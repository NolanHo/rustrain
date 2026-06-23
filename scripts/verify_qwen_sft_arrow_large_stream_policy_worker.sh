#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

WORK_DIR="$(mktemp -d /tmp/rustrain-arrow-large-stream-policy-XXXXXX)"
trap 'rm -rf "${WORK_DIR}"' EXIT

ARROW_PATH="${WORK_DIR}/train.arrow"
UNBOUNDED_CONFIG="${WORK_DIR}/unbounded.toml"
CACHED_CONFIG="${WORK_DIR}/cached.toml"
CACHE_PATH="${WORK_DIR}/arrow-row-index.json"
ERROR_OUTPUT="${WORK_DIR}/unbounded.err"
FIRST_PLAN_OUTPUT="${WORK_DIR}/first-data-plan.out"
SECOND_PLAN_OUTPUT="${WORK_DIR}/second-data-plan.out"

python - "${ARROW_PATH}" <<'PY'
import pathlib
import sys

import pyarrow as pa
import pyarrow.ipc as ipc

path = pathlib.Path(sys.argv[1])
schema = pa.schema(
    [
        ("instruction", pa.string()),
        ("input", pa.string()),
        ("output", pa.string()),
    ]
)
batch = pa.record_batch(
    [
        pa.array(["first", "second", "third", "fourth"], type=pa.string()),
        pa.array(["", "", "", ""], type=pa.string()),
        pa.array(["alpha", "beta", "gamma", "delta"], type=pa.string()),
    ],
    schema=schema,
)
with ipc.new_stream(path, schema) as writer:
    writer.write_batch(batch)
PY

cat >"${UNBOUNDED_CONFIG}" <<EOF
[run]
name = "qwen_sft_arrow_large_stream_policy_unbounded"
base_dir = "/tmp/rustrain-runs"
seed = 777

[model]
name = "qwen2_5_0_5b_arrow_large_stream_policy"
architecture = "qwen_trainable_session"
model_path = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
vocab_size = 151936
hidden_size = 896
num_layers = 24
num_attention_heads = 14
num_key_value_heads = 2
intermediate_size = 4864
seq_len = 128
norm = "rmsnorm"
activation = "swiglu"
rope = true
rms_norm_eps = 0.000001
trainable_layers = [0]

[train]
max_steps = 1
backend = "tch"
micro_batch_size = 1
global_batch_size = 1
gradient_accumulation_steps = 1
learning_rate = 0.000001
weight_decay = 0.0
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_eps = 0.00000001
dtype = "fp32"
device = "cuda"
checkpoint_every = 0

[data]
kind = "instruction_arrow"
paths = ["${ARROW_PATH}"]
train_split = 0.75
shuffle = false
instruction_field = "instruction"
input_field = "input"
response_field = "output"
prompt_template = "Instruction: {instruction}\\nResponse: "
prompt_with_input_template = "Instruction: {instruction}\\nInput: {input}\\nResponse: "

[parallel]
tensor_model_parallel_size = 1
pipeline_model_parallel_size = 1
data_parallel_size = 1
expert_model_parallel_size = 1
context_parallel_size = 1
EOF

cp "${UNBOUNDED_CONFIG}" "${CACHED_CONFIG}"
python - "${CACHED_CONFIG}" "${CACHE_PATH}" <<'PY'
import pathlib
import sys

config = pathlib.Path(sys.argv[1])
cache = pathlib.Path(sys.argv[2]).as_posix()
text = config.read_text()
text = text.replace(
    '[data]\nkind = "instruction_arrow"\n',
    f'[data]\nkind = "instruction_arrow"\nindex_cache = "{cache}"\n',
)
config.write_text(text)
PY

if cargo run -- qwen sft-streaming-data-plan \
  --config "${UNBOUNDED_CONFIG}" \
  --world-size 1 \
  --data-cursor-start 0 >"${ERROR_OUTPUT}" 2>&1; then
  echo "expected unbounded instruction_arrow data plan to fail" >&2
  exit 1
fi
if ! grep -q "requires data.max_samples, data.source_max_samples, or an index cache" "${ERROR_OUTPUT}"; then
  echo "unbounded failure did not mention cache-or-bounds policy" >&2
  cat "${ERROR_OUTPUT}" >&2
  exit 1
fi

cargo run -- qwen sft-streaming-data-plan \
  --config "${CACHED_CONFIG}" \
  --world-size 1 \
  --data-cursor-start 0 \
  | tee "${FIRST_PLAN_OUTPUT}"
cargo run -- qwen sft-streaming-data-plan \
  --config "${CACHED_CONFIG}" \
  --world-size 1 \
  --data-cursor-start 0 \
  | tee "${SECOND_PLAN_OUTPUT}"

python - "${CACHE_PATH}" "${FIRST_PLAN_OUTPUT}" "${SECOND_PLAN_OUTPUT}" <<'PY'
import json
import pathlib
import sys

def load_json_output(path: str) -> dict:
    text = pathlib.Path(path).read_text()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise SystemExit(f"{path} did not contain JSON output: {text}")
    return json.loads(text[start : end + 1])

cache_path = pathlib.Path(sys.argv[1])
first = load_json_output(sys.argv[2])
second = load_json_output(sys.argv[3])

if not cache_path.exists():
    raise SystemExit(f"expected cache file to be written: {cache_path}")
cache = json.loads(cache_path.read_text())
if cache.get("format") != "rustrain.qwen_sft_arrow_row_index.v1":
    raise SystemExit(f"unexpected Arrow cache format: {cache.get('format')}")
if cache.get("max_samples") is not None:
    raise SystemExit(f"unbounded cache should record max_samples null, got {cache.get('max_samples')}")
if len(cache.get("samples", [])) != 4:
    raise SystemExit(f"expected 4 cached row indices, got {len(cache.get('samples', []))}")
if cache.get("summary", {}).get("samples") != 4:
    raise SystemExit(f"expected summary samples 4, got {cache.get('summary')}")

for data, context, expected_hit, expected_written in [
    (first, "first data plan", False, True),
    (second, "second data plan", True, False),
]:
    if data.get("streaming_index_cache_path") != str(cache_path):
        raise SystemExit(f"{context}: unexpected cache path {data.get('streaming_index_cache_path')}")
    if data.get("streaming_index_cache_hit") is not expected_hit:
        raise SystemExit(f"{context}: expected cache hit {expected_hit}, got {data.get('streaming_index_cache_hit')}")
    if data.get("streaming_index_cache_written") is not expected_written:
        raise SystemExit(f"{context}: expected cache written {expected_written}, got {data.get('streaming_index_cache_written')}")
    if data.get("dataset_total_samples") != 4:
        raise SystemExit(f"{context}: expected 4 samples, got {data.get('dataset_total_samples')}")
    if data.get("dataset_train_samples") != 3 or data.get("dataset_eval_samples") != 1:
        raise SystemExit(
            f"{context}: expected 3/1 split, got "
            f"{data.get('dataset_train_samples')}/{data.get('dataset_eval_samples')}"
        )
    if data.get("tokenized_samples_materialized") is not False:
        raise SystemExit(f"{context}: data-plan must not materialize tokenized samples")

print(
    "qwen_sft_arrow_large_stream_policy_verified: "
    f"cache={cache_path} samples={len(cache['samples'])} "
    f"first_written={first['streaming_index_cache_written']} "
    f"second_hit={second['streaming_index_cache_hit']}"
)
PY
