#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

WORK_DIR="$(mktemp -d /tmp/rustrain-sft-hf-cache-XXXXXX)"
trap 'rm -rf "${WORK_DIR}"' EXIT

HF_ARROW="${RUSTRAIN_HF_SFT_ARROW:-/vePFS-Mindverse/share/huggingface/datasets/iamtarun___code_instructions_120k_alpaca/default/0.0.0/31f725b2d714c1b4f038e80fbaa6b977870a50b7/code_instructions_120k_alpaca-train.arrow}"
DATA_DIR="${WORK_DIR}/data"
mkdir -p "${DATA_DIR}"

python scripts/export_instruction_arrow_jsonl.py \
  --input "${HF_ARROW}" \
  --output "${DATA_DIR}/alpaca.jsonl" \
  --limit 128 \
  --shards 2 \
  --response-column output \
  --metadata-output "${DATA_DIR}/export_metadata.json"

python scripts/export_instruction_arrow_jsonl.py \
  --input "${HF_ARROW}" \
  --output "${DATA_DIR}/alpaca_bounded.jsonl" \
  --limit 5 \
  --no-full-row-count \
  --response-column output \
  --metadata-output "${DATA_DIR}/export_metadata_bounded.json" >/dev/null

CONFIG="${WORK_DIR}/hf-cache.toml"
cat >"${CONFIG}" <<EOF
[run]
name = "qwen_sft_streaming_hf_cache"
base_dir = "/tmp/rustrain-runs"
seed = 777

[model]
name = "qwen2_5_0_5b_sft_streaming_hf_cache"
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
max_steps = 2
backend = "tch"
micro_batch_size = 2
global_batch_size = 2
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
kind = "instruction_jsonl"
paths = [
  "${DATA_DIR}/alpaca_0.jsonl",
  "${DATA_DIR}/alpaca_1.jsonl",
]
external_metadata_paths = ["${DATA_DIR}/export_metadata.json"]
train_split = 0.75
shuffle = false
prompt_template = "Instruction: {instruction}\\nResponse: "
prompt_with_input_template = "Instruction: {instruction}\\nInput: {input}\\nResponse: "

[parallel]
tensor_model_parallel_size = 1
pipeline_model_parallel_size = 1
data_parallel_size = 2
expert_model_parallel_size = 1
context_parallel_size = 1
EOF

DATA_OUTPUT="${WORK_DIR}/data-plan.out"
BATCH_OUTPUT="${WORK_DIR}/batch-plan.out"
CACHE_FIRST_OUTPUT="${WORK_DIR}/batch-plan-cache-first.out"
CACHE_SECOND_OUTPUT="${WORK_DIR}/batch-plan-cache-second.out"
CACHE_PATH="${WORK_DIR}/offset-index.json"

cargo run -- qwen-sft-streaming-data-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 94 \
  | tee "${DATA_OUTPUT}"

cargo run -- qwen-sft-streaming-batch-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 94 \
  | tee "${BATCH_OUTPUT}"

cargo run -- qwen-sft-streaming-batch-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 94 \
  --index-cache "${CACHE_PATH}" \
  | tee "${CACHE_FIRST_OUTPUT}"

cargo run -- qwen-sft-streaming-batch-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 94 \
  --index-cache "${CACHE_PATH}" \
  | tee "${CACHE_SECOND_OUTPUT}"

python - \
  "${DATA_OUTPUT}" \
  "${BATCH_OUTPUT}" \
  "${CACHE_FIRST_OUTPUT}" \
  "${CACHE_SECOND_OUTPUT}" \
  "${CACHE_PATH}" \
  "${DATA_DIR}" \
  "${HF_ARROW}" <<'PY'
import json
import pathlib
import sys

data_output = pathlib.Path(sys.argv[1])
batch_output = pathlib.Path(sys.argv[2])
cache_first_output = pathlib.Path(sys.argv[3])
cache_second_output = pathlib.Path(sys.argv[4])
cache_path = pathlib.Path(sys.argv[5])
data_dir = pathlib.Path(sys.argv[6])
hf_arrow = pathlib.Path(sys.argv[7])


def load_json(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8")
    start = text.find("{")
    if start < 0:
        raise SystemExit(f"{path} did not contain JSON output: {text}")
    return json.loads(text[start:])


data_plan = load_json(data_output)
batch_plan = load_json(batch_output)
cache_first = load_json(cache_first_output)
cache_second = load_json(cache_second_output)
metadata = json.loads((data_dir / "export_metadata.json").read_text(encoding="utf-8"))
bounded_metadata = json.loads(
    (data_dir / "export_metadata_bounded.json").read_text(encoding="utf-8")
)

source_files = [str(data_dir / f"alpaca_{index}.jsonl") for index in range(2)]
source_counts = [{"path": path, "samples": 64} for path in source_files]
export_outputs = [{"path": path, "rows": 64} for path in source_files]

if pathlib.Path(metadata["source_arrow"]) != hf_arrow:
    raise SystemExit(f"metadata source_arrow {metadata['source_arrow']} != {hf_arrow}")
if metadata["source_rows"] < 100_000 or metadata["exported_rows"] != 128:
    raise SystemExit(f"unexpected HF export metadata: {metadata}")
if metadata.get("source_rows_exact") is not True or metadata.get("source_rows_lower_bound") is not False:
    raise SystemExit(f"full HF export metadata should record exact source row count: {metadata}")
if metadata.get("arrow_ipc_format") != "stream":
    raise SystemExit(f"HF export should record Arrow stream format: {metadata}")
for column in ["instruction", "input", "output"]:
    if column not in metadata["columns"]:
        raise SystemExit(f"HF export metadata missing column {column}: {metadata}")
if metadata.get("column_map", {}).get("response") != "output":
    raise SystemExit(f"HF export metadata did not record output response source: {metadata}")
if metadata.get("shards") != 2:
    raise SystemExit(f"HF export metadata did not record two shards: {metadata}")
if metadata.get("output_files") != export_outputs:
    raise SystemExit(f"HF export metadata output_files {metadata.get('output_files')} != {export_outputs}")

if pathlib.Path(bounded_metadata["source_arrow"]) != hf_arrow:
    raise SystemExit(f"bounded metadata source_arrow {bounded_metadata['source_arrow']} != {hf_arrow}")
if bounded_metadata.get("arrow_ipc_format") != "stream":
    raise SystemExit(f"bounded HF export should record Arrow stream format: {bounded_metadata}")
if bounded_metadata.get("source_rows_exact") is not False:
    raise SystemExit(f"bounded export should not claim exact source_rows: {bounded_metadata}")
if bounded_metadata.get("source_rows_lower_bound") is not True:
    raise SystemExit(f"bounded export should mark source_rows as lower bound: {bounded_metadata}")
if bounded_metadata.get("exported_rows") != 5 or bounded_metadata.get("source_rows", 0) < 5:
    raise SystemExit(f"bounded export metadata has unexpected counts: {bounded_metadata}")
if bounded_metadata.get("shards") != 1:
    raise SystemExit(f"bounded export should use one shard: {bounded_metadata}")
bounded_output = data_dir / "alpaca_bounded.jsonl"
if bounded_metadata.get("output_files") != [{"path": str(bounded_output), "rows": 5}]:
    raise SystemExit(f"bounded export output_files mismatch: {bounded_metadata}")
bounded_lines = bounded_output.read_text(encoding="utf-8").splitlines()
if len(bounded_lines) != 5:
    raise SystemExit(f"bounded export wrote {len(bounded_lines)} rows instead of 5")
for index, line in enumerate(bounded_lines):
    record = json.loads(line)
    if not isinstance(record.get("instruction"), str) or not isinstance(record.get("response"), str):
        raise SystemExit(f"bounded export row {index} has invalid record: {record}")

expected_window = [
    {
        "cursor": cursor,
        "epoch": 0 if cursor < 96 else 1,
        "sample_offset": cursor if cursor < 96 else cursor - 96,
    }
    for cursor in range(94, 106)
]

for key, expected in {
    "world_size": 2,
    "local_batch_size": 2,
    "global_batch_size": 4,
    "train_steps": 2,
    "data_cursor_start": 94,
    "data_cursor_end": 102,
    "data_cursor_next": 102,
    "dataset_total_samples": 128,
    "dataset_train_samples": 96,
    "dataset_eval_samples": 32,
    "dataset_order_seed": 777,
    "dataset_shuffle": False,
    "tokenizer_loaded": False,
    "tokenized_samples_materialized": False,
}.items():
    if data_plan.get(key) != expected:
        raise SystemExit(f"data-plan {key} {data_plan.get(key)!r} != {expected!r}")
if data_plan.get("data_epoch_start") != 0 or data_plan.get("data_sample_offset_start") != 94:
    raise SystemExit(f"data-plan start cursor metadata is wrong: {data_plan}")
if data_plan.get("data_epoch_end") != 1 or data_plan.get("data_sample_offset_end") != 6:
    raise SystemExit(f"data-plan end cursor metadata is wrong: {data_plan}")
if data_plan.get("train_window_sample_cursors") != expected_window:
    raise SystemExit(
        f"data-plan train_window_sample_cursors {data_plan.get('train_window_sample_cursors')} != {expected_window}"
    )
if data_plan.get("dataset_source_files") != source_files:
    raise SystemExit(f"data-plan dataset_source_files {data_plan.get('dataset_source_files')} != {source_files}")
if data_plan.get("dataset_source_sample_counts") != source_counts:
    raise SystemExit(
        f"data-plan dataset_source_sample_counts {data_plan.get('dataset_source_sample_counts')} != {source_counts}"
    )
if data_plan.get("train_source_files") != source_files or data_plan.get("train_source_sample_counts") != source_counts:
    raise SystemExit("data-plan train source metadata did not match dataset metadata")
if not data_plan.get("dataset_fingerprint") or data_plan.get("dataset_fingerprint") != data_plan.get("train_fingerprint"):
    raise SystemExit("data-plan fingerprint must be non-empty and match train_fingerprint")


def verify_batch_plan(plan: dict, context: str) -> None:
    for key, expected in {
        "world_size": 2,
        "local_batch_size": 2,
        "global_batch_size": 4,
        "train_steps": 2,
        "train_batch_count": 3,
        "data_cursor_start": 94,
        "data_cursor_end": 102,
        "data_cursor_next": 102,
        "dataset_total_samples": 128,
        "dataset_train_samples": 96,
        "dataset_eval_samples": 32,
        "dataset_order_seed": 777,
        "dataset_shuffle": False,
        "tokenizer_loaded": True,
        "tokenized_samples_materialized": True,
        "reference_tokenized_samples_materialized": True,
        "streaming_window_samples": 12,
        "streaming_raw_samples_read": 12,
        "materialized_input_max_delta": 0,
        "materialized_mask_max_delta": 0.0,
    }.items():
        if plan.get(key) != expected:
            raise SystemExit(f"{context} {key} {plan.get(key)!r} != {expected!r}")
    if plan.get("train_window_sample_cursors") != expected_window:
        raise SystemExit(
            f"{context} train_window_sample_cursors {plan.get('train_window_sample_cursors')} != {expected_window}"
        )
    if plan.get("dataset_source_files") != source_files:
        raise SystemExit(f"{context} dataset_source_files {plan.get('dataset_source_files')} != {source_files}")
    if plan.get("dataset_source_sample_counts") != source_counts:
        raise SystemExit(
            f"{context} dataset_source_sample_counts {plan.get('dataset_source_sample_counts')} != {source_counts}"
        )
    if plan.get("dataset_fingerprint") != data_plan.get("dataset_fingerprint"):
        raise SystemExit(f"{context} fingerprint drifted from data-plan")
    for key in [
        "batch_sequence_tokens",
        "batch_masked_positions",
        "batch_padding_tokens",
        "batch_token_fingerprints",
    ]:
        values = plan.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise SystemExit(f"{context} expected three entries for {key}, got {values}")
    if any(value <= 1 for value in plan["batch_sequence_tokens"]):
        raise SystemExit(f"{context} batch_sequence_tokens must be >1: {plan['batch_sequence_tokens']}")
    if any(value <= 0 for value in plan["batch_masked_positions"]):
        raise SystemExit(f"{context} batch_masked_positions must be positive: {plan['batch_masked_positions']}")
    if any(not isinstance(value, str) or len(value) != 16 for value in plan["batch_token_fingerprints"]):
        raise SystemExit(f"{context} bad batch token fingerprints: {plan['batch_token_fingerprints']}")
    raw_indices = plan.get("streaming_raw_sample_indices")
    if not isinstance(raw_indices, list) or len(raw_indices) != 12:
        raise SystemExit(f"{context} expected 12 raw sample indices, got {raw_indices}")
    unique_samples = {
        (entry["path"], entry["index_in_file"], entry["global_index"])
        for entry in raw_indices
    }
    if len(unique_samples) != plan["streaming_raw_samples_read"]:
        raise SystemExit(
            f"{context} unique raw sample count {len(unique_samples)} != streaming_raw_samples_read {plan['streaming_raw_samples_read']}"
        )
    if raw_indices[0]["path"] != source_files[1] or raw_indices[0]["index_in_file"] != 30:
        raise SystemExit(f"{context} first cursor should land near end of second source: {raw_indices[0]}")
    if raw_indices[1]["path"] != source_files[1] or raw_indices[1]["index_in_file"] != 31:
        raise SystemExit(f"{context} second cursor should land at final train record: {raw_indices[1]}")
    if raw_indices[2]["path"] != source_files[0] or raw_indices[2]["index_in_file"] != 0:
        raise SystemExit(f"{context} wrap cursor should return to first source: {raw_indices[2]}")
    for entry in raw_indices:
        offset = entry.get("byte_offset")
        if not isinstance(offset, int) or offset < 0:
            raise SystemExit(f"{context} invalid byte_offset: {entry}")


verify_batch_plan(batch_plan, "batch-plan")
verify_batch_plan(cache_first, "cache-first")
verify_batch_plan(cache_second, "cache-second")

if batch_plan.get("streaming_index_cache_path") is not None:
    raise SystemExit("uncached batch plan unexpectedly reported a cache path")
if batch_plan.get("streaming_index_cache_hit") is not False or batch_plan.get("streaming_index_cache_written") is not False:
    raise SystemExit("uncached batch plan cache flags were not false")
if cache_first.get("streaming_index_cache_path") != str(cache_path):
    raise SystemExit("cache first path mismatch")
if cache_first.get("streaming_index_cache_hit") is not False or cache_first.get("streaming_index_cache_written") is not True:
    raise SystemExit("first cache run should write without hit")
if cache_second.get("streaming_index_cache_path") != str(cache_path):
    raise SystemExit("cache second path mismatch")
if cache_second.get("streaming_index_cache_hit") is not True or cache_second.get("streaming_index_cache_written") is not False:
    raise SystemExit("second cache run should hit without writing")
if cache_first.get("streaming_raw_sample_indices") != cache_second.get("streaming_raw_sample_indices"):
    raise SystemExit("cache hit raw indices differ from cache write raw indices")

cache = json.loads(cache_path.read_text(encoding="utf-8"))
if cache.get("format") != "rustrain.qwen_sft_offset_index.v7":
    raise SystemExit(f"unexpected cache format {cache.get('format')}")
if len(cache.get("samples", [])) != 128:
    raise SystemExit(f"cache should contain all 128 raw offsets, got {len(cache.get('samples', []))}")
if cache.get("source_files") is None or len(cache["source_files"]) != 2:
    raise SystemExit(f"cache should contain two source metadata entries, got {cache.get('source_files')}")
if cache.get("field_map", {}).get("response") != "response":
    raise SystemExit(f"cache field_map response should be exported response: {cache.get('field_map')}")
summary = cache.get("summary")
if not isinstance(summary, dict):
    raise SystemExit(f"cache should contain summary metadata, got {summary}")
if summary.get("samples") != data_plan.get("dataset_total_samples"):
    raise SystemExit(f"cache summary samples {summary.get('samples')} != data-plan total {data_plan.get('dataset_total_samples')}")
if summary.get("source_files") != data_plan.get("dataset_source_files"):
    raise SystemExit("cache summary source_files differ from data-plan source files")
if summary.get("source_sample_counts") != data_plan.get("dataset_source_sample_counts"):
    raise SystemExit("cache summary source sample counts differ from data-plan")
if summary.get("fingerprint") != data_plan.get("dataset_fingerprint"):
    raise SystemExit(f"cache summary fingerprint {summary.get('fingerprint')} != data-plan {data_plan.get('dataset_fingerprint')}")

print(
    "qwen_sft_streaming_hf_cache_verified: "
    f"source_rows={metadata['source_rows']} "
    f"exported_rows={metadata['exported_rows']} "
    f"train_samples={data_plan['dataset_train_samples']} "
    f"streaming_window_samples={batch_plan['streaming_window_samples']} "
    f"streaming_raw_samples_read={batch_plan['streaming_raw_samples_read']} "
    f"cache_hit={cache_second['streaming_index_cache_hit']} "
    f"fingerprint={data_plan['dataset_fingerprint']}"
)
PY
