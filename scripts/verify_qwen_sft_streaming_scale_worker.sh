#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

WORK_DIR="$(mktemp -d /tmp/rustrain-sft-scale-XXXXXX)"
trap 'rm -rf "${WORK_DIR}"' EXIT

DATA_DIR="${WORK_DIR}/data"
mkdir -p "${DATA_DIR}"

python - "${DATA_DIR}" <<'PY'
import json
import pathlib
import sys

data_dir = pathlib.Path(sys.argv[1])
for shard_index, count in enumerate([80, 70, 50]):
    path = data_dir / f"scale_shard_{shard_index}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row_index in range(count):
            global_index = shard_index * 1000 + row_index
            record = {
                "system": f"scale system shard={shard_index}",
                "instruction": f"scale instruction {global_index}",
                "input": f"context shard={shard_index} row={row_index}",
                "response": f"scale response {global_index}",
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
PY

CONFIG="${WORK_DIR}/scale.toml"
cat >"${CONFIG}" <<EOF
[run]
name = "qwen_sft_streaming_scale"
base_dir = "/tmp/rustrain-runs"
seed = 777

[model]
name = "qwen2_5_0_5b_sft_streaming_scale"
architecture = "qwen_trainable_session"
model_path = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
vocab_size = 151936
hidden_size = 896
num_layers = 24
num_attention_heads = 14
num_key_value_heads = 2
intermediate_size = 4864
seq_len = 64
norm = "rmsnorm"
activation = "swiglu"
rope = true
rms_norm_eps = 0.000001
trainable_layers = [0]

[train]
max_steps = 3
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
  "${DATA_DIR}/scale_shard_0.jsonl",
  "${DATA_DIR}/scale_shard_1.jsonl",
  "${DATA_DIR}/scale_shard_2.jsonl",
]
train_split = 0.8
shuffle = false
system_field = "system"
prompt_template = "System: {system}\\nInstruction: {instruction}\\nResponse: "
prompt_with_input_template = "System: {system}\\nInstruction: {instruction}\\nInput: {input}\\nResponse: "

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
  --data-cursor-start 158 \
  | tee "${DATA_OUTPUT}"

cargo run -- qwen-sft-streaming-batch-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 158 \
  | tee "${BATCH_OUTPUT}"

cargo run -- qwen-sft-streaming-batch-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 158 \
  --index-cache "${CACHE_PATH}" \
  | tee "${CACHE_FIRST_OUTPUT}"

cargo run -- qwen-sft-streaming-batch-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 158 \
  --index-cache "${CACHE_PATH}" \
  | tee "${CACHE_SECOND_OUTPUT}"

python - "${DATA_OUTPUT}" "${BATCH_OUTPUT}" "${CACHE_FIRST_OUTPUT}" "${CACHE_SECOND_OUTPUT}" "${CACHE_PATH}" "${DATA_DIR}" <<'PY'
import json
import pathlib
import sys

data_output = pathlib.Path(sys.argv[1])
batch_output = pathlib.Path(sys.argv[2])
cache_first_output = pathlib.Path(sys.argv[3])
cache_second_output = pathlib.Path(sys.argv[4])
cache_path = pathlib.Path(sys.argv[5])
data_dir = pathlib.Path(sys.argv[6])

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

source_files = [str(data_dir / f"scale_shard_{index}.jsonl") for index in range(3)]
source_counts = [
    {"path": source_files[0], "samples": 80},
    {"path": source_files[1], "samples": 70},
    {"path": source_files[2], "samples": 50},
]
expected_window = [
    {
        "cursor": cursor,
        "epoch": 0 if cursor < 160 else 1,
        "sample_offset": cursor if cursor < 160 else cursor - 160,
    }
    for cursor in range(158, 174)
]

for key, expected in {
    "world_size": 2,
    "local_batch_size": 2,
    "global_batch_size": 4,
    "train_steps": 3,
    "required_batches": 13,
    "data_cursor_start": 158,
    "data_cursor_end": 170,
    "data_cursor_next": 170,
    "dataset_total_samples": 200,
    "dataset_train_samples": 160,
    "dataset_eval_samples": 40,
    "dataset_order_seed": 777,
    "dataset_shuffle": False,
    "tokenizer_loaded": False,
    "tokenized_samples_materialized": False,
}.items():
    if data_plan.get(key) != expected:
        raise SystemExit(f"data-plan {key} {data_plan.get(key)!r} != {expected!r}")
if data_plan.get("data_epoch_start") != 0 or data_plan.get("data_sample_offset_start") != 158:
    raise SystemExit(f"data-plan start cursor metadata is wrong: {data_plan}")
if data_plan.get("data_epoch_end") != 1 or data_plan.get("data_sample_offset_end") != 10:
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
        "train_steps": 3,
        "required_batches": 13,
        "train_batch_count": 4,
        "data_cursor_start": 158,
        "data_cursor_end": 170,
        "data_cursor_next": 170,
        "dataset_total_samples": 200,
        "dataset_train_samples": 160,
        "dataset_eval_samples": 40,
        "dataset_order_seed": 777,
        "dataset_shuffle": False,
        "tokenizer_loaded": True,
        "tokenized_samples_materialized": True,
        "reference_tokenized_samples_materialized": True,
        "streaming_window_samples": 16,
        "streaming_raw_samples_read": 16,
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
        if not isinstance(values, list) or len(values) != 4:
            raise SystemExit(f"{context} expected four entries for {key}, got {values}")
    if any(value <= 1 for value in plan["batch_sequence_tokens"]):
        raise SystemExit(f"{context} batch_sequence_tokens must be >1: {plan['batch_sequence_tokens']}")
    if any(value <= 0 for value in plan["batch_masked_positions"]):
        raise SystemExit(f"{context} batch_masked_positions must be positive: {plan['batch_masked_positions']}")
    if any(not isinstance(value, str) or len(value) != 16 for value in plan["batch_token_fingerprints"]):
        raise SystemExit(f"{context} bad batch token fingerprints: {plan['batch_token_fingerprints']}")
    raw_indices = plan.get("streaming_raw_sample_indices")
    if not isinstance(raw_indices, list) or len(raw_indices) != 16:
        raise SystemExit(f"{context} expected 16 raw sample indices, got {raw_indices}")
    unique_samples = {
        (entry["path"], entry["index_in_file"], entry["global_index"])
        for entry in raw_indices
    }
    if len(unique_samples) != plan["streaming_raw_samples_read"]:
        raise SystemExit(
            f"{context} unique raw sample count {len(unique_samples)} != streaming_raw_samples_read {plan['streaming_raw_samples_read']}"
        )
    expected_paths = {source_files[0], source_files[1], source_files[2]}
    if not {entry["path"] for entry in raw_indices}.issubset(expected_paths):
        raise SystemExit(f"{context} raw window used unexpected source files: {raw_indices}")
    if raw_indices[0]["path"] != source_files[2] or raw_indices[0]["index_in_file"] != 8:
        raise SystemExit(f"{context} first cursor should land near end of third source: {raw_indices[0]}")
    if raw_indices[1]["path"] != source_files[2] or raw_indices[1]["index_in_file"] != 9:
        raise SystemExit(f"{context} second cursor should land at final third-source record: {raw_indices[1]}")
    if raw_indices[2]["path"] != source_files[0] or raw_indices[2]["index_in_file"] != 0:
        raise SystemExit(f"{context} wrap cursor should return to first source: {raw_indices[2]}")
    for entry in raw_indices:
        offset = entry.get("byte_offset")
        if not isinstance(offset, int) or offset < 0:
            raise SystemExit(f"{context} invalid byte_offset: {entry}")
    offsets_by_sample = {}
    for entry in raw_indices:
        sample_key = (entry["path"], entry["index_in_file"], entry["global_index"])
        previous = offsets_by_sample.setdefault(sample_key, entry["byte_offset"])
        if previous != entry["byte_offset"]:
            raise SystemExit(f"{context} inconsistent duplicate byte_offset for {sample_key}")

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
if cache.get("format") != "rustrain.qwen_sft_offset_index.v6":
    raise SystemExit(f"unexpected cache format {cache.get('format')}")
if len(cache.get("samples", [])) != 200:
    raise SystemExit(f"cache should contain all 200 raw offsets, got {len(cache.get('samples', []))}")
if cache.get("source_files") is None or len(cache["source_files"]) != 3:
    raise SystemExit(f"cache should contain three source metadata entries, got {cache.get('source_files')}")
if cache.get("field_map", {}).get("system") != "system":
    raise SystemExit(f"cache field_map system should be set: {cache.get('field_map')}")

print(
    "qwen_sft_streaming_scale_verified: "
    f"total_samples={data_plan['dataset_total_samples']} "
    f"train_samples={data_plan['dataset_train_samples']} "
    f"source_files={len(source_files)} "
    f"streaming_window_samples={batch_plan['streaming_window_samples']} "
    f"streaming_raw_samples_read={batch_plan['streaming_raw_samples_read']} "
    f"cache_hit={cache_second['streaming_index_cache_hit']} "
    f"fingerprint={data_plan['dataset_fingerprint']}"
)
PY
