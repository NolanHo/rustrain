#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SFT_STREAMING_BATCH_EVAL_PATHS_CONFIG:-configs/qwen_session_dp2_sft_eval_paths.toml}"
OUTPUT="$(mktemp)"

cargo run -- qwen-sft-streaming-batch-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 4 \
  | tee "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
import json
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
start = text.find("{")
if start < 0:
    raise SystemExit(f"streaming batch eval_paths output did not contain JSON: {text}")
data = json.loads(text[start:])

expected_sources = [
    "data/sft_toy/eval_instructions.jsonl",
    "data/sft_toy/instructions.jsonl",
    "data/sft_toy/more_instructions.jsonl",
]
expected_counts = [
    {"path": "data/sft_toy/eval_instructions.jsonl", "samples": 2},
    {"path": "data/sft_toy/instructions.jsonl", "samples": 6},
    {"path": "data/sft_toy/more_instructions.jsonl", "samples": 2},
]
checks = {
    "world_size": 2,
    "local_batch_size": 1,
    "global_batch_size": 2,
    "train_steps": 2,
    "required_batches": 5,
    "train_batch_count": 3,
    "data_cursor_start": 4,
    "data_cursor_end": 8,
    "data_cursor_next": 8,
    "train_window_start_cursor": 4,
    "train_window_end_cursor_exclusive": 10,
    "dataset_total_samples": 10,
    "dataset_train_samples": 8,
    "dataset_eval_samples": 2,
    "dataset_order_seed": 777,
    "dataset_shuffle": True,
    "tokenizer_loaded": True,
    "tokenized_samples_materialized": True,
    "reference_tokenized_samples_materialized": True,
    "streaming_window_samples": 6,
    "streaming_raw_samples_read": 5,
    "materialized_input_max_delta": 0,
    "materialized_mask_max_delta": 0.0,
}
for key, expected in checks.items():
    if data.get(key) != expected:
        raise SystemExit(f"{key} {data.get(key)} != {expected}")

expected_window = [
    {"cursor": 4, "epoch": 0, "sample_offset": 4},
    {"cursor": 5, "epoch": 0, "sample_offset": 5},
    {"cursor": 6, "epoch": 0, "sample_offset": 6},
    {"cursor": 7, "epoch": 0, "sample_offset": 7},
    {"cursor": 8, "epoch": 1, "sample_offset": 0},
    {"cursor": 9, "epoch": 1, "sample_offset": 1},
]
if data.get("train_window_sample_cursors") != expected_window:
    raise SystemExit(
        f"train_window_sample_cursors {data.get('train_window_sample_cursors')} != {expected_window}"
    )
if data.get("dataset_source_files") != expected_sources:
    raise SystemExit(
        f"dataset_source_files {data.get('dataset_source_files')} != {expected_sources}"
    )
source_counts = data.get("dataset_source_sample_counts") or []
if source_counts != expected_counts:
    raise SystemExit(f"dataset_source_sample_counts {source_counts} != {expected_counts}")
if [entry.get("path") for entry in source_counts] != expected_sources:
    raise SystemExit(
        f"dataset_source_sample_counts paths {[entry.get('path') for entry in source_counts]} != {expected_sources}"
    )
if any(int(entry.get("samples", 0)) <= 0 for entry in source_counts):
    raise SystemExit(f"dataset_source_sample_counts must be positive: {source_counts}")
if sum(int(entry["samples"]) for entry in source_counts) != int(data["dataset_total_samples"]):
    raise SystemExit(
        f"dataset_source_sample_counts total {source_counts} != dataset_total_samples {data['dataset_total_samples']}"
    )
if data.get("dataset_fingerprint") != "f771c261589611be":
    raise SystemExit(f"unexpected dataset_fingerprint {data.get('dataset_fingerprint')}")

raw_indices = data.get("streaming_raw_sample_indices") or []
if len(raw_indices) != 6:
    raise SystemExit(f"expected 6 raw streaming indices, got {raw_indices}")
raw_paths = {entry.get("path") for entry in raw_indices}
if raw_paths != {"data/sft_toy/instructions.jsonl", "data/sft_toy/more_instructions.jsonl"}:
    raise SystemExit(f"streaming raw train paths should exclude eval_paths, got {raw_paths}")
if any(entry.get("path") == "data/sft_toy/eval_instructions.jsonl" for entry in raw_indices):
    raise SystemExit(f"streaming raw indices unexpectedly include eval source: {raw_indices}")
if not any(entry.get("path") == "data/sft_toy/more_instructions.jsonl" for entry in raw_indices):
    raise SystemExit(f"streaming raw indices did not include the second train source: {raw_indices}")

for key, expected_len in [
    ("batch_sequence_tokens", 3),
    ("batch_masked_positions", 3),
    ("batch_padding_tokens", 3),
    ("batch_token_fingerprints", 3),
]:
    values = data.get(key)
    if not isinstance(values, list) or len(values) != expected_len:
        raise SystemExit(f"{key} expected {expected_len} values, got {values}")
if any(value <= 1 for value in data["batch_sequence_tokens"]):
    raise SystemExit(f"batch_sequence_tokens must be > 1, got {data['batch_sequence_tokens']}")
if any(value <= 0 for value in data["batch_masked_positions"]):
    raise SystemExit(f"batch_masked_positions must be positive, got {data['batch_masked_positions']}")
if any(not isinstance(value, str) or len(value) != 16 for value in data["batch_token_fingerprints"]):
    raise SystemExit(f"bad batch_token_fingerprints: {data['batch_token_fingerprints']}")

print(
    "qwen_sft_streaming_batch_eval_paths_verified: "
    f"train_samples={data['dataset_train_samples']} "
    f"eval_samples={data['dataset_eval_samples']} "
    f"raw_paths={sorted(raw_paths)} "
    f"materialized_input_max_delta={data['materialized_input_max_delta']} "
    f"materialized_mask_max_delta={data['materialized_mask_max_delta']}"
)
PY
