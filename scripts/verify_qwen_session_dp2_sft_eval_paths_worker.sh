#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SESSION_DP2_SFT_EVAL_PATHS_CONFIG:-configs/qwen_session_dp2_sft_eval_paths.toml}"
OUTPUT="$(mktemp)"

cargo run -- qwen-session-dp-data-plan --config "${CONFIG}" --world-size 2 \
  | tee "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
import json
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
start = text.find("{")
if start < 0:
    raise SystemExit(f"data plan output did not contain JSON: {text}")
data = json.loads(text[start:])

expected_sources = [
    "data/sft_toy/eval_instructions.jsonl",
    "data/sft_toy/instructions.jsonl",
    "data/sft_toy/more_instructions.jsonl",
]
checks = {
    "world_size": 2,
    "local_batch_size": 1,
    "global_batch_size": 2,
    "train_steps": 2,
    "required_batches": 5,
    "data_cursor_start": 0,
    "data_cursor_end": 4,
    "data_cursor_next": 4,
    "data_epoch_start": 0,
    "data_epoch_end": 0,
    "data_epoch_next": 0,
    "data_sample_offset_start": 0,
    "data_sample_offset_end": 4,
    "data_sample_offset_next": 4,
    "dataset_total_samples": 10,
    "dataset_train_samples": 8,
    "dataset_eval_samples": 2,
    "dataset_order_seed": 777,
}
for key, expected in checks.items():
    if data.get(key) != expected:
        raise SystemExit(f"{key} {data.get(key)} != {expected}")
if data.get("dataset_source_files") != expected_sources:
    raise SystemExit(
        f"dataset_source_files {data.get('dataset_source_files')} != {expected_sources}"
    )
if not data.get("dataset_fingerprint"):
    raise SystemExit("dataset_fingerprint must not be empty")
if data.get("dataset_shuffle") is not True:
    raise SystemExit(f"dataset_shuffle {data.get('dataset_shuffle')} is not true")
source_counts = data.get("dataset_source_sample_counts") or []
expected_counts = {
    "data/sft_toy/eval_instructions.jsonl": 2,
    "data/sft_toy/instructions.jsonl": 6,
    "data/sft_toy/more_instructions.jsonl": 2,
}
actual_counts = {entry["path"]: entry["samples"] for entry in source_counts}
if actual_counts != expected_counts:
    raise SystemExit(f"source counts {actual_counts} != {expected_counts}")
if int(data.get("dataset_total_tokens", 0)) <= 0:
    raise SystemExit(f"dataset_total_tokens must be positive: {data.get('dataset_total_tokens')}")

print(
    "qwen_session_dp2_sft_eval_paths_data_plan_verified: "
    f"train_samples={data['dataset_train_samples']} "
    f"eval_samples={data['dataset_eval_samples']} "
    f"source_files={data['dataset_source_files']} "
    f"fingerprint={data['dataset_fingerprint']} "
    f"data_cursor_next={data['data_cursor_next']}"
)
PY
