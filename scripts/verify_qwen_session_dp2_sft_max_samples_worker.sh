#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SESSION_DP2_SFT_MAX_SAMPLES_CONFIG:-configs/qwen_session_dp2_sft_max_samples.toml}"
EXPECTED_TOTAL="${RUSTRAIN_EXPECTED_MAX_SAMPLES:-4}"
EXPECTED_SOURCE="${RUSTRAIN_EXPECTED_MAX_SAMPLES_SOURCE:-data/sft_toy/instructions.jsonl}"
OUTPUT="$(mktemp)"

cargo run -- qwen-session-dp-data-plan --config "${CONFIG}" --world-size 2 \
  | tee "${OUTPUT}"

python - "${OUTPUT}" "${EXPECTED_TOTAL}" "${EXPECTED_SOURCE}" <<'PY'
import json
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
expected_total = int(sys.argv[2])
expected_source = sys.argv[3]
start = text.find("{")
if start < 0:
    raise SystemExit(f"data plan output did not contain JSON: {text}")
data = json.loads(text[start:])

checks = {
    "world_size": 2,
    "local_batch_size": 1,
    "global_batch_size": 2,
    "train_steps": 1,
    "required_batches": 3,
    "data_cursor_start": 0,
    "data_cursor_end": 2,
    "data_cursor_next": 2,
    "data_epoch_start": 0,
    "data_epoch_end": 0,
    "data_epoch_next": 0,
    "data_sample_offset_start": 0,
    "data_sample_offset_end": 2,
    "data_sample_offset_next": 2,
    "dataset_total_samples": expected_total,
    "dataset_train_samples": 3,
    "dataset_eval_samples": 1,
    "dataset_order_seed": 777,
}
for key, expected in checks.items():
    if data.get(key) != expected:
        raise SystemExit(f"{key} {data.get(key)} != {expected}")
if data.get("dataset_source_files") != [expected_source]:
    raise SystemExit(
        f"dataset_source_files {data.get('dataset_source_files')} != {[expected_source]}"
    )
source_counts = data.get("dataset_source_sample_counts") or []
expected_counts = [{"path": expected_source, "samples": expected_total}]
if source_counts != expected_counts:
    raise SystemExit(f"dataset_source_sample_counts {source_counts} != {expected_counts}")
if not data.get("dataset_fingerprint"):
    raise SystemExit("dataset_fingerprint must not be empty")
if data.get("dataset_shuffle") is not True:
    raise SystemExit(f"dataset_shuffle {data.get('dataset_shuffle')} is not true")
if int(data.get("dataset_total_tokens", 0)) <= 0:
    raise SystemExit(f"dataset_total_tokens must be positive: {data.get('dataset_total_tokens')}")

print(
    "qwen_session_dp2_sft_max_samples_data_plan_verified: "
    f"total_samples={data['dataset_total_samples']} "
    f"train_samples={data['dataset_train_samples']} "
    f"eval_samples={data['dataset_eval_samples']} "
    f"source_files={data['dataset_source_files']} "
    f"source_counts={data['dataset_source_sample_counts']} "
    f"fingerprint={data['dataset_fingerprint']} "
    f"data_cursor_next={data['data_cursor_next']}"
)
PY
