#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SFT_STREAMING_DATA_PLAN_CONFIG:-configs/qwen_session_dp2_sft_max_samples.toml}"
EXPECTED_SOURCE="${RUSTRAIN_EXPECTED_STREAMING_SOURCE:-data/sft_toy/instructions.jsonl}"
EXPECTED_SECOND_SOURCE="${RUSTRAIN_EXPECTED_STREAMING_SECOND_SOURCE:-data/sft_toy/more_instructions.jsonl}"
EXPECTED_FINGERPRINT="${RUSTRAIN_EXPECTED_STREAMING_FINGERPRINT:-3bfd266239e4b9b9}"
OUTPUT="$(mktemp)"

cargo run -- qwen-sft-streaming-data-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 2 \
  | tee "${OUTPUT}"

python - "${OUTPUT}" "${EXPECTED_SOURCE}" "${EXPECTED_SECOND_SOURCE}" "${EXPECTED_FINGERPRINT}" <<'PY'
import json
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
expected_source = sys.argv[2]
expected_second_source = sys.argv[3]
expected_fingerprint = sys.argv[4]
start = text.find("{")
if start < 0:
    raise SystemExit(f"streaming data plan output did not contain JSON: {text}")
data = json.loads(text[start:])

checks = {
    "max_samples": 4,
    "world_size": 2,
    "local_batch_size": 1,
    "global_batch_size": 2,
    "train_steps": 1,
    "required_batches": 3,
    "data_cursor_start": 2,
    "data_cursor_end": 4,
    "data_cursor_next": 4,
    "data_epoch_start": 0,
    "data_epoch_end": 1,
    "data_epoch_next": 1,
    "data_sample_offset_start": 2,
    "data_sample_offset_end": 1,
    "data_sample_offset_next": 1,
    "train_window_start_cursor": 2,
    "train_window_end_cursor_exclusive": 6,
    "dataset_total_samples": 4,
    "dataset_train_samples": 3,
    "dataset_eval_samples": 1,
    "dataset_order_seed": 777,
    "dataset_shuffle": True,
    "tokenizer_loaded": False,
    "tokenized_samples_materialized": False,
}
for key, expected in checks.items():
    if data.get(key) != expected:
        raise SystemExit(f"{key} {data.get(key)} != {expected}")
if data.get("data_paths") != [expected_source, expected_second_source]:
    raise SystemExit(
        f"data_paths {data.get('data_paths')} != {[expected_source, expected_second_source]}"
    )
if data.get("eval_paths") != []:
    raise SystemExit(f"eval_paths should be empty, got {data.get('eval_paths')}")
if data.get("max_eval_samples") is not None:
    raise SystemExit(f"max_eval_samples should be null, got {data.get('max_eval_samples')}")

expected_window = [
    {"cursor": 2, "epoch": 0, "sample_offset": 2},
    {"cursor": 3, "epoch": 1, "sample_offset": 0},
    {"cursor": 4, "epoch": 1, "sample_offset": 1},
    {"cursor": 5, "epoch": 1, "sample_offset": 2},
]
if data.get("train_window_sample_cursors") != expected_window:
    raise SystemExit(
        f"train_window_sample_cursors {data.get('train_window_sample_cursors')} != {expected_window}"
    )

expected_counts = [{"path": expected_source, "samples": 4}]
for key in ["dataset_source_sample_counts", "train_source_sample_counts"]:
    if data.get(key) != expected_counts:
        raise SystemExit(f"{key} {data.get(key)} != {expected_counts}")
for key in ["dataset_source_files", "train_source_files"]:
    if data.get(key) != [expected_source]:
        raise SystemExit(f"{key} {data.get(key)} != {[expected_source]}")
if data.get("dataset_fingerprint") != expected_fingerprint:
    raise SystemExit(
        f"dataset_fingerprint {data.get('dataset_fingerprint')} != {expected_fingerprint}"
    )
if data.get("train_fingerprint") != expected_fingerprint:
    raise SystemExit(
        f"train_fingerprint {data.get('train_fingerprint')} != {expected_fingerprint}"
    )
if data.get("eval_source_files") != []:
    raise SystemExit(f"eval_source_files should be empty, got {data.get('eval_source_files')}")
if data.get("eval_source_sample_counts") != []:
    raise SystemExit(
        f"eval_source_sample_counts should be empty, got {data.get('eval_source_sample_counts')}"
    )

print(
    "qwen_sft_streaming_data_plan_verified: "
    f"total_samples={data['dataset_total_samples']} "
    f"train_samples={data['dataset_train_samples']} "
    f"eval_samples={data['dataset_eval_samples']} "
    f"source_files={data['dataset_source_files']} "
    f"fingerprint={data['dataset_fingerprint']} "
    f"tokenized_samples_materialized={data['tokenized_samples_materialized']}"
)
PY
