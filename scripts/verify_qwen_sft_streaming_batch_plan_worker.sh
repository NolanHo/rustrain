#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SFT_STREAMING_BATCH_PLAN_CONFIG:-configs/qwen_session_dp2_sft_max_samples.toml}"
EXPECTED_SOURCE="${RUSTRAIN_EXPECTED_STREAMING_SOURCE:-data/sft_toy/instructions.jsonl}"
EXPECTED_FINGERPRINT="${RUSTRAIN_EXPECTED_STREAMING_FINGERPRINT:-1f1a505dc2c37e79}"
OUTPUT="$(mktemp)"

cargo run -- qwen-sft-streaming-batch-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 2 \
  | tee "${OUTPUT}"

python - "${OUTPUT}" "${EXPECTED_SOURCE}" "${EXPECTED_FINGERPRINT}" <<'PY'
import json
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
expected_source = sys.argv[2]
expected_fingerprint = sys.argv[3]
start = text.find("{")
if start < 0:
    raise SystemExit(f"streaming batch plan output did not contain JSON: {text}")
data = json.loads(text[start:])

checks = {
    "world_size": 2,
    "local_batch_size": 1,
    "global_batch_size": 2,
    "train_steps": 1,
    "required_batches": 3,
    "train_batch_count": 2,
    "data_cursor_start": 2,
    "data_cursor_end": 4,
    "data_cursor_next": 4,
    "train_window_start_cursor": 2,
    "train_window_end_cursor_exclusive": 6,
    "dataset_total_samples": 4,
    "dataset_train_samples": 3,
    "dataset_eval_samples": 1,
    "dataset_order_seed": 777,
    "dataset_shuffle": True,
    "tokenizer_loaded": True,
    "tokenized_samples_materialized": True,
    "streaming_window_samples": 4,
    "materialized_input_max_delta": 0,
    "materialized_mask_max_delta": 0.0,
}
for key, expected in checks.items():
    if data.get(key) != expected:
        raise SystemExit(f"{key} {data.get(key)} != {expected}")

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

if data.get("dataset_source_files") != [expected_source]:
    raise SystemExit(f"dataset_source_files {data.get('dataset_source_files')} != {[expected_source]}")
if data.get("dataset_source_sample_counts") != [{"path": expected_source, "samples": 4}]:
    raise SystemExit(
        f"dataset_source_sample_counts {data.get('dataset_source_sample_counts')} did not match"
    )
if data.get("dataset_fingerprint") != expected_fingerprint:
    raise SystemExit(
        f"dataset_fingerprint {data.get('dataset_fingerprint')} != {expected_fingerprint}"
    )

batch_sequence_tokens = data.get("batch_sequence_tokens")
batch_masked_positions = data.get("batch_masked_positions")
batch_padding_tokens = data.get("batch_padding_tokens")
batch_token_fingerprints = data.get("batch_token_fingerprints")
if not all(isinstance(values, list) and len(values) == 2 for values in [
    batch_sequence_tokens,
    batch_masked_positions,
    batch_padding_tokens,
    batch_token_fingerprints,
]):
    raise SystemExit("expected two streaming batches worth of tokenized batch metadata")
if any(value <= 1 for value in batch_sequence_tokens):
    raise SystemExit(f"batch_sequence_tokens must be > 1, got {batch_sequence_tokens}")
if any(value <= 0 for value in batch_masked_positions):
    raise SystemExit(f"batch_masked_positions must be positive, got {batch_masked_positions}")
if any(not isinstance(value, str) or len(value) != 16 for value in batch_token_fingerprints):
    raise SystemExit(f"bad batch_token_fingerprints: {batch_token_fingerprints}")

print(
    "qwen_sft_streaming_batch_plan_verified: "
    f"streaming_window_samples={data['streaming_window_samples']} "
    f"batch_sequence_tokens={batch_sequence_tokens} "
    f"materialized_input_max_delta={data['materialized_input_max_delta']} "
    f"materialized_mask_max_delta={data['materialized_mask_max_delta']}"
)
PY
