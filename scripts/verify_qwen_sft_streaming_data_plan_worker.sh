#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SFT_STREAMING_DATA_PLAN_CONFIG:-configs/qwen_session_dp2_sft_max_samples.toml}"
EXPECTED_SOURCE="${RUSTRAIN_EXPECTED_STREAMING_SOURCE:-data/sft_toy/instructions.jsonl}"
EXPECTED_FINGERPRINT="${RUSTRAIN_EXPECTED_STREAMING_FINGERPRINT:-1f1a505dc2c37e79}"
OUTPUT="$(mktemp)"

cargo run -- qwen-sft-streaming-data-plan --config "${CONFIG}" | tee "${OUTPUT}"

python - "${OUTPUT}" "${EXPECTED_SOURCE}" "${EXPECTED_FINGERPRINT}" <<'PY'
import json
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
expected_source = sys.argv[2]
expected_fingerprint = sys.argv[3]
start = text.find("{")
if start < 0:
    raise SystemExit(f"streaming data plan output did not contain JSON: {text}")
data = json.loads(text[start:])

checks = {
    "max_samples": 4,
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
