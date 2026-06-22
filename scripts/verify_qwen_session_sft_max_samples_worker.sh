#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SESSION_SINGLE_SFT_MAX_SAMPLES_CONFIG:-configs/qwen_session_single_sft_max_samples.toml}"
EXPECTED_TOTAL="${RUSTRAIN_EXPECTED_MAX_SAMPLES:-4}"
EXPECTED_SOURCE="${RUSTRAIN_EXPECTED_MAX_SAMPLES_SOURCE:-data/sft_toy/instructions.jsonl}"
OUTPUT="$(mktemp)"

cargo run -- train --config "${CONFIG}" | tee "${OUTPUT}"

python - "${OUTPUT}" "${EXPECTED_TOTAL}" "${EXPECTED_SOURCE}" <<'PY'
import ast
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from qwen_verify_utils import require_complete_qwen_base_model_path

def parse_source_sample_counts(text):
    entries = re.findall(r'QwenSftSourceSampleCount \{ path: "([^"]+)", samples: (\d+) \}', text)
    if not entries:
        raise SystemExit(f"dataset_source_sample_counts did not contain parseable entries: {text}")
    return [{"path": path, "samples": int(samples)} for path, samples in entries]

output_path = pathlib.Path(sys.argv[1])
expected_total = int(sys.argv[2])
expected_source = sys.argv[3]

values = {}
for line in output_path.read_text().splitlines():
    if ": " not in line:
        continue
    key, value = line.split(": ", 1)
    values[key] = value

required = [
    "manifest_output",
    "dataset_total_samples",
    "dataset_train_samples",
    "dataset_eval_samples",
    "dataset_source_files",
    "dataset_source_sample_counts",
    "dataset_fingerprint",
    "data_cursor_start",
    "data_cursor_next",
    "streaming_train_batches",
    "reload_delta",
    "second_step_delta",
    "trainable_tensors",
]
missing = [key for key in required if key not in values]
if missing:
    raise SystemExit(f"max_samples session run is missing fields: {missing}")

if int(values["dataset_total_samples"]) != expected_total:
    raise SystemExit(
        f"expected dataset_total_samples {expected_total}, got {values['dataset_total_samples']}"
    )
if int(values["dataset_train_samples"]) != 3:
    raise SystemExit(f"expected dataset_train_samples 3, got {values['dataset_train_samples']}")
if int(values["dataset_eval_samples"]) != 1:
    raise SystemExit(f"expected dataset_eval_samples 1, got {values['dataset_eval_samples']}")
source_files = ast.literal_eval(values["dataset_source_files"])
if source_files != [expected_source]:
    raise SystemExit(f"expected source_files {[expected_source]}, got {source_files}")
expected_counts = [{"path": expected_source, "samples": expected_total}]
source_sample_counts = parse_source_sample_counts(values["dataset_source_sample_counts"])
if source_sample_counts != expected_counts:
    raise SystemExit(
        f"dataset_source_sample_counts {source_sample_counts} != {expected_counts}"
    )
if int(values["data_cursor_start"]) != 0:
    raise SystemExit(f"expected data_cursor_start 0, got {values['data_cursor_start']}")
if int(values["data_cursor_next"]) != 1:
    raise SystemExit(f"expected data_cursor_next 1, got {values['data_cursor_next']}")
if values["streaming_train_batches"] != "true":
    raise SystemExit(
        f"expected streaming_train_batches true, got {values['streaming_train_batches']}"
    )
if int(values["trainable_tensors"]) != 14:
    raise SystemExit(f"expected trainable_tensors 14, got {values['trainable_tensors']}")
for key in ["reload_delta", "second_step_delta"]:
    if float(values[key]) > 1e-5:
        raise SystemExit(f"{key} too large: {values[key]}")

manifest = json.loads(pathlib.Path(values["manifest_output"]).read_text())
require_complete_qwen_base_model_path(manifest, values["manifest_output"])
if manifest.get("dataset_source_files") != source_files:
    raise SystemExit(
        f"manifest source files {manifest.get('dataset_source_files')} != {source_files}"
    )
if manifest.get("dataset_source_sample_counts") != expected_counts:
    raise SystemExit(
        f"manifest source sample counts {manifest.get('dataset_source_sample_counts')} != {expected_counts}"
    )
if manifest.get("dataset_fingerprint") != values["dataset_fingerprint"]:
    raise SystemExit("manifest dataset_fingerprint does not match stdout")
if int(manifest.get("data_cursor_next")) != int(values["data_cursor_next"]):
    raise SystemExit(
        f"manifest data_cursor_next {manifest.get('data_cursor_next')} != {values['data_cursor_next']}"
    )
if int(manifest.get("data_cursor_start")) != int(values["data_cursor_start"]):
    raise SystemExit(
        f"manifest data_cursor_start {manifest.get('data_cursor_start')} != {values['data_cursor_start']}"
    )
if manifest.get("streaming_train_batches") is not True:
    raise SystemExit(
        f"manifest streaming_train_batches {manifest.get('streaming_train_batches')} is not true"
    )

print(
    "qwen_session_sft_max_samples_verified: "
    f"dataset_total_samples={values['dataset_total_samples']} "
    f"dataset_train_samples={values['dataset_train_samples']} "
    f"dataset_eval_samples={values['dataset_eval_samples']} "
    f"dataset_source_files={source_files} "
    f"dataset_source_sample_counts={values['dataset_source_sample_counts']} "
    f"dataset_fingerprint={values['dataset_fingerprint']} "
    f"data_cursor_next={values['data_cursor_next']} "
    f"streaming_train_batches={values['streaming_train_batches']}"
)
PY
