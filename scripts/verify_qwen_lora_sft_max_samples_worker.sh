#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ray/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_LORA_SFT_MAX_SAMPLES_CONFIG:-configs/qwen_lora_sft_max_samples.toml}"
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
    "adapter_manifest",
    "dataset_total_samples",
    "dataset_source_files",
    "dataset_source_sample_counts",
    "dataset_fingerprint",
    "streaming_train_batches",
    "reload_delta",
    "eval_reload_delta",
    "full_forward_reload_delta",
    "full_generate_reload_match",
]
missing = [key for key in required if key not in values]
if missing:
    raise SystemExit(f"max_samples run is missing fields: {missing}")

if int(values["dataset_total_samples"]) != expected_total:
    raise SystemExit(
        f"expected dataset_total_samples {expected_total}, got {values['dataset_total_samples']}"
    )
if values["streaming_train_batches"] != "true":
    raise SystemExit(
        f"expected streaming_train_batches true, got {values['streaming_train_batches']}"
    )
source_files = ast.literal_eval(values["dataset_source_files"])
if source_files != [expected_source]:
    raise SystemExit(f"expected source_files {[expected_source]}, got {source_files}")
expected_counts = [{"path": expected_source, "samples": expected_total}]
source_sample_counts = parse_source_sample_counts(values["dataset_source_sample_counts"])
if source_sample_counts != expected_counts:
    raise SystemExit(
        f"dataset_source_sample_counts {source_sample_counts} != {expected_counts}"
    )

manifest = json.loads(pathlib.Path(values["adapter_manifest"]).read_text())
require_complete_qwen_base_model_path(manifest, values["adapter_manifest"])
if manifest.get("dataset_total_samples") != expected_total:
    raise SystemExit(
        f"manifest dataset_total_samples {manifest.get('dataset_total_samples')} != {expected_total}"
    )
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
if manifest.get("streaming_train_batches") is not True:
    raise SystemExit(
        f"manifest streaming_train_batches {manifest.get('streaming_train_batches')} is not true"
    )
for key in ["reload_delta", "eval_reload_delta", "full_forward_reload_delta"]:
    if float(values[key]) > 1e-6:
        raise SystemExit(f"{key} too large: {values[key]}")
if values["full_generate_reload_match"] != "true":
    raise SystemExit("full_generate_reload_match must be true")

print(
    "qwen_lora_sft_max_samples_verified: "
    f"dataset_total_samples={values['dataset_total_samples']} "
    f"dataset_source_files={source_files} "
    f"dataset_source_sample_counts={values['dataset_source_sample_counts']} "
    f"dataset_fingerprint={values['dataset_fingerprint']}"
)
PY
