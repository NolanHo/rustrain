#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_LORA_SFT_EVAL_PATHS_CONFIG:-configs/qwen_lora_sft_eval_paths.toml}"
OUTPUT="$(mktemp)"

cargo run -- train --config "${CONFIG}" | tee "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
import ast
import json
import pathlib
import re
import sys

def parse_source_sample_counts(text):
    entries = re.findall(r'QwenSftSourceSampleCount \{ path: "([^"]+)", samples: (\d+) \}', text)
    if not entries:
        raise SystemExit(f"dataset_source_sample_counts did not contain parseable entries: {text}")
    return [{"path": path, "samples": int(samples)} for path, samples in entries]

output_path = pathlib.Path(sys.argv[1])
values = {}
for line in output_path.read_text().splitlines():
    if ": " not in line:
        continue
    key, value = line.split(": ", 1)
    values[key] = value

required = [
    "adapter_manifest",
    "train_samples",
    "eval_samples",
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
    raise SystemExit(f"eval_paths run is missing fields: {missing}")

expected_sources = [
    "data/sft_toy/instructions.jsonl",
    "data/sft_toy/more_instructions.jsonl",
]
source_files = ast.literal_eval(values["dataset_source_files"])
if source_files != expected_sources:
    raise SystemExit(f"expected source_files {expected_sources}, got {source_files}")
if int(values["train_samples"]) != 6:
    raise SystemExit(f"expected train_samples 6, got {values['train_samples']}")
if int(values["eval_samples"]) != 2:
    raise SystemExit(f"expected eval_samples 2, got {values['eval_samples']}")
if int(values["dataset_total_samples"]) != 8:
    raise SystemExit(
        f"expected dataset_total_samples 8, got {values['dataset_total_samples']}"
    )
if values["streaming_train_batches"] != "true":
    raise SystemExit(
        f"expected streaming_train_batches true, got {values['streaming_train_batches']}"
    )
expected_counts = [
    {"path": "data/sft_toy/instructions.jsonl", "samples": 6},
    {"path": "data/sft_toy/more_instructions.jsonl", "samples": 2},
]
source_sample_counts = parse_source_sample_counts(values["dataset_source_sample_counts"])
if source_sample_counts != expected_counts:
    raise SystemExit(
        f"dataset_source_sample_counts {source_sample_counts} != {expected_counts}"
    )

manifest = json.loads(pathlib.Path(values["adapter_manifest"]).read_text())
if manifest.get("dataset_total_samples") != 8:
    raise SystemExit(
        f"manifest dataset_total_samples {manifest.get('dataset_total_samples')} != 8"
    )
if manifest.get("dataset_train_samples") != 6:
    raise SystemExit(
        f"manifest dataset_train_samples {manifest.get('dataset_train_samples')} != 6"
    )
if manifest.get("dataset_eval_samples") != 2:
    raise SystemExit(
        f"manifest dataset_eval_samples {manifest.get('dataset_eval_samples')} != 2"
    )
if manifest.get("dataset_source_files") != expected_sources:
    raise SystemExit(
        f"manifest source files {manifest.get('dataset_source_files')} != {expected_sources}"
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
    "qwen_lora_sft_eval_paths_verified: "
    f"train_samples={values['train_samples']} "
    f"eval_samples={values['eval_samples']} "
    f"dataset_source_files={source_files} "
    f"dataset_fingerprint={values['dataset_fingerprint']}"
)
PY
