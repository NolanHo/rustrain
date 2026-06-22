#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_LORA_SFT_NO_SHUFFLE_CONFIG:-configs/qwen_lora_sft_no_shuffle.toml}"
OUTPUT="$(mktemp)"

cargo run -- train --config "${CONFIG}" | tee "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
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
    "dataset_order_seed",
    "dataset_shuffle",
    "streaming_train_batches",
    "sequence_tokens",
    "reload_delta",
    "eval_reload_delta",
    "full_forward_reload_delta",
    "full_generate_reload_match",
]
missing = [key for key in required if key not in values]
if missing:
    raise SystemExit(f"no-shuffle run is missing fields: {missing}")

if values["dataset_shuffle"] != "false":
    raise SystemExit(f"expected dataset_shuffle false, got {values['dataset_shuffle']}")
if values["streaming_train_batches"] != "true":
    raise SystemExit(
        f"expected streaming_train_batches true, got {values['streaming_train_batches']}"
    )
if int(values["dataset_order_seed"]) != 777:
    raise SystemExit(f"expected dataset_order_seed 777, got {values['dataset_order_seed']}")
if int(values["dataset_total_samples"]) != 4:
    raise SystemExit(f"expected dataset_total_samples 4, got {values['dataset_total_samples']}")
if int(values["sequence_tokens"]) != 13:
    raise SystemExit(
        "shuffle=false should use the first JSONL training sample before shuffling; "
        f"expected sequence_tokens 13, got {values['sequence_tokens']}"
    )
expected_sources = ["data/sft_toy/instructions.jsonl"]
source_files = ast.literal_eval(values["dataset_source_files"])
if source_files != expected_sources:
    raise SystemExit(f"expected dataset_source_files {expected_sources}, got {source_files}")
expected_counts = [{"path": "data/sft_toy/instructions.jsonl", "samples": 4}]
source_sample_counts = parse_source_sample_counts(values["dataset_source_sample_counts"])
if source_sample_counts != expected_counts:
    raise SystemExit(
        f"dataset_source_sample_counts {source_sample_counts} != {expected_counts}"
    )

manifest = json.loads(pathlib.Path(values["adapter_manifest"]).read_text())
require_complete_qwen_base_model_path(manifest, values["adapter_manifest"])
if manifest.get("dataset_shuffle") is not False:
    raise SystemExit(f"manifest dataset_shuffle should be false, got {manifest.get('dataset_shuffle')}")
if manifest.get("dataset_order_seed") != 777:
    raise SystemExit(f"manifest dataset_order_seed should be 777, got {manifest.get('dataset_order_seed')}")
if manifest.get("dataset_total_samples") != 4:
    raise SystemExit(
        f"manifest dataset_total_samples {manifest.get('dataset_total_samples')} != 4"
    )
if manifest.get("dataset_source_files") != expected_sources:
    raise SystemExit(
        f"manifest dataset_source_files {manifest.get('dataset_source_files')} != {expected_sources}"
    )
if manifest.get("dataset_source_sample_counts") != expected_counts:
    raise SystemExit(
        f"manifest dataset_source_sample_counts {manifest.get('dataset_source_sample_counts')} != {expected_counts}"
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
    "qwen_lora_sft_no_shuffle_verified: "
    f"dataset_shuffle={values['dataset_shuffle']} "
    f"dataset_order_seed={values['dataset_order_seed']} "
    f"sequence_tokens={values['sequence_tokens']} "
    f"dataset_fingerprint={values['dataset_fingerprint']}"
)
PY
