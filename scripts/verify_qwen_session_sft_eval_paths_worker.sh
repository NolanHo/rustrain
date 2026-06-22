#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SESSION_SINGLE_SFT_EVAL_PATHS_CONFIG:-configs/qwen_session_single_sft_eval_paths.toml}"
OUTPUT="$(mktemp)"

cargo run -- train --config "${CONFIG}" | tee "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
import ast
import json
import pathlib
import sys

output_path = pathlib.Path(sys.argv[1])
values = {}
for line in output_path.read_text().splitlines():
    if ": " not in line:
        continue
    key, value = line.split(": ", 1)
    values[key] = value

required = [
    "manifest_output",
    "dataset_total_samples",
    "dataset_total_tokens",
    "dataset_train_samples",
    "dataset_eval_samples",
    "dataset_source_files",
    "dataset_source_sample_counts",
    "dataset_fingerprint",
    "dataset_order_seed",
    "batch_size",
    "sequence_tokens",
    "data_cursor_next",
    "streaming_train_batches",
    "reload_delta",
    "second_step_delta",
]
missing = [key for key in required if key not in values]
if missing:
    raise SystemExit(f"eval_paths session run is missing fields: {missing}")

expected_sources = [
    "data/sft_toy/eval_instructions.jsonl",
    "data/sft_toy/instructions.jsonl",
]
source_files = ast.literal_eval(values["dataset_source_files"])
if source_files != expected_sources:
    raise SystemExit(f"expected source_files {expected_sources}, got {source_files}")
if int(values["dataset_train_samples"]) != 6:
    raise SystemExit(f"expected dataset_train_samples 6, got {values['dataset_train_samples']}")
if int(values["dataset_eval_samples"]) != 2:
    raise SystemExit(f"expected dataset_eval_samples 2, got {values['dataset_eval_samples']}")
if int(values["dataset_total_samples"]) != 8:
    raise SystemExit(f"expected dataset_total_samples 8, got {values['dataset_total_samples']}")
if int(values["dataset_order_seed"]) != 777:
    raise SystemExit(f"expected dataset_order_seed 777, got {values['dataset_order_seed']}")
for expected in ["eval_instructions.jsonl\", samples: 2", "instructions.jsonl\", samples: 6"]:
    if expected not in values["dataset_source_sample_counts"]:
        raise SystemExit(
            f"dataset_source_sample_counts missing {expected}: {values['dataset_source_sample_counts']}"
        )
for key in ["batch_size", "sequence_tokens", "dataset_total_tokens"]:
    if int(values[key]) <= 0:
        raise SystemExit(f"{key} must be positive, got {values[key]}")
for key in ["reload_delta", "second_step_delta"]:
    if float(values[key]) > 1e-5:
        raise SystemExit(f"{key} too large: {values[key]}")
if values["streaming_train_batches"] != "true":
    raise SystemExit(
        f"expected streaming_train_batches true, got {values['streaming_train_batches']}"
    )

manifest = json.loads(pathlib.Path(values["manifest_output"]).read_text())
if manifest.get("dataset_source_files") != expected_sources:
    raise SystemExit(
        f"manifest source files {manifest.get('dataset_source_files')} != {expected_sources}"
    )
if manifest.get("dataset_fingerprint") != values["dataset_fingerprint"]:
    raise SystemExit("manifest dataset_fingerprint does not match stdout")
if manifest.get("dataset_shuffle") is not True:
    raise SystemExit(f"manifest dataset_shuffle {manifest.get('dataset_shuffle')} is not true")
if int(manifest.get("data_cursor_next")) != int(values["data_cursor_next"]):
    raise SystemExit(
        f"manifest data_cursor_next {manifest.get('data_cursor_next')} != {values['data_cursor_next']}"
    )
if manifest.get("streaming_train_batches") is not True:
    raise SystemExit(
        f"manifest streaming_train_batches {manifest.get('streaming_train_batches')} is not true"
    )

print(
    "qwen_session_sft_eval_paths_verified: "
    f"train_samples={values['dataset_train_samples']} "
    f"eval_samples={values['dataset_eval_samples']} "
    f"dataset_source_files={source_files} "
    f"dataset_fingerprint={values['dataset_fingerprint']} "
    f"reload_delta={values['reload_delta']} "
    f"second_step_delta={values['second_step_delta']} "
    f"streaming_train_batches={values['streaming_train_batches']}"
)
PY
