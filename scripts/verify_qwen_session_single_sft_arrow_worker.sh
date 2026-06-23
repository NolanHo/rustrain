#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SESSION_SINGLE_SFT_ARROW_CONFIG:-configs/qwen_session_single_sft_arrow.toml}"
EXPECTED_TRAINABLE_TENSORS="${RUSTRAIN_EXPECTED_QWEN_TRAINABLE_TENSORS:-14}"
EXPECTED_COMPUTE_KIND="${RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND:-fp32}"
BASE_OUTPUT="$(mktemp)"
RESUME_OUTPUT="$(mktemp)"

cargo run -- train --config "${CONFIG}" | tee "${BASE_OUTPUT}"

BASE_CURSOR_NEXT="$(
  python - "${BASE_OUTPUT}" <<'PY'
import pathlib
import sys

values = {}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if ": " in line:
        key, value = line.split(": ", 1)
        values[key] = value
manifest = values.get("manifest_output")
cursor_next = values.get("data_cursor_next")
if manifest is None:
    raise SystemExit("base run did not print manifest_output")
if cursor_next is None:
    raise SystemExit("base run did not print data_cursor_next")
print(f"{manifest}\t{cursor_next}")
PY
)"
MANIFEST_OUTPUT="${BASE_CURSOR_NEXT%%$'\t'*}"
BASE_DATA_CURSOR_NEXT="${BASE_CURSOR_NEXT##*$'\t'}"

cargo run -- train --config "${CONFIG}" --resume-from "${MANIFEST_OUTPUT}" \
  | tee "${RESUME_OUTPUT}"

python - "${RESUME_OUTPUT}" "${BASE_DATA_CURSOR_NEXT}" "${EXPECTED_TRAINABLE_TENSORS}" "${EXPECTED_COMPUTE_KIND}" <<'PY'
import ast
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from qwen_verify_utils import require_complete_qwen_base_model_path

values = {}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if ": " not in line:
        continue
    key, value = line.split(": ", 1)
    values[key] = value

base_data_cursor_next = int(sys.argv[2])
expected_trainable_tensors = int(sys.argv[3])
expected_compute_kind = sys.argv[4]
required = [
    "resume_from",
    "resumed_checkpoint",
    "compute_kind",
    "train_steps",
    "step_losses",
    "first_step_grad_norm",
    "final_step_grad_norm",
    "tokens_per_second",
    "samples_per_second",
    "dataset_total_samples",
    "dataset_train_samples",
    "dataset_eval_samples",
    "dataset_source_files",
    "dataset_source_sample_counts",
    "dataset_fingerprint",
    "dataset_order_seed",
    "dataset_shuffle",
    "streaming_train_batches",
    "data_cursor_start",
    "data_cursor_end",
    "data_cursor_next",
    "data_epoch_start",
    "data_epoch_end",
    "data_epoch_next",
    "data_sample_offset_start",
    "data_sample_offset_end",
    "data_sample_offset_next",
    "batch_size",
    "sequence_tokens",
    "reload_delta",
    "second_step_delta",
    "trainable_tensors",
    "manifest_output",
]
missing = [key for key in required if key not in values]
if missing:
    raise SystemExit(f"resume run is missing fields: {missing}")
if values["resumed_checkpoint"] != "true":
    raise SystemExit(f"expected resumed_checkpoint true, got {values['resumed_checkpoint']}")
if values["compute_kind"] != expected_compute_kind:
    raise SystemExit(f"expected compute_kind {expected_compute_kind}, got {values['compute_kind']}")
if values["streaming_train_batches"] != "true":
    raise SystemExit(f"expected streaming_train_batches true, got {values['streaming_train_batches']}")
if "dataset_total_tokens" in values:
    raise SystemExit(
        "instruction_arrow trainer runtime must not report dataset_total_tokens; "
        "full token totals are reserved for materialized batch-plan parity"
    )
if values.get("streaming_index_cache_path"):
    raise SystemExit("instruction_arrow trainer path must not report streaming_index_cache_path")
if int(values["train_steps"]) != 2:
    raise SystemExit(f"expected train_steps 2, got {values['train_steps']}")
if int(values["trainable_tensors"]) != expected_trainable_tensors:
    raise SystemExit(
        f"expected {expected_trainable_tensors} trainable tensors, got {values['trainable_tensors']}"
    )
step_losses = ast.literal_eval(values["step_losses"])
if len(step_losses) != 3:
    raise SystemExit(f"expected 3 step losses, got {step_losses}")
if not all(math.isfinite(float(loss)) for loss in step_losses):
    raise SystemExit(f"step losses must be finite: {step_losses}")
for key in [
    "first_step_grad_norm",
    "final_step_grad_norm",
    "tokens_per_second",
    "samples_per_second",
]:
    if float(values[key]) <= 0.0:
        raise SystemExit(f"{key} must be positive, got {values[key]}")
for key in [
    "dataset_total_samples",
    "dataset_train_samples",
    "dataset_eval_samples",
    "batch_size",
    "sequence_tokens",
]:
    if int(values[key]) <= 0:
        raise SystemExit(f"{key} must be positive, got {values[key]}")
if int(values["dataset_total_samples"]) != 32:
    raise SystemExit(f"expected bounded Arrow dataset_total_samples 32, got {values['dataset_total_samples']}")
if int(values["dataset_train_samples"]) != 24 or int(values["dataset_eval_samples"]) != 8:
    raise SystemExit(
        f"expected 24/8 train/eval split, got {values['dataset_train_samples']}/{values['dataset_eval_samples']}"
    )
dataset_source_files = ast.literal_eval(values["dataset_source_files"])
if len(dataset_source_files) != 1 or not dataset_source_files[0].endswith(".arrow"):
    raise SystemExit(f"expected one Arrow source file, got {dataset_source_files}")
source_counts_text = values["dataset_source_sample_counts"]
if source_counts_text.count("QwenSftSourceSampleCount") != 1 or "samples: 32" not in source_counts_text:
    raise SystemExit(f"expected Arrow source sample count 32, got {source_counts_text}")
if int(values["dataset_order_seed"]) != 777:
    raise SystemExit(f"expected dataset_order_seed 777, got {values['dataset_order_seed']}")
if values["dataset_shuffle"] != "false":
    raise SystemExit(f"expected dataset_shuffle false, got {values['dataset_shuffle']}")
if int(values["data_cursor_start"]) != base_data_cursor_next:
    raise SystemExit(
        f"resume data_cursor_start {values['data_cursor_start']} did not continue from base data_cursor_next {base_data_cursor_next}"
    )
expected_cursor_end = int(values["data_cursor_start"]) + int(values["train_steps"]) * int(values["batch_size"])
if int(values["data_cursor_end"]) != expected_cursor_end:
    raise SystemExit(f"expected data_cursor_end {expected_cursor_end}, got {values['data_cursor_end']}")
if int(values["data_cursor_next"]) != int(values["data_cursor_end"]):
    raise SystemExit(
        f"expected data_cursor_next to equal data_cursor_end, got {values['data_cursor_next']} vs {values['data_cursor_end']}"
    )
train_samples = int(values["dataset_train_samples"])
for cursor_key, epoch_key, offset_key in [
    ("data_cursor_start", "data_epoch_start", "data_sample_offset_start"),
    ("data_cursor_end", "data_epoch_end", "data_sample_offset_end"),
    ("data_cursor_next", "data_epoch_next", "data_sample_offset_next"),
]:
    cursor = int(values[cursor_key])
    if int(values[epoch_key]) != cursor // train_samples:
        raise SystemExit(f"{epoch_key} mismatch for {cursor_key}={cursor}")
    if int(values[offset_key]) != cursor % train_samples:
        raise SystemExit(f"{offset_key} mismatch for {cursor_key}={cursor}")
if float(values["reload_delta"]) > 1e-5:
    raise SystemExit(f"reload_delta too large: {values['reload_delta']}")
if float(values["second_step_delta"]) > 1e-5:
    raise SystemExit(f"second_step_delta too large: {values['second_step_delta']}")

manifest = json.loads(pathlib.Path(values["manifest_output"]).read_text())
require_complete_qwen_base_model_path(manifest, values["manifest_output"])
if manifest.get("dataset_source_files") != dataset_source_files:
    raise SystemExit("manifest dataset_source_files did not match trainer output")
if manifest.get("dataset_fingerprint") != values["dataset_fingerprint"]:
    raise SystemExit("manifest dataset_fingerprint did not match trainer output")
if manifest.get("streaming_train_batches") is not True:
    raise SystemExit(f"manifest streaming_train_batches is not true: {manifest.get('streaming_train_batches')}")

print(
    "qwen_session_single_sft_arrow_verified: "
    f"resume_from={values['resume_from']} "
    f"step_losses={step_losses} "
    f"dataset_total_samples={values['dataset_total_samples']} "
    f"dataset_train_samples={values['dataset_train_samples']} "
    f"dataset_eval_samples={values['dataset_eval_samples']} "
    f"data_cursor_start={values['data_cursor_start']} "
    f"data_cursor_next={values['data_cursor_next']} "
    f"reload_delta={values['reload_delta']} "
    f"second_step_delta={values['second_step_delta']}"
)
PY
