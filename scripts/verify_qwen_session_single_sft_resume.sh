#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SESSION_SINGLE_SFT_CONFIG:-configs/qwen_session_single_sft.toml}"
EXPECTED_TRAINABLE_TENSORS="${RUSTRAIN_EXPECTED_QWEN_TRAINABLE_TENSORS:-}"
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

python - "${RESUME_OUTPUT}" "${BASE_DATA_CURSOR_NEXT}" "${EXPECTED_TRAINABLE_TENSORS}" <<'PY'
import ast
import json
import math
import pathlib
import sys

base_data_cursor_next = int(sys.argv[2])
expected_trainable_tensors = sys.argv[3]
values = {}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if ": " not in line:
        continue
    key, value = line.split(": ", 1)
    values[key] = value

required = [
    "resume_from",
    "resumed_checkpoint",
    "train_steps",
    "step_losses",
    "first_step_grad_norm",
    "final_step_grad_norm",
    "tokens_per_second",
    "samples_per_second",
    "dataset_total_samples",
    "dataset_total_tokens",
    "dataset_train_samples",
    "dataset_eval_samples",
    "dataset_source_files",
    "dataset_fingerprint",
    "dataset_order_seed",
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
]
missing = [key for key in required if key not in values]
if missing:
    raise SystemExit(f"resume run is missing fields: {missing}")
if values["resumed_checkpoint"] != "true":
    raise SystemExit("resume run did not report resumed_checkpoint: true")
if int(values["train_steps"]) != 2:
    raise SystemExit(f"expected train_steps 2, got {values['train_steps']}")
step_losses = ast.literal_eval(values["step_losses"])
if len(step_losses) != 3:
    raise SystemExit(f"expected 3 step losses, got {step_losses}")
if not all(math.isfinite(float(loss)) for loss in step_losses):
    raise SystemExit(f"resume step losses must be finite: {step_losses}")
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
    "dataset_total_tokens",
    "dataset_train_samples",
    "dataset_eval_samples",
    "batch_size",
    "sequence_tokens",
    "trainable_tensors",
]:
    if int(values[key]) <= 0:
        raise SystemExit(f"{key} must be positive, got {values[key]}")
if expected_trainable_tensors and int(values["trainable_tensors"]) != int(expected_trainable_tensors):
    raise SystemExit(
        f"expected {expected_trainable_tensors} trainable tensors, got {values['trainable_tensors']}"
    )
dataset_source_files = ast.literal_eval(values["dataset_source_files"])
if not dataset_source_files:
    raise SystemExit("dataset_source_files must not be empty")
if not all(str(path).endswith(".jsonl") for path in dataset_source_files):
    raise SystemExit(f"dataset_source_files must only contain JSONL paths, got {dataset_source_files}")
if not values["dataset_fingerprint"]:
    raise SystemExit("dataset_fingerprint must not be empty")
manifest = json.loads(pathlib.Path(values["manifest_output"]).read_text())
if manifest.get("dataset_source_files") != dataset_source_files:
    raise SystemExit(
        f"manifest dataset_source_files {manifest.get('dataset_source_files')} did not match summary {dataset_source_files}"
    )
if manifest.get("dataset_fingerprint") != values["dataset_fingerprint"]:
    raise SystemExit(
        f"manifest dataset_fingerprint {manifest.get('dataset_fingerprint')} did not match summary {values['dataset_fingerprint']}"
    )
if int(values["dataset_order_seed"]) != 777:
    raise SystemExit(f"expected dataset_order_seed 777, got {values['dataset_order_seed']}")
if int(values["data_cursor_start"]) != base_data_cursor_next:
    raise SystemExit(
        f"resume data_cursor_start {values['data_cursor_start']} did not continue from base data_cursor_next {base_data_cursor_next}"
    )
expected_cursor_end = int(values["data_cursor_start"]) + int(values["train_steps"]) * int(values["batch_size"])
if int(values["data_cursor_end"]) != expected_cursor_end:
    raise SystemExit(
        f"expected data_cursor_end {expected_cursor_end}, got {values['data_cursor_end']}"
    )
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
    expected_epoch = cursor // train_samples
    expected_offset = cursor % train_samples
    if int(values[epoch_key]) != expected_epoch:
        raise SystemExit(
            f"expected {epoch_key} {expected_epoch}, got {values[epoch_key]} from {cursor_key}={cursor}"
        )
    if int(values[offset_key]) != expected_offset:
        raise SystemExit(
            f"expected {offset_key} {expected_offset}, got {values[offset_key]} from {cursor_key}={cursor}"
        )
if float(values["reload_delta"]) > 1e-5:
    raise SystemExit(f"reload_delta too large: {values['reload_delta']}")
if float(values["second_step_delta"]) > 1e-5:
    raise SystemExit(f"second_step_delta too large: {values['second_step_delta']}")

print(
    "qwen_session_single_sft_resume_verified: "
    f"resume_from={values['resume_from']} "
    f"step_losses={step_losses} "
    f"dataset_total_samples={values['dataset_total_samples']} "
    f"dataset_total_tokens={values['dataset_total_tokens']} "
    f"data_cursor_start={values['data_cursor_start']} "
    f"data_cursor_next={values['data_cursor_next']} "
    f"data_epoch_next={values['data_epoch_next']} "
    f"data_sample_offset_next={values['data_sample_offset_next']} "
    f"reload_delta={values['reload_delta']} "
    f"second_step_delta={values['second_step_delta']}"
)
PY
