#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

BASE_OUTPUT="$(mktemp)"
RESUME_OUTPUT="$(mktemp)"
CONFIG="${RUSTRAIN_QWEN_LORA_SFT_CONFIG:-configs/qwen_lora_sft.toml}"
EXPECTED_COMPUTE_KIND="${RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND:-}"

cargo run -- train --config "${CONFIG}" | tee "${BASE_OUTPUT}"

BASE_MANIFEST_CURSOR="$(
  python - "${BASE_OUTPUT}" <<'PY'
import pathlib
import sys

values = {}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if ": " in line:
        key, value = line.split(": ", 1)
        values[key] = value
manifest = values.get("adapter_manifest")
cursor_next = values.get("data_cursor_next")
if manifest is None:
    raise SystemExit("base run did not print adapter_manifest")
if cursor_next is None:
    raise SystemExit("base run did not print data_cursor_next")
print(f"{manifest}\t{cursor_next}")
PY
)"
ADAPTER_MANIFEST="${BASE_MANIFEST_CURSOR%%$'\t'*}"
BASE_DATA_CURSOR_NEXT="${BASE_MANIFEST_CURSOR##*$'\t'}"

cargo run -- train --config "${CONFIG}" --resume-from "${ADAPTER_MANIFEST}" \
  | tee "${RESUME_OUTPUT}"

python - "${RESUME_OUTPUT}" "${EXPECTED_COMPUTE_KIND}" "${BASE_DATA_CURSOR_NEXT}" <<'PY'
import ast
import math
import pathlib
import sys

base_data_cursor_next = int(sys.argv[3])
values = {}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if ": " not in line:
        continue
    key, value = line.split(": ", 1)
    values[key] = value

required = [
    "compute_kind",
    "resume_from",
    "resumed_adapter",
    "adapter_manifest",
    "dataset_total_samples",
    "dataset_total_tokens",
    "dataset_response_tokens",
    "dataset_masked_positions",
    "dataset_max_sequence_tokens",
    "dataset_order_seed",
    "data_cursor_start",
    "data_cursor_end",
    "data_cursor_next",
    "batch_size",
    "global_batch_size",
    "gradient_accumulation_steps",
    "initial_loss",
    "final_loss",
    "reload_delta",
    "eval_reload_delta",
    "full_forward_reload_delta",
    "full_forward_merge_delta",
    "full_forward_unmerge_delta",
    "full_generate_reload_match",
    "full_generate_merge_match",
    "full_generate_new_token_ids",
    "first_step_grad_norm",
    "final_step_grad_norm",
    "final_step_clipped_grad_norm",
    "tokens_per_second",
    "samples_per_second",
]
missing = [key for key in required if key not in values]
if missing:
    raise SystemExit(f"resume run is missing fields: {missing}")
expected_compute_kind = sys.argv[2]
if expected_compute_kind and values["compute_kind"] != expected_compute_kind:
    raise SystemExit(
        f"compute_kind {values['compute_kind']} does not match expected {expected_compute_kind}"
    )
if values["resumed_adapter"] != "true":
    raise SystemExit("resume run did not report resumed_adapter: true")
for key in [
    "dataset_total_samples",
    "dataset_total_tokens",
    "dataset_response_tokens",
    "dataset_masked_positions",
    "dataset_max_sequence_tokens",
]:
    if int(values[key]) <= 0:
        raise SystemExit(f"{key} must be positive, got {values[key]}")
for key in ["initial_loss", "final_loss"]:
    if not math.isfinite(float(values[key])):
        raise SystemExit(f"{key} must be finite, got {values[key]}")
if int(values["data_cursor_start"]) != base_data_cursor_next:
    raise SystemExit(
        f"resume data_cursor_start {values['data_cursor_start']} did not continue from base data_cursor_next {base_data_cursor_next}"
    )
expected_cursor_end = int(values["data_cursor_start"]) + int(values["steps"]) * int(values["global_batch_size"])
if int(values["data_cursor_end"]) != expected_cursor_end:
    raise SystemExit(
        f"expected data_cursor_end {expected_cursor_end}, got {values['data_cursor_end']}"
    )
if int(values["data_cursor_next"]) != expected_cursor_end:
    raise SystemExit(
        f"expected data_cursor_next {expected_cursor_end}, got {values['data_cursor_next']}"
    )
for key in [
    "reload_delta",
    "eval_reload_delta",
    "full_forward_reload_delta",
]:
    if float(values[key]) > 1e-6:
        raise SystemExit(f"{key} too large: {values[key]}")
merge_tolerance = 5.0 if values["compute_kind"] == "bf16" else 1e-6
if float(values["full_forward_merge_delta"]) > merge_tolerance:
    raise SystemExit(
        f"full_forward_merge_delta too large: {values['full_forward_merge_delta']}"
    )
unmerge_tolerance = 5.0 if values["compute_kind"] == "bf16" else 1e-3
if float(values["full_forward_unmerge_delta"]) > unmerge_tolerance:
    raise SystemExit(
        f"full_forward_unmerge_delta too large: {values['full_forward_unmerge_delta']}"
    )
if values["full_generate_reload_match"] != "true":
    raise SystemExit(
        f"full_generate_reload_match must be true, got {values['full_generate_reload_match']}"
    )
if values["compute_kind"] != "bf16" and values["full_generate_merge_match"] != "true":
    raise SystemExit(
        f"full_generate_merge_match must be true, got {values['full_generate_merge_match']}"
    )
for key in [
    "first_step_grad_norm",
    "final_step_grad_norm",
    "final_step_clipped_grad_norm",
    "tokens_per_second",
    "samples_per_second",
]:
    if float(values[key]) <= 0.0:
        raise SystemExit(f"{key} must be positive, got {values[key]}")
generated = ast.literal_eval(values["full_generate_new_token_ids"])
if not generated:
    raise SystemExit("full_generate_new_token_ids must not be empty")

print(
    "qwen_lora_sft_resume_verified: "
    f"resume_from={values['resume_from']} "
    f"data_cursor_start={values['data_cursor_start']} "
    f"data_cursor_next={values['data_cursor_next']} "
    f"initial_loss={values['initial_loss']} "
    f"final_loss={values['final_loss']} "
    f"reload_delta={values['reload_delta']} "
    f"tokens_per_second={values['tokens_per_second']} "
    f"samples_per_second={values['samples_per_second']}"
)
PY
