#!/usr/bin/env bash
set -euo pipefail

BASE_OUTPUT="$(mktemp)"
RESUME_OUTPUT="$(mktemp)"
CONFIG="${RUSTRAIN_QWEN_LORA_SFT_CONFIG:-configs/qwen_lora_sft.toml}"
EXPECTED_COMPUTE_KIND="${RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND:-}"

cargo run -- train --config "${CONFIG}" | tee "${BASE_OUTPUT}"

ADAPTER_OUTPUT="$(
  python - "${BASE_OUTPUT}" <<'PY'
import pathlib
import sys

for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if line.startswith("adapter_checkpoint: "):
        print(line.split(": ", 1)[1])
        break
else:
    raise SystemExit("base run did not print adapter_checkpoint")
PY
)"

cargo run -- train --config "${CONFIG}" --resume-from "${ADAPTER_OUTPUT}" \
  | tee "${RESUME_OUTPUT}"

python - "${RESUME_OUTPUT}" "${EXPECTED_COMPUTE_KIND}" <<'PY'
import ast
import pathlib
import sys

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
    "dataset_total_samples",
    "dataset_total_tokens",
    "dataset_response_tokens",
    "dataset_masked_positions",
    "dataset_max_sequence_tokens",
    "dataset_order_seed",
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
if not float(values["final_loss"]) < float(values["initial_loss"]):
    raise SystemExit(
        f"resume loss did not improve: initial={values['initial_loss']} final={values['final_loss']}"
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
    f"initial_loss={values['initial_loss']} "
    f"final_loss={values['final_loss']} "
    f"reload_delta={values['reload_delta']} "
    f"tokens_per_second={values['tokens_per_second']} "
    f"samples_per_second={values['samples_per_second']}"
)
PY
