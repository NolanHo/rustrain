#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ray/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SESSION_LAYERS_CONFIG:-configs/qwen_session_single_layers03.toml}"
EXPECTED_COMPUTE_KIND="${RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND:-fp32}"
EXPECTED_TRAINABLE_TENSORS="${RUSTRAIN_EXPECTED_QWEN_TRAINABLE_TENSORS:-50}"
OUTPUT="$(mktemp)"

cargo run -- train --config "${CONFIG}" | tee "${OUTPUT}"

python - "${OUTPUT}" "${EXPECTED_COMPUTE_KIND}" "${EXPECTED_TRAINABLE_TENSORS}" <<'PY'
import ast
import json
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

required = [
    "compute_kind",
    "train_steps",
    "step_losses",
    "first_step_grad_norm",
    "final_step_grad_norm",
    "tokens_per_second",
    "samples_per_second",
    "memory_rss_mb",
    "gpu_memory_allocated_mb",
    "reload_delta",
    "second_step_delta",
    "manifest_output",
    "trainable_tensors",
]
missing = [key for key in required if key not in values]
if missing:
    raise SystemExit(f"layer03 run is missing fields: {missing}")

expected_compute_kind = sys.argv[2]
expected_trainable_tensors = int(sys.argv[3])
if values["compute_kind"] != expected_compute_kind:
    raise SystemExit(
        f"expected compute_kind {expected_compute_kind}, got {values['compute_kind']}"
    )
if int(values["train_steps"]) != 1:
    raise SystemExit(f"expected train_steps 1, got {values['train_steps']}")
if int(values["trainable_tensors"]) != expected_trainable_tensors:
    raise SystemExit(
        f"expected {expected_trainable_tensors} trainable tensors, got {values['trainable_tensors']}"
    )

step_losses = ast.literal_eval(values["step_losses"])
if len(step_losses) != 2:
    raise SystemExit(f"expected 2 step losses, got {step_losses}")
if not step_losses[-1] < step_losses[0]:
    raise SystemExit(f"layer03 step losses did not improve: {step_losses}")

for key in [
    "first_step_grad_norm",
    "final_step_grad_norm",
    "tokens_per_second",
    "samples_per_second",
    "memory_rss_mb",
    "gpu_memory_allocated_mb",
]:
    if float(values[key]) <= 0.0:
        raise SystemExit(f"{key} must be positive, got {values[key]}")
if float(values["reload_delta"]) > 1e-5:
    raise SystemExit(f"reload_delta too large: {values['reload_delta']}")
if float(values["second_step_delta"]) > 1e-5:
    raise SystemExit(f"second_step_delta too large: {values['second_step_delta']}")
manifest_path = pathlib.Path(values["manifest_output"])
manifest = json.loads(manifest_path.read_text())
require_complete_qwen_base_model_path(manifest, manifest_path)

print(
    "qwen_session_layers03_verified: "
    f"compute_kind={values['compute_kind']} "
    f"step_losses={step_losses} "
    f"trainable_tensors={values['trainable_tensors']} "
    f"reload_delta={values['reload_delta']} "
    f"second_step_delta={values['second_step_delta']}"
)
PY
