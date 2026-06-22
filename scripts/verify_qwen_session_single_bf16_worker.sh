#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SESSION_SINGLE_BF16_CONFIG:-configs/qwen_session_single_bf16.toml}"
OUTPUT="$(mktemp)"

cargo run -- train --config "${CONFIG}" | tee "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
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
    "resumed_checkpoint",
    "train_steps",
    "step_losses",
    "first_step_grad_norm",
    "final_step_grad_norm",
    "tokens_per_second",
    "samples_per_second",
    "reload_delta",
    "second_step_delta",
    "manifest_output",
    "trainable_tensors",
]
missing = [key for key in required if key not in values]
if missing:
    raise SystemExit(f"bf16 run is missing fields: {missing}")
if values["compute_kind"] != "bf16":
    raise SystemExit(f"expected compute_kind bf16, got {values['compute_kind']}")
if values["resumed_checkpoint"] != "false":
    raise SystemExit(f"expected fresh run, got resumed_checkpoint={values['resumed_checkpoint']}")
if int(values["train_steps"]) != 2:
    raise SystemExit(f"expected train_steps 2, got {values['train_steps']}")
step_losses = ast.literal_eval(values["step_losses"])
if len(step_losses) != 3:
    raise SystemExit(f"expected 3 step losses, got {step_losses}")
if not step_losses[-1] < step_losses[0]:
    raise SystemExit(f"bf16 step losses did not improve: {step_losses}")
for key in [
    "first_step_grad_norm",
    "final_step_grad_norm",
    "tokens_per_second",
    "samples_per_second",
]:
    if float(values[key]) <= 0.0:
        raise SystemExit(f"{key} must be positive, got {values[key]}")
if float(values["reload_delta"]) > 1e-5:
    raise SystemExit(f"reload_delta too large: {values['reload_delta']}")
if float(values["second_step_delta"]) > 1e-5:
    raise SystemExit(f"second_step_delta too large: {values['second_step_delta']}")
if int(values["trainable_tensors"]) != 14:
    raise SystemExit(f"expected 14 trainable tensors, got {values['trainable_tensors']}")
manifest_path = pathlib.Path(values["manifest_output"])
manifest = json.loads(manifest_path.read_text())
require_complete_qwen_base_model_path(manifest, manifest_path)

print(
    "qwen_session_single_bf16_verified: "
    f"step_losses={step_losses} "
    f"reload_delta={values['reload_delta']} "
    f"second_step_delta={values['second_step_delta']} "
    f"trainable_tensors={values['trainable_tensors']}"
)
PY
