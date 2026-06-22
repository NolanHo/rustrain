#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

BASE_OUTPUT="$(mktemp)"
RESUME_OUTPUT="$(mktemp)"

cargo run -- train --config configs/qwen_session_single.toml | tee "${BASE_OUTPUT}"

MANIFEST_OUTPUT="$(
  python - "${BASE_OUTPUT}" <<'PY'
import pathlib
import sys

for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if line.startswith("manifest_output: "):
        print(line.split(": ", 1)[1])
        break
else:
    raise SystemExit("base run did not print manifest_output")
PY
)"

cargo run -- train --config configs/qwen_session_single.toml --resume-from "${MANIFEST_OUTPUT}" \
  | tee "${RESUME_OUTPUT}"

python - "${RESUME_OUTPUT}" <<'PY'
import ast
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd() / "scripts"))
from qwen_verify_utils import require_complete_qwen_base_model_path

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
    "manifest_output",
    "first_step_grad_norm",
    "final_step_grad_norm",
    "tokens_per_second",
    "samples_per_second",
    "reload_delta",
    "second_step_delta",
]
missing = [key for key in required if key not in values]
if missing:
    raise SystemExit(f"resume run is missing fields: {missing}")
if values["resumed_checkpoint"] != "true":
    raise SystemExit("resume run did not report resumed_checkpoint: true")
base_manifest = pathlib.Path(values["resume_from"])
resume_manifest = pathlib.Path(values["manifest_output"])
if not base_manifest.is_file():
    raise SystemExit(f"resume_from manifest does not exist: {base_manifest}")
if not resume_manifest.is_file():
    raise SystemExit(f"resume manifest_output does not exist: {resume_manifest}")
if resume_manifest == base_manifest:
    raise SystemExit("resume run reused the base manifest path instead of writing a fresh manifest")
for manifest_path in [base_manifest, resume_manifest]:
    manifest = json.loads(manifest_path.read_text())
    require_complete_qwen_base_model_path(manifest, manifest_path)
if int(values["train_steps"]) != 2:
    raise SystemExit(f"expected train_steps 2, got {values['train_steps']}")
step_losses = ast.literal_eval(values["step_losses"])
if len(step_losses) != 3:
    raise SystemExit(f"expected 3 step losses, got {step_losses}")
if not step_losses[-1] < step_losses[0]:
    raise SystemExit(f"resume step losses did not improve: {step_losses}")
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

print(
    "qwen_session_single_resume_verified: "
    f"resume_from={values['resume_from']} "
    f"manifest_output={values['manifest_output']} "
    f"step_losses={step_losses} "
    f"reload_delta={values['reload_delta']} "
    f"second_step_delta={values['second_step_delta']}"
)
PY
