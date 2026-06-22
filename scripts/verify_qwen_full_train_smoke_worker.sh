#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

DTYPE="${RUSTRAIN_QWEN_FULL_TRAIN_DTYPE:-bf16}"
EXPECTED_DTYPE="${RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND:-${DTYPE}}"
DELTA_OUTPUT="${RUSTRAIN_QWEN_FULL_TRAIN_DELTA_OUTPUT:-/tmp/rustrain-qwen-full-train-delta-${DTYPE}.safetensors}"
OUTPUT="$(mktemp)"

cargo run -- qwen-full-train-smoke --dtype "${DTYPE}" --delta-output "${DELTA_OUTPUT}" \
  | tee "${OUTPUT}"

python - "${OUTPUT}" "${EXPECTED_DTYPE}" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from qwen_verify_utils import require_complete_qwen_base_model_path, require_complete_qwen_model_path

text = pathlib.Path(sys.argv[1]).read_text()
expected_dtype = sys.argv[2]
start = text.find("{")
end = text.rfind("}")
if start < 0 or end < start:
    raise SystemExit("qwen-full-train-smoke did not print a JSON summary")
summary = json.loads(text[start : end + 1])

required_fields = [
    "compute_kind",
    "train_steps",
    "step_losses",
    "initial_loss",
    "final_loss",
    "reloaded_loss",
    "reload_delta",
    "resume_loss",
    "continuous_second_loss",
    "resumed_second_loss",
    "second_step_delta",
    "delta_output",
    "optimizer_output",
    "manifest_output",
    "model_path",
    "trainable_tensors",
]
missing = [key for key in required_fields if key not in summary]
if missing:
    raise SystemExit(f"summary is missing fields: {missing}")
if summary["compute_kind"] != expected_dtype:
    raise SystemExit(
        f"compute_kind {summary['compute_kind']} does not match expected {expected_dtype}"
    )
if summary["train_steps"] != 1:
    raise SystemExit(f"expected train_steps 1, got {summary['train_steps']}")
step_losses = summary["step_losses"]
if len(step_losses) != 2:
    raise SystemExit(f"expected 2 step losses, got {step_losses}")
if not summary["final_loss"] < summary["initial_loss"]:
    raise SystemExit(
        f"full-train smoke did not improve loss: {summary['initial_loss']} -> {summary['final_loss']}"
    )
if step_losses != [summary["initial_loss"], summary["final_loss"]]:
    raise SystemExit(f"step_losses do not match initial/final losses: {step_losses}")
if abs(summary["final_loss"] - summary["reloaded_loss"]) > 1e-5:
    raise SystemExit("reloaded_loss does not match final_loss")
if float(summary["reload_delta"]) > 1e-5:
    raise SystemExit(f"reload_delta too large: {summary['reload_delta']}")
if float(summary["second_step_delta"]) > 1e-5:
    raise SystemExit(f"second_step_delta too large: {summary['second_step_delta']}")
if abs(summary["continuous_second_loss"] - summary["resumed_second_loss"]) > 1e-5:
    raise SystemExit("resumed second step does not match continuous second step")

for key in ["delta_output", "optimizer_output", "manifest_output"]:
    path = pathlib.Path(summary[key])
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"{key} missing or empty: {path}")

resolved_model_path = require_complete_qwen_model_path(summary["model_path"], "qwen-full-train-smoke summary")
manifest_path = pathlib.Path(summary["manifest_output"])
manifest = json.loads(manifest_path.read_text())
manifest_model_path = require_complete_qwen_base_model_path(manifest, manifest_path)
if manifest_model_path != resolved_model_path:
    raise SystemExit(
        f"manifest base_model_path {manifest_model_path} did not match summary model_path {resolved_model_path}"
    )

tensors = summary["trainable_tensors"]
if len(tensors) != 14:
    raise SystemExit(f"expected 14 trainable tensors, got {len(tensors)}")
names = {tensor["name"] for tensor in tensors}
required_names = {
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.norm.weight",
}
missing_names = sorted(required_names - names)
if missing_names:
    raise SystemExit(f"missing required trainable tensors: {missing_names}")
for tensor in tensors:
    if not tensor.get("grad_defined"):
        raise SystemExit(f"{tensor['name']} did not report a gradient")
    if float(tensor["grad_norm"]) <= 0.0:
        raise SystemExit(f"{tensor['name']} grad_norm must be positive")
    if float(tensor["delta_norm"]) <= 0.0:
        raise SystemExit(f"{tensor['name']} delta_norm must be positive")

print(
    "qwen_full_train_smoke_verified: "
    f"compute_kind={summary['compute_kind']} "
    f"loss={summary['initial_loss']}->{summary['final_loss']} "
    f"reload_delta={summary['reload_delta']} "
    f"second_step_delta={summary['second_step_delta']} "
    f"trainable_tensors={len(tensors)}"
)
PY
