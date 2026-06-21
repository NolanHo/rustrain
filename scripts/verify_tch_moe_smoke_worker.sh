#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT="$(mktemp)"

cargo run -- tch-moe-smoke | tee "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
import json
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
start = text.find("{")
end = text.rfind("}")
if start < 0 or end < start:
    raise SystemExit("tch-moe-smoke did not print a JSON summary")
summary = json.loads(text[start : end + 1])

required = [
    "device",
    "train_steps",
    "learning_rate",
    "aux_loss_weight",
    "tokens",
    "hidden_size",
    "expert_hidden_size",
    "num_experts",
    "top_k",
    "initial_loss",
    "final_loss",
    "initial_task_loss",
    "final_task_loss",
    "initial_load_balance_loss",
    "final_load_balance_loss",
    "checkpoint_output",
    "optimizer_output",
    "manifest_output",
    "reloaded_loss",
    "reload_delta",
    "reload_optimizer_max_abs",
    "continuous_second_loss",
    "resumed_second_loss",
    "second_step_delta",
    "second_step_router_max_abs",
    "second_step_expert_up_max_abs",
    "second_step_expert_down_max_abs",
    "second_step_optimizer_max_abs",
    "expert_load",
    "total_params",
    "activated_params",
    "router_grad_defined",
    "expert_up_grad_defined",
    "expert_down_grad_defined",
    "router_grad_norm",
    "expert_up_grad_norm",
    "expert_down_grad_norm",
    "router_delta_norm",
    "expert_up_delta_norm",
    "expert_down_delta_norm",
]
missing = [key for key in required if key not in summary]
if missing:
    raise SystemExit(f"summary is missing fields: {missing}")

if not str(summary["device"]).startswith("Cuda"):
    raise SystemExit(f"expected CUDA device, got {summary['device']}")
if summary["train_steps"] != 8:
    raise SystemExit(f"expected train_steps=8, got {summary['train_steps']}")
if summary["tokens"] != 6:
    raise SystemExit(f"expected 6 routed tokens, got {summary['tokens']}")
if summary["hidden_size"] != 4:
    raise SystemExit(f"expected hidden_size=4, got {summary['hidden_size']}")
if summary["expert_hidden_size"] != 6:
    raise SystemExit(
        f"expected expert_hidden_size=6, got {summary['expert_hidden_size']}"
    )
if summary["num_experts"] != 3:
    raise SystemExit(f"expected num_experts=3, got {summary['num_experts']}")
if summary["top_k"] != 1:
    raise SystemExit(f"expected top_k=1, got {summary['top_k']}")
if summary["total_params"] != 156:
    raise SystemExit(f"unexpected total_params: {summary['total_params']}")
if summary["activated_params"] != 60:
    raise SystemExit(f"unexpected activated_params: {summary['activated_params']}")
if not summary["activated_params"] < summary["total_params"]:
    raise SystemExit("activated_params must be less than total_params")
checkpoint = pathlib.Path(summary["checkpoint_output"])
if not checkpoint.exists() or checkpoint.stat().st_size == 0:
    raise SystemExit(f"checkpoint_output missing or empty: {checkpoint}")
optimizer = pathlib.Path(summary["optimizer_output"])
if not optimizer.exists() or optimizer.stat().st_size == 0:
    raise SystemExit(f"optimizer_output missing or empty: {optimizer}")
manifest_path = pathlib.Path(summary["manifest_output"])
if not manifest_path.exists() or manifest_path.stat().st_size == 0:
    raise SystemExit(f"manifest_output missing or empty: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
if manifest.get("format") != "rustrain.tch_moe.v1":
    raise SystemExit(f"unexpected manifest format: {manifest.get('format')}")
if manifest.get("global_step") != summary["train_steps"]:
    raise SystemExit(
        f"manifest global_step should equal train_steps: {manifest.get('global_step')} vs {summary['train_steps']}"
    )
if manifest.get("model_safetensors") != str(checkpoint):
    raise SystemExit("manifest model_safetensors does not match summary checkpoint_output")
if manifest.get("optimizer_safetensors") != str(optimizer):
    raise SystemExit("manifest optimizer_safetensors does not match summary optimizer_output")
model_tensors = {entry["name"]: entry for entry in manifest.get("model_tensors", [])}
expected_model_tensors = {
    "router.weight": [summary["hidden_size"], summary["num_experts"]],
    "experts.up.weight": [
        summary["num_experts"],
        summary["hidden_size"],
        summary["expert_hidden_size"],
    ],
    "experts.down.weight": [
        summary["num_experts"],
        summary["expert_hidden_size"],
        summary["hidden_size"],
    ],
}
if set(model_tensors) != set(expected_model_tensors):
    raise SystemExit(f"unexpected model tensor names: {sorted(model_tensors)}")
for name, shape in expected_model_tensors.items():
    entry = model_tensors[name]
    if entry.get("shape") != shape:
        raise SystemExit(f"{name} shape mismatch: {entry.get('shape')} vs {shape}")
    if entry.get("dtype") != "Float":
        raise SystemExit(f"{name} dtype mismatch: {entry.get('dtype')}")
optimizer_slots = {entry["name"]: entry for entry in manifest.get("optimizer_slots", [])}
expected_slots = {
    "router.weight.adam_m": [summary["hidden_size"], summary["num_experts"]],
    "router.weight.adam_v": [summary["hidden_size"], summary["num_experts"]],
    "experts.up.weight.adam_m": [
        summary["num_experts"],
        summary["hidden_size"],
        summary["expert_hidden_size"],
    ],
    "experts.up.weight.adam_v": [
        summary["num_experts"],
        summary["hidden_size"],
        summary["expert_hidden_size"],
    ],
    "experts.down.weight.adam_m": [
        summary["num_experts"],
        summary["expert_hidden_size"],
        summary["hidden_size"],
    ],
    "experts.down.weight.adam_v": [
        summary["num_experts"],
        summary["expert_hidden_size"],
        summary["hidden_size"],
    ],
}
if set(optimizer_slots) != set(expected_slots):
    raise SystemExit(f"unexpected optimizer slot names: {sorted(optimizer_slots)}")
for name, shape in expected_slots.items():
    entry = optimizer_slots[name]
    if entry.get("shape") != shape:
        raise SystemExit(f"{name} shape mismatch: {entry.get('shape')} vs {shape}")
    if entry.get("dtype") != "Float":
        raise SystemExit(f"{name} dtype mismatch: {entry.get('dtype')}")
step_tensor = manifest.get("optimizer_step_tensor", {})
if step_tensor.get("name") != "optimizer.step":
    raise SystemExit(f"unexpected optimizer step tensor name: {step_tensor.get('name')}")
if step_tensor.get("shape") != [1]:
    raise SystemExit(f"unexpected optimizer step shape: {step_tensor.get('shape')}")
if step_tensor.get("dtype") != "Int64":
    raise SystemExit(f"unexpected optimizer step dtype: {step_tensor.get('dtype')}")
if len(summary["expert_load"]) != summary["num_experts"]:
    raise SystemExit(f"unexpected expert_load length: {summary['expert_load']}")
if sum(summary["expert_load"]) != summary["tokens"]:
    raise SystemExit(
        f"expert_load should cover every token: {summary['expert_load']}"
    )
if not all(summary[key] is True for key in [
    "router_grad_defined",
    "expert_up_grad_defined",
    "expert_down_grad_defined",
]):
    raise SystemExit("all router/expert gradients must be defined")
for key in [
    "router_grad_norm",
    "expert_up_grad_norm",
    "expert_down_grad_norm",
    "router_delta_norm",
    "expert_up_delta_norm",
    "expert_down_delta_norm",
]:
    if float(summary[key]) <= 0.0:
        raise SystemExit(f"{key} must be positive, got {summary[key]}")
if not float(summary["final_loss"]) < float(summary["initial_loss"]):
    raise SystemExit(
        f"loss did not improve: {summary['initial_loss']} -> {summary['final_loss']}"
    )
if not float(summary["final_task_loss"]) < float(summary["initial_task_loss"]):
    raise SystemExit(
        f"task loss did not improve: {summary['initial_task_loss']} -> {summary['final_task_loss']}"
    )
if float(summary["initial_load_balance_loss"]) < 0.0:
    raise SystemExit("initial load-balance loss must be non-negative")
if float(summary["final_load_balance_loss"]) < 0.0:
    raise SystemExit("final load-balance loss must be non-negative")
if abs(float(summary["reloaded_loss"]) - float(summary["final_loss"])) > 1e-7:
    raise SystemExit(
        f"reloaded_loss does not match final_loss: {summary['reloaded_loss']} vs {summary['final_loss']}"
    )
for key in [
    "reload_delta",
    "reload_optimizer_max_abs",
    "second_step_delta",
    "second_step_router_max_abs",
    "second_step_expert_up_max_abs",
    "second_step_expert_down_max_abs",
    "second_step_optimizer_max_abs",
]:
    if float(summary[key]) > 1e-7:
        raise SystemExit(f"{key} too large: {summary[key]}")

print(
    "tch_moe_smoke_verified: "
    f"loss={summary['initial_loss']}->{summary['final_loss']} "
    f"task_loss={summary['initial_task_loss']}->{summary['final_task_loss']} "
    f"expert_load={summary['expert_load']} "
    f"params={summary['activated_params']}/{summary['total_params']} "
    f"reload_delta={summary['reload_delta']} "
    f"second_step_delta={summary['second_step_delta']} "
    f"manifest={summary['manifest_output']}"
)
PY
