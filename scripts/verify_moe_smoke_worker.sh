#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT="$(mktemp)"

cargo run -- moe-smoke | tee "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
import json
import math
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
start = text.find("{")
end = text.rfind("}")
if start < 0 or end < start:
    raise SystemExit("moe-smoke did not print a JSON summary")
summary = json.loads(text[start : end + 1])

required_top = [
    "tiny_hidden_shape",
    "tiny_hidden_sum",
    "tiny",
    "deepseek_hidden_shape",
    "deepseek_hidden_sum",
    "deepseek",
]
missing_top = [key for key in required_top if key not in summary]
if missing_top:
    raise SystemExit(f"summary is missing fields: {missing_top}")

tiny = summary["tiny"]
required_tiny = [
    "expert_load",
    "load_balance_loss",
    "total_params",
    "activated_params",
]
missing_tiny = [key for key in required_tiny if key not in tiny]
if missing_tiny:
    raise SystemExit(f"tiny stats missing fields: {missing_tiny}")
if summary["tiny_hidden_shape"] != [1, 2]:
    raise SystemExit(f"unexpected tiny hidden shape: {summary['tiny_hidden_shape']}")
if not math.isclose(float(summary["tiny_hidden_sum"]), 2.0, abs_tol=1e-6):
    raise SystemExit(f"unexpected tiny hidden sum: {summary['tiny_hidden_sum']}")
if tiny["expert_load"] != [1, 0]:
    raise SystemExit(f"unexpected tiny expert load: {tiny['expert_load']}")
if tiny["total_params"] != 20:
    raise SystemExit(f"unexpected tiny total_params: {tiny['total_params']}")
if tiny["activated_params"] != 12:
    raise SystemExit(f"unexpected tiny activated_params: {tiny['activated_params']}")
if not tiny["activated_params"] < tiny["total_params"]:
    raise SystemExit("tiny activated params should be sparse relative to total params")
if not math.isclose(float(tiny["load_balance_loss"]), 0.5, abs_tol=1e-6):
    raise SystemExit(f"unexpected tiny load_balance_loss: {tiny['load_balance_loss']}")

deepseek = summary["deepseek"]
required_deepseek = [
    "layers",
    "shared_params",
    "routed_params",
    "total_params",
    "activated_params",
]
missing_deepseek = [key for key in required_deepseek if key not in deepseek]
if missing_deepseek:
    raise SystemExit(f"deepseek stats missing fields: {missing_deepseek}")
if summary["deepseek_hidden_shape"] != [2, 2]:
    raise SystemExit(f"unexpected deepseek hidden shape: {summary['deepseek_hidden_shape']}")
if not math.isclose(float(summary["deepseek_hidden_sum"]), 16.0, abs_tol=1e-6):
    raise SystemExit(f"unexpected deepseek hidden sum: {summary['deepseek_hidden_sum']}")
if deepseek["shared_params"] != 16:
    raise SystemExit(f"unexpected deepseek shared_params: {deepseek['shared_params']}")
if deepseek["routed_params"] != 40:
    raise SystemExit(f"unexpected deepseek routed_params: {deepseek['routed_params']}")
if deepseek["total_params"] != 56:
    raise SystemExit(f"unexpected deepseek total_params: {deepseek['total_params']}")
if deepseek["activated_params"] != 40:
    raise SystemExit(f"unexpected deepseek activated_params: {deepseek['activated_params']}")
if not deepseek["activated_params"] < deepseek["total_params"]:
    raise SystemExit("deepseek activated params should be sparse relative to total params")

layers = deepseek["layers"]
if len(layers) != 2:
    raise SystemExit(f"expected 2 deepseek layers, got {len(layers)}")
for expected_index, layer in enumerate(layers):
    required_layer = ["layer_index", "routed_expert_load", "load_balance_loss"]
    missing_layer = [key for key in required_layer if key not in layer]
    if missing_layer:
        raise SystemExit(f"deepseek layer {expected_index} missing fields: {missing_layer}")
    if layer["layer_index"] != expected_index:
        raise SystemExit(f"unexpected layer_index: {layer['layer_index']}")
    if layer["routed_expert_load"] != [2, 0]:
        raise SystemExit(
            f"unexpected routed load for layer {expected_index}: {layer['routed_expert_load']}"
        )
    if not math.isclose(float(layer["load_balance_loss"]), 0.5, abs_tol=1e-6):
        raise SystemExit(
            f"unexpected load balance loss for layer {expected_index}: {layer['load_balance_loss']}"
        )

print(
    "moe_smoke_verified: "
    f"tiny_load={tiny['expert_load']} "
    f"deepseek_layers={len(layers)} "
    f"deepseek_params={deepseek['activated_params']}/{deepseek['total_params']}"
)
PY
