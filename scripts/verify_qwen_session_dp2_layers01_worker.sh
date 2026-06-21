#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT_DIR="${RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR:-/tmp/rustrain-runs/qwen-session-dp2-layers01-verify-$(date +%Y%m%d-%H%M%S)-$$}"
CONFIG="${RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG:-configs/qwen_session_dp2_layers01.toml}"
EXPECTED_TRAINABLE_TENSORS="${RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS:-25}"
EXPECTED_DTYPE="${RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND:-fp32}"
export RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}"
export RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS="${EXPECTED_TRAINABLE_TENSORS}"
export RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND="${EXPECTED_DTYPE}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  train --config "${CONFIG}"

python - <<'PY'
import json
import os
import pathlib

output_dir = pathlib.Path(os.environ["RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR"])
expected_trainable_tensors = int(os.environ["RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS"])
expected_dtype = os.environ.get("RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND")
rank_summaries = sorted(output_dir.rglob("qwen-session-dp-rank-*.json"))
if len(rank_summaries) != 2:
    raise SystemExit(
        f"expected 2 qwen session DP rank summaries under {output_dir}, found {len(rank_summaries)}"
    )

expected_layer_names = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.1.mlp.down_proj.weight",
    "model.norm.weight",
]
evidence = []
rank0_manifest_path = None
for path in rank_summaries:
    data = json.loads(path.read_text())
    if data.get("world_size") != 2:
        raise SystemExit(f"{path} expected world_size 2, got {data.get('world_size')}")
    if data.get("steps") != 2:
        raise SystemExit(f"{path} expected steps 2, got {data.get('steps')}")
    if expected_dtype and data.get("dtype") != expected_dtype:
        raise SystemExit(
            f"{path} dtype {data.get('dtype')} does not match expected {expected_dtype}"
        )

    trainable_tensors = data.get("trainable_tensors")
    if not isinstance(trainable_tensors, list):
        raise SystemExit(f"{path} trainable_tensors must be a list")
    if len(trainable_tensors) != expected_trainable_tensors:
        raise SystemExit(
            f"{path} expected {expected_trainable_tensors} trainable tensors, got {len(trainable_tensors)}"
        )
    if "model.embed_tokens.weight" in trainable_tensors:
        raise SystemExit(f"{path} DP representative path must not train tied embedding")
    for name in expected_layer_names:
        if name not in trainable_tensors:
            raise SystemExit(f"{path} missing expected trainable tensor {name}")

    global_step_losses = data.get("global_step_losses")
    if not isinstance(global_step_losses, list) or len(global_step_losses) != 3:
        raise SystemExit(f"{path} expected 3 global_step_losses, got {global_step_losses}")
    if not global_step_losses[-1] < global_step_losses[0]:
        raise SystemExit(f"{path} global loss did not improve: {global_step_losses}")
    if not data.get("global_loss_improved"):
        raise SystemExit(f"{path} global_loss_improved was not true")
    if float(data.get("max_grad_delta", 1.0)) > 5e-4:
        raise SystemExit(f"{path} max_grad_delta too large: {data.get('max_grad_delta')}")
    for key in ["reload_delta", "next_step_delta", "sharded_reload_delta", "sharded_next_step_delta"]:
        value = data.get(key)
        if value is None:
            raise SystemExit(f"{path} is missing {key}")
        if float(value) > 1e-5:
            raise SystemExit(f"{path} {key} {value} exceeds tolerance")

    manifest_path = pathlib.Path(data["manifest_output"])
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("trainable_tensors") != trainable_tensors:
        raise SystemExit(f"{path} rank0 manifest trainable_tensors do not match summary")
    if len(manifest.get("tensors", [])) != expected_trainable_tensors:
        raise SystemExit(
            f"{manifest_path} expected {expected_trainable_tensors} manifest tensors, got {len(manifest.get('tensors', []))}"
        )
    if int(manifest.get("tensor_count", -1)) != expected_trainable_tensors:
        raise SystemExit(
            f"{manifest_path} tensor_count {manifest.get('tensor_count')} != {expected_trainable_tensors}"
        )
    sharded_global = json.loads(pathlib.Path(data["sharded_global_manifest_output"]).read_text())
    parallel = sharded_global.get("parallel") or {}
    if int(parallel.get("data_parallel_size", -1)) != 2:
        raise SystemExit(
            f"{path} sharded data_parallel_size {parallel.get('data_parallel_size')} != 2"
        )
    for key in [
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "context_parallel_size",
    ]:
        if int(parallel.get(key, -1)) != 1:
            raise SystemExit(f"{path} sharded {key} {parallel.get(key)} != 1")
    if len(sharded_global.get("ranks", [])) != 2:
        raise SystemExit(f"{path} sharded global manifest must embed 2 rank manifests")
    if data["checkpoint_written"]:
        rank0_manifest_path = manifest_path
    evidence.append(
        {
            "rank": data["rank"],
            "dtype": data["dtype"],
            "trainable_tensors": len(trainable_tensors),
            "global_step_losses": global_step_losses,
            "reload_delta": data["reload_delta"],
            "next_step_delta": data["next_step_delta"],
            "sharded_reload_delta": data["sharded_reload_delta"],
            "sharded_next_step_delta": data["sharded_next_step_delta"],
        }
    )

if rank0_manifest_path is None:
    raise SystemExit("expected one rank0 checkpoint writer")

print(json.dumps({"qwen_session_dp2_layers01_verified": evidence}, indent=2, sort_keys=True))
PY
