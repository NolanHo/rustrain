#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT_DIR="${RUSTRAIN_EP_TCH_MOE_OUTPUT_DIR:-/tmp/rustrain-runs/ep-tch-moe-verify-$$}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  moe parallel-ep-tch-moe-rank-smoke \
  --output-dir "${OUTPUT_DIR}/ranks"

python - "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
rank_dir = output_dir / "ranks"
summaries = sorted(rank_dir.glob("ep-tch-moe-rank-*.json"))
if len(summaries) != 2:
    raise SystemExit(f"expected 2 EP tch MoE rank summaries under {rank_dir}, found {len(summaries)}")

expected = {
    0: {
        "source_token_indices": [0, 2],
        "owned_experts": [0, 2],
        "owned_token_indices": [0, 1, 2],
        "dispatch_send_counts": [2, 0],
        "dispatch_recv_counts": [2, 1],
        "combine_send_counts": [2, 1],
        "combine_recv_counts": [2, 0],
    },
    1: {
        "source_token_indices": [1, 3],
        "owned_experts": [2, 4],
        "owned_token_indices": [3],
        "dispatch_send_counts": [1, 1],
        "dispatch_recv_counts": [0, 1],
        "combine_send_counts": [0, 1],
        "combine_recv_counts": [1, 1],
    },
}

source_tokens = []
owned_tokens = []
evidence = []
for path in summaries:
    data = json.loads(path.read_text())
    rank = int(data["rank"])
    if rank not in expected:
        raise SystemExit(f"{path} unexpected rank {rank}")
    if data["world_size"] != 2:
        raise SystemExit(f"{path} world_size {data['world_size']} != 2")
    if data["local_rank"] != rank:
        raise SystemExit(f"{path} local_rank {data['local_rank']} != rank {rank}")
    want = expected[rank]
    owned_range = [int(data["owned_expert_start"]), int(data["owned_expert_end"])]
    if owned_range != want["owned_experts"]:
        raise SystemExit(f"{path} owned expert range {owned_range} != {want['owned_experts']}")
    for key in [
        "source_token_indices",
        "owned_token_indices",
        "dispatch_send_counts",
        "dispatch_recv_counts",
        "combine_send_counts",
        "combine_recv_counts",
    ]:
        if data[key] != want[key]:
            raise SystemExit(f"{path} {key} {data[key]} != {want[key]}")
    if data["assembled_output_shape"] != [2, 3]:
        raise SystemExit(f"{path} assembled_output_shape {data['assembled_output_shape']} != [2, 3]")
    if data["reference_output_shape"] != [2, 3]:
        raise SystemExit(f"{path} reference_output_shape {data['reference_output_shape']} != [2, 3]")
    if float(data["sparse_output_max_abs"]) > 1e-6:
        raise SystemExit(f"{path} sparse_output_max_abs too large: {data['sparse_output_max_abs']}")
    if data["global_expert_load"] != [2, 1, 0, 1]:
        raise SystemExit(f"{path} global_expert_load {data['global_expert_load']} != [2, 1, 0, 1]")
    if abs(float(data["load_balance_loss"]) - 0.125) > 1e-9:
        raise SystemExit(f"{path} load_balance_loss {data['load_balance_loss']} != 0.125")
    for key in ["expert_up_grad_norm", "expert_down_grad_norm"]:
        if float(data[key]) <= 0.0:
            raise SystemExit(f"{path} {key} must be positive: {data[key]}")
    if not data["train_loss_improved"]:
        raise SystemExit(f"{path} did not report train_loss_improved")
    if float(data["train_final_loss"]) >= float(data["train_initial_loss"]):
        raise SystemExit(
            f"{path} train_final_loss {data['train_final_loss']} >= train_initial_loss {data['train_initial_loss']}"
        )

    manifest_path = pathlib.Path(data["checkpoint_manifest_output"])
    model_path = pathlib.Path(data["checkpoint_model_safetensors"])
    optimizer_path = pathlib.Path(data["checkpoint_optimizer_safetensors"])
    for artifact in (manifest_path, model_path, optimizer_path):
        if not artifact.exists():
            raise SystemExit(f"{path} expected checkpoint artifact {artifact} to exist")
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != "rustrain.ep_sharded.v1":
        raise SystemExit(f"{manifest_path} unexpected format {manifest['format']}")
    if manifest["rank"] != rank or manifest["world_size"] != 2:
        raise SystemExit(f"{manifest_path} rank/world_size mismatch")
    if manifest["local_rank"] != rank:
        raise SystemExit(f"{manifest_path} local_rank {manifest['local_rank']} != rank {rank}")
    if [manifest["owned_expert_start"], manifest["owned_expert_end"]] != owned_range:
        raise SystemExit(f"{manifest_path} owned expert range mismatch")
    if manifest["model_safetensors"] != str(model_path):
        raise SystemExit(f"{manifest_path} model_safetensors mismatch")
    if manifest["optimizer_safetensors"] != str(optimizer_path):
        raise SystemExit(f"{manifest_path} optimizer_safetensors mismatch")
    if manifest["optimizer"] != "adamw":
        raise SystemExit(f"{manifest_path} optimizer {manifest['optimizer']} != adamw")
    if int(data["checkpoint_tensor_count"]) != 2:
        raise SystemExit(f"{path} checkpoint_tensor_count {data['checkpoint_tensor_count']} != 2")
    if len(manifest["shards"]) != 2:
        raise SystemExit(f"{manifest_path} expected two owned expert shards")
    shards = {shard["shard_name"]: shard for shard in manifest["shards"]}
    expected_shapes = {
        "experts.up.weight": [2, 3, 5],
        "experts.down.weight": [2, 5, 3],
    }
    expected_global_shapes = {
        "experts.up.weight": [4, 3, 5],
        "experts.down.weight": [4, 5, 3],
    }
    expected_slots = {
        "experts.up.weight": ("experts.up.weight.adam_m", "experts.up.weight.adam_v"),
        "experts.down.weight": ("experts.down.weight.adam_m", "experts.down.weight.adam_v"),
    }
    if set(shards) != set(expected_shapes):
        raise SystemExit(f"{manifest_path} unexpected shard names {sorted(shards)}")
    for name, shape in expected_shapes.items():
        shard = shards[name]
        if shard["partition"] != "expert_model_parallel":
            raise SystemExit(f"{manifest_path} unexpected partition {shard['partition']}")
        if shard["dtype"] != "float32":
            raise SystemExit(f"{manifest_path} unexpected dtype {shard['dtype']}")
        if shard["shard_shape"] != shape or shard["global_shape"] != expected_global_shapes[name]:
            raise SystemExit(f"{manifest_path} unexpected shape for {name}: {shard}")
        if (shard["optimizer_m_name"], shard["optimizer_v_name"]) != expected_slots[name]:
            raise SystemExit(f"{manifest_path} unexpected optimizer slots for {name}: {shard}")
    for key in [
        "reload_expert_up_max_abs",
        "reload_expert_down_max_abs",
        "reload_optimizer_max_abs",
        "reload_loss_delta",
        "second_step_delta",
        "second_step_expert_up_max_abs",
        "second_step_expert_down_max_abs",
        "second_step_optimizer_max_abs",
    ]:
        if float(data[key]) > 1e-7:
            raise SystemExit(f"{path} {key} too large: {data[key]}")
    if int(data["second_step_optimizer_step_delta"]) != 0:
        raise SystemExit(f"{path} second_step_optimizer_step_delta must be 0")
    source_tokens.extend(data["source_token_indices"])
    owned_tokens.extend(data["owned_token_indices"])
    evidence.append(
        {
            "rank": rank,
            "owned_experts": owned_range,
            "owned_token_indices": data["owned_token_indices"],
            "expert_up_grad_norm": data["expert_up_grad_norm"],
            "expert_down_grad_norm": data["expert_down_grad_norm"],
            "train_initial_loss": data["train_initial_loss"],
            "train_final_loss": data["train_final_loss"],
            "checkpoint_manifest_output": data["checkpoint_manifest_output"],
            "second_step_delta": data["second_step_delta"],
        }
    )

if sorted(source_tokens) != [0, 1, 2, 3]:
    raise SystemExit(f"source token coverage is wrong: {source_tokens}")
if sorted(owned_tokens) != [0, 1, 2, 3]:
    raise SystemExit(f"owned token coverage is wrong: {owned_tokens}")

launch_summary = json.loads((output_dir / "launch-summary.json").read_text())
assigned = [rank["assigned_cuda_visible_device"] for rank in launch_summary["ranks"]]
if len(set(assigned)) != 2:
    raise SystemExit(f"expected distinct launch GPU assignments, got {assigned}")

print(
    json.dumps(
        {
            "ep_tch_moe_verified": evidence,
            "assigned_cuda_visible_devices": assigned,
        },
        indent=2,
    )
)
PY
