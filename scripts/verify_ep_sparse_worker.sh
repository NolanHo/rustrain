#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT_DIR="${RUSTRAIN_EP_SPARSE_OUTPUT_DIR:-/tmp/rustrain-runs/ep-sparse-verify-$$}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  parallel-ep-sparse-rank-smoke \
  --output-dir "${OUTPUT_DIR}/ranks"

python - "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
rank_dir = output_dir / "ranks"
summaries = sorted(rank_dir.glob("ep-sparse-rank-*.json"))
if len(summaries) != 2:
    raise SystemExit(f"expected 2 EP sparse rank summaries under {rank_dir}, found {len(summaries)}")

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

evidence = []
source_tokens = []
owned_tokens = []
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
    if data["source_token_indices"] != want["source_token_indices"]:
        raise SystemExit(f"{path} source_token_indices {data['source_token_indices']} != {want['source_token_indices']}")
    owned_range = [int(data["owned_expert_start"]), int(data["owned_expert_end"])]
    if owned_range != want["owned_experts"]:
        raise SystemExit(f"{path} owned expert range {owned_range} != {want['owned_experts']}")
    for key in [
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
    if float(data["scale_grad_norm"]) <= 0.0:
        raise SystemExit(f"{path} scale_grad_norm must be positive: {data['scale_grad_norm']}")
    if not data["train_loss_improved"]:
        raise SystemExit(
            f"{path} expected train_loss_improved, initial={data['train_initial_loss']} final={data['train_final_loss']}"
        )
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
    if int(data["checkpoint_tensor_count"]) != 1:
        raise SystemExit(f"{path} checkpoint_tensor_count {data['checkpoint_tensor_count']} != 1")
    if len(manifest["shards"]) != 1:
        raise SystemExit(f"{manifest_path} expected exactly one owned expert shard")
    shard = manifest["shards"][0]
    if shard["name"] != f"experts.{owned_range[0]}..{owned_range[1]}.scale":
        raise SystemExit(f"{manifest_path} unexpected shard logical name {shard['name']}")
    if shard["shard_name"] != "experts.scale":
        raise SystemExit(f"{manifest_path} unexpected shard_name {shard['shard_name']}")
    if shard["optimizer_m_name"] != "experts.scale.adam_m":
        raise SystemExit(f"{manifest_path} unexpected optimizer_m_name {shard['optimizer_m_name']}")
    if shard["optimizer_v_name"] != "experts.scale.adam_v":
        raise SystemExit(f"{manifest_path} unexpected optimizer_v_name {shard['optimizer_v_name']}")
    if shard["partition"] != "expert_model_parallel":
        raise SystemExit(f"{manifest_path} unexpected partition {shard['partition']}")
    if shard["dtype"] != "float32":
        raise SystemExit(f"{manifest_path} unexpected dtype {shard['dtype']}")
    if shard["shard_shape"] != [2, 3] or shard["global_shape"] != [4, 3]:
        raise SystemExit(f"{manifest_path} unexpected shard/global shape {shard}")
    if float(data["reload_scale_max_abs"]) > 1e-7:
        raise SystemExit(f"{path} reload_scale_max_abs too large: {data['reload_scale_max_abs']}")
    if float(data["reload_optimizer_max_abs"]) > 1e-7:
        raise SystemExit(f"{path} reload_optimizer_max_abs too large: {data['reload_optimizer_max_abs']}")
    if float(data["reload_loss_delta"]) > 1e-7:
        raise SystemExit(f"{path} reload_loss_delta too large: {data['reload_loss_delta']}")
    if float(data["second_step_delta"]) > 1e-7:
        raise SystemExit(f"{path} second_step_delta too large: {data['second_step_delta']}")
    if float(data["second_step_scale_max_abs"]) > 1e-7:
        raise SystemExit(f"{path} second_step_scale_max_abs too large: {data['second_step_scale_max_abs']}")
    if float(data["second_step_optimizer_max_abs"]) > 1e-7:
        raise SystemExit(f"{path} second_step_optimizer_max_abs too large: {data['second_step_optimizer_max_abs']}")
    if int(data["second_step_optimizer_step_delta"]) != 0:
        raise SystemExit(f"{path} second_step_optimizer_step_delta must be 0: {data['second_step_optimizer_step_delta']}")
    source_tokens.extend(data["source_token_indices"])
    owned_tokens.extend(data["owned_token_indices"])
    evidence.append(
        {
            "rank": rank,
            "source_token_indices": data["source_token_indices"],
            "owned_experts": owned_range,
            "owned_token_indices": data["owned_token_indices"],
            "dispatch_send_counts": data["dispatch_send_counts"],
            "dispatch_recv_counts": data["dispatch_recv_counts"],
            "combine_send_counts": data["combine_send_counts"],
            "combine_recv_counts": data["combine_recv_counts"],
            "global_expert_load": data["global_expert_load"],
            "load_balance_loss": data["load_balance_loss"],
            "sparse_output_max_abs": data["sparse_output_max_abs"],
            "scale_grad_norm": data["scale_grad_norm"],
            "train_initial_loss": data["train_initial_loss"],
            "train_final_loss": data["train_final_loss"],
            "train_loss_improved": data["train_loss_improved"],
            "checkpoint_manifest_output": data["checkpoint_manifest_output"],
            "reload_scale_max_abs": data["reload_scale_max_abs"],
            "reload_optimizer_max_abs": data["reload_optimizer_max_abs"],
            "reload_loss": data["reload_loss"],
            "reload_loss_delta": data["reload_loss_delta"],
            "continuous_second_loss": data["continuous_second_loss"],
            "resumed_second_loss": data["resumed_second_loss"],
            "second_step_delta": data["second_step_delta"],
            "second_step_scale_max_abs": data["second_step_scale_max_abs"],
            "second_step_optimizer_max_abs": data["second_step_optimizer_max_abs"],
            "second_step_optimizer_step_delta": data["second_step_optimizer_step_delta"],
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
            "ep_sparse_verified": evidence,
            "assigned_cuda_visible_devices": assigned,
        },
        indent=2,
    )
)
PY
