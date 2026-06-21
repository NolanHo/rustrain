#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT_DIR="${RUSTRAIN_EP_NCCL_OUTPUT_DIR:-/tmp/rustrain-runs/ep-nccl-verify-$$}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  parallel-ep-nccl-rank-smoke \
  --output-dir "${OUTPUT_DIR}/ranks"

python - "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
rank_dir = output_dir / "ranks"
summaries = sorted(rank_dir.glob("ep-nccl-rank-*.json"))
if len(summaries) != 2:
    raise SystemExit(f"expected 2 EP NCCL rank summaries under {rank_dir}, found {len(summaries)}")

evidence = []
owned_ranges = []
covered_tokens = []
expert_load = [0, 0, 0, 0]
for path in summaries:
    data = json.loads(path.read_text())
    rank = int(data["rank"])
    if data["world_size"] != 2:
        raise SystemExit(f"{path} world_size {data['world_size']} != 2")
    if data["local_rank"] != rank:
        raise SystemExit(f"{path} local_rank {data['local_rank']} != rank {rank}")
    owned_range = [int(data["owned_expert_start"]), int(data["owned_expert_end"])]
    owned_ranges.append(tuple(owned_range))
    if data["reduced_output_shape"] != [4, 3]:
        raise SystemExit(f"{path} reduced_output_shape {data['reduced_output_shape']} != [4, 3]")
    if float(data["combine_max_abs"]) > 1e-6:
        raise SystemExit(f"{path} combine_max_abs too large: {data['combine_max_abs']}")
    if float(data["scale_grad_norm"]) <= 0.0:
        raise SystemExit(f"{path} scale_grad_norm must be positive: {data['scale_grad_norm']}")
    if not data["train_loss_improved"]:
        raise SystemExit(f"{path} did not report train_loss_improved")
    if float(data["train_final_loss"]) >= float(data["train_initial_loss"]):
        raise SystemExit(
            f"{path} final loss {data['train_final_loss']} >= initial loss {data['train_initial_loss']}"
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
    if [manifest["owned_expert_start"], manifest["owned_expert_end"]] != owned_range:
        raise SystemExit(f"{manifest_path} owned expert range mismatch")
    if manifest["model_safetensors"] != str(model_path):
        raise SystemExit(f"{manifest_path} model_safetensors mismatch")
    if manifest["optimizer_safetensors"] != str(optimizer_path):
        raise SystemExit(f"{manifest_path} optimizer_safetensors mismatch")
    if manifest["optimizer"] != "adamw":
        raise SystemExit(f"{manifest_path} optimizer {manifest['optimizer']} != adamw")
    if data["checkpoint_tensor_count"] != 1 or len(manifest["shards"]) != 1:
        raise SystemExit(f"{manifest_path} expected exactly one owned expert shard")
    shard = manifest["shards"][0]
    if shard["shard_name"] != "experts.scale":
        raise SystemExit(f"{manifest_path} unexpected shard_name {shard['shard_name']}")
    if shard["optimizer_m_name"] != "experts.scale.adam_m":
        raise SystemExit(f"{manifest_path} unexpected optimizer_m_name {shard['optimizer_m_name']}")
    if shard["optimizer_v_name"] != "experts.scale.adam_v":
        raise SystemExit(f"{manifest_path} unexpected optimizer_v_name {shard['optimizer_v_name']}")
    if shard["partition"] != "expert_model_parallel":
        raise SystemExit(f"{manifest_path} unexpected partition {shard['partition']}")
    if shard["shard_shape"] != [2, 3] or shard["global_shape"] != [4, 3]:
        raise SystemExit(f"{manifest_path} unexpected shard/global shape {shard}")
    if float(data["reload_scale_max_abs"]) > 1e-7:
        raise SystemExit(f"{path} reload_scale_max_abs too large: {data['reload_scale_max_abs']}")
    if float(data["reload_optimizer_max_abs"]) > 1e-7:
        raise SystemExit(f"{path} reload_optimizer_max_abs too large: {data['reload_optimizer_max_abs']}")
    if float(data["second_step_delta"]) > 1e-6:
        raise SystemExit(f"{path} second_step_delta too large: {data['second_step_delta']}")
    if float(data["second_step_scale_max_abs"]) > 1e-6:
        raise SystemExit(
            f"{path} second_step_scale_max_abs too large: {data['second_step_scale_max_abs']}"
        )
    if float(data["second_step_optimizer_max_abs"]) > 1e-6:
        raise SystemExit(
            f"{path} second_step_optimizer_max_abs too large: {data['second_step_optimizer_max_abs']}"
        )
    if float(data["continuous_second_loss"]) != float(data["resumed_second_loss"]):
        raise SystemExit(
            f"{path} continuous/resumed second loss mismatch: "
            f"{data['continuous_second_loss']} != {data['resumed_second_loss']}"
        )
    covered_tokens.extend(int(index) for index in data["owned_token_indices"])
    for index, load in enumerate(data["expert_load"]):
        expert_load[index] += int(load)
    evidence.append(
        {
            "rank": rank,
            "owned_experts": owned_range,
            "owned_token_indices": data["owned_token_indices"],
            "combine_max_abs": data["combine_max_abs"],
            "scale_grad_norm": data["scale_grad_norm"],
            "train_initial_loss": data["train_initial_loss"],
            "train_final_loss": data["train_final_loss"],
            "checkpoint_manifest_output": data["checkpoint_manifest_output"],
            "reload_scale_max_abs": data["reload_scale_max_abs"],
            "reload_optimizer_max_abs": data["reload_optimizer_max_abs"],
            "second_step_delta": data["second_step_delta"],
        }
    )

if sorted(owned_ranges) != [(0, 2), (2, 4)]:
    raise SystemExit(f"unexpected expert ownership ranges: {owned_ranges}")
if sorted(covered_tokens) != [0, 1, 2, 3]:
    raise SystemExit(f"rank-local token coverage is wrong: {covered_tokens}")
if expert_load != [2, 1, 0, 1]:
    raise SystemExit(f"combined expert load {expert_load} != expected [2, 1, 0, 1]")

launch_summary = json.loads((output_dir / "launch-summary.json").read_text())
assigned = [rank["assigned_cuda_visible_device"] for rank in launch_summary["ranks"]]
if len(set(assigned)) != 2:
    raise SystemExit(f"expected distinct launch GPU assignments, got {assigned}")

print(
    json.dumps(
        {
            "ep_nccl_verified": evidence,
            "assigned_cuda_visible_devices": assigned,
        },
        indent=2,
    )
)
PY
