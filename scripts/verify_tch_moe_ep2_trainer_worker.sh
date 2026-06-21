#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT_DIR="${RUSTRAIN_TCH_MOE_EP2_OUTPUT_DIR:-/tmp/rustrain-runs/tch-moe-ep2-trainer-verify-$$}"
CONFIG="${RUSTRAIN_TCH_MOE_EP2_CONFIG:-configs/tch_moe_ep2.toml}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  train --config "${CONFIG}"

python - "${OUTPUT_DIR}" "${CONFIG}" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
config = sys.argv[2]
launch_summary = json.loads((output_dir / "launch-summary.json").read_text())
if launch_summary["command"] != ["train", "--config", config]:
    raise SystemExit(f"unexpected launch command {launch_summary['command']}")
assigned = [rank["assigned_cuda_visible_device"] for rank in launch_summary["ranks"]]
if len(set(assigned)) != 2:
    raise SystemExit(f"expected distinct launch GPU assignments, got {assigned}")

rank_summaries = sorted(output_dir.rglob("ep-tch-moe-rank-*.json"))
if len(rank_summaries) != 2:
    raise SystemExit(f"expected 2 EP tch MoE trainer rank summaries under {output_dir}, found {len(rank_summaries)}")

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
for path in rank_summaries:
    data = json.loads(path.read_text())
    rank = int(data["rank"])
    want = expected.get(rank)
    if want is None:
        raise SystemExit(f"{path} unexpected rank {rank}")
    if data["world_size"] != 2:
        raise SystemExit(f"{path} world_size {data['world_size']} != 2")
    if data["local_rank"] != rank:
        raise SystemExit(f"{path} local_rank {data['local_rank']} != rank {rank}")
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
    if data["global_expert_load"] != [2, 1, 0, 1]:
        raise SystemExit(f"{path} global_expert_load {data['global_expert_load']} != [2, 1, 0, 1]")
    if float(data["sparse_output_max_abs"]) > 1e-6:
        raise SystemExit(f"{path} sparse_output_max_abs too large: {data['sparse_output_max_abs']}")
    if not data["train_loss_improved"]:
        raise SystemExit(f"{path} expected train_loss_improved")
    if float(data["train_final_loss"]) >= float(data["train_initial_loss"]):
        raise SystemExit(f"{path} train loss did not improve")
    for key in ["expert_up_grad_norm", "expert_down_grad_norm"]:
        if float(data[key]) <= 0.0:
            raise SystemExit(f"{path} {key} must be positive: {data[key]}")
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
    manifest_path = pathlib.Path(data["checkpoint_manifest_output"])
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != "rustrain.ep_sharded.v1":
        raise SystemExit(f"{manifest_path} unexpected format {manifest['format']}")
    if manifest["rank"] != rank or manifest["world_size"] != 2:
        raise SystemExit(f"{manifest_path} rank/world_size mismatch")
    if [manifest["owned_expert_start"], manifest["owned_expert_end"]] != owned_range:
        raise SystemExit(f"{manifest_path} owned expert range mismatch")
    if len(manifest["shards"]) != 2:
        raise SystemExit(f"{manifest_path} expected two expert MLP shards")
    shard_names = sorted(shard["shard_name"] for shard in manifest["shards"])
    if shard_names != ["experts.down.weight", "experts.up.weight"]:
        raise SystemExit(f"{manifest_path} unexpected shard names {shard_names}")
    source_tokens.extend(data["source_token_indices"])
    owned_tokens.extend(data["owned_token_indices"])
    evidence.append(
        {
            "rank": rank,
            "owned_experts": owned_range,
            "expert_up_grad_norm": data["expert_up_grad_norm"],
            "expert_down_grad_norm": data["expert_down_grad_norm"],
            "train_initial_loss": data["train_initial_loss"],
            "train_final_loss": data["train_final_loss"],
            "checkpoint_manifest_output": data["checkpoint_manifest_output"],
        }
    )

if sorted(source_tokens) != [0, 1, 2, 3]:
    raise SystemExit(f"source token coverage is wrong: {source_tokens}")
if sorted(owned_tokens) != [0, 1, 2, 3]:
    raise SystemExit(f"owned token coverage is wrong: {owned_tokens}")

for rank in [0, 1]:
    log_text = pathlib.Path(launch_summary["ranks"][rank]["log_path"]).read_text()
    if "rustrain tch MoE EP trainer-entry complete" not in log_text:
        raise SystemExit(f"rank {rank} log did not include trainer-entry completion marker")

print(
    json.dumps(
        {
            "tch_moe_ep2_trainer_verified": evidence,
            "assigned_cuda_visible_devices": assigned,
        },
        indent=2,
    )
)
PY
