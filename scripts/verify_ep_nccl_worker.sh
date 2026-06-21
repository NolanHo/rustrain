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
