#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT_DIR="${RUSTRAIN_LAUNCH_GPU_ASSIGNMENT_OUTPUT_DIR:-/tmp/rustrain-runs/launch-gpu-assignment-verify-$$}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  print-launch-env

python - "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
summary_path = output_dir / "launch-summary.json"
summary = json.loads(summary_path.read_text())
if summary["nproc_per_node"] != 2:
    raise SystemExit(f"expected nproc_per_node=2, got {summary['nproc_per_node']}")
if len(summary["ranks"]) != 2:
    raise SystemExit(f"expected 2 rank summaries, got {len(summary['ranks'])}")

assigned = {}
for rank in summary["ranks"]:
    rank_id = int(rank["rank"])
    if rank["assigned_cuda_device_ordinal"] != rank_id:
        raise SystemExit(
            f"rank {rank_id} assigned ordinal {rank['assigned_cuda_device_ordinal']} != {rank_id}"
        )
    device = rank["assigned_cuda_visible_device"]
    if not device:
        raise SystemExit(f"rank {rank_id} has no assigned CUDA visible device")
    assigned[rank_id] = device

if assigned[0] == assigned[1]:
    raise SystemExit(f"rank0 and rank1 were assigned the same device: {assigned[0]}")

rank_logs = sorted(output_dir.glob("rank-*.log"))
if len(rank_logs) != 2:
    raise SystemExit(f"expected 2 rank logs, got {len(rank_logs)}")

for path in rank_logs:
    data = json.loads(path.read_text())
    rank = int(data["rank"])
    if data["local_rank"] != rank:
        raise SystemExit(f"{path} local_rank {data['local_rank']} != rank {rank}")
    if data["world_size"] != 2 or data["local_world_size"] != 2:
        raise SystemExit(f"{path} has wrong world size fields: {data}")
    if data["assigned_cuda_device_ordinal"] != rank:
        raise SystemExit(
            f"{path} assigned ordinal {data['assigned_cuda_device_ordinal']} != rank {rank}"
        )
    if data["assigned_cuda_visible_device"] != assigned[rank]:
        raise SystemExit(
            f"{path} assigned device {data['assigned_cuda_visible_device']} "
            f"does not match launch summary {assigned[rank]}"
        )

print(
    "launch_gpu_assignment_verified: "
    f"rank0={assigned[0]} rank1={assigned[1]} output_dir={output_dir}"
)
PY
