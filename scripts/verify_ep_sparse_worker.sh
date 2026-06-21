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
            "sparse_output_max_abs": data["sparse_output_max_abs"],
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
