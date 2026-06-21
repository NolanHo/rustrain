#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT_DIR="${RUSTRAIN_QWEN_TP_LINEAR_OUTPUT_DIR:-/tmp/rustrain-runs/qwen-tp-linear-verify-$$}"
MODEL_PATH="${RUSTRAIN_QWEN_TP_LINEAR_MODEL_PATH:-/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  qwen-tp-linear-rank-smoke \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}/ranks"

python - "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
rank_dir = output_dir / "ranks"
summaries = sorted(rank_dir.glob("qwen-tp-linear-rank-*.json"))
if len(summaries) != 2:
    raise SystemExit(f"expected 2 TP rank summaries under {rank_dir}, found {len(summaries)}")

evidence = []
covered = []
for path in summaries:
    data = json.loads(path.read_text())
    rank = int(data["rank"])
    if data["world_size"] != 2:
        raise SystemExit(f"{path} world_size {data['world_size']} != 2")
    if data["local_rank"] != rank:
        raise SystemExit(f"{path} local_rank {data['local_rank']} != rank {rank}")
    shard_shape = data["shard_output_shape"]
    full_shape = data["full_output_shape"]
    if shard_shape[0] != full_shape[0]:
        raise SystemExit(f"{path} batch dimension mismatch: shard={shard_shape}, full={full_shape}")
    if shard_shape[1] * 2 != full_shape[1]:
        raise SystemExit(f"{path} shard dimension {shard_shape[1]} is not half of full {full_shape[1]}")
    covered.append((int(data["shard_start"]), int(data["shard_end"])))
    if rank == 0:
        if data["max_abs"] is None or data["mean_abs"] is None:
            raise SystemExit(f"{path} rank0 missing parity diff fields")
        if float(data["max_abs"]) > 1e-5:
            raise SystemExit(f"{path} max_abs too large: {data['max_abs']}")
    evidence.append(
        {
            "rank": rank,
            "shard_start": data["shard_start"],
            "shard_end": data["shard_end"],
            "shard_output_shape": shard_shape,
            "max_abs": data["max_abs"],
        }
    )

covered.sort()
if covered[0][0] != 0 or covered[0][1] != covered[1][0]:
    raise SystemExit(f"TP shards are not contiguous: {covered}")

launch_summary = json.loads((output_dir / "launch-summary.json").read_text())
assigned = [rank["assigned_cuda_visible_device"] for rank in launch_summary["ranks"]]
if len(set(assigned)) != 2:
    raise SystemExit(f"expected distinct launch GPU assignments, got {assigned}")

print(json.dumps({"qwen_tp_linear_verified": evidence, "assigned_cuda_visible_devices": assigned}, indent=2))
PY
