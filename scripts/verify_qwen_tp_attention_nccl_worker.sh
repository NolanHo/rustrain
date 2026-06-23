#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT_DIR="${RUSTRAIN_QWEN_TP_ATTENTION_NCCL_OUTPUT_DIR:-/tmp/rustrain-runs/qwen-tp-attention-nccl-verify-$$}"
MODEL_PATH="${RUSTRAIN_QWEN_TP_ATTENTION_NCCL_MODEL_PATH:-/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  qwen tp-attention-nccl-rank-smoke \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}/ranks"

python - "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from qwen_verify_utils import require_complete_qwen_model_path

output_dir = pathlib.Path(sys.argv[1])
rank_dir = output_dir / "ranks"
summaries = sorted(rank_dir.glob("qwen-tp-attention-nccl-rank-*.json"))
if len(summaries) != 2:
    raise SystemExit(f"expected 2 TP attention NCCL rank summaries under {rank_dir}, found {len(summaries)}")

evidence = []
q_heads = []
kv_heads = []
resolved_model_paths = set()
for path in summaries:
    data = json.loads(path.read_text())
    resolved_model_paths.add(require_complete_qwen_model_path(data["model_path"], path))
    rank = int(data["rank"])
    if data["world_size"] != 2:
        raise SystemExit(f"{path} world_size {data['world_size']} != 2")
    if data["local_rank"] != rank:
        raise SystemExit(f"{path} local_rank {data['local_rank']} != rank {rank}")
    input_shape = data["input_shape"]
    context_shape = data["context_shard_shape"]
    contribution_shape = data["output_contribution_shape"]
    reduced_shape = data["reduced_output_shape"]
    full_shape = data["full_output_shape"]
    if context_shape[0] != input_shape[0] or context_shape[1] != input_shape[1]:
        raise SystemExit(f"{path} context shape does not preserve batch/seq: {context_shape} vs {input_shape}")
    if context_shape[2] * 2 != input_shape[2]:
        raise SystemExit(f"{path} context hidden shard {context_shape[2]} is not half of input hidden {input_shape[2]}")
    if contribution_shape != full_shape:
        raise SystemExit(f"{path} output contribution shape {contribution_shape} != full output shape {full_shape}")
    if reduced_shape != full_shape:
        raise SystemExit(f"{path} reduced output shape {reduced_shape} != full output shape {full_shape}")
    q_heads.append((int(data["q_head_start"]), int(data["q_head_end"])))
    kv_heads.append((int(data["kv_head_start"]), int(data["kv_head_end"])))
    if float(data["max_abs"]) > 1e-5:
        raise SystemExit(f"{path} max_abs too large: {data['max_abs']}")
    evidence.append(
        {
            "rank": rank,
            "q_heads": [data["q_head_start"], data["q_head_end"]],
            "kv_heads": [data["kv_head_start"], data["kv_head_end"]],
            "context_shard_shape": context_shape,
            "reduced_output_shape": reduced_shape,
            "max_abs": data["max_abs"],
        }
    )

for label, ranges in [("q_heads", q_heads), ("kv_heads", kv_heads)]:
    ranges.sort()
    if ranges[0][0] != 0 or ranges[0][1] != ranges[1][0]:
        raise SystemExit(f"{label} shards are not contiguous: {ranges}")

if len(resolved_model_paths) != 1:
    raise SystemExit(f"expected all ranks to resolve the same Qwen model path, got {sorted(resolved_model_paths)}")

launch_summary = json.loads((output_dir / "launch-summary.json").read_text())
assigned = [rank["assigned_cuda_visible_device"] for rank in launch_summary["ranks"]]
if len(set(assigned)) != 2:
    raise SystemExit(f"expected distinct launch GPU assignments, got {assigned}")

print(json.dumps({
    "qwen_tp_attention_nccl_verified": evidence,
    "assigned_cuda_visible_devices": assigned,
    "resolved_model_path": sorted(resolved_model_paths)[0],
}, indent=2))
PY
