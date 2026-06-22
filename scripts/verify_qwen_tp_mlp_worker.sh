#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT_DIR="${RUSTRAIN_QWEN_TP_MLP_OUTPUT_DIR:-/tmp/rustrain-runs/qwen-tp-mlp-verify-$$}"
MODEL_PATH="${RUSTRAIN_QWEN_TP_MLP_MODEL_PATH:-/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  qwen-tp-mlp-rank-smoke \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}/ranks"

python - "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
rank_dir = output_dir / "ranks"
summaries = sorted(rank_dir.glob("qwen-tp-mlp-rank-*.json"))
if len(summaries) != 2:
    raise SystemExit(f"expected 2 TP MLP rank summaries under {rank_dir}, found {len(summaries)}")

def require_complete_qwen_model_path(path, summary_path):
    model_path = pathlib.Path(path)
    missing = [
        name
        for name in ("config.json", "tokenizer.json", "model.safetensors")
        if not (model_path / name).exists()
    ]
    if missing:
        raise SystemExit(
            f"{summary_path} model_path {model_path} is not a complete Qwen checkpoint; missing {missing}"
        )
    return str(model_path)

evidence = []
intermediate = []
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
    activation_shape = data["activation_shard_shape"]
    contribution_shape = data["output_contribution_shape"]
    full_shape = data["full_output_shape"]
    if contribution_shape != full_shape:
        raise SystemExit(f"{path} output contribution shape {contribution_shape} != full output shape {full_shape}")
    if contribution_shape != input_shape:
        raise SystemExit(f"{path} output contribution shape {contribution_shape} != input shape {input_shape}")
    if activation_shape[0] != input_shape[0] or activation_shape[1] != input_shape[1]:
        raise SystemExit(f"{path} activation shape does not preserve batch/seq: {activation_shape} vs {input_shape}")
    intermediate.append((int(data["intermediate_start"]), int(data["intermediate_end"])))
    if rank == 0:
        if data["max_abs"] is None or data["mean_abs"] is None:
            raise SystemExit(f"{path} rank0 missing parity diff fields")
        if float(data["max_abs"]) > 1e-5:
            raise SystemExit(f"{path} max_abs too large: {data['max_abs']}")
    else:
        if data["max_abs"] is not None or data["mean_abs"] is not None:
            raise SystemExit(f"{path} non-rank0 unexpectedly has parity diff fields")
    evidence.append(
        {
            "rank": rank,
            "intermediate": [data["intermediate_start"], data["intermediate_end"]],
            "activation_shard_shape": activation_shape,
            "max_abs": data["max_abs"],
        }
    )

intermediate.sort()
if intermediate[0][0] != 0 or intermediate[0][1] != intermediate[1][0]:
    raise SystemExit(f"intermediate shards are not contiguous: {intermediate}")

if len(resolved_model_paths) != 1:
    raise SystemExit(f"expected all ranks to resolve the same Qwen model path, got {sorted(resolved_model_paths)}")

launch_summary = json.loads((output_dir / "launch-summary.json").read_text())
assigned = [rank["assigned_cuda_visible_device"] for rank in launch_summary["ranks"]]
if len(set(assigned)) != 2:
    raise SystemExit(f"expected distinct launch GPU assignments, got {assigned}")

print(json.dumps({
    "qwen_tp_mlp_verified": evidence,
    "assigned_cuda_visible_devices": assigned,
    "resolved_model_path": sorted(resolved_model_paths)[0],
}, indent=2))
PY
