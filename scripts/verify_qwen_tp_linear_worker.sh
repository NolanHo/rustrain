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

expected_projection_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
evidence = []
covered_by_projection = {name: [] for name in expected_projection_names}
resolved_model_paths = set()
for path in summaries:
    data = json.loads(path.read_text())
    resolved_model_paths.add(require_complete_qwen_model_path(data["model_path"], path))
    rank = int(data["rank"])
    if data["world_size"] != 2:
        raise SystemExit(f"{path} world_size {data['world_size']} != 2")
    if data["local_rank"] != rank:
        raise SystemExit(f"{path} local_rank {data['local_rank']} != rank {rank}")
    projections = data.get("projections")
    if not isinstance(projections, list):
        raise SystemExit(f"{path} missing projections list")
    names = [projection.get("name") for projection in projections]
    if names != expected_projection_names:
        raise SystemExit(f"{path} projections {names} != {expected_projection_names}")

    rank_evidence = {"rank": rank, "projections": []}
    for projection in projections:
        name = projection["name"]
        shard_shape = projection["shard_output_shape"]
        full_shape = projection["full_output_shape"]
        if shard_shape[0] != full_shape[0]:
            raise SystemExit(
                f"{path} {name} batch dimension mismatch: shard={shard_shape}, full={full_shape}"
            )
        if shard_shape[1] * 2 != full_shape[1]:
            raise SystemExit(
                f"{path} {name} shard dimension {shard_shape[1]} is not half of full {full_shape[1]}"
            )
        covered_by_projection[name].append(
            (int(projection["shard_start"]), int(projection["shard_end"]))
        )
        if rank == 0:
            if projection["max_abs"] is None or projection["mean_abs"] is None:
                raise SystemExit(f"{path} rank0 missing {name} parity diff fields")
            if float(projection["max_abs"]) > 1e-5:
                raise SystemExit(f"{path} {name} max_abs too large: {projection['max_abs']}")
        else:
            if projection["max_abs"] is not None or projection["mean_abs"] is not None:
                raise SystemExit(f"{path} non-rank0 unexpectedly has {name} parity diff fields")
        rank_evidence["projections"].append(
            {
                "name": name,
                "shard_start": projection["shard_start"],
                "shard_end": projection["shard_end"],
                "shard_output_shape": shard_shape,
                "max_abs": projection["max_abs"],
            }
        )
    evidence.append(rank_evidence)

for name, covered in covered_by_projection.items():
    covered.sort()
    if len(covered) != 2:
        raise SystemExit(f"{name} expected two TP shards, found {covered}")
    if covered[0][0] != 0 or covered[0][1] != covered[1][0]:
        raise SystemExit(f"{name} TP shards are not contiguous: {covered}")

if len(resolved_model_paths) != 1:
    raise SystemExit(f"expected all ranks to resolve the same Qwen model path, got {sorted(resolved_model_paths)}")

launch_summary = json.loads((output_dir / "launch-summary.json").read_text())
assigned = [rank["assigned_cuda_visible_device"] for rank in launch_summary["ranks"]]
if len(set(assigned)) != 2:
    raise SystemExit(f"expected distinct launch GPU assignments, got {assigned}")

print(json.dumps({
    "qwen_tp_linear_verified": evidence,
    "assigned_cuda_visible_devices": assigned,
    "resolved_model_path": sorted(resolved_model_paths)[0],
}, indent=2))
PY
