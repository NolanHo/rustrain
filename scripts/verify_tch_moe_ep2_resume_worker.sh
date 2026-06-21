#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

BASE_OUTPUT_DIR="${RUSTRAIN_TCH_MOE_EP2_BASE_OUTPUT_DIR:-/tmp/rustrain-runs/tch-moe-ep2-resume-base-$$}"
RESUME_OUTPUT_DIR="${RUSTRAIN_TCH_MOE_EP2_RESUME_OUTPUT_DIR:-/tmp/rustrain-runs/tch-moe-ep2-resume-continue-$$}"
CONFIG="${RUSTRAIN_TCH_MOE_EP2_CONFIG:-configs/tch_moe_ep2.toml}"
export BASE_OUTPUT_DIR RESUME_OUTPUT_DIR

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${BASE_OUTPUT_DIR}" \
  train --config "${CONFIG}"

BASE_MANIFEST="$(
  python - <<'PY'
import json
import os
import pathlib

output_dir = pathlib.Path(os.environ["BASE_OUTPUT_DIR"])
rank0_paths = sorted(output_dir.rglob("ep-tch-moe-rank-0.json"))
if len(rank0_paths) != 1:
    raise SystemExit(f"expected one base EP rank0 summary under {output_dir}, found {len(rank0_paths)}")
summary = json.loads(rank0_paths[0].read_text())
manifest = summary.get("ep_global_manifest_output")
if not manifest:
    raise SystemExit("base EP run did not report ep_global_manifest_output")
manifest_path = pathlib.Path(manifest)
if not manifest_path.exists():
    raise SystemExit(f"base EP sharded global manifest does not exist: {manifest_path}")
print(manifest)
PY
)"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${RESUME_OUTPUT_DIR}" \
  train --config "${CONFIG}" --resume-from "${BASE_MANIFEST}"

python - "${BASE_MANIFEST}" <<'PY'
import json
import os
import pathlib
import sys

base_manifest = pathlib.Path(sys.argv[1])
base_manifest_data = json.loads(base_manifest.read_text())
resume_output_dir = pathlib.Path(os.environ["RESUME_OUTPUT_DIR"])
summaries = sorted(resume_output_dir.rglob("ep-tch-moe-rank-*.json"))
if len(summaries) != 2:
    raise SystemExit(f"expected 2 resume EP rank summaries under {resume_output_dir}, found {len(summaries)}")

if base_manifest_data["format"] != "rustrain.ep_sharded.v1":
    raise SystemExit(f"unexpected base EP manifest format {base_manifest_data['format']}")
if base_manifest_data.get("manifest_kind") != "global":
    raise SystemExit(f"unexpected base EP manifest_kind {base_manifest_data.get('manifest_kind')}")
parallel = base_manifest_data["parallel"]
if parallel["expert_model_parallel_size"] != 2:
    raise SystemExit(f"unexpected base EP parallel config {parallel}")

evidence = []
for path in summaries:
    data = json.loads(path.read_text())
    rank = int(data["rank"])
    if data.get("resume_from") != str(base_manifest):
        raise SystemExit(f"{path} resume_from {data.get('resume_from')} != {base_manifest}")
    if data.get("resumed_sharded_checkpoint") is not True:
        raise SystemExit(f"{path} did not report resumed_sharded_checkpoint=true")
    if int(data.get("resume_global_step", -1)) != int(base_manifest_data["global_step"]):
        raise SystemExit(
            f"{path} resume_global_step {data.get('resume_global_step')} != {base_manifest_data['global_step']}"
        )
    if int(data.get("resume_sharded_manifest_tensor_count", -1)) != 2:
        raise SystemExit(
            f"{path} expected 2 resumed expert shards, got {data.get('resume_sharded_manifest_tensor_count')}"
        )
    for key in [
        "resume_reload_expert_up_max_abs",
        "resume_reload_expert_down_max_abs",
        "resume_reload_optimizer_max_abs",
        "resume_reload_loss_delta",
        "resume_next_step_delta",
        "resume_next_step_expert_up_max_abs",
        "resume_next_step_expert_down_max_abs",
        "resume_next_step_optimizer_max_abs",
    ]:
        if float(data.get(key, 1.0)) > 1e-7:
            raise SystemExit(f"{path} {key} too large: {data.get(key)}")
    if int(data.get("resume_next_step_optimizer_step_delta", -1)) != 0:
        raise SystemExit(
            f"{path} resume_next_step_optimizer_step_delta must be 0, got {data.get('resume_next_step_optimizer_step_delta')}"
        )
    rank_manifest = next(
        (entry for entry in base_manifest_data["ranks"] if int(entry["rank"]) == rank),
        None,
    )
    if rank_manifest is None:
        raise SystemExit(f"base manifest is missing rank {rank}")
    if data.get("resume_model_safetensors") != rank_manifest["model_safetensors"]:
        raise SystemExit(f"{path} did not resume rank-owned model safetensors for rank {rank}")
    if data.get("resume_optimizer_safetensors") != rank_manifest["optimizer_safetensors"]:
        raise SystemExit(f"{path} did not resume rank-owned optimizer safetensors for rank {rank}")
    if not pathlib.Path(data["resume_model_safetensors"]).exists():
        raise SystemExit(f"{path} resumed model safetensors does not exist")
    if not pathlib.Path(data["resume_optimizer_safetensors"]).exists():
        raise SystemExit(f"{path} resumed optimizer safetensors does not exist")
    evidence.append(
        {
            "rank": rank,
            "resume_from": data["resume_from"],
            "resume_global_step": data["resume_global_step"],
            "resume_reload_loss_delta": data["resume_reload_loss_delta"],
            "resume_next_step_delta": data["resume_next_step_delta"],
            "resume_model_safetensors": data["resume_model_safetensors"],
        }
    )

print(json.dumps({"tch_moe_ep2_external_resume": evidence}, indent=2, sort_keys=True))
PY
