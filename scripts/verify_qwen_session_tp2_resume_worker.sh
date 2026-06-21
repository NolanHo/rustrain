#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

BASE_OUTPUT_DIR="${RUSTRAIN_TP_BASE_OUTPUT_DIR:-/tmp/rustrain-runs/qwen-session-tp2-resume-base-$$}"
RESUME_OUTPUT_DIR="${RUSTRAIN_TP_RESUME_OUTPUT_DIR:-/tmp/rustrain-runs/qwen-session-tp2-resume-continue-$$}"
CONFIG="${RUSTRAIN_QWEN_SESSION_TP_CONFIG:-configs/qwen_session_tp2.toml}"
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
rank0_paths = sorted(output_dir.rglob("qwen-session-tp-rank-0.json"))
if len(rank0_paths) != 1:
    raise SystemExit(f"expected one base TP rank0 summary under {output_dir}, found {len(rank0_paths)}")
summary = json.loads(rank0_paths[0].read_text())
manifest = summary.get("sharded_global_manifest_output")
if not manifest:
    raise SystemExit("base TP run did not report sharded_global_manifest_output")
manifest_path = pathlib.Path(manifest)
if not manifest_path.exists():
    raise SystemExit(f"base TP sharded global manifest does not exist: {manifest_path}")
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

from safetensors import safe_open

base_manifest = pathlib.Path(sys.argv[1])
base_manifest_data = json.loads(base_manifest.read_text())
resume_output_dir = pathlib.Path(os.environ["RESUME_OUTPUT_DIR"])
summaries = sorted(resume_output_dir.rglob("qwen-session-tp-rank-*.json"))
if len(summaries) != 2:
    raise SystemExit(f"expected 2 resume TP rank summaries under {resume_output_dir}, found {len(summaries)}")


def tensor_shapes(path):
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = list(handle.get_tensor(key).shape)
    return tensors


def validate_tp_global_manifest(manifest_path, expected_global_step):
    manifest_path = pathlib.Path(manifest_path)
    if not manifest_path.exists():
        raise SystemExit(f"missing TP sharded global manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != "rustrain.qwen_sharded.v1":
        raise SystemExit(f"{manifest_path} unexpected format {manifest['format']}")
    if manifest["base_model_path"] != "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct":
        raise SystemExit(f"{manifest_path} unexpected base_model_path {manifest['base_model_path']}")
    if manifest["tokenizer_path"] != "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct/tokenizer.json":
        raise SystemExit(f"{manifest_path} unexpected tokenizer_path {manifest['tokenizer_path']}")
    if int(manifest["global_step"]) != expected_global_step:
        raise SystemExit(f"{manifest_path} global_step {manifest['global_step']} != {expected_global_step}")
    if int(manifest["consumed_samples"]) != 2:
        raise SystemExit(f"{manifest_path} consumed_samples {manifest['consumed_samples']} != 2")
    if int(manifest["consumed_tokens"]) != 10:
        raise SystemExit(f"{manifest_path} consumed_tokens {manifest['consumed_tokens']} != 10")
    for key in ["data_cursor_next", "data_epoch_next", "data_sample_offset_next", "data_train_samples"]:
        if manifest[key] is not None:
            raise SystemExit(f"{manifest_path} focused TP manifest should leave {key} null, got {manifest[key]}")
    if manifest["dataset_source_files"] != []:
        raise SystemExit(f"{manifest_path} focused TP manifest should not claim dataset_source_files")
    if manifest["dataset_source_sample_counts"] != []:
        raise SystemExit(f"{manifest_path} focused TP manifest should not claim dataset_source_sample_counts")
    if manifest["dataset_fingerprint"] != "":
        raise SystemExit(f"{manifest_path} focused TP manifest should not claim dataset_fingerprint")
    if manifest["dataset_shuffle"] is not True:
        raise SystemExit(f"{manifest_path} dataset_shuffle {manifest['dataset_shuffle']} != true")
    if int(manifest["seed"]) != 42:
        raise SystemExit(f"{manifest_path} seed {manifest['seed']} != 42")
    if manifest["dtype"] != "fp32":
        raise SystemExit(f"{manifest_path} dtype {manifest['dtype']} != fp32")
    if manifest["optimizer"] != "adamw_zero_slots_smoke":
        raise SystemExit(f"{manifest_path} optimizer {manifest['optimizer']} != adamw_zero_slots_smoke")
    if manifest["scheduler"] != "constant":
        raise SystemExit(f"{manifest_path} scheduler {manifest['scheduler']} != constant")
    expected_parallel = {
        "data_parallel_size": 1,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "context_parallel_size": 1,
    }
    if manifest["parallel"] != expected_parallel:
        raise SystemExit(f"{manifest_path} unexpected parallel config {manifest['parallel']}")
    if len(manifest["ranks"]) != 2:
        raise SystemExit(f"{manifest_path} expected 2 ranks, got {len(manifest['ranks'])}")
    if sorted(int(rank["tensor_model_parallel_rank"]) for rank in manifest["ranks"]) != [0, 1]:
        raise SystemExit(f"{manifest_path} does not cover TP ranks 0 and 1")
    for rank_manifest in manifest["ranks"]:
        rank = int(rank_manifest["rank"])
        if rank_manifest["data_parallel_rank"] != 0:
            raise SystemExit(f"{manifest_path} rank {rank} data_parallel_rank must be 0")
        if rank_manifest["pipeline_model_parallel_rank"] != 0:
            raise SystemExit(f"{manifest_path} rank {rank} pipeline_model_parallel_rank must be 0")
        if rank_manifest["expert_model_parallel_rank"] != 0:
            raise SystemExit(f"{manifest_path} rank {rank} expert_model_parallel_rank must be 0")
        if rank_manifest["context_parallel_rank"] != 0:
            raise SystemExit(f"{manifest_path} rank {rank} context_parallel_rank must be 0")
        if len(rank_manifest["shards"]) != 9:
            raise SystemExit(f"{manifest_path} rank {rank} expected 9 shards, got {len(rank_manifest['shards'])}")
        model_path = pathlib.Path(rank_manifest["model_safetensors"])
        optimizer_path = pathlib.Path(rank_manifest["optimizer_safetensors"])
        if not model_path.exists() or model_path.stat().st_size == 0:
            raise SystemExit(f"{manifest_path} rank {rank} missing model_safetensors {model_path}")
        if not optimizer_path.exists() or optimizer_path.stat().st_size == 0:
            raise SystemExit(f"{manifest_path} rank {rank} missing optimizer_safetensors {optimizer_path}")
        model_shapes = tensor_shapes(model_path)
        optimizer_shapes = tensor_shapes(optimizer_path)
        for shard in rank_manifest["shards"]:
            if shard["shard_name"] not in model_shapes:
                raise SystemExit(f"{manifest_path} rank {rank} missing model shard {shard['shard_name']}")
            if model_shapes[shard["shard_name"]] != shard["shard_shape"]:
                raise SystemExit(
                    f"{manifest_path} rank {rank} shard {shard['shard_name']} shape "
                    f"{model_shapes[shard['shard_name']]} != {shard['shard_shape']}"
                )
            for slot_key in ["optimizer_m_name", "optimizer_v_name"]:
                slot_name = shard[slot_key]
                if slot_name not in optimizer_shapes:
                    raise SystemExit(f"{manifest_path} rank {rank} missing optimizer slot {slot_name}")
                if optimizer_shapes[slot_name] != shard["shard_shape"]:
                    raise SystemExit(
                        f"{manifest_path} rank {rank} optimizer slot {slot_name} shape "
                        f"{optimizer_shapes[slot_name]} != {shard['shard_shape']}"
                    )
    return manifest


base_manifest_data = validate_tp_global_manifest(base_manifest, expected_global_step=1)

evidence = []
resume_global_manifests = set()
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
    if int(data.get("resume_sharded_manifest_tensor_count", -1)) != 9:
        raise SystemExit(
            f"{path} expected 9 resumed focused shards, got {data.get('resume_sharded_manifest_tensor_count')}"
        )
    if float(data.get("resume_restore_max_abs", 1.0)) > 1e-3:
        raise SystemExit(f"{path} resume_restore_max_abs too large: {data.get('resume_restore_max_abs')}")
    if float(data.get("resume_next_update_max_abs", 1.0)) > 1e-3:
        raise SystemExit(
            f"{path} resume_next_update_max_abs too large: {data.get('resume_next_update_max_abs')}"
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
    if pathlib.Path(data["resume_rank_manifest_output"]).name != f"qwen-session-tp-sharded-rank-{rank}.json":
        raise SystemExit(f"{path} unexpected resume_rank_manifest_output {data['resume_rank_manifest_output']}")
    resume_global_manifests.add(data["sharded_global_manifest_output"])
    evidence.append(
        {
            "rank": rank,
            "resume_from": data["resume_from"],
            "resume_global_step": data["resume_global_step"],
            "resume_restore_max_abs": data["resume_restore_max_abs"],
            "resume_next_update_max_abs": data["resume_next_update_max_abs"],
            "resume_model_safetensors": data["resume_model_safetensors"],
        }
    )

if len(resume_global_manifests) != 1:
    raise SystemExit(f"expected one resumed launch global manifest, got {sorted(resume_global_manifests)}")
resume_global_manifest_path = pathlib.Path(next(iter(resume_global_manifests)))
resume_manifest = validate_tp_global_manifest(resume_global_manifest_path, expected_global_step=1)
for rank_manifest in resume_manifest["ranks"]:
    rank = int(rank_manifest["rank"])
    rank_manifest_path = resume_global_manifest_path.parent / f"qwen-session-tp-sharded-rank-{rank}.json"
    if not rank_manifest_path.exists():
        raise SystemExit(f"resumed launch missing rank manifest file {rank_manifest_path}")
    if json.loads(rank_manifest_path.read_text()) != rank_manifest:
        raise SystemExit(f"resumed global manifest embedded rank {rank} does not match {rank_manifest_path}")

print(json.dumps({"qwen_session_tp2_external_resume": evidence}, indent=2, sort_keys=True))
PY
