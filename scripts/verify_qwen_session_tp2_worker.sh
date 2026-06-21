#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT_DIR="${RUSTRAIN_QWEN_SESSION_TP_OUTPUT_DIR:-/tmp/rustrain-runs/qwen-session-tp2-verify-$$}"
CONFIG="${RUSTRAIN_QWEN_SESSION_TP_CONFIG:-configs/qwen_session_tp2.toml}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  train --config "${CONFIG}"

python - "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
summaries = sorted(output_dir.rglob("qwen-session-tp-rank-*.json"))
if len(summaries) != 2:
    raise SystemExit(f"expected 2 Qwen session TP rank summaries under {output_dir}, found {len(summaries)}")

evidence = []
q_heads = []
kv_heads = []
intermediate = []
global_manifests = set()
for path in summaries:
    data = json.loads(path.read_text())
    rank = int(data["rank"])
    if data["world_size"] != 2:
        raise SystemExit(f"{path} world_size {data['world_size']} != 2")
    if data["tensor_model_parallel_size"] != 2:
        raise SystemExit(f"{path} tensor_model_parallel_size {data['tensor_model_parallel_size']} != 2")
    if data["data_parallel_size"] != 1:
        raise SystemExit(f"{path} data_parallel_size {data['data_parallel_size']} != 1")
    if data["local_rank"] != rank:
        raise SystemExit(f"{path} local_rank {data['local_rank']} != rank {rank}")
    if float(data["attention_max_abs"]) > 1e-5:
        raise SystemExit(f"{path} attention_max_abs too large: {data['attention_max_abs']}")
    if float(data["mlp_max_abs"]) > 1e-5:
        raise SystemExit(f"{path} mlp_max_abs too large: {data['mlp_max_abs']}")
    if float(data["layer0_max_abs"]) > 1e-5:
        raise SystemExit(f"{path} layer0_max_abs too large: {data['layer0_max_abs']}")
    if not data.get("attention_train_loss_improved"):
        raise SystemExit(
            f"{path} expected attention train loss to improve, initial={data.get('attention_train_initial_loss')} final={data.get('attention_train_final_loss')}"
        )
    if float(data["attention_train_final_loss"]) >= float(data["attention_train_initial_loss"]):
        raise SystemExit(
            f"{path} attention train final loss {data['attention_train_final_loss']} is not below initial {data['attention_train_initial_loss']}"
        )
    for key in [
        "attention_train_q_grad_norm",
        "attention_train_k_grad_norm",
        "attention_train_v_grad_norm",
        "attention_train_o_grad_norm",
    ]:
        if float(data[key]) <= 0.0:
            raise SystemExit(f"{path} expected positive {key}, got {data[key]}")
    if not data.get("mlp_train_loss_improved"):
        raise SystemExit(
            f"{path} expected MLP train loss to improve, initial={data.get('mlp_train_initial_loss')} final={data.get('mlp_train_final_loss')}"
        )
    if float(data["mlp_train_final_loss"]) >= float(data["mlp_train_initial_loss"]):
        raise SystemExit(
            f"{path} MLP train final loss {data['mlp_train_final_loss']} is not below initial {data['mlp_train_initial_loss']}"
        )
    if not data.get("layer0_train_loss_improved"):
        raise SystemExit(
            f"{path} expected layer0 train loss to improve, initial={data.get('layer0_train_initial_loss')} final={data.get('layer0_train_final_loss')}"
        )
    if float(data["layer0_train_final_loss"]) >= float(data["layer0_train_initial_loss"]):
        raise SystemExit(
            f"{path} layer0 train final loss {data['layer0_train_final_loss']} is not below initial {data['layer0_train_initial_loss']}"
        )
    for key in ["mlp_train_gate_grad_norm", "mlp_train_up_grad_norm", "mlp_train_down_grad_norm"]:
        if float(data[key]) <= 0.0:
            raise SystemExit(f"{path} expected positive {key}, got {data[key]}")
    attention_reduced = data["attention_reduced_output_shape"]
    layer0_reduced = data["layer0_reduced_output_shape"]
    mlp_reduced = data["mlp_reduced_output_shape"]
    if attention_reduced != [1, 9, 896]:
        raise SystemExit(f"{path} unexpected attention reduced output shape {attention_reduced}")
    if layer0_reduced != [1, 6, 896]:
        raise SystemExit(f"{path} unexpected layer0 reduced output shape {layer0_reduced}")
    if mlp_reduced != [1, 7, 896]:
        raise SystemExit(f"{path} unexpected MLP reduced output shape {mlp_reduced}")
    if int(data["sharded_manifest_tensor_count"]) != 9:
        raise SystemExit(f"{path} expected 9 TP sharded manifest tensors, got {data['sharded_manifest_tensor_count']}")
    if float(data["sharded_restore_max_abs"]) > 1e-3:
        raise SystemExit(f"{path} sharded_restore_max_abs too large: {data['sharded_restore_max_abs']}")
    rank_manifest_path = pathlib.Path(data["sharded_rank_manifest_output"])
    if not rank_manifest_path.exists():
        raise SystemExit(f"{path} missing TP rank sharded manifest {rank_manifest_path}")
    rank_manifest = json.loads(rank_manifest_path.read_text())
    if rank_manifest["rank"] != rank:
        raise SystemExit(f"{rank_manifest_path} rank {rank_manifest['rank']} != {rank}")
    if rank_manifest["tensor_model_parallel_rank"] != rank:
        raise SystemExit(
            f"{rank_manifest_path} tensor_model_parallel_rank {rank_manifest['tensor_model_parallel_rank']} != {rank}"
        )
    if len(rank_manifest["shards"]) != 9:
        raise SystemExit(f"{rank_manifest_path} expected 9 shards, got {len(rank_manifest['shards'])}")
    partitions = {entry["partition"] for entry in rank_manifest["shards"]}
    if not {"tp_row", "tp_col", "replicated_norm_smoke"}.issubset(partitions):
        raise SystemExit(f"{rank_manifest_path} missing expected TP partitions, got {sorted(partitions)}")
    for entry in rank_manifest["shards"]:
        if not entry["optimizer_m_name"] or not entry["optimizer_v_name"]:
            raise SystemExit(f"{rank_manifest_path} shard {entry['name']} missing optimizer slots")
        if not entry["global_shape"] or not entry["shard_shape"]:
            raise SystemExit(f"{rank_manifest_path} shard {entry['name']} missing shapes")
    global_manifests.add(data["sharded_global_manifest_output"])
    q_heads.append((int(data["attention_q_head_start"]), int(data["attention_q_head_end"])))
    kv_heads.append((int(data["attention_kv_head_start"]), int(data["attention_kv_head_end"])))
    intermediate.append((int(data["mlp_intermediate_start"]), int(data["mlp_intermediate_end"])))
    evidence.append(
        {
            "rank": rank,
            "attention_q_heads": [data["attention_q_head_start"], data["attention_q_head_end"]],
            "attention_kv_heads": [data["attention_kv_head_start"], data["attention_kv_head_end"]],
            "mlp_intermediate": [data["mlp_intermediate_start"], data["mlp_intermediate_end"]],
            "attention_max_abs": data["attention_max_abs"],
            "mlp_max_abs": data["mlp_max_abs"],
            "attention_train_initial_loss": data["attention_train_initial_loss"],
            "attention_train_final_loss": data["attention_train_final_loss"],
            "layer0_max_abs": data["layer0_max_abs"],
            "layer0_train_initial_loss": data["layer0_train_initial_loss"],
            "layer0_train_final_loss": data["layer0_train_final_loss"],
            "mlp_train_initial_loss": data["mlp_train_initial_loss"],
            "mlp_train_final_loss": data["mlp_train_final_loss"],
            "sharded_restore_max_abs": data["sharded_restore_max_abs"],
            "sharded_restore_mean_abs": data["sharded_restore_mean_abs"],
        }
    )

for label, ranges in [
    ("attention_q_heads", q_heads),
    ("attention_kv_heads", kv_heads),
    ("mlp_intermediate", intermediate),
]:
    ranges.sort()
    if ranges[0][0] != 0 or ranges[0][1] != ranges[1][0]:
        raise SystemExit(f"{label} shards are not contiguous: {ranges}")

launch_summary = json.loads((output_dir / "launch-summary.json").read_text())
assigned = [rank["assigned_cuda_visible_device"] for rank in launch_summary["ranks"]]
if len(set(assigned)) != 2:
    raise SystemExit(f"expected distinct launch GPU assignments, got {assigned}")

if len(global_manifests) != 1:
    raise SystemExit(f"expected one shared TP global manifest, got {global_manifests}")
global_manifest_path = pathlib.Path(next(iter(global_manifests)))
if not global_manifest_path.exists():
    raise SystemExit(f"missing TP global sharded manifest {global_manifest_path}")
global_manifest = json.loads(global_manifest_path.read_text())
if global_manifest["format"] != "rustrain.qwen_sharded.v1":
    raise SystemExit(f"unexpected TP global manifest format {global_manifest['format']}")
parallel = global_manifest["parallel"]
if parallel["tensor_model_parallel_size"] != 2 or parallel["data_parallel_size"] != 1:
    raise SystemExit(f"unexpected TP global parallel config {parallel}")
if len(global_manifest["ranks"]) != 2:
    raise SystemExit(f"expected 2 TP global rank manifests, got {len(global_manifest['ranks'])}")
if sorted(rank["tensor_model_parallel_rank"] for rank in global_manifest["ranks"]) != [0, 1]:
    raise SystemExit("TP global manifest does not cover tensor parallel ranks 0 and 1")

print(json.dumps({"qwen_session_tp2_verified": evidence, "assigned_cuda_visible_devices": assigned}, indent=2))
PY
