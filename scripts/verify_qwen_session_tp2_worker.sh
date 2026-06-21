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

from safetensors import safe_open

output_dir = pathlib.Path(sys.argv[1])
summaries = sorted(output_dir.rglob("qwen-session-tp-rank-*.json"))
if len(summaries) != 2:
    raise SystemExit(f"expected 2 Qwen session TP rank summaries under {output_dir}, found {len(summaries)}")

evidence = []
q_heads = []
kv_heads = []
intermediate = []
global_manifests = set()
rank_manifest_by_rank = {}


def tensor_shapes(path):
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = list(handle.get_tensor(key).shape)
    return tensors


def tensor_shape_sums(path):
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            tensors[key] = {
                "shape": list(tensor.shape),
                "abs_sum": float(tensor.abs().sum().item()),
            }
    return tensors


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
    for key in [
        "layer0_train_q_grad_norm",
        "layer0_train_k_grad_norm",
        "layer0_train_v_grad_norm",
        "layer0_train_o_grad_norm",
        "layer0_train_gate_grad_norm",
        "layer0_train_up_grad_norm",
        "layer0_train_down_grad_norm",
    ]:
        if float(data[key]) <= 0.0:
            raise SystemExit(f"{path} expected positive {key}, got {data[key]}")
    if data["causal_train_input_shape"] != [1, 5]:
        raise SystemExit(f"{path} unexpected causal_train_input_shape {data['causal_train_input_shape']}")
    if float(data["causal_train_initial_loss_delta"]) > 1e-2:
        raise SystemExit(
            f"{path} causal_train_initial_loss_delta too large: {data['causal_train_initial_loss_delta']}"
        )
    if not data.get("causal_train_loss_improved"):
        raise SystemExit(
            f"{path} expected causal LM train loss to improve, initial={data.get('causal_train_initial_loss')} final={data.get('causal_train_final_loss')}"
        )
    if float(data["causal_train_final_loss"]) >= float(data["causal_train_initial_loss"]):
        raise SystemExit(
            f"{path} causal LM final loss {data['causal_train_final_loss']} is not below initial {data['causal_train_initial_loss']}"
        )
    for key in [
        "causal_train_q_grad_norm",
        "causal_train_k_grad_norm",
        "causal_train_v_grad_norm",
        "causal_train_o_grad_norm",
        "causal_train_gate_grad_norm",
        "causal_train_up_grad_norm",
        "causal_train_down_grad_norm",
    ]:
        if float(data[key]) <= 0.0:
            raise SystemExit(f"{path} expected positive {key}, got {data[key]}")
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
    if float(data["sharded_next_update_max_abs"]) > 1e-3:
        raise SystemExit(f"{path} sharded_next_update_max_abs too large: {data['sharded_next_update_max_abs']}")
    rank_manifest_path = pathlib.Path(data["sharded_rank_manifest_output"])
    if not rank_manifest_path.exists():
        raise SystemExit(f"{path} missing TP rank sharded manifest {rank_manifest_path}")
    rank_manifest = json.loads(rank_manifest_path.read_text())
    rank_manifest_by_rank[rank] = (rank_manifest_path, rank_manifest)
    if rank_manifest["rank"] != rank:
        raise SystemExit(f"{rank_manifest_path} rank {rank_manifest['rank']} != {rank}")
    if rank_manifest["tensor_model_parallel_rank"] != rank:
        raise SystemExit(
            f"{rank_manifest_path} tensor_model_parallel_rank {rank_manifest['tensor_model_parallel_rank']} != {rank}"
        )
    if rank_manifest["data_parallel_rank"] != 0:
        raise SystemExit(f"{rank_manifest_path} expected data_parallel_rank 0, got {rank_manifest['data_parallel_rank']}")
    if rank_manifest["pipeline_model_parallel_rank"] != 0:
        raise SystemExit(
            f"{rank_manifest_path} expected pipeline_model_parallel_rank 0, got {rank_manifest['pipeline_model_parallel_rank']}"
        )
    if rank_manifest["expert_model_parallel_rank"] != 0:
        raise SystemExit(
            f"{rank_manifest_path} expected expert_model_parallel_rank 0, got {rank_manifest['expert_model_parallel_rank']}"
        )
    if rank_manifest["context_parallel_rank"] != 0:
        raise SystemExit(
            f"{rank_manifest_path} expected context_parallel_rank 0, got {rank_manifest['context_parallel_rank']}"
        )
    if len(rank_manifest["shards"]) != 9:
        raise SystemExit(f"{rank_manifest_path} expected 9 shards, got {len(rank_manifest['shards'])}")
    partitions = {entry["partition"] for entry in rank_manifest["shards"]}
    if not {"tp_row", "tp_col", "replicated_norm_smoke"}.issubset(partitions):
        raise SystemExit(f"{rank_manifest_path} missing expected TP partitions, got {sorted(partitions)}")
    model_path = pathlib.Path(rank_manifest["model_safetensors"])
    optimizer_path = pathlib.Path(rank_manifest["optimizer_safetensors"])
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise SystemExit(f"{rank_manifest_path} missing or empty model_safetensors {model_path}")
    if not optimizer_path.exists() or optimizer_path.stat().st_size == 0:
        raise SystemExit(f"{rank_manifest_path} missing or empty optimizer_safetensors {optimizer_path}")
    model_shapes = tensor_shapes(model_path)
    optimizer_shapes = tensor_shape_sums(optimizer_path)
    for entry in rank_manifest["shards"]:
        if not entry["optimizer_m_name"] or not entry["optimizer_v_name"]:
            raise SystemExit(f"{rank_manifest_path} shard {entry['name']} missing optimizer slots")
        if not entry["global_shape"] or not entry["shard_shape"]:
            raise SystemExit(f"{rank_manifest_path} shard {entry['name']} missing shapes")
        if entry["shard_name"] not in model_shapes:
            raise SystemExit(f"{rank_manifest_path} model safetensors missing shard {entry['shard_name']}")
        if model_shapes[entry["shard_name"]] != entry["shard_shape"]:
            raise SystemExit(
                f"{rank_manifest_path} shard {entry['shard_name']} shape {model_shapes[entry['shard_name']]} != {entry['shard_shape']}"
            )
        if entry["optimizer_m_name"] not in optimizer_shapes:
            raise SystemExit(f"{rank_manifest_path} optimizer safetensors missing m slot {entry['optimizer_m_name']}")
        if entry["optimizer_v_name"] not in optimizer_shapes:
            raise SystemExit(f"{rank_manifest_path} optimizer safetensors missing v slot {entry['optimizer_v_name']}")
        if optimizer_shapes[entry["optimizer_m_name"]]["shape"] != entry["shard_shape"]:
            raise SystemExit(
                f"{rank_manifest_path} optimizer m slot {entry['optimizer_m_name']} shape "
                f"{optimizer_shapes[entry['optimizer_m_name']]['shape']} != {entry['shard_shape']}"
            )
        if optimizer_shapes[entry["optimizer_v_name"]]["shape"] != entry["shard_shape"]:
            raise SystemExit(
                f"{rank_manifest_path} optimizer v slot {entry['optimizer_v_name']} shape "
                f"{optimizer_shapes[entry['optimizer_v_name']]['shape']} != {entry['shard_shape']}"
            )
        if entry["partition"] in {"tp_row", "tp_col"}:
            for slot_key in ["optimizer_m_name", "optimizer_v_name"]:
                slot_name = entry[slot_key]
                if optimizer_shapes[slot_name]["abs_sum"] <= 0.0:
                    raise SystemExit(
                        f"{rank_manifest_path} optimizer slot {slot_name} for {entry['name']} is all zero"
                    )
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
            "causal_train_full_loss": data["causal_train_full_loss"],
            "causal_train_initial_loss": data["causal_train_initial_loss"],
            "causal_train_final_loss": data["causal_train_final_loss"],
            "mlp_train_initial_loss": data["mlp_train_initial_loss"],
            "mlp_train_final_loss": data["mlp_train_final_loss"],
            "sharded_restore_max_abs": data["sharded_restore_max_abs"],
            "sharded_restore_mean_abs": data["sharded_restore_mean_abs"],
            "sharded_next_update_max_abs": data["sharded_next_update_max_abs"],
            "sharded_next_update_mean_abs": data["sharded_next_update_mean_abs"],
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
if global_manifest["base_model_path"] != "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct":
    raise SystemExit(f"unexpected TP base_model_path {global_manifest['base_model_path']}")
if global_manifest["tokenizer_path"] != "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct/tokenizer.json":
    raise SystemExit(f"unexpected TP tokenizer_path {global_manifest['tokenizer_path']}")
if int(global_manifest["global_step"]) != 1:
    raise SystemExit(f"unexpected TP global_step {global_manifest['global_step']}")
if int(global_manifest["consumed_samples"]) != 2:
    raise SystemExit(f"unexpected TP consumed_samples {global_manifest['consumed_samples']}")
if int(global_manifest["consumed_tokens"]) != 10:
    raise SystemExit(f"unexpected TP consumed_tokens {global_manifest['consumed_tokens']}")
if global_manifest["data_cursor_next"] is not None:
    raise SystemExit(f"focused TP global manifest should not claim data_cursor_next: {global_manifest['data_cursor_next']}")
if global_manifest["data_epoch_next"] is not None:
    raise SystemExit(f"focused TP global manifest should not claim data_epoch_next: {global_manifest['data_epoch_next']}")
if global_manifest["data_sample_offset_next"] is not None:
    raise SystemExit(
        f"focused TP global manifest should not claim data_sample_offset_next: {global_manifest['data_sample_offset_next']}"
    )
if global_manifest["data_train_samples"] is not None:
    raise SystemExit(f"focused TP global manifest should not claim data_train_samples: {global_manifest['data_train_samples']}")
if global_manifest["dataset_source_files"] != []:
    raise SystemExit(f"focused TP global manifest should not claim dataset_source_files: {global_manifest['dataset_source_files']}")
if global_manifest["dataset_source_sample_counts"] != []:
    raise SystemExit(
        "focused TP global manifest should not claim dataset_source_sample_counts: "
        f"{global_manifest['dataset_source_sample_counts']}"
    )
if global_manifest["dataset_fingerprint"] != "":
    raise SystemExit(f"focused TP global manifest should not claim dataset_fingerprint: {global_manifest['dataset_fingerprint']}")
if global_manifest["dataset_shuffle"] is not True:
    raise SystemExit(f"unexpected TP dataset_shuffle {global_manifest['dataset_shuffle']}")
if int(global_manifest["seed"]) != 42:
    raise SystemExit(f"unexpected TP seed {global_manifest['seed']}")
if global_manifest["dtype"] != "fp32":
    raise SystemExit(f"unexpected TP dtype {global_manifest['dtype']}")
if global_manifest["optimizer"] != "adamw_gradient_slots_smoke":
    raise SystemExit(f"unexpected TP optimizer {global_manifest['optimizer']}")
if global_manifest["scheduler"] != "constant":
    raise SystemExit(f"unexpected TP scheduler {global_manifest['scheduler']}")
parallel = global_manifest["parallel"]
expected_parallel = {
    "data_parallel_size": 1,
    "tensor_model_parallel_size": 2,
    "pipeline_model_parallel_size": 1,
    "expert_model_parallel_size": 1,
    "context_parallel_size": 1,
}
if parallel != expected_parallel:
    raise SystemExit(f"unexpected TP global parallel config {parallel}")
if len(global_manifest["ranks"]) != 2:
    raise SystemExit(f"expected 2 TP global rank manifests, got {len(global_manifest['ranks'])}")
if sorted(rank["tensor_model_parallel_rank"] for rank in global_manifest["ranks"]) != [0, 1]:
    raise SystemExit("TP global manifest does not cover tensor parallel ranks 0 and 1")
for global_rank in global_manifest["ranks"]:
    rank = int(global_rank["rank"])
    manifest_pair = rank_manifest_by_rank.get(rank)
    if manifest_pair is None:
        raise SystemExit(f"TP global manifest references rank {rank}, but no rank manifest was verified")
    _, rank_manifest = manifest_pair
    if global_rank != rank_manifest:
        raise SystemExit(f"TP global manifest embedded rank {rank} does not match rank manifest file")

print(json.dumps({"qwen_session_tp2_verified": evidence, "assigned_cuda_visible_devices": assigned}, indent=2))
PY
