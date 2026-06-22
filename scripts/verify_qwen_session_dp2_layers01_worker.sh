#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT_DIR="${RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR:-/tmp/rustrain-runs/qwen-session-dp2-layers01-verify-$(date +%Y%m%d-%H%M%S)-$$}"
CONFIG="${RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG:-configs/qwen_session_dp2_layers01.toml}"
EXPECTED_TRAINABLE_TENSORS="${RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS:-25}"
EXPECTED_DTYPE="${RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND:-fp32}"
EXPECTED_DATASET_SEED="${RUSTRAIN_EXPECTED_DATASET_ORDER_SEED:-}"
EXPECTED_TRAINABLE_NAMES="${RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES:-model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.1.self_attn.q_proj.weight,model.layers.1.mlp.down_proj.weight,model.norm.weight}"
export RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}"
export RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS="${EXPECTED_TRAINABLE_TENSORS}"
export RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND="${EXPECTED_DTYPE}"
export RUSTRAIN_EXPECTED_DATASET_ORDER_SEED="${EXPECTED_DATASET_SEED}"
export RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES="${EXPECTED_TRAINABLE_NAMES}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  train --config "${CONFIG}"

python - <<'PY'
import json
import os
import pathlib

output_dir = pathlib.Path(os.environ["RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR"])
expected_trainable_tensors = int(os.environ["RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS"])
expected_dtype = os.environ.get("RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND")
expected_dataset_seed = os.environ.get("RUSTRAIN_EXPECTED_DATASET_ORDER_SEED")
expected_trainable_names = [
    name.strip()
    for name in os.environ["RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES"].split(",")
    if name.strip()
]
rank_summaries = sorted(output_dir.rglob("qwen-session-dp-rank-*.json"))
if len(rank_summaries) != 2:
    raise SystemExit(
        f"expected 2 qwen session DP rank summaries under {output_dir}, found {len(rank_summaries)}"
    )

evidence = []
rank0_manifest_path = None
def verify_source_sample_counts(value, dataset_source_files, dataset_total_samples, context):
    if not isinstance(value, list) or not value:
        raise SystemExit(f"{context} dataset_source_sample_counts must be a non-empty list")
    actual_files = []
    total_samples = 0
    for entry in value:
        if not isinstance(entry, dict):
            raise SystemExit(f"{context} dataset_source_sample_counts entry must be an object: {entry}")
        source_path = entry.get("path")
        samples = entry.get("samples")
        if not source_path:
            raise SystemExit(f"{context} dataset_source_sample_counts entry is missing path: {entry}")
        if samples is None or int(samples) <= 0:
            raise SystemExit(f"{context} dataset_source_sample_counts entry must have positive samples: {entry}")
        actual_files.append(source_path)
        total_samples += int(samples)
    if actual_files != dataset_source_files:
        raise SystemExit(
            f"{context} dataset_source_sample_counts paths {actual_files} do not match dataset_source_files {dataset_source_files}"
        )
    if total_samples != int(dataset_total_samples):
        raise SystemExit(
            f"{context} dataset_source_sample_counts total {total_samples} does not match dataset_total_samples {dataset_total_samples}"
        )
    return value

for path in rank_summaries:
    data = json.loads(path.read_text())
    if data.get("world_size") != 2:
        raise SystemExit(f"{path} expected world_size 2, got {data.get('world_size')}")
    if data.get("steps") != 2:
        raise SystemExit(f"{path} expected steps 2, got {data.get('steps')}")
    if expected_dtype and data.get("dtype") != expected_dtype:
        raise SystemExit(
            f"{path} dtype {data.get('dtype')} does not match expected {expected_dtype}"
        )

    trainable_tensors = data.get("trainable_tensors")
    if not isinstance(trainable_tensors, list):
        raise SystemExit(f"{path} trainable_tensors must be a list")
    if len(trainable_tensors) != expected_trainable_tensors:
        raise SystemExit(
            f"{path} expected {expected_trainable_tensors} trainable tensors, got {len(trainable_tensors)}"
        )
    if "model.embed_tokens.weight" in trainable_tensors:
        raise SystemExit(f"{path} DP representative path must not train tied embedding")
    for name in expected_trainable_names:
        if name not in trainable_tensors:
            raise SystemExit(f"{path} missing expected trainable tensor {name}")

    global_step_losses = data.get("global_step_losses")
    if not isinstance(global_step_losses, list) or len(global_step_losses) != 3:
        raise SystemExit(f"{path} expected 3 global_step_losses, got {global_step_losses}")
    if not global_step_losses[-1] < global_step_losses[0]:
        raise SystemExit(f"{path} global loss did not improve: {global_step_losses}")
    if not data.get("global_loss_improved"):
        raise SystemExit(f"{path} global_loss_improved was not true")
    max_grad_delta_tolerance = 2.0 if expected_dtype == "bf16" else 5e-4
    if float(data.get("max_grad_delta", 1.0)) > max_grad_delta_tolerance:
        raise SystemExit(
            f"{path} max_grad_delta too large: {data.get('max_grad_delta')} "
            f"> {max_grad_delta_tolerance}"
        )
    for key in ["reload_delta", "next_step_delta", "sharded_reload_delta", "sharded_next_step_delta"]:
        value = data.get(key)
        if value is None:
            raise SystemExit(f"{path} is missing {key}")
        if float(value) > 1e-5:
            raise SystemExit(f"{path} {key} {value} exceeds tolerance")

    manifest_path = pathlib.Path(data["manifest_output"])
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("trainable_tensors") != trainable_tensors:
        raise SystemExit(f"{path} rank0 manifest trainable_tensors do not match summary")
    if len(manifest.get("tensors", [])) != expected_trainable_tensors:
        raise SystemExit(
            f"{manifest_path} expected {expected_trainable_tensors} manifest tensors, got {len(manifest.get('tensors', []))}"
        )
    if int(manifest.get("tensor_count", -1)) != expected_trainable_tensors:
        raise SystemExit(
            f"{manifest_path} tensor_count {manifest.get('tensor_count')} != {expected_trainable_tensors}"
        )
    if expected_dataset_seed:
        required_dataset_fields = [
            "dataset_total_samples",
            "dataset_total_tokens",
            "dataset_train_samples",
            "dataset_eval_samples",
            "dataset_source_files",
            "dataset_source_sample_counts",
            "dataset_fingerprint",
            "dataset_order_seed",
            "streaming_train_batches",
            "data_cursor_start",
            "data_cursor_end",
            "data_cursor_next",
            "data_epoch_start",
            "data_epoch_end",
            "data_epoch_next",
            "data_sample_offset_start",
            "data_sample_offset_end",
            "data_sample_offset_next",
            "sequence_tokens",
            "local_batch_size",
        ]
        missing_dataset_fields = [
            key for key in required_dataset_fields if data.get(key) is None
        ]
        if missing_dataset_fields:
            raise SystemExit(f"{path} is missing dataset fields: {missing_dataset_fields}")
        for key in [
            "dataset_total_samples",
            "dataset_total_tokens",
            "dataset_train_samples",
            "dataset_eval_samples",
            "sequence_tokens",
            "local_batch_size",
        ]:
            if int(data[key]) <= 0:
                raise SystemExit(f"{path} {key} must be positive, got {data[key]}")
        dataset_source_files = data["dataset_source_files"]
        if not dataset_source_files:
            raise SystemExit(f"{path} dataset_source_files must not be empty")
        if not all(str(source).endswith(".jsonl") for source in dataset_source_files):
            raise SystemExit(
                f"{path} dataset_source_files must only contain JSONL paths, got {dataset_source_files}"
            )
        dataset_source_sample_counts = verify_source_sample_counts(
            data["dataset_source_sample_counts"],
            dataset_source_files,
            data["dataset_total_samples"],
            str(path),
        )
        if not data["dataset_fingerprint"]:
            raise SystemExit(f"{path} dataset_fingerprint must not be empty")
        if int(data["dataset_order_seed"]) != int(expected_dataset_seed):
            raise SystemExit(
                f"{path} dataset_order_seed {data['dataset_order_seed']} does not match expected {expected_dataset_seed}"
            )
        if data.get("streaming_train_batches") is not True:
            raise SystemExit(
                f"{path} expected streaming_train_batches true, got {data.get('streaming_train_batches')}"
            )
        expected_cursor_end = int(data["steps"]) * int(data["local_batch_size"]) * int(data["world_size"])
        if int(data["data_cursor_start"]) != 0:
            raise SystemExit(f"{path} expected data_cursor_start 0, got {data['data_cursor_start']}")
        if int(data["data_cursor_end"]) != expected_cursor_end:
            raise SystemExit(
                f"{path} expected data_cursor_end {expected_cursor_end}, got {data['data_cursor_end']}"
            )
        if int(data["data_cursor_next"]) != expected_cursor_end:
            raise SystemExit(
                f"{path} expected data_cursor_next {expected_cursor_end}, got {data['data_cursor_next']}"
            )
        train_samples = int(data["dataset_train_samples"])
        for cursor_key, epoch_key, offset_key in [
            ("data_cursor_start", "data_epoch_start", "data_sample_offset_start"),
            ("data_cursor_end", "data_epoch_end", "data_sample_offset_end"),
            ("data_cursor_next", "data_epoch_next", "data_sample_offset_next"),
        ]:
            cursor = int(data[cursor_key])
            expected_epoch = cursor // train_samples
            expected_offset = cursor % train_samples
            if int(data[epoch_key]) != expected_epoch:
                raise SystemExit(
                    f"{path} expected {epoch_key} {expected_epoch}, got {data[epoch_key]} from {cursor_key}={cursor}"
                )
            if int(data[offset_key]) != expected_offset:
                raise SystemExit(
                    f"{path} expected {offset_key} {expected_offset}, got {data[offset_key]} from {cursor_key}={cursor}"
                )
        if manifest.get("dataset_source_files") != dataset_source_files:
            raise SystemExit(
                f"{path} rank0 manifest dataset_source_files {manifest.get('dataset_source_files')} does not match summary {dataset_source_files}"
            )
        if manifest.get("dataset_source_sample_counts") != dataset_source_sample_counts:
            raise SystemExit(
                f"{path} rank0 manifest dataset_source_sample_counts {manifest.get('dataset_source_sample_counts')} does not match summary {dataset_source_sample_counts}"
            )
        verify_source_sample_counts(
            manifest.get("dataset_source_sample_counts"),
            dataset_source_files,
            data["dataset_total_samples"],
            f"{path} rank0 manifest",
        )
        if manifest.get("dataset_fingerprint") != data["dataset_fingerprint"]:
            raise SystemExit(
                f"{path} rank0 manifest dataset_fingerprint {manifest.get('dataset_fingerprint')} does not match summary {data['dataset_fingerprint']}"
            )
        if manifest.get("streaming_train_batches") is not True:
            raise SystemExit(
                f"{path} rank0 manifest streaming_train_batches {manifest.get('streaming_train_batches')} is not true"
            )
    sharded_global = json.loads(pathlib.Path(data["sharded_global_manifest_output"]).read_text())
    parallel = sharded_global.get("parallel") or {}
    if int(parallel.get("data_parallel_size", -1)) != 2:
        raise SystemExit(
            f"{path} sharded data_parallel_size {parallel.get('data_parallel_size')} != 2"
        )
    for key in [
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "context_parallel_size",
    ]:
        if int(parallel.get(key, -1)) != 1:
            raise SystemExit(f"{path} sharded {key} {parallel.get(key)} != 1")
    if len(sharded_global.get("ranks", [])) != 2:
        raise SystemExit(f"{path} sharded global manifest must embed 2 rank manifests")
    if expected_dataset_seed:
        if sharded_global.get("dataset_source_files") != data["dataset_source_files"]:
            raise SystemExit(
                f"{path} sharded dataset_source_files {sharded_global.get('dataset_source_files')} does not match summary {data['dataset_source_files']}"
            )
        if sharded_global.get("dataset_source_sample_counts") != dataset_source_sample_counts:
            raise SystemExit(
                f"{path} sharded dataset_source_sample_counts {sharded_global.get('dataset_source_sample_counts')} does not match summary {dataset_source_sample_counts}"
            )
        verify_source_sample_counts(
            sharded_global.get("dataset_source_sample_counts"),
            data["dataset_source_files"],
            data["dataset_total_samples"],
            f"{path} sharded global manifest",
        )
        if sharded_global.get("dataset_fingerprint") != data["dataset_fingerprint"]:
            raise SystemExit(
                f"{path} sharded dataset_fingerprint {sharded_global.get('dataset_fingerprint')} does not match summary {data['dataset_fingerprint']}"
            )
        if int(sharded_global["consumed_samples"]) != int(data["data_cursor_next"]):
            raise SystemExit(
                f"{path} sharded consumed_samples {sharded_global['consumed_samples']} does not match data_cursor_next {data['data_cursor_next']}"
            )
        if int(sharded_global["data_cursor_next"]) != int(data["data_cursor_next"]):
            raise SystemExit(
                f"{path} sharded data_cursor_next {sharded_global['data_cursor_next']} does not match summary {data['data_cursor_next']}"
            )
        if sharded_global.get("streaming_train_batches") is not True:
            raise SystemExit(
                f"{path} sharded streaming_train_batches {sharded_global.get('streaming_train_batches')} is not true"
            )
        if int(sharded_global["data_train_samples"]) != int(data["dataset_train_samples"]):
            raise SystemExit(
                f"{path} sharded data_train_samples {sharded_global['data_train_samples']} does not match summary {data['dataset_train_samples']}"
            )
        if int(sharded_global["data_epoch_next"]) != int(data["data_epoch_next"]):
            raise SystemExit(
                f"{path} sharded data_epoch_next {sharded_global['data_epoch_next']} does not match summary {data['data_epoch_next']}"
            )
        if int(sharded_global["data_sample_offset_next"]) != int(data["data_sample_offset_next"]):
            raise SystemExit(
                f"{path} sharded data_sample_offset_next {sharded_global['data_sample_offset_next']} does not match summary {data['data_sample_offset_next']}"
            )
    if data["checkpoint_written"]:
        rank0_manifest_path = manifest_path
    evidence.append(
        {
            "rank": data["rank"],
            "dtype": data["dtype"],
            "dataset_order_seed": data.get("dataset_order_seed"),
            "dataset_total_samples": data.get("dataset_total_samples"),
            "dataset_source_files": data.get("dataset_source_files"),
            "dataset_source_sample_counts": data.get("dataset_source_sample_counts"),
            "dataset_fingerprint": data.get("dataset_fingerprint"),
            "streaming_train_batches": data.get("streaming_train_batches"),
            "data_cursor_next": data.get("data_cursor_next"),
            "data_epoch_next": data.get("data_epoch_next"),
            "data_sample_offset_next": data.get("data_sample_offset_next"),
            "sequence_tokens": data.get("sequence_tokens"),
            "trainable_tensors": len(trainable_tensors),
            "global_step_losses": global_step_losses,
            "reload_delta": data["reload_delta"],
            "next_step_delta": data["next_step_delta"],
            "sharded_reload_delta": data["sharded_reload_delta"],
            "sharded_next_step_delta": data["sharded_next_step_delta"],
        }
    )

if rank0_manifest_path is None:
    raise SystemExit("expected one rank0 checkpoint writer")

print(json.dumps({"qwen_session_dp2_layers01_verified": evidence}, indent=2, sort_keys=True))
PY
