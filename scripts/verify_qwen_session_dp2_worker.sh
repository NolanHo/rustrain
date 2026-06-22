#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

OUTPUT_DIR="${RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR:?RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR is required}"
CONFIG="${RUSTRAIN_QWEN_SESSION_DP_CONFIG:-configs/qwen_session_dp2.toml}"
EXPECTED_DTYPE="${RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND:-}"
EXPECTED_DATASET_SEED="${RUSTRAIN_EXPECTED_DATASET_ORDER_SEED:-}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  train --config "${CONFIG}"

python - <<'PY'
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from qwen_verify_utils import require_complete_qwen_manifest_paths

output_dir = pathlib.Path(os.environ["RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR"])
expected_dtype = os.environ.get("RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND")
expected_dataset_seed = os.environ.get("RUSTRAIN_EXPECTED_DATASET_ORDER_SEED")
rank_summaries = sorted(output_dir.rglob("qwen-session-dp-rank-*.json"))
if len(rank_summaries) != 2:
    raise SystemExit(
        f"expected 2 qwen session DP rank summaries under {output_dir}, found {len(rank_summaries)}"
    )

evidence = []
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
    sharded_reload_delta = data.get("sharded_reload_delta")
    if sharded_reload_delta is None:
        raise SystemExit(f"{path} is missing sharded_reload_delta")
    if sharded_reload_delta > 1e-5:
        raise SystemExit(
            f"{path} sharded_reload_delta {sharded_reload_delta} exceeds tolerance"
        )
    sharded_next_step_delta = data.get("sharded_next_step_delta")
    if sharded_next_step_delta is None:
        raise SystemExit(f"{path} is missing sharded_next_step_delta")
    if sharded_next_step_delta > 1e-5:
        raise SystemExit(
            f"{path} sharded_next_step_delta {sharded_next_step_delta} exceeds tolerance"
        )
    sharded_manifest_path = pathlib.Path(data["sharded_global_manifest_output"])
    sharded_manifest = json.loads(sharded_manifest_path.read_text())
    require_complete_qwen_manifest_paths(sharded_manifest, sharded_manifest_path)
    if expected_dtype and data.get("dtype") != expected_dtype:
        raise SystemExit(
            f"{path} dtype {data.get('dtype')} does not match expected {expected_dtype}"
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
        if sharded_manifest.get("dataset_source_files") != dataset_source_files:
            raise SystemExit(
                f"{path} sharded dataset_source_files {sharded_manifest.get('dataset_source_files')} does not match summary {dataset_source_files}"
            )
        if sharded_manifest.get("dataset_source_sample_counts") != dataset_source_sample_counts:
            raise SystemExit(
                f"{path} sharded dataset_source_sample_counts {sharded_manifest.get('dataset_source_sample_counts')} does not match summary {dataset_source_sample_counts}"
            )
        verify_source_sample_counts(
            sharded_manifest.get("dataset_source_sample_counts"),
            dataset_source_files,
            data["dataset_total_samples"],
            f"{path} sharded manifest",
        )
        if sharded_manifest.get("dataset_fingerprint") != data["dataset_fingerprint"]:
            raise SystemExit(
                f"{path} sharded dataset_fingerprint {sharded_manifest.get('dataset_fingerprint')} does not match summary {data['dataset_fingerprint']}"
            )
        if int(sharded_manifest["consumed_samples"]) != expected_cursor_end:
            raise SystemExit(
                f"{path} sharded consumed_samples {sharded_manifest['consumed_samples']} does not match expected {expected_cursor_end}"
            )
        if int(sharded_manifest["data_cursor_next"]) != expected_cursor_end:
            raise SystemExit(
                f"{path} sharded data_cursor_next {sharded_manifest['data_cursor_next']} does not match expected {expected_cursor_end}"
            )
        if sharded_manifest.get("streaming_train_batches") is not True:
            raise SystemExit(
                f"{path} sharded streaming_train_batches {sharded_manifest.get('streaming_train_batches')} is not true"
            )
        if int(sharded_manifest["data_train_samples"]) != train_samples:
            raise SystemExit(
                f"{path} sharded data_train_samples {sharded_manifest['data_train_samples']} does not match expected {train_samples}"
            )
        if int(sharded_manifest["data_epoch_next"]) != expected_cursor_end // train_samples:
            raise SystemExit(
                f"{path} sharded data_epoch_next {sharded_manifest['data_epoch_next']} does not match rank summary"
            )
        if int(sharded_manifest["data_sample_offset_next"]) != expected_cursor_end % train_samples:
            raise SystemExit(
                f"{path} sharded data_sample_offset_next {sharded_manifest['data_sample_offset_next']} does not match rank summary"
            )
        if data["checkpoint_written"]:
            rank0_manifest = json.loads(pathlib.Path(data["manifest_output"]).read_text())
            if rank0_manifest.get("dataset_source_files") != dataset_source_files:
                raise SystemExit(
                    f"{path} rank0 manifest dataset_source_files {rank0_manifest.get('dataset_source_files')} does not match summary {dataset_source_files}"
                )
            if rank0_manifest.get("dataset_source_sample_counts") != dataset_source_sample_counts:
                raise SystemExit(
                    f"{path} rank0 manifest dataset_source_sample_counts {rank0_manifest.get('dataset_source_sample_counts')} does not match summary {dataset_source_sample_counts}"
                )
            verify_source_sample_counts(
                rank0_manifest.get("dataset_source_sample_counts"),
                dataset_source_files,
                data["dataset_total_samples"],
                f"{path} rank0 manifest",
            )
            if rank0_manifest.get("dataset_fingerprint") != data["dataset_fingerprint"]:
                raise SystemExit(
                    f"{path} rank0 manifest dataset_fingerprint {rank0_manifest.get('dataset_fingerprint')} does not match summary {data['dataset_fingerprint']}"
                )
            if rank0_manifest.get("streaming_train_batches") is not True:
                raise SystemExit(
                    f"{path} rank0 manifest streaming_train_batches {rank0_manifest.get('streaming_train_batches')} is not true"
                )
    evidence.append(
        {
            "rank": data["rank"],
            "dtype": data.get("dtype"),
            "checkpoint_written": data["checkpoint_written"],
            "dataset_order_seed": data.get("dataset_order_seed"),
            "dataset_total_samples": data.get("dataset_total_samples"),
            "dataset_source_files": data.get("dataset_source_files"),
            "dataset_source_sample_counts": data.get("dataset_source_sample_counts"),
            "dataset_fingerprint": data.get("dataset_fingerprint"),
            "streaming_train_batches": data.get("streaming_train_batches"),
            "data_cursor_start": data.get("data_cursor_start"),
            "data_cursor_next": data.get("data_cursor_next"),
            "data_epoch_next": data.get("data_epoch_next"),
            "data_sample_offset_next": data.get("data_sample_offset_next"),
            "sequence_tokens": data.get("sequence_tokens"),
            "reload_delta": data["reload_delta"],
            "sharded_reload_delta": sharded_reload_delta,
            "sharded_next_step_delta": sharded_next_step_delta,
            "sharded_global_manifest_output": data["sharded_global_manifest_output"],
        }
    )

print(json.dumps({"qwen_session_dp_sharded_restore": evidence}, indent=2, sort_keys=True))
PY
