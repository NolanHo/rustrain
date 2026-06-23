#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SESSION_DP2_SFT_ARROW_CONFIG:-configs/qwen_session_dp2_sft_arrow.toml}"
OUTPUT_DIR="${RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR:-/tmp/rustrain-runs/qwen-session-dp2-sft-arrow-$(date +%Y%m%d-%H%M%S)-$$}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  train --config "${CONFIG}"

python - "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from qwen_verify_utils import (
    require_complete_qwen_base_model_path,
    require_complete_qwen_manifest_paths,
)

output_dir = pathlib.Path(sys.argv[1])
rank_summaries = sorted(output_dir.rglob("qwen-session-dp-rank-*.json"))
if len(rank_summaries) != 2:
    raise SystemExit(f"expected 2 DP rank summaries under {output_dir}, found {len(rank_summaries)}")

evidence = []
for path in rank_summaries:
    data = json.loads(path.read_text())
    required = [
        "rank",
        "world_size",
        "dtype",
        "data_kind",
        "steps",
        "local_batch_size",
        "sequence_tokens",
        "dataset_total_samples",
        "dataset_train_samples",
        "dataset_eval_samples",
        "dataset_source_files",
        "dataset_source_sample_counts",
        "dataset_fingerprint",
        "dataset_order_seed",
        "dataset_shuffle",
        "streaming_train_batches",
        "streaming_index_cache_hit",
        "streaming_index_cache_written",
        "data_cursor_start",
        "data_cursor_end",
        "data_cursor_next",
        "data_epoch_start",
        "data_epoch_end",
        "data_epoch_next",
        "data_sample_offset_start",
        "data_sample_offset_end",
        "data_sample_offset_next",
        "reload_delta",
        "next_step_delta",
        "sharded_reload_delta",
        "sharded_next_step_delta",
        "sharded_global_manifest_output",
    ]
    missing = [key for key in required if data.get(key) is None]
    if missing:
        raise SystemExit(f"{path} is missing fields: {missing}")
    if data.get("dataset_total_tokens") is not None:
        raise SystemExit(
            f"{path} instruction_arrow trainer runtime must not report dataset_total_tokens; "
            "full token totals are reserved for materialized batch-plan parity"
        )
    if int(data["world_size"]) != 2:
        raise SystemExit(f"{path} expected world_size 2, got {data['world_size']}")
    if data["dtype"] != "fp32":
        raise SystemExit(f"{path} expected fp32 dtype, got {data['dtype']}")
    if data["data_kind"] != "instruction_arrow":
        raise SystemExit(f"{path} expected instruction_arrow data_kind, got {data['data_kind']}")
    if int(data["steps"]) != 1:
        raise SystemExit(f"{path} expected 1 step, got {data['steps']}")
    if int(data["local_batch_size"]) != 1:
        raise SystemExit(f"{path} expected local_batch_size 1, got {data['local_batch_size']}")
    if int(data["dataset_total_samples"]) != 32:
        raise SystemExit(f"{path} expected 32 Arrow samples, got {data['dataset_total_samples']}")
    if int(data["dataset_train_samples"]) != 24 or int(data["dataset_eval_samples"]) != 8:
        raise SystemExit(
            f"{path} expected 24/8 train/eval split, got {data['dataset_train_samples']}/{data['dataset_eval_samples']}"
        )
    dataset_source_files = data["dataset_source_files"]
    if len(dataset_source_files) != 1 or not dataset_source_files[0].endswith(".arrow"):
        raise SystemExit(f"{path} expected one Arrow source file, got {dataset_source_files}")
    source_counts = data["dataset_source_sample_counts"]
    if source_counts != [{"path": dataset_source_files[0], "samples": 32}]:
        raise SystemExit(f"{path} unexpected source counts: {source_counts}")
    if int(data["dataset_order_seed"]) != 777:
        raise SystemExit(f"{path} expected dataset_order_seed 777, got {data['dataset_order_seed']}")
    if data["dataset_shuffle"] is not False:
        raise SystemExit(f"{path} expected dataset_shuffle false, got {data['dataset_shuffle']}")
    if data["streaming_train_batches"] is not True:
        raise SystemExit(f"{path} expected streaming_train_batches true, got {data['streaming_train_batches']}")
    cache_path = data.get("streaming_index_cache_path")
    if cache_path:
        expected_suffix = f".rank-{data['rank']}.json"
        expected_dir = f"rank-{data['rank']}-cache"
        cache_parts = pathlib.Path(cache_path).parts
        if not str(cache_path).endswith(expected_suffix) and expected_dir not in cache_parts:
            raise SystemExit(
                f"{path} expected rank-local Arrow cache ending {expected_suffix} "
                f"or under {expected_dir}/, got {cache_path}"
            )
        if data["streaming_index_cache_hit"] is not False:
            raise SystemExit(f"{path} first Arrow DP cache read must be miss")
        if data["streaming_index_cache_written"] is not True:
            raise SystemExit(f"{path} first Arrow DP cache run must write the rank-local cache")
    elif data["streaming_index_cache_hit"] is not False or data["streaming_index_cache_written"] is not False:
        raise SystemExit(
            f"{path} Arrow DP cache fields are inconsistent without cache path: "
            f"hit={data['streaming_index_cache_hit']} written={data['streaming_index_cache_written']}"
        )
    expected_cursor_end = int(data["steps"]) * int(data["local_batch_size"]) * int(data["world_size"])
    if int(data["data_cursor_start"]) != 0:
        raise SystemExit(f"{path} expected data_cursor_start 0, got {data['data_cursor_start']}")
    if int(data["data_cursor_end"]) != expected_cursor_end:
        raise SystemExit(f"{path} expected data_cursor_end {expected_cursor_end}, got {data['data_cursor_end']}")
    if int(data["data_cursor_next"]) != expected_cursor_end:
        raise SystemExit(f"{path} expected data_cursor_next {expected_cursor_end}, got {data['data_cursor_next']}")
    train_samples = int(data["dataset_train_samples"])
    for cursor_key, epoch_key, offset_key in [
        ("data_cursor_start", "data_epoch_start", "data_sample_offset_start"),
        ("data_cursor_end", "data_epoch_end", "data_sample_offset_end"),
        ("data_cursor_next", "data_epoch_next", "data_sample_offset_next"),
    ]:
        cursor = int(data[cursor_key])
        if int(data[epoch_key]) != cursor // train_samples:
            raise SystemExit(f"{path} {epoch_key} mismatch for {cursor_key}={cursor}")
        if int(data[offset_key]) != cursor % train_samples:
            raise SystemExit(f"{path} {offset_key} mismatch for {cursor_key}={cursor}")
    for key in ["reload_delta", "next_step_delta", "sharded_reload_delta", "sharded_next_step_delta"]:
        if float(data[key]) > 1e-5:
            raise SystemExit(f"{path} {key} too large: {data[key]}")

    sharded_manifest_path = pathlib.Path(data["sharded_global_manifest_output"])
    sharded_manifest = json.loads(sharded_manifest_path.read_text())
    require_complete_qwen_manifest_paths(sharded_manifest, sharded_manifest_path)
    if sharded_manifest.get("dataset_source_files") != dataset_source_files:
        raise SystemExit(f"{path} sharded manifest dataset_source_files mismatch")
    if sharded_manifest.get("dataset_source_sample_counts") != source_counts:
        raise SystemExit(f"{path} sharded manifest dataset_source_sample_counts mismatch")
    if sharded_manifest.get("dataset_fingerprint") != data["dataset_fingerprint"]:
        raise SystemExit(f"{path} sharded manifest dataset_fingerprint mismatch")
    if sharded_manifest.get("streaming_train_batches") is not True:
        raise SystemExit(f"{path} sharded manifest streaming_train_batches is not true")
    if int(sharded_manifest["consumed_samples"]) != expected_cursor_end:
        raise SystemExit(f"{path} sharded consumed_samples mismatch: {sharded_manifest['consumed_samples']}")
    if int(sharded_manifest["data_cursor_next"]) != expected_cursor_end:
        raise SystemExit(f"{path} sharded data_cursor_next mismatch: {sharded_manifest['data_cursor_next']}")

    if data.get("checkpoint_written"):
        rank0_manifest_path = pathlib.Path(data["manifest_output"])
        rank0_manifest = json.loads(rank0_manifest_path.read_text())
        require_complete_qwen_base_model_path(rank0_manifest, rank0_manifest_path)
        if rank0_manifest.get("dataset_source_files") != dataset_source_files:
            raise SystemExit(f"{path} rank0 manifest dataset_source_files mismatch")
        if rank0_manifest.get("dataset_source_sample_counts") != source_counts:
            raise SystemExit(f"{path} rank0 manifest dataset_source_sample_counts mismatch")
        if rank0_manifest.get("dataset_fingerprint") != data["dataset_fingerprint"]:
            raise SystemExit(f"{path} rank0 manifest dataset_fingerprint mismatch")
        if rank0_manifest.get("streaming_train_batches") is not True:
            raise SystemExit(f"{path} rank0 manifest streaming_train_batches is not true")

    evidence.append(
        {
            "rank": data["rank"],
            "checkpoint_written": data.get("checkpoint_written"),
            "dataset_total_samples": data["dataset_total_samples"],
            "dataset_train_samples": data["dataset_train_samples"],
            "dataset_eval_samples": data["dataset_eval_samples"],
            "dataset_fingerprint": data["dataset_fingerprint"],
            "data_cursor_next": data["data_cursor_next"],
            "reload_delta": data["reload_delta"],
            "sharded_reload_delta": data["sharded_reload_delta"],
            "sharded_next_step_delta": data["sharded_next_step_delta"],
            "streaming_index_cache_path": cache_path,
            "streaming_index_cache_written": data["streaming_index_cache_written"],
        }
    )

print(json.dumps({"qwen_session_dp2_sft_arrow_verified": evidence}, indent=2, sort_keys=True))
PY
