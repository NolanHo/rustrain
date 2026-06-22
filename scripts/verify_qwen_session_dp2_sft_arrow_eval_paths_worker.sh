#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SESSION_DP2_SFT_ARROW_EVAL_PATHS_CONFIG:-configs/qwen_session_dp2_sft_arrow_eval_paths.toml}"
OUTPUT_DIR="${RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR:-/tmp/rustrain-runs/qwen-session-dp2-sft-arrow-eval-paths-$(date +%Y%m%d-%H%M%S)-$$}"
DATA_PLAN_OUTPUT="$(mktemp)"
BATCH_PLAN_OUTPUT="$(mktemp)"

cargo run -- qwen-sft-streaming-data-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 0 \
  | tee "${DATA_PLAN_OUTPUT}"

cargo run -- qwen-sft-streaming-batch-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 0 \
  | tee "${BATCH_PLAN_OUTPUT}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${OUTPUT_DIR}" \
  train --config "${CONFIG}"

python - "${DATA_PLAN_OUTPUT}" "${BATCH_PLAN_OUTPUT}" "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from qwen_verify_utils import (
    require_complete_qwen_base_model_path,
    require_complete_qwen_manifest_paths,
)

def load_json_output(path_text):
    text = pathlib.Path(path_text).read_text()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise SystemExit(f"{path_text} did not contain JSON: {text}")
    return json.loads(text[start : end + 1])

data_plan = load_json_output(sys.argv[1])
batch_plan = load_json_output(sys.argv[2])
output_dir = pathlib.Path(sys.argv[3])

expected_path = data_plan["data_paths"][0]
if data_plan["eval_paths"] != [expected_path]:
    raise SystemExit(f"expected eval_paths to mirror train Arrow path, got {data_plan['eval_paths']}")
if data_plan["max_samples"] != 24 or data_plan["max_eval_samples"] != 8:
    raise SystemExit(
        f"expected max/max_eval samples 24/8, got {data_plan['max_samples']}/{data_plan['max_eval_samples']}"
    )
expected_source_files = [expected_path]
expected_source_counts = [{"path": expected_path, "samples": 32}]
expected_train_counts = [{"path": expected_path, "samples": 24}]
expected_eval_counts = [{"path": expected_path, "samples": 8}]
if data_plan["dataset_total_samples"] != 32:
    raise SystemExit(f"expected total samples 32, got {data_plan['dataset_total_samples']}")
if data_plan["dataset_train_samples"] != 24 or data_plan["dataset_eval_samples"] != 8:
    raise SystemExit(
        f"expected 24/8 train/eval samples, got {data_plan['dataset_train_samples']}/{data_plan['dataset_eval_samples']}"
    )
if data_plan["dataset_source_files"] != expected_source_files:
    raise SystemExit(f"unexpected dataset_source_files: {data_plan['dataset_source_files']}")
if data_plan["dataset_source_sample_counts"] != expected_source_counts:
    raise SystemExit(f"unexpected dataset_source_sample_counts: {data_plan['dataset_source_sample_counts']}")
if data_plan["train_source_sample_counts"] != expected_train_counts:
    raise SystemExit(f"unexpected train source counts: {data_plan['train_source_sample_counts']}")
if data_plan["eval_source_sample_counts"] != expected_eval_counts:
    raise SystemExit(f"unexpected eval source counts: {data_plan['eval_source_sample_counts']}")
if not data_plan["train_fingerprint"] or not data_plan["eval_fingerprint"]:
    raise SystemExit("train/eval fingerprints must be present")
if data_plan["dataset_fingerprint"] in [data_plan["train_fingerprint"], data_plan["eval_fingerprint"]]:
    raise SystemExit("combined fingerprint must differ from train/eval fingerprints")
if data_plan["streaming_index_cache_path"] is not None:
    raise SystemExit("instruction_arrow data plan must not report index cache path")

for key in [
    "dataset_total_samples",
    "dataset_train_samples",
    "dataset_eval_samples",
    "dataset_source_files",
    "dataset_source_sample_counts",
    "dataset_fingerprint",
    "dataset_order_seed",
    "dataset_shuffle",
]:
    if batch_plan[key] != data_plan[key]:
        raise SystemExit(f"batch-plan {key} mismatch: {batch_plan[key]} vs {data_plan[key]}")
if batch_plan["streaming_window_samples"] != 4:
    raise SystemExit(f"expected 4 streaming window samples, got {batch_plan['streaming_window_samples']}")
if batch_plan["streaming_raw_sample_indices"] != []:
    raise SystemExit(f"Arrow batch plan must not report JSONL raw indices, got {batch_plan['streaming_raw_sample_indices']}")
if batch_plan["streaming_index_cache_path"] is not None:
    raise SystemExit("instruction_arrow batch plan must not report index cache path")
if batch_plan["materialized_input_max_delta"] != 0 or float(batch_plan["materialized_mask_max_delta"]) != 0.0:
    raise SystemExit(
        f"batch plan materialized deltas must be zero, got input={batch_plan['materialized_input_max_delta']} mask={batch_plan['materialized_mask_max_delta']}"
    )
if len(batch_plan["batch_token_fingerprints"]) != 2:
    raise SystemExit(f"expected 2 batch token fingerprints, got {batch_plan['batch_token_fingerprints']}")

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
        raise SystemExit(f"{path} expected dataset_total_samples 32, got {data['dataset_total_samples']}")
    if int(data["dataset_train_samples"]) != 24 or int(data["dataset_eval_samples"]) != 8:
        raise SystemExit(
            f"{path} expected 24/8 train/eval samples, got {data['dataset_train_samples']}/{data['dataset_eval_samples']}"
        )
    if data["dataset_source_files"] != expected_source_files:
        raise SystemExit(f"{path} dataset_source_files mismatch: {data['dataset_source_files']}")
    if data["dataset_source_sample_counts"] != expected_source_counts:
        raise SystemExit(f"{path} dataset_source_sample_counts mismatch: {data['dataset_source_sample_counts']}")
    if data["dataset_fingerprint"] != data_plan["dataset_fingerprint"]:
        raise SystemExit(f"{path} dataset_fingerprint mismatch")
    if int(data["dataset_order_seed"]) != 777:
        raise SystemExit(f"{path} expected dataset_order_seed 777, got {data['dataset_order_seed']}")
    if data["dataset_shuffle"] is not False:
        raise SystemExit(f"{path} expected dataset_shuffle false, got {data['dataset_shuffle']}")
    if data["streaming_train_batches"] is not True:
        raise SystemExit(f"{path} expected streaming_train_batches true, got {data['streaming_train_batches']}")
    if data["streaming_index_cache_hit"] is not False or data["streaming_index_cache_written"] is not False:
        raise SystemExit(
            f"{path} Arrow DP path must not use index cache, got hit={data['streaming_index_cache_hit']} written={data['streaming_index_cache_written']}"
        )
    if data.get("streaming_index_cache_path"):
        raise SystemExit(f"{path} Arrow DP path must not report index cache path")
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
    if sharded_manifest.get("dataset_source_files") != expected_source_files:
        raise SystemExit(f"{path} sharded manifest dataset_source_files mismatch")
    if sharded_manifest.get("dataset_source_sample_counts") != expected_source_counts:
        raise SystemExit(f"{path} sharded manifest dataset_source_sample_counts mismatch")
    if sharded_manifest.get("dataset_fingerprint") != data_plan["dataset_fingerprint"]:
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
        if rank0_manifest.get("dataset_source_files") != expected_source_files:
            raise SystemExit(f"{path} rank0 manifest dataset_source_files mismatch")
        if rank0_manifest.get("dataset_source_sample_counts") != expected_source_counts:
            raise SystemExit(f"{path} rank0 manifest dataset_source_sample_counts mismatch")
        if rank0_manifest.get("dataset_fingerprint") != data_plan["dataset_fingerprint"]:
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
        }
    )

print(json.dumps({"qwen_session_dp2_sft_arrow_eval_paths_verified": evidence}, indent=2, sort_keys=True))
PY
