#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ray/require_gpu_worker.sh"

BASE_CONFIG="${RUSTRAIN_QWEN_SESSION_DP2_SFT_ARROW_INDEX_CACHE_CONFIG:-configs/qwen_session_dp2_sft_arrow_index_cache.toml}"
RUN_DIR="$(mktemp -d /tmp/rustrain-dp2-arrow-cache-verify-XXXXXX)"
CONFIG="${RUN_DIR}/config.toml"
CACHE_PATH="${RUN_DIR}/arrow-row-index.json"
FIRST_OUTPUT_DIR="${RUN_DIR}/first"
SECOND_OUTPUT_DIR="${RUN_DIR}/second"

python - "${BASE_CONFIG}" "${CONFIG}" "${CACHE_PATH}" <<'PY'
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
cache = pathlib.Path(sys.argv[3]).as_posix()
lines = []
for line in text.splitlines():
    if line.startswith("index_cache = "):
        lines.append(f'index_cache = "{cache}"')
    else:
        lines.append(line)
pathlib.Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
PY

cargo run -- launch --nproc-per-node 2 --output-dir "${FIRST_OUTPUT_DIR}" train --config "${CONFIG}"
cargo run -- launch --nproc-per-node 2 --output-dir "${SECOND_OUTPUT_DIR}" train --config "${CONFIG}"

python - "${FIRST_OUTPUT_DIR}" "${SECOND_OUTPUT_DIR}" "${CACHE_PATH}" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from qwen_verify_utils import (
    require_complete_qwen_base_model_path,
    require_complete_qwen_manifest_paths,
)

first_dir = pathlib.Path(sys.argv[1])
second_dir = pathlib.Path(sys.argv[2])
cache_path = pathlib.Path(sys.argv[3])

def rank_cache(rank):
    return cache_path.with_name(f"{cache_path.stem}.rank-{rank}{cache_path.suffix}")

def summaries(output_dir):
    paths = sorted(output_dir.rglob("qwen-session-dp-rank-*.json"))
    if len(paths) != 2:
        raise SystemExit(f"expected 2 rank summaries under {output_dir}, found {len(paths)}")
    parsed = {}
    for path in paths:
        data = json.loads(path.read_text())
        parsed[int(data["rank"])] = (path, data)
    return parsed

first = summaries(first_dir)
second = summaries(second_dir)
evidence = []

for rank in [0, 1]:
    expected_cache = rank_cache(rank)
    if not expected_cache.exists():
        raise SystemExit(f"rank {rank} cache was not written: {expected_cache}")
    cache = json.loads(expected_cache.read_text())
    if cache.get("format") != "rustrain.qwen_sft_arrow_row_index.v1":
        raise SystemExit(f"rank {rank} unexpected cache format: {cache.get('format')}")
    if cache.get("summary", {}).get("samples") != 32:
        raise SystemExit(f"rank {rank} expected 32 cached samples, got {cache.get('summary')}")
    if len(cache.get("samples", [])) != 32:
        raise SystemExit(f"rank {rank} expected 32 cached row indices, got {len(cache.get('samples', []))}")

    for context, item, expected_hit, expected_written in [
        ("first", first[rank], False, True),
        ("second", second[rank], True, False),
    ]:
        path, data = item
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
            "streaming_index_cache_path",
            "streaming_index_cache_hit",
            "streaming_index_cache_written",
            "data_cursor_start",
            "data_cursor_end",
            "data_cursor_next",
            "reload_delta",
            "next_step_delta",
            "sharded_reload_delta",
            "sharded_next_step_delta",
            "sharded_global_manifest_output",
        ]
        missing = [key for key in required if data.get(key) is None]
        if missing:
            raise SystemExit(f"{path} is missing fields: {missing}")
        if data["data_kind"] != "instruction_arrow":
            raise SystemExit(f"{path} expected instruction_arrow, got {data['data_kind']}")
        if int(data["world_size"]) != 2:
            raise SystemExit(f"{path} expected world_size 2, got {data['world_size']}")
        if int(data["dataset_total_samples"]) != 32:
            raise SystemExit(f"{path} expected 32 samples, got {data['dataset_total_samples']}")
        if int(data["dataset_train_samples"]) != 24 or int(data["dataset_eval_samples"]) != 8:
            raise SystemExit(
                f"{path} expected 24/8 train/eval split, got {data['dataset_train_samples']}/{data['dataset_eval_samples']}"
            )
        dataset_source_files = data["dataset_source_files"]
        if len(dataset_source_files) != 1 or not dataset_source_files[0].endswith(".arrow"):
            raise SystemExit(f"{path} expected one Arrow source file, got {dataset_source_files}")
        if data["dataset_source_sample_counts"] != [{"path": dataset_source_files[0], "samples": 32}]:
            raise SystemExit(f"{path} unexpected source counts: {data['dataset_source_sample_counts']}")
        if data["dataset_shuffle"] is not False:
            raise SystemExit(f"{path} expected dataset_shuffle false, got {data['dataset_shuffle']}")
        if data["streaming_train_batches"] is not True:
            raise SystemExit(f"{path} expected streaming_train_batches true")
        if pathlib.Path(data["streaming_index_cache_path"]) != expected_cache:
            raise SystemExit(
                f"{context} rank {rank} cache path {data['streaming_index_cache_path']} != {expected_cache}"
            )
        if data["streaming_index_cache_hit"] is not expected_hit:
            raise SystemExit(
                f"{context} rank {rank} expected cache_hit={expected_hit}, got {data['streaming_index_cache_hit']}"
            )
        if data["streaming_index_cache_written"] is not expected_written:
            raise SystemExit(
                f"{context} rank {rank} expected cache_written={expected_written}, got {data['streaming_index_cache_written']}"
            )
        expected_cursor_end = int(data["steps"]) * int(data["local_batch_size"]) * int(data["world_size"])
        if int(data["data_cursor_start"]) != 0:
            raise SystemExit(f"{path} expected cursor start 0, got {data['data_cursor_start']}")
        if int(data["data_cursor_end"]) != expected_cursor_end:
            raise SystemExit(f"{path} expected cursor end {expected_cursor_end}, got {data['data_cursor_end']}")
        if int(data["data_cursor_next"]) != expected_cursor_end:
            raise SystemExit(f"{path} expected cursor next {expected_cursor_end}, got {data['data_cursor_next']}")
        for key in ["reload_delta", "next_step_delta", "sharded_reload_delta", "sharded_next_step_delta"]:
            if float(data[key]) > 1e-5:
                raise SystemExit(f"{path} {key} too large: {data[key]}")

        sharded_manifest_path = pathlib.Path(data["sharded_global_manifest_output"])
        sharded_manifest = json.loads(sharded_manifest_path.read_text())
        require_complete_qwen_manifest_paths(sharded_manifest, sharded_manifest_path)
        if sharded_manifest.get("dataset_source_files") != dataset_source_files:
            raise SystemExit(f"{path} sharded manifest dataset_source_files mismatch")
        if sharded_manifest.get("dataset_source_sample_counts") != data["dataset_source_sample_counts"]:
            raise SystemExit(f"{path} sharded manifest dataset_source_sample_counts mismatch")
        if sharded_manifest.get("streaming_train_batches") is not True:
            raise SystemExit(f"{path} sharded manifest streaming_train_batches is not true")
        if int(sharded_manifest["data_cursor_next"]) != expected_cursor_end:
            raise SystemExit(f"{path} sharded data_cursor_next mismatch: {sharded_manifest['data_cursor_next']}")

        if data.get("checkpoint_written"):
            rank0_manifest_path = pathlib.Path(data["manifest_output"])
            rank0_manifest = json.loads(rank0_manifest_path.read_text())
            require_complete_qwen_base_model_path(rank0_manifest, rank0_manifest_path)
            if rank0_manifest.get("dataset_source_files") != dataset_source_files:
                raise SystemExit(f"{path} rank0 manifest dataset_source_files mismatch")
            if rank0_manifest.get("dataset_source_sample_counts") != data["dataset_source_sample_counts"]:
                raise SystemExit(f"{path} rank0 manifest dataset_source_sample_counts mismatch")
            if rank0_manifest.get("streaming_train_batches") is not True:
                raise SystemExit(f"{path} rank0 streaming_train_batches is not true")

    evidence.append(
        {
            "rank": rank,
            "cache": str(expected_cache),
            "first_written": first[rank][1]["streaming_index_cache_written"],
            "second_hit": second[rank][1]["streaming_index_cache_hit"],
        }
    )

print(json.dumps({"qwen_session_dp2_sft_arrow_index_cache_verified": evidence}, indent=2, sort_keys=True))
PY
