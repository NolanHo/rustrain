#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

BASE_CONFIG="${RUSTRAIN_QWEN_SESSION_DP2_SFT_ARROW_UNBOUNDED_CONFIG:-configs/qwen_session_dp2_sft_arrow.toml}"
RUN_DIR="$(mktemp -d /tmp/rustrain-dp2-arrow-unbounded-cache-XXXXXX)"
ARROW_PATH="${RUN_DIR}/train.arrow"
CONFIG="${RUN_DIR}/config.toml"
FIRST_OUTPUT_DIR="${RUN_DIR}/first"
SECOND_OUTPUT_DIR="${RUN_DIR}/second"
CACHE_PATH="${RUN_DIR}/arrow-row-index.json"
MASTER_PORT="${RUSTRAIN_QWEN_SESSION_DP2_SFT_ARROW_UNBOUNDED_PORT:-29631}"

python - "${ARROW_PATH}" <<'PY'
import pathlib
import sys

import pyarrow as pa
import pyarrow.ipc as ipc

path = pathlib.Path(sys.argv[1])
row_count = 64
schema = pa.schema(
    [
        ("instruction", pa.string()),
        ("input", pa.string()),
        ("output", pa.string()),
    ]
)
batch = pa.record_batch(
    [
        pa.array([f"question {index}" for index in range(row_count)], type=pa.string()),
        pa.array([f"context {index}" for index in range(row_count)], type=pa.string()),
        pa.array([f"answer {index}" for index in range(row_count)], type=pa.string()),
    ],
    schema=schema,
)
with ipc.new_stream(path, schema) as writer:
    writer.write_batch(batch)
PY

python - "${BASE_CONFIG}" "${CONFIG}" "${ARROW_PATH}" "${CACHE_PATH}" <<'PY'
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
arrow_path = pathlib.Path(sys.argv[3]).as_posix()
cache_path = pathlib.Path(sys.argv[4]).as_posix()
lines = []
in_data = False
for line in source.read_text().splitlines():
    if line.strip() == "[data]":
        in_data = True
        lines.append(line)
        continue
    if line.startswith("[") and line.strip() != "[data]":
        in_data = False
    if line.startswith("max_samples = "):
        continue
    if in_data and line.startswith("paths = "):
        lines.append(f'paths = ["{arrow_path}"]')
        lines.append(f'index_cache = "{cache_path}"')
        continue
    lines.append(line)
target.write_text("\n".join(lines) + "\n")
PY

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${FIRST_OUTPUT_DIR}" \
  --master-port "${MASTER_PORT}" \
  train --config "${CONFIG}"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${SECOND_OUTPUT_DIR}" \
  --master-port "${MASTER_PORT}" \
  train --config "${CONFIG}"

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
cache_base = pathlib.Path(sys.argv[3])

def rank_cache(rank: int) -> pathlib.Path:
    return cache_base.with_name(f"{cache_base.stem}.rank-{rank}{cache_base.suffix}")

def summaries(output_dir: pathlib.Path) -> dict[int, tuple[pathlib.Path, dict]]:
    paths = sorted(output_dir.rglob("qwen-session-dp-rank-*.json"))
    if len(paths) != 2:
        raise SystemExit(f"expected 2 DP rank summaries under {output_dir}, found {len(paths)}")
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

    for context, item, expected_hit, expected_written in [
        ("first", first[rank], False, True),
        ("second", second[rank], True, False),
    ]:
        path, data = item
        required = [
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
            raise SystemExit(f"{path} instruction_arrow trainer runtime must not report dataset_total_tokens")
        if data["data_kind"] != "instruction_arrow":
            raise SystemExit(f"{path} expected instruction_arrow, got {data['data_kind']}")
        if int(data["world_size"]) != 2:
            raise SystemExit(f"{path} expected world_size 2, got {data['world_size']}")
        if int(data["steps"]) != 1 or int(data["local_batch_size"]) != 1:
            raise SystemExit(f"{path} expected one DP step with local batch 1, got steps={data['steps']} local={data['local_batch_size']}")
        total_samples = int(data["dataset_total_samples"])
        train_samples = int(data["dataset_train_samples"])
        eval_samples = int(data["dataset_eval_samples"])
        if total_samples != 64:
            raise SystemExit(f"{path} expected 64 unbounded fixture samples, got {total_samples}")
        if train_samples + eval_samples != total_samples:
            raise SystemExit(f"{path} train/eval counts do not sum to total: {train_samples}+{eval_samples}!={total_samples}")
        dataset_source_files = data["dataset_source_files"]
        if len(dataset_source_files) != 1 or not dataset_source_files[0].endswith(".arrow"):
            raise SystemExit(f"{path} expected one Arrow source file, got {dataset_source_files}")
        if data["dataset_source_sample_counts"] != [{"path": dataset_source_files[0], "samples": total_samples}]:
            raise SystemExit(f"{path} unexpected source counts: {data['dataset_source_sample_counts']}")
        if data["streaming_train_batches"] is not True:
            raise SystemExit(f"{path} expected streaming_train_batches true")
        if data["streaming_index_cache_hit"] is not expected_hit:
            raise SystemExit(f"{path} expected cache_hit={expected_hit}, got {data['streaming_index_cache_hit']}")
        if data["streaming_index_cache_written"] is not expected_written:
            raise SystemExit(f"{path} expected cache_written={expected_written}, got {data['streaming_index_cache_written']}")
        if pathlib.Path(data["streaming_index_cache_path"]) != expected_cache:
            raise SystemExit(f"{path} expected rank-local cache {expected_cache}, got {data['streaming_index_cache_path']}")

        cache = json.loads(expected_cache.read_text())
        if cache.get("format") != "rustrain.qwen_sft_arrow_row_index.v1":
            raise SystemExit(f"{path} unexpected cache format: {cache.get('format')}")
        if cache.get("max_samples") is not None:
            raise SystemExit(f"{path} unbounded Arrow cache should record max_samples null, got {cache.get('max_samples')}")
        if cache.get("summary", {}).get("samples") != total_samples:
            raise SystemExit(f"{path} cache summary samples mismatch: {cache.get('summary')}")
        if len(cache.get("samples", [])) != total_samples:
            raise SystemExit(f"{path} expected {total_samples} cached row indices, got {len(cache.get('samples', []))}")

        expected_cursor_end = int(data["steps"]) * int(data["local_batch_size"]) * int(data["world_size"])
        if int(data["data_cursor_start"]) != 0:
            raise SystemExit(f"{path} expected data_cursor_start 0, got {data['data_cursor_start']}")
        if int(data["data_cursor_end"]) != expected_cursor_end:
            raise SystemExit(f"{path} expected data_cursor_end {expected_cursor_end}, got {data['data_cursor_end']}")
        if int(data["data_cursor_next"]) != expected_cursor_end:
            raise SystemExit(f"{path} expected data_cursor_next {expected_cursor_end}, got {data['data_cursor_next']}")
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
        if sharded_manifest.get("dataset_source_sample_counts") != data["dataset_source_sample_counts"]:
            raise SystemExit(f"{path} sharded manifest dataset_source_sample_counts mismatch")
        if sharded_manifest.get("dataset_fingerprint") != data["dataset_fingerprint"]:
            raise SystemExit(f"{path} sharded manifest dataset_fingerprint mismatch")
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
            if rank0_manifest.get("dataset_fingerprint") != data["dataset_fingerprint"]:
                raise SystemExit(f"{path} rank0 manifest dataset_fingerprint mismatch")
            if rank0_manifest.get("streaming_train_batches") is not True:
                raise SystemExit(f"{path} rank0 manifest streaming_train_batches is not true")

    evidence.append(
        {
            "rank": rank,
            "cache": str(expected_cache),
            "first_written": first[rank][1]["streaming_index_cache_written"],
            "second_hit": second[rank][1]["streaming_index_cache_hit"],
            "dataset_total_samples": second[rank][1]["dataset_total_samples"],
            "data_cursor_next": second[rank][1]["data_cursor_next"],
            "sharded_reload_delta": second[rank][1]["sharded_reload_delta"],
            "sharded_next_step_delta": second[rank][1]["sharded_next_step_delta"],
        }
    )

print(json.dumps({"qwen_session_dp2_sft_arrow_unbounded_cache_verified": evidence}, indent=2, sort_keys=True))
PY
