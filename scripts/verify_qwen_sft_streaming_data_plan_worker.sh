#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SFT_STREAMING_DATA_PLAN_CONFIG:-configs/qwen_session_dp2_sft_max_samples.toml}"
EXPECTED_SOURCE="${RUSTRAIN_EXPECTED_STREAMING_SOURCE:-data/sft_toy/instructions.jsonl}"
EXPECTED_SECOND_SOURCE="${RUSTRAIN_EXPECTED_STREAMING_SECOND_SOURCE:-data/sft_toy/more_instructions.jsonl}"
EXPECTED_FINGERPRINT="${RUSTRAIN_EXPECTED_STREAMING_FINGERPRINT:-3bfd266239e4b9b9}"
WORK_DIR="$(mktemp -d /tmp/rustrain-sft-data-plan-XXXXXX)"
trap 'rm -rf "${WORK_DIR}"' EXIT
CONFIG_WITH_CACHE="${WORK_DIR}/data-plan-index-cache.toml"
CACHE_PATH="${WORK_DIR}/offset-index.json"
UNCACHED_OUTPUT="${WORK_DIR}/uncached.out"
CACHE_FIRST_OUTPUT="${WORK_DIR}/cache-first.out"
CACHE_SECOND_OUTPUT="${WORK_DIR}/cache-second.out"

cargo run -- qwen sft-streaming-data-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 2 \
  | tee "${UNCACHED_OUTPUT}"

python - "${CONFIG}" "${CONFIG_WITH_CACHE}" "${CACHE_PATH}" <<'PY'
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
cache = pathlib.Path(sys.argv[3])
text = source.read_text(encoding="utf-8")
needle = "[data]\n"
if needle not in text:
    raise SystemExit(f"{source} does not contain [data] section")
target.write_text(text.replace(needle, needle + f'index_cache = "{cache}"\n', 1), encoding="utf-8")
PY

cargo run -- qwen sft-streaming-data-plan \
  --config "${CONFIG_WITH_CACHE}" \
  --world-size 2 \
  --data-cursor-start 2 \
  | tee "${CACHE_FIRST_OUTPUT}"

cargo run -- qwen sft-streaming-data-plan \
  --config "${CONFIG_WITH_CACHE}" \
  --world-size 2 \
  --data-cursor-start 2 \
  | tee "${CACHE_SECOND_OUTPUT}"

python - "${UNCACHED_OUTPUT}" "${CACHE_FIRST_OUTPUT}" "${CACHE_SECOND_OUTPUT}" "${CACHE_PATH}" "${EXPECTED_SOURCE}" "${EXPECTED_SECOND_SOURCE}" "${EXPECTED_FINGERPRINT}" <<'PY'
import json
import pathlib
import sys

uncached_output = pathlib.Path(sys.argv[1])
cache_first_output = pathlib.Path(sys.argv[2])
cache_second_output = pathlib.Path(sys.argv[3])
cache_path = pathlib.Path(sys.argv[4])
expected_source = sys.argv[5]
expected_second_source = sys.argv[6]
expected_fingerprint = sys.argv[7]

def load_json(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8")
    start = text.find("{")
    if start < 0:
        raise SystemExit(f"streaming data plan output did not contain JSON: {text}")
    return json.loads(text[start:])

uncached = load_json(uncached_output)
cache_first = load_json(cache_first_output)
cache_second = load_json(cache_second_output)

checks = {
    "max_samples": 4,
    "world_size": 2,
    "local_batch_size": 1,
    "global_batch_size": 2,
    "train_steps": 1,
    "required_batches": 3,
    "data_cursor_start": 2,
    "data_cursor_end": 4,
    "data_cursor_next": 4,
    "data_epoch_start": 0,
    "data_epoch_end": 1,
    "data_epoch_next": 1,
    "data_sample_offset_start": 2,
    "data_sample_offset_end": 1,
    "data_sample_offset_next": 1,
    "train_window_start_cursor": 2,
    "train_window_end_cursor_exclusive": 6,
    "dataset_total_samples": 4,
    "dataset_train_samples": 3,
    "dataset_eval_samples": 1,
    "dataset_order_seed": 777,
    "dataset_shuffle": True,
    "tokenizer_loaded": False,
    "tokenized_samples_materialized": False,
}

expected_window = [
    {"cursor": 2, "epoch": 0, "sample_offset": 2},
    {"cursor": 3, "epoch": 1, "sample_offset": 0},
    {"cursor": 4, "epoch": 1, "sample_offset": 1},
    {"cursor": 5, "epoch": 1, "sample_offset": 2},
]
expected_counts = [{"path": expected_source, "samples": 4}]

def verify_plan(data: dict, context: str) -> None:
    for key, expected in checks.items():
        if data.get(key) != expected:
            raise SystemExit(f"{context}: {key} {data.get(key)} != {expected}")
    if data.get("data_paths") != [expected_source, expected_second_source]:
        raise SystemExit(
            f"{context}: data_paths {data.get('data_paths')} != {[expected_source, expected_second_source]}"
        )
    if data.get("eval_paths") != []:
        raise SystemExit(f"{context}: eval_paths should be empty, got {data.get('eval_paths')}")
    if data.get("max_eval_samples") is not None:
        raise SystemExit(f"{context}: max_eval_samples should be null, got {data.get('max_eval_samples')}")
    if data.get("train_window_sample_cursors") != expected_window:
        raise SystemExit(
            f"{context}: train_window_sample_cursors {data.get('train_window_sample_cursors')} != {expected_window}"
        )
    for key in ["dataset_source_sample_counts", "train_source_sample_counts"]:
        if data.get(key) != expected_counts:
            raise SystemExit(f"{context}: {key} {data.get(key)} != {expected_counts}")
    for key in ["dataset_source_files", "train_source_files"]:
        if data.get(key) != [expected_source]:
            raise SystemExit(f"{context}: {key} {data.get(key)} != {[expected_source]}")
    if data.get("dataset_fingerprint") != expected_fingerprint:
        raise SystemExit(
            f"{context}: dataset_fingerprint {data.get('dataset_fingerprint')} != {expected_fingerprint}"
        )
    if data.get("train_fingerprint") != expected_fingerprint:
        raise SystemExit(
            f"{context}: train_fingerprint {data.get('train_fingerprint')} != {expected_fingerprint}"
        )
    if data.get("eval_source_files") != []:
        raise SystemExit(f"{context}: eval_source_files should be empty, got {data.get('eval_source_files')}")
    if data.get("eval_source_sample_counts") != []:
        raise SystemExit(
            f"{context}: eval_source_sample_counts should be empty, got {data.get('eval_source_sample_counts')}"
        )

for context, data in [
    ("uncached", uncached),
    ("cache-first", cache_first),
    ("cache-second", cache_second),
]:
    verify_plan(data, context)

if uncached.get("streaming_index_cache_path") is not None:
    raise SystemExit(f"uncached cache path should be null: {uncached.get('streaming_index_cache_path')}")
if uncached.get("streaming_index_cache_hit") is not False:
    raise SystemExit(f"uncached cache_hit should be false: {uncached.get('streaming_index_cache_hit')}")
if uncached.get("streaming_index_cache_written") is not False:
    raise SystemExit(
        f"uncached cache_written should be false: {uncached.get('streaming_index_cache_written')}"
    )
for context, data, expected_hit, expected_written in [
    ("cache-first", cache_first, False, True),
    ("cache-second", cache_second, True, False),
]:
    if data.get("streaming_index_cache_path") != str(cache_path):
        raise SystemExit(
            f"{context}: cache path {data.get('streaming_index_cache_path')} != {cache_path}"
        )
    if data.get("streaming_index_cache_hit") is not expected_hit:
        raise SystemExit(f"{context}: cache_hit {data.get('streaming_index_cache_hit')} != {expected_hit}")
    if data.get("streaming_index_cache_written") is not expected_written:
        raise SystemExit(
            f"{context}: cache_written {data.get('streaming_index_cache_written')} != {expected_written}"
        )
if not cache_path.exists() or cache_path.stat().st_size == 0:
    raise SystemExit(f"expected non-empty streaming offset index cache at {cache_path}")
cache = json.loads(cache_path.read_text(encoding="utf-8"))
if cache.get("format") != "rustrain.qwen_sft_offset_index.v7":
    raise SystemExit(f"unexpected cache format {cache.get('format')}")
if len(cache.get("samples", [])) != 4:
    raise SystemExit(f"expected 4 cached raw sample offsets, got {len(cache.get('samples', []))}")
summary = cache.get("summary")
if not isinstance(summary, dict):
    raise SystemExit(f"expected cache summary metadata, got {summary}")
if summary.get("samples") != cache_second.get("dataset_total_samples"):
    raise SystemExit(f"cache summary samples {summary.get('samples')} != {cache_second.get('dataset_total_samples')}")
if summary.get("source_files") != cache_second.get("dataset_source_files"):
    raise SystemExit("cache summary source_files differ from cache-hit data-plan")
if summary.get("source_sample_counts") != cache_second.get("dataset_source_sample_counts"):
    raise SystemExit("cache summary source_sample_counts differ from cache-hit data-plan")
if summary.get("fingerprint") != cache_second.get("dataset_fingerprint"):
    raise SystemExit(
        f"cache summary fingerprint {summary.get('fingerprint')} != {cache_second.get('dataset_fingerprint')}"
    )

print(
    "qwen_sft_streaming_data_plan_verified: "
    f"total_samples={cache_second['dataset_total_samples']} "
    f"train_samples={cache_second['dataset_train_samples']} "
    f"eval_samples={cache_second['dataset_eval_samples']} "
    f"source_files={cache_second['dataset_source_files']} "
    f"fingerprint={cache_second['dataset_fingerprint']} "
    f"cache_hit={cache_second['streaming_index_cache_hit']} "
    f"tokenized_samples_materialized={cache_second['tokenized_samples_materialized']}"
)
PY
