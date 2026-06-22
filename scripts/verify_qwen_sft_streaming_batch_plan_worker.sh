#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SFT_STREAMING_BATCH_PLAN_CONFIG:-configs/qwen_session_dp2_sft_max_samples.toml}"
EXPECTED_SOURCE="${RUSTRAIN_EXPECTED_STREAMING_SOURCE:-data/sft_toy/instructions.jsonl}"
EXPECTED_FINGERPRINT="${RUSTRAIN_EXPECTED_STREAMING_FINGERPRINT:-3bfd266239e4b9b9}"
OUTPUT="$(mktemp)"
CACHE_PATH="$(mktemp -u /tmp/rustrain-qwen-sft-offset-index-XXXXXX.json)"
CACHED_OUTPUT_FIRST="$(mktemp)"
CACHED_OUTPUT_SECOND="$(mktemp)"

cargo run -- qwen-sft-streaming-batch-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 2 \
  | tee "${OUTPUT}"

cargo run -- qwen-sft-streaming-batch-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 2 \
  --index-cache "${CACHE_PATH}" \
  | tee "${CACHED_OUTPUT_FIRST}"

cargo run -- qwen-sft-streaming-batch-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 2 \
  --index-cache "${CACHE_PATH}" \
  | tee "${CACHED_OUTPUT_SECOND}"

python - "${OUTPUT}" "${CACHED_OUTPUT_FIRST}" "${CACHED_OUTPUT_SECOND}" "${CACHE_PATH}" "${EXPECTED_SOURCE}" "${EXPECTED_FINGERPRINT}" <<'PY'
import json
import pathlib
import sys

output_paths = [pathlib.Path(sys.argv[index]) for index in range(1, 4)]
cache_path = pathlib.Path(sys.argv[4])
expected_source = sys.argv[5]
expected_fingerprint = sys.argv[6]

def load_json(path):
    text = path.read_text()
    start = text.find("{")
    if start < 0:
        raise SystemExit(f"streaming batch plan output did not contain JSON: {text}")
    return json.loads(text[start:])

summaries = [load_json(path) for path in output_paths]

checks = {
    "world_size": 2,
    "local_batch_size": 1,
    "global_batch_size": 2,
    "train_steps": 1,
    "required_batches": 3,
    "train_batch_count": 2,
    "data_cursor_start": 2,
    "data_cursor_end": 4,
    "data_cursor_next": 4,
    "train_window_start_cursor": 2,
    "train_window_end_cursor_exclusive": 6,
    "dataset_total_samples": 4,
    "dataset_train_samples": 3,
    "dataset_eval_samples": 1,
    "dataset_order_seed": 777,
    "dataset_shuffle": True,
    "tokenizer_loaded": True,
    "tokenized_samples_materialized": True,
    "reference_tokenized_samples_materialized": True,
    "streaming_window_samples": 4,
    "streaming_raw_samples_read": 3,
    "materialized_input_max_delta": 0,
    "materialized_mask_max_delta": 0.0,
}
expected_window = [
    {"cursor": 2, "epoch": 0, "sample_offset": 2},
    {"cursor": 3, "epoch": 1, "sample_offset": 0},
    {"cursor": 4, "epoch": 1, "sample_offset": 1},
    {"cursor": 5, "epoch": 1, "sample_offset": 2},
]
expected_raw_indices = [
    {"path": expected_source, "index_in_file": 3, "global_index": 3},
    {"path": expected_source, "index_in_file": 3, "global_index": 3},
    {"path": expected_source, "index_in_file": 2, "global_index": 2},
    {"path": expected_source, "index_in_file": 1, "global_index": 1},
]

def verify_summary(data, context):
    for key, expected in checks.items():
        if data.get(key) != expected:
            raise SystemExit(f"{context}: {key} {data.get(key)} != {expected}")
    if data.get("train_window_sample_cursors") != expected_window:
        raise SystemExit(
            f"{context}: train_window_sample_cursors {data.get('train_window_sample_cursors')} != {expected_window}"
        )
    if data.get("dataset_source_files") != [expected_source]:
        raise SystemExit(
            f"{context}: dataset_source_files {data.get('dataset_source_files')} != {[expected_source]}"
        )
    if data.get("dataset_source_sample_counts") != [{"path": expected_source, "samples": 4}]:
        raise SystemExit(
            f"{context}: dataset_source_sample_counts {data.get('dataset_source_sample_counts')} did not match"
        )
    if data.get("dataset_fingerprint") != expected_fingerprint:
        raise SystemExit(
            f"{context}: dataset_fingerprint {data.get('dataset_fingerprint')} != {expected_fingerprint}"
        )
    batch_sequence_tokens = data.get("batch_sequence_tokens")
    batch_masked_positions = data.get("batch_masked_positions")
    batch_padding_tokens = data.get("batch_padding_tokens")
    batch_token_fingerprints = data.get("batch_token_fingerprints")
    if not all(isinstance(values, list) and len(values) == 2 for values in [
        batch_sequence_tokens,
        batch_masked_positions,
        batch_padding_tokens,
        batch_token_fingerprints,
    ]):
        raise SystemExit(f"{context}: expected two streaming batches worth of tokenized batch metadata")
    if any(value <= 1 for value in batch_sequence_tokens):
        raise SystemExit(f"{context}: batch_sequence_tokens must be > 1, got {batch_sequence_tokens}")
    if any(value <= 0 for value in batch_masked_positions):
        raise SystemExit(f"{context}: batch_masked_positions must be positive, got {batch_masked_positions}")
    if any(not isinstance(value, str) or len(value) != 16 for value in batch_token_fingerprints):
        raise SystemExit(f"{context}: bad batch_token_fingerprints: {batch_token_fingerprints}")
    raw_indices = data.get("streaming_raw_sample_indices")
    raw_indices_without_offsets = [
        {key: value for key, value in entry.items() if key != "byte_offset"}
        for entry in raw_indices
    ]
    if raw_indices_without_offsets != expected_raw_indices:
        raise SystemExit(
            f"{context}: streaming_raw_sample_indices without byte offsets {raw_indices_without_offsets} != {expected_raw_indices}"
        )
    offsets_by_sample = {}
    for entry in raw_indices:
        offset = entry.get("byte_offset")
        if not isinstance(offset, int) or offset < 0:
            raise SystemExit(f"{context}: streaming raw sample byte_offset must be a non-negative integer: {entry}")
        sample_key = (entry["path"], entry["index_in_file"], entry["global_index"])
        previous = offsets_by_sample.setdefault(sample_key, offset)
        if previous != offset:
            raise SystemExit(
                f"{context}: duplicate streaming raw sample {sample_key} used inconsistent byte offsets: {previous} vs {offset}"
            )
    return batch_sequence_tokens, raw_indices

uncached, cache_write, cache_hit = summaries
batch_sequence_tokens, raw_indices = verify_summary(uncached, "uncached")
verify_summary(cache_write, "cache_write")
verify_summary(cache_hit, "cache_hit")

if uncached.get("streaming_index_cache_path") is not None:
    raise SystemExit("uncached summary should not report streaming_index_cache_path")
if uncached.get("streaming_index_cache_hit") is not False:
    raise SystemExit("uncached summary should report streaming_index_cache_hit=false")
if uncached.get("streaming_index_cache_written") is not False:
    raise SystemExit("uncached summary should report streaming_index_cache_written=false")

for context, summary, expected_hit, expected_written in [
    ("cache_write", cache_write, False, True),
    ("cache_hit", cache_hit, True, False),
]:
    if summary.get("streaming_index_cache_path") != str(cache_path):
        raise SystemExit(
            f"{context}: streaming_index_cache_path {summary.get('streaming_index_cache_path')} != {cache_path}"
        )
    if summary.get("streaming_index_cache_hit") is not expected_hit:
        raise SystemExit(
            f"{context}: streaming_index_cache_hit {summary.get('streaming_index_cache_hit')} != {expected_hit}"
        )
    if summary.get("streaming_index_cache_written") is not expected_written:
        raise SystemExit(
            f"{context}: streaming_index_cache_written {summary.get('streaming_index_cache_written')} != {expected_written}"
        )

if not cache_path.exists() or cache_path.stat().st_size == 0:
    raise SystemExit(f"expected non-empty streaming offset index cache at {cache_path}")
cache = json.loads(cache_path.read_text())
if cache.get("format") != "rustrain.qwen_sft_offset_index.v7":
    raise SystemExit(f"unexpected offset index cache format: {cache.get('format')}")
source_files = cache.get("source_files")
if not isinstance(source_files, list) or not source_files:
    raise SystemExit(f"expected non-empty cache source_files, got {source_files}")
for entry in source_files:
    if not isinstance(entry, dict):
        raise SystemExit(f"cache source_files entry must be an object: {entry}")
    if not str(entry.get("path", "")).endswith(".jsonl"):
        raise SystemExit(f"cache source file path must be JSONL: {entry}")
    if int(entry.get("len", 0)) <= 0:
        raise SystemExit(f"cache source file len must be positive: {entry}")
    if int(entry.get("modified_unix_nanos", 0)) <= 0:
        raise SystemExit(f"cache source file mtime must be positive: {entry}")
field_map = cache.get("field_map")
if not isinstance(field_map, dict):
    raise SystemExit(f"expected cache field_map object, got {field_map}")
expected_field_map = {
    "instruction": "instruction",
    "input": "input",
    "response": "response",
    "system": None,
    "chat_messages": None,
    "prompt_template": "Instruction:\n{instruction}\n\nResponse:\n",
    "prompt_with_input_template": "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n",
    "trim_fields": True,
    "min_response_chars": 1,
    "max_response_chars": None,
    "instruction_contains_any": [],
    "instruction_excludes_any": [],
    "response_contains_any": [],
    "response_excludes_any": [],
    "input_contains_any": [],
    "input_excludes_any": [],
    "min_instruction_chars": None,
    "max_instruction_chars": None,
    "min_input_chars": None,
    "max_input_chars": None,
    "min_system_chars": None,
    "max_system_chars": None,
    "system_contains_any": [],
    "system_excludes_any": [],
    "min_prompt_chars": None,
    "max_prompt_chars": None,
    "min_sample_chars": None,
    "max_sample_chars": None,
    "dedupe_samples": False,
    "field_replacements": [],
    "normalize_whitespace": False,
    "source_weights": [],
    "source_max_samples": [],
    "skip_invalid_records": False,
}
for key, expected in expected_field_map.items():
    if field_map.get(key) != expected:
        raise SystemExit(f"cache field_map[{key!r}] {field_map.get(key)!r} != {expected!r}")
if cache.get("min_response_chars") != 1:
    raise SystemExit(f"cache min_response_chars {cache.get('min_response_chars')} != 1")
if len(cache.get("samples", [])) != 4:
    raise SystemExit(f"expected 4 cached raw sample offsets, got {len(cache.get('samples', []))}")
summary = cache.get("summary")
if not isinstance(summary, dict):
    raise SystemExit(f"expected cache summary object, got {summary}")
if summary.get("samples") != 4:
    raise SystemExit(f"cache summary samples {summary.get('samples')} != 4")
if summary.get("source_files") != [expected_source]:
    raise SystemExit(f"cache summary source_files {summary.get('source_files')} != {[expected_source]}")
if summary.get("source_sample_counts") != [{"path": expected_source, "samples": 4}]:
    raise SystemExit(f"cache summary source_sample_counts {summary.get('source_sample_counts')} != expected")
if summary.get("fingerprint") != expected_fingerprint:
    raise SystemExit(f"cache summary fingerprint {summary.get('fingerprint')} != {expected_fingerprint}")
if cache_hit.get("streaming_raw_sample_indices") != cache_write.get("streaming_raw_sample_indices"):
    raise SystemExit("cache hit raw sample indices differ from cache write run")

print(
    "qwen_sft_streaming_batch_plan_verified: "
    f"streaming_window_samples={uncached['streaming_window_samples']} "
    f"streaming_raw_samples_read={uncached['streaming_raw_samples_read']} "
    f"cache_path={cache_path} "
    f"cache_hit={cache_hit['streaming_index_cache_hit']} "
    f"batch_sequence_tokens={batch_sequence_tokens} "
    f"materialized_input_max_delta={uncached['materialized_input_max_delta']} "
    f"materialized_mask_max_delta={uncached['materialized_mask_max_delta']}"
)
PY
