#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

BASE_CONFIG="${RUSTRAIN_QWEN_SESSION_SINGLE_SFT_ARROW_INDEX_CACHE_CONFIG:-configs/qwen_session_single_sft_arrow_index_cache.toml}"
WORK_DIR="$(mktemp -d /tmp/rustrain-arrow-cache-verify-XXXXXX)"
CONFIG="${WORK_DIR}/config.toml"
CACHE_PATH="${WORK_DIR}/arrow-row-index.json"
FIRST_PLAN_OUTPUT="${WORK_DIR}/first-data-plan.txt"
SECOND_PLAN_OUTPUT="${WORK_DIR}/second-data-plan.txt"
FIRST_BATCH_OUTPUT="${WORK_DIR}/first-batch-plan.txt"
SECOND_BATCH_OUTPUT="${WORK_DIR}/second-batch-plan.txt"
FIRST_TRAIN_OUTPUT="${WORK_DIR}/first-train.txt"
SECOND_TRAIN_OUTPUT="${WORK_DIR}/second-train.txt"

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

cargo run -- qwen-sft-streaming-data-plan --config "${CONFIG}" --world-size 1 --data-cursor-start 0 \
  | tee "${FIRST_PLAN_OUTPUT}"
cargo run -- qwen-sft-streaming-data-plan --config "${CONFIG}" --world-size 1 --data-cursor-start 0 \
  | tee "${SECOND_PLAN_OUTPUT}"

cargo run -- qwen-sft-streaming-batch-plan --config "${CONFIG}" --world-size 1 --data-cursor-start 0 --index-cache "${CACHE_PATH}" \
  | tee "${FIRST_BATCH_OUTPUT}"
cargo run -- qwen-sft-streaming-batch-plan --config "${CONFIG}" --world-size 1 --data-cursor-start 0 --index-cache "${CACHE_PATH}" \
  | tee "${SECOND_BATCH_OUTPUT}"

rm -f "${CACHE_PATH}"
cargo run -- train --config "${CONFIG}" | tee "${FIRST_TRAIN_OUTPUT}"
cargo run -- train --config "${CONFIG}" | tee "${SECOND_TRAIN_OUTPUT}"

python - "${CACHE_PATH}" "${FIRST_PLAN_OUTPUT}" "${SECOND_PLAN_OUTPUT}" "${FIRST_BATCH_OUTPUT}" "${SECOND_BATCH_OUTPUT}" "${FIRST_TRAIN_OUTPUT}" "${SECOND_TRAIN_OUTPUT}" <<'PY'
import json
import pathlib
import sys

def load_json_output(path):
    text = pathlib.Path(path).read_text()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise SystemExit(f"{path} did not contain JSON: {text}")
    return json.loads(text[start : end + 1])

def load_values(path):
    values = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            values[key] = value
    return values

cache_path = pathlib.Path(sys.argv[1])
first_plan = load_json_output(sys.argv[2])
second_plan = load_json_output(sys.argv[3])
first_batch = load_json_output(sys.argv[4])
second_batch = load_json_output(sys.argv[5])
first_train = load_values(sys.argv[6])
second_train = load_values(sys.argv[7])

if not cache_path.exists():
    raise SystemExit(f"expected Arrow index cache to exist: {cache_path}")
cache = json.loads(cache_path.read_text())
if cache.get("format") != "rustrain.qwen_sft_arrow_row_index.v1":
    raise SystemExit(f"unexpected cache format: {cache.get('format')}")
if cache.get("summary", {}).get("samples") != 32:
    raise SystemExit(f"expected 32 cached Arrow samples, got {cache.get('summary')}")
if len(cache.get("samples", [])) != 32:
    raise SystemExit(f"expected 32 cached row indices, got {len(cache.get('samples', []))}")
if cache["samples"][0]["row_index"] != 0 or cache["samples"][-1]["row_index"] != 31:
    raise SystemExit(f"unexpected cached row range: {cache['samples'][0]} .. {cache['samples'][-1]}")

def assert_cache_fields(data, context, expected_hit, expected_written):
    if data.get("streaming_index_cache_path") != str(cache_path):
        raise SystemExit(f"{context}: unexpected cache path {data.get('streaming_index_cache_path')}")
    if data.get("streaming_index_cache_hit") is not expected_hit:
        raise SystemExit(f"{context}: expected hit {expected_hit}, got {data.get('streaming_index_cache_hit')}")
    if data.get("streaming_index_cache_written") is not expected_written:
        raise SystemExit(f"{context}: expected written {expected_written}, got {data.get('streaming_index_cache_written')}")

assert_cache_fields(first_plan, "first data plan", False, True)
assert_cache_fields(second_plan, "second data plan", True, False)
assert_cache_fields(first_batch, "first batch plan", True, False)
assert_cache_fields(second_batch, "second batch plan", True, False)
if first_batch["materialized_input_max_delta"] != 0 or float(first_batch["materialized_mask_max_delta"]) != 0.0:
    raise SystemExit("first batch plan materialized deltas must be zero")
if second_batch["materialized_input_max_delta"] != 0 or float(second_batch["materialized_mask_max_delta"]) != 0.0:
    raise SystemExit("second batch plan materialized deltas must be zero")
if first_batch["streaming_raw_sample_indices"] != [] or second_batch["streaming_raw_sample_indices"] != []:
    raise SystemExit("Arrow batch plan must not report JSONL raw indices")

for values, context, expected_hit, expected_written in [
    (first_train, "first train", "false", "true"),
    (second_train, "second train", "true", "false"),
]:
    if values.get("streaming_index_cache_path") != str(cache_path):
        raise SystemExit(f"{context}: unexpected cache path {values.get('streaming_index_cache_path')}")
    if values.get("streaming_index_cache_hit") != expected_hit:
        raise SystemExit(f"{context}: expected hit {expected_hit}, got {values.get('streaming_index_cache_hit')}")
    if values.get("streaming_index_cache_written") != expected_written:
        raise SystemExit(f"{context}: expected written {expected_written}, got {values.get('streaming_index_cache_written')}")
    if values.get("dataset_total_samples") != "32":
        raise SystemExit(f"{context}: expected 32 samples, got {values.get('dataset_total_samples')}")
    if float(values.get("reload_delta", "nan")) > 1e-5:
        raise SystemExit(f"{context}: reload_delta too large: {values.get('reload_delta')}")
    if float(values.get("second_step_delta", "nan")) > 1e-5:
        raise SystemExit(f"{context}: second_step_delta too large: {values.get('second_step_delta')}")

print(
    "qwen_session_single_sft_arrow_index_cache_verified: "
    f"cache={cache_path} samples={len(cache['samples'])} "
    f"first_written={first_train['streaming_index_cache_written']} "
    f"second_hit={second_train['streaming_index_cache_hit']}"
)
PY
