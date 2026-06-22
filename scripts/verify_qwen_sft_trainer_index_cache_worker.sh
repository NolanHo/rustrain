#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

BASE_CONFIG="${RUSTRAIN_QWEN_SFT_TRAINER_INDEX_CACHE_CONFIG:-configs/qwen_session_single_sft_max_samples.toml}"
RUN_DIR="$(mktemp -d)"
CONFIG="${RUN_DIR}/trainer-index-cache.toml"
CACHE="${RUN_DIR}/qwen-sft-offset-index.json"
FIRST_OUTPUT="${RUN_DIR}/first.out"
SECOND_OUTPUT="${RUN_DIR}/second.out"

python - "${BASE_CONFIG}" "${CONFIG}" "${CACHE}" <<'PY'
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
cache = pathlib.Path(sys.argv[3])
text = base.read_text()
needle = 'max_samples = 4\n'
if needle not in text:
    raise SystemExit(f"{base} does not contain expected {needle!r}")
text = text.replace(needle, needle + f'index_cache = "{cache}"\n', 1)
target.write_text(text)
PY

cargo run -- train --config "${CONFIG}" | tee "${FIRST_OUTPUT}"
cargo run -- train --config "${CONFIG}" | tee "${SECOND_OUTPUT}"

python - "${FIRST_OUTPUT}" "${SECOND_OUTPUT}" "${CACHE}" <<'PY'
import pathlib
import sys

first_path = pathlib.Path(sys.argv[1])
second_path = pathlib.Path(sys.argv[2])
cache_path = pathlib.Path(sys.argv[3])

def parse(path):
    values = {}
    for line in path.read_text().splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        values[key] = value
    return values

first = parse(first_path)
second = parse(second_path)
required = [
    "streaming_train_batches",
    "streaming_index_cache_path",
    "streaming_index_cache_hit",
    "streaming_index_cache_written",
]
for context, values in [("first", first), ("second", second)]:
    missing = [key for key in required if key not in values]
    if missing:
        raise SystemExit(f"{context} trainer run is missing fields: {missing}")
    if values["streaming_train_batches"] != "true":
        raise SystemExit(f"{context} streaming_train_batches must be true")
    if pathlib.Path(values["streaming_index_cache_path"]) != cache_path:
        raise SystemExit(
            f"{context} cache path {values['streaming_index_cache_path']} != {cache_path}"
        )

if first["streaming_index_cache_hit"] != "false":
    raise SystemExit(f"first cache_hit should be false: {first['streaming_index_cache_hit']}")
if first["streaming_index_cache_written"] != "true":
    raise SystemExit(
        f"first cache_written should be true: {first['streaming_index_cache_written']}"
    )
if second["streaming_index_cache_hit"] != "true":
    raise SystemExit(f"second cache_hit should be true: {second['streaming_index_cache_hit']}")
if second["streaming_index_cache_written"] != "false":
    raise SystemExit(
        f"second cache_written should be false: {second['streaming_index_cache_written']}"
    )
if not cache_path.exists():
    raise SystemExit(f"cache was not written: {cache_path}")

print(
    "qwen_sft_trainer_index_cache_verified: "
    f"cache={cache_path} "
    f"first_written={first['streaming_index_cache_written']} "
    f"second_hit={second['streaming_index_cache_hit']}"
)
PY
