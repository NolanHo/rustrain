#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SFT_TRAINER_DEFAULT_INDEX_CACHE_CONFIG:-configs/qwen_session_single_sft_max_samples.toml}"
OUTPUT="$(mktemp)"
RUN_DIR="$(mktemp -d)"
EFFECTIVE_CONFIG="${RUN_DIR}/default-index-cache.toml"

python - "${CONFIG}" "${EFFECTIVE_CONFIG}" <<'PY'
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
text = source.read_text()
configured = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
fallback = (
    "/vePFS-Mindverse/share/huggingface/hub/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
if not (pathlib.Path(configured) / "config.json").exists():
    if not (pathlib.Path(fallback) / "config.json").exists():
        raise SystemExit(
            f"neither configured nor fallback Qwen model path exists: {configured}, {fallback}"
        )
    text = text.replace(configured, fallback)
target.write_text(text)
PY

cargo run -- train --config "${EFFECTIVE_CONFIG}" | tee "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
import pathlib
import sys

output_path = pathlib.Path(sys.argv[1])
values = {}
for line in output_path.read_text().splitlines():
    if ": " not in line:
        continue
    key, value = line.split(": ", 1)
    values[key] = value

required = [
    "run_dir",
    "streaming_train_batches",
    "streaming_index_cache_path",
    "streaming_index_cache_hit",
    "streaming_index_cache_written",
    "reload_delta",
    "second_step_delta",
]
missing = [key for key in required if key not in values]
if missing:
    raise SystemExit(f"default-cache trainer run is missing fields: {missing}")
if values["streaming_train_batches"] != "true":
    raise SystemExit(f"streaming_train_batches must be true: {values['streaming_train_batches']}")
if values["streaming_index_cache_hit"] != "false":
    raise SystemExit(f"default first run should not hit cache: {values['streaming_index_cache_hit']}")
if values["streaming_index_cache_written"] != "true":
    raise SystemExit(
        f"default first run should write cache: {values['streaming_index_cache_written']}"
    )
for key in ["reload_delta", "second_step_delta"]:
    if float(values[key]) > 1e-6:
        raise SystemExit(f"{key} too large: {values[key]}")

run_dir = pathlib.Path(values["run_dir"])
cache_path = pathlib.Path(values["streaming_index_cache_path"])
expected = run_dir / "cache" / "qwen-session-single-offset-index.json"
if cache_path != expected:
    raise SystemExit(f"default cache path {cache_path} != {expected}")
if not cache_path.exists():
    raise SystemExit(f"default cache was not written: {cache_path}")

print(
    "qwen_sft_trainer_default_index_cache_verified: "
    f"cache={cache_path} written={values['streaming_index_cache_written']}"
)
PY
