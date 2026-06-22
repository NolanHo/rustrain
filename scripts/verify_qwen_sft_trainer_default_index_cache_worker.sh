#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SFT_TRAINER_DEFAULT_INDEX_CACHE_CONFIG:-configs/qwen_session_single_sft_max_samples.toml}"
OUTPUT="$(mktemp)"

cargo run -- train --config "${CONFIG}" | tee "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd() / "scripts"))
from qwen_verify_utils import require_complete_qwen_base_model_path

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
    "manifest_output",
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
manifest_path = pathlib.Path(values["manifest_output"])
if not manifest_path.is_file():
    raise SystemExit(f"manifest_output missing: {manifest_path}")
manifest = json.loads(manifest_path.read_text())
require_complete_qwen_base_model_path(manifest, manifest_path)
if manifest.get("streaming_train_batches") is not True:
    raise SystemExit(
        f"manifest streaming_train_batches {manifest.get('streaming_train_batches')} is not true"
    )

print(
    "qwen_sft_trainer_default_index_cache_verified: "
    f"cache={cache_path} written={values['streaming_index_cache_written']}"
)
PY
