#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ray/require_gpu_worker.sh"

BASE_CONFIG="${RUSTRAIN_QWEN_SESSION_DP2_TRAINER_INDEX_CACHE_CONFIG:-configs/qwen_session_dp2_sft_max_samples.toml}"
RUN_DIR="$(mktemp -d)"
CONFIG="${RUN_DIR}/session-dp2-index-cache.toml"
CACHE="${RUN_DIR}/qwen-session-dp-offset-index.json"
FIRST_OUTPUT_DIR="${RUN_DIR}/first"
SECOND_OUTPUT_DIR="${RUN_DIR}/second"

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

cargo run -- launch --nproc-per-node 2 --output-dir "${FIRST_OUTPUT_DIR}" train --config "${CONFIG}"
cargo run -- launch --nproc-per-node 2 --output-dir "${SECOND_OUTPUT_DIR}" train --config "${CONFIG}"

python - "${FIRST_OUTPUT_DIR}" "${SECOND_OUTPUT_DIR}" "${CACHE}" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd() / "scripts"))
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
        parsed[int(data["rank"])] = data
    return parsed

first = summaries(first_dir)
second = summaries(second_dir)

for rank in [0, 1]:
    expected_cache = rank_cache(rank)
    if not expected_cache.exists():
        raise SystemExit(f"rank {rank} cache was not written: {expected_cache}")
    for context, data, expected_hit, expected_written in [
        ("first", first[rank], False, True),
        ("second", second[rank], True, False),
    ]:
        if data.get("streaming_train_batches") is not True:
            raise SystemExit(f"{context} rank {rank} streaming_train_batches must be true")
        if pathlib.Path(data.get("streaming_index_cache_path", "")) != expected_cache:
            raise SystemExit(
                f"{context} rank {rank} cache path {data.get('streaming_index_cache_path')} != {expected_cache}"
            )
        if data.get("streaming_index_cache_hit") is not expected_hit:
            raise SystemExit(
                f"{context} rank {rank} cache_hit {data.get('streaming_index_cache_hit')} != {expected_hit}"
            )
        if data.get("streaming_index_cache_written") is not expected_written:
            raise SystemExit(
                f"{context} rank {rank} cache_written {data.get('streaming_index_cache_written')} != {expected_written}"
            )
        for key in ["reload_delta", "sharded_reload_delta", "sharded_next_step_delta"]:
            if float(data[key]) > 1e-5:
                raise SystemExit(f"{context} rank {rank} {key} too large: {data[key]}")
        sharded_manifest_path = pathlib.Path(data["sharded_global_manifest_output"])
        if not sharded_manifest_path.is_file():
            raise SystemExit(
                f"{context} rank {rank} sharded global manifest missing: {sharded_manifest_path}"
            )
        sharded_manifest = json.loads(sharded_manifest_path.read_text())
        require_complete_qwen_manifest_paths(sharded_manifest, sharded_manifest_path)
        if sharded_manifest.get("streaming_train_batches") is not True:
            raise SystemExit(
                f"{context} rank {rank} sharded streaming_train_batches {sharded_manifest.get('streaming_train_batches')} is not true"
            )
        if data.get("checkpoint_written"):
            rank0_manifest_path = pathlib.Path(data["manifest_output"])
            if not rank0_manifest_path.is_file():
                raise SystemExit(
                    f"{context} rank {rank} rank0 manifest missing: {rank0_manifest_path}"
                )
            rank0_manifest = json.loads(rank0_manifest_path.read_text())
            require_complete_qwen_base_model_path(rank0_manifest, rank0_manifest_path)
            if rank0_manifest.get("streaming_train_batches") is not True:
                raise SystemExit(
                    f"{context} rank {rank} rank0 streaming_train_batches {rank0_manifest.get('streaming_train_batches')} is not true"
                )

print(
    "qwen_session_dp2_trainer_index_cache_verified: "
    f"cache={cache_path} "
    "first_rank_writes=true,true second_rank_hits=true,true"
)
PY
