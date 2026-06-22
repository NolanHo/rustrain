#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SFT_STREAMING_EVAL_PATHS_CONFIG:-configs/qwen_session_dp2_sft_eval_paths.toml}"
OUTPUT="$(mktemp)"

cargo run -- qwen-sft-streaming-data-plan \
  --config "${CONFIG}" \
  --world-size 2 \
  --data-cursor-start 4 \
  | tee "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
import json
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
start = text.find("{")
if start < 0:
    raise SystemExit(f"streaming eval_paths output did not contain JSON: {text}")
data = json.loads(text[start:])

expected_sources = [
    "data/sft_toy/eval_instructions.jsonl",
    "data/sft_toy/instructions.jsonl",
    "data/sft_toy/more_instructions.jsonl",
]
expected_train_sources = [
    "data/sft_toy/instructions.jsonl",
    "data/sft_toy/more_instructions.jsonl",
]
expected_eval_sources = ["data/sft_toy/eval_instructions.jsonl"]
expected_counts = [
    {"path": "data/sft_toy/eval_instructions.jsonl", "samples": 2},
    {"path": "data/sft_toy/instructions.jsonl", "samples": 6},
    {"path": "data/sft_toy/more_instructions.jsonl", "samples": 2},
]
expected_train_counts = [
    {"path": "data/sft_toy/instructions.jsonl", "samples": 6},
    {"path": "data/sft_toy/more_instructions.jsonl", "samples": 2},
]
expected_eval_counts = [{"path": "data/sft_toy/eval_instructions.jsonl", "samples": 2}]

checks = {
    "max_samples": None,
    "world_size": 2,
    "local_batch_size": 1,
    "global_batch_size": 2,
    "train_steps": 2,
    "required_batches": 5,
    "data_cursor_start": 4,
    "data_cursor_end": 8,
    "data_cursor_next": 8,
    "data_epoch_start": 0,
    "data_epoch_end": 1,
    "data_epoch_next": 1,
    "data_sample_offset_start": 4,
    "data_sample_offset_end": 0,
    "data_sample_offset_next": 0,
    "train_window_start_cursor": 4,
    "train_window_end_cursor_exclusive": 10,
    "dataset_total_samples": 10,
    "dataset_train_samples": 8,
    "dataset_eval_samples": 2,
    "dataset_order_seed": 777,
    "dataset_shuffle": True,
    "tokenizer_loaded": False,
    "tokenized_samples_materialized": False,
}
for key, expected in checks.items():
    if data.get(key) != expected:
        raise SystemExit(f"{key} {data.get(key)} != {expected}")
expected_window = [
    {"cursor": 4, "epoch": 0, "sample_offset": 4},
    {"cursor": 5, "epoch": 0, "sample_offset": 5},
    {"cursor": 6, "epoch": 0, "sample_offset": 6},
    {"cursor": 7, "epoch": 0, "sample_offset": 7},
    {"cursor": 8, "epoch": 1, "sample_offset": 0},
    {"cursor": 9, "epoch": 1, "sample_offset": 1},
]
if data.get("train_window_sample_cursors") != expected_window:
    raise SystemExit(
        f"train_window_sample_cursors {data.get('train_window_sample_cursors')} != {expected_window}"
    )
if data.get("dataset_source_files") != expected_sources:
    raise SystemExit(
        f"dataset_source_files {data.get('dataset_source_files')} != {expected_sources}"
    )
if data.get("train_source_files") != expected_train_sources:
    raise SystemExit(
        f"train_source_files {data.get('train_source_files')} != {expected_train_sources}"
    )
if data.get("eval_source_files") != expected_eval_sources:
    raise SystemExit(
        f"eval_source_files {data.get('eval_source_files')} != {expected_eval_sources}"
    )
for key, expected, expected_paths, expected_total in [
    ("dataset_source_sample_counts", expected_counts, expected_sources, data["dataset_total_samples"]),
    ("train_source_sample_counts", expected_train_counts, expected_train_sources, data["dataset_train_samples"]),
    ("eval_source_sample_counts", expected_eval_counts, expected_eval_sources, data["dataset_eval_samples"]),
]:
    actual = data.get(key) or []
    if actual != expected:
        raise SystemExit(f"{key} {actual} != {expected}")
    if [entry.get("path") for entry in actual] != expected_paths:
        raise SystemExit(
            f"{key} paths {[entry.get('path') for entry in actual]} != {expected_paths}"
        )
    if any(int(entry.get("samples", 0)) <= 0 for entry in actual):
        raise SystemExit(f"{key} must contain positive samples: {actual}")
    if sum(int(entry["samples"]) for entry in actual) != int(expected_total):
        raise SystemExit(f"{key} total {actual} != expected total {expected_total}")
for key in ["dataset_fingerprint", "train_fingerprint", "eval_fingerprint"]:
    if not data.get(key):
        raise SystemExit(f"{key} must not be empty")
if data["dataset_fingerprint"] in {data["train_fingerprint"], data["eval_fingerprint"]}:
    raise SystemExit("combined dataset_fingerprint must differ from train/eval fingerprints")

print(
    "qwen_sft_streaming_eval_paths_verified: "
    f"total_samples={data['dataset_total_samples']} "
    f"train_samples={data['dataset_train_samples']} "
    f"eval_samples={data['dataset_eval_samples']} "
    f"source_files={data['dataset_source_files']} "
    f"fingerprint={data['dataset_fingerprint']}"
)
PY
