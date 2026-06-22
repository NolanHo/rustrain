#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SFT_STREAMING_EVAL_PATHS_CONFIG:-configs/qwen_session_dp2_sft_eval_paths.toml}"
OUTPUT="$(mktemp)"

cargo run -- qwen-sft-streaming-data-plan --config "${CONFIG}" | tee "${OUTPUT}"

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
expected_counts = {
    "data/sft_toy/eval_instructions.jsonl": 2,
    "data/sft_toy/instructions.jsonl": 6,
    "data/sft_toy/more_instructions.jsonl": 2,
}
expected_train_counts = {
    "data/sft_toy/instructions.jsonl": 6,
    "data/sft_toy/more_instructions.jsonl": 2,
}
expected_eval_counts = {"data/sft_toy/eval_instructions.jsonl": 2}

checks = {
    "max_samples": None,
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
for key, expected in [
    ("dataset_source_sample_counts", expected_counts),
    ("train_source_sample_counts", expected_train_counts),
    ("eval_source_sample_counts", expected_eval_counts),
]:
    actual = {entry["path"]: entry["samples"] for entry in data.get(key) or []}
    if actual != expected:
        raise SystemExit(f"{key} {actual} != {expected}")
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
