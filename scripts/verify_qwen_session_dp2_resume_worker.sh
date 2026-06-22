#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

BASE_OUTPUT_DIR="${RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR:?RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR is required}"
RESUME_OUTPUT_DIR="${RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR:?RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR is required}"
CONFIG="${RUSTRAIN_QWEN_SESSION_DP_CONFIG:-configs/qwen_session_dp2_sft.toml}"
EXPECTED_DATASET_SEED="${RUSTRAIN_EXPECTED_DATASET_ORDER_SEED:-}"
EXPECTED_TRAINABLE_TENSORS="${RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS:-}"
EXPECTED_TRAINABLE_NAMES="${RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES:-}"
EXPECTED_DTYPE="${RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND:-}"
EXPECTED_DATASET_TOTAL_SAMPLES="${RUSTRAIN_EXPECTED_DATASET_TOTAL_SAMPLES:-}"
EXPECTED_DATASET_TRAIN_SAMPLES="${RUSTRAIN_EXPECTED_DATASET_TRAIN_SAMPLES:-}"
EXPECTED_DATASET_EVAL_SAMPLES="${RUSTRAIN_EXPECTED_DATASET_EVAL_SAMPLES:-}"
EXPECTED_DATASET_SOURCE_FILES="${RUSTRAIN_EXPECTED_DATASET_SOURCE_FILES:-}"
EXPECTED_DATASET_SOURCE_SAMPLE_COUNTS="${RUSTRAIN_EXPECTED_DATASET_SOURCE_SAMPLE_COUNTS:-}"
EXPECTED_DATASET_FINGERPRINT="${RUSTRAIN_EXPECTED_DATASET_FINGERPRINT:-}"
export BASE_OUTPUT_DIR RESUME_OUTPUT_DIR EXPECTED_DATASET_SEED EXPECTED_TRAINABLE_TENSORS EXPECTED_TRAINABLE_NAMES EXPECTED_DTYPE
export EXPECTED_DATASET_TOTAL_SAMPLES EXPECTED_DATASET_TRAIN_SAMPLES EXPECTED_DATASET_EVAL_SAMPLES
export EXPECTED_DATASET_SOURCE_FILES EXPECTED_DATASET_SOURCE_SAMPLE_COUNTS EXPECTED_DATASET_FINGERPRINT

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${BASE_OUTPUT_DIR}" \
  train --config "${CONFIG}"

BASE_MANIFEST="$(
  python - <<'PY'
import json
import os
import pathlib

output_dir = pathlib.Path(os.environ["BASE_OUTPUT_DIR"])
rank0 = sorted(output_dir.rglob("qwen-session-dp-rank-0.json"))
if len(rank0) != 1:
    raise SystemExit(f"expected one rank0 summary under {output_dir}, found {len(rank0)}")
summary = json.loads(rank0[0].read_text())
if not summary.get("checkpoint_written"):
    raise SystemExit("rank0 base run did not write checkpoint")
print(summary["manifest_output"])
PY
)"

cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${RESUME_OUTPUT_DIR}" \
  train --config "${CONFIG}" --resume-from "${BASE_MANIFEST}"

python - <<'PY'
import json
import os
import pathlib

base_output_dir = pathlib.Path(os.environ["BASE_OUTPUT_DIR"])
resume_output_dir = pathlib.Path(os.environ["RESUME_OUTPUT_DIR"])
expected_dataset_seed = os.environ.get("EXPECTED_DATASET_SEED")
expected_trainable_tensors = os.environ.get("EXPECTED_TRAINABLE_TENSORS")
expected_dtype = os.environ.get("EXPECTED_DTYPE")
expected_dataset_total_samples = os.environ.get("EXPECTED_DATASET_TOTAL_SAMPLES")
expected_dataset_train_samples = os.environ.get("EXPECTED_DATASET_TRAIN_SAMPLES")
expected_dataset_eval_samples = os.environ.get("EXPECTED_DATASET_EVAL_SAMPLES")
expected_dataset_source_files = [
    source.strip()
    for source in os.environ.get("EXPECTED_DATASET_SOURCE_FILES", "").split(",")
    if source.strip()
]
expected_dataset_source_sample_counts = {}
for entry in os.environ.get("EXPECTED_DATASET_SOURCE_SAMPLE_COUNTS", "").split(","):
    entry = entry.strip()
    if not entry:
        continue
    path, sep, samples = entry.rpartition(":")
    if not sep or not path:
        raise SystemExit(f"invalid EXPECTED_DATASET_SOURCE_SAMPLE_COUNTS entry: {entry}")
    expected_dataset_source_sample_counts[path] = int(samples)
expected_dataset_fingerprint = os.environ.get("EXPECTED_DATASET_FINGERPRINT")
expected_trainable_names = [
    name.strip()
    for name in os.environ.get("EXPECTED_TRAINABLE_NAMES", "").split(",")
    if name.strip()
]

base_rank0_paths = sorted(base_output_dir.rglob("qwen-session-dp-rank-0.json"))
if len(base_rank0_paths) != 1:
    raise SystemExit(f"expected one base rank0 summary, found {len(base_rank0_paths)}")
base_rank0 = json.loads(base_rank0_paths[0].read_text())
base_cursor_next = int(base_rank0["data_cursor_next"])
if expected_dtype and base_rank0.get("dtype") != expected_dtype:
    raise SystemExit(
        f"base rank0 dtype {base_rank0.get('dtype')} does not match expected {expected_dtype}"
    )
if expected_trainable_tensors:
    base_trainable_tensors = base_rank0.get("trainable_tensors")
    if not isinstance(base_trainable_tensors, list):
        raise SystemExit("base rank0 trainable_tensors must be a list")
    if len(base_trainable_tensors) != int(expected_trainable_tensors):
        raise SystemExit(
            f"base rank0 expected {expected_trainable_tensors} trainable tensors, got {len(base_trainable_tensors)}"
        )
    for name in expected_trainable_names:
        if name not in base_trainable_tensors:
            raise SystemExit(f"base rank0 missing expected trainable tensor {name}")
if expected_dataset_total_samples and int(base_rank0["dataset_total_samples"]) != int(expected_dataset_total_samples):
    raise SystemExit(
        f"base rank0 dataset_total_samples {base_rank0['dataset_total_samples']} != {expected_dataset_total_samples}"
    )
if expected_dataset_train_samples and int(base_rank0["dataset_train_samples"]) != int(expected_dataset_train_samples):
    raise SystemExit(
        f"base rank0 dataset_train_samples {base_rank0['dataset_train_samples']} != {expected_dataset_train_samples}"
    )
if expected_dataset_eval_samples and int(base_rank0["dataset_eval_samples"]) != int(expected_dataset_eval_samples):
    raise SystemExit(
        f"base rank0 dataset_eval_samples {base_rank0['dataset_eval_samples']} != {expected_dataset_eval_samples}"
    )
if expected_dataset_source_files and base_rank0.get("dataset_source_files") != expected_dataset_source_files:
    raise SystemExit(
        f"base rank0 dataset_source_files {base_rank0.get('dataset_source_files')} != {expected_dataset_source_files}"
    )
if expected_dataset_source_sample_counts:
    base_counts = {
        entry["path"]: entry["samples"]
        for entry in base_rank0.get("dataset_source_sample_counts") or []
    }
    if base_counts != expected_dataset_source_sample_counts:
        raise SystemExit(
            f"base rank0 dataset_source_sample_counts {base_counts} != {expected_dataset_source_sample_counts}"
        )
if expected_dataset_fingerprint and base_rank0.get("dataset_fingerprint") != expected_dataset_fingerprint:
    raise SystemExit(
        f"base rank0 dataset_fingerprint {base_rank0.get('dataset_fingerprint')} != {expected_dataset_fingerprint}"
    )

resume_summaries = sorted(resume_output_dir.rglob("qwen-session-dp-rank-*.json"))
if len(resume_summaries) != 2:
    raise SystemExit(
        f"expected 2 resume rank summaries under {resume_output_dir}, found {len(resume_summaries)}"
    )

evidence = []
for path in resume_summaries:
    data = json.loads(path.read_text())
    if data.get("resumed_checkpoint") is not True:
        raise SystemExit(f"{path} did not report resumed_checkpoint=true")
    if expected_dtype and data.get("dtype") != expected_dtype:
        raise SystemExit(f"{path} dtype {data.get('dtype')} does not match expected {expected_dtype}")
    if not data.get("resume_from"):
        raise SystemExit(f"{path} did not report resume_from")
    if int(data["data_cursor_start"]) != base_cursor_next:
        raise SystemExit(
            f"{path} data_cursor_start {data['data_cursor_start']} did not continue from {base_cursor_next}"
        )
    expected_cursor_next = base_cursor_next + int(data["steps"]) * int(data["local_batch_size"]) * int(data["world_size"])
    if int(data["data_cursor_next"]) != expected_cursor_next:
        raise SystemExit(
            f"{path} expected data_cursor_next {expected_cursor_next}, got {data['data_cursor_next']}"
        )
    if expected_dataset_seed and int(data["dataset_order_seed"]) != int(expected_dataset_seed):
        raise SystemExit(
            f"{path} dataset_order_seed {data['dataset_order_seed']} does not match expected {expected_dataset_seed}"
        )
    if not data.get("dataset_source_files"):
        raise SystemExit(f"{path} dataset_source_files must not be empty")
    if not data.get("dataset_fingerprint"):
        raise SystemExit(f"{path} dataset_fingerprint must not be empty")
    if expected_dataset_total_samples and int(data["dataset_total_samples"]) != int(expected_dataset_total_samples):
        raise SystemExit(
            f"{path} dataset_total_samples {data['dataset_total_samples']} != {expected_dataset_total_samples}"
        )
    if expected_dataset_train_samples and int(data["dataset_train_samples"]) != int(expected_dataset_train_samples):
        raise SystemExit(
            f"{path} dataset_train_samples {data['dataset_train_samples']} != {expected_dataset_train_samples}"
        )
    if expected_dataset_eval_samples and int(data["dataset_eval_samples"]) != int(expected_dataset_eval_samples):
        raise SystemExit(
            f"{path} dataset_eval_samples {data['dataset_eval_samples']} != {expected_dataset_eval_samples}"
        )
    if expected_dataset_source_files and data.get("dataset_source_files") != expected_dataset_source_files:
        raise SystemExit(
            f"{path} dataset_source_files {data.get('dataset_source_files')} != {expected_dataset_source_files}"
        )
    if expected_dataset_source_sample_counts:
        counts = {
            entry["path"]: entry["samples"]
            for entry in data.get("dataset_source_sample_counts") or []
        }
        if counts != expected_dataset_source_sample_counts:
            raise SystemExit(
                f"{path} dataset_source_sample_counts {counts} != {expected_dataset_source_sample_counts}"
            )
    if expected_dataset_fingerprint and data.get("dataset_fingerprint") != expected_dataset_fingerprint:
        raise SystemExit(
            f"{path} dataset_fingerprint {data.get('dataset_fingerprint')} != {expected_dataset_fingerprint}"
        )
    trainable_tensors = data.get("trainable_tensors")
    if expected_trainable_tensors:
        if not isinstance(trainable_tensors, list):
            raise SystemExit(f"{path} trainable_tensors must be a list")
        if len(trainable_tensors) != int(expected_trainable_tensors):
            raise SystemExit(
                f"{path} expected {expected_trainable_tensors} trainable tensors, got {len(trainable_tensors)}"
            )
        if "model.embed_tokens.weight" in trainable_tensors:
            raise SystemExit(f"{path} DP representative path must not train tied embedding")
        for name in expected_trainable_names:
            if name not in trainable_tensors:
                raise SystemExit(f"{path} missing expected trainable tensor {name}")
    for key in ["reload_delta", "next_step_delta", "sharded_reload_delta", "sharded_next_step_delta"]:
        if float(data[key]) > 1e-5:
            raise SystemExit(f"{path} {key} too large: {data[key]}")
    manifest = json.loads(pathlib.Path(data["manifest_output"]).read_text())
    if expected_trainable_tensors:
        if manifest.get("trainable_tensors") != trainable_tensors:
            raise SystemExit(f"{path} manifest trainable_tensors do not match summary")
        if int(manifest.get("tensor_count", -1)) != int(expected_trainable_tensors):
            raise SystemExit(
                f"{path} manifest tensor_count {manifest.get('tensor_count')} != {expected_trainable_tensors}"
            )
        if len(manifest.get("tensors", [])) != int(expected_trainable_tensors):
            raise SystemExit(
                f"{path} expected {expected_trainable_tensors} manifest tensors, got {len(manifest.get('tensors', []))}"
            )
    if int(manifest["data_cursor_start"]) != base_cursor_next:
        raise SystemExit(
            f"{path} manifest data_cursor_start {manifest['data_cursor_start']} did not continue from {base_cursor_next}"
        )
    if int(manifest["data_cursor_next"]) != expected_cursor_next:
        raise SystemExit(
            f"{path} manifest data_cursor_next {manifest['data_cursor_next']} != expected {expected_cursor_next}"
        )
    if expected_dataset_source_files and manifest.get("dataset_source_files") != expected_dataset_source_files:
        raise SystemExit(
            f"{path} manifest dataset_source_files {manifest.get('dataset_source_files')} != {expected_dataset_source_files}"
        )
    if expected_dataset_source_sample_counts:
        manifest_counts = {
            entry["path"]: entry["samples"]
            for entry in manifest.get("dataset_source_sample_counts") or []
        }
        if manifest_counts != expected_dataset_source_sample_counts:
            raise SystemExit(
                f"{path} manifest dataset_source_sample_counts {manifest_counts} != {expected_dataset_source_sample_counts}"
            )
    if expected_dataset_fingerprint and manifest.get("dataset_fingerprint") != expected_dataset_fingerprint:
        raise SystemExit(
            f"{path} manifest dataset_fingerprint {manifest.get('dataset_fingerprint')} != {expected_dataset_fingerprint}"
        )
    sharded_manifest = json.loads(pathlib.Path(data["sharded_global_manifest_output"]).read_text())
    if int(sharded_manifest["data_cursor_next"]) != expected_cursor_next:
        raise SystemExit(
            f"{path} sharded data_cursor_next {sharded_manifest['data_cursor_next']} != expected {expected_cursor_next}"
        )
    if expected_dataset_source_files and sharded_manifest.get("dataset_source_files") != expected_dataset_source_files:
        raise SystemExit(
            f"{path} sharded dataset_source_files {sharded_manifest.get('dataset_source_files')} != {expected_dataset_source_files}"
        )
    if expected_dataset_source_sample_counts:
        sharded_counts = {
            entry["path"]: entry["samples"]
            for entry in sharded_manifest.get("dataset_source_sample_counts") or []
        }
        if sharded_counts != expected_dataset_source_sample_counts:
            raise SystemExit(
                f"{path} sharded dataset_source_sample_counts {sharded_counts} != {expected_dataset_source_sample_counts}"
            )
    if expected_dataset_fingerprint and sharded_manifest.get("dataset_fingerprint") != expected_dataset_fingerprint:
        raise SystemExit(
            f"{path} sharded dataset_fingerprint {sharded_manifest.get('dataset_fingerprint')} != {expected_dataset_fingerprint}"
        )
    evidence.append(
        {
            "rank": data["rank"],
            "resume_from": data["resume_from"],
            "data_cursor_start": data["data_cursor_start"],
            "data_cursor_next": data["data_cursor_next"],
            "dataset_fingerprint": data["dataset_fingerprint"],
            "dtype": data.get("dtype"),
            "trainable_tensors": len(trainable_tensors) if isinstance(trainable_tensors, list) else None,
            "reload_delta": data["reload_delta"],
            "sharded_reload_delta": data["sharded_reload_delta"],
            "sharded_next_step_delta": data["sharded_next_step_delta"],
        }
    )

print(json.dumps({"qwen_session_dp2_resume": evidence}, indent=2, sort_keys=True))
PY
