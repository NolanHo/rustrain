#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

CONFIG="${RUSTRAIN_QWEN_SESSION_SINGLE_SFT_ARROW_EVAL_PATHS_CONFIG:-configs/qwen_session_single_sft_arrow_eval_paths.toml}"
EXPECTED_TRAINABLE_TENSORS="${RUSTRAIN_EXPECTED_QWEN_TRAINABLE_TENSORS:-14}"
DATA_PLAN_OUTPUT="$(mktemp)"
BATCH_PLAN_OUTPUT="$(mktemp)"
BASE_OUTPUT="$(mktemp)"
RESUME_OUTPUT="$(mktemp)"

cargo run -- qwen-sft-streaming-data-plan \
  --config "${CONFIG}" \
  --world-size 1 \
  --data-cursor-start 0 \
  | tee "${DATA_PLAN_OUTPUT}"

cargo run -- qwen-sft-streaming-batch-plan \
  --config "${CONFIG}" \
  --world-size 1 \
  --data-cursor-start 0 \
  | tee "${BATCH_PLAN_OUTPUT}"

cargo run -- train --config "${CONFIG}" | tee "${BASE_OUTPUT}"

BASE_CURSOR_NEXT="$(
  python - "${BASE_OUTPUT}" <<'PY'
import pathlib
import sys

values = {}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if ": " in line:
        key, value = line.split(": ", 1)
        values[key] = value
manifest = values.get("manifest_output")
cursor_next = values.get("data_cursor_next")
if manifest is None:
    raise SystemExit("base run did not print manifest_output")
if cursor_next is None:
    raise SystemExit("base run did not print data_cursor_next")
print(f"{manifest}\t{cursor_next}")
PY
)"
MANIFEST_OUTPUT="${BASE_CURSOR_NEXT%%$'\t'*}"
BASE_DATA_CURSOR_NEXT="${BASE_CURSOR_NEXT##*$'\t'}"

cargo run -- train --config "${CONFIG}" --resume-from "${MANIFEST_OUTPUT}" \
  | tee "${RESUME_OUTPUT}"

python - "${DATA_PLAN_OUTPUT}" "${BATCH_PLAN_OUTPUT}" "${RESUME_OUTPUT}" "${BASE_DATA_CURSOR_NEXT}" "${EXPECTED_TRAINABLE_TENSORS}" <<'PY'
import ast
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from qwen_verify_utils import require_complete_qwen_base_model_path

data_plan_text = pathlib.Path(sys.argv[1]).read_text()
start = data_plan_text.find("{")
end = data_plan_text.rfind("}")
if start < 0 or end < start:
    raise SystemExit(f"data plan output did not contain JSON: {data_plan_text}")
data_plan = json.loads(data_plan_text[start : end + 1])

batch_plan_text = pathlib.Path(sys.argv[2]).read_text()
start = batch_plan_text.find("{")
end = batch_plan_text.rfind("}")
if start < 0 or end < start:
    raise SystemExit(f"batch plan output did not contain JSON: {batch_plan_text}")
batch_plan = json.loads(batch_plan_text[start : end + 1])

resume_values = {}
for line in pathlib.Path(sys.argv[3]).read_text().splitlines():
    if ": " not in line:
        continue
    key, value = line.split(": ", 1)
    resume_values[key] = value

base_data_cursor_next = int(sys.argv[4])
expected_trainable_tensors = int(sys.argv[5])

expected_path = data_plan["data_paths"][0]
if data_plan["eval_paths"] != [expected_path]:
    raise SystemExit(f"expected eval_paths to mirror train Arrow path, got {data_plan['eval_paths']}")
if data_plan["max_samples"] != 24 or data_plan["max_eval_samples"] != 8:
    raise SystemExit(
        f"expected max/max_eval samples 24/8, got {data_plan['max_samples']}/{data_plan['max_eval_samples']}"
    )
if data_plan["dataset_total_samples"] != 32:
    raise SystemExit(f"expected data-plan total samples 32, got {data_plan['dataset_total_samples']}")
if data_plan["dataset_train_samples"] != 24 or data_plan["dataset_eval_samples"] != 8:
    raise SystemExit(
        f"expected explicit Arrow eval split 24/8, got {data_plan['dataset_train_samples']}/{data_plan['dataset_eval_samples']}"
    )
if data_plan["dataset_source_files"] != [expected_path]:
    raise SystemExit(f"expected one merged source file, got {data_plan['dataset_source_files']}")
if data_plan["dataset_source_sample_counts"] != [{"path": expected_path, "samples": 32}]:
    raise SystemExit(f"unexpected merged source sample counts: {data_plan['dataset_source_sample_counts']}")
if data_plan["train_source_sample_counts"] != [{"path": expected_path, "samples": 24}]:
    raise SystemExit(f"unexpected train source sample counts: {data_plan['train_source_sample_counts']}")
if data_plan["eval_source_sample_counts"] != [{"path": expected_path, "samples": 8}]:
    raise SystemExit(f"unexpected eval source sample counts: {data_plan['eval_source_sample_counts']}")
if not data_plan["train_fingerprint"] or not data_plan["eval_fingerprint"]:
    raise SystemExit("train/eval fingerprints must be present")
if data_plan["dataset_fingerprint"] in [data_plan["train_fingerprint"], data_plan["eval_fingerprint"]]:
    raise SystemExit("combined dataset fingerprint must differ from train/eval fingerprints")
if data_plan["streaming_index_cache_path"] is not None:
    raise SystemExit("instruction_arrow data plan must not report an index cache path")

for key in [
    "dataset_total_samples",
    "dataset_train_samples",
    "dataset_eval_samples",
    "dataset_source_files",
    "dataset_source_sample_counts",
    "dataset_fingerprint",
    "dataset_order_seed",
    "dataset_shuffle",
]:
    if batch_plan[key] != data_plan[key]:
        raise SystemExit(f"batch-plan {key} mismatch: {batch_plan[key]} vs {data_plan[key]}")
if batch_plan["streaming_window_samples"] != 3:
    raise SystemExit(f"expected 3 streaming window samples, got {batch_plan['streaming_window_samples']}")
if batch_plan["streaming_raw_sample_indices"] != []:
    raise SystemExit(f"Arrow batch plan must not report JSONL raw indices, got {batch_plan['streaming_raw_sample_indices']}")
if batch_plan["streaming_index_cache_path"] is not None:
    raise SystemExit("instruction_arrow batch plan must not report an index cache path")
if batch_plan["materialized_input_max_delta"] != 0 or float(batch_plan["materialized_mask_max_delta"]) != 0.0:
    raise SystemExit(
        f"batch plan materialized deltas must be zero, got input={batch_plan['materialized_input_max_delta']} mask={batch_plan['materialized_mask_max_delta']}"
    )
if len(batch_plan["batch_token_fingerprints"]) != 3:
    raise SystemExit(f"expected 3 batch token fingerprints, got {batch_plan['batch_token_fingerprints']}")

required = [
    "resumed_checkpoint",
    "compute_kind",
    "train_steps",
    "step_losses",
    "first_step_grad_norm",
    "final_step_grad_norm",
    "dataset_total_samples",
    "dataset_train_samples",
    "dataset_eval_samples",
    "dataset_source_files",
    "dataset_source_sample_counts",
    "dataset_fingerprint",
    "dataset_order_seed",
    "dataset_shuffle",
    "streaming_train_batches",
    "data_cursor_start",
    "data_cursor_end",
    "data_cursor_next",
    "data_epoch_start",
    "data_epoch_end",
    "data_epoch_next",
    "data_sample_offset_start",
    "data_sample_offset_end",
    "data_sample_offset_next",
    "batch_size",
    "sequence_tokens",
    "reload_delta",
    "second_step_delta",
    "trainable_tensors",
    "manifest_output",
]
missing = [key for key in required if key not in resume_values]
if missing:
    raise SystemExit(f"resume run is missing fields: {missing}")
if resume_values["resumed_checkpoint"] != "true":
    raise SystemExit(f"expected resumed checkpoint, got {resume_values['resumed_checkpoint']}")
if resume_values["compute_kind"] != "fp32":
    raise SystemExit(f"expected fp32 compute, got {resume_values['compute_kind']}")
if resume_values["streaming_train_batches"] != "true":
    raise SystemExit(f"expected streaming_train_batches true, got {resume_values['streaming_train_batches']}")
if "dataset_total_tokens" in resume_values:
    raise SystemExit(
        "instruction_arrow trainer runtime must not report dataset_total_tokens; "
        "full token totals are reserved for materialized batch-plan parity"
    )
if int(resume_values["train_steps"]) != 2:
    raise SystemExit(f"expected train_steps 2, got {resume_values['train_steps']}")
if int(resume_values["trainable_tensors"]) != expected_trainable_tensors:
    raise SystemExit(
        f"expected {expected_trainable_tensors} trainable tensors, got {resume_values['trainable_tensors']}"
    )
step_losses = ast.literal_eval(resume_values["step_losses"])
if len(step_losses) != 3 or not all(math.isfinite(float(loss)) for loss in step_losses):
    raise SystemExit(f"expected 3 finite step losses, got {step_losses}")
for key in ["first_step_grad_norm", "final_step_grad_norm"]:
    if float(resume_values[key]) <= 0.0:
        raise SystemExit(f"{key} must be positive, got {resume_values[key]}")
if int(resume_values["dataset_total_samples"]) != 32:
    raise SystemExit(f"expected total samples 32, got {resume_values['dataset_total_samples']}")
if int(resume_values["dataset_train_samples"]) != 24 or int(resume_values["dataset_eval_samples"]) != 8:
    raise SystemExit(
        f"expected 24/8 train/eval samples, got {resume_values['dataset_train_samples']}/{resume_values['dataset_eval_samples']}"
    )
dataset_source_files = ast.literal_eval(resume_values["dataset_source_files"])
if dataset_source_files != [expected_path]:
    raise SystemExit(f"expected merged Arrow source file, got {dataset_source_files}")
source_counts_text = resume_values["dataset_source_sample_counts"]
if source_counts_text.count("QwenSftSourceSampleCount") != 1 or "samples: 32" not in source_counts_text:
    raise SystemExit(f"expected merged Arrow source sample count 32, got {source_counts_text}")
if resume_values["dataset_fingerprint"] != data_plan["dataset_fingerprint"]:
    raise SystemExit("resume dataset fingerprint does not match data plan")
if int(resume_values["dataset_order_seed"]) != 777:
    raise SystemExit(f"expected dataset_order_seed 777, got {resume_values['dataset_order_seed']}")
if resume_values["dataset_shuffle"] != "false":
    raise SystemExit(f"expected dataset_shuffle false, got {resume_values['dataset_shuffle']}")
if int(resume_values["data_cursor_start"]) != base_data_cursor_next:
    raise SystemExit(
        f"resume cursor {resume_values['data_cursor_start']} did not continue from base cursor {base_data_cursor_next}"
    )
expected_cursor_end = int(resume_values["data_cursor_start"]) + int(resume_values["train_steps"]) * int(resume_values["batch_size"])
if int(resume_values["data_cursor_end"]) != expected_cursor_end:
    raise SystemExit(f"expected data_cursor_end {expected_cursor_end}, got {resume_values['data_cursor_end']}")
if int(resume_values["data_cursor_next"]) != int(resume_values["data_cursor_end"]):
    raise SystemExit("data_cursor_next must equal data_cursor_end")
train_samples = int(resume_values["dataset_train_samples"])
for cursor_key, epoch_key, offset_key in [
    ("data_cursor_start", "data_epoch_start", "data_sample_offset_start"),
    ("data_cursor_end", "data_epoch_end", "data_sample_offset_end"),
    ("data_cursor_next", "data_epoch_next", "data_sample_offset_next"),
]:
    cursor = int(resume_values[cursor_key])
    if int(resume_values[epoch_key]) != cursor // train_samples:
        raise SystemExit(f"{epoch_key} mismatch for {cursor_key}={cursor}")
    if int(resume_values[offset_key]) != cursor % train_samples:
        raise SystemExit(f"{offset_key} mismatch for {cursor_key}={cursor}")
if float(resume_values["reload_delta"]) > 1e-5:
    raise SystemExit(f"reload_delta too large: {resume_values['reload_delta']}")
if float(resume_values["second_step_delta"]) > 1e-5:
    raise SystemExit(f"second_step_delta too large: {resume_values['second_step_delta']}")

manifest_path = pathlib.Path(resume_values["manifest_output"])
manifest = json.loads(manifest_path.read_text())
require_complete_qwen_base_model_path(manifest, manifest_path)
if manifest.get("dataset_source_files") != [expected_path]:
    raise SystemExit(f"manifest dataset_source_files mismatch: {manifest.get('dataset_source_files')}")
if manifest.get("dataset_source_sample_counts") != [{"path": expected_path, "samples": 32}]:
    raise SystemExit(f"manifest source counts mismatch: {manifest.get('dataset_source_sample_counts')}")
if manifest.get("dataset_fingerprint") != data_plan["dataset_fingerprint"]:
    raise SystemExit("manifest dataset fingerprint does not match data plan")

print(
    "qwen_session_single_sft_arrow_eval_paths_verified: "
    + json.dumps(
        {
            "dataset_total_samples": int(resume_values["dataset_total_samples"]),
            "dataset_train_samples": int(resume_values["dataset_train_samples"]),
            "dataset_eval_samples": int(resume_values["dataset_eval_samples"]),
            "data_cursor_next": int(resume_values["data_cursor_next"]),
            "reload_delta": float(resume_values["reload_delta"]),
            "second_step_delta": float(resume_values["second_step_delta"]),
        },
        sort_keys=True,
    )
)
PY
