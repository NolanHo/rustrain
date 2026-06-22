#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

BASE_OUTPUT="$(mktemp)"
RESUME_OUTPUT="$(mktemp)"
DIRECT_OUTPUT="$(mktemp)"
CONFIG="${RUSTRAIN_QWEN_LORA_SFT_CONFIG:-configs/qwen_lora_sft.toml}"
EXPECTED_COMPUTE_KIND="${RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND:-}"

cargo run -- train --config "${CONFIG}" | tee "${BASE_OUTPUT}"

BASE_RESUME_INPUTS="$(
  python - "${BASE_OUTPUT}" <<'PY'
import pathlib
import sys

values = {}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if ": " in line:
        key, value = line.split(": ", 1)
        values[key] = value
manifest = values.get("adapter_manifest")
adapter = values.get("adapter_checkpoint")
cursor_next = values.get("data_cursor_next")
if manifest is None:
    raise SystemExit("base run did not print adapter_manifest")
if adapter is None:
    raise SystemExit("base run did not print adapter_checkpoint")
if cursor_next is None:
    raise SystemExit("base run did not print data_cursor_next")
print(f"{manifest}\t{adapter}\t{cursor_next}")
PY
)"
IFS=$'\t' read -r ADAPTER_MANIFEST ADAPTER_CHECKPOINT BASE_DATA_CURSOR_NEXT <<<"${BASE_RESUME_INPUTS}"

cargo run -- train --config "${CONFIG}" --resume-from "${ADAPTER_MANIFEST}" \
  | tee "${RESUME_OUTPUT}"

if [ "${EXPECTED_COMPUTE_KIND}" != "bf16" ]; then
  cargo run -- train --config "${CONFIG}" --resume-from "${ADAPTER_CHECKPOINT}" \
    | tee "${DIRECT_OUTPUT}"
else
  echo "Skipping direct .safetensors resume continuation for bf16; manifest-backed bf16 resume covers dtype identity."
fi

python - "${RESUME_OUTPUT}" "${EXPECTED_COMPUTE_KIND}" "${BASE_DATA_CURSOR_NEXT}" <<'PY'
import ast
import json
import math
import pathlib
import re
import sys

def parse_source_sample_counts(text):
    entries = re.findall(r'QwenSftSourceSampleCount \{ path: "([^"]+)", samples: (\d+) \}', text)
    if not entries:
        raise SystemExit(f"dataset_source_sample_counts did not contain parseable entries: {text}")
    return [{"path": path, "samples": int(samples)} for path, samples in entries]

def verify_source_sample_counts(counts, dataset_source_files, dataset_total_samples, context):
    if not isinstance(counts, list) or not counts:
        raise SystemExit(f"{context} dataset_source_sample_counts must be a non-empty list")
    count_paths = []
    total = 0
    for entry in counts:
        if not isinstance(entry, dict):
            raise SystemExit(f"{context} dataset_source_sample_counts entry must be an object: {entry}")
        path = entry.get("path")
        samples = entry.get("samples")
        if not path:
            raise SystemExit(f"{context} dataset_source_sample_counts entry is missing path: {entry}")
        if samples is None or int(samples) <= 0:
            raise SystemExit(f"{context} dataset_source_sample_counts entry must have positive samples: {entry}")
        count_paths.append(path)
        total += int(samples)
    if count_paths != dataset_source_files:
        raise SystemExit(
            f"{context} dataset_source_sample_counts paths {count_paths} do not match dataset_source_files {dataset_source_files}"
        )
    if total != int(dataset_total_samples):
        raise SystemExit(
            f"{context} dataset_source_sample_counts total {total} does not match dataset_total_samples {dataset_total_samples}"
        )
    return counts

base_data_cursor_next = int(sys.argv[3])
values = {}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if ": " not in line:
        continue
    key, value = line.split(": ", 1)
    values[key] = value

required = [
    "compute_kind",
    "resume_from",
    "resumed_adapter",
    "adapter_manifest",
    "dataset_total_samples",
    "dataset_total_tokens",
    "dataset_response_tokens",
    "dataset_masked_positions",
    "dataset_max_sequence_tokens",
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
    "global_batch_size",
    "gradient_accumulation_steps",
    "initial_loss",
    "final_loss",
    "reload_delta",
    "eval_reload_delta",
    "full_forward_reload_delta",
    "full_forward_merge_delta",
    "full_forward_unmerge_delta",
    "full_generate_reload_match",
    "full_generate_merge_match",
    "full_generate_new_token_ids",
    "first_step_grad_norm",
    "final_step_grad_norm",
    "final_step_clipped_grad_norm",
    "tokens_per_second",
    "samples_per_second",
]
missing = [key for key in required if key not in values]
if missing:
    raise SystemExit(f"resume run is missing fields: {missing}")
expected_compute_kind = sys.argv[2]
if expected_compute_kind and values["compute_kind"] != expected_compute_kind:
    raise SystemExit(
        f"compute_kind {values['compute_kind']} does not match expected {expected_compute_kind}"
    )
if values["resumed_adapter"] != "true":
    raise SystemExit("resume run did not report resumed_adapter: true")
if values["streaming_train_batches"] != "true":
    raise SystemExit(
        f"resume run did not report streaming_train_batches: true, got {values['streaming_train_batches']}"
    )
for key in [
    "dataset_total_samples",
    "dataset_total_tokens",
    "dataset_response_tokens",
    "dataset_masked_positions",
    "dataset_max_sequence_tokens",
]:
    if int(values[key]) <= 0:
        raise SystemExit(f"{key} must be positive, got {values[key]}")
dataset_source_files = ast.literal_eval(values["dataset_source_files"])
if not dataset_source_files:
    raise SystemExit("dataset_source_files must not be empty")
if not all(str(path).endswith(".jsonl") for path in dataset_source_files):
    raise SystemExit(f"dataset_source_files must only contain JSONL paths, got {dataset_source_files}")
dataset_source_sample_counts = verify_source_sample_counts(
    parse_source_sample_counts(values["dataset_source_sample_counts"]),
    dataset_source_files,
    values["dataset_total_samples"],
    "resume stdout",
)
if not values["dataset_fingerprint"]:
    raise SystemExit("dataset_fingerprint must not be empty")
adapter_manifest = json.loads(pathlib.Path(values["adapter_manifest"]).read_text())
if adapter_manifest.get("compute_kind") != values["compute_kind"]:
    raise SystemExit(
        f"adapter manifest compute_kind {adapter_manifest.get('compute_kind')} did not match summary {values['compute_kind']}"
    )
if adapter_manifest.get("dataset_source_files") != dataset_source_files:
    raise SystemExit(
        f"adapter manifest dataset_source_files {adapter_manifest.get('dataset_source_files')} did not match summary {dataset_source_files}"
    )
if adapter_manifest.get("dataset_source_sample_counts") != dataset_source_sample_counts:
    raise SystemExit(
        f"adapter manifest dataset_source_sample_counts {adapter_manifest.get('dataset_source_sample_counts')} did not match summary {dataset_source_sample_counts}"
    )
verify_source_sample_counts(
    adapter_manifest.get("dataset_source_sample_counts"),
    dataset_source_files,
    values["dataset_total_samples"],
    "adapter manifest",
)
if adapter_manifest.get("dataset_fingerprint") != values["dataset_fingerprint"]:
    raise SystemExit(
        f"adapter manifest dataset_fingerprint {adapter_manifest.get('dataset_fingerprint')} did not match summary {values['dataset_fingerprint']}"
    )
if str(adapter_manifest.get("dataset_shuffle")).lower() != values["dataset_shuffle"]:
    raise SystemExit(
        f"adapter manifest dataset_shuffle {adapter_manifest.get('dataset_shuffle')} did not match summary {values['dataset_shuffle']}"
    )
if adapter_manifest.get("streaming_train_batches") is not True:
    raise SystemExit(
        f"adapter manifest streaming_train_batches {adapter_manifest.get('streaming_train_batches')} is not true"
    )
expected_target_layers = [0, 1]
expected_target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
if adapter_manifest.get("target_layers") != expected_target_layers:
    raise SystemExit(
        f"adapter manifest target_layers {adapter_manifest.get('target_layers')} != {expected_target_layers}"
    )
if adapter_manifest.get("target_modules") != expected_target_modules:
    raise SystemExit(
        f"adapter manifest target_modules {adapter_manifest.get('target_modules')} != {expected_target_modules}"
    )
for key in ["initial_loss", "final_loss"]:
    if not math.isfinite(float(values[key])):
        raise SystemExit(f"{key} must be finite, got {values[key]}")
if int(values["data_cursor_start"]) != base_data_cursor_next:
    raise SystemExit(
        f"resume data_cursor_start {values['data_cursor_start']} did not continue from base data_cursor_next {base_data_cursor_next}"
    )
expected_cursor_end = int(values["data_cursor_start"]) + int(values["steps"]) * int(values["global_batch_size"])
if int(values["data_cursor_end"]) != expected_cursor_end:
    raise SystemExit(
        f"expected data_cursor_end {expected_cursor_end}, got {values['data_cursor_end']}"
    )
if int(values["data_cursor_next"]) != expected_cursor_end:
    raise SystemExit(
        f"expected data_cursor_next {expected_cursor_end}, got {values['data_cursor_next']}"
    )
train_samples = int(values["train_samples"])
for cursor_key, epoch_key, offset_key in [
    ("data_cursor_start", "data_epoch_start", "data_sample_offset_start"),
    ("data_cursor_end", "data_epoch_end", "data_sample_offset_end"),
    ("data_cursor_next", "data_epoch_next", "data_sample_offset_next"),
]:
    cursor = int(values[cursor_key])
    expected_epoch = cursor // train_samples
    expected_offset = cursor % train_samples
    if int(values[epoch_key]) != expected_epoch:
        raise SystemExit(
            f"expected {epoch_key} {expected_epoch}, got {values[epoch_key]} from {cursor_key}={cursor}"
        )
    if int(values[offset_key]) != expected_offset:
        raise SystemExit(
            f"expected {offset_key} {expected_offset}, got {values[offset_key]} from {cursor_key}={cursor}"
        )
for key in [
    "reload_delta",
    "eval_reload_delta",
    "full_forward_reload_delta",
]:
    if float(values[key]) > 1e-6:
        raise SystemExit(f"{key} too large: {values[key]}")
merge_tolerance = 5.0 if values["compute_kind"] == "bf16" else 1e-6
if float(values["full_forward_merge_delta"]) > merge_tolerance:
    raise SystemExit(
        f"full_forward_merge_delta too large: {values['full_forward_merge_delta']}"
    )
unmerge_tolerance = 5.0 if values["compute_kind"] == "bf16" else 1e-3
if float(values["full_forward_unmerge_delta"]) > unmerge_tolerance:
    raise SystemExit(
        f"full_forward_unmerge_delta too large: {values['full_forward_unmerge_delta']}"
    )
if values["full_generate_reload_match"] != "true":
    raise SystemExit(
        f"full_generate_reload_match must be true, got {values['full_generate_reload_match']}"
    )
if values["compute_kind"] != "bf16" and values["full_generate_merge_match"] != "true":
    raise SystemExit(
        f"full_generate_merge_match must be true, got {values['full_generate_merge_match']}"
    )
for key in [
    "first_step_grad_norm",
    "final_step_grad_norm",
    "final_step_clipped_grad_norm",
    "tokens_per_second",
    "samples_per_second",
]:
    if float(values[key]) <= 0.0:
        raise SystemExit(f"{key} must be positive, got {values[key]}")
generated = ast.literal_eval(values["full_generate_new_token_ids"])
if not generated:
    raise SystemExit("full_generate_new_token_ids must not be empty")

print(
    "qwen_lora_sft_resume_verified: "
    f"resume_from={values['resume_from']} "
    f"data_cursor_start={values['data_cursor_start']} "
    f"data_cursor_next={values['data_cursor_next']} "
    f"data_epoch_next={values['data_epoch_next']} "
    f"data_sample_offset_next={values['data_sample_offset_next']} "
    f"initial_loss={values['initial_loss']} "
    f"final_loss={values['final_loss']} "
    f"reload_delta={values['reload_delta']} "
    f"tokens_per_second={values['tokens_per_second']} "
    f"samples_per_second={values['samples_per_second']}"
)
PY

if [ "${EXPECTED_COMPUTE_KIND}" != "bf16" ]; then
python - "${DIRECT_OUTPUT}" "${EXPECTED_COMPUTE_KIND}" "${ADAPTER_CHECKPOINT}" <<'PY'
import math
import pathlib
import sys

values = {}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if ": " not in line:
        continue
    key, value = line.split(": ", 1)
    values[key] = value

required = [
    "compute_kind",
    "resume_from",
    "resumed_adapter",
    "adapter_manifest",
    "streaming_train_batches",
    "data_cursor_start",
    "data_cursor_next",
    "initial_loss",
    "final_loss",
    "reload_delta",
    "eval_reload_delta",
    "full_forward_reload_delta",
    "full_generate_reload_match",
    "tokens_per_second",
    "samples_per_second",
]
missing = [key for key in required if key not in values]
if missing:
    raise SystemExit(f"direct adapter resume run is missing fields: {missing}")
expected_compute_kind = sys.argv[2]
adapter_checkpoint = sys.argv[3]
if expected_compute_kind and values["compute_kind"] != expected_compute_kind:
    raise SystemExit(
        f"direct resume compute_kind {values['compute_kind']} does not match expected {expected_compute_kind}"
    )
if values["resume_from"] != adapter_checkpoint:
    raise SystemExit(
        f"direct resume_from {values['resume_from']} did not match adapter checkpoint {adapter_checkpoint}"
    )
if values["resumed_adapter"] != "true":
    raise SystemExit("direct resume did not report resumed_adapter: true")
if values["streaming_train_batches"] != "true":
    raise SystemExit(
        f"direct resume did not report streaming_train_batches: true, got {values['streaming_train_batches']}"
    )
if int(values["data_cursor_start"]) != 0:
    raise SystemExit(
        f"direct .safetensors resume should not claim manifest cursor continuity, got data_cursor_start={values['data_cursor_start']}"
    )
if int(values["data_cursor_next"]) <= 0:
    raise SystemExit(f"direct resume data_cursor_next must advance, got {values['data_cursor_next']}")
if not pathlib.Path(values["adapter_manifest"]).is_file():
    raise SystemExit(f"direct resume did not write adapter manifest {values['adapter_manifest']}")
for key in ["initial_loss", "final_loss", "tokens_per_second", "samples_per_second"]:
    value = float(values[key])
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"{key} must be positive and finite, got {values[key]}")
if float(values["final_loss"]) >= float(values["initial_loss"]):
    raise SystemExit(
        f"direct resume did not reduce loss: initial={values['initial_loss']} final={values['final_loss']}"
    )
for key in ["reload_delta", "eval_reload_delta", "full_forward_reload_delta"]:
    if float(values[key]) > 1e-6:
        raise SystemExit(f"direct resume {key} too large: {values[key]}")
if values["full_generate_reload_match"] != "true":
    raise SystemExit(
        f"direct resume full_generate_reload_match must be true, got {values['full_generate_reload_match']}"
    )

print(
    "qwen_lora_sft_direct_resume_verified: "
    f"resume_from={values['resume_from']} "
    f"data_cursor_start={values['data_cursor_start']} "
    f"data_cursor_next={values['data_cursor_next']} "
    f"initial_loss={values['initial_loss']} "
    f"final_loss={values['final_loss']} "
    f"reload_delta={values['reload_delta']}"
)
PY
fi
