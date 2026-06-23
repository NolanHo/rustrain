#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

BASE_CONFIG="${RUSTRAIN_QWEN_SESSION_DP2_SFT_ARROW_HF_MIXED_SCHEMA_CONFIG:-configs/qwen_session_dp2_sft_arrow.toml}"
ALPACA_ARROW="${RUSTRAIN_HF_SFT_ALPACA_ARROW:-/vePFS-Mindverse/share/huggingface/datasets/iamtarun___code_instructions_120k_alpaca/default/0.0.0/31f725b2d714c1b4f038e80fbaa6b977870a50b7/code_instructions_120k_alpaca-train.arrow}"
QA_ARROW="${RUSTRAIN_HF_SFT_QA_ARROW:-/vePFS-Mindverse/share/huggingface/datasets/e53f048856ff4f594e959d75785d2c2d37b678ee/main/0.0.0/62d07aa71a8777a4/e53f048856ff4f594e959d75785d2c2d37b678ee-train.arrow}"
ALPACA_MIN_SAMPLES="${RUSTRAIN_HF_SFT_ALPACA_ARROW_MIN_SAMPLES:-100000}"
QA_MIN_SAMPLES="${RUSTRAIN_HF_SFT_QA_ARROW_MIN_SAMPLES:-1000}"
TRAIN_STEPS="${RUSTRAIN_HF_SFT_MIXED_SCHEMA_ARROW_TRAIN_STEPS:-1}"
TRAINABLE_LAYERS="${RUSTRAIN_HF_SFT_MIXED_SCHEMA_ARROW_TRAINABLE_LAYERS:-0}"
RUN_DIR="$(mktemp -d /tmp/rustrain-dp2-arrow-hf-mixed-schema-cache-XXXXXX)"
CONFIG="${RUN_DIR}/config.toml"
FIRST_OUTPUT_DIR="${RUN_DIR}/first"
SECOND_OUTPUT_DIR="${RUN_DIR}/second"
CACHE_PATH="${RUN_DIR}/arrow-row-index.json"
MASTER_PORT="${RUSTRAIN_QWEN_SESSION_DP2_SFT_ARROW_HF_MIXED_SCHEMA_PORT:-29649}"

if [ ! -f "${ALPACA_ARROW}" ]; then
  echo "expected HF Alpaca Arrow cache file to exist: ${ALPACA_ARROW}" >&2
  exit 1
fi
if [ ! -f "${QA_ARROW}" ]; then
  echo "expected HF QA Arrow cache file to exist: ${QA_ARROW}" >&2
  exit 1
fi

python - "${BASE_CONFIG}" "${CONFIG}" "${ALPACA_ARROW}" "${QA_ARROW}" "${CACHE_PATH}" "${TRAIN_STEPS}" "${TRAINABLE_LAYERS}" <<'PY'
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
alpaca_path = pathlib.Path(sys.argv[3]).as_posix()
qa_path = pathlib.Path(sys.argv[4]).as_posix()
cache_path = pathlib.Path(sys.argv[5]).as_posix()
train_steps = int(sys.argv[6])
trainable_layers = [
    int(value.strip())
    for value in sys.argv[7].split(",")
    if value.strip()
]
lines = []
in_data = False
in_train = False
in_model = False
for line in source.read_text().splitlines():
    stripped = line.strip()
    if line.startswith("[") and stripped != "[data]":
        in_data = False
    if stripped == "[train]":
        in_train = True
        in_model = False
        lines.append(line)
        continue
    if stripped == "[model]":
        in_model = True
        in_train = False
        lines.append(line)
        continue
    if stripped == "[data]":
        in_data = True
        in_train = False
        in_model = False
        lines.append(line)
        continue
    if line.startswith("["):
        in_train = False
        in_model = False
    if in_train and line.startswith("max_steps = "):
        lines.append(f"max_steps = {train_steps}")
        continue
    if in_model and line.startswith("trainable_layers = "):
        layers = ", ".join(str(layer) for layer in trainable_layers)
        lines.append(f"trainable_layers = [{layers}]")
        continue
    if in_data and line.startswith("max_samples = "):
        continue
    if in_data and line.startswith("paths = "):
        lines.append(f'paths = ["{alpaca_path}", "{qa_path}"]')
        lines.append(f'index_cache = "{cache_path}"')
        lines.append('source_instruction_fields = ["instruction", "question"]')
        lines.append('source_input_fields = ["input", ""]')
        lines.append('source_response_fields = ["output", "answer"]')
        continue
    lines.append(line)
target.write_text("\n".join(lines) + "\n")
PY

FIRST_STARTED="$(date +%s)"
cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${FIRST_OUTPUT_DIR}" \
  --master-port "${MASTER_PORT}" \
  train --config "${CONFIG}"
FIRST_ELAPSED="$(( "$(date +%s)" - FIRST_STARTED ))"

SECOND_STARTED="$(date +%s)"
cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir "${SECOND_OUTPUT_DIR}" \
  --master-port "${MASTER_PORT}" \
  train --config "${CONFIG}"
SECOND_ELAPSED="$(( "$(date +%s)" - SECOND_STARTED ))"

python - \
  "${FIRST_OUTPUT_DIR}" \
  "${SECOND_OUTPUT_DIR}" \
  "${CACHE_PATH}" \
  "${ALPACA_ARROW}" \
  "${QA_ARROW}" \
  "${ALPACA_MIN_SAMPLES}" \
  "${QA_MIN_SAMPLES}" \
  "${TRAIN_STEPS}" \
  "${TRAINABLE_LAYERS}" \
  "${FIRST_ELAPSED}" \
  "${SECOND_ELAPSED}" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from qwen_verify_utils import (
    require_complete_qwen_base_model_path,
    require_complete_qwen_manifest_paths,
)

first_dir = pathlib.Path(sys.argv[1])
second_dir = pathlib.Path(sys.argv[2])
cache_base = pathlib.Path(sys.argv[3])
expected_arrows = [
    pathlib.Path(sys.argv[4]).as_posix(),
    pathlib.Path(sys.argv[5]).as_posix(),
]
source_min_samples = [int(sys.argv[6]), int(sys.argv[7])]
expected_steps = int(sys.argv[8])
expected_trainable_layers = [
    int(value.strip())
    for value in sys.argv[9].split(",")
    if value.strip()
]
first_elapsed_secs = int(sys.argv[10])
second_elapsed_secs = int(sys.argv[11])


def rank_cache(rank: int) -> pathlib.Path:
    return cache_base.with_name(f"{cache_base.stem}.rank-{rank}{cache_base.suffix}")


def summaries(output_dir: pathlib.Path) -> dict[int, tuple[pathlib.Path, dict]]:
    paths = sorted(output_dir.rglob("qwen-session-dp-rank-*.json"))
    if len(paths) != 2:
        raise SystemExit(f"expected 2 DP rank summaries under {output_dir}, found {len(paths)}")
    parsed = {}
    for path in paths:
        data = json.loads(path.read_text())
        parsed[int(data["rank"])] = (path, data)
    return parsed


first = summaries(first_dir)
second = summaries(second_dir)

reference_total = None
reference_train = None
reference_eval = None
reference_fingerprint = None
evidence = []
for rank in [0, 1]:
    expected_cache = rank_cache(rank)
    if not expected_cache.exists():
        raise SystemExit(f"rank {rank} cache was not written: {expected_cache}")

    for context, item, expected_hit, expected_written in [
        ("first", first[rank], False, True),
        ("second", second[rank], True, False),
    ]:
        path, data = item
        required = [
            "rank",
            "world_size",
            "dtype",
            "data_kind",
            "steps",
            "local_batch_size",
            "sequence_tokens",
            "tokens_per_second",
            "samples_per_second",
            "memory_rss_mb",
            "gpu_memory_allocated_mb",
            "dataset_total_samples",
            "dataset_train_samples",
            "dataset_eval_samples",
            "dataset_source_files",
            "dataset_source_sample_counts",
            "dataset_fingerprint",
            "dataset_order_seed",
            "dataset_shuffle",
            "streaming_train_batches",
            "streaming_index_cache_path",
            "streaming_index_cache_hit",
            "streaming_index_cache_written",
            "data_cursor_start",
            "data_cursor_end",
            "data_cursor_next",
            "data_epoch_start",
            "data_epoch_end",
            "data_epoch_next",
            "data_sample_offset_start",
            "data_sample_offset_end",
            "data_sample_offset_next",
            "reload_delta",
            "next_step_delta",
            "sharded_reload_delta",
            "sharded_next_step_delta",
            "sharded_global_manifest_output",
        ]
        missing = [key for key in required if data.get(key) is None]
        if missing:
            raise SystemExit(f"{path} is missing fields: {missing}")
        if data.get("dataset_total_tokens") is not None:
            raise SystemExit(f"{path} instruction_arrow trainer runtime must not report dataset_total_tokens")
        if data["data_kind"] != "instruction_arrow":
            raise SystemExit(f"{path} expected instruction_arrow, got {data['data_kind']}")
        if int(data["world_size"]) != 2:
            raise SystemExit(f"{path} expected world_size 2, got {data['world_size']}")
        if int(data["steps"]) != expected_steps or int(data["local_batch_size"]) != 1:
            raise SystemExit(f"{path} expected {expected_steps} DP step(s) with local batch 1, got steps={data['steps']} local={data['local_batch_size']}")
        trainable_tensors = data.get("trainable_tensors") or []
        for layer in expected_trainable_layers:
            for suffix in ["self_attn.q_proj.weight", "mlp.down_proj.weight"]:
                expected_name = f"model.layers.{layer}.{suffix}"
                if expected_name not in trainable_tensors:
                    raise SystemExit(f"{path} missing trainable tensor {expected_name}: {trainable_tensors}")
        if int(data["sequence_tokens"]) <= 0:
            raise SystemExit(f"{path} expected positive sequence_tokens, got {data['sequence_tokens']}")
        if float(data["tokens_per_second"]) <= 0.0:
            raise SystemExit(f"{path} expected positive tokens_per_second, got {data['tokens_per_second']}")
        if float(data["samples_per_second"]) <= 0.0:
            raise SystemExit(f"{path} expected positive samples_per_second, got {data['samples_per_second']}")
        if float(data["memory_rss_mb"]) <= 0.0:
            raise SystemExit(f"{path} expected positive memory_rss_mb, got {data['memory_rss_mb']}")
        if float(data["gpu_memory_allocated_mb"]) < 0.0:
            raise SystemExit(f"{path} expected non-negative gpu_memory_allocated_mb, got {data['gpu_memory_allocated_mb']}")

        total_samples = int(data["dataset_total_samples"])
        train_samples = int(data["dataset_train_samples"])
        eval_samples = int(data["dataset_eval_samples"])
        if train_samples + eval_samples != total_samples:
            raise SystemExit(f"{path} train/eval counts do not sum to total: {train_samples}+{eval_samples}!={total_samples}")
        if reference_total is None:
            reference_total = total_samples
            reference_train = train_samples
            reference_eval = eval_samples
            reference_fingerprint = data["dataset_fingerprint"]
        elif (
            total_samples != reference_total
            or train_samples != reference_train
            or eval_samples != reference_eval
            or data["dataset_fingerprint"] != reference_fingerprint
        ):
            raise SystemExit(f"{path} dataset summary drifted across ranks/runs")

        dataset_source_files = data["dataset_source_files"]
        if sorted(dataset_source_files) != sorted(expected_arrows):
            raise SystemExit(f"{path} expected sources {expected_arrows}, got {dataset_source_files}")
        source_counts = data["dataset_source_sample_counts"]
        if sorted(entry.get("path") for entry in source_counts) != sorted(expected_arrows):
            raise SystemExit(f"{path} unexpected source counts: {data['dataset_source_sample_counts']}")
        samples_by_source = {entry["path"]: int(entry["samples"]) for entry in source_counts}
        per_source_samples = [samples_by_source[source] for source in expected_arrows]
        if sum(per_source_samples) != total_samples:
            raise SystemExit(f"{path} source counts do not sum to total: {per_source_samples} vs {total_samples}")
        for source_path, samples, minimum in zip(expected_arrows, per_source_samples, source_min_samples):
            if samples < minimum:
                raise SystemExit(f"{path} expected at least {minimum} samples from {source_path}, got {samples}")
        if data["streaming_train_batches"] is not True:
            raise SystemExit(f"{path} expected streaming_train_batches true")
        if data["streaming_index_cache_hit"] is not expected_hit:
            raise SystemExit(f"{path} expected cache_hit={expected_hit}, got {data['streaming_index_cache_hit']}")
        if data["streaming_index_cache_written"] is not expected_written:
            raise SystemExit(f"{path} expected cache_written={expected_written}, got {data['streaming_index_cache_written']}")
        if pathlib.Path(data["streaming_index_cache_path"]) != expected_cache:
            raise SystemExit(f"{path} expected rank-local cache {expected_cache}, got {data['streaming_index_cache_path']}")

        cache = json.loads(expected_cache.read_text())
        if cache.get("format") != "rustrain.qwen_sft_arrow_row_index.v1":
            raise SystemExit(f"{path} unexpected cache format: {cache.get('format')}")
        if sorted(cache.get("paths", [])) != sorted(expected_arrows):
            raise SystemExit(f"{path} cache paths mismatch: {cache.get('paths')}")
        if cache.get("max_samples") is not None:
            raise SystemExit(f"{path} unbounded Arrow cache should record max_samples null, got {cache.get('max_samples')}")
        if cache.get("field_map", {}).get("input") != "input":
            raise SystemExit(f"{path} cache field_map input should stay default for mixed Arrow: {cache.get('field_map')}")
        if cache.get("field_map", {}).get("source_instruction_fields") != ["instruction", "question"]:
            raise SystemExit(f"{path} cache source_instruction_fields mismatch: {cache.get('field_map')}")
        if cache.get("field_map", {}).get("source_input_fields") != ["input", ""]:
            raise SystemExit(f"{path} cache source_input_fields mismatch: {cache.get('field_map')}")
        if cache.get("field_map", {}).get("source_response_fields") != ["output", "answer"]:
            raise SystemExit(f"{path} cache source_response_fields mismatch: {cache.get('field_map')}")
        if cache.get("summary", {}).get("samples") != total_samples:
            raise SystemExit(f"{path} cache summary samples mismatch: {cache.get('summary')}")
        if cache.get("summary", {}).get("fingerprint") != data["dataset_fingerprint"]:
            raise SystemExit(f"{path} cache fingerprint mismatch: {cache.get('summary')}")
        if len(cache.get("samples", [])) != total_samples:
            raise SystemExit(f"{path} expected {total_samples} cached row indices, got {len(cache.get('samples', []))}")

        expected_cursor_end = expected_steps * int(data["local_batch_size"]) * int(data["world_size"])
        if int(data["data_cursor_start"]) != 0:
            raise SystemExit(f"{path} expected data_cursor_start 0, got {data['data_cursor_start']}")
        if int(data["data_cursor_end"]) != expected_cursor_end:
            raise SystemExit(f"{path} expected data_cursor_end {expected_cursor_end}, got {data['data_cursor_end']}")
        if int(data["data_cursor_next"]) != expected_cursor_end:
            raise SystemExit(f"{path} expected data_cursor_next {expected_cursor_end}, got {data['data_cursor_next']}")
        for cursor_key, epoch_key, offset_key in [
            ("data_cursor_start", "data_epoch_start", "data_sample_offset_start"),
            ("data_cursor_end", "data_epoch_end", "data_sample_offset_end"),
            ("data_cursor_next", "data_epoch_next", "data_sample_offset_next"),
        ]:
            cursor = int(data[cursor_key])
            if int(data[epoch_key]) != cursor // train_samples:
                raise SystemExit(f"{path} {epoch_key} mismatch for {cursor_key}={cursor}")
            if int(data[offset_key]) != cursor % train_samples:
                raise SystemExit(f"{path} {offset_key} mismatch for {cursor_key}={cursor}")
        for key in ["reload_delta", "next_step_delta", "sharded_reload_delta", "sharded_next_step_delta"]:
            if float(data[key]) > 1e-5:
                raise SystemExit(f"{path} {key} too large: {data[key]}")

        sharded_manifest_path = pathlib.Path(data["sharded_global_manifest_output"])
        sharded_manifest = json.loads(sharded_manifest_path.read_text())
        require_complete_qwen_manifest_paths(sharded_manifest, sharded_manifest_path)
        if sorted(sharded_manifest.get("dataset_source_files", [])) != sorted(dataset_source_files):
            raise SystemExit(f"{path} sharded manifest dataset_source_files mismatch")
        if sorted(sharded_manifest.get("dataset_source_sample_counts", []), key=lambda entry: entry["path"]) != sorted(data["dataset_source_sample_counts"], key=lambda entry: entry["path"]):
            raise SystemExit(f"{path} sharded manifest dataset_source_sample_counts mismatch")
        if sharded_manifest.get("dataset_fingerprint") != data["dataset_fingerprint"]:
            raise SystemExit(f"{path} sharded manifest dataset_fingerprint mismatch")
        if sharded_manifest.get("streaming_train_batches") is not True:
            raise SystemExit(f"{path} sharded manifest streaming_train_batches is not true")
        if int(sharded_manifest["data_cursor_next"]) != expected_cursor_end:
            raise SystemExit(f"{path} sharded data_cursor_next mismatch: {sharded_manifest['data_cursor_next']}")

        if data.get("checkpoint_written"):
            rank0_manifest_path = pathlib.Path(data["manifest_output"])
            rank0_manifest = json.loads(rank0_manifest_path.read_text())
            require_complete_qwen_base_model_path(rank0_manifest, rank0_manifest_path)
            if sorted(rank0_manifest.get("dataset_source_files", [])) != sorted(dataset_source_files):
                raise SystemExit(f"{path} rank0 manifest dataset_source_files mismatch")
            if sorted(rank0_manifest.get("dataset_source_sample_counts", []), key=lambda entry: entry["path"]) != sorted(data["dataset_source_sample_counts"], key=lambda entry: entry["path"]):
                raise SystemExit(f"{path} rank0 manifest dataset_source_sample_counts mismatch")
            if rank0_manifest.get("dataset_fingerprint") != data["dataset_fingerprint"]:
                raise SystemExit(f"{path} rank0 manifest dataset_fingerprint mismatch")
            if rank0_manifest.get("streaming_train_batches") is not True:
                raise SystemExit(f"{path} rank0 manifest streaming_train_batches is not true")

    evidence.append(
        {
            "rank": rank,
            "cache": str(expected_cache),
            "first_written": first[rank][1]["streaming_index_cache_written"],
            "second_hit": second[rank][1]["streaming_index_cache_hit"],
            "dataset_total_samples": second[rank][1]["dataset_total_samples"],
            "data_cursor_next": second[rank][1]["data_cursor_next"],
            "tokens_per_second": second[rank][1]["tokens_per_second"],
            "samples_per_second": second[rank][1]["samples_per_second"],
            "memory_rss_mb": second[rank][1]["memory_rss_mb"],
            "gpu_memory_allocated_mb": second[rank][1]["gpu_memory_allocated_mb"],
            "sharded_reload_delta": second[rank][1]["sharded_reload_delta"],
            "sharded_next_step_delta": second[rank][1]["sharded_next_step_delta"],
        }
    )

print(
    json.dumps(
        {
            "qwen_session_dp2_sft_arrow_hf_mixed_schema_cache_verified": evidence,
            "hf_mixed_schema_arrows": expected_arrows,
            "first_elapsed_secs": first_elapsed_secs,
            "second_elapsed_secs": second_elapsed_secs,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
