#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/require_gpu_worker.sh"

BASE_OUTPUT_DIR="${RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR:?RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR is required}"
RESUME_OUTPUT_DIR="${RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR:?RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR is required}"
CONFIG="${RUSTRAIN_QWEN_SESSION_DP_CONFIG:-configs/qwen_session_dp2_sft.toml}"
EXPECTED_DATASET_SEED="${RUSTRAIN_EXPECTED_DATASET_ORDER_SEED:-}"
export BASE_OUTPUT_DIR RESUME_OUTPUT_DIR EXPECTED_DATASET_SEED

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

base_rank0_paths = sorted(base_output_dir.rglob("qwen-session-dp-rank-0.json"))
if len(base_rank0_paths) != 1:
    raise SystemExit(f"expected one base rank0 summary, found {len(base_rank0_paths)}")
base_rank0 = json.loads(base_rank0_paths[0].read_text())
base_cursor_next = int(base_rank0["data_cursor_next"])

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
    for key in ["reload_delta", "next_step_delta", "sharded_reload_delta", "sharded_next_step_delta"]:
        if float(data[key]) > 1e-5:
            raise SystemExit(f"{path} {key} too large: {data[key]}")
    manifest = json.loads(pathlib.Path(data["manifest_output"]).read_text())
    if int(manifest["data_cursor_start"]) != base_cursor_next:
        raise SystemExit(
            f"{path} manifest data_cursor_start {manifest['data_cursor_start']} did not continue from {base_cursor_next}"
        )
    if int(manifest["data_cursor_next"]) != expected_cursor_next:
        raise SystemExit(
            f"{path} manifest data_cursor_next {manifest['data_cursor_next']} != expected {expected_cursor_next}"
        )
    sharded_manifest = json.loads(pathlib.Path(data["sharded_global_manifest_output"]).read_text())
    if int(sharded_manifest["data_cursor_next"]) != expected_cursor_next:
        raise SystemExit(
            f"{path} sharded data_cursor_next {sharded_manifest['data_cursor_next']} != expected {expected_cursor_next}"
        )
    evidence.append(
        {
            "rank": data["rank"],
            "resume_from": data["resume_from"],
            "data_cursor_start": data["data_cursor_start"],
            "data_cursor_next": data["data_cursor_next"],
            "dataset_fingerprint": data["dataset_fingerprint"],
            "reload_delta": data["reload_delta"],
            "sharded_reload_delta": data["sharded_reload_delta"],
            "sharded_next_step_delta": data["sharded_next_step_delta"],
        }
    )

print(json.dumps({"qwen_session_dp2_resume": evidence}, indent=2, sort_keys=True))
PY
