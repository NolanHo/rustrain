#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${RUSTRAIN_REMOTE_HOST:-root@192.168.42.106}"
REMOTE_PORT="${RUSTRAIN_REMOTE_PORT:-2222}"
REMOTE_DIR="${RUSTRAIN_REMOTE_DIR:-/vePFS-Mindverse/user/nolanho/code/rustrain}"
REMOTE_PYTHON="${RUSTRAIN_REMOTE_PYTHON:-/opt/venv/bin/python}"
SYNC_TO_WORKER="${RUSTRAIN_SYNC_TO_WORKER:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

remote_run_2gpu() {
  RUSTRAIN_REMOTE_HOST="${REMOTE_HOST}" \
    RUSTRAIN_REMOTE_PORT="${REMOTE_PORT}" \
    RUSTRAIN_REMOTE_DIR="${REMOTE_DIR}" \
    RUSTRAIN_REMOTE_PYTHON="${REMOTE_PYTHON}" \
    RUSTRAIN_SYNC_TO_WORKER="${SYNC_TO_WORKER}" \
    RUSTRAIN_RAY_NUM_GPUS=2 \
    "${SCRIPT_DIR}/gpu_run.sh" "$@"
}

remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=600 cargo run -- launch \
  --nproc-per-node 2 \
  --output-dir /tmp/rustrain-runs/qwen-session-trainer-dp2-verify \
  train --config configs/qwen_session_dp2.toml
