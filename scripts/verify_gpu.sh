#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${RUSTRAIN_REMOTE_HOST:-root@192.168.42.106}"
REMOTE_PORT="${RUSTRAIN_REMOTE_PORT:-2222}"
REMOTE_DIR="${RUSTRAIN_REMOTE_DIR:-/vePFS-Mindverse/user/nolanho/code/rustrain}"

remote_run() {
  ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" \
    "cd '${REMOTE_DIR}' && . \"\$HOME/.cargo/env\" 2>/dev/null || true; cd '${REMOTE_DIR}' && source scripts/tch_a800_env.sh && $*"
}

remote_run cargo run -- tch-cuda-probe
remote_run cargo test qwen_delta_manifest_roundtrips
remote_run cargo test qwen_causal_lm_loss_is_finite_for_tiny_weights
remote_run cargo test tch_tiny_lm_trains_all_parameter_groups
remote_run cargo run -- train --config configs/tch_smoke_cuda.toml
