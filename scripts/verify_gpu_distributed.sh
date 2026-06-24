#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${RUSTRAIN_REMOTE_HOST:-${RUSTRAIN_REMOTE_USER:-root}@${RUSTRAIN_REMOTE_HOST:-127.0.0.1}}"
REMOTE_PORT="${RUSTRAIN_REMOTE_PORT:-2222}"
REMOTE_DIR="${RUSTRAIN_REMOTE_DIR:-${RUSTRAIN_REMOTE_DIR:-/root/rustrain}}"
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

# DP=2 config-driven training
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=600 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-trainer-dp2 train --config configs/qwen_session_dp2.toml
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=600 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-trainer-dp2-bf16 train --config configs/qwen_session_dp2_bf16.toml

# DP=2 layer variants
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-layers01 train --config configs/qwen_session_dp2_layers01.toml
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-layers01-bf16 train --config configs/qwen_session_dp2_layers01_bf16.toml
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-layers03 train --config configs/qwen_session_dp2_layers03.toml
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-layers03-bf16 train --config configs/qwen_session_dp2_layers03_bf16.toml
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-layers07 train --config configs/qwen_session_dp2_layers07.toml
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-layers07-bf16 train --config configs/qwen_session_dp2_layers07_bf16.toml

# DP=2 SFT variants
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-sft train --config configs/qwen_session_dp2_sft.toml
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-layers01-sft train --config configs/qwen_session_dp2_layers01_sft.toml
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-layers03-sft train --config configs/qwen_session_dp2_layers03_sft.toml
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-layers07-sft train --config configs/qwen_session_dp2_layers07_sft.toml

# DP=2 SFT bf16 variants
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-layers01-sft-bf16 train --config configs/qwen_session_dp2_layers01_sft_bf16.toml
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-layers03-sft-bf16 train --config configs/qwen_session_dp2_layers03_sft_bf16.toml
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-layers07-sft-bf16 train --config configs/qwen_session_dp2_layers07_sf_bf16.toml

# TP=2
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-trainer-tp2 train --config configs/qwen_session_tp2.toml

# MoE EP=2
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/tch-moe-ep2 train --config configs/tch_moe_ep2.toml

# DP=2 max samples
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-sft-max-samples train --config configs/qwen_session_dp2_sft_max_samples.toml

# DP=2 SFT eval paths
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-sft-eval-paths train --config configs/qwen_session_dp2_sft_eval_paths.toml

# DP=2 SFT Arrow variants
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-sft-arrow train --config configs/qwen_session_dp2_sft_arrow.toml
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-sft-arrow-eval-paths train --config configs/qwen_session_dp2_sft_arrow_eval_paths.toml
remote_run_2gpu env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp2-sft-arrow-index-cache train --config configs/qwen_session_dp2_sft_arrow_index_cache.toml
