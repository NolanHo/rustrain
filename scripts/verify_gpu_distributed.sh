#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${RUSTRAIN_REMOTE_HOST:-root@192.168.42.106}"
REMOTE_PORT="${RUSTRAIN_REMOTE_PORT:-2222}"
REMOTE_DIR="${RUSTRAIN_REMOTE_DIR:-/vePFS-Mindverse/user/nolanho/code/rustrain}"
REMOTE_PYTHON="${RUSTRAIN_REMOTE_PYTHON:-/opt/venv/bin/python}"
SYNC_TO_WORKER="${RUSTRAIN_SYNC_TO_WORKER:-1}"
OUTPUT_DIR="${RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR:-/tmp/rustrain-runs/qwen-session-trainer-dp2-verify-$(date +%Y%m%d-%H%M%S)-$$}"
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

remote_run_2gpu bash scripts/verify_launch_gpu_assignment_worker.sh
remote_run_2gpu bash scripts/verify_qwen_tp_linear_worker.sh
remote_run_2gpu bash scripts/verify_qwen_tp_attention_worker.sh
remote_run_2gpu bash scripts/verify_qwen_tp_attention_nccl_worker.sh
remote_run_2gpu bash scripts/verify_qwen_tp_mlp_worker.sh
remote_run_2gpu bash scripts/verify_qwen_tp_mlp_nccl_worker.sh
remote_run_2gpu bash scripts/verify_ep_rank_local_worker.sh
remote_run_2gpu bash scripts/verify_ep_nccl_worker.sh
remote_run_2gpu bash scripts/verify_ep_sparse_worker.sh
remote_run_2gpu bash scripts/verify_ep_tch_moe_worker.sh
remote_run_2gpu bash scripts/verify_tch_moe_ep2_trainer_worker.sh
remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  bash scripts/verify_tch_moe_ep2_resume_worker.sh
remote_run_2gpu bash scripts/verify_qwen_session_tp2_worker.sh
remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  bash scripts/verify_qwen_session_tp2_resume_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=600 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}" \
  bash scripts/verify_qwen_session_dp2_worker.sh

remote_run_2gpu bash scripts/verify_qwen_session_dp2_sft_max_samples_worker.sh
remote_run_2gpu bash scripts/verify_qwen_session_dp2_sft_eval_paths_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}-layers01" \
  bash scripts/verify_qwen_session_dp2_layers01_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}-layers01-bf16" \
  RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers01_bf16.toml \
  RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 \
  bash scripts/verify_qwen_session_dp2_layers01_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}-layers03" \
  RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers03.toml \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=49 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.3.self_attn.q_proj.weight,model.layers.3.mlp.down_proj.weight,model.norm.weight \
  bash scripts/verify_qwen_session_dp2_layers01_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}-layers03-bf16" \
  RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers03_bf16.toml \
  RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=49 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.3.self_attn.q_proj.weight,model.layers.3.mlp.down_proj.weight,model.norm.weight \
  bash scripts/verify_qwen_session_dp2_layers01_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}-layers07" \
  RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers07.toml \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=97 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.7.self_attn.q_proj.weight,model.layers.7.mlp.down_proj.weight,model.norm.weight \
  bash scripts/verify_qwen_session_dp2_layers01_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}-layers07-bf16" \
  RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers07_bf16.toml \
  RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=97 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.7.self_attn.q_proj.weight,model.layers.7.mlp.down_proj.weight,model.norm.weight \
  bash scripts/verify_qwen_session_dp2_layers01_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}-layers01-sft" \
  RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers01_sft.toml \
  RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 \
  bash scripts/verify_qwen_session_dp2_layers01_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}-layers01-sft-bf16" \
  RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers01_sft_bf16.toml \
  RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 \
  RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 \
  bash scripts/verify_qwen_session_dp2_layers01_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}-layers03-sft" \
  RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers03_sft.toml \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=49 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.3.self_attn.q_proj.weight,model.layers.3.mlp.down_proj.weight,model.norm.weight \
  RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 \
  bash scripts/verify_qwen_session_dp2_layers01_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}-layers07-sft" \
  RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers07_sft.toml \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=97 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.7.self_attn.q_proj.weight,model.layers.7.mlp.down_proj.weight,model.norm.weight \
  RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 \
  bash scripts/verify_qwen_session_dp2_layers01_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}-layers03-sft-bf16" \
  RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers03_sft_bf16.toml \
  RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=49 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.3.self_attn.q_proj.weight,model.layers.3.mlp.down_proj.weight,model.norm.weight \
  RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 \
  bash scripts/verify_qwen_session_dp2_layers01_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}-layers07-sft-bf16" \
  RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers07_sft_bf16.toml \
  RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=97 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.7.self_attn.q_proj.weight,model.layers.7.mlp.down_proj.weight,model.norm.weight \
  RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 \
  bash scripts/verify_qwen_session_dp2_layers01_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=600 \
  RUSTRAIN_DISTRIBUTED_VERIFY_OUTPUT_DIR="${OUTPUT_DIR}-bf16" \
  RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_bf16.toml \
  RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 \
  bash scripts/verify_qwen_session_dp2_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR="${OUTPUT_DIR}-resume-base" \
  RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR="${OUTPUT_DIR}-resume-continue" \
  RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_sft.toml \
  RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 \
  bash scripts/verify_qwen_session_dp2_resume_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR="${OUTPUT_DIR}-sft-max-samples-resume-base" \
  RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR="${OUTPUT_DIR}-sft-max-samples-resume-continue" \
  bash scripts/verify_qwen_session_dp2_sft_max_samples_resume_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR="${OUTPUT_DIR}-layers01-sft-resume-base" \
  RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR="${OUTPUT_DIR}-layers01-sft-resume-continue" \
  RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_layers01_sft.toml \
  RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=25 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.1.self_attn.q_proj.weight,model.layers.1.mlp.down_proj.weight,model.norm.weight \
  bash scripts/verify_qwen_session_dp2_resume_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR="${OUTPUT_DIR}-layers01-sft-bf16-resume-base" \
  RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR="${OUTPUT_DIR}-layers01-sft-bf16-resume-continue" \
  RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_layers01_sft_bf16.toml \
  RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 \
  RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=25 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.1.self_attn.q_proj.weight,model.layers.1.mlp.down_proj.weight,model.norm.weight \
  bash scripts/verify_qwen_session_dp2_resume_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR="${OUTPUT_DIR}-layers03-sft-resume-base" \
  RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR="${OUTPUT_DIR}-layers03-sft-resume-continue" \
  RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_layers03_sft.toml \
  RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=49 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.3.self_attn.q_proj.weight,model.layers.3.mlp.down_proj.weight,model.norm.weight \
  bash scripts/verify_qwen_session_dp2_resume_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR="${OUTPUT_DIR}-layers07-sft-resume-base" \
  RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR="${OUTPUT_DIR}-layers07-sft-resume-continue" \
  RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_layers07_sft.toml \
  RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=97 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.7.self_attn.q_proj.weight,model.layers.7.mlp.down_proj.weight,model.norm.weight \
  bash scripts/verify_qwen_session_dp2_resume_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR="${OUTPUT_DIR}-layers03-sft-bf16-resume-base" \
  RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR="${OUTPUT_DIR}-layers03-sft-bf16-resume-continue" \
  RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_layers03_sft_bf16.toml \
  RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 \
  RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=49 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.3.self_attn.q_proj.weight,model.layers.3.mlp.down_proj.weight,model.norm.weight \
  bash scripts/verify_qwen_session_dp2_resume_worker.sh

remote_run_2gpu env \
  RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 \
  RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR="${OUTPUT_DIR}-layers07-sft-bf16-resume-base" \
  RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR="${OUTPUT_DIR}-layers07-sft-bf16-resume-continue" \
  RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_layers07_sft_bf16.toml \
  RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 \
  RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=97 \
  RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.7.self_attn.q_proj.weight,model.layers.7.mlp.down_proj.weight,model.norm.weight \
  bash scripts/verify_qwen_session_dp2_resume_worker.sh
