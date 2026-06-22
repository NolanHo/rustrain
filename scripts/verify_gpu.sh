#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${RUSTRAIN_REMOTE_HOST:-root@192.168.42.106}"
REMOTE_PORT="${RUSTRAIN_REMOTE_PORT:-2222}"
REMOTE_DIR="${RUSTRAIN_REMOTE_DIR:-/vePFS-Mindverse/user/nolanho/code/rustrain}"
REMOTE_PYTHON="${RUSTRAIN_REMOTE_PYTHON:-/opt/venv/bin/python}"
SYNC_TO_WORKER="${RUSTRAIN_SYNC_TO_WORKER:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

remote_run() {
  RUSTRAIN_REMOTE_HOST="${REMOTE_HOST}" \
    RUSTRAIN_REMOTE_PORT="${REMOTE_PORT}" \
    RUSTRAIN_REMOTE_DIR="${REMOTE_DIR}" \
    RUSTRAIN_REMOTE_PYTHON="${REMOTE_PYTHON}" \
    RUSTRAIN_SYNC_TO_WORKER="${SYNC_TO_WORKER}" \
    "${SCRIPT_DIR}/gpu_run.sh" "$@"
}

remote_run cargo run -- tch-cuda-probe
remote_run cargo fmt --check
remote_run cargo check
remote_run bash scripts/verify_moe_smoke_worker.sh
remote_run bash scripts/verify_tch_moe_smoke_worker.sh
remote_run cargo test qwen_delta_manifest_roundtrips
remote_run cargo test qwen_causal_lm_loss_is_finite_for_tiny_weights
remote_run cargo test qwen_lora
remote_run cargo test tch_tiny_lm_trains_all_parameter_groups
remote_run cargo test tch_dtype_policy_maps_runtime_dtype_to_compute_kind
remote_run cargo run -- train --config configs/tch_smoke_cuda.toml
remote_run cargo run -- train --config configs/tch_smoke_cuda_bf16.toml
remote_run cargo run -- qwen-sampling-smoke
remote_run bash scripts/verify_qwen_kv_cache_worker.sh
remote_run cargo run -- qwen-lora-sft-smoke
remote_run cargo run -- train --config configs/qwen_lora_sft.toml
remote_run bash scripts/verify_qwen_lora_sft_resume.sh
remote_run bash scripts/verify_qwen_lora_sft_max_samples_worker.sh
remote_run bash scripts/verify_qwen_lora_sft_no_shuffle_worker.sh
remote_run bash scripts/verify_qwen_lora_sft_eval_paths_worker.sh
remote_run bash scripts/verify_qwen_sft_streaming_data_plan_worker.sh
remote_run bash scripts/verify_qwen_sft_streaming_batch_plan_worker.sh
remote_run bash scripts/verify_qwen_sft_streaming_scale_worker.sh
remote_run bash scripts/verify_qwen_sft_streaming_hf_cache_worker.sh
remote_run bash scripts/verify_qwen_lora_sft_trainer_index_cache_worker.sh
remote_run bash scripts/verify_qwen_sft_trainer_index_cache_worker.sh
remote_run bash scripts/verify_qwen_sft_trainer_default_index_cache_worker.sh
remote_run bash scripts/verify_qwen_sft_streaming_eval_paths_worker.sh
remote_run bash scripts/verify_qwen_sft_streaming_batch_eval_paths_worker.sh
remote_run bash scripts/verify_qwen_session_sft_max_samples_worker.sh
remote_run bash scripts/verify_qwen_session_sft_eval_paths_worker.sh
remote_run cargo run -- train --config configs/qwen_lora_sft_bf16.toml
remote_run env RUSTRAIN_QWEN_LORA_SFT_CONFIG=configs/qwen_lora_sft_bf16.toml RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 bash scripts/verify_qwen_lora_sft_resume.sh
remote_run bash scripts/verify_qwen_session_single_bf16_worker.sh
remote_run bash scripts/verify_qwen_session_single_resume.sh
remote_run bash scripts/verify_qwen_session_single_sft_resume.sh
remote_run bash scripts/verify_qwen_session_single_sft_arrow_worker.sh
remote_run bash scripts/verify_qwen_session_single_sft_arrow_eval_paths_worker.sh
remote_run env RUSTRAIN_QWEN_SESSION_SINGLE_SFT_CONFIG=configs/qwen_session_single_layers07_sft.toml RUSTRAIN_EXPECTED_QWEN_TRAINABLE_TENSORS=98 bash scripts/verify_qwen_session_single_sft_resume.sh
remote_run env RUSTRAIN_QWEN_SESSION_SINGLE_SFT_CONFIG=configs/qwen_session_single_layers07_sft_bf16.toml RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 RUSTRAIN_EXPECTED_QWEN_TRAINABLE_TENSORS=98 bash scripts/verify_qwen_session_single_sft_resume.sh
remote_run bash scripts/verify_qwen_session_layers01_worker.sh
remote_run bash scripts/verify_qwen_session_layers03_worker.sh
remote_run env RUSTRAIN_QWEN_SESSION_LAYERS_CONFIG=configs/qwen_session_single_layers03_bf16.toml RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 bash scripts/verify_qwen_session_layers03_worker.sh
remote_run env RUSTRAIN_QWEN_SESSION_LAYERS_CONFIG=configs/qwen_session_single_layers07.toml RUSTRAIN_EXPECTED_QWEN_TRAINABLE_TENSORS=98 bash scripts/verify_qwen_session_layers03_worker.sh
remote_run env RUSTRAIN_QWEN_SESSION_LAYERS_CONFIG=configs/qwen_session_single_layers07_bf16.toml RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 RUSTRAIN_EXPECTED_QWEN_TRAINABLE_TENSORS=98 bash scripts/verify_qwen_session_layers03_worker.sh
remote_run bash scripts/verify_qwen_full_train_smoke_worker.sh
