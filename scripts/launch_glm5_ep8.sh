#!/usr/bin/env bash
set -euo pipefail

# launch_glm5_ep8.sh — Launch GLM-5.2 EP=8 LoRA SFT training on 8 GPUs.
#
# Prerequisites:
#   - GLM-5.2 model downloaded to /vePFS-mindverse/user/nolanho/models/GLM-5.2
#   - SFT data at data/sft/deepseek_test.jsonl (or synthetic fallback)
#   - 8× GPU available (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)
#
# Usage:
#   bash scripts/launch_glm5_ep8.sh
#
# Or via SSH to GPU server:
#   scripts/gpu_run_ssh.sh bash scripts/launch_glm5_ep8.sh

CONFIG="${RUSTRAIN_GLM5_CONFIG:-configs/glm5_lora_sft_ep8.toml}"
OUTPUT_DIR="${RUSTRAIN_GLM5_OUTPUT:-/tmp/rustrain-runs/glm5-lora-sft-ep8}"
NUM_GPUS="${RUSTRAIN_NUM_GPUS:-8}"
LAUNCH_TIMEOUT="${RUSTRAIN_LAUNCH_TIMEOUT_SECS:-1800}"

echo "Launching GLM-5.2 EP=${NUM_GPUS} LoRA SFT training"
echo "  config:     $CONFIG"
echo "  output:     $OUTPUT_DIR"
echo "  timeout:    ${LAUNCH_TIMEOUT}s"
echo "  GPUs:       $NUM_GPUS"

export RUSTRAIN_LAUNCH_TIMEOUT_SECS="$LAUNCH_TIMEOUT"

cargo run --release -- \
  launch --nproc-per-node "$NUM_GPUS" \
  --output-dir "$OUTPUT_DIR" \
  train --config "$CONFIG"
