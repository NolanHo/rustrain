#!/usr/bin/env bash
set -euo pipefail

# launch_gpu.sh — Launch GLM-5.2 training on GPU server (no cargo, use pre-built binary)
#
# Usage:
#   bash scripts/launch_gpu.sh configs/glm5_fp8_4layers.toml
#   bash scripts/launch_gpu.sh configs/glm5_fp8_lora_sft_ep8.toml

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-configs/glm5_fp8_4layers.toml}"
OUTPUT_DIR="${RUSTRAIN_OUTPUT:-/tmp/rustrain-runs/glm5-fp8-test}"
NUM_GPUS="${RUSTRAIN_NUM_GPUS:-8}"

# ── Environment ──
VENV="/vePFS-Mindverse/user/nolanho/rustrain-env"
TORCH_LIB="$VENV/lib/python3.13/site-packages/torch/lib"
NVIDIA_LIB="$VENV/lib/python3.13/site-packages/nvidia"
CUDA_LIB="/usr/local/cuda/lib64"

# FP8 kernel .so (find in target/release/build/*/out/)
FP8_DIR=$(find "$PROJECT_DIR/target" -name "libfp8_gemm.so" -path "*/out/*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")

export LD_LIBRARY_PATH="$TORCH_LIB:$NVIDIA_LIB/nccl/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cusparse/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparselt/lib:$CUDA_LIB${FP8_DIR:+:$FP8_DIR}"

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1

# ── Run ──
BINARY="$PROJECT_DIR/target/release/rustrain"
if [ ! -f "$BINARY" ]; then
  echo "ERROR: Binary not found at $BINARY"
  echo "Build on this machine or sync from compile machine."
  exit 1
fi

echo "Launching GLM-5.2 training on ${NUM_GPUS}× GPU"
echo "  binary:  $BINARY"
echo "  config:  $CONFIG"
echo "  output:  $OUTPUT_DIR"
echo "  fp8 lib: ${FP8_DIR:-not found}"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

"$BINARY" launch \
  --nproc-per-node "$NUM_GPUS" \
  --output-dir "$OUTPUT_DIR" \
  train --config "$CONFIG"
