#!/usr/bin/env bash
set -euo pipefail

# gpu_run_ssh.sh — Run commands on a remote GPU server via direct SSH (no Ray).
#
# This is the non-Ray counterpart of scripts/ray/gpu_run.sh. It syncs code to
# the remote, sets up the GPU/CUDA/tch-rs environment, and runs the command
# directly — no Ray task submission, no C wrapper for env injection.
#
# Usage: scripts/gpu_run_ssh.sh <command> [args...]
#
# Environment variables (all optional, with defaults):
#   RUSTRAIN_REMOTE_HOST              SSH target (default: root@H20e3)
#   RUSTRAIN_REMOTE_PORT              SSH port (default: 22)
#   RUSTRAIN_REMOTE_DIR               Remote project directory (default: /root/rustrain)
#   RUSTRAIN_SYNC_TO_REMOTE           1=sync code, 0=use existing (default: 1)
#   RUSTRAIN_NUM_GPUS                 GPUs to expose via CUDA_VISIBLE_DEVICES (default: 1)
#   RUSTRAIN_REMOTE_ENV_FILE          Source this file on remote before running (optional)
#   RUSTRAIN_REMOTE_CARGO_HOME        Remote cargo home (default: $HOME/.cargo)
#   RUSTRAIN_REMOTE_RUSTUP_HOME       Remote rustup home (default: $HOME/.rustup)
#   RUSTRAIN_REMOTE_PYTHON            Remote python binary (default: python3)
#   RUSTRAIN_REMOTE_CUDA_PREFIX       CUDA install prefix (default: /usr/local/cuda)
#   RUSTRAIN_REMOTE_CARGO_TARGET_DIR  Remote cargo target dir (default: $REMOTE_DIR/target)
#   RUSTRAIN_SSH_OPTS                Extra SSH options

if [ "$#" -eq 0 ]; then
  echo "usage: scripts/gpu_run_ssh.sh <command> [args...]" >&2
  echo "set RUSTRAIN_NUM_GPUS=N to expose N GPUs" >&2
  exit 2
fi

# ─── Configuration ───────────────────────────────────────────────────────────

REMOTE_HOST="${RUSTRAIN_REMOTE_HOST:-root@H20e3}"
REMOTE_PORT="${RUSTRAIN_REMOTE_PORT:-22}"
REMOTE_DIR="${RUSTRAIN_REMOTE_DIR:-/root/rustrain}"
SYNC_TO_REMOTE="${RUSTRAIN_SYNC_TO_REMOTE:-1}"
NUM_GPUS="${RUSTRAIN_NUM_GPUS:-1}"
REMOTE_ENV_FILE="${RUSTRAIN_REMOTE_ENV_FILE:-}"
REMOTE_CARGO_HOME="${RUSTRAIN_REMOTE_CARGO_HOME:-}"
REMOTE_RUSTUP_HOME="${RUSTRAIN_REMOTE_RUSTUP_HOME:-}"
REMOTE_PYTHON="${RUSTRAIN_REMOTE_PYTHON:-python3}"
REMOTE_CUDA_PREFIX="${RUSTRAIN_REMOTE_CUDA_PREFIX:-/usr/local/cuda}"
REMOTE_CARGO_TARGET_DIR="${RUSTRAIN_REMOTE_CARGO_TARGET_DIR:-}"
SSH_OPTS="${RUSTRAIN_SSH_OPTS:--o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/rustrain_ssh_known_hosts}"

# Build CUDA_VISIBLE_DEVICES for the remote
if [ "$NUM_GPUS" -gt 0 ]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)) | sed 's/,$//')"
else
  CUDA_VISIBLE_DEVICES=""
fi

# ─── Sync code to remote ────────────────────────────────────────────────────

if [ "$SYNC_TO_REMOTE" = "1" ]; then
  echo "[gpu_run_ssh] Syncing code to ${REMOTE_HOST}:${REMOTE_DIR} ..."
  if command -v rsync &>/dev/null; then
    rsync -az --delete \
      --exclude '.git' --exclude 'target' --exclude 'runs' \
      --exclude '.xbot' --exclude '_internal_docs' \
      -e "ssh ${SSH_OPTS} -p ${REMOTE_PORT}" \
      ./ "${REMOTE_HOST}:${REMOTE_DIR}/"
  else
    # Fallback: tar + scp + remote extract
    LOCAL_ARCHIVE="$(mktemp)"
    tar --exclude .git --exclude target --exclude runs \
      --exclude .xbot --exclude _internal_docs \
      -cf "$LOCAL_ARCHIVE" .
    REMOTE_ARCHIVE="/tmp/rustrain-ssh-sync-$$.tar"
    scp ${SSH_OPTS} -P "$REMOTE_PORT" "$LOCAL_ARCHIVE" "${REMOTE_HOST}:${REMOTE_ARCHIVE}"
    rm -f "$LOCAL_ARCHIVE"
    ssh ${SSH_OPTS} -p "$REMOTE_PORT" "$REMOTE_HOST" \
      "mkdir -p '$REMOTE_DIR' && tar -xf '$REMOTE_ARCHIVE' -C '$REMOTE_DIR' && rm -f '$REMOTE_ARCHIVE'"
  fi
  echo "[gpu_run_ssh] Sync complete."
fi

# ─── Build remote env-setup script ──────────────────────────────────────────
#
# The env setup is written as a bash snippet that runs on the remote. Variable
# expansion with \$ is deferred to the remote shell; local values are injected
# directly (no \$).

REMOTE_CARGO_HOME_ARG="${REMOTE_CARGO_HOME}"
REMOTE_RUSTUP_HOME_ARG="${REMOTE_RUSTUP_HOME}"
REMOTE_CARGO_TARGET_DIR_ARG="${REMOTE_CARGO_TARGET_DIR}"

REMOTE_SETUP=$(cat <<REMOTE_SETUP
set -euo pipefail
cd "${REMOTE_DIR}"

# ── Source optional remote env file ──
if [ -n "${REMOTE_ENV_FILE}" ] && [ -f "${REMOTE_ENV_FILE}" ]; then
  source "${REMOTE_ENV_FILE}"
fi

# ── GPU visibility ──
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

# ── tch-rs / PyTorch integration ──
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1

# ── Rust toolchain ──
REMOTE_CARGO_HOME='${REMOTE_CARGO_HOME_ARG}'
REMOTE_RUSTUP_HOME='${REMOTE_RUSTUP_HOME_ARG}'
REMOTE_CARGO_TARGET_DIR='${REMOTE_CARGO_TARGET_DIR_ARG}'
if [ -z "\$REMOTE_CARGO_HOME" ]; then
  REMOTE_CARGO_HOME="\$HOME/.cargo"
fi
if [ -z "\$REMOTE_RUSTUP_HOME" ]; then
  REMOTE_RUSTUP_HOME="\$HOME/.rustup"
fi
if [ -z "\$REMOTE_CARGO_TARGET_DIR" ]; then
  REMOTE_CARGO_TARGET_DIR="${REMOTE_DIR}/target"
fi
export CARGO_HOME="\$REMOTE_CARGO_HOME"
export RUSTUP_HOME="\$REMOTE_RUSTUP_HOME"
export CARGO_TARGET_DIR="\$REMOTE_CARGO_TARGET_DIR"
export PATH="\$REMOTE_CARGO_HOME/bin:\$PATH"

# ── Auto-detect Python site-packages and torch lib path ──
TORCH_SITE="\$(${REMOTE_PYTHON} -c 'import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))' 2>/dev/null || echo '')"
if [ -z "\$TORCH_SITE" ]; then
  echo "[gpu_run_ssh] WARNING: torch not found via ${REMOTE_PYTHON}; tch-rs build may fail." >&2
  TORCH_SITE="/usr/local/lib/python3.13/dist-packages"
fi
TORCH_LIB="\$TORCH_SITE/torch/lib"
NVIDIA_LIB="\$TORCH_SITE/nvidia"
echo "[gpu_run_ssh] torch_site=\$TORCH_SITE"

# ── LD_LIBRARY_PATH: CUDA + torch + nvidia libs ──
LD_PATHS=(
  "\$TORCH_LIB"
  "\$NVIDIA_LIB/cuda_runtime/lib"
  "\$NVIDIA_LIB/cuda_cupti/lib"
  "\$NVIDIA_LIB/cuda_nvrtc/lib"
  "\$NVIDIA_LIB/cublas/lib"
  "\$NVIDIA_LIB/cudnn/lib"
  "\$NVIDIA_LIB/cufft/lib"
  "\$NVIDIA_LIB/curand/lib"
  "\$NVIDIA_LIB/cusolver/lib"
  "\$NVIDIA_LIB/cusparse/lib"
  "\$NVIDIA_LIB/cusparselt/lib"
  "\$NVIDIA_LIB/nccl/lib"
  "\$NVIDIA_LIB/nvjitlink/lib"
  "\$NVIDIA_LIB/nvtx/lib"
  "${REMOTE_CUDA_PREFIX}/lib64"
  "${REMOTE_CUDA_PREFIX}/compat"
  "/usr/local/nvidia/lib"
  "/usr/local/nvidia/lib64"
)
# Only include paths that exist on the remote
LD_LIBRARY_PATH=""
for p in "\${LD_PATHS[@]}"; do
  [ -d "\$p" ] && LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:+\$LD_LIBRARY_PATH:}\$p"
done
export LD_LIBRARY_PATH

# ── LD_PRELOAD: libtorch_cuda.so (required by tch-rs for CUDA ops) ──
if [ -f "\$TORCH_LIB/libtorch_cuda.so" ]; then
  export LD_PRELOAD="\$TORCH_LIB/libtorch_cuda.so"
fi

# ── PYTHONPATH (for tch-rs build script) ──
export PYTHONPATH="\$TORCH_SITE"

# ── Ready ──
echo "[gpu_run_ssh] host=\$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[gpu_run_ssh] LD_LIBRARY_PATH=\$LD_LIBRARY_PATH"
echo "[gpu_run_ssh] CARGO_HOME=\$CARGO_HOME PATH=\$PATH"
echo "[gpu_run_ssh] Running: $*"
echo "---"

# ── Execute the actual command ──
exec $*
REMOTE_SETUP
)

# ─── Execute on remote ──────────────────────────────────────────────────────

echo "$REMOTE_SETUP" | ssh ${SSH_OPTS} -p "$REMOTE_PORT" "$REMOTE_HOST" bash
