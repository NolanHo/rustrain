#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${RUSTRAIN_REMOTE_HOST:-root@192.168.42.106}"
REMOTE_PORT="${RUSTRAIN_REMOTE_PORT:-2222}"
REMOTE_DIR="${RUSTRAIN_REMOTE_DIR:-/vePFS-Mindverse/user/nolanho/code/rustrain}"
REMOTE_PYTHON="${RUSTRAIN_REMOTE_PYTHON:-/opt/venv/bin/python}"

remote_run() {
  local command="$*"
  ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "${REMOTE_PYTHON} - '${REMOTE_DIR}' '${command}'" <<'PY'
import ray
import subprocess
import sys

remote_dir = sys.argv[1]
command = sys.argv[2]

ray.init(address="auto")

@ray.remote(num_gpus=1)
def run_on_gpu_worker(remote_dir: str, command: str) -> str:
    script = f"""
set -euo pipefail
if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable
fi
. "$HOME/.cargo/env"
cd "{remote_dir}"
source scripts/tch_a800_env.sh
{command}
"""
    result = subprocess.run(
        ["bash", "-lc", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    return result.stdout

print(ray.get(run_on_gpu_worker.remote(remote_dir, command)), end="")
PY
}

remote_run cargo run -- tch-cuda-probe
remote_run cargo test qwen_delta_manifest_roundtrips
remote_run cargo test qwen_causal_lm_loss_is_finite_for_tiny_weights
remote_run cargo test tch_tiny_lm_trains_all_parameter_groups
remote_run cargo run -- train --config configs/tch_smoke_cuda.toml
remote_run cargo run -- qwen-lora-sft-smoke
