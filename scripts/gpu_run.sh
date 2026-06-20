#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: scripts/gpu_run.sh <command> [args...]" >&2
  echo "set RUSTRAIN_RAY_NUM_GPUS=N to reserve more than one Ray GPU for a command" >&2
  exit 2
fi

REMOTE_HOST="${RUSTRAIN_REMOTE_HOST:-root@192.168.42.106}"
REMOTE_PORT="${RUSTRAIN_REMOTE_PORT:-2222}"
REMOTE_DIR="${RUSTRAIN_REMOTE_DIR:-/vePFS-Mindverse/user/nolanho/code/rustrain}"
REMOTE_PYTHON="${RUSTRAIN_REMOTE_PYTHON:-/opt/venv/bin/python}"
RAY_NUM_GPUS="${RUSTRAIN_RAY_NUM_GPUS:-1}"

ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "${REMOTE_PYTHON}" - "${REMOTE_DIR}" "${RAY_NUM_GPUS}" "$@" <<'PY'
import ray
import shlex
import subprocess
import sys

remote_dir = sys.argv[1]
ray_num_gpus = float(sys.argv[2])
command = shlex.join(sys.argv[3:])

ray.init(address="auto")

@ray.remote(num_gpus=ray_num_gpus)
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
