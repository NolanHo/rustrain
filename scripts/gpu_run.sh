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
REMOTE_FALLBACK_DIR="${RUSTRAIN_REMOTE_FALLBACK_DIR:-/root/rustrain}"
REMOTE_PYTHON="${RUSTRAIN_REMOTE_PYTHON:-/opt/venv/bin/python}"
RAY_NUM_GPUS="${RUSTRAIN_RAY_NUM_GPUS:-1}"
SYNC_TO_WORKER="${RUSTRAIN_SYNC_TO_WORKER:-1}"
SSH_OPTS="${RUSTRAIN_SSH_OPTS:--o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/rustrain_gpu_known_hosts -o GlobalKnownHostsFile=/dev/null}"
REMOTE_ARCHIVE=""
NO_REMOTE_ARCHIVE="__RUSTRAIN_NO_ARCHIVE__"

read -r -a SSH_OPT_ARGS <<<"${SSH_OPTS}"

cleanup_remote_archive() {
  if [ -n "${REMOTE_ARCHIVE}" ]; then
    ssh "${SSH_OPT_ARGS[@]}" -p "${REMOTE_PORT}" "${REMOTE_HOST}" "rm -f '${REMOTE_ARCHIVE}'" >/dev/null 2>&1 || true
  fi
}
trap cleanup_remote_archive EXIT

if [ "${SYNC_TO_WORKER}" = "1" ]; then
  LOCAL_ARCHIVE="$(mktemp)"
  tar --exclude .git --exclude target --exclude runs -cf "${LOCAL_ARCHIVE}" .
  if [ -d runs/parity ]; then
    find runs/parity -maxdepth 1 -type f -name '*.safetensors' \
      ! -name 'qwen2_5_0_5b_tied_head_delta.safetensors' \
      -print0 | xargs -0 -r tar -rf "${LOCAL_ARCHIVE}"
  fi
  REMOTE_ARCHIVE="/tmp/rustrain-gpu-run-${USER:-user}-$$.tar"
  scp "${SSH_OPT_ARGS[@]}" -P "${REMOTE_PORT}" "${LOCAL_ARCHIVE}" "${REMOTE_HOST}:${REMOTE_ARCHIVE}" >/dev/null
  rm -f "${LOCAL_ARCHIVE}"
fi

REMOTE_ARCHIVE_ARG="${REMOTE_ARCHIVE:-${NO_REMOTE_ARCHIVE}}"

ssh "${SSH_OPT_ARGS[@]}" -p "${REMOTE_PORT}" "${REMOTE_HOST}" "${REMOTE_PYTHON}" - "${REMOTE_DIR}" "${REMOTE_FALLBACK_DIR}" "${RAY_NUM_GPUS}" "${REMOTE_ARCHIVE_ARG}" "$@" <<'PY'
import ray
import hashlib
import io
import os
import shlex
import shutil
import subprocess
import sys
import tarfile

remote_dir = sys.argv[1]
remote_fallback_dir = sys.argv[2]
ray_num_gpus = float(sys.argv[3])
remote_archive = sys.argv[4]
if remote_archive == "__RUSTRAIN_NO_ARCHIVE__":
    remote_archive = ""
command = shlex.join(sys.argv[5:])
archive_bytes = None
if remote_archive:
    with open(remote_archive, "rb") as archive_file:
        archive_bytes = archive_file.read()

ray.init(address="auto")

@ray.remote(num_gpus=ray_num_gpus)
def run_on_gpu_worker(
    remote_dir: str,
    remote_fallback_dir: str,
    command: str,
    archive_bytes: bytes | None,
) -> str:
    accelerator_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
    subprocess_env = os.environ.copy()
    if accelerator_ids and not subprocess_env.get("CUDA_VISIBLE_DEVICES"):
        subprocess_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in accelerator_ids)
    work_dir = remote_dir
    staged = archive_bytes is not None
    if staged:
        digest = hashlib.sha256(archive_bytes).hexdigest()[:16]
        work_dir = f"/tmp/rustrain-gpu-run-{digest}-{os.getpid()}"
        shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
        work_dir_abs = os.path.abspath(work_dir)
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            for member in archive.getmembers():
                target = os.path.abspath(os.path.join(work_dir_abs, member.name))
                if target != work_dir_abs and not target.startswith(work_dir_abs + os.sep):
                    raise RuntimeError(f"refusing unsafe archive path: {member.name}")
            archive.extractall(work_dir_abs)
    elif not os.path.isdir(work_dir):
        if remote_fallback_dir and os.path.isdir(remote_fallback_dir):
            work_dir = remote_fallback_dir
        else:
            raise RuntimeError(
                "remote checkout does not exist and no fallback is available: "
                f"remote_dir={remote_dir}, fallback={remote_fallback_dir}; "
                "set RUSTRAIN_SYNC_TO_WORKER=1 to stage the local worktree"
            )
    worker_header = (
        f"Ray worker host={os.uname().nodename} "
        f"accelerator_ids={accelerator_ids} "
        f"CUDA_VISIBLE_DEVICES={subprocess_env.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"work_dir={work_dir} staged={staged} "
        f"command={command}\n"
    )
    script = f"""
set -euo pipefail
if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable
fi
. "$HOME/.cargo/env"
cd {shlex.quote(work_dir)}
source scripts/tch_a800_env.sh
/opt/venv/bin/python - <<'RUSTRAIN_GPU_PROBE'
import os
import sys
import torch

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f"Ray CUDA worker ready: device_count={{torch.cuda.device_count()}}")
    sys.exit(0)
print("Ray worker has no usable CUDA GPU; refusing CPU execution.")
print(f"CUDA_VISIBLE_DEVICES={{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}}")
print(f"torch.cuda.is_available={{torch.cuda.is_available()}}")
print(f"torch.cuda.device_count={{torch.cuda.device_count()}}")
sys.exit(1)
RUSTRAIN_GPU_PROBE
{command}
"""
    result = subprocess.run(
        ["bash", "-lc", script],
        env=subprocess_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(worker_header + result.stdout)
    return worker_header + result.stdout

print(
    ray.get(
        run_on_gpu_worker.remote(
            remote_dir,
            remote_fallback_dir,
            command,
            archive_bytes,
        )
    ),
    end="",
)
PY
