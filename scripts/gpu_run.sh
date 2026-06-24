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
REMOTE_PYTHON="${RUSTRAIN_REMOTE_PYTHON:-/usr/local/bin/python3}"
RAY_HEAD_FILE="${RUSTRAIN_RAY_HEAD_FILE:-/vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt}"
RAY_ADDRESS="${RUSTRAIN_RAY_ADDRESS:-}"
RAY_NUM_GPUS="${RUSTRAIN_RAY_NUM_GPUS:-1}"
SYNC_TO_WORKER="${RUSTRAIN_SYNC_TO_WORKER:-1}"
SSH_OPTS="${RUSTRAIN_SSH_OPTS:--o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/rustrain_gpu_known_hosts -o GlobalKnownHostsFile=/dev/null}"
REMOTE_ARCHIVE=""

LOCAL_ARCHIVE=""
NO_REMOTE_ARCHIVE="__RUSTRAIN_NO_REMOTE_ARCHIVE__"
AUTO_RAY_ADDRESS="__RUSTRAIN_AUTO_RAY_ADDRESS__"

cleanup_remote_archive() {
  if [ -n "${REMOTE_ARCHIVE}" ] && [ "${REMOTE_ARCHIVE}" != "${NO_REMOTE_ARCHIVE}" ]; then
    ssh ${SSH_OPTS} -p "${REMOTE_PORT}" "${REMOTE_HOST}" "rm -f '${REMOTE_ARCHIVE}'" >/dev/null 2>&1 || true
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
  scp ${SSH_OPTS} -P "${REMOTE_PORT}" "${LOCAL_ARCHIVE}" "${REMOTE_HOST}:${REMOTE_ARCHIVE}" >/dev/null
  rm -f "${LOCAL_ARCHIVE}"
fi

REMOTE_ARCHIVE_ARG="${REMOTE_ARCHIVE:-${NO_REMOTE_ARCHIVE}}"
RAY_ADDRESS_ARG="${RAY_ADDRESS:-${AUTO_RAY_ADDRESS}}"

ssh ${SSH_OPTS} -p "${REMOTE_PORT}" "${REMOTE_HOST}" "${REMOTE_PYTHON}" - "${REMOTE_DIR}" "${REMOTE_FALLBACK_DIR}" "${RAY_ADDRESS_ARG}" "${RAY_HEAD_FILE}" "${RAY_NUM_GPUS}" "${REMOTE_ARCHIVE_ARG}" "$@" <<'PY'
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile

import ray

remote_dir = sys.argv[1]
remote_fallback_dir = sys.argv[2]
ray_address = sys.argv[3]
ray_head_file = sys.argv[4]
ray_num_gpus = int(sys.argv[5])
remote_archive = sys.argv[6]
command = sys.argv[7:]

if ray_address == "__RUSTRAIN_AUTO_RAY_ADDRESS__":
    try:
        with open(ray_head_file, "r", encoding="utf-8") as address_file:
            ray_head = address_file.read().strip()
    except Exception:
        ray_head = "192.168.42.141"
    ray_address = f"{ray_head}:6379"

if ":" not in ray_address:
    ray_address = f"{ray_address}:6379"

try:
    ray.init(address=ray_address, log_to_driver=False)
except Exception as error:
    raise RuntimeError(
        f"Ray head is not reachable from the SSH submission host; refusing to fall back to local/plain-SSH execution. "
        f"ray_address={ray_address}, ray_head_file={ray_head_file}, error={error}"
    )

archive_bytes = None
if remote_archive and remote_archive != "__RUSTRAIN_NO_REMOTE_ARCHIVE__":
    with open(remote_archive, "rb") as archive_file:
        archive_bytes = archive_file.read()


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

    # Install Rust if not present
    cargo_bin = "/root/.cargo/bin/cargo"
    if not os.path.exists(cargo_bin):
        subprocess.run(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path",
            shell=True, capture_output=True, text=True, timeout=120,
        )
    subprocess_env["PATH"] = f"/root/.cargo/bin:{subprocess_env.get('PATH', '')}"
    subprocess_env["LIBTORCH_USE_PYTORCH"] = "1"
    subprocess_env["LIBTORCH_BYPASS_VERSION_CHECK"] = "1"
    subprocess_env["CARGO_TARGET_DIR"] = "/tmp/rustrain-target-a800"

    torch_site = os.environ.get(
        "RUSTRAIN_TORCH_SITE",
        "/usr/local/lib/python3.12/dist-packages",
    )
    torch_lib = f"{torch_site}/torch/lib"
    nvidia = f"{torch_site}/nvidia"
    ld_library = ":".join([
        torch_lib,
        f"{nvidia}/cuda_runtime/lib",
        f"{nvidia}/cuda_cupti/lib",
        f"{nvidia}/cuda_nvrtc/lib",
        f"{nvidia}/cublas/lib",
        f"{nvidia}/cudnn/lib",
        f"{nvidia}/cufft/lib",
        f"{nvidia}/curand/lib",
        f"{nvidia}/cusolver/lib",
        f"{nvidia}/cusparse/lib",
        f"{nvidia}/cusparselt/lib",
        f"{nvidia}/nccl/lib",
        f"{nvidia}/nvjitlink/lib",
        f"{nvidia}/nvshmem/lib",
        f"{nvidia}/nvtx/lib",
        "/usr/local/cuda/lib64",
    ])
    if "LD_LIBRARY_PATH" in subprocess_env:
        ld_library = f"{ld_library}:{subprocess_env['LD_LIBRARY_PATH']}"
    subprocess_env["LD_LIBRARY_PATH"] = ld_library

    ld_preload = f"{torch_lib}/libtorch_cuda.so"
    if "LD_PRELOAD" in subprocess_env:
        ld_preload = f"{ld_preload}:{subprocess_env['LD_PRELOAD']}"
    subprocess_env["LD_PRELOAD"] = ld_preload

    try:
        result = subprocess.run(
            command,
            cwd=work_dir,
            shell=True,
            env=subprocess_env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        if result.returncode != 0:
            raise RuntimeError(
                f"Ray worker host={os.uname().nodename} accelerator_ids={accelerator_ids} "
                f"CUDA_VISIBLE_DEVICES={subprocess_env.get('CUDA_VISIBLE_DEVICES', 'unset')} "
                f"work_dir={work_dir} staged={staged} command={' '.join(command)}\n"
                f"Ray CUDA worker ready: device_count={__import__('torch').cuda.device_count()}\n"
                f"{output}"
            )
        return (
            f"Ray worker host={os.uname().nodename} accelerator_ids={accelerator_ids} "
            f"CUDA_VISIBLE_DEVICES={subprocess_env.get('CUDA_VISIBLE_DEVICES', 'unset')} "
            f"work_dir={work_dir} staged={staged} command={' '.join(command)}\n"
            f"Ray CUDA worker ready: device_count={__import__('torch').cuda.device_count()}\n"
            f"{output}"
        )
    except subprocess.TimeoutExpired as e:
        output = e.stdout or ""
        if e.stderr:
            output += "\n" + e.stderr
        raise RuntimeError(
            f"Ray worker timed out after {e.timeout}s: command={' '.join(command)}\n{output}"
        )


try:
    result = ray.get(run_on_gpu_worker.remote(remote_dir, remote_fallback_dir, " ".join(command), archive_bytes))
    print(result)
finally:
    ray.shutdown()
PY
