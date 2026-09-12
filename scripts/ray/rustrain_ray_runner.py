"""rustrain_ray_runner — Python helper for running shell commands on Ray GPU workers.

Called via rayrust's task_call_python(). Inherits env from the Ray worker
(which has correct CUDA driver paths), prepends extra paths, runs subprocess.
"""
import subprocess
import os


def run_command(cmd, num_gpus=1, env_overrides=None):
    """Run a shell command on this Ray worker, inheriting the environment.

    Args:
        cmd: Shell command string.
        num_gpus: Number of GPUs to expose (sets CUDA_VISIBLE_DEVICES).
        env_overrides: Dict of additional env vars to set.

    Returns:
        Dict with stdout, stderr, returncode.
    """
    env = os.environ.copy()

    # Prepend extra library paths to the EXISTING LD_LIBRARY_PATH (don't replace)
    extra_lib_paths = [
        "/usr/local/lib/python3.13/dist-packages/ray/cpp/lib",
        "/usr/local/cuda-13.0/compat",
        "/usr/local/lib/python3.13/dist-packages/torch/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/cuda_runtime/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/cuda_cupti/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/cuda_nvrtc/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/cublas/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/cudnn/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/cufft/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/curand/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/cusolver/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/cusparse/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/cusparselt/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/nccl/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/nvjitlink/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/nvshmem/lib",
        "/usr/local/lib/python3.13/dist-packages/nvidia/nvtx/lib",
        "/usr/local/cuda/lib64",
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(extra_lib_paths) + ":" + existing

    # Ensure LD_PRELOAD includes libtorch_cuda.so (needed by tch-rs)
    torch_so = "/usr/local/lib/python3.13/dist-packages/torch/lib/libtorch_cuda.so"
    existing_preload = env.get("LD_PRELOAD", "")
    if torch_so not in existing_preload:
        env["LD_PRELOAD"] = torch_so + (":" + existing_preload if existing_preload else "")

    # Expose first N GPUs
    if num_gpus > 0:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    if env_overrides:
        for k, v in env_overrides.items():
            env[k] = v

    result = subprocess.run(
        cmd,
        shell=True,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
