#!/usr/bin/env bash
set -euo pipefail

/opt/venv/bin/python - <<'PY'
import os
import socket
import sys

try:
    import torch
except Exception as exc:
    print(f"GPU worker guard failed: could not import torch: {exc}", file=sys.stderr)
    sys.exit(1)

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(
        "GPU worker guard passed: "
        f"host={socket.gethostname()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"device_count={torch.cuda.device_count()}"
    )
    sys.exit(0)

print("GPU worker guard failed: refusing to run without a visible CUDA GPU.", file=sys.stderr)
print(
    f"host={socket.gethostname()} "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
    f"torch.cuda.is_available={torch.cuda.is_available()} "
    f"torch.cuda.device_count={torch.cuda.device_count()}",
    file=sys.stderr,
)
print(
    "Run this script through scripts/gpu_run.sh so it executes inside a Ray GPU worker.",
    file=sys.stderr,
)
sys.exit(1)
PY
