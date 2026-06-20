#!/usr/bin/env bash
set -euo pipefail

echo "Local CPU smoke/verification is not a supported development path." >&2
echo "Run scripts/gpu_run.sh <command> or scripts/verify_gpu.sh to execute on a Ray GPU worker." >&2
exit 1
