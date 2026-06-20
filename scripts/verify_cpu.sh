#!/usr/bin/env bash
set -euo pipefail

echo "Local CPU smoke/verification is forbidden for rustrain development." >&2
echo "Run every test, smoke, parity, and training check on a Ray GPU worker:" >&2
echo "  scripts/gpu_run.sh <command>" >&2
echo "  scripts/verify_gpu.sh" >&2
exit 1
