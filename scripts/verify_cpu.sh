#!/usr/bin/env bash
set -euo pipefail

echo "Local CPU smoke/verification is forbidden for rustrain development." >&2
echo "" >&2
echo "Run on a GPU server via one of these:" >&2
echo "" >&2
echo "  Direct SSH (preferred, e.g. H20e3):" >&2
echo "    scripts/gpu_run_ssh.sh <command>" >&2
echo "    scripts/verify_gpu_ssh.sh" >&2
echo "    scripts/verify_gpu_distributed_ssh.sh" >&2
echo "" >&2
echo "  Ray cluster (legacy):" >&2
echo "    scripts/ray/gpu_run.sh <command>" >&2
echo "    scripts/ray/verify_gpu.sh" >&2
echo "    scripts/ray/verify_gpu_distributed.sh" >&2
exit 1
