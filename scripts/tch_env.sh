# Legacy helper for non-A800 tch-rs experiments. Project verification must use
# scripts/gpu_run_ssh.sh (direct SSH) or scripts/ray/gpu_run.sh (Ray GPU workers).
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="/path/to/python-venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
