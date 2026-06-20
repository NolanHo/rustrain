# Legacy helper for non-A800 tch-rs experiments. Project verification must use
# scripts/gpu_run.sh or scripts/verify_gpu.sh so commands run on Ray GPU workers.
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="/share/code/nolanho/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
