# Source this file before running tch-rs commands against the local Python
# PyTorch install.
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="/share/code/nolanho/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
