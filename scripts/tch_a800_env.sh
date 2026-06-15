# Source this file on the A800 worker before running tch-rs commands against
# the worker's CUDA-enabled Python PyTorch install.
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/rustrain-target-a800}"

_RU_TORCH_SITE="${RUSTRAIN_TORCH_SITE:-/usr/local/lib/python3.12/dist-packages}"
_RU_TORCH_LIB="${_RU_TORCH_SITE}/torch/lib"
_RU_NVIDIA="${_RU_TORCH_SITE}/nvidia"

export LD_PRELOAD="${_RU_TORCH_LIB}/libtorch_cuda.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export LD_LIBRARY_PATH="${_RU_TORCH_LIB}:${_RU_NVIDIA}/cuda_runtime/lib:${_RU_NVIDIA}/cuda_cupti/lib:${_RU_NVIDIA}/cuda_nvrtc/lib:${_RU_NVIDIA}/cublas/lib:${_RU_NVIDIA}/cudnn/lib:${_RU_NVIDIA}/cufft/lib:${_RU_NVIDIA}/curand/lib:${_RU_NVIDIA}/cusolver/lib:${_RU_NVIDIA}/cusparse/lib:${_RU_NVIDIA}/cusparselt/lib:${_RU_NVIDIA}/nccl/lib:${_RU_NVIDIA}/nvjitlink/lib:${_RU_NVIDIA}/nvshmem/lib:${_RU_NVIDIA}/nvtx/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
