#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 {smoke|bench-single|bench-tp2}" >&2
    exit 2
}

mode="${1:-}"
case "$mode" in
    smoke|bench-single|bench-tp2) ;;
    *) usage ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python3}"
readarray -t torch_config < <(
    "$python_bin" - <<'PY'
import pathlib
import torch

root = pathlib.Path(torch.__file__).resolve().parent
print(root / "include")
print(root / "lib")
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
print(root.parent)
print(torch.__version__)
PY
)
torch_include="${TORCH_INCLUDE_PATH:-${torch_config[0]}}"
torch_lib="${TORCH_LIB_PATH:-${torch_config[1]}}"
cxx11_abi="${GLIBCXX_USE_CXX11_ABI:-${torch_config[2]}}"
site_packages="${torch_config[3]}"
torch_version="${torch_config[4]}"

cuda_home="${CUDA_HOME:-/usr/local/cuda}"
cuda_include="${CUDA_INCLUDE_PATH:-$cuda_home/include}"
nccl_include="${NCCL_INCLUDE_PATH:-$site_packages/nvidia/nccl/include}"
nccl_lib="${NCCL_LIB_PATH:-$site_packages/nvidia/nccl/lib}"

for required in \
    "$torch_include/ATen/ATen.h" \
    "$torch_lib/libtorch.so" \
    "$cuda_include/cuda_runtime.h" \
    "$nccl_include/nccl.h"; do
    if [[ ! -e "$required" ]]; then
        echo "required prebuilt dependency file not found: $required" >&2
        exit 1
    fi
done

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export TORCH_INCLUDE_PATH="$torch_include"
export TORCH_LIB_PATH="$torch_lib"
export CUDA_INCLUDE_PATH="$cuda_include"
export NCCL_INCLUDE_PATH="$nccl_include"
export NCCL_LIB_PATH="$nccl_lib"
export GLIBCXX_USE_CXX11_ABI="$cxx11_abi"

if [[ -e "$nccl_lib/libnccl.so" ]]; then
    nccl_link="-lnccl"
    nccl_file="$nccl_lib/libnccl.so"
elif [[ -e "$nccl_lib/libnccl.so.2" ]]; then
    nccl_link="-l:libnccl.so.2"
    nccl_file="$nccl_lib/libnccl.so.2"
else
    echo "prebuilt NCCL library not found under $nccl_lib" >&2
    exit 1
fi

nvcc="$cuda_home/bin/nvcc"
if [[ ! -x "$nvcc" ]]; then
    echo "CUDA compiler not found: $nvcc" >&2
    exit 1
fi

for tool in g++ sha256sum stat; do
    if ! command -v "$tool" >/dev/null; then
        echo "required build tool not found: $tool" >&2
        exit 1
    fi
done

fingerprint="$({
    printf '%s\n' \
        "python=$python_bin" \
        "torch_version=$torch_version" \
        "torch_include=$torch_include" \
        "torch_lib=$torch_lib" \
        "cxx11_abi=$cxx11_abi" \
        "cuda_home=$cuda_home" \
        "cuda_include=$cuda_include" \
        "nccl_include=$nccl_include" \
        "nccl_lib=$nccl_lib" \
        "nccl_link=$nccl_link"
    g++ --version
    "$nvcc" --version
    sha256sum "$0"
    stat -Lc '%n:%s:%Y' \
        "$torch_lib/libtorch.so" \
        "$torch_lib/libtorch_cuda.so" \
        "$torch_lib/libc10.so" \
        "$nccl_file" \
        "$cuda_home/lib64/libcudart.so"
} | sha256sum | cut -c1-20)"
native_root="${NATIVE_BUILD_DIR:-target/native-qwen36-gdn}"
native_dir="$native_root/$fingerprint"
mkdir -p "$native_dir"

kernel_lib="$native_dir/libqwen36_kernels.so"
kernel_sources=(
    crates/rustrain-qwen3-6/kernels/qwen3_6_kernels.cpp
    crates/rustrain-qwen3-6/kernels/delta_rule.cu
    crates/rustrain-qwen3-6/kernels/delta_rule.cuh
    crates/rustrain-qwen3-6/kernels/fused_kernels.cu
)
rebuild_kernel=false
if [[ ! -e "$kernel_lib" ]]; then
    rebuild_kernel=true
else
    for source in "${kernel_sources[@]}"; do
        if [[ "$source" -nt "$kernel_lib" ]]; then
            rebuild_kernel=true
            break
        fi
    done
fi

if [[ "$rebuild_kernel" == true ]]; then
    cuda_objects=()
    for source in delta_rule fused_kernels; do
        object="$native_dir/$source.o"
        "$nvcc" -c "crates/rustrain-qwen3-6/kernels/$source.cu" -o "$object" \
            -O2 -std=c++17 "-D_GLIBCXX_USE_CXX11_ABI=$cxx11_abi" \
            "-I$torch_include" "-I$cuda_include" -Xcompiler -fPIC
        cuda_objects+=("$object")
    done

    g++ -shared -fPIC -std=c++17 -O2 \
        "-D_GLIBCXX_USE_CXX11_ABI=$cxx11_abi" \
        crates/rustrain-qwen3-6/kernels/qwen3_6_kernels.cpp \
        "${cuda_objects[@]}" -o "$kernel_lib" \
        "-I$torch_include" "-I$cuda_include" "-I$nccl_include" \
        "-L$torch_lib" "-L$nccl_lib" "-L$cuda_home/lib64" \
        "-Wl,-rpath,$torch_lib" "-Wl,-rpath,$nccl_lib" \
        "-Wl,-rpath,$cuda_home/lib64" -Wl,--no-as-needed \
        -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lcudart "$nccl_link"
fi
kernel_dir="$native_dir"

common_flags=(
    -std=c++17 -O2 "-D_GLIBCXX_USE_CXX11_ABI=$cxx11_abi"
    "-I$torch_include" "-I$cuda_include" "-I$nccl_include"
    "-L$kernel_dir" "-L$torch_lib" "-L$nccl_lib" "-L$cuda_home/lib64"
    "-Wl,-rpath,$kernel_dir" "-Wl,-rpath,$torch_lib"
    "-Wl,-rpath,$nccl_lib" "-Wl,-rpath,$cuda_home/lib64"
    -Wl,--no-as-needed -Wl,--allow-shlib-undefined
    -lqwen36_kernels -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda
    -lcudart "$nccl_link"
)

smoke_bin="$native_dir/native_tp_gdn_smoke"
bench_bin="$native_dir/native_tp_gdn_bench"
smoke_source=crates/rustrain-qwen3-6/tests/native_tp_gdn_smoke.cpp
bench_source=crates/rustrain-qwen3-6/tests/native_tp_gdn_bench.cpp
if [[ ! -e "$smoke_bin" || "$smoke_source" -nt "$smoke_bin" || "$kernel_lib" -nt "$smoke_bin" ]]; then
    g++ "$smoke_source" -o "$smoke_bin" "${common_flags[@]}"
fi
if [[ ! -e "$bench_bin" || "$bench_source" -nt "$bench_bin" || "$kernel_lib" -nt "$bench_bin" ]]; then
    g++ "$bench_source" -o "$bench_bin" "${common_flags[@]}"
fi

cu13_lib="$site_packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$kernel_dir:$torch_lib:$nccl_lib:$cuda_home/lib64:$cu13_lib:${LD_LIBRARY_PATH:-}"
export RUSTRAIN_NCCL_RUN_ID="${RUSTRAIN_NCCL_RUN_ID:-qwen36-gdn-$$}"

case "$mode" in
    smoke)
        TP_SIZE=2 "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node=2 --no-python "$smoke_bin"
        ;;
    bench-single)
        BENCH_MODE=single WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 "$bench_bin"
        ;;
    bench-tp2)
        BENCH_MODE=tp2 TP_SIZE=2 "$python_bin" -m torch.distributed.run \
            --standalone --nnodes=1 --nproc-per-node=2 --no-python "$bench_bin"
        ;;
esac
