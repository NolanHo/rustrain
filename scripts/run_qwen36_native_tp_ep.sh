#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
case "$mode" in
    local-smoke)
        test_name=native_smoke
        test_source=crates/rustrain-qwen3-6/tests/native_smoke.cpp
        ;;
    smoke|gpu-metadata-smoke|tri-smoke|tri-replicated-smoke)
        test_name=native_tp_ep_smoke
        test_source=crates/rustrain-qwen3-6/tests/native_tp_ep_smoke.cpp
        ;;
    sequence-parallel-smoke)
        test_name=native_sequence_parallel_smoke
        test_source=crates/rustrain-qwen3-6/tests/native_sequence_parallel_smoke.cpp
        ;;
    tp-attention-smoke)
        test_name=native_tp_attention_smoke
        test_source=crates/rustrain-qwen3-6/tests/native_tp_attention_smoke.cpp
        ;;
    mtp-dynamic-smoke)
        test_name=native_mtp_dynamic_smoke
        test_source=crates/rustrain-qwen3-6/tests/native_mtp_dynamic_smoke.cpp
        ;;
    mtp-dp-smoke)
        test_name=native_mtp_dp_smoke
        test_source=crates/rustrain-qwen3-6/tests/native_mtp_dp_smoke.cpp
        ;;
    mtp-tp-smoke)
        test_name=native_mtp_tp_smoke
        test_source=crates/rustrain-qwen3-6/tests/native_mtp_tp_smoke.cpp
        ;;
    mtp-tp-ep-smoke)
        test_name=native_mtp_tp_smoke
        test_source=crates/rustrain-qwen3-6/tests/native_mtp_tp_smoke.cpp
        ;;
    cp-gdn-smoke)
        test_name=native_cp_gdn_smoke
        test_source=crates/rustrain-qwen3-6/tests/native_cp_gdn_smoke.cpp
        ;;
    cp-attention-smoke)
        test_name=native_cp_attention_smoke
        test_source=crates/rustrain-qwen3-6/tests/native_cp_attention_smoke.cpp
        ;;
    ep-smoke)
        test_name=native_ep_smoke
        test_source=crates/rustrain-qwen3-6/tests/native_ep_smoke.cpp
        ;;
    ep-bench)
        test_name=native_ep_bench
        test_source=crates/rustrain-qwen3-6/tests/native_ep_bench.cpp
        ;;
    bench)
        test_name=native_tp_ep_bench
        test_source=crates/rustrain-qwen3-6/tests/native_tp_ep_bench.cpp
        ;;
    pp-cp-comm-smoke)
        test_name=native_pp_cp_comm_smoke
        test_source=crates/rustrain-qwen3-6/tests/native_pp_cp_comm_smoke.cpp
        ;;
    pp-train-smoke)
        test_name=native_pp_train_smoke
        test_source=crates/rustrain-qwen3-6/tests/native_pp_train_smoke.cpp
        ;;
    *)
    echo "usage: $0 [local-smoke|smoke|sequence-parallel-smoke|tp-attention-smoke|cp-attention-smoke|mtp-dynamic-smoke|mtp-dp-smoke|mtp-tp-smoke|mtp-tp-ep-smoke|cp-gdn-smoke|gpu-metadata-smoke|tri-smoke|tri-replicated-smoke|ep-smoke|ep-bench|bench|pp-cp-comm-smoke|pp-train-smoke]" >&2
        exit 2
        ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python3}"
if [[ -n "${TORCH_INCLUDE_PATH:-}" && -n "${TORCH_LIB_PATH:-}" && -n "${GLIBCXX_USE_CXX11_ABI:-}" ]]; then
    torch_include="$TORCH_INCLUDE_PATH"
    torch_lib="$TORCH_LIB_PATH"
    cxx11_abi="$GLIBCXX_USE_CXX11_ABI"
    site_packages="${PYTHON_SITE_PACKAGES:-${torch_include%/torch/include}}"
    torch_version="${TORCH_VERSION:-prebuilt}"
else
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
    torch_include="${torch_config[0]}"
    torch_lib="${torch_config[1]}"
    cxx11_abi="${torch_config[2]}"
    site_packages="${torch_config[3]}"
    torch_version="${torch_config[4]}"
fi

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
for tool in "$nvcc" g++ sha256sum stat; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "required build tool not found: $tool" >&2
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

fingerprint="$({
    printf '%s\n' \
        "python=$python_bin" \
        "torch_version=$torch_version" \
        "torch_include=$torch_include" \
        "torch_lib=$torch_lib" \
        "cxx11_abi=$cxx11_abi" \
        "cuda_home=$cuda_home" \
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

native_root="${NATIVE_BUILD_DIR:-target/native-qwen36-tp-ep}"
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

test_bin="$native_dir/$test_name"
if [[ ! -e "$test_bin" || "$test_source" -nt "$test_bin" || "$kernel_lib" -nt "$test_bin" ]]; then
    g++ "$test_source" -o "$test_bin" \
        -std=c++17 -O2 "-D_GLIBCXX_USE_CXX11_ABI=$cxx11_abi" \
        "-I$torch_include" "-I$cuda_include" "-I$nccl_include" \
        "-L$native_dir" "-L$torch_lib" "-L$nccl_lib" "-L$cuda_home/lib64" \
        "-Wl,-rpath,$native_dir" "-Wl,-rpath,$torch_lib" \
        "-Wl,-rpath,$nccl_lib" "-Wl,-rpath,$cuda_home/lib64" \
        -Wl,--no-as-needed -Wl,--allow-shlib-undefined \
        -lqwen36_kernels -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda \
        -lcudart "$nccl_link"
fi

cu13_lib="$site_packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$native_dir:$torch_lib:$nccl_lib:$cuda_home/lib64:$cu13_lib:${LD_LIBRARY_PATH:-}"
if [[ -z "${RUSTRAIN_NCCL_RUN_ID:-}" &&
      ( -z "${RUSTRAIN_RUN_ID:-}" || -z "${RUSTRAIN_ATTEMPT_ID:-}" ) ]]; then
    export RUSTRAIN_NCCL_RUN_ID="qwen36-tp-ep-$$"
fi

if [[ "$mode" == "local-smoke" ]]; then
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 TP_SIZE=1 EP_SIZE=1 DP_SIZE=1 \
        "$test_bin"
elif [[ "$mode" == "sequence-parallel-smoke" ]]; then
    QWEN36_SEQUENCE_PARALLEL=1 TP_SIZE=2 CP_SIZE=1 EP_SIZE=1 DP_SIZE=1 PP_SIZE=1 \
        "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node=2 --no-python "$test_bin"
elif [[ "$mode" == "tp-attention-smoke" ]]; then
    TP_SIZE=2 CP_SIZE=1 EP_SIZE=1 DP_SIZE=1 PP_SIZE=1 \
        "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node=2 --no-python "$test_bin"
elif [[ "$mode" == "mtp-dynamic-smoke" ]]; then
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 TP_SIZE=1 CP_SIZE=1 EP_SIZE=1 DP_SIZE=1 PP_SIZE=1 \
        "$test_bin"
elif [[ "$mode" == "mtp-dp-smoke" ]]; then
    TP_SIZE=1 CP_SIZE=1 EP_SIZE=1 DP_SIZE=2 PP_SIZE=1 \
        "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node=2 --no-python "$test_bin"
elif [[ "$mode" == "mtp-tp-smoke" ]]; then
    TP_SIZE=2 CP_SIZE=1 EP_SIZE=1 DP_SIZE=1 PP_SIZE=1 \
        "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node=2 --no-python "$test_bin"
elif [[ "$mode" == "mtp-tp-ep-smoke" ]]; then
    QWEN36_TEST_MTP_MOE=1 QWEN36_TEST_MTP_EP=1 \
    QWEN36_EP_A2A=1 QWEN36_EP_A2A_SHARDED=1 QWEN36_EP_A2A_PACKED=1 \
    TP_SIZE=2 CP_SIZE=1 EP_SIZE=2 DP_SIZE=1 PP_SIZE=1 \
        "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node=4 --no-python "$test_bin"
elif [[ "$mode" == "cp-gdn-smoke" ]]; then
    TP_SIZE=1 CP_SIZE=2 EP_SIZE=1 DP_SIZE=1 PP_SIZE=1 \
        "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node=2 --no-python "$test_bin"
elif [[ "$mode" == "cp-attention-smoke" ]]; then
    cp_test_size="${QWEN36_TEST_CP_SIZE:-2}"
    QWEN36_TEST_CP_RING="${QWEN36_TEST_CP_RING:-0}" \
    QWEN36_CP_FULL_ATTENTION_KV_GATHER="$([[ "${QWEN36_TEST_CP_RING:-0}" == "1" ]] && echo 0 || echo 1)" \
    QWEN36_CP_FULL_ATTENTION_RING="$([[ "${QWEN36_TEST_CP_RING:-0}" == "1" ]] && echo 1 || echo 0)" \
    TP_SIZE=1 CP_SIZE="$cp_test_size" EP_SIZE=1 DP_SIZE=1 PP_SIZE=1 \
        "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node="$cp_test_size" --no-python "$test_bin"
elif [[ "$mode" == "pp-cp-comm-smoke" ]]; then
    TP_SIZE=1 CP_SIZE=2 EP_SIZE=1 DP_SIZE=1 PP_SIZE=2 \
        "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node=4 --no-python "$test_bin"
elif [[ "$mode" == "pp-train-smoke" ]]; then
    pp_train_world="${PP_TRAIN_WORLD:-2}"
    if [[ "$pp_train_world" -lt 2 ]]; then
        echo "PP_TRAIN_WORLD must be >= 2" >&2
        exit 1
    fi
    TP_SIZE=1 CP_SIZE=1 EP_SIZE=1 DP_SIZE=1 PP_SIZE="$pp_train_world" \
        "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node="$pp_train_world" --no-python "$test_bin"
elif [[ "$mode" == "tri-smoke" || "$mode" == "tri-replicated-smoke" ]]; then
    sharded_a2a=1
    if [[ "$mode" == "tri-replicated-smoke" ]]; then
        sharded_a2a=0
    fi
    TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 RUSTRAIN_DATA_PARALLEL=1 \
    QWEN36_EP_A2A=1 QWEN36_EP_A2A_SHARDED="$sharded_a2a" \
        "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node=8 --no-python "$test_bin"
elif [[ "$mode" == "smoke" || "$mode" == "gpu-metadata-smoke" || "$mode" == "bench" ]]; then
    gpu_metadata_env=()
    if [[ "$mode" == "gpu-metadata-smoke" ]]; then
        gpu_metadata_env=(QWEN36_EP_A2A_GPU_METADATA=1)
    fi
    env "${gpu_metadata_env[@]}" TP_SIZE=2 EP_SIZE=2 DP_SIZE=1 \
    QWEN36_EP_A2A=1 QWEN36_EP_A2A_SHARDED=1 \
        "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node=4 --no-python "$test_bin"
elif [[ "$mode" == "ep-bench" ]]; then
    ep_bench_world="${EP_BENCH_WORLD:-4}"
    TP_SIZE=1 EP_SIZE="$ep_bench_world" DP_SIZE=1 \
    QWEN36_EP_A2A=1 QWEN36_EP_A2A_SHARDED=1 \
        "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node="$ep_bench_world" \
            --no-python "$test_bin"
else
    TP_SIZE=1 EP_SIZE=2 DP_SIZE=1 \
    QWEN36_EP_A2A=1 QWEN36_EP_A2A_SHARDED=1 \
        "$python_bin" -m torch.distributed.run --standalone \
            --nnodes=1 --nproc-per-node=2 --no-python "$test_bin"
fi
