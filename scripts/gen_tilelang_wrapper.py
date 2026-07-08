#!/usr/bin/env python3
"""Generate Tilelang fused kernels as a C .so with extern "C" entry points.

Tilelang's export_library produces a TVM FFI module (not directly dlopen-able).
This script instead:
1. Uses Tilelang to generate CUDA C++ source code
2. Wraps it with extern "C" functions
3. Compiles with nvcc to produce a standard .so
"""
import tilelang
import tilelang.language as T
import torch
import os, sys

@tilelang.jit(out_idx=[2], execution_backend="tvm_ffi", target="cuda")
def fused_swiglu(M=128, I=128, limit=10.0, block_m=16, threads=128):
    @T.prim_func
    def main(gate_out: T.Tensor((M, I), "bfloat16"), up_out: T.Tensor((M, I), "bfloat16"), activated: T.Tensor((M, I), "bfloat16")):
        with T.Kernel(T.ceildiv(M, block_m), T.ceildiv(I, 64), threads=threads) as (bx, by):
            gate_shared = T.alloc_shared((block_m, 64), "bfloat16")
            up_shared = T.alloc_shared((block_m, 64), "bfloat16")
            out_local = T.alloc_fragment((block_m, 64), "bfloat16")
            for ki in T.serial(T.ceildiv(I, 64)):
                T.copy(gate_out[bx*block_m:(bx+1)*block_m, ki*64:(ki+1)*64], gate_shared)
                T.copy(up_out[bx*block_m:(bx+1)*block_m, ki*64:(ki+1)*64], up_shared)
                for i in T.Parallel(block_m):
                    for j in T.serial(64):
                        g = T.cast(gate_shared[i, j], "float32")
                        u = T.cast(up_shared[i, j], "float32")
                        v = T.cast(1.0, "float32") / (T.cast(1.0, "float32") + T.exp(-g)) * u
                        if limit > 0.0:
                            v = T.max(v, T.cast(-limit, "float32"))
                            v = T.min(v, T.cast(limit, "float32"))
                        out_local[i, j] = T.cast(v, "bfloat16")
                T.copy(out_local, activated[bx*block_m:(bx+1)*block_m, ki*64:(ki+1)*64])
    return main

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    
    k = fused_swiglu()
    # Trigger compilation
    gate = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
    _ = k(gate, up)
    
    # Get the CUDA source code
    src = k.get_kernel_source()
    print(f"CUDA source length: {len(src)}")
    
    # Write wrapper .cu file
    wrapper_path = os.path.join(out_dir, "tilelang_wrapper.cu")
    so_path = os.path.join(out_dir, "libtilelang_fused.so")
    
    with open(wrapper_path, "w") as f:
        f.write(src)
        f.write("\n\n")
        f.write('#include <cuda_runtime.h>\n\n')
        f.write('extern "C" void tilelang_fused_swiglu(\n')
        f.write('    void* gate, void* up, void* out,\n')
        f.write('    int64_t M, int64_t I, double limit\n')
        f.write(') {\n')
        f.write('    dim3 grid((M + 15) / 16, (I + 63) / 64);\n')
        f.write('    dim3 block(128);\n')
        f.write('    int shared_mem = 2 * 16 * 64 * 2; // 2 tensors, 16x64, bfloat16\n')
        f.write('    main_kernel<<<grid, block, shared_mem>>>(\n')
        f.write('        (const __nv_bfloat16*)up,\n')
        f.write('        (const __nv_bfloat16*)gate,\n')
        f.write('        (__nv_bfloat16*)out\n')
        f.write('    );\n')
        f.write('}\n')
    
    print(f"Wrote wrapper to {wrapper_path}")
    
    # Compile with nvcc
    import subprocess
    # Find tilelang include path
    import tilelang as _tl
    tl_path = os.path.dirname(_tl.__file__)
    tl_include = os.path.join(tl_path, "src")
    
    result = subprocess.run([
        "nvcc", "-shared", "-Xcompiler", "-fPIC", "-O2", "-std=c++17",
        f"-I{tl_include}",
        "-o", so_path,
        wrapper_path,
        "-lcudart"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Compiled to {so_path}")
    else:
        print(f"nvcc failed: {result.stderr[:500]}")

if __name__ == "__main__":
    main()
