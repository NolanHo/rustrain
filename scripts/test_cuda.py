import os, ctypes

# 1. Write C wrapper that sets CUDA_VISIBLE_DEVICES + LD_LIBRARY_PATH + LD_PRELOAD
c_code = r"""
#include <stdlib.h>
#include <unistd.h>
int main(int argc, char **argv) {
    setenv("CUDA_VISIBLE_DEVICES", argv[1], 1);
    setenv("LD_LIBRARY_PATH",
        "/usr/local/cuda-13.0/compat:"
        "/usr/local/lib/python3.13/dist-packages/torch/lib:"
        "/usr/local/lib/python3.13/dist-packages/nvidia/cuda_runtime/lib:"
        "/usr/local/lib/python3.13/dist-packages/nvidia/cublas/lib:"
        "/usr/local/lib/python3.13/dist-packages/nvidia/cudnn/lib:"
        "/usr/local/lib/python3.13/dist-packages/nvidia/nccl/lib:"
        "/usr/local/lib/python3.13/dist-packages/nvidia/nvjitlink/lib:"
        "/usr/local/cuda/lib64:"
        "/usr/local/nvidia/lib:"
        "/usr/local/nvidia/lib64", 1);
    setenv("LD_PRELOAD",
        "/usr/local/lib/python3.13/dist-packages/torch/lib/libtorch_cuda.so", 1);
    execvp(argv[2], &argv[2]);
    return 1;
}
"""
with open("/dev/shm/cw.c", "w") as f:
    f.write(c_code)
print("C file written")

# 2. Compile
libc = ctypes.CDLL("libc.so.6")
ret = libc.system(b"cc -o /dev/shm/cw /dev/shm/cw.c 2>&1")
print(f"Compile exit: {ret}")

# 3. Verify C file content
with open("/dev/shm/cw.c") as f:
    content = f.read()
    print(f"C file OK: {len(content)} bytes")

# 4. Run: cw 0 python3 -c "import torch; print(cuda)"
ret = libc.system(b'/dev/shm/cw 0 python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())" 2>&1')
print(f"Run exit: {ret}")
