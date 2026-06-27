// fp8_gemm.cpp — C++ shim for FP8 GEMM + safetensors loading
//
// Two functions:
// 1. v4_fp8_scaled_mm — block-wise FP8 GEMM via at::_scaled_mm (CUTLASS)
// 2. v4_create_tensor — create at::Tensor from raw bytes (for FP8 loading without Python)
//
// Compiled with g++ (no nvcc) — links against libtorch.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cstring>
#include <vector>

// at::_scaled_mm is available via #include <ATen/ops/_scaled_mm.h>
// No forward declaration needed — the header provides the inline definition.

extern "C" {

// ── FP8 block-wise GEMM ──

void* v4_fp8_scaled_mm(
    void* a_ptr,
    void* b_ptr,
    void* scale_a_ptr,
    void* scale_b_ptr
) {
    try {
        const at::Tensor& a = *reinterpret_cast<at::Tensor*>(a_ptr);
        const at::Tensor& b = *reinterpret_cast<at::Tensor*>(b_ptr);
        const at::Tensor& scale_a = *reinterpret_cast<at::Tensor*>(scale_a_ptr);
        const at::Tensor& scale_b = *reinterpret_cast<at::Tensor*>(scale_b_ptr);

        // Cast inputs to FP8 if they're not already (tch-rs can't create FP8 tensors)
        at::Tensor a_fp8 = (a.scalar_type() == at::kFloat8_e4m3fn) ? a : a.to(at::kFloat8_e4m3fn);
        at::Tensor b_fp8 = (b.scalar_type() == at::kFloat8_e4m3fn) ? b : b.to(at::kFloat8_e4m3fn);

        at::Tensor result = at::_scaled_mm(
            a_fp8,
            b_fp8.t(),
            scale_a,
            scale_b,
            c10::nullopt,  // bias
            c10::nullopt,  // scale_result
            at::kBFloat16, // output dtype — ALWAYS bf16
            true            // use_fast_accum
        );

        return new at::Tensor(std::move(result));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void v4_fp8_free_tensor(void* tensor_ptr) {
    if (tensor_ptr) {
        delete reinterpret_cast<at::Tensor*>(tensor_ptr);
    }
}

// ── Tensor creation from raw bytes (no Python) ──
//
// Creates an at::Tensor from raw data pointer, shape, and dtype.
// The data is copied to the specified CUDA device.
//
// dtype_code:
//   0 = F8_E4M3 (float8_e4m3fn)  — V4 weights
//   1 = F32                       — float scales
//   2 = BF16                      — bf16 weights
//   3 = U8                        — uint8 (ue8m0 scales, before conversion)
//   4 = I64                       — int64 (token ids)

void* v4_create_tensor(
    const void* data,       // raw bytes (CPU memory)
    const int64_t* shape,   // shape array
    int shape_len,          // number of dimensions
    int dtype_code,         // see above
    int device_id           // CUDA device (-1 for CPU)
) {
    try {
        // Build shape vector
        std::vector<int64_t> sizes(shape, shape + shape_len);

        // Map dtype code to at::ScalarType
        at::ScalarType dtype;
        switch (dtype_code) {
            case 0: dtype = at::kFloat8_e4m3fn; break;
            case 1: dtype = at::kFloat; break;
            case 2: dtype = at::kBFloat16; break;
            case 3: dtype = at::kByte; break;  // uint8
            case 4: dtype = at::kLong; break;  // int64
            default: dtype = at::kFloat; break;
        }

        // Create CPU tensor from raw data
        at::TensorOptions opts = at::TensorOptions().dtype(dtype);
        at::Tensor cpu_tensor = at::from_blob(
            const_cast<void*>(data),
            sizes,
            opts
        ).clone();  // clone to own the memory

        // Move to CUDA if requested
        if (device_id >= 0) {
            cpu_tensor = cpu_tensor.to(at::Device(at::Device::Type::CUDA, device_id));
        }

        return new at::Tensor(std::move(cpu_tensor));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

} // extern "C"
