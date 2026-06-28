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

// ── FP8 E4M3FN byte-level dequant ──
//
// PyTorch's to(kFloat) on FP8 tensors triggers an internal block-wise view
// optimization that crashes on non-128-aligned dimensions (e.g. [576, 6144]).
// This function bypasses PyTorch's FP8 type handling entirely by reading
// raw FP8 bytes and converting each to float32 using bit manipulation.
//
// The output is a plain F32 tensor — safe for any downstream operation.

static inline float fp8_e4m3fn_to_f32(uint8_t b) {
    uint32_t sign = (b >> 7) & 1;
    uint32_t exp   = (b >> 3) & 0x0F;
    uint32_t mant  =  b        & 0x07;

    float val;
    if (exp == 0 && mant == 0) {
        val = 0.0f;
    } else if (exp == 0) {
        // Subnormal: (mant / 8) * 2^(-6)
        val = (static_cast<float>(mant) / 8.0f) * 0.015625f;  // 2^-6
    } else {
        // Normal: (1 + mant/8) * 2^(exp - 7)
        val = (1.0f + static_cast<float>(mant) / 8.0f) *
              std::ldexp(1.0f, static_cast<int>(exp) - 7);
    }
    return sign ? -val : val;
}

// v4_dequant_fp8_raw — byte-level FP8→F32 conversion (no PyTorch FP8 ops)
// Takes a GPU FP8 tensor pointer and returns a GPU F32 tensor.
// Completely bypasses PyTorch's to() / copy_() which trigger view bugs.
void* v4_dequant_fp8_raw(void* tensor_ptr) {
    try {
        at::Tensor& fp8 = *reinterpret_cast<at::Tensor*>(tensor_ptr);

        // Get shape and total elements
        auto sizes = fp8.sizes();
        int64_t numel = fp8.numel();

        // Read FP8 bytes from GPU to CPU
        at::Tensor fp8_cpu = fp8.to(at::kCPU).contiguous();
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(fp8_cpu.data_ptr());

        // Convert each FP8 byte to float32
        std::vector<float> f32_data(numel);
        for (int64_t i = 0; i < numel; i++) {
            f32_data[i] = fp8_e4m3fn_to_f32(bytes[i]);
        }

        // Create F32 tensor from converted data
        at::Tensor f32_cpu = at::from_blob(
            f32_data.data(),
            sizes,
            at::TensorOptions().dtype(at::kFloat)
        ).clone();

        // Move to same device as input
        at::Tensor f32_gpu = f32_cpu.to(fp8.device());

        return new at::Tensor(std::move(f32_gpu));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

} // extern "C"
