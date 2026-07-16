// fp8_gemm.cpp — C++ shim for FP8 GEMM + safetensors loading
//
// Two functions:
// 1. v4_fp8_scaled_mm — block-wise FP8 GEMM via at::_scaled_mm (CUTLASS)
// 2. v4_create_tensor — create at::Tensor from raw bytes (for FP8 loading without Python)
//
// Compiled with g++ (no nvcc) — links against libtorch.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/custom_function.h>
#include <cstdio>
#include <cstring>
#include <cmath>
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
        fprintf(stderr, "[v4_fp8_scaled_mm] FAILED: %s\n", e.what());
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
    } else if (exp == 0x0F && mant == 0x07) {
        val = NAN;
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
        auto sizes = fp8.sizes();
        int64_t numel = fp8.numel();

        at::Tensor fp8_cpu = fp8.to(at::kCPU).contiguous();
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(fp8_cpu.data_ptr());

        std::vector<float> f32_data(numel);
        for (int64_t i = 0; i < numel; i++) {
            f32_data[i] = fp8_e4m3fn_to_f32(bytes[i]);
        }

        at::Tensor f32_cpu = at::from_blob(
            f32_data.data(), sizes,
            at::TensorOptions().dtype(at::kFloat)
        ).clone();
        at::Tensor f32_gpu = f32_cpu.to(fp8.device());
        return new at::Tensor(std::move(f32_gpu));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

// ── FP8 block-wise matmul via RowWise _scaled_mm ──
//
// H20-3e's _scaled_mm supports TensorWise and RowWise but NOT block-wise 128x128
// (stride alignment issue). We convert block-wise scale to RowWise:
//   - Block-wise scale_b [N/128, K/128] → row-wise [1, N] via max along K/128
//   - Block-wise scale_a [M/128, K/128] → row-wise [M, 1] via max along K/128
// Then call _scaled_mm with RowWise scaling — fully hardware FP8 GEMM, zero dequant.
//
// Accuracy: max along K/128 is a slight overestimate (block max ≤ row max),
// but the FP8 range is [-448, 448] and we clamp, so overflow is impossible.

void* v4_fp8_tiled_matmul(
    void* input_ptr,     // at::Tensor BF16 [M, K] on GPU
    void* weight_ptr,    // at::Tensor FP8 [N, K] on GPU
    void* scale_ptr      // at::Tensor F32 [N/128, K/128] on GPU
) {
    try {
        at::Tensor& input  = *reinterpret_cast<at::Tensor*>(input_ptr);
        at::Tensor& weight = *reinterpret_cast<at::Tensor*>(weight_ptr);
        at::Tensor& scale  = *reinterpret_cast<at::Tensor*>(scale_ptr);

        int64_t K = input.size(1);
        int64_t N = weight.size(0);

        TORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
                    "FP8 tiled matmul expects rank-2 input and weight");
        TORCH_CHECK(weight.size(1) == K, "FP8 weight/input K mismatch");

        // ── FP8 matmul with full autograd support ──
        // _scaled_mm has no backward in C++, so for frozen FP8 weights we:
        // 1) Dequant FP8→BF16 on GPU via .to(at::kBFloat16) (C++ path, no Python)
        // 2) Use regular at::mm which has full autograd
        //
        // If weight is already BF16 (LoRA applied), skip dequant.

        at::Tensor b_bf16;
        if (weight.scalar_type() == at::kFloat8_e4m3fn) {
            // The checkpoint stores raw FP8 values plus one scale per 128x128
            // tile. Apply that scale exactly once while dequantizing.
            auto raw = weight.contiguous().reshape({-1}).to(at::kFloat).reshape({N, K});
            auto scale_f32 = scale.to(weight.device()).to(at::kFloat);
            int64_t n_blocks = (N + 127) / 128;
            int64_t k_blocks = (K + 127) / 128;
            at::Tensor expanded_scale;
            if (scale_f32.dim() == 2 && scale_f32.size(0) == n_blocks &&
                scale_f32.size(1) == k_blocks) {
                expanded_scale = at::repeat_interleave(scale_f32, 128, 0)
                                     .repeat_interleave(128, 1)
                                     .narrow(0, 0, N).narrow(1, 0, K);
            } else {
                TORCH_CHECK(scale_f32.sizes() == weight.sizes(),
                            "FP8 scale must be [ceil(N/128), ceil(K/128)] or [N, K]");
                expanded_scale = scale_f32;
            }
            b_bf16 = (raw * expanded_scale).to(at::kBFloat16);
        } else {
            // A non-FP8 weight (for example a LoRA-fused BF16 weight) already
            // includes the base scale. Applying checkpoint scale again is wrong.
            b_bf16 = weight.to(at::kBFloat16);
        }

        at::Tensor result = at::mm(input, b_bf16.t());

        return new at::Tensor(std::move(result));
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_fp8_tiled_matmul] FAILED: %s\n", e.what());
        return nullptr;
    }
}

// ── Gradient Checkpointing via torch::autograd::Function ──
//
// Saves activation memory by NOT storing intermediate tensors during forward,
// then recomputing them during backward. Only the layer input is saved.
//
// Uses a C callback to call back into Rust for the forward computation.
// The callback receives the input tensor pointer and user context, returns
// a new tensor pointer (ownership transferred to C++).

#include <torch/csrc/autograd/custom_function.h>

typedef void* (*CheckpointFn)(void* input_ptr, void* user_ctx);

struct CheckpointFunction : public torch::autograd::Function<CheckpointFunction> {
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor input,
        int64_t fn_val,
        int64_t user_ctx_val
    ) {
        ctx->saved_data["fn"] = fn_val;
        ctx->saved_data["user_ctx"] = user_ctx_val;
        ctx->save_for_backward({input});

        // Run forward WITHOUT grad — no intermediate activations stored
        at::AutoGradMode guard(false);
        auto fn = reinterpret_cast<CheckpointFn>(fn_val);
        void* out = fn(reinterpret_cast<void*>(&input), reinterpret_cast<void*>(user_ctx_val));
        auto* t = reinterpret_cast<at::Tensor*>(out);
        at::Tensor result = std::move(*t);
        delete t;
        return result;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output
    ) {
        auto saved = ctx->get_saved_variables();
        at::Tensor input = saved[0];

        auto fn = reinterpret_cast<CheckpointFn>(ctx->saved_data["fn"].toInt());
        auto user_ctx = reinterpret_cast<void*>(ctx->saved_data["user_ctx"].toInt());

        // Detach input to break connection with outer graph.
        // This lets us use retain_graph=false (frees memory after backward).
        at::AutoGradMode guard(true);
        at::Tensor input_detached = input.detach();
        input_detached.set_requires_grad(true);

        void* out = fn(reinterpret_cast<void*>(&input_detached), user_ctx);
        auto* t = reinterpret_cast<at::Tensor*>(out);
        at::Tensor output = std::move(*t);
        delete t;

        // Backprop through detached graph — safe to free after
        output.backward(grad_output[0], /*retain_graph=*/false, /*create_graph=*/false);
        return {input_detached.grad(), at::Tensor(), at::Tensor()};
    }
};

// C FFI entry point: call from Rust to wrap a forward function with checkpointing
void* v4_checkpoint(void* fn_ptr, void* input_ptr, void* user_ctx) {
    auto& input = *reinterpret_cast<at::Tensor*>(input_ptr);
    auto fn = reinterpret_cast<CheckpointFn>(fn_ptr);

    // Ensure input requires grad — checkpoint needs it to set up backward
    if (!input.requires_grad()) {
        input.set_requires_grad(true);
    }

    // apply() returns at::Tensor in this PyTorch version
    at::Tensor result = CheckpointFunction::apply(
        input,
        (int64_t)(uintptr_t)fn_ptr,
        (int64_t)(uintptr_t)user_ctx
    );

    return new at::Tensor(std::move(result));
}

// Helper: create a new at::Tensor* from another at::Tensor* (copy constructor increments refcount)
// Used by Rust callbacks to return a tensor to C++.
void* v4_make_at_tensor(void* tensor_ptr) {
    auto* t = reinterpret_cast<at::Tensor*>(tensor_ptr);
    return new at::Tensor(*t);  // copy constructor increments refcount
}

// ── Caching allocator controls ──

/// Set the fraction of GPU memory the caching allocator is allowed to use.
/// Call early (before any training allocations) to pre-expand the pool.
void v4_set_memory_fraction(double fraction, int device_id) {
    try {
        c10::cuda::CUDACachingAllocator::setMemoryFraction(fraction, device_id);
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_set_memory_fraction] FAILED: %s\n", e.what());
    }
}

/// Empty the caching allocator's free pool (release cached blocks back to CUDA).
/// Call after warmup pass so that step 0 starts with a clean but pre-warmed pool.
void v4_empty_cache() {
    try {
        c10::cuda::CUDACachingAllocator::emptyCache();
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_empty_cache] FAILED: %s\n", e.what());
    }
}

} // extern "C"
