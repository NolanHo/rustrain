// qwen3_6_kernels.cpp — C++ FFI for Qwen3.6 native kernels
//
// Full C++ training: forward + loss + backward + Adam optimizer.
// LoRA A/B are created in C++ as at::Tensor (requires_grad=true).
// No tch-rs VarStore involved — gradients flow entirely within C++ autograd.

#include <ATen/ATen.h>
#if __has_include(<ATen/ops/_grouped_mm.h>)
#include <ATen/ops/_grouped_mm.h>
#define RUSTRAIN_HAS_ATEN_GROUPED_MM 1
#else
#define RUSTRAIN_HAS_ATEN_GROUPED_MM 0
#endif
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/variable.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <set>
#include <array>
#include <map>
#include <limits>
#include <sstream>
#include <chrono>
#include <sys/stat.h>
#include <unistd.h>

// NCCL for Expert Parallel all-reduce
#include <nccl.h>

struct TrainingContext;  // forward declaration (defined below)
struct LayerConfig;
static int64_t g_context_sequence = 0;

// Forward declarations for functions defined after TrainingContext
at::Tensor apply_multi_lora(TrainingContext* ctx, int64_t layer_idx, int64_t pair_idx, const at::Tensor& base_weight);
static at::Tensor tp_allreduce_lora_delta(
    TrainingContext* ctx, const at::Tensor& local_delta);
static at::Tensor tp_copy_lora_input(
    TrainingContext* ctx, const at::Tensor& input);

// ──────────────────────────────────────────────────────────────────────
// Forward declarations
// ──────────────────────────────────────────────────────────────────────

static at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double eps);

static bool env_enabled(const char* name, bool fallback = false) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') return fallback;
    return std::strcmp(value, "0") != 0 &&
        std::strcmp(value, "false") != 0;
}

static std::string nccl_sync_dir() {
    const char* run_id = std::getenv("RUSTRAIN_NCCL_RUN_ID");
    if (!run_id || run_id[0] == '\0') return "/tmp/rustrain-nccl";
    std::string sanitized;
    for (const unsigned char ch : std::string(run_id)) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') || ch == '-' || ch == '_') {
            sanitized.push_back(static_cast<char>(ch));
        } else {
            sanitized.push_back('_');
        }
    }
    if (sanitized.empty()) sanitized = "default";
    mkdir("/tmp/rustrain-nccl", 0777);
    std::string path = "/tmp/rustrain-nccl/" + sanitized;
    mkdir(path.c_str(), 0777);
    return path;
}

// ──────────────────────────────────────────────────────────────────────
// Hand-written CUDA fused kernels (compiled from fused_kernels.cu)
// ──────────────────────────────────────────────────────────────────────

extern "C" {
    void launch_fused_rmsnorm(const void* input, const void* weight, void* output,
        int M, int K, float eps, int scale_mode, void* stream);
    void launch_fused_swiglu(const void* gate, const void* up, void* output,
        int N, float limit, void* stream);
    void launch_fused_rmsnorm_matmul(const void* input, const void* weight, const void* matmul_w, void* output,
        int M, int N, int K, float eps, int scale_mode, void* stream);
    void launch_fused_adam_multi(
        void** d_param_ptrs, void** d_grad_ptrs,
        float** d_m_ptrs, float** d_v_ptrs,
        int* d_sizes, int n_params,
        float beta1, float beta2,
        float lr_scaled, float eps_scaled,
        float one_minus_beta1, float one_minus_beta2,
        void* stream);
    void launch_fused_adam_multi_out_of_place(
        void** d_src_param_ptrs, void** d_grad_ptrs,
        float** d_src_m_ptrs, float** d_src_v_ptrs,
        void** d_dst_param_ptrs,
        float** d_dst_m_ptrs, float** d_dst_v_ptrs,
        int* d_sizes, float* d_lr_scaled, float* d_eps_scaled,
        int n_params,
        float beta1, float beta2,
        float one_minus_beta1, float one_minus_beta2,
        void* stream);
}

/// Fused RMSNorm — single CUDA kernel (replaces 3 ATen ops)
/// scale_mode: 0 = weight only (V4), 1 = 1+weight (Qwen3.6)
static at::Tensor fused_rmsnorm_op(
    const at::Tensor& input, const at::Tensor& weight, double eps, int scale_mode
) {
    auto output = at::empty_like(input);
    int M = input.size(0);
    int K = input.size(1);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_fused_rmsnorm(
        input.data_ptr(), weight.data_ptr(), output.data_ptr(),
        M, K, (float)eps, scale_mode, stream
    );
    return output;
}

/// Fused SwiGLU with autograd: forward uses CUDA kernel, backward uses ATen ops
/// silu(g) * u, backward:
///   d/dg = sigmoid(g) * (1 + g * (1 - sigmoid(g))) * u
///   d/du = silu(g) = g * sigmoid(g)
struct NcclAllReduceFunction : public torch::autograd::Function<NcclAllReduceFunction> {
    static ncclDataType_t dtype_for(at::ScalarType type) {
        switch (type) {
            case at::kBFloat16: return ncclBfloat16;
            case at::kFloat: return ncclFloat;
            case at::kHalf: return ncclFloat16;
            default:
                TORCH_CHECK(false, "unsupported NCCL all-reduce dtype: ", type);
        }
    }

    // NCCL is normally issued on PyTorch's current stream. When an external
    // NCCL stream is supplied by the EP runtime, fence both sides so the
    // collective observes the producer and its result is visible to the
    // current compute stream. This keeps the operation asynchronous without
    // relying on a device-wide synchronize.
    static at::Tensor allreduce(
        const at::Tensor& input, ncclComm_t comm, cudaStream_t requested_stream,
        ncclRedOp_t reduction = ncclSum
    ) {
        TORCH_CHECK(input.is_cuda(), "NCCL all-reduce requires a CUDA tensor");
        const int dev = input.device().index();
        cudaSetDevice(dev);
        const auto current_stream = c10::cuda::getCurrentCUDAStream(dev).stream();
        const auto comm_stream = requested_stream ? requested_stream : current_stream;
        auto contiguous = input.contiguous();
        auto output = at::empty_like(contiguous);

        cudaEvent_t before = nullptr;
        cudaEvent_t after = nullptr;
        const bool cross_stream = comm_stream != current_stream;
        if (cross_stream) {
            TORCH_CHECK(cudaEventCreateWithFlags(&before, cudaEventDisableTiming) == cudaSuccess,
                "failed to create NCCL producer event");
            TORCH_CHECK(cudaEventCreateWithFlags(&after, cudaEventDisableTiming) == cudaSuccess,
                "failed to create NCCL consumer event");
            TORCH_CHECK(cudaEventRecord(before, current_stream) == cudaSuccess,
                "failed to record NCCL producer event");
            TORCH_CHECK(cudaStreamWaitEvent(comm_stream, before, 0) == cudaSuccess,
                "failed to wait for NCCL producer event");
        }

        auto err = ncclAllReduce(
            contiguous.data_ptr(), output.data_ptr(), contiguous.numel(),
            dtype_for(contiguous.scalar_type()), reduction, comm, comm_stream);
        TORCH_CHECK(err == ncclSuccess, "ncclAllReduce failed: ", ncclGetErrorString(err));

        if (cross_stream) {
            TORCH_CHECK(cudaEventRecord(after, comm_stream) == cudaSuccess,
                "failed to record NCCL consumer event");
            TORCH_CHECK(cudaStreamWaitEvent(current_stream, after, 0) == cudaSuccess,
                "failed to wait for NCCL consumer event");
            // Destruction is asynchronous-safe after the wait has been
            // enqueued on the current stream.
            cudaEventDestroy(before);
            cudaEventDestroy(after);
        }
        return output;
    }

    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
        at::Tensor input, int64_t comm_ptr, int64_t stream_ptr) {
        ctx->saved_data["comm"] = comm_ptr;
        ctx->saved_data["stream"] = stream_ptr;
        auto nccl_comm = reinterpret_cast<ncclComm_t>(comm_ptr);
        ctx->save_for_backward({input});
        return allreduce(input, nccl_comm,
            reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    static std::vector<at::Tensor> backward(torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output) {
        // Expert weights frozen — gradient flows through residual connection.
        // Save input in forward so autograd keeps it alive until backward.
        return {grad_output[0], at::Tensor(), at::Tensor()};
    }
};

// Megatron's copy_to_tensor_model_parallel_region equivalent: replicated
// input in forward, sum the column-parallel input-gradient contributions in
// backward before they flow into the preceding replicated sub-layer.
struct TpCopyToRegionFunction : public torch::autograd::Function<TpCopyToRegionFunction> {
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
        at::Tensor input, int64_t comm_ptr, int64_t stream_ptr) {
        ctx->saved_data["comm"] = comm_ptr;
        ctx->saved_data["stream"] = stream_ptr;
        return input;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output) {
        auto comm = reinterpret_cast<ncclComm_t>(ctx->saved_data["comm"].toInt());
        auto stream = reinterpret_cast<cudaStream_t>(ctx->saved_data["stream"].toInt());
        auto grad_input = NcclAllReduceFunction::allreduce(
            grad_output[0], comm, stream);
        return {grad_input, at::Tensor(), at::Tensor()};
    }
};

struct FusedSwiGLUFunction : public torch::autograd::Function<FusedSwiGLUFunction> {
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
        at::Tensor gate, at::Tensor up, double limit) {
        ctx->saved_data["limit"] = limit;
        ctx->save_for_backward({gate, up});
        auto output = at::empty_like(gate);
        int N = gate.numel();
        auto stream = c10::cuda::getCurrentCUDAStream().stream();
        launch_fused_swiglu(gate.data_ptr(), up.data_ptr(), output.data_ptr(),
            N, (float)limit, stream);
        return output;
    }
    static std::vector<at::Tensor> backward(torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_out) {
        auto saved = ctx->get_saved_variables();
        auto gate = saved[0];
        auto up = saved[1];
        auto grad = grad_out[0];
        // silu'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
        auto sig = at::sigmoid(gate);
        auto silu_grad = sig * (1.0 + gate * (1.0 - sig));
        auto grad_gate = grad * silu_grad * up;
        auto grad_up = grad * (gate * sig);  // silu(g)
        return {grad_gate, grad_up, at::Tensor()};
    }
};

static at::Tensor fused_swiglu_op(
    const at::Tensor& gate_out, const at::Tensor& up_out, double limit
) {
    // Keep the ATen path here because its backward is already part of the
    // autograd graph; the surrounding matmuls remain fused by the layer path.
    auto inter = at::silu(gate_out) * up_out;
    if (limit > 0.0) inter = inter.clamp(-limit, limit);
    return inter;
}

/// Fused RMSNorm + Matmul — single CUDA kernel (normed values stay in SRAM)
/// scale_mode: 0 = weight only (V4), 1 = 1+weight (Qwen3.6)
static at::Tensor fused_rmsnorm_matmul_op(
    const at::Tensor& input, const at::Tensor& norm_w, const at::Tensor& matmul_w,
    double eps, int scale_mode
) {
    int64_t M = input.size(0);
    int64_t K = input.size(1);
    int64_t N = matmul_w.size(0);
    auto output = at::zeros({M, N}, input.options());
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_fused_rmsnorm_matmul(
        input.data_ptr(), norm_w.data_ptr(), matmul_w.data_ptr(), output.data_ptr(),
        (int)M, (int)N, (int)K, (float)eps, scale_mode, stream
    );
    return output;
}

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────
// ──────────────────────────────────────────────────────────────────────

static at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double eps) {
    // Compute in input's dtype (BF16) to avoid 2x memory from FP32 conversion.
    // For large sequences (2M+), FP32 conversion creates [1, 16, 2M, 256] × 4B = 32GB.
    auto kind = input.scalar_type();
    auto variance = input.pow(2).mean(-1, true);
    auto inv_rms = (variance + eps).rsqrt();
    auto normed = input * inv_rms;
    return normed * (1.0 + weight.to(kind));
}

static at::Tensor lora_delta(const at::Tensor& base, const at::Tensor& a, const at::Tensor& b, double scaling) {
    auto kind = base.scalar_type();
    auto delta = b.to(kind).matmul(a.to(kind));
    return base + (delta * scaling).to(kind);
}

// KV cache for chunked full attention (per-layer, per-call)
// Stored on GPU if QWEN36_KV_OFFLOAD not set, CPU if set

// ──────────────────────────────────────────────────────────────────────
// Full attention
// ──────────────────────────────────────────────────────────────────────

// Split-half rotate: x = [a, b] → [-b, a]
static at::Tensor rotate_half(const at::Tensor& x) {
    auto last_dim = x.size(-1);
    auto half = last_dim / 2;
    auto x1 = x.narrow(-1, 0, half);
    auto x2 = x.narrow(-1, half, half);
    return at::cat({x2.neg(), x1}, -1);
}

static at::Tensor full_attention(
    const at::Tensor& hidden,
    const at::Tensor& q_proj, const at::Tensor& q_norm,
    const at::Tensor& k_proj, const at::Tensor& k_norm,
    const at::Tensor& v_proj, const at::Tensor& o_proj,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    double partial_rotary_factor, double rope_theta,
    double rms_eps, at::ScalarType compute_type,
    const at::Tensor& attention_mask  // [batch, seq] — 1 for real tokens, 0 for padding
) {
    auto device = hidden.device();
    int64_t batch = hidden.size(0), seq = hidden.size(1);
    int64_t qkv_dim = num_heads * head_dim;
    int64_t rotary_dim = (int64_t)(head_dim * partial_rotary_factor);

    // Full attention always uses SDPA (Flash Attention) — O(seq) memory.
    // QWEN36_SEQ_CHUNK only affects linear_attention (delta rule state passing).
    // For large sequences, chunk the q/k/v projections to avoid OOM on single matmul.
    const char* proj_chunk_env = getenv("QWEN36_PROJ_CHUNK");
    int64_t proj_chunk = proj_chunk_env ? atoll(proj_chunk_env) : 0;

    // Debug: confirm proj_chunk is read
    if (proj_chunk > 0 && seq > proj_chunk) {
    }

    at::Tensor q, gate, k, v;

    if (proj_chunk > 0 && seq > proj_chunk) {
        // Pre-allocate output tensors, fill in chunks to avoid storing all parts
        q = at::empty({batch, num_heads, seq, head_dim}, hidden.options());
        gate = at::empty({batch, num_heads, seq, head_dim}, hidden.options());
        k = at::empty({batch, num_kv_heads, seq, head_dim}, hidden.options());
        v = at::empty({batch, num_kv_heads, seq, head_dim}, hidden.options());

        for (int64_t s = 0; s < seq; s += proj_chunk) {
            int64_t e = std::min(s + proj_chunk, seq);
            int64_t clen = e - s;
            auto h_chunk = hidden.narrow(1, s, clen);

            // qkv projection
            auto qo = at::matmul(h_chunk, q_proj.t()).view({batch, clen, num_heads, head_dim * 2});
            auto qkc = qo.chunk(2, -1);
            // Write directly into pre-allocated tensors (transpose = view, then copy)
            q.narrow(2, s, clen).copy_(qkc[0].transpose(1, 2));
            gate.narrow(2, s, clen).copy_(qkc[1].transpose(1, 2));
            qo = at::Tensor();

            k.narrow(2, s, clen).copy_(
                at::matmul(h_chunk, k_proj.t()).view({batch, clen, num_kv_heads, head_dim}).transpose(1, 2));
            v.narrow(2, s, clen).copy_(
                at::matmul(h_chunk, v_proj.t()).view({batch, clen, num_kv_heads, head_dim}).transpose(1, 2));

        }
    } else {
        auto q_out = at::matmul(hidden, q_proj.t()).view({batch, seq, num_heads, head_dim * 2});
        auto qk_chunk = q_out.chunk(2, -1);
        q = qk_chunk[0].transpose(1, 2);
        gate = qk_chunk[1].transpose(1, 2);
        k = at::matmul(hidden, k_proj.t()).view({batch, seq, num_kv_heads, head_dim}).transpose(1, 2);
        v = at::matmul(hidden, v_proj.t()).view({batch, seq, num_kv_heads, head_dim}).transpose(1, 2);
    }

    q = rms_norm(q, q_norm, rms_eps);
    k = rms_norm(k, k_norm, rms_eps);

    // Debug: memory after rms_norm
    {
        size_t free2, total2;
        cudaMemGetInfo(&free2, &total2);
    }

    // Release gate before RoPE to save 16GB
    auto gate_saved = gate;
    gate = at::Tensor();

    // Debug: memory after gate release
    {
        size_t free2, total2;
        cudaMemGetInfo(&free2, &total2);
    }

    if (rotary_dim > 0) {
        auto pos = at::arange(seq, at::TensorOptions().dtype(at::kFloat).device(device)).unsqueeze(0);
        auto exponents = at::arange(0, rotary_dim, 2, at::TensorOptions().dtype(at::kFloat).device(device)) / (double)rotary_dim;
        auto inv_freq = (exponents * std::log(rope_theta)).exp().reciprocal();
        auto freqs = pos.unsqueeze(-1) * inv_freq.unsqueeze(0);
        auto emb = at::cat({freqs, freqs}, -1);
        auto cos = emb.cos().unsqueeze(1).to(q.scalar_type());
        auto sin = emb.sin().unsqueeze(1).to(q.scalar_type());

        // In-place RoPE to avoid creating additional 16GB tensors
        auto q_rot = q.narrow(-1, 0, rotary_dim);
        auto k_rot = k.narrow(-1, 0, rotary_dim);
        auto rotate_half_q = at::cat({-q_rot.narrow(-1, rotary_dim/2, rotary_dim/2), q_rot.narrow(-1, 0, rotary_dim/2)}, -1);
        auto rotate_half_k = at::cat({-k_rot.narrow(-1, rotary_dim/2, rotary_dim/2), k_rot.narrow(-1, 0, rotary_dim/2)}, -1);
        // q_rotated = q_rot * cos + rotate_half(q_rot) * sin — in-place
        q_rot.mul_(cos).add_(rotate_half_q * sin);
        k_rot.mul_(cos).add_(rotate_half_k * sin);
        // q and k are now modified in-place (RoPE applied to rotary_dim part)
        rotate_half_q = at::Tensor();
        rotate_half_k = at::Tensor();
        cos = at::Tensor();
        sin = at::Tensor();
    }

    // Restore gate after RoPE
    gate = gate_saved;

    // GQA: use enable_gqa=true — no K/V expansion needed (PT 2.5+)
    // K/V stay [batch, num_kv_heads, seq, head_dim], SDPA handles broadcasting.
    // This saves (n_rep-1) × num_kv_heads × seq × head_dim × BF16 per layer.
    // For N=864, num_kv_heads=2, head_dim=256: saves ~7.2 GB/layer peak.

    double scale = 1.0 / std::sqrt((double)head_dim);

    // Use SDPA with GQA (enable_gqa=true, PT 2.5+).
    // When attn_mask is provided, is_causal must be False (PT constraint).
    // We combine causal + padding into one additive mask.
    if (attention_mask.defined() && attention_mask.numel() > 0) {
        auto kpm = attention_mask.to(at::kBool);
        while (kpm.dim() > 2) kpm = kpm.squeeze(0);
        kpm = kpm.unsqueeze(1).unsqueeze(1);  // [B, 1, 1, S]
        // Build combined mask: causal + padding
        // causal: upper triangular = -inf
        auto causal = at::triu(at::ones({seq, seq}, at::TensorOptions().dtype(at::kBool).device(q.device())), 1);
        causal = causal.unsqueeze(0).unsqueeze(0);  // [1, 1, S, S]
        // padding: [B, 1, 1, S] → broadcast to [B, 1, S, S]
        auto pad_mask = kpm.logical_not();  // [B, 1, 1, S] — True = ignore
        auto combined = causal.logical_or(pad_mask);
        auto additive_mask = at::zeros({batch, 1, seq, seq}, at::TensorOptions().dtype(q.scalar_type()).device(q.device()));
        additive_mask = additive_mask.masked_fill(combined, -std::numeric_limits<float>::infinity());
        auto attn_out = at::scaled_dot_product_attention(
            q, k, v, additive_mask, 0.0, false, c10::nullopt, true  // is_causal=false, enable_gqa=true
        );
        // Qwen3.5/3.6 full attention gates the attention value before o_proj.
        // Applying the gate after o_proj is not equivalent when o_proj mixes features.
        auto gated_attn = attn_out * at::sigmoid(gate).to(attn_out.scalar_type());
        return gated_attn.transpose(1, 2).reshape({batch, seq, qkv_dim}).matmul(o_proj.t());
    } else {
        auto attn_out = at::scaled_dot_product_attention(
            q, k, v, c10::nullopt, 0.0, true, c10::nullopt, true  // is_causal=true, enable_gqa=true
        );
        auto gated_attn = attn_out * at::sigmoid(gate).to(attn_out.scalar_type());
        auto result = gated_attn.transpose(1, 2).reshape({batch, seq, qkv_dim}).matmul(o_proj.t());
        gate = at::Tensor();
        return result;
    }
}

// ──────────────────────────────────────────────────────────────────────
// Linear attention (Gated Delta Rule — matrix formulation)
// ──────────────────────────────────────────────────────────────────────

// Forward declaration for CUDA kernel (defined in delta_rule.cu)
extern "C" int cuda_gated_delta_rule(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    float* state, float* out, float* delta_buf,
    int BH, int seq_len, int key_dim, int val_dim, void* stream
);

extern "C" int cuda_gated_delta_rule_backward(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    const float* final_state, const float* delta_buf,
    const float* grad_out,
    float* grad_q, float* grad_k, float* grad_v,
    float* grad_g, float* grad_beta,
    int BH, int seq_len, int key_dim, int val_dim, void* stream
);

// Correctness reference for the gated delta rule. This deliberately uses
// ATen batched operations inside C++, so it remains outside the Rust hot path
// while providing a complete autograd oracle for the custom CUDA forward.
static at::Tensor gated_delta_rule_reference(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& g_exp, const at::Tensor& beta
) {
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
        "gated_delta_rule_reference expects [BH, S, D] tensors");
    const int64_t bh = q.size(0);
    const int64_t seq = q.size(1);
    const int64_t key_dim = q.size(2);
    const int64_t val_dim = v.size(2);
    TORCH_CHECK(k.size(0) == bh && k.size(1) == seq && k.size(2) == key_dim,
        "q/k shape mismatch");
    TORCH_CHECK(v.size(0) == bh && v.size(1) == seq,
        "v shape mismatch");
    TORCH_CHECK(g_exp.size(0) == bh && g_exp.size(1) == seq &&
                beta.size(0) == bh && beta.size(1) == seq,
        "g/beta shape mismatch");

    auto state = at::zeros({bh, key_dim, val_dim},
        q.options().dtype(at::kFloat));
    auto qf = q.to(at::kFloat);
    auto kf = k.to(at::kFloat);
    auto vf = v.to(at::kFloat);
    auto gf = g_exp.to(at::kFloat);
    auto bf = beta.to(at::kFloat);
    std::vector<at::Tensor> outputs;
    outputs.reserve(seq);
    for (int64_t t = 0; t < seq; ++t) {
        auto gt = gf.select(1, t).view({bh, 1, 1});
        auto kt = kf.select(1, t);
        auto vt = vf.select(1, t);
        auto qt = qf.select(1, t);
        auto bt = bf.select(1, t).view({bh, 1});
        state = state * gt;
        auto kv = at::bmm(kt.unsqueeze(1), state).squeeze(1);
        auto delta = (vt - kv) * bt;
        state = state + kt.unsqueeze(2) * delta.unsqueeze(1);
        outputs.push_back(at::bmm(qt.unsqueeze(1), state).squeeze(1));
    }
    return at::stack(outputs, 1);
}

// The forward and backward CUDA kernels are wrapped in one autograd Function.
// Set QWEN36_DELTA_REFERENCE_BWD=1 to run the ATen recurrence oracle for parity
// debugging; production training uses the current-stream fused backward.
struct GatedDeltaRuleFunction : public torch::autograd::Function<GatedDeltaRuleFunction> {
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor q, at::Tensor k, at::Tensor v,
        at::Tensor g_exp, at::Tensor beta
    ) {
        TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
            "gated delta CUDA path requires CUDA tensors");
        TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat &&
                    v.scalar_type() == at::kFloat && g_exp.scalar_type() == at::kFloat &&
                    beta.scalar_type() == at::kFloat,
            "gated delta CUDA path expects FP32 working tensors");
        const int64_t bh = q.size(0), seq = q.size(1);
        const int64_t key_dim = q.size(2), val_dim = v.size(2);
        auto state = at::zeros({bh, key_dim, val_dim}, q.options());
        auto out = at::empty({bh, seq, val_dim}, q.options());
        auto delta_buf = at::empty({bh, seq, val_dim}, q.options());
        auto stream = c10::cuda::getCurrentCUDAStream(q.device().index()).stream();
        int status = cuda_gated_delta_rule(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
            g_exp.data_ptr<float>(), beta.data_ptr<float>(),
            state.data_ptr<float>(), out.data_ptr<float>(), delta_buf.data_ptr<float>(),
            (int)bh, (int)seq, (int)key_dim, (int)val_dim,
            reinterpret_cast<void*>(stream));
        TORCH_CHECK(status == 0, "gated delta CUDA launch failed: ", status);
        ctx->save_for_backward({q, k, v, g_exp, beta, state, delta_buf});
        return out;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output
    ) {
        auto saved = ctx->get_saved_variables();
        auto q = saved[0];
        auto k = saved[1];
        auto v = saved[2];
        auto g = saved[3];
        auto beta = saved[4];
        if (env_enabled("QWEN36_DELTA_REFERENCE_BWD")) {
            auto q_ref = q.detach().set_requires_grad(true);
            auto k_ref = k.detach().set_requires_grad(true);
            auto v_ref = v.detach().set_requires_grad(true);
            auto g_ref = g.detach().set_requires_grad(true);
            auto beta_ref = beta.detach().set_requires_grad(true);
            at::AutoGradMode guard(true);
            auto reference = gated_delta_rule_reference(q_ref, k_ref, v_ref, g_ref, beta_ref);
            auto grads = torch::autograd::grad(
                {reference}, {q_ref, k_ref, v_ref, g_ref, beta_ref},
                {grad_output[0]}, /*retain_graph=*/false,
                /*create_graph=*/false, /*allow_unused=*/false);
            return {grads[0], grads[1], grads[2], grads[3], grads[4]};
        }

        const int64_t bh = q.size(0), seq = q.size(1);
        const int64_t key_dim = q.size(2), val_dim = v.size(2);
        auto grad_out = grad_output[0].contiguous();
        auto grad_q = at::empty_like(q);
        auto grad_k = at::empty_like(k);
        auto grad_v = at::empty_like(v);
        auto grad_g = at::empty_like(g);
        auto grad_beta = at::empty_like(beta);
        auto stream = c10::cuda::getCurrentCUDAStream(q.device().index()).stream();
        int status = cuda_gated_delta_rule_backward(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
            g.data_ptr<float>(), beta.data_ptr<float>(),
            saved[5].data_ptr<float>(), saved[6].data_ptr<float>(),
            grad_out.data_ptr<float>(), grad_q.data_ptr<float>(),
            grad_k.data_ptr<float>(), grad_v.data_ptr<float>(),
            grad_g.data_ptr<float>(), grad_beta.data_ptr<float>(),
            (int)bh, (int)seq, (int)key_dim, (int)val_dim,
            reinterpret_cast<void*>(stream));
        TORCH_CHECK(status == 0, "gated delta backward launch failed: ", status);
        return {grad_q, grad_k, grad_v, grad_g, grad_beta};
    }
};

static at::Tensor linear_attention(
    const at::Tensor& hidden,
    const at::Tensor& in_proj_qkv, const at::Tensor& in_proj_z,
    const at::Tensor& in_proj_a, const at::Tensor& in_proj_b,
    const at::Tensor& a_log, const at::Tensor& dt_bias,
    const at::Tensor& conv1d_weight, const at::Tensor& norm_w,
    const at::Tensor& out_proj,
    int64_t num_k_heads, int64_t key_dim,
    int64_t num_v_heads, int64_t val_dim,
    int64_t conv_kernel, double rms_eps,
    at::ScalarType compute_type
) {
    auto device = hidden.device();
    int64_t batch = hidden.size(0), seq = hidden.size(1);
    int64_t q_size = num_k_heads * key_dim;
    int64_t v_size = num_v_heads * val_dim;
    int64_t qkv_dim = q_size * 2 + v_size;

    // Check if sequence chunking is enabled
    const char* chunk_env = getenv("QWEN36_SEQ_CHUNK");
    int64_t seq_chunk = 0;
    if (chunk_env) seq_chunk = atoll(chunk_env);

    // Stateful chunking currently has no autograd state input/output contract.
    // Keep it for inference/eval only; training uses the autograd-wrapped full
    // sequence path below until chunk state gradients are implemented.
    if (seq_chunk > 0 && seq > seq_chunk && !at::GradMode::is_enabled()) {
        // Chunked linear attention — mathematically equivalent to full sequence
        // Process in chunks, passing the delta rule state between chunks.
        // This avoids creating [batch, seq, qkv_dim] intermediate tensors.
        int64_t BH = batch * num_v_heads;
        auto state = at::zeros({BH, key_dim, val_dim},
            at::TensorOptions().dtype(at::kFloat).device(device));

        std::vector<at::Tensor> chunk_outputs;
        int64_t offset = 0;
        while (offset < seq) {
            int64_t end = std::min(offset + seq_chunk, seq);
            int64_t chunk_len = end - offset;

            // Slice hidden for this chunk
            auto hidden_chunk = hidden.narrow(1, offset, chunk_len);

            // Process this chunk (same as non-chunked but on subset)
            auto qkv = at::matmul(hidden_chunk, in_proj_qkv.t());
            auto qkv_t = qkv.transpose(1, 2);

            // Conv1d: need overlap from previous chunk (conv_kernel - 1 tokens)
            int64_t pad = conv_kernel - 1;
            at::Tensor padding;
            if (offset == 0) {
                // First chunk: zero padding
                padding = at::zeros({batch, qkv_dim, pad}, qkv.options());
            } else {
                // Need previous chunk's last (pad) tokens for causal conv
                // Recompute qkv for the overlap region
                auto prev_qkv = at::matmul(
                    hidden.narrow(1, offset - pad, pad), in_proj_qkv.t()).transpose(1, 2);
                padding = prev_qkv;
            }
            auto padded = at::cat({padding, qkv_t}, 2);
            auto conv_out = at::conv1d(padded, conv1d_weight, {},
                at::IntArrayRef({1}), at::IntArrayRef({0}), at::IntArrayRef({1}), qkv_dim);
            conv_out = at::silu(conv_out.narrow(2, 0, chunk_len));
            auto qkv_conv = conv_out.transpose(1, 2);

            auto q = qkv_conv.narrow(-1, 0, q_size).view({batch, chunk_len, num_k_heads, key_dim});
            auto k = qkv_conv.narrow(-1, q_size, q_size).view({batch, chunk_len, num_k_heads, key_dim});
            auto v = qkv_conv.narrow(-1, q_size * 2, v_size).view({batch, chunk_len, num_v_heads, val_dim});

            auto a = at::matmul(hidden_chunk, in_proj_a.t());
            auto b = at::matmul(hidden_chunk, in_proj_b.t());
            auto z = at::matmul(hidden_chunk, in_proj_z.t()).view({batch, chunk_len, num_v_heads, val_dim});

            auto a_log_f = a_log.to(at::kFloat);
            auto dt_bias_f = dt_bias.to(at::kFloat);
            auto a_f = a.to(at::kFloat);
            auto g = a_log_f.unsqueeze(0).unsqueeze(0).exp().neg() *
                     at::softplus(a_f + dt_bias_f.unsqueeze(0).unsqueeze(0));
            auto beta = at::sigmoid(b);

            int64_t n_rep = num_v_heads / num_k_heads;
            q = q.repeat_interleave(n_rep, 2);
            k = k.repeat_interleave(n_rep, 2);

            auto q_f = q.to(at::kFloat);
            auto k_f = k.to(at::kFloat);
            q = q_f * (q_f.pow(2).sum(-1, true) + 1e-6).rsqrt();
            k = k_f * (k_f.pow(2).sum(-1, true) + 1e-6).rsqrt();
            q = q * (1.0 / std::sqrt((double)key_dim));

            auto q_t = q.transpose(1, 2).contiguous();
            auto k_t = k.transpose(1, 2).contiguous();
            auto v_t = v.to(at::kFloat).transpose(1, 2).contiguous();
            auto g_t = g.transpose(1, 2).contiguous();
            auto beta_t = beta.to(at::kFloat).transpose(1, 2).contiguous();

            auto g_exp = g_t.exp();
            auto q_contig = q_t.reshape({BH, chunk_len, key_dim}).contiguous().to(at::kFloat);
            auto k_contig = k_t.reshape({BH, chunk_len, key_dim}).contiguous().to(at::kFloat);
            auto v_contig = v_t.reshape({BH, chunk_len, val_dim}).contiguous().to(at::kFloat);
            auto g_contig = g_exp.reshape({BH, chunk_len}).contiguous().to(at::kFloat);
            auto beta_contig = beta_t.reshape({BH, chunk_len}).contiguous().to(at::kFloat);
            auto state_contig = state.contiguous();
            auto outs = at::empty({BH, chunk_len, val_dim}, q_t.options());
            auto delta_buf = at::empty({BH, chunk_len, val_dim}, q_t.options());

            // CUDA kernel — state is passed in and updated in-place
            auto stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();
            int status = cuda_gated_delta_rule(
                q_contig.data_ptr<float>(),
                k_contig.data_ptr<float>(),
                v_contig.data_ptr<float>(),
                g_contig.data_ptr<float>(),
                beta_contig.data_ptr<float>(),
                state_contig.data_ptr<float>(),
                outs.data_ptr<float>(),
                delta_buf.data_ptr<float>(),
                (int)BH, (int)chunk_len, (int)key_dim, (int)val_dim,
                reinterpret_cast<void*>(stream)
            );
            TORCH_CHECK(status == 0, "gated delta CUDA launch failed: ", status);
            state = state_contig;  // updated state for next chunk

            auto core_out = outs.reshape({batch, num_v_heads, chunk_len, val_dim})
                                 .transpose(1, 2).to(compute_type);

            auto core_flat = core_out.reshape({-1, val_dim});
            auto z_flat = z.reshape({-1, val_dim});
            auto variance = core_flat.to(at::kFloat).pow(2).mean(-1, true);
            auto normed = (core_flat.to(at::kFloat) * (variance + rms_eps).rsqrt() *
                           norm_w.to(at::kFloat)).to(core_flat.scalar_type());
            auto gated = (normed * at::silu(z_flat.to(at::kFloat)).to(normed.scalar_type()))
                         .view({batch, chunk_len, num_v_heads * val_dim});
            auto result = at::matmul(gated, out_proj.t());
            chunk_outputs.push_back(result);

            offset = end;
        }

        return at::cat(chunk_outputs, 1);  // [batch, seq, hidden]
    }

    // Non-chunked path (original code)
    auto qkv = at::matmul(hidden, in_proj_qkv.t());

    // DIAG: dump after QKV projection
    if (getenv("QWEN36_DUMP_LAYERS")) {
        auto qkv_f = qkv.to(at::kFloat);
        fprintf(stderr, "[diag-na] qkv_proj: mean=%.6f std=%.6f [0,0,:5]=%.6f,%.6f,%.6f,%.6f,%.6f\n",
                qkv_f.mean().item<float>(), qkv_f.std().item<float>(),
                qkv_f[0][0][0].item<float>(), qkv_f[0][0][1].item<float>(),
                qkv_f[0][0][2].item<float>(), qkv_f[0][0][3].item<float>(),
                qkv_f[0][0][4].item<float>());
    }

    auto qkv_t = qkv.transpose(1, 2);
    int64_t pad = conv_kernel - 1;
    auto padding = at::zeros({batch, qkv_dim, pad}, qkv.options());
    auto padded = at::cat({padding, qkv_t}, 2);
    auto conv_out = at::conv1d(padded, conv1d_weight, /*bias=*/{},
        at::IntArrayRef({1}), at::IntArrayRef({0}), at::IntArrayRef({1}), qkv_dim);
    conv_out = at::silu(conv_out.narrow(2, 0, seq));
    auto qkv_conv = conv_out.transpose(1, 2);

    auto q = qkv_conv.narrow(-1, 0, q_size).view({batch, seq, num_k_heads, key_dim});
    auto k = qkv_conv.narrow(-1, q_size, q_size).view({batch, seq, num_k_heads, key_dim});
    auto v = qkv_conv.narrow(-1, q_size * 2, v_size).view({batch, seq, num_v_heads, val_dim});

    auto a = at::matmul(hidden, in_proj_a.t());
    auto b = at::matmul(hidden, in_proj_b.t());
    auto z = at::matmul(hidden, in_proj_z.t()).view({batch, seq, num_v_heads, val_dim});

    // g = -exp(A_log) * softplus(a + dt_bias)  — HF convention
    auto a_log_f = a_log.to(at::kFloat);
    auto dt_bias_f = dt_bias.to(at::kFloat);
    auto a_f = a.to(at::kFloat);
    auto g = a_log_f.unsqueeze(0).unsqueeze(0).exp().neg() * at::softplus(a_f + dt_bias_f.unsqueeze(0).unsqueeze(0));
    auto beta = at::sigmoid(b);

    int64_t n_rep = num_v_heads / num_k_heads;
    q = q.repeat_interleave(n_rep, 2);
    k = k.repeat_interleave(n_rep, 2);

    // L2 normalize Q, K (HF: use_qk_l2norm_in_kernel=True, eps=1e-6)
    auto q_f = q.to(at::kFloat);
    auto k_f = k.to(at::kFloat);
    q = q_f * (q_f.pow(2).sum(-1, true) + 1e-6).rsqrt();
    k = k_f * (k_f.pow(2).sum(-1, true) + 1e-6).rsqrt();

    // Scale Q by 1/sqrt(key_dim) — matching HF: scale = 1 / (key_dim ** 0.5)
    double scale = 1.0 / std::sqrt((double)key_dim);
    q = q * scale;

    // Transpose to [B, H, S, D] for recurrent loop
    auto q_t = q.transpose(1, 2).contiguous();
    auto k_t = k.transpose(1, 2).contiguous();
    auto v_t = v.to(at::kFloat).transpose(1, 2).contiguous();
    auto g_t = g.transpose(1, 2).contiguous();  // [B, H, S]
    auto beta_t = beta.to(at::kFloat).transpose(1, 2).contiguous();

    // CUDA kernel for gated delta rule — eliminates per-token kernel launch overhead
    // All computation runs in a single kernel launch using shared memory for state
    auto g_exp = g_t.exp();  // [B, H, S]
    int64_t BH = batch * num_v_heads;
    // Prepare contiguous FP32 tensors for the CUDA forward/autograd wrapper.
    auto q_contig = q_t.reshape({BH, seq, key_dim}).contiguous().to(at::kFloat);
    auto k_contig = k_t.reshape({BH, seq, key_dim}).contiguous().to(at::kFloat);
    auto v_contig = v_t.reshape({BH, seq, val_dim}).contiguous().to(at::kFloat);
    auto g_contig = g_exp.reshape({BH, seq}).contiguous().to(at::kFloat);
    auto beta_contig = beta_t.reshape({BH, seq}).contiguous().to(at::kFloat);
    auto outs = GatedDeltaRuleFunction::apply(
        q_contig, k_contig, v_contig, g_contig, beta_contig);

    // Reshape: [B*H, S, D_v] → [B, H, S, D_v] → [B, S, H, D_v]
    auto core_out = outs.reshape({batch, num_v_heads, seq, val_dim})
                         .transpose(1, 2).to(compute_type);

    // DIAG: dump after delta rule
    if (getenv("QWEN36_DUMP_LAYERS")) {
        auto co_f = core_out.to(at::kFloat);
        fprintf(stderr, "[diag-na] after_delta: mean=%.6f std=%.6f [0,0,0,:3]=%.6f,%.6f,%.6f\n",
                co_f.mean().item<float>(), co_f.std().item<float>(),
                co_f[0][0][0][0].item<float>(), co_f[0][0][0][1].item<float>(), co_f[0][0][0][2].item<float>());
    }

    auto core_flat = core_out.reshape({-1, val_dim});
    auto z_flat = z.reshape({-1, val_dim});
    auto variance = core_flat.to(at::kFloat).pow(2).mean(-1, true);
    auto normed = (core_flat.to(at::kFloat) * (variance + rms_eps).rsqrt() * norm_w.to(at::kFloat)).to(core_flat.scalar_type());
    auto gated = (normed * at::silu(z_flat.to(at::kFloat)).to(normed.scalar_type())).view({batch, seq, num_v_heads * val_dim});
    auto result = at::matmul(gated, out_proj.t());

    // DIAG: dump after norm+gate+out_proj
    if (getenv("QWEN36_DUMP_LAYERS")) {
        auto r_f = result.to(at::kFloat);
        fprintf(stderr, "[diag-na] after_out_proj: mean=%.6f std=%.6f [0,0,:3]=%.6f,%.6f,%.6f\n",
                r_f.mean().item<float>(), r_f.std().item<float>(),
                r_f[0][0][0].item<float>(), r_f[0][0][1].item<float>(), r_f[0][0][2].item<float>());
    }

    return result;
}

// Forward declarations for functions defined below
static at::Tensor dense_mlp_forward(const at::Tensor& hidden,
    const at::Tensor& gate_proj, const at::Tensor& up_proj, const at::Tensor& down_proj,
    at::ScalarType compute_type);
static at::Tensor full_attention_batched(TrainingContext* ctx, const at::Tensor& hidden,
    int64_t layer_idx, const at::Tensor& q_proj, const at::Tensor& q_norm,
    const at::Tensor& k_proj, const at::Tensor& k_norm,
    const at::Tensor& v_proj, const at::Tensor& o_proj,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    double partial_rotary_factor, double rope_theta,
    double rms_eps, at::ScalarType kind, const at::Tensor& attention_mask);
static at::Tensor linear_attention_batched(TrainingContext* ctx, const at::Tensor& hidden,
    int64_t layer_idx, const at::Tensor& in_proj_qkv, const at::Tensor& in_proj_z,
    const at::Tensor& in_proj_a, const at::Tensor& in_proj_b,
    const at::Tensor& a_log, const at::Tensor& dt_bias,
    const at::Tensor& conv1d_w, const at::Tensor& norm_w, const at::Tensor& out_proj,
    int64_t num_k_heads, int64_t key_dim, int64_t num_v_heads, int64_t val_dim,
    int64_t conv_kernel, double rms_eps, at::ScalarType compute_type);

// ──────────────────────────────────────────────────────────────────────
// MoE
// ──────────────────────────────────────────────────────────────────────

static ncclDataType_t qwen36_nccl_dtype(at::ScalarType type) {
    switch (type) {
        case at::kFloat: return ncclFloat32;
        case at::kHalf: return ncclFloat16;
        case at::kBFloat16: return ncclBfloat16;
        case at::kInt: return ncclInt;
        case at::kLong: return ncclInt64;
        default:
            TORCH_CHECK(false, "unsupported NCCL dtype: ", type);
    }
}

struct Qwen36A2ACountPlan {
    std::vector<int32_t> send;
    std::vector<int32_t> receive;
    at::Tensor send_tensor;
    at::Tensor receive_tensor;
};

static Qwen36A2ACountPlan qwen36_a2a_counts(
    const at::Tensor& local_metadata, ncclComm_t comm, cudaStream_t stream
) {
    int world = 0;
    auto err = ncclCommCount(comm, &world);
    TORCH_CHECK(err == ncclSuccess, "ncclCommCount failed: ",
        ncclGetErrorString(err));
    TORCH_CHECK(local_metadata.numel() == world + 1,
        "EP A2A metadata must contain one count per peer and a validity flag");
    auto all_counts = at::empty(
        {world, world + 1}, local_metadata.options());
    err = ncclAllGather(
        local_metadata.data_ptr(), all_counts.data_ptr(), world + 1,
        ncclInt, comm, stream);
    TORCH_CHECK(err == ncclSuccess, "EP A2A count all-gather failed: ",
        ncclGetErrorString(err));
    auto host = all_counts.to(at::TensorOptions().device(at::kCPU));
    int rank = 0;
    err = ncclCommUserRank(comm, &rank);
    TORCH_CHECK(err == ncclSuccess, "ncclCommUserRank failed: ",
        ncclGetErrorString(err));

    Qwen36A2ACountPlan plan;
    plan.send_tensor = host.select(0, rank).narrow(0, 0, world).contiguous();
    plan.receive_tensor = host.narrow(1, 0, world)
        .select(1, rank).contiguous();
    plan.send.resize(world);
    plan.receive.resize(world);
    std::memcpy(plan.send.data(), plan.send_tensor.data_ptr<int32_t>(),
        sizeof(int32_t) * world);
    std::memcpy(plan.receive.data(), plan.receive_tensor.data_ptr<int32_t>(),
        sizeof(int32_t) * world);
    for (int peer = 0; peer < world; ++peer) {
        TORCH_CHECK(plan.send[peer] >= 0 && plan.receive[peer] >= 0,
            "negative EP A2A token count");
        TORCH_CHECK(host.data_ptr<int32_t>()[peer * (world + 1) + world] == 0,
            "EP A2A expert index is outside the communicator range");
    }
    return plan;
}

// Variable-split token dispatch. Metadata is deliberately non-differentiable;
// only the hidden activation participates in the custom backward exchange.
struct Qwen36A2ADispatchFunction : public torch::autograd::Function<Qwen36A2ADispatchFunction> {
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor input, at::Tensor expert_indices, at::Tensor token_indices,
        int64_t expert_count, int64_t comm_ptr
    ) {
        auto comm = reinterpret_cast<ncclComm_t>(comm_ptr);
        TORCH_CHECK(comm, "EP A2A requires an NCCL communicator");
        int world = 1, rank = 0;
        TORCH_CHECK(ncclCommCount(comm, &world) == ncclSuccess,
            "ncclCommCount failed");
        TORCH_CHECK(ncclCommUserRank(comm, &rank) == ncclSuccess,
            "ncclCommUserRank failed");
        TORCH_CHECK(expert_count > 0 && expert_indices.scalar_type() == at::kLong,
            "invalid EP A2A expert metadata");
        TORCH_CHECK(expert_indices.numel() == token_indices.numel(),
            "EP A2A expert and source-token metadata must have equal length");
        auto stream = c10::cuda::getCurrentCUDAStream(input.device().index()).stream();
        const int64_t hidden = input.size(1);
        auto gpu_count_opts = at::TensorOptions()
            .device(input.device()).dtype(at::kInt);
        at::Tensor send_index;
        at::Tensor local_metadata;
        // Peer-wise nonzero is cheaper at EP2; one GPU sort wins as the EP
        // fan-out grows for sufficiently large token batches. The override
        // keeps both paths available for A/B and production tuning.
        const bool use_gpu_metadata = env_enabled(
            "QWEN36_EP_A2A_GPU_METADATA",
            world >= 4 && expert_indices.numel() >= 512);
        if (use_gpu_metadata) {
            auto destinations = at::floor_divide(
                expert_indices, expert_count);
            auto valid_destinations = destinations.clamp(0, world - 1);
            auto invalid = destinations.ne(valid_destinations)
                .any().to(at::kInt).reshape({1});
            auto [sorted_destinations, order] = valid_destinations.sort(0);
            send_index = std::move(order);
            auto local_counts = at::bincount(
                sorted_destinations, c10::nullopt, world).to(at::kInt);
            local_metadata = at::cat({local_counts, invalid}, 0);
        } else {
            std::vector<at::Tensor> indices(world);
            std::vector<int32_t> host_metadata(world + 1, 0);
            for (int dst = 0; dst < world; ++dst) {
                auto mask = (expert_indices >= dst * expert_count) &
                    (expert_indices < (dst + 1) * expert_count);
                indices[dst] = at::nonzero(mask).reshape({-1});
                host_metadata[dst] = static_cast<int32_t>(
                    indices[dst].numel());
            }
            send_index = at::cat(indices, 0);
            host_metadata[world] = send_index.numel() != expert_indices.numel();
            auto host_counts = at::from_blob(
                host_metadata.data(), {world + 1},
                at::TensorOptions().device(at::kCPU).dtype(at::kInt));
            local_metadata = host_counts.to(gpu_count_opts);
        }
        auto count_plan = qwen36_a2a_counts(local_metadata, comm, stream);
        const auto& send_counts = count_plan.send;
        const auto& recv_counts = count_plan.receive;
        std::vector<int64_t> send_offsets(world + 1, 0), recv_offsets(world + 1, 0);
        for (int i = 0; i < world; ++i) {
            send_offsets[i + 1] = send_offsets[i] + send_counts[i];
            recv_offsets[i + 1] = recv_offsets[i] + recv_counts[i];
        }
        auto send_token = token_indices.index_select(0, send_index).contiguous();
        auto send_hidden = input.index_select(0, send_token).contiguous();
        auto send_expert = expert_indices.index_select(0, send_index).contiguous();
        auto recv_hidden = at::empty({recv_offsets.back(), hidden}, input.options());
        auto recv_token = at::empty({recv_offsets.back()}, token_indices.options());
        auto recv_expert = at::empty({recv_offsets.back()}, expert_indices.options());
        auto send_ptr = [&](at::Tensor& t, int64_t row, int64_t width) -> const void* {
            return static_cast<const char*>(t.data_ptr()) +
                row * width * t.element_size();
        };
        auto recv_ptr = [&](at::Tensor& t, int64_t row, int64_t width) -> void* {
            return static_cast<char*>(t.data_ptr()) +
                row * width * t.element_size();
        };
        TORCH_CHECK(ncclGroupStart() == ncclSuccess, "ncclGroupStart failed");
        for (int peer = 0; peer < world; ++peer) {
            const int64_t rows = send_counts[peer];
            if (rows) {
                auto err = ncclSend(send_ptr(send_hidden, send_offsets[peer], hidden),
                    rows * hidden, qwen36_nccl_dtype(input.scalar_type()), peer, comm, stream);
                TORCH_CHECK(err == ncclSuccess, "A2A hidden send failed: ", ncclGetErrorString(err));
                err = ncclSend(send_ptr(send_token, send_offsets[peer], 1), rows,
                    ncclInt64, peer, comm, stream);
                TORCH_CHECK(err == ncclSuccess, "A2A token send failed: ", ncclGetErrorString(err));
                err = ncclSend(send_ptr(send_expert, send_offsets[peer], 1), rows,
                    ncclInt64, peer, comm, stream);
                TORCH_CHECK(err == ncclSuccess, "A2A expert send failed: ", ncclGetErrorString(err));
            }
        }
        for (int peer = 0; peer < world; ++peer) {
            const int64_t rows = recv_counts[peer];
            if (rows) {
                auto err = ncclRecv(recv_ptr(recv_hidden, recv_offsets[peer], hidden),
                    rows * hidden, qwen36_nccl_dtype(input.scalar_type()), peer, comm, stream);
                TORCH_CHECK(err == ncclSuccess, "A2A hidden recv failed: ", ncclGetErrorString(err));
                err = ncclRecv(recv_ptr(recv_token, recv_offsets[peer], 1), rows,
                    ncclInt64, peer, comm, stream);
                TORCH_CHECK(err == ncclSuccess, "A2A token recv failed: ", ncclGetErrorString(err));
                err = ncclRecv(recv_ptr(recv_expert, recv_offsets[peer], 1), rows,
                    ncclInt64, peer, comm, stream);
                TORCH_CHECK(err == ncclSuccess, "A2A expert recv failed: ", ncclGetErrorString(err));
            }
        }
        TORCH_CHECK(ncclGroupEnd() == ncclSuccess, "A2A dispatch group failed");
        auto recv_local = recv_expert - rank * expert_count;
        auto send_counts_tensor = std::move(count_plan.send_tensor);
        auto recv_counts_tensor = std::move(count_plan.receive_tensor);
        ctx->save_for_backward({input, send_token, send_counts_tensor, recv_counts_tensor});
        ctx->saved_data["comm"] = comm_ptr;
        ctx->saved_data["expert_count"] = expert_count;
        return {recv_hidden, recv_token, recv_local, send_index, send_counts_tensor, recv_counts_tensor};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx, std::vector<at::Tensor> grad_output
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto send_token = saved[1];
        auto send_counts = saved[2].to(at::TensorOptions().device(at::kCPU));
        auto recv_counts = saved[3].to(at::TensorOptions().device(at::kCPU));
        const int world = send_counts.numel();
        auto comm = reinterpret_cast<ncclComm_t>(ctx->saved_data["comm"].toInt());
        auto stream = c10::cuda::getCurrentCUDAStream(input.device().index()).stream();
        std::vector<int32_t> sc(world), rc(world);
        std::memcpy(sc.data(), send_counts.data_ptr<int32_t>(), sizeof(int32_t) * world);
        std::memcpy(rc.data(), recv_counts.data_ptr<int32_t>(), sizeof(int32_t) * world);
        std::vector<int64_t> so(world + 1, 0), ro(world + 1, 0);
        for (int i = 0; i < world; ++i) { so[i + 1] = so[i] + sc[i]; ro[i + 1] = ro[i] + rc[i]; }
        auto grad_input = at::zeros_like(input);
        auto returned = at::empty({so.back(), input.size(1)}, input.options());
        const size_t elem_bytes = input.element_size();
        TORCH_CHECK(ncclGroupStart() == ncclSuccess, "ncclGroupStart failed");
        for (int peer = 0; peer < world; ++peer) if (sc[peer]) {
            auto err = ncclRecv(static_cast<char*>(returned.data_ptr()) + so[peer] * input.size(1) * elem_bytes,
                sc[peer] * input.size(1), qwen36_nccl_dtype(input.scalar_type()), peer, comm, stream);
            TORCH_CHECK(err == ncclSuccess, "A2A backward recv failed: ", ncclGetErrorString(err));
        }
        for (int peer = 0; peer < world; ++peer) if (rc[peer]) {
            auto err = ncclSend(static_cast<const char*>(grad_output[0].data_ptr()) + ro[peer] * input.size(1) * elem_bytes,
                rc[peer] * input.size(1), qwen36_nccl_dtype(input.scalar_type()), peer, comm, stream);
            TORCH_CHECK(err == ncclSuccess, "A2A backward send failed: ", ncclGetErrorString(err));
        }
        TORCH_CHECK(ncclGroupEnd() == ncclSuccess, "A2A dispatch backward group failed");
        grad_input.index_add_(0, send_token, returned);
        return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

struct Qwen36A2ACombineFunction : public torch::autograd::Function<Qwen36A2ACombineFunction> {
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
        at::Tensor local_output, at::Tensor send_counts,
        at::Tensor recv_counts, int64_t comm_ptr) {
        auto comm = reinterpret_cast<ncclComm_t>(comm_ptr);
        const int world = send_counts.numel();
        auto sc_cpu = send_counts.to(at::TensorOptions().device(at::kCPU));
        auto rc_cpu = recv_counts.to(at::TensorOptions().device(at::kCPU));
        std::vector<int32_t> sc(world), rc(world);
        std::memcpy(sc.data(), sc_cpu.data_ptr<int32_t>(), sizeof(int32_t) * world);
        std::memcpy(rc.data(), rc_cpu.data_ptr<int32_t>(), sizeof(int32_t) * world);
        std::vector<int64_t> so(world + 1, 0), ro(world + 1, 0);
        for (int i = 0; i < world; ++i) { so[i + 1] = so[i] + sc[i]; ro[i + 1] = ro[i] + rc[i]; }
        auto stream = c10::cuda::getCurrentCUDAStream(local_output.device().index()).stream();
        auto returned = at::empty({so.back(), local_output.size(1)}, local_output.options());
        const size_t elem_bytes = local_output.element_size();
        TORCH_CHECK(ncclGroupStart() == ncclSuccess, "ncclGroupStart failed");
        for (int peer = 0; peer < world; ++peer) if (rc[peer]) {
            auto err = ncclSend(static_cast<const char*>(local_output.data_ptr()) + ro[peer] * local_output.size(1) * elem_bytes,
                rc[peer] * local_output.size(1), qwen36_nccl_dtype(local_output.scalar_type()), peer, comm, stream);
            TORCH_CHECK(err == ncclSuccess, "A2A combine send failed: ", ncclGetErrorString(err));
        }
        for (int peer = 0; peer < world; ++peer) if (sc[peer]) {
            auto err = ncclRecv(static_cast<char*>(returned.data_ptr()) + so[peer] * local_output.size(1) * elem_bytes,
                sc[peer] * local_output.size(1), qwen36_nccl_dtype(local_output.scalar_type()), peer, comm, stream);
            TORCH_CHECK(err == ncclSuccess, "A2A combine recv failed: ", ncclGetErrorString(err));
        }
        TORCH_CHECK(ncclGroupEnd() == ncclSuccess, "A2A combine group failed");
        ctx->save_for_backward({send_counts, recv_counts});
        ctx->saved_data["comm"] = comm_ptr;
        return returned;
    }

    static std::vector<at::Tensor> backward(torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output) {
        auto saved = ctx->get_saved_variables();
        auto sc_cpu = saved[0].to(at::TensorOptions().device(at::kCPU));
        auto rc_cpu = saved[1].to(at::TensorOptions().device(at::kCPU));
        const int world = sc_cpu.numel();
        std::vector<int32_t> sc(world), rc(world);
        std::memcpy(sc.data(), sc_cpu.data_ptr<int32_t>(), sizeof(int32_t) * world);
        std::memcpy(rc.data(), rc_cpu.data_ptr<int32_t>(), sizeof(int32_t) * world);
        std::vector<int64_t> so(world + 1, 0), ro(world + 1, 0);
        for (int i = 0; i < world; ++i) { so[i + 1] = so[i] + sc[i]; ro[i + 1] = ro[i] + rc[i]; }
        auto comm = reinterpret_cast<ncclComm_t>(ctx->saved_data["comm"].toInt());
        auto stream = c10::cuda::getCurrentCUDAStream(grad_output[0].device().index()).stream();
        auto packed = grad_output[0].contiguous();
        auto grad_local = at::empty({ro.back(), grad_output[0].size(1)}, grad_output[0].options());
        const size_t elem_bytes = grad_output[0].element_size();
        TORCH_CHECK(ncclGroupStart() == ncclSuccess, "ncclGroupStart failed");
        for (int peer = 0; peer < world; ++peer) if (sc[peer]) {
            auto err = ncclSend(static_cast<const char*>(packed.data_ptr()) + so[peer] * grad_output[0].size(1) * elem_bytes,
                sc[peer] * grad_output[0].size(1), qwen36_nccl_dtype(grad_output[0].scalar_type()), peer, comm, stream);
            TORCH_CHECK(err == ncclSuccess, "A2A combine backward send failed: ", ncclGetErrorString(err));
        }
        for (int peer = 0; peer < world; ++peer) if (rc[peer]) {
            auto err = ncclRecv(static_cast<char*>(grad_local.data_ptr()) + ro[peer] * grad_output[0].size(1) * elem_bytes,
                rc[peer] * grad_output[0].size(1), qwen36_nccl_dtype(grad_output[0].scalar_type()), peer, comm, stream);
            TORCH_CHECK(err == ncclSuccess, "A2A combine backward recv failed: ", ncclGetErrorString(err));
        }
        TORCH_CHECK(ncclGroupEnd() == ncclSuccess, "A2A combine backward group failed");
        return {grad_local, at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

struct RoutedExpertLora {
    const at::Tensor* gate_up_a = nullptr;  // [local_experts, rank, hidden]
    const at::Tensor* gate_up_b = nullptr;  // [local_experts, 2*intermediate, rank]
    const at::Tensor* down_a = nullptr;     // [local_experts, rank, intermediate]
    const at::Tensor* down_b = nullptr;     // [local_experts, hidden, rank]
    double scaling = 0.0;
};

enum class LoraTpLayout : uint8_t {
    LatentRank,
    ColumnParallel,
    RowParallel,
};

// Per-sample adapter projection used by the batched multi-LoRA path.  This is
// intentionally separate from RoutedExpertLora: routed experts carry one
// A/B pair per local expert, while dense/shared projections carry one pair per
// adapter sample.
struct LoraBatchEntry {
    at::Tensor a_stack;    // [N, rank, in]
    at::Tensor b_stack;    // [N, out, rank]
    at::Tensor scaling;    // [N, 1, 1]
    LoraTpLayout layout = LoraTpLayout::LatentRank;
};

static const LoraBatchEntry* lora_batch_entry(
    TrainingContext* ctx, int64_t layer_idx, int64_t pair_idx);
static at::Tensor dense_mlp_forward_batched(
    TrainingContext* ctx, int64_t layer_idx, const at::Tensor& hidden,
    const at::Tensor& gate_proj, const at::Tensor& up_proj,
    const at::Tensor& down_proj, at::ScalarType compute_type);
static bool base_tp_mlp_enabled(const TrainingContext* ctx);
static int64_t base_tp_mlp_world_size(const TrainingContext* ctx);
static at::Tensor tp_allreduce_base_mlp(
    TrainingContext* ctx, const at::Tensor& local_output);
static at::Tensor tp_copy_base_mlp_input(
    TrainingContext* ctx, const at::Tensor& input);

static at::Tensor lora_activation_delta(
    TrainingContext* ctx, const at::Tensor& x,
    const at::Tensor& A, const at::Tensor& B,
    const at::Tensor& scaling, LoraTpLayout layout);

static at::Tensor add_batched_lora(
    TrainingContext* ctx, const at::Tensor& base, const at::Tensor& input,
    const LoraBatchEntry* entry
) {
    if (!entry) return base;
    return base + lora_activation_delta(
        ctx, input, entry->a_stack, entry->b_stack, entry->scaling,
        entry->layout);
}

// Per-token routed-expert LoRA. Dynamic adapters add a leading sample axis to
// the expert-local tensors: A [batch, experts, rank, in],
// B [batch, experts, out, rank]. Flattening (sample, expert) lets one pair of
// index_select + bmm operations select the correct adapter and expert without
// materializing a full-rank delta weight.
static at::Tensor dynamic_expert_lora_delta(
    TrainingContext* ctx,
    const at::Tensor& input,
    const at::Tensor& token_indices,
    const at::Tensor& local_expert_indices,
    int64_t batch,
    int64_t seq,
    const LoraBatchEntry* entry
) {
    if (!entry) return at::zeros({0}, input.options());
    TORCH_CHECK(entry->a_stack.dim() == 4 && entry->b_stack.dim() == 4,
        "dynamic routed-expert LoRA expects rank-4 stacked A/B tensors");
    const int64_t local_experts = entry->a_stack.size(1);
    auto sample_indices = at::floor_divide(token_indices, seq);
    auto pair_indices = sample_indices * local_experts + local_expert_indices;
    auto a_stack = entry->a_stack;
    auto b_stack = entry->b_stack;
    if (a_stack.size(0) == 1 && batch > 1) {
        a_stack = a_stack.expand(
            {batch, a_stack.size(1), a_stack.size(2), a_stack.size(3)});
        b_stack = b_stack.expand(
            {batch, b_stack.size(1), b_stack.size(2), b_stack.size(3)});
    }
    auto a = a_stack.flatten(0, 1)
        .index_select(0, pair_indices).to(input.scalar_type());
    auto b = b_stack.flatten(0, 1)
        .index_select(0, pair_indices).to(input.scalar_type());
    auto lora_input = entry->layout == LoraTpLayout::LatentRank
        ? tp_copy_lora_input(ctx, input) : input;
    auto low_rank = at::bmm(a, lora_input.unsqueeze(-1)).squeeze(-1);
    auto delta = at::bmm(b, low_rank.unsqueeze(-1)).squeeze(-1);
    auto scaling_stack = entry->scaling;
    if (scaling_stack.size(0) == 1 && batch > 1) {
        scaling_stack = scaling_stack.expand({batch, 1, 1});
    }
    auto scaling = scaling_stack.index_select(0, sample_indices)
        .reshape({-1, 1}).to(input.scalar_type());
    auto scaled = delta * scaling;
    return entry->layout == LoraTpLayout::LatentRank
        ? tp_allreduce_lora_delta(ctx, scaled) : scaled;
}

static at::Tensor moe_routed_a2a(
    TrainingContext* training_ctx, ncclComm_t comm,
    const at::Tensor& flat, const at::Tensor& topk_weights,
    const at::Tensor& topk_indices,
    const at::Tensor& experts_gate_up, const at::Tensor& experts_down,
    const RoutedExpertLora& expert_lora,
    int64_t top_k, int64_t intermediate, int64_t expert_count,
    int64_t batch, int64_t seq,
    const LoraBatchEntry* expert_gate_up_lora,
    const LoraBatchEntry* expert_down_lora
) {
    const int64_t hidden_dim = flat.size(1);
    const bool expert_tp = base_tp_mlp_enabled(training_ctx);
    const auto gate_up_layout = expert_tp
        ? LoraTpLayout::ColumnParallel : LoraTpLayout::LatentRank;
    const auto down_layout = expert_tp
        ? LoraTpLayout::RowParallel : LoraTpLayout::LatentRank;
    auto routed_output = at::zeros_like(flat);
    auto token_indices = at::arange(
        flat.size(0), at::TensorOptions().device(flat.device()).dtype(at::kLong));
    auto run_local_experts = [&](const at::Tensor& received,
                                 const at::Tensor& received_tokens,
                                 const at::Tensor& received_experts) {
        auto [sorted_experts, expert_order] = received_experts.sort(0);
        auto selected = received.index_select(0, expert_order);
        auto selected_tokens = received_tokens.index_select(0, expert_order);
        auto counts = at::bincount(
            sorted_experts, c10::nullopt, expert_count);
        auto offsets = counts.cumsum(0).to(at::kInt);
        auto sorted_output = at::zeros_like(selected);

        if (selected.size(0) > 0) {
            at::Tensor expert_out;
#if RUSTRAIN_HAS_ATEN_GROUPED_MM
            const bool use_grouped_mm =
                !env_enabled("QWEN36_DISABLE_GROUPED_MM") &&
                selected.scalar_type() == at::kBFloat16 &&
                hidden_dim % 8 == 0 && intermediate % 8 == 0;
            if (use_grouped_mm) {
                at::Tensor counts_cpu;
                auto fixed_lora_delta = [&](const at::Tensor& input,
                                            const at::Tensor& a,
                                            const at::Tensor& b,
                                            LoraTpLayout layout) {
                    auto lora_input = layout == LoraTpLayout::LatentRank
                        ? tp_copy_lora_input(training_ctx, input) : input;
                    at::Tensor delta;
                    if (a.size(1) % 8 == 0) {
                        auto low_rank = at::_grouped_mm(
                            lora_input, a.transpose(1, 2), offsets);
                        delta = at::_grouped_mm(
                            low_rank, b.transpose(1, 2), offsets);
                    } else {
                        if (!counts_cpu.defined()) {
                            counts_cpu = counts.to(
                                at::TensorOptions().device(at::kCPU));
                        }
                        std::vector<at::Tensor> chunks;
                        chunks.reserve(expert_count);
                        int64_t offset = 0;
                        for (int64_t e_local = 0;
                             e_local < expert_count; ++e_local) {
                            const int64_t rows = counts_cpu.index(
                                {e_local}).item<int64_t>();
                            auto expert_input = lora_input.narrow(
                                0, offset, rows);
                            chunks.push_back(at::matmul(
                                at::matmul(
                                    expert_input, a.select(0, e_local).t()),
                                b.select(0, e_local).t()));
                            offset += rows;
                        }
                        delta = at::cat(chunks, 0);
                    }
                    auto scaled = delta * expert_lora.scaling;
                    return layout == LoraTpLayout::LatentRank
                        ? tp_allreduce_lora_delta(training_ctx, scaled)
                        : scaled;
                };
                auto gu = at::_grouped_mm(
                    selected, experts_gate_up.transpose(1, 2), offsets);
                if (expert_lora.gate_up_a && expert_lora.gate_up_b) {
                    gu = gu + fixed_lora_delta(
                        selected, *expert_lora.gate_up_a,
                        *expert_lora.gate_up_b, gate_up_layout);
                }
                if (expert_gate_up_lora) {
                    gu = gu + dynamic_expert_lora_delta(
                        training_ctx, selected, selected_tokens,
                        sorted_experts, batch, seq, expert_gate_up_lora);
                }
                auto activated = fused_swiglu_op(
                    gu.narrow(-1, 0, intermediate),
                    gu.narrow(-1, intermediate, intermediate), 0.0);
                expert_out = at::_grouped_mm(
                    activated, experts_down.transpose(1, 2), offsets);
                if (expert_lora.down_a && expert_lora.down_b) {
                    expert_out = expert_out + fixed_lora_delta(
                        activated, *expert_lora.down_a,
                        *expert_lora.down_b, down_layout);
                }
                if (expert_down_lora) {
                    expert_out = expert_out + dynamic_expert_lora_delta(
                        training_ctx, activated, selected_tokens,
                        sorted_experts, batch, seq, expert_down_lora);
                }
            } else
#endif
            {
                auto counts_cpu = counts.to(
                    at::TensorOptions().device(at::kCPU));
                expert_out = at::zeros_like(selected);
                int64_t offset = 0;
                for (int64_t e_local = 0; e_local < expert_count; ++e_local) {
                    const int64_t rows =
                        counts_cpu.index({e_local}).item<int64_t>();
                    if (rows > 0) {
                        auto expert_input = selected.narrow(0, offset, rows);
                        auto expert_tokens = selected_tokens.narrow(
                            0, offset, rows);
                        auto local_experts = sorted_experts.narrow(
                            0, offset, rows);
                        auto gu = at::matmul(
                            expert_input,
                            experts_gate_up.select(0, e_local).t());
                        if (expert_lora.gate_up_a && expert_lora.gate_up_b) {
                            auto lora_input = gate_up_layout ==
                                    LoraTpLayout::LatentRank
                                ? tp_copy_lora_input(training_ctx, expert_input)
                                : expert_input;
                            auto delta = at::matmul(
                                at::matmul(lora_input,
                                    expert_lora.gate_up_a->select(
                                        0, e_local).t()),
                                expert_lora.gate_up_b->select(0, e_local).t()) *
                                expert_lora.scaling;
                            gu = gu + (gate_up_layout ==
                                    LoraTpLayout::LatentRank
                                ? tp_allreduce_lora_delta(training_ctx, delta)
                                : delta);
                        }
                        if (expert_gate_up_lora) {
                            gu = gu + dynamic_expert_lora_delta(
                                training_ctx, expert_input, expert_tokens,
                                local_experts, batch, seq,
                                expert_gate_up_lora);
                        }
                        auto activated = fused_swiglu_op(
                            gu.narrow(-1, 0, intermediate),
                            gu.narrow(-1, intermediate, intermediate), 0.0);
                        auto local_out = at::matmul(
                            activated,
                            experts_down.select(0, e_local).t());
                        if (expert_lora.down_a && expert_lora.down_b) {
                            auto lora_input = down_layout ==
                                    LoraTpLayout::LatentRank
                                ? tp_copy_lora_input(training_ctx, activated)
                                : activated;
                            auto delta = at::matmul(
                                at::matmul(lora_input,
                                    expert_lora.down_a->select(
                                        0, e_local).t()),
                                expert_lora.down_b->select(0, e_local).t()) *
                                expert_lora.scaling;
                            local_out = local_out + (down_layout ==
                                    LoraTpLayout::LatentRank
                                ? tp_allreduce_lora_delta(training_ctx, delta)
                                : delta);
                        }
                        if (expert_down_lora) {
                            local_out = local_out + dynamic_expert_lora_delta(
                                training_ctx, activated, expert_tokens,
                                local_experts, batch, seq, expert_down_lora);
                        }
                        expert_out = expert_out.index_add(
                            0,
                            at::arange(offset, offset + rows,
                                expert_order.options()),
                            local_out);
                    }
                    offset += rows;
                }
            }
            sorted_output = expert_out;
        }

        auto local_output = at::zeros_like(received).index_add(
            0, expert_order, sorted_output);
        // Empty destinations still need activation and parameter graph edges
        // so every rank reaches the same optimizer collectives.
        if (!local_output.requires_grad()) {
            at::Tensor anchor = received.sum().to(local_output.scalar_type());
            auto include = [&](const at::Tensor* tensor) {
                if (!tensor || !tensor->defined() ||
                    !tensor->requires_grad()) return;
                anchor = anchor + tensor->sum().to(local_output.scalar_type());
            };
            include(expert_lora.gate_up_a);
            include(expert_lora.gate_up_b);
            include(expert_lora.down_a);
            include(expert_lora.down_b);
            if (expert_gate_up_lora) {
                include(&expert_gate_up_lora->a_stack);
                include(&expert_gate_up_lora->b_stack);
            }
            if (expert_down_lora) {
                include(&expert_down_lora->a_stack);
                include(&expert_down_lora->b_stack);
            }
            local_output = local_output + anchor * 0.0;
        }
        return local_output;
    };

    auto dispatch_and_combine = [&](const at::Tensor& assignment_input,
                                    const at::Tensor& assignment_experts,
                                    const at::Tensor& assignment_tokens,
                                    const at::Tensor& assignment_weights) {
        // `received_tokens` preserves the source flattened row index through
        // dispatch so dynamic multi-LoRA can recover the tenant/sample row.
        auto dispatched = Qwen36A2ADispatchFunction::apply(
            assignment_input, assignment_experts, assignment_tokens,
            expert_count,
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(comm)));
        auto received = dispatched[0];
        auto received_tokens = dispatched[1];
        auto received_experts = dispatched[2];
        auto send_index = dispatched[3];
        auto send_counts = dispatched[4];
        auto recv_counts = dispatched[5];
        auto local_output = run_local_experts(
            received, received_tokens, received_experts);
        auto returned = Qwen36A2ACombineFunction::apply(
            local_output, send_counts, recv_counts,
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(comm)));
        returned = tp_allreduce_base_mlp(training_ctx, returned);
        auto source_tokens = assignment_tokens.index_select(0, send_index);
        auto source_weights = assignment_weights
            .index_select(0, send_index).unsqueeze(-1);
        routed_output = routed_output.index_add(
            0, source_tokens, returned * source_weights);
    };

    if (env_enabled("QWEN36_EP_A2A_PACKED", true)) {
        auto assignment_tokens = token_indices.unsqueeze(1)
            .expand({flat.size(0), top_k}).reshape({-1});
        auto assignment_experts = topk_indices.reshape({-1}).contiguous();
        auto assignment_weights = topk_weights.reshape({-1}).contiguous();
        dispatch_and_combine(
            flat, assignment_experts, assignment_tokens,
            assignment_weights);
    } else {
        for (int64_t kk = 0; kk < top_k; ++kk) {
            auto expert_indices = topk_indices.select(-1, kk).contiguous();
            auto expert_weights = topk_weights.select(-1, kk).contiguous();
            dispatch_and_combine(
                flat, expert_indices, token_indices, expert_weights);
        }
    }
    return routed_output;
}

static at::Tensor moe_forward(
    TrainingContext* training_ctx,
    void* nccl_comm_v, void* nccl_stream_v,
    const at::Tensor& hidden,
    const at::Tensor& gate_w, const at::Tensor& shared_expert_gate_w,
    const at::Tensor& shared_gate_proj, const at::Tensor& shared_up_proj, const at::Tensor& shared_down_proj,
    const at::Tensor& experts_gate_up, const at::Tensor& experts_down,
    const RoutedExpertLora& expert_lora,
    int64_t num_experts, int64_t top_k, int64_t intermediate,
    bool norm_topk_prob, int64_t expert_start, int64_t expert_count,
    at::ScalarType compute_type,
    const LoraBatchEntry* shared_gate_lora = nullptr,
    const LoraBatchEntry* shared_up_lora = nullptr,
    const LoraBatchEntry* shared_down_lora = nullptr,
    const LoraBatchEntry* expert_gate_up_lora = nullptr,
    const LoraBatchEntry* expert_down_lora = nullptr
) {
    int64_t batch = hidden.size(0), seq = hidden.size(1), hidden_dim = hidden.size(2);
    auto device = hidden.device();
    auto flat = hidden.reshape({batch * seq, hidden_dim});
    int64_t N = flat.size(0);
    const bool expert_tp = base_tp_mlp_enabled(training_ctx);
    const int64_t local_intermediate = experts_gate_up.size(1) / 2;
    TORCH_CHECK(experts_gate_up.dim() == 3 && experts_down.dim() == 3 &&
            experts_gate_up.size(0) == expert_count &&
            experts_down.size(0) == expert_count &&
            experts_gate_up.size(1) == 2 * local_intermediate &&
            experts_gate_up.size(2) == hidden_dim &&
            experts_down.size(1) == hidden_dim &&
            experts_down.size(2) == local_intermediate,
        "routed expert local weight shapes are inconsistent: gate_up=",
        experts_gate_up.sizes(), " down=", experts_down.sizes());
    if (expert_tp) {
        const int64_t tp_world_size = base_tp_mlp_world_size(training_ctx);
        TORCH_CHECK(intermediate > 0 &&
                intermediate % tp_world_size == 0 &&
                local_intermediate == intermediate / tp_world_size,
            "routed expert TP intermediate mismatch: global=", intermediate,
            " local=", local_intermediate,
            " TP_SIZE=", tp_world_size);
    } else {
        TORCH_CHECK(local_intermediate == intermediate,
            "unsharded routed expert intermediate mismatch: configured=",
            intermediate, " weight=", local_intermediate);
    }
    auto expert_flat = tp_copy_base_mlp_input(training_ctx, flat);

    auto router_logits = at::matmul(flat, gate_w.t());
    auto routing_weights = router_logits.softmax(-1, at::kFloat);
    auto [topk_weights, topk_indices] = routing_weights.topk(top_k, -1, true, true);
    if (norm_topk_prob) {
        auto denom = topk_weights.sum(-1, true).clamp_min(1e-9);
        topk_weights = topk_weights / denom;
    }
    topk_weights = topk_weights.to(compute_type);

    auto routed_output = at::zeros(flat.sizes(), flat.options());
    if (env_enabled("QWEN36_EP_A2A_SHARDED") && expert_count < num_experts) {
        TORCH_CHECK(nccl_comm_v && env_enabled("QWEN36_EP_A2A"),
            "QWEN36_EP_A2A_SHARDED=1 requires an initialized EP communicator "
            "and QWEN36_EP_A2A=1");
    }
    bool use_a2a = false;
    const bool sharded_a2a_mode = env_enabled("QWEN36_EP_A2A_SHARDED") &&
        expert_count < num_experts;
    if (expert_tp && expert_count < num_experts) {
        TORCH_CHECK(sharded_a2a_mode && env_enabled("QWEN36_EP_A2A") &&
                env_enabled("QWEN36_EP_A2A_PACKED", true),
            "expert TP with expert sharding requires packed sharded A2A: set "
            "QWEN36_EP_A2A=1, QWEN36_EP_A2A_SHARDED=1, and keep "
            "QWEN36_EP_A2A_PACKED enabled");
    }
    if (nccl_comm_v && env_enabled("QWEN36_EP_A2A") &&
        ((!expert_gate_up_lora && !expert_down_lora) || sharded_a2a_mode)) {
        auto comm = reinterpret_cast<ncclComm_t>(nccl_comm_v);
        int world = 1, rank = 0;
        auto err = ncclCommCount(comm, &world);
        TORCH_CHECK(err == ncclSuccess, "ncclCommCount failed: ",
            ncclGetErrorString(err));
        err = ncclCommUserRank(comm, &rank);
        TORCH_CHECK(err == ncclSuccess, "ncclCommUserRank failed: ",
            ncclGetErrorString(err));
        TORCH_CHECK(world * expert_count == num_experts,
            "EP A2A requires equal contiguous expert partitions: world=", world,
            " local_experts=", expert_count, " global_experts=", num_experts);
        TORCH_CHECK(expert_start == rank * expert_count,
            "EP A2A requires rank-contiguous expert ownership: rank=", rank,
            " expert_start=", expert_start, " local_experts=", expert_count);
        use_a2a = world > 1;
        if (use_a2a) {
            routed_output = moe_routed_a2a(
                training_ctx, comm, expert_flat, topk_weights, topk_indices,
                experts_gate_up, experts_down, expert_lora,
                top_k, local_intermediate, expert_count, batch, seq,
                expert_gate_up_lora, expert_down_lora);
        }
    }

    // Debug: dump MoE routing and weight stats
    if (getenv("QWEN36_DUMP_MOE")) {
        auto rl_f = router_logits.to(at::kFloat);
        auto rw_f = topk_weights.to(at::kFloat);
        auto egu_f = experts_gate_up.select(0, 0).to(at::kFloat);
        auto ed_f = experts_down.select(0, 0).to(at::kFloat);
        auto sg_f = shared_gate_proj.to(at::kFloat);
        auto sd_f = shared_down_proj.to(at::kFloat);
        auto seg_f = at::sigmoid(at::matmul(flat, shared_expert_gate_w.t())).to(at::kFloat);
    }

    // Sort-based expert dispatch — eliminates eq/nonzero/index_select per expert.
    // At large N (1000+), the old for-loop with eq+nonzero per expert was O(experts × tokens)
    // with high kernel launch overhead. This approach pre-sorts tokens by expert assignment.
    //
    // Autograd note: routing indices/weights are computed in no-grad forward (detached).
    // Only matmul inputs/outputs participate in autograd. sort/index_select/index_add
    // gradients are handled by PyTorch automatically.
    if (!use_a2a) for (int64_t kk = 0; kk < top_k; kk++) {
        auto expert_indices = topk_indices.select(-1, kk);   // [N*seq]
        auto expert_weights = topk_weights.select(-1, kk);   // [N*seq]

        // Sort tokens by expert index → contiguous groups per expert
        auto [sorted_indices, sort_order] = expert_indices.sort(0);
        // sorted_indices: expert IDs in ascending order
        // sort_order: original token positions in sorted order

        // Find expert boundaries via bincount + cumsum
        auto counts = at::bincount(sorted_indices, c10::nullopt, expert_start + expert_count);
        // counts[e_global] = number of tokens assigned to expert e
        // Gather tokens in sorted order (contiguous per expert)
        auto gathered = expert_flat.index_select(0, sort_order);
        auto local_slot_output = at::zeros_like(flat);

        // The grouped single-rank path knows that every sorted row is local
        // and therefore needs no host-visible expert counts. Materialize the
        // CPU copy lazily only for EP slicing or the legacy per-expert loop.
        at::Tensor counts_cpu;
        auto get_counts_cpu = [&]() -> const at::Tensor& {
            if (!counts_cpu.defined()) {
                counts_cpu = counts.to(at::TensorOptions().device(at::kCPU));
            }
            return counts_cpu;
        };

#if RUSTRAIN_HAS_ATEN_GROUPED_MM
        // PyTorch 2.12+ exposes the same CUTLASS grouped-GEMM primitive used
        // by its native MoE path. It consumes sorted token rows plus cumulative
        // expert offsets, eliminating one GEMM launch per local expert. Older
        // libtorch builds compile the fallback loop below.
        bool grouped_lora_compatible = true;
        if (expert_lora.gate_up_a) {
            grouped_lora_compatible = expert_lora.gate_up_a->size(1) % 8 == 0;
        }
        if (expert_lora.down_a) {
            grouped_lora_compatible = grouped_lora_compatible &&
                expert_lora.down_a->size(1) % 8 == 0;
        }
        const bool use_grouped_mm = !env_enabled("QWEN36_DISABLE_GROUPED_MM") &&
            compute_type == at::kBFloat16 && grouped_lora_compatible &&
            hidden_dim % 8 == 0 && local_intermediate % 8 == 0;
        if (use_grouped_mm) {
            const bool owns_all_experts = expert_start == 0 &&
                expert_count == counts.size(0);
            const int64_t local_start = owns_all_experts || expert_start == 0
                ? 0
                : get_counts_cpu().narrow(0, 0, expert_start).sum().item<int64_t>();
            const int64_t local_tokens = owns_all_experts
                ? gathered.size(0)
                : get_counts_cpu()
                    .narrow(0, expert_start, expert_count).sum().item<int64_t>();
            if (local_tokens > 0) {
                if (env_enabled("QWEN36_REPORT_GROUPED_MM")) {
                    std::fprintf(
                        stderr,
                        "[q36_moe] grouped_mm experts=%ld tokens=%ld hidden=%ld intermediate=%ld\n",
                        static_cast<long>(expert_count),
                        static_cast<long>(local_tokens),
                        static_cast<long>(hidden_dim),
                        static_cast<long>(local_intermediate));
                }
                auto selected = gathered.narrow(0, local_start, local_tokens);
                auto token_indices = sort_order.narrow(
                    0, local_start, local_tokens);
                auto local_expert_indices = sorted_indices.narrow(
                    0, local_start, local_tokens) - expert_start;
                auto offsets = counts.narrow(0, expert_start, expert_count)
                    .cumsum(0).to(at::kInt);
                auto gu = at::_grouped_mm(
                    selected, experts_gate_up.transpose(1, 2), offsets);
                if (expert_lora.gate_up_a && expert_lora.gate_up_b) {
                    auto lora_input = expert_tp
                        ? selected : tp_copy_lora_input(training_ctx, selected);
                    auto low_rank = at::_grouped_mm(
                        lora_input, expert_lora.gate_up_a->transpose(1, 2), offsets);
                    auto delta = at::_grouped_mm(
                        low_rank, expert_lora.gate_up_b->transpose(1, 2), offsets);
                    auto scaled = delta * expert_lora.scaling;
                    gu = gu + (expert_tp
                        ? scaled
                        : tp_allreduce_lora_delta(training_ctx, scaled));
                }
                if (expert_gate_up_lora) {
                    gu = gu + dynamic_expert_lora_delta(
                        training_ctx, selected, token_indices, local_expert_indices,
                        batch, seq, expert_gate_up_lora);
                }
                auto activated = fused_swiglu_op(
                    gu.narrow(-1, 0, local_intermediate),
                    gu.narrow(-1, local_intermediate, local_intermediate), 0.0);
                auto expert_out = at::_grouped_mm(
                    activated, experts_down.transpose(1, 2), offsets);
                if (expert_lora.down_a && expert_lora.down_b) {
                    auto lora_input = expert_tp
                        ? activated : tp_copy_lora_input(training_ctx, activated);
                    auto low_rank = at::_grouped_mm(
                        lora_input, expert_lora.down_a->transpose(1, 2), offsets);
                    auto delta = at::_grouped_mm(
                        low_rank, expert_lora.down_b->transpose(1, 2), offsets);
                    auto scaled = delta * expert_lora.scaling;
                    expert_out = expert_out + (expert_tp
                        ? scaled
                        : tp_allreduce_lora_delta(training_ctx, scaled));
                }
                if (expert_down_lora) {
                    expert_out = expert_out + dynamic_expert_lora_delta(
                        training_ctx, activated, token_indices, local_expert_indices,
                        batch, seq, expert_down_lora);
                }
                local_slot_output = local_slot_output.index_add(
                    0, token_indices, expert_out);
            }
            local_slot_output = tp_allreduce_base_mlp(
                training_ctx, local_slot_output);
            routed_output = routed_output + local_slot_output *
                expert_weights.unsqueeze(-1);
            continue;
        }
#endif

        // Process each expert's contiguous token slice. `gathered` contains
        // all global experts in sorted order, so rank>0 must skip tokens for
        // experts owned by lower ranks before taking its local range.
        int64_t offset = expert_start > 0
            ? get_counts_cpu().narrow(0, 0, expert_start).sum().item<int64_t>()
            : 0;
        for (int64_t e_local = 0; e_local < expert_count; e_local++) {
            int64_t e_global = expert_start + e_local;
            int64_t n_tokens = get_counts_cpu().index({e_global}).item<int64_t>();
            if (n_tokens > 0) {
                auto selected = gathered.narrow(0, offset, n_tokens);  // zero-copy view!
                auto token_indices = sort_order.narrow(0, offset, n_tokens);
                auto local_expert_indices = sorted_indices.narrow(0, offset, n_tokens)
                    - expert_start;
                auto egu = experts_gate_up.select(0, e_local);
                auto ed = experts_down.select(0, e_local);
                auto gu = at::matmul(selected, egu.t());
                if (expert_lora.gate_up_a && expert_lora.gate_up_b) {
                    auto a = expert_lora.gate_up_a->select(0, e_local);
                    auto b = expert_lora.gate_up_b->select(0, e_local);
                    auto lora_input = expert_tp
                        ? selected : tp_copy_lora_input(training_ctx, selected);
                    auto delta = at::matmul(at::matmul(lora_input, a.t()), b.t())
                        * expert_lora.scaling;
                    gu = gu + (expert_tp
                        ? delta
                        : tp_allreduce_lora_delta(training_ctx, delta));
                }
                if (expert_gate_up_lora) {
                    gu = gu + dynamic_expert_lora_delta(
                        training_ctx, selected, token_indices, local_expert_indices,
                        batch, seq, expert_gate_up_lora);
                }
                auto gate_part = gu.narrow(-1, 0, local_intermediate);
                auto up_part = gu.narrow(
                    -1, local_intermediate, local_intermediate);
                auto activated = fused_swiglu_op(gate_part, up_part, 0.0);
                auto expert_out = at::matmul(activated, ed.t());
                if (expert_lora.down_a && expert_lora.down_b) {
                    auto a = expert_lora.down_a->select(0, e_local);
                    auto b = expert_lora.down_b->select(0, e_local);
                    auto lora_input = expert_tp
                        ? activated : tp_copy_lora_input(training_ctx, activated);
                    auto delta = at::matmul(at::matmul(lora_input, a.t()), b.t())
                        * expert_lora.scaling;
                    expert_out = expert_out + (expert_tp
                        ? delta
                        : tp_allreduce_lora_delta(training_ctx, delta));
                }
                if (expert_down_lora) {
                    expert_out = expert_out + dynamic_expert_lora_delta(
                        training_ctx, activated, token_indices, local_expert_indices,
                        batch, seq, expert_down_lora);
                }
                local_slot_output = local_slot_output.index_add(
                    0, token_indices, expert_out);
            }
            offset += n_tokens;
        }
        local_slot_output = tp_allreduce_base_mlp(
            training_ctx, local_slot_output);
        routed_output = routed_output + local_slot_output *
            expert_weights.unsqueeze(-1);
    }

    // A rank can receive no tokens for any of its local experts. Keep a
    // zero-valued dependency on routed-expert LoRA tensors so autograd still
    // produces defined zero gradients and every rank reaches the same NCCL
    // collectives. This changes only the graph, not the routed output values.
    if (!use_a2a && !routed_output.requires_grad()) {
        at::Tensor graph_anchor;
        auto include_anchor = [&](const at::Tensor* tensor) {
            if (!tensor || !tensor->defined() || !tensor->requires_grad()) return;
            auto contribution = tensor->sum().to(routed_output.scalar_type());
            graph_anchor = graph_anchor.defined()
                ? graph_anchor + contribution
                : contribution;
        };
        include_anchor(expert_lora.gate_up_a);
        include_anchor(expert_lora.gate_up_b);
        include_anchor(expert_lora.down_a);
        include_anchor(expert_lora.down_b);
        if (expert_gate_up_lora) {
            include_anchor(&expert_gate_up_lora->a_stack);
            include_anchor(&expert_gate_up_lora->b_stack);
        }
        if (expert_down_lora) {
            include_anchor(&expert_down_lora->a_stack);
            include_anchor(&expert_down_lora->b_stack);
        }
        if (graph_anchor.defined()) {
            routed_output = routed_output + graph_anchor * 0.0;
        }
    }

    // EP all-reduce via NcclAllReduceFunction — custom autograd Function.
    if (nccl_comm_v && !use_a2a) {
        auto nccl_comm = reinterpret_cast<ncclComm_t>(nccl_comm_v);
        routed_output = NcclAllReduceFunction::apply(
            routed_output, (int64_t)nccl_comm,
            (int64_t)reinterpret_cast<uintptr_t>(nccl_stream_v));
    }

    // Shared expert (same as before, with fused SwiGLU)
    auto shared_input = tp_copy_base_mlp_input(training_ctx, flat);
    auto shared_gate = at::matmul(shared_input, shared_gate_proj.t());
    auto shared_up = at::matmul(shared_input, shared_up_proj.t());
    if (shared_gate_lora) {
        shared_gate = add_batched_lora(
            training_ctx, shared_gate.reshape({batch, seq, -1}),
            shared_input.reshape({batch, seq, hidden_dim}),
            shared_gate_lora)
            .reshape({batch * seq, -1});
    }
    if (shared_up_lora) {
        shared_up = add_batched_lora(
            training_ctx, shared_up.reshape({batch, seq, -1}),
            shared_input.reshape({batch, seq, hidden_dim}),
            shared_up_lora)
            .reshape({batch * seq, -1});
    }
    auto shared_hidden = fused_swiglu_op(
        shared_gate.reshape({batch, seq, -1}),
        shared_up.reshape({batch, seq, -1}), 0.0);
    auto shared_out = at::matmul(shared_hidden.reshape({batch * seq, -1}), shared_down_proj.t());
    if (shared_down_lora) {
        shared_out = add_batched_lora(
            training_ctx, shared_out.reshape({batch, seq, -1}), shared_hidden,
            shared_down_lora)
            .reshape({batch * seq, -1});
    }
    shared_out = tp_allreduce_base_mlp(training_ctx, shared_out);
    auto seg = at::sigmoid(at::matmul(flat, shared_expert_gate_w.t())).to(compute_type);
    shared_out = (shared_out * seg).to(compute_type);

    // Debug: dump routed_output AFTER loop
    if (getenv("QWEN36_DUMP_MOE")) {
        auto ro_f = routed_output.to(at::kFloat);
        auto so_f = shared_out.to(at::kFloat);
    }

    return (routed_output + shared_out).reshape({batch, seq, hidden_dim});
}

// ──────────────────────────────────────────────────────────────────────
// Layer config + forward
// ──────────────────────────────────────────────────────────────────────

struct LayerConfig {
    int64_t layer_type, num_heads, num_kv_heads, head_dim;
    int64_t num_k_heads, key_dim, num_v_heads, val_dim, conv_kernel;
    double partial_rotary_factor, rope_theta, rms_eps;
    int64_t num_experts, top_k, moe_intermediate, expert_start, expert_count;
    int64_t intermediate_size;  // dense MLP intermediate size (0 for MoE)
    int32_t norm_topk_prob;
    // NCCL handles for EP all-reduce (set per-layer from TrainingContext)
    // nullptr when single-GPU. Stored here so moe_forward can access them
    // without needing the full TrainingContext definition.
    void* nccl_comm = nullptr;
    void* nccl_stream = nullptr;
};

// Weight count per layer: dense has 3 MLP weights, MoE has 7 MoE weights.
// Full attention: 2 norm + 6 attn + (3 dense | 7 moe) = 11 | 15
// Linear attention: 2 norm + 9 linear_attn + (3 dense | 7 moe) = 14 | 18
static inline int64_t weight_count_for_layer(const LayerConfig& cfg) {
    int64_t attn_w = (cfg.layer_type == 0) ? 6 : 9;
    int64_t mlp_w = (cfg.num_experts > 0) ? 7 : 3;
    return 2 + attn_w + mlp_w;
}

enum class LoraSegment : uint8_t { Attention, Mlp };

static bool is_mlp_lora_target(const std::string& name) {
    return name == "gate_proj" || name == "up_proj" || name == "down_proj" ||
        name == "shared_gate_proj" || name == "shared_up_proj" ||
        name == "shared_down_proj" || name == "experts_gate_up_proj" ||
        name == "experts_down_proj";
}

struct LoraProjectionSpec {
    const char* name;
    int64_t weight_index;
    LoraSegment segment;
    bool grouped_expert;
};

struct LoraProjectionTable {
    std::array<LoraProjectionSpec, 10> entries;
    int64_t count = 0;

    void add(const char* name, int64_t weight_index, LoraSegment segment) {
        TORCH_CHECK(count < (int64_t)entries.size(), "too many LoRA projections in layer");
        entries[count++] = {name, weight_index, segment, false};
    }

    void add_grouped_expert(const char* name, int64_t weight_index) {
        TORCH_CHECK(count < (int64_t)entries.size(), "too many LoRA projections in layer");
        entries[count++] = {name, weight_index, LoraSegment::Mlp, true};
    }
};

static LoraProjectionTable lora_projection_table(const LayerConfig& cfg) {
    LoraProjectionTable table;
    if (cfg.layer_type == 0) {
        table.add("q_proj", 2, LoraSegment::Attention);
        table.add("k_proj", 4, LoraSegment::Attention);
        table.add("v_proj", 6, LoraSegment::Attention);
        table.add("o_proj", 7, LoraSegment::Attention);
    } else {
        table.add("in_proj_qkv", 2, LoraSegment::Attention);
        table.add("in_proj_z", 3, LoraSegment::Attention);
        table.add("in_proj_a", 4, LoraSegment::Attention);
        table.add("in_proj_b", 5, LoraSegment::Attention);
        table.add("out_proj", 10, LoraSegment::Attention);
    }

    const int64_t mlp_start = cfg.layer_type == 0 ? 8 : 11;
    if (cfg.num_experts > 0) {
        table.add("shared_gate_proj", mlp_start + 2, LoraSegment::Mlp);
        table.add("shared_up_proj", mlp_start + 3, LoraSegment::Mlp);
        table.add("shared_down_proj", mlp_start + 4, LoraSegment::Mlp);
        table.add_grouped_expert("experts_gate_up_proj", mlp_start + 5);
        table.add_grouped_expert("experts_down_proj", mlp_start + 6);
    } else {
        table.add("gate_proj", mlp_start, LoraSegment::Mlp);
        table.add("up_proj", mlp_start + 1, LoraSegment::Mlp);
        table.add("down_proj", mlp_start + 2, LoraSegment::Mlp);
    }
    return table;
}

static inline int64_t lora_pair_count(const LayerConfig& cfg) {
    return lora_projection_table(cfg).count;
}

static int64_t lora_pair_index(const LayerConfig& cfg, const char* name) {
    auto table = lora_projection_table(cfg);
    for (int64_t i = 0; i < table.count; ++i) {
        if (std::strcmp(table.entries[i].name, name) == 0) return i;
    }
    return -1;
}

static RoutedExpertLora routed_expert_lora(
    TrainingContext* ctx, int64_t layer_idx, const LayerConfig& cfg);
static at::Tensor tp_allreduce_base_mlp(
    TrainingContext* ctx, const at::Tensor& local_output);
static at::Tensor tp_copy_base_mlp_input(
    TrainingContext* ctx, const at::Tensor& input);
static at::Tensor tp_allreduce_base_attention(
    TrainingContext* ctx, const at::Tensor& local_output);
static at::Tensor tp_copy_base_attention_input(
    TrainingContext* ctx, const at::Tensor& input);
static bool base_tp_attention_enabled(const TrainingContext* ctx);

static at::Tensor forward_single_layer(
    TrainingContext* ctx, const at::Tensor& hidden, at::Tensor** w, const LayerConfig* cfg,
    int64_t layer_idx, at::ScalarType kind,
    const at::Tensor& attention_mask, bool use_batched = false
) {
    auto input_norm = *w[0];
    auto post_norm = *w[1];
    auto attn_input = rms_norm(hidden, input_norm, cfg->rms_eps);
    bool is_moe = (cfg->num_experts > 0);

    at::Tensor attn_output;
    if (cfg->layer_type == 0) {
        // Full attention
        auto q_proj = *w[2], q_norm = *w[3], k_proj = *w[4], k_norm = *w[5], v_proj = *w[6], o_proj = *w[7];
        TORCH_CHECK(!base_tp_attention_enabled(ctx) || use_batched,
            "base full-attention TP requires the activation-level LoRA path");
        if (use_batched) {
            // Activation-level LoRA: pass base weights, apply delta inside attention
            attn_output = full_attention_batched(
                ctx, attn_input, layer_idx,
                q_proj, q_norm, k_proj, k_norm, v_proj, o_proj,
                cfg->num_heads, cfg->num_kv_heads, cfg->head_dim,
                cfg->partial_rotary_factor, cfg->rope_theta, cfg->rms_eps, kind,
                attention_mask);
        } else {
            // Weight-level LoRA (legacy: modify weights, single forward)
            q_proj = apply_multi_lora(ctx, layer_idx, 0, q_proj);
            k_proj = apply_multi_lora(ctx, layer_idx, 1, k_proj);
            v_proj = apply_multi_lora(ctx, layer_idx, 2, v_proj);
            o_proj = apply_multi_lora(ctx, layer_idx, 3, o_proj);
            attn_output = full_attention(attn_input, q_proj, q_norm, k_proj, k_norm, v_proj, o_proj,
                cfg->num_heads, cfg->num_kv_heads, cfg->head_dim,
                cfg->partial_rotary_factor, cfg->rope_theta, cfg->rms_eps, kind,
                attention_mask);
        }
        auto post_attn = rms_norm(hidden + attn_output, post_norm, cfg->rms_eps);
        if (is_moe) {
            const int64_t shared_gate_pair = lora_pair_index(*cfg, "shared_gate_proj");
            const int64_t shared_up_pair = lora_pair_index(*cfg, "shared_up_proj");
            const int64_t shared_down_pair = lora_pair_index(*cfg, "shared_down_proj");
            const int64_t expert_gate_up_pair = lora_pair_index(*cfg, "experts_gate_up_proj");
            const int64_t expert_down_pair = lora_pair_index(*cfg, "experts_down_proj");
            auto shared_gate = use_batched ? *w[10]
                : apply_multi_lora(ctx, layer_idx, shared_gate_pair, *w[10]);
            auto shared_up = use_batched ? *w[11]
                : apply_multi_lora(ctx, layer_idx, shared_up_pair, *w[11]);
            auto shared_down = use_batched ? *w[12]
                : apply_multi_lora(ctx, layer_idx, shared_down_pair, *w[12]);
            auto expert_lora = routed_expert_lora(ctx, layer_idx, *cfg);
            auto mlp_out = moe_forward(ctx, cfg->nccl_comm, cfg->nccl_stream, post_attn,
                *w[8], *w[9], shared_gate, shared_up, shared_down, *w[13], *w[14],
                expert_lora,
                cfg->num_experts, cfg->top_k, cfg->moe_intermediate,
                cfg->norm_topk_prob != 0, cfg->expert_start, cfg->expert_count, kind,
                use_batched ? lora_batch_entry(ctx, layer_idx, shared_gate_pair) : nullptr,
                use_batched ? lora_batch_entry(ctx, layer_idx, shared_up_pair) : nullptr,
                use_batched ? lora_batch_entry(ctx, layer_idx, shared_down_pair) : nullptr,
                use_batched ? lora_batch_entry(ctx, layer_idx, expert_gate_up_pair) : nullptr,
                use_batched ? lora_batch_entry(ctx, layer_idx, expert_down_pair) : nullptr);
            return hidden + attn_output + mlp_out;
        } else {
            if (use_batched) {
                auto mlp_out = dense_mlp_forward_batched(
                    ctx, layer_idx, post_attn, *w[8], *w[9], *w[10], kind);
                return hidden + attn_output + mlp_out;
            }
            auto gate = apply_multi_lora(ctx, layer_idx,
                lora_pair_index(*cfg, "gate_proj"), *w[8]);
            auto up = apply_multi_lora(ctx, layer_idx,
                lora_pair_index(*cfg, "up_proj"), *w[9]);
            auto down = apply_multi_lora(ctx, layer_idx,
                lora_pair_index(*cfg, "down_proj"), *w[10]);
            auto mlp_input = tp_copy_base_mlp_input(ctx, post_attn);
            auto mlp_out = tp_allreduce_base_mlp(
                ctx, dense_mlp_forward(mlp_input, gate, up, down, kind));
            return hidden + attn_output + mlp_out;
        }
    } else {
        // Linear attention
        auto in_proj_qkv = *w[2], in_proj_z = *w[3], in_proj_a = *w[4], in_proj_b = *w[5];
        auto a_log = *w[6], dt_bias = *w[7], conv1d_w = *w[8], norm_w = *w[9], out_proj = *w[10];
        TORCH_CHECK(!base_tp_attention_enabled(ctx) || use_batched,
            "base linear-attention TP requires the activation-level LoRA path");
        if (use_batched) {
            attn_output = linear_attention_batched(
                ctx, attn_input, layer_idx,
                in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
                a_log, dt_bias, conv1d_w, norm_w, out_proj,
                cfg->num_k_heads, cfg->key_dim, cfg->num_v_heads, cfg->val_dim,
                cfg->conv_kernel, cfg->rms_eps, kind);
        } else {
            in_proj_qkv = apply_multi_lora(ctx, layer_idx, 0, in_proj_qkv);
            in_proj_z = apply_multi_lora(ctx, layer_idx, 1, in_proj_z);
            in_proj_a = apply_multi_lora(ctx, layer_idx, 2, in_proj_a);
            in_proj_b = apply_multi_lora(ctx, layer_idx, 3, in_proj_b);
            out_proj = apply_multi_lora(ctx, layer_idx, 4, out_proj);
            attn_output = linear_attention(attn_input, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
                a_log, dt_bias, conv1d_w, norm_w, out_proj,
                cfg->num_k_heads, cfg->key_dim, cfg->num_v_heads, cfg->val_dim,
                cfg->conv_kernel, cfg->rms_eps, kind);
        }
        auto post_attn = rms_norm(hidden + attn_output, post_norm, cfg->rms_eps);
        if (is_moe) {
            const int64_t shared_gate_pair = lora_pair_index(*cfg, "shared_gate_proj");
            const int64_t shared_up_pair = lora_pair_index(*cfg, "shared_up_proj");
            const int64_t shared_down_pair = lora_pair_index(*cfg, "shared_down_proj");
            const int64_t expert_gate_up_pair = lora_pair_index(*cfg, "experts_gate_up_proj");
            const int64_t expert_down_pair = lora_pair_index(*cfg, "experts_down_proj");
            auto shared_gate = use_batched ? *w[13]
                : apply_multi_lora(ctx, layer_idx, shared_gate_pair, *w[13]);
            auto shared_up = use_batched ? *w[14]
                : apply_multi_lora(ctx, layer_idx, shared_up_pair, *w[14]);
            auto shared_down = use_batched ? *w[15]
                : apply_multi_lora(ctx, layer_idx, shared_down_pair, *w[15]);
            auto expert_lora = routed_expert_lora(ctx, layer_idx, *cfg);
            auto mlp_out = moe_forward(ctx, cfg->nccl_comm, cfg->nccl_stream, post_attn,
                *w[11], *w[12], shared_gate, shared_up, shared_down, *w[16], *w[17],
                expert_lora,
                cfg->num_experts, cfg->top_k, cfg->moe_intermediate,
                cfg->norm_topk_prob != 0, cfg->expert_start, cfg->expert_count, kind,
                use_batched ? lora_batch_entry(ctx, layer_idx, shared_gate_pair) : nullptr,
                use_batched ? lora_batch_entry(ctx, layer_idx, shared_up_pair) : nullptr,
                use_batched ? lora_batch_entry(ctx, layer_idx, shared_down_pair) : nullptr,
                use_batched ? lora_batch_entry(ctx, layer_idx, expert_gate_up_pair) : nullptr,
                use_batched ? lora_batch_entry(ctx, layer_idx, expert_down_pair) : nullptr);
            return hidden + attn_output + mlp_out;
        } else {
            if (use_batched) {
                auto mlp_out = dense_mlp_forward_batched(
                    ctx, layer_idx, post_attn, *w[11], *w[12], *w[13], kind);
                return hidden + attn_output + mlp_out;
            }
            auto gate = apply_multi_lora(ctx, layer_idx,
                lora_pair_index(*cfg, "gate_proj"), *w[11]);
            auto up = apply_multi_lora(ctx, layer_idx,
                lora_pair_index(*cfg, "up_proj"), *w[12]);
            auto down = apply_multi_lora(ctx, layer_idx,
                lora_pair_index(*cfg, "down_proj"), *w[13]);
            auto mlp_input = tp_copy_base_mlp_input(ctx, post_attn);
            auto mlp_out = tp_allreduce_base_mlp(
                ctx, dense_mlp_forward(mlp_input, gate, up, down, kind));
            return hidden + attn_output + mlp_out;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────

// Device-side pointer buffer cache for multi-tensor fused Adam.
// Uses at::Tensor (PyTorch caching allocator) — no cudaMalloc/cudaFree needed.
struct AdamDevBuffers {
    at::Tensor params_buf;   // [max_n] kLong — void* stored as int64_t
    at::Tensor grads_buf;    // [max_n] kLong
    at::Tensor m_buf;        // [max_n] kLong — float* stored as int64_t
    at::Tensor v_buf;        // [max_n] kLong
    at::Tensor dst_params_buf;  // [max_n] kLong
    at::Tensor dst_m_buf;       // [max_n] kLong
    at::Tensor dst_v_buf;       // [max_n] kLong
    at::Tensor sizes_buf;    // [max_n] kInt
    at::Tensor lr_buf;       // [max_n] kFloat
    at::Tensor eps_buf;      // [max_n] kFloat
    int capacity = 0;

    void ensure(int n, const at::Tensor& ref) {
        if (n <= capacity) return;
        auto dev = ref.device();
        params_buf = at::empty({n}, at::TensorOptions().dtype(at::kLong).device(dev));
        grads_buf  = at::empty({n}, at::TensorOptions().dtype(at::kLong).device(dev));
        m_buf      = at::empty({n}, at::TensorOptions().dtype(at::kLong).device(dev));
        v_buf      = at::empty({n}, at::TensorOptions().dtype(at::kLong).device(dev));
        dst_params_buf = at::empty({n}, at::TensorOptions().dtype(at::kLong).device(dev));
        dst_m_buf = at::empty({n}, at::TensorOptions().dtype(at::kLong).device(dev));
        dst_v_buf = at::empty({n}, at::TensorOptions().dtype(at::kLong).device(dev));
        sizes_buf  = at::empty({n}, at::TensorOptions().dtype(at::kInt).device(dev));
        lr_buf     = at::empty({n}, at::TensorOptions().dtype(at::kFloat).device(dev));
        eps_buf    = at::empty({n}, at::TensorOptions().dtype(at::kFloat).device(dev));
        capacity = n;
    }
};

struct TrainingContext {
    // All ranks create contexts in the same order. This sequence isolates
    // per-context rendezvous files when a worker reuses one NCCL process.
    int64_t context_sequence = 0;
    // Model weights (frozen, no grad) — pointers to external tensors
    std::vector<at::Tensor*> weight_ptrs;  // flat array, 15 or 18 per layer
    std::vector<at::Tensor*> embed_ptr;
    std::vector<at::Tensor*> final_norm_ptr;
    std::vector<at::Tensor*> lm_head_ptr;
    std::vector<LayerConfig> layer_configs;
    // Frozen GDN alpha/beta projections are tiny and launch-bound. Keep a
    // context-owned concatenation so the activation-level LoRA path computes
    // both with one GEMM without duplicating the large QKV/Z weights.
    std::vector<at::Tensor> fused_gdn_ab_weights;
    int64_t num_layers;

    // Attention mask [batch, seq] — 1 for real tokens, 0 for padding
    at::Tensor attention_mask;

    // ── Multi-LoRA adapter registry ──
    struct LoRAAdapter {
        int64_t id;
        int64_t rank;
        // Each tenant owns an independent Adam bias-correction clock.
        int64_t optimizer_step = 0;
        double alpha;
        std::set<int64_t> target_layers;
        std::set<std::string> target_modules;
        std::map<int64_t, std::vector<std::pair<at::Tensor, at::Tensor>>> params;
        std::map<int64_t, std::vector<std::array<at::Tensor, 4>>> adam_state;
        // Reusable out-of-place Adam destinations. Each entry stores
        // {param_a, m_a, v_a, param_b, m_b, v_b}; a successful transaction
        // swaps these handles with the live registry without allocating.
        std::map<int64_t, std::vector<std::array<at::Tensor, 6>>> adam_shadow;
        // Gradients are harvested after each backward into FP32 tensors. The
        // accumulator tensors are intentionally shared by value when chunk
        // registry guards copy adapters, so their contents survive restore.
        std::map<int64_t, std::vector<std::array<at::Tensor, 2>>> grad_accum;
        at::Tensor grad_slab;
    };

    std::vector<LoRAAdapter> adapters;
    int64_t next_adapter_id = 0;
    int64_t multi_lora_invocation = 0;
    int64_t dynamic_finalizer_count = 0;
    int64_t dynamic_adam_launch_count = 0;
    bool restore_without_parameter_sync = false;
    bool allow_heterogeneous_registration = false;

    // LoRA cache: pre-concatenated A/B per (layer, module) pair
    // Invalidated when adapters change or after Adam update
    bool lora_cache_valid = false;
    std::map<int64_t, at::Tensor> lora_cache;  // cached delta_weight per (layer*10+pair)

    // ── Batched Multi-LoRA (activation-level) ──
    // When active, replaces the weight-level lora_cache.
    // Stores per-(layer, module) stacked A/B tensors for batched B@(A@x) computation.
    bool lora_batch_valid = false;
    int64_t lora_batch_n = 0;  // number of adapters in current batch
    std::map<int64_t, LoraBatchEntry> lora_batch_cache;
    // Legacy single-LoRA (backward compat)
    std::vector<at::Tensor> lora_a;
    std::vector<at::Tensor> lora_b;
    // Fixed-slot FP32 gradient accumulators. Inactive slots are undefined
    // tensors to preserve the positional LoRA ABI without extra allocation.
    std::vector<at::Tensor> grad_accum_a;
    std::vector<at::Tensor> grad_accum_b;
    at::Tensor fixed_grad_slab;
    std::vector<uint8_t> lora_active;
    std::vector<int64_t> lora_layer_offset;
    double lora_scaling;
    std::vector<std::string> lora_names;

    // Adam optimizer state
    std::vector<at::Tensor> adam_m;
    std::vector<at::Tensor> adam_v;
    double lr, beta1, beta2, eps;
    // The fixed adapter's Adam bias-correction clock. Dynamic tenant updates
    // must not advance it because every tenant has its own optimizer_step.
    int64_t fixed_optimizer_step;
    // A failed native call aborts the in-flight gradient window. This flag is
    // consumed by the scoped accumulation guard on stack unwinding.
    bool accumulation_active = false;
    // Sum of (micro weight * supervised tokens) for the fixed-LoRA window.
    // Fixed gradients are accumulated as token-weighted numerators and
    // normalized exactly once at the optimizer boundary.
    double accumulated_token_weight = 0.0;

    // Device buffer cache for multi-tensor fused Adam
    AdamDevBuffers adam_dev_bufs;

    // Config
    at::ScalarType compute_type;
    int64_t vocab_size;
    double rms_eps;

    // MTP weights (optional)
    bool has_mtp;
    double mtp_loss_scale = 0.1;  // NVIDIA Megatron default
    at::Tensor *mtp_fc, *mtp_pre_fc_norm_emb, *mtp_pre_fc_norm_hidden, *mtp_norm;
    std::vector<at::Tensor*> mtp_layer_weights;
    std::vector<LayerConfig> mtp_layer_configs;

    // Gradient checkpointing
    bool use_checkpoint;
    int64_t group_size;
    // Group checkpoint storage for manual sequential backward
    std::vector<at::Tensor> group_inputs;
    std::vector<at::Tensor> group_outputs;
    // Variable-size group ranges for selective checkpointing
    std::vector<std::pair<int64_t, int64_t>> group_ranges;

    // Expert-parallel communicator. Layer dispatch/combine and dense replica
    // reductions use this axis; routed-expert parameters never reduce on it.
    ncclComm_t nccl_comm = nullptr;
    cudaStream_t nccl_stream = nullptr;
    int ep_world_size = 1;
    int ep_rank = 0;
    bool expert_parallel = false;
    // Expert-data-parallel communicator. Routed experts are replicated only
    // across this axis, while dense parameters reduce across both EP and DP.
    ncclComm_t dp_comm = nullptr;
    cudaStream_t dp_stream = nullptr;
    int dp_world_size = 1;
    int dp_rank = 0;
    bool data_parallel = false;
    // LoRA-only tensor parallelism keeps frozen base weights replicated and
    // shards the latent rank. Only the local LoRA delta uses this communicator.
    ncclComm_t tp_comm = nullptr;
    cudaStream_t tp_stream = nullptr;
    int tp_world_size = 1;
    int tp_rank = 0;
    // Frozen dense SwiGLU TP: gate/up are output-sharded and down is
    // input-sharded by the Rust weight loader.  The local row contribution
    // is reduced over the TP communicator before the residual add.
    bool base_tp_mlp = false;
    // Frozen attention TP: full attention and GDN own disjoint head bundles;
    // their output projections own the matching input columns.
    bool base_tp_attention = false;
    // Vocabulary parallelism shards embedding and LM-head rows over TP ranks.
    // Hidden states remain replicated; embedding outputs and CE hidden
    // gradients are summed over the TP communicator.
    bool vocab_parallel = false;
    int64_t local_vocab_size = 0;
    // Set when a legacy NCCL setter supplies an incompatible mixed topology.
    // Training entry points reject the context before touching parameters.
    bool topology_invalid = false;
    int cuda_device = 0;
// ──────────────────────────────────────────────────────────────────────
};

struct AdapterRegistryHash {
    uint64_t first = 1469598103934665603ULL;
    uint64_t second = 1099511628211ULL;

    void add_u64(uint64_t value) {
        first ^= value;
        first *= 1099511628211ULL;
        second ^= value + 0x9e3779b97f4a7c15ULL + (second << 6) +
            (second >> 2);
        second *= 0xbf58476d1ce4e5b9ULL;
    }

    void add_string(const std::string& value) {
        add_u64(value.size());
        for (const unsigned char byte : value) add_u64(byte);
    }
};

static void hash_tensor_layout(
    AdapterRegistryHash& hash, const at::Tensor& tensor
) {
    hash.add_u64(tensor.defined());
    if (!tensor.defined()) return;
    hash.add_u64(static_cast<uint64_t>(tensor.scalar_type()));
    hash.add_u64(tensor.requires_grad());
    hash.add_u64(tensor.dim());
    for (const auto size : tensor.sizes()) hash.add_u64(size);
}

static void hash_adapter_layout(
    AdapterRegistryHash& hash,
    const TrainingContext::LoRAAdapter& adapter
) {
    hash.add_u64(adapter.id);
    hash.add_u64(adapter.rank);
    hash.add_u64(adapter.optimizer_step);
    uint64_t alpha_bits = 0;
    static_assert(sizeof(alpha_bits) == sizeof(adapter.alpha));
    std::memcpy(&alpha_bits, &adapter.alpha, sizeof(alpha_bits));
    hash.add_u64(alpha_bits);
    hash.add_u64(adapter.target_layers.size());
    for (const auto layer : adapter.target_layers) hash.add_u64(layer);
    hash.add_u64(adapter.target_modules.size());
    for (const auto& module : adapter.target_modules) hash.add_string(module);
    hash.add_u64(adapter.params.size());
    for (const auto& [layer, pairs] : adapter.params) {
        hash.add_u64(layer);
        hash.add_u64(pairs.size());
        for (const auto& [a, b] : pairs) {
            hash_tensor_layout(hash, a);
            hash_tensor_layout(hash, b);
        }
    }
    hash.add_u64(adapter.grad_accum.size());
    for (const auto& [layer, pairs] : adapter.grad_accum) {
        hash.add_u64(layer);
        hash.add_u64(pairs.size());
        for (const auto& pair : pairs) {
            hash_tensor_layout(hash, pair[0]);
            hash_tensor_layout(hash, pair[1]);
        }
    }
    hash_tensor_layout(hash, adapter.grad_slab);
}

static void hash_collective_topology(
    AdapterRegistryHash& hash, const TrainingContext* ctx
) {
    hash.add_u64(ctx->tp_world_size);
    hash.add_u64(ctx->ep_world_size);
    hash.add_u64(ctx->dp_world_size);
    hash.add_u64(ctx->base_tp_attention);
    hash.add_u64(ctx->base_tp_mlp);
    hash.add_u64(ctx->vocab_parallel);
    hash.add_u64(ctx->expert_parallel);
    hash.add_u64(ctx->data_parallel);
    hash.add_u64(env_enabled("QWEN36_EP_A2A"));
    hash.add_u64(env_enabled("QWEN36_EP_A2A_SHARDED"));
    hash.add_u64(env_enabled("QWEN36_GROUPED_LORA_SYNC", true));
    hash.add_u64(env_enabled("QWEN36_GRAD_SLAB", true));
}

static void validate_adapter_collective_registry(
    TrainingContext* ctx,
    const int64_t* requested_ids,
    int64_t requested_count,
    int64_t requested_rank,
    bool use_registered_order
) {
    if (!ctx || (!ctx->nccl_comm && !ctx->dp_comm && !ctx->tp_comm)) return;

    AdapterRegistryHash hash;
    hash_collective_topology(hash, ctx);
    hash.add_u64(requested_count);
    hash.add_u64(requested_rank);
    hash.add_u64(use_registered_order);
    int64_t found_count = 0;
    if (use_registered_order) {
        hash.add_u64(ctx->adapters.size());
        for (const auto& adapter : ctx->adapters) {
            hash_adapter_layout(hash, adapter);
            ++found_count;
        }
    } else if (requested_ids && requested_count > 0) {
        for (int64_t index = 0; index < requested_count; ++index) {
            const int64_t requested_id = requested_ids[index];
            hash.add_u64(requested_id);
            const auto it = std::find_if(
                ctx->adapters.begin(), ctx->adapters.end(),
                [&](const auto& adapter) { return adapter.id == requested_id; });
            if (it == ctx->adapters.end()) {
                hash.add_u64(0x6d697373696e67ULL);
                continue;
            }
            hash.add_u64(0x666f756e64ULL);
            hash_adapter_layout(hash, *it);
            ++found_count;
        }
    } else {
        hash.add_u64(0x6e756c6cULL);
    }

    constexpr uint64_t kPositiveInt64Mask =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    const std::vector<int64_t> signature_values{
        requested_count,
        requested_rank,
        found_count,
        static_cast<int64_t>(hash.first & kPositiveInt64Mask),
        static_cast<int64_t>(hash.second & kPositiveInt64Mask),
    };
    c10::cuda::set_device(ctx->cuda_device);
    cudaSetDevice(ctx->cuda_device);
    auto options = at::TensorOptions().dtype(at::kLong).device(
        at::kCUDA, ctx->cuda_device);
    auto minimum = at::tensor(signature_values, options);
    auto maximum = minimum.clone();
    auto stream = c10::cuda::getCurrentCUDAStream(ctx->cuda_device).stream();

    auto reduce_axis = [&](ncclComm_t communicator, const char* axis) {
        if (!communicator) return;
        auto err = ncclAllReduce(
            minimum.data_ptr<int64_t>(), minimum.data_ptr<int64_t>(),
            minimum.numel(), ncclInt64, ncclMin, communicator, stream);
        TORCH_CHECK(err == ncclSuccess, axis,
            " adapter registry minimum all-reduce failed: ",
            ncclGetErrorString(err));
        err = ncclAllReduce(
            maximum.data_ptr<int64_t>(), maximum.data_ptr<int64_t>(),
            maximum.numel(), ncclInt64, ncclMax, communicator, stream);
        TORCH_CHECK(err == ncclSuccess, axis,
            " adapter registry maximum all-reduce failed: ",
            ncclGetErrorString(err));
    };

    // Sequential reductions over the orthogonal axes propagate the extrema
    // over the complete TP x EP x DP grid.
    reduce_axis(ctx->nccl_comm, "EP");
    reduce_axis(ctx->dp_comm, "DP");
    reduce_axis(ctx->tp_comm, "TP");
    const auto minimum_cpu = minimum.to(at::kCPU);
    const auto maximum_cpu = maximum.to(at::kCPU);
    const auto* minimum_data = minimum_cpu.data_ptr<int64_t>();
    const auto* maximum_data = maximum_cpu.data_ptr<int64_t>();
    for (int64_t index = 0; index < minimum.numel(); ++index) {
        TORCH_CHECK(minimum_data[index] == maximum_data[index],
            "dynamic LoRA adapter registry mismatch across distributed ranks; "
            "all ranks must select the same ordered IDs, ranks, optimizer "
            "clocks, targets, and tensor layouts");
    }
}

static void validate_fixed_collective_registry(
    TrainingContext* ctx, int64_t optimizer_phase = -1
) {
    if (!ctx || (!ctx->nccl_comm && !ctx->dp_comm && !ctx->tp_comm)) return;

    AdapterRegistryHash hash;
    hash_collective_topology(hash, ctx);
    hash.add_u64(ctx->fixed_optimizer_step);
    hash.add_u64(ctx->lora_active.size());
    hash.add_u64(ctx->lora_a.size());
    hash.add_u64(ctx->lora_b.size());
    hash.add_u64(ctx->adapters.size());
    for (const auto& adapter : ctx->adapters)
        hash_adapter_layout(hash, adapter);
    for (size_t index = 0; index < ctx->lora_active.size(); ++index) {
        hash.add_u64(ctx->lora_active[index]);
        if (index < ctx->lora_a.size())
            hash_tensor_layout(hash, ctx->lora_a[index]);
        if (index < ctx->lora_b.size())
            hash_tensor_layout(hash, ctx->lora_b[index]);
        if (index < ctx->grad_accum_a.size())
            hash_tensor_layout(hash, ctx->grad_accum_a[index]);
        if (index < ctx->grad_accum_b.size())
            hash_tensor_layout(hash, ctx->grad_accum_b[index]);
    }
    hash_tensor_layout(hash, ctx->fixed_grad_slab);

    constexpr uint64_t kPositiveInt64Mask =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    const std::vector<int64_t> signature_values{
        optimizer_phase,
        ctx->fixed_optimizer_step,
        static_cast<int64_t>(ctx->lora_active.size()),
        static_cast<int64_t>(hash.first & kPositiveInt64Mask),
        static_cast<int64_t>(hash.second & kPositiveInt64Mask),
    };
    c10::cuda::set_device(ctx->cuda_device);
    cudaSetDevice(ctx->cuda_device);
    auto options = at::TensorOptions().dtype(at::kLong).device(
        at::kCUDA, ctx->cuda_device);
    auto minimum = at::tensor(signature_values, options);
    auto maximum = minimum.clone();
    auto stream = c10::cuda::getCurrentCUDAStream(ctx->cuda_device).stream();
    auto reduce_axis = [&](ncclComm_t communicator, const char* axis) {
        if (!communicator) return;
        auto err = ncclAllReduce(
            minimum.data_ptr<int64_t>(), minimum.data_ptr<int64_t>(),
            minimum.numel(), ncclInt64, ncclMin, communicator, stream);
        TORCH_CHECK(err == ncclSuccess, axis,
            " fixed LoRA registry minimum all-reduce failed: ",
            ncclGetErrorString(err));
        err = ncclAllReduce(
            maximum.data_ptr<int64_t>(), maximum.data_ptr<int64_t>(),
            maximum.numel(), ncclInt64, ncclMax, communicator, stream);
        TORCH_CHECK(err == ncclSuccess, axis,
            " fixed LoRA registry maximum all-reduce failed: ",
            ncclGetErrorString(err));
    };
    reduce_axis(ctx->nccl_comm, "EP");
    reduce_axis(ctx->dp_comm, "DP");
    reduce_axis(ctx->tp_comm, "TP");

    const auto minimum_cpu = minimum.to(at::kCPU);
    const auto maximum_cpu = maximum.to(at::kCPU);
    const auto* minimum_data = minimum_cpu.data_ptr<int64_t>();
    const auto* maximum_data = maximum_cpu.data_ptr<int64_t>();
    for (int64_t index = 0; index < minimum.numel(); ++index) {
        TORCH_CHECK(minimum_data[index] == maximum_data[index],
            "fixed LoRA registry mismatch across distributed ranks; all "
            "ranks must use the same active slots, tensor layouts, and "
            "optimizer phase/clock");
    }
}

static bool adapter_collective_all_succeeded(
    TrainingContext* ctx,
    bool local_success
) {
    if (!ctx || (!ctx->nccl_comm && !ctx->dp_comm && !ctx->tp_comm))
        return local_success;
    c10::cuda::set_device(ctx->cuda_device);
    cudaSetDevice(ctx->cuda_device);
    auto options = at::TensorOptions().dtype(at::kInt).device(
        at::kCUDA, ctx->cuda_device);
    auto succeeded = at::full({1}, local_success ? 1 : 0, options);
    auto stream = c10::cuda::getCurrentCUDAStream(ctx->cuda_device).stream();
    auto reduce_axis = [&](ncclComm_t communicator, const char* axis) {
        if (!communicator) return;
        const auto err = ncclAllReduce(
            succeeded.data_ptr<int32_t>(), succeeded.data_ptr<int32_t>(),
            1, ncclInt32, ncclMin, communicator, stream);
        TORCH_CHECK(err == ncclSuccess, axis,
            " adapter success consensus failed: ", ncclGetErrorString(err));
    };
    reduce_axis(ctx->nccl_comm, "EP");
    reduce_axis(ctx->dp_comm, "DP");
    reduce_axis(ctx->tp_comm, "TP");
    return succeeded.to(at::kCPU).item<int32_t>() != 0;
}

static void require_clear_accumulation_for_registry_mutation(
    TrainingContext* ctx
) {
    const bool local_clear = ctx && !ctx->accumulation_active &&
        ctx->accumulated_token_weight == 0.0;
    TORCH_CHECK(local_clear,
        "cannot mutate the dynamic LoRA registry while a gradient "
        "accumulation window is pending; finalize or abort it first");
}

static bool adapter_registration_phase_matches(
    TrainingContext* ctx,
    const TrainingContext::LoRAAdapter& candidate,
    bool local_ready,
    uint64_t phase
) {
    if (!ctx || (!ctx->nccl_comm && !ctx->dp_comm && !ctx->tp_comm))
        return local_ready;

    AdapterRegistryHash hash;
    hash_collective_topology(hash, ctx);
    hash.add_u64(phase);
    hash.add_u64(ctx->restore_without_parameter_sync);
    hash.add_u64(ctx->allow_heterogeneous_registration);
    hash.add_u64(ctx->next_adapter_id);
    hash.add_u64(ctx->adapters.size());
    for (const auto& adapter : ctx->adapters)
        hash_adapter_layout(hash, adapter);
    hash_adapter_layout(hash, candidate);

    constexpr uint64_t kPositiveInt64Mask =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    const std::vector<int64_t> signature_values{
        local_ready ? 1 : 0,
        static_cast<int64_t>(hash.first & kPositiveInt64Mask),
        static_cast<int64_t>(hash.second & kPositiveInt64Mask),
    };
    c10::cuda::set_device(ctx->cuda_device);
    cudaSetDevice(ctx->cuda_device);
    auto options = at::TensorOptions().dtype(at::kLong).device(
        at::kCUDA, ctx->cuda_device);
    auto minimum = at::tensor(signature_values, options);
    auto maximum = minimum.clone();
    auto stream = c10::cuda::getCurrentCUDAStream(ctx->cuda_device).stream();
    auto reduce_axis = [&](ncclComm_t communicator, const char* axis) {
        if (!communicator) return;
        auto err = ncclAllReduce(
            minimum.data_ptr<int64_t>(), minimum.data_ptr<int64_t>(),
            minimum.numel(), ncclInt64, ncclMin, communicator, stream);
        TORCH_CHECK(err == ncclSuccess, axis,
            " adapter registration minimum all-reduce failed: ",
            ncclGetErrorString(err));
        err = ncclAllReduce(
            maximum.data_ptr<int64_t>(), maximum.data_ptr<int64_t>(),
            maximum.numel(), ncclInt64, ncclMax, communicator, stream);
        TORCH_CHECK(err == ncclSuccess, axis,
            " adapter registration maximum all-reduce failed: ",
            ncclGetErrorString(err));
    };
    reduce_axis(ctx->nccl_comm, "EP");
    reduce_axis(ctx->dp_comm, "DP");
    reduce_axis(ctx->tp_comm, "TP");

    const auto minimum_cpu = minimum.to(at::kCPU);
    const auto maximum_cpu = maximum.to(at::kCPU);
    const auto* minimum_data = minimum_cpu.data_ptr<int64_t>();
    const auto* maximum_data = maximum_cpu.data_ptr<int64_t>();
    for (int64_t index = 0; index < minimum.numel(); ++index) {
        if (minimum_data[index] != maximum_data[index]) return false;
    }
    return minimum_data[0] == 1;
}

static bool supported_mask_dtype(const at::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case at::kBool:
        case at::kByte:
        case at::kChar:
        case at::kShort:
        case at::kInt:
        case at::kLong:
        case at::kHalf:
        case at::kBFloat16:
        case at::kFloat:
        case at::kDouble:
            return true;
        default:
            return false;
    }
}

static bool replica_input_signatures_match(
    TrainingContext* ctx,
    const at::Tensor& input_ids,
    const at::Tensor& target_mask,
    const at::Tensor* attention_mask
) {
    if (!ctx || (!ctx->tp_comm && !ctx->nccl_comm)) return true;
    const std::vector<int64_t> signature_values{
        input_ids.size(0),
        input_ids.size(1),
        static_cast<int64_t>(input_ids.scalar_type()),
        target_mask.size(0),
        target_mask.size(1),
        static_cast<int64_t>(target_mask.scalar_type()),
        attention_mask ? 1 : 0,
        attention_mask ? attention_mask->size(0) : 0,
        attention_mask ? attention_mask->size(1) : 0,
        attention_mask
            ? static_cast<int64_t>(attention_mask->scalar_type())
            : 0,
    };
    auto options = at::TensorOptions().dtype(at::kLong).device(
        at::kCUDA, ctx->cuda_device);
    auto minimum = at::tensor(signature_values, options);
    auto maximum = minimum.clone();
    auto stream = c10::cuda::getCurrentCUDAStream(ctx->cuda_device).stream();
    auto reduce_axis = [&](ncclComm_t communicator, const char* axis) {
        if (!communicator) return;
        auto err = ncclAllReduce(
            minimum.data_ptr<int64_t>(), minimum.data_ptr<int64_t>(),
            minimum.numel(), ncclInt64, ncclMin, communicator, stream);
        TORCH_CHECK(err == ncclSuccess, axis,
            " input signature minimum all-reduce failed: ",
            ncclGetErrorString(err));
        err = ncclAllReduce(
            maximum.data_ptr<int64_t>(), maximum.data_ptr<int64_t>(),
            maximum.numel(), ncclInt64, ncclMax, communicator, stream);
        TORCH_CHECK(err == ncclSuccess, axis,
            " input signature maximum all-reduce failed: ",
            ncclGetErrorString(err));
    };
    reduce_axis(ctx->tp_comm, "TP");
    const bool sharded_a2a = ctx->expert_parallel && ctx->nccl_comm &&
        env_enabled("QWEN36_EP_A2A_SHARDED");
    if (ctx->expert_parallel && !sharded_a2a)
        reduce_axis(ctx->nccl_comm, "replicated EP");
    return (minimum == maximum).all().item<bool>();
}

static bool harvest_leaf_grad(
    at::Tensor& param, at::Tensor& fp32_accumulator
) {
    auto grad = param.grad();
    if (!grad.defined()) return false;
    TORCH_CHECK(fp32_accumulator.defined(),
        "active LoRA parameter is missing its FP32 gradient accumulator");
    TORCH_CHECK(fp32_accumulator.scalar_type() == at::kFloat,
        "LoRA gradient accumulator must be FP32");
    TORCH_CHECK(fp32_accumulator.sizes() == param.sizes(),
        "LoRA gradient accumulator shape mismatch");
    at::NoGradGuard guard;
    fp32_accumulator.add_(grad.to(at::kFloat));
    // Never retain BF16 leaf gradients across micro-batches. The next
    // backward starts with a fresh leaf grad and is harvested again.
    param.mutable_grad() = at::Tensor();
    return true;
}

static bool harvest_adapter_gradients(TrainingContext::LoRAAdapter& adapter) {
    bool harvested = false;
    for (auto& [layer_idx, pairs] : adapter.params) {
        auto accum_it = adapter.grad_accum.find(layer_idx);
        TORCH_CHECK(accum_it != adapter.grad_accum.end() &&
                    accum_it->second.size() == pairs.size(),
            "dynamic LoRA gradient accumulator layout mismatch");
        for (size_t i = 0; i < pairs.size(); ++i) {
            auto& [a, b] = pairs[i];
            auto& accum = accum_it->second[i];
            if (a.requires_grad()) harvested |= harvest_leaf_grad(a, accum[0]);
            if (b.requires_grad()) harvested |= harvest_leaf_grad(b, accum[1]);
        }
    }
    return harvested;
}

static bool harvest_gradient_accumulators(TrainingContext* ctx) {
    bool harvested = false;
    for (auto& adapter : ctx->adapters) {
        harvested |= harvest_adapter_gradients(adapter);
    }
    TORCH_CHECK(ctx->grad_accum_a.size() == ctx->lora_a.size() &&
                ctx->grad_accum_b.size() == ctx->lora_b.size(),
        "fixed LoRA gradient accumulator layout mismatch");
    for (size_t i = 0; i < ctx->lora_a.size(); ++i) {
        if (!ctx->lora_active[i]) continue;
        harvested |= harvest_leaf_grad(ctx->lora_a[i], ctx->grad_accum_a[i]);
        harvested |= harvest_leaf_grad(ctx->lora_b[i], ctx->grad_accum_b[i]);
    }
    ctx->accumulation_active |= harvested;
    return harvested;
}

static void clear_adapter_gradient_accumulators(
    TrainingContext::LoRAAdapter& adapter
) {
    at::NoGradGuard guard;
    if (adapter.grad_slab.defined()) adapter.grad_slab.zero_();
    for (auto& [layer_idx, pairs] : adapter.params) {
        auto accum_it = adapter.grad_accum.find(layer_idx);
        for (size_t i = 0; i < pairs.size(); ++i) {
            auto& [a, b] = pairs[i];
            if (a.grad().defined()) a.mutable_grad() = at::Tensor();
            if (b.grad().defined()) b.mutable_grad() = at::Tensor();
            if (accum_it != adapter.grad_accum.end() &&
                i < accum_it->second.size() && !adapter.grad_slab.defined()) {
                if (accum_it->second[i][0].defined()) accum_it->second[i][0].zero_();
                if (accum_it->second[i][1].defined()) accum_it->second[i][1].zero_();
            }
        }
    }
}

static void clear_gradient_accumulators(TrainingContext* ctx) {
    if (!ctx) return;
    at::NoGradGuard guard;
    for (auto& adapter : ctx->adapters) {
        clear_adapter_gradient_accumulators(adapter);
    }
    if (ctx->fixed_grad_slab.defined()) ctx->fixed_grad_slab.zero_();
    for (size_t i = 0; i < ctx->lora_a.size(); ++i) {
        if (ctx->lora_a[i].grad().defined())
            ctx->lora_a[i].mutable_grad() = at::Tensor();
        if (ctx->lora_b[i].grad().defined())
            ctx->lora_b[i].mutable_grad() = at::Tensor();
        if (!ctx->fixed_grad_slab.defined() &&
            i < ctx->grad_accum_a.size() && ctx->grad_accum_a[i].defined())
            ctx->grad_accum_a[i].zero_();
        if (!ctx->fixed_grad_slab.defined() &&
            i < ctx->grad_accum_b.size() && ctx->grad_accum_b[i].defined())
            ctx->grad_accum_b[i].zero_();
    }
    ctx->group_inputs.clear();
    ctx->group_outputs.clear();
    ctx->lora_cache_valid = false;
    ctx->lora_batch_valid = false;
    ctx->accumulation_active = false;
    ctx->accumulated_token_weight = 0.0;
}

struct GradientAccumulationFailureGuard {
    TrainingContext* ctx;
    bool disarmed = false;
    ~GradientAccumulationFailureGuard() noexcept {
        if (disarmed) return;
        try {
            clear_gradient_accumulators(ctx);
        } catch (const std::exception& e) {
            fprintf(stderr,
                "[q36] secondary gradient cleanup failure: %s\n", e.what());
        } catch (...) {
            fprintf(stderr,
                "[q36] secondary gradient cleanup failure: unknown error\n");
        }
    }
};

static at::Tensor tp_allreduce_lora_delta(
    TrainingContext* ctx, const at::Tensor& local_delta
) {
    if (!ctx || ctx->tp_world_size <= 1) return local_delta;
    TORCH_CHECK(ctx->tp_comm,
        "LoRA TP communicator is not initialized for TP_SIZE=",
        ctx->tp_world_size);
    return NcclAllReduceFunction::apply(
        local_delta, (int64_t)ctx->tp_comm,
        (int64_t)reinterpret_cast<uintptr_t>(ctx->tp_stream));
}

static at::Tensor tp_copy_lora_input(
    TrainingContext* ctx, const at::Tensor& input
) {
    if (!ctx || ctx->tp_world_size <= 1) return input;
    TORCH_CHECK(ctx->tp_comm,
        "LoRA TP communicator is not initialized for TP_SIZE=",
        ctx->tp_world_size);
    return TpCopyToRegionFunction::apply(
        input, (int64_t)ctx->tp_comm,
        (int64_t)reinterpret_cast<uintptr_t>(ctx->tp_stream));
}

static bool base_tp_attention_enabled(const TrainingContext* ctx) {
    return ctx && ctx->base_tp_attention && ctx->tp_world_size > 1;
}

static bool vocab_parallel_enabled(const TrainingContext* ctx) {
    return ctx && ctx->vocab_parallel && ctx->tp_world_size > 1;
}

static at::Tensor tp_allreduce_value(
    TrainingContext* ctx, const at::Tensor& input, ncclRedOp_t reduction
) {
    TORCH_CHECK(ctx && ctx->tp_comm,
        "vocabulary TP communicator is not initialized for TP_SIZE=",
        ctx ? ctx->tp_world_size : 0);
    return NcclAllReduceFunction::allreduce(
        input, ctx->tp_comm, ctx->tp_stream, reduction);
}

static at::Tensor vocabulary_embedding(
    TrainingContext* ctx, const at::Tensor& input_ids
) {
    auto embed = *ctx->embed_ptr[0];
    if (!vocab_parallel_enabled(ctx)) return at::embedding(embed, input_ids);

    const int64_t vocab_start = ctx->tp_rank * ctx->local_vocab_size;
    const int64_t vocab_end = vocab_start + ctx->local_vocab_size;
    auto in_range = (input_ids >= vocab_start) & (input_ids < vocab_end);
    auto local_ids = (input_ids - vocab_start).clamp(0, ctx->local_vocab_size - 1);
    auto local_hidden = at::embedding(embed, local_ids);
    local_hidden = local_hidden * in_range.unsqueeze(-1).to(local_hidden.scalar_type());
    return tp_allreduce_value(ctx, local_hidden, ncclSum);
}

static bool base_tp_mlp_enabled(const TrainingContext* ctx) {
    return ctx && ctx->base_tp_mlp && ctx->tp_world_size > 1;
}

static int64_t base_tp_mlp_world_size(const TrainingContext* ctx) {
    TORCH_CHECK(base_tp_mlp_enabled(ctx),
        "base MLP TP context is not enabled");
    return ctx->tp_world_size;
}

static at::Tensor tp_allreduce_base_mlp(
    TrainingContext* ctx, const at::Tensor& local_output
) {
    if (!ctx || !ctx->base_tp_mlp || ctx->tp_world_size <= 1)
        return local_output;
    TORCH_CHECK(ctx->tp_comm,
        "base MLP TP communicator is not initialized for TP_SIZE=",
        ctx->tp_world_size);
    return NcclAllReduceFunction::apply(
        local_output, (int64_t)ctx->tp_comm,
        (int64_t)reinterpret_cast<uintptr_t>(ctx->tp_stream));
}

static at::Tensor tp_copy_base_mlp_input(
    TrainingContext* ctx, const at::Tensor& input
) {
    if (!ctx || !ctx->base_tp_mlp || ctx->tp_world_size <= 1)
        return input;
    TORCH_CHECK(ctx->tp_comm,
        "base MLP TP communicator is not initialized for TP_SIZE=",
        ctx->tp_world_size);
    return TpCopyToRegionFunction::apply(
        input, (int64_t)ctx->tp_comm,
        (int64_t)reinterpret_cast<uintptr_t>(ctx->tp_stream));
}

static at::Tensor tp_allreduce_base_attention(
    TrainingContext* ctx, const at::Tensor& local_output
) {
    if (!ctx || !ctx->base_tp_attention || ctx->tp_world_size <= 1)
        return local_output;
    TORCH_CHECK(ctx->tp_comm,
        "base attention TP communicator is not initialized for TP_SIZE=",
        ctx->tp_world_size);
    return NcclAllReduceFunction::apply(
        local_output, (int64_t)ctx->tp_comm,
        (int64_t)reinterpret_cast<uintptr_t>(ctx->tp_stream));
}

static at::Tensor tp_copy_base_attention_input(
    TrainingContext* ctx, const at::Tensor& input
) {
    if (!ctx || !ctx->base_tp_attention || ctx->tp_world_size <= 1)
        return input;
    TORCH_CHECK(ctx->tp_comm,
        "base attention TP communicator is not initialized for TP_SIZE=",
        ctx->tp_world_size);
    return TpCopyToRegionFunction::apply(
        input, (int64_t)ctx->tp_comm,
        (int64_t)reinterpret_cast<uintptr_t>(ctx->tp_stream));
}

static LoraTpLayout lora_tp_layout(
    const TrainingContext* ctx, int64_t layer_idx, int64_t pair_idx
) {
    if (!ctx || ctx->tp_world_size <= 1 ||
        layer_idx < 0 || layer_idx >= ctx->num_layers)
        return LoraTpLayout::LatentRank;
    const auto& cfg = ctx->layer_configs[layer_idx];
    auto table = lora_projection_table(cfg);
    TORCH_CHECK(pair_idx >= 0 && pair_idx < table.count,
        "invalid LoRA pair for TP layout");
    const std::string name(table.entries[pair_idx].name);
    if (ctx->base_tp_attention &&
        (name == "q_proj" || name == "k_proj" || name == "v_proj" ||
         name == "in_proj_qkv" || name == "in_proj_z" ||
         name == "in_proj_a" || name == "in_proj_b"))
        return LoraTpLayout::ColumnParallel;
    if (ctx->base_tp_attention &&
        (name == "o_proj" || name == "out_proj"))
        return LoraTpLayout::RowParallel;
    if (ctx->base_tp_mlp &&
        (name == "gate_proj" || name == "up_proj" ||
         name == "shared_gate_proj" || name == "shared_up_proj" ||
         name == "experts_gate_up_proj"))
        return LoraTpLayout::ColumnParallel;
    if (ctx->base_tp_mlp &&
        (name == "down_proj" || name == "shared_down_proj" ||
         name == "experts_down_proj"))
        return LoraTpLayout::RowParallel;
    return LoraTpLayout::LatentRank;
}

struct LoraGradientSlabBinding {
    at::Tensor* accumulator = nullptr;
    const at::Tensor* parameter = nullptr;
    uint8_t bucket_key = 0;
    int64_t offset = 0;
};

static uint8_t lora_gradient_bucket_key(
    const TrainingContext* ctx, int64_t layer, int64_t pair, bool is_a
) {
    const auto table = lora_projection_table(ctx->layer_configs[layer]);
    const bool grouped_expert = table.entries[pair].grouped_expert;
    const auto layout = lora_tp_layout(ctx, layer, pair);
    const bool tp_replicated =
        (layout == LoraTpLayout::ColumnParallel && is_a) ||
        (layout == LoraTpLayout::RowParallel && !is_a);
    return (grouped_expert ? 2 : 0) | (tp_replicated ? 1 : 0);
}

static void bind_lora_gradient_slab(
    std::vector<LoraGradientSlabBinding>& bindings,
    at::Tensor& slab
) {
    if (bindings.empty()) {
        slab = at::Tensor();
        return;
    }
    constexpr int64_t alignment = 64;
    int64_t total = 0;
    for (uint8_t key = 0; key < 4; ++key) {
        total = ((total + alignment - 1) / alignment) * alignment;
        for (auto& binding : bindings) {
            if (binding.bucket_key != key) continue;
            binding.offset = total;
            total += binding.parameter->numel();
        }
    }
    slab = at::zeros({total}, at::TensorOptions()
        .dtype(at::kFloat).device(bindings.front().parameter->device()));
    for (auto& binding : bindings) {
        *binding.accumulator = slab.narrow(
            0, binding.offset, binding.parameter->numel())
            .view(binding.parameter->sizes());
    }
}

static void bind_fixed_lora_gradient_slab(TrainingContext* ctx) {
    if (!env_enabled("QWEN36_GRAD_SLAB", true)) {
        for (size_t index = 0; index < ctx->lora_a.size(); ++index) {
            if (!ctx->lora_active[index]) continue;
            const auto options = at::TensorOptions().dtype(at::kFloat)
                .device(ctx->lora_a[index].device());
            ctx->grad_accum_a[index] = at::zeros(
                ctx->lora_a[index].sizes(), options);
            ctx->grad_accum_b[index] = at::zeros(
                ctx->lora_b[index].sizes(), options);
        }
        return;
    }
    std::vector<LoraGradientSlabBinding> bindings;
    for (int64_t layer = 0; layer < ctx->num_layers; ++layer) {
        const int64_t offset = ctx->lora_layer_offset[layer];
        const int64_t count = lora_pair_count(ctx->layer_configs[layer]);
        for (int64_t pair = 0; pair < count; ++pair) {
            const int64_t index = offset + pair;
            if (!ctx->lora_active[index]) continue;
            bindings.push_back({
                &ctx->grad_accum_a[index], &ctx->lora_a[index],
                lora_gradient_bucket_key(ctx, layer, pair, true), 0});
            bindings.push_back({
                &ctx->grad_accum_b[index], &ctx->lora_b[index],
                lora_gradient_bucket_key(ctx, layer, pair, false), 0});
        }
    }
    bind_lora_gradient_slab(bindings, ctx->fixed_grad_slab);
}

static void bind_adapter_lora_gradient_slab(
    TrainingContext* ctx, TrainingContext::LoRAAdapter& adapter
) {
    if (!env_enabled("QWEN36_GRAD_SLAB", true)) {
        for (auto& [layer, pairs] : adapter.params) {
            auto& accumulators = adapter.grad_accum.at(layer);
            for (size_t pair = 0; pair < pairs.size(); ++pair) {
                auto& [a, b] = pairs[pair];
                if (!a.requires_grad()) continue;
                const auto options = at::TensorOptions().dtype(at::kFloat)
                    .device(a.device());
                accumulators[pair][0] = at::zeros(a.sizes(), options);
                accumulators[pair][1] = at::zeros(b.sizes(), options);
            }
        }
        return;
    }
    std::vector<LoraGradientSlabBinding> bindings;
    for (auto& [layer, pairs] : adapter.params) {
        auto accum_it = adapter.grad_accum.find(layer);
        TORCH_CHECK(accum_it != adapter.grad_accum.end() &&
                accum_it->second.size() == pairs.size(),
            "dynamic LoRA gradient slab layout mismatch");
        for (int64_t pair = 0; pair < static_cast<int64_t>(pairs.size()); ++pair) {
            auto& [a, b] = pairs[pair];
            if (!a.requires_grad()) continue;
            bindings.push_back({
                &accum_it->second[pair][0], &a,
                lora_gradient_bucket_key(ctx, layer, pair, true), 0});
            bindings.push_back({
                &accum_it->second[pair][1], &b,
                lora_gradient_bucket_key(ctx, layer, pair, false), 0});
        }
    }
    bind_lora_gradient_slab(bindings, adapter.grad_slab);
}

static bool active_lora_targets_use_latent_rank_layout(
    const TrainingContext* ctx,
    const std::set<int64_t>& target_layers,
    const std::set<std::string>& target_modules,
    bool empty_modules_mean_attention_only
) {
    for (int64_t layer = 0; layer < ctx->num_layers; ++layer) {
        if (!target_layers.empty() && target_layers.count(layer) == 0) continue;
        const auto table = lora_projection_table(ctx->layer_configs[layer]);
        for (int64_t pair = 0; pair < table.count; ++pair) {
            const auto& projection = table.entries[pair];
            const bool active = target_modules.empty()
                ? !empty_modules_mean_attention_only ||
                    (!projection.grouped_expert &&
                     projection.segment == LoraSegment::Attention)
                : target_modules.count(projection.name) > 0;
            if (active &&
                lora_tp_layout(ctx, layer, pair) == LoraTpLayout::LatentRank)
                return true;
        }
    }
    return false;
}

static int64_t local_lora_rank_for_active_targets(
    const TrainingContext* ctx,
    int64_t global_rank,
    const std::set<int64_t>& target_layers,
    const std::set<std::string>& target_modules,
    bool empty_modules_mean_attention_only,
    const char* adapter_kind
) {
    TORCH_CHECK(global_rank > 0, adapter_kind, " LoRA rank must be positive");
    const bool uses_latent_rank = active_lora_targets_use_latent_rank_layout(
        ctx, target_layers, target_modules, empty_modules_mean_attention_only);
    TORCH_CHECK(!uses_latent_rank || global_rank % ctx->tp_world_size == 0,
        adapter_kind, " LoRA rank ", global_rank,
        " must be divisible by TP_SIZE=", ctx->tp_world_size,
        " because at least one active projection uses latent-rank sharding");
    return uses_latent_rank ? global_rank / ctx->tp_world_size : global_rank;
}

static at::Tensor initialize_lora_a(
    TrainingContext* ctx, const at::TensorOptions& options,
    int64_t experts, int64_t global_rank, int64_t in_features
) {
    const int64_t local_rank = global_rank / ctx->tp_world_size;
    const int64_t rank_start = ctx->tp_rank * local_rank;
    if (experts > 0) {
        auto global = at::randn({experts, global_rank, in_features}, options);
        return global.narrow(1, rank_start, local_rank).contiguous() * 0.01;
    }
    auto global = at::randn({global_rank, in_features}, options);
    return global.narrow(0, rank_start, local_rank).contiguous() * 0.01;
}

static const char* lora_pair_name(const LayerConfig& cfg, int64_t pair_idx) {
    auto table = lora_projection_table(cfg);
    TORCH_CHECK(pair_idx >= 0 && pair_idx < table.count, "invalid LoRA projection index");
    return table.entries[pair_idx].name;
}

static constexpr int64_t LORA_CACHE_STRIDE = 32;

static inline int64_t lora_cache_key(int64_t layer_idx, int64_t pair_idx) {
    return layer_idx * LORA_CACHE_STRIDE + pair_idx;
}

static inline bool legacy_lora_slot_active(const TrainingContext* ctx, int64_t slot) {
    return slot >= 0 && slot < (int64_t)ctx->lora_active.size() && ctx->lora_active[slot] != 0;
}

static RoutedExpertLora routed_expert_lora(
    TrainingContext* ctx, int64_t layer_idx, const LayerConfig& cfg
) {
    RoutedExpertLora result;
    // Dynamic multi-LoRA supplies per-sample activation-level tensors. Do not
    // mix the fixed adapter's expert tensors into those batches.
    if (!ctx->adapters.empty() || cfg.num_experts <= 0) return result;

    const int64_t offset = ctx->lora_layer_offset[layer_idx];
    const int64_t gate_up_pair = lora_pair_index(cfg, "experts_gate_up_proj");
    const int64_t down_pair = lora_pair_index(cfg, "experts_down_proj");
    if (gate_up_pair >= 0 && legacy_lora_slot_active(ctx, offset + gate_up_pair)) {
        result.gate_up_a = &ctx->lora_a[offset + gate_up_pair];
        result.gate_up_b = &ctx->lora_b[offset + gate_up_pair];
    }
    if (down_pair >= 0 && legacy_lora_slot_active(ctx, offset + down_pair)) {
        result.down_a = &ctx->lora_a[offset + down_pair];
        result.down_b = &ctx->lora_b[offset + down_pair];
    }
    result.scaling = ctx->lora_scaling;
    return result;
}

static ncclDataType_t nccl_dtype_for(const at::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case at::kBFloat16: return ncclBfloat16;
        case at::kFloat: return ncclFloat;
        case at::kHalf: return ncclFloat16;
        default:
            TORCH_CHECK(false, "unsupported LoRA gradient dtype for EP all-reduce: ",
                        tensor.scalar_type());
    }
}

static void tp_broadcast_lora_parameter(
    TrainingContext* ctx, at::Tensor& tensor
) {
    if (!ctx || ctx->tp_world_size <= 1 || !tensor.defined()) return;
    TORCH_CHECK(ctx->tp_comm,
        "LoRA TP communicator is not initialized for parameter broadcast");
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(),
        "LoRA TP parameter broadcast requires a contiguous CUDA tensor");
    const int dev = tensor.device().index();
    cudaSetDevice(dev);
    auto stream = c10::cuda::getCurrentCUDAStream(dev).stream();
    auto err = ncclBroadcast(
        tensor.data_ptr(), tensor.data_ptr(), tensor.numel(),
        nccl_dtype_for(tensor), 0, ctx->tp_comm, stream);
    TORCH_CHECK(err == ncclSuccess,
        "NCCL LoRA parameter broadcast failed: ", ncclGetErrorString(err));
}

static void replica_broadcast_lora_parameter(
    at::Tensor& tensor, ncclComm_t communicator, int world_size,
    const char* axis
) {
    if (world_size <= 1 || !tensor.defined()) return;
    TORCH_CHECK(communicator, "LoRA ", axis,
        " communicator is not initialized for parameter broadcast");
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(),
        "LoRA replica parameter broadcast requires a contiguous CUDA tensor");
    const int dev = tensor.device().index();
    cudaSetDevice(dev);
    auto stream = c10::cuda::getCurrentCUDAStream(dev).stream();
    auto err = ncclBroadcast(
        tensor.data_ptr(), tensor.data_ptr(), tensor.numel(),
        nccl_dtype_for(tensor), 0, communicator, stream);
    TORCH_CHECK(err == ncclSuccess,
        "NCCL LoRA ", axis, " parameter broadcast failed: ",
        ncclGetErrorString(err));
}

static void synchronize_adapter_replicated_lora_parameters(
    TrainingContext* ctx, TrainingContext::LoRAAdapter& adapter);

static void synchronize_fixed_replicated_lora_parameters(TrainingContext* ctx) {
    if (!ctx) return;
    for (int64_t layer = 0; layer < ctx->num_layers; ++layer) {
        const int64_t offset = ctx->lora_layer_offset[layer];
        const int64_t pairs = lora_pair_count(ctx->layer_configs[layer]);
        for (int64_t pair = 0; pair < pairs; ++pair) {
            const int64_t slot = offset + pair;
            if (!legacy_lora_slot_active(ctx, slot)) continue;
            const auto layout = lora_tp_layout(ctx, layer, pair);
            if (layout == LoraTpLayout::ColumnParallel)
                tp_broadcast_lora_parameter(ctx, ctx->lora_a[slot]);
            else if (layout == LoraTpLayout::RowParallel)
                tp_broadcast_lora_parameter(ctx, ctx->lora_b[slot]);
            const bool grouped_expert =
                lora_projection_table(ctx->layer_configs[layer])
                    .entries[pair].grouped_expert;
            if (!grouped_expert) {
                replica_broadcast_lora_parameter(
                    ctx->lora_a[slot], ctx->nccl_comm, ctx->ep_world_size, "EP");
                replica_broadcast_lora_parameter(
                    ctx->lora_b[slot], ctx->nccl_comm, ctx->ep_world_size, "EP");
            }
            replica_broadcast_lora_parameter(
                ctx->lora_a[slot], ctx->dp_comm, ctx->dp_world_size, "DP");
            replica_broadcast_lora_parameter(
                ctx->lora_b[slot], ctx->dp_comm, ctx->dp_world_size, "DP");
        }
    }
    for (auto& adapter : ctx->adapters)
        synchronize_adapter_replicated_lora_parameters(ctx, adapter);
}

static void synchronize_adapter_replicated_lora_parameters(
    TrainingContext* ctx, TrainingContext::LoRAAdapter& adapter
) {
    if (!ctx) return;
    if ((ctx->base_tp_attention || ctx->base_tp_mlp) &&
        ctx->tp_world_size > 1 && !ctx->tp_comm)
        return;  // qwen36_init_nccl synchronizes deferred adapters.
    if (ctx->expert_parallel && ctx->ep_world_size > 1 && !ctx->nccl_comm)
        return;
    if (ctx->data_parallel && ctx->dp_world_size > 1 && !ctx->dp_comm)
        return;
    for (auto& [layer, pairs] : adapter.params) {
        for (int64_t pair = 0; pair < static_cast<int64_t>(pairs.size()); ++pair) {
            auto& [a, b] = pairs[pair];
            if (!a.requires_grad() && !b.requires_grad()) continue;
            const auto layout = lora_tp_layout(ctx, layer, pair);
            if (layout == LoraTpLayout::ColumnParallel)
                tp_broadcast_lora_parameter(ctx, a);
            else if (layout == LoraTpLayout::RowParallel)
                tp_broadcast_lora_parameter(ctx, b);
            const bool grouped_expert =
                lora_projection_table(ctx->layer_configs[layer])
                    .entries[pair].grouped_expert;
            if (!grouped_expert) {
                replica_broadcast_lora_parameter(
                    a, ctx->nccl_comm, ctx->ep_world_size, "EP");
                replica_broadcast_lora_parameter(
                    b, ctx->nccl_comm, ctx->ep_world_size, "EP");
            }
            replica_broadcast_lora_parameter(
                a, ctx->dp_comm, ctx->dp_world_size, "DP");
            replica_broadcast_lora_parameter(
                b, ctx->dp_comm, ctx->dp_world_size, "DP");
        }
    }
}

static void tp_sum_replicated_lora_accumulator(
    TrainingContext* ctx, at::Tensor& accumulator,
    LoraTpLayout layout, bool is_a
) {
    const bool replicated =
        (layout == LoraTpLayout::ColumnParallel && is_a) ||
        (layout == LoraTpLayout::RowParallel && !is_a);
    if (!replicated || !accumulator.defined() || ctx->tp_world_size <= 1) return;
    TORCH_CHECK(ctx->tp_comm,
        "LoRA TP communicator is not initialized for replicated gradient sum");
    auto reduced = NcclAllReduceFunction::allreduce(
        accumulator, ctx->tp_comm, ctx->tp_stream);
    at::NoGradGuard guard;
    accumulator.copy_(reduced);
}

static void reduce_lora_accumulator(
    TrainingContext* ctx, at::Tensor& accumulator, double scale,
    bool reduce_ep, bool reduce_dp
) {
    if (!accumulator.defined()) return;
    TORCH_CHECK(accumulator.scalar_type() == at::kFloat,
        "LoRA DP gradient accumulator must be FP32");
    auto contiguous = accumulator.contiguous();
    if (scale != 1.0) {
        contiguous = contiguous * scale;
    }
    auto reduce_axis = [&](ncclComm_t communicator, const char* axis) {
        TORCH_CHECK(communicator, "LoRA ", axis,
            " gradient all-reduce has no communicator");
        auto reduced = at::empty_like(contiguous);
        const int dev = contiguous.device().index();
        cudaSetDevice(dev);
        auto stream = c10::cuda::getCurrentCUDAStream(dev).stream();
        auto err = ncclAllReduce(
            contiguous.data_ptr(), reduced.data_ptr(), contiguous.numel(),
            nccl_dtype_for(contiguous), ncclSum, communicator, stream);
        TORCH_CHECK(err == ncclSuccess, "NCCL LoRA ", axis,
            " gradient all-reduce failed: ", ncclGetErrorString(err));
        contiguous = reduced;
    };
    if (reduce_ep) reduce_axis(ctx->nccl_comm, "EP");
    if (reduce_dp) reduce_axis(ctx->dp_comm, "DP");
    at::NoGradGuard guard;
    accumulator.copy_(contiguous);
}

static void normalize_lora_accumulator_numerator(
    TrainingContext* ctx, at::Tensor& accumulator,
    const at::Tensor& global_weight, bool reduce_ep, bool reduce_dp
) {
    if (!accumulator.defined()) return;
    TORCH_CHECK(accumulator.scalar_type() == at::kFloat,
        "LoRA DP gradient accumulator must be FP32");
    TORCH_CHECK(global_weight.numel() == 1,
        "per-adapter LoRA global token weight must be scalar");
    reduce_lora_accumulator(
        ctx, accumulator, 1.0, reduce_ep, reduce_dp);
    at::NoGradGuard guard;
    accumulator.div_(global_weight.clamp_min(1.0));
}

static void reduce_lora_accumulator_weighted(
    TrainingContext* ctx, at::Tensor& accumulator,
    const at::Tensor& local_weight, const at::Tensor& global_weight,
    bool reduce_ep, bool reduce_dp
) {
    if (!accumulator.defined()) return;
    TORCH_CHECK(accumulator.scalar_type() == at::kFloat,
        "LoRA DP gradient accumulator must be FP32");
    TORCH_CHECK(local_weight.numel() == 1 && global_weight.numel() == 1,
        "per-adapter LoRA token weights must be scalar");
    at::NoGradGuard guard;
    accumulator.mul_(local_weight);
    reduce_lora_accumulator(
        ctx, accumulator, 1.0, reduce_ep, reduce_dp);
    accumulator.div_(global_weight.clamp_min(1.0));
}

struct GroupedLoraGradientSyncPlan {
    struct Entry {
        at::Tensor* accumulator = nullptr;
        at::Tensor local_weight;
        at::Tensor global_weight;
        at::Tensor work;
        double pre_scale = 1.0;
        double post_scale = 1.0;
        bool reduce_ep = false;
        bool reduce_dp = false;
        bool reduce_tp = false;
    };

    std::vector<Entry> entries;

    static bool tp_replicated(LoraTpLayout layout, bool is_a) {
        return (layout == LoraTpLayout::ColumnParallel && is_a) ||
            (layout == LoraTpLayout::RowParallel && !is_a);
    }

    void add(
        TrainingContext* ctx,
        at::Tensor& accumulator,
        double pre_scale,
        const at::Tensor& local_weight,
        const at::Tensor& global_weight,
        double post_scale,
        bool reduce_ep,
        bool reduce_dp,
        LoraTpLayout tp_layout,
        bool is_a
    ) {
        if (!accumulator.defined()) return;
        Entry entry;
        entry.accumulator = &accumulator;
        entry.local_weight = local_weight;
        entry.global_weight = global_weight;
        entry.pre_scale = pre_scale;
        entry.post_scale = post_scale;
        entry.reduce_ep = reduce_ep;
        entry.reduce_dp = reduce_dp;
        entry.reduce_tp = ctx->tp_world_size > 1 &&
            tp_replicated(tp_layout, is_a);
        entries.push_back(std::move(entry));
    }

    void prepare(TrainingContext* ctx) {
        for (auto& entry : entries) {
            TORCH_CHECK(entry.accumulator && entry.accumulator->is_cuda() &&
                    entry.accumulator->scalar_type() == at::kFloat &&
                    entry.accumulator->device().index() == ctx->cuda_device,
                "grouped LoRA gradient sync requires CUDA FP32 accumulators "
                "on the context device");
            TORCH_CHECK(std::isfinite(entry.pre_scale) &&
                    std::isfinite(entry.post_scale),
                "grouped LoRA gradient sync scale must be finite");
            if (entry.local_weight.defined()) {
                TORCH_CHECK(entry.local_weight.is_cuda() &&
                        entry.local_weight.numel() == 1 &&
                        entry.local_weight.device() == entry.accumulator->device(),
                    "grouped LoRA local token weight must be a CUDA scalar");
            }
            if (entry.global_weight.defined()) {
                TORCH_CHECK(entry.global_weight.is_cuda() &&
                        entry.global_weight.numel() == 1 &&
                        entry.global_weight.device() == entry.accumulator->device(),
                    "grouped LoRA global token weight must be a CUDA scalar");
            }
            TORCH_CHECK(entry.accumulator->is_contiguous(),
                "grouped LoRA gradient sync requires contiguous accumulators");
            entry.work = *entry.accumulator;
        }
    }

    template <typename Predicate>
    static void run_group(
        std::vector<Entry>& entries,
        ncclComm_t communicator,
        cudaStream_t stream,
        const char* axis,
        Predicate predicate
    ) {
        const bool any = std::any_of(
            entries.begin(), entries.end(), predicate);
        if (!any) return;
        TORCH_CHECK(communicator, "grouped LoRA ", axis,
            " gradient sync has no communicator");
        const auto start_error = ncclGroupStart();
        TORCH_CHECK(start_error == ncclSuccess,
            "grouped LoRA ", axis, " ncclGroupStart failed: ",
            ncclGetErrorString(start_error));
        ncclResult_t first_error = ncclSuccess;
        for (auto& entry : entries) {
            if (!predicate(entry)) continue;
            const auto error = ncclAllReduce(
                entry.work.data_ptr(), entry.work.data_ptr(), entry.work.numel(),
                nccl_dtype_for(entry.work), ncclSum, communicator, stream);
            if (first_error == ncclSuccess && error != ncclSuccess)
                first_error = error;
        }
        const auto end_error = ncclGroupEnd();
        TORCH_CHECK(first_error == ncclSuccess,
            "grouped LoRA ", axis, " gradient all-reduce failed: ",
            ncclGetErrorString(first_error));
        TORCH_CHECK(end_error == ncclSuccess,
            "grouped LoRA ", axis, " ncclGroupEnd failed: ",
            ncclGetErrorString(end_error));
    }

    void execute(TrainingContext* ctx) {
        std::exception_ptr preparation_error;
        try {
            prepare(ctx);
        } catch (...) {
            preparation_error = std::current_exception();
        }
        const bool local_ready = !preparation_error;
        const bool globally_ready = adapter_collective_all_succeeded(
            ctx, local_ready);
        if (!local_ready) std::rethrow_exception(preparation_error);
        TORCH_CHECK(globally_ready,
            "grouped LoRA gradient sync preparation failed on another rank");

        at::NoGradGuard guard;
        for (auto& entry : entries) {
            if (entry.local_weight.defined())
                entry.work.mul_(entry.local_weight);
            if (entry.pre_scale != 1.0)
                entry.work.mul_(entry.pre_scale);
        }

        const auto stream = c10::cuda::getCurrentCUDAStream(
            ctx->cuda_device).stream();
        run_group(entries, ctx->nccl_comm, stream, "EP",
            [](const Entry& entry) { return entry.reduce_ep; });
        run_group(entries, ctx->dp_comm, stream, "DP",
            [](const Entry& entry) { return entry.reduce_dp; });

        for (auto& entry : entries) {
            if (entry.global_weight.defined())
                entry.work.div_(entry.global_weight);
            if (entry.post_scale != 1.0)
                entry.work.mul_(entry.post_scale);
        }

        run_group(entries, ctx->tp_comm, stream, "TP",
            [](const Entry& entry) { return entry.reduce_tp; });
    }
};

// Fixed-LoRA gradients are accumulated as token-weighted numerators. Replicated
// DP sums all replicated parameters and divides by the global token count;
// legacy EP keeps its replicated batch/local expert semantics. Sharded A2A
// sums non-expert parameters across source ranks, while expert parameters have
// already received all source numerators through the inverse A2A and therefore
// only divide by the global token count.
static bool replica_token_weights_match(
    TrainingContext* ctx, const at::Tensor& weights,
    ncclComm_t communicator, int32_t world_size, const char* axis
) {
    if (!communicator || world_size <= 1) return true;
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == at::kFloat &&
            weights.is_contiguous(),
        "LoRA replica token weights must be contiguous CUDA FP32");
    auto minimum = weights.clone();
    auto maximum = weights.clone();
    auto stream = c10::cuda::getCurrentCUDAStream(
        weights.device().index()).stream();
    const auto minimum_error = ncclAllReduce(
        minimum.data_ptr<float>(), minimum.data_ptr<float>(), minimum.numel(),
        ncclFloat, ncclMin, communicator, stream);
    TORCH_CHECK(minimum_error == ncclSuccess,
        "NCCL LoRA ", axis,
        " token-count minimum validation failed: ",
        ncclGetErrorString(minimum_error));
    const auto maximum_error = ncclAllReduce(
        maximum.data_ptr<float>(), maximum.data_ptr<float>(), maximum.numel(),
        ncclFloat, ncclMax, communicator, stream);
    TORCH_CHECK(maximum_error == ncclSuccess,
        "NCCL LoRA ", axis,
        " token-count maximum validation failed: ",
        ncclGetErrorString(maximum_error));
    return (minimum == maximum).all().item<bool>();
}

static bool synchronize_lora_gradients(
    TrainingContext* ctx, const at::Tensor& target_mask,
    double accumulated_token_weight = 0.0,
    const at::Tensor* per_adapter_token_counts = nullptr,
    std::vector<uint8_t>* adapter_has_global_tokens = nullptr,
    bool adapter_token_counts_prevalidated = false
) {
    const bool sharded_a2a = ctx->expert_parallel && ctx->nccl_comm &&
        env_enabled("QWEN36_EP_A2A_SHARDED");
    TORCH_CHECK(!sharded_a2a || env_enabled("QWEN36_EP_A2A"),
        "QWEN36_EP_A2A_SHARDED=1 requires QWEN36_EP_A2A=1");
    const bool dp_allreduce = ctx->data_parallel && ctx->dp_comm;
    const bool normalization_allreduce = dp_allreduce || sharded_a2a;
    const double replicated_a2a_expert_scale =
        ctx->expert_parallel && env_enabled("QWEN36_EP_A2A") && !sharded_a2a
        ? 1.0 / static_cast<double>(ctx->ep_world_size)
        : 1.0;
    auto sum_replica_axes = [&](at::Tensor value, bool reduce_ep) {
        auto reduce_axis = [&](ncclComm_t communicator, const char* axis) {
            auto reduced = at::empty_like(value);
            auto stream = c10::cuda::getCurrentCUDAStream(
                value.device().index()).stream();
            auto err = ncclAllReduce(
                value.data_ptr(), reduced.data_ptr(), value.numel(),
                nccl_dtype_for(value), ncclSum, communicator, stream);
            TORCH_CHECK(err == ncclSuccess, "NCCL ", axis,
                " replica all-reduce failed: ", ncclGetErrorString(err));
            value = reduced;
        };
        if (reduce_ep) reduce_axis(ctx->nccl_comm, "EP");
        if (dp_allreduce) reduce_axis(ctx->dp_comm, "DP");
        return value;
    };
    const bool per_adapter_weighting = per_adapter_token_counts &&
        per_adapter_token_counts->defined();
    const bool grouped_sync = env_enabled("QWEN36_GROUPED_LORA_SYNC", true);
    GroupedLoraGradientSyncPlan grouped_plan;
    at::Tensor local_adapter_weights;
    at::Tensor global_adapter_weights;
    double scale = 1.0;
    if (per_adapter_weighting) {
        TORCH_CHECK(per_adapter_token_counts->dim() == 1 &&
                    per_adapter_token_counts->size(0) ==
                        static_cast<int64_t>(ctx->adapters.size()),
            "dynamic LoRA token-count vector must match adapter registry");
        local_adapter_weights = per_adapter_token_counts->to(at::kFloat)
            .contiguous();
        if (!adapter_token_counts_prevalidated) {
            TORCH_CHECK(at::logical_and(
                    at::isfinite(local_adapter_weights),
                    local_adapter_weights >= 0).all().item<bool>(),
                "dynamic LoRA token counts must be finite and non-negative");
        }
        bool replica_weights_match = replica_token_weights_match(
            ctx, local_adapter_weights, ctx->tp_comm,
            ctx->tp_world_size, "TP");
        if (ctx->expert_parallel && !sharded_a2a) {
            const bool ep_weights_match = replica_token_weights_match(
                ctx, local_adapter_weights, ctx->nccl_comm,
                ctx->ep_world_size, "replicated EP");
            replica_weights_match = replica_weights_match && ep_weights_match;
        }
        TORCH_CHECK(adapter_collective_all_succeeded(
                ctx, replica_weights_match) && replica_weights_match,
            "dynamic LoRA token counts differ across TP or replicated EP "
            "ranks; replicas must use identical target masks");
        global_adapter_weights = normalization_allreduce
            ? sum_replica_axes(local_adapter_weights, sharded_a2a)
            : local_adapter_weights;
        if (adapter_has_global_tokens) {
            adapter_has_global_tokens->assign(ctx->adapters.size(), 0);
            auto global_cpu = global_adapter_weights.to(
                at::TensorOptions().device(at::kCPU));
            const auto* counts = global_cpu.data_ptr<float>();
            for (size_t i = 0; i < ctx->adapters.size(); ++i) {
                (*adapter_has_global_tokens)[i] = counts[i] > 0.0f ? 1 : 0;
            }
        }
    } else {
        auto local = at::full({1}, accumulated_token_weight,
            at::TensorOptions().dtype(at::kFloat).device(target_mask.device()));
        bool replica_weights_match = replica_token_weights_match(
            ctx, local, ctx->tp_comm, ctx->tp_world_size, "TP");
        if (ctx->expert_parallel && !sharded_a2a) {
            const bool ep_weights_match = replica_token_weights_match(
                ctx, local, ctx->nccl_comm,
                ctx->ep_world_size, "replicated EP");
            replica_weights_match = replica_weights_match && ep_weights_match;
        }
        TORCH_CHECK(adapter_collective_all_succeeded(
                ctx, replica_weights_match) && replica_weights_match,
            "fixed LoRA token weights differ across TP or replicated EP "
            "ranks; replicas must use identical accumulation windows");
        auto global = normalization_allreduce
            ? sum_replica_axes(local, sharded_a2a)
            : local;
        const double global_weight = global.item<double>();
        if (global_weight <= 0.0) return false;
        scale = 1.0 / global_weight;
    }
    for (size_t adapter_index = 0;
         adapter_index < ctx->adapters.size(); ++adapter_index) {
        auto& adapter = ctx->adapters[adapter_index];
        for (auto& [layer_idx, pairs] : adapter.params) {
            auto table = lora_projection_table(ctx->layer_configs[layer_idx]);
            for (int64_t pair = 0; pair < (int64_t)pairs.size(); ++pair) {
                auto accum_it = adapter.grad_accum.find(layer_idx);
                TORCH_CHECK(accum_it != adapter.grad_accum.end() &&
                            pair < (int64_t)accum_it->second.size(),
                    "dynamic LoRA gradient accumulator layout mismatch");
                if (per_adapter_weighting) {
                    auto local_weight = local_adapter_weights.index(
                        {static_cast<int64_t>(adapter_index)});
                    auto global_weight = global_adapter_weights.index(
                        {static_cast<int64_t>(adapter_index)});
                    TORCH_CHECK(adapter_has_global_tokens &&
                            adapter_has_global_tokens->size() == ctx->adapters.size(),
                        "dynamic LoRA global-token activity is unavailable");
                    if (!(*adapter_has_global_tokens)[adapter_index]) {
                        if (accum_it->second[pair][0].defined())
                            accum_it->second[pair][0].zero_();
                        if (accum_it->second[pair][1].defined())
                            accum_it->second[pair][1].zero_();
                        continue;
                    }
                    const bool grouped_expert = table.entries[pair].grouped_expert;
                    const auto layout = lora_tp_layout(ctx, layer_idx, pair);
                    if (sharded_a2a) {
                        // Sharded A2A restores each source row to a numerator
                        // before backward. Expert owners already receive all
                        // source numerators; replicated projections still need
                        // one all-reduce before division by the global count.
                        if (grouped_sync) {
                            grouped_plan.add(
                                ctx, accum_it->second[pair][0], 1.0,
                                at::Tensor(), global_weight, 1.0,
                                !grouped_expert, dp_allreduce, layout, true);
                            grouped_plan.add(
                                ctx, accum_it->second[pair][1], 1.0,
                                at::Tensor(), global_weight, 1.0,
                                !grouped_expert, dp_allreduce, layout, false);
                        } else {
                            normalize_lora_accumulator_numerator(
                                ctx, accum_it->second[pair][0], global_weight,
                                !grouped_expert, dp_allreduce);
                            normalize_lora_accumulator_numerator(
                                ctx, accum_it->second[pair][1], global_weight,
                                !grouped_expert, dp_allreduce);
                        }
                    } else {
                        if (grouped_sync) {
                            const double expert_scale = grouped_expert
                                ? replicated_a2a_expert_scale
                                : 1.0;
                            grouped_plan.add(
                                ctx, accum_it->second[pair][0], 1.0,
                                local_weight, global_weight, expert_scale,
                                false, dp_allreduce, layout, true);
                            grouped_plan.add(
                                ctx, accum_it->second[pair][1], 1.0,
                                local_weight, global_weight, expert_scale,
                                false, dp_allreduce, layout, false);
                        } else {
                            reduce_lora_accumulator_weighted(
                                ctx, accum_it->second[pair][0], local_weight,
                                global_weight, false, dp_allreduce);
                            reduce_lora_accumulator_weighted(
                                ctx, accum_it->second[pair][1], local_weight,
                                global_weight, false, dp_allreduce);
                        }
                        if (!grouped_sync && grouped_expert &&
                            replicated_a2a_expert_scale != 1.0) {
                            at::NoGradGuard guard;
                            accum_it->second[pair][0].mul_(
                                replicated_a2a_expert_scale);
                            accum_it->second[pair][1].mul_(
                                replicated_a2a_expert_scale);
                        }
                    }
                    continue;
                }
                // Routed-expert tensors are sharded only in EP. Pure DP has
                // replicated experts and therefore uses the same reduction as
                // shared projections; the legacy accumulation path preserves
                // local normalization when no DP communicator is present.
                if (table.entries[pair].grouped_expert) {
                    if (accumulated_token_weight > 0.0) {
                        const auto layout = lora_tp_layout(ctx, layer_idx, pair);
                        if (grouped_sync) {
                            grouped_plan.add(
                                ctx, accum_it->second[pair][0], scale,
                                at::Tensor(), at::Tensor(), 1.0,
                                false, dp_allreduce, layout, true);
                            grouped_plan.add(
                                ctx, accum_it->second[pair][1], scale,
                                at::Tensor(), at::Tensor(), 1.0,
                                false, dp_allreduce, layout, false);
                        } else {
                            reduce_lora_accumulator(
                                ctx, accum_it->second[pair][0], scale,
                                false, dp_allreduce);
                            reduce_lora_accumulator(
                                ctx, accum_it->second[pair][1], scale,
                                false, dp_allreduce);
                        }
                    }
                    continue;
                }
                const auto layout = lora_tp_layout(ctx, layer_idx, pair);
                if (grouped_sync) {
                    grouped_plan.add(
                        ctx, accum_it->second[pair][0], scale,
                        at::Tensor(), at::Tensor(), 1.0,
                        sharded_a2a, dp_allreduce, layout, true);
                    grouped_plan.add(
                        ctx, accum_it->second[pair][1], scale,
                        at::Tensor(), at::Tensor(), 1.0,
                        sharded_a2a, dp_allreduce, layout, false);
                } else {
                    reduce_lora_accumulator(
                        ctx, accum_it->second[pair][0], scale,
                        sharded_a2a, dp_allreduce);
                    reduce_lora_accumulator(
                        ctx, accum_it->second[pair][1], scale,
                        sharded_a2a, dp_allreduce);
                }
            }
        }
    }
    // Projection-aware TP keeps one LoRA factor replicated. Its gradient is
    // the sum of disjoint output-head (column) or input-column (row)
    // contributions and is synchronized once at the optimizer boundary.
    if (!grouped_sync) {
        for (auto& adapter : ctx->adapters) {
            for (auto& [layer_idx, pairs] : adapter.grad_accum) {
                for (int64_t pair = 0;
                     pair < static_cast<int64_t>(pairs.size()); ++pair) {
                    const auto layout = lora_tp_layout(ctx, layer_idx, pair);
                    tp_sum_replicated_lora_accumulator(
                        ctx, pairs[pair][0], layout, true);
                    tp_sum_replicated_lora_accumulator(
                        ctx, pairs[pair][1], layout, false);
                }
            }
        }
    }
    if (per_adapter_weighting) {
        if (grouped_sync) grouped_plan.execute(ctx);
        return adapter_has_global_tokens && std::any_of(
            adapter_has_global_tokens->begin(),
            adapter_has_global_tokens->end(),
            [](uint8_t active) { return active != 0; });
    }
    // The replicated-source A2A path sends an identical token batch from every
    // EP rank. Average only its sharded expert parameter gradients here;
    // scaling the combine backward would also under-scale the activation
    // gradient returned to each source graph. Sharded A2A is handled above via
    // global token-count weighting and does not enter this branch.
    for (int64_t layer = 0; layer < ctx->num_layers; ++layer) {
        auto table = lora_projection_table(ctx->layer_configs[layer]);
        int64_t offset = ctx->lora_layer_offset[layer];
        for (int64_t pair = 0; pair < table.count; ++pair) {
            // Routed expert LoRA is local-only for EP, while pure DP owns the
            // complete replicated expert tensor and must all-reduce it.
            if (table.entries[pair].grouped_expert) {
                const auto layout = lora_tp_layout(ctx, layer, pair);
                if (grouped_sync) {
                    grouped_plan.add(
                        ctx, ctx->grad_accum_a[offset + pair],
                        scale * replicated_a2a_expert_scale,
                        at::Tensor(), at::Tensor(), 1.0,
                        false, dp_allreduce, layout, true);
                    grouped_plan.add(
                        ctx, ctx->grad_accum_b[offset + pair],
                        scale * replicated_a2a_expert_scale,
                        at::Tensor(), at::Tensor(), 1.0,
                        false, dp_allreduce, layout, false);
                } else {
                    reduce_lora_accumulator(
                        ctx, ctx->grad_accum_a[offset + pair],
                        scale * replicated_a2a_expert_scale,
                        false, dp_allreduce);
                    reduce_lora_accumulator(
                        ctx, ctx->grad_accum_b[offset + pair],
                        scale * replicated_a2a_expert_scale,
                        false, dp_allreduce);
                }
                continue;
            }
            const auto layout = lora_tp_layout(ctx, layer, pair);
            if (grouped_sync) {
                grouped_plan.add(
                    ctx, ctx->grad_accum_a[offset + pair], scale,
                    at::Tensor(), at::Tensor(), 1.0,
                    sharded_a2a, dp_allreduce, layout, true);
                grouped_plan.add(
                    ctx, ctx->grad_accum_b[offset + pair], scale,
                    at::Tensor(), at::Tensor(), 1.0,
                    sharded_a2a, dp_allreduce, layout, false);
            } else {
                reduce_lora_accumulator(
                    ctx, ctx->grad_accum_a[offset + pair], scale,
                    sharded_a2a, dp_allreduce);
                reduce_lora_accumulator(
                    ctx, ctx->grad_accum_b[offset + pair], scale,
                    sharded_a2a, dp_allreduce);
            }
        }
    }
    if (grouped_sync) {
        grouped_plan.execute(ctx);
    } else {
        for (int64_t layer = 0; layer < ctx->num_layers; ++layer) {
            const int64_t offset = ctx->lora_layer_offset[layer];
            const int64_t pairs = lora_pair_count(ctx->layer_configs[layer]);
            for (int64_t pair = 0; pair < pairs; ++pair) {
                const auto layout = lora_tp_layout(ctx, layer, pair);
                tp_sum_replicated_lora_accumulator(
                    ctx, ctx->grad_accum_a[offset + pair], layout, true);
                tp_sum_replicated_lora_accumulator(
                    ctx, ctx->grad_accum_b[offset + pair], layout, false);
            }
        }
    }
    return true;
}

static void elide_trivial_attention_mask(TrainingContext* ctx) {
    if (!ctx->attention_mask.defined() || ctx->attention_mask.numel() == 0) return;
    // A padding-free batch can use SDPA's native causal fast path. This is one
    // scalar synchronization per step, instead of materializing [B,S,S] in
    // every full-attention layer.
    if (at::all(ctx->attention_mask != 0).item<bool>()) {
        ctx->attention_mask = at::Tensor();
    }
}

// Linear attention carries recurrent state across the sequence. Until the
// kernel accepts packed cu_seqlens, only full sequences and strict right
// padding are safe; a 0 -> 1 transition would let padding state leak into a
// later real token (left padding and internal holes).
static void validate_linear_attention_mask(
    TrainingContext* ctx, const at::Tensor& attention_mask
) {
    if (!ctx || !attention_mask.defined() || attention_mask.numel() == 0) return;
    bool has_linear = false;
    for (const auto& cfg : ctx->layer_configs) {
        if (cfg.layer_type != 0) {
            has_linear = true;
            break;
        }
    }
    if (!has_linear) {
        for (const auto& cfg : ctx->mtp_layer_configs) {
            if (cfg.layer_type != 0) {
                has_linear = true;
                break;
            }
        }
    }
    if (!has_linear) return;
    TORCH_CHECK(attention_mask.dim() == 2,
        "linear-attention mask must be [batch, seq]");
    auto mask = attention_mask.to(at::kBool);
    if (mask.size(1) <= 1) return;
    auto leading = mask.narrow(1, 0, mask.size(1) - 1);
    auto trailing = mask.narrow(1, 1, mask.size(1) - 1);
    auto invalid_transition = leading.logical_not().logical_and(trailing);
    TORCH_CHECK(!invalid_transition.any().item<bool>(),
        "linear attention only supports full or strict right-padding masks; "
        "left-padding/internal holes require packed cu_seqlens support");
}

// ── Multi-LoRA: concat all adapters' A/B, 2x GEMM ──
// Pre-build cache of concatenated A/B per (layer, module) pair.
// Called once at start of forward; reused across all layers.

static void precompute_lora_cache(TrainingContext* ctx) {
    if (ctx->lora_cache_valid) return;
    ctx->lora_cache.clear();

    // Phase 1: collect all (cache_key, a_concat, b_concat) tuples
    struct LoraEntry {
        int64_t key;
        at::Tensor a_concat;  // [sum_ranks, in]
        at::Tensor b_concat;  // [out, sum_ranks]
    };
    std::vector<LoraEntry> entries;

    for (int64_t layer_idx = 0; layer_idx < ctx->num_layers; layer_idx++) {
        int64_t num_pairs = lora_pair_count(ctx->layer_configs[layer_idx]);
        for (int64_t pair_idx = 0; pair_idx < num_pairs; pair_idx++) {
            auto projection_table = lora_projection_table(ctx->layer_configs[layer_idx]);
            // Routed experts use activation-level low-rank GEMMs. Materializing
            // B@A for every local expert would erase LoRA's memory advantage.
            if (projection_table.entries[pair_idx].grouped_expert) continue;
            std::vector<at::Tensor> a_list, b_list;
            const char* module_name = lora_pair_name(ctx->layer_configs[layer_idx], pair_idx);
            for (auto& adapter : ctx->adapters) {
                if (!adapter.target_modules.empty() &&
                    adapter.target_modules.find(module_name) == adapter.target_modules.end())
                    continue;
                if (!adapter.target_layers.empty() && adapter.target_layers.find(layer_idx) == adapter.target_layers.end())
                    continue;
                auto it = adapter.params.find(layer_idx);
                if (it == adapter.params.end()) continue;
                if (pair_idx >= (int64_t)it->second.size()) continue;
                auto& [a, b] = it->second[pair_idx];
                if (!a.requires_grad() && !b.requires_grad()) continue;
                double scaling = adapter.alpha / (double)adapter.rank;
                b_list.push_back(b * scaling);
                a_list.push_back(a);
            }
            if (a_list.empty()) {
                if (!ctx->lora_a.empty() && layer_idx < (int64_t)ctx->lora_layer_offset.size()) {
                    int64_t la_offset = ctx->lora_layer_offset[layer_idx];
                    if (la_offset + pair_idx < (int64_t)ctx->lora_a.size()) {
                        if (!ctx->lora_active.empty() &&
                            !ctx->lora_active[la_offset + pair_idx]) {
                            continue;
                        }
                        b_list.push_back(ctx->lora_b[la_offset + pair_idx] * ctx->lora_scaling);
                        a_list.push_back(ctx->lora_a[la_offset + pair_idx]);
                    }
                }
            }
            if (!a_list.empty()) {
                at::Tensor a_concat, b_concat;
                if (a_list.size() == 1) {
                    a_concat = a_list[0];
                    b_concat = b_list[0];
                } else {
                    a_concat = at::cat(a_list, 0);  // [sum_ranks, in]
                    b_concat = at::cat(b_list, 1);  // [out, sum_ranks]
                }
                entries.push_back({lora_cache_key(layer_idx, pair_idx), a_concat, b_concat});
            }
        }
    }

    // Phase 2: group by (out_dim, in_dim, sum_ranks) and batch matmul
    // delta = b_concat @ a_concat  → [out, in]
    // Group entries with identical shapes to use at::bmm
    struct ShapeGroup {
        int64_t out_dim, in_dim, sum_ranks;
        std::vector<size_t> indices;
    };
    std::vector<ShapeGroup> groups;
    for (size_t i = 0; i < entries.size(); i++) {
        auto& e = entries[i];
        int64_t out_dim = e.b_concat.size(0);
        int64_t sum_ranks = e.b_concat.size(1);
        int64_t in_dim = e.a_concat.size(1);
        bool found = false;
        for (auto& g : groups) {
            if (g.out_dim == out_dim && g.in_dim == in_dim && g.sum_ranks == sum_ranks) {
                g.indices.push_back(i);
                found = true;
                break;
            }
        }
        if (!found) {
            groups.push_back({out_dim, in_dim, sum_ranks, {i}});
        }
    }

    // Phase 3: for each group, stack and bmm in BF16 (2x faster tensor cores)
    // LoRA params are FP32 for gradient stability; cast to BF16 for matmul only.
    // Autograd handles the cast backward (grad → FP32 automatically).
    auto bf16 = at::kBFloat16;
    for (auto& g : groups) {
        int n = (int)g.indices.size();
        if (n == 1) {
            auto& e = entries[g.indices[0]];
            auto delta = at::matmul(e.b_concat, e.a_concat);
            ctx->lora_cache[e.key] = delta;  // already BF16
        } else {
            std::vector<at::Tensor> b_stack_vec, a_stack_vec;
            b_stack_vec.reserve(n);
            a_stack_vec.reserve(n);
            for (auto idx : g.indices) {
                b_stack_vec.push_back(entries[idx].b_concat);
                a_stack_vec.push_back(entries[idx].a_concat);
            }
            auto b_stack = at::stack(b_stack_vec, 0);  // [N, out, sum_ranks] BF16
            auto a_stack = at::stack(a_stack_vec, 0);  // [N, sum_ranks, in] BF16
            auto deltas = at::bmm(b_stack, a_stack);   // [N, out, in] BF16
            for (int i = 0; i < n; i++) {
                ctx->lora_cache[entries[g.indices[i]].key] = deltas[i];
            }
        }
    }

    ctx->lora_cache_valid = true;
}

// ── Batched Multi-LoRA: activation-level B@(A@x) ──

/// Prepare stacked A/B tensors for all adapters per (layer, module).
/// Stores in ctx->lora_batch_cache. Called once before forward.
/// Replaces precompute_lora_cache when N > 1.
static void prepare_lora_batch(TrainingContext* ctx) {
    ctx->lora_batch_cache.clear();
    ctx->lora_batch_n = 0;

    for (int64_t layer_idx = 0; layer_idx < ctx->num_layers; layer_idx++) {
        int64_t num_pairs = lora_pair_count(ctx->layer_configs[layer_idx]);
        for (int64_t pair_idx = 0; pair_idx < num_pairs; pair_idx++) {
            std::vector<at::Tensor> a_list, b_list;
            std::vector<double> scalings;
            const char* module_name = lora_pair_name(ctx->layer_configs[layer_idx], pair_idx);
            for (auto& adapter : ctx->adapters) {
                if (!adapter.target_modules.empty() &&
                    adapter.target_modules.find(module_name) == adapter.target_modules.end())
                    continue;
                if (!adapter.target_layers.empty() && adapter.target_layers.find(layer_idx) == adapter.target_layers.end())
                    continue;
                auto it = adapter.params.find(layer_idx);
                if (it == adapter.params.end()) continue;
                if (pair_idx >= (int64_t)it->second.size()) continue;
                auto& [a, b] = it->second[pair_idx];
                if (!a.requires_grad() && !b.requires_grad()) continue;
                b_list.push_back(b);
                a_list.push_back(a);
                scalings.push_back(adapter.alpha / (double)adapter.rank);
            }
            if (a_list.empty()) continue;

            int64_t n = (int64_t)a_list.size();
            if (ctx->lora_batch_n == 0) ctx->lora_batch_n = n;

            auto a_stack = at::stack(a_list, 0);  // [N, rank, in]
            auto b_stack = at::stack(b_list, 0);  // [N, out, rank]
            // Create scaling tensor on GPU — from_blob only wraps CPU pointer,
            // so we must explicitly move it to the right device.
            auto scaling_cpu = at::from_blob(
                scalings.data(), {(int64_t)n, 1, 1},
                at::TensorOptions().dtype(at::kDouble)
            );
            auto scaling = scaling_cpu.to(a_stack.device()).to(at::kBFloat16);  // [N, 1, 1]

            ctx->lora_batch_cache[lora_cache_key(layer_idx, pair_idx)] = {
                a_stack, b_stack, scaling,
                lora_tp_layout(ctx, layer_idx, pair_idx)
            };
        }
    }
    ctx->lora_batch_valid = true;
}

/// Build activation-level entries for the fixed adapter in TP mode. The
/// singleton adapter dimension is expanded lazily to the input batch by
/// lora_activation_delta/dynamic_expert_lora_delta.
static void prepare_fixed_lora_batch(TrainingContext* ctx) {
    ctx->lora_batch_cache.clear();
    ctx->lora_batch_n = 1;
    for (int64_t layer_idx = 0; layer_idx < ctx->num_layers; ++layer_idx) {
        const int64_t pair_count = lora_pair_count(ctx->layer_configs[layer_idx]);
        const int64_t offset = ctx->lora_layer_offset[layer_idx];
        for (int64_t pair_idx = 0; pair_idx < pair_count; ++pair_idx) {
            if (lora_projection_table(ctx->layer_configs[layer_idx])
                    .entries[pair_idx].grouped_expert)
                continue;
            const int64_t slot = offset + pair_idx;
            if (!legacy_lora_slot_active(ctx, slot)) continue;
            auto& a = ctx->lora_a[slot];
            auto& b = ctx->lora_b[slot];
            auto scaling = at::full(
                {1, 1, 1}, ctx->lora_scaling,
                at::TensorOptions().dtype(a.scalar_type()).device(a.device()));
            ctx->lora_batch_cache[lora_cache_key(layer_idx, pair_idx)] = {
                a.unsqueeze(0), b.unsqueeze(0), scaling,
                lora_tp_layout(ctx, layer_idx, pair_idx)};
        }
    }
    ctx->lora_batch_valid = true;
}

/// Compute B@(A@x) * scaling — never materializes B@A.
/// x: [N, seq, in], A: [N, rank, in], B: [N, out, rank], scaling: [N, 1, 1]
/// returns: [N, seq, out]
static at::Tensor lora_activation_delta(
    TrainingContext* ctx,
    const at::Tensor& x,          // [N, seq, in]
    const at::Tensor& A,          // [N, rank, in]
    const at::Tensor& B,          // [N, out, rank]
    const at::Tensor& scaling,    // [N, 1, 1]
    LoraTpLayout layout
) {
    // Cast to compute dtype (BF16)
    auto kind = x.scalar_type();
    auto A_c = A.to(kind);
    auto B_c = B.to(kind);
    auto s_c = scaling.to(kind);
    if (A_c.size(0) == 1 && x.size(0) > 1) {
        A_c = A_c.expand({x.size(0), A_c.size(1), A_c.size(2)});
        B_c = B_c.expand({x.size(0), B_c.size(1), B_c.size(2)});
        s_c = s_c.expand({x.size(0), 1, 1});
    }
    // Latent-rank TP sums local deltas in forward. Its replicated input must
    // likewise sum the rank-local input-gradient contributions in backward;
    // otherwise a later sharded LoRA branch feeds only a partial dgrad into
    // preceding replicated layers.
    auto lora_input = layout == LoraTpLayout::LatentRank
        ? tp_copy_lora_input(ctx, x) : x;
    // Ax = A @ x^T  → [N, rank, seq]
    auto Ax = at::bmm(A_c, lora_input.transpose(-2, -1));
    // delta = B @ Ax → [N, out, seq] → transpose → [N, seq, out]
    auto delta = at::bmm(B_c, Ax).transpose(-2, -1);
    auto scaled = delta * s_c;
    return layout == LoraTpLayout::LatentRank
        ? tp_allreduce_lora_delta(ctx, scaled) : scaled;
}

static const LoraBatchEntry* lora_batch_entry(
    TrainingContext* ctx, int64_t layer_idx, int64_t pair_idx
) {
    if (!ctx || !ctx->lora_batch_valid || pair_idx < 0) return nullptr;
    auto it = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, pair_idx));
    return it == ctx->lora_batch_cache.end() ? nullptr : &it->second;
}

static at::Tensor dense_mlp_forward_batched(
    TrainingContext* ctx, int64_t layer_idx, const at::Tensor& hidden,
    const at::Tensor& gate_proj, const at::Tensor& up_proj,
    const at::Tensor& down_proj, at::ScalarType compute_type
) {
    const auto& cfg = ctx->layer_configs[layer_idx];
    const int64_t gate_pair = lora_pair_index(cfg, "gate_proj");
    const int64_t up_pair = lora_pair_index(cfg, "up_proj");
    const int64_t down_pair = lora_pair_index(cfg, "down_proj");

    auto mlp_input = tp_copy_base_mlp_input(ctx, hidden);
    auto gate_out = at::matmul(mlp_input, gate_proj.t());
    auto up_out = at::matmul(mlp_input, up_proj.t());
    gate_out = add_batched_lora(
        ctx, gate_out, mlp_input, lora_batch_entry(ctx, layer_idx, gate_pair));
    up_out = add_batched_lora(
        ctx, up_out, mlp_input, lora_batch_entry(ctx, layer_idx, up_pair));
    auto activated = fused_swiglu_op(gate_out, up_out, 0.0);
    auto result = at::matmul(activated, down_proj.t());
    result = add_batched_lora(
        ctx, result, activated, lora_batch_entry(ctx, layer_idx, down_pair));
    // Base TP uses a row-parallel down projection.  Reduce its local hidden
    // contribution before the residual add.
    return tp_allreduce_base_mlp(ctx, result.to(compute_type));
}

__attribute__((noinline, visibility("default")))
at::Tensor apply_multi_lora(
    TrainingContext* ctx, int64_t layer_idx, int64_t pair_idx,
    const at::Tensor& base_weight
) {
    auto it = ctx->lora_cache.find(lora_cache_key(layer_idx, pair_idx));
    if (it == ctx->lora_cache.end()) return base_weight;

    // Cached delta_weight = b_concat @ a_concat (BF16, precomputed in batched bmm)
    auto& delta = it->second;
    return base_weight + delta;  // both BF16, no conversion needed
}

// Forward declarations for sub-layer checkpointing
// Sub-layer checkpointing: split each layer into attn + mlp segments
// Enabled by QWEN36_SUBCKPT=1 env var. Reduces peak memory by ~2x
// at the cost of 2x extra recomputation per layer during backward.
// ──────────────────────────────────────────────────────────────────────

// Compute attention output from hidden
at::Tensor compute_attn_only(
    TrainingContext* ctx, const at::Tensor& hidden, int64_t layer_idx, at::ScalarType kind
) {
    const auto& cfg = ctx->layer_configs[layer_idx];
    int64_t w_offset = 0;
    for (int64_t j = 0; j < layer_idx; j++)
        w_offset += weight_count_for_layer(ctx->layer_configs[j]);
    auto attn_input = rms_norm(hidden, *ctx->weight_ptrs[w_offset + 0], cfg.rms_eps);

    // Use batched path if lora_batch is active
    if (ctx->lora_batch_valid) {
        if (cfg.layer_type == 0) {
            auto qp = *ctx->weight_ptrs[w_offset+2], qn = *ctx->weight_ptrs[w_offset+3];
            auto kp = *ctx->weight_ptrs[w_offset+4], kn = *ctx->weight_ptrs[w_offset+5];
            auto vp = *ctx->weight_ptrs[w_offset+6], op = *ctx->weight_ptrs[w_offset+7];
            return full_attention_batched(
                ctx, attn_input, layer_idx, qp, qn, kp, kn, vp, op,
                cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
                cfg.partial_rotary_factor, cfg.rope_theta, cfg.rms_eps, kind,
                ctx->attention_mask);
        } else {
            auto qkv = *ctx->weight_ptrs[w_offset+2], z = *ctx->weight_ptrs[w_offset+3];
            auto a = *ctx->weight_ptrs[w_offset+4], b = *ctx->weight_ptrs[w_offset+5];
            auto al = *ctx->weight_ptrs[w_offset+6], db = *ctx->weight_ptrs[w_offset+7];
            auto cw = *ctx->weight_ptrs[w_offset+8], nw = *ctx->weight_ptrs[w_offset+9];
            auto op = *ctx->weight_ptrs[w_offset+10];
            return linear_attention_batched(
                ctx, attn_input, layer_idx, qkv, z, a, b, al, db, cw, nw, op,
                cfg.num_k_heads, cfg.key_dim, cfg.num_v_heads, cfg.val_dim,
                cfg.conv_kernel, cfg.rms_eps, kind);
        }
    }

    // Legacy path: weight-level LoRA
    TORCH_CHECK(!ctx->base_tp_attention,
        "base attention TP requires the activation-level LoRA path");
    int64_t lora_count = lora_pair_count(cfg);
    int64_t la_offset = ctx->lora_layer_offset[layer_idx];
    bool has_lora = (la_offset + lora_count) <= (int64_t)ctx->lora_a.size();
    std::vector<at::Tensor*> la(lora_count, nullptr), lb(lora_count, nullptr);
    if (has_lora) for (int64_t k = 0; k < lora_count; k++) { la[k] = &ctx->lora_a[la_offset + k]; lb[k] = &ctx->lora_b[la_offset + k]; }

    if (cfg.layer_type == 0) {
        auto qp = *ctx->weight_ptrs[w_offset+2], qn = *ctx->weight_ptrs[w_offset+3];
        auto kp = *ctx->weight_ptrs[w_offset+4], kn = *ctx->weight_ptrs[w_offset+5];
        auto vp = *ctx->weight_ptrs[w_offset+6], op = *ctx->weight_ptrs[w_offset+7];
        if (has_lora) {
            if (la[0]) qp = lora_delta(qp, *la[0], *lb[0], ctx->lora_scaling);
            if (la[1]) kp = lora_delta(kp, *la[1], *lb[1], ctx->lora_scaling);
            if (la[2]) vp = lora_delta(vp, *la[2], *lb[2], ctx->lora_scaling);
            if (la[3]) op = lora_delta(op, *la[3], *lb[3], ctx->lora_scaling);
        }
        return full_attention(attn_input, qp, qn, kp, kn, vp, op,
            cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
            cfg.partial_rotary_factor, cfg.rope_theta, cfg.rms_eps, kind,
            ctx->attention_mask);
    } else {
        auto qkv = *ctx->weight_ptrs[w_offset+2], z = *ctx->weight_ptrs[w_offset+3];
        auto a = *ctx->weight_ptrs[w_offset+4], b = *ctx->weight_ptrs[w_offset+5];
        auto al = *ctx->weight_ptrs[w_offset+6], db = *ctx->weight_ptrs[w_offset+7];
        auto cw = *ctx->weight_ptrs[w_offset+8], nw = *ctx->weight_ptrs[w_offset+9];
        auto op = *ctx->weight_ptrs[w_offset+10];
        if (has_lora) {
            if (la[0]) qkv = lora_delta(qkv, *la[0], *lb[0], ctx->lora_scaling);
            if (la[1]) z = lora_delta(z, *la[1], *lb[1], ctx->lora_scaling);
            if (la[2]) a = lora_delta(a, *la[2], *lb[2], ctx->lora_scaling);
            if (la[3]) b = lora_delta(b, *la[3], *lb[3], ctx->lora_scaling);
            if (la[4]) op = lora_delta(op, *la[4], *lb[4], ctx->lora_scaling);
        }
        return linear_attention(attn_input, qkv, z, a, b, al, db, cw, nw, op,
            cfg.num_k_heads, cfg.key_dim, cfg.num_v_heads, cfg.val_dim,
            cfg.conv_kernel, cfg.rms_eps, kind);
    }
}

// Compute MLP output from residual (hidden + attn_output)
at::Tensor compute_mlp_only(
    TrainingContext* ctx, const at::Tensor& residual, int64_t layer_idx, at::ScalarType kind
) {
    const auto& cfg = ctx->layer_configs[layer_idx];
    int64_t w_offset = 0;
    for (int64_t j = 0; j < layer_idx; j++)
        w_offset += weight_count_for_layer(ctx->layer_configs[j]);
    int64_t mlp_start = (cfg.layer_type == 0) ? 8 : 11;
    auto post_attn = rms_norm(residual, *ctx->weight_ptrs[w_offset + 1], cfg.rms_eps);
    if (cfg.num_experts > 0) {
        const bool use_batched = ctx->lora_batch_valid;
        const int64_t shared_gate_pair = lora_pair_index(cfg, "shared_gate_proj");
        const int64_t shared_up_pair = lora_pair_index(cfg, "shared_up_proj");
        const int64_t shared_down_pair = lora_pair_index(cfg, "shared_down_proj");
        const int64_t expert_gate_up_pair = lora_pair_index(cfg, "experts_gate_up_proj");
        const int64_t expert_down_pair = lora_pair_index(cfg, "experts_down_proj");
        auto shared_gate = use_batched ? *ctx->weight_ptrs[w_offset+mlp_start+2]
            : apply_multi_lora(ctx, layer_idx, shared_gate_pair,
                *ctx->weight_ptrs[w_offset+mlp_start+2]);
        auto shared_up = use_batched ? *ctx->weight_ptrs[w_offset+mlp_start+3]
            : apply_multi_lora(ctx, layer_idx, shared_up_pair,
                *ctx->weight_ptrs[w_offset+mlp_start+3]);
        auto shared_down = use_batched ? *ctx->weight_ptrs[w_offset+mlp_start+4]
            : apply_multi_lora(ctx, layer_idx, shared_down_pair,
                *ctx->weight_ptrs[w_offset+mlp_start+4]);
        auto expert_lora = routed_expert_lora(ctx, layer_idx, cfg);
        return moe_forward(ctx, cfg.nccl_comm, cfg.nccl_stream, post_attn,
            *ctx->weight_ptrs[w_offset+mlp_start], *ctx->weight_ptrs[w_offset+mlp_start+1],
            shared_gate, shared_up,
            shared_down, *ctx->weight_ptrs[w_offset+mlp_start+5],
            *ctx->weight_ptrs[w_offset+mlp_start+6],
            expert_lora,
            cfg.num_experts, cfg.top_k, cfg.moe_intermediate,
            cfg.norm_topk_prob != 0, cfg.expert_start, cfg.expert_count, kind,
            use_batched ? lora_batch_entry(ctx, layer_idx, shared_gate_pair) : nullptr,
            use_batched ? lora_batch_entry(ctx, layer_idx, shared_up_pair) : nullptr,
            use_batched ? lora_batch_entry(ctx, layer_idx, shared_down_pair) : nullptr,
            use_batched ? lora_batch_entry(ctx, layer_idx, expert_gate_up_pair) : nullptr,
            use_batched ? lora_batch_entry(ctx, layer_idx, expert_down_pair) : nullptr);
    } else {
        if (ctx->lora_batch_valid) {
            return dense_mlp_forward_batched(
                ctx, layer_idx, post_attn,
                *ctx->weight_ptrs[w_offset+mlp_start],
                *ctx->weight_ptrs[w_offset+mlp_start+1],
                *ctx->weight_ptrs[w_offset+mlp_start+2], kind);
        }
        auto gate = apply_multi_lora(ctx, layer_idx,
            lora_pair_index(cfg, "gate_proj"), *ctx->weight_ptrs[w_offset+mlp_start]);
        auto up = apply_multi_lora(ctx, layer_idx,
            lora_pair_index(cfg, "up_proj"), *ctx->weight_ptrs[w_offset+mlp_start+1]);
        auto down = apply_multi_lora(ctx, layer_idx,
            lora_pair_index(cfg, "down_proj"), *ctx->weight_ptrs[w_offset+mlp_start+2]);
        auto mlp_input = tp_copy_base_mlp_input(ctx, post_attn);
        return tp_allreduce_base_mlp(
            ctx, dense_mlp_forward(mlp_input, gate, up, down, kind));
    }
}

// Sub-layer checkpoint: wraps a function call with no-grad forward + recompute on backward
struct SubLayerCkpt : public torch::autograd::Function<SubLayerCkpt> {
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor input, int64_t tc_val, int64_t layer_idx, bool is_attn
    ) {
        ctx->saved_data["tc"] = tc_val;
        ctx->saved_data["layer"] = layer_idx;
        ctx->saved_data["is_attn"] = is_attn;
        bool offload = getenv("QWEN36_OFFLOAD_ACTIVATIONS");
        if (offload) {
            ctx->saved_data["input_cpu"] = input.detach().to(
                at::TensorOptions().dtype(input.scalar_type()).device(at::kCPU).pinned_memory(true));
            ctx->saved_data["device"] = input.device();
        } else {
            ctx->save_for_backward({input});
        }
        at::AutoGradMode guard(false);
        auto* tc = reinterpret_cast<TrainingContext*>(tc_val);
        if (is_attn) {
            // Attn segment: returns hidden + attn_output (residual connection)
            return input + compute_attn_only(tc, input, layer_idx, tc->compute_type);
        } else {
            // MLP segment: returns residual + mlp_out (full layer output)
            return input + compute_mlp_only(tc, input, layer_idx, tc->compute_type);
        }
    }
    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx, std::vector<at::Tensor> grad_output
    ) {
        at::Tensor input;
        if (ctx->saved_data.count("input_cpu") > 0) {
            input = ctx->saved_data["input_cpu"].toTensor().to(ctx->saved_data["device"].toDevice());
        } else {
            input = ctx->get_saved_variables()[0];
        }
        auto* tc = reinterpret_cast<TrainingContext*>(ctx->saved_data["tc"].toInt());
        int64_t layer = ctx->saved_data["layer"].toInt();
        bool is_attn = ctx->saved_data["is_attn"].toBool();
        at::AutoGradMode guard(true);
        input.set_requires_grad(true);

        // Derived LoRA deltas belong to one recomputed segment's graph and are
        // released by grad(..., retain_graph=false). Attention and MLP both
        // need a fresh cache now that both segments can own adapters.
        tc->lora_batch_valid = false;
        tc->lora_cache_valid = false;
        if (!tc->adapters.empty()) prepare_lora_batch(tc);
        else if (tc->tp_world_size > 1) prepare_fixed_lora_batch(tc);
        else precompute_lora_cache(tc);
        auto output = is_attn ? compute_attn_only(tc, input, layer, tc->compute_type)
                              : compute_mlp_only(tc, input, layer, tc->compute_type);

        // Collect all tensors to compute gradients for: input + LoRA params for this layer
        // This way grad() accumulates gradients into LoRA params (leaf nodes) too.
        auto projection_table = lora_projection_table(tc->layer_configs[layer]);
        int64_t lora_count = projection_table.count;
        int64_t la_offset = tc->lora_layer_offset[layer];
        bool has_lora = (la_offset + lora_count) <= (int64_t)tc->lora_a.size();

        std::vector<at::Tensor> grad_inputs = {input};
        std::vector<std::pair<at::Tensor*, at::Tensor*>> active_params;
        if (!tc->adapters.empty()) {
            for (auto& adapter : tc->adapters) {
                auto it = adapter.params.find(layer);
                if (it == adapter.params.end()) continue;
                for (int64_t k = 0; k < lora_count && k < (int64_t)it->second.size(); ++k) {
                    auto segment = projection_table.entries[k].segment;
                    if ((is_attn && segment != LoraSegment::Attention) ||
                        (!is_attn && segment != LoraSegment::Mlp)) continue;
                    auto& [a, b] = it->second[k];
                    if (!a.requires_grad() && !b.requires_grad()) continue;
                    active_params.push_back({&a, &b});
                    grad_inputs.push_back(a);
                    grad_inputs.push_back(b);
                }
            }
        } else if (has_lora) {
            for (int64_t k = 0; k < lora_count; k++) {
                auto segment = projection_table.entries[k].segment;
                if ((is_attn && segment != LoraSegment::Attention) ||
                    (!is_attn && segment != LoraSegment::Mlp)) continue;
                int64_t slot = la_offset + k;
                if (!legacy_lora_slot_active(tc, slot)) continue;
                active_params.push_back({&tc->lora_a[slot], &tc->lora_b[slot]});
                grad_inputs.push_back(tc->lora_a[slot]);
                grad_inputs.push_back(tc->lora_b[slot]);
            }
        }

        auto grads = torch::autograd::grad(
            {output}, grad_inputs, {grad_output[0]},
            /*retain_graph=*/false, /*create_graph=*/false,
            /*allow_unused=*/true
        );

        // Manually accumulate LoRA param gradients
        if (!active_params.empty()) {
            int64_t gi = 1;  // skip input grad (index 0)
            for (auto& [param_a_ptr, param_b_ptr] : active_params) {
                if (grads[gi].defined()) {
                    auto& param_a = *param_a_ptr;
                    if (param_a.grad().defined())
                        param_a.grad().add_(grads[gi]);
                    else
                        param_a.mutable_grad() = grads[gi].clone();
                }
                gi++;
                if (grads[gi].defined()) {
                    auto& param_b = *param_b_ptr;
                    if (param_b.grad().defined())
                        param_b.grad().add_(grads[gi]);
                    else
                        param_b.mutable_grad() = grads[gi].clone();
                }
                gi++;
            }
        }

        return {grads[0], at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

// Forward a single layer with sub-layer checkpointing
// attn segment returns (hidden + attn_output) to avoid extra GPU tensor
// mlp segment returns (residual + mlp_out) = full layer output
at::Tensor forward_single_layer_subckpt(
    TrainingContext* ctx, const at::Tensor& hidden, int64_t layer_idx
) {
    // attn segment: computes attn_output, returns hidden + attn_output
    auto residual = SubLayerCkpt::apply(
        hidden, (int64_t)(uintptr_t)ctx, layer_idx, true);
    // mlp segment: computes mlp_out from residual, returns residual + mlp_out
    auto result = SubLayerCkpt::apply(
        residual, (int64_t)(uintptr_t)ctx, layer_idx, false);
    return result;
}

// ── Batched attention variants (activation-level LoRA) ──
// These wrap the base attention functions, applying LoRA delta as
// B@(A@x) * scaling on the projected outputs instead of modifying weights.

static at::Tensor full_attention_batched(
    TrainingContext* ctx, const at::Tensor& hidden, int64_t layer_idx,
    const at::Tensor& q_proj, const at::Tensor& q_norm,
    const at::Tensor& k_proj, const at::Tensor& k_norm,
    const at::Tensor& v_proj, const at::Tensor& o_proj,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    double partial_rotary_factor, double rope_theta,
    double rms_eps, at::ScalarType kind,
    const at::Tensor& attention_mask
) {
    // Compute Q/K/V with base weight, then add LoRA delta if present
    int64_t batch = hidden.size(0), seq = hidden.size(1);
    auto projection_input = tp_copy_base_attention_input(ctx, hidden);
    if (ctx->base_tp_attention) {
        TORCH_CHECK(num_heads % ctx->tp_world_size == 0 &&
                num_kv_heads % ctx->tp_world_size == 0,
            "full-attention heads must be divisible by TP_SIZE");
        num_heads /= ctx->tp_world_size;
        num_kv_heads /= ctx->tp_world_size;
    }
    int64_t qkv_dim = num_heads * head_dim;

    auto q = at::matmul(projection_input, q_proj.t());
    auto k = at::matmul(projection_input, k_proj.t());
    auto v = at::matmul(projection_input, v_proj.t());

    // Apply activation-level LoRA: q += B@(A@hidden) * scaling
    auto it_q = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 0));
    if (it_q != ctx->lora_batch_cache.end()) {
        q = q + lora_activation_delta(ctx, projection_input, it_q->second.a_stack,
            it_q->second.b_stack, it_q->second.scaling, it_q->second.layout);
    }
    auto it_k = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 1));
    if (it_k != ctx->lora_batch_cache.end()) {
        k = k + lora_activation_delta(ctx, projection_input, it_k->second.a_stack,
            it_k->second.b_stack, it_k->second.scaling, it_k->second.layout);
    }
    auto it_v = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 2));
    if (it_v != ctx->lora_batch_cache.end()) {
        v = v + lora_activation_delta(ctx, projection_input, it_v->second.a_stack,
            it_v->second.b_stack, it_v->second.scaling, it_v->second.layout);
    }

    // Reshape Q: [batch, seq, num_heads, head_dim*2] → split into q and gate
    q = q.view({batch, seq, num_heads, head_dim * 2});
    auto qk = q.chunk(2, -1);
    auto q_out = qk[0].transpose(1, 2);   // [batch, heads, seq, head_dim]
    auto gate = qk[1].transpose(1, 2);

    k = k.view({batch, seq, num_kv_heads, head_dim}).transpose(1, 2);
    v = v.view({batch, seq, num_kv_heads, head_dim}).transpose(1, 2);

    q_out = rms_norm(q_out, q_norm, rms_eps);
    k = rms_norm(k, k_norm, rms_eps);

    // Keep gate alive until after SDPA. The architecture applies it to the
    // attention value before the output projection.

    // RoPE
    int64_t rotary_dim = (int64_t)(head_dim * partial_rotary_factor);
    if (rotary_dim > 0) {
        auto device = hidden.device();
        auto pos = at::arange(seq, at::TensorOptions().dtype(at::kFloat).device(device)).unsqueeze(0);
        auto exponents = at::arange(0, rotary_dim, 2, at::TensorOptions().dtype(at::kFloat).device(device)) / (double)rotary_dim;
        auto inv_freq = (exponents * std::log(rope_theta)).exp().reciprocal();
        auto freqs = pos.unsqueeze(-1) * inv_freq.unsqueeze(0);
        auto emb = at::cat({freqs, freqs}, -1);
        auto cos = emb.cos().unsqueeze(1).to(q_out.scalar_type());
        auto sin = emb.sin().unsqueeze(1).to(q_out.scalar_type());
        auto q_rot = q_out.narrow(-1, 0, rotary_dim);
        auto k_rot = k.narrow(-1, 0, rotary_dim);
        auto rotate_half_q = at::cat({-q_rot.narrow(-1, rotary_dim/2, rotary_dim/2), q_rot.narrow(-1, 0, rotary_dim/2)}, -1);
        auto rotate_half_k = at::cat({-k_rot.narrow(-1, rotary_dim/2, rotary_dim/2), k_rot.narrow(-1, 0, rotary_dim/2)}, -1);
        q_rot.mul_(cos).add_(rotate_half_q * sin);
        k_rot.mul_(cos).add_(rotate_half_k * sin);
        cos = at::Tensor(); sin = at::Tensor();
    }

    // GQA: no K/V expansion needed (PT 2.5+ enable_gqa=true)
    double scale = 1.0 / std::sqrt((double)head_dim);

    // SDPA with GQA
    at::Tensor attn_out;
    if (attention_mask.defined() && attention_mask.numel() > 0) {
        auto kpm = attention_mask.to(at::kBool);
        while (kpm.dim() > 2) kpm = kpm.squeeze(0);
        if (kpm.size(0) == 1) {
            kpm = kpm.unsqueeze(1).unsqueeze(1).expand({batch, 1, 1, seq});
        } else {
            kpm = kpm.unsqueeze(1).unsqueeze(1);
        }
        // Combined causal + padding mask
        auto causal = at::triu(at::ones({seq, seq}, at::TensorOptions().dtype(at::kBool).device(q_out.device())), 1);
        causal = causal.unsqueeze(0).unsqueeze(0);  // [1, 1, S, S]
        auto pad_mask = kpm.logical_not();  // [B, 1, 1, S]
        auto combined = causal.logical_or(pad_mask);
        auto additive_mask = at::zeros({batch, 1, seq, seq}, at::TensorOptions().dtype(q_out.scalar_type()).device(q_out.device()));
        additive_mask = additive_mask.masked_fill(combined, -std::numeric_limits<float>::infinity());
        attn_out = at::scaled_dot_product_attention(q_out, k, v, additive_mask, 0.0, false, c10::nullopt, true);
    } else {
        attn_out = at::scaled_dot_product_attention(q_out, k, v, c10::nullopt, 0.0, true, c10::nullopt, true);
    }
    auto gated_attn = attn_out * at::sigmoid(gate).to(attn_out.scalar_type());
    auto attn_flat = gated_attn.transpose(1, 2).reshape({batch, seq, qkv_dim});
    auto result = attn_flat.matmul(o_proj.t());

    // Apply LoRA delta on o_proj output
    auto it_o = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 3));
    if (it_o != ctx->lora_batch_cache.end()) {
        result = result + lora_activation_delta(ctx, attn_flat,
            it_o->second.a_stack, it_o->second.b_stack, it_o->second.scaling,
            it_o->second.layout);
    }
    return tp_allreduce_base_attention(ctx, result);
}

static at::Tensor linear_attention_batched(
    TrainingContext* ctx, const at::Tensor& hidden, int64_t layer_idx,
    const at::Tensor& in_proj_qkv, const at::Tensor& in_proj_z,
    const at::Tensor& in_proj_a, const at::Tensor& in_proj_b,
    const at::Tensor& a_log, const at::Tensor& dt_bias,
    const at::Tensor& conv1d_w, const at::Tensor& norm_w, const at::Tensor& out_proj,
    int64_t num_k_heads, int64_t key_dim, int64_t num_v_heads, int64_t val_dim,
    int64_t conv_kernel, double rms_eps, at::ScalarType compute_type
) {
    // Full reimplementation of linear_attention non-chunked path with
    // activation-level LoRA delta on QKV, Z, and out_proj.
    int64_t batch = hidden.size(0), seq = hidden.size(1);
    auto projection_input = tp_copy_base_attention_input(ctx, hidden);
    if (base_tp_attention_enabled(ctx)) {
        TORCH_CHECK(num_k_heads > 0 && num_v_heads > 0 &&
                num_v_heads % num_k_heads == 0 &&
                num_k_heads % ctx->tp_world_size == 0 &&
                num_v_heads % ctx->tp_world_size == 0,
            "linear-attention heads must preserve value-head groups and be divisible by TP_SIZE");
        num_k_heads /= ctx->tp_world_size;
        num_v_heads /= ctx->tp_world_size;
    }
    int64_t q_size = num_k_heads * key_dim;
    int64_t v_size = num_v_heads * val_dim;
    int64_t qkv_dim = q_size * 2 + v_size;

    // QKV projection + LoRA delta
    auto qkv = at::matmul(projection_input, in_proj_qkv.t());
    auto it_qkv = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 0));
    if (it_qkv != ctx->lora_batch_cache.end()) {
        qkv = qkv + lora_activation_delta(ctx, projection_input, it_qkv->second.a_stack,
            it_qkv->second.b_stack, it_qkv->second.scaling, it_qkv->second.layout);
    }

    // DIAG: dump after QKV projection
    if (getenv("QWEN36_DUMP_LAYERS") && layer_idx == 0) {
        auto qkv_f = qkv.to(at::kFloat);
        fprintf(stderr, "[diag-la] layer %ld qkv_proj: shape=[%ld,%ld,%ld] mean=%.6f std=%.6f [0,0,:5]=%.6f,%.6f,%.6f,%.6f,%.6f\n",
                (long)layer_idx, (long)qkv_f.size(0), (long)qkv_f.size(1), (long)qkv_f.size(2),
                qkv_f.mean().item<float>(), qkv_f.std().item<float>(),
                qkv_f[0][0][0].item<float>(), qkv_f[0][0][1].item<float>(),
                qkv_f[0][0][2].item<float>(), qkv_f[0][0][3].item<float>(),
                qkv_f[0][0][4].item<float>());
        // Dump weight layout info
        auto w_f = in_proj_qkv.to(at::kFloat);
        fprintf(stderr, "[diag-la] in_proj_qkv weight: shape=[%ld,%ld] mean=%.6f std=%.6f\n",
                (long)w_f.size(0), (long)w_f.size(1), w_f.mean().item<float>(), w_f.std().item<float>());
        auto conv_w_f = conv1d_w.to(at::kFloat);
        fprintf(stderr, "[diag-la] conv1d weight: shape=[%ld,%ld,%ld] mean=%.6f std=%.6f\n",
                (long)conv_w_f.size(0), (long)conv_w_f.size(1), (long)conv_w_f.size(2),
                conv_w_f.mean().item<float>(), conv_w_f.std().item<float>());
    }

    auto qkv_t = qkv.transpose(1, 2);
    int64_t pad = conv_kernel - 1;
    auto padding = at::zeros({batch, qkv_dim, pad}, qkv.options());
    auto padded = at::cat({padding, qkv_t}, 2);
    auto conv_out = at::conv1d(padded, conv1d_w, /*bias=*/{},
        at::IntArrayRef({1}), at::IntArrayRef({0}), at::IntArrayRef({1}), qkv_dim);
    conv_out = at::silu(conv_out.narrow(2, 0, seq));
    auto qkv_conv = conv_out.transpose(1, 2);

    // DIAG: dump after conv1d
    if (getenv("QWEN36_DUMP_LAYERS") && layer_idx == 0) {
        auto qc_f = qkv_conv.to(at::kFloat);
        fprintf(stderr, "[diag-la] layer %ld after_conv1d: mean=%.6f std=%.6f [0,0,:5]=%.6f,%.6f,%.6f,%.6f,%.6f\n",
                (long)layer_idx, qc_f.mean().item<float>(), qc_f.std().item<float>(),
                qc_f[0][0][0].item<float>(), qc_f[0][0][1].item<float>(),
                qc_f[0][0][2].item<float>(), qc_f[0][0][3].item<float>(),
                qc_f[0][0][4].item<float>());
    }

    // Flat QKV split (matches transformers Qwen3_5GatedDeltaNet.forward)
    // in_proj_qkv outputs flat layout: [Q_all(2048) | K_all(2048) | V_all(4096)]
    // NOT per-head interleaved. This matches the non-batched path (line ~517).
    int64_t head_k_dim = key_dim;                        // 128 (already per-head)
    int64_t head_v_dim = val_dim;                        // 128 (already per-head)
    int64_t q_total = num_k_heads * head_k_dim;          // 2048
    int64_t v_total = num_v_heads * head_v_dim;          // 4096
    auto q = qkv_conv.narrow(-1, 0, q_total).reshape({batch, seq, num_k_heads, head_k_dim});
    auto k = qkv_conv.narrow(-1, q_total, q_total).reshape({batch, seq, num_k_heads, head_k_dim});
    auto v = qkv_conv.narrow(-1, q_total * 2, v_total).reshape({batch, seq, num_v_heads, head_v_dim});

    // DIAG: dump Q/K/V after per-head split
    if (getenv("QWEN36_DUMP_LAYERS") && layer_idx == 0) {
        auto q_f = q.to(at::kFloat);
        auto k_f = k.to(at::kFloat);
        auto v_f = v.to(at::kFloat);
        fprintf(stderr, "[diag-la] after_split q: mean=%.6f std=%.6f [0,0,0,:3]=%.6f,%.6f,%.6f\n",
                q_f.mean().item<float>(), q_f.std().item<float>(),
                q_f[0][0][0][0].item<float>(), q_f[0][0][0][1].item<float>(), q_f[0][0][0][2].item<float>());
        fprintf(stderr, "[diag-la] after_split k: mean=%.6f std=%.6f [0,0,0,:3]=%.6f,%.6f,%.6f\n",
                k_f.mean().item<float>(), k_f.std().item<float>(),
                k_f[0][0][0][0].item<float>(), k_f[0][0][0][1].item<float>(), k_f[0][0][0][2].item<float>());
        fprintf(stderr, "[diag-la] after_split v: mean=%.6f std=%.6f [0,0,0,:3]=%.6f,%.6f,%.6f\n",
                v_f.mean().item<float>(), v_f.std().item<float>(),
                v_f[0][0][0][0].item<float>(), v_f[0][0][0][1].item<float>(), v_f[0][0][0][2].item<float>());
    }

    at::Tensor a;
    at::Tensor b;
    const bool use_fused_ab = layer_idx >= 0 &&
        layer_idx < static_cast<int64_t>(ctx->fused_gdn_ab_weights.size()) &&
        ctx->fused_gdn_ab_weights[layer_idx].defined();
    if (use_fused_ab) {
        auto ab = at::matmul(
            projection_input, ctx->fused_gdn_ab_weights[layer_idx].t());
        TORCH_CHECK(ab.size(-1) == num_v_heads * 2,
            "fused GDN A/B projection output shape mismatch");
        a = ab.narrow(-1, 0, num_v_heads);
        b = ab.narrow(-1, num_v_heads, num_v_heads);
    } else {
        a = at::matmul(projection_input, in_proj_a.t());
        b = at::matmul(projection_input, in_proj_b.t());
    }
    auto it_a = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 2));
    if (it_a != ctx->lora_batch_cache.end()) {
        a = a + lora_activation_delta(ctx, projection_input, it_a->second.a_stack,
            it_a->second.b_stack, it_a->second.scaling, it_a->second.layout);
    }
    auto it_b = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 3));
    if (it_b != ctx->lora_batch_cache.end()) {
        b = b + lora_activation_delta(ctx, projection_input, it_b->second.a_stack,
            it_b->second.b_stack, it_b->second.scaling, it_b->second.layout);
    }

    // Z projection + LoRA delta
    auto z = at::matmul(projection_input, in_proj_z.t());
    auto it_z = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 1));
    if (it_z != ctx->lora_batch_cache.end()) {
        z = z + lora_activation_delta(ctx, projection_input, it_z->second.a_stack,
            it_z->second.b_stack, it_z->second.scaling, it_z->second.layout);
    }
    z = z.reshape({batch, seq, num_v_heads, head_v_dim});

    // g = -exp(A_log) * softplus(a + dt_bias)
    auto a_log_f = a_log.to(at::kFloat);
    auto dt_bias_f = dt_bias.to(at::kFloat);
    auto a_f = a.to(at::kFloat);
    auto g = a_log_f.unsqueeze(0).unsqueeze(0).exp().neg() * at::softplus(a_f + dt_bias_f.unsqueeze(0).unsqueeze(0));
    auto beta = at::sigmoid(b);

    // Expand Q/K to num_v_heads
    int64_t n_rep = num_v_heads / num_k_heads;
    q = q.repeat_interleave(n_rep, 2);
    k = k.repeat_interleave(n_rep, 2);

    // L2 normalize Q, K (per-head, matching HF)
    auto q_f = q.to(at::kFloat);
    auto k_f = k.to(at::kFloat);
    q = q_f * (q_f.pow(2).sum(-1, true) + 1e-6).rsqrt();
    k = k_f * (k_f.pow(2).sum(-1, true) + 1e-6).rsqrt();

    // DIAG: dump after L2 norm
    if (getenv("QWEN36_DUMP_LAYERS") && layer_idx == 0) {
        auto q_f = q.to(at::kFloat);
        auto k_f = k.to(at::kFloat);
        fprintf(stderr, "[diag-la] after_l2norm q: mean=%.6f std=%.6f [0,0,0,:3]=%.6f,%.6f,%.6f\n",
                q_f.mean().item<float>(), q_f.std().item<float>(),
                q_f[0][0][0][0].item<float>(), q_f[0][0][0][1].item<float>(), q_f[0][0][0][2].item<float>());
        fprintf(stderr, "[diag-la] after_l2norm k: mean=%.6f std=%.6f [0,0,0,:3]=%.6f,%.6f,%.6f\n",
                k_f.mean().item<float>(), k_f.std().item<float>(),
                k_f[0][0][0][0].item<float>(), k_f[0][0][0][1].item<float>(), k_f[0][0][0][2].item<float>());
        auto g_f = g.to(at::kFloat);
        fprintf(stderr, "[diag-la] g: mean=%.6f std=%.6f [0,0,:3]=%.6f,%.6f,%.6f\n",
                g_f.mean().item<float>(), g_f.std().item<float>(),
                g_f[0][0][0].item<float>(), g_f[0][0][1].item<float>(), g_f[0][0][2].item<float>());
        auto beta_f = beta.to(at::kFloat);
        fprintf(stderr, "[diag-la] beta: mean=%.6f std=%.6f [0,0,:3]=%.6f,%.6f,%.6f\n",
                beta_f.mean().item<float>(), beta_f.std().item<float>(),
                beta_f[0][0][0].item<float>(), beta_f[0][0][1].item<float>(), beta_f[0][0][2].item<float>());
    }

    double scale = 1.0 / std::sqrt((double)head_k_dim);
    q = q * scale;

    auto q_t = q.transpose(1, 2).contiguous();
    auto k_t = k.transpose(1, 2).contiguous();
    auto v_t = v.to(at::kFloat).transpose(1, 2).contiguous();
    auto g_t = g.transpose(1, 2).contiguous();
    auto beta_t = beta.to(at::kFloat).transpose(1, 2).contiguous();

    auto g_exp = g_t.exp();

    // N-aware sub-batching: process N adapters in groups of 256 to limit
    // state tensor size and improve SM occupancy.
    // State is independent per adapter — safe to split by N dimension.
    int64_t BH_total = batch * num_v_heads;
    int64_t sub_batch = (BH_total > 8192) ? 256 : batch;  // 256 adapters or all if small
    sub_batch = std::min(sub_batch, batch);

    std::vector<at::Tensor> sub_outputs;
    sub_outputs.reserve((batch + sub_batch - 1) / sub_batch);

    for (int64_t sb = 0; sb < batch; sb += sub_batch) {
        int64_t n = std::min(sub_batch, batch - sb);
        int64_t BH = n * num_v_heads;

        // Narrow on dim 0 (batch/adapter dimension), then reshape to [BH, seq, dim]
        auto q_sub = q_t.narrow(0, sb, n);
        auto k_sub = k_t.narrow(0, sb, n);
        auto v_sub = v_t.narrow(0, sb, n);
        auto g_sub = g_exp.narrow(0, sb, n);
        auto beta_sub = beta_t.narrow(0, sb, n);

        auto q_contig = q_sub.reshape({BH, seq, head_k_dim}).contiguous().to(at::kFloat);
        auto k_contig = k_sub.reshape({BH, seq, head_k_dim}).contiguous().to(at::kFloat);
        auto v_contig = v_sub.reshape({BH, seq, head_v_dim}).contiguous().to(at::kFloat);
        auto g_contig = g_sub.reshape({BH, seq}).contiguous().to(at::kFloat);
        auto beta_contig = beta_sub.reshape({BH, seq}).contiguous().to(at::kFloat);
        sub_outputs.push_back(GatedDeltaRuleFunction::apply(
            q_contig, k_contig, v_contig, g_contig, beta_contig));
    }

    auto outs = at::cat(sub_outputs, 0);

    auto core_out = outs.reshape({batch, num_v_heads, seq, head_v_dim})
                         .transpose(1, 2).to(compute_type);

    // DIAG: dump after delta rule
    if (getenv("QWEN36_DUMP_LAYERS") && layer_idx == 0) {
        auto co_f = core_out.to(at::kFloat);
        fprintf(stderr, "[diag-la] after_delta_rule: mean=%.6f std=%.6f [0,0,0,:3]=%.6f,%.6f,%.6f\n",
                co_f.mean().item<float>(), co_f.std().item<float>(),
                co_f[0][0][0][0].item<float>(), co_f[0][0][0][1].item<float>(), co_f[0][0][0][2].item<float>());
    }

    auto core_flat = core_out.reshape({-1, head_v_dim});
    auto z_flat = z.reshape({-1, head_v_dim});
    auto variance = core_flat.to(at::kFloat).pow(2).mean(-1, true);
    auto normed = (core_flat.to(at::kFloat) * (variance + rms_eps).rsqrt() * norm_w.to(at::kFloat)).to(core_flat.scalar_type());
    auto gated = (normed * at::silu(z_flat.to(at::kFloat)).to(normed.scalar_type())).reshape({batch, seq, num_v_heads * head_v_dim});
    auto result = at::matmul(gated, out_proj.t());

    // DIAG: dump after norm+gate+out_proj
    if (getenv("QWEN36_DUMP_LAYERS") && layer_idx == 0) {
        auto r_f = result.to(at::kFloat);
        fprintf(stderr, "[diag-la] after_out_proj: mean=%.6f std=%.6f [0,0,:3]=%.6f,%.6f,%.6f\n",
                r_f.mean().item<float>(), r_f.std().item<float>(),
                r_f[0][0][0].item<float>(), r_f[0][0][1].item<float>(), r_f[0][0][2].item<float>());
    }

    // out_proj LoRA delta: result += B@(A@gated) * scaling
    auto it_op = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 4));
    if (it_op != ctx->lora_batch_cache.end()) {
        result = result + lora_activation_delta(ctx, gated, it_op->second.a_stack,
            it_op->second.b_stack, it_op->second.scaling, it_op->second.layout);
    }

    return tp_allreduce_base_attention(ctx, result);
}
// ──────────────────────────────────────────────────────────────────────

static at::Tensor dense_mlp_forward(
    const at::Tensor& hidden,
    const at::Tensor& gate_proj, const at::Tensor& up_proj, const at::Tensor& down_proj,
    at::ScalarType compute_type
) {
    int64_t batch = hidden.size(0), seq = hidden.size(1), hidden_dim = hidden.size(2);
    auto flat = hidden.reshape({batch * seq, hidden_dim});
    auto gate_out = at::matmul(flat, gate_proj.t());
    auto up_out = at::matmul(flat, up_proj.t());
    // Fused silu * up via Tilelang (falls back to ATen)
    auto activated = fused_swiglu_op(gate_out, up_out, 0.0);  // Qwen3.6 dense MLP has no clamp
    return at::matmul(activated, down_proj.t()).reshape({batch, seq, hidden_dim});
}

// Forward pass (no checkpointing)
static at::Tensor forward_full(
    TrainingContext* ctx,
    const at::Tensor& input_ids
) {
    if (!ctx->adapters.empty()) prepare_lora_batch(ctx);
    else if (ctx->tp_world_size > 1) prepare_fixed_lora_batch(ctx);
    else precompute_lora_cache(ctx);
    auto kind = ctx->compute_type;
    at::AutoGradMode guard(true);
    at::Tensor hidden = vocabulary_embedding(ctx, input_ids);

    // Debug: dump embedding output stats
    if (getenv("QWEN36_DUMP_LAYERS")) {
        auto h_f = hidden.to(at::kFloat);
        fprintf(stderr, "[dump] embedding: mean=%.6f std=%.6f [0,:3]=%.6f,%.6f,%.6f\n",
                h_f.mean().item<float>(), h_f.std().item<float>(),
                h_f[0][0][0].item<float>(), h_f[0][0][1].item<float>(), h_f[0][0][2].item<float>());
    }

    for (int64_t i = 0; i < ctx->num_layers; i++) {
        // Get weight pointers for this layer
        int64_t w_offset = 0;
        for (int64_t j = 0; j < i; j++)
            w_offset += weight_count_for_layer(ctx->layer_configs[j]);
        int64_t w_count = weight_count_for_layer(ctx->layer_configs[i]);
        std::vector<at::Tensor*> layer_w(ctx->weight_ptrs.begin() + w_offset,
                                         ctx->weight_ptrs.begin() + w_offset + w_count);

        // Get LoRA pointers for this layer (nullptr if no LoRA for this layer)
        int64_t lora_count = lora_pair_count(ctx->layer_configs[i]);
        int64_t la_offset = ctx->lora_layer_offset[i];
        // Check if this layer has LoRA params (la_offset < lora_a.size())
        bool has_lora = (la_offset + lora_count) <= (int64_t)ctx->lora_a.size();
        std::vector<at::Tensor*> la_ptrs(lora_count, nullptr), lb_ptrs(lora_count, nullptr);
        if (has_lora) {
            for (int64_t k = 0; k < lora_count; k++) {
                la_ptrs[k] = &ctx->lora_a[la_offset + k];
                lb_ptrs[k] = &ctx->lora_b[la_offset + k];
            }
        }

        hidden = forward_single_layer(ctx, hidden, layer_w.data(), &ctx->layer_configs[i], i,
            kind, ctx->attention_mask, ctx->lora_batch_valid);

        // Debug: dump per-layer hidden state stats
        if (getenv("QWEN36_DUMP_LAYERS")) {
            auto h_f = hidden.to(at::kFloat);
            fprintf(stderr, "[dump] layer %ld: mean=%.6f std=%.6f [0,0,:3]=%.6f,%.6f,%.6f\n",
                    (long)i, h_f.mean().item<float>(), h_f.std().item<float>(),
                    h_f[0][0][0].item<float>(), h_f[0][0][1].item<float>(), h_f[0][0][2].item<float>());
        }

        // No per-layer sync — let CUDA pipeline run asynchronously.
        // emptyCache() here was the #1 cause of GPU underutilization (6% util).
    }

    return hidden;  // pre-norm hidden (for MTP)
}

// ──────────────────────────────────────────────────────────────────────
// Gradient checkpointing: per-group recomputation
// ──────────────────────────────────────────────────────────────────────

// Run a group of layers forward (with grad enabled, for recomputation)
static at::Tensor forward_layer_group(
    TrainingContext* ctx,
    const at::Tensor& input,
    int64_t start_layer,
    int64_t end_layer
) {
    auto kind = ctx->compute_type;
    at::Tensor hidden = input;

    // Normal path: full layer forward (sub-layer checkpointing is handled
    // in forward_full_checkpoint, not here)
    for (int64_t i = start_layer; i < end_layer; i++) {
        int64_t w_offset = 0;
        for (int64_t j = 0; j < i; j++)
            w_offset += weight_count_for_layer(ctx->layer_configs[j]);
        int64_t w_count = weight_count_for_layer(ctx->layer_configs[i]);
        std::vector<at::Tensor*> layer_w(ctx->weight_ptrs.begin() + w_offset,
                                         ctx->weight_ptrs.begin() + w_offset + w_count);

        int64_t lora_count = lora_pair_count(ctx->layer_configs[i]);
        int64_t la_offset = ctx->lora_layer_offset[i];
        bool has_lora = (la_offset + lora_count) <= (int64_t)ctx->lora_a.size();
        std::vector<at::Tensor*> la_ptrs(lora_count, nullptr), lb_ptrs(lora_count, nullptr);
        if (has_lora) {
            for (int64_t k = 0; k < lora_count; k++) {
                la_ptrs[k] = &ctx->lora_a[la_offset + k];
                lb_ptrs[k] = &ctx->lora_b[la_offset + k];
            }
        }

        hidden = forward_single_layer(ctx, hidden, layer_w.data(), &ctx->layer_configs[i], i,
            kind, ctx->attention_mask, ctx->lora_batch_valid);

        // Debug: dump per-layer hidden state stats (also in checkpoint recompute path)
        if (getenv("QWEN36_DUMP_LAYERS")) {
            auto h_f = hidden.to(at::kFloat);
            fprintf(stderr, "[dump] layer %ld: mean=%.6f std=%.6f [0,0,:3]=%.6f,%.6f,%.6f\n",
                    (long)i, h_f.mean().item<float>(), h_f.std().item<float>(),
                    h_f[0][0][0].item<float>(), h_f[0][0][1].item<float>(), h_f[0][0][2].item<float>());
        }
    }
    return hidden;
}

// autograd::Function for checkpointing a group of layers.
// Forward: run group WITHOUT grad (no intermediate activations stored).
// Backward: recompute group WITH grad, then backprop through recomputed graph.
struct GroupCheckpointFunction : public torch::autograd::Function<GroupCheckpointFunction> {
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor input,
        int64_t tc_val,
        int64_t start_layer,
        int64_t end_layer
    ) {
        ctx->saved_data["tc"] = tc_val;
        ctx->saved_data["start"] = start_layer;
        ctx->saved_data["end"] = end_layer;

        // Check if activation offload is enabled
        bool offload = getenv("QWEN36_OFFLOAD_ACTIVATIONS");

        if (offload) {
            // Save input to CPU — frees GPU memory between groups
            // We store the CPU copy in saved_data (not save_for_backward,
            // because save_for_backward would keep it on GPU)
            auto input_cpu = input.detach().to(at::TensorOptions().dtype(input.scalar_type()).device(at::kCPU).pinned_memory(true));
            ctx->saved_data["input_cpu"] = input_cpu;
            // Also store device for restoring later
            ctx->saved_data["device"] = input.device();
        } else {
            ctx->save_for_backward({input});
        }

        // Run forward in NO-GRAD mode — intermediate activations are NOT stored.
        at::AutoGradMode guard(false);
        auto* tc = reinterpret_cast<TrainingContext*>(tc_val);
        return forward_layer_group(tc, input, start_layer, end_layer);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output
    ) {
        bool offload = ctx->saved_data.count("input_cpu") > 0;

        at::Tensor input;
        if (offload) {
            // Restore input from CPU → GPU
            auto input_cpu = ctx->saved_data["input_cpu"].toTensor();
            auto device = ctx->saved_data["device"].toDevice();
            input = input_cpu.to(device);
        } else {
            auto saved = ctx->get_saved_variables();
            input = saved[0];
        }

        auto tc = reinterpret_cast<TrainingContext*>(ctx->saved_data["tc"].toInt());
        int64_t start_layer = ctx->saved_data["start"].toInt();
        int64_t end_layer = ctx->saved_data["end"].toInt();

        // Recompute forward WITH grad enabled — builds autograd graph for LoRA params.
        at::AutoGradMode guard(true);
        input.set_requires_grad(true);
        auto output = forward_layer_group(tc, input, start_layer, end_layer);

        // Backprop through recomputed graph.
        // retain_graph=false: each group's recomputed graph is independent.
        // LoRA param gradients accumulate via autograd's accumulator (leaf nodes).
        // The graph is freed immediately after backward — critical for memory.
        torch::autograd::backward({output}, {grad_output[0]},
            /*retain_graph=*/false, /*create_graph=*/false);
        return {input.grad(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

// ──────────────────────────────────────────────────────────────────────
// FusedLayerFunction: autograd::Function for single layer forward+backward.
// Forward: run WITH grad (PyTorch saves intermediates in graph).
// Backward: PyTorch autograd traverses graph — NO recompute needed.
// This eliminates checkpoint recompute (the main bottleneck).
// Controlled by QWEN36_FUSED_LAYER=1 env var.
// ──────────────────────────────────────────────────────────────────────

struct FusedLayerFunction : public torch::autograd::Function<FusedLayerFunction> {
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor input,
        int64_t tc_val,
        int64_t layer_idx
    ) {
        ctx->saved_data["tc"] = tc_val;
        ctx->saved_data["layer"] = layer_idx;
        // No save_for_backward — PyTorch autograd graph handles it.
        // Forward runs WITH grad — all intermediates saved in graph.
        auto* tc = reinterpret_cast<TrainingContext*>(tc_val);
        auto kind = tc->compute_type;

        int64_t w_offset = 0;
        for (int64_t j = 0; j < layer_idx; j++)
            w_offset += weight_count_for_layer(tc->layer_configs[j]);
        int64_t w_count = weight_count_for_layer(tc->layer_configs[layer_idx]);
        std::vector<at::Tensor*> layer_w(tc->weight_ptrs.begin() + w_offset,
                                         tc->weight_ptrs.begin() + w_offset + w_count);
        int64_t lora_count = lora_pair_count(tc->layer_configs[layer_idx]);
        int64_t la_offset = tc->lora_layer_offset[layer_idx];
        bool has_lora = (la_offset + lora_count) <= (int64_t)tc->lora_a.size();
        std::vector<at::Tensor*> la(lora_count, nullptr), lb(lora_count, nullptr);
        if (has_lora) for (int64_t k = 0; k < lora_count; k++) {
            la[k] = &tc->lora_a[la_offset + k]; lb[k] = &tc->lora_b[la_offset + k];
        }
        return forward_single_layer(tc, input, layer_w.data(), &tc->layer_configs[layer_idx],
            layer_idx, kind, tc->attention_mask, tc->lora_batch_valid);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output
    ) {
        // PyTorch autograd handles backward through the graph built during forward.
        // No recompute needed — just return grad_output as grad_input.
        // The actual backward computation happens via PyTorch's autograd engine
        // traversing the graph nodes (matmul backward, SDPA backward, etc.).
        return {grad_output[0], at::Tensor(), at::Tensor()};
    }
};

// Forward pass with fused layer (no checkpoint, no recompute).
// Uses FusedLayerFunction per layer — PyTorch autograd handles backward.
// QWEN36_FUSED_LAYER=1 enables this path.
static at::Tensor forward_full_fused(
    TrainingContext* ctx,
    const at::Tensor& input_ids
) {
    if (!ctx->adapters.empty()) prepare_lora_batch(ctx);
    else if (ctx->tp_world_size > 1) prepare_fixed_lora_batch(ctx);
    else precompute_lora_cache(ctx);
    at::Tensor hidden = vocabulary_embedding(ctx, input_ids);
    hidden = hidden.detach().set_requires_grad(true);

    for (int64_t i = 0; i < ctx->num_layers; i++) {
        hidden = FusedLayerFunction::apply(
            hidden,
            (int64_t)(uintptr_t)ctx,
            i
        );
    }

    return hidden;
}

// Forward pass with gradient checkpointing — manual checkpoint (no autograd::Function)
// Forward: no-grad, save group inputs (offloaded to CPU). Backward: manual recompute per group.
// This avoids autograd engine retaining all group outputs simultaneously.
static at::Tensor forward_full_checkpoint(
    TrainingContext* ctx,
    const at::Tensor& input_ids
) {
    // Use activation-level paths for dynamic adapters and LoRA TP; the
    // legacy weight cache remains the single-rank fast path.
    if (!ctx->adapters.empty()) {
        prepare_lora_batch(ctx);
    } else if (ctx->tp_world_size > 1) {
        prepare_fixed_lora_batch(ctx);
    } else {
        precompute_lora_cache(ctx);
    }
    at::Tensor hidden = vocabulary_embedding(ctx, input_ids);

    if (getenv("QWEN36_DUMP_LAYERS")) {
        auto h_f = hidden.to(at::kFloat);
        fprintf(stderr, "[dump] ckpt embedding: mean=%.6f std=%.6f [0,0,:3]=%.6f,%.6f,%.6f\n",
                h_f.mean().item<float>(), h_f.std().item<float>(),
                h_f[0][0][0].item<float>(), h_f[0][0][1].item<float>(), h_f[0][0][2].item<float>());
    }

    bool use_subckpt = env_enabled("QWEN36_SUBCKPT");

    if (use_subckpt) {
        at::AutoGradMode restore(true);
        hidden = hidden.detach().set_requires_grad(true);
        for (int64_t i = 0; i < ctx->num_layers; i++) {
            hidden = forward_single_layer_subckpt(ctx, hidden, i);
        }
        return hidden;
    }

    // Group-level manual checkpointing with variable group size.
    // Larger group_size = fewer recomputations in backward (faster) but more
    // peak memory. Default gs=4 (from ctx->group_size), overridable via env.
    at::AutoGradMode no_grad(false);
    hidden = hidden.detach();

    ctx->group_inputs.clear();
    bool offload = getenv("QWEN36_OFFLOAD_ACTIVATIONS");

    // Build group list using ctx->group_size (default 4).
    // Env override: QWEN36_GROUP_SIZE=10 sets gs=10.
    int64_t gs = ctx->group_size;
    if (gs < 1) gs = 1;
    const char* gs_env = getenv("QWEN36_GROUP_SIZE");
    if (gs_env) { gs = atol(gs_env); if (gs < 1) gs = 1; }
    fprintf(stderr, "[checkpoint] group_size=%ld (num_layers=%ld → %ld groups)\n",
            (long)gs, (long)ctx->num_layers,
            (long)((ctx->num_layers + gs - 1) / gs));

    std::vector<std::pair<int64_t, int64_t>> groups;
    for (int64_t i = 0; i < ctx->num_layers; i += gs) {
        groups.push_back({i, std::min(i + gs, ctx->num_layers)});
    }

    // Save groups for backward
    ctx->group_ranges = groups;

    for (auto& [start, end] : groups) {
        if (offload && start < groups.back().first) {
            ctx->group_inputs.push_back(
                hidden.to(at::TensorOptions().dtype(hidden.scalar_type()).device(at::kCPU).pinned_memory(true))
            );
        } else {
            ctx->group_inputs.push_back(hidden.clone());
        }

        hidden = forward_layer_group(ctx, hidden, start, end);

        if (getenv("QWEN36_DUMP_LAYERS")) {
            auto h_f = hidden.to(at::kFloat);
            fprintf(stderr, "[dump] ckpt group [%ld,%ld): mean=%.6f std=%.6f [0,0,:3]=%.6f,%.6f,%.6f\n",
                    (long)start, (long)end, h_f.mean().item<float>(), h_f.std().item<float>(),
                    h_f[0][0][0].item<float>(), h_f[0][0][1].item<float>(), h_f[0][0][2].item<float>());
        }
    }

    // Return hidden on GPU with requires_grad for CE backward
    at::AutoGradMode restore(true);
    hidden = hidden.set_requires_grad(true);
    return hidden;
}

// Manual sequential backward — recompute each group with grad, backprop, free.
// Only 1 group's intermediate tensors exist at any time.
static void manual_group_backward(
    TrainingContext* ctx,
    const at::Tensor& hidden_grad
) {
    auto& groups = ctx->group_ranges;
    int64_t num_groups = (int64_t)groups.size();
    at::Tensor grad = hidden_grad;
    at::AutoGradMode grad_mode(true);

    for (int64_t g = num_groups - 1; g >= 0; g--) {
        int64_t start = groups[g].first;
        int64_t end = groups[g].second;

        // Restore input from saved (CPU if offloaded)
        auto input = ctx->group_inputs[g].to(hidden_grad.device()).detach().set_requires_grad(true);

        // Each recomputed group builds a fresh autograd graph. Reusing a
        // differentiable LoRA cache after grad(..., retain_graph=false) would
        // point the next group at freed graph nodes.
        ctx->lora_batch_valid = false;
        ctx->lora_cache_valid = false;
        if (!ctx->adapters.empty()) prepare_lora_batch(ctx);
        else if (ctx->tp_world_size > 1) prepare_fixed_lora_batch(ctx);
        else precompute_lora_cache(ctx);

        // Recompute forward with grad for this group only
        auto output = forward_layer_group(ctx, input, start, end);

        // Backprop through this group using grad() instead of backward().
        // grad() only computes gradients for specified inputs — faster than
        // backward() which traverses all leaf nodes.
        // LoRA params are shared across groups, so we accumulate their gradients.
        std::vector<at::Tensor> grad_inputs = {input};

        if (!ctx->adapters.empty()) {
            // Multi-LoRA: collect A/B from ctx->adapters
            for (int64_t l = start; l < end; l++) {
                int64_t lora_count = lora_pair_count(ctx->layer_configs[l]);
                for (auto& adapter : ctx->adapters) {
                    auto it = adapter.params.find(l);
                    if (it == adapter.params.end()) continue;
                    for (int64_t k = 0; k < lora_count && k < (int64_t)it->second.size(); k++) {
                        auto& [a, b] = it->second[k];
                        if (!a.requires_grad() && !b.requires_grad()) continue;
                        grad_inputs.push_back(a);
                        grad_inputs.push_back(b);
                    }
                }
            }
        } else {
            // Legacy single-LoRA
            for (int64_t l = start; l < end; l++) {
                int64_t lora_count = lora_pair_count(ctx->layer_configs[l]);
                int64_t la_offset = ctx->lora_layer_offset[l];
                bool has_lora = (la_offset + lora_count) <= (int64_t)ctx->lora_a.size();
                if (has_lora) {
                    for (int64_t k = 0; k < lora_count; k++) {
                        if (!legacy_lora_slot_active(ctx, la_offset + k)) continue;
                        grad_inputs.push_back(ctx->lora_a[la_offset + k]);
                        grad_inputs.push_back(ctx->lora_b[la_offset + k]);
                    }
                }
            }
        }

        auto grads = torch::autograd::grad(
            {output}, grad_inputs, {grad},
            /*retain_graph=*/false, /*create_graph=*/false,
            /*allow_unused=*/true
        );


        // Manually accumulate LoRA param gradients
        if (!ctx->adapters.empty()) {
            // Multi-LoRA: accumulate into ctx->adapters
            int64_t gi = 1;  // skip input grad (index 0)
            for (int64_t l = start; l < end; l++) {
                int64_t lora_count = lora_pair_count(ctx->layer_configs[l]);
                for (auto& adapter : ctx->adapters) {
                    auto it = adapter.params.find(l);
                    if (it == adapter.params.end()) continue;
                    for (int64_t k = 0; k < lora_count && k < (int64_t)it->second.size(); k++) {
                        auto& [a, b] = it->second[k];
                        if (!a.requires_grad() && !b.requires_grad()) continue;
                        if (gi < (int64_t)grads.size() && grads[gi].defined()) {
                            if (a.grad().defined()) a.grad().add_(grads[gi]);
                            else a.mutable_grad() = grads[gi].clone();
                        }
                        gi++;
                        if (gi < (int64_t)grads.size() && grads[gi].defined()) {
                            if (b.grad().defined()) b.grad().add_(grads[gi]);
                            else b.mutable_grad() = grads[gi].clone();
                        }
                        gi++;
                    }
                }
            }
        } else {
            // Legacy single-LoRA
            int64_t gi = 1;  // skip input grad (index 0)
            for (int64_t l = start; l < end; l++) {
                int64_t lora_count = lora_pair_count(ctx->layer_configs[l]);
                int64_t la_offset = ctx->lora_layer_offset[l];
                bool has_lora = (la_offset + lora_count) <= (int64_t)ctx->lora_a.size();
                if (has_lora) {
                    for (int64_t k = 0; k < lora_count; k++) {
                        if (!legacy_lora_slot_active(ctx, la_offset + k)) continue;
                        if (grads[gi].defined()) {
                            auto& pa = ctx->lora_a[la_offset + k];
                            if (pa.grad().defined()) pa.grad().add_(grads[gi]);
                            else pa.mutable_grad() = grads[gi].clone();
                        }
                        gi++;
                        if (grads[gi].defined()) {
                            auto& pb = ctx->lora_b[la_offset + k];
                            if (pb.grad().defined()) pb.grad().add_(grads[gi]);
                            else pb.mutable_grad() = grads[gi].clone();
                        }
                        gi++;
                    }
                }
            }
        }

        // Gradient for this group's input = gradient for next group's output
        grad = grads[0];

        // Free saved input to release memory
        ctx->group_inputs[g] = at::Tensor();
    }
}

// ──────────────────────────────────────────────────────────────────────
// Fused Cross-Entropy Loss with online softmax (FlashAttention-style).
//
// Instead of materializing [n_tokens, vocab] logits (8+ GB), we tile over
// the vocabulary dimension:
//   1. Forward: iterate vocab tiles, compute partial logits, accumulate
//      global max + sum_exp via online softmax → loss
//   2. Backward: iterate vocab tiles again, compute softmax = exp(logit-max)/sum_exp,
//      subtract one_hot for target tokens → accumulate grad into hidden_normed
//
// Peak memory: [n_tokens, tile_size] instead of [n_tokens, vocab].
// For N=100, seq=512: n_tokens=51200, vocab=248320, tile=8192
//   Old: [51200, 248320] × 4 bytes = 49 GB per chunk
//   New: [51200, 8192]  × 4 bytes = 1.6 GB per tile
//
// Returns: scalar loss tensor (with autograd graph for hidden_normed)
// ──────────────────────────────────────────────────────────────────────
struct LossResult {
    at::Tensor value;
    at::Tensor hidden_grad;
};

// Vocabulary-parallel cross entropy. The first LM-head pass computes local
// online-softmax statistics and target logits. TP MAX/SUM reductions produce
// global statistics; the backward projection reuses cached local logits when
// they fit under a bounded workspace cap, otherwise it recomputes them.
static LossResult compute_vocab_parallel_loss(
    TrainingContext* ctx,
    const at::Tensor& hidden,
    const at::Tensor& input_ids,
    const at::Tensor& target_mask,
    bool independent_samples,
    bool compute_hidden_grad
) {
    TORCH_CHECK(vocab_parallel_enabled(ctx),
        "distributed vocabulary loss requires vocabulary TP");
    TORCH_CHECK(ctx->tp_comm, "distributed vocabulary loss requires a TP communicator");

    auto final_norm = *ctx->final_norm_ptr[0];
    auto lm_head = *ctx->lm_head_ptr[0];
    auto hidden_detached = hidden.detach();
    if (compute_hidden_grad) hidden_detached.set_requires_grad(true);

    at::Tensor hidden_normed;
    {
        at::AutoGradMode no_grad(false);
        hidden_normed = rms_norm(hidden_detached, final_norm, ctx->rms_eps);
    }

    const int64_t batch_size = hidden_normed.size(0);
    const int64_t seq_len = hidden_normed.size(1);
    const int64_t hidden_dim = hidden_normed.size(2);
    const int64_t shifted_seq_len = seq_len - 1;
    auto hidden_flat = hidden_normed.narrow(1, 0, shifted_seq_len)
        .reshape({-1, hidden_dim}).contiguous();
    auto shifted_targets = input_ids.narrow(1, 1, shifted_seq_len).reshape({-1});
    auto shifted_mask = target_mask.narrow(1, 1, shifted_seq_len)
        .reshape({-1}).to(at::kFloat);
    const int64_t total_tokens = shifted_targets.size(0);

    int64_t token_tile = 512;
    if (const char* value = getenv("QWEN36_CE_TOKEN_TILE")) {
        token_tile = std::max<int64_t>(1, std::atoll(value));
    }
    int64_t vocab_tile = ctx->local_vocab_size;
    if (const char* value = getenv("QWEN36_CE_TILE")) {
        vocab_tile = std::max<int64_t>(1, std::atoll(value));
    }
    int64_t logits_cache_bytes = 512LL * 1024LL * 1024LL;
    if (const char* value = getenv("QWEN36_CE_LOGITS_CACHE_BYTES")) {
        logits_cache_bytes = std::max<int64_t>(0, std::atoll(value));
    }

    auto total_count = shifted_mask.sum().clamp_min(1.0);
    at::Tensor token_denominators;
    if (independent_samples) {
        auto per_sample_count = target_mask.narrow(1, 1, shifted_seq_len)
            .sum(1, true).clamp_min(1.0);
        token_denominators = per_sample_count
            .expand({batch_size, shifted_seq_len}).reshape({-1}).to(at::kFloat);
    }

    auto total_loss = at::zeros({1},
        at::TensorOptions().dtype(at::kFloat).device(hidden.device()));
    at::Tensor grad_hidden_flat;
    if (compute_hidden_grad) {
        grad_hidden_flat = at::empty({total_tokens, hidden_dim},
            at::TensorOptions().dtype(at::kFloat).device(hidden.device()));
    }

    at::AutoGradMode no_grad(false);
    const int64_t vocab_start = ctx->tp_rank * ctx->local_vocab_size;
    const int64_t num_token_tiles =
        (total_tokens + token_tile - 1) / token_tile;
    const int64_t num_vocab_tiles =
        (ctx->local_vocab_size + vocab_tile - 1) / vocab_tile;

    for (int64_t token_index = 0; token_index < num_token_tiles; ++token_index) {
        const int64_t token_start = token_index * token_tile;
        const int64_t token_count = std::min(
            token_tile, total_tokens - token_start);
        auto chunk_hidden = hidden_flat.narrow(0, token_start, token_count);
        auto chunk_targets = shifted_targets.narrow(0, token_start, token_count);
        auto chunk_mask = shifted_mask.narrow(0, token_start, token_count);

        auto local_max = at::full({token_count, 1},
            -std::numeric_limits<float>::infinity(),
            at::TensorOptions().dtype(at::kFloat).device(hidden.device()));
        auto local_sum_exp = at::zeros_like(local_max);
        auto local_target_logit = at::zeros_like(local_max);
        const bool cache_logits =
            token_count * ctx->local_vocab_size * static_cast<int64_t>(sizeof(float)) <=
            logits_cache_bytes;
        std::vector<at::Tensor> cached_logits;
        if (cache_logits) cached_logits.reserve(num_vocab_tiles);

        for (int64_t vocab_index = 0; vocab_index < num_vocab_tiles; ++vocab_index) {
            const int64_t local_start = vocab_index * vocab_tile;
            const int64_t local_count = std::min(
                vocab_tile, ctx->local_vocab_size - local_start);
            const int64_t global_start = vocab_start + local_start;
            const int64_t global_end = global_start + local_count;
            auto head_tile = lm_head.narrow(0, local_start, local_count);
            auto logits = at::matmul(chunk_hidden, head_tile.t()).to(at::kFloat);
            if (cache_logits) cached_logits.push_back(logits);

            auto tile_max = std::get<0>(at::max(logits, 1, true));
            auto new_max = at::max(local_max, tile_max);
            local_sum_exp = at::exp(local_max - new_max) * local_sum_exp +
                at::exp(logits - new_max).sum(1, true);
            local_max = new_max;

            auto in_range = (chunk_targets >= global_start) &
                (chunk_targets < global_end);
            auto local_targets = (chunk_targets - global_start)
                .clamp(0, local_count - 1).reshape({-1, 1});
            auto gathered = at::gather(logits, 1, local_targets);
            local_target_logit.add_(at::where(
                in_range.reshape({-1, 1}), gathered, at::zeros_like(gathered)));
        }

        auto global_max = tp_allreduce_value(ctx, local_max, ncclMax);
        local_sum_exp.mul_(at::exp(local_max - global_max));
        auto global_stats = tp_allreduce_value(
            ctx, at::cat({local_sum_exp, local_target_logit}, 1), ncclSum);
        auto global_sum_exp = global_stats.narrow(1, 0, 1);
        auto target_logit = global_stats.narrow(1, 1, 1);
        auto per_token_loss =
            (at::log(global_sum_exp) + global_max - target_logit).squeeze(1);
        auto masked_loss = per_token_loss * chunk_mask;
        if (independent_samples) {
            total_loss.add_((masked_loss /
                token_denominators.narrow(0, token_start, token_count)).sum());
        } else {
            total_loss.add_(masked_loss.sum() / total_count);
        }

        if (!compute_hidden_grad) continue;

        auto grad_scale = independent_samples
            ? chunk_mask /
                token_denominators.narrow(0, token_start, token_count)
            : chunk_mask / total_count;
        auto local_grad_hidden = at::zeros({token_count, hidden_dim},
            at::TensorOptions().dtype(at::kFloat).device(hidden.device()));
        for (int64_t vocab_index = 0; vocab_index < num_vocab_tiles; ++vocab_index) {
            const int64_t local_start = vocab_index * vocab_tile;
            const int64_t local_count = std::min(
                vocab_tile, ctx->local_vocab_size - local_start);
            const int64_t global_start = vocab_start + local_start;
            const int64_t global_end = global_start + local_count;
            auto head_tile = lm_head.narrow(0, local_start, local_count);
            auto logits = cache_logits
                ? cached_logits[vocab_index]
                : at::matmul(chunk_hidden, head_tile.t()).to(at::kFloat);
            auto grad_logits = at::exp(logits - global_max) / global_sum_exp;

            auto in_range = (chunk_targets >= global_start) &
                (chunk_targets < global_end);
            auto local_targets = (chunk_targets - global_start)
                .clamp(0, local_count - 1).reshape({-1, 1});
            auto one_hot = at::zeros_like(grad_logits);
            one_hot.scatter_(1, local_targets, 1.0);
            grad_logits.sub_(one_hot * in_range.to(at::kFloat).reshape({-1, 1}));
            grad_logits.mul_(grad_scale.reshape({-1, 1}));
            local_grad_hidden.add_(at::matmul(
                grad_logits.to(head_tile.scalar_type()), head_tile).to(at::kFloat));
        }
        auto global_grad_hidden = tp_allreduce_value(
            ctx, local_grad_hidden, ncclSum);
        grad_hidden_flat.narrow(0, token_start, token_count)
            .copy_(global_grad_hidden);
    }

    at::Tensor hidden_grad;
    if (compute_hidden_grad) {
        auto grad_shifted = grad_hidden_flat
            .reshape({batch_size, shifted_seq_len, hidden_dim})
            .to(hidden_normed.scalar_type());
        auto grad_full = at::cat({
            grad_shifted,
            at::zeros({batch_size, 1, hidden_dim},
                at::TensorOptions().dtype(hidden_normed.scalar_type())
                    .device(hidden_normed.device()))
        }, 1);
        at::AutoGradMode grad_mode(true);
        auto hidden_normed_recompute = rms_norm(
            hidden_detached, final_norm, ctx->rms_eps);
        hidden_grad = torch::autograd::grad(
            {hidden_normed_recompute}, {hidden_detached}, {grad_full},
            /*retain_graph=*/false, /*create_graph=*/false,
            /*allow_unused=*/false)[0];
    }

    return {total_loss, hidden_grad};
}

static at::Tensor compute_loss_fused(
    TrainingContext* ctx,
    const at::Tensor& hidden,       // [batch, seq, hidden] (requires_grad)
    const at::Tensor& input_ids,    // [batch, seq]
    const at::Tensor& target_mask,  // [batch, seq]
    int64_t vocab_size
) {
    if (vocab_parallel_enabled(ctx)) {
        return compute_vocab_parallel_loss(
            ctx, hidden, input_ids, target_mask,
            /*independent_samples=*/false,
            /*compute_hidden_grad=*/false).value;
    }
    auto final_norm = *ctx->final_norm_ptr[0];
    auto lm_head = *ctx->lm_head_ptr[0];  // [vocab, hidden]

    // Compute hidden_normed in no-grad, then set requires_grad.
    auto hidden_detached = hidden.detach();
    at::Tensor hidden_normed;
    {
        at::AutoGradMode no_grad_mode(false);
        hidden_normed = rms_norm(hidden_detached, final_norm, ctx->rms_eps);
    }
    hidden_normed.set_requires_grad(true);

    int64_t seq_len = hidden_normed.size(1);
    int64_t hidden_dim = hidden_normed.size(2);

    auto shifted_hidden = hidden_normed.narrow(1, 0, seq_len - 1);
    auto shifted_targets = input_ids.narrow(1, 1, seq_len - 1).reshape({-1});
    auto shifted_mask = target_mask.narrow(1, 1, seq_len - 1).reshape({-1});

    int64_t total_tokens = shifted_targets.size(0);
    auto hidden_flat = shifted_hidden.reshape({-1, hidden_dim});

    // Ensure hidden_flat is contiguous for narrow + matmul
    hidden_flat = hidden_flat.contiguous();

    auto mask_f = shifted_mask.to(at::kFloat);

    // ── Tile configuration ──
    // tile_size controls vocab granularity. Larger = fewer iterations but more memory.
    // [total_tokens, tile] × 4 bytes (FP32). 8192 → ~1.6 GB for 51200 tokens.
    int64_t tile_size = 8192;
    const char* ts_env = getenv("QWEN36_CE_TILE");
    if (ts_env) { tile_size = atol(ts_env); if (tile_size < 1) tile_size = 8192; }
    int64_t num_tiles = (vocab_size + tile_size - 1) / tile_size;

    // lm_head transpose: we need [hidden, vocab] for matmul
    // lm_head is [vocab, hidden], so lm_head.t() is [hidden, vocab]
    // We narrow on dim 0 of lm_head (vocab dim), then transpose.
    // lm_head_w: [tile, hidden] → .t() → [hidden, tile]
    // hidden_flat: [total_tokens, hidden] × [hidden, tile] → [total_tokens, tile]

    // ── Forward pass: compute loss via online softmax ──
    // For each token i: loss_i = log(sum_v exp(logit_iv)) - logit_i,target_i
    // We compute in two phases:
    //   Phase 1: find global max and sum_exp across all vocab tiles
    //   Phase 2: compute loss = log(sum_exp) - target_logit (for masked tokens)

    // Running max and sum_exp per token
    auto logit_max = at::full({total_tokens, 1}, -std::numeric_limits<float>::infinity(),
        at::TensorOptions().dtype(at::kFloat).device(hidden_flat.device()));
    auto sum_exp = at::zeros({total_tokens, 1},
        at::TensorOptions().dtype(at::kFloat).device(hidden_flat.device()));

    // Phase 1: accumulate max and sum_exp
    for (int64_t t = 0; t < num_tiles; t++) {
        int64_t v_start = t * tile_size;
        int64_t v_end = std::min(v_start + tile_size, vocab_size);
        int64_t v_n = v_end - v_start;

        auto lm_head_tile = lm_head.narrow(0, v_start, v_n);  // [v_n, hidden]
        // [total_tokens, hidden] × [hidden, v_n] → [total_tokens, v_n]
        auto logits_tile = at::matmul(hidden_flat, lm_head_tile.t()).to(at::kFloat);

        // Online softmax update: new_max = max(old_max, tile_max)
        auto tile_max = std::get<0>(at::max(logits_tile, /*dim=*/1, /*keepdim=*/true));
        auto new_max = at::max(logit_max, tile_max);

        // Adjust sum_exp: exp(old - new_max) * old_sum + exp(tile - new_max) * tile_sum
        auto old_exp = at::exp(logit_max - new_max);
        auto tile_exp = at::exp(logits_tile - new_max);
        sum_exp = old_exp * sum_exp + tile_exp.sum(/*dim=*/1, /*keepdim=*/true);
        logit_max = new_max;

    }

    // Phase 2: compute target logit for each token
    // We need logits[target] — gather the target position's logit.
    // Do a second pass to find target logit.
    auto target_logit = at::full({total_tokens, 1},
        -std::numeric_limits<float>::infinity(),
        at::TensorOptions().dtype(at::kFloat).device(hidden_flat.device()));

    for (int64_t t = 0; t < num_tiles; t++) {
        int64_t v_start = t * tile_size;
        int64_t v_end = std::min(v_start + tile_size, vocab_size);
        int64_t v_n = v_end - v_start;

        // Check if any target falls in this tile's vocab range
        auto in_range = (shifted_targets >= v_start) & (shifted_targets < v_end);
        if (!in_range.any().item<bool>()) continue;

        auto lm_head_tile = lm_head.narrow(0, v_start, v_n);
        auto logits_tile = at::matmul(hidden_flat, lm_head_tile.t()).to(at::kFloat);

        // Gather target logits: subtract v_start to get local index
        auto local_targets = (shifted_targets - v_start).clamp(0, v_n - 1);
        // Gather: logits_tile[i, local_targets[i]] for tokens in range
        auto gathered = at::gather(logits_tile, /*dim=*/1, local_targets.reshape({-1, 1}));
        // Only keep tokens that are actually in range
        gathered = at::where(
            in_range.reshape({-1, 1}), gathered,
            at::full_like(gathered, -std::numeric_limits<float>::infinity()));
        target_logit = at::max(target_logit, gathered);

    }

    // Loss per token: log(sum_exp) - target_logit  (= -log(softmax[target]))
    auto log_sum_exp = at::log(sum_exp) + logit_max;  // log(sum(exp(x-max))) + max = logsumexp
    auto per_token_loss = log_sum_exp - target_logit;  // [total_tokens, 1]
    per_token_loss = at::where(
        mask_f.reshape({-1, 1}) > 0, per_token_loss,
        at::zeros_like(per_token_loss));
    auto masked_loss = per_token_loss.squeeze(1) * mask_f;
    auto total_count = mask_f.sum().clamp_min(1.0);
    double loss_val = (masked_loss.sum().item<double>()) / total_count.item<double>();

    // Evaluation runs with AutoGradMode disabled. The online-softmax passes
    // above already produced the exact scalar value, so do not execute the
    // training-only manual gradient pass below. Besides fixing eval on a
    // no-grad graph, this avoids a third traversal of the vocabulary tiles.
    if (!at::GradMode::is_enabled()) {
        return at::tensor({loss_val},
            at::TensorOptions().dtype(at::kFloat).device(hidden.device()));
    }

    // ── Backward pass: compute grad_hidden_normed manually ──
    // dL/dhidden_normed = (softmax - one_hot) / count * mask
    // softmax = exp(logit - logit_max) / sum_exp
    // We iterate tiles again, accumulate grad = softmax_tile @ lm_head_tile + target_grad
    auto grad_hidden = at::zeros({total_tokens, hidden_dim},
        at::TensorOptions().dtype(at::kFloat).device(hidden_flat.device()));
    auto grad_scale = mask_f / total_count;  // [total_tokens]

    for (int64_t t = 0; t < num_tiles; t++) {
        int64_t v_start = t * tile_size;
        int64_t v_end = std::min(v_start + tile_size, vocab_size);
        int64_t v_n = v_end - v_start;

        auto lm_head_tile = lm_head.narrow(0, v_start, v_n);  // [v_n, hidden]
        auto logits_tile = at::matmul(hidden_flat, lm_head_tile.t()).to(at::kFloat);

        // softmax_tile = exp(logits - max) / sum_exp  (reuse from forward)
        auto softmax_tile = at::exp(logits_tile - logit_max) / sum_exp;  // [total_tokens, v_n]

        // Subtract one_hot for target tokens in this tile
        auto in_range = (shifted_targets >= v_start) & (shifted_targets < v_end);
        if (in_range.any().item<bool>()) {
            auto local_targets = (shifted_targets - v_start).clamp_min(0);
            // scatter -1 at target positions
            auto one_hot = at::zeros({total_tokens, v_n},
                at::TensorOptions().dtype(at::kFloat).device(hidden_flat.device()));
            one_hot.scatter_(1, local_targets.reshape({-1, 1}), 1.0);
            one_hot = one_hot * in_range.to(at::kFloat).reshape({-1, 1});
            softmax_tile = softmax_tile - one_hot;
        }

        // grad_hidden += grad_scale * softmax_tile @ lm_head_tile
        // softmax_tile: [total_tokens, v_n] (Float), lm_head_tile: [v_n, hidden] (BF16)
        // → [total_tokens, hidden] (Float)
        auto grad_tile = at::matmul(
            (softmax_tile * grad_scale.reshape({-1, 1})),
            lm_head_tile.to(at::kFloat)
        );
        grad_hidden.add_(grad_tile);

    }

    // Set gradient on hidden_normed (leaf tensor).
    // grad_hidden covers [batch, seq-1, hidden] (shifted tokens).
    // hidden_normed is [batch, seq, hidden] — the final position has no target.
    auto grad_reshaped = grad_hidden.to(hidden_normed.scalar_type())
        .reshape({hidden_normed.size(0), seq_len - 1, hidden_dim});
    auto grad_full = at::cat({
        grad_reshaped,
        at::zeros({hidden_normed.size(0), 1, hidden_dim},
            at::TensorOptions().dtype(hidden_normed.scalar_type()).device(hidden_normed.device()))
    }, /*dim=*/1);  // [batch, seq, hidden]
    hidden_normed.mutable_grad() = grad_full;

    // Backprop hidden_normed gradient to hidden via rms_norm recompute.
    if (hidden_normed.grad().defined()) {
        hidden.set_requires_grad(true);
        auto hidden_normed_recompute = rms_norm(hidden, final_norm, ctx->rms_eps);
        hidden_normed_recompute.backward(hidden_normed.grad());
    }

    return at::tensor({loss_val},
        at::TensorOptions().dtype(at::kFloat).device(hidden.device()));
}

// Cross-entropy loss with response-only masking — chunked with detach.
// Return the hidden gradient instead of mutating/backpropagating through the
// main model graph. The caller combines auxiliary gradients and performs one
// main backward, which is required for checkpointed execution.
static LossResult compute_loss(
    TrainingContext* ctx,
    const at::Tensor& hidden,
    const at::Tensor& input_ids,
    const at::Tensor& target_mask,
    int64_t vocab_size,
    bool independent_samples = false
) {
    if (vocab_parallel_enabled(ctx)) {
        return compute_vocab_parallel_loss(
            ctx, hidden, input_ids, target_mask,
            independent_samples, /*compute_hidden_grad=*/true);
    }
    auto final_norm = *ctx->final_norm_ptr[0];
    auto lm_head = *ctx->lm_head_ptr[0];

    // Detach hidden for CE computation — CE backward won't touch main model graph.
    // We accumulate gradient into hidden_normed, then manually backprop to hidden.
    auto hidden_detached = hidden.detach().set_requires_grad(true);

    // Compute hidden_normed in no-grad, then set requires_grad on it.
    // This way CE backward only builds a tiny graph (hidden_normed → logits → loss),
    // not connected to hidden_detached at all.
    at::Tensor hidden_normed;
    {
        at::AutoGradMode no_grad_mode(false);
        hidden_normed = rms_norm(hidden_detached, final_norm, ctx->rms_eps);
    }
    hidden_normed.set_requires_grad(true);

    int64_t seq_len = hidden_normed.size(1);
    auto shifted_hidden = hidden_normed.narrow(1, 0, seq_len - 1);
    auto shifted_targets = input_ids.narrow(1, 1, seq_len - 1).reshape({-1});
    auto shifted_mask = target_mask.narrow(1, 1, seq_len - 1).reshape({-1});

    int64_t total_tokens = shifted_targets.size(0);
    // Smaller chunks = less peak memory (4GB vs 16GB per chunk at vocab=248K).
    // This allows removing emptyCache between chunks — GPU stays async.
    int64_t chunk_size = 4096;
    int64_t num_chunks = (total_tokens + chunk_size - 1) / chunk_size;

    auto total_count = shifted_mask.sum().clamp_min(1.0);
    at::Tensor token_denominators;
    if (independent_samples) {
        auto per_sample_count = target_mask.narrow(1, 1, seq_len - 1)
            .sum(1, true).clamp_min(1.0);
        token_denominators = per_sample_count
            .expand({target_mask.size(0), seq_len - 1}).reshape({-1});
    }
    auto hidden_flat = shifted_hidden.reshape({-1, hidden_normed.size(2)});

    double total_loss_val = 0.0;

    for (int64_t c = 0; c < num_chunks; c++) {
        int64_t start = c * chunk_size;
        int64_t end = std::min(start + chunk_size, total_tokens);
        int64_t n = end - start;

        auto chunk_hidden = hidden_flat.narrow(0, start, n);
        auto chunk_logits = at::matmul(chunk_hidden, lm_head.t());
        auto chunk_targets = shifted_targets.narrow(0, start, n);
        auto chunk_mask = shifted_mask.narrow(0, start, n);

        // Diagnostic: print logits stats for first chunk, first token
        if (c == 0 && getenv("QWEN36_LOSS_DIAG")) {
            auto logits_f = chunk_logits[0].to(at::kFloat);
            auto lsm = at::log_softmax(logits_f, -1);
            int64_t tgt = chunk_targets[0].item<int64_t>();
            fprintf(stderr, "[diag] logits shape: [%ld, %ld]\n", (long)n, (long)chunk_logits.size(1));
            fprintf(stderr, "[diag] logits[0,:5]: %.6f %.6f %.6f %.6f %.6f\n",
                    logits_f[0].item<float>(), logits_f[1].item<float>(),
                    logits_f[2].item<float>(), logits_f[3].item<float>(),
                    logits_f[4].item<float>());
            fprintf(stderr, "[diag] logits mean: %.6f, std: %.6f\n",
                    logits_f.mean().item<float>(), logits_f.std().item<float>());
            fprintf(stderr, "[diag] target token: %ld\n", (long)tgt);
            fprintf(stderr, "[diag] target logit: %.6f\n", logits_f[tgt].item<float>());
            fprintf(stderr, "[diag] log_softmax[target]: %.6f\n", lsm[tgt].item<float>());
            fprintf(stderr, "[diag] -log_softmax[target] (per-token loss): %.6f\n", -lsm[tgt].item<float>());
        }

        auto per_token_loss = at::cross_entropy_loss(
            chunk_logits.to(at::kFloat), chunk_targets,
            at::Tensor(), at::Reduction::None, -100, 0.0
        );
        auto masked_loss = per_token_loss * chunk_mask.to(at::kFloat);
        // Single-sample training uses the global response-token mean. Batched
        // multi-LoRA instead divides each row by its own response-token count
        // so tenant gradients do not depend on neighboring rows.
        auto chunk_loss = independent_samples
            ? (masked_loss / token_denominators.narrow(0, start, n)).sum()
            : masked_loss.sum() / total_count;

        // Backward this chunk — each chunk creates an independent CE subgraph
        // because hidden_normed is a leaf tensor. retain_graph=false is safe
        // and much faster than retain_graph=true (which accumulates graph).
        torch::autograd::backward({chunk_loss}, {},
            /*retain_graph=*/false, /*create_graph=*/false);

        total_loss_val += chunk_loss.item<double>();

    }

    TORCH_CHECK(hidden_normed.grad().defined(),
        "cross-entropy did not produce a hidden gradient");
    auto hidden_normed_recompute = rms_norm(hidden_detached, final_norm, ctx->rms_eps);
    auto hidden_grad = torch::autograd::grad(
        {hidden_normed_recompute}, {hidden_detached}, {hidden_normed.grad()},
        /*retain_graph=*/false, /*create_graph=*/false,
        /*allow_unused=*/false)[0];

    return {
        at::tensor({total_loss_val},
            at::TensorOptions().dtype(at::kFloat).device(hidden.device())),
        hidden_grad
    };
}

// ──────────────────────────────────────────────────────────────────────
// MTP (Multi-Token Prediction) forward + loss
// ──────────────────────────────────────────────────────────────────────

// MTP forward: produce hidden states (not logits) for chunked loss computation.
// hidden: [batch, seq, hidden] — pre-norm hidden from main model
// Returns: [batch, seq-1, hidden] — MTP hidden (after final norm, before lm_head)
static at::Tensor mtp_forward(
    TrainingContext* ctx,
    const at::Tensor& hidden,
    const at::Tensor& input_ids
) {
    auto kind = ctx->compute_type;
    int64_t seq_len = hidden.size(1);

    // hidden[t] + embed[t+1] → predict token t+2 (Megatron convention)
    auto hidden_shifted = hidden.narrow(1, 0, seq_len - 1);  // [batch, seq-1, hidden]
    auto embed_next = vocabulary_embedding(
        ctx, input_ids.narrow(1, 1, seq_len - 1));  // [batch, seq-1, hidden]

    // RMSNorm both
    auto h_normed = rms_norm(hidden_shifted, *ctx->mtp_pre_fc_norm_hidden, ctx->rms_eps).to(kind);
    auto e_normed = rms_norm(embed_next, *ctx->mtp_pre_fc_norm_emb, ctx->rms_eps).to(kind);

    // Combine: embed first, then hidden → fc projection
    auto combined = at::cat({e_normed, h_normed}, /*dim=*/-1);
    auto projected = at::matmul(combined, ctx->mtp_fc->t());  // fc: [hidden, 2*hidden]

    // MTP layers (full attention + MoE/dense, no LoRA)
    at::Tensor h = projected;
    int64_t num_mtp_layers = (int64_t)ctx->mtp_layer_configs.size();
    for (int64_t i = 0; i < num_mtp_layers; i++) {
        int64_t w_offset = 0;
        for (int64_t j = 0; j < i; j++)
            w_offset += weight_count_for_layer(ctx->mtp_layer_configs[j]);
        int64_t w_count = weight_count_for_layer(ctx->mtp_layer_configs[i]);
        std::vector<at::Tensor*> layer_w(ctx->mtp_layer_weights.begin() + w_offset,
                                         ctx->mtp_layer_weights.begin() + w_offset + w_count);
        // MTP processes seq-1 tokens — slice attention mask's last dim to match
        auto mtp_mask = ctx->attention_mask.defined()
            ? ctx->attention_mask.narrow(-1, 0, h.size(1))
            : at::Tensor();
        h = forward_single_layer(ctx, h, layer_w.data(), &ctx->mtp_layer_configs[i],
            ctx->num_layers + i, kind, mtp_mask);
    }

    // Final norm only — return hidden, not logits
    return rms_norm(h, *ctx->mtp_norm, ctx->rms_eps).to(kind);
}

// MTP loss: chunked matmul + cross-entropy, weighted by 0.5
// mtp_hidden[t] (from hidden[t] + embed[t+1]) predicts token t+2 (Megatron convention)
// No full logits tensor — chunked matmul + fused CE
static at::Tensor mtp_compute_loss(
    TrainingContext* ctx,
    const at::Tensor& mtp_hidden,
    const at::Tensor& input_ids,
    const at::Tensor& target_mask,
    bool independent_samples = false
) {
    int64_t vocab_size = ctx->vocab_size;
    int64_t seq_len = input_ids.size(1);
    auto lm_head = *ctx->lm_head_ptr[0];

    // MTP hidden: [batch, seq-1, hidden], drop last → predict t+2
    int64_t n_tokens = seq_len - 2;
    auto hidden_flat = mtp_hidden.narrow(1, 0, n_tokens).reshape({-1, mtp_hidden.size(2)});
    auto shifted_targets = input_ids.narrow(1, 2, n_tokens).reshape({-1});
    auto shifted_mask = target_mask.narrow(1, 2, n_tokens).reshape({-1});

    // Chunked matmul + cross-entropy
    int64_t total_tokens = shifted_targets.size(0);
    int64_t chunk_size = 4096;  // smaller chunks = less peak memory
    int64_t num_chunks = (total_tokens + chunk_size - 1) / chunk_size;

    auto total_loss = at::zeros({1}, at::TensorOptions().dtype(at::kFloat).device(mtp_hidden.device()));
    auto total_count = shifted_mask.sum().clamp_min(1.0);
    at::Tensor token_denominators;
    if (independent_samples) {
        auto per_sample_count = target_mask.narrow(1, 2, n_tokens)
            .sum(1, true).clamp_min(1.0);
        token_denominators = per_sample_count
            .expand({target_mask.size(0), n_tokens}).reshape({-1});
    }

    for (int64_t c = 0; c < num_chunks; c++) {
        int64_t start = c * chunk_size;
        int64_t end = std::min(start + chunk_size, total_tokens);
        int64_t n = end - start;

        auto chunk_hidden = hidden_flat.narrow(0, start, n);
        auto chunk_logits = at::matmul(chunk_hidden, lm_head.t());  // [n, vocab]
        auto chunk_targets = shifted_targets.narrow(0, start, n);
        auto chunk_mask = shifted_mask.narrow(0, start, n);

        auto per_token_loss = at::cross_entropy_loss(
            chunk_logits.to(at::kFloat), chunk_targets,
            /*weight=*/at::Tensor(), /*reduction=*/at::Reduction::None,
            /*ignore_index=*/-100, /*label_smoothing=*/0.0
        );
        auto masked_loss = per_token_loss * chunk_mask.to(at::kFloat);
        // Avoid in-place accumulation into a non-grad leaf: out-of-place add
        // keeps the MTP loss connected to the frozen-head input graph.
        total_loss = total_loss + (independent_samples
            ? (masked_loss / token_denominators.narrow(0, start, n)).sum()
            : masked_loss.sum());
    }

    if (!independent_samples) total_loss = total_loss / total_count;
    return total_loss * ctx->mtp_loss_scale;
}

// ──────────────────────────────────────────────────────────────────────
// C FFI
// ──────────────────────────────────────────────────────────────────────

extern "C" {

__attribute__((visibility("default"))) int64_t qwen36_kernel_abi_version() {
    return 24;
}

// Benchmark/metrics helper. The orthogonal EP, DP, and TP reductions propagate
// the maximum over the complete process grid without requiring a world group.
__attribute__((visibility("default"))) double qwen36_parallel_max_double(
    void* ctx_ptr, double value
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        TORCH_CHECK(ctx, "parallel max requires a valid training context");
        c10::cuda::set_device(ctx->cuda_device);
        cudaSetDevice(ctx->cuda_device);
        auto maximum = at::full(
            {1}, value,
            at::TensorOptions().device(at::kCUDA, ctx->cuda_device)
                .dtype(at::kDouble));
        auto stream = c10::cuda::getCurrentCUDAStream(ctx->cuda_device).stream();
        auto reduce_axis = [&](ncclComm_t communicator, const char* axis) {
            if (!communicator) return;
            auto err = ncclAllReduce(
                maximum.data_ptr<double>(), maximum.data_ptr<double>(), 1,
                ncclDouble, ncclMax, communicator, stream);
            TORCH_CHECK(err == ncclSuccess, axis,
                " benchmark max all-reduce failed: ", ncclGetErrorString(err));
        };
        reduce_axis(ctx->nccl_comm, "EP");
        reduce_axis(ctx->dp_comm, "DP");
        reduce_axis(ctx->tp_comm, "TP");
        return maximum.to(at::kCPU).item<double>();
    } catch (const std::exception& error) {
        fprintf(stderr, "[q36] parallel max FAILED: %s\n", error.what());
        return std::numeric_limits<double>::quiet_NaN();
    }
}

static constexpr int32_t QWEN36_CONTEXT_BASE_TP_ATTENTION = 1 << 0;
static constexpr int32_t QWEN36_CONTEXT_DATA_PARALLEL = 1 << 1;
static constexpr int32_t QWEN36_CONTEXT_VOCAB_PARALLEL = 1 << 2;
static constexpr int32_t QWEN36_CONTEXT_EXPERT_PARALLEL = 1 << 3;
static constexpr int32_t QWEN36_CONTEXT_BASE_TP_MLP = 1 << 4;

// Create training context — called once at startup
// lora_rank: LoRA rank (from config)
// target_layers: array of layer indices to apply LoRA (nullptr = all layers)
// num_target_layers: length of target_layers array
static void* qwen36_create_training_context_impl(
    void** weight_ptrs, int64_t num_weight_ptrs,
    void* embed_ptr, void* final_norm_ptr, void* lm_head_ptr,
    void* layer_configs_ptr, int64_t num_layers,
    int32_t compute_type,
    double lora_scaling, double lr, double beta1, double beta2, double eps,
    int64_t vocab_size, double rms_eps,
    int64_t lora_rank,
    const int64_t* target_layers, int64_t num_target_layers,
    const char* target_modules_str,
    int32_t context_flags
) {
    try {
        auto* ctx = new TrainingContext();
        ctx->context_sequence = ++g_context_sequence;
        ctx->compute_type = static_cast<at::ScalarType>(compute_type);
        ctx->lr = lr; ctx->beta1 = beta1; ctx->beta2 = beta2; ctx->eps = eps;
        ctx->vocab_size = vocab_size; ctx->rms_eps = rms_eps;
        ctx->fixed_optimizer_step = 0; ctx->lora_scaling = lora_scaling;
        ctx->num_layers = num_layers;
        ctx->base_tp_attention =
            (context_flags & QWEN36_CONTEXT_BASE_TP_ATTENTION) != 0;
        ctx->base_tp_mlp =
            (context_flags & QWEN36_CONTEXT_BASE_TP_MLP) != 0;
        ctx->vocab_parallel =
            (context_flags & QWEN36_CONTEXT_VOCAB_PARALLEL) != 0;
        ctx->has_mtp = false;
        ctx->use_checkpoint = false; ctx->group_size = 4;
        const char* tp_size_env = getenv("TP_SIZE");
        if (!tp_size_env) tp_size_env = getenv("RUSTRAIN_TP_SIZE");
        ctx->tp_world_size = tp_size_env ? atoi(tp_size_env) : 1;
        TORCH_CHECK(ctx->tp_world_size > 0, "TP_SIZE must be positive");
        const char* world_size_env = getenv("WORLD_SIZE");
        const int configured_world_size = world_size_env ? atoi(world_size_env) : 1;
        TORCH_CHECK(configured_world_size > 0, "WORLD_SIZE must be positive");
        const bool data_parallel_requested =
            (context_flags & QWEN36_CONTEXT_DATA_PARALLEL) != 0 ||
            env_enabled("RUSTRAIN_DATA_PARALLEL");
        const char* dp_size_env = getenv("DP_SIZE");
        if (!dp_size_env) dp_size_env = getenv("RUSTRAIN_DP_SIZE");
        int configured_dp_size = dp_size_env ? atoi(dp_size_env) : 1;
        const char* ep_size_env = getenv("EP_SIZE");
        if (!ep_size_env) ep_size_env = getenv("RUSTRAIN_EP_SIZE");
        const int configured_ep_size = ep_size_env ? atoi(ep_size_env) : 1;
        const bool expert_parallel_requested =
            (context_flags & QWEN36_CONTEXT_EXPERT_PARALLEL) != 0 ||
            configured_ep_size > 1;
        if (data_parallel_requested && !dp_size_env &&
            configured_world_size %
                (ctx->tp_world_size * configured_ep_size) == 0) {
            configured_dp_size = configured_world_size /
                (ctx->tp_world_size * configured_ep_size);
        }
        const char* pp_size_env = getenv("PP_SIZE");
        if (!pp_size_env) pp_size_env = getenv("RUSTRAIN_PP_SIZE");
        const char* cp_size_env = getenv("CP_SIZE");
        if (!cp_size_env) cp_size_env = getenv("RUSTRAIN_CP_SIZE");
        const int configured_pp_size = pp_size_env ? atoi(pp_size_env) : 1;
        const int configured_cp_size = cp_size_env ? atoi(cp_size_env) : 1;
        TORCH_CHECK(configured_pp_size > 0 && configured_cp_size > 0 &&
                    configured_dp_size > 0 && configured_ep_size > 0,
            "PP_SIZE, CP_SIZE, DP_SIZE, and EP_SIZE must be positive");
        TORCH_CHECK(configured_pp_size == 1 && configured_cp_size == 1,
            "native Qwen LoRA does not implement PP/CP yet; ",
            "PP_SIZE=", configured_pp_size, " CP_SIZE=", configured_cp_size);
        TORCH_CHECK(configured_world_size ==
                ctx->tp_world_size * configured_ep_size * configured_dp_size,
            "native Qwen LoRA requires WORLD_SIZE=TP_SIZE*EP_SIZE*DP_SIZE; ",
            "TP_SIZE=", ctx->tp_world_size, " EP_SIZE=", configured_ep_size,
            " DP_SIZE=", configured_dp_size,
            " WORLD_SIZE=", configured_world_size);
        TORCH_CHECK(expert_parallel_requested == (configured_ep_size > 1),
            "expert-parallel context flag must match EP_SIZE");
        TORCH_CHECK(data_parallel_requested == (configured_dp_size > 1),
            "data-parallel context flag must match DP_SIZE");
        ctx->ep_world_size = configured_ep_size;
        ctx->expert_parallel = expert_parallel_requested;
        ctx->dp_world_size = configured_dp_size;
        ctx->data_parallel = data_parallel_requested;
        const char* rank_env = getenv("RANK");
        const int global_rank = rank_env ? atoi(rank_env) : 0;
        const char* tp_rank_env = getenv("RUSTRAIN_TP_RANK");
        const char* ep_rank_env = getenv("RUSTRAIN_EP_RANK");
        const char* dp_rank_env = getenv("RUSTRAIN_DP_RANK");
        ctx->tp_rank = tp_rank_env ? atoi(tp_rank_env)
            : global_rank % ctx->tp_world_size;
        ctx->ep_rank = ep_rank_env ? atoi(ep_rank_env)
            : (global_rank / ctx->tp_world_size) % configured_ep_size;
        ctx->dp_rank = dp_rank_env ? atoi(dp_rank_env)
            : global_rank / (ctx->tp_world_size * configured_ep_size);
        TORCH_CHECK(ctx->tp_rank >= 0 && ctx->tp_rank < ctx->tp_world_size &&
                    ctx->ep_rank >= 0 && ctx->ep_rank < ctx->ep_world_size &&
                    ctx->dp_rank >= 0 && ctx->dp_rank < ctx->dp_world_size,
            "parallel rank coordinates are outside their configured axes");
        TORCH_CHECK(lora_rank > 0, "LoRA rank must be positive");
        if (const char* mtp_scale = getenv("QWEN36_MTP_LOSS_SCALE")) {
            ctx->mtp_loss_scale = std::strtod(mtp_scale, nullptr);
        }

        // Store weight pointers
        auto** wp = reinterpret_cast<at::Tensor**>(weight_ptrs);
        for (int64_t i = 0; i < num_weight_ptrs; i++) {
            ctx->weight_ptrs.push_back(wp[i]);
        }
        ctx->embed_ptr.push_back(reinterpret_cast<at::Tensor*>(embed_ptr));
        ctx->final_norm_ptr.push_back(reinterpret_cast<at::Tensor*>(final_norm_ptr));
        ctx->lm_head_ptr.push_back(reinterpret_cast<at::Tensor*>(lm_head_ptr));
        if (ctx->vocab_parallel) {
            TORCH_CHECK(ctx->tp_world_size > 1,
                "vocabulary parallelism requires TP_SIZE>1");
            TORCH_CHECK(ctx->vocab_size > 0 &&
                    ctx->vocab_size % ctx->tp_world_size == 0,
                "vocab_size=", ctx->vocab_size,
                " must be divisible by TP_SIZE=", ctx->tp_world_size);
            ctx->local_vocab_size = ctx->vocab_size / ctx->tp_world_size;
            auto* embed = ctx->embed_ptr[0];
            auto* lm_head = ctx->lm_head_ptr[0];
            TORCH_CHECK(embed && lm_head && embed->dim() == 2 && lm_head->dim() == 2,
                "vocabulary parallelism requires matrix embedding and LM-head weights");
            TORCH_CHECK(embed->size(0) == ctx->local_vocab_size &&
                    lm_head->size(0) == ctx->local_vocab_size &&
                    embed->size(1) == lm_head->size(1),
                "vocabulary parallelism received inconsistent local weight shapes: embed=",
                embed->sizes(), " lm_head=", lm_head->sizes(),
                " expected local vocab=", ctx->local_vocab_size);
        } else {
            ctx->local_vocab_size = ctx->vocab_size;
        }

        // Copy layer configs
        auto* lcfgs = reinterpret_cast<LayerConfig*>(layer_configs_ptr);
        for (int64_t i = 0; i < num_layers; i++) {
            ctx->layer_configs.push_back(lcfgs[i]);
        }

        ctx->fused_gdn_ab_weights.resize(num_layers);
        if (env_enabled("QWEN36_GDN_FUSED_AB_PROJECTION", true)) {
            at::NoGradGuard guard;
            int64_t weight_offset = 0;
            for (int64_t layer = 0; layer < num_layers; ++layer) {
                const auto& cfg = ctx->layer_configs[layer];
                if (cfg.layer_type != 0) {
                    auto* a = ctx->weight_ptrs[weight_offset + 4];
                    auto* b = ctx->weight_ptrs[weight_offset + 5];
                    TORCH_CHECK(a && b && a->dim() == 2 && b->dim() == 2 &&
                            a->sizes() == b->sizes(),
                        "GDN fused A/B projection requires matching matrix weights at layer ",
                        layer);
                    ctx->fused_gdn_ab_weights[layer] =
                        at::cat({*a, *b}, 0).contiguous();
                }
                weight_offset += weight_count_for_layer(cfg);
            }
        }

        if (ctx->base_tp_attention) {
            TORCH_CHECK(ctx->tp_world_size > 1,
                "base attention TP requires TP_SIZE>1");
            int64_t weight_offset = 0;
            for (int64_t layer = 0; layer < num_layers; ++layer) {
                const auto& cfg = ctx->layer_configs[layer];
                if (cfg.layer_type == 0) {
                    TORCH_CHECK(cfg.num_heads > 0 && cfg.num_kv_heads > 0 &&
                            cfg.head_dim > 0,
                        "invalid full-attention head configuration at layer ", layer);
                    TORCH_CHECK(cfg.num_heads % cfg.num_kv_heads == 0 &&
                            cfg.num_heads % ctx->tp_world_size == 0 &&
                            cfg.num_kv_heads % ctx->tp_world_size == 0,
                        "full-attention heads must preserve GQA groups and be divisible by TP_SIZE at layer ",
                        layer);
                    const int64_t rotary_dim = static_cast<int64_t>(
                        cfg.head_dim * cfg.partial_rotary_factor);
                    TORCH_CHECK(rotary_dim >= 0 && rotary_dim <= cfg.head_dim &&
                            rotary_dim % 2 == 0,
                        "full-attention rotary dimension must be even and within head_dim at layer ",
                        layer, ": rotary_dim=", rotary_dim,
                        " head_dim=", cfg.head_dim);
                    auto* q = ctx->weight_ptrs[weight_offset + 2];
                    auto* k = ctx->weight_ptrs[weight_offset + 4];
                    auto* v = ctx->weight_ptrs[weight_offset + 6];
                    auto* o = ctx->weight_ptrs[weight_offset + 7];
                    TORCH_CHECK(q && k && v && o && q->dim() == 2 &&
                            k->dim() == 2 && v->dim() == 2 && o->dim() == 2,
                        "base full-attention TP requires matrix Q/K/V/O weights at layer ",
                        layer);
                    const int64_t local_heads = cfg.num_heads / ctx->tp_world_size;
                    const int64_t local_kv_heads = cfg.num_kv_heads / ctx->tp_world_size;
                    TORCH_CHECK(q->size(0) == local_heads * cfg.head_dim * 2 &&
                            k->size(0) == local_kv_heads * cfg.head_dim &&
                            v->size(0) == local_kv_heads * cfg.head_dim &&
                            o->size(1) == local_heads * cfg.head_dim &&
                            q->size(1) == k->size(1) && q->size(1) == v->size(1) &&
                            q->size(1) == o->size(0),
                        "base full-attention TP received inconsistent local weight shapes at layer ",
                        layer, ": q=", q->sizes(), " k=", k->sizes(),
                        " v=", v->sizes(), " o=", o->sizes());
                } else {
                    TORCH_CHECK(cfg.num_k_heads > 0 && cfg.num_v_heads > 0 &&
                            cfg.key_dim == 128 && cfg.val_dim == 128 &&
                            cfg.conv_kernel > 0,
                        "invalid linear-attention configuration at layer ", layer);
                    TORCH_CHECK(cfg.num_v_heads % cfg.num_k_heads == 0 &&
                            cfg.num_k_heads % ctx->tp_world_size == 0 &&
                            cfg.num_v_heads % ctx->tp_world_size == 0,
                        "linear-attention heads must preserve value-head groups and be divisible by TP_SIZE at layer ",
                        layer);
                    auto* qkv = ctx->weight_ptrs[weight_offset + 2];
                    auto* z = ctx->weight_ptrs[weight_offset + 3];
                    auto* a = ctx->weight_ptrs[weight_offset + 4];
                    auto* b = ctx->weight_ptrs[weight_offset + 5];
                    auto* a_log = ctx->weight_ptrs[weight_offset + 6];
                    auto* dt_bias = ctx->weight_ptrs[weight_offset + 7];
                    auto* conv = ctx->weight_ptrs[weight_offset + 8];
                    auto* norm = ctx->weight_ptrs[weight_offset + 9];
                    auto* out = ctx->weight_ptrs[weight_offset + 10];
                    TORCH_CHECK(qkv && z && a && b && a_log && dt_bias &&
                            conv && norm && out,
                        "base linear-attention TP received null weights at layer ", layer);
                    const int64_t local_k_heads =
                        cfg.num_k_heads / ctx->tp_world_size;
                    const int64_t local_v_heads =
                        cfg.num_v_heads / ctx->tp_world_size;
                    const int64_t local_q = local_k_heads * cfg.key_dim;
                    const int64_t local_v = local_v_heads * cfg.val_dim;
                    const int64_t local_qkv = local_q * 2 + local_v;
                    TORCH_CHECK(qkv->dim() == 2 &&
                            qkv->size(0) == local_qkv,
                        "base linear-attention TP QKV shape mismatch at layer ",
                        layer, ": ", qkv->sizes(), " expected rows=", local_qkv);
                    const int64_t hidden_size = qkv->size(1);
                    TORCH_CHECK(z->dim() == 2 && z->size(0) == local_v &&
                            z->size(1) == hidden_size &&
                            a->dim() == 2 && a->size(0) == local_v_heads &&
                            a->size(1) == hidden_size &&
                            b->dim() == 2 && b->sizes() == a->sizes(),
                        "base linear-attention TP Z/A/B shapes mismatch at layer ", layer,
                        ": z=", z->sizes(), " a=", a->sizes(), " b=", b->sizes());
                    TORCH_CHECK(a_log->dim() == 1 &&
                            a_log->size(0) == local_v_heads &&
                            dt_bias->dim() == 1 &&
                            dt_bias->size(0) == local_v_heads,
                        "base linear-attention TP A_log/dt_bias shapes mismatch at layer ",
                        layer, ": A_log=", a_log->sizes(),
                        " dt_bias=", dt_bias->sizes());
                    TORCH_CHECK(conv->dim() == 3 &&
                            conv->size(0) == local_qkv && conv->size(1) == 1 &&
                            conv->size(2) == cfg.conv_kernel,
                        "base linear-attention TP depthwise-conv shape mismatch at layer ",
                        layer, ": ", conv->sizes());
                    TORCH_CHECK(norm->dim() == 1 && norm->size(0) == cfg.val_dim,
                        "base linear-attention TP norm must remain replicated at layer ",
                        layer, ": ", norm->sizes());
                    TORCH_CHECK(out->dim() == 2 && out->size(0) == hidden_size &&
                            out->size(1) == local_v,
                        "base linear-attention TP output shape mismatch at layer ",
                        layer, ": ", out->sizes(), " expected [", hidden_size,
                        ", ", local_v, "]");
                }
                weight_offset += weight_count_for_layer(cfg);
            }
        }

        // Build target layer set
        std::set<int64_t> target_set;
        const bool all_target_layers = !target_layers || num_target_layers == 0;
        if (target_layers && num_target_layers > 0) {
            for (int64_t j = 0; j < num_target_layers; j++) {
                TORCH_CHECK(target_layers[j] >= 0 && target_layers[j] < num_layers,
                    "LoRA target layer out of range: ", target_layers[j],
                    " for model with ", num_layers, " layers");
                target_set.insert(target_layers[j]);
            }
        }

        std::set<std::string> target_modules;
        if (target_modules_str && target_modules_str[0] != '\0') {
            std::stringstream ss(target_modules_str);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) target_modules.insert(item);
            }
        }
        for (const auto& name : target_modules) {
            TORCH_CHECK(
                name == "q_proj" || name == "k_proj" || name == "v_proj" ||
                name == "o_proj" || name == "in_proj_qkv" ||
                name == "in_proj_z" || name == "in_proj_a" ||
                name == "in_proj_b" || name == "out_proj" ||
                name == "gate_proj" || name == "up_proj" ||
                name == "down_proj" || name == "shared_gate_proj" ||
                name == "shared_up_proj" || name == "shared_down_proj" ||
                name == "experts_gate_up_proj" || name == "experts_down_proj",
                "unsupported native Qwen LoRA target module: ", name,
                "; supported routed expert targets are experts_gate_up_proj/experts_down_proj");
        }
        for (const auto& name : target_modules) {
            bool resolved = false;
            for (const auto& layer_cfg : ctx->layer_configs) {
                auto table = lora_projection_table(layer_cfg);
                for (int64_t k = 0; k < table.count; ++k) {
                    if (name == table.entries[k].name) {
                        resolved = true;
                        break;
                    }
                }
                if (resolved) break;
            }
            TORCH_CHECK(resolved,
                "LoRA target module does not exist in this model: ", name);
        }
        const int64_t local_lora_rank = local_lora_rank_for_active_targets(
            ctx, lora_rank, target_set, target_modules,
            /*empty_modules_mean_attention_only=*/false, "fixed");

        // Create fixed positional slots for every layer so layer offsets stay
        // stable. Inactive slots are zero tensors without grad and are skipped
        // by cache construction/Adam; this preserves the existing FFI export
        // indexing while honoring target_modules exactly.
        int64_t offset = 0;
        for (int64_t i = 0; i < num_layers; i++) {
            auto projection_table = lora_projection_table(ctx->layer_configs[i]);
            int64_t lora_count = projection_table.count;
            ctx->lora_layer_offset.push_back(offset);

            // Get base weight shapes from the weight pointers
            int64_t w_offset = 0;
            for (int64_t j = 0; j < i; j++)
                w_offset += weight_count_for_layer(ctx->layer_configs[j]);

            for (int64_t k = 0; k < projection_table.count; ++k) {
                const auto& projection = projection_table.entries[k];
                auto* base = ctx->weight_ptrs[w_offset + projection.weight_index];
                TORCH_CHECK(base, "null LoRA base projection: layer=", i,
                    " module=", projection.name);
                TORCH_CHECK(
                    (!projection.grouped_expert && base->dim() == 2)
                        || (projection.grouped_expert && base->dim() == 3),
                    "LoRA projection rank mismatch: layer=", i,
                    " module=", projection.name, " base_dim=", base->dim());
                bool active = (all_target_layers || target_set.find(i) != target_set.end()) &&
                    (target_modules.empty() || target_modules.count(projection.name) > 0);
                auto opts = at::TensorOptions().dtype(ctx->compute_type).device(base->device());
                at::Tensor a, b;
                if (!active) {
                    // Stable slot index without allocating potentially hundreds
                    // of MB for inactive expert-local tensors.
                    a = at::zeros({}, opts);
                    b = at::zeros({}, opts);
                } else if (projection.grouped_expert) {
                    int64_t experts = base->size(0);
                    int64_t out_f = base->size(1), in_f = base->size(2);
                    const auto layout = lora_tp_layout(ctx, i, k);
                    if (layout == LoraTpLayout::ColumnParallel ||
                        layout == LoraTpLayout::RowParallel) {
                        a = at::randn({experts, lora_rank, in_f}, opts) * 0.01;
                        b = at::zeros({experts, out_f, lora_rank}, opts);
                    } else {
                        a = initialize_lora_a(ctx, opts, experts, lora_rank, in_f);
                        b = at::zeros({experts, out_f, local_lora_rank}, opts);
                    }
                } else {
                    int64_t out_f = base->size(0), in_f = base->size(1);
                    const auto layout = lora_tp_layout(ctx, i, k);
                    if (layout == LoraTpLayout::ColumnParallel) {
                        a = at::randn({lora_rank, in_f}, opts) * 0.01;
                        b = at::zeros({out_f, lora_rank}, opts);
                    } else if (layout == LoraTpLayout::RowParallel) {
                        a = at::randn({lora_rank, in_f}, opts) * 0.01;
                        b = at::zeros({out_f, lora_rank}, opts);
                    } else {
                        a = initialize_lora_a(ctx, opts, 0, lora_rank, in_f);
                        b = at::zeros({out_f, local_lora_rank}, opts);
                    }
                }
                a.set_requires_grad(active);
                b.set_requires_grad(active);
                ctx->grad_accum_a.push_back(at::Tensor());
                ctx->grad_accum_b.push_back(at::Tensor());
                ctx->lora_a.push_back(std::move(a));
                ctx->lora_b.push_back(std::move(b));
                ctx->lora_active.push_back(active ? 1 : 0);
                auto prefix = "layers." + std::to_string(i) + "." + projection.name;
                ctx->lora_names.push_back(prefix + ".lora_A.weight");
                ctx->lora_names.push_back(prefix + ".lora_B.weight");
            }
            offset += lora_count;
        }
        bind_fixed_lora_gradient_slab(ctx);

        // Initialize Adam state (FP32 for numerical stability, even if params are BF16)
        for (size_t i = 0; i < ctx->lora_a.size(); i++) {
            auto opts_f32 = at::TensorOptions().dtype(at::kFloat).device(ctx->lora_a[i].device());
            ctx->adam_m.push_back(at::zeros(ctx->lora_a[i].sizes(), opts_f32));
            ctx->adam_m.push_back(at::zeros(ctx->lora_b[i].sizes(), opts_f32));
            ctx->adam_v.push_back(at::zeros(ctx->lora_a[i].sizes(), opts_f32));
            ctx->adam_v.push_back(at::zeros(ctx->lora_b[i].sizes(), opts_f32));
        }

        fprintf(stderr, "[q36_ctx] created: %ld layers, %ld LoRA params, %ld Adam states\n",
            (long)num_layers, (long)ctx->lora_a.size(), (long)ctx->adam_m.size());
        return ctx;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] create FAILED: %s\n", e.what());
        return nullptr;
    }
}

__attribute__((visibility("default"))) void* qwen36_create_training_context(
    void** weight_ptrs, int64_t num_weight_ptrs,
    void* embed_ptr, void* final_norm_ptr, void* lm_head_ptr,
    void* layer_configs_ptr, int64_t num_layers,
    int32_t compute_type,
    double lora_scaling, double lr, double beta1, double beta2, double eps,
    int64_t vocab_size, double rms_eps,
    int64_t lora_rank,
    const int64_t* target_layers, int64_t num_target_layers,
    const char* target_modules_str
) {
    return qwen36_create_training_context_impl(
        weight_ptrs, num_weight_ptrs, embed_ptr, final_norm_ptr, lm_head_ptr,
        layer_configs_ptr, num_layers, compute_type, lora_scaling, lr, beta1,
        beta2, eps, vocab_size, rms_eps, lora_rank, target_layers,
        num_target_layers, target_modules_str, 0);
}

__attribute__((visibility("default"))) void* qwen36_create_training_context_ex(
    void** weight_ptrs, int64_t num_weight_ptrs,
    void* embed_ptr, void* final_norm_ptr, void* lm_head_ptr,
    void* layer_configs_ptr, int64_t num_layers,
    int32_t compute_type,
    double lora_scaling, double lr, double beta1, double beta2, double eps,
    int64_t vocab_size, double rms_eps,
    int64_t lora_rank,
    const int64_t* target_layers, int64_t num_target_layers,
    const char* target_modules_str, int32_t context_flags
) {
    return qwen36_create_training_context_impl(
        weight_ptrs, num_weight_ptrs, embed_ptr, final_norm_ptr, lm_head_ptr,
        layer_configs_ptr, num_layers, compute_type, lora_scaling, lr, beta1,
        beta2, eps, vocab_size, rms_eps, lora_rank, target_layers,
        num_target_layers, target_modules_str, context_flags);
}

__attribute__((visibility("default"))) int32_t qwen36_set_base_tp_mlp(
    void* ctx_ptr, int32_t enabled
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        TORCH_CHECK(ctx, "null training context");
        if (!enabled) {
            TORCH_CHECK(!ctx->base_tp_mlp,
                "base MLP TP cannot be disabled after enablement because "
                "the context owns TP-sharded weights");
            return 0;
        }
        const bool preconfigured = ctx->base_tp_mlp;
        TORCH_CHECK(ctx->tp_world_size > 1,
            "base MLP TP requires TP_SIZE>1");
        TORCH_CHECK(!ctx->has_mtp,
            "base MLP TP requires MTP to be disabled");
        if (!preconfigured) {
            for (const auto& adapter : ctx->adapters) {
                TORCH_CHECK(!adapter.target_modules.empty(),
                    "base MLP TP must be selected when the context is created "
                    "before adding a dynamic adapter that targets all modules");
                for (const auto& name : adapter.target_modules) {
                    TORCH_CHECK(!is_mlp_lora_target(name),
                        "base MLP TP must be selected when the context is created "
                        "before adding dynamic MLP LoRA target ", name);
                }
            }
        }

        int64_t weight_offset = 0;
        for (int64_t layer = 0; layer < ctx->num_layers; ++layer) {
            const auto& cfg = ctx->layer_configs[layer];
            const int64_t mlp_start = cfg.layer_type == 0 ? 8 : 11;
            if (cfg.num_experts == 0) {
                TORCH_CHECK(cfg.intermediate_size > 0 &&
                        cfg.intermediate_size % ctx->tp_world_size == 0,
                    "dense intermediate_size must be divisible by TP_SIZE");
                const int64_t local_intermediate =
                    cfg.intermediate_size / ctx->tp_world_size;
                auto* gate = ctx->weight_ptrs[weight_offset + mlp_start];
                auto* up = ctx->weight_ptrs[weight_offset + mlp_start + 1];
                auto* down = ctx->weight_ptrs[weight_offset + mlp_start + 2];
                TORCH_CHECK(gate && up && down && gate->dim() == 2 &&
                        up->dim() == 2 && down->dim() == 2,
                    "base dense MLP TP requires matrix gate/up/down weights");
                TORCH_CHECK(gate->size(0) == local_intermediate &&
                        up->size(0) == local_intermediate &&
                        down->size(1) == local_intermediate &&
                        gate->size(1) == up->size(1) &&
                        gate->size(1) == down->size(0),
                    "base dense MLP TP received inconsistent local weight shapes: gate=",
                    gate->sizes(), " up=", up->sizes(), " down=", down->sizes(),
                    " expected local intermediate=", local_intermediate);
            } else {
                TORCH_CHECK(cfg.moe_intermediate > 0 &&
                        cfg.moe_intermediate % ctx->tp_world_size == 0,
                    "routed expert intermediate must be divisible by TP_SIZE");
                const int64_t local_intermediate =
                    cfg.moe_intermediate / ctx->tp_world_size;
                auto* shared_gate = ctx->weight_ptrs[weight_offset + mlp_start + 2];
                auto* shared_up = ctx->weight_ptrs[weight_offset + mlp_start + 3];
                auto* shared_down = ctx->weight_ptrs[weight_offset + mlp_start + 4];
                auto* experts_gate_up =
                    ctx->weight_ptrs[weight_offset + mlp_start + 5];
                auto* experts_down =
                    ctx->weight_ptrs[weight_offset + mlp_start + 6];
                TORCH_CHECK(shared_gate && shared_up && shared_down &&
                        experts_gate_up && experts_down &&
                        shared_gate->dim() == 2 && shared_up->dim() == 2 &&
                        shared_down->dim() == 2 &&
                        experts_gate_up->dim() == 3 && experts_down->dim() == 3,
                    "base expert TP requires matrix shared weights and rank-3 "
                    "routed expert weights");
                TORCH_CHECK(shared_gate->size(0) > 0 &&
                        shared_gate->sizes() == shared_up->sizes() &&
                        shared_down->size(1) == shared_gate->size(0) &&
                        shared_down->size(0) == shared_gate->size(1),
                    "base shared expert TP received inconsistent local weight shapes: gate=",
                    shared_gate->sizes(), " up=", shared_up->sizes(),
                    " down=", shared_down->sizes());
                TORCH_CHECK(experts_gate_up->size(0) == cfg.expert_count &&
                        experts_down->size(0) == cfg.expert_count &&
                        experts_gate_up->size(1) == 2 * local_intermediate &&
                        experts_down->size(2) == local_intermediate &&
                        experts_gate_up->size(2) == shared_gate->size(1) &&
                        experts_down->size(1) == shared_gate->size(1),
                    "base routed expert TP received inconsistent local weight shapes: gate_up=",
                    experts_gate_up->sizes(), " down=", experts_down->sizes(),
                    " expected local intermediate=", local_intermediate,
                    " local experts=", cfg.expert_count);
            }

            if (!preconfigured) {
                const int64_t lora_offset = ctx->lora_layer_offset[layer];
                auto projections = lora_projection_table(cfg);
                for (int64_t pair = 0; pair < projections.count; ++pair) {
                    if (projections.entries[pair].segment == LoraSegment::Mlp) {
                        TORCH_CHECK(!ctx->lora_active[lora_offset + pair],
                            "base MLP TP must be selected when the context is "
                            "created before enabling fixed MLP LoRA target ",
                            projections.entries[pair].name);
                    }
                }
            }
            weight_offset += weight_count_for_layer(cfg);
        }
        ctx->base_tp_mlp = true;
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] set_base_tp_mlp FAILED: %s\n", e.what());
        return -1;
    }
}

// Set MTP weights on an existing training context.
// Called after create_training_context if MTP is enabled.
__attribute__((visibility("default"))) int32_t qwen36_set_mtp_weights(
    void* ctx_ptr,
    void* mtp_fc_ptr,
    void* mtp_pre_fc_norm_emb_ptr,
    void* mtp_pre_fc_norm_hidden_ptr,
    void* mtp_norm_ptr,
    void** mtp_layer_weight_ptrs, int64_t num_mtp_layer_weights,
    void* mtp_layer_configs_ptr, int64_t num_mtp_layers
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        TORCH_CHECK(ctx, "null training context");
        TORCH_CHECK(!ctx->base_tp_mlp && !ctx->base_tp_attention &&
                !ctx->vocab_parallel,
            "MTP cannot be enabled after frozen base or vocabulary TP because MTP weights are not sharded");
        TORCH_CHECK(!ctx->has_mtp, "MTP weights are already configured");
        ctx->has_mtp = true;
        ctx->mtp_fc = reinterpret_cast<at::Tensor*>(mtp_fc_ptr);
        ctx->mtp_pre_fc_norm_emb = reinterpret_cast<at::Tensor*>(mtp_pre_fc_norm_emb_ptr);
        ctx->mtp_pre_fc_norm_hidden = reinterpret_cast<at::Tensor*>(mtp_pre_fc_norm_hidden_ptr);
        ctx->mtp_norm = reinterpret_cast<at::Tensor*>(mtp_norm_ptr);

        auto** wp = reinterpret_cast<at::Tensor**>(mtp_layer_weight_ptrs);
        for (int64_t i = 0; i < num_mtp_layer_weights; i++) {
            ctx->mtp_layer_weights.push_back(wp[i]);
        }

        auto* lcfgs = reinterpret_cast<LayerConfig*>(mtp_layer_configs_ptr);
        for (int64_t i = 0; i < num_mtp_layers; i++) {
            ctx->mtp_layer_configs.push_back(lcfgs[i]);
        }

        fprintf(stderr, "[q36_ctx] MTP set: %ld MTP layers, %ld MTP weight pointers\n",
            (long)num_mtp_layers, (long)num_mtp_layer_weights);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] set_mtp_weights FAILED: %s\n", e.what());
        return -1;
    }
}

// One training micro-step. Non-final micro-steps accumulate scaled leaf
// gradients; only the final micro-step synchronizes and updates parameters.
__attribute__((visibility("default"))) double qwen36_train_micro_step(
    void* ctx_ptr,
    void* input_ids_ptr,
    void* target_mask_ptr,
    void* attention_mask_ptr,
    double gradient_scale,
    int32_t apply_optimizer
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        TORCH_CHECK(!ctx->topology_invalid,
            "native Qwen context rejected an incompatible TP/DP/EP topology");
        GradientAccumulationFailureGuard accumulation_guard{ctx};
        auto* input_ids_tensor = reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto* target_mask_tensor = reinterpret_cast<at::Tensor*>(target_mask_ptr);
        auto* attention_mask_tensor = attention_mask_ptr
            ? reinterpret_cast<at::Tensor*>(attention_mask_ptr)
            : nullptr;
        bool local_preflight_valid = input_ids_tensor && target_mask_tensor &&
            input_ids_tensor->is_cuda() && target_mask_tensor->is_cuda() &&
            input_ids_tensor->device() == target_mask_tensor->device() &&
            input_ids_tensor->dim() == 2 && target_mask_tensor->dim() == 2 &&
            input_ids_tensor->size(1) > 1 &&
            input_ids_tensor->sizes() == target_mask_tensor->sizes() &&
            input_ids_tensor->scalar_type() == at::kLong &&
            supported_mask_dtype(*target_mask_tensor) &&
            ctx->adapters.empty() && gradient_scale > 0.0 &&
            std::isfinite(gradient_scale) &&
            (apply_optimizer == 0 || apply_optimizer == 1);
        if (local_preflight_valid && attention_mask_tensor) {
            local_preflight_valid = attention_mask_tensor->is_cuda() &&
                attention_mask_tensor->device() == input_ids_tensor->device() &&
                attention_mask_tensor->dim() == 2 &&
                attention_mask_tensor->sizes() == input_ids_tensor->sizes() &&
                supported_mask_dtype(*attention_mask_tensor);
        }
        if (local_preflight_valid &&
            (ctx->nccl_comm || ctx->dp_comm || ctx->tp_comm)) {
            local_preflight_valid =
                input_ids_tensor->device().index() == ctx->cuda_device;
        }
        TORCH_CHECK(adapter_collective_all_succeeded(
                ctx, local_preflight_valid) && local_preflight_valid,
            "native Qwen fixed-LoRA call must use CUDA inputs on the NCCL "
            "context device, a finite positive gradient scale, and no "
            "dynamic adapters");
        auto& input_ids = *input_ids_tensor;
        const int input_device = input_ids.device().index();
        if (!ctx->nccl_comm && !ctx->dp_comm && !ctx->tp_comm)
            ctx->cuda_device = input_device;
        c10::cuda::set_device(ctx->cuda_device);
        cudaSetDevice(ctx->cuda_device);
        const bool replica_inputs_match = replica_input_signatures_match(
            ctx, input_ids, *target_mask_tensor, attention_mask_tensor);
        TORCH_CHECK(adapter_collective_all_succeeded(
                ctx, replica_inputs_match) && replica_inputs_match,
            "native Qwen fixed-LoRA input shape and dtype differ across TP "
            "or replicated EP ranks");
        const double supervised_tokens = target_mask_tensor
            ->narrow(1, 1, target_mask_tensor->size(1) - 1)
            .to(at::kFloat).sum().item<double>();
        const double micro_token_weight =
            gradient_scale * supervised_tokens;
        const double next_accumulated_token_weight =
            ctx->accumulated_token_weight + micro_token_weight;
        const bool local_token_weight_valid =
            std::isfinite(supervised_tokens) && supervised_tokens >= 0.0 &&
            std::isfinite(micro_token_weight) &&
            std::isfinite(next_accumulated_token_weight);
        TORCH_CHECK(adapter_collective_all_succeeded(
                ctx, local_token_weight_valid) && local_token_weight_valid,
            "native Qwen fixed-LoRA token weights must remain finite and "
            "non-negative on every distributed rank");
        // Fail collective-layout or optimizer-clock disagreement before
        // forward/backward launches any work. The failure guard aborts any
        // pending accumulation window without leaving queued graph work.
        validate_fixed_collective_registry(ctx, apply_optimizer);
        auto& target_mask = *target_mask_tensor;
        if (attention_mask_ptr) {
            auto& attention_mask = *reinterpret_cast<at::Tensor*>(attention_mask_ptr);
            validate_linear_attention_mask(ctx, attention_mask);
            ctx->attention_mask = attention_mask;
            elide_trivial_attention_mask(ctx);
        }

        // Forward: checkpoint (default) or fused layer (QWEN36_FUSED_LAYER=1)
        bool use_fused = env_enabled("QWEN36_FUSED_LAYER");
        TORCH_CHECK(!use_fused,
            "QWEN36_FUSED_LAYER is disabled until its custom backward preserves the layer graph");
        auto hidden = use_fused
            ? forward_full_fused(ctx, input_ids)
            : ctx->use_checkpoint
                ? forward_full_checkpoint(ctx, input_ids)
                : forward_full(ctx, input_ids);

        // Debug: GPU memory after forward
        {
            size_t free, total;
            cudaMemGetInfo(&free, &total);
        }

        // Main loss — compute_loss does chunked CE with immediate backward per chunk.
        // This avoids accumulating 250 chunks of [512, vocab] logits in autograd graph.
        auto main_loss = compute_loss(ctx, hidden, input_ids, target_mask, ctx->vocab_size);
        double loss_val = main_loss.value.item<double>();
        auto total_hidden_grad = main_loss.hidden_grad;

        // MTP must be differentiated before the main model backward. Its
        // frozen head still contributes a hidden-state gradient to trainable
        // main-layer LoRA parameters.
        if (ctx->has_mtp && !env_enabled("QWEN36_DISABLE_MTP")) {
            auto mtp_input = hidden.detach().set_requires_grad(true);
            auto mtp_hidden = mtp_forward(ctx, mtp_input, input_ids);
            auto mtp_loss = mtp_compute_loss(ctx, mtp_hidden, input_ids, target_mask);
            mtp_loss.backward();
            TORCH_CHECK(mtp_input.grad().defined(), "MTP did not produce a hidden gradient");
            total_hidden_grad.add_(mtp_input.grad());
            loss_val += mtp_loss.item<double>();
        }

        // compute_loss returns a local token mean. Convert it to a weighted
        // numerator before backward; the FP32 window is divided by the global
        // accumulated token weight exactly once at the optimizer boundary.
        total_hidden_grad.mul_(micro_token_weight);

        // Trigger exactly one main-model backward with the combined hidden
        // gradient. Manual groups are the non-autograd checkpoint fallback;
        // normal and sub-checkpoint paths use the real graph.
        if (!ctx->group_inputs.empty() && !env_enabled("QWEN36_SUBCKPT")) {
            manual_group_backward(ctx, total_hidden_grad);
        } else {
            hidden.backward(total_hidden_grad);
        }

        // Consume every BF16 leaf gradient immediately. Only the FP32 buffers
        // survive the micro-step boundary.
        const bool accumulation_was_active = ctx->accumulation_active;
        harvest_gradient_accumulators(ctx);
        if (micro_token_weight == 0.0)
            ctx->accumulation_active = accumulation_was_active;
        ctx->accumulated_token_weight = next_accumulated_token_weight;

        if (!apply_optimizer) {
            // The forward graph has been consumed, but parameters remain
            // live for the next micro-batch. Never reuse cached LoRA deltas
            // whose autograd nodes were freed by this backward.
            ctx->lora_cache_valid = false;
            ctx->lora_batch_valid = false;
            accumulation_guard.disarmed = true;
            return loss_val;
        }

        // Replicated DP gradients are synchronized before the local Adam
        // update. EP keeps replicated gradients local because its forward
        // routed activation already contains the cross-rank sum.
        const bool has_global_tokens = synchronize_lora_gradients(
            ctx, target_mask, ctx->accumulated_token_weight);
        if (!has_global_tokens) {
            clear_gradient_accumulators(ctx);
            accumulation_guard.disarmed = true;
            return loss_val;
        }

        // ── Adam optimizer step — CUDA multi-tensor fused kernel ──
        at::AutoGradMode guard(false);
        ctx->lora_cache_valid = false;
        ctx->lora_batch_valid = false;
        const int64_t next_step = ctx->fixed_optimizer_step + 1;
        double step_f = (double)next_step;
        double bias_correction1 = 1.0 - std::pow(ctx->beta1, step_f);
        double bias_correction2 = 1.0 - std::pow(ctx->beta2, step_f);
        double sqrt_bias_correction2 = std::sqrt(bias_correction2);
        float lr_scaled = (float)(
            ctx->lr * sqrt_bias_correction2 / bias_correction1);
        float eps_scaled = (float)(ctx->eps * sqrt_bias_correction2);
        float one_minus_b1 = (float)(1.0 - ctx->beta1);
        float one_minus_b2 = (float)(1.0 - ctx->beta2);

        // Collect all fixed-adapter (param, grad, m, v, size) tuples. Dynamic
        // registries are rejected at function entry and have a separate Adam
        // boundary in qwen36_train_multi_lora.
        std::vector<void*> h_params, h_grads;
        std::vector<float*> h_m, h_v;
        std::vector<int> h_sizes;

        size_t adam_idx = 0;
        for (size_t i = 0; i < ctx->lora_a.size(); i++) {
            {
                auto& param = ctx->lora_a[i];
                auto& accum = ctx->grad_accum_a[i];
                if (ctx->lora_active[i] && accum.defined() &&
                    param.scalar_type() == at::kBFloat16) {
                    h_params.push_back(param.data_ptr());
                    h_grads.push_back(accum.data_ptr());
                    h_m.push_back((float*)ctx->adam_m[adam_idx].data_ptr());
                    h_v.push_back((float*)ctx->adam_v[adam_idx].data_ptr());
                    h_sizes.push_back((int)param.numel());
                }
            }
            adam_idx++;
            {
                auto& param = ctx->lora_b[i];
                auto& accum = ctx->grad_accum_b[i];
                if (ctx->lora_active[i] && accum.defined() &&
                    param.scalar_type() == at::kBFloat16) {
                    h_params.push_back(param.data_ptr());
                    h_grads.push_back(accum.data_ptr());
                    h_m.push_back((float*)ctx->adam_m[adam_idx].data_ptr());
                    h_v.push_back((float*)ctx->adam_v[adam_idx].data_ptr());
                    h_sizes.push_back((int)param.numel());
                }
            }
            adam_idx++;
        }

        int n_params = (int)h_params.size();
        if (n_params > 0) {
            // ── CUDA multi-tensor fused Adam: 1 launch for all params ──
            // Ensure device buffers are large enough
            ctx->adam_dev_bufs.ensure(n_params, ctx->lora_a.empty()
                ? ctx->adapters[0].params.begin()->second[0].first
                : ctx->lora_a[0]);

            // Copy pointer arrays to device
            auto opts_cpu_long = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
            auto opts_cpu_int  = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
            auto params_cpu = at::from_blob(h_params.data(), {n_params}, opts_cpu_long);
            auto grads_cpu  = at::from_blob(h_grads.data(),  {n_params}, opts_cpu_long);
            auto m_cpu      = at::from_blob(h_m.data(),      {n_params}, opts_cpu_long);
            auto v_cpu      = at::from_blob(h_v.data(),      {n_params}, opts_cpu_long);
            auto sizes_cpu  = at::from_blob(h_sizes.data(),  {n_params}, opts_cpu_int);
            ctx->adam_dev_bufs.params_buf.narrow(0, 0, n_params).copy_(params_cpu);
            ctx->adam_dev_bufs.grads_buf.narrow(0, 0, n_params).copy_(grads_cpu);
            ctx->adam_dev_bufs.m_buf.narrow(0, 0, n_params).copy_(m_cpu);
            ctx->adam_dev_bufs.v_buf.narrow(0, 0, n_params).copy_(v_cpu);
            ctx->adam_dev_bufs.sizes_buf.narrow(0, 0, n_params).copy_(sizes_cpu);

            // Single kernel launch for ALL params
            auto stream = c10::cuda::getCurrentCUDAStream().stream();
            launch_fused_adam_multi(
                (void**)ctx->adam_dev_bufs.params_buf.data_ptr(),
                (void**)ctx->adam_dev_bufs.grads_buf.data_ptr(),
                (float**)ctx->adam_dev_bufs.m_buf.data_ptr(),
                (float**)ctx->adam_dev_bufs.v_buf.data_ptr(),
                (int*)ctx->adam_dev_bufs.sizes_buf.data_ptr(),
                n_params,
                (float)ctx->beta1, (float)ctx->beta2,
                lr_scaled, eps_scaled,
                one_minus_b1, one_minus_b2,
                (void*)stream
            );
            auto launch_error = cudaGetLastError();
            TORCH_CHECK(launch_error == cudaSuccess,
                "fused FP32-gradient Adam launch failed: ",
                cudaGetErrorString(launch_error));
            ctx->fixed_optimizer_step = next_step;
        }

        clear_gradient_accumulators(ctx);
        accumulation_guard.disarmed = true;
        return loss_val;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] train_step FAILED: %s\n", e.what());
        return -1.0;
    }
}

// Backward-compatible complete optimizer step.
__attribute__((visibility("default"))) double qwen36_train_step(
    void* ctx_ptr,
    void* input_ids_ptr,
    void* target_mask_ptr,
    void* attention_mask_ptr
) {
    return qwen36_train_micro_step(
        ctx_ptr, input_ids_ptr, target_mask_ptr, attention_mask_ptr, 1.0, 1);
}

// Get LoRA A tensor pointer by index
__attribute__((visibility("default"))) void* qwen36_get_lora_a(void* ctx_ptr, int64_t index) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    if (index < 0 || index >= (int64_t)ctx->lora_a.size()) return nullptr;
    return &ctx->lora_a[index];
}

// Get LoRA B tensor pointer by index
__attribute__((visibility("default"))) void* qwen36_get_lora_b(void* ctx_ptr, int64_t index) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    if (index < 0 || index >= (int64_t)ctx->lora_b.size()) return nullptr;
    return &ctx->lora_b[index];
}

// Read-only diagnostic accessor used by native validation and runtime
// observability. The returned tensor is owned by the training context.
__attribute__((visibility("default"))) void* qwen36_get_lora_grad_accumulator(
    void* ctx_ptr, int64_t index, int32_t is_b
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    if (!ctx || index < 0 || index >= (int64_t)ctx->lora_a.size()) return nullptr;
    auto& accumulators = is_b ? ctx->grad_accum_b : ctx->grad_accum_a;
    if (index >= (int64_t)accumulators.size() || !accumulators[index].defined())
        return nullptr;
    return &accumulators[index];
}

// Explicitly abort an incomplete accumulation window. The operation is
// idempotent and leaves parameters, Adam state, and optimizer clocks intact.
__attribute__((visibility("default"))) int32_t qwen36_abort_gradient_accumulation(
    void* ctx_ptr
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        TORCH_CHECK(ctx, "null training context");
        clear_gradient_accumulators(ctx);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] abort_gradient_accumulation FAILED: %s\n", e.what());
        return -1;
    }
}

// Copy one exported LoRA tensor back into the native leaf parameter. This is
// used by checkpoint resume and adapter import; derived delta caches are
// invalidated so the next forward rebuilds the graph from the new leaf.
__attribute__((visibility("default"))) int32_t qwen36_set_lora_tensor(
    void* ctx_ptr, int64_t index, int32_t is_b, void* tensor_ptr
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        TORCH_CHECK(tensor_ptr, "null LoRA tensor");
        auto& slots = is_b ? ctx->lora_b : ctx->lora_a;
        TORCH_CHECK(index >= 0 && index < (int64_t)slots.size(), "invalid LoRA slot");
        auto& target = slots[index];
        auto& source = *reinterpret_cast<at::Tensor*>(tensor_ptr);
        TORCH_CHECK(source.sizes() == target.sizes(),
            "LoRA tensor shape mismatch at slot ", index,
            ": expected ", target.sizes(), " got ", source.sizes());
        at::NoGradGuard guard;
        target.copy_(source.to(target.device()).to(target.scalar_type()));
        ctx->lora_cache_valid = false;
        ctx->lora_batch_valid = false;
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] set_lora_tensor FAILED: %s\n", e.what());
        return -1;
    }
}

// Free training context
__attribute__((visibility("default"))) void qwen36_free_training_context(void* ctx_ptr) {
    if (ctx_ptr) {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        // Don't destroy NCCL communicator — it's a process-level singleton
        // (g_nccl_comm). It must survive context destruction so the next
        // session can reuse it. Destroying it would break NCCL for all
        // subsequent sessions in the same worker process.
        // ncclCommDestroy is called only on process exit (via atexit or Drop).
        delete ctx;
    }
}

// ── Batched Multi-LoRA Training ──

/// Compute max adapters that fit in available GPU memory.
/// Based on per-adapter activation memory (dominant) + LoRA params + Adam state.
static int64_t compute_n_max(
    int64_t free_gpu_bytes, int64_t rank, int64_t seq,
    int64_t hidden, int64_t group_size, int64_t num_layers,
    bool vocab_parallel, int64_t local_vocab_size
) {
    // Per-adapter activation memory (BF16, group_size=2 optimal):
    // group_inputs: (num_layers / group_size) × seq × hidden × 2 bytes
    // peak recompute: ~34 MB per layer pair × group_size (rough avg)
    int64_t gs = group_size < 1 ? 1 : group_size;
    int64_t num_groups = (num_layers + gs - 1) / gs;
    int64_t group_input_mem = num_groups * seq * hidden * 2;  // BF16

    // Average per-layer saved tensors during recompute:
    // full_attn: ~17MB, linear_attn: ~25MB, moe: ~8MB → avg ~17MB
    int64_t avg_layer_saved = 17 * 1024 * 1024;  // bytes
    int64_t peak_mem = gs * avg_layer_saved;

    // LoRA params + Adam state: 280 modules × (A+B) × (BF16 param + FP32 m + FP32 v)
    // Per module: rank × hidden × 2 (BF16) + hidden × rank × 2 (BF16) + 2 × 4 (FP32 m+v)
    // Simplified: 280 × rank × hidden × (2 + 2 + 8) = 280 × rank × hidden × 12
    int64_t num_modules = 280;
    int64_t lora_mem = num_modules * rank * hidden * 12;  // conservative

    // Vocabulary TP keeps a cached FP32 logits tile while backward also owns
    // FP32 softmax and one-hot buffers. Reserve four tile-sized buffers plus
    // the [token, hidden] gradient so tenant chunking stays conservative.
    int64_t ce_peak = vocab_parallel
        ? 512LL * local_vocab_size * 16LL + 512LL * hidden * 4LL
        : 16384LL * 248320LL * 4LL;

    // Attention intermediate: Q/K/V + attn_weights ≈ 4 × N × heads × seq × head_dim × 2 bytes
    // Flash attention keeps this O(seq) not O(seq²), but still significant at seq=16K
    int64_t attn_heads = 16;
    int64_t head_dim = 256;
    int64_t attn_mem = 4 * attn_heads * seq * head_dim * 2;  // per adapter, BF16

    int64_t per_adapter = group_input_mem + peak_mem + lora_mem + attn_mem;
    // Empirical multiplier: residual add, MoE routing, LoRA delta, etc.
    // Multiplier scales with seq: small seq needs less (CPU dispatch dominant),
    // large seq needs more (attention/MoE intermediates dominate).
    // seq=512: 3x → n_max=100. seq=16K: 8x → n_max≈8 (auto-chunks N=20+).
    int64_t mult = (seq > 4096) ? 8 : 3;
    per_adapter = per_adapter * mult;
    if (per_adapter <= 0) return 1;

    // Reserve 15% for fragmentation + overhead, minus CE peak (constant overhead)
    int64_t usable = (free_gpu_bytes - ce_peak) * 85 / 100;
    if (usable < per_adapter) usable = free_gpu_bytes * 50 / 100;  // fallback: aggressive
    int64_t n_max = usable / per_adapter;
    return n_max < 1 ? 1 : n_max;
}

enum class DynamicMultiLoraMode {
    TrainAndFinalize,
    TrainOnly,
    FinalizeOnly,
};

/// Train all adapters in chunks. Inputs may be [1, seq] (shared prompt,
/// repeated per chunk) or [n_total, seq] (one independent sample per adapter).
/// Heterogeneous selected training reuses the same implementation in two
/// phases so all signature groups share one synchronization/Adam boundary.
static double qwen36_train_multi_lora_impl(
    void* ctx_ptr,
    void* input_ids_ptr,
    void* target_mask_ptr,
    void* attention_mask_ptr,
    int32_t n_total,
    int32_t lora_rank,
    DynamicMultiLoraMode mode,
    int32_t* finalizer_phase = nullptr
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        const bool train_only = mode == DynamicMultiLoraMode::TrainOnly;
        const bool finalize_only = mode == DynamicMultiLoraMode::FinalizeOnly;
        if (finalize_only && finalizer_phase) *finalizer_phase = 0;
        TORCH_CHECK(!ctx->topology_invalid,
            "native Qwen context rejected an incompatible TP/DP/EP topology");
        auto* input_ids_tensor = reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto* target_mask_tensor = reinterpret_cast<at::Tensor*>(target_mask_ptr);
        auto* attention_mask_tensor = attention_mask_ptr
            ? reinterpret_cast<at::Tensor*>(attention_mask_ptr)
            : nullptr;
        const int64_t total_adapters = (int64_t)ctx->adapters.size();
        bool local_input_valid = input_ids_tensor && target_mask_tensor &&
            input_ids_tensor->is_cuda() && target_mask_tensor->is_cuda() &&
            input_ids_tensor->device() == target_mask_tensor->device() &&
            input_ids_tensor->dim() == 2 && target_mask_tensor->dim() == 2 &&
            input_ids_tensor->size(1) > 1 &&
            input_ids_tensor->scalar_type() == at::kLong &&
            supported_mask_dtype(*target_mask_tensor) &&
            target_mask_tensor->sizes() == input_ids_tensor->sizes() &&
            n_total > 0 &&
            (input_ids_tensor->size(0) == 1 ||
                input_ids_tensor->size(0) == n_total) &&
            (mode != DynamicMultiLoraMode::TrainAndFinalize ||
                (!ctx->accumulation_active &&
                    ctx->accumulated_token_weight == 0.0)) &&
            (!ctx->has_mtp || env_enabled("QWEN36_DISABLE_MTP"));
        if (local_input_valid && attention_mask_tensor) {
            local_input_valid = attention_mask_tensor->is_cuda() &&
                attention_mask_tensor->device() == input_ids_tensor->device() &&
                attention_mask_tensor->dim() == 2 &&
                attention_mask_tensor->sizes() == input_ids_tensor->sizes() &&
                supported_mask_dtype(*attention_mask_tensor);
        }
        if (local_input_valid &&
            (ctx->nccl_comm || ctx->dp_comm || ctx->tp_comm)) {
            local_input_valid =
                input_ids_tensor->device().index() == ctx->cuda_device;
        }
        TORCH_CHECK(adapter_collective_all_succeeded(
                ctx, local_input_valid) && local_input_valid,
            "native Qwen multi-LoRA inputs must be CUDA tensors on the "
            "NCCL context device");
        auto& input_ids = *input_ids_tensor;
        auto& target_mask = *target_mask_tensor;
        const int input_device = input_ids.device().index();
        if (!ctx->nccl_comm && !ctx->dp_comm && !ctx->tp_comm)
            ctx->cuda_device = input_device;
        c10::cuda::set_device(ctx->cuda_device);
        cudaSetDevice(ctx->cuda_device);
        const bool replica_inputs_match = replica_input_signatures_match(
            ctx, input_ids, target_mask, attention_mask_tensor);
        TORCH_CHECK(adapter_collective_all_succeeded(
                ctx, replica_inputs_match) && replica_inputs_match,
            "native Qwen multi-LoRA input shape and dtype differ across TP "
            "or replicated EP ranks");
        GradientAccumulationFailureGuard accumulation_guard{ctx};

        validate_adapter_collective_registry(
            ctx, nullptr, n_total, finalize_only ? 0 : lora_rank,
            true);
        if (total_adapters == 0) return -1.0;
        TORCH_CHECK(n_total > 0 && total_adapters == n_total,
            "n_total must equal the number of registered adapters (n_total=",
            n_total, ", registered=", total_adapters, ")");
        if (!finalize_only) {
            const auto& reference_adapter = ctx->adapters.front();
            for (const auto& adapter : ctx->adapters) {
                TORCH_CHECK(adapter.rank == lora_rank,
                    "lora_rank argument must match every registered adapter; adapter=",
                    adapter.id, " registered_rank=", adapter.rank,
                    " requested_rank=", lora_rank);
                TORCH_CHECK(
                    adapter.target_layers == reference_adapter.target_layers &&
                        adapter.target_modules == reference_adapter.target_modules,
                    "legacy multi-LoRA training requires homogeneous target layers/modules; "
                    "use the grouped v2 trainer for heterogeneous adapters");
            }
        }
        const int64_t input_batch = input_ids.size(0);
        auto input_row_token_counts = target_mask
            .narrow(1, 1, target_mask.size(1) - 1)
            .to(at::kFloat).sum(1);
        auto adapter_token_counts = input_batch == 1
            ? input_row_token_counts.repeat({total_adapters})
            : input_row_token_counts;
        // Keep the caller's mask intact. Each chunk receives either the
        // corresponding rows or a repeated batch-1 mask; this also prevents
        // elide_trivial_attention_mask from leaking the last chunk into the
        // next train step.
        at::Tensor provided_attention_mask;
        if (attention_mask_tensor) {
            provided_attention_mask = *attention_mask_tensor;
            validate_linear_attention_mask(ctx, provided_attention_mask);
        }
        const at::Tensor saved_attention_mask = ctx->attention_mask;
        const bool saved_use_checkpoint = ctx->use_checkpoint;
        struct AttentionMaskGuard {
            TrainingContext* ctx;
            at::Tensor saved;
            ~AttentionMaskGuard() { ctx->attention_mask = saved; }
        } attention_mask_guard{ctx, saved_attention_mask};
        struct CheckpointModeGuard {
            TrainingContext* ctx;
            bool saved;
            ~CheckpointModeGuard() { ctx->use_checkpoint = saved; }
        } checkpoint_mode_guard{ctx, saved_use_checkpoint};

        struct AdapterRegistryChunkGuard {
            TrainingContext* ctx;
            std::vector<TrainingContext::LoRAAdapter> all;
            bool active = false;

            AdapterRegistryChunkGuard(
                TrainingContext* context, int64_t start, int64_t end,
                bool scope_registry)
                : ctx(context) {
                if (!scope_registry) return;
                all.swap(ctx->adapters);
                try {
                    ctx->adapters.assign(all.begin() + start, all.begin() + end);
                    active = true;
                } catch (...) {
                    ctx->adapters.clear();
                    ctx->adapters.swap(all);
                    throw;
                }
            }

            void restore() {
                if (!active) return;
                ctx->adapters.clear();
                ctx->adapters.swap(all);
                active = false;
            }

            ~AdapterRegistryChunkGuard() { restore(); }
        };

        int64_t n_max = total_adapters;
        if (!finalize_only) {
            // All workers in the TP x EP x DP grid must agree on the
            // activation chunk schedule to keep forward collectives ordered.
            size_t free_mem = 0;
            size_t total_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);
            n_max = compute_n_max(
                (int64_t)free_mem, lora_rank,
                input_ids.size(-1), 2048,
                ctx->group_size, ctx->num_layers, ctx->vocab_parallel,
                ctx->local_vocab_size
            );
            n_max = std::min(n_max, total_adapters);
            if (n_max < 1) n_max = 1;
            if ((ctx->nccl_comm && ctx->ep_world_size > 1) ||
                (ctx->dp_comm && ctx->dp_world_size > 1) ||
                (ctx->tp_comm && ctx->tp_world_size > 1)) {
                auto published_n_max = at::full(
                    {1}, n_max, input_ids.options().dtype(at::kLong));
                auto stream = c10::cuda::getCurrentCUDAStream(
                    input_ids.device().index()).stream();
                if (ctx->nccl_comm && ctx->ep_world_size > 1) {
                    auto err = ncclAllReduce(
                        published_n_max.data_ptr<int64_t>(),
                        published_n_max.data_ptr<int64_t>(), 1, ncclInt64,
                        ncclMin, reinterpret_cast<ncclComm_t>(ctx->nccl_comm),
                        stream);
                    TORCH_CHECK(err == ncclSuccess,
                        "EP n_max all-reduce failed: ", ncclGetErrorString(err));
                }
                if (ctx->dp_comm && ctx->dp_world_size > 1) {
                    auto err = ncclAllReduce(
                        published_n_max.data_ptr<int64_t>(),
                        published_n_max.data_ptr<int64_t>(), 1, ncclInt64,
                        ncclMin, reinterpret_cast<ncclComm_t>(ctx->dp_comm),
                        stream);
                    TORCH_CHECK(err == ncclSuccess,
                        "DP n_max all-reduce failed: ", ncclGetErrorString(err));
                }
                if (ctx->tp_comm && ctx->tp_world_size > 1) {
                    auto err = ncclAllReduce(
                        published_n_max.data_ptr<int64_t>(),
                        published_n_max.data_ptr<int64_t>(), 1, ncclInt64,
                        ncclMin, reinterpret_cast<ncclComm_t>(ctx->tp_comm),
                        stream);
                    TORCH_CHECK(err == ncclSuccess,
                        "TP n_max all-reduce failed: ", ncclGetErrorString(err));
                }
                n_max = published_n_max.to(
                    at::TensorOptions().device(at::kCPU)).item<int64_t>();
            }
            n_max = std::min(n_max, total_adapters);
            if (n_max < 1) n_max = 1;
            fprintf(stderr,
                "[train_multi] total=%ld n_max=%ld free=%.1fGB rank=%d\n",
                (long)total_adapters, (long)n_max,
                (double)free_mem / 1e9, lora_rank);
        }

        double total_loss = 0.0;
        int64_t num_chunks = (total_adapters + n_max - 1) / n_max;
        // Chunking is a memory scheduling detail, not an optimizer step. Each
        // tenant clock commits only after its Adam launch succeeds.

        for (int64_t chunk = 0; chunk < num_chunks; chunk++) {
            int64_t start = chunk * n_max;
            int64_t end = std::min(start + n_max, total_adapters);
            int64_t n = end - start;

            // Invalidate cache for this chunk's adapter set
            ctx->lora_batch_valid = false;
            ctx->lora_cache_valid = false;

            // Scope the registry to this activation-memory chunk. The guard
            // restores the full registry even if forward/backward fails.
            AdapterRegistryChunkGuard registry_guard(
                ctx, start, end, !finalize_only);

            double loss_val = 0.0;
            if (!finalize_only) {
                // Mark batched mode active
                ctx->lora_batch_valid = true;  // triggers prepare_lora_batch in forward

            // Expand only a shared batch-1 sample. With a tenant-specific
            // [n_total, seq] batch, preserve each adapter's own row and slice
            // the same chunk range as the adapter registry.
            auto ids_expanded = input_batch == 1
                ? input_ids.repeat({n, 1})
                : input_ids.narrow(0, start, n).contiguous();
            auto mask_expanded = input_batch == 1
                ? target_mask.repeat({n, 1})
                : target_mask.narrow(0, start, n).contiguous();
            if (provided_attention_mask.defined()) {
                ctx->attention_mask = input_batch == 1
                    ? provided_attention_mask.repeat({n, 1})
                    : provided_attention_mask.narrow(0, start, n).contiguous();
                elide_trivial_attention_mask(ctx);
            }

            // Run train_step (reuses existing forward + loss + backward + Adam)
            // But we need to pass the expanded tensors
            auto& input_ref = ids_expanded;
            auto& mask_ref = mask_expanded;

            // Forward — force checkpoint for multi-LoRA (needed for group_inputs)
            bool use_fused = env_enabled("QWEN36_FUSED_LAYER");
            TORCH_CHECK(!use_fused,
                "QWEN36_FUSED_LAYER is disabled until its custom backward preserves the layer graph");
            ctx->use_checkpoint = true;  // force checkpoint for manual_group_backward

            auto t_fwd_start = std::chrono::steady_clock::now();
            auto hidden = use_fused
                ? forward_full_fused(ctx, input_ref)
                : forward_full_checkpoint(ctx, input_ref);
            auto t_fwd_end = std::chrono::steady_clock::now();
            double fwd_ms = std::chrono::duration<double, std::milli>(t_fwd_end - t_fwd_start).count();

            // Batched CE: compute loss with autograd enabled.
            at::Tensor hidden_grad;
            auto t_loss_start = std::chrono::steady_clock::now();
            {
                at::AutoGradMode grad_enable(true);
                // Re-attach hidden to autograd graph
                hidden = hidden.detach().set_requires_grad(true);
                TORCH_CHECK(!env_enabled("QWEN36_FUSED_CE"),
                    "QWEN36_FUSED_CE is disabled until its tile gather and gradient normalization are validated");
                auto loss = compute_loss(
                    ctx, hidden, input_ref, mask_ref, ctx->vocab_size,
                    /*independent_samples=*/true);
                loss_val = loss.value.item<double>();
                hidden_grad = loss.hidden_grad;
            }
            // independent_samples produces a local per-tenant mean. Restore
            // each row to a token-sum numerator before A2A backward so the
            // expert owner can combine unequal source shards correctly.
            if (ctx->expert_parallel && ctx->nccl_comm &&
                env_enabled("QWEN36_EP_A2A_SHARDED")) {
                auto chunk_token_counts = adapter_token_counts
                    .narrow(0, start, n).reshape({n, 1, 1});
                hidden_grad.mul_(chunk_token_counts);
            }
            auto t_loss_end = std::chrono::steady_clock::now();
            double loss_ms = std::chrono::duration<double, std::milli>(t_loss_end - t_loss_start).count();

            // Backward
            auto t_bwd_start = std::chrono::steady_clock::now();
            if (ctx->has_mtp && !env_enabled("QWEN36_DISABLE_MTP")) {
                auto mtp_input = hidden.detach().set_requires_grad(true);
                auto mtp_hidden = mtp_forward(ctx, mtp_input, input_ref);
                auto mtp_loss = mtp_compute_loss(
                    ctx, mtp_hidden, input_ref, mask_ref,
                    /*independent_samples=*/true);
                mtp_loss.backward();
                TORCH_CHECK(mtp_input.grad().defined(), "MTP did not produce a hidden gradient");
                hidden_grad.add_(mtp_input.grad());
                loss_val += mtp_loss.item<double>();
            }

            if (!ctx->group_inputs.empty() && !env_enabled("QWEN36_SUBCKPT")) {
                manual_group_backward(ctx, hidden_grad);
            } else {
                hidden.backward(hidden_grad);
            }
            // The chunk registry contains intrusive copies of the selected
            // adapter tensors. Harvest now; FP32 accumulator contents remain
            // shared when the full registry is restored below.
            harvest_gradient_accumulators(ctx);
            // No cudaDeviceSynchronize — let GPU pipeline run asynchronously.
            // The next chunk's CPU prep (LoRA batch, input expand) will overlap
            // with the tail of this chunk's GPU backward.
            auto t_bwd_end = std::chrono::steady_clock::now();
            double bwd_ms = std::chrono::duration<double, std::milli>(t_bwd_end - t_bwd_start).count();

            fprintf(stderr, "[train_multi] chunk %ld/%ld: n=%ld loss=%f  fwd=%.0fms loss=%.0fms bwd=%.0fms\n",
                    (long)(chunk+1), (long)num_chunks, (long)n, loss_val, fwd_ms, loss_ms, bwd_ms);
            }

            // Restore the complete registry before the next chunk. Gradients
            // remain attached to the intrusive tensor handles, so all chunks
            // can accumulate and the optimizer runs exactly once below.
            registry_guard.restore();

            if (chunk == num_chunks - 1 && !train_only) {
                // DP gradient synchronization and Adam belong to the logical
                // multi-tenant step, never to an activation-memory chunk.
                ++ctx->dynamic_finalizer_count;
                TORCH_CHECK(!finalize_only || !env_enabled(
                        "QWEN36_TEST_FAIL_FINALIZER_BEFORE_TOKEN_PREFLIGHT"),
                    "injected dynamic finalizer failure before token preflight");
                bool local_token_counts_valid = false;
                try {
                    local_token_counts_valid =
                        at::logical_and(
                            at::isfinite(adapter_token_counts),
                            adapter_token_counts >= 0).all().item<bool>();
                } catch (...) {
                    local_token_counts_valid = false;
                }
                bool global_token_counts_valid = local_token_counts_valid;
                if ((ctx->nccl_comm && ctx->ep_world_size > 1) ||
                    (ctx->dp_comm && ctx->dp_world_size > 1) ||
                    (ctx->tp_comm && ctx->tp_world_size > 1)) {
                    global_token_counts_valid =
                        adapter_collective_all_succeeded(
                            ctx, local_token_counts_valid);
                }
                if (finalize_only && finalizer_phase) *finalizer_phase = 1;
                TORCH_CHECK(global_token_counts_valid &&
                        local_token_counts_valid,
                    "dynamic LoRA token counts must be finite, non-negative, "
                    "and valid on every distributed rank");
                std::vector<uint8_t> adapter_has_global_tokens;
                synchronize_lora_gradients(
                    ctx, target_mask, 0.0, &adapter_token_counts,
                    &adapter_has_global_tokens,
                    /*adapter_token_counts_prevalidated=*/true);
                TORCH_CHECK(adapter_has_global_tokens.size() ==
                        ctx->adapters.size(),
                    "dynamic LoRA global-token activity vector mismatch");

                // Build every Adam result out of place. No live parameter,
                // optimizer tensor, or tenant clock changes until the unified
                // launch has passed preflight and CUDA launch validation.
                at::AutoGradMode guard(false);
                ctx->lora_cache_valid = false;
                ctx->lora_batch_valid = false;
                struct DynamicAdamCommit {
                    at::Tensor* param;
                    at::Tensor* accumulator;
                    at::Tensor* m;
                    at::Tensor* v;
                    at::Tensor* next_param;
                    at::Tensor* next_m;
                    at::Tensor* next_v;
                    float lr_scaled;
                    float eps_scaled;
                };
                struct DynamicAdamClockCommit {
                    TrainingContext::LoRAAdapter* adapter;
                    int64_t logical_step;
                };
                std::vector<DynamicAdamCommit> commits;
                std::vector<DynamicAdamClockCommit> clock_commits;
                auto append_commit = [&](at::Tensor& param,
                                         at::Tensor& accumulator,
                                         at::Tensor& m,
                                         at::Tensor& v,
                                         at::Tensor& next_param,
                                         at::Tensor& next_m,
                                         at::Tensor& next_v,
                                         float lr_scaled,
                                         float eps_scaled) {
                    if (!param.requires_grad() || !accumulator.defined()) return;
                    TORCH_CHECK(param.defined() && param.is_cuda() &&
                            param.is_contiguous() &&
                            param.scalar_type() == at::kBFloat16,
                        "dynamic Adam parameter must be contiguous CUDA BF16");
                    TORCH_CHECK(accumulator.is_cuda() &&
                            accumulator.is_contiguous() &&
                            accumulator.scalar_type() == at::kFloat &&
                            accumulator.sizes() == param.sizes() &&
                            accumulator.device() == param.device(),
                        "dynamic Adam accumulator must be matching contiguous CUDA FP32");
                    TORCH_CHECK(m.defined() && v.defined() && m.is_cuda() &&
                            v.is_cuda() && m.is_contiguous() && v.is_contiguous() &&
                            m.scalar_type() == at::kFloat &&
                            v.scalar_type() == at::kFloat &&
                            m.sizes() == param.sizes() &&
                            v.sizes() == param.sizes() &&
                            m.device() == param.device() &&
                            v.device() == param.device(),
                        "dynamic Adam state must be matching contiguous CUDA FP32");
                    TORCH_CHECK(next_param.defined() && next_param.is_cuda() &&
                            next_param.is_contiguous() &&
                            next_param.scalar_type() == at::kBFloat16 &&
                            next_param.sizes() == param.sizes() &&
                            next_param.device() == param.device() &&
                            next_param.requires_grad() == param.requires_grad() &&
                            next_m.defined() && next_v.defined() &&
                            next_m.is_cuda() && next_v.is_cuda() &&
                            next_m.is_contiguous() && next_v.is_contiguous() &&
                            next_m.scalar_type() == at::kFloat &&
                            next_v.scalar_type() == at::kFloat &&
                            next_m.sizes() == param.sizes() &&
                            next_v.sizes() == param.sizes() &&
                            next_m.device() == param.device() &&
                            next_v.device() == param.device(),
                        "dynamic Adam shadow must match live tensor layout");
                    TORCH_CHECK(param.numel() > 0 &&
                            param.numel() <= std::numeric_limits<int>::max(),
                        "dynamic Adam tensor size is outside fused-kernel range: ",
                        param.numel());
                    commits.push_back(DynamicAdamCommit{
                        &param, &accumulator, &m, &v,
                        &next_param, &next_m, &next_v,
                        lr_scaled, eps_scaled});
                };
                for (size_t adapter_index = 0;
                     adapter_index < ctx->adapters.size(); ++adapter_index) {
                    if (!adapter_has_global_tokens[adapter_index]) continue;
                    auto& adapter = ctx->adapters[adapter_index];
                    TORCH_CHECK(adapter.optimizer_step >= 0 &&
                            adapter.optimizer_step <
                                std::numeric_limits<int64_t>::max(),
                        "dynamic Adam optimizer clock is outside valid range for adapter ",
                        adapter.id, ": ", adapter.optimizer_step);
                    const int64_t logical_step = adapter.optimizer_step + 1;
                    const double step_f = (double)logical_step;
                    const double bias_correction1 =
                        1.0 - std::pow(ctx->beta1, step_f);
                    const double bias_correction2 =
                        1.0 - std::pow(ctx->beta2, step_f);
                    const double sqrt_bias_correction2 =
                        std::sqrt(bias_correction2);
                    const float lr_scaled = (float)(
                        ctx->lr * sqrt_bias_correction2 / bias_correction1);
                    const float eps_scaled = (float)(
                        ctx->eps * sqrt_bias_correction2);
                    TORCH_CHECK(std::isfinite(lr_scaled) &&
                            std::isfinite(eps_scaled) &&
                            bias_correction1 > 0.0 &&
                            bias_correction2 > 0.0,
                        "dynamic Adam bias correction is invalid for adapter ",
                        adapter.id, " at logical step ", logical_step);
                    const size_t adapter_commit_begin = commits.size();
                    for (auto& [layer_idx, pairs] : adapter.params) {
                        auto state_it = adapter.adam_state.find(layer_idx);
                        auto shadow_it = adapter.adam_shadow.find(layer_idx);
                        auto accum_it = adapter.grad_accum.find(layer_idx);
                        TORCH_CHECK(state_it != adapter.adam_state.end() &&
                                shadow_it != adapter.adam_shadow.end() &&
                                accum_it != adapter.grad_accum.end() &&
                                state_it->second.size() == pairs.size() &&
                                shadow_it->second.size() == pairs.size() &&
                                accum_it->second.size() == pairs.size(),
                            "dynamic Adam registry layout mismatch for adapter ",
                            adapter.id, " layer ", layer_idx);
                        for (size_t i = 0; i < pairs.size(); ++i) {
                            auto& [a, b] = pairs[i];
                            auto& [m_a, v_a, m_b, v_b] = state_it->second[i];
                            auto& [next_a, next_m_a, next_v_a,
                                   next_b, next_m_b, next_v_b] =
                                shadow_it->second[i];
                            auto& [accum_a, accum_b] = accum_it->second[i];
                            append_commit(
                                a, accum_a, m_a, v_a,
                                next_a, next_m_a, next_v_a,
                                lr_scaled, eps_scaled);
                            append_commit(
                                b, accum_b, m_b, v_b,
                                next_b, next_m_b, next_v_b,
                                lr_scaled, eps_scaled);
                        }
                    }
                    if (commits.size() > adapter_commit_begin) {
                        clock_commits.push_back(
                            DynamicAdamClockCommit{&adapter, logical_step});
                    }
                }
                if (!commits.empty()) {
                    TORCH_CHECK(commits.size() <=
                            static_cast<size_t>(std::numeric_limits<int>::max()),
                        "dynamic Adam tensor count exceeds fused-kernel range");
                    std::vector<void*> h_params, h_grads, h_dst_params;
                    std::vector<float*> h_m, h_v, h_dst_m, h_dst_v;
                    std::vector<int> h_sizes;
                    std::vector<float> h_lr_scaled, h_eps_scaled;
                    const size_t tensor_count = commits.size();
                    h_params.reserve(tensor_count);
                    h_grads.reserve(tensor_count);
                    h_m.reserve(tensor_count);
                    h_v.reserve(tensor_count);
                    h_dst_params.reserve(tensor_count);
                    h_dst_m.reserve(tensor_count);
                    h_dst_v.reserve(tensor_count);
                    h_sizes.reserve(tensor_count);
                    h_lr_scaled.reserve(tensor_count);
                    h_eps_scaled.reserve(tensor_count);
                    for (auto& commit : commits) {
                        h_params.push_back(commit.param->data_ptr());
                        h_grads.push_back(commit.accumulator->data_ptr());
                        h_m.push_back((float*)commit.m->data_ptr());
                        h_v.push_back((float*)commit.v->data_ptr());
                        h_dst_params.push_back(commit.next_param->data_ptr());
                        h_dst_m.push_back((float*)commit.next_m->data_ptr());
                        h_dst_v.push_back((float*)commit.next_v->data_ptr());
                        h_sizes.push_back((int)commit.param->numel());
                        h_lr_scaled.push_back(commit.lr_scaled);
                        h_eps_scaled.push_back(commit.eps_scaled);
                    }
                    const int n_params = (int)commits.size();
                    auto opts_cpu_long = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
                    auto opts_cpu_int  = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
                    auto opts_cpu_float = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
                    auto params_cpu = at::from_blob(h_params.data(), {n_params}, opts_cpu_long);
                    auto grads_cpu  = at::from_blob(h_grads.data(),  {n_params}, opts_cpu_long);
                    auto m_cpu      = at::from_blob(h_m.data(),      {n_params}, opts_cpu_long);
                    auto v_cpu      = at::from_blob(h_v.data(),      {n_params}, opts_cpu_long);
                    auto dst_params_cpu = at::from_blob(
                        h_dst_params.data(), {n_params}, opts_cpu_long);
                    auto dst_m_cpu = at::from_blob(
                        h_dst_m.data(), {n_params}, opts_cpu_long);
                    auto dst_v_cpu = at::from_blob(
                        h_dst_v.data(), {n_params}, opts_cpu_long);
                    auto sizes_cpu  = at::from_blob(h_sizes.data(),  {n_params}, opts_cpu_int);
                    auto lr_cpu = at::from_blob(
                        h_lr_scaled.data(), {n_params}, opts_cpu_float);
                    auto eps_cpu = at::from_blob(
                        h_eps_scaled.data(), {n_params}, opts_cpu_float);
                    ctx->adam_dev_bufs.ensure(n_params, *commits[0].param);
                    ctx->adam_dev_bufs.params_buf.narrow(0, 0, n_params).copy_(params_cpu);
                    ctx->adam_dev_bufs.grads_buf.narrow(0, 0, n_params).copy_(grads_cpu);
                    ctx->adam_dev_bufs.m_buf.narrow(0, 0, n_params).copy_(m_cpu);
                    ctx->adam_dev_bufs.v_buf.narrow(0, 0, n_params).copy_(v_cpu);
                    ctx->adam_dev_bufs.dst_params_buf.narrow(0, 0, n_params).copy_(dst_params_cpu);
                    ctx->adam_dev_bufs.dst_m_buf.narrow(0, 0, n_params).copy_(dst_m_cpu);
                    ctx->adam_dev_bufs.dst_v_buf.narrow(0, 0, n_params).copy_(dst_v_cpu);
                    ctx->adam_dev_bufs.sizes_buf.narrow(0, 0, n_params).copy_(sizes_cpu);
                    ctx->adam_dev_bufs.lr_buf.narrow(0, 0, n_params).copy_(lr_cpu);
                    ctx->adam_dev_bufs.eps_buf.narrow(0, 0, n_params).copy_(eps_cpu);
                    auto stream = c10::cuda::getCurrentCUDAStream().stream();
                    ++ctx->dynamic_adam_launch_count;
                    launch_fused_adam_multi_out_of_place(
                        (void**)ctx->adam_dev_bufs.params_buf.data_ptr(),
                        (void**)ctx->adam_dev_bufs.grads_buf.data_ptr(),
                        (float**)ctx->adam_dev_bufs.m_buf.data_ptr(),
                        (float**)ctx->adam_dev_bufs.v_buf.data_ptr(),
                        (void**)ctx->adam_dev_bufs.dst_params_buf.data_ptr(),
                        (float**)ctx->adam_dev_bufs.dst_m_buf.data_ptr(),
                        (float**)ctx->adam_dev_bufs.dst_v_buf.data_ptr(),
                        (int*)ctx->adam_dev_bufs.sizes_buf.data_ptr(),
                        (float*)ctx->adam_dev_bufs.lr_buf.data_ptr(),
                        (float*)ctx->adam_dev_bufs.eps_buf.data_ptr(),
                        n_params, (float)ctx->beta1, (float)ctx->beta2,
                        (float)(1.0 - ctx->beta1),
                        (float)(1.0 - ctx->beta2),
                        (void*)stream);
                    auto launch_error = cudaGetLastError();
                    TORCH_CHECK(launch_error == cudaSuccess,
                        "dynamic transactional Adam launch failed: ",
                        cudaGetErrorString(launch_error));
                    const bool inject_failure = env_enabled(
                        "QWEN36_TEST_FAIL_DYNAMIC_ADAM_BEFORE_COMMIT");
                    if (inject_failure || env_enabled(
                            "QWEN36_STRICT_DYNAMIC_ADAM_COMMIT")) {
                        auto completion_error = cudaStreamSynchronize(stream);
                        TORCH_CHECK(completion_error == cudaSuccess,
                            "dynamic transactional Adam execution failed: ",
                            cudaGetErrorString(completion_error));
                    }
                    // Production commits remain asynchronous: the next
                    // forward consumes the swapped destinations on this same
                    // stream. Strict mode is available for validation without
                    // imposing a device barrier on every training step.
                    TORCH_CHECK(!inject_failure,
                        "injected dynamic Adam failure before commit");
                    for (auto& commit : commits) {
                        std::swap(*commit.param, *commit.next_param);
                        std::swap(*commit.m, *commit.next_m);
                        std::swap(*commit.v, *commit.next_v);
                    }
                }
                for (const auto& clock_commit : clock_commits) {
                    clock_commit.adapter->optimizer_step =
                        clock_commit.logical_step;
                }
            }

            if (!finalize_only) {
                total_loss += loss_val;
                fprintf(stderr, "[train_multi] chunk %ld/%ld: n=%ld loss=%.6f\n",
                        (long)(chunk + 1), (long)num_chunks, (long)n, loss_val);
            }
        }

        if (!train_only) clear_gradient_accumulators(ctx);
        accumulation_guard.disarmed = true;
        return finalize_only ? 0.0 : total_loss / total_adapters;
    } catch (const std::exception& e) {
        fprintf(stderr, "[train_multi] FAILED: %s\n", e.what());
        return -1.0;
    } catch (...) {
        fprintf(stderr, "[train_multi] FAILED: unknown exception\n");
        return -1.0;
    }
}

__attribute__((visibility("default"))) double qwen36_train_multi_lora(
    void* ctx_ptr,
    void* input_ids_ptr,
    void* target_mask_ptr,
    void* attention_mask_ptr,
    int32_t n_total,
    int32_t lora_rank
) {
    return qwen36_train_multi_lora_impl(
        ctx_ptr, input_ids_ptr, target_mask_ptr, attention_mask_ptr,
        n_total, lora_rank, DynamicMultiLoraMode::TrainAndFinalize);
}

// Train only the requested dynamic tenants. The existing train_multi_lora
// implementation already batches activation-level LoRA projections and owns
// the logical Adam boundary; this wrapper scopes its adapter registry to the
// selected IDs and restores the original order on every exit.
__attribute__((visibility("default"))) double qwen36_train_multi_lora_selected(
    void* ctx_ptr,
    void* input_ids_ptr,
    void* target_mask_ptr,
    void* attention_mask_ptr,
    const int64_t* adapter_ids,
    int32_t n_adapters,
    int32_t lora_rank
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    std::vector<TrainingContext::LoRAAdapter> original;
    std::vector<TrainingContext::LoRAAdapter> selected;
    std::vector<TrainingContext::LoRAAdapter> merged;
    std::vector<size_t> selected_indexes;
    std::vector<uint8_t> moved;
    bool registry_detached = false;
    bool selected_installed = false;
    auto restore_registry = [&]() {
        if (!ctx || !registry_detached) return;
        if (selected_installed) {
            selected.swap(ctx->adapters);
            selected_installed = false;
        }
        for (size_t i = 0; i < original.size(); ++i) {
            if (!moved[i]) {
                merged.push_back(std::move(original[i]));
                continue;
            }
            auto selected_it = std::find(
                selected_indexes.begin(), selected_indexes.end(), i);
            TORCH_CHECK(selected_it != selected_indexes.end(),
                "selected adapter index disappeared: ", i);
            const auto selected_index = static_cast<size_t>(
                selected_it - selected_indexes.begin());
            merged.push_back(std::move(selected[selected_index]));
        }
        ctx->adapters.swap(merged);
        registry_detached = false;
    };
    try {
        TORCH_CHECK(ctx,
            "selected multi-LoRA requires a valid training context");
        TORCH_CHECK(!ctx->topology_invalid,
            "native Qwen context rejected an incompatible TP/DP/EP topology");
        validate_adapter_collective_registry(
            ctx, adapter_ids, n_adapters, lora_rank, false);
        TORCH_CHECK(adapter_ids && n_adapters > 0,
            "selected multi-LoRA requires at least one adapter ID");
        const auto original_count = ctx->adapters.size();
        original.reserve(original_count);
        selected.reserve(n_adapters);
        merged.reserve(original_count);
        selected_indexes.reserve(n_adapters);
        moved.resize(original_count, 0);
        original.swap(ctx->adapters);
        registry_detached = true;
        for (int32_t i = 0; i < n_adapters; ++i) {
            TORCH_CHECK(adapter_ids[i] > 0, "selected adapter IDs must be positive");
            auto it = std::find_if(original.begin(), original.end(),
                [&](const auto& adapter) { return adapter.id == adapter_ids[i]; });
            TORCH_CHECK(it != original.end(), "unknown selected adapter ID: ", adapter_ids[i]);
            const auto index = static_cast<size_t>(it - original.begin());
            TORCH_CHECK(!moved[index], "duplicate selected adapter ID: ", adapter_ids[i]);
            moved[index] = 1;
            selected_indexes.push_back(index);
            selected.push_back(std::move(*it));
        }
        ctx->adapters.swap(selected);
        selected_installed = true;
        const double loss = qwen36_train_multi_lora(
            ctx_ptr, input_ids_ptr, target_mask_ptr, attention_mask_ptr,
            n_adapters, lora_rank);
        restore_registry();
        return loss;
    } catch (const std::exception& e) {
        try { restore_registry(); } catch (...) {}
        fprintf(stderr, "[train_multi_selected] FAILED: %s\n", e.what());
        return -1.0;
    } catch (...) {
        try { restore_registry(); } catch (...) {}
        fprintf(stderr, "[train_multi_selected] FAILED: unknown exception\n");
        return -1.0;
    }
}

static std::string dynamic_adapter_group_key(
    const TrainingContext::LoRAAdapter& adapter
) {
    std::ostringstream key;
    key << "rank=" << adapter.rank;
    key << "|layers=";
    for (const auto layer : adapter.target_layers) key << layer << ",";
    key << "|modules=";
    for (const auto& module : adapter.target_modules) key << module << ",";
    for (const auto& [layer_idx, pairs] : adapter.params) {
        for (size_t pair_idx = 0; pair_idx < pairs.size(); ++pair_idx) {
            const auto& [a, b] = pairs[pair_idx];
            if (!a.requires_grad() && !b.requires_grad()) continue;
            key << "|" << layer_idx << ":" << pair_idx << ":";
            for (const auto size : a.sizes()) key << size << ",";
            key << "/";
            for (const auto size : b.sizes()) key << size << ",";
        }
    }
    return key.str();
}

static void rollback_dynamic_adapter_commit(
    TrainingContext::LoRAAdapter& adapter,
    int64_t optimizer_step
) {
    if (adapter.optimizer_step == optimizer_step) return;
    TORCH_CHECK(adapter.optimizer_step == optimizer_step + 1,
        "heterogeneous rollback observed unexpected optimizer clock for adapter ",
        adapter.id, ": expected ", optimizer_step + 1,
        " got ", adapter.optimizer_step);
    for (auto& [layer_idx, pairs] : adapter.params) {
        auto state_it = adapter.adam_state.find(layer_idx);
        auto shadow_it = adapter.adam_shadow.find(layer_idx);
        TORCH_CHECK(state_it != adapter.adam_state.end() &&
                shadow_it != adapter.adam_shadow.end() &&
                state_it->second.size() == pairs.size() &&
                shadow_it->second.size() == pairs.size(),
            "heterogeneous rollback registry mismatch for adapter ",
            adapter.id, " layer ", layer_idx);
        for (size_t pair_idx = 0; pair_idx < pairs.size(); ++pair_idx) {
            auto& [a, b] = pairs[pair_idx];
            auto& [m_a, v_a, m_b, v_b] = state_it->second[pair_idx];
            auto& [old_a, old_m_a, old_v_a,
                   old_b, old_m_b, old_v_b] = shadow_it->second[pair_idx];
            if (a.requires_grad()) {
                std::swap(a, old_a);
                std::swap(m_a, old_m_a);
                std::swap(v_a, old_v_a);
            }
            if (b.requires_grad()) {
                std::swap(b, old_b);
                std::swap(m_b, old_m_b);
                std::swap(v_b, old_v_b);
            }
        }
    }
    adapter.optimizer_step = optimizer_step;
}

// Heterogeneous selected training groups only adapters whose active tensor
// layouts can be stacked safely. Groups execute forward/backward in canonical
// key order, then the complete selected registry shares one gradient
// synchronization and one transactional Adam finalizer.
__attribute__((visibility("default"))) double
qwen36_train_multi_lora_selected_v2(
    void* ctx_ptr,
    void* input_ids_ptr,
    void* target_mask_ptr,
    void* attention_mask_ptr,
    const int64_t* adapter_ids,
    int32_t n_adapters
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    std::vector<TrainingContext::LoRAAdapter> original;
    std::vector<TrainingContext::LoRAAdapter> selected;
    std::vector<TrainingContext::LoRAAdapter> merged;
    std::vector<size_t> selected_indexes;
    std::vector<uint8_t> moved;
    std::vector<int64_t> original_steps;
    bool registry_detached = false;
    bool selected_registry_installed = false;
    auto restore_registry = [&]() {
        if (!ctx || !registry_detached) return;
        for (size_t index = 0; index < original.size(); ++index) {
            if (!moved[index]) {
                merged.push_back(std::move(original[index]));
                continue;
            }
            auto selected_it = std::find(
                selected_indexes.begin(), selected_indexes.end(), index);
            TORCH_CHECK(selected_it != selected_indexes.end(),
                "heterogeneous selected adapter index disappeared: ", index);
            const size_t selected_index = static_cast<size_t>(
                selected_it - selected_indexes.begin());
            merged.push_back(std::move(selected[selected_index]));
        }
        ctx->adapters.swap(merged);
        registry_detached = false;
    };
    auto rollback = [&]() {
        const size_t rollback_count = std::min(
            selected.size(), original_steps.size());
        for (size_t index = 0; index < rollback_count; ++index) {
            rollback_dynamic_adapter_commit(
                selected[index], original_steps[index]);
        }
    };
    auto recover = [&]() {
        try {
            if (ctx && registry_detached && selected_registry_installed) {
                selected.swap(ctx->adapters);
                selected_registry_installed = false;
            } else if (ctx && registry_detached && !ctx->adapters.empty()) {
                // A failing group returns with its scoped registry installed.
                for (auto& adapter : ctx->adapters) {
                    auto selected_it = std::find_if(
                        selected.begin(), selected.end(),
                        [&](const auto& item) { return item.id == adapter.id; });
                    if (selected_it != selected.end())
                        *selected_it = std::move(adapter);
                }
                ctx->adapters.clear();
            }
        } catch (const std::exception& e) {
            fprintf(stderr,
                "[train_multi_selected_v2] group recovery FAILED: %s\n",
                e.what());
        } catch (...) {
            fprintf(stderr,
                "[train_multi_selected_v2] group recovery FAILED\n");
        }
        try {
            if (ctx && registry_detached) {
                for (auto& adapter : selected)
                    clear_adapter_gradient_accumulators(adapter);
                ctx->group_inputs.clear();
                ctx->group_outputs.clear();
                ctx->lora_cache_valid = false;
                ctx->lora_batch_valid = false;
                ctx->accumulation_active = false;
                ctx->accumulated_token_weight = 0.0;
            }
        } catch (const std::exception& e) {
            fprintf(stderr,
                "[train_multi_selected_v2] gradient cleanup FAILED: %s\n",
                e.what());
        } catch (...) {
            fprintf(stderr,
                "[train_multi_selected_v2] gradient cleanup FAILED\n");
        }
        try {
            rollback();
        } catch (const std::exception& e) {
            fprintf(stderr,
                "[train_multi_selected_v2] rollback FAILED: %s\n", e.what());
        } catch (...) {
            fprintf(stderr, "[train_multi_selected_v2] rollback FAILED\n");
        }
        try {
            restore_registry();
        } catch (const std::exception& e) {
            fprintf(stderr,
                "[train_multi_selected_v2] registry restore FAILED: %s\n",
                e.what());
        } catch (...) {
            fprintf(stderr,
                "[train_multi_selected_v2] registry restore FAILED\n");
        }
    };
    try {
        TORCH_CHECK(ctx,
            "heterogeneous selected training requires a context");
        TORCH_CHECK(!ctx->topology_invalid,
            "native Qwen context rejected an incompatible TP/DP/EP topology");
        const bool local_request_valid = adapter_ids && n_adapters > 0;
        TORCH_CHECK(adapter_collective_all_succeeded(
                ctx, local_request_valid) && local_request_valid,
            "heterogeneous selected training requires rank-consistent "
            "adapter IDs and a positive adapter count");
        const bool local_accumulation_clear = !ctx->accumulation_active &&
            ctx->accumulated_token_weight == 0.0;
        TORCH_CHECK(adapter_collective_all_succeeded(
                ctx, local_accumulation_clear) && local_accumulation_clear,
            "cannot start heterogeneous multi-LoRA while a fixed or dynamic "
            "gradient accumulation window is pending");
        validate_adapter_collective_registry(
            ctx, adapter_ids, n_adapters, 0, false);
        auto* input_ids_tensor = reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto* target_mask_tensor = reinterpret_cast<at::Tensor*>(target_mask_ptr);
        bool local_input_valid = input_ids_tensor && target_mask_tensor;
        if (local_input_valid) {
            local_input_valid = input_ids_tensor->dim() == 2 &&
                target_mask_tensor->dim() == 2 &&
                input_ids_tensor->sizes() == target_mask_tensor->sizes();
        }
        int64_t input_batch = 0;
        if (local_input_valid) {
            input_batch = input_ids_tensor->size(0);
            local_input_valid = input_batch == 1 || input_batch == n_adapters;
        }
        at::Tensor attention_mask;
        if (local_input_valid && attention_mask_ptr) {
            attention_mask = *reinterpret_cast<at::Tensor*>(attention_mask_ptr);
            local_input_valid =
                attention_mask.sizes() == input_ids_tensor->sizes();
        }
        const bool global_input_valid = adapter_collective_all_succeeded(
            ctx, local_input_valid);
        TORCH_CHECK(global_input_valid && local_input_valid,
            "heterogeneous multi-LoRA inputs must be rank-consistent [batch, seq] "
            "tensors with batch 1 or adapter count");
        auto& input_ids = *input_ids_tensor;
        auto& target_mask = *target_mask_tensor;

        const size_t original_count = ctx->adapters.size();
        std::vector<size_t> requested_indexes;
        std::vector<uint8_t> requested(original_count, 0);
        requested_indexes.reserve(n_adapters);
        for (int32_t request_index = 0;
             request_index < n_adapters; ++request_index) {
            TORCH_CHECK(adapter_ids[request_index] > 0,
                "heterogeneous selected adapter IDs must be positive");
            auto it = std::find_if(
                ctx->adapters.begin(), ctx->adapters.end(),
                [&](const auto& adapter) {
                    return adapter.id == adapter_ids[request_index];
                });
            TORCH_CHECK(it != ctx->adapters.end(),
                "unknown heterogeneous selected adapter ID: ",
                adapter_ids[request_index]);
            const size_t index = static_cast<size_t>(
                it - ctx->adapters.begin());
            TORCH_CHECK(!requested[index],
                "duplicate heterogeneous selected adapter ID: ",
                adapter_ids[request_index]);
            requested[index] = 1;
            requested_indexes.push_back(index);
        }
        original.reserve(original_count);
        selected.reserve(n_adapters);
        merged.reserve(original_count);
        selected_indexes.reserve(n_adapters);
        moved.resize(original_count, 0);
        original.swap(ctx->adapters);
        registry_detached = true;
        original_steps.reserve(requested_indexes.size());
        for (const size_t index : requested_indexes) {
            moved[index] = 1;
            selected_indexes.push_back(index);
            original_steps.push_back(original[index].optimizer_step);
            selected.push_back(std::move(original[index]));
        }
        std::map<std::string, std::vector<size_t>> groups;
        for (size_t index = 0; index < selected.size(); ++index) {
            groups[dynamic_adapter_group_key(selected[index])].push_back(index);
        }

        double weighted_loss = 0.0;
        int32_t completed_groups = 0;
        for (const auto& [group_key, indexes] : groups) {
            std::vector<int64_t> row_indexes;
            row_indexes.reserve(indexes.size());
            for (const size_t selected_index : indexes)
                row_indexes.push_back(static_cast<int64_t>(selected_index));

            at::Tensor group_input;
            at::Tensor group_targets;
            at::Tensor group_attention;
            void* group_input_ptr = input_ids_ptr;
            void* group_target_ptr = target_mask_ptr;
            void* group_attention_ptr = attention_mask_ptr;
            bool local_prepared = true;
            try {
                if (input_batch != 1) {
                    auto row_tensor = at::tensor(
                        row_indexes,
                        input_ids.options().dtype(at::kLong));
                    group_input = input_ids.index_select(0, row_tensor);
                    group_targets = target_mask.index_select(0, row_tensor);
                    group_input_ptr = &group_input;
                    group_target_ptr = &group_targets;
                    if (attention_mask.defined()) {
                        group_attention = attention_mask.index_select(
                            0, row_tensor);
                        group_attention_ptr = &group_attention;
                    }
                }
            } catch (...) {
                local_prepared = false;
            }
            TORCH_CHECK(adapter_collective_all_succeeded(ctx, local_prepared),
                "heterogeneous adapter group preparation failed on at least "
                "one distributed rank: ", group_key);

            std::vector<TrainingContext::LoRAAdapter> group;
            group.reserve(indexes.size());
            for (const size_t selected_index : indexes)
                group.push_back(std::move(selected[selected_index]));
            ctx->adapters.swap(group);
            const int32_t group_size = static_cast<int32_t>(indexes.size());
            const int32_t group_rank = static_cast<int32_t>(
                ctx->adapters.front().rank);
            const double group_loss = qwen36_train_multi_lora_impl(
                ctx, group_input_ptr, group_target_ptr, group_attention_ptr,
                group_size, group_rank, DynamicMultiLoraMode::TrainOnly);

            group.swap(ctx->adapters);
            for (size_t group_index = 0;
                 group_index < indexes.size(); ++group_index) {
                selected[indexes[group_index]] =
                    std::move(group[group_index]);
            }
            ++completed_groups;
            bool local_group_succeeded =
                std::isfinite(group_loss) && group_loss >= 0.0;
            const char* fail_after = std::getenv(
                "QWEN36_TEST_FAIL_HETERO_GROUP_AFTER");
            if (fail_after && fail_after[0] != '\0') {
                char* end = nullptr;
                const long requested = std::strtol(fail_after, &end, 10);
                local_group_succeeded = local_group_succeeded &&
                    end && *end == '\0' && requested > 0 &&
                    completed_groups != requested;
            }
            TORCH_CHECK(adapter_collective_all_succeeded(
                    ctx, local_group_succeeded),
                "heterogeneous adapter group failed on at least one "
                "distributed rank: ", group_key);
            weighted_loss += group_loss * indexes.size();
        }

        ctx->adapters.swap(selected);
        selected_registry_installed = true;
        int32_t finalizer_phase = 0;
        const double finalizer_result = qwen36_train_multi_lora_impl(
            ctx, input_ids_ptr, target_mask_ptr, attention_mask_ptr,
            n_adapters, 0, DynamicMultiLoraMode::FinalizeOnly,
            &finalizer_phase);
        selected.swap(ctx->adapters);
        selected_registry_installed = false;
        if (finalizer_phase < 1) {
            // Match peers already waiting in the token preflight phase when
            // this rank failed during finalizer-local preparation.
            adapter_collective_all_succeeded(ctx, false);
        }
        const bool local_finalizer_succeeded =
            std::isfinite(finalizer_result) && finalizer_result >= 0.0;
        TORCH_CHECK(adapter_collective_all_succeeded(
                ctx, local_finalizer_succeeded),
            "heterogeneous adapter finalizer failed on at least one "
            "distributed rank");
        restore_registry();
        return weighted_loss / static_cast<double>(n_adapters);
    } catch (const std::exception& e) {
        recover();
        fprintf(stderr, "[train_multi_selected_v2] FAILED: %s\n", e.what());
        return -1.0;
    } catch (...) {
        recover();
        fprintf(stderr,
            "[train_multi_selected_v2] FAILED: unknown exception\n");
        return -1.0;
    }
}

// Set NCCL communicator for Expert Parallel all-reduce
// Creates NCCL communicator directly in C++ using env vars RANK/WORLD_SIZE.
// Rank 0 generates unique ID and writes to /tmp/rustrain-nccl/nccl-id.bin
// Other ranks read it. All ranks call ncclCommInitRank.
// Returns 0 on success, -1 on failure.
// Process-level NCCL singleton — created once, reused across sessions.
static ncclComm_t g_nccl_comm = nullptr;
static cudaStream_t g_nccl_stream = nullptr;
static ncclComm_t g_tp_comm = nullptr;
static cudaStream_t g_tp_stream = nullptr;
static ncclComm_t g_ep_comm = nullptr;
static cudaStream_t g_ep_stream = nullptr;
static ncclComm_t g_dp_comm = nullptr;
static cudaStream_t g_dp_stream = nullptr;
static bool g_nccl_initialized = false;
static bool g_nccl_cleanup_registered = false;
static int g_parallel_rank = 0;
static int g_parallel_world_size = 1;
static int g_parallel_tp_rank = 0;
static int g_parallel_tp_size = 1;
static int g_parallel_tp_color = 0;
static int g_parallel_ep_rank = 0;
static int g_parallel_ep_size = 1;
static int g_parallel_ep_color = 0;
static int g_parallel_dp_rank = 0;
static int g_parallel_dp_size = 1;
static int g_parallel_dp_color = 0;

static void qwen36_destroy_process_communicators() {
    if (g_dp_comm) ncclCommDestroy(g_dp_comm);
    if (g_ep_comm) ncclCommDestroy(g_ep_comm);
    if (g_tp_comm) ncclCommDestroy(g_tp_comm);
    if (g_nccl_comm) ncclCommDestroy(g_nccl_comm);
    g_dp_comm = nullptr;
    g_ep_comm = nullptr;
    g_tp_comm = nullptr;
    g_nccl_comm = nullptr;
    g_nccl_initialized = false;
}

static bool same_cached_parallel_topology(
    int rank, int world_size,
    int tp_rank, int tp_size, int tp_color,
    int ep_rank, int ep_size, int ep_color,
    int dp_rank, int dp_size, int dp_color
) {
    return g_parallel_rank == rank &&
        g_parallel_world_size == world_size &&
        g_parallel_tp_rank == tp_rank &&
        g_parallel_tp_size == tp_size &&
        g_parallel_tp_color == tp_color &&
        g_parallel_ep_rank == ep_rank &&
        g_parallel_ep_size == ep_size &&
        g_parallel_ep_color == ep_color &&
        g_parallel_dp_rank == dp_rank &&
        g_parallel_dp_size == dp_size &&
        g_parallel_dp_color == dp_color;
}
static int g_cuda_device = 0;

// Set CUDA device — called from Rust worker before any GPU operation.
// Ensures PyTorch initializes CUDA context on the correct device.
__attribute__((visibility("default"))) void qwen36_set_cuda_device(int32_t device) {
    // Use PyTorch's device API — this updates both cudaSetDevice AND
    // PyTorch's internal device tracking (c10::cuda::current_device).
    // Must be called before any GPU operation in exec'd worker processes.
    c10::cuda::set_device(device);
    cudaSetDevice(device);
    g_cuda_device = device;
    // Force PyTorch to create CUDA context on this device
    auto opts = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device);
    auto dummy = at::empty({1}, opts);
    dummy.sizes();  // touch to ensure materialization
}

static int32_t qwen36_init_parallel_nccl_impl(
    void* ctx_ptr,
    int rank, int world_size,
    int tp_rank, int tp_size, int tp_color,
    int ep_rank, int ep_size, int ep_color,
    int dp_rank, int dp_size, int dp_color,
    bool synchronize_parameters
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);

    if (!ctx) return -1;

    if (rank < 0 || rank >= world_size || world_size <= 0 ||
        tp_rank < 0 || tp_rank >= tp_size || tp_size <= 0 || tp_color < 0 ||
        ep_rank < 0 || ep_rank >= ep_size || ep_size <= 0 || ep_color < 0 ||
        dp_rank < 0 || dp_rank >= dp_size || dp_size <= 0 || dp_color < 0 ||
        world_size != tp_size * ep_size * dp_size) {
        ctx->topology_invalid = true;
        fprintf(stderr,
            "[parallel_nccl] invalid topology: rank=%d world=%d tp=%d/%d color=%d "
            "ep=%d/%d color=%d dp=%d/%d color=%d\n",
            rank, world_size, tp_rank, tp_size, tp_color,
            ep_rank, ep_size, ep_color, dp_rank, dp_size, dp_color);
        return -1;
    }
    if (ctx->tp_world_size != tp_size || ctx->ep_world_size != ep_size ||
        ctx->dp_world_size != dp_size ||
        ctx->expert_parallel != (ep_size > 1) ||
        ctx->data_parallel != (dp_size > 1)) {
        ctx->topology_invalid = true;
        fprintf(stderr,
            "[parallel_nccl] init topology does not match context: "
            "context tp=%d ep=%d dp=%d, init tp=%d ep=%d dp=%d\n",
            ctx->tp_world_size, ctx->ep_world_size, ctx->dp_world_size,
            tp_size, ep_size, dp_size);
        return -1;
    }
    ctx->topology_invalid = false;

    // If already initialized, just set the pointer on this context
    if (g_nccl_initialized) {
        if (!same_cached_parallel_topology(
                rank, world_size, tp_rank, tp_size, tp_color,
                ep_rank, ep_size, ep_color,
                dp_rank, dp_size, dp_color)) {
            ctx->topology_invalid = true;
            fprintf(stderr,
                "[parallel_nccl] process communicator topology cannot change after initialization\n");
            return -1;
        }
        ctx->expert_parallel = ep_size > 1;
        ctx->ep_rank = ep_rank;
        ctx->ep_world_size = ep_size;
        ctx->data_parallel = dp_size > 1;
        ctx->dp_rank = dp_rank;
        ctx->dp_world_size = dp_size;
        const char* local_rank_str2 = getenv("LOCAL_RANK");
        ctx->cuda_device = local_rank_str2 ? atoi(local_rank_str2) : g_cuda_device;
        ctx->tp_world_size = tp_size;
        ctx->tp_rank = tp_rank;
        ctx->topology_invalid = false;
        ctx->tp_comm = tp_size > 1 ? g_tp_comm : nullptr;
        ctx->tp_stream = tp_size > 1 ? g_tp_stream : nullptr;
        ctx->nccl_comm = ep_size > 1 ? g_ep_comm : nullptr;
        ctx->nccl_stream = ep_size > 1 ? g_ep_stream : nullptr;
        ctx->dp_comm = dp_size > 1 ? g_dp_comm : nullptr;
        ctx->dp_stream = dp_size > 1 ? g_dp_stream : nullptr;
        void* layer_comm = ep_size > 1 ? (void*)g_ep_comm : nullptr;
        void* layer_stream = ep_size > 1 ? (void*)g_ep_stream : nullptr;
        for (auto& lc : ctx->layer_configs) {
            lc.nccl_comm = layer_comm;
            lc.nccl_stream = layer_stream;
        }
        for (auto& lc : ctx->mtp_layer_configs) {
            lc.nccl_comm = layer_comm;
            lc.nccl_stream = layer_stream;
        }
        if (synchronize_parameters) {
            validate_fixed_collective_registry(ctx);
            synchronize_fixed_replicated_lora_parameters(ctx);
        }
        return 0;
    }
    if (world_size <= 1) return 0;  // no EP needed

    // Set CUDA device and initialize PyTorch CUDA context on this device.
    // NCCL communicator binds to the current CUDA context. If PyTorch hasn't
    // initialized CUDA on this device yet, NCCL gets a wrong context → "invalid argument".
    // Creating a dummy tensor forces PyTorch to initialize its CUDA context.
    const char* local_rank_str = getenv("LOCAL_RANK");
    int local_rank = local_rank_str ? atoi(local_rank_str) : rank;
    c10::cuda::set_device(local_rank);
    cudaSetDevice(local_rank);
    g_cuda_device = local_rank;
    ctx->cuda_device = local_rank;
    {
        // Force PyTorch CUDA context initialization on this device
        auto opts = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, local_rank);
        auto dummy = at::empty({1}, opts);
        dummy.sizes();  // touch to ensure materialization
    }

    // Exchange unique ID via file
    // Rank 0 generates ID, writes to file, then writes "ready" sentinel.
    // Other ranks wait for "ready" file, then read ID.
    // This ensures rank 0's write is visible before others read.
    const std::string rendezvous_dir = nccl_sync_dir();
    const std::string id_path = rendezvous_dir + "/nccl-id.bin";
    const std::string ready_path = rendezvous_dir + "/nccl-ready.txt";
    ncclUniqueId unique_id;
    if (rank == 0) {
        // Clean up old files first
        remove(ready_path.c_str());
        for (int peer = 0; peer < world_size; ++peer) {
            const std::string stale_barrier =
                rendezvous_dir + "/barrier_" + std::to_string(peer);
            remove(stale_barrier.c_str());
        }
        ncclGetUniqueId(&unique_id);
        FILE* f = fopen(id_path.c_str(), "wb");
        fwrite(&unique_id, sizeof(unique_id), 1, f);
        fclose(f);
        // Write ready sentinel AFTER id file
        FILE* rf = fopen(ready_path.c_str(), "w");
        fprintf(rf, "ready\n");
        fclose(rf);
    } else {
        // Wait for ready sentinel
        for (int i = 0; i < 600; i++) {
            FILE* rf = fopen(ready_path.c_str(), "r");
            if (rf) { fclose(rf); break; }
            usleep(10000);  // 10ms
        }
        // Now read ID file
        FILE* f = fopen(id_path.c_str(), "rb");
        if (!f || fread(&unique_id, sizeof(unique_id), 1, f) != 1) {
            fprintf(stderr, "[ep_nccl] rank %d: failed to read ID file\n", rank);
            if (f) fclose(f);
            return -1;
        }
        fclose(f);
    }

    // Barrier: ensure all ranks reach ncclCommInitRank simultaneously.
    // Without this, rank 0 (fast load_model) reaches ncclCommInitRank before
    // rank 3 (slow load_model) → NCCL timeout.
    {
        const char* barrier_dir = rendezvous_dir.c_str();
        char bpath[256];
        snprintf(bpath, sizeof(bpath), "%s/barrier_%d", barrier_dir, rank);
        FILE* bf = fopen(bpath, "w"); fprintf(bf, "1\n"); fclose(bf);
        // Wait for all ranks
        for (int i = 0; i < world_size; i++) {
            char p[256];
            snprintf(p, sizeof(p), "%s/barrier_%d", barrier_dir, i);
            for (int w = 0; w < 6000; w++) {  // 60s timeout
                FILE* f2 = fopen(p, "r");
                if (f2) { fclose(f2); break; }
                usleep(10000);
            }
        }
    }

    // Initialize communicator — only once per process
    ncclComm_t comm;
    ncclResult_t err = ncclCommInitRank(&comm, world_size, unique_id, rank);
    if (err != ncclSuccess) {
        fprintf(stderr, "[ep_nccl] ncclCommInitRank failed: %d (%s)\n", err, ncclGetErrorString(err));
        return -1;
    }

    // Use PyTorch's compute stream for NCCL — NOT a separate stream.
    // Separate stream causes "invalid argument" because NCCL communicator
    // is bound to a CUDA context, and PyTorch's caching allocator stream
    // may be on a different context. Using the same stream ensures same context.
    // We store nullptr for stream — moe_forward will use getCurrentCUDAStream(dev).
    cudaStream_t nccl_stream = nullptr;  // nullptr = use default stream

    // Store as process-level singleton
    g_nccl_comm = comm;
    g_nccl_stream = nccl_stream;
    if (tp_size <= 0 || world_size % tp_size != 0) {
        fprintf(stderr, "[tp_nccl] invalid TP_SIZE=%d for WORLD_SIZE=%d\n",
            tp_size, world_size);
        return -1;
    }
    if (tp_size > 1) {
        ncclResult_t tp_err = ncclCommSplit(
            comm, tp_color, tp_rank, &g_tp_comm, nullptr);
        if (tp_err != ncclSuccess) {
            fprintf(stderr, "[tp_nccl] ncclCommSplit failed: %d (%s)\n",
                tp_err, ncclGetErrorString(tp_err));
            ncclCommDestroy(comm);
            g_nccl_comm = nullptr;
            return -1;
        }
        g_tp_stream = nccl_stream;
    }
    if (ep_size > 1) {
        ncclResult_t ep_err = ncclCommSplit(
            comm, ep_color, ep_rank, &g_ep_comm, nullptr);
        if (ep_err != ncclSuccess) {
            fprintf(stderr, "[ep_nccl] ncclCommSplit failed: %d (%s)\n",
                ep_err, ncclGetErrorString(ep_err));
            if (g_tp_comm) ncclCommDestroy(g_tp_comm);
            ncclCommDestroy(comm);
            g_tp_comm = nullptr;
            g_nccl_comm = nullptr;
            return -1;
        }
        g_ep_stream = nccl_stream;
    }
    if (dp_size > 1) {
        ncclResult_t dp_err = ncclCommSplit(
            comm, dp_color, dp_rank, &g_dp_comm, nullptr);
        if (dp_err != ncclSuccess) {
            fprintf(stderr, "[dp_nccl] ncclCommSplit failed: %d (%s)\n",
                dp_err, ncclGetErrorString(dp_err));
            if (g_ep_comm) ncclCommDestroy(g_ep_comm);
            if (g_tp_comm) ncclCommDestroy(g_tp_comm);
            ncclCommDestroy(comm);
            g_ep_comm = nullptr;
            g_tp_comm = nullptr;
            g_nccl_comm = nullptr;
            return -1;
        }
        g_dp_stream = nccl_stream;
    }
    g_parallel_rank = rank;
    g_parallel_world_size = world_size;
    g_parallel_tp_rank = tp_rank;
    g_parallel_tp_size = tp_size;
    g_parallel_tp_color = tp_color;
    g_parallel_ep_rank = ep_rank;
    g_parallel_ep_size = ep_size;
    g_parallel_ep_color = ep_color;
    g_parallel_dp_rank = dp_rank;
    g_parallel_dp_size = dp_size;
    g_parallel_dp_color = dp_color;
    g_nccl_initialized = true;
    if (!g_nccl_cleanup_registered) {
        std::atexit(qwen36_destroy_process_communicators);
        g_nccl_cleanup_registered = true;
    }

    ctx->nccl_comm = ep_size > 1 ? g_ep_comm : nullptr;
    ctx->nccl_stream = ep_size > 1 ? g_ep_stream : nullptr;
    ctx->ep_rank = ep_rank;
    ctx->ep_world_size = ep_size;
    ctx->expert_parallel = ep_size > 1;
    ctx->dp_comm = dp_size > 1 ? g_dp_comm : nullptr;
    ctx->dp_stream = dp_size > 1 ? g_dp_stream : nullptr;
    ctx->dp_rank = dp_rank;
    ctx->dp_world_size = dp_size;
    ctx->data_parallel = dp_size > 1;
    ctx->tp_world_size = tp_size;
    ctx->tp_rank = tp_rank;
    ctx->tp_comm = tp_size > 1 ? g_tp_comm : nullptr;
    ctx->tp_stream = tp_size > 1 ? g_tp_stream : nullptr;

    // Propagate to layer configs
    void* layer_comm = ep_size > 1 ? (void*)g_ep_comm : nullptr;
    void* layer_stream = ep_size > 1 ? (void*)g_ep_stream : nullptr;
    for (auto& lc : ctx->layer_configs) {
        lc.nccl_comm = layer_comm;
        lc.nccl_stream = layer_stream;
    }
    for (auto& lc : ctx->mtp_layer_configs) {
        lc.nccl_comm = layer_comm;
        lc.nccl_stream = layer_stream;
    }

    if (synchronize_parameters) {
        validate_fixed_collective_registry(ctx);
        synchronize_fixed_replicated_lora_parameters(ctx);
    }
    return 0;
}

__attribute__((visibility("default"))) int32_t qwen36_init_parallel_nccl(
    void* ctx_ptr,
    int32_t rank, int32_t world_size,
    int32_t tp_rank, int32_t tp_size, int32_t tp_color,
    int32_t ep_rank, int32_t ep_size, int32_t ep_color,
    int32_t dp_rank, int32_t dp_size, int32_t dp_color
) {
    return qwen36_init_parallel_nccl_impl(
        ctx_ptr, rank, world_size,
        tp_rank, tp_size, tp_color,
        ep_rank, ep_size, ep_color,
        dp_rank, dp_size, dp_color,
        /*synchronize_parameters=*/true);
}

// Attach a shadow restore context to process-cached communicators without
// broadcasting its temporary random LoRA initialization. Checkpoint tensors
// replace every active parameter before the context can become live.
__attribute__((visibility("default"))) int32_t qwen36_attach_parallel_nccl_no_sync(
    void* ctx_ptr,
    int32_t rank, int32_t world_size,
    int32_t tp_rank, int32_t tp_size, int32_t tp_color,
    int32_t ep_rank, int32_t ep_size, int32_t ep_color,
    int32_t dp_rank, int32_t dp_size, int32_t dp_color
) {
    if (world_size > 1 && !g_nccl_initialized) {
        fprintf(stderr,
            "[parallel_nccl] restore attach requires initialized process communicators\n");
        return -1;
    }
    return qwen36_init_parallel_nccl_impl(
        ctx_ptr, rank, world_size,
        tp_rank, tp_size, tp_color,
        ep_rank, ep_size, ep_color,
        dp_rank, dp_size, dp_color,
        /*synchronize_parameters=*/false);
}

__attribute__((visibility("default"))) int32_t qwen36_init_nccl(
    void* ctx_ptr
) {
    const char* rank_str = getenv("RANK");
    const char* world_str = getenv("WORLD_SIZE");
    if (!rank_str || !world_str) return -1;
    const int rank = atoi(rank_str);
    const int world_size = atoi(world_str);
    const char* tp_size_str = getenv("TP_SIZE");
    if (!tp_size_str) tp_size_str = getenv("RUSTRAIN_TP_SIZE");
    const int tp_size = tp_size_str ? atoi(tp_size_str) : 1;
    const bool data_parallel = env_enabled("RUSTRAIN_DATA_PARALLEL");
    const char* ep_size_str = getenv("EP_SIZE");
    if (!ep_size_str) ep_size_str = getenv("RUSTRAIN_EP_SIZE");
    const int ep_size = ep_size_str ? atoi(ep_size_str) : 1;
    const char* dp_size_str = getenv("DP_SIZE");
    if (!dp_size_str) dp_size_str = getenv("RUSTRAIN_DP_SIZE");
    const int dp_size = dp_size_str
        ? atoi(dp_size_str)
        : (data_parallel && tp_size > 0 && ep_size > 0
            ? world_size / (tp_size * ep_size) : 1);
    const int tp_rank = tp_size > 0 ? rank % tp_size : 0;
    const int ep_rank = tp_size > 0 && ep_size > 0
        ? (rank / tp_size) % ep_size : 0;
    const int dp_rank = tp_size > 0 && ep_size > 0
        ? rank / (tp_size * ep_size) : 0;
    return qwen36_init_parallel_nccl_impl(
        ctx_ptr, rank, world_size,
        tp_rank, tp_size, rank / std::max(tp_size, 1),
        ep_rank, ep_size, dp_rank * std::max(tp_size, 1) + tp_rank,
        dp_rank, dp_size, ep_rank * std::max(tp_size, 1) + tp_rank,
        /*synchronize_parameters=*/true);
}

// Set NCCL communicator for Expert Parallel all-reduce (legacy, from Rust)
__attribute__((visibility("default"))) void qwen36_set_nccl_comm(
    void* ctx_ptr, void* comm_ptr, void* stream_ptr,
    int32_t ep_rank, int32_t ep_world_size
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    ctx->data_parallel = env_enabled("RUSTRAIN_DATA_PARALLEL");
    if (ctx->tp_world_size <= 0 || ctx->tp_world_size > 1) {
        ctx->topology_invalid = true;
        fprintf(stderr,
            "[parallel_nccl] legacy setter only supports TP_SIZE=1: "
            "TP_SIZE=%d WORLD_SIZE=%d DATA_PARALLEL=%d\n",
            ctx->tp_world_size, ep_world_size, ctx->data_parallel ? 1 : 0);
        return;
    }
    if (ctx->data_parallel) {
        ctx->dp_comm = reinterpret_cast<ncclComm_t>(comm_ptr);
        ctx->dp_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        ctx->dp_rank = ep_rank;
        ctx->dp_world_size = ep_world_size;
        ctx->nccl_comm = nullptr;
        ctx->nccl_stream = nullptr;
        ctx->ep_rank = 0;
        ctx->ep_world_size = 1;
        ctx->expert_parallel = false;
    } else {
        ctx->nccl_comm = reinterpret_cast<ncclComm_t>(comm_ptr);
        ctx->nccl_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        ctx->ep_rank = ep_rank;
        ctx->ep_world_size = ep_world_size;
        ctx->expert_parallel = ep_world_size > 1;
        ctx->dp_comm = nullptr;
        ctx->dp_stream = nullptr;
        ctx->dp_rank = 0;
        ctx->dp_world_size = 1;
    }
    ctx->topology_invalid = false;
    int current_device = g_cuda_device;
    cudaGetDevice(&current_device);
    ctx->cuda_device = current_device;
    // Only EP owns routed-output collectives. In pure replicated DP the world
    // communicator belongs exclusively to LoRA gradient synchronization;
    // exposing it to moe_forward would mix activations from unrelated samples.
    void* layer_comm = ctx->expert_parallel ? comm_ptr : nullptr;
    void* layer_stream = ctx->expert_parallel ? stream_ptr : nullptr;
    for (auto& lc : ctx->layer_configs) {
        lc.nccl_comm = layer_comm;
        lc.nccl_stream = layer_stream;
    }
    for (auto& lc : ctx->mtp_layer_configs) {
        lc.nccl_comm = layer_comm;
        lc.nccl_stream = layer_stream;
    }
}

// Enable/disable gradient checkpointing
__attribute__((visibility("default"))) void qwen36_set_checkpoint(void* ctx_ptr, int32_t enable, int64_t group_size) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    ctx->use_checkpoint = (enable != 0);
    ctx->group_size = (group_size > 0) ? group_size : 4;
    fprintf(stderr, "[q36_ctx] checkpoint: %s, group_size=%ld\n",
        ctx->use_checkpoint ? "ON" : "OFF", (long)ctx->group_size);
}

// Set attention mask for padding tokens
__attribute__((visibility("default"), used))
void qwen36_set_attention_mask(void* ctx_ptr, void* mask_ptr) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    if (mask_ptr) {
        auto& mask = *reinterpret_cast<at::Tensor*>(mask_ptr);
        validate_linear_attention_mask(ctx, mask);
        ctx->attention_mask = mask;
    }
}

// Utility functions (kept for compatibility)
__attribute__((visibility("default"))) void* qwen36_gemm(void* a_ptr, void* b_ptr, int transpose_b) {
    auto& a = *reinterpret_cast<at::Tensor*>(a_ptr);
    auto& b = *reinterpret_cast<at::Tensor*>(b_ptr);
    if (transpose_b) return new at::Tensor(at::matmul(a, b.t()));
    return new at::Tensor(at::matmul(a, b));
}

__attribute__((visibility("default"))) void qwen36_free_tensor(void* tensor_ptr) {
    if (tensor_ptr) delete reinterpret_cast<at::Tensor*>(tensor_ptr);
}

// ── Multi-LoRA adapter management ──

__attribute__((visibility("default")))
int64_t qwen36_add_lora(
    void* ctx_ptr,
    int64_t rank, double alpha,
    const int64_t* target_layers, int64_t num_target_layers,
    const char* target_modules_str
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        TORCH_CHECK(ctx, "dynamic LoRA registration requires a context");
        TrainingContext::LoRAAdapter adapter{};
        int64_t local_rank = 0;
        std::string request_error;
        try {
            TORCH_CHECK(!ctx->topology_invalid,
                "native Qwen context rejected an incompatible distributed "
                "topology");
            TORCH_CHECK(!ctx->accumulation_active &&
                    ctx->accumulated_token_weight == 0.0,
                "cannot mutate the dynamic LoRA registry while a gradient "
                "accumulation window is pending; finalize or abort it first");
            TORCH_CHECK(rank > 0, "LoRA rank must be positive");
            TORCH_CHECK(std::isfinite(alpha) && alpha > 0.0,
                "LoRA alpha must be finite and positive");
            TORCH_CHECK(num_target_layers >= 0 &&
                    (num_target_layers == 0 || target_layers),
                "dynamic LoRA target layer list is invalid");
            TORCH_CHECK(ctx->next_adapter_id <
                    std::numeric_limits<int64_t>::max(),
                "dynamic LoRA adapter ID space is exhausted");
            adapter.id = ctx->next_adapter_id + 1;
            adapter.rank = rank;
            adapter.alpha = alpha;
            for (int64_t i = 0; i < num_target_layers; i++)
                adapter.target_layers.insert(target_layers[i]);
            if (target_modules_str) {
                std::string s(target_modules_str);
                std::stringstream ss(s);
                std::string item;
                while (std::getline(ss, item, ','))
                    adapter.target_modules.insert(item);
            }
            for (auto layer : adapter.target_layers) {
                TORCH_CHECK(layer >= 0 && layer < ctx->num_layers,
                    "dynamic LoRA target layer out of range: ", layer,
                    " for model with ", ctx->num_layers, " layers");
            }
            // The activation-level batch path stacks A/B across adapters.
            // Keep the batch rectangular and semantically aligned instead of
            // waiting for an opaque stack failure during the first step.
            if (!ctx->adapters.empty() &&
                !ctx->restore_without_parameter_sync &&
                !ctx->allow_heterogeneous_registration) {
                const auto& reference = ctx->adapters.front();
                TORCH_CHECK(rank == reference.rank,
                    "dynamic LoRA adapters in one batch must use the same rank");
                TORCH_CHECK(adapter.target_layers == reference.target_layers,
                    "dynamic LoRA adapters in one batch must use identical "
                    "target_layers");
                TORCH_CHECK(adapter.target_modules == reference.target_modules,
                    "dynamic LoRA adapters in one batch must use identical "
                    "target_modules");
            }
            for (const auto& name : adapter.target_modules) {
                TORCH_CHECK(
                    name == "q_proj" || name == "k_proj" ||
                    name == "v_proj" || name == "o_proj" ||
                    name == "in_proj_qkv" || name == "in_proj_z" ||
                    name == "in_proj_a" || name == "in_proj_b" ||
                    name == "out_proj" || name == "gate_proj" ||
                    name == "up_proj" || name == "down_proj" ||
                    name == "shared_gate_proj" ||
                    name == "shared_up_proj" ||
                    name == "shared_down_proj" ||
                    name == "experts_gate_up_proj" ||
                    name == "experts_down_proj",
                    "unsupported dynamic Qwen LoRA target module: ", name);
                bool resolved = false;
                for (const auto& layer_cfg : ctx->layer_configs) {
                    auto table = lora_projection_table(layer_cfg);
                    for (int64_t pair = 0; pair < table.count; ++pair) {
                        if (name == table.entries[pair].name) {
                            resolved = true;
                            break;
                        }
                    }
                    if (resolved) break;
                }
                TORCH_CHECK(resolved,
                    "dynamic LoRA target module does not exist in this model: ",
                    name);
            }
            local_rank = local_lora_rank_for_active_targets(
                ctx, rank, adapter.target_layers, adapter.target_modules,
                /*empty_modules_mean_attention_only=*/true, "dynamic");
        } catch (const std::exception& e) {
            request_error = e.what();
        }
        const bool local_request_valid = request_error.empty();
        const bool request_matches = adapter_registration_phase_matches(
            ctx, adapter, local_request_valid, /*phase=*/0);
        TORCH_CHECK(request_matches && local_request_valid,
            "dynamic LoRA registration request is invalid or differs across "
            "distributed ranks", request_error.empty() ? "" : ": ",
            request_error);

        std::string preparation_error;
        try {
            for (int64_t i = 0; i < ctx->num_layers; i++) {
                if (!adapter.target_layers.empty() &&
                    adapter.target_layers.find(i) ==
                        adapter.target_layers.end())
                    continue;
                int64_t w_offset = 0;
                for (int64_t j = 0; j < i; j++)
                    w_offset += weight_count_for_layer(ctx->layer_configs[j]);
                auto projection_table =
                    lora_projection_table(ctx->layer_configs[i]);
                int64_t num_pairs = projection_table.count;
                std::vector<std::pair<at::Tensor, at::Tensor>> pairs;
                std::vector<std::array<at::Tensor, 4>> adam_states;
                std::vector<std::array<at::Tensor, 6>> adam_shadows;
                std::vector<std::array<at::Tensor, 2>> grad_accumulators;
                for (int64_t k = 0; k < num_pairs; k++) {
                    const auto& projection = projection_table.entries[k];
                    auto* base =
                        ctx->weight_ptrs[w_offset + projection.weight_index];
                    // Preserve the historical empty-target default (attention
                    // projections only). Explicit lists may select any 2D
                    // dense/shared MLP projection.
                    bool active = adapter.target_modules.empty()
                        ? !projection.grouped_expert &&
                            projection.segment == LoraSegment::Attention
                        : adapter.target_modules.find(projection.name) !=
                            adapter.target_modules.end();
                    auto opts = at::TensorOptions().dtype(ctx->compute_type)
                        .device(base->device());
                    at::Tensor a, b;
                    if (active) {
                        if (projection.grouped_expert) {
                            TORCH_CHECK(base->dim() == 3,
                                "dynamic routed-expert LoRA projection must "
                                "be rank 3: ", projection.name);
                            int64_t experts = base->size(0);
                            int64_t out_f = base->size(1);
                            int64_t in_f = base->size(2);
                            const auto layout = lora_tp_layout(ctx, i, k);
                            if (layout == LoraTpLayout::ColumnParallel ||
                                layout == LoraTpLayout::RowParallel) {
                                a = at::randn(
                                    {experts, rank, in_f}, opts) * 0.01;
                                b = at::zeros(
                                    {experts, out_f, rank}, opts);
                            } else {
                                a = initialize_lora_a(
                                    ctx, opts, experts, rank, in_f);
                                b = at::zeros(
                                    {experts, out_f, local_rank}, opts);
                            }
                        } else {
                            TORCH_CHECK(base->dim() == 2,
                                "dynamic LoRA projection must be a matrix: ",
                                projection.name);
                            int64_t out_f = base->size(0);
                            int64_t in_f = base->size(1);
                            const auto layout = lora_tp_layout(ctx, i, k);
                            if (layout == LoraTpLayout::ColumnParallel ||
                                layout == LoraTpLayout::RowParallel) {
                                a = at::randn({rank, in_f}, opts) * 0.01;
                                b = at::zeros({out_f, rank}, opts);
                            } else {
                                a = initialize_lora_a(
                                    ctx, opts, 0, rank, in_f);
                                b = at::zeros({out_f, local_rank}, opts);
                            }
                        }
                    } else {
                        a = at::zeros({}, opts);
                        b = at::zeros({}, opts);
                    }
                    a.set_requires_grad(active);
                    b.set_requires_grad(active);
                    auto opts_f32 = at::TensorOptions().dtype(at::kFloat)
                        .device(base->device());
                    adam_states.push_back({
                        at::zeros(a.sizes(), opts_f32),
                        at::zeros(a.sizes(), opts_f32),
                        at::zeros(b.sizes(), opts_f32),
                        at::zeros(b.sizes(), opts_f32)});
                    adam_shadows.push_back(active
                        ? std::array<at::Tensor, 6>{
                            at::empty_like(a).set_requires_grad(true),
                            at::empty(a.sizes(), opts_f32),
                            at::empty(a.sizes(), opts_f32),
                            at::empty_like(b).set_requires_grad(true),
                            at::empty(b.sizes(), opts_f32),
                            at::empty(b.sizes(), opts_f32)}
                        : std::array<at::Tensor, 6>{});
                    grad_accumulators.push_back(
                        std::array<at::Tensor, 2>{
                            at::Tensor(), at::Tensor()});
                    pairs.emplace_back(std::move(a), std::move(b));
                }
                adapter.params[i] = std::move(pairs);
                adapter.adam_state[i] = std::move(adam_states);
                adapter.adam_shadow[i] = std::move(adam_shadows);
                adapter.grad_accum[i] = std::move(grad_accumulators);
            }
            bind_adapter_lora_gradient_slab(ctx, adapter);
            ctx->adapters.reserve(ctx->adapters.size() + 1);
        } catch (const std::exception& e) {
            preparation_error = e.what();
        }
        const bool local_prepared = preparation_error.empty();
        const bool preparation_matches = adapter_registration_phase_matches(
            ctx, adapter, local_prepared, /*phase=*/1);
        TORCH_CHECK(preparation_matches && local_prepared,
            "dynamic LoRA registration preparation failed or differs across "
            "distributed ranks", preparation_error.empty() ? "" : ": ",
            preparation_error);
        std::string synchronization_error;
        try {
            if (!ctx->restore_without_parameter_sync)
                synchronize_adapter_replicated_lora_parameters(ctx, adapter);
            TORCH_CHECK(!std::getenv(
                    "QWEN36_TEST_FAIL_ADAPTER_REGISTRATION_AFTER_SYNC"),
                "injected dynamic LoRA registration failure after "
                "parameter synchronization");
        } catch (const std::exception& e) {
            synchronization_error = e.what();
        }
        const bool local_synchronized = synchronization_error.empty();
        const bool synchronization_matches = adapter_registration_phase_matches(
            ctx, adapter, local_synchronized, /*phase=*/2);
        TORCH_CHECK(synchronization_matches && local_synchronized,
            "dynamic LoRA parameter synchronization failed on at least one "
            "distributed rank",
            synchronization_error.empty() ? "" : ": ",
            synchronization_error);
        int64_t id = adapter.id;
        ctx->adapters.push_back(std::move(adapter));
        ctx->next_adapter_id = id;
        ctx->lora_cache_valid = false;
        ctx->lora_batch_valid = false;
        fprintf(stderr, "[q36_lora] added adapter %ld: rank=%ld alpha=%.1f\n", (long)id, (long)rank, alpha);
        return id;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] add_lora FAILED: %s\n", e.what());
        return -1;
    }
}

struct ScopedBoolOverride {
    bool& target;
    bool previous;

    ScopedBoolOverride(bool& target_value, bool value)
        : target(target_value), previous(target_value) {
        target = value;
    }

    ~ScopedBoolOverride() { target = previous; }
};

__attribute__((visibility("default")))
int64_t qwen36_add_lora_v2(
    void* ctx_ptr,
    int64_t rank, double alpha,
    const int64_t* target_layers, int64_t num_target_layers,
    const char* target_modules_str
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    if (!ctx) return -1;
    ScopedBoolOverride guard(ctx->allow_heterogeneous_registration, true);
    return qwen36_add_lora(
        ctx_ptr, rank, alpha, target_layers, num_target_layers,
        target_modules_str);
}

__attribute__((visibility("default")))
int64_t qwen36_add_lora_for_restore(
    void* ctx_ptr,
    int64_t rank, double alpha,
    const int64_t* target_layers, int64_t num_target_layers,
    const char* target_modules_str
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    if (!ctx) return -1;
    ScopedBoolOverride guard(ctx->restore_without_parameter_sync, true);
    return qwen36_add_lora(
        ctx_ptr, rank, alpha, target_layers, num_target_layers,
        target_modules_str);
}

// Restore a dynamic adapter's externally visible ID during checkpoint load.
// IDs are positive and unique; the monotonic allocator is advanced so future
// additions cannot collide with a restored tenant.
__attribute__((visibility("default")))
int32_t qwen36_set_adapter_id(void* ctx_ptr, int64_t current_id, int64_t requested_id) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        TORCH_CHECK(ctx && current_id > 0 && requested_id > 0,
            "dynamic adapter IDs must be positive");
        require_clear_accumulation_for_registry_mutation(ctx);
        for (const auto& adapter : ctx->adapters) {
            TORCH_CHECK(adapter.id != requested_id || adapter.id == current_id,
                "dynamic adapter ID already exists: ", requested_id);
        }
        for (auto& adapter : ctx->adapters) {
            if (adapter.id == current_id) {
                adapter.id = requested_id;
                ctx->next_adapter_id = std::max(ctx->next_adapter_id, requested_id);
                ctx->lora_cache_valid = false;
                ctx->lora_batch_valid = false;
                return 0;
            }
        }
        TORCH_CHECK(false, "dynamic adapter not found: ", current_id);
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] set_adapter_id FAILED: %s\n", e.what());
        return -1;
    }
}

__attribute__((visibility("default")))
int32_t qwen36_remove_lora(void* ctx_ptr, int64_t adapter_id) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        TORCH_CHECK(ctx && adapter_id > 0,
            "dynamic LoRA removal requires a context and positive adapter ID");
        require_clear_accumulation_for_registry_mutation(ctx);
        for (auto it = ctx->adapters.begin(); it != ctx->adapters.end(); ++it) {
            if (it->id == adapter_id) {
                ctx->adapters.erase(it);
                ctx->lora_cache_valid = false;
                ctx->lora_batch_valid = false;
                return 1;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] remove_lora FAILED: %s\n", e.what());
        return -1;
    }
}

__attribute__((visibility("default")))
int64_t qwen36_list_lora(void* ctx_ptr, int64_t* out_ids, int64_t max_count) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    int64_t count = (int64_t)ctx->adapters.size();
    if (count > max_count) count = max_count;
    for (int64_t i = 0; i < count; i++)
        out_ids[i] = ctx->adapters[i].id;
    return count;
}

__attribute__((visibility("default")))
void* qwen36_get_adapter_lora_tensor(
    void* ctx_ptr, int64_t adapter_id, int64_t layer_idx,
    const char* module_name, int32_t is_b
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    if (!ctx || !module_name || layer_idx < 0 || layer_idx >= ctx->num_layers)
        return nullptr;
    const int64_t pair_idx = lora_pair_index(
        ctx->layer_configs[layer_idx], module_name);
    if (pair_idx < 0) return nullptr;
    for (auto& adapter : ctx->adapters) {
        if (adapter.id != adapter_id) continue;
        auto it = adapter.params.find(layer_idx);
        if (it == adapter.params.end() ||
            pair_idx >= static_cast<int64_t>(it->second.size()))
            return nullptr;
        auto& pair = it->second[pair_idx];
        auto& tensor = is_b ? pair.second : pair.first;
        return tensor.requires_grad() ? &tensor : nullptr;
    }
    return nullptr;
}

__attribute__((visibility("default")))
int32_t qwen36_set_adapter_lora_tensor(
    void* ctx_ptr, int64_t adapter_id, int64_t layer_idx,
    const char* module_name, int32_t is_b, void* tensor_ptr
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        TORCH_CHECK(ctx && module_name && tensor_ptr,
            "invalid dynamic LoRA tensor setter arguments");
        auto* target = reinterpret_cast<at::Tensor*>(
            qwen36_get_adapter_lora_tensor(
                ctx, adapter_id, layer_idx, module_name, is_b));
        TORCH_CHECK(target, "dynamic LoRA target not found: adapter=", adapter_id,
            " layer=", layer_idx, " module=", module_name);
        auto& source = *reinterpret_cast<at::Tensor*>(tensor_ptr);
        TORCH_CHECK(source.sizes() == target->sizes(),
            "dynamic LoRA tensor shape mismatch: expected ", target->sizes(),
            " got ", source.sizes());
        at::NoGradGuard guard;
        target->copy_(source.to(target->device()).to(target->scalar_type()));
        ctx->lora_cache_valid = false;
        ctx->lora_batch_valid = false;
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] set_adapter_lora_tensor FAILED: %s\n", e.what());
        return -1;
    }
}

// Access one dynamic adapter's Adam state. The per-layer state is stored as
// {m_a, v_a, m_b, v_b}; `is_b` selects A/B and `is_v` selects m/v.
__attribute__((visibility("default")))
void* qwen36_get_adapter_optimizer_tensor(
    void* ctx_ptr, int64_t adapter_id, int64_t layer_idx,
    const char* module_name, int32_t is_b, int32_t is_v
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    if (!ctx || !module_name || layer_idx < 0 || layer_idx >= ctx->num_layers)
        return nullptr;
    const int64_t pair_idx = lora_pair_index(
        ctx->layer_configs[layer_idx], module_name);
    if (pair_idx < 0) return nullptr;
    for (auto& adapter : ctx->adapters) {
        if (adapter.id != adapter_id) continue;
        auto state_it = adapter.adam_state.find(layer_idx);
        if (state_it == adapter.adam_state.end() ||
            pair_idx >= static_cast<int64_t>(state_it->second.size()))
            return nullptr;
        // array order: m_a, v_a, m_b, v_b
        const int index = (is_b ? 2 : 0) + (is_v ? 1 : 0);
        return &state_it->second[pair_idx][index];
    }
    return nullptr;
}

__attribute__((visibility("default")))
int32_t qwen36_set_adapter_optimizer_tensor(
    void* ctx_ptr, int64_t adapter_id, int64_t layer_idx,
    const char* module_name, int32_t is_b, int32_t is_v, void* tensor_ptr
) {
    try {
        auto* target = reinterpret_cast<at::Tensor*>(
            qwen36_get_adapter_optimizer_tensor(
                ctx_ptr, adapter_id, layer_idx, module_name, is_b, is_v));
        TORCH_CHECK(target && tensor_ptr, "dynamic optimizer tensor not found");
        auto& source = *reinterpret_cast<at::Tensor*>(tensor_ptr);
        TORCH_CHECK(source.sizes() == target->sizes(),
            "dynamic optimizer tensor shape mismatch: expected ", target->sizes(),
            " got ", source.sizes());
        at::NoGradGuard guard;
        target->copy_(source.to(target->device()).to(target->scalar_type()));
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] set_adapter_optimizer_tensor FAILED: %s\n", e.what());
        return -1;
    }
}

__attribute__((visibility("default")))
int64_t qwen36_get_adapter_step_count(void* ctx_ptr, int64_t adapter_id) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    if (!ctx || adapter_id <= 0) return -1;
    for (const auto& adapter : ctx->adapters) {
        if (adapter.id == adapter_id) return adapter.optimizer_step;
    }
    return -1;
}

__attribute__((visibility("default")))
int32_t qwen36_set_adapter_step_count(
    void* ctx_ptr, int64_t adapter_id, int64_t step_count
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    if (!ctx || adapter_id <= 0 || step_count < 0) return -1;
    for (auto& adapter : ctx->adapters) {
        if (adapter.id == adapter_id) {
            adapter.optimizer_step = step_count;
            return 0;
        }
    }
    return -1;
}

__attribute__((visibility("default")))
int64_t qwen36_get_lora_count(void* ctx_ptr) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    // This legacy accessor is paired with get_lora_a/get_lora_b and therefore
    // counts only the fixed single-adapter slots. Dynamic adapters have their
    // own registry and must not make this count exceed those arrays.
    return (int64_t)ctx->lora_a.size();
}

__attribute__((visibility("default")))
double qwen36_eval_step(void* ctx_ptr, void* input_ids_ptr, void* target_mask_ptr, void* attention_mask_ptr) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        struct EvalCacheGuard {
            TrainingContext* ctx;
            ~EvalCacheGuard() {
                // Evaluation builds projection caches under no-grad. They
                // must never be reused by the following training step.
                if (!ctx) return;
                ctx->lora_cache_valid = false;
                ctx->lora_batch_valid = false;
            }
        } cache_guard{ctx};
        TORCH_CHECK(ctx, "null training context");
        TORCH_CHECK(ctx->adapters.empty(),
            "dynamic LoRA adapters require selected multi-LoRA evaluation; "
            "ordinary eval_step has no tenant mapping");
        auto& input_ids = *reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto& target_mask = *reinterpret_cast<at::Tensor*>(target_mask_ptr);
        if (attention_mask_ptr) {
            auto& attention_mask = *reinterpret_cast<at::Tensor*>(attention_mask_ptr);
            validate_linear_attention_mask(ctx, attention_mask);
            ctx->attention_mask = attention_mask;
        }
        elide_trivial_attention_mask(ctx);
        at::AutoGradMode no_grad(false);
        auto hidden = ctx->use_checkpoint ? forward_full_checkpoint(ctx, input_ids) : forward_full(ctx, input_ids);
        auto loss = compute_loss_fused(ctx, hidden, input_ids, target_mask, ctx->vocab_size);
        return loss.item<double>();
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] eval_step FAILED: %s\n", e.what());
        return -1.0;
    }
}

static at::Tensor qwen36_copy_host_i64_batch(
    TrainingContext* ctx,
    const int64_t* data,
    int64_t batch_size,
    int64_t seq_len,
    const char* name
) {
    TORCH_CHECK(ctx, name, " requires a valid training context");
    TORCH_CHECK(data, name, " requires a non-null host pointer");
    TORCH_CHECK(batch_size > 0 && seq_len > 0,
        name, " requires positive batch_size and seq_len");
    TORCH_CHECK(batch_size <= std::numeric_limits<int64_t>::max() / seq_len,
        name, " batch shape overflows int64");
    c10::cuda::set_device(ctx->cuda_device);
    cudaSetDevice(ctx->cuda_device);
    auto host = at::from_blob(
        const_cast<int64_t*>(data), {batch_size, seq_len},
        at::TensorOptions().device(at::kCPU).dtype(at::kLong));
    // Shared-memory storage remains owned by the IPC channel. A blocking H2D
    // copy makes the returned CUDA tensor independent before the worker posts
    // completion and permits the coordinator to reuse the slab.
    return host.to(at::Device(at::kCUDA, ctx->cuda_device), at::kLong,
        /*non_blocking=*/false, /*copy=*/true);
}

__attribute__((visibility("default")))
double qwen36_train_step_host_i64(
    void* ctx_ptr,
    const int64_t* input_ids,
    const int64_t* target_mask,
    const int64_t* attention_mask,
    int64_t batch_size,
    int64_t seq_len
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        auto input = qwen36_copy_host_i64_batch(
            ctx, input_ids, batch_size, seq_len, "host train input_ids");
        auto targets = qwen36_copy_host_i64_batch(
            ctx, target_mask, batch_size, seq_len, "host train target_mask");
        auto attention = qwen36_copy_host_i64_batch(
            ctx, attention_mask, batch_size, seq_len, "host train attention_mask");
        return qwen36_train_step(ctx, &input, &targets, &attention);
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] train_step_host_i64 FAILED: %s\n", e.what());
        return -1.0;
    }
}

__attribute__((visibility("default")))
double qwen36_train_multi_lora_host_i64(
    void* ctx_ptr,
    const int64_t* input_ids,
    const int64_t* target_mask,
    const int64_t* attention_mask,
    int64_t batch_size,
    int64_t seq_len,
    int32_t n_total,
    int32_t lora_rank,
    const int64_t* adapter_ids,
    int32_t n_adapter_ids
) {
    try {
        (void)lora_rank;  // Retained for the ABI22 host wire contract.
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        TORCH_CHECK(n_adapter_ids >= 0,
            "host multi-LoRA adapter count must be non-negative");
        TORCH_CHECK(n_adapter_ids == 0 || n_adapter_ids == n_total,
            "host multi-LoRA selected adapter count must match n_total");
        auto input = qwen36_copy_host_i64_batch(
            ctx, input_ids, batch_size, seq_len, "host multi-LoRA input_ids");
        auto targets = qwen36_copy_host_i64_batch(
            ctx, target_mask, batch_size, seq_len, "host multi-LoRA target_mask");
        auto attention = qwen36_copy_host_i64_batch(
            ctx, attention_mask, batch_size, seq_len,
            "host multi-LoRA attention_mask");
        if (n_adapter_ids == 0) {
            TORCH_CHECK(ctx && static_cast<int32_t>(ctx->adapters.size()) == n_total,
                "host multi-LoRA adapter count must match the live registry");
            std::vector<int64_t> all_adapter_ids;
            all_adapter_ids.reserve(ctx->adapters.size());
            for (const auto& adapter : ctx->adapters)
                all_adapter_ids.push_back(adapter.id);
            return qwen36_train_multi_lora_selected_v2(
                ctx, &input, &targets, &attention,
                all_adapter_ids.data(), n_total);
        }
        TORCH_CHECK(adapter_ids,
            "host selected multi-LoRA requires adapter IDs");
        return qwen36_train_multi_lora_selected_v2(
            ctx, &input, &targets, &attention,
            adapter_ids, n_adapter_ids);
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] train_multi_lora_host_i64 FAILED: %s\n", e.what());
        return -1.0;
    }
}

__attribute__((visibility("default")))
double qwen36_eval_step_host_i64(
    void* ctx_ptr,
    const int64_t* input_ids,
    const int64_t* target_mask,
    const int64_t* attention_mask,
    int64_t batch_size,
    int64_t seq_len
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        auto input = qwen36_copy_host_i64_batch(
            ctx, input_ids, batch_size, seq_len, "host eval input_ids");
        auto targets = qwen36_copy_host_i64_batch(
            ctx, target_mask, batch_size, seq_len, "host eval target_mask");
        auto attention = qwen36_copy_host_i64_batch(
            ctx, attention_mask, batch_size, seq_len, "host eval attention_mask");
        return qwen36_eval_step(ctx, &input, &targets, &attention);
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] eval_step_host_i64 FAILED: %s\n", e.what());
        return -1.0;
    }
}

__attribute__((visibility("default")))
int64_t qwen36_get_step_count(void* ctx_ptr) {
    return (int64_t)reinterpret_cast<TrainingContext*>(ctx_ptr)->fixed_optimizer_step;
}

__attribute__((visibility("default")))
int64_t qwen36_get_dynamic_finalizer_count(void* ctx_ptr) {
    if (!ctx_ptr) return -1;
    return reinterpret_cast<TrainingContext*>(ctx_ptr)->dynamic_finalizer_count;
}

__attribute__((visibility("default")))
int64_t qwen36_get_dynamic_adam_launch_count(void* ctx_ptr) {
    if (!ctx_ptr) return -1;
    return reinterpret_cast<TrainingContext*>(ctx_ptr)->dynamic_adam_launch_count;
}

__attribute__((visibility("default")))
int32_t qwen36_get_accumulation_active(void* ctx_ptr) {
    if (!ctx_ptr) return -1;
    return reinterpret_cast<TrainingContext*>(ctx_ptr)->accumulation_active
        ? 1 : 0;
}

__attribute__((visibility("default")))
double qwen36_get_accumulated_token_weight(void* ctx_ptr) {
    if (!ctx_ptr) return -1.0;
    return reinterpret_cast<TrainingContext*>(
        ctx_ptr)->accumulated_token_weight;
}

// Restore the Adam bias-correction clock independently from tensor state.
// Checkpoint loading imports m/v through a separate ABI, so omitting this
// value would resume the next update as step 1 even for a mature optimizer.
__attribute__((visibility("default")))
int32_t qwen36_set_step_count(void* ctx_ptr, int64_t step_count) {
    if (!ctx_ptr || step_count < 0) return -1;
    reinterpret_cast<TrainingContext*>(ctx_ptr)->fixed_optimizer_step = step_count;
    return 0;
}

__attribute__((visibility("default")))
int64_t qwen36_export_optimizer_state(void* ctx_ptr, void** m_ptrs, void** v_ptrs, int64_t max_count) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    int64_t count = (int64_t)ctx->adam_m.size();
    if (count > max_count) count = max_count;
    for (int64_t i = 0; i < count; i++) {
        m_ptrs[i] = &ctx->adam_m[i];
        v_ptrs[i] = &ctx->adam_v[i];
    }
    return count;
}

__attribute__((visibility("default")))
int64_t qwen36_import_optimizer_state(void* ctx_ptr, void** m_ptrs, void** v_ptrs, int64_t count) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        TORCH_CHECK(ctx, "null training context");
        TORCH_CHECK(count >= 0, "negative optimizer state count");
        TORCH_CHECK(count <= (int64_t)ctx->adam_m.size() &&
                count <= (int64_t)ctx->adam_v.size(),
            "optimizer state count exceeds native state: count=", count,
            " m=", ctx->adam_m.size(), " v=", ctx->adam_v.size());
        TORCH_CHECK(count == 0 || (m_ptrs && v_ptrs),
            "null optimizer state pointer array");
        at::NoGradGuard guard;
        for (int64_t i = 0; i < count; i++) {
            auto* src_m = reinterpret_cast<at::Tensor*>(m_ptrs[i]);
            auto* src_v = reinterpret_cast<at::Tensor*>(v_ptrs[i]);
            TORCH_CHECK(src_m && src_v, "null optimizer tensor at index ", i);
            auto& target_m = ctx->adam_m[i];
            auto& target_v = ctx->adam_v[i];
            TORCH_CHECK(src_m->sizes() == target_m.sizes(),
                "Adam m shape mismatch at index ", i,
                ": expected ", target_m.sizes(), " got ", src_m->sizes());
            TORCH_CHECK(src_v->sizes() == target_v.sizes(),
                "Adam v shape mismatch at index ", i,
                ": expected ", target_v.sizes(), " got ", src_v->sizes());
            target_m.copy_(src_m->to(target_m.device()).to(target_m.scalar_type()));
            target_v.copy_(src_v->to(target_v.device()).to(target_v.scalar_type()));
        }
        return count;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] import_optimizer_state FAILED: %s\n", e.what());
        return -1;
    }
}

}  // extern "C"
