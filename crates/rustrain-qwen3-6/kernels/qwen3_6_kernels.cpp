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

// ──────────────────────────────────────────────────────────────────────
// Forward declarations
// ──────────────────────────────────────────────────────────────────────

static at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double eps);

static bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
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
        const at::Tensor& input, ncclComm_t comm, cudaStream_t requested_stream
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
            dtype_for(contiguous.scalar_type()), ncclSum, comm, comm_stream);
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

            q = (q.to(at::kFloat) / q.to(at::kFloat).norm(2, -1, true).clamp_min(1e-6));
            k = (k.to(at::kFloat) / k.to(at::kFloat).norm(2, -1, true).clamp_min(1e-6));
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
    q = (q.to(at::kFloat) / q.to(at::kFloat).norm(2, -1, true).clamp_min(1e-6));
    k = (k.to(at::kFloat) / k.to(at::kFloat).norm(2, -1, true).clamp_min(1e-6));

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

static std::vector<int32_t> qwen36_a2a_counts(
    const at::Tensor& local_counts, ncclComm_t comm, cudaStream_t stream
) {
    const int world = static_cast<int>(local_counts.numel());
    auto all_counts = at::empty({world, world}, local_counts.options());
    auto err = ncclAllGather(
        local_counts.data_ptr(), all_counts.data_ptr(), world,
        ncclInt, comm, stream);
    TORCH_CHECK(err == ncclSuccess, "EP A2A count all-gather failed: ",
        ncclGetErrorString(err));
    auto host = all_counts.to(at::TensorOptions().device(at::kCPU));
    std::vector<int32_t> recv_counts(world);
    auto ptr = host.data_ptr<int32_t>();
    int rank = 0;
    err = ncclCommUserRank(comm, &rank);
    TORCH_CHECK(err == ncclSuccess, "ncclCommUserRank failed: ",
        ncclGetErrorString(err));
    for (int src = 0; src < world; ++src) {
        recv_counts[src] = ptr[src * world + rank];
        TORCH_CHECK(recv_counts[src] >= 0, "negative EP A2A receive count");
    }
    return recv_counts;
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
        auto stream = c10::cuda::getCurrentCUDAStream(input.device().index()).stream();
        const int64_t hidden = input.size(1);
        std::vector<at::Tensor> indices(world);
        std::vector<int32_t> send_counts(world, 0);
        for (int dst = 0; dst < world; ++dst) {
            auto mask = (expert_indices >= dst * expert_count) &
                (expert_indices < (dst + 1) * expert_count);
            indices[dst] = at::nonzero(mask).reshape({-1});
            send_counts[dst] = static_cast<int32_t>(indices[dst].numel());
        }
        auto count_opts = at::TensorOptions().device(input.device()).dtype(at::kInt);
        auto local_counts = at::empty({world}, count_opts);
        auto host_counts = at::empty({world}, at::TensorOptions().device(at::kCPU).dtype(at::kInt));
        std::memcpy(host_counts.data_ptr<int32_t>(), send_counts.data(), sizeof(int32_t) * world);
        local_counts.copy_(host_counts);
        auto recv_counts = qwen36_a2a_counts(local_counts, comm, stream);
        std::vector<int64_t> send_offsets(world + 1, 0), recv_offsets(world + 1, 0);
        for (int i = 0; i < world; ++i) {
            send_offsets[i + 1] = send_offsets[i] + send_counts[i];
            recv_offsets[i + 1] = recv_offsets[i] + recv_counts[i];
        }
        auto send_index = at::cat(indices, 0);
        auto send_hidden = input.index_select(0, send_index).contiguous();
        auto send_token = token_indices.index_select(0, send_index).contiguous();
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
        auto send_counts_tensor = at::empty({world}, count_opts);
        auto recv_counts_tensor = at::empty({world}, count_opts);
        std::memcpy(host_counts.data_ptr<int32_t>(), send_counts.data(), sizeof(int32_t) * world);
        send_counts_tensor.copy_(host_counts);
        std::memcpy(host_counts.data_ptr<int32_t>(), recv_counts.data(), sizeof(int32_t) * world);
        recv_counts_tensor.copy_(host_counts);
        ctx->save_for_backward({input, send_index, send_counts_tensor, recv_counts_tensor});
        ctx->saved_data["comm"] = comm_ptr;
        ctx->saved_data["expert_count"] = expert_count;
        return {recv_hidden, recv_token, recv_local, send_index, send_counts_tensor, recv_counts_tensor};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx, std::vector<at::Tensor> grad_output
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto send_index = saved[1];
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
        grad_input.index_add_(0, send_index, returned);
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
    auto low_rank = at::bmm(a, input.unsqueeze(-1)).squeeze(-1);
    auto delta = at::bmm(b, low_rank.unsqueeze(-1)).squeeze(-1);
    auto scaling_stack = entry->scaling;
    if (scaling_stack.size(0) == 1 && batch > 1) {
        scaling_stack = scaling_stack.expand({batch, 1, 1});
    }
    auto scaling = scaling_stack.index_select(0, sample_indices)
        .reshape({-1, 1}).to(input.scalar_type());
    return tp_allreduce_lora_delta(ctx, delta * scaling);
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
    auto routed_output = at::zeros_like(flat);
    auto token_indices = at::arange(
        flat.size(0), at::TensorOptions().device(flat.device()).dtype(at::kLong));
    for (int64_t kk = 0; kk < top_k; ++kk) {
        auto expert_indices = topk_indices.select(-1, kk).contiguous();
        // `received_tokens` preserves the source flattened row index through
        // dispatch. Dynamic multi-LoRA uses floor_divide(row, seq) to recover
        // the tenant/sample row, so sharded A2A does not need a second host
        // metadata exchange for adapter IDs.
        auto dispatched = Qwen36A2ADispatchFunction::apply(
            flat, expert_indices, token_indices, expert_count,
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(comm)));
        auto received = dispatched[0];
        auto received_tokens = dispatched[1];
        auto received_experts = dispatched[2];
        auto send_index = dispatched[3];
        auto send_counts = dispatched[4];
        auto recv_counts = dispatched[5];
        auto local_output = at::zeros_like(received);
        for (int64_t e_local = 0; e_local < expert_count; ++e_local) {
            auto rows = at::nonzero(received_experts == e_local).reshape({-1});
            if (rows.numel() == 0) continue;
            auto selected = received.index_select(0, rows);
            auto selected_tokens = received_tokens.index_select(0, rows);
            auto selected_experts = received_experts.index_select(0, rows);
            auto gu = at::matmul(selected, experts_gate_up.select(0, e_local).t());
            if (expert_lora.gate_up_a && expert_lora.gate_up_b) {
                auto a = expert_lora.gate_up_a->select(0, e_local);
                auto b = expert_lora.gate_up_b->select(0, e_local);
                auto delta = at::matmul(at::matmul(selected, a.t()), b.t()) *
                    expert_lora.scaling;
                gu = gu + tp_allreduce_lora_delta(training_ctx, delta);
            }
            if (expert_gate_up_lora) {
                gu = gu + dynamic_expert_lora_delta(
                    training_ctx, selected, selected_tokens, selected_experts,
                    batch, seq, expert_gate_up_lora);
            }
            auto activated = fused_swiglu_op(
                gu.narrow(-1, 0, intermediate),
                gu.narrow(-1, intermediate, intermediate), 0.0);
            auto expert_out = at::matmul(
                activated, experts_down.select(0, e_local).t());
            if (expert_lora.down_a && expert_lora.down_b) {
                auto a = expert_lora.down_a->select(0, e_local);
                auto b = expert_lora.down_b->select(0, e_local);
                auto delta = at::matmul(at::matmul(activated, a.t()), b.t()) *
                    expert_lora.scaling;
                expert_out = expert_out +
                    tp_allreduce_lora_delta(training_ctx, delta);
            }
            if (expert_down_lora) {
                expert_out = expert_out + dynamic_expert_lora_delta(
                    training_ctx, activated, selected_tokens, selected_experts,
                    batch, seq, expert_down_lora);
            }
            local_output = local_output.index_add(0, rows, expert_out);
        }

        // Preserve a zero-valued dependency on every local expert LoRA when a
        // rank receives no tokens. This keeps optimizer collective order equal
        // across ranks without changing the output.
        if (!local_output.requires_grad()) {
            // Even without routed-expert LoRA, an empty destination must keep
            // the dispatch activation edge alive. Its zero gradient is sent
            // back to the source in the inverse dispatch backward.
            at::Tensor anchor = received.sum().to(local_output.scalar_type());
            auto include = [&](const at::Tensor* tensor) {
                if (!tensor || !tensor->defined() || !tensor->requires_grad()) return;
                auto value = tensor->sum().to(local_output.scalar_type());
                anchor = anchor + value;
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

        // Expert results return in the source's packed dispatch order. Apply
        // routing weights on the source rank, then restore source token order;
        // this keeps router/hidden gradients on the correct source graph.
        auto returned = Qwen36A2ACombineFunction::apply(
            local_output, send_counts, recv_counts,
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(comm)));
        auto source_weights = topk_weights.select(-1, kk)
            .index_select(0, send_index).unsqueeze(-1);
        routed_output = routed_output.index_add(
            0, send_index, returned * source_weights);
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
                training_ctx, comm, flat, topk_weights, topk_indices,
                experts_gate_up, experts_down, expert_lora,
                top_k, intermediate, expert_count, batch, seq,
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
        auto gathered = flat.index_select(0, sort_order);
        auto gathered_weights = expert_weights.index_select(0, sort_order).unsqueeze(-1);

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
            hidden_dim % 8 == 0 && intermediate % 8 == 0;
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
                        static_cast<long>(intermediate));
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
                    auto low_rank = at::_grouped_mm(
                        selected, expert_lora.gate_up_a->transpose(1, 2), offsets);
                    auto delta = at::_grouped_mm(
                        low_rank, expert_lora.gate_up_b->transpose(1, 2), offsets);
                    gu = gu + tp_allreduce_lora_delta(
                        training_ctx, delta * expert_lora.scaling);
                }
                if (expert_gate_up_lora) {
                    gu = gu + dynamic_expert_lora_delta(
                        training_ctx, selected, token_indices, local_expert_indices,
                        batch, seq, expert_gate_up_lora);
                }
                auto activated = fused_swiglu_op(
                    gu.narrow(-1, 0, intermediate),
                    gu.narrow(-1, intermediate, intermediate), 0.0);
                auto expert_out = at::_grouped_mm(
                    activated, experts_down.transpose(1, 2), offsets);
                if (expert_lora.down_a && expert_lora.down_b) {
                    auto low_rank = at::_grouped_mm(
                        activated, expert_lora.down_a->transpose(1, 2), offsets);
                    auto delta = at::_grouped_mm(
                        low_rank, expert_lora.down_b->transpose(1, 2), offsets);
                    expert_out = expert_out + tp_allreduce_lora_delta(
                        training_ctx, delta * expert_lora.scaling);
                }
                if (expert_down_lora) {
                    expert_out = expert_out + dynamic_expert_lora_delta(
                        training_ctx, activated, token_indices, local_expert_indices,
                        batch, seq, expert_down_lora);
                }
                auto weights = gathered_weights.narrow(
                    0, local_start, local_tokens);
                routed_output = routed_output.index_add_(
                    0, token_indices, expert_out * weights);
            }
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
                    auto delta = at::matmul(at::matmul(selected, a.t()), b.t())
                        * expert_lora.scaling;
                    gu = gu + tp_allreduce_lora_delta(training_ctx, delta);
                }
                if (expert_gate_up_lora) {
                    gu = gu + dynamic_expert_lora_delta(
                        training_ctx, selected, token_indices, local_expert_indices,
                        batch, seq, expert_gate_up_lora);
                }
                auto gate_part = gu.narrow(-1, 0, intermediate);
                auto up_part = gu.narrow(-1, intermediate, intermediate);
                auto activated = fused_swiglu_op(gate_part, up_part, 0.0);
                auto expert_out = at::matmul(activated, ed.t());
                if (expert_lora.down_a && expert_lora.down_b) {
                    auto a = expert_lora.down_a->select(0, e_local);
                    auto b = expert_lora.down_b->select(0, e_local);
                    auto delta = at::matmul(at::matmul(activated, a.t()), b.t())
                        * expert_lora.scaling;
                    expert_out = expert_out
                        + tp_allreduce_lora_delta(training_ctx, delta);
                }
                if (expert_down_lora) {
                    expert_out = expert_out + dynamic_expert_lora_delta(
                        training_ctx, activated, token_indices, local_expert_indices,
                        batch, seq, expert_down_lora);
                }
                auto weights = gathered_weights.narrow(0, offset, n_tokens);
                routed_output = routed_output.index_add_(0, token_indices, expert_out * weights);
            }
            offset += n_tokens;
        }
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
    auto shared_gate = at::matmul(flat, shared_gate_proj.t());
    auto shared_up = at::matmul(flat, shared_up_proj.t());
    if (shared_gate_lora) {
        shared_gate = add_batched_lora(
            training_ctx, shared_gate.reshape({batch, seq, -1}), hidden,
            shared_gate_lora)
            .reshape({batch * seq, -1});
    }
    if (shared_up_lora) {
        shared_up = add_batched_lora(
            training_ctx, shared_up.reshape({batch, seq, -1}), hidden,
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
    at::Tensor sizes_buf;    // [max_n] kInt
    int capacity = 0;

    void ensure(int n, const at::Tensor& ref) {
        if (n <= capacity) return;
        auto dev = ref.device();
        params_buf = at::empty({n}, at::TensorOptions().dtype(at::kLong).device(dev));
        grads_buf  = at::empty({n}, at::TensorOptions().dtype(at::kLong).device(dev));
        m_buf      = at::empty({n}, at::TensorOptions().dtype(at::kLong).device(dev));
        v_buf      = at::empty({n}, at::TensorOptions().dtype(at::kLong).device(dev));
        sizes_buf  = at::empty({n}, at::TensorOptions().dtype(at::kInt).device(dev));
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
    int64_t num_layers;

    // Attention mask [batch, seq] — 1 for real tokens, 0 for padding
    at::Tensor attention_mask;

    // ── Multi-LoRA adapter registry ──
    struct LoRAAdapter {
        int64_t id;
        int64_t rank;
        // Each tenant owns an independent Adam bias-correction clock. The
        // session-wide step_count remains a transport/metric clock only.
        int64_t optimizer_step = 0;
        double alpha;
        std::set<int64_t> target_layers;
        std::set<std::string> target_modules;
        std::map<int64_t, std::vector<std::pair<at::Tensor, at::Tensor>>> params;
        std::map<int64_t, std::vector<std::array<at::Tensor, 4>>> adam_state;
        // Gradients are harvested after each backward into FP32 tensors. The
        // accumulator tensors are intentionally shared by value when chunk
        // registry guards copy adapters, so their contents survive restore.
        std::map<int64_t, std::vector<std::array<at::Tensor, 2>>> grad_accum;
    };

    std::vector<LoRAAdapter> adapters;
    int64_t next_adapter_id = 0;
    int64_t multi_lora_invocation = 0;

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
    std::vector<uint8_t> lora_active;
    std::vector<int64_t> lora_layer_offset;
    double lora_scaling;
    std::vector<std::string> lora_names;

    // Adam optimizer state
    std::vector<at::Tensor> adam_m;
    std::vector<at::Tensor> adam_v;
    double lr, beta1, beta2, eps;
    int64_t step_count;
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

    // NCCL for Expert Parallel all-reduce (nullptr if single-GPU)
    ncclComm_t nccl_comm = nullptr;
    cudaStream_t nccl_stream = nullptr;
    int ep_world_size = 1;
    int ep_rank = 0;
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
    // Set when a legacy NCCL setter supplies an incompatible mixed topology.
    // Training entry points reject the context before touching parameters.
    bool topology_invalid = false;
    int cuda_device = 0;
// ──────────────────────────────────────────────────────────────────────
};

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
    for (auto& [layer_idx, pairs] : adapter.params) {
        auto accum_it = adapter.grad_accum.find(layer_idx);
        for (size_t i = 0; i < pairs.size(); ++i) {
            auto& [a, b] = pairs[i];
            if (a.grad().defined()) a.mutable_grad() = at::Tensor();
            if (b.grad().defined()) b.mutable_grad() = at::Tensor();
            if (accum_it != adapter.grad_accum.end() &&
                i < accum_it->second.size()) {
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
    for (size_t i = 0; i < ctx->lora_a.size(); ++i) {
        if (ctx->lora_a[i].grad().defined())
            ctx->lora_a[i].mutable_grad() = at::Tensor();
        if (ctx->lora_b[i].grad().defined())
            ctx->lora_b[i].mutable_grad() = at::Tensor();
        if (i < ctx->grad_accum_a.size() && ctx->grad_accum_a[i].defined())
            ctx->grad_accum_a[i].zero_();
        if (i < ctx->grad_accum_b.size() && ctx->grad_accum_b[i].defined())
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
    if (!ctx || !ctx->base_tp_attention || ctx->tp_world_size <= 1 ||
        layer_idx < 0 || layer_idx >= ctx->num_layers)
        return LoraTpLayout::LatentRank;
    const auto& cfg = ctx->layer_configs[layer_idx];
    auto table = lora_projection_table(cfg);
    TORCH_CHECK(pair_idx >= 0 && pair_idx < table.count,
        "invalid LoRA pair for TP layout");
    const std::string name(table.entries[pair_idx].name);
    if (name == "q_proj" || name == "k_proj" || name == "v_proj" ||
        name == "in_proj_qkv" || name == "in_proj_z" ||
        name == "in_proj_a" || name == "in_proj_b")
        return LoraTpLayout::ColumnParallel;
    if (name == "o_proj" || name == "out_proj")
        return LoraTpLayout::RowParallel;
    return LoraTpLayout::LatentRank;
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

static void synchronize_adapter_replicated_lora_parameters(
    TrainingContext* ctx, TrainingContext::LoRAAdapter& adapter);

static void synchronize_fixed_replicated_lora_parameters(TrainingContext* ctx) {
    if (!ctx || !ctx->base_tp_attention || ctx->tp_world_size <= 1) return;
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
        }
    }
    for (auto& adapter : ctx->adapters)
        synchronize_adapter_replicated_lora_parameters(ctx, adapter);
}

static void synchronize_adapter_replicated_lora_parameters(
    TrainingContext* ctx, TrainingContext::LoRAAdapter& adapter
) {
    if (!ctx || !ctx->base_tp_attention || ctx->tp_world_size <= 1) return;
    if (!ctx->tp_comm) return;  // qwen36_init_nccl synchronizes deferred adapters.
    for (auto& [layer, pairs] : adapter.params) {
        for (int64_t pair = 0; pair < static_cast<int64_t>(pairs.size()); ++pair) {
            auto& [a, b] = pairs[pair];
            if (!a.requires_grad() && !b.requires_grad()) continue;
            const auto layout = lora_tp_layout(ctx, layer, pair);
            if (layout == LoraTpLayout::ColumnParallel)
                tp_broadcast_lora_parameter(ctx, a);
            else if (layout == LoraTpLayout::RowParallel)
                tp_broadcast_lora_parameter(ctx, b);
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
    bool allreduce
) {
    if (!accumulator.defined()) return;
    TORCH_CHECK(accumulator.scalar_type() == at::kFloat,
        "LoRA DP gradient accumulator must be FP32");
    auto contiguous = accumulator.contiguous();
    if (scale != 1.0) {
        contiguous = contiguous * scale;
    }
    if (!allreduce) {
        at::NoGradGuard guard;
        accumulator.copy_(contiguous);
        return;
    }
    TORCH_CHECK(ctx->nccl_comm, "LoRA gradient all-reduce has no communicator");
    auto reduced = at::empty_like(contiguous);
    int dev = contiguous.device().index();
    cudaSetDevice(dev);
    auto stream = c10::cuda::getCurrentCUDAStream(dev).stream();
    auto err = ncclAllReduce(
        contiguous.data_ptr(), reduced.data_ptr(), contiguous.numel(),
        nccl_dtype_for(contiguous), ncclSum, ctx->nccl_comm, stream);
    TORCH_CHECK(err == ncclSuccess, "NCCL LoRA gradient all-reduce failed: ",
                ncclGetErrorString(err));
    at::NoGradGuard guard;
    accumulator.copy_(reduced);
}

static void normalize_lora_accumulator_numerator(
    TrainingContext* ctx, at::Tensor& accumulator,
    const at::Tensor& global_weight, bool allreduce
) {
    if (!accumulator.defined()) return;
    TORCH_CHECK(accumulator.scalar_type() == at::kFloat,
        "LoRA DP gradient accumulator must be FP32");
    TORCH_CHECK(global_weight.numel() == 1,
        "per-adapter LoRA global token weight must be scalar");
    auto numerator = accumulator.contiguous();
    at::Tensor reduced;
    if (allreduce) {
        TORCH_CHECK(ctx->nccl_comm,
            "LoRA gradient all-reduce has no communicator");
        reduced = at::empty_like(numerator);
        int dev = numerator.device().index();
        cudaSetDevice(dev);
        auto stream = c10::cuda::getCurrentCUDAStream(dev).stream();
        auto err = ncclAllReduce(
            numerator.data_ptr(), reduced.data_ptr(), numerator.numel(),
            nccl_dtype_for(numerator), ncclSum, ctx->nccl_comm, stream);
        TORCH_CHECK(err == ncclSuccess,
            "NCCL LoRA numerator all-reduce failed: ",
            ncclGetErrorString(err));
    } else {
        reduced = numerator;
    }
    reduced = reduced / global_weight.clamp_min(1.0);
    at::NoGradGuard guard;
    accumulator.copy_(reduced);
}

static void reduce_lora_accumulator_weighted(
    TrainingContext* ctx, at::Tensor& accumulator,
    const at::Tensor& local_weight, const at::Tensor& global_weight,
    bool allreduce
) {
    if (!accumulator.defined()) return;
    TORCH_CHECK(accumulator.scalar_type() == at::kFloat,
        "LoRA DP gradient accumulator must be FP32");
    TORCH_CHECK(local_weight.numel() == 1 && global_weight.numel() == 1,
        "per-adapter LoRA token weights must be scalar");
    auto weighted = accumulator.contiguous() * local_weight;
    at::Tensor reduced;
    if (allreduce) {
        TORCH_CHECK(ctx->nccl_comm,
            "LoRA gradient all-reduce has no communicator");
        reduced = at::empty_like(weighted);
        int dev = weighted.device().index();
        cudaSetDevice(dev);
        auto stream = c10::cuda::getCurrentCUDAStream(dev).stream();
        auto err = ncclAllReduce(
            weighted.data_ptr(), reduced.data_ptr(), weighted.numel(),
            nccl_dtype_for(weighted), ncclSum, ctx->nccl_comm, stream);
        TORCH_CHECK(err == ncclSuccess,
            "NCCL weighted LoRA gradient all-reduce failed: ",
            ncclGetErrorString(err));
    } else {
        reduced = weighted;
    }
    reduced = reduced / global_weight.clamp_min(1.0);
    at::NoGradGuard guard;
    accumulator.copy_(reduced);
}

// Fixed-LoRA gradients are accumulated as token-weighted numerators. Replicated
// DP sums all replicated parameters and divides by the global token count;
// legacy EP keeps its replicated batch/local expert semantics. Sharded A2A
// sums non-expert parameters across source ranks, while expert parameters have
// already received all source numerators through the inverse A2A and therefore
// only divide by the global token count.
static void synchronize_lora_gradients(
    TrainingContext* ctx, const at::Tensor& target_mask,
    double accumulated_token_weight = 0.0,
    const at::Tensor* per_adapter_token_counts = nullptr,
    std::vector<uint8_t>* adapter_has_global_tokens = nullptr
) {
    const bool sharded_a2a = ctx->nccl_comm && !ctx->data_parallel &&
        env_enabled("QWEN36_EP_A2A_SHARDED");
    TORCH_CHECK(!sharded_a2a || env_enabled("QWEN36_EP_A2A"),
        "QWEN36_EP_A2A_SHARDED=1 requires QWEN36_EP_A2A=1");
    const bool dp_allreduce = ctx->nccl_comm && ctx->data_parallel;
    const bool normalization_allreduce = dp_allreduce || sharded_a2a;
    const bool per_adapter_weighting = per_adapter_token_counts &&
        per_adapter_token_counts->defined();
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
        TORCH_CHECK(at::isfinite(local_adapter_weights).all().item<bool>(),
            "dynamic LoRA token counts must be finite");
        TORCH_CHECK((local_adapter_weights >= 0).all().item<bool>(),
            "dynamic LoRA token counts must be non-negative");
        if (dp_allreduce || sharded_a2a) {
            global_adapter_weights = at::empty_like(local_adapter_weights);
            auto stream = c10::cuda::getCurrentCUDAStream(
                local_adapter_weights.device().index()).stream();
            auto err = ncclAllReduce(
                local_adapter_weights.data_ptr(),
                global_adapter_weights.data_ptr(),
                local_adapter_weights.numel(), ncclFloat, ncclSum,
                ctx->nccl_comm, stream);
            TORCH_CHECK(err == ncclSuccess,
                "NCCL per-adapter token-count all-reduce failed: ",
                ncclGetErrorString(err));
        } else {
            global_adapter_weights = local_adapter_weights;
        }
        if (adapter_has_global_tokens) {
            adapter_has_global_tokens->assign(ctx->adapters.size(), 0);
            auto global_cpu = global_adapter_weights.to(
                at::TensorOptions().device(at::kCPU));
            const auto* counts = global_cpu.data_ptr<float>();
            for (size_t i = 0; i < ctx->adapters.size(); ++i) {
                (*adapter_has_global_tokens)[i] = counts[i] > 0.0f ? 1 : 0;
            }
        }
    } else if (accumulated_token_weight > 0.0) {
        double global_weight = accumulated_token_weight;
        if (normalization_allreduce) {
            auto local = at::full({1}, accumulated_token_weight,
                at::TensorOptions().dtype(at::kFloat).device(target_mask.device()));
            auto global = at::empty_like(local);
            auto stream = c10::cuda::getCurrentCUDAStream(
                target_mask.device().index()).stream();
            auto err = ncclAllReduce(
                local.data_ptr(), global.data_ptr(), 1,
                ncclFloat, ncclSum, ctx->nccl_comm, stream);
            TORCH_CHECK(err == ncclSuccess,
                "NCCL accumulated token-count all-reduce failed: ",
                ncclGetErrorString(err));
            global_weight = global.item<double>();
        }
        scale = 1.0 / std::max(global_weight, 1.0);
    } else {
        // Dynamic multi-LoRA currently contributes one independently-normalized
        // row per tenant. Preserve that contract while weighting replicated DP
        // ranks by the selected batch's token count.
        if (!dp_allreduce) {
            if (!ctx->base_tp_attention) return;
        } else {
            auto shifted_mask = target_mask.narrow(1, 1, target_mask.size(1) - 1)
                .to(at::kFloat).sum().reshape({1});
            auto global_mask = at::empty_like(shifted_mask);
            auto stream = c10::cuda::getCurrentCUDAStream(
                shifted_mask.device().index()).stream();
            auto err = ncclAllReduce(
                shifted_mask.data_ptr(), global_mask.data_ptr(), 1,
                ncclFloat, ncclSum, ctx->nccl_comm, stream);
            TORCH_CHECK(err == ncclSuccess, "NCCL token-count all-reduce failed: ",
                        ncclGetErrorString(err));
            const double local_tokens = shifted_mask.item<double>();
            const double global_tokens = global_mask.item<double>();
            scale = local_tokens / std::max(global_tokens, 1.0);
        }
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
                    if (sharded_a2a) {
                        // Sharded A2A restores each source row to a numerator
                        // before backward. Expert owners already receive all
                        // source numerators; replicated projections still need
                        // one all-reduce before division by the global count.
                        normalize_lora_accumulator_numerator(
                            ctx, accum_it->second[pair][0], global_weight,
                            !grouped_expert);
                        normalize_lora_accumulator_numerator(
                            ctx, accum_it->second[pair][1], global_weight,
                            !grouped_expert);
                    } else {
                        reduce_lora_accumulator_weighted(
                            ctx, accum_it->second[pair][0], local_weight,
                            global_weight, dp_allreduce);
                        reduce_lora_accumulator_weighted(
                            ctx, accum_it->second[pair][1], local_weight,
                            global_weight, dp_allreduce);
                    }
                    continue;
                }
                // Routed-expert tensors are sharded only in EP. Pure DP has
                // replicated experts and therefore uses the same reduction as
                // shared projections; the legacy accumulation path preserves
                // local normalization when no DP communicator is present.
                if (table.entries[pair].grouped_expert) {
                    if (accumulated_token_weight > 0.0) {
                        reduce_lora_accumulator(
                            ctx, accum_it->second[pair][0], scale, dp_allreduce);
                        reduce_lora_accumulator(
                            ctx, accum_it->second[pair][1], scale, dp_allreduce);
                    }
                    continue;
                }
                reduce_lora_accumulator(
                    ctx, accum_it->second[pair][0], scale, dp_allreduce);
                reduce_lora_accumulator(
                    ctx, accum_it->second[pair][1], scale, dp_allreduce);
            }
        }
    }
    // Projection-aware TP keeps one LoRA factor replicated. Its gradient is
    // the sum of disjoint output-head (column) or input-column (row)
    // contributions and is synchronized once at the optimizer boundary.
    for (auto& adapter : ctx->adapters) {
        for (auto& [layer_idx, pairs] : adapter.grad_accum) {
            for (int64_t pair = 0; pair < static_cast<int64_t>(pairs.size()); ++pair) {
                const auto layout = lora_tp_layout(ctx, layer_idx, pair);
                tp_sum_replicated_lora_accumulator(
                    ctx, pairs[pair][0], layout, true);
                tp_sum_replicated_lora_accumulator(
                    ctx, pairs[pair][1], layout, false);
            }
        }
    }
    if (per_adapter_weighting) return;
    // The replicated-source A2A path sends an identical token batch from every
    // EP rank. Average only its sharded expert parameter gradients here;
    // scaling the combine backward would also under-scale the activation
    // gradient returned to each source graph. Sharded A2A is handled above via
    // global token-count weighting and does not enter this branch.
    const double replicated_a2a_expert_scale =
        ctx->nccl_comm && env_enabled("QWEN36_EP_A2A") && !sharded_a2a &&
            !ctx->data_parallel && ctx->ep_world_size > 1
        ? 1.0 / static_cast<double>(ctx->ep_world_size)
        : 1.0;
    for (int64_t layer = 0; layer < ctx->num_layers; ++layer) {
        auto table = lora_projection_table(ctx->layer_configs[layer]);
        int64_t offset = ctx->lora_layer_offset[layer];
        for (int64_t pair = 0; pair < table.count; ++pair) {
            // Routed expert LoRA is local-only for EP, while pure DP owns the
            // complete replicated expert tensor and must all-reduce it.
            if (table.entries[pair].grouped_expert) {
                if (accumulated_token_weight > 0.0) {
                    reduce_lora_accumulator(
                        ctx, ctx->grad_accum_a[offset + pair],
                        scale * replicated_a2a_expert_scale, dp_allreduce);
                    reduce_lora_accumulator(
                        ctx, ctx->grad_accum_b[offset + pair],
                        scale * replicated_a2a_expert_scale, dp_allreduce);
                }
                continue;
            }
            reduce_lora_accumulator(
                ctx, ctx->grad_accum_a[offset + pair], scale,
                dp_allreduce || sharded_a2a);
            reduce_lora_accumulator(
                ctx, ctx->grad_accum_b[offset + pair], scale,
                dp_allreduce || sharded_a2a);
        }
    }
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

static void elide_trivial_attention_mask(TrainingContext* ctx) {
    if (!ctx->attention_mask.defined() || ctx->attention_mask.numel() == 0) return;
    // A padding-free batch can use SDPA's native causal fast path. This is one
    // scalar synchronization per step, instead of materializing [B,S,S] in
    // every full-attention layer.
    if (at::all(ctx->attention_mask != 0).item<bool>()) {
        ctx->attention_mask = at::Tensor();
    }
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

    auto a = at::matmul(projection_input, in_proj_a.t());
    auto b = at::matmul(projection_input, in_proj_b.t());
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
    q = (q.to(at::kFloat) / q.to(at::kFloat).norm(2, -1, true).clamp_min(1e-6));
    k = (k.to(at::kFloat) / k.to(at::kFloat).norm(2, -1, true).clamp_min(1e-6));

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
    auto embed = *ctx->embed_ptr[0];
    auto final_norm = *ctx->final_norm_ptr[0];

    at::AutoGradMode guard(true);
    at::Tensor hidden = at::embedding(embed, input_ids);

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
    auto embed = *ctx->embed_ptr[0];
    at::Tensor hidden = at::embedding(embed, input_ids);
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
    auto embed = *ctx->embed_ptr[0];
    at::Tensor hidden = at::embedding(embed, input_ids);

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
static at::Tensor compute_loss_fused(
    TrainingContext* ctx,
    const at::Tensor& hidden,       // [batch, seq, hidden] (requires_grad)
    const at::Tensor& input_ids,    // [batch, seq]
    const at::Tensor& target_mask,  // [batch, seq]
    int64_t vocab_size
) {
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
    // hidden_normed is [batch, seq, hidden] — pad first token with zeros.
    auto grad_reshaped = grad_hidden.to(hidden_normed.scalar_type())
        .reshape({hidden_normed.size(0), seq_len - 1, hidden_dim});
    auto grad_full = at::cat({
        at::zeros({hidden_normed.size(0), 1, hidden_dim},
            at::TensorOptions().dtype(hidden_normed.scalar_type()).device(hidden_normed.device())),
        grad_reshaped
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

struct LossResult {
    at::Tensor value;
    at::Tensor hidden_grad;
};

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
    auto embed = *ctx->embed_ptr[0];

    int64_t seq_len = hidden.size(1);

    // hidden[t] + embed[t+1] → predict token t+2 (Megatron convention)
    auto hidden_shifted = hidden.narrow(1, 0, seq_len - 1);  // [batch, seq-1, hidden]
    auto embed_next = at::embedding(embed, input_ids.narrow(1, 1, seq_len - 1));  // [batch, seq-1, hidden]

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
    return 15;
}

static constexpr int32_t QWEN36_CONTEXT_BASE_TP_ATTENTION = 1 << 0;

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
        ctx->step_count = 0; ctx->lora_scaling = lora_scaling;
        ctx->num_layers = num_layers;
        ctx->base_tp_attention =
            (context_flags & QWEN36_CONTEXT_BASE_TP_ATTENTION) != 0;
        ctx->has_mtp = false;
        ctx->use_checkpoint = false; ctx->group_size = 4;
        const char* tp_size_env = getenv("TP_SIZE");
        if (!tp_size_env) tp_size_env = getenv("RUSTRAIN_TP_SIZE");
        ctx->tp_world_size = tp_size_env ? atoi(tp_size_env) : 1;
        TORCH_CHECK(ctx->tp_world_size > 0, "TP_SIZE must be positive");
        const char* world_size_env = getenv("WORLD_SIZE");
        const int configured_world_size = world_size_env ? atoi(world_size_env) : 1;
        TORCH_CHECK(configured_world_size > 0, "WORLD_SIZE must be positive");
        const bool data_parallel_requested = env_enabled("RUSTRAIN_DATA_PARALLEL");
        const char* pp_size_env = getenv("PP_SIZE");
        if (!pp_size_env) pp_size_env = getenv("RUSTRAIN_PP_SIZE");
        const char* cp_size_env = getenv("CP_SIZE");
        if (!cp_size_env) cp_size_env = getenv("RUSTRAIN_CP_SIZE");
        const int configured_pp_size = pp_size_env ? atoi(pp_size_env) : 1;
        const int configured_cp_size = cp_size_env ? atoi(cp_size_env) : 1;
        TORCH_CHECK(configured_pp_size > 0 && configured_cp_size > 0,
            "PP_SIZE and CP_SIZE must be positive");
        TORCH_CHECK(configured_pp_size == 1 && configured_cp_size == 1,
            "native Qwen LoRA does not implement PP/CP yet; ",
            "PP_SIZE=", configured_pp_size, " CP_SIZE=", configured_cp_size);
        TORCH_CHECK(
            ctx->tp_world_size <= 1 ||
                (!data_parallel_requested && configured_world_size == ctx->tp_world_size),
            "native Qwen LoRA supports TP-only topology when TP_SIZE>1; "
            "TP_SIZE=", ctx->tp_world_size, " WORLD_SIZE=", configured_world_size,
            " DATA_PARALLEL=", data_parallel_requested ? 1 : 0,
            " is an incompatible mixed TP/DP/EP topology");
        ctx->ep_world_size = configured_world_size;
        ctx->data_parallel = data_parallel_requested;
        const char* rank_env = getenv("RANK");
        const int global_rank = rank_env ? atoi(rank_env) : 0;
        ctx->tp_rank = global_rank % ctx->tp_world_size;
        TORCH_CHECK(lora_rank > 0 && lora_rank % ctx->tp_world_size == 0,
            "LoRA rank ", lora_rank, " must be divisible by TP_SIZE=",
            ctx->tp_world_size);
        const int64_t local_lora_rank = lora_rank / ctx->tp_world_size;
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

        // Copy layer configs
        auto* lcfgs = reinterpret_cast<LayerConfig*>(layer_configs_ptr);
        for (int64_t i = 0; i < num_layers; i++) {
            ctx->layer_configs.push_back(lcfgs[i]);
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
                    a = initialize_lora_a(ctx, opts, experts, lora_rank, in_f);
                    b = at::zeros({experts, out_f, local_lora_rank}, opts);
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
                auto grad_opts = at::TensorOptions().dtype(at::kFloat).device(base->device());
                ctx->grad_accum_a.push_back(
                    active ? at::zeros(a.sizes(), grad_opts) : at::Tensor());
                ctx->grad_accum_b.push_back(
                    active ? at::zeros(b.sizes(), grad_opts) : at::Tensor());
                ctx->lora_a.push_back(std::move(a));
                ctx->lora_b.push_back(std::move(b));
                ctx->lora_active.push_back(active ? 1 : 0);
                auto prefix = "layers." + std::to_string(i) + "." + projection.name;
                ctx->lora_names.push_back(prefix + ".lora_A.weight");
                ctx->lora_names.push_back(prefix + ".lora_B.weight");
            }
            offset += lora_count;
        }

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
                "base dense MLP TP cannot be disabled after enablement because "
                "the context owns TP-sharded weights");
            return 0;
        }
        if (ctx->base_tp_mlp) return 0;
        TORCH_CHECK(ctx->tp_world_size > 1,
            "base dense MLP TP requires TP_SIZE>1");
        TORCH_CHECK(!ctx->has_mtp,
            "base dense MLP TP requires MTP to be disabled");
        for (const auto& adapter : ctx->adapters) {
            TORCH_CHECK(!adapter.target_modules.empty(),
                "base dense MLP TP does not support an existing dynamic LoRA "
                "adapter targeting all modules");
            for (const auto& name : adapter.target_modules) {
                TORCH_CHECK(!is_mlp_lora_target(name),
                    "base dense MLP TP does not support existing dynamic MLP LoRA target ",
                    name, "; use attention-only targets");
            }
        }

        int64_t weight_offset = 0;
        for (int64_t layer = 0; layer < ctx->num_layers; ++layer) {
            const auto& cfg = ctx->layer_configs[layer];
            TORCH_CHECK(cfg.num_experts == 0,
                "base dense MLP TP currently supports dense models only");
            TORCH_CHECK(cfg.intermediate_size > 0 &&
                    cfg.intermediate_size % ctx->tp_world_size == 0,
                "dense intermediate_size must be divisible by TP_SIZE");
            const int64_t local_intermediate =
                cfg.intermediate_size / ctx->tp_world_size;
            const int64_t mlp_start = cfg.layer_type == 0 ? 8 : 11;
            auto* gate = ctx->weight_ptrs[weight_offset + mlp_start];
            auto* up = ctx->weight_ptrs[weight_offset + mlp_start + 1];
            auto* down = ctx->weight_ptrs[weight_offset + mlp_start + 2];
            TORCH_CHECK(gate && up && down &&
                    gate->dim() == 2 && up->dim() == 2 && down->dim() == 2,
                "base dense MLP TP requires matrix gate/up/down weights");
            TORCH_CHECK(gate->size(0) == local_intermediate &&
                    up->size(0) == local_intermediate &&
                    down->size(1) == local_intermediate &&
                    gate->size(1) == up->size(1) &&
                    gate->size(1) == down->size(0),
                "base dense MLP TP received inconsistent local weight shapes: gate=",
                gate->sizes(), " up=", up->sizes(), " down=", down->sizes(),
                " expected local intermediate=", local_intermediate);

            const int64_t lora_offset = ctx->lora_layer_offset[layer];
            auto projections = lora_projection_table(cfg);
            for (int64_t pair = 0; pair < projections.count; ++pair) {
                if (projections.entries[pair].segment == LoraSegment::Mlp) {
                    TORCH_CHECK(!ctx->lora_active[lora_offset + pair],
                        "base dense MLP TP does not yet support MLP LoRA target ",
                        projections.entries[pair].name,
                        "; use explicit attention-only targets");
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
        TORCH_CHECK(!ctx->base_tp_mlp && !ctx->base_tp_attention,
            "MTP cannot be enabled after frozen base TP because MTP weights are not sharded");
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
        TORCH_CHECK(ctx->adapters.empty(),
            "dynamic LoRA adapters require qwen36_train_multi_lora or "
            "qwen36_train_multi_lora_selected");
        GradientAccumulationFailureGuard accumulation_guard{ctx};
        TORCH_CHECK(gradient_scale > 0.0 && std::isfinite(gradient_scale),
            "gradient_scale must be finite and positive");
        // Set CUDA device for EP
        if (ctx->nccl_comm) {
            c10::cuda::set_device(ctx->cuda_device);
            cudaSetDevice(ctx->cuda_device);
        }
        auto& input_ids = *reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto& target_mask = *reinterpret_cast<at::Tensor*>(target_mask_ptr);
        if (attention_mask_ptr) {
            ctx->attention_mask = *reinterpret_cast<at::Tensor*>(attention_mask_ptr);
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

        const double supervised_tokens = target_mask
            .narrow(1, 1, target_mask.size(1) - 1)
            .to(at::kFloat).sum().item<double>();
        const double micro_token_weight =
            gradient_scale * std::max(supervised_tokens, 1.0);
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
        harvest_gradient_accumulators(ctx);
        ctx->accumulated_token_weight += micro_token_weight;

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
        synchronize_lora_gradients(
            ctx, target_mask, ctx->accumulated_token_weight);

        // ── Adam optimizer step — CUDA multi-tensor fused kernel ──
        at::AutoGradMode guard(false);
        ctx->lora_cache_valid = false;
        ctx->lora_batch_valid = false;
        const int64_t next_step = ctx->step_count + 1;
        double step_f = (double)next_step;
        double bias_correction1 = 1.0 - std::pow(ctx->beta1, step_f);
        double bias_correction2 = 1.0 - std::pow(ctx->beta2, step_f);
        double sqrt_bias_correction2 = std::sqrt(bias_correction2);
        float lr_scaled = (float)(
            ctx->lr * sqrt_bias_correction2 / bias_correction1);
        float eps_scaled = (float)(ctx->eps * sqrt_bias_correction2);
        float one_minus_b1 = (float)(1.0 - ctx->beta1);
        float one_minus_b2 = (float)(1.0 - ctx->beta2);

        // Collect all (param, grad, m, v, size) tuples from multi-LoRA + legacy
        std::vector<void*> h_params, h_grads;
        std::vector<float*> h_m, h_v;
        std::vector<int> h_sizes;

        // Multi-LoRA adapters
        for (auto& adapter : ctx->adapters) {
            for (auto& [layer_idx, pairs] : adapter.params) {
                auto& adam_states = adapter.adam_state[layer_idx];
                auto& accumulators = adapter.grad_accum[layer_idx];
                for (size_t i = 0; i < pairs.size(); i++) {
                    auto& [a, b] = pairs[i];
                    auto& [m_a, v_a, m_b, v_b] = adam_states[i];
                    auto& [accum_a, accum_b] = accumulators[i];
                    if (a.requires_grad() && accum_a.defined() &&
                        a.scalar_type() == at::kBFloat16) {
                        h_params.push_back(a.data_ptr());
                        h_grads.push_back(accum_a.data_ptr());
                        h_m.push_back((float*)m_a.data_ptr());
                        h_v.push_back((float*)v_a.data_ptr());
                        h_sizes.push_back((int)a.numel());
                    }
                    if (b.requires_grad() && accum_b.defined() &&
                        b.scalar_type() == at::kBFloat16) {
                        h_params.push_back(b.data_ptr());
                        h_grads.push_back(accum_b.data_ptr());
                        h_m.push_back((float*)m_b.data_ptr());
                        h_v.push_back((float*)v_b.data_ptr());
                        h_sizes.push_back((int)b.numel());
                    }
                }
            }
        }
        // Legacy single-LoRA
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
            ctx->step_count = next_step;
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
    int64_t hidden, int64_t group_size, int64_t num_layers
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

    // CE peak: one chunk of [16384, vocab] logits at a time.
    // BF16 logits (8GB) + FP32 CE loss (16GB) + grad (~8GB) ≈ 16GB.
    // But backward releases immediately, so effective peak is lower.
    int64_t ce_peak = 16384LL * 248320LL * 4LL;  // ~16GB (FP32 logits only, others freed by autograd)

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

/// Train all adapters in chunks. Each chunk: independent forward → loss → backward → Adam.
/// Inputs may be [1, seq] (shared prompt, repeated per chunk) or
/// [n_total, seq] (one independent sample per adapter).
__attribute__((visibility("default"))) double qwen36_train_multi_lora(
    void* ctx_ptr,
    void* input_ids_ptr,
    void* target_mask_ptr,
    void* attention_mask_ptr,
    int32_t n_total,
    int32_t lora_rank
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        TORCH_CHECK(!ctx->topology_invalid,
            "native Qwen context rejected an incompatible TP/DP/EP topology");
        TORCH_CHECK(!ctx->accumulation_active &&
                ctx->accumulated_token_weight == 0.0,
            "cannot start dynamic multi-LoRA while a fixed-LoRA gradient "
            "accumulation window is pending");
        GradientAccumulationFailureGuard accumulation_guard{ctx};
        if (ctx->nccl_comm) {
            c10::cuda::set_device(ctx->cuda_device);
            cudaSetDevice(ctx->cuda_device);
        }

        auto& input_ids = *reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto& target_mask = *reinterpret_cast<at::Tensor*>(target_mask_ptr);

        int64_t total_adapters = (int64_t)ctx->adapters.size();
        if (total_adapters == 0) return -1.0;
        TORCH_CHECK(n_total > 0 && total_adapters == n_total,
            "n_total must equal the number of registered adapters (n_total=",
            n_total, ", registered=", total_adapters, ")");
        for (const auto& adapter : ctx->adapters) {
            TORCH_CHECK(adapter.rank == lora_rank,
                "lora_rank argument must match every registered adapter; adapter=",
                adapter.id, " registered_rank=", adapter.rank,
                " requested_rank=", lora_rank);
        }
        TORCH_CHECK(!ctx->has_mtp || env_enabled("QWEN36_DISABLE_MTP"),
            "dynamic multi-LoRA with MTP is not supported until main and MTP "
            "objectives have independent global token denominators");
        TORCH_CHECK(input_ids.dim() == 2 && target_mask.dim() == 2,
            "multi-LoRA inputs must have shape [batch, seq]");
        const int64_t input_batch = input_ids.size(0);
        TORCH_CHECK(input_batch == 1 || input_batch == n_total,
            "multi-LoRA input batch must be 1 or n_total (batch=", input_batch,
            ", n_total=", n_total, ")");
        TORCH_CHECK(target_mask.size(0) == input_batch &&
                    target_mask.size(1) == input_ids.size(1),
            "target_mask must match input_ids shape");
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
        if (attention_mask_ptr) {
            provided_attention_mask = *reinterpret_cast<at::Tensor*>(attention_mask_ptr);
            TORCH_CHECK(provided_attention_mask.dim() == 2,
                "multi-LoRA attention_mask must have shape [batch, seq]");
            TORCH_CHECK(provided_attention_mask.size(0) == input_batch &&
                        provided_attention_mask.size(1) == input_ids.size(1),
                "attention_mask must match input_ids shape");
        }
        const at::Tensor saved_attention_mask = ctx->attention_mask;
        struct AttentionMaskGuard {
            TrainingContext* ctx;
            at::Tensor saved;
            ~AttentionMaskGuard() { ctx->attention_mask = saved; }
        } attention_mask_guard{ctx, saved_attention_mask};

        struct AdapterRegistryChunkGuard {
            TrainingContext* ctx;
            std::vector<TrainingContext::LoRAAdapter> all;
            bool active = false;

            AdapterRegistryChunkGuard(
                TrainingContext* context, int64_t start, int64_t end)
                : ctx(context) {
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

        // Compute N_max from available GPU memory. All workers must agree on
        // the chunk schedule to keep later collectives in the same order, so
        // rank 0 publishes its value through the existing communicator.
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        int64_t n_max = 0;
        if (ctx->nccl_comm && ctx->ep_world_size > 1) {
            if (ctx->ep_rank == 0) {
                n_max = compute_n_max(
                    (int64_t)free_mem, lora_rank,
                    input_ids.size(-1), 2048,
                    ctx->group_size, ctx->num_layers
                );
                n_max = std::min(n_max, total_adapters);
                if (n_max < 1) n_max = 1;
            }
            auto published_n_max = at::full(
                {1}, n_max, input_ids.options().dtype(at::kLong));
            auto stream = c10::cuda::getCurrentCUDAStream(
                input_ids.device().index()).stream();
            auto err = ncclBroadcast(
                published_n_max.data_ptr<int64_t>(),
                published_n_max.data_ptr<int64_t>(), 1, ncclInt64, 0,
                reinterpret_cast<ncclComm_t>(ctx->nccl_comm), stream);
            TORCH_CHECK(err == ncclSuccess, "n_max broadcast failed: ",
                ncclGetErrorString(err));
            n_max = published_n_max.to(
                at::TensorOptions().device(at::kCPU)).item<int64_t>();
        } else {
            n_max = compute_n_max(
                (int64_t)free_mem, lora_rank,
                input_ids.size(-1), 2048,
                ctx->group_size, ctx->num_layers
            );
        }
        n_max = std::min(n_max, total_adapters);
        if (n_max < 1) n_max = 1;

        fprintf(stderr, "[train_multi] total=%ld n_max=%ld free=%.1fGB rank=%d\n",
                (long)total_adapters, (long)n_max, (double)free_mem / 1e9, lora_rank);

        double total_loss = 0.0;
        int64_t num_chunks = (total_adapters + n_max - 1) / n_max;
        // Chunking is a memory scheduling detail, not an optimizer step. Both
        // the session clock and tenant clocks commit only after Adam launches.
        const int64_t next_session_step = ctx->step_count + 1;
        bool any_update = false;

        for (int64_t chunk = 0; chunk < num_chunks; chunk++) {
            int64_t start = chunk * n_max;
            int64_t end = std::min(start + n_max, total_adapters);
            int64_t n = end - start;

            // Invalidate cache for this chunk's adapter set
            ctx->lora_batch_valid = false;
            ctx->lora_cache_valid = false;

            // Scope the registry to this activation-memory chunk. The guard
            // restores the full registry even if forward/backward fails.
            AdapterRegistryChunkGuard registry_guard(ctx, start, end);

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
            double loss_val;
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
            if (ctx->nccl_comm && !ctx->data_parallel &&
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

            // Restore the complete registry before the next chunk. Gradients
            // remain attached to the intrusive tensor handles, so all chunks
            // can accumulate and the optimizer runs exactly once below.
            registry_guard.restore();

            if (chunk == num_chunks - 1) {
                // DP gradient synchronization and Adam belong to the logical
                // multi-tenant step, never to an activation-memory chunk.
                std::vector<uint8_t> adapter_has_global_tokens;
                synchronize_lora_gradients(
                    ctx, target_mask, 0.0, &adapter_token_counts,
                    &adapter_has_global_tokens);
                TORCH_CHECK(adapter_has_global_tokens.size() ==
                        ctx->adapters.size(),
                    "dynamic LoRA global-token activity vector mismatch");

                // Adam step. Group tenants by their own logical clock so
                // newly-added or resumed tenants do not inherit another
                // tenant's bias correction. Adapters with the same clock
                // still share one fused multi-tensor launch.
                at::AutoGradMode guard(false);
                ctx->lora_cache_valid = false;
                ctx->lora_batch_valid = false;
                std::map<int64_t, std::vector<TrainingContext::LoRAAdapter*>> groups;
                for (size_t adapter_index = 0;
                     adapter_index < ctx->adapters.size(); ++adapter_index) {
                    if (!adapter_has_global_tokens[adapter_index]) continue;
                    auto& adapter = ctx->adapters[adapter_index];
                    groups[adapter.optimizer_step + 1].push_back(&adapter);
                }
                for (auto& [logical_step, adapters] : groups) {
                    std::vector<void*> h_params, h_grads;
                    std::vector<float*> h_m, h_v;
                    std::vector<int> h_sizes;
                    for (auto* adapter : adapters) {
                        for (auto& [layer_idx, pairs] : adapter->params) {
                            auto& adam_states = adapter->adam_state[layer_idx];
                            auto& accumulators = adapter->grad_accum[layer_idx];
                            for (size_t i = 0; i < pairs.size(); i++) {
                                auto& [a, b] = pairs[i];
                                auto& [m_a, v_a, m_b, v_b] = adam_states[i];
                                auto& [accum_a, accum_b] = accumulators[i];
                                if (a.requires_grad() && accum_a.defined() &&
                                    a.scalar_type() == at::kBFloat16) {
                                    h_params.push_back(a.data_ptr());
                                    h_grads.push_back(accum_a.data_ptr());
                                    h_m.push_back((float*)m_a.data_ptr());
                                    h_v.push_back((float*)v_a.data_ptr());
                                    h_sizes.push_back((int)a.numel());
                                }
                                if (b.requires_grad() && accum_b.defined() &&
                                    b.scalar_type() == at::kBFloat16) {
                                    h_params.push_back(b.data_ptr());
                                    h_grads.push_back(accum_b.data_ptr());
                                    h_m.push_back((float*)m_b.data_ptr());
                                    h_v.push_back((float*)v_b.data_ptr());
                                    h_sizes.push_back((int)b.numel());
                                }
                            }
                        }
                    }
                    if (h_params.empty()) continue;
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
                    const float one_minus_b1 = (float)(1.0 - ctx->beta1);
                    const float one_minus_b2 = (float)(1.0 - ctx->beta2);
                    int n_params = (int)h_params.size();
                    auto opts_cpu_long = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
                    auto opts_cpu_int  = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
                    auto params_cpu = at::from_blob(h_params.data(), {n_params}, opts_cpu_long);
                    auto grads_cpu  = at::from_blob(h_grads.data(),  {n_params}, opts_cpu_long);
                    auto m_cpu      = at::from_blob(h_m.data(),      {n_params}, opts_cpu_long);
                    auto v_cpu      = at::from_blob(h_v.data(),      {n_params}, opts_cpu_long);
                    auto sizes_cpu  = at::from_blob(h_sizes.data(),  {n_params}, opts_cpu_int);
                    ctx->adam_dev_bufs.ensure(n_params, adapters[0]->params.begin()->second[0].first);
                    ctx->adam_dev_bufs.params_buf.narrow(0, 0, n_params).copy_(params_cpu);
                    ctx->adam_dev_bufs.grads_buf.narrow(0, 0, n_params).copy_(grads_cpu);
                    ctx->adam_dev_bufs.m_buf.narrow(0, 0, n_params).copy_(m_cpu);
                    ctx->adam_dev_bufs.v_buf.narrow(0, 0, n_params).copy_(v_cpu);
                    ctx->adam_dev_bufs.sizes_buf.narrow(0, 0, n_params).copy_(sizes_cpu);
                    auto stream = c10::cuda::getCurrentCUDAStream().stream();
                    launch_fused_adam_multi(
                        (void**)ctx->adam_dev_bufs.params_buf.data_ptr(),
                        (void**)ctx->adam_dev_bufs.grads_buf.data_ptr(),
                        (float**)ctx->adam_dev_bufs.m_buf.data_ptr(),
                        (float**)ctx->adam_dev_bufs.v_buf.data_ptr(),
                        (int*)ctx->adam_dev_bufs.sizes_buf.data_ptr(),
                        n_params, (float)ctx->beta1, (float)ctx->beta2,
                        lr_scaled, eps_scaled, one_minus_b1, one_minus_b2,
                        (void*)stream);
                    auto launch_error = cudaGetLastError();
                    TORCH_CHECK(launch_error == cudaSuccess,
                        "dynamic fused FP32-gradient Adam launch failed: ",
                        cudaGetErrorString(launch_error));
                    for (auto* adapter : adapters) {
                        adapter->optimizer_step = logical_step;
                    }
                    any_update = true;
                }
            }

            total_loss += loss_val;

            fprintf(stderr, "[train_multi] chunk %ld/%ld: n=%ld loss=%.6f\n",
                    (long)(chunk + 1), (long)num_chunks, (long)n, loss_val);
        }

        if (any_update) ctx->step_count = next_session_step;
        clear_gradient_accumulators(ctx);
        accumulation_guard.disarmed = true;
        return total_loss / total_adapters;
    } catch (const std::exception& e) {
        fprintf(stderr, "[train_multi] FAILED: %s\n", e.what());
        return -1.0;
    } catch (...) {
        fprintf(stderr, "[train_multi] FAILED: unknown exception\n");
        return -1.0;
    }
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
        TORCH_CHECK(ctx && adapter_ids && n_adapters > 0,
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
static bool g_nccl_initialized = false;
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

__attribute__((visibility("default"))) int32_t qwen36_init_nccl(
    void* ctx_ptr
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);

    // If already initialized, just set the pointer on this context
    if (g_nccl_initialized) {
        ctx->nccl_comm = g_nccl_comm;
        ctx->nccl_stream = g_nccl_stream;
        ctx->data_parallel = env_enabled("RUSTRAIN_DATA_PARALLEL");
        // CRITICAL: also set ep_rank/ep_world_size — needed for new TrainingContext
        // created by subsequent CreateSession commands. The fast path previously
        // skipped this, leaving ep_rank=0 → cudaSetDevice(0) on all ranks → crash.
        const char* rank_str2 = getenv("RANK");
        const char* world_str2 = getenv("WORLD_SIZE");
        if (rank_str2) ctx->ep_rank = atoi(rank_str2);
        if (world_str2) ctx->ep_world_size = atoi(world_str2);
        const char* local_rank_str2 = getenv("LOCAL_RANK");
        ctx->cuda_device = local_rank_str2 ? atoi(local_rank_str2) : g_cuda_device;
        const char* tp_size_str2 = getenv("TP_SIZE");
        if (!tp_size_str2) tp_size_str2 = getenv("RUSTRAIN_TP_SIZE");
        ctx->tp_world_size = tp_size_str2 ? atoi(tp_size_str2) : 1;
        ctx->tp_rank = ctx->tp_world_size > 0
            ? ctx->ep_rank % ctx->tp_world_size : 0;
        if (ctx->tp_world_size <= 0 ||
            (ctx->tp_world_size > 1 &&
                (ctx->data_parallel || ctx->ep_world_size != ctx->tp_world_size))) {
            ctx->topology_invalid = true;
            fprintf(stderr,
                "[tp_nccl] reject mixed topology: TP_SIZE=%d WORLD_SIZE=%d DATA_PARALLEL=%d\n",
                ctx->tp_world_size, ctx->ep_world_size, ctx->data_parallel ? 1 : 0);
            return -1;
        }
        ctx->topology_invalid = false;
        ctx->tp_comm = ctx->tp_world_size > 1 ? g_tp_comm : nullptr;
        ctx->tp_stream = ctx->tp_world_size > 1 ? g_tp_stream : nullptr;
        // In TP-only mode the parent communicator is reserved for the TP
        // split; EP layer collectives must remain disabled on replicated MoE.
        if (ctx->tp_world_size <= 1) {
            void* layer_comm = ctx->data_parallel
                ? nullptr : (void*)g_nccl_comm;
            void* layer_stream = ctx->data_parallel
                ? nullptr : (void*)g_nccl_stream;
            for (auto& lc : ctx->layer_configs) {
                lc.nccl_comm = layer_comm;
                lc.nccl_stream = layer_stream;
            }
            for (auto& lc : ctx->mtp_layer_configs) {
                lc.nccl_comm = layer_comm;
                lc.nccl_stream = layer_stream;
            }
        }
        synchronize_fixed_replicated_lora_parameters(ctx);
        return 0;
    }

    const char* rank_str = getenv("RANK");
    const char* world_str = getenv("WORLD_SIZE");
    if (!rank_str || !world_str) return -1;
    int rank = atoi(rank_str);
    int world_size = atoi(world_str);
    const char* tp_size_str = getenv("TP_SIZE");
    if (!tp_size_str) tp_size_str = getenv("RUSTRAIN_TP_SIZE");
    const int configured_tp_size = tp_size_str ? atoi(tp_size_str) : 1;
    const bool data_parallel_requested = env_enabled("RUSTRAIN_DATA_PARALLEL");
    if (configured_tp_size <= 0 ||
        (configured_tp_size > 1 &&
            (data_parallel_requested || world_size != configured_tp_size))) {
        ctx->topology_invalid = true;
        fprintf(stderr,
            "[tp_nccl] reject mixed topology: TP_SIZE=%d WORLD_SIZE=%d DATA_PARALLEL=%d\n",
            configured_tp_size, world_size, data_parallel_requested ? 1 : 0);
        return -1;
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
    const int tp_size = tp_size_str ? atoi(tp_size_str) : 1;
    if (tp_size <= 0 || world_size % tp_size != 0) {
        fprintf(stderr, "[tp_nccl] invalid TP_SIZE=%d for WORLD_SIZE=%d\n",
            tp_size, world_size);
        return -1;
    }
    if (tp_size > 1) {
        // The default rank order makes TP the least-significant axis. A
        // future multi-axis implementation must pass an explicit color/key
        // mapping instead of reusing this world-rank split.
        ncclResult_t tp_err = ncclCommSplit(
            comm, rank / tp_size, rank % tp_size, &g_tp_comm, nullptr);
        if (tp_err != ncclSuccess) {
            fprintf(stderr, "[tp_nccl] ncclCommSplit failed: %d (%s)\n",
                tp_err, ncclGetErrorString(tp_err));
            return -1;
        }
        g_tp_stream = nccl_stream;
    }
    g_nccl_initialized = true;

    ctx->nccl_comm = comm;
    ctx->nccl_stream = nccl_stream;
    ctx->ep_rank = rank;
    ctx->ep_world_size = world_size;
    ctx->data_parallel = env_enabled("RUSTRAIN_DATA_PARALLEL");
    ctx->tp_world_size = tp_size;
    ctx->tp_rank = rank % tp_size;
    ctx->tp_comm = tp_size > 1 ? g_tp_comm : nullptr;
    ctx->tp_stream = tp_size > 1 ? g_tp_stream : nullptr;

    // Propagate to layer configs
    if (tp_size <= 1) {
        void* layer_comm = ctx->data_parallel ? nullptr : (void*)comm;
        void* layer_stream = ctx->data_parallel
            ? nullptr : (void*)nccl_stream;
        for (auto& lc : ctx->layer_configs) {
            lc.nccl_comm = layer_comm;
            lc.nccl_stream = layer_stream;
        }
        for (auto& lc : ctx->mtp_layer_configs) {
            lc.nccl_comm = layer_comm;
            lc.nccl_stream = layer_stream;
        }
    }

    synchronize_fixed_replicated_lora_parameters(ctx);
    return 0;
}

// Set NCCL communicator for Expert Parallel all-reduce (legacy, from Rust)
__attribute__((visibility("default"))) void qwen36_set_nccl_comm(
    void* ctx_ptr, void* comm_ptr, void* stream_ptr,
    int32_t ep_rank, int32_t ep_world_size
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    ctx->nccl_comm = reinterpret_cast<ncclComm_t>(comm_ptr);
    ctx->nccl_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    ctx->ep_rank = ep_rank;
    ctx->ep_world_size = ep_world_size;
    ctx->data_parallel = env_enabled("RUSTRAIN_DATA_PARALLEL");
    if (ctx->tp_world_size <= 0 ||
        (ctx->tp_world_size > 1 &&
            (ctx->data_parallel || ep_world_size != ctx->tp_world_size))) {
        ctx->topology_invalid = true;
        fprintf(stderr,
            "[tp_nccl] reject mixed topology: TP_SIZE=%d WORLD_SIZE=%d DATA_PARALLEL=%d\n",
            ctx->tp_world_size, ep_world_size, ctx->data_parallel ? 1 : 0);
        return;
    }
    ctx->topology_invalid = false;
    int current_device = g_cuda_device;
    cudaGetDevice(&current_device);
    ctx->cuda_device = current_device;
    // Only EP owns routed-output collectives. In pure replicated DP the world
    // communicator belongs exclusively to LoRA gradient synchronization;
    // exposing it to moe_forward would mix activations from unrelated samples.
    void* layer_comm = ctx->data_parallel ? nullptr : comm_ptr;
    void* layer_stream = ctx->data_parallel ? nullptr : stream_ptr;
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
        ctx->attention_mask = *reinterpret_cast<at::Tensor*>(mask_ptr);
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
        TORCH_CHECK(rank > 0, "LoRA rank must be positive");
        TORCH_CHECK(rank % ctx->tp_world_size == 0,
            "dynamic LoRA rank ", rank, " must be divisible by TP_SIZE=",
            ctx->tp_world_size);
        const int64_t local_rank = rank / ctx->tp_world_size;
        TORCH_CHECK(alpha > 0.0, "LoRA alpha must be positive");
        TrainingContext::LoRAAdapter adapter;
        adapter.id = ++ctx->next_adapter_id;
        adapter.rank = rank;
        adapter.alpha = alpha;
        if (target_layers && num_target_layers > 0) {
            for (int64_t i = 0; i < num_target_layers; i++)
                adapter.target_layers.insert(target_layers[i]);
        }
        if (target_modules_str) {
            std::string s(target_modules_str);
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ','))
                adapter.target_modules.insert(item);
        }
        if (ctx->base_tp_mlp) {
            TORCH_CHECK(!adapter.target_modules.empty(),
                "base MLP tensor parallelism does not support a dynamic LoRA "
                "adapter targeting all modules; use explicit attention-only targets");
            for (const auto& name : adapter.target_modules) {
                TORCH_CHECK(!is_mlp_lora_target(name),
                    "base MLP tensor parallelism does not yet support dynamic MLP LoRA target ", name,
                    "; use attention-only targets until projection-axis LoRA collectives are implemented");
            }
        }
        for (auto layer : adapter.target_layers) {
            TORCH_CHECK(layer >= 0 && layer < ctx->num_layers,
                "dynamic LoRA target layer out of range: ", layer,
                " for model with ", ctx->num_layers, " layers");
        }
        // The activation-level batch path stacks A/B across adapters. Keep
        // the batch rectangular and semantically aligned instead of waiting
        // for an opaque ATen stack/shape failure during the first step.
        if (!ctx->adapters.empty()) {
            const auto& reference = ctx->adapters.front();
            TORCH_CHECK(rank == reference.rank,
                "dynamic LoRA adapters in one batch must use the same rank");
            TORCH_CHECK(adapter.target_layers == reference.target_layers,
                "dynamic LoRA adapters in one batch must use identical target_layers");
            TORCH_CHECK(adapter.target_modules == reference.target_modules,
                "dynamic LoRA adapters in one batch must use identical target_modules");
        }
        for (const auto& name : adapter.target_modules) {
            TORCH_CHECK(
                name == "q_proj" || name == "k_proj" || name == "v_proj" ||
                name == "o_proj" || name == "in_proj_qkv" ||
                name == "in_proj_z" || name == "in_proj_a" ||
                name == "in_proj_b" || name == "out_proj" ||
                name == "gate_proj" || name == "up_proj" ||
                name == "down_proj" || name == "shared_gate_proj" ||
                name == "shared_up_proj" || name == "shared_down_proj" ||
                name == "experts_gate_up_proj" || name == "experts_down_proj",
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
                "dynamic LoRA target module does not exist in this model: ", name);
        }
        for (int64_t i = 0; i < ctx->num_layers; i++) {
            if (!adapter.target_layers.empty() && adapter.target_layers.find(i) == adapter.target_layers.end())
                continue;
            int64_t w_offset = 0;
            for (int64_t j = 0; j < i; j++)
                w_offset += weight_count_for_layer(ctx->layer_configs[j]);
            auto projection_table = lora_projection_table(ctx->layer_configs[i]);
            int64_t num_pairs = projection_table.count;
            std::vector<std::pair<at::Tensor, at::Tensor>> pairs;
            std::vector<std::array<at::Tensor, 4>> adam_states;
            std::vector<std::array<at::Tensor, 2>> grad_accumulators;
            for (int64_t k = 0; k < num_pairs; k++) {
                const auto& projection = projection_table.entries[k];
                auto* base = ctx->weight_ptrs[w_offset + projection.weight_index];
                // Preserve the historical empty-target default (attention
                // projections only). Explicit target lists may additionally
                // select any 2D dense/shared MLP projection.
                bool active = adapter.target_modules.empty()
                    ? !projection.grouped_expert &&
                        projection.segment == LoraSegment::Attention
                    : adapter.target_modules.find(projection.name) !=
                        adapter.target_modules.end();
                auto opts = at::TensorOptions().dtype(ctx->compute_type).device(base->device());
                at::Tensor a, b;
                if (active) {
                    if (projection.grouped_expert) {
                        TORCH_CHECK(base->dim() == 3,
                            "dynamic routed-expert LoRA projection must be rank 3: ",
                            projection.name);
                        int64_t experts = base->size(0);
                        int64_t out_f = base->size(1), in_f = base->size(2);
                        a = initialize_lora_a(ctx, opts, experts, rank, in_f);
                        b = at::zeros({experts, out_f, local_rank}, opts);
                    } else {
                        TORCH_CHECK(base->dim() == 2,
                            "dynamic LoRA projection must be a matrix: ", projection.name);
                        int64_t out_f = base->size(0), in_f = base->size(1);
                        const auto layout = lora_tp_layout(ctx, i, k);
                        if (layout == LoraTpLayout::ColumnParallel ||
                            layout == LoraTpLayout::RowParallel) {
                            a = at::randn({rank, in_f}, opts) * 0.01;
                            b = at::zeros({out_f, rank}, opts);
                        } else {
                            a = initialize_lora_a(ctx, opts, 0, rank, in_f);
                            b = at::zeros({out_f, local_rank}, opts);
                        }
                    }
                } else {
                    a = at::zeros({}, opts);
                    b = at::zeros({}, opts);
                }
                a.set_requires_grad(active);
                b.set_requires_grad(active);
                // Adam state: FP32 for numerical stability
                auto opts_f32 = at::TensorOptions().dtype(at::kFloat).device(base->device());
                adam_states.push_back({
                    at::zeros(a.sizes(), opts_f32), at::zeros(a.sizes(), opts_f32),
                    at::zeros(b.sizes(), opts_f32), at::zeros(b.sizes(), opts_f32)
                });
                grad_accumulators.push_back(active
                    ? std::array<at::Tensor, 2>{
                        at::zeros(a.sizes(), opts_f32),
                        at::zeros(b.sizes(), opts_f32)}
                    : std::array<at::Tensor, 2>{at::Tensor(), at::Tensor()});
                pairs.emplace_back(std::move(a), std::move(b));
            }
            adapter.params[i] = std::move(pairs);
            adapter.adam_state[i] = std::move(adam_states);
            adapter.grad_accum[i] = std::move(grad_accumulators);
        }
        synchronize_adapter_replicated_lora_parameters(ctx, adapter);
        int64_t id = adapter.id;
        ctx->adapters.push_back(std::move(adapter));
        ctx->lora_cache_valid = false;
        ctx->lora_batch_valid = false;
        fprintf(stderr, "[q36_lora] added adapter %ld: rank=%ld alpha=%.1f\n", (long)id, (long)rank, alpha);
        return id;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36] add_lora FAILED: %s\n", e.what());
        return -1;
    }
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
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    for (auto it = ctx->adapters.begin(); it != ctx->adapters.end(); ++it) {
        if (it->id == adapter_id) {
            ctx->adapters.erase(it);
            ctx->lora_cache_valid = false;
            ctx->lora_batch_valid = false;
            return 1;
        }
    }
    return 0;
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
        if (attention_mask_ptr)
            ctx->attention_mask = *reinterpret_cast<at::Tensor*>(attention_mask_ptr);
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

__attribute__((visibility("default")))
int64_t qwen36_get_step_count(void* ctx_ptr) {
    return (int64_t)reinterpret_cast<TrainingContext*>(ctx_ptr)->step_count;
}

// Restore the Adam bias-correction clock independently from tensor state.
// Checkpoint loading imports m/v through a separate ABI, so omitting this
// value would resume the next update as step 1 even for a mature optimizer.
__attribute__((visibility("default")))
int32_t qwen36_set_step_count(void* ctx_ptr, int64_t step_count) {
    if (!ctx_ptr || step_count < 0) return -1;
    reinterpret_cast<TrainingContext*>(ctx_ptr)->step_count = step_count;
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
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    int64_t imported = 0;
    for (int64_t i = 0; i < count && i < (int64_t)ctx->adam_m.size(); i++) {
        auto* src_m = reinterpret_cast<at::Tensor*>(m_ptrs[i]);
        auto* src_v = reinterpret_cast<at::Tensor*>(v_ptrs[i]);
        if (src_m && src_v) {
            ctx->adam_m[i] = src_m->clone();
            ctx->adam_v[i] = src_v->clone();
            imported++;
        }
    }
    return imported;
}


}  // extern "C"
