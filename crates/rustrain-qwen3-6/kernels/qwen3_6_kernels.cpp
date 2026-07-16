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

// Forward declarations for functions defined after TrainingContext
at::Tensor apply_multi_lora(TrainingContext* ctx, int64_t layer_idx, int64_t pair_idx, const at::Tensor& base_weight);

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

struct RoutedExpertLora {
    const at::Tensor* gate_up_a = nullptr;  // [local_experts, rank, hidden]
    const at::Tensor* gate_up_b = nullptr;  // [local_experts, 2*intermediate, rank]
    const at::Tensor* down_a = nullptr;     // [local_experts, rank, intermediate]
    const at::Tensor* down_b = nullptr;     // [local_experts, hidden, rank]
    double scaling = 0.0;
};

// Per-sample adapter projection used by the batched multi-LoRA path.  This is
// intentionally separate from RoutedExpertLora: routed experts carry one
// A/B pair per local expert, while dense/shared projections carry one pair per
// adapter sample.
struct LoraBatchEntry {
    at::Tensor a_stack;    // [N, rank, in]
    at::Tensor b_stack;    // [N, out, rank]
    at::Tensor scaling;     // [N, 1, 1]
};

static const LoraBatchEntry* lora_batch_entry(
    TrainingContext* ctx, int64_t layer_idx, int64_t pair_idx);
static at::Tensor dense_mlp_forward_batched(
    TrainingContext* ctx, int64_t layer_idx, const at::Tensor& hidden,
    const at::Tensor& gate_proj, const at::Tensor& up_proj,
    const at::Tensor& down_proj, at::ScalarType compute_type);

static at::Tensor lora_activation_delta(
    const at::Tensor& x, const at::Tensor& A, const at::Tensor& B,
    const at::Tensor& scaling);

static at::Tensor add_batched_lora(
    const at::Tensor& base, const at::Tensor& input,
    const LoraBatchEntry* entry
) {
    if (!entry) return base;
    return base + lora_activation_delta(
        input, entry->a_stack, entry->b_stack, entry->scaling);
}

// Per-token routed-expert LoRA. Dynamic adapters add a leading sample axis to
// the expert-local tensors: A [batch, experts, rank, in],
// B [batch, experts, out, rank]. Flattening (sample, expert) lets one pair of
// index_select + bmm operations select the correct adapter and expert without
// materializing a full-rank delta weight.
static at::Tensor dynamic_expert_lora_delta(
    const at::Tensor& input,
    const at::Tensor& token_indices,
    const at::Tensor& local_expert_indices,
    int64_t seq,
    const LoraBatchEntry* entry
) {
    if (!entry) return at::zeros({0}, input.options());
    TORCH_CHECK(entry->a_stack.dim() == 4 && entry->b_stack.dim() == 4,
        "dynamic routed-expert LoRA expects rank-4 stacked A/B tensors");
    const int64_t local_experts = entry->a_stack.size(1);
    auto sample_indices = at::floor_divide(token_indices, seq);
    auto pair_indices = sample_indices * local_experts + local_expert_indices;
    auto a = entry->a_stack.flatten(0, 1)
        .index_select(0, pair_indices).to(input.scalar_type());
    auto b = entry->b_stack.flatten(0, 1)
        .index_select(0, pair_indices).to(input.scalar_type());
    auto low_rank = at::bmm(a, input.unsqueeze(-1)).squeeze(-1);
    auto delta = at::bmm(b, low_rank.unsqueeze(-1)).squeeze(-1);
    auto scaling = entry->scaling.index_select(0, sample_indices)
        .reshape({-1, 1}).to(input.scalar_type());
    return delta * scaling;
}

static at::Tensor moe_forward(
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
    for (int64_t kk = 0; kk < top_k; kk++) {
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
                    gu = gu + delta * expert_lora.scaling;
                }
                if (expert_gate_up_lora) {
                    gu = gu + dynamic_expert_lora_delta(
                        selected, token_indices, local_expert_indices,
                        seq, expert_gate_up_lora);
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
                    expert_out = expert_out + delta * expert_lora.scaling;
                }
                if (expert_down_lora) {
                    expert_out = expert_out + dynamic_expert_lora_delta(
                        activated, token_indices, local_expert_indices,
                        seq, expert_down_lora);
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
                    gu = gu + at::matmul(at::matmul(selected, a.t()), b.t()) * expert_lora.scaling;
                }
                if (expert_gate_up_lora) {
                    gu = gu + dynamic_expert_lora_delta(
                        selected, token_indices, local_expert_indices,
                        seq, expert_gate_up_lora);
                }
                auto gate_part = gu.narrow(-1, 0, intermediate);
                auto up_part = gu.narrow(-1, intermediate, intermediate);
                auto activated = fused_swiglu_op(gate_part, up_part, 0.0);
                auto expert_out = at::matmul(activated, ed.t());
                if (expert_lora.down_a && expert_lora.down_b) {
                    auto a = expert_lora.down_a->select(0, e_local);
                    auto b = expert_lora.down_b->select(0, e_local);
                    expert_out = expert_out
                        + at::matmul(at::matmul(activated, a.t()), b.t()) * expert_lora.scaling;
                }
                if (expert_down_lora) {
                    expert_out = expert_out + dynamic_expert_lora_delta(
                        activated, token_indices, local_expert_indices,
                        seq, expert_down_lora);
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
    if (!routed_output.requires_grad()) {
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
    if (nccl_comm_v) {
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
            shared_gate.reshape({batch, seq, -1}), hidden, shared_gate_lora)
            .reshape({batch * seq, -1});
    }
    if (shared_up_lora) {
        shared_up = add_batched_lora(
            shared_up.reshape({batch, seq, -1}), hidden, shared_up_lora)
            .reshape({batch * seq, -1});
    }
    auto shared_hidden = fused_swiglu_op(
        shared_gate.reshape({batch, seq, -1}),
        shared_up.reshape({batch, seq, -1}), 0.0);
    auto shared_out = at::matmul(shared_hidden.reshape({batch * seq, -1}), shared_down_proj.t());
    if (shared_down_lora) {
        shared_out = add_batched_lora(
            shared_out.reshape({batch, seq, -1}), shared_hidden, shared_down_lora)
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
            auto mlp_out = moe_forward(cfg->nccl_comm, cfg->nccl_stream, post_attn,
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
            auto mlp_out = dense_mlp_forward(post_attn, gate, up, down, kind);
            return hidden + attn_output + mlp_out;
        }
    } else {
        // Linear attention
        auto in_proj_qkv = *w[2], in_proj_z = *w[3], in_proj_a = *w[4], in_proj_b = *w[5];
        auto a_log = *w[6], dt_bias = *w[7], conv1d_w = *w[8], norm_w = *w[9], out_proj = *w[10];
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
            auto mlp_out = moe_forward(cfg->nccl_comm, cfg->nccl_stream, post_attn,
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
            auto mlp_out = dense_mlp_forward(post_attn, gate, up, down, kind);
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
        double alpha;
        std::set<int64_t> target_layers;
        std::set<std::string> target_modules;
        std::map<int64_t, std::vector<std::pair<at::Tensor, at::Tensor>>> params;
        std::map<int64_t, std::vector<std::array<at::Tensor, 4>>> adam_state;
    };

    std::vector<LoRAAdapter> adapters;
    int64_t next_adapter_id = 0;

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
    std::vector<uint8_t> lora_active;
    std::vector<int64_t> lora_layer_offset;
    double lora_scaling;
    std::vector<std::string> lora_names;

    // Adam optimizer state
    std::vector<at::Tensor> adam_m;
    std::vector<at::Tensor> adam_v;
    double lr, beta1, beta2, eps;
    int64_t step_count;

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
    int cuda_device = 0;
// ──────────────────────────────────────────────────────────────────────
};

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

static void allreduce_lora_grad(
    TrainingContext* ctx, at::Tensor& param, double local_token_scale
) {
    auto grad = param.grad();
    if (!ctx->nccl_comm || !grad.defined()) return;
    auto contiguous = grad.contiguous();
    if (local_token_scale != 1.0) {
        contiguous = contiguous * local_token_scale;
    }
    auto reduced = at::empty_like(contiguous);
    int dev = contiguous.device().index();
    cudaSetDevice(dev);
    auto stream = c10::cuda::getCurrentCUDAStream(dev).stream();
    auto err = ncclAllReduce(
        contiguous.data_ptr(), reduced.data_ptr(), contiguous.numel(),
        nccl_dtype_for(contiguous), ncclSum, ctx->nccl_comm, stream);
    TORCH_CHECK(err == ncclSuccess, "NCCL LoRA gradient all-reduce failed: ",
                ncclGetErrorString(err));
    param.mutable_grad() = reduced;
}

// Every rank evaluates the complete loss. Average replicated LoRA gradients
// across DP ranks with token-count weighting so their Adam update matches a
// single global batch. EP ranks already receive the complete routed activation
// in forward and keep replicated gradients local; routed expert adapters remain
// local because their parameter tensors are sharded.
static void synchronize_lora_gradients(
    TrainingContext* ctx, const at::Tensor& target_mask
) {
    if (!ctx->nccl_comm || !ctx->data_parallel) return;
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
    const double token_scale = local_tokens / std::max(global_tokens, 1.0);
    for (auto& adapter : ctx->adapters) {
        for (auto& [layer_idx, pairs] : adapter.params) {
            auto table = lora_projection_table(ctx->layer_configs[layer_idx]);
            for (int64_t pair = 0; pair < (int64_t)pairs.size(); ++pair) {
                // Dynamic routed-expert tensors are sharded exactly like the
                // base experts; only replicated adapter tensors are reduced.
                if (table.entries[pair].grouped_expert) continue;
                auto& [a, b] = pairs[pair];
                allreduce_lora_grad(ctx, a, token_scale);
                allreduce_lora_grad(ctx, b, token_scale);
            }
        }
    }
    for (int64_t layer = 0; layer < ctx->num_layers; ++layer) {
        auto table = lora_projection_table(ctx->layer_configs[layer]);
        int64_t offset = ctx->lora_layer_offset[layer];
        for (int64_t pair = 0; pair < table.count; ++pair) {
            // Routed expert LoRA is sharded with the base expert weights. Its
            // local gradients belong only to this EP rank and must not be
            // summed with a different expert shard on another rank.
            if (table.entries[pair].grouped_expert) continue;
            allreduce_lora_grad(ctx, ctx->lora_a[offset + pair], token_scale);
            allreduce_lora_grad(ctx, ctx->lora_b[offset + pair], token_scale);
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
                a_stack, b_stack, scaling
            };
        }
    }
    ctx->lora_batch_valid = true;
}

/// Compute B@(A@x) * scaling — never materializes B@A.
/// x: [N, seq, in], A: [N, rank, in], B: [N, out, rank], scaling: [N, 1, 1]
/// returns: [N, seq, out]
static at::Tensor lora_activation_delta(
    const at::Tensor& x,          // [N, seq, in]
    const at::Tensor& A,          // [N, rank, in]
    const at::Tensor& B,          // [N, out, rank]
    const at::Tensor& scaling     // [N, 1, 1]
) {
    // Cast to compute dtype (BF16)
    auto kind = x.scalar_type();
    auto A_c = A.to(kind);
    auto B_c = B.to(kind);
    auto s_c = scaling.to(kind);
    // Ax = A @ x^T  → [N, rank, seq]
    auto Ax = at::bmm(A_c, x.transpose(-2, -1));
    // delta = B @ Ax → [N, out, seq] → transpose → [N, seq, out]
    auto delta = at::bmm(B_c, Ax).transpose(-2, -1);
    return delta * s_c;
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

    auto gate_out = at::matmul(hidden, gate_proj.t());
    auto up_out = at::matmul(hidden, up_proj.t());
    gate_out = add_batched_lora(
        gate_out, hidden, lora_batch_entry(ctx, layer_idx, gate_pair));
    up_out = add_batched_lora(
        up_out, hidden, lora_batch_entry(ctx, layer_idx, up_pair));
    auto activated = fused_swiglu_op(gate_out, up_out, 0.0);
    auto result = at::matmul(activated, down_proj.t());
    result = add_batched_lora(
        result, activated, lora_batch_entry(ctx, layer_idx, down_pair));
    return result.to(compute_type);
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
        return moe_forward(cfg.nccl_comm, cfg.nccl_stream, post_attn,
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
        return dense_mlp_forward(post_attn, gate, up, down, kind);
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
    int64_t qkv_dim = num_heads * head_dim;

    auto q = at::matmul(hidden, q_proj.t());
    auto k = at::matmul(hidden, k_proj.t());
    auto v = at::matmul(hidden, v_proj.t());

    // Apply activation-level LoRA: q += B@(A@hidden) * scaling
    auto it_q = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 0));
    if (it_q != ctx->lora_batch_cache.end()) {
        q = q + lora_activation_delta(hidden, it_q->second.a_stack, it_q->second.b_stack, it_q->second.scaling);
    }
    auto it_k = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 1));
    if (it_k != ctx->lora_batch_cache.end()) {
        k = k + lora_activation_delta(hidden, it_k->second.a_stack, it_k->second.b_stack, it_k->second.scaling);
    }
    auto it_v = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 2));
    if (it_v != ctx->lora_batch_cache.end()) {
        v = v + lora_activation_delta(hidden, it_v->second.a_stack, it_v->second.b_stack, it_v->second.scaling);
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
        result = result + lora_activation_delta(attn_flat,
            it_o->second.a_stack, it_o->second.b_stack, it_o->second.scaling);
    }
    return result;
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
    auto device = hidden.device();
    int64_t batch = hidden.size(0), seq = hidden.size(1);
    int64_t q_size = num_k_heads * key_dim;
    int64_t v_size = num_v_heads * val_dim;
    int64_t qkv_dim = q_size * 2 + v_size;

    // QKV projection + LoRA delta
    auto qkv = at::matmul(hidden, in_proj_qkv.t());
    auto it_qkv = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 0));
    if (it_qkv != ctx->lora_batch_cache.end()) {
        qkv = qkv + lora_activation_delta(hidden, it_qkv->second.a_stack, it_qkv->second.b_stack, it_qkv->second.scaling);
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

    auto a = at::matmul(hidden, in_proj_a.t());
    auto b = at::matmul(hidden, in_proj_b.t());
    auto it_a = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 2));
    if (it_a != ctx->lora_batch_cache.end()) {
        a = a + lora_activation_delta(hidden, it_a->second.a_stack, it_a->second.b_stack, it_a->second.scaling);
    }
    auto it_b = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 3));
    if (it_b != ctx->lora_batch_cache.end()) {
        b = b + lora_activation_delta(hidden, it_b->second.a_stack, it_b->second.b_stack, it_b->second.scaling);
    }

    // Z projection + LoRA delta
    auto z = at::matmul(hidden, in_proj_z.t());
    auto it_z = ctx->lora_batch_cache.find(lora_cache_key(layer_idx, 1));
    if (it_z != ctx->lora_batch_cache.end()) {
        z = z + lora_activation_delta(hidden, it_z->second.a_stack, it_z->second.b_stack, it_z->second.scaling);
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
        result = result + lora_activation_delta(gated, it_op->second.a_stack, it_op->second.b_stack, it_op->second.scaling);
    }

    return result;
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
    if (ctx->lora_batch_valid) prepare_lora_batch(ctx);
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
    if (ctx->lora_batch_valid) prepare_lora_batch(ctx);
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
    // Use batched path if multiple adapters, else legacy weight-level
    if (ctx->lora_batch_valid) {
        prepare_lora_batch(ctx);
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
        else precompute_lora_cache(ctx);

        // Recompute forward with grad for this group only
        auto output = forward_layer_group(ctx, input, start, end);

        // Backprop through this group using grad() instead of backward().
        // grad() only computes gradients for specified inputs — faster than
        // backward() which traverses all leaf nodes.
        // LoRA params are shared across groups, so we accumulate their gradients.
        std::vector<at::Tensor> grad_inputs = {input};

        if (ctx->lora_batch_valid) {
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
        if (ctx->lora_batch_valid) {
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
    return 8;
}

// Create training context — called once at startup
// lora_rank: LoRA rank (from config)
// target_layers: array of layer indices to apply LoRA (nullptr = all layers)
// num_target_layers: length of target_layers array
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
    try {
        auto* ctx = new TrainingContext();
        ctx->compute_type = static_cast<at::ScalarType>(compute_type);
        ctx->lr = lr; ctx->beta1 = beta1; ctx->beta2 = beta2; ctx->eps = eps;
        ctx->vocab_size = vocab_size; ctx->rms_eps = rms_eps;
        ctx->step_count = 0; ctx->lora_scaling = lora_scaling;
        ctx->num_layers = num_layers;
        ctx->use_checkpoint = false; ctx->group_size = 4;
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
                    a = at::randn({experts, lora_rank, in_f}, opts) * 0.01;
                    b = at::zeros({experts, out_f, lora_rank}, opts);
                } else {
                    int64_t out_f = base->size(0), in_f = base->size(1);
                    a = at::randn({lora_rank, in_f}, opts) * 0.01;
                    b = at::zeros({out_f, lora_rank}, opts);
                }
                a.set_requires_grad(active);
                b.set_requires_grad(active);
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

// Set MTP weights on an existing training context.
// Called after create_training_context if MTP is enabled.
__attribute__((visibility("default"))) void qwen36_set_mtp_weights(
    void* ctx_ptr,
    void* mtp_fc_ptr,
    void* mtp_pre_fc_norm_emb_ptr,
    void* mtp_pre_fc_norm_hidden_ptr,
    void* mtp_norm_ptr,
    void** mtp_layer_weight_ptrs, int64_t num_mtp_layer_weights,
    void* mtp_layer_configs_ptr, int64_t num_mtp_layers
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
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

        if (gradient_scale != 1.0) {
            total_hidden_grad.mul_(gradient_scale);
        }

        // Trigger exactly one main-model backward with the combined hidden
        // gradient. Manual groups are the non-autograd checkpoint fallback;
        // normal and sub-checkpoint paths use the real graph.
        if (!ctx->group_inputs.empty() && !env_enabled("QWEN36_SUBCKPT")) {
            manual_group_backward(ctx, total_hidden_grad);
        } else {
            hidden.backward(total_hidden_grad);
        }

        if (!apply_optimizer) {
            // The forward graph has been consumed, but parameters remain
            // live for the next micro-batch. Never reuse cached LoRA deltas
            // whose autograd nodes were freed by this backward.
            ctx->lora_cache_valid = false;
            ctx->lora_batch_valid = false;
            return loss_val;
        }

        // Replicated DP gradients are synchronized before the local Adam
        // update. EP keeps replicated gradients local because its forward
        // routed activation already contains the cross-rank sum.
        synchronize_lora_gradients(ctx, target_mask);

        // ── Adam optimizer step — CUDA multi-tensor fused kernel ──
        at::AutoGradMode guard(false);
        ctx->step_count++;
        ctx->lora_cache_valid = false;
        ctx->lora_batch_valid = false;
        double step_f = (double)ctx->step_count;
        double bias_correction1 = 1.0 - std::pow(ctx->beta1, step_f);
        double bias_correction2 = 1.0 - std::pow(ctx->beta2, step_f);
        float lr_scaled = (float)(ctx->lr / bias_correction1);
        float eps_scaled = (float)(ctx->eps / std::sqrt(bias_correction2));
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
                for (size_t i = 0; i < pairs.size(); i++) {
                    auto& [a, b] = pairs[i];
                    auto& [m_a, v_a, m_b, v_b] = adam_states[i];
                    if (a.grad().defined() && a.scalar_type() == at::kBFloat16) {
                        h_params.push_back(a.data_ptr());
                        h_grads.push_back(a.grad().data_ptr());
                        h_m.push_back((float*)m_a.data_ptr());
                        h_v.push_back((float*)v_a.data_ptr());
                        h_sizes.push_back((int)a.numel());
                    }
                    if (b.grad().defined() && b.scalar_type() == at::kBFloat16) {
                        h_params.push_back(b.data_ptr());
                        h_grads.push_back(b.grad().data_ptr());
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
                auto& grad = param.grad();
                if (grad.defined() && param.scalar_type() == at::kBFloat16) {
                    h_params.push_back(param.data_ptr());
                    h_grads.push_back(grad.data_ptr());
                    h_m.push_back((float*)ctx->adam_m[adam_idx].data_ptr());
                    h_v.push_back((float*)ctx->adam_v[adam_idx].data_ptr());
                    h_sizes.push_back((int)param.numel());
                }
            }
            adam_idx++;
            {
                auto& param = ctx->lora_b[i];
                auto& grad = param.grad();
                if (grad.defined() && param.scalar_type() == at::kBFloat16) {
                    h_params.push_back(param.data_ptr());
                    h_grads.push_back(grad.data_ptr());
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
        }

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
        TORCH_CHECK(input_ids.dim() == 2 && target_mask.dim() == 2,
            "multi-LoRA inputs must have shape [batch, seq]");
        const int64_t input_batch = input_ids.size(0);
        TORCH_CHECK(input_batch == 1 || input_batch == n_total,
            "multi-LoRA input batch must be 1 or n_total (batch=", input_batch,
            ", n_total=", n_total, ")");
        TORCH_CHECK(target_mask.size(0) == input_batch &&
                    target_mask.size(1) == input_ids.size(1),
            "target_mask must match input_ids shape");

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

        // Compute N_max from available GPU memory.
        // CRITICAL: all workers must agree on n_max to keep NCCL all-reduce in sync.
        // Use file-based barrier: rank 0 computes n_max, writes to file, others read.
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        int64_t n_max;
        if (ctx->nccl_comm && ctx->ep_world_size > 1) {
            const std::string sync_path = nccl_sync_dir() + "/nmax_sync.txt";
            if (ctx->ep_rank == 0) {
                n_max = compute_n_max(
                    (int64_t)free_mem, lora_rank,
                    input_ids.size(-1), 2048,
                    ctx->group_size, ctx->num_layers
                );
                n_max = std::min(n_max, total_adapters);
                if (n_max < 1) n_max = 1;
                FILE* f = fopen(sync_path.c_str(), "w");
                fprintf(f, "%ld\n", (long)n_max);
                fclose(f);
            } else {
                for (int i = 0; i < 600; i++) {
                    FILE* f = fopen(sync_path.c_str(), "r");
                    if (f) { fscanf(f, "%ld", (long*)&n_max); fclose(f); break; }
                    usleep(10000);
                }
            }
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
        // Chunking is a memory scheduling detail, not an optimizer step. All
        // adapters in this call must use the same Adam bias correction.
        ctx->step_count++;
        const double logical_step = (double)ctx->step_count;
        const double bias_correction1 = 1.0 - std::pow(ctx->beta1, logical_step);
        const double bias_correction2 = 1.0 - std::pow(ctx->beta2, logical_step);
        const float lr_scaled = (float)(ctx->lr / bias_correction1);
        const float eps_scaled = (float)(ctx->eps / std::sqrt(bias_correction2));
        const float one_minus_b1 = (float)(1.0 - ctx->beta1);
        const float one_minus_b2 = (float)(1.0 - ctx->beta2);

        for (int64_t chunk = 0; chunk < num_chunks; chunk++) {
            int64_t start = chunk * n_max;
            int64_t end = std::min(start + n_max, total_adapters);
            int64_t n = end - start;

            // Invalidate cache for this chunk's adapter set
            ctx->lora_batch_valid = false;
            ctx->lora_cache_valid = false;

            // Temporarily set lora_batch_valid so prepare_lora_batch runs
            // We need to select only adapters[start:end]
            // HACK: move non-chunk adapters to a temp vector, run, then restore
            std::vector<TrainingContext::LoRAAdapter> all_adapters;
            all_adapters.swap(ctx->adapters);
            ctx->adapters.assign(
                all_adapters.begin() + start, all_adapters.begin() + end);

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
            ctx->adapters.swap(all_adapters);

            if (chunk == num_chunks - 1) {
                // DP gradient synchronization and Adam belong to the logical
                // multi-tenant step, never to an activation-memory chunk.
                synchronize_lora_gradients(ctx, target_mask);

                // Adam step
                at::AutoGradMode guard(false);
                ctx->lora_cache_valid = false;
                ctx->lora_batch_valid = false;

                std::vector<void*> h_params, h_grads;
                std::vector<float*> h_m, h_v;
                std::vector<int> h_sizes;

                for (auto& adapter : ctx->adapters) {
                    for (auto& [layer_idx, pairs] : adapter.params) {
                        auto& adam_states = adapter.adam_state[layer_idx];
                        for (size_t i = 0; i < pairs.size(); i++) {
                            auto& [a, b] = pairs[i];
                            auto& [m_a, v_a, m_b, v_b] = adam_states[i];
                            if (a.grad().defined() && a.scalar_type() == at::kBFloat16) {
                                h_params.push_back(a.data_ptr());
                                h_grads.push_back(a.grad().data_ptr());
                                h_m.push_back((float*)m_a.data_ptr());
                                h_v.push_back((float*)v_a.data_ptr());
                                h_sizes.push_back((int)a.numel());
                            }
                            if (b.grad().defined() && b.scalar_type() == at::kBFloat16) {
                                h_params.push_back(b.data_ptr());
                                h_grads.push_back(b.grad().data_ptr());
                                h_m.push_back((float*)m_b.data_ptr());
                                h_v.push_back((float*)v_b.data_ptr());
                                h_sizes.push_back((int)b.numel());
                            }
                        }
                    }
                }

                if (!h_params.empty()) {
                    int n_params = (int)h_params.size();
                    auto opts_cpu_long = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
                    auto opts_cpu_int  = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
                    auto params_cpu = at::from_blob(h_params.data(), {n_params}, opts_cpu_long);
                    auto grads_cpu  = at::from_blob(h_grads.data(),  {n_params}, opts_cpu_long);
                    auto m_cpu      = at::from_blob(h_m.data(),      {n_params}, opts_cpu_long);
                    auto v_cpu      = at::from_blob(h_v.data(),      {n_params}, opts_cpu_long);
                    auto sizes_cpu  = at::from_blob(h_sizes.data(),  {n_params}, opts_cpu_int);
                    ctx->adam_dev_bufs.ensure(n_params, ctx->adapters[0].params.begin()->second[0].first);
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
                        n_params,
                        (float)ctx->beta1, (float)ctx->beta2,
                        lr_scaled, eps_scaled,
                        one_minus_b1, one_minus_b2,
                        (void*)stream
                    );
                }
            }

            total_loss += loss_val;

            fprintf(stderr, "[train_multi] chunk %ld/%ld: n=%ld loss=%.6f\n",
                    (long)(chunk + 1), (long)num_chunks, (long)n, loss_val);
        }

        ctx->attention_mask = saved_attention_mask;
        return total_loss / total_adapters;
    } catch (const std::exception& e) {
        fprintf(stderr, "[train_multi] FAILED: %s\n", e.what());
        return -1.0;
    } catch (...) {
        fprintf(stderr, "[train_multi] FAILED: unknown exception\n");
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
        for (auto& lc : ctx->layer_configs) { lc.nccl_comm = (void*)g_nccl_comm; lc.nccl_stream = (void*)g_nccl_stream; }
        for (auto& lc : ctx->mtp_layer_configs) { lc.nccl_comm = (void*)g_nccl_comm; lc.nccl_stream = (void*)g_nccl_stream; }
        return 0;
    }

    const char* rank_str = getenv("RANK");
    const char* world_str = getenv("WORLD_SIZE");
    if (!rank_str || !world_str) return -1;
    int rank = atoi(rank_str);
    int world_size = atoi(world_str);
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
    g_nccl_initialized = true;

    ctx->nccl_comm = comm;
    ctx->nccl_stream = nccl_stream;
    ctx->ep_rank = rank;
    ctx->ep_world_size = world_size;
    ctx->data_parallel = env_enabled("RUSTRAIN_DATA_PARALLEL");

    // Propagate to layer configs
    for (auto& lc : ctx->layer_configs) {
        lc.nccl_comm = (void*)comm;
        lc.nccl_stream = (void*)nccl_stream;
    }
    for (auto& lc : ctx->mtp_layer_configs) {
        lc.nccl_comm = (void*)comm;
        lc.nccl_stream = (void*)nccl_stream;
    }

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
    int current_device = g_cuda_device;
    cudaGetDevice(&current_device);
    ctx->cuda_device = current_device;
    // Propagate NCCL handles to all layer configs so moe_forward can access them
    for (auto& lc : ctx->layer_configs) {
        lc.nccl_comm = comm_ptr;
        lc.nccl_stream = stream_ptr;
    }
    for (auto& lc : ctx->mtp_layer_configs) {
        lc.nccl_comm = comm_ptr;
        lc.nccl_stream = stream_ptr;
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
                        a = at::randn({experts, rank, in_f}, opts) * 0.01;
                        b = at::zeros({experts, out_f, rank}, opts);
                    } else {
                        TORCH_CHECK(base->dim() == 2,
                            "dynamic LoRA projection must be a matrix: ", projection.name);
                        int64_t out_f = base->size(0), in_f = base->size(1);
                        a = at::randn({rank, in_f}, opts) * 0.01;
                        b = at::zeros({out_f, rank}, opts);
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
                pairs.emplace_back(std::move(a), std::move(b));
            }
            adapter.params[i] = std::move(pairs);
            adapter.adam_state[i] = std::move(adam_states);
        }
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
