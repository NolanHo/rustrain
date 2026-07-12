// qwen3_6_kernels.cpp — C++ FFI for Qwen3.6 native kernels
//
// Full C++ training: forward + loss + backward + Adam optimizer.
// LoRA A/B are created in C++ as at::Tensor (requires_grad=true).
// No tch-rs VarStore involved — gradients flow entirely within C++ autograd.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/variable.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <set>
#include <array>
#include <map>
#include <sstream>
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
    // ATen ops (keeps autograd graph) — TODO: re-enable CUDA kernel with autograd
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
        fprintf(stderr, "[proj_debug] full_attention: seq=%ld proj_chunk=%ld (chunked)\n", (long)seq, (long)proj_chunk);
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

            cudaDeviceSynchronize();
            c10::cuda::CUDACachingAllocator::emptyCache();
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
        fprintf(stderr, "[mem_debug] after rms_norm: used=%.1f GB, free=%.1f GB\n",
            (total2 - free2) / 1e9, free2 / 1e9);
    }

    // Release gate before RoPE to save 16GB
    auto gate_saved = gate;
    gate = at::Tensor();
    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();

    // Debug: memory after gate release
    {
        size_t free2, total2;
        cudaMemGetInfo(&free2, &total2);
        fprintf(stderr, "[mem_debug] after gate release: used=%.1f GB, free=%.1f GB\n",
            (total2 - free2) / 1e9, free2 / 1e9);
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
        cudaDeviceSynchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    // Restore gate after RoPE
    gate = gate_saved;

    int64_t n_rep = num_heads / num_kv_heads;
    auto k_expanded = k.repeat_interleave(n_rep, 1);
    k = at::Tensor();  // release old small K
    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
    auto v_expanded = v.repeat_interleave(n_rep, 1);
    v = at::Tensor();  // release old small V

    // Release intermediate tensors before SDPA
    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();

    double scale = 1.0 / std::sqrt((double)head_dim);

    // Use SDPA (Flash Attention) — O(seq) memory instead of O(seq²)
    // Build attention mask: causal + padding
    // SDPA accepts is_causal=true for causal, but we also need padding mask.
    // Combine: use a 2D bias [batch*num_heads, seq, seq] = causal_mask & key_padding_mask
    at::Tensor attn_mask;
    if (attention_mask.defined() && attention_mask.numel() > 0) {
        // key_padding_mask: [batch, seq] → [batch, 1, 1, seq]
        // Expand to [batch, num_heads, seq, seq] via broadcasting
        // mask=1 means "attend", mask=0 means "ignore"
        auto kpm = attention_mask.to(at::kBool);
        // Ensure kpm is [batch, seq] (squeeze extra dims from Rust side)
        while (kpm.dim() > 2) kpm = kpm.squeeze(0);
        kpm = kpm.unsqueeze(1).unsqueeze(1);  // [B, 1, 1, S]
        // For SDPA, attn_mask should be additive bias: 0 for attend, -inf for ignore
        auto additive_mask = at::zeros({batch, 1, 1, seq}, at::TensorOptions().dtype(q.scalar_type()).device(q.device()));
        additive_mask = additive_mask.masked_fill(kpm.logical_not(), -std::numeric_limits<float>::infinity());
        auto attn_out = at::scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            additive_mask,
            0.0,
            true
        );
        return attn_out.transpose(1, 2).reshape({batch, seq, qkv_dim}).matmul(o_proj.t());
    } else {
        auto attn_out = at::scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            c10::nullopt,
            0.0,
            true
        );
        auto result = attn_out.transpose(1, 2).reshape({batch, seq, qkv_dim}).matmul(o_proj.t());
        k_expanded = at::Tensor();
        v_expanded = at::Tensor();
        cudaDeviceSynchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
        result = result * at::sigmoid(gate).to(result.scalar_type());
        gate = at::Tensor();
        cudaDeviceSynchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
        return result;
    }
}

// ──────────────────────────────────────────────────────────────────────
// Linear attention (Gated Delta Rule — matrix formulation)
// ──────────────────────────────────────────────────────────────────────

// Forward declaration for CUDA kernel (defined in delta_rule.cu)
extern "C" void cuda_gated_delta_rule(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    float* state, float* out, float* delta_buf,
    int BH, int seq_len, int key_dim, int val_dim
);

extern "C" void cuda_gated_delta_rule_backward(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    const float* final_state, const float* delta_buf,
    const float* grad_out,
    float* grad_q, float* grad_k, float* grad_v,
    float* grad_g, float* grad_beta,
    int BH, int seq_len, int key_dim, int val_dim
);

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

    if (seq_chunk > 0 && seq > seq_chunk) {
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
            cuda_gated_delta_rule(
                q_contig.data_ptr<float>(),
                k_contig.data_ptr<float>(),
                v_contig.data_ptr<float>(),
                g_contig.data_ptr<float>(),
                beta_contig.data_ptr<float>(),
                state_contig.data_ptr<float>(),
                outs.data_ptr<float>(),
                delta_buf.data_ptr<float>(),
                (int)BH, (int)chunk_len, (int)key_dim, (int)val_dim
            );
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
    auto state = at::zeros({BH, key_dim, val_dim}, q_t.options());  // [B*H, D_k, D_v]

    // Prepare contiguous FP32 tensors for CUDA kernel
    auto q_contig = q_t.reshape({BH, seq, key_dim}).contiguous().to(at::kFloat);
    auto k_contig = k_t.reshape({BH, seq, key_dim}).contiguous().to(at::kFloat);
    auto v_contig = v_t.reshape({BH, seq, val_dim}).contiguous().to(at::kFloat);
    auto g_contig = g_exp.reshape({BH, seq}).contiguous().to(at::kFloat);
    auto beta_contig = beta_t.reshape({BH, seq}).contiguous().to(at::kFloat);
    auto state_contig = state.contiguous();
    auto outs = at::empty({BH, seq, val_dim}, q_t.options());
    auto delta_buf = at::empty({BH, seq, val_dim}, q_t.options());

    // Launch CUDA kernel — single launch replaces seq×3 bmm calls
    cuda_gated_delta_rule(
        q_contig.data_ptr<float>(),
        k_contig.data_ptr<float>(),
        v_contig.data_ptr<float>(),
        g_contig.data_ptr<float>(),
        beta_contig.data_ptr<float>(),
        state_contig.data_ptr<float>(),
        outs.data_ptr<float>(),
        delta_buf.data_ptr<float>(),
        (int)BH, (int)seq, (int)key_dim, (int)val_dim
    );

    // Reshape: [B*H, S, D_v] → [B, H, S, D_v] → [B, S, H, D_v]
    auto core_out = outs.reshape({batch, num_v_heads, seq, val_dim})
                         .transpose(1, 2).to(compute_type);

    auto core_flat = core_out.reshape({-1, val_dim});
    auto z_flat = z.reshape({-1, val_dim});
    auto variance = core_flat.to(at::kFloat).pow(2).mean(-1, true);
    auto normed = (core_flat.to(at::kFloat) * (variance + rms_eps).rsqrt() * norm_w.to(at::kFloat)).to(core_flat.scalar_type());
    auto gated = (normed * at::silu(z_flat.to(at::kFloat)).to(normed.scalar_type())).view({batch, seq, num_v_heads * val_dim});
    auto result = at::matmul(gated, out_proj.t());
    return result;
}

// ──────────────────────────────────────────────────────────────────────
// Dense MLP (SwiGLU) — for non-MoE models (Qwen3.5 dense)
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

// ──────────────────────────────────────────────────────────────────────
// MoE
// ──────────────────────────────────────────────────────────────────────

static at::Tensor moe_forward(
    void* nccl_comm_v, void* nccl_stream_v,
    const at::Tensor& hidden,
    const at::Tensor& gate_w, const at::Tensor& shared_expert_gate_w,
    const at::Tensor& shared_gate_proj, const at::Tensor& shared_up_proj, const at::Tensor& shared_down_proj,
    const at::Tensor& experts_gate_up, const at::Tensor& experts_down,
    int64_t num_experts, int64_t top_k, int64_t intermediate,
    bool norm_topk_prob, int64_t expert_start, int64_t expert_count,
    at::ScalarType compute_type
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
        fprintf(stderr, "  [moe] router_logits: mean=%.6f std=%.6f max=%.6f\n", rl_f.mean().item<float>(), rl_f.std().item<float>(), rl_f.abs().max().item<float>());
        fprintf(stderr, "  [moe] topk_weights: mean=%.6f std=%.6f max=%.6f\n", rw_f.mean().item<float>(), rw_f.std().item<float>(), rw_f.abs().max().item<float>());
        fprintf(stderr, "  [moe] experts_gate_up[0]: mean=%.6f std=%.6f max=%.6f\n", egu_f.mean().item<float>(), egu_f.std().item<float>(), egu_f.abs().max().item<float>());
        fprintf(stderr, "  [moe] experts_down[0]: mean=%.6f std=%.6f max=%.6f\n", ed_f.mean().item<float>(), ed_f.std().item<float>(), ed_f.abs().max().item<float>());
        auto sg_f = shared_gate_proj.to(at::kFloat);
        auto sd_f = shared_down_proj.to(at::kFloat);
        fprintf(stderr, "  [moe] shared_gate: mean=%.6f std=%.6f max=%.6f\n", sg_f.mean().item<float>(), sg_f.std().item<float>(), sg_f.abs().max().item<float>());
        fprintf(stderr, "  [moe] shared_down: mean=%.6f std=%.6f max=%.6f\n", sd_f.mean().item<float>(), sd_f.std().item<float>(), sd_f.abs().max().item<float>());
        auto seg_f = at::sigmoid(at::matmul(flat, shared_expert_gate_w.t())).to(at::kFloat);
        fprintf(stderr, "  [moe] shared_gate_sig: mean=%.6f std=%.6f max=%.6f\n", seg_f.mean().item<float>(), seg_f.std().item<float>(), seg_f.abs().max().item<float>());
    }

    for (int64_t kk = 0; kk < top_k; kk++) {
        auto expert_indices = topk_indices.select(-1, kk);
        auto expert_weights = topk_weights.select(-1, kk);
        for (int64_t e_local = 0; e_local < expert_count; e_local++) {
            int64_t e_global = expert_start + e_local;
            auto mask = expert_indices.eq(e_global);
            // Use nonzero().size(0) instead of mask.sum().item<double>()
            // This avoids GPU→CPU synchronization — size(0) is metadata only
            auto token_indices = mask.nonzero().squeeze(-1);
            if (token_indices.size(0) == 0) continue;
            auto selected = flat.index_select(0, token_indices);
            auto egu = experts_gate_up.select(0, e_local);
            auto ed = experts_down.select(0, e_local);
            auto gu = at::matmul(selected, egu.t());
            auto gate_part = gu.narrow(-1, 0, intermediate);
            auto up_part = gu.narrow(-1, intermediate, intermediate);
            auto expert_out = at::matmul(fused_swiglu_op(gate_part, up_part, 0.0), ed.t());
            auto weights = expert_weights.index_select(0, token_indices).unsqueeze(-1);
            routed_output = routed_output.index_add_(0, token_indices, expert_out * weights);
        }
    }

    // EP all-reduce: sum routed_output across all EP ranks
    // Each rank only has contributions from its local experts.
    // The non-local expert tokens are still zero — after all-reduce, every
    // rank gets the complete routed_output.
    // EP all-reduce: sum routed_output across all EP ranks
    // CRITICAL: must use out-of-place NCCL (separate input/output buffers).
    // In-place NCCL (ptr, ptr) modifies autograd-tracked tensor via raw CUDA call,
    // bypassing PyTorch's version counter. During backward, autograd accesses
    // corrupted saved tensors → cudaErrorIllegalAddress.
    if (nccl_comm_v) {
        auto nccl_comm = reinterpret_cast<ncclComm_t>(nccl_comm_v);
        int dev = routed_output.device().index();
        cudaSetDevice(dev);
        auto compute_stream = c10::cuda::getCurrentCUDAStream(dev).stream();
        auto ro = routed_output.is_contiguous() ? routed_output : routed_output.contiguous();
        // Allocate separate output buffer — NOT connected to autograd graph
        auto reduced = at::empty_like(ro);
        ncclResult_t nccl_err = ncclAllReduce(
            ro.data_ptr(),
            reduced.data_ptr(),   // output: separate buffer
            ro.numel(),
            ncclBfloat16,
            ncclSum,
            nccl_comm,
            compute_stream
        );
        if (nccl_err != ncclSuccess) {
            fprintf(stderr, "[ep_debug] ncclAllReduce FAILED: %d (%s)\n", nccl_err, ncclGetErrorString(nccl_err));
        }
        // CRITICAL: sync before ro goes out of scope.
        // NCCL is async on compute stream. If ro (input buffer) is freed by
        // PyTorch's caching allocator when routed_output = reduced replaces it,
        // NCCL may still be reading from ro's memory → cudaErrorIllegalAddress.
        // This sync only blocks until NCCL completes, not the entire stream.
        cudaStreamSynchronize(compute_stream);
        // Replace routed_output with reduced version.
        routed_output = reduced;
    }

    // Shared expert (same as before, with fused SwiGLU)
    auto shared_gate = at::matmul(flat, shared_gate_proj.t());
    auto shared_up = at::matmul(flat, shared_up_proj.t());
    auto shared_out = at::matmul(fused_swiglu_op(shared_gate, shared_up, 0.0), shared_down_proj.t());
    auto seg = at::sigmoid(at::matmul(flat, shared_expert_gate_w.t())).to(compute_type);
    shared_out = (shared_out * seg).to(compute_type);

    // Debug: dump routed_output AFTER loop
    if (getenv("QWEN36_DUMP_MOE")) {
        auto ro_f = routed_output.to(at::kFloat);
        auto so_f = shared_out.to(at::kFloat);
        fprintf(stderr, "  [moe] routed_output (after loop): mean=%.6f std=%.6f max=%.6f\n", ro_f.mean().item<float>(), ro_f.std().item<float>(), ro_f.abs().max().item<float>());
        fprintf(stderr, "  [moe] shared_out: mean=%.6f std=%.6f max=%.6f\n", so_f.mean().item<float>(), so_f.std().item<float>(), so_f.abs().max().item<float>());
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

static at::Tensor forward_single_layer(
    TrainingContext* ctx, const at::Tensor& hidden, at::Tensor** w, const LayerConfig* cfg,
    int64_t layer_idx, at::ScalarType kind,
    const at::Tensor& attention_mask
) {
    auto input_norm = *w[0];
    auto post_norm = *w[1];
    auto attn_input = rms_norm(hidden, input_norm, cfg->rms_eps);
    bool is_moe = (cfg->num_experts > 0);

    at::Tensor attn_output;
    if (cfg->layer_type == 0) {
        // Full attention
        auto q_proj = *w[2], q_norm = *w[3], k_proj = *w[4], k_norm = *w[5], v_proj = *w[6], o_proj = *w[7];
        // Apply multi-LoRA (handles both adapters and legacy arrays)
        q_proj = apply_multi_lora(ctx, layer_idx, 0, q_proj);
        k_proj = apply_multi_lora(ctx, layer_idx, 1, k_proj);
        v_proj = apply_multi_lora(ctx, layer_idx, 2, v_proj);
        o_proj = apply_multi_lora(ctx, layer_idx, 3, o_proj);
        attn_output = full_attention(attn_input, q_proj, q_norm, k_proj, k_norm, v_proj, o_proj,
            cfg->num_heads, cfg->num_kv_heads, cfg->head_dim,
            cfg->partial_rotary_factor, cfg->rope_theta, cfg->rms_eps, kind,
            attention_mask);
        auto post_attn = rms_norm(hidden + attn_output, post_norm, cfg->rms_eps);
        if (is_moe) {
            auto mlp_out = moe_forward(cfg->nccl_comm, cfg->nccl_stream, post_attn,
                *w[8], *w[9], *w[10], *w[11], *w[12], *w[13], *w[14],
                cfg->num_experts, cfg->top_k, cfg->moe_intermediate,
                cfg->norm_topk_prob != 0, cfg->expert_start, cfg->expert_count, kind);
            // Debug: dump sub-component stats for last layer
            if (getenv("QWEN36_DUMP_LAST_LAYER")) {
                auto af = attn_output.to(at::kFloat);
                auto mf = mlp_out.to(at::kFloat);
                auto pf = post_attn.to(at::kFloat);
                auto hf = hidden.to(at::kFloat);
                fprintf(stderr, "  [last_layer] hidden_in:  mean=%.6f std=%.6f max=%.6f\n", hf.mean().item<float>(), hf.std().item<float>(), hf.abs().max().item<float>());
                fprintf(stderr, "  [last_layer] attn_out:   mean=%.6f std=%.6f max=%.6f\n", af.mean().item<float>(), af.std().item<float>(), af.abs().max().item<float>());
                fprintf(stderr, "  [last_layer] post_attn:   mean=%.6f std=%.6f max=%.6f\n", pf.mean().item<float>(), pf.std().item<float>(), pf.abs().max().item<float>());
                fprintf(stderr, "  [last_layer] mlp_out:     mean=%.6f std=%.6f max=%.6f\n", mf.mean().item<float>(), mf.std().item<float>(), mf.abs().max().item<float>());
            }
            return hidden + attn_output + mlp_out;
        } else {
            auto mlp_out = dense_mlp_forward(post_attn, *w[8], *w[9], *w[10], kind);
            return hidden + attn_output + mlp_out;
        }
    } else {
        // Linear attention
        auto in_proj_qkv = *w[2], in_proj_z = *w[3], in_proj_a = *w[4], in_proj_b = *w[5];
        auto a_log = *w[6], dt_bias = *w[7], conv1d_w = *w[8], norm_w = *w[9], out_proj = *w[10];
        in_proj_qkv = apply_multi_lora(ctx, layer_idx, 0, in_proj_qkv);
        in_proj_z = apply_multi_lora(ctx, layer_idx, 1, in_proj_z);
        out_proj = apply_multi_lora(ctx, layer_idx, 2, out_proj);
        attn_output = linear_attention(attn_input, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
            a_log, dt_bias, conv1d_w, norm_w, out_proj,
            cfg->num_k_heads, cfg->key_dim, cfg->num_v_heads, cfg->val_dim,
            cfg->conv_kernel, cfg->rms_eps, kind);
        auto post_attn = rms_norm(hidden + attn_output, post_norm, cfg->rms_eps);
        if (is_moe) {
            auto mlp_out = moe_forward(cfg->nccl_comm, cfg->nccl_stream, post_attn,
                *w[11], *w[12], *w[13], *w[14], *w[15], *w[16], *w[17],
                cfg->num_experts, cfg->top_k, cfg->moe_intermediate,
                cfg->norm_topk_prob != 0, cfg->expert_start, cfg->expert_count, kind);
            return hidden + attn_output + mlp_out;
        } else {
            auto mlp_out = dense_mlp_forward(post_attn, *w[11], *w[12], *w[13], kind);
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
    // Legacy single-LoRA (backward compat)
    std::vector<at::Tensor> lora_a;
    std::vector<at::Tensor> lora_b;
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
    at::Tensor *mtp_fc, *mtp_pre_fc_norm_emb, *mtp_pre_fc_norm_hidden, *mtp_norm;
    std::vector<at::Tensor*> mtp_layer_weights;
    std::vector<LayerConfig> mtp_layer_configs;

    // Gradient checkpointing
    bool use_checkpoint;
    int64_t group_size;
    // Group checkpoint storage for manual sequential backward
    std::vector<at::Tensor> group_inputs;
    std::vector<at::Tensor> group_outputs;

    // NCCL for Expert Parallel all-reduce (nullptr if single-GPU)
    ncclComm_t nccl_comm = nullptr;
    cudaStream_t nccl_stream = nullptr;
    int ep_world_size = 1;
    int ep_rank = 0;
// ──────────────────────────────────────────────────────────────────────
};

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
        int64_t num_pairs = (ctx->layer_configs[layer_idx].layer_type == 0) ? 4 : 3;
        for (int64_t pair_idx = 0; pair_idx < num_pairs; pair_idx++) {
            std::vector<at::Tensor> a_list, b_list;
            for (auto& adapter : ctx->adapters) {
                if (!adapter.target_layers.empty() && adapter.target_layers.find(layer_idx) == adapter.target_layers.end())
                    continue;
                auto it = adapter.params.find(layer_idx);
                if (it == adapter.params.end()) continue;
                if (pair_idx >= (int64_t)it->second.size()) continue;
                auto& [a, b] = it->second[pair_idx];
                double scaling = adapter.alpha / (double)adapter.rank;
                b_list.push_back(b * scaling);
                a_list.push_back(a);
            }
            if (a_list.empty()) {
                if (!ctx->lora_a.empty() && layer_idx < (int64_t)ctx->lora_layer_offset.size()) {
                    int64_t la_offset = ctx->lora_layer_offset[layer_idx];
                    if (la_offset + pair_idx < (int64_t)ctx->lora_a.size()) {
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
                entries.push_back({layer_idx * 10 + pair_idx, a_concat, b_concat});
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

__attribute__((noinline, visibility("default")))
at::Tensor apply_multi_lora(
    TrainingContext* ctx, int64_t layer_idx, int64_t pair_idx,
    const at::Tensor& base_weight
) {
    auto it = ctx->lora_cache.find(layer_idx * 10 + pair_idx);
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
    int64_t lora_count = (cfg.layer_type == 0) ? 4 : 3;
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
            if (la[2]) op = lora_delta(op, *la[2], *lb[2], ctx->lora_scaling);
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
        return moe_forward(cfg.nccl_comm, cfg.nccl_stream, post_attn,
            *ctx->weight_ptrs[w_offset+mlp_start], *ctx->weight_ptrs[w_offset+mlp_start+1],
            *ctx->weight_ptrs[w_offset+mlp_start+2], *ctx->weight_ptrs[w_offset+mlp_start+3],
            *ctx->weight_ptrs[w_offset+mlp_start+4], *ctx->weight_ptrs[w_offset+mlp_start+5],
            *ctx->weight_ptrs[w_offset+mlp_start+6],
            cfg.num_experts, cfg.top_k, cfg.moe_intermediate,
            cfg.norm_topk_prob != 0, cfg.expert_start, cfg.expert_count, kind);
    } else {
        return dense_mlp_forward(post_attn,
            *ctx->weight_ptrs[w_offset+mlp_start], *ctx->weight_ptrs[w_offset+mlp_start+1],
            *ctx->weight_ptrs[w_offset+mlp_start+2], kind);
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
        auto output = is_attn ? compute_attn_only(tc, input, layer, tc->compute_type)
                              : compute_mlp_only(tc, input, layer, tc->compute_type);

        // Collect all tensors to compute gradients for: input + LoRA params for this layer
        // This way grad() accumulates gradients into LoRA params (leaf nodes) too.
        int64_t lora_count = (tc->layer_configs[layer].layer_type == 0) ? 4 : 3;
        int64_t la_offset = tc->lora_layer_offset[layer];
        bool has_lora = (la_offset + lora_count) <= (int64_t)tc->lora_a.size();

        std::vector<at::Tensor> grad_inputs = {input};
        if (has_lora) {
            for (int64_t k = 0; k < lora_count; k++) {
                grad_inputs.push_back(tc->lora_a[la_offset + k]);
                grad_inputs.push_back(tc->lora_b[la_offset + k]);
            }
        }

        auto grads = torch::autograd::grad(
            {output}, grad_inputs, {grad_output[0]},
            /*retain_graph=*/false, /*create_graph=*/false,
            /*allow_unused=*/true
        );

        // Manually accumulate LoRA param gradients
        if (has_lora) {
            int64_t gi = 1;  // skip input grad (index 0)
            for (int64_t k = 0; k < lora_count; k++) {
                if (grads[gi].defined()) {
                    auto& param_a = tc->lora_a[la_offset + k];
                    if (param_a.grad().defined())
                        param_a.grad().add_(grads[gi]);
                    else
                        param_a.mutable_grad() = grads[gi].clone();
                }
                gi++;
                if (grads[gi].defined()) {
                    auto& param_b = tc->lora_b[la_offset + k];
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

// Forward pass (no checkpointing)
static at::Tensor forward_full(
    TrainingContext* ctx,
    const at::Tensor& input_ids
) {
    precompute_lora_cache(ctx);
    auto kind = ctx->compute_type;
    auto embed = *ctx->embed_ptr[0];
    auto final_norm = *ctx->final_norm_ptr[0];

    at::AutoGradMode guard(true);
    at::Tensor hidden = at::embedding(embed, input_ids);

    // Debug: dump embedding output stats
    if (getenv("QWEN36_DUMP_LAYERS")) {
        auto h_f = hidden.to(at::kFloat);
        fprintf(stderr, "Layer  0 (embed): mean=%.6f std=%.6f max_abs=%.6f\n",
            h_f.mean().item<float>(), h_f.std().item<float>(), h_f.abs().max().item<float>());
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
        int64_t lora_count = (ctx->layer_configs[i].layer_type == 0) ? 4 : 3;
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
            kind, ctx->attention_mask);

        // Debug: dump per-layer hidden state stats (matching HF output_hidden_states)
        if (getenv("QWEN36_DUMP_LAYERS")) {
            auto h_f = hidden.to(at::kFloat);
            fprintf(stderr, "Layer %2ld: mean=%.6f std=%.6f max_abs=%.6f\n",
                i, h_f.mean().item<float>(), h_f.std().item<float>(), h_f.abs().max().item<float>());
        }

        // Sync + release CUDA allocator cache after each layer in no-grad forward.
        // Without sync, pending CUDA ops hold references to intermediates,
        // preventing emptyCache from freeing them.
        cudaDeviceSynchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
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

        int64_t lora_count = (ctx->layer_configs[i].layer_type == 0) ? 4 : 3;
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
            kind, ctx->attention_mask);
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
        int64_t lora_count = (tc->layer_configs[layer_idx].layer_type == 0) ? 4 : 3;
        int64_t la_offset = tc->lora_layer_offset[layer_idx];
        bool has_lora = (la_offset + lora_count) <= (int64_t)tc->lora_a.size();
        std::vector<at::Tensor*> la(lora_count, nullptr), lb(lora_count, nullptr);
        if (has_lora) for (int64_t k = 0; k < lora_count; k++) {
            la[k] = &tc->lora_a[la_offset + k]; lb[k] = &tc->lora_b[la_offset + k];
        }
        return forward_single_layer(tc, input, layer_w.data(), &tc->layer_configs[layer_idx],
            layer_idx, kind, tc->attention_mask);
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
    auto embed = *ctx->embed_ptr[0];
    at::Tensor hidden = at::embedding(embed, input_ids);
    hidden = hidden.detach().set_requires_grad(true);

    for (int64_t i = 0; i < ctx->num_layers; i++) {
        hidden = FusedLayerFunction::apply(
            hidden,
            (int64_t)(uintptr_t)ctx,
            i
        );
        // Release allocator cache between layers to prevent accumulation
        c10::cuda::CUDACachingAllocator::emptyCache();
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
    precompute_lora_cache(ctx);
    auto embed = *ctx->embed_ptr[0];
    at::Tensor hidden = at::embedding(embed, input_ids);

    bool use_subckpt = getenv("QWEN36_SUBCKPT");

    if (use_subckpt) {
        at::AutoGradMode restore(true);
        hidden = hidden.detach().set_requires_grad(true);
        for (int64_t i = 0; i < ctx->num_layers; i++) {
            hidden = forward_single_layer_subckpt(ctx, hidden, i);
        }
        return hidden;
    }

    // Group-level manual checkpointing — no autograd::Function, no graph edges
    int64_t gs = ctx->group_size;
    if (gs < 1) gs = 1;

    // Forward in no-grad mode
    at::AutoGradMode no_grad(false);
    hidden = hidden.detach();

    ctx->group_inputs.clear();
    bool offload = getenv("QWEN36_OFFLOAD_ACTIVATIONS");
    int64_t num_groups = (ctx->num_layers + gs - 1) / gs;

    for (int64_t g = 0; g < num_groups; g++) {
        int64_t start = g * gs;
        int64_t end = std::min(start + gs, ctx->num_layers);

        // Save this group's input (offload to CPU if enabled)
        if (offload && g < num_groups - 1) {
            ctx->group_inputs.push_back(
                hidden.to(at::TensorOptions().dtype(hidden.scalar_type()).device(at::kCPU).pinned_memory(true))
            );
        } else {
            ctx->group_inputs.push_back(hidden.clone());
        }

        hidden = forward_layer_group(ctx, hidden, start, end);

        // Debug: GPU memory after each group
        {
            // Force release cached memory from no-grad intermediates
            c10::cuda::CUDACachingAllocator::emptyCache();
            size_t free, total;
            cudaMemGetInfo(&free, &total);
            fprintf(stderr, "[mem_debug] after group %ld/%ld: used=%.1f GB, free=%.1f GB\n",
                (long)g, (long)num_groups, (total - free) / 1e9, free / 1e9);
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
    int64_t gs = ctx->group_size;
    if (gs < 1) gs = 1;

    int64_t num_groups = (ctx->num_layers + gs - 1) / gs;
    at::Tensor grad = hidden_grad;
    at::AutoGradMode grad_mode(true);

    for (int64_t g = num_groups - 1; g >= 0; g--) {
        int64_t start = g * gs;
        int64_t end = std::min(start + gs, ctx->num_layers);

        // Restore input from saved (CPU if offloaded)
        auto input = ctx->group_inputs[g].to(hidden_grad.device()).detach().set_requires_grad(true);

        // Recompute forward with grad for this group only
        auto output = forward_layer_group(ctx, input, start, end);

        // Backprop through this group using grad() instead of backward().
        // grad() only computes gradients for specified inputs — faster than
        // backward() which traverses all leaf nodes.
        // LoRA params are shared across groups, so we accumulate their gradients.
        std::vector<at::Tensor> grad_inputs = {input};
        // Add LoRA params for this group's layers
        for (int64_t l = start; l < end; l++) {
            int64_t lora_count = (ctx->layer_configs[l].layer_type == 0) ? 4 : 3;
            int64_t la_offset = ctx->lora_layer_offset[l];
            bool has_lora = (la_offset + lora_count) <= (int64_t)ctx->lora_a.size();
            if (has_lora) {
                for (int64_t k = 0; k < lora_count; k++) {
                    grad_inputs.push_back(ctx->lora_a[la_offset + k]);
                    grad_inputs.push_back(ctx->lora_b[la_offset + k]);
                }
            }
        }

        auto grads = torch::autograd::grad(
            {output}, grad_inputs, {grad},
            /*retain_graph=*/false, /*create_graph=*/false,
            /*allow_unused=*/true
        );

        // Manually accumulate LoRA param gradients
        int64_t gi = 1;  // skip input grad (index 0)
        for (int64_t l = start; l < end; l++) {
            int64_t lora_count = (ctx->layer_configs[l].layer_type == 0) ? 4 : 3;
            int64_t la_offset = ctx->lora_layer_offset[l];
            bool has_lora = (la_offset + lora_count) <= (int64_t)ctx->lora_a.size();
            if (has_lora) {
                for (int64_t k = 0; k < lora_count; k++) {
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

        // Force release cached intermediates from this group's recompute+backward
        c10::cuda::CUDACachingAllocator::emptyCache();

        // Gradient for this group's input = gradient for next group's output
        grad = grads[0];

        // Free saved input to release memory
        ctx->group_inputs[g] = at::Tensor();
    }
}

// Cross-entropy loss with response-only masking — chunked with detach.
// Detach hidden_normed so CE backward doesn't traverse main model graph.
// Accumulate hidden_normed gradient, then backprop to hidden separately.
static at::Tensor compute_loss(
    TrainingContext* ctx,
    const at::Tensor& hidden,
    const at::Tensor& input_ids,
    const at::Tensor& target_mask,
    int64_t vocab_size
) {
    auto final_norm = *ctx->final_norm_ptr[0];
    auto lm_head = *ctx->lm_head_ptr[0];

    // Detach hidden for CE computation — CE backward won't touch main model graph.
    // We accumulate gradient into hidden_normed, then manually backprop to hidden.
    auto hidden_detached = hidden.detach();

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
    int64_t chunk_size = 16384;  // [16384, vocab] = 15GB per chunk
    int64_t num_chunks = (total_tokens + chunk_size - 1) / chunk_size;

    auto total_count = shifted_mask.sum().clamp_min(1.0);
    auto hidden_flat = shifted_hidden.reshape({-1, hidden_normed.size(2)});

    double total_loss_val = 0.0;
    auto total_count_val = total_count.item<double>();

    for (int64_t c = 0; c < num_chunks; c++) {
        int64_t start = c * chunk_size;
        int64_t end = std::min(start + chunk_size, total_tokens);
        int64_t n = end - start;

        auto chunk_hidden = hidden_flat.narrow(0, start, n);
        auto chunk_logits = at::matmul(chunk_hidden, lm_head.t());
        auto chunk_targets = shifted_targets.narrow(0, start, n);
        auto chunk_mask = shifted_mask.narrow(0, start, n);

        auto per_token_loss = at::cross_entropy_loss(
            chunk_logits.to(at::kFloat), chunk_targets,
            at::Tensor(), at::Reduction::None, -100, 0.0
        );
        auto masked_loss = per_token_loss * chunk_mask.to(at::kFloat);
        auto chunk_loss = masked_loss.sum();

        // Backward this chunk — only traverses CE graph (detached from main model)
        // retain_graph=true: needed because all chunks share hidden_normed graph
        // (but graph is tiny — just matmul + CE, not connected to main model)
        torch::autograd::backward({chunk_loss}, {},
            /*retain_graph=*/true, /*create_graph=*/false);

        total_loss_val += chunk_loss.item<double>();

        // Periodically release CUDA allocator cache to prevent accumulation
        // of freed chunk intermediates (logits, log_softmax, etc.)
        if ((c + 1) % 8 == 0) {
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
    }

    // Backprop hidden_normed gradient to hidden via rms_norm.
    // hidden_normed was computed in no-grad mode (detached from hidden_detached).
    // Recompute rms_norm with grad tracking to get hidden's gradient.
    if (hidden_normed.grad().defined()) {
        hidden.set_requires_grad(true);
        auto hidden_normed_recompute = rms_norm(hidden, final_norm, ctx->rms_eps);
        hidden_normed_recompute.backward(hidden_normed.grad());
        // hidden.grad() now has the CE gradient contribution
    }

    // Release all CE intermediate tensors at once
    c10::cuda::CUDACachingAllocator::emptyCache();

    return at::tensor({total_loss_val / total_count_val},
        at::TensorOptions().dtype(at::kFloat).device(hidden.device()));
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
    const at::Tensor& target_mask
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
    int64_t chunk_size = 16384;  // [16384, vocab] = 15GB per chunk  // larger chunks = fewer backward passes
    int64_t num_chunks = (total_tokens + chunk_size - 1) / chunk_size;

    auto total_loss = at::zeros({1}, at::TensorOptions().dtype(at::kFloat).device(mtp_hidden.device()));
    auto total_count = shifted_mask.sum().clamp_min(1.0);

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
        total_loss += masked_loss.sum();
    }

    return (total_loss / total_count) * 0.5;
}

// ──────────────────────────────────────────────────────────────────────
// C FFI
// ──────────────────────────────────────────────────────────────────────

extern "C" {

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
    const int64_t* target_layers, int64_t num_target_layers
) {
    try {
        auto* ctx = new TrainingContext();
        ctx->compute_type = static_cast<at::ScalarType>(compute_type);
        ctx->lr = lr; ctx->beta1 = beta1; ctx->beta2 = beta2; ctx->eps = eps;
        ctx->vocab_size = vocab_size; ctx->rms_eps = rms_eps;
        ctx->step_count = 0; ctx->lora_scaling = lora_scaling;
        ctx->num_layers = num_layers;
        ctx->use_checkpoint = false; ctx->group_size = 4;

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
        if (target_layers && num_target_layers > 0) {
            for (int64_t j = 0; j < num_target_layers; j++)
                target_set.insert(target_layers[j]);
        }

        // Create LoRA parameters for target layers only
        int64_t offset = 0;
        auto kind = ctx->compute_type;
        for (int64_t i = 0; i < num_layers; i++) {
            int64_t lora_count = (ctx->layer_configs[i].layer_type == 0) ? 4 : 3;
            ctx->lora_layer_offset.push_back(offset);

            if (target_set.find(i) == target_set.end()) {
                // Not a target layer — no LoRA params, offset stays same
                continue;
            }

            // Get base weight shapes from the weight pointers
            int64_t w_offset = 0;
            for (int64_t j = 0; j < i; j++)
                w_offset += weight_count_for_layer(ctx->layer_configs[j]);

            if (ctx->layer_configs[i].layer_type == 0) {
                // Full attention: q_proj, k_proj, v_proj, o_proj
                int64_t proj_indices[] = {2, 4, 6, 7};  // q, k, v, o
                for (int k = 0; k < 4; k++) {
                    auto* base = ctx->weight_ptrs[w_offset + proj_indices[k]];
                    int64_t out_f = base->size(0), in_f = base->size(1);
                    auto a = at::randn({lora_rank, in_f}, at::TensorOptions().dtype(ctx->compute_type).device(base->device())) * 0.01;
                    auto b = at::zeros({out_f, lora_rank}, at::TensorOptions().dtype(ctx->compute_type).device(base->device()));
                    a.set_requires_grad(true);
                    b.set_requires_grad(true);
                    ctx->lora_a.push_back(std::move(a));
                    ctx->lora_b.push_back(std::move(b));
                    ctx->lora_names.push_back("lora_a_" + std::to_string(i) + "_" + std::to_string(k));
                    ctx->lora_names.push_back("lora_b_" + std::to_string(i) + "_" + std::to_string(k));
                }
            } else {
                // Linear attention: in_proj_qkv, in_proj_z, out_proj
                int64_t proj_indices[] = {2, 3, 10};  // qkv, z, out
                for (int k = 0; k < 3; k++) {
                    auto* base = ctx->weight_ptrs[w_offset + proj_indices[k]];
                    int64_t out_f = base->size(0), in_f = base->size(1);
                    auto a = at::randn({lora_rank, in_f}, at::TensorOptions().dtype(ctx->compute_type).device(base->device())) * 0.01;
                    auto b = at::zeros({out_f, lora_rank}, at::TensorOptions().dtype(ctx->compute_type).device(base->device()));
                    a.set_requires_grad(true);
                    b.set_requires_grad(true);
                    ctx->lora_a.push_back(std::move(a));
                    ctx->lora_b.push_back(std::move(b));
                    ctx->lora_names.push_back("lora_a_" + std::to_string(i) + "_" + std::to_string(k));
                    ctx->lora_names.push_back("lora_b_" + std::to_string(i) + "_" + std::to_string(k));
                }
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
        fprintf(stderr, "[q36_create_ctx] FAILED: %s\n", e.what());
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

// Single training step: forward + loss + backward + Adam update
// Returns loss value, or -1 on error.
__attribute__((visibility("default"))) double qwen36_train_step(
    void* ctx_ptr,
    void* input_ids_ptr,
    void* target_mask_ptr,
    void* attention_mask_ptr
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        // Set CUDA device for EP — tensors and NCCL comm are on local_rank's GPU
        if (ctx->nccl_comm) {
            cudaSetDevice(ctx->ep_rank);
        }
        auto& input_ids = *reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto& target_mask = *reinterpret_cast<at::Tensor*>(target_mask_ptr);
        if (attention_mask_ptr) {
            ctx->attention_mask = *reinterpret_cast<at::Tensor*>(attention_mask_ptr);
        }

        // Forward: checkpoint (default) or fused layer (QWEN36_FUSED_LAYER=1)
        bool use_fused = getenv("QWEN36_FUSED_LAYER");
        auto hidden = use_fused
            ? forward_full_fused(ctx, input_ids)
            : ctx->use_checkpoint
                ? forward_full_checkpoint(ctx, input_ids)
                : forward_full(ctx, input_ids);

        // Debug: GPU memory after forward
        {
            size_t free, total;
            cudaMemGetInfo(&free, &total);
            fprintf(stderr, "[mem_debug] after forward: used=%.1f GB, free=%.1f GB\n",
                (total - free) / 1e9, free / 1e9);
        }

        // Main loss — compute_loss does chunked CE with immediate backward per chunk.
        // This avoids accumulating 250 chunks of [512, vocab] logits in autograd graph.
        auto loss = compute_loss(ctx, hidden, input_ids, target_mask, ctx->vocab_size);

        // compute_loss did chunked CE backward on detached hidden,
        // accumulated gradient into hidden.grad().
        // Now trigger main model backward using hidden's gradient.
        double loss_val = loss.item<double>();

        // Debug: GPU memory after CE backward
        {
            size_t free, total;
            cudaMemGetInfo(&free, &total);
            fprintf(stderr, "[mem_debug] after CE backward: used=%.1f GB, free=%.1f GB\n",
                (total - free) / 1e9, free / 1e9);
        }

        // Release CE's retained graph (hidden_normed etc.) before backward.
        // CE gradient is already accumulated into hidden.grad().
        // hidden.detach() breaks the autograd graph from CE.
        hidden = hidden.detach();
        c10::cuda::CUDACachingAllocator::emptyCache();

        // Debug: after releasing CE graph
        {
            size_t free, total;
            cudaMemGetInfo(&free, &total);
            fprintf(stderr, "[mem_debug] before manual backward: used=%.1f GB, free=%.1f GB\n",
                (total - free) / 1e9, free / 1e9);
        }

        // Trigger main model backward.
        // FusedLayer: PyTorch autograd handles backward per layer.
        // Checkpoint: manual_group_backward recomputes each group with PyTorch.
        if (use_fused && hidden.grad().defined()) {
            hidden.backward(hidden.grad());
        } else if (hidden.grad().defined() && !ctx->group_inputs.empty() && !getenv("QWEN36_SUBCKPT")) {
            manual_group_backward(ctx, hidden.grad());
        } else if (hidden.grad().defined()) {
            hidden.backward(hidden.grad());
        }

        // MTP loss (if enabled) — run AFTER main backward to reuse freed GPU memory
        if (ctx->has_mtp && !getenv("QWEN36_DISABLE_MTP")) {
            // Detach hidden so MTP forward doesn't rebuild main model graph
            auto hidden_detached = hidden.detach().set_requires_grad(true);
            auto mtp_hidden = mtp_forward(ctx, hidden_detached, input_ids);
            auto mtp_loss = mtp_compute_loss(ctx, mtp_hidden, input_ids, target_mask);
            if (ctx->step_count == 0) {
                fprintf(stderr, "[mtp_debug] main_loss=%.4f mtp_loss=%.4f (x0.5=%.4f) total=%.4f\n",
                    loss_val, mtp_loss.item<double>() / 0.5, mtp_loss.item<double>(),
                    (loss_val + mtp_loss.item<double>()));
            }
            // Backward MTP loss — frees MTP intermediate tensors immediately
            mtp_loss.backward();
            // Add MTP gradient to hidden's gradient (already populated by main backward)
            if (hidden_detached.grad().defined()) {
                if (hidden.grad().defined()) {
                    hidden.grad().add_(hidden_detached.grad());
                } else {
                    hidden.mutable_grad() = hidden_detached.grad().clone();
                }
            }
        }

        // ── Adam optimizer step — CUDA multi-tensor fused kernel ──
        at::AutoGradMode guard(false);
        ctx->step_count++;
        ctx->lora_cache_valid = false;
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
            ctx->adam_dev_bufs.params_buf.copy_(params_cpu);
            ctx->adam_dev_bufs.grads_buf.copy_(grads_cpu);
            ctx->adam_dev_bufs.m_buf.copy_(m_cpu);
            ctx->adam_dev_bufs.v_buf.copy_(v_cpu);
            ctx->adam_dev_bufs.sizes_buf.copy_(sizes_cpu);

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

        // Sync all CUDA operations before returning — ensures no async errors
        // leak into the next train_step (critical for EP with NCCL)
        if (ctx->nccl_comm) {
            cudaDeviceSynchronize();
        }

        return loss_val;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36_train_step] FAILED: %s\n", e.what());
        return -1.0;
    }
}

// Get LoRA A tensor pointer by index
__attribute__((visibility("default"))) void* qwen36_get_lora_a(void* ctx_ptr, int64_t index) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    return &ctx->lora_a[index];
}

// Get LoRA B tensor pointer by index
__attribute__((visibility("default"))) void* qwen36_get_lora_b(void* ctx_ptr, int64_t index) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    return &ctx->lora_b[index];
}

// Free training context
__attribute__((visibility("default"))) void qwen36_free_training_context(void* ctx_ptr) {
    if (ctx_ptr) {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        if (ctx->nccl_comm) ncclCommDestroy(ctx->nccl_comm);
        if (ctx->nccl_stream) cudaStreamDestroy(ctx->nccl_stream);
        delete ctx;
    }
}

// Set NCCL communicator for Expert Parallel all-reduce
// Creates NCCL communicator directly in C++ using env vars RANK/WORLD_SIZE.
// Rank 0 generates unique ID and writes to /tmp/rustrain-nccl/nccl-id.bin
// Other ranks read it. All ranks call ncclCommInitRank.
// Returns 0 on success, -1 on failure.
__attribute__((visibility("default"))) int32_t qwen36_init_nccl(
    void* ctx_ptr
) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);

    const char* rank_str = getenv("RANK");
    const char* world_str = getenv("WORLD_SIZE");
    if (!rank_str || !world_str) return -1;
    int rank = atoi(rank_str);
    int world_size = atoi(world_str);
    if (world_size <= 1) return 0;  // no EP needed

    // Set CUDA device
    const char* local_rank_str = getenv("LOCAL_RANK");
    int local_rank = local_rank_str ? atoi(local_rank_str) : rank;
    cudaSetDevice(local_rank);

    // Exchange unique ID via file
    const char* id_path = "/tmp/rustrain-nccl/nccl-id.bin";
    ncclUniqueId unique_id;
    if (rank == 0) {
        mkdir("/tmp/rustrain-nccl", 0777);
        ncclGetUniqueId(&unique_id);
        FILE* f = fopen(id_path, "wb");
        fwrite(&unique_id, sizeof(unique_id), 1, f);
        fclose(f);
    } else {
        // Wait for rank 0 to write the ID file
        for (int i = 0; i < 600; i++) {  // 60 second timeout
            FILE* f = fopen(id_path, "rb");
            if (f) {
                if (fread(&unique_id, sizeof(unique_id), 1, f) == 1) {
                    fclose(f);
                    break;
                }
                fclose(f);
            }
            usleep(100000);  // 100ms
        }
    }

    // Initialize communicator
    ncclComm_t comm;
    ncclResult_t err = ncclCommInitRank(&comm, world_size, unique_id, rank);
    if (err != ncclSuccess) {
        fprintf(stderr, "[ep_nccl] ncclCommInitRank failed: %d (%s)\n", err, ncclGetErrorString(err));
        return -1;
    }

    // Create dedicated NCCL stream on current device
    cudaStream_t nccl_stream;
    cudaStreamCreate(&nccl_stream);

    ctx->nccl_comm = comm;
    ctx->nccl_stream = nccl_stream;
    ctx->ep_rank = rank;
    ctx->ep_world_size = world_size;

    // Propagate to layer configs
    for (auto& lc : ctx->layer_configs) {
        lc.nccl_comm = (void*)comm;
        lc.nccl_stream = (void*)nccl_stream;
    }
    for (auto& lc : ctx->mtp_layer_configs) {
        lc.nccl_comm = (void*)comm;
        lc.nccl_stream = (void*)nccl_stream;
    }

    fprintf(stderr, "[ep_nccl] rank=%d world=%d comm=%p stream=%p\n", rank, world_size, (void*)comm, (void*)nccl_stream);
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
        for (int64_t i = 0; i < ctx->num_layers; i++) {
            if (!adapter.target_layers.empty() && adapter.target_layers.find(i) == adapter.target_layers.end())
                continue;
            int64_t w_offset = 0;
            for (int64_t j = 0; j < i; j++)
                w_offset += weight_count_for_layer(ctx->layer_configs[j]);
            int64_t num_pairs;
            const int64_t* proj_indices;
            if (ctx->layer_configs[i].layer_type == 0) {
                static const int64_t full_indices[] = {2, 4, 6, 7};
                proj_indices = full_indices;
                num_pairs = 4;
            } else {
                static const int64_t linear_indices[] = {2, 3, 10};
                proj_indices = linear_indices;
                num_pairs = 3;
            }
            std::vector<std::pair<at::Tensor, at::Tensor>> pairs;
            std::vector<std::array<at::Tensor, 4>> adam_states;
            for (int k = 0; k < num_pairs; k++) {
                auto* base = ctx->weight_ptrs[w_offset + proj_indices[k]];
                int64_t out_f = base->size(0), in_f = base->size(1);
                auto a = at::randn({rank, in_f}, at::TensorOptions().dtype(ctx->compute_type).device(base->device())) * 0.01;
                auto b = at::zeros({out_f, rank}, at::TensorOptions().dtype(ctx->compute_type).device(base->device()));
                a.set_requires_grad(true);
                b.set_requires_grad(true);
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
        fprintf(stderr, "[q36_lora] added adapter %ld: rank=%ld alpha=%.1f\n", (long)id, (long)rank, alpha);
        return id;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36_add_lora] FAILED: %s\n", e.what());
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
int64_t qwen36_get_lora_count(void* ctx_ptr) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    int64_t total = (int64_t)ctx->lora_a.size();
    for (auto& adapter : ctx->adapters)
        for (auto& [layer_idx, pairs] : adapter.params)
            total += (int64_t)pairs.size() * 2;
    return total;
}

__attribute__((visibility("default")))
double qwen36_eval_step(void* ctx_ptr, void* input_ids_ptr, void* target_mask_ptr, void* attention_mask_ptr) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        auto& input_ids = *reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto& target_mask = *reinterpret_cast<at::Tensor*>(target_mask_ptr);
        if (attention_mask_ptr)
            ctx->attention_mask = *reinterpret_cast<at::Tensor*>(attention_mask_ptr);
        at::AutoGradMode no_grad(false);
        auto hidden = ctx->use_checkpoint ? forward_full_checkpoint(ctx, input_ids) : forward_full(ctx, input_ids);
        auto loss = compute_loss(ctx, hidden, input_ids, target_mask, ctx->vocab_size);
        return loss.item<double>();
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36_eval_step] FAILED: %s\n", e.what());
        return -1.0;
    }
}

__attribute__((visibility("default")))
int64_t qwen36_get_step_count(void* ctx_ptr) {
    return (int64_t)reinterpret_cast<TrainingContext*>(ctx_ptr)->step_count;
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
