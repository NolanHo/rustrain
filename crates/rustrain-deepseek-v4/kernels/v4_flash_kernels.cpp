// v4_flash_kernels.cpp — C++ FFI for DeepSeek V4 Flash training
//
// All compute in C++: forward + loss + backward + Adam optimizer.
// LoRA A/B created in C++ as at::Tensor (requires_grad=true).
// No tch-rs VarStore — gradients flow entirely within C++ autograd.
// Async NCCL: MoE output all_reduce overlapped with next layer's compute.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/autograd.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <memory>

// ── NCCL forward declarations (avoid #include <nccl.h> which pulls in cuda_fp16.h → nv/target) ──
typedef void* ncclComm_t;
typedef int ncclResult_t;
typedef enum { ncclSum = 0 } ncclRedOp_t;
// ncclDataType_t: we only need BFloat16 and Float
typedef enum {
    ncclInt8 = 0, ncclChar = 0,
    ncclUint8 = 1, ncclInt32 = 2, ncclInt = 2, ncclFloat32 = 3, ncclFloat = 3,
    ncclBfloat16 = 9,
} ncclDataType_t;
extern "C" {
    ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                                ncclDataType_t datatype, ncclRedOp_t op,
                                ncclComm_t comm, void* stream);
}

// ──────────────────────────────────────────────────────────────────────
// Forward declarations
// ──────────────────────────────────────────────────────────────────────

static at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double eps);

// ──────────────────────────────────────────────────────────────────────
// Tilelang fused kernel loading (dlopen)
// ──────────────────────────────────────────────────────────────────────

// Tilelang compiles Python DSL → .so with C entry points.
// We dlopen at runtime; if not found, fall back to ATen ops.

struct TilelangKernels {
    // fused_rmsnorm_matmul(X [M,K], W_norm [K], W_matmul [N,K], Y [M,N], M, N, K, eps)
    void (*fused_rmsnorm_matmul)(void*, void*, void*, void*, int64_t, int64_t, int64_t, double);
    // fused_swiglu(gate [M,I], up [M,I], out [M,I], M, I, limit)
    void (*fused_swiglu)(void*, void*, void*, int64_t, int64_t, double);
};

static TilelangKernels* g_tilelang = nullptr;

static void load_tilelang_kernels() {
    if (g_tilelang) return;
    void* handle = dlopen("libtilelang_fused.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) handle = dlopen("libtilelang_fused.so", RTLD_LAZY);
    if (!handle) return;

    auto* k = new TilelangKernels{};
    auto load_sym = [&](const char* name, auto* fn_ptr) {
        void* p = dlsym(handle, name);
        if (p) *(void**)fn_ptr = p;
    };
    load_sym("tilelang_fused_rmsnorm_matmul", &k->fused_rmsnorm_matmul);
    load_sym("tilelang_fused_swiglu", &k->fused_swiglu);
    g_tilelang = k;
    fprintf(stderr, "[v4_tilelang] loaded: rmsnorm_matmul=%s swiglu=%s\n",
        k->fused_rmsnorm_matmul ? "yes" : "no",
        k->fused_swiglu ? "yes" : "no");
}

/// Fused RMSNorm + Matmul via Tilelang (falls back to ATen if not available)
static at::Tensor fused_rmsnorm_matmul(
    const at::Tensor& input, const at::Tensor& norm_w, const at::Tensor& matmul_w,
    double eps)
{
    if (g_tilelang && g_tilelang->fused_rmsnorm_matmul) {
        int64_t M = input.size(0);
        int64_t K = input.size(1);
        int64_t N = matmul_w.size(0);
        auto output = at::zeros({M, N}, input.options());
        g_tilelang->fused_rmsnorm_matmul(
            input.data_ptr(), norm_w.data_ptr(), matmul_w.data_ptr(),
            output.data_ptr(), M, N, K, eps);
        return output;
    }
    // Fallback: ATen
    auto normed = rms_norm(input, norm_w, eps);
    return at::linear(normed, matmul_w);
}

/// Fused SwiGLU via Tilelang (falls back to ATen if not available)
static at::Tensor fused_swiglu_op(
    const at::Tensor& gate_out, const at::Tensor& up_out, double limit)
{
    if (g_tilelang && g_tilelang->fused_swiglu) {
        int64_t M = gate_out.size(0);
        int64_t I = gate_out.size(1);
        auto output = at::zeros({M, I}, gate_out.options());
        g_tilelang->fused_swiglu(
            gate_out.data_ptr(), up_out.data_ptr(), output.data_ptr(),
            M, I, limit);
        return output;
    }
    // Fallback: ATen
    auto inter = at::silu(gate_out) * up_out;
    if (limit > 0.0) inter = inter.clamp(-limit, limit);
    return inter;
}

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────

static at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double eps) {
    auto input_f32 = input.to(at::kFloat);
    auto variance = input_f32.pow(2).mean(-1, true);
    auto inv_rms = (variance + eps).rsqrt();
    return (input_f32 * inv_rms * weight.to(at::kFloat)).to(input.scalar_type());
}

static at::Tensor lora_delta(const at::Tensor& base, const at::Tensor& a, const at::Tensor& b, double scaling) {
    auto kind = base.scalar_type();
    auto delta = b.to(kind).matmul(a.to(kind));
    return base + (delta * scaling).to(kind);
}

static at::Tensor v4_swiglu(const at::Tensor& input, const at::Tensor& w1, const at::Tensor& w2,
                            const at::Tensor& w3, double limit) {
    auto gate = at::matmul(input, w1.t());
    auto up = at::matmul(input, w3.t());
    // Fused silu * clamp via Tilelang (falls back to ATen)
    auto inter = fused_swiglu_op(gate, up, limit);
    return at::matmul(inter, w2.t());
}

// ── FP8 safe_linear (reuse from fp8_gemm.cpp pattern) ──

static at::Tensor safe_linear(const at::Tensor& input, const at::Tensor& weight,
                               std::optional<at::Tensor> weight_scale) {
    auto dtype = input.scalar_type();
    auto device = input.device();
    if (weight_scale.has_value() && weight.scalar_type() == at::kFloat8_e4m3fn) {
        // FP8 path: dequant with scale, then linear
        auto& s = weight_scale.value();
        auto s_dev = s.to(device).to(at::kFloat);
        int64_t n = weight.size(0), k = weight.size(1);
        int64_t n_blocks = (n + 127) / 128;
        int64_t k_blocks = (k + 127) / 128;
        at::Tensor scale_expanded;
        if (s_dev.size(0) == n_blocks && s_dev.size(1) == k_blocks) {
            int64_t n_padded = n_blocks * 128;
            int64_t k_padded = k_blocks * 128;
            auto expanded = s_dev.unsqueeze(-1).unsqueeze(-1)
                .expand({n_blocks, k_blocks, 128, 128})
                .reshape({n_padded, k_padded})
                .contiguous()
                .narrow(0, 0, n).narrow(1, 0, k);
            scale_expanded = expanded;
        } else {
            scale_expanded = s_dev;
        }
        auto w_bf16 = (weight.to(at::kFloat) * scale_expanded).to(dtype);
        return at::linear(input, w_bf16);
    }
    auto w = weight.to(device).to(dtype);
    return at::linear(input, w);
}

// ── RoPE (YaRN + compress_rope_theta) ──

static at::Tensor rotate_half(const at::Tensor& x) {
    auto last_dim = x.size(-1);
    auto half = last_dim / 2;
    return at::cat({x.narrow(-1, half, half).neg(), x.narrow(-1, 0, half)}, -1);
}

// Compute cos/sin for RoPE with optional YaRN scaling.
// Returns (cos, sin) each [seq_len, rope_dim] where rope_dim = qk_rope_head_dim.
static void rope_cos_sin(int64_t seq_len, int64_t rope_dim, double theta,
                         bool use_yarn, double yarn_factor, double beta_fast, double beta_slow,
                         int64_t orig_max_pos, at::Device device,
                         at::Tensor& cos, at::Tensor& sin) {
    int64_t half = rope_dim / 2;
    auto exponents = at::arange(0, rope_dim, 2, at::TensorOptions().dtype(at::kFloat).device(device)) / (double)rope_dim;
    at::Tensor inv_freq;
    if (use_yarn) {
        auto base_inv = (exponents * std::log(theta)).exp().reciprocal();
        auto range = (double)orig_max_pos * base_inv;
        auto low = beta_slow;
        auto high = beta_fast;
        auto t = (range - low) / (high - low);
        auto scale_factor = (1.0 / yarn_factor) * (1.0 - t) + 1.0 * t;
        auto mask_high = range > high;
        auto mask_low = range < low;
        inv_freq = at::where(mask_high, base_inv, at::where(mask_low, base_inv / yarn_factor, base_inv * scale_factor));
    } else {
        inv_freq = (exponents * std::log(theta)).exp().reciprocal();
    }
    auto pos = at::arange(seq_len, at::TensorOptions().dtype(at::kFloat).device(device)).unsqueeze(-1);
    auto freqs = pos * inv_freq.unsqueeze(0);  // [seq, half]
    auto emb = at::cat({freqs, freqs}, -1);    // [seq, rope_dim]
    cos = emb.cos();
    sin = emb.sin();
}

// ── Compress / Decompress ──

static at::Tensor compress_seq(const at::Tensor& hidden, int64_t ratio) {
    if (ratio <= 1) return hidden;
    int64_t seq = hidden.size(1);
    int64_t new_seq = seq / ratio;
    if (new_seq == 0) return hidden;
    auto batch = hidden.size(0), dim = hidden.size(2);
    return hidden.narrow(1, 0, new_seq * ratio)
        .reshape({batch, new_seq, ratio, dim})
        .mean(2, false);
}

static at::Tensor decompress_seq(const at::Tensor& hidden, int64_t ratio, int64_t target_seq) {
    if (ratio <= 1) return hidden;
    int64_t seq = hidden.size(1);
    if (seq >= target_seq) return hidden.narrow(1, 0, target_seq);
    auto batch = hidden.size(0), dim = hidden.size(2);
    return hidden.reshape({batch, seq, 1, dim})
        .expand({batch, seq, ratio, dim})
        .reshape({batch, seq * ratio, dim})
        .narrow(1, 0, target_seq);
}

// ── HC Attention Bias ──

static at::Tensor compute_hc_attn_bias(
    const at::Tensor& hc_attn_base, const at::Tensor& hc_attn_fn,
    const at::Tensor& hc_attn_scale,
    int64_t n_hash, int64_t n_heads, int64_t head_dim)
{
    int64_t hash_dim = 8;
    auto pattern = at::zeros({n_heads * 2 * head_dim},
        hc_attn_fn.options());
    for (int64_t i = 0; i < n_hash; i++) {
        auto base_i = hc_attn_base.narrow(0, i * hash_dim, hash_dim);
        auto fn_i = hc_attn_fn.narrow(0, i * hash_dim, hash_dim);
        auto scale_i = hc_attn_scale[i].item<double>();
        auto pattern_i = base_i.unsqueeze(0).matmul(fn_i).squeeze(0);
        pattern = pattern + pattern_i * scale_i;
    }
    auto p = pattern.reshape({n_heads, 2, head_dim});
    auto q_hash = p.select(1, 0);  // [n_heads, head_dim]
    auto k_hash = p.select(1, 1);
    return q_hash.unsqueeze(-1).matmul(k_hash.unsqueeze(-2));  // [n_heads, head_dim, head_dim]
}

// ── V4 MLA Attention ──

struct V4LayerConfig {
    int64_t head_dim;       // 512
    int64_t num_heads;
    int64_t qk_rope_dim;    // 64
    int64_t o_groups;
    int64_t kv_lora_rank;   // 512
    double rope_theta;
    bool use_yarn;
    double yarn_factor, beta_fast, beta_slow;
    int64_t orig_max_pos;
    double rms_eps;
    double swiglu_limit;
    int64_t sliding_window;
    // MoE
    int64_t n_experts, top_k, moe_intermediate, n_shared_experts;
    double routed_scaling_factor;
    bool norm_topk_prob;  // always false for V4 (noaux_tc handles it)
    std::string scoring_func;
    std::string topk_method;
    int64_t hc_sinkhorn_iters, hc_mult;
    double hc_eps;
    int64_t index_n_heads, index_head_dim, index_topk, num_hash_layers;
    // EP
    int64_t expert_start, expert_count;
    // Compression
    int64_t compress_ratio;
    // LoRA
    bool has_lora;
};

static at::Tensor v4_attention(
    const at::Tensor& hidden,
    const at::Tensor& wq_a, const at::Tensor& wq_b,
    const at::Tensor& wkv, const at::Tensor& wo_a, const at::Tensor& wo_b,
    const at::Tensor& q_norm, const at::Tensor& kv_norm,
    const at::Tensor& attn_sink,
    at::Tensor** hc_weights,  // [base, fn, scale, ffn_base, ffn_fn, ffn_scale] or null
    const V4LayerConfig* cfg,
    at::ScalarType compute_type)
{
    auto device = hidden.device();
    int64_t batch = hidden.size(0), seq = hidden.size(1);
    int64_t head_dim = cfg->head_dim;
    int64_t qk_rope = cfg->qk_rope_dim;
    int64_t qk_nope = head_dim - qk_rope;
    int64_t num_heads = cfg->num_heads;
    int64_t o_groups = cfg->o_groups;

    // Q path: wq_a → q_norm → wq_b
    auto q_a = safe_linear(hidden, wq_a, std::nullopt);
    q_a = rms_norm(q_a, q_norm, cfg->rms_eps);
    auto q_b = safe_linear(q_a, wq_b, std::nullopt);
    auto q = q_b.reshape({batch, seq, num_heads, head_dim}).transpose(1, 2);
    auto q_nope = q.narrow(-1, 0, qk_nope);
    auto q_rope = q.narrow(-1, qk_nope, qk_rope);

    // KV path: wkv → kv_norm (MQA, shared across heads)
    auto wkv_out = safe_linear(hidden, wkv, std::nullopt);
    auto kv = rms_norm(wkv_out, kv_norm, cfg->rms_eps);
    auto k_nope = kv.narrow(-1, 0, qk_nope).reshape({batch, 1, seq, qk_nope}).expand({batch, num_heads, seq, qk_nope});
    auto k_rope = kv.narrow(-1, qk_nope, qk_rope).reshape({batch, 1, seq, qk_rope}).expand({batch, num_heads, seq, qk_rope});
    auto v = kv.reshape({batch, 1, seq, head_dim}).expand({batch, num_heads, seq, head_dim});

    // RoPE
    at::Tensor cos, sin;
    rope_cos_sin(seq, qk_rope, cfg->rope_theta, cfg->use_yarn, cfg->yarn_factor,
                 cfg->beta_fast, cfg->beta_slow, cfg->orig_max_pos, device, cos, sin);
    cos = cos.to(compute_type);
    sin = sin.to(compute_type);
    auto q_rope_rot = q_rope * cos.unsqueeze(0).unsqueeze(0) + rotate_half(q_rope) * sin.unsqueeze(0).unsqueeze(0);
    auto k_rope_rot = k_rope * cos.unsqueeze(0).unsqueeze(0) + rotate_half(k_rope) * sin.unsqueeze(0).unsqueeze(0);

    auto q_full = at::cat({q_nope, q_rope_rot}, -1);
    auto k_full = at::cat({k_nope, k_rope_rot}, -1);

    // Attention scores with sink
    double scale = 1.0 / std::sqrt((double)head_dim);
    auto scores = at::matmul(q_full, k_full.transpose(-1, -2)) * scale;
    auto sink = attn_sink.reshape({1, num_heads, 1, 1}).to(scores.scalar_type());
    scores = scores + sink;

    // HC bias (if compressed layer with HC weights)
    if (hc_weights) {
        auto hc_bias = compute_hc_attn_bias(*hc_weights[0], *hc_weights[1], *hc_weights[2],
            cfg->num_hash_layers, cfg->index_n_heads, cfg->index_head_dim);
        if (cfg->index_n_heads == num_heads) {
            auto hc_trace = hc_bias.diagonal(0, -1, -2).sum(1, false);
            scores = scores + hc_trace.reshape({1, num_heads, 1, 1}).to(scores.scalar_type());
        } else {
            auto hc_scalar = hc_bias.mean(0).sum().item<double>() / (double)head_dim;
            scores = scores + hc_scalar;
        }
    }

    // Causal mask with optional sliding window
    at::Tensor mask;
    if (cfg->sliding_window > 0 && seq > cfg->sliding_window) {
        auto sw = cfg->sliding_window;
        auto pos = at::arange(seq, at::TensorOptions().dtype(at::kFloat).device(device));
        auto diff = pos.unsqueeze(0) - pos.unsqueeze(1);
        mask = (diff.ge(0) * diff.lt((double)sw)).to(at::kBool);
    } else {
        auto pos = at::arange(seq, at::TensorOptions().dtype(at::kFloat).device(device));
        auto diff = pos.unsqueeze(0) - pos.unsqueeze(1);
        mask = diff.ge(0).to(at::kBool);
    }
    scores = scores.masked_fill(mask.logical_not().unsqueeze(0).unsqueeze(0),
                                -std::numeric_limits<float>::infinity());

    auto probs = scores.softmax(-1, at::kFloat).to(v.scalar_type());
    auto context = at::matmul(probs, v);  // [batch, heads, seq, head_dim]

    // Group reduction: [batch, heads, seq, head_dim] → [batch, o_groups, seq, head_dim]
    int64_t heads_per_group = num_heads / o_groups;
    context = context.reshape({batch, o_groups, heads_per_group, seq, head_dim}).sum(2, false);
    context = context.reshape({batch, seq, o_groups * head_dim}).to(compute_type);

    // Output: wo_a → wo_b
    auto o_comp = safe_linear(context, wo_a, std::nullopt);
    return safe_linear(o_comp, wo_b, std::nullopt);
}

// ── V4 MoE MLP (noaux_tc routing) ──

static at::Tensor v4_moe_mlp(
    const at::Tensor& hidden,
    const at::Tensor& gate_w,
    const at::Tensor& shared_w1, const at::Tensor& shared_w2, const at::Tensor& shared_w3,
    at::Tensor** expert_w1, at::Tensor** expert_w2, at::Tensor** expert_w3,
    int64_t n_local_experts,
    int64_t top_k, int64_t moe_inter, double swiglu_limit,
    double routed_scaling_factor,
    const std::string& scoring_func, const std::string& topk_method,
    int64_t hc_sinkhorn_iters, int64_t hc_mult, double hc_eps,
    int64_t n_experts_total, int64_t expert_start,
    at::ScalarType compute_type)
{
    int64_t batch = hidden.size(0), seq = hidden.size(1), hidden_dim = hidden.size(2);
    auto flat = hidden.reshape({batch * seq, hidden_dim});

    // Shared expert (always computed, replicated)
    auto shared_out = v4_swiglu(flat, shared_w1, shared_w2, shared_w3, swiglu_limit);

    // Router logits
    auto router_logits = at::matmul(flat, gate_w.t());

    // Scoring function
    at::Tensor scores;
    if (scoring_func == "sqrtsoftplus") {
        scores = (router_logits.exp() + 1.0).log().sqrt();
    } else {
        scores = router_logits.softmax(-1, at::kFloat);
    }

    // Top-k selection with optional Sinkhorn (noaux_tc)
    at::Tensor topk_weights, topk_indices;
    if (topk_method == "noaux_tc" && hc_mult > 1 && n_experts_total > top_k) {
        int64_t k_ext = std::min(top_k * hc_mult, n_experts_total);
        auto [ext_scores, ext_indices] = scores.topk(k_ext, -1, true, true);
        // Sinkhorn on over-selected scores
        auto flat_scores = ext_scores.reshape({-1, k_ext});
        for (int64_t it = 0; it < hc_sinkhorn_iters; it++) {
            auto row_sum = flat_scores.sum(-1, true).clamp_min(hc_eps);
            flat_scores = flat_scores / row_sum;
            auto col_sum = flat_scores.sum(0, true).clamp_min(hc_eps);
            flat_scores = flat_scores / col_sum;
        }
        auto normalized = flat_scores.reshape(ext_scores.sizes());
        auto [_, final_local_idx] = normalized.topk(top_k, -1, true, true);
        topk_indices = ext_indices.gather(-1, final_local_idx);
        topk_weights = ext_scores.gather(-1, final_local_idx);
    } else {
        auto [tw, ti] = scores.topk(top_k, -1, true, true);
        topk_weights = tw;
        topk_indices = ti;
    }

    // Normalize
    auto denom = topk_weights.sum(-1, true).clamp_min(1e-9);
    topk_weights = (topk_weights / denom) * routed_scaling_factor;
    topk_weights = topk_weights.to(compute_type);

    // Accumulate expert outputs (local experts only)
    auto output = shared_out;
    for (int64_t k = 0; k < top_k; k++) {
        auto expert_indices = topk_indices.select(-1, k);
        auto expert_weights = topk_weights.select(-1, k);
        for (int64_t e_local = 0; e_local < n_local_experts; e_local++) {
            int64_t e_global = expert_start + e_local;
            auto mask = expert_indices.eq(e_global).to(compute_type);
            if (mask.sum().item<double>() > 0.0) {
                auto token_idx = mask.nonzero().squeeze(-1);
                if (token_idx.size(0) == 0) continue;
                auto selected = flat.index_select(0, token_idx);
                auto expert_out = v4_swiglu(selected, *expert_w1[e_local], *expert_w2[e_local], *expert_w3[e_local], swiglu_limit);
                auto w = expert_weights.index_select(0, token_idx).unsqueeze(-1);
                output = output.index_add(0, token_idx, expert_out * w);
            }
        }
    }
    return output.reshape({batch, seq, hidden_dim});
}

// ── Cross-entropy loss ──

static at::Tensor compute_loss(
    const at::Tensor& hidden, const at::Tensor& input_ids, const at::Tensor& target_mask,
    const at::Tensor& final_norm_w, const at::Tensor& lm_head,
    double rms_eps, int64_t vocab_size)
{
    auto hidden_normed = rms_norm(hidden, final_norm_w, rms_eps);
    auto logits = at::matmul(hidden_normed, lm_head.t());

    int64_t seq_len = logits.size(1);
    auto shifted_logits = logits.narrow(1, 0, seq_len - 1).reshape({-1, vocab_size});
    auto shifted_targets = input_ids.narrow(1, 1, seq_len - 1).reshape({-1});
    auto shifted_mask = target_mask.narrow(1, 1, seq_len - 1).reshape({-1});

    auto log_probs = shifted_logits.log_softmax(-1, at::kFloat);
    auto per_token_loss = log_probs.gather(1, shifted_targets.unsqueeze(1)).squeeze(1).neg();
    auto masked = per_token_loss * shifted_mask.to(at::kFloat);
    return masked.sum() / shifted_mask.sum().clamp_min(1.0);
}

// ── MTP ──

struct MtpWeights {
    at::Tensor norm, hnorm, head, ffn_norm, ffn_w1, ffn_w2, ffn_w3;
};

static at::Tensor mtp_forward(
    const at::Tensor& hidden, const at::Tensor& input_ids, const at::Tensor& embed,
    const MtpWeights& mtp, double rms_eps)
{
    int64_t seq_len = hidden.size(1);
    auto hidden_shifted = hidden.narrow(1, 0, seq_len - 1);
    auto embed_next = at::embedding(embed, input_ids.narrow(1, 1, seq_len - 1));
    auto combined = (hidden_shifted + embed_next) / 2.0;
    auto normed = rms_norm(combined, mtp.norm, rms_eps);
    auto ffn_out = v4_swiglu(normed, mtp.ffn_w1, mtp.ffn_w2, mtp.ffn_w3, 10.0);
    auto after_ffn = normed + ffn_out;
    auto final_h = rms_norm(after_ffn, mtp.hnorm, rms_eps);
    return at::matmul(final_h, mtp.head.t());
}

static at::Tensor mtp_compute_loss(
    const at::Tensor& mtp_logits, const at::Tensor& input_ids,
    const at::Tensor& target_mask, int64_t vocab_size)
{
    int64_t seq_len = input_ids.size(1);
    auto shifted_logits = mtp_logits.reshape({-1, vocab_size});
    auto shifted_targets = input_ids.narrow(1, 2, seq_len - 2).reshape({-1});
    auto shifted_mask = target_mask.narrow(1, 2, seq_len - 2).reshape({-1});
    auto log_probs = shifted_logits.log_softmax(-1, at::kFloat);
    auto per_token_loss = log_probs.gather(1, shifted_targets.unsqueeze(1)).squeeze(1).neg();
    auto masked = per_token_loss * shifted_mask.to(at::kFloat);
    return (masked.sum() / masked.sum().clamp_min(1.0)) * 0.5;
}

// ──────────────────────────────────────────────────────────────────────
// NCCL helpers (async all_reduce with CUDA events)
// ──────────────────────────────────────────────────────────────────────

// Launch async all_reduce on comm_stream, return output + event for later sync.
// Does NOT block the compute stream — caller must cudaStreamWaitEvent before using output.
static void async_all_reduce(
    ncclComm_t comm, cudaStream_t comm_stream,
    const at::Tensor& input, at::Tensor& output, cudaEvent_t& event)
{
    int64_t count = input.numel();
    ncclDataType_t dtype;
    switch (input.scalar_type()) {
        case at::kBFloat16: dtype = ncclBfloat16; break;
        case at::kFloat:    dtype = ncclFloat;    break;
        default:            dtype = ncclFloat;    break;
    }
    ncclAllReduce(input.data_ptr(), output.data_ptr(), count, dtype, ncclSum, comm, comm_stream);
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    cudaEventRecord(event, comm_stream);
}

// Blocking all_reduce (for LoRA gradients before Adam).
static void sync_all_reduce(
    ncclComm_t comm, cudaStream_t comm_stream, at::Tensor& tensor)
{
    int64_t count = tensor.numel();
    auto buf = at::zeros_like(tensor);
    ncclDataType_t dtype;
    switch (tensor.scalar_type()) {
        case at::kBFloat16: dtype = ncclBfloat16; break;
        case at::kFloat:    dtype = ncclFloat;    break;
        default:            dtype = ncclFloat;    break;
    }
    ncclAllReduce(tensor.data_ptr(), buf.data_ptr(), count, dtype, ncclSum, comm, comm_stream);
    cudaStreamSynchronize(comm_stream);
    tensor.copy_(buf);
}

// ──────────────────────────────────────────────────────────────────────
// Training Context
// ──────────────────────────────────────────────────────────────────────

struct TrainingContext {
    // Model weights (frozen) — pointers to external tensors
    // Per layer: [attn_norm, ffn_norm, wq_a, wq_b, wkv, wo_a, wo_b, q_norm, kv_norm, attn_sink,
    //             gate, shared_w1, shared_w2, shared_w3,
    //             expert_w1[0..N-1], expert_w2[0..N-1], expert_w3[0..N-1],
    //             hc_attn_base, hc_attn_fn, hc_attn_scale, hc_ffn_base, hc_ffn_fn, hc_ffn_scale]
    std::vector<at::Tensor*> weight_ptrs;
    std::vector<int64_t> weight_count_per_layer;  // weights per layer (varies with EP)
    std::vector<bool> layer_has_hc;
    at::Tensor *embed_ptr, *final_norm_ptr, *lm_head_ptr;
    std::vector<V4LayerConfig> layer_configs;
    int64_t num_layers;

    // LoRA (owned in C++, requires_grad=true)
    std::vector<at::Tensor> lora_a, lora_b;
    std::vector<int64_t> lora_layer_offset;
    double lora_scaling;

    // Adam
    std::vector<at::Tensor> adam_m, adam_v;
    double lr, beta1, beta2, eps;
    int64_t step_count;

    at::ScalarType compute_type;
    int64_t vocab_size;
    double rms_eps;

    // MTP
    bool has_mtp;
    MtpWeights mtp;
    at::Tensor embed_for_mtp;

    // Checkpointing
    bool use_checkpoint;
    int64_t group_size;

    // NCCL (for EP gradient sync)
    ncclComm_t nccl_comm;
    cudaStream_t comm_stream;
    int world_size;
    int rank;
};

// Weight count per layer: 14 base + 3*n_local_experts + (6 if HC else 0)
static int64_t weight_count_for_layer(const V4LayerConfig& cfg) {
    return 14 + 3 * cfg.expert_count + (cfg.num_hash_layers > 0 ? 6 : 0);
}

// Forward pass for a single layer — returns (residual, mlp_output) separately
// so the caller can all_reduce the MoE output for EP.
static std::pair<at::Tensor, at::Tensor> forward_single_layer_split(
    TrainingContext* ctx, const at::Tensor& hidden, int64_t layer_idx)
{
    auto& cfg = ctx->layer_configs[layer_idx];
    int64_t w_offset = 0;
    for (int64_t j = 0; j < layer_idx; j++)
        w_offset += weight_count_for_layer(ctx->layer_configs[j]);

    auto** w = ctx->weight_ptrs.data() + w_offset;
    auto kind = ctx->compute_type;

    auto h = hidden;
    if (cfg.compress_ratio > 1) {
        h = compress_seq(hidden, cfg.compress_ratio);
    }

    // Fused RMSNorm + Q projection (Tilelang if available, else ATen fallback)
    auto attn_input = fused_rmsnorm_matmul(h, *w[0], *w[2], cfg.rms_eps);

    auto wq_a = *w[2];
    auto wq_b = *w[3], wkv = *w[4], wo_a = *w[5], wo_b = *w[6];
    if (cfg.has_lora) {
        // LoRA modifies the projection weights — can't use fused path for LoRA layers
        // Re-do with separate rms_norm + lora_delta + safe_linear
        attn_input = rms_norm(h, *w[0], cfg.rms_eps);
        int64_t la_off = ctx->lora_layer_offset[layer_idx];
        wq_a = lora_delta(*w[2], ctx->lora_a[la_off], ctx->lora_b[la_off], ctx->lora_scaling);
        wq_b = lora_delta(*w[3], ctx->lora_a[la_off+1], ctx->lora_b[la_off+1], ctx->lora_scaling);
        wkv = lora_delta(*w[4], ctx->lora_a[la_off+2], ctx->lora_b[la_off+2], ctx->lora_scaling);
        wo_a = lora_delta(*w[5], ctx->lora_a[la_off+3], ctx->lora_b[la_off+3], ctx->lora_scaling);
        wo_b = lora_delta(*w[6], ctx->lora_a[la_off+4], ctx->lora_b[la_off+4], ctx->lora_scaling);
        attn_input = safe_linear(attn_input, wq_a, std::nullopt);
    }

    at::Tensor** hc_ptr = nullptr;
    if (cfg.num_hash_layers > 0) {
        hc_ptr = w + 14 + 3 * cfg.expert_count;
    }
    auto attn_out = v4_attention(attn_input, wq_a, wq_b, wkv, wo_a, wo_b,
        *w[7], *w[8], *w[9], hc_ptr, &cfg, kind);

    auto residual = h + attn_out;
    auto mlp_input = rms_norm(residual, *w[1], cfg.rms_eps);

    auto mlp_out = v4_moe_mlp(mlp_input, *w[10],
        *w[11], *w[12], *w[13],
        w + 14, w + 14 + cfg.expert_count, w + 14 + 2*cfg.expert_count,
        cfg.expert_count, cfg.top_k, cfg.moe_intermediate, cfg.swiglu_limit,
        cfg.routed_scaling_factor, cfg.scoring_func, cfg.topk_method,
        cfg.hc_sinkhorn_iters, cfg.hc_mult, cfg.hc_eps,
        cfg.n_experts, cfg.expert_start, kind);

    return {residual, mlp_out};
}

// Full forward with async NCCL pipeline (no checkpointing)
static at::Tensor forward_full(TrainingContext* ctx, const at::Tensor& input_ids) {
    at::AutoGradMode guard(true);
    auto hidden = at::embedding(*ctx->embed_ptr, input_ids);

    // Async pipeline: (pending_output, pending_event) from previous layer's all_reduce.
    at::Tensor pending_output;
    cudaEvent_t pending_event = nullptr;

    for (int64_t i = 0; i < ctx->num_layers; i++) {
        // Wait for previous layer's async all_reduce to complete (GPU-side, no CPU block)
        if (pending_event) {
            auto compute_stream = at::cuda::getCurrentCUDAStream();
            cudaStreamWaitEvent(compute_stream, pending_event);
            cudaEventDestroy(pending_event);
            pending_event = nullptr;
            hidden = pending_output;
        }

        auto [residual, mlp_out] = forward_single_layer_split(ctx, hidden, i);

        if (ctx->world_size > 1) {
            // Async all_reduce on MoE output (shared expert is replicated → divide by world_size)
            auto pd = mlp_out.detach();
            auto reduced = at::zeros_like(pd);
            async_all_reduce(ctx->nccl_comm, ctx->comm_stream, pd, reduced, pending_event);
            auto full = (reduced / (double)ctx->world_size).to(ctx->compute_type);
            full.set_requires_grad(true);
            hidden = residual + full;
            pending_output = hidden;
        } else {
            hidden = residual + mlp_out;
        }
    }

    // Wait for final all_reduce
    if (pending_event) {
        auto compute_stream = at::cuda::getCurrentCUDAStream();
        cudaStreamWaitEvent(compute_stream, pending_event);
        cudaEventDestroy(pending_event);
    }

    // Decompress
    int64_t orig_seq = input_ids.size(1);
    int64_t max_ratio = 0;
    for (int64_t i = 0; i < ctx->num_layers; i++) {
        if (ctx->layer_configs[i].compress_ratio > max_ratio)
            max_ratio = ctx->layer_configs[i].compress_ratio;
    }
    if (max_ratio > 1) {
        hidden = decompress_seq(hidden, max_ratio, orig_seq);
    }
    return hidden;
}

// Gradient checkpointing — uses the non-split forward (no per-layer all_reduce).
// For EP + checkpointing, all_reduce is done after backward on LoRA grads.
static at::Tensor forward_layer_group(TrainingContext* ctx, const at::Tensor& input,
                                       int64_t start, int64_t end) {
    at::Tensor h = input;
    for (int64_t i = start; i < end; i++) {
        auto [residual, mlp_out] = forward_single_layer_split(ctx, h, i);
        if (ctx->world_size > 1) {
            // Synchronous all_reduce within checkpoint groups (simpler, no async overlap)
            auto pd = mlp_out.detach();
            sync_all_reduce(ctx->nccl_comm, ctx->comm_stream, pd);
            auto full = (pd / (double)ctx->world_size).to(ctx->compute_type);
            full.set_requires_grad(true);
            h = residual + full;
        } else {
            h = residual + mlp_out;
        }
    }
    return h;
}

struct GroupCheckpointFunction : public torch::autograd::Function<GroupCheckpointFunction> {
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
        at::Tensor input, int64_t tc_val, int64_t start, int64_t end) {
        ctx->saved_data["tc"] = tc_val;
        ctx->saved_data["start"] = start;
        ctx->saved_data["end"] = end;
        ctx->save_for_backward({input});
        auto* tc = reinterpret_cast<TrainingContext*>(tc_val);
        return forward_layer_group(tc, input, start, end);
    }
    static std::vector<at::Tensor> backward(torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output) {
        auto saved = ctx->get_saved_variables();
        at::Tensor input = saved[0];
        auto tc = reinterpret_cast<TrainingContext*>(ctx->saved_data["tc"].toInt());
        int64_t start = ctx->saved_data["start"].toInt();
        int64_t end = ctx->saved_data["end"].toInt();
        at::AutoGradMode guard(true);
        input.set_requires_grad(true);
        auto output = forward_layer_group(tc, input, start, end);
        torch::autograd::backward({output}, {grad_output[0]}, true, false);
        return {input.grad(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

static at::Tensor forward_full_checkpoint(TrainingContext* ctx, const at::Tensor& input_ids) {
    auto hidden = at::embedding(*ctx->embed_ptr, input_ids);
    hidden = hidden.detach().set_requires_grad(true);
    int64_t gs = ctx->group_size > 0 ? ctx->group_size : 4;
    for (int64_t start = 0; start < ctx->num_layers; start += gs) {
        int64_t end = std::min(start + gs, ctx->num_layers);
        hidden = GroupCheckpointFunction::apply(hidden, (int64_t)(uintptr_t)ctx, start, end);
    }
    // Decompress
    int64_t orig_seq = input_ids.size(1);
    int64_t max_ratio = 0;
    for (int64_t i = 0; i < ctx->num_layers; i++) {
        if (ctx->layer_configs[i].compress_ratio > max_ratio)
            max_ratio = ctx->layer_configs[i].compress_ratio;
    }
    if (max_ratio > 1) {
        hidden = decompress_seq(hidden, max_ratio, orig_seq);
    }
    return hidden;
}

// ──────────────────────────────────────────────────────────────────────
// C FFI
// ──────────────────────────────────────────────────────────────────────

extern "C" {

void* v4_create_training_context(
    void** weight_ptrs, int64_t num_weight_ptrs,
    void* embed_ptr, void* final_norm_ptr, void* lm_head_ptr,
    void* layer_configs_ptr, int64_t num_layers,
    int32_t compute_type,
    double lora_scaling, double lr, double beta1, double beta2, double eps,
    int64_t vocab_size, double rms_eps,
    void** mtp_ptrs, int64_t num_mtp_ptrs,
    void* embed_for_mtp_ptr
) {
    try {
        load_tilelang_kernels();  // Load Tilelang fused kernels if available
        auto* ctx = new TrainingContext();
        ctx->compute_type = static_cast<at::ScalarType>(compute_type);
        ctx->lr = lr; ctx->beta1 = beta1; ctx->beta2 = beta2; ctx->eps = eps;
        ctx->vocab_size = vocab_size; ctx->rms_eps = rms_eps;
        ctx->step_count = 0; ctx->lora_scaling = lora_scaling;
        ctx->num_layers = num_layers;
        ctx->use_checkpoint = false; ctx->group_size = 4;
        ctx->has_mtp = (mtp_ptrs != nullptr && num_mtp_ptrs >= 7);

        auto** wp = reinterpret_cast<at::Tensor**>(weight_ptrs);
        for (int64_t i = 0; i < num_weight_ptrs; i++) {
            ctx->weight_ptrs.push_back(wp[i]);
        }
        ctx->embed_ptr = reinterpret_cast<at::Tensor*>(embed_ptr);
        ctx->final_norm_ptr = reinterpret_cast<at::Tensor*>(final_norm_ptr);
        ctx->lm_head_ptr = reinterpret_cast<at::Tensor*>(lm_head_ptr);

        auto* lcfgs = reinterpret_cast<V4LayerConfig*>(layer_configs_ptr);
        for (int64_t i = 0; i < num_layers; i++) {
            ctx->layer_configs.push_back(lcfgs[i]);
        }

        // MTP weights
        if (ctx->has_mtp) {
            auto** mp = reinterpret_cast<at::Tensor**>(mtp_ptrs);
            ctx->mtp.norm = *mp[0];
            ctx->mtp.hnorm = *mp[1];
            ctx->mtp.head = *mp[2];
            ctx->mtp.ffn_norm = *mp[3];
            ctx->mtp.ffn_w1 = *mp[4];
            ctx->mtp.ffn_w2 = *mp[5];
            ctx->mtp.ffn_w3 = *mp[6];
            ctx->embed_for_mtp = *reinterpret_cast<at::Tensor*>(embed_for_mtp_ptr);
        }

        // Create LoRA params (5 per layer: wq_a, wq_b, wkv, wo_a, wo_b)
        int64_t offset = 0;
        for (int64_t i = 0; i < num_layers; i++) {
            ctx->lora_layer_offset.push_back(offset);
            auto& cfg = ctx->layer_configs[i];
            if (!cfg.has_lora) { offset += 0; continue; }

            int64_t w_offset = 0;
            for (int64_t j = 0; j < i; j++)
                w_offset += weight_count_for_layer(ctx->layer_configs[j]);

            // 5 LoRA targets: wq_a=w[2], wq_b=w[3], wkv=w[4], wo_a=w[5], wo_b=w[6]
            int64_t proj_indices[] = {2, 3, 4, 5, 6};
            for (int k = 0; k < 5; k++) {
                auto* base = ctx->weight_ptrs[w_offset + proj_indices[k]];
                int64_t out_f = base->size(0), in_f = base->size(1);
                int64_t rank = 8;  // TODO: from config
                auto a = at::randn({rank, in_f}, at::TensorOptions().dtype(at::kFloat).device(base->device())) * 0.01;
                auto b = at::zeros({out_f, rank}, at::TensorOptions().dtype(at::kFloat).device(base->device()));
                a.set_requires_grad(true);
                b.set_requires_grad(true);
                ctx->lora_a.push_back(std::move(a));
                ctx->lora_b.push_back(std::move(b));
            }
            offset += 5;
        }

        // Adam state
        for (size_t i = 0; i < ctx->lora_a.size(); i++) {
            ctx->adam_m.push_back(at::zeros_like(ctx->lora_a[i]));
            ctx->adam_m.push_back(at::zeros_like(ctx->lora_b[i]));
            ctx->adam_v.push_back(at::zeros_like(ctx->lora_a[i]));
            ctx->adam_v.push_back(at::zeros_like(ctx->lora_b[i]));
        }

        fprintf(stderr, "[v4_ctx] created: %ld layers, %ld LoRA params, MTP=%s\n",
            (long)num_layers, (long)ctx->lora_a.size(), ctx->has_mtp ? "yes" : "no");
        return ctx;
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_create_ctx] FAILED: %s\n", e.what());
        return nullptr;
    }
}

double v4_train_step(void* ctx_ptr, void* input_ids_ptr, void* target_mask_ptr) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        auto& input_ids = *reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto& target_mask = *reinterpret_cast<at::Tensor*>(target_mask_ptr);

        auto hidden = ctx->use_checkpoint
            ? forward_full_checkpoint(ctx, input_ids)
            : forward_full(ctx, input_ids);

        auto loss = compute_loss(hidden, input_ids, target_mask,
            *ctx->final_norm_ptr, *ctx->lm_head_ptr, ctx->rms_eps, ctx->vocab_size);

        // MTP loss
        if (ctx->has_mtp) {
            auto mtp_logits = mtp_forward(hidden, input_ids, ctx->embed_for_mtp, ctx->mtp, ctx->rms_eps);
            auto mlp_loss = mtp_compute_loss(mtp_logits, input_ids, target_mask, ctx->vocab_size);
            loss = loss + mlp_loss;
        }

        double loss_val = loss.item<double>();
        loss.backward();

        // Sync LoRA gradients across ranks (EP mode: attention is replicated)
        at::AutoGradMode guard(false);
        if (ctx->world_size > 1) {
            for (size_t i = 0; i < ctx->lora_a.size(); i++) {
                auto& ga = ctx->lora_a[i].mutable_grad();
                if (ga.defined() && ga.numel() > 0) {
                    sync_all_reduce(ctx->nccl_comm, ctx->comm_stream, ga);
                    ga.div_((double)ctx->world_size);
                }
                auto& gb = ctx->lora_b[i].mutable_grad();
                if (gb.defined() && gb.numel() > 0) {
                    sync_all_reduce(ctx->nccl_comm, ctx->comm_stream, gb);
                    gb.div_((double)ctx->world_size);
                }
            }
        }

        // Adam
        ctx->step_count++;
        double sf = (double)ctx->step_count;
        double bc1 = 1.0 - std::pow(ctx->beta1, sf);
        double bc2 = 1.0 - std::pow(ctx->beta2, sf);

        size_t ai = 0;
        for (size_t i = 0; i < ctx->lora_a.size(); i++) {
            {
                auto& p = ctx->lora_a[i];
                auto& g = p.grad();
                if (g.defined()) {
                    auto& m = ctx->adam_m[ai];
                    auto& v = ctx->adam_v[ai];
                    m = m * ctx->beta1 + g * (1.0 - ctx->beta1);
                    v = v * ctx->beta2 + g * g * (1.0 - ctx->beta2);
                    p.add_((m / bc1) / ((v / bc2).sqrt() + ctx->eps) * (-ctx->lr));
                    p.grad().zero_();
                }
            }
            ai++;
            {
                auto& p = ctx->lora_b[i];
                auto& g = p.grad();
                if (g.defined()) {
                    auto& m = ctx->adam_m[ai];
                    auto& v = ctx->adam_v[ai];
                    m = m * ctx->beta1 + g * (1.0 - ctx->beta1);
                    v = v * ctx->beta2 + g * g * (1.0 - ctx->beta2);
                    p.add_((m / bc1) / ((v / bc2).sqrt() + ctx->eps) * (-ctx->lr));
                    p.grad().zero_();
                }
            }
            ai++;
        }
        return loss_val;
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_train_step] FAILED: %s\n", e.what());
        return -1.0;
    }
}

int64_t v4_get_lora_count(void* ctx_ptr) {
    return (int64_t)reinterpret_cast<TrainingContext*>(ctx_ptr)->lora_a.size();
}

void* v4_get_lora_a(void* ctx_ptr, int64_t idx) {
    return &reinterpret_cast<TrainingContext*>(ctx_ptr)->lora_a[idx];
}

void* v4_get_lora_b(void* ctx_ptr, int64_t idx) {
    return &reinterpret_cast<TrainingContext*>(ctx_ptr)->lora_b[idx];
}

void v4_free_training_context(void* ctx_ptr) {
    if (ctx_ptr) delete reinterpret_cast<TrainingContext*>(ctx_ptr);
}

void v4_set_checkpoint(void* ctx_ptr, int32_t enable, int64_t group_size) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    ctx->use_checkpoint = (enable != 0);
    ctx->group_size = (group_size > 0) ? group_size : 4;
}

void v4_set_nccl_comm(void* ctx_ptr, void* comm, void* stream, int32_t rank, int32_t world_size) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    ctx->nccl_comm = reinterpret_cast<ncclComm_t>(comm);
    ctx->comm_stream = reinterpret_cast<cudaStream_t>(stream);
    ctx->rank = rank;
    ctx->world_size = world_size;
    fprintf(stderr, "[v4_ctx] NCCL set: rank=%d world_size=%d\n", rank, world_size);
}

}  // extern "C"
