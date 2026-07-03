// qwen3_6_kernels.cpp — C++ FFI for Qwen3.6 native kernels
//
// Full C++ training: forward + loss + backward + Adam optimizer.
// LoRA A/B are created in C++ as at::Tensor (requires_grad=true).
// No tch-rs VarStore involved — gradients flow entirely within C++ autograd.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/autograd.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <set>

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────

static at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double eps) {
    auto input_f32 = input.to(at::kFloat);
    auto variance = input_f32.pow(2).mean(-1, true);
    auto inv_rms = (variance + eps).rsqrt();
    // HF Qwen3.5-MoE: output = (x * rsqrt(var+eps)) * (1.0 + weight)
    auto normed = input_f32 * inv_rms;
    return (normed * (1.0 + weight.to(at::kFloat))).to(input.scalar_type());
}

static at::Tensor lora_delta(const at::Tensor& base, const at::Tensor& a, const at::Tensor& b, double scaling) {
    auto kind = base.scalar_type();
    auto delta = b.to(kind).matmul(a.to(kind));
    return base + (delta * scaling).to(kind);
}

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
    double rms_eps, at::ScalarType compute_type
) {
    auto device = hidden.device();
    int64_t batch = hidden.size(0), seq = hidden.size(1);
    int64_t qkv_dim = num_heads * head_dim;
    int64_t rotary_dim = (int64_t)(head_dim * partial_rotary_factor);

    auto q_out = at::matmul(hidden, q_proj.t()).view({batch, seq, num_heads, head_dim * 2});
    auto qk_chunk = q_out.chunk(2, -1);
    auto q = qk_chunk[0].transpose(1, 2);   // [batch, heads, seq, head_dim]
    auto gate = qk_chunk[1].transpose(1, 2); // [batch, heads, seq, head_dim]

    auto k = at::matmul(hidden, k_proj.t()).view({batch, seq, num_kv_heads, head_dim}).transpose(1, 2);
    auto v = at::matmul(hidden, v_proj.t()).view({batch, seq, num_kv_heads, head_dim}).transpose(1, 2);

    q = rms_norm(q, q_norm, rms_eps);
    k = rms_norm(k, k_norm, rms_eps);

    if (rotary_dim > 0) {
        // HF: inv_freq = 1 / (theta ^ (arange(0, dim, 2) / dim))
        auto pos = at::arange(seq, at::TensorOptions().dtype(at::kFloat).device(device)).unsqueeze(0);
        auto exponents = at::arange(0, rotary_dim, 2, at::TensorOptions().dtype(at::kFloat).device(device)) / (double)rotary_dim;
        auto inv_freq = (exponents * std::log(rope_theta)).exp().reciprocal();
        auto freqs = pos.unsqueeze(-1) * inv_freq.unsqueeze(0);
        // HF: emb = cat(freqs, freqs, dim=-1) → cos, sin of [batch, seq, rotary_dim]
        auto emb = at::cat({freqs, freqs}, -1);
        auto cos = emb.cos().unsqueeze(1).to(q.scalar_type());
        auto sin = emb.sin().unsqueeze(1).to(q.scalar_type());

        auto q_rot = q.narrow(-1, 0, rotary_dim);
        auto k_rot = k.narrow(-1, 0, rotary_dim);
        auto q_pass = q.narrow(-1, rotary_dim, head_dim - rotary_dim);
        auto k_pass = k.narrow(-1, rotary_dim, head_dim - rotary_dim);
        auto q_rotated = q_rot * cos + rotate_half(q_rot) * sin;
        auto k_rotated = k_rot * cos + rotate_half(k_rot) * sin;
        q = (head_dim > rotary_dim) ? at::cat({q_rotated, q_pass}, -1) : q_rotated;
        k = (head_dim > rotary_dim) ? at::cat({k_rotated, k_pass}, -1) : k_rotated;
    }

    int64_t n_rep = num_heads / num_kv_heads;
    k = k.repeat_interleave(n_rep, 1);
    v = v.repeat_interleave(n_rep, 1);

    double scale = 1.0 / std::sqrt((double)head_dim);

    // Use SDPA (Flash Attention) — O(seq) memory instead of O(seq²)
    // Pass is_causal=true, no explicit attn_mask needed
    auto attn_out = at::scaled_dot_product_attention(
        q, k, v,
        /*attn_mask=*/c10::nullopt,  // no explicit mask
        0.0,  // dropout_p
        true   // is_causal
    );
    // Apply scale manually since SDPA uses 1/sqrt(head_dim) by default
    // Actually SDPA already applies 1/sqrt(E) scaling, but our scale matches
    // since head_dim == E. So no extra scaling needed.

    attn_out = attn_out * at::sigmoid(gate).to(attn_out.scalar_type());
    return attn_out.transpose(1, 2).reshape({batch, seq, qkv_dim}).matmul(o_proj.t());
}

// ──────────────────────────────────────────────────────────────────────
// Linear attention (Gated Delta Rule — matrix formulation)
// ──────────────────────────────────────────────────────────────────────

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

    // Optimized Gated Delta Rule — reduce kernel launches via bmm
    // The recurrence is inherently sequential (state update depends on previous token),
    // but we minimize per-token overhead by using bmm for the matrix operations
    // instead of select + elementwise mul + sum.
    //
    // Key optimization: pre-reshape to [B*H, S, D] format so each per-token
    // operation is a single bmm call rather than multiple select/mul/sum.
    auto g_exp = g_t.exp();  // [B, H, S]
    int64_t BH = batch * num_v_heads;
    auto state = at::zeros({BH, key_dim, val_dim}, q_t.options());  // [B*H, D_k, D_v]

    // Pre-reshape to [B*H, S, D] for efficient indexing
    auto q_bh = q_t.reshape({BH, seq, key_dim});     // [B*H, S, D_k]
    auto k_bh = k_t.reshape({BH, seq, key_dim});     // [B*H, S, D_k]
    auto v_bh = v_t.reshape({BH, seq, val_dim});     // [B*H, S, D_v]
    auto g_bh = g_exp.reshape({BH, seq});             // [B*H, S]
    auto beta_bh = beta_t.reshape({BH, seq});        // [B*H, S]

    auto outs = at::empty({BH, seq, val_dim}, q_t.options());  // [B*H, S, D_v]

    for (int64_t i = 0; i < seq; i++) {
        // k_i: [B*H, 1, D_k], v_i: [B*H, 1, D_v], q_i: [B*H, 1, D_k]
        auto k_i = k_bh.select(1, i).unsqueeze(1);  // [B*H, 1, D_k]
        auto v_i = v_bh.select(1, i).unsqueeze(1);  // [B*H, 1, D_v]
        auto q_i = q_bh.select(1, i).unsqueeze(1);  // [B*H, 1, D_k]
        auto g_i = g_bh.select(1, i).reshape({BH, 1, 1});  // [B*H, 1, 1]
        auto beta_i = beta_bh.select(1, i).reshape({BH, 1, 1});  // [B*H, 1, 1]

        // S = S * g_i (decay) — elementwise, [B*H, D_k, D_v]
        state = state * g_i;

        // kv_mem = bmm(k_i, S) → [B*H, 1, D_v]
        auto kv_mem = at::bmm(k_i, state);  // [B*H, 1, D_v]

        // delta = (v_i - kv_mem) * beta_i
        auto delta = (v_i - kv_mem) * beta_i;  // [B*H, 1, D_v]

        // S += k_i^T @ delta → outer product via bmm
        // k_i^T: [B*H, D_k, 1], delta: [B*H, 1, D_v] → [B*H, D_k, D_v]
        state = state + at::bmm(k_i.transpose(1, 2), delta);

        // out_i = bmm(q_i, S) → [B*H, 1, D_v]
        auto out_i = at::bmm(q_i, state);  // [B*H, 1, D_v]

        // Store output
        outs.select(1, i).copy_(out_i.squeeze(1));
    }

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
    auto activated = at::silu(gate_out) * up_out;
    return at::matmul(activated, down_proj.t()).reshape({batch, seq, hidden_dim});
}

// ──────────────────────────────────────────────────────────────────────
// MoE
// ──────────────────────────────────────────────────────────────────────

static at::Tensor moe_forward(
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

    auto router_logits = at::matmul(flat, gate_w.t());
    auto routing_weights = router_logits.softmax(-1, at::kFloat);  // FP32 for precision (matches Rust)
    auto [topk_weights, topk_indices] = routing_weights.topk(top_k, -1, true, true);
    if (norm_topk_prob) {
        auto denom = topk_weights.sum(-1, true).clamp_min(1e-9);
        topk_weights = topk_weights / denom;
    }
    topk_weights = topk_weights.to(compute_type);

    auto routed_output = at::zeros(flat.sizes(), flat.options());
    for (int64_t kk = 0; kk < top_k; kk++) {
        auto expert_indices = topk_indices.select(-1, kk);
        auto expert_weights = topk_weights.select(-1, kk);
        for (int64_t e_local = 0; e_local < expert_count; e_local++) {
            int64_t e_global = expert_start + e_local;
            auto mask = expert_indices.eq(e_global).to(compute_type);
            if (mask.sum().item<double>() > 0.0) {
                auto token_indices = mask.nonzero().squeeze(-1);
                if (token_indices.size(0) == 0) continue;
                auto selected = flat.index_select(0, token_indices);
                auto egu = experts_gate_up.select(0, e_local);
                auto ed = experts_down.select(0, e_local);
                auto gu = at::matmul(selected, egu.t());
                auto gate_part = gu.narrow(-1, 0, intermediate);
                auto up_part = gu.narrow(-1, intermediate, intermediate);
                auto expert_out = at::matmul(at::silu(gate_part) * up_part, ed.t());
                auto weights = expert_weights.index_select(0, token_indices).unsqueeze(-1);
                routed_output = routed_output.index_add_(0, token_indices, expert_out * weights);
            }
        }
    }

    auto shared_out = at::matmul(at::silu(at::matmul(flat, shared_gate_proj.t())) * at::matmul(flat, shared_up_proj.t()), shared_down_proj.t());
    auto seg = at::sigmoid(at::matmul(flat, shared_expert_gate_w.t())).to(compute_type);
    shared_out = (shared_out * seg).to(compute_type);
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
    const at::Tensor& hidden, at::Tensor** w, const LayerConfig* cfg,
    at::ScalarType kind, double lora_scaling,
    at::Tensor** la, at::Tensor** lb
) {
    auto input_norm = *w[0];
    auto post_norm = *w[1];
    auto attn_input = rms_norm(hidden, input_norm, cfg->rms_eps);
    bool is_moe = (cfg->num_experts > 0);

    at::Tensor attn_output;
    if (cfg->layer_type == 0) {
        // Full attention
        auto q_proj = *w[2], q_norm = *w[3], k_proj = *w[4], k_norm = *w[5], v_proj = *w[6], o_proj = *w[7];
        if (la && la[0] && lb && lb[0]) {
            if (la[0] && lb[0]) q_proj = lora_delta(q_proj, *la[0], *lb[0], lora_scaling);
            if (la[1] && lb[1]) k_proj = lora_delta(k_proj, *la[1], *lb[1], lora_scaling);
            if (la[2] && lb[2]) v_proj = lora_delta(v_proj, *la[2], *lb[2], lora_scaling);
            if (la[3] && lb[3]) o_proj = lora_delta(o_proj, *la[3], *lb[3], lora_scaling);
        }
        attn_output = full_attention(attn_input, q_proj, q_norm, k_proj, k_norm, v_proj, o_proj,
            cfg->num_heads, cfg->num_kv_heads, cfg->head_dim,
            cfg->partial_rotary_factor, cfg->rope_theta, cfg->rms_eps, kind);
        auto post_attn = rms_norm(hidden + attn_output, post_norm, cfg->rms_eps);
        if (is_moe) {
            auto mlp_out = moe_forward(post_attn,
                *w[8], *w[9], *w[10], *w[11], *w[12], *w[13], *w[14],
                cfg->num_experts, cfg->top_k, cfg->moe_intermediate,
                cfg->norm_topk_prob != 0, cfg->expert_start, cfg->expert_count, kind);
            return hidden + attn_output + mlp_out;
        } else {
            auto mlp_out = dense_mlp_forward(post_attn, *w[8], *w[9], *w[10], kind);
            return hidden + attn_output + mlp_out;
        }
    } else {
        // Linear attention
        auto in_proj_qkv = *w[2], in_proj_z = *w[3], in_proj_a = *w[4], in_proj_b = *w[5];
        auto a_log = *w[6], dt_bias = *w[7], conv1d_w = *w[8], norm_w = *w[9], out_proj = *w[10];
        if (la && la[0] && lb && lb[0]) {
            if (la[0] && lb[0]) in_proj_qkv = lora_delta(in_proj_qkv, *la[0], *lb[0], lora_scaling);
            if (la[1] && lb[1]) in_proj_z = lora_delta(in_proj_z, *la[1], *lb[1], lora_scaling);
            if (la[2] && lb[2]) out_proj = lora_delta(out_proj, *la[2], *lb[2], lora_scaling);
        }
        attn_output = linear_attention(attn_input, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
            a_log, dt_bias, conv1d_w, norm_w, out_proj,
            cfg->num_k_heads, cfg->key_dim, cfg->num_v_heads, cfg->val_dim,
            cfg->conv_kernel, cfg->rms_eps, kind);
        auto post_attn = rms_norm(hidden + attn_output, post_norm, cfg->rms_eps);
        if (is_moe) {
            auto mlp_out = moe_forward(post_attn,
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
// Training Context — all state lives in C++
// ──────────────────────────────────────────────────────────────────────

struct TrainingContext {
    // Model weights (frozen, no grad) — pointers to external tensors
    std::vector<at::Tensor*> weight_ptrs;  // flat array, 15 or 18 per layer
    std::vector<at::Tensor*> embed_ptr;
    std::vector<at::Tensor*> final_norm_ptr;
    std::vector<at::Tensor*> lm_head_ptr;
    std::vector<LayerConfig> layer_configs;
    int64_t num_layers;

    // LoRA parameters (owned in C++, requires_grad=true)
    std::vector<at::Tensor> lora_a;  // one per (layer, module)
    std::vector<at::Tensor> lora_b;
    std::vector<int64_t> lora_layer_offset;  // offset into lora_a/b per layer
    double lora_scaling;
    std::vector<std::string> lora_names;  // for saving

    // Adam optimizer state
    std::vector<at::Tensor> adam_m;
    std::vector<at::Tensor> adam_v;
    double lr, beta1, beta2, eps;
    int64_t step_count;

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
};

// Forward pass (no checkpointing)
static at::Tensor forward_full(
    TrainingContext* ctx,
    const at::Tensor& input_ids
) {
    auto kind = ctx->compute_type;
    auto embed = *ctx->embed_ptr[0];
    auto final_norm = *ctx->final_norm_ptr[0];

    at::AutoGradMode guard(true);
    at::Tensor hidden = at::embedding(embed, input_ids);

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

        hidden = forward_single_layer(hidden, layer_w.data(), &ctx->layer_configs[i],
            kind, ctx->lora_scaling, la_ptrs.data(), lb_ptrs.data());
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

        hidden = forward_single_layer(hidden, layer_w.data(), &ctx->layer_configs[i],
            kind, ctx->lora_scaling, la_ptrs.data(), lb_ptrs.data());
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
        ctx->save_for_backward({input});

        // Run forward WITH grad — the Function will only save the input (not intermediate activations).
        // The key is that save_for_backward only stores the input tensor.
        // We don't use no-grad here because the LoRA params need to build a graph for backward.
        auto* tc = reinterpret_cast<TrainingContext*>(tc_val);
        return forward_layer_group(tc, input, start_layer, end_layer);
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output
    ) {
        auto saved = ctx->get_saved_variables();
        at::Tensor input = saved[0];

        auto tc = reinterpret_cast<TrainingContext*>(ctx->saved_data["tc"].toInt());
        int64_t start_layer = ctx->saved_data["start"].toInt();
        int64_t end_layer = ctx->saved_data["end"].toInt();

        // Recompute forward WITH grad enabled, backprop through recomputed graph.
        // Must explicitly enable autograd since backward runs in no-grad context.
        at::AutoGradMode guard(true);
        input.set_requires_grad(true);
        auto output = forward_layer_group(tc, input, start_layer, end_layer);

        // Backprop through recomputed graph — gradients accumulate into LoRA params.
        // Use backward with retain_graph to avoid freeing graph that other checkpoint
        // groups' backward functions still need.
        torch::autograd::backward({output}, {grad_output[0]},
            /*retain_graph=*/true, /*create_graph=*/false);
        return {input.grad(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

// Forward pass with gradient checkpointing
static at::Tensor forward_full_checkpoint(
    TrainingContext* ctx,
    const at::Tensor& input_ids
) {
    auto embed = *ctx->embed_ptr[0];
    at::Tensor hidden = at::embedding(embed, input_ids);

    // Detach and set requires_grad so autograd::Function can track the graph
    hidden = hidden.detach().set_requires_grad(true);

    int64_t gs = ctx->group_size;
    if (gs < 1) gs = 1;

    for (int64_t start = 0; start < ctx->num_layers; start += gs) {
        int64_t end = std::min(start + gs, ctx->num_layers);
        hidden = GroupCheckpointFunction::apply(
            hidden,
            (int64_t)(uintptr_t)ctx,
            start,
            end
        );
    }

    return hidden;
}

// Cross-entropy loss with response-only masking — chunked to avoid full logits in memory
// Uses at::cross_entropy_loss (fused log_softmax + nll_loss) for efficiency
static at::Tensor compute_loss(
    TrainingContext* ctx,
    const at::Tensor& hidden,
    const at::Tensor& input_ids,
    const at::Tensor& target_mask,
    int64_t vocab_size
) {
    auto kind = ctx->compute_type;
    auto final_norm = *ctx->final_norm_ptr[0];
    auto lm_head = *ctx->lm_head_ptr[0];

    auto hidden_normed = rms_norm(hidden, final_norm, ctx->rms_eps);

    int64_t seq_len = hidden_normed.size(1);
    // Shift: logits[t] predicts token t+1
    auto shifted_hidden = hidden_normed.narrow(1, 0, seq_len - 1);  // [batch, seq-1, hidden]
    auto shifted_targets = input_ids.narrow(1, 1, seq_len - 1).reshape({-1});
    auto shifted_mask = target_mask.narrow(1, 1, seq_len - 1).reshape({-1});

    // Chunked cross-entropy: process tokens in chunks to limit peak memory
    int64_t total_tokens = shifted_targets.size(0);
    int64_t chunk_size = 512;
    int64_t num_chunks = (total_tokens + chunk_size - 1) / chunk_size;

    auto total_loss = at::zeros({1}, at::TensorOptions().dtype(at::kFloat).device(hidden_normed.device()));
    auto total_count = shifted_mask.sum().clamp_min(1.0);
    auto hidden_flat = shifted_hidden.reshape({-1, hidden_normed.size(2)});

    for (int64_t c = 0; c < num_chunks; c++) {
        int64_t start = c * chunk_size;
        int64_t end = std::min(start + chunk_size, total_tokens);
        int64_t n = end - start;

        // Compute logits for this chunk only: [n, vocab]
        auto chunk_hidden = hidden_flat.narrow(0, start, n);
        auto chunk_logits = at::matmul(chunk_hidden, lm_head.t());  // [n, vocab]

        auto chunk_targets = shifted_targets.narrow(0, start, n);
        auto chunk_mask = shifted_mask.narrow(0, start, n);

        // Fused cross-entropy (log_softmax + nll_loss in one kernel)
        auto per_token_loss = at::cross_entropy_loss(
            chunk_logits.to(at::kFloat), chunk_targets,
            /*weight=*/at::Tensor(), /*reduction=*/at::Reduction::None,
            /*ignore_index=*/-100, /*label_smoothing=*/0.0
        );
        auto masked_loss = per_token_loss * chunk_mask.to(at::kFloat);
        total_loss += masked_loss.sum();
    }

    return total_loss / total_count;
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
        h = forward_single_layer(h, layer_w.data(), &ctx->mtp_layer_configs[i],
            kind, ctx->lora_scaling, nullptr, nullptr);
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
    int64_t chunk_size = 512;
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
void* qwen36_create_training_context(
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
                    auto a = at::randn({lora_rank, in_f}, at::TensorOptions().dtype(at::kFloat).device(base->device())) * 0.01;
                    auto b = at::zeros({out_f, lora_rank}, at::TensorOptions().dtype(at::kFloat).device(base->device()));
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
                    auto a = at::randn({lora_rank, in_f}, at::TensorOptions().dtype(at::kFloat).device(base->device())) * 0.01;
                    auto b = at::zeros({out_f, lora_rank}, at::TensorOptions().dtype(at::kFloat).device(base->device()));
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

        // Initialize Adam state (zeros for each LoRA param)
        for (size_t i = 0; i < ctx->lora_a.size(); i++) {
            ctx->adam_m.push_back(at::zeros_like(ctx->lora_a[i]));
            ctx->adam_m.push_back(at::zeros_like(ctx->lora_b[i]));
            ctx->adam_v.push_back(at::zeros_like(ctx->lora_a[i]));
            ctx->adam_v.push_back(at::zeros_like(ctx->lora_b[i]));
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
void qwen36_set_mtp_weights(
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
double qwen36_train_step(
    void* ctx_ptr,
    void* input_ids_ptr,
    void* target_mask_ptr
) {
    try {
        auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
        auto& input_ids = *reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto& target_mask = *reinterpret_cast<at::Tensor*>(target_mask_ptr);

        // Forward (with or without checkpointing)
        auto hidden = ctx->use_checkpoint
            ? forward_full_checkpoint(ctx, input_ids)
            : forward_full(ctx, input_ids);

        // Main loss (cross-entropy on shifted tokens)
        auto loss = compute_loss(ctx, hidden, input_ids, target_mask, ctx->vocab_size);

        // MTP loss (if enabled and not disabled by env)
        if (ctx->has_mtp && !getenv("QWEN36_DISABLE_MTP")) {
            auto mtp_hidden = mtp_forward(ctx, hidden, input_ids);
            auto mtp_loss = mtp_compute_loss(ctx, mtp_hidden, input_ids, target_mask);
            // Print both losses for debugging
            if (ctx->step_count == 0) {
                fprintf(stderr, "[mtp_debug] main_loss=%.4f mtp_loss=%.4f (x0.5=%.4f) total=%.4f\n",
                    loss.item<double>(), mtp_loss.item<double>() / 0.5, mtp_loss.item<double>(),
                    (loss + mtp_loss).item<double>());
            }
            loss = loss + mtp_loss;
        }

        double loss_val = loss.item<double>();

        // Backward — gradients accumulate into LoRA A/B (requires_grad=true)
        loss.backward();

        // Adam optimizer step (no grad)
        at::AutoGradMode guard(false);
        ctx->step_count++;
        double step_f = (double)ctx->step_count;
        double bias_correction1 = 1.0 - std::pow(ctx->beta1, step_f);
        double bias_correction2 = 1.0 - std::pow(ctx->beta2, step_f);

        size_t adam_idx = 0;
        for (size_t i = 0; i < ctx->lora_a.size(); i++) {
            // Update LoRA A
            {
                auto& param = ctx->lora_a[i];
                auto& grad = param.grad();
                if (grad.defined()) {
                    auto& m = ctx->adam_m[adam_idx];
                    auto& v = ctx->adam_v[adam_idx];
                    m = m * ctx->beta1 + grad * (1.0 - ctx->beta1);
                    v = v * ctx->beta2 + grad * grad * (1.0 - ctx->beta2);
                    auto mh = m / bias_correction1;
                    auto vh = v / bias_correction2;
                    param.add_((mh / (vh.sqrt() + ctx->eps)) * -ctx->lr);
                    param.grad().zero_();
                }
            }
            adam_idx++;
            // Update LoRA B
            {
                auto& param = ctx->lora_b[i];
                auto& grad = param.grad();
                if (grad.defined()) {
                    auto& m = ctx->adam_m[adam_idx];
                    auto& v = ctx->adam_v[adam_idx];
                    m = m * ctx->beta1 + grad * (1.0 - ctx->beta1);
                    v = v * ctx->beta2 + grad * grad * (1.0 - ctx->beta2);
                    auto mh = m / bias_correction1;
                    auto vh = v / bias_correction2;
                    param.add_((mh / (vh.sqrt() + ctx->eps)) * -ctx->lr);
                    param.grad().zero_();
                }
            }
            adam_idx++;
        }

        return loss_val;
    } catch (const std::exception& e) {
        fprintf(stderr, "[q36_train_step] FAILED: %s\n", e.what());
        return -1.0;
    }
}

// Get LoRA parameter count
int64_t qwen36_get_lora_count(void* ctx_ptr) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    return (int64_t)ctx->lora_a.size();
}

// Get LoRA A tensor pointer by index
void* qwen36_get_lora_a(void* ctx_ptr, int64_t index) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    return &ctx->lora_a[index];
}

// Get LoRA B tensor pointer by index
void* qwen36_get_lora_b(void* ctx_ptr, int64_t index) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    return &ctx->lora_b[index];
}

// Free training context
void qwen36_free_training_context(void* ctx_ptr) {
    if (ctx_ptr) {
        delete reinterpret_cast<TrainingContext*>(ctx_ptr);
    }
}

// Enable/disable gradient checkpointing
void qwen36_set_checkpoint(void* ctx_ptr, int32_t enable, int64_t group_size) {
    auto* ctx = reinterpret_cast<TrainingContext*>(ctx_ptr);
    ctx->use_checkpoint = (enable != 0);
    ctx->group_size = (group_size > 0) ? group_size : 4;
    fprintf(stderr, "[q36_ctx] checkpoint: %s, group_size=%ld\n",
        ctx->use_checkpoint ? "ON" : "OFF", (long)ctx->group_size);
}

// Utility functions (kept for compatibility)
void* qwen36_gemm(void* a_ptr, void* b_ptr, int transpose_b) {
    auto& a = *reinterpret_cast<at::Tensor*>(a_ptr);
    auto& b = *reinterpret_cast<at::Tensor*>(b_ptr);
    if (transpose_b) return new at::Tensor(at::matmul(a, b.t()));
    return new at::Tensor(at::matmul(a, b));
}

void qwen36_free_tensor(void* tensor_ptr) {
    if (tensor_ptr) delete reinterpret_cast<at::Tensor*>(tensor_ptr);
}

}  // extern "C"
