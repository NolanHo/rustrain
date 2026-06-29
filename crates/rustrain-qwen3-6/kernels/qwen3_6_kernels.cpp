// qwen3_6_kernels.cpp — C++ FFI for Qwen3.6 native kernels
//
// Full C++ layer forward + gradient checkpointing.
// All forward logic (RMSNorm, full/linear attention, MoE, LoRA delta) is in C++
// to preserve autograd graph across checkpoint boundaries.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/autograd.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

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
    // Cast LoRA A/B from FP32 (VarStore) to base weight dtype (BF16) for matmul
    auto kind = base.scalar_type();
    auto delta = b.to(kind).matmul(a.to(kind));
    return base + (delta * scaling).to(kind);
}

// ──────────────────────────────────────────────────────────────────────
// Full attention (with MRoPE + partial rotary + output gate)
// ──────────────────────────────────────────────────────────────────────

static at::Tensor rotate_half_interleaved(const at::Tensor& x) {
    auto n = x.size(-1);
    auto half = n / 2;
    // Preserve leading dimensions: [..., n] → [..., half, 2] → [..., n]
    std::vector<int64_t> shape;
    for (int i = 0; i < x.dim() - 1; i++) shape.push_back(x.size(i));
    shape.push_back(half);
    shape.push_back(2);
    auto x_pairs = x.reshape(shape);
    auto rotated = at::stack({x_pairs.select(-1, 1).neg(), x_pairs.select(-1, 0)}, -1);
    return rotated.flatten(-2, -1);
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

    auto q_out = at::matmul(hidden, q_proj.t());  // [batch, seq, 2*qkv_dim]
    auto q = q_out.narrow(-1, 0, qkv_dim).view({batch, seq, num_heads, head_dim}).transpose(1, 2);
    auto gate = q_out.narrow(-1, qkv_dim, qkv_dim).view({batch, seq, num_heads, head_dim}).transpose(1, 2);

    auto k = at::matmul(hidden, k_proj.t()).view({batch, seq, num_kv_heads, head_dim}).transpose(1, 2);
    auto v = at::matmul(hidden, v_proj.t()).view({batch, seq, num_kv_heads, head_dim}).transpose(1, 2);

    q = rms_norm(q, q_norm, rms_eps);
    k = rms_norm(k, k_norm, rms_eps);

    // RoPE
    if (rotary_dim > 0) {
        int64_t half = rotary_dim / 2;
        auto pos = at::arange(seq, at::TensorOptions().dtype(at::kFloat).device(device)).unsqueeze(0);
        auto exponents = at::arange(half, at::TensorOptions().dtype(at::kFloat).device(device)) * (2.0 / rotary_dim);
        auto inv_freq = (exponents * std::log(rope_theta)).exp().reciprocal();
        auto freqs = pos.unsqueeze(-1) * inv_freq.unsqueeze(0);
        auto cos = freqs.cos().unsqueeze(1);
        auto sin = freqs.sin().unsqueeze(1);
        auto cos_full = at::stack({cos, cos}, -1).flatten(-2, -1).narrow(-1, 0, rotary_dim).to(q.scalar_type());
        auto sin_full = at::stack({sin, sin}, -1).flatten(-2, -1).narrow(-1, 0, rotary_dim).to(q.scalar_type());

        auto q_rot = q.narrow(-1, 0, rotary_dim);
        auto k_rot = k.narrow(-1, 0, rotary_dim);
        auto q_pass = q.narrow(-1, rotary_dim, head_dim - rotary_dim);
        auto k_pass = k.narrow(-1, rotary_dim, head_dim - rotary_dim);

        auto q_rotated = q_rot * cos_full + rotate_half_interleaved(q_rot) * sin_full;
        auto k_rotated = k_rot * cos_full + rotate_half_interleaved(k_rot) * sin_full;

        q = (head_dim > rotary_dim) ? at::cat({q_rotated, q_pass}, -1) : q_rotated;
        k = (head_dim > rotary_dim) ? at::cat({k_rotated, k_pass}, -1) : k_rotated;
    }

    // GQA
    int64_t n_rep = num_heads / num_kv_heads;
    k = k.repeat_interleave(n_rep, 1);
    v = v.repeat_interleave(n_rep, 1);

    // Attention
    double scale = 1.0 / std::sqrt((double)head_dim);
    auto attn_weights = at::matmul(q * scale, k.transpose(-1, -2));
    auto causal_mask = at::ones({seq, seq}, at::TensorOptions().dtype(at::kBool).device(device)).triu(1);
    attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -std::numeric_limits<float>::infinity());
    auto attn_out = attn_weights.softmax(-1).matmul(v);

    // Output gate
    attn_out = attn_out * at::sigmoid(gate).to(attn_out.scalar_type());

    // Output projection
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

    // QKV projection
    auto qkv = at::matmul(hidden, in_proj_qkv.t());  // [batch, seq, qkv_dim]

    // Conv1d (causal, SiLU)
    auto qkv_t = qkv.transpose(1, 2);  // [batch, qkv_dim, seq]
    int64_t pad = conv_kernel - 1;
    auto padding = at::zeros({batch, qkv_dim, pad}, qkv.options());
    auto padded = at::cat({padding, qkv_t}, 2);
    auto conv_out = at::conv1d(padded, conv1d_weight, /*bias=*/{},
        /*stride=*/at::IntArrayRef({1}), /*padding=*/at::IntArrayRef({0}), /*dilation=*/at::IntArrayRef({1}), /*groups=*/qkv_dim);
    conv_out = at::silu(conv_out.narrow(2, 0, seq));
    auto qkv_conv = conv_out.transpose(1, 2);  // [batch, seq, qkv_dim]

    // Split Q, K, V
    auto q = qkv_conv.narrow(-1, 0, q_size).view({batch, seq, num_k_heads, key_dim});
    auto k = qkv_conv.narrow(-1, q_size, q_size).view({batch, seq, num_k_heads, key_dim});
    auto v = qkv_conv.narrow(-1, q_size * 2, v_size).view({batch, seq, num_v_heads, val_dim});

    // Project A, B, Z
    auto a = at::matmul(hidden, in_proj_a.t());
    auto b = at::matmul(hidden, in_proj_b.t());
    auto z = at::matmul(hidden, in_proj_z.t()).view({batch, seq, num_v_heads, val_dim});

    // g = -exp(A_log) * softplus(a + dt_bias)
    auto a_log_f = a_log.to(at::kFloat);
    auto dt_bias_f = dt_bias.to(at::kFloat);
    auto a_f = a.to(at::kFloat);
    auto g = a_log_f.unsqueeze(0).unsqueeze(0).exp().neg() * at::softplus(a_f + dt_bias_f.unsqueeze(0).unsqueeze(0));
    g = g.to(compute_type);
    auto beta = at::sigmoid(b).to(compute_type);

    // Repeat K heads to V heads
    int64_t n_rep = num_v_heads / num_k_heads;
    q = q.repeat_interleave(n_rep, 2);
    k = k.repeat_interleave(n_rep, 2);

    // L2 normalize
    q = (q / q.norm(2, -1, true).clamp_min(1e-6)).to(compute_type);
    k = (k / k.norm(2, -1, true).clamp_min(1e-6)).to(compute_type);

    // Matrix GDN: A[i,j] = ratio[i,j] * beta[j] * QK[i,j] for j<=i
    auto q_t = q.transpose(1, 2);  // [batch, heads, seq, key_dim]
    auto k_t = k.transpose(1, 2);
    auto v_t = v.transpose(1, 2);
    auto g_t = g.transpose(1, 2);  // [batch, heads, seq]
    auto beta_t = beta.transpose(1, 2);

    auto qk = at::matmul(q_t, k_t.transpose(-1, -2));
    auto g_abs = g_t.abs().clamp_min(1e-20);
    auto cum_log_g = g_abs.log().cumsum(2);
    auto cum_sign = g_t.sign().cumprod(2);
    auto log_ratio = (cum_log_g.unsqueeze(-1) - cum_log_g.unsqueeze(-2)).clamp_max(50.0);
    auto ratio = log_ratio.exp() * (cum_sign.unsqueeze(-1) * cum_sign.unsqueeze(-2));
    auto mask = at::ones({seq, seq}, q_t.options()).tril(0);
    auto attn = (ratio * beta_t.unsqueeze(-2).to(at::kFloat) * qk.to(at::kFloat)) * mask;
    attn = attn.to(q_t.scalar_type());
    auto core_out = at::matmul(attn, v_t);  // [batch, heads, seq, val_dim]

    // Gated RMSNorm
    core_out = core_out.transpose(1, 2);  // [batch, seq, heads, val_dim]
    auto core_flat = core_out.reshape({-1, val_dim});
    auto z_flat = z.reshape({-1, val_dim});
    auto variance = core_flat.to(at::kFloat).pow(2).mean(-1, true);
    auto normed = (core_flat.to(at::kFloat) * (variance + rms_eps).rsqrt() * norm_w.to(at::kFloat)).to(core_flat.scalar_type());
    auto gated = normed * z_flat;
    auto normed_out = gated.view({batch, seq, num_v_heads * val_dim});

    return at::matmul(normed_out, out_proj.t());
}

// ──────────────────────────────────────────────────────────────────────
// MoE (routed + shared expert + gate)
// ──────────────────────────────────────────────────────────────────────

static at::Tensor moe_forward(
    const at::Tensor& hidden,
    const at::Tensor& gate_w, const at::Tensor& shared_expert_gate_w,
    const at::Tensor& shared_gate_proj, const at::Tensor& shared_up_proj, const at::Tensor& shared_down_proj,
    const at::Tensor& experts_gate_up, const at::Tensor& experts_down,
    int64_t num_experts, int64_t top_k, int64_t intermediate,
    bool norm_topk_prob, double router_aux_loss_coef,
    int64_t expert_start, int64_t expert_count,
    at::ScalarType compute_type
) {
    int64_t batch = hidden.size(0), seq = hidden.size(1), hidden_dim = hidden.size(2);
    auto device = hidden.device();
    auto flat = hidden.reshape({batch * seq, hidden_dim});

    // Router
    auto router_logits = at::matmul(flat, gate_w.t());
    auto routing_weights = router_logits.softmax(-1);
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
                auto weighted = expert_out * weights;
                routed_output = routed_output.index_add_(0, token_indices, weighted);
            }
        }
    }

    // Shared expert
    auto shared_gate = at::matmul(flat, shared_gate_proj.t());
    auto shared_up = at::matmul(flat, shared_up_proj.t());
    auto shared_out = at::matmul(at::silu(shared_gate) * shared_up, shared_down_proj.t());
    auto seg_logit = at::matmul(flat, shared_expert_gate_w.t());
    auto seg = at::sigmoid(seg_logit).to(compute_type);
    shared_out = (shared_out * seg).to(compute_type);

    return (routed_output + shared_out).reshape({batch, seq, hidden_dim});
}

// ──────────────────────────────────────────────────────────────────────
// Single layer forward (dispatches full vs linear attention)
// ──────────────────────────────────────────────────────────────────────

// Layer config struct passed from Rust
struct LayerConfig {
    int64_t layer_type;      // 0=full, 1=linear
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t head_dim;
    int64_t num_k_heads;
    int64_t key_dim;
    int64_t num_v_heads;
    int64_t val_dim;
    int64_t conv_kernel;
    double partial_rotary_factor;
    double rope_theta;
    double rms_eps;
    int64_t num_experts;
    int64_t top_k;
    int64_t moe_intermediate;
    bool norm_topk_prob;
    int64_t expert_start;
    int64_t expert_count;
};

// Weight pointers for a single layer (passed as void** array from Rust)
// Full attention: [0]=input_norm, [1]=post_norm, [2]=q_proj, [3]=q_norm, [4]=k_proj, [5]=k_norm, [6]=v_proj, [7]=o_proj
// Linear attn:    [0]=input_norm, [1]=post_norm, [2]=in_proj_qkv, [3]=in_proj_z, [4]=in_proj_a, [5]=in_proj_b, [6]=A_log, [7]=dt_bias, [8]=conv1d, [9]=norm, [10]=out_proj
// MoE (same for both): gate, shared_expert_gate, shared_gate_proj, shared_up_proj, shared_down_proj, experts_gate_up, experts_down

static at::Tensor forward_single_layer_cpp(
    const at::Tensor& hidden,
    void** weight_ptrs,
    const LayerConfig* cfg,
    at::ScalarType compute_type,
    double lora_scaling,
    void** lora_a_ptrs,  // may be null
    void** lora_b_ptrs
) {
    auto kind = compute_type;
    auto** w = reinterpret_cast<at::Tensor**>(weight_ptrs);

    auto input_norm = *w[0];
    auto post_norm = *w[1];

    auto attn_input = rms_norm(hidden, input_norm, cfg->rms_eps);

    at::Tensor attn_output;
    if (cfg->layer_type == 0) {
        // Full attention
        auto q_proj = *w[2];
        auto q_norm = *w[3];
        auto k_proj = *w[4];
        auto k_norm = *w[5];
        auto v_proj = *w[6];
        auto o_proj = *w[7];

        // Apply LoRA to all full-attn target modules
        if (lora_a_ptrs && lora_a_ptrs[0]) {
            auto** la = reinterpret_cast<at::Tensor**>(lora_a_ptrs);
            auto** lb = reinterpret_cast<at::Tensor**>(lora_b_ptrs);
            // LoRA pointer layout: [q_a, k_a, v_a, o_a, q_b, k_b, v_b, o_b]
            // Or if only q_proj: [q_a, q_b]
            // We check each pointer — if null, skip
            if (la[0] && lb[0]) q_proj = lora_delta(q_proj, *la[0], *lb[0], lora_scaling);
            if (la[1] && lb[1]) k_proj = lora_delta(k_proj, *la[1], *lb[1], lora_scaling);
            if (la[2] && lb[2]) v_proj = lora_delta(v_proj, *la[2], *lb[2], lora_scaling);
            if (la[3] && lb[3]) o_proj = lora_delta(o_proj, *la[3], *lb[3], lora_scaling);
        }

        int64_t moe_offset = 8;
        attn_output = full_attention(
            attn_input, q_proj, q_norm, k_proj, k_norm, v_proj, o_proj,
            cfg->num_heads, cfg->num_kv_heads, cfg->head_dim,
            cfg->partial_rotary_factor, cfg->rope_theta, cfg->rms_eps, kind
        );

        // MoE weights start at offset 8
        auto moe_output = moe_forward(
            rms_norm(hidden + attn_output, post_norm, cfg->rms_eps),
            *w[8], *w[9], *w[10], *w[11], *w[12],
            *w[13], *w[14],
            cfg->num_experts, cfg->top_k, cfg->moe_intermediate,
            cfg->norm_topk_prob, 0.001, cfg->expert_start, cfg->expert_count, kind
        );
        return (hidden + attn_output + moe_output);
    } else {
        // Linear attention
        auto in_proj_qkv = *w[2];
        auto in_proj_z = *w[3];
        auto in_proj_a = *w[4];
        auto in_proj_b = *w[5];
        auto a_log = *w[6];
        auto dt_bias = *w[7];
        auto conv1d_w = *w[8];
        auto norm_w = *w[9];
        auto out_proj = *w[10];

        // Apply LoRA to linear_attn target modules
        if (lora_a_ptrs && lora_a_ptrs[0]) {
            auto** la = reinterpret_cast<at::Tensor**>(lora_a_ptrs);
            auto** lb = reinterpret_cast<at::Tensor**>(lora_b_ptrs);
            // For linear_attn, LoRA targets are: [in_proj_qkv, in_proj_z, out_proj]
            // Mapped to indices 0, 1, 2 in the LoRA pointer array
            if (la[0] && lb[0]) in_proj_qkv = lora_delta(in_proj_qkv, *la[0], *lb[0], lora_scaling);
            if (la[1] && lb[1]) in_proj_z = lora_delta(in_proj_z, *la[1], *lb[1], lora_scaling);
            if (la[2] && lb[2]) out_proj = lora_delta(out_proj, *la[2], *lb[2], lora_scaling);
        }

        attn_output = linear_attention(
            attn_input, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
            a_log, dt_bias, conv1d_w, norm_w, out_proj,
            cfg->num_k_heads, cfg->key_dim, cfg->num_v_heads, cfg->val_dim,
            cfg->conv_kernel, cfg->rms_eps, kind
        );

        // MoE weights start at offset 11
        auto moe_output = moe_forward(
            rms_norm(hidden + attn_output, post_norm, cfg->rms_eps),
            *w[11], *w[12], *w[13], *w[14], *w[15],
            *w[16], *w[17],
            cfg->num_experts, cfg->top_k, cfg->moe_intermediate,
            cfg->norm_topk_prob, 0.001, cfg->expert_start, cfg->expert_count, kind
        );
        return (hidden + attn_output + moe_output);
    }
    return hidden;  // unreachable
}

// ──────────────────────────────────────────────────────────────────────
// Gradient Checkpointing via autograd::Function
// ──────────────────────────────────────────────────────────────────────

struct Qwen36CheckpointFunction : public torch::autograd::Function<Qwen36CheckpointFunction> {
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor input,
        int64_t num_layers,
        int64_t weight_ptrs_val,
        int64_t layer_configs_val,
        int64_t compute_type_val,
        double lora_scaling,
        int64_t lora_a_ptrs_val,
        int64_t lora_b_ptrs_val
    ) {
        ctx->saved_data["num_layers"] = num_layers;
        ctx->saved_data["weight_ptrs"] = weight_ptrs_val;
        ctx->saved_data["layer_configs"] = layer_configs_val;
        ctx->saved_data["compute_type"] = compute_type_val;
        ctx->saved_data["lora_scaling"] = lora_scaling;
        ctx->saved_data["lora_a_ptrs"] = lora_a_ptrs_val;
        ctx->saved_data["lora_b_ptrs"] = lora_b_ptrs_val;
        ctx->save_for_backward({input.detach()});

        at::AutoGradMode guard(false);
        auto* wptrs = reinterpret_cast<void**>(weight_ptrs_val);
        auto* lcfgs = reinterpret_cast<LayerConfig*>(layer_configs_val);
        auto ctype = static_cast<at::ScalarType>(compute_type_val);
        auto** la = lora_a_ptrs_val ? reinterpret_cast<void**>(lora_a_ptrs_val) : nullptr;
        auto** lb = lora_b_ptrs_val ? reinterpret_cast<void**>(lora_b_ptrs_val) : nullptr;

        at::Tensor h = input;
        // Calculate weight stride per layer (full=15, linear=18)
        for (int64_t i = 0; i < num_layers; i++) {
            int64_t w_offset = 0;
            for (int64_t j = 0; j < i; j++) {
                w_offset += (lcfgs[j].layer_type == 0) ? 15 : 18;
            }
            int64_t w_count = (lcfgs[i].layer_type == 0) ? 15 : 18;
            // Create a sub-array of weight pointers for this layer
            std::vector<void*> layer_w(wptrs + w_offset, wptrs + w_offset + w_count);
            // LoRA pointers: only for target layers, indexed per-layer
            void** la_layer = la ? la + i : nullptr;
            void** lb_layer = lb ? lb + i : nullptr;
            h = forward_single_layer_cpp(h, layer_w.data(), &lcfgs[i], ctype, lora_scaling, la_layer, lb_layer);
        }
        return h;
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output
    ) {
        auto saved = ctx->get_saved_variables();
        at::Tensor input = saved[0];
        auto num_layers = ctx->saved_data["num_layers"].toInt();
        auto weight_ptrs_val = ctx->saved_data["weight_ptrs"].toInt();
        auto layer_configs_val = ctx->saved_data["layer_configs"].toInt();
        auto compute_type_val = ctx->saved_data["compute_type"].toInt();
        double lora_scaling = ctx->saved_data["lora_scaling"].toDouble();
        auto lora_a_ptrs_val = ctx->saved_data["lora_a_ptrs"].toInt();
        auto lora_b_ptrs_val = ctx->saved_data["lora_b_ptrs"].toInt();

        at::AutoGradMode guard(true);
        input.set_requires_grad(true);

        auto* wptrs = reinterpret_cast<void**>(weight_ptrs_val);
        auto* lcfgs = reinterpret_cast<LayerConfig*>(layer_configs_val);
        auto ctype = static_cast<at::ScalarType>(compute_type_val);
        auto** la = lora_a_ptrs_val ? reinterpret_cast<void**>(lora_a_ptrs_val) : nullptr;
        auto** lb = lora_b_ptrs_val ? reinterpret_cast<void**>(lora_b_ptrs_val) : nullptr;

        // Set requires_grad on ALL LoRA A/B tensors so autograd builds a graph through them
        {
            int64_t off = 0;
            for (int64_t i = 0; i < num_layers; i++) {
                int64_t lc = (lcfgs[i].layer_type == 0) ? 4 : 3;
                if (la) {
                    auto** la_t = reinterpret_cast<at::Tensor**>(la + off);
                    auto** lb_t = reinterpret_cast<at::Tensor**>(lb + off);
                    for (int64_t k = 0; k < lc; k++) {
                        if (la_t[k]) la_t[k]->set_requires_grad(true);
                        if (lb_t[k]) lb_t[k]->set_requires_grad(true);
                    }
                }
                off += lc;
            }
        }

        // Recompute forward with grad enabled — builds a fresh graph
        at::Tensor h = input;
        {
            int64_t off = 0;
            for (int64_t i = 0; i < num_layers; i++) {
                int64_t w_offset = 0;
                for (int64_t j = 0; j < i; j++)
                    w_offset += (lcfgs[j].layer_type == 0) ? 15 : 18;
                int64_t w_count = (lcfgs[i].layer_type == 0) ? 15 : 18;
                std::vector<void*> layer_w(wptrs + w_offset, wptrs + w_offset + w_count);
                int64_t lc = (lcfgs[i].layer_type == 0) ? 4 : 3;
                void** la_layer = la ? la + off : nullptr;
                void** lb_layer = lb ? lb + off : nullptr;
                h = forward_single_layer_cpp(h, layer_w.data(), &lcfgs[i], ctype, lora_scaling, la_layer, lb_layer);
                off += lc;
            }
        }

        // Backward through the recomputed graph.
        // Gradients accumulate into ALL leaf tensors (input + LoRA A/B).
        // input.detach() in forward ensures no "second time backward" error.
        h.backward(grad_output[0]);

        return {input.grad(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

// ──────────────────────────────────────────────────────────────────────
// C FFI
// ──────────────────────────────────────────────────────────────────────

extern "C" {

void* qwen36_checkpoint_forward(
    void* input_ptr,
    int64_t num_layers,
    void** weight_ptrs,
    void* layer_configs,
    int32_t compute_type,
    double lora_scaling,
    void** lora_a_ptrs,
    void** lora_b_ptrs
) {
    try {
        auto& input = *reinterpret_cast<at::Tensor*>(input_ptr);
        auto result = Qwen36CheckpointFunction::apply(
            input,
            num_layers,
            (int64_t)(uintptr_t)weight_ptrs,
            (int64_t)(uintptr_t)layer_configs,
            (int64_t)compute_type,
            lora_scaling,
            (int64_t)(uintptr_t)lora_a_ptrs,
            (int64_t)(uintptr_t)lora_b_ptrs
        );
        return new at::Tensor(std::move(result));
    } catch (const std::exception& e) {
        fprintf(stderr, "[qwen36_checkpoint_forward] FAILED: %s\n", e.what());
        return nullptr;
    }
}

void* qwen36_gemm(void* a_ptr, void* b_ptr, int transpose_b) {
    try {
        auto& a = *reinterpret_cast<at::Tensor*>(a_ptr);
        auto& b = *reinterpret_cast<at::Tensor*>(b_ptr);
        if (transpose_b) return new at::Tensor(at::matmul(a, b.t()));
        return new at::Tensor(at::matmul(a, b));
    } catch (const std::exception& e) {
        fprintf(stderr, "[qwen36_gemm] FAILED: %s\n", e.what());
        return nullptr;
    }
}

void* qwen36_swiglu_gemm(void* a_ptr, void* gate_up_ptr, void* down_ptr) {
    try {
        auto& a = *reinterpret_cast<at::Tensor*>(a_ptr);
        auto& gate_up = *reinterpret_cast<at::Tensor*>(gate_up_ptr);
        auto& down = *reinterpret_cast<at::Tensor*>(down_ptr);
        int64_t inter = gate_up.size(0) / 2;
        auto gate = gate_up.narrow(0, 0, inter);
        auto up = gate_up.narrow(0, inter, inter);
        auto activated = at::silu(at::matmul(a, gate.t())) * at::matmul(a, up.t());
        return new at::Tensor(at::matmul(activated, down.t()));
    } catch (const std::exception& e) {
        fprintf(stderr, "[qwen36_swiglu_gemm] FAILED: %s\n", e.what());
        return nullptr;
    }
}

void* qwen36_chunked_delta_rule(
    void* q_ptr, void* k_ptr, void* v_ptr,
    void* g_ptr, void* beta_ptr, int64_t chunk_size
) {
    // Delegated to matrix formulation inside linear_attention
    return nullptr;
}

void* qwen36_sdpa(void* q_ptr, void* k_ptr, void* v_ptr, int is_causal, double scale) {
    try {
        auto& q = *reinterpret_cast<at::Tensor*>(q_ptr);
        auto& k = *reinterpret_cast<at::Tensor*>(k_ptr);
        auto& v = *reinterpret_cast<at::Tensor*>(v_ptr);
        return new at::Tensor(at::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, scale, is_causal != 0));
    } catch (const std::exception& e) {
        fprintf(stderr, "[qwen36_sdpa] FAILED: %s\n", e.what());
        return nullptr;
    }
}

void qwen36_set_grad_enabled(int enabled) {
    at::GradMode::set_enabled(enabled != 0);
}

int qwen36_get_grad_enabled() {
    return at::GradMode::is_enabled() ? 1 : 0;
}

void qwen36_free_tensor(void* tensor_ptr) {
    if (tensor_ptr) delete reinterpret_cast<at::Tensor*>(tensor_ptr);
}

}  // extern "C"
