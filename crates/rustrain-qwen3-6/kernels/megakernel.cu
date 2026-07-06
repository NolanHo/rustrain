// megakernel.cu — Fused forward+backward for Qwen3.6 layers.
//
// Each layer is a single autograd::Function that:
// - Forward: computes output, saves MINIMAL intermediates
// - Backward: hand-written gradient computation (no PyTorch graph traversal)
//
// This eliminates checkpoint recompute (the main bottleneck at 1M+).
// Forward runs once. Backward uses saved intermediates directly.
//
// Saved intermediates per linear attention layer (~20GB for 1M seq):
//   attn_input [B, S, H]     — for rms_norm backward
//   qkv_conv   [B, S, qkv_dim] — for conv1d/q/k/v backward
//   a, b, z    [B, S, ...]  — for g/beta/z backward
//   core_out   [B, S, V*D]   — for gated_norm backward
//   q_normed, k_normed — for L2 norm backward
//
// Saved intermediates per full attention layer (~30GB for 1M seq):
//   attn_input, q/k/v/gate, sdpa_output, cos/sin

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/variable.h>
#include "backward.h"

// Forward declaration
extern "C" void cuda_gated_delta_rule(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    float* state, float* out,
    int BH, int seq_len, int key_dim, int val_dim
);

// ──────────────────────────────────────────────────────────────────────
// Linear Attention Layer: Fused Forward + Backward
// ──────────────────────────────────────────────────────────────────────

struct LinearAttnLayer : public torch::autograd::Function<LinearAttnLayer> {
    // Saved data keys:
    // "attn_input" — rms_norm output [B, S, H]
    // "qkv_conv"   — conv1d+silu output [B, S, qkv_dim]
    // "a_proj"     — projection a [B, S, num_v_heads]
    // "b_proj"     — projection b [B, S, num_v_heads]
    // "z_proj"     — projection z [B, S, v_size]
    // "q_normed"   — L2-normalized q [BH, S, D_k]
    // "k_normed"   — L2-normalized k [BH, S, D_k]
    // "g"           — decay factor [B, H, S]
    // "beta"        — sigmoid(b) [B, H, S]
    // "core_out"    — delta rule output [BH, S, D_v]
    // "gated"       — gated norm output [B, S, V*D]

    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor hidden,           // [B, S, H]
        at::Tensor input_norm,       // [H]
        at::Tensor in_proj_qkv,      // [qkv_dim, H]
        at::Tensor in_proj_z,        // [v_size, H]
        at::Tensor in_proj_a,        // [num_v_heads, H]
        at::Tensor in_proj_b,        // [num_v_heads, H]
        at::Tensor a_log,            // [num_v_heads]
        at::Tensor dt_bias,          // [num_v_heads]
        at::Tensor conv1d_w,         // [qkv_dim, 1, conv_k]
        at::Tensor norm_w,           // [val_dim]
        at::Tensor out_proj,         // [H, v_size]
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

        // --- RMSNorm (BF16, no FP32 conversion) ---
        auto attn_input = rms_norm(hidden, input_norm, rms_eps);

        // --- QKV projection ---
        auto qkv = at::matmul(attn_input, in_proj_qkv.t());  // [B, S, qkv_dim]
        auto qkv_t = qkv.transpose(1, 2);                     // [B, qkv_dim, S]

        // --- Conv1d (depthwise, causal, + SiLU) ---
        int64_t pad = conv_kernel - 1;
        auto padding = at::zeros({batch, qkv_dim, pad}, qkv.options());
        auto padded = at::cat({padding, qkv_t}, 2);
        auto conv_out = at::conv1d(padded, conv1d_w, {},
            at::IntArrayRef({1}), at::IntArrayRef({0}), at::IntArrayRef({1}), qkv_dim);
        conv_out = at::silu(conv_out.narrow(2, 0, seq));
        auto qkv_conv = conv_out.transpose(1, 2);  // [B, S, qkv_dim]

        // Split Q, K, V
        auto q = qkv_conv.narrow(-1, 0, q_size).view({batch, seq, num_k_heads, key_dim});
        auto k = qkv_conv.narrow(-1, q_size, q_size).view({batch, seq, num_k_heads, key_dim});
        auto v = qkv_conv.narrow(-1, q_size * 2, v_size).view({batch, seq, num_v_heads, val_dim});

        // --- A, B, Z projections ---
        auto a = at::matmul(attn_input, in_proj_a.t());  // [B, S, num_v_heads]
        auto b = at::matmul(attn_input, in_proj_b.t());
        auto z = at::matmul(attn_input, in_proj_z.t()).view({batch, seq, num_v_heads, val_dim});

        // --- g = -exp(A_log) * softplus(a + dt_bias) ---
        auto a_log_f = a_log.to(at::kFloat);
        auto dt_bias_f = dt_bias.to(at::kFloat);
        auto a_f = a.to(at::kFloat);
        auto g = a_log_f.unsqueeze(0).unsqueeze(0).exp().neg() *
                 at::softplus(a_f + dt_bias_f.unsqueeze(0).unsqueeze(0));
        auto beta = at::sigmoid(b);

        // --- Repeat K heads to V heads ---
        int64_t n_rep = num_v_heads / num_k_heads;
        q = q.repeat_interleave(n_rep, 2);
        k = k.repeat_interleave(n_rep, 2);

        // --- L2 normalize Q, K ---
        auto q_normed = (q.to(at::kFloat) / q.to(at::kFloat).norm(2, -1, true).clamp_min(1e-6));
        auto k_normed = (k.to(at::kFloat) / k.to(at::kFloat).norm(2, -1, true).clamp_min(1e-6));

        // --- Scale Q ---
        double scale = 1.0 / std::sqrt((double)key_dim);
        q_normed = q_normed * scale;

        // --- Transpose for delta rule ---
        auto q_t = q_normed.transpose(1, 2).contiguous();  // [B, H, S, D_k]
        auto k_t = k_normed.transpose(1, 2).contiguous();
        auto v_t = v.to(at::kFloat).transpose(1, 2).contiguous();
        auto g_t = g.transpose(1, 2).contiguous();
        auto beta_t = beta.to(at::kFloat).transpose(1, 2).contiguous();

        auto g_exp = g_t.exp();
        int64_t BH = batch * num_v_heads;
        auto state = at::zeros({BH, key_dim, val_dim},
            at::TensorOptions().dtype(at::kFloat).device(device));

        auto q_contig = q_t.reshape({BH, seq, key_dim}).contiguous().to(at::kFloat);
        auto k_contig = k_t.reshape({BH, seq, key_dim}).contiguous().to(at::kFloat);
        auto v_contig = v_t.reshape({BH, seq, val_dim}).contiguous().to(at::kFloat);
        auto g_contig = g_exp.reshape({BH, seq}).contiguous().to(at::kFloat);
        auto beta_contig = beta_t.reshape({BH, seq}).contiguous().to(at::kFloat);
        auto state_contig = state.contiguous();
        auto outs = at::empty({BH, seq, val_dim}, q_t.options());

        cuda_gated_delta_rule(
            q_contig.data_ptr<float>(),
            k_contig.data_ptr<float>(),
            v_contig.data_ptr<float>(),
            g_contig.data_ptr<float>(),
            beta_contig.data_ptr<float>(),
            state_contig.data_ptr<float>(),
            outs.data_ptr<float>(),
            (int)BH, (int)seq, (int)key_dim, (int)val_dim
        );

        auto core_out = outs.reshape({batch, num_v_heads, seq, val_dim})
                             .transpose(1, 2).to(compute_type);  // [B, S, V, D_v]

        // --- Gated RMSNorm (raw weight, not 1+weight) ---
        auto core_flat = core_out.reshape({-1, val_dim});
        auto z_flat = z.reshape({-1, val_dim});
        auto variance = core_flat.to(at::kFloat).pow(2).mean(-1, true);
        auto normed = (core_flat.to(at::kFloat) * (variance + rms_eps).rsqrt() *
                       norm_w.to(at::kFloat)).to(core_flat.scalar_type());
        auto gated = (normed * at::silu(z_flat.to(at::kFloat)).to(normed.scalar_type()))
                     .view({batch, seq, num_v_heads * val_dim});

        // --- Output projection ---
        auto result = at::matmul(gated, out_proj.t());  // [B, S, H]

        // --- Save intermediates for backward ---
        // Only save what's needed for backward — skip large temporaries
        ctx->save_for_backward({
            hidden,          // for rms_norm backward
            attn_input,      // for matmul backward (qkv, a, b, z projections)
            qkv_conv,        // for conv1d backward + q/k/v split
            a,               // for g backward
            b,               // for beta backward
            z_flat.reshape({batch, seq, num_v_heads, val_dim}),  // for gated backward
            q_normed.reshape({batch, seq, num_v_heads, key_dim}),  // for L2 backward
            k_normed.reshape({batch, seq, num_k_heads * n_rep, key_dim}),
            core_out,        // for gated norm backward
            gated,           // for out_proj backward
            result           // for residual
        });
        ctx->saved_data["rms_eps"] = rms_eps;
        ctx->saved_data["num_k_heads"] = num_k_heads;
        ctx->saved_data["key_dim"] = key_dim;
        ctx->saved_data["num_v_heads"] = num_v_heads;
        ctx->saved_data["val_dim"] = val_dim;
        ctx->saved_data["conv_kernel"] = conv_kernel;
        ctx->saved_data["scale"] = scale;
        ctx->saved_data["n_rep"] = n_rep;
        ctx->saved_data["a_log"] = a_log;
        ctx->saved_data["dt_bias"] = dt_bias;
        ctx->saved_data["conv1d_w"] = conv1d_w;
        ctx->saved_data["norm_w"] = norm_w;
        ctx->saved_data["out_proj"] = out_proj;
        ctx->saved_data["in_proj_qkv"] = in_proj_qkv;
        ctx->saved_data["in_proj_z"] = in_proj_z;
        ctx->saved_data["in_proj_a"] = in_proj_a;
        ctx->saved_data["in_proj_b"] = in_proj_b;
        ctx->saved_data["input_norm"] = input_norm;

        return hidden + result;  // residual connection
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<at::Tensor> grad_output
    ) {
        auto saved = ctx->get_saved_variables();
        auto hidden = saved[0];
        auto attn_input = saved[1];
        auto qkv_conv = saved[2];
        auto a = saved[3];
        auto b = saved[4];
        auto z = saved[5];
        auto q_normed_saved = saved[6];
        auto k_normed_saved = saved[7];
        auto core_out = saved[8];
        auto gated = saved[9];
        // saved[10] = result (not needed for backward)

        double rms_eps = ctx->saved_data["rms_eps"].toDouble();
        int64_t num_k_heads = ctx->saved_data["num_k_heads"].toInt();
        int64_t key_dim = ctx->saved_data["key_dim"].toInt();
        int64_t num_v_heads = ctx->saved_data["num_v_heads"].toInt();
        int64_t val_dim = ctx->saved_data["val_dim"].toInt();
        int64_t conv_kernel = ctx->saved_data["conv_kernel"].toInt();
        double scale = ctx->saved_data["scale"].toDouble();
        int64_t n_rep = ctx->saved_data["n_rep"].toInt();

        auto a_log = ctx->saved_data["a_log"].toTensor();
        auto dt_bias = ctx->saved_data["dt_bias"].toTensor();
        auto conv1d_w = ctx->saved_data["conv1d_w"].toTensor();
        auto norm_w = ctx->saved_data["norm_w"].toTensor();
        auto out_proj = ctx->saved_data["out_proj"].toTensor();
        auto in_proj_qkv = ctx->saved_data["in_proj_qkv"].toTensor();
        auto in_proj_z = ctx->saved_data["in_proj_z"].toTensor();
        auto in_proj_a = ctx->saved_data["in_proj_a"].toTensor();
        auto in_proj_b = ctx->saved_data["in_proj_b"].toTensor();
        auto input_norm = ctx->saved_data["input_norm"].toTensor();

        int64_t batch = hidden.size(0), seq = hidden.size(1);
        int64_t q_size = num_k_heads * key_dim;
        int64_t v_size = num_v_heads * val_dim;
        int64_t qkv_dim = q_size * 2 + v_size;
        auto device = hidden.device();

        auto grad_hidden = grad_output[0];  // grad w.r.t. (hidden + result) = grad_output
        auto grad_result = grad_hidden;     // d(hidden+result)/d(result) = grad_hidden

        // === Backward: output projection ===
        // forward: result = matmul(gated, out_proj.t())
        // grad_gated = grad_result @ out_proj  [B, S, v_size]
        // grad_out_proj = grad_result^T @ gated  [H, v_size]
        auto grad_result_flat = grad_result.reshape({-1, grad_result.size(-1)});
        auto gated_flat = gated.reshape({-1, gated.size(-1)});
        auto grad_gated_flat = at::matmul(grad_result_flat, out_proj);  // [B*S, v_size]
        // grad_out_proj computed separately (weight grad, handled by caller for LoRA)

        // === Backward: gated RMSNorm ===
        // forward: normed = rms_norm(core, norm_w) (raw weight)
        //          gated = normed * silu(z)
        // grad_normed = grad_gated * silu(z)
        // grad_z = grad_gated * normed * silu'(z)
        auto core_flat = core_out.reshape({-1, val_dim}).to(at::kFloat);
        auto z_flat = z.reshape({-1, val_dim}).to(at::kFloat);
        auto grad_gated_f = grad_gated_flat.to(at::kFloat);

        auto var = core_flat.pow(2).mean(-1, true);
        auto inv_rms = (var + rms_eps).rsqrt();
        auto normed_raw = core_flat * inv_rms;  // before weight
        auto normed = normed_raw * norm_w.to(at::kFloat);
        auto silu_z = at::silu(z_flat);

        auto grad_normed = grad_gated_f * silu_z;       // [B*S, V*D]
        auto grad_z = grad_gated_f * normed * (silu_z * (1.0 - silu_z));  // silu'(z) = sig*(1+x*(1-sig))

        // === Backward: rms_norm (gated norm, raw weight) ===
        // grad_normed_scaled = grad_normed * norm_w
        // grad_core = inv_rms * (grad_normed_scaled - (core*inv_rms) * mean(grad_normed_scaled * core*inv_rms))
        auto grad_normed_scaled = grad_normed * norm_w.to(at::kFloat);
        auto core_normed = core_flat * inv_rms;
        auto dot = (grad_normed_scaled * core_normed).mean(-1, true);
        auto grad_core = inv_rms * (grad_normed_scaled - core_normed * dot);  // [B*S, V*D]

        // === Backward: delta rule (CUDA kernel — TODO: write backward kernel) ===
        // For now, use PyTorch autograd for delta rule backward.
        // This is the one part that needs a custom CUDA kernel.
        // TODO: implement cuda_gated_delta_rule_backward
        //
        // For now, approximate: grad_q = grad_core @ S^T, grad_k, grad_v, grad_g, grad_beta
        // This is an approximation — the exact backward requires the saved state.

        // === Backward: L2 normalize ===
        // q_normed = q / ||q||, grad_q = (grad_q_normed - q_normed * dot(grad_q_normed, q_normed)) / ||q||
        // (Using saved q_normed and k_normed)

        // === Backward: scale ===
        // grad_q_normed *= scale

        // === Backward: conv1d + SiLU ===
        // grad_qkv_conv = silu_backward(conv_out_pre_silu, grad_qkv_conv)
        // grad_qkv = conv1d_backward(grad_qkv_conv, conv1d_w)

        // === Backward: QKV projection ===
        // grad_attn_input = grad_qkv @ in_proj_qkv
        // (plus grad from a, b, z projections)

        // === Backward: rms_norm (input) ===
        // grad_hidden_from_attn = rms_norm_backward(hidden, input_norm, grad_attn_input)

        // Final grad_hidden = grad_output (residual) + grad_hidden_from_attn

        // For now, return grad_output as grad_input (placeholder — will be filled in)
        // This is NOT correct — it's a placeholder for the full backward.
        auto grad_input = grad_output[0];  // residual: d(hidden+result)/d(hidden) = grad_output

        // Return gradients for all inputs (in order)
        int64_t num_inputs = 11 + 6 + 4;  // tensors + config ints
        std::vector<at::Tensor> grads;
        grads.push_back(grad_input);  // hidden
        // Fill remaining with empty tensors (weight grads handled by LoRA)
        for (int i = 1; i < 11 + 6 + 4; i++) {
            grads.push_back(at::Tensor());
        }
        return grads;
    }
};

// ──────────────────────────────────────────────────────────────────────
// Host launcher for megakernel forward
// ──────────────────────────────────────────────────────────────────────

extern "C" void* megakernel_linear_attn_create() {
    return nullptr;  // No persistent state needed
}

extern "C" at::Tensor megakernel_linear_attn_forward_backward(
    void* handle,
    const at::Tensor& hidden,
    const at::Tensor& grad_output,
    void** weights,
    void* layer_config,
    double lora_scaling,
    void** lora_a, void** lora_b,
    bool is_backward
) {
    // Placeholder — full implementation uses LinearAttnLayer::apply
    return at::Tensor();
}
