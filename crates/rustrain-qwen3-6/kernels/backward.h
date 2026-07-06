// backward.h — Hand-written backward implementations for megakernel.
// Each function computes gradients without PyTorch autograd graph.
// This eliminates checkpoint recompute — forward runs once, backward uses
// saved intermediates directly.

#pragma once

#include <ATen/ATen.h>
#include <cmath>

// ──────────────────────────────────────────────────────────────────────
// RMSNorm backward
// Forward: out = input * rsqrt(var(input) + eps) * (1 + weight)
//   where var = mean(input^2, dim=-1)
// Backward: given grad_out, compute grad_input, grad_weight
//   Let inv_rms = rsqrt(var + eps)
//   normed = input * inv_rms
//   out = normed * (1 + weight)
//   grad_normed = grad_out * (1 + weight)
//   grad_input = inv_rms * (grad_normed - normed * mean(grad_normed * normed, dim=-1))
//   grad_weight = sum(grad_out * normed, dim=0..S-1)
// ──────────────────────────────────────────────────────────────────────

struct RMSNormBackward {
    at::Tensor grad_input;
    at::Tensor grad_weight;
};

inline RMSNormBackward rms_norm_backward(
    const at::Tensor& input,       // [B, S, H] or [..., H]
    const at::Tensor& weight,     // [H]
    const at::Tensor& grad_out,   // same shape as input
    double eps
) {
    // Recompute forward values (no need to save — just recompute, it's cheap)
    auto kind = input.scalar_type();
    auto input_sq = input.pow(2);
    auto var = input_sq.mean(-1, true);
    auto inv_rms = (var + eps).rsqrt();
    auto normed = input * inv_rms;
    auto one_plus_w = (1.0 + weight.to(kind));

    // grad_normed = grad_out * (1 + weight)
    auto grad_normed = grad_out * one_plus_w;

    // grad_input = inv_rms * (grad_normed - normed * mean(grad_normed * normed))
    auto dot = (grad_normed * normed).mean(-1, true);
    auto grad_input = inv_rms * (grad_normed - normed * dot);

    // grad_weight = sum(grad_out * normed, dim=0..S-1) → [H]
    auto grad_weight = (grad_out * normed).sum(at::IntArrayRef({0, 1}));

    return RMSNormBackward{grad_input, grad_weight};
}

// ──────────────────────────────────────────────────────────────────────
// SiLU backward
// Forward: out = x * sigmoid(x)
// Backward: grad_x = grad_out * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
//          = grad_out * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
// ──────────────────────────────────────────────────────────────────────

inline at::Tensor mk_silu_backward(const at::Tensor& x, const at::Tensor& grad_out) {
    auto sig = at::sigmoid(x);
    return grad_out * sig * (1.0 + x * (1.0 - sig));
}

// ──────────────────────────────────────────────────────────────────────
// Sigmoid backward
// Forward: out = sigmoid(x)
// Backward: grad_x = grad_out * out * (1 - out)
// ──────────────────────────────────────────────────────────────────────

inline at::Tensor sigmoid_backward(const at::Tensor& out, const at::Tensor& grad_out) {
    return grad_out * out * (1.0 - out);
}

// ──────────────────────────────────────────────────────────────────────
// Softplus backward
// Forward: out = softplus(x) = log(1 + exp(x))
// Backward: grad_x = grad_out * sigmoid(x)
// ──────────────────────────────────────────────────────────────────────

inline at::Tensor softplus_backward(const at::Tensor& x, const at::Tensor& grad_out) {
    return grad_out * at::sigmoid(x);
}

// ──────────────────────────────────────────────────────────────────────
// Matmul backward: y = x @ W^T  (x: [B*S, H_in], W: [H_out, H_in], y: [B*S, H_out])
// grad_x = grad_y @ W         [B*S, H_in]
// grad_W = grad_y^T @ x       [H_out, H_in]
// ──────────────────────────────────────────────────────────────────────

struct MatmulBackward {
    at::Tensor grad_input;
    at::Tensor grad_weight;
};

inline MatmulBackward mk_matmul_backward(
    const at::Tensor& x,          // [..., H_in]
    const at::Tensor& weight,     // [H_out, H_in]
    const at::Tensor& grad_y      // [..., H_out]
) {
    auto x_2d = x.reshape({-1, x.size(-1)});
    auto grad_y_2d = grad_y.reshape({-1, grad_y.size(-1)});

    auto grad_x = at::matmul(grad_y_2d, weight).reshape(x.sizes());
    auto grad_w = at::matmul(grad_y_2d.t(), x_2d);

    return MatmulBackward{grad_x, grad_w};
}

// ──────────────────────────────────────────────────────────────────────
// Rotate_half backward (identity — rotate_half is its own inverse)
// rotate_half([a, b]) = [-b, a]
// rotate_half(rotate_half(x)) = x
// So grad_x = rotate_half(grad_out)
// ──────────────────────────────────────────────────────────────────────

inline at::Tensor rotate_half_backward(const at::Tensor& grad_out) {
    auto last_dim = grad_out.size(-1);
    auto half = last_dim / 2;
    auto x1 = grad_out.narrow(-1, 0, half);
    auto x2 = grad_out.narrow(-1, half, half);
    return at::cat({x2, x1.neg()}, -1);
}

// ──────────────────────────────────────────────────────────────────────
// RoPE backward
// Forward: q_rotated = q * cos + rotate_half(q) * sin
// Backward: grad_q = grad_out * cos + rotate_half(grad_out) * sin
// (cos and sin are constants — no gradient needed)
// ──────────────────────────────────────────────────────────────────────

inline at::Tensor rope_backward(
    const at::Tensor& grad_out,
    const at::Tensor& cos,
    const at::Tensor& sin
) {
    return grad_out * cos + rotate_half_backward(grad_out) * sin;
}

// ──────────────────────────────────────────────────────────────────────
// L2 normalize backward
// Forward: out = x / ||x||  (where ||x|| = norm(x, dim=-1))
// Backward: grad_x = (grad_out - out * dot(grad_out, out)) / ||x||
// ──────────────────────────────────────────────────────────────────────

inline at::Tensor l2norm_backward(
    const at::Tensor& x,          // [..., D]
    const at::Tensor& out,        // [..., D]  (normalized)
    const at::Tensor& grad_out   // [..., D]
) {
    auto norm = x.norm(2, -1, true).clamp_min(1e-6);
    auto dot = (grad_out * out).sum(-1, true);
    return (grad_out - out * dot) / norm;
}

// ──────────────────────────────────────────────────────────────────────
// Gated RMSNorm backward
// Forward: normed = rms_norm(core) (raw weight, not 1+weight)
//          gated = normed * silu(z)
//          out = matmul(gated, out_proj.t())
// Backward: given grad_out (from matmul backward)
//   grad_gated = grad_out @ out_proj          [B*S, V*D]
//   grad_normed = grad_gated * silu(z)        [B*S, V*D]
//   grad_z = grad_gated * normed * silu'(z)   [B*S, V*D]
//   grad_core = rms_norm_backward(core, norm_w, grad_normed)
//   grad_out_proj = grad_out.t() @ gated
// ──────────────────────────────────────────────────────────────────────

struct GatedNormBackward {
    at::Tensor grad_core;      // [B*S, V*D]
    at::Tensor grad_z;         // [B*S, V*D]
    at::Tensor grad_out_proj;  // [H_out, H_in]
    at::Tensor grad_norm_w;    // [V*D]
};

inline GatedNormBackward gated_norm_backward(
    const at::Tensor& core_flat,   // [B*S, V*D]
    const at::Tensor& z_flat,      // [B*S, V*D]
    const at::Tensor& norm_w,      // [V*D]
    const at::Tensor& out_proj,    // [H_out, V*D]
    const at::Tensor& grad_gated, // [B*S, V*D] — gradient after out_proj matmul
    double eps
) {
    // Recompute normed (cheap)
    auto var = core_flat.pow(2).mean(-1, true);
    auto inv_rms = (var + eps).rsqrt();
    auto normed = core_flat * inv_rms * norm_w;  // raw weight (not 1+weight)

    // grad_normed = grad_gated * silu(z)
    auto silu_z = at::silu(z_flat);
    auto grad_normed_raw = grad_gated * silu_z;

    // grad_z = grad_gated * normed * silu'(z)
    auto grad_z = grad_gated * normed * mk_silu_backward(z_flat, at::ones_like(z_flat));

    // grad_core = rms_norm_backward
    auto grad_normed_scaled = grad_normed_raw * norm_w;  // undo weight scaling
    auto dot = (grad_normed_scaled * (core_flat * inv_rms)).mean(-1, true);
    auto grad_core = inv_rms * (grad_normed_scaled - (core_flat * inv_rms) * dot);

    // grad_norm_w = sum(grad_normed_raw * (core_flat * inv_rms))
    auto grad_norm_w = (grad_normed_raw * (core_flat * inv_rms)).sum(at::IntArrayRef({0}));

    // grad_out_proj computed by caller (mk_matmul_backward)
    return GatedNormBackward{grad_core, grad_z, at::Tensor(), grad_norm_w};
}
