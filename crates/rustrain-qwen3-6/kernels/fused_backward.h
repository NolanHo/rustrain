// fused_backward.h — C++ wrappers for fused CUDA backward kernels
#pragma once
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

// Forward declarations — implemented in fused_backward.cu

// Fused SiLU backward: grad_x = grad_out * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
at::Tensor fused_silu_backward(const at::Tensor& x, const at::Tensor& grad_out);

// Fused gate sigmoid backward:
//   grad_sdpa = grad_attn_out * sigmoid(gate)
//   grad_gate = grad_attn_out * sigmoid(gate) * (1 - sigmoid(gate))
// Returns {grad_sdpa, grad_gate}
std::tuple<at::Tensor, at::Tensor> fused_gate_backward(
    const at::Tensor& gate, const at::Tensor& grad_attn_out);

// Fused RMSNorm backward (2-pass kernel):
//   grad_input = inv_rms * (grad_normed - normed * mean(grad_normed * normed))
//   grad_weight = sum(grad_out * normed)
// Returns {grad_input, grad_weight}
std::tuple<at::Tensor, at::Tensor> fused_rms_norm_backward(
    const at::Tensor& input, const at::Tensor& weight,
    const at::Tensor& grad_out, double eps);

// Fused gated norm backward (2-pass kernel):
//   grad_core, grad_z, grad_norm_w
// Returns {grad_core, grad_z, grad_norm_w}
std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_gated_norm_backward(
    const at::Tensor& core_flat, const at::Tensor& z_flat,
    const at::Tensor& norm_w, const at::Tensor& grad_gated,
    double eps);

// Fused L2 norm backward (single pass with reduction):
//   grad_x = (grad_out - out * dot(grad_out, out)) / ||x|| * scale
at::Tensor fused_l2norm_backward(
    const at::Tensor& x, const at::Tensor& out,
    const at::Tensor& grad_out, float scale);

// Fused RoPE backward:
//   grad_q = grad_out * cos + rotate_half(grad_out) * sin
at::Tensor fused_rope_backward(
    const at::Tensor& grad_out, const at::Tensor& cos,
    const at::Tensor& sin, int64_t rotary_dim);

// Fused g backward (linear attention):
//   grad_a = grad_g * (-exp(A_log)) * sigmoid(a + dt_bias)
at::Tensor fused_g_backward(
    const at::Tensor& a, const at::Tensor& a_log,
    const at::Tensor& dt_bias, const at::Tensor& grad_g);

// Fused beta sigmoid backward:
//   grad_b = grad_beta * sigmoid(b) * (1 - sigmoid(b))
at::Tensor fused_beta_backward(
    const at::Tensor& b, const at::Tensor& grad_beta);
