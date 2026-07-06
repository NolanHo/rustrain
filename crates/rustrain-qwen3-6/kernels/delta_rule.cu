// delta_rule.cu — CUDA kernel implementation for gated delta rule
// Compiled with nvcc, linked into libqwen36_kernels.so

#include "delta_rule.cuh"

// C-linkage wrapper for the forward host launcher
extern "C" void cuda_gated_delta_rule(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    float* state, float* out, float* delta_buf,
    int BH, int seq_len, int key_dim, int val_dim
) {
    launch_gated_delta_rule(q, k, v, g_exp, beta, state, out, delta_buf,
                            BH, seq_len, key_dim, val_dim);
}

// C-linkage wrapper for the backward host launcher
extern "C" void cuda_gated_delta_rule_backward(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    const float* final_state,
    const float* delta_buf,
    const float* grad_out,
    float* grad_q, float* grad_k, float* grad_v,
    float* grad_g, float* grad_beta,
    int BH, int seq_len, int key_dim, int val_dim
) {
    launch_gated_delta_rule_backward(q, k, v, g_exp, beta,
        final_state, delta_buf, grad_out,
        grad_q, grad_k, grad_v, grad_g, grad_beta,
        BH, seq_len, key_dim, val_dim);
}
