// delta_rule.cu — CUDA kernel implementation for gated delta rule
// Compiled with nvcc, linked into libqwen36_kernels.so

#include "delta_rule.cuh"

// C-linkage wrapper for the forward host launcher
extern "C" int cuda_gated_delta_rule(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    float* state, float* out, float* delta_buf,
    const int32_t* lengths, int heads_per_batch,
    int BH, int seq_len, int key_dim, int val_dim, void* stream_ptr
) {
    if (key_dim != DR_D_K || val_dim != DR_D_V || heads_per_batch <= 0) {
        return -1;
    }
    launch_gated_delta_rule(q, k, v, g_exp, beta, state, out, delta_buf,
                            lengths, heads_per_batch,
                            BH, seq_len, key_dim, val_dim,
                            reinterpret_cast<cudaStream_t>(stream_ptr));
    return static_cast<int>(cudaGetLastError());
}

// C-linkage wrapper for the backward host launcher
extern "C" int cuda_gated_delta_rule_backward(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    const float* final_state,
    const float* delta_buf,
    const float* grad_out,
    float* grad_q, float* grad_k, float* grad_v,
    float* grad_g, float* grad_beta,
    const int32_t* lengths, int heads_per_batch,
    int BH, int seq_len, int key_dim, int val_dim, void* stream_ptr
) {
    if (key_dim != DR_D_K || val_dim != DR_D_V || heads_per_batch <= 0) return -1;
    int launch_status = launch_gated_delta_rule_backward(q, k, v, g_exp, beta,
        final_state, delta_buf, grad_out,
        grad_q, grad_k, grad_v, grad_g, grad_beta,
        lengths, heads_per_batch,
        BH, seq_len, key_dim, val_dim,
        reinterpret_cast<cudaStream_t>(stream_ptr));
    if (launch_status != 0) return launch_status;
    return static_cast<int>(cudaGetLastError());
}
