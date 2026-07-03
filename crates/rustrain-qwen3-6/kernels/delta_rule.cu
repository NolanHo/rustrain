// delta_rule.cu — CUDA kernel implementation for gated delta rule
// Compiled with nvcc, linked into libqwen36_kernels.so

#include "delta_rule.cuh"

// C-linkage wrapper for the host launcher
extern "C" void cuda_gated_delta_rule(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    float* state, float* out,
    int BH, int seq_len, int key_dim, int val_dim
) {
    launch_gated_delta_rule(q, k, v, g_exp, beta, state, out,
                            BH, seq_len, key_dim, val_dim);
}
