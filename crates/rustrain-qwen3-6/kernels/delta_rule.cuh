// delta_rule.cuh — CUDA kernel for chunk-wise parallel gated delta rule
//
// Eliminates per-token kernel launch overhead by running the entire
// recurrent loop inside a single CUDA kernel using shared memory.
//
// Grid: (1, BH) — one block per (batch * head), processes entire sequence
// Block: THREADS threads cooperate on D_K × D_V state matrix
//
// All data (q, k, v, g, beta, state) is in FP32.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

// Single-block-per-head kernel: processes entire sequence sequentially
// but vectorizes over D_V dimension using thread parallelism.
//
// q:      [BH, S, D_K]
// k:      [BH, S, D_K]
// v:      [BH, S, D_V]
// g:      [BH, S]       (already exp'd)
// beta:   [BH, S]
// state:  [BH, D_K, D_V] (persistent, modified in-place)
// out:    [BH, S, D_V]

template<int D_K, int D_V, int THREADS>
__global__ void gated_delta_rule_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    int S
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;

    // Dynamic shared memory for state: [D_K * D_V]
    extern __shared__ float state_s[];

    // Load state into shared memory
    const float* state_ro = state + bh * D_K * D_V;
    #pragma unroll
    for (int i = tid; i < D_K * D_V; i += THREADS) {
        state_s[i] = state_ro[i];
    }
    __syncthreads();

    // Base pointers for this (batch, head)
    const float* q_bh = q + bh * S * D_K;
    const float* k_bh = k + bh * S * D_K;
    const float* v_bh = v + bh * S * D_V;
    const float* g_bh = g + bh * S;
    const float* beta_bh = beta + bh * S;
    float* out_bh = out + bh * S * D_V;

    // Process sequence token by token (sequential dependency)
    // but parallelize over D_V via thread parallelism
    for (int t = 0; t < S; t++) {
        float g_t = g_bh[t];
        float beta_t = beta_bh[t];
        const float* q_t = q_bh + t * D_K;
        const float* k_t = k_bh + t * D_K;
        const float* v_t = v_bh + t * D_V;

        // --- Decay state: S *= g[t] ---
        #pragma unroll
        for (int i = tid; i < D_K * D_V; i += THREADS) {
            state_s[i] *= g_t;
        }
        __syncthreads();

        // --- kv_mem[dv] = sum_dk(k[dk] * S[dk, dv]) ---
        // Each thread computes a subset of D_V
        // To support arbitrary THREADS vs D_V ratio, use a loop:
        for (int dv = tid; dv < D_V; dv += THREADS) {
            float sum = 0.0f;
            #pragma unroll
            for (int dk = 0; dk < D_K; dk++) {
                sum += k_t[dk] * state_s[dk * D_V + dv];
            }
            // delta[dv] = (v[dv] - kv_mem[dv]) * beta
            float delta_dv = (v_t[dv] - sum) * beta_t;

            // State update: S[dk, dv] += k[dk] * delta[dv]
            #pragma unroll
            for (int dk = 0; dk < D_K; dk++) {
                state_s[dk * D_V + dv] += k_t[dk] * delta_dv;
            }

            // Output: out[t, dv] = sum_dk(q[dk] * S[dk, dv])
            float out_val = 0.0f;
            #pragma unroll
            for (int dk = 0; dk < D_K; dk++) {
                out_val += q_t[dk] * state_s[dk * D_V + dv];
            }
            out_bh[t * D_V + dv] = out_val;
        }
        __syncthreads();
    }

    // --- Write back state ---
    float* state_w = state + bh * D_K * D_V;
    #pragma unroll
    for (int i = tid; i < D_K * D_V; i += THREADS) {
        state_w[i] = state_s[i];
    }
}

// --- Host launcher ---
// Called from C++ code (qwen3_6_kernels.cpp)
// All tensors must be contiguous, FP32, on CUDA device
inline void launch_gated_delta_rule(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    float* state, float* out,
    int BH, int seq_len, int key_dim, int val_dim,
    cudaStream_t stream = 0
) {
    constexpr int D_K = 128;
    constexpr int D_V = 128;
    constexpr int THREADS = 128;  // D_V=128, each thread handles 1 dv

    if (key_dim != D_K || val_dim != D_V) {
        // Should not happen for Qwen3.5/3.6
        fprintf(stderr, "[delta_rule] ERROR: D_K=%d or D_V=%d mismatch\n", key_dim, val_dim);
        return;
    }

    dim3 grid(BH);      // one block per (batch * head)
    dim3 block(THREADS);

    // Dynamic shared memory for state [D_K * D_V] = 128*128*4 = 64KB
    // Exceeds default 48KB, need to opt-in to dynamic shared memory
    size_t smem_size = D_K * D_V * sizeof(float);
    cudaFuncSetAttribute(
        gated_delta_rule_kernel<D_K, D_V, THREADS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    gated_delta_rule_kernel<D_K, D_V, THREADS>
        <<<grid, block, smem_size, stream>>>(
            q, k, v, g_exp, beta, state, out, seq_len
        );
}
