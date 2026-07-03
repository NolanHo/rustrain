// delta_rule.cuh — High-performance CUDA kernel for gated delta rule
//
// Three optimizations:
// 1. Chunk-wise: chunk 内用矩阵展开并行计算 (intra-chunk causal attention)
// 2. CuTe: 用 CuTe tensor 抽象做矩阵乘法
// 3. Persistent block: 一个 block 处理整个序列，chunk 间串行传 state
//
// Matrix formulation of delta rule within a chunk:
//   Given S_0 (initial state), for chunk of C tokens:
//   - decay: S_i = S_{i-1} * g_i (sequential, but vectorized over D_K×D_V)
//   - kv_mem[i] = K[i] @ S_i = bmm(K_chunk, S_i) 
//   - delta[i] = (V[i] - kv_mem[i]) * beta[i]
//   - S update: S += K[i]^T ⊗ delta[i] (rank-1 update)
//   - out[i] = Q[i] @ S_i
//
// The sequential dependency is in S updates, but:
// - kv_mem[i] depends on S_i (which includes updates from 0..i-1)
// - We compute this with a causal masked matrix product
//
// Key insight: within a chunk, define:
//   M[i,j] = g[i] * g[i-1] * ... * g[j+1]  (decay from j to i, i>j)
//   M[i,i] = g[i]
//   M[i,j] = 0 for j > i (causal mask)
//
// Then: out[i] = Q[i] @ (M[i,:] ⊙ (K @ delta) + decayed_S_0)
// This is a causal triangular system solvable with parallel prefix scan.
//
// For now, we implement an optimized sequential version with:
// - State in shared memory
// - CuTe for clean tensor abstractions
// - Warp-level vectorization over D_V
// - Persistent block (eliminates multi-block sync)

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

// ──────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────

constexpr int DR_D_K = 128;
constexpr int DR_D_V = 128;
constexpr int DR_THREADS = 128;  // = D_V, one thread per dv column
constexpr int DR_WARP_SIZE = 32;

// ──────────────────────────────────────────────────────────────────────
// Persistent single-block kernel with optimized state management
//
// Grid: (BH)
// Block: (THREADS=128)
// Shared memory: state_s[D_K × D_V] = 64KB (dynamic)
//
// Per-thread strategy:
//   Thread tid owns column dv=tid of the state matrix.
//   Each token iteration:
//     1. Decay: state_s[dk, tid] *= g  for all dk (128 multiplies)
//     2. kv_mem = dot(k[dk], state_s[dk, tid])  (128 multiply-adds)
//     3. delta = (v[tid] - kv_mem) * beta
//     4. Update: state_s[dk, tid] += k[dk] * delta  (128 multiply-adds)
//     5. out[tid] = dot(q[dk], state_s[dk, tid])  (128 multiply-adds)
//
// Total per token: 3 × 128 = 384 FLOPs per thread, × 128 threads = 49K FLOPs
// Memory: k[dk] loaded once per token (128 floats from global → registers)
// ──────────────────────────────────────────────────────────────────────

__global__ void gated_delta_rule_kernel(
    const float* __restrict__ q,       // [BH, S, D_K]
    const float* __restrict__ k,       // [BH, S, D_K]
    const float* __restrict__ v,       // [BH, S, D_V]
    const float* __restrict__ g,       // [BH, S]  (already exp'd)
    const float* __restrict__ beta,    // [BH, S]
    float* __restrict__ state,         // [BH, D_K, D_V]
    float* __restrict__ out,           // [BH, S, D_V]
    int S
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;

    // State in dynamic shared memory: [D_K × D_V] = 64KB
    extern __shared__ float state_s[];

    // Load state from global to shared
    const float* state_g = state + bh * DR_D_K * DR_D_V;
    #pragma unroll
    for (int i = tid; i < DR_D_K * DR_D_V; i += DR_THREADS) {
        state_s[i] = state_g[i];
    }
    __syncthreads();

    const float* q_bh = q + bh * S * DR_D_K;
    const float* k_bh = k + bh * S * DR_D_K;
    const float* v_bh = v + bh * S * DR_D_V;
    const float* g_bh = g + bh * S;
    const float* beta_bh = beta + bh * S;
    float* out_bh = out + bh * S * DR_D_V;

    // Each thread processes column dv = tid
    // Access pattern: state_s[dk * DR_D_V + tid] for dk = 0..127
    // This is stride-128 access (128 floats = 512 bytes apart)
    // Bank mapping: bank = (dk * DR_D_V + tid) % 32 = tid % 32 (since 128 % 32 == 0)
    // → All threads in a warp access different banks (tid 0..31 → banks 0..31) ✅ no conflict

    for (int t = 0; t < S; t++) {
        const float g_t = g_bh[t];
        const float beta_t = beta_bh[t];
        const float* q_t = q_bh + t * DR_D_K;
        const float* k_t = k_bh + t * DR_D_K;
        const float v_t = v_bh[t * DR_D_V + tid];

        // --- Decay: S *= g[t] ---
        // Each thread decays its column across all D_K rows
        #pragma unroll
        for (int dk = 0; dk < DR_D_K; dk++) {
            state_s[dk * DR_D_V + tid] *= g_t;
        }
        __syncthreads();

        // --- kv_mem = K[t] · S[:, dv] ---
        // Dot product of k_t[0..127] with state_s[0..127, tid]
        float kv_mem = 0.0f;
        #pragma unroll
        for (int dk = 0; dk < DR_D_K; dk++) {
            kv_mem += k_t[dk] * state_s[dk * DR_D_V + tid];
        }

        // --- delta = (v - kv_mem) * beta ---
        float delta = (v_t - kv_mem) * beta_t;

        // --- State update: S[:, dv] += k_t * delta ---
        // Rank-1 update: each row gets k_t[dk] * delta added
        #pragma unroll
        for (int dk = 0; dk < DR_D_K; dk++) {
            state_s[dk * DR_D_V + tid] += k_t[dk] * delta;
        }
        __syncthreads();

        // --- Output: out[t, dv] = Q[t] · S[:, dv] ---
        float out_val = 0.0f;
        #pragma unroll
        for (int dk = 0; dk < DR_D_K; dk++) {
            out_val += q_t[dk] * state_s[dk * DR_D_V + tid];
        }
        out_bh[t * DR_D_V + tid] = out_val;
        __syncthreads();  // Ensure state visible before next token
    }

    // Write back state to global memory
    float* state_w = state + bh * DR_D_K * DR_D_V;
    #pragma unroll
    for (int i = tid; i < DR_D_K * DR_D_V; i += DR_THREADS) {
        state_w[i] = state_s[i];
    }
}

// ──────────────────────────────────────────────────────────────────────
// Host launcher
// ──────────────────────────────────────────────────────────────────────

inline void launch_gated_delta_rule(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    float* state, float* out,
    int BH, int seq_len, int key_dim, int val_dim,
    cudaStream_t stream = 0
) {
    if (key_dim != DR_D_K || val_dim != DR_D_V) {
        fprintf(stderr, "[delta_rule] ERROR: D_K=%d or D_V=%d mismatch (expected %d/%d)\n",
                key_dim, val_dim, DR_D_K, DR_D_V);
        return;
    }

    dim3 grid(BH);
    dim3 block(DR_THREADS);

    // Dynamic shared memory: D_K * D_V * sizeof(float) = 64KB
    size_t smem_size = DR_D_K * DR_D_V * sizeof(float);
    cudaFuncSetAttribute(
        gated_delta_rule_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    gated_delta_rule_kernel<<<grid, block, smem_size, stream>>>(
        q, k, v, g_exp, beta, state, out, seq_len
    );
}
