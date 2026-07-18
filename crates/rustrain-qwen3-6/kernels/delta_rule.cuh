// delta_rule.cuh — High-performance CUDA kernel for gated delta rule
//
// Forward: gated_delta_rule_kernel — sequential state update
// Backward: gated_delta_rule_backward_kernel — reverse pass with gradient
//
// Forward per token t:
//   S = S * g[t]                        (decay)
//   kv = sum_dk(S * k[t])                (key-value memory)
//   delta = (v[t] - kv) * beta[t]        (innovation)
//   S = S + k[t] ⊗ delta                  (state update)
//   out[t] = sum_dk(S * q[t])             (output)
//
// Backward per token t (reverse):
//   grad_S += outer(grad_out[t], q[t])
//   grad_q[t] = S · grad_out[t]
//   grad_delta = grad_S · k[t]
//   grad_k += grad_S · delta
//   grad_v = grad_delta * beta
//   grad_kv = -grad_delta * beta
//   grad_beta = grad_delta · (v - kv)
//   grad_S_from_kv = outer(k[t], grad_kv)
//   grad_k += S_d · grad_kv
//   grad_g += sum(grad_S_d * S_prev)
//   grad_S_prev = grad_S_d * g[t]
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
    float* __restrict__ delta_buf,     // [BH, S, D_V] — saved for backward
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

        // Save delta for the fused autograd backward. The reference backward
        // path is selected explicitly and does not use this CUDA kernel.
        if (delta_buf != nullptr) {
            delta_buf[bh * S * DR_D_V + t * DR_D_V + tid] = delta;
        }

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
// Host launcher (forward)
// ──────────────────────────────────────────────────────────────────────

inline void launch_gated_delta_rule(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    float* state, float* out, float* delta_buf,
    int BH, int seq_len, int key_dim, int val_dim,
    cudaStream_t stream
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
        q, k, v, g_exp, beta, state, out, delta_buf, seq_len
    );
}

// ──────────────────────────────────────────────────────────────────────
// Backward kernel: reverse pass with gradient tracking
//
// Inputs: forward saved values (q, k, v, g_exp, beta, final_state)
//         grad_out [BH, S, D_V]
// Outputs: grad_q, grad_k, grad_v, grad_g, grad_beta [BH, S, D_*]
//
// Algorithm (reverse, per token t = S-1 down to 0):
//   1. Undo state update: S_before = S - k[t] ⊗ delta[t]
//      Need delta[t] = (v[t] - kv_mem[t]) * beta[t]
//      kv_mem[t] = S_before · k[t]  (recomputed)
//   2. grad_S += outer(grad_out[t], q[t])     → grad_q[t] = S · grad_out[t]
//   3. grad_delta = grad_S · k[t]             → grad_k_update = grad_S · delta
//   4. grad_v[t] = grad_delta * beta
//      grad_kv = -grad_delta * beta
//      grad_beta[t] = grad_delta · (v - kv)
//   5. grad_S_from_kv = outer(k[t], grad_kv)  → grad_k_kv = S_d · grad_kv
//   6. grad_g[t] = sum(grad_S_d * S_before)
//      grad_S = grad_S_d * g[t]  (undo decay)
//
// Shared memory: 3 × [D_K × D_V] = 192KB (state, grad_S, S_before)
// ──────────────────────────────────────────────────────────────────────

__global__ void gated_delta_rule_backward_kernel(
    const float* __restrict__ q,         // [BH, S, D_K]
    const float* __restrict__ k,         // [BH, S, D_K]
    const float* __restrict__ v,         // [BH, S, D_V]
    const float* __restrict__ g_exp,     // [BH, S]  (already exp'd)
    const float* __restrict__ beta,      // [BH, S]
    const float* __restrict__ final_state,// [BH, D_K, D_V] — state after all tokens
    const float* __restrict__ delta_buf, // [BH, S, D_V] — saved from forward
    const float* __restrict__ grad_out, // [BH, S, D_V]
    float* __restrict__ grad_q,          // [BH, S, D_K]
    float* __restrict__ grad_k,           // [BH, S, D_K]
    float* __restrict__ grad_v,           // [BH, S, D_V]
    float* __restrict__ grad_g,           // [BH, S]
    float* __restrict__ grad_beta,        // [BH, S]
    int S
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;

    // Shared memory: 3 state matrices
    extern __shared__ float smem[];
    float* state_s = smem;                          // [D_K * D_V] — current state (forward)
    float* grad_S_s = state_s + DR_D_K * DR_D_V;   // [D_K * D_V] — grad w.r.t. state
    // No third matrix — recompute delta inline

    // Load final state (after all tokens) into shared
    const float* state_g = final_state + bh * DR_D_K * DR_D_V;
    for (int i = tid; i < DR_D_K * DR_D_V; i += DR_THREADS) {
        state_s[i] = state_g[i];
        grad_S_s[i] = 0.0f;
    }
    __syncthreads();

    const float* q_bh = q + bh * S * DR_D_K;
    const float* k_bh = k + bh * S * DR_D_K;
    const float* v_bh = v + bh * S * DR_D_V;
    const float* g_bh = g_exp + bh * S;
    const float* beta_bh = beta + bh * S;
    const float* go_bh = grad_out + bh * S * DR_D_V;

    float* gq_bh = grad_q + bh * S * DR_D_K;
    float* gk_bh = grad_k + bh * S * DR_D_K;
    float* gv_bh = grad_v + bh * S * DR_D_V;
    float* gg_bh = grad_g + bh * S;
    float* gb_bh = grad_beta + bh * S;

    // Reverse pass: t = S-1 down to 0
    for (int t = S - 1; t >= 0; t--) {
        const float g_t = g_bh[t];
        const float beta_t = beta_bh[t];
        const float* q_t = q_bh + t * DR_D_K;
        const float* k_t = k_bh + t * DR_D_K;
        const float v_t = v_bh[t * DR_D_V + tid];
        const float go_t = go_bh[t * DR_D_V + tid];
        const float delta_t = delta_buf[bh * S * DR_D_V + t * DR_D_V + tid];

        // --- Step 1: Undo state update to get S_before ---
        // Forward: S_after = S_before + k ⊗ delta
        // Undo: S_before = S_after - k ⊗ delta
        #pragma unroll
        for (int dk = 0; dk < DR_D_K; dk++) {
            state_s[dk * DR_D_V + tid] -= k_t[dk] * delta_t;
        }
        __syncthreads();

        // Now state_s = S_before * g[t] (state after decay, before update)
        // S_before = state_s / g[t]

        // --- Step 2: grad_q[t] = S · grad_out[t] ---
        // grad_S += outer(grad_out[t], q[t])
        float gq_val = 0.0f;
        #pragma unroll
        for (int dk = 0; dk < DR_D_K; dk++) {
            float s_val = state_s[dk * DR_D_V + tid];
            grad_S_s[dk * DR_D_V + tid] += go_t * q_t[dk];
            gq_val += s_val * go_t;
        }
        gq_bh[t * DR_D_K + tid] = gq_val;

        // --- Step 3: grad_delta = grad_S · k[t], grad_k from update ---
        // grad_S contains d(loss)/d(S_after_update)
        // d(S_update)/d(delta) = k[t] → grad_delta = grad_S · k[t]
        // d(S_update)/d(k) = delta[t] → grad_k_update = grad_S · delta[t]
        float gdelta = 0.0f;
        float gk_update = 0.0f;
        #pragma unroll
        for (int dk = 0; dk < DR_D_K; dk++) {
            float gs = grad_S_s[dk * DR_D_V + tid];
            gdelta += gs * k_t[dk];
            gk_update += gs * delta_t;
        }

        // --- Step 4: grad_v = grad_delta * beta ---
        gv_bh[t * DR_D_V + tid] = gdelta * beta_t;

        // grad_kv = -grad_delta * beta
        float gkv = -gdelta * beta_t;

        // grad_beta = grad_delta · (v - kv)
        // kv = (S_before_g · k).sum(dk) — recompute
        float kv_mem = 0.0f;
        #pragma unroll
        for (int dk = 0; dk < DR_D_K; dk++) {
            kv_mem += state_s[dk * DR_D_V + tid] * k_t[dk];
        }
        if (tid == 0) {
            // grad_beta is scalar per (bh, t) — accumulate across dv
            // Actually beta is [BH, S], not per-dv. Need reduction.
            // For now, store per-dv and reduce later.
        }
        gb_bh[t] = 0.0f;  // legacy kernel; not referenced by the host launcher

        // --- Step 5: grad_k from kv ---
        // d(kv)/d(k) = S_before_g → grad_k_kv = S_before_g · grad_kv
        float gk_kv = 0.0f;
        #pragma unroll
        for (int dk = 0; dk < DR_D_K; dk++) {
            gk_kv += state_s[dk * DR_D_V + tid] * gkv;
        }
        gk_bh[t * DR_D_K + tid] = gk_update + gk_kv;

        // --- Step 6: grad_S from kv → undo decay ---
        // d(kv)/d(S_before_g) = k[t] → grad_S_from_kv = outer(k[t], grad_kv)
        // d(decay)/d(S_before) = g[t] → grad_S = grad_S_from_kv * g[t]
        // Plus grad_S from output already accumulated
        #pragma unroll
        for (int dk = 0; dk < DR_D_K; dk++) {
            grad_S_s[dk * DR_D_V + tid] = (grad_S_s[dk * DR_D_V + tid] + k_t[dk] * gkv) * g_t;
        }

        // --- Step 7: grad_g = sum(grad_S_before_decay * S_before) ---
        // This needs S_before = state_s / g[t], and grad_S_before_decay
        // For simplicity, approximate grad_g as 0 (g is exp of a_log, small gradient)
        gg_bh[t] = 0.0f;  // legacy kernel; not referenced by the host launcher

        __syncthreads();
    }
}

// Correct reverse-mode recurrence. The older kernel above is retained only as
// a historical reference and is not used by any launcher; this kernel computes
// all q/k/v/g/beta gradients.
// The forward recurrence is:
//   R = g * S_prev, kv = k^T R, delta = beta * (v - kv),
//   S = R + k outer delta, out = q^T S.
__global__ void gated_delta_rule_backward_kernel_correct(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g_exp,
    const float* __restrict__ beta,
    const float* __restrict__ final_state,
    const float* __restrict__ delta_buf,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    float* __restrict__ grad_g,
    float* __restrict__ grad_beta,
    int S
) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    extern __shared__ float smem[];
    float* state_s = smem;  // [D_K, D_V] = S_t while entering each step
    float* grad_s = state_s + DR_D_K * DR_D_V;
    // Four warp partials for reductions over D_V, followed by two scalar
    // reductions. This avoids atomics and keeps the 128-thread block intact.
    float* reduce_q = grad_s + DR_D_K * DR_D_V;
    float* reduce_k = reduce_q + 4 * DR_D_K;
    float* reduce_beta = reduce_k + 4 * DR_D_K;
    float* reduce_g = reduce_beta + 4;

    const float* state_g = final_state + bh * DR_D_K * DR_D_V;
    for (int i = tid; i < DR_D_K * DR_D_V; i += DR_THREADS) {
        state_s[i] = state_g[i];
        grad_s[i] = 0.0f;
    }
    __syncthreads();

    const float* q_bh = q + bh * S * DR_D_K;
    const float* k_bh = k + bh * S * DR_D_K;
    const float* v_bh = v + bh * S * DR_D_V;
    const float* g_bh = g_exp + bh * S;
    const float* beta_bh = beta + bh * S;
    const float* go_bh = grad_out + bh * S * DR_D_V;
    float* gq_bh = grad_q + bh * S * DR_D_K;
    float* gk_bh = grad_k + bh * S * DR_D_K;
    float* gv_bh = grad_v + bh * S * DR_D_V;
    float* gg_bh = grad_g + bh * S;
    float* gb_bh = grad_beta + bh * S;

    for (int t = S - 1; t >= 0; --t) {
        const float* q_t = q_bh + t * DR_D_K;
        const float* k_t = k_bh + t * DR_D_K;
        const float g_t = g_bh[t];
        const float beta_t = beta_bh[t];
        const float go_t = go_bh[t * DR_D_V + tid];
        const float delta_t = delta_buf[bh * S * DR_D_V + t * DR_D_V + tid];

        // Add the direct output contribution to dS_t and reduce dQ_t over
        // value columns while state_s still contains the post-update state.
        for (int dk = 0; dk < DR_D_K; ++dk) {
            const int idx = dk * DR_D_V + tid;
            const float s_after = state_s[idx];
            grad_s[idx] += q_t[dk] * go_t;
            float q_part = s_after * go_t;
            for (int off = 16; off > 0; off >>= 1)
                q_part += __shfl_down_sync(0xffffffff, q_part, off);
            if (lane == 0) reduce_q[warp * DR_D_K + dk] = q_part;
        }
        __syncthreads();
        if (tid < DR_D_K) {
            gq_bh[t * DR_D_K + tid] =
                reduce_q[tid] + reduce_q[DR_D_K + tid] +
                reduce_q[2 * DR_D_K + tid] + reduce_q[3 * DR_D_K + tid];
        }
        __syncthreads();

        // Undo S_t = R_t + k outer delta, leaving R_t = g_t*S_prev.
        for (int dk = 0; dk < DR_D_K; ++dk)
            state_s[dk * DR_D_V + tid] -= k_t[dk] * delta_t;
        __syncthreads();

        // ddelta = dS_t^T k and h = ddelta * beta. The same value-column
        // thread computes dV and one partial for dBeta/dG.
        float gdelta = 0.0f;
        float kv_mem = 0.0f;
        for (int dk = 0; dk < DR_D_K; ++dk) {
            const int idx = dk * DR_D_V + tid;
            gdelta += grad_s[idx] * k_t[dk];
            kv_mem += state_s[idx] * k_t[dk];
        }
        const float h = gdelta * beta_t;
        gv_bh[t * DR_D_V + tid] = h;
        const float beta_part = gdelta * (v_bh[t * DR_D_V + tid] - kv_mem);
        const float safe_g = fmaxf(g_t, 1.0e-8f);
        float g_part = 0.0f;

        // dR = dS - k outer h. Reduce dK over value columns and update dS_prev.
        for (int dk = 0; dk < DR_D_K; ++dk) {
            const int idx = dk * DR_D_V + tid;
            const float r = state_s[idx];
            const float s_prev = r / safe_g;
            const float grad_r = grad_s[idx] - k_t[dk] * h;
            const float k_part = grad_s[idx] * delta_t - r * h;
            g_part += grad_r * s_prev;
            float reduced_k = k_part;
            for (int off = 16; off > 0; off >>= 1)
                reduced_k += __shfl_down_sync(0xffffffff, reduced_k, off);
            if (lane == 0) reduce_k[warp * DR_D_K + dk] = reduced_k;
            grad_s[idx] = grad_r * g_t;
            // The next reverse iteration starts from S_prev, not R_t.
            state_s[idx] = s_prev;
        }

        float reduced_beta = beta_part;
        float reduced_g = g_part;
        for (int off = 16; off > 0; off >>= 1) {
            reduced_beta += __shfl_down_sync(0xffffffff, reduced_beta, off);
            reduced_g += __shfl_down_sync(0xffffffff, reduced_g, off);
        }
        if (lane == 0) {
            reduce_beta[warp] = reduced_beta;
            reduce_g[warp] = reduced_g;
        }
        __syncthreads();
        if (tid < DR_D_K) {
            gk_bh[t * DR_D_K + tid] =
                reduce_k[tid] + reduce_k[DR_D_K + tid] +
                reduce_k[2 * DR_D_K + tid] + reduce_k[3 * DR_D_K + tid];
        }
        if (tid == 0) {
            gb_bh[t] = reduce_beta[0] + reduce_beta[1] +
                       reduce_beta[2] + reduce_beta[3];
            gg_bh[t] = reduce_g[0] + reduce_g[1] +
                       reduce_g[2] + reduce_g[3];
        }
        __syncthreads();
    }
}

// ──────────────────────────────────────────────────────────────────────
// Host launcher (backward)
// ──────────────────────────────────────────────────────────────────────

inline int launch_gated_delta_rule_backward(
    const float* q, const float* k, const float* v,
    const float* g_exp, const float* beta,
    const float* final_state,
    const float* delta_buf,
    const float* grad_out,
    float* grad_q, float* grad_k, float* grad_v,
    float* grad_g, float* grad_beta,
    int BH, int seq_len, int key_dim, int val_dim,
    cudaStream_t stream
) {
    if (key_dim != DR_D_K || val_dim != DR_D_V) {
        fprintf(stderr, "[delta_rule_backward] ERROR: D_K=%d or D_V=%d mismatch\n", key_dim, val_dim);
        return -1;
    }

    dim3 grid(BH);
    dim3 block(DR_THREADS);

    // 2 state matrices plus four warp partials per D_K and scalar reductions.
    size_t smem_size = (2 * DR_D_K * DR_D_V + 8 * DR_D_K + 8) * sizeof(float);
    auto attr_status = cudaFuncSetAttribute(
        gated_delta_rule_backward_kernel_correct,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (attr_status != cudaSuccess) {
        fprintf(stderr, "[delta_rule_backward] shared-memory attribute failed: %s\n",
                cudaGetErrorString(attr_status));
        return static_cast<int>(attr_status);
    }

    gated_delta_rule_backward_kernel_correct<<<grid, block, smem_size, stream>>>(
        q, k, v, g_exp, beta, final_state, delta_buf, grad_out,
        grad_q, grad_k, grad_v, grad_g, grad_beta, seq_len
    );
    return 0;
}
