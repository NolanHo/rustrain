// fused_kernels.cu — Hand-written CUDA kernels for element-wise fusion.
//
// Shape-generic, no Python/Tilelang dependency.
// Compiled with nvcc, linked into libqwen36_kernels.so / libv4_flash_kernels.so.
//
// Kernels:
//   1. fused_rmsnorm: RMSNorm in single kernel (replaces 3 ATen ops)
//   2. fused_swiglu: silu(gate) * up + clamp in single kernel (replaces 3 ATen ops)
//   3. fused_rmsnorm_matmul: RMSNorm + matmul fused (normed values stay in SRAM)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// ──────────────────────────────────────────────────────────────────────
// 1. Fused RMSNorm
// ──────────────────────────────────────────────────────────────────────
// input [M, K] → output [M, K], weight [K], eps
// output[i, k] = input[i, k] * rsqrt(mean(input[i,:]^2) + eps) * weight[k]
//
// One block per row, 256 threads, K must be <= 8192 (tiled if larger).

__global__ void fused_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int M, int K, float eps, float scale_mode  // scale_mode: 0=weight, 1=1+weight
) {
    int row = blockIdx.x;
    if (row >= M) return;

    const __nv_bfloat16* in_row = input + row * K;
    __nv_bfloat16* out_row = output + row * K;

    // Phase 1: compute sum of squares (reduction within block)
    extern __shared__ __nv_bfloat16 smem_rms[];
    float* smem = (float*)smem_rms;
    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = __bfloat162float(in_row[k]);
        local_sum += v * v;
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }

    // Block reduce via shared memory
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = local_sum;
    __syncthreads();

    float inv_rms;
    if (threadIdx.x < blockDim.x / 32) {
        float v = smem[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, offset);
        if (threadIdx.x == 0) smem[0] = v;
    }
    __syncthreads();
    inv_rms = rsqrtf(smem[0] / float(K) + eps);

    // Phase 2: normalize and write
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float w = __bfloat162float(weight[k]);
        if (scale_mode > 0.5f) w = 1.0f + w;  // Qwen3.6: 1 + weight
        float v = __bfloat162float(in_row[k]) * inv_rms * w;
        out_row[k] = __float2bfloat16_rn(v);
    }
}

// ──────────────────────────────────────────────────────────────────────
// 2. Fused SwiGLU: silu(gate) * up, with optional clamp
// ──────────────────────────────────────────────────────────────────────
// gate [M, I], up [M, I] → out [M, I]
// out = silu(gate) * up, clamped to [-limit, limit]

__global__ void fused_swiglu_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int N, float limit  // N = M * I
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
    float silu_g = g / (1.0f + expf(-g));
    float v = silu_g * u;
    if (limit > 0.0f) {
        v = fmaxf(fminf(v, limit), -limit);
    }
    out[idx] = __float2bfloat16_rn(v);
}

// ──────────────────────────────────────────────────────────────────────
// 3. Fused RMSNorm + Matmul (for attention input norm + q_proj)
// ──────────────────────────────────────────────────────────────────────
// input [M, K], weight [K], matmul_w [N, K] → output [M, N]
// output[m, n] = sum_k(rmsnorm(input[m, k], weight, eps) * matmul_w[n, k])
//
// One block per (M_tile, N_tile), tiled matmul with RMSNorm fused.
// Block size: 16x16 tiles, 256 threads.

#define FMM_BLOCK_M 16
#define FMM_BLOCK_N 64
#define FMM_THREADS 256

__global__ void fused_rmsnorm_matmul_kernel(
    const __nv_bfloat16* __restrict__ input,   // [M, K]
    const __nv_bfloat16* __restrict__ weight,  // [K]
    const __nv_bfloat16* __restrict__ matmul_w, // [N, K]
    __nv_bfloat16* __restrict__ output,        // [M, N]
    int M, int N, int K, float eps, float scale_mode
) {
    int m_start = blockIdx.x * FMM_BLOCK_M;
    int n_start = blockIdx.y * FMM_BLOCK_N;
    int tid = threadIdx.x;

    // Shared memory: X tile [FMM_BLOCK_M, K_tile] + W tile [FMM_BLOCK_N, K_tile] + rms_val
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* X_s = smem;                          // [FMM_BLOCK_M, K]
    __nv_bfloat16* W_s = X_s + FMM_BLOCK_M * K;        // [FMM_BLOCK_N, K]
    float* rms_val = (float*)(W_s + FMM_BLOCK_N * K);  // [FMM_BLOCK_M]

    // Load X tile and compute RMSNorm per row
    for (int k = tid; k < K; k += FMM_THREADS) {
        for (int i = 0; i < FMM_BLOCK_M && m_start + i < M; i++) {
            X_s[i * K + k] = input[(m_start + i) * K + k];
        }
    }
    __syncthreads();

    // Compute RMSNorm for each row in the tile
    if (tid < FMM_BLOCK_M && m_start + tid < M) {
        float sum_sq = 0.0f;
        for (int k = 0; k < K; k++) {
            float v = __bfloat162float(X_s[tid * K + k]);
            sum_sq += v * v;
        }
        rms_val[tid] = rsqrtf(sum_sq / float(K) + eps);
    }
    __syncthreads();

    // Apply RMSNorm + weight to X tile
    for (int k = tid; k < K; k += FMM_THREADS) {
        float w = __bfloat162float(weight[k]);
        if (scale_mode > 0.5f) w = 1.0f + w;
        for (int i = 0; i < FMM_BLOCK_M && m_start + i < M; i++) {
            float v = __bfloat162float(X_s[i * K + k]) * rms_val[i] * w;
            X_s[i * K + k] = __float2bfloat16_rn(v);
        }
    }
    __syncthreads();

    // Load W tile
    for (int k = tid; k < K; k += FMM_THREADS) {
        for (int j = 0; j < FMM_BLOCK_N && n_start + j < N; j++) {
            W_s[j * K + k] = matmul_w[(n_start + j) * K + k];
        }
    }
    __syncthreads();

    // Tiled matmul: output[m, n] = sum_k(X_s[m, k] * W_s[n, k])
    // Each thread computes one output element
    int m_local = tid / (FMM_BLOCK_N / 4);  // m within tile
    int n_local = tid % (FMM_BLOCK_N / 4);  // n within tile (stride 4 for coalescing)

    for (int m_off = 0; m_off < FMM_BLOCK_M; m_off += FMM_THREADS / FMM_BLOCK_N) {
        int m = m_local + m_off;
        if (m >= FMM_BLOCK_M || m_start + m >= M) continue;
        for (int n_off = 0; n_off < 4; n_off++) {
            int n = n_local * 4 + n_off;
            if (n >= FMM_BLOCK_N || n_start + n >= N) continue;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += __bfloat162float(X_s[m * K + k]) * __bfloat162float(W_s[n * K + k]);
            }
            output[(m_start + m) * N + (n_start + n)] = __float2bfloat16_rn(acc);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// C wrapper functions (called from C++ via extern "C")
// ──────────────────────────────────────────────────────────────────────

extern "C" {

void launch_fused_rmsnorm(
    const void* input, const void* weight, void* output,
    int M, int K, float eps, int scale_mode, cudaStream_t stream
) {
    int threads = 256;
    int blocks = M;
    int smem_size = (threads / 32) * sizeof(float);
    fused_rmsnorm_kernel<<<blocks, threads, smem_size, stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)output, M, K, eps, (float)scale_mode
    );
}

void launch_fused_swiglu(
    const void* gate, const void* up, void* output,
    int N, float limit, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fused_swiglu_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
        (__nv_bfloat16*)output, N, limit
    );
}

void launch_fused_rmsnorm_matmul(
    const void* input, const void* weight, const void* matmul_w, void* output,
    int M, int N, int K, float eps, int scale_mode, cudaStream_t stream
) {
    dim3 grid((M + FMM_BLOCK_M - 1) / FMM_BLOCK_M, (N + FMM_BLOCK_N - 1) / FMM_BLOCK_N);
    dim3 block(FMM_THREADS);
    int smem_size = (FMM_BLOCK_M * K + FMM_BLOCK_N * K) * sizeof(__nv_bfloat16) + FMM_BLOCK_M * sizeof(float);
    fused_rmsnorm_matmul_kernel<<<grid, block, smem_size, stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)weight,
        (const __nv_bfloat16*)matmul_w, (__nv_bfloat16*)output,
        M, N, K, eps, (float)scale_mode
    );
}

}  // extern "C"
