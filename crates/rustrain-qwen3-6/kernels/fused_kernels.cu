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
    // silu(g) = g * sigmoid(g) — F32 compute, BF16 output (matches PyTorch internal)
    float sig = 1.0f / (1.0f + expf(-g));
    float v = g * sig * u;
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
// 4. Multi-tensor Fused Adam
// ──────────────────────────────────────────────────────────────────────
// One block per param tensor. Each thread processes multiple elements.
// Handles BF16 params with FP32 accumulated grads and FP32 m/v.
//
// Replaces 7 ATen ops per param with 1 kernel launch for ALL params:
//   m = m * beta1 + grad * (1 - beta1)
//   v = v * beta2 + grad^2 * (1 - beta2)
//   param -= lr_scaled * m / (sqrt(v) + eps_scaled)
//   grad = 0

__global__ void fused_adam_multi_kernel(
    void** __restrict__ param_ptrs,   // [n_params] BF16
    void** __restrict__ grad_ptrs,    // [n_params] FP32
    float** __restrict__ m_ptrs,      // [n_params] FP32
    float** __restrict__ v_ptrs,      // [n_params] FP32
    const int* __restrict__ sizes,   // [n_params]
    int n_params,
    float beta1, float beta2,
    float lr_scaled, float eps_scaled,
    float one_minus_beta1, float one_minus_beta2
) {
    int pidx = blockIdx.x;
    if (pidx >= n_params) return;

    int size = sizes[pidx];
    __nv_bfloat16* param = (__nv_bfloat16*)param_ptrs[pidx];
    float* grad = (float*)grad_ptrs[pidx];
    float* m = m_ptrs[pidx];
    float* v = v_ptrs[pidx];

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float g = grad[i];
        float m_new = m[i] * beta1 + g * one_minus_beta1;
        float v_new = v[i] * beta2 + g * g * one_minus_beta2;
        m[i] = m_new;
        v[i] = v_new;
        float p = __bfloat162float(param[i]);
        p -= lr_scaled * m_new / (sqrtf(v_new) + eps_scaled);
        param[i] = __float2bfloat16_rn(p);
        grad[i] = 0.0f;
    }
}

// Out-of-place variant used by dynamic multi-LoRA transactions. It writes
// only destination parameter/state tensors and never mutates source tensors
// or accumulated gradients.
__global__ void fused_adam_multi_out_of_place_kernel(
    void** __restrict__ src_param_ptrs,
    void** __restrict__ grad_ptrs,
    float** __restrict__ src_m_ptrs,
    float** __restrict__ src_v_ptrs,
    void** __restrict__ dst_param_ptrs,
    float** __restrict__ dst_m_ptrs,
    float** __restrict__ dst_v_ptrs,
    const int* __restrict__ sizes,
    const float* __restrict__ lr_scaled,
    const float* __restrict__ eps_scaled,
    const float* __restrict__ beta1,
    const float* __restrict__ beta2,
    int n_params
) {
    int pidx = blockIdx.x;
    if (pidx >= n_params) return;

    int size = sizes[pidx];
    const __nv_bfloat16* src_param =
        (const __nv_bfloat16*)src_param_ptrs[pidx];
    const float* grad = (const float*)grad_ptrs[pidx];
    const float* src_m = src_m_ptrs[pidx];
    const float* src_v = src_v_ptrs[pidx];
    __nv_bfloat16* dst_param = (__nv_bfloat16*)dst_param_ptrs[pidx];
    float* dst_m = dst_m_ptrs[pidx];
    float* dst_v = dst_v_ptrs[pidx];
    const float tensor_lr = lr_scaled[pidx];
    const float tensor_eps = eps_scaled[pidx];
    const float tensor_beta1 = beta1[pidx];
    const float tensor_beta2 = beta2[pidx];
    const float tensor_one_minus_beta1 = 1.0f - tensor_beta1;
    const float tensor_one_minus_beta2 = 1.0f - tensor_beta2;

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float g = grad[i];
        float m_new = src_m[i] * tensor_beta1 +
            g * tensor_one_minus_beta1;
        float v_new = src_v[i] * tensor_beta2 +
            g * g * tensor_one_minus_beta2;
        dst_m[i] = m_new;
        dst_v[i] = v_new;
        float p = __bfloat162float(src_param[i]);
        p -= tensor_lr * m_new / (sqrtf(v_new) + tensor_eps);
        dst_param[i] = __float2bfloat16_rn(p);
    }
}

// Accumulate one FP32 squared L2 norm per logical adapter. Pointer lists only
// contain the unique owners of replicated parameters, so the caller can sum
// these device scalars over the orthogonal process grid without double count.
__global__ void fused_multi_tensor_l2_norm_kernel(
    void** __restrict__ grad_ptrs,
    const int* __restrict__ sizes,
    const int* __restrict__ groups,
    int n_tensors,
    float* __restrict__ norm_squares
) {
    const int tensor_index = blockIdx.x;
    if (tensor_index >= n_tensors) return;

    const float* grad = (const float*)grad_ptrs[tensor_index];
    const int size = sizes[tensor_index];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const float value = grad[i];
        sum += value * value;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) atomicAdd(norm_squares + groups[tensor_index], sum);
    }
}

__global__ void fused_multi_tensor_clip_kernel(
    void** __restrict__ grad_ptrs,
    const int* __restrict__ sizes,
    const int* __restrict__ groups,
    int n_tensors,
    const float* __restrict__ norm_squares,
    float max_norm
) {
    const int tensor_index = blockIdx.x;
    if (tensor_index >= n_tensors) return;
    const float total_norm = sqrtf(norm_squares[groups[tensor_index]]);
    const float scale = fminf(1.0f, max_norm / (total_norm + 1.0e-6f));
    if (scale >= 1.0f) return;

    float* grad = (float*)grad_ptrs[tensor_index];
    const int size = sizes[tensor_index];
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        grad[i] *= scale;
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

void launch_fused_adam_multi(
    void** d_param_ptrs, void** d_grad_ptrs,
    float** d_m_ptrs, float** d_v_ptrs,
    int* d_sizes, int n_params,
    float beta1, float beta2,
    float lr_scaled, float eps_scaled,
    float one_minus_beta1, float one_minus_beta2,
    cudaStream_t stream
) {
    if (n_params <= 0) return;
    int threads = 256;
    int blocks = n_params;
    fused_adam_multi_kernel<<<blocks, threads, 0, stream>>>(
        d_param_ptrs, d_grad_ptrs, d_m_ptrs, d_v_ptrs,
        d_sizes, n_params,
        beta1, beta2, lr_scaled, eps_scaled,
        one_minus_beta1, one_minus_beta2
    );
}

void launch_fused_adam_multi_out_of_place(
    void** d_src_param_ptrs, void** d_grad_ptrs,
    float** d_src_m_ptrs, float** d_src_v_ptrs,
    void** d_dst_param_ptrs,
    float** d_dst_m_ptrs, float** d_dst_v_ptrs,
    int* d_sizes, float* d_lr_scaled, float* d_eps_scaled,
    float* d_beta1, float* d_beta2,
    int n_params,
    cudaStream_t stream
) {
    if (n_params <= 0) return;
    int threads = 256;
    int blocks = n_params;
    fused_adam_multi_out_of_place_kernel<<<blocks, threads, 0, stream>>>(
        d_src_param_ptrs, d_grad_ptrs, d_src_m_ptrs, d_src_v_ptrs,
        d_dst_param_ptrs, d_dst_m_ptrs, d_dst_v_ptrs,
        d_sizes, d_lr_scaled, d_eps_scaled,
        d_beta1, d_beta2,
        n_params
    );
}

void launch_fused_multi_tensor_l2_norm(
    void** d_grad_ptrs, int* d_sizes, int* d_groups,
    int n_tensors, float* d_norm_squares, cudaStream_t stream
) {
    if (n_tensors <= 0) return;
    fused_multi_tensor_l2_norm_kernel<<<n_tensors, 256, 0, stream>>>(
        d_grad_ptrs, d_sizes, d_groups, n_tensors, d_norm_squares);
}

void launch_fused_multi_tensor_clip(
    void** d_grad_ptrs, int* d_sizes, int* d_groups,
    int n_tensors, const float* d_norm_squares, float max_norm,
    cudaStream_t stream
) {
    if (n_tensors <= 0) return;
    fused_multi_tensor_clip_kernel<<<n_tensors, 256, 0, stream>>>(
        d_grad_ptrs, d_sizes, d_groups, n_tensors,
        d_norm_squares, max_norm);
}

}  // extern "C"
