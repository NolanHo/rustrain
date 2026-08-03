// fused_backward.cu — Fused CUDA backward kernels for megakernel.
// Each kernel replaces multiple PyTorch ops with a single kernel launch.
// This eliminates kernel launch overhead and intermediate memory allocation.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// ──────────────────────────────────────────────────────────────────────
// Fused RMSNorm backward: 1 kernel replaces 11 PyTorch ops
//
// Forward: out = input * rsqrt(mean(input^2) + eps) * (1 + weight)
// Backward: grad_input = inv_rms * (grad_normed - normed * mean(grad_normed * normed))
//           grad_weight = sum(grad_out * normed)
//
// Two-pass: pass 1 computes variance + dot (reduction), pass 2 computes grad_input
// ──────────────────────────────────────────────────────────────────────

// Pass 1: compute per-row variance and dot product
// One block per row, each thread handles multiple elements
template<typename T>
__global__ void rms_norm_bwd_pass1_kernel(
    const T* __restrict__ input,     // [N, H]
    const T* __restrict__ grad_out,  // [N, H]
    const T* __restrict__ weight,   // [H]
    float* __restrict__ stats,       // [N, 2] — (var, dot)
    int H, double eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    float partial_sq = 0.0f, partial_dot = 0.0f;

    for (int i = tid; i < H; i += bs) {
        float x = (float)input[row * H + i];
        float g = (float)grad_out[row * H + i];
        float w = (float)weight[i];
        float normed = x * rsqrtf(0.0f); // placeholder, will fix
        partial_sq += x * x;
        // normed = x * inv_rms, but inv_rms depends on variance (reduction)
        // So we compute dot in pass 2 using saved normed
        // Actually, we need to save normed or recompute. Let's save variance first.
    }
    // This approach needs two separate reductions. Let me restructure.

    // Actually, let's just do a single-pass approach:
    // Pass 1: compute variance (sum of x^2) per row → store in stats[row, 0]
    // Pass 2: compute dot(grad_normed, normed) per row → store in stats[row, 1]
    //         Also compute grad_weight contribution
    // Pass 3: compute grad_input

    // But 3 passes is still better than 11 kernels. Let me do 2 passes:
    // Pass 1: variance + grad_weight partial sum
    // Pass 2: dot + grad_input
}

// Actually, let me take a simpler approach. Instead of writing raw CUDA,
// let me use PyTorch's custom op registration with torch::autograd
// to fuse the elementwise ops using at::Tensor operations but with
// fewer intermediate allocations.

// Wait — the user said "we need fusion too". The best approach for fusion
// without writing complex CUDA kernels is to use torch::jit or custom
// CUDA kernels. But actually, the simplest high-impact approach is:

// 1. Write a single CUDA kernel that fuses all elementwise ops per layer
// 2. Use cuBLAS for matmuls (can't fuse those)
// 3. Use Flash Attention for SDPA backward (already optimal)

// Let me write proper CUDA kernels using a simpler design:
// - Each kernel handles one "fused" operation
// - Use cooperative groups or simple block-level reduction

// ──────────────────────────────────────────────────────────────────────
// Fused elementwise: silu_backward + multiply
// grad_x = grad_out * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
// Also: if x2 is provided, grad_x2 = grad_out * silu(x) * sigmoid'(x)
// This fuses silu_backward + gate multiply in one kernel
// ──────────────────────────────────────────────────────────────────────

template<typename T>
__global__ void fused_silu_bwd_kernel(
    const T* __restrict__ x,        // [N]
    const T* __restrict__ grad_out, // [N]
    T* __restrict__ grad_x,         // [N]
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float xv = (float)x[idx];
    float gv = (float)grad_out[idx];
    float sig = 1.0f / (1.0f + expf(-xv));
    float grad = gv * sig * (1.0f + xv * (1.0f - sig));
    grad_x[idx] = (T)grad;
}

// ──────────────────────────────────────────────────────────────────────
// Fused: gate sigmoid backward
// Given: attn_out = sdpa_out * sigmoid(gate)
// grad_sdpa = grad_attn_out * sigmoid(gate)
// grad_gate = grad_attn_out * sigmoid(gate) * (1 - sigmoid(gate))
// ──────────────────────────────────────────────────────────────────────

template<typename T>
__global__ void fused_gate_bwd_kernel(
    const T* __restrict__ gate,          // [N]
    const T* __restrict__ grad_attn_out,// [N]
    T* __restrict__ grad_sdpa,           // [N]
    T* __restrict__ grad_gate,          // [N]
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = (float)gate[idx];
    float ga = (float)grad_attn_out[idx];
    float sig = 1.0f / (1.0f + expf(-g));
    grad_sdpa[idx] = (T)(ga * sig);
    grad_gate[idx] = (T)(ga * sig * (1.0f - sig));
}

// ──────────────────────────────────────────────────────────────────────
// Fused RMSNorm backward: 2 passes
// Pass 1: compute variance per row (reduction)
// Pass 2: compute grad_input + accumulate grad_weight
// ──────────────────────────────────────────────────────────────────────

template<typename T>
__global__ void rms_norm_bwd_variance_kernel(
    const T* __restrict__ input,  // [N, H]
    float* __restrict__ variance,  // [N]
    int H
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    float sum = 0.0f;
    for (int i = tid; i < H; i += bs) {
        float x = (float)input[row * H + i];
        sum += x * x;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Block reduction
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        variance[row] = sdata[0] / (float)H;
    }
}

template<typename T>
__global__ void rms_norm_bwd_grad_kernel(
    const T* __restrict__ input,       // [N, H]
    const T* __restrict__ weight,      // [H]
    const T* __restrict__ grad_out,    // [N, H]
    const float* __restrict__ variance,// [N]
    T* __restrict__ grad_input,        // [N, H]
    T* __restrict__ grad_weight,       // [H] — must be zero-initialized
    int H, double eps
) {
    extern __shared__ float smem[];
    // smem layout: [bs] for dot reduction, [H_aligned] for grad_weight reduction
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    float var = variance[row];
    float inv_rms = rsqrtf(var + (float)eps);

    // First: compute dot = mean(grad_normed * normed)
    // grad_normed = grad_out * (1 + weight)
    // normed = input * inv_rms
    float partial_dot = 0.0f;
    for (int i = tid; i < H; i += bs) {
        float x = (float)input[row * H + i];
        float g = (float)grad_out[row * H + i];
        float w = (float)weight[i];
        float normed = x * inv_rms;
        float grad_normed = g * (1.0f + w);
        partial_dot += grad_normed * normed;
    }
    smem[tid] = partial_dot;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float dot = smem[0] / (float)H;
    __syncthreads();

    // Second: compute grad_input and accumulate grad_weight
    for (int i = tid; i < H; i += bs) {
        float x = (float)input[row * H + i];
        float g = (float)grad_out[row * H + i];
        float w = (float)weight[i];
        float normed = x * inv_rms;
        float grad_normed = g * (1.0f + w);
        float gi = inv_rms * (grad_normed - normed * dot);
        grad_input[row * H + i] = (T)gi;

        // Accumulate grad_weight (atomic since multiple rows write to same column)
        float gw = g * normed;
        atomicAdd((float*)&grad_weight[i], gw);
    }
}

// ──────────────────────────────────────────────────────────────────────
// Fused gated norm backward: combines silu + rms_norm + gate in 2 passes
// Forward: normed = core * inv_rms * norm_w
//          gated = normed * silu(z)
//          out = matmul(gated, out_proj.t())
// Backward: grad_gated = grad_out @ out_proj (done by caller via cuBLAS)
//           grad_normed = grad_gated * silu(z)
//           grad_z = grad_gated * normed * silu'(z)
//           grad_core = rms_norm_backward(core, norm_w, grad_normed)
// ──────────────────────────────────────────────────────────────────────

// Pass 1: compute variance of core per row
template<typename T>
__global__ void gated_norm_bwd_variance_kernel(
    const T* __restrict__ core,    // [N, D]
    float* __restrict__ variance,  // [N]
    int D
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    float sum = 0.0f;
    for (int i = tid; i < D; i += bs) {
        float x = (float)core[row * D + i];
        sum += x * x;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) variance[row] = sdata[0] / (float)D;
}

// Pass 2: compute grad_core, grad_z, accumulate grad_norm_w
template<typename T>
__global__ void gated_norm_bwd_grad_kernel(
    const T* __restrict__ core,         // [N, D]
    const T* __restrict__ z,            // [N, D]
    const T* __restrict__ norm_w,       // [D]
    const T* __restrict__ grad_gated,   // [N, D]
    const float* __restrict__ variance, // [N]
    T* __restrict__ grad_core,          // [N, D]
    T* __restrict__ grad_z,             // [N, D]
    T* __restrict__ grad_norm_w,        // [D] — must be zero-initialized
    int D, double eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    float var = variance[row];
    float inv_rms = rsqrtf(var + (float)eps);

    // Pass 1: compute dot = mean(grad_normed_scaled * (core * inv_rms))
    // grad_normed = grad_gated * silu(z)
    // grad_normed_scaled = grad_normed * norm_w
    float partial_dot = 0.0f;
    for (int i = tid; i < D; i += bs) {
        float c = (float)core[row * D + i];
        float zv = (float)z[row * D + i];
        float gg = (float)grad_gated[row * D + i];
        float nw = (float)norm_w[i];
        float silu_z = zv / (1.0f + expf(-zv));
        float grad_normed = gg * silu_z;
        float grad_normed_scaled = grad_normed * nw;
        float normed_core = c * inv_rms;
        partial_dot += grad_normed_scaled * normed_core;
    }
    smem[tid] = partial_dot;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float dot = smem[0] / (float)D;
    __syncthreads();

    // Pass 2: compute grad_core, grad_z, accumulate grad_norm_w
    for (int i = tid; i < D; i += bs) {
        float c = (float)core[row * D + i];
        float zv = (float)z[row * D + i];
        float gg = (float)grad_gated[row * D + i];
        float nw = (float)norm_w[i];
        float silu_z = zv / (1.0f + expf(-zv));
        float sig_z = 1.0f / (1.0f + expf(-zv));

        float grad_normed = gg * silu_z;
        float normed_core = c * inv_rms;
        float grad_normed_scaled = grad_normed * nw;

        // grad_core
        float gc = inv_rms * (grad_normed_scaled - normed_core * dot);
        grad_core[row * D + i] = (T)gc;

        // grad_z = grad_gated * normed * silu'(z)
        float normed = normed_core * nw;
        float gz = gg * normed * sig_z * (1.0f + zv * (1.0f - sig_z));
        grad_z[row * D + i] = (T)gz;

        // grad_norm_w (atomic accumulate across rows)
        float gnw = grad_normed * normed_core;
        atomicAdd((float*)&grad_norm_w[i], gnw);
    }
}

// ──────────────────────────────────────────────────────────────────────
// Fused L2 norm backward: 2 passes
// Forward: out = x / ||x|| * scale
// Backward: grad_x = (grad_out - out * dot(grad_out, out)) / ||x|| * scale
// ──────────────────────────────────────────────────────────────────────

template<typename T>
__global__ void l2norm_bwd_kernel(
    const T* __restrict__ x,        // [N, D]
    const T* __restrict__ out,      // [N, D] (normalized * scale)
    const T* __restrict__ grad_out, // [N, D]
    T* __restrict__ grad_x,         // [N, D]
    int D, float scale
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    // Pass 1: compute norm and dot
    float partial_sq = 0.0f, partial_dot = 0.0f;
    for (int i = tid; i < D; i += bs) {
        float xv = (float)x[row * D + i];
        float ov = (float)out[row * D + i];
        float gv = (float)grad_out[row * D + i];
        partial_sq += xv * xv;
        partial_dot += gv * ov;
    }
    smem[tid] = partial_sq;
    smem[bs + tid] = partial_dot;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
            smem[bs + tid] += smem[bs + tid + s];
        }
        __syncthreads();
    }
    float norm = sqrtf(smem[0]);
    if (norm < 1e-6f) norm = 1e-6f;
    float dot = smem[bs];
    __syncthreads();

    // Pass 2: compute grad_x
    for (int i = tid; i < D; i += bs) {
        float ov = (float)out[row * D + i];
        float gv = (float)grad_out[row * D + i];
        float gx = (gv - ov * dot) / norm * scale;
        grad_x[row * D + i] = (T)gx;
    }
}

// ──────────────────────────────────────────────────────────────────────
// Fused RoPE backward: single pass elementwise
// grad_q = grad_out * cos + rotate_half(grad_out) * sin
// ──────────────────────────────────────────────────────────────────────

template<typename T>
__global__ void rope_bwd_kernel(
    const T* __restrict__ grad_out,  // [B, H, S, D]
    const T* __restrict__ cos,       // [1, 1, S, D_rot]
    const T* __restrict__ sin,       // [1, 1, S, D_rot]
    T* __restrict__ grad_q,          // [B, H, S, D]
    int N, int rotary_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int d = idx % rotary_dim;
    int rest = idx / rotary_dim;
    int half = rotary_dim / 2;

    float gv = (float)grad_out[idx];
    float cv = (float)cos[d];
    float sv = (float)sin[d];

    if (d < half) {
        // grad_q[d] = grad_out[d] * cos[d] + grad_out[d + half] * sin[d]
        // (rotate_half: [-b, a] → grad_out[d] gets -grad_out[d+half] in forward,
        //  so backward: rotate_half(grad_out)[d] = -grad_out[d+half])
        // Actually: rotate_half([a, b]) = [-b, a]
        // So rotate_half(grad_out)[d] for d < half = -grad_out[d + half]
        float rh = -(float)grad_out[idx + half];
        grad_q[idx] = (T)(gv * cv + rh * sv);
    } else if (d < rotary_dim) {
        // rotate_half(grad_out)[d] for d >= half = grad_out[d - half]
        float rh = (float)grad_out[idx - half];
        grad_q[idx] = (T)(gv * cv + rh * sv);
    } else {
        // Non-rotary part: grad flows directly
        grad_q[idx] = grad_out[idx];
    }
}

// ──────────────────────────────────────────────────────────────────────
// Fused: g backward (linear attention)
// g = -exp(A_log) * softplus(a + dt_bias)
// grad_a = grad_g * (-exp(A_log)) * sigmoid(a + dt_bias)
// ──────────────────────────────────────────────────────────────────────

template<typename T>
__global__ void g_bwd_kernel(
    const T* __restrict__ a,          // [N]
    const T* __restrict__ a_log,      // [H] or [1]
    const T* __restrict__ dt_bias,    // [H] or [1]
    const T* __restrict__ grad_g,     // [N]
    T* __restrict__ grad_a,           // [N]
    int N, int H_per_row
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float av = (float)a[idx];
    int h = idx % H_per_row;
    float al = (float)a_log[h];
    float dt = (float)dt_bias[h];
    float gg = (float)grad_g[idx];

    float neg_exp = -expf(al);
    float x = av + dt;
    float sig = 1.0f / (1.0f + expf(-x));

    grad_a[idx] = (T)(gg * neg_exp * sig);
}

// ──────────────────────────────────────────────────────────────────────
// Fused: beta sigmoid backward
// beta = sigmoid(b)
// grad_b = grad_beta * beta * (1 - beta)
// ──────────────────────────────────────────────────────────────────────

template<typename T>
__global__ void beta_bwd_kernel(
    const T* __restrict__ b,           // [N]
    const T* __restrict__ grad_beta,   // [N]
    T* __restrict__ grad_b,            // [N]
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float bv = (float)b[idx];
    float gb = (float)grad_beta[idx];
    float sig = 1.0f / (1.0f + expf(-bv));
    grad_b[idx] = (T)(gb * sig * (1.0f - sig));
}

// ──────────────────────────────────────────────────────────────────────
// C++ launcher functions
// ──────────────────────────────────────────────────────────────────────

#define LAUNCH_ELEMENTWISE(kernel, ...) \
    int threads = 256; \
    int blocks = (N + threads - 1) / threads; \
    kernel<<<blocks, threads>>>(__VA_ARGS__, (int)N); \

template<typename T>
void launch_fused_silu_bwd(const T* x, const T* grad_out, T* grad_x, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fused_silu_bwd_kernel<T><<<blocks, threads>>>(x, grad_out, grad_x, N);
}

template<typename T>
void launch_fused_gate_bwd(const T* gate, const T* grad_attn_out,
                           T* grad_sdpa, T* grad_gate, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fused_gate_bwd_kernel<T><<<blocks, threads>>>(gate, grad_attn_out, grad_sdpa, grad_gate, N);
}

template<typename T>
void launch_rms_norm_bwd(const T* input, const T* weight, const T* grad_out,
                         T* grad_input, T* grad_weight, int N, int H, double eps,
                         cudaStream_t stream) {
    // Allocate temp
    float* variance;
    cudaMalloc(&variance, N * sizeof(float));

    // Zero grad_weight
    cudaMemsetAsync(grad_weight, 0, H * sizeof(T), stream);

    int threads = (H < 1024) ? H : 1024;
    // Round up to power of 2 for reduction
    int t = 32;
    while (t < threads) t <<= 1;
    threads = t;

    int smem = threads * sizeof(float);

    rms_norm_bwd_variance_kernel<T><<<N, threads, smem, stream>>>(
        input, variance, H);
    rms_norm_bwd_grad_kernel<T><<<N, threads, smem, stream>>>(
        input, weight, grad_out, variance, grad_input, grad_weight, H, eps);

    cudaFree(variance);
}

template<typename T>
void launch_gated_norm_bwd(const T* core, const T* z, const T* norm_w,
                           const T* grad_gated, T* grad_core, T* grad_z,
                           T* grad_norm_w, int N, int D, double eps,
                           cudaStream_t stream) {
    float* variance;
    cudaMalloc(&variance, N * sizeof(float));

    cudaMemsetAsync(grad_norm_w, 0, D * sizeof(T), stream);

    int threads = (D < 1024) ? D : 1024;
    int t = 32;
    while (t < threads) t <<= 1;
    threads = t;
    int smem = threads * sizeof(float);

    gated_norm_bwd_variance_kernel<T><<<N, threads, smem, stream>>>(
        core, variance, D);
    gated_norm_bwd_grad_kernel<T><<<N, threads, smem, stream>>>(
        core, z, norm_w, grad_gated, variance, grad_core, grad_z,
        grad_norm_w, D, eps);

    cudaFree(variance);
}

template<typename T>
void launch_l2norm_bwd(const T* x, const T* out, const T* grad_out,
                       T* grad_x, int N, int D, float scale, cudaStream_t stream) {
    int threads = (D < 1024) ? D : 1024;
    int t = 32;
    while (t < threads) t <<= 1;
    threads = t;
    int smem = 2 * threads * sizeof(float);  // two reductions

    l2norm_bwd_kernel<T><<<N, threads, smem, stream>>>(
        x, out, grad_out, grad_x, D, scale);
}

template<typename T>
void launch_rope_bwd(const T* grad_out, const T* cos, const T* sin,
                     T* grad_q, int N, int rotary_dim) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    rope_bwd_kernel<T><<<blocks, threads>>>(grad_out, cos, sin, grad_q, N, rotary_dim);
}

template<typename T>
void launch_g_bwd(const T* a, const T* a_log, const T* dt_bias,
                  const T* grad_g, T* grad_a, int N, int H_per_row) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    g_bwd_kernel<T><<<blocks, threads>>>(a, a_log, dt_bias, grad_g, grad_a, N, H_per_row);
}

template<typename T>
void launch_beta_bwd(const T* b, const T* grad_beta, T* grad_b, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    beta_bwd_kernel<T><<<blocks, threads>>>(b, grad_beta, grad_b, N);
}

// Explicit instantiation for __nv_bfloat16 and float
#define INSTANTIATE(T) \
    template void launch_fused_silu_bwd<T>(const T*, const T*, T*, int); \
    template void launch_fused_gate_bwd<T>(const T*, const T*, T*, T*, int); \
    template void launch_rms_norm_bwd<T>(const T*, const T*, const T*, T*, T*, int, int, double, cudaStream_t); \
    template void launch_gated_norm_bwd<T>(const T*, const T*, const T*, const T*, T*, T*, T*, int, int, double, cudaStream_t); \
    template void launch_l2norm_bwd<T>(const T*, const T*, const T*, T*, int, int, float, cudaStream_t); \
    template void launch_rope_bwd<T>(const T*, const T*, const T*, T*, int, int); \
    template void launch_g_bwd<T>(const T*, const T*, const T*, const T*, T*, int, int); \
    template void launch_beta_bwd<T>(const T*, const T*, T*, int);

INSTANTIATE(__nv_bfloat16)
INSTANTIATE(float)

// ──────────────────────────────────────────────────────────────────────
// C++ wrappers using at::Tensor — called from qwen3_6_kernels.cpp
// ──────────────────────────────────────────────────────────────────────

#include "fused_backward.h"

at::Tensor fused_silu_backward(const at::Tensor& x, const at::Tensor& grad_out) {
    auto grad_x = at::empty_like(x);
    int N = x.numel();
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kBFloat16) {
        launch_fused_silu_bwd<__nv_bfloat16>(
            (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)grad_out.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)grad_x.data_ptr<at::BFloat16>(), N);
    } else {
        launch_fused_silu_bwd<float>(
            x.data_ptr<float>(), grad_out.data_ptr<float>(),
            grad_x.data_ptr<float>(), N);
    }
    return grad_x;
}

std::tuple<at::Tensor, at::Tensor> fused_gate_backward(
    const at::Tensor& gate, const at::Tensor& grad_attn_out) {
    auto grad_sdpa = at::empty_like(gate);
    auto grad_gate = at::empty_like(gate);
    int N = gate.numel();
    if (gate.scalar_type() == at::kBFloat16) {
        launch_fused_gate_bwd<__nv_bfloat16>(
            (const __nv_bfloat16*)gate.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)grad_attn_out.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)grad_sdpa.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)grad_gate.data_ptr<at::BFloat16>(), N);
    } else {
        launch_fused_gate_bwd<float>(
            gate.data_ptr<float>(), grad_attn_out.data_ptr<float>(),
            grad_sdpa.data_ptr<float>(), grad_gate.data_ptr<float>(), N);
    }
    return {grad_sdpa, grad_gate};
}

std::tuple<at::Tensor, at::Tensor> fused_rms_norm_backward(
    const at::Tensor& input, const at::Tensor& weight,
    const at::Tensor& grad_out, double eps) {
    int N = input.numel() / input.size(-1);
    int H = input.size(-1);
    auto grad_input = at::empty_like(input);
    auto grad_weight = at::empty_like(weight);
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == at::kBFloat16) {
        launch_rms_norm_bwd<__nv_bfloat16>(
            (const __nv_bfloat16*)input.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)weight.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)grad_out.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)grad_input.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)grad_weight.data_ptr<at::BFloat16>(),
            N, H, eps, stream);
    } else {
        launch_rms_norm_bwd<float>(
            input.data_ptr<float>(), weight.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            grad_input.data_ptr<float>(), grad_weight.data_ptr<float>(),
            N, H, eps, stream);
    }
    return {grad_input, grad_weight};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_gated_norm_backward(
    const at::Tensor& core_flat, const at::Tensor& z_flat,
    const at::Tensor& norm_w, const at::Tensor& grad_gated, double eps) {
    int N = core_flat.size(0);
    int D = core_flat.size(1);
    auto grad_core = at::empty_like(core_flat);
    auto grad_z = at::empty_like(z_flat);
    auto grad_norm_w = at::empty_like(norm_w);
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (core_flat.scalar_type() == at::kBFloat16) {
        launch_gated_norm_bwd<__nv_bfloat16>(
            (const __nv_bfloat16*)core_flat.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)z_flat.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)norm_w.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)grad_gated.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)grad_core.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)grad_z.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)grad_norm_w.data_ptr<at::BFloat16>(),
            N, D, eps, stream);
    } else {
        launch_gated_norm_bwd<float>(
            core_flat.data_ptr<float>(), z_flat.data_ptr<float>(),
            norm_w.data_ptr<float>(), grad_gated.data_ptr<float>(),
            grad_core.data_ptr<float>(), grad_z.data_ptr<float>(),
            grad_norm_w.data_ptr<float>(), N, D, eps, stream);
    }
    return {grad_core, grad_z, grad_norm_w};
}

at::Tensor fused_l2norm_backward(
    const at::Tensor& x, const at::Tensor& out,
    const at::Tensor& grad_out, float scale) {
    int N = x.numel() / x.size(-1);
    int D = x.size(-1);
    auto grad_x = at::empty_like(x);
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kBFloat16) {
        launch_l2norm_bwd<__nv_bfloat16>(
            (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)out.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)grad_out.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)grad_x.data_ptr<at::BFloat16>(),
            N, D, scale, stream);
    } else {
        launch_l2norm_bwd<float>(
            x.data_ptr<float>(), out.data_ptr<float>(),
            grad_out.data_ptr<float>(), grad_x.data_ptr<float>(),
            N, D, scale, stream);
    }
    return grad_x;
}

at::Tensor fused_rope_backward(
    const at::Tensor& grad_out, const at::Tensor& cos,
    const at::Tensor& sin, int64_t rotary_dim) {
    auto grad_q = at::empty_like(grad_out);
    int N = grad_out.numel();
    if (grad_out.scalar_type() == at::kBFloat16) {
        launch_rope_bwd<__nv_bfloat16>(
            (const __nv_bfloat16*)grad_out.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)cos.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)sin.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)grad_q.data_ptr<at::BFloat16>(),
            N, (int)rotary_dim);
    } else {
        launch_rope_bwd<float>(
            grad_out.data_ptr<float>(), cos.data_ptr<float>(),
            sin.data_ptr<float>(), grad_q.data_ptr<float>(),
            N, (int)rotary_dim);
    }
    return grad_q;
}

at::Tensor fused_g_backward(
    const at::Tensor& a, const at::Tensor& a_log,
    const at::Tensor& dt_bias, const at::Tensor& grad_g) {
    auto grad_a = at::empty_like(a);
    int N = a.numel();
    int H_per_row = a_log.numel();
    if (a.scalar_type() == at::kBFloat16) {
        launch_g_bwd<__nv_bfloat16>(
            (const __nv_bfloat16*)a.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)a_log.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)dt_bias.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)grad_g.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)grad_a.data_ptr<at::BFloat16>(),
            N, H_per_row);
    } else {
        launch_g_bwd<float>(
            a.data_ptr<float>(), a_log.data_ptr<float>(),
            dt_bias.data_ptr<float>(), grad_g.data_ptr<float>(),
            grad_a.data_ptr<float>(), N, H_per_row);
    }
    return grad_a;
}

at::Tensor fused_beta_backward(
    const at::Tensor& b, const at::Tensor& grad_beta) {
    auto grad_b = at::empty_like(b);
    int N = b.numel();
    if (b.scalar_type() == at::kBFloat16) {
        launch_beta_bwd<__nv_bfloat16>(
            (const __nv_bfloat16*)b.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)grad_beta.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)grad_b.data_ptr<at::BFloat16>(), N);
    } else {
        launch_beta_bwd<float>(
            b.data_ptr<float>(), grad_beta.data_ptr<float>(),
            grad_b.data_ptr<float>(), N);
    }
    return grad_b;
}
