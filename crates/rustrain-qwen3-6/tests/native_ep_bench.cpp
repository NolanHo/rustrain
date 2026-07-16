#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <numeric>
#include <vector>

struct LayerConfig {
    int64_t layer_type, num_heads, num_kv_heads, head_dim;
    int64_t num_k_heads, key_dim, num_v_heads, val_dim, conv_kernel;
    double partial_rotary_factor, rope_theta, rms_eps;
    int64_t num_experts, top_k, moe_intermediate, expert_start, expert_count;
    int64_t intermediate_size;
    int32_t norm_topk_prob;
    void* nccl_comm;
    void* nccl_stream;
};

extern "C" void* qwen36_create_training_context(
    void**, int64_t, void*, void*, void*, void*, int64_t, int32_t,
    double, double, double, double, double, int64_t, double, int64_t,
    const int64_t*, int64_t, const char*);
extern "C" int32_t qwen36_init_nccl(void*);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" void qwen36_free_training_context(void*);

static int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    return value ? std::max(1, std::atoi(value)) : fallback;
}

static at::Tensor cuda_rand(std::initializer_list<int64_t> shape) {
    return at::randn(shape,
        at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
}

static std::vector<void*> tensor_ptrs(std::vector<at::Tensor>& tensors) {
    std::vector<void*> result;
    result.reserve(tensors.size());
    for (auto& tensor : tensors) result.push_back(&tensor);
    return result;
}

static double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

int main() {
    const int rank = std::atoi(std::getenv("RANK") ? std::getenv("RANK") : "0");
    const int world = std::atoi(
        std::getenv("WORLD_SIZE") ? std::getenv("WORLD_SIZE") : "1");
    const int local_rank = std::atoi(
        std::getenv("LOCAL_RANK") ? std::getenv("LOCAL_RANK") : "0");
    const bool a2a = std::getenv("QWEN36_EP_A2A") &&
        std::atoi(std::getenv("QWEN36_EP_A2A")) != 0;
    const bool sharded = a2a && std::getenv("QWEN36_EP_A2A_SHARDED") &&
        std::atoi(std::getenv("QWEN36_EP_A2A_SHARDED")) != 0;
    const int seq = env_int("BENCH_SEQ", 128);
    const int hidden = env_int("BENCH_HIDDEN", 256);
    const int experts = env_int("BENCH_EXPERTS", 8);
    const int intermediate = env_int("BENCH_INTERMEDIATE", 256);
    const int warmup = env_int("BENCH_WARMUP", 2);
    const int iters = env_int("BENCH_ITERS", 10);
    assert(world >= 2 && rank >= 0 && rank < world);
    assert(hidden % 8 == 0 && experts % world == 0);
    assert(!sharded || a2a);
    c10::cuda::CUDAGuard guard(local_rank);
    at::manual_seed(1234);

    const int local_experts = experts / world;
    const int64_t expert_start = rank * local_experts;
    const int vocab = std::max(1024, hidden * 4);
    const int head_dim = hidden;

    // Build one deterministic global model, then narrow only expert tensors.
    std::vector<at::Tensor> global_weights;
    global_weights.push_back(cuda_rand({hidden}));
    global_weights.push_back(cuda_rand({hidden}));
    global_weights.push_back(cuda_rand({2 * head_dim, hidden}));
    global_weights.push_back(cuda_rand({head_dim}));
    global_weights.push_back(cuda_rand({head_dim, hidden}));
    global_weights.push_back(cuda_rand({head_dim}));
    global_weights.push_back(cuda_rand({head_dim, hidden}));
    global_weights.push_back(cuda_rand({hidden, head_dim}));
    global_weights.push_back(cuda_rand({experts, hidden}));
    global_weights.push_back(cuda_rand({1, hidden}));
    global_weights.push_back(cuda_rand({intermediate, hidden}));
    global_weights.push_back(cuda_rand({intermediate, hidden}));
    global_weights.push_back(cuda_rand({hidden, intermediate}));
    global_weights.push_back(cuda_rand({experts, 2 * intermediate, hidden}));
    global_weights.push_back(cuda_rand({experts, hidden, intermediate}));
    for (auto& weight : global_weights) weight.set_requires_grad(false);

    std::vector<at::Tensor> local_weights = global_weights;
    local_weights[13] = global_weights[13]
        .narrow(0, expert_start, local_experts).contiguous();
    local_weights[14] = global_weights[14]
        .narrow(0, expert_start, local_experts).contiguous();
    auto weights = tensor_ptrs(local_weights);
    auto embed = cuda_rand({vocab, hidden});
    auto final_norm = cuda_rand({hidden});
    auto lm_head = cuda_rand({vocab, hidden});
    embed.set_requires_grad(false);
    final_norm.set_requires_grad(false);
    lm_head.set_requires_grad(false);

    LayerConfig config{};
    config.layer_type = 0;
    config.num_heads = 1;
    config.num_kv_heads = 1;
    config.head_dim = head_dim;
    config.partial_rotary_factor = 1.0;
    config.rope_theta = 10000.0;
    config.rms_eps = 1e-5;
    config.num_experts = experts;
    config.top_k = 2;
    config.moe_intermediate = intermediate;
    config.expert_start = expert_start;
    config.expert_count = local_experts;
    config.norm_topk_prob = 1;

    const int64_t target_layer = 0;
    const char* targets = "experts_gate_up_proj,experts_down_proj";
    void* ctx = qwen36_create_training_context(
        weights.data(), static_cast<int64_t>(weights.size()),
        &embed, &final_norm, &lm_head, &config, 1,
        static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, 8,
        &target_layer, 1, targets);
    assert(ctx);
    assert(qwen36_init_nccl(ctx) == 0);

    std::vector<int64_t> ids(seq);
    for (int i = 0; i < seq; ++i) {
        const int offset = sharded ? rank * seq : 0;
        ids[i] = (offset + i + 1) % vocab;
    }
    auto input_ids = at::from_blob(ids.data(), {1, seq},
        at::TensorOptions().device(at::kCPU).dtype(at::kLong)).clone().to(at::kCUDA);
    auto target_mask = at::ones({1, seq},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    auto attention_mask = at::ones({1, seq},
        at::TensorOptions().device(at::kCUDA).dtype(at::kBool));

    for (int i = 0; i < warmup; ++i) {
        (void)qwen36_train_step(ctx, &input_ids, &target_mask, &attention_mask);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start = nullptr, stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<double> times;
    times.reserve(iters);
    double last_loss = 0.0;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start, c10::cuda::getCurrentCUDAStream().stream());
        last_loss = qwen36_train_step(
            ctx, &input_ids, &target_mask, &attention_mask);
        cudaEventRecord(stop, c10::cuda::getCurrentCUDAStream().stream());
        cudaEventSynchronize(stop);
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        times.push_back(static_cast<double>(elapsed_ms));
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const double mean = std::accumulate(times.begin(), times.end(), 0.0) /
        static_cast<double>(times.size());
    double variance = 0.0;
    for (const double value : times) variance += (value - mean) * (value - mean);
    variance /= static_cast<double>(times.size());
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    const double local_tokens = static_cast<double>(std::max(seq - 1, 1));
    const double processed_tokens = local_tokens * world;
    // Legacy EP replicates the input batch on every rank; only sharded A2A
    // represents distinct global samples in this synthetic harness.
    const double unique_tokens = sharded ? processed_tokens : local_tokens;
    const double median_seconds = percentile(times, 0.5) / 1000.0;
    const double processed_tokens_per_sec = processed_tokens / median_seconds;
    const double unique_tokens_per_sec = unique_tokens / median_seconds;
    std::printf(
        "native_qwen36_ep_bench rank=%d world=%d a2a=%d sharded=%d "
        "seq=%d hidden=%d experts=%d intermediate=%d warmup=%d iters=%d "
        "local_tokens=%.0f processed_tokens=%.0f unique_tokens=%.0f last_loss=%.8f "
        "step_ms_mean=%.4f step_ms_median=%.4f step_ms_std=%.4f "
        "processed_tokens_per_sec=%.2f unique_tokens_per_sec=%.2f "
        "free_mem_gib=%.3f\n",
        rank, world, a2a, sharded, seq, hidden, experts, intermediate,
        warmup, iters, local_tokens, processed_tokens, unique_tokens, last_loss,
        mean, percentile(times, 0.5), std::sqrt(variance),
        processed_tokens_per_sec, unique_tokens_per_sec,
        static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0));
    std::fflush(stdout);
    qwen36_free_training_context(ctx);
    return 0;
}
