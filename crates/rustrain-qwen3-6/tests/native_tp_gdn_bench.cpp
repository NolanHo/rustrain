#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <numeric>
#include <sstream>
#include <string>
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

extern "C" int64_t qwen36_kernel_abi_version();
extern "C" void qwen36_set_cuda_device(int32_t);
extern "C" void* qwen36_create_training_context(
    void**, int64_t, void*, void*, void*, void*, int64_t, int32_t,
    double, double, double, double, double, int64_t, double, int64_t,
    const int64_t*, int64_t, const char*);
extern "C" void* qwen36_create_training_context_ex(
    void**, int64_t, void*, void*, void*, void*, int64_t, int32_t,
    double, double, double, double, double, int64_t, double, int64_t,
    const int64_t*, int64_t, const char*, int32_t);
extern "C" int32_t qwen36_init_nccl(void*);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" void qwen36_free_training_context(void*);

namespace {

constexpr int64_t kAbiVersion = 15;
constexpr int32_t kBaseTpAttention = 1 << 0;

static int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') return fallback;
    const int parsed = std::atoi(value);
    assert(parsed > 0);
    return parsed;
}

static int env_int_or(const char* primary, const char* secondary, int fallback) {
    return std::getenv(primary) ? env_int(primary, fallback)
                                : env_int(secondary, fallback);
}

static at::Tensor seeded_randn(
    std::initializer_list<int64_t> shape, double scale, int64_t seed
) {
    at::manual_seed(seed);
    return (at::randn(shape,
        at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)) * scale)
        .to(at::kBFloat16);
}

static at::Tensor unit_weight(std::initializer_list<int64_t> shape) {
    return at::ones(
        shape, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
}

static std::vector<void*> pointers(std::vector<at::Tensor>& tensors) {
    std::vector<void*> result;
    result.reserve(tensors.size());
    for (auto& tensor : tensors) result.push_back(&tensor);
    return result;
}

static void append_gdn_layer(
    std::vector<at::Tensor>& weights, int64_t layer,
    int64_t hidden, int64_t intermediate,
    int64_t global_k_heads, int64_t global_v_heads,
    int64_t key_dim, int64_t value_dim, int64_t conv_kernel,
    int tp_world
) {
    const int64_t local_k_heads = global_k_heads / tp_world;
    const int64_t local_v_heads = global_v_heads / tp_world;
    const int64_t q_size = local_k_heads * key_dim;
    const int64_t v_size = local_v_heads * value_dim;
    const int64_t qkv_size = 2 * q_size + v_size;
    const int64_t seed = 1000 + layer * 100;

    weights.push_back(unit_weight({hidden}));
    weights.push_back(unit_weight({hidden}));
    weights.push_back(seeded_randn({qkv_size, hidden}, 0.0020, seed + 2));
    weights.push_back(seeded_randn({v_size, hidden}, 0.0020, seed + 3));
    weights.push_back(seeded_randn({local_v_heads, hidden}, 0.0020, seed + 4));
    weights.push_back(seeded_randn({local_v_heads, hidden}, 0.0020, seed + 5));
    // A realistic negative time-step bias keeps the recurrent decay near one.
    // A zero bias would produce decay ~= 0.5 and make reverse state recovery
    // exponentially ill-conditioned over this synthetic long sequence.
    weights.push_back(at::zeros({local_v_heads},
        at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    weights.push_back(at::full({local_v_heads}, -4.0,
        at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    weights.push_back(seeded_randn(
        {qkv_size, 1, conv_kernel}, 0.0020, seed + 8));
    weights.push_back(unit_weight({value_dim}));
    weights.push_back(seeded_randn({hidden, v_size}, 0.0020, seed + 10));
    weights.push_back(seeded_randn(
        {intermediate, hidden}, 0.0020, seed + 11));
    weights.push_back(seeded_randn(
        {intermediate, hidden}, 0.0020, seed + 12));
    weights.push_back(seeded_randn(
        {hidden, intermediate}, 0.0020, seed + 13));
}

static double percentile(std::vector<double> values, double quantile) {
    assert(!values.empty());
    std::sort(values.begin(), values.end());
    const double position = quantile * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

static double gib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

static size_t used_since(size_t initial_free, size_t observed_free) {
    return initial_free > observed_free ? initial_free - observed_free : 0;
}

}  // namespace

int main() {
    const char* mode_env = std::getenv("BENCH_MODE");
    const std::string mode = mode_env ? mode_env : "single";
    assert(mode == "single" || mode == "tp2");
    const bool use_tp = mode == "tp2";
    const int expected_world = use_tp ? 2 : 1;
    const int rank = std::getenv("RANK") ? std::atoi(std::getenv("RANK")) : 0;
    const int world = std::getenv("WORLD_SIZE")
        ? std::atoi(std::getenv("WORLD_SIZE")) : 1;
    const int local_rank = std::getenv("LOCAL_RANK")
        ? std::atoi(std::getenv("LOCAL_RANK")) : rank;
    assert(world == expected_world && rank >= 0 && rank < world);
    assert(qwen36_kernel_abi_version() == kAbiVersion);
    qwen36_set_cuda_device(local_rank);
    assert(cudaFree(nullptr) == cudaSuccess);

    const int batch = env_int("BENCH_BATCH", 2);
    const int seq = env_int("BENCH_SEQ", 512);
    const int hidden = env_int("BENCH_HIDDEN", 2048);
    const int key_heads = env_int("BENCH_K_HEADS", 16);
    const int value_heads = env_int("BENCH_V_HEADS", 32);
    const int key_dim = env_int_or("BENCH_KEY_DIM", "BENCH_HEAD_DIM", 128);
    const int value_dim = env_int_or("BENCH_VALUE_DIM", "BENCH_HEAD_DIM", 128);
    const int conv_kernel = env_int("BENCH_CONV", 4);
    const int layers = env_int("BENCH_LAYERS", 3);
    const int intermediate = env_int("BENCH_INTERMEDIATE", 2048);
    const int vocab = env_int("BENCH_VOCAB", 4096);
    const int lora_rank = env_int("BENCH_LORA_RANK", 8);
    const int warmup = env_int("BENCH_WARMUP", 5);
    const int iters = env_int("BENCH_ITERS", 30);
    assert(seq >= 2 && hidden > 0 && key_dim > 0 && value_dim > 0);
    assert(value_heads % key_heads == 0);
    assert(key_heads % expected_world == 0 && value_heads % expected_world == 0);
    assert(lora_rank % expected_world == 0);

    size_t free_start = 0, total_bytes = 0;
    assert(cudaMemGetInfo(&free_start, &total_bytes) == cudaSuccess);
    size_t min_observed_free = free_start;

    std::vector<at::Tensor> weights;
    weights.reserve(layers * 14);
    for (int64_t layer = 0; layer < layers; ++layer) {
        append_gdn_layer(weights, layer, hidden, intermediate,
            key_heads, value_heads, key_dim, value_dim, conv_kernel,
            expected_world);
    }
    for (auto& weight : weights) weight.set_requires_grad(false);
    auto weight_ptrs = pointers(weights);

    auto embed = seeded_randn({vocab, hidden}, 0.0020, 31);
    auto final_norm = unit_weight({hidden});
    auto lm_head = seeded_randn({vocab, hidden}, 0.0020, 37);
    embed.set_requires_grad(false);
    final_norm.set_requires_grad(false);
    lm_head.set_requires_grad(false);

    std::vector<LayerConfig> configs(layers);
    for (auto& config : configs) {
        config.layer_type = 1;
        config.num_k_heads = key_heads;
        config.key_dim = key_dim;
        config.num_v_heads = value_heads;
        config.val_dim = value_dim;
        config.conv_kernel = conv_kernel;
        config.rms_eps = 1e-5;
        config.intermediate_size = intermediate;
    }
    std::vector<int64_t> target_layers(layers);
    std::iota(target_layers.begin(), target_layers.end(), 0);
    constexpr const char* targets =
        "in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj";

    std::vector<int64_t> host_ids(batch * seq);
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s)
            host_ids[b * seq + s] = 1 + (b * 131 + s * 17) % (vocab - 1);
    }
    auto input_ids = at::from_blob(host_ids.data(), {batch, seq},
        at::TensorOptions().device(at::kCPU).dtype(at::kLong)).clone().to(at::kCUDA);
    auto target_mask = at::ones({batch, seq},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    auto attention_mask = at::ones({batch, seq},
        at::TensorOptions().device(at::kCUDA).dtype(at::kBool));

    size_t free_before_context = 0;
    assert(cudaMemGetInfo(&free_before_context, &total_bytes) == cudaSuccess);
    min_observed_free = std::min(min_observed_free, free_before_context);
    setenv("TP_SIZE", use_tp ? "2" : "1", 1);
    unsetenv("RUSTRAIN_DATA_PARALLEL");
    void* context = use_tp
        ? qwen36_create_training_context_ex(
            weight_ptrs.data(), weight_ptrs.size(), &embed, &final_norm, &lm_head,
            configs.data(), layers, static_cast<int32_t>(at::kBFloat16),
            1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, lora_rank,
            target_layers.data(), layers, targets, kBaseTpAttention)
        : qwen36_create_training_context(
            weight_ptrs.data(), weight_ptrs.size(), &embed, &final_norm, &lm_head,
            configs.data(), layers, static_cast<int32_t>(at::kBFloat16),
            1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, lora_rank,
            target_layers.data(), layers, targets);
    assert(context);
    if (use_tp) assert(qwen36_init_nccl(context) == 0);

    size_t free_after_context = 0;
    assert(cudaMemGetInfo(&free_after_context, &total_bytes) == cudaSuccess);
    min_observed_free = std::min(min_observed_free, free_after_context);
    double last_loss = 0.0;
    for (int i = 0; i < warmup; ++i) {
        last_loss = qwen36_train_step(
            context, &input_ids, &target_mask, &attention_mask);
        assert(std::isfinite(last_loss));
    }
    assert(cudaDeviceSynchronize() == cudaSuccess);

    size_t free_after_warmup = 0;
    assert(cudaMemGetInfo(&free_after_warmup, &total_bytes) == cudaSuccess);
    min_observed_free = std::min(min_observed_free, free_after_warmup);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    assert(cudaEventCreate(&start) == cudaSuccess);
    assert(cudaEventCreate(&stop) == cudaSuccess);
    std::vector<double> times;
    times.reserve(iters);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    for (int i = 0; i < iters; ++i) {
        assert(cudaEventRecord(start, stream) == cudaSuccess);
        last_loss = qwen36_train_step(
            context, &input_ids, &target_mask, &attention_mask);
        assert(cudaEventRecord(stop, stream) == cudaSuccess);
        assert(cudaEventSynchronize(stop) == cudaSuccess);
        assert(std::isfinite(last_loss));
        float elapsed_ms = 0.0f;
        assert(cudaEventElapsedTime(&elapsed_ms, start, stop) == cudaSuccess);
        times.push_back(elapsed_ms);
        size_t free_now = 0;
        assert(cudaMemGetInfo(&free_now, &total_bytes) == cudaSuccess);
        min_observed_free = std::min(min_observed_free, free_now);
    }
    assert(cudaEventDestroy(start) == cudaSuccess);
    assert(cudaEventDestroy(stop) == cudaSuccess);

    const double mean = std::accumulate(times.begin(), times.end(), 0.0) /
        static_cast<double>(times.size());
    double variance = 0.0;
    for (const double value : times) variance += (value - mean) * (value - mean);
    variance /= static_cast<double>(times.size());
    const double p50 = percentile(times, 0.50);
    const double p90 = percentile(times, 0.90);
    const double model_tokens = static_cast<double>(batch) * (seq - 1);
    const double layer_tokens = model_tokens * layers;
    const double model_tokens_per_second = model_tokens / (p50 / 1000.0);
    const double layer_tokens_per_second = layer_tokens / (p50 / 1000.0);

    cudaDeviceProp properties{};
    assert(cudaGetDeviceProperties(&properties, local_rank) == cudaSuccess);
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
        << "native_tp_gdn_bench {"
        << "\"mode\":\"" << mode << "\","
        << "\"rank\":" << rank << ",\"world\":" << world << ","
        << "\"gpu\":\"" << properties.name << "\","
        << "\"abi\":" << kAbiVersion << ","
        << "\"batch\":" << batch << ",\"seq\":" << seq << ","
        << "\"hidden\":" << hidden << ",\"layers\":" << layers << ","
        << "\"key_heads\":" << key_heads << ","
        << "\"value_heads\":" << value_heads << ","
        << "\"key_dim\":" << key_dim << ","
        << "\"value_dim\":" << value_dim << ","
        << "\"n_rep\":" << (value_heads / key_heads) << ","
        << "\"conv\":" << conv_kernel << ","
        << "\"intermediate\":" << intermediate << ","
        << "\"vocab\":" << vocab << ",\"lora_rank\":" << lora_rank << ","
        << "\"warmup\":" << warmup << ",\"iters\":" << iters << ","
        << "\"last_loss\":" << last_loss << ","
        << "\"step_ms_mean\":" << mean << ","
        << "\"step_ms_p50\":" << p50 << ","
        << "\"step_ms_p90\":" << p90 << ","
        << "\"step_ms_std\":" << std::sqrt(variance) << ","
        << "\"model_tokens\":" << model_tokens << ","
        << "\"gdn_layer_tokens\":" << layer_tokens << ","
        << "\"model_tokens_per_sec\":" << model_tokens_per_second << ","
        << "\"gdn_layer_tokens_per_sec\":" << layer_tokens_per_second << ","
        << "\"device_total_gib\":" << gib(total_bytes) << ","
        << "\"free_start_gib\":" << gib(free_start) << ","
        << "\"free_before_context_gib\":" << gib(free_before_context) << ","
        << "\"free_after_context_gib\":" << gib(free_after_context) << ","
        << "\"free_after_warmup_gib\":" << gib(free_after_warmup) << ","
        << "\"context_resident_delta_gib\":"
        << gib(used_since(free_before_context, free_after_context)) << ","
        << "\"max_observed_resident_gib\":"
        << gib(used_since(free_start, min_observed_free)) << ","
        << "\"samples_ms\":[";
    for (size_t i = 0; i < times.size(); ++i) {
        if (i) output << ',';
        output << times[i];
    }
    output << "]}\n";
    const std::string line = output.str();
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fflush(stdout);

    qwen36_free_training_context(context);
    return 0;
}
