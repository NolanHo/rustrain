#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
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
extern "C" double qwen36_parallel_max_double(void*, double);
extern "C" void qwen36_set_cuda_device(int32_t);
extern "C" void* qwen36_create_training_context_ex(
    void**, int64_t, void*, void*, void*, void*, int64_t, int32_t,
    double, double, double, double, double, int64_t, double, int64_t,
    const int64_t*, int64_t, const char*, int32_t);
extern "C" int32_t qwen36_init_parallel_nccl(
    void*, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" int64_t qwen36_add_lora(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" double qwen36_train_multi_lora_selected(
    void*, void*, void*, void*, const int64_t*, int32_t, int32_t);
extern "C" void qwen36_free_training_context(void*);

namespace {

constexpr int64_t kAbiVersion = 20;
constexpr int32_t kBaseTpAttention = 1 << 0;
constexpr int32_t kVocabParallel = 1 << 2;
constexpr int32_t kExpertParallel = 1 << 3;
constexpr int32_t kBaseTpMlp = 1 << 4;
constexpr int kTpSize = 2;
constexpr int kEpSize = 2;
constexpr int kDpSize = 1;
constexpr int kWorldSize = kTpSize * kEpSize * kDpSize;

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') return fallback;
    const int parsed = std::atoi(value);
    assert(parsed > 0);
    return parsed;
}

int required_env_nonnegative(const char* name) {
    const char* value = std::getenv(name);
    assert(value && value[0] != '\0');
    const int parsed = std::atoi(value);
    assert(parsed >= 0);
    return parsed;
}

bool env_enabled(const char* name, bool fallback = false) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') return fallback;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0;
}

std::string env_string(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && value[0] != '\0' ? value : fallback;
}

std::string json_escape(const std::string& input) {
    std::string output;
    output.reserve(input.size());
    for (const char value : input) {
        switch (value) {
            case '\\': output += "\\\\"; break;
            case '"': output += "\\\""; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default: output += value; break;
        }
    }
    return output;
}

at::Tensor seeded_cpu_randn(
    std::initializer_list<int64_t> shape, double scale, int64_t seed
) {
    at::manual_seed(seed);
    return (at::randn(shape,
        at::TensorOptions().device(at::kCPU).dtype(at::kFloat)) * scale)
        .to(at::kBFloat16);
}

at::Tensor cuda_local(const at::Tensor& tensor) {
    return tensor.contiguous().to(at::kCUDA);
}

at::Tensor unit(std::initializer_list<int64_t> shape) {
    return at::ones(
        shape, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
}

std::vector<void*> pointers(std::vector<at::Tensor>& tensors) {
    std::vector<void*> result;
    result.reserve(tensors.size());
    for (auto& tensor : tensors) result.push_back(&tensor);
    return result;
}

void append_layer_weights(
    std::vector<at::Tensor>& weights, int layer, int tp_rank, int ep_rank,
    int hidden, int heads, int kv_heads, int head_dim,
    int experts, int intermediate, bool expert_tp
) {
    const int local_heads = heads / kTpSize;
    const int local_kv_heads = kv_heads / kTpSize;
    const int local_experts = experts / kEpSize;
    const int local_intermediate = expert_tp
        ? intermediate / kTpSize : intermediate;
    const int64_t common_seed = 1000 + layer * 100;

    weights.push_back(unit({hidden}));
    weights.push_back(unit({hidden}));
    weights.push_back(cuda_local(seeded_cpu_randn(
        {2 * heads * head_dim, hidden}, 0.0020, common_seed + 2).narrow(
            0, tp_rank * 2 * local_heads * head_dim,
            2 * local_heads * head_dim)));
    weights.push_back(unit({head_dim}));
    weights.push_back(cuda_local(seeded_cpu_randn(
        {kv_heads * head_dim, hidden}, 0.0020, common_seed + 4).narrow(
            0, tp_rank * local_kv_heads * head_dim,
            local_kv_heads * head_dim)));
    weights.push_back(unit({head_dim}));
    weights.push_back(cuda_local(seeded_cpu_randn(
        {kv_heads * head_dim, hidden}, 0.0020, common_seed + 6).narrow(
            0, tp_rank * local_kv_heads * head_dim,
            local_kv_heads * head_dim)));
    weights.push_back(cuda_local(seeded_cpu_randn(
        {hidden, heads * head_dim}, 0.0020, common_seed + 7).narrow(
            1, tp_rank * local_heads * head_dim,
            local_heads * head_dim)));
    weights.push_back(cuda_local(seeded_cpu_randn(
        {experts, hidden}, 0.0020, common_seed + 8)));
    weights.push_back(cuda_local(seeded_cpu_randn(
        {1, hidden}, 0.0020, common_seed + 9)));

    auto shared_gate = seeded_cpu_randn(
        {intermediate, hidden}, 0.0020, common_seed + 10);
    auto shared_up = seeded_cpu_randn(
        {intermediate, hidden}, 0.0020, common_seed + 11);
    auto shared_down = seeded_cpu_randn(
        {hidden, intermediate}, 0.0020, common_seed + 12);
    if (expert_tp) {
        shared_gate = shared_gate.narrow(
            0, tp_rank * local_intermediate, local_intermediate);
        shared_up = shared_up.narrow(
            0, tp_rank * local_intermediate, local_intermediate);
        shared_down = shared_down.narrow(
            1, tp_rank * local_intermediate, local_intermediate);
    }
    weights.push_back(cuda_local(shared_gate));
    weights.push_back(cuda_local(shared_up));
    weights.push_back(cuda_local(shared_down));

    auto expert_gate_up = seeded_cpu_randn(
        {experts, 2 * intermediate, hidden}, 0.0020, common_seed + 13)
        .narrow(0, ep_rank * local_experts, local_experts);
    auto expert_down = seeded_cpu_randn(
        {experts, hidden, intermediate}, 0.0020, common_seed + 14)
        .narrow(0, ep_rank * local_experts, local_experts);
    if (expert_tp) {
        expert_gate_up = at::cat({
            expert_gate_up.narrow(
                1, tp_rank * local_intermediate, local_intermediate),
            expert_gate_up.narrow(
                1, intermediate + tp_rank * local_intermediate,
                local_intermediate),
        }, 1);
        expert_down = expert_down.narrow(
            2, tp_rank * local_intermediate, local_intermediate);
    }
    weights.push_back(cuda_local(expert_gate_up));
    weights.push_back(cuda_local(expert_down));

    assert(weights[weights.size() - 5].sizes() ==
        at::IntArrayRef({local_intermediate, hidden}));
    assert(weights[weights.size() - 4].sizes() ==
        at::IntArrayRef({local_intermediate, hidden}));
    assert(weights[weights.size() - 3].sizes() ==
        at::IntArrayRef({hidden, local_intermediate}));
    assert(weights[weights.size() - 2].sizes() ==
        at::IntArrayRef({local_experts, 2 * local_intermediate, hidden}));
    assert(weights[weights.size() - 1].sizes() ==
        at::IntArrayRef({local_experts, hidden, local_intermediate}));
}

double percentile(std::vector<double> values, double quantile) {
    assert(!values.empty());
    std::sort(values.begin(), values.end());
    const double position = quantile * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

double gib(int64_t bytes) {
    return static_cast<double>(std::max<int64_t>(bytes, 0)) /
        (1024.0 * 1024.0 * 1024.0);
}

size_t used_since(size_t initial_free, size_t observed_free) {
    return initial_free > observed_free ? initial_free - observed_free : 0;
}

void set_parallel_env(int rank, int tp_rank, int ep_rank) {
    setenv("WORLD_SIZE", "4", 1);
    setenv("TP_SIZE", "2", 1);
    setenv("EP_SIZE", "2", 1);
    setenv("DP_SIZE", "1", 1);
    setenv("RUSTRAIN_TP_RANK", std::to_string(tp_rank).c_str(), 1);
    setenv("RUSTRAIN_EP_RANK", std::to_string(ep_rank).c_str(), 1);
    setenv("RUSTRAIN_DP_RANK", "0", 1);
    setenv("RANK", std::to_string(rank).c_str(), 1);
}

}  // namespace

int main() {
    const int rank = required_env_nonnegative("RANK");
    const int world = env_int("WORLD_SIZE", -1);
    const int local_rank = required_env_nonnegative("LOCAL_RANK");
    assert(world == kWorldSize && rank >= 0 && rank < world);
    const int tp_rank = rank % kTpSize;
    const int ep_rank = (rank / kTpSize) % kEpSize;
    const int tp_color = ep_rank;
    const int ep_color = tp_rank;
    const int dp_color = rank;
    assert(qwen36_kernel_abi_version() == kAbiVersion);
    assert(env_enabled("QWEN36_EP_A2A"));
    assert(env_enabled("QWEN36_EP_A2A_SHARDED"));
    qwen36_set_cuda_device(local_rank);
    assert(cudaFree(nullptr) == cudaSuccess);
    set_parallel_env(rank, tp_rank, ep_rank);

    const std::string lora_mode = env_string("BENCH_LORA_MODE", "fixed");
    assert(lora_mode == "fixed" || lora_mode == "dynamic");
    const std::string expert_tp_mode = env_string(
        "BENCH_EXPERT_TP_MODE", "etp");
    assert(expert_tp_mode == "replicated" || expert_tp_mode == "etp");
    const bool expert_tp = expert_tp_mode == "etp";
    const std::string variant = env_string("BENCH_VARIANT", "baseline");
    const bool packed_a2a = env_enabled("QWEN36_EP_A2A_PACKED", true);
    const std::string targets = env_string(
        "BENCH_TARGETS", "q_proj,experts_gate_up_proj,experts_down_proj");
    const int batch = env_int("BENCH_BATCH", 2);
    const int seq = env_int("BENCH_SEQ", 128);
    const int hidden = env_int("BENCH_HIDDEN", 1024);
    const int head_dim = env_int("BENCH_HEAD_DIM", 128);
    const int heads = env_int("BENCH_HEADS", hidden / head_dim);
    const int kv_heads = env_int("BENCH_KV_HEADS", heads);
    const int experts = env_int("BENCH_EXPERTS", 8);
    const int intermediate = env_int("BENCH_INTERMEDIATE", 2048);
    const int top_k = env_int("BENCH_TOP_K", 2);
    const int lora_rank = env_int("BENCH_LORA_RANK", 16);
    const int layers = env_int("BENCH_LAYERS", 1);
    const int vocab = env_int("BENCH_VOCAB", 8192);
    const int warmup = env_int("BENCH_WARMUP", 3);
    const int iters = env_int("BENCH_ITERS", 20);
    const int tenants = env_int("BENCH_TENANTS", std::max(batch, 8));
    const int active_tenants = env_int("BENCH_ACTIVE_TENANTS", batch);
    const bool rotate_tenants = env_enabled("BENCH_ROTATE_TENANTS");

    assert(seq >= 2 && hidden > 0 && hidden == heads * head_dim);
    assert(heads % kTpSize == 0 && kv_heads % kTpSize == 0);
    assert(heads % kv_heads == 0);
    assert(experts % kEpSize == 0 && top_k <= experts);
    assert(lora_rank % kTpSize == 0 && vocab % kTpSize == 0);
    assert(!expert_tp || intermediate % kTpSize == 0);
    assert(lora_mode != "dynamic" ||
        (active_tenants == batch && tenants >= active_tenants));

    c10::cuda::CUDACachingAllocator::resetPeakStats(local_rank);
    size_t free_start = 0;
    size_t total_bytes = 0;
    assert(cudaMemGetInfo(&free_start, &total_bytes) == cudaSuccess);
    size_t min_observed_free = free_start;

    std::vector<at::Tensor> weights;
    weights.reserve(static_cast<size_t>(layers) * 15);
    for (int layer = 0; layer < layers; ++layer) {
        append_layer_weights(weights, layer, tp_rank, ep_rank,
            hidden, heads, kv_heads, head_dim, experts, intermediate,
            expert_tp);
    }
    for (auto& weight : weights) weight.set_requires_grad(false);
    auto weight_ptrs = pointers(weights);

    const int local_vocab = vocab / kTpSize;
    auto embed = cuda_local(seeded_cpu_randn(
        {vocab, hidden}, 0.0020, 41).narrow(
            0, tp_rank * local_vocab, local_vocab));
    auto final_norm = unit({hidden});
    auto lm_head = cuda_local(seeded_cpu_randn(
        {vocab, hidden}, 0.0020, 47).narrow(
            0, tp_rank * local_vocab, local_vocab));
    embed.set_requires_grad(false);
    final_norm.set_requires_grad(false);
    lm_head.set_requires_grad(false);

    std::vector<LayerConfig> configs(layers);
    for (auto& config : configs) {
        config.layer_type = 0;
        config.num_heads = heads;
        config.num_kv_heads = kv_heads;
        config.head_dim = head_dim;
        config.partial_rotary_factor = 1.0;
        config.rope_theta = 10000.0;
        config.rms_eps = 1e-5;
        config.num_experts = experts;
        config.top_k = top_k;
        config.moe_intermediate = intermediate;
        config.expert_start = ep_rank * (experts / kEpSize);
        config.expert_count = experts / kEpSize;
        config.norm_topk_prob = 1;
    }
    std::vector<int64_t> target_layers(layers);
    std::iota(target_layers.begin(), target_layers.end(), 0);

    void* context = qwen36_create_training_context_ex(
        weight_ptrs.data(), static_cast<int64_t>(weight_ptrs.size()),
        &embed, &final_norm, &lm_head, configs.data(), layers,
        static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, lora_rank,
        target_layers.data(), layers, targets.c_str(),
        kBaseTpAttention | kVocabParallel | kExpertParallel |
            (expert_tp ? kBaseTpMlp : 0));
    assert(context);
    assert(qwen36_init_parallel_nccl(
        context, rank, world,
        tp_rank, kTpSize, tp_color,
        ep_rank, kEpSize, ep_color,
        0, kDpSize, dp_color) == 0);

    std::vector<int64_t> adapter_ids;
    if (lora_mode == "dynamic") {
        adapter_ids.reserve(tenants);
        for (int tenant = 0; tenant < tenants; ++tenant) {
            const int64_t adapter_id = qwen36_add_lora(
                context, lora_rank, static_cast<double>(lora_rank),
                target_layers.data(), layers, targets.c_str());
            assert(adapter_id > 0);
            adapter_ids.push_back(adapter_id);
        }
    }

    std::vector<int64_t> host_ids(static_cast<size_t>(batch) * seq);
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq; ++s) {
            const int64_t global_source =
                static_cast<int64_t>(ep_rank) * batch + b;
            host_ids[static_cast<size_t>(b) * seq + s] =
                1 + (global_source * 131 + s * 17) % (vocab - 1);
        }
    }
    auto input_ids = at::from_blob(host_ids.data(), {batch, seq},
        at::TensorOptions().device(at::kCPU).dtype(at::kLong))
        .clone().to(at::kCUDA);
    auto target_mask = at::ones({batch, seq},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    auto attention_mask = at::ones({batch, seq},
        at::TensorOptions().device(at::kCUDA).dtype(at::kBool));

    size_t free_after_context = 0;
    assert(cudaMemGetInfo(&free_after_context, &total_bytes) == cudaSuccess);
    min_observed_free = std::min(min_observed_free, free_after_context);

    std::vector<int64_t> selected(active_tenants);
    auto select_tenants = [&](int step) {
        if (lora_mode != "dynamic") return;
        const int start = rotate_tenants
            ? (step * active_tenants) % tenants : 0;
        for (int index = 0; index < active_tenants; ++index) {
            selected[index] = adapter_ids[(start + index) % tenants];
        }
    };
    auto train_once = [&](int step) {
        if (lora_mode == "fixed") {
            return qwen36_train_step(
                context, &input_ids, &target_mask, &attention_mask);
        }
        select_tenants(step);
        return qwen36_train_multi_lora_selected(
            context, &input_ids, &target_mask, &attention_mask,
            selected.data(), active_tenants, lora_rank);
    };

    double last_loss = 0.0;
    for (int step = 0; step < warmup; ++step) {
        last_loss = train_once(step);
        assert(last_loss > 0.0 && std::isfinite(last_loss));
    }
    assert(cudaDeviceSynchronize() == cudaSuccess);
    size_t free_after_warmup = 0;
    assert(cudaMemGetInfo(&free_after_warmup, &total_bytes) == cudaSuccess);
    min_observed_free = std::min(min_observed_free, free_after_warmup);

    std::vector<double> times_ms;
    times_ms.reserve(iters);
    for (int iteration = 0; iteration < iters; ++iteration) {
        assert(cudaDeviceSynchronize() == cudaSuccess);
        const auto start = std::chrono::steady_clock::now();
        last_loss = train_once(warmup + iteration);
        assert(cudaDeviceSynchronize() == cudaSuccess);
        const auto stop = std::chrono::steady_clock::now();
        assert(last_loss > 0.0 && std::isfinite(last_loss));
        const double local_ms = std::chrono::duration<double, std::milli>(
            stop - start).count();
        const double world_max_ms = qwen36_parallel_max_double(context, local_ms);
        assert(world_max_ms > 0.0 && std::isfinite(world_max_ms));
        times_ms.push_back(world_max_ms);
        size_t free_now = 0;
        assert(cudaMemGetInfo(&free_now, &total_bytes) == cudaSuccess);
        min_observed_free = std::min(min_observed_free, free_now);
    }

    const double mean = std::accumulate(
        times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
    double variance = 0.0;
    for (const double value : times_ms) {
        variance += (value - mean) * (value - mean);
    }
    variance /= times_ms.size();
    const double p50 = percentile(times_ms, 0.50);
    const double p95 = percentile(times_ms, 0.95);
    const double unique_tokens =
        static_cast<double>(batch) * seq * kEpSize;
    const double loss_tokens =
        static_cast<double>(batch) * (seq - 1) * kEpSize;
    const double routed_tokens = unique_tokens * top_k * layers;
    const double unique_tokens_per_sec = unique_tokens / (p50 / 1000.0);
    const double routed_tokens_per_sec = routed_tokens / (p50 / 1000.0);

    const auto allocator_stats =
        c10::cuda::CUDACachingAllocator::getDeviceStats(local_rank);
    // Aggregate is the stable first entry of CUDACachingAllocator StatArray;
    // the enum namespace differs across prebuilt PyTorch releases.
    constexpr size_t aggregate = 0;
    const auto& allocated = allocator_stats.allocated_bytes[aggregate];
    const auto& reserved = allocator_stats.reserved_bytes[aggregate];
    cudaDeviceProp properties{};
    assert(cudaGetDeviceProperties(&properties, local_rank) == cudaSuccess);

    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
        << "native_tp_ep_bench {"
        << "\"variant\":\"" << json_escape(variant) << "\","
        << "\"lora_mode\":\"" << lora_mode << "\","
        << "\"expert_tp_mode\":\"" << expert_tp_mode << "\","
        << "\"targets\":\"" << json_escape(targets) << "\","
        << "\"rank\":" << rank << ",\"world\":" << world << ','
        << "\"tp_rank\":" << tp_rank << ",\"tp_size\":" << kTpSize << ','
        << "\"ep_rank\":" << ep_rank << ",\"ep_size\":" << kEpSize << ','
        << "\"gpu\":\"" << json_escape(properties.name) << "\","
        << "\"abi\":" << kAbiVersion << ','
        << "\"ep_a2a\":true,\"ep_a2a_sharded\":true,"
        << "\"ep_a2a_packed\":" << (packed_a2a ? "true" : "false") << ','
        << "\"timing_scope\":\"world_max\","
        << "\"batch\":" << batch << ",\"seq\":" << seq << ','
        << "\"hidden\":" << hidden << ",\"heads\":" << heads << ','
        << "\"kv_heads\":" << kv_heads << ",\"head_dim\":" << head_dim << ','
        << "\"layers\":" << layers << ",\"experts\":" << experts << ','
        << "\"intermediate\":" << intermediate << ','
        << "\"global_intermediate\":" << intermediate << ','
        << "\"local_intermediate\":"
        << (expert_tp ? intermediate / kTpSize : intermediate) << ','
        << "\"expert_base_replication_factor\":"
        << (expert_tp ? 1 : kTpSize) << ','
        << "\"top_k\":" << top_k << ','
        << "\"vocab\":" << vocab << ",\"lora_rank\":" << lora_rank << ','
        << "\"tenants\":" << (lora_mode == "dynamic" ? tenants : 0) << ','
        << "\"active_tenants\":"
        << (lora_mode == "dynamic" ? active_tenants : 0) << ','
        << "\"rotate_tenants\":" << (rotate_tenants ? "true" : "false") << ','
        << "\"warmup\":" << warmup << ",\"iters\":" << iters << ','
        << "\"last_loss\":" << last_loss << ','
        << "\"step_ms_mean\":" << mean << ','
        << "\"step_ms_p50\":" << p50 << ','
        << "\"step_ms_p95\":" << p95 << ','
        << "\"step_ms_std\":" << std::sqrt(variance) << ','
        << "\"unique_tokens\":" << unique_tokens << ','
        << "\"loss_tokens\":" << loss_tokens << ','
        << "\"routed_tokens\":" << routed_tokens << ','
        << "\"unique_tokens_per_sec\":" << unique_tokens_per_sec << ','
        << "\"routed_tokens_per_sec\":" << routed_tokens_per_sec << ','
        << "\"device_total_gib\":" << gib(total_bytes) << ','
        << "\"free_start_gib\":" << gib(free_start) << ','
        << "\"free_after_context_gib\":" << gib(free_after_context) << ','
        << "\"free_after_warmup_gib\":" << gib(free_after_warmup) << ','
        << "\"max_observed_resident_gib\":"
        << gib(static_cast<int64_t>(used_since(free_start, min_observed_free))) << ','
        << "\"allocator_current_allocated_gib\":" << gib(allocated.current) << ','
        << "\"allocator_peak_allocated_gib\":" << gib(allocated.peak) << ','
        << "\"allocator_current_reserved_gib\":" << gib(reserved.current) << ','
        << "\"allocator_peak_reserved_gib\":" << gib(reserved.peak) << ','
        << "\"samples_ms\":[";
    for (size_t index = 0; index < times_ms.size(); ++index) {
        if (index) output << ',';
        output << times_ms[index];
    }
    output << "]}\n";
    const std::string line = output.str();
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fflush(stdout);

    qwen36_free_training_context(context);
    return 0;
}
