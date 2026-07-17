#include <ATen/ATen.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
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
extern "C" int64_t qwen36_add_lora(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" void* qwen36_get_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t);
extern "C" int32_t qwen36_set_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t, void*);
extern "C" void* qwen36_get_adapter_optimizer_tensor(
    void*, int64_t, int64_t, const char*, int32_t, int32_t);
extern "C" void* qwen36_get_lora_a(void*, int64_t);
extern "C" void* qwen36_get_lora_b(void*, int64_t);
extern "C" int32_t qwen36_set_lora_tensor(void*, int64_t, int32_t, void*);
extern "C" void* qwen36_get_lora_grad_accumulator(void*, int64_t, int32_t);
extern "C" int32_t qwen36_abort_gradient_accumulation(void*);
extern "C" int64_t qwen36_export_optimizer_state(
    void*, void**, void**, int64_t);
extern "C" double qwen36_eval_step(void*, void*, void*, void*);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" double qwen36_train_micro_step(
    void*, void*, void*, void*, double, int32_t);
extern "C" double qwen36_train_multi_lora_selected(
    void*, void*, void*, void*, const int64_t*, int32_t, int32_t);
extern "C" void qwen36_free_training_context(void*);

namespace {

constexpr int64_t kAbiVersion = 15;
constexpr int32_t kBaseTpAttention = 1 << 0;
constexpr int64_t kLayers = 2;
constexpr int64_t kHidden = 32;
constexpr int64_t kIntermediate = 48;
constexpr int64_t kKeyHeads = 4;
constexpr int64_t kValueHeads = 8;
constexpr int64_t kKeyDim = 128;
constexpr int64_t kValueDim = 128;
constexpr int64_t kConvKernel = 4;
constexpr int64_t kVocab = 64;
constexpr int64_t kLoraRank = 4;
constexpr int64_t kPairsPerLayer = 8;
constexpr int64_t kQSize = kKeyHeads * kKeyDim;
constexpr int64_t kVSize = kValueHeads * kValueDim;
constexpr int64_t kQkvSize = 2 * kQSize + kVSize;

constexpr std::array<const char*, 5> kGdnModules = {
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"};

static int required_env_int(const char* name) {
    const char* value = std::getenv(name);
    assert(value && value[0] != '\0');
    return std::atoi(value);
}

static at::Tensor fingerprint(
    std::initializer_list<int64_t> shape, double scale, int64_t offset
) {
    int64_t count = 1;
    for (const int64_t dim : shape) count *= dim;
    auto values = at::arange(
        count, at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    return ((values.add(offset).remainder(97) - 48.0) * scale)
        .reshape(shape).to(at::kBFloat16);
}

static at::Tensor unit_weight(std::initializer_list<int64_t> shape) {
    return at::ones(
        shape, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
}

static void append_full_gdn_layer(
    std::vector<at::Tensor>& weights, int64_t layer
) {
    const int64_t base = 1000 * layer;
    weights.push_back(unit_weight({kHidden}));
    weights.push_back(unit_weight({kHidden}));

    auto q = fingerprint({kQSize, kHidden}, 0.00035, base + 11);
    auto k = fingerprint({kQSize, kHidden}, 0.00041, base + 211);
    auto v = fingerprint({kVSize, kHidden}, 0.00029, base + 421);
    weights.push_back(at::cat({q, k, v}, 0).contiguous());
    weights.push_back(fingerprint({kVSize, kHidden}, 0.00031, base + 631));
    weights.push_back(fingerprint({kValueHeads, kHidden}, 0.00043, base + 719));
    weights.push_back(fingerprint({kValueHeads, kHidden}, 0.00047, base + 811));
    weights.push_back(fingerprint({kValueHeads}, 0.0008, base + 907));
    weights.push_back(fingerprint({kValueHeads}, 0.0007, base + 953));

    auto q_conv = fingerprint(
        {kQSize, 1, kConvKernel}, 0.0009, base + 101);
    auto k_conv = fingerprint(
        {kQSize, 1, kConvKernel}, 0.0011, base + 307);
    auto v_conv = fingerprint(
        {kVSize, 1, kConvKernel}, 0.0007, base + 509);
    weights.push_back(at::cat({q_conv, k_conv, v_conv}, 0).contiguous());
    weights.push_back(unit_weight({kValueDim}));
    weights.push_back(fingerprint({kHidden, kVSize}, 0.00033, base + 613));

    weights.push_back(fingerprint(
        {kIntermediate, kHidden}, 0.00045, base + 701));
    weights.push_back(fingerprint(
        {kIntermediate, kHidden}, 0.00039, base + 797));
    weights.push_back(fingerprint(
        {kHidden, kIntermediate}, 0.00037, base + 887));
}

static at::Tensor shard_flat_qkv(const at::Tensor& full, int rank, int dim) {
    const int64_t local_q = kQSize / 2;
    const int64_t local_v = kVSize / 2;
    auto q = full.narrow(dim, rank * local_q, local_q);
    auto k = full.narrow(dim, kQSize + rank * local_q, local_q);
    auto v = full.narrow(dim, 2 * kQSize + rank * local_v, local_v);
    return at::cat({q, k, v}, dim).contiguous();
}

static std::vector<at::Tensor> make_local_weights(
    const std::vector<at::Tensor>& full, int rank
) {
    std::vector<at::Tensor> local;
    local.reserve(full.size());
    for (int64_t layer = 0; layer < kLayers; ++layer) {
        const int64_t offset = layer * 14;
        local.push_back(full[offset + 0]);
        local.push_back(full[offset + 1]);
        local.push_back(shard_flat_qkv(full[offset + 2], rank, 0));
        local.push_back(full[offset + 3]
            .narrow(0, rank * (kVSize / 2), kVSize / 2).contiguous());
        local.push_back(full[offset + 4]
            .narrow(0, rank * (kValueHeads / 2), kValueHeads / 2).contiguous());
        local.push_back(full[offset + 5]
            .narrow(0, rank * (kValueHeads / 2), kValueHeads / 2).contiguous());
        local.push_back(full[offset + 6]
            .narrow(0, rank * (kValueHeads / 2), kValueHeads / 2).contiguous());
        local.push_back(full[offset + 7]
            .narrow(0, rank * (kValueHeads / 2), kValueHeads / 2).contiguous());
        local.push_back(shard_flat_qkv(full[offset + 8], rank, 0));
        local.push_back(full[offset + 9]);
        local.push_back(full[offset + 10]
            .narrow(1, rank * (kVSize / 2), kVSize / 2).contiguous());
        local.insert(local.end(), full.begin() + offset + 11,
            full.begin() + offset + 14);
    }
    return local;
}

static std::vector<void*> pointers(std::vector<at::Tensor>& tensors) {
    std::vector<void*> result;
    result.reserve(tensors.size());
    for (auto& tensor : tensors) result.push_back(&tensor);
    return result;
}

static LayerConfig gdn_config() {
    LayerConfig config{};
    config.layer_type = 1;
    config.num_k_heads = kKeyHeads;
    config.key_dim = kKeyDim;
    config.num_v_heads = kValueHeads;
    config.val_dim = kValueDim;
    config.conv_kernel = kConvKernel;
    config.rms_eps = 1e-5;
    config.intermediate_size = kIntermediate;
    return config;
}

struct Batch {
    at::Tensor input_ids;
    at::Tensor target_mask;
    at::Tensor attention_mask;
};

static Batch make_batch(int64_t batch, int64_t seq, int64_t offset) {
    std::vector<int64_t> host(batch * seq);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            host[b * seq + s] = 1 + (offset + b * 17 + s * 5) % (kVocab - 1);
        }
    }
    auto ids = at::from_blob(host.data(), {batch, seq},
        at::TensorOptions().device(at::kCPU).dtype(at::kLong)).clone().to(at::kCUDA);
    auto target = at::ones({batch, seq},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    auto attention = at::ones({batch, seq},
        at::TensorOptions().device(at::kCUDA).dtype(at::kBool));
    return {std::move(ids), std::move(target), std::move(attention)};
}

static double max_diff(const at::Tensor& lhs, const at::Tensor& rhs) {
    assert(lhs.sizes() == rhs.sizes());
    return (lhs.to(at::kFloat) - rhs.to(at::kFloat))
        .abs().max().item<double>();
}

static double relative_l2(const at::Tensor& lhs, const at::Tensor& rhs) {
    assert(lhs.sizes() == rhs.sizes());
    auto delta = lhs.to(at::kFloat) - rhs.to(at::kFloat);
    const double denom = std::max(rhs.to(at::kFloat).norm().item<double>(), 1e-12);
    return delta.norm().item<double>() / denom;
}

static bool is_column_parallel(const char* module) {
    return std::strcmp(module, "out_proj") != 0;
}

static int64_t full_output_size(const char* module) {
    if (std::strcmp(module, "in_proj_qkv") == 0) return kQkvSize;
    if (std::strcmp(module, "in_proj_z") == 0) return kVSize;
    if (std::strcmp(module, "in_proj_a") == 0 ||
        std::strcmp(module, "in_proj_b") == 0) return kValueHeads;
    assert(std::strcmp(module, "out_proj") == 0);
    return kHidden;
}

static int64_t full_input_size(const char* module) {
    return std::strcmp(module, "out_proj") == 0 ? kVSize : kHidden;
}

static at::Tensor shard_projection_output(
    const at::Tensor& full, const char* module, int rank, int dim
) {
    if (std::strcmp(module, "in_proj_qkv") == 0)
        return shard_flat_qkv(full, rank, dim);
    const int64_t local = full.size(dim) / 2;
    return full.narrow(dim, rank * local, local).contiguous();
}

static at::Tensor expected_local_factor(
    const at::Tensor& full, const char* module, bool is_b, int rank
) {
    if (is_column_parallel(module)) {
        return is_b ? shard_projection_output(full, module, rank, 0)
                    : full;
    }
    return is_b ? full
                : full.narrow(1, rank * (kVSize / 2), kVSize / 2).contiguous();
}

struct LoraFixture {
    int64_t layer;
    int64_t pair;
    const char* module;
    at::Tensor full_a;
    at::Tensor full_b;
    at::Tensor local_a;
    at::Tensor local_b;
};

static std::vector<LoraFixture> make_lora_fixtures(int rank, int64_t seed_base) {
    std::vector<LoraFixture> fixtures;
    fixtures.reserve(kLayers * kGdnModules.size());
    for (int64_t layer = 0; layer < kLayers; ++layer) {
        for (int64_t pair = 0; pair < static_cast<int64_t>(kGdnModules.size()); ++pair) {
            const char* module = kGdnModules[pair];
            const int64_t base = seed_base + layer * 100 + pair * 13;
            auto full_a = fingerprint(
                {kLoraRank, full_input_size(module)}, 0.0007, base + 1);
            auto full_b = fingerprint(
                {full_output_size(module), kLoraRank}, 0.0006, base + 7);
            auto local_a = expected_local_factor(full_a, module, false, rank);
            auto local_b = expected_local_factor(full_b, module, true, rank);
            fixtures.push_back({layer, pair, module, std::move(full_a),
                std::move(full_b), std::move(local_a), std::move(local_b)});
        }
    }
    return fixtures;
}

static void assert_weight_contract(
    const std::vector<at::Tensor>& full,
    const std::vector<at::Tensor>& local, int rank
) {
    assert(full.size() == 28 && local.size() == full.size());
    for (int64_t layer = 0; layer < kLayers; ++layer) {
        const int64_t offset = layer * 14;
        assert(local[offset + 2].sizes() ==
            at::IntArrayRef({kQkvSize / 2, kHidden}));
        assert(local[offset + 8].sizes() ==
            at::IntArrayRef({kQkvSize / 2, 1, kConvKernel}));
        assert(local[offset + 3].sizes() ==
            at::IntArrayRef({kVSize / 2, kHidden}));
        assert(local[offset + 4].sizes() ==
            at::IntArrayRef({kValueHeads / 2, kHidden}));
        assert(local[offset + 9].sizes() == at::IntArrayRef({kValueDim}));
        assert(local[offset + 10].sizes() ==
            at::IntArrayRef({kHidden, kVSize / 2}));
        assert(max_diff(local[offset + 2],
            shard_flat_qkv(full[offset + 2], rank, 0)) == 0.0);
        assert(max_diff(local[offset + 8],
            shard_flat_qkv(full[offset + 8], rank, 0)) == 0.0);
    }
}

struct ContextPair {
    void* distributed = nullptr;
    void* reference = nullptr;
};

static ContextPair create_context_pair(
    std::vector<at::Tensor>& local_weights,
    std::vector<at::Tensor>& full_weights,
    at::Tensor& embed, at::Tensor& final_norm, at::Tensor& lm_head,
    LayerConfig* configs, const char* fixed_targets,
    bool run_negative_guard
) {
    const int64_t target_layers[kLayers] = {0, 1};
    auto local_ptrs = pointers(local_weights);
    setenv("TP_SIZE", "2", 1);
    unsetenv("RUSTRAIN_DATA_PARALLEL");

    if (run_negative_guard) {
        auto invalid_weights = local_weights;
        invalid_weights[2] = invalid_weights[2]
            .narrow(0, 0, invalid_weights[2].size(0) - 1).contiguous();
        auto invalid_ptrs = pointers(invalid_weights);
        void* invalid = qwen36_create_training_context_ex(
            invalid_ptrs.data(), invalid_ptrs.size(), &embed, &final_norm, &lm_head,
            configs, kLayers, static_cast<int32_t>(at::kBFloat16),
            1.0, 1e-3, 0.9, 0.999, 1e-8, kVocab, 1e-5, kLoraRank,
            target_layers, kLayers, fixed_targets, kBaseTpAttention);
        assert(!invalid && "invalid local flat-QKV shape must be rejected");
    }

    void* distributed = qwen36_create_training_context_ex(
        local_ptrs.data(), local_ptrs.size(), &embed, &final_norm, &lm_head,
        configs, kLayers, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, kVocab, 1e-5, kLoraRank,
        target_layers, kLayers, fixed_targets, kBaseTpAttention);
    assert(distributed && qwen36_init_nccl(distributed) == 0);

    setenv("TP_SIZE", "1", 1);
    auto full_ptrs = pointers(full_weights);
    void* reference = qwen36_create_training_context(
        full_ptrs.data(), full_ptrs.size(), &embed, &final_norm, &lm_head,
        configs, kLayers, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, kVocab, 1e-5, kLoraRank,
        target_layers, kLayers, fixed_targets);
    assert(reference);
    return {distributed, reference};
}

static void set_fixed_lora(
    ContextPair contexts, std::vector<LoraFixture>& fixtures
) {
    for (auto& fixture : fixtures) {
        const int64_t slot = fixture.layer * kPairsPerLayer + fixture.pair;
        assert(qwen36_set_lora_tensor(
            contexts.distributed, slot, 0, &fixture.local_a) == 0);
        assert(qwen36_set_lora_tensor(
            contexts.distributed, slot, 1, &fixture.local_b) == 0);
        assert(qwen36_set_lora_tensor(
            contexts.reference, slot, 0, &fixture.full_a) == 0);
        assert(qwen36_set_lora_tensor(
            contexts.reference, slot, 1, &fixture.full_b) == 0);
    }
}

struct ErrorSummary {
    double a = 0.0;
    double b = 0.0;
    double relative = 0.0;
    double m = 0.0;
    double v = 0.0;
    double param = 0.0;
    double adam = 0.0;
};

static void check_fixed_path(
    ContextPair contexts, std::vector<LoraFixture>& fixtures,
    Batch& short_batch, Batch& batch, int rank
) {
    const double short_distributed = qwen36_eval_step(contexts.distributed,
        &short_batch.input_ids, &short_batch.target_mask,
        &short_batch.attention_mask);
    const double short_reference = qwen36_eval_step(contexts.reference,
        &short_batch.input_ids, &short_batch.target_mask,
        &short_batch.attention_mask);
    const double eval_distributed = qwen36_eval_step(contexts.distributed,
        &batch.input_ids, &batch.target_mask, &batch.attention_mask);
    const double eval_reference = qwen36_eval_step(contexts.reference,
        &batch.input_ids, &batch.target_mask, &batch.attention_mask);
    assert(std::isfinite(short_distributed) && std::isfinite(short_reference));
    assert(std::isfinite(eval_distributed) && std::isfinite(eval_reference));

    const double micro_distributed = qwen36_train_micro_step(
        contexts.distributed, &batch.input_ids, &batch.target_mask,
        &batch.attention_mask, 1.0, 0);
    const double micro_reference = qwen36_train_micro_step(
        contexts.reference, &batch.input_ids, &batch.target_mask,
        &batch.attention_mask, 1.0, 0);
    assert(std::isfinite(micro_distributed) && std::isfinite(micro_reference));

    ErrorSummary errors;
    for (auto& fixture : fixtures) {
        const int64_t slot = fixture.layer * kPairsPerLayer + fixture.pair;
        auto* local_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_grad_accumulator(contexts.distributed, slot, 0));
        auto* local_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_grad_accumulator(contexts.distributed, slot, 1));
        auto* full_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_grad_accumulator(contexts.reference, slot, 0));
        auto* full_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_grad_accumulator(contexts.reference, slot, 1));
        assert(local_a && local_b && full_a && full_b);
        assert(local_a->scalar_type() == at::kFloat &&
            local_b->scalar_type() == at::kFloat);
        auto expected_a = expected_local_factor(*full_a, fixture.module, false, rank);
        auto expected_b = expected_local_factor(*full_b, fixture.module, true, rank);
        const double a_diff = max_diff(*local_a, expected_a);
        const double b_diff = max_diff(*local_b, expected_b);
        const double a_relative = relative_l2(*local_a, expected_a);
        const double b_relative = relative_l2(*local_b, expected_b);
        std::printf(
            "native_tp_gdn_grad rank=%d layer=%ld module=%s "
            "a_diff=%0.8e b_diff=%0.8e a_relative_l2=%0.8e "
            "b_relative_l2=%0.8e a_reference_norm=%0.8e "
            "b_reference_norm=%0.8e\n",
            rank, fixture.layer, fixture.module, a_diff, b_diff,
            a_relative, b_relative,
            expected_a.to(at::kFloat).norm().item<double>(),
            expected_b.to(at::kFloat).norm().item<double>());
        // Before the final optimizer boundary, replicated factors have only
        // their rank-local contribution. Compare shard-local factors here;
        // final Adam state checks below cover the synchronized replicas.
        if (is_column_parallel(fixture.module)) {
            errors.b = std::max(errors.b, b_diff);
            errors.relative = std::max(errors.relative, b_relative);
        } else {
            errors.a = std::max(errors.a, a_diff);
            errors.relative = std::max(errors.relative, a_relative);
        }
    }
    assert(qwen36_abort_gradient_accumulation(contexts.distributed) == 0);
    assert(qwen36_abort_gradient_accumulation(contexts.reference) == 0);

    const double loss_distributed = qwen36_train_step(contexts.distributed,
        &batch.input_ids, &batch.target_mask, &batch.attention_mask);
    const double loss_reference = qwen36_train_step(contexts.reference,
        &batch.input_ids, &batch.target_mask, &batch.attention_mask);
    assert(std::isfinite(loss_distributed) && std::isfinite(loss_reference));

    constexpr int64_t optimizer_count = 2 * kLayers * kPairsPerLayer;
    std::vector<void*> local_m(optimizer_count), local_v(optimizer_count);
    std::vector<void*> full_m(optimizer_count), full_v(optimizer_count);
    assert(qwen36_export_optimizer_state(contexts.distributed,
        local_m.data(), local_v.data(), optimizer_count) == optimizer_count);
    assert(qwen36_export_optimizer_state(contexts.reference,
        full_m.data(), full_v.data(), optimizer_count) == optimizer_count);

    for (auto& fixture : fixtures) {
        const int64_t slot = fixture.layer * kPairsPerLayer + fixture.pair;
        auto* updated_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(contexts.distributed, slot));
        auto* updated_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(contexts.distributed, slot));
        auto* reference_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(contexts.reference, slot));
        auto* reference_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(contexts.reference, slot));
        assert(updated_a && updated_b && reference_a && reference_b);

        auto* m_a = reinterpret_cast<at::Tensor*>(local_m[2 * slot]);
        auto* m_b = reinterpret_cast<at::Tensor*>(local_m[2 * slot + 1]);
        auto* v_a = reinterpret_cast<at::Tensor*>(local_v[2 * slot]);
        auto* v_b = reinterpret_cast<at::Tensor*>(local_v[2 * slot + 1]);
        auto* ref_m_a = reinterpret_cast<at::Tensor*>(full_m[2 * slot]);
        auto* ref_m_b = reinterpret_cast<at::Tensor*>(full_m[2 * slot + 1]);
        auto* ref_v_a = reinterpret_cast<at::Tensor*>(full_v[2 * slot]);
        auto* ref_v_b = reinterpret_cast<at::Tensor*>(full_v[2 * slot + 1]);
        assert(m_a && m_b && v_a && v_b && ref_m_a && ref_m_b &&
            ref_v_a && ref_v_b);

        errors.m = std::max({errors.m,
            max_diff(*m_a, expected_local_factor(
                *ref_m_a, fixture.module, false, rank)),
            max_diff(*m_b, expected_local_factor(
                *ref_m_b, fixture.module, true, rank))});
        errors.v = std::max({errors.v,
            max_diff(*v_a, expected_local_factor(
                *ref_v_a, fixture.module, false, rank)),
            max_diff(*v_b, expected_local_factor(
                *ref_v_b, fixture.module, true, rank))});
        errors.param = std::max({errors.param,
            max_diff(*updated_a, expected_local_factor(
                *reference_a, fixture.module, false, rank)),
            max_diff(*updated_b, expected_local_factor(
                *reference_b, fixture.module, true, rank))});

        auto expected_a = (fixture.local_a.to(at::kFloat) - 1e-3 *
            (*m_a / (1.0 - 0.9)) /
            (((*v_a / (1.0 - 0.999)).sqrt()) + 1e-8)).to(at::kBFloat16);
        auto expected_b = (fixture.local_b.to(at::kFloat) - 1e-3 *
            (*m_b / (1.0 - 0.9)) /
            (((*v_b / (1.0 - 0.999)).sqrt()) + 1e-8)).to(at::kBFloat16);
        errors.adam = std::max({errors.adam,
            max_diff(*updated_a, expected_a), max_diff(*updated_b, expected_b)});
    }

    const double short_diff = std::abs(short_distributed - short_reference);
    const double eval_diff = std::abs(eval_distributed - eval_reference);
    const double micro_diff = std::abs(micro_distributed - micro_reference);
    const double loss_diff = std::abs(loss_distributed - loss_reference);
    std::printf(
        "native_tp_gdn_fixed rank=%d short_eval_diff=%0.8e eval_diff=%0.8e "
        "micro_diff=%0.8e loss_diff=%0.8e a_grad_diff=%0.8e "
        "b_grad_diff=%0.8e grad_relative_l2=%0.8e m_diff=%0.8e "
        "v_diff=%0.8e param_diff=%0.8e adam_error=%0.8e\n",
        rank, short_diff, eval_diff, micro_diff, loss_diff, errors.a, errors.b,
        errors.relative, errors.m, errors.v, errors.param, errors.adam);
    std::fflush(stdout);

    assert(short_diff < 5e-3 && eval_diff < 5e-3);
    assert(micro_diff < 5e-3 && loss_diff < 5e-3);
    assert(errors.a < 5e-4 && errors.b < 5e-4);
    assert(errors.relative < 2e-2);
    assert(errors.m < 5e-5 && errors.v < 5e-8);
    assert(errors.param <= 2e-3);
    assert(errors.adam < 1e-8);
}

static void set_dynamic_lora(
    void* context, int64_t adapter_id,
    std::vector<LoraFixture>& fixtures, bool local
) {
    for (auto& fixture : fixtures) {
        auto& a = local ? fixture.local_a : fixture.full_a;
        auto& b = local ? fixture.local_b : fixture.full_b;
        assert(qwen36_set_adapter_lora_tensor(context, adapter_id,
            fixture.layer, fixture.module, 0, &a) == 0);
        assert(qwen36_set_adapter_lora_tensor(context, adapter_id,
            fixture.layer, fixture.module, 1, &b) == 0);
    }
}

static void check_dynamic_path(
    ContextPair contexts, std::vector<LoraFixture>& fixtures,
    Batch& batch, int rank
) {
    const int64_t target_layers[kLayers] = {0, 1};
    constexpr const char* targets =
        "in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj";
    const int64_t distributed_id = qwen36_add_lora(contexts.distributed,
        kLoraRank, kLoraRank, target_layers, kLayers, targets);
    const int64_t reference_id = qwen36_add_lora(contexts.reference,
        kLoraRank, kLoraRank, target_layers, kLayers, targets);
    assert(distributed_id > 0 && reference_id > 0);
    set_dynamic_lora(contexts.distributed, distributed_id, fixtures, true);
    set_dynamic_lora(contexts.reference, reference_id, fixtures, false);

    const double distributed_loss = qwen36_train_multi_lora_selected(
        contexts.distributed, &batch.input_ids, &batch.target_mask,
        &batch.attention_mask, &distributed_id, 1, kLoraRank);
    const double reference_loss = qwen36_train_multi_lora_selected(
        contexts.reference, &batch.input_ids, &batch.target_mask,
        &batch.attention_mask, &reference_id, 1, kLoraRank);
    assert(std::isfinite(distributed_loss) && std::isfinite(reference_loss));

    ErrorSummary errors;
    for (auto& fixture : fixtures) {
        auto* local_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_adapter_lora_tensor(contexts.distributed,
                distributed_id, fixture.layer, fixture.module, 0));
        auto* local_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_adapter_lora_tensor(contexts.distributed,
                distributed_id, fixture.layer, fixture.module, 1));
        auto* full_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_adapter_lora_tensor(contexts.reference,
                reference_id, fixture.layer, fixture.module, 0));
        auto* full_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_adapter_lora_tensor(contexts.reference,
                reference_id, fixture.layer, fixture.module, 1));
        assert(local_a && local_b && full_a && full_b);
        errors.param = std::max({errors.param,
            max_diff(*local_a, expected_local_factor(
                *full_a, fixture.module, false, rank)),
            max_diff(*local_b, expected_local_factor(
                *full_b, fixture.module, true, rank))});

        auto state = [&](void* context, int64_t adapter, bool is_b, bool is_v) {
            auto* tensor = reinterpret_cast<at::Tensor*>(
                qwen36_get_adapter_optimizer_tensor(context, adapter,
                    fixture.layer, fixture.module, is_b, is_v));
            assert(tensor && tensor->scalar_type() == at::kFloat);
            return tensor;
        };
        auto* m_a = state(contexts.distributed, distributed_id, false, false);
        auto* m_b = state(contexts.distributed, distributed_id, true, false);
        auto* v_a = state(contexts.distributed, distributed_id, false, true);
        auto* v_b = state(contexts.distributed, distributed_id, true, true);
        auto* full_m_a = state(contexts.reference, reference_id, false, false);
        auto* full_m_b = state(contexts.reference, reference_id, true, false);
        auto* full_v_a = state(contexts.reference, reference_id, false, true);
        auto* full_v_b = state(contexts.reference, reference_id, true, true);
        errors.m = std::max({errors.m,
            max_diff(*m_a, expected_local_factor(
                *full_m_a, fixture.module, false, rank)),
            max_diff(*m_b, expected_local_factor(
                *full_m_b, fixture.module, true, rank))});
        errors.v = std::max({errors.v,
            max_diff(*v_a, expected_local_factor(
                *full_v_a, fixture.module, false, rank)),
            max_diff(*v_b, expected_local_factor(
                *full_v_b, fixture.module, true, rank))});

        auto expected_a = (fixture.local_a.to(at::kFloat) - 1e-3 *
            (*m_a / (1.0 - 0.9)) /
            (((*v_a / (1.0 - 0.999)).sqrt()) + 1e-8)).to(at::kBFloat16);
        auto expected_b = (fixture.local_b.to(at::kFloat) - 1e-3 *
            (*m_b / (1.0 - 0.9)) /
            (((*v_b / (1.0 - 0.999)).sqrt()) + 1e-8)).to(at::kBFloat16);
        errors.adam = std::max({errors.adam,
            max_diff(*local_a, expected_a), max_diff(*local_b, expected_b)});
    }

    const double loss_diff = std::abs(distributed_loss - reference_loss);
    std::printf(
        "native_tp_gdn_dynamic rank=%d loss_diff=%0.8e m_diff=%0.8e "
        "v_diff=%0.8e param_diff=%0.8e adam_error=%0.8e\n",
        rank, loss_diff, errors.m, errors.v, errors.param, errors.adam);
    std::fflush(stdout);
    assert(loss_diff < 5e-3);
    assert(errors.m < 5e-5 && errors.v < 5e-8);
    assert(errors.param <= 2e-3);
    assert(errors.adam < 1e-8);
}

}  // namespace

int main() {
    const int rank = required_env_int("RANK");
    const int world = required_env_int("WORLD_SIZE");
    const int local_rank = std::getenv("LOCAL_RANK")
        ? std::atoi(std::getenv("LOCAL_RANK")) : rank;
    assert(world == 2 && rank >= 0 && rank < world);
    assert(qwen36_kernel_abi_version() == kAbiVersion);
    qwen36_set_cuda_device(local_rank);

    std::vector<at::Tensor> full_weights;
    full_weights.reserve(kLayers * 14);
    for (int64_t layer = 0; layer < kLayers; ++layer)
        append_full_gdn_layer(full_weights, layer);
    for (auto& weight : full_weights) weight.set_requires_grad(false);
    auto local_weights = make_local_weights(full_weights, rank);
    for (auto& weight : local_weights) weight.set_requires_grad(false);
    assert_weight_contract(full_weights, local_weights, rank);

    auto embed = fingerprint({kVocab, kHidden}, 0.0013, 17);
    auto final_norm = unit_weight({kHidden});
    auto lm_head = fingerprint({kVocab, kHidden}, 0.0011, 59);
    embed.set_requires_grad(false);
    final_norm.set_requires_grad(false);
    lm_head.set_requires_grad(false);
    LayerConfig configs[kLayers] = {gdn_config(), gdn_config()};
    auto short_batch = make_batch(2, 3, 3);
    auto batch = make_batch(2, 9, 11);
    auto dynamic_batch = make_batch(1, 9, 23);

    constexpr const char* all_targets =
        "in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj";
    auto fixed_contexts = create_context_pair(local_weights, full_weights,
        embed, final_norm, lm_head, configs, all_targets, true);
    auto fixed_fixtures = make_lora_fixtures(rank, 2000);
    set_fixed_lora(fixed_contexts, fixed_fixtures);
    check_fixed_path(
        fixed_contexts, fixed_fixtures, short_batch, batch, rank);
    qwen36_free_training_context(fixed_contexts.reference);
    qwen36_free_training_context(fixed_contexts.distributed);

    auto dynamic_contexts = create_context_pair(local_weights, full_weights,
        embed, final_norm, lm_head, configs, "in_proj_qkv", false);
    auto dynamic_fixtures = make_lora_fixtures(rank, 4000);
    check_dynamic_path(
        dynamic_contexts, dynamic_fixtures, dynamic_batch, rank);
    qwen36_free_training_context(dynamic_contexts.reference);
    qwen36_free_training_context(dynamic_contexts.distributed);
    return 0;
}
