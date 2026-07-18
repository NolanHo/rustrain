#include <ATen/ATen.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
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
extern "C" int32_t qwen36_init_parallel_nccl(
    void*, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t);
extern "C" int32_t qwen36_attach_parallel_nccl_no_sync(
    void*, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t);
extern "C" int32_t qwen36_set_mtp_weights(
    void*, void*, void*, void*, void*, void**, int64_t, void*, int64_t);
extern "C" int64_t qwen36_get_lora_count(void*);
extern "C" void* qwen36_get_lora_a(void*, int64_t);
extern "C" void* qwen36_get_lora_b(void*, int64_t);
extern "C" int32_t qwen36_set_lora_tensor(void*, int64_t, int32_t, void*);
extern "C" int64_t qwen36_export_optimizer_state(
    void*, void**, void**, int64_t);
extern "C" int64_t qwen36_import_optimizer_state(
    void*, void**, void**, int64_t);
extern "C" int64_t qwen36_get_step_count(void*);
extern "C" int32_t qwen36_set_step_count(void*, int64_t);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" double qwen36_train_micro_step(
    void*, void*, void*, void*, double, int32_t);
extern "C" int32_t qwen36_abort_gradient_accumulation(void*);
extern "C" int64_t qwen36_add_lora(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" int64_t qwen36_add_lora_v2(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" int64_t qwen36_add_lora_for_restore(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" void* qwen36_get_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t);
extern "C" int32_t qwen36_set_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t, void*);
extern "C" void* qwen36_get_adapter_optimizer_tensor(
    void*, int64_t, int64_t, const char*, int32_t, int32_t);
extern "C" int32_t qwen36_set_adapter_optimizer_tensor(
    void*, int64_t, int64_t, const char*, int32_t, int32_t, void*);
extern "C" int64_t qwen36_get_adapter_step_count(void*, int64_t);
extern "C" int64_t qwen36_get_dynamic_finalizer_count(void*);
extern "C" int64_t qwen36_get_dynamic_adam_launch_count(void*);
extern "C" int32_t qwen36_set_adapter_step_count(
    void*, int64_t, int64_t);
extern "C" double qwen36_train_multi_lora_selected(
    void*, void*, void*, void*, const int64_t*, int32_t, int32_t);
extern "C" double qwen36_train_multi_lora_selected_v2(
    void*, void*, void*, void*, const int64_t*, int32_t);
extern "C" void qwen36_free_training_context(void*);

namespace {

constexpr int64_t kAbiVersion = 24;
constexpr int32_t kBaseTpAttention = 1 << 0;
constexpr int32_t kDataParallel = 1 << 1;
constexpr int32_t kVocabParallel = 1 << 2;
constexpr int32_t kExpertParallel = 1 << 3;
constexpr int32_t kBaseTpMlp = 1 << 4;
constexpr int64_t kHidden = 16;
constexpr int64_t kVocab = 16;
constexpr int64_t kHeads = 2;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kHeadDim = 8;
constexpr int64_t kExperts = 2;
constexpr int64_t kIntermediate = 8;
constexpr int64_t kLoraRank = 4;
constexpr int64_t kLoraPairs = 9;
constexpr int64_t kOptimizerSlots = 2 * kLoraPairs;
constexpr double kLearningRate = 1e-3;
constexpr double kBeta1 = 0.9;
constexpr double kBeta2 = 0.999;
constexpr double kAdamEps = 1e-8;

int required_env_int(const char* name) {
    const char* value = std::getenv(name);
    assert(value && value[0] != '\0');
    return std::atoi(value);
}

at::Tensor fingerprint(
    std::initializer_list<int64_t> shape, double scale, int64_t offset
) {
    int64_t count = 1;
    for (int64_t dim : shape) count *= dim;
    auto values = at::arange(
        count, at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    return ((values.add(offset).remainder(37) - 18.0) * scale)
        .reshape(shape).to(at::kBFloat16);
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

double max_diff(const at::Tensor& lhs, const at::Tensor& rhs) {
    assert(lhs.sizes() == rhs.sizes());
    return (lhs.to(at::kFloat) - rhs.to(at::kFloat))
        .abs().max().item<double>();
}

double update_norm(const at::Tensor& after, const at::Tensor& before) {
    return (after.to(at::kFloat) - before.to(at::kFloat))
        .abs().sum().item<double>();
}

at::Tensor adam_expected(
    const at::Tensor& before, const at::Tensor& m, const at::Tensor& v
) {
    auto m_hat = m.to(at::kFloat) / (1.0 - kBeta1);
    auto v_hat = v.to(at::kFloat) / (1.0 - kBeta2);
    return (before.to(at::kFloat) -
        kLearningRate * m_hat / (v_hat.sqrt() + kAdamEps))
        .to(before.scalar_type());
}

struct Batch {
    at::Tensor input_ids;
    at::Tensor target_mask;
    at::Tensor attention_mask;
};

Batch source_batch(int source_rank) {
    auto long_opts = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
    auto float_opts = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    auto bool_opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBool);
    if (source_rank == 0) {
        return {
            at::tensor({1, 2, 3, 4}, long_opts).reshape({1, 4}),
            at::tensor({0.0, 1.0, 0.0, 0.0}, float_opts).reshape({1, 4}),
            at::ones({1, 4}, bool_opts),
        };
    }
    if (source_rank == 1) {
        return {
            at::tensor({5, 6, 7, 8}, long_opts).reshape({1, 4}),
            at::tensor({0.0, 1.0, 1.0, 1.0}, float_opts).reshape({1, 4}),
            at::ones({1, 4}, bool_opts),
        };
    }
    if (source_rank == 2) {
        return {
            at::tensor({9, 10, 11, 12}, long_opts).reshape({1, 4}),
            at::tensor({0.0, 1.0, 0.0, 0.0}, float_opts).reshape({1, 4}),
            at::ones({1, 4}, bool_opts),
        };
    }
    assert(source_rank == 3);
    return {
        at::tensor({13, 14, 15, 1}, long_opts).reshape({1, 4}),
        at::tensor({0.0, 1.0, 1.0, 0.0}, float_opts).reshape({1, 4}),
        at::ones({1, 4}, bool_opts),
    };
}

Batch full_batch(int source_count) {
    auto first = source_batch(0);
    auto second = source_batch(1);
    if (source_count == 2) {
        return {
            at::cat({first.input_ids, second.input_ids}, 0),
            at::cat({first.target_mask, second.target_mask}, 0),
            at::cat({first.attention_mask, second.attention_mask}, 0),
        };
    }
    assert(source_count == 4);
    auto third = source_batch(2);
    auto fourth = source_batch(3);
    return {
        at::cat({first.input_ids, second.input_ids,
            third.input_ids, fourth.input_ids}, 0),
        at::cat({first.target_mask, second.target_mask,
            third.target_mask, fourth.target_mask}, 0),
        at::cat({first.attention_mask, second.attention_mask,
            third.attention_mask, fourth.attention_mask}, 0),
    };
}

std::vector<at::Tensor> make_full_weights() {
    std::vector<at::Tensor> weights;
    weights.reserve(15);
    weights.push_back(unit({kHidden}));
    weights.push_back(unit({kHidden}));
    weights.push_back(fingerprint(
        {2 * kHeads * kHeadDim, kHidden}, 0.0011, 11));
    weights.push_back(unit({kHeadDim}));
    weights.push_back(fingerprint(
        {kKvHeads * kHeadDim, kHidden}, 0.0012, 101));
    weights.push_back(unit({kHeadDim}));
    weights.push_back(fingerprint(
        {kKvHeads * kHeadDim, kHidden}, 0.0013, 211));
    weights.push_back(fingerprint(
        {kHidden, kHeads * kHeadDim}, 0.0010, 307));
    weights.push_back(fingerprint({kExperts, kHidden}, 0.0020, 401));
    weights.push_back(fingerprint({1, kHidden}, 0.0015, 503));
    weights.push_back(fingerprint({kIntermediate, kHidden}, 0.0010, 601));
    weights.push_back(fingerprint({kIntermediate, kHidden}, 0.0011, 701));
    weights.push_back(fingerprint({kHidden, kIntermediate}, 0.0012, 809));
    weights.push_back(fingerprint(
        {kExperts, 2 * kIntermediate, kHidden}, 0.0010, 907));
    weights.push_back(fingerprint(
        {kExperts, kHidden, kIntermediate}, 0.0011, 1009));
    for (auto& weight : weights) weight.set_requires_grad(false);
    return weights;
}

std::vector<at::Tensor> make_local_weights(
    const std::vector<at::Tensor>& full, int tp_rank, int ep_rank,
    bool base_tp_mlp
) {
    const int64_t local_heads = kHeads / 2;
    const int64_t local_kv_heads = kKvHeads / 2;
    const int64_t local_intermediate = kIntermediate / 2;
    std::vector<at::Tensor> local;
    local.reserve(full.size());
    local.push_back(full[0]);
    local.push_back(full[1]);
    local.push_back(full[2].narrow(
        0, tp_rank * 2 * local_heads * kHeadDim,
        2 * local_heads * kHeadDim).contiguous());
    local.push_back(full[3]);
    local.push_back(full[4].narrow(
        0, tp_rank * local_kv_heads * kHeadDim,
        local_kv_heads * kHeadDim).contiguous());
    local.push_back(full[5]);
    local.push_back(full[6].narrow(
        0, tp_rank * local_kv_heads * kHeadDim,
        local_kv_heads * kHeadDim).contiguous());
    local.push_back(full[7].narrow(
        1, tp_rank * local_heads * kHeadDim,
        local_heads * kHeadDim).contiguous());
    local.push_back(full[8]);
    local.push_back(full[9]);
    local.push_back(base_tp_mlp ? full[10].narrow(
        0, tp_rank * local_intermediate, local_intermediate).contiguous()
        : full[10]);
    local.push_back(base_tp_mlp ? full[11].narrow(
        0, tp_rank * local_intermediate, local_intermediate).contiguous()
        : full[11]);
    local.push_back(base_tp_mlp ? full[12].narrow(
        1, tp_rank * local_intermediate, local_intermediate).contiguous()
        : full[12]);
    auto local_gate_up = full[13].narrow(0, ep_rank, 1);
    local.push_back(base_tp_mlp ? at::cat({
            local_gate_up.narrow(
                1, tp_rank * local_intermediate, local_intermediate),
            local_gate_up.narrow(
                1, kIntermediate + tp_rank * local_intermediate,
                local_intermediate),
        }, 1).contiguous()
        : local_gate_up.contiguous());
    auto local_down = full[14].narrow(0, ep_rank, 1);
    local.push_back(base_tp_mlp ? local_down
        .narrow(2, tp_rank * local_intermediate, local_intermediate)
        .contiguous() : local_down.contiguous());
    const int64_t expected_intermediate =
        base_tp_mlp ? local_intermediate : kIntermediate;
    assert(local[10].sizes() == at::IntArrayRef({expected_intermediate, kHidden}));
    assert(local[11].sizes() == at::IntArrayRef({expected_intermediate, kHidden}));
    assert(local[12].sizes() == at::IntArrayRef({kHidden, expected_intermediate}));
    assert(local[13].sizes() == at::IntArrayRef(
        {1, 2 * expected_intermediate, kHidden}));
    assert(local[14].sizes() == at::IntArrayRef(
        {1, kHidden, expected_intermediate}));
    return local;
}

LayerConfig make_config(int expert_start, int expert_count) {
    LayerConfig config{};
    config.layer_type = 0;
    config.num_heads = kHeads;
    config.num_kv_heads = kKvHeads;
    config.head_dim = kHeadDim;
    config.partial_rotary_factor = 1.0;
    config.rope_theta = 10000.0;
    config.rms_eps = 1e-5;
    config.num_experts = kExperts;
    // top_k=2 makes every source exercise both owner ranks and every expert
    // optimizer slot, independent of small-model router tie behavior.
    config.top_k = 2;
    config.moe_intermediate = kIntermediate;
    config.expert_start = expert_start;
    config.expert_count = expert_count;
    config.norm_topk_prob = 1;
    return config;
}

enum class FixtureKind {
    Q,
    SharedGate,
    SharedDown,
    ExpertGateUp,
    ExpertDown,
};

struct LoraFixture {
    int64_t slot;
    const char* module;
    FixtureKind kind;
    at::Tensor full_a;
    at::Tensor full_b;
    at::Tensor local_a;
    at::Tensor local_b;
};

at::Tensor local_factor(
    const at::Tensor& full, FixtureKind kind, bool is_b,
    int tp_rank, int ep_rank, bool base_tp_mlp
) {
    if (kind == FixtureKind::Q) {
        if (!is_b) return full.clone();
        const int64_t local_rows = full.size(0) / 2;
        return full.narrow(0, tp_rank * local_rows, local_rows).contiguous();
    }
    const int64_t local_intermediate = kIntermediate / 2;
    if (!base_tp_mlp) {
        if (kind == FixtureKind::SharedGate ||
            kind == FixtureKind::SharedDown) {
            const int64_t rank_dim = is_b ? 1 : 0;
            const int64_t local_rank = full.size(rank_dim) / 2;
            return full.narrow(
                rank_dim, tp_rank * local_rank, local_rank).contiguous();
        }
        auto local = full.narrow(0, ep_rank, 1);
        const int64_t rank_dim = is_b ? 2 : 1;
        const int64_t local_rank = local.size(rank_dim) / 2;
        return local.narrow(
            rank_dim, tp_rank * local_rank, local_rank).contiguous();
    }
    if (kind == FixtureKind::SharedGate) {
        if (!is_b) return full.clone();
        return full.narrow(
            0, tp_rank * local_intermediate, local_intermediate).contiguous();
    }
    if (kind == FixtureKind::SharedDown) {
        if (is_b) return full.clone();
        return full.narrow(
            1, tp_rank * local_intermediate, local_intermediate).contiguous();
    }
    auto local = full.narrow(0, ep_rank, 1);
    if (kind == FixtureKind::ExpertGateUp) {
        if (!is_b) return local.contiguous();
        return at::cat({
            local.narrow(
                1, tp_rank * local_intermediate, local_intermediate),
            local.narrow(
                1, kIntermediate + tp_rank * local_intermediate,
                local_intermediate),
        }, 1).contiguous();
    }
    if (!is_b) {
        return local.narrow(
            2, tp_rank * local_intermediate, local_intermediate).contiguous();
    }
    return local.contiguous();
}

std::vector<LoraFixture> make_lora_fixtures(
    int tp_rank, int ep_rank, int64_t offset, bool base_tp_mlp
) {
    std::vector<LoraFixture> result;
    auto add = [&](int64_t slot, const char* module, FixtureKind kind,
                   at::Tensor full_a, at::Tensor full_b) {
        // Some narrow slices are already contiguous, so contiguous() may keep
        // aliasing the full-reference tensor. DP perturbations must only touch
        // the local fixture.
        auto local_a = local_factor(
            full_a, kind, false, tp_rank, ep_rank, base_tp_mlp).clone();
        auto local_b = local_factor(
            full_b, kind, true, tp_rank, ep_rank, base_tp_mlp).clone();
        result.push_back({slot, module, kind, std::move(full_a),
            std::move(full_b), std::move(local_a), std::move(local_b)});
    };
    add(0, "q_proj", FixtureKind::Q,
        fingerprint({kLoraRank, kHidden}, 0.0007, offset + 1),
        fingerprint({2 * kHeads * kHeadDim, kLoraRank},
            0.0006, offset + 101));
    add(4, "shared_gate_proj", FixtureKind::SharedGate,
        fingerprint({kLoraRank, kHidden}, 0.0007, offset + 151),
        fingerprint({kIntermediate, kLoraRank}, 0.0006, offset + 181));
    add(6, "shared_down_proj", FixtureKind::SharedDown,
        fingerprint({kLoraRank, kIntermediate}, 0.0007, offset + 191),
        fingerprint({kHidden, kLoraRank}, 0.0006, offset + 201));
    add(7, "experts_gate_up_proj", FixtureKind::ExpertGateUp,
        fingerprint({kExperts, kLoraRank, kHidden}, 0.0007, offset + 211),
        fingerprint({kExperts, 2 * kIntermediate, kLoraRank},
            0.0006, offset + 307));
    add(8, "experts_down_proj", FixtureKind::ExpertDown,
        fingerprint({kExperts, kLoraRank, kIntermediate},
            0.0007, offset + 401),
        fingerprint({kExperts, kHidden, kLoraRank},
            0.0006, offset + 503));
    assert(result[0].local_a.sizes() == at::IntArrayRef({kLoraRank, kHidden}));
    assert(result[0].local_b.sizes() == at::IntArrayRef({kHidden, kLoraRank}));
    const int64_t local_rank = base_tp_mlp ? kLoraRank : kLoraRank / 2;
    const int64_t local_intermediate =
        base_tp_mlp ? kIntermediate / 2 : kIntermediate;
    assert(result[1].local_a.sizes() == at::IntArrayRef({local_rank, kHidden}));
    assert(result[1].local_b.sizes() == at::IntArrayRef({local_intermediate, local_rank}));
    assert(result[2].local_a.sizes() == at::IntArrayRef({local_rank, local_intermediate}));
    assert(result[2].local_b.sizes() == at::IntArrayRef({kHidden, local_rank}));
    assert(result[3].local_a.sizes() == at::IntArrayRef({1, local_rank, kHidden}));
    const int64_t local_gate_up_rows =
        base_tp_mlp ? kIntermediate : 2 * kIntermediate;
    assert(result[3].local_b.sizes() ==
        at::IntArrayRef({1, local_gate_up_rows, local_rank}));
    assert(result[4].local_a.sizes() == at::IntArrayRef({1, local_rank, local_intermediate}));
    assert(result[4].local_b.sizes() == at::IntArrayRef({1, kHidden, local_rank}));
    return result;
}

void install_fixed(
    void* context, std::vector<LoraFixture>& fixtures, bool local
) {
    assert(qwen36_get_lora_count(context) == kLoraPairs);
    for (auto& fixture : fixtures) {
        auto& a = local ? fixture.local_a : fixture.full_a;
        auto& b = local ? fixture.local_b : fixture.full_b;
        assert(qwen36_set_lora_tensor(context, fixture.slot, 0, &a) == 0);
        assert(qwen36_set_lora_tensor(context, fixture.slot, 1, &b) == 0);
    }
}

void install_dynamic(
    void* context, int64_t adapter_id,
    std::vector<LoraFixture>& fixtures, bool local
) {
    for (auto& fixture : fixtures) {
        auto& a = local ? fixture.local_a : fixture.full_a;
        auto& b = local ? fixture.local_b : fixture.full_b;
        assert(qwen36_set_adapter_lora_tensor(
            context, adapter_id, 0, fixture.module, 0, &a) == 0);
        assert(qwen36_set_adapter_lora_tensor(
            context, adapter_id, 0, fixture.module, 1, &b) == 0);
    }
}

at::Tensor* dynamic_tensor(
    void* context, int64_t adapter_id, const char* module, bool is_b
) {
    auto* tensor = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            context, adapter_id, 0, module, is_b ? 1 : 0));
    assert(tensor);
    return tensor;
}

at::Tensor* dynamic_state(
    void* context, int64_t adapter_id, const char* module,
    bool is_b, bool is_v
) {
    auto* tensor = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_optimizer_tensor(
            context, adapter_id, 0, module,
            is_b ? 1 : 0, is_v ? 1 : 0));
    assert(tensor && tensor->scalar_type() == at::kFloat);
    return tensor;
}

void set_distributed_env(
    int rank, int world, int tp_rank, int ep_rank, int dp_rank, int dp_size
) {
    setenv("WORLD_SIZE", std::to_string(world).c_str(), 1);
    setenv("TP_SIZE", "2", 1);
    setenv("EP_SIZE", "2", 1);
    setenv("DP_SIZE", std::to_string(dp_size).c_str(), 1);
    setenv("RUSTRAIN_TP_RANK", std::to_string(tp_rank).c_str(), 1);
    setenv("RUSTRAIN_EP_RANK", std::to_string(ep_rank).c_str(), 1);
    setenv("RUSTRAIN_DP_RANK", std::to_string(dp_rank).c_str(), 1);
    setenv("RUSTRAIN_DATA_PARALLEL", dp_size > 1 ? "1" : "0", 1);
    setenv("RANK", std::to_string(rank).c_str(), 1);
}

void set_reference_env() {
    setenv("WORLD_SIZE", "1", 1);
    setenv("TP_SIZE", "1", 1);
    setenv("EP_SIZE", "1", 1);
    setenv("DP_SIZE", "1", 1);
    setenv("RUSTRAIN_TP_RANK", "0", 1);
    setenv("RUSTRAIN_EP_RANK", "0", 1);
    setenv("RUSTRAIN_DP_RANK", "0", 1);
    setenv("RUSTRAIN_DATA_PARALLEL", "0", 1);
    setenv("RANK", "0", 1);
}

struct ParityErrors {
    double param = 0.0;
    double m = 0.0;
    double v = 0.0;
    double adam = 0.0;
};

}  // namespace

int main() {
    const int rank = required_env_int("RANK");
    const int world = required_env_int("WORLD_SIZE");
    const int local_rank = required_env_int("LOCAL_RANK");
    assert((world == 4 || world == 8) && rank >= 0 && rank < world);
    const int dp_size = world / 4;
    const int tp_rank = rank % 2;
    const int ep_rank = (rank / 2) % 2;
    const int dp_rank = rank / 4;
    const int tp_color = ep_rank + 2 * dp_rank;
    const int ep_color = tp_rank + 2 * dp_rank;
    const int dp_color = tp_rank + 2 * ep_rank;
    const char* sharded_a2a_env = std::getenv("QWEN36_EP_A2A_SHARDED");
    assert(sharded_a2a_env);
    const bool sharded_source = std::strcmp(sharded_a2a_env, "0") != 0;
    const bool base_tp_mlp = sharded_source;
    const int32_t distributed_flags =
        kBaseTpAttention | kVocabParallel | kExpertParallel |
        (base_tp_mlp ? kBaseTpMlp : 0) |
        (dp_size > 1 ? kDataParallel : 0);
    assert(qwen36_kernel_abi_version() == kAbiVersion);
    assert(std::getenv("QWEN36_EP_A2A") &&
        std::strcmp(std::getenv("QWEN36_EP_A2A"), "0") != 0);
    qwen36_set_cuda_device(local_rank);

    auto full_weights = make_full_weights();
    auto local_weights = make_local_weights(
        full_weights, tp_rank, ep_rank, base_tp_mlp);
    auto full_ptrs = pointers(full_weights);
    auto local_ptrs = pointers(local_weights);
    auto embed = fingerprint({kVocab, kHidden}, 0.0012, 1201);
    auto final_norm = unit({kHidden});
    auto lm_head = fingerprint({kVocab, kHidden}, 0.0010, 1301);
    auto local_embed = embed.narrow(
        0, tp_rank * (kVocab / 2), kVocab / 2).contiguous();
    auto local_lm_head = lm_head.narrow(
        0, tp_rank * (kVocab / 2), kVocab / 2).contiguous();
    embed.set_requires_grad(false);
    final_norm.set_requires_grad(false);
    lm_head.set_requires_grad(false);

    auto distributed_config = make_config(ep_rank, 1);
    auto reference_config = make_config(0, kExperts);
    const int64_t target_layer = 0;
    constexpr const char* targets =
        "q_proj,shared_gate_proj,shared_down_proj,"
        "experts_gate_up_proj,experts_down_proj";
    constexpr const char* projection_targets = "q_proj";

    // Reject a process grid that cannot cover WORLD_SIZE before any NCCL
    // communicator is created.
    set_distributed_env(rank, world, tp_rank, ep_rank, dp_rank, dp_size);
    setenv("EP_SIZE", "3", 1);
    void* invalid = qwen36_create_training_context_ex(
        local_ptrs.data(), local_ptrs.size(), &local_embed, &final_norm,
        &local_lm_head, &distributed_config, 1,
        static_cast<int32_t>(at::kBFloat16),
        1.0, kLearningRate, kBeta1, kBeta2, kAdamEps,
        kVocab, 1e-5, kLoraRank, &target_layer, 1, targets,
        distributed_flags);
    assert(invalid == nullptr);
    set_distributed_env(rank, world, tp_rank, ep_rank, dp_rank, dp_size);

    void* projection_rank_three = qwen36_create_training_context_ex(
        local_ptrs.data(), local_ptrs.size(), &local_embed, &final_norm,
        &local_lm_head, &distributed_config, 1,
        static_cast<int32_t>(at::kBFloat16),
        1.0, kLearningRate, kBeta1, kBeta2, kAdamEps,
        kVocab, 1e-5, 3, &target_layer, 1, projection_targets,
        distributed_flags);
    assert(projection_rank_three);
    assert(qwen36_add_lora(
        projection_rank_three, 3, 3.0, &target_layer, 1,
        projection_targets) > 0);
    qwen36_free_training_context(projection_rank_three);

    constexpr const char* mixed_targets = "q_proj,shared_gate_proj";
    const int32_t mixed_flags = distributed_flags & ~kBaseTpMlp;
    void* invalid_mixed_rank_three = qwen36_create_training_context_ex(
        local_ptrs.data(), local_ptrs.size(), &local_embed, &final_norm,
        &local_lm_head, &distributed_config, 1,
        static_cast<int32_t>(at::kBFloat16),
        1.0, kLearningRate, kBeta1, kBeta2, kAdamEps,
        kVocab, 1e-5, 3, &target_layer, 1, mixed_targets,
        mixed_flags);
    assert(invalid_mixed_rank_three == nullptr);
    void* mixed_rank_four = qwen36_create_training_context_ex(
        local_ptrs.data(), local_ptrs.size(), &local_embed, &final_norm,
        &local_lm_head, &distributed_config, 1,
        static_cast<int32_t>(at::kBFloat16),
        1.0, kLearningRate, kBeta1, kBeta2, kAdamEps,
        kVocab, 1e-5, kLoraRank, &target_layer, 1, mixed_targets,
        mixed_flags);
    assert(mixed_rank_four);
    assert(qwen36_add_lora(
        mixed_rank_four, 3, 3.0, &target_layer, 1, mixed_targets) == -1);
    qwen36_free_training_context(mixed_rank_four);

    void* distributed = qwen36_create_training_context_ex(
        local_ptrs.data(), local_ptrs.size(), &local_embed, &final_norm,
        &local_lm_head, &distributed_config, 1,
        static_cast<int32_t>(at::kBFloat16),
        1.0, kLearningRate, kBeta1, kBeta2, kAdamEps,
        kVocab, 1e-5, kLoraRank, &target_layer, 1, targets,
        distributed_flags);
    assert(distributed);

    set_reference_env();
    void* reference = qwen36_create_training_context(
        full_ptrs.data(), full_ptrs.size(), &embed, &final_norm, &lm_head,
        &reference_config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, kLearningRate, kBeta1, kBeta2, kAdamEps,
        kVocab, 1e-5, kLoraRank, &target_layer, 1, targets);
    assert(reference);
    set_distributed_env(rank, world, tp_rank, ep_rank, dp_rank, dp_size);

    // Frozen-base/vocabulary TP must reject unsharded MTP weights directly.
    auto mtp_dummy = unit({1});
    assert(qwen36_set_mtp_weights(
        distributed, &mtp_dummy, &mtp_dummy, &mtp_dummy, &mtp_dummy,
        nullptr, 0, nullptr, 0) == -1);

    auto fixed_fixtures = make_lora_fixtures(
        tp_rank, ep_rank, 2001, base_tp_mlp);
    if (dp_rank > 0) {
        for (auto& fixture : fixed_fixtures) {
            fixture.local_a.add_(0.25);
            fixture.local_b.add_(0.25);
        }
    }
    install_fixed(distributed, fixed_fixtures, true);
    install_fixed(reference, fixed_fixtures, false);

    // Default tp-ep-dp rank order makes TP the least-significant coordinate.
    assert(qwen36_init_parallel_nccl(
        distributed, rank, world,
        0, 1, 0,
        rank % 2, 2, rank / 2,
        rank / 2, 2, rank % 2) == -1);
    assert(qwen36_init_parallel_nccl(
        distributed, rank, world,
        tp_rank, 2, tp_color,
        ep_rank, 2, ep_color,
        dp_rank, dp_size, dp_color) == 0);

    std::vector<std::array<at::Tensor, 2>> fixed_before;
    double broadcast_diff = 0.0;
    for (auto& fixture : fixed_fixtures) {
        auto* a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(distributed, fixture.slot));
        auto* b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(distributed, fixture.slot));
        assert(a && b);
        auto expected_a = local_factor(
            fixture.full_a, fixture.kind, false, tp_rank, ep_rank,
            base_tp_mlp);
        auto expected_b = local_factor(
            fixture.full_b, fixture.kind, true, tp_rank, ep_rank,
            base_tp_mlp);
        const double a_diff = max_diff(*a, expected_a);
        const double b_diff = max_diff(*b, expected_b);
        if (a_diff != 0.0 || b_diff != 0.0) {
            std::fprintf(stderr,
                "fixed_broadcast_mismatch rank=%d tp=%d ep=%d dp=%d "
                "module=%s a_diff=%0.8e b_diff=%0.8e\n",
                rank, tp_rank, ep_rank, dp_rank, fixture.module,
                a_diff, b_diff);
        }
        broadcast_diff = std::max({broadcast_diff,
            a_diff, b_diff});
        fixed_before.push_back({a->clone(), b->clone()});
    }
    assert(broadcast_diff == 0.0);

    const int source_rank = sharded_source ? 2 * dp_rank + ep_rank : dp_rank;
    auto local_batch = source_batch(source_rank);
    const int source_count = sharded_source ? 2 * dp_size : dp_size;
    auto global_batch = full_batch(source_count);

    auto* fixed_preflight_probe = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_b(distributed, fixed_fixtures.front().slot));
    assert(fixed_preflight_probe);
    const auto fixed_preflight_probe_before = fixed_preflight_probe->clone();

    // Replica input signatures must agree before TP/EP forward collectives.
    auto mismatched_shape_input = local_batch.input_ids;
    auto mismatched_shape_targets = local_batch.target_mask;
    auto mismatched_shape_attention = local_batch.attention_mask;
    if (rank == 0) {
        mismatched_shape_input = mismatched_shape_input.narrow(1, 0, 3).contiguous();
        mismatched_shape_targets = mismatched_shape_targets.narrow(1, 0, 3).contiguous();
        mismatched_shape_attention = mismatched_shape_attention.narrow(1, 0, 3).contiguous();
    }
    assert(qwen36_train_micro_step(
        distributed, &mismatched_shape_input, &mismatched_shape_targets,
        &mismatched_shape_attention, 1.0, 1) < 0.0);
    assert(qwen36_get_step_count(distributed) == 0);
    assert(max_diff(*fixed_preflight_probe, fixed_preflight_probe_before) == 0.0);

    // The scale itself is finite, but sources with multiple supervised tokens
    // overflow the product. Every rank must fail before forward/backward.
    assert(qwen36_train_micro_step(
        distributed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask,
        std::numeric_limits<double>::max(), 1) < 0.0);
    assert(qwen36_get_step_count(distributed) == 0);
    assert(max_diff(*fixed_preflight_probe, fixed_preflight_probe_before) == 0.0);

    // Optimizer phase is part of the distributed fixed-LoRA clock. A rank-local
    // apply decision must fail before forward or accumulation can diverge.
    const double mismatched_phase_loss = qwen36_train_micro_step(
        distributed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask, 1.0, rank == 0 ? 0 : 1);
    assert(mismatched_phase_loss < 0.0);
    assert(qwen36_get_step_count(distributed) == 0);
    assert(max_diff(*fixed_preflight_probe, fixed_preflight_probe_before) == 0.0);

    // A rank-local fixed optimizer clock must fail through full-topology
    // consensus before gradient collectives or Adam can diverge.
    auto* fixed_clock_probe = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_b(distributed, fixed_fixtures.front().slot));
    assert(fixed_clock_probe);
    const auto fixed_clock_probe_before = fixed_clock_probe->clone();
    if (rank == 0)
        assert(qwen36_set_step_count(distributed, 1) == 0);
    const double mismatched_clock_loss = qwen36_train_step(
        distributed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask);
    assert(mismatched_clock_loss < 0.0);
    if (rank == 0)
        assert(qwen36_set_step_count(distributed, 0) == 0);
    assert(qwen36_get_step_count(distributed) == 0);
    assert(max_diff(*fixed_clock_probe, fixed_clock_probe_before) == 0.0);

    // Registry mutation is collective and must reject disagreement about a
    // pending accumulation window before adapter parameter synchronization.
    assert(qwen36_train_micro_step(
        distributed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask, 1.0, 0) > 0.0);
    if (rank == 0)
        assert(qwen36_abort_gradient_accumulation(distributed) == 0);
    assert(qwen36_add_lora(
        distributed, kLoraRank, 1.0, &target_layer, 1, targets) < 0);
    assert(qwen36_abort_gradient_accumulation(distributed) == 0);
    assert(qwen36_get_step_count(distributed) == 0);

    const double distributed_loss = qwen36_train_step(
        distributed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask);
    const double reference_loss = qwen36_train_step(
        reference, &global_batch.input_ids, &global_batch.target_mask,
        &global_batch.attention_mask);
    assert(distributed_loss > 0.0 && std::isfinite(distributed_loss));
    assert(reference_loss > 0.0 && std::isfinite(reference_loss));

    std::vector<void*> local_m(kOptimizerSlots), local_v(kOptimizerSlots);
    std::vector<void*> full_m(kOptimizerSlots), full_v(kOptimizerSlots);
    assert(qwen36_export_optimizer_state(
        distributed, local_m.data(), local_v.data(), kOptimizerSlots) ==
        kOptimizerSlots);
    assert(qwen36_export_optimizer_state(
        reference, full_m.data(), full_v.data(), kOptimizerSlots) ==
        kOptimizerSlots);

    ParityErrors fixed_errors;
    for (size_t fixture_index = 0;
         fixture_index < fixed_fixtures.size(); ++fixture_index) {
        auto& fixture = fixed_fixtures[fixture_index];
        auto* local_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(distributed, fixture.slot));
        auto* local_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(distributed, fixture.slot));
        auto* ref_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(reference, fixture.slot));
        auto* ref_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(reference, fixture.slot));
        assert(local_a && local_b && ref_a && ref_b);
        fixed_errors.param = std::max({fixed_errors.param,
            max_diff(*local_a, local_factor(
                *ref_a, fixture.kind, false, tp_rank, ep_rank,
                base_tp_mlp)),
            max_diff(*local_b, local_factor(
                *ref_b, fixture.kind, true, tp_rank, ep_rank,
                base_tp_mlp))});

        const int64_t a_state = 2 * fixture.slot;
        const int64_t b_state = a_state + 1;
        auto* m_a = reinterpret_cast<at::Tensor*>(local_m[a_state]);
        auto* m_b = reinterpret_cast<at::Tensor*>(local_m[b_state]);
        auto* v_a = reinterpret_cast<at::Tensor*>(local_v[a_state]);
        auto* v_b = reinterpret_cast<at::Tensor*>(local_v[b_state]);
        auto* ref_m_a = reinterpret_cast<at::Tensor*>(full_m[a_state]);
        auto* ref_m_b = reinterpret_cast<at::Tensor*>(full_m[b_state]);
        auto* ref_v_a = reinterpret_cast<at::Tensor*>(full_v[a_state]);
        auto* ref_v_b = reinterpret_cast<at::Tensor*>(full_v[b_state]);
        assert(m_a && m_b && v_a && v_b &&
            ref_m_a && ref_m_b && ref_v_a && ref_v_b);
        fixed_errors.m = std::max({fixed_errors.m,
            max_diff(*m_a, local_factor(
                *ref_m_a, fixture.kind, false, tp_rank, ep_rank,
                base_tp_mlp)),
            max_diff(*m_b, local_factor(
                *ref_m_b, fixture.kind, true, tp_rank, ep_rank,
                base_tp_mlp))});
        fixed_errors.v = std::max({fixed_errors.v,
            max_diff(*v_a, local_factor(
                *ref_v_a, fixture.kind, false, tp_rank, ep_rank,
                base_tp_mlp)),
            max_diff(*v_b, local_factor(
                *ref_v_b, fixture.kind, true, tp_rank, ep_rank,
                base_tp_mlp))});
        fixed_errors.adam = std::max({fixed_errors.adam,
            max_diff(*local_a, adam_expected(
                fixed_before[fixture_index][0], *m_a, *v_a)),
            max_diff(*local_b, adam_expected(
                fixed_before[fixture_index][1], *m_b, *v_b))});
        assert(update_norm(*local_a, fixed_before[fixture_index][0]) > 0.0);
        assert(update_norm(*local_b, fixed_before[fixture_index][1]) > 0.0);
    }

    std::printf(
        "native_tp_ep_fixed rank=%d tp=%d ep=%d dp=%d source_mode=%s "
        "local_loss=%0.8f "
        "reference_loss=%0.8f param_diff=%0.8e m_diff=%0.8e "
        "v_diff=%0.8e adam_error=%0.8e\n",
        rank, tp_rank, ep_rank, dp_rank,
        sharded_source ? "sharded" : "replicated",
        distributed_loss, reference_loss,
        fixed_errors.param, fixed_errors.m, fixed_errors.v,
        fixed_errors.adam);
    std::fflush(stdout);
    // TP matmul reductions and EP dispatch round BF16 in a different order
    // from the full model. FP32 optimizer states remain the tight oracle.
    assert(fixed_errors.param <= 3e-3);
    assert(fixed_errors.m <= 7e-3);
    assert(fixed_errors.v <= 2e-5);
    assert(fixed_errors.adam <= 1e-5);

    // Simulate a safetensors checkpoint: fixed LoRA and Adam state leave the
    // device, then a fresh distributed context restores them from CPU tensors.
    std::vector<at::Tensor> checkpoint_a;
    std::vector<at::Tensor> checkpoint_b;
    std::vector<at::Tensor> checkpoint_m;
    std::vector<at::Tensor> checkpoint_v;
    checkpoint_a.reserve(kLoraPairs);
    checkpoint_b.reserve(kLoraPairs);
    checkpoint_m.reserve(kOptimizerSlots);
    checkpoint_v.reserve(kOptimizerSlots);
    for (int64_t slot = 0; slot < kLoraPairs; ++slot) {
        auto* a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(distributed, slot));
        auto* b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(distributed, slot));
        assert(a && b);
        checkpoint_a.push_back(a->to(at::kCPU).clone());
        checkpoint_b.push_back(b->to(at::kCPU).clone());
    }
    for (int64_t index = 0; index < kOptimizerSlots; ++index) {
        auto* m = reinterpret_cast<at::Tensor*>(local_m[index]);
        auto* v = reinterpret_cast<at::Tensor*>(local_v[index]);
        assert(m && v);
        checkpoint_m.push_back(m->to(at::kCPU).clone());
        checkpoint_v.push_back(v->to(at::kCPU).clone());
    }

    void* resumed = qwen36_create_training_context_ex(
        local_ptrs.data(), local_ptrs.size(), &local_embed, &final_norm,
        &local_lm_head, &distributed_config, 1,
        static_cast<int32_t>(at::kBFloat16),
        1.0, kLearningRate, kBeta1, kBeta2, kAdamEps,
        kVocab, 1e-5, kLoraRank, &target_layer, 1, targets,
        distributed_flags);
    assert(resumed);
    std::vector<std::array<at::Tensor, 2>> shadow_fixed_before_attach;
    shadow_fixed_before_attach.reserve(kLoraPairs);
    for (int64_t slot = 0; slot < kLoraPairs; ++slot) {
        auto* a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(resumed, slot));
        auto* b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(resumed, slot));
        assert(a && b);
        if (dp_rank > 0) {
            auto divergent_a = a->detach().clone().add_(0.125);
            auto divergent_b = b->detach().clone().add_(0.25);
            assert(qwen36_set_lora_tensor(
                resumed, slot, 0, &divergent_a) == 0);
            assert(qwen36_set_lora_tensor(
                resumed, slot, 1, &divergent_b) == 0);
        }
        shadow_fixed_before_attach.push_back({a->clone(), b->clone()});
    }
    assert(qwen36_attach_parallel_nccl_no_sync(
        resumed, rank, world,
        tp_rank, 2, tp_color,
        ep_rank, 2, ep_color,
        dp_rank, dp_size, dp_color) == 0);
    for (int64_t slot = 0; slot < kLoraPairs; ++slot) {
        auto* a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(resumed, slot));
        auto* b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(resumed, slot));
        assert(a && b);
        assert(max_diff(*a, shadow_fixed_before_attach[slot][0]) == 0.0);
        assert(max_diff(*b, shadow_fixed_before_attach[slot][1]) == 0.0);
    }
    for (int64_t slot = 0; slot < kLoraPairs; ++slot) {
        assert(qwen36_set_lora_tensor(
            resumed, slot, 0, &checkpoint_a[slot]) == 0);
        assert(qwen36_set_lora_tensor(
            resumed, slot, 1, &checkpoint_b[slot]) == 0);
    }
    auto checkpoint_m_ptrs = pointers(checkpoint_m);
    auto checkpoint_v_ptrs = pointers(checkpoint_v);
    assert(qwen36_import_optimizer_state(
        resumed, checkpoint_m_ptrs.data(), checkpoint_v_ptrs.data(),
        kOptimizerSlots) == kOptimizerSlots);
    assert(qwen36_set_step_count(
        resumed, qwen36_get_step_count(distributed)) == 0);

    std::vector<void*> resumed_m(kOptimizerSlots), resumed_v(kOptimizerSlots);
    assert(qwen36_export_optimizer_state(
        resumed, resumed_m.data(), resumed_v.data(), kOptimizerSlots) ==
        kOptimizerSlots);
    double restored_param_diff = 0.0;
    double restored_m_diff = 0.0;
    double restored_v_diff = 0.0;
    for (int64_t slot = 0; slot < kLoraPairs; ++slot) {
        auto* original_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(distributed, slot));
        auto* original_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(distributed, slot));
        auto* restored_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(resumed, slot));
        auto* restored_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(resumed, slot));
        assert(original_a && original_b && restored_a && restored_b);
        assert(restored_a->is_cuda() && restored_b->is_cuda());
        restored_param_diff = std::max({restored_param_diff,
            max_diff(*original_a, *restored_a),
            max_diff(*original_b, *restored_b)});
    }
    for (int64_t index = 0; index < kOptimizerSlots; ++index) {
        auto* original_m = reinterpret_cast<at::Tensor*>(local_m[index]);
        auto* original_v = reinterpret_cast<at::Tensor*>(local_v[index]);
        auto* restored_m = reinterpret_cast<at::Tensor*>(resumed_m[index]);
        auto* restored_v = reinterpret_cast<at::Tensor*>(resumed_v[index]);
        assert(original_m && original_v && restored_m && restored_v);
        assert(restored_m->is_cuda() && restored_v->is_cuda());
        assert(restored_m->scalar_type() == at::kFloat);
        assert(restored_v->scalar_type() == at::kFloat);
        restored_m_diff = std::max(
            restored_m_diff, max_diff(*original_m, *restored_m));
        restored_v_diff = std::max(
            restored_v_diff, max_diff(*original_v, *restored_v));
    }
    assert(restored_param_diff == 0.0);
    assert(restored_m_diff == 0.0);
    assert(restored_v_diff == 0.0);
    assert(qwen36_get_step_count(resumed) ==
        qwen36_get_step_count(distributed));

    const double continued_loss = qwen36_train_step(
        distributed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask);
    const double resumed_loss = qwen36_train_step(
        resumed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask);
    assert(std::abs(continued_loss - resumed_loss) <= 1e-6);
    assert(qwen36_export_optimizer_state(
        distributed, local_m.data(), local_v.data(), kOptimizerSlots) ==
        kOptimizerSlots);
    assert(qwen36_export_optimizer_state(
        resumed, resumed_m.data(), resumed_v.data(), kOptimizerSlots) ==
        kOptimizerSlots);
    double continued_param_diff = 0.0;
    double continued_m_diff = 0.0;
    double continued_v_diff = 0.0;
    for (int64_t slot = 0; slot < kLoraPairs; ++slot) {
        auto* continued_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(distributed, slot));
        auto* continued_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(distributed, slot));
        auto* restored_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(resumed, slot));
        auto* restored_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(resumed, slot));
        continued_param_diff = std::max({continued_param_diff,
            max_diff(*continued_a, *restored_a),
            max_diff(*continued_b, *restored_b)});
    }
    for (int64_t index = 0; index < kOptimizerSlots; ++index) {
        continued_m_diff = std::max(continued_m_diff, max_diff(
            *reinterpret_cast<at::Tensor*>(local_m[index]),
            *reinterpret_cast<at::Tensor*>(resumed_m[index])));
        continued_v_diff = std::max(continued_v_diff, max_diff(
            *reinterpret_cast<at::Tensor*>(local_v[index]),
            *reinterpret_cast<at::Tensor*>(resumed_v[index])));
    }
    std::printf(
        "native_tp_ep_resume rank=%d loss=%0.8f param_diff=%0.8e "
        "m_diff=%0.8e v_diff=%0.8e step=%ld\n",
        rank, resumed_loss, continued_param_diff, continued_m_diff,
        continued_v_diff,
        static_cast<long>(qwen36_get_step_count(resumed)));
    std::fflush(stdout);
    assert(continued_param_diff == 0.0);
    assert(continued_m_diff == 0.0);
    assert(continued_v_diff == 0.0);
    assert(qwen36_get_step_count(resumed) == 2);
    const int64_t fixed_step_before_dynamic =
        qwen36_get_step_count(distributed);

    const int64_t dynamic_targets[] = {0};
    const int64_t tenant_one = qwen36_add_lora(
        distributed, kLoraRank, kLoraRank,
        dynamic_targets, 1, targets);
    const int64_t tenant_two = qwen36_add_lora(
        distributed, kLoraRank, kLoraRank,
        dynamic_targets, 1, targets);
    assert(tenant_one > 0 && tenant_two > tenant_one);
    auto dynamic_one = make_lora_fixtures(
        tp_rank, ep_rank, 4001, base_tp_mlp);
    auto dynamic_two = make_lora_fixtures(
        tp_rank, ep_rank, 6001, base_tp_mlp);
    auto reference_dynamic_one = make_lora_fixtures(
        tp_rank, ep_rank, 4001, base_tp_mlp);
    install_dynamic(distributed, tenant_one, dynamic_one, true);
    install_dynamic(distributed, tenant_two, dynamic_two, true);
    // A selected dynamic tenant owns one activation row locally. Its full-rank
    // oracle must aggregate every source row into one adapter update, which a
    // fresh fixed-LoRA context does without conflating rows with tenant IDs.
    set_reference_env();
    void* dynamic_reference = qwen36_create_training_context(
        full_ptrs.data(), full_ptrs.size(), &embed, &final_norm, &lm_head,
        &reference_config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, kLearningRate, kBeta1, kBeta2, kAdamEps,
        kVocab, 1e-5, kLoraRank, &target_layer, 1, targets);
    assert(dynamic_reference);
    install_fixed(dynamic_reference, reference_dynamic_one, false);
    set_distributed_env(rank, world, tp_rank, ep_rank, dp_rank, dp_size);
    assert(qwen36_train_step(
        distributed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask) < 0.0);
    assert(qwen36_get_step_count(distributed) == fixed_step_before_dynamic);
    assert(qwen36_get_adapter_step_count(distributed, tenant_one) == 0);
    assert(qwen36_get_adapter_step_count(distributed, tenant_two) == 0);

    std::vector<std::array<at::Tensor, 2>> tenant_one_before;
    std::vector<std::array<at::Tensor, 2>> tenant_two_before;
    for (size_t index = 0; index < dynamic_one.size(); ++index) {
        tenant_one_before.push_back({
            dynamic_tensor(distributed, tenant_one,
                dynamic_one[index].module, false)->clone(),
            dynamic_tensor(distributed, tenant_one,
                dynamic_one[index].module, true)->clone(),
        });
        tenant_two_before.push_back({
            dynamic_tensor(distributed, tenant_two,
                dynamic_two[index].module, false)->clone(),
            dynamic_tensor(distributed, tenant_two,
                dynamic_two[index].module, true)->clone(),
        });
    }

    const double dynamic_loss = qwen36_train_multi_lora_selected(
        distributed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask, &tenant_one, 1, kLoraRank);
    const double reference_dynamic_loss = qwen36_train_step(
        dynamic_reference, &global_batch.input_ids, &global_batch.target_mask,
        &global_batch.attention_mask);
    assert(dynamic_loss > 0.0 && std::isfinite(dynamic_loss));
    assert(reference_dynamic_loss > 0.0 && std::isfinite(reference_dynamic_loss));
    assert(qwen36_get_adapter_step_count(distributed, tenant_one) == 1);
    assert(qwen36_get_adapter_step_count(distributed, tenant_two) == 0);
    assert(qwen36_get_step_count(distributed) == fixed_step_before_dynamic);

    double selected_update = 0.0;
    double isolated_diff = 0.0;
    double dynamic_adam_error = 0.0;
    double dynamic_param_diff = 0.0;
    double dynamic_m_diff = 0.0;
    double dynamic_v_diff = 0.0;
    std::vector<void*> dynamic_reference_m(kOptimizerSlots);
    std::vector<void*> dynamic_reference_v(kOptimizerSlots);
    assert(qwen36_export_optimizer_state(
        dynamic_reference, dynamic_reference_m.data(),
        dynamic_reference_v.data(), kOptimizerSlots) == kOptimizerSlots);
    for (size_t index = 0; index < dynamic_one.size(); ++index) {
        const char* module = dynamic_one[index].module;
        auto* selected_a = dynamic_tensor(
            distributed, tenant_one, module, false);
        auto* selected_b = dynamic_tensor(
            distributed, tenant_one, module, true);
        auto* isolated_a = dynamic_tensor(
            distributed, tenant_two, module, false);
        auto* isolated_b = dynamic_tensor(
            distributed, tenant_two, module, true);
        const int64_t a_state = 2 * dynamic_one[index].slot;
        const int64_t b_state = a_state + 1;
        auto* reference_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(dynamic_reference, dynamic_one[index].slot));
        auto* reference_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(dynamic_reference, dynamic_one[index].slot));
        selected_update += update_norm(
            *selected_a, tenant_one_before[index][0]);
        selected_update += update_norm(
            *selected_b, tenant_one_before[index][1]);
        isolated_diff = std::max({isolated_diff,
            max_diff(*isolated_a, tenant_two_before[index][0]),
            max_diff(*isolated_b, tenant_two_before[index][1])});

        auto* m_a = dynamic_state(
            distributed, tenant_one, module, false, false);
        auto* v_a = dynamic_state(
            distributed, tenant_one, module, false, true);
        auto* m_b = dynamic_state(
            distributed, tenant_one, module, true, false);
        auto* v_b = dynamic_state(
            distributed, tenant_one, module, true, true);
        auto* reference_m_a = reinterpret_cast<at::Tensor*>(
            dynamic_reference_m[a_state]);
        auto* reference_v_a = reinterpret_cast<at::Tensor*>(
            dynamic_reference_v[a_state]);
        auto* reference_m_b = reinterpret_cast<at::Tensor*>(
            dynamic_reference_m[b_state]);
        auto* reference_v_b = reinterpret_cast<at::Tensor*>(
            dynamic_reference_v[b_state]);
        assert(reference_a && reference_b && reference_m_a && reference_v_a &&
            reference_m_b && reference_v_b);
        dynamic_param_diff = std::max({dynamic_param_diff,
            max_diff(*selected_a, local_factor(
                *reference_a, dynamic_one[index].kind, false,
                tp_rank, ep_rank, base_tp_mlp)),
            max_diff(*selected_b, local_factor(
                *reference_b, dynamic_one[index].kind, true,
                tp_rank, ep_rank, base_tp_mlp))});
        dynamic_m_diff = std::max({dynamic_m_diff,
            max_diff(*m_a, local_factor(
                *reference_m_a, dynamic_one[index].kind, false,
                tp_rank, ep_rank, base_tp_mlp)),
            max_diff(*m_b, local_factor(
                *reference_m_b, dynamic_one[index].kind, true,
                tp_rank, ep_rank, base_tp_mlp))});
        dynamic_v_diff = std::max({dynamic_v_diff,
            max_diff(*v_a, local_factor(
                *reference_v_a, dynamic_one[index].kind, false,
                tp_rank, ep_rank, base_tp_mlp)),
            max_diff(*v_b, local_factor(
                *reference_v_b, dynamic_one[index].kind, true,
                tp_rank, ep_rank, base_tp_mlp))});
        dynamic_adam_error = std::max({dynamic_adam_error,
            max_diff(*selected_a, adam_expected(
                tenant_one_before[index][0], *m_a, *v_a)),
            max_diff(*selected_b, adam_expected(
                tenant_one_before[index][1], *m_b, *v_b))});
        assert(dynamic_state(
            distributed, tenant_two, module, false, false)
            ->abs().max().item<double>() == 0.0);
        assert(dynamic_state(
            distributed, tenant_two, module, true, false)
            ->abs().max().item<double>() == 0.0);
    }
    std::printf(
        "native_tp_ep_dynamic rank=%d tp=%d ep=%d dp=%d loss=%0.8f "
        "selected_update=%0.8e isolated_diff=%0.8e adam_error=%0.8e "
        "param_diff=%0.8e m_diff=%0.8e v_diff=%0.8e "
        "steps=[%ld,%ld]\n",
        rank, tp_rank, ep_rank, dp_rank, dynamic_loss, selected_update,
        isolated_diff, dynamic_adam_error, dynamic_param_diff,
        dynamic_m_diff, dynamic_v_diff,
        static_cast<long>(qwen36_get_adapter_step_count(
            distributed, tenant_one)),
        static_cast<long>(qwen36_get_adapter_step_count(
            distributed, tenant_two)));
    std::fflush(stdout);
    assert(selected_update > 0.0);
    assert(isolated_diff == 0.0);
    assert(dynamic_adam_error <= 1e-5);
    assert(dynamic_param_diff <= 3e-3);
    assert(dynamic_m_diff <= 7e-3);
    assert(dynamic_v_diff <= 2e-5);

    // Round-trip one tenant through CPU tensors and its independent optimizer
    // clock, then require exact next-step parity with the uninterrupted context.
    std::vector<std::array<at::Tensor, 6>> dynamic_checkpoint;
    dynamic_checkpoint.reserve(dynamic_one.size());
    for (const auto& fixture : dynamic_one) {
        dynamic_checkpoint.push_back({
            dynamic_tensor(distributed, tenant_one, fixture.module, false)
                ->to(at::kCPU).clone(),
            dynamic_tensor(distributed, tenant_one, fixture.module, true)
                ->to(at::kCPU).clone(),
            dynamic_state(distributed, tenant_one, fixture.module, false, false)
                ->to(at::kCPU).clone(),
            dynamic_state(distributed, tenant_one, fixture.module, false, true)
                ->to(at::kCPU).clone(),
            dynamic_state(distributed, tenant_one, fixture.module, true, false)
                ->to(at::kCPU).clone(),
            dynamic_state(distributed, tenant_one, fixture.module, true, true)
                ->to(at::kCPU).clone(),
        });
    }
    const int64_t resumed_tenant_one = qwen36_add_lora_for_restore(
        resumed, kLoraRank, kLoraRank, dynamic_targets, 1, targets);
    const int64_t resumed_tenant_two = qwen36_add_lora_for_restore(
        resumed, kLoraRank, kLoraRank, dynamic_targets, 1, targets);
    assert(resumed_tenant_one == tenant_one);
    assert(resumed_tenant_two == tenant_two);
    for (size_t index = 0; index < dynamic_one.size(); ++index) {
        const char* module = dynamic_one[index].module;
        auto& state = dynamic_checkpoint[index];
        assert(qwen36_set_adapter_lora_tensor(
            resumed, resumed_tenant_one, 0, module, 0, &state[0]) == 0);
        assert(qwen36_set_adapter_lora_tensor(
            resumed, resumed_tenant_one, 0, module, 1, &state[1]) == 0);
        assert(qwen36_set_adapter_optimizer_tensor(
            resumed, resumed_tenant_one, 0, module, 0, 0, &state[2]) == 0);
        assert(qwen36_set_adapter_optimizer_tensor(
            resumed, resumed_tenant_one, 0, module, 0, 1, &state[3]) == 0);
        assert(qwen36_set_adapter_optimizer_tensor(
            resumed, resumed_tenant_one, 0, module, 1, 0, &state[4]) == 0);
        assert(qwen36_set_adapter_optimizer_tensor(
            resumed, resumed_tenant_one, 0, module, 1, 1, &state[5]) == 0);
    }
    assert(qwen36_set_adapter_step_count(
        resumed, resumed_tenant_one,
        qwen36_get_adapter_step_count(distributed, tenant_one)) == 0);
    assert(qwen36_get_adapter_step_count(resumed, resumed_tenant_one) ==
        qwen36_get_adapter_step_count(distributed, tenant_one));
    double restored_dynamic_diff = 0.0;
    for (const auto& fixture : dynamic_one) {
        for (int is_b = 0; is_b < 2; ++is_b) {
            restored_dynamic_diff = std::max(restored_dynamic_diff, max_diff(
                *dynamic_tensor(distributed, tenant_one, fixture.module, is_b),
                *dynamic_tensor(resumed, resumed_tenant_one, fixture.module, is_b)));
            for (int is_v = 0; is_v < 2; ++is_v) {
                restored_dynamic_diff = std::max(restored_dynamic_diff, max_diff(
                    *dynamic_state(
                        distributed, tenant_one, fixture.module, is_b, is_v),
                    *dynamic_state(
                        resumed, resumed_tenant_one, fixture.module, is_b, is_v)));
            }
        }
    }
    assert(restored_dynamic_diff == 0.0);
    const int64_t fixed_step_before_resumed_dynamic =
        qwen36_get_step_count(distributed);
    const double continued_dynamic_loss = qwen36_train_multi_lora_selected(
        distributed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask, &tenant_one, 1, kLoraRank);
    const double resumed_dynamic_loss = qwen36_train_multi_lora_selected(
        resumed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask, &resumed_tenant_one, 1, kLoraRank);
    assert(std::abs(continued_dynamic_loss - resumed_dynamic_loss) <= 1e-6);
    assert(qwen36_get_adapter_step_count(distributed, tenant_one) == 2);
    assert(qwen36_get_adapter_step_count(resumed, resumed_tenant_one) == 2);
    assert(qwen36_get_adapter_step_count(resumed, resumed_tenant_two) == 0);
    assert(qwen36_get_step_count(distributed) == fixed_step_before_resumed_dynamic);
    assert(qwen36_get_step_count(resumed) == fixed_step_before_resumed_dynamic);
    double resumed_dynamic_diff = 0.0;
    for (const auto& fixture : dynamic_one) {
        for (int is_b = 0; is_b < 2; ++is_b) {
            resumed_dynamic_diff = std::max(resumed_dynamic_diff, max_diff(
                *dynamic_tensor(distributed, tenant_one, fixture.module, is_b),
                *dynamic_tensor(resumed, resumed_tenant_one, fixture.module, is_b)));
            for (int is_v = 0; is_v < 2; ++is_v) {
                resumed_dynamic_diff = std::max(resumed_dynamic_diff, max_diff(
                    *dynamic_state(distributed, tenant_one, fixture.module, is_b, is_v),
                    *dynamic_state(resumed, resumed_tenant_one, fixture.module, is_b, is_v)));
            }
        }
    }
    assert(resumed_dynamic_diff == 0.0);

    const int64_t heterogeneous_tenant = qwen36_add_lora_v2(
        distributed, 3, 3.0, &target_layer, 1, projection_targets);
    assert(heterogeneous_tenant > tenant_two);
    auto* heterogeneous_b = dynamic_tensor(
        distributed, heterogeneous_tenant, projection_targets, true);
    auto* homogeneous_b = dynamic_tensor(
        distributed, tenant_one, projection_targets, true);
    auto heterogeneous_before = heterogeneous_b->clone();
    auto homogeneous_before = homogeneous_b->clone();
    const int64_t heterogeneous_ids[] = {tenant_one, heterogeneous_tenant};
    const int64_t tenant_one_step_before =
        qwen36_get_adapter_step_count(distributed, tenant_one);

    // TP peers are replicas of the same source row. A rank-local target mask
    // change must fail on every topology coordinate before gradient sync.
    auto mismatched_tp_targets = local_batch.target_mask.clone();
    if (ep_rank == 0 &&
        ((dp_rank == 0 && tp_rank == 0) ||
         (dp_rank == 1 && tp_rank == 1))) {
        mismatched_tp_targets.zero_();
    }
    const int64_t finalizers_before_tp_mismatch =
        qwen36_get_dynamic_finalizer_count(distributed);
    const int64_t adam_before_tp_mismatch =
        qwen36_get_dynamic_adam_launch_count(distributed);
    assert(qwen36_train_multi_lora_selected_v2(
        distributed, &local_batch.input_ids, &mismatched_tp_targets,
        &local_batch.attention_mask, heterogeneous_ids, 2) < 0.0);
    assert(qwen36_get_adapter_step_count(distributed, tenant_one) ==
        tenant_one_step_before);
    assert(qwen36_get_adapter_step_count(distributed, heterogeneous_tenant) == 0);
    assert(max_diff(*homogeneous_b, homogeneous_before) == 0.0);
    assert(max_diff(*heterogeneous_b, heterogeneous_before) == 0.0);
    assert(qwen36_get_dynamic_finalizer_count(distributed) ==
        finalizers_before_tp_mismatch + 1);
    assert(qwen36_get_dynamic_adam_launch_count(distributed) ==
        adam_before_tp_mismatch);

    const int64_t finalizers_before_heterogeneous =
        qwen36_get_dynamic_finalizer_count(distributed);
    const int64_t adam_launches_before_heterogeneous =
        qwen36_get_dynamic_adam_launch_count(distributed);

    if (rank == 0)
        setenv("QWEN36_TEST_FAIL_HETERO_GROUP_AFTER", "1", 1);
    assert(qwen36_train_multi_lora_selected_v2(
        distributed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask, heterogeneous_ids, 2) < 0.0);
    if (rank == 0)
        unsetenv("QWEN36_TEST_FAIL_HETERO_GROUP_AFTER");
    assert(qwen36_get_adapter_step_count(distributed, tenant_one) ==
        tenant_one_step_before);
    assert(qwen36_get_adapter_step_count(distributed, heterogeneous_tenant) == 0);
    assert(max_diff(*homogeneous_b, homogeneous_before) == 0.0);
    assert(max_diff(*heterogeneous_b, heterogeneous_before) == 0.0);
    assert(qwen36_get_dynamic_finalizer_count(distributed) ==
        finalizers_before_heterogeneous);
    assert(qwen36_get_dynamic_adam_launch_count(distributed) ==
        adam_launches_before_heterogeneous);

    if (rank == 0)
        setenv("QWEN36_TEST_FAIL_FINALIZER_BEFORE_TOKEN_PREFLIGHT", "1", 1);
    assert(qwen36_train_multi_lora_selected_v2(
        distributed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask, heterogeneous_ids, 2) < 0.0);
    if (rank == 0)
        unsetenv("QWEN36_TEST_FAIL_FINALIZER_BEFORE_TOKEN_PREFLIGHT");
    assert(qwen36_get_adapter_step_count(distributed, tenant_one) ==
        tenant_one_step_before);
    assert(qwen36_get_adapter_step_count(distributed, heterogeneous_tenant) == 0);
    assert(max_diff(*homogeneous_b, homogeneous_before) == 0.0);
    assert(max_diff(*heterogeneous_b, heterogeneous_before) == 0.0);
    assert(qwen36_get_dynamic_finalizer_count(distributed) ==
        finalizers_before_heterogeneous + 1);
    assert(qwen36_get_dynamic_adam_launch_count(distributed) ==
        adam_launches_before_heterogeneous);

    if (rank == 0)
        setenv("QWEN36_TEST_FAIL_DYNAMIC_ADAM_BEFORE_COMMIT", "1", 1);
    assert(qwen36_train_multi_lora_selected_v2(
        distributed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask, heterogeneous_ids, 2) < 0.0);
    if (rank == 0)
        unsetenv("QWEN36_TEST_FAIL_DYNAMIC_ADAM_BEFORE_COMMIT");
    assert(qwen36_get_adapter_step_count(distributed, tenant_one) ==
        tenant_one_step_before);
    assert(qwen36_get_adapter_step_count(distributed, heterogeneous_tenant) == 0);
    assert(max_diff(*homogeneous_b, homogeneous_before) == 0.0);
    assert(max_diff(*heterogeneous_b, heterogeneous_before) == 0.0);
    assert(qwen36_get_dynamic_finalizer_count(distributed) ==
        finalizers_before_heterogeneous + 2);
    assert(qwen36_get_dynamic_adam_launch_count(distributed) ==
        adam_launches_before_heterogeneous + 1);

    // Exercise request-order token normalization across two signatures with
    // unequal positive counts and a third globally-empty tenant. The empty
    // tenant must retain its parameters, Adam state, and private clock.
    const int64_t heterogeneous_batch_ids[] = {
        tenant_one, heterogeneous_tenant, tenant_two};
    auto dense_targets = at::ones_like(local_batch.target_mask);
    dense_targets.select(1, 0).zero_();
    auto empty_targets = at::zeros_like(local_batch.target_mask);
    auto heterogeneous_input = local_batch.input_ids.repeat({3, 1});
    auto heterogeneous_targets = at::cat({
        local_batch.target_mask, dense_targets, empty_targets}, 0);
    auto heterogeneous_attention =
        local_batch.attention_mask.repeat({3, 1});
    auto* empty_tenant_b = dynamic_tensor(
        distributed, tenant_two, projection_targets, true);
    auto* empty_tenant_m = dynamic_state(
        distributed, tenant_two, projection_targets, true, false);
    auto* empty_tenant_v = dynamic_state(
        distributed, tenant_two, projection_targets, true, true);
    assert(empty_tenant_b && empty_tenant_m && empty_tenant_v);
    const auto empty_tenant_b_before = empty_tenant_b->clone();
    const auto empty_tenant_m_before = empty_tenant_m->clone();
    const auto empty_tenant_v_before = empty_tenant_v->clone();
    const int64_t empty_tenant_step_before =
        qwen36_get_adapter_step_count(distributed, tenant_two);
    const double heterogeneous_loss = qwen36_train_multi_lora_selected_v2(
        distributed, &heterogeneous_input, &heterogeneous_targets,
        &heterogeneous_attention, heterogeneous_batch_ids, 3);
    assert(heterogeneous_loss > 0.0 && std::isfinite(heterogeneous_loss));
    assert(qwen36_get_adapter_step_count(distributed, tenant_one) ==
        tenant_one_step_before + 1);
    assert(qwen36_get_adapter_step_count(distributed, heterogeneous_tenant) == 1);
    assert(qwen36_get_adapter_step_count(distributed, tenant_two) ==
        empty_tenant_step_before);
    assert(qwen36_get_dynamic_finalizer_count(distributed) ==
        finalizers_before_heterogeneous + 3);
    assert(qwen36_get_dynamic_adam_launch_count(distributed) ==
        adam_launches_before_heterogeneous + 2);
    assert(update_norm(*homogeneous_b, homogeneous_before) > 0.0);
    assert(update_norm(*heterogeneous_b, heterogeneous_before) > 0.0);
    assert(max_diff(*empty_tenant_b, empty_tenant_b_before) == 0.0);
    assert(max_diff(*empty_tenant_m, empty_tenant_m_before) == 0.0);
    assert(max_diff(*empty_tenant_v, empty_tenant_v_before) == 0.0);
    std::printf(
        "native_tp_ep_heterogeneous_v2 rank=%d loss=%0.8f ok\n",
        rank, heterogeneous_loss);
    std::fflush(stdout);

    qwen36_free_training_context(reference);
    qwen36_free_training_context(dynamic_reference);
    qwen36_free_training_context(resumed);
    qwen36_free_training_context(distributed);
    return 0;
}
