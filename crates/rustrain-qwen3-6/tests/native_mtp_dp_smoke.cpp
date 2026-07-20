#include <ATen/ATen.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    void*, int32_t, int32_t, int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
extern "C" int32_t qwen36_set_mtp_weights(
    void*, void*, void*, void*, void*, void**, int64_t, void*, int64_t);
extern "C" int64_t qwen36_add_lora(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" int32_t qwen36_set_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t, void*);
extern "C" void* qwen36_get_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t);
extern "C" void* qwen36_get_adapter_optimizer_tensor(
    void*, int64_t, int64_t, const char*, int32_t, int32_t);
extern "C" int64_t qwen36_get_adapter_step_count(void*, int64_t);
extern "C" int32_t qwen36_train_multi_lora_selected_v3(
    void*, void*, void*, void*, const int64_t*, int32_t,
    double*, double*, int32_t);
extern "C" void qwen36_free_training_context(void*);

static constexpr int32_t kDataParallel = 1 << 1;
static constexpr int64_t kHidden = 8;
static constexpr int64_t kHeads = 4;
static constexpr int64_t kKvHeads = 2;
static constexpr int64_t kHeadDim = 2;
static constexpr int64_t kIntermediate = 12;
static constexpr int64_t kVocab = 16;
static constexpr int64_t kLoraRank = 2;
static constexpr double kLearningRate = 1e-3;
static constexpr double kBeta1 = 0.9;
static constexpr double kBeta2 = 0.999;
static constexpr double kAdamEps = 1e-8;

static at::Tensor values(std::initializer_list<int64_t> shape, double scale) {
    int64_t count = 1;
    for (const auto dim : shape) count *= dim;
    return ((at::arange(count, at::TensorOptions().device(at::kCUDA)
        .dtype(at::kFloat)).remainder(23) - 11.0) * scale)
        .reshape(shape).to(at::kBFloat16);
}

static std::vector<void*> pointers(std::vector<at::Tensor>& tensors) {
    std::vector<void*> result;
    result.reserve(tensors.size());
    for (auto& tensor : tensors) result.push_back(&tensor);
    return result;
}

static double max_diff(const at::Tensor& lhs, const at::Tensor& rhs) {
    return (lhs.to(at::kFloat) - rhs.to(at::kFloat)).abs().max().item<double>();
}

struct ModelFixture {
    std::vector<at::Tensor> weights;
    std::vector<void*> weight_ptrs;
    at::Tensor embed;
    at::Tensor final_norm;
    at::Tensor lm_head;
    at::Tensor mtp_fc;
    at::Tensor mtp_pre_emb;
    at::Tensor mtp_pre_hidden;
    at::Tensor mtp_norm;
    LayerConfig config{};

    ModelFixture()
        : embed(values({kVocab, kHidden}, .020)),
          final_norm(at::ones({kHidden}, at::TensorOptions().device(at::kCUDA)
              .dtype(at::kBFloat16))),
          lm_head(values({kVocab, kHidden}, .015)),
          mtp_fc(values({kHidden, 2 * kHidden}, .006)),
          mtp_pre_emb(at::ones({kHidden}, at::TensorOptions().device(at::kCUDA)
              .dtype(at::kBFloat16))),
          mtp_pre_hidden(at::ones({kHidden}, at::TensorOptions().device(at::kCUDA)
              .dtype(at::kBFloat16))),
          mtp_norm(at::ones({kHidden}, at::TensorOptions().device(at::kCUDA)
              .dtype(at::kBFloat16))) {
        weights.push_back(at::ones({kHidden}, at::TensorOptions()
            .device(at::kCUDA).dtype(at::kBFloat16)));
        weights.push_back(at::ones({kHidden}, at::TensorOptions()
            .device(at::kCUDA).dtype(at::kBFloat16)));
        weights.push_back(values({2 * kHeads * kHeadDim, kHidden}, .010));
        weights.push_back(at::ones({kHeadDim}, at::TensorOptions()
            .device(at::kCUDA).dtype(at::kBFloat16)));
        weights.push_back(values({kKvHeads * kHeadDim, kHidden}, .012));
        weights.push_back(at::ones({kHeadDim}, at::TensorOptions()
            .device(at::kCUDA).dtype(at::kBFloat16)));
        weights.push_back(values({kKvHeads * kHeadDim, kHidden}, .008));
        weights.push_back(values({kHidden, kHeads * kHeadDim}, .011));
        weights.push_back(values({kIntermediate, kHidden}, .009));
        weights.push_back(values({kIntermediate, kHidden}, .007));
        weights.push_back(values({kHidden, kIntermediate}, .010));
        for (auto& tensor : weights) tensor.set_requires_grad(false);
        for (auto* tensor : {&embed, &final_norm, &lm_head, &mtp_fc,
                             &mtp_pre_emb, &mtp_pre_hidden, &mtp_norm})
            tensor->set_requires_grad(false);
        weight_ptrs = pointers(weights);

        config.layer_type = 0;
        config.num_heads = kHeads;
        config.num_kv_heads = kKvHeads;
        config.head_dim = kHeadDim;
        config.partial_rotary_factor = 1.0;
        config.rope_theta = 10000.0;
        config.rms_eps = 1e-5;
        config.intermediate_size = kIntermediate;
    }
};

struct LoraFixture {
    at::Tensor a;
    at::Tensor b;
};

static LoraFixture lora_fixture(double a_scale, double b_scale) {
    return {
        values({kLoraRank, kHidden}, a_scale),
        values({2 * kHeads * kHeadDim, kLoraRank}, b_scale),
    };
}

static void install_lora(
    void* context, int64_t adapter, const LoraFixture& fixture
) {
    auto a = fixture.a;
    auto b = fixture.b;
    assert(qwen36_set_adapter_lora_tensor(
        context, adapter, 0, "q_proj", 0, &a) == 0);
    assert(qwen36_set_adapter_lora_tensor(
        context, adapter, 0, "q_proj", 1, &b) == 0);
}

static void set_distributed_environment(int rank, int local_rank) {
    setenv("WORLD_SIZE", "2", 1);
    setenv("RANK", std::to_string(rank).c_str(), 1);
    setenv("LOCAL_RANK", std::to_string(local_rank).c_str(), 1);
    setenv("TP_SIZE", "1", 1);
    setenv("CP_SIZE", "1", 1);
    setenv("EP_SIZE", "1", 1);
    setenv("DP_SIZE", "2", 1);
    setenv("PP_SIZE", "1", 1);
    setenv("RUSTRAIN_TP_RANK", "0", 1);
    setenv("RUSTRAIN_CP_RANK", "0", 1);
    setenv("RUSTRAIN_EP_RANK", "0", 1);
    setenv("RUSTRAIN_DP_RANK", std::to_string(rank).c_str(), 1);
    setenv("RUSTRAIN_PP_RANK", "0", 1);
    setenv("RUSTRAIN_DATA_PARALLEL", "1", 1);
    unsetenv("QWEN36_DISABLE_MTP");
    setenv("QWEN36_MTP_LOSS_SCALE", "0.4", 1);
}

static void set_reference_environment(int local_rank) {
    setenv("WORLD_SIZE", "1", 1);
    setenv("RANK", "0", 1);
    setenv("LOCAL_RANK", std::to_string(local_rank).c_str(), 1);
    setenv("TP_SIZE", "1", 1);
    setenv("CP_SIZE", "1", 1);
    setenv("EP_SIZE", "1", 1);
    setenv("DP_SIZE", "1", 1);
    setenv("PP_SIZE", "1", 1);
    setenv("RUSTRAIN_TP_RANK", "0", 1);
    setenv("RUSTRAIN_CP_RANK", "0", 1);
    setenv("RUSTRAIN_EP_RANK", "0", 1);
    setenv("RUSTRAIN_DP_RANK", "0", 1);
    setenv("RUSTRAIN_PP_RANK", "0", 1);
    unsetenv("RUSTRAIN_DATA_PARALLEL");
    unsetenv("QWEN36_DISABLE_MTP");
    setenv("QWEN36_MTP_LOSS_SCALE", "0.4", 1);
}

static void* create_context(ModelFixture& model, bool data_parallel) {
    const int64_t target_layer = 0;
    const int32_t flags = data_parallel ? kDataParallel : 0;
    return qwen36_create_training_context_ex(
        model.weight_ptrs.data(), model.weight_ptrs.size(), &model.embed,
        &model.final_norm, &model.lm_head, &model.config, 1,
        static_cast<int32_t>(at::kBFloat16), 1.0, kLearningRate,
        kBeta1, kBeta2, kAdamEps, kVocab, 1e-5, kLoraRank,
        &target_layer, 1, "q_proj", flags);
}

static void install_mtp(void* context, ModelFixture& model) {
    auto mtp_weight_ptrs = pointers(model.weights);
    assert(qwen36_set_mtp_weights(
        context, &model.mtp_fc, &model.mtp_pre_emb,
        &model.mtp_pre_hidden, &model.mtp_norm,
        mtp_weight_ptrs.data(), mtp_weight_ptrs.size(),
        &model.config, 1) == 0);
}

struct Batch {
    at::Tensor ids;
    at::Tensor targets;
    at::Tensor attention;
};

static Batch tenant_one_batch() {
    auto ids = at::tensor({1, 2, 3, 4, 5, 6},
        at::TensorOptions().device(at::kCUDA).dtype(at::kLong)).reshape({1, 6});
    auto targets = at::tensor({0., 1., 1., 1., 1., 1.},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat)).reshape({1, 6});
    return {ids, targets, at::ones({1, 6},
        at::TensorOptions().device(at::kCUDA).dtype(at::kBool))};
}

static Batch tenant_two_batch() {
    auto ids = at::tensor({7, 8, 9, 10, 11, 12},
        at::TensorOptions().device(at::kCUDA).dtype(at::kLong)).reshape({1, 6});
    auto targets = at::tensor({0., 1., 1., 0., 0., 0.},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat)).reshape({1, 6});
    return {ids, targets, at::ones({1, 6},
        at::TensorOptions().device(at::kCUDA).dtype(at::kBool))};
}

static Batch distributed_batch(int dp_rank) {
    auto tenant_one = tenant_one_batch();
    auto tenant_two = tenant_two_batch();
    auto zero_one = at::zeros_like(tenant_one.targets);
    auto zero_two = at::zeros_like(tenant_two.targets);
    return {
        at::cat({tenant_one.ids, tenant_two.ids}, 0),
        dp_rank == 0
            ? at::cat({tenant_one.targets, zero_two}, 0)
            : at::cat({zero_one, tenant_two.targets}, 0),
        at::cat({tenant_one.attention, tenant_two.attention}, 0),
    };
}

struct AdapterState {
    std::array<at::Tensor, 2> parameters;
    std::array<at::Tensor, 2> adam_m;
    std::array<at::Tensor, 2> adam_v;
};

static AdapterState adapter_state(void* context, int64_t adapter) {
    AdapterState result;
    for (int is_b = 0; is_b < 2; ++is_b) {
        auto* parameter = reinterpret_cast<at::Tensor*>(
            qwen36_get_adapter_lora_tensor(
                context, adapter, 0, "q_proj", is_b));
        auto* adam_m = reinterpret_cast<at::Tensor*>(
            qwen36_get_adapter_optimizer_tensor(
                context, adapter, 0, "q_proj", is_b, 0));
        auto* adam_v = reinterpret_cast<at::Tensor*>(
            qwen36_get_adapter_optimizer_tensor(
                context, adapter, 0, "q_proj", is_b, 1));
        assert(parameter && adam_m && adam_v);
        result.parameters[is_b] = parameter->clone();
        result.adam_m[is_b] = adam_m->clone();
        result.adam_v[is_b] = adam_v->clone();
    }
    return result;
}

static double state_diff(const AdapterState& lhs, const AdapterState& rhs) {
    double result = 0.0;
    for (int index = 0; index < 2; ++index) {
        result = std::max(result,
            max_diff(lhs.parameters[index], rhs.parameters[index]));
        result = std::max(result, max_diff(lhs.adam_m[index], rhs.adam_m[index]));
        result = std::max(result, max_diff(lhs.adam_v[index], rhs.adam_v[index]));
    }
    return result;
}

struct ReferenceResult {
    void* context;
    int64_t adapter;
    double loss;
    AdapterState state;
};

static ReferenceResult run_reference(
    ModelFixture& model, const LoraFixture& lora, Batch& batch, bool enable_mtp
) {
    void* context = create_context(model, false);
    assert(context);
    if (enable_mtp) install_mtp(context, model);
    const int64_t target_layer = 0;
    const int64_t adapter = qwen36_add_lora(
        context, kLoraRank, 2.0, &target_layer, 1, "q_proj");
    assert(adapter > 0);
    install_lora(context, adapter, lora);
    double aggregate = -1.0;
    double loss = -1.0;
    assert(qwen36_train_multi_lora_selected_v3(
        context, &batch.ids, &batch.targets, &batch.attention,
        &adapter, 1, &aggregate, &loss, 1) == 0);
    assert(std::isfinite(aggregate) && std::isfinite(loss));
    assert(std::abs(aggregate - loss) < 1e-12);
    return {context, adapter, loss, adapter_state(context, adapter)};
}

int main() {
    const int rank = std::atoi(std::getenv("RANK"));
    const int world = std::atoi(std::getenv("WORLD_SIZE"));
    const int local_rank = std::atoi(
        std::getenv("LOCAL_RANK") ? std::getenv("LOCAL_RANK") : "0");
    assert(world == 2 && rank >= 0 && rank < world);
    qwen36_set_cuda_device(local_rank);
    set_distributed_environment(rank, local_rank);

    ModelFixture model;
    void* distributed = create_context(model, true);
    assert(distributed);
    install_mtp(distributed, model);
    assert(qwen36_init_parallel_nccl(
        distributed, rank, world,
        0, 1, rank,
        0, 1, rank,
        rank, 2, 0) == 0);

    const int64_t target_layer = 0;
    const LoraFixture lora_one = lora_fixture(.0020, .0010);
    const LoraFixture lora_two = lora_fixture(.0026, .0007);
    const LoraFixture lora_unselected = lora_fixture(.0008, .0016);
    const int64_t adapter_one = qwen36_add_lora(
        distributed, kLoraRank, 2.0, &target_layer, 1, "q_proj");
    const int64_t adapter_two = qwen36_add_lora(
        distributed, kLoraRank, 2.0, &target_layer, 1, "q_proj");
    const int64_t adapter_unselected = qwen36_add_lora(
        distributed, kLoraRank, 2.0, &target_layer, 1, "q_proj");
    assert(adapter_one > 0 && adapter_two > adapter_one &&
        adapter_unselected > adapter_two);
    install_lora(distributed, adapter_one, lora_one);
    install_lora(distributed, adapter_two, lora_two);
    install_lora(distributed, adapter_unselected, lora_unselected);
    const auto unselected_before = adapter_state(
        distributed, adapter_unselected);

    auto local_batch = distributed_batch(rank);
    const int64_t selected[] = {adapter_one, adapter_two};
    double aggregate_loss = -1.0;
    double adapter_losses[2] = {-1.0, -1.0};
    assert(qwen36_train_multi_lora_selected_v3(
        distributed, &local_batch.ids, &local_batch.targets,
        &local_batch.attention, selected, 2,
        &aggregate_loss, adapter_losses, 2) == 0);
    assert(std::isfinite(aggregate_loss) &&
        std::isfinite(adapter_losses[0]) &&
        std::isfinite(adapter_losses[1]));

    set_reference_environment(local_rank);
    auto tenant_one = tenant_one_batch();
    auto tenant_two = tenant_two_batch();
    const auto reference_one = run_reference(
        model, lora_one, tenant_one, true);
    const auto reference_two = run_reference(
        model, lora_two, tenant_two, true);
    const auto main_only = run_reference(
        model, lora_one, tenant_one, false);
    set_distributed_environment(rank, local_rank);

    const auto distributed_one = adapter_state(distributed, adapter_one);
    const auto distributed_two = adapter_state(distributed, adapter_two);
    const auto unselected_after = adapter_state(
        distributed, adapter_unselected);
    double parameter_diff = 0.0;
    double adam_m_diff = 0.0;
    double adam_v_diff = 0.0;
    for (int index = 0; index < 2; ++index) {
        parameter_diff = std::max({parameter_diff,
            max_diff(distributed_one.parameters[index],
                reference_one.state.parameters[index]),
            max_diff(distributed_two.parameters[index],
                reference_two.state.parameters[index])});
        adam_m_diff = std::max({adam_m_diff,
            max_diff(distributed_one.adam_m[index],
                reference_one.state.adam_m[index]),
            max_diff(distributed_two.adam_m[index],
                reference_two.state.adam_m[index])});
        adam_v_diff = std::max({adam_v_diff,
            max_diff(distributed_one.adam_v[index],
                reference_one.state.adam_v[index]),
            max_diff(distributed_two.adam_v[index],
                reference_two.state.adam_v[index])});
    }
    const double adapter_loss_diff = std::max(
        std::abs(adapter_losses[0] - reference_one.loss),
        std::abs(adapter_losses[1] - reference_two.loss));
    const double aggregate_reference =
        0.5 * (reference_one.loss + reference_two.loss);
    const double aggregate_diff = std::abs(
        aggregate_loss - aggregate_reference);
    const double unselected_diff = state_diff(
        unselected_before, unselected_after);
    const double mtp_loss_effect = std::abs(
        reference_one.loss - main_only.loss);
    double mtp_parameter_effect = 0.0;
    for (int index = 0; index < 2; ++index) {
        mtp_parameter_effect = std::max(mtp_parameter_effect, max_diff(
            reference_one.state.parameters[index],
            main_only.state.parameters[index]));
    }

    std::printf(
        "native_qwen36_mtp_dp rank=%d aggregate_diff=%0.8e "
        "adapter_loss_diff=%0.8e parameter_diff=%0.8e m_diff=%0.8e "
        "v_diff=%0.8e unselected_diff=%0.8e "
        "mtp_loss_effect=%0.8e mtp_parameter_effect=%0.8e "
        "losses=%0.8g,%0.8g\n",
        rank, aggregate_diff, adapter_loss_diff, parameter_diff,
        adam_m_diff, adam_v_diff, unselected_diff,
        mtp_loss_effect, mtp_parameter_effect,
        adapter_losses[0], adapter_losses[1]);
    std::fflush(stdout);

    assert(aggregate_diff < 1e-4 && adapter_loss_diff < 1e-4);
    assert(parameter_diff <= 2e-3);
    assert(adam_m_diff < 5e-5 && adam_v_diff < 5e-8);
    assert(unselected_diff == 0.0);
    assert(qwen36_get_adapter_step_count(distributed, adapter_one) == 1);
    assert(qwen36_get_adapter_step_count(distributed, adapter_two) == 1);
    assert(qwen36_get_adapter_step_count(
        distributed, adapter_unselected) == 0);
    assert(mtp_loss_effect > 1e-6 || mtp_parameter_effect > 0.0);

    qwen36_free_training_context(main_only.context);
    qwen36_free_training_context(reference_two.context);
    qwen36_free_training_context(reference_one.context);
    qwen36_free_training_context(distributed);
    return 0;
}
