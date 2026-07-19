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
extern "C" int32_t qwen36_init_nccl(void*);
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

static constexpr int32_t kBaseTpAttention = 1 << 0;
static constexpr int32_t kBaseTpMlp = 1 << 4;
static constexpr int64_t kTpSize = 2;
static constexpr int64_t kHidden = 8;
static constexpr int64_t kHeads = 4;
static constexpr int64_t kKvHeads = 2;
static constexpr int64_t kHeadDim = 2;
static constexpr int64_t kIntermediate = 12;
static constexpr int64_t kVocab = 16;
static constexpr int64_t kLoraRank = 2;

static at::Tensor values(std::initializer_list<int64_t> shape, double scale,
                         int64_t offset = 0) {
    int64_t count = 1;
    for (const auto dim : shape) count *= dim;
    return (((at::arange(count, at::TensorOptions().device(at::kCUDA)
        .dtype(at::kFloat)) + offset).remainder(23) - 11.0) * scale)
        .reshape(shape).to(at::kBFloat16);
}

static std::vector<void*> pointers(std::vector<at::Tensor>& tensors) {
    std::vector<void*> result;
    result.reserve(tensors.size());
    for (auto& tensor : tensors) result.push_back(&tensor);
    return result;
}

static double max_diff(const at::Tensor& lhs, const at::Tensor& rhs) {
    return (lhs.to(at::kFloat) - rhs.to(at::kFloat))
        .abs().max().item<double>();
}

static std::vector<at::Tensor> make_layer_weights(double scale, int64_t offset) {
    auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    std::vector<at::Tensor> result;
    result.push_back(at::ones({kHidden}, options));
    result.push_back(at::ones({kHidden}, options));
    result.push_back(values({2 * kHeads * kHeadDim, kHidden}, scale, offset));
    result.push_back(at::ones({kHeadDim}, options));
    result.push_back(values({kKvHeads * kHeadDim, kHidden}, scale * 1.2, offset + 3));
    result.push_back(at::ones({kHeadDim}, options));
    result.push_back(values({kKvHeads * kHeadDim, kHidden}, scale * .8, offset + 5));
    result.push_back(values({kHidden, kHeads * kHeadDim}, scale * 1.1, offset + 7));
    result.push_back(values({kIntermediate, kHidden}, scale * .9, offset + 11));
    result.push_back(values({kIntermediate, kHidden}, scale * .7, offset + 13));
    result.push_back(values({kHidden, kIntermediate}, scale, offset + 17));
    for (auto& tensor : result) tensor.set_requires_grad(false);
    return result;
}

static std::vector<at::Tensor> shard_layer_weights(
    const std::vector<at::Tensor>& full, int rank
) {
    const int64_t local_heads = kHeads / kTpSize;
    const int64_t local_kv_heads = kKvHeads / kTpSize;
    const int64_t local_intermediate = kIntermediate / kTpSize;
    return {
        full[0], full[1],
        full[2].narrow(0, rank * 2 * local_heads * kHeadDim,
            2 * local_heads * kHeadDim).contiguous(),
        full[3],
        full[4].narrow(0, rank * local_kv_heads * kHeadDim,
            local_kv_heads * kHeadDim).contiguous(),
        full[5],
        full[6].narrow(0, rank * local_kv_heads * kHeadDim,
            local_kv_heads * kHeadDim).contiguous(),
        full[7].narrow(1, rank * local_heads * kHeadDim,
            local_heads * kHeadDim).contiguous(),
        full[8].narrow(0, rank * local_intermediate,
            local_intermediate).contiguous(),
        full[9].narrow(0, rank * local_intermediate,
            local_intermediate).contiguous(),
        full[10].narrow(1, rank * local_intermediate,
            local_intermediate).contiguous(),
    };
}

struct ModelFixture {
    std::vector<at::Tensor> full_weights;
    std::vector<at::Tensor> full_mtp_weights;
    at::Tensor embed;
    at::Tensor final_norm;
    at::Tensor lm_head;
    at::Tensor mtp_fc;
    at::Tensor mtp_pre_emb;
    at::Tensor mtp_pre_hidden;
    at::Tensor mtp_norm;
    LayerConfig config{};

    ModelFixture()
        : full_weights(make_layer_weights(.010, 0)),
          full_mtp_weights(make_layer_weights(.008, 41)),
          embed(values({kVocab, kHidden}, .020, 19)),
          final_norm(at::ones({kHidden}, at::TensorOptions()
              .device(at::kCUDA).dtype(at::kBFloat16))),
          lm_head(values({kVocab, kHidden}, .015, 29)),
          mtp_fc(values({kHidden, 2 * kHidden}, .006, 37)),
          mtp_pre_emb(at::ones({kHidden}, at::TensorOptions()
              .device(at::kCUDA).dtype(at::kBFloat16))),
          mtp_pre_hidden(at::ones({kHidden}, at::TensorOptions()
              .device(at::kCUDA).dtype(at::kBFloat16))),
          mtp_norm(at::ones({kHidden}, at::TensorOptions()
              .device(at::kCUDA).dtype(at::kBFloat16))) {
        for (auto* tensor : {&embed, &final_norm, &lm_head, &mtp_fc,
                             &mtp_pre_emb, &mtp_pre_hidden, &mtp_norm})
            tensor->set_requires_grad(false);
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

static LoraFixture make_lora(double a_scale, double b_scale, int64_t offset) {
    return {
        values({kLoraRank, kHidden}, a_scale, offset),
        values({2 * kHeads * kHeadDim, kLoraRank}, b_scale, offset + 5),
    };
}

static void install_lora(
    void* context, int64_t adapter, const LoraFixture& fixture,
    int tp_rank, bool sharded
) {
    auto a = fixture.a;
    auto b = sharded
        ? fixture.b.narrow(0,
            tp_rank * fixture.b.size(0) / kTpSize,
            fixture.b.size(0) / kTpSize).contiguous()
        : fixture.b;
    assert(qwen36_set_adapter_lora_tensor(
        context, adapter, 0, "q_proj", 0, &a) == 0);
    assert(qwen36_set_adapter_lora_tensor(
        context, adapter, 0, "q_proj", 1, &b) == 0);
}

static void set_distributed_environment(int rank, int local_rank) {
    setenv("WORLD_SIZE", "2", 1);
    setenv("RANK", std::to_string(rank).c_str(), 1);
    setenv("LOCAL_RANK", std::to_string(local_rank).c_str(), 1);
    setenv("TP_SIZE", "2", 1);
    setenv("CP_SIZE", "1", 1);
    setenv("EP_SIZE", "1", 1);
    setenv("DP_SIZE", "1", 1);
    setenv("PP_SIZE", "1", 1);
    setenv("RUSTRAIN_TP_RANK", std::to_string(rank).c_str(), 1);
    setenv("RUSTRAIN_CP_RANK", "0", 1);
    setenv("RUSTRAIN_EP_RANK", "0", 1);
    setenv("RUSTRAIN_DP_RANK", "0", 1);
    setenv("RUSTRAIN_PP_RANK", "0", 1);
    unsetenv("RUSTRAIN_DATA_PARALLEL");
    unsetenv("QWEN36_SEQUENCE_PARALLEL");
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
    unsetenv("QWEN36_SEQUENCE_PARALLEL");
    unsetenv("QWEN36_DISABLE_MTP");
    setenv("QWEN36_MTP_LOSS_SCALE", "0.4", 1);
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
        result = std::max(result,
            max_diff(lhs.adam_m[index], rhs.adam_m[index]));
        result = std::max(result,
            max_diff(lhs.adam_v[index], rhs.adam_v[index]));
    }
    return result;
}

static void install_mtp(
    void* context, ModelFixture& model, std::vector<at::Tensor>& weights
) {
    auto weight_ptrs = pointers(weights);
    assert(qwen36_set_mtp_weights(
        context, &model.mtp_fc, &model.mtp_pre_emb,
        &model.mtp_pre_hidden, &model.mtp_norm,
        weight_ptrs.data(), weight_ptrs.size(), &model.config, 1) == 0);
}

int main() {
    const int rank = std::atoi(std::getenv("RANK"));
    const int world = std::atoi(std::getenv("WORLD_SIZE"));
    const int local_rank = std::atoi(
        std::getenv("LOCAL_RANK") ? std::getenv("LOCAL_RANK") : "0");
    assert(world == kTpSize && rank >= 0 && rank < world);
    qwen36_set_cuda_device(local_rank);
    set_distributed_environment(rank, local_rank);

    ModelFixture model;
    auto local_weights = shard_layer_weights(model.full_weights, rank);
    auto local_mtp_weights = shard_layer_weights(model.full_mtp_weights, rank);
    auto local_weight_ptrs = pointers(local_weights);
    const int64_t target_layer = 0;
    void* distributed = qwen36_create_training_context_ex(
        local_weight_ptrs.data(), local_weight_ptrs.size(),
        &model.embed, &model.final_norm, &model.lm_head,
        &model.config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, .9, .999, 1e-8, kVocab, 1e-5, kLoraRank,
        &target_layer, 1, "q_proj", kBaseTpAttention | kBaseTpMlp);
    assert(distributed);
    install_mtp(distributed, model, local_mtp_weights);
    assert(qwen36_init_nccl(distributed) == 0);

    const auto lora_one = make_lora(.0020, .0010, 0);
    const auto lora_two = make_lora(.0026, .0007, 17);
    const auto lora_unselected = make_lora(.0008, .0016, 31);
    const int64_t adapter_one = qwen36_add_lora(
        distributed, kLoraRank, 2.0, &target_layer, 1, "q_proj");
    const int64_t adapter_two = qwen36_add_lora(
        distributed, kLoraRank, 2.0, &target_layer, 1, "q_proj");
    const int64_t adapter_unselected = qwen36_add_lora(
        distributed, kLoraRank, 2.0, &target_layer, 1, "q_proj");
    assert(adapter_one > 0 && adapter_two > adapter_one &&
        adapter_unselected > adapter_two);
    install_lora(distributed, adapter_one, lora_one, rank, true);
    install_lora(distributed, adapter_two, lora_two, rank, true);
    install_lora(distributed, adapter_unselected, lora_unselected, rank, true);
    const auto unselected_before = adapter_state(distributed, adapter_unselected);

    auto input_ids = at::tensor({1, 2, 3, 4, 5, 6,
                                 7, 8, 9, 10, 11, 12},
        at::TensorOptions().device(at::kCUDA).dtype(at::kLong)).reshape({2, 6});
    auto target_mask = at::tensor({0., 1., 1., 1., 1., 1.,
                                   0., 1., 1., 0., 0., 0.},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat)).reshape({2, 6});
    auto attention_mask = at::ones({2, 6},
        at::TensorOptions().device(at::kCUDA).dtype(at::kBool));
    const int64_t selected[] = {adapter_one, adapter_two};
    double aggregate = -1.0;
    double losses[2] = {-1.0, -1.0};
    assert(qwen36_train_multi_lora_selected_v3(
        distributed, &input_ids, &target_mask, &attention_mask,
        selected, 2, &aggregate, losses, 2) == 0);

    set_reference_environment(local_rank);
    auto full_weight_ptrs = pointers(model.full_weights);
    void* reference = qwen36_create_training_context(
        full_weight_ptrs.data(), full_weight_ptrs.size(),
        &model.embed, &model.final_norm, &model.lm_head,
        &model.config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, .9, .999, 1e-8, kVocab, 1e-5, kLoraRank,
        &target_layer, 1, "q_proj");
    assert(reference);
    install_mtp(reference, model, model.full_mtp_weights);
    const int64_t reference_one = qwen36_add_lora(
        reference, kLoraRank, 2.0, &target_layer, 1, "q_proj");
    const int64_t reference_two = qwen36_add_lora(
        reference, kLoraRank, 2.0, &target_layer, 1, "q_proj");
    const int64_t reference_unselected = qwen36_add_lora(
        reference, kLoraRank, 2.0, &target_layer, 1, "q_proj");
    assert(reference_one > 0 && reference_two > reference_one &&
        reference_unselected > reference_two);
    install_lora(reference, reference_one, lora_one, 0, false);
    install_lora(reference, reference_two, lora_two, 0, false);
    install_lora(reference, reference_unselected, lora_unselected, 0, false);
    const int64_t reference_selected[] = {reference_one, reference_two};
    double reference_aggregate = -1.0;
    double reference_losses[2] = {-1.0, -1.0};
    assert(qwen36_train_multi_lora_selected_v3(
        reference, &input_ids, &target_mask, &attention_mask,
        reference_selected, 2, &reference_aggregate,
        reference_losses, 2) == 0);

    // A main-loss-only singleton makes a silently skipped MTP branch visible.
    void* main_only = qwen36_create_training_context(
        full_weight_ptrs.data(), full_weight_ptrs.size(),
        &model.embed, &model.final_norm, &model.lm_head,
        &model.config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, .9, .999, 1e-8, kVocab, 1e-5, kLoraRank,
        &target_layer, 1, "q_proj");
    assert(main_only);
    const int64_t main_only_adapter = qwen36_add_lora(
        main_only, kLoraRank, 2.0, &target_layer, 1, "q_proj");
    assert(main_only_adapter > 0);
    install_lora(main_only, main_only_adapter, lora_one, 0, false);
    auto main_only_ids = input_ids.narrow(0, 0, 1).contiguous();
    auto main_only_targets = target_mask.narrow(0, 0, 1).contiguous();
    auto main_only_attention = attention_mask.narrow(0, 0, 1).contiguous();
    double main_only_aggregate = -1.0;
    double main_only_loss = -1.0;
    assert(qwen36_train_multi_lora_selected_v3(
        main_only, &main_only_ids, &main_only_targets, &main_only_attention,
        &main_only_adapter, 1, &main_only_aggregate,
        &main_only_loss, 1) == 0);

    const auto distributed_one = adapter_state(distributed, adapter_one);
    const auto distributed_two = adapter_state(distributed, adapter_two);
    const auto reference_one_state = adapter_state(reference, reference_one);
    const auto reference_two_state = adapter_state(reference, reference_two);
    const auto main_only_state = adapter_state(main_only, main_only_adapter);
    const auto unselected_after = adapter_state(distributed, adapter_unselected);
    double parameter_diff = 0.0;
    double adam_m_diff = 0.0;
    double adam_v_diff = 0.0;
    const int64_t local_q_rows = 2 * (kHeads / kTpSize) * kHeadDim;
    auto compare_state = [&](const AdapterState& actual,
                             const AdapterState& expected) {
        for (int factor = 0; factor < 2; ++factor) {
            auto expected_parameter = factor == 0
                ? expected.parameters[factor]
                : expected.parameters[factor].narrow(
                    0, rank * local_q_rows, local_q_rows);
            auto expected_m = factor == 0
                ? expected.adam_m[factor]
                : expected.adam_m[factor].narrow(
                    0, rank * local_q_rows, local_q_rows);
            auto expected_v = factor == 0
                ? expected.adam_v[factor]
                : expected.adam_v[factor].narrow(
                    0, rank * local_q_rows, local_q_rows);
            parameter_diff = std::max(parameter_diff,
                max_diff(actual.parameters[factor], expected_parameter));
            adam_m_diff = std::max(adam_m_diff,
                max_diff(actual.adam_m[factor], expected_m));
            adam_v_diff = std::max(adam_v_diff,
                max_diff(actual.adam_v[factor], expected_v));
        }
    };
    compare_state(distributed_one, reference_one_state);
    compare_state(distributed_two, reference_two_state);

    const double aggregate_diff = std::abs(aggregate - reference_aggregate);
    const double loss_diff = std::max(
        std::abs(losses[0] - reference_losses[0]),
        std::abs(losses[1] - reference_losses[1]));
    const double unselected_diff = state_diff(
        unselected_before, unselected_after);
    const double mtp_loss_effect = std::abs(
        reference_losses[0] - main_only_loss);
    double mtp_parameter_effect = 0.0;
    for (int factor = 0; factor < 2; ++factor) {
        mtp_parameter_effect = std::max(mtp_parameter_effect, max_diff(
            reference_one_state.parameters[factor],
            main_only_state.parameters[factor]));
    }
    std::printf(
        "native_qwen36_mtp_tp rank=%d aggregate_diff=%0.8e "
        "loss_diff=%0.8e parameter_diff=%0.8e m_diff=%0.8e "
        "v_diff=%0.8e unselected_diff=%0.8e mtp_loss_effect=%0.8e "
        "mtp_parameter_effect=%0.8e losses=%0.8g,%0.8g\n",
        rank, aggregate_diff, loss_diff, parameter_diff, adam_m_diff,
        adam_v_diff, unselected_diff, mtp_loss_effect,
        mtp_parameter_effect, losses[0], losses[1]);
    std::fflush(stdout);

    assert(std::isfinite(aggregate) && std::isfinite(losses[0]) &&
        std::isfinite(losses[1]));
    assert(aggregate_diff < 5e-3 && loss_diff < 5e-3);
    assert(parameter_diff <= 2e-3);
    assert(adam_m_diff < 5e-4 && adam_v_diff < 5e-7);
    assert(unselected_diff == 0.0);
    assert(qwen36_get_adapter_step_count(distributed, adapter_one) == 1);
    assert(qwen36_get_adapter_step_count(distributed, adapter_two) == 1);
    assert(qwen36_get_adapter_step_count(distributed, adapter_unselected) == 0);
    assert(mtp_loss_effect > 1e-6 || mtp_parameter_effect > 0.0);

    qwen36_free_training_context(main_only);
    qwen36_free_training_context(reference);
    qwen36_free_training_context(distributed);
    return 0;
}
