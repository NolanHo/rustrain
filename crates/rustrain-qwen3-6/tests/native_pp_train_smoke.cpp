#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
extern "C" void* qwen36_create_training_context_v2(
    void**, int64_t, void*, void*, void*, void*, int64_t,
    int64_t, int64_t, int32_t, int32_t,
    double, double, double, double, double, int64_t, double, int64_t,
    const int64_t*, int64_t, const char*, int32_t);
extern "C" int32_t qwen36_init_parallel_nccl_v2(
    void*, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t);
extern "C" double qwen36_train_micro_step(
    void*, void*, void*, void*, double, int32_t);
struct Qwen36PipelineWindowV1 {
    uint32_t struct_size;
    uint32_t version;
    int64_t window_id;
    int64_t num_microbatches;
    int32_t schedule;
    int32_t num_chunks;
    int32_t flags;
};
struct Qwen36PipelineTickV1 {
    uint32_t struct_size;
    uint32_t version;
    int64_t window_id;
    int64_t forward_mb;
    int64_t backward_mb;
    int32_t chunk_id;
    int32_t phase;
    void* input_ids;
    void* target_mask;
    void* attention_mask;
    double gradient_scale;
};
struct Qwen36PipelineResultV1 {
    uint32_t struct_size;
    uint32_t version;
    int32_t status;
    int64_t completed_fwd;
    int64_t completed_bwd;
    int64_t in_flight;
    int64_t optimizer_step;
    double loss;
};
extern "C" int32_t qwen36_pipeline_begin_v1(
    void*, const Qwen36PipelineWindowV1*);
extern "C" int32_t qwen36_pipeline_tick_v1(
    void*, const Qwen36PipelineTickV1*, Qwen36PipelineResultV1*);
extern "C" int32_t qwen36_pipeline_finish_v1(
    void*, int32_t, Qwen36PipelineResultV1*);
extern "C" int32_t qwen36_pipeline_abort_v1(void*);
extern "C" int64_t qwen36_get_lora_count(void*);
extern "C" void* qwen36_get_lora_a(void*, int64_t);
extern "C" void* qwen36_get_lora_b(void*, int64_t);
extern "C" void* qwen36_get_lora_grad_accumulator(
    void*, int64_t, int32_t);
extern "C" int32_t qwen36_set_lora_tensor(
    void*, int64_t, int32_t, void*);
extern "C" int32_t qwen36_abort_gradient_accumulation(void*);
extern "C" int64_t qwen36_export_optimizer_state(
    void*, void**, void**, int64_t);
extern "C" int64_t qwen36_get_step_count(void*);
extern "C" void qwen36_free_training_context(void*);

static at::Tensor cuda_tensor(std::initializer_list<int64_t> shape, double value) {
    auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    return at::full(shape, value, options);
}

static at::Tensor cuda_pattern(
    std::initializer_list<int64_t> shape,
    double scale,
    double offset
) {
    int64_t numel = 1;
    for (const auto dimension : shape) numel *= dimension;
    auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    return at::linspace(-scale, scale, numel, options)
        .add(offset).reshape(shape).to(at::kBFloat16);
}

static at::Tensor& tensor_from_ptr(void* pointer) {
    assert(pointer);
    return *reinterpret_cast<at::Tensor*>(pointer);
}

static double max_diff(const at::Tensor& left, const at::Tensor& right) {
    assert(left.sizes() == right.sizes());
    return left.to(at::kFloat).sub(right.to(at::kFloat))
        .abs().max().item<double>();
}

static double max_abs(const at::Tensor& tensor) {
    return tensor.to(at::kFloat).abs().max().item<double>();
}

struct FixedState {
    std::vector<at::Tensor> lora_a;
    std::vector<at::Tensor> lora_b;
    std::vector<at::Tensor> adam_m;
    std::vector<at::Tensor> adam_v;
    int64_t step = 0;
};

static FixedState snapshot_fixed_state(void* context) {
    FixedState state;
    const int64_t lora_count = qwen36_get_lora_count(context);
    assert(lora_count > 0);
    state.lora_a.reserve(lora_count);
    state.lora_b.reserve(lora_count);
    for (int64_t index = 0; index < lora_count; ++index) {
        state.lora_a.push_back(tensor_from_ptr(
            qwen36_get_lora_a(context, index)).detach().cpu().clone());
        state.lora_b.push_back(tensor_from_ptr(
            qwen36_get_lora_b(context, index)).detach().cpu().clone());
    }

    const int64_t optimizer_count = 2 * lora_count;
    std::vector<void*> m_ptrs(optimizer_count);
    std::vector<void*> v_ptrs(optimizer_count);
    assert(qwen36_export_optimizer_state(
        context, m_ptrs.data(), v_ptrs.data(), optimizer_count) ==
        optimizer_count);
    state.adam_m.reserve(optimizer_count);
    state.adam_v.reserve(optimizer_count);
    for (int64_t index = 0; index < optimizer_count; ++index) {
        state.adam_m.push_back(
            tensor_from_ptr(m_ptrs[index]).detach().cpu().clone());
        state.adam_v.push_back(
            tensor_from_ptr(v_ptrs[index]).detach().cpu().clone());
    }
    state.step = qwen36_get_step_count(context);
    return state;
}

static void copy_fixed_lora(void* source, void* destination) {
    const int64_t count = qwen36_get_lora_count(source);
    assert(count == qwen36_get_lora_count(destination));
    for (int64_t index = 0; index < count; ++index) {
        assert(qwen36_set_lora_tensor(
            destination, index, 0, qwen36_get_lora_a(source, index)) == 0);
        assert(qwen36_set_lora_tensor(
            destination, index, 1, qwen36_get_lora_b(source, index)) == 0);
    }
}

struct StateDiff {
    double parameter = 0.0;
    double adam_m = 0.0;
    double adam_v = 0.0;
};

static StateDiff compare_fixed_state(
    const FixedState& left,
    const FixedState& right
) {
    assert(left.lora_a.size() == right.lora_a.size());
    assert(left.lora_b.size() == right.lora_b.size());
    assert(left.adam_m.size() == right.adam_m.size());
    assert(left.adam_v.size() == right.adam_v.size());
    StateDiff difference;
    for (size_t index = 0; index < left.lora_a.size(); ++index) {
        difference.parameter = std::max(
            difference.parameter,
            max_diff(left.lora_a[index], right.lora_a[index]));
        difference.parameter = std::max(
            difference.parameter,
            max_diff(left.lora_b[index], right.lora_b[index]));
    }
    for (size_t index = 0; index < left.adam_m.size(); ++index) {
        difference.adam_m = std::max(
            difference.adam_m,
            max_diff(left.adam_m[index], right.adam_m[index]));
        difference.adam_v = std::max(
            difference.adam_v,
            max_diff(left.adam_v[index], right.adam_v[index]));
    }
    return difference;
}

int main() {
    assert(qwen36_kernel_abi_version() == 28);
    const int rank = std::atoi(std::getenv("RANK") ? std::getenv("RANK") : "0");
    const int local_rank = std::atoi(
        std::getenv("LOCAL_RANK") ? std::getenv("LOCAL_RANK") : "0");
    assert(rank == 0 || rank == 1);
    c10::cuda::CUDAGuard guard(local_rank);

    const int pp_rank = rank;
    std::vector<at::Tensor> weights = {
        cuda_tensor({4}, 1.0),
        cuda_tensor({4}, 1.0),
        cuda_pattern({8, 4}, 0.08, 0.02),
        cuda_tensor({4}, 1.0),
        cuda_pattern({4, 4}, 0.07, -0.01),
        cuda_tensor({4}, 1.0),
        cuda_pattern({4, 4}, 0.06, 0.015),
        cuda_pattern({4, 4}, 0.08, 0.01),
        cuda_pattern({8, 4}, 0.04, 0.02),
        cuda_pattern({8, 4}, 0.05, -0.01),
        cuda_pattern({4, 8}, 0.04, 0.01),
    };
    for (auto& weight : weights) weight.set_requires_grad(false);
    auto embed = cuda_pattern({4, 4}, 0.20, 0.05);
    auto final_norm = cuda_tensor({4}, 1.0);
    auto lm_head = cuda_pattern({4, 4}, 0.15, -0.025);
    embed.set_requires_grad(false);
    final_norm.set_requires_grad(false);
    lm_head.set_requires_grad(false);

    std::vector<void*> weight_ptrs;
    for (auto& weight : weights) weight_ptrs.push_back(&weight);
    LayerConfig config{};
    config.layer_type = 0;
    config.num_heads = 1;
    config.num_kv_heads = 1;
    config.head_dim = 4;
    config.partial_rotary_factor = 1.0;
    config.rope_theta = 10000.0;
    config.rms_eps = 1e-6;
    config.intermediate_size = 8;
    const int32_t stage_flags = pp_rank == 0 ? 1 : 2;
    const int64_t target_layers[] = {0, 1};
    auto create_context = [&]() {
        void* context = qwen36_create_training_context_v2(
            weight_ptrs.data(), static_cast<int64_t>(weight_ptrs.size()),
            pp_rank == 0 ? &embed : nullptr,
            pp_rank == 1 ? &final_norm : nullptr,
            pp_rank == 1 ? &lm_head : nullptr,
            &config, 1, pp_rank, 2, stage_flags,
            static_cast<int32_t>(at::kBFloat16),
            1.0, 1e-3, 0.9, 0.999, 1e-8, 4, 1e-6, 1,
            target_layers, 2, "q_proj", 0);
        assert(context);
        assert(qwen36_init_parallel_nccl_v2(
            context, rank, 2,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            pp_rank, 2, 0) == 0);
        return context;
    };
    void* window_context = create_context();
    void* legacy_context = create_context();
    copy_fixed_lora(window_context, legacy_context);
    const auto initial_state = snapshot_fixed_state(window_context);
    const auto copied_state = snapshot_fixed_state(legacy_context);
    const auto initial_difference = compare_fixed_state(
        initial_state, copied_state);
    assert(initial_difference.parameter == 0.0);
    assert(initial_difference.adam_m == 0.0);
    assert(initial_difference.adam_v == 0.0);

    const int64_t sequence_length = std::atoll(
        std::getenv("QWEN36_PP_SMOKE_SEQ") ?
            std::getenv("QWEN36_PP_SMOKE_SEQ") : "3");
    assert(sequence_length > 1);
    auto long_options = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
    auto token_positions = at::arange(sequence_length, long_options);
    std::vector<at::Tensor> input_microbatches;
    std::vector<at::Tensor> target_microbatches;
    input_microbatches.reserve(4);
    target_microbatches.reserve(4);
    for (int64_t microbatch = 0; microbatch < 4; ++microbatch) {
        input_microbatches.push_back(
            token_positions.add(microbatch).remainder(4)
                .reshape({1, sequence_length}));
        auto target = at::ones({1, sequence_length}, long_options);
        const int64_t masked_tokens = std::min(
            microbatch, std::max<int64_t>(sequence_length - 2, 0));
        if (masked_tokens > 0) {
            target.narrow(
                1, sequence_length - masked_tokens, masked_tokens).zero_();
        }
        target_microbatches.push_back(std::move(target));
    }
    Qwen36PipelineWindowV1 window{
        sizeof(Qwen36PipelineWindowV1), 1, 7, 4, 0, 1, 0};
    assert(qwen36_pipeline_begin_v1(window_context, &window) == 0);
    int64_t max_in_flight = 0;
    Qwen36PipelineResultV1 result{};
    result.struct_size = sizeof(Qwen36PipelineResultV1);
    result.version = 1;
    const int ticks = pp_rank == 0 ? 5 : 4;
    for (int tick_index = 0; tick_index < ticks; ++tick_index) {
        const int64_t forward_mb = pp_rank == 0
            ? (tick_index < 4 ? tick_index : -1) : tick_index;
        const int64_t backward_mb = pp_rank == 0
            ? (tick_index == 0 ? -1 : tick_index - 1) : tick_index;
        Qwen36PipelineTickV1 tick{
            sizeof(Qwen36PipelineTickV1), 1, 7, forward_mb, backward_mb,
            0, forward_mb < 0 ? 2 : (forward_mb == 0 ? 0 : 1),
            forward_mb >= 0 ? &input_microbatches[forward_mb] : nullptr,
            forward_mb >= 0 ? &target_microbatches[forward_mb] : nullptr,
            nullptr, 0.25};
        assert(qwen36_pipeline_tick_v1(window_context, &tick, &result) == 0);
        max_in_flight = std::max(max_in_flight, result.in_flight);
    }
    assert(max_in_flight == (pp_rank == 0 ? 1 : 0));
    assert(qwen36_get_step_count(window_context) == 0);

    double gradient_difference = 0.0;
    double gradient_magnitude = 0.0;
    double legacy_probe_loss = 0.0;
    for (int64_t microbatch = 0; microbatch < 4; ++microbatch) {
        legacy_probe_loss = qwen36_train_micro_step(
            legacy_context, &input_microbatches[microbatch],
            &target_microbatches[microbatch], nullptr, 0.25, 0);
        assert(std::isfinite(legacy_probe_loss) && legacy_probe_loss >= 0.0);
    }
    const int64_t lora_count = qwen36_get_lora_count(window_context);
    assert(lora_count == qwen36_get_lora_count(legacy_context));
    for (int64_t index = 0; index < lora_count; ++index) {
        for (int32_t is_b = 0; is_b <= 1; ++is_b) {
            auto* window_gradient = reinterpret_cast<at::Tensor*>(
                qwen36_get_lora_grad_accumulator(
                    window_context, index, is_b));
            auto* legacy_gradient = reinterpret_cast<at::Tensor*>(
                qwen36_get_lora_grad_accumulator(
                    legacy_context, index, is_b));
            assert((window_gradient == nullptr) == (legacy_gradient == nullptr));
            if (!window_gradient) continue;
            gradient_difference = std::max(
                gradient_difference,
                max_diff(*window_gradient, *legacy_gradient));
            gradient_magnitude = std::max(
                gradient_magnitude, max_abs(*window_gradient));
        }
    }
    assert(gradient_magnitude > 1e-8);
    assert(gradient_difference <= 1e-6);
    assert(qwen36_abort_gradient_accumulation(legacy_context) == 0);
    assert(qwen36_get_step_count(legacy_context) == 0);

    assert(qwen36_pipeline_finish_v1(window_context, 1, &result) == 0);
    assert(result.completed_fwd == 4 && result.completed_bwd == 4);
    assert(result.in_flight == 0 && std::isfinite(result.loss));
    assert(qwen36_get_step_count(window_context) == 1);
    const double window_loss = result.loss;

    double legacy_loss_sum = 0.0;
    for (int64_t microbatch = 0; microbatch < 4; ++microbatch) {
        const double legacy_microbatch_loss = qwen36_train_micro_step(
            legacy_context, &input_microbatches[microbatch],
            &target_microbatches[microbatch], nullptr, 0.25,
            microbatch == 3 ? 1 : 0);
        assert(std::isfinite(legacy_microbatch_loss) &&
            legacy_microbatch_loss >= 0.0);
        legacy_loss_sum += legacy_microbatch_loss;
    }
    assert(qwen36_get_step_count(legacy_context) == 1);
    const double legacy_loss = legacy_loss_sum / 4.0;
    assert(std::abs(window_loss - legacy_loss) <= 1e-5);

    const auto window_state = snapshot_fixed_state(window_context);
    const auto legacy_state = snapshot_fixed_state(legacy_context);
    const auto parity = compare_fixed_state(window_state, legacy_state);
    assert(window_state.step == 1 && legacy_state.step == 1);
    assert(parity.parameter <= 2e-3);
    assert(parity.adam_m <= 1e-5);
    assert(parity.adam_v <= 1e-7);
    const auto update = compare_fixed_state(initial_state, window_state);
    assert(update.parameter > 0.0);
    assert(std::max(update.adam_m, update.adam_v) > 0.0);

    auto bad_target_mask = at::ones(
        {1, sequence_length - 1},
        at::TensorOptions().device(at::kCUDA).dtype(at::kLong));
    // Deliberately make only rank 0 violate the next window's shape contract.
    // Both ranks must fail at the PP min/max preflight, before any NCCL P2P
    // count can diverge.
    auto* bad_target_ptr = rank == 0
        ? &bad_target_mask : &target_microbatches[0];
    Qwen36PipelineWindowV1 bad_window{
        sizeof(Qwen36PipelineWindowV1), 1, 8, 1, 0, 1, 0};
    assert(qwen36_pipeline_begin_v1(window_context, &bad_window) == 0);
    Qwen36PipelineTickV1 bad_tick{
        sizeof(Qwen36PipelineTickV1), 1, 8, 0, pp_rank == 0 ? -1 : 0, 0, 0,
        &input_microbatches[0], bad_target_ptr, nullptr, 1.0};
    assert(qwen36_pipeline_tick_v1(window_context, &bad_tick, &result) != 0);
    std::printf("native_qwen36_pp_train rank=%d loss=%0.6f "
        "grad_diff=%0.8e param_diff=%0.8e m_diff=%0.8e v_diff=%0.8e "
        "max_in_flight=%ld step=1 ok\n", rank,
        window_loss, gradient_difference, parity.parameter, parity.adam_m,
        parity.adam_v, (long)max_in_flight);
    qwen36_free_training_context(legacy_context);
    qwen36_free_training_context(window_context);
    return 0;
}
