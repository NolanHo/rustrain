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
extern "C" int64_t qwen36_get_step_count(void*);
extern "C" void qwen36_free_training_context(void*);

static at::Tensor cuda_tensor(std::initializer_list<int64_t> shape, double value) {
    auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    return at::full(shape, value, options);
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
        cuda_tensor({8, 4}, 0.0),
        cuda_tensor({4}, 1.0),
        cuda_tensor({4, 4}, 0.0),
        cuda_tensor({4}, 1.0),
        cuda_tensor({4, 4}, 0.0),
        cuda_tensor({4, 4}, 0.0),
        cuda_tensor({8, 4}, 0.0),
        cuda_tensor({8, 4}, 0.0),
        cuda_tensor({4, 8}, 0.0),
    };
    for (auto& weight : weights) weight.set_requires_grad(false);
    auto embed = cuda_tensor({4, 4}, 0.0);
    auto final_norm = cuda_tensor({4}, 1.0);
    auto lm_head = cuda_tensor({4, 4}, 0.0);
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
    const int64_t target_layer = 0;
    void* context = qwen36_create_training_context_v2(
        weight_ptrs.data(), static_cast<int64_t>(weight_ptrs.size()),
        pp_rank == 0 ? &embed : nullptr,
        pp_rank == 1 ? &final_norm : nullptr,
        pp_rank == 1 ? &lm_head : nullptr,
        &config, 1, pp_rank, 2, stage_flags,
        static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, 4, 1e-6, 1,
        &target_layer, 1, "q_proj", 0);
    assert(context);
    assert(qwen36_init_parallel_nccl_v2(
        context, rank, 2,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        pp_rank, 2, 0) == 0);

    const int64_t sequence_length = std::atoll(
        std::getenv("QWEN36_PP_SMOKE_SEQ") ?
            std::getenv("QWEN36_PP_SMOKE_SEQ") : "3");
    assert(sequence_length > 1);
    auto input_ids = at::arange(
        sequence_length, at::TensorOptions().device(at::kCUDA).dtype(at::kLong))
        .remainder(4).reshape({1, sequence_length});
    auto target_mask = at::ones(
        {1, sequence_length},
        at::TensorOptions().device(at::kCUDA).dtype(at::kLong));
    Qwen36PipelineWindowV1 window{
        sizeof(Qwen36PipelineWindowV1), 1, 7, 4, 0, 1, 0};
    assert(qwen36_pipeline_begin_v1(context, &window) == 0);
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
            forward_mb >= 0 ? &input_ids : nullptr,
            forward_mb >= 0 ? &target_mask : nullptr,
            nullptr, 0.25};
        assert(qwen36_pipeline_tick_v1(context, &tick, &result) == 0);
        max_in_flight = std::max(max_in_flight, result.in_flight);
    }
    assert(max_in_flight == (pp_rank == 0 ? 1 : 0));
    assert(qwen36_get_step_count(context) == 0);
    assert(qwen36_pipeline_finish_v1(context, 1, &result) == 0);
    assert(result.completed_fwd == 4 && result.completed_bwd == 4);
    assert(result.in_flight == 0 && std::isfinite(result.loss));
    assert(qwen36_get_step_count(context) == 1);
    auto bad_target_mask = at::ones(
        {1, sequence_length - 1},
        at::TensorOptions().device(at::kCUDA).dtype(at::kLong));
    Qwen36PipelineWindowV1 bad_window{
        sizeof(Qwen36PipelineWindowV1), 1, 8, 1, 0, 1, 0};
    assert(qwen36_pipeline_begin_v1(context, &bad_window) == 0);
    Qwen36PipelineTickV1 bad_tick{
        sizeof(Qwen36PipelineTickV1), 1, 8, 0, pp_rank == 0 ? -1 : 0, 0, 0,
        &input_ids, &bad_target_mask, nullptr, 1.0};
    assert(qwen36_pipeline_tick_v1(context, &bad_tick, &result) != 0);
    std::printf("native_qwen36_pp_train rank=%d loss=%0.6f "
        "fwd=%ld bwd=%ld max_in_flight=%ld step=1 ok\n", rank,
        result.loss, (long)result.completed_fwd, (long)result.completed_bwd,
        (long)max_in_flight);
    qwen36_free_training_context(context);
    return 0;
}
