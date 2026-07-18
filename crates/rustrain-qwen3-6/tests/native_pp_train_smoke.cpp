#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

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
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" int64_t qwen36_get_step_count(void*);
extern "C" void qwen36_free_training_context(void*);

static at::Tensor cuda_tensor(std::initializer_list<int64_t> shape, double value) {
    auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    return at::full(shape, value, options);
}

int main() {
    assert(qwen36_kernel_abi_version() == 27);
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
    void* context = qwen36_create_training_context_v2(
        weight_ptrs.data(), static_cast<int64_t>(weight_ptrs.size()),
        pp_rank == 0 ? &embed : nullptr,
        pp_rank == 1 ? &final_norm : nullptr,
        pp_rank == 1 ? &lm_head : nullptr,
        &config, 1, pp_rank, 2, stage_flags,
        static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, 4, 1e-6, 1,
        nullptr, 0, "q_proj", 0);
    assert(context);
    assert(qwen36_init_parallel_nccl_v2(
        context, rank, 2,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        pp_rank, 2, 0) == 0);

    auto input_ids = at::arange(
        1, 4, at::TensorOptions().device(at::kCUDA).dtype(at::kLong))
        .reshape({1, 3});
    auto target_mask = at::ones(
        {1, 3}, at::TensorOptions().device(at::kCUDA).dtype(at::kLong));
    const double loss = qwen36_train_step(
        context, &input_ids, &target_mask, nullptr);
    assert(std::isfinite(loss) && loss >= 0.0);
    assert(qwen36_get_step_count(context) == 1);
    std::printf("native_qwen36_pp_train rank=%d loss=%0.6f step=1 ok\n",
        rank, loss);
    qwen36_free_training_context(context);
    return 0;
}
