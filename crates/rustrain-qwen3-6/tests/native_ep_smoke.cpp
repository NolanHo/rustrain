#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <cassert>
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

extern "C" void* qwen36_create_training_context(
    void**, int64_t, void*, void*, void*, void*, int64_t, int32_t,
    double, double, double, double, double, int64_t, double, int64_t,
    const int64_t*, int64_t, const char*);
extern "C" int32_t qwen36_init_nccl(void*);
extern "C" void* qwen36_get_lora_b(void*, int64_t);
extern "C" int32_t qwen36_set_lora_tensor(void*, int64_t, int32_t, void*);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" void qwen36_free_training_context(void*);

static at::Tensor cuda_rand(std::initializer_list<int64_t> shape) {
    return at::randn(shape, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
}

int main() {
    const int rank = std::atoi(std::getenv("RANK") ? std::getenv("RANK") : "0");
    const int world = std::atoi(std::getenv("WORLD_SIZE") ? std::getenv("WORLD_SIZE") : "1");
    const int local_rank = std::atoi(std::getenv("LOCAL_RANK") ? std::getenv("LOCAL_RANK") : "0");
    assert(world == 2 && rank >= 0 && rank < world);
    c10::cuda::CUDAGuard guard(local_rank);
    // Replicated weights must be identical across EP ranks.
    at::manual_seed(100);

    constexpr int64_t hidden = 16;
    constexpr int64_t vocab = 8;
    constexpr int64_t experts = 2;
    constexpr int64_t head_dim = 8;
    constexpr int64_t intermediate = 8;
    constexpr int64_t rank_lora = 4;
    std::vector<at::Tensor> weights;
    weights.push_back(cuda_rand({hidden}));
    weights.push_back(cuda_rand({hidden}));
    weights.push_back(cuda_rand({2 * head_dim, hidden}));
    weights.push_back(cuda_rand({head_dim}));
    weights.push_back(cuda_rand({head_dim, hidden}));
    weights.push_back(cuda_rand({head_dim}));
    weights.push_back(cuda_rand({head_dim, hidden}));
    weights.push_back(cuda_rand({hidden, head_dim}));
    weights.push_back(cuda_rand({experts, hidden}));
    weights.push_back(cuda_rand({1, hidden}));
    weights.push_back(cuda_rand({intermediate, hidden}));
    weights.push_back(cuda_rand({intermediate, hidden}));
    weights.push_back(cuda_rand({hidden, intermediate}));
    // Each process owns exactly one expert row.
    weights.push_back(cuda_rand({1, 2 * intermediate, hidden}));
    weights.push_back(cuda_rand({1, hidden, intermediate}));
    for (auto& weight : weights) weight.set_requires_grad(false);
    auto embed = cuda_rand({vocab, hidden});
    auto final_norm = cuda_rand({hidden});
    auto lm_head = cuda_rand({vocab, hidden});
    embed.set_requires_grad(false);
    final_norm.set_requires_grad(false);
    lm_head.set_requires_grad(false);

    std::vector<void*> weight_ptrs;
    for (auto& weight : weights) weight_ptrs.push_back(&weight);
    LayerConfig config{};
    config.layer_type = 0;
    config.num_heads = 1;
    config.num_kv_heads = 1;
    config.head_dim = head_dim;
    config.partial_rotary_factor = 1.0;
    config.rope_theta = 10000.0;
    config.rms_eps = 1e-5;
    config.num_experts = experts;
    // Route every token to both experts so both local shards exercise their
    // expert LoRA optimizer path in this two-rank smoke.
    config.top_k = 2;
    config.moe_intermediate = intermediate;
    config.expert_start = rank;
    config.expert_count = 1;
    config.norm_topk_prob = 1;

    const int64_t target_layer = 0;
    void* ctx = qwen36_create_training_context(
        weight_ptrs.data(), static_cast<int64_t>(weight_ptrs.size()),
        &embed, &final_norm, &lm_head, &config, 1,
        static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, rank_lora,
        &target_layer, 1, "experts_gate_up_proj,experts_down_proj");
    assert(ctx);
    assert(qwen36_init_nccl(ctx) == 0);
    auto* lora_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(ctx, 7));
    assert(lora_b);
    auto lora_b_value = at::ones(lora_b->sizes(), lora_b->options());
    assert(qwen36_set_lora_tensor(ctx, 7, 1, &lora_b_value) == 0);
    auto before = lora_b->clone();

    auto input_ids = at::tensor({1, 2},
        at::TensorOptions().device(at::kCUDA).dtype(at::kLong)).reshape({1, 2});
    auto target_mask = at::ones({1, 2},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    auto attention_mask = at::ones({1, 2},
        at::TensorOptions().device(at::kCUDA).dtype(at::kBool));
    const double loss = qwen36_train_step(ctx, &input_ids, &target_mask, &attention_mask);
    c10::cuda::device_synchronize();
    const double update = (*lora_b - before).abs().sum().item<double>();
    std::printf("native_qwen36_ep_smoke rank=%d world=%d loss=%0.8f lora_b_update=%0.8e\n",
        rank, world, loss, update);
    assert(loss == loss && loss > 0.0);
    assert(update > 0.0);
    qwen36_free_training_context(ctx);
    return 0;
}
