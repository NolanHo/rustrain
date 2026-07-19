#include <ATen/ATen.h>

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

extern "C" int64_t qwen36_kernel_abi_version();
extern "C" void qwen36_set_cuda_device(int32_t);
extern "C" void* qwen36_create_training_context_ex(
    void**, int64_t, void*, void*, void*, void*, int64_t, int32_t,
    double, double, double, double, double, int64_t, double, int64_t,
    const int64_t*, int64_t, const char*, int32_t);
extern "C" int32_t qwen36_init_parallel_nccl(
    void*, int32_t, int32_t, int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
extern "C" int32_t qwen36_set_lora_tensor(void*, int64_t, int32_t, void*);
extern "C" void* qwen36_get_lora_b(void*, int64_t);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" int64_t qwen36_get_sequence_parallel_counter(void*, int32_t);
extern "C" void qwen36_free_training_context(void*);

static at::Tensor values(std::initializer_list<int64_t> shape, double scale) {
    int64_t count = 1;
    for (auto dim : shape) count *= dim;
    return ((at::arange(count, at::TensorOptions().device(at::kCUDA)
        .dtype(at::kFloat)).remainder(23) - 11.0) * scale)
        .reshape(shape).to(at::kBFloat16);
}

static std::vector<void*> ptrs(std::vector<at::Tensor>& tensors) {
    std::vector<void*> result;
    for (auto& tensor : tensors) result.push_back(&tensor);
    return result;
}

int main() {
    assert(qwen36_kernel_abi_version() == 28);
    const int rank = std::atoi(std::getenv("RANK"));
    const int world = std::atoi(std::getenv("WORLD_SIZE"));
    const int local_rank = std::atoi(std::getenv("LOCAL_RANK"));
    assert(world == 2 && (rank == 0 || rank == 1));
    qwen36_set_cuda_device(local_rank);
    setenv("QWEN36_SEQUENCE_PARALLEL", "1", 1);
    setenv("TP_SIZE", "2", 1);
    setenv("CP_SIZE", "1", 1);
    setenv("EP_SIZE", "1", 1);
    setenv("DP_SIZE", "1", 1);
    setenv("PP_SIZE", "1", 1);
    setenv("RUSTRAIN_TP_RANK", std::to_string(rank).c_str(), 1);
    setenv("RUSTRAIN_EP_RANK", "0", 1);
    setenv("RUSTRAIN_DP_RANK", "0", 1);

    constexpr int64_t hidden = 8, heads = 4, kv_heads = 2, head_dim = 2;
    constexpr int64_t intermediate = 12, vocab = 16, rank_lora = 2;
    constexpr int64_t local_heads = heads / 2, local_kv = kv_heads / 2;
    std::vector<at::Tensor> full;
    full.push_back(at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full.push_back(at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full.push_back(values({2 * heads * head_dim, hidden}, .01));
    full.push_back(at::ones({head_dim}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full.push_back(values({kv_heads * head_dim, hidden}, .012));
    full.push_back(at::ones({head_dim}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full.push_back(values({kv_heads * head_dim, hidden}, .008));
    full.push_back(values({hidden, heads * head_dim}, .011));
    full.push_back(values({intermediate, hidden}, .009));
    full.push_back(values({intermediate, hidden}, .007));
    full.push_back(values({hidden, intermediate}, .010));
    std::vector<at::Tensor> local = {
        full[0], full[1], full[2].narrow(0, rank * local_heads * 2 * head_dim,
            local_heads * 2 * head_dim).contiguous(), full[3],
        full[4].narrow(0, rank * local_kv * head_dim, local_kv * head_dim).contiguous(),
        full[5], full[6].narrow(0, rank * local_kv * head_dim, local_kv * head_dim).contiguous(),
        full[7].narrow(1, rank * local_heads * head_dim, local_heads * head_dim).contiguous(),
        full[8].narrow(0, rank * intermediate / 2, intermediate / 2).contiguous(),
        full[9].narrow(0, rank * intermediate / 2, intermediate / 2).contiguous(),
        full[10].narrow(1, rank * intermediate / 2, intermediate / 2).contiguous(),
    };
    auto embed = values({vocab, hidden}, .02);
    auto final_norm = at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
    auto lm_head = values({vocab, hidden}, .015);
    auto local_embed = embed.narrow(0, rank * vocab / 2, vocab / 2).contiguous();
    auto local_lm_head = lm_head.narrow(0, rank * vocab / 2, vocab / 2).contiguous();
    LayerConfig config{};
    config.num_heads = heads; config.num_kv_heads = kv_heads; config.head_dim = head_dim;
    config.partial_rotary_factor = 1.0; config.rope_theta = 10000.0;
    config.rms_eps = 1e-5; config.intermediate_size = intermediate;
    const int64_t target_layer = 0;
    auto local_ptrs = ptrs(local);
    constexpr int32_t flags = (1 << 0) | (1 << 2) | (1 << 4);
    void* ctx = qwen36_create_training_context_ex(
        local_ptrs.data(), local_ptrs.size(), &local_embed, &final_norm,
        &local_lm_head, &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, .9, .999, 1e-8, vocab, 1e-5, rank_lora,
        &target_layer, 1, "q_proj,k_proj,v_proj,o_proj", flags);
    assert(ctx);
    auto q_a = values({rank_lora, hidden}, .002);
    auto q_b = values({local_heads * 2 * head_dim, rank_lora}, .001);
    auto k_a = values({rank_lora, hidden}, .002);
    auto k_b = values({local_kv * head_dim, rank_lora}, .001);
    auto v_a = values({rank_lora, hidden}, .002);
    auto v_b = values({local_kv * head_dim, rank_lora}, .001);
    auto o_a = values({rank_lora, local_heads * head_dim}, .002);
    auto o_b = values({hidden, rank_lora}, .001);
    at::Tensor* factors[] = {&q_a, &q_b, &k_a, &k_b, &v_a, &v_b, &o_a, &o_b};
    for (int64_t slot = 0; slot < 4; ++slot) {
        assert(qwen36_set_lora_tensor(ctx, slot, 0, factors[2 * slot]) == 0);
        assert(qwen36_set_lora_tensor(ctx, slot, 1, factors[2 * slot + 1]) == 0);
    }
    assert(qwen36_init_parallel_nccl(ctx, rank, world, rank, 2, 0,
        0, 1, 0, 0, 1, 0) == 0);
    auto* b_before_ptr = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(ctx, 0));
    assert(b_before_ptr);
    auto b_before = b_before_ptr->clone();
    auto ids = at::tensor({1, 2, 3, 4}, at::TensorOptions().device(at::kCUDA).dtype(at::kLong)).reshape({1, 4});
    auto target = at::ones({1, 4}, at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    auto mask = at::ones({1, 4}, at::TensorOptions().device(at::kCUDA).dtype(at::kBool));
    const double loss = qwen36_train_step(ctx, &ids, &target, &mask);
    assert(std::isfinite(loss) && loss > 0.0);
    auto* b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(ctx, 0));
    assert(b && (*b - b_before).abs().sum().item<double>() > 0.0);
    assert(qwen36_get_sequence_parallel_counter(ctx, 0) == 1);
    assert(qwen36_get_sequence_parallel_counter(ctx, 1) > 0);
    assert(qwen36_get_sequence_parallel_counter(ctx, 2) > 0);
    assert(qwen36_get_sequence_parallel_counter(ctx, 3) == 1);
    std::printf("native_qwen36_sequence_parallel_smoke rank=%d loss=%0.8f ag=%lld rs=%lld local_seq=%lld\n",
        rank, loss, (long long)qwen36_get_sequence_parallel_counter(ctx, 1),
        (long long)qwen36_get_sequence_parallel_counter(ctx, 2),
        (long long)qwen36_get_sequence_parallel_counter(ctx, 4));
    qwen36_free_training_context(ctx);
    return 0;
}
