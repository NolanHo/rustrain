#include <ATen/ATen.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
extern "C" int32_t qwen36_init_parallel_nccl_v2(
    void*, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t);
extern "C" int64_t qwen36_add_lora(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" void* qwen36_get_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t);
extern "C" int32_t qwen36_set_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t, void*);
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

static at::Tensor deterministic(std::initializer_list<int64_t> shape, double scale) {
    int64_t count = 1;
    for (int64_t dim : shape) count *= dim;
    return ((at::arange(count,
                 at::TensorOptions().device(at::kCUDA).dtype(at::kFloat))
                 .remainder(19) - 9.0) * scale)
        .reshape(shape).to(at::kBFloat16);
}

static std::vector<void*> pointers(std::vector<at::Tensor>& tensors) {
    std::vector<void*> result;
    result.reserve(tensors.size());
    for (auto& tensor : tensors) result.push_back(&tensor);
    return result;
}

static double max_diff(const at::Tensor& lhs, const at::Tensor& rhs) {
    return (lhs - rhs).abs().max().item<double>();
}

int main() {
    const int rank = std::atoi(std::getenv("RANK"));
    const int world = std::atoi(std::getenv("WORLD_SIZE"));
    assert(world == 2 && rank >= 0 && rank < world);
    qwen36_set_cuda_device(rank);
    auto set_distributed_env = [&]() {
        setenv("WORLD_SIZE", std::to_string(world).c_str(), 1);
        setenv("RANK", std::to_string(rank).c_str(), 1);
        setenv("TP_SIZE", "1", 1);
        setenv("CP_SIZE", "2", 1);
        setenv("EP_SIZE", "1", 1);
        setenv("DP_SIZE", "1", 1);
        setenv("PP_SIZE", "1", 1);
        setenv("RUSTRAIN_TP_RANK", "0", 1);
        setenv("RUSTRAIN_CP_RANK", std::to_string(rank).c_str(), 1);
        setenv("QWEN36_CP_FULL_ATTENTION_KV_GATHER", "1", 1);
    };
    auto set_reference_env = []() {
        setenv("WORLD_SIZE", "1", 1);
        setenv("RANK", "0", 1);
        setenv("TP_SIZE", "1", 1);
        setenv("CP_SIZE", "1", 1);
        setenv("EP_SIZE", "1", 1);
        setenv("DP_SIZE", "1", 1);
        setenv("PP_SIZE", "1", 1);
        setenv("RUSTRAIN_TP_RANK", "0", 1);
        setenv("RUSTRAIN_CP_RANK", "0", 1);
        setenv("QWEN36_CP_FULL_ATTENTION_KV_GATHER", "1", 1);
    };

    constexpr int64_t hidden = 8;
    constexpr int64_t heads = 4;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t head_dim = 2;
    constexpr int64_t intermediate = 12;
    constexpr int64_t vocab = 16;
    constexpr int64_t lora_rank = 4;

    std::vector<at::Tensor> full_weights;
    full_weights.push_back(at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full_weights.push_back(at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full_weights.push_back(deterministic({2 * heads * head_dim, hidden}, 0.010));
    full_weights.push_back(at::ones({head_dim}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full_weights.push_back(deterministic({kv_heads * head_dim, hidden}, 0.012));
    full_weights.push_back(at::ones({head_dim}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full_weights.push_back(deterministic({kv_heads * head_dim, hidden}, 0.008));
    full_weights.push_back(deterministic({hidden, heads * head_dim}, 0.011));
    full_weights.push_back(deterministic({intermediate, hidden}, 0.009));
    full_weights.push_back(deterministic({intermediate, hidden}, 0.007));
    full_weights.push_back(deterministic({hidden, intermediate}, 0.010));
    for (auto& weight : full_weights) weight.set_requires_grad(false);

    std::vector<at::Tensor> local_weights = full_weights;

    auto embed = deterministic({vocab, hidden}, 0.020);
    auto final_norm = at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
    auto lm_head = deterministic({vocab, hidden}, 0.015);
    LayerConfig config{};
    config.layer_type = 0;
    config.num_heads = heads;
    config.num_kv_heads = kv_heads;
    config.head_dim = head_dim;
    config.partial_rotary_factor = 1.0;
    config.rope_theta = 10000.0;
    config.rms_eps = 1e-5;
    config.intermediate_size = intermediate;

    const int64_t target_layer = 0;
    auto local_ptrs = pointers(local_weights);
    set_distributed_env();
    unsetenv("RUSTRAIN_DATA_PARALLEL");
    void* distributed = qwen36_create_training_context_ex(
        local_ptrs.data(), local_ptrs.size(), &embed, &final_norm, &lm_head,
        &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, lora_rank,
        &target_layer, 1, "q_proj,k_proj,v_proj,o_proj", 0);
    assert(distributed);
    assert(qwen36_init_parallel_nccl_v2(
        distributed, rank, world,
        0, 1, 0,
        rank, 2, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0) == 0);

    set_reference_env();
    auto full_ptrs = pointers(full_weights);
    void* reference = qwen36_create_training_context(
        full_ptrs.data(), full_ptrs.size(), &embed, &final_norm, &lm_head,
        &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, lora_rank,
        &target_layer, 1, "q_proj,k_proj,v_proj,o_proj");
    assert(reference);
    set_distributed_env();

    auto q_a = deterministic({lora_rank, hidden}, 0.0020);
    auto q_b = deterministic({2 * heads * head_dim, lora_rank}, 0.0010);
    auto k_a = deterministic({lora_rank, hidden}, 0.0018);
    auto k_b = deterministic({kv_heads * head_dim, lora_rank}, 0.0011);
    auto v_a = deterministic({lora_rank, hidden}, 0.0016);
    auto v_b = deterministic({kv_heads * head_dim, lora_rank}, 0.0009);
    auto o_a = deterministic({lora_rank, heads * head_dim}, 0.0015);
    auto o_b = deterministic({hidden, lora_rank}, 0.0012);
    auto local_q_b = q_b.clone();
    auto local_o_a = o_a.clone();
    auto local_k_b = k_b.clone();
    auto local_v_b = v_b.clone();
    assert(qwen36_set_lora_tensor(distributed, 0, 0, &q_a) == 0);
    assert(qwen36_set_lora_tensor(distributed, 0, 1, &local_q_b) == 0);
    assert(qwen36_set_lora_tensor(distributed, 1, 0, &k_a) == 0);
    assert(qwen36_set_lora_tensor(distributed, 1, 1, &local_k_b) == 0);
    assert(qwen36_set_lora_tensor(distributed, 2, 0, &v_a) == 0);
    assert(qwen36_set_lora_tensor(distributed, 2, 1, &local_v_b) == 0);
    assert(qwen36_set_lora_tensor(distributed, 3, 0, &local_o_a) == 0);
    assert(qwen36_set_lora_tensor(distributed, 3, 1, &o_b) == 0);
    assert(qwen36_set_lora_tensor(reference, 0, 0, &q_a) == 0);
    assert(qwen36_set_lora_tensor(reference, 0, 1, &q_b) == 0);
    assert(qwen36_set_lora_tensor(reference, 1, 0, &k_a) == 0);
    assert(qwen36_set_lora_tensor(reference, 1, 1, &k_b) == 0);
    assert(qwen36_set_lora_tensor(reference, 2, 0, &v_a) == 0);
    assert(qwen36_set_lora_tensor(reference, 2, 1, &v_b) == 0);
    assert(qwen36_set_lora_tensor(reference, 3, 0, &o_a) == 0);
    assert(qwen36_set_lora_tensor(reference, 3, 1, &o_b) == 0);

    auto input_ids = at::tensor({
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 0, 0},
        at::TensorOptions().device(at::kCUDA).dtype(at::kLong)).reshape({2, 8});
    auto target_mask = at::tensor({
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat)).reshape({2, 8});
    auto attention_mask = target_mask.to(at::kBool);

    const double distributed_eval = qwen36_eval_step(
        distributed, &input_ids, &target_mask, &attention_mask);
    const double reference_eval = qwen36_eval_step(
        reference, &input_ids, &target_mask, &attention_mask);
    assert(distributed_eval > 0.0 && reference_eval > 0.0);

    const double distributed_micro = qwen36_train_micro_step(
        distributed, &input_ids, &target_mask, &attention_mask, 1.0, 0);
    const double reference_micro = qwen36_train_micro_step(
        reference, &input_ids, &target_mask, &attention_mask, 1.0, 0);
    assert(distributed_micro > 0.0 && reference_micro > 0.0);
    auto* local_q_b_grad = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_grad_accumulator(distributed, 0, 1));
    auto* full_q_b_grad = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_grad_accumulator(reference, 0, 1));
    auto* local_k_b_grad = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_grad_accumulator(distributed, 1, 1));
    auto* full_k_b_grad = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_grad_accumulator(reference, 1, 1));
    auto* local_v_b_grad = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_grad_accumulator(distributed, 2, 1));
    auto* full_v_b_grad = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_grad_accumulator(reference, 2, 1));
    auto* local_o_a_grad = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_grad_accumulator(distributed, 3, 0));
    auto* full_o_a_grad = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_grad_accumulator(reference, 3, 0));
    assert(local_q_b_grad && full_q_b_grad && local_k_b_grad && full_k_b_grad);
    assert(local_v_b_grad && full_v_b_grad && local_o_a_grad && full_o_a_grad);
    const double q_b_grad_diff = max_diff(*local_q_b_grad, *full_q_b_grad);
    const double o_a_grad_diff = max_diff(*local_o_a_grad, *full_o_a_grad);
    const double k_b_grad_diff = max_diff(*local_k_b_grad, *full_k_b_grad);
    const double v_b_grad_diff = max_diff(*local_v_b_grad, *full_v_b_grad);
    assert(qwen36_abort_gradient_accumulation(distributed) == 0);
    assert(qwen36_abort_gradient_accumulation(reference) == 0);

    const double distributed_loss = qwen36_train_step(
        distributed, &input_ids, &target_mask, &attention_mask);
    const double reference_loss = qwen36_train_step(
        reference, &input_ids, &target_mask, &attention_mask);
    assert(distributed_loss > 0.0 && reference_loss > 0.0);

    auto* updated_q_a = reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(distributed, 0));
    auto* updated_q_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(distributed, 0));
    auto* updated_k_a = reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(distributed, 1));
    auto* updated_k_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(distributed, 1));
    auto* updated_v_a = reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(distributed, 2));
    auto* updated_v_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(distributed, 2));
    auto* updated_o_a = reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(distributed, 3));
    auto* updated_o_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(distributed, 3));
    auto* reference_q_a = reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(reference, 0));
    auto* reference_q_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(reference, 0));
    auto* reference_k_a = reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(reference, 1));
    auto* reference_k_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(reference, 1));
    auto* reference_v_a = reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(reference, 2));
    auto* reference_v_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(reference, 2));
    auto* reference_o_a = reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(reference, 3));
    auto* reference_o_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(reference, 3));
    assert(updated_q_a && updated_q_b && updated_k_a && updated_k_b);
    assert(updated_v_a && updated_v_b && updated_o_a && updated_o_b);
    assert(reference_q_a && reference_q_b && reference_k_a && reference_k_b);
    assert(reference_v_a && reference_v_b && reference_o_a && reference_o_b);

    constexpr int64_t optimizer_count = 14;
    std::vector<void*> local_m(optimizer_count), local_v(optimizer_count);
    std::vector<void*> full_m(optimizer_count), full_v(optimizer_count);
    assert(qwen36_export_optimizer_state(
        distributed, local_m.data(), local_v.data(), optimizer_count) ==
        optimizer_count);
    assert(qwen36_export_optimizer_state(
        reference, full_m.data(), full_v.data(), optimizer_count) ==
        optimizer_count);
    double optimizer_m_diff = 0.0;
    double optimizer_v_diff = 0.0;
    double adam_error = 0.0;
    auto observe_optimizer = [&](int64_t index, const at::Tensor& expected_m,
                                 const at::Tensor& expected_v,
                                 const at::Tensor& updated,
                                 const at::Tensor& before) {
        auto* m = reinterpret_cast<at::Tensor*>(local_m[index]);
        auto* v = reinterpret_cast<at::Tensor*>(local_v[index]);
        assert(m && v);
        optimizer_m_diff = std::max(optimizer_m_diff, max_diff(*m, expected_m));
        optimizer_v_diff = std::max(optimizer_v_diff, max_diff(*v, expected_v));
        auto expected_param = (
            before.to(at::kFloat) - 1e-3 *
                (*m / (1.0 - 0.9)) /
                (((*v / (1.0 - 0.999)).sqrt()) + 1e-8))
            .to(at::kBFloat16);
        adam_error = std::max(adam_error, max_diff(updated, expected_param));
    };
    auto state = [](std::vector<void*>& tensors, int64_t index) -> at::Tensor& {
        auto* tensor = reinterpret_cast<at::Tensor*>(tensors[index]);
        assert(tensor);
        return *tensor;
    };
    observe_optimizer(0, state(full_m, 0), state(full_v, 0), *updated_q_a, q_a);
    observe_optimizer(1, state(full_m, 1), state(full_v, 1), *updated_q_b, local_q_b);
    observe_optimizer(2, state(full_m, 2), state(full_v, 2), *updated_k_a, k_a);
    observe_optimizer(3, state(full_m, 3), state(full_v, 3), *updated_k_b, local_k_b);
    observe_optimizer(4, state(full_m, 4), state(full_v, 4), *updated_v_a, v_a);
    observe_optimizer(5, state(full_m, 5), state(full_v, 5), *updated_v_b, local_v_b);
    observe_optimizer(6, state(full_m, 6), state(full_v, 6), *updated_o_a, local_o_a);
    observe_optimizer(7, state(full_m, 7), state(full_v, 7), *updated_o_b, o_b);
    const double q_a_diff = max_diff(*updated_q_a, *reference_q_a);
    const double q_b_diff = max_diff(*updated_q_b, *reference_q_b);
    const double o_a_diff = max_diff(*updated_o_a, *reference_o_a);
    const double o_b_diff = max_diff(*updated_o_b, *reference_o_b);
    const double k_a_diff = max_diff(*updated_k_a, *reference_k_a);
    const double k_b_diff = max_diff(*updated_k_b, *reference_k_b);
    const double v_a_diff = max_diff(*updated_v_a, *reference_v_a);
    const double v_b_diff = max_diff(*updated_v_b, *reference_v_b);
    std::printf(
        "base_cp_attention_smoke rank=%d eval_diff=%0.8e loss_diff=%0.8e q_b_grad_diff=%0.8e k_b_grad_diff=%0.8e v_b_grad_diff=%0.8e o_a_grad_diff=%0.8e m_diff=%0.8e v_diff=%0.8e adam_error=%0.8e q_a_diff=%0.8e q_b_diff=%0.8e k_a_diff=%0.8e k_b_diff=%0.8e v_a_diff=%0.8e v_b_diff=%0.8e o_a_diff=%0.8e o_b_diff=%0.8e\n",
        rank, std::abs(distributed_eval - reference_eval),
        std::abs(distributed_loss - reference_loss), q_b_grad_diff,
        k_b_grad_diff, v_b_grad_diff, o_a_grad_diff, optimizer_m_diff,
        optimizer_v_diff, adam_error, q_a_diff, q_b_diff, k_a_diff,
        k_b_diff, v_a_diff, v_b_diff, o_a_diff, o_b_diff);
    std::fflush(stdout);
    // The CP path shards token gradients before reducing LoRA gradients.
    assert(std::abs(distributed_eval - reference_eval) < 5e-3);
    assert(std::abs(distributed_loss - reference_loss) < 5e-3);
    // Micro-step accumulators are rank-local until the optimizer boundary.
    // Their non-zero K/V values exercise the gather backward path; the
    // synchronized Adam moments below are the CP2-versus-CP1 oracle.
    assert(std::isfinite(q_b_grad_diff) && std::isfinite(k_b_grad_diff));
    assert(std::isfinite(v_b_grad_diff) && std::isfinite(o_a_grad_diff));
    assert(optimizer_m_diff < 5e-5 && optimizer_v_diff < 5e-8);
    assert(adam_error < 1e-8);
    assert(std::max({q_a_diff, q_b_diff, k_a_diff, k_b_diff,
        v_a_diff, v_b_diff, o_a_diff, o_b_diff}) <= 2e-3);

    set_distributed_env();
    if (rank == 1) setenv("QWEN36_CP_FULL_ATTENTION_KV_GATHER", "0", 1);
    const double flag_mismatch = qwen36_eval_step(
        distributed, &input_ids, &target_mask, &attention_mask);
    assert(flag_mismatch < 0.0);
    setenv("QWEN36_CP_FULL_ATTENTION_KV_GATHER", "1", 1);

    qwen36_free_training_context(reference);
    qwen36_free_training_context(distributed);

    std::printf("cp_full_attention_kv_gather_smoke rank=%d flag_mismatch=ok\n", rank);
    std::fflush(stdout);
    return 0;
}
