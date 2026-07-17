#include <ATen/ATen.h>

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
extern "C" void* qwen36_get_lora_a(void*, int64_t);
extern "C" void* qwen36_get_lora_b(void*, int64_t);
extern "C" int32_t qwen36_set_lora_tensor(void*, int64_t, int32_t, void*);
extern "C" void* qwen36_get_lora_grad_accumulator(void*, int64_t, int32_t);
extern "C" int32_t qwen36_abort_gradient_accumulation(void*);
extern "C" double qwen36_eval_step(void*, void*, void*, void*);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" double qwen36_train_micro_step(
    void*, void*, void*, void*, double, int32_t);
extern "C" int64_t qwen36_export_optimizer_state(
    void*, void**, void**, int64_t);
extern "C" void qwen36_free_training_context(void*);

static at::Tensor deterministic(
    std::initializer_list<int64_t> shape, double scale, int64_t offset = 0
) {
    int64_t count = 1;
    for (int64_t dim : shape) count *= dim;
    return ((at::arange(count,
                 at::TensorOptions().device(at::kCUDA).dtype(at::kFloat))
                 .add(offset).remainder(23) - 11.0) * scale)
        .reshape(shape).to(at::kBFloat16);
}

static void append_gdn_layer(
    std::vector<at::Tensor>& weights, int64_t hidden, int64_t intermediate,
    int64_t state_dim, int64_t layer
) {
    const int64_t qkv_dim = 3 * state_dim;
    const auto ones = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    weights.push_back(at::ones({hidden}, ones));
    weights.push_back(at::ones({hidden}, ones));
    weights.push_back(deterministic({qkv_dim, hidden}, 0.0020, layer));
    weights.push_back(deterministic({state_dim, hidden}, 0.0022, layer + 1));
    weights.push_back(deterministic({1, hidden}, 0.0015, layer + 2));
    weights.push_back(deterministic({1, hidden}, 0.0017, layer + 3));
    weights.push_back(deterministic({1}, 0.0010, layer + 4));
    weights.push_back(deterministic({1}, 0.0010, layer + 5));
    weights.push_back(deterministic({qkv_dim, 1, 4}, 0.0012, layer + 6));
    weights.push_back(at::ones({state_dim}, ones));
    weights.push_back(deterministic({hidden, state_dim}, 0.0020, layer + 7));
    weights.push_back(deterministic({intermediate, hidden}, 0.0020, layer + 8));
    weights.push_back(deterministic({intermediate, hidden}, 0.0018, layer + 9));
    weights.push_back(deterministic({hidden, intermediate}, 0.0020, layer + 10));
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

    constexpr int64_t hidden = 8;
    constexpr int64_t intermediate = 12;
    constexpr int64_t state_dim = 128;
    constexpr int64_t qkv_dim = 3 * state_dim;
    constexpr int64_t vocab = 16;
    constexpr int64_t lora_rank = 4;
    constexpr int64_t local_rank = lora_rank / 2;
    constexpr int64_t slots_per_layer = 8;

    std::vector<at::Tensor> weights;
    append_gdn_layer(weights, hidden, intermediate, state_dim, 0);
    append_gdn_layer(weights, hidden, intermediate, state_dim, 1);
    for (auto& weight : weights) weight.set_requires_grad(false);
    auto weight_ptrs = pointers(weights);

    auto embed = deterministic({vocab, hidden}, 0.0030);
    auto final_norm = at::ones(
        {hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
    auto lm_head = deterministic({vocab, hidden}, 0.0025);
    LayerConfig configs[2]{};
    for (auto& config : configs) {
        config.layer_type = 1;
        config.num_k_heads = 1;
        config.key_dim = state_dim;
        config.num_v_heads = 1;
        config.val_dim = state_dim;
        config.conv_kernel = 4;
        config.rms_eps = 1e-5;
        config.intermediate_size = intermediate;
    }
    const int64_t target_layers[2] = {0, 1};

    setenv("TP_SIZE", "2", 1);
    unsetenv("RUSTRAIN_DATA_PARALLEL");
    void* distributed = qwen36_create_training_context_ex(
        weight_ptrs.data(), weight_ptrs.size(), &embed, &final_norm, &lm_head,
        configs, 2, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, lora_rank,
        // This regression exercises latent-rank-only TP with replicated GDN
        // base weights. ABI15 bit 0 now explicitly enables base GDN head TP.
        target_layers, 2, "in_proj_qkv", 0);
    assert(distributed && qwen36_init_nccl(distributed) == 0);

    setenv("TP_SIZE", "1", 1);
    void* reference = qwen36_create_training_context(
        weight_ptrs.data(), weight_ptrs.size(), &embed, &final_norm, &lm_head,
        configs, 2, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, lora_rank,
        target_layers, 2, "in_proj_qkv");
    assert(reference);

    std::vector<at::Tensor> full_a;
    std::vector<at::Tensor> full_b;
    for (int64_t layer = 0; layer < 2; ++layer) {
        full_a.push_back(deterministic(
            {lora_rank, hidden}, 0.0010, 20 + layer));
        full_b.push_back(deterministic(
            {qkv_dim, lora_rank}, 0.0008, 30 + layer));
        auto local_a = full_a.back().narrow(
            0, rank * local_rank, local_rank).contiguous();
        auto local_b = full_b.back().narrow(
            1, rank * local_rank, local_rank).contiguous();
        const int64_t slot = layer * slots_per_layer;
        assert(qwen36_set_lora_tensor(distributed, slot, 0, &local_a) == 0);
        assert(qwen36_set_lora_tensor(distributed, slot, 1, &local_b) == 0);
        assert(qwen36_set_lora_tensor(reference, slot, 0, &full_a.back()) == 0);
        assert(qwen36_set_lora_tensor(reference, slot, 1, &full_b.back()) == 0);
    }

    auto input_ids = at::tensor({1, 2, 3},
        at::TensorOptions().device(at::kCUDA).dtype(at::kLong)).reshape({1, 3});
    auto target_mask = at::ones({1, 3},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    auto attention_mask = at::ones({1, 3},
        at::TensorOptions().device(at::kCUDA).dtype(at::kBool));

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

    double max_a_grad_diff = 0.0;
    double max_b_grad_diff = 0.0;
    for (int64_t layer = 0; layer < 2; ++layer) {
        const int64_t slot = layer * slots_per_layer;
        auto* local_a_grad = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_grad_accumulator(distributed, slot, 0));
        auto* local_b_grad = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_grad_accumulator(distributed, slot, 1));
        auto* full_a_grad = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_grad_accumulator(reference, slot, 0));
        auto* full_b_grad = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_grad_accumulator(reference, slot, 1));
        assert(local_a_grad && local_b_grad && full_a_grad && full_b_grad);
        max_a_grad_diff = std::max(max_a_grad_diff, max_diff(
            *local_a_grad, full_a_grad->narrow(
                0, rank * local_rank, local_rank)));
        max_b_grad_diff = std::max(max_b_grad_diff, max_diff(
            *local_b_grad, full_b_grad->narrow(
                1, rank * local_rank, local_rank)));
    }
    assert(qwen36_abort_gradient_accumulation(distributed) == 0);
    assert(qwen36_abort_gradient_accumulation(reference) == 0);

    const double distributed_loss = qwen36_train_step(
        distributed, &input_ids, &target_mask, &attention_mask);
    const double reference_loss = qwen36_train_step(
        reference, &input_ids, &target_mask, &attention_mask);
    assert(distributed_loss > 0.0 && reference_loss > 0.0);

    constexpr int64_t optimizer_count = 2 * 2 * slots_per_layer;
    std::vector<void*> local_m(optimizer_count), local_v(optimizer_count);
    std::vector<void*> full_m(optimizer_count), full_v(optimizer_count);
    assert(qwen36_export_optimizer_state(
        distributed, local_m.data(), local_v.data(), optimizer_count) ==
        optimizer_count);
    assert(qwen36_export_optimizer_state(
        reference, full_m.data(), full_v.data(), optimizer_count) ==
        optimizer_count);

    double max_a_param_diff = 0.0;
    double max_b_param_diff = 0.0;
    double max_m_diff = 0.0;
    double max_v_diff = 0.0;
    double max_adam_error = 0.0;
    for (int64_t layer = 0; layer < 2; ++layer) {
        const int64_t slot = layer * slots_per_layer;
        auto* local_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(distributed, slot));
        auto* local_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(distributed, slot));
        auto* full_a_param = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(reference, slot));
        auto* full_b_param = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(reference, slot));
        assert(local_a && local_b && full_a_param && full_b_param);
        max_a_param_diff = std::max(max_a_param_diff, max_diff(
            *local_a, full_a_param->narrow(
                0, rank * local_rank, local_rank)));
        max_b_param_diff = std::max(max_b_param_diff, max_diff(
            *local_b, full_b_param->narrow(
                1, rank * local_rank, local_rank)));

        auto* local_m_a = reinterpret_cast<at::Tensor*>(local_m[2 * slot]);
        auto* local_m_b = reinterpret_cast<at::Tensor*>(local_m[2 * slot + 1]);
        auto* local_v_a = reinterpret_cast<at::Tensor*>(local_v[2 * slot]);
        auto* local_v_b = reinterpret_cast<at::Tensor*>(local_v[2 * slot + 1]);
        auto* full_m_a = reinterpret_cast<at::Tensor*>(full_m[2 * slot]);
        auto* full_m_b = reinterpret_cast<at::Tensor*>(full_m[2 * slot + 1]);
        auto* full_v_a = reinterpret_cast<at::Tensor*>(full_v[2 * slot]);
        auto* full_v_b = reinterpret_cast<at::Tensor*>(full_v[2 * slot + 1]);
        assert(local_m_a && local_m_b && local_v_a && local_v_b);
        assert(full_m_a && full_m_b && full_v_a && full_v_b);
        max_m_diff = std::max({max_m_diff,
            max_diff(*local_m_a, full_m_a->narrow(
                0, rank * local_rank, local_rank)),
            max_diff(*local_m_b, full_m_b->narrow(
                1, rank * local_rank, local_rank))});
        max_v_diff = std::max({max_v_diff,
            max_diff(*local_v_a, full_v_a->narrow(
                0, rank * local_rank, local_rank)),
            max_diff(*local_v_b, full_v_b->narrow(
                1, rank * local_rank, local_rank))});

        auto local_a_before = full_a[layer].narrow(
            0, rank * local_rank, local_rank).contiguous();
        auto local_b_before = full_b[layer].narrow(
            1, rank * local_rank, local_rank).contiguous();
        auto expected_a = (
            local_a_before.to(at::kFloat) - 1e-3 *
                (*local_m_a / (1.0 - 0.9)) /
                (((*local_v_a / (1.0 - 0.999)).sqrt()) + 1e-8))
            .to(at::kBFloat16);
        auto expected_b = (
            local_b_before.to(at::kFloat) - 1e-3 *
                (*local_m_b / (1.0 - 0.9)) /
                (((*local_v_b / (1.0 - 0.999)).sqrt()) + 1e-8))
            .to(at::kBFloat16);
        max_adam_error = std::max({max_adam_error,
            max_diff(*local_a, expected_a), max_diff(*local_b, expected_b)});
    }

    std::printf(
        "latent_tp_two_layer_smoke rank=%d eval_diff=%0.8e loss_diff=%0.8e a_grad_diff=%0.8e b_grad_diff=%0.8e m_diff=%0.8e v_diff=%0.8e adam_error=%0.8e a_param_diff=%0.8e b_param_diff=%0.8e\n",
        rank, std::abs(distributed_eval - reference_eval),
        std::abs(distributed_loss - reference_loss), max_a_grad_diff,
        max_b_grad_diff, max_m_diff, max_v_diff, max_adam_error,
        max_a_param_diff, max_b_param_diff);
    std::fflush(stdout);
    assert(std::abs(distributed_eval - reference_eval) < 5e-3);
    assert(std::abs(distributed_loss - reference_loss) < 5e-3);
    assert(max_a_grad_diff < 5e-4 && max_b_grad_diff < 5e-4);
    assert(max_m_diff < 5e-5 && max_v_diff < 5e-8);
    assert(max_adam_error == 0.0);
    assert(max_a_param_diff < 1e-3 && max_b_param_diff < 1e-3);

    qwen36_free_training_context(reference);
    qwen36_free_training_context(distributed);
    return 0;
}
