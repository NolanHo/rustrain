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

extern "C" void qwen36_set_cuda_device(int32_t);
extern "C" void* qwen36_create_training_context(
    void**, int64_t, void*, void*, void*, void*, int64_t, int32_t,
    double, double, double, double, double, int64_t, double, int64_t,
    const int64_t*, int64_t, const char*);
extern "C" int32_t qwen36_init_nccl(void*);
extern "C" int32_t qwen36_set_base_tp_mlp(void*, int32_t);
extern "C" int32_t qwen36_set_mtp_weights(
    void*, void*, void*, void*, void*, void**, int64_t, void*, int64_t);
extern "C" int64_t qwen36_add_lora(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" void* qwen36_get_lora_a(void*, int64_t);
extern "C" void* qwen36_get_lora_b(void*, int64_t);
extern "C" int32_t qwen36_set_lora_tensor(void*, int64_t, int32_t, void*);
extern "C" void* qwen36_get_lora_grad_accumulator(void*, int64_t, int32_t);
extern "C" int32_t qwen36_abort_gradient_accumulation(void*);
extern "C" double qwen36_eval_step(void*, void*, void*, void*);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" double qwen36_train_micro_step(
    void*, void*, void*, void*, double, int32_t);
extern "C" void qwen36_free_training_context(void*);

static at::Tensor deterministic(std::initializer_list<int64_t> shape, double scale) {
    int64_t count = 1;
    for (int64_t dim : shape) count *= dim;
    return ((at::arange(count,
                 at::TensorOptions().device(at::kCUDA).dtype(at::kFloat))
                 .remainder(17) - 8.0) * scale)
        .reshape(shape).to(at::kBFloat16);
}

static std::vector<void*> pointers(std::vector<at::Tensor>& tensors) {
    std::vector<void*> result;
    result.reserve(tensors.size());
    for (auto& tensor : tensors) result.push_back(&tensor);
    return result;
}

int main() {
    const int process_rank = std::atoi(std::getenv("RANK"));
    const int world = std::atoi(std::getenv("WORLD_SIZE"));
    assert(world == 2 && process_rank >= 0 && process_rank < world);
    qwen36_set_cuda_device(process_rank);

    constexpr int64_t hidden = 8;
    constexpr int64_t intermediate = 12;
    constexpr int64_t vocab = 16;
    constexpr int64_t lora_rank = 4;
    constexpr int64_t local_lora_rank = lora_rank / 2;

    std::vector<at::Tensor> full_weights;
    full_weights.push_back(at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full_weights.push_back(at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full_weights.push_back(deterministic({2 * hidden, hidden}, 0.01));
    full_weights.push_back(at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full_weights.push_back(deterministic({hidden, hidden}, 0.012));
    full_weights.push_back(at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full_weights.push_back(deterministic({hidden, hidden}, 0.008));
    full_weights.push_back(deterministic({hidden, hidden}, 0.011));
    full_weights.push_back(deterministic({intermediate, hidden}, 0.009));
    full_weights.push_back(deterministic({intermediate, hidden}, 0.007));
    full_weights.push_back(deterministic({hidden, intermediate}, 0.01));
    for (auto& weight : full_weights) weight.set_requires_grad(false);

    const int64_t intermediate_start = process_rank * (intermediate / world);
    std::vector<at::Tensor> local_weights(full_weights.begin(), full_weights.begin() + 8);
    local_weights.push_back(full_weights[8].narrow(0, intermediate_start, intermediate / world).contiguous());
    local_weights.push_back(full_weights[9].narrow(0, intermediate_start, intermediate / world).contiguous());
    local_weights.push_back(full_weights[10].narrow(1, intermediate_start, intermediate / world).contiguous());
    assert(local_weights[8].sizes() == at::IntArrayRef({intermediate / 2, hidden}));
    assert(local_weights[9].sizes() == at::IntArrayRef({intermediate / 2, hidden}));
    assert(local_weights[10].sizes() == at::IntArrayRef({hidden, intermediate / 2}));

    auto embed = deterministic({vocab, hidden}, 0.02);
    auto final_norm = at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
    auto lm_head = deterministic({vocab, hidden}, 0.015);
    LayerConfig config{};
    config.layer_type = 0;
    config.num_heads = 1;
    config.num_kv_heads = 1;
    config.head_dim = hidden;
    config.partial_rotary_factor = 1.0;
    config.rope_theta = 10000.0;
    config.rms_eps = 1e-5;
    config.intermediate_size = intermediate;

    const int64_t target_layer = 0;
    auto local_ptrs = pointers(local_weights);
    setenv("TP_SIZE", "2", 1);
    unsetenv("RUSTRAIN_DATA_PARALLEL");
    void* rejected = qwen36_create_training_context(
        local_ptrs.data(), local_ptrs.size(), &embed, &final_norm, &lm_head,
        &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, lora_rank,
        &target_layer, 1, "gate_proj");
    assert(rejected && qwen36_set_base_tp_mlp(rejected, 1) != 0);
    qwen36_free_training_context(rejected);

    void* dynamic_rejected = qwen36_create_training_context(
        local_ptrs.data(), local_ptrs.size(), &embed, &final_norm, &lm_head,
        &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, lora_rank,
        &target_layer, 1, "q_proj");
    assert(dynamic_rejected);
    assert(qwen36_add_lora(
        dynamic_rejected, lora_rank, 1.0, &target_layer, 1, "gate_proj") > 0);
    assert(qwen36_set_base_tp_mlp(dynamic_rejected, 1) != 0);
    qwen36_free_training_context(dynamic_rejected);

    void* distributed = qwen36_create_training_context(
        local_ptrs.data(), local_ptrs.size(), &embed, &final_norm, &lm_head,
        &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, lora_rank,
        &target_layer, 1, "q_proj");
    assert(distributed && qwen36_set_base_tp_mlp(distributed, 1) == 0);
    assert(qwen36_set_base_tp_mlp(distributed, 0) != 0);
    assert(qwen36_set_mtp_weights(
        distributed, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr, 0) != 0);
    assert(qwen36_add_lora(
        distributed, lora_rank, 1.0, &target_layer, 1, nullptr) < 0);
    assert(qwen36_init_nccl(distributed) == 0);

    setenv("TP_SIZE", "1", 1);
    auto full_ptrs = pointers(full_weights);
    void* reference = qwen36_create_training_context(
        full_ptrs.data(), full_ptrs.size(), &embed, &final_norm, &lm_head,
        &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, lora_rank,
        &target_layer, 1, "q_proj");
    assert(reference);

    auto full_a = deterministic({lora_rank, hidden}, 0.002);
    auto full_b = deterministic({2 * hidden, lora_rank}, 0.001);
    auto local_a = full_a.narrow(0, process_rank * local_lora_rank, local_lora_rank).contiguous();
    auto local_b = full_b.narrow(1, process_rank * local_lora_rank, local_lora_rank).contiguous();
    assert(qwen36_set_lora_tensor(distributed, 0, 0, &local_a) == 0);
    assert(qwen36_set_lora_tensor(distributed, 0, 1, &local_b) == 0);
    assert(qwen36_set_lora_tensor(reference, 0, 0, &full_a) == 0);
    assert(qwen36_set_lora_tensor(reference, 0, 1, &full_b) == 0);
    auto local_a_before = local_a.clone();
    auto local_b_before = local_b.clone();

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
    assert(std::abs(distributed_eval - reference_eval) < 1e-4);

    const double distributed_micro = qwen36_train_micro_step(
        distributed, &input_ids, &target_mask, &attention_mask, 1.0, 0);
    const double reference_micro = qwen36_train_micro_step(
        reference, &input_ids, &target_mask, &attention_mask, 1.0, 0);
    assert(distributed_micro > 0.0 && reference_micro > 0.0);
    auto* distributed_b_grad = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_grad_accumulator(distributed, 0, 1));
    auto* reference_b_grad = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_grad_accumulator(reference, 0, 1));
    assert(distributed_b_grad && reference_b_grad);
    auto reference_b_grad_slice = reference_b_grad->narrow(
        1, process_rank * local_lora_rank, local_lora_rank);
    const double b_grad_diff =
        (*distributed_b_grad - reference_b_grad_slice).abs().max().item<double>();
    assert(distributed_b_grad->abs().max().item<double>() > 0.0);
    assert(b_grad_diff < 1e-4);
    assert(qwen36_abort_gradient_accumulation(distributed) == 0);
    assert(qwen36_abort_gradient_accumulation(reference) == 0);

    const double distributed_loss = qwen36_train_step(
        distributed, &input_ids, &target_mask, &attention_mask);
    const double reference_loss = qwen36_train_step(
        reference, &input_ids, &target_mask, &attention_mask);
    assert(distributed_loss > 0.0 && reference_loss > 0.0);
    assert(std::abs(distributed_loss - reference_loss) < 1e-4);

    auto* updated_a = reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(distributed, 0));
    auto* updated_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(distributed, 0));
    auto* reference_a = reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(reference, 0));
    auto* reference_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(reference, 0));
    auto reference_a_slice = reference_a->narrow(
        0, process_rank * local_lora_rank, local_lora_rank);
    auto reference_b_slice = reference_b->narrow(
        1, process_rank * local_lora_rank, local_lora_rank);
    const double a_diff = (*updated_a - reference_a_slice).abs().max().item<double>();
    const double b_diff = (*updated_b - reference_b_slice).abs().max().item<double>();
    const double a_update = (*updated_a - local_a_before).abs().max().item<double>();
    const double b_update = (*updated_b - local_b_before).abs().max().item<double>();
    std::printf(
        "base_tp_mlp_smoke rank=%d eval_diff=%0.8e loss_diff=%0.8e b_grad_diff=%0.8e a_diff=%0.8e b_diff=%0.8e\n",
        process_rank, std::abs(distributed_eval - reference_eval),
        std::abs(distributed_loss - reference_loss), b_grad_diff, a_diff, b_diff);
    assert(a_diff < 1e-5 && b_diff < 1e-5);
    assert(a_update > 0.0 || b_update > 0.0);

    qwen36_free_training_context(reference);
    qwen36_free_training_context(distributed);
    return 0;
}
