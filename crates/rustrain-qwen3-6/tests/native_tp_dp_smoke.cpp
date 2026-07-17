#include <ATen/ATen.h>

#include <algorithm>
#include <array>
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
extern "C" int32_t qwen36_init_parallel_nccl(
    void*, int32_t, int32_t, int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t);
extern "C" int32_t qwen36_set_lora_tensor(void*, int64_t, int32_t, void*);
extern "C" void* qwen36_get_lora_a(void*, int64_t);
extern "C" void* qwen36_get_lora_b(void*, int64_t);
extern "C" int64_t qwen36_export_optimizer_state(
    void*, void**, void**, int64_t);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" int64_t qwen36_add_lora(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" double qwen36_train_multi_lora_selected(
    void*, void*, void*, void*, const int64_t*, int32_t, int32_t);
extern "C" int64_t qwen36_get_adapter_step_count(void*, int64_t);
extern "C" void qwen36_free_training_context(void*);

static constexpr double kLearningRate = 1e-3;
static constexpr double kBeta1 = 0.9;
static constexpr double kBeta2 = 0.999;
static constexpr double kAdamEps = 1e-8;
static constexpr int64_t kOptimizerSlots = 14;

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

struct Batch {
    at::Tensor input_ids;
    at::Tensor target_mask;
    at::Tensor attention_mask;
};

static Batch make_batch(int dp_rank) {
    auto options_long = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
    auto options_float = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    auto options_bool = at::TensorOptions().device(at::kCUDA).dtype(at::kBool);
    if (dp_rank == 0) {
        return {
            at::tensor({1, 2, 3, 4}, options_long).reshape({1, 4}),
            at::tensor({1.0, 1.0, 1.0, 1.0}, options_float).reshape({1, 4}),
            at::ones({1, 4}, options_bool),
        };
    }
    return {
        at::tensor({4, 2, 5, 1}, options_long).reshape({1, 4}),
        at::tensor({1.0, 1.0, 0.0, 0.0}, options_float).reshape({1, 4}),
        at::ones({1, 4}, options_bool),
    };
}

struct LoraWeights {
    at::Tensor q_a, q_b;
    at::Tensor k_a, k_b;
    at::Tensor v_a, v_b;
    at::Tensor o_a, o_b;
};

static LoraWeights make_lora_weights(
    int64_t hidden, int64_t heads, int64_t kv_heads,
    int64_t head_dim, int64_t lora_rank
) {
    return {
        deterministic({lora_rank, hidden}, 0.0020),
        deterministic({2 * heads * head_dim, lora_rank}, 0.0010),
        deterministic({lora_rank, hidden}, 0.0018),
        deterministic({kv_heads * head_dim, lora_rank}, 0.0011),
        deterministic({lora_rank, hidden}, 0.0016),
        deterministic({kv_heads * head_dim, lora_rank}, 0.0009),
        deterministic({lora_rank, heads * head_dim}, 0.0015),
        deterministic({hidden, lora_rank}, 0.0012),
    };
}

static LoraWeights clone_lora_weights(const LoraWeights& weights) {
    return {
        weights.q_a.clone(), weights.q_b.clone(),
        weights.k_a.clone(), weights.k_b.clone(),
        weights.v_a.clone(), weights.v_b.clone(),
        weights.o_a.clone(), weights.o_b.clone(),
    };
}

static void offset_lora_weights(LoraWeights& weights, double offset) {
    std::array<at::Tensor*, 8> tensors = {
        &weights.q_a, &weights.q_b, &weights.k_a, &weights.k_b,
        &weights.v_a, &weights.v_b, &weights.o_a, &weights.o_b,
    };
    for (auto* tensor : tensors) tensor->add_(offset);
}

static void install_full_lora(void* context, LoraWeights& weights) {
    std::array<at::Tensor*, 8> tensors = {
        &weights.q_a, &weights.q_b, &weights.k_a, &weights.k_b,
        &weights.v_a, &weights.v_b, &weights.o_a, &weights.o_b,
    };
    for (int64_t index = 0; index < 4; ++index) {
        assert(qwen36_set_lora_tensor(context, index, 0, tensors[2 * index]) == 0);
        assert(qwen36_set_lora_tensor(context, index, 1, tensors[2 * index + 1]) == 0);
    }
}

static LoraWeights shard_lora_for_tp(
    const LoraWeights& full, int tp_rank, int64_t local_heads,
    int64_t local_kv_heads, int64_t head_dim
) {
    return {
        full.q_a,
        full.q_b.narrow(
            0, tp_rank * 2 * local_heads * head_dim,
            2 * local_heads * head_dim).contiguous(),
        full.k_a,
        full.k_b.narrow(
            0, tp_rank * local_kv_heads * head_dim,
            local_kv_heads * head_dim).contiguous(),
        full.v_a,
        full.v_b.narrow(
            0, tp_rank * local_kv_heads * head_dim,
            local_kv_heads * head_dim).contiguous(),
        full.o_a.narrow(
            1, tp_rank * local_heads * head_dim,
            local_heads * head_dim).contiguous(),
        full.o_b,
    };
}

static std::array<at::Tensor*, 8> lora_parameters(void* context) {
    std::array<at::Tensor*, 8> result{};
    for (int64_t index = 0; index < 4; ++index) {
        result[2 * index] = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(context, index));
        result[2 * index + 1] = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(context, index));
        assert(result[2 * index] && result[2 * index + 1]);
    }
    return result;
}

struct OptimizerState {
    std::vector<void*> m;
    std::vector<void*> v;
};

static OptimizerState optimizer_state(void* context) {
    OptimizerState result{
        std::vector<void*>(kOptimizerSlots),
        std::vector<void*>(kOptimizerSlots),
    };
    assert(qwen36_export_optimizer_state(
        context, result.m.data(), result.v.data(), kOptimizerSlots) ==
        kOptimizerSlots);
    return result;
}

static at::Tensor& state_tensor(std::vector<void*>& tensors, int64_t index) {
    auto* tensor = reinterpret_cast<at::Tensor*>(tensors[index]);
    assert(tensor);
    return *tensor;
}

struct FullReference {
    void* context;
    double loss;
    OptimizerState optimizer;
};

static FullReference run_full_reference(
    std::vector<void*>& full_ptrs, at::Tensor& embed, at::Tensor& final_norm,
    at::Tensor& lm_head, LayerConfig& config, LoraWeights& lora,
    Batch& batch, int64_t vocab, int64_t lora_rank
) {
    const int64_t target_layer = 0;
    void* context = qwen36_create_training_context(
        full_ptrs.data(), full_ptrs.size(), &embed, &final_norm, &lm_head,
        &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, kLearningRate, kBeta1, kBeta2, kAdamEps,
        vocab, 1e-5, lora_rank, &target_layer, 1,
        "q_proj,k_proj,v_proj,o_proj");
    assert(context);
    install_full_lora(context, lora);
    const double loss = qwen36_train_step(
        context, &batch.input_ids, &batch.target_mask, &batch.attention_mask);
    assert(loss > 0.0 && std::isfinite(loss));
    return {context, loss, optimizer_state(context)};
}

int main() {
    assert(qwen36_kernel_abi_version() == 16);
    const int rank = std::atoi(std::getenv("RANK"));
    const int world = std::atoi(std::getenv("WORLD_SIZE"));
    const int local_rank = std::atoi(
        std::getenv("LOCAL_RANK") ? std::getenv("LOCAL_RANK") : "0");
    assert(world == 4 && rank >= 0 && rank < world);
    const int tp_rank = rank % 2;
    const int dp_rank = rank / 2;
    qwen36_set_cuda_device(local_rank);

    constexpr int64_t hidden = 8;
    constexpr int64_t heads = 4;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t head_dim = 2;
    constexpr int64_t intermediate = 12;
    constexpr int64_t vocab = 16;
    constexpr int64_t lora_rank = 4;
    constexpr int64_t local_heads = heads / 2;
    constexpr int64_t local_kv_heads = kv_heads / 2;

    std::vector<at::Tensor> full_weights;
    full_weights.push_back(at::ones(
        {hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full_weights.push_back(at::ones(
        {hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full_weights.push_back(deterministic({2 * heads * head_dim, hidden}, 0.010));
    full_weights.push_back(at::ones(
        {head_dim}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full_weights.push_back(deterministic({kv_heads * head_dim, hidden}, 0.012));
    full_weights.push_back(at::ones(
        {head_dim}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    full_weights.push_back(deterministic({kv_heads * head_dim, hidden}, 0.008));
    full_weights.push_back(deterministic({hidden, heads * head_dim}, 0.011));
    full_weights.push_back(deterministic({intermediate, hidden}, 0.009));
    full_weights.push_back(deterministic({intermediate, hidden}, 0.007));
    full_weights.push_back(deterministic({hidden, intermediate}, 0.010));
    for (auto& weight : full_weights) weight.set_requires_grad(false);

    std::vector<at::Tensor> local_weights;
    local_weights.push_back(full_weights[0]);
    local_weights.push_back(full_weights[1]);
    local_weights.push_back(full_weights[2].narrow(
        0, tp_rank * 2 * local_heads * head_dim,
        2 * local_heads * head_dim).contiguous());
    local_weights.push_back(full_weights[3]);
    local_weights.push_back(full_weights[4].narrow(
        0, tp_rank * local_kv_heads * head_dim,
        local_kv_heads * head_dim).contiguous());
    local_weights.push_back(full_weights[5]);
    local_weights.push_back(full_weights[6].narrow(
        0, tp_rank * local_kv_heads * head_dim,
        local_kv_heads * head_dim).contiguous());
    local_weights.push_back(full_weights[7].narrow(
        1, tp_rank * local_heads * head_dim,
        local_heads * head_dim).contiguous());
    local_weights.insert(
        local_weights.end(), full_weights.begin() + 8, full_weights.end());

    auto embed = deterministic({vocab, hidden}, 0.020);
    auto final_norm = at::ones(
        {hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
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

    auto local_ptrs = pointers(local_weights);
    auto full_ptrs = pointers(full_weights);
    const int64_t target_layer = 0;

    // Native validation must reject a topology whose declared axes do not
    // cover the process world before any communicator is initialized.
    setenv("TP_SIZE", "2", 1);
    setenv("DP_SIZE", "3", 1);
    setenv("RUSTRAIN_DATA_PARALLEL", "1", 1);
    void* invalid = qwen36_create_training_context_ex(
        local_ptrs.data(), local_ptrs.size(), &embed, &final_norm, &lm_head,
        &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, kLearningRate, kBeta1, kBeta2, kAdamEps,
        vocab, 1e-5, lora_rank, &target_layer, 1,
        "q_proj,k_proj,v_proj,o_proj", 3);
    assert(invalid == nullptr);

    setenv("DP_SIZE", "2", 1);
    auto full_lora = make_lora_weights(
        hidden, heads, kv_heads, head_dim, lora_rank);
    auto local_lora = shard_lora_for_tp(
        full_lora, tp_rank, local_heads, local_kv_heads, head_dim);
    auto synchronized_lora = clone_lora_weights(local_lora);
    if (dp_rank == 1) {
        // DP must restore every local shard from the matching dp_rank=0 peer.
        offset_lora_weights(local_lora, 0.25);
    } else if (tp_rank == 1) {
        // TP must restore projection-replicated factors from tp_rank=0.
        local_lora.q_a.add_(0.25);
        local_lora.k_a.add_(0.25);
        local_lora.v_a.add_(0.25);
        local_lora.o_b.add_(0.25);
    }
    void* distributed = qwen36_create_training_context_ex(
        local_ptrs.data(), local_ptrs.size(), &embed, &final_norm, &lm_head,
        &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, kLearningRate, kBeta1, kBeta2, kAdamEps,
        vocab, 1e-5, lora_rank, &target_layer, 1,
        "q_proj,k_proj,v_proj,o_proj", 3);
    assert(distributed);
    install_full_lora(distributed, local_lora);
    assert(qwen36_init_parallel_nccl(
        distributed, rank, world,
        tp_rank, 2, dp_rank * 2,
        dp_rank, 2, tp_rank) == 0);

    const auto synchronized_params = lora_parameters(distributed);
    const std::array<at::Tensor*, 8> expected_synchronized = {
        &synchronized_lora.q_a, &synchronized_lora.q_b,
        &synchronized_lora.k_a, &synchronized_lora.k_b,
        &synchronized_lora.v_a, &synchronized_lora.v_b,
        &synchronized_lora.o_a, &synchronized_lora.o_b,
    };
    double broadcast_diff = 0.0;
    for (int64_t index = 0; index < 8; ++index) {
        broadcast_diff = std::max(
            broadcast_diff,
            max_diff(*synchronized_params[index], *expected_synchronized[index]));
    }
    assert(broadcast_diff == 0.0);

    auto batch0 = make_batch(0);
    auto batch1 = make_batch(1);
    Batch full_batch{
        at::cat({batch0.input_ids, batch1.input_ids}, 0),
        at::cat({batch0.target_mask, batch1.target_mask}, 0),
        at::cat({batch0.attention_mask, batch1.attention_mask}, 0),
    };
    Batch& local_batch = dp_rank == 0 ? batch0 : batch1;
    const double distributed_loss = qwen36_train_step(
        distributed, &local_batch.input_ids, &local_batch.target_mask,
        &local_batch.attention_mask);
    assert(distributed_loss > 0.0 && std::isfinite(distributed_loss));

    // Reference contexts stay process-local. The concatenated batch is the
    // hard oracle for the DP-reduced update; the two local references prove
    // that unequal token counts are not accidentally averaged by replica.
    setenv("TP_SIZE", "1", 1);
    setenv("DP_SIZE", "1", 1);
    unsetenv("RUSTRAIN_DATA_PARALLEL");
    auto reference_lora = make_lora_weights(
        hidden, heads, kv_heads, head_dim, lora_rank);
    auto reference = run_full_reference(
        full_ptrs, embed, final_norm, lm_head, config, reference_lora,
        full_batch, vocab, lora_rank);
    auto local0_lora = make_lora_weights(
        hidden, heads, kv_heads, head_dim, lora_rank);
    auto local0 = run_full_reference(
        full_ptrs, embed, final_norm, lm_head, config, local0_lora,
        batch0, vocab, lora_rank);
    auto local1_lora = make_lora_weights(
        hidden, heads, kv_heads, head_dim, lora_rank);
    auto local1 = run_full_reference(
        full_ptrs, embed, final_norm, lm_head, config, local1_lora,
        batch1, vocab, lora_rank);

    auto distributed_params = lora_parameters(distributed);
    auto reference_params = lora_parameters(reference.context);
    auto distributed_optimizer = optimizer_state(distributed);
    std::array<at::Tensor, 8> before = {
        synchronized_lora.q_a, synchronized_lora.q_b,
        synchronized_lora.k_a, synchronized_lora.k_b,
        synchronized_lora.v_a, synchronized_lora.v_b,
        synchronized_lora.o_a, synchronized_lora.o_b,
    };
    auto reference_slice = [&](const at::Tensor& tensor, int64_t index) {
        switch (index) {
            case 1:
                return tensor.narrow(
                    0, tp_rank * 2 * local_heads * head_dim,
                    2 * local_heads * head_dim);
            case 3:
            case 5:
                return tensor.narrow(
                    0, tp_rank * local_kv_heads * head_dim,
                    local_kv_heads * head_dim);
            case 6:
                return tensor.narrow(
                    1, tp_rank * local_heads * head_dim,
                    local_heads * head_dim);
            default:
                return tensor;
        }
    };

    double parameter_diff = 0.0;
    double optimizer_m_diff = 0.0;
    double optimizer_v_diff = 0.0;
    double adam_error = 0.0;
    double weighted_reference_error = 0.0;
    double equal_replica_gap = 0.0;
    constexpr double token_count0 = 3.0;
    constexpr double token_count1 = 1.0;
    for (int64_t index = 0; index < 8; ++index) {
        auto expected_parameter = reference_slice(*reference_params[index], index);
        auto expected_m = reference_slice(
            state_tensor(reference.optimizer.m, index), index);
        auto expected_v = reference_slice(
            state_tensor(reference.optimizer.v, index), index);
        parameter_diff = std::max(
            parameter_diff, max_diff(*distributed_params[index], expected_parameter));
        optimizer_m_diff = std::max(
            optimizer_m_diff,
            max_diff(state_tensor(distributed_optimizer.m, index), expected_m));
        optimizer_v_diff = std::max(
            optimizer_v_diff,
            max_diff(state_tensor(distributed_optimizer.v, index), expected_v));

        auto expected_adam_parameter = (
            before[index].to(at::kFloat) - kLearningRate *
                (state_tensor(distributed_optimizer.m, index) / (1.0 - kBeta1)) /
                ((state_tensor(distributed_optimizer.v, index) /
                    (1.0 - kBeta2)).sqrt() + kAdamEps))
            .to(at::kBFloat16);
        adam_error = std::max(
            adam_error,
            max_diff(*distributed_params[index], expected_adam_parameter));

        auto local_m0 = state_tensor(local0.optimizer.m, index);
        auto local_m1 = state_tensor(local1.optimizer.m, index);
        auto token_weighted_m =
            (local_m0 * token_count0 + local_m1 * token_count1) /
            (token_count0 + token_count1);
        auto equal_replica_m = (local_m0 + local_m1) * 0.5;
        weighted_reference_error = std::max(
            weighted_reference_error, max_diff(
                state_tensor(reference.optimizer.m, index), token_weighted_m));
        equal_replica_gap = std::max(
            equal_replica_gap, max_diff(
                state_tensor(reference.optimizer.m, index), equal_replica_m));
    }

    const double expected_local_loss =
        dp_rank == 0 ? local0.loss : local1.loss;
    const double local_loss_diff = std::abs(distributed_loss - expected_local_loss);
    std::printf(
        "native_tp_dp_smoke rank=%d tp_rank=%d dp_rank=%d "
        "broadcast_diff=%0.8e local_loss_diff=%0.8e "
        "parameter_diff=%0.8e m_diff=%0.8e "
        "v_diff=%0.8e adam_error=%0.8e weighted_reference_error=%0.8e "
        "equal_replica_gap=%0.8e\n",
        rank, tp_rank, dp_rank, broadcast_diff, local_loss_diff, parameter_diff,
        optimizer_m_diff, optimizer_v_diff, adam_error,
        weighted_reference_error, equal_replica_gap);
    std::fflush(stdout);

    // TP row-parallel BF16 rounds each local matmul before the collective.
    assert(local_loss_diff < 5e-3);
    assert(parameter_diff <= 2e-3);
    assert(optimizer_m_diff < 5e-5 && optimizer_v_diff < 5e-8);
    assert(adam_error < 1e-8);
    assert(weighted_reference_error < 5e-5);
    assert(equal_replica_gap > 1e-8);

    // A fixed-size registry signature must reject a different tenant order
    // before any adapter-shaped collective can mix gradients or deadlock.
    const int64_t adapter_one = qwen36_add_lora(
        distributed, lora_rank, lora_rank, &target_layer, 1, "q_proj");
    const int64_t adapter_two = qwen36_add_lora(
        distributed, lora_rank, lora_rank, &target_layer, 1, "q_proj");
    assert(adapter_one > 0 && adapter_two > adapter_one);
    std::array<int64_t, 2> mismatched_ids = dp_rank == 0
        ? std::array<int64_t, 2>{adapter_one, adapter_two}
        : std::array<int64_t, 2>{adapter_two, adapter_one};
    auto dynamic_input = local_batch.input_ids.repeat({2, 1});
    auto dynamic_target = local_batch.target_mask.repeat({2, 1});
    auto dynamic_attention = local_batch.attention_mask.repeat({2, 1});
    const double mismatched_loss = qwen36_train_multi_lora_selected(
        distributed, &dynamic_input, &dynamic_target, &dynamic_attention,
        mismatched_ids.data(), mismatched_ids.size(), lora_rank);
    assert(mismatched_loss < 0.0);
    assert(qwen36_get_adapter_step_count(distributed, adapter_one) == 0);
    assert(qwen36_get_adapter_step_count(distributed, adapter_two) == 0);

    const std::array<int64_t, 2> ordered_ids{adapter_one, adapter_two};
    const double dynamic_loss = qwen36_train_multi_lora_selected(
        distributed, &dynamic_input, &dynamic_target, &dynamic_attention,
        ordered_ids.data(), ordered_ids.size(), lora_rank);
    assert(dynamic_loss > 0.0 && std::isfinite(dynamic_loss));
    assert(qwen36_get_adapter_step_count(distributed, adapter_one) == 1);
    assert(qwen36_get_adapter_step_count(distributed, adapter_two) == 1);

    qwen36_free_training_context(local1.context);
    qwen36_free_training_context(local0.context);
    qwen36_free_training_context(reference.context);
    qwen36_free_training_context(distributed);
    return 0;
}
