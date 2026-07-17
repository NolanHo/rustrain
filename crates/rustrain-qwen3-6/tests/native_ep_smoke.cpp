#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
extern "C" int64_t qwen36_add_lora(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" void* qwen36_get_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t);
extern "C" int32_t qwen36_set_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t, void*);
extern "C" void* qwen36_get_adapter_optimizer_tensor(
    void*, int64_t, int64_t, const char*, int32_t, int32_t);
extern "C" int64_t qwen36_get_adapter_step_count(void*, int64_t);
extern "C" double qwen36_train_multi_lora(
    void*, void*, void*, void*, int32_t, int32_t);
extern "C" int64_t qwen36_get_lora_count(void*);
extern "C" void* qwen36_get_lora_a(void*, int64_t);
extern "C" void* qwen36_get_lora_b(void*, int64_t);
extern "C" int32_t qwen36_set_lora_tensor(void*, int64_t, int32_t, void*);
extern "C" int64_t qwen36_export_optimizer_state(void*, void**, void**, int64_t);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" void qwen36_free_training_context(void*);

static at::Tensor cuda_rand(std::initializer_list<int64_t> shape) {
    return at::randn(shape, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
}

static std::vector<void*> tensor_ptrs(std::vector<at::Tensor>& tensors) {
    std::vector<void*> ptrs;
    ptrs.reserve(tensors.size());
    for (auto& tensor : tensors) ptrs.push_back(&tensor);
    return ptrs;
}

static double max_abs_diff(const at::Tensor& actual, const at::Tensor& expected) {
    assert(actual.sizes() == expected.sizes());
    return (actual.to(at::kFloat) - expected.to(at::kFloat))
        .abs().max().item<double>();
}

static double update_norm(const at::Tensor& after, const at::Tensor& before) {
    return (after.to(at::kFloat) - before.to(at::kFloat))
        .abs().sum().item<double>();
}

static double first_adam_step_diff(
    const at::Tensor& actual,
    const at::Tensor& before,
    const at::Tensor& first_m,
    const at::Tensor& first_v
) {
    constexpr double lr = 1e-3;
    constexpr double beta1 = 0.9;
    constexpr double beta2 = 0.999;
    constexpr double eps = 1e-8;
    auto m_hat = first_m.to(at::kFloat) / (1.0 - beta1);
    auto v_hat = first_v.to(at::kFloat) / (1.0 - beta2);
    auto expected = (before.to(at::kFloat) -
        lr * m_hat / (v_hat.sqrt() + eps)).to(actual.scalar_type());
    return max_abs_diff(actual, expected);
}

struct ExpertLora {
    at::Tensor* gate_up_a;
    at::Tensor* gate_up_b;
    at::Tensor* down_a;
    at::Tensor* down_b;
};

static ExpertLora get_expert_lora(void* ctx) {
    assert(qwen36_get_lora_count(ctx) == 9);
    ExpertLora lora{
        reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(ctx, 7)),
        reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(ctx, 7)),
        reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(ctx, 8)),
        reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(ctx, 8)),
    };
    assert(lora.gate_up_a && lora.gate_up_b && lora.down_a && lora.down_b);
    return lora;
}

static void set_expert_lora(
    void* ctx,
    at::Tensor& gate_up_a,
    at::Tensor& gate_up_b,
    at::Tensor& down_a,
    at::Tensor& down_b
) {
    assert(qwen36_set_lora_tensor(ctx, 7, 0, &gate_up_a) == 0);
    assert(qwen36_set_lora_tensor(ctx, 7, 1, &gate_up_b) == 0);
    assert(qwen36_set_lora_tensor(ctx, 8, 0, &down_a) == 0);
    assert(qwen36_set_lora_tensor(ctx, 8, 1, &down_b) == 0);
}

int main() {
    const int rank = std::atoi(std::getenv("RANK") ? std::getenv("RANK") : "0");
    const int world = std::atoi(std::getenv("WORLD_SIZE") ? std::getenv("WORLD_SIZE") : "1");
    const int local_rank = std::atoi(
        std::getenv("LOCAL_RANK") ? std::getenv("LOCAL_RANK") : "0");
    const int a2a = std::getenv("QWEN36_EP_A2A") &&
        std::strcmp(std::getenv("QWEN36_EP_A2A"), "0") != 0;
    const int sharded = a2a && std::getenv("QWEN36_EP_A2A_SHARDED") &&
        std::strcmp(std::getenv("QWEN36_EP_A2A_SHARDED"), "0") != 0;
    const bool dynamic_only = std::getenv("QWEN36_DYNAMIC_ONLY") &&
        std::strcmp(std::getenv("QWEN36_DYNAMIC_ONLY"), "0") != 0;
    // Replicated A2A receives duplicate source rows in rank order, so its BF16
    // accumulation is not bit-identical to the full-expert reference. Sharded
    // A2A uses distinct rows; its optimizer state matches closely, while a
    // near-boundary update can land in the adjacent BF16 parameter bin. Keep
    // the legacy threshold strict, bound each BF16 case separately, and retain
    // the exact standard-Adam oracle below for all modes.
    const double param_tol = sharded ? 2e-3 : (a2a ? 2e-4 : 1e-5);
    const double m_tol = sharded ? 5e-4 : (a2a ? 5e-3 : 1e-5);
    const double v_tol = a2a ? 1e-5 : 1e-6;
    assert(world == 2 && rank >= 0 && rank < world);
    assert(!std::getenv("TP_SIZE") || std::atoi(std::getenv("TP_SIZE")) == 1);
    c10::cuda::CUDAGuard guard(local_rank);

    // Every process deterministically creates the same global model. The EP
    // context receives a distinct contiguous expert slice from that model.
    at::manual_seed(100);
    constexpr int64_t hidden = 16;
    constexpr int64_t vocab = 8;
    constexpr int64_t experts = 2;
    constexpr int64_t head_dim = 8;
    constexpr int64_t intermediate = 8;
    constexpr int64_t lora_rank = 4;

    std::vector<at::Tensor> global_weights;
    global_weights.push_back(cuda_rand({hidden}));
    global_weights.push_back(cuda_rand({hidden}));
    global_weights.push_back(cuda_rand({2 * head_dim, hidden}));
    global_weights.push_back(cuda_rand({head_dim}));
    global_weights.push_back(cuda_rand({head_dim, hidden}));
    global_weights.push_back(cuda_rand({head_dim}));
    global_weights.push_back(cuda_rand({head_dim, hidden}));
    global_weights.push_back(cuda_rand({hidden, head_dim}));
    global_weights.push_back(cuda_rand({experts, hidden}));
    global_weights.push_back(cuda_rand({1, hidden}));
    global_weights.push_back(cuda_rand({intermediate, hidden}));
    global_weights.push_back(cuda_rand({intermediate, hidden}));
    global_weights.push_back(cuda_rand({hidden, intermediate}));
    global_weights.push_back(cuda_rand({experts, 2 * intermediate, hidden}));
    global_weights.push_back(cuda_rand({experts, hidden, intermediate}));
    for (auto& weight : global_weights) weight.set_requires_grad(false);

    assert(max_abs_diff(
        global_weights[13].narrow(0, 0, 1),
        global_weights[13].narrow(0, 1, 1)) > 0.0);
    assert(max_abs_diff(
        global_weights[14].narrow(0, 0, 1),
        global_weights[14].narrow(0, 1, 1)) > 0.0);

    std::vector<at::Tensor> distributed_weights = global_weights;
    distributed_weights[13] = global_weights[13].narrow(0, rank, 1).contiguous();
    distributed_weights[14] = global_weights[14].narrow(0, rank, 1).contiguous();
    auto distributed_weight_ptrs = tensor_ptrs(distributed_weights);
    auto reference_weight_ptrs = tensor_ptrs(global_weights);

    auto embed = cuda_rand({vocab, hidden});
    auto final_norm = cuda_rand({hidden});
    auto lm_head = cuda_rand({vocab, hidden});
    embed.set_requires_grad(false);
    final_norm.set_requires_grad(false);
    lm_head.set_requires_grad(false);

    LayerConfig distributed_config{};
    distributed_config.layer_type = 0;
    distributed_config.num_heads = 1;
    distributed_config.num_kv_heads = 1;
    distributed_config.head_dim = head_dim;
    distributed_config.partial_rotary_factor = 1.0;
    distributed_config.rope_theta = 10000.0;
    distributed_config.rms_eps = 1e-5;
    distributed_config.num_experts = experts;
    // With two experts, top_k=2 guarantees every token exercises both EP
    // shards and both corresponding expert LoRA optimizer paths.
    distributed_config.top_k = 2;
    distributed_config.moe_intermediate = intermediate;
    distributed_config.expert_start = rank;
    distributed_config.expert_count = 1;
    distributed_config.norm_topk_prob = 1;
    distributed_config.nccl_comm = nullptr;
    distributed_config.nccl_stream = nullptr;

    LayerConfig reference_config = distributed_config;
    reference_config.expert_start = 0;
    reference_config.expert_count = experts;

    const int64_t target_layer = 0;
    const char* targets = "experts_gate_up_proj,experts_down_proj";
    void* distributed_ctx = qwen36_create_training_context(
        distributed_weight_ptrs.data(),
        static_cast<int64_t>(distributed_weight_ptrs.size()),
        &embed, &final_norm, &lm_head, &distributed_config, 1,
        static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, lora_rank,
        &target_layer, 1, targets);
    assert(distributed_ctx);

    // The reference owns all experts and deliberately never initializes NCCL.
    // Its copied LayerConfig therefore retains null communication handles even
    // though WORLD_SIZE remains two for the distributed process.
    void* reference_ctx = qwen36_create_training_context(
        reference_weight_ptrs.data(),
        static_cast<int64_t>(reference_weight_ptrs.size()),
        &embed, &final_norm, &lm_head, &reference_config, 1,
        static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, lora_rank,
        &target_layer, 1, targets);
    assert(reference_ctx);

    auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    auto global_gate_up_a =
        ((at::arange(experts * lora_rank * hidden, opts) + 1.0) * 1e-4)
            .reshape({experts, lora_rank, hidden}).to(at::kBFloat16);
    auto global_gate_up_b =
        ((at::arange(experts * 2 * intermediate * lora_rank, opts) + 3.0) * 5e-5)
            .reshape({experts, 2 * intermediate, lora_rank}).to(at::kBFloat16);
    auto global_down_a =
        ((at::arange(experts * lora_rank * intermediate, opts) + 5.0) * 8e-5)
            .reshape({experts, lora_rank, intermediate}).to(at::kBFloat16);
    auto global_down_b =
        ((at::arange(experts * hidden * lora_rank, opts) + 7.0) * 6e-5)
            .reshape({experts, hidden, lora_rank}).to(at::kBFloat16);

    assert(max_abs_diff(
        global_gate_up_a.narrow(0, 0, 1),
        global_gate_up_a.narrow(0, 1, 1)) > 0.0);
    assert(max_abs_diff(
        global_down_b.narrow(0, 0, 1),
        global_down_b.narrow(0, 1, 1)) > 0.0);

    auto local_gate_up_a = global_gate_up_a.narrow(0, rank, 1).contiguous();
    auto local_gate_up_b = global_gate_up_b.narrow(0, rank, 1).contiguous();
    auto local_down_a = global_down_a.narrow(0, rank, 1).contiguous();
    auto local_down_b = global_down_b.narrow(0, rank, 1).contiguous();
    set_expert_lora(distributed_ctx,
        local_gate_up_a, local_gate_up_b, local_down_a, local_down_b);
    set_expert_lora(reference_ctx,
        global_gate_up_a, global_gate_up_b, global_down_a, global_down_b);

    auto distributed_lora = get_expert_lora(distributed_ctx);
    auto reference_lora = get_expert_lora(reference_ctx);
    assert(distributed_lora.gate_up_a->sizes() ==
        at::IntArrayRef({1, lora_rank, hidden}));
    assert(reference_lora.gate_up_a->sizes() ==
        at::IntArrayRef({experts, lora_rank, hidden}));
    assert(distributed_lora.gate_up_b->sizes() ==
        at::IntArrayRef({1, 2 * intermediate, lora_rank}));
    assert(distributed_lora.down_a->sizes() ==
        at::IntArrayRef({1, lora_rank, intermediate}));
    assert(distributed_lora.down_b->sizes() ==
        at::IntArrayRef({1, hidden, lora_rank}));

    auto gate_up_a_before = distributed_lora.gate_up_a->clone();
    auto gate_up_b_before = distributed_lora.gate_up_b->clone();
    auto down_a_before = distributed_lora.down_a->clone();
    auto down_b_before = distributed_lora.down_b->clone();

    // Initialize NCCL only after the reference context is fully constructed.
    // qwen36_init_nccl mutates only the distributed context's copied configs.
    assert(qwen36_init_nccl(distributed_ctx) == 0);

    auto long_opts = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
    auto float_opts = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    auto bool_opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBool);
    if (!dynamic_only) {
    at::Tensor input_ids;
    at::Tensor target_mask;
    at::Tensor attention_mask;
    at::Tensor reference_input_ids;
    at::Tensor reference_target_mask;
    at::Tensor reference_attention_mask;
    if (sharded) {
        // Deliberately unequal supervised-token counts: rank 0 contributes one
        // response token, rank 1 contributes three. The reference evaluates
        // the deterministic global batch on a no-NCCL full-expert context.
        input_ids = rank == 0
            ? at::tensor({1, 2, 3, 4}, long_opts).reshape({1, 4})
            : at::tensor({4, 5, 6, 7}, long_opts).reshape({1, 4});
        target_mask = rank == 0
            ? at::tensor({0, 1, 0, 0}, float_opts).reshape({1, 4})
            : at::tensor({0, 1, 1, 1}, float_opts).reshape({1, 4});
        attention_mask = at::ones({1, 4}, bool_opts);
        reference_input_ids = at::cat({
            at::tensor({1, 2, 3, 4}, long_opts).reshape({1, 4}),
            at::tensor({4, 5, 6, 7}, long_opts).reshape({1, 4})}, 0);
        reference_target_mask = at::cat({
            at::tensor({0, 1, 0, 0}, float_opts).reshape({1, 4}),
            at::tensor({0, 1, 1, 1}, float_opts).reshape({1, 4})}, 0);
        reference_attention_mask = at::ones({2, 4}, bool_opts);
    } else {
        input_ids = at::tensor({1, 2, 3}, long_opts).reshape({1, 3});
        target_mask = at::ones({1, 3}, float_opts);
        attention_mask = at::ones({1, 3}, bool_opts);
        reference_input_ids = input_ids;
        reference_target_mask = target_mask;
        reference_attention_mask = attention_mask;
    }

    const double distributed_loss = qwen36_train_step(
        distributed_ctx, &input_ids, &target_mask, &attention_mask);
    const double reference_loss = qwen36_train_step(
        reference_ctx, &reference_input_ids, &reference_target_mask,
        &reference_attention_mask);
    c10::cuda::device_synchronize();
    assert(distributed_loss > 0.0 && std::isfinite(distributed_loss));
    assert(reference_loss > 0.0 && std::isfinite(reference_loss));

    const auto reference_slice = [rank](const at::Tensor& tensor) {
        return tensor.narrow(0, rank, 1);
    };
    const double gate_up_a_diff = max_abs_diff(
        *distributed_lora.gate_up_a, reference_slice(*reference_lora.gate_up_a));
    const double gate_up_b_diff = max_abs_diff(
        *distributed_lora.gate_up_b, reference_slice(*reference_lora.gate_up_b));
    const double down_a_diff = max_abs_diff(
        *distributed_lora.down_a, reference_slice(*reference_lora.down_a));
    const double down_b_diff = max_abs_diff(
        *distributed_lora.down_b, reference_slice(*reference_lora.down_b));
    // qwen36_train_step returns each rank's local mean in sharded mode; the
    // global weighted scalar is intentionally compared through parameter/state
    // parity below rather than against the full-batch reference scalar.
    const double loss_diff = sharded ? -1.0 :
        std::abs(distributed_loss - reference_loss);

    const double gate_up_a_update = update_norm(
        *distributed_lora.gate_up_a, gate_up_a_before);
    const double gate_up_b_update = update_norm(
        *distributed_lora.gate_up_b, gate_up_b_before);
    const double down_a_update = update_norm(
        *distributed_lora.down_a, down_a_before);
    const double down_b_update = update_norm(
        *distributed_lora.down_b, down_b_before);

    constexpr int64_t optimizer_slots = 18;
    std::vector<void*> distributed_m(optimizer_slots), distributed_v(optimizer_slots);
    std::vector<void*> reference_m(optimizer_slots), reference_v(optimizer_slots);
    assert(qwen36_export_optimizer_state(distributed_ctx,
        distributed_m.data(), distributed_v.data(), optimizer_slots) == optimizer_slots);
    assert(qwen36_export_optimizer_state(reference_ctx,
        reference_m.data(), reference_v.data(), optimizer_slots) == optimizer_slots);

    double optimizer_m_diff = 0.0;
    double optimizer_v_diff = 0.0;
    for (int64_t state_idx = 14; state_idx < optimizer_slots; ++state_idx) {
        auto* local_m = reinterpret_cast<at::Tensor*>(distributed_m[state_idx]);
        auto* local_v = reinterpret_cast<at::Tensor*>(distributed_v[state_idx]);
        auto* full_m = reinterpret_cast<at::Tensor*>(reference_m[state_idx]);
        auto* full_v = reinterpret_cast<at::Tensor*>(reference_v[state_idx]);
        assert(local_m && local_v && full_m && full_v);
        optimizer_m_diff = std::max(
            optimizer_m_diff, max_abs_diff(*local_m, reference_slice(*full_m)));
        optimizer_v_diff = std::max(
            optimizer_v_diff, max_abs_diff(*local_v, reference_slice(*full_v)));
    }

    const double gate_up_a_adam_diff = first_adam_step_diff(
        *distributed_lora.gate_up_a, gate_up_a_before,
        *reinterpret_cast<at::Tensor*>(distributed_m[14]),
        *reinterpret_cast<at::Tensor*>(distributed_v[14]));
    const double gate_up_b_adam_diff = first_adam_step_diff(
        *distributed_lora.gate_up_b, gate_up_b_before,
        *reinterpret_cast<at::Tensor*>(distributed_m[15]),
        *reinterpret_cast<at::Tensor*>(distributed_v[15]));
    const double down_a_adam_diff = first_adam_step_diff(
        *distributed_lora.down_a, down_a_before,
        *reinterpret_cast<at::Tensor*>(distributed_m[16]),
        *reinterpret_cast<at::Tensor*>(distributed_v[16]));
    const double down_b_adam_diff = first_adam_step_diff(
        *distributed_lora.down_b, down_b_before,
        *reinterpret_cast<at::Tensor*>(distributed_m[17]),
        *reinterpret_cast<at::Tensor*>(distributed_v[17]));

    std::printf(
        "native_qwen36_ep_parity rank=%d world=%d top_k=2 a2a=%d sharded=%d "
        "loss_compare=%s local_tokens=%d "
        "distributed_loss=%0.8f reference_loss=%0.8f loss_diff=%0.8e "
        "gate_up_a_diff=%0.8e gate_up_b_diff=%0.8e "
        "down_a_diff=%0.8e down_b_diff=%0.8e "
        "adam_m_diff=%0.8e adam_v_diff=%0.8e "
        "adam_step_diffs=[%0.8e,%0.8e,%0.8e,%0.8e] "
        "updates=[%0.8e,%0.8e,%0.8e,%0.8e]\n",
        rank, world, a2a, sharded, sharded ? "skipped" : "direct",
        sharded ? (rank == 0 ? 1 : 3) : 2,
        distributed_loss, reference_loss, loss_diff,
        gate_up_a_diff, gate_up_b_diff, down_a_diff, down_b_diff,
        optimizer_m_diff, optimizer_v_diff,
        gate_up_a_adam_diff, gate_up_b_adam_diff,
        down_a_adam_diff, down_b_adam_diff,
        gate_up_a_update, gate_up_b_update, down_a_update, down_b_update);
    std::fflush(stdout);

    assert(gate_up_a_update > 0.0);
    assert(gate_up_b_update > 0.0);
    assert(down_a_update > 0.0);
    assert(down_b_update > 0.0);
    assert(sharded || loss_diff <= 2e-2);
    assert(gate_up_a_diff <= param_tol);
    assert(gate_up_b_diff <= param_tol);
    assert(down_a_diff <= param_tol);
    assert(down_b_diff <= param_tol);
    assert(optimizer_m_diff <= m_tol);
    assert(optimizer_v_diff <= v_tol);
    assert(gate_up_a_adam_diff <= 1e-5);
    assert(gate_up_b_adam_diff <= 1e-5);
    assert(down_a_adam_diff <= 1e-5);
    assert(down_b_adam_diff <= 1e-5);
    }

    if (sharded) {
        // Dynamic tenant rows use the source flattened token index that
        // sharded A2A already transports. Rank 0 and rank 1 deliberately swap
        // token counts [1,2]/[2,1] for the two adapters, exercising the
        // all-reduced per-adapter denominator and owner-local expert update.
        const int64_t dynamic_targets[] = {0};
        const char* dynamic_modules =
            "experts_gate_up_proj,experts_down_proj";
        const int64_t adapter_a = qwen36_add_lora(
            distributed_ctx, lora_rank, 1.0, dynamic_targets, 1,
            dynamic_modules);
        const int64_t adapter_b = qwen36_add_lora(
            distributed_ctx, lora_rank, 1.0, dynamic_targets, 1,
            dynamic_modules);
        const int64_t reference_adapter_a = qwen36_add_lora(
            reference_ctx, lora_rank, 1.0, dynamic_targets, 1,
            dynamic_modules);
        const int64_t reference_adapter_b = qwen36_add_lora(
            reference_ctx, lora_rank, 1.0, dynamic_targets, 1,
            dynamic_modules);
        assert(adapter_a > 0 && adapter_b > adapter_a);
        assert(reference_adapter_a > 0 &&
            reference_adapter_b > reference_adapter_a);
        auto dynamic_tensor = [&](void* ctx, int64_t adapter,
                                  const char* module, int b) {
            auto* ptr = qwen36_get_adapter_lora_tensor(
                ctx, adapter, 0, module, b);
            assert(ptr);
            return reinterpret_cast<at::Tensor*>(ptr);
        };
        auto initialize_dynamic_adapter = [&](int64_t distributed_adapter,
                                              int64_t reference_adapter,
                                              double offset) {
            auto set_pair = [&](const char* module, int b,
                                std::vector<int64_t> shape) {
                int64_t numel = 1;
                for (int64_t dim : shape) numel *= dim;
                auto global = ((at::arange(numel, opts) + offset) * 5e-5)
                    .reshape(shape).to(at::kBFloat16);
                auto local = global.narrow(0, rank, 1).contiguous();
                assert(qwen36_set_adapter_lora_tensor(
                    distributed_ctx, distributed_adapter, 0, module, b,
                    &local) == 0);
                assert(qwen36_set_adapter_lora_tensor(
                    reference_ctx, reference_adapter, 0, module, b,
                    &global) == 0);
            };
            set_pair("experts_gate_up_proj", 0,
                {experts, lora_rank, hidden});
            set_pair("experts_gate_up_proj", 1,
                {experts, 2 * intermediate, lora_rank});
            set_pair("experts_down_proj", 0,
                {experts, lora_rank, intermediate});
            set_pair("experts_down_proj", 1,
                {experts, hidden, lora_rank});
        };
        initialize_dynamic_adapter(
            adapter_a, reference_adapter_a, 11.0);
        initialize_dynamic_adapter(
            adapter_b, reference_adapter_b, 1011.0);
        auto* dynamic_a_gate = dynamic_tensor(
            distributed_ctx, adapter_a, "experts_gate_up_proj", 1);
        auto* dynamic_b_gate = dynamic_tensor(
            distributed_ctx, adapter_b, "experts_gate_up_proj", 1);
        auto dynamic_a_before = dynamic_a_gate->clone();
        auto dynamic_b_before = dynamic_b_gate->clone();

        auto dynamic_ids = at::tensor(
            {1, 2, 3, 4, 4, 5, 6, 7}, long_opts).reshape({2, 4});
        auto dynamic_mask = rank == 0
            ? at::tensor({0, 1, 0, 0, 0, 1, 1, 0}, float_opts)
                .reshape({2, 4})
            : at::tensor({0, 0, 1, 1, 0, 0, 0, 1}, float_opts)
                .reshape({2, 4});
        auto dynamic_attention = at::ones({2, 4}, bool_opts);
        const double dynamic_loss = qwen36_train_multi_lora(
            distributed_ctx, &dynamic_ids, &dynamic_mask,
            &dynamic_attention, 2, static_cast<int32_t>(lora_rank));
        auto reference_dynamic_mask = at::tensor(
            {0, 1, 1, 1, 0, 1, 1, 1}, float_opts).reshape({2, 4});
        const double reference_dynamic_loss = dynamic_only
            ? qwen36_train_multi_lora(
                reference_ctx, &dynamic_ids, &reference_dynamic_mask,
                &dynamic_attention, 2, static_cast<int32_t>(lora_rank))
            : -1.0;
        c10::cuda::device_synchronize();
        assert(dynamic_loss > 0.0 && std::isfinite(dynamic_loss));
        assert(!dynamic_only || (reference_dynamic_loss > 0.0 &&
            std::isfinite(reference_dynamic_loss)));
        assert(qwen36_get_adapter_step_count(distributed_ctx, adapter_a) == 1);
        assert(qwen36_get_adapter_step_count(distributed_ctx, adapter_b) == 1);
        const double dynamic_update_a = update_norm(
            *dynamic_a_gate, dynamic_a_before);
        const double dynamic_update_b = update_norm(
            *dynamic_b_gate, dynamic_b_before);
        assert(dynamic_update_a > 0.0 && dynamic_update_b > 0.0);

        double dynamic_param_diff = -1.0;
        double dynamic_m_diff = -1.0;
        double dynamic_v_diff = -1.0;
        if (dynamic_only) {
            const auto reference_slice = [rank](const at::Tensor& tensor) {
                return tensor.narrow(0, rank, 1);
            };
            auto optimizer_tensor = [](void* ctx, int64_t adapter,
                                       const char* module, int b, int is_v) {
                auto* ptr = qwen36_get_adapter_optimizer_tensor(
                    ctx, adapter, 0, module, b, is_v);
                assert(ptr);
                return reinterpret_cast<at::Tensor*>(ptr);
            };
            const int64_t distributed_adapters[] = {adapter_a, adapter_b};
            const int64_t reference_adapters[] = {
                reference_adapter_a, reference_adapter_b};
            dynamic_param_diff = 0.0;
            dynamic_m_diff = 0.0;
            dynamic_v_diff = 0.0;
            for (int adapter_index = 0; adapter_index < 2; ++adapter_index) {
                for (const char* module : {
                         "experts_gate_up_proj", "experts_down_proj"}) {
                    auto* distributed_b = dynamic_tensor(
                        distributed_ctx, distributed_adapters[adapter_index],
                        module, 1);
                    auto* reference_b = dynamic_tensor(
                        reference_ctx, reference_adapters[adapter_index],
                        module, 1);
                    dynamic_param_diff = std::max(
                        dynamic_param_diff,
                        max_abs_diff(*distributed_b,
                            reference_slice(*reference_b)));
                    for (int is_v = 0; is_v < 2; ++is_v) {
                        auto* distributed_state = optimizer_tensor(
                            distributed_ctx, distributed_adapters[adapter_index],
                            module, 1, is_v);
                        auto* reference_state = optimizer_tensor(
                            reference_ctx, reference_adapters[adapter_index],
                            module, 1, is_v);
                        double diff = max_abs_diff(
                            *distributed_state,
                            reference_slice(*reference_state));
                        if (is_v) dynamic_v_diff = std::max(dynamic_v_diff, diff);
                        else dynamic_m_diff = std::max(dynamic_m_diff, diff);
                    }
                }
            }
            std::fprintf(stderr,
                "native_qwen36_dynamic_reference rank=%d "
                "param_diff=%0.8e m_diff=%0.8e v_diff=%0.8e\n",
                rank, dynamic_param_diff, dynamic_m_diff, dynamic_v_diff);
            assert(dynamic_param_diff <= 2e-3);
            assert(dynamic_m_diff <= 5e-3);
            assert(dynamic_v_diff <= 1e-5);
        }

        // A tenant with no supervised tokens on any rank is a valid no-op:
        // it must not abort the logical step or advance its private Adam
        // clock. This also covers the final chunk when n_max is smaller than
        // the registry size.
        const int64_t adapter_c = qwen36_add_lora(
            distributed_ctx, lora_rank, 1.0, dynamic_targets, 1,
            dynamic_modules);
        assert(adapter_c > adapter_b);
        auto* dynamic_c_gate = dynamic_tensor(
            distributed_ctx, adapter_c, "experts_gate_up_proj", 1);
        auto dynamic_c_before = dynamic_c_gate->clone();
        auto zero_ids = at::tensor({
            1, 2, 3, 4, 4, 5, 6, 7, 1, 2, 3, 4}, long_opts)
            .reshape({3, 4});
        auto zero_mask = at::tensor({
            0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}, float_opts)
            .reshape({3, 4});
        auto zero_attention = at::ones({3, 4}, bool_opts);
        const double zero_loss = qwen36_train_multi_lora(
            distributed_ctx, &zero_ids, &zero_mask,
            &zero_attention, 3, static_cast<int32_t>(lora_rank));
        assert(zero_loss > 0.0 && std::isfinite(zero_loss));
        assert(qwen36_get_adapter_step_count(distributed_ctx, adapter_a) == 2);
        assert(qwen36_get_adapter_step_count(distributed_ctx, adapter_b) == 2);
        assert(qwen36_get_adapter_step_count(distributed_ctx, adapter_c) == 0);
        assert(update_norm(*dynamic_c_gate, dynamic_c_before) == 0.0);

        // Ordinary single-adapter entry points must not silently pick an
        // arbitrary tenant or crash in a batched BMM after registration.
        auto ordinary_ids = dynamic_ids.narrow(0, 0, 1).contiguous();
        auto ordinary_mask = dynamic_mask.narrow(0, 0, 1).contiguous();
        auto ordinary_attention = dynamic_attention.narrow(0, 0, 1).contiguous();
        assert(qwen36_train_step(
            distributed_ctx, &ordinary_ids, &ordinary_mask,
            &ordinary_attention) < 0.0);
        assert(qwen36_get_adapter_step_count(distributed_ctx, adapter_a) == 2);
        assert(qwen36_get_adapter_step_count(distributed_ctx, adapter_b) == 2);
        std::printf(
            "native_qwen36_dynamic_sharded rank=%d world=%d loss=%0.8f "
            "adapter_steps=[%ld,%ld,%ld] updates=[%0.8e,%0.8e] "
            "zero_loss=%0.8f reference_loss=%0.8f "
            "param_diff=%0.8e m_diff=%0.8e v_diff=%0.8e\n",
            rank, world, dynamic_loss,
            (long)qwen36_get_adapter_step_count(distributed_ctx, adapter_a),
            (long)qwen36_get_adapter_step_count(distributed_ctx, adapter_b),
            (long)qwen36_get_adapter_step_count(distributed_ctx, adapter_c),
            dynamic_update_a, dynamic_update_b, zero_loss,
            reference_dynamic_loss, dynamic_param_diff,
            dynamic_m_diff, dynamic_v_diff);
        std::fflush(stdout);
    }

    qwen36_free_training_context(reference_ctx);
    qwen36_free_training_context(distributed_ctx);
    return 0;
}
