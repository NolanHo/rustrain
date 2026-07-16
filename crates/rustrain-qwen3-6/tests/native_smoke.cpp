#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <cassert>
#include <cmath>
#include <cstdint>
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
extern "C" int64_t qwen36_kernel_abi_version();
extern "C" int32_t qwen36_init_nccl(void*);
extern "C" int64_t qwen36_get_lora_count(void*);
extern "C" void* qwen36_get_lora_a(void*, int64_t);
extern "C" void* qwen36_get_lora_b(void*, int64_t);
extern "C" void* qwen36_get_lora_grad_accumulator(void*, int64_t, int32_t);
extern "C" int32_t qwen36_abort_gradient_accumulation(void*);
extern "C" int32_t qwen36_set_lora_tensor(void*, int64_t, int32_t, void*);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" double qwen36_train_micro_step(
    void*, void*, void*, void*, double, int32_t);
extern "C" int64_t qwen36_get_step_count(void*);
extern "C" int32_t qwen36_set_step_count(void*, int64_t);
extern "C" int64_t qwen36_get_adapter_step_count(void*, int64_t);
extern "C" int32_t qwen36_set_adapter_step_count(void*, int64_t, int64_t);
extern "C" double qwen36_eval_step(void*, void*, void*, void*);
extern "C" double qwen36_train_multi_lora(
    void*, void*, void*, void*, int32_t, int32_t);
extern "C" double qwen36_train_multi_lora_selected(
    void*, void*, void*, void*, const int64_t*, int32_t, int32_t);
extern "C" int64_t qwen36_add_lora(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" void* qwen36_get_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t);
extern "C" int32_t qwen36_set_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t, void*);
extern "C" void qwen36_free_training_context(void*);

static at::Tensor cuda_rand(std::initializer_list<int64_t> shape) {
    return at::randn(shape, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
}

int main() {
    assert(qwen36_kernel_abi_version() == 11);
    const int world = std::atoi(std::getenv("WORLD_SIZE") ? std::getenv("WORLD_SIZE") : "1");
    const int process_rank = std::atoi(std::getenv("RANK") ? std::getenv("RANK") : "0");
    const int local_rank = std::atoi(std::getenv("LOCAL_RANK") ? std::getenv("LOCAL_RANK") : "0");
    assert(world == 1 || (world == 2 && process_rank >= 0 && process_rank < world));
    const int tp_size = std::atoi(std::getenv("TP_SIZE") ? std::getenv("TP_SIZE") : "1");
    assert(tp_size == 1 || (tp_size == world && tp_size == 2));
    c10::cuda::CUDAGuard guard(local_rank);
    at::manual_seed(7);

    constexpr int64_t hidden = 16;
    constexpr int64_t vocab = 8;
    constexpr int64_t experts = 2;
    constexpr int64_t head_dim = 8;
    constexpr int64_t intermediate = 8;
    constexpr int64_t rank = 8;
    const int64_t local_lora_rank = rank / tp_size;

    // One full-attention MoE layer. Shapes intentionally match the native
    // weight order used by build_weight_ptrs/kernel.cpp.
    std::vector<at::Tensor> weights;
    weights.push_back(cuda_rand({hidden}));                 // input RMSNorm
    weights.push_back(cuda_rand({hidden}));                 // post-attention RMSNorm
    weights.push_back(cuda_rand({2 * head_dim, hidden}));   // q_proj
    weights.push_back(cuda_rand({head_dim}));               // q_norm
    weights.push_back(cuda_rand({head_dim, hidden}));       // k_proj
    weights.push_back(cuda_rand({head_dim}));               // k_norm
    weights.push_back(cuda_rand({head_dim, hidden}));       // v_proj
    weights.push_back(cuda_rand({hidden, head_dim}));       // o_proj
    weights.push_back(cuda_rand({experts, hidden}));        // router
    weights.push_back(cuda_rand({1, hidden}));              // shared expert gate
    weights.push_back(cuda_rand({intermediate, hidden}));   // shared gate
    weights.push_back(cuda_rand({intermediate, hidden}));   // shared up
    weights.push_back(cuda_rand({hidden, intermediate}));   // shared down
    weights.push_back(cuda_rand({experts, 2 * intermediate, hidden}));
    weights.push_back(cuda_rand({experts, hidden, intermediate}));
    for (auto& weight : weights) weight.set_requires_grad(false);

    auto embed = cuda_rand({vocab, hidden});
    auto final_norm = cuda_rand({hidden});
    auto lm_head = cuda_rand({vocab, hidden});
    embed.set_requires_grad(false);
    final_norm.set_requires_grad(false);
    lm_head.set_requires_grad(false);

    std::vector<void*> weight_ptrs;
    weight_ptrs.reserve(weights.size());
    for (auto& weight : weights) weight_ptrs.push_back(&weight);
    LayerConfig config{};
    config.layer_type = 0;
    config.num_heads = 1;
    config.num_kv_heads = 1;
    config.head_dim = head_dim;
    config.num_k_heads = 0;
    config.key_dim = 0;
    config.num_v_heads = 0;
    config.val_dim = 0;
    config.conv_kernel = 0;
    config.partial_rotary_factor = 1.0;
    config.rope_theta = 10000.0;
    config.rms_eps = 1e-5;
    config.num_experts = experts;
    config.top_k = 1;
    config.moe_intermediate = intermediate;
    config.expert_start = 0;
    config.expert_count = experts;
    config.intermediate_size = 0;
    config.norm_topk_prob = 1;
    config.nccl_comm = nullptr;
    config.nccl_stream = nullptr;

    const int64_t target_layer = 0;
    void* ctx = qwen36_create_training_context(
        weight_ptrs.data(), static_cast<int64_t>(weight_ptrs.size()),
        &embed, &final_norm, &lm_head, &config, 1,
        static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, rank,
        &target_layer, 1, "experts_gate_up_proj,experts_down_proj");
    if (!ctx) return 2;
    if (world > 1) assert(qwen36_init_nccl(ctx) == 0);

    const int64_t count = qwen36_get_lora_count(ctx);
    assert(count == 9);
    auto* expert_a = reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(ctx, 7));
    auto* expert_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(ctx, 7));
    auto* down_a = reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(ctx, 8));
    auto* down_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(ctx, 8));
    assert(expert_a && expert_b && down_a && down_b);
    assert(expert_a->sizes() == at::IntArrayRef({experts, local_lora_rank, hidden}));
    assert(expert_b->sizes() == at::IntArrayRef({experts, 2 * intermediate, local_lora_rank}));
    assert(down_a->sizes() == at::IntArrayRef({experts, local_lora_rank, intermediate}));
    assert(down_b->sizes() == at::IntArrayRef({experts, hidden, local_lora_rank}));

    // Make both B tensors nonzero so the step exercises the LoRA branches.
    auto expert_b_value = at::ones(expert_b->sizes(), expert_b->options());
    auto down_b_value = at::ones(down_b->sizes(), down_b->options());
    assert(qwen36_set_lora_tensor(ctx, 7, 1, &expert_b_value) == 0);
    assert(qwen36_set_lora_tensor(ctx, 8, 1, &down_b_value) == 0);
    auto expert_a_before = expert_a->clone();
    auto expert_b_before = expert_b->clone();

    auto input_ids = at::arange(1, 3, at::TensorOptions().device(at::kCUDA).dtype(at::kLong)).reshape({1, 2});
    auto target_mask = at::ones({1, 2}, at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    auto attention_mask = at::ones({1, 2}, at::TensorOptions().device(at::kCUDA).dtype(at::kBool));
    // Distinct tenant rows exercise the production [n_total, seq] path. The
    // rows are intentionally different so a repeated batch-1 implementation
    // cannot satisfy the assertions below.
    auto multi_input_ids = at::tensor({1, 2, 3, 4},
        at::TensorOptions().device(at::kCUDA).dtype(at::kLong)).reshape({2, 2});
    auto multi_target_mask = at::ones({2, 2},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    auto multi_attention_mask = at::ones({2, 2},
        at::TensorOptions().device(at::kCUDA).dtype(at::kBool));

    // The optimized and fallback routed-expert paths must agree before the
    // optimizer changes any parameters. On older libtorch both calls use the
    // fallback, preserving the same compatibility check.
    setenv("QWEN36_DISABLE_GROUPED_MM", "1", 1);
    const double fallback_loss =
        qwen36_eval_step(ctx, &input_ids, &target_mask, &attention_mask);
    unsetenv("QWEN36_DISABLE_GROUPED_MM");
    setenv("QWEN36_REPORT_GROUPED_MM", "1", 1);
    const double grouped_loss =
        qwen36_eval_step(ctx, &input_ids, &target_mask, &attention_mask);
    std::printf(
        "native_qwen36_moe_lora_parity fallback=%0.8f grouped=%0.8f diff=%0.8e\n",
        fallback_loss, grouped_loss, std::abs(fallback_loss - grouped_loss));
    assert(fallback_loss > 0.0 && grouped_loss > 0.0);
    assert(std::abs(fallback_loss - grouped_loss) <= 2e-2);

    const double loss = qwen36_train_step(ctx, &input_ids, &target_mask, &attention_mask);
    c10::cuda::device_synchronize();
    std::printf("native_qwen36_moe_lora_smoke loss=%0.8f\n", loss);
    assert(loss == loss);
    assert(loss > 0.0);
    const double update_norm = (*expert_a - expert_a_before).abs().sum().item<double>();
    const double b_update_norm = (*expert_b - expert_b_before).abs().sum().item<double>();
    std::printf("native_qwen36_moe_lora_smoke expert_a_update=%0.8e expert_b_update=%0.8e\n",
                update_norm, b_update_norm);
    assert(update_norm > 0.0 || b_update_norm > 0.0);

    // Dynamic multi-adapter batches must apply shared-expert MLP LoRA per
    // sample and preserve the parameter updates after chunk registry restore.
    const char* shared_targets =
        "shared_gate_proj,shared_up_proj,shared_down_proj,"
        "experts_gate_up_proj,experts_down_proj";
    const int64_t adapter_one = qwen36_add_lora(
        ctx, rank, 1.0, &target_layer, 1, shared_targets);
    const int64_t adapter_two = qwen36_add_lora(
        ctx, rank, 1.0, &target_layer, 1, shared_targets);
    assert(adapter_one > 0 && adapter_two > adapter_one);
    assert(qwen36_get_adapter_step_count(ctx, adapter_one) == 0);
    assert(qwen36_get_adapter_step_count(ctx, adapter_two) == 0);
    assert(qwen36_set_adapter_step_count(ctx, adapter_one, -1) != 0);
    assert(qwen36_set_adapter_step_count(ctx, adapter_one, 4) == 0);
    assert(qwen36_get_adapter_step_count(ctx, adapter_one) == 4);
    assert(qwen36_set_adapter_step_count(ctx, adapter_one, 0) == 0);
    auto* dynamic_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_one, 0, "shared_gate_proj", 1));
    assert(dynamic_b && dynamic_b->sizes() == at::IntArrayRef({intermediate, local_lora_rank}));
    auto dynamic_b_value = at::ones(dynamic_b->sizes(), dynamic_b->options());
    assert(qwen36_set_adapter_lora_tensor(
        ctx, adapter_one, 0, "shared_gate_proj", 1, &dynamic_b_value) == 0);
    auto dynamic_b_before = dynamic_b->clone();
    auto* dynamic_b_two = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_two, 0, "shared_gate_proj", 1));
    assert(dynamic_b_two && dynamic_b_two->sizes() == dynamic_b->sizes());
    auto dynamic_b_two_value = at::full(dynamic_b_two->sizes(), -1.0, dynamic_b_two->options());
    assert(qwen36_set_adapter_lora_tensor(
        ctx, adapter_two, 0, "shared_gate_proj", 1, &dynamic_b_two_value) == 0);
    auto dynamic_b_two_before = dynamic_b_two->clone();
    auto* dynamic_expert_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_one, 0, "experts_gate_up_proj", 1));
    assert(dynamic_expert_b && dynamic_expert_b->sizes() ==
        at::IntArrayRef({experts, 2 * intermediate, local_lora_rank}));
    auto dynamic_expert_b_value = at::ones(
        dynamic_expert_b->sizes(), dynamic_expert_b->options());
    assert(qwen36_set_adapter_lora_tensor(
        ctx, adapter_one, 0, "experts_gate_up_proj", 1,
        &dynamic_expert_b_value) == 0);
    auto dynamic_expert_b_before = dynamic_expert_b->clone();
    const double multi_loss = qwen36_train_multi_lora(
        ctx, &multi_input_ids, &multi_target_mask, &multi_attention_mask, 2, rank);
    c10::cuda::device_synchronize();
    assert(qwen36_get_adapter_step_count(ctx, adapter_one) == 1);
    assert(qwen36_get_adapter_step_count(ctx, adapter_two) == 1);
    dynamic_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_one, 0, "shared_gate_proj", 1));
    assert(dynamic_b);
    dynamic_expert_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_one, 0, "experts_gate_up_proj", 1));
    assert(dynamic_expert_b);
    const double dynamic_update =
        (*dynamic_b - dynamic_b_before).abs().sum().item<double>();
    const double dynamic_expert_update =
        (*dynamic_expert_b - dynamic_expert_b_before).abs().sum().item<double>();
    dynamic_b_two = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_two, 0, "shared_gate_proj", 1));
    assert(dynamic_b_two);
    const double dynamic_two_update =
        (*dynamic_b_two - dynamic_b_two_before).abs().sum().item<double>();
    std::printf(
        "native_qwen36_multi_lora_smoke loss=%0.8f shared_gate_b_update=%0.8e "
        "expert_gate_up_b_update=%0.8e adapter_two_gate_b_update=%0.8e\n",
        multi_loss, dynamic_update, dynamic_expert_update, dynamic_two_update);
    assert(multi_loss == multi_loss && multi_loss > 0.0);
    assert(dynamic_update > 0.0);
    assert(dynamic_expert_update > 0.0);
    assert(dynamic_two_update > 0.0);

    auto adapter_one_before_selected = dynamic_b->clone();
    auto adapter_two_before_selected = dynamic_b_two->clone();
    auto selected_input_ids = multi_input_ids.narrow(0, 0, 1).contiguous();
    auto selected_target_mask = multi_target_mask.narrow(0, 0, 1).contiguous();
    auto selected_attention_mask = multi_attention_mask.narrow(0, 0, 1).contiguous();
    const int64_t selected_adapter_ids[] = {adapter_one};
    const double selected_loss = qwen36_train_multi_lora_selected(
        ctx, &selected_input_ids, &selected_target_mask,
        &selected_attention_mask, selected_adapter_ids, 1, rank);
    c10::cuda::device_synchronize();
    assert(selected_loss == selected_loss && selected_loss > 0.0);
    assert(qwen36_get_adapter_step_count(ctx, adapter_one) == 2);
    assert(qwen36_get_adapter_step_count(ctx, adapter_two) == 1);
    dynamic_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_one, 0, "shared_gate_proj", 1));
    dynamic_b_two = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_two, 0, "shared_gate_proj", 1));
    assert(dynamic_b && dynamic_b_two);
    const double selected_update =
        (*dynamic_b - adapter_one_before_selected).abs().sum().item<double>();
    const double unselected_update =
        (*dynamic_b_two - adapter_two_before_selected).abs().sum().item<double>();
    std::printf(
        "native_qwen36_selected_multi_lora_smoke loss=%0.8f "
        "selected_update=%0.8e unselected_update=%0.8e\n",
        selected_loss, selected_update, unselected_update);
    assert(selected_update > 0.0);
    assert(unselected_update == 0.0);

    const int64_t unknown_adapter_ids[] = {adapter_two + 1000};
    assert(qwen36_train_multi_lora_selected(
        ctx, &selected_input_ids, &selected_target_mask,
        &selected_attention_mask, unknown_adapter_ids, 1, rank) < 0.0);
    assert(qwen36_get_adapter_step_count(ctx, adapter_one) == 2);
    assert(qwen36_get_adapter_step_count(ctx, adapter_two) == 1);
    assert(qwen36_get_adapter_lora_tensor(
        ctx, adapter_one, 0, "shared_gate_proj", 1) != nullptr);
    assert(qwen36_get_adapter_lora_tensor(
        ctx, adapter_two, 0, "shared_gate_proj", 1) != nullptr);
    qwen36_free_training_context(ctx);

    // Dense Qwen3.5 variants use the same per-sample activation path for
    // gate/up/down projections.
    std::vector<at::Tensor> dense_weights(weights.begin(), weights.begin() + 8);
    dense_weights.push_back(cuda_rand({intermediate, hidden}));
    dense_weights.push_back(cuda_rand({intermediate, hidden}));
    dense_weights.push_back(cuda_rand({hidden, intermediate}));
    weight_ptrs.clear();
    for (auto& weight : dense_weights) weight_ptrs.push_back(&weight);
    LayerConfig dense_config = config;
    dense_config.num_experts = 0;
    dense_config.top_k = 0;
    dense_config.moe_intermediate = 0;
    dense_config.expert_count = 0;
    dense_config.intermediate_size = intermediate;
    ctx = qwen36_create_training_context(
        weight_ptrs.data(), static_cast<int64_t>(weight_ptrs.size()),
        &embed, &final_norm, &lm_head, &dense_config, 1,
        static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, rank,
        &target_layer, 1, "q_proj");
    assert(ctx);
    if (world > 1) assert(qwen36_init_nccl(ctx) == 0);
    const char* dense_targets = "gate_proj,up_proj,down_proj";
    const int64_t dense_one = qwen36_add_lora(
        ctx, rank, 1.0, &target_layer, 1, dense_targets);
    const int64_t dense_two = qwen36_add_lora(
        ctx, rank, 1.0, &target_layer, 1, dense_targets);
    assert(dense_one > 0 && dense_two > dense_one);
    auto* dense_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(ctx, dense_one, 0, "gate_proj", 1));
    assert(dense_b && dense_b->sizes() == at::IntArrayRef({intermediate, local_lora_rank}));
    auto dense_b_value = at::ones(dense_b->sizes(), dense_b->options());
    assert(qwen36_set_adapter_lora_tensor(
        ctx, dense_one, 0, "gate_proj", 1, &dense_b_value) == 0);
    auto dense_b_before = dense_b->clone();
    const double dense_loss = qwen36_train_multi_lora(
        ctx, &multi_input_ids, &multi_target_mask, &multi_attention_mask, 2, rank);
    c10::cuda::device_synchronize();
    dense_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(ctx, dense_one, 0, "gate_proj", 1));
    assert(dense_b);
    const double dense_update =
        (*dense_b - dense_b_before).abs().sum().item<double>();
    std::printf(
        "native_qwen35_dense_multi_lora_smoke loss=%0.8f gate_b_update=%0.8e\n",
        dense_loss, dense_update);
    assert(dense_loss == dense_loss && dense_loss > 0.0);
    assert(dense_update > 0.0);
    qwen36_free_training_context(ctx);

    // Qwen3.5/3.6 linear-attention layers use the model's actual 128-wide
    // delta-rule state. Exercise both fixed and per-sample dynamic GDN LoRA,
    // including the custom CUDA backward for q/k/v/g/beta.
    constexpr int64_t linear_heads = 1;
    constexpr int64_t linear_dim = 128;
    constexpr int64_t linear_qkv = 3 * linear_dim;
    std::vector<at::Tensor> linear_weights;
    linear_weights.push_back(cuda_rand({hidden}));
    linear_weights.push_back(cuda_rand({hidden}));
    linear_weights.push_back(cuda_rand({linear_qkv, hidden}));
    linear_weights.push_back(cuda_rand({linear_dim, hidden}));
    linear_weights.push_back(cuda_rand({linear_heads, hidden}));
    linear_weights.push_back(cuda_rand({linear_heads, hidden}));
    linear_weights.push_back(cuda_rand({linear_heads}));
    linear_weights.push_back(cuda_rand({linear_heads}));
    linear_weights.push_back(cuda_rand({linear_qkv, 1, 4}));
    linear_weights.push_back(cuda_rand({linear_dim}));
    linear_weights.push_back(cuda_rand({hidden, linear_dim}));
    linear_weights.push_back(cuda_rand({intermediate, hidden}));
    linear_weights.push_back(cuda_rand({intermediate, hidden}));
    linear_weights.push_back(cuda_rand({hidden, intermediate}));
    for (auto& weight : linear_weights) weight.set_requires_grad(false);
    weight_ptrs.clear();
    for (auto& weight : linear_weights) weight_ptrs.push_back(&weight);
    LayerConfig linear_config = dense_config;
    linear_config.layer_type = 1;
    linear_config.num_heads = 0;
    linear_config.num_kv_heads = 0;
    linear_config.head_dim = 0;
    linear_config.num_k_heads = linear_heads;
    linear_config.key_dim = linear_dim;
    linear_config.num_v_heads = linear_heads;
    linear_config.val_dim = linear_dim;
    linear_config.conv_kernel = 4;
    ctx = qwen36_create_training_context(
        weight_ptrs.data(), static_cast<int64_t>(weight_ptrs.size()),
        &embed, &final_norm, &lm_head, &linear_config, 1,
        static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, rank,
        &target_layer, 1,
        "in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj");
    assert(ctx);
    if (world > 1) assert(qwen36_init_nccl(ctx) == 0);
    assert(qwen36_get_lora_count(ctx) == 8);
    auto* linear_a = reinterpret_cast<at::Tensor*>(qwen36_get_lora_a(ctx, 0));
    auto* linear_b = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(ctx, 0));
    assert(linear_a && linear_b);
    assert(linear_a->sizes() == at::IntArrayRef({local_lora_rank, hidden}));
    assert(linear_b->sizes() == at::IntArrayRef({linear_qkv, local_lora_rank}));
    auto linear_b_value = at::ones(linear_b->sizes(), linear_b->options());
    assert(qwen36_set_lora_tensor(ctx, 0, 1, &linear_b_value) == 0);
    auto linear_a_before = linear_a->clone();
    auto* linear_a_accum = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_grad_accumulator(ctx, 0, 0));
    assert(linear_a_accum);
    assert(linear_a_accum->scalar_type() == at::kFloat);
    assert(linear_a_accum->sizes() == linear_a->sizes());
    assert(linear_a_accum->abs().sum().item<double>() == 0.0);
    assert(qwen36_get_step_count(ctx) == 0);
    assert(qwen36_set_step_count(ctx, -1) != 0);
    assert(qwen36_get_step_count(ctx) == 0);
    assert(qwen36_set_step_count(ctx, 7) == 0);
    assert(qwen36_get_step_count(ctx) == 7);
    assert(qwen36_set_step_count(ctx, 0) == 0);
    const double accum_loss_0 = qwen36_train_micro_step(
        ctx, &input_ids, &target_mask, &attention_mask, 0.5, 0);
    c10::cuda::device_synchronize();
    assert(accum_loss_0 == accum_loss_0);
    assert(qwen36_get_step_count(ctx) == 0);
    assert((*linear_a - linear_a_before).abs().sum().item<double>() == 0.0);
    assert(linear_a_accum->abs().sum().item<double>() > 0.0);
    // The BF16 leaf gradient is consumed at the micro-step boundary; only the
    // FP32 accumulator may survive.
    assert(!linear_a->grad().defined());

    // A failing final micro-step aborts the existing window transactionally.
    const double failed_accum = qwen36_train_micro_step(
        ctx, &input_ids, &target_mask, &attention_mask, NAN, 1);
    c10::cuda::device_synchronize();
    assert(failed_accum < 0.0);
    assert(qwen36_get_step_count(ctx) == 0);
    assert(linear_a_accum->abs().sum().item<double>() == 0.0);
    assert((*linear_a - linear_a_before).abs().sum().item<double>() == 0.0);

    // Explicit abort is idempotent and clears a successful non-final micro.
    assert(qwen36_train_micro_step(
        ctx, &input_ids, &target_mask, &attention_mask, 0.5, 0) > 0.0);
    c10::cuda::device_synchronize();
    assert(linear_a_accum->abs().sum().item<double>() > 0.0);
    assert(qwen36_abort_gradient_accumulation(ctx) == 0);
    c10::cuda::device_synchronize();
    assert(linear_a_accum->abs().sum().item<double>() == 0.0);
    assert(qwen36_get_step_count(ctx) == 0);
    assert((*linear_a - linear_a_before).abs().sum().item<double>() == 0.0);

    // Two clean micro-batches accumulate into FP32 and commit one Adam step.
    const double clean_accum_loss_0 = qwen36_train_micro_step(
        ctx, &input_ids, &target_mask, &attention_mask, 0.5, 0);
    c10::cuda::device_synchronize();
    assert(clean_accum_loss_0 == clean_accum_loss_0);
    assert(linear_a_accum->scalar_type() == at::kFloat);
    assert(linear_a_accum->abs().sum().item<double>() > 0.0);
    const double accum_loss_1 = qwen36_train_micro_step(
        ctx, &input_ids, &target_mask, &attention_mask, 0.5, 1);
    c10::cuda::device_synchronize();
    assert(accum_loss_1 == accum_loss_1);
    assert(qwen36_get_step_count(ctx) == 1);
    assert((*linear_a - linear_a_before).abs().sum().item<double>() > 0.0);
    assert(linear_a_accum->abs().sum().item<double>() == 0.0);
    linear_a_before = linear_a->clone();
    const double linear_loss = qwen36_train_step(
        ctx, &input_ids, &target_mask, &attention_mask);
    c10::cuda::device_synchronize();
    const double linear_update =
        (*linear_a - linear_a_before).abs().sum().item<double>();
    std::printf(
        "native_qwen35_linear_lora_smoke loss=%0.8f qkv_a_update=%0.8e\n",
        linear_loss, linear_update);
    assert(linear_loss == linear_loss && linear_loss > 0.0);
    assert(linear_update > 0.0);

    const int64_t linear_adapter_one = qwen36_add_lora(
        ctx, rank, 1.0, &target_layer, 1, "in_proj_qkv");
    const int64_t linear_adapter_two = qwen36_add_lora(
        ctx, rank, 1.0, &target_layer, 1, "in_proj_qkv");
    assert(linear_adapter_one > 0 && linear_adapter_two > linear_adapter_one);
    auto* dynamic_linear_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, linear_adapter_one, 0, "in_proj_qkv", 1));
    assert(dynamic_linear_b && dynamic_linear_b->sizes() ==
        at::IntArrayRef({linear_qkv, local_lora_rank}));
    auto dynamic_linear_b_value = at::ones(
        dynamic_linear_b->sizes(), dynamic_linear_b->options());
    assert(qwen36_set_adapter_lora_tensor(
        ctx, linear_adapter_one, 0, "in_proj_qkv", 1,
        &dynamic_linear_b_value) == 0);
    auto dynamic_linear_b_before = dynamic_linear_b->clone();
    const double dynamic_linear_loss = qwen36_train_multi_lora(
        ctx, &multi_input_ids, &multi_target_mask, &multi_attention_mask, 2, rank);
    c10::cuda::device_synchronize();
    dynamic_linear_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, linear_adapter_one, 0, "in_proj_qkv", 1));
    assert(dynamic_linear_b);
    const double dynamic_linear_update =
        (*dynamic_linear_b - dynamic_linear_b_before).abs().sum().item<double>();
    std::printf(
        "native_qwen35_linear_multi_lora_smoke loss=%0.8f qkv_b_update=%0.8e\n",
        dynamic_linear_loss, dynamic_linear_update);
    assert(dynamic_linear_loss == dynamic_linear_loss && dynamic_linear_loss > 0.0);
    assert(dynamic_linear_update > 0.0);
    qwen36_free_training_context(ctx);
    return 0;
}
