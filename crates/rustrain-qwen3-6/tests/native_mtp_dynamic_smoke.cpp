#include <ATen/ATen.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
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
extern "C" int32_t qwen36_set_mtp_weights(
    void*, void*, void*, void*, void*, void**, int64_t, void*, int64_t);
extern "C" int64_t qwen36_add_lora(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" int32_t qwen36_set_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t, void*);
extern "C" void* qwen36_get_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t);
extern "C" int32_t qwen36_train_multi_lora_selected_v3(
    void*, void*, void*, void*, const int64_t*, int32_t,
    double*, double*, int32_t);
extern "C" void qwen36_free_training_context(void*);

static at::Tensor values(std::initializer_list<int64_t> shape, double scale) {
    int64_t count = 1;
    for (const auto dim : shape) count *= dim;
    return ((at::arange(count, at::TensorOptions().device(at::kCUDA)
        .dtype(at::kFloat)).remainder(23) - 11.0) * scale)
        .reshape(shape).to(at::kBFloat16);
}

static std::vector<void*> pointers(std::vector<at::Tensor>& tensors) {
    std::vector<void*> result;
    result.reserve(tensors.size());
    for (auto& tensor : tensors) result.push_back(&tensor);
    return result;
}

static double max_diff(const at::Tensor& lhs, const at::Tensor& rhs) {
    return (lhs.to(at::kFloat) - rhs.to(at::kFloat)).abs().max().item<double>();
}

static void install_q_lora(void* ctx, int64_t adapter, const at::Tensor& a,
                           const at::Tensor& b) {
    auto a_copy = a;
    auto b_copy = b;
    assert(qwen36_set_adapter_lora_tensor(
        ctx, adapter, 0, "q_proj", 0, &a_copy) == 0);
    assert(qwen36_set_adapter_lora_tensor(
        ctx, adapter, 0, "q_proj", 1, &b_copy) == 0);
}

static void set_environment() {
    setenv("WORLD_SIZE", "1", 1);
    setenv("RANK", "0", 1);
    setenv("LOCAL_RANK", "0", 1);
    setenv("TP_SIZE", "1", 1);
    setenv("CP_SIZE", "1", 1);
    setenv("EP_SIZE", "1", 1);
    setenv("DP_SIZE", "1", 1);
    setenv("PP_SIZE", "1", 1);
    unsetenv("RUSTRAIN_DATA_PARALLEL");
    unsetenv("QWEN36_DISABLE_MTP");
    unsetenv("QWEN36_MTP_LOSS_SCALE");
}

int main() {
    set_environment();
    qwen36_set_cuda_device(0);

    constexpr int64_t hidden = 8;
    constexpr int64_t heads = 4;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t head_dim = 2;
    constexpr int64_t intermediate = 12;
    constexpr int64_t vocab = 16;
    constexpr int64_t rank = 2;

    std::vector<at::Tensor> weights;
    weights.push_back(at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    weights.push_back(at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    weights.push_back(values({2 * heads * head_dim, hidden}, .010));
    weights.push_back(at::ones({head_dim}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    weights.push_back(values({kv_heads * head_dim, hidden}, .012));
    weights.push_back(at::ones({head_dim}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16)));
    weights.push_back(values({kv_heads * head_dim, hidden}, .008));
    weights.push_back(values({hidden, heads * head_dim}, .011));
    weights.push_back(values({intermediate, hidden}, .009));
    weights.push_back(values({intermediate, hidden}, .007));
    weights.push_back(values({hidden, intermediate}, .010));
    for (auto& tensor : weights) tensor.set_requires_grad(false);
    auto embed = values({vocab, hidden}, .020);
    auto final_norm = at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
    auto lm_head = values({vocab, hidden}, .015);
    embed.set_requires_grad(false);
    final_norm.set_requires_grad(false);
    lm_head.set_requires_grad(false);

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
    auto weight_ptrs = pointers(weights);

    void* ctx = qwen36_create_training_context(
        weight_ptrs.data(), weight_ptrs.size(), &embed, &final_norm, &lm_head,
        &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, .9, .999, 1e-8, vocab, 1e-5, rank,
        &target_layer, 1, "q_proj");
    assert(ctx);

    auto mtp_fc = values({hidden, 2 * hidden}, .006);
    auto mtp_pre_emb = at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
    auto mtp_pre_hidden = at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
    auto mtp_norm = at::ones({hidden}, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
    mtp_fc.set_requires_grad(false);
    mtp_pre_emb.set_requires_grad(false);
    mtp_pre_hidden.set_requires_grad(false);
    mtp_norm.set_requires_grad(false);
    auto mtp_weight_ptrs = pointers(weights);
    assert(qwen36_set_mtp_weights(
        ctx, &mtp_fc, &mtp_pre_emb, &mtp_pre_hidden, &mtp_norm,
        mtp_weight_ptrs.data(), mtp_weight_ptrs.size(), &config, 1) == 0);

    auto q_a = values({rank, hidden}, .002);
    auto q_b = values({2 * heads * head_dim, rank}, .001);
    const int64_t adapter_one = qwen36_add_lora(
        ctx, rank, 2.0, &target_layer, 1, "q_proj");
    const int64_t adapter_two = qwen36_add_lora(
        ctx, rank, 2.0, &target_layer, 1, "q_proj");
    const int64_t adapter_three = qwen36_add_lora(
        ctx, rank, 2.0, &target_layer, 1, "q_proj");
    assert(adapter_one > 0 && adapter_two > adapter_one &&
        adapter_three > adapter_two);
    install_q_lora(ctx, adapter_one, q_a, q_b);
    install_q_lora(ctx, adapter_two, q_a * 1.3, q_b * 0.7);
    install_q_lora(ctx, adapter_three, q_a * 0.4, q_b * 1.6);
    auto* unselected_a_before_ptr = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(ctx, adapter_three, 0, "q_proj", 0));
    auto* unselected_b_before_ptr = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(ctx, adapter_three, 0, "q_proj", 1));
    assert(unselected_a_before_ptr && unselected_b_before_ptr);
    auto unselected_a_before = unselected_a_before_ptr->clone();
    auto unselected_b_before = unselected_b_before_ptr->clone();

    // Row 0 has five main-loss tokens and four MTP tokens; row 1 has two and
    // one respectively. This makes accidental use of one shared denominator
    // observable while keeping the first row exactly comparable to a singleton.
    auto input_ids = at::tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
        at::TensorOptions().device(at::kCUDA).dtype(at::kLong)).reshape({2, 6});
    auto target_mask = at::tensor({0., 1., 1., 1., 1., 1.,
                                    0., 1., 1., 0., 0., 0.},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat)).reshape({2, 6});
    const int64_t selected_ids[] = {adapter_one, adapter_two};
    double aggregate = -1.0;
    double adapter_losses[2] = {-1.0, -1.0};
    assert(qwen36_train_multi_lora_selected_v3(
        ctx, &input_ids, &target_mask, nullptr, selected_ids, 2,
        &aggregate, adapter_losses, 2) == 0);

    void* ref = qwen36_create_training_context(
        weight_ptrs.data(), weight_ptrs.size(), &embed, &final_norm, &lm_head,
        &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, .9, .999, 1e-8, vocab, 1e-5, rank,
        &target_layer, 1, "q_proj");
    assert(ref);
    auto ref_mtp_weight_ptrs = pointers(weights);
    assert(qwen36_set_mtp_weights(
        ref, &mtp_fc, &mtp_pre_emb, &mtp_pre_hidden, &mtp_norm,
        ref_mtp_weight_ptrs.data(), ref_mtp_weight_ptrs.size(), &config, 1) == 0);
    const int64_t ref_adapter = qwen36_add_lora(
        ref, rank, 2.0, &target_layer, 1, "q_proj");
    assert(ref_adapter > 0);
    install_q_lora(ref, ref_adapter, q_a, q_b);
    auto ref_ids = input_ids.narrow(0, 0, 1).contiguous();
    auto ref_mask = target_mask.narrow(0, 0, 1).contiguous();
    const int64_t ref_selected[] = {ref_adapter};
    double ref_aggregate = -1.0;
    double ref_loss[1] = {-1.0};
    assert(qwen36_train_multi_lora_selected_v3(
        ref, &ref_ids, &ref_mask, nullptr, ref_selected, 1,
        &ref_aggregate, ref_loss, 1) == 0);

    auto* updated_a = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(ctx, adapter_one, 0, "q_proj", 0));
    auto* updated_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(ctx, adapter_one, 0, "q_proj", 1));
    auto* ref_a = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(ref, ref_adapter, 0, "q_proj", 0));
    auto* ref_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(ref, ref_adapter, 0, "q_proj", 1));
    auto* unselected_a_after_ptr = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(ctx, adapter_three, 0, "q_proj", 0));
    auto* unselected_b_after_ptr = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(ctx, adapter_three, 0, "q_proj", 1));
    assert(updated_a && updated_b && ref_a && ref_b &&
        unselected_a_after_ptr && unselected_b_after_ptr);
    const double parameter_diff = std::max(
        max_diff(*updated_a, *ref_a), max_diff(*updated_b, *ref_b));
    const double unselected_diff = std::max(
        max_diff(*unselected_a_after_ptr, unselected_a_before),
        max_diff(*unselected_b_after_ptr, unselected_b_before));
    const double loss_diff = std::abs(adapter_losses[0] - ref_loss[0]);
    std::printf(
        "native_qwen36_mtp_dynamic_smoke aggregate=%0.8g adapter0=%0.8g "
        "adapter1=%0.8g ref=%0.8g loss_diff=%0.8e parameter_diff=%0.8e "
        "unselected_diff=%0.8e\n", aggregate, adapter_losses[0],
        adapter_losses[1], ref_loss[0], loss_diff, parameter_diff,
        unselected_diff);
    std::fflush(stdout);
    assert(std::isfinite(aggregate) && std::isfinite(adapter_losses[0]) &&
        std::isfinite(adapter_losses[1]) && std::isfinite(ref_loss[0]));
    assert(loss_diff < 2e-5);
    assert(parameter_diff < 2e-3);
    assert(unselected_diff == 0.0);

    qwen36_free_training_context(ref);
    qwen36_free_training_context(ctx);
    return 0;
}
