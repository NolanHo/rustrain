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
extern "C" int32_t qwen36_set_pad_token_id(void*, int64_t);
extern "C" int32_t qwen36_set_lora_tensor(void*, int64_t, int32_t, void*);
extern "C" void* qwen36_get_lora_b(void*, int64_t);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" int64_t qwen36_get_sequence_parallel_counter(void*, int32_t);
extern "C" int64_t qwen36_add_lora(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" int32_t qwen36_set_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t, void*);
extern "C" void* qwen36_get_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t);
extern "C" void* qwen36_get_adapter_optimizer_tensor(
    void*, int64_t, int64_t, const char*, int32_t, int32_t);
extern "C" int64_t qwen36_get_adapter_step_count(void*, int64_t);
extern "C" int32_t qwen36_train_multi_lora_selected_v3(
    void*, void*, void*, void*, const int64_t*, int32_t,
    double*, double*, int32_t);
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

static double max_diff(const at::Tensor& left, const at::Tensor& right) {
    assert(left.sizes() == right.sizes());
    return left.to(at::kFloat).sub(right.to(at::kFloat)).abs().max().item<double>();
}

static void set_dynamic_pair(
    void* context, int64_t adapter, const char* module,
    at::Tensor& a, at::Tensor& b
) {
    assert(qwen36_set_adapter_lora_tensor(
        context, adapter, 0, module, 0, &a) == 0);
    assert(qwen36_set_adapter_lora_tensor(
        context, adapter, 0, module, 1, &b) == 0);
}

static void install_dynamic_fixture(
    void* context, int64_t adapter, int rank, bool sharded, double scale
) {
    constexpr int64_t hidden = 8;
    constexpr int64_t q_out = 16;
    constexpr int64_t kv_out = 4;
    constexpr int64_t intermediate = 12;
    constexpr int64_t local_q_out = q_out / 2;
    constexpr int64_t local_kv_out = kv_out / 2;
    constexpr int64_t local_intermediate = intermediate / 2;
    constexpr int64_t lora_rank = 2;

    auto q_a = values({lora_rank, hidden}, scale);
    auto q_b_full = values({q_out, lora_rank}, scale * 1.1);
    auto k_a = values({lora_rank, hidden}, scale * 1.2);
    auto k_b_full = values({kv_out, lora_rank}, scale * 1.3);
    auto v_a = values({lora_rank, hidden}, scale * 1.4);
    auto v_b_full = values({kv_out, lora_rank}, scale * 1.5);
    auto o_a_full = values({lora_rank, hidden}, scale * 1.6);
    auto o_b = values({hidden, lora_rank}, scale * 1.7);
    auto gate_a = values({lora_rank, hidden}, scale * 1.8);
    auto gate_b_full = values({intermediate, lora_rank}, scale * 1.9);
    auto up_a = values({lora_rank, hidden}, scale * 2.0);
    auto up_b_full = values({intermediate, lora_rank}, scale * 2.1);
    auto down_a_full = values({lora_rank, intermediate}, scale * 2.2);
    auto down_b = values({hidden, lora_rank}, scale * 2.3);

    auto shard_rows = [rank](const at::Tensor& tensor, int64_t rows) {
        return tensor.narrow(0, rank * rows, rows).contiguous();
    };
    auto shard_cols = [rank](const at::Tensor& tensor, int64_t cols) {
        return tensor.narrow(1, rank * cols, cols).contiguous();
    };
    auto q_b = sharded ? shard_rows(q_b_full, local_q_out) : q_b_full;
    auto k_b = sharded ? shard_rows(k_b_full, local_kv_out) : k_b_full;
    auto v_b = sharded ? shard_rows(v_b_full, local_kv_out) : v_b_full;
    auto o_a = sharded ? shard_cols(o_a_full, hidden / 2) : o_a_full;
    auto gate_b = sharded
        ? shard_rows(gate_b_full, local_intermediate) : gate_b_full;
    auto up_b = sharded
        ? shard_rows(up_b_full, local_intermediate) : up_b_full;
    auto down_a = sharded
        ? shard_cols(down_a_full, local_intermediate) : down_a_full;

    set_dynamic_pair(context, adapter, "q_proj", q_a, q_b);
    set_dynamic_pair(context, adapter, "k_proj", k_a, k_b);
    set_dynamic_pair(context, adapter, "v_proj", v_a, v_b);
    set_dynamic_pair(context, adapter, "o_proj", o_a, o_b);
    set_dynamic_pair(context, adapter, "gate_proj", gate_a, gate_b);
    set_dynamic_pair(context, adapter, "up_proj", up_a, up_b);
    set_dynamic_pair(context, adapter, "down_proj", down_a, down_b);
}

int main() {
    assert(qwen36_kernel_abi_version() == 31);
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
    assert(qwen36_set_pad_token_id(ctx, 0) == 0);
    auto* b_before_ptr = reinterpret_cast<at::Tensor*>(qwen36_get_lora_b(ctx, 0));
    assert(b_before_ptr);
    auto b_before = b_before_ptr->clone();
    auto ids = at::tensor({1, 2, 3, 4}, at::TensorOptions().device(at::kCUDA).dtype(at::kLong)).reshape({1, 4});
    auto target = at::ones({1, 4}, at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    const double loss = qwen36_train_step(ctx, &ids, &target, nullptr);
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

    // Dynamic multi-LoRA + sequence parallel oracle. Two selected tenants
    // receive rows with different token counts; the third tenant must remain
    // bitwise unchanged. The reference context owns the complete TP weights.
    auto full_ptrs = ptrs(full);
    const char* dynamic_targets =
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj";
    const int64_t dynamic_target_layer = 0;
    void* dynamic_distributed = qwen36_create_training_context_ex(
        local_ptrs.data(), local_ptrs.size(), &local_embed, &final_norm,
        &local_lm_head, &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, .9, .999, 1e-8, vocab, 1e-5, rank_lora,
        &dynamic_target_layer, 1, dynamic_targets, flags);
    assert(dynamic_distributed);
    assert(qwen36_init_parallel_nccl(
        dynamic_distributed, rank, world, rank, 2, 0,
        0, 1, 0, 0, 1, 0) == 0);
    assert(qwen36_set_pad_token_id(dynamic_distributed, 0) == 0);

    unsetenv("QWEN36_SEQUENCE_PARALLEL");
    setenv("WORLD_SIZE", "1", 1);
    setenv("RANK", "0", 1);
    setenv("TP_SIZE", "1", 1);
    setenv("RUSTRAIN_TP_RANK", "0", 1);
    void* dynamic_reference = qwen36_create_training_context_ex(
        full_ptrs.data(), full_ptrs.size(), &embed, &final_norm, &lm_head,
        &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, .9, .999, 1e-8, vocab, 1e-5, rank_lora,
        &dynamic_target_layer, 1, dynamic_targets, 0);
    assert(dynamic_reference);
    assert(qwen36_set_pad_token_id(dynamic_reference, 0) == 0);
    setenv("QWEN36_SEQUENCE_PARALLEL", "1", 1);
    setenv("WORLD_SIZE", "2", 1);
    setenv("RANK", std::to_string(rank).c_str(), 1);
    setenv("TP_SIZE", "2", 1);
    setenv("RUSTRAIN_TP_RANK", std::to_string(rank).c_str(), 1);

    std::array<int64_t, 3> distributed_adapters{};
    std::array<int64_t, 3> reference_adapters{};
    for (int64_t index = 0; index < 3; ++index) {
        distributed_adapters[index] = qwen36_add_lora(
            dynamic_distributed, rank_lora, rank_lora,
            &dynamic_target_layer, 1, dynamic_targets);
        reference_adapters[index] = qwen36_add_lora(
            dynamic_reference, rank_lora, rank_lora,
            &dynamic_target_layer, 1, dynamic_targets);
        assert(distributed_adapters[index] > 0 &&
            reference_adapters[index] > 0);
        install_dynamic_fixture(
            dynamic_distributed, distributed_adapters[index], rank, true,
            0.0008 + index * 0.00017);
        install_dynamic_fixture(
            dynamic_reference, reference_adapters[index], 0, false,
            0.0008 + index * 0.00017);
    }

    auto dynamic_ids = at::tensor(
        {1, 2, 3, 4, 5, 6, 7, 8},
        at::TensorOptions().device(at::kCUDA).dtype(at::kLong)).reshape({2, 4});
    auto dynamic_targets_mask = at::tensor(
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat)).reshape({2, 4});
    const int64_t selected_dynamic_ids[] = {
        distributed_adapters[0], distributed_adapters[1]};
    const int64_t selected_reference_ids[] = {
        reference_adapters[0], reference_adapters[1]};
    double distributed_dynamic_loss = -1.0;
    double reference_dynamic_loss = -1.0;
    std::array<double, 2> distributed_tenant_losses{-1.0, -1.0};
    std::array<double, 2> reference_tenant_losses{-1.0, -1.0};
    const std::array<const char*, 7> dynamic_modules = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"};
    auto snapshot = [](void* context, int64_t adapter, const char* module) {
        std::array<at::Tensor, 6> state;
        for (int is_b = 0; is_b < 2; ++is_b) {
            auto* parameter = reinterpret_cast<at::Tensor*>(
                qwen36_get_adapter_lora_tensor(
                    context, adapter, 0, module, is_b));
            auto* m = reinterpret_cast<at::Tensor*>(
                qwen36_get_adapter_optimizer_tensor(
                    context, adapter, 0, module, is_b, 0));
            auto* v = reinterpret_cast<at::Tensor*>(
                qwen36_get_adapter_optimizer_tensor(
                    context, adapter, 0, module, is_b, 1));
            assert(parameter && m && v);
            state[is_b] = parameter->clone();
            state[2 + is_b] = m->clone();
            state[4 + is_b] = v->clone();
        }
        return state;
    };
    std::array<std::array<at::Tensor, 6>, 7> unselected_before{};
    for (size_t module = 0; module < dynamic_modules.size(); ++module)
        unselected_before[module] = snapshot(
            dynamic_distributed, distributed_adapters[2], dynamic_modules[module]);
    assert(qwen36_train_multi_lora_selected_v3(
        dynamic_distributed, &dynamic_ids, &dynamic_targets_mask,
        nullptr, selected_dynamic_ids, 2,
        &distributed_dynamic_loss, distributed_tenant_losses.data(), 2) == 0);
    assert(qwen36_train_multi_lora_selected_v3(
        dynamic_reference, &dynamic_ids, &dynamic_targets_mask,
        nullptr, selected_reference_ids, 2,
        &reference_dynamic_loss, reference_tenant_losses.data(), 2) == 0);
    assert(std::isfinite(distributed_dynamic_loss) &&
        std::isfinite(reference_dynamic_loss));

    double dynamic_parameter_diff = 0.0;
    double dynamic_m_diff = 0.0;
    double dynamic_v_diff = 0.0;
    for (int64_t tenant = 0; tenant < 2; ++tenant) {
        for (size_t module_index = 0;
             module_index < dynamic_modules.size(); ++module_index) {
            const char* module = dynamic_modules[module_index];
            const bool column_parallel = module_index != 3 && module_index != 6;
            auto distributed = snapshot(
                dynamic_distributed, distributed_adapters[tenant], module);
            auto reference = snapshot(
                dynamic_reference, reference_adapters[tenant], module);
            const int64_t local_rows = module_index == 0 ? 8
                : (module_index == 1 || module_index == 2 ? 2
                : (module_index == 4 || module_index == 5 ? 6 : 8));
            const int64_t local_cols = module_index == 3 ? 4
                : (module_index == 6 ? 6 : 8);
            for (int state = 0; state < 6; ++state) {
                at::Tensor expected = reference[state];
                if (column_parallel && (state == 1 || state == 3 || state == 5))
                    expected = expected.narrow(0, rank * local_rows, local_rows);
                if (!column_parallel && (state == 0 || state == 2 || state == 4))
                    expected = expected.narrow(1, rank * local_cols, local_cols);
                const double difference = max_diff(distributed[state], expected);
                if (state == 0 || state == 1) dynamic_parameter_diff =
                    std::max(dynamic_parameter_diff, difference);
                else if (state == 2 || state == 3) dynamic_m_diff =
                    std::max(dynamic_m_diff, difference);
                else dynamic_v_diff = std::max(dynamic_v_diff, difference);
            }
        }
    }
    double unselected_diff = 0.0;
    for (size_t module = 0; module < dynamic_modules.size(); ++module) {
        auto after = snapshot(
            dynamic_distributed, distributed_adapters[2], dynamic_modules[module]);
        for (int state = 0; state < 6; ++state)
            unselected_diff = std::max(
                unselected_diff, max_diff(after[state], unselected_before[module][state]));
    }
    std::printf(
        "native_qwen36_sequence_parallel_dynamic_raw rank=%d dist_loss=%0.8g ref_loss=%0.8g "
        "dist_tenant=%0.8g,%0.8g ref_tenant=%0.8g,%0.8g\n",
        rank, distributed_dynamic_loss, reference_dynamic_loss,
        distributed_tenant_losses[0], distributed_tenant_losses[1],
        reference_tenant_losses[0], reference_tenant_losses[1]);
    std::fflush(stdout);
    // TP/SP changes BF16 reduction order relative to the single-rank oracle.
    // Keep the report tolerance wide enough for that rounding while state
    // comparisons below remain substantially tighter.
    assert(std::abs(distributed_dynamic_loss - reference_dynamic_loss) < 1e-2);
    for (int tenant = 0; tenant < 2; ++tenant) {
        assert(std::abs(distributed_tenant_losses[tenant] -
            reference_tenant_losses[tenant]) < 1e-2);
        assert(qwen36_get_adapter_step_count(
            dynamic_distributed, distributed_adapters[tenant]) == 1);
    }
    assert(qwen36_get_adapter_step_count(
        dynamic_distributed, distributed_adapters[2]) == 0);
    assert(dynamic_parameter_diff < 5e-3 && dynamic_m_diff < 5e-4 &&
        dynamic_v_diff < 5e-7 && unselected_diff == 0.0);
    std::printf(
        "native_qwen36_sequence_parallel_dynamic rank=%d loss_diff=%0.8g "
        "tenant_loss_diff=%0.8g param_diff=%0.8g m_diff=%0.8g v_diff=%0.8g\n",
        rank, std::abs(distributed_dynamic_loss - reference_dynamic_loss),
        std::max(std::abs(distributed_tenant_losses[0] - reference_tenant_losses[0]),
            std::abs(distributed_tenant_losses[1] - reference_tenant_losses[1])),
        dynamic_parameter_diff, dynamic_m_diff, dynamic_v_diff);
    qwen36_free_training_context(dynamic_reference);
    qwen36_free_training_context(dynamic_distributed);
    qwen36_free_training_context(ctx);
    return 0;
}
