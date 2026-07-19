#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

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

extern "C" int64_t qwen36_kernel_abi_version();
extern "C" void* qwen36_create_training_context(
    void**, int64_t, void*, void*, void*, void*, int64_t, int32_t,
    double, double, double, double, double, int64_t, double, int64_t,
    const int64_t*, int64_t, const char*);
extern "C" void* qwen36_create_training_context_v2(
    void**, int64_t, void*, void*, void*, void*, int64_t,
    int64_t, int64_t, int32_t, int32_t,
    double, double, double, double, double, int64_t, double, int64_t,
    const int64_t*, int64_t, const char*, int32_t);
extern "C" int32_t qwen36_init_parallel_nccl_v2(
    void*, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t);
extern "C" int32_t qwen36_attach_parallel_nccl_no_sync_v2(
    void*, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t);
extern "C" double qwen36_parallel_max_double(void*, double);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" double qwen36_eval_step(void*, void*, void*, void*);
extern "C" int64_t qwen36_get_lora_count(void*);
extern "C" void* qwen36_get_lora_a(void*, int64_t);
extern "C" int64_t qwen36_add_lora(
    void*, int64_t, double, const int64_t*, int64_t, const char*);
extern "C" int32_t qwen36_set_adapter_id(void*, int64_t, int64_t);
extern "C" int32_t qwen36_remove_lora(void*, int64_t);
extern "C" void* qwen36_get_adapter_lora_tensor(
    void*, int64_t, int64_t, const char*, int32_t);
extern "C" void qwen36_free_training_context(void*);

static void set_rank_environment(int rank, int cp_rank, int pp_rank) {
    const auto rank_string = std::to_string(rank);
    const auto cp_string = std::to_string(cp_rank);
    const auto pp_string = std::to_string(pp_rank);
    setenv("RANK", rank_string.c_str(), 1);
    setenv("WORLD_SIZE", "4", 1);
    setenv("TP_SIZE", "1", 1);
    setenv("CP_SIZE", "2", 1);
    setenv("EP_SIZE", "1", 1);
    setenv("DP_SIZE", "1", 1);
    setenv("PP_SIZE", "2", 1);
    setenv("RUSTRAIN_TP_RANK", "0", 1);
    setenv("RUSTRAIN_CP_RANK", cp_string.c_str(), 1);
    setenv("RUSTRAIN_EP_RANK", "0", 1);
    setenv("RUSTRAIN_DP_RANK", "0", 1);
    setenv("RUSTRAIN_PP_RANK", pp_string.c_str(), 1);
}

static void* create_empty_context(at::Tensor& embed, at::Tensor& norm,
                                  at::Tensor& lm_head) {
    return qwen36_create_training_context(
        nullptr, 0, &embed, &norm, &lm_head, nullptr, 0,
        static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, 4, 1e-6, 1,
        nullptr, 0, "");
}

static void* create_stage_context(
    int pp_rank, at::Tensor& embed, at::Tensor& norm, at::Tensor& lm_head,
    std::vector<at::Tensor>& weights, LayerConfig& config
) {
    std::vector<void*> weight_ptrs;
    for (auto& weight : weights) weight_ptrs.push_back(&weight);
    const int64_t target_layer = 0;
    return qwen36_create_training_context_v2(
        weight_ptrs.data(), static_cast<int64_t>(weight_ptrs.size()),
        pp_rank == 0 ? &embed : nullptr,
        pp_rank == 1 ? &norm : nullptr,
        pp_rank == 1 ? &lm_head : nullptr,
        &config, 1, pp_rank, 2, pp_rank == 0 ? 1 : 2,
        static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, 4, 1e-6, 1,
        &target_layer, 1, "q_proj", 0);
}

static int32_t initialize_five_axis_context(
    void* ctx, int rank, int cp_rank, int pp_rank, bool attach
) {
    const int cp_color = pp_rank * 2;  // [0,1] and [2,3]
    const int pp_color = cp_rank;      // [0,2] and [1,3]
    auto init = attach ? qwen36_attach_parallel_nccl_no_sync_v2
                       : qwen36_init_parallel_nccl_v2;
    return init(
        ctx, rank, 4,
        0, 1, 0,
        cp_rank, 2, cp_color,
        0, 1, 0,
        0, 1, 0,
        pp_rank, 2, pp_color);
}

int main() {
    assert(qwen36_kernel_abi_version() == 29);
    const int rank = std::atoi(std::getenv("RANK"));
    const int local_rank = std::atoi(std::getenv("LOCAL_RANK"));
    assert(rank >= 0 && rank < 4);
    const int cp_rank = rank % 2;
    const int pp_rank = rank / 2;
    set_rank_environment(rank, cp_rank, pp_rank);

    c10::cuda::CUDAGuard guard(local_rank);
    auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    auto embed = at::zeros({4, 4}, options);
    auto norm = at::ones({4}, options);
    auto lm_head = at::zeros({4, 4}, options);

    std::vector<at::Tensor> stage_weights = {
        at::ones({4}, options), at::ones({4}, options),
        at::zeros({8, 4}, options), at::ones({4}, options),
        at::zeros({4, 4}, options), at::ones({4}, options),
        at::zeros({4, 4}, options), at::zeros({4, 4}, options),
        at::zeros({8, 4}, options), at::zeros({8, 4}, options),
        at::zeros({4, 8}, options),
    };
    LayerConfig stage_config{};
    stage_config.layer_type = 0;
    stage_config.num_heads = 1;
    stage_config.num_kv_heads = 1;
    stage_config.head_dim = 4;
    stage_config.partial_rotary_factor = 1.0;
    stage_config.rope_theta = 10000.0;
    stage_config.rms_eps = 1e-6;
    stage_config.intermediate_size = 8;

    void* premature_shadow = create_empty_context(embed, norm, lm_head);
    assert(premature_shadow);
    assert(initialize_five_axis_context(
        premature_shadow, rank, cp_rank, pp_rank, true) == -1);
    assert(std::isnan(qwen36_parallel_max_double(premature_shadow, rank)));
    qwen36_free_training_context(premature_shadow);

    const char* configured_run_id = std::getenv("RUSTRAIN_NCCL_RUN_ID");
    const bool had_configured_run_id = configured_run_id != nullptr;
    const std::string saved_run_id = configured_run_id ? configured_run_id : "";
    setenv("RUSTRAIN_NCCL_RUN_ID", "..", 1);
    void* invalid_run_id = create_empty_context(embed, norm, lm_head);
    assert(invalid_run_id);
    assert(initialize_five_axis_context(
        invalid_run_id, rank, cp_rank, pp_rank, false) == -1);
    assert(std::isnan(qwen36_parallel_max_double(invalid_run_id, rank)));
    qwen36_free_training_context(invalid_run_id);
    if (had_configured_run_id) {
        setenv("RUSTRAIN_NCCL_RUN_ID", saved_run_id.c_str(), 1);
    } else {
        unsetenv("RUSTRAIN_NCCL_RUN_ID");
    }

    void* context = create_empty_context(embed, norm, lm_head);
    assert(context);
    assert(initialize_five_axis_context(
        context, rank, cp_rank, pp_rank, false) == 0);
    const double maximum = qwen36_parallel_max_double(context, rank);
    assert(std::isfinite(maximum));
    assert(maximum == 3.0);
    assert(qwen36_train_step(context, nullptr, nullptr, nullptr) < 0.0);
    assert(qwen36_eval_step(context, nullptr, nullptr, nullptr) < 0.0);

    void* stage_context = create_stage_context(
        pp_rank, embed, norm, lm_head, stage_weights, stage_config);
    assert(stage_context);
    assert(initialize_five_axis_context(
        stage_context, rank, cp_rank, pp_rank, true) == 0);
    assert(qwen36_get_lora_count(stage_context) == 7);
    auto* stage_lora_a = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_a(stage_context, 0));
    assert(stage_lora_a);
    if (pp_rank == 0) {
        assert(stage_lora_a->sizes() == at::IntArrayRef({1, 4}));
    } else {
        assert(stage_lora_a->dim() == 0);
    }
    // A single CP replica presenting a different canonical request must make
    // every rank fail without letting unaffected PP stages skip the request
    // consensus and deadlock.
    const int64_t mismatched_target_layer = rank == 0 ? 1 : 0;
    assert(qwen36_add_lora(
        stage_context, 4, 8.0, &mismatched_target_layer, 1, "q_proj") < 0);
    const int64_t dynamic_target_layer = 0;
    const int64_t adapter_id = qwen36_add_lora(
        stage_context, 4, 8.0, &dynamic_target_layer, 1, "q_proj");
    assert(adapter_id == 1);
    auto* dynamic_a = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            stage_context, adapter_id, dynamic_target_layer, "q_proj", 0));
    if (pp_rank == 0) {
        assert(dynamic_a);
        assert(dynamic_a->sizes() == at::IntArrayRef({4, 4}));
    } else {
        assert(dynamic_a == nullptr);
    }
    assert(qwen36_get_adapter_lora_tensor(
        stage_context, adapter_id, 1, "q_proj", 0) == nullptr);
    assert(qwen36_set_adapter_id(stage_context, adapter_id, 7) == 0);
    assert(qwen36_get_adapter_lora_tensor(
        stage_context, adapter_id, dynamic_target_layer, "q_proj", 0) ==
        nullptr);
    auto* renamed_dynamic_a = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            stage_context, 7, dynamic_target_layer, "q_proj", 0));
    if (pp_rank == 0) {
        assert(renamed_dynamic_a);
        assert(renamed_dynamic_a->sizes() == at::IntArrayRef({4, 4}));
    } else {
        assert(renamed_dynamic_a == nullptr);
    }
    assert(qwen36_remove_lora(stage_context, 7) == 1);
    assert(qwen36_get_adapter_lora_tensor(
        stage_context, 7, dynamic_target_layer, "q_proj", 0) == nullptr);

    void* shadow = create_empty_context(embed, norm, lm_head);
    assert(shadow);
    assert(initialize_five_axis_context(
        shadow, rank, cp_rank, pp_rank, true) == 0);
    assert(qwen36_parallel_max_double(shadow, rank) == 3.0);

    set_rank_environment(rank, (cp_rank + 1) % 2, pp_rank);
    void* invalid = create_empty_context(embed, norm, lm_head);
    assert(invalid);
    assert(initialize_five_axis_context(
        invalid, rank, cp_rank, pp_rank, true) == -1);
    assert(std::isnan(qwen36_parallel_max_double(invalid, rank)));

    std::printf(
        "native_qwen36_pp_cp_comm rank=%d cp=%d pp=%d max=%0.1f ok\n",
        rank, cp_rank, pp_rank, maximum);
    qwen36_free_training_context(invalid);
    qwen36_free_training_context(shadow);
    qwen36_free_training_context(stage_context);
    qwen36_free_training_context(context);
    return 0;
}
