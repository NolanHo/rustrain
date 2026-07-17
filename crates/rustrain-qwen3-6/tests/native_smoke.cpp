#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
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
extern "C" void qwen36_set_nccl_comm(void*, void*, void*, int32_t, int32_t);
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
extern "C" int64_t qwen36_export_optimizer_state(
    void*, void**, void**, int64_t);
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
extern "C" void* qwen36_get_adapter_optimizer_tensor(
    void*, int64_t, int64_t, const char*, int32_t, int32_t);
extern "C" void qwen36_free_training_context(void*);

static at::Tensor cuda_rand(std::initializer_list<int64_t> shape) {
    return at::randn(shape, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
}

static std::string dp_smoke_sync_dir() {
    const char* run_id = std::getenv("RUSTRAIN_NCCL_RUN_ID");
    std::string sanitized = run_id && run_id[0] ? run_id : "native-dp-smoke";
    for (char& ch : sanitized) {
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
              (ch >= '0' && ch <= '9') || ch == '-' || ch == '_')) {
            ch = '_';
        }
    }
    mkdir("/tmp/rustrain-nccl", 0777);
    std::string path = "/tmp/rustrain-nccl/" + sanitized;
    mkdir(path.c_str(), 0777);
    return path;
}

static void write_tensor_file(const std::string& path, const at::Tensor& tensor) {
    auto cpu = tensor.to(at::kCPU).to(at::kFloat).contiguous();
    const std::string temporary_path = path + ".tmp." +
        std::to_string(static_cast<long long>(getpid()));
    FILE* file = std::fopen(temporary_path.c_str(), "wb");
    assert(file);
    const int64_t rows = cpu.size(0);
    const int64_t cols = cpu.size(1);
    assert(std::fwrite(&rows, sizeof(rows), 1, file) == 1);
    assert(std::fwrite(&cols, sizeof(cols), 1, file) == 1);
    assert(std::fwrite(cpu.data_ptr<float>(), sizeof(float), cpu.numel(), file) ==
        static_cast<size_t>(cpu.numel()));
    std::fclose(file);
    assert(std::rename(temporary_path.c_str(), path.c_str()) == 0);
}

static at::Tensor read_tensor_file(const std::string& path) {
    FILE* file = nullptr;
    for (int attempt = 0; attempt < 6000 && !file; ++attempt) {
        file = std::fopen(path.c_str(), "rb");
        if (!file) usleep(10000);
    }
    assert(file);
    int64_t rows = 0;
    int64_t cols = 0;
    assert(std::fread(&rows, sizeof(rows), 1, file) == 1);
    assert(std::fread(&cols, sizeof(cols), 1, file) == 1);
    auto result = at::empty({rows, cols}, at::TensorOptions().dtype(at::kFloat));
    assert(std::fread(result.data_ptr<float>(), sizeof(float), result.numel(), file) ==
        static_cast<size_t>(result.numel()));
    std::fclose(file);
    return result;
}

struct DpAdapterInitial {
    at::Tensor shared_a;
    at::Tensor shared_b;
    at::Tensor expert_a;
    at::Tensor expert_b;
};

static DpAdapterInitial capture_dp_adapter(void* ctx, int64_t adapter_id) {
    auto* shared_a = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_id, 0, "shared_gate_proj", 0));
    auto* shared_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_id, 0, "shared_gate_proj", 1));
    auto* expert_a = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_id, 0, "experts_gate_up_proj", 0));
    auto* expert_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_id, 0, "experts_gate_up_proj", 1));
    assert(shared_a && shared_b && expert_a && expert_b);
    return {
        shared_a->clone(), shared_b->clone(),
        expert_a->clone(), expert_b->clone()
    };
}

static void restore_dp_adapter(
    void* ctx, int64_t adapter_id, const DpAdapterInitial& initial
) {
    auto shared_a = initial.shared_a;
    auto shared_b = initial.shared_b;
    auto expert_a = initial.expert_a;
    auto expert_b = initial.expert_b;
    assert(qwen36_set_adapter_lora_tensor(
        ctx, adapter_id, 0, "shared_gate_proj", 0, &shared_a) == 0);
    assert(qwen36_set_adapter_lora_tensor(
        ctx, adapter_id, 0, "shared_gate_proj", 1, &shared_b) == 0);
    assert(qwen36_set_adapter_lora_tensor(
        ctx, adapter_id, 0, "experts_gate_up_proj", 0, &expert_a) == 0);
    assert(qwen36_set_adapter_lora_tensor(
        ctx, adapter_id, 0, "experts_gate_up_proj", 1, &expert_b) == 0);
}

static at::Tensor dp_adapter_b_delta(
    void* ctx, int64_t adapter_id, const DpAdapterInitial& initial
) {
    auto* shared_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_id, 0, "shared_gate_proj", 1));
    auto* expert_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_id, 0, "experts_gate_up_proj", 1));
    assert(shared_b && expert_b);
    return at::cat({
        (*shared_b - initial.shared_b).to(at::kFloat).reshape({-1}),
        (*expert_b - initial.expert_b).to(at::kFloat).reshape({-1})
    });
}

static at::Tensor dp_adapter_b_optimizer(
    void* ctx, int64_t adapter_id, bool is_v
) {
    auto* shared_state = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_optimizer_tensor(
            ctx, adapter_id, 0, "shared_gate_proj", 1, is_v ? 1 : 0));
    auto* expert_state = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_optimizer_tensor(
            ctx, adapter_id, 0, "experts_gate_up_proj", 1, is_v ? 1 : 0));
    assert(shared_state && expert_state);
    assert(shared_state->scalar_type() == at::kFloat);
    assert(expert_state->scalar_type() == at::kFloat);
    return at::cat({
        shared_state->reshape({-1}), expert_state->reshape({-1})});
}

static at::Tensor dp_adapter_initial_b(const DpAdapterInitial& initial) {
    return at::cat({
        initial.shared_b.to(at::kFloat).reshape({-1}),
        initial.expert_b.to(at::kFloat).reshape({-1})
    });
}

static int run_dynamic_dp_smoke(
    std::vector<void*>& weight_ptrs,
    at::Tensor& embed, at::Tensor& final_norm, at::Tensor& lm_head,
    LayerConfig& config, int process_rank, int64_t lora_rank,
    int64_t vocab, int64_t intermediate, int64_t experts
) {
    constexpr double learning_rate = 1e-3;
    constexpr double adam_eps = 1e-8;
    constexpr double beta1 = 0.9;
    constexpr double beta2 = 0.999;
    const int64_t target_layer = 0;
    const char* targets = "shared_gate_proj,experts_gate_up_proj";
    auto create_context = [&]() {
        return qwen36_create_training_context(
            weight_ptrs.data(), static_cast<int64_t>(weight_ptrs.size()),
            &embed, &final_norm, &lm_head, &config, 1,
            static_cast<int32_t>(at::kBFloat16),
            1.0, learning_rate, beta1, beta2, adam_eps,
            vocab, 1e-5, lora_rank, &target_layer, 1, targets);
    };
    auto add_adapters = [&](void* ctx) {
        std::vector<int64_t> ids;
        ids.push_back(qwen36_add_lora(
            ctx, lora_rank, 1.0, &target_layer, 1, targets));
        ids.push_back(qwen36_add_lora(
            ctx, lora_rank, 1.0, &target_layer, 1, targets));
        assert(ids[0] > 0 && ids[1] > ids[0]);
        return ids;
    };

    auto input_ids = process_rank == 0
        ? at::tensor({1, 2, 3, 4, 4, 3, 2, 1},
            at::TensorOptions().device(at::kCUDA).dtype(at::kLong))
        : at::tensor({2, 4, 1, 3, 3, 1, 4, 2},
            at::TensorOptions().device(at::kCUDA).dtype(at::kLong));
    input_ids = input_ids.reshape({2, 4});
    auto target_mask = process_rank == 0
        ? at::tensor({1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0},
            at::TensorOptions().device(at::kCUDA).dtype(at::kFloat))
        : at::tensor({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0},
            at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    target_mask = target_mask.reshape({2, 4});
    auto attention_mask = at::ones({2, 4},
        at::TensorOptions().device(at::kCUDA).dtype(at::kBool));

    // First obtain each rank's independently-normalized local first moment.
    // At the first step m=(1-beta1)*g, so token weighting remains linear and
    // can be checked without inferring gradients from BF16 parameter deltas.
    void* local_ctx = create_context();
    assert(local_ctx);
    auto local_ids = add_adapters(local_ctx);
    std::vector<DpAdapterInitial> initial;
    initial.push_back(capture_dp_adapter(local_ctx, local_ids[0]));
    initial.push_back(capture_dp_adapter(local_ctx, local_ids[1]));
    const double local_loss = qwen36_train_multi_lora(
        local_ctx, &input_ids, &target_mask, &attention_mask, 2, lora_rank);
    assert(local_loss == local_loss && local_loss > 0.0);
    c10::cuda::device_synchronize();
    auto local_gradients = at::stack({
        dp_adapter_b_optimizer(local_ctx, local_ids[0], false),
        dp_adapter_b_optimizer(local_ctx, local_ids[1], false)
    }).to(at::kCPU);
    qwen36_free_training_context(local_ctx);

    const std::string sync_dir = dp_smoke_sync_dir();
    const std::string local_path = sync_dir + "/dp-local-" +
        std::to_string(process_rank) + ".bin";
    const std::string peer_path = sync_dir + "/dp-local-" +
        std::to_string(1 - process_rank) + ".bin";
    std::remove(local_path.c_str());
    write_tensor_file(local_path, local_gradients);
    auto peer_gradients = read_tensor_file(peer_path);
    assert(peer_gradients.sizes() == local_gradients.sizes());

    void* distributed_ctx = create_context();
    assert(distributed_ctx);
    assert(qwen36_init_nccl(distributed_ctx) == 0);
    auto distributed_ids = add_adapters(distributed_ctx);
    restore_dp_adapter(distributed_ctx, distributed_ids[0], initial[0]);
    restore_dp_adapter(distributed_ctx, distributed_ids[1], initial[1]);
    const double distributed_loss = qwen36_train_multi_lora(
        distributed_ctx, &input_ids, &target_mask, &attention_mask,
        2, lora_rank);
    assert(distributed_loss == distributed_loss && distributed_loss > 0.0);
    c10::cuda::device_synchronize();
    auto distributed_deltas = at::stack({
        dp_adapter_b_delta(
            distributed_ctx, distributed_ids[0], initial[0]),
        dp_adapter_b_delta(
            distributed_ctx, distributed_ids[1], initial[1])
    }).to(at::kCPU);
    auto distributed_gradients = at::stack({
        dp_adapter_b_optimizer(distributed_ctx, distributed_ids[0], false),
        dp_adapter_b_optimizer(distributed_ctx, distributed_ids[1], false)
    }).to(at::kCPU);
    auto distributed_v = at::stack({
        dp_adapter_b_optimizer(distributed_ctx, distributed_ids[0], true),
        dp_adapter_b_optimizer(distributed_ctx, distributed_ids[1], true)
    }).to(at::kCPU);
    auto initial_b = at::stack({
        dp_adapter_initial_b(initial[0]), dp_adapter_initial_b(initial[1])
    }).to(at::kCPU);

    const std::string distributed_path = sync_dir + "/dp-global-" +
        std::to_string(process_rank) + ".bin";
    const std::string peer_distributed_path = sync_dir + "/dp-global-" +
        std::to_string(1 - process_rank) + ".bin";
    std::remove(distributed_path.c_str());
    write_tensor_file(distributed_path, distributed_gradients);
    auto peer_distributed_gradients = read_tensor_file(peer_distributed_path);
    assert(at::allclose(
        distributed_gradients, peer_distributed_gradients, 0.0, 0.0));

    auto gradient0 = process_rank == 0 ? local_gradients : peer_gradients;
    auto gradient1 = process_rank == 0 ? peer_gradients : local_gradients;
    auto counts0 = at::tensor(
        {1.0, 3.0}, at::TensorOptions().dtype(at::kFloat)).reshape({2, 1});
    auto counts1 = at::tensor(
        {3.0, 1.0}, at::TensorOptions().dtype(at::kFloat)).reshape({2, 1});
    auto expected_gradient =
        (gradient0 * counts0 + gradient1 * counts1) / (counts0 + counts1);
    auto aggregate_count_gradient = (gradient0 + gradient1) * 0.5;

    auto relative_error = [&](const at::Tensor& actual,
                              const at::Tensor& expected) {
        return (actual - expected).abs().sum().item<double>() /
            std::max(expected.abs().sum().item<double>(), 1e-12);
    };
    const int64_t shared_numel = intermediate * lora_rank;
    const int64_t expert_numel = experts * 2 * intermediate * lora_rank;
    assert(distributed_gradients.size(1) == shared_numel + expert_numel);
    const double all_error = relative_error(
        distributed_gradients, expected_gradient);
    const double grouped_error = relative_error(
        distributed_gradients.narrow(1, shared_numel, expert_numel),
        expected_gradient.narrow(1, shared_numel, expert_numel));
    const double old_formula_gap =
        (expected_gradient - aggregate_count_gradient).abs().sum().item<double>();
    auto expected_v = distributed_gradients.square() *
        ((1.0 - beta2) / ((1.0 - beta1) * (1.0 - beta1)));
    const double v_error = relative_error(distributed_v, expected_v);
    auto expected_parameter = (
        initial_b - learning_rate *
            (distributed_gradients / (1.0 - beta1)) /
            ((distributed_v / (1.0 - beta2)).sqrt() + adam_eps))
        .to(at::kBFloat16).to(at::kFloat);
    auto expected_parameter_delta = expected_parameter - initial_b;
    const double adam_delta_max_error =
        (distributed_deltas - expected_parameter_delta)
            .abs().max().item<double>();
    const double grouped_update = distributed_deltas
        .narrow(1, shared_numel, expert_numel).abs().sum().item<double>();
    std::printf(
        "native_qwen36_dynamic_dp_weighting rank=%d loss=%0.8f "
        "relative_error=%0.6e grouped_error=%0.6e old_formula_gap=%0.6e "
        "v_error=%0.6e adam_delta_max_error=%0.6e grouped_update=%0.6e\n",
        process_rank, distributed_loss, all_error, grouped_error,
        old_formula_gap, v_error, adam_delta_max_error, grouped_update);
    assert(old_formula_gap > 1e-8);
    assert(grouped_update > 0.0);
    assert(all_error < 2e-5);
    assert(grouped_error < 2e-5);
    assert(v_error < 2e-5);
    assert(adam_delta_max_error < 2e-5);
    qwen36_free_training_context(distributed_ctx);
    return 0;
}

int main() {
    assert(qwen36_kernel_abi_version() == 19);
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

    const bool data_parallel = std::getenv("RUSTRAIN_DATA_PARALLEL") &&
        std::strcmp(std::getenv("RUSTRAIN_DATA_PARALLEL"), "0") != 0;
    if (world == 2 && tp_size == 1 && data_parallel) {
        return run_dynamic_dp_smoke(
            weight_ptrs, embed, final_norm, lm_head, config,
            process_rank, rank, vocab, intermediate, experts);
    }

    const int64_t target_layer = 0;
    // Native C++ must reject a mixed TP/DP/EP topology even when Rust-side
    // validation is bypassed. The smoke process is single-rank, so simulate
    // an invalid world before creating the real context.
    if (world == 1 && tp_size == 1) {
        setenv("TP_SIZE", "2", 1);
        setenv("WORLD_SIZE", "4", 1);
        void* invalid_topology_ctx = qwen36_create_training_context(
            weight_ptrs.data(), static_cast<int64_t>(weight_ptrs.size()),
            &embed, &final_norm, &lm_head, &config, 1,
            static_cast<int32_t>(at::kBFloat16),
            1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, rank,
            &target_layer, 1, "experts_gate_up_proj,experts_down_proj");
        assert(invalid_topology_ctx == nullptr);
        setenv("TP_SIZE", "1", 1);
        setenv("WORLD_SIZE", "1", 1);
    }
    if (world == 1 && tp_size == 1) {
        setenv("PP_SIZE", "2", 1);
        void* invalid_pp_ctx = qwen36_create_training_context(
            weight_ptrs.data(), static_cast<int64_t>(weight_ptrs.size()),
            &embed, &final_norm, &lm_head, &config, 1,
            static_cast<int32_t>(at::kBFloat16),
            1.0, 1e-3, 0.9, 0.999, 1e-8, vocab, 1e-5, rank,
            &target_layer, 1, "experts_gate_up_proj,experts_down_proj");
        assert(invalid_pp_ctx == nullptr);
        unsetenv("PP_SIZE");
    }
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
    std::vector<void*> fixed_m(2 * count);
    std::vector<void*> fixed_v(2 * count);
    assert(qwen36_export_optimizer_state(
        ctx, fixed_m.data(), fixed_v.data(), 2 * count) == 2 * count);
    auto* expert_b_m = reinterpret_cast<at::Tensor*>(fixed_m[2 * 7 + 1]);
    auto* expert_b_v = reinterpret_cast<at::Tensor*>(fixed_v[2 * 7 + 1]);
    assert(expert_b_m && expert_b_v);
    auto expected_expert_b = (
        expert_b_before.to(at::kFloat) - 1e-3 *
            (expert_b_m->to(at::kFloat) / (1.0 - 0.9)) /
            ((expert_b_v->to(at::kFloat) / (1.0 - 0.999)).sqrt() + 1e-8))
        .to(at::kBFloat16);
    const double fixed_adam_max_error =
        (*expert_b - expected_expert_b).abs().max().item<double>();
    std::printf("native_qwen36_moe_lora_smoke expert_a_update=%0.8e expert_b_update=%0.8e\n",
                update_norm, b_update_norm);
    assert(update_norm > 0.0 || b_update_norm > 0.0);
    std::printf(
        "native_qwen36_fixed_adam_oracle max_error=%0.8e\n",
        fixed_adam_max_error);
    assert(fixed_adam_max_error == 0.0);

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
    auto dynamic_b_value = at::full(
        dynamic_b->sizes(), 0.01, dynamic_b->options());
    assert(qwen36_set_adapter_lora_tensor(
        ctx, adapter_one, 0, "shared_gate_proj", 1, &dynamic_b_value) == 0);
    auto dynamic_b_before = dynamic_b->clone();
    auto* dynamic_b_two = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_two, 0, "shared_gate_proj", 1));
    assert(dynamic_b_two && dynamic_b_two->sizes() == dynamic_b->sizes());
    auto dynamic_b_two_value = at::full(
        dynamic_b_two->sizes(), -0.01, dynamic_b_two->options());
    assert(qwen36_set_adapter_lora_tensor(
        ctx, adapter_two, 0, "shared_gate_proj", 1, &dynamic_b_two_value) == 0);
    auto dynamic_b_two_before = dynamic_b_two->clone();
    auto* dynamic_expert_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_lora_tensor(
            ctx, adapter_one, 0, "experts_gate_up_proj", 1));
    assert(dynamic_expert_b && dynamic_expert_b->sizes() ==
        at::IntArrayRef({experts, 2 * intermediate, local_lora_rank}));
    auto dynamic_expert_b_value = at::full(
        dynamic_expert_b->sizes(), 0.01, dynamic_expert_b->options());
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
    // A globally empty tenant is a successful no-op: active tenants still
    // update, while the empty tenant preserves parameters, optimizer state,
    // and its private Adam clock.
    auto zero_target_mask = multi_target_mask.clone();
    zero_target_mask.select(0, 1).zero_();
    auto empty_tenant_b_before = dynamic_b_two->clone();
    auto* empty_tenant_m = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_optimizer_tensor(
            ctx, adapter_two, 0, "shared_gate_proj", 1, 0));
    auto* empty_tenant_v = reinterpret_cast<at::Tensor*>(
        qwen36_get_adapter_optimizer_tensor(
            ctx, adapter_two, 0, "shared_gate_proj", 1, 1));
    assert(empty_tenant_m && empty_tenant_v);
    auto empty_tenant_m_before = empty_tenant_m->clone();
    auto empty_tenant_v_before = empty_tenant_v->clone();
    assert(qwen36_train_multi_lora(
        ctx, &multi_input_ids, &zero_target_mask, &multi_attention_mask,
        2, rank) > 0.0);
    c10::cuda::device_synchronize();
    assert(qwen36_get_adapter_step_count(ctx, adapter_one) == 3);
    assert(qwen36_get_adapter_step_count(ctx, adapter_two) == 1);
    assert((*dynamic_b_two - empty_tenant_b_before).abs().max().item<double>() == 0.0);
    assert((*empty_tenant_m - empty_tenant_m_before).abs().max().item<double>() == 0.0);
    assert((*empty_tenant_v - empty_tenant_v_before).abs().max().item<double>() == 0.0);
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
    auto dense_b_value = at::full(
        dense_b->sizes(), 0.01, dense_b->options());
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
    auto dynamic_linear_b_value = at::full(
        dynamic_linear_b->sizes(), 0.01, dynamic_linear_b->options());
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
