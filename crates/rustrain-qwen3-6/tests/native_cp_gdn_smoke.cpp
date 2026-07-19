#include <ATen/ATen.h>

#include <algorithm>
#include <array>
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
extern "C" int32_t qwen36_set_lora_tensor(void*, int64_t, int32_t, void*);
extern "C" void* qwen36_get_lora_a(void*, int64_t);
extern "C" void* qwen36_get_lora_b(void*, int64_t);
extern "C" int64_t qwen36_export_optimizer_state(
    void*, void**, void**, int64_t);
extern "C" int64_t qwen36_get_step_count(void*);
extern "C" double qwen36_train_step(void*, void*, void*, void*);
extern "C" double qwen36_eval_step(void*, void*, void*, void*);
extern "C" double qwen36_eval_step_host_i64(
    void*, const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t);
extern "C" void qwen36_set_checkpoint(void*, int32_t, int64_t);
extern "C" void qwen36_free_training_context(void*);

namespace {

constexpr int64_t kAbiVersion = 29;
constexpr int64_t kHidden = 32;
constexpr int64_t kIntermediate = 48;
constexpr int64_t kKeyHeads = 2;
constexpr int64_t kValueHeads = 4;
constexpr int64_t kKeyDim = 128;
constexpr int64_t kValueDim = 128;
constexpr int64_t kConvKernel = 4;
constexpr int64_t kVocab = 64;
constexpr int64_t kLoraRank = 4;
constexpr int64_t kPairsPerLayer = 8;
constexpr int64_t kQSize = kKeyHeads * kKeyDim;
constexpr int64_t kVSize = kValueHeads * kValueDim;
constexpr int64_t kQkvSize = 2 * kQSize + kVSize;
constexpr int64_t kBatch = 2;
constexpr int64_t kSequence = 8;

constexpr std::array<const char*, 5> kModules = {
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"};

static at::Tensor fingerprint(
    std::initializer_list<int64_t> shape, double scale, int64_t offset
) {
    int64_t count = 1;
    for (const int64_t dim : shape) count *= dim;
    auto values = at::arange(
        count, at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));
    return ((values.add(offset).remainder(97) - 48.0) * scale)
        .reshape(shape).to(at::kBFloat16);
}

static at::Tensor ones(std::initializer_list<int64_t> shape) {
    return at::ones(
        shape, at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
}

static std::vector<at::Tensor> make_weights() {
    std::vector<at::Tensor> weights;
    weights.reserve(14);
    weights.push_back(ones({kHidden}));
    weights.push_back(ones({kHidden}));
    auto q = fingerprint({kQSize, kHidden}, 0.00035, 11);
    auto k = fingerprint({kQSize, kHidden}, 0.00041, 211);
    auto v = fingerprint({kVSize, kHidden}, 0.00029, 421);
    weights.push_back(at::cat({q, k, v}, 0).contiguous());
    weights.push_back(fingerprint({kVSize, kHidden}, 0.00031, 631));
    weights.push_back(fingerprint({kValueHeads, kHidden}, 0.00043, 719));
    weights.push_back(fingerprint({kValueHeads, kHidden}, 0.00047, 811));
    weights.push_back(fingerprint({kValueHeads}, 0.0008, 907));
    weights.push_back(fingerprint({kValueHeads}, 0.0007, 953));
    auto q_conv = fingerprint({kQSize, 1, kConvKernel}, 0.0009, 101);
    auto k_conv = fingerprint({kQSize, 1, kConvKernel}, 0.0011, 307);
    auto v_conv = fingerprint({kVSize, 1, kConvKernel}, 0.0007, 509);
    weights.push_back(at::cat({q_conv, k_conv, v_conv}, 0).contiguous());
    weights.push_back(ones({kValueDim}));
    weights.push_back(fingerprint({kHidden, kVSize}, 0.00033, 613));
    weights.push_back(fingerprint({kIntermediate, kHidden}, 0.00045, 701));
    weights.push_back(fingerprint({kIntermediate, kHidden}, 0.00039, 797));
    weights.push_back(fingerprint({kHidden, kIntermediate}, 0.00037, 887));
    return weights;
}

static std::vector<void*> pointers(std::vector<at::Tensor>& tensors) {
    std::vector<void*> result;
    result.reserve(tensors.size());
    for (auto& tensor : tensors) result.push_back(&tensor);
    return result;
}

static LayerConfig make_config() {
    LayerConfig config{};
    config.layer_type = 1;
    config.num_k_heads = kKeyHeads;
    config.key_dim = kKeyDim;
    config.num_v_heads = kValueHeads;
    config.val_dim = kValueDim;
    config.conv_kernel = kConvKernel;
    config.rms_eps = 1e-5;
    config.intermediate_size = kIntermediate;
    return config;
}

struct LoraFixture {
    int64_t slot;
    const char* module;
    at::Tensor a;
    at::Tensor b;
};

static int64_t input_size(const char* module) {
    return std::string(module) == "out_proj" ? kVSize : kHidden;
}

static int64_t output_size(const char* module) {
    const std::string name(module);
    if (name == "in_proj_qkv") return kQkvSize;
    if (name == "in_proj_z") return kVSize;
    if (name == "in_proj_a" || name == "in_proj_b") return kValueHeads;
    assert(name == "out_proj");
    return kHidden;
}

static std::vector<LoraFixture> make_lora() {
    std::vector<LoraFixture> fixtures;
    fixtures.reserve(kModules.size());
    for (int64_t slot = 0; slot < static_cast<int64_t>(kModules.size()); ++slot) {
        const char* module = kModules[slot];
        fixtures.push_back({
            slot,
            module,
            fingerprint({kLoraRank, input_size(module)}, 0.0007, 1201 + slot * 31),
            fingerprint({output_size(module), kLoraRank}, 0.0006, 1601 + slot * 37),
        });
    }
    return fixtures;
}

struct Batch {
    at::Tensor ids;
    at::Tensor targets;
    at::Tensor attention;
};

static Batch make_right_padded_batch() {
    std::vector<int64_t> ids(kBatch * kSequence);
    std::vector<float> targets(kBatch * kSequence, 1.0f);
    std::vector<uint8_t> attention(kBatch * kSequence, 1);
    for (int64_t batch = 0; batch < kBatch; ++batch) {
        for (int64_t token = 0; token < kSequence; ++token) {
            ids[batch * kSequence + token] =
                1 + (batch * 17 + token * 5) % (kVocab - 1);
        }
    }
    for (int64_t token = 6; token < kSequence; ++token) {
        targets[kSequence + token] = 0.0f;
        attention[kSequence + token] = 0;
    }
    auto ids_tensor = at::from_blob(
        ids.data(), {kBatch, kSequence},
        at::TensorOptions().device(at::kCPU).dtype(at::kLong)).clone().to(at::kCUDA);
    auto targets_tensor = at::from_blob(
        targets.data(), {kBatch, kSequence},
        at::TensorOptions().device(at::kCPU).dtype(at::kFloat)).clone().to(at::kCUDA);
    auto attention_tensor = at::from_blob(
        attention.data(), {kBatch, kSequence},
        at::TensorOptions().device(at::kCPU).dtype(at::kByte)).clone()
        .to(at::kCUDA).to(at::kBool);
    return {std::move(ids_tensor), std::move(targets_tensor),
        std::move(attention_tensor)};
}

static double max_diff(const at::Tensor& lhs, const at::Tensor& rhs) {
    assert(lhs.sizes() == rhs.sizes());
    return (lhs.to(at::kFloat) - rhs.to(at::kFloat))
        .abs().max().item<double>();
}

static void set_parallel_environment(int rank, int world, int cp_rank, int cp_size) {
    setenv("WORLD_SIZE", std::to_string(world).c_str(), 1);
    setenv("RANK", std::to_string(rank).c_str(), 1);
    setenv("TP_SIZE", "1", 1);
    setenv("CP_SIZE", std::to_string(cp_size).c_str(), 1);
    setenv("EP_SIZE", "1", 1);
    setenv("DP_SIZE", "1", 1);
    setenv("PP_SIZE", "1", 1);
    setenv("RUSTRAIN_TP_RANK", "0", 1);
    setenv("RUSTRAIN_CP_RANK", std::to_string(cp_rank).c_str(), 1);
    setenv("RUSTRAIN_EP_RANK", "0", 1);
    setenv("RUSTRAIN_DP_RANK", "0", 1);
    setenv("RUSTRAIN_PP_RANK", "0", 1);
}

static void* create_context(
    std::vector<at::Tensor>& weights, at::Tensor& embedding,
    at::Tensor& final_norm, at::Tensor& lm_head, LayerConfig& config,
    bool extended
) {
    auto weight_ptrs = pointers(weights);
    const int64_t target_layer = 0;
    if (extended) {
        return qwen36_create_training_context_ex(
            weight_ptrs.data(), weight_ptrs.size(), &embedding, &final_norm,
            &lm_head, &config, 1, static_cast<int32_t>(at::kBFloat16),
            1.0, 1e-3, 0.9, 0.999, 1e-8, kVocab, 1e-5, kLoraRank,
            &target_layer, 1,
            "in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj", 0);
    }
    return qwen36_create_training_context(
        weight_ptrs.data(), weight_ptrs.size(), &embedding, &final_norm,
        &lm_head, &config, 1, static_cast<int32_t>(at::kBFloat16),
        1.0, 1e-3, 0.9, 0.999, 1e-8, kVocab, 1e-5, kLoraRank,
        &target_layer, 1,
        "in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj");
}

}  // namespace

int main() {
    assert(qwen36_kernel_abi_version() == kAbiVersion);
    const int rank = std::atoi(std::getenv("RANK"));
    const int world = std::atoi(std::getenv("WORLD_SIZE"));
    const int local_rank = std::atoi(std::getenv("LOCAL_RANK"));
    assert(world == 2 && (rank == 0 || rank == 1));
    qwen36_set_cuda_device(local_rank);
    unsetenv("QWEN36_SEQ_CHUNK");
    auto weights = make_weights();
    auto embedding = fingerprint({kVocab, kHidden}, 0.0021, 2001);
    auto final_norm = ones({kHidden});
    auto lm_head = fingerprint({kVocab, kHidden}, 0.0017, 2201);
    auto config = make_config();

    set_parallel_environment(rank, world, rank, 2);
    void* distributed = create_context(
        weights, embedding, final_norm, lm_head, config, true);
    assert(distributed);
    auto* initial_a = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_a(distributed, 0));
    auto* initial_b = reinterpret_cast<at::Tensor*>(
        qwen36_get_lora_b(distributed, 0));
    assert(initial_a && initial_b);
    initial_a->detach().fill_(rank == 0 ? 0.125 : -0.25);
    initial_b->detach().fill_(rank == 0 ? 0.25 : -0.5);

    set_parallel_environment(0, 1, 0, 1);
    void* reference = create_context(
        weights, embedding, final_norm, lm_head, config, false);
    assert(reference);

    set_parallel_environment(rank, world, rank, 2);
    assert(qwen36_init_parallel_nccl_v2(
        distributed, rank, world,
        0, 1, 0,
        rank, 2, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0) == 0);
    assert((initial_a->to(at::kFloat) - 0.125).abs().max().item<float>() == 0.0f);
    assert((initial_b->to(at::kFloat) - 0.25).abs().max().item<float>() == 0.0f);
    qwen36_set_checkpoint(distributed, 1, 1);
    qwen36_set_checkpoint(reference, 1, 1);

    auto fixtures = make_lora();
    for (auto& fixture : fixtures) {
        assert(qwen36_set_lora_tensor(
            distributed, fixture.slot, 0, &fixture.a) == 0);
        assert(qwen36_set_lora_tensor(
            distributed, fixture.slot, 1, &fixture.b) == 0);
        assert(qwen36_set_lora_tensor(
            reference, fixture.slot, 0, &fixture.a) == 0);
        assert(qwen36_set_lora_tensor(
            reference, fixture.slot, 1, &fixture.b) == 0);
    }

    auto batch = make_right_padded_batch();
    const double distributed_eval_loss = qwen36_eval_step(
        distributed, &batch.ids, &batch.targets, &batch.attention);
    const double reference_eval_loss = qwen36_eval_step(
        reference, &batch.ids, &batch.targets, &batch.attention);
    assert(std::isfinite(distributed_eval_loss) &&
        std::isfinite(reference_eval_loss));
    assert(std::abs(distributed_eval_loss - reference_eval_loss) < 1e-2);

    const double distributed_loss = qwen36_train_step(
        distributed, &batch.ids, &batch.targets, &batch.attention);
    const double reference_loss = qwen36_train_step(
        reference, &batch.ids, &batch.targets, &batch.attention);
    assert(std::isfinite(distributed_loss) && std::isfinite(reference_loss));

    constexpr int64_t optimizer_count = 2 * kPairsPerLayer;
    std::vector<void*> distributed_m(optimizer_count), distributed_v(optimizer_count);
    std::vector<void*> reference_m(optimizer_count), reference_v(optimizer_count);
    assert(qwen36_export_optimizer_state(
        distributed, distributed_m.data(), distributed_v.data(),
        optimizer_count) == optimizer_count);
    assert(qwen36_export_optimizer_state(
        reference, reference_m.data(), reference_v.data(),
        optimizer_count) == optimizer_count);

    double max_a_diff = 0.0;
    double max_b_diff = 0.0;
    double max_m_diff = 0.0;
    double max_v_diff = 0.0;
    double max_update = 0.0;
    for (const auto& fixture : fixtures) {
        auto* distributed_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(distributed, fixture.slot));
        auto* distributed_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(distributed, fixture.slot));
        auto* reference_a = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_a(reference, fixture.slot));
        auto* reference_b = reinterpret_cast<at::Tensor*>(
            qwen36_get_lora_b(reference, fixture.slot));
        assert(distributed_a && distributed_b && reference_a && reference_b);
        max_a_diff = std::max(max_a_diff, max_diff(*distributed_a, *reference_a));
        max_b_diff = std::max(max_b_diff, max_diff(*distributed_b, *reference_b));
        max_update = std::max({max_update,
            max_diff(*distributed_a, fixture.a),
            max_diff(*distributed_b, fixture.b)});

        for (int is_b = 0; is_b < 2; ++is_b) {
            const int64_t state_index = 2 * fixture.slot + is_b;
            auto* dist_m = reinterpret_cast<at::Tensor*>(distributed_m[state_index]);
            auto* dist_v = reinterpret_cast<at::Tensor*>(distributed_v[state_index]);
            auto* ref_m = reinterpret_cast<at::Tensor*>(reference_m[state_index]);
            auto* ref_v = reinterpret_cast<at::Tensor*>(reference_v[state_index]);
            assert(dist_m && dist_v && ref_m && ref_v);
            max_m_diff = std::max(max_m_diff, max_diff(*dist_m, *ref_m));
            max_v_diff = std::max(max_v_diff, max_diff(*dist_v, *ref_v));
        }
    }

    const double loss_diff = std::abs(distributed_loss - reference_loss);
    const int64_t distributed_step = qwen36_get_step_count(distributed);
    const int64_t reference_step = qwen36_get_step_count(reference);
    std::printf(
        "native_cp_gdn_smoke rank=%d eval_loss_diff=%0.8e "
        "loss_diff=%0.8e a_diff=%0.8e "
        "b_diff=%0.8e m_diff=%0.8e v_diff=%0.8e max_update=%0.8e "
        "step=%ld reference_step=%ld\n",
        rank, std::abs(distributed_eval_loss - reference_eval_loss),
        loss_diff, max_a_diff, max_b_diff, max_m_diff, max_v_diff,
        max_update, static_cast<long>(distributed_step),
        static_cast<long>(reference_step));
    std::fflush(stdout);

    assert(loss_diff < 1e-2);
    assert(max_a_diff <= 2.1e-3 && max_b_diff <= 2.1e-3);
    assert(max_m_diff < 5e-4 && max_v_diff < 5e-7);
    assert(max_update > 0.0);
    assert(distributed_step == 1 && reference_step == 1);

    if (rank == 1) setenv("QWEN36_SEQ_CHUNK", "4", 1);
    assert(qwen36_eval_step(
        distributed, &batch.ids, &batch.targets, &batch.attention) < 0.0);
    unsetenv("QWEN36_SEQ_CHUNK");
    if (rank == 1) batch.ids[0][0].fill_(batch.ids[0][0].item<int64_t>() + 1);
    assert(qwen36_eval_step(
        distributed, &batch.ids, &batch.targets, &batch.attention) < 0.0);

    std::vector<int64_t> host_ids(kBatch * kSequence, 1);
    std::vector<int64_t> host_targets(kBatch * kSequence, 1);
    std::vector<int64_t> host_attention(kBatch * kSequence, 1);
    const int64_t* host_input = rank == 1 ? nullptr : host_ids.data();
    assert(qwen36_eval_step_host_i64(
        distributed, host_input, host_targets.data(), host_attention.data(),
        kBatch, kSequence) < 0.0);

    qwen36_free_training_context(reference);
    qwen36_free_training_context(distributed);
    return 0;
}
