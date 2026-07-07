// glm5_attention.cpp — C++ implementation of GLM-5.2 DSA attention
//
// Migrated from Rust (model.rs glm5_dsa_attention) to C++ for:
// - Coarse-grained kernel fusion (one FFI call per layer instead of ~30)
// - Direct CUDA stream control (for async overlap with NCCL)
// - No tch-rs dependency in compute path
//
// All intermediate tensors live on C++ stack — zero FFI crossings per operation.

#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/topk.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/transpose.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/scatter.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/log_softmax.h>
#include <ATen/ops/nll_loss.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/where.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/triu.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <optional>

extern "C" {

// ── Helper: RMSNorm ──
static at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double eps) {
    auto dtype = input.scalar_type();
    auto w = weight.to(dtype);
    // variance = mean(input^2, -1, keepdim)
    auto sq = input.pow(2.0);
    auto variance = sq.mean(-1, /*keepdim=*/true);
    auto result = input * (variance + eps).rsqrt().to(dtype) * w;
    return result.to(dtype);
}

// ── Helper: RMSNorm with bias (indexer k_norm) ──
static at::Tensor rms_norm_with_bias(const at::Tensor& input, const at::Tensor& weight,
                                       const at::Tensor& bias, double eps) {
    auto dtype = input.scalar_type();
    auto w = weight.to(dtype);
    auto b = bias.to(dtype);
    auto sq = input.pow(2.0);
    auto variance = sq.mean(-1, /*keepdim=*/true);
    auto result = input * (variance + eps).rsqrt().to(dtype) * w + b;
    return result.to(dtype);
}

// ── Helper: RoPE cos/sin ──
// ── RoPE cache: avoid recomputing cos/sin every layer (78 layers × 2 calls/layer = 156×/step) ──
#include <map>
#include <tuple>
static std::map<std::tuple<int64_t, int64_t, double, int>, std::pair<at::Tensor, at::Tensor>> g_rope_cache;

static std::pair<at::Tensor, at::Tensor> rope_cos_sin(int64_t seq_len, int64_t head_dim,
                                                       double theta, int device_id) {
    auto key = std::make_tuple(seq_len, head_dim, theta, device_id);
    auto it = g_rope_cache.find(key);
    if (it != g_rope_cache.end()) {
        return it->second;
    }
    auto device = at::Device(at::Device::Type::CUDA, device_id);
    auto positions = at::arange(seq_len, at::TensorOptions().dtype(at::kFloat).device(device));
    auto dim_indices = at::arange(head_dim / 2, at::TensorOptions().dtype(at::kFloat).device(device));
    auto inv_freq = (dim_indices * (2.0 / (double)head_dim)) * (1.0 / std::log(theta));
    inv_freq = inv_freq.exp();
    auto freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0); // [S, D/2]
    auto cos = at::cos(freqs);
    auto sin = at::sin(freqs);
    cos = at::cat({cos, cos}, -1);
    sin = at::cat({sin, sin}, -1);
    auto result = std::make_pair(cos, sin);
    g_rope_cache[key] = result;
    return result;
}

// ── Helper: apply rotary interleave ──
static at::Tensor apply_rotary_interleave(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin) {
    int64_t seq = x.size(2);
    auto c = cos.narrow(0, 0, seq).unsqueeze(0).unsqueeze(0);
    auto s = sin.narrow(0, 0, seq).unsqueeze(0).unsqueeze(0);
    int64_t half = x.size(-1) / 2;
    auto x_even = x.slice(-1, 0, at::nullopt, 2);
    auto x_odd = x.slice(-1, 1, at::nullopt, 2);
    auto rotated = at::cat({x_odd.neg(), x_even}, -1);
    return x * c + rotated * s;
}

// ── Helper: apply rotary (non-interleave) ──
static at::Tensor apply_rotary(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin) {
    int64_t seq = x.size(2);
    auto c = cos.narrow(0, 0, seq).unsqueeze(0).unsqueeze(0);
    auto s = sin.narrow(0, 0, seq).unsqueeze(0).unsqueeze(0);
    int64_t half = x.size(-1) / 2;
    auto x1 = x.narrow(-1, 0, half);
    auto x2 = x.narrow(-1, half, half);
    auto rotated = at::cat({x2.neg(), x1}, -1);
    return x * c + rotated * s;
}

// ── Helper: FP8 dequant (byte-level FP8→F32, then × scale, →BF16) ──
// Mirrors Rust's dequant_fp8_weight: expand block-wise scale [n_blocks, k_blocks]
// to [N, K], multiply, convert to target dtype.
static at::Tensor dequant_fp8(const at::Tensor& fp8_weight, const at::Tensor& scale,
                                at::ScalarType target_dtype) {
    int64_t n = fp8_weight.size(0);
    int64_t k = fp8_weight.size(1);
    auto device = fp8_weight.device();

    // Step 1: FP8 → F32 on the same device
    auto f32_weight = fp8_weight.to(at::kFloat);

    // Step 2: expand scale from [n_blocks, k_blocks] to [N, K]
    // Ensure scale is on the same device as weight
    auto scale_on_device = scale.to(device).to(at::kFloat);
    int64_t n_blocks = (n + 127) / 128;
    int64_t k_blocks = (k + 127) / 128;
    at::Tensor scale_expanded;
    if (scale_on_device.size(0) == n_blocks && scale_on_device.size(1) == k_blocks) {
        int64_t n_padded = n_blocks * 128;
        int64_t k_padded = k_blocks * 128;
        // [n_blocks, k_blocks] → [n_blocks, k_blocks, 128, 128] → [n_padded, k_padded]
        auto expanded = scale_on_device.unsqueeze(-1).unsqueeze(-1)
                              .expand({n_blocks, k_blocks, 128, 128})
                              .reshape({n_padded, k_padded})
                              .contiguous();
        // Crop to actual [N, K]
        scale_expanded = expanded.narrow(0, 0, n).narrow(1, 0, k);
    } else {
        // Scale already matches [N, K]
        scale_expanded = scale_on_device;
    }

    // Step 3: apply scale and convert to target dtype
    auto result = (f32_weight * scale_expanded).to(target_dtype);
    return result;
}

// ── Helper: FP8-safe linear ──
// If weight is FP8 and scale is provided: dequant (FP8→BF16 with scale) then at::linear.
// If weight is already BF16: ensure on same device as input, then at::linear.
static at::Tensor safe_linear(const at::Tensor& input, const at::Tensor& weight,
                               std::optional<at::Tensor> weight_scale) {
    auto dtype = input.scalar_type();
    auto device = input.device();
    if (weight_scale.has_value() && weight.scalar_type() == at::kFloat8_e4m3fn) {
        // FP8 path: dequant with scale, then linear
        auto w_bf16 = dequant_fp8(weight, weight_scale.value(), dtype);
        return at::linear(input, w_bf16);
    }
    // Standard path: ensure weight is on the same device as input
    auto w = weight.to(device).to(dtype);
    return at::linear(input, w);
}

// ── Helper: chunked topk (for large seq) ──
static at::Tensor chunked_topk(const at::Tensor& idx_q, const at::Tensor& idx_k,
                                double idx_scale, int64_t actual_topk,
                                int64_t num_heads, int64_t idx_n_heads,
                                int64_t batch, int64_t seq, at::ScalarType compute_dtype,
                                int device_id) {
    // For seq <= 4096, compute full scores in one matmul (fast, no chunking overhead)
    // [B, idx_n_heads, S, S] * 2B (BF16) = 32MB at 4K, 128MB at 8K — fits in GPU memory
    if (seq <= 4096) {
        auto scores = at::matmul(idx_q, idx_k.transpose(-2, -1)) * idx_scale;
        if (idx_n_heads != num_heads) {
            scores = scores.mean(1, /*keepdim=*/true)
                         .expand({batch, num_heads, seq, seq}, /*implicit=*/false);
        }
        auto [_, indices] = scores.topk(actual_topk, -1, true, true);
        return indices;
    }

    // For seq > 4096, use chunked approach but with larger chunks (2048)
    // to reduce kernel launch overhead
    int64_t score_chunk = 2048;
    at::Tensor best_scores, best_indices;
    bool has_best = false;
    for (int64_t k_start = 0; k_start < seq; k_start += score_chunk) {
        int64_t k_end = std::min(k_start + score_chunk, seq);
        int64_t k_len = k_end - k_start;
        auto idx_k_chunk = idx_k.narrow(-2, k_start, k_len);
        auto scores_chunk = at::matmul(idx_q, idx_k_chunk.transpose(-2, -1)) * idx_scale;
        if (idx_n_heads != num_heads) {
            scores_chunk = scores_chunk.mean(1, /*keepdim=*/true)
                                 .expand({batch, num_heads, seq, k_len}, /*implicit=*/false);
        }
        int64_t local_topk = std::min(actual_topk, k_len);
        auto [ls, li] = scores_chunk.topk(local_topk, -1, true, true);
        auto offset = at::full(li.sizes(), (double)k_start,
                              at::TensorOptions().dtype(at::kFloat).device(at::Device(at::Device::Type::CUDA, device_id)));
        auto li_offset = li.to(at::kFloat) + offset;
        if (has_best) {
            auto merged = at::cat({best_scores, ls}, -1);
            auto merged_idx = at::cat({best_indices, li_offset.to(at::kLong)}, -1);
            auto [s, pos] = merged.topk(actual_topk, -1, true, true);
            best_scores = s;
            best_indices = merged_idx.gather(-1, pos, false);
        } else {
            best_scores = ls;
            best_indices = li_offset.to(at::kLong);
            has_best = true;
        }
        // Free chunk intermediates
        scores_chunk = at::Tensor();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    return best_indices;
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_dsa_attention — full DSA attention in one C++ call
//
// Input: hidden [B, S, H] BF16 on GPU
// Output: attention output [B, S, num_heads * v_head] BF16 on GPU
//
// IndexShare state is passed in/out via raw pointers:
//   topk_indices_ptr: void** — points to at::Tensor* (or nullptr to recompute)
//   idx_bias_keys_ptr: void** — points to at::Tensor*
// ══════════════════════════════════════════════════════════════════════

void* v4_glm5_dsa_attention(
    // Input
    void* input_ptr,            // at::Tensor* [B, S, H]
    // Attention weights (all at::Tensor*)
    void* q_a_proj, void* q_a_layernorm, void* q_b_proj,
    void* kv_a_proj, void* kv_a_layernorm, void* kv_b_proj,
    void* o_proj,
    // FP8 scales (nullable: pass nullptr if not FP8)
    void* q_a_scale, void* q_b_scale, void* kv_a_scale, void* kv_b_scale, void* o_scale,
    // Indexer weights (nullable for non-full layers)
    void* idx_wq_b, void* idx_wk, void* idx_k_norm_w, void* idx_k_norm_b,
    void* idx_weights_proj,
    void* idx_wq_b_scale, void* idx_wk_scale,
    // Config
    int batch_i, int seq_i, int num_heads_i, int qk_nope_i, int qk_rope_i,
    int v_head_i, int kv_lora_i, int idx_head_dim_i, int idx_n_heads_i,
    int idx_topk_i, int layer_i, bool is_full_layer,
    double rms_eps, double rope_theta, bool rope_interleave,
    int device_id,
    // IndexShare state (in/out)
    void** topk_indices_ptr,    // &at::Tensor* or &nullptr
    void** idx_bias_keys_ptr,   // &at::Tensor* or &nullptr
    int* source_layer           // &int
) {
    try {
        auto& input = *reinterpret_cast<at::Tensor*>(input_ptr);
        auto compute_dtype = input.scalar_type();
        int64_t batch = batch_i, seq = seq_i;
        int64_t nh = num_heads_i, qn = qk_nope_i, qr = qk_rope_i, vh = v_head_i;
        int64_t kvl = kv_lora_i, ihd = idx_head_dim_i, inh = idx_n_heads_i;
        int64_t itk = idx_topk_i;
        auto device = at::Device(at::Device::Type::CUDA, device_id);

        // ── Q/K/V projections ──
        auto& q_a_w = *reinterpret_cast<at::Tensor*>(q_a_proj);
        auto& q_a_ln = *reinterpret_cast<at::Tensor*>(q_a_layernorm);
        auto& q_b_w = *reinterpret_cast<at::Tensor*>(q_b_proj);
        auto& kv_a_w = *reinterpret_cast<at::Tensor*>(kv_a_proj);
        auto& kv_a_ln = *reinterpret_cast<at::Tensor*>(kv_a_layernorm);
        auto& kv_b_w = *reinterpret_cast<at::Tensor*>(kv_b_proj);
        auto& o_w = *reinterpret_cast<at::Tensor*>(o_proj);

        // FP8 scales
        auto qa_s = q_a_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(q_a_scale)) : std::nullopt;
        auto qb_s = q_b_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(q_b_scale)) : std::nullopt;
        auto kva_s = kv_a_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(kv_a_scale)) : std::nullopt;
        auto kvb_s = kv_b_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(kv_b_scale)) : std::nullopt;
        auto o_s = o_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(o_scale)) : std::nullopt;
        auto iwq_s = idx_wq_b_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(idx_wq_b_scale)) : std::nullopt;
        auto iwk_s = idx_wk_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(idx_wk_scale)) : std::nullopt;

        auto q_a = safe_linear(input, q_a_w, qa_s);
        auto q_a_normed = rms_norm(q_a, q_a_ln.to(compute_dtype), rms_eps);
        auto q_b = safe_linear(q_a_normed, q_b_w, qb_s);
        auto q = q_b.reshape({batch, seq, nh, qn + qr}).transpose(1, 2);
        auto q_nope = q.narrow(-1, 0, qn);
        auto q_rope = q.narrow(-1, qn, qr);

        auto kv_a = safe_linear(input, kv_a_w, kva_s);
        auto kv_lora_raw = kv_a.narrow(-1, 0, kvl);
        auto k_rope = kv_a.narrow(-1, kvl, qr);
        auto kv_lora_part = rms_norm(kv_lora_raw, kv_a_ln.to(compute_dtype), rms_eps);
        auto kv_b = safe_linear(kv_lora_part, kv_b_w, kvb_s);
        kv_b = kv_b.reshape({batch, seq, nh, qn + vh});
        auto k_nope = kv_b.narrow(-1, 0, qn).transpose(1, 2);
        auto v = kv_b.narrow(-1, qn, vh).transpose(1, 2);

        // ── RoPE ──
        auto k_rope_expanded = k_rope.unsqueeze(2).transpose(1, 2)
                                   .expand({batch, nh, seq, qr}, /*implicit=*/false);
        auto [cos, sin] = rope_cos_sin(seq, qr, rope_theta, device_id);
        cos = cos.to(compute_dtype);
        sin = sin.to(compute_dtype);
        at::Tensor q_rope_rot, k_rope_rot;
        if (rope_interleave) {
            q_rope_rot = apply_rotary_interleave(q_rope, cos, sin);
            k_rope_rot = apply_rotary_interleave(k_rope_expanded, cos, sin);
        } else {
            q_rope_rot = apply_rotary(q_rope, cos, sin);
            k_rope_rot = apply_rotary(k_rope_expanded, cos, sin);
        }

        auto q_full = at::cat({q_nope, q_rope_rot}, -1);
        auto k_full = at::cat({k_nope, k_rope_rot}, -1);
        double attn_scale = 1.0 / std::sqrt((double)(qn + qr));

        // ── DSA Indexer ──
        bool should_compute_topk = is_full_layer &&
            (*topk_indices_ptr == nullptr || layer_i % 4 == 0); // index_topk_freq=4

        if (idx_wq_b && idx_wk && idx_k_norm_w && idx_k_norm_b && idx_weights_proj) {
            if (should_compute_topk) {
                auto& wq_b = *reinterpret_cast<at::Tensor*>(idx_wq_b);
                auto& wk = *reinterpret_cast<at::Tensor*>(idx_wk);
                auto& kn_w = *reinterpret_cast<at::Tensor*>(idx_k_norm_w);
                auto& kn_b = *reinterpret_cast<at::Tensor*>(idx_k_norm_b);
                auto& wproj = *reinterpret_cast<at::Tensor*>(idx_weights_proj);

                // Indexer Q — with FP8 scale
                auto idx_q = safe_linear(q_a, wq_b, iwq_s);
                idx_q = idx_q.reshape({batch, seq, inh, ihd}).transpose(1, 2);

                // Indexer K — with FP8 scale
                auto idx_k_raw = safe_linear(input, wk, iwk_s);
                auto idx_k = rms_norm_with_bias(idx_k_raw, kn_w.to(compute_dtype),
                                                kn_b.to(compute_dtype), rms_eps);
                auto idx_k_expanded = idx_k.unsqueeze(1).expand({batch, inh, seq, ihd}, /*implicit=*/false);

                // Indexer RoPE
                at::Tensor idx_q_rot, idx_k_rot;
                if (rope_interleave) {
                    auto [ci, si] = rope_cos_sin(seq, ihd, rope_theta, device_id);
                    ci = ci.to(compute_dtype);
                    si = si.to(compute_dtype);
                    idx_q_rot = apply_rotary_interleave(idx_q, ci, si);
                    idx_k_rot = apply_rotary_interleave(idx_k_expanded, ci, si);
                } else {
                    idx_q_rot = idx_q;
                    idx_k_rot = idx_k_expanded;
                }

                double idx_scale = 1.0 / std::sqrt((double)ihd);
                int64_t actual_topk = std::min(itk, seq);

                auto topk_indices = chunked_topk(idx_q_rot, idx_k_rot, idx_scale,
                                                  actual_topk, nh, inh, batch, seq,
                                                  compute_dtype, device_id);

                // Indexer bias keys
                auto idx_bias_keys = safe_linear(input, wproj, std::nullopt);
                idx_bias_keys = idx_bias_keys.reshape({batch, seq, inh}).transpose(1, 2);

                // Save state
                *topk_indices_ptr = new at::Tensor(topk_indices);
                *idx_bias_keys_ptr = new at::Tensor(idx_bias_keys);
                *source_layer = layer_i;
            }
        } else {
            *topk_indices_ptr = nullptr;
            *idx_bias_keys_ptr = nullptr;
        }

        // ── Attention via SDPA ──
        at::Tensor context;
        if (*topk_indices_ptr) {
            auto& topk_indices = *reinterpret_cast<at::Tensor*>(*topk_indices_ptr);
            auto& idx_bias_keys = *reinterpret_cast<at::Tensor*>(*idx_bias_keys_ptr);
            int64_t actual_topk = topk_indices.size(-1);

            // Per-key bias
            at::Tensor bias_per_key;
            if (inh != nh) {
                bias_per_key = idx_bias_keys.mean(1, /*keepdim=*/true)
                                .expand({batch, nh, seq}, /*implicit=*/false);
            } else {
                bias_per_key = idx_bias_keys;
            }

            int64_t attn_chunk = (seq > 2048) ? 512 : seq;
            if (attn_chunk >= seq) {
                // Small seq: single pass — optimized bias construction
                // Start with bias_per_key broadcast to [B, nh, S, S]
                auto bias = bias_per_key.unsqueeze(2)
                            .expand({batch, nh, seq, seq}, /*implicit=*/false)
                            .to(compute_dtype).clone();

                // Mask out positions not in topk OR future positions (causal)
                // Build mask: 1 where attended, 0 where masked
                auto sparse_mask = at::zeros({batch, nh, seq, seq},
                    at::TensorOptions().dtype(compute_dtype).device(device));
                auto ones = at::ones({batch, nh, seq, actual_topk},
                    at::TensorOptions().dtype(compute_dtype).device(device));
                sparse_mask.scatter_(-1, topk_indices, ones);

                // Causal: position j > i is masked (future)
                // Combined: attend only if (in topk) AND (not future)
                auto causal_mask = at::ones({seq, seq},
                    at::TensorOptions().dtype(at::kBool).device(device)).triu(1);
                auto causal_f = causal_mask.unsqueeze(0).unsqueeze(0)
                                .expand({batch, nh, seq, seq}, /*implicit=*/false)
                                .to(compute_dtype);
                auto combined = sparse_mask * (1.0 - causal_f);
                // Set -inf where combined == 0 (not attended)
                bias.masked_fill_(combined.eq(0.0), -std::numeric_limits<double>::infinity());

                // Free intermediates
                sparse_mask = at::Tensor(); causal_f = at::Tensor(); combined = at::Tensor();
                c10::cuda::CUDACachingAllocator::emptyCache();

                context = at::scaled_dot_product_attention(
                    q_full, k_full, v, bias, 0.0, false, attn_scale);
            } else {
                // Chunked attention
                std::vector<at::Tensor> outputs;
                for (int64_t q_start = 0; q_start < seq; q_start += attn_chunk) {
                    int64_t q_end = std::min(q_start + attn_chunk, seq);
                    int64_t q_len = q_end - q_start;
                    auto q_chunk = q_full.narrow(2, q_start, q_len);

                    auto chunk_topk = topk_indices.narrow(2, q_start, q_len);
                    auto sparse_mask = at::zeros({batch, nh, q_len, seq},
                        at::TensorOptions().dtype(compute_dtype).device(device));
                    auto ones = at::ones({batch, nh, q_len, actual_topk},
                        at::TensorOptions().dtype(compute_dtype).device(device));
                    sparse_mask.scatter_(-1, chunk_topk, ones);

                    auto q_pos = (at::arange(q_len, at::TensorOptions().dtype(at::kLong).device(device)) + q_start).to(compute_dtype);
                    auto k_pos = at::arange(seq, at::TensorOptions().dtype(at::kLong).device(device)).to(compute_dtype);
                    auto diff = k_pos.unsqueeze(0) - q_pos.unsqueeze(1);
                    auto cm = diff.gt(0.0);
                    auto causal_f = cm.unsqueeze(0).unsqueeze(0)
                                   .expand({batch, nh, q_len, seq}, /*implicit=*/false)
                                   .to(compute_dtype);

                    auto combined = sparse_mask * (1.0 - causal_f);
                    auto mask_bool = combined.eq(0.0);
                    auto bias = bias_per_key.unsqueeze(2)
                                .expand({batch, nh, q_len, seq}, /*implicit=*/false)
                                .to(compute_dtype);
                    bias = bias.masked_fill(mask_bool, -std::numeric_limits<double>::infinity());

                    auto chunk_out = at::scaled_dot_product_attention(
                        q_chunk, k_full, v, bias, 0.0, false, attn_scale);
                    outputs.push_back(chunk_out);
                }
                context = at::cat(outputs, 2);
            }
        } else {
            // Full causal SDPA
            context = at::scaled_dot_product_attention(
                q_full, k_full, v, std::nullopt, 0.0, true, attn_scale);
        }

        // ── Output projection ──
        auto out = context.transpose(1, 2).reshape({batch, seq, nh * vh});
        auto result = safe_linear(out, o_w, o_s);
        return new at::Tensor(std::move(result));
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_glm5_dsa_attention] FAILED: %s\n", e.what());
        return nullptr;
    }
}

// ── Helper: free at::Tensor* created by v4_glm5_dsa_attention ──
void v4_glm5_free_at_tensor(void* tensor_ptr) {
    if (tensor_ptr) {
        delete reinterpret_cast<at::Tensor*>(tensor_ptr);
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_mlp_fp8 — SwiGLU MLP with optional FP8 weights
// ══════════════════════════════════════════════════════════════════════
void* v4_glm5_mlp_fp8(
    void* input_ptr,
    void* gate_ptr, void* up_ptr, void* down_ptr,
    void* gate_scale_ptr, void* up_scale_ptr, void* down_scale_ptr
) {
    try {
        auto& input = *reinterpret_cast<at::Tensor*>(input_ptr);
        auto& gate = *reinterpret_cast<at::Tensor*>(gate_ptr);
        auto& up = *reinterpret_cast<at::Tensor*>(up_ptr);
        auto& down = *reinterpret_cast<at::Tensor*>(down_ptr);
        auto dtype = input.scalar_type();

        auto gate_scale = gate_scale_ptr ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(gate_scale_ptr)) : std::nullopt;
        auto up_scale = up_scale_ptr ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(up_scale_ptr)) : std::nullopt;
        auto down_scale = down_scale_ptr ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(down_scale_ptr)) : std::nullopt;

        auto gate_out = safe_linear(input, gate, gate_scale);
        auto up_out = safe_linear(input, up, up_scale);
        auto activated = at::silu(gate_out) * up_out;
        auto result = safe_linear(activated, down, down_scale);
        return new at::Tensor(std::move(result));
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_glm5_mlp_fp8] FAILED: %s\n", e.what());
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_rms_norm — RMSNorm
// ══════════════════════════════════════════════════════════════════════
void* v4_glm5_rms_norm(void* input_ptr, void* weight_ptr, double eps) {
    try {
        auto& input = *reinterpret_cast<at::Tensor*>(input_ptr);
        auto& weight = *reinterpret_cast<at::Tensor*>(weight_ptr);
        return new at::Tensor(rms_norm(input, weight, eps));
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_glm5_rms_norm] FAILED: %s\n", e.what());
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_cross_entropy_loss — chunked cross-entropy loss
// Returns scalar loss tensor (F32)
// ══════════════════════════════════════════════════════════════════════
void* v4_glm5_cross_entropy_loss(
    void* hidden_ptr,       // [B, S, H] BF16
    void* lm_head_ptr,      // [vocab, H] BF16
    void* targets_ptr,      // [B, S] int64
    void* mask_ptr,          // [B, S] float
    int seq_len_i, int vocab_i, int chunk_size_i, int device_id
) {
    try {
        auto& hidden = *reinterpret_cast<at::Tensor*>(hidden_ptr);
        auto& lm_head = *reinterpret_cast<at::Tensor*>(lm_head_ptr);
        auto& targets = *reinterpret_cast<at::Tensor*>(targets_ptr);
        auto& mask = *reinterpret_cast<at::Tensor*>(mask_ptr);
        auto device = at::Device(at::Device::Type::CUDA, device_id);
        int64_t seq_len = seq_len_i, vocab = vocab_i, chunk = chunk_size_i;

        // Shifted: targets[1:], mask[1:]
        auto shifted_targets = targets.narrow(1, 1, seq_len - 1);
        auto shifted_mask = mask.narrow(1, 1, seq_len - 1).to(at::kFloat);
        auto total_mask = shifted_mask.sum(at::kFloat);

        auto loss_acc = at::zeros({}, at::TensorOptions().dtype(at::kFloat).device(device));
        for (int64_t start = 0; start < seq_len - 1; start += chunk) {
            int64_t end = std::min(start + chunk, seq_len - 1);
            int64_t len = end - start;
            auto normed_chunk = hidden.narrow(1, start, len);
            auto logits = at::linear(normed_chunk, lm_head);
            auto log_probs = logits.reshape({-1, vocab}).log_softmax(-1, at::kFloat);
            auto t = shifted_targets.narrow(1, start, len).reshape({-1});
            auto m = shifted_mask.narrow(1, start, len);
            auto per_token = at::nll_loss(log_probs, t, std::nullopt, at::Reduction::None, -100)
                                 .reshape({1, len});
            loss_acc = loss_acc + (per_token * m).sum(at::kFloat);
        }
        return new at::Tensor(loss_acc / total_mask.clamp_min(1.0));
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_glm5_cross_entropy_loss] FAILED: %s\n", e.what());
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_adam_step — Adam optimizer in C++
// Updates params, m, v in-place
// ══════════════════════════════════════════════════════════════════════
void v4_adam_step(
    void** params,     // array of at::Tensor* (trainable params)
    void** grads,       // array of at::Tensor* (gradients)
    void** m_state,     // array of at::Tensor* (Adam m)
    void** v_state,     // array of at::Tensor* (Adam v)
    int n_params,
    double lr, double beta1, double beta2, double eps, int step_i
) {
    try {
        double sn = (double)(step_i + 1);
        double bias1 = 1.0 - std::pow(beta1, sn);
        double bias2 = 1.0 - std::pow(beta2, sn);

        for (int i = 0; i < n_params; i++) {
            auto& param = *reinterpret_cast<at::Tensor*>(params[i]);
            auto& grad = *reinterpret_cast<at::Tensor*>(grads[i]);
            auto& m = *reinterpret_cast<at::Tensor*>(m_state[i]);
            auto& v = *reinterpret_cast<at::Tensor*>(v_state[i]);

            if (!grad.defined() || grad.numel() == 0) continue;

            auto g = grad.to(at::kFloat);
            m = m * beta1 + g * (1.0 - beta1);
            v = v * beta2 + (g * g) * (1.0 - beta2);
            auto mh = m / bias1;
            auto vh = v / bias2;
            auto update = mh / (vh.sqrt() + eps);
            param.add_(update * (-lr));
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_adam_step] FAILED: %s\n", e.what());
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_moe_layer — MoE routing + expert dispatch + shared expert + combine
//
// Replaces ~20 tch-rs calls per layer with 1 FFI call.
// Expert weights are passed as CPU at::Tensor* arrays — C++ does to_device internally.
// ══════════════════════════════════════════════════════════════════════
void* v4_glm5_moe_layer(
    void* mlp_input_ptr,          // [B, S, H] BF16 on GPU
    // Shared expert weights (GPU, replicated)
    void* shared_gate, void* shared_up, void* shared_down,
    void* shared_gate_scale, void* shared_up_scale, void* shared_down_scale,
    // Router gate weight (GPU)
    void* gate_weight,
    // Expert weights (CPU, prefetched to GPU on demand)
    // Arrays of at::Tensor* — length n_local_experts × 3 (gate, up, down)
    void** expert_gate_weights,   // [n_local_experts] CPU tensors
    void** expert_up_weights,     // [n_local_experts] CPU tensors
    void** expert_down_weights,   // [n_local_experts] CPU tensors
    void** expert_gate_scales,    // [n_local_experts] or nullptr
    void** expert_up_scales,
    void** expert_down_scales,
    int n_local_experts,
    int* local_expert_indices,    // [n_local_experts] global expert IDs
    int n_routed_experts, int topk,
    double routed_scaling_factor,
    int device_id
) {
    try {
        auto& mlp_input = *reinterpret_cast<at::Tensor*>(mlp_input_ptr);
        auto dtype = mlp_input.scalar_type();
        auto device = at::Device(at::Device::Type::CUDA, device_id);
        int64_t batch = mlp_input.size(0);
        int64_t seq = mlp_input.size(1);
        int64_t hidden = mlp_input.size(2);
        int64_t k = topk;
        int64_t n_experts = n_routed_experts;

        // ── Shared expert: merge gate+up into single matmul ──
        auto& sg = *reinterpret_cast<at::Tensor*>(shared_gate);
        auto& su = *reinterpret_cast<at::Tensor*>(shared_up);
        auto& sd = *reinterpret_cast<at::Tensor*>(shared_down);
        auto sg_scale = shared_gate_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(shared_gate_scale)) : std::nullopt;
        auto su_scale = shared_up_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(shared_up_scale)) : std::nullopt;
        auto sd_scale = shared_down_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(shared_down_scale)) : std::nullopt;
        // If both gate and up have same shape and no FP8, concatenate weights for single matmul
        int64_t inter_dim = sg.size(0);
        auto shared_output = [&]() -> at::Tensor {
            if (sg_scale.has_value() || su_scale.has_value()) {
                // FP8 path: can't concatenate FP8 weights, use separate calls
                auto gate_out = safe_linear(mlp_input, sg, sg_scale);
                auto up_out = safe_linear(mlp_input, su, su_scale);
                auto activated = at::silu(gate_out) * up_out;
                return safe_linear(activated, sd, sd_scale);
            } else {
                // BF16 path: merge gate+up weights → single matmul
                auto gate_up_w = at::cat({sg, su}, 0);  // [2*inter, hidden]
                auto gu = at::linear(mlp_input, gate_up_w);  // [B, S, 2*inter]
                auto gate_part = gu.narrow(-1, 0, inter_dim);
                auto up_part = gu.narrow(-1, inter_dim, inter_dim);
                auto activated = at::silu(gate_part) * up_part;
                return safe_linear(activated, sd, sd_scale);
            }
        }();

        // ── Router: sigmoid + topk ──
        auto& gate_w = *reinterpret_cast<at::Tensor*>(gate_weight);
        auto router_logits = at::linear(mlp_input, gate_w.to(dtype));
        auto scores = at::sigmoid(router_logits);
        auto [topk_weights, topk_indices] = scores.topk(k, -1, true, true);
        auto denom = topk_weights.sum(-1, /*keepdim=*/true);
        topk_weights = (topk_weights / denom) * routed_scaling_factor;

        // Flatten for per-token dispatch
        auto flat_input = mlp_input.reshape({-1, hidden});
        auto tk_indices = topk_indices.reshape({-1, k});
        auto tk_weights = topk_weights.reshape({-1, k});

        auto partial_output = at::zeros(flat_input.sizes(),
            at::TensorOptions().dtype(dtype).device(device));

        // ── Expert dispatch — batched matmul (Megatron-style GroupedLinear) ──
        // Pre-stacked expert weights: cached per (layer, n_local_experts) to avoid
        // redundant at::stack calls (64 tensors × 3 = 192 copies per layer per step)
        static std::map<std::pair<int, int>, std::tuple<at::Tensor, at::Tensor, at::Tensor>> s_expert_weight_cache;
        auto cache_key = std::make_pair((int)mlp_input.size(1), n_local_experts);  // seq + n_experts as key
        at::Tensor gate_weights_stack, up_weights_stack, down_weights_stack;
        auto cache_it = s_expert_weight_cache.find(cache_key);
        if (cache_it != s_expert_weight_cache.end()) {
            auto& [gw, uw, dw] = cache_it->second;
            gate_weights_stack = gw;
            up_weights_stack = uw;
            down_weights_stack = dw;
        } else {
            // Stack expert weights: [n_experts, out, in]
            gate_weights_stack = at::stack(
                std::vector<at::Tensor>([&]{
                    std::vector<at::Tensor> v;
                    for (int e = 0; e < n_local_experts; e++)
                        v.push_back(*reinterpret_cast<at::Tensor*>(expert_gate_weights[e]));
                    return v;
                }()), 0);
            up_weights_stack = at::stack(
                std::vector<at::Tensor>([&]{
                    std::vector<at::Tensor> v;
                    for (int e = 0; e < n_local_experts; e++)
                        v.push_back(*reinterpret_cast<at::Tensor*>(expert_up_weights[e]));
                    return v;
                }()), 0);
            down_weights_stack = at::stack(
                std::vector<at::Tensor>([&]{
                    std::vector<at::Tensor> v;
                    for (int e = 0; e < n_local_experts; e++)
                        v.push_back(*reinterpret_cast<at::Tensor*>(expert_down_weights[e]));
                    return v;
                }()), 0);
            s_expert_weight_cache[cache_key] = std::make_tuple(gate_weights_stack, up_weights_stack, down_weights_stack);
        }

        // Determine token assignment per expert
        int64_t N = flat_input.size(0);
        // For each token, find which local expert slot it goes to
        // tk_indices: [N, k] with global expert IDs
        // Build local expert index: [N, k] → local index or -1
        auto local_idx = at::full({N, k}, -1.0, at::TensorOptions().dtype(at::kFloat).device(device));
        for (int e = 0; e < n_local_experts; e++) {
            int global_e = local_expert_indices[e];
            auto mask = tk_indices.eq(global_e);
            local_idx = local_idx.masked_fill(mask, (float)e);
        }
        // For each token, pick the first matching local expert (if any)
        auto has_expert = local_idx.ge(0).any(-1);  // [N] bool
        auto assigned_expert = local_idx.clamp_min(0).to(at::kLong).narrow(-1, 0, 1).squeeze(-1);  // [N]

        // Sort by expert to group tokens
        auto [sorted_experts, sort_order] = assigned_expert.sort(0);
        auto sorted_tokens = flat_input.index_select(0, sort_order);
        auto sorted_weights = tk_weights.gather(0, sort_order.narrow(1, 0, 1).expand({-1, k}));

        // Count per-expert (avoid .item() sync — use a fixed max)
        int64_t max_per_expert = (N + n_local_experts - 1) / n_local_experts + 16;

        // Pad to [n_local_experts, max_per_expert, hidden]
        auto gathered = at::zeros({n_local_experts, max_per_expert, hidden},
            at::TensorOptions().dtype(dtype).device(device));
        auto gather_valid = at::zeros({n_local_experts, max_per_expert},
            at::TensorOptions().dtype(at::kBool).device(device));
        auto weight_gathered = at::zeros({n_local_experts, max_per_expert, 1},
            at::TensorOptions().dtype(dtype).device(device));

        // Fill using per-expert offsets
        {
            auto expert_offsets = at::cumsum(
                at::cat(std::vector<at::Tensor>({
                    at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(device)),
                    at::histc(sorted_experts.to(at::kFloat), n_local_experts, 0, n_local_experts - 1).to(at::kLong)
                        .narrow(0, 0, n_local_experts - 1)
                })), 0);
            auto arange_idx = at::arange(N, at::TensorOptions().dtype(at::kLong).device(device));
            auto group_starts = expert_offsets.index_select(0, sorted_experts.clamp_max(n_local_experts - 1));
            auto local_pos = arange_idx - group_starts;
            auto valid = local_pos.lt(max_per_expert);
            auto valid_idx = valid.nonzero().squeeze(-1);

            if (valid_idx.size(0) > 0) {
                auto v_expert = sorted_experts.index_select(0, valid_idx);
                auto v_pos = local_pos.index_select(0, valid_idx);
                auto v_tokens = sorted_tokens.index_select(0, valid_idx);
                auto v_weights = sorted_weights.index_select(0, valid_idx)
                    .sum(-1, true).to(dtype);  // [valid, 1]

                // Scatter into batched tensor
                gathered.index_put_({v_expert, v_pos}, v_tokens);
                gather_valid.index_put_({v_expert, v_pos}, true);
                weight_gathered.index_put_({v_expert, v_pos}, v_weights);
            }
        }

        // Batched gate+up matmul: [E, max_tok, hidden] @ [E, hidden, inter]^T → [E, max_tok, inter]
        auto gate_out = at::bmm(gathered, gate_weights_stack.transpose(-1, -2));
        auto up_out = at::bmm(gathered, up_weights_stack.transpose(-1, -2));
        auto activated = at::silu(gate_out) * up_out;

        // Batched down matmul: [E, max_tok, inter] @ [E, inter, hidden]^T → [E, max_tok, hidden]
        auto expert_outputs = at::bmm(activated, down_weights_stack.transpose(-1, -2));

        // Zero invalid positions and apply weights
        expert_outputs = expert_outputs * gather_valid.unsqueeze(-1).to(dtype);
        expert_outputs = expert_outputs * weight_gathered;

        // Scatter-add back to original token positions
        {
            auto expert_offsets = at::cumsum(
                at::cat(std::vector<at::Tensor>({
                    at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(device)),
                    at::histc(sorted_experts.to(at::kFloat), n_local_experts, 0, n_local_experts - 1).to(at::kLong)
                        .narrow(0, 0, n_local_experts - 1)
                })), 0);
            auto arange_idx = at::arange(N, at::TensorOptions().dtype(at::kLong).device(device));
            auto group_starts = expert_offsets.index_select(0, sorted_experts.clamp_max(n_local_experts - 1));
            auto local_pos = arange_idx - group_starts;
            auto valid = local_pos.lt(max_per_expert);
            auto valid_idx = valid.nonzero().squeeze(-1);

            if (valid_idx.size(0) > 0) {
                auto v_expert = sorted_experts.index_select(0, valid_idx);
                auto v_pos = local_pos.index_select(0, valid_idx);
                auto v_token_ids = sort_order.index_select(0, valid_idx).squeeze(-1);

                // Gather expert outputs for valid positions
                auto out_flat = expert_outputs.index_select(0, v_expert)
                    .index_select(1, v_pos);  // [valid, hidden]
                partial_output.index_add_(0, v_token_ids, out_flat);
            }
        }

        // Free batched intermediates (but not stacked weights — they're cached)
        gathered = at::Tensor(); gate_out = at::Tensor(); up_out = at::Tensor();
        activated = at::Tensor(); expert_outputs = at::Tensor();
        c10::cuda::CUDACachingAllocator::emptyCache();

        // Combine: partial + shared
        auto partial_mlp = partial_output.reshape({1, -1, hidden}) + shared_output;
        c10::cuda::CUDACachingAllocator::emptyCache();
        return new at::Tensor(std::move(partial_mlp));
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_glm5_moe_layer] FAILED: %s\n", e.what());
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_embedding — embedding lookup
// ══════════════════════════════════════════════════════════════════════
void* v4_glm5_embedding(void* embed_weight_ptr, void* input_ids_ptr, int device_id) {
    try {
        auto& embed_weight = *reinterpret_cast<at::Tensor*>(embed_weight_ptr);
        auto& input_ids = *reinterpret_cast<at::Tensor*>(input_ids_ptr);
        auto result = at::embedding(embed_weight, input_ids, -1, false, false);
        return new at::Tensor(std::move(result));
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_glm5_embedding] FAILED: %s\n", e.what());
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_glm5_layer_forward — full transformer layer in one C++ call
//
// Combines: RMSNorm → attention → residual → RMSNorm → MoE/dense → residual
// All intermediate tensors stay on C++ stack — zero FFI crossings.
// Returns new hidden state [B, S, H].
// ══════════════════════════════════════════════════════════════════════
void* v4_glm5_layer_forward(
    void* hidden_ptr,              // [B, S, H] BF16 on GPU
    // Norm weights
    void* input_norm_weight,       // RMSNorm weight for attention input
    void* post_norm_weight,        // RMSNorm weight for MLP input
    // Attention weights (all at::Tensor* — see v4_glm5_dsa_attention)
    void* q_a_proj, void* q_a_layernorm, void* q_b_proj,
    void* kv_a_proj, void* kv_a_layernorm, void* kv_b_proj, void* o_proj,
    void* q_a_scale, void* q_b_scale, void* kv_a_scale, void* kv_b_scale, void* o_scale,
    void* idx_wq_b, void* idx_wk, void* idx_k_norm_w, void* idx_k_norm_b,
    void* idx_weights_proj, void* idx_wq_b_scale, void* idx_wk_scale,
    // MLP/MoE weights
    void* gate_weight,             // MoE router (or nullptr for dense)
    void* shared_gate, void* shared_up, void* shared_down,  // shared expert (or nullptr for dense)
    void* shared_gate_scale, void* shared_up_scale, void* shared_down_scale,
    // Dense MLP weights (if not MoE)
    void* dense_gate, void* dense_up, void* dense_down,
    void* dense_gate_scale, void* dense_up_scale, void* dense_down_scale,
    // Expert weights (CPU, for MoE only)
    void** expert_gate_weights, void** expert_up_weights, void** expert_down_weights,
    void** expert_gate_scales, void** expert_up_scales, void** expert_down_scales,
    int n_local_experts,
    const int* local_expert_indices,
    // Config
    int batch_i, int seq_i, int num_heads_i, int qk_nope_i, int qk_rope_i,
    int v_head_i, int kv_lora_i, int idx_head_dim_i, int idx_n_heads_i,
    int idx_topk_i, int layer_i, bool is_full_layer,
    bool is_moe_layer, int n_routed_experts, int topk,
    double rms_eps, double rope_theta, bool rope_interleave,
    double routed_scaling_factor,
    int device_id,
    // IndexShare state (in/out)
    void** topk_indices_ptr, void** idx_bias_keys_ptr, int* source_layer
) {
    try {
        auto& hidden = *reinterpret_cast<at::Tensor*>(hidden_ptr);
        auto dtype = hidden.scalar_type();
        int64_t batch = batch_i, seq = seq_i;
        int64_t nh = num_heads_i;
        auto device = at::Device(at::Device::Type::CUDA, device_id);



        // ── 1. Attention RMSNorm ──
        auto& attn_norm_w = *reinterpret_cast<at::Tensor*>(input_norm_weight);
        auto hidden_norm = rms_norm(hidden, attn_norm_w.to(dtype), rms_eps);

        // ── 2. Attention ──
        // Build indexer weights check
        bool has_indexer = (idx_wq_b != nullptr && idx_wk != nullptr &&
                           idx_k_norm_w != nullptr && idx_k_norm_b != nullptr &&
                           idx_weights_proj != nullptr);

        // Delegate to v4_glm5_dsa_attention logic (inline for zero FFI overhead)
        // Q/K/V projections
        auto& q_a_w = *reinterpret_cast<at::Tensor*>(q_a_proj);
        auto& q_a_ln = *reinterpret_cast<at::Tensor*>(q_a_layernorm);
        auto& q_b_w = *reinterpret_cast<at::Tensor*>(q_b_proj);
        auto& kv_a_w = *reinterpret_cast<at::Tensor*>(kv_a_proj);
        auto& kv_a_ln = *reinterpret_cast<at::Tensor*>(kv_a_layernorm);
        auto& kv_b_w = *reinterpret_cast<at::Tensor*>(kv_b_proj);
        auto& o_w = *reinterpret_cast<at::Tensor*>(o_proj);
        int64_t qn = qk_nope_i, qr = qk_rope_i, vh = v_head_i, kvl = kv_lora_i;
        int64_t ihd = idx_head_dim_i, inh = idx_n_heads_i, itk = idx_topk_i;

        // Q/K/V projections — use safe_linear for FP8 scale support
        auto q_a_scale_t = q_a_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(q_a_scale)) : std::nullopt;
        auto q_b_scale_t = q_b_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(q_b_scale)) : std::nullopt;
        auto kv_a_scale_t = kv_a_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(kv_a_scale)) : std::nullopt;
        auto kv_b_scale_t = kv_b_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(kv_b_scale)) : std::nullopt;
        auto o_scale_t = o_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(o_scale)) : std::nullopt;

        auto q_a = safe_linear(hidden_norm, q_a_w, q_a_scale_t);
        auto q_a_normed = rms_norm(q_a, q_a_ln.to(dtype), rms_eps);
        auto q_b = safe_linear(q_a_normed, q_b_w, q_b_scale_t);
        auto q = q_b.reshape({batch, seq, nh, qn + qr}).transpose(1, 2);
        auto q_nope = q.narrow(-1, 0, qn);
        auto q_rope = q.narrow(-1, qn, qr);

        auto kv_a = safe_linear(hidden_norm, kv_a_w, kv_a_scale_t);
        auto kv_lora_raw = kv_a.narrow(-1, 0, kvl);
        auto k_rope = kv_a.narrow(-1, kvl, qr);
        auto kv_lora_part = rms_norm(kv_lora_raw, kv_a_ln.to(dtype), rms_eps);
        auto kv_b = safe_linear(kv_lora_part, kv_b_w, kv_b_scale_t);
        kv_b = kv_b.reshape({batch, seq, nh, qn + vh});
        auto k_nope = kv_b.narrow(-1, 0, qn).transpose(1, 2);
        auto v = kv_b.narrow(-1, qn, vh).transpose(1, 2);

        auto k_rope_expanded = k_rope.unsqueeze(2).transpose(1, 2)
                                   .expand({batch, nh, seq, qr}, /*implicit=*/false);
        auto [cos, sin] = rope_cos_sin(seq, qr, rope_theta, device_id);
        cos = cos.to(dtype); sin = sin.to(dtype);
        at::Tensor q_rope_rot, k_rope_rot;
        if (rope_interleave) {
            q_rope_rot = apply_rotary_interleave(q_rope, cos, sin);
            k_rope_rot = apply_rotary_interleave(k_rope_expanded, cos, sin);
        } else {
            q_rope_rot = apply_rotary(q_rope, cos, sin);
            k_rope_rot = apply_rotary(k_rope_expanded, cos, sin);
        }

        auto q_full = at::cat({q_nope, q_rope_rot}, -1);
        auto k_full = at::cat({k_nope, k_rope_rot}, -1);
        double attn_scale = 1.0 / std::sqrt((double)(qn + qr));

        // DSA indexer
        bool should_compute = is_full_layer &&
            (*topk_indices_ptr == nullptr || layer_i % 4 == 0);

        if (has_indexer && should_compute) {
            auto& wq_b = *reinterpret_cast<at::Tensor*>(idx_wq_b);
            auto& wk = *reinterpret_cast<at::Tensor*>(idx_wk);
            auto& kn_w = *reinterpret_cast<at::Tensor*>(idx_k_norm_w);
            auto& kn_b = *reinterpret_cast<at::Tensor*>(idx_k_norm_b);
            auto& wproj = *reinterpret_cast<at::Tensor*>(idx_weights_proj);
            auto idx_wq_b_scale_t = idx_wq_b_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(idx_wq_b_scale)) : std::nullopt;
            auto idx_wk_scale_t = idx_wk_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(idx_wk_scale)) : std::nullopt;
            auto idx_q = safe_linear(q_a, wq_b, idx_wq_b_scale_t);
            idx_q = idx_q.reshape({batch, seq, inh, ihd}).transpose(1, 2);
            auto idx_k_raw = safe_linear(hidden_norm, wk, idx_wk_scale_t);
            auto idx_k = rms_norm_with_bias(idx_k_raw, kn_w.to(dtype), kn_b.to(dtype), rms_eps);
            auto idx_k_exp = idx_k.unsqueeze(1).expand({batch, inh, seq, ihd}, /*implicit=*/false);
            at::Tensor idx_q_rot, idx_k_rot;
            if (rope_interleave) {
                auto [ci, si] = rope_cos_sin(seq, ihd, rope_theta, device_id);
                ci = ci.to(dtype); si = si.to(dtype);
                idx_q_rot = apply_rotary_interleave(idx_q, ci, si);
                idx_k_rot = apply_rotary_interleave(idx_k_exp, ci, si);
            } else { idx_q_rot = idx_q; idx_k_rot = idx_k_exp; }
            double idx_scale = 1.0 / std::sqrt((double)ihd);
            int64_t actual_topk = std::min(itk, seq);
            auto topk_indices = chunked_topk(idx_q_rot, idx_k_rot, idx_scale, actual_topk, nh, inh, batch, seq, dtype, device_id);
            auto idx_bias_keys = safe_linear(hidden_norm, wproj, std::nullopt);
            idx_bias_keys = idx_bias_keys.reshape({batch, seq, inh}).transpose(1, 2);
            *topk_indices_ptr = new at::Tensor(topk_indices);
            *idx_bias_keys_ptr = new at::Tensor(idx_bias_keys);
            *source_layer = layer_i;
        }

        // Attention computation (same as v4_glm5_dsa_attention)
        at::Tensor context;
        if (*topk_indices_ptr) {
            auto& topk_indices = *reinterpret_cast<at::Tensor*>(*topk_indices_ptr);
            auto& idx_bias_keys = *reinterpret_cast<at::Tensor*>(*idx_bias_keys_ptr);
            int64_t actual_topk = topk_indices.size(-1);
            at::Tensor bias_per_key;
            if (inh != nh) {
                bias_per_key = idx_bias_keys.mean(1, /*keepdim=*/true).expand({batch, nh, seq}, /*implicit=*/false);
            } else { bias_per_key = idx_bias_keys; }
            int64_t attn_chunk = (seq > 2048) ? 512 : seq;
            if (attn_chunk >= seq) {
                auto sm = at::zeros({batch, nh, seq, seq}, at::TensorOptions().dtype(dtype).device(device));
                auto ones = at::ones({batch, nh, seq, actual_topk}, at::TensorOptions().dtype(dtype).device(device));
                sm.scatter_(-1, topk_indices, ones);
                auto cm = at::ones({seq, seq}, at::TensorOptions().dtype(at::kBool).device(device)).triu(1);
                auto cf = cm.unsqueeze(0).unsqueeze(0).expand({batch, nh, seq, seq}, /*implicit=*/false).to(dtype);
                auto combined = sm * (1.0 - cf);
                auto mb = combined.eq(0.0);
                auto bias = bias_per_key.unsqueeze(2).expand({batch, nh, seq, seq}, /*implicit=*/false).to(dtype);
                bias = bias.masked_fill(mb, -std::numeric_limits<double>::infinity());
                context = at::scaled_dot_product_attention(q_full, k_full, v, bias, 0.0, false, attn_scale);
            } else {
                std::vector<at::Tensor> outputs;
                for (int64_t qs = 0; qs < seq; qs += attn_chunk) {
                    int64_t qe = std::min(qs + attn_chunk, seq), ql = qe - qs;
                    auto qc = q_full.narrow(2, qs, ql);
                    auto ct = topk_indices.narrow(2, qs, ql);
                    auto sm = at::zeros({batch, nh, ql, seq}, at::TensorOptions().dtype(dtype).device(device));
                    auto ones = at::ones({batch, nh, ql, actual_topk}, at::TensorOptions().dtype(dtype).device(device));
                    sm.scatter_(-1, ct, ones);
                    auto qp = (at::arange(ql, at::TensorOptions().dtype(at::kLong).device(device)) + qs).to(dtype);
                    auto kp = at::arange(seq, at::TensorOptions().dtype(at::kLong).device(device)).to(dtype);
                    auto diff = kp.unsqueeze(0) - qp.unsqueeze(1);
                    auto cf = diff.gt(0.0).unsqueeze(0).unsqueeze(0).expand({batch, nh, ql, seq}, /*implicit=*/false).to(dtype);
                    auto combined = sm * (1.0 - cf);
                    auto mb = combined.eq(0.0);
                    auto bias = bias_per_key.unsqueeze(2).expand({batch, nh, ql, seq}, /*implicit=*/false).to(dtype);
                    bias = bias.masked_fill(mb, -std::numeric_limits<double>::infinity());
                    outputs.push_back(at::scaled_dot_product_attention(qc, k_full, v, bias, 0.0, false, attn_scale));
                }
                context = at::cat(outputs, 2);
            }
        } else {
            context = at::scaled_dot_product_attention(q_full, k_full, v, std::nullopt, 0.0, true, attn_scale);
        }

        auto attn_out = safe_linear(context.transpose(1, 2).reshape({batch, seq, nh * vh}), o_w, o_scale_t);

        // ── 3. Residual ──
        auto residual = hidden + attn_out;

        // ── 4. MLP RMSNorm ──
        auto& post_norm_w = *reinterpret_cast<at::Tensor*>(post_norm_weight);
        auto mlp_input = rms_norm(residual, post_norm_w.to(dtype), rms_eps);

        // ── 5. MoE or Dense MLP ──
        at::Tensor mlp_output;
        if (is_moe_layer && gate_weight) {
            // MoE: reuse v4_glm5_moe_layer logic inline
            auto& gate_w = *reinterpret_cast<at::Tensor*>(gate_weight);
            auto& sg = *reinterpret_cast<at::Tensor*>(shared_gate);
            auto& su = *reinterpret_cast<at::Tensor*>(shared_up);
            auto& sd = *reinterpret_cast<at::Tensor*>(shared_down);
            auto sg_scale = shared_gate_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(shared_gate_scale)) : std::nullopt;
            auto su_scale = shared_up_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(shared_up_scale)) : std::nullopt;
            auto sd_scale = shared_down_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(shared_down_scale)) : std::nullopt;
            // Shared expert: merge gate+up for BF16 path
            int64_t s_inter = sg.size(0);
            auto shared_out = [&]() -> at::Tensor {
                if (sg_scale.has_value() || su_scale.has_value()) {
                    auto sgo = safe_linear(mlp_input, sg, sg_scale);
                    auto suo = safe_linear(mlp_input, su, su_scale);
                    return safe_linear(at::silu(sgo) * suo, sd, sd_scale);
                } else {
                    auto gate_up_w = at::cat({sg, su}, 0);
                    auto gu = at::linear(mlp_input, gate_up_w);
                    auto activated = at::silu(gu.narrow(-1, 0, s_inter)) * gu.narrow(-1, s_inter, s_inter);
                    return safe_linear(activated, sd, sd_scale);
                }
            }();

            auto router_logits = at::linear(mlp_input, gate_w.to(dtype));
            auto scores = at::sigmoid(router_logits);
            auto [tw, ti] = scores.topk(topk, -1, true, true);
            auto denom = tw.sum(-1, /*keepdim=*/true);
            tw = (tw / denom) * routed_scaling_factor;

            int64_t hidden_dim = mlp_input.size(2);
            auto flat_input = mlp_input.reshape({-1, hidden_dim});
            auto tk_indices = ti.reshape({-1, topk});
            auto tk_weights = tw.reshape({-1, topk});
            auto partial = at::zeros(flat_input.sizes(), at::TensorOptions().dtype(dtype).device(device));

            // Batched MoE: stack expert weights → bmm (Megatron GroupedLinear style)
            {
                // Use cached stacked weights (avoid per-step at::stack overhead)
                auto cache_key = std::make_pair((int)mlp_input.size(1), n_local_experts);
                static std::map<std::pair<int, int>, std::tuple<at::Tensor, at::Tensor, at::Tensor>> s_inline_cache;
                at::Tensor gw_stack, uw_stack, dw_stack;
                auto cit = s_inline_cache.find(cache_key);
                if (cit != s_inline_cache.end()) {
                    auto& [a, b, c] = cit->second;
                    gw_stack = a; uw_stack = b; dw_stack = c;
                } else {
                    gw_stack = at::stack(std::vector<at::Tensor>([&]{
                        std::vector<at::Tensor> v;
                        for (int e = 0; e < n_local_experts; e++)
                            v.push_back(*reinterpret_cast<at::Tensor*>(expert_gate_weights[e]));
                        return v;
                    }()), 0);
                    uw_stack = at::stack(std::vector<at::Tensor>([&]{
                        std::vector<at::Tensor> v;
                        for (int e = 0; e < n_local_experts; e++)
                            v.push_back(*reinterpret_cast<at::Tensor*>(expert_up_weights[e]));
                        return v;
                    }()), 0);
                    dw_stack = at::stack(std::vector<at::Tensor>([&]{
                        std::vector<at::Tensor> v;
                        for (int e = 0; e < n_local_experts; e++)
                            v.push_back(*reinterpret_cast<at::Tensor*>(expert_down_weights[e]));
                        return v;
                    }()), 0);
                    s_inline_cache[cache_key] = std::make_tuple(gw_stack, uw_stack, dw_stack);
                }

                int64_t N = flat_input.size(0);
                // Assign each token to its first local expert
                auto local_idx = at::full({N, topk}, -1.0, at::TensorOptions().dtype(at::kFloat).device(device));
                for (int e = 0; e < n_local_experts; e++) {
                    auto mask = tk_indices.eq(local_expert_indices[e]);
                    local_idx = local_idx.masked_fill(mask, (float)e);
                }
                auto assigned = local_idx.clamp_min(0).to(at::kLong).narrow(-1, 0, 1).squeeze(-1);
                auto [sorted_e, sort_order] = assigned.sort(0);
                auto sorted_tokens = flat_input.index_select(0, sort_order);
                auto sorted_w = tk_weights.gather(0, sort_order.narrow(1, 0, 1).expand({-1, topk}));

                int64_t max_per = (N + n_local_experts - 1) / n_local_experts + 16;
                auto gathered = at::zeros({n_local_experts, max_per, hidden_dim},
                    at::TensorOptions().dtype(dtype).device(device));
                auto valid_mask = at::zeros({n_local_experts, max_per},
                    at::TensorOptions().dtype(at::kBool).device(device));
                auto w_gathered = at::zeros({n_local_experts, max_per, 1},
                    at::TensorOptions().dtype(dtype).device(device));

                auto offsets = at::cumsum(at::cat(std::vector<at::Tensor>({
                    at::zeros({1}, at::TensorOptions().dtype(at::kLong).device(device)),
                    at::histc(sorted_e.to(at::kFloat), n_local_experts, 0, n_local_experts - 1).to(at::kLong)
                        .narrow(0, 0, n_local_experts - 1)
                })), 0);
                auto arange = at::arange(N, at::TensorOptions().dtype(at::kLong).device(device));
                auto starts = offsets.index_select(0, sorted_e.clamp_max(n_local_experts - 1));
                auto pos = arange - starts;
                auto valid = pos.lt(max_per);
                auto vidx = valid.nonzero().squeeze(-1);

                if (vidx.size(0) > 0) {
                    auto ve = sorted_e.index_select(0, vidx);
                    auto vp = pos.index_select(0, vidx);
                    auto vt = sorted_tokens.index_select(0, vidx);
                    auto vw = sorted_w.index_select(0, vidx).sum(-1, true).to(dtype);
                    gathered.index_put_({ve, vp}, vt);
                    valid_mask.index_put_({ve, vp}, true);
                    w_gathered.index_put_({ve, vp}, vw);
                }

                auto go = at::bmm(gathered, gw_stack.transpose(-1, -2));
                auto uo = at::bmm(gathered, uw_stack.transpose(-1, -2));
                auto act = at::silu(go) * uo;
                auto outs = at::bmm(act, dw_stack.transpose(-1, -2));
                outs = outs * valid_mask.unsqueeze(-1).to(dtype) * w_gathered;

                if (vidx.size(0) > 0) {
                    auto ve2 = sorted_e.index_select(0, vidx);
                    auto vp2 = pos.index_select(0, vidx);
                    auto vt2 = sort_order.index_select(0, vidx).squeeze(-1);
                    auto out_flat = outs.index_select(0, ve2).index_select(1, vp2);
                    partial.index_add_(0, vt2, out_flat);
                }

                gathered = at::Tensor(); go = at::Tensor(); uo = at::Tensor(); act = at::Tensor();
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
            mlp_output = partial.reshape({1, -1, hidden_dim}) + shared_out;
        } else {
            // Dense MLP
            auto& dg = *reinterpret_cast<at::Tensor*>(dense_gate);
            auto& du = *reinterpret_cast<at::Tensor*>(dense_up);
            auto& dd = *reinterpret_cast<at::Tensor*>(dense_down);
            auto dg_scale = dense_gate_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(dense_gate_scale)) : std::nullopt;
            auto du_scale = dense_up_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(dense_up_scale)) : std::nullopt;
            auto dd_scale = dense_down_scale ? std::optional<at::Tensor>(*reinterpret_cast<at::Tensor*>(dense_down_scale)) : std::nullopt;
            auto go = safe_linear(mlp_input, dg, dg_scale);
            auto uo = safe_linear(mlp_input, du, du_scale);
            auto act = at::silu(go) * uo;
            mlp_output = safe_linear(act, dd, dd_scale);
        }

        // ── 6. Residual ──
        auto new_hidden = residual + mlp_output;
        // Release intermediate tensors from this layer to prevent memory accumulation
        c10::cuda::CUDACachingAllocator::emptyCache();
        return new at::Tensor(std::move(new_hidden));
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_glm5_layer_forward] FAILED: %s\n", e.what());
        return nullptr;
    }
}

// ══════════════════════════════════════════════════════════════════════
// v4_stream_wait_event — make PyTorch's current CUDA stream wait for an event
// This is the key for async overlap: CPU doesn't block, GPU handles the dependency.
// ══════════════════════════════════════════════════════════════════════
void v4_stream_wait_event(int device_id, void* event_ptr) {
    try {
        cudaSetDevice(device_id);
        auto stream = c10::cuda::getCurrentCUDAStream(c10::cuda::current_device());
        cudaEvent_t event = reinterpret_cast<cudaEvent_t>(event_ptr);
        cudaStreamWaitEvent(stream.stream(), event, 0);
    } catch (const std::exception& e) {
        fprintf(stderr, "[v4_stream_wait_event] FAILED: %s\n", e.what());
    }
}

} // extern "C"
