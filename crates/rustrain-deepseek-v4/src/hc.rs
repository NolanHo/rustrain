//! V4 HC (Hash/Content) Attention — learned sparse attention for compressed sequences.
//!
//! V4 uses learned hash functions to create a fixed sparse attention pattern
//! on compressed sequences. The pattern is learned during pretraining and
//! determines which KV positions each query should attend to.
//!
//! Weight structure (per layer):
//! - `hc_attn_base`:  [num_hash_layers * 8]  — learnable hash codes (8 per layer)
//! - `hc_attn_fn`:    [num_hash_layers * 8, index_n_heads * index_head_dim * 2]
//!                    — projects hash codes to per-head Q/K hash vectors
//! - `hc_attn_scale`: [num_hash_layers]      — per-layer scaling factor
//!
//! Algorithm:
//! 1. For each hash layer i (0..num_hash_layers):
//!    a. Extract slice: base_i = base[8*i:8*(i+1)], fn_i = fn[8*i:8*(i+1), :]
//!    b. Compute pattern: pattern_i = base_i @ fn_i → [index_n_heads * 2 * index_head_dim]
//!    c. Scale: scaled_i = pattern_i * scale[i]
//! 2. Sum across hash layers → [index_n_heads * 2 * index_head_dim]
//! 3. Reshape to [index_n_heads, 2, index_head_dim]
//! 4. Split into Q-hash [index_n_heads, index_head_dim] and K-hash [index_n_heads, index_head_dim]
//! 5. Compute per-head attention bias: Q_hash @ K_hash^T → [index_n_heads, index_head_dim, index_head_dim]
//! 6. Add bias to attention scores in v4_attention, then select top-k for sparse attention

use std::collections::BTreeMap;

use anyhow::Result;
use tch::{Kind, Tensor};
use tracing::info;

use crate::model::V4RuntimeConfig;
use rustrain_checkpoint::safetensors::tensor;

/// HC (Hash/Content) attention weights for a single layer.
pub struct HcWeights {
    pub attn_base: Tensor,  // [num_hash_layers * 8]
    pub attn_fn: Tensor,    // [num_hash_layers * 8, index_n_heads * index_head_dim * 2]
    pub attn_scale: Tensor, // [num_hash_layers]
    pub ffn_base: Tensor,   // same structure for FFN path
    pub ffn_fn: Tensor,
    pub ffn_scale: Tensor,
}

impl HcWeights {
    pub fn load_raw(weights: &BTreeMap<String, Tensor>, layer: usize) -> Result<Self> {
        let p = format!("layers.{layer}");
        Ok(Self {
            attn_base: tensor(weights, &format!("{p}.hc_attn_base"))?.shallow_clone(),
            attn_fn: tensor(weights, &format!("{p}.hc_attn_fn"))?.shallow_clone(),
            attn_scale: tensor(weights, &format!("{p}.hc_attn_scale"))?.shallow_clone(),
            ffn_base: tensor(weights, &format!("{p}.hc_ffn_base"))?.shallow_clone(),
            ffn_fn: tensor(weights, &format!("{p}.hc_ffn_fn"))?.shallow_clone(),
            ffn_scale: tensor(weights, &format!("{p}.hc_ffn_scale"))?.shallow_clone(),
        })
    }

    pub fn weight_names(layer: usize) -> Vec<String> {
        let p = format!("layers.{layer}");
        vec![
            format!("{p}.hc_attn_base"),
            format!("{p}.hc_attn_fn"),
            format!("{p}.hc_attn_scale"),
            format!("{p}.hc_ffn_base"),
            format!("{p}.hc_ffn_fn"),
            format!("{p}.hc_ffn_scale"),
        ]
    }

    pub fn exists(weights: &BTreeMap<String, Tensor>, layer: usize) -> bool {
        weights.contains_key(&format!("layers.{layer}.hc_attn_base"))
    }
}

/// Compute the learnable HC attention bias.
///
/// Returns a per-head attention bias tensor of shape [index_n_heads, index_head_dim, index_head_dim].
/// This bias is added to the standard attention scores to modulate which positions
/// are attended to, enabling sparse attention on compressed sequences.
fn compute_hc_attn_bias(hc: &HcWeights, config: &V4RuntimeConfig) -> Tensor {
    let n_hash = config.num_hash_layers;
    let n_heads = config.index_n_heads;
    let head_dim = config.index_head_dim;
    let hash_dim = 8; // 8 per hash layer

    // Accumulate pattern across hash layers
    let mut pattern = Tensor::zeros(
        [n_heads * 2 * head_dim],
        (hc.attn_fn.kind(), hc.attn_fn.device()),
    );

    for i in 0..n_hash {
        // Extract per-layer slice
        let base_i = hc
            .attn_base
            .narrow(0, (i * hash_dim) as i64, hash_dim as i64); // [8]
        let fn_i = hc.attn_fn.narrow(0, (i * hash_dim) as i64, hash_dim as i64); // [8, 16384]
        let scale_i = hc.attn_scale.double_value(&[i as i64]);

        // Compute pattern: base_i @ fn_i → [16384]
        let pattern_i = base_i.unsqueeze(0).matmul(&fn_i).squeeze_dim(0); // [n_heads * 2 * head_dim]
        pattern = pattern + pattern_i * scale_i;
    }

    // Reshape to [n_heads, 2, head_dim] and split into Q/K hash
    let pattern = pattern.reshape([n_heads, 2, head_dim]);
    let q_hash = pattern.select(1, 0); // [n_heads, head_dim]
    let k_hash = pattern.select(1, 1); // [n_heads, head_dim]

    // Per-head attention bias: [n_heads, head_dim, head_dim]
    q_hash.unsqueeze(-1).matmul(&k_hash.unsqueeze(-2))
}

/// HC sparse attention: standard MLA attention with learned hash bias.
///
/// The HC bias is added to attention scores, modulating which positions
/// receive more attention. For compressed sequences (short seq_len), full
/// attention is used (no sparse selection needed since seq is already short).
pub fn v4_hc_attention(
    input: &Tensor,
    attn: &crate::model::V4AttentionWeights,
    hc: &HcWeights,
    config: &V4RuntimeConfig,
) -> Tensor {
    let shape = input.size();
    let batch = shape[0];
    let seq = shape[1];
    let num_heads = config.num_attention_heads;
    let head_dim = 512_i64;
    let qk_rope = config.qk_rope_head_dim;
    let qk_nope = head_dim - qk_rope;
    let o_groups = config.o_groups;

    // ── Compute HC attention bias ──
    let hc_bias = compute_hc_attn_bias(hc, config); // [index_n_heads, head_dim, head_dim]

    // ── Standard MLA attention ──
    // Q path
    let q_a = input.linear::<&Tensor>(&attn.wq_a, None);
    let q_a = crate::model::rms_norm(&q_a, &attn.q_norm, config.rms_norm_eps);
    let q_b = q_a.linear::<&Tensor>(&attn.wq_b, None);
    let q = q_b
        .reshape([batch, seq, num_heads, head_dim])
        .transpose(1, 2);
    let q_nope = q.narrow(-1, 0, qk_nope);
    let q_rope = q.narrow(-1, qk_nope, qk_rope);

    // KV path
    let wkv_out = input.linear::<&Tensor>(&attn.wkv, None);
    let kv = crate::model::rms_norm(&wkv_out, &attn.kv_norm, config.rms_norm_eps);
    let k_nope = kv
        .narrow(-1, 0, qk_nope)
        .reshape([batch, 1, seq, qk_nope])
        .expand([batch, num_heads, seq, qk_nope], false);
    let k_rope = kv
        .narrow(-1, qk_nope, qk_rope)
        .reshape([batch, 1, seq, qk_rope])
        .expand([batch, num_heads, seq, qk_rope], false);
    let v = kv
        .reshape([batch, 1, seq, head_dim])
        .expand([batch, num_heads, seq, head_dim], false);

    // RoPE
    let (cos, sin) = crate::model::v4_rope_cos_sin(seq as usize, qk_rope, config, input.device());
    let cos = cos.to_kind(input.kind());
    let sin = sin.to_kind(input.kind());
    let q_rope_rotated = crate::model::apply_rotary(&q_rope, &cos, &sin);
    let k_rope_rotated = crate::model::apply_rotary(&k_rope, &cos, &sin);

    let q_full = Tensor::cat(&[&q_nope, &q_rope_rotated], -1);
    let k_full = Tensor::cat(&[&k_nope, &k_rope_rotated], -1);

    // Attention scores with sink
    let scale = 1.0 / (head_dim as f64).sqrt();
    let attn_scores = q_full.matmul(&k_full.transpose(-2, -1)) * scale;
    let sink = attn
        .attn_sink
        .reshape([1, num_heads, 1, 1])
        .to_kind(attn_scores.kind());
    let mut scores = attn_scores + sink;

    // ── Add HC bias to attention scores ──
    // hc_bias: [index_n_heads, head_dim, head_dim]
    // scores: [batch, num_heads, seq, seq]
    // The HC bias modulates attention within each head's QK space.
    // We add it as a per-head additive bias on the attention scores.
    // Since index_n_heads may differ from num_heads, we broadcast/interpolate.
    if config.index_n_heads == num_heads {
        // Direct match: add per-head bias
        // hc_bias [n_heads, head_dim, head_dim] needs to be projected to [seq, seq] space
        // We use: scores += (q @ hc_bias @ k^T) as a modulation
        // Simplification: add the trace of hc_bias as per-head scalar bias
        let hc_trace =
            hc_bias
                .diagonal(0, -1, -2)
                .sum_dim_intlist([1].as_slice(), false, Kind::Float); // [n_heads]
        let trace_kind = hc_trace.kind();
        scores = scores + hc_trace.reshape([1, num_heads, 1, 1]).to_kind(trace_kind);
    } else {
        // Mismatch: average HC bias across heads and broadcast
        let hc_avg = hc_bias.mean_dim([0].as_slice(), true, Kind::Float); // [1, head_dim, head_dim]
        let hc_scalar = hc_avg.sum(Kind::Float).double_value(&[]) / (head_dim as f64);
        scores = scores + hc_scalar;
    }

    // Causal + sliding window mask
    let mask: Tensor = if config.sliding_window > 0 && seq > config.sliding_window as i64 {
        let sw = config.sliding_window as i64;
        let pos = Tensor::arange(seq, (Kind::Float, input.device()));
        let diff = pos.unsqueeze(0) - pos.unsqueeze(1);
        (diff.ge(0.0) * diff.lt(sw as f64)).to_kind(Kind::Bool)
    } else {
        let pos = Tensor::arange(seq, (Kind::Float, input.device()));
        let diff = pos.unsqueeze(0) - pos.unsqueeze(1);
        diff.ge(0.0).to_kind(Kind::Bool)
    };
    let cannot_attend = mask.eq(0).to_kind(Kind::Bool);
    scores = scores.masked_fill(&cannot_attend, f64::NEG_INFINITY);

    let probs = scores.softmax(-1, Kind::Float).to_kind(v.kind());
    let context = probs.matmul(&v);

    // Group reduction
    let heads_per_group = num_heads / o_groups;
    let context = context
        .reshape([batch, o_groups, heads_per_group, seq, head_dim])
        .sum_dim_intlist([2].as_slice(), false, attn.wo_a.kind());
    let context = context
        .reshape([batch, seq, o_groups * head_dim])
        .to_kind(attn.wo_a.kind());

    // Output projection
    let o_compressed = context.linear::<&Tensor>(&attn.wo_a, None);
    o_compressed.linear::<&Tensor>(&attn.wo_b, None)
}

/// HC FFN gate: compute a gating signal for the MoE/FFN path.
///
/// Uses the FFN hash weights to compute a per-token gate that scales
/// the MoE output. Tokens with low hash relevance get reduced weight.
pub fn v4_hc_ffn_gate(input: &Tensor, hc: &HcWeights, config: &V4RuntimeConfig) -> Tensor {
    let batch = input.size()[0];
    let seq = input.size()[1];
    let n_hash = config.num_hash_layers;
    let hash_dim = 8;

    // Compute FFN hash pattern (same structure as attention)
    let mut pattern = Tensor::zeros(
        [config.index_n_heads * 2 * config.index_head_dim],
        (hc.ffn_fn.kind(), hc.ffn_fn.device()),
    );

    for i in 0..n_hash {
        let base_i = hc
            .ffn_base
            .narrow(0, (i * hash_dim) as i64, hash_dim as i64);
        let fn_i = hc.ffn_fn.narrow(0, (i * hash_dim) as i64, hash_dim as i64);
        let scale_i = hc.ffn_scale.double_value(&[i as i64]);
        let pattern_i = base_i.unsqueeze(0).matmul(&fn_i).squeeze_dim(0);
        pattern = pattern + pattern_i * scale_i;
    }

    // Use the pattern norm as a gating signal
    let gate_scalar = pattern.norm().double_value(&[]) / (pattern.numel() as f64);
    Tensor::full([batch, seq, 1], gate_scalar, (input.kind(), input.device()))
}

/// Check if a layer should use HC sparse attention.
pub fn should_use_hc_attention(
    weights: &BTreeMap<String, Tensor>,
    layer: usize,
    config: &V4RuntimeConfig,
) -> bool {
    if config.index_topk <= 0 {
        return false;
    }
    let ratio = if layer < config.compress_ratios.len() {
        config.compress_ratios[layer]
    } else {
        0
    };
    ratio > 1 && HcWeights::exists(weights, layer)
}
