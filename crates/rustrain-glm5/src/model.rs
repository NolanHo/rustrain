use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use tch::{no_grad, Device, Kind, Reduction, Tensor};
use tracing::info;

use rustrain_checkpoint::safetensors::{read_safetensors_dir, tensor};
use rustrain_core::runtime::{Config, RunPaths};

// ── GLM-5.2 Config ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Glm5RuntimeConfig {
    pub num_hidden_layers: usize,
    pub num_attention_heads: i64,
    pub hidden_size: i64,
    pub kv_lora_rank: i64,
    pub q_lora_rank: i64,
    pub qk_nope_head_dim: i64,
    pub qk_rope_head_dim: i64,
    pub v_head_dim: i64,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub first_k_dense_replace: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub moe_intermediate_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: i64,
    pub scoring_func: String,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
    // DSA indexer
    pub index_head_dim: i64,
    pub index_n_heads: i64,
    pub index_topk: i64,
    // IndexShare
    pub indexer_types: Vec<String>, // ["full","full","full","shared",...]
    pub index_topk_freq: i64,       // recompute top-k every N layers
    pub index_skip_topk_offset: i64, // skip first N layers
    pub index_share_for_mtp_iteration: bool,
    // RoPE
    pub rope_interleave: bool,
    // YaRN
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: f64,
    pub rope_beta_fast: f64,
    pub rope_beta_slow: f64,
    pub rope_original_max_pos: i64,
    // FP8
    pub expert_dtype: String, // "fp8" or "bf16"
}

impl Glm5RuntimeConfig {
    pub fn is_moe_layer(&self, layer: usize) -> bool {
        layer >= self.first_k_dense_replace
    }

    /// Returns the layer index whose indexer weights this layer should use.
    /// For "full" layers, returns self. For "shared" layers, returns the
    /// nearest preceding "full" layer.
    pub fn indexer_source_layer(&self, layer: usize) -> usize {
        if layer >= self.indexer_types.len() {
            return layer;
        }
        if self.indexer_types[layer] == "full" {
            return layer;
        }
        // Walk backwards to find the nearest "full" layer
        for l in (0..layer).rev() {
            if l < self.indexer_types.len() && self.indexer_types[l] == "full" {
                return l;
            }
        }
        layer // fallback
    }

    /// Whether this layer should skip sparse attention (first N layers)
    pub fn should_skip_topk(&self, layer: usize) -> bool {
        (layer as i64) < self.index_skip_topk_offset
    }
}

// ── Config Parsing ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Glm5ModelConfig {
    model_type: String,
    hidden_size: i64,
    num_hidden_layers: usize,
    num_attention_heads: i64,
    vocab_size: i64,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    kv_lora_rank: Option<i64>,
    #[serde(default)]
    q_lora_rank: Option<i64>,
    #[serde(default)]
    qk_nope_head_dim: Option<i64>,
    #[serde(default)]
    qk_rope_head_dim: Option<i64>,
    #[serde(default)]
    v_head_dim: Option<i64>,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    #[serde(default)]
    first_k_dense_replace: Option<usize>,
    #[serde(default)]
    n_routed_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
    #[serde(default)]
    n_shared_experts: Option<usize>,
    #[serde(default)]
    moe_intermediate_size: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    scoring_func: Option<String>,
    #[serde(default)]
    n_group: Option<usize>,
    #[serde(default)]
    topk_group: Option<usize>,
    #[serde(default)]
    routed_scaling_factor: Option<f64>,
    // DSA indexer
    #[serde(default)]
    index_head_dim: Option<i64>,
    #[serde(default)]
    index_n_heads: Option<i64>,
    #[serde(default)]
    index_topk: Option<i64>,
    #[serde(default)]
    indexer_types: Option<Vec<String>>,
    #[serde(default)]
    index_topk_freq: Option<i64>,
    #[serde(default)]
    index_skip_topk_offset: Option<i64>,
    #[serde(default)]
    index_share_for_mtp_iteration: Option<bool>,
    // RoPE
    #[serde(default)]
    rope_interleave: Option<bool>,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    max_position_embeddings: Option<i64>,
    #[serde(default)]
    expert_dtype: Option<String>,
}

pub fn read_glm5_config(path: &Path) -> Result<Glm5RuntimeConfig> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let c: Glm5ModelConfig = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse {}", path.display()))?;

    let n_layers = c.num_hidden_layers;
    let indexer_types = c.indexer_types.unwrap_or_else(|| {
        // Default: all "full"
        vec!["full".to_string(); n_layers]
    });

    Ok(Glm5RuntimeConfig {
        num_hidden_layers: n_layers,
        num_attention_heads: c.num_attention_heads,
        hidden_size: c.hidden_size,
        kv_lora_rank: c.kv_lora_rank.unwrap_or(512),
        q_lora_rank: c.q_lora_rank.unwrap_or(1536),
        qk_nope_head_dim: c.qk_nope_head_dim.unwrap_or(128),
        qk_rope_head_dim: c.qk_rope_head_dim.unwrap_or(64),
        v_head_dim: c.v_head_dim.unwrap_or(256),
        rope_theta: c.rope_theta.unwrap_or(8000000.0),
        rms_norm_eps: c.rms_norm_eps.unwrap_or(1e-6),
        tie_word_embeddings: c.tie_word_embeddings,
        first_k_dense_replace: c.first_k_dense_replace.unwrap_or(3),
        n_routed_experts: c.n_routed_experts.unwrap_or(0),
        num_experts_per_tok: c.num_experts_per_tok.unwrap_or(0),
        n_shared_experts: c.n_shared_experts.unwrap_or(0),
        moe_intermediate_size: c.moe_intermediate_size.unwrap_or(0),
        intermediate_size: c.intermediate_size.unwrap_or(18432),
        vocab_size: c.vocab_size,
        scoring_func: c.scoring_func.unwrap_or_else(|| "sigmoid".to_string()),
        n_group: c.n_group.unwrap_or(1),
        topk_group: c.topk_group.unwrap_or(1),
        routed_scaling_factor: c.routed_scaling_factor.unwrap_or(1.0),
        // DSA indexer
        index_head_dim: c.index_head_dim.unwrap_or(128),
        index_n_heads: c.index_n_heads.unwrap_or(64),
        index_topk: c.index_topk.unwrap_or(2048),
        indexer_types,
        index_topk_freq: c.index_topk_freq.unwrap_or(1),
        index_skip_topk_offset: c.index_skip_topk_offset.unwrap_or(0),
        index_share_for_mtp_iteration: c.index_share_for_mtp_iteration.unwrap_or(false),
        // RoPE
        rope_interleave: c.rope_interleave.unwrap_or(true),
        // YaRN
        rope_scaling_type: c.rope_scaling.as_ref().and_then(|v| {
            v.get("type").and_then(|t| t.as_str()).map(|s| s.to_string())
        }),
        rope_scaling_factor: c.rope_scaling.as_ref().and_then(|v| v.get("factor")).and_then(|f| f.as_f64()).unwrap_or(1.0),
        rope_beta_fast: c.rope_scaling.as_ref().and_then(|v| v.get("beta_fast")).and_then(|f| f.as_f64()).unwrap_or(32.0),
        rope_beta_slow: c.rope_scaling.as_ref().and_then(|v| v.get("beta_slow")).and_then(|f| f.as_f64()).unwrap_or(1.0),
        rope_original_max_pos: c.rope_scaling.as_ref().and_then(|v| v.get("original_max_position_embeddings")).and_then(|f| f.as_i64()).unwrap_or(4096),
        expert_dtype: c.expert_dtype.unwrap_or_else(|| "bf16".to_string()),
    })
}

// ── Compute dtype ───────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm5ComputeDType {
    Fp32,
    Bf16,
}

impl Glm5ComputeDType {
    pub fn kind(self) -> Kind {
        match self {
            Self::Fp32 => Kind::Float,
            Self::Bf16 => Kind::BFloat16,
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────

pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let dtype = input.kind();
    let variance = input
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    let result = input * (variance + eps).rsqrt().to_kind(dtype) * weight;
    result.to_kind(dtype)
}

/// RMSNorm with bias (indexer k_norm uses weight + bias)
fn rms_norm_with_bias(input: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Tensor {
    let dtype = input.kind();
    let variance = input
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    let result = input * (variance + eps).rsqrt().to_kind(dtype) * weight + bias;
    result.to_kind(dtype)
}

// ── RoPE ────────────────────────────────────────────────────────

pub fn rope_cos_sin(seq_len: usize, head_dim: i64, theta: f64, device: Device) -> (Tensor, Tensor) {
    let positions = Tensor::arange(seq_len as i64, (Kind::Float, device));
    let dim_indices = Tensor::arange(head_dim / 2, (Kind::Float, device));
    let inv_freq = (dim_indices * (2.0 / head_dim as f64)) * (1.0 / theta.ln());
    let inv_freq = inv_freq.exp();
    let freqs = positions.outer(&inv_freq); // [seq_len, head_dim/2]
    let cos = freqs.cos();
    let sin = freqs.sin();
    let cos = Tensor::cat(&[&cos, &cos], -1);
    let sin = Tensor::cat(&[&sin, &sin], -1);
    (cos, sin)
}

pub fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let seq_len = x.size()[2];
    let cos = cos.narrow(0, 0, seq_len).unsqueeze(0).unsqueeze(0);
    let sin = sin.narrow(0, 0, seq_len).unsqueeze(0).unsqueeze(0);
    let half = x.size()[x.size().len() - 1] / 2;
    let x1 = x.narrow(-1, 0, half);
    let x2 = x.narrow(-1, half, half);
    let rotated = Tensor::cat(&[&x2.neg(), &x1], -1);
    x * cos + rotated * sin
}

pub fn apply_rotary_interleave(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let seq_len = x.size()[2];
    let half = x.size()[x.size().len() - 1] / 2;
    let cos = cos.narrow(0, 0, seq_len).unsqueeze(0).unsqueeze(0);
    let sin = sin.narrow(0, 0, seq_len).unsqueeze(0).unsqueeze(0);
    let x_even = x.slice(-1, 0, None, 2);
    let x_odd = x.slice(-1, 1, None, 2);
    let rotated = Tensor::cat(&[&x_odd.neg(), &x_even], -1);
    x * cos + rotated * sin
}

pub fn apply_rotary_dispatch(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    interleave: bool,
) -> Tensor {
    if interleave {
        apply_rotary_interleave(x, cos, sin)
    } else {
        apply_rotary(x, cos, sin)
    }
}

// ── Attention Weights ────────────────────────────────────────────

pub struct Glm5AttentionWeights {
    pub q_a_proj: Tensor,
    pub q_a_layernorm: Tensor,
    pub q_b_proj: Tensor,
    pub kv_a_proj_with_mqa: Tensor,
    pub kv_a_layernorm: Tensor,
    pub kv_b_proj: Tensor,
    pub o_proj: Tensor,
    // DSA indexer (optional, present on "full" layers)
    pub indexer_k_norm_weight: Option<Tensor>,
    pub indexer_k_norm_bias: Option<Tensor>,
    pub indexer_weights_proj: Option<Tensor>,
    pub indexer_wk: Option<Tensor>,
    pub indexer_wq_b: Option<Tensor>,
}

impl Glm5AttentionWeights {
    pub fn load_with_kind(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        kind: Kind,
    ) -> Result<Self> {
        let p = format!("model.layers.{layer}.self_attn");
        let q_a_proj = tensor(weights, &format!("{p}.wq_a.weight"))?.to_kind(kind);
        let q_a_layernorm = tensor(weights, &format!("{p}.wq_a_layernorm.weight"))?.to_kind(kind);
        let q_b_proj = tensor(weights, &format!("{p}.wq_b.weight"))?.to_kind(kind);
        let kv_a = tensor(weights, &format!("{p}.wkv.weight"))?.to_kind(kind);
        let kv_a_ln = tensor(weights, &format!("{p}.wkv_a_layernorm.weight"))?.to_kind(kind);
        let kv_b = tensor(weights, &format!("{p}.wkv_b.weight"))?.to_kind(kind);
        let o_proj = tensor(weights, &format!("{p}.wo.weight"))?.to_kind(kind);

        // Indexer weights — may not exist for "shared" layers
        let indexer_k_norm_weight = weights
            .get(&format!("{p}.indexer.k_norm.weight"))
            .map(|t| t.to_kind(kind));
        let indexer_k_norm_bias = weights
            .get(&format!("{p}.indexer.k_norm.bias"))
            .map(|t| t.to_kind(kind));
        let indexer_weights_proj = weights
            .get(&format!("{p}.indexer.weights_proj.weight"))
            .map(|t| t.to_kind(kind));
        let indexer_wk = weights
            .get(&format!("{p}.indexer.wk.weight"))
            .map(|t| t.to_kind(kind));
        let indexer_wq_b = weights
            .get(&format!("{p}.indexer.wq_b.weight"))
            .map(|t| t.to_kind(kind));

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa: kv_a,
            kv_a_layernorm: kv_a_ln,
            kv_b_proj: kv_b,
            o_proj,
            indexer_k_norm_weight,
            indexer_k_norm_bias,
            indexer_weights_proj,
            indexer_wk,
            indexer_wq_b,
        })
    }

    pub fn load_raw(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
    ) -> Result<Self> {
        Self::load_with_kind(weights, layer, Kind::Float)
    }
}

// ── DSA Indexer State (for IndexShare) ───────────────────────────

/// Holds the top-k selection result from a "full" indexer layer.
/// "shared" layers reuse this instead of recomputing.
pub struct IndexShareState {
    /// Sparse mask: [batch, num_heads, seq, seq] — which KV positions to attend to
    pub sparse_mask: Tensor,
    /// Indexer bias: [batch, num_heads, seq, seq] — additive bias to attention scores
    pub idx_bias: Tensor,
    /// Which layer produced this state
    pub source_layer: usize,
}

impl IndexShareState {
    pub fn empty(batch: i64, num_heads: i64, seq: i64, device: Device) -> Self {
        Self {
            sparse_mask: Tensor::zeros([batch, num_heads, seq, seq], (Kind::Float, device)),
            idx_bias: Tensor::zeros([batch, num_heads, seq, seq], (Kind::Float, device)),
            source_layer: 0,
        }
    }
}

// ── DSA Attention ────────────────────────────────────────────────

/// GLM-5.2 DSA attention with IndexShare support.
///
/// - `indexer_weights`: weights from the source layer (may differ from `attn` if shared)
/// - `index_share_state`: if Some, reuse the top-k mask from a previous "full" layer
/// - Returns: (attention_output, updated_index_share_state)
pub fn glm5_dsa_attention(
    input: &Tensor,
    attn: &Glm5AttentionWeights,
    indexer_weights: &Glm5AttentionWeights, // may be same as attn for "full" layers
    config: &Glm5RuntimeConfig,
    index_share_state: &mut Option<IndexShareState>,
    layer: usize,
) -> Tensor {
    let batch = input.size()[0];
    let seq = input.size()[1];
    let num_heads = config.num_attention_heads;
    let qk_nope = config.qk_nope_head_dim;
    let qk_rope = config.qk_rope_head_dim;
    let v_head = config.v_head_dim;
    let kv_lora = config.kv_lora_rank;
    let idx_head_dim = config.index_head_dim;
    let idx_n_heads = config.index_n_heads;
    let idx_topk = config.index_topk;

    // ── Standard MLA Q/K/V projections ──
    let q_a = input.linear::<&Tensor>(&attn.q_a_proj, None);
    let q_a_normed = rms_norm(&q_a, &attn.q_a_layernorm, config.rms_norm_eps);
    let q_b = q_a_normed.linear::<&Tensor>(&attn.q_b_proj, None);
    let q = q_b
        .reshape([batch, seq, num_heads, qk_nope + qk_rope])
        .transpose(1, 2);
    let q_nope = q.narrow(-1, 0, qk_nope);
    let q_rope = q.narrow(-1, qk_nope, qk_rope);

    let kv_a = input.linear::<&Tensor>(&attn.kv_a_proj_with_mqa, None);
    let kv_a_normed = rms_norm(&kv_a, &attn.kv_a_layernorm, config.rms_norm_eps);
    let kv_lora_part = kv_a_normed.narrow(-1, 0, kv_lora);
    let k_rope = kv_a_normed.narrow(-1, kv_lora, qk_rope);
    let kv_b = kv_lora_part.linear::<&Tensor>(&attn.kv_b_proj, None);
    let kv_b = kv_b.reshape([batch, seq, num_heads, qk_nope + v_head]);
    let k_nope = kv_b.narrow(-1, 0, qk_nope).transpose(1, 2);
    let v = kv_b.narrow(-1, qk_nope, v_head).transpose(1, 2);

    let k_rope_expanded = k_rope
        .unsqueeze(2)
        .transpose(1, 2)
        .expand([batch, num_heads, seq, qk_rope], false);
    let (cos, sin) = rope_cos_sin(seq as usize, qk_rope, config.rope_theta, input.device());
    let cos = cos.to_kind(input.kind());
    let sin = sin.to_kind(input.kind());
    let q_rope_rotated = apply_rotary_dispatch(&q_rope, &cos, &sin, config.rope_interleave);
    let k_rope_rotated = apply_rotary_dispatch(&k_rope_expanded, &cos, &sin, config.rope_interleave);

    let q_full = Tensor::cat(&[&q_nope, &q_rope_rotated], -1);
    let k_full = Tensor::cat(&[&k_nope, &k_rope_rotated], -1);

    let scale = 1.0 / ((qk_nope + qk_rope) as f64).sqrt();
    let mut scores = q_full.matmul(&k_full.transpose(-2, -1)) * scale;

    // ── DSA Indexer ──
    let should_compute_topk = !config.should_skip_topk(layer)
        && (index_share_state.is_none() || layer % (config.index_topk_freq as usize) == 0);

    if let (Some(wq_b), Some(wk), Some(k_norm_w), Some(k_norm_b), Some(weights_proj)) = (
        &indexer_weights.indexer_wq_b,
        &indexer_weights.indexer_wk,
        &indexer_weights.indexer_k_norm_weight,
        &indexer_weights.indexer_k_norm_bias,
        &indexer_weights.indexer_weights_proj,
    ) {
        if should_compute_topk {
            // 1. Indexer Q: from q_a → wq_b → [batch, seq, n_heads * idx_head_dim]
            let idx_q = q_a.linear::<&Tensor>(wq_b, None);
            let idx_q = idx_q
                .reshape([batch, seq, num_heads, idx_head_dim])
                .transpose(1, 2);

            // 2. Indexer K: from kv_lora_normed → wk → k_norm → [batch, seq, idx_n_heads * idx_head_dim]
            let idx_k_raw = kv_lora_part.linear::<&Tensor>(wk, None);
            let idx_k_normed = rms_norm_with_bias(&idx_k_raw, k_norm_w, k_norm_b, config.rms_norm_eps);
            let idx_k = idx_k_normed
                .reshape([batch, seq, idx_n_heads, idx_head_dim])
                .transpose(1, 2);

            // 3. Indexer attention scores
            let idx_k_expanded = if idx_n_heads != num_heads {
                idx_k
                    .mean_dim([1].as_slice(), true, Kind::Float)
                    .expand([batch, num_heads, seq, idx_head_dim], false)
            } else {
                idx_k
            };
            let idx_scale = 1.0 / (idx_head_dim as f64).sqrt();
            let idx_scores = idx_q.matmul(&idx_k_expanded.transpose(-2, -1)) * idx_scale;

            // 4. Top-k selection
            let actual_topk = idx_topk.min(seq as i64);
            let (_, topk_indices) = idx_scores.topk(actual_topk, -1, true, true);

            // 5. Sparse mask
            let sparse_mask = {
                let ones = Tensor::ones(
                    [batch, num_heads, seq, actual_topk],
                    (Kind::Float, input.device()),
                );
                let mut mask = Tensor::zeros(
                    [batch as i64, num_heads, seq as i64, seq as i64],
                    (Kind::Float, input.device()),
                );
                mask.scatter_(-1, &topk_indices, &ones);
                mask
            };

            // 6. Indexer bias
            let idx_k_proj = idx_k_normed
                .linear::<&Tensor>(weights_proj, None)
                .reshape([batch, seq, num_heads, idx_head_dim])
                .transpose(1, 2);
            let idx_bias = idx_q.matmul(&idx_k_proj.transpose(-2, -1)) * idx_scale;

            // Save state for IndexShare
            *index_share_state = Some(IndexShareState {
                sparse_mask,
                idx_bias,
                source_layer: layer,
            });
        }
    } else {
        // No indexer weights → full causal attention fallback
        *index_share_state = None;
    }

    // Apply sparse mask
    if let Some(state) = index_share_state {
        let causal_mask =
            Tensor::ones([seq as i64, seq as i64], (Kind::Bool, input.device())).triu(1);
        let causal_f = causal_mask
            .unsqueeze(0)
            .unsqueeze(0)
            .expand([batch as i64, num_heads, seq as i64, seq as i64], false)
            .to_kind(Kind::Float);
        let combined = &state.sparse_mask * &causal_f;
        let valid = combined.gt(0.0).to_kind(Kind::Bool);
        scores = scores.masked_fill(&valid.logical_not(), f64::NEG_INFINITY);
        scores = scores + &state.idx_bias;
    } else {
        let causal_mask =
            Tensor::ones([seq as i64, seq as i64], (Kind::Bool, input.device())).triu(1);
        scores = scores.masked_fill(&causal_mask, f64::NEG_INFINITY);
    }

    let probs = scores.softmax(-1, Kind::Float).to_kind(v.kind());
    let context = probs.matmul(&v);
    let context = context
        .transpose(1, 2)
        .reshape([batch, seq, num_heads * v_head]);

    context.linear::<&Tensor>(&attn.o_proj, None)
}

// ── MLP ─────────────────────────────────────────────────────────

pub fn glm5_mlp(input: &Tensor, gate: &Tensor, up: &Tensor, down: &Tensor) -> Tensor {
    let gate_out = input.linear::<&Tensor>(gate, None);
    let up_out = input.linear::<&Tensor>(up, None);
    let activated = gate_out.silu() * up_out;
    activated.linear::<&Tensor>(down, None)
}

/// FP8-aware MLP: uses V4's fp8_linear for GEMM when weights are FP8 tensors.
/// Falls back to standard bf16 linear otherwise.
pub fn glm5_mlp_fp8(
    input: &Tensor,
    gate: &Tensor,
    up: &Tensor,
    down: &Tensor,
    use_fp8: bool,
) -> Tensor {
    if use_fp8 && rustrain_deepseek_v4::fp8_kernel::is_fp8_kernel_available() {
        // FP8 weights have .scale_f companion tensors.
        // For now, fall back to standard linear — FP8 weight loading
        // is handled in session_ep.rs where scale tensors are loaded.
        // This will be activated when GLM-5.2 FP8 weights are available.
        glm5_mlp(input, gate, up, down)
    } else {
        glm5_mlp(input, gate, up, down)
    }
}

// ── MoE ─────────────────────────────────────────────────────────

pub fn glm5_moe_mlp(
    input: &Tensor,
    gate: &Tensor,
    shared_gate: &Tensor,
    shared_up: &Tensor,
    shared_down: &Tensor,
    experts: &[(Tensor, Tensor, Tensor)],
    num_experts_per_tok: usize,
    scoring_func: &str,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f64,
) -> Tensor {
    let shared_output = glm5_mlp(input, shared_gate, shared_up, shared_down);
    let router_logits = input.linear::<&Tensor>(gate, None);
    let n_experts = experts.len() as i64;

    let (topk_weights, topk_indices) = if scoring_func == "sigmoid" {
        let scores = router_logits.sigmoid();
        if n_group > 1 {
            let epg = n_experts / n_group as i64;
            let group_scores = scores
                .reshape([-1, n_group as i64, epg])
                .sum_dim_intlist([-1].as_slice(), true, Kind::Float)
                .squeeze_dim(-1);
            let (_, group_idx) = group_scores.topk(topk_group as i64, -1, true, true);
            let mut group_mask = Tensor::zeros(
                [group_scores.size()[0], n_group as i64],
                (scores.kind(), scores.device()),
            );
            let _ = group_mask.scatter_(
                -1,
                &group_idx,
                &Tensor::ones([group_scores.size()[0], n_group as i64], (scores.kind(), scores.device())),
            );
            let expert_mask = group_mask
                .unsqueeze(-1)
                .expand([-1, -1, epg], false)
                .reshape([-1, n_experts]);
            let masked_scores = scores * &expert_mask;
            let (w, i) = masked_scores.topk(num_experts_per_tok as i64, -1, true, true);
            (w, i)
        } else {
            scores.topk(num_experts_per_tok as i64, -1, true, true)
        }
    } else {
        // softmax (sqrtsoftplus variant)
        let logits = router_logits.softmax(-1, Kind::Float);
        logits.topk(num_experts_per_tok as i64, -1, true, true)
    };

    // Normalize weights
    let denom = topk_weights.sum_dim_intlist([-1].as_slice(), true, Kind::Float);
    let topk_weights = (topk_weights / denom) * routed_scaling_factor;

    let batch = input.size()[0];
    let seq = input.size()[1];
    let flat_input = input.reshape([-1, input.size()[2]]);

    let mut output = Tensor::zeros(
        flat_input.size(),
        (input.kind(), input.device()),
    );

    for e in 0..n_experts as usize {
        let mask = topk_indices.eq(e as i64).to_kind(input.kind());
        let count = mask.sum(Kind::Float).double_value(&[]) as i64;
        if count == 0 {
            continue;
        }
        let (gate_w, up_w, down_w) = &experts[e];
        let expert_out = glm5_mlp(&flat_input, gate_w, up_w, down_w);
        let weight = topk_weights.narrow(-1, e as i64, 1).unsqueeze(-1);
        let mask_expanded = mask.unsqueeze(-1).expand([-1, expert_out.size()[1]], false);
        let contribution = expert_out * &mask_expanded * weight;
        output = output + contribution;
    }

    let output = output.reshape([batch, seq, -1]);
    output + shared_output
}

// ── Dense Layer Weights ─────────────────────────────────────────

pub struct Glm5DenseLayerWeights {
    pub input_norm: Tensor,
    pub attn: Glm5AttentionWeights,
    pub post_attention_norm: Tensor,
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
}

impl Glm5DenseLayerWeights {
    pub fn load_raw(weights: &BTreeMap<String, Tensor>, layer: usize) -> Result<Self> {
        let p = format!("model.layers.{layer}");
        Ok(Self {
            input_norm: tensor(weights, &format!("{p}.input_layernorm.weight"))?.shallow_clone(),
            attn: Glm5AttentionWeights::load_raw(weights, layer)?,
            post_attention_norm: tensor(weights, &format!("{p}.post_attention_layernorm.weight"))?.shallow_clone(),
            gate_proj: tensor(weights, &format!("{p}.mlp.gate_proj.weight"))?.shallow_clone(),
            up_proj: tensor(weights, &format!("{p}.mlp.up_proj.weight"))?.shallow_clone(),
            down_proj: tensor(weights, &format!("{p}.mlp.down_proj.weight"))?.shallow_clone(),
        })
    }
}

// ── MoE Layer Weights ───────────────────────────────────────────

pub struct Glm5MoeLayerWeights {
    pub input_norm: Tensor,
    pub attn: Glm5AttentionWeights,
    pub post_attention_norm: Tensor,
    pub gate: Tensor,
    pub shared_gate_proj: Tensor,
    pub shared_up_proj: Tensor,
    pub shared_down_proj: Tensor,
    pub experts: Vec<(Tensor, Tensor, Tensor)>,
}

impl Glm5MoeLayerWeights {
    pub fn load_raw(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        n_experts: usize,
    ) -> Result<Self> {
        let p = format!("model.layers.{layer}");
        let shared_prefix = format!("{p}.mlp.shared_experts");
        let mut experts = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let ep = format!("{p}.mlp.experts.{e}");
            let gate = tensor(weights, &format!("{ep}.gate_proj.weight"))?.shallow_clone();
            let up = tensor(weights, &format!("{ep}.up_proj.weight"))?.shallow_clone();
            let down = tensor(weights, &format!("{ep}.down_proj.weight"))?.shallow_clone();
            experts.push((gate, up, down));
        }
        Ok(Self {
            input_norm: tensor(weights, &format!("{p}.input_layernorm.weight"))?.shallow_clone(),
            attn: Glm5AttentionWeights::load_raw(weights, layer)?,
            post_attention_norm: tensor(weights, &format!("{p}.post_attention_layernorm.weight"))?.shallow_clone(),
            gate: tensor(weights, &format!("{p}.mlp.gate.weight"))?.shallow_clone(),
            shared_gate_proj: tensor(weights, &format!("{shared_prefix}.gate_proj.weight"))?.shallow_clone(),
            shared_up_proj: tensor(weights, &format!("{shared_prefix}.up_proj.weight"))?.shallow_clone(),
            shared_down_proj: tensor(weights, &format!("{shared_prefix}.down_proj.weight"))?.shallow_clone(),
            experts,
        })
    }
}

// ── Layer Forward ────────────────────────────────────────────────

pub fn glm5_dense_layer(
    input: &Tensor,
    weights: &Glm5DenseLayerWeights,
    config: &Glm5RuntimeConfig,
    index_share_state: &mut Option<IndexShareState>,
    layer: usize,
) -> Tensor {
    let hidden = rms_norm(input, &weights.input_norm, config.rms_norm_eps);
    let source = config.indexer_source_layer(layer);
    let indexer_weights = if source == layer {
        &weights.attn
    } else {
        // For shared layers, we need the source layer's indexer weights.
        // In the forward path, we'll load them separately. For now, use own attn.
        &weights.attn
    };
    let attn = glm5_dsa_attention(&hidden, &weights.attn, indexer_weights, config, index_share_state, layer);
    let residual = input + &attn;
    let mlp_input = rms_norm(&residual, &weights.post_attention_norm, config.rms_norm_eps);
    let mlp = glm5_mlp(&mlp_input, &weights.gate_proj, &weights.up_proj, &weights.down_proj);
    residual + mlp
}

pub fn glm5_moe_layer(
    input: &Tensor,
    weights: &Glm5MoeLayerWeights,
    config: &Glm5RuntimeConfig,
    index_share_state: &mut Option<IndexShareState>,
    indexer_weights_map: &BTreeMap<usize, Glm5AttentionWeights>,
    layer: usize,
) -> Tensor {
    let hidden = rms_norm(input, &weights.input_norm, config.rms_norm_eps);
    let source = config.indexer_source_layer(layer);
    let indexer_weights = indexer_weights_map.get(&source).unwrap_or(&weights.attn);
    let attn = glm5_dsa_attention(&hidden, &weights.attn, indexer_weights, config, index_share_state, layer);
    let residual = input + &attn;
    let mlp_input = rms_norm(&residual, &weights.post_attention_norm, config.rms_norm_eps);
    let mlp = glm5_moe_mlp(
        &mlp_input,
        &weights.gate,
        &weights.shared_gate_proj,
        &weights.shared_up_proj,
        &weights.shared_down_proj,
        &weights.experts,
        config.num_experts_per_tok,
        &config.scoring_func,
        config.n_group,
        config.topk_group,
        config.routed_scaling_factor,
    );
    residual + mlp
}

// ── Forward ──────────────────────────────────────────────────────

pub fn glm5_forward_from_ids(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &Glm5RuntimeConfig,
) -> Result<Tensor> {
    glm5_forward_from_ids_with_kind(input_ids, weights, config, Kind::Float)
}

pub fn glm5_forward_from_ids_with_kind(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &Glm5RuntimeConfig,
    kind: Kind,
) -> Result<Tensor> {
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?;
    let final_norm = tensor(weights, "model.norm.weight")?;
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false).to_kind(kind);

    let mut index_share_state: Option<IndexShareState> = None;

    // Pre-load indexer weights for all "full" layers (for IndexShare)
    let mut indexer_weights_map: BTreeMap<usize, Glm5AttentionWeights> = BTreeMap::new();
    for layer in 0..config.num_hidden_layers {
        if layer < config.indexer_types.len() && config.indexer_types[layer] == "full" {
            let attn = Glm5AttentionWeights::load_with_kind(weights, layer, kind)?;
            indexer_weights_map.insert(layer, attn);
        }
    }

    for layer in 0..config.num_hidden_layers {
        if config.is_moe_layer(layer) {
            let lw = Glm5MoeLayerWeights::load_raw(weights, layer, config.n_routed_experts)?;
            hidden = glm5_moe_layer(
                &hidden,
                &lw,
                config,
                &mut index_share_state,
                &indexer_weights_map,
                layer,
            );
        } else {
            let lw = Glm5DenseLayerWeights::load_raw(weights, layer)?;
            hidden = glm5_dense_layer(&hidden, &lw, config, &mut index_share_state, layer);
        }
        // Reset index_share_state at "full" layer boundaries
        if layer < config.indexer_types.len() && config.indexer_types[layer] == "full" {
            // State will be recomputed at next full layer
        }
    }

    let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps);
    let lm_head = if config.tie_word_embeddings {
        embed_tokens.shallow_clone()
    } else {
        tensor(weights, "lm_head.weight")?.shallow_clone()
    };
    let logits = hidden.linear::<&Tensor>(&lm_head, None);
    Ok(logits)
}

// ── Cross-entropy loss ───────────────────────────────────────────

pub fn glm5_cross_entropy_loss(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &Glm5RuntimeConfig,
) -> Result<Tensor> {
    let logits = glm5_forward_from_ids(input_ids, weights, config)?;
    let shifted = logits.narrow(1, 0, logits.size()[1] - 1);
    let targets = input_ids.narrow(1, 1, input_ids.size()[1] - 1);
    Ok(shifted
        .reshape([-1, config.vocab_size])
        .log_softmax(-1, Kind::Float)
        .g_nll_loss::<&Tensor>(&targets.reshape([-1]), None, Reduction::Mean, -100))
}
