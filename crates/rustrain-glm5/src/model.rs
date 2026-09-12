use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Reduction, Tensor, no_grad};
use tracing::info;

use rustrain_checkpoint::safetensors::{read_safetensors_dir, tensor};
use rustrain_core::runtime::{Config, RunPaths};

/// Keep FP8 tensors as-is; convert other dtypes to `kind`.
/// This prevents `.to_kind(BFloat16)` from destroying FP8 weights
/// that need to stay FP8 for `_scaled_mm`.
pub trait KeepIfFp8 {
    fn keep_if_fp8(&self, kind: Kind) -> Tensor;
}

impl KeepIfFp8 for Tensor {
    fn keep_if_fp8(&self, kind: Kind) -> Tensor {
        if self.kind() == Kind::Float8e4m3fn {
            self.shallow_clone()
        } else {
            self.to_kind(kind)
        }
    }
}

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
    pub norm_topk_prob: bool,
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
    pub rope_attention_factor: f64,
    // Indexer RoPE (separate from main attention RoPE)
    pub indexer_rope_interleave: bool,
    // MLP layer types: "dense" or "sparse" per layer
    pub mlp_layer_types: Vec<String>,
    // Top-k method: "noaux_tc" or "groupwise"
    pub topk_method: String,
    // RoPE type: "default" or "yarn"
    pub rope_type: String,
    // MTP
    pub num_nextn_predict_layers: usize,
    // FP8
    pub expert_dtype: String, // "fp8" or "bf16"
}

impl Glm5RuntimeConfig {
    pub fn is_moe_layer(&self, layer: usize) -> bool {
        // Prefer mlp_layer_types if available, fall back to first_k_dense_replace
        if layer < self.mlp_layer_types.len() {
            self.mlp_layer_types[layer] == "sparse"
        } else {
            layer >= self.first_k_dense_replace
        }
    }

    /// Returns the layer index whose indexer weights this layer should use.
    /// For "full" layers, returns self. For "shared" layers, returns the
    /// nearest preceding "full" layer.
    pub fn try_indexer_source_layer(&self, layer: usize) -> Result<usize> {
        let indexer_type = self
            .indexer_types
            .get(layer)
            .ok_or_else(|| anyhow!("indexer type missing for layer {layer}"))?;
        if indexer_type == "full" {
            return Ok(layer);
        }
        (0..layer)
            .rev()
            .find(|&candidate| self.indexer_types[candidate] == "full")
            .ok_or_else(|| {
                anyhow!("shared indexer layer {layer} has no preceding full source layer")
            })
    }

    pub fn indexer_source_layer(&self, layer: usize) -> usize {
        self.try_indexer_source_layer(layer)
            .expect("GLM-5 config must be validated before selecting indexer weights")
    }

    /// Whether this layer reuses the preceding full layer's DSA top-k.
    pub fn should_skip_topk(&self, layer: usize) -> bool {
        self.indexer_types
            .get(layer)
            .is_some_and(|kind| kind == "shared")
    }

    pub fn should_recompute_indexer(&self, layer: usize) -> bool {
        self.indexer_types
            .get(layer)
            .is_some_and(|kind| kind == "full")
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_hidden_layers == 0 {
            bail!("num_hidden_layers must be positive");
        }
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("num_attention_heads", self.num_attention_heads),
            ("q_lora_rank", self.q_lora_rank),
            ("kv_lora_rank", self.kv_lora_rank),
            ("qk_nope_head_dim", self.qk_nope_head_dim),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
            ("v_head_dim", self.v_head_dim),
            ("index_head_dim", self.index_head_dim),
            ("index_n_heads", self.index_n_heads),
            ("index_topk", self.index_topk),
            ("vocab_size", self.vocab_size),
        ] {
            if value <= 0 {
                bail!("{name} must be positive, got {value}");
            }
        }
        if self.qk_rope_head_dim % 2 != 0 {
            bail!("qk_rope_head_dim must be even for RoPE");
        }
        if self.index_head_dim < self.qk_rope_head_dim {
            bail!("index_head_dim must be at least qk_rope_head_dim");
        }
        if self.rope_theta <= 1.0 || !self.rope_theta.is_finite() {
            bail!("rope_theta must be finite and greater than 1");
        }
        if self.rms_norm_eps <= 0.0 || !self.rms_norm_eps.is_finite() {
            bail!("rms_norm_eps must be finite and positive");
        }
        if self.index_topk_freq <= 0 {
            bail!("index_topk_freq must be positive");
        }
        if self.index_skip_topk_offset < 0
            || self.index_skip_topk_offset as usize > self.num_hidden_layers
        {
            bail!("index_skip_topk_offset is outside the layer range");
        }
        if self.indexer_types.len() != self.num_hidden_layers {
            bail!(
                "indexer_types length {} does not match num_hidden_layers {}",
                self.indexer_types.len(),
                self.num_hidden_layers
            );
        }
        if self.mlp_layer_types.len() != self.num_hidden_layers {
            bail!(
                "mlp_layer_types length {} does not match num_hidden_layers {}",
                self.mlp_layer_types.len(),
                self.num_hidden_layers
            );
        }
        for (layer, kind) in self.indexer_types.iter().enumerate() {
            if !matches!(kind.as_str(), "full" | "shared") {
                bail!("invalid indexer type {kind:?} at layer {layer}");
            }
            self.try_indexer_source_layer(layer)?;
        }
        for (layer, kind) in self.mlp_layer_types.iter().enumerate() {
            if !matches!(kind.as_str(), "dense" | "sparse") {
                bail!("invalid MLP layer type {kind:?} at layer {layer}");
            }
        }
        if self.n_routed_experts == 0 || self.num_experts_per_tok == 0 {
            bail!("GLM-5 MoE requires routed experts and experts-per-token");
        }
        if self.num_experts_per_tok > self.n_routed_experts {
            bail!("num_experts_per_tok exceeds n_routed_experts");
        }
        if self.n_group == 0 || self.n_routed_experts % self.n_group != 0 {
            bail!("n_routed_experts must be divisible by positive n_group");
        }
        if self.topk_group == 0 || self.topk_group > self.n_group {
            bail!("topk_group must be in 1..=n_group");
        }
        if !matches!(self.scoring_func.as_str(), "sigmoid" | "softmax") {
            bail!("unsupported scoring_func {:?}", self.scoring_func);
        }
        if !matches!(self.topk_method.as_str(), "noaux_tc" | "groupwise") {
            bail!("unsupported topk_method {:?}", self.topk_method);
        }
        if self.topk_method == "noaux_tc" && self.scoring_func != "sigmoid" {
            bail!("topk_method=noaux_tc requires scoring_func=sigmoid");
        }
        if !matches!(self.rope_type.as_str(), "default" | "yarn") {
            bail!("unsupported rope_type {:?}", self.rope_type);
        }
        if self.rope_type == "yarn" {
            if self.rope_scaling_type.as_deref() != Some("yarn") {
                bail!("rope_type=yarn requires a complete YaRN scaling object");
            }
            if self.rope_scaling_factor <= 1.0
                || self.rope_original_max_pos <= 0
                || self.rope_beta_fast <= self.rope_beta_slow
                || self.rope_beta_slow <= 0.0
            {
                bail!("invalid YaRN factor/original context/beta boundaries");
            }
        }
        if !matches!(self.expert_dtype.as_str(), "bf16" | "fp8") {
            bail!("unsupported expert_dtype {:?}", self.expert_dtype);
        }
        Ok(())
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
    #[serde(default = "default_true")]
    norm_topk_prob: bool,
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
    #[serde(default)]
    indexer_rope_interleave: Option<bool>,
    #[serde(default)]
    mlp_layer_types: Option<Vec<String>>,
    #[serde(default)]
    topk_method: Option<String>,
    #[serde(default)]
    num_nextn_predict_layers: Option<usize>,
    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,
}

fn default_true() -> bool {
    true
}

fn derive_indexer_types(num_layers: usize, topk_freq: i64, skip_topk_offset: i64) -> Vec<String> {
    let freq = topk_freq.max(1);
    let offset = skip_topk_offset.max(1);
    (0..num_layers)
        .map(|layer| {
            // Megatron evaluates max(layer_number - offset, 0) % freq on
            // 1-indexed layer numbers. HF exposes the same schedule in its
            // zero-indexed `indexer_types` list.
            let phase = (layer as i64 + 1 - offset).max(0);
            if phase % freq == 0 {
                "full".to_string()
            } else {
                "shared".to_string()
            }
        })
        .collect()
}

pub fn read_glm5_config(path: &Path) -> Result<Glm5RuntimeConfig> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let c: Glm5ModelConfig = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse {}", path.display()))?;

    let n_layers = c.num_hidden_layers;
    let index_topk_freq = c.index_topk_freq.unwrap_or(1);
    let index_skip_topk_offset = c.index_skip_topk_offset.unwrap_or(0);
    let indexer_types = c
        .indexer_types
        .unwrap_or_else(|| derive_indexer_types(n_layers, index_topk_freq, index_skip_topk_offset));
    let mlp_layer_types = c.mlp_layer_types.unwrap_or_else(|| {
        // Default: first_k_dense_replace layers are "dense", rest are "sparse"
        let mut v = vec!["dense".to_string(); n_layers];
        for i in c.first_k_dense_replace.unwrap_or(3)..n_layers {
            v[i] = "sparse".to_string();
        }
        v
    });

    // GLM-5.2 stores the complete RoPE schema in `rope_parameters`.  Older
    // synthetic fixtures used top-level `rope_theta`; retain that only for
    // default RoPE.  YaRN fields must never be synthesized from defaults.
    let rope_schema = c.rope_parameters.as_ref().or(c.rope_scaling.as_ref());
    let rope_type = rope_schema
        .and_then(|v| v.get("rope_type"))
        .or_else(|| rope_schema.and_then(|v| v.get("type")))
        .and_then(|t| t.as_str())
        .unwrap_or("default")
        .to_string();
    let rope_theta = rope_schema
        .and_then(|v| v.get("rope_theta"))
        .and_then(|t| t.as_f64())
        .or(c.rope_theta)
        .ok_or_else(|| {
            anyhow!("GLM-5 config must define rope_parameters.rope_theta or rope_theta")
        })?;

    let yarn_field = |name: &str| rope_schema.and_then(|v| v.get(name));
    let rope_scaling_type = if rope_type == "yarn" {
        Some("yarn".to_string())
    } else {
        None
    };
    let rope_scaling_factor = yarn_field("factor").and_then(|v| v.as_f64());
    let rope_beta_fast = yarn_field("beta_fast").and_then(|v| v.as_f64());
    let rope_beta_slow = yarn_field("beta_slow").and_then(|v| v.as_f64());
    let rope_original_max_pos =
        yarn_field("original_max_position_embeddings").and_then(|v| v.as_i64());
    let rope_attention_factor = yarn_field("attention_factor")
        .and_then(|v| v.as_f64())
        .or_else(|| {
            rope_scaling_factor.map(|factor| {
                if factor <= 1.0 {
                    1.0
                } else {
                    0.1 * factor.ln() + 1.0
                }
            })
        })
        .unwrap_or(1.0);

    if rope_type == "yarn" {
        for (name, present) in [
            ("factor", rope_scaling_factor.is_some()),
            ("beta_fast", rope_beta_fast.is_some()),
            ("beta_slow", rope_beta_slow.is_some()),
            (
                "original_max_position_embeddings",
                rope_original_max_pos.is_some(),
            ),
        ] {
            if !present {
                bail!("rope_type=yarn requires {name} in the checkpoint RoPE schema");
            }
        }
    }

    let config = Glm5RuntimeConfig {
        num_hidden_layers: n_layers,
        num_attention_heads: c.num_attention_heads,
        hidden_size: c.hidden_size,
        kv_lora_rank: c.kv_lora_rank.unwrap_or(512),
        q_lora_rank: c.q_lora_rank.unwrap_or(1536),
        qk_nope_head_dim: c.qk_nope_head_dim.unwrap_or(128),
        qk_rope_head_dim: c.qk_rope_head_dim.unwrap_or(64),
        v_head_dim: c.v_head_dim.unwrap_or(256),
        rope_theta,
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
        norm_topk_prob: c.norm_topk_prob,
        // DSA indexer
        index_head_dim: c.index_head_dim.unwrap_or(128),
        index_n_heads: c.index_n_heads.unwrap_or(64),
        index_topk: c.index_topk.unwrap_or(2048),
        indexer_types,
        index_topk_freq,
        index_skip_topk_offset,
        index_share_for_mtp_iteration: c.index_share_for_mtp_iteration.unwrap_or(false),
        // RoPE
        rope_interleave: c.rope_interleave.unwrap_or(true),
        // YaRN
        rope_scaling_type,
        rope_scaling_factor: rope_scaling_factor.unwrap_or(1.0),
        rope_beta_fast: rope_beta_fast.unwrap_or(0.0),
        rope_beta_slow: rope_beta_slow.unwrap_or(0.0),
        rope_original_max_pos: rope_original_max_pos.unwrap_or(0),
        rope_attention_factor,
        // Indexer RoPE (separate from main attention RoPE)
        indexer_rope_interleave: c.indexer_rope_interleave.unwrap_or(true),
        // MLP layer types
        mlp_layer_types,
        // Top-k method
        topk_method: c.topk_method.unwrap_or_else(|| "noaux_tc".to_string()),
        // RoPE type
        rope_type,
        // MTP
        num_nextn_predict_layers: c.num_nextn_predict_layers.unwrap_or(0),
        // FP8
        expert_dtype: c.expert_dtype.unwrap_or_else(|| "bf16".to_string()),
    };
    config.validate()?;
    Ok(config)
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
    let weight = weight.to_kind(dtype);
    let variance = input
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    let result = input * (variance + eps).rsqrt().to_kind(dtype) * &weight;
    result.to_kind(dtype)
}

/// RMSNorm with bias (indexer k_norm uses weight + bias)
pub fn rms_norm_with_bias(input: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Tensor {
    let dtype = input.kind();
    let weight = weight.to_kind(dtype);
    let bias = bias.to_kind(dtype);
    let variance = input
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    let result = input * (variance + eps).rsqrt().to_kind(dtype) * &weight + &bias;
    result.to_kind(dtype)
}

// ── RoPE ────────────────────────────────────────────────────────

pub fn rope_inv_frequencies(head_dim: i64, theta: f64) -> Result<Vec<f64>> {
    if head_dim <= 0 || head_dim % 2 != 0 {
        bail!("RoPE head_dim must be a positive even number");
    }
    if theta <= 1.0 || !theta.is_finite() {
        bail!("RoPE theta must be finite and greater than 1");
    }
    Ok((0..head_dim / 2)
        .map(|index| theta.powf(-(2.0 * index as f64 / head_dim as f64)))
        .collect())
}

pub fn yarn_inv_frequencies(head_dim: i64, config: &Glm5RuntimeConfig) -> Result<Vec<f64>> {
    let base = rope_inv_frequencies(head_dim, config.rope_theta)?;
    if config.rope_type != "yarn" {
        return Ok(base);
    }

    let correction_dim = |rotations: f64| {
        head_dim as f64
            * (config.rope_original_max_pos as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * config.rope_theta.ln())
    };
    let low = correction_dim(config.rope_beta_fast).floor().max(0.0);
    let high = correction_dim(config.rope_beta_slow)
        .ceil()
        .min((head_dim - 1) as f64);
    let high = if (high - low).abs() < f64::EPSILON {
        high + 0.001
    } else {
        high
    };

    Ok(base
        .into_iter()
        .enumerate()
        .map(|(index, extrapolated)| {
            let ramp = ((index as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let interpolation = extrapolated / config.rope_scaling_factor;
            let extrapolation_weight = 1.0 - ramp;
            interpolation * (1.0 - extrapolation_weight) + extrapolated * extrapolation_weight
        })
        .collect())
}

fn expand_rope_frequencies(values: &Tensor, interleave: bool) -> Tensor {
    if interleave {
        Tensor::stack(&[values, values], -1).flatten(-2, -1)
    } else {
        Tensor::cat(&[values, values], -1)
    }
}

fn rope_cos_sin_from_inv_freq(
    seq_len: usize,
    inv_freq: &[f64],
    attention_factor: f64,
    interleave: bool,
    device: Device,
) -> (Tensor, Tensor) {
    let positions = Tensor::arange(seq_len as i64, (Kind::Float, device));
    let inv_freq = Tensor::from_slice(inv_freq)
        .to_kind(Kind::Float)
        .to_device(device);
    let freqs = positions.outer(&inv_freq);
    let cos = freqs.cos() * attention_factor;
    let sin = freqs.sin() * attention_factor;
    (
        expand_rope_frequencies(&cos, interleave),
        expand_rope_frequencies(&sin, interleave),
    )
}

pub fn rope_cos_sin(seq_len: usize, head_dim: i64, theta: f64, device: Device) -> (Tensor, Tensor) {
    let inv_freq =
        rope_inv_frequencies(head_dim, theta).expect("validated GLM-5 RoPE dimensions and theta");
    rope_cos_sin_from_inv_freq(seq_len, &inv_freq, 1.0, false, device)
}

pub fn rope_cos_sin_for_config(
    seq_len: usize,
    head_dim: i64,
    config: &Glm5RuntimeConfig,
    interleave: bool,
    device: Device,
) -> Result<(Tensor, Tensor)> {
    let inv_freq = yarn_inv_frequencies(head_dim, config)?;
    Ok(rope_cos_sin_from_inv_freq(
        seq_len,
        &inv_freq,
        config.rope_attention_factor,
        interleave,
        device,
    ))
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
    let cos = cos.narrow(0, 0, seq_len).unsqueeze(0).unsqueeze(0);
    let sin = sin.narrow(0, 0, seq_len).unsqueeze(0).unsqueeze(0);
    let x_even = x.slice(-1, 0, None, 2);
    let x_odd = x.slice(-1, 1, None, 2);
    let rotated = Tensor::stack(&[&x_odd.neg(), &x_even], -1).flatten(-2, -1);
    x * cos + rotated * sin
}

pub fn apply_rotary_dispatch(x: &Tensor, cos: &Tensor, sin: &Tensor, interleave: bool) -> Tensor {
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
    // FP8 scales (optional, present when weights are F8_E4M3)
    pub q_a_proj_scale: Option<Tensor>,
    pub q_b_proj_scale: Option<Tensor>,
    pub kv_a_proj_scale: Option<Tensor>,
    pub kv_b_proj_scale: Option<Tensor>,
    pub o_proj_scale: Option<Tensor>,
    pub indexer_weights_proj_scale: Option<Tensor>,
    pub indexer_wk_scale: Option<Tensor>,
    pub indexer_wq_b_scale: Option<Tensor>,
}

impl Clone for Glm5AttentionWeights {
    fn clone(&self) -> Self {
        Self {
            q_a_proj: self.q_a_proj.shallow_clone(),
            q_a_layernorm: self.q_a_layernorm.shallow_clone(),
            q_b_proj: self.q_b_proj.shallow_clone(),
            kv_a_proj_with_mqa: self.kv_a_proj_with_mqa.shallow_clone(),
            kv_a_layernorm: self.kv_a_layernorm.shallow_clone(),
            kv_b_proj: self.kv_b_proj.shallow_clone(),
            o_proj: self.o_proj.shallow_clone(),
            indexer_k_norm_weight: self
                .indexer_k_norm_weight
                .as_ref()
                .map(|t| t.shallow_clone()),
            indexer_k_norm_bias: self.indexer_k_norm_bias.as_ref().map(|t| t.shallow_clone()),
            indexer_weights_proj: self
                .indexer_weights_proj
                .as_ref()
                .map(|t| t.shallow_clone()),
            indexer_wk: self.indexer_wk.as_ref().map(|t| t.shallow_clone()),
            indexer_wq_b: self.indexer_wq_b.as_ref().map(|t| t.shallow_clone()),
            q_a_proj_scale: self.q_a_proj_scale.as_ref().map(|t| t.shallow_clone()),
            q_b_proj_scale: self.q_b_proj_scale.as_ref().map(|t| t.shallow_clone()),
            kv_a_proj_scale: self.kv_a_proj_scale.as_ref().map(|t| t.shallow_clone()),
            kv_b_proj_scale: self.kv_b_proj_scale.as_ref().map(|t| t.shallow_clone()),
            o_proj_scale: self.o_proj_scale.as_ref().map(|t| t.shallow_clone()),
            indexer_weights_proj_scale: self
                .indexer_weights_proj_scale
                .as_ref()
                .map(|t| t.shallow_clone()),
            indexer_wk_scale: self.indexer_wk_scale.as_ref().map(|t| t.shallow_clone()),
            indexer_wq_b_scale: self.indexer_wq_b_scale.as_ref().map(|t| t.shallow_clone()),
        }
    }
}

impl Glm5AttentionWeights {
    pub fn load_with_kind(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        kind: Kind,
    ) -> Result<Self> {
        let p = format!("model.layers.{layer}.self_attn");
        // Keep FP8 weights as-is (forward uses _scaled_mm); only convert non-FP8 to compute kind.
        let q_a_proj = tensor(weights, &format!("{p}.q_a_proj.weight"))?.keep_if_fp8(kind);
        let q_a_layernorm = tensor(weights, &format!("{p}.q_a_layernorm.weight"))?.to_kind(kind);
        let q_b_proj = tensor(weights, &format!("{p}.q_b_proj.weight"))?.keep_if_fp8(kind);
        let kv_a = tensor(weights, &format!("{p}.kv_a_proj_with_mqa.weight"))?.keep_if_fp8(kind);
        let kv_a_ln = tensor(weights, &format!("{p}.kv_a_layernorm.weight"))?.to_kind(kind);
        let kv_b = tensor(weights, &format!("{p}.kv_b_proj.weight"))?.keep_if_fp8(kind);
        let o_proj = tensor(weights, &format!("{p}.o_proj.weight"))?.keep_if_fp8(kind);

        // Indexer weights — may not exist for "shared" layers
        let indexer_k_norm_weight = weights
            .get(&format!("{p}.indexer.k_norm.weight"))
            .map(|t| t.to_kind(kind));
        let indexer_k_norm_bias = weights
            .get(&format!("{p}.indexer.k_norm.bias"))
            .map(|t| t.to_kind(kind));
        let indexer_weights_proj = weights
            .get(&format!("{p}.indexer.weights_proj.weight"))
            .map(|t| t.keep_if_fp8(kind));
        let indexer_wk = weights
            .get(&format!("{p}.indexer.wk.weight"))
            .map(|t| t.keep_if_fp8(kind));
        let indexer_wq_b = weights
            .get(&format!("{p}.indexer.wq_b.weight"))
            .map(|t| t.keep_if_fp8(kind));

        // FP8 scales
        let q_a_proj_scale = weights
            .get(&format!("{p}.q_a_proj.weight_scale_inv"))
            .map(|t| t.shallow_clone());
        let q_b_proj_scale = weights
            .get(&format!("{p}.q_b_proj.weight_scale_inv"))
            .map(|t| t.shallow_clone());
        let kv_a_proj_scale = weights
            .get(&format!("{p}.kv_a_proj_with_mqa.weight_scale_inv"))
            .map(|t| t.shallow_clone());
        let kv_b_proj_scale = weights
            .get(&format!("{p}.kv_b_proj.weight_scale_inv"))
            .map(|t| t.shallow_clone());
        let o_proj_scale = weights
            .get(&format!("{p}.o_proj.weight_scale_inv"))
            .map(|t| t.shallow_clone());
        let indexer_weights_proj_scale = weights
            .get(&format!("{p}.indexer.weights_proj.weight_scale_inv"))
            .map(|t| t.shallow_clone());
        let indexer_wk_scale = weights
            .get(&format!("{p}.indexer.wk.weight_scale_inv"))
            .map(|t| t.shallow_clone());
        let indexer_wq_b_scale = weights
            .get(&format!("{p}.indexer.wq_b.weight_scale_inv"))
            .map(|t| t.shallow_clone());

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
            q_a_proj_scale,
            q_b_proj_scale,
            kv_a_proj_scale,
            kv_b_proj_scale,
            o_proj_scale,
            indexer_weights_proj_scale,
            indexer_wk_scale,
            indexer_wq_b_scale,
        })
    }

    pub fn load_raw(weights: &BTreeMap<String, Tensor>, layer: usize) -> Result<Self> {
        Self::load_with_kind(weights, layer, Kind::Float)
    }
}

// ── DSA Indexer State (for IndexShare) ───────────────────────────

pub struct IndexShareState {
    /// Top-k indices: [batch, num_heads, seq, topk] int64 — which KV positions to attend to.
    /// Compact representation: O(S × topk) instead of O(S²) for sparse_mask.
    pub topk_indices: Tensor,
    /// Per-key bias: [batch, idx_n_heads, seq] — NOT expanded to S×S (saves O(S²) memory).
    /// Expanded lazily during attention bias construction.
    pub idx_bias_keys: Tensor,
    /// Which layer produced this state
    pub source_layer: usize,
}

impl Clone for IndexShareState {
    fn clone(&self) -> Self {
        Self {
            topk_indices: self.topk_indices.shallow_clone(),
            idx_bias_keys: self.idx_bias_keys.shallow_clone(),
            source_layer: self.source_layer,
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
    // Ensure input is in a consistent dtype for all matmul operations
    let compute_kind = input.kind();
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
    // Use glm5_safe_linear for FP8 dispatch when scale is available
    let q_a = glm5_safe_linear(input, &attn.q_a_proj, attn.q_a_proj_scale.as_ref());
    let q_a_normed = rms_norm(
        &q_a,
        &attn.q_a_layernorm.to_kind(compute_kind),
        config.rms_norm_eps,
    );
    let q_b = glm5_safe_linear(&q_a_normed, &attn.q_b_proj, attn.q_b_proj_scale.as_ref());
    let q = q_b
        .reshape([batch, seq, num_heads, qk_nope + qk_rope])
        .transpose(1, 2);
    let q_nope = q.narrow(-1, 0, qk_nope);
    let q_rope = q.narrow(-1, qk_nope, qk_rope);

    let kv_a = glm5_safe_linear(
        input,
        &attn.kv_a_proj_with_mqa,
        attn.kv_a_proj_scale.as_ref(),
    );
    // Split first: kv_lora part gets RMSNorm, RoPE part does not
    let kv_lora_raw = kv_a.narrow(-1, 0, kv_lora);
    let k_rope = kv_a.narrow(-1, kv_lora, qk_rope);
    let kv_lora_part = rms_norm(
        &kv_lora_raw,
        &attn.kv_a_layernorm.to_kind(compute_kind),
        config.rms_norm_eps,
    );
    let kv_b = glm5_safe_linear(
        &kv_lora_part,
        &attn.kv_b_proj,
        attn.kv_b_proj_scale.as_ref(),
    );
    let kv_b = kv_b.reshape([batch, seq, num_heads, qk_nope + v_head]);
    let k_nope = kv_b.narrow(-1, 0, qk_nope).transpose(1, 2);
    let v = kv_b.narrow(-1, qk_nope, v_head).transpose(1, 2);

    let k_rope_expanded = k_rope
        .unsqueeze(2)
        .transpose(1, 2)
        .expand([batch, num_heads, seq, qk_rope], false);
    let (cos, sin) = rope_cos_sin_for_config(
        seq as usize,
        qk_rope,
        config,
        config.rope_interleave,
        input.device(),
    )
    .expect("validated GLM-5 RoPE configuration");
    let cos = cos.to_kind(input.kind());
    let sin = sin.to_kind(input.kind());
    let q_rope_rotated = apply_rotary_dispatch(&q_rope, &cos, &sin, config.rope_interleave);
    let k_rope_rotated =
        apply_rotary_dispatch(&k_rope_expanded, &cos, &sin, config.rope_interleave);

    let q_full = Tensor::cat(&[&q_nope, &q_rope_rotated], -1);
    let k_full = Tensor::cat(&[&k_nope, &k_rope_rotated], -1);

    let attn_scale = 1.0 / ((qk_nope + qk_rope) as f64).sqrt();
    // Note: scores matrix is NOT materialized — SDPA handles attention internally

    // ── DSA Indexer ──
    let should_compute_topk = config.should_recompute_indexer(layer);

    if let (Some(wq_b), Some(wk), Some(k_norm_w), Some(k_norm_b), Some(weights_proj)) = (
        &indexer_weights.indexer_wq_b,
        &indexer_weights.indexer_wk,
        &indexer_weights.indexer_k_norm_weight,
        &indexer_weights.indexer_k_norm_bias,
        &indexer_weights.indexer_weights_proj,
    ) {
        if should_compute_topk {
            // 1. Indexer Q: from q_a → wq_b → [batch, seq, idx_n_heads * idx_head_dim]
            let idx_q = glm5_safe_linear(&q_a, wq_b, indexer_weights.indexer_wq_b_scale.as_ref());
            // Reshape to [batch, idx_n_heads, seq, idx_head_dim]
            let idx_q = idx_q
                .reshape([batch, seq, idx_n_heads, idx_head_dim])
                .transpose(1, 2);

            // 2. Indexer K: from HIDDEN (not kv_lora) → wk → k_norm → [batch, seq, idx_head_dim]
            //    wk: [idx_head_dim, hidden_size] — single-head key, NO idx_n_heads dimension
            let idx_k_raw = glm5_safe_linear(input, wk, indexer_weights.indexer_wk_scale.as_ref());
            let idx_k = rms_norm_with_bias(
                &idx_k_raw,
                &k_norm_w.to_kind(compute_kind),
                &k_norm_b.to_kind(compute_kind),
                config.rms_norm_eps,
            );
            // idx_k: [batch, seq, idx_head_dim] — broadcast across idx_n_heads heads
            let idx_k_expanded = idx_k
                .unsqueeze(1) // [b, 1, seq, dim]
                .expand([batch, idx_n_heads, seq, idx_head_dim], false);

            // 3. Apply RoPE only to the configured positional subspace. GLM-5
            // stores it in the trailing dimensions for interleaved checkpoints;
            // DeepSeek-V3.2's contiguous layout stores it first.
            let (cos_i, sin_i) = rope_cos_sin_for_config(
                seq as usize,
                qk_rope,
                config,
                config.indexer_rope_interleave,
                input.device(),
            )
            .expect("validated GLM-5 indexer RoPE configuration");
            let cos_i = cos_i.to_kind(input.kind());
            let sin_i = sin_i.to_kind(input.kind());
            let rotate_indexer = |value: &Tensor| {
                let nope_dim = idx_head_dim - qk_rope;
                if config.indexer_rope_interleave {
                    let nope = value.narrow(-1, 0, nope_dim);
                    let rope = value.narrow(-1, nope_dim, qk_rope);
                    Tensor::cat(
                        &[&nope, &apply_rotary_interleave(&rope, &cos_i, &sin_i)],
                        -1,
                    )
                } else {
                    let rope = value.narrow(-1, 0, qk_rope);
                    let nope = value.narrow(-1, qk_rope, nope_dim);
                    Tensor::cat(&[&apply_rotary(&rope, &cos_i, &sin_i), &nope], -1)
                }
            };
            let idx_q_rotated = rotate_indexer(&idx_q);
            let idx_k_rotated = rotate_indexer(&idx_k_expanded);

            let actual_topk = idx_topk.min(seq as i64);
            let score_chunk = 512_i64;
            // weights_proj is a per-query head weight for the indexer score:
            // sum_h(relu(q_h dot k) * weight_h). It is not an SDPA bias.
            let head_weights = glm5_safe_linear(
                input,
                weights_proj,
                indexer_weights.indexer_weights_proj_scale.as_ref(),
            )
            .reshape([batch, seq, idx_n_heads])
            .transpose(1, 2)
            .to_kind(Kind::Float)
                * ((idx_n_heads * idx_head_dim) as f64).sqrt().recip();

            let causal_mask = |k_start: i64, k_len: i64| {
                let q_pos = Tensor::arange(seq, (Kind::Int64, input.device()));
                let k_pos = Tensor::arange(k_len, (Kind::Int64, input.device())) + k_start;
                k_pos
                    .unsqueeze(0)
                    .gt_tensor(&q_pos.unsqueeze(1))
                    .unsqueeze(0)
                    .unsqueeze(0)
            };

            let score_block = |keys: &Tensor, k_start: i64, k_len: i64| {
                let per_head = idx_q_rotated
                    .matmul(&keys.transpose(-2, -1))
                    .relu()
                    .to_kind(Kind::Float);
                let scores = (per_head * head_weights.unsqueeze(-1)).sum_dim_intlist(
                    [1].as_slice(),
                    true,
                    Kind::Float,
                );
                scores.masked_fill(&causal_mask(k_start, k_len), f64::NEG_INFINITY)
            };

            let topk_indices = if seq <= score_chunk {
                let scores = score_block(&idx_k_rotated, 0, seq);
                let (_, indices) = scores.topk(actual_topk, -1, true, true);
                indices.expand([batch, num_heads, seq, actual_topk], false)
            } else {
                let mut best_scores: Option<Tensor> = None;
                let mut best_indices: Option<Tensor> = None;

                for k_start in (0..seq).step_by(score_chunk as usize) {
                    let k_end = (k_start + score_chunk).min(seq as i64);
                    let k_len = k_end - k_start;
                    let idx_k_chunk = idx_k_rotated.narrow(-2, k_start, k_len);
                    let scores_chunk = score_block(&idx_k_chunk, k_start, k_len);
                    let local_topk = actual_topk.min(k_len);
                    let (local_scores, local_indices) =
                        scores_chunk.topk(local_topk, -1, true, true);
                    let local_indices = local_indices + k_start;

                    match (&best_scores, &best_indices) {
                        (Some(bs), Some(bi)) => {
                            let merged = Tensor::cat(&[bs, &local_scores], -1);
                            let merged_idx = Tensor::cat(&[bi, &local_indices], -1);
                            let (s, pos) = merged.topk(actual_topk, -1, true, true);
                            best_scores = Some(s);
                            best_indices = Some(merged_idx.gather(-1, &pos, false));
                        }
                        _ => {
                            best_scores = Some(local_scores);
                            best_indices = Some(local_indices);
                        }
                    }
                }
                best_indices
                    .expect("at least one indexer key block")
                    .expand([batch, num_heads, seq, actual_topk], false)
            };

            // Keep the legacy state field shape for TP/CP ABI compatibility; it
            // is intentionally zero and is not used as an attention bias.
            let idx_bias_keys = Tensor::zeros([batch, 1, seq], (input.kind(), input.device()));
            *index_share_state = Some(IndexShareState {
                topk_indices,
                idx_bias_keys,
                source_layer: layer,
            });
        }
    } else {
        // No indexer weights → full causal attention fallback
        *index_share_state = None;
    }

    // ── Flash Attention via SDPA ──
    let context = if let Some(state) = index_share_state {
        // DSA sparse attention with chunked bias construction.
        // Two optimizations vs. original:
        // 1. drop() early release — frees O(S²) intermediates immediately after use,
        //    reducing peak from 5 simultaneous tensors to 2.
        // 2. Query-dim chunking — for large seq, builds [B,H,C,S] bias per chunk
        //    instead of [B,H,S,S]. Peak: O(C×S) instead of O(S²).
        //
        // Combined effect at S=8192, C=512:
        //   Old: 5 × 64 × 8192² × 2B = 40 GB  →  New: 2 × 64 × 512 × 8192 × 2B = 1 GB

        let actual_topk = state.topk_indices.size()[state.topk_indices.size().len() - 1];

        // Chunk size: 512 query positions per chunk.
        // At S=8192: 16 chunks × [B,H,512,S] = 16 × 1 GB, but only 2 simultaneous.
        let attn_chunk: i64 = if seq > 2048 { 512 } else { seq as i64 };

        if attn_chunk >= seq as i64 {
            // ── Small seq (≤2048): single pass with early drop ──
            let sparse_mask = {
                let mut m = Tensor::zeros(
                    [batch as i64, num_heads, seq as i64, seq as i64],
                    (input.kind(), input.device()),
                );
                let ones = Tensor::ones(
                    [batch as i64, num_heads, seq as i64, actual_topk],
                    (input.kind(), input.device()),
                );
                let _ = m.scatter_(-1, &state.topk_indices, &ones);
                m
            };
            let causal_f = {
                let cm =
                    Tensor::ones([seq as i64, seq as i64], (Kind::Bool, input.device())).triu(1);
                cm.unsqueeze(0)
                    .unsqueeze(0)
                    .expand([batch as i64, num_heads, seq as i64, seq as i64], false)
                    .to_kind(input.kind())
            };

            // Combine sparse + causal, then free intermediates (5→3→2)
            let combined = &sparse_mask * (1.0 - &causal_f);
            drop(sparse_mask);
            drop(causal_f);

            let bias =
                Tensor::zeros_like(&combined).masked_fill(&combined.eq(0), f64::NEG_INFINITY);
            drop(combined);

            Tensor::scaled_dot_product_attention(
                &q_full,
                &k_full,
                &v,
                Some(&bias),
                0.0,
                false,
                Some(attn_scale),
                false,
            )
        } else {
            // ── Large seq (>2048): chunked attention ──
            // Process query dim in chunks of C=512. Each chunk builds [B,H,C,S]
            // bias instead of [B,H,S,S]. Peak: 2×B×H×C×S×2B.
            let n_chunks = (seq as i64 + attn_chunk - 1) / attn_chunk;
            let mut outputs: Vec<Tensor> = Vec::with_capacity(n_chunks as usize);

            for q_start in (0..seq as i64).step_by(attn_chunk as usize) {
                let q_end = (q_start + attn_chunk).min(seq as i64);
                let q_len = q_end - q_start;
                let q_chunk = q_full.narrow(2, q_start, q_len);

                // 1. Sparse mask for this chunk: [B, H, q_len, S]
                let sparse_mask = {
                    let chunk_topk = state.topk_indices.narrow(2, q_start, q_len);
                    let mut m = Tensor::zeros(
                        [batch as i64, num_heads, q_len, seq as i64],
                        (input.kind(), input.device()),
                    );
                    let ones = Tensor::ones(
                        [batch as i64, num_heads, q_len, actual_topk],
                        (input.kind(), input.device()),
                    );
                    let _ = m.scatter_(-1, &chunk_topk, &ones);
                    m
                };

                // 2. Causal mask: key j is masked if j > q_start + i (query pos)
                let causal_f = {
                    let q_pos = (Tensor::arange(q_len, (Kind::Int64, input.device())) + q_start)
                        .to_kind(input.kind());
                    let k_pos = Tensor::arange(seq as i64, (Kind::Int64, input.device()))
                        .to_kind(input.kind());
                    // cm[i, j] = true if j > q_start + i (causal: mask future keys)
                    let diff = k_pos.unsqueeze(0) - q_pos.unsqueeze(1); // [q_len, S]
                    let cm = diff.gt(0.0); // bool tensor
                    cm.unsqueeze(0)
                        .unsqueeze(0)
                        .expand([batch as i64, num_heads, q_len, seq as i64], false)
                        .to_kind(input.kind())
                };

                // 3. Combine, then drop intermediates (3→2→1)
                let combined = &sparse_mask * (1.0 - &causal_f);
                drop(sparse_mask);
                drop(causal_f);

                // 4. Bias = idx_bias where valid, -inf where masked
                let bias =
                    Tensor::zeros_like(&combined).masked_fill(&combined.eq(0), f64::NEG_INFINITY);
                drop(combined);

                // 5. SDPA for this chunk: Q[C,d] × K[S,d] → out[C,d]
                let chunk_out = Tensor::scaled_dot_product_attention(
                    &q_chunk,
                    &k_full,
                    &v,
                    Some(&bias),
                    0.0,
                    false,
                    Some(attn_scale),
                    false,
                );
                drop(bias);
                outputs.push(chunk_out);
            }

            let refs: Vec<&Tensor> = outputs.iter().collect();
            Tensor::cat(&refs, 2)
        }
    } else {
        // Full causal: SDPA's built-in causal mask — fastest path
        Tensor::scaled_dot_product_attention::<&Tensor>(
            &q_full,
            &k_full,
            &v,
            None,
            0.0,  // dropout
            true, // is_causal
            Some(attn_scale),
            false, // enable_gqa
        )
    };

    let context = context
        .transpose(1, 2)
        .reshape([batch, seq, num_heads * v_head]);

    glm5_safe_linear(&context, &attn.o_proj, attn.o_proj_scale.as_ref())
}

// ── MLP ─────────────────────────────────────────────────────────

pub fn glm5_mlp(input: &Tensor, gate: &Tensor, up: &Tensor, down: &Tensor) -> Tensor {
    let k = input.kind();
    let gate_out = input.linear::<&Tensor>(&gate.to_kind(k), None);
    let up_out = input.linear::<&Tensor>(&up.to_kind(k), None);
    let activated = gate_out.silu() * up_out;
    activated.linear::<&Tensor>(&down.to_kind(k), None)
}

/// Safe FP8 linear dispatch: uses V4's fp8_linear (_scaled_mm) when scale is available
/// and dimensions are 128-aligned. For non-128-aligned FP8 weights (e.g. kv_a_proj [576, 6144]),
/// uses byte-level C++ dequant (dequant_fp8_weight) to bypass PyTorch's view bug.
/// Fails loudly when an FP8 weight has no usable scale or the FP8 kernel
/// cannot dequantize it. Silently treating FP8 bytes as bf16 produces
/// numerically invalid training results.
pub fn glm5_safe_linear(input: &Tensor, weight: &Tensor, scale: Option<&Tensor>) -> Tensor {
    if let Some(s) = scale {
        let n = weight.size()[0];
        let k = weight.size()[1];

        // V4 path: fp8_linear (_scaled_mm) for 128-aligned weights
        if n % 128 == 0
            && k % 128 == 0
            && matches!(input.kind(), Kind::BFloat16 | Kind::Float)
            && matches!(input.device(), tch::Device::Cuda(_))
        {
            match rustrain_deepseek_v4::fp8_kernel::fp8_linear(input, weight, s) {
                Ok(out) => return out,
                Err(e) => {
                    tracing::debug!("fp8_linear failed ({e:?}), trying explicit dequant");
                }
            }
        }

        // Fallback for non-128-aligned FP8 weights: byte-level dequant + scale + linear
        match rustrain_deepseek_v4::fp8_kernel::dequant_fp8_weight(weight, s) {
            Ok(w_bf16) => {
                let compute_kind = input.kind();
                return input.linear::<&Tensor>(&w_bf16.to_kind(compute_kind), None);
            }
            Err(e) => {
                panic!("FP8 dequantization failed for a GLM-5 weight: {e:?}");
            }
        }
    } else {
        if weight.kind() == Kind::Float8e4m3fn {
            panic!("FP8 GLM-5 weight is missing its block scale tensor");
        }
        // No scale: standard bf16 linear
        let k = input.kind();
        input.linear::<&Tensor>(&weight.to_kind(k), None)
    }
}

/// FP8-aware MLP: uses V4's fp8_linear for GEMM when FP8 weights + scales are provided.
/// Falls back to standard bf16 linear otherwise.
pub fn glm5_mlp_fp8(
    input: &Tensor,
    gate: &Tensor,
    up: &Tensor,
    down: &Tensor,
    gate_scale: Option<&Tensor>,
    up_scale: Option<&Tensor>,
    down_scale: Option<&Tensor>,
) -> Tensor {
    let gate_out = glm5_safe_linear(input, gate, gate_scale);
    let up_out = glm5_safe_linear(input, up, up_scale);
    let activated = gate_out.silu() * up_out;
    glm5_safe_linear(&activated, down, down_scale)
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
    glm5_moe_mlp_with_router(
        input,
        gate,
        None,
        shared_gate,
        shared_up,
        shared_down,
        experts,
        num_experts_per_tok,
        scoring_func,
        "groupwise",
        n_group,
        topk_group,
        true,
        routed_scaling_factor,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn glm5_router_topk(
    router_logits: &Tensor,
    correction_bias: Option<&Tensor>,
    num_experts_per_tok: usize,
    scoring_func: &str,
    topk_method: &str,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
) -> (Tensor, Tensor) {
    let n_experts = router_logits.size()[router_logits.size().len() - 1];
    let scores = match scoring_func {
        "sigmoid" => router_logits.sigmoid(),
        "softmax" => router_logits.softmax(-1, Kind::Float),
        other => panic!("unsupported GLM-5 scoring_func {other:?}"),
    };
    let selection_scores = match topk_method {
        "noaux_tc" => {
            scores.shallow_clone()
                + correction_bias
                    .expect("topk_method=noaux_tc requires e_score_correction_bias")
                    .to_kind(Kind::Float)
        }
        "groupwise" => scores.shallow_clone(),
        other => panic!("unsupported GLM-5 topk_method {other:?}"),
    };
    let epg = n_experts / n_group as i64;
    let grouped = selection_scores.reshape([-1, n_group as i64, epg]);
    let group_scores = if topk_method == "noaux_tc" {
        grouped
            .topk(2_i64.min(epg), -1, true, true)
            .0
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
    } else {
        grouped.max_dim(-1, false).0
    };
    let (_, group_idx) = group_scores.topk(topk_group as i64, -1, true, true);
    let mut group_mask = Tensor::zeros_like(&group_scores);
    let ones = Tensor::ones(group_idx.size(), (Kind::Float, group_idx.device()));
    let _ = group_mask.scatter_(-1, &group_idx, &ones);
    let expert_mask = group_mask
        .unsqueeze(-1)
        .expand([-1, -1, epg], false)
        .reshape([-1, n_experts])
        .eq(0);
    let masked_scores = selection_scores
        .reshape([-1, n_experts])
        .masked_fill(&expert_mask, f64::NEG_INFINITY);
    let (_, topk_indices) = masked_scores.topk(num_experts_per_tok as i64, -1, true, true);
    let mut topk_weights = scores
        .reshape([-1, n_experts])
        .gather(-1, &topk_indices, false);
    if norm_topk_prob {
        let denom = topk_weights.sum_dim_intlist([-1].as_slice(), true, Kind::Float);
        topk_weights = topk_weights / denom.clamp_min(1e-20);
    }
    (topk_weights * routed_scaling_factor, topk_indices)
}

#[allow(clippy::too_many_arguments)]
pub fn glm5_moe_mlp_with_router(
    input: &Tensor,
    gate: &Tensor,
    correction_bias: Option<&Tensor>,
    shared_gate: &Tensor,
    shared_up: &Tensor,
    shared_down: &Tensor,
    experts: &[(Tensor, Tensor, Tensor)],
    num_experts_per_tok: usize,
    scoring_func: &str,
    topk_method: &str,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
) -> Tensor {
    let shared_output = glm5_mlp(input, shared_gate, shared_up, shared_down);
    let router_logits = input
        .linear::<&Tensor>(&gate.to_kind(input.kind()), None)
        .to_kind(Kind::Float);
    let n_experts = experts.len() as i64;
    let (topk_weights, topk_indices) = glm5_router_topk(
        &router_logits,
        correction_bias,
        num_experts_per_tok,
        scoring_func,
        topk_method,
        n_group,
        topk_group,
        norm_topk_prob,
        routed_scaling_factor,
    );

    // Flatten topk to [batch*seq, num_experts_per_tok] for per-token expert dispatch
    let topk_weights = topk_weights.reshape([-1, num_experts_per_tok as i64]);
    let topk_indices = topk_indices.reshape([-1, num_experts_per_tok as i64]);

    let batch = input.size()[0];
    let seq = input.size()[1];
    let flat_input = input.reshape([-1, input.size()[2]]);

    let mut output = Tensor::zeros(flat_input.size(), (input.kind(), input.device()));

    for e in 0..n_experts as usize {
        let slot_mask = topk_indices.eq(e as i64).to_kind(Kind::Float);
        let expert_weight = (&topk_weights * &slot_mask)
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
            .to_kind(input.kind());
        let count = slot_mask.sum(Kind::Float).double_value(&[]) as i64;
        if count == 0 {
            continue;
        }
        let (gate_w, up_w, down_w) = &experts[e];
        let expert_out = glm5_mlp(&flat_input, gate_w, up_w, down_w);
        let contribution = expert_out * expert_weight.unsqueeze(-1);
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
            post_attention_norm: tensor(weights, &format!("{p}.post_attention_layernorm.weight"))?
                .shallow_clone(),
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
    pub gate_correction_bias: Option<Tensor>,
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
            post_attention_norm: tensor(weights, &format!("{p}.post_attention_layernorm.weight"))?
                .shallow_clone(),
            gate: tensor(weights, &format!("{p}.mlp.gate.weight"))?.shallow_clone(),
            gate_correction_bias: weights
                .get(&format!("{p}.mlp.gate.e_score_correction_bias"))
                .map(|tensor| tensor.shallow_clone()),
            shared_gate_proj: tensor(weights, &format!("{shared_prefix}.gate_proj.weight"))?
                .shallow_clone(),
            shared_up_proj: tensor(weights, &format!("{shared_prefix}.up_proj.weight"))?
                .shallow_clone(),
            shared_down_proj: tensor(weights, &format!("{shared_prefix}.down_proj.weight"))?
                .shallow_clone(),
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
    indexer_weights_map: &BTreeMap<usize, Glm5AttentionWeights>,
    layer: usize,
) -> Tensor {
    let hidden = rms_norm(input, &weights.input_norm, config.rms_norm_eps);
    let source = config.indexer_source_layer(layer);
    let indexer_weights = indexer_weights_map.get(&source).unwrap_or(&weights.attn);
    let attn = glm5_dsa_attention(
        &hidden,
        &weights.attn,
        indexer_weights,
        config,
        index_share_state,
        layer,
    );
    let residual = input + &attn;
    let mlp_input = rms_norm(&residual, &weights.post_attention_norm, config.rms_norm_eps);
    let mlp = glm5_mlp(
        &mlp_input,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
    );
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
    let attn = glm5_dsa_attention(
        &hidden,
        &weights.attn,
        indexer_weights,
        config,
        index_share_state,
        layer,
    );
    let residual = input + &attn;
    let mlp_input = rms_norm(&residual, &weights.post_attention_norm, config.rms_norm_eps);
    let mlp = glm5_moe_mlp_with_router(
        &mlp_input,
        &weights.gate,
        weights.gate_correction_bias.as_ref(),
        &weights.shared_gate_proj,
        &weights.shared_up_proj,
        &weights.shared_down_proj,
        &weights.experts,
        config.num_experts_per_tok,
        &config.scoring_func,
        &config.topk_method,
        config.n_group,
        config.topk_group,
        config.norm_topk_prob,
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
            hidden = glm5_dense_layer(
                &hidden,
                &lw,
                config,
                &mut index_share_state,
                &indexer_weights_map,
                layer,
            );
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

// ── MTP (Multi-Token Prediction) ─────────────────────────────────

pub const GLM5_MTP_LOSS_SCALING_FACTOR_DEFAULT: f64 =
    rustrain_core::runtime::DEFAULT_MTP_LOSS_SCALING_FACTOR;

/// Validate the native MTP topology before loading weights or initializing
/// NCCL. This implementation supports the checkpoint's one native layer with
/// TP-only and EP-only, while retaining an explicit CP and combined TP+EP gate.
pub fn validate_glm5_mtp_contract(num_layers: usize, cp_size: usize) -> Result<()> {
    if num_layers > 1 {
        bail!(
            "GLM-5 native MTP supports exactly one layer in this implementation; configured {num_layers}"
        );
    }
    if num_layers > 0 && cp_size > 1 {
        bail!(
            "GLM-5 native MTP with context_parallel_size > 1 is unsupported: CP activation ring is not Megatron-autograd-aware"
        );
    }
    Ok(())
}

/// Validate the complete distributed topology for the native MTP path. TP-only
/// and EP-only are supported. Combined TP+EP remains gated until the session
/// builds Megatron's independent expert-EP/expert-DP groups and sequence-
/// parallel token movement.
pub fn validate_glm5_mtp_distributed_contract(
    num_layers: usize,
    tp_size: usize,
    cp_size: usize,
    ep_size: usize,
) -> Result<()> {
    validate_glm5_mtp_contract(num_layers, cp_size)?;
    if tp_size == 0 || ep_size == 0 {
        bail!("GLM-5 native MTP TP and EP sizes must be positive");
    }
    if num_layers > 0 && tp_size > 1 && ep_size > 1 {
        bail!(
            "GLM-5 native MTP with combined TP+EP is unsupported until the session builds Megatron independent expert-EP/expert-DP groups and sequence-parallel token scatter/gather (tp_size={tp_size}, ep_size={ep_size})"
        );
    }
    Ok(())
}

/// Megatron carries MTP loss numerators and token counts across microbatches
/// before applying global per-token normalization. The GLM5 sessions now
/// preserve those tensors through the optimizer step; this helper keeps the
/// pre-load validation for malformed zero-sized accumulation settings.
pub fn validate_glm5_mtp_accumulation_contract(
    num_layers: usize,
    gradient_accumulation_steps: usize,
) -> Result<()> {
    if num_layers > 0 && gradient_accumulation_steps == 0 {
        bail!(
            "GLM-5 native MTP requires gradient_accumulation_steps > 0; got {gradient_accumulation_steps}"
        );
    }
    Ok(())
}

/// Fusion and output-normalization weights that surround GLM-5.2's native
/// MTP decoder layer. The decoder itself is stored as the extra layer at
/// `model.layers.{num_hidden_layers + mtp_layer}` and uses the normal
/// attention/MoE weight containers.
pub struct Glm5MtpProjectionWeights {
    pub enorm: Tensor,
    pub hnorm: Tensor,
    pub eh_proj: Tensor,
    pub eh_proj_scale: Option<Tensor>,
    pub shared_head_norm: Tensor,
}

impl Glm5MtpProjectionWeights {
    pub fn load_with_kind(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        kind: Kind,
    ) -> Result<Self> {
        let p = format!("model.layers.{layer}");
        Ok(Self {
            enorm: tensor(weights, &format!("{p}.enorm.weight"))?.to_kind(kind),
            hnorm: tensor(weights, &format!("{p}.hnorm.weight"))?.to_kind(kind),
            eh_proj: tensor(weights, &format!("{p}.eh_proj.weight"))?.keep_if_fp8(kind),
            eh_proj_scale: weights
                .get(&format!("{p}.eh_proj.weight_scale_inv"))
                .map(Tensor::shallow_clone),
            shared_head_norm: tensor(weights, &format!("{p}.shared_head.norm.weight"))?
                .to_kind(kind),
        })
    }

    /// Load the Megatron column-parallel MTP fusion projection. Norm weights
    /// remain replicated, while `eh_proj` is sharded from `[H, 2H]` to
    /// `[H / TP, 2H]` over its output rows.
    pub fn load_tp_sharded(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        kind: Kind,
        hidden_size: i64,
        tp_rank: usize,
        tp_size: usize,
    ) -> Result<Self> {
        let full = Self::load_with_kind(weights, layer, kind)?;
        let p = format!("model.layers.{layer}");
        for (name, norm) in [
            ("enorm", &full.enorm),
            ("hnorm", &full.hnorm),
            ("shared_head.norm", &full.shared_head_norm),
        ] {
            if norm.size() != [hidden_size] {
                bail!(
                    "{p}.{name}.weight must have shape [{hidden_size}], got {:?}",
                    norm.size()
                );
            }
        }
        if full.eh_proj.size() != [hidden_size, hidden_size * 2] {
            bail!(
                "{p}.eh_proj.weight must have shape [{hidden_size}, {}], got {:?}",
                hidden_size * 2,
                full.eh_proj.size()
            );
        }
        let eh_proj = crate::tp_cp::shard_column_parallel_linear(
            &full.eh_proj,
            full.eh_proj_scale.as_ref(),
            tp_rank,
            tp_size,
            &format!("{p}.eh_proj.weight"),
        )?;
        Ok(Self {
            enorm: full.enorm,
            hnorm: full.hnorm,
            eh_proj: eh_proj.weight,
            eh_proj_scale: eh_proj.weight_scale,
            shared_head_norm: full.shared_head_norm,
        })
    }

    pub fn weight_names(layer: usize) -> Vec<String> {
        let p = format!("model.layers.{layer}");
        vec![
            format!("{p}.enorm.weight"),
            format!("{p}.hnorm.weight"),
            format!("{p}.eh_proj.weight"),
            format!("{p}.eh_proj.weight_scale_inv"),
            format!("{p}.shared_head.norm.weight"),
        ]
    }
}

pub fn glm5_mtp_layer_index(config: &Glm5RuntimeConfig, mtp_layer: usize) -> Result<usize> {
    if mtp_layer >= config.num_nextn_predict_layers {
        bail!(
            "MTP layer {mtp_layer} is outside configured range 0..{}",
            config.num_nextn_predict_layers
        );
    }
    config
        .num_hidden_layers
        .checked_add(mtp_layer)
        .ok_or_else(|| anyhow!("MTP decoder layer index overflows usize"))
}

/// Return the extra decoder-layer indices used by native GLM-5.2 MTP.
///
/// GLM-5 stores MTP layers directly after the trunk decoder layers, so a
/// checkpoint configured with `num_nextn_predict_layers = N` owns layers
/// `model.layers.{num_hidden_layers}` through
/// `model.layers.{num_hidden_layers + N - 1}`.
pub fn glm5_mtp_layer_indices(config: &Glm5RuntimeConfig) -> Result<Vec<usize>> {
    (0..config.num_nextn_predict_layers)
        .map(|i| glm5_mtp_layer_index(config, i))
        .collect()
}

pub fn glm5_mtp_prediction_len(seq_len: i64) -> Result<i64> {
    if seq_len < 3 {
        bail!("GLM-5 MTP requires at least three tokens, got {seq_len}");
    }
    Ok(seq_len - 2)
}

/// Megatron's GPT dataset reads one extra raw token, then splits it into
/// `sequence_length` inputs and the same number of next-token labels.
pub fn glm5_megatron_raw_seq_len(sequence_len: i64) -> Result<i64> {
    if sequence_len <= 0 {
        bail!("Megatron sequence length must be positive, got {sequence_len}");
    }
    sequence_len
        .checked_add(1)
        .ok_or_else(|| anyhow!("Megatron raw sequence length overflows i64"))
}

/// Number of valid teacher-forced positions after all configured MTP layers.
pub fn glm5_mtp_prediction_len_for_layers(seq_len: i64, num_layers: usize) -> Result<i64> {
    let num_layers_i64 =
        i64::try_from(num_layers).map_err(|_| anyhow!("MTP layer count does not fit in i64"))?;
    let required = num_layers_i64
        .checked_add(2)
        .ok_or_else(|| anyhow!("MTP required sequence length overflows i64"))?;
    if seq_len < required {
        bail!("GLM-5 {num_layers}-layer MTP requires at least {required} tokens, got {seq_len}");
    }
    Ok(seq_len - num_layers_i64 - 1)
}

pub fn has_mtp_weights(weights: &BTreeMap<String, Tensor>, config: &Glm5RuntimeConfig) -> bool {
    config.num_nextn_predict_layers > 0
        && glm5_mtp_layer_indices(config).map_or(false, |layers| {
            layers
                .iter()
                .all(|layer| weights.contains_key(&format!("model.layers.{layer}.eh_proj.weight")))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_rope_uses_inverse_power_frequencies() {
        let frequencies = rope_inv_frequencies(64, 8_000_000.0).unwrap();
        assert_eq!(frequencies[0], 1.0);
        let expected = 8_000_000_f64.powf(-62.0 / 64.0);
        assert!((frequencies[31] - expected).abs() < 1e-15);
        assert!(frequencies.windows(2).all(|pair| pair[1] < pair[0]));
    }

    #[test]
    fn interleaved_rotary_preserves_adjacent_pairs() {
        let x = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape([1, 1, 1, 4]);
        let cos = Tensor::zeros([1, 4], (Kind::Float, Device::Cpu));
        let sin = Tensor::ones([1, 4], (Kind::Float, Device::Cpu));
        let rotated = apply_rotary_interleave(&x, &cos, &sin).reshape([-1]);
        let values: Vec<f32> = Vec::<f32>::try_from(&rotated).unwrap();
        assert_eq!(values, vec![-2.0, 1.0, -4.0, 3.0]);
    }

    #[test]
    fn native_mtp_uses_extra_decoder_layer_contract() {
        let names = Glm5MtpProjectionWeights::weight_names(78);
        assert!(names.contains(&"model.layers.78.enorm.weight".to_string()));
        assert!(names.contains(&"model.layers.78.hnorm.weight".to_string()));
        assert!(names.contains(&"model.layers.78.eh_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.78.shared_head.norm.weight".to_string()));
        assert!(names.iter().all(|name| !name.starts_with("mtp.")));
        assert_eq!(glm5_mtp_prediction_len(7).unwrap(), 5);
        assert!(glm5_mtp_prediction_len(2).is_err());
        assert_eq!(glm5_mtp_prediction_len_for_layers(7, 2).unwrap(), 4);
        assert!(glm5_mtp_prediction_len_for_layers(3, 2).is_err());
    }

    #[test]
    fn glm52_indexer_schedule_matches_megatron_phase() {
        let types = derive_indexer_types(78, 4, 3);
        let compute_layers: Vec<_> = types
            .iter()
            .enumerate()
            .filter_map(|(layer, kind)| (kind == "full").then_some(layer))
            .collect();
        let mut expected = vec![0, 1, 2];
        expected.extend((6..=74).step_by(4));
        assert_eq!(compute_layers, expected);
        for layer in [3, 4, 5, 75, 76, 77] {
            assert_eq!(types[layer], "shared");
        }

        let zero_offset = derive_indexer_types(8, 4, 0);
        let zero_offset_compute: Vec<_> = zero_offset
            .iter()
            .enumerate()
            .filter_map(|(layer, kind)| (kind == "full").then_some(layer))
            .collect();
        assert_eq!(zero_offset_compute, vec![0, 4]);
    }

    #[test]
    fn config_rejects_shared_indexer_without_source() {
        let path = std::env::temp_dir().join(format!(
            "rustrain-glm5-invalid-config-{}.json",
            std::process::id()
        ));
        let json = serde_json::json!({
            "model_type": "glm_moe_dsa",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 128,
            "q_lora_rank": 16,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "v_head_dim": 8,
            "rope_parameters": {"rope_theta": 8000000.0, "rope_type": "default"},
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "index_head_dim": 16,
            "index_n_heads": 2,
            "index_topk": 4,
            "indexer_types": ["shared", "full"],
            "mlp_layer_types": ["dense", "sparse"],
            "rope_interleave": true,
            "indexer_rope_interleave": true
        });
        fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        let error = read_glm5_config(&path).unwrap_err();
        let _ = fs::remove_file(path);
        assert!(error.to_string().contains("no preceding full source layer"));
    }

    #[test]
    fn sparse_bias_mask_contains_no_nan() {
        let selected = Tensor::from_slice(&[1.0_f32, 0.0]).reshape([1, 2]);
        let bias = Tensor::zeros_like(&selected).masked_fill(&selected.eq(0), f64::NEG_INFINITY);
        assert_eq!(bias.isnan().any().int64_value(&[]), 0);
        assert!(bias.double_value(&[0, 1]).is_infinite());
    }
}
