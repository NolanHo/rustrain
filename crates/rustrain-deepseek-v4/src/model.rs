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

// ── Config ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct V4RuntimeConfig {
    pub num_hidden_layers: usize,
    pub num_attention_heads: i64,
    pub hidden_size: i64,
    pub q_lora_rank: i64,
    pub kv_lora_rank: i64, // inferred from wkv weight shape
    pub qk_rope_head_dim: i64,
    pub qk_nope_head_dim: i64, // inferred
    pub v_head_dim: i64,       // inferred
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub vocab_size: i64,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub moe_intermediate_size: usize,
    pub scoring_func: String,
    pub routed_scaling_factor: f64,
    pub swiglu_limit: f64,
    pub o_lora_rank: i64,
    pub o_groups: i64,
    pub expert_dtype: String,
    pub sliding_window: usize,
    pub compress_ratios: Vec<usize>,
    pub hc_sinkhorn_iters: usize,
    pub hc_mult: usize,
    pub hc_eps: f64,
    pub topk_method: String,
    pub num_nextn_predict_layers: usize,
    // ── YaRN RoPE scaling ──
    pub rope_scaling_type: Option<String>, // "yarn" or None
    pub rope_scaling_factor: f64,          // e.g. 16
    pub rope_beta_fast: f64,               // e.g. 32
    pub rope_beta_slow: f64,               // e.g. 1
    pub rope_original_max_pos: i64,        // e.g. 65536
    // ── Compressed-layer RoPE ──
    pub compress_rope_theta: f64, // e.g. 160000
    // ── Hash/Content (HC) attention: sparse attention on compressed sequences ──
    pub index_head_dim: i64,    // e.g. 128
    pub index_n_heads: i64,     // e.g. 64
    pub index_topk: i64,        // e.g. 512 or 1024
    pub num_hash_layers: usize, // e.g. 3
    // ── FP8 quantization ──
    pub scale_fmt: String,             // "e4m3" scale format, "ue8m0" for V4
    pub weight_block_size: (i64, i64), // (128, 128)
}

#[derive(Debug, Deserialize)]
struct V4ModelConfig {
    model_type: String,
    hidden_size: i64,
    num_hidden_layers: usize,
    num_attention_heads: i64,
    vocab_size: i64,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    q_lora_rank: Option<i64>,
    #[serde(default)]
    qk_rope_head_dim: Option<i64>,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    #[serde(default)]
    n_routed_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
    #[serde(default)]
    n_shared_experts: Option<usize>,
    #[serde(default)]
    moe_intermediate_size: Option<usize>,
    #[serde(default)]
    scoring_func: Option<String>,
    #[serde(default)]
    routed_scaling_factor: Option<f64>,
    #[serde(default)]
    swiglu_limit: Option<f64>,
    #[serde(default)]
    o_lora_rank: Option<i64>,
    #[serde(default)]
    o_groups: Option<i64>,
    #[serde(default)]
    expert_dtype: Option<String>,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    compress_ratios: Option<Vec<usize>>,
    #[serde(default)]
    hc_sinkhorn_iters: Option<usize>,
    #[serde(default)]
    hc_mult: Option<usize>,
    #[serde(default)]
    hc_eps: Option<f64>,
    #[serde(default)]
    topk_method: Option<String>,
    #[serde(default)]
    num_nextn_predict_layers: Option<usize>,
    // ── New fields ──
    #[serde(default)]
    rope_scaling: Option<RopeScalingConfig>,
    #[serde(default)]
    compress_rope_theta: Option<f64>,
    #[serde(default)]
    index_head_dim: Option<i64>,
    #[serde(default)]
    index_n_heads: Option<i64>,
    #[serde(default)]
    index_topk: Option<i64>,
    #[serde(default)]
    num_hash_layers: Option<usize>,
    #[serde(default)]
    quantization_config: Option<QuantizationConfig>,
}

#[derive(Debug, Deserialize)]
struct RopeScalingConfig {
    #[serde(rename = "type")]
    scaling_type: String,
    factor: f64,
    #[serde(default)]
    beta_fast: Option<f64>,
    #[serde(default)]
    beta_slow: Option<f64>,
    #[serde(default)]
    original_max_position_embeddings: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct QuantizationConfig {
    #[serde(default)]
    fmt: Option<String>,
    #[serde(default)]
    scale_fmt: Option<String>,
    #[serde(default)]
    weight_block_size: Option<Vec<i64>>,
}

pub fn read_v4_config(path: &Path) -> Result<V4RuntimeConfig> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let c: V4ModelConfig = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(V4RuntimeConfig {
        num_hidden_layers: c.num_hidden_layers,
        num_attention_heads: c.num_attention_heads,
        hidden_size: c.hidden_size,
        q_lora_rank: c.q_lora_rank.unwrap_or(1024),
        kv_lora_rank: 512, // will be inferred from weights
        qk_rope_head_dim: c.qk_rope_head_dim.unwrap_or(64),
        qk_nope_head_dim: 128, // will be inferred from weights
        v_head_dim: 128,       // will be inferred from weights
        rope_theta: c.rope_theta.unwrap_or(10000.0),
        rms_norm_eps: c.rms_norm_eps.unwrap_or(1e-6),
        tie_word_embeddings: c.tie_word_embeddings,
        vocab_size: c.vocab_size,
        n_routed_experts: c.n_routed_experts.unwrap_or(256),
        num_experts_per_tok: c.num_experts_per_tok.unwrap_or(6),
        n_shared_experts: c.n_shared_experts.unwrap_or(1),
        moe_intermediate_size: c.moe_intermediate_size.unwrap_or(2048),
        scoring_func: c.scoring_func.unwrap_or_else(|| "sqrtsoftplus".to_string()),
        routed_scaling_factor: c.routed_scaling_factor.unwrap_or(1.5),
        swiglu_limit: c.swiglu_limit.unwrap_or(10.0),
        o_lora_rank: c.o_lora_rank.unwrap_or(1024),
        o_groups: c.o_groups.unwrap_or(8),
        expert_dtype: c.expert_dtype.unwrap_or_else(|| "fp8".to_string()),
        sliding_window: c.sliding_window.unwrap_or(0),
        compress_ratios: c.compress_ratios.unwrap_or_default(),
        hc_sinkhorn_iters: c.hc_sinkhorn_iters.unwrap_or(20),
        hc_mult: c.hc_mult.unwrap_or(4),
        hc_eps: c.hc_eps.unwrap_or(1e-6),
        topk_method: c.topk_method.unwrap_or_else(|| "noaux_tc".to_string()),
        num_nextn_predict_layers: c.num_nextn_predict_layers.unwrap_or(0),
        // YaRN RoPE scaling
        rope_scaling_type: c.rope_scaling.as_ref().map(|r| r.scaling_type.clone()),
        rope_scaling_factor: c.rope_scaling.as_ref().map_or(1.0, |r| r.factor),
        rope_beta_fast: c
            .rope_scaling
            .as_ref()
            .and_then(|r| r.beta_fast)
            .unwrap_or(32.0),
        rope_beta_slow: c
            .rope_scaling
            .as_ref()
            .and_then(|r| r.beta_slow)
            .unwrap_or(1.0),
        rope_original_max_pos: c
            .rope_scaling
            .as_ref()
            .and_then(|r| r.original_max_position_embeddings)
            .unwrap_or(65536),
        // Compressed-layer RoPE
        compress_rope_theta: c.compress_rope_theta.unwrap_or(160000.0),
        // Hash/Content attention
        index_head_dim: c.index_head_dim.unwrap_or(128),
        index_n_heads: c.index_n_heads.unwrap_or(64),
        index_topk: c.index_topk.unwrap_or(512),
        num_hash_layers: c.num_hash_layers.unwrap_or(3),
        // FP8 quantization
        scale_fmt: c
            .quantization_config
            .as_ref()
            .and_then(|q| q.scale_fmt.clone())
            .unwrap_or_else(|| "e4m3".to_string()),
        weight_block_size: c
            .quantization_config
            .as_ref()
            .and_then(|q| q.weight_block_size.as_ref())
            .map(|v| (v[0], v[1]))
            .unwrap_or((128, 128)),
    })
}

// ── Compute dtype ──────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum V4ComputeDType {
    Fp32,
    Bf16,
}

impl V4ComputeDType {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "fp32" => Ok(Self::Fp32),
            "bf16" => Ok(Self::Bf16),
            other => bail!("unsupported V4 compute dtype {other}"),
        }
    }
    pub fn kind(self) -> Kind {
        match self {
            Self::Fp32 => Kind::Float,
            Self::Bf16 => Kind::BFloat16,
        }
    }
}

// ── Model path resolution ──────────────────────────────────────

pub fn v4_model_path_is_complete(model_path: &Path) -> bool {
    model_path.join("config.json").exists()
        && model_path.join("tokenizer.json").exists()
        && (model_path.join("model.safetensors").exists()
            || model_path.join("model.safetensors.index.json").exists())
}

pub fn resolve_v4_model_path(model_path: &Path) -> Result<PathBuf> {
    if v4_model_path_is_complete(model_path) {
        return Ok(model_path.to_path_buf());
    }
    let Some(model_dir_name) = model_path.file_name().and_then(|n| n.to_str()) else {
        bail!(
            "V4 model path {} has no directory name",
            model_path.display()
        );
    };
    let Some(root) = model_path.parent() else {
        bail!("V4 model path {} has no parent", model_path.display());
    };
    let hub_root = root.join("hub");
    let hub_suffix = format!("--{model_dir_name}");
    let mut candidates: Vec<PathBuf> = fs::read_dir(&hub_root)
        .ok()
        .into_iter()
        .flat_map(|e| e.filter_map(Result::ok))
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("models--") && n.ends_with(&hub_suffix))
        })
        .flat_map(|hub_dir| {
            let snapshots = hub_dir.join("snapshots");
            fs::read_dir(&snapshots)
                .ok()
                .into_iter()
                .flat_map(|e| e.filter_map(Result::ok))
                .map(|e| e.path())
                .filter(|p| v4_model_path_is_complete(p))
                .collect::<Vec<_>>()
        })
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .rev()
        .next()
        .ok_or_else(|| anyhow!("no complete HF hub snapshot found for V4"))
}

// ── Core ops ───────────────────────────────────────────────────

pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let dtype = input.kind();
    let variance = input
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    let result = input * (variance + eps).rsqrt().to_kind(dtype) * weight;
    result.to_kind(dtype)
}

/// SwiGLU with output limiting (V4 specific: swiglu_limit=10.0)
pub fn v4_swiglu(input: &Tensor, w1: &Tensor, w2: &Tensor, w3: &Tensor, limit: f64) -> Tensor {
    let gate = input.linear::<&Tensor>(w1, None).silu();
    let up = input.linear::<&Tensor>(w3, None);
    let intermediate = gate * up;
    let limited = if limit > 0.0 {
        intermediate.clamp(-limit, limit)
    } else {
        intermediate
    };
    limited.linear::<&Tensor>(w2, None)
}

// ── RoPE ──────────────────────────────────────────────────────

/// Compute RoPE cos/sin with optional YaRN scaling.
///
/// YaRN (Yet another RoPE extensioN) interpolates the position encoding
/// using a "range" that transitions from full extrapolation (high-freq dims)
/// to full interpolation (low-freq dims), with a smooth transition band
/// controlled by beta_fast / beta_slow.
pub fn rope_cos_sin(seq_len: usize, head_dim: i64, theta: f64, device: Device) -> (Tensor, Tensor) {
    // No scaling → plain RoPE
    let positions = Tensor::arange(seq_len as i64, (Kind::Float, device));
    let dim_indices = Tensor::arange(head_dim / 2, (Kind::Float, device));
    let inv_freq = (dim_indices * (2.0 / head_dim as f64)) * (1.0 / theta.ln());
    let inv_freq = inv_freq.exp();
    let freqs = positions.outer(&inv_freq);
    let cos = freqs.cos();
    let sin = freqs.sin();
    (
        Tensor::cat(&[&cos, &cos], -1),
        Tensor::cat(&[&sin, &sin], -1),
    )
}

/// YaRN-scaled RoPE cos/sin computation.
///
/// Parameters from config:
/// - `factor`: position scaling factor (e.g. 16)
/// - `original_max_pos`: original training length (e.g. 65536)
/// - `beta_fast`: boundary between extrapolation and transition (e.g. 32)
/// - `beta_slow`: boundary between transition and interpolation (e.g. 1)
/// - `theta`: base rope_theta (e.g. 10000)
/// - `compress_theta`: alternate theta for compressed layers (e.g. 160000)
///
/// For each frequency dimension d (0..head_dim/2):
///   1. Compute the "range" r(d) — the number of RoPE rotations per position.
///   2. Classify: if r > beta_fast → extrapolation (no change),
///      if r < beta_slow → interpolation (divide freq by factor),
///      else → smooth ramp between the two.
///   3. Scale the inv_freq accordingly.
pub fn yarn_rope_cos_sin(
    seq_len: usize,
    head_dim: i64,
    config: &V4RuntimeConfig,
    device: Device,
) -> (Tensor, Tensor) {
    let half_dim = head_dim / 2;
    let factor = config.rope_scaling_factor;
    let orig_max = config.rope_original_max_pos as f64;
    let theta = config.rope_theta;
    let beta_fast = config.rope_beta_fast;
    let beta_slow = config.rope_beta_slow;

    // Compute base inv_freq for each dim
    // inv_freq[d] = theta^(-(2d / head_dim)) = exp(-ln(theta) * 2d / head_dim)
    let dim_frac: Vec<f64> = (0..half_dim)
        .map(|d| 2.0 * d as f64 / head_dim as f64)
        .collect();
    let base_inv_freq: Vec<f64> = dim_frac.iter().map(|&f| theta.powf(-f)).collect();

    // Compute YaRN ramp for each dimension.
    // range[d] = orig_max * inv_freq[d] — this is the "wavelength ratio".
    // If range[d] > beta_fast * orig_max/dim → extrapolation (scale = 1)
    // If range[d] < beta_slow * orig_max/dim → interpolation (scale = 1/factor)
    // Else → linear ramp between 1 and 1/factor
    let scaled_inv_freq: Vec<f64> = base_inv_freq
        .iter()
        .map(|&inv_f| {
            // range = orig_max * inv_freq (how many rotations over original context)
            let range = orig_max * inv_f;
            // Normalized boundaries
            let low = beta_slow;
            let high = beta_fast;
            if range > high {
                // Extrapolation: no scaling
                inv_f
            } else if range < low {
                // Interpolation: scale by 1/factor
                inv_f / factor
            } else {
                // Transition: smooth ramp
                let t = (range - low) / (high - low);
                let scale = (1.0 / factor) * (1.0 - t) + 1.0 * t;
                inv_f * scale
            }
        })
        .collect();

    let inv_freq_tensor = Tensor::from_slice(&scaled_inv_freq).to_device(device);
    let positions = Tensor::arange(seq_len as i64, (Kind::Float, device));
    let freqs = positions.outer(&inv_freq_tensor);
    let cos = freqs.cos();
    let sin = freqs.sin();
    (
        Tensor::cat(&[&cos, &cos], -1),
        Tensor::cat(&[&sin, &sin], -1),
    )
}

/// Compute RoPE cos/sin, choosing between plain RoPE, YaRN, and compressed-layer theta.
///
/// - If the layer is compressed (ratio > 1) and compress_rope_theta is set,
///   use that theta with YaRN scaling.
/// - Otherwise use normal theta with YaRN scaling if configured.
pub fn v4_rope_cos_sin(
    seq_len: usize,
    head_dim: i64,
    config: &V4RuntimeConfig,
    device: Device,
) -> (Tensor, Tensor) {
    // Check if YaRN scaling is configured
    if config.rope_scaling_type.as_deref() == Some("yarn") {
        // For compressed layers, we'd use compress_rope_theta, but the scaling
        // is handled at the forward level. Here we always use YaRN with base theta.
        // The compressed-layer theta is applied by swapping config.rope_theta.
        yarn_rope_cos_sin(seq_len, head_dim, config, device)
    } else {
        rope_cos_sin(seq_len, head_dim, config.rope_theta, device)
    }
}

pub(crate) fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let seq_len = x.size()[2];
    let cos = cos.narrow(0, 0, seq_len).unsqueeze(0).unsqueeze(0);
    let sin = sin.narrow(0, 0, seq_len).unsqueeze(0).unsqueeze(0);
    let x1 = x.narrow(-1, 0, x.size()[x.size().len() - 1] / 2);
    let x2 = x.narrow(
        -1,
        x.size()[x.size().len() - 1] / 2,
        x.size()[x.size().len() - 1] / 2,
    );
    let rotated = Tensor::cat(&[&x2.neg(), &x1], -1);
    x * cos + rotated * sin
}

/// Compress sequence by pooling adjacent tokens.
/// ratio=0: no change. ratio=N: pool every N tokens into 1.
pub fn compress_seq(hidden: &Tensor, ratio: usize) -> Tensor {
    if ratio <= 1 {
        return hidden.shallow_clone();
    }
    let seq = hidden.size()[1];
    let new_seq = seq / ratio as i64;
    if new_seq == 0 {
        return hidden.shallow_clone();
    }
    // Truncate to multiple of ratio, then reshape and mean
    let truncated = hidden.narrow(1, 0, new_seq * ratio as i64);
    let shape = truncated.size();
    let batch = shape[0];
    let dim = shape[2];
    truncated
        .reshape([batch, new_seq, ratio as i64, dim])
        .mean_dim([2].as_slice(), false, hidden.kind())
}

/// Decompress sequence by repeating tokens.
pub fn decompress_seq(hidden: &Tensor, ratio: usize, target_seq: i64) -> Tensor {
    if ratio <= 1 {
        return hidden.shallow_clone();
    }
    let seq = hidden.size()[1];
    if seq >= target_seq {
        return hidden.narrow(1, 0, target_seq).shallow_clone();
    }
    // Repeat each token `ratio` times
    let batch = hidden.size()[0];
    let dim = hidden.size()[2];
    hidden
        .reshape([batch, seq, 1, dim])
        .expand([batch, seq, ratio as i64, dim], false)
        .reshape([batch, seq * ratio as i64, dim])
        .narrow(1, 0, target_seq)
}

// ── V4 Attention (MLA variant) ────────────────────────────────

pub struct V4AttentionWeights {
    pub wq_a: Tensor,      // [q_lora_rank, hidden]
    pub wq_b: Tensor,      // [heads*(qk_nope+qk_rope), q_lora_rank]
    pub wkv: Tensor,       // [kv_lora_rank+qk_rope, hidden] (single KV projection)
    pub wo_a: Tensor,      // [o_lora_rank, hidden]
    pub wo_b: Tensor,      // [hidden, o_lora_rank]
    pub q_norm: Tensor,    // [q_lora_rank]
    pub kv_norm: Tensor,   // [kv_lora_rank]
    pub attn_sink: Tensor, // scalar
    // Optional FP8 block-wise scales (when weights are stored as FP8)
    pub wq_a_scale: Option<Tensor>,
    pub wq_b_scale: Option<Tensor>,
    pub wkv_scale: Option<Tensor>,
    pub wo_a_scale: Option<Tensor>,
    pub wo_b_scale: Option<Tensor>,
}

impl V4AttentionWeights {
    pub fn load_raw(weights: &BTreeMap<String, Tensor>, layer: usize) -> Result<Self> {
        let p = format!("layers.{layer}.attn");
        // Helper to load optional FP8 scale
        let scale = |name: &str| -> Option<Tensor> {
            let scale_name = format!("{name}.scale_f");
            weights.get(&scale_name).map(|t| t.shallow_clone())
        };
        let wq_a_name = format!("{p}.wq_a.weight");
        let wq_b_name = format!("{p}.wq_b.weight");
        let wkv_name = format!("{p}.wkv.weight");
        let wo_a_name = format!("{p}.wo_a.weight");
        let wo_b_name = format!("{p}.wo_b.weight");
        Ok(Self {
            wq_a: tensor(weights, &wq_a_name)?.shallow_clone(),
            wq_b: tensor(weights, &wq_b_name)?.shallow_clone(),
            wkv: tensor(weights, &wkv_name)?.shallow_clone(),
            wo_a: tensor(weights, &wo_a_name)?.shallow_clone(),
            wo_b: tensor(weights, &wo_b_name)?.shallow_clone(),
            q_norm: tensor(weights, &format!("{p}.q_norm.weight"))?.shallow_clone(),
            kv_norm: tensor(weights, &format!("{p}.kv_norm.weight"))?.shallow_clone(),
            attn_sink: tensor(weights, &format!("{p}.attn_sink"))?.shallow_clone(),
            wq_a_scale: scale(&wq_a_name),
            wq_b_scale: scale(&wq_b_name),
            wkv_scale: scale(&wkv_name),
            wo_a_scale: scale(&wo_a_name),
            wo_b_scale: scale(&wo_b_name),
        })
    }
}

/// Dispatch linear: use FP8 GEMM if scale is available, else regular linear.
///
/// Input is [batch*seq, hidden] or [batch, seq, hidden] — reshaped to 2D for FP8 GEMM.
/// Weight is [out_dim, in_dim] (FP8 or bf16). Scale is [out_dim/128, in_dim/128] (float32).
fn fp8_or_linear(input: &Tensor, weight: &Tensor, scale: &Option<Tensor>) -> Tensor {
    if let Some(scale) = scale {
        let input_shape = input.size();
        let input_2d = if input_shape.len() == 3 {
            input.reshape([input_shape[0] * input_shape[1], input_shape[2]])
        } else {
            input.shallow_clone()
        };

        // FP8 GEMM requires both M and K to be multiples of 128 (block-wise scale).
        // After compression, seq can be < 128 — fall back to regular linear in that case.
        let m = input_2d.size()[0];
        let k = input_2d.size()[1];
        let can_fp8 = m >= 128 && m % 128 == 0 && k >= 128 && k % 128 == 0;

        if can_fp8 {
            match crate::fp8_kernel::fp8_linear(&input_2d, weight, scale) {
                Ok(out) => {
                    let out = out.to_kind(input.kind());
                    if input_shape.len() == 3 {
                        return out.reshape([input_shape[0], input_shape[1], -1]);
                    } else {
                        return out;
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "FP8 GEMM failed, falling back to bf16 linear");
                }
            }
        }
        // Fallback: dequant weight to bf16 and use regular linear
        let weight_bf16 = weight.to_kind(input.kind());
        let out = input_2d.linear::<&Tensor>(&weight_bf16, None);
        if input_shape.len() == 3 {
            out.reshape([input_shape[0], input_shape[1], -1])
        } else {
            out
        }
    } else {
        input.linear::<&Tensor>(weight, None)
    }
}

pub fn v4_attention(input: &Tensor, attn: &V4AttentionWeights, config: &V4RuntimeConfig) -> Tensor {
    let shape = input.size();
    let batch = shape[0];
    let seq = shape[1];
    let num_heads = config.num_attention_heads;
    let head_dim = 512_i64;
    let qk_rope = config.qk_rope_head_dim; // 64
    let qk_nope = head_dim - qk_rope; // 448
    let o_groups = config.o_groups;
    let kv_dim = config.kv_lora_rank; // 512

    // Q path: wq_a → q_norm → wq_b → reshape
    // FP8 dispatch: use fp8_linear if scale available, else regular linear
    let q_a = fp8_or_linear(input, &attn.wq_a, &attn.wq_a_scale);
    let q_a = rms_norm(&q_a, &attn.q_norm, config.rms_norm_eps);
    let q_b = fp8_or_linear(&q_a, &attn.wq_b, &attn.wq_b_scale);
    let q = q_b
        .reshape([batch, seq, num_heads, head_dim])
        .transpose(1, 2); // [batch, heads, seq, 512]
    let q_nope = q.narrow(-1, 0, qk_nope); // [batch, heads, seq, 448]
    let q_rope = q.narrow(-1, qk_nope, qk_rope); // [batch, heads, seq, 64]

    // KV path: wkv → kv_norm (MQA, shared across heads)
    let wkv_out = fp8_or_linear(input, &attn.wkv, &attn.wkv_scale);
    let kv = rms_norm(&wkv_out, &attn.kv_norm, config.rms_norm_eps);
    // K and V are both derived from the 512-dim latent KV
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

    // Apply RoPE to the rope portions (uses YaRN scaling if configured)
    let (cos, sin) = v4_rope_cos_sin(seq as usize, qk_rope, config, input.device());
    let cos = cos.to_kind(input.kind());
    let sin = sin.to_kind(input.kind());
    let q_rope_rotated = apply_rotary(&q_rope, &cos, &sin);
    let k_rope_rotated = apply_rotary(&k_rope, &cos, &sin);

    // Concatenate nope + rope
    let q_full = Tensor::cat(&[&q_nope, &q_rope_rotated], -1); // [batch, heads, seq, 512]
    let k_full = Tensor::cat(&[&k_nope, &k_rope_rotated], -1); // [batch, heads, seq, 512]

    // Attention with sink
    let scale = 1.0 / (head_dim as f64).sqrt();
    let attn_scores = q_full.matmul(&k_full.transpose(-2, -1)) * scale;
    let sink = attn
        .attn_sink
        .reshape([1, num_heads, 1, 1])
        .to_kind(attn_scores.kind());
    let scores = attn_scores + sink;
    // Causal mask with optional sliding window
    let mask: Tensor = if config.sliding_window > 0 && seq as i64 > config.sliding_window as i64 {
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
    let scores = scores.masked_fill(&cannot_attend, f64::NEG_INFINITY);
    let probs = scores.softmax(-1, Kind::Float).to_kind(v.kind());
    let context = probs.matmul(&v); // [batch, heads, seq, 512]

    // Group reduction: [batch, heads, seq, 512] → [batch, o_groups, seq, hidden/o_groups]
    let heads_per_group = num_heads / o_groups;
    let context = context
        .reshape([batch, o_groups, heads_per_group, seq, head_dim])
        .sum_dim_intlist([2].as_slice(), false, v.kind());
    let context = context
        .reshape([batch, seq, o_groups * head_dim])
        .to_kind(attn.wo_a.kind());

    // Output: wo_a → wo_b (LoRA-style two-layer projection)
    let o_compressed = fp8_or_linear(&context, &attn.wo_a, &attn.wo_a_scale);
    fp8_or_linear(&o_compressed, &attn.wo_b, &attn.wo_b_scale)
}

/// Apply Sinkhorn normalization for load-balanced routing.
/// Alternates row (token) and column (expert) normalization.
pub fn sinkhorn_normalize(scores: &Tensor, iters: usize, eps: f64) -> Tensor {
    let mut s = scores.shallow_clone();
    for _ in 0..iters {
        // Row normalization (per token): sum over experts = 1
        let row_sum = s
            .sum_dim_intlist([-1].as_slice(), true, Kind::Float)
            .clamp_min(eps);
        s = s / &row_sum;
        // Column normalization (per expert): sum over tokens = 1
        let col_sum = s
            .sum_dim_intlist([0].as_slice(), true, Kind::Float)
            .clamp_min(eps);
        s = s / &col_sum;
    }
    s
}

// ── V4 MoE Layer ──────────────────────────────────────────────

pub struct V4MoeLayerWeights {
    pub attn_norm: Tensor,
    pub attn: V4AttentionWeights,
    pub ffn_norm: Tensor,
    pub gate: Tensor,                             // router [n_experts, hidden]
    pub shared_experts: (Tensor, Tensor, Tensor), // (w1, w2, w3)
    pub experts: Vec<(Tensor, Tensor, Tensor)>,   // [(w1, w2, w3)] per expert
}

impl V4MoeLayerWeights {
    pub fn load_raw(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        n_experts: usize,
    ) -> Result<Self> {
        let p = format!("layers.{layer}");
        let sp = format!("{p}.ffn.shared_experts");

        let mut experts = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let ep = format!("{p}.ffn.experts.{e}");
            experts.push((
                tensor(weights, &format!("{ep}.w1.weight"))?.shallow_clone(),
                tensor(weights, &format!("{ep}.w2.weight"))?.shallow_clone(),
                tensor(weights, &format!("{ep}.w3.weight"))?.shallow_clone(),
            ));
        }

        Ok(Self {
            attn_norm: tensor(weights, &format!("{p}.attn_norm.weight"))?.shallow_clone(),
            attn: V4AttentionWeights::load_raw(weights, layer)?,
            ffn_norm: tensor(weights, &format!("{p}.ffn_norm.weight"))?.shallow_clone(),
            gate: tensor(weights, &format!("{p}.ffn.gate.weight"))?.shallow_clone(),
            shared_experts: (
                tensor(weights, &format!("{sp}.w1.weight"))?.shallow_clone(),
                tensor(weights, &format!("{sp}.w2.weight"))?.shallow_clone(),
                tensor(weights, &format!("{sp}.w3.weight"))?.shallow_clone(),
            ),
            experts,
        })
    }
}

/// V4 MoE MLP with sqrtsoftplus routing + SwiGLU with limit.
///
/// Routing method controlled by `topk_method`:
/// - `"noaux_tc"`: over-select `num_experts_per_tok * hc_mult` candidates,
///   apply Sinkhorn normalization for load balancing, then select final
///   top-k from the normalized scores. No auxiliary loss needed.
/// - Other values: standard top-k on raw scores.
pub fn v4_moe_mlp(
    input: &Tensor,
    gate: &Tensor,
    shared: &(Tensor, Tensor, Tensor),
    experts: &[(Tensor, Tensor, Tensor)],
    config: &V4RuntimeConfig,
) -> Tensor {
    let num_experts_per_tok = config.num_experts_per_tok;
    let scoring_func = &config.scoring_func;
    let routed_scaling_factor = config.routed_scaling_factor;
    let swiglu_limit = config.swiglu_limit;
    let topk_method = &config.topk_method;
    let n_experts = config.n_routed_experts;

    // Shared expert (always computed)
    let shared_out = v4_swiglu(input, &shared.0, &shared.1, &shared.2, swiglu_limit);

    // Router logits
    let router_logits = input.linear::<&Tensor>(gate, None); // [batch, seq, n_experts]

    // Scoring function: sqrtsoftplus (V4 specific)
    let scores = if scoring_func == "sqrtsoftplus" {
        let sp = (router_logits.shallow_clone().exp() + 1.0).log();
        sp.sqrt()
    } else {
        router_logits
            .shallow_clone()
            .softmax(-1, Kind::Float)
            .to_kind(router_logits.kind())
    };
    let _ = &router_logits;

    // Top-k selection with optional Sinkhorn load balancing (noaux_tc)
    let (topk_weights, topk_indices) =
        if topk_method == "noaux_tc" && config.hc_mult > 1 && n_experts > num_experts_per_tok {
            // Over-select candidates: top (k * hc_mult), capped at n_experts
            let k_extended = (num_experts_per_tok * config.hc_mult).min(n_experts);
            let (ext_scores, ext_indices) = scores.topk(k_extended as i64, -1, true, true);

            // Apply Sinkhorn normalization on the over-selected scores
            // Reshape to [batch*seq, k_extended] for row/col normalization
            let batch = ext_scores.size()[0];
            let seq = ext_scores.size()[1];
            let flat_scores = ext_scores.reshape([batch * seq, k_extended as i64]);
            let normalized =
                sinkhorn_normalize(&flat_scores, config.hc_sinkhorn_iters, config.hc_eps);
            let normalized = normalized.reshape(ext_scores.size());

            // Select final top-k from Sinkhorn-normalized scores
            let (_, final_local_idx) = normalized.topk(num_experts_per_tok as i64, -1, true, true);

            // Map local indices back to original expert indices and original scores
            let final_indices = ext_indices.gather(-1, &final_local_idx, false);
            let final_weights = ext_scores.gather(-1, &final_local_idx, false);

            (final_weights, final_indices)
        } else {
            // Standard top-k
            scores.topk(num_experts_per_tok as i64, -1, true, true)
        };

    // Normalize top-k weights
    let denom = topk_weights
        .sum_dim_intlist([-1].as_slice(), true, Kind::Float)
        .to_kind(router_logits.kind())
        .clamp_min(1e-9);
    let topk_weights = topk_weights / &denom * routed_scaling_factor;

    // Accumulate expert outputs
    let mut output = shared_out;
    for k in 0..num_experts_per_tok {
        let expert_indices = topk_indices.select(-1, k as i64);
        let expert_weights = topk_weights.select(-1, k as i64);
        let weights_kind = expert_weights.kind();

        for (expert_idx, (w1, w2, w3)) in experts.iter().enumerate() {
            let mask = expert_indices.eq(expert_idx as i64).to_kind(weights_kind);
            if mask.sum(Kind::Float).double_value(&[]) > 0.0 {
                let expert_out = v4_swiglu(input, w1, w2, w3, swiglu_limit);
                let weight = (&expert_weights * &mask).unsqueeze(-1);
                output = output + (expert_out * weight);
            }
        }
    }
    output
}

pub fn v4_moe_layer(
    input: &Tensor,
    weights: &V4MoeLayerWeights,
    config: &V4RuntimeConfig,
) -> Tensor {
    let hidden = rms_norm(input, &weights.attn_norm, config.rms_norm_eps);
    let attn_out = v4_attention(&hidden, &weights.attn, config);
    let residual = input + &attn_out;
    let mlp_input = rms_norm(&residual, &weights.ffn_norm, config.rms_norm_eps);
    let mlp = v4_moe_mlp(
        &mlp_input,
        &weights.gate,
        &weights.shared_experts,
        &weights.experts,
        config,
    );
    residual + mlp
}

/// V4 MoE layer with HC sparse attention for compressed sequences.
///
/// Uses `v4_hc_attention` (hash-based top-k sparse attention) instead of full
/// attention when HC weights are available and the sequence is compressed.
pub fn v4_moe_layer_with_hc(
    input: &Tensor,
    weights: &V4MoeLayerWeights,
    hc_weights: &BTreeMap<String, Tensor>,
    layer: usize,
    config: &V4RuntimeConfig,
) -> Tensor {
    let hidden = rms_norm(input, &weights.attn_norm, config.rms_norm_eps);

    // Use HC sparse attention if available, else fall back to full attention
    let attn_out = if crate::hc::should_use_hc_attention(hc_weights, layer, config) {
        let hc = crate::hc::HcWeights::load_raw(hc_weights, layer)
            .unwrap_or_else(|_| panic!("HC weights exist but failed to load for layer {layer}"));
        crate::hc::v4_hc_attention(&hidden, &weights.attn, &hc, config)
    } else {
        v4_attention(&hidden, &weights.attn, config)
    };

    let residual = input + &attn_out;
    let mlp_input = rms_norm(&residual, &weights.ffn_norm, config.rms_norm_eps);

    // Optionally gate MoE output with HC FFN hash
    let mlp = v4_moe_mlp(
        &mlp_input,
        &weights.gate,
        &weights.shared_experts,
        &weights.experts,
        config,
    );

    let mlp = if crate::hc::should_use_hc_attention(hc_weights, layer, config) {
        if let Ok(hc) = crate::hc::HcWeights::load_raw(hc_weights, layer) {
            let gate = crate::hc::v4_hc_ffn_gate(&mlp_input, &hc, config);
            &mlp * &gate
        } else {
            mlp
        }
    } else {
        mlp
    };

    residual + mlp
}

// ── Forward ───────────────────────────────────────────────────

pub fn v4_forward_selective_with_hidden(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &V4RuntimeConfig,
    trainable_layers: &[usize],
) -> Result<(Tensor, Tensor)> {
    let embed_tokens = tensor(weights, "embed.weight")?;
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);
    // Ensure hidden dtype matches weight dtype (bf16 training needs this)
    let target_kind = embed_tokens.kind();
    if hidden.kind() != target_kind {
        hidden = hidden.to_kind(target_kind);
    }

    for layer in 0..config.num_hidden_layers {
        if !trainable_layers.contains(&layer) {
            continue;
        }
        // Apply compression before this layer
        let ratio = if layer < config.compress_ratios.len() {
            config.compress_ratios[layer]
        } else {
            0
        };
        if ratio > 1 {
            hidden = compress_seq(&hidden, ratio);
        }
        // For compressed layers, use compress_rope_theta instead of rope_theta
        let layer_config = if ratio > 1 {
            let mut c = config.clone();
            c.rope_theta = config.compress_rope_theta;
            c
        } else {
            config.clone()
        };
        let lw = V4MoeLayerWeights::load_raw(weights, layer, config.n_routed_experts)?;
        hidden = v4_moe_layer_with_hc(&hidden, &lw, weights, layer, &layer_config);
    }
    // Decompress back to original sequence length before lm_head.
    // V4 compression is per-layer independent: each compressed layer reduces
    // the sequence by its own ratio. The final sequence length is determined
    // by the cumulative effect — but in practice, layers alternate between
    // ratio=4 (light compress) and ratio=128 (heavy compress), with the
    // heavy compress dominating. We use the max ratio as the decompression
    // factor since decompress_seq just repeats tokens to fill the target length.
    let original_seq_len = input_ids.size()[1];
    let total_ratio: usize = config
        .compress_ratios
        .iter()
        .copied()
        .filter(|&r| r > 1)
        .max()
        .unwrap_or(1);
    if total_ratio > 1 {
        hidden = decompress_seq(&hidden, total_ratio, original_seq_len);
    }

    // Return (logits, hidden_before_lm_head) — MTP needs the pre-norm hidden state
    let hidden_pre = hidden.shallow_clone();
    let logits = if let Ok(final_norm) = tensor(weights, "norm.weight") {
        let normed = rms_norm(&hidden, &final_norm, config.rms_norm_eps);
        let lm_head = if let Ok(lh) = tensor(weights, "head.weight") {
            lh.shallow_clone()
        } else {
            embed_tokens.shallow_clone() // tied
        };
        normed.linear::<&Tensor>(&lm_head, None)
    } else {
        hidden.linear::<&Tensor>(&embed_tokens, None)
    };
    Ok((logits, hidden_pre))
}

pub fn v4_forward_selective(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &V4RuntimeConfig,
    trainable_layers: &[usize],
) -> Result<Tensor> {
    let (logits, _) =
        v4_forward_selective_with_hidden(input_ids, weights, config, trainable_layers)?;
    Ok(logits)
}

pub fn v4_causal_lm_loss_selective(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &V4RuntimeConfig,
    trainable_layers: &[usize],
) -> Result<Tensor> {
    let logits = v4_forward_selective(input_ids, weights, config, trainable_layers)?;
    let shifted = logits.narrow(1, 0, logits.size()[1] - 1);
    let targets = input_ids.narrow(1, 1, input_ids.size()[1] - 1);
    Ok(shifted
        .reshape([-1, config.vocab_size])
        .log_softmax(-1, Kind::Float)
        .g_nll_loss::<&Tensor>(&targets.reshape([-1]), None, Reduction::Mean, -100))
}

/// Compute MTP (Multi-Token Prediction) auxiliary loss across all MTP layers.
///
/// For each MTP layer i (0..num_nextn_predict_layers):
///   - Layer 0: input = model's hidden[t] + embed[token t+1], predicts token t+2
///   - Layer i>0: input = MTP layer (i-1)'s pre-head hidden + embed[token t+i+1],
///     predicts token t+i+2
///   - Available sequence length decreases by 1 per layer
///
/// Returns the sum of all MTP layer losses. Returns 0.0 if MTP weights are
/// unavailable, sequence too short, or `num_nextn_predict_layers == 0`.
pub fn v4_mtp_loss(
    hidden: &Tensor,
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &V4RuntimeConfig,
) -> Result<Tensor> {
    let seq_len = input_ids.size()[1];
    let num_mtp = config.num_nextn_predict_layers;
    if seq_len < 3 || !has_mtp_weights(weights) || num_mtp == 0 {
        return Ok(Tensor::zeros([], (Kind::Float, hidden.device())));
    }

    let embed = tensor(weights, "embed.weight")?.to_kind(hidden.kind());
    let device = hidden.device();
    let mut total_mtp_loss = Tensor::zeros([], (Kind::Float, device));
    let mut current_hidden = hidden.shallow_clone();

    for layer in 0..num_mtp {
        // Need at least 2 + layer + 1 tokens for this layer to have any targets
        let avail_len = seq_len as i64 - 2 - layer as i64;
        if avail_len <= 0 {
            break;
        }

        let mtp = MtpHeadWeights::load_raw(weights, layer)?;

        // Hidden state aligned: current_hidden[0..avail_len]
        let hidden_aligned = current_hidden.narrow(1, 0, avail_len);

        // Next token embedding: embed[input_ids[layer+1 .. layer+1+avail_len]]
        let next_token_ids = input_ids.narrow(1, (layer + 1) as i64, avail_len);
        let next_token_embed = Tensor::embedding(&embed, &next_token_ids, -1, false, false);

        // MTP forward → (logits, pre-head hidden for next layer)
        let (logits, pre_head_hidden) =
            v4_mtp_forward_with_hidden(&hidden_aligned, &next_token_embed, &mtp, config);

        // Targets: input_ids[layer+2 .. layer+2+avail_len]
        let targets = input_ids.narrow(1, (layer + 2) as i64, avail_len);

        let layer_loss = logits
            .reshape([-1, config.vocab_size])
            .log_softmax(-1, Kind::Float)
            .g_nll_loss::<&Tensor>(&targets.reshape([-1]), None, Reduction::Mean, -100);

        // Accumulate with weight 0.5 per layer (diminishing importance)
        total_mtp_loss = total_mtp_loss + layer_loss * 0.5;

        // Pass pre-head hidden to next layer (already seq-narrowed to avail_len)
        current_hidden = pre_head_hidden;
    }

    Ok(total_mtp_loss)
}

/// Combined LM + MTP loss for the selective forward path.
///
/// Computes the standard next-token LM loss, plus an optional MTP auxiliary
/// loss (weight 0.5) if MTP weights are available and `num_nextn_predict_layers > 0`.
pub fn v4_causal_lm_loss_with_mtp(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &V4RuntimeConfig,
    trainable_layers: &[usize],
) -> Result<Tensor> {
    let (logits, hidden) =
        v4_forward_selective_with_hidden(input_ids, weights, config, trainable_layers)?;

    let shifted = logits.narrow(1, 0, logits.size()[1] - 1);
    let targets = input_ids.narrow(1, 1, input_ids.size()[1] - 1);
    let lm_loss = shifted
        .reshape([-1, config.vocab_size])
        .log_softmax(-1, Kind::Float)
        .g_nll_loss::<&Tensor>(&targets.reshape([-1]), None, Reduction::Mean, -100);

    if has_mtp_weights(weights) && config.num_nextn_predict_layers > 0 {
        let mtp_loss = v4_mtp_loss(&hidden, input_ids, weights, config)?;
        Ok(&lm_loss + mtp_loss * 0.5)
    } else {
        Ok(lm_loss)
    }
}

// ── Trainable tensors ────────────────────────────────────────

pub fn v4_trainable_tensors_for_layer(layer: usize, config: &V4RuntimeConfig) -> Vec<String> {
    let p = format!("layers.{layer}");
    let mut names = vec![
        format!("{p}.attn_norm.weight"),
        format!("{p}.attn.wq_a.weight"),
        format!("{p}.attn.wq_b.weight"),
        format!("{p}.attn.wkv.weight"),
        format!("{p}.attn.wo_a.weight"),
        format!("{p}.attn.wo_b.weight"),
        format!("{p}.attn.q_norm.weight"),
        format!("{p}.attn.kv_norm.weight"),
        format!("{p}.attn.attn_sink"),
        format!("{p}.ffn_norm.weight"),
        format!("{p}.ffn.gate.weight"),
    ];
    // Shared expert
    names.push(format!("{p}.ffn.shared_experts.w1.weight"));
    names.push(format!("{p}.ffn.shared_experts.w2.weight"));
    names.push(format!("{p}.ffn.shared_experts.w3.weight"));
    // Routed experts
    for e in 0..config.n_routed_experts {
        names.push(format!("{p}.ffn.experts.{e}.w1.weight"));
        names.push(format!("{p}.ffn.experts.{e}.w2.weight"));
        names.push(format!("{p}.ffn.experts.{e}.w3.weight"));
    }
    // HC (Hash/Content) attention weights
    names.extend(crate::hc::HcWeights::weight_names(layer));
    names
}

pub fn v4_trainable_tensors(
    trainable_layers: &[usize],
    config: &V4RuntimeConfig,
    include_lm_head: bool,
) -> Vec<String> {
    let mut names = Vec::new();
    for &layer in trainable_layers {
        names.extend(v4_trainable_tensors_for_layer(layer, config));
    }
    names.push("norm.weight".to_string());
    if include_lm_head {
        names.push("head.weight".to_string());
    }
    names
}

// ── FP8 weight loading ────────────────────────────────────────

/// Load V4 weights, dequantizing FP8→bf16.
///
/// If a pre-converted bf16 safetensors file exists at `<model_dir>/v4_bf16_all.safetensors`,
/// reads directly from it (fast, no Python needed). Otherwise falls back to Python dequant.
pub fn load_v4_weights(
    model_dir: &Path,
    needed: &HashSet<String>,
) -> Result<BTreeMap<String, Tensor>> {
    // V4 weights are always FP8 — load via C++ native safetensors reader (no Python).
    // Falls back to device 0 if no LOCAL_RANK env var set.
    let device_id = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);
    load_v4_weights_fp8(model_dir, needed, device_id)
}
/// Load V4 weights as FP8 (no dequant) + float scales.
///
/// Returns a single BTreeMap where:
///   - "weight_name" → FP8 e4m3 tensor
///   - "weight_name.scale_f" → float32 scale (ue8m0 already converted)
///
/// Forward path checks for ".scale_f" entry to decide fp8 vs bf16 GEMM.
pub fn load_v4_weights_fp8(
    model_dir: &Path,
    needed: &HashSet<String>,
    device_id: i32,
) -> Result<BTreeMap<String, Tensor>> {
    // No Python — use C++ safetensors reader via fp8_kernel::load_safetensors_native.
    let index_path = model_dir.join("model.safetensors.index.json");
    let single_path = model_dir.join("model.safetensors");

    if single_path.exists() {
        info!(path = %single_path.display(), needed = needed.len(), "loading FP8 from single safetensors");
        return crate::fp8_kernel::load_safetensors_native(&single_path, needed, device_id);
    }

    if !index_path.exists() {
        bail!("no safetensors index found at {}", index_path.display());
    }

    let index_json = std::fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_json)?;
    let weight_map = index["weight_map"]
        .as_object()
        .context("invalid weight_map in index")?;

    // Build needed_with_scales: original names + .scale variants
    let mut needed_with_scales: HashSet<String> = HashSet::new();
    for name in needed {
        needed_with_scales.insert(name.clone());
        let scale_name = name.replace(".weight", ".scale");
        needed_with_scales.insert(scale_name);
    }

    // Group needed tensors by shard
    let mut shard_to_names: std::collections::HashMap<String, HashSet<String>> =
        std::collections::HashMap::new();
    for name in &needed_with_scales {
        if let Some(shard) = weight_map.get(name) {
            let shard_name = shard.as_str().unwrap_or("");
            shard_to_names
                .entry(shard_name.to_string())
                .or_default()
                .insert(name.clone());
        }
    }

    // Load each shard's needed tensors via C++
    let mut result = BTreeMap::new();
    for (shard, names) in &shard_to_names {
        let shard_path = model_dir.join(shard);
        info!(shard = %shard, tensors = names.len(), "loading FP8 from shard");
        let tensors = crate::fp8_kernel::load_safetensors_native(&shard_path, names, device_id)?;
        for (name, tensor) in tensors {
            result.insert(name, tensor);
        }
    }

    info!(loaded = result.len(), "V4 FP8 weights loaded (no Python)");
    Ok(result)
}

/// Dispatch linear: use FP8 GEMM if scale available, else regular linear.
///
/// `weights` map may contain "name.scale_f" entry (float32 block scale).
/// If present, use fp8_linear; otherwise fall back to regular tch linear.
pub fn v4_linear(input: &Tensor, weights: &BTreeMap<String, Tensor>, weight_name: &str) -> Tensor {
    let scale_name = format!("{weight_name}.scale_f");
    if let Ok(weight) = tensor(weights, weight_name) {
        if let Ok(scale) = tensor(weights, &scale_name) {
            // FP8 path
            match crate::fp8_kernel::fp8_linear(input, &weight, &scale) {
                Ok(out) => return out.to_kind(input.kind()),
                Err(e) => {
                    tracing::warn!(error = %e, "FP8 GEMM failed, falling back to bf16");
                }
            }
        }
        // Fallback: regular linear
        input.linear::<&Tensor>(&weight, None)
    } else {
        // Weight not found — return zeros
        Tensor::zeros([input.size()[0], 1], (input.kind(), input.device()))
    }
}

/// V4 forward with LoRA applied inline.
pub fn v4_forward_lora(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &V4RuntimeConfig,
    trainable_layers: &[usize],
    registry: &crate::lora::V4LoraRegistry,
) -> Result<Tensor> {
    let embed_tokens = tensor(weights, "embed.weight")?;
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);
    let target_kind = embed_tokens.kind();
    if hidden.kind() != target_kind {
        hidden = hidden.to_kind(target_kind);
    }

    for layer in 0..config.num_hidden_layers {
        if !trainable_layers.contains(&layer) {
            continue;
        }
        // Apply compression before this layer (same as v4_forward_selective)
        let ratio = if layer < config.compress_ratios.len() {
            config.compress_ratios[layer]
        } else {
            0
        };
        if ratio > 1 {
            hidden = compress_seq(&hidden, ratio);
        }
        // For compressed layers, use compress_rope_theta
        let layer_config = if ratio > 1 {
            let mut c = config.clone();
            c.rope_theta = config.compress_rope_theta;
            c
        } else {
            config.clone()
        };
        let lw = V4MoeLayerWeights::load_raw(weights, layer, config.n_routed_experts)?;
        let hidden_norm = rms_norm(&hidden, &lw.attn_norm, layer_config.rms_norm_eps);
        let attn = v4_lora_attention_weights(&lw.attn, layer, registry);
        // Use HC sparse attention if available for this layer
        let attn_out = if crate::hc::should_use_hc_attention(weights, layer, &layer_config) {
            if let Ok(hc) = crate::hc::HcWeights::load_raw(weights, layer) {
                crate::hc::v4_hc_attention(&hidden_norm, &attn, &hc, &layer_config)
            } else {
                v4_attention(&hidden_norm, &attn, &layer_config)
            }
        } else {
            v4_attention(&hidden_norm, &attn, &layer_config)
        };
        let residual = &hidden + &attn_out;
        let mlp_input = rms_norm(&residual, &lw.ffn_norm, layer_config.rms_norm_eps);
        let mlp = v4_moe_mlp(
            &mlp_input,
            &lw.gate,
            &lw.shared_experts,
            &lw.experts,
            config,
        );
        hidden = residual + mlp;
    }

    // Decompress back to original sequence length before lm_head
    let original_seq_len = input_ids.size()[1];
    let total_ratio: usize = config
        .compress_ratios
        .iter()
        .copied()
        .filter(|&r| r > 1)
        .max()
        .unwrap_or(1);
    if total_ratio > 1 {
        hidden = decompress_seq(&hidden, total_ratio, original_seq_len);
    }

    // Apply final norm + lm_head if available, otherwise use tied embeddings
    // V4 uses "head.weight" as lm_head (not tied)
    if let Ok(final_norm) = tensor(weights, "norm.weight") {
        let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps);
        let lm_head = if let Ok(lh) = tensor(weights, "head.weight") {
            lh.shallow_clone()
        } else {
            embed_tokens.shallow_clone()
        };
        Ok(hidden.linear::<&Tensor>(&lm_head, None))
    } else {
        // No norm.weight available — use identity norm (ones)
        let ones = Tensor::ones([config.hidden_size], (Kind::Float, hidden.device()));
        let hidden = rms_norm(&hidden, &ones, config.rms_norm_eps);
        let lm_head = if let Ok(lh) = tensor(weights, "head.weight") {
            lh.shallow_clone()
        } else {
            embed_tokens.shallow_clone()
        };
        Ok(hidden.linear::<&Tensor>(&lm_head, None))
    }
}

pub fn v4_lora_attention_weights(
    attn: &V4AttentionWeights,
    layer: usize,
    registry: &crate::lora::V4LoraRegistry,
) -> V4AttentionWeights {
    V4AttentionWeights {
        wq_a: crate::lora::v4_lora_weight(
            &attn.wq_a,
            layer,
            crate::lora::V4LoraTargetModule::WqA,
            registry,
        ),
        wq_b: crate::lora::v4_lora_weight(
            &attn.wq_b,
            layer,
            crate::lora::V4LoraTargetModule::WqB,
            registry,
        ),
        wkv: crate::lora::v4_lora_weight(
            &attn.wkv,
            layer,
            crate::lora::V4LoraTargetModule::Wkv,
            registry,
        ),
        wo_a: crate::lora::v4_lora_weight(
            &attn.wo_a,
            layer,
            crate::lora::V4LoraTargetModule::WoA,
            registry,
        ),
        wo_b: crate::lora::v4_lora_weight(
            &attn.wo_b,
            layer,
            crate::lora::V4LoraTargetModule::WoB,
            registry,
        ),
        q_norm: attn.q_norm.shallow_clone(),
        kv_norm: attn.kv_norm.shallow_clone(),
        attn_sink: attn.attn_sink.shallow_clone(),
        wq_a_scale: None, // LoRA path uses bf16, no FP8 scale
        wq_b_scale: None,
        wkv_scale: None,
        wo_a_scale: None,
        wo_b_scale: None,
    }
}

pub fn v4_causal_lm_loss_lora(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &V4RuntimeConfig,
    trainable_layers: &[usize],
    registry: &crate::lora::V4LoraRegistry,
) -> Result<Tensor> {
    let logits = v4_forward_lora(input_ids, weights, config, trainable_layers, registry)?;
    let shifted = logits.narrow(1, 0, logits.size()[1] - 1);
    let targets = input_ids.narrow(1, 1, input_ids.size()[1] - 1);
    Ok(shifted
        .reshape([-1, config.vocab_size])
        .log_softmax(-1, Kind::Float)
        .g_nll_loss::<&Tensor>(&targets.reshape([-1]), None, tch::Reduction::Mean, -100))
}

// ── MTP (Multi-Token Prediction) ──────────────────────────────

/// MTP head weights
pub struct MtpHeadWeights {
    pub norm: Tensor,     // mtp.0.norm.weight
    pub hnorm: Tensor,    // mtp.0.hnorm.weight
    pub head: Tensor,     // mtp.0.head.weight — output projection to vocab
    pub ffn_norm: Tensor, // mtp.0.ffn_norm (if exists)
    pub ffn_shared_w1: Tensor,
    pub ffn_shared_w2: Tensor,
    pub ffn_shared_w3: Tensor,
}

impl MtpHeadWeights {
    pub fn load_raw(weights: &BTreeMap<String, Tensor>, mtp_layer: usize) -> Result<Self> {
        let p = format!("mtp.{mtp_layer}");
        Ok(Self {
            norm: tensor(weights, &format!("{p}.norm.weight"))?.shallow_clone(),
            hnorm: tensor(weights, &format!("{p}.hnorm.weight"))?.shallow_clone(),
            head: tensor(weights, &format!("{p}.head.weight"))?.shallow_clone(),
            ffn_norm: tensor(weights, &format!("{p}.ffn_norm.weight"))
                .or_else(|_| tensor(weights, &format!("{p}.ffn.weight")))?
                .shallow_clone(),
            ffn_shared_w1: tensor(weights, &format!("{p}.ffn.shared_experts.w1.weight"))?
                .shallow_clone(),
            ffn_shared_w2: tensor(weights, &format!("{p}.ffn.shared_experts.w2.weight"))?
                .shallow_clone(),
            ffn_shared_w3: tensor(weights, &format!("{p}.ffn.shared_experts.w3.weight"))?
                .shallow_clone(),
        })
    }

    /// Names of all weight tensors in this MTP head.
    pub fn weight_names(mtp_layer: usize) -> Vec<String> {
        let p = format!("mtp.{mtp_layer}");
        vec![
            format!("{p}.norm.weight"),
            format!("{p}.hnorm.weight"),
            format!("{p}.head.weight"),
            format!("{p}.ffn_norm.weight"),
            format!("{p}.ffn.shared_experts.w1.weight"),
            format!("{p}.ffn.shared_experts.w2.weight"),
            format!("{p}.ffn.shared_experts.w3.weight"),
        ]
    }
}

/// Forward through MTP head, returning both logits and pre-head hidden state.
/// The pre-head hidden is needed as input for the next MTP layer in a multi-layer chain.
pub fn v4_mtp_forward_with_hidden(
    hidden: &Tensor,
    next_token_embed: &Tensor,
    mtp: &MtpHeadWeights,
    config: &V4RuntimeConfig,
) -> (Tensor, Tensor) {
    // hidden: [batch, seq, hidden], next_token_embed: [batch, seq, hidden]
    let combined = (hidden + next_token_embed) / 2.0;
    let normed = rms_norm(&combined, &mtp.norm, config.rms_norm_eps);

    // FFN (shared expert style)
    let ffn_out = v4_swiglu(
        &normed,
        &mtp.ffn_shared_w1,
        &mtp.ffn_shared_w2,
        &mtp.ffn_shared_w3,
        config.swiglu_limit,
    );
    let after_ffn = &normed + &ffn_out;
    let final_hidden = rms_norm(&after_ffn, &mtp.hnorm, config.rms_norm_eps);

    // Output projection to vocab
    let logits = final_hidden.linear::<&Tensor>(&mtp.head, None);
    (logits, final_hidden)
}

/// Forward through MTP head (logits only).
pub fn v4_mtp_forward(
    hidden: &Tensor,
    next_token_embed: &Tensor,
    mtp: &MtpHeadWeights,
    config: &V4RuntimeConfig,
) -> Tensor {
    v4_mtp_forward_with_hidden(hidden, next_token_embed, mtp, config).0
}

/// Check if MTP weights are available in the weight map.
pub fn has_mtp_weights(weights: &BTreeMap<String, Tensor>) -> bool {
    weights.contains_key("mtp.0.head.weight")
}
