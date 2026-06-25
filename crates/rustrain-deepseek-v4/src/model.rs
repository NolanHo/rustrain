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
    let variance = input
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    let result = input * (variance + eps).rsqrt() * weight;
    result.to_kind(input.kind())
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

pub fn rope_cos_sin(seq_len: usize, head_dim: i64, theta: f64, device: Device) -> (Tensor, Tensor) {
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
}

impl V4AttentionWeights {
    pub fn load_raw(weights: &BTreeMap<String, Tensor>, layer: usize) -> Result<Self> {
        let p = format!("layers.{layer}.attn");
        Ok(Self {
            wq_a: tensor(weights, &format!("{p}.wq_a.weight"))?.shallow_clone(),
            wq_b: tensor(weights, &format!("{p}.wq_b.weight"))?.shallow_clone(),
            wkv: tensor(weights, &format!("{p}.wkv.weight"))?.shallow_clone(),
            wo_a: tensor(weights, &format!("{p}.wo_a.weight"))?.shallow_clone(),
            wo_b: tensor(weights, &format!("{p}.wo_b.weight"))?.shallow_clone(),
            q_norm: tensor(weights, &format!("{p}.q_norm.weight"))?.shallow_clone(),
            kv_norm: tensor(weights, &format!("{p}.kv_norm.weight"))?.shallow_clone(),
            attn_sink: tensor(weights, &format!("{p}.attn_sink"))?.shallow_clone(),
        })
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
    let q_a = input.linear::<&Tensor>(&attn.wq_a, None); // [batch, seq, 1024]
    let q_a = rms_norm(&q_a, &attn.q_norm, config.rms_norm_eps);
    let q_b = q_a.linear::<&Tensor>(&attn.wq_b, None); // [batch, seq, 32768]
    let q = q_b
        .reshape([batch, seq, num_heads, head_dim])
        .transpose(1, 2); // [batch, heads, seq, 512]
    let q_nope = q.narrow(-1, 0, qk_nope); // [batch, heads, seq, 448]
    let q_rope = q.narrow(-1, qk_nope, qk_rope); // [batch, heads, seq, 64]

    // KV path: wkv → kv_norm (MQA, shared across heads)
    let wkv_out = input.linear::<&Tensor>(&attn.wkv, None); // [batch, seq, 512]
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

    // Apply RoPE to the rope portions
    let (cos, sin) = rope_cos_sin(seq as usize, qk_rope, config.rope_theta, input.device());
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
    let causal_mask = Tensor::ones([seq as i64, seq as i64], (Kind::Bool, input.device())).triu(1);
    let scores = scores.masked_fill(&causal_mask, f64::NEG_INFINITY);
    let probs = scores.softmax(-1, Kind::Float).to_kind(v.kind());
    let context = probs.matmul(&v); // [batch, heads, seq, 512]

    // Group reduction: [batch, heads, seq, 512] → [batch, o_groups, seq, hidden/o_groups]
    let heads_per_group = num_heads / o_groups;
    let context = context
        .reshape([batch, o_groups, heads_per_group, seq, head_dim])
        .sum_dim_intlist([2].as_slice(), false, Kind::Float);
    let context = context.reshape([batch, seq, o_groups * head_dim]); // [batch, seq, 4096]

    // Output: wo_a → wo_b (LoRA-style two-layer projection)
    let o_compressed = context.linear::<&Tensor>(&attn.wo_a, None); // [batch, seq, 8192]
    o_compressed.linear::<&Tensor>(&attn.wo_b, None) // [batch, seq, 4096]
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

/// V4 MoE MLP with sqrtsoftplus routing + SwiGLU with limit
pub fn v4_moe_mlp(
    input: &Tensor,
    gate: &Tensor,
    shared: &(Tensor, Tensor, Tensor),
    experts: &[(Tensor, Tensor, Tensor)],
    num_experts_per_tok: usize,
    scoring_func: &str,
    routed_scaling_factor: f64,
    swiglu_limit: f64,
) -> Tensor {
    // Shared expert (always computed)
    let shared_out = v4_swiglu(input, &shared.0, &shared.1, &shared.2, swiglu_limit);

    // Router logits
    let router_logits = input.linear::<&Tensor>(gate, None); // [batch, seq, n_experts]

    // Scoring function: sqrtsoftplus (V4 specific)
    let scores = if scoring_func == "sqrtsoftplus" {
        let sp = (router_logits.shallow_clone().exp() + 1.0).log();
        sp.sqrt()
    } else {
        router_logits.shallow_clone().softmax(-1, Kind::Float)
    };
    let _ = &router_logits;

    // Top-k selection
    let (topk_weights, topk_indices) = scores.topk(num_experts_per_tok as i64, -1, true, true);

    // Normalize top-k weights
    let denom = topk_weights
        .sum_dim_intlist([-1].as_slice(), true, Kind::Float)
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
        config.num_experts_per_tok,
        &config.scoring_func,
        config.routed_scaling_factor,
        config.swiglu_limit,
    );
    residual + mlp
}

// ── Forward ───────────────────────────────────────────────────

pub fn v4_forward_selective(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &V4RuntimeConfig,
    trainable_layers: &[usize],
) -> Result<Tensor> {
    let embed_tokens = tensor(weights, "embed.weight")?;
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);

    for layer in 0..config.num_hidden_layers {
        if !trainable_layers.contains(&layer) {
            continue;
        }
        let lw = V4MoeLayerWeights::load_raw(weights, layer, config.n_routed_experts)?;
        hidden = v4_moe_layer(&hidden, &lw, config);
    }

    // Apply final norm + lm_head if available, otherwise use tied embeddings
    if let Ok(final_norm) = tensor(weights, "norm.weight") {
        let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps);
        let lm_head = if let Ok(lh) = tensor(weights, "head.weight") {
            lh.shallow_clone()
        } else {
            embed_tokens.shallow_clone() // tied
        };
        Ok(hidden.linear::<&Tensor>(&lm_head, None))
    } else {
        // Skip final norm + lm_head (for partial weight loading)
        Ok(hidden.linear::<&Tensor>(&embed_tokens, None))
    }
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

/// Load V4 weights, dequantizing FP8→bf16 via Python.
/// V4 uses FP8 for all weights (each has .scale tensor).
pub fn load_v4_weights(
    model_dir: &Path,
    needed: &HashSet<String>,
) -> Result<BTreeMap<String, Tensor>> {
    let script = format!(
        r#"
import json, sys, torch, os
from safetensors import safe_open
from safetensors.torch import save_file

model_dir = {dir:?}
needed = set(sys.argv[1:])
out = "/tmp/v4_bf16_converted.safetensors"

idx = os.path.join(model_dir, "model.safetensors.index.json")
single = os.path.join(model_dir, "model.safetensors")

tensors = {{}}
if os.path.exists(single):
    with safe_open(single, framework="pt") as f:
        for k in f.keys():
            if k in needed:
                t = f.get_tensor(k)
                if t.dtype == torch.float8_e4m3fn:
                    scale_key = k.replace(".weight", ".scale")
                    if scale_key in f.keys():
                        scale = f.get_tensor(scale_key).to(torch.float32)
                        # Block-wise dequant: 128x128 blocks
                        out_dim, in_dim = t.shape
                        block_out, block_in = scale.shape
                        bs_out, bs_in = out_dim // block_out, in_dim // block_in
                        t = t.to(torch.float32).reshape(block_out, bs_out, block_in, bs_in)
                        t = t * scale.reshape(block_out, 1, block_in, 1)
                        t = t.reshape(out_dim, in_dim).to(torch.bfloat16)
                    else:
                        t = t.to(torch.bfloat16)
                tensors[k] = t.cpu()
elif os.path.exists(idx):
    with open(idx) as f: wm = json.load(f)["weight_map"]
    shards = set(wm[n] for n in needed if n in wm)
    for s in sorted(shards):
        with safe_open(os.path.join(model_dir, s), framework="pt") as f:
            for k in f.keys():
                if k in needed:
                    t = f.get_tensor(k)
                    if t.dtype == torch.float8_e4m3fn: t = t.to(torch.bfloat16)
                    tensors[k] = t.cpu()
else:
    sys.exit(1)

save_file(tensors, out)
print(out)
"#,
        dir = model_dir.display()
    );

    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(&script)
        .args(needed.iter())
        .output()?;

    if !output.status.success() {
        bail!(
            "V4 FP8 conversion failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    info!(path = %path, "V4 FP8→bf16 conversion complete");

    let tensors = Tensor::read_safetensors(Path::new(&path))?;
    Ok(tensors.into_iter().collect())
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

    for layer in 0..config.num_hidden_layers {
        if !trainable_layers.contains(&layer) {
            continue;
        }
        let lw = V4MoeLayerWeights::load_raw(weights, layer, config.n_routed_experts)?;
        let hidden_norm = rms_norm(&hidden, &lw.attn_norm, config.rms_norm_eps);
        let attn = v4_lora_attention_weights(&lw.attn, layer, registry);
        let attn_out = v4_attention(&hidden_norm, &attn, config);
        let residual = &hidden + &attn_out;
        let mlp_input = rms_norm(&residual, &lw.ffn_norm, config.rms_norm_eps);
        let mlp = v4_moe_mlp(
            &mlp_input,
            &lw.gate,
            &lw.shared_experts,
            &lw.experts,
            config.num_experts_per_tok,
            &config.scoring_func,
            config.routed_scaling_factor,
            config.swiglu_limit,
        );
        hidden = residual + mlp;
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

fn v4_lora_attention_weights(
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
