use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Reduction, Tensor, no_grad};
use tracing::info;

use rustrain_checkpoint::io::*;
use rustrain_checkpoint::manifest::*;
use rustrain_checkpoint::safetensors::{read_safetensors_dir, read_safetensors_map, tensor};
use rustrain_core::runtime::{Config, LrScheduler, RunPaths, load_config};
use rustrain_nccl::nccl as nccl_smoke;
use rustrain_train::metrics::{gpu_memory_allocated_mb, memory_rss_mb};

// ── Config ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DeepSeekRuntimeConfig {
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
}

impl DeepSeekRuntimeConfig {
    pub fn is_moe_layer(&self, layer: usize) -> bool {
        layer >= self.first_k_dense_replace
    }
}

#[derive(Debug, Deserialize)]
struct DeepSeekModelConfig {
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
}

pub fn read_deepseek_config(path: &Path) -> Result<DeepSeekRuntimeConfig> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let c: DeepSeekModelConfig = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(DeepSeekRuntimeConfig {
        num_hidden_layers: c.num_hidden_layers,
        num_attention_heads: c.num_attention_heads,
        hidden_size: c.hidden_size,
        kv_lora_rank: c.kv_lora_rank.unwrap_or(512),
        q_lora_rank: c.q_lora_rank.unwrap_or(1536),
        qk_nope_head_dim: c.qk_nope_head_dim.unwrap_or(128),
        qk_rope_head_dim: c.qk_rope_head_dim.unwrap_or(64),
        v_head_dim: c.v_head_dim.unwrap_or(128),
        rope_theta: c.rope_theta.unwrap_or(10000.0),
        rms_norm_eps: c.rms_norm_eps.unwrap_or(1e-6),
        tie_word_embeddings: c.tie_word_embeddings,
        first_k_dense_replace: c.first_k_dense_replace.unwrap_or(3),
        n_routed_experts: c.n_routed_experts.unwrap_or(0),
        num_experts_per_tok: c.num_experts_per_tok.unwrap_or(0),
        n_shared_experts: c.n_shared_experts.unwrap_or(0),
        moe_intermediate_size: c.moe_intermediate_size.unwrap_or(0),
        intermediate_size: c.intermediate_size.unwrap_or(18432),
        vocab_size: c.vocab_size,
    })
}

// ── Compute dtype ──────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeepSeekComputeDType {
    Fp32,
    Bf16,
}

impl DeepSeekComputeDType {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "fp32" => Ok(Self::Fp32),
            "bf16" => Ok(Self::Bf16),
            other => bail!("unsupported DeepSeek compute dtype {other}"),
        }
    }
    pub fn kind(self) -> Kind {
        match self {
            Self::Fp32 => Kind::Float,
            Self::Bf16 => Kind::BFloat16,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
        }
    }
}

// ── Model path resolution ──────────────────────────────────────

pub fn deepseek_model_path_is_complete(model_path: &Path) -> bool {
    model_path.join("config.json").exists()
        && model_path.join("tokenizer.json").exists()
        && (model_path.join("model.safetensors").exists()
            || model_path.join("model.safetensors.index.json").exists())
}

pub fn resolve_deepseek_model_path(model_path: &Path) -> Result<PathBuf> {
    if deepseek_model_path_is_complete(model_path) {
        return Ok(model_path.to_path_buf());
    }
    let Some(model_dir_name) = model_path.file_name().and_then(|name| name.to_str()) else {
        bail!(
            "DeepSeek model path {} has no directory name",
            model_path.display()
        );
    };
    let Some(root) = model_path.parent() else {
        bail!("DeepSeek model path {} has no parent", model_path.display());
    };
    let hub_root = root.join("hub");
    let hub_suffix = format!("--{model_dir_name}");
    let hub_model_dirs = fs::read_dir(&hub_root)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("models--") && name.ends_with(&hub_suffix))
        })
        .collect::<Vec<_>>();
    if hub_model_dirs.is_empty() {
        bail!(
            "DeepSeek model path {} is missing config/tokenizer/model files and no matching HF hub cache entry was found under {}",
            model_path.display(),
            hub_root.display()
        );
    }
    let mut candidates: Vec<PathBuf> = hub_model_dirs
        .into_iter()
        .flat_map(|hub_dir| {
            let snapshots = hub_dir.join("snapshots");
            fs::read_dir(&snapshots)
                .ok()
                .into_iter()
                .flat_map(|entries| entries.filter_map(Result::ok))
                .map(|entry| entry.path())
                .filter(|path| deepseek_model_path_is_complete(path))
                .collect::<Vec<_>>()
        })
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .rev()
        .next()
        .ok_or_else(|| anyhow!("no complete HF hub snapshot found"))
}

// ── Core ops ───────────────────────────────────────────────────

pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let variance = input
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    input * (variance + eps).rsqrt() * weight
}

pub fn deepseek_mlp(
    input: &Tensor,
    gate_proj: &Tensor,
    up_proj: &Tensor,
    down_proj: &Tensor,
) -> Tensor {
    let gate = input.linear::<&Tensor>(gate_proj, None);
    let up = input.linear::<&Tensor>(up_proj, None);
    (gate.silu() * up).linear::<&Tensor>(down_proj, None)
}

// ── RoPE ──────────────────────────────────────────────────────

pub fn rope_cos_sin(seq_len: usize, head_dim: i64, theta: f64, device: Device) -> (Tensor, Tensor) {
    let positions = Tensor::arange(seq_len as i64, (Kind::Float, device));
    let dim_indices = Tensor::arange(head_dim / 2, (Kind::Float, device));
    let inv_freq = (dim_indices * (2.0 / head_dim as f64)) * (1.0 / theta.ln());
    let inv_freq = inv_freq.exp();
    let freqs = positions.outer(&inv_freq); // [seq_len, head_dim/2]
    let cos = freqs.cos();
    let sin = freqs.sin();
    // Interleave: [seq_len, head_dim]
    let cos = Tensor::cat(&[&cos, &cos], -1);
    let sin = Tensor::cat(&[&sin, &sin], -1);
    (cos, sin)
}

fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    // x: [batch, heads, seq, head_dim]
    let seq_len = x.size()[2];
    let cos = cos.narrow(0, 0, seq_len).unsqueeze(0).unsqueeze(0); // [1, 1, seq, head_dim]
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

// ── MLA Attention ─────────────────────────────────────────────

pub struct DeepSeekAttentionWeights {
    pub q_a_proj: Tensor,
    pub q_a_layernorm: Tensor,
    pub q_b_proj: Tensor,
    pub kv_a_proj_with_mqa: Tensor,
    pub kv_a_layernorm: Tensor,
    pub kv_b_proj: Tensor,
    pub o_proj: Tensor,
}

impl DeepSeekAttentionWeights {
    pub fn load(weights: &BTreeMap<String, Tensor>, layer: usize) -> Result<Self> {
        Self::load_with_kind(weights, layer, Kind::Float)
    }

    pub fn load_with_kind(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        kind: Kind,
    ) -> Result<Self> {
        let p = format!("model.layers.{layer}.self_attn");
        Ok(Self {
            q_a_proj: tensor(weights, &format!("{p}.q_a_proj.weight"))?.to_kind(kind),
            q_a_layernorm: tensor(weights, &format!("{p}.q_a_layernorm.weight"))?.to_kind(kind),
            q_b_proj: tensor(weights, &format!("{p}.q_b_proj.weight"))?.to_kind(kind),
            kv_a_proj_with_mqa: tensor(weights, &format!("{p}.kv_a_proj_with_mqa.weight"))?
                .to_kind(kind),
            kv_a_layernorm: tensor(weights, &format!("{p}.kv_a_layernorm.weight"))?.to_kind(kind),
            kv_b_proj: tensor(weights, &format!("{p}.kv_b_proj.weight"))?.to_kind(kind),
            o_proj: tensor(weights, &format!("{p}.o_proj.weight"))?.to_kind(kind),
        })
    }
}

pub fn deepseek_mla_attention(
    input: &Tensor,
    attn: &DeepSeekAttentionWeights,
    config: &DeepSeekRuntimeConfig,
) -> Tensor {
    let shape = input.size();
    let batch = shape[0];
    let seq = shape[1];
    let hidden = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let qk_nope = config.qk_nope_head_dim;
    let qk_rope = config.qk_rope_head_dim;
    let v_head = config.v_head_dim;
    let kv_lora = config.kv_lora_rank;

    // Q path: input → q_a_proj → rms_norm → q_b_proj → reshape
    let q_a = input.linear::<&Tensor>(&attn.q_a_proj, None); // [batch, seq, q_lora_rank]
    let q_a = rms_norm(&q_a, &attn.q_a_layernorm, config.rms_norm_eps);
    let q_b = q_a.linear::<&Tensor>(&attn.q_b_proj, None); // [batch, seq, num_heads*(qk_nope+qk_rope)]
    let q = q_b
        .reshape([batch, seq, num_heads, qk_nope + qk_rope])
        .transpose(1, 2); // [batch, heads, seq, qk_nope+qk_rope]
    let q_nope = q.narrow(-1, 0, qk_nope);
    let q_rope = q.narrow(-1, qk_nope, qk_rope);

    // KV path: input → kv_a_proj_with_mqa → split → rms_norm → kv_b_proj → reshape
    let kv_a = input.linear::<&Tensor>(&attn.kv_a_proj_with_mqa, None); // [batch, seq, kv_lora_rank+qk_rope]
    let kv_lora_part = kv_a.narrow(-1, 0, kv_lora);
    let k_rope = kv_a.narrow(-1, kv_lora, qk_rope); // [batch, seq, qk_rope]

    let kv_lora_normed = rms_norm(&kv_lora_part, &attn.kv_a_layernorm, config.rms_norm_eps);
    let kv_b = kv_lora_normed.linear::<&Tensor>(&attn.kv_b_proj, None); // [batch, seq, num_heads*(qk_nope+v_head)]
    let kv_b = kv_b.reshape([batch, seq, num_heads, qk_nope + v_head]);
    let k_nope = kv_b.narrow(-1, 0, qk_nope).transpose(1, 2); // [batch, heads, seq, qk_nope]
    let v = kv_b.narrow(-1, qk_nope, v_head).transpose(1, 2); // [batch, heads, seq, v_head]

    // Apply RoPE to q_rope and k_rope
    // k_rope is [batch, seq, qk_rope] -> expand to [batch, num_heads, seq, qk_rope]
    let k_rope_expanded = k_rope
        .unsqueeze(2) // [batch, seq, 1, qk_rope]
        .transpose(1, 2) // [batch, 1, seq, qk_rope]
        .expand([batch, num_heads, seq, qk_rope], false); // [batch, heads, seq, qk_rope]
    let (cos, sin) = rope_cos_sin(seq as usize, qk_rope, config.rope_theta, input.device());
    let cos = cos.to_kind(input.kind());
    let sin = sin.to_kind(input.kind());
    let q_rope_rotated = apply_rotary(&q_rope, &cos, &sin);
    let k_rope_rotated = apply_rotary(&k_rope_expanded, &cos, &sin);

    // Concat nope + rope
    let q_full = Tensor::cat(&[&q_nope, &q_rope_rotated], -1); // [batch, heads, seq, qk_nope+qk_rope]
    let k_full = Tensor::cat(&[&k_nope, &k_rope_rotated], -1); // [batch, heads, seq, qk_nope+qk_rope]

    // Attention
    let scores = q_full.matmul(&k_full.transpose(-2, -1)) / ((qk_nope + qk_rope) as f64).sqrt();
    let causal_mask = Tensor::ones([seq as i64, seq as i64], (Kind::Bool, input.device())).triu(1);
    let scores = scores.masked_fill(&causal_mask, f64::NEG_INFINITY);
    let probs = scores.softmax(-1, Kind::Float).to_kind(v.kind());
    let context = probs.matmul(&v); // [batch, heads, seq, v_head]
    let context = context
        .transpose(1, 2)
        .reshape([batch, seq, num_heads * v_head]);

    context.linear::<&Tensor>(&attn.o_proj, None)
}

// ── Dense Layer ───────────────────────────────────────────────

pub struct DeepSeekDenseLayerWeights {
    pub input_norm: Tensor,
    pub attn: DeepSeekAttentionWeights,
    pub post_attention_norm: Tensor,
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
}

impl DeepSeekDenseLayerWeights {
    pub fn load_with_kind(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        kind: Kind,
    ) -> Result<Self> {
        let p = format!("model.layers.{layer}");
        Ok(Self {
            input_norm: tensor(weights, &format!("{p}.input_layernorm.weight"))?.to_kind(kind),
            attn: DeepSeekAttentionWeights::load_with_kind(weights, layer, kind)?,
            post_attention_norm: tensor(weights, &format!("{p}.post_attention_layernorm.weight"))?
                .to_kind(kind),
            gate_proj: tensor(weights, &format!("{p}.mlp.gate_proj.weight"))?.to_kind(kind),
            up_proj: tensor(weights, &format!("{p}.mlp.up_proj.weight"))?.to_kind(kind),
            down_proj: tensor(weights, &format!("{p}.mlp.down_proj.weight"))?.to_kind(kind),
        })
    }
}

pub fn deepseek_layer(
    input: &Tensor,
    weights: &DeepSeekDenseLayerWeights,
    config: &DeepSeekRuntimeConfig,
) -> Tensor {
    let hidden = rms_norm(input, &weights.input_norm, config.rms_norm_eps);
    let attn = deepseek_mla_attention(&hidden, &weights.attn, config);
    let residual = input + &attn;
    let mlp_input = rms_norm(&residual, &weights.post_attention_norm, config.rms_norm_eps);
    let mlp = deepseek_mlp(
        &mlp_input,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
    );
    residual + mlp
}

// ── MoE Layer ─────────────────────────────────────────────────

pub struct DeepSeekMoeLayerWeights {
    pub input_norm: Tensor,
    pub attn: DeepSeekAttentionWeights,
    pub post_attention_norm: Tensor,
    pub gate: Tensor, // router: [n_routed_experts, hidden_size]
    pub shared_gate_proj: Tensor,
    pub shared_up_proj: Tensor,
    pub shared_down_proj: Tensor,
    pub experts: Vec<(Tensor, Tensor, Tensor)>, // [(gate, up, down)] per expert
}

impl DeepSeekMoeLayerWeights {
    pub fn load_with_kind(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        n_experts: usize,
        kind: Kind,
    ) -> Result<Self> {
        let p = format!("model.layers.{layer}");
        let shared_prefix = format!("{p}.mlp.shared_experts");
        let mut experts = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let ep = format!("{p}.mlp.experts.{e}");
            let gate = tensor(weights, &format!("{ep}.gate_proj.weight"))?.to_kind(kind);
            let up = tensor(weights, &format!("{ep}.up_proj.weight"))?.to_kind(kind);
            let down = tensor(weights, &format!("{ep}.down_proj.weight"))?.to_kind(kind);
            experts.push((gate, up, down));
        }
        Ok(Self {
            input_norm: tensor(weights, &format!("{p}.input_layernorm.weight"))?.to_kind(kind),
            attn: DeepSeekAttentionWeights::load_with_kind(weights, layer, kind)?,
            post_attention_norm: tensor(weights, &format!("{p}.post_attention_layernorm.weight"))?
                .to_kind(kind),
            gate: tensor(weights, &format!("{p}.mlp.gate.weight"))?.to_kind(kind),
            shared_gate_proj: tensor(weights, &format!("{shared_prefix}.gate_proj.weight"))?
                .to_kind(kind),
            shared_up_proj: tensor(weights, &format!("{shared_prefix}.up_proj.weight"))?
                .to_kind(kind),
            shared_down_proj: tensor(weights, &format!("{shared_prefix}.down_proj.weight"))?
                .to_kind(kind),
            experts,
        })
    }
}

pub fn deepseek_moe_mlp(
    input: &Tensor,
    gate: &Tensor,
    shared_gate: &Tensor,
    shared_up: &Tensor,
    shared_down: &Tensor,
    experts: &[(Tensor, Tensor, Tensor)],
    num_experts_per_tok: usize,
) -> Tensor {
    // Shared expert (always computed)
    let shared_output = deepseek_mlp(input, shared_gate, shared_up, shared_down);

    // Router logits
    let router_logits = input.linear::<&Tensor>(gate, None); // [batch, seq, n_experts]
    let n_experts = experts.len() as i64;

    // Top-k selection with softmax over selected experts
    let (topk_weights, topk_indices) =
        router_logits.topk(num_experts_per_tok as i64, -1, true, true);
    // Softmax over selected
    let topk_weights = topk_weights.softmax(-1, Kind::Float);

    // Accumulate expert outputs
    let mut output = shared_output;

    for k in 0..num_experts_per_tok {
        let expert_indices = topk_indices.select(-1, k as i64); // [batch, seq]
        let expert_weights = topk_weights.select(-1, k as i64); // [batch, seq]
        let weights_kind = expert_weights.kind();

        for (expert_idx, (gate_p, up_p, down_p)) in experts.iter().enumerate() {
            let mask = expert_indices.eq(expert_idx as i64).to_kind(weights_kind);
            let mask_sum = mask.sum(Kind::Float).double_value(&[]);
            if mask_sum > 0.0 {
                let expert_out = deepseek_mlp(input, gate_p, up_p, down_p);
                let weight = (&expert_weights * &mask).unsqueeze(-1);
                output = output + (expert_out * weight);
            }
        }
    }

    output
}

pub fn deepseek_moe_layer(
    input: &Tensor,
    weights: &DeepSeekMoeLayerWeights,
    config: &DeepSeekRuntimeConfig,
) -> Tensor {
    let hidden = rms_norm(input, &weights.input_norm, config.rms_norm_eps);
    let attn = deepseek_mla_attention(&hidden, &weights.attn, config);
    let residual = input + &attn;
    let mlp_input = rms_norm(&residual, &weights.post_attention_norm, config.rms_norm_eps);
    let mlp = deepseek_moe_mlp(
        &mlp_input,
        &weights.gate,
        &weights.shared_gate_proj,
        &weights.shared_up_proj,
        &weights.shared_down_proj,
        &weights.experts,
        config.num_experts_per_tok,
    );
    residual + mlp
}

// ── Forward ───────────────────────────────────────────────────

pub fn deepseek_forward_from_ids(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &DeepSeekRuntimeConfig,
) -> Result<Tensor> {
    deepseek_forward_from_ids_with_kind(input_ids, weights, config, Kind::Float)
}

pub fn deepseek_forward_from_ids_with_kind(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &DeepSeekRuntimeConfig,
    compute_kind: Kind,
) -> Result<Tensor> {
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?.to_kind(compute_kind);
    let final_norm = tensor(weights, "model.norm.weight")?.to_kind(compute_kind);
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);
    for layer in 0..config.num_hidden_layers {
        if config.is_moe_layer(layer) {
            let lw = DeepSeekMoeLayerWeights::load_with_kind(
                weights,
                layer,
                config.n_routed_experts,
                compute_kind,
            )?;
            hidden = deepseek_moe_layer(&hidden, &lw, config);
        } else {
            let lw = DeepSeekDenseLayerWeights::load_with_kind(weights, layer, compute_kind)?;
            hidden = deepseek_layer(&hidden, &lw, config);
        }
    }
    let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps).to_kind(compute_kind);
    let lm_head = if config.tie_word_embeddings {
        embed_tokens.shallow_clone()
    } else {
        tensor(weights, "lm_head.weight")?.to_kind(compute_kind)
    };
    Ok(hidden.linear::<&Tensor>(&lm_head, None))
}

pub fn deepseek_causal_lm_loss(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &DeepSeekRuntimeConfig,
) -> Result<Tensor> {
    deepseek_causal_lm_loss_with_kind(input_ids, weights, config, Kind::Float)
}

pub fn deepseek_causal_lm_loss_with_kind(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &DeepSeekRuntimeConfig,
    compute_kind: Kind,
) -> Result<Tensor> {
    let logits = deepseek_forward_from_ids_with_kind(input_ids, weights, config, compute_kind)?;
    let shifted = logits.narrow(1, 0, logits.size()[1] - 1);
    let targets = input_ids.narrow(1, 1, input_ids.size()[1] - 1);
    let vocab_size = config.vocab_size;
    Ok(shifted
        .reshape([-1, vocab_size])
        .log_softmax(-1, Kind::Float)
        .g_nll_loss::<&Tensor>(&targets.reshape([-1]), None, Reduction::Mean, -100))
}

// ── Trainable tensors ────────────────────────────────────────

pub fn deepseek_trainable_tensors_for_layer(
    layer: usize,
    config: &DeepSeekRuntimeConfig,
) -> Vec<String> {
    let p = format!("model.layers.{layer}");
    let mut names = vec![
        format!("{p}.input_layernorm.weight"),
        format!("{p}.self_attn.q_a_proj.weight"),
        format!("{p}.self_attn.q_a_layernorm.weight"),
        format!("{p}.self_attn.q_b_proj.weight"),
        format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
        format!("{p}.self_attn.kv_a_layernorm.weight"),
        format!("{p}.self_attn.kv_b_proj.weight"),
        format!("{p}.self_attn.o_proj.weight"),
        format!("{p}.post_attention_layernorm.weight"),
    ];
    if config.is_moe_layer(layer) {
        names.push(format!("{p}.mlp.gate.weight"));
        names.push(format!("{p}.mlp.shared_experts.gate_proj.weight"));
        names.push(format!("{p}.mlp.shared_experts.up_proj.weight"));
        names.push(format!("{p}.mlp.shared_experts.down_proj.weight"));
        for e in 0..config.n_routed_experts {
            names.push(format!("{p}.mlp.experts.{e}.gate_proj.weight"));
            names.push(format!("{p}.mlp.experts.{e}.up_proj.weight"));
            names.push(format!("{p}.mlp.experts.{e}.down_proj.weight"));
        }
    } else {
        names.push(format!("{p}.mlp.gate_proj.weight"));
        names.push(format!("{p}.mlp.up_proj.weight"));
        names.push(format!("{p}.mlp.down_proj.weight"));
    }
    names
}

pub fn deepseek_trainable_tensors(
    trainable_layers: &[usize],
    config: &DeepSeekRuntimeConfig,
    include_lm_head: bool,
) -> Vec<String> {
    let mut names = Vec::new();
    for &layer in trainable_layers {
        names.extend(deepseek_trainable_tensors_for_layer(layer, config));
    }
    names.push("model.norm.weight".to_string());
    if include_lm_head {
        names.push("lm_head.weight".to_string());
    }
    names
}

/// Load only specific tensors from a multi-file safetensors model.
/// Uses the index.json to find which shard files contain the needed tensors.
pub fn read_deepseek_tensors(
    model_dir: &Path,
    needed_names: &std::collections::HashSet<String>,
) -> Result<BTreeMap<String, Tensor>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        let tensors = Tensor::read_safetensors(&single)
            .with_context(|| format!("failed to read {}", single.display()))?;
        let mut result = BTreeMap::new();
        for (name, tensor) in tensors {
            if needed_names.contains(&name) {
                result.insert(name, tensor);
            }
        }
        return Ok(result);
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        bail!(
            "no model.safetensors or model.safetensors.index.json in {}",
            model_dir.display()
        );
    }
    let index_text = fs::read_to_string(&index_path)?;
    #[derive(Deserialize)]
    struct Index {
        weight_map: std::collections::HashMap<String, String>,
    }
    let index: Index = serde_json::from_str(&index_text)?;

    // Find which shard files contain our needed tensors
    let mut shard_files: std::collections::HashSet<String> = std::collections::HashSet::new();
    for name in needed_names {
        if let Some(shard) = index.weight_map.get(name) {
            shard_files.insert(shard.clone());
        }
    }

    info!(
        shards = shard_files.len(),
        needed = needed_names.len(),
        "loading selective shards"
    );

    let mut weights = BTreeMap::new();
    for shard_file in &shard_files {
        let shard_path = model_dir.join(shard_file);
        let shard_tensors = Tensor::read_safetensors(&shard_path)
            .with_context(|| format!("failed to read {}", shard_path.display()))?;
        for (name, tensor) in shard_tensors {
            if needed_names.contains(&name) {
                weights.insert(name, tensor);
            }
        }
    }
    Ok(weights)
}

/// Forward through only the specified layers (for training verification).
/// Skips layers not present in the weights map.
pub fn deepseek_forward_selective(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &DeepSeekRuntimeConfig,
    trainable_layers: &[usize],
) -> Result<Tensor> {
    let compute_kind = Kind::Float;
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?.to_kind(compute_kind);
    let final_norm = tensor(weights, "model.norm.weight")?.to_kind(compute_kind);
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);

    for layer in 0..config.num_hidden_layers {
        if !trainable_layers.contains(&layer) {
            continue;
        }
        if config.is_moe_layer(layer) {
            let lw = DeepSeekMoeLayerWeights::load_with_kind(
                weights,
                layer,
                config.n_routed_experts,
                compute_kind,
            )?;
            hidden = deepseek_moe_layer(&hidden, &lw, config);
        } else {
            let lw = DeepSeekDenseLayerWeights::load_with_kind(weights, layer, compute_kind)?;
            hidden = deepseek_layer(&hidden, &lw, config);
        }
    }

    let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps).to_kind(compute_kind);
    let lm_head = if config.tie_word_embeddings {
        embed_tokens.shallow_clone()
    } else {
        tensor(weights, "lm_head.weight")?.to_kind(compute_kind)
    };
    Ok(hidden.linear::<&Tensor>(&lm_head, None))
}

pub fn deepseek_causal_lm_loss_selective(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &DeepSeekRuntimeConfig,
    trainable_layers: &[usize],
) -> Result<Tensor> {
    let logits = deepseek_forward_selective(input_ids, weights, config, trainable_layers)?;
    let shifted = logits.narrow(1, 0, logits.size()[1] - 1);
    let targets = input_ids.narrow(1, 1, input_ids.size()[1] - 1);
    Ok(shifted
        .reshape([-1, config.vocab_size])
        .log_softmax(-1, Kind::Float)
        .g_nll_loss::<&Tensor>(&targets.reshape([-1]), None, Reduction::Mean, -100))
}
