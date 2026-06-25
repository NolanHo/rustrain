//! DeepSeek TP (Tensor Parallel) support.
//!
//! MLA TP strategy:
//! - q_a_proj: NOT split (compresses to shared low-rank space)
//! - q_b_proj: column parallel (split by heads)
//! - kv_a_proj_with_mqa: NOT split (MQA shared)
//! - kv_b_proj: column parallel (split by heads)
//! - o_proj: row parallel (split by heads, all-reduce output)
//! - MLP gate/up: column parallel, down: row parallel (all-reduce)

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use tch::{Device, Kind, Tensor, no_grad};
use tracing::info;

use rustrain_checkpoint::safetensors::{read_safetensors_dir, tensor};
use rustrain_nccl::nccl as nccl_smoke;

use crate::model::*;

/// Parse env var as usize (set by launcher).
fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set; run through rustrain launch"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

/// TP shard config for MLA attention.
pub struct MlaTpShard {
    pub rank: usize,
    pub world_size: usize,
    pub heads_per_rank: i64,
    pub head_start: i64,
}

impl MlaTpShard {
    pub fn new(rank: usize, world_size: usize, num_heads: i64) -> Self {
        let heads_per_rank = num_heads / world_size as i64;
        Self {
            rank,
            world_size,
            heads_per_rank,
            head_start: rank as i64 * heads_per_rank,
        }
    }
}

/// TP-sharded MLA attention weights.
/// Only q_b_proj and kv_b_proj are sharded (by heads).
/// q_a_proj, kv_a_proj_with_mqa, layernorms are replicated (shared).
pub struct TpAttentionWeights {
    pub q_a_proj: Tensor,           // replicated
    pub q_a_layernorm: Tensor,      // replicated
    pub q_b_proj: Tensor,           // sharded: [heads_per_rank*(qk_nope+qk_rope), q_lora_rank]
    pub kv_a_proj_with_mqa: Tensor, // replicated
    pub kv_a_layernorm: Tensor,     // replicated
    pub kv_b_proj: Tensor,          // sharded: [heads_per_rank*(qk_nope+v_head), kv_lora_rank]
    pub o_proj: Tensor,             // sharded: [hidden, heads_per_rank*v_head]
}

impl TpAttentionWeights {
    pub fn load_sharded(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        shard: &MlaTpShard,
        config: &DeepSeekRuntimeConfig,
        kind: Kind,
    ) -> Result<Self> {
        let p = format!("model.layers.{layer}.self_attn");
        let num_heads = config.num_attention_heads;
        let qk_nope = config.qk_nope_head_dim;
        let qk_rope = config.qk_rope_head_dim;
        let v_head = config.v_head_dim;

        // q_b_proj: [num_heads*(qk_nope+qk_rope), q_lora_rank]
        // Split by heads: each rank gets heads_per_rank rows
        let q_b_full = tensor(weights, &format!("{p}.q_b_proj.weight"))?.to_kind(kind);
        let q_b_row_start = shard.head_start * (qk_nope + qk_rope);
        let q_b_row_size = shard.heads_per_rank * (qk_nope + qk_rope);
        let q_b_proj = q_b_full
            .narrow(0, q_b_row_start, q_b_row_size)
            .shallow_clone();

        // kv_b_proj: [num_heads*(qk_nope+v_head), kv_lora_rank]
        // Split by heads
        let kv_b_full = tensor(weights, &format!("{p}.kv_b_proj.weight"))?.to_kind(kind);
        let kv_b_row_start = shard.head_start * (qk_nope + v_head);
        let kv_b_row_size = shard.heads_per_rank * (qk_nope + v_head);
        let kv_b_proj = kv_b_full
            .narrow(0, kv_b_row_start, kv_b_row_size)
            .shallow_clone();

        // o_proj: [hidden, num_heads*v_head]
        // Row parallel: split by input dimension (heads)
        let o_full = tensor(weights, &format!("{p}.o_proj.weight"))?.to_kind(kind);
        let o_col_start = shard.head_start * v_head;
        let o_col_size = shard.heads_per_rank * v_head;
        let o_proj = o_full.narrow(1, o_col_start, o_col_size).shallow_clone();

        Ok(Self {
            q_a_proj: tensor(weights, &format!("{p}.q_a_proj.weight"))?.to_kind(kind),
            q_a_layernorm: tensor(weights, &format!("{p}.q_a_layernorm.weight"))?.to_kind(kind),
            q_b_proj,
            kv_a_proj_with_mqa: tensor(weights, &format!("{p}.kv_a_proj_with_mqa.weight"))?
                .to_kind(kind),
            kv_a_layernorm: tensor(weights, &format!("{p}.kv_a_layernorm.weight"))?.to_kind(kind),
            kv_b_proj,
            o_proj,
        })
    }
}

/// TP-sharded dense layer weights (layer 0-2).
pub struct TpDenseLayerWeights {
    pub input_norm: Tensor,
    pub attn: TpAttentionWeights,
    pub post_attention_norm: Tensor,
    pub gate_proj: Tensor, // column parallel
    pub up_proj: Tensor,   // column parallel
    pub down_proj: Tensor, // row parallel
}

impl TpDenseLayerWeights {
    pub fn load_sharded(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        shard: &MlaTpShard,
        config: &DeepSeekRuntimeConfig,
        kind: Kind,
    ) -> Result<Self> {
        let p = format!("model.layers.{layer}");
        let intermediate = config.intermediate_size as i64;

        // MLP gate_proj: [intermediate, hidden] - column parallel (split rows)
        let gate_full = tensor(weights, &format!("{p}.mlp.gate_proj.weight"))?.to_kind(kind);
        let inter_per_rank = intermediate / shard.world_size as i64;
        let gate_proj = gate_full
            .narrow(0, shard.rank as i64 * inter_per_rank, inter_per_rank)
            .shallow_clone();

        let up_full = tensor(weights, &format!("{p}.mlp.up_proj.weight"))?.to_kind(kind);
        let up_proj = up_full
            .narrow(0, shard.rank as i64 * inter_per_rank, inter_per_rank)
            .shallow_clone();

        // down_proj: [hidden, intermediate] - row parallel (split cols)
        let down_full = tensor(weights, &format!("{p}.mlp.down_proj.weight"))?.to_kind(kind);
        let down_proj = down_full
            .narrow(1, shard.rank as i64 * inter_per_rank, inter_per_rank)
            .shallow_clone();

        Ok(Self {
            input_norm: tensor(weights, &format!("{p}.input_layernorm.weight"))?.to_kind(kind),
            attn: TpAttentionWeights::load_sharded(weights, layer, shard, config, kind)?,
            post_attention_norm: tensor(weights, &format!("{p}.post_attention_layernorm.weight"))?
                .to_kind(kind),
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

/// TP forward for MLA attention (sharded version).
/// Each rank computes its head subset. o_proj output needs all-reduce.
pub fn tp_mla_attention(
    input: &Tensor,
    attn: &TpAttentionWeights,
    config: &DeepSeekRuntimeConfig,
    shard: &MlaTpShard,
) -> Tensor {
    let shape = input.size();
    let batch = shape[0];
    let seq = shape[1];
    let heads = shard.heads_per_rank;
    let qk_nope = config.qk_nope_head_dim;
    let qk_rope = config.qk_rope_head_dim;
    let v_head = config.v_head_dim;
    let kv_lora = config.kv_lora_rank;

    // Q path (replicated q_a_proj, sharded q_b_proj)
    let q_a = input.linear::<&Tensor>(&attn.q_a_proj, None);
    let q_a = rms_norm(&q_a, &attn.q_a_layernorm, config.rms_norm_eps);
    let q_b = q_a.linear::<&Tensor>(&attn.q_b_proj, None); // [batch, seq, heads_per_rank*(qk_nope+qk_rope)]
    let q = q_b
        .reshape([batch, seq, heads, qk_nope + qk_rope])
        .transpose(1, 2);
    let q_nope = q.narrow(-1, 0, qk_nope);
    let q_rope = q.narrow(-1, qk_nope, qk_rope);

    // KV path (replicated kv_a_proj, sharded kv_b_proj)
    let kv_a = input.linear::<&Tensor>(&attn.kv_a_proj_with_mqa, None);
    let kv_lora_part = kv_a.narrow(-1, 0, kv_lora);
    let k_rope = kv_a.narrow(-1, kv_lora, qk_rope);

    let kv_lora_normed = rms_norm(&kv_lora_part, &attn.kv_a_layernorm, config.rms_norm_eps);
    let kv_b = kv_lora_normed.linear::<&Tensor>(&attn.kv_b_proj, None); // [batch, seq, heads_per_rank*(qk_nope+v_head)]
    let kv_b = kv_b.reshape([batch, seq, heads, qk_nope + v_head]);
    let k_nope = kv_b.narrow(-1, 0, qk_nope).transpose(1, 2);
    let v = kv_b.narrow(-1, qk_nope, v_head).transpose(1, 2);

    // RoPE
    let k_rope_expanded = k_rope
        .unsqueeze(2)
        .transpose(1, 2)
        .expand([batch, heads, seq, qk_rope], false);
    let (cos, sin) = rope_cos_sin(seq as usize, qk_rope, config.rope_theta, input.device());
    let cos = cos.to_kind(input.kind());
    let sin = sin.to_kind(input.kind());
    let q_rope_rotated = apply_rotary(&q_rope, &cos, &sin);
    let k_rope_rotated = apply_rotary(&k_rope_expanded, &cos, &sin);

    let q_full = Tensor::cat(&[&q_nope, &q_rope_rotated], -1);
    let k_full = Tensor::cat(&[&k_nope, &k_rope_rotated], -1);

    let scores = q_full.matmul(&k_full.transpose(-2, -1)) / ((qk_nope + qk_rope) as f64).sqrt();
    let causal_mask = Tensor::ones([seq as i64, seq as i64], (Kind::Bool, input.device())).triu(1);
    let scores = scores.masked_fill(&causal_mask, f64::NEG_INFINITY);
    let probs = scores.softmax(-1, Kind::Float).to_kind(v.kind());
    let context = probs.matmul(&v);
    let context = context
        .transpose(1, 2)
        .reshape([batch, seq, heads * v_head]);

    // o_proj (row parallel - partial output, needs all-reduce)
    context.linear::<&Tensor>(&attn.o_proj, None)
}

/// TP forward for dense layer (sharded).
/// Returns partial output (needs all-reduce for MLP down_proj).
pub fn tp_dense_layer(
    input: &Tensor,
    weights: &TpDenseLayerWeights,
    config: &DeepSeekRuntimeConfig,
    shard: &MlaTpShard,
) -> Tensor {
    let hidden = rms_norm(input, &weights.input_norm, config.rms_norm_eps);
    let attn_out = tp_mla_attention(&hidden, &weights.attn, config, shard);
    // Attention output is row-parallel (o_proj split), needs all-reduce
    // For now, return partial - caller must all-reduce
    let residual = input + &attn_out;
    let mlp_input = rms_norm(&residual, &weights.post_attention_norm, config.rms_norm_eps);
    // MLP: gate and up are column parallel (independent), down is row parallel (needs all-reduce)
    let gate = mlp_input.linear::<&Tensor>(&weights.gate_proj, None);
    let up = mlp_input.linear::<&Tensor>(&weights.up_proj, None);
    let intermediate = gate.silu() * up;
    let mlp_out = intermediate.linear::<&Tensor>(&weights.down_proj, None);
    // Partial output: attn_out + mlp_out both need all-reduce
    // We return the full residual + partial (attn + mlp), caller all-reduces (attn + mlp)
    residual + mlp_out
}

/// Run TP rank process for DeepSeek.
/// This is called by `launch --nproc-per-node 2 -- train --config configs/deepseek_v3_tp2.toml`
pub fn deepseek_tp_rank(
    model_path: &Path,
    output_dir: &Path,
    config: &DeepSeekRuntimeConfig,
    kind: Kind,
) -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;

    if world_size != 2 && world_size != 1 {
        bail!("DeepSeek TP currently expects WORLD_SIZE=2, got {world_size}");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }

    let device = Device::Cuda(local_rank);
    info!(rank, world_size, local_rank, "DeepSeek TP rank starting");

    let shard = MlaTpShard::new(rank, world_size, config.num_attention_heads);
    info!(
        heads_per_rank = shard.heads_per_rank,
        head_start = shard.head_start,
        "TP shard configured"
    );

    // Load weights (FP8→bf16 conversion via Python)
    let mut needed: std::collections::HashSet<String> = std::collections::HashSet::new();
    needed.insert("model.embed_tokens.weight".to_string());
    needed.insert("model.norm.weight".to_string());
    if !config.tie_word_embeddings {
        needed.insert("lm_head.weight".to_string());
    }
    let layer0_names = deepseek_trainable_tensors_for_layer(0, config);
    needed.extend(layer0_names);
    let weights = crate::session::load_deepseek_weights(model_path, &needed)?;

    // Move to GPU
    let weights_gpu: BTreeMap<String, Tensor> = weights
        .into_iter()
        .map(|(name, t)| (name, t.to_device(device).to_kind(kind)))
        .collect();

    // Load TP-sharded layer 0 (dense)
    let layer_weights = TpDenseLayerWeights::load_sharded(&weights_gpu, 0, &shard, config, kind)?;

    // Create test input
    let input_ids = Tensor::from_slice(&[1i64, 2, 3, 4, 5])
        .reshape([1, 5])
        .to_device(device);
    let embed_tokens = tensor(&weights_gpu, "model.embed_tokens.weight")?.to_kind(kind);
    let hidden = Tensor::embedding(&embed_tokens, &input_ids, -1, false, false);

    // TP forward: each rank computes its partial output
    let partial = tp_dense_layer(&hidden, &layer_weights, config, &shard);

    // All-reduce: combine partial outputs from all ranks
    let reduced = if world_size > 1 {
        nccl_smoke::all_reduce_tensor_f32_for_launch(
            &std::path::PathBuf::from(output_dir).join(format!("tp-rank-{rank}-output")),
            &partial,
        )?
    } else {
        partial.shallow_clone()
    };

    info!(
        rank,
        output_norm = reduced.norm().double_value(&[]),
        "TP output reduced"
    );

    // Compare with full (non-TP) forward for parity
    let full_layer = DeepSeekDenseLayerWeights::load_with_kind(&weights_gpu, 0, kind)?;
    let full_output = deepseek_layer(&hidden, &full_layer, config);
    let diff = (&reduced - &full_output).abs().max().double_value(&[]);

    info!(rank, max_diff = diff, "TP parity check");
    if diff > 1e-3 {
        bail!("DeepSeek TP parity failed: rank={rank}, max_diff={diff}");
    }

    info!(rank, "DeepSeek TP rank complete");
    println!("rank={rank} tp_parity_max_diff={diff:.6}");
    Ok(())
}
