//! V4 TP (Tensor Parallel) support.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use tch::{Device, Kind, Tensor};
use tracing::info;

use rustrain_checkpoint::safetensors::tensor;
use rustrain_nccl::nccl as nccl_smoke;

use crate::model::*;

fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

pub struct V4TpShard {
    pub rank: usize,
    pub world_size: usize,
    pub heads_per_rank: i64,
    pub head_start: i64,
}

impl V4TpShard {
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

/// TP-sharded V4 attention weights.
/// wq_a: replicated, wq_b: column parallel (by heads), wkv: replicated,
/// wo_a: row parallel input, wo_b: row parallel output (all-reduce).
pub struct TpV4AttentionWeights {
    pub wq_a: Tensor,
    pub q_norm: Tensor,
    pub wq_b: Tensor, // sharded: [heads_per_rank * 512, q_lora_rank]
    pub wkv: Tensor,  // replicated
    pub kv_norm: Tensor,
    pub attn_sink: Tensor, // sharded: [heads_per_rank]
    pub wo_a: Tensor,      // replicated input, sharded output
    pub wo_b: Tensor,      // sharded: [hidden, o_groups_per_rank * 512]
}

impl TpV4AttentionWeights {
    pub fn load_sharded(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        shard: &V4TpShard,
        config: &V4RuntimeConfig,
        kind: Kind,
    ) -> Result<Self> {
        let p = format!("layers.{layer}.attn");
        let head_dim = 512_i64;
        let num_heads = config.num_attention_heads;

        // wq_b: [num_heads * 512, q_lora_rank] → split by heads
        let wq_b_full = tensor(weights, &format!("{p}.wq_b.weight"))?.to_kind(kind);
        let wq_b_row_start = shard.head_start * head_dim;
        let wq_b_row_size = shard.heads_per_rank * head_dim;
        let wq_b = wq_b_full
            .narrow(0, wq_b_row_start, wq_b_row_size)
            .shallow_clone();

        // attn_sink: [num_heads] → split by heads
        let sink_full = tensor(weights, &format!("{p}.attn_sink"))?.to_kind(kind);
        let attn_sink = sink_full
            .narrow(0, shard.head_start, shard.heads_per_rank)
            .shallow_clone();

        // wo_a: [o_groups * 512, hidden] → split by o_groups
        // wo_b: [hidden, o_groups * 512] → split by o_groups (column)
        let o_groups_per_rank = config.o_groups / shard.world_size as i64;
        let wo_a_full = tensor(weights, &format!("{p}.wo_a.weight"))?.to_kind(kind);
        let wo_b_full = tensor(weights, &format!("{p}.wo_b.weight"))?.to_kind(kind);
        let wo_row_start = shard.rank as i64 * o_groups_per_rank * 512;
        let wo_row_size = o_groups_per_rank * 512;
        let wo_a = wo_a_full
            .narrow(0, wo_row_start, wo_row_size)
            .shallow_clone();
        let wo_b = wo_b_full
            .narrow(1, wo_row_start, wo_row_size)
            .shallow_clone();

        Ok(Self {
            wq_a: tensor(weights, &format!("{p}.wq_a.weight"))?.to_kind(kind),
            q_norm: tensor(weights, &format!("{p}.q_norm.weight"))?.to_kind(kind),
            wq_b,
            wkv: tensor(weights, &format!("{p}.wkv.weight"))?.to_kind(kind),
            kv_norm: tensor(weights, &format!("{p}.kv_norm.weight"))?.to_kind(kind),
            attn_sink,
            wo_a,
            wo_b,
        })
    }
}

pub fn tp_v4_attention(
    input: &Tensor,
    attn: &TpV4AttentionWeights,
    config: &V4RuntimeConfig,
    shard: &V4TpShard,
) -> Tensor {
    let shape = input.size();
    let batch = shape[0];
    let seq = shape[1];
    let heads = shard.heads_per_rank;
    let head_dim = 512_i64;
    let qk_rope = config.qk_rope_head_dim;
    let qk_nope = head_dim - qk_rope;
    let o_groups_per_rank = config.o_groups / shard.world_size as i64;

    // Q path (replicated wq_a, sharded wq_b)
    let q_a = input.linear::<&Tensor>(&attn.wq_a, None);
    let q_a = rms_norm(&q_a, &attn.q_norm, config.rms_norm_eps);
    let q_b = q_a.linear::<&Tensor>(&attn.wq_b, None);
    let q = q_b.reshape([batch, seq, heads, head_dim]).transpose(1, 2);
    let q_nope = q.narrow(-1, 0, qk_nope);
    let q_rope = q.narrow(-1, qk_nope, qk_rope);

    // KV path (replicated)
    let wkv_out = input.linear::<&Tensor>(&attn.wkv, None);
    let kv = rms_norm(&wkv_out, &attn.kv_norm, config.rms_norm_eps);
    let k_nope = kv
        .narrow(-1, 0, qk_nope)
        .reshape([batch, 1, seq, qk_nope])
        .expand([batch, heads, seq, qk_nope], false);
    let k_rope = kv
        .narrow(-1, qk_nope, qk_rope)
        .reshape([batch, 1, seq, qk_rope])
        .expand([batch, heads, seq, qk_rope], false);
    let v = kv
        .reshape([batch, 1, seq, head_dim])
        .expand([batch, heads, seq, head_dim], false);

    let (cos, sin) = rope_cos_sin(seq as usize, qk_rope, config.rope_theta, input.device());
    let cos = cos.to_kind(input.kind());
    let sin = sin.to_kind(input.kind());
    let q_rope_rotated = apply_rotary(&q_rope, &cos, &sin);
    let k_rope_rotated = apply_rotary(&k_rope, &cos, &sin);

    let q_full = Tensor::cat(&[&q_nope, &q_rope_rotated], -1);
    let k_full = Tensor::cat(&[&k_nope, &k_rope_rotated], -1);

    let scale = 1.0 / (head_dim as f64).sqrt();
    let attn_scores = q_full.matmul(&k_full.transpose(-2, -1)) * scale;
    let sink = attn
        .attn_sink
        .reshape([1, heads, 1, 1])
        .to_kind(attn_scores.kind());
    let scores = attn_scores + sink;
    let causal_mask = Tensor::ones([seq as i64, seq as i64], (Kind::Bool, input.device())).triu(1);
    let scores = scores.masked_fill(&causal_mask, f64::NEG_INFINITY);
    let probs = scores.softmax(-1, Kind::Float).to_kind(v.kind());
    let context = probs.matmul(&v);

    // Group reduction (local heads only)
    let heads_per_group = heads / o_groups_per_rank;
    let context = context
        .reshape([batch, o_groups_per_rank, heads_per_group, seq, head_dim])
        .sum_dim_intlist([2].as_slice(), false, Kind::Float);
    let context = context.reshape([batch, seq, o_groups_per_rank * head_dim]);

    // Output: wo_a → wo_b (partial, needs all-reduce for wo_b)
    let o_compressed = context.linear::<&Tensor>(&attn.wo_a, None);
    o_compressed.linear::<&Tensor>(&attn.wo_b, None)
}

pub fn deepseek_v4_tp_rank(
    model_path: &Path,
    output_dir: &Path,
    config: &V4RuntimeConfig,
    kind: Kind,
) -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;

    if world_size > 1 && world_size != 2 {
        bail!("V4 TP currently expects WORLD_SIZE<=2, got {world_size}");
    }

    let device = Device::Cuda(local_rank);
    info!(rank, world_size, local_rank, "V4 TP rank starting");

    let shard = V4TpShard::new(rank, world_size, config.num_attention_heads);
    info!(
        heads_per_rank = shard.heads_per_rank,
        head_start = shard.head_start,
        "TP shard"
    );

    let mut needed: std::collections::HashSet<String> = std::collections::HashSet::new();
    needed.insert("embed.weight".to_string());
    needed.insert("norm.weight".to_string());
    needed.insert("head.weight".to_string());
    let layer0_names = v4_trainable_tensors_for_layer(0, config);
    needed.extend(layer0_names);

    let weights = crate::model::load_v4_weights(model_path, &needed)?;
    let weights_gpu: BTreeMap<String, Tensor> = weights
        .into_iter()
        .map(|(name, t)| (name, t.to_device(device).to_kind(kind)))
        .collect();

    let input_ids = Tensor::from_slice(&[1i64, 2, 3, 4, 5])
        .reshape([1, 5])
        .to_device(device);
    let embed = tensor(&weights_gpu, "embed.weight")?.to_kind(kind);
    let hidden = Tensor::embedding(&embed, &input_ids, -1, false, false);
    let hidden_norm = rms_norm(
        &hidden,
        &tensor(&weights_gpu, "layers.0.attn_norm.weight")?.to_kind(kind),
        config.rms_norm_eps,
    );

    if world_size == 1 {
        // TP=1: verify full attention works
        let full_attn = V4AttentionWeights::load_raw(&weights_gpu, 0)?;
        let out = v4_attention(&hidden_norm, &full_attn, config);
        info!(
            rank,
            output_norm = out.norm().double_value(&[]),
            "V4 TP=1 attention verified"
        );
        println!("rank=0 v4_tp_parity_max_diff=0.000000");
        return Ok(());
    }

    let tp_attn = TpV4AttentionWeights::load_sharded(&weights_gpu, 0, &shard, config, kind)?;
    let partial = tp_v4_attention(&hidden_norm, &tp_attn, config, &shard);

    let reduced = if world_size > 1 {
        nccl_smoke::all_reduce_tensor_f32_for_launch(
            &std::path::PathBuf::from(output_dir).join(format!("v4-tp-rank-{rank}")),
            &partial,
        )?
    } else {
        partial.shallow_clone()
    };

    let full_attn = V4AttentionWeights::load_raw(&weights_gpu, 0)?;
    let hidden_norm = rms_norm(
        &hidden,
        &tensor(&weights_gpu, "layers.0.attn_norm.weight")?.to_kind(kind),
        config.rms_norm_eps,
    );
    let full_out = v4_attention(&hidden_norm, &full_attn, config);
    let diff = (&reduced - &full_out).abs().max().double_value(&[]);

    info!(rank, max_diff = diff, "V4 TP parity");
    if diff > 1e-3 {
        bail!("V4 TP parity failed: rank={rank}, max_diff={diff}");
    }

    info!(rank, "V4 TP rank complete");
    println!("rank={rank} v4_tp_parity_max_diff={diff:.6}");
    Ok(())
}
