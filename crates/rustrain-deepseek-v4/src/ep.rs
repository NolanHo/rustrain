//! V4 EP (Expert Parallel) support.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use tch::{Device, Kind, Tensor};
use tracing::info;

use crate::model::*;
use rustrain_checkpoint::safetensors::tensor;

fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

pub struct V4EpShard {
    pub rank: usize,
    pub world_size: usize,
    pub experts_per_rank: usize,
    pub expert_start: usize,
    pub local_expert_indices: Vec<usize>,
}

impl V4EpShard {
    pub fn new(rank: usize, world_size: usize, num_experts: usize) -> Self {
        let experts_per_rank = num_experts / world_size;
        let expert_start = rank * experts_per_rank;
        Self {
            rank,
            world_size,
            experts_per_rank,
            expert_start,
            local_expert_indices: (expert_start..expert_start + experts_per_rank).collect(),
        }
    }
}

pub fn deepseek_v4_ep_rank(
    model_path: &Path,
    output_dir: &Path,
    config: &V4RuntimeConfig,
    kind: Kind,
) -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;

    if world_size > 1 && world_size != 2 {
        bail!("V4 EP currently expects WORLD_SIZE<=2, got {world_size}");
    }

    let device = Device::Cuda(local_rank);
    info!(rank, world_size, local_rank, "V4 EP rank starting");

    let shard = V4EpShard::new(rank, world_size, config.n_routed_experts);
    info!(
        experts_per_rank = shard.experts_per_rank,
        expert_start = shard.expert_start,
        "EP shard"
    );

    // Load only needed weights
    let mut needed: std::collections::HashSet<String> = std::collections::HashSet::new();
    needed.insert("embed.weight".to_string());
    needed.insert("norm.weight".to_string());
    needed.insert("head.weight".to_string());

    let p = "layers.0";
    // Attention + norm + gate + shared
    for suffix in &[
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn.wq_a.weight",
        "attn.wq_b.weight",
        "attn.wkv.weight",
        "attn.wo_a.weight",
        "attn.wo_b.weight",
        "attn.q_norm.weight",
        "attn.kv_norm.weight",
        "attn.attn_sink",
        "ffn.gate.weight",
        "ffn.shared_experts.w1.weight",
        "ffn.shared_experts.w2.weight",
        "ffn.shared_experts.w3.weight",
    ] {
        needed.insert(format!("{p}.{suffix}"));
    }
    // Only local experts
    for &e in &shard.local_expert_indices {
        for suffix in &["w1.weight", "w2.weight", "w3.weight"] {
            needed.insert(format!("{p}.ffn.experts.{e}.{suffix}"));
        }
    }

    let weights = crate::model::load_v4_weights(model_path, &needed)?;
    let weights_gpu: BTreeMap<String, Tensor> = weights
        .into_iter()
        .map(|(name, t)| (name, t.to_device(device).to_kind(kind)))
        .collect();

    // Load full layer for comparison
    let lw = V4MoeLayerWeights::load_raw(&weights_gpu, 0, shard.experts_per_rank)?;

    let input_ids = Tensor::from_slice(&[1i64, 2, 3, 4, 5])
        .reshape([1, 5])
        .to_device(device);
    let embed = tensor(&weights_gpu, "embed.weight")?.to_kind(kind);
    let hidden = Tensor::embedding(&embed, &input_ids, -1, false, false);

    let local_output = v4_moe_layer(&hidden, &lw, config);
    info!(
        rank,
        output_norm = local_output.norm().double_value(&[]),
        "EP local output"
    );

    // Check expert contribution
    let mlp_input = rms_norm(&hidden, &lw.ffn_norm, config.rms_norm_eps);
    let shared_only = v4_swiglu(
        &mlp_input,
        &lw.shared_experts.0,
        &lw.shared_experts.1,
        &lw.shared_experts.2,
        config.swiglu_limit,
    );
    let diff = (&local_output - &hidden).abs().max().double_value(&[])
        - shared_only.abs().max().double_value(&[]);

    info!(rank, expert_contribution = diff, "EP expert contribution");
    println!("rank={rank} v4_ep_expert_contribution={diff:.6}");
    Ok(())
}
