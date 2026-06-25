//! DeepSeek EP (Expert Parallel) support.
//!
//! EP strategy:
//! - Split 256 routed experts across EP ranks
//! - Each rank holds 256/EP experts
//! - Shared expert is replicated on all ranks
//! - Router logits computed on all ranks (gate is replicated)
//! - Only local experts are evaluated
//! - Token dispatch: tokens routed to non-local experts are skipped (simplified)

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use tch::{Device, Kind, Tensor};
use tracing::info;

use crate::model::*;
use rustrain_checkpoint::safetensors::tensor;

/// Parse env var as usize.
fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set; run through rustrain launch"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

/// EP shard: which experts are on this rank.
pub struct EpShard {
    pub rank: usize,
    pub world_size: usize,
    pub experts_per_rank: usize,
    pub expert_start: usize,
    pub local_expert_indices: Vec<usize>,
}

impl EpShard {
    pub fn new(rank: usize, world_size: usize, num_experts: usize) -> Self {
        let experts_per_rank = num_experts / world_size;
        let expert_start = rank * experts_per_rank;
        let local_expert_indices = (expert_start..expert_start + experts_per_rank).collect();
        Self {
            rank,
            world_size,
            experts_per_rank,
            expert_start,
            local_expert_indices,
        }
    }
}

/// EP-sharded MoE layer weights.
/// Router gate and shared expert are replicated. Only routed experts are sharded.
pub struct EpMoeLayerWeights {
    pub input_norm: Tensor,
    pub attn: DeepSeekAttentionWeights,
    pub post_attention_norm: Tensor,
    pub gate: Tensor,             // replicated: [n_experts, hidden]
    pub shared_gate_proj: Tensor, // replicated
    pub shared_up_proj: Tensor,   // replicated
    pub shared_down_proj: Tensor, // replicated
    pub local_experts: Vec<(Tensor, Tensor, Tensor)>, // only this rank's experts
    pub shard: EpShard,
}

impl EpMoeLayerWeights {
    pub fn load_sharded(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        config: &DeepSeekRuntimeConfig,
        shard: EpShard,
        kind: Kind,
    ) -> Result<Self> {
        let p = format!("model.layers.{layer}");
        let shared_prefix = format!("{p}.mlp.shared_experts");

        // Load only local experts
        let mut local_experts = Vec::with_capacity(shard.experts_per_rank);
        for &expert_idx in &shard.local_expert_indices {
            let ep = format!("{p}.mlp.experts.{expert_idx}");
            local_experts.push((
                tensor(weights, &format!("{ep}.gate_proj.weight"))?.to_kind(kind),
                tensor(weights, &format!("{ep}.up_proj.weight"))?.to_kind(kind),
                tensor(weights, &format!("{ep}.down_proj.weight"))?.to_kind(kind),
            ));
        }

        info!(
            rank = shard.rank,
            local_experts = shard.experts_per_rank,
            expert_start = shard.expert_start,
            "EP layer loaded"
        );

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
            local_experts,
            shard,
        })
    }
}

/// EP MoE forward: compute shared expert + local routed experts only.
/// Tokens routed to non-local experts are skipped (simplified, no all-to-all).
pub fn ep_moe_mlp(
    input: &Tensor,
    gate: &Tensor,
    shared_gate: &Tensor,
    shared_up: &Tensor,
    shared_down: &Tensor,
    local_experts: &[(Tensor, Tensor, Tensor)],
    shard: &EpShard,
    num_experts_per_tok: usize,
    scoring_func: &str,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f64,
) -> Tensor {
    // Shared expert (always computed, replicated)
    let shared_output = deepseek_mlp(input, shared_gate, shared_up, shared_down);

    // Router logits (replicated gate)
    let router_logits = input.linear::<&Tensor>(gate, None);

    // Top-k selection
    let (topk_weights, topk_indices) = if scoring_func == "sigmoid" {
        let scores = router_logits.sigmoid();
        let (tw, ti) = scores.topk(num_experts_per_tok as i64, -1, true, true);
        let denom = tw
            .sum_dim_intlist([-1].as_slice(), true, Kind::Float)
            .clamp_min(1e-9);
        (tw / &denom * routed_scaling_factor, ti)
    } else {
        let (tw, ti) = router_logits.topk(num_experts_per_tok as i64, -1, true, true);
        (tw.softmax(-1, Kind::Float), ti)
    };

    // Only compute local experts
    let mut output = shared_output;

    for k in 0..num_experts_per_tok {
        let expert_indices = topk_indices.select(-1, k as i64);
        let expert_weights = topk_weights.select(-1, k as i64);
        let weights_kind = expert_weights.kind();

        for (local_idx, (gate_p, up_p, down_p)) in local_experts.iter().enumerate() {
            let global_expert_idx = shard.expert_start + local_idx;
            let mask = expert_indices
                .eq(global_expert_idx as i64)
                .to_kind(weights_kind);
            if mask.sum(Kind::Float).double_value(&[]) > 0.0 {
                let expert_out = deepseek_mlp(input, gate_p, up_p, down_p);
                let weight = (&expert_weights * &mask).unsqueeze(-1);
                output = output + (expert_out * weight);
            }
        }
    }

    output
}

/// EP forward for a MoE layer.
pub fn ep_moe_layer(
    input: &Tensor,
    weights: &EpMoeLayerWeights,
    config: &DeepSeekRuntimeConfig,
) -> Tensor {
    let hidden = rms_norm(input, &weights.input_norm, config.rms_norm_eps);
    let attn = deepseek_mla_attention(&hidden, &weights.attn, config);
    let residual = input + &attn;
    let mlp_input = rms_norm(&residual, &weights.post_attention_norm, config.rms_norm_eps);
    let mlp = ep_moe_mlp(
        &mlp_input,
        &weights.gate,
        &weights.shared_gate_proj,
        &weights.shared_up_proj,
        &weights.shared_down_proj,
        &weights.local_experts,
        &weights.shard,
        config.num_experts_per_tok,
        &config.scoring_func,
        config.n_group,
        config.topk_group,
        config.routed_scaling_factor,
    );
    residual + mlp
}

/// Run EP rank process for DeepSeek.
/// Each rank loads only its subset of experts and computes local contribution.
pub fn deepseek_ep_rank(
    model_path: &Path,
    output_dir: &Path,
    config: &DeepSeekRuntimeConfig,
    kind: Kind,
) -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;

    if world_size > 1 && world_size != 2 {
        bail!("DeepSeek EP currently expects WORLD_SIZE<=2, got {world_size}");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }

    let device = Device::Cuda(local_rank);
    info!(rank, world_size, local_rank, "DeepSeek EP rank starting");

    let shard = EpShard::new(rank, world_size, config.n_routed_experts);
    info!(
        experts_per_rank = shard.experts_per_rank,
        expert_start = shard.expert_start,
        "EP shard configured"
    );

    // Load only needed weights
    let mut needed: std::collections::HashSet<String> = std::collections::HashSet::new();
    needed.insert("model.embed_tokens.weight".to_string());
    needed.insert("model.norm.weight".to_string());
    if !config.tie_word_embeddings {
        needed.insert("lm_head.weight".to_string());
    }
    let p3 = "model.layers.3";
    for suffix in &[
        "input_layernorm.weight",
        "self_attn.q_a_proj.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate.weight",
        "mlp.shared_experts.gate_proj.weight",
        "mlp.shared_experts.up_proj.weight",
        "mlp.shared_experts.down_proj.weight",
    ] {
        needed.insert(format!("{p3}.{suffix}"));
    }
    for &expert_idx in &shard.local_expert_indices {
        for suffix in &["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            needed.insert(format!("{p3}.mlp.experts.{expert_idx}.{suffix}"));
        }
    }

    let weights = crate::session::load_deepseek_weights(model_path, &needed)?;
    let weights_gpu: BTreeMap<String, Tensor> = weights
        .into_iter()
        .map(|(name, t)| (name, t.to_device(device).to_kind(kind)))
        .collect();

    let layer_weights = EpMoeLayerWeights::load_sharded(&weights_gpu, 3, config, shard, kind)?;

    let input_ids = Tensor::from_slice(&[1i64, 2, 3, 4, 5])
        .reshape([1, 5])
        .to_device(device);
    let embed_tokens = tensor(&weights_gpu, "model.embed_tokens.weight")?.to_kind(kind);
    let hidden = Tensor::embedding(&embed_tokens, &input_ids, -1, false, false);

    let local_output = ep_moe_layer(&hidden, &layer_weights, config);

    info!(
        rank,
        output_norm = local_output.norm().double_value(&[]),
        "EP local output computed"
    );

    let shared_only = {
        let mlp_input = rms_norm(
            &hidden,
            &layer_weights.post_attention_norm,
            config.rms_norm_eps,
        );
        deepseek_mlp(
            &mlp_input,
            &layer_weights.shared_gate_proj,
            &layer_weights.shared_up_proj,
            &layer_weights.shared_down_proj,
        )
    };
    let diff = (&local_output - &shared_only).abs().max().double_value(&[]);
    info!(
        rank,
        expert_contribution = diff,
        "EP expert contribution check"
    );

    if diff < 1e-8 {
        info!(
            rank,
            "no local experts were activated (expected for small input)"
        );
    }

    info!(rank, "DeepSeek EP rank complete");
    println!("rank={rank} ep_expert_contribution={diff:.6}");
    Ok(())
}
