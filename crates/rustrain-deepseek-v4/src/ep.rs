//! V4 EP (Expert Parallel) support.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use anyhow::{bail, Context, Result};
use tch::{no_grad, Device, Kind, Reduction, Tensor};
use tracing::info;

use crate::model::*;
use rustrain_checkpoint::safetensors::tensor;
use rustrain_nccl::nccl as nccl_smoke;

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

    /// Returns true if the given global expert index belongs to this rank.
    pub fn owns_expert(&self, global_idx: usize) -> bool {
        (self.expert_start..self.expert_start + self.experts_per_rank).contains(&global_idx)
    }

    /// Maps a global expert index to a local index (0-based within this rank).
    pub fn local_index(&self, global_idx: usize) -> usize {
        global_idx - self.expert_start
    }
}

/// EP-sharded MoE layer weights.
///
/// Attention and shared experts are replicated across all ranks.
/// Only routed experts are sharded: each rank holds `experts_per_rank` experts
/// with global indices `[expert_start, expert_start + experts_per_rank)`.
pub struct V4EpMoeLayerWeights {
    pub attn_norm: Tensor,
    pub attn: V4AttentionWeights,
    pub ffn_norm: Tensor,
    pub gate: Tensor,
    pub shared_experts: (Tensor, Tensor, Tensor),
    pub local_experts: Vec<(Tensor, Tensor, Tensor)>,
    pub shard: V4EpShard,
}

impl V4EpMoeLayerWeights {
    pub fn load_raw(
        weights: &BTreeMap<String, Tensor>,
        layer: usize,
        shard: &V4EpShard,
    ) -> Result<Self> {
        let p = format!("layers.{layer}");
        let sp = format!("{p}.ffn.shared_experts");

        // Load local experts (global indices → local storage)
        let mut local_experts = Vec::with_capacity(shard.experts_per_rank);
        for &global_e in &shard.local_expert_indices {
            let ep = format!("{p}.ffn.experts.{global_e}");
            local_experts.push((
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
            local_experts,
            shard: V4EpShard::new(
                shard.rank,
                shard.world_size,
                shard.experts_per_rank * shard.world_size,
            ),
        })
    }
}

/// EP-sharded MoE MLP.
///
/// Computes shared expert (replicated) + local routed experts.
/// Only processes experts that belong to this rank; other experts' contributions
/// come from other ranks via all-reduce.
///
/// Returns the partial output (shared + local expert contributions).
/// Caller must all-reduce across ranks to get the full output.
pub fn v4_moe_mlp_ep(
    input: &Tensor,
    gate: &Tensor,
    shared: &(Tensor, Tensor, Tensor),
    local_experts: &[(Tensor, Tensor, Tensor)],
    shard: &V4EpShard,
    config: &V4RuntimeConfig,
) -> Tensor {
    let num_experts_per_tok = config.num_experts_per_tok;
    let swiglu_limit = config.swiglu_limit;
    let n_experts = config.n_routed_experts;

    // Shared expert (always computed, replicated)
    let shared_out = v4_swiglu(input, &shared.0, &shared.1, &shared.2, swiglu_limit);

    // Router logits — all ranks compute the same (gate is replicated)
    let router_logits = input.linear::<&Tensor>(gate, None); // [batch, seq, n_experts]

    // Scoring function: sqrtsoftplus (V4 specific)
    let scores = if config.scoring_func == "sqrtsoftplus" {
        let sp = (router_logits.shallow_clone().exp() + 1.0).log();
        sp.sqrt()
    } else {
        router_logits.shallow_clone().softmax(-1, Kind::Float)
    };

    // Top-k selection (same on all ranks since gate is replicated)
    let (topk_weights, topk_indices) = scores.topk(num_experts_per_tok as i64, -1, true, true);

    // Normalize top-k weights
    let denom = topk_weights
        .sum_dim_intlist([-1].as_slice(), true, Kind::Float)
        .clamp_min(1e-9);
    let topk_weights = topk_weights / &denom * config.routed_scaling_factor;

    // Accumulate expert outputs — only for local experts
    let mut output = shared_out;
    for k in 0..num_experts_per_tok {
        let expert_indices = topk_indices.select(-1, k as i64); // [batch, seq] global expert ids
        let expert_weights = topk_weights.select(-1, k as i64); // [batch, seq]
        let weights_kind = expert_weights.kind();

        for (local_idx, (w1, w2, w3)) in local_experts.iter().enumerate() {
            let global_expert_idx = shard.expert_start + local_idx;
            let mask = expert_indices
                .eq(global_expert_idx as i64)
                .to_kind(weights_kind);
            if mask.sum(Kind::Float).double_value(&[]) > 0.0 {
                let expert_out = v4_swiglu(input, w1, w2, w3, swiglu_limit);
                let weight = (&expert_weights * &mask).unsqueeze(-1);
                output = output + (expert_out * weight);
            }
        }
    }
    output
}

/// EP-sharded MoE routed-only (no shared expert).
/// Computes only the routed expert contributions for this rank's local experts.
/// Caller must all-reduce and add the shared expert separately.
pub fn v4_moe_routed_only_ep(
    input: &Tensor,
    gate: &Tensor,
    local_experts: &[(Tensor, Tensor, Tensor)],
    shard: &V4EpShard,
    config: &V4RuntimeConfig,
) -> Tensor {
    let num_experts_per_tok = config.num_experts_per_tok;
    let swiglu_limit = config.swiglu_limit;

    let router_logits = input.linear::<&Tensor>(gate, None);
    let scores = if config.scoring_func == "sqrtsoftplus" {
        let sp = (router_logits.shallow_clone().exp() + 1.0).log();
        sp.sqrt()
    } else {
        router_logits.shallow_clone().softmax(-1, Kind::Float)
    };
    let (topk_weights, topk_indices) = scores.topk(num_experts_per_tok as i64, -1, true, true);
    let denom = topk_weights
        .sum_dim_intlist([-1].as_slice(), true, Kind::Float)
        .clamp_min(1e-9);
    let topk_weights = topk_weights / &denom * config.routed_scaling_factor;

    let mut output = Tensor::zeros(input.size(), (input.kind(), input.device()));
    for k in 0..num_experts_per_tok {
        let expert_indices = topk_indices.select(-1, k as i64);
        let expert_weights = topk_weights.select(-1, k as i64);
        let weights_kind = expert_weights.kind();

        for (local_idx, (w1, w2, w3)) in local_experts.iter().enumerate() {
            let global_expert_idx = shard.expert_start + local_idx;
            let mask = expert_indices
                .eq(global_expert_idx as i64)
                .to_kind(weights_kind);
            if mask.sum(Kind::Float).double_value(&[]) > 0.0 {
                let expert_out = v4_swiglu(input, w1, w2, w3, swiglu_limit);
                let weight = (&expert_weights * &mask).unsqueeze(-1);
                output = output + (expert_out * weight);
            }
        }
    }
    output
}

/// EP-sharded MoE layer: attention (replicated) + EP MoE MLP.
///
/// Returns partial output. Caller must all-reduce the MoE part.
/// For simplicity, this computes the full layer and returns it;
/// the all-reduce happens at the MoE output level (before residual).
pub fn v4_moe_layer_ep(
    input: &Tensor,
    weights: &V4EpMoeLayerWeights,
    config: &V4RuntimeConfig,
) -> Tensor {
    let hidden = rms_norm(input, &weights.attn_norm, config.rms_norm_eps);
    let attn_out = v4_attention(&hidden, &weights.attn, config);
    let residual = input + &attn_out;
    let mlp_input = rms_norm(&residual, &weights.ffn_norm, config.rms_norm_eps);
    let mlp = v4_moe_mlp_ep(
        &mlp_input,
        &weights.gate,
        &weights.shared_experts,
        &weights.local_experts,
        &weights.shard,
        config,
    );
    residual + mlp
}

// ── EP Parity Test ────────────────────────────────────────────────────────────

pub fn deepseek_v4_ep_rank(
    model_path: &Path,
    output_dir: &Path,
    config: &V4RuntimeConfig,
    kind: Kind,
) -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;

    if world_size > 1 && world_size > config.n_routed_experts {
        bail!(
            "V4 EP world_size {world_size} exceeds n_routed_experts {}",
            config.n_routed_experts
        );
    }
    if config.n_routed_experts % world_size != 0 {
        bail!(
            "V4 EP n_routed_experts {} must be divisible by world_size {world_size}",
            config.n_routed_experts
        );
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
    let mut needed: HashSet<String> = HashSet::new();
    needed.insert("embed.weight".to_string());
    needed.insert("norm.weight".to_string());
    needed.insert("head.weight".to_string());

    let p = "layers.0";
    // Attention + norm + gate + shared (all replicated)
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

    let ep_lw = V4EpMoeLayerWeights::load_raw(&weights_gpu, 0, &shard)?;

    let input_ids = Tensor::from_slice(&[1i64, 2, 3, 4, 5])
        .reshape([1, 5])
        .to_device(device);
    let embed = tensor(&weights_gpu, "embed.weight")?.to_kind(kind);
    let hidden = Tensor::embedding(&embed, &input_ids, -1, false, false);

    // EP forward: each rank computes partial output (shared + local experts)
    let partial = v4_moe_layer_ep(&hidden, &ep_lw, config);
    info!(
        rank,
        partial_norm = partial.norm().double_value(&[]),
        "EP partial output (shared + local experts)"
    );

    // All-reduce to get the full output (sum of all ranks' expert contributions)
    let full = if world_size > 1 {
        nccl_smoke::all_reduce_tensor_f32_for_launch(
            &output_dir.to_path_buf().join(format!("v4-ep-rank-{rank}")),
            &partial,
        )?
    } else {
        partial.shallow_clone()
    };

    info!(
        rank,
        full_norm = full.norm().double_value(&[]),
        "EP full output (after all-reduce)"
    );

    // Verify: compare against full (non-EP) computation if world_size == 1
    if world_size == 1 {
        let full_lw = V4MoeLayerWeights::load_raw(&weights_gpu, 0, config.n_routed_experts)?;
        let reference = v4_moe_layer(&hidden, &full_lw, config);
        let diff = (&full - &reference).abs().max().double_value(&[]);
        info!(rank, max_diff = diff, "EP=1 parity check");
        println!("rank={rank} v4_ep_parity_max_diff={diff:.6}");
    } else {
        println!(
            "rank={rank} v4_ep_full_norm={:.6}",
            full.norm().double_value(&[])
        );
    }

    Ok(())
}

// ── EP Full Training Loop ────────────────────────────────────────────────────
//
// Each rank:
//   1. Forward: attention (replicated) + EP MoE (local experts only)
//   2. All-reduce MoE output (sum across ranks)
//   3. Loss + backward
//   4. All-reduce gradients of replicated params (attention, gate, shared)
//   5. Optimizer step

pub fn deepseek_v4_ep_train(
    model_path: &Path,
    output_dir: &Path,
    config: &V4RuntimeConfig,
    kind: Kind,
) -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;

    if world_size > 1 && world_size > config.n_routed_experts {
        bail!(
            "V4 EP world_size {world_size} exceeds n_routed_experts {}",
            config.n_routed_experts
        );
    }
    if config.n_routed_experts % world_size != 0 {
        bail!(
            "V4 EP n_routed_experts {} must be divisible by world_size {world_size}",
            config.n_routed_experts
        );
    }

    let device = Device::Cuda(local_rank);
    info!(rank, world_size, local_rank, "V4 EP train starting");

    let shard = V4EpShard::new(rank, world_size, config.n_routed_experts);

    // ── Load weights ──
    let trainable_layer = 0;
    let mut needed: HashSet<String> = HashSet::new();
    needed.insert("embed.weight".to_string());
    needed.insert("norm.weight".to_string());
    needed.insert("head.weight".to_string());
    needed.insert(format!("layers.{trainable_layer}.attn_norm.weight"));
    needed.insert(format!("layers.{trainable_layer}.ffn_norm.weight"));
    for suffix in &[
        "wq_a.weight",
        "wq_b.weight",
        "wkv.weight",
        "wo_a.weight",
        "wo_b.weight",
        "q_norm.weight",
        "kv_norm.weight",
        "attn_sink",
    ] {
        needed.insert(format!("layers.{trainable_layer}.attn.{suffix}"));
    }
    needed.insert(format!("layers.{trainable_layer}.ffn.gate.weight"));
    for suffix in &["w1.weight", "w2.weight", "w3.weight"] {
        needed.insert(format!(
            "layers.{trainable_layer}.ffn.shared_experts.{suffix}"
        ));
    }
    // Only local experts
    for &e in &shard.local_expert_indices {
        for suffix in &["w1.weight", "w2.weight", "w3.weight"] {
            needed.insert(format!("layers.{trainable_layer}.ffn.experts.{e}.{suffix}"));
        }
    }

    let weights = crate::model::load_v4_weights(model_path, &needed)?;
    let mut weights_gpu: BTreeMap<String, Tensor> = weights
        .into_iter()
        .map(|(name, t)| (name, t.to_device(device).to_kind(kind)))
        .collect();
    info!(rank, tensors = weights_gpu.len(), "weights loaded");

    // ── Set local expert params as trainable ──
    let mut trainable_params: Vec<(String, Tensor)> = Vec::new();
    for &global_e in &shard.local_expert_indices {
        for suffix in &["w1.weight", "w2.weight", "w3.weight"] {
            let name = format!("layers.{trainable_layer}.ffn.experts.{global_e}.{suffix}");
            if let Some(t) = weights_gpu.get_mut(&name) {
                let trainable = t.shallow_clone().to_kind(kind).set_requires_grad(true);
                weights_gpu.insert(name.clone(), trainable.shallow_clone());
                trainable_params.push((name.clone(), trainable));
            }
        }
    }
    info!(
        rank,
        trainable_tensors = trainable_params.len(),
        "local expert params set trainable"
    );

    // ── Training loop ──
    let input_ids = Tensor::from_slice(&[1i64, 2, 3, 4, 5])
        .reshape([1, 5])
        .to_device(device);
    let lr = 1e-6_f64;
    let num_steps = 3;
    let mut initial_loss = 0.0_f64;

    for step in 0..num_steps {
        // ── Forward ──
        let embed = tensor(&weights_gpu, "embed.weight")?.to_kind(kind);
        let hidden = Tensor::embedding(&embed, &input_ids, -1, false, false);

        // Attention (replicated, no autograd needed for parity test)
        let ep_lw = V4EpMoeLayerWeights::load_raw(&weights_gpu, trainable_layer, &shard)?;
        let attn_norm = rms_norm(
            &hidden,
            &tensor(
                &weights_gpu,
                &format!("layers.{trainable_layer}.attn_norm.weight"),
            )?
            .to_kind(kind),
            config.rms_norm_eps,
        );
        let attn_out = v4_attention(&attn_norm, &ep_lw.attn, config);
        let residual = &hidden + &attn_out;

        // EP MoE: partial output (shared + local experts)
        let mlp_input = rms_norm(&residual, &ep_lw.ffn_norm, config.rms_norm_eps);
        let partial_mlp = v4_moe_mlp_ep(
            &mlp_input,
            &ep_lw.gate,
            &ep_lw.shared_experts,
            &ep_lw.local_experts,
            &shard,
            config,
        );

        // All-reduce MoE output (sum across ranks)
        // Shared expert is on every rank, so it will be summed world_size times.
        // We need to divide by world_size for the shared part, or better:
        // subtract (world_size-1) * shared_out after all-reduce.
        let full_mlp = if world_size > 1 {
            let partial_det = no_grad(|| partial_mlp.shallow_clone()).detach();
            nccl_smoke::all_reduce_tensor_f32_for_launch(
                &output_dir
                    .to_path_buf()
                    .join(format!("v4-ep-train-{rank}-{step}")),
                &partial_det,
            )?
        } else {
            partial_mlp.shallow_clone()
        };

        // Correct for shared expert being replicated: the all-reduce sums
        // shared_out from all ranks. Divide by world_size to correct.
        let mlp = if world_size > 1 {
            &full_mlp / (world_size as f64)
        } else {
            full_mlp
        };
        let mlp = no_grad(|| mlp).detach();
        let mut mlp = mlp.set_requires_grad(true);

        let layer_out = &residual + &mlp;

        // Final norm + lm_head
        let final_norm = tensor(&weights_gpu, "norm.weight")?.to_kind(kind);
        let normed = rms_norm(&layer_out, &final_norm, config.rms_norm_eps);
        let lm_head = tensor(&weights_gpu, "head.weight")?.to_kind(kind);
        let logits = normed.linear::<&Tensor>(&lm_head, None);

        // Loss
        let shifted = logits.narrow(1, 0, logits.size()[1] - 1);
        let targets = input_ids.narrow(1, 1, input_ids.size()[1] - 1);
        let loss = shifted
            .reshape([-1, config.vocab_size])
            .log_softmax(-1, Kind::Float)
            .g_nll_loss::<&Tensor>(&targets.reshape([-1]), None, Reduction::Mean, -100);

        let loss_val = loss.double_value(&[]);
        if step == 0 {
            initial_loss = loss_val;
        }
        info!(rank, step = step + 1, loss = loss_val, "EP train step");

        // ── Backward ──
        // Phase 1: backward through lm_head → mlp (autograd for local experts)
        loss.backward();

        // Phase 2: manual backward for local expert params
        // grad_mlp = d(loss)/d(mlp) — from mlp.grad()
        // aux_loss = (partial_mlp * grad_mlp).sum()
        // aux_loss.backward() → gradients for local expert params
        if world_size > 1 {
            let grad_mlp = mlp.grad();
            if grad_mlp.defined() {
                let grad_mlp_det = no_grad(|| grad_mlp.shallow_clone()).detach();
                // Scale by 1/world_size to match the forward correction
                let grad_mlp_scaled = &grad_mlp_det / (world_size as f64);
                let aux_loss = (&partial_mlp * &grad_mlp_scaled).sum(Kind::Float);
                aux_loss.backward();
            }
        }

        // ── Optimizer step (SGD) ──
        for (_, param) in trainable_params.iter_mut() {
            let grad = param.grad();
            if grad.defined() {
                let grad_norm = grad.norm().double_value(&[]);
                if grad_norm > 0.0 {
                    let _ = no_grad(|| param.f_sub_(&(grad * lr)));
                }
            }
            param.zero_grad();
        }

        // Sync trainable params back to weights_gpu
        for (name, param) in &trainable_params {
            weights_gpu.insert(name.clone(), param.shallow_clone());
        }

        // Clear mlp gradient
        if world_size > 1 {
            mlp.zero_grad();
        }
    }

    info!(rank, initial_loss, "V4 EP train complete");
    println!("rank={rank} ep_train_initial_loss={initial_loss:.9}");
    println!("rank={rank} ep_train_steps={num_steps}");
    Ok(())
}
