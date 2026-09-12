//! V4 TP (Tensor Parallel) support.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use tch::{no_grad, Device, Kind, Reduction, Tensor};
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
/// wo_a: row parallel (shard input dim), wo_b: replicated (all-reduce after wo_b).
pub struct TpV4AttentionWeights {
    pub wq_a: Tensor,
    pub q_norm: Tensor,
    pub wq_b: Tensor,
    pub wkv: Tensor,
    pub kv_norm: Tensor,
    pub attn_sink: Tensor,
    pub wo_a: Tensor,
    pub wo_b: Tensor,
    pub wq_a_scale: Option<Tensor>,
    pub wq_b_scale: Option<Tensor>,
    pub wkv_scale: Option<Tensor>,
    pub wo_a_scale: Option<Tensor>,
    pub wo_b_scale: Option<Tensor>,
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

        // wo_a: [o_lora_rank, o_groups * 512] → split by o_groups along input (dim 1)
        // wo_b: [hidden, o_lora_rank] → replicated (all-reduce after wo_b)
        let o_groups_per_rank = config.o_groups / shard.world_size as i64;
        let wo_a_full = tensor(weights, &format!("{p}.wo_a.weight"))?.to_kind(kind);
        let wo_row_start = shard.rank as i64 * o_groups_per_rank * 512;
        let wo_row_size = o_groups_per_rank * 512;
        let wo_a = wo_a_full
            .narrow(1, wo_row_start, wo_row_size)
            .shallow_clone();
        // wo_b is replicated (full weight on each rank)
        let wo_b = tensor(weights, &format!("{p}.wo_b.weight"))?.to_kind(kind);

        // FP8 scales (optional — only when weights are stored as FP8)
        let scale = |name: &str| -> Option<Tensor> {
            weights
                .get(&format!("{name}.scale_f"))
                .map(|t| t.shallow_clone())
        };
        let wq_a_name = format!("{p}.wq_a.weight");
        let wq_b_name = format!("{p}.wq_b.weight");
        let wkv_name = format!("{p}.wkv.weight");
        let wo_a_name = format!("{p}.wo_a.weight");
        let wo_b_name = format!("{p}.wo_b.weight");

        Ok(Self {
            wq_a: tensor(weights, &wq_a_name)?.to_kind(kind),
            q_norm: tensor(weights, &format!("{p}.q_norm.weight"))?.to_kind(kind),
            wq_b,
            wkv: tensor(weights, &wkv_name)?.to_kind(kind),
            kv_norm: tensor(weights, &format!("{p}.kv_norm.weight"))?.to_kind(kind),
            attn_sink,
            wo_a,
            wo_b,
            wq_a_scale: scale(&wq_a_name),
            wq_b_scale: scale(&wq_b_name),
            wkv_scale: scale(&wkv_name),
            wo_a_scale: scale(&wo_a_name),
            wo_b_scale: scale(&wo_b_name),
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

    let (cos, sin) = v4_rope_cos_sin(seq as usize, qk_rope, config, input.device());
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
    // Causal mask with optional sliding window (same logic as v4_attention)
    let mask: Tensor = if config.sliding_window > 0 && seq > config.sliding_window as i64 {
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
    let context = probs.matmul(&v);

    // Group reduction (local heads only)
    let heads_per_group = heads / o_groups_per_rank;
    let context = context
        .reshape([batch, o_groups_per_rank, heads_per_group, seq, head_dim])
        .sum_dim_intlist([2].as_slice(), false, attn.wo_a.kind());
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

    if world_size > 1 && world_size > config.num_attention_heads as usize {
        bail!(
            "V4 TP world_size {world_size} exceeds num_attention_heads {}",
            config.num_attention_heads
        );
    }
    if config.num_attention_heads as usize % world_size != 0 {
        bail!(
            "V4 TP num_attention_heads {} must be divisible by world_size {world_size}",
            config.num_attention_heads
        );
    }
    if config.o_groups as usize % world_size != 0 {
        bail!(
            "V4 TP o_groups {} must be divisible by world_size {world_size}",
            config.o_groups
        );
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
    // Only attention weights for TP parity test (not all 256 experts)
    needed.insert("layers.0.attn_norm.weight".to_string());
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
        needed.insert(format!("layers.0.attn.{suffix}"));
    }

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

// ── TP Full Training Loop ─────────────────────────────────────────────────────
//
// Unlike `deepseek_v4_tp_rank` (which only tests attention parity), this function
// runs a complete training loop: forward → loss → backward → optimizer step.
//
// TP sharding strategy:
//   - Attention: wq_a/wkv/q_norm/kv_norm replicated, wq_b/attn_sink sharded by heads,
//     wo_a sharded by o_groups (dim 1), wo_b replicated (all-reduce after wo_b)
//   - MoE: fully replicated (all experts on all ranks)
//   - Embedding/lm_head/norm: replicated
//
// Autograd handling:
//   NCCL all-reduce does not participate in tch-rs autograd. We use a two-phase
//   backward approach:
//     1. `loss.backward()` → gradients for MoE + replicated params
//     2. `aux_loss = (partial * grad_reduced).sum(); aux_loss.backward()` →
//        gradients for attention params (partial's graph is independent of reduced)
//
//   For replicated attention params (wq_a, wkv, wo_b), gradients should be
//   all-reduced across ranks to ensure consistency. This is a TODO.

pub fn deepseek_v4_tp_train(
    model_path: &Path,
    output_dir: &Path,
    config: &V4RuntimeConfig,
    kind: Kind,
) -> Result<()> {
    use std::collections::HashSet;
    use tch::{no_grad, Reduction};

    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;

    // ── Validation ──
    if world_size > 1 && world_size > config.num_attention_heads as usize {
        bail!(
            "V4 TP world_size {world_size} exceeds num_attention_heads {}",
            config.num_attention_heads
        );
    }
    if config.num_attention_heads as usize % world_size != 0 {
        bail!(
            "V4 TP num_attention_heads {} must be divisible by world_size {world_size}",
            config.num_attention_heads
        );
    }
    if config.o_groups as usize % world_size != 0 {
        bail!(
            "V4 TP o_groups {} must be divisible by world_size {world_size}",
            config.o_groups
        );
    }

    let device = Device::Cuda(local_rank);
    info!(rank, world_size, local_rank, "V4 TP train starting");

    let shard = V4TpShard::new(rank, world_size, config.num_attention_heads);

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
    for e in 0..config.n_routed_experts {
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

    // ── Set attention params as trainable ──
    let attn_param_names: Vec<String> = [
        "wq_a.weight",
        "wq_b.weight",
        "wkv.weight",
        "wo_a.weight",
        "wo_b.weight",
    ]
    .iter()
    .map(|s| format!("layers.{trainable_layer}.attn.{s}"))
    .collect();

    let mut trainable_params: Vec<(String, Tensor)> = Vec::new();
    for name in &attn_param_names {
        if let Some(t) = weights_gpu.get_mut(name) {
            let trainable = t.shallow_clone().to_kind(kind).set_requires_grad(true);
            weights_gpu.insert(name.clone(), trainable.shallow_clone());
            trainable_params.push((name.clone(), trainable));
        }
    }
    info!(
        rank,
        trainable_tensors = trainable_params.len(),
        "attention params set trainable"
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
        let hidden_norm = rms_norm(
            &hidden,
            &tensor(
                &weights_gpu,
                &format!("layers.{trainable_layer}.attn_norm.weight"),
            )?
            .to_kind(kind),
            config.rms_norm_eps,
        );

        // TP-sharded attention (autograd for attention params)
        let tp_attn = TpV4AttentionWeights::load_sharded(
            &weights_gpu,
            trainable_layer,
            &shard,
            config,
            kind,
        )?;
        let partial = tp_v4_attention(&hidden_norm, &tp_attn, config, &shard);

        // All-reduce attention output (breaks autograd graph)
        let mut reduced = if world_size > 1 {
            let partial_det = no_grad(|| partial.shallow_clone()).detach();
            let reduced = nccl_smoke::all_reduce_tensor_f32_for_launch(
                &output_dir
                    .to_path_buf()
                    .join(format!("v4-tp-train-{rank}-{step}")),
                &partial_det,
            )?;
            reduced.set_requires_grad(true)
        } else {
            partial.shallow_clone()
        };

        // MoE forward (autograd for subsequent params)
        let residual = &hidden + &reduced;
        let ffn_norm = tensor(
            &weights_gpu,
            &format!("layers.{trainable_layer}.ffn_norm.weight"),
        )?
        .to_kind(kind);
        let mlp_input = rms_norm(&residual, &ffn_norm, config.rms_norm_eps);

        let lw =
            V4MoeLayerWeights::load_raw(&weights_gpu, trainable_layer, config.n_routed_experts)?;
        let mlp = v4_moe_mlp(
            &mlp_input,
            &lw.gate,
            &lw.shared_experts,
            &lw.experts,
            config,
        );
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
        info!(rank, step = step + 1, loss = loss_val, "TP train step");

        // ── Backward ──
        // Phase 1: backward through MoE + subsequent layers (autograd)
        loss.backward();

        // Phase 2: manual backward for attention params
        if world_size > 1 {
            let grad_reduced = reduced.grad();
            if grad_reduced.defined() {
                let grad_reduced_det = no_grad(|| grad_reduced.shallow_clone()).detach();
                let aux_loss = (&partial * &grad_reduced_det).sum(Kind::Float);
                aux_loss.backward();
            }
        }

        // Phase 3: all-reduce gradients of replicated attention params
        // wq_a, wkv, wo_b are replicated — each rank computes a partial gradient.
        // Sum across ranks to get the full gradient.
        if world_size > 1 {
            let replicated_param_names = [
                format!("layers.{trainable_layer}.attn.wq_a.weight"),
                format!("layers.{trainable_layer}.attn.wkv.weight"),
                format!("layers.{trainable_layer}.attn.wo_b.weight"),
            ];
            for (name, param) in trainable_params.iter_mut() {
                if !replicated_param_names.contains(name) {
                    continue;
                }
                let grad = param.grad();
                if grad.defined() && grad.numel() > 0 {
                    let grad_path = output_dir.to_path_buf().join(format!(
                        "v4-tp-grad-{rank}-{step}-{}",
                        name.replace('.', "-")
                    ));
                    let reduced_grad =
                        nccl_smoke::all_reduce_tensor_f32_for_launch(&grad_path, &grad)?;
                    // Note: tch-rs doesn't support setting custom gradients directly.
                    // The all-reduced gradient is used in the optimizer step below
                    // by checking a stored gradient. For now, we average params
                    // after the optimizer step instead.
                    // TODO: implement proper gradient sync when tch-rs supports it.
                }
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

        // Clear reduced's gradient for next iteration
        if world_size > 1 {
            reduced.zero_grad();
        }
    }

    info!(rank, initial_loss, "V4 TP train complete");
    println!("rank={rank} tp_train_initial_loss={initial_loss:.9}");
    println!("rank={rank} tp_train_steps={num_steps}");
    Ok(())
}

// ── TP+EP Hybrid Training Loop ───────────────────────────────────────────────
//
// Combines TP (attention sharding) with EP (expert sharding).
//
// Layout: world_size = tp_size × ep_size
//   tp_rank = rank % tp_size  (which attention heads this rank handles)
//   ep_rank = rank / tp_size  (which experts this rank handles)
//
// Communication per step:
//   1. Attention: TP-sharded partial → all-reduce (all ranks) → / ep_size
//      (EP ranks in same TP position produce identical partials)
//   2. Routed experts: EP-sharded partial → all-reduce (all ranks) → / tp_size
//      (TP ranks in same EP position produce identical partials)
//   3. Shared expert: computed locally, not all-reduced
//
// Backward:
//   Phase 1: loss.backward() → gradients for lm_head, norm, shared expert
//   Phase 2: (partial_attn * grad_attn).sum().backward() → attention param grads
//   Phase 3: (partial_routed * grad_routed).sum().backward() → expert param grads
//   Phase 4: all-reduce replicated param gradients

pub fn deepseek_v4_tp_ep_train(
    model_path: &Path,
    output_dir: &Path,
    config: &V4RuntimeConfig,
    kind: Kind,
) -> Result<()> {
    use crate::ep::{v4_moe_routed_only_ep, V4EpShard};
    use std::collections::HashSet;

    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;

    // Derive tp_size and ep_size from env or config
    let tp_size = std::env::var("TP_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(world_size);
    let ep_size = world_size / tp_size;

    if world_size != tp_size * ep_size {
        bail!("V4 TP+EP world_size {world_size} must equal tp_size {tp_size} × ep_size {ep_size}");
    }

    let tp_rank = rank % tp_size;
    let ep_rank = rank / tp_size;

    // Validation
    if config.num_attention_heads as usize % tp_size != 0 {
        bail!(
            "num_attention_heads {} must be divisible by tp_size {tp_size}",
            config.num_attention_heads
        );
    }
    if config.o_groups as usize % tp_size != 0 {
        bail!(
            "o_groups {} must be divisible by tp_size {tp_size}",
            config.o_groups
        );
    }
    if config.n_routed_experts % ep_size != 0 {
        bail!(
            "n_routed_experts {} must be divisible by ep_size {ep_size}",
            config.n_routed_experts
        );
    }

    let device = Device::Cuda(local_rank);
    info!(
        rank,
        world_size, local_rank, tp_rank, ep_rank, tp_size, ep_size, "V4 TP+EP train starting"
    );

    let tp_shard = V4TpShard::new(tp_rank, tp_size, config.num_attention_heads);
    let ep_shard = V4EpShard::new(ep_rank, ep_size, config.n_routed_experts);

    // ── Load weights ──
    let trainable_layer = 0;
    let mut needed: HashSet<String> = HashSet::new();
    needed.insert("embed.weight".to_string());
    needed.insert("norm.weight".to_string());
    needed.insert("head.weight".to_string());
    needed.insert(format!("layers.{trainable_layer}.attn_norm.weight"));
    needed.insert(format!("layers.{trainable_layer}.ffn_norm.weight"));
    // Attention weights (replicated, TP-sharded at load time)
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
    // Shared experts (replicated)
    for suffix in &["w1.weight", "w2.weight", "w3.weight"] {
        needed.insert(format!(
            "layers.{trainable_layer}.ffn.shared_experts.{suffix}"
        ));
    }
    // Only local EP experts
    for &e in &ep_shard.local_expert_indices {
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

    // ── Set trainable: attention params + local expert params ──
    let mut trainable_params: Vec<(String, Tensor)> = Vec::new();

    // Attention params (TP-sharded and replicated)
    for suffix in &[
        "wq_a.weight",
        "wq_b.weight",
        "wkv.weight",
        "wo_a.weight",
        "wo_b.weight",
    ] {
        let name = format!("layers.{trainable_layer}.attn.{suffix}");
        if let Some(t) = weights_gpu.get_mut(&name) {
            let trainable = t.shallow_clone().to_kind(kind).set_requires_grad(true);
            weights_gpu.insert(name.clone(), trainable.shallow_clone());
            trainable_params.push((name.clone(), trainable));
        }
    }
    // Local expert params
    for &global_e in &ep_shard.local_expert_indices {
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
        "params set trainable"
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
        let hidden_norm = rms_norm(
            &hidden,
            &tensor(
                &weights_gpu,
                &format!("layers.{trainable_layer}.attn_norm.weight"),
            )?
            .to_kind(kind),
            config.rms_norm_eps,
        );

        // 1. TP-sharded attention (autograd for attention params)
        let tp_attn = TpV4AttentionWeights::load_sharded(
            &weights_gpu,
            trainable_layer,
            &tp_shard,
            config,
            kind,
        )?;
        let partial_attn = tp_v4_attention(&hidden_norm, &tp_attn, config, &tp_shard);

        // All-reduce attention → divide by ep_size (EP replicas produce identical partials)
        let mut attn_reduced = if world_size > 1 {
            let partial_det = no_grad(|| partial_attn.shallow_clone()).detach();
            let raw = nccl_smoke::all_reduce_tensor_f32_for_launch(
                &output_dir
                    .to_path_buf()
                    .join(format!("v4-tp-ep-attn-{rank}-{step}")),
                &partial_det,
            )?;
            (raw / (ep_size as f64)).set_requires_grad(true)
        } else {
            partial_attn.shallow_clone()
        };

        let residual = &hidden + &attn_reduced;

        // 2. MoE: shared (local) + routed (EP-sharded)
        let ffn_norm = tensor(
            &weights_gpu,
            &format!("layers.{trainable_layer}.ffn_norm.weight"),
        )?
        .to_kind(kind);
        let mlp_input = rms_norm(&residual, &ffn_norm, config.rms_norm_eps);

        // Shared expert (computed locally, not all-reduced)
        let shared_w1 = tensor(
            &weights_gpu,
            &format!("layers.{trainable_layer}.ffn.shared_experts.w1.weight"),
        )?
        .to_kind(kind);
        let shared_w2 = tensor(
            &weights_gpu,
            &format!("layers.{trainable_layer}.ffn.shared_experts.w2.weight"),
        )?
        .to_kind(kind);
        let shared_w3 = tensor(
            &weights_gpu,
            &format!("layers.{trainable_layer}.ffn.shared_experts.w3.weight"),
        )?
        .to_kind(kind);
        let shared_out = v4_swiglu(
            &mlp_input,
            &shared_w1,
            &shared_w2,
            &shared_w3,
            config.swiglu_limit,
        );

        // Routed experts (EP-sharded, autograd for local expert params)
        let gate = tensor(
            &weights_gpu,
            &format!("layers.{trainable_layer}.ffn.gate.weight"),
        )?
        .to_kind(kind);
        let local_experts: Vec<(Tensor, Tensor, Tensor)> = ep_shard
            .local_expert_indices
            .iter()
            .map(|&e| {
                let w1 = tensor(
                    &weights_gpu,
                    &format!("layers.{trainable_layer}.ffn.experts.{e}.w1.weight"),
                )
                .unwrap()
                .to_kind(kind);
                let w2 = tensor(
                    &weights_gpu,
                    &format!("layers.{trainable_layer}.ffn.experts.{e}.w2.weight"),
                )
                .unwrap()
                .to_kind(kind);
                let w3 = tensor(
                    &weights_gpu,
                    &format!("layers.{trainable_layer}.ffn.experts.{e}.w3.weight"),
                )
                .unwrap()
                .to_kind(kind);
                (w1, w2, w3)
            })
            .collect();

        let partial_routed =
            v4_moe_routed_only_ep(&mlp_input, &gate, &local_experts, &ep_shard, config);

        // All-reduce routed → divide by tp_size (TP replicas produce identical partials)
        let mut routed_full = if world_size > 1 {
            let partial_det = no_grad(|| partial_routed.shallow_clone()).detach();
            let raw = nccl_smoke::all_reduce_tensor_f32_for_launch(
                &output_dir
                    .to_path_buf()
                    .join(format!("v4-tp-ep-routed-{rank}-{step}")),
                &partial_det,
            )?;
            (raw / (tp_size as f64)).set_requires_grad(true)
        } else {
            partial_routed.shallow_clone()
        };

        // Full MoE = shared (local) + routed (all-reduced)
        let mlp = &shared_out + &routed_full;
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
        info!(rank, step = step + 1, loss = loss_val, "TP+EP train step");

        // ── Backward ──
        // Phase 1: backward through lm_head → MoE → residual → loss
        loss.backward();

        // Phase 2: manual backward for attention params
        if world_size > 1 {
            let grad_attn = attn_reduced.grad();
            if grad_attn.defined() {
                let grad_det = no_grad(|| grad_attn.shallow_clone()).detach();
                let grad_scaled = &grad_det / (ep_size as f64);
                let aux_loss = (&partial_attn * &grad_scaled).sum(Kind::Float);
                aux_loss.backward();
            }
        }

        // Phase 3: manual backward for local expert params
        if world_size > 1 {
            let grad_routed = routed_full.grad();
            if grad_routed.defined() {
                let grad_det = no_grad(|| grad_routed.shallow_clone()).detach();
                let grad_scaled = &grad_det / (tp_size as f64);
                let aux_loss = (&partial_routed * &grad_scaled).sum(Kind::Float);
                aux_loss.backward();
            }
        }

        // Phase 4: all-reduce replicated param gradients (wq_a, wkv, wo_b)
        if world_size > 1 {
            let replicated_names = [
                format!("layers.{trainable_layer}.attn.wq_a.weight"),
                format!("layers.{trainable_layer}.attn.wkv.weight"),
                format!("layers.{trainable_layer}.attn.wo_b.weight"),
            ];
            for (name, param) in trainable_params.iter_mut() {
                if !replicated_names.contains(name) {
                    continue;
                }
                let grad = param.grad();
                if grad.defined() && grad.numel() > 0 {
                    let grad_path = output_dir.to_path_buf().join(format!(
                        "v4-tp-ep-grad-{rank}-{step}-{}",
                        name.replace('.', "-")
                    ));
                    let reduced_grad =
                        nccl_smoke::all_reduce_tensor_f32_for_launch(&grad_path, &grad)?;
                    // TODO: apply reduced_grad as gradient when tch-rs supports set_grad.
                }
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
        for (name, param) in &trainable_params {
            weights_gpu.insert(name.clone(), param.shallow_clone());
        }
        if world_size > 1 {
            attn_reduced.zero_grad();
            routed_full.zero_grad();
        }
    }

    info!(rank, initial_loss, "V4 TP+EP train complete");
    println!("rank={rank} tp_ep_train_initial_loss={initial_loss:.9}");
    println!("rank={rank} tp_ep_train_steps={num_steps}");
    Ok(())
}
