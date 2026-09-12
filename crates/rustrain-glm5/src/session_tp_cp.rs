//! TP+CP+EP training loop for GLM-5.2.
//!
//! This is a focused implementation that reuses the weight loading, barrier,
//! NCCL, LoRA, and optimizer infrastructure from session_ep.rs, but replaces
//! the attention path with TP+CP sharding.
//!
//! The non-MTP legacy path uses a Cartesian TP×CP×EP decomposition. Native MTP
//! rejects combined TP+EP before entering this session because Megatron uses
//! separate dense and expert rank generators plus sequence parallelism.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use tch::{Device, Kind, Tensor, no_grad};
use tracing::info;

use crate::lora::*;
use crate::model::*;
use crate::model::{glm5_mlp, rms_norm};
use crate::session_ep::Glm5EpShard;
use crate::sft::*;
use crate::tp_cp::*;
use rustrain_checkpoint::safetensors::tensor;
use rustrain_nccl::nccl::{self as nccl_smoke, NcclPersistentComm};

struct CheckpointedCppMoeTpLayer {
    shared_gate: Tensor,
    shared_up: Tensor,
    shared_down: Tensor,
    shared_gate_scale: Option<Tensor>,
    shared_up_scale: Option<Tensor>,
    shared_down_scale: Option<Tensor>,
    gate: Tensor,
    correction_bias: Option<Tensor>,
    expert_gate: Vec<Tensor>,
    expert_up: Vec<Tensor>,
    expert_down: Vec<Tensor>,
    expert_gate_scales: Vec<Option<Tensor>>,
    expert_up_scales: Vec<Option<Tensor>>,
    expert_down_scales: Vec<Option<Tensor>>,
    expert_indices: Vec<usize>,
    n_routed_experts: i32,
    topk: i32,
    n_group: i32,
    topk_group: i32,
    scoring_func: i32,
    topk_method: i32,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
    tp_comm: usize,
    tp_size: i32,
    device_id: i32,
}

impl CheckpointedCppMoeTpLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let expert_gate = self.expert_gate.iter().collect::<Vec<_>>();
        let expert_up = self.expert_up.iter().collect::<Vec<_>>();
        let expert_down = self.expert_down.iter().collect::<Vec<_>>();
        let expert_gate_scales = self
            .expert_gate_scales
            .iter()
            .map(Option::as_ref)
            .collect::<Vec<_>>();
        let expert_up_scales = self
            .expert_up_scales
            .iter()
            .map(Option::as_ref)
            .collect::<Vec<_>>();
        let expert_down_scales = self
            .expert_down_scales
            .iter()
            .map(Option::as_ref)
            .collect::<Vec<_>>();
        rustrain_deepseek_v4::fp8_kernel::glm5_moe_layer_tp_cpp(
            input,
            &self.shared_gate,
            &self.shared_up,
            &self.shared_down,
            self.shared_gate_scale.as_ref(),
            self.shared_up_scale.as_ref(),
            self.shared_down_scale.as_ref(),
            &self.gate,
            self.correction_bias.as_ref(),
            &expert_gate,
            &expert_up,
            &expert_down,
            &expert_gate_scales,
            &expert_up_scales,
            &expert_down_scales,
            &self.expert_indices,
            self.n_routed_experts,
            self.topk,
            self.n_group,
            self.topk_group,
            self.scoring_func,
            self.topk_method,
            self.norm_topk_prob,
            self.routed_scaling_factor,
            self.tp_comm as *mut std::ffi::c_void,
            self.tp_size,
            self.device_id,
        )
    }
}

struct CheckpointRegistryGuard;

impl Drop for CheckpointRegistryGuard {
    fn drop(&mut self) {
        rustrain_deepseek_v4::fp8_kernel::clear_checkpoint_registry();
    }
}

fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

/// The current frozen trunk only has Megatron owner dispatch/return for the
/// EP-only path. Combining attention TP with EP would require sequence-
/// parallel token ownership plus the independent expert groups built below.
/// Reject it before loading checkpoint tensors or initializing NCCL.
pub fn validate_glm5_tp_ep_session_topology(tp_size: usize, ep_size: usize) -> Result<()> {
    if tp_size == 0 || ep_size == 0 {
        bail!("GLM-5 TP and EP sizes must be positive");
    }
    if tp_size > 1 && ep_size > 1 {
        bail!(
            "GLM-5 combined TP+EP is unsupported until the frozen trunk uses Megatron sequence-parallel owner dispatch/return with independent expert-EP and expert-DP groups (tp_size={tp_size}, ep_size={ep_size})"
        );
    }
    Ok(())
}

pub fn validate_glm5_staged_expert_topology(
    predequant_expert_weights: bool,
    tp_size: usize,
    cp_size: usize,
    ep_size: usize,
) -> Result<()> {
    if predequant_expert_weights {
        return Ok(());
    }
    if tp_size <= 1 || cp_size != 1 || ep_size != 1 {
        bail!(
            "GLM-5 CPU-staged expert tensor parallelism currently requires TP>1, CP=1, and EP=1 (tp_size={tp_size}, cp_size={cp_size}, ep_size={ep_size})"
        );
    }
    Ok(())
}

/// Whether a named full-sized LoRA variable still needs a TP SUM at step end.
/// WqA/Wkv are replicated projections whose downstream column-parallel input
/// mapping already sums their dgrad. WqB/Wo are locally sliced and need SUM to
/// reconstruct the full adapter gradient.
pub fn glm5_lora_gradient_requires_step_end_tp_sum(name: &str) -> Result<bool> {
    if name.contains("/WqA/") || name.contains("/Wkv/") {
        Ok(false)
    } else if name.contains("/WqB/") || name.contains("/Wo/") {
        Ok(true)
    } else {
        bail!("unrecognized GLM-5 LoRA variable name {name:?}")
    }
}

fn trainable_variable_names(registry: &Glm5LoraRegistry) -> Result<Vec<String>> {
    let names_by_storage = registry
        .var_store
        .variables()
        .into_iter()
        .map(|(name, tensor)| (tensor.data_ptr() as usize, name))
        .collect::<std::collections::HashMap<_, _>>();
    registry
        .var_store
        .trainable_variables()
        .iter()
        .map(|tensor| {
            names_by_storage
                .get(&(tensor.data_ptr() as usize))
                .cloned()
                .context("GLM-5 trainable LoRA variable has no stable VarStore name")
        })
        .collect()
}

/// Materialize one rank-local Megatron expert-TP tensor after its safetensors
/// file has loaded. This bounds transient full expert storage to one checkpoint
/// shard; the loader API does not stream individual tensors. `copy()` is
/// intentional because a contiguous dim-0 narrow can retain the full storage.
pub fn materialize_glm5_expert_tensor_for_tp(
    name: &str,
    tensor: Tensor,
    tp_rank: usize,
    tp_size: usize,
) -> Result<Tensor> {
    if tp_size == 1 || !name.contains(".mlp.experts.") {
        return Ok(tensor);
    }
    if tp_size == 0 || tp_rank >= tp_size {
        bail!("GLM-5 expert TP rank {tp_rank} is outside tensor parallel size {tp_size}");
    }
    let axis = if name.ends_with(".gate_proj.weight")
        || name.ends_with(".up_proj.weight")
        || name.ends_with(".gate_proj.weight_scale_inv")
        || name.ends_with(".up_proj.weight_scale_inv")
    {
        0
    } else if name.ends_with(".down_proj.weight") || name.ends_with(".down_proj.weight_scale_inv") {
        1
    } else {
        bail!("unsupported GLM-5 expert tensor name {name:?}");
    };
    let sizes = tensor.size();
    if sizes.len() != 2 {
        bail!("GLM-5 expert TP tensor {name:?} must be rank 2, got {sizes:?}");
    }
    let width = sizes[axis];
    let tp_size_i64 = i64::try_from(tp_size).context("GLM-5 expert TP size exceeds i64")?;
    if width % tp_size_i64 != 0 {
        bail!(
            "GLM-5 expert TP tensor {name:?} axis {axis} width {width} is not divisible by TP {tp_size}"
        );
    }
    let local = width / tp_size_i64;
    let start = i64::try_from(tp_rank).context("GLM-5 expert TP rank exceeds i64")? * local;
    Ok(tensor.narrow(axis as i64, start, local).copy())
}

fn keep_fp8(t: &Tensor, kind: Kind) -> Tensor {
    if t.kind() == Kind::Float8e4m3fn {
        t.shallow_clone()
    } else {
        t.to_kind(kind)
    }
}

fn reattach_cp_token_mean(
    local_sum: &Tensor,
    reduced_sum: &Tensor,
    reduced_count: &Tensor,
    cp_size: usize,
) -> Tensor {
    debug_assert!(cp_size > 0);
    let average_sum = reduced_sum / cp_size as f64;
    let average_count = reduced_count.clamp_min(1.0) / cp_size as f64;
    let visible_sum = local_sum + &(&average_sum - local_sum).detach();
    visible_sum / average_count
}

fn reattach_global_token_mean(
    local_sum: &Tensor,
    reduced_sum: &Tensor,
    reduced_count: &Tensor,
) -> Tensor {
    let visible_sum = local_sum + &(reduced_sum - local_sum).detach();
    visible_sum / reduced_count.clamp_min(1.0)
}

fn insert_glm5_dense_mlp_weights(needed: &mut HashSet<String>, prefix: &str) {
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        needed.insert(format!("{prefix}.{projection}.weight"));
        needed.insert(format!("{prefix}.{projection}.weight_scale_inv"));
    }
}

fn glm5_next_token_target_offset(global_offset: i64) -> Result<i32> {
    let target_offset = global_offset
        .checked_add(1)
        .context("GLM5 next-token target offset overflows i64")?;
    i32::try_from(target_offset).context("GLM5 next-token target offset exceeds i32")
}

#[allow(clippy::too_many_arguments)]
fn run_mtp_decoder_layer_tp_ep(
    input: &Tensor,
    weights_gpu: &BTreeMap<String, Tensor>,
    expert_weights_gpu: &BTreeMap<String, Tensor>,
    layer: usize,
    config: &Glm5RuntimeConfig,
    attn: &Glm5TpAttentionWeights,
    tp_shard: &Glm5TpShard,
    ep_shard: &Glm5EpShard,
    tp_comm: Option<&NcclPersistentComm>,
    cp_comm: Option<&NcclPersistentComm>,
    ep_comm: Option<&NcclPersistentComm>,
    tp_size: usize,
    cp_rank: usize,
    cp_size: usize,
    ep_size: usize,
) -> Result<Tensor> {
    use rustrain_deepseek_v4::fp8_kernel::{Glm5MtpDecoderDescriptor, glm5_mtp_decoder_layer_cpp};
    fn ptr(t: &Tensor) -> *mut std::ffi::c_void {
        t.as_ptr() as *mut _
    }
    fn opt(t: Option<&Tensor>) -> *mut std::ffi::c_void {
        t.map_or(std::ptr::null_mut(), ptr)
    }
    fn comm(t: Option<&NcclPersistentComm>, enabled: bool) -> *mut std::ffi::c_void {
        if enabled {
            t.map_or(std::ptr::null_mut(), NcclPersistentComm::raw_comm_ptr)
        } else {
            std::ptr::null_mut()
        }
    }
    let p = format!("model.layers.{layer}");
    if attn.indexer_wq_b.is_none()
        || attn.indexer_wk.is_none()
        || attn.indexer_k_norm_weight.is_none()
        || attn.indexer_k_norm_bias.is_none()
        || attn.indexer_weights_proj.is_none()
    {
        bail!("GLM-5 MTP decoder layer {layer} is missing its full DSA indexer weights");
    }
    let input_norm = tensor(weights_gpu, &format!("{p}.input_layernorm.weight"))?;
    let post_norm = tensor(weights_gpu, &format!("{p}.post_attention_layernorm.weight"))?;
    let is_moe = config.is_moe_layer(layer);
    let shared = format!("{p}.mlp.shared_experts");
    let shared_mlp = is_moe
        .then(|| {
            Glm5TpMlpWeights::load_sharded(
                weights_gpu,
                &shared,
                input.kind(),
                tp_shard.tp_rank,
                tp_shard.tp_size,
            )
        })
        .transpose()?;
    let dense_mlp = (!is_moe)
        .then(|| {
            Glm5TpMlpWeights::load_sharded(
                weights_gpu,
                &format!("{p}.mlp"),
                input.kind(),
                tp_shard.tp_rank,
                tp_shard.tp_size,
            )
        })
        .transpose()?;
    let mut eg = Vec::new();
    let mut eu = Vec::new();
    let mut ed = Vec::new();
    let mut egs = Vec::new();
    let mut eus = Vec::new();
    let mut eds = Vec::new();
    for &expert in &ep_shard.local_expert_indices {
        let q = format!("{p}.mlp.experts.{expert}");
        eg.push(ptr(tensor(
            expert_weights_gpu,
            &format!("{q}.gate_proj.weight"),
        )?));
        eu.push(ptr(tensor(
            expert_weights_gpu,
            &format!("{q}.up_proj.weight"),
        )?));
        ed.push(ptr(tensor(
            expert_weights_gpu,
            &format!("{q}.down_proj.weight"),
        )?));
        egs.push(opt(
            expert_weights_gpu.get(&format!("{q}.gate_proj.weight_scale_inv"))
        ));
        eus.push(opt(
            expert_weights_gpu.get(&format!("{q}.up_proj.weight_scale_inv"))
        ));
        eds.push(opt(
            expert_weights_gpu.get(&format!("{q}.down_proj.weight_scale_inv"))
        ));
    }
    let indices: Vec<i32> = ep_shard
        .local_expert_indices
        .iter()
        .map(|&i| i32::try_from(i).context("GLM-5 expert index exceeds C++ ABI"))
        .collect::<Result<_>>()?;
    let descriptor = Glm5MtpDecoderDescriptor {
        hidden: ptr(input),
        input_norm_weight: ptr(input_norm),
        post_norm_weight: ptr(post_norm),
        q_a_proj: ptr(&attn.q_a_proj),
        q_a_layernorm: ptr(&attn.q_a_layernorm),
        q_b_proj: ptr(&attn.q_b_proj),
        kv_a_proj: ptr(&attn.kv_a_proj_with_mqa),
        kv_a_layernorm: ptr(&attn.kv_a_layernorm),
        kv_b_proj: ptr(&attn.kv_b_proj),
        o_proj: ptr(&attn.o_proj),
        q_a_scale: opt(attn.q_a_proj_scale.as_ref()),
        q_b_scale: opt(attn.q_b_proj_scale.as_ref()),
        kv_a_scale: opt(attn.kv_a_proj_scale.as_ref()),
        kv_b_scale: opt(attn.kv_b_proj_scale.as_ref()),
        o_scale: opt(attn.o_proj_scale.as_ref()),
        idx_wq_b: opt(attn.indexer_wq_b.as_ref()),
        idx_wk: opt(attn.indexer_wk.as_ref()),
        idx_k_norm_w: opt(attn.indexer_k_norm_weight.as_ref()),
        idx_k_norm_b: opt(attn.indexer_k_norm_bias.as_ref()),
        idx_weights_proj: opt(attn.indexer_weights_proj.as_ref()),
        idx_weights_proj_scale: opt(attn.indexer_weights_proj_scale.as_ref()),
        idx_wq_b_scale: opt(attn.indexer_wq_b_scale.as_ref()),
        idx_wk_scale: opt(attn.indexer_wk_scale.as_ref()),
        gate_weight: opt(is_moe
            .then(|| tensor(weights_gpu, &format!("{p}.mlp.gate.weight")))
            .transpose()?),
        correction_bias: opt(weights_gpu.get(&format!("{p}.mlp.gate.e_score_correction_bias"))),
        shared_gate: opt(shared_mlp.as_ref().map(|mlp| &mlp.gate_proj)),
        shared_up: opt(shared_mlp.as_ref().map(|mlp| &mlp.up_proj)),
        shared_down: opt(shared_mlp.as_ref().map(|mlp| &mlp.down_proj)),
        shared_gate_scale: opt(shared_mlp
            .as_ref()
            .and_then(|mlp| mlp.gate_proj_scale.as_ref())),
        shared_up_scale: opt(shared_mlp
            .as_ref()
            .and_then(|mlp| mlp.up_proj_scale.as_ref())),
        shared_down_scale: opt(shared_mlp
            .as_ref()
            .and_then(|mlp| mlp.down_proj_scale.as_ref())),
        dense_gate: opt(dense_mlp.as_ref().map(|mlp| &mlp.gate_proj)),
        dense_up: opt(dense_mlp.as_ref().map(|mlp| &mlp.up_proj)),
        dense_down: opt(dense_mlp.as_ref().map(|mlp| &mlp.down_proj)),
        dense_gate_scale: opt(dense_mlp
            .as_ref()
            .and_then(|mlp| mlp.gate_proj_scale.as_ref())),
        dense_up_scale: opt(dense_mlp
            .as_ref()
            .and_then(|mlp| mlp.up_proj_scale.as_ref())),
        dense_down_scale: opt(dense_mlp
            .as_ref()
            .and_then(|mlp| mlp.down_proj_scale.as_ref())),
        expert_gate_weights: eg.as_mut_ptr(),
        expert_up_weights: eu.as_mut_ptr(),
        expert_down_weights: ed.as_mut_ptr(),
        expert_gate_scales: egs.as_mut_ptr(),
        expert_up_scales: eus.as_mut_ptr(),
        expert_down_scales: eds.as_mut_ptr(),
        local_expert_indices: indices.as_ptr(),
        tp_comm: comm(tp_comm, tp_size > 1),
        cp_comm: comm(cp_comm, cp_size > 1),
        ep_comm: comm(ep_comm, ep_size > 1),
        tp_size: tp_size as i32,
        cp_rank: cp_rank as i32,
        cp_size: cp_size as i32,
        ep_rank: ep_shard.rank as i32,
        ep_size: ep_size as i32,
        n_local_experts: indices.len() as i32,
        n_routed_experts: config.n_routed_experts as i32,
        topk: config.num_experts_per_tok as i32,
        n_group: config.n_group as i32,
        topk_group: config.topk_group as i32,
        scoring_func: match config.scoring_func.as_str() {
            "sigmoid" => 0,
            "softmax" => 1,
            other => bail!("unsupported GLM-5 scoring_func {other:?}"),
        },
        topk_method: match config.topk_method.as_str() {
            "groupwise" => 0,
            "noaux_tc" => 1,
            other => bail!("unsupported GLM-5 topk_method {other:?}"),
        },
        norm_topk_prob: i32::from(config.norm_topk_prob),
        is_moe_layer: i32::from(is_moe),
        num_heads: tp_shard.heads_per_rank as i32,
        qk_nope: config.qk_nope_head_dim as i32,
        qk_rope: config.qk_rope_head_dim as i32,
        v_head: config.v_head_dim as i32,
        kv_lora: config.kv_lora_rank as i32,
        idx_head_dim: config.index_head_dim as i32,
        idx_n_heads: tp_shard.idx_heads_per_rank as i32,
        idx_n_heads_global: config.index_n_heads as i32,
        idx_topk: config.index_topk as i32,
        rope_interleave: i32::from(config.rope_interleave),
        indexer_rope_interleave: i32::from(config.indexer_rope_interleave),
        rms_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        routed_scaling_factor: config.routed_scaling_factor,
        rope_scaling_factor: config.rope_scaling_factor,
        rope_beta_fast: config.rope_beta_fast,
        rope_beta_slow: config.rope_beta_slow,
        rope_attention_factor: config.rope_attention_factor,
        rope_original_max_pos: config.rope_original_max_pos,
        rope_is_yarn: i32::from(config.rope_type == "yarn"),
    };
    glm5_mtp_decoder_layer_cpp(&descriptor)
}

#[allow(clippy::too_many_arguments)]
fn run_mtp_decoder_layer_tp_ep_rust_fallback(
    input: &Tensor,
    weights_gpu: &BTreeMap<String, Tensor>,
    expert_weights_gpu: &BTreeMap<String, Tensor>,
    layer: usize,
    config: &Glm5RuntimeConfig,
    tp_shard: &Glm5TpShard,
    ep_shard: &Glm5EpShard,
    tp_comm: Option<&NcclPersistentComm>,
    cp_comm: Option<&NcclPersistentComm>,
    ep_comm: Option<&NcclPersistentComm>,
    tp_size: usize,
    cp_rank: usize,
    cp_size: usize,
    ep_size: usize,
    compute_kind: Kind,
) -> Result<Tensor> {
    let p = format!("model.layers.{layer}");
    let attn_norm =
        tensor(weights_gpu, &format!("{p}.input_layernorm.weight"))?.to_kind(compute_kind);
    let hidden_norm = rms_norm(input, &attn_norm, config.rms_norm_eps);
    let attn_weights =
        Glm5TpAttentionWeights::load_sharded(weights_gpu, layer, compute_kind, tp_shard, config)?;

    // The MTP decoder is an independent full DSA layer. Never inherit the
    // trunk's IndexShare state, even when the checkpoint's last trunk layer is
    // shared.
    let mut mtp_index_state: Option<IndexShareState> = None;
    let mut mtp_config = config.clone();
    mtp_config.index_topk_freq = 1;
    mtp_config.index_skip_topk_offset = 0;
    while mtp_config.indexer_types.len() <= layer {
        mtp_config.indexer_types.push("full".to_string());
    }
    mtp_config.indexer_types[layer] = "full".to_string();
    let attn_out = glm5_dsa_attention_tp_cp(
        &hidden_norm,
        &attn_weights,
        &attn_weights,
        &mtp_config,
        &mut mtp_index_state,
        layer,
        tp_shard,
        cp_rank,
        cp_size,
        tp_comm,
        cp_comm,
    )
    .to_kind(compute_kind);
    let attn_out = if tp_size > 1 {
        let detached = no_grad(|| attn_out.shallow_clone()).detach();
        let reduced = tp_comm
            .context("MTP TP communicator missing")?
            .all_reduce(&detached)?;
        let full = reduced.to_kind(compute_kind);
        &attn_out + &(&full - &attn_out).detach()
    } else {
        attn_out
    };

    let residual = input + &attn_out;
    let post_norm =
        tensor(weights_gpu, &format!("{p}.post_attention_layernorm.weight"))?.to_kind(compute_kind);
    let mlp_input = rms_norm(&residual, &post_norm, config.rms_norm_eps);

    if config.is_moe_layer(layer) {
        let gate = tensor(weights_gpu, &format!("{p}.mlp.gate.weight"))?.to_kind(compute_kind);
        let correction_bias = weights_gpu.get(&format!("{p}.mlp.gate.e_score_correction_bias"));
        let shared_gate = keep_fp8(
            tensor(
                weights_gpu,
                &format!("{p}.mlp.shared_experts.gate_proj.weight"),
            )?,
            compute_kind,
        );
        let shared_up = keep_fp8(
            tensor(
                weights_gpu,
                &format!("{p}.mlp.shared_experts.up_proj.weight"),
            )?,
            compute_kind,
        );
        let shared_down = keep_fp8(
            tensor(
                weights_gpu,
                &format!("{p}.mlp.shared_experts.down_proj.weight"),
            )?,
            compute_kind,
        );
        let shared_output = glm5_mlp_fp8(
            &mlp_input,
            &shared_gate,
            &shared_up,
            &shared_down,
            weights_gpu.get(&format!(
                "{p}.mlp.shared_experts.gate_proj.weight_scale_inv"
            )),
            weights_gpu.get(&format!("{p}.mlp.shared_experts.up_proj.weight_scale_inv")),
            weights_gpu.get(&format!(
                "{p}.mlp.shared_experts.down_proj.weight_scale_inv"
            )),
        );

        let router_logits = mlp_input
            .linear::<&Tensor>(&gate, None)
            .to_kind(Kind::Float);
        let k = config.num_experts_per_tok as i64;
        let (topk_weights, topk_indices) = glm5_router_topk(
            &router_logits,
            correction_bias,
            config.num_experts_per_tok,
            &config.scoring_func,
            &config.topk_method,
            config.n_group,
            config.topk_group,
            config.norm_topk_prob,
            config.routed_scaling_factor,
        );
        let flat_input = mlp_input.reshape([-1, mlp_input.size()[2]]);
        let tk_indices = topk_indices.reshape([-1, k]);
        let tk_weights = topk_weights.reshape([-1, k]);
        let mut routed_partial =
            Tensor::zeros(flat_input.size(), (compute_kind, flat_input.device()));

        for &global_e in &ep_shard.local_expert_indices {
            let mask = tk_indices.eq(global_e as i64).to_kind(compute_kind);
            if mask.sum(Kind::Int64).int64_value(&[]) == 0 {
                continue;
            }
            let ep = format!("{p}.mlp.experts.{global_e}");
            let gate_w = expert_weights_gpu
                .get(&format!("{ep}.gate_proj.weight"))
                .with_context(|| format!("MTP expert {global_e} gate weight missing"))?;
            let up_w = expert_weights_gpu
                .get(&format!("{ep}.up_proj.weight"))
                .with_context(|| format!("MTP expert {global_e} up weight missing"))?;
            let down_w = expert_weights_gpu
                .get(&format!("{ep}.down_proj.weight"))
                .with_context(|| format!("MTP expert {global_e} down weight missing"))?;
            let expert_out = glm5_mlp_fp8(
                &flat_input,
                gate_w,
                up_w,
                down_w,
                expert_weights_gpu.get(&format!("{ep}.gate_proj.weight_scale_inv")),
                expert_weights_gpu.get(&format!("{ep}.up_proj.weight_scale_inv")),
                expert_weights_gpu.get(&format!("{ep}.down_proj.weight_scale_inv")),
            );
            let weighted_mask = (mask * &tk_weights)
                .sum_dim_intlist([-1].as_slice(), false, compute_kind)
                .unsqueeze(-1);
            routed_partial += &(expert_out * weighted_mask);
        }

        let routed_partial = routed_partial.reshape([1, -1, mlp_input.size()[2]]);
        let routed_full = if ep_size > 1 {
            let detached = no_grad(|| routed_partial.shallow_clone()).detach();
            let reduced = ep_comm
                .context("MTP EP communicator missing")?
                .all_reduce(&detached)?;
            let full = reduced.to_kind(compute_kind);
            &routed_partial + &(&full - &routed_partial).detach()
        } else {
            routed_partial
        };
        Ok(&residual + routed_full + shared_output)
    } else {
        let gate = keep_fp8(
            tensor(weights_gpu, &format!("{p}.mlp.gate_proj.weight"))?,
            compute_kind,
        );
        let up = keep_fp8(
            tensor(weights_gpu, &format!("{p}.mlp.up_proj.weight"))?,
            compute_kind,
        );
        let down = keep_fp8(
            tensor(weights_gpu, &format!("{p}.mlp.down_proj.weight"))?,
            compute_kind,
        );
        let mlp = glm5_mlp_fp8(
            &mlp_input,
            &gate,
            &up,
            &down,
            weights_gpu.get(&format!("{p}.mlp.gate_proj.weight_scale_inv")),
            weights_gpu.get(&format!("{p}.mlp.up_proj.weight_scale_inv")),
            weights_gpu.get(&format!("{p}.mlp.down_proj.weight_scale_inv")),
        );
        Ok(&residual + mlp)
    }
}

#[derive(serde::Serialize)]
pub struct TpCpEpSummary {
    pub adapter_output: String,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub trainable_params: usize,
}

pub fn train_glm5_lora_sft_tp_cp_ep(
    config: &rustrain_core::runtime::Config,
    run_paths: &rustrain_core::runtime::RunPaths,
) -> Result<TpCpEpSummary> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;

    // ── Parallelism decomposition ──
    if world_size == 0 {
        bail!("GLM-5 TP+CP+EP: WORLD_SIZE must be positive");
    }
    if rank >= world_size {
        bail!("GLM-5 rank {rank} must be smaller than WORLD_SIZE {world_size}");
    }
    let tp_size = config.parallel.tensor_model_parallel_size.max(1);
    let cp_size = config.parallel.context_parallel_size.max(1);
    let ep_size = config.parallel.expert_model_parallel_size.max(1);
    validate_glm5_tp_ep_session_topology(tp_size, ep_size)?;
    validate_glm5_staged_expert_topology(
        config.train.predequant_expert_weights,
        tp_size,
        cp_size,
        ep_size,
    )?;

    // ── Model config ──
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("GLM-5 TP+CP+EP requires model.model_path")?;
    let model_path = std::path::PathBuf::from(model_path);
    let runtime_config = read_glm5_config(&model_path.join("config.json"))?;

    let cpp_router_supported =
        matches!(runtime_config.scoring_func.as_str(), "sigmoid" | "softmax")
            && matches!(
                runtime_config.topk_method.as_str(),
                "groupwise" | "noaux_tc"
            )
            && (runtime_config.topk_method != "noaux_tc"
                || runtime_config.scoring_func == "sigmoid");
    let use_cpp_moe =
        rustrain_deepseek_v4::fp8_kernel::is_glm5_attention_available() && cpp_router_supported;
    if tp_size > 1 && !use_cpp_moe {
        bail!(
            "GLM-5 expert TP requires the compiled C++ MoE kernel and a supported router (scoring_func={:?}, topk_method={:?})",
            runtime_config.scoring_func,
            runtime_config.topk_method
        );
    }

    validate_glm5_mtp_distributed_contract(
        runtime_config.num_nextn_predict_layers,
        tp_size,
        cp_size,
        ep_size,
    )?;

    // With CP=1, Megatron overlays two rank generators on the same ranks:
    // dense coordinates are TP x dense-DP while expert coordinates are
    // ETP(=1) x EP x expert-DP. EP is therefore not rank/(TP*CP).
    let megatron_coords = if cp_size == 1 {
        Some(glm5_megatron_rank_coordinates(
            rank, world_size, tp_size, ep_size,
        )?)
    } else {
        let product = tp_size
            .checked_mul(cp_size)
            .and_then(|value| value.checked_mul(ep_size))
            .context("GLM-5 legacy TP×CP×EP topology overflows usize")?;
        if world_size != product {
            bail!(
                "GLM-5 CP topology has no data-parallel rank generator: WORLD_SIZE {world_size} must equal TP {tp_size} × CP {cp_size} × EP {ep_size} = {product}"
            );
        }
        None
    };
    let (tp_rank, cp_rank, ep_rank, dense_dp_rank, dense_dp_size, expert_dp_rank, expert_dp_size) =
        if let Some(coords) = megatron_coords {
            (
                coords.tp_rank,
                0,
                coords.ep_rank,
                coords.dense_dp_rank,
                coords.dense_dp_size,
                coords.expert_dp_rank,
                coords.expert_dp_size,
            )
        } else {
            (
                rank % tp_size,
                (rank / tp_size) % cp_size,
                rank / (tp_size * cp_size),
                0,
                1,
                0,
                1,
            )
        };
    if runtime_config.num_nextn_predict_layers > 0 {
        let required_seq_len = runtime_config
            .num_nextn_predict_layers
            .checked_add(2)
            .context("GLM-5 MTP required sequence length overflows usize")?;
        if config.model.seq_len < required_seq_len {
            bail!(
                "GLM-5 {}-layer MTP requires model.seq_len >= {}",
                runtime_config.num_nextn_predict_layers,
                required_seq_len
            );
        }
    }

    // Validate
    if runtime_config.num_attention_heads as usize % tp_size != 0 {
        bail!(
            "num_attention_heads {} must be divisible by tp_size {tp_size}",
            runtime_config.num_attention_heads
        );
    }
    if runtime_config.index_n_heads as usize > 0
        && (runtime_config.index_n_heads as usize) < tp_size
    {
        bail!(
            "index_n_heads {} must be at least tp_size {tp_size}",
            runtime_config.index_n_heads
        );
    }
    if runtime_config.index_n_heads as usize % tp_size != 0 {
        bail!(
            "index_n_heads {} must be divisible by tp_size {tp_size}",
            runtime_config.index_n_heads
        );
    }
    for (name, width) in [
        (
            "q_b_proj rows",
            runtime_config.qk_nope_head_dim + runtime_config.qk_rope_head_dim,
        ),
        (
            "kv_b_proj rows",
            runtime_config.qk_nope_head_dim + runtime_config.v_head_dim,
        ),
        ("o_proj columns", runtime_config.v_head_dim),
    ] {
        let shard = width * (runtime_config.num_attention_heads / tp_size as i64);
        if shard % 128 != 0 {
            bail!("{name} per-TP shard {shard} must align to FP8 block size 128");
        }
    }
    let indexer_shard =
        runtime_config.index_head_dim * (runtime_config.index_n_heads / tp_size as i64);
    if indexer_shard % 128 != 0 {
        bail!("indexer wq_b per-TP shard {indexer_shard} must align to FP8 block size 128");
    }
    if runtime_config.n_routed_experts % ep_size != 0 {
        bail!(
            "n_routed_experts {} must be divisible by ep_size {ep_size}",
            runtime_config.n_routed_experts
        );
    }
    if runtime_config.index_topk_freq <= 0 {
        bail!("GLM-5 TP+CP+EP: index_topk_freq must be positive");
    }
    if runtime_config.num_experts_per_tok <= 0
        || runtime_config.num_experts_per_tok > runtime_config.n_routed_experts
    {
        bail!(
            "GLM-5 TP+CP+EP: invalid num_experts_per_tok {} for {} experts",
            runtime_config.num_experts_per_tok,
            runtime_config.n_routed_experts
        );
    }
    let experts_per_group = runtime_config.n_routed_experts / runtime_config.n_group;
    if runtime_config.num_experts_per_tok > runtime_config.topk_group * experts_per_group {
        bail!(
            "num_experts_per_tok {} exceeds selected router capacity {}",
            runtime_config.num_experts_per_tok,
            runtime_config.topk_group * experts_per_group
        );
    }
    if config.model.seq_len == 0 || config.model.seq_len % cp_size != 0 {
        bail!(
            "GLM-5 TP+CP+EP: model.seq_len {} must be divisible by cp_size {cp_size}",
            config.model.seq_len
        );
    }

    info!(
        rank,
        world_size,
        local_rank,
        tp_rank,
        cp_rank,
        ep_rank,
        dense_dp_rank,
        dense_dp_size,
        expert_dp_rank,
        expert_dp_size,
        tp_size,
        cp_size,
        ep_size,
        layers = runtime_config.num_hidden_layers,
        "GLM-5.2 TP+CP+EP config loaded"
    );

    // ── EP shard ──
    let ep_shard = Glm5EpShard::new(ep_rank, ep_size, runtime_config.n_routed_experts);

    // ── TP shard ──
    let tp_shard = Glm5TpShard::new(
        tp_rank,
        tp_size,
        runtime_config.num_attention_heads,
        runtime_config.index_n_heads,
    );

    // ── LoRA config ──
    let lora_config_raw = config
        .lora
        .as_ref()
        .context("GLM-5 LoRA SFT requires [lora] config section")?;
    let trainable_layer_indices: Vec<usize> = lora_config_raw
        .target_layers
        .iter()
        .map(|l| *l as usize)
        .collect();
    if let Some(layer) = trainable_layer_indices
        .iter()
        .find(|&&layer| layer >= runtime_config.num_hidden_layers)
    {
        bail!(
            "LoRA target layer {layer} is outside the frozen trunk range 0..{}; native MTP decoder weights are not trainable",
            runtime_config.num_hidden_layers
        );
    }
    let target_modules: Vec<Glm5LoraTargetModule> = lora_config_raw
        .target_modules
        .iter()
        .map(|s| Glm5LoraTargetModule::from_name(s))
        .collect::<Result<Vec<_>>>()?;
    let lora_config = Glm5LoraConfig {
        rank: lora_config_raw.rank,
        alpha: lora_config_raw.alpha as i64,
        target_layers: trainable_layer_indices.clone(),
        target_modules,
    };

    let compute_kind = match config.train.dtype {
        rustrain_core::runtime::DType::Bf16 => Kind::BFloat16,
        _ => Kind::Float,
    };
    let device = Device::Cuda(local_rank);

    // ── Staggered loading ──
    if rank > 0 {
        std::thread::sleep(std::time::Duration::from_secs((rank * 5) as u64));
    }

    // ── Build needed weight set ──
    let n_layers = runtime_config.num_hidden_layers;
    let mut needed: HashSet<String> = HashSet::new();
    needed.insert("model.embed_tokens.weight".to_string());
    needed.insert("model.norm.weight".to_string());
    if !runtime_config.tie_word_embeddings {
        needed.insert("lm_head.weight".to_string());
        needed.insert("lm_head.weight_scale_inv".to_string());
    }

    let total_decoder_layers = n_layers + runtime_config.num_nextn_predict_layers;
    for layer in 0..total_decoder_layers {
        let p = format!("model.layers.{layer}");
        needed.insert(format!("{p}.input_layernorm.weight"));
        needed.insert(format!("{p}.post_attention_layernorm.weight"));
        for suffix in &[
            "q_a_proj.weight",
            "q_a_layernorm.weight",
            "q_b_proj.weight",
            "kv_a_proj_with_mqa.weight",
            "kv_a_layernorm.weight",
            "kv_b_proj.weight",
            "o_proj.weight",
        ] {
            needed.insert(format!("{p}.self_attn.{suffix}"));
            needed.insert(format!("{p}.self_attn.{suffix}_scale_inv"));
        }
        let indexer_type = runtime_config
            .indexer_types
            .get(layer)
            .map(|s| s.as_str())
            .unwrap_or("full");
        if indexer_type == "full" {
            for suffix in &[
                "k_norm.weight",
                "k_norm.bias",
                "weights_proj.weight",
                "wk.weight",
                "wq_b.weight",
            ] {
                needed.insert(format!("{p}.self_attn.indexer.{suffix}"));
                if matches!(*suffix, "weights_proj.weight" | "wk.weight" | "wq_b.weight") {
                    needed.insert(format!("{p}.self_attn.indexer.{suffix}_scale_inv"));
                }
            }
        }
        needed.insert(format!("{p}.mlp.gate.weight"));
        if runtime_config.topk_method == "noaux_tc" {
            needed.insert(format!("{p}.mlp.gate.e_score_correction_bias"));
        }
        for suffix in &["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            needed.insert(format!("{p}.mlp.shared_experts.{suffix}"));
            needed.insert(format!("{p}.mlp.shared_experts.{suffix}_scale_inv"));
        }
        for &e in &ep_shard.local_expert_indices {
            for suffix in &["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
                needed.insert(format!("{p}.mlp.experts.{e}.{suffix}"));
                needed.insert(format!("{p}.mlp.experts.{e}.{suffix}_scale_inv"));
            }
        }
        if !runtime_config.is_moe_layer(layer) {
            insert_glm5_dense_mlp_weights(&mut needed, &format!("{p}.mlp"));
        }
    }
    for mtp_layer in glm5_mtp_layer_indices(&runtime_config)? {
        needed.extend(Glm5MtpProjectionWeights::weight_names(mtp_layer));
    }

    // ── Load weights ──
    let weights = load_glm5_weights_shared(&model_path, &needed, tp_rank, tp_size)?;
    info!(rank, tensors = weights.len(), "weights loaded");

    // Expert offloading: experts on CPU, rest on GPU
    let mut weights_gpu: BTreeMap<String, Tensor> = BTreeMap::new();
    let mut expert_weights_cpu: BTreeMap<String, Tensor> = BTreeMap::new();
    for (name, t) in weights {
        if name.contains(".mlp.experts.") {
            expert_weights_cpu.insert(name, t);
        } else {
            let t = if t.kind() == Kind::Float8e4m3fn || t.kind() == Kind::Float {
                t.to_device(device)
            } else {
                t.to_device(device).to_kind(compute_kind)
            };
            weights_gpu.insert(name, t);
        }
    }
    info!(
        rank,
        tensors_on_gpu = weights_gpu.len(),
        expert_tensors_on_cpu = expert_weights_cpu.len(),
        "weights loaded (TP+CP+EP, experts offloaded)"
    );

    // Megatron defaults expert tensor parallelism to TP. The per-file loader
    // materialized only this rank's gate/up rows and down columns, so this final
    // map contains no full routed-expert tensors. Staged mode keeps rank-local
    // FP8 shards on CPU; eager mode moves only those shards to GPU.
    let expert_weights_tp = expert_weights_cpu;
    let mut expert_weights_runtime = if config.train.predequant_expert_weights {
        expert_weights_tp
            .into_iter()
            .map(|(name, tensor)| {
                let tensor = if tensor.kind() == Kind::Float8e4m3fn || tensor.kind() == Kind::Float
                {
                    tensor.to_device(device)
                } else {
                    tensor.to_device(device).to_kind(compute_kind)
                };
                (name, tensor)
            })
            .collect::<BTreeMap<_, _>>()
    } else {
        expert_weights_tp
    };
    if config.train.predequant_expert_weights {
        let weight_names = expert_weights_runtime.keys().cloned().collect::<Vec<_>>();
        for name in weight_names {
            let Some(weight) = expert_weights_runtime.get(&name) else {
                continue;
            };
            if weight.kind() != Kind::Float8e4m3fn || !name.ends_with(".weight") {
                continue;
            }
            let scale_name = name.replace(".weight", ".weight_scale_inv");
            let scale = expert_weights_runtime
                .get(&scale_name)
                .with_context(|| format!("FP8 expert TP weight {name:?} is missing its scale"))?;
            let dequantized = rustrain_deepseek_v4::fp8_kernel::dequant_fp8_weight(weight, scale)
                .with_context(|| format!("failed to pre-dequantize expert TP weight {name:?}"))?
                .to_kind(compute_kind);
            expert_weights_runtime.insert(name, dequantized);
        }
    }
    info!(
        rank,
        expert_tensor_count = expert_weights_runtime.len(),
        expert_tensor_parallel_size = tp_size,
        residency = if config.train.predequant_expert_weights {
            "gpu_bf16_tp_shard"
        } else {
            "cpu_fp8_tp_shard_staged"
        },
        predequant_fp8 = config.train.predequant_expert_weights,
        "expert tensor-parallel residency initialized"
    );

    let tp_vocab = Glm5TpVocabWeights::load_sharded(
        &weights_gpu,
        compute_kind,
        runtime_config.vocab_size,
        runtime_config.tie_word_embeddings,
        tp_rank,
        tp_size,
    )?;

    // ── LoRA registry ──
    // For TP, LoRA is on the local shard of attention weights.
    // We load full attention weights, create LoRA on them, then narrow.
    // Simpler: create LoRA registry on full weights, then narrow in forward.
    let registry = Glm5LoraRegistry::new(&weights_gpu, lora_config, device)?;
    let trainable_count = registry.var_store.trainable_variables().len();
    info!(
        rank,
        trainable_params = trainable_count,
        "LoRA adapters created"
    );

    // ── Barrier ──
    let barrier_dir = config.run.base_dir.join(&config.run.name).join("barrier");
    std::fs::create_dir_all(&barrier_dir)?;
    let ready_file = barrier_dir.join(format!("rank_{rank}.ready"));
    std::fs::write(&ready_file, b"ready")?;
    info!(rank, "waiting at barrier");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
    loop {
        let ready_count = std::fs::read_dir(&barrier_dir)
            .map(|d| {
                d.filter_map(|e| e.ok())
                    .filter(|e| e.file_name().to_string_lossy().starts_with("rank_"))
                    .count()
            })
            .unwrap_or(0);
        if ready_count >= world_size {
            break;
        }
        if std::time::Instant::now() > deadline {
            bail!("barrier timeout: {ready_count}/{world_size}");
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    info!(rank, "all ranks ready");

    // ── NCCL communication domains ──
    // Each parallel axis must use its own communicator. For CP=1 these are
    // Megatron's independent dense and expert rank generators, not Cartesian
    // TP x EP subgroups.
    let comm_root = config.run.base_dir.join(&config.run.name).join("nccl-comm");
    let world_comm = if world_size > 1 {
        Some(NcclPersistentComm::new_group(
            &comm_root.join("world"),
            rank,
            world_size,
            local_rank,
        )?)
    } else {
        None
    };
    let tp_comm = if tp_size > 1 {
        let (group_name, group_rank) = if megatron_coords.is_some() {
            (format!("tp-dense-dp{dense_dp_rank}"), tp_rank)
        } else {
            (format!("tp-ep{ep_rank}-cp{cp_rank}"), tp_rank)
        };
        Some(NcclPersistentComm::new_group(
            &comm_root.join(group_name),
            group_rank,
            tp_size,
            local_rank,
        )?)
    } else {
        None
    };
    let cp_comm = if cp_size > 1 {
        Some(NcclPersistentComm::new_group(
            &comm_root.join(format!("cp-ep{ep_rank}-tp{tp_rank}")),
            cp_rank,
            cp_size,
            local_rank,
        )?)
    } else {
        None
    };
    let ep_comm = if ep_size > 1 {
        let (group_name, group_rank) = if megatron_coords.is_some() {
            (format!("expert-ep-edp{expert_dp_rank}"), ep_rank)
        } else {
            (format!("ep-cp{cp_rank}-tp{tp_rank}"), ep_rank)
        };
        Some(NcclPersistentComm::new_group(
            &comm_root.join(group_name),
            group_rank,
            ep_size,
            local_rank,
        )?)
    } else {
        None
    };
    let dense_dp_comm = if dense_dp_size > 1 {
        Some(NcclPersistentComm::new_group(
            &comm_root.join(format!("dense-dp-tp{tp_rank}")),
            dense_dp_rank,
            dense_dp_size,
            local_rank,
        )?)
    } else {
        None
    };
    // Routed experts are frozen in this LoRA session. Still create their
    // parameter-replica group so a future trainable-expert path cannot reuse
    // dense DP or EP by accident.
    let _expert_dp_comm = if expert_dp_size > 1 {
        Some(NcclPersistentComm::new_group(
            &comm_root.join(format!("expert-dp-ep{ep_rank}")),
            expert_dp_rank,
            expert_dp_size,
            local_rank,
        )?)
    } else {
        None
    };
    info!(
        rank,
        tp_size,
        cp_size,
        ep_size,
        dense_dp_size,
        expert_dp_size,
        "NCCL communication domains created"
    );

    // ── SFT data ──
    let tokenizer = tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let sft_jsonl = std::path::Path::new("data/sft/deepseek_test.jsonl");
    let train_dataset = if sft_jsonl.exists() {
        Glm5SftDataset::from_jsonl_simple(sft_jsonl, &tokenizer)?
    } else {
        Glm5SftDataset::synthetic(&tokenizer)?
    };
    if train_dataset.samples.is_empty() {
        bail!("GLM-5 SFT dataset contains no samples");
    }
    // Pad to config seq_len, then slice for CP
    let mtp_enabled = runtime_config.num_nextn_predict_layers > 0;
    let accumulation_steps = if mtp_enabled {
        config.train.gradient_accumulation_steps
    } else {
        1
    };
    let micro_batch_size = if mtp_enabled {
        config.train.micro_batch_size
    } else {
        1
    };
    if accumulation_steps == 0 || micro_batch_size == 0 {
        bail!("GLM-5 MTP requires positive micro-batch and accumulation sizes");
    }
    let accumulation_batch_size = micro_batch_size
        .checked_mul(accumulation_steps)
        .context("GLM-5 MTP accumulation batch size overflows usize")?;
    let distributed_batch_size = accumulation_batch_size
        .checked_mul(dense_dp_size)
        .context("GLM-5 dense-DP accumulation batch size overflows usize")?;
    let accumulation_dataset = Glm5SftDataset {
        samples: (0..distributed_batch_size)
            .map(|index| train_dataset.samples[index % train_dataset.samples.len()].clone())
            .collect(),
        pad_token_id: train_dataset.pad_token_id,
    };
    // Ranks in one dense TP group share a sample; different dense-DP ranks own
    // disjoint sample slots. Tiny fixtures may repeat content, but ownership
    // and loss/gradient collectives still follow the Megatron contract.
    let dense_batch_start = dense_dp_rank
        .checked_mul(accumulation_batch_size)
        .context("GLM-5 dense-DP batch offset overflows usize")?;
    let raw_batch =
        accumulation_dataset.padded_batch(dense_batch_start, accumulation_batch_size, device);

    let target_seq = if mtp_enabled {
        glm5_megatron_raw_seq_len(config.model.seq_len as i64)?
    } else {
        config.model.seq_len as i64
    };
    let actual_seq = raw_batch.input_ids.size()[1];
    let full_batch = if actual_seq > target_seq {
        bail!(
            "GLM-5 SFT batch has {actual_seq} raw tokens, exceeding Megatron raw sequence length {target_seq}"
        );
    } else if actual_seq < target_seq {
        let pad_token = train_dataset.pad_token_id;
        let pad_ids = Tensor::full(
            [raw_batch.input_ids.size()[0], target_seq - actual_seq],
            pad_token,
            (Kind::Int64, device),
        );
        let input_ids = Tensor::cat(&[&raw_batch.input_ids, &pad_ids], 1);
        let pad_mask = Tensor::zeros(
            [raw_batch.input_ids.size()[0], target_seq - actual_seq],
            (Kind::Int64, device),
        );
        let target_mask = Tensor::cat(&[&raw_batch.target_mask, &pad_mask], 1);
        Glm5SftBatch {
            input_ids,
            target_mask,
            num_masked: raw_batch.num_masked,
        }
    } else {
        raw_batch
    };

    // CP slice: each rank handles [cp_rank * s_local, (cp_rank+1) * s_local)
    let model_seq = config.model.seq_len as i64;
    let s_local = model_seq / cp_size as i64;
    let cp_batch = if cp_size > 1 {
        let input_ids = full_batch
            .input_ids
            .narrow(1, cp_rank as i64 * s_local, s_local);
        let target_mask = full_batch
            .target_mask
            .narrow(1, cp_rank as i64 * s_local, s_local);
        Glm5SftBatch {
            input_ids,
            target_mask,
            num_masked: full_batch.num_masked,
        }
    } else {
        Glm5SftBatch {
            input_ids: full_batch.input_ids.narrow(1, 0, model_seq),
            target_mask: full_batch.target_mask.narrow(1, 0, model_seq),
            num_masked: full_batch.num_masked,
        }
    };

    // Keep the CP-local MTP sequence length identical on every rank. The
    // right halo is allocated once, outside the training loop; C++ selects
    // each layer's absolute embedding and target offsets without per-step
    // Rust padding/slicing kernels.
    let mtp_embedding_ids = if runtime_config.num_nextn_predict_layers > 0 {
        Tensor::cat(
            &[
                &cp_batch.input_ids,
                &Tensor::full(
                    [cp_batch.input_ids.size()[0], 2],
                    train_dataset.pad_token_id,
                    (Kind::Int64, device),
                ),
            ],
            1,
        )
    } else {
        full_batch.input_ids.shallow_clone()
    };
    let mtp_target_ids = if runtime_config.num_nextn_predict_layers > 0 {
        Tensor::cat(
            &[
                &full_batch.input_ids,
                &Tensor::full(
                    [full_batch.input_ids.size()[0], 1],
                    train_dataset.pad_token_id,
                    (Kind::Int64, device),
                ),
            ],
            1,
        )
    } else {
        full_batch.input_ids.shallow_clone()
    };
    let mtp_target_mask = if runtime_config.num_nextn_predict_layers > 0 {
        Tensor::cat(
            &[
                &full_batch.target_mask,
                &Tensor::zeros(
                    [full_batch.target_mask.size()[0], 1],
                    (full_batch.target_mask.kind(), device),
                ),
            ],
            1,
        )
    } else {
        full_batch.target_mask.shallow_clone()
    };

    // ── Optimizer ──
    let lr = config.train.learning_rate as f64;
    let beta1 = config.train.adam_beta1 as f64;
    let beta2 = config.train.adam_beta2 as f64;
    let eps = config.train.adam_eps as f64;
    let trainable_vars = registry.var_store.trainable_variables();
    let trainable_var_names = trainable_variable_names(&registry)?;
    if trainable_var_names.len() != trainable_vars.len() {
        bail!("GLM-5 LoRA variable name/count mismatch");
    }
    let mut adam_m: Vec<Tensor> = trainable_vars.iter().map(Tensor::zeros_like).collect();
    let mut adam_v: Vec<Tensor> = trainable_vars.iter().map(Tensor::zeros_like).collect();

    let mut initial_loss = 0.0_f64;
    let mut last_loss = 0.0_f64;

    // Pre-load TP-sharded indexer weights for "full" layers. The native MTP
    // decoder is an extra full layer and owns its indexer state.
    let mut indexer_weights_map: BTreeMap<usize, Glm5TpAttentionWeights> = BTreeMap::new();
    for layer in 0..total_decoder_layers {
        let indexer_type = runtime_config
            .indexer_types
            .get(layer)
            .map(|s| s.as_str())
            .unwrap_or("full");
        if indexer_type == "full" {
            let attn = Glm5TpAttentionWeights::load_sharded(
                &weights_gpu,
                layer,
                compute_kind,
                &tp_shard,
                &runtime_config,
            )?;
            indexer_weights_map.insert(layer, attn);
        }
    }
    let mut shared_mlp_weights_map: BTreeMap<usize, Glm5TpMlpWeights> = BTreeMap::new();
    if use_cpp_moe {
        for layer in 0..n_layers {
            if runtime_config.is_moe_layer(layer) {
                let prefix = format!("model.layers.{layer}.mlp.shared_experts");
                shared_mlp_weights_map.insert(
                    layer,
                    Glm5TpMlpWeights::load_sharded(
                        &weights_gpu,
                        &prefix,
                        compute_kind,
                        tp_rank,
                        tp_size,
                    )?,
                );
            }
        }
    }

    let use_checkpointing = true;
    // IndexShare state is mutable across layers; avoid checkpointing attention
    // until forward/backward state snapshots are separate.
    let use_attention_checkpointing = false;
    rustrain_deepseek_v4::fp8_kernel::set_memory_fraction(
        config.train.cuda_memory_fraction,
        local_rank as i32,
    );
    info!(
        rank,
        memory_fraction = config.train.cuda_memory_fraction,
        "set caching allocator memory fraction"
    );

    // ── Training loop ──
    for step in 0..config.train.max_steps {
        let local_aggregate_base_token_count = full_batch
            .target_mask
            .narrow(1, 1, full_batch.target_mask.size()[1] - 1)
            .to_kind(Kind::Float)
            .sum(Kind::Float)
            .clamp_min(1.0);
        let aggregate_base_token_count = if dense_dp_size > 1 {
            dense_dp_comm
                .as_ref()
                .unwrap()
                .all_reduce(&local_aggregate_base_token_count)?
                .clamp_min(1.0)
        } else {
            local_aggregate_base_token_count
        };
        let mut accumulated_loss_val = 0.0_f64;
        let mut accumulated_mtp_loss_val = 0.0_f64;

        for accumulation_index in 0..accumulation_steps {
            let batch_start = (accumulation_index * micro_batch_size) as i64;
            let batch_len = micro_batch_size as i64;
            let full_batch = Glm5SftBatch {
                input_ids: full_batch.input_ids.narrow(0, batch_start, batch_len),
                target_mask: full_batch.target_mask.narrow(0, batch_start, batch_len),
                num_masked: full_batch.num_masked,
            };
            let cp_batch = Glm5SftBatch {
                input_ids: cp_batch.input_ids.narrow(0, batch_start, batch_len),
                target_mask: cp_batch.target_mask.narrow(0, batch_start, batch_len),
                num_masked: cp_batch.num_masked,
            };
            let mtp_embedding_ids = mtp_embedding_ids.narrow(0, batch_start, batch_len);
            let mtp_target_ids = mtp_target_ids.narrow(0, batch_start, batch_len);
            let mtp_target_mask = mtp_target_mask.narrow(0, batch_start, batch_len);
            let base_token_count = full_batch
                .target_mask
                .narrow(1, 1, full_batch.target_mask.size()[1] - 1)
                .to_kind(Kind::Float)
                .sum(Kind::Float);
            let global_base_token_count = if dense_dp_size > 1 {
                dense_dp_comm
                    .as_ref()
                    .unwrap()
                    .all_reduce(&base_token_count)?
            } else {
                base_token_count.shallow_clone()
            };
            let microbatch_weight = &global_base_token_count / &aggregate_base_token_count;
            let _checkpoint_registry_guard = CheckpointRegistryGuard;

            let embed = tensor(&weights_gpu, "model.embed_tokens.weight")?.to_kind(compute_kind);
            let mut hidden = Tensor::embedding(&embed, &cp_batch.input_ids, -1, false, false);
            if hidden.kind() != compute_kind {
                hidden = hidden.to_kind(compute_kind);
            }

            let mut index_share_state: Option<IndexShareState> = None;

            for layer in 0..n_layers {
                let p = format!("model.layers.{layer}");

                // ── Attention (TP+CP sharded) ──
                let attn_norm = tensor(&weights_gpu, &format!("{p}.input_layernorm.weight"))?
                    .to_kind(compute_kind);
                let hidden_norm = rms_norm(&hidden, &attn_norm, runtime_config.rms_norm_eps);

                // Load TP-sharded attention weights
                let attn_weights = Glm5TpAttentionWeights::load_sharded(
                    &weights_gpu,
                    layer,
                    compute_kind,
                    &tp_shard,
                    &runtime_config,
                )?;
                // Apply the full adapter deltas to this rank's local shard.  The
                // shard helper slices B (column parallel) or A (row parallel) so
                // every TP rank receives the corresponding LoRA update.
                let attn_weights =
                    attn_weights.with_lora(layer, &registry, &tp_shard, &runtime_config)?;

                let source = runtime_config.indexer_source_layer(layer);
                let indexer_weights = indexer_weights_map.get(&source).unwrap_or(&attn_weights);

                let is_full_layer = !runtime_config.should_skip_topk(layer)
                    && runtime_config
                        .indexer_types
                        .get(layer)
                        .map(|kind| kind == "full")
                        .unwrap_or(true);

                let attn_out = if use_attention_checkpointing {
                    let state_mutex = Arc::new(Mutex::new(index_share_state.take()));
                    let state_for_closure = state_mutex.clone();
                    let attn_clone = attn_weights.clone();
                    let indexer_clone = indexer_weights.clone();
                    let runtime_clone = runtime_config.clone();
                    let tp_clone = Glm5TpShard::new(
                        tp_rank,
                        tp_size,
                        runtime_config.num_attention_heads,
                        runtime_config.index_n_heads,
                    );
                    let layer_copy = layer;
                    let cp_rank_copy = cp_rank;
                    // Distributed collectives are stateful and cannot be captured by
                    // the Send + 'static checkpoint callback. Use the direct path for
                    // TP/CP and checkpoint only the single-rank case.
                    if cp_size > 1 || tp_size > 1 {
                        glm5_dsa_attention_tp_cp(
                            &hidden_norm,
                            &attn_weights,
                            indexer_weights,
                            &runtime_config,
                            &mut index_share_state,
                            layer,
                            &tp_shard,
                            cp_rank,
                            cp_size,
                            tp_comm.as_ref(),
                            cp_comm.as_ref(),
                        )
                    } else {
                        // CP=1: checkpoint (no comm needed)
                        let result = rustrain_deepseek_v4::fp8_kernel::checkpoint(
                            &hidden_norm,
                            move |input| {
                                let mut guard = state_for_closure.lock().unwrap();
                                let mut local_state = guard.take();
                                if is_full_layer {
                                    local_state = None;
                                }
                                let output = glm5_dsa_attention_tp_cp(
                                    input,
                                    &attn_clone,
                                    &indexer_clone,
                                    &runtime_clone,
                                    &mut local_state,
                                    layer_copy,
                                    &tp_clone,
                                    cp_rank_copy,
                                    1,
                                    None,
                                    None,
                                );
                                *guard = local_state;
                                output
                            },
                        );
                        index_share_state = state_mutex.lock().unwrap().take();
                        result
                    }
                } else {
                    glm5_dsa_attention_tp_cp(
                        &hidden_norm,
                        &attn_weights,
                        indexer_weights,
                        &runtime_config,
                        &mut index_share_state,
                        layer,
                        &tp_shard,
                        cp_rank,
                        cp_size,
                        tp_comm.as_ref(),
                        cp_comm.as_ref(),
                    )
                }
                .to_kind(compute_kind);

                // TP all-reduce: sum row-parallel output projections within the TP group.
                let attn_out = if tp_size > 1 {
                    let pd = no_grad(|| attn_out.shallow_clone()).detach();
                    let reduced = tp_comm.as_ref().unwrap().all_reduce(&pd)?;
                    let full = reduced.to_kind(compute_kind);
                    &attn_out + &(&full - &attn_out).detach()
                } else {
                    attn_out
                };

                let residual = &hidden + &attn_out;

                // ── MoE / Dense MLP ── (same as EP, divide by tp_size)
                let post_norm = tensor(
                    &weights_gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?
                .to_kind(compute_kind);
                let mlp_input = rms_norm(&residual, &post_norm, runtime_config.rms_norm_eps);

                if runtime_config.is_moe_layer(layer) {
                    let gate = tensor(&weights_gpu, &format!("{p}.mlp.gate.weight"))?
                        .to_kind(compute_kind);
                    let correction_bias =
                        weights_gpu.get(&format!("{p}.mlp.gate.e_score_correction_bias"));
                    if use_cpp_moe && tp_size > 1 {
                        let shared = shared_mlp_weights_map
                            .get(&layer)
                            .context("cached TP-sharded shared expert weights are missing")?;
                        let mut expert_gate = Vec::new();
                        let mut expert_up = Vec::new();
                        let mut expert_down = Vec::new();
                        let mut expert_gate_scales = Vec::new();
                        let mut expert_up_scales = Vec::new();
                        let mut expert_down_scales = Vec::new();
                        for &expert in &ep_shard.local_expert_indices {
                            let prefix = format!("{p}.mlp.experts.{expert}");
                            expert_gate.push(
                                tensor(
                                    &expert_weights_runtime,
                                    &format!("{prefix}.gate_proj.weight"),
                                )?
                                .shallow_clone(),
                            );
                            expert_up.push(
                                tensor(
                                    &expert_weights_runtime,
                                    &format!("{prefix}.up_proj.weight"),
                                )?
                                .shallow_clone(),
                            );
                            expert_down.push(
                                tensor(
                                    &expert_weights_runtime,
                                    &format!("{prefix}.down_proj.weight"),
                                )?
                                .shallow_clone(),
                            );
                            expert_gate_scales.push(
                                expert_weights_runtime
                                    .get(&format!("{prefix}.gate_proj.weight_scale_inv"))
                                    .map(Tensor::shallow_clone),
                            );
                            expert_up_scales.push(
                                expert_weights_runtime
                                    .get(&format!("{prefix}.up_proj.weight_scale_inv"))
                                    .map(Tensor::shallow_clone),
                            );
                            expert_down_scales.push(
                                expert_weights_runtime
                                    .get(&format!("{prefix}.down_proj.weight_scale_inv"))
                                    .map(Tensor::shallow_clone),
                            );
                        }
                        let cpp_moe = CheckpointedCppMoeTpLayer {
                            shared_gate: shared.gate_proj.shallow_clone(),
                            shared_up: shared.up_proj.shallow_clone(),
                            shared_down: shared.down_proj.shallow_clone(),
                            shared_gate_scale: shared
                                .gate_proj_scale
                                .as_ref()
                                .map(Tensor::shallow_clone),
                            shared_up_scale: shared
                                .up_proj_scale
                                .as_ref()
                                .map(Tensor::shallow_clone),
                            shared_down_scale: shared
                                .down_proj_scale
                                .as_ref()
                                .map(Tensor::shallow_clone),
                            gate,
                            correction_bias: correction_bias.map(Tensor::shallow_clone),
                            expert_gate,
                            expert_up,
                            expert_down,
                            expert_gate_scales,
                            expert_up_scales,
                            expert_down_scales,
                            expert_indices: ep_shard.local_expert_indices.clone(),
                            n_routed_experts: runtime_config.n_routed_experts as i32,
                            topk: runtime_config.num_experts_per_tok as i32,
                            n_group: runtime_config.n_group as i32,
                            topk_group: runtime_config.topk_group as i32,
                            scoring_func: match runtime_config.scoring_func.as_str() {
                                "sigmoid" => 0,
                                "softmax" => 1,
                                other => bail!("unsupported GLM5 scoring_func {other:?}"),
                            },
                            topk_method: match runtime_config.topk_method.as_str() {
                                "groupwise" => 0,
                                "noaux_tc" => 1,
                                other => bail!("unsupported GLM5 topk_method {other:?}"),
                            },
                            norm_topk_prob: runtime_config.norm_topk_prob,
                            routed_scaling_factor: runtime_config.routed_scaling_factor,
                            tp_comm: tp_comm
                                .as_ref()
                                .context("expert TP communicator is missing")?
                                .raw_comm_ptr() as usize,
                            tp_size: tp_size as i32,
                            device_id: local_rank as i32,
                        };
                        let full_mlp = if config.train.predequant_expert_weights {
                            cpp_moe.forward(&mlp_input)?
                        } else {
                            rustrain_deepseek_v4::fp8_kernel::checkpoint_result(
                                &mlp_input,
                                move |input| cpp_moe.forward(input),
                            )?
                        };
                        hidden = &residual + &full_mlp;
                    } else {
                        let shared_gate = keep_fp8(
                            tensor(
                                &weights_gpu,
                                &format!("{p}.mlp.shared_experts.gate_proj.weight"),
                            )?,
                            compute_kind,
                        );
                        let shared_up = keep_fp8(
                            tensor(
                                &weights_gpu,
                                &format!("{p}.mlp.shared_experts.up_proj.weight"),
                            )?,
                            compute_kind,
                        );
                        let shared_down = keep_fp8(
                            tensor(
                                &weights_gpu,
                                &format!("{p}.mlp.shared_experts.down_proj.weight"),
                            )?,
                            compute_kind,
                        );
                        let shared_gate_scale = weights_gpu.get(&format!(
                            "{p}.mlp.shared_experts.gate_proj.weight_scale_inv"
                        ));
                        let shared_up_scale = weights_gpu
                            .get(&format!("{p}.mlp.shared_experts.up_proj.weight_scale_inv"));
                        let shared_down_scale = weights_gpu.get(&format!(
                            "{p}.mlp.shared_experts.down_proj.weight_scale_inv"
                        ));

                        let shared_output = if use_checkpointing {
                            let sg = shared_gate.shallow_clone();
                            let su = shared_up.shallow_clone();
                            let sd = shared_down.shallow_clone();
                            let sgs = shared_gate_scale.map(|t| t.shallow_clone());
                            let sus = shared_up_scale.map(|t| t.shallow_clone());
                            let sds = shared_down_scale.map(|t| t.shallow_clone());
                            rustrain_deepseek_v4::fp8_kernel::checkpoint(&mlp_input, move |input| {
                                glm5_mlp_fp8(
                                    input,
                                    &sg,
                                    &su,
                                    &sd,
                                    sgs.as_ref(),
                                    sus.as_ref(),
                                    sds.as_ref(),
                                )
                            })
                        } else {
                            glm5_mlp_fp8(
                                &mlp_input,
                                &shared_gate,
                                &shared_up,
                                &shared_down,
                                shared_gate_scale,
                                shared_up_scale,
                                shared_down_scale,
                            )
                        };

                        let router_logits = mlp_input
                            .linear::<&Tensor>(&gate, None)
                            .to_kind(Kind::Float);
                        let k = runtime_config.num_experts_per_tok as i64;
                        let (topk_weights, topk_indices) = glm5_router_topk(
                            &router_logits,
                            correction_bias,
                            runtime_config.num_experts_per_tok,
                            &runtime_config.scoring_func,
                            &runtime_config.topk_method,
                            runtime_config.n_group,
                            runtime_config.topk_group,
                            runtime_config.norm_topk_prob,
                            runtime_config.routed_scaling_factor,
                        );

                        let flat_input = mlp_input.reshape([-1, mlp_input.size()[2]]);
                        let tk_indices = topk_indices.reshape([-1, k]);
                        let tk_weights = topk_weights.reshape([-1, k]);

                        let mut partial_output =
                            Tensor::zeros(flat_input.size(), (compute_kind, flat_input.device()));

                        for &global_e in &ep_shard.local_expert_indices {
                            let mask = tk_indices.eq(global_e as i64).to_kind(compute_kind);
                            let mask_flat = mask
                                .sum_dim_intlist([-1].as_slice(), false, compute_kind)
                                .to_kind(compute_kind);
                            let count = mask_flat.sum(compute_kind).double_value(&[]) as i64;
                            if count == 0 {
                                continue;
                            }
                            let eg = format!("{p}.mlp.experts.{global_e}");
                            let gate_w = expert_weights_runtime
                                .get(&format!("{eg}.gate_proj.weight"))
                                .context("expert weight")?;
                            let up_w = expert_weights_runtime
                                .get(&format!("{eg}.up_proj.weight"))
                                .context("expert weight")?;
                            let down_w = expert_weights_runtime
                                .get(&format!("{eg}.down_proj.weight"))
                                .context("expert weight")?;
                            let gate_w_scale = expert_weights_runtime
                                .get(&format!("{eg}.gate_proj.weight_scale_inv"));
                            let up_w_scale = expert_weights_runtime
                                .get(&format!("{eg}.up_proj.weight_scale_inv"));
                            let down_w_scale = expert_weights_runtime
                                .get(&format!("{eg}.down_proj.weight_scale_inv"));

                            let expert_out = glm5_mlp_fp8(
                                &flat_input,
                                &gate_w,
                                &up_w,
                                &down_w,
                                gate_w_scale,
                                up_w_scale,
                                down_w_scale,
                            );
                            let weighted_mask = (mask * &tk_weights)
                                .sum_dim_intlist([-1].as_slice(), false, compute_kind)
                                .to_kind(compute_kind);
                            let mask_expanded = weighted_mask
                                .unsqueeze(-1)
                                .expand([-1, expert_out.size()[1]], false);
                            partial_output = partial_output + &(expert_out * &mask_expanded);
                        }

                        let routed_partial = partial_output.reshape([1, -1, mlp_input.size()[2]]);
                        let routed_full = if ep_size > 1 {
                            let pd = no_grad(|| routed_partial.shallow_clone()).detach();
                            let reduced = ep_comm.as_ref().unwrap().all_reduce(&pd)?;
                            let full = reduced.to_kind(routed_partial.kind());
                            &routed_partial + &(&full - &routed_partial).detach()
                        } else {
                            routed_partial
                        };
                        // Shared experts are replicated, so add them once after the EP sum.
                        let full_mlp = routed_full + shared_output;
                        hidden = &residual + &full_mlp;
                    }
                } else {
                    let gate = keep_fp8(
                        tensor(&weights_gpu, &format!("{p}.mlp.gate_proj.weight"))?,
                        compute_kind,
                    );
                    let up = keep_fp8(
                        tensor(&weights_gpu, &format!("{p}.mlp.up_proj.weight"))?,
                        compute_kind,
                    );
                    let down = keep_fp8(
                        tensor(&weights_gpu, &format!("{p}.mlp.down_proj.weight"))?,
                        compute_kind,
                    );
                    let gate_scale = weights_gpu
                        .get(&format!("{p}.mlp.gate_proj.weight_scale_inv"))
                        .map(|t| t.shallow_clone());
                    let up_scale = weights_gpu
                        .get(&format!("{p}.mlp.up_proj.weight_scale_inv"))
                        .map(|t| t.shallow_clone());
                    let down_scale = weights_gpu
                        .get(&format!("{p}.mlp.down_proj.weight_scale_inv"))
                        .map(|t| t.shallow_clone());

                    let mlp = if use_checkpointing {
                        rustrain_deepseek_v4::fp8_kernel::checkpoint(&mlp_input, move |input| {
                            glm5_mlp_fp8(
                                input,
                                &gate,
                                &up,
                                &down,
                                gate_scale.as_ref(),
                                up_scale.as_ref(),
                                down_scale.as_ref(),
                            )
                        })
                    } else {
                        glm5_mlp_fp8(
                            &mlp_input,
                            &gate,
                            &up,
                            &down,
                            gate_scale.as_ref(),
                            up_scale.as_ref(),
                            down_scale.as_ref(),
                        )
                    };
                    hidden = &residual + &mlp;
                }

                if hidden.kind() != compute_kind {
                    hidden = hidden.to_kind(compute_kind);
                }
            }

            // Final norm, FP8 lm_head, and vocabulary-parallel CE stay in one
            // C++ dispatch. The returned normalized state also feeds native MTP.
            let final_norm = tensor(&weights_gpu, "model.norm.weight")?.to_kind(compute_kind);
            let global_offset = cp_rank as i64 * s_local;
            let seq_len_local = cp_batch.input_ids.size()[1];
            let target_len = if global_offset + seq_len_local < target_seq {
                seq_len_local
            } else {
                seq_len_local - 1
            };
            let base_output = rustrain_deepseek_v4::fp8_kernel::
                glm5_next_token_postprocess_loss_vocab_parallel_cpp(
                    &hidden,
                    &final_norm,
                    &tp_vocab.lm_head.weight,
                    tp_vocab.lm_head.weight_scale.as_ref(),
                    &full_batch.input_ids,
                    &full_batch.target_mask,
                    runtime_config.rms_norm_eps,
                    glm5_next_token_target_offset(global_offset)?,
                    4096,
                    tp_vocab.lm_head.range.vocab_start,
                    tp_vocab.lm_head.range.padded_vocab_size,
                    tp_comm
                        .as_ref()
                        .map_or(std::ptr::null_mut(), |comm| comm.raw_comm_ptr()),
                    tp_size as i32,
                )?;
            let normed = base_output.normalized;
            if normed.size()[1] != target_len {
                bail!(
                    "GLM5 base CE normalized {} positions, expected {target_len}",
                    normed.size()[1]
                );
            }

            let mtp_losses = if runtime_config.num_nextn_predict_layers > 0 {
                let mut previous_mtp_block: Option<Tensor> = None;
                let mut mtp_losses = Vec::with_capacity(runtime_config.num_nextn_predict_layers);

                for (mtp_idx, mtp_layer) in glm5_mtp_layer_indices(&runtime_config)?
                    .into_iter()
                    .enumerate()
                {
                    let offset = mtp_idx as i64;
                    let projection = Glm5MtpProjectionWeights::load_tp_sharded(
                        &weights_gpu,
                        mtp_layer,
                        compute_kind,
                        runtime_config.hidden_size,
                        tp_rank,
                        tp_size,
                    )?;
                    let source_hidden = previous_mtp_block.as_ref().unwrap_or(&normed);
                    let prepared = rustrain_deepseek_v4::fp8_kernel::glm5_mtp_prepare_tp_cpp(
                        source_hidden,
                        &mtp_embedding_ids,
                        &tp_vocab.embed_tokens.weight,
                        &projection.enorm,
                        &projection.hnorm,
                        &projection.eh_proj,
                        projection.eh_proj_scale.as_ref(),
                        runtime_config.rms_norm_eps,
                        (global_offset + offset + 1) as i32,
                        tp_vocab.embed_tokens.range.vocab_start,
                        tp_vocab.embed_tokens.range.padded_vocab_size,
                        tp_comm
                            .as_ref()
                            .map_or(std::ptr::null_mut(), |comm| comm.raw_comm_ptr()),
                        tp_rank as i32,
                        tp_size as i32,
                    )?;
                    let mtp_block = run_mtp_decoder_layer_tp_ep(
                        &prepared,
                        &weights_gpu,
                        &expert_weights_runtime,
                        mtp_layer,
                        &runtime_config,
                        indexer_weights_map
                            .get(&mtp_layer)
                            .context("cached MTP TP attention weights are missing")?,
                        &tp_shard,
                        &ep_shard,
                        tp_comm.as_ref(),
                        cp_comm.as_ref(),
                        ep_comm.as_ref(),
                        tp_size,
                        cp_rank,
                        cp_size,
                        ep_size,
                    )?;

                    let mtp_output =
                    rustrain_deepseek_v4::fp8_kernel::glm5_mtp_postprocess_loss_vocab_parallel_cpp(
                        &mtp_block,
                        &projection.shared_head_norm,
                        &tp_vocab.lm_head.weight,
                        tp_vocab.lm_head.weight_scale.as_ref(),
                        &mtp_target_ids,
                        &mtp_target_mask,
                        runtime_config.rms_norm_eps,
                        (global_offset + offset) as i32,
                        4096,
                        tp_vocab.lm_head.range.vocab_start,
                        tp_vocab.lm_head.range.padded_vocab_size,
                        tp_comm
                            .as_ref()
                            .map_or(std::ptr::null_mut(), |comm| comm.raw_comm_ptr()),
                        tp_size as i32,
                    )?;
                    let local_loss = mtp_output.loss;
                    let local_sum = mtp_output.loss_sum;
                    let local_count = mtp_output.token_count;
                    let layer_loss = if cp_size > 1 {
                        let reduced_sum = cp_comm.as_ref().unwrap().all_reduce(&local_sum)?;
                        let reduced_count = cp_comm.as_ref().unwrap().all_reduce(&local_count)?;
                        // Match the main LM identity reattachment: expose the
                        // global token-normalized value while multiplying the
                        // local autograd edge by cp_size. The final gradient sync
                        // divides by cp_size once for both LM and MTP.
                        reattach_cp_token_mean(&local_sum, &reduced_sum, &reduced_count, cp_size)
                    } else if dense_dp_size > 1 {
                        // EP owns expert routing, not data-parallel loss
                        // normalization. Only ranks with the same TP shard
                        // participate in this dense-DP reduction.
                        let reduced_sum = dense_dp_comm.as_ref().unwrap().all_reduce(&local_sum)?;
                        let reduced_count =
                            dense_dp_comm.as_ref().unwrap().all_reduce(&local_count)?;
                        reattach_global_token_mean(&local_sum, &reduced_sum, &reduced_count)
                    } else {
                        local_loss
                    };
                    mtp_losses.push(layer_loss);
                    previous_mtp_block = Some(mtp_output.normalized);
                }
                mtp_losses
            } else {
                Vec::new()
            };

            // CP all-reduce: preserve the local graph while exposing the global
            // token-normalized loss value on every CP rank.
            let local_lm_loss = base_output.loss;
            let local_lm_sum = base_output.loss_sum;
            let local_lm_count = base_output.token_count;
            let lm_loss = if cp_size > 1 {
                let reduced_sum = cp_comm.as_ref().unwrap().all_reduce(&local_lm_sum)?;
                let reduced_count = cp_comm.as_ref().unwrap().all_reduce(&local_lm_count)?;
                reattach_cp_token_mean(&local_lm_sum, &reduced_sum, &reduced_count, cp_size)
            } else if dense_dp_size > 1 {
                let reduced_sum = dense_dp_comm.as_ref().unwrap().all_reduce(&local_lm_sum)?;
                let reduced_count = dense_dp_comm
                    .as_ref()
                    .unwrap()
                    .all_reduce(&local_lm_count)?;
                reattach_global_token_mean(&local_lm_sum, &reduced_sum, &reduced_count)
            } else {
                local_lm_loss
            };
            let weighted_lm_loss = &lm_loss * &microbatch_weight;
            let loss = if mtp_losses.is_empty() {
                weighted_lm_loss
            } else {
                let weighted_mtp_losses: Vec<Tensor> = mtp_losses
                    .into_iter()
                    .map(|layer_loss| {
                        let weighted = &layer_loss * &microbatch_weight;
                        accumulated_mtp_loss_val += weighted.double_value(&[]);
                        weighted
                    })
                    .collect();
                let combined = rustrain_deepseek_v4::fp8_kernel::glm5_combine_losses_cpp(
                    &weighted_lm_loss,
                    &weighted_mtp_losses,
                    config.train.mtp_loss_scaling_factor,
                )?;
                combined.total
            };

            let loss_val = loss.double_value(&[]);
            accumulated_loss_val += loss_val;

            // ── Backward ──
            loss.backward();
            rustrain_deepseek_v4::fp8_kernel::clear_checkpoint_registry();
        }

        if step == 0 {
            rustrain_deepseek_v4::fp8_kernel::empty_cache();
        }

        // These adapters target dense attention weights only. Their full-sized
        // A/B tensors are locally sliced by TP, so first reconstruct the full
        // adapter gradient over normal TP, then accumulate samples over dense
        // DP. Routed experts are frozen; trainable expert adapters would need
        // the independent expert-DP communicator created above.
        let synced_grads: Vec<Tensor> = if world_size > 1 {
            let vars = registry.var_store.trainable_variables();
            let mut synced = Vec::with_capacity(vars.len());
            for (var_index, var) in vars.iter().enumerate() {
                let g = var.grad();
                if g.defined() && g.numel() > 0 {
                    let reduced = if megatron_coords.is_some() {
                        let tp_reduced = if tp_size > 1
                            && glm5_lora_gradient_requires_step_end_tp_sum(
                                &trainable_var_names[var_index],
                            )? {
                            tp_comm.as_ref().unwrap().all_reduce(&g)?
                        } else {
                            g.shallow_clone()
                        };
                        if dense_dp_size > 1 {
                            dense_dp_comm.as_ref().unwrap().all_reduce(&tp_reduced)?
                        } else {
                            tp_reduced
                        }
                    } else {
                        // Legacy CP ranks differentiate disjoint token chunks.
                        // Its identity reattachment contributes cp_size, so the
                        // world sum is divided by CP exactly once.
                        let reduced = world_comm.as_ref().unwrap().all_reduce(&g)?;
                        &reduced / (cp_size as f64)
                    };
                    synced.push(no_grad(|| reduced.to_kind(g.kind())));
                } else {
                    synced.push(g.shallow_clone());
                }
            }
            synced
        } else {
            Vec::new()
        };

        // ── Adam ──
        let mut current_vars = registry.var_store.trainable_variables();
        for (i, var) in current_vars.iter_mut().enumerate() {
            let grad = if world_size > 1 {
                synced_grads[i].shallow_clone()
            } else {
                var.grad()
            };
            if grad.defined() {
                let g = grad.to_kind(Kind::Float);
                let m = &mut adam_m[i];
                let v = &mut adam_v[i];
                *m = m.shallow_clone() * beta1 + &(&g * (1.0 - beta1));
                *v = v.shallow_clone() * beta2 + &(&g * &g * (1.0 - beta2));
                let sn = (step + 1) as f64;
                let mh = m.shallow_clone() / (1.0 - beta1.powf(sn));
                let vh = v.shallow_clone() / (1.0 - beta2.powf(sn));
                let update = &mh / (vh.sqrt() + eps);
                let _ = no_grad(|| var.f_add_(&(update * (-lr))));
            }
            var.zero_grad();
        }

        if step == 0 {
            initial_loss = accumulated_loss_val;
        }
        last_loss = accumulated_loss_val;
        info!(
            rank,
            step = step + 1,
            loss = accumulated_loss_val,
            mtp_loss = if !mtp_enabled {
                None
            } else {
                Some(accumulated_mtp_loss_val)
            },
            accumulation_steps,
            "GLM-5 TP+CP+EP train step"
        );
    }

    // ── Save LoRA adapter ──
    let adapter_output = run_paths
        .checkpoints
        .join("glm5-lora-adapter-tp-cp-ep.safetensors");
    registry.save(&adapter_output)?;
    info!(rank, adapter = %adapter_output.display(), "adapter saved");

    let final_loss = last_loss;
    info!(rank, initial_loss, final_loss, "GLM-5 TP+CP+EP complete");

    Ok(TpCpEpSummary {
        adapter_output: adapter_output.display().to_string(),
        initial_loss,
        final_loss,
        trainable_params: trainable_count,
    })
}

/// Shared weight loading function (same as session_ep.rs)
fn load_glm5_weights_shared(
    model_path: &std::path::Path,
    needed: &HashSet<String>,
    tp_rank: usize,
    tp_size: usize,
) -> Result<BTreeMap<String, Tensor>> {
    // Delegate to the existing function in session_ep
    // We can't call it directly since it's private, so we inline the logic.
    #[derive(serde::Deserialize)]
    struct SafetensorsIndex {
        weight_map: std::collections::HashMap<String, String>,
    }

    let index_path = model_path.join("model.safetensors.index.json");
    let index_text = std::fs::read_to_string(&index_path)
        .with_context(|| format!("failed to read {}", index_path.display()))?;
    let index: SafetensorsIndex = serde_json::from_str(&index_text)?;

    let mut shard_to_tensors: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for name in needed {
        if let Some(shard) = index.weight_map.get(name) {
            shard_to_tensors
                .entry(shard.clone())
                .or_default()
                .push(name.clone());
        }
    }

    let mut weights = BTreeMap::new();
    for (shard_file, tensor_names) in &shard_to_tensors {
        let shard_path = model_path.join(shard_file);
        let shard_needed: HashSet<String> = tensor_names.iter().cloned().collect();
        match rustrain_deepseek_v4::fp8_kernel::load_safetensors_native(
            &shard_path,
            &shard_needed,
            -1,
        ) {
            Ok(shard_weights) => {
                for (name, t) in shard_weights {
                    let t = materialize_glm5_expert_tensor_for_tp(&name, t, tp_rank, tp_size)?;
                    weights.insert(name, t);
                }
            }
            Err(_) => {
                // Fallback to tch-rs for this one shard only. Loading the full
                // model here would defeat expert-TP's bounded CPU residency.
                let all_tensors =
                    rustrain_checkpoint::safetensors::read_safetensors_map(&shard_path)?;
                for name in &shard_needed {
                    if let Some(t) = all_tensors.get(name) {
                        let t = materialize_glm5_expert_tensor_for_tp(
                            name,
                            t.shallow_clone(),
                            tp_rank,
                            tp_size,
                        )?;
                        weights.insert(name.clone(), t);
                    }
                }
            }
        }
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cp_token_mean_matches_global_value_and_post_sync_gradient() {
        let local_sum = Tensor::from(2.0_f32);
        let _ = local_sum.set_requires_grad(true);
        let reduced_sum = Tensor::from(10.0_f32);
        let reduced_count = Tensor::from(4.0_f32);

        let loss = reattach_cp_token_mean(&local_sum, &reduced_sum, &reduced_count, 2);
        assert!((loss.double_value(&[]) - 2.5).abs() < 1e-6);
        loss.backward();

        let local_grad = local_sum.grad().double_value(&[]);
        // The training loop synchronizes this local derivative by /cp_size.
        assert!((local_grad / 2.0 - 0.25).abs() < 1e-6);

        let sparse_local_sum = Tensor::from(0.5_f32);
        let _ = sparse_local_sum.set_requires_grad(true);
        let sparse_loss = reattach_cp_token_mean(
            &sparse_local_sum,
            &Tensor::from(3.0_f32),
            &Tensor::from(1.0_f32),
            4,
        );
        assert!((sparse_loss.double_value(&[]) - 3.0).abs() < 1e-6);
        sparse_loss.backward();
        assert!((sparse_local_sum.grad().double_value(&[]) / 4.0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn staged_experts_accept_only_tp_without_cp_or_ep() {
        assert!(validate_glm5_staged_expert_topology(false, 2, 1, 1).is_ok());
        assert!(validate_glm5_staged_expert_topology(false, 8, 1, 1).is_ok());
        assert!(validate_glm5_staged_expert_topology(true, 1, 4, 1).is_ok());
        assert!(validate_glm5_staged_expert_topology(false, 1, 1, 1).is_err());
        assert!(validate_glm5_staged_expert_topology(false, 2, 2, 1).is_err());
        assert!(validate_glm5_staged_expert_topology(false, 2, 1, 2).is_err());
    }

    #[test]
    fn expert_tp_materializes_weight_and_scale_on_matching_axes() {
        let gate = Tensor::arange(8 * 4, (Kind::Float, Device::Cpu)).reshape([8, 4]);
        let gate = materialize_glm5_expert_tensor_for_tp(
            "model.layers.1.mlp.experts.0.gate_proj.weight",
            gate,
            1,
            2,
        )
        .unwrap();
        assert_eq!(gate.size(), vec![4, 4]);
        assert_eq!(gate.double_value(&[0, 0]), 16.0);

        let down = Tensor::arange(4 * 8, (Kind::Float, Device::Cpu)).reshape([4, 8]);
        let down = materialize_glm5_expert_tensor_for_tp(
            "model.layers.1.mlp.experts.0.down_proj.weight",
            down,
            1,
            2,
        )
        .unwrap();
        assert_eq!(down.size(), vec![4, 4]);
        assert_eq!(down.double_value(&[0, 0]), 4.0);

        let scale = Tensor::from_slice(&[0.25_f32, 0.5]).reshape([2, 1]);
        let scale = materialize_glm5_expert_tensor_for_tp(
            "model.layers.1.mlp.experts.0.up_proj.weight_scale_inv",
            scale,
            1,
            2,
        )
        .unwrap();
        assert_eq!(scale.size(), vec![1, 1]);
        assert!((scale.double_value(&[0, 0]) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn dense_mlp_needed_set_keeps_fp8_weights_and_scales_paired() {
        let mut needed = HashSet::new();
        insert_glm5_dense_mlp_weights(&mut needed, "model.layers.0.mlp");

        assert_eq!(needed.len(), 6);
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            assert!(needed.contains(&format!("model.layers.0.mlp.{projection}.weight")));
            assert!(needed.contains(&format!("model.layers.0.mlp.{projection}.weight_scale_inv")));
        }
    }

    #[test]
    fn base_ce_starts_at_the_first_shifted_raw_token() {
        assert_eq!(glm5_next_token_target_offset(0).unwrap(), 1);
        assert_eq!(glm5_next_token_target_offset(64).unwrap(), 65);
    }

    #[test]
    fn lora_tp_gradient_sync_policy_avoids_double_reducing_replicated_modules() {
        assert!(!glm5_lora_gradient_requires_step_end_tp_sum("layer0/WqA/lora_a").unwrap());
        assert!(!glm5_lora_gradient_requires_step_end_tp_sum("layer7/Wkv/lora_b").unwrap());
        assert!(glm5_lora_gradient_requires_step_end_tp_sum("layer0/WqB/lora_a").unwrap());
        assert!(glm5_lora_gradient_requires_step_end_tp_sum("layer7/Wo/lora_b").unwrap());
        assert!(glm5_lora_gradient_requires_step_end_tp_sum("unknown").is_err());
    }
}
