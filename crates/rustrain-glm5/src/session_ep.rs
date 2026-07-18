use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use tch::{Device, Kind, Reduction, Tensor, no_grad};
use tracing::{info, warn};

use crate::lora::*;
use crate::model::*;
use crate::model::{glm5_mlp, rms_norm};
use crate::sft::*;
use rustrain_checkpoint::safetensors::tensor;
use rustrain_nccl::nccl::{self as nccl_smoke, NcclPersistentComm};

/// Keep FP8 weights as-is; only convert non-FP8 tensors to `kind`.
fn keep_fp8(t: &Tensor, kind: Kind) -> Tensor {
    if t.kind() == Kind::Float8e4m3fn {
        t.shallow_clone()
    } else {
        t.to_kind(kind)
    }
}

fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

fn reattach_ep_token_mean(
    local_sum: &Tensor,
    reduced_sum: &Tensor,
    reduced_count: &Tensor,
) -> Tensor {
    let visible_sum = local_sum + &(reduced_sum - local_sum).detach();
    visible_sum / reduced_count.clamp_min(1.0)
}

/// EP shard for GLM-5.2 (same pattern as V4).
pub struct Glm5EpShard {
    pub rank: usize,
    pub world_size: usize,
    pub experts_per_rank: usize,
    pub expert_start: usize,
    pub local_expert_indices: Vec<usize>,
}

impl Glm5EpShard {
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

#[derive(Debug, Serialize)]
pub struct Glm5LoraSftSummary {
    pub adapter_output: String,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub trainable_params: usize,
}

fn add_mtp_decoder_weights(
    needed: &mut HashSet<String>,
    layer: usize,
    local_expert_indices: &[usize],
    topk_method: &str,
) {
    needed.extend(Glm5MtpProjectionWeights::weight_names(layer));
    let p = format!("model.layers.{layer}");
    needed.insert(format!("{p}.input_layernorm.weight"));
    needed.insert(format!("{p}.post_attention_layernorm.weight"));
    for suffix in [
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
    for suffix in [
        "k_norm.weight",
        "k_norm.bias",
        "weights_proj.weight",
        "wk.weight",
        "wq_b.weight",
    ] {
        needed.insert(format!("{p}.self_attn.indexer.{suffix}"));
        if matches!(suffix, "weights_proj.weight" | "wk.weight" | "wq_b.weight") {
            needed.insert(format!("{p}.self_attn.indexer.{suffix}_scale_inv"));
        }
    }

    needed.insert(format!("{p}.mlp.gate.weight"));
    if topk_method == "noaux_tc" {
        needed.insert(format!("{p}.mlp.gate.e_score_correction_bias"));
    }
    for suffix in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
        needed.insert(format!("{p}.mlp.shared_experts.{suffix}"));
        needed.insert(format!("{p}.mlp.shared_experts.{suffix}_scale_inv"));
    }
    for expert in local_expert_indices {
        for suffix in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            needed.insert(format!("{p}.mlp.experts.{expert}.{suffix}"));
            needed.insert(format!("{p}.mlp.experts.{expert}.{suffix}_scale_inv"));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_mtp_decoder_layer_ep(
    input: &Tensor,
    layer: usize,
    attn: &Glm5AttentionWeights,
    weights_gpu: &BTreeMap<String, Tensor>,
    expert_weights_gpu: &BTreeMap<String, Tensor>,
    config: &Glm5RuntimeConfig,
    ep_shard: &Glm5EpShard,
    nccl_comm: Option<&NcclPersistentComm>,
) -> Result<Tensor> {
    use rustrain_deepseek_v4::fp8_kernel::{Glm5MtpDecoderDescriptor, glm5_mtp_decoder_layer_cpp};

    fn ptr(tensor: &Tensor) -> *mut std::ffi::c_void {
        tensor.as_ptr() as *mut _
    }
    fn opt_ptr(tensor: Option<&Tensor>) -> *mut std::ffi::c_void {
        tensor.map_or(std::ptr::null_mut(), ptr)
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
    let shared_prefix = format!("{p}.mlp.shared_experts");

    let mut expert_gate = Vec::with_capacity(ep_shard.local_expert_indices.len());
    let mut expert_up = Vec::with_capacity(ep_shard.local_expert_indices.len());
    let mut expert_down = Vec::with_capacity(ep_shard.local_expert_indices.len());
    let mut expert_gate_scales = Vec::with_capacity(ep_shard.local_expert_indices.len());
    let mut expert_up_scales = Vec::with_capacity(ep_shard.local_expert_indices.len());
    let mut expert_down_scales = Vec::with_capacity(ep_shard.local_expert_indices.len());
    for expert in &ep_shard.local_expert_indices {
        let prefix = format!("{p}.mlp.experts.{expert}");
        expert_gate.push(ptr(tensor(
            expert_weights_gpu,
            &format!("{prefix}.gate_proj.weight"),
        )?));
        expert_up.push(ptr(tensor(
            expert_weights_gpu,
            &format!("{prefix}.up_proj.weight"),
        )?));
        expert_down.push(ptr(tensor(
            expert_weights_gpu,
            &format!("{prefix}.down_proj.weight"),
        )?));
        expert_gate_scales.push(opt_ptr(
            expert_weights_gpu.get(&format!("{prefix}.gate_proj.weight_scale_inv")),
        ));
        expert_up_scales.push(opt_ptr(
            expert_weights_gpu.get(&format!("{prefix}.up_proj.weight_scale_inv")),
        ));
        expert_down_scales.push(opt_ptr(
            expert_weights_gpu.get(&format!("{prefix}.down_proj.weight_scale_inv")),
        ));
    }
    let local_expert_indices: Vec<i32> = ep_shard
        .local_expert_indices
        .iter()
        .map(|&expert| i32::try_from(expert).context("GLM-5 expert index exceeds C++ ABI"))
        .collect::<Result<_>>()?;
    let is_moe = config.is_moe_layer(layer);

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
        q_a_scale: opt_ptr(attn.q_a_proj_scale.as_ref()),
        q_b_scale: opt_ptr(attn.q_b_proj_scale.as_ref()),
        kv_a_scale: opt_ptr(attn.kv_a_proj_scale.as_ref()),
        kv_b_scale: opt_ptr(attn.kv_b_proj_scale.as_ref()),
        o_scale: opt_ptr(attn.o_proj_scale.as_ref()),
        idx_wq_b: opt_ptr(attn.indexer_wq_b.as_ref()),
        idx_wk: opt_ptr(attn.indexer_wk.as_ref()),
        idx_k_norm_w: opt_ptr(attn.indexer_k_norm_weight.as_ref()),
        idx_k_norm_b: opt_ptr(attn.indexer_k_norm_bias.as_ref()),
        idx_weights_proj: opt_ptr(attn.indexer_weights_proj.as_ref()),
        idx_weights_proj_scale: opt_ptr(attn.indexer_weights_proj_scale.as_ref()),
        idx_wq_b_scale: opt_ptr(attn.indexer_wq_b_scale.as_ref()),
        idx_wk_scale: opt_ptr(attn.indexer_wk_scale.as_ref()),
        gate_weight: opt_ptr(
            is_moe
                .then(|| tensor(weights_gpu, &format!("{p}.mlp.gate.weight")))
                .transpose()?,
        ),
        correction_bias: opt_ptr(weights_gpu.get(&format!("{p}.mlp.gate.e_score_correction_bias"))),
        shared_gate: opt_ptr(
            is_moe
                .then(|| tensor(weights_gpu, &format!("{shared_prefix}.gate_proj.weight")))
                .transpose()?,
        ),
        shared_up: opt_ptr(
            is_moe
                .then(|| tensor(weights_gpu, &format!("{shared_prefix}.up_proj.weight")))
                .transpose()?,
        ),
        shared_down: opt_ptr(
            is_moe
                .then(|| tensor(weights_gpu, &format!("{shared_prefix}.down_proj.weight")))
                .transpose()?,
        ),
        shared_gate_scale: opt_ptr(
            weights_gpu.get(&format!("{shared_prefix}.gate_proj.weight_scale_inv")),
        ),
        shared_up_scale: opt_ptr(
            weights_gpu.get(&format!("{shared_prefix}.up_proj.weight_scale_inv")),
        ),
        shared_down_scale: opt_ptr(
            weights_gpu.get(&format!("{shared_prefix}.down_proj.weight_scale_inv")),
        ),
        dense_gate: opt_ptr(
            (!is_moe)
                .then(|| tensor(weights_gpu, &format!("{p}.mlp.gate_proj.weight")))
                .transpose()?,
        ),
        dense_up: opt_ptr(
            (!is_moe)
                .then(|| tensor(weights_gpu, &format!("{p}.mlp.up_proj.weight")))
                .transpose()?,
        ),
        dense_down: opt_ptr(
            (!is_moe)
                .then(|| tensor(weights_gpu, &format!("{p}.mlp.down_proj.weight")))
                .transpose()?,
        ),
        dense_gate_scale: opt_ptr(weights_gpu.get(&format!("{p}.mlp.gate_proj.weight_scale_inv"))),
        dense_up_scale: opt_ptr(weights_gpu.get(&format!("{p}.mlp.up_proj.weight_scale_inv"))),
        dense_down_scale: opt_ptr(weights_gpu.get(&format!("{p}.mlp.down_proj.weight_scale_inv"))),
        expert_gate_weights: expert_gate.as_mut_ptr(),
        expert_up_weights: expert_up.as_mut_ptr(),
        expert_down_weights: expert_down.as_mut_ptr(),
        expert_gate_scales: expert_gate_scales.as_mut_ptr(),
        expert_up_scales: expert_up_scales.as_mut_ptr(),
        expert_down_scales: expert_down_scales.as_mut_ptr(),
        local_expert_indices: local_expert_indices.as_ptr(),
        ep_comm: nccl_comm.map_or(std::ptr::null_mut(), NcclPersistentComm::raw_comm_ptr),
        ep_rank: ep_shard.rank as i32,
        ep_size: ep_shard.world_size as i32,
        n_local_experts: local_expert_indices.len() as i32,
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
        num_heads: config.num_attention_heads as i32,
        qk_nope: config.qk_nope_head_dim as i32,
        qk_rope: config.qk_rope_head_dim as i32,
        v_head: config.v_head_dim as i32,
        kv_lora: config.kv_lora_rank as i32,
        idx_head_dim: config.index_head_dim as i32,
        idx_n_heads: config.index_n_heads as i32,
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
        tp_size: 1,
        cp_rank: 0,
        cp_size: 1,
        ..Default::default()
    };
    glm5_mtp_decoder_layer_cpp(&descriptor)
}

/// EP-distributed LoRA SFT training for GLM-5.2.
///
/// Each rank loads: all attention weights (replicated) + 1/world_size experts (sharded)
/// + shared experts + gate + embed + head + norm + LoRA adapter.
///
/// Forward: loop through ALL layers, DSA attention with LoRA (autograd),
/// and MoE with EP owner dispatch/return. Shared experts remain local to the
/// originating rank and are never reconstructed by an EP all-reduce.
/// Backward: loss.backward() + LoRA gradient all-reduce.
/// IndexShare: shared layers reuse full layer's indexer weights + top-k mask.
pub fn train_glm5_lora_sft_ep(
    config: &rustrain_core::runtime::Config,
    run_paths: &rustrain_core::runtime::RunPaths,
) -> Result<Glm5LoraSftSummary> {
    // ── Parse distributed env ──
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;

    // ── Model config ──
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("GLM-5 LoRA SFT EP requires model.model_path")?;
    let model_path = std::path::PathBuf::from(model_path);
    let runtime_config = read_glm5_config(&model_path.join("config.json"))?;
    let declared_ep_size = config.parallel.expert_model_parallel_size.max(1);
    if world_size != declared_ep_size {
        bail!(
            "GLM-5 EP runtime WORLD_SIZE {world_size} does not match configured expert_model_parallel_size {declared_ep_size}"
        );
    }
    validate_glm5_mtp_distributed_contract(
        runtime_config.num_nextn_predict_layers,
        1,
        config.parallel.context_parallel_size,
        world_size,
    )?;
    if runtime_config.num_nextn_predict_layers > 0 {
        let required_seq_len = runtime_config
            .num_nextn_predict_layers
            .checked_add(2)
            .context("GLM-5 MTP required sequence length overflows usize")?;
        if config.model.seq_len < required_seq_len {
            bail!(
                "GLM-5 {}-layer native MTP requires model.seq_len >= {}, got {}",
                runtime_config.num_nextn_predict_layers,
                required_seq_len,
                config.model.seq_len
            );
        }
    }
    info!(
        rank,
        world_size,
        local_rank,
        layers = runtime_config.num_hidden_layers,
        indexer_types = ?runtime_config.indexer_types,
        "GLM-5.2 LoRA SFT EP config loaded"
    );

    if world_size == 0 {
        bail!("GLM-5 EP: WORLD_SIZE must be positive");
    }
    if runtime_config.index_topk_freq <= 0 {
        bail!(
            "GLM-5 EP: index_topk_freq must be positive, got {}",
            runtime_config.index_topk_freq
        );
    }
    if runtime_config.num_experts_per_tok <= 0
        || runtime_config.num_experts_per_tok > runtime_config.n_routed_experts
    {
        bail!(
            "GLM-5 EP: num_experts_per_tok {} must be in 1..={} ",
            runtime_config.num_experts_per_tok,
            runtime_config.n_routed_experts
        );
    }
    if config.model.seq_len == 0 {
        bail!("GLM-5 EP: model.seq_len must be positive");
    }
    if runtime_config.n_routed_experts % world_size != 0 {
        bail!(
            "GLM-5 EP: n_routed_experts {} must be divisible by world_size {world_size}",
            runtime_config.n_routed_experts
        );
    }

    // ── EP shard ──
    let ep_shard = Glm5EpShard::new(rank, world_size, runtime_config.n_routed_experts);
    info!(
        rank,
        experts_per_rank = ep_shard.experts_per_rank,
        expert_start = ep_shard.expert_start,
        "EP shard"
    );

    // ── LoRA config ──
    let lora_config_raw = config
        .lora
        .as_ref()
        .context("GLM-5 LoRA SFT EP requires [lora] config section")?;
    let trainable_layer_indices: Vec<usize> = lora_config_raw
        .target_layers
        .iter()
        .map(|l| *l as usize)
        .collect();
    if let Some(layer) = trainable_layer_indices
        .iter()
        .copied()
        .find(|layer| *layer >= runtime_config.num_hidden_layers)
    {
        bail!(
            "GLM-5 EP LoRA target layer {layer} is outside the trunk range 0..{}; native MTP decoder layers remain frozen",
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

    // ── Compute dtype ──
    let compute_kind = match config.train.dtype {
        rustrain_core::runtime::DType::Bf16 => Kind::BFloat16,
        _ => Kind::Float,
    };
    let device = Device::Cuda(local_rank);

    // ── Staggered loading to avoid OOM ──
    if rank > 0 {
        info!(
            rank,
            delay_secs = rank * 5,
            "waiting before weight loading (staggered)"
        );
        std::thread::sleep(std::time::Duration::from_secs((rank * 5) as u64));
    }

    // ── Build needed weight set ──
    // Only load weights for target_layers (trainable) + embed/head/norm.
    // Non-target layers are skipped in forward (hidden passes through unchanged).
    let n_layers = runtime_config.num_hidden_layers;
    let _n_experts = runtime_config.n_routed_experts;
    let mut needed: HashSet<String> = HashSet::new();
    needed.insert("model.embed_tokens.weight".to_string());
    needed.insert("model.norm.weight".to_string());
    if !runtime_config.tie_word_embeddings {
        needed.insert("lm_head.weight".to_string());
    }

    for layer in 0..n_layers {
        let p = format!("model.layers.{layer}");
        // Attention (all layers, replicated)
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
            // FP8: also load scale_inv companion
            needed.insert(format!("{p}.self_attn.{suffix}_scale_inv"));
        }
        // Indexer weights (only for "full" layers)
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
                // FP8 scale for indexer wk and wq_b
                if matches!(
                    *suffix,
                    "weights_proj.weight" | "wk.weight" | "wq_b.weight"
                ) {
                    needed.insert(format!("{p}.self_attn.indexer.{suffix}_scale_inv"));
                }
            }
        }
        // Gate + shared experts (all layers, replicated)
        needed.insert(format!("{p}.mlp.gate.weight"));
        if runtime_config.topk_method == "noaux_tc" {
            needed.insert(format!("{p}.mlp.gate.e_score_correction_bias"));
        }
        for suffix in &["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            needed.insert(format!("{p}.mlp.shared_experts.{suffix}"));
            // FP8 scale
            needed.insert(format!("{p}.mlp.shared_experts.{suffix}_scale_inv"));
        }
        // Only LOCAL experts (EP sharded)
        for &e in &ep_shard.local_expert_indices {
            for suffix in &["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
                needed.insert(format!("{p}.mlp.experts.{e}.{suffix}"));
                // FP8 scale
                needed.insert(format!("{p}.mlp.experts.{e}.{suffix}_scale_inv"));
            }
        }
        // Dense layers: gate_proj, up_proj, down_proj (not under shared_experts)
        if !runtime_config.is_moe_layer(layer) {
            for suffix in &["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
                needed.insert(format!("{p}.mlp.{suffix}"));
                needed.insert(format!("{p}.mlp.{suffix}_scale_inv"));
            }
        }
    }
    for mtp_idx in 0..runtime_config.num_nextn_predict_layers {
        let mtp_layer = glm5_mtp_layer_index(&runtime_config, mtp_idx)?;
        add_mtp_decoder_weights(
            &mut needed,
            mtp_layer,
            &ep_shard.local_expert_indices,
            &runtime_config.topk_method,
        );
        info!(
            rank,
            mtp_idx, mtp_layer, "loading native GLM-5 MTP decoder layer"
        );
    }

    info!(rank, needed_tensors = needed.len(), "loading weights");

    // ── Load weights (mmap safetensors) ──
    let weights = load_glm5_weights(&model_path, &needed)?;
    info!(rank, tensors = weights.len(), "weights loaded");

    // Expert offloading: keep routed expert weights on CPU, prefetch to GPU on demand.
    // This saves ~87GB GPU memory (32 experts/rank × ~2.8GB → 0, only ~1-3 experts on GPU at a time).
    // Experts are frozen (LoRA targets attention only), so no gradient/autograd complexity.
    let mut weights_gpu: BTreeMap<String, Tensor> = BTreeMap::new();
    let mut expert_weights_cpu: BTreeMap<String, Tensor> = BTreeMap::new();
    for (name, t) in weights {
        if name.contains(".mlp.experts.") {
            // Expert weights stay on CPU — will be prefetched to GPU on demand
            expert_weights_cpu.insert(name, t);
        } else {
            let t = if t.kind() == Kind::Float8e4m3fn {
                // Keep FP8 — move to GPU as-is (forward uses _scaled_mm)
                t.to_device(device)
            } else if t.kind() == Kind::Float {
                // Scales (weight_scale_inv) stay F32 on GPU
                t.to_device(device)
            } else {
                // BF16 and other weights
                t.to_device(device).to_kind(compute_kind)
            };
            weights_gpu.insert(name, t);
        }
    }
    let expert_cpu_count = expert_weights_cpu.len();
    info!(
        rank,
        tensors_on_gpu = weights_gpu.len(),
        expert_tensors_on_cpu = expert_cpu_count,
        "weights loaded (experts offloaded to CPU)"
    );

    // ── Create LoRA registry ──
    let mut registry = Glm5LoraRegistry::new(&weights_gpu, lora_config, device)?;
    let trainable_count = registry.var_store.trainable_variables().len();
    info!(
        rank,
        trainable_params = trainable_count,
        "LoRA adapters created"
    );

    // ── Create persistent NCCL communicator FIRST (before barrier) ──
    let nccl_comm = if world_size > 1 {
        let comm_dir = config.run.base_dir.join(&config.run.name).join("nccl-comm");
        let comm = NcclPersistentComm::new(&comm_dir)?;
        info!(rank, "persistent NCCL communicator created");
        Some(comm)
    } else {
        None
    };

    // ── Barrier: NCCL all-reduce (fast, no file system polling) ──
    if world_size > 1 {
        info!(rank, "NCCL barrier: waiting for all ranks");
        let barrier_tensor = Tensor::zeros([], (Kind::Float, device));
        let _ = nccl_comm.as_ref().unwrap().all_reduce(&barrier_tensor);
        info!(rank, "all ranks ready (NCCL barrier passed)");
    } else {
        info!(rank, "single rank, no barrier needed");
    }

    // ── SFT data ──
    let tokenizer = tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
    let sft_jsonl = {
        let p = std::path::Path::new("/vePFS-Mindverse/user/nolanho/bin/glm5/sft_data.jsonl");
        if p.exists() {
            p.to_path_buf()
        } else {
            std::path::PathBuf::from("data/sft/deepseek_test.jsonl")
        }
    };
    let train_dataset = if sft_jsonl.exists() {
        info!(rank, path = %sft_jsonl.display(), "loading real SFT data");
        Glm5SftDataset::from_jsonl_simple(&sft_jsonl, &tokenizer)?
    } else {
        info!(rank, "no SFT JSONL found, using synthetic data");
        Glm5SftDataset::synthetic(&tokenizer)?
    };
    if train_dataset.samples.is_empty() {
        bail!("GLM-5 SFT dataset contains no samples");
    }
    // Megatron reads one extra raw token for S model positions so that the
    // input and next-token label tensors both have length S.
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
    // Build one optimizer-step batch in deterministic rank-strided dataset
    // order. EP ranks intentionally see different samples: the owner
    // dispatch must combine token assignments across the expert group rather
    // than summing same-position partial outputs.
    let accumulation_dataset = Glm5SftDataset {
        samples: (0..accumulation_batch_size)
            .map(|index| {
                let sample_index =
                    (rank * accumulation_batch_size + index) % train_dataset.samples.len();
                train_dataset.samples[sample_index].clone()
            })
            .collect(),
        pad_token_id: train_dataset.pad_token_id,
    };
    // Each EP rank owns a distinct local batch. The C++ MoE path performs the
    // corresponding differentiable owner dispatch/return for every trunk and
    // native MTP layer.
    let raw_batch = accumulation_dataset.padded_batch(0, accumulation_batch_size, device);

    let target_seq = if mtp_enabled {
        glm5_megatron_raw_seq_len(config.model.seq_len as i64)?
    } else {
        config.model.seq_len as i64
    };
    let actual_seq = raw_batch.input_ids.size()[1];
    let train_batch = if actual_seq > target_seq {
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

    // Megatron's raw extra token is a label source, not a model position. The
    // trunk consumes exactly S tokens; MTP CE keeps the S+1 raw-token tensor.
    let model_input_ids = if mtp_enabled {
        train_batch
            .input_ids
            .narrow(1, 0, config.model.seq_len as i64)
    } else {
        train_batch.input_ids.shallow_clone()
    };
    let mtp_embedding_ids = if mtp_enabled {
        Tensor::cat(
            &[
                &model_input_ids,
                &Tensor::full(
                    [model_input_ids.size()[0], 2],
                    train_dataset.pad_token_id,
                    (Kind::Int64, device),
                ),
            ],
            1,
        )
    } else {
        train_batch.input_ids.shallow_clone()
    };
    let mtp_target_ids = if mtp_enabled {
        Tensor::cat(
            &[
                &train_batch.input_ids,
                &Tensor::full(
                    [train_batch.input_ids.size()[0], 1],
                    train_dataset.pad_token_id,
                    (Kind::Int64, device),
                ),
            ],
            1,
        )
    } else {
        train_batch.input_ids.shallow_clone()
    };
    let mtp_target_mask = if mtp_enabled {
        Tensor::cat(
            &[
                &train_batch.target_mask,
                &Tensor::zeros(
                    [train_batch.target_mask.size()[0], 1],
                    (train_batch.target_mask.kind(), device),
                ),
            ],
            1,
        )
    } else {
        train_batch.target_mask.shallow_clone()
    };

    // ── Optimizer state (Adam) ──
    let lr = config.train.learning_rate as f64;
    let beta1 = config.train.adam_beta1 as f64;
    let beta2 = config.train.adam_beta2 as f64;
    let eps = config.train.adam_eps as f64;
    let trainable_vars = registry.var_store.trainable_variables();
    let mut adam_m: Vec<Tensor> = trainable_vars.iter().map(Tensor::zeros_like).collect();
    let mut adam_v: Vec<Tensor> = trainable_vars.iter().map(Tensor::zeros_like).collect();

    let mut initial_loss = 0.0_f64;
    let mut loss_history_last = 0.0_f64;

    // Pre-load indexer weights for every full source layer (including sources
    // shared by a LoRA target layer).
    let mut indexer_weights_map: BTreeMap<usize, Glm5AttentionWeights> = BTreeMap::new();
    for layer in 0..n_layers {
        let indexer_type = runtime_config
            .indexer_types
            .get(layer)
            .map(|s| s.as_str())
            .unwrap_or("full");
        if indexer_type == "full" {
            let attn = Glm5AttentionWeights::load_with_kind(&weights_gpu, layer, compute_kind)?;
            indexer_weights_map.insert(layer, attn);
        }
    }

    // Gradient checkpointing: saves MLP intermediate activations by recomputing during backward.
    // MLP is pure (no index_share_state mutation), so it's safe to checkpoint.
    // Attention uses SDPA (O(S) memory), so no checkpointing needed there.
    let use_checkpointing = true;
    // Attention mutates IndexShare state; checkpoint recomputation would need
    // a separate immutable input state and is disabled until that contract is
    // implemented. MLP checkpointing remains enabled below.
    let use_attention_checkpointing = false;

    // ── C++ kernel availability ──
    let cpp_router_supported =
        matches!(runtime_config.scoring_func.as_str(), "sigmoid" | "softmax")
            && matches!(
                runtime_config.topk_method.as_str(),
                "groupwise" | "noaux_tc"
            )
            && (runtime_config.topk_method != "noaux_tc"
                || runtime_config.scoring_func == "sigmoid");
    // The C++ attention path remains single-rank because its full-layer ABI has
    // no sequence-parallel/EP group descriptors. The standalone C++ MoE path,
    // however, owns differentiable EP dispatch/return and is safe for EP-only.
    let use_cpp_attention = rustrain_deepseek_v4::fp8_kernel::is_glm5_attention_available()
        && world_size == 1
        && runtime_config
            .rope_scaling_type
            .as_deref()
            .is_none_or(|kind| kind == "default")
        && cpp_router_supported;
    if use_cpp_attention {
        info!(
            rank,
            "C++ GLM5 attention kernel available — using coarse-grained C++ path"
        );
    } else {
        info!(
            rank,
            "C++ GLM5 attention kernel not available — using Rust tch-rs path"
        );
    }
    let use_cpp_mlp = rustrain_deepseek_v4::fp8_kernel::is_glm5_attention_available()
        && runtime_config
            .rope_scaling_type
            .as_deref()
            .is_none_or(|kind| kind == "default")
        && cpp_router_supported;
    if runtime_config.num_nextn_predict_layers > 0 && world_size > 1 && !use_cpp_mlp {
        bail!(
            "GLM-5 native MTP with EP>1 requires the compiled C++ autograd-aware owner dispatch/return kernel; Rust same-position MoE fallback is disabled"
        );
    }
    let use_cpp_loss = use_cpp_attention; // same .so provides loss
    let use_cpp_optimizer = use_cpp_attention; // same .so provides optimizer

    // ── Pre-expand caching allocator ──
    // Apply the configured PyTorch caching-allocator device limit.
    // This causes it to pre-allocate large segments upfront instead of growing incrementally.
    rustrain_deepseek_v4::fp8_kernel::set_memory_fraction(
        config.train.cuda_memory_fraction,
        local_rank as i32,
    );
    info!(
        rank,
        memory_fraction = config.train.cuda_memory_fraction,
        "set caching allocator memory fraction"
    );

    // ── Cache expert weights on GPU (eliminates per-layer CPU→GPU transfer) ──
    // Expert weights are frozen (LoRA targets attention only), so they can be
    // loaded once and reused every step. This eliminates ~87GB/layer PCIe transfer.
    info!(
        rank,
        predequant_fp8 = config.train.predequant_expert_weights,
        "caching expert weights on GPU"
    );
    let mut expert_weights_gpu: BTreeMap<String, Tensor> = BTreeMap::new();
    // First pass: load all to GPU
    for (name, t) in &expert_weights_cpu {
        let gpu_t = t.to_device(device);
        expert_weights_gpu.insert(name.clone(), gpu_t);
    }
    if config.train.predequant_expert_weights {
        // Pre-dequant FP8 weights once to avoid repeating it in safe_linear.
        let expert_weight_names: Vec<String> = expert_weights_gpu.keys().cloned().collect();
        for name in &expert_weight_names {
            if let Some(t) = expert_weights_gpu.get(name) {
                if t.kind() == Kind::Float8e4m3fn {
                    let scale_name = name.replace(".weight", ".weight_scale_inv");
                    if let Some(scale) = expert_weights_gpu.get(&scale_name) {
                        match rustrain_deepseek_v4::fp8_kernel::dequant_fp8_weight(t, scale) {
                            Ok(bf16) => {
                                expert_weights_gpu.insert(name.clone(), bf16.to_kind(compute_kind));
                            }
                            Err(e) => {
                                tracing::warn!(
                                    rank,
                                    "pre-dequant failed for {}: {:?}, keeping FP8",
                                    name,
                                    e
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    let expert_gpu_count = expert_weights_gpu.len();
    info!(
        rank,
        expert_tensors_on_gpu = expert_gpu_count,
        predequant_fp8 = config.train.predequant_expert_weights,
        "expert weights cached on GPU"
    );

    // ── Pre-extract constant tensors (avoid per-step BTreeMap lookups) ──
    let embed_weight = tensor(&weights_gpu, "model.embed_tokens.weight")?.to_kind(compute_kind);
    let final_norm_weight = tensor(&weights_gpu, "model.norm.weight")?.to_kind(compute_kind);
    let lm_head_weight = if runtime_config.tie_word_embeddings {
        embed_weight.shallow_clone()
    } else {
        tensor(&weights_gpu, "lm_head.weight")?.to_kind(compute_kind)
    };
    let mtp_projection: Vec<(usize, Glm5MtpProjectionWeights)> =
        glm5_mtp_layer_indices(&runtime_config)?
            .into_iter()
            .map(|mtp_layer| {
                Ok((
                    mtp_layer,
                    Glm5MtpProjectionWeights::load_with_kind(
                        &weights_gpu,
                        mtp_layer,
                        compute_kind,
                    )?,
                ))
            })
            .collect::<Result<_>>()?;
    let mtp_decoder_attention: BTreeMap<usize, Glm5AttentionWeights> = mtp_projection
        .iter()
        .map(|(layer, _)| {
            Ok((
                *layer,
                Glm5AttentionWeights::load_with_kind(&weights_gpu, *layer, compute_kind)?,
            ))
        })
        .collect::<Result<_>>()?;

    // ── Training loop ──
    for step in 0..config.train.max_steps {
        let aggregate_base_token_count_local = train_batch
            .target_mask
            .narrow(1, 1, train_batch.target_mask.size()[1] - 1)
            .to_kind(Kind::Float)
            .sum(Kind::Float);
        let aggregate_base_token_count = if world_size > 1 {
            nccl_comm
                .as_ref()
                .unwrap()
                .all_reduce(&aggregate_base_token_count_local)?
        } else {
            aggregate_base_token_count_local
        }
        .clamp_min(1.0);
        let mut accumulated_loss_val = 0.0_f64;
        let mut accumulated_mtp_loss_val = 0.0_f64;

        for accumulation_index in 0..accumulation_steps {
            let batch_start = (accumulation_index * micro_batch_size) as i64;
            let batch_len = micro_batch_size as i64;
            let train_batch = Glm5SftBatch {
                input_ids: train_batch.input_ids.narrow(0, batch_start, batch_len),
                target_mask: train_batch.target_mask.narrow(0, batch_start, batch_len),
                num_masked: train_batch.num_masked,
            };
            let model_input_ids = model_input_ids.narrow(0, batch_start, batch_len);
            let mtp_embedding_ids = mtp_embedding_ids.narrow(0, batch_start, batch_len);
            let mtp_target_ids = mtp_target_ids.narrow(0, batch_start, batch_len);
            let mtp_target_mask = mtp_target_mask.narrow(0, batch_start, batch_len);
            let base_token_count_local = train_batch
                .target_mask
                .narrow(1, 1, train_batch.target_mask.size()[1] - 1)
                .to_kind(Kind::Float)
                .sum(Kind::Float);
            let base_token_count = if world_size > 1 {
                nccl_comm
                    .as_ref()
                    .unwrap()
                    .all_reduce(&base_token_count_local)?
            } else {
                base_token_count_local
            };
            let microbatch_weight = &base_token_count / &aggregate_base_token_count;

            // ── Forward ──
            let embed = &embed_weight;
            let mut hidden = Tensor::embedding(embed, &model_input_ids, -1, false, false);
            if hidden.kind() != compute_kind {
                hidden = hidden.to_kind(compute_kind);
            }

            let mut index_share_state: Option<IndexShareState> = None;
            // C++ IndexShare state for layer_forward path (separate from Rust path)
            let mut cpp_layer_state = rustrain_deepseek_v4::fp8_kernel::Glm5IndexState::default();

            // Async pipeline: (output_tensor, cuda_event) from previous layer's all-reduce.
            // Next layer waits on event (GPU-side, no CPU block) before using output.
            let mut pending_allreduce: Option<(Tensor, rustrain_nccl::nccl::CudaEventHandle)> =
                None;

            for layer in 0..n_layers {
                let p = format!("model.layers.{layer}");

                if use_cpp_attention {
                    // ── C++ unified layer forward: 1 FFI call for entire layer ──
                    // Combines: RMSNorm → attention → residual → RMSNorm → MoE/dense → residual
                    let attn_norm = tensor(&weights_gpu, &format!("{p}.input_layernorm.weight"))?
                        .to_kind(compute_kind);
                    let post_norm = tensor(
                        &weights_gpu,
                        &format!("{p}.post_attention_layernorm.weight"),
                    )?
                    .to_kind(compute_kind);

                    // Load attention weights (with LoRA applied)
                    let attn_weights =
                        Glm5AttentionWeights::load_with_kind(&weights_gpu, layer, compute_kind)?;
                    let lora_attn = lora_attention_weights(&attn_weights, layer, &mut registry)?;

                    // Get indexer weights
                    let source = runtime_config.indexer_source_layer(layer);
                    let indexer_weights = indexer_weights_map.get(&source).unwrap_or(&lora_attn);

                    let is_full_layer = !runtime_config.should_skip_topk(layer)
                        && runtime_config
                            .indexer_types
                            .get(layer)
                            .map(|kind| kind == "full")
                            .unwrap_or(true);

                    let is_moe = runtime_config.is_moe_layer(layer);

                    // ── Wait for previous layer's async all-reduce to complete (GPU-side, no CPU block) ──
                    if let Some((prev_output, prev_event)) = pending_allreduce.take() {
                        rustrain_deepseek_v4::fp8_kernel::stream_wait_event(
                            local_rank as i32,
                            &prev_event,
                        );
                        hidden = prev_output;
                    }

                    if is_moe {
                        // ── MoE layer ──
                        let gate = tensor(&weights_gpu, &format!("{p}.mlp.gate.weight"))?
                            .to_kind(compute_kind);
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

                        // Build expert weight arrays from GPU-cached weights
                        let p_str = format!("{p}.mlp.experts.");
                        let mut egw: Vec<&Tensor> = Vec::new();
                        let mut euw: Vec<&Tensor> = Vec::new();
                        let mut edw: Vec<&Tensor> = Vec::new();
                        let mut egs: Vec<Option<&Tensor>> = Vec::new();
                        let mut eus: Vec<Option<&Tensor>> = Vec::new();
                        let mut eds: Vec<Option<&Tensor>> = Vec::new();
                        for &global_e in &ep_shard.local_expert_indices {
                            let eg = format!("{p_str}{global_e}");
                            egw.push(
                                expert_weights_gpu
                                    .get(&format!("{eg}.gate_proj.weight"))
                                    .unwrap(),
                            );
                            euw.push(
                                expert_weights_gpu
                                    .get(&format!("{eg}.up_proj.weight"))
                                    .unwrap(),
                            );
                            edw.push(
                                expert_weights_gpu
                                    .get(&format!("{eg}.down_proj.weight"))
                                    .unwrap(),
                            );
                            egs.push(
                                expert_weights_gpu.get(&format!("{eg}.gate_proj.weight_scale_inv")),
                            );
                            eus.push(
                                expert_weights_gpu.get(&format!("{eg}.up_proj.weight_scale_inv")),
                            );
                            eds.push(
                                expert_weights_gpu.get(&format!("{eg}.down_proj.weight_scale_inv")),
                            );
                        }

                        let partial_mlp = rustrain_deepseek_v4::fp8_kernel::glm5_layer_forward_cpp(
                            &hidden,
                            &attn_norm,
                            &post_norm,
                            &lora_attn.q_a_proj,
                            &lora_attn.q_a_layernorm,
                            &lora_attn.q_b_proj,
                            &lora_attn.kv_a_proj_with_mqa,
                            &lora_attn.kv_a_layernorm,
                            &lora_attn.kv_b_proj,
                            &lora_attn.o_proj,
                            lora_attn.q_a_proj_scale.as_ref(),
                            lora_attn.q_b_proj_scale.as_ref(),
                            lora_attn.kv_a_proj_scale.as_ref(),
                            lora_attn.kv_b_proj_scale.as_ref(),
                            lora_attn.o_proj_scale.as_ref(),
                            indexer_weights.indexer_wq_b.as_ref(),
                            indexer_weights.indexer_wk.as_ref(),
                            indexer_weights.indexer_k_norm_weight.as_ref(),
                            indexer_weights.indexer_k_norm_bias.as_ref(),
                            indexer_weights.indexer_weights_proj.as_ref(),
                            indexer_weights.indexer_weights_proj_scale.as_ref(),
                            indexer_weights.indexer_wq_b_scale.as_ref(),
                            indexer_weights.indexer_wk_scale.as_ref(),
                            Some(&gate),
                            Some(&shared_gate),
                            Some(&shared_up),
                            Some(&shared_down),
                            shared_gate_scale,
                            shared_up_scale,
                            shared_down_scale,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None, // dense weights (MoE)
                            &egw,
                            &euw,
                            &edw,
                            &egs,
                            &eus,
                            &eds,
                            &ep_shard.local_expert_indices,
                            hidden.size()[0] as i32,
                            hidden.size()[1] as i32,
                            runtime_config.num_attention_heads as i32,
                            runtime_config.qk_nope_head_dim as i32,
                            runtime_config.qk_rope_head_dim as i32,
                            runtime_config.v_head_dim as i32,
                            runtime_config.kv_lora_rank as i32,
                            runtime_config.index_head_dim as i32,
                            runtime_config.index_n_heads as i32,
                            runtime_config.index_topk as i32,
                            runtime_config.index_topk_freq as i32,
                            layer as i32,
                            is_full_layer,
                            true,
                            runtime_config.n_routed_experts as i32,
                            runtime_config.num_experts_per_tok as i32,
                            runtime_config.rms_norm_eps,
                            runtime_config.rope_theta,
                            runtime_config.rope_interleave,
                            runtime_config.routed_scaling_factor,
                            local_rank as i32,
                            &mut cpp_layer_state,
                        )?;

                        // All-reduce MoE output — preserve autograd graph for LoRA backward
                        // Trick: hidden = partial_mlp + (reduced - partial_mlp).detach()
                        // Forward value = reduced (correct), backward gradient flows to partial_mlp (coef 1)
                        // LoRA gradient all-reduce later averages across ranks (÷ world_size)
                        let mlp_kind = partial_mlp.kind();
                        if world_size > 1 {
                            let reduced = nccl_comm
                                .as_ref()
                                .unwrap()
                                .all_reduce(&partial_mlp)
                                .unwrap_or_else(|_| partial_mlp.shallow_clone());
                            let full = (&reduced / (world_size as f64)).to_kind(mlp_kind);
                            // identity trick: forward = full, backward grad → partial_mlp (coef 1)
                            hidden = &partial_mlp + &(&full - &partial_mlp).detach();
                        } else {
                            hidden = partial_mlp.shallow_clone();
                        }
                    } else {
                        // ── Dense layer ──
                        let gate_w = keep_fp8(
                            tensor(&weights_gpu, &format!("{p}.mlp.gate_proj.weight"))?,
                            compute_kind,
                        );
                        let up_w = keep_fp8(
                            tensor(&weights_gpu, &format!("{p}.mlp.up_proj.weight"))?,
                            compute_kind,
                        );
                        let down_w = keep_fp8(
                            tensor(&weights_gpu, &format!("{p}.mlp.down_proj.weight"))?,
                            compute_kind,
                        );
                        let gate_scale =
                            weights_gpu.get(&format!("{p}.mlp.gate_proj.weight_scale_inv"));
                        let up_scale =
                            weights_gpu.get(&format!("{p}.mlp.up_proj.weight_scale_inv"));
                        let down_scale =
                            weights_gpu.get(&format!("{p}.mlp.down_proj.weight_scale_inv"));

                        hidden = rustrain_deepseek_v4::fp8_kernel::glm5_layer_forward_cpp(
                            &hidden,
                            &attn_norm,
                            &post_norm,
                            &lora_attn.q_a_proj,
                            &lora_attn.q_a_layernorm,
                            &lora_attn.q_b_proj,
                            &lora_attn.kv_a_proj_with_mqa,
                            &lora_attn.kv_a_layernorm,
                            &lora_attn.kv_b_proj,
                            &lora_attn.o_proj,
                            lora_attn.q_a_proj_scale.as_ref(),
                            lora_attn.q_b_proj_scale.as_ref(),
                            lora_attn.kv_a_proj_scale.as_ref(),
                            lora_attn.kv_b_proj_scale.as_ref(),
                            lora_attn.o_proj_scale.as_ref(),
                            indexer_weights.indexer_wq_b.as_ref(),
                            indexer_weights.indexer_wk.as_ref(),
                            indexer_weights.indexer_k_norm_weight.as_ref(),
                            indexer_weights.indexer_k_norm_bias.as_ref(),
                            indexer_weights.indexer_weights_proj.as_ref(),
                            indexer_weights.indexer_weights_proj_scale.as_ref(),
                            indexer_weights.indexer_wq_b_scale.as_ref(),
                            indexer_weights.indexer_wk_scale.as_ref(),
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None, // MoE weights (none for dense)
                            Some(&gate_w),
                            Some(&up_w),
                            Some(&down_w),
                            gate_scale,
                            up_scale,
                            down_scale,
                            &[],
                            &[],
                            &[],
                            &[],
                            &[],
                            &[], // expert weights (none)
                            &[],
                            hidden.size()[0] as i32,
                            hidden.size()[1] as i32,
                            runtime_config.num_attention_heads as i32,
                            runtime_config.qk_nope_head_dim as i32,
                            runtime_config.qk_rope_head_dim as i32,
                            runtime_config.v_head_dim as i32,
                            runtime_config.kv_lora_rank as i32,
                            runtime_config.index_head_dim as i32,
                            runtime_config.index_n_heads as i32,
                            runtime_config.index_topk as i32,
                            runtime_config.index_topk_freq as i32,
                            layer as i32,
                            is_full_layer,
                            false,
                            runtime_config.n_routed_experts as i32,
                            runtime_config.num_experts_per_tok as i32,
                            runtime_config.rms_norm_eps,
                            runtime_config.rope_theta,
                            runtime_config.rope_interleave,
                            runtime_config.routed_scaling_factor,
                            local_rank as i32,
                            &mut cpp_layer_state,
                        )?;
                    }

                    if hidden.kind() != compute_kind {
                        hidden = hidden.to_kind(compute_kind);
                    }
                    continue; // Skip the old Rust path below
                }

                // ── Rust fallback path (when C++ kernel not available) ──
                // ── Attention ──
                let attn_norm = tensor(&weights_gpu, &format!("{p}.input_layernorm.weight"))?
                    .to_kind(compute_kind);
                let hidden_norm = if use_cpp_attention {
                    rustrain_deepseek_v4::fp8_kernel::glm5_rms_norm_cpp(
                        &hidden,
                        &attn_norm,
                        runtime_config.rms_norm_eps,
                    )?
                } else {
                    rms_norm(&hidden, &attn_norm, runtime_config.rms_norm_eps)
                };

                // Load attention weights
                let attn_weights =
                    Glm5AttentionWeights::load_with_kind(&weights_gpu, layer, compute_kind)?;
                let lora_attn = lora_attention_weights(&attn_weights, layer, &mut registry)?;

                // Get indexer weights (from source layer for IndexShare)
                let source = runtime_config.indexer_source_layer(layer);
                let indexer_weights = indexer_weights_map.get(&source).unwrap_or(&lora_attn);

                // Determine if this is a "full" layer (computes indexer state) or "shared" (reuses)
                let is_full_layer = !runtime_config.should_skip_topk(layer)
                    && runtime_config
                        .indexer_types
                        .get(layer)
                        .map(|kind| kind == "full")
                        .unwrap_or(true);

                // ── C++ attention path: coarse-grained, one FFI call per layer ──
                // Rust path: fine-grained, ~30 tch-rs calls per layer (with checkpoint)
                let attn_out = if use_cpp_attention {
                    // C++ path — no checkpointing needed (C++ manages intermediates on stack)
                    // Convert Rust IndexShareState to C++ Glm5IndexState
                    let mut cpp_state = rustrain_deepseek_v4::fp8_kernel::Glm5IndexState::default();
                    if index_share_state.is_some() {
                        // Rust state → C++ state: pass topk_indices/idx_bias_keys as at::Tensor*
                        // For now, C++ recomputes — we pass null state and let C++ compute fresh
                        // TODO: share state between Rust and C++ (needs at::Tensor* ↔ tch::Tensor conversion)
                    }
                    let result = rustrain_deepseek_v4::fp8_kernel::glm5_dsa_attention_cpp(
                        &hidden_norm,
                        &lora_attn.q_a_proj,
                        &lora_attn.q_a_layernorm,
                        &lora_attn.q_b_proj,
                        &lora_attn.kv_a_proj_with_mqa,
                        &lora_attn.kv_a_layernorm,
                        &lora_attn.kv_b_proj,
                        &lora_attn.o_proj,
                        lora_attn.q_a_proj_scale.as_ref(),
                        lora_attn.q_b_proj_scale.as_ref(),
                        lora_attn.kv_a_proj_scale.as_ref(),
                        lora_attn.kv_b_proj_scale.as_ref(),
                        lora_attn.o_proj_scale.as_ref(),
                        indexer_weights.indexer_wq_b.as_ref(),
                        indexer_weights.indexer_wk.as_ref(),
                        indexer_weights.indexer_k_norm_weight.as_ref(),
                        indexer_weights.indexer_k_norm_bias.as_ref(),
                        indexer_weights.indexer_weights_proj.as_ref(),
                        indexer_weights.indexer_weights_proj_scale.as_ref(),
                        indexer_weights.indexer_wq_b_scale.as_ref(),
                        indexer_weights.indexer_wk_scale.as_ref(),
                        hidden.size()[0] as i32,
                        hidden.size()[1] as i32,
                        runtime_config.num_attention_heads as i32,
                        runtime_config.qk_nope_head_dim as i32,
                        runtime_config.qk_rope_head_dim as i32,
                        runtime_config.v_head_dim as i32,
                        runtime_config.kv_lora_rank as i32,
                        runtime_config.index_head_dim as i32,
                        runtime_config.index_n_heads as i32,
                        runtime_config.index_topk as i32,
                        runtime_config.index_topk_freq as i32,
                        layer as i32,
                        is_full_layer,
                        runtime_config.rms_norm_eps,
                        runtime_config.rope_theta,
                        runtime_config.rope_interleave,
                        local_rank as i32,
                        &mut cpp_state,
                    )?;
                    // C++ state is internal — for now, reset Rust state (C++ recomputes each full layer)
                    if is_full_layer {
                        index_share_state = None;
                    }
                    result
                } else if use_attention_checkpointing {
                    // Rust path with checkpointing
                    let state_mutex = Arc::new(Mutex::new(index_share_state.take()));
                    let state_for_closure = state_mutex.clone();
                    let attn_clone = lora_attn.clone();
                    let indexer_clone = indexer_weights.clone();
                    let runtime_clone = runtime_config.clone();
                    let layer_copy = layer;
                    let full_layer = is_full_layer;
                    let result =
                        rustrain_deepseek_v4::fp8_kernel::checkpoint(&hidden_norm, move |input| {
                            let mut guard = state_for_closure.lock().unwrap();
                            let mut local_state = guard.take();
                            if full_layer {
                                local_state = None;
                            }
                            let output = glm5_dsa_attention(
                                input,
                                &attn_clone,
                                &indexer_clone,
                                &runtime_clone,
                                &mut local_state,
                                layer_copy,
                            );
                            *guard = local_state;
                            output
                        });
                    index_share_state = state_mutex.lock().unwrap().take();
                    result
                } else {
                    glm5_dsa_attention(
                        &hidden_norm,
                        &lora_attn,
                        indexer_weights,
                        &runtime_config,
                        &mut index_share_state,
                        layer,
                    )
                }
                .to_kind(compute_kind);

                let residual = &hidden + &attn_out;

                // ── MoE / Dense MLP ──
                let post_norm = tensor(
                    &weights_gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?
                .to_kind(compute_kind);
                let mlp_input = if use_cpp_attention {
                    rustrain_deepseek_v4::fp8_kernel::glm5_rms_norm_cpp(
                        &residual,
                        &post_norm,
                        runtime_config.rms_norm_eps,
                    )?
                } else {
                    rms_norm(&residual, &post_norm, runtime_config.rms_norm_eps)
                };

                if runtime_config.is_moe_layer(layer) {
                    // MoE with EP
                    let gate = tensor(&weights_gpu, &format!("{p}.mlp.gate.weight"))?
                        .to_kind(compute_kind);
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

                    if use_cpp_mlp {
                        // ── C++ MoE: one FFI call for routing + dispatch + shared + combine ──
                        // Build expert weight arrays from CPU-offloaded weights
                        let p_str = format!("{p}.mlp.experts.");
                        let mut egw: Vec<&Tensor> = Vec::new();
                        let mut euw: Vec<&Tensor> = Vec::new();
                        let mut edw: Vec<&Tensor> = Vec::new();
                        let mut egs: Vec<Option<&Tensor>> = Vec::new();
                        let mut eus: Vec<Option<&Tensor>> = Vec::new();
                        let mut eds: Vec<Option<&Tensor>> = Vec::new();
                        for &global_e in &ep_shard.local_expert_indices {
                            let eg = format!("{p_str}{global_e}");
                            egw.push(
                                expert_weights_gpu
                                    .get(&format!("{eg}.gate_proj.weight"))
                                    .unwrap(),
                            );
                            euw.push(
                                expert_weights_gpu
                                    .get(&format!("{eg}.up_proj.weight"))
                                    .unwrap(),
                            );
                            edw.push(
                                expert_weights_gpu
                                    .get(&format!("{eg}.down_proj.weight"))
                                    .unwrap(),
                            );
                            egs.push(
                                expert_weights_gpu.get(&format!("{eg}.gate_proj.weight_scale_inv")),
                            );
                            eus.push(
                                expert_weights_gpu.get(&format!("{eg}.up_proj.weight_scale_inv")),
                            );
                            eds.push(
                                expert_weights_gpu.get(&format!("{eg}.down_proj.weight_scale_inv")),
                            );
                        }
                        let full_mlp = rustrain_deepseek_v4::fp8_kernel::glm5_moe_layer_ep_cpp(
                            &mlp_input,
                            &shared_gate,
                            &shared_up,
                            &shared_down,
                            shared_gate_scale,
                            shared_up_scale,
                            shared_down_scale,
                            &gate,
                            weights_gpu.get(&format!("{p}.mlp.gate.e_score_correction_bias")),
                            &egw,
                            &euw,
                            &edw,
                            &egs,
                            &eus,
                            &eds,
                            &ep_shard.local_expert_indices,
                            runtime_config.n_routed_experts as i32,
                            runtime_config.num_experts_per_tok as i32,
                            runtime_config.n_group as i32,
                            runtime_config.topk_group as i32,
                            match runtime_config.scoring_func.as_str() {
                                "sigmoid" => 0,
                                "softmax" => 1,
                                other => bail!("unsupported GLM5 scoring_func {other:?}"),
                            },
                            match runtime_config.topk_method.as_str() {
                                "groupwise" => 0,
                                "noaux_tc" => 1,
                                other => bail!("unsupported GLM5 topk_method {other:?}"),
                            },
                            runtime_config.norm_topk_prob,
                            runtime_config.routed_scaling_factor,
                            nccl_comm
                                .as_ref()
                                .map_or(std::ptr::null_mut(), NcclPersistentComm::raw_comm_ptr),
                            rank as i32,
                            world_size as i32,
                            local_rank as i32,
                        )?;
                        hidden = &residual + &full_mlp;
                    } else {
                        // ── Rust MoE path (fallback) ──
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

                        // Router logits — computed over ALL experts
                        let router_logits = mlp_input
                            .linear::<&Tensor>(&gate, None)
                            .to_kind(Kind::Float);
                        let correction_bias =
                            weights_gpu.get(&format!("{p}.mlp.gate.e_score_correction_bias"));
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

                        // Flatten for per-token dispatch
                        let flat_input = mlp_input.reshape([-1, mlp_input.size()[2]]);
                        let tk_indices = topk_indices.reshape([-1, k]);
                        let tk_weights = topk_weights.reshape([-1, k]);

                        // Only apply LOCAL experts (EP sharded)
                        let mut partial_output =
                            Tensor::zeros(flat_input.size(), (compute_kind, flat_input.device()));

                        for &global_e in &ep_shard.local_expert_indices {
                            // Check which tokens selected this expert
                            let mask = tk_indices.eq(global_e as i64).to_kind(compute_kind);
                            let mask_flat = mask
                                .sum_dim_intlist([-1].as_slice(), false, compute_kind)
                                .to_kind(compute_kind);
                            let count = mask_flat.sum(compute_kind).double_value(&[]) as i64;
                            if count == 0 {
                                continue;
                            }
                            let eg = format!("{p}.mlp.experts.{global_e}");
                            // Expert weights from GPU cache (no CPU→GPU transfer)
                            let gate_w = expert_weights_gpu
                                .get(&format!("{eg}.gate_proj.weight"))
                                .with_context(|| {
                                    format!("expert weight not found: {eg}.gate_proj.weight")
                                })?;
                            let up_w = expert_weights_gpu
                                .get(&format!("{eg}.up_proj.weight"))
                                .with_context(|| {
                                    format!("expert weight not found: {eg}.up_proj.weight")
                                })?;
                            let down_w = expert_weights_gpu
                                .get(&format!("{eg}.down_proj.weight"))
                                .with_context(|| {
                                    format!("expert weight not found: {eg}.down_proj.weight")
                                })?;
                            let gate_w_scale =
                                expert_weights_gpu.get(&format!("{eg}.gate_proj.weight_scale_inv"));
                            let up_w_scale =
                                expert_weights_gpu.get(&format!("{eg}.up_proj.weight_scale_inv"));
                            let down_w_scale =
                                expert_weights_gpu.get(&format!("{eg}.down_proj.weight_scale_inv"));
                            let expert_out = if use_cpp_mlp {
                                rustrain_deepseek_v4::fp8_kernel::glm5_mlp_fp8_cpp(
                                    &flat_input,
                                    gate_w,
                                    up_w,
                                    down_w,
                                    gate_w_scale,
                                    up_w_scale,
                                    down_w_scale,
                                )?
                            } else {
                                glm5_mlp_fp8(
                                    &flat_input,
                                    gate_w,
                                    up_w,
                                    down_w,
                                    gate_w_scale,
                                    up_w_scale,
                                    down_w_scale,
                                )
                            };
                            let weighted_mask = (mask * &tk_weights)
                                .sum_dim_intlist([-1].as_slice(), false, compute_kind)
                                .to_kind(compute_kind);
                            let mask_expanded = weighted_mask
                                .unsqueeze(-1)
                                .expand([-1, expert_out.size()[1]], false);
                            let contribution = expert_out * &mask_expanded;
                            partial_output = partial_output + contribution;
                        }

                        let routed_partial = partial_output.reshape([1, -1, mlp_input.size()[2]]);
                        let routed_full = if world_size > 1 {
                            let reduced =
                                nccl_comm.as_ref().unwrap().all_reduce(&routed_partial)?;
                            let full = reduced.to_kind(routed_partial.kind());
                            &routed_partial + &(&full - &routed_partial).detach()
                        } else {
                            routed_partial
                        };
                        let full_mlp = routed_full + shared_output;

                        hidden = &residual + &full_mlp;
                    } // end Rust MoE fallback
                } else {
                    // Dense MLP — checkpointed to save intermediate activations
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

                    let mlp = if use_cpp_mlp {
                        rustrain_deepseek_v4::fp8_kernel::glm5_mlp_fp8_cpp(
                            &mlp_input,
                            &gate,
                            &up,
                            &down,
                            gate_scale.as_ref(),
                            up_scale.as_ref(),
                            down_scale.as_ref(),
                        )?
                    } else if use_checkpointing {
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

            // ── Drain pending async all-reduce before loss computation ──
            if let Some((final_output, event)) = pending_allreduce.take() {
                rustrain_deepseek_v4::fp8_kernel::stream_wait_event(local_rank as i32, &event);
                hidden = final_output;
            }

            // ── Final norm + lm_head ──
            let normed = if use_cpp_attention {
                rustrain_deepseek_v4::fp8_kernel::glm5_rms_norm_cpp(
                    &hidden,
                    &final_norm_weight,
                    runtime_config.rms_norm_eps,
                )?
            } else {
                rms_norm(&hidden, &final_norm_weight, runtime_config.rms_norm_eps)
            };
            let lm_head = &lm_head_weight;

            // ── Chunked SFT Loss ──
            let model_seq_len = model_input_ids.size()[1];
            let raw_seq_len = train_batch.input_ids.size()[1];
            let vocab = runtime_config.vocab_size;

            let lm_loss = if use_cpp_loss {
                // C++ cross-entropy loss (single call, chunked internally)
                rustrain_deepseek_v4::fp8_kernel::glm5_cross_entropy_loss_cpp(
                    &normed,
                    lm_head,
                    &train_batch.input_ids,
                    &train_batch.target_mask,
                    raw_seq_len as i32,
                    vocab as i32,
                    4096,
                    local_rank as i32,
                )?
            } else {
                // Rust chunked cross-entropy loss
                let shifted_targets = train_batch.input_ids.narrow(1, 1, raw_seq_len - 1);
                let shifted_mask = train_batch
                    .target_mask
                    .narrow(1, 1, raw_seq_len - 1)
                    .to_kind(Kind::Float);
                let total_mask = shifted_mask.sum(Kind::Float);
                let ce_chunk_size = 4096;
                let mut loss_acc = Tensor::zeros([], (Kind::Float, device));
                for start in (0..raw_seq_len - 1).step_by(ce_chunk_size as usize) {
                    let end = (start + ce_chunk_size as i64).min(raw_seq_len - 1);
                    let chunk_len = end - start;
                    let normed_chunk = normed.narrow(1, start, chunk_len);
                    let logits_chunk = normed_chunk.linear::<&Tensor>(lm_head, None);
                    let log_probs = logits_chunk
                        .reshape([-1, vocab])
                        .log_softmax(-1, Kind::Float);
                    let targets_chunk = shifted_targets.narrow(1, start, chunk_len).reshape([-1]);
                    let mask_chunk = shifted_mask.narrow(1, start, chunk_len);
                    let per_token_loss = log_probs
                        .g_nll_loss::<&Tensor>(&targets_chunk, None, Reduction::None, -100)
                        .reshape([1, chunk_len]);
                    let masked = &per_token_loss * &mask_chunk;
                    loss_acc = loss_acc + masked.sum(Kind::Float);
                }
                if world_size > 1 {
                    let reduced_sum = nccl_comm.as_ref().unwrap().all_reduce(&loss_acc)?;
                    let reduced_count = nccl_comm.as_ref().unwrap().all_reduce(&total_mask)?;
                    reattach_ep_token_mean(&loss_acc, &reduced_sum, &reduced_count)
                } else {
                    loss_acc / total_mask.clamp_min(1.0)
                }
            };

            let weighted_lm_loss = &lm_loss * &microbatch_weight;
            let loss = if mtp_projection.is_empty() {
                weighted_lm_loss
            } else {
                let seq_len = model_seq_len;
                let mut previous_mtp_block: Option<Tensor> = None;
                let mut mtp_losses = Vec::with_capacity(mtp_projection.len());

                for (mtp_idx, (mtp_layer, mtp)) in mtp_projection.iter().enumerate() {
                    let offset = mtp_idx as i64;
                    if seq_len < offset + 3 {
                        bail!(
                            "GLM-5 MTP layer {mtp_idx} requires sequence length >= {}, got {}",
                            offset + 3,
                            seq_len
                        );
                    }

                    // Layer 0 starts from the trunk post-final-norm hidden state;
                    // later layers consume the preceding MTP shared-head-normalized
                    // output. Offset alignment stays inside the C++ fusion call.
                    let prepare_hidden = previous_mtp_block.as_ref().unwrap_or(&normed);

                    let mtp_input = rustrain_deepseek_v4::fp8_kernel::glm5_mtp_prepare_cpp(
                        prepare_hidden,
                        &mtp_embedding_ids,
                        &embed_weight,
                        &mtp.enorm,
                        &mtp.hnorm,
                        &mtp.eh_proj,
                        mtp.eh_proj_scale.as_ref(),
                        runtime_config.rms_norm_eps,
                        (offset + 1) as i32,
                    )?;
                    let mtp_block = forward_mtp_decoder_layer_ep(
                        &mtp_input,
                        *mtp_layer,
                        mtp_decoder_attention
                            .get(mtp_layer)
                            .context("cached MTP attention weights are missing")?,
                        &weights_gpu,
                        &expert_weights_gpu,
                        &runtime_config,
                        &ep_shard,
                        nccl_comm.as_ref(),
                    )?;
                    let mtp_output =
                        rustrain_deepseek_v4::fp8_kernel::glm5_mtp_postprocess_loss_cpp(
                            &mtp_block,
                            &mtp.shared_head_norm,
                            &lm_head_weight,
                            None,
                            &mtp_target_ids,
                            &mtp_target_mask,
                            runtime_config.rms_norm_eps,
                            offset as i32,
                            512,
                        )?;
                    let layer_loss = if world_size > 1 {
                        let reduced_sum = nccl_comm
                            .as_ref()
                            .unwrap()
                            .all_reduce(&mtp_output.loss_sum)?;
                        let reduced_count = nccl_comm
                            .as_ref()
                            .unwrap()
                            .all_reduce(&mtp_output.token_count)?;
                        reattach_ep_token_mean(&mtp_output.loss_sum, &reduced_sum, &reduced_count)
                    } else {
                        mtp_output.loss
                    };
                    let weighted_layer_loss = &layer_loss * &microbatch_weight;
                    accumulated_mtp_loss_val += weighted_layer_loss.double_value(&[]);
                    mtp_losses.push(weighted_layer_loss);
                    previous_mtp_block = Some(mtp_output.normalized);
                }

                let combined = rustrain_deepseek_v4::fp8_kernel::glm5_combine_losses_cpp(
                    &weighted_lm_loss,
                    &mtp_losses,
                    config.train.mtp_loss_scaling_factor,
                )?;
                combined.total
            };

            let loss_val = loss.double_value(&[]);
            accumulated_loss_val += loss_val;

            // ── Backward ──
            loss.backward();

            // Free checkpoint closures (they hold GPU tensor references)
            rustrain_deepseek_v4::fp8_kernel::clear_checkpoint_registry();
        }

        loss_history_last = accumulated_loss_val;
        if step == 0 {
            initial_loss = accumulated_loss_val;
        }
        info!(
            rank,
            step = step + 1,
            loss = accumulated_loss_val,
            mtp_loss = if mtp_projection.is_empty() {
                None
            } else {
                Some(accumulated_mtp_loss_val)
            },
            accumulation_steps,
            "GLM-5 EP LoRA SFT train step"
        );

        if step == 0 {
            rustrain_deepseek_v4::fp8_kernel::empty_cache();
        }

        // ── LoRA gradient all-reduce ──
        // Note: all_reduce_async was attempted but caused undefined tensor issues
        // because the output tensor on comm_stream isn't visible to compute stream
        // without proper synchronization. Using sync all_reduce for correctness.
        // The MoE output all-reduce (line 547) uses async because it's on the
        // critical path of the layer loop; gradient sync is off the critical path.
        let synced_grads: Vec<Tensor> = if world_size > 1 {
            let vars = registry.var_store.trainable_variables();
            let mut synced = Vec::with_capacity(vars.len());
            for var in &vars {
                let g = var.grad();
                if g.defined() && g.numel() > 0 {
                    let reduced = nccl_comm.as_ref().unwrap().all_reduce(&g)?;
                    synced.push(no_grad(|| reduced.to_kind(g.kind())));
                } else {
                    synced.push(g.shallow_clone());
                }
            }
            synced
        } else {
            Vec::new()
        };

        // ── Adam optimizer step ──
        let mut current_vars = registry.var_store.trainable_variables();
        // Disable requires_grad during optimizer step (C++ uses in-place ops which
        // fail on leaf Variables with requires_grad=true)
        for v in &current_vars {
            let _ = v.set_requires_grad(false);
        }
        if use_cpp_optimizer {
            let grads: Vec<Tensor> = current_vars
                .iter()
                .enumerate()
                .map(|(i, _var)| {
                    if world_size > 1 {
                        synced_grads[i].shallow_clone()
                    } else {
                        current_vars[i].grad()
                    }
                })
                .collect();
            rustrain_deepseek_v4::fp8_kernel::adam_step_cpp(
                &mut current_vars,
                &grads,
                &mut adam_m,
                &mut adam_v,
                lr,
                beta1,
                beta2,
                eps,
                step as i32,
            );
            for var in current_vars.iter_mut() {
                var.zero_grad();
            }
        } else {
            // Rust Adam
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
        }
        // Re-enable requires_grad for next forward pass
        for v in &current_vars {
            let _ = v.set_requires_grad(true);
        }
        // Invalidate LoRA delta cache (params changed)
        registry.invalidate_delta_cache();
    }

    // ── Save LoRA adapter ──
    let adapter_output = run_paths
        .checkpoints
        .join("glm5-lora-adapter-ep.safetensors");
    registry.save(&adapter_output)?;
    info!(rank, adapter = %adapter_output.display(), "adapter saved");

    // This is the last measured training forward (the optimizer update for
    // that forward is applied after the measurement). Keep it separate from
    // `initial_loss`; callers must not interpret it as a post-update evaluation.
    let final_loss = if config.train.max_steps > 0 {
        loss_history_last
    } else {
        initial_loss
    };
    info!(rank, initial_loss, final_loss, "GLM-5 LoRA SFT EP complete");

    Ok(Glm5LoraSftSummary {
        adapter_output: adapter_output.display().to_string(),
        initial_loss,
        final_loss,
        trainable_params: trainable_count,
    })
}

/// Load GLM-5.2 weights from safetensors directory using mmap (V4 native parser)
/// Only reads needed tensors from each shard — much faster than Tensor::read_safetensors
fn load_glm5_weights(
    model_path: &std::path::Path,
    needed: &HashSet<String>,
) -> Result<BTreeMap<String, Tensor>> {
    // First, read the index to find which shards contain needed tensors
    let index_path = model_path.join("model.safetensors.index.json");
    let single = model_path.join("model.safetensors");

    // If single file, use V4's native loader directly
    if single.exists() {
        return rustrain_deepseek_v4::fp8_kernel::load_safetensors_native(
            &single, needed, -1, // CPU
        )
        .with_context(|| format!("failed to load {}", single.display()));
    }

    if !index_path.exists() {
        anyhow::bail!(
            "no model.safetensors or index file in {}",
            model_path.display()
        );
    }

    // Parse index to group needed tensors by shard
    let index_text = std::fs::read_to_string(&index_path)?;
    #[derive(serde::Deserialize)]
    struct SafetensorsIndex {
        weight_map: std::collections::HashMap<String, String>,
    }
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

    info!(
        shards_needed = shard_to_tensors.len(),
        total_shards = index
            .weight_map
            .values()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        tensors_needed = needed.len(),
        "loading weights via mmap (V4 native parser)"
    );

    let mut weights = BTreeMap::new();
    for (shard_file, tensor_names) in &shard_to_tensors {
        let shard_path = model_path.join(shard_file);
        let shard_needed: HashSet<String> = tensor_names.iter().cloned().collect();
        match rustrain_deepseek_v4::fp8_kernel::load_safetensors_native(
            &shard_path,
            &shard_needed,
            -1, // CPU first, will move to GPU later
        ) {
            Ok(shard_weights) => {
                for (name, t) in shard_weights {
                    weights.insert(name, t);
                }
            }
            Err(e) => {
                // V4 native parser may not support all dtypes — fall back to tch-rs
                tracing::warn!(shard = %shard_file, error = %e, "V4 native loader failed, falling back to tch-rs");
                let shard_tensors = Tensor::read_safetensors(&shard_path)?;
                let shard_map: BTreeMap<String, Tensor> = shard_tensors.into_iter().collect();
                for name in tensor_names {
                    if let Some(t) = shard_map.get(name) {
                        weights.insert(name.clone(), t.shallow_clone());
                    }
                }
            }
        }
    }

    Ok(weights)
}
