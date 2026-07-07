use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use tch::{no_grad, Device, Kind, Reduction, Tensor};
use tracing::{info, warn};

use crate::lora::*;
use crate::model::*;
use crate::model::{rms_norm, glm5_mlp};
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

/// EP-distributed LoRA SFT training for GLM-5.2.
///
/// Each rank loads: all attention weights (replicated) + 1/world_size experts (sharded)
/// + shared experts + gate + embed + head + norm + LoRA adapter.
///
/// Forward: loop through ALL layers, DSA attention with LoRA (autograd),
/// MoE with EP (local experts → all-reduce → /world_size).
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
    info!(
        rank,
        world_size,
        local_rank,
        layers = runtime_config.num_hidden_layers,
        indexer_types = ?runtime_config.indexer_types,
        "GLM-5.2 LoRA SFT EP config loaded"
    );

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
    let trainable_layers: HashSet<usize> = lora_config_raw
        .target_layers
        .iter()
        .map(|l| *l as usize)
        .collect();
    let mut needed: HashSet<String> = HashSet::new();
    needed.insert("model.embed_tokens.weight".to_string());
    needed.insert("model.norm.weight".to_string());
    if !runtime_config.tie_word_embeddings {
        needed.insert("lm_head.weight".to_string());
    }

    for layer in 0..n_layers {
        // Skip non-trainable layers entirely — they won't be loaded to GPU
        if !trainable_layers.contains(&layer) {
            continue;
        }
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
                "q_b_proj.weight",
            ] {
                needed.insert(format!("{p}.self_attn.indexer.{suffix}"));
                // FP8 scale for indexer wk and wq_b
                if suffix == &"wk.weight" || suffix == &"q_b_proj.weight" {
                    needed.insert(format!("{p}.self_attn.indexer.{suffix}_scale_inv"));
                }
            }
        }
        // Gate + shared experts (all layers, replicated)
        needed.insert(format!("{p}.mlp.gate.weight"));
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
            needed.insert(format!("{p}.mlp.gate_proj.weight"));
            needed.insert(format!("{p}.mlp.up_proj.weight"));
            needed.insert(format!("{p}.mlp.down_proj.weight"));
        }
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
        if p.exists() { p.to_path_buf() }
        else { std::path::PathBuf::from("data/sft/deepseek_test.jsonl") }
    };
    let train_dataset = if sft_jsonl.exists() {
        info!(rank, path = %sft_jsonl.display(), "loading real SFT data");
        Glm5SftDataset::from_jsonl_simple(&sft_jsonl, &tokenizer)?
    } else {
        info!(rank, "no SFT JSONL found, using synthetic data");
        Glm5SftDataset::synthetic(&tokenizer)?
    };
    let raw_batch = train_dataset.padded_batch(0, 1, device);

    // Pad input to config seq_len
    let target_seq = config.model.seq_len as i64;
    let actual_seq = raw_batch.input_ids.size()[1];
    let train_batch = if actual_seq < target_seq {
        let pad_token = train_dataset.pad_token_id;
        let pad_ids = Tensor::full(
            [1, target_seq - actual_seq],
            pad_token,
            (Kind::Int64, device),
        );
        let input_ids = Tensor::cat(&[&raw_batch.input_ids, &pad_ids], 1);
        let pad_mask = Tensor::zeros([1, target_seq - actual_seq], (Kind::Int64, device));
        let target_mask = Tensor::cat(&[&raw_batch.target_mask, &pad_mask], 1);
        Glm5SftBatch {
            input_ids,
            target_mask,
            num_masked: raw_batch.num_masked,
        }
    } else {
        raw_batch
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

    // Pre-load indexer weights for all "full" TRAINABLE layers (for IndexShare)
    let mut indexer_weights_map: BTreeMap<usize, Glm5AttentionWeights> = BTreeMap::new();
    for layer in 0..n_layers {
        if !trainable_layers.contains(&layer) {
            continue;
        }
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

    // ── C++ kernel availability ──
    let use_cpp_attention = rustrain_deepseek_v4::fp8_kernel::is_glm5_attention_available();
    if use_cpp_attention {
        info!(rank, "C++ GLM5 attention kernel available — using coarse-grained C++ path");
    } else {
        info!(rank, "C++ GLM5 attention kernel not available — using Rust tch-rs path");
    }
    let use_cpp_mlp = use_cpp_attention; // same .so provides MLP
    let use_cpp_loss = use_cpp_attention; // same .so provides loss
    let use_cpp_optimizer = use_cpp_attention; // same .so provides optimizer

    // ── Pre-expand caching allocator ──
    // Tell PyTorch's caching allocator it can use 95% of GPU memory.
    // This causes it to pre-allocate large segments upfront instead of growing incrementally.
    rustrain_deepseek_v4::fp8_kernel::set_memory_fraction(0.95, local_rank as i32);
    info!(rank, "set caching allocator memory fraction to 0.95");

    // ── Cache expert weights on GPU (eliminates per-layer CPU→GPU transfer) ──
    // Expert weights are frozen (LoRA targets attention only), so they can be
    // loaded once and reused every step. This eliminates ~87GB/layer PCIe transfer.
    info!(rank, "caching expert weights on GPU (with FP8 pre-dequant)...");
    let mut expert_weights_gpu: BTreeMap<String, Tensor> = BTreeMap::new();
    // First pass: load all to GPU
    for (name, t) in &expert_weights_cpu {
        let gpu_t = t.to_device(device);
        expert_weights_gpu.insert(name.clone(), gpu_t);
    }
    // Second pass: pre-dequant FP8 weights to BF16 (saves per-step dequant in safe_linear)
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
                            tracing::warn!(rank, "pre-dequant failed for {}: {:?}, keeping FP8", name, e);
                        }
                    }
                }
            }
        }
    }
    let expert_gpu_count = expert_weights_gpu.len();
    info!(rank, expert_tensors_on_gpu = expert_gpu_count, "expert weights cached on GPU (FP8 pre-dequanted)");

    // ── Training loop ──
    for step in 0..config.train.max_steps {
        // ── Forward ──
        let embed = tensor(&weights_gpu, "model.embed_tokens.weight")?.to_kind(compute_kind);
        let mut hidden = Tensor::embedding(&embed, &train_batch.input_ids, -1, false, false);
        if hidden.kind() != compute_kind {
            hidden = hidden.to_kind(compute_kind);
        }

        let mut index_share_state: Option<IndexShareState> = None;
        // C++ IndexShare state for layer_forward path (separate from Rust path)
        let mut cpp_layer_state = rustrain_deepseek_v4::fp8_kernel::Glm5IndexState::default();

        // Async pipeline: (output_tensor, cuda_event) from previous layer's all-reduce.
        // Next layer waits on event (GPU-side, no CPU block) before using output.
        let mut pending_allreduce: Option<(Tensor, rustrain_nccl::nccl::CudaEventHandle)> = None;

        for layer in 0..n_layers {
            // Skip non-trainable layers — hidden passes through unchanged
            if !trainable_layers.contains(&layer) {
                continue;
            }
            let p = format!("model.layers.{layer}");

            if use_cpp_attention {
                // ── C++ unified layer forward: 1 FFI call for entire layer ──
                // Combines: RMSNorm → attention → residual → RMSNorm → MoE/dense → residual
                let attn_norm = tensor(&weights_gpu, &format!("{p}.input_layernorm.weight"))?.to_kind(compute_kind);
                let post_norm = tensor(&weights_gpu, &format!("{p}.post_attention_layernorm.weight"))?.to_kind(compute_kind);

                // Load attention weights (with LoRA applied)
                let attn_weights = Glm5AttentionWeights::load_with_kind(&weights_gpu, layer, compute_kind)?;
                let lora_attn = lora_attention_weights(&attn_weights, layer, &mut registry);

                // Get indexer weights
                let source = runtime_config.indexer_source_layer(layer);
                let indexer_weights = indexer_weights_map.get(&source).unwrap_or(&lora_attn);

                let is_full_layer = !runtime_config.should_skip_topk(layer)
                    && (cpp_layer_state.is_none() || layer % (runtime_config.index_topk_freq as usize) == 0);

                let is_moe = runtime_config.is_moe_layer(layer);

                // ── Wait for previous layer's async all-reduce to complete (GPU-side, no CPU block) ──
                if let Some((prev_output, prev_event)) = pending_allreduce.take() {
                    rustrain_deepseek_v4::fp8_kernel::stream_wait_event(local_rank as i32, &prev_event);
                    hidden = prev_output;
                }

                if is_moe {
                    // ── MoE layer ──
                    let gate = tensor(&weights_gpu, &format!("{p}.mlp.gate.weight"))?.to_kind(compute_kind);
                    let shared_gate = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.shared_experts.gate_proj.weight"))?, compute_kind);
                    let shared_up = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.shared_experts.up_proj.weight"))?, compute_kind);
                    let shared_down = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.shared_experts.down_proj.weight"))?, compute_kind);
                    let shared_gate_scale = weights_gpu.get(&format!("{p}.mlp.shared_experts.gate_proj.weight_scale_inv"));
                    let shared_up_scale = weights_gpu.get(&format!("{p}.mlp.shared_experts.up_proj.weight_scale_inv"));
                    let shared_down_scale = weights_gpu.get(&format!("{p}.mlp.shared_experts.down_proj.weight_scale_inv"));

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
                        egw.push(expert_weights_gpu.get(&format!("{eg}.gate_proj.weight")).unwrap());
                        euw.push(expert_weights_gpu.get(&format!("{eg}.up_proj.weight")).unwrap());
                        edw.push(expert_weights_gpu.get(&format!("{eg}.down_proj.weight")).unwrap());
                        egs.push(expert_weights_gpu.get(&format!("{eg}.gate_proj.weight_scale_inv")));
                        eus.push(expert_weights_gpu.get(&format!("{eg}.up_proj.weight_scale_inv")));
                        eds.push(expert_weights_gpu.get(&format!("{eg}.down_proj.weight_scale_inv")));
                    }

                    let partial_mlp = rustrain_deepseek_v4::fp8_kernel::glm5_layer_forward_cpp(
                        &hidden,
                        &attn_norm, &post_norm,
                        &lora_attn.q_a_proj, &lora_attn.q_a_layernorm, &lora_attn.q_b_proj,
                        &lora_attn.kv_a_proj_with_mqa, &lora_attn.kv_a_layernorm, &lora_attn.kv_b_proj,
                        &lora_attn.o_proj,
                        lora_attn.q_a_proj_scale.as_ref(), lora_attn.q_b_proj_scale.as_ref(),
                        lora_attn.kv_a_proj_scale.as_ref(), lora_attn.kv_b_proj_scale.as_ref(),
                        lora_attn.o_proj_scale.as_ref(),
                        lora_attn.indexer_wq_b.as_ref(), lora_attn.indexer_wk.as_ref(),
                        lora_attn.indexer_k_norm_weight.as_ref(), lora_attn.indexer_k_norm_bias.as_ref(),
                        lora_attn.indexer_weights_proj.as_ref(),
                        lora_attn.indexer_wq_b_scale.as_ref(), lora_attn.indexer_wk_scale.as_ref(),
                        Some(&gate),
                        Some(&shared_gate), Some(&shared_up), Some(&shared_down),
                        shared_gate_scale, shared_up_scale, shared_down_scale,
                        None, None, None, None, None, None, // dense weights (MoE)
                        &egw, &euw, &edw, &egs, &eus, &eds,
                        &ep_shard.local_expert_indices,
                        hidden.size()[0] as i32, hidden.size()[1] as i32,
                        runtime_config.num_attention_heads as i32,
                        runtime_config.qk_nope_head_dim as i32,
                        runtime_config.qk_rope_head_dim as i32,
                        runtime_config.v_head_dim as i32,
                        runtime_config.kv_lora_rank as i32,
                        runtime_config.index_head_dim as i32,
                        runtime_config.index_n_heads as i32,
                        runtime_config.index_topk as i32,
                        layer as i32, is_full_layer, true,
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
                        let reduced = nccl_comm.as_ref().unwrap().all_reduce(&partial_mlp)
                            .unwrap_or_else(|_| partial_mlp.shallow_clone());
                        let full = (&reduced / (world_size as f64)).to_kind(mlp_kind);
                        // identity trick: forward = full, backward grad → partial_mlp (coef 1)
                        hidden = &partial_mlp + &(&full - &partial_mlp).detach();
                    } else {
                        hidden = partial_mlp.shallow_clone();
                    }
                } else {
                    // ── Dense layer ──
                    let gate_w = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.gate_proj.weight"))?, compute_kind);
                    let up_w = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.up_proj.weight"))?, compute_kind);
                    let down_w = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.down_proj.weight"))?, compute_kind);
                    let gate_scale = weights_gpu.get(&format!("{p}.mlp.gate_proj.weight_scale_inv"));
                    let up_scale = weights_gpu.get(&format!("{p}.mlp.up_proj.weight_scale_inv"));
                    let down_scale = weights_gpu.get(&format!("{p}.mlp.down_proj.weight_scale_inv"));

                    hidden = rustrain_deepseek_v4::fp8_kernel::glm5_layer_forward_cpp(
                        &hidden,
                        &attn_norm, &post_norm,
                        &lora_attn.q_a_proj, &lora_attn.q_a_layernorm, &lora_attn.q_b_proj,
                        &lora_attn.kv_a_proj_with_mqa, &lora_attn.kv_a_layernorm, &lora_attn.kv_b_proj,
                        &lora_attn.o_proj,
                        lora_attn.q_a_proj_scale.as_ref(), lora_attn.q_b_proj_scale.as_ref(),
                        lora_attn.kv_a_proj_scale.as_ref(), lora_attn.kv_b_proj_scale.as_ref(),
                        lora_attn.o_proj_scale.as_ref(),
                        lora_attn.indexer_wq_b.as_ref(), lora_attn.indexer_wk.as_ref(),
                        lora_attn.indexer_k_norm_weight.as_ref(), lora_attn.indexer_k_norm_bias.as_ref(),
                        lora_attn.indexer_weights_proj.as_ref(),
                        lora_attn.indexer_wq_b_scale.as_ref(), lora_attn.indexer_wk_scale.as_ref(),
                        None, None, None, None, None, None, None, // MoE weights (none for dense)
                        Some(&gate_w), Some(&up_w), Some(&down_w),
                        gate_scale, up_scale, down_scale,
                        &[], &[], &[], &[], &[], &[], // expert weights (none)
                        &[],
                        hidden.size()[0] as i32, hidden.size()[1] as i32,
                        runtime_config.num_attention_heads as i32,
                        runtime_config.qk_nope_head_dim as i32,
                        runtime_config.qk_rope_head_dim as i32,
                        runtime_config.v_head_dim as i32,
                        runtime_config.kv_lora_rank as i32,
                        runtime_config.index_head_dim as i32,
                        runtime_config.index_n_heads as i32,
                        runtime_config.index_topk as i32,
                        layer as i32, is_full_layer, false,
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
                rustrain_deepseek_v4::fp8_kernel::glm5_rms_norm_cpp(&hidden, &attn_norm, runtime_config.rms_norm_eps)?
            } else {
                rms_norm(&hidden, &attn_norm, runtime_config.rms_norm_eps)
            };

            // Load attention weights
            let attn_weights = Glm5AttentionWeights::load_with_kind(&weights_gpu, layer, compute_kind)?;
            let lora_attn = lora_attention_weights(&attn_weights, layer, &mut registry);

            // Get indexer weights (from source layer for IndexShare)
            let source = runtime_config.indexer_source_layer(layer);
            let indexer_weights = indexer_weights_map.get(&source).unwrap_or(&lora_attn);

            // Determine if this is a "full" layer (computes indexer state) or "shared" (reuses)
            let is_full_layer = !runtime_config.should_skip_topk(layer)
                && (index_share_state.is_none() || layer % (runtime_config.index_topk_freq as usize) == 0);

            // ── C++ attention path: coarse-grained, one FFI call per layer ──
            // Rust path: fine-grained, ~30 tch-rs calls per layer (with checkpoint)
            let attn_out = if use_cpp_attention {
                // C++ path — no checkpointing needed (C++ manages intermediates on stack)
                // Convert Rust IndexShareState to C++ Glm5IndexState
                let mut cpp_state = rustrain_deepseek_v4::fp8_kernel::Glm5IndexState::default();
                if let Some(ref s) = index_share_state {
                    // Rust state → C++ state: pass topk_indices/idx_bias_keys as at::Tensor*
                    // For now, C++ recomputes — we pass null state and let C++ compute fresh
                    // TODO: share state between Rust and C++ (needs at::Tensor* ↔ tch::Tensor conversion)
                }
                let result = rustrain_deepseek_v4::fp8_kernel::glm5_dsa_attention_cpp(
                    &hidden_norm,
                    &lora_attn.q_a_proj, &lora_attn.q_a_layernorm, &lora_attn.q_b_proj,
                    &lora_attn.kv_a_proj_with_mqa, &lora_attn.kv_a_layernorm, &lora_attn.kv_b_proj,
                    &lora_attn.o_proj,
                    lora_attn.q_a_proj_scale.as_ref(), lora_attn.q_b_proj_scale.as_ref(),
                    lora_attn.kv_a_proj_scale.as_ref(), lora_attn.kv_b_proj_scale.as_ref(),
                    lora_attn.o_proj_scale.as_ref(),
                    lora_attn.indexer_wq_b.as_ref(), lora_attn.indexer_wk.as_ref(),
                    lora_attn.indexer_k_norm_weight.as_ref(), lora_attn.indexer_k_norm_bias.as_ref(),
                    lora_attn.indexer_weights_proj.as_ref(),
                    lora_attn.indexer_wq_b_scale.as_ref(), lora_attn.indexer_wk_scale.as_ref(),
                    hidden.size()[0] as i32, hidden.size()[1] as i32,
                    runtime_config.num_attention_heads as i32,
                    runtime_config.qk_nope_head_dim as i32,
                    runtime_config.qk_rope_head_dim as i32,
                    runtime_config.v_head_dim as i32,
                    runtime_config.kv_lora_rank as i32,
                    runtime_config.index_head_dim as i32,
                    runtime_config.index_n_heads as i32,
                    runtime_config.index_topk as i32,
                    layer as i32, is_full_layer,
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
            } else if use_checkpointing {
                // Rust path with checkpointing
                let state_mutex = Arc::new(Mutex::new(index_share_state.take()));
                let state_for_closure = state_mutex.clone();
                let attn_clone = lora_attn.clone();
                let indexer_clone = indexer_weights.clone();
                let runtime_clone = runtime_config.clone();
                let layer_copy = layer;
                let full_layer = is_full_layer;
                let result = rustrain_deepseek_v4::fp8_kernel::checkpoint(&hidden_norm, move |input| {
                    let mut guard = state_for_closure.lock().unwrap();
                    let mut local_state = guard.take();
                    if full_layer {
                        local_state = None;
                    }
                    let output = glm5_dsa_attention(input, &attn_clone, &indexer_clone, &runtime_clone, &mut local_state, layer_copy);
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
            let post_norm = tensor(&weights_gpu, &format!("{p}.post_attention_layernorm.weight"))?
                .to_kind(compute_kind);
            let mlp_input = if use_cpp_attention {
                rustrain_deepseek_v4::fp8_kernel::glm5_rms_norm_cpp(&residual, &post_norm, runtime_config.rms_norm_eps)?
            } else {
                rms_norm(&residual, &post_norm, runtime_config.rms_norm_eps)
            };

            if runtime_config.is_moe_layer(layer) {
                // MoE with EP
                let gate = tensor(&weights_gpu, &format!("{p}.mlp.gate.weight"))?.to_kind(compute_kind);
                let shared_gate = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.shared_experts.gate_proj.weight"))?, compute_kind);
                let shared_up = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.shared_experts.up_proj.weight"))?, compute_kind);
                let shared_down = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.shared_experts.down_proj.weight"))?, compute_kind);
                let shared_gate_scale = weights_gpu.get(&format!("{p}.mlp.shared_experts.gate_proj.weight_scale_inv"));
                let shared_up_scale = weights_gpu.get(&format!("{p}.mlp.shared_experts.up_proj.weight_scale_inv"));
                let shared_down_scale = weights_gpu.get(&format!("{p}.mlp.shared_experts.down_proj.weight_scale_inv"));

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
                        egw.push(expert_weights_gpu.get(&format!("{eg}.gate_proj.weight")).unwrap());
                        euw.push(expert_weights_gpu.get(&format!("{eg}.up_proj.weight")).unwrap());
                        edw.push(expert_weights_gpu.get(&format!("{eg}.down_proj.weight")).unwrap());
                        egs.push(expert_weights_gpu.get(&format!("{eg}.gate_proj.weight_scale_inv")));
                        eus.push(expert_weights_gpu.get(&format!("{eg}.up_proj.weight_scale_inv")));
                        eds.push(expert_weights_gpu.get(&format!("{eg}.down_proj.weight_scale_inv")));
                    }
                    let partial_mlp = rustrain_deepseek_v4::fp8_kernel::glm5_moe_layer_cpp(
                        &mlp_input,
                        &shared_gate, &shared_up, &shared_down,
                        shared_gate_scale, shared_up_scale, shared_down_scale,
                        &gate,
                        &egw, &euw, &edw,
                        &egs, &eus, &eds,
                        &ep_shard.local_expert_indices,
                        runtime_config.n_routed_experts as i32,
                        runtime_config.num_experts_per_tok as i32,
                        runtime_config.routed_scaling_factor,
                        local_rank as i32,
                    )?;

                    // All-reduce MoE output (shared expert counted world_size times → divide)
                    let mlp_kind = partial_mlp.kind();
                    let full_mlp = if world_size > 1 {
                        let pd = no_grad(|| partial_mlp.shallow_clone()).detach();
                        let reduced = nccl_comm.as_ref().unwrap().all_reduce(&pd)?;
                        let full = no_grad(|| (&reduced / (world_size as f64)).to_kind(mlp_kind)).detach();
                        full.set_requires_grad(true)
                    } else {
                        partial_mlp.shallow_clone()
                    };
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
                        glm5_mlp_fp8(input, &sg, &su, &sd, sgs.as_ref(), sus.as_ref(), sds.as_ref())
                    })
                } else {
                    glm5_mlp_fp8(
                        &mlp_input, &shared_gate, &shared_up, &shared_down,
                        shared_gate_scale, shared_up_scale, shared_down_scale,
                    )
                };

                // Router logits — computed over ALL experts
                let router_logits = mlp_input.linear::<&Tensor>(&gate, None);
                let n_experts = runtime_config.n_routed_experts as i64;
                let k = runtime_config.num_experts_per_tok as i64;

                // Sigmoid scoring + top-k
                let scores = router_logits.sigmoid();
                let (topk_weights, topk_indices) = scores.topk(k, -1, true, true);
                // Normalize
                let denom = topk_weights.sum_dim_intlist([-1].as_slice(), true, topk_weights.kind());
                let topk_weights = (topk_weights / denom) * runtime_config.routed_scaling_factor;

                // Flatten for per-token dispatch
                let flat_input = mlp_input.reshape([-1, mlp_input.size()[2]]);
                let tk_indices = topk_indices.reshape([-1, k]);
                let tk_weights = topk_weights.reshape([-1, k]);

                // Only apply LOCAL experts (EP sharded)
                let mut partial_output = Tensor::zeros(
                    flat_input.size(),
                    (compute_kind, flat_input.device()),
                );

                for (local_idx, &global_e) in ep_shard.local_expert_indices.iter().enumerate() {
                    // Check which tokens selected this expert
                    let mask = tk_indices.eq(global_e as i64).to_kind(compute_kind);
                    let mask_flat = mask.sum_dim_intlist([-1].as_slice(), false, compute_kind).to_kind(compute_kind);
                    let count = mask_flat.sum(compute_kind).double_value(&[]) as i64;
                    if count == 0 {
                        continue;
                    }
                    let eg = format!("{p}.mlp.experts.{global_e}");
                    // Expert weights from GPU cache (no CPU→GPU transfer)
                    let gate_w = expert_weights_gpu.get(&format!("{eg}.gate_proj.weight"))
                        .with_context(|| format!("expert weight not found: {eg}.gate_proj.weight"))?;
                    let up_w = expert_weights_gpu.get(&format!("{eg}.up_proj.weight"))
                        .with_context(|| format!("expert weight not found: {eg}.up_proj.weight"))?;
                    let down_w = expert_weights_gpu.get(&format!("{eg}.down_proj.weight"))
                        .with_context(|| format!("expert weight not found: {eg}.down_proj.weight"))?;
                    let gate_w_scale = expert_weights_gpu.get(&format!("{eg}.gate_proj.weight_scale_inv"));
                    let up_w_scale = expert_weights_gpu.get(&format!("{eg}.up_proj.weight_scale_inv"));
                    let down_w_scale = expert_weights_gpu.get(&format!("{eg}.down_proj.weight_scale_inv"));
                    let expert_out = if use_cpp_mlp {
                        rustrain_deepseek_v4::fp8_kernel::glm5_mlp_fp8_cpp(
                            &flat_input, gate_w, up_w, down_w,
                            gate_w_scale, up_w_scale, down_w_scale,
                        )?
                    } else {
                        glm5_mlp_fp8(
                            &flat_input, gate_w, up_w, down_w,
                            gate_w_scale, up_w_scale, down_w_scale,
                        )
                    };
                    // Find which position (0..k) this expert was selected at, and use that weight
                    // For simplicity, sum contributions from all k positions
                    let weight = tk_weights.narrow(-1, 0, 1).unsqueeze(-1); // simplified: use first position
                    // Actually need to find the weight corresponding to where global_e appears in topk_indices
                    // Use mask & tk_weights summed over k
                    let weighted_mask = (mask * &tk_weights).sum_dim_intlist([-1].as_slice(), false, compute_kind).to_kind(compute_kind);
                    let mask_expanded = weighted_mask.unsqueeze(-1).expand([-1, expert_out.size()[1]], false);
                    let contribution = expert_out * &mask_expanded;
                    partial_output = partial_output + contribution;
                }

                // partial_output = local experts only. Add shared (replicated).
                let partial_mlp = partial_output.reshape([1, -1, mlp_input.size()[2]]) + &shared_output;

                // All-reduce MoE output (shared expert counted world_size times → divide)
                let mlp_kind = partial_mlp.kind();
                let full_mlp = if world_size > 1 {
                    let pd = no_grad(|| partial_mlp.shallow_clone()).detach();
                    let reduced = nccl_comm.as_ref().unwrap().all_reduce(&pd)?;
                    let full = no_grad(|| (&reduced / (world_size as f64)).to_kind(mlp_kind)).detach();
                    full.set_requires_grad(true)
                } else {
                    partial_mlp.shallow_clone()
                };

                hidden = &residual + &full_mlp;
                } // end Rust MoE fallback
            } else {
                // Dense MLP — checkpointed to save intermediate activations
                let gate = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.gate_proj.weight"))?, compute_kind);
                let up = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.up_proj.weight"))?, compute_kind);
                let down = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.down_proj.weight"))?, compute_kind);
                let gate_scale = weights_gpu.get(&format!("{p}.mlp.gate_proj.weight_scale_inv")).map(|t| t.shallow_clone());
                let up_scale = weights_gpu.get(&format!("{p}.mlp.up_proj.weight_scale_inv")).map(|t| t.shallow_clone());
                let down_scale = weights_gpu.get(&format!("{p}.mlp.down_proj.weight_scale_inv")).map(|t| t.shallow_clone());

                let mlp = if use_cpp_mlp {
                    rustrain_deepseek_v4::fp8_kernel::glm5_mlp_fp8_cpp(
                        &mlp_input, &gate, &up, &down,
                        gate_scale.as_ref(), up_scale.as_ref(), down_scale.as_ref(),
                    )?
                } else if use_checkpointing {
                    rustrain_deepseek_v4::fp8_kernel::checkpoint(&mlp_input, move |input| {
                        glm5_mlp_fp8(input, &gate, &up, &down, gate_scale.as_ref(), up_scale.as_ref(), down_scale.as_ref())
                    })
                } else {
                    glm5_mlp_fp8(&mlp_input, &gate, &up, &down, gate_scale.as_ref(), up_scale.as_ref(), down_scale.as_ref())
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
        let final_norm = tensor(&weights_gpu, "model.norm.weight")?.to_kind(compute_kind);
        let normed = if use_cpp_attention {
            rustrain_deepseek_v4::fp8_kernel::glm5_rms_norm_cpp(&hidden, &final_norm, runtime_config.rms_norm_eps)?
        } else {
            rms_norm(&hidden, &final_norm, runtime_config.rms_norm_eps)
        };
        let lm_head = if runtime_config.tie_word_embeddings {
            embed.shallow_clone()
        } else {
            tensor(&weights_gpu, "lm_head.weight")?.to_kind(compute_kind)
        };

        // ── Chunked SFT Loss ──
        let seq_len = train_batch.input_ids.size()[1];
        let vocab = runtime_config.vocab_size;

        let loss = if use_cpp_loss {
            // C++ cross-entropy loss (single call, chunked internally)
            rustrain_deepseek_v4::fp8_kernel::glm5_cross_entropy_loss_cpp(
                &normed, &lm_head, &train_batch.input_ids, &train_batch.target_mask,
                seq_len as i32, vocab as i32, 256, local_rank as i32,
            )?
        } else {
            // Rust chunked cross-entropy loss
            let shifted_targets = train_batch.input_ids.narrow(1, 1, seq_len - 1);
            let shifted_mask = train_batch.target_mask.narrow(1, 1, seq_len - 1).to_kind(Kind::Float);
            let total_mask = shifted_mask.sum(Kind::Float);
            let ce_chunk_size = 256;
            let mut loss_acc = Tensor::zeros([], (Kind::Float, device));
            for start in (0..seq_len - 1).step_by(ce_chunk_size as usize) {
                let end = (start + ce_chunk_size as i64).min(seq_len - 1);
                let chunk_len = end - start;
                let normed_chunk = normed.narrow(1, start, chunk_len);
                let logits_chunk = normed_chunk.linear::<&Tensor>(&lm_head, None);
                let log_probs = logits_chunk.reshape([-1, vocab]).log_softmax(-1, Kind::Float);
                let targets_chunk = shifted_targets.narrow(1, start, chunk_len).reshape([-1]);
                let mask_chunk = shifted_mask.narrow(1, start, chunk_len);
                let per_token_loss = log_probs
                    .g_nll_loss::<&Tensor>(&targets_chunk, None, Reduction::None, -100)
                    .reshape([1, chunk_len]);
                let masked = &per_token_loss * &mask_chunk;
                loss_acc = loss_acc + masked.sum(Kind::Float);
            }
            loss_acc / total_mask.clamp_min(1.0)
        };

        let loss_val = loss.double_value(&[]);
        if step == 0 {
            initial_loss = loss_val;
        }

        info!(
            rank,
            step = step + 1,
            loss = loss_val,
            "GLM-5 EP LoRA SFT train step"
        );

        // ── Backward ──
        loss.backward();

        // Free checkpoint closures (they hold GPU tensor references)
        rustrain_deepseek_v4::fp8_kernel::clear_checkpoint_registry();

        // ── Warmup: empty cache after first step ──
        // Empty cache every step to release intermediate tensors and prevent fragmentation.
        // (Previously only done on step 0 — this caused memory pressure on later steps.)
        rustrain_deepseek_v4::fp8_kernel::empty_cache();

        // ── LoRA gradient all-reduce ──
        // Note: all_reduce_async was attempted but caused undefined tensor issues
        // because the output tensor on comm_stream isn't visible to compute stream
        // without proper synchronization. Using sync all_reduce for correctness.
        // The MoE output all-reduce (line 547) uses async because it's on the
        // critical path of the layer loop; gradient sync is off the critical path.
        let synced_grads: Vec<Tensor> = if world_size > 1 {
            let vars = registry.var_store.trainable_variables();
            vars.iter()
                .map(|var| {
                    let g = var.grad();
                    if g.defined() && g.numel() > 0 {
                        let reduced = nccl_comm
                            .as_ref()
                            .unwrap()
                            .all_reduce(&g)
                            .unwrap_or_else(|_| g.shallow_clone());
                        no_grad(|| (&reduced / (world_size as f64)).to_kind(g.kind()))
                    } else {
                        g.shallow_clone()
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        // ── Adam optimizer step ──
        let mut current_vars = registry.var_store.trainable_variables();
        // Disable requires_grad during optimizer step (C++ uses in-place ops which
        // fail on leaf Variables with requires_grad=true)
        for v in &current_vars { v.set_requires_grad(false); }
        if use_cpp_optimizer {
            let grads: Vec<Tensor> = current_vars.iter().enumerate().map(|(i, _var)| {
                if world_size > 1 {
                    synced_grads[i].shallow_clone()
                } else {
                    current_vars[i].grad()
                }
            }).collect();
            rustrain_deepseek_v4::fp8_kernel::adam_step_cpp(
                &mut current_vars, &grads, &mut adam_m, &mut adam_v,
                lr, beta1, beta2, eps, step as i32,
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
        for v in &current_vars { v.set_requires_grad(true); }
        // Invalidate LoRA delta cache (params changed)
        registry.invalidate_delta_cache();
    }

    // ── Save LoRA adapter ──
    let adapter_output = run_paths.checkpoints.join("glm5-lora-adapter-ep.safetensors");
    registry.save(&adapter_output)?;
    info!(rank, adapter = %adapter_output.display(), "adapter saved");

    let final_loss = initial_loss; // TODO: proper final loss eval
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
            &single,
            needed,
            -1, // CPU
        )
        .with_context(|| format!("failed to load {}", single.display()));
    }

    if !index_path.exists() {
        anyhow::bail!("no model.safetensors or index file in {}", model_path.display());
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
        total_shards = index.weight_map.values().collect::<std::collections::HashSet<_>>().len(),
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
