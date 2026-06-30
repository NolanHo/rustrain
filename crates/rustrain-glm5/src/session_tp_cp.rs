//! TP+CP+EP training loop for GLM-5.2.
//!
//! This is a focused implementation that reuses the weight loading, barrier,
//! NCCL, LoRA, and optimizer infrastructure from session_ep.rs, but replaces
//! the attention path with TP+CP sharding.
//!
//! Rank decomposition: world_size = tp_size × cp_size × ep_size
//!   tp_rank = rank % tp_size
//!   cp_rank = (rank / tp_size) % cp_size
//!   ep_rank = rank / (tp_size * cp_size)

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use tch::{no_grad, Device, Kind, Reduction, Tensor};
use tracing::info;

use crate::lora::*;
use crate::model::*;
use crate::model::{rms_norm, glm5_mlp};
use crate::sft::*;
use crate::tp_cp::*;
use crate::session_ep::Glm5EpShard;
use rustrain_checkpoint::safetensors::tensor;
use rustrain_nccl::nccl::{self as nccl_smoke, NcclPersistentComm};

fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

fn keep_fp8(t: &Tensor, kind: Kind) -> Tensor {
    if t.kind() == Kind::Float8e4m3fn {
        t.shallow_clone()
    } else {
        t.to_kind(kind)
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
    let tp_size = config.parallel.tensor_model_parallel_size.max(1);
    let cp_size = config.parallel.context_parallel_size.max(1);
    let ep_size = (world_size / (tp_size * cp_size)).max(1);

    if world_size != tp_size * cp_size * ep_size {
        bail!(
            "world_size {world_size} must equal tp_size {tp_size} × cp_size {cp_size} × ep_size {ep_size}"
        );
    }

    let tp_rank = rank % tp_size;
    let cp_rank = (rank / tp_size) % cp_size;
    let ep_rank = rank / (tp_size * cp_size);

    // ── Model config ──
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("GLM-5 TP+CP+EP requires model.model_path")?;
    let model_path = std::path::PathBuf::from(model_path);
    let runtime_config = read_glm5_config(&model_path.join("config.json"))?;

    // Validate
    if runtime_config.num_attention_heads as usize % tp_size != 0 {
        bail!(
            "num_attention_heads {} must be divisible by tp_size {tp_size}",
            runtime_config.num_attention_heads
        );
    }
    if runtime_config.n_routed_experts % ep_size != 0 {
        bail!(
            "n_routed_experts {} must be divisible by ep_size {ep_size}",
            runtime_config.n_routed_experts
        );
    }

    info!(
        rank, world_size, local_rank, tp_rank, cp_rank, ep_rank, tp_size, cp_size, ep_size,
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
    let trainable_layers: HashSet<usize> = trainable_layer_indices.iter().copied().collect();
    let mut needed: HashSet<String> = HashSet::new();
    needed.insert("model.embed_tokens.weight".to_string());
    needed.insert("model.norm.weight".to_string());
    if !runtime_config.tie_word_embeddings {
        needed.insert("lm_head.weight".to_string());
    }

    for layer in 0..n_layers {
        if !trainable_layers.contains(&layer) {
            continue;
        }
        let p = format!("model.layers.{layer}");
        needed.insert(format!("{p}.input_layernorm.weight"));
        needed.insert(format!("{p}.post_attention_layernorm.weight"));
        for suffix in &[
            "q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
            "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight",
            "o_proj.weight",
        ] {
            needed.insert(format!("{p}.self_attn.{suffix}"));
            needed.insert(format!("{p}.self_attn.{suffix}_scale_inv"));
        }
        let indexer_type = runtime_config.indexer_types.get(layer).map(|s| s.as_str()).unwrap_or("full");
        if indexer_type == "full" {
            for suffix in &["k_norm.weight", "k_norm.bias", "weights_proj.weight", "wk.weight", "q_b_proj.weight"] {
                needed.insert(format!("{p}.self_attn.indexer.{suffix}"));
                if suffix == &"wk.weight" || suffix == &"q_b_proj.weight" {
                    needed.insert(format!("{p}.self_attn.indexer.{suffix}_scale_inv"));
                }
            }
        }
        needed.insert(format!("{p}.mlp.gate.weight"));
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
            needed.insert(format!("{p}.mlp.gate_proj.weight"));
            needed.insert(format!("{p}.mlp.up_proj.weight"));
            needed.insert(format!("{p}.mlp.down_proj.weight"));
        }
    }

    // ── Load weights ──
    let weights = load_glm5_weights_shared(&model_path, &needed)?;
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

    // ── LoRA registry ──
    // For TP, LoRA is on the local shard of attention weights.
    // We load full attention weights, create LoRA on them, then narrow.
    // Simpler: create LoRA registry on full weights, then narrow in forward.
    let registry = Glm5LoraRegistry::new(&weights_gpu, lora_config, device)?;
    let trainable_count = registry.var_store.trainable_variables().len();
    info!(rank, trainable_params = trainable_count, "LoRA adapters created");

    // ── Barrier ──
    let barrier_dir = config.run.base_dir.join(&config.run.name).join("barrier");
    std::fs::create_dir_all(&barrier_dir)?;
    let ready_file = barrier_dir.join(format!("rank_{rank}.ready"));
    std::fs::write(&ready_file, b"ready")?;
    info!(rank, "waiting at barrier");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
    loop {
        let ready_count = std::fs::read_dir(&barrier_dir)
            .map(|d| d.filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().starts_with("rank_")).count())
            .unwrap_or(0);
        if ready_count >= world_size { break; }
        if std::time::Instant::now() > deadline { bail!("barrier timeout: {ready_count}/{world_size}"); }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    info!(rank, "all ranks ready");

    // ── NCCL ──
    let nccl_comm = if world_size > 1 {
        let comm_dir = config.run.base_dir.join(&config.run.name).join("nccl-comm");
        let comm = NcclPersistentComm::new(&comm_dir)?;
        info!(rank, "NCCL communicator created");
        Some(comm)
    } else {
        None
    };

    // ── SFT data ──
    let tokenizer = tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let sft_jsonl = std::path::Path::new("data/sft/deepseek_test.jsonl");
    let train_dataset = if sft_jsonl.exists() {
        Glm5SftDataset::from_jsonl_simple(sft_jsonl, &tokenizer)?
    } else {
        Glm5SftDataset::synthetic(&tokenizer)?
    };
    let raw_batch = train_dataset.padded_batch(0, 1, device);

    // Pad to config seq_len, then slice for CP
    let target_seq = config.model.seq_len as i64;
    let actual_seq = raw_batch.input_ids.size()[1];
    let full_batch = if actual_seq < target_seq {
        let pad_token = train_dataset.pad_token_id;
        let pad_ids = Tensor::full([1, target_seq - actual_seq], pad_token, (Kind::Int64, device));
        let input_ids = Tensor::cat(&[&raw_batch.input_ids, &pad_ids], 1);
        let pad_mask = Tensor::zeros([1, target_seq - actual_seq], (Kind::Int64, device));
        let target_mask = Tensor::cat(&[&raw_batch.target_mask, &pad_mask], 1);
        Glm5SftBatch { input_ids, target_mask, num_masked: raw_batch.num_masked }
    } else {
        raw_batch
    };

    // CP slice: each rank handles [cp_rank * s_local, (cp_rank+1) * s_local)
    let s_local = target_seq / cp_size as i64;
    let cp_batch = if cp_size > 1 {
        let input_ids = full_batch.input_ids.narrow(1, cp_rank as i64 * s_local, s_local);
        let target_mask = full_batch.target_mask.narrow(1, cp_rank as i64 * s_local, s_local);
        Glm5SftBatch {
            input_ids,
            target_mask,
            num_masked: full_batch.num_masked,
        }
    } else {
        full_batch
    };

    // ── Optimizer ──
    let lr = config.train.learning_rate as f64;
    let beta1 = config.train.adam_beta1 as f64;
    let beta2 = config.train.adam_beta2 as f64;
    let eps = config.train.adam_eps as f64;
    let trainable_vars = registry.var_store.trainable_variables();
    let mut adam_m: Vec<Tensor> = trainable_vars.iter().map(Tensor::zeros_like).collect();
    let mut adam_v: Vec<Tensor> = trainable_vars.iter().map(Tensor::zeros_like).collect();

    let mut initial_loss = 0.0_f64;

    // Pre-load TP-sharded indexer weights for "full" layers
    let mut indexer_weights_map: BTreeMap<usize, Glm5TpAttentionWeights> = BTreeMap::new();
    for layer in 0..n_layers {
        if !trainable_layers.contains(&layer) { continue; }
        let indexer_type = runtime_config.indexer_types.get(layer).map(|s| s.as_str()).unwrap_or("full");
        if indexer_type == "full" {
            let attn = Glm5TpAttentionWeights::load_sharded(
                &weights_gpu, layer, compute_kind, &tp_shard, &runtime_config,
            )?;
            indexer_weights_map.insert(layer, attn);
        }
    }

    let use_checkpointing = true;
    rustrain_deepseek_v4::fp8_kernel::set_memory_fraction(0.95, local_rank as i32);
    info!(rank, "caching allocator set to 0.95");

    // ── Training loop ──
    for step in 0..config.train.max_steps {
        let embed = tensor(&weights_gpu, "model.embed_tokens.weight")?.to_kind(compute_kind);
        let mut hidden = Tensor::embedding(&embed, &cp_batch.input_ids, -1, false, false);
        if hidden.kind() != compute_kind {
            hidden = hidden.to_kind(compute_kind);
        }

        let mut index_share_state: Option<IndexShareState> = None;

        for layer in 0..n_layers {
            if !trainable_layers.contains(&layer) { continue; }
            let p = format!("model.layers.{layer}");

            // ── Attention (TP+CP sharded) ──
            let attn_norm = tensor(&weights_gpu, &format!("{p}.input_layernorm.weight"))?.to_kind(compute_kind);
            let hidden_norm = rms_norm(&hidden, &attn_norm, runtime_config.rms_norm_eps);

            // Load TP-sharded attention weights
            let attn_weights = Glm5TpAttentionWeights::load_sharded(
                &weights_gpu, layer, compute_kind, &tp_shard, &runtime_config,
            )?;
            // TODO: apply LoRA to TP-sharded weights (for now, no LoRA in TP path)

            let source = runtime_config.indexer_source_layer(layer);
            let indexer_weights = indexer_weights_map.get(&source).unwrap_or(&attn_weights);

            let is_full_layer = !runtime_config.should_skip_topk(layer)
                && (index_share_state.is_none() || layer % (runtime_config.index_topk_freq as usize) == 0);

            let attn_out = if use_checkpointing {
                let state_mutex = Arc::new(Mutex::new(index_share_state.take()));
                let state_for_closure = state_mutex.clone();
                let attn_clone = attn_weights.clone();
                let indexer_clone = indexer_weights.clone();
                let runtime_clone = runtime_config.clone();
                let tp_clone = Glm5TpShard::new(tp_rank, tp_size, runtime_config.num_attention_heads, runtime_config.index_n_heads);
                let layer_copy = layer;
                let cp_rank_copy = cp_rank;
                let cp_size_copy = cp_size;
                // For CP=1, pass None (no ring needed). For CP>1, skip checkpointing.
                if cp_size > 1 {
                    // CP>1: no checkpointing, direct call with comm
                    glm5_dsa_attention_tp_cp(
                        &hidden_norm, &attn_weights, indexer_weights, &runtime_config,
                        &mut index_share_state, layer, &tp_shard,
                        cp_rank, cp_size, ep_rank, nccl_comm.as_ref(),
                    )
                } else {
                    // CP=1: checkpoint (no comm needed)
                    let result = rustrain_deepseek_v4::fp8_kernel::checkpoint(&hidden_norm, move |input| {
                        let mut guard = state_for_closure.lock().unwrap();
                        let mut local_state = guard.take();
                        if is_full_layer {
                            local_state = None;
                        }
                        let output = glm5_dsa_attention_tp_cp(
                            input, &attn_clone, &indexer_clone, &runtime_clone,
                            &mut local_state, layer_copy, &tp_clone,
                            cp_rank_copy, 1, 0, None,
                        );
                        *guard = local_state;
                        output
                    });
                    index_share_state = state_mutex.lock().unwrap().take();
                    result
                }
            } else {
                glm5_dsa_attention_tp_cp(
                    &hidden_norm, &attn_weights, indexer_weights, &runtime_config,
                    &mut index_share_state, layer, &tp_shard,
                    cp_rank, cp_size, ep_rank, nccl_comm.as_ref(),
                )
            }
            .to_kind(compute_kind);

            // TP all-reduce: attention output is partial (only local heads).
            // All-reduce across ALL ranks, divide by ep_size.
            // (TP ranks in same EP position produce identical partials → sum = tp_size × partial)
            let attn_out = if tp_size > 1 {
                let pd = no_grad(|| attn_out.shallow_clone()).detach();
                let reduced = nccl_comm.as_ref().unwrap().all_reduce(&pd)?;
                let full = no_grad(|| (&reduced / (ep_size as f64)).to_kind(compute_kind)).detach();
                full.set_requires_grad(true)
            } else {
                attn_out
            };

            let residual = &hidden + &attn_out;

            // ── MoE / Dense MLP ── (same as EP, divide by tp_size)
            let post_norm = tensor(&weights_gpu, &format!("{p}.post_attention_layernorm.weight"))?.to_kind(compute_kind);
            let mlp_input = rms_norm(&residual, &post_norm, runtime_config.rms_norm_eps);

            if runtime_config.is_moe_layer(layer) {
                let gate = tensor(&weights_gpu, &format!("{p}.mlp.gate.weight"))?.to_kind(compute_kind);
                let shared_gate = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.shared_experts.gate_proj.weight"))?, compute_kind);
                let shared_up = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.shared_experts.up_proj.weight"))?, compute_kind);
                let shared_down = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.shared_experts.down_proj.weight"))?, compute_kind);
                let shared_gate_scale = weights_gpu.get(&format!("{p}.mlp.shared_experts.gate_proj.weight_scale_inv"));
                let shared_up_scale = weights_gpu.get(&format!("{p}.mlp.shared_experts.up_proj.weight_scale_inv"));
                let shared_down_scale = weights_gpu.get(&format!("{p}.mlp.shared_experts.down_proj.weight_scale_inv"));

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
                    glm5_mlp_fp8(&mlp_input, &shared_gate, &shared_up, &shared_down,
                        shared_gate_scale, shared_up_scale, shared_down_scale)
                };

                let router_logits = mlp_input.linear::<&Tensor>(&gate, None);
                let n_experts = runtime_config.n_routed_experts as i64;
                let k = runtime_config.num_experts_per_tok as i64;
                let scores = router_logits.sigmoid();
                let (topk_weights, topk_indices) = scores.topk(k, -1, true, true);
                let denom = topk_weights.sum_dim_intlist([-1].as_slice(), true, topk_weights.kind());
                let topk_weights = (topk_weights / denom) * runtime_config.routed_scaling_factor;

                let flat_input = mlp_input.reshape([-1, mlp_input.size()[2]]);
                let tk_indices = topk_indices.reshape([-1, k]);
                let tk_weights = topk_weights.reshape([-1, k]);

                let mut partial_output = Tensor::zeros(flat_input.size(), (compute_kind, flat_input.device()));

                for &global_e in &ep_shard.local_expert_indices {
                    let mask = tk_indices.eq(global_e as i64).to_kind(compute_kind);
                    let mask_flat = mask.sum_dim_intlist([-1].as_slice(), false, compute_kind).to_kind(compute_kind);
                    let count = mask_flat.sum(compute_kind).double_value(&[]) as i64;
                    if count == 0 { continue; }
                    let eg = format!("{p}.mlp.experts.{global_e}");
                    let gate_w_cpu = expert_weights_cpu.get(&format!("{eg}.gate_proj.weight")).context("expert weight")?;
                    let gate_w = keep_fp8(&gate_w_cpu.to_device(device), compute_kind);
                    let up_w_cpu = expert_weights_cpu.get(&format!("{eg}.up_proj.weight")).context("expert weight")?;
                    let up_w = keep_fp8(&up_w_cpu.to_device(device), compute_kind);
                    let down_w_cpu = expert_weights_cpu.get(&format!("{eg}.down_proj.weight")).context("expert weight")?;
                    let down_w = keep_fp8(&down_w_cpu.to_device(device), compute_kind);
                    let gate_w_scale = expert_weights_cpu.get(&format!("{eg}.gate_proj.weight_scale_inv")).map(|t| t.to_device(device));
                    let up_w_scale = expert_weights_cpu.get(&format!("{eg}.up_proj.weight_scale_inv")).map(|t| t.to_device(device));
                    let down_w_scale = expert_weights_cpu.get(&format!("{eg}.down_proj.weight_scale_inv")).map(|t| t.to_device(device));

                    let expert_out = glm5_mlp_fp8(&flat_input, &gate_w, &up_w, &down_w,
                        gate_w_scale.as_ref(), up_w_scale.as_ref(), down_w_scale.as_ref());
                    let weighted_mask = (mask * &tk_weights).sum_dim_intlist([-1].as_slice(), false, compute_kind).to_kind(compute_kind);
                    let mask_expanded = weighted_mask.unsqueeze(-1).expand([-1, expert_out.size()[1]], false);
                    partial_output = partial_output + &(expert_out * &mask_expanded);
                }

                let partial_mlp = partial_output.reshape([1, -1, mlp_input.size()[2]]) + &shared_output;
                // MoE all-reduce: divide by tp_size (EP ranks in same TP position produce identical partials)
                let mlp_kind = partial_mlp.kind();
                let full_mlp = if world_size > 1 {
                    let pd = no_grad(|| partial_mlp.shallow_clone()).detach();
                    let reduced = nccl_comm.as_ref().unwrap().all_reduce(&pd)?;
                    let full = no_grad(|| (&reduced / (tp_size as f64)).to_kind(mlp_kind)).detach();
                    full.set_requires_grad(true)
                } else {
                    partial_mlp.shallow_clone()
                };
                hidden = &residual + &full_mlp;
            } else {
                let gate = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.gate_proj.weight"))?, compute_kind);
                let up = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.up_proj.weight"))?, compute_kind);
                let down = keep_fp8(tensor(&weights_gpu, &format!("{p}.mlp.down_proj.weight"))?, compute_kind);
                let gate_scale = weights_gpu.get(&format!("{p}.mlp.gate_proj.weight_scale_inv")).map(|t| t.shallow_clone());
                let up_scale = weights_gpu.get(&format!("{p}.mlp.up_proj.weight_scale_inv")).map(|t| t.shallow_clone());
                let down_scale = weights_gpu.get(&format!("{p}.mlp.down_proj.weight_scale_inv")).map(|t| t.shallow_clone());

                let mlp = if use_checkpointing {
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

        // ── Final norm + lm_head + chunked CE loss ──
        let final_norm = tensor(&weights_gpu, "model.norm.weight")?.to_kind(compute_kind);
        let normed = rms_norm(&hidden, &final_norm, runtime_config.rms_norm_eps);
        let lm_head = if runtime_config.tie_word_embeddings {
            embed.shallow_clone()
        } else {
            tensor(&weights_gpu, "lm_head.weight")?.to_kind(compute_kind)
        };

        let seq_len_local = cp_batch.input_ids.size()[1];
        let vocab = runtime_config.vocab_size;
        let shifted_targets = cp_batch.input_ids.narrow(1, 1, seq_len_local - 1);
        let shifted_mask = cp_batch.target_mask.narrow(1, 1, seq_len_local - 1).to_kind(Kind::Float);
        let total_mask = shifted_mask.sum(Kind::Float);

        let ce_chunk_size = 256;
        let mut loss_acc = Tensor::zeros([], (Kind::Float, device));

        for start in (0..seq_len_local - 1).step_by(ce_chunk_size as usize) {
            let end = (start + ce_chunk_size as i64).min(seq_len_local - 1);
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

        // CP all-reduce: sum loss across CP ranks, divide by cp_size
        let loss = if cp_size > 1 {
            let pd = no_grad(|| loss_acc.shallow_clone()).detach();
            let reduced = nccl_comm.as_ref().unwrap().all_reduce(&pd)?;
            no_grad(|| (reduced / (cp_size as f64)).to_kind(Kind::Float))
        } else {
            loss_acc
        };
        let loss = loss / total_mask.clamp_min(1.0);

        let loss_val = loss.double_value(&[]);
        if step == 0 {
            initial_loss = loss_val;
        }

        info!(rank, step = step + 1, loss = loss_val, "GLM-5 TP+CP+EP train step");

        // ── Backward ──
        loss.backward();
        rustrain_deepseek_v4::fp8_kernel::clear_checkpoint_registry();

        if step == 0 {
            rustrain_deepseek_v4::fp8_kernel::empty_cache();
            info!(rank, "cache warmed up");
        }

        // ── LoRA gradient all-reduce ── (divide by world_size)
        let synced_grads: Vec<Tensor> = if world_size > 1 {
            let vars = registry.var_store.trainable_variables();
            vars.iter()
                .map(|var| {
                    let g = var.grad();
                    if g.defined() && g.numel() > 0 {
                        let reduced = nccl_comm.as_ref().unwrap().all_reduce(&g)
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
    }

    // ── Save LoRA adapter ──
    let adapter_output = run_paths.checkpoints.join("glm5-lora-adapter-tp-cp-ep.safetensors");
    registry.save(&adapter_output)?;
    info!(rank, adapter = %adapter_output.display(), "adapter saved");

    let final_loss = initial_loss;
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
            shard_to_tensors.entry(shard.clone()).or_default().push(name.clone());
        }
    }

    let mut weights = BTreeMap::new();
    for (shard_file, tensor_names) in &shard_to_tensors {
        let shard_path = model_path.join(shard_file);
        let shard_needed: HashSet<String> = tensor_names.iter().cloned().collect();
        match rustrain_deepseek_v4::fp8_kernel::load_safetensors_native(
            &shard_path, &shard_needed, -1,
        ) {
            Ok(shard_weights) => {
                for (name, t) in shard_weights {
                    weights.insert(name, t);
                }
            }
            Err(_) => {
                // Fallback to tch-rs
                let all_tensors = rustrain_checkpoint::safetensors::read_safetensors_dir(model_path)?;
                for name in &shard_needed {
                    if let Some(t) = all_tensors.get(name) {
                        weights.insert(name.clone(), t.shallow_clone());
                    }
                }
            }
        }
    }
    Ok(weights)
}
