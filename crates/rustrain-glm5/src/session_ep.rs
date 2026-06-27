use std::collections::{BTreeMap, HashSet};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use tch::{no_grad, Device, Kind, Reduction, Tensor};
use tracing::info;

use crate::lora::*;
use crate::model::*;
use crate::model::{rms_norm, glm5_mlp};
use crate::sft::*;
use rustrain_checkpoint::safetensors::{read_safetensors_dir, tensor};
use rustrain_nccl::nccl::{self as nccl_smoke, NcclPersistentComm};

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
    let all_layers: Vec<usize> = (0..runtime_config.num_hidden_layers).collect();
    let target_modules: Vec<Glm5LoraTargetModule> = lora_config_raw
        .target_modules
        .iter()
        .map(|s| Glm5LoraTargetModule::from_name(s))
        .collect::<Result<Vec<_>>>()?;
    let lora_config = Glm5LoraConfig {
        rank: lora_config_raw.rank,
        alpha: lora_config_raw.alpha as i64,
        target_layers: all_layers.clone(),
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
            "wq_a.weight",
            "wq_a_layernorm.weight",
            "wq_b.weight",
            "wkv.weight",
            "wkv_a_layernorm.weight",
            "wkv_b.weight",
            "wo.weight",
        ] {
            needed.insert(format!("{p}.self_attn.{suffix}"));
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
            }
        }
        // Gate + shared experts (all layers, replicated)
        needed.insert(format!("{p}.mlp.gate.weight"));
        for suffix in &["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            needed.insert(format!("{p}.mlp.shared_experts.{suffix}"));
        }
        // Only LOCAL experts (EP sharded)
        for &e in &ep_shard.local_expert_indices {
            for suffix in &["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
                needed.insert(format!("{p}.mlp.experts.{e}.{suffix}"));
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

    let weights_gpu: BTreeMap<String, Tensor> = weights
        .into_iter()
        .map(|(name, t)| {
            let t = t.to_device(device).to_kind(compute_kind);
            (name, t)
        })
        .collect();
    info!(rank, tensors_on_gpu = weights_gpu.len(), "weights on GPU");

    // ── Create LoRA registry ──
    let registry = Glm5LoraRegistry::new(&weights_gpu, lora_config, device)?;
    let trainable_count = registry.var_store.trainable_variables().len();
    info!(
        rank,
        trainable_params = trainable_count,
        "LoRA adapters created"
    );

    // ── Barrier: wait for all ranks to finish loading ──
    let barrier_dir = run_paths.root.join("barrier");
    std::fs::create_dir_all(&barrier_dir)?;
    let ready_file = barrier_dir.join(format!("rank_{rank}.ready"));
    std::fs::write(&ready_file, b"ready")?;
    info!(rank, "waiting at barrier for all ranks to load weights");
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
            bail!("barrier timeout: only {ready_count}/{world_size} ranks ready");
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    info!(rank, "all ranks ready, starting training");

    // ── Create persistent NCCL communicator ──
    let nccl_comm = if world_size > 1 {
        let comm_dir = run_paths.root.join("nccl-comm");
        let comm = NcclPersistentComm::new(&comm_dir)?;
        info!(rank, "persistent NCCL communicator created");
        Some(comm)
    } else {
        None
    };

    // ── SFT data ──
    let tokenizer = tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
    let sft_jsonl = std::path::Path::new("data/sft/deepseek_test.jsonl");
    let train_dataset = if sft_jsonl.exists() {
        info!(rank, path = %sft_jsonl.display(), "loading real SFT data");
        Glm5SftDataset::from_jsonl_simple(sft_jsonl, &tokenizer)?
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

    // Pre-load indexer weights for all "full" layers (for IndexShare)
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

    // ── Training loop ──
    for step in 0..config.train.max_steps {
        // ── Forward ──
        let embed = tensor(&weights_gpu, "model.embed_tokens.weight")?.to_kind(compute_kind);
        let mut hidden = Tensor::embedding(&embed, &train_batch.input_ids, -1, false, false);
        if hidden.kind() != compute_kind {
            hidden = hidden.to_kind(compute_kind);
        }

        let mut index_share_state: Option<IndexShareState> = None;

        for layer in 0..n_layers {
            let p = format!("model.layers.{layer}");

            // ── Attention ──
            let attn_norm = tensor(&weights_gpu, &format!("{p}.input_layernorm.weight"))?
                .to_kind(compute_kind);
            let hidden_norm = rms_norm(&hidden, &attn_norm, runtime_config.rms_norm_eps);

            // Load attention weights
            let attn_weights = Glm5AttentionWeights::load_with_kind(&weights_gpu, layer, compute_kind)?;
            let lora_attn = lora_attention_weights(&attn_weights, layer, &registry);

            // Get indexer weights (from source layer for IndexShare)
            let source = runtime_config.indexer_source_layer(layer);
            let indexer_weights = indexer_weights_map.get(&source).unwrap_or(&lora_attn);

            let attn_out = glm5_dsa_attention(
                &hidden_norm,
                &lora_attn,
                indexer_weights,
                &runtime_config,
                &mut index_share_state,
                layer,
            )
            .to_kind(compute_kind);

            let residual = &hidden + &attn_out;

            // ── MoE / Dense MLP ──
            let post_norm = tensor(&weights_gpu, &format!("{p}.post_attention_layernorm.weight"))?
                .to_kind(compute_kind);
            let mlp_input = rms_norm(&residual, &post_norm, runtime_config.rms_norm_eps);

            if runtime_config.is_moe_layer(layer) {
                // MoE with EP — inline implementation (not glm5_moe_mlp which assumes all experts local)
                let gate = tensor(&weights_gpu, &format!("{p}.mlp.gate.weight"))?
                    .to_kind(compute_kind);
                let shared_gate = tensor(&weights_gpu, &format!("{p}.mlp.shared_experts.gate_proj.weight"))?
                    .to_kind(compute_kind);
                let shared_up = tensor(&weights_gpu, &format!("{p}.mlp.shared_experts.up_proj.weight"))?
                    .to_kind(compute_kind);
                let shared_down = tensor(&weights_gpu, &format!("{p}.mlp.shared_experts.down_proj.weight"))?
                    .to_kind(compute_kind);

                // Shared expert (replicated across all ranks)
                let shared_output = glm5_mlp(&mlp_input, &shared_gate, &shared_up, &shared_down);

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
                    let gate_w = tensor(&weights_gpu, &format!("{eg}.gate_proj.weight"))?.to_kind(compute_kind);
                    let up_w = tensor(&weights_gpu, &format!("{eg}.up_proj.weight"))?.to_kind(compute_kind);
                    let down_w = tensor(&weights_gpu, &format!("{eg}.down_proj.weight"))?.to_kind(compute_kind);
                    let expert_out = glm5_mlp(&flat_input, &gate_w, &up_w, &down_w);
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
            } else {
                // Dense MLP
                let gate = tensor(&weights_gpu, &format!("{p}.mlp.gate_proj.weight"))?
                    .to_kind(compute_kind);
                let up = tensor(&weights_gpu, &format!("{p}.mlp.up_proj.weight"))?
                    .to_kind(compute_kind);
                let down = tensor(&weights_gpu, &format!("{p}.mlp.down_proj.weight"))?
                    .to_kind(compute_kind);
                let mlp = glm5_mlp(&mlp_input, &gate, &up, &down);
                hidden = &residual + &mlp;
            }

            if hidden.kind() != compute_kind {
                hidden = hidden.to_kind(compute_kind);
            }
        }

        // ── Final norm + lm_head ──
        let final_norm = tensor(&weights_gpu, "model.norm.weight")?.to_kind(compute_kind);
        let normed = rms_norm(&hidden, &final_norm, runtime_config.rms_norm_eps);
        let lm_head = if runtime_config.tie_word_embeddings {
            embed.shallow_clone()
        } else {
            tensor(&weights_gpu, "lm_head.weight")?.to_kind(compute_kind)
        };
        let logits = normed.linear::<&Tensor>(&lm_head, None);

        // ── SFT Loss ──
        let shifted_logits = logits.narrow(1, 0, logits.size()[1] - 1);
        let shifted_targets = train_batch
            .input_ids
            .narrow(1, 1, train_batch.input_ids.size()[1] - 1);
        let shifted_mask = train_batch
            .target_mask
            .narrow(1, 1, train_batch.target_mask.size()[1] - 1)
            .to_kind(Kind::Float);
        let batch_size = shifted_logits.size()[0];
        let seq_len = shifted_logits.size()[1];

        let log_probs = shifted_logits
            .reshape([-1, runtime_config.vocab_size])
            .log_softmax(-1, Kind::Float);
        let per_token_loss = log_probs
            .g_nll_loss::<&Tensor>(&shifted_targets.reshape([-1]), None, Reduction::None, -100)
            .reshape([batch_size, seq_len]);
        let masked_loss = &per_token_loss * &shifted_mask;
        let total_mask = shifted_mask.sum(Kind::Float);
        let loss = masked_loss.sum(Kind::Float) / total_mask.clamp_min(1.0);

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

        // ── LoRA gradient all-reduce ──
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

/// Load GLM-5.2 weights from safetensors directory (mmap)
fn load_glm5_weights(
    model_path: &std::path::Path,
    needed: &HashSet<String>,
) -> Result<BTreeMap<String, Tensor>> {
    let weights = read_safetensors_dir(model_path)
        .with_context(|| format!("failed to read safetensors from {}", model_path.display()))?;

    // Filter to only needed tensors
    let filtered: BTreeMap<String, Tensor> = weights
        .into_iter()
        .filter(|(k, _)| {
            needed.contains(k.as_str())
                || needed.iter().any(|n| k.starts_with(n))
        })
        .collect();

    Ok(filtered)
}
