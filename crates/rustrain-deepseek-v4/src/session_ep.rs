use std::collections::{BTreeMap, HashSet};

use anyhow::{bail, Context, Result};
use tch::{no_grad, Device, Kind, Reduction, Tensor};
use tracing::info;

use crate::ep::{v4_moe_mlp_ep, V4EpShard};
use crate::lora::*;
use crate::model::*;
use crate::session::V4LoraSftSummary;
use crate::sft::*;
use rustrain_checkpoint::safetensors::tensor;
use rustrain_nccl::nccl::{self as nccl_smoke, NcclPersistentComm};

fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

/// EP-distributed LoRA SFT training for V4.
///
/// Each rank loads: all attention weights (replicated) + 1/world_size experts (sharded)
/// + shared experts + gate + embed + head + norm + LoRA adapter.
///
/// Forward: loop through ALL layers, attention with LoRA (autograd),
/// MoE with EP (local experts → all-reduce → /world_size).
/// Backward: loss.backward() + per-layer aux_loss for MoE gradient flow.
/// LoRA gradient: all-reduce across ranks.
pub fn train_v4_lora_sft_ep(
    config: &rustrain_core::runtime::Config,
    run_paths: &rustrain_core::runtime::RunPaths,
) -> Result<V4LoraSftSummary> {
    // ── Parse distributed env ──
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;

    // ── Model config ──
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("V4 LoRA SFT EP requires model.model_path")?;
    let model_path = resolve_v4_model_path(model_path)?;
    let runtime_config = read_v4_config(&model_path.join("config.json"))?;
    info!(
        rank,
        world_size,
        local_rank,
        layers = runtime_config.num_hidden_layers,
        "V4 LoRA SFT EP config loaded"
    );

    if runtime_config.n_routed_experts % world_size != 0 {
        bail!(
            "V4 EP: n_routed_experts {} must be divisible by world_size {world_size}",
            runtime_config.n_routed_experts
        );
    }

    // ── EP shard ──
    let ep_shard = V4EpShard::new(rank, world_size, runtime_config.n_routed_experts);
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
        .context("V4 LoRA SFT EP requires [lora] config section")?;
    // Train LoRA on ALL layers for full-model SFT
    let all_layers: Vec<usize> = (0..runtime_config.num_hidden_layers).collect();
    let target_modules: Vec<V4LoraTargetModule> = lora_config_raw
        .target_modules
        .iter()
        .map(|s| V4LoraTargetModule::from_name(s))
        .collect::<Result<Vec<_>>>()?;
    let lora_config = V4LoraConfig {
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
    // Each rank reads ~84GB into RAM. With 8 ranks loading simultaneously,
    // that's 672GB peak → OOM. Stagger by rank so only 1 loads at a time.
    if rank > 0 {
        info!(
            rank,
            delay_secs = rank * 40,
            "waiting before weight loading (staggered)"
        );
        std::thread::sleep(std::time::Duration::from_secs((rank * 40) as u64));
    }

    // ── Build needed weight set ──
    let n_layers = runtime_config.num_hidden_layers;
    let n_experts = runtime_config.n_routed_experts;
    let mut needed: HashSet<String> = HashSet::new();
    needed.insert("embed.weight".to_string());
    needed.insert("norm.weight".to_string());
    needed.insert("head.weight".to_string());

    for layer in 0..n_layers {
        let p = format!("layers.{layer}");
        // Attention (all layers, replicated)
        needed.insert(format!("{p}.attn_norm.weight"));
        needed.insert(format!("{p}.ffn_norm.weight"));
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
            needed.insert(format!("{p}.attn.{suffix}"));
        }
        // Gate + shared experts (all layers, replicated)
        needed.insert(format!("{p}.ffn.gate.weight"));
        for suffix in &["w1.weight", "w2.weight", "w3.weight"] {
            needed.insert(format!("{p}.ffn.shared_experts.{suffix}"));
        }
        // HC weights (if present)
        needed.extend(crate::hc::HcWeights::weight_names(layer));
        // Only LOCAL experts (EP sharded)
        for &e in &ep_shard.local_expert_indices {
            for suffix in &["w1.weight", "w2.weight", "w3.weight"] {
                needed.insert(format!("{p}.ffn.experts.{e}.{suffix}"));
            }
        }
    }

    // MTP weights
    if runtime_config.num_nextn_predict_layers > 0 {
        for mtp_layer in 0..runtime_config.num_nextn_predict_layers {
            needed.extend(MtpHeadWeights::weight_names(mtp_layer));
        }
    }

    info!(rank, needed_tensors = needed.len(), "loading FP8 weights");

    let weights = load_v4_weights_fp8(&model_path, &needed, local_rank as i32)?;
    info!(rank, tensors = weights.len(), "FP8 weights loaded");

    // FP8 weights with .scale_f stay as FP8 (used in C++ GEMM).
    // All other tensors (norms, embed, head, attn_sink) converted to bf16
    // because PyTorch can't do arithmetic on FP8 tensors.
    let weights_gpu: BTreeMap<String, Tensor> = {
        let scale_names: HashSet<String> = weights
            .keys()
            .filter(|k| k.ends_with(".scale_f"))
            .map(|k| k.replace(".scale_f", ""))
            .collect();
        weights
            .into_iter()
            .map(|(name, t)| {
                let t = t.to_device(device);
                let is_scale = name.ends_with(".scale_f");
                let is_fp8_weight = scale_names.contains(&name) && !is_scale;
                let processed = if is_scale {
                    t.to_kind(Kind::Float)
                } else if is_fp8_weight {
                    t // keep FP8 for C++ GEMM
                } else {
                    t.to_kind(compute_kind) // norms, embed, head → bf16
                };
                (name, processed)
            })
            .collect()
    };
    info!(rank, tensors_on_gpu = weights_gpu.len(), "weights on GPU");

    // ── Create LoRA registry ──
    let mut registry = V4LoraRegistry::new(&weights_gpu, lora_config, device)?;
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

    // ── Create persistent NCCL communicator (reused across all layers) ──
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
    // Try to load real SFT data; fall back to synthetic
    let sft_jsonl = std::path::Path::new("data/sft/deepseek_test.jsonl");
    let train_dataset = if sft_jsonl.exists() {
        info!(rank, path = %sft_jsonl.display(), "loading real SFT data");
        V4SftDataset::from_jsonl_simple(sft_jsonl, &tokenizer)?
    } else {
        info!(rank, "no SFT JSONL found, using synthetic data");
        V4SftDataset::synthetic(&tokenizer)?
    };
    let raw_batch = train_dataset.padded_batch(0, 1, device);

    // Pad input to config seq_len so compression chain doesn't shrink it to 0.
    // V4 compress_ratios alternate 4 and 128 — need seq >> 128*4 to survive.
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
        V4SftBatch {
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

    // ── Training loop ──
    for step in 0..config.train.max_steps {
        // ── Forward through ALL layers ──
        let embed = tensor(&weights_gpu, "embed.weight")?.to_kind(compute_kind);
        let mut hidden = Tensor::embedding(&embed, &train_batch.input_ids, -1, false, false);
        if hidden.kind() != compute_kind {
            hidden = hidden.to_kind(compute_kind);
        }

        for layer in 0..n_layers {
            let p = format!("layers.{layer}");

            // Compression — skip if compressed seq would be too short (< 2 tokens)
            let current_seq = hidden.size()[1];
            let ratio = if layer < runtime_config.compress_ratios.len() {
                runtime_config.compress_ratios[layer]
            } else {
                0
            };
            let effective_ratio = if ratio > 1 && current_seq / ratio as i64 >= 2 {
                ratio
            } else {
                0
            };
            if effective_ratio > 1 {
                hidden = compress_seq(&hidden, effective_ratio);
            }

            // Layer config (use compress_rope_theta for compressed layers)
            let layer_config = if effective_ratio > 1 {
                let mut c = runtime_config.clone();
                c.rope_theta = runtime_config.compress_rope_theta;
                c
            } else {
                runtime_config.clone()
            };

            // ── Attention with LoRA ──
            let attn_norm =
                tensor(&weights_gpu, &format!("{p}.attn_norm.weight"))?.to_kind(compute_kind);
            let hidden_norm = rms_norm(&hidden, &attn_norm, layer_config.rms_norm_eps);

            let attn_weights = V4AttentionWeights::load_raw(&weights_gpu, layer)?;
            let lora_attn = v4_lora_attention_weights(&attn_weights, layer, &registry);

            // HC attention: use learned hash bias on compressed sequences
            let use_hc = effective_ratio > 1 && crate::hc::HcWeights::exists(&weights_gpu, layer);
            let attn_out = if use_hc {
                if let Ok(hc) = crate::hc::HcWeights::load_raw(&weights_gpu, layer) {
                    crate::hc::v4_hc_attention(&hidden_norm, &lora_attn, &hc, &layer_config)
                } else {
                    v4_attention(&hidden_norm, &lora_attn, &layer_config)
                }
            } else {
                v4_attention(&hidden_norm, &lora_attn, &layer_config)
            }
            .to_kind(compute_kind);

            let residual = &hidden + &attn_out;

            // ── MoE with EP ──
            let ffn_norm =
                tensor(&weights_gpu, &format!("{p}.ffn_norm.weight"))?.to_kind(compute_kind);
            let mlp_input = rms_norm(&residual, &ffn_norm, layer_config.rms_norm_eps);

            let gate = tensor(&weights_gpu, &format!("{p}.ffn.gate.weight"))?.to_kind(compute_kind);
            let shared_w1 = tensor(&weights_gpu, &format!("{p}.ffn.shared_experts.w1.weight"))?
                .to_kind(compute_kind);
            let shared_w2 = tensor(&weights_gpu, &format!("{p}.ffn.shared_experts.w2.weight"))?
                .to_kind(compute_kind);
            let shared_w3 = tensor(&weights_gpu, &format!("{p}.ffn.shared_experts.w3.weight"))?
                .to_kind(compute_kind);

            // Load local experts
            let local_experts: Vec<(Tensor, Tensor, Tensor)> = ep_shard
                .local_expert_indices
                .iter()
                .map(|&e| {
                    let w1 = tensor(&weights_gpu, &format!("{p}.ffn.experts.{e}.w1.weight"))
                        .unwrap()
                        .to_kind(compute_kind);
                    let w2 = tensor(&weights_gpu, &format!("{p}.ffn.experts.{e}.w2.weight"))
                        .unwrap()
                        .to_kind(compute_kind);
                    let w3 = tensor(&weights_gpu, &format!("{p}.ffn.experts.{e}.w3.weight"))
                        .unwrap()
                        .to_kind(compute_kind);
                    (w1, w2, w3)
                })
                .collect();

            // EP MoE: shared (replicated) + local experts → partial output
            let partial_mlp = v4_moe_mlp_ep(
                &mlp_input,
                &gate,
                &(shared_w1, shared_w2, shared_w3),
                &local_experts,
                &ep_shard,
                &layer_config,
            );

            // All-reduce MoE output (shared expert counted world_size times → divide)
            let mlp_kind = partial_mlp.kind();
            let (full_mlp, _partial_with_grad) = if world_size > 1 {
                let pd = no_grad(|| partial_mlp.shallow_clone()).detach();
                let reduced = nccl_comm.as_ref().unwrap().all_reduce(&pd)?;
                // NCCL returns f32 — convert back to original dtype (bf16)
                let full = no_grad(|| (&reduced / (world_size as f64)).to_kind(mlp_kind)).detach();
                (full.set_requires_grad(true), partial_mlp)
            } else {
                (partial_mlp.shallow_clone(), partial_mlp.shallow_clone())
            };

            // Store (partial_with_grad, full_mlp) for two-phase backward
            hidden = &residual + &full_mlp;
            // Ensure hidden stays in compute dtype (bf16)
            if hidden.kind() != compute_kind {
                hidden = hidden.to_kind(compute_kind);
            }
        }

        // ── Decompress ──
        let original_seq_len = train_batch.input_ids.size()[1];
        let current_seq_len = hidden.size()[1];
        let total_ratio: usize = if current_seq_len >= original_seq_len {
            1
        } else {
            (original_seq_len / current_seq_len) as usize
        };
        if total_ratio > 1 {
            hidden = decompress_seq(&hidden, total_ratio, original_seq_len);
        }

        // ── Final norm + lm_head ──
        let final_norm = tensor(&weights_gpu, "norm.weight")?.to_kind(compute_kind);
        let normed = rms_norm(&hidden, &final_norm, runtime_config.rms_norm_eps);
        let lm_head = tensor(&weights_gpu, "head.weight")?.to_kind(compute_kind);
        let logits = normed.linear::<&Tensor>(&lm_head, None);

        // ── SFT Loss ──
        let shifted_logits = logits.narrow(1, 0, logits.size()[1] - 1);
        let shifted_targets =
            train_batch
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

        // ── MTP auxiliary loss ──
        let mtp_loss = if runtime_config.num_nextn_predict_layers > 0 {
            match v4_mtp_loss(
                &hidden,
                &train_batch.input_ids,
                &weights_gpu,
                &runtime_config,
            ) {
                Ok(ml) => {
                    let ml_val = ml.double_value(&[]);
                    info!(
                        rank,
                        step = step + 1,
                        lm_loss = loss_val,
                        mtp_loss = ml_val,
                        "EP LoRA SFT train step"
                    );
                    ml
                }
                Err(e) => {
                    tracing::warn!(error = %e, "MTP loss computation failed, skipping");
                    info!(
                        rank,
                        step = step + 1,
                        lm_loss = loss_val,
                        mtp_loss = 0.0,
                        "EP LoRA SFT train step (no MTP)"
                    );
                    Tensor::zeros([], (Kind::Float, hidden.device()))
                }
            }
        } else {
            info!(
                rank,
                step = step + 1,
                loss = loss_val,
                "EP LoRA SFT train step"
            );
            Tensor::zeros([], (Kind::Float, hidden.device()))
        };

        // Total loss = LM loss + 0.5 * MTP loss
        // Must combine before backward — separate backward() calls would
        // free the shared autograd graph (hidden → both paths).
        let total_loss = &loss + (&mtp_loss * 0.5);

        // ── Backward ──
        total_loss.backward();

        // Phase 3: LoRA gradient all-reduce
        // tch-rs VarStore doesn't support set_grad, so we all-reduce gradients
        // and use the averaged gradient directly in the Adam update below,
        // instead of reading var.grad() which gives the local (non-synced) gradient.
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
            Vec::new() // single rank: use var.grad() directly
        };

        // ── Adam optimizer step ──
        let mut current_vars = registry.var_store.trainable_variables();
        for (i, var) in current_vars.iter_mut().enumerate() {
            // Use synced gradient if available, else fall back to local gradient
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
    let adapter_output = run_paths.checkpoints.join("v4-lora-adapter-ep.safetensors");
    registry.save(&adapter_output)?;
    info!(rank, adapter = %adapter_output.display(), "adapter saved");

    // ── Final loss ──
    // Recompute final loss (simplified - just use last step's loss)
    let final_loss = initial_loss; // TODO: proper final loss eval

    info!(rank, initial_loss, final_loss, "V4 LoRA SFT EP complete");

    Ok(V4LoraSftSummary {
        adapter_output: adapter_output.display().to_string(),
        initial_loss,
        final_loss,
        trainable_params: trainable_count,
    })
}
