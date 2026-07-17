//! Qwen3.6 training session — single-GPU LoRA SFT + EP4 distributed training.

use std::collections::{BTreeMap, HashSet};
use std::env;

use anyhow::{Context, Result, anyhow, bail};
use tch::{Kind, Tensor};
use tracing::info;

use crate::config::{
    LayerType, Qwen36RuntimeConfig, read_qwen36_runtime_config, resolve_qwen36_model_path,
};
use crate::lora::{
    Qwen36AdapterArtifact, Qwen36LoraConfig, Qwen36LoraTargetModule, validate_lora_targets,
};
use crate::sft::SftDataset;
use rustrain_checkpoint::safetensors::read_safetensors_dir_filtered;
use rustrain_core::runtime::{Config, RunPaths};

// ──────────────────────────────────────────────────────────────────────
// EP Shard
// ──────────────────────────────────────────────────────────────────────

pub struct EpShard {
    pub rank: usize,
    pub world_size: usize,
    pub experts_per_rank: usize,
    pub expert_start: usize,
    pub local_expert_indices: Vec<usize>,
}

impl EpShard {
    pub fn new(rank: usize, world_size: usize, num_experts: usize) -> Self {
        assert!(
            num_experts % world_size == 0,
            "num_experts {num_experts} not divisible by world_size {world_size}"
        );
        let epr = num_experts / world_size;
        let start = rank * epr;
        Self {
            rank,
            world_size,
            experts_per_rank: epr,
            expert_start: start,
            local_expert_indices: (start..start + epr).collect(),
        }
    }

    pub fn owns_expert(&self, global_idx: usize) -> bool {
        global_idx >= self.expert_start && global_idx < self.expert_start + self.experts_per_rank
    }
}

// ──────────────────────────────────────────────────────────────────────
// Summary
// ──────────────────────────────────────────────────────────────────────

pub struct Qwen36LoraSftSummary {
    pub adapter_output: String,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub trainable_params: usize,
}

// ──────────────────────────────────────────────────────────────────────
// Weight name helpers
// ──────────────────────────────────────────────────────────────────────

fn parse_env_usize(key: &str) -> Result<usize> {
    env::var(key)
        .with_context(|| format!("{key} not set"))
        .and_then(|v| {
            v.parse::<usize>()
                .with_context(|| format!("invalid {key}: {v}"))
        })
}

fn lora_config_from_config(config: &Config) -> Result<Qwen36LoraConfig> {
    let lora = config
        .lora
        .as_ref()
        .ok_or_else(|| anyhow!("[lora] section required"))?;
    let target_layers: Vec<usize> = lora.target_layers.clone();
    let target_modules: Vec<Qwen36LoraTargetModule> = lora
        .target_modules
        .iter()
        .map(|m| Qwen36LoraTargetModule::parse(m))
        .collect::<Result<Vec<_>>>()?;
    Ok(Qwen36LoraConfig {
        rank: lora.rank,
        alpha: lora.alpha,
        target_layers,
        target_modules,
    })
}

/// Build the set of weight names needed for training.
/// For EP mode, only load local expert slice.
fn build_needed_weights(
    config: &Qwen36RuntimeConfig,
    lora_config: &Qwen36LoraConfig,
    ep_shard: Option<&EpShard>,
) -> HashSet<String> {
    let p = &config.weight_prefix;
    let mut needed = HashSet::new();

    // Embed, norm, lm_head
    needed.insert(format!("{p}embed_tokens.weight"));
    needed.insert(format!("{p}norm.weight"));
    if !config.tie_word_embeddings {
        needed.insert("lm_head.weight".to_string());
    }

    // Per-layer weights for ALL layers (not just LoRA targets)
    for layer in 0..config.num_hidden_layers {
        let lp = format!("{p}layers.{layer}");
        needed.insert(format!("{lp}.input_layernorm.weight"));
        needed.insert(format!("{lp}.post_attention_layernorm.weight"));

        // Attention weights (full or linear)
        match config.layer_types[layer] {
            LayerType::FullAttention => {
                for w in &["q_proj", "q_norm", "k_proj", "k_norm", "v_proj", "o_proj"] {
                    needed.insert(format!("{lp}.self_attn.{w}.weight"));
                }
            }
            LayerType::LinearAttention => {
                needed.insert(format!("{lp}.linear_attn.A_log"));
                needed.insert(format!("{lp}.linear_attn.conv1d.weight"));
                needed.insert(format!("{lp}.linear_attn.dt_bias"));
                needed.insert(format!("{lp}.linear_attn.norm.weight"));
                for w in &[
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_a",
                    "in_proj_b",
                    "out_proj",
                ] {
                    needed.insert(format!("{lp}.linear_attn.{w}.weight"));
                }
            }
        }

        // MLP weights — dense vs MoE
        if config.is_moe {
            needed.insert(format!("{lp}.mlp.gate.weight"));
            needed.insert(format!("{lp}.mlp.shared_expert_gate.weight"));
            needed.insert(format!("{lp}.mlp.shared_expert.gate_proj.weight"));
            needed.insert(format!("{lp}.mlp.shared_expert.up_proj.weight"));
            needed.insert(format!("{lp}.mlp.shared_expert.down_proj.weight"));
            // Fused expert tensors are 3D [num_experts, ...], loaded as a whole
            needed.insert(format!("{lp}.mlp.experts.gate_up_proj"));
            needed.insert(format!("{lp}.mlp.experts.down_proj"));
        } else {
            // Dense MLP: standard SwiGLU (gate_proj, up_proj, down_proj)
            needed.insert(format!("{lp}.mlp.gate_proj.weight"));
            needed.insert(format!("{lp}.mlp.up_proj.weight"));
            needed.insert(format!("{lp}.mlp.down_proj.weight"));
        }
    }

    // Vision encoder (for multimodal)
    if config.has_vision {
        for name in crate::vision::VisionWeights::weight_names(config) {
            needed.insert(name);
        }
    }

    needed
}

// ──────────────────────────────────────────────────────────────────────
// Entry point: single-GPU LoRA SFT
// ──────────────────────────────────────────────────────────────────────

pub fn train_qwen3_6_lora_sft(
    config: &Config,
    run_paths: &RunPaths,
) -> Result<Qwen36LoraSftSummary> {
    train_impl(config, run_paths, None)
}

// ──────────────────────────────────────────────────────────────────────
// Entry point: EP4 distributed LoRA SFT
// ──────────────────────────────────────────────────────────────────────

pub fn train_qwen3_6_lora_sft_ep(
    config: &Config,
    run_paths: &RunPaths,
) -> Result<Qwen36LoraSftSummary> {
    let rank = parse_env_usize("RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    // For MoE models, shard experts. For dense models, EP is a no-op (no experts to shard).
    let model_path = config
        .model
        .model_path
        .as_ref()
        .ok_or_else(|| anyhow!("model.model_path required"))?;
    let model_path = resolve_qwen36_model_path(model_path)?;
    let runtime_config = read_qwen36_runtime_config(&model_path)?;
    let ep_shard = if runtime_config.is_moe {
        Some(EpShard::new(rank, world_size, runtime_config.num_experts))
    } else {
        None
    };
    train_impl(config, run_paths, ep_shard)
}

// ──────────────────────────────────────────────────────────────────────
// Core training implementation
// ──────────────────────────────────────────────────────────────────────

fn train_impl(
    config: &Config,
    run_paths: &RunPaths,
    ep_shard: Option<EpShard>,
) -> Result<Qwen36LoraSftSummary> {
    let model_path = config
        .model
        .model_path
        .as_ref()
        .ok_or_else(|| anyhow!("model.model_path required"))?;
    let model_path = resolve_qwen36_model_path(model_path)?;
    let runtime_config = read_qwen36_runtime_config(&model_path)?;
    let lora_config = lora_config_from_config(config)?;
    validate_lora_targets(&runtime_config, &lora_config)?;
    let device = match config.train.device {
        rustrain_core::runtime::Device::Cuda => {
            // EP mode: use LOCAL_RANK to select the correct GPU
            let local_rank = std::env::var("LOCAL_RANK")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            tch::Device::Cuda(local_rank)
        }
        rustrain_core::runtime::Device::Cpu => tch::Device::Cpu,
    };
    let compute_kind = match config.train.dtype {
        rustrain_core::runtime::DType::Fp16 => Kind::Half,
        rustrain_core::runtime::DType::Bf16 => Kind::BFloat16,
        rustrain_core::runtime::DType::Fp32 => Kind::Float,
    };
    if compute_kind != Kind::BFloat16 {
        bail!(
            "native Qwen3.5/3.6 LoRA currently supports bf16 only; {:?} would not be updated by the fused Adam kernel",
            config.train.dtype
        );
    }

    let shard_ref = ep_shard.as_ref();
    let is_ep = shard_ref.is_some();
    let env_enabled = |name: &str| {
        std::env::var(name)
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false)
    };
    let ep_a2a = env_enabled("QWEN36_EP_A2A");
    let ep_a2a_sharded = env_enabled("QWEN36_EP_A2A_SHARDED");
    if ep_a2a_sharded && !is_ep {
        bail!("QWEN36_EP_A2A_SHARDED=1 requires expert-parallel training");
    }
    if ep_a2a_sharded && !ep_a2a {
        bail!("QWEN36_EP_A2A_SHARDED=1 requires QWEN36_EP_A2A=1");
    }
    // Non-EP Qwen sessions may run replicated-weight LoRA data parallelism.
    // The launcher supplies the standard torchrun environment; EP keeps its
    // explicit shard metadata as the source of truth.
    let env_world_size = std::env::var("WORLD_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    let env_rank = std::env::var("RANK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let world_size = shard_ref.map(|s| s.world_size).unwrap_or(env_world_size);
    let rank = shard_ref.map(|s| s.rank).unwrap_or(env_rank);
    let tp_size = config.parallel.tensor_model_parallel_size;
    let env_tp_size = std::env::var("TP_SIZE")
        .or_else(|_| std::env::var("RUSTRAIN_TP_SIZE"))
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    if env_tp_size > 1 && env_tp_size != tp_size {
        bail!(
            "TP_SIZE environment ({env_tp_size}) does not match config tensor_model_parallel_size ({tp_size})"
        );
    }
    if tp_size > 1 {
        if world_size != tp_size
            || config.parallel.pipeline_model_parallel_size != 1
            || config.parallel.data_parallel_size != 1
            || config.parallel.expert_model_parallel_size != 1
            || config.parallel.context_parallel_size != 1
        {
            bail!(
                "native Qwen LoRA currently supports TP-only topology: TP={} WORLD_SIZE={} PP={} DP={} EP={} CP={}",
                tp_size,
                world_size,
                config.parallel.pipeline_model_parallel_size,
                config.parallel.data_parallel_size,
                config.parallel.expert_model_parallel_size,
                config.parallel.context_parallel_size
            );
        }
        if lora_config.rank % tp_size as i64 != 0 {
            bail!(
                "LoRA rank {} must be divisible by TP_SIZE={tp_size}",
                lora_config.rank
            );
        }
        unsafe {
            std::env::set_var("TP_SIZE", tp_size.to_string());
        }
    }
    let is_data_parallel = !is_ep && world_size > 1 && tp_size == 1;
    if is_data_parallel && runtime_config.is_moe {
        bail!(
            "replicated Qwen data parallelism is only supported for dense/linear-attention models; use *_ep for MoE"
        );
    }
    if is_data_parallel || tp_size > 1 {
        crate::kernel::CppTrainingContext::set_cuda_device(
            std::env::var("LOCAL_RANK")
                .ok()
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(0),
        );
    }

    // Dense base-weight TP follows Megatron's ColumnParallel/RowParallel MLP:
    // gate/up rows are owned by one TP rank and down columns are owned by the
    // same rank.  Attention and embeddings remain replicated in this first
    // unit.  MoE, MTP, and mixed TP/DP/EP are rejected rather than silently
    // running with an unsharded path.
    let base_tp_mlp = tp_size > 1 && !runtime_config.is_moe;
    if base_tp_mlp {
        if runtime_config.mtp_num_hidden_layers > 0 {
            bail!("base dense TP MLP currently requires MTP to be disabled");
        }
        if runtime_config.intermediate_size <= 0
            || runtime_config.intermediate_size % tp_size as i64 != 0
        {
            bail!(
                "dense intermediate_size={} must be divisible by TP_SIZE={tp_size}",
                runtime_config.intermediate_size
            );
        }
        if lora_config.target_modules.is_empty()
            || lora_config.target_modules.iter().any(|module| {
                matches!(
                    module.cpp_name(),
                    "gate_proj"
                        | "up_proj"
                        | "down_proj"
                        | "shared_gate_proj"
                        | "shared_up_proj"
                        | "shared_down_proj"
                        | "experts_gate_up_proj"
                        | "experts_down_proj"
                )
            })
        {
            bail!(
                "base dense TP MLP currently requires explicit attention-only LoRA target_modules"
            );
        }
    }

    // Build needed weight set
    let needed = build_needed_weights(&runtime_config, &lora_config, shard_ref);

    // Stagger loading for EP to avoid OOM
    if is_ep {
        std::thread::sleep(std::time::Duration::from_secs(rank as u64 * 5));
    }

    info!(
        "loading {} weight tensors from {}",
        needed.len(),
        model_path.display()
    );
    let weights = read_safetensors_dir_filtered(&model_path, &needed)?;

    // Move to device — for EP, narrow expert tensors on CPU first to save GPU memory
    let mut weights_gpu = BTreeMap::new();
    if let Some(shard) = shard_ref {
        // EP mode: narrow expert tensors on CPU before transferring to GPU
        let num_experts = runtime_config.num_experts as i64;
        for (name, tensor) in &weights {
            // Check if this is an expert tensor that needs narrowing
            let needs_narrow = name.contains(".mlp.experts.gate_up_proj")
                || name.contains(".mlp.experts.down_proj");
            if needs_narrow && tensor.size()[0] == num_experts {
                // Narrow to local shard: [rank * experts_per_rank, ...]
                let narrowed = tensor
                    .narrow(0, shard.expert_start as i64, shard.experts_per_rank as i64)
                    .contiguous()
                    .to_device(device)
                    .to_kind(compute_kind);
                weights_gpu.insert(name.clone(), narrowed);
            } else if needs_narrow && tensor.size()[0] != num_experts {
                // Already narrowed (e.g., MTP layer experts) — load as-is
                weights_gpu.insert(name.clone(), tensor.to_device(device).to_kind(compute_kind));
            } else {
                // Non-expert weights: replicate on all GPUs
                weights_gpu.insert(name.clone(), tensor.to_device(device).to_kind(compute_kind));
            }
        }
        info!(
            "EP{}: narrowed expert tensors to {} experts per rank",
            world_size, shard.experts_per_rank
        );
    } else {
        for (name, tensor) in &weights {
            let local_shard = if base_tp_mlp {
                crate::kernel::shard_dense_mlp_weight_for_tp(name, tensor, tp_size, rank % tp_size)?
            } else {
                None
            };
            let gpu_tensor = local_shard
                .as_ref()
                .unwrap_or(tensor)
                .to_device(device)
                .to_kind(compute_kind);
            weights_gpu.insert(name.clone(), gpu_tensor);
        }
        if base_tp_mlp {
            info!(
                tp_size,
                tp_rank = rank % tp_size,
                "base dense MLP TP enabled: gate/up row shards and down column shards"
            );
        }
    }

    info!(
        "LoRA config: rank={}, alpha={}",
        lora_config.rank, lora_config.alpha
    );

    // Load SFT data
    let tokenizer_path = model_path.join("tokenizer.json");
    let data = if let Some(data_config) = &config.data {
        let path = &data_config.paths[0];
        SftDataset::from_jsonl(path, &tokenizer_path, config.model.seq_len)?
    } else {
        bail!("[data] section required for SFT training");
    };

    // Load MTP weights into weights_gpu if available (C++ uses them via set_mtp_weights)
    if runtime_config.mtp_num_hidden_layers > 0 {
        let mtp_names = crate::mtp::MtpWeights::weight_names(&runtime_config);
        let mtp_needed: HashSet<String> = mtp_names.into_iter().collect();
        let mtp_tensors = read_safetensors_dir_filtered(&model_path, &mtp_needed)?;
        let mut mtp_gpu = BTreeMap::new();
        for (name, tensor) in &mtp_tensors {
            mtp_gpu.insert(name.clone(), tensor.to_device(device).to_kind(compute_kind));
        }
        for (name, tensor) in mtp_gpu {
            weights_gpu.insert(name, tensor);
        }
    }

    let batch_size = config.train.micro_batch_size;
    let max_steps = config.train.max_steps as usize;
    let gradient_accumulation_steps = config.train.gradient_accumulation_steps;

    // ── C++ all-in-C++ training path (required) ──
    // LoRA A/B, Adam optimizer, forward, loss, backward all in C++.
    if !crate::kernel::kernels_available() {
        bail!(
            "C++ kernels (libqwen36_kernels.so) not found — required for training. Ensure the .so is in LD_LIBRARY_PATH."
        );
    }

    let expert_start = shard_ref.map(|s| s.expert_start).unwrap_or(0);
    let expert_count = shard_ref
        .map(|s| s.experts_per_rank)
        .unwrap_or(runtime_config.num_experts);
    let ctx = crate::kernel::CppTrainingContext::new(
        &weights_gpu,
        &runtime_config,
        compute_kind,
        config.train.learning_rate as f64,
        config.train.adam_beta1 as f64,
        config.train.adam_beta2 as f64,
        config.train.adam_eps as f64,
        lora_config.alpha as f64 / lora_config.rank as f64, // lora scaling = alpha / rank
        lora_config.rank as i64,
        base_tp_mlp,
        &lora_config.target_layers,
        &lora_config.target_modules,
        expert_start,
        expert_count,
    )?;

    if world_size > 1 {
        // Tell the native reducer whether this communicator is replicated DP
        // or expert-parallel. EP already all-reduces routed activations and
        // must not average replicated LoRA gradients a second time.
        unsafe {
            std::env::set_var(
                "RUSTRAIN_DATA_PARALLEL",
                if is_data_parallel { "1" } else { "0" },
            );
        }
        let ret = ctx.init_nccl();
        if ret != 0 {
            bail!("C++ NCCL init failed (code {})", ret);
        }
        info!(
            rank,
            world_size, is_ep, "NCCL communicator created for Qwen LoRA parallel training"
        );
    }
    info!("C++ TrainingContext: {} LoRA params", ctx.lora_count());

    // Set MTP weights if available
    if runtime_config.mtp_num_hidden_layers > 0 {
        ctx.set_mtp_weights(&weights_gpu, &runtime_config, expert_start, expert_count)?;
        info!(
            "C++ TrainingContext: MTP weights set ({} layers)",
            runtime_config.mtp_num_hidden_layers
        );
    }

    // Enable gradient checkpointing if env var set
    if let Ok(gs) = std::env::var("QWEN36_CHECKPOINT_GROUP") {
        let group_size: i64 = gs.parse().unwrap_or(4);
        ctx.set_checkpoint(true, group_size);
        info!("C++ TrainingContext: gradient checkpointing ON (group_size={group_size})");
    }

    // Training loop
    let mut initial_loss = 0.0_f64;
    let mut final_loss = 0.0_f64;

    for step in 0..max_steps {
        let mut loss_value = 0.0;
        for accumulation_index in 0..gradient_accumulation_steps {
            let micro_step = step * gradient_accumulation_steps + accumulation_index;
            let data_start = if is_data_parallel || ep_a2a_sharded {
                (micro_step * batch_size * world_size + rank * batch_size) % data.len()
            } else {
                (micro_step * batch_size) % data.len()
            };
            let sft_batch = data.batch(data_start, batch_size);
            let (input_ids, target_mask) = sft_batch.to_tensors(device, compute_kind);

            // Build attention mask: 1 for real tokens, 0 for padding
            // target_mask > 0 means the token is either response (loss) or prompt (no loss but attend)
            // We need: 1 for all non-padding tokens, 0 for padding tokens
            // The SFT batch's target_mask is: 0=prompt, 1=response, but padding has mask=0 too.
            // We need attention_mask = (target_mask >= 0).to(float) but that's all 1s.
            // Actually: padding tokens have target_mask=0 AND are after EOS.
            // The SftBatch already has padding at the end with mask=0.
            // We need: attention_mask = 1 where token is NOT padding.
            // Since prompt tokens have mask=0 and response tokens have mask=1,
            // but padding also has mask=0, we can't distinguish prompt from padding using mask alone.
            // Solution: use the pad_token_id to build attention mask from input_ids.
            let pad_id = data.pad_token_id();
            let attention_mask = input_ids.ne(pad_id).to_kind(Kind::Float).unsqueeze(0); // [1, seq]

            loss_value += ctx.train_micro_step(
                &input_ids,
                &target_mask,
                &attention_mask,
                1.0 / gradient_accumulation_steps as f64,
                accumulation_index + 1 == gradient_accumulation_steps,
            )? / gradient_accumulation_steps as f64;
        }
        if step == 0 {
            initial_loss = loss_value;
        }
        final_loss = loss_value;
        if step % 10 == 0 || step == max_steps - 1 {
            info!("step {step}/{max_steps} loss={loss_value:.6}");
        }
    }

    // Export every positional native slot, then let the artifact mapper omit
    // inactive slots and assign stable projection-aware tensor names.
    let mut exported = Vec::with_capacity(ctx.lora_count() as usize);
    for index in 0..ctx.lora_count() {
        let a = ctx
            .get_lora_a(index)
            .with_context(|| format!("native LoRA slot {index} is missing A"))?;
        let b = ctx
            .get_lora_b(index)
            .with_context(|| format!("native LoRA slot {index} is missing B"))?;
        exported.push((a, b));
    }
    let artifact = Qwen36AdapterArtifact::from_native_exports(
        &config.model.name,
        &config.model.architecture,
        Some(&model_path),
        &runtime_config,
        &lora_config,
        exported,
    )?;
    let trainable_params = artifact.tensors.len();
    let adapter_path = artifact.save(&run_paths.root)?;
    info!("saved adapter to {}", adapter_path.display());

    Ok(Qwen36LoraSftSummary {
        adapter_output: adapter_path.to_string_lossy().to_string(),
        initial_loss,
        final_loss,
        trainable_params,
    })
}
