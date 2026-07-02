//! V4 Flash EP-distributed LoRA SFT training — C++ all-in-C++ path.
//!
//! All compute (forward, loss, backward, Adam) happens in C++ via a single
//! `v4_train_step()` FFI call per training step. Rust handles weight loading,
//! SFT data, NCCL gradient sync, and adapter save.

use std::collections::{BTreeMap, HashSet};

use anyhow::{bail, Context, Result};
use tch::{no_grad, Device, Kind, Tensor};
use tracing::info;

use crate::ep::V4EpShard;
use crate::lora::*;
use crate::model::*;
use crate::session::V4LoraSftSummary;
use crate::sft::*;
use crate::v4_kernel::V4CppTrainingContext;
use rustrain_checkpoint::safetensors::tensor;
use rustrain_nccl::nccl::{self as nccl_smoke, NcclPersistentComm};

fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

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
        rank, world_size, local_rank,
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
    if rank > 0 {
        info!(rank, delay_secs = rank * 40, "waiting before weight loading (staggered)");
        std::thread::sleep(std::time::Duration::from_secs((rank * 40) as u64));
    }

    // ── Build needed weight set ──
    let n_layers = runtime_config.num_hidden_layers;
    let mut needed: HashSet<String> = HashSet::new();
    needed.insert("embed.weight".to_string());
    needed.insert("norm.weight".to_string());
    if !runtime_config.tie_word_embeddings {
        needed.insert("head.weight".to_string());
    }

    for layer in 0..n_layers {
        let p = format!("layers.{layer}");
        needed.insert(format!("{p}.attn_norm.weight"));
        needed.insert(format!("{p}.ffn_norm.weight"));
        for suffix in &[
            "wq_a.weight", "wq_b.weight", "wkv.weight",
            "wo_a.weight", "wo_b.weight",
            "q_norm.weight", "kv_norm.weight", "attn_sink",
        ] {
            needed.insert(format!("{p}.attn.{suffix}"));
        }
        needed.insert(format!("{p}.ffn.gate.weight"));
        for suffix in &["w1.weight", "w2.weight", "w3.weight"] {
            needed.insert(format!("{p}.ffn.shared_experts.{suffix}"));
        }
        needed.extend(crate::hc::HcWeights::weight_names(layer));
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

    // Move to device — FP8 weights with .scale_f stay as FP8, others to bf16
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
                    t.to_kind(compute_kind)
                };
                (name, processed)
            })
            .collect()
    };
    info!(rank, tensors_on_gpu = weights_gpu.len(), "weights on GPU");

    // ── Barrier: wait for all ranks ──
    let barrier_dir = run_paths.root.join("barrier");
    std::fs::create_dir_all(&barrier_dir)?;
    let ready_file = barrier_dir.join(format!("rank_{rank}.ready"));
    std::fs::write(&ready_file, b"ready")?;
    info!(rank, "waiting at barrier for all ranks to load weights");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
    loop {
        let ready_count = std::fs::read_dir(&barrier_dir)
            .map(|d| d.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with("rank_"))
                .count())
            .unwrap_or(0);
        if ready_count >= world_size { break; }
        if std::time::Instant::now() > deadline {
            bail!("barrier timeout: only {ready_count}/{world_size} ranks ready");
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    info!(rank, "all ranks ready, starting training");

    // ── C++ training context ──
    if !crate::v4_kernel::kernels_available() {
        bail!("C++ kernels (libv4_flash_kernels.so) not found — required for training");
    }

    let lora_scaling = lora_config.alpha as f64 / lora_config.rank as f64;
    let ctx = V4CppTrainingContext::new(
        &weights_gpu, &runtime_config, compute_kind,
        config.train.learning_rate as f64,
        config.train.adam_beta1 as f64,
        config.train.adam_beta2 as f64,
        config.train.adam_eps as f64,
        lora_scaling,
        Some(&ep_shard),
        true,  // has_lora
    )?;
    info!(rank, lora_params = ctx.lora_count(), "C++ TrainingContext created");

    // Gradient checkpointing
    if let Ok(gs) = std::env::var("V4_CHECKPOINT_GROUP") {
        let group_size: i64 = gs.parse().unwrap_or(4);
        ctx.set_checkpoint(true, group_size);
        info!(rank, "gradient checkpointing ON (group_size={group_size})");
    }

    // ── SFT data ──
    let tokenizer = tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
    let sft_jsonl = std::path::Path::new("data/sft/deepseek_test.jsonl");
    let train_dataset = if sft_jsonl.exists() {
        info!(rank, path = %sft_jsonl.display(), "loading real SFT data");
        V4SftDataset::from_jsonl_simple(sft_jsonl, &tokenizer)?
    } else {
        info!(rank, "no SFT JSONL found, using synthetic data");
        V4SftDataset::synthetic(&tokenizer)?
    };

    let batch_size = config.train.micro_batch_size;
    let max_steps = config.train.max_steps as usize;

    let mut initial_loss = 0.0_f64;
    let mut final_loss = 0.0_f64;

    for step in 0..max_steps {
        let data_start = (step * batch_size) % train_dataset.samples.len();
        let sft_batch = train_dataset.padded_batch(data_start, batch_size, device);

        // Pad input to config seq_len
        let target_seq = config.model.seq_len as i64;
        let actual_seq = sft_batch.input_ids.size()[1];
        let (input_ids, target_mask) = if actual_seq < target_seq {
            let pad_token = train_dataset.pad_token_id;
            let pad_ids = Tensor::full([1, target_seq - actual_seq], pad_token, (Kind::Int64, device));
            let input_ids = Tensor::cat(&[&sft_batch.input_ids, &pad_ids], 1);
            let pad_mask = Tensor::zeros([1, target_seq - actual_seq], (Kind::Int64, device));
            let target_mask = Tensor::cat(&[&sft_batch.target_mask, &pad_mask], 1);
            (input_ids, target_mask)
        } else {
            (sft_batch.input_ids.shallow_clone(), sft_batch.target_mask.shallow_clone())
        };

        let loss_value = ctx.train_step(&input_ids, &target_mask)?;
        if step == 0 { initial_loss = loss_value; }
        final_loss = loss_value;
        if step % 10 == 0 || step == max_steps - 1 {
            info!(rank, step = step + 1, max_steps, loss = format!("{loss_value:.6}"), "V4 C++ train step");
        }
    }

    // ── Save LoRA adapter ──
    let adapter_path = run_paths.checkpoints.join("v4-lora-adapter-ep.safetensors");
    {
        let mut named_tensors: BTreeMap<String, Tensor> = BTreeMap::new();
        for i in 0..ctx.lora_count() {
            if let (Some(a), Some(b)) = (ctx.get_lora_a(i), ctx.get_lora_b(i)) {
                named_tensors.insert(format!("lora_a_{i}"), a.to_kind(Kind::Float).to_device(tch::Device::Cpu));
                named_tensors.insert(format!("lora_b_{i}"), b.to_kind(Kind::Float).to_device(tch::Device::Cpu));
            }
        }
        use std::io::Write;
        let mut tensors_data = Vec::new();
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        for (name, t) in &named_tensors {
            let t = t.contiguous().to_kind(Kind::Float);
            let shape: Vec<i64> = t.size().iter().copied().collect();
            let data: Vec<f32> = Vec::<f32>::try_from(&t.reshape([-1]))?;
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            header.insert(name.clone(), serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[offset,offset+bytes.len() as u64]}));
            offset += bytes.len() as u64;
            tensors_data.push(bytes);
        }
        let header_str = serde_json::to_string(&serde_json::Value::Object(header))?;
        let file = std::fs::File::create(&adapter_path)
            .with_context(|| format!("create {}", adapter_path.display()))?;
        let mut writer = std::io::BufWriter::new(file);
        writer.write_all(&(header_str.len() as u64).to_le_bytes())?;
        writer.write_all(header_str.as_bytes())?;
        for data in &tensors_data { writer.write_all(data)?; }
        info!(rank, adapter = %adapter_path.display(), "adapter saved");
    }

    info!(rank, initial_loss, final_loss, "V4 LoRA SFT EP complete");

    Ok(V4LoraSftSummary {
        adapter_output: adapter_path.display().to_string(),
        initial_loss,
        final_loss,
        trainable_params: ctx.lora_count() as usize * 2,
    })
}
