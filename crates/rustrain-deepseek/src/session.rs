use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail};
use tch::{Device, Kind, Tensor, no_grad};
use tracing::info;

use rustrain_core::runtime::{Config, RunPaths};
use rustrain_train::metrics::{gpu_memory_allocated_mb, memory_rss_mb};

use crate::model::*;

pub fn train_deepseek_session_single_from_config(
    config: &Config,
    _run_paths: &RunPaths,
) -> Result<()> {
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("DeepSeek session trainer requires model.model_path")?;
    let model_path = resolve_deepseek_model_path(model_path)?;

    let runtime_config = read_deepseek_config(&model_path.join("config.json"))?;
    info!(
        layers = runtime_config.num_hidden_layers,
        "DeepSeek config loaded"
    );

    let trainable_layers = config
        .model
        .trainable_layers
        .clone()
        .unwrap_or_else(|| vec![0]);

    let trainable_names = deepseek_trainable_tensors(
        &trainable_layers,
        &runtime_config,
        !runtime_config.tie_word_embeddings,
    );

    let mut needed: HashSet<String> = HashSet::new();
    needed.insert("model.embed_tokens.weight".to_string());
    needed.insert("model.norm.weight".to_string());
    if !runtime_config.tie_word_embeddings {
        needed.insert("lm_head.weight".to_string());
    }
    needed.extend(trainable_names.iter().cloned());

    let weights = load_deepseek_weights(&model_path, &needed)?;
    info!(tensors = weights.len(), "weights loaded");

    let dtype = match config.train.dtype {
        rustrain_core::runtime::DType::Fp32 => DeepSeekComputeDType::Fp32,
        rustrain_core::runtime::DType::Bf16 => DeepSeekComputeDType::Bf16,
        _ => bail!("unsupported dtype"),
    };
    let compute_kind = dtype.kind();

    let mut weights_gpu: BTreeMap<String, Tensor> = weights
        .into_iter()
        .map(|(name, t)| (name, t.to_device(Device::Cuda(0))))
        .collect();

    let mut trainable_params: Vec<(String, Tensor)> = Vec::new();
    for name in &trainable_names {
        if let Some(t) = weights_gpu.get_mut(name) {
            let trainable = t
                .shallow_clone()
                .to_kind(compute_kind)
                .set_requires_grad(true);
            weights_gpu.insert(name.clone(), trainable.shallow_clone());
            trainable_params.push((name.clone(), trainable));
        }
    }

    info!(
        trainable_tensors = trainable_params.len(),
        "trainable parameters set"
    );

    let input_ids = Tensor::from_slice(&[1i64, 2, 3, 4, 5])
        .reshape([1, 5])
        .to_device(Device::Cuda(0));

    let lr = config.train.learning_rate as f64;
    let mut initial_loss = 0.0_f64;

    for step in 0..config.train.max_steps {
        let loss = deepseek_causal_lm_loss_selective(
            &input_ids,
            &weights_gpu,
            &runtime_config,
            &trainable_layers,
        )?;
        let loss_val = loss.double_value(&[]);
        if step == 0 {
            initial_loss = loss_val;
        }
        info!(step = step + 1, loss = loss_val, "train step");
        loss.backward();

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
    }

    let final_loss = deepseek_causal_lm_loss_selective(
        &input_ids,
        &weights_gpu,
        &runtime_config,
        &trainable_layers,
    )?
    .double_value(&[]);

    info!(initial_loss, final_loss, "DeepSeek training complete");
    println!("initial_loss: {:.9}", initial_loss);
    println!("final_loss: {:.9}", final_loss);
    println!("trainable_tensors: {}", trainable_params.len());
    if let Some(rss) = memory_rss_mb() {
        println!("memory_rss_mb: {:.2}", rss);
    }
    if let Some(gpu) = gpu_memory_allocated_mb() {
        println!("gpu_memory_allocated_mb: {:.2}", gpu);
    }

    if final_loss >= initial_loss {
        bail!(
            "DeepSeek training failed to reduce loss: initial={initial_loss}, final={final_loss}"
        );
    }
    Ok(())
}

pub fn load_deepseek_weights(
    model_dir: &Path,
    needed: &HashSet<String>,
) -> Result<BTreeMap<String, Tensor>> {
    // DeepSeek-V3 uses FP8 safetensors which tch-rs can't read.
    // Use Python to convert needed shards to bf16.
    let script = format!(
        r#"
import json, sys, torch, os
from safetensors import safe_open
from safetensors.torch import save_file

model_dir = {dir:?}
needed = set(sys.argv[1:])
out = "/tmp/deepseek_bf16_converted.safetensors"

idx = os.path.join(model_dir, "model.safetensors.index.json")
single = os.path.join(model_dir, "model.safetensors")

tensors = {{}}
if os.path.exists(single):
    with safe_open(single, framework="pt") as f:
        for k in f.keys():
            if k in needed:
                t = f.get_tensor(k)
                if t.dtype == torch.float8_e4m3fn: t = t.to(torch.bfloat16)
                tensors[k] = t.cpu()
elif os.path.exists(idx):
    with open(idx) as f: wm = json.load(f)["weight_map"]
    shards = set(wm[n] for n in needed if n in wm)
    for s in sorted(shards):
        with safe_open(os.path.join(model_dir, s), framework="pt") as f:
            for k in f.keys():
                if k in needed:
                    t = f.get_tensor(k)
                    if t.dtype == torch.float8_e4m3fn: t = t.to(torch.bfloat16)
                    tensors[k] = t.cpu()
else:
    sys.exit(1)

save_file(tensors, out)
print(out)
"#,
        dir = model_dir.display()
    );

    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(&script)
        .args(needed.iter())
        .output()?;

    if !output.status.success() {
        bail!(
            "FP8 conversion failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    info!(path = %path, "FP8→bf16 conversion complete");

    let tensors = Tensor::read_safetensors(Path::new(&path))?;
    Ok(tensors.into_iter().collect())
}

// ── LoRA SFT ─────────────────────────────────────────────────

use crate::lora::*;
use crate::sft::*;
use rustrain_core::runtime::LrScheduler;

pub struct DeepSeekLoraSftSummary {
    pub adapter_output: String,
    pub manifest_output: String,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub steps: usize,
    pub trainable_params: usize,
}

pub fn train_deepseek_lora_sft_from_config(
    config: &Config,
    run_paths: &RunPaths,
) -> Result<DeepSeekLoraSftSummary> {
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("DeepSeek LoRA SFT requires model.model_path")?;
    let model_path = resolve_deepseek_model_path(model_path)?;

    let runtime_config = read_deepseek_config(&model_path.join("config.json"))?;
    info!(
        layers = runtime_config.num_hidden_layers,
        "DeepSeek config loaded"
    );

    let trainable_layers = config
        .model
        .trainable_layers
        .clone()
        .unwrap_or_else(|| vec![0]);

    // Build LoRA config from TOML [lora] section
    let lora_config_raw = config
        .lora
        .as_ref()
        .context("DeepSeek LoRA SFT requires [lora] config section")?;
    let target_modules: Vec<DeepSeekLoraTargetModule> = lora_config_raw
        .target_modules
        .iter()
        .map(|s| DeepSeekLoraTargetModule::from_name(s))
        .collect::<Result<Vec<_>>>()?;
    let lora_config = DeepSeekLoraConfig {
        rank: lora_config_raw.rank,
        alpha: lora_config_raw.alpha as i64,
        target_layers: trainable_layers.clone(),
        target_modules,
    };

    // Determine what weights we need: LoRA targets + embed + norm + lm_head
    let mut needed: HashSet<String> = HashSet::new();
    needed.insert("model.embed_tokens.weight".to_string());
    needed.insert("model.norm.weight".to_string());
    if !runtime_config.tie_word_embeddings {
        needed.insert("lm_head.weight".to_string());
    }
    for &layer in &trainable_layers {
        let names = deepseek_trainable_tensors_for_layer(layer, &runtime_config);
        needed.extend(names.iter().cloned());
    }

    // Load weights (FP8→bf16 conversion)
    let weights = load_deepseek_weights(&model_path, &needed)?;
    info!(tensors = weights.len(), "weights loaded");

    // Determine compute dtype
    let dtype = match config.train.dtype {
        rustrain_core::runtime::DType::Fp32 => DeepSeekComputeDType::Fp32,
        rustrain_core::runtime::DType::Bf16 => DeepSeekComputeDType::Bf16,
        _ => bail!("unsupported dtype"),
    };
    let compute_kind = dtype.kind();
    let device = Device::Cuda(0);

    // Move weights to GPU
    let mut weights_gpu: BTreeMap<String, Tensor> = weights
        .into_iter()
        .map(|(name, t)| (name, t.to_device(device)))
        .collect();

    // Convert all weights to compute dtype
    for t in weights_gpu.values_mut() {
        *t = t.to_kind(compute_kind);
    }

    // Create LoRA registry (or load from checkpoint if resume_from is set)
    let mut registry = if let Some(resume_path) = config.train.resume_from.as_ref() {
        info!(resume_from = %resume_path.display(), "resuming LoRA SFT from checkpoint");
        // resume_path is the adapter safetensors file
        DeepSeekLoraRegistry::load(resume_path, lora_config.clone())?
    } else {
        DeepSeekLoraRegistry::new(&weights_gpu, lora_config.clone(), device)?
    };
    let trainable_count = registry.var_store.trainable_variables().len();
    info!(trainable_params = trainable_count, "LoRA adapters created");

    // Store original base weights (no requires_grad)
    let original_weights: BTreeMap<String, Tensor> = weights_gpu
        .iter()
        .map(|(k, v)| (k.clone(), v.shallow_clone()))
        .collect();

    // Explicitly set requires_grad on LoRA params

    // Create training input (simple 5-token sequence for verification)
    let input_ids = Tensor::from_slice(&[1i64, 2, 3, 4, 5])
        .reshape([1, 5])
        .to_device(device);

    // AdamW optimizer (manual, same as Qwen3)
    let lr = config.train.learning_rate as f64;
    let beta1 = config.train.adam_beta1 as f64;
    let beta2 = config.train.adam_beta2 as f64;
    let eps = config.train.adam_eps as f64;
    let trainable_vars = registry.var_store.trainable_variables();
    let mut adam_m: Vec<Tensor> = trainable_vars.iter().map(Tensor::zeros_like).collect();
    let mut adam_v: Vec<Tensor> = trainable_vars.iter().map(Tensor::zeros_like).collect();

    let mut initial_loss = 0.0_f64;

    for step in 0..config.train.max_steps {
        // Forward (LoRA applied inline via lora_weight)
        let loss = deepseek_causal_lm_loss_lora(
            &input_ids,
            &original_weights,
            &runtime_config,
            &trainable_layers,
            &registry,
        )?;
        let loss_val = loss.double_value(&[]);
        if step == 0 {
            initial_loss = loss_val;
        }
        info!(step = step + 1, loss = loss_val, "LoRA SFT train step");

        // Backward
        loss.backward();

        // AdamW update on VarStore trainable variables
        let mut current_vars = registry.var_store.trainable_variables();
        for (i, var) in current_vars.iter_mut().enumerate() {
            let grad = var.grad();
            if grad.defined() {
                let g = grad.to_kind(Kind::Float); // always compute AdamW in fp32
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

    // Final loss
    let final_loss = deepseek_causal_lm_loss_lora(
        &input_ids,
        &original_weights,
        &runtime_config,
        &trainable_layers,
        &registry,
    )?
    .double_value(&[]);

    info!(initial_loss, final_loss, "DeepSeek LoRA SFT complete");

    // Save LoRA adapter
    let adapter_output = run_paths
        .checkpoints
        .join("deepseek-lora-adapter.safetensors");
    registry.save(&adapter_output)?;
    info!(adapter = %adapter_output.display(), "LoRA adapter saved");

    // Save manifest
    let manifest_output = lora_manifest_path(&adapter_output);
    let manifest = DeepSeekLoraManifest {
        format: "rustrain.deepseek_lora_sft.v1".to_string(),
        base_model_path: model_path.display().to_string(),
        adapter_safetensors: adapter_output.display().to_string(),
        rank: lora_config.rank,
        alpha: lora_config.alpha,
        target_layers: lora_config.target_layers.clone(),
        target_modules: lora_config
            .target_modules
            .iter()
            .map(|m| m.weight_suffix().to_string())
            .collect(),
        steps: config.train.max_steps as usize,
        initial_loss,
        final_loss,
    };
    write_lora_manifest(&manifest_output, &manifest)?;
    info!(manifest = %manifest_output.display(), "manifest saved");

    // Verify: reload adapter
    let reloaded = DeepSeekLoraRegistry::load(&adapter_output, lora_config)?;
    info!(reloaded_params = reloaded.param_count(), "adapter reloaded");

    println!("initial_loss: {:.9}", initial_loss);
    println!("final_loss: {:.9}", final_loss);
    println!("trainable_params: {}", trainable_count);
    println!("adapter_checkpoint: {}", adapter_output.display());

    if final_loss >= initial_loss {
        bail!(
            "DeepSeek LoRA SFT failed to reduce loss: initial={initial_loss}, final={final_loss}"
        );
    }

    Ok(DeepSeekLoraSftSummary {
        adapter_output: adapter_output.display().to_string(),
        manifest_output: manifest_output.display().to_string(),
        initial_loss,
        final_loss,
        steps: config.train.max_steps as usize,
        trainable_params: trainable_count,
    })
}
