use std::collections::{BTreeMap, HashSet};

use anyhow::{Context, Result, bail};
use tch::{Device, Kind, Tensor, no_grad};
use tracing::info;

use rustrain_core::runtime::{Config, RunPaths};
use rustrain_train::metrics::{gpu_memory_allocated_mb, memory_rss_mb};

use crate::model::*;

pub fn train_v4_session_single_from_config(config: &Config, _run_paths: &RunPaths) -> Result<()> {
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("V4 session trainer requires model.model_path")?;
    let model_path = resolve_v4_model_path(model_path)?;

    let runtime_config = read_v4_config(&model_path.join("config.json"))?;
    info!(
        layers = runtime_config.num_hidden_layers,
        "V4 config loaded"
    );

    let trainable_layers = config
        .model
        .trainable_layers
        .clone()
        .unwrap_or_else(|| vec![0]);

    let mut needed: HashSet<String> = HashSet::new();
    needed.insert("embed.weight".to_string());
    needed.insert("norm.weight".to_string());
    needed.insert("head.weight".to_string());
    if !runtime_config.tie_word_embeddings {
        // lm_head.weight skipped (V4 uses tied embeddings)
    }
    let names = v4_trainable_tensors_for_layer(trainable_layers[0], &runtime_config);
    needed.extend(names);

    let weights = load_v4_weights(&model_path, &needed)?;
    info!(tensors = weights.len(), "weights loaded");

    let compute_kind = Kind::Float;
    let device = Device::Cuda(0);

    let mut weights_gpu: BTreeMap<String, Tensor> = weights
        .into_iter()
        .map(|(name, t)| (name, t.to_device(device).to_kind(compute_kind)))
        .collect();

    // Set trainable
    let trainable_names = v4_trainable_tensors_for_layer(trainable_layers[0], &runtime_config);
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
        .to_device(device);

    let lr = config.train.learning_rate as f64;
    let mut initial_loss = 0.0_f64;

    for step in 0..config.train.max_steps {
        let loss = v4_causal_lm_loss_selective(
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

    let final_loss =
        v4_causal_lm_loss_selective(&input_ids, &weights_gpu, &runtime_config, &trainable_layers)?
            .double_value(&[]);

    info!(initial_loss, final_loss, "V4 training complete");
    println!("initial_loss: {:.9}", initial_loss);
    println!("final_loss: {:.9}", final_loss);
    println!("trainable_tensors: {}", trainable_params.len());
    Ok(())
}

use crate::lora::*;
use crate::sft::*;

pub struct V4LoraSftSummary {
    pub adapter_output: String,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub trainable_params: usize,
}

pub fn train_v4_lora_sft_from_config(
    config: &Config,
    run_paths: &RunPaths,
) -> Result<V4LoraSftSummary> {
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("V4 LoRA SFT requires model.model_path")?;
    let model_path = resolve_v4_model_path(model_path)?;

    let runtime_config = read_v4_config(&model_path.join("config.json"))?;
    info!(
        layers = runtime_config.num_hidden_layers,
        "V4 config loaded"
    );

    let trainable_layers = config
        .model
        .trainable_layers
        .clone()
        .unwrap_or_else(|| vec![0]);

    let lora_config_raw = config
        .lora
        .as_ref()
        .context("V4 LoRA SFT requires [lora] config section")?;
    let target_modules: Vec<V4LoraTargetModule> = lora_config_raw
        .target_modules
        .iter()
        .map(|s| V4LoraTargetModule::from_name(s))
        .collect::<Result<Vec<_>>>()?;
    let lora_config = V4LoraConfig {
        rank: lora_config_raw.rank,
        alpha: lora_config_raw.alpha as i64,
        target_layers: trainable_layers.clone(),
        target_modules,
    };

    let mut needed: HashSet<String> = HashSet::new();
    needed.insert("embed.weight".to_string());
    needed.insert("norm.weight".to_string());
    needed.insert("head.weight".to_string());
    if !runtime_config.tie_word_embeddings {
        // lm_head.weight skipped (V4 uses tied embeddings)
    }
    let names = v4_trainable_tensors_for_layer(trainable_layers[0], &runtime_config);
    needed.extend(names);

    let weights = load_v4_weights(&model_path, &needed)?;
    info!(tensors = weights.len(), "weights loaded");

    let compute_kind = Kind::Float;
    let device = Device::Cuda(0);

    let weights_gpu: BTreeMap<String, Tensor> = weights
        .into_iter()
        .map(|(name, t)| (name, t.to_device(device).to_kind(compute_kind)))
        .collect();

    let mut registry = if let Some(resume_path) = config.train.resume_from.as_ref() {
        info!(resume_from = %resume_path.display(), "resuming V4 LoRA SFT from checkpoint");
        V4LoraRegistry::load(resume_path, lora_config.clone())?
    } else {
        V4LoraRegistry::new(&weights_gpu, lora_config.clone(), device)?
    };

    let trainable_count = registry.var_store.trainable_variables().len();
    info!(trainable_params = trainable_count, "LoRA adapters created");

    // SFT data
    let tokenizer = tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
    let train_dataset = V4SftDataset::synthetic(&tokenizer)?;
    let train_batch = train_dataset.padded_batch(0, 1, device);

    let lr = config.train.learning_rate as f64;
    let beta1 = config.train.adam_beta1 as f64;
    let beta2 = config.train.adam_beta2 as f64;
    let eps = config.train.adam_eps as f64;
    let trainable_vars = registry.var_store.trainable_variables();
    let mut adam_m: Vec<Tensor> = trainable_vars.iter().map(Tensor::zeros_like).collect();
    let mut adam_v: Vec<Tensor> = trainable_vars.iter().map(Tensor::zeros_like).collect();

    let mut initial_loss = 0.0_f64;

    for step in 0..config.train.max_steps {
        let loss = v4_lora_sft_loss(
            &train_batch.input_ids,
            &train_batch.target_mask,
            &weights_gpu,
            &runtime_config,
            &trainable_layers,
            &registry,
        )?;
        let loss_val = loss.double_value(&[]);
        if step == 0 {
            initial_loss = loss_val;
        }
        info!(step = step + 1, loss = loss_val, "V4 LoRA SFT train step");
        loss.backward();

        let mut current_vars = registry.var_store.trainable_variables();
        for (i, var) in current_vars.iter_mut().enumerate() {
            let grad = var.grad();
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

    let final_loss = v4_lora_sft_loss(
        &train_batch.input_ids,
        &train_batch.target_mask,
        &weights_gpu,
        &runtime_config,
        &trainable_layers,
        &registry,
    )?
    .double_value(&[]);

    info!(initial_loss, final_loss, "V4 LoRA SFT complete");

    let adapter_output = run_paths.checkpoints.join("v4-lora-adapter.safetensors");
    registry.save(&adapter_output)?;
    info!(adapter = %adapter_output.display(), "adapter saved");

    let manifest = V4LoraManifest {
        format: "rustrain.deepseek_v4_lora_sft.v1".to_string(),
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
    let manifest_output = lora_manifest_path(&adapter_output);
    write_lora_manifest(&manifest_output, &manifest)?;

    println!("initial_loss: {:.9}", initial_loss);
    println!("final_loss: {:.9}", final_loss);
    println!("trainable_params: {}", trainable_count);
    println!("adapter_checkpoint: {}", adapter_output.display());

    if final_loss >= initial_loss {
        bail!("V4 LoRA SFT failed to reduce loss: initial={initial_loss}, final={final_loss}");
    }

    Ok(V4LoraSftSummary {
        adapter_output: adapter_output.display().to_string(),
        initial_loss,
        final_loss,
        trainable_params: trainable_count,
    })
}
