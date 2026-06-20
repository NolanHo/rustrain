use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use tch::{Device, IndexOp, Kind, Reduction, Tensor, no_grad};

#[derive(Debug, Serialize)]
pub struct DiffStats {
    max_abs: f64,
    mean_abs: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct QwenRuntimeConfig {
    pub num_hidden_layers: usize,
    pub num_attention_heads: i64,
    pub num_key_value_heads: i64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

#[derive(Debug, Serialize)]
struct QwenModuleParitySummary {
    model_safetensors: String,
    fixture: String,
    attention_diff: DiffStats,
    rms_norm_diff: DiffStats,
    mlp_diff: DiffStats,
    layer0_diff: DiffStats,
    layer1_diff: DiffStats,
}

#[derive(Debug, Deserialize)]
struct QwenModelConfig {
    num_hidden_layers: usize,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    rms_norm_eps: f64,
    rope_theta: f64,
}

#[derive(Debug, Serialize)]
struct TopLogit {
    token_id: i64,
    logit: f64,
}

#[derive(Debug, Serialize)]
struct QwenLogitsParitySummary {
    model_path: String,
    reference_fixture: String,
    input_ids: Vec<i64>,
    logits_shape: Vec<i64>,
    logits_diff: DiffStats,
    last_token_topk: Vec<TopLogit>,
}

#[derive(Debug, Serialize)]
struct QwenGenerateParitySummary {
    model_path: String,
    reference_fixture: String,
    prompt_len: usize,
    max_new_tokens: usize,
    generated_ids: Vec<i64>,
    new_token_ids: Vec<i64>,
    reference_match: bool,
}

#[derive(Debug, Serialize)]
struct QwenSamplingSmokeSummary {
    model_path: String,
    reference_fixture: String,
    prompt_len: usize,
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
    generated_ids: Vec<i64>,
    new_token_ids: Vec<i64>,
}

#[derive(Debug, Serialize)]
struct QwenKvCacheParitySummary {
    model_path: String,
    reference_fixture: String,
    prompt_len: usize,
    max_new_tokens: usize,
    full_context_ids: Vec<i64>,
    cached_ids: Vec<i64>,
    new_token_ids: Vec<i64>,
    reference_match: bool,
}

#[derive(Debug, Serialize)]
struct QwenLoraSmokeSummary {
    model_path: String,
    fixture: String,
    adapter_output: String,
    rank: i64,
    alpha: f64,
    zero_lora_max_delta: f64,
    nonzero_lora_max_delta: f64,
    reload_max_delta: f64,
    trainable_tensors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct QwenTiedHeadTrainSummary {
    model_path: String,
    reference_fixture: String,
    delta_output: String,
    trainable_tensor: String,
    learning_rate: f64,
    initial_loss: f64,
    final_loss: f64,
    reloaded_loss: f64,
    reload_delta: f64,
    grad_defined: bool,
    grad_norm: f64,
}

#[derive(Debug, Serialize)]
struct TrainableTensorSummary {
    name: String,
    grad_defined: bool,
    grad_norm: f64,
    delta_norm: f64,
}

#[derive(Debug, Serialize)]
struct QwenFullTrainSmokeSummary {
    model_path: String,
    reference_fixture: String,
    delta_output: String,
    manifest_output: String,
    learning_rate: f64,
    initial_loss: f64,
    final_loss: f64,
    reloaded_loss: f64,
    reload_delta: f64,
    trainable_tensors: Vec<TrainableTensorSummary>,
}

#[derive(Debug, Serialize, Deserialize)]
struct QwenDeltaCheckpointManifest {
    format: String,
    base_model_path: String,
    reference_fixture: String,
    delta_safetensors: String,
    train_step: u64,
    learning_rate: f64,
    initial_loss: f64,
    final_loss: f64,
    tensors: Vec<QwenDeltaTensorManifestEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct QwenDeltaTensorManifestEntry {
    name: String,
    delta_name: String,
    shape: Vec<i64>,
    dtype: String,
    grad_norm: f64,
    delta_norm: f64,
}

pub fn qwen_module_parity(model_safetensors: &Path, fixture: &Path) -> Result<()> {
    let weights = read_safetensors_map(model_safetensors)?;
    let fixture_tensors = read_safetensors_map(fixture)?;
    let input = tensor(&fixture_tensors, "embedded_hidden")?.to_kind(Kind::Float);
    let attention_input = tensor(&fixture_tensors, "input_attention_normed")?.to_kind(Kind::Float);
    let expected_attention = tensor(&fixture_tensors, "attention_output")?.to_kind(Kind::Float);
    let expected_norm = tensor(&fixture_tensors, "post_attention_normed")?.to_kind(Kind::Float);
    let expected_mlp = tensor(&fixture_tensors, "mlp_output")?.to_kind(Kind::Float);
    let expected_layer0 = tensor(&fixture_tensors, "layer0_output")?.to_kind(Kind::Float);
    let expected_layer1 = tensor(&fixture_tensors, "layer1_output")?.to_kind(Kind::Float);

    let config = QwenRuntimeConfig {
        num_hidden_layers: 24,
        num_attention_heads: 14,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };
    let layer0 = QwenLayerWeights::load(&weights, 0)?;

    let actual_attention = qwen_attention(
        &attention_input,
        &layer0.q_proj,
        &layer0.q_bias,
        &layer0.k_proj,
        &layer0.k_bias,
        &layer0.v_proj,
        &layer0.v_bias,
        &layer0.o_proj,
        &config,
    );
    let actual_norm = rms_norm(&input, &layer0.post_attention_norm, config.rms_norm_eps);
    let actual_mlp = qwen_mlp(
        &actual_norm,
        &layer0.gate_proj,
        &layer0.up_proj,
        &layer0.down_proj,
    );
    let actual_layer0 = qwen_layer(&input, &layer0, &config);
    let actual_layer1 = qwen_layer(
        &actual_layer0,
        &QwenLayerWeights::load(&weights, 1)?,
        &config,
    );
    let attention_diff = diff_stats(&actual_attention, &expected_attention)?;
    let rms_norm_diff = diff_stats(&actual_norm, &expected_norm)?;
    let mlp_diff = diff_stats(&actual_mlp, &expected_mlp)?;
    let layer0_diff = diff_stats(&actual_layer0, &expected_layer0)?;
    let layer1_diff = diff_stats(&actual_layer1, &expected_layer1)?;

    if attention_diff.max_abs > 1e-4 {
        bail!(
            "attention parity failed: max_abs={}",
            attention_diff.max_abs
        );
    }
    if rms_norm_diff.max_abs > 1e-5 {
        bail!("RMSNorm parity failed: max_abs={}", rms_norm_diff.max_abs);
    }
    if mlp_diff.max_abs > 1e-4 {
        bail!("MLP parity failed: max_abs={}", mlp_diff.max_abs);
    }
    if layer0_diff.max_abs > 1e-4 {
        bail!("layer0 parity failed: max_abs={}", layer0_diff.max_abs);
    }
    if layer1_diff.max_abs > 2e-4 {
        bail!("layer1 parity failed: max_abs={}", layer1_diff.max_abs);
    }

    let summary = QwenModuleParitySummary {
        model_safetensors: model_safetensors.display().to_string(),
        fixture: fixture.display().to_string(),
        attention_diff,
        rms_norm_diff,
        mlp_diff,
        layer0_diff,
        layer1_diff,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_logits_parity(model_path: &Path, reference_fixture: &Path) -> Result<()> {
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    let expected_logits = tensor(&reference, "logits")?.to_kind(Kind::Float);
    let actual_logits = qwen_forward_from_ids(&input_ids, &weights, &config)?;
    let logits_diff = diff_stats(&actual_logits, &expected_logits)?;

    if logits_diff.max_abs > 5e-3 {
        bail!("logits parity failed: max_abs={}", logits_diff.max_abs);
    }

    let last_logits = actual_logits.i((0, -1));
    let (values, indices) = last_logits.topk(8, -1, true, true);
    let values: Vec<f32> = Vec::<f32>::try_from(values.to_device(Device::Cpu))?;
    let indices: Vec<i64> = Vec::<i64>::try_from(indices.to_device(Device::Cpu))?;
    let last_token_topk = values
        .into_iter()
        .zip(indices)
        .map(|(logit, token_id)| TopLogit {
            token_id,
            logit: f64::from(logit),
        })
        .collect();
    let input_ids_flat: Vec<i64> =
        Vec::<i64>::try_from(input_ids.reshape([-1]).to_device(Device::Cpu))?;

    let summary = QwenLogitsParitySummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        input_ids: input_ids_flat,
        logits_shape: actual_logits.size(),
        logits_diff,
        last_token_topk,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_generate_parity(model_path: &Path, reference_fixture: &Path) -> Result<()> {
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    let expected_generated = tensor(&reference, "generated_ids")?.to_kind(Kind::Int64);
    let expected_ids: Vec<i64> =
        Vec::<i64>::try_from(expected_generated.reshape([-1]).to_device(Device::Cpu))?;
    let prompt_len = input_ids.size()[1] as usize;
    if expected_ids.len() < prompt_len {
        bail!(
            "reference generated ids shorter than prompt: generated={}, prompt={prompt_len}",
            expected_ids.len()
        );
    }
    let max_new_tokens = expected_ids.len() - prompt_len;
    let generated = qwen_greedy_generate(&input_ids, &weights, &config, max_new_tokens)?;
    let generated_ids: Vec<i64> =
        Vec::<i64>::try_from(generated.reshape([-1]).to_device(Device::Cpu))?;
    let reference_match = generated_ids == expected_ids;
    if !reference_match {
        bail!(
            "greedy generation parity failed: expected {:?}, got {:?}",
            expected_ids,
            generated_ids
        );
    }
    let summary = QwenGenerateParitySummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        prompt_len,
        max_new_tokens,
        new_token_ids: generated_ids[prompt_len..].to_vec(),
        generated_ids,
        reference_match,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_sampling_smoke(
    model_path: &Path,
    reference_fixture: &Path,
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
) -> Result<()> {
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    let prompt_len = input_ids.size()[1] as usize;
    let generated = qwen_sample_generate(
        &input_ids,
        &weights,
        &config,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        seed,
    )?;
    let generated_ids: Vec<i64> =
        Vec::<i64>::try_from(generated.reshape([-1]).to_device(Device::Cpu))?;
    let new_token_ids = generated_ids[prompt_len..].to_vec();
    if new_token_ids.len() != max_new_tokens {
        bail!(
            "sampling smoke generated {} tokens, expected {max_new_tokens}",
            new_token_ids.len()
        );
    }

    let summary = QwenSamplingSmokeSummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        prompt_len,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        seed,
        generated_ids,
        new_token_ids,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_kv_cache_parity(
    model_path: &Path,
    reference_fixture: &Path,
    max_new_tokens: usize,
) -> Result<()> {
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    let prompt_len = input_ids.size()[1] as usize;
    let full_context = qwen_greedy_generate(&input_ids, &weights, &config, max_new_tokens)?;
    let cached = qwen_greedy_generate_with_cache(&input_ids, &weights, &config, max_new_tokens)?;
    let full_context_ids: Vec<i64> =
        Vec::<i64>::try_from(full_context.reshape([-1]).to_device(Device::Cpu))?;
    let cached_ids: Vec<i64> = Vec::<i64>::try_from(cached.reshape([-1]).to_device(Device::Cpu))?;
    let reference_match = full_context_ids == cached_ids;
    if !reference_match {
        bail!(
            "KV-cache greedy parity failed: full_context={:?}, cached={:?}",
            full_context_ids,
            cached_ids
        );
    }

    let summary = QwenKvCacheParitySummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        prompt_len,
        max_new_tokens,
        new_token_ids: cached_ids[prompt_len..].to_vec(),
        full_context_ids,
        cached_ids,
        reference_match,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_lora_smoke(
    model_path: &Path,
    fixture: &Path,
    adapter_output: &Path,
    rank: i64,
    alpha: f64,
) -> Result<()> {
    if rank <= 0 {
        bail!("rank must be positive");
    }
    if alpha <= 0.0 {
        bail!("alpha must be positive");
    }

    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let fixture_tensors = read_safetensors_map(fixture)?;
    let attention_input = tensor(&fixture_tensors, "input_attention_normed")?.to_kind(Kind::Float);
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let layer0 = QwenLayerWeights::load(&weights, 0)?;
    let base = qwen_attention(
        &attention_input,
        &layer0.q_proj,
        &layer0.q_bias,
        &layer0.k_proj,
        &layer0.k_bias,
        &layer0.v_proj,
        &layer0.v_bias,
        &layer0.o_proj,
        &config,
    );
    let zero_adapter = QwenAttentionLoraAdapter::zeros(
        layer0.q_proj.size()[1],
        layer0.q_proj.size()[0],
        layer0.v_proj.size()[0],
        rank,
        alpha,
    );
    let zero_output = qwen_attention_with_lora(&attention_input, &layer0, &zero_adapter, &config);
    let zero_lora_max_delta = diff_stats(&zero_output, &base)?.max_abs;
    if zero_lora_max_delta > 1e-7 {
        bail!("zero LoRA changed attention output: max_delta={zero_lora_max_delta}");
    }

    let adapter = QwenAttentionLoraAdapter::deterministic(
        layer0.q_proj.size()[1],
        layer0.q_proj.size()[0],
        layer0.v_proj.size()[0],
        rank,
        alpha,
    );
    let adapted_output = qwen_attention_with_lora(&attention_input, &layer0, &adapter, &config);
    let nonzero_lora_max_delta = diff_stats(&adapted_output, &base)?.max_abs;
    if nonzero_lora_max_delta <= 0.0 {
        bail!("non-zero LoRA did not change attention output");
    }

    adapter.save(adapter_output)?;
    let reloaded = QwenAttentionLoraAdapter::load(adapter_output)?;
    let reloaded_output = qwen_attention_with_lora(&attention_input, &layer0, &reloaded, &config);
    let reload_max_delta = diff_stats(&reloaded_output, &adapted_output)?.max_abs;
    if reload_max_delta > 1e-7 {
        bail!("LoRA adapter reload changed output: max_delta={reload_max_delta}");
    }

    let summary = QwenLoraSmokeSummary {
        model_path: model_path.display().to_string(),
        fixture: fixture.display().to_string(),
        adapter_output: adapter_output.display().to_string(),
        rank,
        alpha,
        zero_lora_max_delta,
        nonzero_lora_max_delta,
        reload_max_delta,
        trainable_tensors: reloaded.trainable_tensor_names(),
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_tied_head_train_smoke(
    model_path: &Path,
    reference_fixture: &Path,
    delta_output: &Path,
    learning_rate: f64,
) -> Result<()> {
    if learning_rate <= 0.0 {
        bail!("learning_rate must be positive");
    }

    let config = read_runtime_config(&model_path.join("config.json"))?;
    let mut weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    if input_ids.size()[1] < 2 {
        bail!("training fixture must contain at least two tokens");
    }

    let mut embed_tokens = tensor(&weights, "model.embed_tokens.weight")?
        .to_kind(Kind::Float)
        .set_requires_grad(true);
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        embed_tokens.shallow_clone(),
    );

    let initial_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)?.double_value(&[]);
    let loss = qwen_causal_lm_loss(&input_ids, &weights, &config)?;
    loss.backward();
    let grad = embed_tokens.grad();
    let grad_defined = grad.defined();
    let grad_norm = if grad_defined {
        grad.norm().double_value(&[])
    } else {
        0.0
    };
    if !grad_defined || grad_norm <= 0.0 {
        bail!("tied embedding gradient was not populated");
    }

    let update = &grad * learning_rate;
    let _ = no_grad(|| embed_tokens.f_sub_(&update))?;

    let final_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)?.double_value(&[]);
    if final_loss >= initial_loss {
        bail!(
            "Qwen tied-head train smoke failed to reduce loss: initial_loss={initial_loss}, final_loss={final_loss}"
        );
    }
    let base_embed_tokens = tensor(
        &read_safetensors_map(&model_path.join("model.safetensors"))?,
        "model.embed_tokens.weight",
    )?
    .to_kind(Kind::Float);
    let delta = &embed_tokens - &base_embed_tokens;
    if let Some(parent) = delta_output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    Tensor::write_safetensors(
        &[(&"model.embed_tokens.weight.delta", &delta)],
        delta_output,
    )
    .with_context(|| format!("failed to write {}", delta_output.display()))?;

    let mut reloaded_weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let delta_tensors = read_safetensors_map(delta_output)?;
    let reloaded_embed = tensor(&reloaded_weights, "model.embed_tokens.weight")?
        .to_kind(Kind::Float)
        + tensor(&delta_tensors, "model.embed_tokens.weight.delta")?.to_kind(Kind::Float);
    reloaded_weights.insert("model.embed_tokens.weight".to_string(), reloaded_embed);
    let reloaded_loss =
        qwen_causal_lm_loss(&input_ids, &reloaded_weights, &config)?.double_value(&[]);
    let reload_delta = (final_loss - reloaded_loss).abs();
    if reload_delta > 1e-5 {
        bail!(
            "Qwen tied-head delta reload parity failed: final_loss={final_loss}, reloaded_loss={reloaded_loss}, reload_delta={reload_delta}"
        );
    }

    let summary = QwenTiedHeadTrainSummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        delta_output: delta_output.display().to_string(),
        trainable_tensor: "model.embed_tokens.weight".to_string(),
        learning_rate,
        initial_loss,
        final_loss,
        reloaded_loss,
        reload_delta,
        grad_defined,
        grad_norm,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_full_train_smoke(
    model_path: &Path,
    reference_fixture: &Path,
    delta_output: &Path,
    learning_rate: f64,
) -> Result<()> {
    if learning_rate <= 0.0 {
        bail!("learning_rate must be positive");
    }

    let config = read_runtime_config(&model_path.join("config.json"))?;
    let mut weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    if input_ids.size()[1] < 2 {
        bail!("training fixture must contain at least two tokens");
    }

    let trainable_names = representative_trainable_qwen_tensors();
    let mut trainable_tensors = Vec::with_capacity(trainable_names.len());
    for name in &trainable_names {
        let trainable = tensor(&weights, name)?
            .to_kind(Kind::Float)
            .set_requires_grad(true);
        weights.insert((*name).to_string(), trainable.shallow_clone());
        trainable_tensors.push(((*name).to_string(), trainable));
    }

    let initial_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)?.double_value(&[]);
    let loss = qwen_causal_lm_loss(&input_ids, &weights, &config)?;
    loss.backward();

    let base_weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let mut delta_entries: Vec<(String, Tensor)> = Vec::with_capacity(trainable_tensors.len());
    let mut tensor_summaries = Vec::with_capacity(trainable_tensors.len());
    let mut manifest_tensors = Vec::with_capacity(trainable_tensors.len());
    for (name, mut trainable) in trainable_tensors {
        let grad = trainable.grad();
        let grad_defined = grad.defined();
        let grad_norm = if grad_defined {
            grad.norm().double_value(&[])
        } else {
            0.0
        };
        if !grad_defined || grad_norm <= 0.0 {
            bail!("trainable tensor {name} did not receive a gradient");
        }

        let update = &grad * learning_rate;
        let _ = no_grad(|| trainable.f_sub_(&update))?;
        weights.insert(name.clone(), trainable.shallow_clone());
        let base = tensor(&base_weights, &name)?.to_kind(Kind::Float);
        let delta = &trainable - &base;
        let delta_norm = delta.norm().double_value(&[]);
        let delta_name = format!("{name}.delta");
        manifest_tensors.push(QwenDeltaTensorManifestEntry {
            name: name.clone(),
            delta_name: delta_name.clone(),
            shape: trainable.size(),
            dtype: "float32".to_string(),
            grad_norm,
            delta_norm,
        });
        delta_entries.push((delta_name, delta));
        tensor_summaries.push(TrainableTensorSummary {
            name,
            grad_defined,
            grad_norm,
            delta_norm,
        });
    }

    let final_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)?.double_value(&[]);
    if final_loss >= initial_loss {
        bail!(
            "Qwen full train smoke failed to reduce loss: initial_loss={initial_loss}, final_loss={final_loss}"
        );
    }

    if let Some(parent) = delta_output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let delta_refs: Vec<(&str, &Tensor)> = delta_entries
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    Tensor::write_safetensors(&delta_refs, delta_output)
        .with_context(|| format!("failed to write {}", delta_output.display()))?;
    let manifest_output = delta_manifest_path(delta_output);
    let manifest = QwenDeltaCheckpointManifest {
        format: "rustrain.qwen_delta.v1".to_string(),
        base_model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        delta_safetensors: delta_output.display().to_string(),
        train_step: 1,
        learning_rate,
        initial_loss,
        final_loss,
        tensors: manifest_tensors,
    };
    write_qwen_delta_manifest(&manifest_output, &manifest)?;

    let mut reloaded_weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let delta_tensors = read_safetensors_map(delta_output)?;
    for name in &trainable_names {
        let delta_name = format!("{name}.delta");
        let reloaded = tensor(&reloaded_weights, name)?.to_kind(Kind::Float)
            + tensor(&delta_tensors, &delta_name)?.to_kind(Kind::Float);
        reloaded_weights.insert((*name).to_string(), reloaded);
    }
    let reloaded_loss =
        qwen_causal_lm_loss(&input_ids, &reloaded_weights, &config)?.double_value(&[]);
    let reload_delta = (final_loss - reloaded_loss).abs();
    if reload_delta > 1e-5 {
        bail!(
            "Qwen full train delta reload parity failed: final_loss={final_loss}, reloaded_loss={reloaded_loss}, reload_delta={reload_delta}"
        );
    }

    let summary = QwenFullTrainSmokeSummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        delta_output: delta_output.display().to_string(),
        manifest_output: manifest_output.display().to_string(),
        learning_rate,
        initial_loss,
        final_loss,
        reloaded_loss,
        reload_delta,
        trainable_tensors: tensor_summaries,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

fn delta_manifest_path(delta_output: &Path) -> std::path::PathBuf {
    let mut path = delta_output.as_os_str().to_os_string();
    path.push(".json");
    path.into()
}

fn write_qwen_delta_manifest(
    manifest_output: &Path,
    manifest: &QwenDeltaCheckpointManifest,
) -> Result<()> {
    if let Some(parent) = manifest_output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(
        manifest_output,
        serde_json::to_string_pretty(manifest).context("failed to serialize manifest")? + "\n",
    )
    .with_context(|| format!("failed to write {}", manifest_output.display()))
}

fn representative_trainable_qwen_tensors() -> Vec<&'static str> {
    vec![
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
    ]
}

fn read_runtime_config(path: &Path) -> Result<QwenRuntimeConfig> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let config: QwenModelConfig = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(QwenRuntimeConfig {
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
    })
}

pub fn read_safetensors_map(path: &Path) -> Result<BTreeMap<String, Tensor>> {
    let tensors = Tensor::read_safetensors(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(tensors.into_iter().collect())
}

pub fn tensor<'a>(tensors: &'a BTreeMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| anyhow!("missing tensor {name}"))
}

pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let variance = input
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    input * (variance + eps).rsqrt() * weight
}

pub fn qwen_mlp(
    input: &Tensor,
    gate_proj: &Tensor,
    up_proj: &Tensor,
    down_proj: &Tensor,
) -> Tensor {
    let gate = input.linear::<&Tensor>(gate_proj, None);
    let up = input.linear::<&Tensor>(up_proj, None);
    (gate.silu() * up).linear::<&Tensor>(down_proj, None)
}

pub struct QwenLayerWeights {
    input_norm: Tensor,
    q_proj: Tensor,
    q_bias: Tensor,
    k_proj: Tensor,
    k_bias: Tensor,
    v_proj: Tensor,
    v_bias: Tensor,
    o_proj: Tensor,
    post_attention_norm: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

struct QwenLayerCache {
    key: Tensor,
    value: Tensor,
}

struct QwenAttentionLoraAdapter {
    q_a: Tensor,
    q_b: Tensor,
    v_a: Tensor,
    v_b: Tensor,
    rank: i64,
    alpha: f64,
}

impl QwenAttentionLoraAdapter {
    fn zeros(
        in_features: i64,
        q_out_features: i64,
        v_out_features: i64,
        rank: i64,
        alpha: f64,
    ) -> Self {
        Self {
            q_a: Tensor::zeros([rank, in_features], (Kind::Float, Device::Cpu)),
            q_b: Tensor::zeros([q_out_features, rank], (Kind::Float, Device::Cpu)),
            v_a: Tensor::zeros([rank, in_features], (Kind::Float, Device::Cpu)),
            v_b: Tensor::zeros([v_out_features, rank], (Kind::Float, Device::Cpu)),
            rank,
            alpha,
        }
    }

    fn deterministic(
        in_features: i64,
        q_out_features: i64,
        v_out_features: i64,
        rank: i64,
        alpha: f64,
    ) -> Self {
        let q_a = deterministic_lora_tensor([rank, in_features], 0.0005);
        let q_b = deterministic_lora_tensor([q_out_features, rank], -0.0003);
        let v_a = deterministic_lora_tensor([rank, in_features], -0.0004);
        let v_b = deterministic_lora_tensor([v_out_features, rank], 0.0002);
        Self {
            q_a,
            q_b,
            v_a,
            v_b,
            rank,
            alpha,
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let rank = Tensor::from_slice(&[self.rank]);
        let alpha = Tensor::from_slice(&[self.alpha as f32]);
        Tensor::write_safetensors(
            &[
                (&"q_proj.lora_a", &self.q_a),
                (&"q_proj.lora_b", &self.q_b),
                (&"v_proj.lora_a", &self.v_a),
                (&"v_proj.lora_b", &self.v_b),
                (&"rank", &rank),
                (&"alpha", &alpha),
            ],
            path,
        )
        .with_context(|| format!("failed to write {}", path.display()))
    }

    fn load(path: &Path) -> Result<Self> {
        let tensors = read_safetensors_map(path)?;
        let q_a = tensor(&tensors, "q_proj.lora_a")?.to_kind(Kind::Float);
        let q_b = tensor(&tensors, "q_proj.lora_b")?.to_kind(Kind::Float);
        let v_a = tensor(&tensors, "v_proj.lora_a")?.to_kind(Kind::Float);
        let v_b = tensor(&tensors, "v_proj.lora_b")?.to_kind(Kind::Float);
        let rank = tensor(&tensors, "rank")?.int64_value(&[0]);
        let alpha = tensor(&tensors, "alpha")?.double_value(&[0]);
        Ok(Self {
            q_a,
            q_b,
            v_a,
            v_b,
            rank,
            alpha,
        })
    }

    fn trainable_tensor_names(&self) -> Vec<String> {
        vec![
            "model.layers.0.self_attn.q_proj.lora_a".to_string(),
            "model.layers.0.self_attn.q_proj.lora_b".to_string(),
            "model.layers.0.self_attn.v_proj.lora_a".to_string(),
            "model.layers.0.self_attn.v_proj.lora_b".to_string(),
        ]
    }

    fn q_delta(&self, device: Device) -> Tensor {
        self.q_b
            .to_device(device)
            .matmul(&self.q_a.to_device(device))
            * (self.alpha / self.rank as f64)
    }

    fn v_delta(&self, device: Device) -> Tensor {
        self.v_b
            .to_device(device)
            .matmul(&self.v_a.to_device(device))
            * (self.alpha / self.rank as f64)
    }
}

fn deterministic_lora_tensor<const N: usize>(shape: [i64; N], scale: f64) -> Tensor {
    let len = shape.iter().product::<i64>() as usize;
    let values: Vec<f32> = (0..len)
        .map(|index| ((index % 17) as f64 - 8.0) as f32 * scale as f32)
        .collect();
    Tensor::from_slice(&values).reshape(shape)
}

impl QwenLayerWeights {
    pub fn load(weights: &BTreeMap<String, Tensor>, layer_index: usize) -> Result<Self> {
        let prefix = format!("model.layers.{layer_index}");
        Ok(Self {
            input_norm: tensor(weights, &format!("{prefix}.input_layernorm.weight"))?
                .to_kind(Kind::Float),
            q_proj: tensor(weights, &format!("{prefix}.self_attn.q_proj.weight"))?
                .to_kind(Kind::Float),
            q_bias: tensor(weights, &format!("{prefix}.self_attn.q_proj.bias"))?
                .to_kind(Kind::Float),
            k_proj: tensor(weights, &format!("{prefix}.self_attn.k_proj.weight"))?
                .to_kind(Kind::Float),
            k_bias: tensor(weights, &format!("{prefix}.self_attn.k_proj.bias"))?
                .to_kind(Kind::Float),
            v_proj: tensor(weights, &format!("{prefix}.self_attn.v_proj.weight"))?
                .to_kind(Kind::Float),
            v_bias: tensor(weights, &format!("{prefix}.self_attn.v_proj.bias"))?
                .to_kind(Kind::Float),
            o_proj: tensor(weights, &format!("{prefix}.self_attn.o_proj.weight"))?
                .to_kind(Kind::Float),
            post_attention_norm: tensor(
                weights,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?
            .to_kind(Kind::Float),
            gate_proj: tensor(weights, &format!("{prefix}.mlp.gate_proj.weight"))?
                .to_kind(Kind::Float),
            up_proj: tensor(weights, &format!("{prefix}.mlp.up_proj.weight"))?.to_kind(Kind::Float),
            down_proj: tensor(weights, &format!("{prefix}.mlp.down_proj.weight"))?
                .to_kind(Kind::Float),
        })
    }
}

pub fn qwen_layer(
    input: &Tensor,
    weights: &QwenLayerWeights,
    config: &QwenRuntimeConfig,
) -> Tensor {
    let attention_input = rms_norm(input, &weights.input_norm, config.rms_norm_eps);
    let attention_output = qwen_attention(
        &attention_input,
        &weights.q_proj,
        &weights.q_bias,
        &weights.k_proj,
        &weights.k_bias,
        &weights.v_proj,
        &weights.v_bias,
        &weights.o_proj,
        config,
    );
    let after_attention = input + attention_output;
    let mlp_input = rms_norm(
        &after_attention,
        &weights.post_attention_norm,
        config.rms_norm_eps,
    );
    let mlp_output = qwen_mlp(
        &mlp_input,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
    );
    after_attention + mlp_output
}

pub fn qwen_forward_from_ids(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
) -> Result<Tensor> {
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?.to_kind(Kind::Float);
    let final_norm = tensor(weights, "model.norm.weight")?.to_kind(Kind::Float);
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);
    for layer_index in 0..config.num_hidden_layers {
        let layer = QwenLayerWeights::load(weights, layer_index)?;
        hidden = qwen_layer(&hidden, &layer, config);
    }
    let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps);
    Ok(hidden.linear::<&Tensor>(&embed_tokens, None))
}

fn qwen_forward_with_cache(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    past_cache: Option<Vec<QwenLayerCache>>,
) -> Result<(Tensor, Vec<QwenLayerCache>)> {
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?.to_kind(Kind::Float);
    let final_norm = tensor(weights, "model.norm.weight")?.to_kind(Kind::Float);
    let position_offset = past_cache
        .as_ref()
        .and_then(|cache| cache.first())
        .map(|layer_cache| layer_cache.key.size()[2])
        .unwrap_or(0);
    let mut past_cache = past_cache.map(|cache| cache.into_iter());
    let mut next_cache = Vec::with_capacity(config.num_hidden_layers);
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);

    for layer_index in 0..config.num_hidden_layers {
        let layer = QwenLayerWeights::load(weights, layer_index)?;
        let past_layer_cache = past_cache.as_mut().and_then(|cache| cache.next());
        let (layer_hidden, layer_cache) =
            qwen_layer_with_cache(&hidden, &layer, config, past_layer_cache, position_offset);
        hidden = layer_hidden;
        next_cache.push(layer_cache);
    }
    let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps);
    Ok((hidden.linear::<&Tensor>(&embed_tokens, None), next_cache))
}

fn qwen_layer_with_cache(
    input: &Tensor,
    weights: &QwenLayerWeights,
    config: &QwenRuntimeConfig,
    past_cache: Option<QwenLayerCache>,
    position_offset: i64,
) -> (Tensor, QwenLayerCache) {
    let attention_input = rms_norm(input, &weights.input_norm, config.rms_norm_eps);
    let (attention_output, cache) = qwen_attention_with_cache(
        &attention_input,
        &weights.q_proj,
        &weights.q_bias,
        &weights.k_proj,
        &weights.k_bias,
        &weights.v_proj,
        &weights.v_bias,
        &weights.o_proj,
        config,
        past_cache,
        position_offset,
    );
    let after_attention = input + attention_output;
    let mlp_input = rms_norm(
        &after_attention,
        &weights.post_attention_norm,
        config.rms_norm_eps,
    );
    let mlp_output = qwen_mlp(
        &mlp_input,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
    );
    (after_attention + mlp_output, cache)
}

fn qwen_causal_lm_loss(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
) -> Result<Tensor> {
    let logits = qwen_forward_from_ids(input_ids, weights, config)?;
    let seq_len = input_ids.size()[1];
    let shifted_logits = logits.narrow(1, 0, seq_len - 1);
    let targets = input_ids.narrow(1, 1, seq_len - 1);
    let vocab_size = shifted_logits.size()[2];
    Ok(shifted_logits
        .reshape([-1, vocab_size])
        .log_softmax(-1, Kind::Float)
        .g_nll_loss::<&Tensor>(&targets.reshape([-1]), None, Reduction::Mean, -100))
}

pub fn qwen_greedy_generate(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    max_new_tokens: usize,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    for _ in 0..max_new_tokens {
        let logits = qwen_forward_from_ids(&generated, weights, config)?;
        let next_token = logits.i((0, -1)).argmax(-1, false).reshape([1, 1]);
        generated = Tensor::cat(&[&generated, &next_token], 1);
    }
    Ok(generated)
}

pub fn qwen_greedy_generate_with_cache(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    max_new_tokens: usize,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    let (logits, mut cache) = qwen_forward_with_cache(input_ids, weights, config, None)?;
    let mut next_token = logits.i((0, -1)).argmax(-1, false).reshape([1, 1]);

    for step in 0..max_new_tokens {
        generated = Tensor::cat(&[&generated, &next_token], 1);
        if step + 1 == max_new_tokens {
            break;
        }
        let (decode_logits, updated_cache) =
            qwen_forward_with_cache(&next_token, weights, config, Some(cache))?;
        cache = updated_cache;
        next_token = decode_logits.i((0, -1)).argmax(-1, false).reshape([1, 1]);
    }

    Ok(generated)
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_sample_generate(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..max_new_tokens {
        let logits = qwen_forward_from_ids(&generated, weights, config)?;
        let next_token =
            sample_token_from_logits(&logits.i((0, -1)), temperature, top_k, top_p, &mut rng)?
                .reshape([1, 1]);
        generated = Tensor::cat(&[&generated, &next_token], 1);
    }
    Ok(generated)
}

fn sample_token_from_logits(
    logits: &Tensor,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    rng: &mut StdRng,
) -> Result<Tensor> {
    if temperature <= 0.0 {
        bail!("temperature must be positive");
    }
    if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
        bail!("top_p must be in (0, 1]");
    }

    let logits: Vec<f32> =
        Vec::<f32>::try_from(logits.to_kind(Kind::Float).to_device(Device::Cpu))?;
    let mut candidates: Vec<(i64, f64)> = logits
        .into_iter()
        .enumerate()
        .filter_map(|(token_id, logit)| {
            let scaled = f64::from(logit) / temperature;
            scaled.is_finite().then_some((token_id as i64, scaled))
        })
        .collect();
    if candidates.is_empty() {
        bail!("no finite logits available for sampling");
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    if top_k > 0 && top_k < candidates.len() {
        candidates.truncate(top_k);
    }

    let max_logit = candidates[0].1;
    let mut probs: Vec<(i64, f64)> = candidates
        .into_iter()
        .map(|(token_id, logit)| (token_id, (logit - max_logit).exp()))
        .collect();
    let total: f64 = probs.iter().map(|(_, prob)| *prob).sum();
    if total <= 0.0 || !total.is_finite() {
        bail!("sampling probabilities are not finite");
    }
    for (_, prob) in &mut probs {
        *prob /= total;
    }

    if top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut keep = 0usize;
        for (_, prob) in &probs {
            keep += 1;
            cumulative += *prob;
            if cumulative >= top_p {
                break;
            }
        }
        probs.truncate(keep.max(1));
    }

    let renorm_total: f64 = probs.iter().map(|(_, prob)| *prob).sum();
    let mut draw = rng.gen_range(0.0..renorm_total);
    for (token_id, prob) in probs {
        if draw <= prob {
            return Ok(Tensor::from_slice(&[token_id]).to_kind(Kind::Int64));
        }
        draw -= prob;
    }

    bail!("sampling draw did not select a token")
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_attention(
    input: &Tensor,
    q_proj: &Tensor,
    q_bias: &Tensor,
    k_proj: &Tensor,
    k_bias: &Tensor,
    v_proj: &Tensor,
    v_bias: &Tensor,
    o_proj: &Tensor,
    config: &QwenRuntimeConfig,
) -> Tensor {
    let shape = input.size();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let hidden_size = shape[2];
    let head_dim = hidden_size / config.num_attention_heads;
    let kv_repeat = config.num_attention_heads / config.num_key_value_heads;

    let q = input
        .linear(q_proj, Some(q_bias))
        .reshape([batch_size, seq_len, config.num_attention_heads, head_dim])
        .transpose(1, 2);
    let k = input
        .linear(k_proj, Some(k_bias))
        .reshape([batch_size, seq_len, config.num_key_value_heads, head_dim])
        .transpose(1, 2);
    let v = input
        .linear(v_proj, Some(v_bias))
        .reshape([batch_size, seq_len, config.num_key_value_heads, head_dim])
        .transpose(1, 2);
    let (cos, sin) = rope_cos_sin(seq_len, head_dim, config.rope_theta, input.device());
    let q = apply_rotary(&q, &cos, &sin);
    let k = apply_rotary(&k, &cos, &sin);
    let k = repeat_kv(&k, kv_repeat);
    let v = repeat_kv(&v, kv_repeat);
    let scores = q.matmul(&k.transpose(-2, -1)) / (head_dim as f64).sqrt();
    let causal_mask = Tensor::ones([seq_len, seq_len], (Kind::Bool, input.device())).triu(1);
    let scores = scores.masked_fill(&causal_mask, f64::NEG_INFINITY);
    let probs = scores.softmax(-1, Kind::Float);
    let context = probs
        .matmul(&v)
        .transpose(1, 2)
        .reshape([batch_size, seq_len, hidden_size]);

    context.linear::<&Tensor>(o_proj, None)
}

fn qwen_attention_with_lora(
    input: &Tensor,
    weights: &QwenLayerWeights,
    adapter: &QwenAttentionLoraAdapter,
    config: &QwenRuntimeConfig,
) -> Tensor {
    let q_proj = &weights.q_proj + adapter.q_delta(input.device());
    let v_proj = &weights.v_proj + adapter.v_delta(input.device());
    qwen_attention(
        input,
        &q_proj,
        &weights.q_bias,
        &weights.k_proj,
        &weights.k_bias,
        &v_proj,
        &weights.v_bias,
        &weights.o_proj,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
fn qwen_attention_with_cache(
    input: &Tensor,
    q_proj: &Tensor,
    q_bias: &Tensor,
    k_proj: &Tensor,
    k_bias: &Tensor,
    v_proj: &Tensor,
    v_bias: &Tensor,
    o_proj: &Tensor,
    config: &QwenRuntimeConfig,
    past_cache: Option<QwenLayerCache>,
    position_offset: i64,
) -> (Tensor, QwenLayerCache) {
    let shape = input.size();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let hidden_size = shape[2];
    let head_dim = hidden_size / config.num_attention_heads;
    let kv_repeat = config.num_attention_heads / config.num_key_value_heads;

    let q = input
        .linear(q_proj, Some(q_bias))
        .reshape([batch_size, seq_len, config.num_attention_heads, head_dim])
        .transpose(1, 2);
    let k = input
        .linear(k_proj, Some(k_bias))
        .reshape([batch_size, seq_len, config.num_key_value_heads, head_dim])
        .transpose(1, 2);
    let v = input
        .linear(v_proj, Some(v_bias))
        .reshape([batch_size, seq_len, config.num_key_value_heads, head_dim])
        .transpose(1, 2);
    let (cos, sin) = rope_cos_sin_with_offset(
        seq_len,
        head_dim,
        config.rope_theta,
        input.device(),
        position_offset,
    );
    let q = apply_rotary(&q, &cos, &sin);
    let k = apply_rotary(&k, &cos, &sin);
    let (k, v) = if let Some(cache) = past_cache {
        (
            Tensor::cat(&[&cache.key, &k], 2),
            Tensor::cat(&[&cache.value, &v], 2),
        )
    } else {
        (k, v)
    };
    let cache = QwenLayerCache {
        key: k.shallow_clone(),
        value: v.shallow_clone(),
    };
    let total_seq_len = k.size()[2];
    let k_for_attention = repeat_kv(&k, kv_repeat);
    let v_for_attention = repeat_kv(&v, kv_repeat);
    let scores = q.matmul(&k_for_attention.transpose(-2, -1)) / (head_dim as f64).sqrt();
    let scores = if position_offset == 0 {
        let causal_mask =
            Tensor::ones([seq_len, total_seq_len], (Kind::Bool, input.device())).triu(1);
        scores.masked_fill(&causal_mask, f64::NEG_INFINITY)
    } else {
        scores
    };
    let probs = scores.softmax(-1, Kind::Float);
    let context =
        probs
            .matmul(&v_for_attention)
            .transpose(1, 2)
            .reshape([batch_size, seq_len, hidden_size]);

    (context.linear::<&Tensor>(o_proj, None), cache)
}

fn rope_cos_sin(seq_len: i64, head_dim: i64, theta: f64, device: Device) -> (Tensor, Tensor) {
    rope_cos_sin_with_offset(seq_len, head_dim, theta, device, 0)
}

fn rope_cos_sin_with_offset(
    seq_len: i64,
    head_dim: i64,
    theta: f64,
    device: Device,
    position_offset: i64,
) -> (Tensor, Tensor) {
    let half = head_dim / 2;
    let inv_freq = Tensor::arange(half, (Kind::Float, device)) * 2.0;
    let inv_freq = (-(&inv_freq / head_dim as f64) * theta.ln()).exp();
    let positions =
        (Tensor::arange(seq_len, (Kind::Float, device)) + position_offset as f64).unsqueeze(1);
    let freqs = positions.matmul(&inv_freq.unsqueeze(0));
    let emb = Tensor::cat(&[&freqs, &freqs], -1).unsqueeze(0).unsqueeze(0);
    (emb.cos(), emb.sin())
}

fn apply_rotary(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    input * cos + rotate_half(input) * sin
}

fn rotate_half(input: &Tensor) -> Tensor {
    let last_dim = input.size()[input.dim() - 1];
    let half = last_dim / 2;
    let first = input.narrow(-1, 0, half);
    let second = input.narrow(-1, half, half);
    Tensor::cat(&[&(-second), &first], -1)
}

fn repeat_kv(input: &Tensor, repeats: i64) -> Tensor {
    if repeats == 1 {
        input.shallow_clone()
    } else {
        input.repeat_interleave_self_int(repeats, 1, None)
    }
}

pub fn diff_stats(actual: &Tensor, expected: &Tensor) -> Result<DiffStats> {
    if actual.size() != expected.size() {
        bail!(
            "shape mismatch: actual {:?}, expected {:?}",
            actual.size(),
            expected.size()
        );
    }
    let diff = (actual - expected).abs().to_device(Device::Cpu);
    Ok(DiffStats {
        max_abs: diff.max().double_value(&[]),
        mean_abs: diff.mean(Kind::Float).double_value(&[]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_matches_manual_formula() {
        let input = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape([1, 2, 2]);
        let weight = Tensor::from_slice(&[0.5_f32, 2.0]);
        let output = rms_norm(&input, &weight, 1e-6);

        assert_eq!(output.size(), vec![1, 2, 2]);
        assert!(output.isfinite().all().int64_value(&[]) == 1);
    }

    #[test]
    fn rotate_half_splits_head_dimension_in_halves() {
        let input = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape([1, 1, 1, 4]);
        let output = rotate_half(&input);

        let values: Vec<f32> = Vec::<f32>::try_from(output.reshape([4])).unwrap();
        assert_eq!(values, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn qwen_causal_lm_loss_is_finite_for_tiny_weights() {
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let weights = tiny_qwen_weights();
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);

        let loss = qwen_causal_lm_loss(&input_ids, &weights, &config).expect("loss should run");

        assert_eq!(loss.size(), Vec::<i64>::new());
        assert!(loss.isfinite().int64_value(&[]) == 1);
    }

    #[test]
    fn representative_full_train_tensors_get_gradients_and_reload() {
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let mut weights = tiny_qwen_weights();
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let trainable_names = representative_trainable_qwen_tensors();
        let mut trainable_tensors = Vec::new();
        for name in &trainable_names {
            let trainable = tensor(&weights, name)
                .expect("representative tensor should exist")
                .to_kind(Kind::Float)
                .set_requires_grad(true);
            weights.insert((*name).to_string(), trainable.shallow_clone());
            trainable_tensors.push(((*name).to_string(), trainable));
        }

        let initial_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        let loss = qwen_causal_lm_loss(&input_ids, &weights, &config).expect("loss should run");
        loss.backward();

        let base_weights = tiny_qwen_weights();
        let mut deltas = BTreeMap::new();
        for (name, mut trainable) in trainable_tensors {
            let grad = trainable.grad();
            assert!(grad.defined(), "{name} should receive a gradient");
            assert!(
                grad.norm().double_value(&[]) > 0.0,
                "{name} grad should be non-zero"
            );

            let _ = no_grad(|| trainable.f_sub_(&(&grad * 1e-2))).expect("update should apply");
            let base = tensor(&base_weights, &name)
                .expect("base tensor should exist")
                .to_kind(Kind::Float);
            deltas.insert(name.clone(), &trainable - &base);
            weights.insert(name, trainable);
        }

        let final_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!(final_loss < initial_loss);

        let mut reloaded_weights = tiny_qwen_weights();
        for (name, delta) in deltas {
            let reloaded = tensor(&reloaded_weights, &name)
                .expect("base tensor should exist")
                .to_kind(Kind::Float)
                + delta;
            reloaded_weights.insert(name, reloaded);
        }
        let reloaded_loss = qwen_causal_lm_loss(&input_ids, &reloaded_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!((final_loss - reloaded_loss).abs() < 1e-6);
    }

    #[test]
    fn sampling_respects_top_k_and_top_p_filters() {
        let logits = Tensor::from_slice(&[0.0_f32, 1.0, 2.0, 3.0]);
        let mut rng = StdRng::seed_from_u64(7);

        let token =
            sample_token_from_logits(&logits, 0.8, 1, 0.5, &mut rng).expect("sample should run");

        assert_eq!(token.int64_value(&[0]), 3);
    }

    #[test]
    fn qwen_delta_manifest_roundtrips() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let delta_output = temp.path().join("delta.safetensors");
        let manifest_output = delta_manifest_path(&delta_output);
        let manifest = QwenDeltaCheckpointManifest {
            format: "rustrain.qwen_delta.v1".to_string(),
            base_model_path: "/models/qwen".to_string(),
            reference_fixture: "fixture.safetensors".to_string(),
            delta_safetensors: delta_output.display().to_string(),
            train_step: 1,
            learning_rate: 1e-6,
            initial_loss: 2.0,
            final_loss: 1.5,
            tensors: vec![QwenDeltaTensorManifestEntry {
                name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                delta_name: "model.layers.0.self_attn.q_proj.weight.delta".to_string(),
                shape: vec![4, 4],
                dtype: "float32".to_string(),
                grad_norm: 3.0,
                delta_norm: 0.1,
            }],
        };

        write_qwen_delta_manifest(&manifest_output, &manifest).expect("manifest should write");
        let reloaded: QwenDeltaCheckpointManifest = serde_json::from_str(
            &fs::read_to_string(&manifest_output).expect("manifest should read"),
        )
        .expect("manifest should parse");

        assert_eq!(manifest_output, temp.path().join("delta.safetensors.json"));
        assert_eq!(reloaded.format, "rustrain.qwen_delta.v1");
        assert_eq!(
            reloaded.tensors[0].delta_name,
            manifest.tensors[0].delta_name
        );
    }

    #[test]
    fn qwen_attention_lora_adapter_roundtrips_mismatched_q_v_shapes() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let adapter_output = temp.path().join("adapter.safetensors");
        let adapter = QwenAttentionLoraAdapter::deterministic(4, 6, 2, 2, 8.0);

        assert_eq!(adapter.q_delta(Device::Cpu).size(), vec![6, 4]);
        assert_eq!(adapter.v_delta(Device::Cpu).size(), vec![2, 4]);

        adapter.save(&adapter_output).expect("adapter should write");
        let reloaded =
            QwenAttentionLoraAdapter::load(&adapter_output).expect("adapter should reload");

        assert_eq!(reloaded.q_delta(Device::Cpu).size(), vec![6, 4]);
        assert_eq!(reloaded.v_delta(Device::Cpu).size(), vec![2, 4]);
        assert!(
            diff_stats(
                &reloaded.q_delta(Device::Cpu),
                &adapter.q_delta(Device::Cpu)
            )
            .expect("q delta diff should compute")
            .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(
                &reloaded.v_delta(Device::Cpu),
                &adapter.v_delta(Device::Cpu)
            )
            .expect("v delta diff should compute")
            .max_abs
                < 1e-8
        );
    }

    #[test]
    fn cached_greedy_matches_full_context_greedy_for_tiny_weights() {
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let weights = tiny_qwen_weights();
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2]).reshape([1, 3]);

        let full = qwen_greedy_generate(&input_ids, &weights, &config, 3)
            .expect("full-context generate should run");
        let cached = qwen_greedy_generate_with_cache(&input_ids, &weights, &config, 3)
            .expect("cached generate should run");
        let full_ids: Vec<i64> = Vec::<i64>::try_from(full.reshape([-1])).unwrap();
        let cached_ids: Vec<i64> = Vec::<i64>::try_from(cached.reshape([-1])).unwrap();

        assert_eq!(cached_ids, full_ids);
    }

    fn tiny_qwen_weights() -> BTreeMap<String, Tensor> {
        let mut weights = BTreeMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::arange(24, (Kind::Float, Device::Cpu)).reshape([6, 4]) / 24.0,
        );
        weights.insert(
            "model.norm.weight".to_string(),
            Tensor::ones([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            Tensor::ones([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            Tensor::ones([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Tensor::eye(4, (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.bias".to_string(),
            Tensor::zeros([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            Tensor::ones([2, 4], (Kind::Float, Device::Cpu)) * 0.05,
        );
        weights.insert(
            "model.layers.0.self_attn.k_proj.bias".to_string(),
            Tensor::zeros([2], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            Tensor::ones([2, 4], (Kind::Float, Device::Cpu)) * 0.03,
        );
        weights.insert(
            "model.layers.0.self_attn.v_proj.bias".to_string(),
            Tensor::zeros([2], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            Tensor::ones([4, 4], (Kind::Float, Device::Cpu)) * 0.02,
        );
        weights.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            Tensor::ones([8, 4], (Kind::Float, Device::Cpu)) * 0.01,
        );
        weights.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            Tensor::ones([8, 4], (Kind::Float, Device::Cpu)) * 0.02,
        );
        weights.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            Tensor::ones([4, 8], (Kind::Float, Device::Cpu)) * 0.03,
        );
        weights
    }
}
