//! parity module - split from qwen_module.rs

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    io::{BufRead, BufReader, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use arrow::{
    array::{Array, LargeStringArray, RecordBatch, StringArray},
    datatypes::{DataType, SchemaRef},
    ipc::reader::{FileReader as ArrowFileReader, StreamReader as ArrowStreamReader},
};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use tch::{Device, IndexOp, Kind, Reduction, Tensor, no_grad};
use tokenizers::Tokenizer;
use tracing::info;

use rustrain_checkpoint::io::{
    delta_manifest_path, optimizer_state_path, qwen_lora_sft_adapter_manifest_path,
    read_qwen_lora_sft_resume_manifest, write_qwen_delta_manifest,
    write_qwen_lora_sft_adapter_manifest,
};
use rustrain_checkpoint::manifest::*;
use rustrain_checkpoint::safetensors::{read_safetensors_map, tensor};
use rustrain_core::runtime::{
    Config, DataConfig as RuntimeDataConfig, DataKind as RuntimeDataKind, Device as RuntimeDevice,
    FieldAffix, FieldCaseTransform, FieldCaseTransformKind, FieldDefault, FieldDefaultTarget,
    FieldRegexFilter, FieldRegexReplacement, FieldReplacement, FieldReplacementTarget, FieldSplit,
    FieldSplitSide, FieldStrip, FieldTransform, FieldTransformOp, FieldTruncation,
    LoraConfig as RuntimeLoraConfig, LrScheduler, RunPaths, load_config,
};
use rustrain_nccl::nccl_smoke;

use crate::generate::*;
use crate::lora::*;
use crate::model::*;
use crate::rank_smoke::*;
use crate::session::*;
use crate::sft::*;

#[derive(Debug, Serialize)]
pub(crate) struct QwenModuleParitySummary {
    pub(crate) model_safetensors: String,
    pub(crate) fixture: String,
    pub(crate) attention_diff: DiffStats,
    pub(crate) rms_norm_diff: DiffStats,
    pub(crate) mlp_diff: DiffStats,
    pub(crate) layer0_diff: DiffStats,
    pub(crate) layer1_diff: DiffStats,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenLogitsParitySummary {
    pub(crate) model_path: String,
    pub(crate) reference_fixture: String,
    pub(crate) input_ids: Vec<i64>,
    pub(crate) logits_shape: Vec<i64>,
    pub(crate) logits_diff: DiffStats,
    pub(crate) last_token_topk: Vec<TopLogit>,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenGenerateParitySummary {
    pub(crate) model_path: String,
    pub(crate) reference_fixture: String,
    pub(crate) prompt_len: usize,
    pub(crate) max_new_tokens: usize,
    pub(crate) generated_ids: Vec<i64>,
    pub(crate) cached_generated_ids: Option<Vec<i64>>,
    pub(crate) new_token_ids: Vec<i64>,
    pub(crate) reference_match: bool,
    pub(crate) cached_reference_match: Option<bool>,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenSamplingSmokeSummary {
    pub(crate) model_path: String,
    pub(crate) reference_fixture: String,
    pub(crate) prompt_len: usize,
    pub(crate) max_new_tokens: usize,
    pub(crate) temperature: f64,
    pub(crate) top_k: usize,
    pub(crate) top_p: f64,
    pub(crate) seed: u64,
    pub(crate) generated_ids: Vec<i64>,
    pub(crate) cached_ids: Vec<i64>,
    pub(crate) new_token_ids: Vec<i64>,
    pub(crate) cache_match: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenKvCacheParitySummary {
    pub(crate) model_path: String,
    pub(crate) reference_fixture: String,
    pub(crate) prompt_len: usize,
    pub(crate) max_new_tokens: usize,
    pub(crate) python_cached_ids: Option<Vec<i64>>,
    pub(crate) full_context_ids: Vec<i64>,
    pub(crate) cached_ids: Vec<i64>,
    pub(crate) new_token_ids: Vec<i64>,
    pub(crate) reference_match: bool,
    pub(crate) python_cached_reference_match: Option<bool>,
}

pub fn qwen_module_parity(model_safetensors: &Path, fixture: &Path) -> Result<()> {
    let model_safetensors = resolve_qwen_model_safetensors_path(model_safetensors)?;
    let weights = read_safetensors_map(&model_safetensors)?;
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
    let model_path = resolve_qwen_model_path(model_path)?;
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
    let model_path = resolve_qwen_model_path(model_path)?;
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    let expected_generated = tensor(&reference, "generated_ids")?.to_kind(Kind::Int64);
    let expected_cached_ids = if let Some(expected_cached) = reference.get("cached_generated_ids") {
        Some(Vec::<i64>::try_from(
            expected_cached
                .to_kind(Kind::Int64)
                .reshape([-1])
                .to_device(Device::Cpu),
        )?)
    } else {
        None
    };
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
    let cached_reference_match = expected_cached_ids
        .as_ref()
        .map(|cached_ids| cached_ids == &generated_ids);
    if cached_reference_match == Some(false) {
        bail!(
            "cached Python greedy generation fixture differs from Rust full-context generation: expected {:?}, got {:?}",
            expected_cached_ids.as_ref().expect("checked Some"),
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
        cached_generated_ids: expected_cached_ids,
        reference_match,
        cached_reference_match,
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
    let model_path = resolve_qwen_model_path(model_path)?;
    let runtime_config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    let prompt_len = input_ids.size()[1] as usize;
    let generated = qwen_sample_generate(
        &input_ids,
        &weights,
        &runtime_config,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        seed,
    )?;
    let cached = qwen_sample_generate_with_cache(
        &input_ids,
        &weights,
        &runtime_config,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        seed,
    )?;
    let generated_ids: Vec<i64> =
        Vec::<i64>::try_from(generated.reshape([-1]).to_device(Device::Cpu))?;
    let cached_ids: Vec<i64> = Vec::<i64>::try_from(cached.reshape([-1]).to_device(Device::Cpu))?;
    let cache_match = generated_ids == cached_ids;
    if !cache_match {
        bail!(
            "cached sampling diverged from full-context sampling: full={:?}, cached={:?}",
            generated_ids,
            cached_ids
        );
    }
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
        cached_ids,
        new_token_ids,
        cache_match,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_kv_cache_parity(
    model_path: &Path,
    reference_fixture: &Path,
    max_new_tokens: usize,
) -> Result<()> {
    let model_path = resolve_qwen_model_path(model_path)?;
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    let python_cached_ids = if let Some(expected_cached) = reference.get("cached_generated_ids") {
        Some(Vec::<i64>::try_from(
            expected_cached
                .to_kind(Kind::Int64)
                .reshape([-1])
                .to_device(Device::Cpu),
        )?)
    } else {
        None
    };
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
    let python_cached_reference_match = python_cached_ids
        .as_ref()
        .map(|reference_ids| reference_ids == &cached_ids);
    if python_cached_reference_match == Some(false) {
        bail!(
            "KV-cache greedy parity failed against Python cached generation: python={:?}, rust={:?}",
            python_cached_ids.as_ref().expect("checked Some"),
            cached_ids
        );
    }

    let summary = QwenKvCacheParitySummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        prompt_len,
        max_new_tokens,
        python_cached_ids,
        new_token_ids: cached_ids[prompt_len..].to_vec(),
        full_context_ids,
        cached_ids,
        reference_match,
        python_cached_reference_match,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}
