//! session module - split from qwen_module.rs

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
use crate::parity::*;
use crate::rank_smoke::*;
use crate::sft::*;

#[derive(Debug, Serialize)]
pub(crate) struct QwenTiedHeadTrainSummary {
    pub(crate) model_path: String,
    pub(crate) reference_fixture: String,
    pub(crate) delta_output: String,
    pub(crate) trainable_tensor: String,
    pub(crate) learning_rate: f64,
    pub(crate) initial_loss: f64,
    pub(crate) final_loss: f64,
    pub(crate) reloaded_loss: f64,
    pub(crate) reload_delta: f64,
    pub(crate) grad_defined: bool,
    pub(crate) grad_norm: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainableTensorSummary {
    pub name: String,
    pub grad_defined: bool,
    pub grad_norm: f64,
    pub delta_norm: f64,
}

#[derive(Debug, Serialize)]
pub struct QwenFullTrainSmokeSummary {
    pub model_path: String,
    pub reference_fixture: String,
    pub delta_output: String,
    pub optimizer_output: String,
    pub manifest_output: String,
    pub compute_kind: String,
    pub resume_from: Option<String>,
    pub resumed_checkpoint: bool,
    pub train_steps: usize,
    pub learning_rate: f64,
    pub step_losses: Vec<f64>,
    pub first_step_grad_norm: f64,
    pub final_step_grad_norm: f64,
    pub tokens_per_second: f64,
    pub samples_per_second: f64,
    pub memory_rss_mb: Option<f64>,
    pub gpu_memory_allocated_mb: Option<f64>,
    pub dataset_total_samples: Option<usize>,
    pub dataset_total_tokens: Option<usize>,
    pub dataset_train_samples: Option<usize>,
    pub dataset_eval_samples: Option<usize>,
    pub dataset_source_files: Option<Vec<String>>,
    pub dataset_source_sample_counts: Option<Vec<QwenSftSourceSampleCount>>,
    pub dataset_fingerprint: Option<String>,
    pub dataset_order_seed: Option<u64>,
    pub dataset_shuffle: Option<bool>,
    pub streaming_train_batches: Option<bool>,
    pub streaming_index_cache_path: Option<String>,
    pub streaming_index_cache_hit: Option<bool>,
    pub streaming_index_cache_written: Option<bool>,
    pub data_cursor_start: Option<usize>,
    pub data_cursor_end: Option<usize>,
    pub data_cursor_next: Option<usize>,
    pub data_epoch_start: Option<usize>,
    pub data_epoch_end: Option<usize>,
    pub data_epoch_next: Option<usize>,
    pub data_sample_offset_start: Option<usize>,
    pub data_sample_offset_end: Option<usize>,
    pub data_sample_offset_next: Option<usize>,
    pub batch_size: usize,
    pub sequence_tokens: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub reloaded_loss: f64,
    pub reload_delta: f64,
    pub resume_loss: f64,
    pub continuous_second_loss: f64,
    pub resumed_second_loss: f64,
    pub second_step_delta: f64,
    pub trainable_tensors: Vec<TrainableTensorSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct QwenGradSignature {
    pub(crate) name: String,
    pub(crate) shape: Vec<i64>,
    pub(crate) samples: Vec<f32>,
}

pub(crate) struct AdamSlotNames {
    pub(crate) m: String,
    pub(crate) v: String,
}

pub(crate) struct AdamState {
    pub(crate) m: Tensor,
    pub(crate) v: Tensor,
}

pub(crate) struct QwenTrainableParameter {
    pub(crate) name: String,
    pub(crate) tensor: Tensor,
    pub(crate) base: Tensor,
    pub(crate) adam: Option<AdamState>,
}

pub(crate) struct QwenTrainStepArtifacts {
    pub(crate) tensor_summaries: Vec<TrainableTensorSummary>,
    pub(crate) manifest_tensors: Vec<QwenDeltaTensorManifestEntry>,
    pub(crate) delta_entries: Vec<(String, Tensor)>,
    pub(crate) optimizer_entries: Vec<(String, Tensor)>,
}

pub(crate) struct QwenTrainableRegistry {
    pub(crate) parameters: Vec<QwenTrainableParameter>,
}

pub(crate) struct QwenTrainStepResult {
    pub(crate) loss_before: f64,
    pub(crate) loss_after: f64,
    pub(crate) artifacts: QwenTrainStepArtifacts,
}

pub(crate) struct QwenTrainableSession {
    pub(crate) config: QwenRuntimeConfig,
    pub(crate) weights: BTreeMap<String, Tensor>,
    pub(crate) input_ids: Tensor,
    pub(crate) compute_kind: Kind,
    pub(crate) registry: QwenTrainableRegistry,
}

pub(crate) struct QwenSessionBatchPlan {
    pub(crate) initial_input_ids: Tensor,
    pub(crate) train_batches: Vec<Tensor>,
    pub(crate) reference_fixture: String,
    pub(crate) dataset_total_samples: Option<usize>,
    pub(crate) dataset_total_tokens: Option<usize>,
    pub(crate) dataset_train_samples: Option<usize>,
    pub(crate) dataset_eval_samples: Option<usize>,
    pub(crate) dataset_source_files: Option<Vec<String>>,
    pub(crate) dataset_source_sample_counts: Option<Vec<QwenSftSourceSampleCount>>,
    pub(crate) dataset_fingerprint: Option<String>,
    pub(crate) dataset_order_seed: Option<u64>,
    pub(crate) dataset_shuffle: Option<bool>,
    pub(crate) streaming_train_batches: Option<bool>,
    pub(crate) streaming_index_cache_path: Option<String>,
    pub(crate) streaming_index_cache_hit: Option<bool>,
    pub(crate) streaming_index_cache_written: Option<bool>,
    pub(crate) train_sample_count: Option<usize>,
    pub(crate) data_epoch_start: Option<usize>,
    pub(crate) data_epoch_end: Option<usize>,
    pub(crate) data_epoch_next: Option<usize>,
    pub(crate) data_sample_offset_start: Option<usize>,
    pub(crate) data_sample_offset_end: Option<usize>,
    pub(crate) data_sample_offset_next: Option<usize>,
    pub(crate) batch_size: usize,
    pub(crate) sequence_tokens: usize,
}

pub(crate) struct QwenSessionDpBatchPlan {
    pub(crate) global_initial_input_ids: Tensor,
    pub(crate) global_train_batches: Vec<Tensor>,
    pub(crate) data_kind: Option<String>,
    pub(crate) dataset_total_samples: Option<usize>,
    pub(crate) dataset_total_tokens: Option<usize>,
    pub(crate) dataset_train_samples: Option<usize>,
    pub(crate) dataset_eval_samples: Option<usize>,
    pub(crate) dataset_source_files: Option<Vec<String>>,
    pub(crate) dataset_source_sample_counts: Option<Vec<QwenSftSourceSampleCount>>,
    pub(crate) dataset_fingerprint: Option<String>,
    pub(crate) dataset_order_seed: Option<u64>,
    pub(crate) dataset_shuffle: Option<bool>,
    pub(crate) streaming_train_batches: Option<bool>,
    pub(crate) streaming_index_cache_path: Option<String>,
    pub(crate) streaming_index_cache_hit: Option<bool>,
    pub(crate) streaming_index_cache_written: Option<bool>,
    pub(crate) train_sample_count: Option<usize>,
    pub(crate) data_epoch_start: Option<usize>,
    pub(crate) data_epoch_end: Option<usize>,
    pub(crate) data_epoch_next: Option<usize>,
    pub(crate) data_sample_offset_start: Option<usize>,
    pub(crate) data_sample_offset_end: Option<usize>,
    pub(crate) data_sample_offset_next: Option<usize>,
    pub(crate) local_batch_size: usize,
    pub(crate) sequence_tokens: usize,
}

pub(crate) struct QwenAttentionDpSession {
    pub(crate) config: QwenRuntimeConfig,
    pub(crate) input: Tensor,
    pub(crate) target: Tensor,
    pub(crate) q_proj: Tensor,
    pub(crate) q_bias: Tensor,
    pub(crate) k_proj: Tensor,
    pub(crate) k_bias: Tensor,
    pub(crate) v_proj: Tensor,
    pub(crate) v_bias: Tensor,
    pub(crate) o_proj: Tensor,
    pub(crate) compute_kind: Kind,
}

pub(crate) fn qwen_data_epoch_and_offset(
    cursor: usize,
    sample_count: usize,
) -> Result<(usize, usize)> {
    if sample_count == 0 {
        bail!("data epoch metadata requires at least one training sample");
    }
    Ok((cursor / sample_count, cursor % sample_count))
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

    let model_path = resolve_qwen_model_path(model_path)?;
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
    dtype: QwenComputeDType,
    learning_rate: f64,
) -> Result<()> {
    let summary = qwen_full_train_summary(
        model_path,
        reference_fixture,
        delta_output,
        dtype,
        learning_rate,
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub(crate) fn qwen_full_train_summary(
    model_path: &Path,
    reference_fixture: &Path,
    delta_output: &Path,
    dtype: QwenComputeDType,
    learning_rate: f64,
) -> Result<QwenFullTrainSmokeSummary> {
    if learning_rate <= 0.0 {
        bail!("learning_rate must be positive");
    }

    let model_path = resolve_qwen_model_path(model_path)?;
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);

    let mut session = QwenTrainableSession::from_weights(config, weights, input_ids, dtype.kind())?;
    let first_step = session.train_step(learning_rate, 1)?;
    let initial_loss = first_step.loss_before;
    let final_loss = first_step.loss_after;
    if final_loss >= initial_loss {
        bail!(
            "Qwen full train smoke failed to reduce loss: initial_loss={initial_loss}, final_loss={final_loss}"
        );
    }

    if let Some(parent) = delta_output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let delta_refs: Vec<(&str, &Tensor)> = first_step
        .artifacts
        .delta_entries
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    Tensor::write_safetensors(&delta_refs, delta_output)
        .with_context(|| format!("failed to write {}", delta_output.display()))?;
    let optimizer_output = optimizer_state_path(delta_output);
    let optimizer_refs: Vec<(&str, &Tensor)> = first_step
        .artifacts
        .optimizer_entries
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    Tensor::write_safetensors(&optimizer_refs, &optimizer_output)
        .with_context(|| format!("failed to write {}", optimizer_output.display()))?;
    let manifest_output = delta_manifest_path(delta_output);
    let manifest = QwenDeltaCheckpointManifest {
        format: "rustrain.qwen_delta.v1".to_string(),
        base_model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        delta_safetensors: delta_output.display().to_string(),
        optimizer_safetensors: Some(optimizer_output.display().to_string()),
        train_step: 1,
        data_cursor_start: None,
        data_cursor_end: None,
        data_cursor_next: None,
        data_epoch_start: None,
        data_epoch_end: None,
        data_epoch_next: None,
        data_sample_offset_start: None,
        data_sample_offset_end: None,
        data_sample_offset_next: None,
        dataset_source_files: Vec::new(),
        dataset_source_sample_counts: Vec::new(),
        dataset_fingerprint: String::new(),
        dataset_shuffle: true,
        streaming_train_batches: None,
        learning_rate,
        initial_loss,
        final_loss,
        tensors: first_step.artifacts.manifest_tensors,
    };
    write_qwen_delta_manifest(&manifest_output, &manifest)?;

    let mut resumed_session = QwenTrainableSession::from_manifest(
        session.config,
        read_safetensors_map(&model_path.join("model.safetensors"))?,
        session.input_ids.shallow_clone(),
        dtype.kind(),
        &manifest,
    )?;
    let reloaded_loss = resumed_session.loss_value()?;
    let reload_delta = (final_loss - reloaded_loss).abs();
    if reload_delta > 1e-5 {
        bail!(
            "Qwen full train delta reload parity failed: final_loss={final_loss}, reloaded_loss={reloaded_loss}, reload_delta={reload_delta}"
        );
    }

    let resumed_second_step = resumed_session.train_step(learning_rate, 2)?;
    let resume_loss_value = resumed_second_step.loss_before;
    let resumed_second_loss = resumed_second_step.loss_after;

    let continuous_second_step = session.train_step(learning_rate, 2)?;
    let continuous_second_loss = continuous_second_step.loss_after;
    let second_step_delta = (continuous_second_loss - resumed_second_loss).abs();
    if second_step_delta > 1e-5 {
        bail!(
            "Qwen full train manifest resume parity failed: continuous_second_loss={continuous_second_loss}, resumed_second_loss={resumed_second_loss}, second_step_delta={second_step_delta}"
        );
    }

    Ok(QwenFullTrainSmokeSummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        delta_output: delta_output.display().to_string(),
        optimizer_output: optimizer_output.display().to_string(),
        manifest_output: manifest_output.display().to_string(),
        compute_kind: dtype.label().to_string(),
        resume_from: None,
        resumed_checkpoint: false,
        train_steps: 1,
        learning_rate,
        step_losses: vec![initial_loss, final_loss],
        first_step_grad_norm: first_step
            .artifacts
            .tensor_summaries
            .iter()
            .map(|summary| summary.grad_norm * summary.grad_norm)
            .sum::<f64>()
            .sqrt(),
        final_step_grad_norm: first_step
            .artifacts
            .tensor_summaries
            .iter()
            .map(|summary| summary.grad_norm * summary.grad_norm)
            .sum::<f64>()
            .sqrt(),
        tokens_per_second: 0.0,
        samples_per_second: 0.0,
        memory_rss_mb: rustrain_train::metrics::memory_rss_mb(),
        gpu_memory_allocated_mb: rustrain_train::metrics::gpu_memory_allocated_mb(),
        dataset_total_samples: None,
        dataset_total_tokens: None,
        dataset_train_samples: None,
        dataset_eval_samples: None,
        dataset_source_files: None,
        dataset_source_sample_counts: None,
        dataset_fingerprint: None,
        dataset_order_seed: None,
        dataset_shuffle: None,
        streaming_train_batches: None,
        streaming_index_cache_path: None,
        streaming_index_cache_hit: None,
        streaming_index_cache_written: None,
        data_cursor_start: None,
        data_cursor_end: None,
        data_cursor_next: None,
        data_epoch_start: None,
        data_epoch_end: None,
        data_epoch_next: None,
        data_sample_offset_start: None,
        data_sample_offset_end: None,
        data_sample_offset_next: None,
        batch_size: session.input_ids.size()[0] as usize,
        sequence_tokens: session.input_ids.size()[1] as usize,
        initial_loss,
        final_loss,
        reloaded_loss,
        reload_delta,
        resume_loss: resume_loss_value,
        continuous_second_loss,
        resumed_second_loss,
        second_step_delta,
        trainable_tensors: first_step.artifacts.tensor_summaries,
    })
}

pub(crate) fn qwen_session_single_summary(
    model_path: &Path,
    delta_output: &Path,
    dtype: QwenComputeDType,
    train_steps: usize,
    learning_rate: f64,
    resume_from: Option<&Path>,
    trainable_layers: &[usize],
    runtime_config: Option<&Config>,
    streaming_index_cache: Option<&Path>,
) -> Result<QwenFullTrainSmokeSummary> {
    if train_steps == 0 {
        bail!("qwen session single trainer requires max_steps > 0");
    }
    if learning_rate <= 0.0 {
        bail!("learning_rate must be positive");
    }

    let model_path = resolve_qwen_model_path(model_path)?;
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let loaded_manifest = resume_from
        .map(|resume_from| {
            let manifest_text = fs::read_to_string(resume_from)
                .with_context(|| format!("failed to read {}", resume_from.display()))?;
            serde_json::from_str::<QwenDeltaCheckpointManifest>(&manifest_text)
                .with_context(|| format!("failed to parse {}", resume_from.display()))
        })
        .transpose()?
        .map(Arc::new);
    let (start_step, data_cursor_start) = if let Some(manifest) = loaded_manifest.as_ref() {
        let start_step = manifest
            .train_step
            .checked_add(1)
            .ok_or_else(|| anyhow!("Qwen session resume train_step overflowed"))?
            as usize;
        let inferred_cursor = manifest.train_step as usize;
        (
            start_step,
            manifest.data_cursor_next.unwrap_or(inferred_cursor),
        )
    } else {
        (1, 0)
    };
    let batch_plan = qwen_session_batch_plan_from_config(
        &model_path,
        &weights,
        data_cursor_start,
        train_steps,
        runtime_config,
        streaming_index_cache,
    )?;
    if let Some(manifest) = loaded_manifest.as_ref() {
        qwen_validate_optional_sft_resume_dataset(
            &manifest.dataset_source_files,
            &manifest.dataset_source_sample_counts,
            &manifest.dataset_fingerprint,
            manifest.dataset_shuffle,
            batch_plan.dataset_source_files.as_deref(),
            batch_plan.dataset_source_sample_counts.as_deref(),
            batch_plan.dataset_fingerprint.as_deref(),
            batch_plan.dataset_shuffle,
            "Qwen session checkpoint resume",
        )?;
    }
    let (mut session, start_step, data_cursor_start) =
        if let Some(manifest) = loaded_manifest.as_ref() {
            (
                QwenTrainableSession::from_manifest(
                    config,
                    weights,
                    batch_plan.initial_input_ids.shallow_clone(),
                    dtype.kind(),
                    manifest,
                )?,
                start_step,
                data_cursor_start,
            )
        } else {
            (
                QwenTrainableSession::from_trainable_layers(
                    config,
                    weights,
                    batch_plan.initial_input_ids.shallow_clone(),
                    dtype.kind(),
                    trainable_layers,
                )?,
                1,
                0,
            )
        };
    let mut step_losses = Vec::with_capacity(train_steps + 1);
    let mut last_step = None;
    let end_step = start_step + train_steps - 1;
    let mut first_step_grad_norm = 0.0;
    let mut final_step_grad_norm = 0.0;
    let train_started = Instant::now();
    for step in start_step..=end_step {
        let batch_index = if batch_plan.train_sample_count.is_some() {
            (step - start_step) * batch_plan.batch_size
        } else {
            data_cursor_start + (step - start_step) * batch_plan.batch_size
        };
        let input_ids = batch_plan
            .train_batches
            .get(batch_index)
            .ok_or_else(|| anyhow!("missing qwen trainable session batch for step {step}"))?;
        session.set_input_ids(input_ids);
        let step_result = session.train_step(learning_rate, step as i32)?;
        if step == start_step {
            step_losses.push(step_result.loss_before);
        }
        step_losses.push(step_result.loss_after);
        let step_grad_norm = qwen_train_artifacts_grad_norm(&step_result.artifacts);
        if step == start_step {
            first_step_grad_norm = step_grad_norm;
        }
        final_step_grad_norm = step_grad_norm;
        last_step = Some(step_result);
    }
    let train_elapsed_secs = train_started.elapsed().as_secs_f64().max(1e-9);
    let local_batch_size = session.input_ids.size()[0] as f64;
    let sequence_tokens = session.input_ids.size()[1] as f64;
    let samples_per_second = local_batch_size * train_steps as f64 / train_elapsed_secs;
    let tokens_per_second =
        local_batch_size * sequence_tokens * train_steps as f64 / train_elapsed_secs;
    let final_step = last_step.expect("train_steps > 0 guarantees a final step");
    let final_artifacts = final_step.artifacts;
    let data_cursor_end = data_cursor_start + train_steps * batch_plan.batch_size;
    let data_cursor_next = data_cursor_end;
    let initial_loss = *step_losses
        .first()
        .expect("step_losses should contain initial loss");
    let final_loss = *step_losses
        .last()
        .expect("step_losses should contain final loss");
    if final_loss >= initial_loss && batch_plan.train_sample_count.is_none() {
        bail!(
            "Qwen session single trainer failed to reduce loss: initial_loss={initial_loss}, final_loss={final_loss}"
        );
    }
    if !initial_loss.is_finite() || !final_loss.is_finite() {
        bail!(
            "Qwen session single trainer produced non-finite loss: initial_loss={initial_loss}, final_loss={final_loss}"
        );
    }

    if let Some(parent) = delta_output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let delta_refs: Vec<(&str, &Tensor)> = final_artifacts
        .delta_entries
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    Tensor::write_safetensors(&delta_refs, delta_output)
        .with_context(|| format!("failed to write {}", delta_output.display()))?;
    let optimizer_output = optimizer_state_path(delta_output);
    let optimizer_refs: Vec<(&str, &Tensor)> = final_artifacts
        .optimizer_entries
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    Tensor::write_safetensors(&optimizer_refs, &optimizer_output)
        .with_context(|| format!("failed to write {}", optimizer_output.display()))?;
    let manifest_output = delta_manifest_path(delta_output);
    let manifest = QwenDeltaCheckpointManifest {
        format: "rustrain.qwen_delta.v1".to_string(),
        base_model_path: model_path.display().to_string(),
        reference_fixture: "qwen_session_single_fixed_tokens".to_string(),
        delta_safetensors: delta_output.display().to_string(),
        optimizer_safetensors: Some(optimizer_output.display().to_string()),
        train_step: end_step as u64,
        data_cursor_start: Some(data_cursor_start),
        data_cursor_end: Some(data_cursor_end),
        data_cursor_next: Some(data_cursor_next),
        data_epoch_start: batch_plan.data_epoch_start,
        data_epoch_end: batch_plan.data_epoch_end,
        data_epoch_next: batch_plan.data_epoch_next,
        data_sample_offset_start: batch_plan.data_sample_offset_start,
        data_sample_offset_end: batch_plan.data_sample_offset_end,
        data_sample_offset_next: batch_plan.data_sample_offset_next,
        dataset_source_files: batch_plan.dataset_source_files.clone().unwrap_or_default(),
        dataset_source_sample_counts: batch_plan
            .dataset_source_sample_counts
            .clone()
            .unwrap_or_default(),
        dataset_fingerprint: batch_plan.dataset_fingerprint.clone().unwrap_or_default(),
        dataset_shuffle: batch_plan.dataset_shuffle.unwrap_or(true),
        streaming_train_batches: batch_plan.streaming_train_batches,
        learning_rate,
        initial_loss,
        final_loss,
        tensors: final_artifacts.manifest_tensors.clone(),
    };
    write_qwen_delta_manifest(&manifest_output, &manifest)?;

    let mut resumed_session = QwenTrainableSession::from_manifest(
        session.config,
        read_safetensors_map(&model_path.join("model.safetensors"))?,
        session.input_ids.shallow_clone(),
        dtype.kind(),
        &manifest,
    )?;
    let reloaded_loss = resumed_session.loss_value()?;
    let reload_delta = (final_loss - reloaded_loss).abs();
    if reload_delta > 1e-5 {
        bail!(
            "Qwen session single delta reload parity failed: final_loss={final_loss}, reloaded_loss={reloaded_loss}, reload_delta={reload_delta}"
        );
    }

    let next_step = end_step + 1;
    let next_batch_index = if batch_plan.train_sample_count.is_some() {
        train_steps * batch_plan.batch_size
    } else {
        data_cursor_next
    };
    let next_batch = batch_plan
        .train_batches
        .get(next_batch_index)
        .ok_or_else(|| anyhow!("missing qwen trainable session next-step batch"))?;
    resumed_session.set_input_ids(next_batch);
    let resumed_second_step = resumed_session.train_step(learning_rate, next_step as i32)?;
    let resume_loss_value = resumed_second_step.loss_before;
    let resumed_second_loss = resumed_second_step.loss_after;
    session.set_input_ids(next_batch);
    let continuous_second_step = session.train_step(learning_rate, next_step as i32)?;
    let continuous_second_loss = continuous_second_step.loss_after;
    let second_step_delta = (continuous_second_loss - resumed_second_loss).abs();
    if second_step_delta > 1e-5 {
        bail!(
            "Qwen session single manifest resume parity failed: continuous_second_loss={continuous_second_loss}, resumed_second_loss={resumed_second_loss}, second_step_delta={second_step_delta}"
        );
    }

    Ok(QwenFullTrainSmokeSummary {
        model_path: model_path.display().to_string(),
        reference_fixture: batch_plan.reference_fixture,
        delta_output: delta_output.display().to_string(),
        optimizer_output: optimizer_output.display().to_string(),
        manifest_output: manifest_output.display().to_string(),
        compute_kind: dtype.label().to_string(),
        resume_from: resume_from.map(|path| path.display().to_string()),
        resumed_checkpoint: resume_from.is_some(),
        train_steps,
        learning_rate,
        step_losses,
        first_step_grad_norm,
        final_step_grad_norm,
        tokens_per_second,
        samples_per_second,
        memory_rss_mb: rustrain_train::metrics::memory_rss_mb(),
        gpu_memory_allocated_mb: rustrain_train::metrics::gpu_memory_allocated_mb(),
        dataset_total_samples: batch_plan.dataset_total_samples,
        dataset_total_tokens: batch_plan.dataset_total_tokens,
        dataset_train_samples: batch_plan.dataset_train_samples,
        dataset_eval_samples: batch_plan.dataset_eval_samples,
        dataset_source_files: batch_plan.dataset_source_files,
        dataset_source_sample_counts: batch_plan.dataset_source_sample_counts,
        dataset_fingerprint: batch_plan.dataset_fingerprint,
        dataset_order_seed: batch_plan.dataset_order_seed,
        dataset_shuffle: batch_plan.dataset_shuffle,
        streaming_train_batches: batch_plan.streaming_train_batches,
        streaming_index_cache_path: batch_plan.streaming_index_cache_path,
        streaming_index_cache_hit: batch_plan.streaming_index_cache_hit,
        streaming_index_cache_written: batch_plan.streaming_index_cache_written,
        data_cursor_start: batch_plan.train_sample_count.map(|_| data_cursor_start),
        data_cursor_end: batch_plan.train_sample_count.map(|_| data_cursor_end),
        data_cursor_next: batch_plan.train_sample_count.map(|_| data_cursor_next),
        data_epoch_start: batch_plan.data_epoch_start,
        data_epoch_end: batch_plan.data_epoch_end,
        data_epoch_next: batch_plan.data_epoch_next,
        data_sample_offset_start: batch_plan.data_sample_offset_start,
        data_sample_offset_end: batch_plan.data_sample_offset_end,
        data_sample_offset_next: batch_plan.data_sample_offset_next,
        batch_size: batch_plan.batch_size,
        sequence_tokens: batch_plan.sequence_tokens,
        initial_loss,
        final_loss,
        reloaded_loss,
        reload_delta,
        resume_loss: resume_loss_value,
        continuous_second_loss,
        resumed_second_loss,
        second_step_delta,
        trainable_tensors: final_artifacts.tensor_summaries,
    })
}

pub(crate) fn qwen_train_artifacts_grad_norm(artifacts: &QwenTrainStepArtifacts) -> f64 {
    artifacts
        .tensor_summaries
        .iter()
        .map(|summary| summary.grad_norm * summary.grad_norm)
        .sum::<f64>()
        .sqrt()
}

pub fn train_qwen_session_dp_from_config(config: &Config, _run_paths: &RunPaths) -> Result<()> {
    if config.model.architecture != "qwen_trainable_session" {
        bail!(
            "qwen session trainer expects architecture = qwen_trainable_session, got {}",
            config.model.architecture
        );
    }
    if !matches!(config.train.device, RuntimeDevice::Cuda) {
        bail!("qwen session trainer requires device = cuda");
    }
    if config.parallel.data_parallel_size != 2 {
        bail!("qwen session trainer currently expects data_parallel_size = 2");
    }
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("qwen session trainer requires model.model_path")?;
    let model_path = resolve_qwen_model_path(model_path)?;
    let output_dir = std::env::var("RUSTRAIN_LAUNCH_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            config
                .run
                .base_dir
                .join("qwen-session-trainer-dp")
                .join(&config.run.name)
        })
        .join("qwen-session-dp-ranks");
    let dtype = match config.train.dtype {
        rustrain_core::runtime::DType::Fp32 => QwenComputeDType::Fp32,
        rustrain_core::runtime::DType::Bf16 => QwenComputeDType::Bf16,
        rustrain_core::runtime::DType::Fp16 => {
            bail!("qwen session trainer does not support fp16 yet; use fp32 or bf16")
        }
    };
    qwen_session_dp_rank_smoke(
        &model_path,
        output_dir,
        dtype,
        config.train.max_steps as usize,
        config.train.learning_rate as f64,
        &qwen_session_trainable_layers_from_config(config),
        config.train.resume_from.as_deref(),
        Some(config),
    )
}

pub fn train_qwen_session_tp_from_config(config: &Config, _run_paths: &RunPaths) -> Result<()> {
    if config.model.architecture != "qwen_trainable_session" {
        bail!(
            "qwen session trainer expects architecture = qwen_trainable_session, got {}",
            config.model.architecture
        );
    }
    if !matches!(config.train.device, RuntimeDevice::Cuda) {
        bail!("qwen session trainer requires device = cuda");
    }
    if config.parallel.tensor_model_parallel_size != 2 {
        bail!("qwen session TP trainer currently expects tensor_model_parallel_size = 2");
    }
    if config.parallel.data_parallel_size != 1 {
        bail!("qwen session TP trainer currently expects data_parallel_size = 1");
    }
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("qwen session trainer requires model.model_path")?;
    let model_path = resolve_qwen_model_path(model_path)?;
    let output_dir = std::env::var("RUSTRAIN_LAUNCH_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            config
                .run
                .base_dir
                .join("qwen-session-trainer-tp")
                .join(&config.run.name)
        })
        .join("qwen-session-tp-ranks");
    qwen_session_tp_rank_smoke(&model_path, output_dir, config)
}

pub fn train_qwen_session_single_from_config(
    config: &Config,
    run_paths: &RunPaths,
) -> Result<QwenFullTrainSmokeSummary> {
    if config.model.architecture != "qwen_trainable_session" {
        bail!(
            "qwen session trainer expects architecture = qwen_trainable_session, got {}",
            config.model.architecture
        );
    }
    if !matches!(config.train.device, RuntimeDevice::Cuda) {
        bail!("qwen session trainer requires device = cuda");
    }
    if config.parallel.data_parallel_size != 1 {
        bail!("qwen session single trainer expects data_parallel_size = 1");
    }
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("qwen session trainer requires model.model_path")?;
    let model_path = resolve_qwen_model_path(model_path)?;
    let dtype = match config.train.dtype {
        rustrain_core::runtime::DType::Fp32 => QwenComputeDType::Fp32,
        rustrain_core::runtime::DType::Bf16 => QwenComputeDType::Bf16,
        rustrain_core::runtime::DType::Fp16 => {
            bail!("qwen session trainer does not support fp16 yet; use fp32 or bf16")
        }
    };
    let streaming_index_cache = config.data.as_ref().and_then(|data| {
        if data.kind == RuntimeDataKind::InstructionJsonl {
            Some(data.index_cache.clone().unwrap_or_else(|| {
                qwen_sft_streaming_index_cache_path(&run_paths.cache, "qwen-session-single")
            }))
        } else {
            data.index_cache.clone()
        }
    });
    qwen_session_single_summary(
        &model_path,
        &run_paths
            .checkpoints
            .join("qwen-session-single-delta.safetensors"),
        dtype,
        config.train.max_steps as usize,
        config.train.learning_rate as f64,
        config.train.resume_from.as_deref(),
        &qwen_session_trainable_layers_from_config(config),
        Some(config),
        streaming_index_cache.as_deref(),
    )
}

pub(crate) fn qwen_dp_artifact_dir(output_dir: &Path) -> Result<PathBuf> {
    let port = std::env::var("MASTER_PORT")
        .context("MASTER_PORT is not set; run through rustrain launch")?;
    Ok(output_dir.join(format!("launch-{port}")))
}

pub(crate) fn qwen_tp_artifact_dir(output_dir: &Path) -> Result<PathBuf> {
    let port = std::env::var("MASTER_PORT")
        .context("MASTER_PORT is not set; run through rustrain launch")?;
    Ok(output_dir.join(format!("launch-{port}")))
}

pub(crate) fn adam_slot_names(name: &str) -> AdamSlotNames {
    AdamSlotNames {
        m: format!("{name}.adam_m"),
        v: format!("{name}.adam_v"),
    }
}

impl QwenTrainableRegistry {
    pub(crate) fn representative(weights: &mut BTreeMap<String, Tensor>) -> Result<Self> {
        Self::from_names(weights, representative_trainable_qwen_tensors())
    }

    pub(crate) fn from_names(
        weights: &mut BTreeMap<String, Tensor>,
        names: Vec<String>,
    ) -> Result<Self> {
        let mut parameters = Vec::with_capacity(names.len());
        for name in names {
            let base = tensor(weights, &name)?.to_kind(Kind::Float);
            let trainable = base.shallow_clone().set_requires_grad(true);
            weights.insert(name.clone(), trainable.shallow_clone());
            parameters.push(QwenTrainableParameter {
                name,
                tensor: trainable,
                base: tensor_snapshot(&base),
                adam: None,
            });
        }
        Ok(Self { parameters })
    }

    pub(crate) fn from_names_on_device(
        weights: &mut BTreeMap<String, Tensor>,
        names: Vec<String>,
        device: Device,
    ) -> Result<Self> {
        let mut parameters = Vec::with_capacity(names.len());
        for name in names {
            let base = tensor(weights, &name)?
                .to_kind(Kind::Float)
                .to_device(device);
            let trainable = base.shallow_clone().set_requires_grad(true);
            weights.insert(name.clone(), trainable.shallow_clone());
            parameters.push(QwenTrainableParameter {
                name,
                tensor: trainable,
                base: tensor_snapshot(&base),
                adam: None,
            });
        }
        Ok(Self { parameters })
    }

    pub(crate) fn adamw_step(
        &mut self,
        weights: &mut BTreeMap<String, Tensor>,
        learning_rate: f64,
        step: i32,
    ) -> Result<QwenTrainStepArtifacts> {
        let grads = self.grad_entries()?;
        self.adamw_step_with_grads(weights, &grads, learning_rate, step)
    }

    pub(crate) fn adamw_step_with_grads(
        &mut self,
        weights: &mut BTreeMap<String, Tensor>,
        averaged_grads: &[(String, Tensor)],
        learning_rate: f64,
        step: i32,
    ) -> Result<QwenTrainStepArtifacts> {
        if averaged_grads.len() != self.parameters.len() {
            bail!(
                "averaged gradient count mismatch: got {}, expected {}",
                averaged_grads.len(),
                self.parameters.len()
            );
        }
        let mut tensor_summaries = Vec::with_capacity(self.parameters.len());
        let mut manifest_tensors = Vec::with_capacity(self.parameters.len());
        let mut delta_entries = Vec::with_capacity(self.parameters.len());
        let mut optimizer_entries = Vec::with_capacity(self.parameters.len() * 2);

        for (parameter, (grad_name, grad)) in self.parameters.iter_mut().zip(averaged_grads.iter())
        {
            if &parameter.name != grad_name {
                bail!(
                    "averaged gradient order mismatch: got {}, expected {}",
                    grad_name,
                    parameter.name
                );
            }
            let grad = grad.to_device(parameter.tensor.device());
            let grad_norm = grad.norm().double_value(&[]);
            if grad_norm <= 0.0 {
                bail!("averaged gradient for {} has zero norm", parameter.name);
            }
            let grad_defined = true;

            let adam_state = adamw_next_state(parameter.adam.as_ref(), &grad, 0.9, 0.999);
            let update = adamw_update(&adam_state, learning_rate, 0.9, 0.999, step, 1e-8);
            let _ = no_grad(|| parameter.tensor.f_sub_(&update))?;
            weights.insert(parameter.name.clone(), parameter.tensor.shallow_clone());

            let delta = &parameter.tensor - &parameter.base;
            let delta_norm = delta.norm().double_value(&[]);
            let delta_name = format!("{}.delta", parameter.name);
            let adam_names = adam_slot_names(&parameter.name);
            manifest_tensors.push(QwenDeltaTensorManifestEntry {
                name: parameter.name.clone(),
                delta_name: delta_name.clone(),
                adam_m_name: Some(adam_names.m.clone()),
                adam_v_name: Some(adam_names.v.clone()),
                shape: parameter.tensor.size(),
                dtype: "float32".to_string(),
                grad_norm,
                delta_norm,
            });
            delta_entries.push((delta_name, delta));
            optimizer_entries.push((adam_names.m, adam_state.m.shallow_clone()));
            optimizer_entries.push((adam_names.v, adam_state.v.shallow_clone()));
            tensor_summaries.push(TrainableTensorSummary {
                name: parameter.name.clone(),
                grad_defined,
                grad_norm,
                delta_norm,
            });
            parameter.adam = Some(adam_state);
        }

        Ok(QwenTrainStepArtifacts {
            tensor_summaries,
            manifest_tensors,
            delta_entries,
            optimizer_entries,
        })
    }

    pub(crate) fn zero_grad(&mut self) {
        for parameter in &mut self.parameters {
            parameter.tensor.zero_grad();
        }
    }

    pub(crate) fn grad_entries(&self) -> Result<Vec<(String, Tensor)>> {
        let mut entries = Vec::with_capacity(self.parameters.len());
        for parameter in &self.parameters {
            let grad = parameter.tensor.grad();
            if !grad.defined() {
                bail!(
                    "trainable tensor {} did not receive a gradient",
                    parameter.name
                );
            }
            entries.push((parameter.name.clone(), grad.to_kind(Kind::Float)));
        }
        Ok(entries)
    }

    pub(crate) fn parameter_names(&self) -> Vec<String> {
        self.parameters
            .iter()
            .map(|parameter| parameter.name.clone())
            .collect()
    }

    pub(crate) fn apply_delta_checkpoint(
        weights: &mut BTreeMap<String, Tensor>,
        delta_tensors: &BTreeMap<String, Tensor>,
        manifest_tensors: &[QwenDeltaTensorManifestEntry],
    ) -> Result<()> {
        for entry in manifest_tensors {
            let base = tensor(weights, &entry.name)?.to_kind(Kind::Float);
            let delta = tensor(delta_tensors, &entry.delta_name)?
                .to_kind(Kind::Float)
                .to_device(base.device());
            let reloaded = base + delta;
            weights.insert(entry.name.clone(), reloaded);
        }
        Ok(())
    }

    pub(crate) fn load_from_manifest(
        weights: &mut BTreeMap<String, Tensor>,
        manifest: &QwenDeltaCheckpointManifest,
    ) -> Result<Self> {
        if manifest.format != "rustrain.qwen_delta.v1" {
            bail!(
                "unsupported Qwen delta checkpoint format {}",
                manifest.format
            );
        }
        let delta_tensors = read_safetensors_map(Path::new(&manifest.delta_safetensors))?;
        Self::apply_delta_checkpoint(weights, &delta_tensors, &manifest.tensors)?;
        let optimizer_tensors = if let Some(path) = &manifest.optimizer_safetensors {
            Some(read_safetensors_map(Path::new(path))?)
        } else {
            None
        };

        let mut parameters = Vec::with_capacity(manifest.tensors.len());
        for entry in &manifest.tensors {
            let reloaded = tensor(weights, &entry.name)?.to_kind(Kind::Float);
            let delta = tensor(&delta_tensors, &entry.delta_name)?
                .to_kind(Kind::Float)
                .to_device(reloaded.device());
            let base = tensor_snapshot(&(reloaded.shallow_clone() - delta));
            let trainable = reloaded.set_requires_grad(true);
            weights.insert(entry.name.clone(), trainable.shallow_clone());
            let adam = match (
                optimizer_tensors.as_ref(),
                entry.adam_m_name.as_ref(),
                entry.adam_v_name.as_ref(),
            ) {
                (Some(optimizer_tensors), Some(m_name), Some(v_name)) => Some(AdamState {
                    m: tensor(optimizer_tensors, m_name)?
                        .to_kind(Kind::Float)
                        .to_device(trainable.device()),
                    v: tensor(optimizer_tensors, v_name)?
                        .to_kind(Kind::Float)
                        .to_device(trainable.device()),
                }),
                (None, None, None) => None,
                _ => bail!(
                    "incomplete optimizer state for trainable tensor {}",
                    entry.name
                ),
            };
            parameters.push(QwenTrainableParameter {
                name: entry.name.clone(),
                tensor: trainable,
                base,
                adam,
            });
        }

        Ok(Self { parameters })
    }
}

impl QwenTrainableSession {
    pub(crate) fn from_weights(
        config: QwenRuntimeConfig,
        mut weights: BTreeMap<String, Tensor>,
        input_ids: Tensor,
        compute_kind: Kind,
    ) -> Result<Self> {
        if input_ids.size()[1] < 2 {
            bail!("training fixture must contain at least two tokens");
        }
        let registry = QwenTrainableRegistry::representative(&mut weights)?;
        Ok(Self {
            config,
            weights,
            input_ids,
            compute_kind,
            registry,
        })
    }

    pub(crate) fn from_trainable_layers(
        config: QwenRuntimeConfig,
        weights: BTreeMap<String, Tensor>,
        input_ids: Tensor,
        compute_kind: Kind,
        trainable_layers: &[usize],
    ) -> Result<Self> {
        Self::from_names(
            config,
            weights,
            input_ids,
            compute_kind,
            qwen_trainable_tensors_for_layers(trainable_layers, true),
        )
    }

    pub(crate) fn from_names(
        config: QwenRuntimeConfig,
        mut weights: BTreeMap<String, Tensor>,
        input_ids: Tensor,
        compute_kind: Kind,
        names: Vec<String>,
    ) -> Result<Self> {
        if input_ids.size()[1] < 2 {
            bail!("training fixture must contain at least two tokens");
        }
        let registry = QwenTrainableRegistry::from_names(&mut weights, names)?;
        Ok(Self {
            config,
            weights,
            input_ids,
            compute_kind,
            registry,
        })
    }

    pub(crate) fn from_names_on_device(
        config: QwenRuntimeConfig,
        mut weights: BTreeMap<String, Tensor>,
        input_ids: Tensor,
        compute_kind: Kind,
        names: Vec<String>,
        device: Device,
    ) -> Result<Self> {
        if input_ids.size()[1] < 2 {
            bail!("training fixture must contain at least two tokens");
        }
        for tensor in weights.values_mut() {
            *tensor = tensor.to_device(device);
        }
        let registry = QwenTrainableRegistry::from_names_on_device(&mut weights, names, device)?;
        Ok(Self {
            config,
            weights,
            input_ids: input_ids.to_device(device),
            compute_kind,
            registry,
        })
    }

    pub(crate) fn from_trainable_layers_on_device(
        config: QwenRuntimeConfig,
        weights: BTreeMap<String, Tensor>,
        input_ids: Tensor,
        compute_kind: Kind,
        trainable_layers: &[usize],
        include_embed_tokens: bool,
        device: Device,
    ) -> Result<Self> {
        Self::from_names_on_device(
            config,
            weights,
            input_ids,
            compute_kind,
            qwen_trainable_tensors_for_layers(trainable_layers, include_embed_tokens),
            device,
        )
    }

    pub(crate) fn from_manifest(
        config: QwenRuntimeConfig,
        weights: BTreeMap<String, Tensor>,
        input_ids: Tensor,
        compute_kind: Kind,
        manifest: &QwenDeltaCheckpointManifest,
    ) -> Result<Self> {
        Self::from_manifest_on_device(config, weights, input_ids, compute_kind, manifest, None)
    }

    pub(crate) fn from_manifest_on_device(
        config: QwenRuntimeConfig,
        mut weights: BTreeMap<String, Tensor>,
        input_ids: Tensor,
        compute_kind: Kind,
        manifest: &QwenDeltaCheckpointManifest,
        device: Option<Device>,
    ) -> Result<Self> {
        if input_ids.size()[1] < 2 {
            bail!("training fixture must contain at least two tokens");
        }
        if let Some(device) = device {
            for tensor in weights.values_mut() {
                *tensor = tensor.to_device(device);
            }
        }
        let registry = QwenTrainableRegistry::load_from_manifest(&mut weights, manifest)?;
        Ok(Self {
            config,
            weights,
            input_ids: match device {
                Some(device) => input_ids.to_device(device),
                None => input_ids,
            },
            compute_kind,
            registry,
        })
    }

    pub(crate) fn loss_value(&self) -> Result<f64> {
        Ok(qwen_causal_lm_loss_with_kind(
            &self.input_ids,
            &self.weights,
            &self.config,
            self.compute_kind,
        )?
        .double_value(&[]))
    }

    pub(crate) fn loss_and_backward(&mut self) -> Result<f64> {
        self.registry.zero_grad();
        let loss = qwen_causal_lm_loss_with_kind(
            &self.input_ids,
            &self.weights,
            &self.config,
            self.compute_kind,
        )?;
        let loss_value = loss.double_value(&[]);
        loss.backward();
        Ok(loss_value)
    }

    pub(crate) fn grad_entries(&self) -> Result<Vec<(String, Tensor)>> {
        self.registry.grad_entries()
    }

    pub(crate) fn parameter_names(&self) -> Vec<String> {
        self.registry.parameter_names()
    }

    pub(crate) fn set_input_ids(&mut self, input_ids: &Tensor) {
        self.input_ids = input_ids.to_device(self.input_ids.device());
    }

    pub(crate) fn all_reduce_average_grads(
        &self,
        output_dir: &Path,
        world_size: usize,
    ) -> Result<Vec<(String, Tensor)>> {
        let mut averaged = Vec::new();
        for (index, (name, grad)) in self.grad_entries()?.into_iter().enumerate() {
            let reduced = nccl_smoke::all_reduce_tensor_f32_for_launch(
                &output_dir.join(format!("grad-{index}")),
                &grad,
            )?;
            averaged.push((name, reduced / world_size as f64));
        }
        Ok(averaged)
    }

    pub(crate) fn apply_adamw_step(
        &mut self,
        averaged_grads: &[(String, Tensor)],
        learning_rate: f64,
        step: i32,
    ) -> Result<QwenTrainStepArtifacts> {
        self.registry
            .adamw_step_with_grads(&mut self.weights, averaged_grads, learning_rate, step)
    }

    pub(crate) fn train_step(
        &mut self,
        learning_rate: f64,
        step: i32,
    ) -> Result<QwenTrainStepResult> {
        let loss_before = self.loss_and_backward()?;
        let artifacts = self
            .registry
            .adamw_step(&mut self.weights, learning_rate, step)?;
        let loss_after = self.loss_value()?;
        Ok(QwenTrainStepResult {
            loss_before,
            loss_after,
            artifacts,
        })
    }
}

impl QwenAttentionDpSession {
    pub(crate) fn from_weights(
        weights: BTreeMap<String, Tensor>,
        input: Tensor,
        target: Tensor,
        config: QwenRuntimeConfig,
        compute_kind: Kind,
        device: Device,
    ) -> Result<Self> {
        let q_proj = tensor(&weights, "model.layers.0.self_attn.q_proj.weight")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        let q_bias = tensor(&weights, "model.layers.0.self_attn.q_proj.bias")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        let k_proj = tensor(&weights, "model.layers.0.self_attn.k_proj.weight")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        let k_bias = tensor(&weights, "model.layers.0.self_attn.k_proj.bias")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        let v_proj = tensor(&weights, "model.layers.0.self_attn.v_proj.weight")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        let v_bias = tensor(&weights, "model.layers.0.self_attn.v_proj.bias")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        let o_proj = tensor(&weights, "model.layers.0.self_attn.o_proj.weight")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        Ok(Self {
            config,
            input: input.to_kind(compute_kind).to_device(device),
            target: target.to_kind(compute_kind).to_device(device),
            q_proj,
            q_bias,
            k_proj,
            k_bias,
            v_proj,
            v_bias,
            o_proj,
            compute_kind,
        })
    }

    pub(crate) fn loss_and_backward(&mut self) -> Result<f64> {
        for (_, parameter) in self.parameters_mut() {
            parameter.zero_grad();
        }
        let loss = self.loss_tensor();
        let loss_value = loss.double_value(&[]);
        loss.backward();
        Ok(loss_value)
    }

    pub(crate) fn loss_value(&self) -> f64 {
        self.loss_tensor().double_value(&[])
    }

    pub(crate) fn loss_tensor(&self) -> Tensor {
        let output = qwen_attention(
            &self.input,
            &self.q_proj.to_kind(self.compute_kind),
            &self.q_bias.to_kind(self.compute_kind),
            &self.k_proj.to_kind(self.compute_kind),
            &self.k_bias.to_kind(self.compute_kind),
            &self.v_proj.to_kind(self.compute_kind),
            &self.v_bias.to_kind(self.compute_kind),
            &self.o_proj.to_kind(self.compute_kind),
            &self.config,
        );
        output.mse_loss(&self.target, Reduction::Mean)
    }

    pub(crate) fn all_reduce_average_grads(
        &self,
        output_dir: &Path,
        world_size: usize,
    ) -> Result<Vec<Tensor>> {
        let mut averaged = Vec::new();
        for (index, (name, parameter)) in self.parameters().iter().enumerate() {
            let grad = parameter.grad();
            if !grad.defined() {
                bail!("trainable tensor {name} did not receive a gradient");
            }
            let reduced = nccl_smoke::all_reduce_tensor_f32_for_launch(
                &output_dir.join(format!("grad-{index}")),
                &grad,
            )?;
            averaged.push(reduced / world_size as f64);
        }
        Ok(averaged)
    }

    pub(crate) fn apply_sgd_step(
        &mut self,
        averaged_grads: &[Tensor],
        learning_rate: f64,
    ) -> Result<()> {
        let mut parameters = self.parameters_mut();
        if averaged_grads.len() != parameters.len() {
            bail!(
                "averaged gradient count mismatch: got {}, expected {}",
                averaged_grads.len(),
                parameters.len()
            );
        }
        for ((_, parameter), grad) in parameters.iter_mut().zip(averaged_grads.iter()) {
            let update = grad.to_device(parameter.device()) * learning_rate;
            let _ = no_grad(|| parameter.f_sub_(&update))?;
        }
        Ok(())
    }

    pub(crate) fn grad_entries(&self) -> Result<Vec<(String, Tensor)>> {
        let mut entries = Vec::new();
        for (name, parameter) in self.parameters() {
            let grad = parameter.grad();
            if !grad.defined() {
                bail!("trainable tensor {name} did not receive a gradient");
            }
            entries.push((name.to_string(), grad.to_kind(Kind::Float)));
        }
        Ok(entries)
    }

    pub(crate) fn parameters(&self) -> [(&'static str, &Tensor); 7] {
        [
            ("model.layers.0.self_attn.q_proj.weight", &self.q_proj),
            ("model.layers.0.self_attn.q_proj.bias", &self.q_bias),
            ("model.layers.0.self_attn.k_proj.weight", &self.k_proj),
            ("model.layers.0.self_attn.k_proj.bias", &self.k_bias),
            ("model.layers.0.self_attn.v_proj.weight", &self.v_proj),
            ("model.layers.0.self_attn.v_proj.bias", &self.v_bias),
            ("model.layers.0.self_attn.o_proj.weight", &self.o_proj),
        ]
    }

    pub(crate) fn parameters_mut(&mut self) -> [(&'static str, &mut Tensor); 7] {
        [
            ("model.layers.0.self_attn.q_proj.weight", &mut self.q_proj),
            ("model.layers.0.self_attn.q_proj.bias", &mut self.q_bias),
            ("model.layers.0.self_attn.k_proj.weight", &mut self.k_proj),
            ("model.layers.0.self_attn.k_proj.bias", &mut self.k_bias),
            ("model.layers.0.self_attn.v_proj.weight", &mut self.v_proj),
            ("model.layers.0.self_attn.v_proj.bias", &mut self.v_bias),
            ("model.layers.0.self_attn.o_proj.weight", &mut self.o_proj),
        ]
    }
}

pub(crate) fn adamw_next_state(
    previous: Option<&AdamState>,
    grad: &Tensor,
    beta1: f64,
    beta2: f64,
) -> AdamState {
    let m = if let Some(previous) = previous {
        &previous.m * beta1 + grad * (1.0 - beta1)
    } else {
        grad * (1.0 - beta1)
    };
    let grad_sq = grad.pow_tensor_scalar(2.0);
    let v = if let Some(previous) = previous {
        &previous.v * beta2 + grad_sq * (1.0 - beta2)
    } else {
        grad_sq * (1.0 - beta2)
    };
    AdamState { m, v }
}

pub(crate) fn qwen_dp_attention_global(input: &Tensor) -> Result<Tensor> {
    if input.size().len() != 3 || input.size()[0] != 1 || input.size()[1] < 2 {
        bail!("Qwen attention DP fixture expects shape [1, seq_len>=2, hidden]");
    }
    let reversed = input.flip([1]);
    Ok(Tensor::cat(&[input.shallow_clone(), reversed], 0))
}

pub(crate) fn qwen_dp_attention_input_for_rank(
    input: &Tensor,
    rank: usize,
    world_size: usize,
) -> Result<Tensor> {
    if world_size != 2 {
        bail!("Qwen attention DP fixture currently expects world_size=2");
    }
    let global = qwen_dp_attention_global(input)?;
    Ok(global.narrow(0, rank as i64, 1))
}

pub(crate) fn qwen_dp_attention_target_for_rank(
    target: &Tensor,
    rank: usize,
    world_size: usize,
) -> Result<Tensor> {
    qwen_dp_attention_input_for_rank(target, rank, world_size)
}

pub(crate) fn grad_signatures(grads: &[(String, Tensor)]) -> Result<Vec<QwenGradSignature>> {
    grads
        .iter()
        .map(|(name, grad)| grad_signature(name, grad))
        .collect()
}

pub(crate) fn grad_signature(name: &str, grad: &Tensor) -> Result<QwenGradSignature> {
    let shape = grad.size();
    let flat = grad.to_kind(Kind::Float).reshape([-1]);
    let numel = flat.numel();
    if numel == 0 {
        bail!("gradient tensor {name} is empty");
    }
    let sample_count = numel.min(16);
    let stride = (numel / sample_count).max(1);
    let samples = (0..sample_count)
        .map(|index| flat.double_value(&[((index * stride).min(numel - 1)) as i64]) as f32)
        .collect();
    Ok(QwenGradSignature {
        name: name.to_string(),
        shape,
        samples,
    })
}

impl QwenGradSignature {
    pub(crate) fn values(&self) -> Vec<f32> {
        self.samples.clone()
    }
}

pub(crate) fn signature_values_max_delta(actual: &[f32], expected: &[f32]) -> Result<f32> {
    if actual.len() != expected.len() {
        bail!(
            "gradient signature length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        );
    }
    Ok(actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max))
}

pub(crate) fn wait_for_expected_signatures(
    path: &Path,
    timeout: Duration,
) -> Result<(f64, Vec<QwenGradSignature>)> {
    let start = Instant::now();
    loop {
        match fs::read_to_string(path) {
            Ok(contents) => {
                return serde_json::from_str(&contents)
                    .with_context(|| format!("failed to parse {}", path.display()));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if start.elapsed() > timeout {
                    bail!("timed out waiting for {}", path.display());
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(error) => {
                return Err(error).with_context(|| format!("failed to read {}", path.display()));
            }
        }
    }
}

pub(crate) fn wait_for_rank_barrier(
    dir: &Path,
    rank: usize,
    world_size: usize,
    timeout: Duration,
) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let ready_path = dir.join(format!("rank-{rank}.ready"));
    fs::write(&ready_path, b"ready")
        .with_context(|| format!("failed to write {}", ready_path.display()))?;
    let start = Instant::now();
    loop {
        let all_ready = (0..world_size).all(|rank| dir.join(format!("rank-{rank}.ready")).exists());
        if all_ready {
            return Ok(());
        }
        if start.elapsed() > timeout {
            bail!("timed out waiting for barrier {}", dir.display());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

pub(crate) fn wait_for_rank_barrier_or_error(
    dir: &Path,
    rank: usize,
    world_size: usize,
    timeout: Duration,
    error_path: &Path,
) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let ready_path = dir.join(format!("rank-{rank}.ready"));
    fs::write(&ready_path, b"ready")
        .with_context(|| format!("failed to write {}", ready_path.display()))?;
    let start = Instant::now();
    loop {
        if error_path.exists() {
            let error_text = fs::read_to_string(error_path)
                .unwrap_or_else(|_| format!("failed to read {}", error_path.display()));
            bail!("{}", error_text.trim());
        }
        let all_ready = (0..world_size).all(|rank| dir.join(format!("rank-{rank}.ready")).exists());
        if all_ready {
            return Ok(());
        }
        if start.elapsed() > timeout {
            bail!("timed out waiting for barrier {}", dir.display());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

pub(crate) fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set; run through rustrain launch"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

pub(crate) fn adamw_update(
    state: &AdamState,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    step: i32,
    eps: f64,
) -> Tensor {
    let m_hat = &state.m / (1.0 - beta1.powi(step));
    let v_hat = &state.v / (1.0 - beta2.powi(step));
    (m_hat / v_hat.sqrt().g_add_scalar(eps)) * learning_rate
}

pub(crate) fn representative_trainable_qwen_tensors() -> Vec<String> {
    qwen_trainable_tensors_for_layers(&[0], true)
}

pub(crate) fn qwen_session_default_trainable_layers() -> Vec<usize> {
    vec![0]
}

pub(crate) fn qwen_session_trainable_layers_from_config(config: &Config) -> Vec<usize> {
    config
        .model
        .trainable_layers
        .clone()
        .unwrap_or_else(qwen_session_default_trainable_layers)
}

pub(crate) fn qwen_trainable_tensors_for_layers(
    trainable_layers: &[usize],
    include_embed_tokens: bool,
) -> Vec<String> {
    let mut names = Vec::new();
    if include_embed_tokens {
        names.push("model.embed_tokens.weight".to_string());
    }
    for layer in trainable_layers {
        let prefix = format!("model.layers.{layer}");
        names.extend([
            format!("{prefix}.input_layernorm.weight"),
            format!("{prefix}.self_attn.q_proj.weight"),
            format!("{prefix}.self_attn.q_proj.bias"),
            format!("{prefix}.self_attn.k_proj.weight"),
            format!("{prefix}.self_attn.k_proj.bias"),
            format!("{prefix}.self_attn.v_proj.weight"),
            format!("{prefix}.self_attn.v_proj.bias"),
            format!("{prefix}.self_attn.o_proj.weight"),
            format!("{prefix}.post_attention_layernorm.weight"),
            format!("{prefix}.mlp.gate_proj.weight"),
            format!("{prefix}.mlp.up_proj.weight"),
            format!("{prefix}.mlp.down_proj.weight"),
        ]);
    }
    names.push("model.norm.weight".to_string());
    names
}

pub(crate) fn qwen_session_dp_global_input(
    weights: &BTreeMap<String, Tensor>,
    device: Device,
) -> Result<Tensor> {
    let vocab_size = tensor(weights, "model.embed_tokens.weight")?.size()[0];
    if vocab_size < 2048 {
        bail!("Qwen session DP smoke expects vocab_size >= 2048, got {vocab_size}");
    }
    Ok(
        Tensor::from_slice(&[101_i64, 872, 198, 3838, 645, 211, 777, 198, 1339, 899])
            .reshape([2, 5])
            .to_kind(Kind::Int64)
            .to_device(device),
    )
}

pub(crate) fn qwen_session_fixed_batch_plan(
    weights: &BTreeMap<String, Tensor>,
    data_cursor_start: usize,
    train_steps: usize,
) -> Result<QwenSessionBatchPlan> {
    let input_ids = qwen_session_dp_global_input(weights, Device::Cpu)?.narrow(0, 0, 1);
    let required_batches = data_cursor_start + train_steps + 1;
    let train_batches = (0..required_batches)
        .map(|_| input_ids.shallow_clone())
        .collect();
    Ok(QwenSessionBatchPlan {
        initial_input_ids: input_ids.shallow_clone(),
        train_batches,
        reference_fixture: "qwen_session_single_fixed_tokens".to_string(),
        dataset_total_samples: None,
        dataset_total_tokens: None,
        dataset_train_samples: None,
        dataset_eval_samples: None,
        dataset_source_files: None,
        dataset_source_sample_counts: None,
        dataset_fingerprint: None,
        dataset_order_seed: None,
        dataset_shuffle: None,
        streaming_train_batches: None,
        streaming_index_cache_path: None,
        streaming_index_cache_hit: None,
        streaming_index_cache_written: None,
        train_sample_count: None,
        data_epoch_start: None,
        data_epoch_end: None,
        data_epoch_next: None,
        data_sample_offset_start: None,
        data_sample_offset_end: None,
        data_sample_offset_next: None,
        batch_size: 1,
        sequence_tokens: 5,
    })
}

pub(crate) fn qwen_session_batch_plan_from_config(
    model_path: &Path,
    weights: &BTreeMap<String, Tensor>,
    data_cursor_start: usize,
    train_steps: usize,
    runtime_config: Option<&Config>,
    streaming_index_cache: Option<&Path>,
) -> Result<QwenSessionBatchPlan> {
    let Some(runtime_config) = runtime_config else {
        return qwen_session_fixed_batch_plan(weights, data_cursor_start, train_steps);
    };
    let Some(data_config) = runtime_config.data.as_ref() else {
        return qwen_session_fixed_batch_plan(weights, data_cursor_start, train_steps);
    };
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let field_map = QwenSftFieldMap::from_runtime_data(data_config)?;
    if data_config.kind == RuntimeDataKind::InstructionArrow {
        let streaming_plan = qwen_sft_arrow_streaming_dataset_plan(
            &tokenizer,
            data_config,
            runtime_config.run.seed,
            data_cursor_start,
            train_steps,
            runtime_config.train.micro_batch_size,
            1,
            streaming_index_cache,
            &field_map,
        )?;
        return Ok(QwenSessionBatchPlan {
            sequence_tokens: streaming_plan.sequence_tokens,
            initial_input_ids: streaming_plan.initial_input_ids,
            train_batches: streaming_plan.train_batches,
            reference_fixture: "qwen_session_single_arrow_streaming".to_string(),
            dataset_total_samples: Some(streaming_plan.dataset_total_samples),
            dataset_total_tokens: None,
            dataset_train_samples: Some(streaming_plan.dataset_train_samples),
            dataset_eval_samples: Some(streaming_plan.dataset_eval_samples),
            dataset_source_files: Some(streaming_plan.dataset_source_files),
            dataset_source_sample_counts: Some(streaming_plan.dataset_source_sample_counts),
            dataset_fingerprint: Some(streaming_plan.dataset_fingerprint),
            dataset_order_seed: Some(runtime_config.run.seed),
            dataset_shuffle: Some(streaming_plan.dataset_shuffle),
            streaming_train_batches: Some(true),
            streaming_index_cache_path: streaming_index_cache
                .map(|path| path.display().to_string()),
            streaming_index_cache_hit: Some(streaming_plan.streaming_index_cache_hit),
            streaming_index_cache_written: Some(streaming_plan.streaming_index_cache_written),
            train_sample_count: Some(streaming_plan.dataset_train_samples),
            data_epoch_start: Some(streaming_plan.data_epoch_start),
            data_epoch_end: Some(streaming_plan.data_epoch_end),
            data_epoch_next: Some(streaming_plan.data_epoch_next),
            data_sample_offset_start: Some(streaming_plan.data_sample_offset_start),
            data_sample_offset_end: Some(streaming_plan.data_sample_offset_end),
            data_sample_offset_next: Some(streaming_plan.data_sample_offset_next),
            batch_size: streaming_plan.local_batch_size,
        });
    }
    let datasets = qwen_session_single_sft_datasets_from_config(
        &tokenizer,
        data_config,
        runtime_config.run.seed,
        &field_map,
    )?;
    let dataset_summary = datasets.combined_summary;
    let train_dataset = datasets.train_dataset;
    let eval_dataset = datasets.eval_dataset;
    let batch_size = runtime_config
        .train
        .micro_batch_size
        .min(train_dataset.len())
        .max(1);
    let required_batches = train_steps * batch_size + 1;
    let data_cursor_end = data_cursor_start + train_steps * batch_size;
    let data_cursor_next = data_cursor_end;
    let (data_epoch_start, data_sample_offset_start) =
        qwen_data_epoch_and_offset(data_cursor_start, train_dataset.len())?;
    let (data_epoch_end, data_sample_offset_end) =
        qwen_data_epoch_and_offset(data_cursor_end, train_dataset.len())?;
    let (data_epoch_next, data_sample_offset_next) =
        qwen_data_epoch_and_offset(data_cursor_next, train_dataset.len())?;
    let window_samples = required_batches + batch_size - 1;
    let (streaming_samples, streaming_index_cache_hit, streaming_index_cache_written) =
        match data_config.kind {
            RuntimeDataKind::InstructionJsonl => {
                let streaming_window = qwen_sft_streaming_token_window_from_jsonl(
                    &tokenizer,
                    &data_config.paths,
                    &data_config.eval_paths,
                    data_config.max_samples,
                    data_config.train_split,
                    data_config.shuffle,
                    runtime_config.run.seed,
                    data_cursor_start,
                    window_samples,
                    streaming_index_cache,
                    &field_map,
                )?;
                (
                    streaming_window.samples,
                    streaming_window.source_index_cache_hit,
                    streaming_window.source_index_cache_written,
                )
            }
            RuntimeDataKind::InstructionArrow => {
                let streaming_window = qwen_sft_arrow_streaming_token_window(
                    &tokenizer,
                    &data_config.paths,
                    &data_config.eval_paths,
                    data_config.max_samples,
                    data_config.train_split,
                    data_config.shuffle,
                    runtime_config.run.seed,
                    data_cursor_start,
                    window_samples,
                    streaming_index_cache,
                    &field_map,
                )?;
                (
                    streaming_window.samples,
                    streaming_window.source_index_cache_hit,
                    streaming_window.source_index_cache_written,
                )
            }
            _ => bail!(
                "qwen trainable session data path supports kind = instruction_jsonl or instruction_arrow"
            ),
        };
    let train_batches = (0..required_batches)
        .map(|relative_cursor| {
            let end = relative_cursor + batch_size;
            let streaming_batch = qwen_sft_padded_batch(
                &streaming_samples[relative_cursor..end],
                train_dataset.pad_token_id,
            )?;
            let reference_batch =
                train_dataset.padded_batch(data_cursor_start + relative_cursor, batch_size)?;
            let input_delta =
                tensor_i64_max_abs_diff(&streaming_batch.input_ids, &reference_batch.input_ids)?;
            let mask_delta =
                tensor_max_abs_diff(&streaming_batch.target_mask, &reference_batch.target_mask)?;
            if input_delta != 0 || mask_delta > 0.0 {
                bail!(
                    "Qwen session streaming batch mismatch at cursor {}: input_delta={}, mask_delta={}",
                    data_cursor_start + relative_cursor,
                    input_delta,
                    mask_delta
                );
            }
            Ok(streaming_batch.input_ids)
        })
        .collect::<Result<Vec<_>>>()?;
    let initial_input_ids = train_batches
        .first()
        .ok_or_else(|| anyhow!("qwen trainable session batch plan produced no batches"))?
        .shallow_clone();
    Ok(QwenSessionBatchPlan {
        sequence_tokens: initial_input_ids.size()[1] as usize,
        initial_input_ids,
        train_batches,
        reference_fixture: "qwen_session_single_jsonl".to_string(),
        dataset_total_samples: Some(dataset_summary.samples),
        dataset_total_tokens: Some(dataset_summary.total_tokens),
        dataset_train_samples: Some(train_dataset.len()),
        dataset_eval_samples: Some(eval_dataset.len()),
        dataset_source_files: Some(dataset_summary.source_files),
        dataset_source_sample_counts: Some(dataset_summary.source_sample_counts),
        dataset_fingerprint: Some(dataset_summary.fingerprint),
        dataset_order_seed: Some(runtime_config.run.seed),
        dataset_shuffle: Some(dataset_summary.shuffle),
        streaming_train_batches: Some(true),
        streaming_index_cache_path: match data_config.kind {
            RuntimeDataKind::InstructionJsonl => {
                streaming_index_cache.map(|path| path.display().to_string())
            }
            RuntimeDataKind::InstructionArrow => {
                streaming_index_cache.map(|path| path.display().to_string())
            }
            _ => None,
        },
        streaming_index_cache_hit: Some(streaming_index_cache_hit),
        streaming_index_cache_written: Some(streaming_index_cache_written),
        train_sample_count: Some(train_dataset.len()),
        data_epoch_start: Some(data_epoch_start),
        data_epoch_end: Some(data_epoch_end),
        data_epoch_next: Some(data_epoch_next),
        data_sample_offset_start: Some(data_sample_offset_start),
        data_sample_offset_end: Some(data_sample_offset_end),
        data_sample_offset_next: Some(data_sample_offset_next),
        batch_size,
    })
}

pub(crate) fn qwen_session_single_sft_datasets_from_config(
    tokenizer: &Tokenizer,
    data_config: &RuntimeDataConfig,
    seed: u64,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftTrainEvalDatasets> {
    match data_config.kind {
        RuntimeDataKind::InstructionJsonl => qwen_sft_train_eval_datasets_from_paths(
            tokenizer,
            &data_config.paths,
            &data_config.eval_paths,
            data_config.max_samples,
            data_config.max_eval_samples,
            data_config.train_split,
            data_config.shuffle,
            seed,
            field_map,
        ),
        RuntimeDataKind::InstructionArrow => {
            qwen_sft_arrow_validate_config_scope(
                data_config,
                "qwen trainable session instruction_arrow data path",
            )?;
            if data_config.max_samples.is_none() {
                bail!(
                    "qwen trainable session instruction_arrow requires data.max_samples for the bounded trainer path"
                );
            }
            let plan_data = qwen_sft_arrow_plan_data(tokenizer, data_config, seed, field_map)?;
            Ok(QwenSftTrainEvalDatasets {
                combined_summary: plan_data.dataset_summary,
                train_dataset: plan_data.train_dataset,
                eval_dataset: plan_data.eval_dataset,
            })
        }
        _ => bail!(
            "qwen trainable session data path supports kind = instruction_jsonl or instruction_arrow"
        ),
    }
}

pub(crate) fn qwen_session_fixed_dp_batch_plan(
    weights: &BTreeMap<String, Tensor>,
    device: Device,
    train_steps: usize,
) -> Result<QwenSessionDpBatchPlan> {
    let global_input = qwen_session_dp_global_input(weights, device)?;
    let global_train_batches = (0..train_steps + 2)
        .map(|_| global_input.shallow_clone())
        .collect();
    Ok(QwenSessionDpBatchPlan {
        global_initial_input_ids: global_input.shallow_clone(),
        global_train_batches,
        data_kind: None,
        dataset_total_samples: None,
        dataset_total_tokens: None,
        dataset_train_samples: None,
        dataset_eval_samples: None,
        dataset_source_files: None,
        dataset_source_sample_counts: None,
        dataset_fingerprint: None,
        dataset_order_seed: None,
        dataset_shuffle: None,
        streaming_train_batches: None,
        streaming_index_cache_path: None,
        streaming_index_cache_hit: None,
        streaming_index_cache_written: None,
        train_sample_count: None,
        data_epoch_start: None,
        data_epoch_end: None,
        data_epoch_next: None,
        data_sample_offset_start: None,
        data_sample_offset_end: None,
        data_sample_offset_next: None,
        local_batch_size: 1,
        sequence_tokens: 5,
    })
}

pub(crate) fn qwen_session_dp_batch_plan_from_config(
    model_path: &Path,
    weights: &BTreeMap<String, Tensor>,
    data_cursor_start: usize,
    train_steps: usize,
    world_size: usize,
    device: Device,
    runtime_config: Option<&Config>,
    streaming_index_cache: Option<&Path>,
) -> Result<QwenSessionDpBatchPlan> {
    let Some(runtime_config) = runtime_config else {
        return qwen_session_fixed_dp_batch_plan(weights, device, train_steps);
    };
    let Some(data_config) = runtime_config.data.as_ref() else {
        return qwen_session_fixed_dp_batch_plan(weights, device, train_steps);
    };
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let field_map = QwenSftFieldMap::from_runtime_data(data_config)?;
    if data_config.kind == RuntimeDataKind::InstructionArrow {
        let streaming_plan = qwen_sft_arrow_streaming_dataset_plan(
            &tokenizer,
            data_config,
            runtime_config.run.seed,
            data_cursor_start,
            train_steps,
            runtime_config.train.micro_batch_size,
            world_size,
            streaming_index_cache,
            &field_map,
        )?;
        let global_train_batches = streaming_plan
            .train_batches
            .into_iter()
            .map(|batch| batch.to_device(device))
            .collect::<Vec<_>>();
        return Ok(QwenSessionDpBatchPlan {
            sequence_tokens: streaming_plan.sequence_tokens,
            global_initial_input_ids: streaming_plan.initial_input_ids.to_device(device),
            global_train_batches,
            data_kind: Some("instruction_arrow".to_string()),
            dataset_total_samples: Some(streaming_plan.dataset_total_samples),
            dataset_total_tokens: None,
            dataset_train_samples: Some(streaming_plan.dataset_train_samples),
            dataset_eval_samples: Some(streaming_plan.dataset_eval_samples),
            dataset_source_files: Some(streaming_plan.dataset_source_files),
            dataset_source_sample_counts: Some(streaming_plan.dataset_source_sample_counts),
            dataset_fingerprint: Some(streaming_plan.dataset_fingerprint),
            dataset_order_seed: Some(runtime_config.run.seed),
            dataset_shuffle: Some(streaming_plan.dataset_shuffle),
            streaming_train_batches: Some(true),
            streaming_index_cache_path: streaming_index_cache
                .map(|path| path.display().to_string()),
            streaming_index_cache_hit: Some(streaming_plan.streaming_index_cache_hit),
            streaming_index_cache_written: Some(streaming_plan.streaming_index_cache_written),
            train_sample_count: Some(streaming_plan.dataset_train_samples),
            data_epoch_start: Some(streaming_plan.data_epoch_start),
            data_epoch_end: Some(streaming_plan.data_epoch_end),
            data_epoch_next: Some(streaming_plan.data_epoch_next),
            data_sample_offset_start: Some(streaming_plan.data_sample_offset_start),
            data_sample_offset_end: Some(streaming_plan.data_sample_offset_end),
            data_sample_offset_next: Some(streaming_plan.data_sample_offset_next),
            local_batch_size: streaming_plan.local_batch_size,
        });
    }
    let datasets = qwen_session_single_sft_datasets_from_config(
        &tokenizer,
        data_config,
        runtime_config.run.seed,
        &field_map,
    )?;
    let dataset_summary = datasets.combined_summary;
    let train_dataset = datasets.train_dataset;
    let eval_dataset = datasets.eval_dataset;
    let local_batch_size = runtime_config
        .train
        .micro_batch_size
        .min(train_dataset.len())
        .max(1);
    let global_batch_size = local_batch_size * world_size;
    let required_batches = train_steps * global_batch_size + 1;
    let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
    let data_cursor_next = data_cursor_end;
    let (data_epoch_start, data_sample_offset_start) =
        qwen_data_epoch_and_offset(data_cursor_start, train_dataset.len())?;
    let (data_epoch_end, data_sample_offset_end) =
        qwen_data_epoch_and_offset(data_cursor_end, train_dataset.len())?;
    let (data_epoch_next, data_sample_offset_next) =
        qwen_data_epoch_and_offset(data_cursor_next, train_dataset.len())?;
    let window_samples = required_batches + global_batch_size - 1;
    let (streaming_samples, streaming_index_cache_hit, streaming_index_cache_written) =
        match data_config.kind {
            RuntimeDataKind::InstructionJsonl => {
                let streaming_window = qwen_sft_streaming_token_window_from_jsonl(
                    &tokenizer,
                    &data_config.paths,
                    &data_config.eval_paths,
                    data_config.max_samples,
                    data_config.train_split,
                    data_config.shuffle,
                    runtime_config.run.seed,
                    data_cursor_start,
                    window_samples,
                    streaming_index_cache,
                    &field_map,
                )?;
                (
                    streaming_window.samples,
                    streaming_window.source_index_cache_hit,
                    streaming_window.source_index_cache_written,
                )
            }
            RuntimeDataKind::InstructionArrow => {
                let streaming_window = qwen_sft_arrow_streaming_token_window(
                    &tokenizer,
                    &data_config.paths,
                    &data_config.eval_paths,
                    data_config.max_samples,
                    data_config.train_split,
                    data_config.shuffle,
                    runtime_config.run.seed,
                    data_cursor_start,
                    window_samples,
                    streaming_index_cache,
                    &field_map,
                )?;
                (
                    streaming_window.samples,
                    streaming_window.source_index_cache_hit,
                    streaming_window.source_index_cache_written,
                )
            }
            _ => bail!(
                "qwen trainable session DP data path supports kind = instruction_jsonl or instruction_arrow"
            ),
        };
    let global_train_batches = (0..required_batches)
        .map(|relative_cursor| {
            let end = relative_cursor + global_batch_size;
            let streaming_batch = qwen_sft_padded_batch(
                &streaming_samples[relative_cursor..end],
                train_dataset.pad_token_id,
            )?;
            let reference_batch =
                train_dataset.padded_batch(data_cursor_start + relative_cursor, global_batch_size)?;
            let input_delta =
                tensor_i64_max_abs_diff(&streaming_batch.input_ids, &reference_batch.input_ids)?;
            let mask_delta =
                tensor_max_abs_diff(&streaming_batch.target_mask, &reference_batch.target_mask)?;
            if input_delta != 0 || mask_delta > 0.0 {
                bail!(
                    "Qwen session DP streaming batch mismatch at cursor {}: input_delta={}, mask_delta={}",
                    data_cursor_start + relative_cursor,
                    input_delta,
                    mask_delta
                );
            }
            Ok(streaming_batch.input_ids.to_device(device))
        })
        .collect::<Result<Vec<_>>>()?;
    let global_initial_input_ids = global_train_batches
        .first()
        .ok_or_else(|| anyhow!("qwen trainable session DP batch plan produced no batches"))?
        .shallow_clone();
    Ok(QwenSessionDpBatchPlan {
        sequence_tokens: global_initial_input_ids.size()[1] as usize,
        global_initial_input_ids,
        global_train_batches,
        data_kind: Some(
            match data_config.kind {
                RuntimeDataKind::InstructionJsonl => "instruction_jsonl",
                RuntimeDataKind::InstructionArrow => "instruction_arrow",
                _ => "other",
            }
            .to_string(),
        ),
        dataset_total_samples: Some(dataset_summary.samples),
        dataset_total_tokens: Some(dataset_summary.total_tokens),
        dataset_train_samples: Some(train_dataset.len()),
        dataset_eval_samples: Some(eval_dataset.len()),
        dataset_source_files: Some(dataset_summary.source_files),
        dataset_source_sample_counts: Some(dataset_summary.source_sample_counts),
        dataset_fingerprint: Some(dataset_summary.fingerprint),
        dataset_order_seed: Some(runtime_config.run.seed),
        dataset_shuffle: Some(dataset_summary.shuffle),
        streaming_train_batches: Some(true),
        streaming_index_cache_path: match data_config.kind {
            RuntimeDataKind::InstructionJsonl => {
                streaming_index_cache.map(|path| path.display().to_string())
            }
            RuntimeDataKind::InstructionArrow => {
                streaming_index_cache.map(|path| path.display().to_string())
            }
            _ => None,
        },
        streaming_index_cache_hit: Some(streaming_index_cache_hit),
        streaming_index_cache_written: Some(streaming_index_cache_written),
        train_sample_count: Some(train_dataset.len()),
        data_epoch_start: Some(data_epoch_start),
        data_epoch_end: Some(data_epoch_end),
        data_epoch_next: Some(data_epoch_next),
        data_sample_offset_start: Some(data_sample_offset_start),
        data_sample_offset_end: Some(data_sample_offset_end),
        data_sample_offset_next: Some(data_sample_offset_next),
        local_batch_size,
    })
}
