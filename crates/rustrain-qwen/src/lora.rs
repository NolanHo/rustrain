//! lora module - split from qwen_module.rs

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
use crate::model::*;
use crate::rank_smoke::*;
use crate::session::*;
use crate::sft::*;
pub struct QwenLoraSftTrainSummary {
    pub model_path: String,
    pub adapter_output: String,
    pub adapter_manifest_output: String,
    pub compute_kind: String,
    pub step_adapter_checkpoints: Vec<String>,
    pub resume_from: Option<String>,
    pub resumed_adapter: bool,
    pub target_layers: Vec<usize>,
    pub target_modules: Vec<String>,
    pub train_samples: usize,
    pub eval_samples: usize,
    pub dataset_total_samples: usize,
    pub dataset_total_tokens: usize,
    pub dataset_response_tokens: usize,
    pub dataset_masked_positions: usize,
    pub dataset_max_sequence_tokens: usize,
    pub dataset_source_files: Vec<String>,
    pub dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub dataset_fingerprint: String,
    pub dataset_order_seed: u64,
    pub dataset_shuffle: bool,
    pub streaming_train_batches: bool,
    pub streaming_index_cache_path: Option<String>,
    pub streaming_index_cache_hit: bool,
    pub streaming_index_cache_written: bool,
    pub data_cursor_start: usize,
    pub data_cursor_end: usize,
    pub data_cursor_next: usize,
    pub data_epoch_start: usize,
    pub data_epoch_end: usize,
    pub data_epoch_next: usize,
    pub data_sample_offset_start: usize,
    pub data_sample_offset_end: usize,
    pub data_sample_offset_next: usize,
    pub batch_size: usize,
    pub global_batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub eval_batch_size: usize,
    pub prompt_tokens: Vec<usize>,
    pub response_tokens: Vec<usize>,
    pub sequence_tokens: usize,
    pub response_masked_positions: usize,
    pub padding_tokens: usize,
    pub rank: i64,
    pub alpha: f64,
    pub learning_rate: f64,
    pub final_learning_rate: f64,
    pub steps: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub initial_eval_loss: f64,
    pub eval_history: Vec<QwenLoraSftEvalStep>,
    pub final_eval_loss: f64,
    pub reloaded_eval_loss: f64,
    pub eval_reload_delta: f64,
    pub reloaded_loss: f64,
    pub reload_delta: f64,
    pub full_forward_adapter_delta: f64,
    pub full_forward_reload_delta: f64,
    pub full_forward_merge_delta: f64,
    pub full_forward_unmerge_delta: f64,
    pub full_generate_reload_match: bool,
    pub full_generate_merge_match: bool,
    pub full_generate_new_token_ids: Vec<i64>,
    pub base_requires_grad: bool,
    pub first_step_grad_norm: f64,
    pub final_step_grad_norm: f64,
    pub final_step_clipped_grad_norm: f64,
    pub tokens_per_second: f64,
    pub samples_per_second: f64,
    pub memory_rss_mb: Option<f64>,
    pub gpu_memory_allocated_mb: Option<f64>,
    pub trainable_tensors: Vec<TrainableTensorSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QwenLoraSftEvalStep {
    pub step: usize,
    pub eval_loss: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct QwenLoraConfig {
    pub(crate) target_layers: Vec<usize>,
    pub(crate) target_modules: Vec<QwenLoraTargetModule>,
    pub(crate) rank: i64,
    pub(crate) alpha: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum QwenLoraTargetModule {
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
}

pub(crate) struct QwenLoraRegistry {
    pub(crate) config: QwenLoraConfig,
    pub(crate) adapters: BTreeMap<usize, QwenAttentionLoraAdapter>,
}

#[derive(Clone)]
pub(crate) struct QwenLoraSftTrainPolicy {
    pub(crate) lr_scheduler: LrScheduler,
    pub(crate) max_grad_norm: Option<f64>,
    pub(crate) dataset_order_seed: u64,
    pub(crate) dataset_shuffle: bool,
}

impl QwenLoraSftTrainPolicy {
    pub(crate) fn from_config(config: &Config) -> Self {
        Self {
            lr_scheduler: config.train.lr_scheduler.clone(),
            max_grad_norm: config.train.max_grad_norm.map(f64::from),
            dataset_order_seed: config.run.seed,
            dataset_shuffle: config
                .data
                .as_ref()
                .map(|data| data.shuffle)
                .unwrap_or(true),
        }
    }

    pub(crate) fn constant_without_clip() -> Self {
        Self {
            lr_scheduler: LrScheduler::Constant,
            max_grad_norm: None,
            dataset_order_seed: 0,
            dataset_shuffle: true,
        }
    }
}

pub(crate) fn qwen_validate_lora_resume_config(
    manifest: Option<&QwenLoraSftAdapterManifest>,
    adapter_config: &QwenLoraConfig,
    current_config: &QwenLoraConfig,
    current_compute_kind: &str,
) -> Result<()> {
    if let Some(manifest) = manifest {
        if manifest.compute_kind != current_compute_kind {
            bail!(
                "Qwen LoRA SFT resume manifest compute_kind does not match current train dtype: manifest={}, current={}",
                manifest.compute_kind,
                current_compute_kind
            );
        }
        if manifest.target_layers != current_config.target_layers {
            bail!(
                "Qwen LoRA SFT resume manifest target_layers do not match current [lora] config: manifest={:?}, current={:?}",
                manifest.target_layers,
                current_config.target_layers
            );
        }
        if manifest.target_modules != current_config.target_module_names() {
            bail!(
                "Qwen LoRA SFT resume manifest target_modules do not match current [lora] config: manifest={:?}, current={:?}",
                manifest.target_modules,
                current_config.target_module_names()
            );
        }
    }
    if adapter_config != current_config {
        bail!(
            "Qwen LoRA SFT resume adapter config does not match current [lora] config: resume={:?}, current={:?}",
            adapter_config,
            current_config
        );
    }
    Ok(())
}
pub fn train_qwen_lora_sft_from_config(
    config: &Config,
    run_paths: &RunPaths,
) -> Result<QwenLoraSftTrainSummary> {
    if config.model.architecture != "qwen_lora_sft" {
        bail!(
            "qwen LoRA SFT trainer expects architecture = qwen_lora_sft, got {}",
            config.model.architecture
        );
    }
    if !matches!(config.train.device, RuntimeDevice::Cuda) {
        bail!("qwen LoRA SFT trainer requires device = cuda");
    }
    if config.parallel.data_parallel_size != 1 {
        bail!("qwen LoRA SFT trainer currently expects data_parallel_size = 1");
    }
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("qwen LoRA SFT trainer requires model.model_path")?;
    let model_path = resolve_qwen_model_path(model_path)?;
    let data = config
        .data
        .as_ref()
        .context("qwen LoRA SFT trainer requires [data] instruction_jsonl")?;
    if data.kind != RuntimeDataKind::InstructionJsonl {
        bail!("qwen LoRA SFT trainer requires data.kind = instruction_jsonl");
    }
    if data.paths.is_empty() {
        bail!("qwen LoRA SFT trainer requires at least one data path");
    }
    let lora_config = QwenLoraConfig::from_runtime(
        config
            .lora
            .as_ref()
            .ok_or_else(|| anyhow!("qwen_lora_sft requires [lora] config"))?,
    )?;
    let dtype = match config.train.dtype {
        rustrain_core::runtime::DType::Fp32 => QwenComputeDType::Fp32,
        rustrain_core::runtime::DType::Bf16 => QwenComputeDType::Bf16,
        rustrain_core::runtime::DType::Fp16 => {
            bail!("qwen LoRA SFT trainer does not support fp16 yet; use fp32 or bf16")
        }
    };
    let adapter_output = run_paths
        .checkpoints
        .join("qwen-lora-sft-adapter.safetensors");
    let streaming_index_cache = data
        .index_cache
        .clone()
        .unwrap_or_else(|| qwen_sft_streaming_index_cache_path(&run_paths.cache, "qwen-lora-sft"));
    qwen_lora_sft_train(
        &model_path,
        &adapter_output,
        &run_paths.checkpoints,
        Some(&data.paths),
        &data.eval_paths,
        data.max_samples,
        config.train.resume_from.as_deref(),
        Some(QwenSftFieldMap::from_runtime_data(data)?),
        config.train.micro_batch_size,
        "Reply with rustrain.",
        "rustrain",
        lora_config,
        config.train.learning_rate as f64,
        config.train.max_steps as usize,
        data.train_split,
        config.train.gradient_accumulation_steps,
        config.train.checkpoint_every,
        config.train.eval_every,
        dtype,
        QwenLoraSftTrainPolicy::from_config(config),
        Some(&streaming_index_cache),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_lora_sft_train(
    model_path: &Path,
    adapter_output: &Path,
    checkpoint_dir: &Path,
    sft_paths: Option<&[PathBuf]>,
    eval_paths: &[PathBuf],
    max_samples: Option<usize>,
    resume_from: Option<&Path>,
    field_map: Option<QwenSftFieldMap>,
    sft_batch_size: usize,
    instruction: &str,
    response: &str,
    lora_config: QwenLoraConfig,
    learning_rate: f64,
    steps: usize,
    train_split: f32,
    gradient_accumulation_steps: usize,
    checkpoint_every: u64,
    eval_every: u64,
    dtype: QwenComputeDType,
    policy: QwenLoraSftTrainPolicy,
    streaming_index_cache: Option<&Path>,
) -> Result<QwenLoraSftTrainSummary> {
    if learning_rate <= 0.0 {
        bail!("learning_rate must be positive");
    }
    if sft_batch_size == 0 {
        bail!("sft_batch_size must be positive");
    }
    if steps == 0 {
        bail!("steps must be positive");
    }
    if gradient_accumulation_steps == 0 {
        bail!("gradient_accumulation_steps must be positive");
    }
    if !(0.0..1.0).contains(&train_split) {
        bail!("train_split must be in (0, 1)");
    }

    let model_path = resolve_qwen_model_path(model_path)?;
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let field_map = field_map.unwrap_or_default();
    let dataset = if let Some(sft_paths) = sft_paths {
        qwen_sft_train_eval_datasets_from_paths(
            &tokenizer,
            sft_paths,
            eval_paths,
            max_samples,
            None,
            train_split,
            policy.dataset_shuffle,
            policy.dataset_order_seed,
            &field_map,
        )?
    } else {
        let dataset = QwenSftDataset::from_instruction_pairs(
            &tokenizer,
            &[
                QwenSftExample {
                    system: String::new(),
                    instruction: instruction.to_string(),
                    input: String::new(),
                    response: response.to_string(),
                },
                QwenSftExample {
                    system: String::new(),
                    instruction: "Name the project.".to_string(),
                    input: String::new(),
                    response: "rustrain".to_string(),
                },
            ],
        )?;
        let dataset =
            qwen_apply_sft_shuffle(dataset, policy.dataset_shuffle, policy.dataset_order_seed);
        let combined_summary = dataset.summary();
        let (train_dataset, eval_dataset) = dataset.train_eval_split(train_split)?;
        QwenSftTrainEvalDatasets {
            combined_summary,
            train_dataset,
            eval_dataset,
        }
    };
    let dataset_summary = dataset.combined_summary;
    let train_dataset = dataset.train_dataset;
    let eval_dataset = dataset.eval_dataset;
    let train_batch_size = sft_batch_size.min(train_dataset.len());
    let eval_batch_size = sft_batch_size.min(eval_dataset.len());
    let resume_manifest = resume_from
        .map(read_qwen_lora_sft_resume_manifest)
        .transpose()?
        .flatten();
    if let Some(manifest) = resume_manifest.as_ref() {
        qwen_validate_sft_resume_dataset(
            &manifest.dataset_source_files,
            &manifest.dataset_source_sample_counts,
            &manifest.dataset_fingerprint,
            manifest.dataset_shuffle,
            &dataset_summary,
            "Qwen LoRA SFT adapter resume",
        )?;
    }
    let data_cursor_start = resume_manifest
        .as_ref()
        .map(|manifest| manifest.data_cursor_next)
        .unwrap_or(0);
    let initial_batch = train_dataset.padded_batch(data_cursor_start, train_batch_size)?;
    let eval_batch = eval_dataset.padded_batch(0, eval_batch_size)?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let resume_adapter_path = resume_manifest
        .as_ref()
        .map(|manifest| PathBuf::from(&manifest.adapter_safetensors))
        .or_else(|| resume_from.map(PathBuf::from));
    let registry = if let Some(resume_adapter_path) = resume_adapter_path.as_ref() {
        let registry = QwenLoraRegistry::load(resume_adapter_path)?;
        qwen_validate_lora_resume_config(
            resume_manifest.as_ref(),
            &registry.config,
            &lora_config,
            dtype.label(),
        )?;
        registry
    } else {
        QwenLoraRegistry::deterministic(&weights, &lora_config, true)?
    };
    let mut base_requires_grad = false;
    for layer_index in &lora_config.target_layers {
        let base_layer = QwenLayerWeights::load(&weights, *layer_index)?;
        base_requires_grad = base_requires_grad
            || base_layer.q_proj.requires_grad()
            || base_layer.k_proj.requires_grad()
            || base_layer.v_proj.requires_grad()
            || base_layer.o_proj.requires_grad()
            || base_layer.gate_proj.requires_grad()
            || base_layer.up_proj.requires_grad()
            || base_layer.down_proj.requires_grad();
    }
    let rank = lora_config.rank;
    let alpha = lora_config.alpha_f64();

    let initial_loss = qwen_lora_sft_loss(
        &initial_batch.input_ids,
        &initial_batch.target_mask,
        &weights,
        &lora_config,
        &registry,
        &config,
        dtype.kind(),
    )?
    .double_value(&[]);
    let initial_eval_loss = qwen_lora_sft_loss(
        &eval_batch.input_ids,
        &eval_batch.target_mask,
        &weights,
        &lora_config,
        &registry,
        &config,
        dtype.kind(),
    )?
    .double_value(&[]);

    let base_tensors: BTreeMap<String, Tensor> = registry
        .trainable_tensors()
        .into_iter()
        .map(|(name, tensor)| (name, tensor_snapshot(&tensor)))
        .collect();
    let mut tensor_summaries = Vec::new();
    let mut first_step_grad_norm = 0.0;
    let mut final_step_grad_norm = 0.0;
    let mut final_step_clipped_grad_norm = 0.0;
    let mut final_learning_rate = learning_rate;
    let mut step_adapter_checkpoints = Vec::new();
    let mut eval_history = Vec::new();
    let train_started = Instant::now();
    let data_cursor_end =
        data_cursor_start + steps * gradient_accumulation_steps * train_batch_size;
    let data_cursor_next = data_cursor_end;
    let (data_epoch_start, data_sample_offset_start) =
        qwen_data_epoch_and_offset(data_cursor_start, train_dataset.len())?;
    let (data_epoch_end, data_sample_offset_end) =
        qwen_data_epoch_and_offset(data_cursor_end, train_dataset.len())?;
    let (data_epoch_next, data_sample_offset_next) =
        qwen_data_epoch_and_offset(data_cursor_next, train_dataset.len())?;
    let streaming_window = sft_paths
        .map(|paths| {
            qwen_sft_streaming_token_window_from_jsonl(
                &tokenizer,
                paths,
                eval_paths,
                max_samples,
                train_split,
                policy.dataset_shuffle,
                policy.dataset_order_seed,
                data_cursor_start,
                steps * gradient_accumulation_steps * train_batch_size,
                streaming_index_cache,
                &field_map,
            )
        })
        .transpose()?;

    for step in 0..steps {
        for (_, mut tensor) in registry.trainable_tensors() {
            tensor.zero_grad();
        }
        for accumulation_index in 0..gradient_accumulation_steps {
            let sample_start = data_cursor_start
                + (step * gradient_accumulation_steps + accumulation_index) * train_batch_size;
            let step_batch = if let Some(streaming_window) = streaming_window.as_ref() {
                let relative_start =
                    (step * gradient_accumulation_steps + accumulation_index) * train_batch_size;
                let relative_end = relative_start + train_batch_size;
                let streaming_batch = qwen_sft_padded_batch(
                    &streaming_window.samples[relative_start..relative_end],
                    train_dataset.pad_token_id,
                )?;
                let reference_batch = train_dataset.padded_batch(sample_start, train_batch_size)?;
                let input_delta = tensor_i64_max_abs_diff(
                    &streaming_batch.input_ids,
                    &reference_batch.input_ids,
                )?;
                let mask_delta = tensor_max_abs_diff(
                    &streaming_batch.target_mask,
                    &reference_batch.target_mask,
                )?;
                if input_delta != 0 || mask_delta > 0.0 {
                    bail!(
                        "Qwen LoRA SFT streaming batch mismatch at cursor {sample_start}: input_delta={input_delta}, mask_delta={mask_delta}"
                    );
                }
                streaming_batch
            } else {
                train_dataset.padded_batch(sample_start, train_batch_size)?
            };
            let loss = qwen_lora_sft_loss(
                &step_batch.input_ids,
                &step_batch.target_mask,
                &weights,
                &lora_config,
                &registry,
                &config,
                dtype.kind(),
            )? / gradient_accumulation_steps as f64;
            loss.backward();
        }

        tensor_summaries.clear();
        let trainable_tensors = registry.trainable_tensors();
        let grad_entries = trainable_tensors
            .iter()
            .map(|(name, tensor)| {
                let grad = tensor.grad();
                let grad_defined = grad.defined();
                let grad_norm = if grad_defined {
                    grad.norm().double_value(&[])
                } else {
                    0.0
                };
                if !grad_defined || grad_norm <= 0.0 {
                    bail!("LoRA tensor {name} did not receive a gradient");
                }
                Ok((name.clone(), tensor.shallow_clone(), grad, grad_norm))
            })
            .collect::<Result<Vec<_>>>()?;
        let grad_norm = grad_entries
            .iter()
            .map(|(_, _, _, norm)| norm.powi(2))
            .sum::<f64>()
            .sqrt();
        let (clipped_grad_norm, clip_scale) = qwen_lora_clip_scale(grad_norm, policy.max_grad_norm);
        let step_number = step + 1;
        let step_lr = qwen_lora_sft_learning_rate(
            learning_rate,
            policy.lr_scheduler.clone(),
            step_number,
            steps,
        );
        if step == 0 {
            first_step_grad_norm = grad_norm;
        }
        final_step_grad_norm = grad_norm;
        final_step_clipped_grad_norm = clipped_grad_norm;
        final_learning_rate = step_lr;

        for (name, mut tensor, grad, grad_norm) in grad_entries {
            let clipped_grad = grad * clip_scale;
            let _ = no_grad(|| tensor.f_sub_(&(clipped_grad * step_lr)))?;
            let delta_norm = (&tensor
                - base_tensors
                    .get(&name)
                    .ok_or_else(|| anyhow!("missing base LoRA tensor {name}"))?)
            .norm()
            .double_value(&[]);
            tensor_summaries.push(TrainableTensorSummary {
                name,
                grad_defined: true,
                grad_norm,
                delta_norm,
            });
        }
        if checkpoint_every > 0 && (step_number as u64) % checkpoint_every == 0 {
            let step_adapter_output =
                checkpoint_dir.join(format!("qwen-lora-sft-step-{step_number}.safetensors"));
            registry.save(&step_adapter_output)?;
            step_adapter_checkpoints.push(step_adapter_output.display().to_string());
        }
        if qwen_lora_sft_should_eval_step(step_number, eval_every) {
            let eval_loss = qwen_lora_sft_loss(
                &eval_batch.input_ids,
                &eval_batch.target_mask,
                &weights,
                &lora_config,
                &registry,
                &config,
                dtype.kind(),
            )?
            .double_value(&[]);
            info!(step = step_number, eval_loss, "Qwen LoRA SFT eval step");
            eval_history.push(QwenLoraSftEvalStep {
                step: step_number,
                eval_loss,
            });
        }
    }
    let train_elapsed_secs = train_started.elapsed().as_secs_f64().max(1e-9);
    let trained_samples = train_batch_size * gradient_accumulation_steps * steps;
    let trained_tokens = trained_samples * initial_batch.input_ids.size()[1] as usize;
    let samples_per_second = trained_samples as f64 / train_elapsed_secs;
    let tokens_per_second = trained_tokens as f64 / train_elapsed_secs;

    let final_loss = qwen_lora_sft_loss(
        &initial_batch.input_ids,
        &initial_batch.target_mask,
        &weights,
        &lora_config,
        &registry,
        &config,
        dtype.kind(),
    )?
    .double_value(&[]);
    let final_eval_loss = qwen_lora_sft_loss(
        &eval_batch.input_ids,
        &eval_batch.target_mask,
        &weights,
        &lora_config,
        &registry,
        &config,
        dtype.kind(),
    )?
    .double_value(&[]);
    if final_loss >= initial_loss {
        bail!(
            "Qwen LoRA SFT smoke failed to reduce response-only loss: initial_loss={initial_loss}, final_loss={final_loss}"
        );
    }

    registry.save(adapter_output)?;
    let adapter_manifest_output = qwen_lora_sft_adapter_manifest_path(adapter_output);
    let adapter_manifest = QwenLoraSftAdapterManifest {
        format: "rustrain.qwen_lora_sft_adapter.v1".to_string(),
        base_model_path: model_path.display().to_string(),
        adapter_safetensors: adapter_output.display().to_string(),
        compute_kind: dtype.label().to_string(),
        steps,
        train_step: data_cursor_next as u64,
        data_cursor_start,
        data_cursor_end,
        data_cursor_next,
        data_epoch_start,
        data_epoch_end,
        data_epoch_next,
        data_sample_offset_start,
        data_sample_offset_end,
        data_sample_offset_next,
        dataset_source_files: dataset_summary.source_files.clone(),
        dataset_source_sample_counts: dataset_summary.source_sample_counts.clone(),
        dataset_fingerprint: dataset_summary.fingerprint.clone(),
        dataset_order_seed: policy.dataset_order_seed,
        dataset_shuffle: dataset_summary.shuffle,
        streaming_train_batches: streaming_window.is_some(),
        dataset_total_samples: dataset_summary.samples,
        dataset_train_samples: train_dataset.len(),
        dataset_eval_samples: eval_dataset.len(),
        batch_size: train_batch_size,
        gradient_accumulation_steps,
        target_layers: lora_config.target_layers.clone(),
        target_modules: lora_config.target_module_names(),
    };
    write_qwen_lora_sft_adapter_manifest(&adapter_manifest_output, &adapter_manifest)?;
    let reloaded = QwenLoraRegistry::load(adapter_output)?;
    let reloaded_loss = qwen_lora_sft_loss(
        &initial_batch.input_ids,
        &initial_batch.target_mask,
        &weights,
        &lora_config,
        &reloaded,
        &config,
        dtype.kind(),
    )?
    .double_value(&[]);
    let reload_delta = (final_loss - reloaded_loss).abs();
    if reload_delta > 1e-7 {
        bail!(
            "Qwen LoRA SFT adapter reload loss parity failed: final_loss={final_loss}, reloaded_loss={reloaded_loss}, reload_delta={reload_delta}"
        );
    }
    let reloaded_eval_loss = qwen_lora_sft_loss(
        &eval_batch.input_ids,
        &eval_batch.target_mask,
        &weights,
        &lora_config,
        &reloaded,
        &config,
        dtype.kind(),
    )?
    .double_value(&[]);
    let eval_reload_delta = (final_eval_loss - reloaded_eval_loss).abs();
    if eval_reload_delta > 1e-7 {
        bail!(
            "Qwen LoRA SFT adapter reload eval parity failed: final_eval_loss={final_eval_loss}, reloaded_eval_loss={reloaded_eval_loss}, eval_reload_delta={eval_reload_delta}"
        );
    }
    let base_logits =
        qwen_forward_from_ids_with_kind(&initial_batch.input_ids, &weights, &config, dtype.kind())?;
    let adapted_logits = qwen_forward_from_ids_with_lora(
        &initial_batch.input_ids,
        &weights,
        &config,
        &registry,
        dtype.kind(),
    )?;
    let reloaded_logits = qwen_forward_from_ids_with_lora(
        &initial_batch.input_ids,
        &weights,
        &config,
        &reloaded,
        dtype.kind(),
    )?;
    let full_forward_adapter_delta = diff_stats(&adapted_logits, &base_logits)?.max_abs;
    if full_forward_adapter_delta <= 0.0 {
        bail!("Qwen LoRA SFT adapter did not change full forward logits");
    }
    let full_forward_reload_delta = diff_stats(&reloaded_logits, &adapted_logits)?.max_abs;
    if full_forward_reload_delta > 1e-7 {
        bail!(
            "Qwen LoRA SFT full forward reload parity failed: max_delta={full_forward_reload_delta}"
        );
    }
    let merged_weights = reloaded.merge_into_weights(&weights)?;
    let merged_logits = qwen_forward_from_ids_with_kind(
        &initial_batch.input_ids,
        &merged_weights,
        &config,
        dtype.kind(),
    )?;
    let full_forward_merge_delta = diff_stats(&merged_logits, &adapted_logits)?.max_abs;
    let full_forward_merge_tolerance = match dtype {
        QwenComputeDType::Fp32 => 1e-7,
        QwenComputeDType::Bf16 => 5.0,
    };
    if full_forward_merge_delta > full_forward_merge_tolerance {
        bail!(
            "Qwen LoRA SFT merge parity failed: max_delta={full_forward_merge_delta}, tolerance={full_forward_merge_tolerance}"
        );
    }
    let unmerged_weights = reloaded.unmerge_from_weights(&merged_weights)?;
    let unmerged_logits = qwen_forward_from_ids_with_kind(
        &initial_batch.input_ids,
        &unmerged_weights,
        &config,
        dtype.kind(),
    )?;
    let full_forward_unmerge_delta = diff_stats(&unmerged_logits, &base_logits)?.max_abs;
    let full_forward_unmerge_tolerance = match dtype {
        QwenComputeDType::Fp32 => 5e-4,
        QwenComputeDType::Bf16 => 5.0,
    };
    if full_forward_unmerge_delta > full_forward_unmerge_tolerance {
        bail!(
            "Qwen LoRA SFT unmerge parity failed: max_delta={full_forward_unmerge_delta}, tolerance={full_forward_unmerge_tolerance}"
        );
    }
    let prompt_ids = initial_batch
        .input_ids
        .i(0)
        .reshape([1, initial_batch.input_ids.size()[1]]);
    let generated =
        qwen_greedy_generate_with_lora(&prompt_ids, &weights, &config, &registry, 2, dtype.kind())?;
    let reloaded_generated =
        qwen_greedy_generate_with_lora(&prompt_ids, &weights, &config, &reloaded, 2, dtype.kind())?;
    let merged_generated =
        qwen_greedy_generate_with_kind(&prompt_ids, &merged_weights, &config, 2, dtype.kind())?;
    let generated_ids: Vec<i64> =
        Vec::<i64>::try_from(generated.reshape([-1]).to_device(Device::Cpu))?;
    let reloaded_generated_ids: Vec<i64> =
        Vec::<i64>::try_from(reloaded_generated.reshape([-1]).to_device(Device::Cpu))?;
    let merged_generated_ids: Vec<i64> =
        Vec::<i64>::try_from(merged_generated.reshape([-1]).to_device(Device::Cpu))?;
    let full_generate_reload_match = generated_ids == reloaded_generated_ids;
    if !full_generate_reload_match {
        bail!(
            "Qwen LoRA SFT full generate reload parity failed: generated={generated_ids:?}, reloaded={reloaded_generated_ids:?}"
        );
    }
    let full_generate_merge_match = generated_ids == merged_generated_ids;
    if !full_generate_merge_match && dtype == QwenComputeDType::Fp32 {
        bail!(
            "Qwen LoRA SFT full generate merge parity failed: generated={generated_ids:?}, merged={merged_generated_ids:?}"
        );
    }
    let full_generate_new_token_ids =
        generated_ids[initial_batch.input_ids.size()[1] as usize..].to_vec();

    let summary = QwenLoraSftTrainSummary {
        model_path: model_path.display().to_string(),
        adapter_output: adapter_output.display().to_string(),
        adapter_manifest_output: adapter_manifest_output.display().to_string(),
        compute_kind: dtype.label().to_string(),
        step_adapter_checkpoints,
        resume_from: resume_from.map(|path| path.display().to_string()),
        resumed_adapter: resume_from.is_some(),
        target_layers: lora_config.target_layers.clone(),
        target_modules: lora_config.target_module_names(),
        train_samples: train_dataset.len(),
        eval_samples: eval_dataset.len(),
        dataset_total_samples: dataset_summary.samples,
        dataset_total_tokens: dataset_summary.total_tokens,
        dataset_response_tokens: dataset_summary.response_tokens,
        dataset_masked_positions: dataset_summary.masked_positions,
        dataset_max_sequence_tokens: dataset_summary.max_sequence_tokens,
        dataset_source_files: dataset_summary.source_files,
        dataset_source_sample_counts: dataset_summary.source_sample_counts,
        dataset_fingerprint: dataset_summary.fingerprint,
        dataset_order_seed: policy.dataset_order_seed,
        dataset_shuffle: dataset_summary.shuffle,
        streaming_train_batches: streaming_window.is_some(),
        streaming_index_cache_path: streaming_index_cache.map(|path| path.display().to_string()),
        streaming_index_cache_hit: streaming_window
            .as_ref()
            .is_some_and(|window| window.source_index_cache_hit),
        streaming_index_cache_written: streaming_window
            .as_ref()
            .is_some_and(|window| window.source_index_cache_written),
        data_cursor_start,
        data_cursor_end,
        data_cursor_next,
        data_epoch_start,
        data_epoch_end,
        data_epoch_next,
        data_sample_offset_start,
        data_sample_offset_end,
        data_sample_offset_next,
        batch_size: initial_batch.prompt_tokens.len(),
        global_batch_size: train_batch_size * gradient_accumulation_steps,
        gradient_accumulation_steps,
        eval_batch_size,
        prompt_tokens: initial_batch.prompt_tokens,
        response_tokens: initial_batch.response_tokens,
        sequence_tokens: initial_batch.input_ids.size()[1] as usize,
        response_masked_positions: initial_batch.masked_positions,
        padding_tokens: initial_batch.padding_tokens,
        rank,
        alpha,
        learning_rate,
        final_learning_rate,
        steps,
        initial_loss,
        final_loss,
        initial_eval_loss,
        eval_history,
        final_eval_loss,
        reloaded_eval_loss,
        eval_reload_delta,
        reloaded_loss,
        reload_delta,
        full_forward_adapter_delta,
        full_forward_reload_delta,
        full_forward_merge_delta,
        full_forward_unmerge_delta,
        full_generate_reload_match,
        full_generate_merge_match,
        full_generate_new_token_ids,
        base_requires_grad,
        first_step_grad_norm,
        final_step_grad_norm,
        final_step_clipped_grad_norm,
        tokens_per_second,
        samples_per_second,
        memory_rss_mb: rustrain_train::metrics::memory_rss_mb(),
        gpu_memory_allocated_mb: rustrain_train::metrics::gpu_memory_allocated_mb(),
        trainable_tensors: tensor_summaries,
    };
    Ok(summary)
}

pub(crate) fn qwen_lora_sft_should_eval_step(step_number: usize, eval_every: u64) -> bool {
    eval_every > 0 && (step_number as u64) % eval_every == 0
}

pub(crate) struct QwenAttentionLoraAdapter {
    pub(crate) modules: BTreeMap<QwenLoraTargetModule, QwenLoraModuleAdapter>,
    pub(crate) rank: i64,
    pub(crate) alpha: f64,
}

pub(crate) struct QwenLoraModuleAdapter {
    pub(crate) a: Tensor,
    pub(crate) b: Tensor,
}

impl QwenLoraTargetModule {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::QProj => "q_proj",
            Self::KProj => "k_proj",
            Self::VProj => "v_proj",
            Self::OProj => "o_proj",
            Self::GateProj => "gate_proj",
            Self::UpProj => "up_proj",
            Self::DownProj => "down_proj",
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "q_proj" => Ok(Self::QProj),
            "k_proj" => Ok(Self::KProj),
            "v_proj" => Ok(Self::VProj),
            "o_proj" => Ok(Self::OProj),
            "gate_proj" => Ok(Self::GateProj),
            "up_proj" => Ok(Self::UpProj),
            "down_proj" => Ok(Self::DownProj),
            other => {
                bail!(
                    "unsupported Qwen LoRA target module {other}; supported: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"
                )
            }
        }
    }

    pub(crate) fn id(self) -> i64 {
        match self {
            Self::QProj => 0,
            Self::KProj => 1,
            Self::VProj => 2,
            Self::OProj => 3,
            Self::GateProj => 4,
            Self::UpProj => 5,
            Self::DownProj => 6,
        }
    }

    pub(crate) fn from_id(id: i64) -> Result<Self> {
        match id {
            0 => Ok(Self::QProj),
            1 => Ok(Self::KProj),
            2 => Ok(Self::VProj),
            3 => Ok(Self::OProj),
            4 => Ok(Self::GateProj),
            5 => Ok(Self::UpProj),
            6 => Ok(Self::DownProj),
            other => Err(anyhow!("unknown LoRA target module id {other}")),
        }
    }

    pub(crate) fn parent_path(self) -> &'static str {
        match self {
            Self::QProj | Self::KProj | Self::VProj | Self::OProj => "self_attn",
            Self::GateProj | Self::UpProj | Self::DownProj => "mlp",
        }
    }

    pub(crate) fn weight_name(self, layer_index: usize) -> String {
        format!(
            "model.layers.{layer_index}.{}.{}.weight",
            self.parent_path(),
            self.as_str()
        )
    }

    pub(crate) fn adapter_prefix(self, layer_index: usize) -> String {
        format!(
            "model.layers.{layer_index}.{}.{}",
            self.parent_path(),
            self.as_str()
        )
    }
}

impl QwenLoraConfig {
    pub(crate) fn layer0_qv(rank: i64, alpha: f64) -> Result<Self> {
        Self::new(
            vec![0],
            vec![QwenLoraTargetModule::QProj, QwenLoraTargetModule::VProj],
            rank,
            alpha,
        )
    }

    pub(crate) fn from_runtime(config: &RuntimeLoraConfig) -> Result<Self> {
        let target_modules = config
            .target_modules
            .iter()
            .map(|module| QwenLoraTargetModule::parse(module))
            .collect::<Result<Vec<_>>>()?;
        Self::new(
            config.target_layers.clone(),
            target_modules,
            config.rank,
            config.alpha,
        )
    }

    pub(crate) fn new(
        target_layers: Vec<usize>,
        target_modules: Vec<QwenLoraTargetModule>,
        rank: i64,
        alpha: f64,
    ) -> Result<Self> {
        if rank <= 0 {
            bail!("rank must be positive");
        }
        if alpha <= 0.0 {
            bail!("alpha must be positive");
        }
        if target_layers.is_empty() {
            bail!("target_layers must not be empty");
        }
        if target_modules.is_empty() {
            bail!("target_modules must not be empty");
        }
        let mut seen_modules = BTreeSet::new();
        for module in &target_modules {
            if !seen_modules.insert(*module) {
                bail!("target_modules must not contain duplicates");
            }
        }
        if alpha.fract() != 0.0 || alpha > i64::MAX as f64 {
            bail!("alpha must be representable as an integer for safetensors metadata");
        }
        Ok(Self {
            target_layers,
            target_modules,
            rank,
            alpha: alpha as i64,
        })
    }

    pub(crate) fn alpha_f64(&self) -> f64 {
        self.alpha as f64
    }

    pub(crate) fn target_module_names(&self) -> Vec<String> {
        self.target_modules
            .iter()
            .map(|module| module.as_str().to_string())
            .collect()
    }
}

impl QwenLoraRegistry {
    pub(crate) fn zeros(
        weights: &BTreeMap<String, Tensor>,
        config: &QwenLoraConfig,
    ) -> Result<Self> {
        Self::build(weights, config, false, false)
    }

    pub(crate) fn deterministic(
        weights: &BTreeMap<String, Tensor>,
        config: &QwenLoraConfig,
        trainable: bool,
    ) -> Result<Self> {
        Self::build(weights, config, true, trainable)
    }

    pub(crate) fn build(
        weights: &BTreeMap<String, Tensor>,
        config: &QwenLoraConfig,
        deterministic: bool,
        trainable: bool,
    ) -> Result<Self> {
        let mut adapters = BTreeMap::new();
        for layer_index in &config.target_layers {
            let layer = QwenLayerWeights::load(weights, *layer_index)?;
            let module_specs = config
                .target_modules
                .iter()
                .map(|module| {
                    let weight = layer.lora_target_weight(*module);
                    (*module, weight.size()[1], weight.size()[0])
                })
                .collect::<Vec<_>>();
            let adapter = if deterministic {
                if trainable {
                    QwenAttentionLoraAdapter::deterministic_trainable(
                        &module_specs,
                        config.rank,
                        config.alpha_f64(),
                    )
                } else {
                    QwenAttentionLoraAdapter::deterministic(
                        &module_specs,
                        config.rank,
                        config.alpha_f64(),
                    )
                }
            } else {
                QwenAttentionLoraAdapter::zeros(&module_specs, config.rank, config.alpha_f64())
            };
            adapters.insert(*layer_index, adapter);
        }
        Ok(Self {
            config: config.clone(),
            adapters,
        })
    }

    pub(crate) fn layer_adapter(&self, layer_index: usize) -> Result<&QwenAttentionLoraAdapter> {
        self.adapters
            .get(&layer_index)
            .ok_or_else(|| anyhow!("missing LoRA adapter for layer {layer_index}"))
    }

    pub(crate) fn adapter_for_layer(
        &self,
        layer_index: usize,
    ) -> Option<&QwenAttentionLoraAdapter> {
        self.adapters.get(&layer_index)
    }

    pub(crate) fn merge_into_weights(
        &self,
        weights: &BTreeMap<String, Tensor>,
    ) -> Result<BTreeMap<String, Tensor>> {
        self.apply_to_weights(weights, 1.0)
    }

    pub(crate) fn unmerge_from_weights(
        &self,
        weights: &BTreeMap<String, Tensor>,
    ) -> Result<BTreeMap<String, Tensor>> {
        self.apply_to_weights(weights, -1.0)
    }

    pub(crate) fn apply_to_weights(
        &self,
        weights: &BTreeMap<String, Tensor>,
        scale: f64,
    ) -> Result<BTreeMap<String, Tensor>> {
        let mut merged = weights
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor_snapshot(tensor)))
            .collect::<BTreeMap<_, _>>();
        for (layer_index, adapter) in &self.adapters {
            for module in &self.config.target_modules {
                let name = module.weight_name(*layer_index);
                let weight = tensor(&merged, &name)?.to_kind(Kind::Float);
                let delta = adapter
                    .delta(*module, weight.device())?
                    .to_kind(Kind::Float);
                merged.insert(name, weight.shallow_clone() + delta * scale);
            }
        }
        Ok(merged)
    }

    pub(crate) fn trainable_tensor_names(&self) -> Vec<String> {
        self.trainable_tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    pub(crate) fn trainable_tensors(&self) -> Vec<(String, Tensor)> {
        self.adapters
            .iter()
            .flat_map(|(layer_index, adapter)| adapter.trainable_tensors(*layer_index))
            .collect()
    }

    pub(crate) fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let mut entries: Vec<(String, Tensor)> = Vec::new();
        entries.push((
            "config.rank".to_string(),
            Tensor::from_slice(&[self.config.rank]),
        ));
        entries.push((
            "config.alpha".to_string(),
            Tensor::from_slice(&[self.config.alpha]),
        ));
        let layers: Vec<i64> = self
            .config
            .target_layers
            .iter()
            .map(|layer| *layer as i64)
            .collect();
        entries.push((
            "config.target_layers".to_string(),
            Tensor::from_slice(&layers),
        ));
        let modules: Vec<i64> = self
            .config
            .target_modules
            .iter()
            .map(|module| module.id())
            .collect();
        entries.push((
            "config.target_modules".to_string(),
            Tensor::from_slice(&modules),
        ));
        for (layer_index, adapter) in &self.adapters {
            entries.extend(adapter.safetensor_entries(*layer_index));
        }
        let refs: Vec<(&str, &Tensor)> = entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&refs, path)
            .with_context(|| format!("failed to write {}", path.display()))
    }

    pub(crate) fn load(path: &Path) -> Result<Self> {
        let tensors = read_safetensors_map(path)?;
        let rank = tensor(&tensors, "config.rank")?.int64_value(&[0]);
        let alpha = tensor(&tensors, "config.alpha")?.int64_value(&[0]);
        let target_layers: Vec<usize> =
            Vec::<i64>::try_from(tensor(&tensors, "config.target_layers")?)?
                .into_iter()
                .map(|layer| layer as usize)
                .collect();
        let target_modules: Vec<QwenLoraTargetModule> =
            Vec::<i64>::try_from(tensor(&tensors, "config.target_modules")?)?
                .into_iter()
                .map(QwenLoraTargetModule::from_id)
                .collect::<Result<Vec<_>>>()?;
        let config = QwenLoraConfig {
            target_layers,
            target_modules,
            rank,
            alpha,
        };
        let mut adapters = BTreeMap::new();
        for layer_index in &config.target_layers {
            adapters.insert(
                *layer_index,
                QwenAttentionLoraAdapter::load_from_tensors(&tensors, *layer_index, &config)?,
            );
        }
        Ok(Self { config, adapters })
    }
}

impl QwenAttentionLoraAdapter {
    pub(crate) fn zeros(
        module_specs: &[(QwenLoraTargetModule, i64, i64)],
        rank: i64,
        alpha: f64,
    ) -> Self {
        let modules = module_specs
            .iter()
            .map(|(module, in_features, out_features)| {
                (
                    *module,
                    QwenLoraModuleAdapter {
                        a: Tensor::zeros([rank, *in_features], (Kind::Float, Device::Cpu)),
                        b: Tensor::zeros([*out_features, rank], (Kind::Float, Device::Cpu)),
                    },
                )
            })
            .collect();
        Self {
            modules,
            rank,
            alpha,
        }
    }

    pub(crate) fn deterministic(
        module_specs: &[(QwenLoraTargetModule, i64, i64)],
        rank: i64,
        alpha: f64,
    ) -> Self {
        let modules = module_specs
            .iter()
            .enumerate()
            .map(|(index, (module, in_features, out_features))| {
                let scale = 0.0002 + index as f64 * 0.0001;
                (
                    *module,
                    QwenLoraModuleAdapter {
                        a: deterministic_lora_tensor([rank, *in_features], scale),
                        b: deterministic_lora_tensor([*out_features, rank], -scale * 0.6),
                    },
                )
            })
            .collect();
        Self {
            modules,
            rank,
            alpha,
        }
    }

    pub(crate) fn deterministic_trainable(
        module_specs: &[(QwenLoraTargetModule, i64, i64)],
        rank: i64,
        alpha: f64,
    ) -> Self {
        let adapter = Self::deterministic(module_specs, rank, alpha);
        for module_adapter in adapter.modules.values() {
            let _ = module_adapter.a.set_requires_grad(true);
            let _ = module_adapter.b.set_requires_grad(true);
        }
        adapter
    }

    #[cfg(test)]
    pub(crate) fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let rank = Tensor::from_slice(&[self.rank]);
        let alpha = Tensor::from_slice(&[self.alpha as f32]);
        Tensor::write_safetensors(
            &[
                (
                    &"q_proj.lora_a",
                    &self
                        .modules
                        .get(&QwenLoraTargetModule::QProj)
                        .context("missing q_proj adapter")?
                        .a,
                ),
                (
                    &"q_proj.lora_b",
                    &self
                        .modules
                        .get(&QwenLoraTargetModule::QProj)
                        .context("missing q_proj adapter")?
                        .b,
                ),
                (
                    &"v_proj.lora_a",
                    &self
                        .modules
                        .get(&QwenLoraTargetModule::VProj)
                        .context("missing v_proj adapter")?
                        .a,
                ),
                (
                    &"v_proj.lora_b",
                    &self
                        .modules
                        .get(&QwenLoraTargetModule::VProj)
                        .context("missing v_proj adapter")?
                        .b,
                ),
                (&"rank", &rank),
                (&"alpha", &alpha),
            ],
            path,
        )
        .with_context(|| format!("failed to write {}", path.display()))
    }

    #[cfg(test)]
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let tensors = read_safetensors_map(path)?;
        let q_a = tensor(&tensors, "q_proj.lora_a")?.to_kind(Kind::Float);
        let q_b = tensor(&tensors, "q_proj.lora_b")?.to_kind(Kind::Float);
        let v_a = tensor(&tensors, "v_proj.lora_a")?.to_kind(Kind::Float);
        let v_b = tensor(&tensors, "v_proj.lora_b")?.to_kind(Kind::Float);
        let rank = tensor(&tensors, "rank")?.int64_value(&[0]);
        let alpha = tensor(&tensors, "alpha")?.double_value(&[0]);
        let mut modules = BTreeMap::new();
        modules.insert(
            QwenLoraTargetModule::QProj,
            QwenLoraModuleAdapter { a: q_a, b: q_b },
        );
        modules.insert(
            QwenLoraTargetModule::VProj,
            QwenLoraModuleAdapter { a: v_a, b: v_b },
        );
        Ok(Self {
            modules,
            rank,
            alpha,
        })
    }

    pub(crate) fn load_from_tensors(
        tensors: &BTreeMap<String, Tensor>,
        layer_index: usize,
        config: &QwenLoraConfig,
    ) -> Result<Self> {
        let mut modules = BTreeMap::new();
        for module in &config.target_modules {
            let prefix = module.adapter_prefix(layer_index);
            modules.insert(
                *module,
                QwenLoraModuleAdapter {
                    a: tensor(tensors, &format!("{prefix}.lora_a"))?
                        .to_kind(Kind::Float)
                        .set_requires_grad(true),
                    b: tensor(tensors, &format!("{prefix}.lora_b"))?
                        .to_kind(Kind::Float)
                        .set_requires_grad(true),
                },
            );
        }
        Ok(Self {
            modules,
            rank: config.rank,
            alpha: config.alpha_f64(),
        })
    }

    pub(crate) fn safetensor_entries(&self, layer_index: usize) -> Vec<(String, Tensor)> {
        self.modules
            .iter()
            .flat_map(|(module, adapter)| {
                let prefix = module.adapter_prefix(layer_index);
                [
                    (format!("{prefix}.lora_a"), adapter.a.shallow_clone()),
                    (format!("{prefix}.lora_b"), adapter.b.shallow_clone()),
                ]
            })
            .collect()
    }

    pub(crate) fn trainable_tensors(&self, layer_index: usize) -> Vec<(String, Tensor)> {
        self.modules
            .iter()
            .flat_map(|(module, adapter)| {
                let prefix = module.adapter_prefix(layer_index);
                [
                    (format!("{prefix}.lora_a"), adapter.a.shallow_clone()),
                    (format!("{prefix}.lora_b"), adapter.b.shallow_clone()),
                ]
            })
            .collect()
    }

    pub(crate) fn delta(&self, module: QwenLoraTargetModule, device: Device) -> Result<Tensor> {
        let adapter = self
            .modules
            .get(&module)
            .ok_or_else(|| anyhow!("missing {} LoRA adapter", module.as_str()))?;
        Ok(adapter
            .b
            .to_device(device)
            .matmul(&adapter.a.to_device(device))
            * (self.alpha / self.rank as f64))
    }
}

pub(crate) fn deterministic_lora_tensor<const N: usize>(shape: [i64; N], scale: f64) -> Tensor {
    let len = shape.iter().product::<i64>() as usize;
    let values: Vec<f32> = (0..len)
        .map(|index| ((index % 17) as f64 - 8.0) as f32 * scale as f32)
        .collect();
    Tensor::from_slice(&values).reshape(shape)
}

pub(crate) fn tensor_snapshot(tensor: &Tensor) -> Tensor {
    let mut snapshot = Tensor::zeros_like(tensor);
    snapshot.copy_(tensor);
    snapshot
}

pub(crate) fn qwen_lora_sft_loss(
    input_ids: &Tensor,
    target_mask: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    lora_config: &QwenLoraConfig,
    registry: &QwenLoraRegistry,
    config: &QwenRuntimeConfig,
    compute_kind: Kind,
) -> Result<Tensor> {
    if lora_config.target_layers.is_empty() {
        bail!("LoRA config must include at least one target layer");
    }

    let mut layer_losses = Vec::with_capacity(lora_config.target_layers.len());
    for layer_index in &lora_config.target_layers {
        layer_losses.push(qwen_layer_lora_sft_loss(
            input_ids,
            target_mask,
            weights,
            *layer_index,
            registry.layer_adapter(*layer_index)?,
            config,
            compute_kind,
        )?);
    }
    Ok(Tensor::stack(&layer_losses.iter().collect::<Vec<_>>(), 0).mean(Kind::Float))
}

pub(crate) fn qwen_layer_lora_sft_loss(
    input_ids: &Tensor,
    target_mask: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    layer_index: usize,
    adapter: &QwenAttentionLoraAdapter,
    config: &QwenRuntimeConfig,
    compute_kind: Kind,
) -> Result<Tensor> {
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?.to_kind(compute_kind);
    let layer = QwenLayerWeights::load_with_kind(weights, layer_index, compute_kind)?;
    let hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);
    let base_output = qwen_layer(&hidden, &layer, config);
    let target = lora_train_target(&base_output);
    let adapted = qwen_layer_with_lora(&hidden, &layer, adapter, config);
    let shifted_adapted = adapted.narrow(1, 0, input_ids.size()[1] - 1);
    let shifted_target = target.narrow(1, 0, input_ids.size()[1] - 1);
    let mask = target_mask.to_device(adapted.device());
    let squared = (shifted_adapted - shifted_target).pow_tensor_scalar(2.0) * &mask;
    Ok(squared.sum(Kind::Float) / mask.sum(Kind::Float))
}

pub(crate) fn qwen_lora_sft_learning_rate(
    base_learning_rate: f64,
    scheduler: LrScheduler,
    step: usize,
    max_steps: usize,
) -> f64 {
    match scheduler {
        LrScheduler::Constant => base_learning_rate,
        LrScheduler::LinearDecay => {
            let max_steps = max_steps.max(1) as f64;
            let progress = ((step.saturating_sub(1)) as f64 / max_steps).clamp(0.0, 1.0);
            base_learning_rate * (1.0 - progress)
        }
    }
}

pub(crate) fn qwen_lora_clip_scale(grad_norm: f64, max_grad_norm: Option<f64>) -> (f64, f64) {
    if let Some(max_grad_norm) = max_grad_norm {
        if grad_norm > max_grad_norm {
            let scale = max_grad_norm / (grad_norm + 1e-12);
            return (max_grad_norm, scale);
        }
    }
    (grad_norm, 1.0)
}

pub(crate) fn lora_train_target(base_output: &Tensor) -> Tensor {
    let values = Tensor::arange(
        base_output.numel() as i64,
        (Kind::Float, base_output.device()),
    )
    .reshape(base_output.size())
    .fmod(11.0)
        / 10_000.0;
    base_output + values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_module::test_utils::*;

    #[test]
    fn qwen_attention_lora_adapter_roundtrips_mismatched_q_v_shapes() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let adapter_output = temp.path().join("adapter.safetensors");
        let adapter = QwenAttentionLoraAdapter::deterministic(
            &[
                (QwenLoraTargetModule::QProj, 4, 6),
                (QwenLoraTargetModule::VProj, 4, 2),
            ],
            2,
            8.0,
        );

        assert_eq!(
            adapter
                .delta(QwenLoraTargetModule::QProj, Device::Cpu)
                .expect("q delta")
                .size(),
            vec![6, 4]
        );
        assert_eq!(
            adapter
                .delta(QwenLoraTargetModule::VProj, Device::Cpu)
                .expect("v delta")
                .size(),
            vec![2, 4]
        );

        adapter.save(&adapter_output).expect("adapter should write");
        let reloaded =
            QwenAttentionLoraAdapter::load(&adapter_output).expect("adapter should reload");

        assert_eq!(
            reloaded
                .delta(QwenLoraTargetModule::QProj, Device::Cpu)
                .expect("q delta")
                .size(),
            vec![6, 4]
        );
        assert_eq!(
            reloaded
                .delta(QwenLoraTargetModule::VProj, Device::Cpu)
                .expect("v delta")
                .size(),
            vec![2, 4]
        );
        assert!(
            diff_stats(
                &reloaded
                    .delta(QwenLoraTargetModule::QProj, Device::Cpu)
                    .expect("reloaded q delta"),
                &adapter
                    .delta(QwenLoraTargetModule::QProj, Device::Cpu)
                    .expect("q delta")
            )
            .expect("q delta diff should compute")
            .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(
                &reloaded
                    .delta(QwenLoraTargetModule::VProj, Device::Cpu)
                    .expect("reloaded v delta"),
                &adapter
                    .delta(QwenLoraTargetModule::VProj, Device::Cpu)
                    .expect("v delta")
            )
            .expect("v delta diff should compute")
            .max_abs
                < 1e-8
        );
    }

    #[test]
    fn qwen_attention_lora_train_step_reduces_tiny_mse_and_reloads() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let adapter_output = temp.path().join("adapter.safetensors");
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let weights = tiny_qwen_weights();
        let layer = QwenLayerWeights::load(&weights, 0).expect("layer should load");
        let input = Tensor::arange(12, (Kind::Float, Device::Cpu)).reshape([1, 3, 4]) / 12.0;
        let target = qwen_attention(
            &input,
            &layer.q_proj,
            &layer.q_bias,
            &layer.k_proj,
            &layer.k_bias,
            &layer.v_proj,
            &layer.v_bias,
            &layer.o_proj,
            &config,
        ) + Tensor::ones([1, 3, 4], (Kind::Float, Device::Cpu)) * 0.01;
        let adapter = QwenAttentionLoraAdapter::deterministic_trainable(
            &[
                (QwenLoraTargetModule::QProj, 4, 4),
                (QwenLoraTargetModule::VProj, 4, 2),
            ],
            2,
            8.0,
        );

        let initial_loss = qwen_attention_lora_mse_loss(&input, &target, &layer, &adapter, &config)
            .double_value(&[]);
        let loss = qwen_attention_lora_mse_loss(&input, &target, &layer, &adapter, &config);
        loss.backward();
        for (_, mut tensor) in adapter.trainable_tensors(0) {
            let grad = tensor.grad();
            assert!(grad.defined());
            let _ = no_grad(|| tensor.f_sub_(&(&grad * 1.0))).expect("update should apply");
        }
        let final_loss = qwen_attention_lora_mse_loss(&input, &target, &layer, &adapter, &config)
            .double_value(&[]);
        assert!(final_loss < initial_loss);

        adapter.save(&adapter_output).expect("adapter should save");
        let reloaded =
            QwenAttentionLoraAdapter::load(&adapter_output).expect("adapter should reload");
        let reloaded_loss =
            qwen_attention_lora_mse_loss(&input, &target, &layer, &reloaded, &config)
                .double_value(&[]);
        assert!((final_loss - reloaded_loss).abs() < 1e-8);
    }

    #[test]
    fn qwen_lora_registry_roundtrips_configured_layer_targets() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let adapter_output = temp.path().join("adapter.safetensors");
        let weights = tiny_qwen_weights();
        let runtime_config = RuntimeLoraConfig {
            rank: 2,
            alpha: 8.0,
            target_layers: vec![0],
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
        };
        let config = QwenLoraConfig::from_runtime(&runtime_config).expect("config should build");
        let registry = QwenLoraRegistry::deterministic(&weights, &config, true)
            .expect("registry should build");

        assert_eq!(registry.config.target_layers, vec![0]);
        assert_eq!(
            registry.config.target_modules,
            vec![
                QwenLoraTargetModule::QProj,
                QwenLoraTargetModule::KProj,
                QwenLoraTargetModule::VProj,
                QwenLoraTargetModule::OProj,
                QwenLoraTargetModule::GateProj,
                QwenLoraTargetModule::UpProj,
                QwenLoraTargetModule::DownProj,
            ]
        );
        assert_eq!(
            registry.trainable_tensor_names(),
            vec![
                "model.layers.0.self_attn.q_proj.lora_a".to_string(),
                "model.layers.0.self_attn.q_proj.lora_b".to_string(),
                "model.layers.0.self_attn.k_proj.lora_a".to_string(),
                "model.layers.0.self_attn.k_proj.lora_b".to_string(),
                "model.layers.0.self_attn.v_proj.lora_a".to_string(),
                "model.layers.0.self_attn.v_proj.lora_b".to_string(),
                "model.layers.0.self_attn.o_proj.lora_a".to_string(),
                "model.layers.0.self_attn.o_proj.lora_b".to_string(),
                "model.layers.0.mlp.gate_proj.lora_a".to_string(),
                "model.layers.0.mlp.gate_proj.lora_b".to_string(),
                "model.layers.0.mlp.up_proj.lora_a".to_string(),
                "model.layers.0.mlp.up_proj.lora_b".to_string(),
                "model.layers.0.mlp.down_proj.lora_a".to_string(),
                "model.layers.0.mlp.down_proj.lora_b".to_string(),
            ]
        );

        registry
            .save(&adapter_output)
            .expect("registry should save");
        let reloaded = QwenLoraRegistry::load(&adapter_output).expect("registry should reload");

        assert_eq!(reloaded.config, config);
        for (name, tensor) in reloaded.trainable_tensors() {
            assert!(
                tensor.requires_grad(),
                "{name} should remain trainable after reload"
            );
        }
        assert!(
            diff_stats(
                &reloaded
                    .layer_adapter(0)
                    .expect("layer adapter")
                    .delta(QwenLoraTargetModule::QProj, Device::Cpu)
                    .expect("reloaded q delta"),
                &registry
                    .layer_adapter(0)
                    .expect("layer adapter")
                    .delta(QwenLoraTargetModule::QProj, Device::Cpu)
                    .expect("q delta"),
            )
            .expect("q delta diff should compute")
            .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(
                &reloaded
                    .layer_adapter(0)
                    .expect("layer adapter")
                    .delta(QwenLoraTargetModule::VProj, Device::Cpu)
                    .expect("reloaded v delta"),
                &registry
                    .layer_adapter(0)
                    .expect("layer adapter")
                    .delta(QwenLoraTargetModule::VProj, Device::Cpu)
                    .expect("v delta"),
            )
            .expect("v delta diff should compute")
            .max_abs
                < 1e-8
        );
        assert_eq!(
            reloaded
                .layer_adapter(0)
                .expect("layer adapter")
                .delta(QwenLoraTargetModule::KProj, Device::Cpu)
                .expect("k delta")
                .size(),
            vec![2, 4]
        );
        assert_eq!(
            reloaded
                .layer_adapter(0)
                .expect("layer adapter")
                .delta(QwenLoraTargetModule::OProj, Device::Cpu)
                .expect("o delta")
                .size(),
            vec![4, 4]
        );
        assert_eq!(
            reloaded
                .layer_adapter(0)
                .expect("layer adapter")
                .delta(QwenLoraTargetModule::GateProj, Device::Cpu)
                .expect("gate delta")
                .size(),
            vec![8, 4]
        );
        assert_eq!(
            reloaded
                .layer_adapter(0)
                .expect("layer adapter")
                .delta(QwenLoraTargetModule::DownProj, Device::Cpu)
                .expect("down delta")
                .size(),
            vec![4, 8]
        );
    }

    #[test]
    fn qwen_lora_registry_applies_all_projection_targets_to_layer() {
        let weights = tiny_qwen_weights();
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let lora_config = QwenLoraConfig::new(
            vec![0],
            vec![
                QwenLoraTargetModule::QProj,
                QwenLoraTargetModule::KProj,
                QwenLoraTargetModule::VProj,
                QwenLoraTargetModule::OProj,
                QwenLoraTargetModule::GateProj,
                QwenLoraTargetModule::UpProj,
                QwenLoraTargetModule::DownProj,
            ],
            2,
            8.0,
        )
        .expect("LoRA config should build");
        let zero_registry =
            QwenLoraRegistry::zeros(&weights, &lora_config).expect("zero registry should build");
        let registry = QwenLoraRegistry::deterministic(&weights, &lora_config, false)
            .expect("registry should build");
        let input = Tensor::arange(12, (Kind::Float, Device::Cpu)).reshape([1, 3, 4]) / 12.0;
        let base_layer = QwenLayerWeights::load(&weights, 0).expect("base layer should load");
        let base_output = qwen_layer(&input, &base_layer, &config);
        let zero_output = qwen_layer_with_lora(
            &input,
            &base_layer,
            zero_registry.layer_adapter(0).expect("zero adapter"),
            &config,
        );
        let adapted_output = qwen_layer_with_lora(
            &input,
            &base_layer,
            registry.layer_adapter(0).expect("adapter"),
            &config,
        );
        assert!(
            diff_stats(&base_output, &zero_output)
                .expect("zero diff should compute")
                .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(&base_output, &adapted_output)
                .expect("adapted diff should compute")
                .max_abs
                > 0.0
        );

        let merged_weights = registry
            .merge_into_weights(&weights)
            .expect("registry should merge all targets");
        let merged_layer =
            QwenLayerWeights::load(&merged_weights, 0).expect("merged layer should load");
        let merged_output = qwen_layer(&input, &merged_layer, &config);
        assert!(
            diff_stats(&adapted_output, &merged_output)
                .expect("merged diff should compute")
                .max_abs
                < 1e-6
        );

        let unmerged_weights = registry
            .unmerge_from_weights(&merged_weights)
            .expect("registry should unmerge all targets");
        let unmerged_layer =
            QwenLayerWeights::load(&unmerged_weights, 0).expect("unmerged layer should load");
        let unmerged_output = qwen_layer(&input, &unmerged_layer, &config);
        assert!(
            diff_stats(&base_output, &unmerged_output)
                .expect("unmerged diff should compute")
                .max_abs
                < 1e-6
        );
    }

    #[test]
    fn qwen_lora_sft_resume_config_validation_checks_manifest_and_adapter() {
        let current = QwenLoraConfig::new(
            vec![0, 1],
            vec![
                QwenLoraTargetModule::QProj,
                QwenLoraTargetModule::VProj,
                QwenLoraTargetModule::DownProj,
            ],
            4,
            8.0,
        )
        .expect("current config should build");
        let mut manifest = QwenLoraSftAdapterManifest {
            format: "rustrain.qwen_lora_sft_adapter.v1".to_string(),
            base_model_path: "/models/qwen".to_string(),
            adapter_safetensors: "/tmp/adapter.safetensors".to_string(),
            compute_kind: "fp32".to_string(),
            steps: 2,
            train_step: 4,
            data_cursor_start: 0,
            data_cursor_end: 4,
            data_cursor_next: 4,
            data_epoch_start: 0,
            data_epoch_end: 0,
            data_epoch_next: 0,
            data_sample_offset_start: 0,
            data_sample_offset_end: 4,
            data_sample_offset_next: 4,
            dataset_source_files: vec!["data/train.jsonl".to_string()],
            dataset_source_sample_counts: vec![QwenSftSourceSampleCount {
                path: "data/train.jsonl".to_string(),
                samples: 4,
            }],
            dataset_fingerprint: "abc123".to_string(),
            dataset_order_seed: 777,
            dataset_shuffle: true,
            streaming_train_batches: true,
            dataset_total_samples: 4,
            dataset_train_samples: 3,
            dataset_eval_samples: 1,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            target_layers: current.target_layers.clone(),
            target_modules: current.target_module_names(),
        };

        qwen_validate_lora_resume_config(Some(&manifest), &current, &current, "fp32")
            .expect("matching manifest and adapter config should pass");
        qwen_validate_lora_resume_config(None, &current, &current, "bf16")
            .expect("direct adapter resume should pass without manifest metadata");

        let compute_kind_error =
            qwen_validate_lora_resume_config(Some(&manifest), &current, &current, "bf16")
                .expect_err("manifest compute kind mismatch should fail")
                .to_string();
        assert!(compute_kind_error.contains("resume manifest compute_kind"));

        let adapter_mismatch = QwenLoraConfig::new(
            vec![0, 1],
            vec![QwenLoraTargetModule::QProj, QwenLoraTargetModule::VProj],
            4,
            8.0,
        )
        .expect("adapter mismatch config should build");
        let adapter_error =
            qwen_validate_lora_resume_config(Some(&manifest), &adapter_mismatch, &current, "fp32")
                .expect_err("adapter config mismatch should fail")
                .to_string();
        assert!(adapter_error.contains("resume adapter config does not match"));

        manifest.target_modules = vec!["q_proj".to_string(), "v_proj".to_string()];
        let manifest_module_error =
            qwen_validate_lora_resume_config(Some(&manifest), &current, &current, "fp32")
                .expect_err("manifest module mismatch should fail")
                .to_string();
        assert!(manifest_module_error.contains("resume manifest target_modules"));

        manifest.target_modules = current.target_module_names();
        manifest.target_layers = vec![0];
        let manifest_layer_error =
            qwen_validate_lora_resume_config(Some(&manifest), &current, &current, "fp32")
                .expect_err("manifest layer mismatch should fail")
                .to_string();
        assert!(manifest_layer_error.contains("resume manifest target_layers"));
    }

    #[test]
    fn qwen_lora_config_rejects_unsupported_target_module() {
        let runtime_config = RuntimeLoraConfig {
            rank: 2,
            alpha: 8.0,
            target_layers: vec![0],
            target_modules: vec!["score_proj".to_string()],
        };
        let error = QwenLoraConfig::from_runtime(&runtime_config)
            .expect_err("unsupported target should fail");

        assert!(
            error
                .to_string()
                .contains("unsupported Qwen LoRA target module score_proj")
        );
    }

    #[test]
    fn qwen_lora_full_forward_and_generate_reload_parity() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let adapter_output = temp.path().join("adapter.safetensors");
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let weights = tiny_qwen_weights();
        let lora_config = QwenLoraConfig::layer0_qv(2, 8.0).expect("config should build");
        let registry = QwenLoraRegistry::deterministic(&weights, &lora_config, false)
            .expect("registry should build");
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2]).reshape([1, 3]);

        let base_logits =
            qwen_forward_from_ids(&input_ids, &weights, &config).expect("base forward should run");
        let adapted_logits =
            qwen_forward_from_ids_with_lora(&input_ids, &weights, &config, &registry, Kind::Float)
                .expect("LoRA forward should run");
        assert!(
            diff_stats(&adapted_logits, &base_logits)
                .expect("adapter diff should compute")
                .max_abs
                > 0.0
        );

        registry
            .save(&adapter_output)
            .expect("registry should save");
        let reloaded = QwenLoraRegistry::load(&adapter_output).expect("registry should reload");
        let reloaded_logits =
            qwen_forward_from_ids_with_lora(&input_ids, &weights, &config, &reloaded, Kind::Float)
                .expect("reloaded LoRA forward should run");
        assert!(
            diff_stats(&reloaded_logits, &adapted_logits)
                .expect("reload diff should compute")
                .max_abs
                < 1e-8
        );
        let merged_weights = reloaded
            .merge_into_weights(&weights)
            .expect("LoRA weights should merge");
        let merged_logits = qwen_forward_from_ids(&input_ids, &merged_weights, &config)
            .expect("merged forward should run");
        assert!(
            diff_stats(&merged_logits, &adapted_logits)
                .expect("merge diff should compute")
                .max_abs
                < 1e-8
        );
        let unmerged_weights = reloaded
            .unmerge_from_weights(&merged_weights)
            .expect("LoRA weights should unmerge");
        let unmerged_logits = qwen_forward_from_ids(&input_ids, &unmerged_weights, &config)
            .expect("unmerged forward should run");
        assert!(
            diff_stats(&unmerged_logits, &base_logits)
                .expect("unmerge diff should compute")
                .max_abs
                < 1e-8
        );

        let generated = qwen_greedy_generate_with_lora(
            &input_ids,
            &weights,
            &config,
            &registry,
            2,
            Kind::Float,
        )
        .expect("LoRA generate should run");
        let reloaded_generated = qwen_greedy_generate_with_lora(
            &input_ids,
            &weights,
            &config,
            &reloaded,
            2,
            Kind::Float,
        )
        .expect("reloaded LoRA generate should run");
        let merged_generated = qwen_greedy_generate(&input_ids, &merged_weights, &config, 2)
            .expect("merged LoRA generate should run");
        let generated_ids: Vec<i64> = Vec::<i64>::try_from(generated.reshape([-1])).unwrap();
        let reloaded_generated_ids: Vec<i64> =
            Vec::<i64>::try_from(reloaded_generated.reshape([-1])).unwrap();
        let merged_generated_ids: Vec<i64> =
            Vec::<i64>::try_from(merged_generated.reshape([-1])).unwrap();
        assert_eq!(reloaded_generated_ids, generated_ids);
        assert_eq!(merged_generated_ids, generated_ids);
    }

    #[test]
    fn qwen_lora_full_layer_targets_affect_forward_and_merge() {
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let weights = tiny_qwen_weights();
        let lora_config = QwenLoraConfig::new(
            vec![0],
            vec![
                QwenLoraTargetModule::QProj,
                QwenLoraTargetModule::KProj,
                QwenLoraTargetModule::VProj,
                QwenLoraTargetModule::OProj,
                QwenLoraTargetModule::GateProj,
                QwenLoraTargetModule::UpProj,
                QwenLoraTargetModule::DownProj,
            ],
            2,
            8.0,
        )
        .expect("config should build");
        let registry = QwenLoraRegistry::deterministic(&weights, &lora_config, false)
            .expect("registry should build");
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2]).reshape([1, 3]);

        for module in &lora_config.target_modules {
            let weight =
                tensor(&weights, &module.weight_name(0)).expect("base weight should exist");
            assert_eq!(
                registry
                    .layer_adapter(0)
                    .expect("layer adapter should exist")
                    .delta(*module, Device::Cpu)
                    .expect("delta should build")
                    .size(),
                weight.size()
            );
        }

        let base_logits =
            qwen_forward_from_ids(&input_ids, &weights, &config).expect("base forward should run");
        let adapted_logits =
            qwen_forward_from_ids_with_lora(&input_ids, &weights, &config, &registry, Kind::Float)
                .expect("LoRA forward should run");
        assert!(
            diff_stats(&adapted_logits, &base_logits)
                .expect("adapter diff should compute")
                .max_abs
                > 0.0
        );

        let merged_weights = registry
            .merge_into_weights(&weights)
            .expect("LoRA weights should merge");
        let merged_logits = qwen_forward_from_ids(&input_ids, &merged_weights, &config)
            .expect("merged forward should run");
        assert!(
            diff_stats(&merged_logits, &adapted_logits)
                .expect("merge diff should compute")
                .max_abs
                < 1e-8
        );
    }

    #[test]
    fn qwen_lora_sft_eval_every_selects_periodic_steps() {
        assert!(!qwen_lora_sft_should_eval_step(1, 0));
        assert!(qwen_lora_sft_should_eval_step(1, 1));
        assert!(!qwen_lora_sft_should_eval_step(1, 2));
        assert!(qwen_lora_sft_should_eval_step(2, 2));
        assert!(qwen_lora_sft_should_eval_step(4, 2));
    }
}
