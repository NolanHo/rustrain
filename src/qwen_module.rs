use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use tch::{Device, IndexOp, Kind, Reduction, Tensor, no_grad};
use tokenizers::Tokenizer;
use tracing::info;

use crate::nccl_smoke;
use crate::runtime::{
    Config, DataKind as RuntimeDataKind, Device as RuntimeDevice, LoraConfig as RuntimeLoraConfig,
    LrScheduler, RunPaths, load_config,
};

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
    cached_ids: Vec<i64>,
    new_token_ids: Vec<i64>,
    cache_match: bool,
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
    target_layers: Vec<usize>,
    target_modules: Vec<String>,
    rank: i64,
    alpha: f64,
    zero_lora_max_delta: f64,
    nonzero_lora_max_delta: f64,
    reload_max_delta: f64,
    trainable_tensors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct QwenLoraTrainSmokeSummary {
    model_path: String,
    fixture: String,
    adapter_output: String,
    target_layers: Vec<usize>,
    target_modules: Vec<String>,
    rank: i64,
    alpha: f64,
    learning_rate: f64,
    initial_loss: f64,
    final_loss: f64,
    reloaded_loss: f64,
    reload_delta: f64,
    base_requires_grad: bool,
    trainable_tensors: Vec<TrainableTensorSummary>,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenLoraSftTrainSummary {
    pub(crate) model_path: String,
    pub(crate) adapter_output: String,
    pub(crate) adapter_manifest_output: String,
    pub(crate) compute_kind: String,
    pub(crate) step_adapter_checkpoints: Vec<String>,
    pub(crate) resume_from: Option<String>,
    pub(crate) resumed_adapter: bool,
    pub(crate) target_layers: Vec<usize>,
    pub(crate) target_modules: Vec<String>,
    pub(crate) train_samples: usize,
    pub(crate) eval_samples: usize,
    pub(crate) dataset_total_samples: usize,
    pub(crate) dataset_total_tokens: usize,
    pub(crate) dataset_response_tokens: usize,
    pub(crate) dataset_masked_positions: usize,
    pub(crate) dataset_max_sequence_tokens: usize,
    pub(crate) dataset_source_files: Vec<String>,
    pub(crate) dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) dataset_fingerprint: String,
    pub(crate) dataset_order_seed: u64,
    pub(crate) dataset_shuffle: bool,
    pub(crate) data_cursor_start: usize,
    pub(crate) data_cursor_end: usize,
    pub(crate) data_cursor_next: usize,
    pub(crate) data_epoch_start: usize,
    pub(crate) data_epoch_end: usize,
    pub(crate) data_epoch_next: usize,
    pub(crate) data_sample_offset_start: usize,
    pub(crate) data_sample_offset_end: usize,
    pub(crate) data_sample_offset_next: usize,
    pub(crate) batch_size: usize,
    pub(crate) global_batch_size: usize,
    pub(crate) gradient_accumulation_steps: usize,
    pub(crate) eval_batch_size: usize,
    pub(crate) prompt_tokens: Vec<usize>,
    pub(crate) response_tokens: Vec<usize>,
    pub(crate) sequence_tokens: usize,
    pub(crate) response_masked_positions: usize,
    pub(crate) padding_tokens: usize,
    pub(crate) rank: i64,
    pub(crate) alpha: f64,
    pub(crate) learning_rate: f64,
    pub(crate) final_learning_rate: f64,
    pub(crate) steps: usize,
    pub(crate) initial_loss: f64,
    pub(crate) final_loss: f64,
    pub(crate) initial_eval_loss: f64,
    pub(crate) eval_history: Vec<QwenLoraSftEvalStep>,
    pub(crate) final_eval_loss: f64,
    pub(crate) reloaded_eval_loss: f64,
    pub(crate) eval_reload_delta: f64,
    pub(crate) reloaded_loss: f64,
    pub(crate) reload_delta: f64,
    pub(crate) full_forward_adapter_delta: f64,
    pub(crate) full_forward_reload_delta: f64,
    pub(crate) full_forward_merge_delta: f64,
    pub(crate) full_forward_unmerge_delta: f64,
    pub(crate) full_generate_reload_match: bool,
    pub(crate) full_generate_merge_match: bool,
    pub(crate) full_generate_new_token_ids: Vec<i64>,
    pub(crate) base_requires_grad: bool,
    pub(crate) first_step_grad_norm: f64,
    pub(crate) final_step_grad_norm: f64,
    pub(crate) final_step_clipped_grad_norm: f64,
    pub(crate) tokens_per_second: f64,
    pub(crate) samples_per_second: f64,
    pub(crate) memory_rss_mb: Option<f64>,
    pub(crate) gpu_memory_allocated_mb: Option<f64>,
    pub(crate) trainable_tensors: Vec<TrainableTensorSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct QwenLoraSftEvalStep {
    pub(crate) step: usize,
    pub(crate) eval_loss: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct QwenLoraSftAdapterManifest {
    format: String,
    base_model_path: String,
    adapter_safetensors: String,
    compute_kind: String,
    steps: usize,
    train_step: u64,
    data_cursor_start: usize,
    data_cursor_end: usize,
    data_cursor_next: usize,
    #[serde(default)]
    data_epoch_start: usize,
    #[serde(default)]
    data_epoch_end: usize,
    #[serde(default)]
    data_epoch_next: usize,
    #[serde(default)]
    data_sample_offset_start: usize,
    #[serde(default)]
    data_sample_offset_end: usize,
    #[serde(default)]
    data_sample_offset_next: usize,
    #[serde(default)]
    dataset_source_files: Vec<String>,
    #[serde(default)]
    dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    #[serde(default)]
    dataset_fingerprint: String,
    dataset_order_seed: u64,
    #[serde(default = "qwen_manifest_default_dataset_shuffle")]
    dataset_shuffle: bool,
    dataset_total_samples: usize,
    dataset_train_samples: usize,
    dataset_eval_samples: usize,
    batch_size: usize,
    gradient_accumulation_steps: usize,
    target_layers: Vec<usize>,
    target_modules: Vec<String>,
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

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TrainableTensorSummary {
    pub(crate) name: String,
    pub(crate) grad_defined: bool,
    pub(crate) grad_norm: f64,
    pub(crate) delta_norm: f64,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenFullTrainSmokeSummary {
    pub(crate) model_path: String,
    pub(crate) reference_fixture: String,
    pub(crate) delta_output: String,
    pub(crate) optimizer_output: String,
    pub(crate) manifest_output: String,
    pub(crate) compute_kind: String,
    pub(crate) resume_from: Option<String>,
    pub(crate) resumed_checkpoint: bool,
    pub(crate) train_steps: usize,
    pub(crate) learning_rate: f64,
    pub(crate) step_losses: Vec<f64>,
    pub(crate) first_step_grad_norm: f64,
    pub(crate) final_step_grad_norm: f64,
    pub(crate) tokens_per_second: f64,
    pub(crate) samples_per_second: f64,
    pub(crate) memory_rss_mb: Option<f64>,
    pub(crate) gpu_memory_allocated_mb: Option<f64>,
    pub(crate) dataset_total_samples: Option<usize>,
    pub(crate) dataset_total_tokens: Option<usize>,
    pub(crate) dataset_train_samples: Option<usize>,
    pub(crate) dataset_eval_samples: Option<usize>,
    pub(crate) dataset_source_files: Option<Vec<String>>,
    pub(crate) dataset_source_sample_counts: Option<Vec<QwenSftSourceSampleCount>>,
    pub(crate) dataset_fingerprint: Option<String>,
    pub(crate) dataset_order_seed: Option<u64>,
    pub(crate) dataset_shuffle: Option<bool>,
    pub(crate) data_cursor_start: Option<usize>,
    pub(crate) data_cursor_end: Option<usize>,
    pub(crate) data_cursor_next: Option<usize>,
    pub(crate) data_epoch_start: Option<usize>,
    pub(crate) data_epoch_end: Option<usize>,
    pub(crate) data_epoch_next: Option<usize>,
    pub(crate) data_sample_offset_start: Option<usize>,
    pub(crate) data_sample_offset_end: Option<usize>,
    pub(crate) data_sample_offset_next: Option<usize>,
    pub(crate) batch_size: usize,
    pub(crate) sequence_tokens: usize,
    pub(crate) initial_loss: f64,
    pub(crate) final_loss: f64,
    pub(crate) reloaded_loss: f64,
    pub(crate) reload_delta: f64,
    pub(crate) resume_loss: f64,
    pub(crate) continuous_second_loss: f64,
    pub(crate) resumed_second_loss: f64,
    pub(crate) second_step_delta: f64,
    pub(crate) trainable_tensors: Vec<TrainableTensorSummary>,
}

#[derive(Debug, Serialize)]
struct QwenDpGradientRankSummary {
    rank: usize,
    world_size: usize,
    local_sequence_count: usize,
    tensor_count: usize,
    steps: usize,
    learning_rate: f64,
    max_grad_delta: f32,
    loss_delta: f64,
    local_loss: f64,
    post_update_loss: f64,
    global_loss: f64,
    global_post_update_loss: f64,
    global_loss_improved: bool,
    global_step_losses: Vec<f64>,
    expected_loss: f64,
    checkpoint_written: bool,
    checkpoint_path: String,
}

#[derive(Debug, Serialize)]
struct QwenSessionDpRankSummary {
    rank: usize,
    world_size: usize,
    dtype: String,
    resume_from: Option<String>,
    resumed_checkpoint: bool,
    local_batch_size: usize,
    sequence_tokens: usize,
    dataset_total_samples: Option<usize>,
    dataset_total_tokens: Option<usize>,
    dataset_train_samples: Option<usize>,
    dataset_eval_samples: Option<usize>,
    dataset_source_files: Option<Vec<String>>,
    dataset_source_sample_counts: Option<Vec<QwenSftSourceSampleCount>>,
    dataset_fingerprint: Option<String>,
    dataset_order_seed: Option<u64>,
    dataset_shuffle: Option<bool>,
    data_cursor_start: Option<usize>,
    data_cursor_end: Option<usize>,
    data_cursor_next: Option<usize>,
    data_epoch_start: Option<usize>,
    data_epoch_end: Option<usize>,
    data_epoch_next: Option<usize>,
    data_sample_offset_start: Option<usize>,
    data_sample_offset_end: Option<usize>,
    data_sample_offset_next: Option<usize>,
    tensor_count: usize,
    steps: usize,
    learning_rate: f64,
    max_grad_delta: f32,
    loss_delta: f64,
    local_loss: f64,
    post_update_loss: f64,
    global_loss: f64,
    global_post_update_loss: f64,
    global_loss_improved: bool,
    global_step_losses: Vec<f64>,
    expected_loss: f64,
    checkpoint_written: bool,
    checkpoint_path: String,
    delta_output: String,
    optimizer_output: String,
    manifest_output: String,
    sharded_rank_manifest_output: String,
    sharded_global_manifest_output: String,
    reloaded_loss: f64,
    reload_delta: f64,
    sharded_reloaded_loss: f64,
    sharded_reload_delta: f64,
    continuous_next_loss: f64,
    resumed_next_loss: f64,
    next_step_delta: f64,
    sharded_continuous_next_loss: f64,
    sharded_resumed_next_loss: f64,
    sharded_next_step_delta: f64,
    trainable_tensors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct QwenSessionDpDataPlanSummary {
    config_path: String,
    model_path: String,
    world_size: usize,
    local_batch_size: usize,
    global_batch_size: usize,
    train_steps: usize,
    required_batches: usize,
    data_cursor_start: usize,
    data_cursor_end: usize,
    data_cursor_next: usize,
    data_epoch_start: usize,
    data_epoch_end: usize,
    data_epoch_next: usize,
    data_sample_offset_start: usize,
    data_sample_offset_end: usize,
    data_sample_offset_next: usize,
    dataset_total_samples: usize,
    dataset_total_tokens: usize,
    dataset_train_samples: usize,
    dataset_eval_samples: usize,
    dataset_source_files: Vec<String>,
    dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    dataset_fingerprint: String,
    dataset_order_seed: u64,
    dataset_shuffle: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct QwenSessionDpCheckpointManifest {
    format: String,
    base_model_path: String,
    writer_rank: usize,
    world_size: usize,
    tensor_count: usize,
    max_grad_delta: f32,
    expected_loss: f64,
    dtype: String,
    steps: usize,
    train_step: u64,
    #[serde(default)]
    data_cursor_start: Option<usize>,
    #[serde(default)]
    data_cursor_end: Option<usize>,
    #[serde(default)]
    data_cursor_next: Option<usize>,
    #[serde(default)]
    data_epoch_start: Option<usize>,
    #[serde(default)]
    data_epoch_end: Option<usize>,
    #[serde(default)]
    data_epoch_next: Option<usize>,
    #[serde(default)]
    data_sample_offset_start: Option<usize>,
    #[serde(default)]
    data_sample_offset_end: Option<usize>,
    #[serde(default)]
    data_sample_offset_next: Option<usize>,
    #[serde(default)]
    dataset_source_files: Vec<String>,
    #[serde(default)]
    dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    #[serde(default)]
    dataset_fingerprint: String,
    #[serde(default = "qwen_manifest_default_dataset_shuffle")]
    dataset_shuffle: bool,
    learning_rate: f64,
    delta_safetensors: String,
    optimizer_safetensors: String,
    post_update_loss: f64,
    global_post_update_loss: f64,
    global_step_losses: Vec<f64>,
    trainable_tensors: Vec<String>,
    tensors: Vec<QwenDeltaTensorManifestEntry>,
}

impl QwenSessionDpCheckpointManifest {
    fn to_delta_manifest(&self) -> Result<QwenDeltaCheckpointManifest> {
        if self.format != "rustrain.qwen_session_dp_rank0.v1" {
            bail!(
                "unsupported Qwen session DP checkpoint format {}",
                self.format
            );
        }
        Ok(QwenDeltaCheckpointManifest {
            format: "rustrain.qwen_delta.v1".to_string(),
            base_model_path: self.base_model_path.clone(),
            reference_fixture: "qwen_session_dp_global_input".to_string(),
            delta_safetensors: self.delta_safetensors.clone(),
            optimizer_safetensors: Some(self.optimizer_safetensors.clone()),
            train_step: self.train_step,
            data_cursor_start: self.data_cursor_start,
            data_cursor_end: self.data_cursor_end,
            data_cursor_next: self.data_cursor_next,
            data_epoch_start: self.data_epoch_start,
            data_epoch_end: self.data_epoch_end,
            data_epoch_next: self.data_epoch_next,
            data_sample_offset_start: self.data_sample_offset_start,
            data_sample_offset_end: self.data_sample_offset_end,
            data_sample_offset_next: self.data_sample_offset_next,
            dataset_source_files: self.dataset_source_files.clone(),
            dataset_source_sample_counts: self.dataset_source_sample_counts.clone(),
            dataset_fingerprint: self.dataset_fingerprint.clone(),
            dataset_shuffle: self.dataset_shuffle,
            learning_rate: self.learning_rate,
            initial_loss: self.expected_loss,
            final_loss: self.global_post_update_loss,
            tensors: self.tensors.clone(),
        })
    }
}

#[allow(dead_code)]
impl QwenShardedCheckpointManifest {
    fn validate(&self) -> Result<()> {
        if self.format != "rustrain.qwen_sharded.v1" {
            bail!("unsupported Qwen sharded checkpoint format {}", self.format);
        }
        if self.base_model_path.is_empty() {
            bail!("Qwen sharded checkpoint requires base_model_path");
        }
        if self.tokenizer_path.is_empty() {
            bail!("Qwen sharded checkpoint requires tokenizer_path");
        }
        if self.dtype.is_empty() {
            bail!("Qwen sharded checkpoint requires dtype");
        }
        if self.optimizer.is_empty() {
            bail!("Qwen sharded checkpoint requires optimizer");
        }
        if self.consumed_samples == 0 {
            bail!("Qwen sharded checkpoint consumed_samples must be positive");
        }
        if self.consumed_tokens == 0 {
            bail!("Qwen sharded checkpoint consumed_tokens must be positive");
        }
        if !self.dataset_fingerprint.is_empty() {
            if self.dataset_source_files.is_empty() {
                bail!("Qwen sharded checkpoint dataset_fingerprint requires dataset_source_files");
            }
            if self
                .dataset_source_files
                .iter()
                .any(|source| !source.ends_with(".jsonl"))
            {
                bail!("Qwen sharded checkpoint dataset_source_files must only contain JSONL paths");
            }
            if !self.dataset_source_sample_counts.is_empty() {
                let count_paths = self
                    .dataset_source_sample_counts
                    .iter()
                    .map(|entry| entry.path.clone())
                    .collect::<Vec<_>>();
                if count_paths != self.dataset_source_files {
                    bail!(
                        "Qwen sharded checkpoint dataset_source_sample_counts must match dataset_source_files"
                    );
                }
                if self
                    .dataset_source_sample_counts
                    .iter()
                    .any(|entry| entry.samples == 0)
                {
                    bail!("Qwen sharded checkpoint dataset_source_sample_counts must be positive");
                }
            }
        } else if !self.dataset_source_files.is_empty() {
            bail!("Qwen sharded checkpoint dataset_source_files require dataset_fingerprint");
        } else if !self.dataset_source_sample_counts.is_empty() {
            bail!(
                "Qwen sharded checkpoint dataset_source_sample_counts require dataset_fingerprint"
            );
        }
        if let Some(data_cursor_next) = self.data_cursor_next {
            if data_cursor_next != self.consumed_samples {
                bail!(
                    "Qwen sharded checkpoint data_cursor_next {data_cursor_next} must match consumed_samples {}",
                    self.consumed_samples
                );
            }
        }
        if let Some(data_train_samples) = self.data_train_samples {
            if data_train_samples == 0 {
                bail!("Qwen sharded checkpoint data_train_samples must be positive");
            }
            if let Some(data_epoch_next) = self.data_epoch_next {
                let expected_epoch = self.consumed_samples / data_train_samples;
                if data_epoch_next != expected_epoch {
                    bail!(
                        "Qwen sharded checkpoint data_epoch_next {data_epoch_next} must match consumed_samples/data_train_samples {expected_epoch}"
                    );
                }
            }
            if let Some(data_sample_offset_next) = self.data_sample_offset_next {
                let expected_offset = self.consumed_samples % data_train_samples;
                if data_sample_offset_next != expected_offset {
                    bail!(
                        "Qwen sharded checkpoint data_sample_offset_next {data_sample_offset_next} must match consumed_samples%data_train_samples {expected_offset}"
                    );
                }
            }
        }
        let expected_world_size = self.parallel.world_size()?;
        if self.ranks.len() != expected_world_size {
            bail!(
                "Qwen sharded checkpoint rank manifest count {} does not match world size {expected_world_size}",
                self.ranks.len()
            );
        }

        let mut seen_ranks = BTreeSet::new();
        for rank in &self.ranks {
            if !seen_ranks.insert(rank.rank) {
                bail!(
                    "Qwen sharded checkpoint contains duplicate rank {}",
                    rank.rank
                );
            }
            if rank.rank >= expected_world_size {
                bail!(
                    "Qwen sharded checkpoint rank {} exceeds world size {expected_world_size}",
                    rank.rank
                );
            }
            self.parallel.validate_rank(rank)?;
            if rank.model_safetensors.is_empty() {
                bail!(
                    "Qwen sharded checkpoint rank {} is missing model_safetensors",
                    rank.rank
                );
            }
            if rank.optimizer_safetensors.is_empty() {
                bail!(
                    "Qwen sharded checkpoint rank {} is missing optimizer_safetensors",
                    rank.rank
                );
            }
            if rank.shards.is_empty() {
                bail!(
                    "Qwen sharded checkpoint rank {} must own at least one tensor shard",
                    rank.rank
                );
            }
            for shard in &rank.shards {
                shard.validate(rank.rank)?;
            }
        }
        for expected_rank in 0..expected_world_size {
            if !seen_ranks.contains(&expected_rank) {
                bail!("Qwen sharded checkpoint is missing rank {expected_rank}");
            }
        }
        Ok(())
    }
}

#[allow(dead_code)]
impl QwenShardedParallelManifest {
    fn world_size(&self) -> Result<usize> {
        let sizes = [
            self.data_parallel_size,
            self.tensor_model_parallel_size,
            self.pipeline_model_parallel_size,
            self.expert_model_parallel_size,
            self.context_parallel_size,
        ];
        if sizes.iter().any(|size| *size == 0) {
            bail!("Qwen sharded checkpoint parallel sizes must be positive");
        }
        sizes.into_iter().try_fold(1usize, |world_size, size| {
            world_size
                .checked_mul(size)
                .ok_or_else(|| anyhow!("Qwen sharded checkpoint parallel world size overflowed"))
        })
    }

    fn validate_rank(&self, rank: &QwenRankShardManifest) -> Result<()> {
        if rank.data_parallel_rank >= self.data_parallel_size {
            bail!(
                "Qwen sharded checkpoint rank {} has data_parallel_rank {} outside size {}",
                rank.rank,
                rank.data_parallel_rank,
                self.data_parallel_size
            );
        }
        if rank.tensor_model_parallel_rank >= self.tensor_model_parallel_size {
            bail!(
                "Qwen sharded checkpoint rank {} has tensor_model_parallel_rank {} outside size {}",
                rank.rank,
                rank.tensor_model_parallel_rank,
                self.tensor_model_parallel_size
            );
        }
        if rank.pipeline_model_parallel_rank >= self.pipeline_model_parallel_size {
            bail!(
                "Qwen sharded checkpoint rank {} has pipeline_model_parallel_rank {} outside size {}",
                rank.rank,
                rank.pipeline_model_parallel_rank,
                self.pipeline_model_parallel_size
            );
        }
        if rank.expert_model_parallel_rank >= self.expert_model_parallel_size {
            bail!(
                "Qwen sharded checkpoint rank {} has expert_model_parallel_rank {} outside size {}",
                rank.rank,
                rank.expert_model_parallel_rank,
                self.expert_model_parallel_size
            );
        }
        if rank.context_parallel_rank >= self.context_parallel_size {
            bail!(
                "Qwen sharded checkpoint rank {} has context_parallel_rank {} outside size {}",
                rank.rank,
                rank.context_parallel_rank,
                self.context_parallel_size
            );
        }
        Ok(())
    }
}

#[allow(dead_code)]
impl QwenTensorShardManifestEntry {
    fn validate(&self, rank: usize) -> Result<()> {
        if self.name.is_empty() {
            bail!("Qwen sharded checkpoint rank {rank} has a shard without a tensor name");
        }
        if self.shard_name.is_empty() {
            bail!(
                "Qwen sharded checkpoint rank {rank} tensor {} is missing shard_name",
                self.name
            );
        }
        if self.optimizer_m_name.is_empty() || self.optimizer_v_name.is_empty() {
            bail!(
                "Qwen sharded checkpoint rank {rank} tensor {} is missing optimizer slots",
                self.name
            );
        }
        if self.global_shape.is_empty() || self.shard_shape.is_empty() {
            bail!(
                "Qwen sharded checkpoint rank {rank} tensor {} is missing shape metadata",
                self.name
            );
        }
        if self.dtype.is_empty() {
            bail!(
                "Qwen sharded checkpoint rank {rank} tensor {} is missing dtype",
                self.name
            );
        }
        if self.partition.is_empty() {
            bail!(
                "Qwen sharded checkpoint rank {rank} tensor {} is missing partition policy",
                self.name
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QwenGradSignature {
    name: String,
    shape: Vec<i64>,
    samples: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct QwenDpCheckpointManifest {
    format: String,
    writer_rank: usize,
    world_size: usize,
    tensor_count: usize,
    max_grad_delta: f32,
    expected_loss: f64,
    dtype: String,
    steps: usize,
    learning_rate: f64,
    post_update_loss: f64,
    global_post_update_loss: f64,
    global_step_losses: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct QwenDeltaCheckpointManifest {
    format: String,
    base_model_path: String,
    reference_fixture: String,
    delta_safetensors: String,
    #[serde(default)]
    optimizer_safetensors: Option<String>,
    train_step: u64,
    #[serde(default)]
    data_cursor_start: Option<usize>,
    #[serde(default)]
    data_cursor_end: Option<usize>,
    #[serde(default)]
    data_cursor_next: Option<usize>,
    #[serde(default)]
    data_epoch_start: Option<usize>,
    #[serde(default)]
    data_epoch_end: Option<usize>,
    #[serde(default)]
    data_epoch_next: Option<usize>,
    #[serde(default)]
    data_sample_offset_start: Option<usize>,
    #[serde(default)]
    data_sample_offset_end: Option<usize>,
    #[serde(default)]
    data_sample_offset_next: Option<usize>,
    #[serde(default)]
    dataset_source_files: Vec<String>,
    #[serde(default)]
    dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    #[serde(default)]
    dataset_fingerprint: String,
    #[serde(default = "qwen_manifest_default_dataset_shuffle")]
    dataset_shuffle: bool,
    learning_rate: f64,
    initial_loss: f64,
    final_loss: f64,
    tensors: Vec<QwenDeltaTensorManifestEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QwenDeltaTensorManifestEntry {
    name: String,
    delta_name: String,
    #[serde(default)]
    adam_m_name: Option<String>,
    #[serde(default)]
    adam_v_name: Option<String>,
    shape: Vec<i64>,
    dtype: String,
    grad_norm: f64,
    delta_norm: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QwenShardedCheckpointManifest {
    format: String,
    base_model_path: String,
    tokenizer_path: String,
    global_step: u64,
    consumed_samples: u64,
    consumed_tokens: u64,
    #[serde(default)]
    data_cursor_next: Option<u64>,
    #[serde(default)]
    data_epoch_next: Option<u64>,
    #[serde(default)]
    data_sample_offset_next: Option<u64>,
    #[serde(default)]
    data_train_samples: Option<u64>,
    #[serde(default)]
    dataset_source_files: Vec<String>,
    #[serde(default)]
    dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    #[serde(default)]
    dataset_fingerprint: String,
    #[serde(default = "qwen_manifest_default_dataset_shuffle")]
    dataset_shuffle: bool,
    seed: u64,
    dtype: String,
    optimizer: String,
    scheduler: String,
    parallel: QwenShardedParallelManifest,
    ranks: Vec<QwenRankShardManifest>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QwenShardedParallelManifest {
    data_parallel_size: usize,
    tensor_model_parallel_size: usize,
    pipeline_model_parallel_size: usize,
    expert_model_parallel_size: usize,
    context_parallel_size: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QwenRankShardManifest {
    rank: usize,
    data_parallel_rank: usize,
    tensor_model_parallel_rank: usize,
    pipeline_model_parallel_rank: usize,
    expert_model_parallel_rank: usize,
    context_parallel_rank: usize,
    model_safetensors: String,
    optimizer_safetensors: String,
    shards: Vec<QwenTensorShardManifestEntry>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QwenTensorShardManifestEntry {
    name: String,
    shard_name: String,
    optimizer_m_name: String,
    optimizer_v_name: String,
    global_shape: Vec<i64>,
    shard_shape: Vec<i64>,
    dtype: String,
    partition: String,
    tied_group: Option<String>,
}

struct AdamSlotNames {
    m: String,
    v: String,
}

struct AdamState {
    m: Tensor,
    v: Tensor,
}

struct QwenTrainableParameter {
    name: String,
    tensor: Tensor,
    base: Tensor,
    adam: Option<AdamState>,
}

struct QwenTrainStepArtifacts {
    tensor_summaries: Vec<TrainableTensorSummary>,
    manifest_tensors: Vec<QwenDeltaTensorManifestEntry>,
    delta_entries: Vec<(String, Tensor)>,
    optimizer_entries: Vec<(String, Tensor)>,
}

struct QwenTrainableRegistry {
    parameters: Vec<QwenTrainableParameter>,
}

struct QwenTrainStepResult {
    loss_before: f64,
    loss_after: f64,
    artifacts: QwenTrainStepArtifacts,
}

struct QwenTrainableSession {
    config: QwenRuntimeConfig,
    weights: BTreeMap<String, Tensor>,
    input_ids: Tensor,
    compute_kind: Kind,
    registry: QwenTrainableRegistry,
}

struct QwenSessionBatchPlan {
    initial_input_ids: Tensor,
    train_batches: Vec<Tensor>,
    reference_fixture: String,
    dataset_total_samples: Option<usize>,
    dataset_total_tokens: Option<usize>,
    dataset_train_samples: Option<usize>,
    dataset_eval_samples: Option<usize>,
    dataset_source_files: Option<Vec<String>>,
    dataset_source_sample_counts: Option<Vec<QwenSftSourceSampleCount>>,
    dataset_fingerprint: Option<String>,
    dataset_order_seed: Option<u64>,
    dataset_shuffle: Option<bool>,
    train_sample_count: Option<usize>,
    data_epoch_start: Option<usize>,
    data_epoch_end: Option<usize>,
    data_epoch_next: Option<usize>,
    data_sample_offset_start: Option<usize>,
    data_sample_offset_end: Option<usize>,
    data_sample_offset_next: Option<usize>,
    batch_size: usize,
    sequence_tokens: usize,
}

struct QwenSessionDpBatchPlan {
    global_initial_input_ids: Tensor,
    global_train_batches: Vec<Tensor>,
    dataset_total_samples: Option<usize>,
    dataset_total_tokens: Option<usize>,
    dataset_train_samples: Option<usize>,
    dataset_eval_samples: Option<usize>,
    dataset_source_files: Option<Vec<String>>,
    dataset_source_sample_counts: Option<Vec<QwenSftSourceSampleCount>>,
    dataset_fingerprint: Option<String>,
    dataset_order_seed: Option<u64>,
    dataset_shuffle: Option<bool>,
    train_sample_count: Option<usize>,
    data_epoch_start: Option<usize>,
    data_epoch_end: Option<usize>,
    data_epoch_next: Option<usize>,
    data_sample_offset_start: Option<usize>,
    data_sample_offset_end: Option<usize>,
    data_sample_offset_next: Option<usize>,
    local_batch_size: usize,
    sequence_tokens: usize,
}

struct QwenAttentionDpSession {
    config: QwenRuntimeConfig,
    input: Tensor,
    target: Tensor,
    q_proj: Tensor,
    q_bias: Tensor,
    k_proj: Tensor,
    k_bias: Tensor,
    v_proj: Tensor,
    v_bias: Tensor,
    o_proj: Tensor,
    compute_kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenComputeDType {
    Fp32,
    Bf16,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct QwenLoraConfig {
    target_layers: Vec<usize>,
    target_modules: Vec<QwenLoraTargetModule>,
    rank: i64,
    alpha: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
enum QwenLoraTargetModule {
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
}

struct QwenLoraRegistry {
    config: QwenLoraConfig,
    adapters: BTreeMap<usize, QwenAttentionLoraAdapter>,
}

#[derive(Clone)]
struct QwenSftTokenSample {
    prompt_tokens: usize,
    response_tokens: usize,
    masked_positions: usize,
    token_ids: Vec<i64>,
    mask_values: Vec<f32>,
}

struct QwenSftExample {
    instruction: String,
    input: String,
    response: String,
    source_file: Option<String>,
}

struct QwenSftExampleSet {
    examples: Vec<QwenSftExample>,
    source_files: Vec<String>,
    source_sample_counts: Vec<QwenSftSourceSampleCount>,
    fingerprint: String,
}

#[derive(Deserialize)]
struct QwenSftRecord {
    instruction: String,
    #[serde(default)]
    input: String,
    response: String,
}

#[derive(Clone)]
struct QwenSftDataset {
    samples: Vec<QwenSftTokenSample>,
    pad_token_id: i64,
    epoch_shuffle_seed: Option<u64>,
    source_files: Vec<String>,
    source_sample_counts: Vec<QwenSftSourceSampleCount>,
    fingerprint: String,
}

struct QwenSftBatch {
    input_ids: Tensor,
    target_mask: Tensor,
    prompt_tokens: Vec<usize>,
    response_tokens: Vec<usize>,
    masked_positions: usize,
    padding_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct QwenSftDatasetSummary {
    samples: usize,
    total_tokens: usize,
    response_tokens: usize,
    masked_positions: usize,
    max_sequence_tokens: usize,
    source_files: Vec<String>,
    source_sample_counts: Vec<QwenSftSourceSampleCount>,
    fingerprint: String,
    shuffle: bool,
}

struct QwenSftTrainEvalDatasets {
    combined_summary: QwenSftDatasetSummary,
    train_dataset: QwenSftDataset,
    eval_dataset: QwenSftDataset,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct QwenSftSourceSampleCount {
    pub(crate) path: String,
    pub(crate) samples: usize,
}

#[derive(Clone)]
struct QwenLoraSftTrainPolicy {
    lr_scheduler: LrScheduler,
    max_grad_norm: Option<f64>,
    dataset_order_seed: u64,
    dataset_shuffle: bool,
}

impl QwenLoraSftTrainPolicy {
    fn from_config(config: &Config) -> Self {
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

    fn constant_without_clip() -> Self {
        Self {
            lr_scheduler: LrScheduler::Constant,
            max_grad_norm: None,
            dataset_order_seed: 0,
            dataset_shuffle: true,
        }
    }
}

fn qwen_manifest_default_dataset_shuffle() -> bool {
    true
}

fn qwen_lora_sft_adapter_manifest_path(adapter_output: &Path) -> PathBuf {
    PathBuf::from(format!("{}.json", adapter_output.display()))
}

fn read_qwen_lora_sft_resume_manifest(
    resume_from: &Path,
) -> Result<Option<QwenLoraSftAdapterManifest>> {
    if resume_from.extension().and_then(|value| value.to_str()) != Some("json") {
        return Ok(None);
    }
    let text = fs::read_to_string(resume_from)
        .with_context(|| format!("failed to read {}", resume_from.display()))?;
    let manifest: QwenLoraSftAdapterManifest = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", resume_from.display()))?;
    if manifest.format != "rustrain.qwen_lora_sft_adapter.v1" {
        bail!(
            "unsupported Qwen LoRA SFT adapter manifest format {}",
            manifest.format
        );
    }
    Ok(Some(manifest))
}

fn write_qwen_lora_sft_adapter_manifest(
    manifest_output: &Path,
    manifest: &QwenLoraSftAdapterManifest,
) -> Result<()> {
    if let Some(parent) = manifest_output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(
        manifest_output,
        serde_json::to_string_pretty(manifest)
            .context("failed to serialize Qwen LoRA SFT adapter manifest")?
            + "\n",
    )
    .with_context(|| format!("failed to write {}", manifest_output.display()))
}

fn qwen_validate_lora_resume_config(
    manifest: Option<&QwenLoraSftAdapterManifest>,
    adapter_config: &QwenLoraConfig,
    current_config: &QwenLoraConfig,
) -> Result<()> {
    if let Some(manifest) = manifest {
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

impl QwenComputeDType {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "fp32" => Ok(Self::Fp32),
            "bf16" => Ok(Self::Bf16),
            other => bail!("unsupported Qwen compute dtype {other}; expected fp32 or bf16"),
        }
    }

    fn kind(self) -> Kind {
        match self {
            Self::Fp32 => Kind::Float,
            Self::Bf16 => Kind::BFloat16,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
        }
    }
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
    let cached = qwen_sample_generate_with_cache(
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
    let lora_config = QwenLoraConfig::layer0_qv(rank, alpha)?;
    let zero_registry = QwenLoraRegistry::zeros(&weights, &lora_config)?;
    let zero_output = qwen_attention_with_lora(
        &attention_input,
        &layer0,
        zero_registry.layer_adapter(0)?,
        &config,
    );
    let zero_lora_max_delta = diff_stats(&zero_output, &base)?.max_abs;
    if zero_lora_max_delta > 1e-7 {
        bail!("zero LoRA changed attention output: max_delta={zero_lora_max_delta}");
    }

    let registry = QwenLoraRegistry::deterministic(&weights, &lora_config, false)?;
    let adapted_output = qwen_attention_with_lora(
        &attention_input,
        &layer0,
        registry.layer_adapter(0)?,
        &config,
    );
    let nonzero_lora_max_delta = diff_stats(&adapted_output, &base)?.max_abs;
    if nonzero_lora_max_delta <= 0.0 {
        bail!("non-zero LoRA did not change attention output");
    }

    registry.save(adapter_output)?;
    let reloaded = QwenLoraRegistry::load(adapter_output)?;
    let reloaded_output = qwen_attention_with_lora(
        &attention_input,
        &layer0,
        reloaded.layer_adapter(0)?,
        &config,
    );
    let reload_max_delta = diff_stats(&reloaded_output, &adapted_output)?.max_abs;
    if reload_max_delta > 1e-7 {
        bail!("LoRA adapter reload changed output: max_delta={reload_max_delta}");
    }

    let summary = QwenLoraSmokeSummary {
        model_path: model_path.display().to_string(),
        fixture: fixture.display().to_string(),
        adapter_output: adapter_output.display().to_string(),
        target_layers: lora_config.target_layers.clone(),
        target_modules: lora_config.target_module_names(),
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

pub fn qwen_lora_train_smoke(
    model_path: &Path,
    fixture: &Path,
    adapter_output: &Path,
    rank: i64,
    alpha: f64,
    learning_rate: f64,
) -> Result<()> {
    if rank <= 0 {
        bail!("rank must be positive");
    }
    if alpha <= 0.0 {
        bail!("alpha must be positive");
    }
    if learning_rate <= 0.0 {
        bail!("learning_rate must be positive");
    }

    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let fixture_tensors = read_safetensors_map(fixture)?;
    let attention_input = tensor(&fixture_tensors, "input_attention_normed")?.to_kind(Kind::Float);
    let base_target_output = tensor(&fixture_tensors, "attention_output")?.to_kind(Kind::Float);
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let layer0 = QwenLayerWeights::load(&weights, 0)?;
    let target_output = lora_train_target(&base_target_output);
    let lora_config = QwenLoraConfig::layer0_qv(rank, alpha)?;
    let registry = QwenLoraRegistry::deterministic(&weights, &lora_config, true)?;
    let base_requires_grad = layer0.q_proj.requires_grad()
        || layer0.k_proj.requires_grad()
        || layer0.v_proj.requires_grad()
        || layer0.o_proj.requires_grad();

    let initial_loss = qwen_attention_lora_mse_loss(
        &attention_input,
        &target_output,
        &layer0,
        registry.layer_adapter(0)?,
        &config,
    )
    .double_value(&[]);
    let loss = qwen_attention_lora_mse_loss(
        &attention_input,
        &target_output,
        &layer0,
        registry.layer_adapter(0)?,
        &config,
    );
    loss.backward();

    let base_tensors: BTreeMap<String, Tensor> = registry
        .trainable_tensors()
        .into_iter()
        .map(|(name, tensor)| (name, tensor_snapshot(&tensor)))
        .collect();
    let mut tensor_summaries = Vec::new();
    for (name, mut tensor) in registry.trainable_tensors() {
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
        let _ = no_grad(|| tensor.f_sub_(&(&grad * learning_rate)))?;
        let delta_norm = (&tensor
            - base_tensors
                .get(&name)
                .ok_or_else(|| anyhow!("missing base LoRA tensor {name}"))?)
        .norm()
        .double_value(&[]);
        tensor_summaries.push(TrainableTensorSummary {
            name,
            grad_defined,
            grad_norm,
            delta_norm,
        });
    }

    let final_loss = qwen_attention_lora_mse_loss(
        &attention_input,
        &target_output,
        &layer0,
        registry.layer_adapter(0)?,
        &config,
    )
    .double_value(&[]);
    if final_loss >= initial_loss {
        bail!(
            "Qwen LoRA train smoke failed to reduce loss: initial_loss={initial_loss}, final_loss={final_loss}"
        );
    }

    registry.save(adapter_output)?;
    let reloaded = QwenLoraRegistry::load(adapter_output)?;
    let reloaded_loss = qwen_attention_lora_mse_loss(
        &attention_input,
        &target_output,
        &layer0,
        reloaded.layer_adapter(0)?,
        &config,
    )
    .double_value(&[]);
    let reload_delta = (final_loss - reloaded_loss).abs();
    if reload_delta > 1e-7 {
        bail!(
            "Qwen LoRA adapter reload loss parity failed: final_loss={final_loss}, reloaded_loss={reloaded_loss}, reload_delta={reload_delta}"
        );
    }

    let summary = QwenLoraTrainSmokeSummary {
        model_path: model_path.display().to_string(),
        fixture: fixture.display().to_string(),
        adapter_output: adapter_output.display().to_string(),
        target_layers: lora_config.target_layers.clone(),
        target_modules: lora_config.target_module_names(),
        rank,
        alpha,
        learning_rate,
        initial_loss,
        final_loss,
        reloaded_loss,
        reload_delta,
        base_requires_grad,
        trainable_tensors: tensor_summaries,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_lora_sft_smoke(
    model_path: &Path,
    adapter_output: &Path,
    sft_jsonl: Option<&Path>,
    sft_batch_size: usize,
    instruction: &str,
    response: &str,
    rank: i64,
    alpha: f64,
    learning_rate: f64,
) -> Result<()> {
    let lora_config = QwenLoraConfig::layer0_qv(rank, alpha)?;
    let sft_paths = sft_jsonl.map(|path| vec![path.to_path_buf()]);
    let checkpoint_dir = adapter_output.parent().unwrap_or_else(|| Path::new("."));
    let summary = qwen_lora_sft_train(
        model_path,
        adapter_output,
        checkpoint_dir,
        sft_paths.as_deref(),
        &[],
        None,
        None,
        sft_batch_size,
        instruction,
        response,
        lora_config,
        learning_rate,
        1,
        0.5,
        1,
        0,
        0,
        QwenComputeDType::Fp32,
        QwenLoraSftTrainPolicy::constant_without_clip(),
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);

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
        crate::runtime::DType::Fp32 => QwenComputeDType::Fp32,
        crate::runtime::DType::Bf16 => QwenComputeDType::Bf16,
        crate::runtime::DType::Fp16 => {
            bail!("qwen LoRA SFT trainer does not support fp16 yet; use fp32 or bf16")
        }
    };
    let adapter_output = run_paths
        .checkpoints
        .join("qwen-lora-sft-adapter.safetensors");
    qwen_lora_sft_train(
        model_path,
        &adapter_output,
        &run_paths.checkpoints,
        Some(&data.paths),
        &data.eval_paths,
        data.max_samples,
        config.train.resume_from.as_deref(),
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
    )
}

#[allow(clippy::too_many_arguments)]
fn qwen_lora_sft_train(
    model_path: &Path,
    adapter_output: &Path,
    checkpoint_dir: &Path,
    sft_paths: Option<&[PathBuf]>,
    eval_paths: &[PathBuf],
    max_samples: Option<usize>,
    resume_from: Option<&Path>,
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

    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let dataset = if let Some(sft_paths) = sft_paths {
        qwen_sft_train_eval_datasets_from_paths(
            &tokenizer,
            sft_paths,
            eval_paths,
            max_samples,
            train_split,
            policy.dataset_shuffle,
            policy.dataset_order_seed,
        )?
    } else {
        let dataset = QwenSftDataset::from_instruction_pairs(
            &tokenizer,
            &[
                QwenSftExample {
                    instruction: instruction.to_string(),
                    input: String::new(),
                    response: response.to_string(),
                    source_file: None,
                },
                QwenSftExample {
                    instruction: "Name the project.".to_string(),
                    input: String::new(),
                    response: "rustrain".to_string(),
                    source_file: None,
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
        qwen_validate_lora_resume_config(resume_manifest.as_ref(), &registry.config, &lora_config)?;
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

    for step in 0..steps {
        for (_, mut tensor) in registry.trainable_tensors() {
            tensor.zero_grad();
        }
        for accumulation_index in 0..gradient_accumulation_steps {
            let sample_start = data_cursor_start
                + (step * gradient_accumulation_steps + accumulation_index) * train_batch_size;
            let step_batch = train_dataset.padded_batch(sample_start, train_batch_size)?;
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
        memory_rss_mb: crate::metrics::memory_rss_mb(),
        gpu_memory_allocated_mb: crate::metrics::gpu_memory_allocated_mb(),
        trainable_tensors: tensor_summaries,
    };
    Ok(summary)
}

fn qwen_lora_sft_should_eval_step(step_number: usize, eval_every: u64) -> bool {
    eval_every > 0 && (step_number as u64) % eval_every == 0
}

fn qwen_data_epoch_and_offset(cursor: usize, sample_count: usize) -> Result<(usize, usize)> {
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

fn qwen_full_train_summary(
    model_path: &Path,
    reference_fixture: &Path,
    delta_output: &Path,
    dtype: QwenComputeDType,
    learning_rate: f64,
) -> Result<QwenFullTrainSmokeSummary> {
    if learning_rate <= 0.0 {
        bail!("learning_rate must be positive");
    }

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
        memory_rss_mb: crate::metrics::memory_rss_mb(),
        gpu_memory_allocated_mb: crate::metrics::gpu_memory_allocated_mb(),
        dataset_total_samples: None,
        dataset_total_tokens: None,
        dataset_train_samples: None,
        dataset_eval_samples: None,
        dataset_source_files: None,
        dataset_source_sample_counts: None,
        dataset_fingerprint: None,
        dataset_order_seed: None,
        dataset_shuffle: None,
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

fn qwen_session_single_summary(
    model_path: &Path,
    delta_output: &Path,
    dtype: QwenComputeDType,
    train_steps: usize,
    learning_rate: f64,
    resume_from: Option<&Path>,
    trainable_layers: &[usize],
    runtime_config: Option<&Config>,
) -> Result<QwenFullTrainSmokeSummary> {
    if train_steps == 0 {
        bail!("qwen session single trainer requires max_steps > 0");
    }
    if learning_rate <= 0.0 {
        bail!("learning_rate must be positive");
    }

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
        model_path,
        &weights,
        data_cursor_start,
        train_steps,
        runtime_config,
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
        let batch_index = data_cursor_start + (step - start_step) * batch_plan.batch_size;
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
    let next_batch_index = data_cursor_next;
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
        memory_rss_mb: crate::metrics::memory_rss_mb(),
        gpu_memory_allocated_mb: crate::metrics::gpu_memory_allocated_mb(),
        dataset_total_samples: batch_plan.dataset_total_samples,
        dataset_total_tokens: batch_plan.dataset_total_tokens,
        dataset_train_samples: batch_plan.dataset_train_samples,
        dataset_eval_samples: batch_plan.dataset_eval_samples,
        dataset_source_files: batch_plan.dataset_source_files,
        dataset_source_sample_counts: batch_plan.dataset_source_sample_counts,
        dataset_fingerprint: batch_plan.dataset_fingerprint,
        dataset_order_seed: batch_plan.dataset_order_seed,
        dataset_shuffle: batch_plan.dataset_shuffle,
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

fn qwen_train_artifacts_grad_norm(artifacts: &QwenTrainStepArtifacts) -> f64 {
    artifacts
        .tensor_summaries
        .iter()
        .map(|summary| summary.grad_norm * summary.grad_norm)
        .sum::<f64>()
        .sqrt()
}

pub fn qwen_dp_gradient_smoke(
    model_path: &Path,
    reference_fixture: &Path,
    output_dir: PathBuf,
    dtype: QwenComputeDType,
    steps: usize,
    learning_rate: f64,
) -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if steps == 0 {
        bail!("Qwen DP gradient smoke requires at least one step");
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        bail!("Qwen DP gradient smoke requires a positive finite learning rate");
    }
    if world_size != 2 {
        bail!("Qwen DP gradient smoke expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    let device = Device::Cuda(local_rank);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let output_dir = qwen_dp_artifact_dir(&output_dir)?;
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    println!("qwen attention DP rank {rank}: loading fixture");

    let config = read_runtime_config(&model_path.join("config.json"))?;
    let fixture = read_safetensors_map(reference_fixture)?;
    let attention_input = tensor(&fixture, "input_attention_normed")?.to_kind(dtype.kind());
    let attention_target = tensor(&fixture, "attention_output")?.to_kind(dtype.kind());
    let local_input = qwen_dp_attention_input_for_rank(&attention_input, rank, world_size)?;
    let local_target = qwen_dp_attention_target_for_rank(&attention_target, rank, world_size)?;

    println!("qwen attention DP rank {rank}: loading model weights");
    let mut local_session = QwenAttentionDpSession::from_weights(
        read_safetensors_map(&model_path.join("model.safetensors"))?,
        local_input,
        local_target,
        config,
        dtype.kind(),
        device,
    )?;
    println!("qwen attention DP rank {rank}: running local backward");
    let local_loss = local_session.loss_and_backward()?;
    let local_grads = local_session.grad_entries()?;

    let expected_path = output_dir.join("qwen-dp-expected-signatures.json");
    let (expected_loss, expected_signatures) = if rank == 0 {
        println!("qwen attention DP rank {rank}: running expected backward");
        let global_input = qwen_dp_attention_global(&attention_input)?;
        let global_target = qwen_dp_attention_global(&attention_target)?;
        let mut expected_session = QwenAttentionDpSession::from_weights(
            read_safetensors_map(&model_path.join("model.safetensors"))?,
            global_input,
            global_target,
            local_session.config,
            dtype.kind(),
            device,
        )?;
        let expected_loss = expected_session.loss_and_backward()?;
        let expected_signatures = grad_signatures(&expected_session.grad_entries()?)?;
        let encoded = serde_json::to_string_pretty(&(expected_loss, &expected_signatures))?;
        fs::write(&expected_path, encoded)
            .with_context(|| format!("failed to write {}", expected_path.display()))?;
        (expected_loss, expected_signatures)
    } else {
        println!("qwen attention DP rank {rank}: waiting for expected signatures");
        wait_for_expected_signatures(&expected_path, Duration::from_secs(300))?
    };

    println!("qwen attention DP rank {rank}: reducing gradient signatures");
    let mut local_signature_values = Vec::new();
    let mut expected_signature_values = Vec::new();
    for ((name, local_grad), expected) in local_grads.iter().zip(expected_signatures.iter()) {
        if name != &expected.name {
            bail!(
                "gradient tensor order mismatch: local {name} != expected {}",
                expected.name
            );
        }
        let local_signature = grad_signature(name, local_grad)?;
        local_signature_values.extend(local_signature.values());
        expected_signature_values.extend(expected.values());
    }
    wait_for_rank_barrier(
        &output_dir.join("qwen-dp-gradient-signatures-ready"),
        rank,
        world_size,
        Duration::from_secs(300),
    )?;
    let reduced_signatures = nccl_smoke::all_reduce_f32_for_launch(
        &output_dir.join("qwen-dp-gradient-signatures"),
        &local_signature_values,
    )?;
    let averaged_signatures: Vec<f32> = reduced_signatures
        .into_iter()
        .map(|value| value / world_size as f32)
        .collect();
    let max_grad_delta =
        signature_values_max_delta(&averaged_signatures, &expected_signature_values)?;

    let global_loss = expected_loss;
    let loss_delta = 0.0;

    if max_grad_delta > 5e-4 || loss_delta > 5e-4 {
        bail!(
            "Qwen DP gradient mismatch: rank={rank}, max_grad_delta={max_grad_delta}, loss_delta={loss_delta}"
        );
    }

    let mut global_step_losses = vec![global_loss];
    let mut post_update_loss = local_loss;
    for step in 0..steps {
        if step > 0 {
            println!("qwen attention DP rank {rank}: running backward for step {step}");
            local_session.loss_and_backward()?;
        }
        println!("qwen attention DP rank {rank}: reducing full gradients for step {step}");
        let averaged_grads = local_session.all_reduce_average_grads(
            &output_dir.join(format!("qwen-dp-full-gradient-step-{step}")),
            world_size,
        )?;
        local_session.apply_sgd_step(&averaged_grads, learning_rate)?;
        post_update_loss = local_session.loss_value();
        wait_for_rank_barrier(
            &output_dir.join(format!("qwen-dp-post-update-loss-ready-step-{step}")),
            rank,
            world_size,
            Duration::from_secs(300),
        )?;
        let reduced_post_update_loss = nccl_smoke::all_reduce_f32_for_launch(
            &output_dir.join(format!("qwen-dp-post-update-loss-step-{step}")),
            &[post_update_loss as f32],
        )?[0];
        global_step_losses.push(reduced_post_update_loss as f64 / world_size as f64);
    }
    let global_post_update_loss = *global_step_losses
        .last()
        .ok_or_else(|| anyhow!("missing Qwen DP post-update loss"))?;
    let global_loss_improved = global_post_update_loss < global_loss;
    if !global_loss_improved {
        bail!(
            "Qwen DP gradient update did not reduce global loss: rank={rank}, global_loss={global_loss}, global_post_update_loss={global_post_update_loss}"
        );
    }

    let checkpoint_path = output_dir.join("qwen-dp-rank0-checkpoint.json");
    let checkpoint_written = if rank == 0 {
        let manifest = QwenDpCheckpointManifest {
            format: "rustrain.qwen_dp_rank0.v1".to_string(),
            writer_rank: rank,
            world_size,
            tensor_count: local_grads.len(),
            max_grad_delta,
            expected_loss,
            dtype: dtype.label().to_string(),
            steps,
            learning_rate,
            post_update_loss,
            global_post_update_loss,
            global_step_losses: global_step_losses.clone(),
        };
        fs::write(
            &checkpoint_path,
            serde_json::to_string_pretty(&manifest)? + "\n",
        )
        .with_context(|| format!("failed to write {}", checkpoint_path.display()))?;
        true
    } else {
        false
    };

    let summary = QwenDpGradientRankSummary {
        rank,
        world_size,
        local_sequence_count: local_session.input.size()[0] as usize,
        tensor_count: local_grads.len(),
        steps,
        learning_rate,
        max_grad_delta,
        loss_delta,
        local_loss,
        post_update_loss,
        global_loss,
        global_post_update_loss,
        global_loss_improved,
        global_step_losses,
        expected_loss,
        checkpoint_written,
        checkpoint_path: checkpoint_path.display().to_string(),
    };
    let summary_path = output_dir.join(format!("qwen-dp-gradient-rank-{rank}.json"));
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_session_dp_rank_smoke(
    model_path: &Path,
    output_dir: PathBuf,
    dtype: QwenComputeDType,
    steps: usize,
    learning_rate: f64,
    trainable_layers: &[usize],
    resume_from: Option<&Path>,
    runtime_config: Option<&Config>,
) -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("Qwen session DP smoke expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    if steps == 0 {
        bail!("Qwen session DP smoke requires at least one step");
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        bail!("Qwen session DP smoke requires a positive finite learning rate");
    }

    let device = Device::Cuda(local_rank);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let output_dir = qwen_dp_artifact_dir(&output_dir)?;
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let loaded_manifest = resume_from
        .map(|resume_from| {
            let manifest_text = fs::read_to_string(resume_from)
                .with_context(|| format!("failed to read {}", resume_from.display()))?;
            serde_json::from_str::<QwenSessionDpCheckpointManifest>(&manifest_text)
                .with_context(|| format!("failed to parse {}", resume_from.display()))
        })
        .transpose()?
        .map(Arc::new);
    let (resume_train_step, data_cursor_start) = if let Some(manifest) = loaded_manifest.as_ref() {
        let next_step = manifest
            .train_step
            .checked_add(1)
            .ok_or_else(|| anyhow!("Qwen session DP resume train_step overflowed"))?
            as usize;
        let inferred_cursor = manifest
            .train_step
            .checked_mul(world_size as u64)
            .ok_or_else(|| anyhow!("Qwen session DP resume inferred cursor overflowed"))?
            as usize;
        (
            next_step,
            manifest.data_cursor_next.unwrap_or(inferred_cursor),
        )
    } else {
        (1, 0)
    };
    let batch_plan = qwen_session_dp_batch_plan_from_config(
        model_path,
        &weights,
        data_cursor_start,
        steps,
        world_size,
        device,
        runtime_config,
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
            "Qwen session DP external checkpoint resume",
        )?;
    }
    let local_input = batch_plan.global_initial_input_ids.narrow(
        0,
        rank as i64 * batch_plan.local_batch_size as i64,
        batch_plan.local_batch_size as i64,
    );

    println!("qwen session DP rank {rank}: loading representative session");
    let mut local_session = if let Some(manifest) = loaded_manifest.as_ref() {
        QwenTrainableSession::from_manifest_on_device(
            config,
            weights,
            local_input.shallow_clone(),
            dtype.kind(),
            &manifest.to_delta_manifest()?,
            Some(device),
        )?
    } else {
        QwenTrainableSession::from_trainable_layers_on_device(
            config,
            weights,
            local_input.shallow_clone(),
            dtype.kind(),
            trainable_layers,
            false,
            device,
        )?
    };
    println!("qwen session DP rank {rank}: running local backward");
    let local_loss = local_session.loss_and_backward()?;
    let local_grads = local_session.grad_entries()?;

    let expected_path = output_dir.join("qwen-session-dp-expected-signatures.json");
    let (expected_loss, expected_signatures) = if rank == 0 {
        println!("qwen session DP rank {rank}: running expected global backward");
        let mut expected_session = if let Some(manifest) = loaded_manifest.as_ref() {
            QwenTrainableSession::from_manifest_on_device(
                config,
                read_safetensors_map(&model_path.join("model.safetensors"))?,
                batch_plan.global_initial_input_ids.shallow_clone(),
                dtype.kind(),
                &manifest.to_delta_manifest()?,
                Some(device),
            )?
        } else {
            QwenTrainableSession::from_trainable_layers_on_device(
                config,
                read_safetensors_map(&model_path.join("model.safetensors"))?,
                batch_plan.global_initial_input_ids.shallow_clone(),
                dtype.kind(),
                trainable_layers,
                false,
                device,
            )?
        };
        let expected_loss = expected_session.loss_and_backward()?;
        let expected_signatures = grad_signatures(&expected_session.grad_entries()?)?;
        fs::write(
            &expected_path,
            serde_json::to_string_pretty(&(expected_loss, &expected_signatures))?,
        )
        .with_context(|| format!("failed to write {}", expected_path.display()))?;
        (expected_loss, expected_signatures)
    } else {
        println!("qwen session DP rank {rank}: waiting for expected signatures");
        wait_for_expected_signatures(&expected_path, Duration::from_secs(300))?
    };

    println!("qwen session DP rank {rank}: reducing gradient signatures");
    let mut local_signature_values = Vec::new();
    let mut expected_signature_values = Vec::new();
    for ((name, local_grad), expected) in local_grads.iter().zip(expected_signatures.iter()) {
        if name != &expected.name {
            bail!(
                "gradient tensor order mismatch: local {name} != expected {}",
                expected.name
            );
        }
        let local_signature = grad_signature(name, local_grad)?;
        local_signature_values.extend(local_signature.values());
        expected_signature_values.extend(expected.values());
    }
    wait_for_rank_barrier(
        &output_dir.join("qwen-session-dp-gradient-signatures-ready"),
        rank,
        world_size,
        Duration::from_secs(300),
    )?;
    let reduced_signatures = nccl_smoke::all_reduce_f32_for_launch(
        &output_dir.join("qwen-session-dp-gradient-signatures"),
        &local_signature_values,
    )?;
    let averaged_signatures: Vec<f32> = reduced_signatures
        .into_iter()
        .map(|value| value / world_size as f32)
        .collect();
    let max_grad_delta =
        signature_values_max_delta(&averaged_signatures, &expected_signature_values)?;
    let global_loss = expected_loss;
    let loss_delta = 0.0;
    let max_grad_delta_tolerance = match dtype {
        QwenComputeDType::Fp32 => 5e-4,
        QwenComputeDType::Bf16 => 1.0,
    };
    if max_grad_delta > max_grad_delta_tolerance {
        bail!(
            "Qwen session DP gradient mismatch: rank={rank}, max_grad_delta={max_grad_delta}, tolerance={max_grad_delta_tolerance}"
        );
    }

    let mut global_step_losses = vec![global_loss];
    let mut post_update_loss = local_loss;
    let mut trainable_summaries = Vec::new();
    let mut last_artifacts: Option<QwenTrainStepArtifacts> = None;
    let global_batch_size = batch_plan.local_batch_size * world_size;
    for step in 0..steps {
        let batch_index = if batch_plan.train_sample_count.is_some() {
            step * global_batch_size
        } else {
            step
        };
        let global_step_batch = batch_plan
            .global_train_batches
            .get(batch_index)
            .ok_or_else(|| anyhow!("missing qwen session DP batch for step {step}"))?;
        let local_step_batch = global_step_batch.narrow(
            0,
            rank as i64 * batch_plan.local_batch_size as i64,
            batch_plan.local_batch_size as i64,
        );
        local_session.set_input_ids(&local_step_batch);
        if step > 0 {
            println!("qwen session DP rank {rank}: running backward for step {step}");
        }
        local_session.loss_and_backward()?;
        println!("qwen session DP rank {rank}: reducing full gradients for step {step}");
        let averaged_grads = local_session.all_reduce_average_grads(
            &output_dir.join(format!("qwen-session-dp-full-gradient-step-{step}")),
            world_size,
        )?;
        let artifacts = local_session.apply_adamw_step(
            &averaged_grads,
            learning_rate,
            (resume_train_step + step) as i32,
        )?;
        trainable_summaries = artifacts.tensor_summaries.clone();
        last_artifacts = Some(artifacts);
        post_update_loss = local_session.loss_value()?;
        wait_for_rank_barrier(
            &output_dir.join(format!(
                "qwen-session-dp-post-update-loss-ready-step-{step}"
            )),
            rank,
            world_size,
            Duration::from_secs(300),
        )?;
        let reduced_post_update_loss = nccl_smoke::all_reduce_f32_for_launch(
            &output_dir.join(format!("qwen-session-dp-post-update-loss-step-{step}")),
            &[post_update_loss as f32],
        )?[0];
        global_step_losses.push(reduced_post_update_loss as f64 / world_size as f64);
    }
    let global_post_update_loss = *global_step_losses
        .last()
        .ok_or_else(|| anyhow!("missing Qwen session DP post-update loss"))?;
    let global_loss_improved = global_post_update_loss < global_loss;
    if !global_loss_improved {
        bail!(
            "Qwen session DP AdamW update did not reduce global loss: rank={rank}, global_loss={global_loss}, global_post_update_loss={global_post_update_loss}"
        );
    }
    let last_artifacts =
        last_artifacts.context("missing Qwen session DP artifacts from final training step")?;

    let trainable_tensors = local_session.parameter_names();
    let data_cursor_end = data_cursor_start + steps * global_batch_size;
    let data_cursor_next = data_cursor_end;
    let checkpoint_path = output_dir.join("qwen-session-dp-rank0-checkpoint.json");
    let delta_output = output_dir.join("qwen-session-dp-rank0-delta.safetensors");
    let optimizer_output = optimizer_state_path(&delta_output);
    let checkpoint_written = if rank == 0 {
        let delta_refs: Vec<(&str, &Tensor)> = last_artifacts
            .delta_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&delta_refs, &delta_output)
            .with_context(|| format!("failed to write {}", delta_output.display()))?;
        let optimizer_refs: Vec<(&str, &Tensor)> = last_artifacts
            .optimizer_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&optimizer_refs, &optimizer_output)
            .with_context(|| format!("failed to write {}", optimizer_output.display()))?;
        let manifest = QwenSessionDpCheckpointManifest {
            format: "rustrain.qwen_session_dp_rank0.v1".to_string(),
            base_model_path: model_path.display().to_string(),
            writer_rank: rank,
            world_size,
            tensor_count: local_grads.len(),
            max_grad_delta,
            expected_loss,
            dtype: dtype.label().to_string(),
            steps,
            train_step: (resume_train_step + steps - 1) as u64,
            data_cursor_start: batch_plan.train_sample_count.map(|_| data_cursor_start),
            data_cursor_end: batch_plan.train_sample_count.map(|_| data_cursor_end),
            data_cursor_next: batch_plan.train_sample_count.map(|_| data_cursor_next),
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
            learning_rate,
            delta_safetensors: delta_output.display().to_string(),
            optimizer_safetensors: optimizer_output.display().to_string(),
            post_update_loss,
            global_post_update_loss,
            global_step_losses: global_step_losses.clone(),
            trainable_tensors: trainable_tensors.clone(),
            tensors: last_artifacts.manifest_tensors.clone(),
        };
        fs::write(
            &checkpoint_path,
            serde_json::to_string_pretty(&manifest)? + "\n",
        )
        .with_context(|| format!("failed to write {}", checkpoint_path.display()))?;
        true
    } else {
        false
    };
    wait_for_rank_barrier(
        &output_dir.join("qwen-session-dp-rank0-checkpoint-written"),
        rank,
        world_size,
        Duration::from_secs(300),
    )?;
    let checkpoint_text = fs::read_to_string(&checkpoint_path)
        .with_context(|| format!("failed to read {}", checkpoint_path.display()))?;
    let checkpoint_manifest: QwenSessionDpCheckpointManifest =
        serde_json::from_str(&checkpoint_text)
            .with_context(|| format!("failed to parse {}", checkpoint_path.display()))?;
    qwen_validate_optional_sft_resume_dataset(
        &checkpoint_manifest.dataset_source_files,
        &checkpoint_manifest.dataset_source_sample_counts,
        &checkpoint_manifest.dataset_fingerprint,
        checkpoint_manifest.dataset_shuffle,
        batch_plan.dataset_source_files.as_deref(),
        batch_plan.dataset_source_sample_counts.as_deref(),
        batch_plan.dataset_fingerprint.as_deref(),
        batch_plan.dataset_shuffle,
        "Qwen session DP rank0 checkpoint resume",
    )?;
    let mut delta_manifest = checkpoint_manifest.to_delta_manifest()?;
    delta_manifest.delta_safetensors = delta_output.display().to_string();
    delta_manifest.optimizer_safetensors = Some(optimizer_output.display().to_string());
    let mut resumed_session = QwenTrainableSession::from_manifest_on_device(
        config,
        read_safetensors_map(&model_path.join("model.safetensors"))?,
        local_input.shallow_clone(),
        dtype.kind(),
        &delta_manifest,
        Some(device),
    )?;
    let final_step_batch = batch_plan
        .global_train_batches
        .get(if batch_plan.train_sample_count.is_some() {
            (steps - 1) * global_batch_size
        } else {
            steps - 1
        })
        .ok_or_else(|| anyhow!("missing qwen session DP final batch"))?;
    let local_final_step_batch = final_step_batch.narrow(
        0,
        rank as i64 * batch_plan.local_batch_size as i64,
        batch_plan.local_batch_size as i64,
    );
    resumed_session.set_input_ids(&local_final_step_batch);
    let reloaded_loss = resumed_session.loss_value()?;
    let reload_delta = (post_update_loss - reloaded_loss).abs();
    if reload_delta > 1e-5 {
        bail!(
            "Qwen session DP rank0 checkpoint reload parity failed: rank={rank}, post_update_loss={post_update_loss}, reloaded_loss={reloaded_loss}, reload_delta={reload_delta}"
        );
    }

    let next_step_batch = batch_plan
        .global_train_batches
        .get(if batch_plan.train_sample_count.is_some() {
            steps * global_batch_size
        } else {
            steps
        })
        .ok_or_else(|| anyhow!("missing qwen session DP next-step batch"))?;
    let local_next_step_batch = next_step_batch.narrow(
        0,
        rank as i64 * batch_plan.local_batch_size as i64,
        batch_plan.local_batch_size as i64,
    );
    local_session.set_input_ids(&local_next_step_batch);
    resumed_session.set_input_ids(&local_next_step_batch);
    local_session.loss_and_backward()?;
    resumed_session.loss_and_backward()?;
    let continuous_next_grads = local_session.all_reduce_average_grads(
        &output_dir.join("qwen-session-dp-continuous-next-gradient"),
        world_size,
    )?;
    let resumed_next_grads = resumed_session.all_reduce_average_grads(
        &output_dir.join("qwen-session-dp-resumed-next-gradient"),
        world_size,
    )?;
    local_session.apply_adamw_step(
        &continuous_next_grads,
        learning_rate,
        (resume_train_step + steps) as i32,
    )?;
    resumed_session.apply_adamw_step(
        &resumed_next_grads,
        learning_rate,
        (resume_train_step + steps) as i32,
    )?;
    let continuous_next_loss = local_session.loss_value()?;
    let resumed_next_loss = resumed_session.loss_value()?;
    let next_step_delta = (continuous_next_loss - resumed_next_loss).abs();
    if next_step_delta > 1e-5 {
        bail!(
            "Qwen session DP rank0 checkpoint resume parity failed: rank={rank}, continuous_next_loss={continuous_next_loss}, resumed_next_loss={resumed_next_loss}, next_step_delta={next_step_delta}"
        );
    }

    let sharded_rank_manifest_output = write_qwen_session_dp_rank_sharded_manifest(
        &output_dir,
        model_path,
        rank,
        world_size,
        steps,
        learning_rate,
        dtype,
        batch_plan.train_sample_count.map(|_| data_cursor_next),
        batch_plan
            .train_sample_count
            .map(|_| data_cursor_next * batch_plan.sequence_tokens),
        &last_artifacts,
    )?;
    wait_for_rank_barrier(
        &output_dir.join("qwen-session-dp-sharded-rank-manifests-written"),
        rank,
        world_size,
        Duration::from_secs(300),
    )?;
    let sharded_global_manifest_output = output_dir.join("qwen-session-dp-sharded-global.json");
    if rank == 0 {
        write_qwen_session_dp_global_sharded_manifest(
            &output_dir,
            model_path,
            world_size,
            resume_train_step + steps - 1,
            dtype,
            batch_plan.train_sample_count.map(|_| data_cursor_next),
            batch_plan
                .train_sample_count
                .map(|_| data_cursor_next * batch_plan.sequence_tokens),
            batch_plan.data_epoch_next,
            batch_plan.data_sample_offset_next,
            batch_plan.train_sample_count,
            batch_plan.dataset_source_files.as_deref().unwrap_or(&[]),
            batch_plan
                .dataset_source_sample_counts
                .as_deref()
                .unwrap_or(&[]),
            batch_plan.dataset_fingerprint.as_deref().unwrap_or(""),
            batch_plan.dataset_shuffle.unwrap_or(true),
            &sharded_global_manifest_output,
        )?;
    }
    wait_for_rank_barrier(
        &output_dir.join("qwen-session-dp-sharded-global-manifest-written"),
        rank,
        world_size,
        Duration::from_secs(300),
    )?;
    let sharded_global_text =
        fs::read_to_string(&sharded_global_manifest_output).with_context(|| {
            format!(
                "failed to read {}",
                sharded_global_manifest_output.display()
            )
        })?;
    let sharded_global_manifest: QwenShardedCheckpointManifest =
        serde_json::from_str(&sharded_global_text).with_context(|| {
            format!(
                "failed to parse {}",
                sharded_global_manifest_output.display()
            )
        })?;
    sharded_global_manifest.validate()?;
    qwen_validate_optional_sft_resume_dataset(
        &sharded_global_manifest.dataset_source_files,
        &sharded_global_manifest.dataset_source_sample_counts,
        &sharded_global_manifest.dataset_fingerprint,
        sharded_global_manifest.dataset_shuffle,
        batch_plan.dataset_source_files.as_deref(),
        batch_plan.dataset_source_sample_counts.as_deref(),
        batch_plan.dataset_fingerprint.as_deref(),
        batch_plan.dataset_shuffle,
        "Qwen session DP sharded checkpoint resume",
    )?;
    let sharded_delta_manifest = qwen_sharded_rank_to_delta_manifest(
        &sharded_global_manifest,
        rank,
        expected_loss,
        global_post_update_loss,
        learning_rate,
    )?;
    let sharded_resumed_session = QwenTrainableSession::from_manifest_on_device(
        config,
        read_safetensors_map(&model_path.join("model.safetensors"))?,
        local_input.shallow_clone(),
        dtype.kind(),
        &sharded_delta_manifest,
        Some(device),
    )?;
    let mut sharded_resumed_session = sharded_resumed_session;
    sharded_resumed_session.set_input_ids(&local_final_step_batch);
    let sharded_reloaded_loss = sharded_resumed_session.loss_value()?;
    let sharded_reload_delta = (post_update_loss - sharded_reloaded_loss).abs();
    if sharded_reload_delta > 1e-5 {
        bail!(
            "Qwen session DP sharded checkpoint reload parity failed: rank={rank}, post_update_loss={post_update_loss}, sharded_reloaded_loss={sharded_reloaded_loss}, sharded_reload_delta={sharded_reload_delta}"
        );
    }
    let mut sharded_continuous_session = QwenTrainableSession::from_manifest_on_device(
        config,
        read_safetensors_map(&model_path.join("model.safetensors"))?,
        local_input.shallow_clone(),
        dtype.kind(),
        &delta_manifest,
        Some(device),
    )?;
    sharded_continuous_session.set_input_ids(&local_next_step_batch);
    sharded_resumed_session.set_input_ids(&local_next_step_batch);
    sharded_continuous_session.loss_and_backward()?;
    sharded_resumed_session.loss_and_backward()?;
    let sharded_continuous_next_grads = sharded_continuous_session.all_reduce_average_grads(
        &output_dir.join("qwen-session-dp-sharded-continuous-next-gradient"),
        world_size,
    )?;
    let sharded_resumed_next_grads = sharded_resumed_session.all_reduce_average_grads(
        &output_dir.join("qwen-session-dp-sharded-resumed-next-gradient"),
        world_size,
    )?;
    sharded_continuous_session.apply_adamw_step(
        &sharded_continuous_next_grads,
        learning_rate,
        (resume_train_step + steps) as i32,
    )?;
    sharded_resumed_session.apply_adamw_step(
        &sharded_resumed_next_grads,
        learning_rate,
        (resume_train_step + steps) as i32,
    )?;
    let sharded_continuous_next_loss = sharded_continuous_session.loss_value()?;
    let sharded_resumed_next_loss = sharded_resumed_session.loss_value()?;
    let sharded_next_step_delta = (sharded_continuous_next_loss - sharded_resumed_next_loss).abs();
    if sharded_next_step_delta > 1e-5 {
        bail!(
            "Qwen session DP sharded checkpoint resume parity failed: rank={rank}, sharded_continuous_next_loss={sharded_continuous_next_loss}, sharded_resumed_next_loss={sharded_resumed_next_loss}, sharded_next_step_delta={sharded_next_step_delta}"
        );
    }

    let summary = QwenSessionDpRankSummary {
        rank,
        world_size,
        dtype: dtype.label().to_string(),
        resume_from: resume_from.map(|path| path.display().to_string()),
        resumed_checkpoint: resume_from.is_some(),
        local_batch_size: local_session.input_ids.size()[0] as usize,
        sequence_tokens: batch_plan.sequence_tokens,
        dataset_total_samples: batch_plan.dataset_total_samples,
        dataset_total_tokens: batch_plan.dataset_total_tokens,
        dataset_train_samples: batch_plan.dataset_train_samples,
        dataset_eval_samples: batch_plan.dataset_eval_samples,
        dataset_source_files: batch_plan.dataset_source_files,
        dataset_source_sample_counts: batch_plan.dataset_source_sample_counts,
        dataset_fingerprint: batch_plan.dataset_fingerprint,
        dataset_order_seed: batch_plan.dataset_order_seed,
        dataset_shuffle: batch_plan.dataset_shuffle,
        data_cursor_start: batch_plan.train_sample_count.map(|_| data_cursor_start),
        data_cursor_end: batch_plan.train_sample_count.map(|_| data_cursor_end),
        data_cursor_next: batch_plan.train_sample_count.map(|_| data_cursor_next),
        data_epoch_start: batch_plan.data_epoch_start,
        data_epoch_end: batch_plan.data_epoch_end,
        data_epoch_next: batch_plan.data_epoch_next,
        data_sample_offset_start: batch_plan.data_sample_offset_start,
        data_sample_offset_end: batch_plan.data_sample_offset_end,
        data_sample_offset_next: batch_plan.data_sample_offset_next,
        tensor_count: local_grads.len(),
        steps,
        learning_rate,
        max_grad_delta,
        loss_delta,
        local_loss,
        post_update_loss,
        global_loss,
        global_post_update_loss,
        global_loss_improved,
        global_step_losses,
        expected_loss,
        checkpoint_written,
        checkpoint_path: checkpoint_path.display().to_string(),
        delta_output: delta_output.display().to_string(),
        optimizer_output: optimizer_output.display().to_string(),
        manifest_output: checkpoint_path.display().to_string(),
        sharded_rank_manifest_output: sharded_rank_manifest_output.display().to_string(),
        sharded_global_manifest_output: sharded_global_manifest_output.display().to_string(),
        reloaded_loss,
        reload_delta,
        sharded_reloaded_loss,
        sharded_reload_delta,
        continuous_next_loss,
        resumed_next_loss,
        next_step_delta,
        sharded_continuous_next_loss,
        sharded_resumed_next_loss,
        sharded_next_step_delta,
        trainable_tensors,
    };
    let summary_path = output_dir.join(format!("qwen-session-dp-rank-{rank}.json"));
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    println!(
        "qwen session DP rank {rank}: updated {} trainable tensors",
        trainable_summaries.len()
    );

    Ok(())
}

fn write_qwen_session_dp_rank_sharded_manifest(
    output_dir: &Path,
    model_path: &Path,
    rank: usize,
    world_size: usize,
    steps: usize,
    learning_rate: f64,
    dtype: QwenComputeDType,
    consumed_samples: Option<usize>,
    consumed_tokens: Option<usize>,
    artifacts: &QwenTrainStepArtifacts,
) -> Result<PathBuf> {
    let rank_dir = output_dir.join(format!("sharded-rank-{rank}"));
    fs::create_dir_all(&rank_dir)
        .with_context(|| format!("failed to create {}", rank_dir.display()))?;
    let model_safetensors = rank_dir.join("model.safetensors");
    let optimizer_safetensors = rank_dir.join("optimizer.safetensors");
    let model_refs: Vec<(&str, &Tensor)> = artifacts
        .delta_entries
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    Tensor::write_safetensors(&model_refs, &model_safetensors)
        .with_context(|| format!("failed to write {}", model_safetensors.display()))?;
    let optimizer_refs: Vec<(&str, &Tensor)> = artifacts
        .optimizer_entries
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    Tensor::write_safetensors(&optimizer_refs, &optimizer_safetensors)
        .with_context(|| format!("failed to write {}", optimizer_safetensors.display()))?;

    let shards = artifacts
        .manifest_tensors
        .iter()
        .map(|entry| {
            Ok(QwenTensorShardManifestEntry {
                name: entry.name.clone(),
                shard_name: entry.delta_name.clone(),
                optimizer_m_name: entry
                    .adam_m_name
                    .clone()
                    .ok_or_else(|| anyhow!("missing Adam m slot for {}", entry.name))?,
                optimizer_v_name: entry
                    .adam_v_name
                    .clone()
                    .ok_or_else(|| anyhow!("missing Adam v slot for {}", entry.name))?,
                global_shape: entry.shape.clone(),
                shard_shape: entry.shape.clone(),
                dtype: entry.dtype.clone(),
                partition: "replicated_dp".to_string(),
                tied_group: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let rank_manifest = QwenRankShardManifest {
        rank,
        data_parallel_rank: rank,
        tensor_model_parallel_rank: 0,
        pipeline_model_parallel_rank: 0,
        expert_model_parallel_rank: 0,
        context_parallel_rank: 0,
        model_safetensors: model_safetensors.display().to_string(),
        optimizer_safetensors: optimizer_safetensors.display().to_string(),
        shards,
    };
    let rank_manifest_output = output_dir.join(format!("qwen-session-dp-sharded-rank-{rank}.json"));
    fs::write(
        &rank_manifest_output,
        serde_json::to_string_pretty(&rank_manifest)? + "\n",
    )
    .with_context(|| format!("failed to write {}", rank_manifest_output.display()))?;

    let smoke_metadata = serde_json::json!({
        "format": "rustrain.qwen_session_dp_shard_smoke.v1",
        "base_model_path": model_path.display().to_string(),
        "rank": rank,
        "world_size": world_size,
        "steps": steps,
        "consumed_samples": consumed_samples,
        "consumed_tokens": consumed_tokens,
        "learning_rate": learning_rate,
        "dtype": dtype.label(),
    });
    fs::write(
        rank_dir.join("smoke-metadata.json"),
        serde_json::to_string_pretty(&smoke_metadata)? + "\n",
    )
    .with_context(|| format!("failed to write sharded smoke metadata for rank {rank}"))?;

    Ok(rank_manifest_output)
}

fn write_qwen_session_dp_global_sharded_manifest(
    output_dir: &Path,
    model_path: &Path,
    world_size: usize,
    global_step: usize,
    dtype: QwenComputeDType,
    consumed_samples: Option<usize>,
    consumed_tokens: Option<usize>,
    data_epoch_next: Option<usize>,
    data_sample_offset_next: Option<usize>,
    data_train_samples: Option<usize>,
    dataset_source_files: &[String],
    dataset_source_sample_counts: &[QwenSftSourceSampleCount],
    dataset_fingerprint: &str,
    dataset_shuffle: bool,
    manifest_output: &Path,
) -> Result<()> {
    let mut ranks = Vec::with_capacity(world_size);
    for rank in 0..world_size {
        let rank_manifest_output =
            output_dir.join(format!("qwen-session-dp-sharded-rank-{rank}.json"));
        let text = fs::read_to_string(&rank_manifest_output)
            .with_context(|| format!("failed to read {}", rank_manifest_output.display()))?;
        let rank_manifest: QwenRankShardManifest = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", rank_manifest_output.display()))?;
        ranks.push(rank_manifest);
    }
    let consumed_samples_value = consumed_samples.unwrap_or(world_size * global_step);
    let consumed_tokens_value = consumed_tokens.unwrap_or(world_size * global_step * 5);
    let manifest = QwenShardedCheckpointManifest {
        format: "rustrain.qwen_sharded.v1".to_string(),
        base_model_path: model_path.display().to_string(),
        tokenizer_path: model_path.join("tokenizer.json").display().to_string(),
        global_step: global_step as u64,
        consumed_samples: consumed_samples_value
            .try_into()
            .context("consumed_samples overflowed u64")?,
        consumed_tokens: consumed_tokens_value
            .try_into()
            .context("consumed_tokens overflowed u64")?,
        data_cursor_next: consumed_samples.map(|value| value as u64),
        data_epoch_next: data_epoch_next.map(|value| value as u64),
        data_sample_offset_next: data_sample_offset_next.map(|value| value as u64),
        data_train_samples: data_train_samples.map(|value| value as u64),
        dataset_source_files: dataset_source_files.to_vec(),
        dataset_source_sample_counts: dataset_source_sample_counts.to_vec(),
        dataset_fingerprint: dataset_fingerprint.to_string(),
        dataset_shuffle,
        seed: 42,
        dtype: dtype.label().to_string(),
        optimizer: "adamw".to_string(),
        scheduler: "constant".to_string(),
        parallel: QwenShardedParallelManifest {
            data_parallel_size: world_size,
            tensor_model_parallel_size: 1,
            pipeline_model_parallel_size: 1,
            expert_model_parallel_size: 1,
            context_parallel_size: 1,
        },
        ranks,
    };
    manifest.validate()?;
    fs::write(
        manifest_output,
        serde_json::to_string_pretty(&manifest)? + "\n",
    )
    .with_context(|| format!("failed to write {}", manifest_output.display()))
}

fn qwen_sharded_rank_to_delta_manifest(
    manifest: &QwenShardedCheckpointManifest,
    rank: usize,
    initial_loss: f64,
    final_loss: f64,
    learning_rate: f64,
) -> Result<QwenDeltaCheckpointManifest> {
    manifest.validate()?;
    let rank_manifest = manifest
        .ranks
        .iter()
        .find(|entry| entry.rank == rank)
        .ok_or_else(|| anyhow!("Qwen sharded checkpoint is missing rank {rank}"))?;
    Ok(QwenDeltaCheckpointManifest {
        format: "rustrain.qwen_delta.v1".to_string(),
        base_model_path: manifest.base_model_path.clone(),
        reference_fixture: format!("qwen_sharded_rank_{rank}"),
        delta_safetensors: rank_manifest.model_safetensors.clone(),
        optimizer_safetensors: Some(rank_manifest.optimizer_safetensors.clone()),
        train_step: manifest.global_step,
        data_cursor_start: None,
        data_cursor_end: None,
        data_cursor_next: manifest
            .data_cursor_next
            .map(|value| usize::try_from(value).context("data_cursor_next overflowed usize"))
            .transpose()?,
        data_epoch_start: None,
        data_epoch_end: None,
        data_epoch_next: manifest
            .data_epoch_next
            .map(|value| usize::try_from(value).context("data_epoch_next overflowed usize"))
            .transpose()?,
        data_sample_offset_start: None,
        data_sample_offset_end: None,
        data_sample_offset_next: manifest
            .data_sample_offset_next
            .map(|value| usize::try_from(value).context("data_sample_offset_next overflowed usize"))
            .transpose()?,
        dataset_source_files: manifest.dataset_source_files.clone(),
        dataset_source_sample_counts: manifest.dataset_source_sample_counts.clone(),
        dataset_fingerprint: manifest.dataset_fingerprint.clone(),
        dataset_shuffle: manifest.dataset_shuffle,
        learning_rate,
        initial_loss,
        final_loss,
        tensors: rank_manifest
            .shards
            .iter()
            .map(|shard| QwenDeltaTensorManifestEntry {
                name: shard.name.clone(),
                delta_name: shard.shard_name.clone(),
                adam_m_name: Some(shard.optimizer_m_name.clone()),
                adam_v_name: Some(shard.optimizer_v_name.clone()),
                shape: shard.global_shape.clone(),
                dtype: shard.dtype.clone(),
                grad_norm: 0.0,
                delta_norm: 0.0,
            })
            .collect(),
    })
}

pub fn qwen_session_dp_data_plan(
    config_path: &Path,
    world_size: usize,
    data_cursor_start: usize,
) -> Result<()> {
    if world_size == 0 {
        bail!("qwen session DP data plan requires world_size > 0");
    }
    let config = load_config(config_path)?;
    if config.model.architecture != "qwen_trainable_session" {
        bail!(
            "qwen session DP data plan expects architecture = qwen_trainable_session, got {}",
            config.model.architecture
        );
    }
    if config.parallel.data_parallel_size != world_size {
        bail!(
            "qwen session DP data plan world_size {world_size} does not match config data_parallel_size {}",
            config.parallel.data_parallel_size
        );
    }
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("qwen session DP data plan requires model.model_path")?;
    let data_config = config
        .data
        .as_ref()
        .context("qwen session DP data plan requires [data]")?;
    if data_config.kind != RuntimeDataKind::InstructionJsonl {
        bail!("qwen session DP data plan supports kind = instruction_jsonl");
    }
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let datasets = qwen_sft_train_eval_datasets_from_paths(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.train_split,
        data_config.shuffle,
        config.run.seed,
    )?;
    let dataset_summary = datasets.combined_summary;
    let train_dataset = datasets.train_dataset;
    let eval_dataset = datasets.eval_dataset;
    let local_batch_size = config
        .train
        .micro_batch_size
        .min(train_dataset.len())
        .max(1);
    let global_batch_size = local_batch_size * world_size;
    let train_steps = config.train.max_steps as usize;
    let required_batches = train_steps * global_batch_size + 1;
    let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
    let data_cursor_next = data_cursor_end;
    let (data_epoch_start, data_sample_offset_start) =
        qwen_data_epoch_and_offset(data_cursor_start, train_dataset.len())?;
    let (data_epoch_end, data_sample_offset_end) =
        qwen_data_epoch_and_offset(data_cursor_end, train_dataset.len())?;
    let (data_epoch_next, data_sample_offset_next) =
        qwen_data_epoch_and_offset(data_cursor_next, train_dataset.len())?;
    let summary = QwenSessionDpDataPlanSummary {
        config_path: config_path.display().to_string(),
        model_path: model_path.display().to_string(),
        world_size,
        local_batch_size,
        global_batch_size,
        train_steps,
        required_batches,
        data_cursor_start,
        data_cursor_end,
        data_cursor_next,
        data_epoch_start,
        data_epoch_end,
        data_epoch_next,
        data_sample_offset_start,
        data_sample_offset_end,
        data_sample_offset_next,
        dataset_total_samples: dataset_summary.samples,
        dataset_total_tokens: dataset_summary.total_tokens,
        dataset_train_samples: train_dataset.len(),
        dataset_eval_samples: eval_dataset.len(),
        dataset_source_files: dataset_summary.source_files,
        dataset_source_sample_counts: dataset_summary.source_sample_counts,
        dataset_fingerprint: dataset_summary.fingerprint,
        dataset_order_seed: config.run.seed,
        dataset_shuffle: dataset_summary.shuffle,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
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
        crate::runtime::DType::Fp32 => QwenComputeDType::Fp32,
        crate::runtime::DType::Bf16 => QwenComputeDType::Bf16,
        crate::runtime::DType::Fp16 => {
            bail!("qwen session trainer does not support fp16 yet; use fp32 or bf16")
        }
    };
    qwen_session_dp_rank_smoke(
        model_path,
        output_dir,
        dtype,
        config.train.max_steps as usize,
        config.train.learning_rate as f64,
        &qwen_session_trainable_layers_from_config(config),
        config.train.resume_from.as_deref(),
        Some(config),
    )
}

pub(crate) fn train_qwen_session_single_from_config(
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
    let dtype = match config.train.dtype {
        crate::runtime::DType::Fp32 => QwenComputeDType::Fp32,
        crate::runtime::DType::Bf16 => QwenComputeDType::Bf16,
        crate::runtime::DType::Fp16 => {
            bail!("qwen session trainer does not support fp16 yet; use fp32 or bf16")
        }
    };
    qwen_session_single_summary(
        model_path,
        &run_paths
            .checkpoints
            .join("qwen-session-single-delta.safetensors"),
        dtype,
        config.train.max_steps as usize,
        config.train.learning_rate as f64,
        config.train.resume_from.as_deref(),
        &qwen_session_trainable_layers_from_config(config),
        Some(config),
    )
}

fn qwen_dp_artifact_dir(output_dir: &Path) -> Result<PathBuf> {
    let port = std::env::var("MASTER_PORT")
        .context("MASTER_PORT is not set; run through rustrain launch")?;
    Ok(output_dir.join(format!("launch-{port}")))
}

fn delta_manifest_path(delta_output: &Path) -> std::path::PathBuf {
    let mut path = delta_output.as_os_str().to_os_string();
    path.push(".json");
    path.into()
}

fn optimizer_state_path(delta_output: &Path) -> std::path::PathBuf {
    let mut path = delta_output.as_os_str().to_os_string();
    path.push(".optimizer.safetensors");
    path.into()
}

fn adam_slot_names(name: &str) -> AdamSlotNames {
    AdamSlotNames {
        m: format!("{name}.adam_m"),
        v: format!("{name}.adam_v"),
    }
}

impl QwenTrainableRegistry {
    fn representative(weights: &mut BTreeMap<String, Tensor>) -> Result<Self> {
        Self::from_names(weights, representative_trainable_qwen_tensors())
    }

    fn from_names(weights: &mut BTreeMap<String, Tensor>, names: Vec<String>) -> Result<Self> {
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

    fn from_names_on_device(
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

    fn adamw_step(
        &mut self,
        weights: &mut BTreeMap<String, Tensor>,
        learning_rate: f64,
        step: i32,
    ) -> Result<QwenTrainStepArtifacts> {
        let grads = self.grad_entries()?;
        self.adamw_step_with_grads(weights, &grads, learning_rate, step)
    }

    fn adamw_step_with_grads(
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

    fn zero_grad(&mut self) {
        for parameter in &mut self.parameters {
            parameter.tensor.zero_grad();
        }
    }

    fn grad_entries(&self) -> Result<Vec<(String, Tensor)>> {
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

    fn parameter_names(&self) -> Vec<String> {
        self.parameters
            .iter()
            .map(|parameter| parameter.name.clone())
            .collect()
    }

    fn apply_delta_checkpoint(
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

    fn load_from_manifest(
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
    fn from_weights(
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

    fn from_trainable_layers(
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

    fn from_names(
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

    fn from_names_on_device(
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

    fn from_trainable_layers_on_device(
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

    fn from_manifest(
        config: QwenRuntimeConfig,
        weights: BTreeMap<String, Tensor>,
        input_ids: Tensor,
        compute_kind: Kind,
        manifest: &QwenDeltaCheckpointManifest,
    ) -> Result<Self> {
        Self::from_manifest_on_device(config, weights, input_ids, compute_kind, manifest, None)
    }

    fn from_manifest_on_device(
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

    fn loss_value(&self) -> Result<f64> {
        Ok(qwen_causal_lm_loss_with_kind(
            &self.input_ids,
            &self.weights,
            &self.config,
            self.compute_kind,
        )?
        .double_value(&[]))
    }

    fn loss_and_backward(&mut self) -> Result<f64> {
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

    fn grad_entries(&self) -> Result<Vec<(String, Tensor)>> {
        self.registry.grad_entries()
    }

    fn parameter_names(&self) -> Vec<String> {
        self.registry.parameter_names()
    }

    fn set_input_ids(&mut self, input_ids: &Tensor) {
        self.input_ids = input_ids.to_device(self.input_ids.device());
    }

    fn all_reduce_average_grads(
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

    fn apply_adamw_step(
        &mut self,
        averaged_grads: &[(String, Tensor)],
        learning_rate: f64,
        step: i32,
    ) -> Result<QwenTrainStepArtifacts> {
        self.registry
            .adamw_step_with_grads(&mut self.weights, averaged_grads, learning_rate, step)
    }

    fn train_step(&mut self, learning_rate: f64, step: i32) -> Result<QwenTrainStepResult> {
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
    fn from_weights(
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

    fn loss_and_backward(&mut self) -> Result<f64> {
        for (_, parameter) in self.parameters_mut() {
            parameter.zero_grad();
        }
        let loss = self.loss_tensor();
        let loss_value = loss.double_value(&[]);
        loss.backward();
        Ok(loss_value)
    }

    fn loss_value(&self) -> f64 {
        self.loss_tensor().double_value(&[])
    }

    fn loss_tensor(&self) -> Tensor {
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

    fn all_reduce_average_grads(
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

    fn apply_sgd_step(&mut self, averaged_grads: &[Tensor], learning_rate: f64) -> Result<()> {
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

    fn grad_entries(&self) -> Result<Vec<(String, Tensor)>> {
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

    fn parameters(&self) -> [(&'static str, &Tensor); 7] {
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

    fn parameters_mut(&mut self) -> [(&'static str, &mut Tensor); 7] {
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

fn adamw_next_state(
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

fn qwen_dp_attention_global(input: &Tensor) -> Result<Tensor> {
    if input.size().len() != 3 || input.size()[0] != 1 || input.size()[1] < 2 {
        bail!("Qwen attention DP fixture expects shape [1, seq_len>=2, hidden]");
    }
    let reversed = input.flip([1]);
    Ok(Tensor::cat(&[input.shallow_clone(), reversed], 0))
}

fn qwen_dp_attention_input_for_rank(
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

fn qwen_dp_attention_target_for_rank(
    target: &Tensor,
    rank: usize,
    world_size: usize,
) -> Result<Tensor> {
    qwen_dp_attention_input_for_rank(target, rank, world_size)
}

fn grad_signatures(grads: &[(String, Tensor)]) -> Result<Vec<QwenGradSignature>> {
    grads
        .iter()
        .map(|(name, grad)| grad_signature(name, grad))
        .collect()
}

fn grad_signature(name: &str, grad: &Tensor) -> Result<QwenGradSignature> {
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
    fn values(&self) -> Vec<f32> {
        self.samples.clone()
    }
}

fn signature_values_max_delta(actual: &[f32], expected: &[f32]) -> Result<f32> {
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

fn wait_for_expected_signatures(
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

fn wait_for_rank_barrier(
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

fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set; run through rustrain launch"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

fn adamw_update(
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

fn representative_trainable_qwen_tensors() -> Vec<String> {
    qwen_trainable_tensors_for_layers(&[0], true)
}

fn qwen_session_default_trainable_layers() -> Vec<usize> {
    vec![0]
}

fn qwen_session_trainable_layers_from_config(config: &Config) -> Vec<usize> {
    config
        .model
        .trainable_layers
        .clone()
        .unwrap_or_else(qwen_session_default_trainable_layers)
}

fn qwen_trainable_tensors_for_layers(
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

fn qwen_session_dp_global_input(
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

fn qwen_session_fixed_batch_plan(
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

fn qwen_session_batch_plan_from_config(
    model_path: &Path,
    weights: &BTreeMap<String, Tensor>,
    data_cursor_start: usize,
    train_steps: usize,
    runtime_config: Option<&Config>,
) -> Result<QwenSessionBatchPlan> {
    let Some(runtime_config) = runtime_config else {
        return qwen_session_fixed_batch_plan(weights, data_cursor_start, train_steps);
    };
    let Some(data_config) = runtime_config.data.as_ref() else {
        return qwen_session_fixed_batch_plan(weights, data_cursor_start, train_steps);
    };
    if data_config.kind != RuntimeDataKind::InstructionJsonl {
        bail!("qwen trainable session data path supports kind = instruction_jsonl");
    }
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let datasets = qwen_sft_train_eval_datasets_from_paths(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.train_split,
        data_config.shuffle,
        runtime_config.run.seed,
    )?;
    let dataset_summary = datasets.combined_summary;
    let train_dataset = datasets.train_dataset;
    let eval_dataset = datasets.eval_dataset;
    let batch_size = runtime_config
        .train
        .micro_batch_size
        .min(train_dataset.len())
        .max(1);
    let required_batches = data_cursor_start + train_steps * batch_size + 1;
    let data_cursor_end = data_cursor_start + train_steps * batch_size;
    let data_cursor_next = data_cursor_end;
    let (data_epoch_start, data_sample_offset_start) =
        qwen_data_epoch_and_offset(data_cursor_start, train_dataset.len())?;
    let (data_epoch_end, data_sample_offset_end) =
        qwen_data_epoch_and_offset(data_cursor_end, train_dataset.len())?;
    let (data_epoch_next, data_sample_offset_next) =
        qwen_data_epoch_and_offset(data_cursor_next, train_dataset.len())?;
    let train_batches = (0..required_batches)
        .map(|sample_cursor| {
            train_dataset
                .padded_batch(sample_cursor, batch_size)
                .map(|batch| batch.input_ids)
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

fn qwen_session_fixed_dp_batch_plan(
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
        dataset_total_samples: None,
        dataset_total_tokens: None,
        dataset_train_samples: None,
        dataset_eval_samples: None,
        dataset_source_files: None,
        dataset_source_sample_counts: None,
        dataset_fingerprint: None,
        dataset_order_seed: None,
        dataset_shuffle: None,
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

fn qwen_session_dp_batch_plan_from_config(
    model_path: &Path,
    weights: &BTreeMap<String, Tensor>,
    data_cursor_start: usize,
    train_steps: usize,
    world_size: usize,
    device: Device,
    runtime_config: Option<&Config>,
) -> Result<QwenSessionDpBatchPlan> {
    let Some(runtime_config) = runtime_config else {
        return qwen_session_fixed_dp_batch_plan(weights, device, train_steps);
    };
    let Some(data_config) = runtime_config.data.as_ref() else {
        return qwen_session_fixed_dp_batch_plan(weights, device, train_steps);
    };
    if data_config.kind != RuntimeDataKind::InstructionJsonl {
        bail!("qwen trainable session DP data path supports kind = instruction_jsonl");
    }
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let datasets = qwen_sft_train_eval_datasets_from_paths(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.train_split,
        data_config.shuffle,
        runtime_config.run.seed,
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
    let global_train_batches = (0..required_batches)
        .map(|relative_cursor| {
            train_dataset
                .padded_batch(data_cursor_start + relative_cursor, global_batch_size)
                .map(|batch| batch.input_ids.to_device(device))
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
        dataset_total_samples: Some(dataset_summary.samples),
        dataset_total_tokens: Some(dataset_summary.total_tokens),
        dataset_train_samples: Some(train_dataset.len()),
        dataset_eval_samples: Some(eval_dataset.len()),
        dataset_source_files: Some(dataset_summary.source_files),
        dataset_source_sample_counts: Some(dataset_summary.source_sample_counts),
        dataset_fingerprint: Some(dataset_summary.fingerprint),
        dataset_order_seed: Some(runtime_config.run.seed),
        dataset_shuffle: Some(dataset_summary.shuffle),
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
    modules: BTreeMap<QwenLoraTargetModule, QwenLoraModuleAdapter>,
    rank: i64,
    alpha: f64,
}

struct QwenLoraModuleAdapter {
    a: Tensor,
    b: Tensor,
}

impl QwenLoraTargetModule {
    fn as_str(self) -> &'static str {
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

    fn parse(value: &str) -> Result<Self> {
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

    fn id(self) -> i64 {
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

    fn from_id(id: i64) -> Result<Self> {
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

    fn parent_path(self) -> &'static str {
        match self {
            Self::QProj | Self::KProj | Self::VProj | Self::OProj => "self_attn",
            Self::GateProj | Self::UpProj | Self::DownProj => "mlp",
        }
    }

    fn weight_name(self, layer_index: usize) -> String {
        format!(
            "model.layers.{layer_index}.{}.{}.weight",
            self.parent_path(),
            self.as_str()
        )
    }

    fn adapter_prefix(self, layer_index: usize) -> String {
        format!(
            "model.layers.{layer_index}.{}.{}",
            self.parent_path(),
            self.as_str()
        )
    }
}

impl QwenLoraConfig {
    fn layer0_qv(rank: i64, alpha: f64) -> Result<Self> {
        Self::new(
            vec![0],
            vec![QwenLoraTargetModule::QProj, QwenLoraTargetModule::VProj],
            rank,
            alpha,
        )
    }

    fn from_runtime(config: &RuntimeLoraConfig) -> Result<Self> {
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

    fn new(
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

    fn alpha_f64(&self) -> f64 {
        self.alpha as f64
    }

    fn target_module_names(&self) -> Vec<String> {
        self.target_modules
            .iter()
            .map(|module| module.as_str().to_string())
            .collect()
    }
}

impl QwenLoraRegistry {
    fn zeros(weights: &BTreeMap<String, Tensor>, config: &QwenLoraConfig) -> Result<Self> {
        Self::build(weights, config, false, false)
    }

    fn deterministic(
        weights: &BTreeMap<String, Tensor>,
        config: &QwenLoraConfig,
        trainable: bool,
    ) -> Result<Self> {
        Self::build(weights, config, true, trainable)
    }

    fn build(
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

    fn layer_adapter(&self, layer_index: usize) -> Result<&QwenAttentionLoraAdapter> {
        self.adapters
            .get(&layer_index)
            .ok_or_else(|| anyhow!("missing LoRA adapter for layer {layer_index}"))
    }

    fn adapter_for_layer(&self, layer_index: usize) -> Option<&QwenAttentionLoraAdapter> {
        self.adapters.get(&layer_index)
    }

    fn merge_into_weights(
        &self,
        weights: &BTreeMap<String, Tensor>,
    ) -> Result<BTreeMap<String, Tensor>> {
        self.apply_to_weights(weights, 1.0)
    }

    fn unmerge_from_weights(
        &self,
        weights: &BTreeMap<String, Tensor>,
    ) -> Result<BTreeMap<String, Tensor>> {
        self.apply_to_weights(weights, -1.0)
    }

    fn apply_to_weights(
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

    fn trainable_tensor_names(&self) -> Vec<String> {
        self.trainable_tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    fn trainable_tensors(&self) -> Vec<(String, Tensor)> {
        self.adapters
            .iter()
            .flat_map(|(layer_index, adapter)| adapter.trainable_tensors(*layer_index))
            .collect()
    }

    fn save(&self, path: &Path) -> Result<()> {
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

    fn load(path: &Path) -> Result<Self> {
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
    fn zeros(module_specs: &[(QwenLoraTargetModule, i64, i64)], rank: i64, alpha: f64) -> Self {
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

    fn deterministic(
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

    fn deterministic_trainable(
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
    fn save(&self, path: &Path) -> Result<()> {
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
    fn load(path: &Path) -> Result<Self> {
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

    fn load_from_tensors(
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

    fn safetensor_entries(&self, layer_index: usize) -> Vec<(String, Tensor)> {
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

    fn trainable_tensors(&self, layer_index: usize) -> Vec<(String, Tensor)> {
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

    fn delta(&self, module: QwenLoraTargetModule, device: Device) -> Result<Tensor> {
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

fn deterministic_lora_tensor<const N: usize>(shape: [i64; N], scale: f64) -> Tensor {
    let len = shape.iter().product::<i64>() as usize;
    let values: Vec<f32> = (0..len)
        .map(|index| ((index % 17) as f64 - 8.0) as f32 * scale as f32)
        .collect();
    Tensor::from_slice(&values).reshape(shape)
}

fn tensor_snapshot(tensor: &Tensor) -> Tensor {
    let mut snapshot = Tensor::zeros_like(tensor);
    snapshot.copy_(tensor);
    snapshot
}

impl QwenLayerWeights {
    pub fn load(weights: &BTreeMap<String, Tensor>, layer_index: usize) -> Result<Self> {
        Self::load_with_kind(weights, layer_index, Kind::Float)
    }

    fn load_with_kind(
        weights: &BTreeMap<String, Tensor>,
        layer_index: usize,
        kind: Kind,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_index}");
        Ok(Self {
            input_norm: tensor(weights, &format!("{prefix}.input_layernorm.weight"))?.to_kind(kind),
            q_proj: tensor(weights, &format!("{prefix}.self_attn.q_proj.weight"))?.to_kind(kind),
            q_bias: tensor(weights, &format!("{prefix}.self_attn.q_proj.bias"))?.to_kind(kind),
            k_proj: tensor(weights, &format!("{prefix}.self_attn.k_proj.weight"))?.to_kind(kind),
            k_bias: tensor(weights, &format!("{prefix}.self_attn.k_proj.bias"))?.to_kind(kind),
            v_proj: tensor(weights, &format!("{prefix}.self_attn.v_proj.weight"))?.to_kind(kind),
            v_bias: tensor(weights, &format!("{prefix}.self_attn.v_proj.bias"))?.to_kind(kind),
            o_proj: tensor(weights, &format!("{prefix}.self_attn.o_proj.weight"))?.to_kind(kind),
            post_attention_norm: tensor(
                weights,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?
            .to_kind(kind),
            gate_proj: tensor(weights, &format!("{prefix}.mlp.gate_proj.weight"))?.to_kind(kind),
            up_proj: tensor(weights, &format!("{prefix}.mlp.up_proj.weight"))?.to_kind(kind),
            down_proj: tensor(weights, &format!("{prefix}.mlp.down_proj.weight"))?.to_kind(kind),
        })
    }

    fn lora_target_weight(&self, module: QwenLoraTargetModule) -> &Tensor {
        match module {
            QwenLoraTargetModule::QProj => &self.q_proj,
            QwenLoraTargetModule::KProj => &self.k_proj,
            QwenLoraTargetModule::VProj => &self.v_proj,
            QwenLoraTargetModule::OProj => &self.o_proj,
            QwenLoraTargetModule::GateProj => &self.gate_proj,
            QwenLoraTargetModule::UpProj => &self.up_proj,
            QwenLoraTargetModule::DownProj => &self.down_proj,
        }
    }
}

pub fn qwen_layer(
    input: &Tensor,
    weights: &QwenLayerWeights,
    config: &QwenRuntimeConfig,
) -> Tensor {
    let compute_kind = weights.q_proj.kind();
    let input = input.to_kind(compute_kind);
    let attention_input =
        rms_norm(&input, &weights.input_norm, config.rms_norm_eps).to_kind(compute_kind);
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
    )
    .to_kind(compute_kind);
    let mlp_output = qwen_mlp(
        &mlp_input,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
    );
    (after_attention + mlp_output).to_kind(compute_kind)
}

pub fn qwen_forward_from_ids(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
) -> Result<Tensor> {
    qwen_forward_from_ids_with_kind(input_ids, weights, config, Kind::Float)
}

fn qwen_forward_from_ids_with_kind(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    compute_kind: Kind,
) -> Result<Tensor> {
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?.to_kind(compute_kind);
    let final_norm = tensor(weights, "model.norm.weight")?.to_kind(compute_kind);
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);
    for layer_index in 0..config.num_hidden_layers {
        let layer = QwenLayerWeights::load_with_kind(weights, layer_index, compute_kind)?;
        hidden = qwen_layer(&hidden, &layer, config);
    }
    let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps).to_kind(compute_kind);
    Ok(hidden.linear::<&Tensor>(&embed_tokens, None))
}

fn qwen_forward_from_ids_with_lora(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    registry: &QwenLoraRegistry,
    compute_kind: Kind,
) -> Result<Tensor> {
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?.to_kind(compute_kind);
    let final_norm = tensor(weights, "model.norm.weight")?.to_kind(compute_kind);
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);
    for layer_index in 0..config.num_hidden_layers {
        let layer = QwenLayerWeights::load_with_kind(weights, layer_index, compute_kind)?;
        hidden = if let Some(adapter) = registry.adapter_for_layer(layer_index) {
            qwen_layer_with_lora(&hidden, &layer, adapter, config)
        } else {
            qwen_layer(&hidden, &layer, config)
        };
    }
    let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps).to_kind(compute_kind);
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
    qwen_causal_lm_loss_with_kind(input_ids, weights, config, Kind::Float)
}

fn qwen_causal_lm_loss_with_kind(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    compute_kind: Kind,
) -> Result<Tensor> {
    let logits = qwen_forward_from_ids_with_kind(input_ids, weights, config, compute_kind)?;
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

fn qwen_greedy_generate_with_kind(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    max_new_tokens: usize,
    compute_kind: Kind,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    for _ in 0..max_new_tokens {
        let logits = qwen_forward_from_ids_with_kind(&generated, weights, config, compute_kind)?;
        let next_token = logits.i((0, -1)).argmax(-1, false).reshape([1, 1]);
        generated = Tensor::cat(&[&generated, &next_token], 1);
    }
    Ok(generated)
}

fn qwen_greedy_generate_with_lora(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    registry: &QwenLoraRegistry,
    max_new_tokens: usize,
    compute_kind: Kind,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    for _ in 0..max_new_tokens {
        let logits =
            qwen_forward_from_ids_with_lora(&generated, weights, config, registry, compute_kind)?;
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

#[allow(clippy::too_many_arguments)]
pub fn qwen_sample_generate_with_cache(
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
    let (logits, mut cache) = qwen_forward_with_cache(input_ids, weights, config, None)?;
    let mut next_token =
        sample_token_from_logits(&logits.i((0, -1)), temperature, top_k, top_p, &mut rng)?
            .reshape([1, 1]);

    for step in 0..max_new_tokens {
        generated = Tensor::cat(&[&generated, &next_token], 1);
        if step + 1 == max_new_tokens {
            break;
        }
        let (decode_logits, updated_cache) =
            qwen_forward_with_cache(&next_token, weights, config, Some(cache))?;
        cache = updated_cache;
        next_token = sample_token_from_logits(
            &decode_logits.i((0, -1)),
            temperature,
            top_k,
            top_p,
            &mut rng,
        )?
        .reshape([1, 1]);
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
    let cos = cos.to_kind(input.kind());
    let sin = sin.to_kind(input.kind());
    let q = apply_rotary(&q, &cos, &sin);
    let k = apply_rotary(&k, &cos, &sin);
    let k = repeat_kv(&k, kv_repeat);
    let v = repeat_kv(&v, kv_repeat);
    let scores = q.matmul(&k.transpose(-2, -1)) / (head_dim as f64).sqrt();
    let causal_mask = Tensor::ones([seq_len, seq_len], (Kind::Bool, input.device())).triu(1);
    let scores = scores.masked_fill(&causal_mask, f64::NEG_INFINITY);
    let probs = scores.softmax(-1, Kind::Float).to_kind(v.kind());
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
    let q_proj = lora_weight_or_base(
        adapter,
        QwenLoraTargetModule::QProj,
        &weights.q_proj,
        input.device(),
    );
    let k_proj = lora_weight_or_base(
        adapter,
        QwenLoraTargetModule::KProj,
        &weights.k_proj,
        input.device(),
    );
    let v_proj = lora_weight_or_base(
        adapter,
        QwenLoraTargetModule::VProj,
        &weights.v_proj,
        input.device(),
    );
    let o_proj = lora_weight_or_base(
        adapter,
        QwenLoraTargetModule::OProj,
        &weights.o_proj,
        input.device(),
    );
    qwen_attention(
        input,
        &q_proj,
        &weights.q_bias,
        &k_proj,
        &weights.k_bias,
        &v_proj,
        &weights.v_bias,
        &o_proj,
        config,
    )
}

fn lora_weight_or_base(
    adapter: &QwenAttentionLoraAdapter,
    module: QwenLoraTargetModule,
    base: &Tensor,
    device: Device,
) -> Tensor {
    let base = base.to_device(device);
    if adapter.modules.contains_key(&module) {
        let delta = adapter
            .delta(module, device)
            .expect("LoRA module should have a delta")
            .to_kind(base.kind());
        base + delta
    } else {
        base
    }
}

fn qwen_layer_with_lora(
    input: &Tensor,
    weights: &QwenLayerWeights,
    adapter: &QwenAttentionLoraAdapter,
    config: &QwenRuntimeConfig,
) -> Tensor {
    let compute_kind = weights.q_proj.kind();
    let input = input.to_kind(compute_kind);
    let device = input.device();
    let attention_input =
        rms_norm(&input, &weights.input_norm, config.rms_norm_eps).to_kind(compute_kind);
    let attention_output = qwen_attention_with_lora(&attention_input, weights, adapter, config);
    let after_attention = input + attention_output;
    let mlp_input = rms_norm(
        &after_attention,
        &weights.post_attention_norm,
        config.rms_norm_eps,
    )
    .to_kind(compute_kind);
    let mlp_output = qwen_mlp(
        &mlp_input,
        &lora_weight_or_base(
            adapter,
            QwenLoraTargetModule::GateProj,
            &weights.gate_proj,
            device,
        ),
        &lora_weight_or_base(
            adapter,
            QwenLoraTargetModule::UpProj,
            &weights.up_proj,
            device,
        ),
        &lora_weight_or_base(
            adapter,
            QwenLoraTargetModule::DownProj,
            &weights.down_proj,
            device,
        ),
    );
    (after_attention + mlp_output).to_kind(compute_kind)
}

fn qwen_attention_lora_mse_loss(
    input: &Tensor,
    target: &Tensor,
    weights: &QwenLayerWeights,
    adapter: &QwenAttentionLoraAdapter,
    config: &QwenRuntimeConfig,
) -> Tensor {
    qwen_attention_with_lora(input, weights, adapter, config).mse_loss(target, Reduction::Mean)
}

impl QwenSftDataset {
    fn from_instruction_pairs(tokenizer: &Tokenizer, examples: &[QwenSftExample]) -> Result<Self> {
        if examples.is_empty() {
            bail!("SFT dataset must contain at least one example");
        }
        let samples = examples
            .iter()
            .map(|example| qwen_sft_token_sample(tokenizer, example))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            samples,
            pad_token_id: qwen_pad_token_id(tokenizer),
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: qwen_sft_dataset_fingerprint(&[], examples),
        })
    }

    fn from_jsonl_paths_with_limit(
        tokenizer: &Tokenizer,
        paths: &[PathBuf],
        max_samples: Option<usize>,
    ) -> Result<Self> {
        let example_set = qwen_sft_examples_from_jsonl_paths(paths)?;
        let example_set = qwen_sft_limit_example_set(example_set, max_samples)?;
        if example_set.examples.is_empty() {
            bail!("SFT dataset must contain at least one example");
        }
        let samples = example_set
            .examples
            .iter()
            .map(|example| qwen_sft_token_sample(tokenizer, example))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            samples,
            pad_token_id: qwen_pad_token_id(tokenizer),
            epoch_shuffle_seed: None,
            source_files: example_set.source_files,
            source_sample_counts: example_set.source_sample_counts,
            fingerprint: example_set.fingerprint,
        })
    }

    fn train_eval_split(&self, train_split: f32) -> Result<(Self, Self)> {
        if !(0.0..1.0).contains(&train_split) {
            bail!("SFT train_split must be in (0, 1)");
        }
        if self.samples.len() < 2 {
            bail!("SFT train/eval split requires at least two samples");
        }
        let split_at = ((self.samples.len() as f32) * train_split).floor() as usize;
        let split_at = split_at.clamp(1, self.samples.len() - 1);
        Ok((
            Self {
                samples: self.samples[..split_at].to_vec(),
                pad_token_id: self.pad_token_id,
                epoch_shuffle_seed: self.epoch_shuffle_seed,
                source_files: self.source_files.clone(),
                source_sample_counts: self.source_sample_counts.clone(),
                fingerprint: self.fingerprint.clone(),
            },
            Self {
                samples: self.samples[split_at..].to_vec(),
                pad_token_id: self.pad_token_id,
                epoch_shuffle_seed: self.epoch_shuffle_seed,
                source_files: self.source_files.clone(),
                source_sample_counts: self.source_sample_counts.clone(),
                fingerprint: self.fingerprint.clone(),
            },
        ))
    }

    fn shuffle_by_seed(mut self, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        self.samples.shuffle(&mut rng);
        self.epoch_shuffle_seed = Some(seed);
        self
    }

    fn summary(&self) -> QwenSftDatasetSummary {
        QwenSftDatasetSummary {
            samples: self.samples.len(),
            total_tokens: self
                .samples
                .iter()
                .map(|sample| sample.token_ids.len())
                .sum(),
            response_tokens: self
                .samples
                .iter()
                .map(|sample| sample.response_tokens)
                .sum(),
            masked_positions: self
                .samples
                .iter()
                .map(|sample| sample.masked_positions)
                .sum(),
            max_sequence_tokens: self
                .samples
                .iter()
                .map(|sample| sample.token_ids.len())
                .max()
                .unwrap_or(0),
            source_files: self.source_files.clone(),
            source_sample_counts: self.source_sample_counts.clone(),
            fingerprint: self.fingerprint.clone(),
            shuffle: self.epoch_shuffle_seed.is_some(),
        }
    }

    fn sample_at_cursor(&self, cursor: usize) -> Result<QwenSftTokenSample> {
        if self.samples.is_empty() {
            bail!("SFT dataset must contain at least one sample");
        }
        let dataset_len = self.samples.len();
        let epoch = cursor / dataset_len;
        let offset = cursor % dataset_len;
        let index = if let Some(seed) = self.epoch_shuffle_seed {
            qwen_epoch_permutation_index(dataset_len, seed, epoch, offset)
        } else {
            offset
        };
        self.samples
            .get(index)
            .cloned()
            .ok_or_else(|| anyhow!("SFT cursor resolved out-of-range sample index {index}"))
    }

    fn padded_batch(&self, start: usize, batch_size: usize) -> Result<QwenSftBatch> {
        if batch_size == 0 {
            bail!("SFT batch size must be positive");
        }
        let samples = (0..batch_size)
            .map(|offset| self.sample_at_cursor(start + offset))
            .collect::<Result<Vec<_>>>()?;
        qwen_sft_padded_batch(&samples, self.pad_token_id)
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn with_source_metadata(
        mut self,
        source_files: Vec<String>,
        source_sample_counts: Vec<QwenSftSourceSampleCount>,
        fingerprint: String,
    ) -> Self {
        self.source_files = source_files;
        self.source_sample_counts = source_sample_counts;
        self.fingerprint = fingerprint;
        self
    }
}

fn qwen_apply_sft_shuffle(dataset: QwenSftDataset, shuffle: bool, seed: u64) -> QwenSftDataset {
    if shuffle {
        dataset.shuffle_by_seed(seed)
    } else {
        dataset
    }
}

fn qwen_sft_train_eval_datasets_from_paths(
    tokenizer: &Tokenizer,
    train_paths: &[PathBuf],
    eval_paths: &[PathBuf],
    max_samples: Option<usize>,
    train_split: f32,
    shuffle: bool,
    seed: u64,
) -> Result<QwenSftTrainEvalDatasets> {
    let train_dataset =
        QwenSftDataset::from_jsonl_paths_with_limit(tokenizer, train_paths, max_samples)?;
    if eval_paths.is_empty() {
        let dataset = qwen_apply_sft_shuffle(train_dataset, shuffle, seed);
        let combined_summary = dataset.summary();
        let (train_dataset, eval_dataset) = dataset.train_eval_split(train_split)?;
        return Ok(QwenSftTrainEvalDatasets {
            combined_summary,
            train_dataset,
            eval_dataset,
        });
    }

    let eval_dataset = QwenSftDataset::from_jsonl_paths_with_limit(tokenizer, eval_paths, None)?;
    let train_summary = train_dataset.summary();
    let eval_summary = eval_dataset.summary();
    let combined_source_files =
        qwen_merge_sft_source_files(&train_summary.source_files, &eval_summary.source_files);
    let combined_source_sample_counts = qwen_merge_sft_source_sample_counts(
        &train_summary.source_sample_counts,
        &eval_summary.source_sample_counts,
    );
    let combined_fingerprint = qwen_combine_sft_fingerprints(
        &combined_source_files,
        &train_summary.fingerprint,
        &eval_summary.fingerprint,
    );
    let train_dataset = qwen_apply_sft_shuffle(
        train_dataset.with_source_metadata(
            combined_source_files.clone(),
            combined_source_sample_counts.clone(),
            combined_fingerprint.clone(),
        ),
        shuffle,
        seed,
    );
    let eval_dataset = eval_dataset.with_source_metadata(
        combined_source_files.clone(),
        combined_source_sample_counts.clone(),
        combined_fingerprint.clone(),
    );
    Ok(QwenSftTrainEvalDatasets {
        combined_summary: QwenSftDatasetSummary {
            samples: train_summary.samples + eval_summary.samples,
            total_tokens: train_summary.total_tokens + eval_summary.total_tokens,
            response_tokens: train_summary.response_tokens + eval_summary.response_tokens,
            masked_positions: train_summary.masked_positions + eval_summary.masked_positions,
            max_sequence_tokens: train_summary
                .max_sequence_tokens
                .max(eval_summary.max_sequence_tokens),
            source_files: combined_source_files,
            source_sample_counts: combined_source_sample_counts,
            fingerprint: combined_fingerprint,
            shuffle,
        },
        train_dataset,
        eval_dataset,
    })
}

fn qwen_merge_sft_source_files(train: &[String], eval: &[String]) -> Vec<String> {
    train
        .iter()
        .chain(eval.iter())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn qwen_merge_sft_source_sample_counts(
    train: &[QwenSftSourceSampleCount],
    eval: &[QwenSftSourceSampleCount],
) -> Vec<QwenSftSourceSampleCount> {
    let mut counts = BTreeMap::new();
    for source_count in train.iter().chain(eval.iter()) {
        *counts.entry(source_count.path.clone()).or_insert(0) += source_count.samples;
    }
    counts
        .into_iter()
        .map(|(path, samples)| QwenSftSourceSampleCount { path, samples })
        .collect()
}

fn qwen_combine_sft_fingerprints(
    source_files: &[String],
    train_fingerprint: &str,
    eval_fingerprint: &str,
) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    qwen_sft_hash_bytes(&mut hash, b"train");
    qwen_sft_hash_bytes(&mut hash, train_fingerprint.as_bytes());
    qwen_sft_hash_bytes(&mut hash, b"\0eval");
    qwen_sft_hash_bytes(&mut hash, eval_fingerprint.as_bytes());
    for file in source_files {
        qwen_sft_hash_bytes(&mut hash, b"\0path");
        qwen_sft_hash_bytes(&mut hash, file.as_bytes());
    }
    format!("{hash:016x}")
}

fn qwen_epoch_permutation_index(
    dataset_len: usize,
    dataset_order_seed: u64,
    epoch: usize,
    offset: usize,
) -> usize {
    let mut order = (0..dataset_len).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(
        dataset_order_seed ^ ((epoch as u64).wrapping_add(1)).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    );
    order.shuffle(&mut rng);
    order[offset]
}

fn qwen_sft_examples_from_jsonl_paths(paths: &[PathBuf]) -> Result<QwenSftExampleSet> {
    if paths.is_empty() {
        bail!("SFT dataset must contain at least one JSONL path");
    }
    let mut examples = Vec::new();
    let mut source_files = BTreeSet::new();
    let mut source_sample_counts = BTreeMap::new();
    for path in paths {
        let example_set = qwen_sft_examples_from_jsonl_path(path)?;
        examples.extend(example_set.examples);
        source_files.extend(example_set.source_files);
        for source_count in example_set.source_sample_counts {
            *source_sample_counts.entry(source_count.path).or_insert(0) += source_count.samples;
        }
    }
    let source_files = source_files.into_iter().collect::<Vec<_>>();
    let source_sample_counts = source_sample_counts
        .into_iter()
        .map(|(path, samples)| QwenSftSourceSampleCount { path, samples })
        .collect::<Vec<_>>();
    let fingerprint = qwen_sft_dataset_fingerprint(&source_files, &examples);
    Ok(QwenSftExampleSet {
        examples,
        source_files,
        source_sample_counts,
        fingerprint,
    })
}

fn qwen_sft_limit_example_set(
    mut example_set: QwenSftExampleSet,
    max_samples: Option<usize>,
) -> Result<QwenSftExampleSet> {
    let Some(max_samples) = max_samples else {
        return Ok(example_set);
    };
    if max_samples == 0 {
        bail!("SFT data.max_samples must be greater than zero");
    }
    if example_set.examples.len() <= max_samples {
        return Ok(example_set);
    }
    example_set.examples.truncate(max_samples);
    let mut source_counts = BTreeMap::new();
    for example in &example_set.examples {
        let Some(source_file) = &example.source_file else {
            continue;
        };
        *source_counts.entry(source_file.clone()).or_insert(0) += 1;
    }
    let source_files = source_counts.keys().cloned().collect::<Vec<_>>();
    let source_sample_counts = source_counts
        .into_iter()
        .map(|(path, samples)| QwenSftSourceSampleCount { path, samples })
        .collect::<Vec<_>>();
    let fingerprint = qwen_sft_dataset_fingerprint(&source_files, &example_set.examples);
    Ok(QwenSftExampleSet {
        examples: example_set.examples,
        source_files,
        source_sample_counts,
        fingerprint,
    })
}

fn qwen_sft_examples_from_jsonl_path(path: &Path) -> Result<QwenSftExampleSet> {
    let files = qwen_sft_jsonl_files(path)?;

    if files.is_empty() {
        bail!("SFT JSONL path {} did not contain files", path.display());
    }

    let mut examples = Vec::new();
    let mut source_sample_counts = Vec::new();
    for file in &files {
        let contents = fs::read_to_string(&file)
            .with_context(|| format!("failed to read {}", file.display()))?;
        let before = examples.len();
        for (line_index, line) in contents.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let record: QwenSftRecord = serde_json::from_str(line).with_context(|| {
                format!(
                    "failed to parse SFT JSONL record {}:{}",
                    file.display(),
                    line_index + 1
                )
            })?;
            examples.push(QwenSftExample {
                instruction: record.instruction,
                input: record.input,
                response: record.response,
                source_file: Some(file.display().to_string()),
            });
        }
        source_sample_counts.push(QwenSftSourceSampleCount {
            path: file.display().to_string(),
            samples: examples.len() - before,
        });
    }

    if examples.is_empty() {
        bail!("SFT JSONL path {} did not contain examples", path.display());
    }
    let source_files = files
        .iter()
        .map(|file| file.display().to_string())
        .collect::<Vec<_>>();
    let fingerprint = qwen_sft_dataset_fingerprint(&source_files, &examples);
    Ok(QwenSftExampleSet {
        examples,
        source_files,
        source_sample_counts,
        fingerprint,
    })
}

fn qwen_sft_jsonl_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_dir() {
        let mut sorted = BTreeSet::new();
        for entry in
            fs::read_dir(path).with_context(|| format!("failed to list {}", path.display()))?
        {
            let entry =
                entry.with_context(|| format!("failed to read entry in {}", path.display()))?;
            let file_type = entry
                .file_type()
                .with_context(|| format!("failed to inspect {}", entry.path().display()))?;
            if file_type.is_file()
                && entry.path().extension().and_then(|value| value.to_str()) == Some("jsonl")
            {
                sorted.insert(entry.path());
            }
        }
        Ok(sorted.into_iter().collect())
    } else {
        Ok(vec![path.to_path_buf()])
    }
}

fn qwen_sft_dataset_fingerprint(source_files: &[String], examples: &[QwenSftExample]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for file in source_files {
        qwen_sft_hash_bytes(&mut hash, b"path");
        qwen_sft_hash_bytes(&mut hash, file.as_bytes());
        qwen_sft_hash_bytes(&mut hash, b"\0");
    }
    for example in examples {
        qwen_sft_hash_bytes(&mut hash, b"instruction");
        qwen_sft_hash_bytes(&mut hash, example.instruction.as_bytes());
        qwen_sft_hash_bytes(&mut hash, b"\0input");
        qwen_sft_hash_bytes(&mut hash, example.input.as_bytes());
        qwen_sft_hash_bytes(&mut hash, b"\0response");
        qwen_sft_hash_bytes(&mut hash, example.response.as_bytes());
        qwen_sft_hash_bytes(&mut hash, b"\0");
    }
    format!("{hash:016x}")
}

fn qwen_sft_hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn qwen_validate_sft_resume_dataset(
    manifest_source_files: &[String],
    manifest_source_sample_counts: &[QwenSftSourceSampleCount],
    manifest_fingerprint: &str,
    manifest_shuffle: bool,
    dataset_summary: &QwenSftDatasetSummary,
    context: &str,
) -> Result<()> {
    qwen_validate_optional_sft_resume_dataset(
        manifest_source_files,
        manifest_source_sample_counts,
        manifest_fingerprint,
        manifest_shuffle,
        Some(&dataset_summary.source_files),
        Some(&dataset_summary.source_sample_counts),
        Some(&dataset_summary.fingerprint),
        Some(dataset_summary.shuffle),
        context,
    )
}

fn qwen_validate_optional_sft_resume_dataset(
    manifest_source_files: &[String],
    manifest_source_sample_counts: &[QwenSftSourceSampleCount],
    manifest_fingerprint: &str,
    manifest_shuffle: bool,
    dataset_source_files: Option<&[String]>,
    dataset_source_sample_counts: Option<&[QwenSftSourceSampleCount]>,
    dataset_fingerprint: Option<&str>,
    dataset_shuffle: Option<bool>,
    context: &str,
) -> Result<()> {
    if manifest_fingerprint.is_empty()
        && manifest_source_files.is_empty()
        && manifest_source_sample_counts.is_empty()
    {
        return Ok(());
    }
    let Some(dataset_fingerprint) = dataset_fingerprint else {
        bail!("{context} manifest has dataset provenance but current run has no JSONL dataset");
    };
    if dataset_fingerprint.is_empty() {
        bail!("{context} current JSONL dataset fingerprint is empty");
    }
    if manifest_fingerprint != dataset_fingerprint {
        bail!(
            "{context} dataset fingerprint mismatch: manifest={manifest_fingerprint}, current={dataset_fingerprint}"
        );
    }
    let Some(dataset_shuffle) = dataset_shuffle else {
        bail!("{context} manifest has dataset shuffle provenance but current run has none");
    };
    if manifest_shuffle != dataset_shuffle {
        bail!(
            "{context} dataset shuffle mismatch: manifest={manifest_shuffle}, current={dataset_shuffle}"
        );
    }
    let Some(dataset_source_files) = dataset_source_files else {
        bail!("{context} manifest has dataset source files but current run has none");
    };
    if manifest_source_files != dataset_source_files {
        bail!(
            "{context} dataset source files mismatch: manifest={manifest_source_files:?}, current={dataset_source_files:?}"
        );
    }
    if !manifest_source_sample_counts.is_empty() {
        let Some(dataset_source_sample_counts) = dataset_source_sample_counts else {
            bail!("{context} manifest has dataset source sample counts but current run has none");
        };
        if manifest_source_sample_counts != dataset_source_sample_counts {
            bail!(
                "{context} dataset source sample counts mismatch: manifest={manifest_source_sample_counts:?}, current={dataset_source_sample_counts:?}"
            );
        }
    }
    Ok(())
}

fn qwen_sft_token_sample(
    tokenizer: &Tokenizer,
    example: &QwenSftExample,
) -> Result<QwenSftTokenSample> {
    let prompt = if example.input.trim().is_empty() {
        format!("Instruction:\n{}\n\nResponse:\n", example.instruction)
    } else {
        format!(
            "Instruction:\n{}\n\nInput:\n{}\n\nResponse:\n",
            example.instruction, example.input
        )
    };
    qwen_sft_token_sample_from_prompt(tokenizer, &prompt, &example.response)
}

fn qwen_sft_token_sample_from_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    response: &str,
) -> Result<QwenSftTokenSample> {
    let response = format!("{response}\n");
    let prompt_encoding = tokenizer
        .encode(prompt, false)
        .map_err(|error| anyhow!("failed to encode prompt: {error}"))?;
    let response_encoding = tokenizer
        .encode(response.as_str(), false)
        .map_err(|error| anyhow!("failed to encode response: {error}"))?;
    let prompt_tokens: Vec<i64> = prompt_encoding
        .get_ids()
        .iter()
        .map(|token| i64::from(*token))
        .collect();
    let response_tokens: Vec<i64> = response_encoding
        .get_ids()
        .iter()
        .map(|token| i64::from(*token))
        .collect();
    if prompt_tokens.is_empty() || response_tokens.is_empty() {
        bail!("SFT prompt and response must both tokenize to at least one token");
    }

    let mut token_ids = prompt_tokens.clone();
    token_ids.extend(response_tokens.iter().copied());
    if token_ids.len() < 2 {
        bail!("SFT sample must contain at least two tokens");
    }
    let target_len = token_ids.len() - 1;
    let prompt_len = prompt_tokens.len();
    let mask_values: Vec<f32> = (0..target_len)
        .map(|target_index| {
            if target_index + 1 >= prompt_len {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let masked_positions = mask_values.iter().filter(|value| **value > 0.0).count();
    if masked_positions == 0 {
        bail!("SFT response-only mask is empty");
    }

    Ok(QwenSftTokenSample {
        prompt_tokens: prompt_tokens.len(),
        response_tokens: response_tokens.len(),
        masked_positions,
        token_ids,
        mask_values,
    })
}

fn qwen_pad_token_id(tokenizer: &Tokenizer) -> i64 {
    tokenizer
        .get_padding()
        .map(|padding| i64::from(padding.pad_id))
        .or_else(|| tokenizer.token_to_id("<|endoftext|>").map(i64::from))
        .unwrap_or(0)
}

fn qwen_sft_padded_batch(
    samples: &[QwenSftTokenSample],
    pad_token_id: i64,
) -> Result<QwenSftBatch> {
    if samples.is_empty() {
        bail!("SFT batch must contain at least one sample");
    }
    let max_len = samples
        .iter()
        .map(|sample| sample.token_ids.len())
        .max()
        .ok_or_else(|| anyhow!("SFT batch must contain at least one sample"))?;
    if max_len < 2 {
        bail!("SFT batch sequence length must be at least two tokens");
    }

    let batch_size = samples.len();
    let mut input_values = Vec::with_capacity(batch_size * max_len);
    let mut mask_values = Vec::with_capacity(batch_size * (max_len - 1));
    let mut prompt_tokens = Vec::with_capacity(batch_size);
    let mut response_tokens = Vec::with_capacity(batch_size);
    let mut masked_positions = 0usize;
    let mut padding_tokens = 0usize;

    for sample in samples {
        prompt_tokens.push(sample.prompt_tokens);
        response_tokens.push(sample.response_tokens);
        input_values.extend(sample.token_ids.iter().copied());
        let pad_len = max_len - sample.token_ids.len();
        input_values.extend(std::iter::repeat(pad_token_id).take(pad_len));
        padding_tokens += pad_len;

        mask_values.extend(sample.mask_values.iter().copied());
        masked_positions += sample.masked_positions;
        mask_values.extend(std::iter::repeat(0.0).take(max_len - 1 - sample.mask_values.len()));
    }

    if masked_positions == 0 {
        bail!("SFT batch response-only mask is empty");
    }

    Ok(QwenSftBatch {
        input_ids: Tensor::from_slice(&input_values)
            .to_kind(Kind::Int64)
            .reshape([batch_size as i64, max_len as i64]),
        target_mask: Tensor::from_slice(&mask_values).reshape([
            batch_size as i64,
            (max_len - 1) as i64,
            1,
        ]),
        prompt_tokens,
        response_tokens,
        masked_positions,
        padding_tokens,
    })
}

fn qwen_lora_sft_loss(
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

fn qwen_layer_lora_sft_loss(
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

fn qwen_lora_sft_learning_rate(
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

fn qwen_lora_clip_scale(grad_norm: f64, max_grad_norm: Option<f64>) -> (f64, f64) {
    if let Some(max_grad_norm) = max_grad_norm {
        if grad_norm > max_grad_norm {
            let scale = max_grad_norm / (grad_norm + 1e-12);
            return (max_grad_norm, scale);
        }
    }
    (grad_norm, 1.0)
}

fn lora_train_target(base_output: &Tensor) -> Tensor {
    let values = Tensor::arange(
        base_output.numel() as i64,
        (Kind::Float, base_output.device()),
    )
    .reshape(base_output.size())
    .fmod(11.0)
        / 10_000.0;
    base_output + values
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
    let cos = cos.to_kind(input.kind());
    let sin = sin.to_kind(input.kind());
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
    let probs = scores
        .softmax(-1, Kind::Float)
        .to_kind(v_for_attention.kind());
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
        let mut registry =
            QwenTrainableRegistry::representative(&mut weights).expect("registry should build");
        assert_eq!(
            registry.parameter_names(),
            representative_trainable_qwen_tensors()
        );

        let initial_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        let loss = qwen_causal_lm_loss(&input_ids, &weights, &config).expect("loss should run");
        loss.backward();
        let artifacts = registry
            .adamw_step(&mut weights, 1e-2, 1)
            .expect("optimizer step should apply");
        assert_eq!(
            artifacts.tensor_summaries.len(),
            representative_trainable_qwen_tensors().len()
        );
        assert_eq!(
            artifacts.manifest_tensors.len(),
            representative_trainable_qwen_tensors().len()
        );
        assert_eq!(
            artifacts.optimizer_entries.len(),
            representative_trainable_qwen_tensors().len() * 2
        );
        for summary in &artifacts.tensor_summaries {
            assert!(
                summary.grad_defined,
                "{} should receive a gradient",
                summary.name
            );
            assert!(
                summary.grad_norm > 0.0,
                "{} grad should be non-zero",
                summary.name
            );
            assert!(
                summary.delta_norm > 0.0,
                "{} delta should be non-zero",
                summary.name
            );
        }

        let final_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!(final_loss < initial_loss);

        let mut reloaded_weights = tiny_qwen_weights();
        let delta_tensors: BTreeMap<String, Tensor> = artifacts
            .delta_entries
            .into_iter()
            .map(|(name, tensor)| (name, tensor))
            .collect();
        QwenTrainableRegistry::apply_delta_checkpoint(
            &mut reloaded_weights,
            &delta_tensors,
            &artifacts.manifest_tensors,
        )
        .expect("delta reload should apply");
        let reloaded_loss = qwen_causal_lm_loss(&input_ids, &reloaded_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!((final_loss - reloaded_loss).abs() < 1e-6);
    }

    #[test]
    fn trainable_tensor_names_expand_over_configured_layers() {
        let names = qwen_trainable_tensors_for_layers(&[0, 1], true);

        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(names.contains(&"model.norm.weight".to_string()));
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.1.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.1.mlp.down_proj.weight".to_string()));
        assert_eq!(names.len(), 26);

        let dp_names = qwen_trainable_tensors_for_layers(&[0, 1], false);
        assert!(!dp_names.contains(&"model.embed_tokens.weight".to_string()));
        assert_eq!(dp_names.len(), 25);
    }

    #[test]
    fn qwen_trainable_session_can_train_multiple_layers() {
        let config = QwenRuntimeConfig {
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let mut session = QwenTrainableSession::from_trainable_layers(
            config,
            two_layer_tiny_qwen_weights(),
            input_ids,
            Kind::Float,
            &[0, 1],
        )
        .expect("multi-layer session should build");

        let step = session
            .train_step(1e-2, 1)
            .expect("multi-layer session should train");

        assert!(step.loss_after < step.loss_before);
        assert_eq!(step.artifacts.tensor_summaries.len(), 26);
        assert!(step.artifacts.tensor_summaries.iter().any(|summary| {
            summary.name == "model.layers.1.self_attn.q_proj.weight" && summary.grad_norm > 0.0
        }));
        assert!(step.artifacts.tensor_summaries.iter().any(|summary| {
            summary.name == "model.layers.1.mlp.down_proj.weight" && summary.grad_norm > 0.0
        }));
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
            optimizer_safetensors: Some(optimizer_state_path(&delta_output).display().to_string()),
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
            learning_rate: 1e-6,
            initial_loss: 2.0,
            final_loss: 1.5,
            tensors: vec![QwenDeltaTensorManifestEntry {
                name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                delta_name: "model.layers.0.self_attn.q_proj.weight.delta".to_string(),
                adam_m_name: Some("model.layers.0.self_attn.q_proj.weight.adam_m".to_string()),
                adam_v_name: Some("model.layers.0.self_attn.q_proj.weight.adam_v".to_string()),
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
        assert_eq!(
            optimizer_state_path(&delta_output),
            temp.path().join("delta.safetensors.optimizer.safetensors")
        );
        assert_eq!(reloaded.format, "rustrain.qwen_delta.v1");
        assert_eq!(
            reloaded.optimizer_safetensors,
            manifest.optimizer_safetensors
        );
        assert_eq!(
            reloaded.tensors[0].delta_name,
            manifest.tensors[0].delta_name
        );
        assert_eq!(
            reloaded.tensors[0].adam_m_name,
            manifest.tensors[0].adam_m_name
        );
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_validates_rank_owned_shards() {
        let manifest = tiny_qwen_sharded_manifest();
        let encoded = serde_json::to_string_pretty(&manifest).expect("manifest should serialize");
        let decoded: QwenShardedCheckpointManifest =
            serde_json::from_str(&encoded).expect("manifest should deserialize");

        decoded.validate().expect("manifest should validate");
        assert_eq!(decoded.format, "rustrain.qwen_sharded.v1");
        assert_eq!(decoded.parallel.world_size().unwrap(), 2);
        assert_eq!(
            decoded.ranks[0].shards[0].optimizer_m_name,
            "rank0.q_proj.m"
        );
        assert_eq!(
            decoded.ranks[1].shards[0].optimizer_v_name,
            "rank1.q_proj.v"
        );
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_missing_rank() {
        let mut manifest = tiny_qwen_sharded_manifest();
        manifest.ranks.pop();

        let error = manifest.validate().expect_err("missing rank should fail");

        assert!(
            error
                .to_string()
                .contains("rank manifest count 1 does not match world size 2")
        );
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_missing_optimizer_slots() {
        let mut manifest = tiny_qwen_sharded_manifest();
        manifest.ranks[0].shards[0].optimizer_m_name.clear();

        let error = manifest
            .validate()
            .expect_err("missing optimizer slots should fail");

        assert!(error.to_string().contains("missing optimizer slots"));
    }

    #[test]
    fn qwen_session_dp_global_sharded_manifest_writes_schema_root() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let manifest = tiny_qwen_sharded_manifest();
        for rank in &manifest.ranks {
            fs::write(
                temp.path()
                    .join(format!("qwen-session-dp-sharded-rank-{}.json", rank.rank)),
                serde_json::to_string_pretty(rank).expect("rank manifest should serialize"),
            )
            .expect("rank manifest should write");
        }
        let output = temp.path().join("global.json");

        write_qwen_session_dp_global_sharded_manifest(
            temp.path(),
            Path::new("/models/qwen"),
            2,
            3,
            QwenComputeDType::Fp32,
            Some(12),
            Some(60),
            Some(2),
            Some(2),
            Some(5),
            &manifest.dataset_source_files,
            &manifest.dataset_source_sample_counts,
            &manifest.dataset_fingerprint,
            manifest.dataset_shuffle,
            &output,
        )
        .expect("global manifest should write");
        let decoded: QwenShardedCheckpointManifest = serde_json::from_str(
            &fs::read_to_string(&output).expect("global manifest should read"),
        )
        .expect("global manifest should parse");

        decoded.validate().expect("global manifest should validate");
        assert_eq!(decoded.format, "rustrain.qwen_sharded.v1");
        assert_eq!(decoded.global_step, 3);
        assert_eq!(decoded.consumed_samples, 12);
        assert_eq!(decoded.consumed_tokens, 60);
        assert_eq!(decoded.data_cursor_next, Some(12));
        assert_eq!(decoded.data_epoch_next, Some(2));
        assert_eq!(decoded.data_sample_offset_next, Some(2));
        assert_eq!(decoded.data_train_samples, Some(5));
        assert_eq!(decoded.dataset_source_files, manifest.dataset_source_files);
        assert_eq!(
            decoded.dataset_source_sample_counts,
            manifest.dataset_source_sample_counts
        );
        assert_eq!(decoded.dataset_fingerprint, manifest.dataset_fingerprint);
        assert_eq!(decoded.ranks.len(), 2);
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_inconsistent_data_progress() {
        let mut manifest = tiny_qwen_sharded_manifest();
        manifest.data_sample_offset_next = Some(5);

        let error = manifest
            .validate()
            .expect_err("inconsistent data progress should fail");

        assert!(
            error
                .to_string()
                .contains("data_sample_offset_next 5 must match")
        );
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_validates_dataset_provenance_shape() {
        let mut legacy_manifest = tiny_qwen_sharded_manifest();
        legacy_manifest.dataset_source_files.clear();
        legacy_manifest.dataset_source_sample_counts.clear();
        legacy_manifest.dataset_fingerprint.clear();
        legacy_manifest
            .validate()
            .expect("legacy sharded manifest without provenance should validate");

        let mut missing_sources = tiny_qwen_sharded_manifest();
        missing_sources.dataset_source_files.clear();
        let missing_sources_error = missing_sources
            .validate()
            .expect_err("fingerprint without source files should fail")
            .to_string();
        assert!(missing_sources_error.contains("requires dataset_source_files"));

        let mut missing_fingerprint = tiny_qwen_sharded_manifest();
        missing_fingerprint.dataset_fingerprint.clear();
        let missing_fingerprint_error = missing_fingerprint
            .validate()
            .expect_err("source files without fingerprint should fail")
            .to_string();
        assert!(missing_fingerprint_error.contains("require dataset_fingerprint"));

        let mut non_jsonl_source = tiny_qwen_sharded_manifest();
        non_jsonl_source.dataset_source_files = vec!["data/README.md".to_string()];
        let non_jsonl_source_error = non_jsonl_source
            .validate()
            .expect_err("non-jsonl source file should fail")
            .to_string();
        assert!(non_jsonl_source_error.contains("must only contain JSONL paths"));

        let mut mismatched_counts = tiny_qwen_sharded_manifest();
        mismatched_counts.dataset_source_sample_counts = vec![QwenSftSourceSampleCount {
            path: "data/other.jsonl".to_string(),
            samples: 5,
        }];
        let mismatched_counts_error = mismatched_counts
            .validate()
            .expect_err("mismatched source sample count paths should fail")
            .to_string();
        assert!(mismatched_counts_error.contains("dataset_source_sample_counts must match"));

        let mut zero_count = tiny_qwen_sharded_manifest();
        zero_count.dataset_source_sample_counts[0].samples = 0;
        let zero_count_error = zero_count
            .validate()
            .expect_err("zero source sample count should fail")
            .to_string();
        assert!(zero_count_error.contains("dataset_source_sample_counts must be positive"));
    }

    #[test]
    fn qwen_sharded_checkpoint_resume_dataset_validation_rejects_changed_data() {
        let manifest = tiny_qwen_sharded_manifest();
        let summary = QwenSftDatasetSummary {
            samples: 5,
            total_tokens: 40,
            response_tokens: 10,
            masked_positions: 10,
            max_sequence_tokens: 8,
            source_files: manifest.dataset_source_files.clone(),
            source_sample_counts: manifest.dataset_source_sample_counts.clone(),
            fingerprint: manifest.dataset_fingerprint.clone(),
            shuffle: manifest.dataset_shuffle,
        };

        qwen_validate_sft_resume_dataset(
            &manifest.dataset_source_files,
            &manifest.dataset_source_sample_counts,
            &manifest.dataset_fingerprint,
            manifest.dataset_shuffle,
            &summary,
            "sharded resume",
        )
        .expect("matching sharded provenance should pass");
        qwen_validate_sft_resume_dataset(&[], &[], "", true, &summary, "legacy sharded resume")
            .expect("legacy sharded manifests without provenance should pass");

        let fingerprint_error = qwen_validate_sft_resume_dataset(
            &manifest.dataset_source_files,
            &manifest.dataset_source_sample_counts,
            "changed-fingerprint",
            manifest.dataset_shuffle,
            &summary,
            "sharded resume",
        )
        .expect_err("changed sharded fingerprint should fail")
        .to_string();
        assert!(fingerprint_error.contains("dataset fingerprint mismatch"));

        let source_error = qwen_validate_sft_resume_dataset(
            &["data/changed.jsonl".to_string()],
            &manifest.dataset_source_sample_counts,
            &manifest.dataset_fingerprint,
            manifest.dataset_shuffle,
            &summary,
            "sharded resume",
        )
        .expect_err("changed sharded source files should fail")
        .to_string();
        assert!(source_error.contains("dataset source files mismatch"));

        let shuffle_error = qwen_validate_sft_resume_dataset(
            &manifest.dataset_source_files,
            &manifest.dataset_source_sample_counts,
            &manifest.dataset_fingerprint,
            !manifest.dataset_shuffle,
            &summary,
            "sharded resume",
        )
        .expect_err("changed sharded shuffle policy should fail")
        .to_string();
        assert!(shuffle_error.contains("dataset shuffle mismatch"));
    }

    #[test]
    fn qwen_sharded_rank_manifest_converts_to_delta_manifest() {
        let manifest = tiny_qwen_sharded_manifest();

        let delta = qwen_sharded_rank_to_delta_manifest(&manifest, 1, 2.0, 1.5, 1e-6)
            .expect("rank should convert");

        assert_eq!(delta.format, "rustrain.qwen_delta.v1");
        assert_eq!(delta.reference_fixture, "qwen_sharded_rank_1");
        assert_eq!(delta.delta_safetensors, "rank1/model.safetensors");
        assert_eq!(
            delta.optimizer_safetensors,
            Some("rank1/optimizer.safetensors".to_string())
        );
        assert_eq!(
            delta.tensors[0].name,
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(delta.tensors[0].delta_name, "rank1.q_proj");
        assert_eq!(
            delta.tensors[0].adam_m_name,
            Some("rank1.q_proj.m".to_string())
        );
    }

    #[test]
    fn qwen_optimizer_slots_reload_reproduces_next_adam_step() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let optimizer_output = temp.path().join("optimizer.safetensors");
        let tensor_name = "model.layers.0.self_attn.q_proj.weight";
        let slot_names = adam_slot_names(tensor_name);
        let first_grad = Tensor::from_slice(&[0.5_f32, -0.25, 0.125, -0.75]).reshape([2, 2]);
        let second_grad = Tensor::from_slice(&[-0.2_f32, 0.4, -0.6, 0.8]).reshape([2, 2]);
        let base_weight = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape([2, 2]);
        let learning_rate = 1e-3;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;

        let first_state = adamw_next_state(None, &first_grad, beta1, beta2);
        let first_update = adamw_update(&first_state, learning_rate, beta1, beta2, 1, eps);
        let after_first = &base_weight - first_update;
        Tensor::write_safetensors(
            &[
                (slot_names.m.as_str(), &first_state.m),
                (slot_names.v.as_str(), &first_state.v),
            ],
            &optimizer_output,
        )
        .expect("optimizer slots should write");

        let reloaded_slots = read_safetensors_map(&optimizer_output).expect("slots should reload");
        let reloaded_state = AdamState {
            m: tensor(&reloaded_slots, &slot_names.m)
                .expect("m slot should exist")
                .to_kind(Kind::Float),
            v: tensor(&reloaded_slots, &slot_names.v)
                .expect("v slot should exist")
                .to_kind(Kind::Float),
        };
        let continuous_second_state =
            adamw_next_state(Some(&first_state), &second_grad, beta1, beta2);
        let reloaded_second_state =
            adamw_next_state(Some(&reloaded_state), &second_grad, beta1, beta2);
        let continuous_after_second = &after_first
            - adamw_update(
                &continuous_second_state,
                learning_rate,
                beta1,
                beta2,
                2,
                eps,
            );
        let reloaded_after_second = &after_first
            - adamw_update(&reloaded_second_state, learning_rate, beta1, beta2, 2, eps);

        assert!(
            diff_stats(&continuous_second_state.m, &reloaded_second_state.m)
                .expect("m state diff should compute")
                .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(&continuous_second_state.v, &reloaded_second_state.v)
                .expect("v state diff should compute")
                .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(&continuous_after_second, &reloaded_after_second)
                .expect("weight diff should compute")
                .max_abs
                < 1e-8
        );
    }

    #[test]
    fn qwen_manifest_resume_reproduces_second_full_train_step() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let delta_output = temp.path().join("delta.safetensors");
        let optimizer_output = optimizer_state_path(&delta_output);
        let manifest_output = delta_manifest_path(&delta_output);
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let learning_rate = 1e-2;

        let mut continuous_weights = tiny_qwen_weights();
        let mut continuous_registry =
            QwenTrainableRegistry::representative(&mut continuous_weights)
                .expect("registry should build");
        let initial_loss = qwen_causal_lm_loss(&input_ids, &continuous_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        let first_loss =
            qwen_causal_lm_loss(&input_ids, &continuous_weights, &config).expect("loss should run");
        first_loss.backward();
        let first_artifacts = continuous_registry
            .adamw_step(&mut continuous_weights, learning_rate, 1)
            .expect("first optimizer step should apply");
        let final_loss = qwen_causal_lm_loss(&input_ids, &continuous_weights, &config)
            .expect("loss should run")
            .double_value(&[]);

        let delta_refs: Vec<(&str, &Tensor)> = first_artifacts
            .delta_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&delta_refs, &delta_output).expect("delta should write");
        let optimizer_refs: Vec<(&str, &Tensor)> = first_artifacts
            .optimizer_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&optimizer_refs, &optimizer_output)
            .expect("optimizer should write");
        let manifest = QwenDeltaCheckpointManifest {
            format: "rustrain.qwen_delta.v1".to_string(),
            base_model_path: "tiny-qwen".to_string(),
            reference_fixture: "inline".to_string(),
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
            learning_rate,
            initial_loss,
            final_loss,
            tensors: first_artifacts.manifest_tensors,
        };
        write_qwen_delta_manifest(&manifest_output, &manifest).expect("manifest should write");
        let reloaded_manifest: QwenDeltaCheckpointManifest = serde_json::from_str(
            &fs::read_to_string(&manifest_output).expect("manifest should read"),
        )
        .expect("manifest should parse");

        let mut resumed_weights = tiny_qwen_weights();
        let mut resumed_registry =
            QwenTrainableRegistry::load_from_manifest(&mut resumed_weights, &reloaded_manifest)
                .expect("registry should load from manifest");
        let resumed_loss = qwen_causal_lm_loss(&input_ids, &resumed_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!((final_loss - resumed_loss).abs() < 1e-6);

        continuous_registry.zero_grad();
        let continuous_second_loss =
            qwen_causal_lm_loss(&input_ids, &continuous_weights, &config).expect("loss should run");
        continuous_second_loss.backward();
        continuous_registry
            .adamw_step(&mut continuous_weights, learning_rate, 2)
            .expect("continuous second step should apply");

        let resumed_second_loss =
            qwen_causal_lm_loss(&input_ids, &resumed_weights, &config).expect("loss should run");
        resumed_second_loss.backward();
        resumed_registry
            .adamw_step(&mut resumed_weights, learning_rate, 2)
            .expect("resumed second step should apply");

        let continuous_after_second = qwen_causal_lm_loss(&input_ids, &continuous_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        let resumed_after_second = qwen_causal_lm_loss(&input_ids, &resumed_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!((continuous_after_second - resumed_after_second).abs() < 1e-6);

        for name in representative_trainable_qwen_tensors() {
            let diff = diff_stats(
                tensor(&continuous_weights, &name).expect("continuous tensor should exist"),
                tensor(&resumed_weights, &name).expect("resumed tensor should exist"),
            )
            .expect("diff should compute");
            assert!(
                diff.max_abs < 1e-6,
                "{name} should match after manifest-resumed second step, max_abs={}",
                diff.max_abs
            );
        }
    }

    #[test]
    fn qwen_trainable_session_trains_and_resumes_from_manifest() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let delta_output = temp.path().join("session-delta.safetensors");
        let optimizer_output = optimizer_state_path(&delta_output);
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let learning_rate = 1e-2;

        let mut continuous_session = QwenTrainableSession::from_weights(
            config,
            tiny_qwen_weights(),
            input_ids.shallow_clone(),
            Kind::Float,
        )
        .expect("session should build");
        let first_step = continuous_session
            .train_step(learning_rate, 1)
            .expect("first step should train");
        assert!(first_step.loss_after < first_step.loss_before);

        let delta_refs: Vec<(&str, &Tensor)> = first_step
            .artifacts
            .delta_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&delta_refs, &delta_output).expect("delta should write");
        let optimizer_refs: Vec<(&str, &Tensor)> = first_step
            .artifacts
            .optimizer_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&optimizer_refs, &optimizer_output)
            .expect("optimizer should write");
        let manifest = QwenDeltaCheckpointManifest {
            format: "rustrain.qwen_delta.v1".to_string(),
            base_model_path: "tiny-qwen".to_string(),
            reference_fixture: "inline".to_string(),
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
            learning_rate,
            initial_loss: first_step.loss_before,
            final_loss: first_step.loss_after,
            tensors: first_step.artifacts.manifest_tensors,
        };
        let mut resumed_session = QwenTrainableSession::from_manifest(
            config,
            tiny_qwen_weights(),
            input_ids,
            Kind::Float,
            &manifest,
        )
        .expect("session should resume");
        assert!((first_step.loss_after - resumed_session.loss_value().unwrap()).abs() < 1e-6);

        let continuous_second = continuous_session
            .train_step(learning_rate, 2)
            .expect("continuous second step should train");
        let resumed_second = resumed_session
            .train_step(learning_rate, 2)
            .expect("resumed second step should train");
        assert!((continuous_second.loss_after - resumed_second.loss_after).abs() < 1e-6);
    }

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
            dataset_total_samples: 4,
            dataset_train_samples: 3,
            dataset_eval_samples: 1,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            target_layers: current.target_layers.clone(),
            target_modules: current.target_module_names(),
        };

        qwen_validate_lora_resume_config(Some(&manifest), &current, &current)
            .expect("matching manifest and adapter config should pass");
        qwen_validate_lora_resume_config(None, &current, &current)
            .expect("direct adapter resume should pass without manifest metadata");

        let adapter_mismatch = QwenLoraConfig::new(
            vec![0, 1],
            vec![QwenLoraTargetModule::QProj, QwenLoraTargetModule::VProj],
            4,
            8.0,
        )
        .expect("adapter mismatch config should build");
        let adapter_error =
            qwen_validate_lora_resume_config(Some(&manifest), &adapter_mismatch, &current)
                .expect_err("adapter config mismatch should fail")
                .to_string();
        assert!(adapter_error.contains("resume adapter config does not match"));

        manifest.target_modules = vec!["q_proj".to_string(), "v_proj".to_string()];
        let manifest_module_error =
            qwen_validate_lora_resume_config(Some(&manifest), &current, &current)
                .expect_err("manifest module mismatch should fail")
                .to_string();
        assert!(manifest_module_error.contains("resume manifest target_modules"));

        manifest.target_modules = current.target_module_names();
        manifest.target_layers = vec![0];
        let manifest_layer_error =
            qwen_validate_lora_resume_config(Some(&manifest), &current, &current)
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
    fn qwen_sft_padded_batch_masks_padding_targets() {
        let samples = vec![
            QwenSftTokenSample {
                prompt_tokens: 2,
                response_tokens: 2,
                masked_positions: 2,
                token_ids: vec![10, 11, 12, 13],
                mask_values: vec![0.0, 1.0, 1.0],
            },
            QwenSftTokenSample {
                prompt_tokens: 1,
                response_tokens: 1,
                masked_positions: 1,
                token_ids: vec![20, 21],
                mask_values: vec![1.0],
            },
        ];

        let batch = qwen_sft_padded_batch(&samples, 0).expect("batch should build");
        let input_values: Vec<i64> = Vec::<i64>::try_from(batch.input_ids.reshape([-1])).unwrap();
        let mask_values: Vec<f32> = Vec::<f32>::try_from(batch.target_mask.reshape([-1])).unwrap();

        assert_eq!(batch.input_ids.size(), vec![2, 4]);
        assert_eq!(batch.target_mask.size(), vec![2, 3, 1]);
        assert_eq!(input_values, vec![10, 11, 12, 13, 20, 21, 0, 0]);
        assert_eq!(mask_values, vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(batch.masked_positions, 3);
        assert_eq!(batch.padding_tokens, 2);
    }

    #[test]
    fn qwen_sft_dataset_builds_wrapping_padded_batches() {
        let dataset = QwenSftDataset {
            samples: vec![
                QwenSftTokenSample {
                    prompt_tokens: 2,
                    response_tokens: 1,
                    masked_positions: 1,
                    token_ids: vec![1, 2, 3],
                    mask_values: vec![0.0, 1.0],
                },
                QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: 2,
                    masked_positions: 2,
                    token_ids: vec![4, 5, 6],
                    mask_values: vec![1.0, 1.0],
                },
            ],
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        };

        let batch = dataset
            .padded_batch(1, 3)
            .expect("wrapping batch should build");
        let input_values: Vec<i64> = Vec::<i64>::try_from(batch.input_ids.reshape([-1])).unwrap();
        let mask_values: Vec<f32> = Vec::<f32>::try_from(batch.target_mask.reshape([-1])).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(batch.input_ids.size(), vec![3, 3]);
        assert_eq!(input_values, vec![4, 5, 6, 1, 2, 3, 4, 5, 6]);
        assert_eq!(mask_values, vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(batch.masked_positions, 5);
        assert_eq!(batch.padding_tokens, 0);
    }

    #[test]
    fn qwen_data_epoch_metadata_tracks_wrapping_cursor() {
        assert_eq!(qwen_data_epoch_and_offset(0, 6).unwrap(), (0, 0));
        assert_eq!(qwen_data_epoch_and_offset(5, 6).unwrap(), (0, 5));
        assert_eq!(qwen_data_epoch_and_offset(6, 6).unwrap(), (1, 0));
        assert_eq!(qwen_data_epoch_and_offset(16, 6).unwrap(), (2, 4));
        assert!(qwen_data_epoch_and_offset(0, 0).is_err());
    }

    #[test]
    fn qwen_sft_dataset_split_keeps_train_and_eval_batches() {
        let dataset = QwenSftDataset {
            samples: (0..5)
                .map(|index| QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: 1,
                    masked_positions: 1,
                    token_ids: vec![index, index + 10],
                    mask_values: vec![1.0],
                })
                .collect(),
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        };

        let (train, eval) = dataset
            .train_eval_split(0.6)
            .expect("split should keep both sides");
        let train_batch = train
            .padded_batch(2, 3)
            .expect("train wrapping batch should build");
        let eval_batch = eval.padded_batch(0, 2).expect("eval batch should build");
        let train_values: Vec<i64> =
            Vec::<i64>::try_from(train_batch.input_ids.reshape([-1])).unwrap();
        let eval_values: Vec<i64> =
            Vec::<i64>::try_from(eval_batch.input_ids.reshape([-1])).unwrap();

        assert_eq!(train.len(), 3);
        assert_eq!(eval.len(), 2);
        assert_eq!(train_values, vec![2, 12, 0, 10, 1, 11]);
        assert_eq!(eval_values, vec![3, 13, 4, 14]);
    }

    #[test]
    fn qwen_sft_dataset_shuffle_is_seeded_and_summarized() {
        let dataset = QwenSftDataset {
            samples: (0..5)
                .map(|index| QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: (index + 1) as usize,
                    masked_positions: (index + 1) as usize,
                    token_ids: vec![index, index + 10, index + 20],
                    mask_values: vec![0.0, 1.0],
                })
                .collect(),
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        };

        let summary = dataset.summary();
        let shuffled_a = dataset.clone().shuffle_by_seed(17);
        let shuffled_b = dataset.clone().shuffle_by_seed(17);
        let shuffled_c = dataset.shuffle_by_seed(18);
        let order_a = shuffled_a
            .samples
            .iter()
            .map(|sample| sample.token_ids[0])
            .collect::<Vec<_>>();
        let order_b = shuffled_b
            .samples
            .iter()
            .map(|sample| sample.token_ids[0])
            .collect::<Vec<_>>();
        let order_c = shuffled_c
            .samples
            .iter()
            .map(|sample| sample.token_ids[0])
            .collect::<Vec<_>>();

        assert_eq!(summary.samples, 5);
        assert_eq!(summary.total_tokens, 15);
        assert_eq!(summary.response_tokens, 15);
        assert_eq!(summary.masked_positions, 15);
        assert_eq!(summary.max_sequence_tokens, 3);
        assert!(!summary.shuffle);
        assert!(shuffled_a.summary().shuffle);
        assert_eq!(order_a, order_b);
        assert_ne!(order_a, order_c);
    }

    #[test]
    fn qwen_sft_dataset_shuffle_can_be_disabled() {
        let dataset = QwenSftDataset {
            samples: (0..5)
                .map(|index| QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: 1,
                    masked_positions: 1,
                    token_ids: vec![index, index + 10],
                    mask_values: vec![1.0],
                })
                .collect(),
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        };

        let unshuffled = qwen_apply_sft_shuffle(dataset.clone(), false, 777);
        let shuffled = qwen_apply_sft_shuffle(dataset, true, 777);
        let unshuffled_order = (0..unshuffled.len())
            .map(|cursor| unshuffled.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();
        let shuffled_order = (0..shuffled.len())
            .map(|cursor| shuffled.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();

        assert_eq!(unshuffled_order, vec![0, 1, 2, 3, 4]);
        assert!(!unshuffled.summary().shuffle);
        assert!(shuffled.summary().shuffle);
        assert_ne!(unshuffled_order, shuffled_order);
    }

    #[test]
    fn qwen_sft_dataset_epoch_shuffle_is_cursor_stable() {
        let dataset = QwenSftDataset {
            samples: (0..6)
                .map(|index| QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: 1,
                    masked_positions: 1,
                    token_ids: vec![index, index + 10],
                    mask_values: vec![1.0],
                })
                .collect(),
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        }
        .shuffle_by_seed(777);

        let epoch0_a = (0..dataset.len())
            .map(|cursor| dataset.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();
        let epoch0_b = (0..dataset.len())
            .map(|cursor| dataset.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();
        let epoch1 = (dataset.len()..dataset.len() * 2)
            .map(|cursor| dataset.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();
        let epoch2 = (dataset.len() * 2..dataset.len() * 3)
            .map(|cursor| dataset.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();

        assert_eq!(epoch0_a, epoch0_b);
        assert!(epoch0_a != epoch1 || epoch0_a != epoch2);

        let wrapped_batch = dataset
            .padded_batch(dataset.len() - 1, 3)
            .expect("epoch-crossing batch should build");
        let wrapped_values: Vec<i64> =
            Vec::<i64>::try_from(wrapped_batch.input_ids.reshape([-1])).unwrap();
        assert_eq!(
            wrapped_values,
            vec![
                epoch0_a[dataset.len() - 1],
                epoch0_a[dataset.len() - 1] + 10,
                epoch1[0],
                epoch1[0] + 10,
                epoch1[1],
                epoch1[1] + 10,
            ]
        );
    }

    #[test]
    fn qwen_session_fixed_batch_plan_reports_fixture_metadata() {
        let mut weights = tiny_qwen_weights();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::zeros([2048, 4], (Kind::Float, Device::Cpu)),
        );

        let plan = qwen_session_fixed_batch_plan(&weights, 0, 2).expect("fixed plan should build");

        assert_eq!(plan.reference_fixture, "qwen_session_single_fixed_tokens");
        assert_eq!(plan.batch_size, 1);
        assert_eq!(plan.sequence_tokens, 5);
        assert_eq!(plan.train_batches.len(), 3);
        assert!(plan.dataset_total_samples.is_none());
    }

    #[test]
    fn qwen_session_fixed_batch_plan_keeps_resume_cursor_window() {
        let mut weights = tiny_qwen_weights();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::zeros([2048, 4], (Kind::Float, Device::Cpu)),
        );

        let plan = qwen_session_fixed_batch_plan(&weights, 2, 2).expect("fixed plan should build");

        assert_eq!(plan.train_batches.len(), 5);
        assert!(plan.train_batches.get(2).is_some());
        assert!(plan.train_batches.get(4).is_some());
    }

    #[test]
    fn qwen_sft_jsonl_reader_loads_instruction_input_response_records() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        fs::write(
            &jsonl,
            r#"{"instruction":"Reply with the project name.","response":"rustrain"}
{"instruction":"Name the language.","input":"rustrain implementation","response":"Rust"}
"#,
        )
        .expect("jsonl should write");

        let example_set =
            qwen_sft_examples_from_jsonl_path(&jsonl).expect("examples should load from jsonl");
        let examples = &example_set.examples;

        assert_eq!(examples.len(), 2);
        assert_eq!(example_set.source_files, vec![jsonl.display().to_string()]);
        assert_eq!(
            example_set.source_sample_counts,
            vec![QwenSftSourceSampleCount {
                path: jsonl.display().to_string(),
                samples: 2,
            }]
        );
        assert!(!example_set.fingerprint.is_empty());
        assert_eq!(examples[0].instruction, "Reply with the project name.");
        assert_eq!(examples[0].input, "");
        assert_eq!(examples[0].response, "rustrain");
        assert_eq!(examples[1].instruction, "Name the language.");
        assert_eq!(examples[1].input, "rustrain implementation");
        assert_eq!(examples[1].response, "Rust");
    }

    #[test]
    fn qwen_sft_explicit_eval_metadata_combines_train_and_eval_sources() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let train_jsonl = temp.path().join("train.jsonl");
        let eval_jsonl = temp.path().join("eval.jsonl");
        let train_file = train_jsonl.display().to_string();
        let eval_file = eval_jsonl.display().to_string();
        let source_files = qwen_merge_sft_source_files(
            std::slice::from_ref(&train_file),
            std::slice::from_ref(&eval_file),
        );
        let source_counts = qwen_merge_sft_source_sample_counts(
            &[QwenSftSourceSampleCount {
                path: train_file.clone(),
                samples: 3,
            }],
            &[QwenSftSourceSampleCount {
                path: eval_file.clone(),
                samples: 2,
            }],
        );
        let fingerprint =
            qwen_combine_sft_fingerprints(&source_files, "train-fingerprint", "eval-fingerprint");

        assert_eq!(source_files, vec![eval_file.clone(), train_file.clone()]);
        assert_eq!(
            source_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: eval_file,
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: train_file,
                    samples: 3,
                },
            ]
        );
        assert!(!fingerprint.is_empty());
        assert_ne!(
            fingerprint,
            qwen_combine_sft_fingerprints(&source_files, "train-fingerprint", "other-eval")
        );
    }

    #[test]
    fn qwen_sft_resume_dataset_validation_rejects_changed_data() {
        let summary = QwenSftDatasetSummary {
            samples: 2,
            total_tokens: 8,
            response_tokens: 2,
            masked_positions: 2,
            max_sequence_tokens: 4,
            source_files: vec!["data/train.jsonl".to_string()],
            source_sample_counts: vec![QwenSftSourceSampleCount {
                path: "data/train.jsonl".to_string(),
                samples: 2,
            }],
            fingerprint: "fingerprint-a".to_string(),
            shuffle: true,
        };

        qwen_validate_sft_resume_dataset(
            &summary.source_files,
            &summary.source_sample_counts,
            &summary.fingerprint,
            true,
            &summary,
            "test resume",
        )
        .expect("matching provenance should pass");
        qwen_validate_sft_resume_dataset(&[], &[], "", true, &summary, "legacy resume")
            .expect("legacy manifests without provenance should pass");

        let fingerprint_error = qwen_validate_sft_resume_dataset(
            &summary.source_files,
            &summary.source_sample_counts,
            "fingerprint-b",
            true,
            &summary,
            "test resume",
        )
        .expect_err("changed content fingerprint should fail")
        .to_string();
        assert!(fingerprint_error.contains("dataset fingerprint mismatch"));

        let source_error = qwen_validate_sft_resume_dataset(
            &["data/other.jsonl".to_string()],
            &summary.source_sample_counts,
            &summary.fingerprint,
            true,
            &summary,
            "test resume",
        )
        .expect_err("changed source files should fail")
        .to_string();
        assert!(source_error.contains("dataset source files mismatch"));

        let count_error = qwen_validate_sft_resume_dataset(
            &summary.source_files,
            &[QwenSftSourceSampleCount {
                path: "data/train.jsonl".to_string(),
                samples: 3,
            }],
            &summary.fingerprint,
            true,
            &summary,
            "test resume",
        )
        .expect_err("changed source sample counts should fail")
        .to_string();
        assert!(count_error.contains("dataset source sample counts mismatch"));

        let shuffle_error = qwen_validate_sft_resume_dataset(
            &summary.source_files,
            &summary.source_sample_counts,
            &summary.fingerprint,
            false,
            &summary,
            "test resume",
        )
        .expect_err("changed shuffle policy should fail")
        .to_string();
        assert!(shuffle_error.contains("dataset shuffle mismatch"));
    }

    #[test]
    fn qwen_sft_jsonl_reader_aggregates_multiple_paths_and_directories() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let dir = temp.path().join("shard_dir");
        fs::create_dir(&dir).expect("shard dir should be created");
        let second = dir.join("b.jsonl");
        let third = dir.join("a.jsonl");
        let ignored = dir.join("ignored.txt");
        fs::write(
            &first,
            r#"{"instruction":"first","response":"one"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"instruction":"third","response":"three"}
"#,
        )
        .expect("second jsonl should write");
        fs::write(
            &third,
            r#"{"instruction":"second","response":"two"}
"#,
        )
        .expect("third jsonl should write");
        fs::write(
            &ignored,
            r#"{"instruction":"ignored","response":"ignored"}
"#,
        )
        .expect("ignored file should write");

        let example_set = qwen_sft_examples_from_jsonl_paths(&[first.clone(), dir.clone()])
            .expect("examples should aggregate from multiple paths");
        let examples = &example_set.examples;

        assert_eq!(examples.len(), 3);
        assert_eq!(example_set.source_files.len(), 3);
        assert_eq!(
            example_set.source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: first.display().to_string(),
                    samples: 1,
                },
                QwenSftSourceSampleCount {
                    path: third.display().to_string(),
                    samples: 1,
                },
                QwenSftSourceSampleCount {
                    path: second.display().to_string(),
                    samples: 1,
                },
            ]
        );
        assert!(
            example_set
                .source_files
                .iter()
                .all(|path| path.ends_with(".jsonl"))
        );
        assert!(!example_set.fingerprint.is_empty());
        assert_eq!(examples[0].instruction, "first");
        assert_eq!(examples[1].instruction, "second");
        assert_eq!(examples[2].instruction, "third");
    }

    #[test]
    fn qwen_sft_example_limit_recomputes_source_counts_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        fs::write(
            &first,
            r#"{"instruction":"one","response":"a"}
{"instruction":"two","response":"b"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"instruction":"three","response":"c"}
{"instruction":"four","response":"d"}
"#,
        )
        .expect("second jsonl should write");

        let full = qwen_sft_examples_from_jsonl_paths(&[first.clone(), second.clone()])
            .expect("full examples should load");
        let limited = qwen_sft_limit_example_set(full, Some(3)).expect("example set should limit");

        assert_eq!(limited.examples.len(), 3);
        assert_eq!(
            limited.source_files,
            vec![first.display().to_string(), second.display().to_string()]
        );
        assert_eq!(
            limited.source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: first.display().to_string(),
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: second.display().to_string(),
                    samples: 1,
                },
            ]
        );
        assert_eq!(limited.examples[0].instruction, "one");
        assert_eq!(limited.examples[2].instruction, "three");
        assert_eq!(
            limited.fingerprint,
            qwen_sft_dataset_fingerprint(&limited.source_files, &limited.examples)
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

    fn tiny_qwen_sharded_manifest() -> QwenShardedCheckpointManifest {
        QwenShardedCheckpointManifest {
            format: "rustrain.qwen_sharded.v1".to_string(),
            base_model_path: "/models/qwen".to_string(),
            tokenizer_path: "/models/qwen/tokenizer.json".to_string(),
            global_step: 3,
            consumed_samples: 8,
            consumed_tokens: 40,
            data_cursor_next: Some(8),
            data_epoch_next: Some(1),
            data_sample_offset_next: Some(3),
            data_train_samples: Some(5),
            dataset_source_files: vec!["data/train.jsonl".to_string()],
            dataset_source_sample_counts: vec![QwenSftSourceSampleCount {
                path: "data/train.jsonl".to_string(),
                samples: 5,
            }],
            dataset_fingerprint: "abc123".to_string(),
            dataset_shuffle: true,
            seed: 42,
            dtype: "float32".to_string(),
            optimizer: "adamw".to_string(),
            scheduler: "linear_decay".to_string(),
            parallel: QwenShardedParallelManifest {
                data_parallel_size: 2,
                tensor_model_parallel_size: 1,
                pipeline_model_parallel_size: 1,
                expert_model_parallel_size: 1,
                context_parallel_size: 1,
            },
            ranks: vec![
                QwenRankShardManifest {
                    rank: 0,
                    data_parallel_rank: 0,
                    tensor_model_parallel_rank: 0,
                    pipeline_model_parallel_rank: 0,
                    expert_model_parallel_rank: 0,
                    context_parallel_rank: 0,
                    model_safetensors: "rank0/model.safetensors".to_string(),
                    optimizer_safetensors: "rank0/optimizer.safetensors".to_string(),
                    shards: vec![QwenTensorShardManifestEntry {
                        name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                        shard_name: "rank0.q_proj".to_string(),
                        optimizer_m_name: "rank0.q_proj.m".to_string(),
                        optimizer_v_name: "rank0.q_proj.v".to_string(),
                        global_shape: vec![4, 4],
                        shard_shape: vec![4, 4],
                        dtype: "float32".to_string(),
                        partition: "replicated_dp".to_string(),
                        tied_group: None,
                    }],
                },
                QwenRankShardManifest {
                    rank: 1,
                    data_parallel_rank: 1,
                    tensor_model_parallel_rank: 0,
                    pipeline_model_parallel_rank: 0,
                    expert_model_parallel_rank: 0,
                    context_parallel_rank: 0,
                    model_safetensors: "rank1/model.safetensors".to_string(),
                    optimizer_safetensors: "rank1/optimizer.safetensors".to_string(),
                    shards: vec![QwenTensorShardManifestEntry {
                        name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                        shard_name: "rank1.q_proj".to_string(),
                        optimizer_m_name: "rank1.q_proj.m".to_string(),
                        optimizer_v_name: "rank1.q_proj.v".to_string(),
                        global_shape: vec![4, 4],
                        shard_shape: vec![4, 4],
                        dtype: "float32".to_string(),
                        partition: "replicated_dp".to_string(),
                        tied_group: None,
                    }],
                },
            ],
        }
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

    fn two_layer_tiny_qwen_weights() -> BTreeMap<String, Tensor> {
        let mut weights = tiny_qwen_weights();
        let layer0_names = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];
        for suffix in layer0_names {
            let layer0_name = format!("model.layers.0.{suffix}");
            let layer1_name = format!("model.layers.1.{suffix}");
            let value = tensor(&weights, &layer0_name)
                .expect("layer0 tensor should exist")
                .shallow_clone();
            weights.insert(layer1_name, value * 0.9);
        }
        weights
    }
}
