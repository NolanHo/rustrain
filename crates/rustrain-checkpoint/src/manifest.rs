use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};

#[cfg(feature = "tch")]
use anyhow::Context;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenSftSourceSampleCount {
    pub path: String,
    pub samples: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QwenLoraSftAdapterManifest {
    pub format: String,
    pub base_model_path: String,
    pub adapter_safetensors: String,
    pub compute_kind: String,
    pub steps: usize,
    pub train_step: u64,
    pub data_cursor_start: usize,
    pub data_cursor_end: usize,
    pub data_cursor_next: usize,
    #[serde(default)]
    pub data_epoch_start: usize,
    #[serde(default)]
    pub data_epoch_end: usize,
    #[serde(default)]
    pub data_epoch_next: usize,
    #[serde(default)]
    pub data_sample_offset_start: usize,
    #[serde(default)]
    pub data_sample_offset_end: usize,
    #[serde(default)]
    pub data_sample_offset_next: usize,
    #[serde(default)]
    pub dataset_source_files: Vec<String>,
    #[serde(default)]
    pub dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    #[serde(default)]
    pub dataset_fingerprint: String,
    pub dataset_order_seed: u64,
    #[serde(default = "qwen_manifest_default_dataset_shuffle")]
    pub dataset_shuffle: bool,
    #[serde(default)]
    pub streaming_train_batches: bool,
    pub dataset_total_samples: usize,
    pub dataset_train_samples: usize,
    pub dataset_eval_samples: usize,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub target_layers: Vec<usize>,
    pub target_modules: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QwenSessionDpCheckpointManifest {
    pub format: String,
    pub base_model_path: String,
    pub writer_rank: usize,
    pub world_size: usize,
    pub tensor_count: usize,
    pub max_grad_delta: f32,
    pub expected_loss: f64,
    pub dtype: String,
    pub steps: usize,
    pub train_step: u64,
    #[serde(default)]
    pub data_cursor_start: Option<usize>,
    #[serde(default)]
    pub data_cursor_end: Option<usize>,
    #[serde(default)]
    pub data_cursor_next: Option<usize>,
    #[serde(default)]
    pub data_epoch_start: Option<usize>,
    #[serde(default)]
    pub data_epoch_end: Option<usize>,
    #[serde(default)]
    pub data_epoch_next: Option<usize>,
    #[serde(default)]
    pub data_sample_offset_start: Option<usize>,
    #[serde(default)]
    pub data_sample_offset_end: Option<usize>,
    #[serde(default)]
    pub data_sample_offset_next: Option<usize>,
    #[serde(default)]
    pub dataset_source_files: Vec<String>,
    #[serde(default)]
    pub dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    #[serde(default)]
    pub dataset_fingerprint: String,
    #[serde(default = "qwen_manifest_default_dataset_shuffle")]
    pub dataset_shuffle: bool,
    #[serde(default)]
    pub streaming_train_batches: Option<bool>,
    pub learning_rate: f64,
    pub delta_safetensors: String,
    pub optimizer_safetensors: String,
    pub post_update_loss: f64,
    pub global_post_update_loss: f64,
    pub global_step_losses: Vec<f64>,
    pub trainable_tensors: Vec<String>,
    pub tensors: Vec<QwenDeltaTensorManifestEntry>,
}

impl QwenSessionDpCheckpointManifest {
    pub fn to_delta_manifest(&self) -> Result<QwenDeltaCheckpointManifest> {
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
            streaming_train_batches: self.streaming_train_batches,
            learning_rate: self.learning_rate,
            initial_loss: self.expected_loss,
            final_loss: self.global_post_update_loss,
            tensors: self.tensors.clone(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QwenDpCheckpointManifest {
    pub format: String,
    pub writer_rank: usize,
    pub world_size: usize,
    pub tensor_count: usize,
    pub max_grad_delta: f32,
    pub expected_loss: f64,
    pub dtype: String,
    pub steps: usize,
    pub learning_rate: f64,
    pub post_update_loss: f64,
    pub global_post_update_loss: f64,
    pub global_step_losses: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QwenDeltaCheckpointManifest {
    pub format: String,
    pub base_model_path: String,
    pub reference_fixture: String,
    pub delta_safetensors: String,
    #[serde(default)]
    pub optimizer_safetensors: Option<String>,
    pub train_step: u64,
    #[serde(default)]
    pub data_cursor_start: Option<usize>,
    #[serde(default)]
    pub data_cursor_end: Option<usize>,
    #[serde(default)]
    pub data_cursor_next: Option<usize>,
    #[serde(default)]
    pub data_epoch_start: Option<usize>,
    #[serde(default)]
    pub data_epoch_end: Option<usize>,
    #[serde(default)]
    pub data_epoch_next: Option<usize>,
    #[serde(default)]
    pub data_sample_offset_start: Option<usize>,
    #[serde(default)]
    pub data_sample_offset_end: Option<usize>,
    #[serde(default)]
    pub data_sample_offset_next: Option<usize>,
    #[serde(default)]
    pub dataset_source_files: Vec<String>,
    #[serde(default)]
    pub dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    #[serde(default)]
    pub dataset_fingerprint: String,
    #[serde(default = "qwen_manifest_default_dataset_shuffle")]
    pub dataset_shuffle: bool,
    #[serde(default)]
    pub streaming_train_batches: Option<bool>,
    pub learning_rate: f64,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub tensors: Vec<QwenDeltaTensorManifestEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenDeltaTensorManifestEntry {
    pub name: String,
    pub delta_name: String,
    #[serde(default)]
    pub adam_m_name: Option<String>,
    #[serde(default)]
    pub adam_v_name: Option<String>,
    pub shape: Vec<i64>,
    pub dtype: String,
    pub grad_norm: f64,
    pub delta_norm: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenShardedCheckpointManifest {
    pub format: String,
    pub base_model_path: String,
    pub tokenizer_path: String,
    pub global_step: u64,
    pub consumed_samples: u64,
    pub consumed_tokens: u64,
    #[serde(default)]
    pub data_cursor_next: Option<u64>,
    #[serde(default)]
    pub data_epoch_next: Option<u64>,
    #[serde(default)]
    pub data_sample_offset_next: Option<u64>,
    #[serde(default)]
    pub data_train_samples: Option<u64>,
    #[serde(default)]
    pub dataset_source_files: Vec<String>,
    #[serde(default)]
    pub dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    #[serde(default)]
    pub dataset_fingerprint: String,
    #[serde(default = "qwen_manifest_default_dataset_shuffle")]
    pub dataset_shuffle: bool,
    #[serde(default)]
    pub streaming_train_batches: Option<bool>,
    pub seed: u64,
    pub dtype: String,
    pub optimizer: String,
    pub scheduler: String,
    pub parallel: QwenShardedParallelManifest,
    pub ranks: Vec<QwenRankShardManifest>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenShardedParallelManifest {
    pub data_parallel_size: usize,
    pub tensor_model_parallel_size: usize,
    pub pipeline_model_parallel_size: usize,
    pub expert_model_parallel_size: usize,
    pub context_parallel_size: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenRankShardManifest {
    pub rank: usize,
    pub data_parallel_rank: usize,
    pub tensor_model_parallel_rank: usize,
    pub pipeline_model_parallel_rank: usize,
    pub expert_model_parallel_rank: usize,
    pub context_parallel_rank: usize,
    pub model_safetensors: String,
    pub optimizer_safetensors: String,
    pub shards: Vec<QwenTensorShardManifestEntry>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenTensorShardManifestEntry {
    pub name: String,
    pub shard_name: String,
    pub optimizer_m_name: String,
    pub optimizer_v_name: String,
    pub global_shape: Vec<i64>,
    pub shard_shape: Vec<i64>,
    pub dtype: String,
    pub partition: String,
    pub tied_group: Option<String>,
}

fn qwen_manifest_default_dataset_shuffle() -> bool {
    true
}

#[allow(dead_code)]
impl QwenShardedCheckpointManifest {
    pub fn validate(&self) -> Result<()> {
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
        if self.scheduler.is_empty() {
            bail!("Qwen sharded checkpoint requires scheduler");
        }
        if self.global_step == 0 {
            bail!("Qwen sharded checkpoint global_step must be positive");
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
                .any(|source| !source.ends_with(".jsonl") && !source.ends_with(".arrow"))
            {
                bail!(
                    "Qwen sharded checkpoint dataset_source_files must only contain JSONL or Arrow paths"
                );
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

        let mut seen_ranks = std::collections::BTreeSet::new();
        let mut seen_rank_axes = std::collections::BTreeSet::new();
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
            let rank_axes = (
                rank.data_parallel_rank,
                rank.tensor_model_parallel_rank,
                rank.pipeline_model_parallel_rank,
                rank.expert_model_parallel_rank,
                rank.context_parallel_rank,
            );
            if !seen_rank_axes.insert(rank_axes) {
                bail!(
                    "Qwen sharded checkpoint contains duplicate parallel rank axes {:?}",
                    rank_axes
                );
            }
            let expected_linear_rank = self.parallel.linear_rank(rank);
            if rank.rank != expected_linear_rank {
                bail!(
                    "Qwen sharded checkpoint rank {} does not match linear parallel rank {}",
                    rank.rank,
                    expected_linear_rank
                );
            }
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
            let mut seen_tensor_names = std::collections::BTreeSet::new();
            let mut seen_model_shards = std::collections::BTreeSet::new();
            let mut seen_optimizer_slots = std::collections::BTreeSet::new();
            for shard in &rank.shards {
                shard.validate(rank.rank)?;
                if !seen_tensor_names.insert(shard.name.as_str()) {
                    bail!(
                        "Qwen sharded checkpoint rank {} contains duplicate tensor shard {}",
                        rank.rank,
                        shard.name
                    );
                }
                if !seen_model_shards.insert(shard.shard_name.as_str()) {
                    bail!(
                        "Qwen sharded checkpoint rank {} contains duplicate shard_name {}",
                        rank.rank,
                        shard.shard_name
                    );
                }
                for slot in [&shard.optimizer_m_name, &shard.optimizer_v_name] {
                    if !seen_optimizer_slots.insert(slot.as_str()) {
                        bail!(
                            "Qwen sharded checkpoint rank {} contains duplicate optimizer slot {}",
                            rank.rank,
                            slot
                        );
                    }
                    if slot == &shard.shard_name {
                        bail!(
                            "Qwen sharded checkpoint rank {} tensor {} optimizer slot {} collides with shard_name",
                            rank.rank,
                            shard.name,
                            slot
                        );
                    }
                }
            }
        }
        for expected_rank in 0..expected_world_size {
            if !seen_ranks.contains(&expected_rank) {
                bail!("Qwen sharded checkpoint is missing rank {expected_rank}");
            }
        }
        Ok(())
    }

    #[cfg(feature = "tch")]
    pub fn validate_artifacts(&self) -> Result<()> {
        self.validate()?;
        for rank in &self.ranks {
            let model_tensors = crate::safetensors::read_safetensors_map(std::path::Path::new(
                &rank.model_safetensors,
            ))
            .with_context(|| {
                format!(
                    "failed to validate Qwen sharded checkpoint rank {} model artifacts",
                    rank.rank
                )
            })?;
            let optimizer_tensors = crate::safetensors::read_safetensors_map(std::path::Path::new(
                &rank.optimizer_safetensors,
            ))
            .with_context(|| {
                format!(
                    "failed to validate Qwen sharded checkpoint rank {} optimizer artifacts",
                    rank.rank
                )
            })?;
            for shard in &rank.shards {
                let model_tensor = crate::safetensors::tensor(&model_tensors, &shard.shard_name)
                    .with_context(|| {
                        format!(
                            "Qwen sharded checkpoint rank {} missing model shard {} for {}",
                            rank.rank, shard.shard_name, shard.name
                        )
                    })?;
                if model_tensor.size() != shard.shard_shape {
                    bail!(
                        "Qwen sharded checkpoint rank {} model shard {} shape {:?} does not match manifest shard_shape {:?}",
                        rank.rank,
                        shard.shard_name,
                        model_tensor.size(),
                        shard.shard_shape
                    );
                }
                let optimizer_m = crate::safetensors::tensor(
                    &optimizer_tensors,
                    &shard.optimizer_m_name,
                )
                .with_context(|| {
                    format!(
                        "Qwen sharded checkpoint rank {} missing optimizer m slot {} for {}",
                        rank.rank, shard.optimizer_m_name, shard.name
                    )
                })?;
                let optimizer_v = crate::safetensors::tensor(
                    &optimizer_tensors,
                    &shard.optimizer_v_name,
                )
                .with_context(|| {
                    format!(
                        "Qwen sharded checkpoint rank {} missing optimizer v slot {} for {}",
                        rank.rank, shard.optimizer_v_name, shard.name
                    )
                })?;
                if optimizer_m.size() != shard.shard_shape {
                    bail!(
                        "Qwen sharded checkpoint rank {} optimizer m slot {} shape {:?} does not match manifest shard_shape {:?}",
                        rank.rank,
                        shard.optimizer_m_name,
                        optimizer_m.size(),
                        shard.shard_shape
                    );
                }
                if optimizer_v.size() != shard.shard_shape {
                    bail!(
                        "Qwen sharded checkpoint rank {} optimizer v slot {} shape {:?} does not match manifest shard_shape {:?}",
                        rank.rank,
                        shard.optimizer_v_name,
                        optimizer_v.size(),
                        shard.shard_shape
                    );
                }
            }
        }
        Ok(())
    }
}

#[allow(dead_code)]
impl QwenShardedParallelManifest {
    pub fn world_size(&self) -> Result<usize> {
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

    pub fn validate_rank(&self, rank: &QwenRankShardManifest) -> Result<()> {
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

    pub fn linear_rank(&self, rank: &QwenRankShardManifest) -> usize {
        (((rank.data_parallel_rank * self.tensor_model_parallel_size
            + rank.tensor_model_parallel_rank)
            * self.pipeline_model_parallel_size
            + rank.pipeline_model_parallel_rank)
            * self.expert_model_parallel_size
            + rank.expert_model_parallel_rank)
            * self.context_parallel_size
            + rank.context_parallel_rank
    }
}

#[allow(dead_code)]
impl QwenTensorShardManifestEntry {
    pub fn validate(&self, rank: usize) -> Result<()> {
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
        if self.global_shape.len() != self.shard_shape.len() {
            bail!(
                "Qwen sharded checkpoint rank {rank} tensor {} global_shape rank {} does not match shard_shape rank {}",
                self.name,
                self.global_shape.len(),
                self.shard_shape.len()
            );
        }
        for (dim_index, (&global_dim, &shard_dim)) in self
            .global_shape
            .iter()
            .zip(self.shard_shape.iter())
            .enumerate()
        {
            if global_dim <= 0 || shard_dim <= 0 {
                bail!(
                    "Qwen sharded checkpoint rank {rank} tensor {} shape dim {dim_index} must be positive",
                    self.name
                );
            }
            if shard_dim > global_dim {
                bail!(
                    "Qwen sharded checkpoint rank {rank} tensor {} shard_shape dim {dim_index} exceeds global_shape",
                    self.name
                );
            }
        }
        if self.dtype.is_empty() {
            bail!(
                "Qwen sharded checkpoint rank {rank} tensor {} is missing dtype",
                self.name
            );
        }
        if !matches!(self.dtype.as_str(), "float32" | "fp32" | "bf16") {
            bail!(
                "Qwen sharded checkpoint rank {rank} tensor {} has unsupported dtype {}",
                self.name,
                self.dtype
            );
        }
        if self.partition.is_empty() {
            bail!(
                "Qwen sharded checkpoint rank {rank} tensor {} is missing partition policy",
                self.name
            );
        }
        if !matches!(
            self.partition.as_str(),
            "replicated_dp" | "replicated_norm_smoke" | "tp_row" | "tp_col"
        ) {
            bail!(
                "Qwen sharded checkpoint rank {rank} tensor {} has unsupported partition policy {}",
                self.name,
                self.partition
            );
        }
        Ok(())
    }
}
