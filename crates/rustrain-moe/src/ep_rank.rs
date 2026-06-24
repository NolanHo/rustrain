use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array2, array};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor};

use rustrain_nccl::nccl as nccl_smoke;
#[derive(Debug, Serialize, Deserialize)]
struct ExpertParallelRankSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    owned_expert_start: usize,
    owned_expert_end: usize,
    owned_token_indices: Vec<usize>,
    expert_load: Vec<usize>,
    local_output_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExpertParallelNcclRankSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    owned_expert_start: usize,
    owned_expert_end: usize,
    owned_token_indices: Vec<usize>,
    expert_load: Vec<usize>,
    reduced_output_shape: Vec<i64>,
    combine_max_abs: f64,
    combine_mean_abs: f64,
    train_initial_loss: f64,
    train_final_loss: f64,
    train_loss_improved: bool,
    scale_grad_norm: f64,
    checkpoint_manifest_output: String,
    checkpoint_model_safetensors: String,
    checkpoint_optimizer_safetensors: String,
    checkpoint_tensor_count: usize,
    reload_scale_max_abs: f64,
    reload_optimizer_max_abs: f64,
    continuous_second_loss: f64,
    resumed_second_loss: f64,
    second_step_delta: f64,
    second_step_scale_max_abs: f64,
    second_step_optimizer_max_abs: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExpertParallelSparseRankSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    source_token_indices: Vec<usize>,
    owned_expert_start: usize,
    owned_expert_end: usize,
    owned_token_indices: Vec<usize>,
    global_expert_load: Vec<usize>,
    load_balance_loss: f64,
    dispatch_send_counts: Vec<usize>,
    dispatch_recv_counts: Vec<usize>,
    combine_send_counts: Vec<usize>,
    combine_recv_counts: Vec<usize>,
    assembled_output_shape: Vec<i64>,
    reference_output_shape: Vec<i64>,
    sparse_output_max_abs: f64,
    sparse_output_mean_abs: f64,
    train_initial_loss: f64,
    train_final_loss: f64,
    train_loss_improved: bool,
    scale_grad_norm: f64,
    checkpoint_manifest_output: String,
    checkpoint_model_safetensors: String,
    checkpoint_optimizer_safetensors: String,
    checkpoint_tensor_count: usize,
    reload_scale_max_abs: f64,
    reload_optimizer_max_abs: f64,
    reload_loss: f64,
    reload_loss_delta: f64,
    continuous_second_loss: f64,
    resumed_second_loss: f64,
    second_step_delta: f64,
    second_step_scale_max_abs: f64,
    second_step_optimizer_max_abs: f64,
    second_step_optimizer_step_delta: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExpertParallelTchMoeRankSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    resume_from: Option<String>,
    resumed_sharded_checkpoint: bool,
    source_token_indices: Vec<usize>,
    owned_expert_start: usize,
    owned_expert_end: usize,
    owned_token_indices: Vec<usize>,
    global_expert_load: Vec<usize>,
    load_balance_loss: f64,
    dispatch_send_counts: Vec<usize>,
    dispatch_recv_counts: Vec<usize>,
    combine_send_counts: Vec<usize>,
    combine_recv_counts: Vec<usize>,
    assembled_output_shape: Vec<i64>,
    reference_output_shape: Vec<i64>,
    sparse_output_max_abs: f64,
    sparse_output_mean_abs: f64,
    train_initial_loss: f64,
    train_final_loss: f64,
    train_loss_improved: bool,
    expert_up_grad_norm: f64,
    expert_down_grad_norm: f64,
    checkpoint_manifest_output: String,
    ep_global_manifest_output: String,
    checkpoint_model_safetensors: String,
    checkpoint_optimizer_safetensors: String,
    checkpoint_tensor_count: usize,
    reload_expert_up_max_abs: f64,
    reload_expert_down_max_abs: f64,
    reload_optimizer_max_abs: f64,
    reload_loss: f64,
    reload_loss_delta: f64,
    continuous_second_loss: f64,
    resumed_second_loss: f64,
    second_step_delta: f64,
    second_step_expert_up_max_abs: f64,
    second_step_expert_down_max_abs: f64,
    second_step_optimizer_max_abs: f64,
    second_step_optimizer_step_delta: i64,
    resume_global_step: Option<u64>,
    resume_rank_manifest_output: Option<String>,
    resume_model_safetensors: Option<String>,
    resume_optimizer_safetensors: Option<String>,
    resume_sharded_manifest_tensor_count: Option<usize>,
    resume_reload_expert_up_max_abs: Option<f64>,
    resume_reload_expert_down_max_abs: Option<f64>,
    resume_reload_optimizer_max_abs: Option<f64>,
    resume_reload_loss_delta: Option<f64>,
    resume_next_step_delta: Option<f64>,
    resume_next_step_expert_up_max_abs: Option<f64>,
    resume_next_step_expert_down_max_abs: Option<f64>,
    resume_next_step_optimizer_max_abs: Option<f64>,
    resume_next_step_optimizer_step_delta: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExpertParallelCheckpointManifest {
    format: String,
    #[serde(default = "ep_rank_manifest_kind")]
    manifest_kind: String,
    rank: usize,
    world_size: usize,
    local_rank: usize,
    global_step: u64,
    owned_expert_start: usize,
    owned_expert_end: usize,
    model_safetensors: String,
    optimizer_safetensors: String,
    optimizer: String,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    shards: Vec<ExpertParallelCheckpointShard>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExpertParallelGlobalCheckpointManifest {
    format: String,
    manifest_kind: String,
    #[serde(default)]
    base_model_path: String,
    #[serde(default)]
    tokenizer_path: String,
    global_step: u64,
    consumed_samples: u64,
    consumed_tokens: u64,
    #[serde(default)]
    data_cursor_next: Option<usize>,
    #[serde(default)]
    data_epoch_next: Option<usize>,
    #[serde(default)]
    data_sample_offset_next: Option<usize>,
    #[serde(default)]
    data_train_samples: Option<usize>,
    #[serde(default)]
    dataset_source_files: Vec<String>,
    #[serde(default)]
    dataset_source_sample_counts: Vec<ExpertParallelDatasetSourceSampleCount>,
    #[serde(default)]
    dataset_fingerprint: String,
    #[serde(default = "ep_manifest_default_dataset_shuffle")]
    dataset_shuffle: bool,
    #[serde(default)]
    seed: u64,
    dtype: String,
    optimizer: String,
    scheduler: String,
    parallel: ExpertParallelGlobalParallelManifest,
    ranks: Vec<ExpertParallelCheckpointManifest>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExpertParallelDatasetSourceSampleCount {
    path: String,
    samples: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExpertParallelGlobalParallelManifest {
    data_parallel_size: usize,
    tensor_model_parallel_size: usize,
    pipeline_model_parallel_size: usize,
    expert_model_parallel_size: usize,
    context_parallel_size: usize,
}

fn ep_rank_manifest_kind() -> String {
    "rank".to_string()
}

fn ep_manifest_default_dataset_shuffle() -> bool {
    true
}

impl ExpertParallelGlobalCheckpointManifest {
    fn validate(&self) -> Result<()> {
        if self.format != "rustrain.ep_sharded.v1" {
            bail!("unsupported EP sharded checkpoint format {}", self.format);
        }
        if self.manifest_kind != "global" {
            bail!(
                "EP global checkpoint manifest_kind must be global, got {}",
                self.manifest_kind
            );
        }
        if self.global_step == 0 {
            bail!("EP global checkpoint global_step must be positive");
        }
        if self.consumed_samples == 0 || self.consumed_tokens == 0 {
            bail!("EP global checkpoint requires positive consumed samples/tokens");
        }
        if !self.dataset_fingerprint.is_empty() {
            if self.dataset_source_files.is_empty() {
                bail!("EP global checkpoint dataset_fingerprint requires dataset_source_files");
            }
            if self
                .dataset_source_files
                .iter()
                .any(|source| !source.ends_with(".jsonl"))
            {
                bail!("EP global checkpoint dataset_source_files must only contain JSONL paths");
            }
            if !self.dataset_source_sample_counts.is_empty() {
                let count_paths = self
                    .dataset_source_sample_counts
                    .iter()
                    .map(|entry| entry.path.as_str())
                    .collect::<Vec<_>>();
                let source_paths = self
                    .dataset_source_files
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                if count_paths != source_paths {
                    bail!(
                        "EP global checkpoint dataset_source_sample_counts must match dataset_source_files order"
                    );
                }
                if self
                    .dataset_source_sample_counts
                    .iter()
                    .any(|entry| entry.samples == 0)
                {
                    bail!("EP global checkpoint dataset source sample counts must be positive");
                }
            }
        }
        if let Some(train_samples) = self.data_train_samples {
            if train_samples == 0 {
                bail!("EP global checkpoint data_train_samples must be positive when set");
            }
        }
        if self.data_cursor_next.is_some()
            != (self.data_epoch_next.is_some() && self.data_sample_offset_next.is_some())
        {
            bail!(
                "EP global checkpoint data cursor, epoch, and sample offset must be present together"
            );
        }
        if let Some(cursor_next) = self.data_cursor_next {
            if cursor_next as u64 != self.consumed_samples {
                bail!(
                    "EP global checkpoint data_cursor_next {} must match consumed_samples {}",
                    cursor_next,
                    self.consumed_samples
                );
            }
        }
        if self.dtype.is_empty() || self.optimizer.is_empty() || self.scheduler.is_empty() {
            bail!("EP global checkpoint requires dtype, optimizer, and scheduler");
        }
        if self.parallel.expert_model_parallel_size == 0
            || self.parallel.data_parallel_size == 0
            || self.parallel.tensor_model_parallel_size == 0
            || self.parallel.pipeline_model_parallel_size == 0
            || self.parallel.context_parallel_size == 0
        {
            bail!("EP global checkpoint parallel sizes must be positive");
        }
        if self.parallel.data_parallel_size != 1
            || self.parallel.tensor_model_parallel_size != 1
            || self.parallel.pipeline_model_parallel_size != 1
            || self.parallel.context_parallel_size != 1
        {
            bail!(
                "EP global checkpoint currently expects DP/TP/PP/CP sizes to be 1, got DP={} TP={} PP={} CP={}",
                self.parallel.data_parallel_size,
                self.parallel.tensor_model_parallel_size,
                self.parallel.pipeline_model_parallel_size,
                self.parallel.context_parallel_size
            );
        }
        if self.ranks.len() != self.parallel.expert_model_parallel_size {
            bail!(
                "EP global checkpoint expected {} rank manifests, got {}",
                self.parallel.expert_model_parallel_size,
                self.ranks.len()
            );
        }
        let mut expected_start = 0;
        let mut ranks = self
            .ranks
            .iter()
            .map(|rank| {
                if rank.format != "rustrain.ep_sharded.v1" {
                    bail!(
                        "EP rank {} has unsupported checkpoint format {}",
                        rank.rank,
                        rank.format
                    );
                }
                if rank.manifest_kind != "rank" {
                    bail!(
                        "EP rank {} manifest_kind must be rank, got {}",
                        rank.rank,
                        rank.manifest_kind
                    );
                }
                if rank.world_size != self.parallel.expert_model_parallel_size {
                    bail!(
                        "EP rank {} world_size {} does not match global EP size {}",
                        rank.rank,
                        rank.world_size,
                        self.parallel.expert_model_parallel_size
                    );
                }
                if rank.global_step != self.global_step {
                    bail!(
                        "EP rank {} global_step {} does not match global step {}",
                        rank.rank,
                        rank.global_step,
                        self.global_step
                    );
                }
                if rank.optimizer != self.optimizer {
                    bail!(
                        "EP rank {} optimizer {} does not match global optimizer {}",
                        rank.rank,
                        rank.optimizer,
                        self.optimizer
                    );
                }
                if rank.shards.is_empty() {
                    bail!("EP rank {} checkpoint has no shards", rank.rank);
                }
                if rank.model_safetensors.is_empty() || rank.optimizer_safetensors.is_empty() {
                    bail!(
                        "EP rank {} checkpoint requires model and optimizer safetensors",
                        rank.rank
                    );
                }
                Ok(rank)
            })
            .collect::<Result<Vec<_>>>()?;
        ranks.sort_by_key(|rank| rank.rank);
        for (expected_rank, rank) in ranks.iter().enumerate() {
            if rank.rank != expected_rank || rank.local_rank != expected_rank {
                bail!(
                    "EP global checkpoint rank ordering mismatch: expected rank/local_rank {}, got {}/{}",
                    expected_rank,
                    rank.rank,
                    rank.local_rank
                );
            }
            if rank.owned_expert_start != expected_start {
                bail!(
                    "EP rank {} owned expert start {} does not continue from {}",
                    rank.rank,
                    rank.owned_expert_start,
                    expected_start
                );
            }
            if rank.owned_expert_end <= rank.owned_expert_start {
                bail!("EP rank {} owns an empty expert range", rank.rank);
            }
            expected_start = rank.owned_expert_end;
            for shard in &rank.shards {
                if shard.partition != "expert_model_parallel" {
                    bail!(
                        "EP rank {} shard {} has unexpected partition {}",
                        rank.rank,
                        shard.name,
                        shard.partition
                    );
                }
                if shard.global_shape.is_empty() || shard.shard_shape.is_empty() {
                    bail!(
                        "EP rank {} shard {} is missing shape metadata",
                        rank.rank,
                        shard.name
                    );
                }
                if shard.optimizer_m_name.is_empty() || shard.optimizer_v_name.is_empty() {
                    bail!(
                        "EP rank {} shard {} is missing optimizer slot metadata",
                        rank.rank,
                        shard.name
                    );
                }
                if shard.global_shape[0] != ep_expert_count() as i64 {
                    bail!(
                        "EP rank {} shard {} global expert dimension {} does not match {}",
                        rank.rank,
                        shard.name,
                        shard.global_shape[0],
                        ep_expert_count()
                    );
                }
                if shard.shard_shape[0] as usize != rank.owned_expert_end - rank.owned_expert_start
                {
                    bail!(
                        "EP rank {} shard {} shard expert dimension {} does not match owned expert count {}",
                        rank.rank,
                        shard.name,
                        shard.shard_shape[0],
                        rank.owned_expert_end - rank.owned_expert_start
                    );
                }
                if shard.dtype.is_empty() {
                    bail!(
                        "EP rank {} shard {} is missing dtype",
                        rank.rank,
                        shard.name
                    );
                }
            }
        }
        if expected_start != ep_expert_count() {
            bail!(
                "EP global checkpoint covers {} experts, expected {}",
                expected_start,
                ep_expert_count()
            );
        }
        Ok(())
    }

    fn validate_artifacts(&self) -> Result<()> {
        self.validate()?;
        for rank in &self.ranks {
            let model_tensors =
                read_tensor_map(Path::new(&rank.model_safetensors)).with_context(|| {
                    format!("failed to validate EP rank {} model artifacts", rank.rank)
                })?;
            let optimizer_tensors = read_tensor_map(Path::new(&rank.optimizer_safetensors))
                .with_context(|| {
                    format!(
                        "failed to validate EP rank {} optimizer artifacts",
                        rank.rank
                    )
                })?;
            for shard in &rank.shards {
                let model_tensor = tensor_from_map(&model_tensors, &shard.shard_name)
                    .with_context(|| {
                        format!(
                            "EP rank {} missing model shard {} for {}",
                            rank.rank, shard.shard_name, shard.name
                        )
                    })?;
                if model_tensor.size() != shard.shard_shape {
                    bail!(
                        "EP rank {} model shard {} shape {:?} does not match manifest shard_shape {:?}",
                        rank.rank,
                        shard.shard_name,
                        model_tensor.size(),
                        shard.shard_shape
                    );
                }
                let optimizer_m = tensor_from_map(&optimizer_tensors, &shard.optimizer_m_name)
                    .with_context(|| {
                        format!(
                            "EP rank {} missing optimizer m slot {} for {}",
                            rank.rank, shard.optimizer_m_name, shard.name
                        )
                    })?;
                let optimizer_v = tensor_from_map(&optimizer_tensors, &shard.optimizer_v_name)
                    .with_context(|| {
                        format!(
                            "EP rank {} missing optimizer v slot {} for {}",
                            rank.rank, shard.optimizer_v_name, shard.name
                        )
                    })?;
                if optimizer_m.size() != shard.shard_shape {
                    bail!(
                        "EP rank {} optimizer m slot {} shape {:?} does not match manifest shard_shape {:?}",
                        rank.rank,
                        shard.optimizer_m_name,
                        optimizer_m.size(),
                        shard.shard_shape
                    );
                }
                if optimizer_v.size() != shard.shard_shape {
                    bail!(
                        "EP rank {} optimizer v slot {} shape {:?} does not match manifest shard_shape {:?}",
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

#[derive(Debug, Serialize, Deserialize)]
struct ExpertParallelCheckpointShard {
    name: String,
    shard_name: String,
    optimizer_m_name: String,
    optimizer_v_name: String,
    global_shape: Vec<i64>,
    shard_shape: Vec<i64>,
    dtype: String,
    partition: String,
}

struct ExpertParallelTchMoeExternalResume {
    global_step: u64,
    rank_manifest_output: String,
    model_safetensors: String,
    optimizer_safetensors: String,
    sharded_manifest_tensor_count: usize,
    expert_up: Tensor,
    expert_down: Tensor,
    state: ExpertParallelTchMoeAdamState,
}

struct ExpertParallelAdamConfig {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

struct ExpertParallelAdamState {
    m: Tensor,
    v: Tensor,
    step: i64,
}

struct ExpertParallelTchMoeAdamState {
    up_m: Tensor,
    up_v: Tensor,
    down_m: Tensor,
    down_v: Tensor,
    step: i64,
}

struct ExpertParallelAdamStep {
    pre_loss: f64,
    post_loss: Option<f64>,
    updated_scales: Tensor,
    updated_state: ExpertParallelAdamState,
    grad_norm: f64,
    reduced_output_shape: Vec<i64>,
    combine_max_abs: f64,
    combine_mean_abs: f64,
}

struct ExpertParallelSparseSgdStep {
    initial_loss: f64,
    final_loss: Option<f64>,
    updated_scales: Tensor,
    updated_state: ExpertParallelAdamState,
    grad_norm: f64,
}

struct ExpertParallelTchMoeAdamStep {
    initial_loss: f64,
    final_loss: Option<f64>,
    updated_expert_up: Tensor,
    updated_expert_down: Tensor,
    updated_state: ExpertParallelTchMoeAdamState,
    up_grad_norm: f64,
    down_grad_norm: f64,
}

struct ExpertParallelCheckpointWrite {
    manifest_path: PathBuf,
    model_safetensors: PathBuf,
    optimizer_safetensors: PathBuf,
    tensor_count: usize,
}

struct ExpertParallelSparsePeerPlan {
    peer: usize,
    token_indices: Vec<usize>,
}

struct ExpertParallelSparseForward {
    assembled_output: Tensor,
    source_token_indices: Vec<usize>,
    owned_token_indices: Vec<usize>,
    dispatch_send_counts: Vec<usize>,
    dispatch_recv_counts: Vec<usize>,
    combine_send_counts: Vec<usize>,
    combine_recv_counts: Vec<usize>,
}

struct ExpertParallelTchMoeCheckpointRead {
    expert_up: Tensor,
    expert_down: Tensor,
    state: ExpertParallelTchMoeAdamState,
}
pub fn run_expert_parallel_rank(output_dir: PathBuf) -> Result<()> {
    let rank = parse_launcher_usize_env("RANK")?;
    let local_rank = parse_launcher_usize_env("LOCAL_RANK")?;
    let world_size = parse_launcher_usize_env("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("EP rank-local currently expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let tokens = ep_tokens();
    let router = ep_router();
    let expert_scales = ep_expert_scales();
    let assignments = route_top1(&tokens, &router);
    let experts_per_rank = expert_scales.len() / world_size;
    let owned_expert_start = rank * experts_per_rank;
    let owned_expert_end = owned_expert_start + experts_per_rank;
    let mut local_output = Array2::<f64>::zeros(tokens.dim());
    let mut owned_token_indices = Vec::new();
    let mut expert_load = vec![0usize; expert_scales.len()];
    for (token_index, expert_index) in assignments.iter().copied().enumerate() {
        if !(owned_expert_start..owned_expert_end).contains(&expert_index) {
            continue;
        }
        owned_token_indices.push(token_index);
        expert_load[expert_index] += 1;
        for hidden_index in 0..tokens.ncols() {
            local_output[[token_index, hidden_index]] =
                tokens[[token_index, hidden_index]] * expert_scales[expert_index][hidden_index];
        }
    }

    let local_output_path = output_dir.join(format!("ep-rank-{rank}-output.json"));
    fs::write(
        &local_output_path,
        serde_json::to_string_pretty(&array2_to_rows(&local_output))? + "\n",
    )
    .with_context(|| format!("failed to write {}", local_output_path.display()))?;

    let summary = ExpertParallelRankSummary {
        rank,
        world_size,
        local_rank,
        owned_expert_start,
        owned_expert_end,
        owned_token_indices,
        expert_load,
        local_output_path: local_output_path.display().to_string(),
    };
    let summary_path = output_dir.join(format!("ep-rank-{rank}.json"));
    fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary)? + "\n",
    )
    .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

pub fn run_expert_parallel_nccl_rank(output_dir: PathBuf) -> Result<()> {
    let rank = parse_launcher_usize_env("RANK")?;
    let local_rank = parse_launcher_usize_env("LOCAL_RANK")?;
    let world_size = parse_launcher_usize_env("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("EP NCCL rank currently expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let device = Device::Cuda(local_rank);
    let tokens = ep_tokens_tensor(device);
    let router = ep_router_tensor(device);
    let assignments = route_top1_tensor(&tokens, &router)?;
    let reference = ep_reference_output_tensor(&tokens, &assignments)?;
    let target = &reference * 0.5;
    let experts_per_rank = ep_expert_count() / world_size;
    let owned_expert_start = rank * experts_per_rank;
    let owned_expert_end = owned_expert_start + experts_per_rank;
    let owned_token_indices = assignments
        .iter()
        .enumerate()
        .filter_map(|(token_index, expert_index)| {
            (owned_expert_start..owned_expert_end)
                .contains(expert_index)
                .then_some(token_index)
        })
        .collect::<Vec<_>>();
    let expert_load = ep_owned_expert_load(&assignments, owned_expert_start, owned_expert_end);

    let local_scales = ep_owned_expert_scales_tensor(owned_expert_start, owned_expert_end, device);
    let adam = ExpertParallelAdamConfig {
        learning_rate: 0.25,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let initial_state = ExpertParallelAdamState {
        m: Tensor::zeros_like(&local_scales),
        v: Tensor::zeros_like(&local_scales),
        step: 0,
    };
    let first_step = ep_nccl_adam_step(
        &output_dir.join("ep-nccl-combine"),
        Some(&output_dir.join("ep-nccl-updated-combine")),
        &tokens,
        &assignments,
        owned_expert_start,
        owned_expert_end,
        &local_scales,
        &initial_state,
        &target,
        Some(&reference),
        &adam,
    )?;
    if first_step.combine_max_abs > 1e-6 {
        bail!(
            "EP NCCL combine mismatch: rank={rank}, max_abs={}, mean_abs={}",
            first_step.combine_max_abs,
            first_step.combine_mean_abs
        );
    }

    if first_step.grad_norm <= 0.0 {
        bail!("EP NCCL local expert scale gradient is missing or zero on rank {rank}");
    }

    let final_loss = first_step
        .post_loss
        .ok_or_else(|| anyhow!("EP NCCL first step did not compute post-update loss"))?;
    let train_loss_improved = final_loss < first_step.pre_loss;
    if !train_loss_improved {
        bail!(
            "EP NCCL train did not lower loss on rank {rank}: initial={}, final={final_loss}",
            first_step.pre_loss
        );
    }

    let checkpoint = write_ep_rank_checkpoint(
        &output_dir,
        rank,
        world_size,
        local_rank,
        owned_expert_start,
        owned_expert_end,
        &first_step.updated_scales,
        &first_step.updated_state,
        &adam,
    )?;
    let (reloaded_scales, reloaded_state) = read_ep_rank_checkpoint(&checkpoint)?;
    let reloaded_scales = reloaded_scales.to_device(device);
    let reloaded_state = ExpertParallelAdamState {
        m: reloaded_state.m.to_device(device),
        v: reloaded_state.v.to_device(device),
        step: reloaded_state.step,
    };
    let reload_scale_max_abs = tensor_max_abs_diff(&reloaded_scales, &first_step.updated_scales)?;
    let reload_optimizer_max_abs =
        tensor_max_abs_diff(&reloaded_state.m, &first_step.updated_state.m)?.max(
            tensor_max_abs_diff(&reloaded_state.v, &first_step.updated_state.v)?,
        );
    if reload_scale_max_abs > 1e-7 || reload_optimizer_max_abs > 1e-7 {
        bail!(
            "EP checkpoint reload mismatch on rank {rank}: scale={reload_scale_max_abs}, optimizer={reload_optimizer_max_abs}"
        );
    }

    let continuous_second = ep_nccl_adam_step(
        &output_dir.join("ep-nccl-continuous-second"),
        None,
        &tokens,
        &assignments,
        owned_expert_start,
        owned_expert_end,
        &first_step.updated_scales,
        &first_step.updated_state,
        &target,
        None,
        &adam,
    )?;
    let resumed_second = ep_nccl_adam_step(
        &output_dir.join("ep-nccl-resumed-second"),
        None,
        &tokens,
        &assignments,
        owned_expert_start,
        owned_expert_end,
        &reloaded_scales,
        &reloaded_state,
        &target,
        None,
        &adam,
    )?;
    let second_step_delta = (continuous_second.pre_loss - resumed_second.pre_loss).abs();
    let second_step_scale_max_abs = tensor_max_abs_diff(
        &continuous_second.updated_scales,
        &resumed_second.updated_scales,
    )?;
    let second_step_optimizer_max_abs = tensor_max_abs_diff(
        &continuous_second.updated_state.m,
        &resumed_second.updated_state.m,
    )?
    .max(tensor_max_abs_diff(
        &continuous_second.updated_state.v,
        &resumed_second.updated_state.v,
    )?);
    if second_step_delta > 1e-6
        || second_step_scale_max_abs > 1e-6
        || second_step_optimizer_max_abs > 1e-6
    {
        bail!(
            "EP checkpoint next-step parity failed on rank {rank}: loss_delta={second_step_delta}, scale_delta={second_step_scale_max_abs}, optimizer_delta={second_step_optimizer_max_abs}"
        );
    }

    let summary = ExpertParallelNcclRankSummary {
        rank,
        world_size,
        local_rank,
        owned_expert_start,
        owned_expert_end,
        owned_token_indices,
        expert_load,
        reduced_output_shape: first_step.reduced_output_shape,
        combine_max_abs: first_step.combine_max_abs,
        combine_mean_abs: first_step.combine_mean_abs,
        train_initial_loss: first_step.pre_loss,
        train_final_loss: final_loss,
        train_loss_improved,
        scale_grad_norm: first_step.grad_norm,
        checkpoint_manifest_output: checkpoint.manifest_path.display().to_string(),
        checkpoint_model_safetensors: checkpoint.model_safetensors.display().to_string(),
        checkpoint_optimizer_safetensors: checkpoint.optimizer_safetensors.display().to_string(),
        checkpoint_tensor_count: checkpoint.tensor_count,
        reload_scale_max_abs,
        reload_optimizer_max_abs,
        continuous_second_loss: continuous_second.pre_loss,
        resumed_second_loss: resumed_second.pre_loss,
        second_step_delta,
        second_step_scale_max_abs,
        second_step_optimizer_max_abs,
    };
    let summary_path = output_dir.join(format!("ep-nccl-rank-{rank}.json"));
    fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary)? + "\n",
    )
    .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

pub fn run_expert_parallel_sparse_rank(output_dir: PathBuf) -> Result<()> {
    let rank = parse_launcher_usize_env("RANK")?;
    let local_rank = parse_launcher_usize_env("LOCAL_RANK")?;
    let world_size = parse_launcher_usize_env("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("EP sparse rank currently expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let device = Device::Cuda(local_rank);
    let tokens = ep_tokens_tensor(device);
    let router = ep_router_tensor(device);
    let assignments = route_top1_tensor(&tokens, &router)?;
    let global_expert_load = ep_global_expert_load(&assignments);
    let load_balance_loss = ep_load_balance_loss(&global_expert_load);
    let experts_per_rank = ep_expert_count() / world_size;
    let owned_expert_start = rank * experts_per_rank;
    let owned_expert_end = owned_expert_start + experts_per_rank;
    let local_scales = ep_owned_expert_scales_tensor(owned_expert_start, owned_expert_end, device);

    let forward = ep_sparse_forward(
        &output_dir,
        "ep-sparse-forward",
        rank,
        world_size,
        &tokens,
        &assignments,
        owned_expert_start,
        owned_expert_end,
        &local_scales,
    )?;
    let reference = ep_reference_output_tensor(&tokens, &assignments)?;
    let reference_rows = ep_sparse_pack_token_rows(&reference, &forward.source_token_indices);
    let sparse_diff = tensor_diff_stats(&forward.assembled_output, &reference_rows)?;
    if sparse_diff.0 > 1e-6 {
        bail!(
            "EP sparse dispatch/combine mismatch: rank={rank}, max_abs={}, mean_abs={}",
            sparse_diff.0,
            sparse_diff.1
        );
    }

    let target_rows = &reference_rows * 0.5;
    let first_step = ep_sparse_sgd_step(
        &output_dir,
        "ep-sparse-first-step",
        rank,
        world_size,
        &tokens,
        &assignments,
        owned_expert_start,
        owned_expert_end,
        &local_scales,
        &ExpertParallelAdamState {
            m: Tensor::zeros_like(&local_scales),
            v: Tensor::zeros_like(&local_scales),
            step: 0,
        },
        &target_rows,
        Some("ep-sparse-updated-forward"),
    )?;
    let train_final_loss = first_step
        .final_loss
        .ok_or_else(|| anyhow!("EP sparse first step did not compute post-update loss"))?;
    let train_loss_improved = train_final_loss < first_step.initial_loss;
    if !train_loss_improved {
        bail!(
            "EP sparse train did not lower loss on rank {rank}: initial={}, final={train_final_loss}",
            first_step.initial_loss
        );
    }
    let sparse_adam = ExpertParallelAdamConfig {
        learning_rate: 0.25,
        beta1: 0.0,
        beta2: 0.0,
        eps: 0.0,
        weight_decay: 0.0,
    };
    let checkpoint = write_ep_rank_checkpoint(
        &output_dir,
        rank,
        world_size,
        local_rank,
        owned_expert_start,
        owned_expert_end,
        &first_step.updated_scales,
        &first_step.updated_state,
        &sparse_adam,
    )?;
    let (reloaded_scales, reloaded_state) = read_ep_rank_checkpoint(&checkpoint)?;
    let reloaded_scales = reloaded_scales.to_device(device);
    let reloaded_state = ExpertParallelAdamState {
        m: reloaded_state.m.to_device(device),
        v: reloaded_state.v.to_device(device),
        step: reloaded_state.step,
    };
    let reload_scale_max_abs = tensor_max_abs_diff(&reloaded_scales, &first_step.updated_scales)?;
    let reload_optimizer_max_abs =
        tensor_max_abs_diff(&reloaded_state.m, &first_step.updated_state.m)?.max(
            tensor_max_abs_diff(&reloaded_state.v, &first_step.updated_state.v)?,
        );
    if reload_scale_max_abs > 1e-7 {
        bail!("EP sparse checkpoint reload mismatch on rank {rank}: scale={reload_scale_max_abs}");
    }
    let reloaded_forward = ep_sparse_forward(
        &output_dir,
        "ep-sparse-reloaded-forward",
        rank,
        world_size,
        &tokens,
        &assignments,
        owned_expert_start,
        owned_expert_end,
        &reloaded_scales,
    )?;
    let reload_loss = (&reloaded_forward.assembled_output - &target_rows)
        .square()
        .mean(Kind::Float)
        .double_value(&[]);
    let reload_loss_delta = (reload_loss - train_final_loss).abs();
    if reload_loss_delta > 1e-7 {
        bail!(
            "EP sparse checkpoint reload loss mismatch on rank {rank}: reload_loss={reload_loss}, train_final_loss={train_final_loss}, delta={reload_loss_delta}"
        );
    }
    if reload_optimizer_max_abs > 1e-7 {
        bail!(
            "EP sparse checkpoint optimizer reload mismatch on rank {rank}: optimizer={reload_optimizer_max_abs}"
        );
    }
    let continuous_second = ep_sparse_sgd_step(
        &output_dir,
        "ep-sparse-continuous-second",
        rank,
        world_size,
        &tokens,
        &assignments,
        owned_expert_start,
        owned_expert_end,
        &first_step.updated_scales,
        &first_step.updated_state,
        &target_rows,
        None,
    )?;
    let resumed_second = ep_sparse_sgd_step(
        &output_dir,
        "ep-sparse-resumed-second",
        rank,
        world_size,
        &tokens,
        &assignments,
        owned_expert_start,
        owned_expert_end,
        &reloaded_scales,
        &reloaded_state,
        &target_rows,
        None,
    )?;
    let second_step_delta = (continuous_second.initial_loss - resumed_second.initial_loss).abs();
    let second_step_scale_max_abs = tensor_max_abs_diff(
        &continuous_second.updated_scales,
        &resumed_second.updated_scales,
    )?;
    let second_step_optimizer_max_abs = tensor_max_abs_diff(
        &continuous_second.updated_state.m,
        &resumed_second.updated_state.m,
    )?
    .max(tensor_max_abs_diff(
        &continuous_second.updated_state.v,
        &resumed_second.updated_state.v,
    )?);
    let second_step_optimizer_step_delta =
        (continuous_second.updated_state.step - resumed_second.updated_state.step).abs();
    if second_step_delta > 1e-7
        || second_step_scale_max_abs > 1e-7
        || second_step_optimizer_max_abs > 1e-7
        || second_step_optimizer_step_delta != 0
    {
        bail!(
            "EP sparse checkpoint next-step parity failed on rank {rank}: loss_delta={second_step_delta}, scale_delta={second_step_scale_max_abs}, optimizer_delta={second_step_optimizer_max_abs}, optimizer_step_delta={second_step_optimizer_step_delta}"
        );
    }

    let summary = ExpertParallelSparseRankSummary {
        rank,
        world_size,
        local_rank,
        source_token_indices: forward.source_token_indices,
        owned_expert_start,
        owned_expert_end,
        owned_token_indices: forward.owned_token_indices,
        global_expert_load,
        load_balance_loss,
        dispatch_send_counts: forward.dispatch_send_counts,
        dispatch_recv_counts: forward.dispatch_recv_counts,
        combine_send_counts: forward.combine_send_counts,
        combine_recv_counts: forward.combine_recv_counts,
        assembled_output_shape: forward.assembled_output.size(),
        reference_output_shape: reference_rows.size(),
        sparse_output_max_abs: sparse_diff.0,
        sparse_output_mean_abs: sparse_diff.1,
        train_initial_loss: first_step.initial_loss,
        train_final_loss,
        train_loss_improved,
        scale_grad_norm: first_step.grad_norm,
        checkpoint_manifest_output: checkpoint.manifest_path.display().to_string(),
        checkpoint_model_safetensors: checkpoint.model_safetensors.display().to_string(),
        checkpoint_optimizer_safetensors: checkpoint.optimizer_safetensors.display().to_string(),
        checkpoint_tensor_count: checkpoint.tensor_count,
        reload_scale_max_abs,
        reload_optimizer_max_abs,
        reload_loss,
        reload_loss_delta,
        continuous_second_loss: continuous_second.initial_loss,
        resumed_second_loss: resumed_second.initial_loss,
        second_step_delta,
        second_step_scale_max_abs,
        second_step_optimizer_max_abs,
        second_step_optimizer_step_delta,
    };
    let summary_path = output_dir.join(format!("ep-sparse-rank-{rank}.json"));
    fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary)? + "\n",
    )
    .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

pub fn run_expert_parallel_tch_moe_rank(
    output_dir: PathBuf,
    resume_from: Option<&Path>,
) -> Result<()> {
    let rank = parse_launcher_usize_env("RANK")?;
    let local_rank = parse_launcher_usize_env("LOCAL_RANK")?;
    let world_size = parse_launcher_usize_env("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("EP tch MoE rank currently expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let device = Device::Cuda(local_rank);
    let tokens = ep_tokens_tensor(device);
    let router = ep_router_tensor(device);
    let assignments = route_top1_tensor(&tokens, &router)?;
    let global_expert_load = ep_global_expert_load(&assignments);
    let load_balance_loss = ep_load_balance_loss(&global_expert_load);
    let experts_per_rank = ep_expert_count() / world_size;
    let owned_expert_start = rank * experts_per_rank;
    let owned_expert_end = owned_expert_start + experts_per_rank;
    let local_up = ep_tch_moe_owned_expert_up_tensor(owned_expert_start, owned_expert_end, device);
    let local_down =
        ep_tch_moe_owned_expert_down_tensor(owned_expert_start, owned_expert_end, device);

    let forward = ep_tch_moe_sparse_forward(
        &output_dir,
        "ep-tch-moe-forward",
        rank,
        world_size,
        &tokens,
        &assignments,
        owned_expert_start,
        owned_expert_end,
        &local_up,
        &local_down,
    )?;
    let reference = ep_tch_moe_reference_output_tensor(&tokens, &assignments)?;
    let reference_rows = ep_sparse_pack_token_rows(&reference, &forward.source_token_indices);
    let sparse_diff = tensor_diff_stats(&forward.assembled_output, &reference_rows)?;
    if sparse_diff.0 > 1e-6 {
        bail!(
            "EP tch MoE sparse dispatch/combine mismatch: rank={rank}, max_abs={}, mean_abs={}",
            sparse_diff.0,
            sparse_diff.1
        );
    }

    let target_rows = &reference_rows * 0.25;
    let adam = ExpertParallelAdamConfig {
        learning_rate: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let initial_state = ExpertParallelTchMoeAdamState {
        up_m: Tensor::zeros_like(&local_up),
        up_v: Tensor::zeros_like(&local_up),
        down_m: Tensor::zeros_like(&local_down),
        down_v: Tensor::zeros_like(&local_down),
        step: 0,
    };
    let first_step = ep_tch_moe_adam_step(
        &output_dir,
        "ep-tch-moe-first-step",
        rank,
        world_size,
        &tokens,
        &assignments,
        owned_expert_start,
        owned_expert_end,
        &local_up,
        &local_down,
        &initial_state,
        &target_rows,
        &adam,
        Some("ep-tch-moe-updated-forward"),
    )?;
    let train_final_loss = first_step
        .final_loss
        .ok_or_else(|| anyhow!("EP tch MoE first step did not compute post-update loss"))?;
    let train_loss_improved = train_final_loss < first_step.initial_loss;
    if !train_loss_improved {
        bail!(
            "EP tch MoE train did not lower loss on rank {rank}: initial={}, final={train_final_loss}",
            first_step.initial_loss
        );
    }

    let checkpoint = write_ep_tch_moe_rank_checkpoint(
        &output_dir,
        rank,
        world_size,
        local_rank,
        owned_expert_start,
        owned_expert_end,
        &first_step.updated_expert_up,
        &first_step.updated_expert_down,
        &first_step.updated_state,
        &adam,
    )?;
    let ep_global_manifest_output =
        write_ep_tch_moe_global_checkpoint_manifest(&output_dir, rank, world_size, &checkpoint)?;
    let reloaded = read_ep_tch_moe_rank_checkpoint(&checkpoint)?;
    let reloaded_up = reloaded.expert_up.to_device(device);
    let reloaded_down = reloaded.expert_down.to_device(device);
    let reloaded_state = ExpertParallelTchMoeAdamState {
        up_m: reloaded.state.up_m.to_device(device),
        up_v: reloaded.state.up_v.to_device(device),
        down_m: reloaded.state.down_m.to_device(device),
        down_v: reloaded.state.down_v.to_device(device),
        step: reloaded.state.step,
    };
    let reload_expert_up_max_abs =
        tensor_max_abs_diff(&reloaded_up, &first_step.updated_expert_up)?;
    let reload_expert_down_max_abs =
        tensor_max_abs_diff(&reloaded_down, &first_step.updated_expert_down)?;
    let reload_optimizer_max_abs =
        ep_tch_moe_adam_state_max_abs_diff(&reloaded_state, &first_step.updated_state)?;
    if reload_expert_up_max_abs > 1e-7
        || reload_expert_down_max_abs > 1e-7
        || reload_optimizer_max_abs > 1e-7
    {
        bail!(
            "EP tch MoE checkpoint reload mismatch on rank {rank}: up={reload_expert_up_max_abs}, down={reload_expert_down_max_abs}, optimizer={reload_optimizer_max_abs}"
        );
    }
    let reloaded_forward = ep_tch_moe_sparse_forward(
        &output_dir,
        "ep-tch-moe-reloaded-forward",
        rank,
        world_size,
        &tokens,
        &assignments,
        owned_expert_start,
        owned_expert_end,
        &reloaded_up,
        &reloaded_down,
    )?;
    let reload_loss = (&reloaded_forward.assembled_output - &target_rows)
        .square()
        .mean(Kind::Float)
        .double_value(&[]);
    let reload_loss_delta = (reload_loss - train_final_loss).abs();
    if reload_loss_delta > 1e-7 {
        bail!(
            "EP tch MoE checkpoint reload loss mismatch on rank {rank}: reload_loss={reload_loss}, train_final_loss={train_final_loss}, delta={reload_loss_delta}"
        );
    }

    let continuous_second = ep_tch_moe_adam_step(
        &output_dir,
        "ep-tch-moe-continuous-second",
        rank,
        world_size,
        &tokens,
        &assignments,
        owned_expert_start,
        owned_expert_end,
        &first_step.updated_expert_up,
        &first_step.updated_expert_down,
        &first_step.updated_state,
        &target_rows,
        &adam,
        None,
    )?;
    let resumed_second = ep_tch_moe_adam_step(
        &output_dir,
        "ep-tch-moe-resumed-second",
        rank,
        world_size,
        &tokens,
        &assignments,
        owned_expert_start,
        owned_expert_end,
        &reloaded_up,
        &reloaded_down,
        &reloaded_state,
        &target_rows,
        &adam,
        None,
    )?;
    let second_step_delta = (continuous_second.initial_loss - resumed_second.initial_loss).abs();
    let second_step_expert_up_max_abs = tensor_max_abs_diff(
        &continuous_second.updated_expert_up,
        &resumed_second.updated_expert_up,
    )?;
    let second_step_expert_down_max_abs = tensor_max_abs_diff(
        &continuous_second.updated_expert_down,
        &resumed_second.updated_expert_down,
    )?;
    let second_step_optimizer_max_abs = ep_tch_moe_adam_state_max_abs_diff(
        &continuous_second.updated_state,
        &resumed_second.updated_state,
    )?;
    let second_step_optimizer_step_delta =
        (continuous_second.updated_state.step - resumed_second.updated_state.step).abs();
    if second_step_delta > 1e-7
        || second_step_expert_up_max_abs > 1e-7
        || second_step_expert_down_max_abs > 1e-7
        || second_step_optimizer_max_abs > 1e-7
        || second_step_optimizer_step_delta != 0
    {
        bail!(
            "EP tch MoE checkpoint next-step parity failed on rank {rank}: loss_delta={second_step_delta}, up_delta={second_step_expert_up_max_abs}, down_delta={second_step_expert_down_max_abs}, optimizer_delta={second_step_optimizer_max_abs}, optimizer_step_delta={second_step_optimizer_step_delta}"
        );
    }

    let external_resume = resume_from
        .map(|resume_from| {
            let resumed =
                read_ep_tch_moe_rank_from_global_manifest(resume_from, rank, world_size)?;
            let resume_up = resumed.expert_up.to_device(device);
            let resume_down = resumed.expert_down.to_device(device);
            let resume_state = ExpertParallelTchMoeAdamState {
                up_m: resumed.state.up_m.to_device(device),
                up_v: resumed.state.up_v.to_device(device),
                down_m: resumed.state.down_m.to_device(device),
                down_v: resumed.state.down_v.to_device(device),
                step: resumed.state.step,
            };
            let resume_reload_expert_up_max_abs =
                tensor_max_abs_diff(&resume_up, &first_step.updated_expert_up)?;
            let resume_reload_expert_down_max_abs =
                tensor_max_abs_diff(&resume_down, &first_step.updated_expert_down)?;
            let resume_reload_optimizer_max_abs =
                ep_tch_moe_adam_state_max_abs_diff(&resume_state, &first_step.updated_state)?;
            let resume_forward = ep_tch_moe_sparse_forward(
                &output_dir,
                "ep-tch-moe-external-resume-forward",
                rank,
                world_size,
                &tokens,
                &assignments,
                owned_expert_start,
                owned_expert_end,
                &resume_up,
                &resume_down,
            )?;
            let resume_reload_loss = (&resume_forward.assembled_output - &target_rows)
                .square()
                .mean(Kind::Float)
                .double_value(&[]);
            let resume_reload_loss_delta = (resume_reload_loss - train_final_loss).abs();
            let external_resumed_second = ep_tch_moe_adam_step(
                &output_dir,
                "ep-tch-moe-external-resume-second",
                rank,
                world_size,
                &tokens,
                &assignments,
                owned_expert_start,
                owned_expert_end,
                &resume_up,
                &resume_down,
                &resume_state,
                &target_rows,
                &adam,
                None,
            )?;
            let resume_next_step_delta =
                (continuous_second.initial_loss - external_resumed_second.initial_loss).abs();
            let resume_next_step_expert_up_max_abs = tensor_max_abs_diff(
                &continuous_second.updated_expert_up,
                &external_resumed_second.updated_expert_up,
            )?;
            let resume_next_step_expert_down_max_abs = tensor_max_abs_diff(
                &continuous_second.updated_expert_down,
                &external_resumed_second.updated_expert_down,
            )?;
            let resume_next_step_optimizer_max_abs = ep_tch_moe_adam_state_max_abs_diff(
                &continuous_second.updated_state,
                &external_resumed_second.updated_state,
            )?;
            let resume_next_step_optimizer_step_delta = (continuous_second.updated_state.step
                - external_resumed_second.updated_state.step)
                .abs();
            if resume_reload_expert_up_max_abs > 1e-7
                || resume_reload_expert_down_max_abs > 1e-7
                || resume_reload_optimizer_max_abs > 1e-7
                || resume_reload_loss_delta > 1e-7
                || resume_next_step_delta > 1e-7
                || resume_next_step_expert_up_max_abs > 1e-7
                || resume_next_step_expert_down_max_abs > 1e-7
                || resume_next_step_optimizer_max_abs > 1e-7
                || resume_next_step_optimizer_step_delta != 0
            {
                bail!(
                    "EP tch MoE external resume parity failed on rank {rank}: reload_up={resume_reload_expert_up_max_abs}, reload_down={resume_reload_expert_down_max_abs}, reload_optimizer={resume_reload_optimizer_max_abs}, reload_loss_delta={resume_reload_loss_delta}, next_loss_delta={resume_next_step_delta}, next_up_delta={resume_next_step_expert_up_max_abs}, next_down_delta={resume_next_step_expert_down_max_abs}, next_optimizer_delta={resume_next_step_optimizer_max_abs}, next_step_delta={resume_next_step_optimizer_step_delta}"
                );
            }
            Ok::<_, anyhow::Error>((
                resumed,
                resume_reload_expert_up_max_abs,
                resume_reload_expert_down_max_abs,
                resume_reload_optimizer_max_abs,
                resume_reload_loss_delta,
                resume_next_step_delta,
                resume_next_step_expert_up_max_abs,
                resume_next_step_expert_down_max_abs,
                resume_next_step_optimizer_max_abs,
                resume_next_step_optimizer_step_delta,
            ))
        })
        .transpose()?;

    let summary = ExpertParallelTchMoeRankSummary {
        rank,
        world_size,
        local_rank,
        resume_from: resume_from.map(|path| path.display().to_string()),
        resumed_sharded_checkpoint: external_resume.is_some(),
        source_token_indices: forward.source_token_indices,
        owned_expert_start,
        owned_expert_end,
        owned_token_indices: forward.owned_token_indices,
        global_expert_load,
        load_balance_loss,
        dispatch_send_counts: forward.dispatch_send_counts,
        dispatch_recv_counts: forward.dispatch_recv_counts,
        combine_send_counts: forward.combine_send_counts,
        combine_recv_counts: forward.combine_recv_counts,
        assembled_output_shape: forward.assembled_output.size(),
        reference_output_shape: reference_rows.size(),
        sparse_output_max_abs: sparse_diff.0,
        sparse_output_mean_abs: sparse_diff.1,
        train_initial_loss: first_step.initial_loss,
        train_final_loss,
        train_loss_improved,
        expert_up_grad_norm: first_step.up_grad_norm,
        expert_down_grad_norm: first_step.down_grad_norm,
        checkpoint_manifest_output: checkpoint.manifest_path.display().to_string(),
        ep_global_manifest_output: ep_global_manifest_output.display().to_string(),
        checkpoint_model_safetensors: checkpoint.model_safetensors.display().to_string(),
        checkpoint_optimizer_safetensors: checkpoint.optimizer_safetensors.display().to_string(),
        checkpoint_tensor_count: checkpoint.tensor_count,
        reload_expert_up_max_abs,
        reload_expert_down_max_abs,
        reload_optimizer_max_abs,
        reload_loss,
        reload_loss_delta,
        continuous_second_loss: continuous_second.initial_loss,
        resumed_second_loss: resumed_second.initial_loss,
        second_step_delta,
        second_step_expert_up_max_abs,
        second_step_expert_down_max_abs,
        second_step_optimizer_max_abs,
        second_step_optimizer_step_delta,
        resume_global_step: external_resume
            .as_ref()
            .map(|(resume, ..)| resume.global_step),
        resume_rank_manifest_output: external_resume
            .as_ref()
            .map(|(resume, ..)| resume.rank_manifest_output.clone()),
        resume_model_safetensors: external_resume
            .as_ref()
            .map(|(resume, ..)| resume.model_safetensors.clone()),
        resume_optimizer_safetensors: external_resume
            .as_ref()
            .map(|(resume, ..)| resume.optimizer_safetensors.clone()),
        resume_sharded_manifest_tensor_count: external_resume
            .as_ref()
            .map(|(resume, ..)| resume.sharded_manifest_tensor_count),
        resume_reload_expert_up_max_abs: external_resume.as_ref().map(|(_, value, ..)| *value),
        resume_reload_expert_down_max_abs: external_resume.as_ref().map(|(_, _, value, ..)| *value),
        resume_reload_optimizer_max_abs: external_resume
            .as_ref()
            .map(|(_, _, _, value, ..)| *value),
        resume_reload_loss_delta: external_resume
            .as_ref()
            .map(|(_, _, _, _, value, ..)| *value),
        resume_next_step_delta: external_resume
            .as_ref()
            .map(|(_, _, _, _, _, value, ..)| *value),
        resume_next_step_expert_up_max_abs: external_resume
            .as_ref()
            .map(|(_, _, _, _, _, _, value, ..)| *value),
        resume_next_step_expert_down_max_abs: external_resume
            .as_ref()
            .map(|(_, _, _, _, _, _, _, value, ..)| *value),
        resume_next_step_optimizer_max_abs: external_resume
            .as_ref()
            .map(|(_, _, _, _, _, _, _, _, value, ..)| *value),
        resume_next_step_optimizer_step_delta: external_resume
            .as_ref()
            .map(|(_, _, _, _, _, _, _, _, _, value)| *value),
    };
    let summary_path = output_dir.join(format!("ep-tch-moe-rank-{rank}.json"));
    fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary)? + "\n",
    )
    .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn ep_tokens() -> Array2<f64> {
    array![
        [1.0_f64, 0.0, 0.5],
        [0.0, 1.0, -0.5],
        [1.0, 1.0, 0.0],
        [-1.0, 0.5, 1.0],
    ]
}

fn ep_router() -> Array2<f64> {
    array![
        [0.9_f64, -0.2, 0.1, 0.0],
        [0.1, 0.8, -0.4, 0.2],
        [0.0, -0.3, 0.7, 0.6],
    ]
}

fn ep_expert_scales() -> [[f64; 3]; 4] {
    [
        [1.0_f64, 0.5, -0.25],
        [-0.5, 1.5, 0.25],
        [0.25, -1.0, 1.25],
        [1.2, 0.3, 0.8],
    ]
}

fn ep_expert_count() -> usize {
    ep_expert_scales().len()
}

fn array2_to_rows(array: &Array2<f64>) -> Vec<Vec<f64>> {
    array.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn ep_tokens_tensor(device: Device) -> Tensor {
    Tensor::from_slice(&[
        1.0_f32, 0.0, 0.5, 0.0, 1.0, -0.5, 1.0, 1.0, 0.0, -1.0, 0.5, 1.0,
    ])
    .reshape([4, 3])
    .to_device(device)
}

fn ep_router_tensor(device: Device) -> Tensor {
    Tensor::from_slice(&[
        0.9_f32, -0.2, 0.1, 0.0, 0.1, 0.8, -0.4, 0.2, 0.0, -0.3, 0.7, 0.6,
    ])
    .reshape([3, 4])
    .to_device(device)
}

fn ep_all_expert_scales_tensor(device: Device) -> Tensor {
    Tensor::from_slice(&[
        1.0_f32, 0.5, -0.25, -0.5, 1.5, 0.25, 0.25, -1.0, 1.25, 1.2, 0.3, 0.8,
    ])
    .reshape([4, 3])
    .to_device(device)
}

fn ep_owned_expert_scales_tensor(start: usize, end: usize, device: Device) -> Tensor {
    ep_all_expert_scales_tensor(device).narrow(0, start as i64, (end - start) as i64)
}

fn ep_tch_moe_expert_hidden_size() -> i64 {
    5
}

fn ep_tch_moe_all_expert_up_tensor(device: Device) -> Tensor {
    let count = ep_expert_count() as i64 * 3 * ep_tch_moe_expert_hidden_size();
    Tensor::arange(count, (Kind::Float, device)).reshape([
        ep_expert_count() as i64,
        3,
        ep_tch_moe_expert_hidden_size(),
    ]) / 50.0
        - 0.3
}

fn ep_tch_moe_all_expert_down_tensor(device: Device) -> Tensor {
    let count = ep_expert_count() as i64 * ep_tch_moe_expert_hidden_size() * 3;
    Tensor::arange(count, (Kind::Float, device)).reshape([
        ep_expert_count() as i64,
        ep_tch_moe_expert_hidden_size(),
        3,
    ]) / 60.0
        - 0.25
}

fn ep_tch_moe_owned_expert_up_tensor(start: usize, end: usize, device: Device) -> Tensor {
    ep_tch_moe_all_expert_up_tensor(device).narrow(0, start as i64, (end - start) as i64)
}

fn ep_tch_moe_owned_expert_down_tensor(start: usize, end: usize, device: Device) -> Tensor {
    ep_tch_moe_all_expert_down_tensor(device).narrow(0, start as i64, (end - start) as i64)
}

fn route_top1_tensor(tokens: &Tensor, router: &Tensor) -> Result<Vec<usize>> {
    let assignments = tokens
        .matmul(router)
        .argmax(1, false)
        .to_device(Device::Cpu);
    Vec::<i64>::try_from(assignments)?
        .into_iter()
        .map(|value| {
            usize::try_from(value).with_context(|| format!("invalid expert assignment {value}"))
        })
        .collect()
}

fn ep_owned_expert_load(
    assignments: &[usize],
    owned_expert_start: usize,
    owned_expert_end: usize,
) -> Vec<usize> {
    let mut expert_load = vec![0usize; ep_expert_count()];
    for &expert_index in assignments {
        if (owned_expert_start..owned_expert_end).contains(&expert_index) {
            expert_load[expert_index] += 1;
        }
    }
    expert_load
}

fn ep_global_expert_load(assignments: &[usize]) -> Vec<usize> {
    let mut expert_load = vec![0usize; ep_expert_count()];
    for &expert_index in assignments {
        expert_load[expert_index] += 1;
    }
    expert_load
}

fn ep_load_balance_loss(expert_load: &[usize]) -> f64 {
    let total = expert_load.iter().sum::<usize>().max(1) as f64;
    let expected = 1.0 / expert_load.len().max(1) as f64;
    expert_load
        .iter()
        .map(|load| {
            let fraction = *load as f64 / total;
            (fraction - expected).powi(2)
        })
        .sum()
}

fn ep_source_token_indices(rank: usize, world_size: usize, token_count: usize) -> Vec<usize> {
    (0..token_count)
        .filter(|token_index| token_index % world_size == rank)
        .collect()
}

fn ep_expert_owner_rank(expert_index: usize, world_size: usize) -> usize {
    expert_index / (ep_expert_count() / world_size)
}

fn ep_sparse_dispatch_send_plan(
    rank: usize,
    world_size: usize,
    assignments: &[usize],
    token_count: usize,
) -> Result<Vec<ExpertParallelSparsePeerPlan>> {
    let source_tokens = ep_source_token_indices(rank, world_size, token_count);
    (0..world_size)
        .map(|peer| {
            let token_indices = source_tokens
                .iter()
                .copied()
                .filter(|token_index| {
                    ep_expert_owner_rank(assignments[*token_index], world_size) == peer
                })
                .collect::<Vec<_>>();
            Ok(ExpertParallelSparsePeerPlan {
                peer,
                token_indices,
            })
        })
        .collect()
}

fn ep_sparse_dispatch_recv_plan(
    rank: usize,
    world_size: usize,
    assignments: &[usize],
) -> Result<Vec<ExpertParallelSparsePeerPlan>> {
    (0..world_size)
        .map(|peer| {
            let token_indices = ep_source_token_indices(peer, world_size, assignments.len())
                .into_iter()
                .filter(|token_index| {
                    ep_expert_owner_rank(assignments[*token_index], world_size) == rank
                })
                .collect::<Vec<_>>();
            Ok(ExpertParallelSparsePeerPlan {
                peer,
                token_indices,
            })
        })
        .collect()
}

fn ep_sparse_combine_recv_plan(
    rank: usize,
    world_size: usize,
    assignments: &[usize],
) -> Result<Vec<ExpertParallelSparsePeerPlan>> {
    let source_tokens = ep_source_token_indices(rank, world_size, assignments.len());
    (0..world_size)
        .map(|peer| {
            let token_indices = source_tokens
                .iter()
                .copied()
                .filter(|token_index| {
                    ep_expert_owner_rank(assignments[*token_index], world_size) == peer
                })
                .collect::<Vec<_>>();
            Ok(ExpertParallelSparsePeerPlan {
                peer,
                token_indices,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn ep_sparse_forward(
    output_dir: &Path,
    step_name: &str,
    rank: usize,
    world_size: usize,
    tokens: &Tensor,
    assignments: &[usize],
    owned_expert_start: usize,
    owned_expert_end: usize,
    local_scales: &Tensor,
) -> Result<ExpertParallelSparseForward> {
    let source_token_indices = ep_source_token_indices(rank, world_size, assignments.len());
    let dispatch_send_plan =
        ep_sparse_dispatch_send_plan(rank, world_size, assignments, assignments.len())?;
    let dispatch_recv_plan = ep_sparse_dispatch_recv_plan(rank, world_size, assignments)?;
    let dispatch_sends = dispatch_send_plan
        .iter()
        .map(|plan| {
            (
                plan.peer,
                ep_sparse_pack_token_rows(tokens, &plan.token_indices),
            )
        })
        .collect::<Vec<_>>();
    let dispatch_recvs = dispatch_recv_plan
        .iter()
        .map(|plan| {
            (
                plan.peer,
                vec![plan.token_indices.len() as i64, tokens.size()[1]],
            )
        })
        .collect::<Vec<_>>();
    let dispatched = nccl_smoke::send_recv_tensors_f32_for_launch(
        &output_dir.join(format!("{step_name}-dispatch")),
        &dispatch_sends,
        &dispatch_recvs,
    )?;

    let mut combine_sends = Vec::with_capacity(dispatched.len());
    let mut owned_token_indices = Vec::new();
    for (peer, payload) in dispatched {
        let plan = dispatch_recv_plan
            .iter()
            .find(|plan| plan.peer == peer)
            .ok_or_else(|| anyhow!("missing dispatch recv plan from peer {peer}"))?;
        owned_token_indices.extend(plan.token_indices.iter().copied());
        let output = ep_sparse_local_expert_outputs(
            &payload,
            &plan.token_indices,
            assignments,
            owned_expert_start,
            owned_expert_end,
            local_scales,
        )?;
        combine_sends.push((peer, output));
    }
    owned_token_indices.sort_unstable();

    let combine_recv_plan = ep_sparse_combine_recv_plan(rank, world_size, assignments)?;
    let combine_recvs = combine_recv_plan
        .iter()
        .map(|plan| {
            (
                plan.peer,
                vec![plan.token_indices.len() as i64, tokens.size()[1]],
            )
        })
        .collect::<Vec<_>>();
    let combined = nccl_smoke::send_recv_tensors_f32_for_launch(
        &output_dir.join(format!("{step_name}-combine")),
        &combine_sends,
        &combine_recvs,
    )?;

    let mut assembled_rows = Vec::with_capacity(source_token_indices.len());
    for token_index in &source_token_indices {
        let owner_rank = ep_expert_owner_rank(assignments[*token_index], world_size);
        let payload = combined
            .iter()
            .find(|(peer, _)| *peer == owner_rank)
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| {
                anyhow!("missing combined output from expert owner rank {owner_rank}")
            })?;
        let plan = combine_recv_plan
            .iter()
            .find(|plan| plan.peer == owner_rank)
            .ok_or_else(|| anyhow!("missing combine recv plan from rank {owner_rank}"))?;
        let row_position = plan
            .token_indices
            .iter()
            .position(|candidate| candidate == token_index)
            .ok_or_else(|| anyhow!("token {token_index} missing from combine recv plan"))?;
        assembled_rows.push(payload.get(row_position as i64));
    }
    let assembled_output = Tensor::stack(&assembled_rows.iter().collect::<Vec<_>>(), 0);

    Ok(ExpertParallelSparseForward {
        assembled_output,
        source_token_indices,
        owned_token_indices,
        dispatch_send_counts: dispatch_send_plan
            .iter()
            .map(|plan| plan.token_indices.len())
            .collect(),
        dispatch_recv_counts: dispatch_recv_plan
            .iter()
            .map(|plan| plan.token_indices.len())
            .collect(),
        combine_send_counts: dispatch_recv_plan
            .iter()
            .map(|plan| plan.token_indices.len())
            .collect(),
        combine_recv_counts: combine_recv_plan
            .iter()
            .map(|plan| plan.token_indices.len())
            .collect(),
    })
}

fn ep_reference_output_tensor(tokens: &Tensor, assignments: &[usize]) -> Result<Tensor> {
    ep_local_output_tensor(
        tokens,
        assignments,
        0,
        ep_expert_count(),
        &ep_all_expert_scales_tensor(tokens.device()),
    )
}

fn ep_tch_moe_reference_output_tensor(tokens: &Tensor, assignments: &[usize]) -> Result<Tensor> {
    ep_tch_moe_local_output_tensor(
        tokens,
        assignments,
        0,
        ep_expert_count(),
        &ep_tch_moe_all_expert_up_tensor(tokens.device()),
        &ep_tch_moe_all_expert_down_tensor(tokens.device()),
    )
}

fn ep_local_output_tensor(
    tokens: &Tensor,
    assignments: &[usize],
    owned_expert_start: usize,
    owned_expert_end: usize,
    owned_scales: &Tensor,
) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(assignments.len());
    for (token_index, expert_index) in assignments.iter().copied().enumerate() {
        let token = tokens.get(token_index as i64);
        let row = if (owned_expert_start..owned_expert_end).contains(&expert_index) {
            let local_expert_index = (expert_index - owned_expert_start) as i64;
            token * owned_scales.get(local_expert_index)
        } else {
            Tensor::zeros([tokens.size()[1]], (Kind::Float, tokens.device()))
        };
        rows.push(row);
    }
    Ok(Tensor::stack(&rows.iter().collect::<Vec<_>>(), 0))
}

fn ep_tch_moe_local_output_tensor(
    tokens: &Tensor,
    assignments: &[usize],
    owned_expert_start: usize,
    owned_expert_end: usize,
    owned_up: &Tensor,
    owned_down: &Tensor,
) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(assignments.len());
    for (token_index, expert_index) in assignments.iter().copied().enumerate() {
        let token = tokens.get(token_index as i64);
        let row = if (owned_expert_start..owned_expert_end).contains(&expert_index) {
            let local_expert_index = (expert_index - owned_expert_start) as i64;
            let hidden = token
                .unsqueeze(0)
                .matmul(&owned_up.get(local_expert_index))
                .gelu("none");
            hidden
                .matmul(&owned_down.get(local_expert_index))
                .squeeze_dim(0)
        } else {
            Tensor::zeros([tokens.size()[1]], (Kind::Float, tokens.device()))
        };
        rows.push(row);
    }
    Ok(Tensor::stack(&rows.iter().collect::<Vec<_>>(), 0))
}

fn ep_sparse_pack_token_rows(tensor: &Tensor, token_indices: &[usize]) -> Tensor {
    if token_indices.is_empty() {
        Tensor::zeros([0, tensor.size()[1]], (Kind::Float, tensor.device()))
    } else {
        let rows = token_indices
            .iter()
            .map(|token_index| tensor.get(*token_index as i64))
            .collect::<Vec<_>>();
        Tensor::stack(&rows.iter().collect::<Vec<_>>(), 0)
    }
}

fn ep_sparse_local_expert_outputs(
    dispatched_tokens: &Tensor,
    token_indices: &[usize],
    assignments: &[usize],
    owned_expert_start: usize,
    owned_expert_end: usize,
    owned_scales: &Tensor,
) -> Result<Tensor> {
    if dispatched_tokens.size()[0] != token_indices.len() as i64 {
        bail!(
            "EP sparse dispatched token count mismatch: payload={}, metadata={}",
            dispatched_tokens.size()[0],
            token_indices.len()
        );
    }
    let mut rows = Vec::with_capacity(token_indices.len());
    for (row_index, token_index) in token_indices.iter().copied().enumerate() {
        let expert_index = assignments[token_index];
        if !(owned_expert_start..owned_expert_end).contains(&expert_index) {
            bail!("EP sparse rank received token {token_index} for unowned expert {expert_index}");
        }
        let local_expert_index = (expert_index - owned_expert_start) as i64;
        rows.push(dispatched_tokens.get(row_index as i64) * owned_scales.get(local_expert_index));
    }
    Ok(if rows.is_empty() {
        Tensor::zeros(
            [0, dispatched_tokens.size()[1]],
            (Kind::Float, dispatched_tokens.device()),
        )
    } else {
        Tensor::stack(&rows.iter().collect::<Vec<_>>(), 0)
    })
}

#[allow(clippy::too_many_arguments)]
fn ep_tch_moe_sparse_forward(
    output_dir: &Path,
    step_name: &str,
    rank: usize,
    world_size: usize,
    tokens: &Tensor,
    assignments: &[usize],
    owned_expert_start: usize,
    owned_expert_end: usize,
    local_up: &Tensor,
    local_down: &Tensor,
) -> Result<ExpertParallelSparseForward> {
    let source_token_indices = ep_source_token_indices(rank, world_size, assignments.len());
    let dispatch_send_plan =
        ep_sparse_dispatch_send_plan(rank, world_size, assignments, assignments.len())?;
    let dispatch_recv_plan = ep_sparse_dispatch_recv_plan(rank, world_size, assignments)?;
    let dispatch_sends = dispatch_send_plan
        .iter()
        .map(|plan| {
            (
                plan.peer,
                ep_sparse_pack_token_rows(tokens, &plan.token_indices),
            )
        })
        .collect::<Vec<_>>();
    let dispatch_recvs = dispatch_recv_plan
        .iter()
        .map(|plan| {
            (
                plan.peer,
                vec![plan.token_indices.len() as i64, tokens.size()[1]],
            )
        })
        .collect::<Vec<_>>();
    let dispatched = nccl_smoke::send_recv_tensors_f32_for_launch(
        &output_dir.join(format!("{step_name}-dispatch")),
        &dispatch_sends,
        &dispatch_recvs,
    )?;

    let mut combine_sends = Vec::with_capacity(dispatched.len());
    let mut owned_token_indices = Vec::new();
    for (peer, payload) in dispatched {
        let plan = dispatch_recv_plan
            .iter()
            .find(|plan| plan.peer == peer)
            .ok_or_else(|| anyhow!("missing dispatch recv plan from peer {peer}"))?;
        owned_token_indices.extend(plan.token_indices.iter().copied());
        let output = ep_tch_moe_sparse_local_expert_outputs(
            &payload,
            &plan.token_indices,
            assignments,
            owned_expert_start,
            owned_expert_end,
            local_up,
            local_down,
        )?;
        combine_sends.push((peer, output));
    }
    owned_token_indices.sort_unstable();

    let combine_recv_plan = ep_sparse_combine_recv_plan(rank, world_size, assignments)?;
    let combine_recvs = combine_recv_plan
        .iter()
        .map(|plan| {
            (
                plan.peer,
                vec![plan.token_indices.len() as i64, tokens.size()[1]],
            )
        })
        .collect::<Vec<_>>();
    let combined = nccl_smoke::send_recv_tensors_f32_for_launch(
        &output_dir.join(format!("{step_name}-combine")),
        &combine_sends,
        &combine_recvs,
    )?;

    let mut assembled_rows = Vec::with_capacity(source_token_indices.len());
    for token_index in &source_token_indices {
        let owner_rank = ep_expert_owner_rank(assignments[*token_index], world_size);
        let payload = combined
            .iter()
            .find(|(peer, _)| *peer == owner_rank)
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| {
                anyhow!("missing combined output from expert owner rank {owner_rank}")
            })?;
        let plan = combine_recv_plan
            .iter()
            .find(|plan| plan.peer == owner_rank)
            .ok_or_else(|| anyhow!("missing combine recv plan from rank {owner_rank}"))?;
        let row_position = plan
            .token_indices
            .iter()
            .position(|candidate| candidate == token_index)
            .ok_or_else(|| anyhow!("token {token_index} missing from combine recv plan"))?;
        assembled_rows.push(payload.get(row_position as i64));
    }
    let assembled_output = Tensor::stack(&assembled_rows.iter().collect::<Vec<_>>(), 0);

    Ok(ExpertParallelSparseForward {
        assembled_output,
        source_token_indices,
        owned_token_indices,
        dispatch_send_counts: dispatch_send_plan
            .iter()
            .map(|plan| plan.token_indices.len())
            .collect(),
        dispatch_recv_counts: dispatch_recv_plan
            .iter()
            .map(|plan| plan.token_indices.len())
            .collect(),
        combine_send_counts: dispatch_recv_plan
            .iter()
            .map(|plan| plan.token_indices.len())
            .collect(),
        combine_recv_counts: combine_recv_plan
            .iter()
            .map(|plan| plan.token_indices.len())
            .collect(),
    })
}

fn ep_tch_moe_sparse_local_expert_outputs(
    dispatched_tokens: &Tensor,
    token_indices: &[usize],
    assignments: &[usize],
    owned_expert_start: usize,
    owned_expert_end: usize,
    owned_up: &Tensor,
    owned_down: &Tensor,
) -> Result<Tensor> {
    if dispatched_tokens.size()[0] != token_indices.len() as i64 {
        bail!(
            "EP tch MoE dispatched token count mismatch: payload={}, metadata={}",
            dispatched_tokens.size()[0],
            token_indices.len()
        );
    }
    let mut rows = Vec::with_capacity(token_indices.len());
    for (row_index, token_index) in token_indices.iter().copied().enumerate() {
        let expert_index = assignments[token_index];
        if !(owned_expert_start..owned_expert_end).contains(&expert_index) {
            bail!("EP tch MoE rank received token {token_index} for unowned expert {expert_index}");
        }
        let local_expert_index = (expert_index - owned_expert_start) as i64;
        let token = dispatched_tokens.get(row_index as i64);
        let hidden = token
            .unsqueeze(0)
            .matmul(&owned_up.get(local_expert_index))
            .gelu("none");
        rows.push(
            hidden
                .matmul(&owned_down.get(local_expert_index))
                .squeeze_dim(0),
        );
    }
    Ok(if rows.is_empty() {
        Tensor::zeros(
            [0, dispatched_tokens.size()[1]],
            (Kind::Float, dispatched_tokens.device()),
        )
    } else {
        Tensor::stack(&rows.iter().collect::<Vec<_>>(), 0)
    })
}

#[allow(clippy::too_many_arguments)]
fn ep_sparse_sgd_step(
    output_dir: &Path,
    step_name: &str,
    rank: usize,
    world_size: usize,
    tokens: &Tensor,
    assignments: &[usize],
    owned_expert_start: usize,
    owned_expert_end: usize,
    local_scales: &Tensor,
    state: &ExpertParallelAdamState,
    target_rows: &Tensor,
    post_update_step_name: Option<&str>,
) -> Result<ExpertParallelSparseSgdStep> {
    let forward = ep_sparse_forward(
        output_dir,
        &format!("{step_name}-forward"),
        rank,
        world_size,
        tokens,
        assignments,
        owned_expert_start,
        owned_expert_end,
        local_scales,
    )?;
    let source_grad =
        (&forward.assembled_output - target_rows) * (2.0 / target_rows.numel() as f64);
    let initial_loss = (&forward.assembled_output - target_rows)
        .square()
        .mean(Kind::Float)
        .double_value(&[]);
    let gradient_sends = ep_sparse_output_gradient_sends(
        rank,
        world_size,
        assignments,
        &forward.source_token_indices,
        &source_grad,
    )?;
    let gradient_recv_plan = ep_sparse_dispatch_recv_plan(rank, world_size, assignments)?;
    let gradient_recvs = gradient_recv_plan
        .iter()
        .map(|plan| {
            (
                plan.peer,
                vec![plan.token_indices.len() as i64, tokens.size()[1]],
            )
        })
        .collect::<Vec<_>>();
    let received_gradients = nccl_smoke::send_recv_tensors_f32_for_launch(
        &output_dir.join(format!("{step_name}-output-grad")),
        &gradient_sends,
        &gradient_recvs,
    )?;
    let scale_grad = ep_sparse_scale_grad_from_output_grads(
        &received_gradients,
        &gradient_recv_plan,
        tokens,
        assignments,
        owned_expert_start,
        owned_expert_end,
        local_scales,
    )?;
    let grad_norm = scale_grad.norm().double_value(&[]);
    if grad_norm <= 0.0 {
        bail!("EP sparse train expected positive local expert scale grad norm");
    }
    let updated_scales = (local_scales - &(&scale_grad * 0.25)).detach();
    let next_step = state.step + 1;
    let updated_state = ExpertParallelAdamState {
        m: scale_grad.detach(),
        v: scale_grad.square().detach(),
        step: next_step,
    };
    let final_loss = if let Some(step_name) = post_update_step_name {
        let updated_forward = ep_sparse_forward(
            output_dir,
            step_name,
            rank,
            world_size,
            tokens,
            assignments,
            owned_expert_start,
            owned_expert_end,
            &updated_scales,
        )?;
        Some(
            (&updated_forward.assembled_output - target_rows)
                .square()
                .mean(Kind::Float)
                .double_value(&[]),
        )
    } else {
        None
    };

    Ok(ExpertParallelSparseSgdStep {
        initial_loss,
        final_loss,
        updated_scales,
        updated_state,
        grad_norm,
    })
}

#[allow(clippy::too_many_arguments)]
fn ep_tch_moe_adam_step(
    output_dir: &Path,
    step_name: &str,
    rank: usize,
    world_size: usize,
    tokens: &Tensor,
    assignments: &[usize],
    owned_expert_start: usize,
    owned_expert_end: usize,
    local_up: &Tensor,
    local_down: &Tensor,
    state: &ExpertParallelTchMoeAdamState,
    target_rows: &Tensor,
    adam: &ExpertParallelAdamConfig,
    post_update_step_name: Option<&str>,
) -> Result<ExpertParallelTchMoeAdamStep> {
    let forward = ep_tch_moe_sparse_forward(
        output_dir,
        &format!("{step_name}-forward"),
        rank,
        world_size,
        tokens,
        assignments,
        owned_expert_start,
        owned_expert_end,
        local_up,
        local_down,
    )?;
    let source_grad =
        (&forward.assembled_output - target_rows) * (2.0 / target_rows.numel() as f64);
    let initial_loss = (&forward.assembled_output - target_rows)
        .square()
        .mean(Kind::Float)
        .double_value(&[]);
    let gradient_sends = ep_sparse_output_gradient_sends(
        rank,
        world_size,
        assignments,
        &forward.source_token_indices,
        &source_grad,
    )?;
    let gradient_recv_plan = ep_sparse_dispatch_recv_plan(rank, world_size, assignments)?;
    let gradient_recvs = gradient_recv_plan
        .iter()
        .map(|plan| {
            (
                plan.peer,
                vec![plan.token_indices.len() as i64, tokens.size()[1]],
            )
        })
        .collect::<Vec<_>>();
    let received_gradients = nccl_smoke::send_recv_tensors_f32_for_launch(
        &output_dir.join(format!("{step_name}-output-grad")),
        &gradient_sends,
        &gradient_recvs,
    )?;

    let train_up = local_up.detach().set_requires_grad(true);
    let train_down = local_down.detach().set_requires_grad(true);
    let local_output_grad_loss = ep_tch_moe_local_output_grad_bridge(
        &received_gradients,
        &gradient_recv_plan,
        tokens,
        assignments,
        owned_expert_start,
        owned_expert_end,
        &train_up,
        &train_down,
    )?;
    local_output_grad_loss.backward();
    let up_grad = train_up.grad();
    let down_grad = train_down.grad();
    let up_grad_norm = up_grad.norm().double_value(&[]);
    let down_grad_norm = down_grad.norm().double_value(&[]);
    if !up_grad.defined() || !down_grad.defined() || up_grad_norm <= 0.0 || down_grad_norm <= 0.0 {
        bail!(
            "EP tch MoE train expected positive expert gradients: up={up_grad_norm}, down={down_grad_norm}"
        );
    }

    let next_step = state.step + 1;
    let (updated_up, up_m, up_v) = ep_adamw_update_tensor(
        local_up,
        &up_grad,
        &state.up_m,
        &state.up_v,
        next_step,
        adam,
    );
    let (updated_down, down_m, down_v) = ep_adamw_update_tensor(
        local_down,
        &down_grad,
        &state.down_m,
        &state.down_v,
        next_step,
        adam,
    );
    let updated_state = ExpertParallelTchMoeAdamState {
        up_m,
        up_v,
        down_m,
        down_v,
        step: next_step,
    };
    let final_loss = if let Some(step_name) = post_update_step_name {
        let updated_forward = ep_tch_moe_sparse_forward(
            output_dir,
            step_name,
            rank,
            world_size,
            tokens,
            assignments,
            owned_expert_start,
            owned_expert_end,
            &updated_up,
            &updated_down,
        )?;
        Some(
            (&updated_forward.assembled_output - target_rows)
                .square()
                .mean(Kind::Float)
                .double_value(&[]),
        )
    } else {
        None
    };

    Ok(ExpertParallelTchMoeAdamStep {
        initial_loss,
        final_loss,
        updated_expert_up: updated_up,
        updated_expert_down: updated_down,
        updated_state,
        up_grad_norm,
        down_grad_norm,
    })
}

fn ep_sparse_output_gradient_sends(
    _rank: usize,
    world_size: usize,
    assignments: &[usize],
    source_token_indices: &[usize],
    source_grad: &Tensor,
) -> Result<Vec<(usize, Tensor)>> {
    if source_grad.size()[0] != source_token_indices.len() as i64 {
        bail!(
            "EP sparse source grad count mismatch: payload={}, metadata={}",
            source_grad.size()[0],
            source_token_indices.len()
        );
    }
    let feature_dim = source_grad.size()[1];
    let mut peer_rows = (0..world_size)
        .map(|_| Vec::<Tensor>::new())
        .collect::<Vec<_>>();
    for (row_index, token_index) in source_token_indices.iter().copied().enumerate() {
        let owner_rank = ep_expert_owner_rank(assignments[token_index], world_size);
        peer_rows[owner_rank].push(source_grad.get(row_index as i64));
    }
    Ok(peer_rows
        .into_iter()
        .enumerate()
        .map(|(peer, rows)| {
            let payload = if rows.is_empty() {
                Tensor::zeros([0, feature_dim], (Kind::Float, source_grad.device()))
            } else {
                Tensor::stack(&rows.iter().collect::<Vec<_>>(), 0)
            };
            (peer, payload)
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn ep_tch_moe_local_output_grad_bridge(
    received_gradients: &[(usize, Tensor)],
    gradient_recv_plan: &[ExpertParallelSparsePeerPlan],
    tokens: &Tensor,
    assignments: &[usize],
    owned_expert_start: usize,
    owned_expert_end: usize,
    local_up: &Tensor,
    local_down: &Tensor,
) -> Result<Tensor> {
    let mut terms = Vec::new();
    for plan in gradient_recv_plan {
        let payload = received_gradients
            .iter()
            .find(|(peer, _)| *peer == plan.peer)
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| anyhow!("missing output gradient payload from peer {}", plan.peer))?;
        if payload.size()[0] != plan.token_indices.len() as i64 {
            bail!(
                "EP tch MoE output grad count mismatch from peer {}: payload={}, metadata={}",
                plan.peer,
                payload.size()[0],
                plan.token_indices.len()
            );
        }
        for (row_index, token_index) in plan.token_indices.iter().copied().enumerate() {
            let expert_index = assignments[token_index];
            if !(owned_expert_start..owned_expert_end).contains(&expert_index) {
                bail!(
                    "EP tch MoE rank received output grad for token {token_index} assigned to unowned expert {expert_index}"
                );
            }
            let local_expert_index = (expert_index - owned_expert_start) as i64;
            let token = tokens.get(token_index as i64);
            let hidden = token
                .unsqueeze(0)
                .matmul(&local_up.get(local_expert_index))
                .gelu("none");
            let output = hidden
                .matmul(&local_down.get(local_expert_index))
                .squeeze_dim(0);
            terms.push((output * payload.get(row_index as i64)).sum(Kind::Float));
        }
    }
    if terms.is_empty() {
        bail!("EP tch MoE rank has no owned tokens and cannot produce gradient evidence");
    }
    Ok(Tensor::stack(&terms.iter().collect::<Vec<_>>(), 0).sum(Kind::Float))
}

#[allow(clippy::too_many_arguments)]
fn ep_sparse_scale_grad_from_output_grads(
    received_gradients: &[(usize, Tensor)],
    gradient_recv_plan: &[ExpertParallelSparsePeerPlan],
    tokens: &Tensor,
    assignments: &[usize],
    owned_expert_start: usize,
    owned_expert_end: usize,
    local_scales: &Tensor,
) -> Result<Tensor> {
    let scale_grad = Tensor::zeros_like(local_scales);
    for plan in gradient_recv_plan {
        let payload = received_gradients
            .iter()
            .find(|(peer, _)| *peer == plan.peer)
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| anyhow!("missing output gradient payload from peer {}", plan.peer))?;
        if payload.size()[0] != plan.token_indices.len() as i64 {
            bail!(
                "EP sparse output grad count mismatch from peer {}: payload={}, metadata={}",
                plan.peer,
                payload.size()[0],
                plan.token_indices.len()
            );
        }
        for (row_index, token_index) in plan.token_indices.iter().copied().enumerate() {
            let expert_index = assignments[token_index];
            if !(owned_expert_start..owned_expert_end).contains(&expert_index) {
                bail!(
                    "EP sparse rank received output grad for token {token_index} assigned to unowned expert {expert_index}"
                );
            }
            let local_expert_index = (expert_index - owned_expert_start) as i64;
            let grad_row = payload.get(row_index as i64);
            let token_row = tokens.get(token_index as i64);
            let row_scale_grad = grad_row * token_row;
            let updated_row = scale_grad.get(local_expert_index) + row_scale_grad;
            scale_grad.get(local_expert_index).copy_(&updated_row);
        }
    }
    Ok(scale_grad)
}

fn ep_adamw_update_tensor(
    parameter: &Tensor,
    grad: &Tensor,
    m: &Tensor,
    v: &Tensor,
    step: i64,
    adam: &ExpertParallelAdamConfig,
) -> (Tensor, Tensor, Tensor) {
    let next_m = m * adam.beta1 + grad * (1.0 - adam.beta1);
    let next_v = v * adam.beta2 + grad.square() * (1.0 - adam.beta2);
    let m_hat = &next_m / (1.0 - adam.beta1.powi(step as i32));
    let v_hat = &next_v / (1.0 - adam.beta2.powi(step as i32));
    let decayed = if adam.weight_decay == 0.0 {
        parameter.shallow_clone()
    } else {
        parameter * (1.0 - adam.learning_rate * adam.weight_decay)
    };
    let updated = (decayed - (m_hat / (v_hat.sqrt() + adam.eps)) * adam.learning_rate).detach();
    (updated, next_m.detach(), next_v.detach())
}

#[allow(clippy::too_many_arguments)]
fn ep_nccl_adam_step(
    reduce_dir: &Path,
    post_update_reduce_dir: Option<&Path>,
    tokens: &Tensor,
    assignments: &[usize],
    owned_expert_start: usize,
    owned_expert_end: usize,
    scales: &Tensor,
    state: &ExpertParallelAdamState,
    target: &Tensor,
    reference: Option<&Tensor>,
    adam: &ExpertParallelAdamConfig,
) -> Result<ExpertParallelAdamStep> {
    let train_scales = scales.detach().set_requires_grad(true);
    let local_output = ep_local_output_tensor(
        tokens,
        assignments,
        owned_expert_start,
        owned_expert_end,
        &train_scales,
    )?;
    let reduced_output = nccl_smoke::all_reduce_tensor_f32_for_launch(reduce_dir, &local_output)?;
    let reduced_output = reduced_output.set_requires_grad(true);
    let reduced_output_shape = reduced_output.size();
    let (combine_max_abs, combine_mean_abs) = if let Some(reference) = reference {
        tensor_diff_stats(&reduced_output, reference)?
    } else {
        (0.0, 0.0)
    };

    let loss_tensor = (&reduced_output - target).square().mean(Kind::Float);
    let pre_loss = loss_tensor.double_value(&[]);
    let output_grads = Tensor::run_backward(&[loss_tensor], &[&reduced_output], false, false);
    let output_grad = output_grads
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("EP NCCL Adam step did not produce an output gradient"))?;
    if !output_grad.defined() || output_grad.norm().double_value(&[]) <= 0.0 {
        bail!("EP NCCL Adam step output gradient is missing or zero");
    }

    let local_train_output = ep_local_output_tensor(
        tokens,
        assignments,
        owned_expert_start,
        owned_expert_end,
        &train_scales,
    )?;
    (local_train_output * &output_grad.detach())
        .sum(Kind::Float)
        .backward();
    let grad = train_scales.grad();
    let grad_norm = grad.norm().double_value(&[]);
    if !grad.defined() || grad_norm <= 0.0 {
        bail!("EP NCCL Adam step local expert scale gradient is missing or zero");
    }

    let next_step = state.step + 1;
    let m = &state.m * adam.beta1 + &grad * (1.0 - adam.beta1);
    let v = &state.v * adam.beta2 + grad.square() * (1.0 - adam.beta2);
    let m_hat = &m / (1.0 - adam.beta1.powi(next_step as i32));
    let v_hat = &v / (1.0 - adam.beta2.powi(next_step as i32));
    let decayed = if adam.weight_decay == 0.0 {
        scales.shallow_clone()
    } else {
        scales * (1.0 - adam.learning_rate * adam.weight_decay)
    };
    let updated_scales =
        (decayed - (m_hat / (v_hat.sqrt() + adam.eps)) * adam.learning_rate).detach();
    let updated_state = ExpertParallelAdamState {
        m: m.detach(),
        v: v.detach(),
        step: next_step,
    };

    let post_loss = if let Some(post_reduce_dir) = post_update_reduce_dir {
        let updated_local_output = ep_local_output_tensor(
            tokens,
            assignments,
            owned_expert_start,
            owned_expert_end,
            &updated_scales,
        )?;
        let updated_reduced =
            nccl_smoke::all_reduce_tensor_f32_for_launch(post_reduce_dir, &updated_local_output)?;
        Some(
            (&updated_reduced - target)
                .square()
                .mean(Kind::Float)
                .double_value(&[]),
        )
    } else {
        None
    };

    Ok(ExpertParallelAdamStep {
        pre_loss,
        post_loss,
        updated_scales,
        updated_state,
        grad_norm,
        reduced_output_shape,
        combine_max_abs,
        combine_mean_abs,
    })
}

#[allow(clippy::too_many_arguments)]
fn write_ep_rank_checkpoint(
    output_dir: &Path,
    rank: usize,
    world_size: usize,
    local_rank: usize,
    owned_expert_start: usize,
    owned_expert_end: usize,
    scales: &Tensor,
    state: &ExpertParallelAdamState,
    adam: &ExpertParallelAdamConfig,
) -> Result<ExpertParallelCheckpointWrite> {
    let rank_dir = output_dir.join(format!("ep-checkpoint-rank-{rank}"));
    fs::create_dir_all(&rank_dir)
        .with_context(|| format!("failed to create {}", rank_dir.display()))?;
    let model_safetensors = rank_dir.join("model.safetensors");
    let optimizer_safetensors = rank_dir.join("optimizer.safetensors");
    let scale_name = "experts.scale";
    let m_name = "experts.scale.adam_m";
    let v_name = "experts.scale.adam_v";

    Tensor::write_safetensors(&[(scale_name, scales)], &model_safetensors)
        .with_context(|| format!("failed to write {}", model_safetensors.display()))?;
    Tensor::write_safetensors(
        &[(m_name, &state.m), (v_name, &state.v)],
        &optimizer_safetensors,
    )
    .with_context(|| format!("failed to write {}", optimizer_safetensors.display()))?;

    let manifest = ExpertParallelCheckpointManifest {
        format: "rustrain.ep_sharded.v1".to_string(),
        manifest_kind: "rank".to_string(),
        rank,
        world_size,
        local_rank,
        global_step: state.step as u64,
        owned_expert_start,
        owned_expert_end,
        model_safetensors: model_safetensors.display().to_string(),
        optimizer_safetensors: optimizer_safetensors.display().to_string(),
        optimizer: "adamw".to_string(),
        learning_rate: adam.learning_rate,
        beta1: adam.beta1,
        beta2: adam.beta2,
        eps: adam.eps,
        weight_decay: adam.weight_decay,
        shards: vec![ExpertParallelCheckpointShard {
            name: format!("experts.{owned_expert_start}..{owned_expert_end}.scale"),
            shard_name: scale_name.to_string(),
            optimizer_m_name: m_name.to_string(),
            optimizer_v_name: v_name.to_string(),
            global_shape: vec![ep_expert_count() as i64, scales.size()[1]],
            shard_shape: scales.size(),
            dtype: "float32".to_string(),
            partition: "expert_model_parallel".to_string(),
        }],
    };
    let manifest_path = rank_dir.join("manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest)? + "\n",
    )
    .with_context(|| format!("failed to write {}", manifest_path.display()))?;

    Ok(ExpertParallelCheckpointWrite {
        manifest_path,
        model_safetensors,
        optimizer_safetensors,
        tensor_count: manifest.shards.len(),
    })
}

fn read_ep_rank_checkpoint(
    checkpoint: &ExpertParallelCheckpointWrite,
) -> Result<(Tensor, ExpertParallelAdamState)> {
    let manifest: ExpertParallelCheckpointManifest = serde_json::from_str(
        &fs::read_to_string(&checkpoint.manifest_path)
            .with_context(|| format!("failed to read {}", checkpoint.manifest_path.display()))?,
    )?;
    if manifest.format != "rustrain.ep_sharded.v1" {
        bail!("unsupported EP checkpoint format {}", manifest.format);
    }
    if manifest.shards.len() != 1 {
        bail!(
            "EP checkpoint expected exactly one rank-owned expert shard, got {}",
            manifest.shards.len()
        );
    }
    let shard = &manifest.shards[0];
    let model_tensors = read_tensor_map(Path::new(&manifest.model_safetensors))?;
    let optimizer_tensors = read_tensor_map(Path::new(&manifest.optimizer_safetensors))?;
    let scales = tensor_from_map(&model_tensors, &shard.shard_name)?.shallow_clone();
    let m = tensor_from_map(&optimizer_tensors, &shard.optimizer_m_name)?.shallow_clone();
    let v = tensor_from_map(&optimizer_tensors, &shard.optimizer_v_name)?.shallow_clone();
    if scales.size() != shard.shard_shape
        || m.size() != shard.shard_shape
        || v.size() != shard.shard_shape
    {
        bail!(
            "EP checkpoint shard shape mismatch: shard={:?}, scales={:?}, m={:?}, v={:?}",
            shard.shard_shape,
            scales.size(),
            m.size(),
            v.size()
        );
    }
    Ok((
        scales,
        ExpertParallelAdamState {
            m,
            v,
            step: manifest.global_step as i64,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn write_ep_tch_moe_rank_checkpoint(
    output_dir: &Path,
    rank: usize,
    world_size: usize,
    local_rank: usize,
    owned_expert_start: usize,
    owned_expert_end: usize,
    expert_up: &Tensor,
    expert_down: &Tensor,
    state: &ExpertParallelTchMoeAdamState,
    adam: &ExpertParallelAdamConfig,
) -> Result<ExpertParallelCheckpointWrite> {
    let rank_dir = output_dir.join(format!("ep-tch-moe-checkpoint-rank-{rank}"));
    fs::create_dir_all(&rank_dir)
        .with_context(|| format!("failed to create {}", rank_dir.display()))?;
    let model_safetensors = rank_dir.join("model.safetensors");
    let optimizer_safetensors = rank_dir.join("optimizer.safetensors");
    let up_name = "experts.up.weight";
    let down_name = "experts.down.weight";
    let up_m_name = "experts.up.weight.adam_m";
    let up_v_name = "experts.up.weight.adam_v";
    let down_m_name = "experts.down.weight.adam_m";
    let down_v_name = "experts.down.weight.adam_v";

    Tensor::write_safetensors(
        &[(up_name, expert_up), (down_name, expert_down)],
        &model_safetensors,
    )
    .with_context(|| format!("failed to write {}", model_safetensors.display()))?;
    Tensor::write_safetensors(
        &[
            (up_m_name, &state.up_m),
            (up_v_name, &state.up_v),
            (down_m_name, &state.down_m),
            (down_v_name, &state.down_v),
        ],
        &optimizer_safetensors,
    )
    .with_context(|| format!("failed to write {}", optimizer_safetensors.display()))?;

    let manifest = ExpertParallelCheckpointManifest {
        format: "rustrain.ep_sharded.v1".to_string(),
        manifest_kind: "rank".to_string(),
        rank,
        world_size,
        local_rank,
        global_step: state.step as u64,
        owned_expert_start,
        owned_expert_end,
        model_safetensors: model_safetensors.display().to_string(),
        optimizer_safetensors: optimizer_safetensors.display().to_string(),
        optimizer: "adamw".to_string(),
        learning_rate: adam.learning_rate,
        beta1: adam.beta1,
        beta2: adam.beta2,
        eps: adam.eps,
        weight_decay: adam.weight_decay,
        shards: vec![
            ExpertParallelCheckpointShard {
                name: format!("experts.{owned_expert_start}..{owned_expert_end}.up.weight"),
                shard_name: up_name.to_string(),
                optimizer_m_name: up_m_name.to_string(),
                optimizer_v_name: up_v_name.to_string(),
                global_shape: vec![
                    ep_expert_count() as i64,
                    expert_up.size()[1],
                    expert_up.size()[2],
                ],
                shard_shape: expert_up.size(),
                dtype: "float32".to_string(),
                partition: "expert_model_parallel".to_string(),
            },
            ExpertParallelCheckpointShard {
                name: format!("experts.{owned_expert_start}..{owned_expert_end}.down.weight"),
                shard_name: down_name.to_string(),
                optimizer_m_name: down_m_name.to_string(),
                optimizer_v_name: down_v_name.to_string(),
                global_shape: vec![
                    ep_expert_count() as i64,
                    expert_down.size()[1],
                    expert_down.size()[2],
                ],
                shard_shape: expert_down.size(),
                dtype: "float32".to_string(),
                partition: "expert_model_parallel".to_string(),
            },
        ],
    };
    let manifest_path = rank_dir.join("manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest)? + "\n",
    )
    .with_context(|| format!("failed to write {}", manifest_path.display()))?;

    Ok(ExpertParallelCheckpointWrite {
        manifest_path,
        model_safetensors,
        optimizer_safetensors,
        tensor_count: manifest.shards.len(),
    })
}

fn wait_for_ep_rank_barrier(
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
            bail!("timed out waiting for EP barrier {}", dir.display());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn write_ep_tch_moe_global_checkpoint_manifest(
    output_dir: &Path,
    rank: usize,
    world_size: usize,
    checkpoint: &ExpertParallelCheckpointWrite,
) -> Result<PathBuf> {
    let rank_manifest_link = output_dir.join(format!("ep-tch-moe-manifest-rank-{rank}.json"));
    let rank_manifest_text = fs::read_to_string(&checkpoint.manifest_path)
        .with_context(|| format!("failed to read {}", checkpoint.manifest_path.display()))?;
    fs::write(&rank_manifest_link, &rank_manifest_text)
        .with_context(|| format!("failed to write {}", rank_manifest_link.display()))?;
    wait_for_ep_rank_barrier(
        &output_dir.join("ep-tch-moe-rank-manifests-written"),
        rank,
        world_size,
        Duration::from_secs(300),
    )?;

    let global_manifest_output = output_dir.join("ep-tch-moe-sharded-global.json");
    if rank == 0 {
        let mut ranks = Vec::with_capacity(world_size);
        for shard_rank in 0..world_size {
            let rank_manifest_path =
                output_dir.join(format!("ep-tch-moe-manifest-rank-{shard_rank}.json"));
            let text = fs::read_to_string(&rank_manifest_path)
                .with_context(|| format!("failed to read {}", rank_manifest_path.display()))?;
            let rank_manifest: ExpertParallelCheckpointManifest = serde_json::from_str(&text)
                .with_context(|| format!("failed to parse {}", rank_manifest_path.display()))?;
            ranks.push(rank_manifest);
        }
        let manifest = ExpertParallelGlobalCheckpointManifest {
            format: "rustrain.ep_sharded.v1".to_string(),
            manifest_kind: "global".to_string(),
            base_model_path: "rustrain://focused-tch-moe-ep-smoke".to_string(),
            tokenizer_path: String::new(),
            global_step: 1,
            consumed_samples: ep_tokens().nrows() as u64,
            consumed_tokens: ep_tokens().nrows() as u64,
            data_cursor_next: None,
            data_epoch_next: None,
            data_sample_offset_next: None,
            data_train_samples: None,
            dataset_source_files: Vec::new(),
            dataset_source_sample_counts: Vec::new(),
            dataset_fingerprint: String::new(),
            dataset_shuffle: true,
            seed: 0,
            dtype: "float32".to_string(),
            optimizer: "adamw".to_string(),
            scheduler: "constant".to_string(),
            parallel: ExpertParallelGlobalParallelManifest {
                data_parallel_size: 1,
                tensor_model_parallel_size: 1,
                pipeline_model_parallel_size: 1,
                expert_model_parallel_size: world_size,
                context_parallel_size: 1,
            },
            ranks,
        };
        manifest.validate_artifacts()?;
        fs::write(
            &global_manifest_output,
            serde_json::to_string_pretty(&manifest)? + "\n",
        )
        .with_context(|| format!("failed to write {}", global_manifest_output.display()))?;
    }
    wait_for_ep_rank_barrier(
        &output_dir.join("ep-tch-moe-global-manifest-written"),
        rank,
        world_size,
        Duration::from_secs(300),
    )?;
    let global_text = fs::read_to_string(&global_manifest_output)
        .with_context(|| format!("failed to read {}", global_manifest_output.display()))?;
    let global_manifest: ExpertParallelGlobalCheckpointManifest =
        serde_json::from_str(&global_text)
            .with_context(|| format!("failed to parse {}", global_manifest_output.display()))?;
    global_manifest.validate_artifacts()?;
    Ok(global_manifest_output)
}

fn read_ep_tch_moe_rank_from_global_manifest(
    resume_from: &Path,
    rank: usize,
    world_size: usize,
) -> Result<ExpertParallelTchMoeExternalResume> {
    let text = fs::read_to_string(resume_from)
        .with_context(|| format!("failed to read {}", resume_from.display()))?;
    let manifest: ExpertParallelGlobalCheckpointManifest = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", resume_from.display()))?;
    manifest.validate_artifacts()?;
    if manifest.parallel.expert_model_parallel_size != world_size {
        bail!(
            "EP tch MoE resume expert_model_parallel_size {} does not match WORLD_SIZE {world_size}",
            manifest.parallel.expert_model_parallel_size
        );
    }
    if manifest.parallel.data_parallel_size != 1
        || manifest.parallel.tensor_model_parallel_size != 1
        || manifest.parallel.pipeline_model_parallel_size != 1
        || manifest.parallel.context_parallel_size != 1
    {
        bail!(
            "EP tch MoE resume expects only expert parallelism, got DP={} TP={} PP={} CP={}",
            manifest.parallel.data_parallel_size,
            manifest.parallel.tensor_model_parallel_size,
            manifest.parallel.pipeline_model_parallel_size,
            manifest.parallel.context_parallel_size
        );
    }
    let rank_manifest = manifest
        .ranks
        .iter()
        .find(|rank_manifest| rank_manifest.rank == rank)
        .ok_or_else(|| anyhow!("EP tch MoE resume manifest is missing rank {rank}"))?;
    if rank_manifest.local_rank != rank {
        bail!(
            "EP tch MoE resume expected local_rank {rank}, got {}",
            rank_manifest.local_rank
        );
    }
    if rank_manifest.shards.len() != 2 {
        bail!(
            "EP tch MoE resume expected two expert MLP shards, got {}",
            rank_manifest.shards.len()
        );
    }
    let up_shard = rank_manifest
        .shards
        .iter()
        .find(|shard| shard.shard_name == "experts.up.weight")
        .ok_or_else(|| anyhow!("EP tch MoE resume missing experts.up.weight shard"))?;
    let down_shard = rank_manifest
        .shards
        .iter()
        .find(|shard| shard.shard_name == "experts.down.weight")
        .ok_or_else(|| anyhow!("EP tch MoE resume missing experts.down.weight shard"))?;
    let model_tensors = read_tensor_map(Path::new(&rank_manifest.model_safetensors))?;
    let optimizer_tensors = read_tensor_map(Path::new(&rank_manifest.optimizer_safetensors))?;
    let expert_up = tensor_from_map(&model_tensors, &up_shard.shard_name)?.shallow_clone();
    let expert_down = tensor_from_map(&model_tensors, &down_shard.shard_name)?.shallow_clone();
    let up_m = tensor_from_map(&optimizer_tensors, &up_shard.optimizer_m_name)?.shallow_clone();
    let up_v = tensor_from_map(&optimizer_tensors, &up_shard.optimizer_v_name)?.shallow_clone();
    let down_m = tensor_from_map(&optimizer_tensors, &down_shard.optimizer_m_name)?.shallow_clone();
    let down_v = tensor_from_map(&optimizer_tensors, &down_shard.optimizer_v_name)?.shallow_clone();
    if expert_up.size() != up_shard.shard_shape
        || up_m.size() != up_shard.shard_shape
        || up_v.size() != up_shard.shard_shape
        || expert_down.size() != down_shard.shard_shape
        || down_m.size() != down_shard.shard_shape
        || down_v.size() != down_shard.shard_shape
    {
        bail!(
            "EP tch MoE resume shard shape mismatch: up={:?}/{:?}, down={:?}/{:?}",
            expert_up.size(),
            up_shard.shard_shape,
            expert_down.size(),
            down_shard.shard_shape
        );
    }
    Ok(ExpertParallelTchMoeExternalResume {
        global_step: manifest.global_step,
        rank_manifest_output: resume_from.display().to_string(),
        model_safetensors: rank_manifest.model_safetensors.clone(),
        optimizer_safetensors: rank_manifest.optimizer_safetensors.clone(),
        sharded_manifest_tensor_count: rank_manifest.shards.len(),
        expert_up,
        expert_down,
        state: ExpertParallelTchMoeAdamState {
            up_m,
            up_v,
            down_m,
            down_v,
            step: rank_manifest.global_step as i64,
        },
    })
}

fn read_ep_tch_moe_rank_checkpoint(
    checkpoint: &ExpertParallelCheckpointWrite,
) -> Result<ExpertParallelTchMoeCheckpointRead> {
    let manifest: ExpertParallelCheckpointManifest = serde_json::from_str(
        &fs::read_to_string(&checkpoint.manifest_path)
            .with_context(|| format!("failed to read {}", checkpoint.manifest_path.display()))?,
    )?;
    if manifest.format != "rustrain.ep_sharded.v1" {
        bail!(
            "unsupported EP tch MoE checkpoint format {}",
            manifest.format
        );
    }
    if manifest.shards.len() != 2 {
        bail!(
            "EP tch MoE checkpoint expected two rank-owned expert shards, got {}",
            manifest.shards.len()
        );
    }
    let up_shard = manifest
        .shards
        .iter()
        .find(|shard| shard.shard_name == "experts.up.weight")
        .ok_or_else(|| anyhow!("missing experts.up.weight shard"))?;
    let down_shard = manifest
        .shards
        .iter()
        .find(|shard| shard.shard_name == "experts.down.weight")
        .ok_or_else(|| anyhow!("missing experts.down.weight shard"))?;
    let model_tensors = read_tensor_map(Path::new(&manifest.model_safetensors))?;
    let optimizer_tensors = read_tensor_map(Path::new(&manifest.optimizer_safetensors))?;
    let expert_up = tensor_from_map(&model_tensors, &up_shard.shard_name)?.shallow_clone();
    let expert_down = tensor_from_map(&model_tensors, &down_shard.shard_name)?.shallow_clone();
    let up_m = tensor_from_map(&optimizer_tensors, &up_shard.optimizer_m_name)?.shallow_clone();
    let up_v = tensor_from_map(&optimizer_tensors, &up_shard.optimizer_v_name)?.shallow_clone();
    let down_m = tensor_from_map(&optimizer_tensors, &down_shard.optimizer_m_name)?.shallow_clone();
    let down_v = tensor_from_map(&optimizer_tensors, &down_shard.optimizer_v_name)?.shallow_clone();
    if expert_up.size() != up_shard.shard_shape
        || up_m.size() != up_shard.shard_shape
        || up_v.size() != up_shard.shard_shape
        || expert_down.size() != down_shard.shard_shape
        || down_m.size() != down_shard.shard_shape
        || down_v.size() != down_shard.shard_shape
    {
        bail!(
            "EP tch MoE checkpoint shard shape mismatch: up={:?}/{:?}, down={:?}/{:?}",
            expert_up.size(),
            up_shard.shard_shape,
            expert_down.size(),
            down_shard.shard_shape
        );
    }
    Ok(ExpertParallelTchMoeCheckpointRead {
        expert_up,
        expert_down,
        state: ExpertParallelTchMoeAdamState {
            up_m,
            up_v,
            down_m,
            down_v,
            step: manifest.global_step as i64,
        },
    })
}

fn ep_tch_moe_adam_state_max_abs_diff(
    actual: &ExpertParallelTchMoeAdamState,
    expected: &ExpertParallelTchMoeAdamState,
) -> Result<f64> {
    Ok(tensor_max_abs_diff(&actual.up_m, &expected.up_m)?
        .max(tensor_max_abs_diff(&actual.up_v, &expected.up_v)?)
        .max(tensor_max_abs_diff(&actual.down_m, &expected.down_m)?)
        .max(tensor_max_abs_diff(&actual.down_v, &expected.down_v)?)
        .max((actual.step - expected.step).abs() as f64))
}

fn read_tensor_map(path: &Path) -> Result<BTreeMap<String, Tensor>> {
    let tensors = Tensor::read_safetensors(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(tensors.into_iter().collect())
}

fn tensor_from_map<'a>(tensors: &'a BTreeMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| anyhow!("missing tensor {name}"))
}

fn tensor_diff_stats(actual: &Tensor, expected: &Tensor) -> Result<(f64, f64)> {
    if actual.size() != expected.size() {
        bail!(
            "shape mismatch: actual {:?}, expected {:?}",
            actual.size(),
            expected.size()
        );
    }
    let diff = (actual - expected).abs().to_device(Device::Cpu);
    Ok((
        diff.max().double_value(&[]),
        diff.mean(Kind::Float).double_value(&[]),
    ))
}

fn tensor_max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<f64> {
    if actual.size() != expected.size() {
        bail!(
            "shape mismatch: actual {:?}, expected {:?}",
            actual.size(),
            expected.size()
        );
    }
    let actual = actual.to_device(Device::Cpu);
    let expected = expected.to_device(Device::Cpu);
    Ok((actual - expected).abs().max().double_value(&[]))
}

fn parse_launcher_usize_env(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| {
            format!(
                "{name} is not set; pass --{} or run through rustrain launch",
                name.to_ascii_lowercase()
            )
        })?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}
fn route_top1(tokens: &Array2<f64>, router: &Array2<f64>) -> Vec<usize> {
    tokens
        .dot(router)
        .rows()
        .into_iter()
        .map(|scores| {
            scores
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index)
                .expect("router should produce at least one expert score")
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[test]
    fn ep_global_checkpoint_manifest_validates_rank_owned_shards() {
        let manifest = tiny_ep_global_manifest();

        manifest
            .validate()
            .expect("valid EP global manifest should pass");
    }

    #[test]
    fn ep_global_checkpoint_manifest_rejects_missing_rank() {
        let mut manifest = tiny_ep_global_manifest();
        manifest.ranks.pop();

        let error = manifest
            .validate()
            .expect_err("missing rank should fail")
            .to_string();
        assert!(error.contains("expected 2 rank manifests"));
    }

    #[test]
    fn ep_global_checkpoint_manifest_rejects_wrong_world_size() {
        let mut manifest = tiny_ep_global_manifest();
        manifest.ranks[1].world_size = 3;

        let error = manifest
            .validate()
            .expect_err("wrong rank world size should fail")
            .to_string();
        assert!(error.contains("does not match global EP size"));
    }

    #[test]
    fn ep_global_checkpoint_manifest_rejects_non_ep_parallel_axes() {
        let mut manifest = tiny_ep_global_manifest();
        manifest.parallel.tensor_model_parallel_size = 2;

        let error = manifest
            .validate()
            .expect_err("non-EP parallel axis should fail")
            .to_string();
        assert!(error.contains("expects DP/TP/PP/CP sizes to be 1"));
    }

    #[test]
    fn ep_global_checkpoint_manifest_rejects_mismatched_rank_step() {
        let mut manifest = tiny_ep_global_manifest();
        manifest.ranks[1].global_step = 2;

        let error = manifest
            .validate()
            .expect_err("mismatched rank step should fail")
            .to_string();
        assert!(error.contains("does not match global step"));
    }

    #[test]
    fn ep_global_checkpoint_manifest_rejects_non_contiguous_experts() {
        let mut manifest = tiny_ep_global_manifest();
        manifest.ranks[1].owned_expert_start = 3;

        let error = manifest
            .validate()
            .expect_err("expert range gap should fail")
            .to_string();
        assert!(error.contains("does not continue"));
    }

    #[test]
    fn ep_global_checkpoint_manifest_rejects_missing_optimizer_slots() {
        let mut manifest = tiny_ep_global_manifest();
        manifest.ranks[0].shards[0].optimizer_m_name.clear();

        let error = manifest
            .validate()
            .expect_err("missing optimizer slot should fail")
            .to_string();
        assert!(error.contains("missing optimizer slot metadata"));
    }

    #[test]
    fn ep_global_checkpoint_manifest_rejects_wrong_shard_expert_dimensions() {
        let mut wrong_global = tiny_ep_global_manifest();
        wrong_global.ranks[0].shards[0].global_shape[0] = 3;

        let wrong_global_error = wrong_global
            .validate()
            .expect_err("wrong global expert dimension should fail")
            .to_string();
        assert!(wrong_global_error.contains("global expert dimension"));

        let mut wrong_shard = tiny_ep_global_manifest();
        wrong_shard.ranks[0].shards[0].shard_shape[0] = 1;

        let wrong_shard_error = wrong_shard
            .validate()
            .expect_err("wrong shard expert dimension should fail")
            .to_string();
        assert!(wrong_shard_error.contains("shard expert dimension"));
    }

    #[test]
    fn ep_global_checkpoint_manifest_rejects_partial_dataset_provenance() {
        let mut manifest = tiny_ep_global_manifest();
        manifest.dataset_fingerprint = "abc123".to_string();

        let error = manifest
            .validate()
            .expect_err("fingerprint without source files should fail")
            .to_string();
        assert!(error.contains("dataset_fingerprint requires dataset_source_files"));
    }

    #[test]
    fn ep_global_checkpoint_manifest_rejects_non_jsonl_dataset_sources() {
        let mut manifest = tiny_ep_global_manifest();
        manifest.dataset_source_files = vec!["data/train.txt".to_string()];
        manifest.dataset_fingerprint = "abc123".to_string();

        let error = manifest
            .validate()
            .expect_err("non-jsonl source should fail")
            .to_string();
        assert!(error.contains("must only contain JSONL paths"));
    }

    #[test]
    fn ep_global_checkpoint_manifest_rejects_inconsistent_data_progress() {
        let mut missing_epoch = tiny_ep_global_manifest();
        missing_epoch.data_cursor_next = Some(4);
        missing_epoch.data_sample_offset_next = Some(4);

        let missing_epoch_error = missing_epoch
            .validate()
            .expect_err("cursor without epoch should fail")
            .to_string();
        assert!(missing_epoch_error.contains("must be present together"));

        let mut mismatched_cursor = tiny_ep_global_manifest();
        mismatched_cursor.data_cursor_next = Some(3);
        mismatched_cursor.data_epoch_next = Some(0);
        mismatched_cursor.data_sample_offset_next = Some(3);

        let mismatched_cursor_error = mismatched_cursor
            .validate()
            .expect_err("cursor not matching consumed samples should fail")
            .to_string();
        assert!(mismatched_cursor_error.contains("must match consumed_samples"));
    }

    #[test]
    fn ep_global_checkpoint_manifest_validates_rank_owned_artifacts() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let manifest =
            tiny_ep_global_manifest_with_artifacts(temp.path()).expect("artifacts should write");

        manifest
            .validate_artifacts()
            .expect("rank-owned artifacts should validate");
    }

    #[test]
    fn ep_global_checkpoint_manifest_rejects_missing_model_artifact_tensor() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let mut manifest =
            tiny_ep_global_manifest_with_artifacts(temp.path()).expect("artifacts should write");
        manifest.ranks[0].shards[0].shard_name = "experts.missing_up.weight".to_string();

        let error = manifest
            .validate_artifacts()
            .expect_err("missing model shard should fail")
            .to_string();

        assert!(error.contains("missing model shard experts.missing_up.weight"));
    }

    #[test]
    fn ep_global_checkpoint_manifest_rejects_missing_optimizer_artifact_tensor() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let mut manifest =
            tiny_ep_global_manifest_with_artifacts(temp.path()).expect("artifacts should write");
        manifest.ranks[0].shards[0].optimizer_m_name = "experts.up.weight.missing_m".to_string();

        let error = manifest
            .validate_artifacts()
            .expect_err("missing optimizer slot should fail")
            .to_string();

        assert!(error.contains("missing optimizer m slot experts.up.weight.missing_m"));
    }

    #[test]
    fn ep_global_checkpoint_manifest_rejects_artifact_shape_mismatch() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let mut manifest =
            tiny_ep_global_manifest_with_artifacts(temp.path()).expect("artifacts should write");
        manifest.ranks[0].shards[0].shard_shape = vec![2, 3, 1];

        let error = manifest
            .validate_artifacts()
            .expect_err("artifact shape mismatch should fail")
            .to_string();

        assert!(error.contains("shape [2, 3, 2] does not match manifest shard_shape [2, 3, 1]"));
    }

    fn tiny_ep_global_manifest() -> ExpertParallelGlobalCheckpointManifest {
        ExpertParallelGlobalCheckpointManifest {
            format: "rustrain.ep_sharded.v1".to_string(),
            manifest_kind: "global".to_string(),
            base_model_path: "rustrain://focused-tch-moe-ep-smoke".to_string(),
            tokenizer_path: String::new(),
            global_step: 1,
            consumed_samples: 4,
            consumed_tokens: 4,
            data_cursor_next: None,
            data_epoch_next: None,
            data_sample_offset_next: None,
            data_train_samples: None,
            dataset_source_files: Vec::new(),
            dataset_source_sample_counts: Vec::new(),
            dataset_fingerprint: String::new(),
            dataset_shuffle: true,
            seed: 0,
            dtype: "float32".to_string(),
            optimizer: "adamw".to_string(),
            scheduler: "constant".to_string(),
            parallel: ExpertParallelGlobalParallelManifest {
                data_parallel_size: 1,
                tensor_model_parallel_size: 1,
                pipeline_model_parallel_size: 1,
                expert_model_parallel_size: 2,
                context_parallel_size: 1,
            },
            ranks: vec![
                tiny_ep_rank_manifest(0, 0, 2),
                tiny_ep_rank_manifest(1, 2, 4),
            ],
        }
    }

    fn tiny_ep_global_manifest_with_artifacts(
        root: &Path,
    ) -> Result<ExpertParallelGlobalCheckpointManifest> {
        let mut manifest = tiny_ep_global_manifest();
        for rank in &mut manifest.ranks {
            let rank_dir = root.join(format!("rank{}", rank.rank));
            fs::create_dir_all(&rank_dir)
                .with_context(|| format!("failed to create {}", rank_dir.display()))?;
            let model_safetensors = rank_dir.join("model.safetensors");
            let optimizer_safetensors = rank_dir.join("optimizer.safetensors");
            let model_entries = rank
                .shards
                .iter()
                .map(|shard| {
                    (
                        shard.shard_name.clone(),
                        Tensor::ones(shard.shard_shape.as_slice(), (Kind::Float, Device::Cpu)),
                    )
                })
                .collect::<Vec<_>>();
            let optimizer_entries = rank
                .shards
                .iter()
                .flat_map(|shard| {
                    [
                        (
                            shard.optimizer_m_name.clone(),
                            Tensor::zeros(shard.shard_shape.as_slice(), (Kind::Float, Device::Cpu)),
                        ),
                        (
                            shard.optimizer_v_name.clone(),
                            Tensor::zeros(shard.shard_shape.as_slice(), (Kind::Float, Device::Cpu)),
                        ),
                    ]
                })
                .collect::<Vec<_>>();
            let model_refs = model_entries
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor))
                .collect::<Vec<_>>();
            let optimizer_refs = optimizer_entries
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor))
                .collect::<Vec<_>>();
            Tensor::write_safetensors(&model_refs, &model_safetensors)
                .with_context(|| format!("failed to write {}", model_safetensors.display()))?;
            Tensor::write_safetensors(&optimizer_refs, &optimizer_safetensors)
                .with_context(|| format!("failed to write {}", optimizer_safetensors.display()))?;
            rank.model_safetensors = model_safetensors.display().to_string();
            rank.optimizer_safetensors = optimizer_safetensors.display().to_string();
        }
        Ok(manifest)
    }

    fn tiny_ep_rank_manifest(
        rank: usize,
        owned_expert_start: usize,
        owned_expert_end: usize,
    ) -> ExpertParallelCheckpointManifest {
        ExpertParallelCheckpointManifest {
            format: "rustrain.ep_sharded.v1".to_string(),
            manifest_kind: "rank".to_string(),
            rank,
            world_size: 2,
            local_rank: rank,
            global_step: 1,
            owned_expert_start,
            owned_expert_end,
            model_safetensors: format!("rank{rank}/model.safetensors"),
            optimizer_safetensors: format!("rank{rank}/optimizer.safetensors"),
            optimizer: "adamw".to_string(),
            learning_rate: 0.05,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            shards: vec![
                tiny_ep_checkpoint_shard(
                    format!("experts.{owned_expert_start}..{owned_expert_end}.up.weight"),
                    "experts.up.weight",
                    vec![4, 3, 2],
                    vec![2, 3, 2],
                ),
                tiny_ep_checkpoint_shard(
                    format!("experts.{owned_expert_start}..{owned_expert_end}.down.weight"),
                    "experts.down.weight",
                    vec![4, 2, 3],
                    vec![2, 2, 3],
                ),
            ],
        }
    }

    fn tiny_ep_checkpoint_shard(
        name: String,
        shard_name: &str,
        global_shape: Vec<i64>,
        shard_shape: Vec<i64>,
    ) -> ExpertParallelCheckpointShard {
        ExpertParallelCheckpointShard {
            name,
            shard_name: shard_name.to_string(),
            optimizer_m_name: format!("{shard_name}.adam_m"),
            optimizer_v_name: format!("{shard_name}.adam_v"),
            global_shape,
            shard_shape,
            dtype: "float32".to_string(),
            partition: "expert_model_parallel".to_string(),
        }
    }
}
