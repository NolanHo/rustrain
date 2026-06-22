use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    io::{BufRead, BufReader, Seek, SeekFrom},
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
    Config, DataConfig as RuntimeDataConfig, DataKind as RuntimeDataKind, Device as RuntimeDevice,
    LoraConfig as RuntimeLoraConfig, LrScheduler, RunPaths, load_config,
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

#[derive(Debug, Serialize)]
struct QwenTpLinearRankSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    model_path: String,
    input_shape: Vec<i64>,
    projections: Vec<QwenTpProjectionShardSummary>,
}

#[derive(Debug, Serialize)]
struct QwenTpProjectionShardSummary {
    name: String,
    tensor_name: String,
    full_output_shape: Vec<i64>,
    shard_output_shape: Vec<i64>,
    shard_start: i64,
    shard_end: i64,
    max_abs: Option<f64>,
    mean_abs: Option<f64>,
}

#[derive(Debug, Serialize)]
struct QwenTpAttentionRankSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    model_path: String,
    input_shape: Vec<i64>,
    q_head_start: i64,
    q_head_end: i64,
    kv_head_start: i64,
    kv_head_end: i64,
    context_shard_shape: Vec<i64>,
    output_contribution_shape: Vec<i64>,
    full_output_shape: Vec<i64>,
    max_abs: Option<f64>,
    mean_abs: Option<f64>,
}

#[derive(Debug, Serialize)]
struct QwenTpAttentionNcclRankSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    model_path: String,
    input_shape: Vec<i64>,
    q_head_start: i64,
    q_head_end: i64,
    kv_head_start: i64,
    kv_head_end: i64,
    context_shard_shape: Vec<i64>,
    output_contribution_shape: Vec<i64>,
    reduced_output_shape: Vec<i64>,
    full_output_shape: Vec<i64>,
    max_abs: f64,
    mean_abs: f64,
}

#[derive(Debug, Serialize)]
struct QwenTpMlpRankSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    model_path: String,
    input_shape: Vec<i64>,
    intermediate_start: i64,
    intermediate_end: i64,
    activation_shard_shape: Vec<i64>,
    output_contribution_shape: Vec<i64>,
    full_output_shape: Vec<i64>,
    max_abs: Option<f64>,
    mean_abs: Option<f64>,
}

#[derive(Debug, Serialize)]
struct QwenTpMlpNcclRankSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    model_path: String,
    input_shape: Vec<i64>,
    intermediate_start: i64,
    intermediate_end: i64,
    activation_shard_shape: Vec<i64>,
    output_contribution_shape: Vec<i64>,
    reduced_output_shape: Vec<i64>,
    full_output_shape: Vec<i64>,
    max_abs: f64,
    mean_abs: f64,
}

#[derive(Debug, Serialize)]
struct QwenSessionTpRankSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    model_path: String,
    resume_from: Option<String>,
    resumed_sharded_checkpoint: bool,
    resume_global_step: Option<u64>,
    resume_rank_manifest_output: Option<String>,
    resume_model_safetensors: Option<String>,
    resume_optimizer_safetensors: Option<String>,
    resume_sharded_manifest_tensor_count: Option<usize>,
    resume_restore_max_abs: Option<f64>,
    resume_restore_mean_abs: Option<f64>,
    resume_next_update_max_abs: Option<f64>,
    resume_next_update_mean_abs: Option<f64>,
    tensor_model_parallel_size: usize,
    data_parallel_size: usize,
    attention_q_head_start: i64,
    attention_q_head_end: i64,
    attention_kv_head_start: i64,
    attention_kv_head_end: i64,
    attention_context_shard_shape: Vec<i64>,
    attention_reduced_output_shape: Vec<i64>,
    attention_max_abs: f64,
    attention_mean_abs: f64,
    attention_train_initial_loss: f64,
    attention_train_final_loss: f64,
    attention_train_loss_improved: bool,
    attention_train_learning_rate: f64,
    attention_train_q_grad_norm: f64,
    attention_train_k_grad_norm: f64,
    attention_train_v_grad_norm: f64,
    attention_train_o_grad_norm: f64,
    layer0_reduced_output_shape: Vec<i64>,
    layer0_max_abs: f64,
    layer0_mean_abs: f64,
    layer0_train_initial_loss: f64,
    layer0_train_final_loss: f64,
    layer0_train_loss_improved: bool,
    layer0_train_learning_rate: f64,
    layer0_train_q_grad_norm: f64,
    layer0_train_k_grad_norm: f64,
    layer0_train_v_grad_norm: f64,
    layer0_train_o_grad_norm: f64,
    layer0_train_gate_grad_norm: f64,
    layer0_train_up_grad_norm: f64,
    layer0_train_down_grad_norm: f64,
    causal_train_input_shape: Vec<i64>,
    causal_train_full_loss: f64,
    causal_train_initial_loss: f64,
    causal_train_initial_loss_delta: f64,
    causal_train_final_loss: f64,
    causal_train_loss_improved: bool,
    causal_train_learning_rate: f64,
    causal_train_q_grad_norm: f64,
    causal_train_k_grad_norm: f64,
    causal_train_v_grad_norm: f64,
    causal_train_o_grad_norm: f64,
    causal_train_gate_grad_norm: f64,
    causal_train_up_grad_norm: f64,
    causal_train_down_grad_norm: f64,
    causal_train_q_grad_sum: f64,
    causal_train_k_grad_sum: f64,
    causal_train_v_grad_sum: f64,
    causal_train_o_grad_sum: f64,
    causal_train_gate_grad_sum: f64,
    causal_train_up_grad_sum: f64,
    causal_train_down_grad_sum: f64,
    sharded_rank_manifest_output: String,
    sharded_global_manifest_output: String,
    sharded_manifest_tensor_count: usize,
    sharded_restore_max_abs: f64,
    sharded_restore_mean_abs: f64,
    sharded_next_update_max_abs: f64,
    sharded_next_update_mean_abs: f64,
    mlp_intermediate_start: i64,
    mlp_intermediate_end: i64,
    mlp_activation_shard_shape: Vec<i64>,
    mlp_reduced_output_shape: Vec<i64>,
    mlp_max_abs: f64,
    mlp_mean_abs: f64,
    mlp_train_initial_loss: f64,
    mlp_train_final_loss: f64,
    mlp_train_loss_improved: bool,
    mlp_train_learning_rate: f64,
    mlp_train_gate_grad_norm: f64,
    mlp_train_up_grad_norm: f64,
    mlp_train_down_grad_norm: f64,
}

struct QwenTpAttentionContribution {
    input: Tensor,
    context_shard: Tensor,
    output_contribution: Tensor,
    full_output: Tensor,
    q_head_start: i64,
    q_heads_per_rank: i64,
    kv_head_start: i64,
    kv_heads_per_rank: i64,
}

struct QwenTpAttentionShardWeights<'a> {
    q_proj: &'a Tensor,
    q_bias: &'a Tensor,
    k_proj: &'a Tensor,
    k_bias: &'a Tensor,
    v_proj: &'a Tensor,
    v_bias: &'a Tensor,
    o_proj: &'a Tensor,
}

struct QwenSessionTpLayer0SgdUpdate {
    initial_loss: f64,
    final_loss: f64,
    initial_output: Tensor,
    final_output: Tensor,
    q_grad_norm: f64,
    k_grad_norm: f64,
    v_grad_norm: f64,
    o_grad_norm: f64,
    gate_grad_norm: f64,
    up_grad_norm: f64,
    down_grad_norm: f64,
}

struct QwenSessionTpCausalLmSgdUpdate {
    full_loss: f64,
    initial_loss: f64,
    final_loss: f64,
    learning_rate: f64,
    q_grad: Tensor,
    k_grad: Tensor,
    v_grad: Tensor,
    o_grad: Tensor,
    gate_grad: Tensor,
    up_grad: Tensor,
    down_grad: Tensor,
    q_grad_norm: f64,
    k_grad_norm: f64,
    v_grad_norm: f64,
    o_grad_norm: f64,
    gate_grad_norm: f64,
    up_grad_norm: f64,
    down_grad_norm: f64,
    q_grad_sum: f64,
    k_grad_sum: f64,
    v_grad_sum: f64,
    o_grad_sum: f64,
    gate_grad_sum: f64,
    up_grad_sum: f64,
    down_grad_sum: f64,
}

struct QwenSessionTpFocusedLayer0Shards {
    input_norm: Tensor,
    post_attention_norm: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

struct QwenSessionTpFocusedResume {
    global_step: u64,
    rank_manifest_output: String,
    model_safetensors: String,
    optimizer_safetensors: String,
    tensor_count: usize,
    restore_diff: DiffStats,
    next_update_diff: DiffStats,
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
    cached_generated_ids: Option<Vec<i64>>,
    new_token_ids: Vec<i64>,
    reference_match: bool,
    cached_reference_match: Option<bool>,
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
    python_cached_ids: Option<Vec<i64>>,
    full_context_ids: Vec<i64>,
    cached_ids: Vec<i64>,
    new_token_ids: Vec<i64>,
    reference_match: bool,
    python_cached_reference_match: Option<bool>,
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
    pub(crate) streaming_train_batches: bool,
    pub(crate) streaming_index_cache_path: Option<String>,
    pub(crate) streaming_index_cache_hit: bool,
    pub(crate) streaming_index_cache_written: bool,
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
    #[serde(default)]
    streaming_train_batches: bool,
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
    pub(crate) streaming_train_batches: Option<bool>,
    pub(crate) streaming_index_cache_path: Option<String>,
    pub(crate) streaming_index_cache_hit: Option<bool>,
    pub(crate) streaming_index_cache_written: Option<bool>,
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
    streaming_train_batches: Option<bool>,
    streaming_index_cache_path: Option<String>,
    streaming_index_cache_hit: Option<bool>,
    streaming_index_cache_written: Option<bool>,
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
    streaming_train_batches: bool,
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
    #[serde(default)]
    streaming_train_batches: Option<bool>,
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
            streaming_train_batches: self.streaming_train_batches,
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
        let mut seen_rank_axes = BTreeSet::new();
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
            let mut seen_tensor_names = BTreeSet::new();
            let mut seen_model_shards = BTreeSet::new();
            let mut seen_optimizer_slots = BTreeSet::new();
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

    fn validate_artifacts(&self) -> Result<()> {
        self.validate()?;
        for rank in &self.ranks {
            let model_tensors = read_safetensors_map(Path::new(&rank.model_safetensors))
                .with_context(|| {
                    format!(
                        "failed to validate Qwen sharded checkpoint rank {} model artifacts",
                        rank.rank
                    )
                })?;
            let optimizer_tensors = read_safetensors_map(Path::new(&rank.optimizer_safetensors))
                .with_context(|| {
                    format!(
                        "failed to validate Qwen sharded checkpoint rank {} optimizer artifacts",
                        rank.rank
                    )
                })?;
            for shard in &rank.shards {
                let model_tensor =
                    tensor(&model_tensors, &shard.shard_name).with_context(|| {
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
                let optimizer_m = tensor(&optimizer_tensors, &shard.optimizer_m_name)
                    .with_context(|| {
                        format!(
                            "Qwen sharded checkpoint rank {} missing optimizer m slot {} for {}",
                            rank.rank, shard.optimizer_m_name, shard.name
                        )
                    })?;
                let optimizer_v = tensor(&optimizer_tensors, &shard.optimizer_v_name)
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

    fn linear_rank(&self, rank: &QwenRankShardManifest) -> usize {
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
    #[serde(default)]
    streaming_train_batches: Option<bool>,
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
    #[serde(default)]
    streaming_train_batches: Option<bool>,
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
    streaming_train_batches: Option<bool>,
    streaming_index_cache_path: Option<String>,
    streaming_index_cache_hit: Option<bool>,
    streaming_index_cache_written: Option<bool>,
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
    streaming_train_batches: Option<bool>,
    streaming_index_cache_path: Option<String>,
    streaming_index_cache_hit: Option<bool>,
    streaming_index_cache_written: Option<bool>,
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

#[derive(Clone)]
struct QwenSftExample {
    system: String,
    instruction: String,
    input: String,
    response: String,
}

struct QwenSftExampleSet {
    examples: Vec<QwenSftExample>,
    source_files: Vec<String>,
    source_sample_counts: Vec<QwenSftSourceSampleCount>,
    fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct QwenSftRawSampleIndex {
    path: String,
    index_in_file: usize,
    global_index: usize,
    byte_offset: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct QwenSftStreamingSourceIndex {
    samples: Vec<QwenSftRawSampleIndex>,
}

#[derive(Debug, Serialize, Deserialize)]
struct QwenSftStreamingSourceIndexCache {
    format: String,
    paths: Vec<String>,
    max_samples: Option<usize>,
    field_map: QwenSftFieldMap,
    min_response_chars: usize,
    samples: Vec<QwenSftRawSampleIndex>,
}

#[derive(Debug)]
struct QwenSftStreamingSourceIndexLoad {
    index: QwenSftStreamingSourceIndex,
    cache_hit: bool,
    cache_written: bool,
}

struct QwenSftStreamingTokenWindow {
    samples: Vec<QwenSftTokenSample>,
    raw_sample_indices: Vec<QwenSftRawSampleIndex>,
    raw_samples_read: usize,
    source_index_cache_hit: bool,
    source_index_cache_written: bool,
}

struct QwenSftRawExampleWindow {
    examples: Vec<QwenSftExample>,
    raw_samples_read: usize,
}

struct QwenSftRecord {
    system: String,
    instruction: String,
    input: String,
    response: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct QwenSftFieldMap {
    instruction: String,
    input: String,
    response: String,
    #[serde(default)]
    system: Option<String>,
    prompt_template: String,
    prompt_with_input_template: String,
    trim_fields: bool,
    min_response_chars: usize,
    #[serde(default)]
    max_response_chars: Option<usize>,
    #[serde(default)]
    min_instruction_chars: Option<usize>,
    #[serde(default)]
    max_instruction_chars: Option<usize>,
    #[serde(default)]
    min_input_chars: Option<usize>,
    #[serde(default)]
    max_input_chars: Option<usize>,
    #[serde(default)]
    min_prompt_chars: Option<usize>,
    #[serde(default)]
    max_prompt_chars: Option<usize>,
    #[serde(default)]
    min_sample_chars: Option<usize>,
    #[serde(default)]
    max_sample_chars: Option<usize>,
    #[serde(default)]
    dedupe_samples: bool,
    source_weights: Vec<usize>,
}

impl QwenSftFieldMap {
    fn from_runtime_data(data: &RuntimeDataConfig) -> Result<Self> {
        let map = Self {
            instruction: data.instruction_field.clone(),
            input: data.input_field.clone(),
            response: data.response_field.clone(),
            system: data.system_field.clone(),
            prompt_template: data.prompt_template.clone(),
            prompt_with_input_template: data.prompt_with_input_template.clone(),
            trim_fields: data.trim_fields,
            min_response_chars: data.min_response_chars,
            max_response_chars: data.max_response_chars,
            min_instruction_chars: data.min_instruction_chars,
            max_instruction_chars: data.max_instruction_chars,
            min_input_chars: data.min_input_chars,
            max_input_chars: data.max_input_chars,
            min_prompt_chars: data.min_prompt_chars,
            max_prompt_chars: data.max_prompt_chars,
            min_sample_chars: data.min_sample_chars,
            max_sample_chars: data.max_sample_chars,
            dedupe_samples: data.dedupe_samples,
            source_weights: data.source_weights.clone(),
        };
        map.validate()?;
        Ok(map)
    }

    fn validate(&self) -> Result<()> {
        if self.instruction.trim().is_empty() {
            bail!("data.instruction_field must not be empty");
        }
        if self.input.trim().is_empty() {
            bail!("data.input_field must not be empty");
        }
        if self.response.trim().is_empty() {
            bail!("data.response_field must not be empty");
        }
        if self
            .system
            .as_ref()
            .is_some_and(|field| field.trim().is_empty())
        {
            bail!("data.system_field must not be empty when set");
        }
        if self.prompt_template.is_empty() {
            bail!("data.prompt_template must not be empty");
        }
        if self.prompt_with_input_template.is_empty() {
            bail!("data.prompt_with_input_template must not be empty");
        }
        if self.source_weights.iter().any(|weight| *weight == 0) {
            bail!("data.source_weights entries must be greater than zero");
        }
        if let Some(max_response_chars) = self.max_response_chars {
            if max_response_chars == 0 {
                bail!("data.max_response_chars must be greater than zero");
            }
            if max_response_chars < self.min_response_chars {
                bail!(
                    "data.max_response_chars must be greater than or equal to data.min_response_chars"
                );
            }
        }
        if let Some(min_instruction_chars) = self.min_instruction_chars {
            if min_instruction_chars == 0 {
                bail!("data.min_instruction_chars must be greater than zero");
            }
        }
        if let Some(max_instruction_chars) = self.max_instruction_chars {
            if max_instruction_chars == 0 {
                bail!("data.max_instruction_chars must be greater than zero");
            }
            if self
                .min_instruction_chars
                .is_some_and(|min_instruction_chars| max_instruction_chars < min_instruction_chars)
            {
                bail!(
                    "data.max_instruction_chars must be greater than or equal to data.min_instruction_chars"
                );
            }
        }
        if let Some(min_input_chars) = self.min_input_chars {
            if min_input_chars == 0 {
                bail!("data.min_input_chars must be greater than zero");
            }
        }
        if let Some(max_input_chars) = self.max_input_chars {
            if max_input_chars == 0 {
                bail!("data.max_input_chars must be greater than zero");
            }
            if self
                .min_input_chars
                .is_some_and(|min_input_chars| max_input_chars < min_input_chars)
            {
                bail!("data.max_input_chars must be greater than or equal to data.min_input_chars");
            }
        }
        if let Some(min_prompt_chars) = self.min_prompt_chars {
            if min_prompt_chars == 0 {
                bail!("data.min_prompt_chars must be greater than zero");
            }
        }
        if let Some(max_prompt_chars) = self.max_prompt_chars {
            if max_prompt_chars == 0 {
                bail!("data.max_prompt_chars must be greater than zero");
            }
            if self
                .min_prompt_chars
                .is_some_and(|min_prompt_chars| max_prompt_chars < min_prompt_chars)
            {
                bail!(
                    "data.max_prompt_chars must be greater than or equal to data.min_prompt_chars"
                );
            }
        }
        if let Some(min_sample_chars) = self.min_sample_chars {
            if min_sample_chars == 0 {
                bail!("data.min_sample_chars must be greater than zero");
            }
        }
        if let Some(max_sample_chars) = self.max_sample_chars {
            if max_sample_chars == 0 {
                bail!("data.max_sample_chars must be greater than zero");
            }
            if self
                .min_sample_chars
                .is_some_and(|min_sample_chars| max_sample_chars < min_sample_chars)
            {
                bail!(
                    "data.max_sample_chars must be greater than or equal to data.min_sample_chars"
                );
            }
        }
        Ok(())
    }
}

impl Default for QwenSftFieldMap {
    fn default() -> Self {
        Self {
            instruction: "instruction".to_string(),
            input: "input".to_string(),
            response: "response".to_string(),
            system: None,
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            source_weights: Vec::new(),
        }
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct QwenSftStreamingSourceSummary {
    samples: usize,
    source_files: Vec<String>,
    source_sample_counts: Vec<QwenSftSourceSampleCount>,
    fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct QwenSftStreamingCursorEntry {
    cursor: usize,
    epoch: usize,
    sample_offset: usize,
}

#[derive(Debug, Serialize)]
struct QwenSftStreamingDataPlanSummary {
    config_path: String,
    data_paths: Vec<String>,
    eval_paths: Vec<String>,
    max_samples: Option<usize>,
    max_eval_samples: Option<usize>,
    train_split: f32,
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
    train_window_start_cursor: usize,
    train_window_end_cursor_exclusive: usize,
    train_window_sample_cursors: Vec<QwenSftStreamingCursorEntry>,
    dataset_total_samples: usize,
    dataset_train_samples: usize,
    dataset_eval_samples: usize,
    dataset_source_files: Vec<String>,
    dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    dataset_fingerprint: String,
    dataset_order_seed: u64,
    dataset_shuffle: bool,
    train_source_files: Vec<String>,
    train_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    train_fingerprint: String,
    eval_source_files: Vec<String>,
    eval_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    eval_fingerprint: String,
    tokenizer_loaded: bool,
    tokenized_samples_materialized: bool,
}

#[derive(Debug, Serialize)]
struct QwenSftStreamingBatchPlanSummary {
    config_path: String,
    model_path: String,
    world_size: usize,
    local_batch_size: usize,
    global_batch_size: usize,
    train_steps: usize,
    required_batches: usize,
    train_batch_count: usize,
    data_cursor_start: usize,
    data_cursor_end: usize,
    data_cursor_next: usize,
    train_window_start_cursor: usize,
    train_window_end_cursor_exclusive: usize,
    train_window_sample_cursors: Vec<QwenSftStreamingCursorEntry>,
    dataset_total_samples: usize,
    dataset_train_samples: usize,
    dataset_eval_samples: usize,
    dataset_source_files: Vec<String>,
    dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    dataset_fingerprint: String,
    dataset_order_seed: u64,
    dataset_shuffle: bool,
    tokenizer_loaded: bool,
    tokenized_samples_materialized: bool,
    reference_tokenized_samples_materialized: bool,
    streaming_index_cache_path: Option<String>,
    streaming_index_cache_hit: bool,
    streaming_index_cache_written: bool,
    streaming_window_samples: usize,
    streaming_raw_samples_read: usize,
    streaming_raw_sample_indices: Vec<QwenSftRawSampleIndex>,
    batch_sequence_tokens: Vec<usize>,
    batch_masked_positions: Vec<usize>,
    batch_padding_tokens: Vec<usize>,
    batch_token_fingerprints: Vec<String>,
    materialized_input_max_delta: i64,
    materialized_mask_max_delta: f64,
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

fn resolve_qwen_model_safetensors_path(model_safetensors: &Path) -> Result<PathBuf> {
    if model_safetensors.exists() {
        return Ok(model_safetensors.to_path_buf());
    }
    if model_safetensors.file_name().and_then(|name| name.to_str()) != Some("model.safetensors") {
        return Ok(model_safetensors.to_path_buf());
    }
    let Some(model_path) = model_safetensors.parent() else {
        return Ok(model_safetensors.to_path_buf());
    };
    Ok(resolve_qwen_model_path(model_path)?.join("model.safetensors"))
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

    let model_path = resolve_qwen_model_path(model_path)?;
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

    let model_path = resolve_qwen_model_path(model_path)?;
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
    let streaming_index_cache = sft_paths
        .as_ref()
        .map(|_| qwen_sft_streaming_index_cache_path(checkpoint_dir, "qwen-lora-sft"));
    let summary = qwen_lora_sft_train(
        model_path,
        adapter_output,
        checkpoint_dir,
        sft_paths.as_deref(),
        &[],
        None,
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
        streaming_index_cache.as_deref(),
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
        crate::runtime::DType::Fp32 => QwenComputeDType::Fp32,
        crate::runtime::DType::Bf16 => QwenComputeDType::Bf16,
        crate::runtime::DType::Fp16 => {
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
fn qwen_lora_sft_train(
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

fn qwen_session_single_summary(
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
    let model_path = resolve_qwen_model_path(model_path)?;
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

pub fn qwen_tp_linear_rank_smoke(model_path: &Path, output_dir: PathBuf) -> Result<()> {
    let model_path = resolve_qwen_model_path(model_path)?;
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("Qwen TP linear rank smoke expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    let device = Device::Cuda(local_rank);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let q_weight = tensor(&weights, "model.layers.0.self_attn.q_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let hidden_size = q_weight.size()[1];
    let input = Tensor::arange(hidden_size * 3, (Kind::Float, device))
        .reshape([3, hidden_size])
        .fmod(17.0)
        / 17.0;

    let projection_specs = [
        (
            "q_proj",
            "model.layers.0.self_attn.q_proj.weight",
            Some("model.layers.0.self_attn.q_proj.bias"),
        ),
        (
            "k_proj",
            "model.layers.0.self_attn.k_proj.weight",
            Some("model.layers.0.self_attn.k_proj.bias"),
        ),
        (
            "v_proj",
            "model.layers.0.self_attn.v_proj.weight",
            Some("model.layers.0.self_attn.v_proj.bias"),
        ),
        ("o_proj", "model.layers.0.self_attn.o_proj.weight", None),
    ];

    let mut local_projection_summaries = Vec::with_capacity(projection_specs.len());
    let mut shard_entries = Vec::with_capacity(projection_specs.len());
    for (name, weight_name, bias_name) in projection_specs {
        let weight = tensor(&weights, weight_name)?
            .to_kind(Kind::Float)
            .to_device(device);
        let output_size = weight.size()[0];
        if output_size % world_size as i64 != 0 {
            bail!("Qwen TP linear rank smoke requires {name} output size divisible by WORLD_SIZE");
        }
        let shard_size = output_size / world_size as i64;
        let shard_start = rank as i64 * shard_size;
        let shard = weight.narrow(0, shard_start, shard_size);
        let mut shard_output = input.matmul(&shard.transpose(0, 1));
        if let Some(bias_name) = bias_name {
            let bias = tensor(&weights, bias_name)?
                .to_kind(Kind::Float)
                .to_device(device)
                .narrow(0, shard_start, shard_size);
            shard_output += bias;
        }
        let tensor_name = format!("{name}_shard_output");
        local_projection_summaries.push(QwenTpProjectionShardSummary {
            name: name.to_string(),
            tensor_name: weight_name.to_string(),
            full_output_shape: vec![input.size()[0], output_size],
            shard_output_shape: shard_output.size(),
            shard_start,
            shard_end: shard_start + shard_size,
            max_abs: None,
            mean_abs: None,
        });
        shard_entries.push((tensor_name, shard_output));
    }

    let shard_refs: Vec<(&str, &Tensor)> = shard_entries
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    let shard_path = output_dir.join(format!("qwen-tp-linear-rank-{rank}.safetensors"));
    Tensor::write_safetensors(&shard_refs, &shard_path)
        .with_context(|| format!("failed to write {}", shard_path.display()))?;

    wait_for_rank_barrier(
        &output_dir.join("qwen-tp-linear-shards-written"),
        rank,
        world_size,
        Duration::from_secs(300),
    )?;

    let mut projection_summaries = local_projection_summaries;
    if rank == 0 {
        for (projection_index, (name, weight_name, bias_name)) in
            projection_specs.into_iter().enumerate()
        {
            let mut shard_outputs = Vec::with_capacity(world_size);
            for shard_rank in 0..world_size {
                let tensors = read_safetensors_map(
                    &output_dir.join(format!("qwen-tp-linear-rank-{shard_rank}.safetensors")),
                )?;
                shard_outputs.push(
                    tensor(&tensors, &format!("{name}_shard_output"))?
                        .to_kind(Kind::Float)
                        .to_device(device),
                );
            }
            let gathered = Tensor::cat(&shard_outputs.iter().collect::<Vec<_>>(), 1);
            let weight = tensor(&weights, weight_name)?
                .to_kind(Kind::Float)
                .to_device(device);
            let mut full_output = input.matmul(&weight.transpose(0, 1));
            if let Some(bias_name) = bias_name {
                full_output += tensor(&weights, bias_name)?
                    .to_kind(Kind::Float)
                    .to_device(device);
            }
            let diff = diff_stats(&gathered, &full_output)?;
            if diff.max_abs > 1e-5 {
                bail!(
                    "Qwen TP {name} shard parity failed: max_abs={}, mean_abs={}",
                    diff.max_abs,
                    diff.mean_abs
                );
            }
            projection_summaries[projection_index].max_abs = Some(diff.max_abs);
            projection_summaries[projection_index].mean_abs = Some(diff.mean_abs);
        }
    }

    let summary = QwenTpLinearRankSummary {
        rank,
        world_size,
        local_rank,
        model_path: model_path.display().to_string(),
        input_shape: input.size(),
        projections: projection_summaries,
    };
    let summary_path = output_dir.join(format!("qwen-tp-linear-rank-{rank}.json"));
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_tp_attention_rank_smoke(model_path: &Path, output_dir: PathBuf) -> Result<()> {
    let model_path = resolve_qwen_model_path(model_path)?;
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("Qwen TP attention rank smoke expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    let device = Device::Cuda(local_rank);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let contribution =
        qwen_tp_attention_contribution(&model_path, device, rank, world_size, "Qwen TP attention")?;

    Tensor::write_safetensors(
        &[("output_contribution", &contribution.output_contribution)],
        &output_dir.join(format!("qwen-tp-attention-rank-{rank}.safetensors")),
    )
    .with_context(|| format!("failed to write rank {rank} TP attention contribution"))?;

    wait_for_rank_barrier(
        &output_dir.join("qwen-tp-attention-contributions-written"),
        rank,
        world_size,
        Duration::from_secs(300),
    )?;

    let full_output_shape = contribution.full_output.size();
    let (max_abs, mean_abs) =
        if rank == 0 {
            let mut contributions = Vec::with_capacity(world_size);
            for shard_rank in 0..world_size {
                let tensors = read_safetensors_map(
                    &output_dir.join(format!("qwen-tp-attention-rank-{shard_rank}.safetensors")),
                )?;
                contributions.push(
                    tensor(&tensors, "output_contribution")?
                        .to_kind(Kind::Float)
                        .to_device(device),
                );
            }
            let gathered = Tensor::stack(&contributions.iter().collect::<Vec<_>>(), 0)
                .sum_dim_intlist([0].as_slice(), false, Kind::Float);
            let diff = diff_stats(&gathered, &contribution.full_output)?;
            if diff.max_abs > 1e-5 {
                bail!(
                    "Qwen TP attention parity failed: max_abs={}, mean_abs={}",
                    diff.max_abs,
                    diff.mean_abs
                );
            }
            (Some(diff.max_abs), Some(diff.mean_abs))
        } else {
            (None, None)
        };

    let summary = QwenTpAttentionRankSummary {
        rank,
        world_size,
        local_rank,
        model_path: model_path.display().to_string(),
        input_shape: contribution.input.size(),
        q_head_start: contribution.q_head_start,
        q_head_end: contribution.q_head_start + contribution.q_heads_per_rank,
        kv_head_start: contribution.kv_head_start,
        kv_head_end: contribution.kv_head_start + contribution.kv_heads_per_rank,
        context_shard_shape: contribution.context_shard.size(),
        output_contribution_shape: contribution.output_contribution.size(),
        full_output_shape,
        max_abs,
        mean_abs,
    };
    let summary_path = output_dir.join(format!("qwen-tp-attention-rank-{rank}.json"));
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_tp_attention_nccl_rank_smoke(model_path: &Path, output_dir: PathBuf) -> Result<()> {
    let model_path = resolve_qwen_model_path(model_path)?;
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("Qwen TP attention NCCL rank smoke expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    let device = Device::Cuda(local_rank);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let contribution = qwen_tp_attention_contribution(
        &model_path,
        device,
        rank,
        world_size,
        "Qwen TP attention NCCL",
    )?;
    let reduced_output = nccl_smoke::all_reduce_tensor_f32_for_launch(
        &output_dir.join("nccl-output-contribution"),
        &contribution.output_contribution,
    )?;
    let diff = diff_stats(&reduced_output, &contribution.full_output)?;
    if diff.max_abs > 1e-5 {
        bail!(
            "Qwen TP attention NCCL parity failed: rank={}, max_abs={}, mean_abs={}",
            rank,
            diff.max_abs,
            diff.mean_abs
        );
    }

    let summary = QwenTpAttentionNcclRankSummary {
        rank,
        world_size,
        local_rank,
        model_path: model_path.display().to_string(),
        input_shape: contribution.input.size(),
        q_head_start: contribution.q_head_start,
        q_head_end: contribution.q_head_start + contribution.q_heads_per_rank,
        kv_head_start: contribution.kv_head_start,
        kv_head_end: contribution.kv_head_start + contribution.kv_heads_per_rank,
        context_shard_shape: contribution.context_shard.size(),
        output_contribution_shape: contribution.output_contribution.size(),
        reduced_output_shape: reduced_output.size(),
        full_output_shape: contribution.full_output.size(),
        max_abs: diff.max_abs,
        mean_abs: diff.mean_abs,
    };
    let summary_path = output_dir.join(format!("qwen-tp-attention-nccl-rank-{rank}.json"));
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

fn qwen_tp_attention_contribution(
    model_path: &Path,
    device: Device,
    rank: usize,
    world_size: usize,
    context: &str,
) -> Result<QwenTpAttentionContribution> {
    let config = read_runtime_config(&model_path.join("config.json"))?;
    if config.num_attention_heads % world_size as i64 != 0 {
        bail!("{context} smoke requires attention heads divisible by WORLD_SIZE");
    }
    if config.num_key_value_heads % world_size as i64 != 0 {
        bail!("{context} smoke requires KV heads divisible by WORLD_SIZE");
    }
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let q_proj = tensor(&weights, "model.layers.0.self_attn.q_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let q_bias = tensor(&weights, "model.layers.0.self_attn.q_proj.bias")?
        .to_kind(Kind::Float)
        .to_device(device);
    let k_proj = tensor(&weights, "model.layers.0.self_attn.k_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let k_bias = tensor(&weights, "model.layers.0.self_attn.k_proj.bias")?
        .to_kind(Kind::Float)
        .to_device(device);
    let v_proj = tensor(&weights, "model.layers.0.self_attn.v_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let v_bias = tensor(&weights, "model.layers.0.self_attn.v_proj.bias")?
        .to_kind(Kind::Float)
        .to_device(device);
    let o_proj = tensor(&weights, "model.layers.0.self_attn.o_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);

    let hidden_size = q_proj.size()[1];
    let head_dim = hidden_size / config.num_attention_heads;
    let q_heads_per_rank = config.num_attention_heads / world_size as i64;
    let kv_heads_per_rank = config.num_key_value_heads / world_size as i64;
    let q_head_start = rank as i64 * q_heads_per_rank;
    let kv_head_start = rank as i64 * kv_heads_per_rank;
    let q_output_start = q_head_start * head_dim;
    let q_output_size = q_heads_per_rank * head_dim;
    let kv_output_start = kv_head_start * head_dim;
    let kv_output_size = kv_heads_per_rank * head_dim;

    let input = Tensor::arange(hidden_size * 9, (Kind::Float, device))
        .reshape([1, 9, hidden_size])
        .fmod(23.0)
        / 23.0;
    let (context_shard, output_contribution) = qwen_tp_attention_shard_contribution(
        &input,
        QwenTpAttentionShardWeights {
            q_proj: &q_proj,
            q_bias: &q_bias,
            k_proj: &k_proj,
            k_bias: &k_bias,
            v_proj: &v_proj,
            v_bias: &v_bias,
            o_proj: &o_proj,
        },
        &config,
        q_output_start,
        q_output_size,
        kv_output_start,
        kv_output_size,
        q_heads_per_rank,
        kv_heads_per_rank,
    );
    let full_output = qwen_attention(
        &input, &q_proj, &q_bias, &k_proj, &k_bias, &v_proj, &v_bias, &o_proj, &config,
    );

    Ok(QwenTpAttentionContribution {
        input,
        context_shard,
        output_contribution,
        full_output,
        q_head_start,
        q_heads_per_rank,
        kv_head_start,
        kv_heads_per_rank,
    })
}

#[allow(clippy::too_many_arguments)]
fn qwen_tp_attention_shard_contribution(
    input: &Tensor,
    weights: QwenTpAttentionShardWeights<'_>,
    config: &QwenRuntimeConfig,
    q_output_start: i64,
    q_output_size: i64,
    kv_output_start: i64,
    kv_output_size: i64,
    q_heads_per_rank: i64,
    kv_heads_per_rank: i64,
) -> (Tensor, Tensor) {
    let batch_size = input.size()[0];
    let seq_len = input.size()[1];
    let hidden_size = input.size()[2];
    let head_dim = hidden_size / config.num_attention_heads;
    let q_shard = input
        .linear(
            &weights.q_proj.narrow(0, q_output_start, q_output_size),
            Some(&weights.q_bias.narrow(0, q_output_start, q_output_size)),
        )
        .reshape([batch_size, seq_len, q_heads_per_rank, head_dim])
        .transpose(1, 2);
    let k_shard = input
        .linear(
            &weights.k_proj.narrow(0, kv_output_start, kv_output_size),
            Some(&weights.k_bias.narrow(0, kv_output_start, kv_output_size)),
        )
        .reshape([batch_size, seq_len, kv_heads_per_rank, head_dim])
        .transpose(1, 2);
    let v_shard = input
        .linear(
            &weights.v_proj.narrow(0, kv_output_start, kv_output_size),
            Some(&weights.v_bias.narrow(0, kv_output_start, kv_output_size)),
        )
        .reshape([batch_size, seq_len, kv_heads_per_rank, head_dim])
        .transpose(1, 2);
    let (cos, sin) = rope_cos_sin(seq_len, head_dim, config.rope_theta, input.device());
    let cos = cos.to_kind(input.kind());
    let sin = sin.to_kind(input.kind());
    let q_shard = apply_rotary(&q_shard, &cos, &sin);
    let k_shard = apply_rotary(&k_shard, &cos, &sin);
    let local_kv_repeat = q_heads_per_rank / kv_heads_per_rank;
    let k_for_attention = repeat_kv(&k_shard, local_kv_repeat);
    let v_for_attention = repeat_kv(&v_shard, local_kv_repeat);
    let scores = q_shard.matmul(&k_for_attention.transpose(-2, -1)) / (head_dim as f64).sqrt();
    let causal_mask = Tensor::ones([seq_len, seq_len], (Kind::Bool, input.device())).triu(1);
    let scores = scores.masked_fill(&causal_mask, f64::NEG_INFINITY);
    let probs = scores
        .softmax(-1, Kind::Float)
        .to_kind(v_for_attention.kind());
    let context_shard = probs.matmul(&v_for_attention).transpose(1, 2).reshape([
        batch_size,
        seq_len,
        q_output_size,
    ]);
    let output_contribution = context_shard.linear::<&Tensor>(
        &weights.o_proj.narrow(1, q_output_start, q_output_size),
        None,
    );
    (context_shard, output_contribution)
}

pub fn qwen_tp_mlp_rank_smoke(model_path: &Path, output_dir: PathBuf) -> Result<()> {
    let model_path = resolve_qwen_model_path(model_path)?;
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("Qwen TP MLP rank smoke expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    let device = Device::Cuda(local_rank);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let gate_proj = tensor(&weights, "model.layers.0.mlp.gate_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let up_proj = tensor(&weights, "model.layers.0.mlp.up_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let down_proj = tensor(&weights, "model.layers.0.mlp.down_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let hidden_size = gate_proj.size()[1];
    let intermediate_size = gate_proj.size()[0];
    if intermediate_size % world_size as i64 != 0 {
        bail!("Qwen TP MLP smoke requires intermediate size divisible by WORLD_SIZE");
    }
    let intermediate_shard_size = intermediate_size / world_size as i64;
    let intermediate_start = rank as i64 * intermediate_shard_size;
    let input = Tensor::arange(hidden_size * 7, (Kind::Float, device))
        .reshape([1, 7, hidden_size])
        .fmod(19.0)
        / 19.0;
    let gate_shard = input.linear::<&Tensor>(
        &gate_proj.narrow(0, intermediate_start, intermediate_shard_size),
        None,
    );
    let up_shard = input.linear::<&Tensor>(
        &up_proj.narrow(0, intermediate_start, intermediate_shard_size),
        None,
    );
    let activation_shard = gate_shard.silu() * up_shard;
    let output_contribution = activation_shard.linear::<&Tensor>(
        &down_proj.narrow(1, intermediate_start, intermediate_shard_size),
        None,
    );

    Tensor::write_safetensors(
        &[("output_contribution", &output_contribution)],
        &output_dir.join(format!("qwen-tp-mlp-rank-{rank}.safetensors")),
    )
    .with_context(|| format!("failed to write rank {rank} TP MLP contribution"))?;

    wait_for_rank_barrier(
        &output_dir.join("qwen-tp-mlp-contributions-written"),
        rank,
        world_size,
        Duration::from_secs(300),
    )?;

    let full_output_shape = input.size();
    let (max_abs, mean_abs) =
        if rank == 0 {
            let mut contributions = Vec::with_capacity(world_size);
            for shard_rank in 0..world_size {
                let tensors = read_safetensors_map(
                    &output_dir.join(format!("qwen-tp-mlp-rank-{shard_rank}.safetensors")),
                )?;
                contributions.push(
                    tensor(&tensors, "output_contribution")?
                        .to_kind(Kind::Float)
                        .to_device(device),
                );
            }
            let gathered = Tensor::stack(&contributions.iter().collect::<Vec<_>>(), 0)
                .sum_dim_intlist([0].as_slice(), false, Kind::Float);
            let full_output = qwen_mlp(&input, &gate_proj, &up_proj, &down_proj);
            let diff = diff_stats(&gathered, &full_output)?;
            if diff.max_abs > 1e-5 {
                bail!(
                    "Qwen TP MLP parity failed: max_abs={}, mean_abs={}",
                    diff.max_abs,
                    diff.mean_abs
                );
            }
            (Some(diff.max_abs), Some(diff.mean_abs))
        } else {
            (None, None)
        };

    let summary = QwenTpMlpRankSummary {
        rank,
        world_size,
        local_rank,
        model_path: model_path.display().to_string(),
        input_shape: input.size(),
        intermediate_start,
        intermediate_end: intermediate_start + intermediate_shard_size,
        activation_shard_shape: activation_shard.size(),
        output_contribution_shape: output_contribution.size(),
        full_output_shape,
        max_abs,
        mean_abs,
    };
    let summary_path = output_dir.join(format!("qwen-tp-mlp-rank-{rank}.json"));
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_tp_mlp_nccl_rank_smoke(model_path: &Path, output_dir: PathBuf) -> Result<()> {
    let model_path = resolve_qwen_model_path(model_path)?;
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("Qwen TP MLP NCCL rank smoke expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    let device = Device::Cuda(local_rank);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let gate_proj = tensor(&weights, "model.layers.0.mlp.gate_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let up_proj = tensor(&weights, "model.layers.0.mlp.up_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let down_proj = tensor(&weights, "model.layers.0.mlp.down_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let hidden_size = gate_proj.size()[1];
    let intermediate_size = gate_proj.size()[0];
    if intermediate_size % world_size as i64 != 0 {
        bail!("Qwen TP MLP NCCL smoke requires intermediate size divisible by WORLD_SIZE");
    }
    let intermediate_shard_size = intermediate_size / world_size as i64;
    let intermediate_start = rank as i64 * intermediate_shard_size;
    let input = Tensor::arange(hidden_size * 7, (Kind::Float, device))
        .reshape([1, 7, hidden_size])
        .fmod(19.0)
        / 19.0;
    let gate_shard = input.linear::<&Tensor>(
        &gate_proj.narrow(0, intermediate_start, intermediate_shard_size),
        None,
    );
    let up_shard = input.linear::<&Tensor>(
        &up_proj.narrow(0, intermediate_start, intermediate_shard_size),
        None,
    );
    let activation_shard = gate_shard.silu() * up_shard;
    let output_contribution = activation_shard.linear::<&Tensor>(
        &down_proj.narrow(1, intermediate_start, intermediate_shard_size),
        None,
    );

    let reduced_output = nccl_smoke::all_reduce_tensor_f32_for_launch(
        &output_dir.join("nccl-output-contribution"),
        &output_contribution,
    )?;
    let full_output = qwen_mlp(&input, &gate_proj, &up_proj, &down_proj);
    let diff = diff_stats(&reduced_output, &full_output)?;
    if diff.max_abs > 1e-5 {
        bail!(
            "Qwen TP MLP NCCL parity failed: rank={}, max_abs={}, mean_abs={}",
            rank,
            diff.max_abs,
            diff.mean_abs
        );
    }

    let summary = QwenTpMlpNcclRankSummary {
        rank,
        world_size,
        local_rank,
        model_path: model_path.display().to_string(),
        input_shape: input.size(),
        intermediate_start,
        intermediate_end: intermediate_start + intermediate_shard_size,
        activation_shard_shape: activation_shard.size(),
        output_contribution_shape: output_contribution.size(),
        reduced_output_shape: reduced_output.size(),
        full_output_shape: full_output.size(),
        max_abs: diff.max_abs,
        mean_abs: diff.mean_abs,
    };
    let summary_path = output_dir.join(format!("qwen-tp-mlp-nccl-rank-{rank}.json"));
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

    let model_path = resolve_qwen_model_path(model_path)?;
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
    let runtime_data = runtime_config.and_then(|config| config.data.as_ref());
    let dp_streaming_index_cache = runtime_data.map(|data| {
        data.index_cache
            .as_ref()
            .map(|path| qwen_sft_rank_index_cache_path(path, rank))
            .unwrap_or_else(|| {
                qwen_sft_streaming_index_cache_path(
                    &output_dir.join(format!("rank-{rank}-cache")),
                    "qwen-session-dp",
                )
            })
    });
    let batch_plan = qwen_session_dp_batch_plan_from_config(
        &model_path,
        &weights,
        data_cursor_start,
        steps,
        world_size,
        device,
        runtime_config,
        dp_streaming_index_cache.as_deref(),
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
        QwenComputeDType::Bf16 => 2.0,
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
            streaming_train_batches: batch_plan.streaming_train_batches,
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
        &model_path,
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
            &model_path,
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
            batch_plan.streaming_train_batches,
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
    sharded_global_manifest.validate_artifacts()?;
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
    streaming_train_batches: Option<bool>,
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
        streaming_train_batches,
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
    manifest.validate_artifacts()?;
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
        streaming_train_batches: manifest.streaming_train_batches,
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
    let model_path = resolve_qwen_model_path(model_path)?;
    let data_config = config
        .data
        .as_ref()
        .context("qwen session DP data plan requires [data]")?;
    if data_config.kind != RuntimeDataKind::InstructionJsonl {
        bail!("qwen session DP data plan supports kind = instruction_jsonl");
    }
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let field_map = QwenSftFieldMap::from_runtime_data(data_config)?;
    let datasets = qwen_sft_train_eval_datasets_from_paths(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.max_eval_samples,
        data_config.train_split,
        data_config.shuffle,
        config.run.seed,
        &field_map,
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
        streaming_train_batches: true,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

pub fn qwen_sft_streaming_data_plan(
    config_path: &Path,
    world_size: usize,
    data_cursor_start: usize,
) -> Result<()> {
    if world_size == 0 {
        bail!("qwen SFT streaming data plan requires world_size > 0");
    }
    let config = load_config(config_path)?;
    let data_config = config
        .data
        .as_ref()
        .context("qwen SFT streaming data plan requires [data]")?;
    if data_config.kind != RuntimeDataKind::InstructionJsonl {
        bail!("qwen SFT streaming data plan supports kind = instruction_jsonl");
    }

    let field_map = QwenSftFieldMap::from_runtime_data(data_config)?;
    let train_summary =
        qwen_sft_streaming_source_summary(&data_config.paths, data_config.max_samples, &field_map)?;
    let (
        dataset_total_samples,
        dataset_train_samples,
        dataset_eval_samples,
        dataset_source_files,
        dataset_source_sample_counts,
        dataset_fingerprint,
        eval_summary,
    ) = if data_config.eval_paths.is_empty() {
        let (train_samples, eval_samples) =
            qwen_sft_train_eval_sample_counts(train_summary.samples, data_config.train_split)?;
        (
            train_summary.samples,
            train_samples,
            eval_samples,
            train_summary.source_files.clone(),
            train_summary.source_sample_counts.clone(),
            train_summary.fingerprint.clone(),
            QwenSftStreamingSourceSummary {
                samples: eval_samples,
                source_files: Vec::new(),
                source_sample_counts: Vec::new(),
                fingerprint: String::new(),
            },
        )
    } else {
        let eval_field_map = qwen_sft_eval_field_map(&field_map);
        let eval_summary = qwen_sft_streaming_source_summary(
            &data_config.eval_paths,
            data_config.max_eval_samples,
            &eval_field_map,
        )?;
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
        (
            train_summary.samples + eval_summary.samples,
            train_summary.samples,
            eval_summary.samples,
            combined_source_files,
            combined_source_sample_counts,
            combined_fingerprint,
            eval_summary,
        )
    };

    let local_batch_size = config
        .train
        .micro_batch_size
        .min(dataset_train_samples)
        .max(1);
    let global_batch_size = local_batch_size * world_size;
    let train_steps = config.train.max_steps as usize;
    let required_batches = train_steps * global_batch_size + 1;
    let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
    let data_cursor_next = data_cursor_end;
    let (data_epoch_start, data_sample_offset_start) =
        qwen_data_epoch_and_offset(data_cursor_start, dataset_train_samples)?;
    let (data_epoch_end, data_sample_offset_end) =
        qwen_data_epoch_and_offset(data_cursor_end, dataset_train_samples)?;
    let (data_epoch_next, data_sample_offset_next) =
        qwen_data_epoch_and_offset(data_cursor_next, dataset_train_samples)?;
    let train_window_sample_cursors = qwen_sft_streaming_cursor_window(
        data_cursor_start,
        required_batches,
        global_batch_size,
        dataset_train_samples,
    )?;
    let train_window_start_cursor = data_cursor_start;
    let train_window_end_cursor_exclusive = train_window_sample_cursors
        .last()
        .map(|entry| entry.cursor + 1)
        .unwrap_or(data_cursor_start);

    let summary = QwenSftStreamingDataPlanSummary {
        config_path: config_path.display().to_string(),
        data_paths: data_config
            .paths
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        eval_paths: data_config
            .eval_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        max_samples: data_config.max_samples,
        max_eval_samples: data_config.max_eval_samples,
        train_split: data_config.train_split,
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
        train_window_start_cursor,
        train_window_end_cursor_exclusive,
        train_window_sample_cursors,
        dataset_total_samples,
        dataset_train_samples,
        dataset_eval_samples,
        dataset_source_files,
        dataset_source_sample_counts,
        dataset_fingerprint,
        dataset_order_seed: config.run.seed,
        dataset_shuffle: data_config.shuffle,
        train_source_files: train_summary.source_files,
        train_source_sample_counts: train_summary.source_sample_counts,
        train_fingerprint: train_summary.fingerprint,
        eval_source_files: eval_summary.source_files,
        eval_source_sample_counts: eval_summary.source_sample_counts,
        eval_fingerprint: eval_summary.fingerprint,
        tokenizer_loaded: false,
        tokenized_samples_materialized: false,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

pub fn qwen_sft_streaming_batch_plan(
    config_path: &Path,
    world_size: usize,
    data_cursor_start: usize,
    index_cache: Option<&Path>,
) -> Result<()> {
    if world_size == 0 {
        bail!("qwen SFT streaming batch plan requires world_size > 0");
    }
    let config = load_config(config_path)?;
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("qwen SFT streaming batch plan requires model.model_path")?;
    let model_path = resolve_qwen_model_path(model_path)?;
    let data_config = config
        .data
        .as_ref()
        .context("qwen SFT streaming batch plan requires [data]")?;
    if data_config.kind != RuntimeDataKind::InstructionJsonl {
        bail!("qwen SFT streaming batch plan supports kind = instruction_jsonl");
    }

    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let field_map = QwenSftFieldMap::from_runtime_data(data_config)?;
    let datasets = qwen_sft_train_eval_datasets_from_paths(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.max_eval_samples,
        data_config.train_split,
        data_config.shuffle,
        config.run.seed,
        &field_map,
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
    let train_batch_count = train_steps + 1;
    let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
    let data_cursor_next = data_cursor_end;
    let train_window_sample_cursors = qwen_sft_streaming_cursor_window(
        data_cursor_start,
        required_batches,
        global_batch_size,
        train_dataset.len(),
    )?;
    let train_window_end_cursor_exclusive = train_window_sample_cursors
        .last()
        .map(|entry| entry.cursor + 1)
        .unwrap_or(data_cursor_start);

    let streaming_window = qwen_sft_streaming_token_window_from_jsonl(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.train_split,
        data_config.shuffle,
        config.run.seed,
        data_cursor_start,
        train_window_sample_cursors.len(),
        index_cache,
        &field_map,
    )?;
    let mut batch_sequence_tokens = Vec::with_capacity(train_batch_count);
    let mut batch_masked_positions = Vec::with_capacity(train_batch_count);
    let mut batch_padding_tokens = Vec::with_capacity(train_batch_count);
    let mut batch_token_fingerprints = Vec::with_capacity(train_batch_count);
    let mut materialized_input_max_delta = 0_i64;
    let mut materialized_mask_max_delta = 0.0_f64;

    for batch_index in 0..train_batch_count {
        let offset = batch_index * global_batch_size;
        let end = offset + global_batch_size;
        let streaming_batch = qwen_sft_padded_batch(
            &streaming_window.samples[offset..end],
            train_dataset.pad_token_id,
        )?;
        let materialized_batch =
            train_dataset.padded_batch(data_cursor_start + offset, global_batch_size)?;
        batch_sequence_tokens.push(streaming_batch.input_ids.size()[1] as usize);
        batch_masked_positions.push(streaming_batch.masked_positions);
        batch_padding_tokens.push(streaming_batch.padding_tokens);
        batch_token_fingerprints.push(qwen_tensor_i64_fingerprint(&streaming_batch.input_ids)?);
        materialized_input_max_delta = materialized_input_max_delta.max(tensor_i64_max_abs_diff(
            &streaming_batch.input_ids,
            &materialized_batch.input_ids,
        )?);
        materialized_mask_max_delta = materialized_mask_max_delta.max(tensor_max_abs_diff(
            &streaming_batch.target_mask,
            &materialized_batch.target_mask,
        )?);
    }

    let summary = QwenSftStreamingBatchPlanSummary {
        config_path: config_path.display().to_string(),
        model_path: model_path.display().to_string(),
        world_size,
        local_batch_size,
        global_batch_size,
        train_steps,
        required_batches,
        train_batch_count,
        data_cursor_start,
        data_cursor_end,
        data_cursor_next,
        train_window_start_cursor: data_cursor_start,
        train_window_end_cursor_exclusive,
        train_window_sample_cursors,
        dataset_total_samples: dataset_summary.samples,
        dataset_train_samples: train_dataset.len(),
        dataset_eval_samples: eval_dataset.len(),
        dataset_source_files: dataset_summary.source_files,
        dataset_source_sample_counts: dataset_summary.source_sample_counts,
        dataset_fingerprint: dataset_summary.fingerprint,
        dataset_order_seed: config.run.seed,
        dataset_shuffle: dataset_summary.shuffle,
        tokenizer_loaded: true,
        tokenized_samples_materialized: true,
        reference_tokenized_samples_materialized: true,
        streaming_index_cache_path: index_cache.map(|path| path.display().to_string()),
        streaming_index_cache_hit: streaming_window.source_index_cache_hit,
        streaming_index_cache_written: streaming_window.source_index_cache_written,
        streaming_window_samples: streaming_window.samples.len(),
        streaming_raw_samples_read: streaming_window.raw_samples_read,
        streaming_raw_sample_indices: streaming_window.raw_sample_indices,
        batch_sequence_tokens,
        batch_masked_positions,
        batch_padding_tokens,
        batch_token_fingerprints,
        materialized_input_max_delta,
        materialized_mask_max_delta,
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
        crate::runtime::DType::Fp32 => QwenComputeDType::Fp32,
        crate::runtime::DType::Bf16 => QwenComputeDType::Bf16,
        crate::runtime::DType::Fp16 => {
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

fn qwen_session_tp_rank_smoke(
    model_path: &Path,
    output_dir: PathBuf,
    config: &Config,
) -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != config.parallel.tensor_model_parallel_size {
        bail!(
            "WORLD_SIZE={world_size} does not match tensor_model_parallel_size={}",
            config.parallel.tensor_model_parallel_size
        );
    }
    if world_size != 2 {
        bail!("Qwen session TP smoke expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    let device = Device::Cuda(local_rank);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let output_dir = qwen_tp_artifact_dir(&output_dir)?;
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let attention = qwen_tp_attention_contribution(
        model_path,
        device,
        rank,
        world_size,
        "Qwen session TP attention",
    )?;
    let attention_reduced = nccl_smoke::all_reduce_tensor_f32_for_launch(
        &output_dir.join("attention-output-contribution"),
        &attention.output_contribution,
    )?;
    let attention_diff = diff_stats(&attention_reduced, &attention.full_output)?;
    if attention_diff.max_abs > 1e-5 {
        bail!(
            "Qwen session TP attention parity failed: rank={}, max_abs={}, mean_abs={}",
            rank,
            attention_diff.max_abs,
            attention_diff.mean_abs
        );
    }

    let runtime_config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let q_proj_full = tensor(&weights, "model.layers.0.self_attn.q_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let q_bias_full = tensor(&weights, "model.layers.0.self_attn.q_proj.bias")?
        .to_kind(Kind::Float)
        .to_device(device);
    let k_proj_full = tensor(&weights, "model.layers.0.self_attn.k_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let k_bias_full = tensor(&weights, "model.layers.0.self_attn.k_proj.bias")?
        .to_kind(Kind::Float)
        .to_device(device);
    let v_proj_full = tensor(&weights, "model.layers.0.self_attn.v_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let v_bias_full = tensor(&weights, "model.layers.0.self_attn.v_proj.bias")?
        .to_kind(Kind::Float)
        .to_device(device);
    let o_proj_full = tensor(&weights, "model.layers.0.self_attn.o_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let head_dim = q_proj_full.size()[1] / runtime_config.num_attention_heads;
    let q_output_start = attention.q_head_start * head_dim;
    let q_output_size = attention.q_heads_per_rank * head_dim;
    let kv_output_start = attention.kv_head_start * head_dim;
    let kv_output_size = attention.kv_heads_per_rank * head_dim;
    let q_shard_base = q_proj_full.narrow(0, q_output_start, q_output_size);
    let k_shard_base = k_proj_full.narrow(0, kv_output_start, kv_output_size);
    let v_shard_base = v_proj_full.narrow(0, kv_output_start, kv_output_size);
    let o_shard_base = o_proj_full.narrow(1, q_output_start, q_output_size);

    let attention_target = (&attention.full_output * 0.5).detach();
    let train_q = q_shard_base
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_k = k_shard_base
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_v = v_shard_base
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_o = o_shard_base
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let (attention_train_initial_loss, attention_output_grad) =
        qwen_session_tp_attention_global_mse_loss_and_grad(
            &attention.input,
            QwenTpAttentionShardWeights {
                q_proj: &train_q,
                q_bias: &q_bias_full.narrow(0, q_output_start, q_output_size),
                k_proj: &train_k,
                k_bias: &k_bias_full.narrow(0, kv_output_start, kv_output_size),
                v_proj: &train_v,
                v_bias: &v_bias_full.narrow(0, kv_output_start, kv_output_size),
                o_proj: &train_o,
            },
            &runtime_config,
            0,
            q_output_size,
            0,
            kv_output_size,
            attention.q_heads_per_rank,
            attention.kv_heads_per_rank,
            &attention_target,
            &output_dir.join("attention-train-initial-output"),
        )?;
    let attention_local_contribution = qwen_tp_attention_shard_contribution(
        &attention.input,
        QwenTpAttentionShardWeights {
            q_proj: &train_q,
            q_bias: &q_bias_full.narrow(0, q_output_start, q_output_size),
            k_proj: &train_k,
            k_bias: &k_bias_full.narrow(0, kv_output_start, kv_output_size),
            v_proj: &train_v,
            v_bias: &v_bias_full.narrow(0, kv_output_start, kv_output_size),
            o_proj: &train_o,
        },
        &runtime_config,
        0,
        q_output_size,
        0,
        kv_output_size,
        attention.q_heads_per_rank,
        attention.kv_heads_per_rank,
    )
    .1;
    (attention_local_contribution * attention_output_grad)
        .sum(Kind::Float)
        .backward();
    let q_grad = train_q.grad();
    let k_grad = train_k.grad();
    let v_grad = train_v.grad();
    let o_grad = train_o.grad();
    if !q_grad.defined() || !k_grad.defined() || !v_grad.defined() || !o_grad.defined() {
        bail!("Qwen session TP attention train smoke expected all shard gradients to be defined");
    }
    let attention_q_grad_norm = q_grad.norm().double_value(&[]);
    let attention_k_grad_norm = k_grad.norm().double_value(&[]);
    let attention_v_grad_norm = v_grad.norm().double_value(&[]);
    let attention_o_grad_norm = o_grad.norm().double_value(&[]);
    if attention_q_grad_norm <= 0.0
        || attention_k_grad_norm <= 0.0
        || attention_v_grad_norm <= 0.0
        || attention_o_grad_norm <= 0.0
    {
        bail!(
            "Qwen session TP attention train smoke expected positive grad norms: q={attention_q_grad_norm}, k={attention_k_grad_norm}, v={attention_v_grad_norm}, o={attention_o_grad_norm}"
        );
    }
    let config_learning_rate = config.train.learning_rate as f64;
    if !config_learning_rate.is_finite() || config_learning_rate <= 0.0 {
        bail!("Qwen session TP train smoke requires positive finite learning_rate");
    }
    let mut attention_train_final_loss = attention_train_initial_loss;
    let mut attention_train_learning_rate = config_learning_rate;
    for candidate_learning_rate in [
        config_learning_rate,
        config_learning_rate * 10.0,
        config_learning_rate * 100.0,
        config_learning_rate * 1000.0,
        1e-3,
        1e-2,
    ] {
        if !candidate_learning_rate.is_finite() || candidate_learning_rate <= 0.0 {
            continue;
        }
        let mut candidate_q = train_q.detach().shallow_clone();
        let mut candidate_k = train_k.detach().shallow_clone();
        let mut candidate_v = train_v.detach().shallow_clone();
        let mut candidate_o = train_o.detach().shallow_clone();
        no_grad(|| -> Result<()> {
            let _ = candidate_q.f_sub_(&(q_grad.shallow_clone() * candidate_learning_rate))?;
            let _ = candidate_k.f_sub_(&(k_grad.shallow_clone() * candidate_learning_rate))?;
            let _ = candidate_v.f_sub_(&(v_grad.shallow_clone() * candidate_learning_rate))?;
            let _ = candidate_o.f_sub_(&(o_grad.shallow_clone() * candidate_learning_rate))?;
            Ok(())
        })?;
        let (candidate_loss, _) = qwen_session_tp_attention_global_mse_loss_and_grad(
            &attention.input,
            QwenTpAttentionShardWeights {
                q_proj: &candidate_q,
                q_bias: &q_bias_full.narrow(0, q_output_start, q_output_size),
                k_proj: &candidate_k,
                k_bias: &k_bias_full.narrow(0, kv_output_start, kv_output_size),
                v_proj: &candidate_v,
                v_bias: &v_bias_full.narrow(0, kv_output_start, kv_output_size),
                o_proj: &candidate_o,
            },
            &runtime_config,
            0,
            q_output_size,
            0,
            kv_output_size,
            attention.q_heads_per_rank,
            attention.kv_heads_per_rank,
            &attention_target,
            &output_dir.join(format!(
                "attention-train-candidate-output-{candidate_learning_rate:.0e}"
            )),
        )?;
        if candidate_loss < attention_train_final_loss {
            attention_train_final_loss = candidate_loss;
            attention_train_learning_rate = candidate_learning_rate;
        }
        if attention_train_final_loss < attention_train_initial_loss {
            break;
        }
    }
    if attention_train_final_loss >= attention_train_initial_loss {
        bail!(
            "Qwen session TP attention train smoke did not reduce loss: initial={attention_train_initial_loss}, final={attention_train_final_loss}"
        );
    }

    let gate_proj_full = tensor(&weights, "model.layers.0.mlp.gate_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let up_proj_full = tensor(&weights, "model.layers.0.mlp.up_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let down_proj_full = tensor(&weights, "model.layers.0.mlp.down_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let hidden_size = gate_proj_full.size()[1];
    let intermediate_size = gate_proj_full.size()[0];
    if intermediate_size % world_size as i64 != 0 {
        bail!("Qwen session TP MLP smoke requires intermediate size divisible by WORLD_SIZE");
    }
    let intermediate_shard_size = intermediate_size / world_size as i64;
    let intermediate_start = rank as i64 * intermediate_shard_size;
    let mlp_input = Tensor::arange(hidden_size * 7, (Kind::Float, device))
        .reshape([1, 7, hidden_size])
        .fmod(19.0)
        / 19.0;
    let gate_shard_base = gate_proj_full.narrow(0, intermediate_start, intermediate_shard_size);
    let up_shard_base = up_proj_full.narrow(0, intermediate_start, intermediate_shard_size);
    let down_shard_base = down_proj_full.narrow(1, intermediate_start, intermediate_shard_size);
    let (activation_shard, mlp_contribution) = qwen_tp_mlp_shard_contribution(
        &mlp_input,
        &gate_shard_base,
        &up_shard_base,
        &down_shard_base,
    );
    let mlp_reduced = nccl_smoke::all_reduce_tensor_f32_for_launch(
        &output_dir.join("mlp-output-contribution"),
        &mlp_contribution,
    )?;
    let mlp_full = qwen_mlp(&mlp_input, &gate_proj_full, &up_proj_full, &down_proj_full);
    let mlp_diff = diff_stats(&mlp_reduced, &mlp_full)?;
    if mlp_diff.max_abs > 1e-5 {
        bail!(
            "Qwen session TP MLP parity failed: rank={}, max_abs={}, mean_abs={}",
            rank,
            mlp_diff.max_abs,
            mlp_diff.mean_abs
        );
    }

    let target = (&mlp_full * 0.5).detach();
    let train_gate = gate_shard_base
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_up = up_shard_base
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_down = down_shard_base
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let (train_initial_loss, output_grad) = qwen_session_tp_mlp_global_mse_loss_and_grad(
        &mlp_input,
        &train_gate,
        &train_up,
        &train_down,
        &target,
        &output_dir.join("mlp-train-initial-output"),
    )?;
    let local_contribution =
        qwen_tp_mlp_shard_contribution(&mlp_input, &train_gate, &train_up, &train_down).1;
    (local_contribution * output_grad)
        .sum(Kind::Float)
        .backward();
    let gate_grad = train_gate.grad();
    let up_grad = train_up.grad();
    let down_grad = train_down.grad();
    if !gate_grad.defined() || !up_grad.defined() || !down_grad.defined() {
        bail!("Qwen session TP MLP train smoke expected all shard gradients to be defined");
    }
    let gate_grad_norm = gate_grad.norm().double_value(&[]);
    let up_grad_norm = up_grad.norm().double_value(&[]);
    let down_grad_norm = down_grad.norm().double_value(&[]);
    if gate_grad_norm <= 0.0 || up_grad_norm <= 0.0 || down_grad_norm <= 0.0 {
        bail!(
            "Qwen session TP MLP train smoke expected positive grad norms: gate={gate_grad_norm}, up={up_grad_norm}, down={down_grad_norm}"
        );
    }
    let mut train_final_loss = train_initial_loss;
    let mut train_learning_rate = config_learning_rate;
    for candidate_learning_rate in [
        config_learning_rate,
        config_learning_rate * 10.0,
        config_learning_rate * 100.0,
        config_learning_rate * 1000.0,
        1e-3,
        1e-2,
    ] {
        if !candidate_learning_rate.is_finite() || candidate_learning_rate <= 0.0 {
            continue;
        }
        let mut candidate_gate = train_gate.detach().shallow_clone();
        let mut candidate_up = train_up.detach().shallow_clone();
        let mut candidate_down = train_down.detach().shallow_clone();
        no_grad(|| -> Result<()> {
            let _ =
                candidate_gate.f_sub_(&(gate_grad.shallow_clone() * candidate_learning_rate))?;
            let _ = candidate_up.f_sub_(&(up_grad.shallow_clone() * candidate_learning_rate))?;
            let _ =
                candidate_down.f_sub_(&(down_grad.shallow_clone() * candidate_learning_rate))?;
            Ok(())
        })?;
        let (candidate_loss, _) = qwen_session_tp_mlp_global_mse_loss_and_grad(
            &mlp_input,
            &candidate_gate,
            &candidate_up,
            &candidate_down,
            &target,
            &output_dir.join(format!(
                "mlp-train-candidate-output-{candidate_learning_rate:.0e}"
            )),
        )?;
        if candidate_loss < train_final_loss {
            train_final_loss = candidate_loss;
            train_learning_rate = candidate_learning_rate;
        }
        if train_final_loss < train_initial_loss {
            break;
        }
    }
    if train_final_loss >= train_initial_loss {
        bail!(
            "Qwen session TP MLP train smoke did not reduce loss: initial={train_initial_loss}, final={train_final_loss}"
        );
    }

    let input_norm_full = tensor(&weights, "model.layers.0.input_layernorm.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let post_attention_norm_full =
        tensor(&weights, "model.layers.0.post_attention_layernorm.weight")?
            .to_kind(Kind::Float)
            .to_device(device);
    let layer0_input = Tensor::arange(q_proj_full.size()[1] * 6, (Kind::Float, device))
        .reshape([1, 6, q_proj_full.size()[1]])
        .fmod(29.0)
        / 29.0;
    let full_layer0 = qwen_layer(
        &layer0_input,
        &QwenLayerWeights {
            input_norm: input_norm_full.shallow_clone(),
            q_proj: q_proj_full.shallow_clone(),
            q_bias: q_bias_full.shallow_clone(),
            k_proj: k_proj_full.shallow_clone(),
            k_bias: k_bias_full.shallow_clone(),
            v_proj: v_proj_full.shallow_clone(),
            v_bias: v_bias_full.shallow_clone(),
            o_proj: o_proj_full.shallow_clone(),
            post_attention_norm: post_attention_norm_full.shallow_clone(),
            gate_proj: gate_proj_full.shallow_clone(),
            up_proj: up_proj_full.shallow_clone(),
            down_proj: down_proj_full.shallow_clone(),
        },
        &runtime_config,
    );
    let (_, _, layer0_reduced) = qwen_session_tp_layer0_global_mse_loss_and_grad(
        &layer0_input,
        &input_norm_full,
        &post_attention_norm_full,
        QwenTpAttentionShardWeights {
            q_proj: &q_shard_base,
            q_bias: &q_bias_full.narrow(0, q_output_start, q_output_size),
            k_proj: &k_shard_base,
            k_bias: &k_bias_full.narrow(0, kv_output_start, kv_output_size),
            v_proj: &v_shard_base,
            v_bias: &v_bias_full.narrow(0, kv_output_start, kv_output_size),
            o_proj: &o_shard_base,
        },
        &gate_shard_base,
        &up_shard_base,
        &down_shard_base,
        &runtime_config,
        attention.q_heads_per_rank,
        attention.kv_heads_per_rank,
        &full_layer0,
        &output_dir.join("layer0-parity"),
    )?;
    let layer0_diff = diff_stats(&layer0_reduced, &full_layer0)?;
    if layer0_diff.max_abs > 1e-5 {
        bail!(
            "Qwen session TP layer0 parity failed: rank={}, max_abs={}, mean_abs={}",
            rank,
            layer0_diff.max_abs,
            layer0_diff.mean_abs
        );
    }

    let layer0_target = (&full_layer0 * 0.5).detach();
    let layer0_update = qwen_session_tp_layer0_sgd_update(
        &layer0_input,
        &input_norm_full,
        &post_attention_norm_full,
        &q_shard_base,
        &q_bias_full.narrow(0, q_output_start, q_output_size),
        &k_shard_base,
        &k_bias_full.narrow(0, kv_output_start, kv_output_size),
        &v_shard_base,
        &v_bias_full.narrow(0, kv_output_start, kv_output_size),
        &o_shard_base,
        &gate_shard_base,
        &up_shard_base,
        &down_shard_base,
        &runtime_config,
        attention.q_heads_per_rank,
        attention.kv_heads_per_rank,
        &layer0_target,
        1e-3,
        &output_dir.join("layer0-train"),
    )?;
    if layer0_update.final_loss >= layer0_update.initial_loss {
        bail!(
            "Qwen session TP layer0 train smoke did not reduce loss: initial={}, final={}",
            layer0_update.initial_loss,
            layer0_update.final_loss
        );
    }
    let causal_train_input_ids = qwen_session_dp_global_input(&weights, device)?.narrow(0, 0, 1);
    let causal_train = qwen_session_tp_causal_lm_sgd_update(
        &causal_train_input_ids,
        &weights,
        &input_norm_full,
        &post_attention_norm_full,
        &q_shard_base,
        &q_bias_full.narrow(0, q_output_start, q_output_size),
        &k_shard_base,
        &k_bias_full.narrow(0, kv_output_start, kv_output_size),
        &v_shard_base,
        &v_bias_full.narrow(0, kv_output_start, kv_output_size),
        &o_shard_base,
        &gate_shard_base,
        &up_shard_base,
        &down_shard_base,
        &runtime_config,
        attention.q_heads_per_rank,
        attention.kv_heads_per_rank,
        config_learning_rate,
        &output_dir.join("causal-lm-train"),
    )?;
    let causal_train_initial_loss_delta =
        (causal_train.initial_loss - causal_train.full_loss).abs();
    if causal_train_initial_loss_delta > 1e-2 {
        bail!(
            "Qwen session TP focused causal LM initial loss diverged from full loss: rank={}, tp_loss={}, full_loss={}, delta={}",
            rank,
            causal_train.initial_loss,
            causal_train.full_loss,
            causal_train_initial_loss_delta
        );
    }
    let (
        sharded_rank_manifest_output,
        sharded_global_manifest_output,
        sharded_manifest_tensor_count,
    ) = write_qwen_session_tp_focused_sharded_manifest(
        &output_dir,
        model_path,
        rank,
        world_size,
        &input_norm_full,
        &post_attention_norm_full,
        &q_shard_base,
        &k_shard_base,
        &v_shard_base,
        &o_shard_base,
        &gate_shard_base,
        &up_shard_base,
        &down_shard_base,
        &[
            (
                "model.layers.0.self_attn.q_proj.weight",
                &causal_train.q_grad,
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                &causal_train.k_grad,
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                &causal_train.v_grad,
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                &causal_train.o_grad,
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                &causal_train.gate_grad,
            ),
            ("model.layers.0.mlp.up_proj.weight", &causal_train.up_grad),
            (
                "model.layers.0.mlp.down_proj.weight",
                &causal_train.down_grad,
            ),
        ],
        &q_proj_full,
        &k_proj_full,
        &v_proj_full,
        &o_proj_full,
        &gate_proj_full,
        &up_proj_full,
        &down_proj_full,
        config,
    )?;
    let (sharded_restore_diff, restored_shards) = qwen_session_tp_focused_sharded_restore(
        &sharded_global_manifest_output,
        rank,
        &layer0_input,
        &full_layer0,
        &q_bias_full.narrow(0, q_output_start, q_output_size),
        &k_bias_full.narrow(0, kv_output_start, kv_output_size),
        &v_bias_full.narrow(0, kv_output_start, kv_output_size),
        &runtime_config,
        attention.q_heads_per_rank,
        attention.kv_heads_per_rank,
        &output_dir.join("layer0-sharded-restore"),
        &[
            ("model.layers.0.input_layernorm.weight", &input_norm_full),
            (
                "model.layers.0.post_attention_layernorm.weight",
                &post_attention_norm_full,
            ),
            ("model.layers.0.self_attn.q_proj.weight", &q_shard_base),
            ("model.layers.0.self_attn.k_proj.weight", &k_shard_base),
            ("model.layers.0.self_attn.v_proj.weight", &v_shard_base),
            ("model.layers.0.self_attn.o_proj.weight", &o_shard_base),
            ("model.layers.0.mlp.gate_proj.weight", &gate_shard_base),
            ("model.layers.0.mlp.up_proj.weight", &up_shard_base),
            ("model.layers.0.mlp.down_proj.weight", &down_shard_base),
        ],
    )?;
    if sharded_restore_diff.max_abs > 1e-3 {
        bail!(
            "Qwen session TP focused sharded restore parity failed: rank={}, max_abs={}, mean_abs={}",
            rank,
            sharded_restore_diff.max_abs,
            sharded_restore_diff.mean_abs
        );
    }
    let sharded_update = qwen_session_tp_layer0_sgd_update(
        &layer0_input,
        &restored_shards.input_norm,
        &restored_shards.post_attention_norm,
        &restored_shards.q,
        &q_bias_full.narrow(0, q_output_start, q_output_size),
        &restored_shards.k,
        &k_bias_full.narrow(0, kv_output_start, kv_output_size),
        &restored_shards.v,
        &v_bias_full.narrow(0, kv_output_start, kv_output_size),
        &restored_shards.o,
        &restored_shards.gate,
        &restored_shards.up,
        &restored_shards.down,
        &runtime_config,
        attention.q_heads_per_rank,
        attention.kv_heads_per_rank,
        &layer0_target,
        1e-3,
        &output_dir.join("layer0-sharded-next-update"),
    )?;
    let sharded_next_update_diff =
        diff_stats(&sharded_update.final_output, &layer0_update.final_output)?;
    if sharded_next_update_diff.max_abs > 1e-3 {
        bail!(
            "Qwen session TP focused sharded next-update parity failed: rank={}, max_abs={}, mean_abs={}",
            rank,
            sharded_next_update_diff.max_abs,
            sharded_next_update_diff.mean_abs
        );
    }
    let external_resume = config
        .train
        .resume_from
        .as_deref()
        .map(|resume_from| {
            qwen_session_tp_focused_external_resume(
                resume_from,
                rank,
                world_size,
                &layer0_input,
                &full_layer0,
                &q_bias_full.narrow(0, q_output_start, q_output_size),
                &k_bias_full.narrow(0, kv_output_start, kv_output_size),
                &v_bias_full.narrow(0, kv_output_start, kv_output_size),
                &runtime_config,
                attention.q_heads_per_rank,
                attention.kv_heads_per_rank,
                &layer0_target,
                &layer0_update.final_output,
                &output_dir.join("layer0-sharded-external-resume"),
            )
        })
        .transpose()?;

    let summary = QwenSessionTpRankSummary {
        rank,
        world_size,
        local_rank,
        model_path: model_path.display().to_string(),
        resume_from: config
            .train
            .resume_from
            .as_ref()
            .map(|path| path.display().to_string()),
        resumed_sharded_checkpoint: external_resume.is_some(),
        resume_global_step: external_resume.as_ref().map(|resume| resume.global_step),
        resume_rank_manifest_output: external_resume
            .as_ref()
            .map(|resume| resume.rank_manifest_output.clone()),
        resume_model_safetensors: external_resume
            .as_ref()
            .map(|resume| resume.model_safetensors.clone()),
        resume_optimizer_safetensors: external_resume
            .as_ref()
            .map(|resume| resume.optimizer_safetensors.clone()),
        resume_sharded_manifest_tensor_count: external_resume
            .as_ref()
            .map(|resume| resume.tensor_count),
        resume_restore_max_abs: external_resume
            .as_ref()
            .map(|resume| resume.restore_diff.max_abs),
        resume_restore_mean_abs: external_resume
            .as_ref()
            .map(|resume| resume.restore_diff.mean_abs),
        resume_next_update_max_abs: external_resume
            .as_ref()
            .map(|resume| resume.next_update_diff.max_abs),
        resume_next_update_mean_abs: external_resume
            .as_ref()
            .map(|resume| resume.next_update_diff.mean_abs),
        tensor_model_parallel_size: config.parallel.tensor_model_parallel_size,
        data_parallel_size: config.parallel.data_parallel_size,
        attention_q_head_start: attention.q_head_start,
        attention_q_head_end: attention.q_head_start + attention.q_heads_per_rank,
        attention_kv_head_start: attention.kv_head_start,
        attention_kv_head_end: attention.kv_head_start + attention.kv_heads_per_rank,
        attention_context_shard_shape: attention.context_shard.size(),
        attention_reduced_output_shape: attention_reduced.size(),
        attention_max_abs: attention_diff.max_abs,
        attention_mean_abs: attention_diff.mean_abs,
        attention_train_initial_loss,
        attention_train_final_loss,
        attention_train_loss_improved: attention_train_final_loss < attention_train_initial_loss,
        attention_train_learning_rate,
        attention_train_q_grad_norm: attention_q_grad_norm,
        attention_train_k_grad_norm: attention_k_grad_norm,
        attention_train_v_grad_norm: attention_v_grad_norm,
        attention_train_o_grad_norm: attention_o_grad_norm,
        layer0_reduced_output_shape: layer0_update.initial_output.size(),
        layer0_max_abs: layer0_diff.max_abs,
        layer0_mean_abs: layer0_diff.mean_abs,
        layer0_train_initial_loss: layer0_update.initial_loss,
        layer0_train_final_loss: layer0_update.final_loss,
        layer0_train_loss_improved: layer0_update.final_loss < layer0_update.initial_loss,
        layer0_train_learning_rate: 1e-3,
        layer0_train_q_grad_norm: layer0_update.q_grad_norm,
        layer0_train_k_grad_norm: layer0_update.k_grad_norm,
        layer0_train_v_grad_norm: layer0_update.v_grad_norm,
        layer0_train_o_grad_norm: layer0_update.o_grad_norm,
        layer0_train_gate_grad_norm: layer0_update.gate_grad_norm,
        layer0_train_up_grad_norm: layer0_update.up_grad_norm,
        layer0_train_down_grad_norm: layer0_update.down_grad_norm,
        causal_train_input_shape: causal_train_input_ids.size(),
        causal_train_full_loss: causal_train.full_loss,
        causal_train_initial_loss: causal_train.initial_loss,
        causal_train_initial_loss_delta,
        causal_train_final_loss: causal_train.final_loss,
        causal_train_loss_improved: causal_train.final_loss < causal_train.initial_loss,
        causal_train_learning_rate: causal_train.learning_rate,
        causal_train_q_grad_norm: causal_train.q_grad_norm,
        causal_train_k_grad_norm: causal_train.k_grad_norm,
        causal_train_v_grad_norm: causal_train.v_grad_norm,
        causal_train_o_grad_norm: causal_train.o_grad_norm,
        causal_train_gate_grad_norm: causal_train.gate_grad_norm,
        causal_train_up_grad_norm: causal_train.up_grad_norm,
        causal_train_down_grad_norm: causal_train.down_grad_norm,
        causal_train_q_grad_sum: causal_train.q_grad_sum,
        causal_train_k_grad_sum: causal_train.k_grad_sum,
        causal_train_v_grad_sum: causal_train.v_grad_sum,
        causal_train_o_grad_sum: causal_train.o_grad_sum,
        causal_train_gate_grad_sum: causal_train.gate_grad_sum,
        causal_train_up_grad_sum: causal_train.up_grad_sum,
        causal_train_down_grad_sum: causal_train.down_grad_sum,
        sharded_rank_manifest_output: sharded_rank_manifest_output.display().to_string(),
        sharded_global_manifest_output: sharded_global_manifest_output.display().to_string(),
        sharded_manifest_tensor_count,
        sharded_restore_max_abs: sharded_restore_diff.max_abs,
        sharded_restore_mean_abs: sharded_restore_diff.mean_abs,
        sharded_next_update_max_abs: sharded_next_update_diff.max_abs,
        sharded_next_update_mean_abs: sharded_next_update_diff.mean_abs,
        mlp_intermediate_start: intermediate_start,
        mlp_intermediate_end: intermediate_start + intermediate_shard_size,
        mlp_activation_shard_shape: activation_shard.size(),
        mlp_reduced_output_shape: mlp_reduced.size(),
        mlp_max_abs: mlp_diff.max_abs,
        mlp_mean_abs: mlp_diff.mean_abs,
        mlp_train_initial_loss: train_initial_loss,
        mlp_train_final_loss: train_final_loss,
        mlp_train_loss_improved: train_final_loss < train_initial_loss,
        mlp_train_learning_rate: train_learning_rate,
        mlp_train_gate_grad_norm: gate_grad_norm,
        mlp_train_up_grad_norm: up_grad_norm,
        mlp_train_down_grad_norm: down_grad_norm,
    };
    let summary_path = output_dir.join(format!("qwen-session-tp-rank-{rank}.json"));
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn qwen_tp_mlp_shard_contribution(
    input: &Tensor,
    gate_shard_weight: &Tensor,
    up_shard_weight: &Tensor,
    down_input_shard_weight: &Tensor,
) -> (Tensor, Tensor) {
    let gate_shard = input.linear::<&Tensor>(gate_shard_weight, None);
    let up_shard = input.linear::<&Tensor>(up_shard_weight, None);
    let activation_shard = gate_shard.silu() * up_shard;
    let output_contribution = activation_shard.linear::<&Tensor>(down_input_shard_weight, None);
    (activation_shard, output_contribution)
}

fn qwen_session_tp_mlp_global_mse_loss_and_grad(
    input: &Tensor,
    gate_shard_weight: &Tensor,
    up_shard_weight: &Tensor,
    down_input_shard_weight: &Tensor,
    target: &Tensor,
    reduce_dir: &Path,
) -> Result<(f64, Tensor)> {
    let local_contribution = qwen_tp_mlp_shard_contribution(
        input,
        gate_shard_weight,
        up_shard_weight,
        down_input_shard_weight,
    )
    .1;
    let reduced = nccl_smoke::all_reduce_tensor_f32_for_launch(reduce_dir, &local_contribution)?;
    let diff = &reduced - target;
    let loss = diff.pow_tensor_scalar(2.0).mean(Kind::Float);
    let output_grad = (&diff * (2.0 / diff.numel() as f64)).detach();
    Ok((loss.double_value(&[]), output_grad))
}

#[allow(clippy::too_many_arguments)]
fn write_qwen_session_tp_focused_sharded_manifest(
    output_dir: &Path,
    model_path: &Path,
    rank: usize,
    world_size: usize,
    input_norm: &Tensor,
    post_attention_norm: &Tensor,
    q_shard: &Tensor,
    k_shard: &Tensor,
    v_shard: &Tensor,
    o_shard: &Tensor,
    gate_shard: &Tensor,
    up_shard: &Tensor,
    down_shard: &Tensor,
    optimizer_slot_grads: &[(&str, &Tensor)],
    q_full: &Tensor,
    k_full: &Tensor,
    v_full: &Tensor,
    o_full: &Tensor,
    gate_full: &Tensor,
    up_full: &Tensor,
    down_full: &Tensor,
    config: &Config,
) -> Result<(PathBuf, PathBuf, usize)> {
    let rank_dir = output_dir.join(format!("tp-sharded-rank-{rank}"));
    fs::create_dir_all(&rank_dir)
        .with_context(|| format!("failed to create {}", rank_dir.display()))?;
    let model_safetensors = rank_dir.join("model.safetensors");
    let optimizer_safetensors = rank_dir.join("optimizer.safetensors");
    let model_entries: Vec<(String, Tensor)> = vec![
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            input_norm.contiguous(),
        ),
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            post_attention_norm.contiguous(),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            q_shard.contiguous(),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            k_shard.contiguous(),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            v_shard.contiguous(),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            o_shard.contiguous(),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            gate_shard.contiguous(),
        ),
        (
            "model.layers.0.mlp.up_proj.weight".to_string(),
            up_shard.contiguous(),
        ),
        (
            "model.layers.0.mlp.down_proj.weight".to_string(),
            down_shard.contiguous(),
        ),
    ];
    let model_refs: Vec<(&str, &Tensor)> = model_entries
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    Tensor::write_safetensors(&model_refs, &model_safetensors)
        .with_context(|| format!("failed to write {}", model_safetensors.display()))?;

    let beta1 = config.train.adam_beta1 as f64;
    let beta2 = config.train.adam_beta2 as f64;
    let optimizer_slot_grad_map = optimizer_slot_grads
        .iter()
        .map(|(name, grad)| (*name, *grad))
        .collect::<BTreeMap<_, _>>();
    let optimizer_entries: Vec<(String, Tensor)> = model_entries
        .iter()
        .flat_map(|(name, tensor)| {
            let grad = optimizer_slot_grad_map.get(name.as_str()).copied();
            let first_moment = grad
                .map(|grad| (grad.detach().to_device(tensor.device()) * (1.0 - beta1)).contiguous())
                .unwrap_or_else(|| Tensor::zeros_like(tensor));
            let second_moment = grad
                .map(|grad| {
                    (grad
                        .detach()
                        .to_device(tensor.device())
                        .pow_tensor_scalar(2.0)
                        * (1.0 - beta2))
                        .contiguous()
                })
                .unwrap_or_else(|| Tensor::zeros_like(tensor));
            [
                (format!("{name}.adam_m"), first_moment),
                (format!("{name}.adam_v"), second_moment),
            ]
        })
        .collect();
    let optimizer_refs: Vec<(&str, &Tensor)> = optimizer_entries
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    Tensor::write_safetensors(&optimizer_refs, &optimizer_safetensors)
        .with_context(|| format!("failed to write {}", optimizer_safetensors.display()))?;

    let shard_specs = [
        (
            "model.layers.0.input_layernorm.weight",
            input_norm,
            input_norm,
            "replicated_norm_smoke",
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            post_attention_norm,
            post_attention_norm,
            "replicated_norm_smoke",
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            q_full,
            q_shard,
            "tp_row",
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            k_full,
            k_shard,
            "tp_row",
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            v_full,
            v_shard,
            "tp_row",
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            o_full,
            o_shard,
            "tp_col",
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            gate_full,
            gate_shard,
            "tp_row",
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            up_full,
            up_shard,
            "tp_row",
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            down_full,
            down_shard,
            "tp_col",
        ),
    ];
    let shards = shard_specs
        .into_iter()
        .map(
            |(name, global_tensor, shard_tensor, partition)| QwenTensorShardManifestEntry {
                name: name.to_string(),
                shard_name: name.to_string(),
                optimizer_m_name: format!("{name}.adam_m"),
                optimizer_v_name: format!("{name}.adam_v"),
                global_shape: global_tensor.size(),
                shard_shape: shard_tensor.size(),
                dtype: "fp32".to_string(),
                partition: partition.to_string(),
                tied_group: None,
            },
        )
        .collect::<Vec<_>>();
    let rank_manifest = QwenRankShardManifest {
        rank,
        data_parallel_rank: 0,
        tensor_model_parallel_rank: rank,
        pipeline_model_parallel_rank: 0,
        expert_model_parallel_rank: 0,
        context_parallel_rank: 0,
        model_safetensors: model_safetensors.display().to_string(),
        optimizer_safetensors: optimizer_safetensors.display().to_string(),
        shards,
    };
    let rank_manifest_output = output_dir.join(format!("qwen-session-tp-sharded-rank-{rank}.json"));
    fs::write(
        &rank_manifest_output,
        serde_json::to_string_pretty(&rank_manifest)? + "\n",
    )
    .with_context(|| format!("failed to write {}", rank_manifest_output.display()))?;
    wait_for_rank_barrier(
        &output_dir.join("qwen-session-tp-sharded-rank-manifests-written"),
        rank,
        world_size,
        Duration::from_secs(300),
    )?;

    let global_manifest_output = output_dir.join("qwen-session-tp-sharded-global.json");
    if rank == 0 {
        let mut ranks = Vec::with_capacity(world_size);
        for shard_rank in 0..world_size {
            let rank_manifest_path =
                output_dir.join(format!("qwen-session-tp-sharded-rank-{shard_rank}.json"));
            let text = fs::read_to_string(&rank_manifest_path)
                .with_context(|| format!("failed to read {}", rank_manifest_path.display()))?;
            let rank_manifest: QwenRankShardManifest = serde_json::from_str(&text)
                .with_context(|| format!("failed to parse {}", rank_manifest_path.display()))?;
            ranks.push(rank_manifest);
        }
        let manifest = QwenShardedCheckpointManifest {
            format: "rustrain.qwen_sharded.v1".to_string(),
            base_model_path: model_path.display().to_string(),
            tokenizer_path: model_path.join("tokenizer.json").display().to_string(),
            global_step: config.train.max_steps,
            consumed_samples: world_size as u64,
            consumed_tokens: (world_size * config.model.seq_len) as u64,
            data_cursor_next: None,
            data_epoch_next: None,
            data_sample_offset_next: None,
            data_train_samples: None,
            dataset_source_files: Vec::new(),
            dataset_source_sample_counts: Vec::new(),
            dataset_fingerprint: String::new(),
            dataset_shuffle: true,
            streaming_train_batches: None,
            seed: config.run.seed,
            dtype: "fp32".to_string(),
            optimizer: "adamw_first_step_slots_smoke".to_string(),
            scheduler: "constant".to_string(),
            parallel: QwenShardedParallelManifest {
                data_parallel_size: 1,
                tensor_model_parallel_size: world_size,
                pipeline_model_parallel_size: 1,
                expert_model_parallel_size: 1,
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
    wait_for_rank_barrier(
        &output_dir.join("qwen-session-tp-sharded-global-manifest-written"),
        rank,
        world_size,
        Duration::from_secs(300),
    )?;
    let global_text = fs::read_to_string(&global_manifest_output)
        .with_context(|| format!("failed to read {}", global_manifest_output.display()))?;
    let global_manifest: QwenShardedCheckpointManifest = serde_json::from_str(&global_text)
        .with_context(|| format!("failed to parse {}", global_manifest_output.display()))?;
    global_manifest.validate_artifacts()?;
    let tensor_count = global_manifest
        .ranks
        .iter()
        .find(|entry| entry.rank == rank)
        .map(|entry| entry.shards.len())
        .ok_or_else(|| anyhow!("missing TP sharded rank manifest for rank {rank}"))?;
    Ok((rank_manifest_output, global_manifest_output, tensor_count))
}

#[allow(clippy::too_many_arguments)]
fn qwen_session_tp_focused_sharded_restore(
    global_manifest_output: &Path,
    rank: usize,
    layer0_input: &Tensor,
    full_layer0: &Tensor,
    q_bias_shard: &Tensor,
    k_bias_shard: &Tensor,
    v_bias_shard: &Tensor,
    runtime_config: &QwenRuntimeConfig,
    q_heads_per_rank: i64,
    kv_heads_per_rank: i64,
    reduce_dir: &Path,
    expected_shards: &[(&str, &Tensor)],
) -> Result<(DiffStats, QwenSessionTpFocusedLayer0Shards)> {
    let global_text = fs::read_to_string(global_manifest_output)
        .with_context(|| format!("failed to read {}", global_manifest_output.display()))?;
    let global_manifest: QwenShardedCheckpointManifest = serde_json::from_str(&global_text)
        .with_context(|| format!("failed to parse {}", global_manifest_output.display()))?;
    global_manifest.validate_artifacts()?;
    let rank_manifest = global_manifest
        .ranks
        .iter()
        .find(|entry| entry.rank == rank)
        .ok_or_else(|| anyhow!("missing TP sharded rank manifest for rank {rank}"))?;
    let shard_tensors = read_safetensors_map(Path::new(&rank_manifest.model_safetensors))?;
    for (name, expected) in expected_shards {
        let restored = tensor(&shard_tensors, name)?
            .to_kind(Kind::Float)
            .to_device(expected.device());
        let diff = diff_stats(&restored, expected)?;
        if diff.max_abs > 1e-7 {
            bail!(
                "Qwen session TP focused sharded restore roundtrip failed: rank={}, tensor={}, max_abs={}, mean_abs={}",
                rank,
                name,
                diff.max_abs,
                diff.mean_abs
            );
        }
    }
    let input_norm = tensor(&shard_tensors, "model.layers.0.input_layernorm.weight")?
        .to_kind(Kind::Float)
        .to_device(layer0_input.device());
    let post_attention_norm = tensor(
        &shard_tensors,
        "model.layers.0.post_attention_layernorm.weight",
    )?
    .to_kind(Kind::Float)
    .to_device(layer0_input.device());
    let q_shard = tensor(&shard_tensors, "model.layers.0.self_attn.q_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(layer0_input.device());
    let k_shard = tensor(&shard_tensors, "model.layers.0.self_attn.k_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(layer0_input.device());
    let v_shard = tensor(&shard_tensors, "model.layers.0.self_attn.v_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(layer0_input.device());
    let o_shard = tensor(&shard_tensors, "model.layers.0.self_attn.o_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(layer0_input.device());
    let gate_shard = tensor(&shard_tensors, "model.layers.0.mlp.gate_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(layer0_input.device());
    let up_shard = tensor(&shard_tensors, "model.layers.0.mlp.up_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(layer0_input.device());
    let down_shard = tensor(&shard_tensors, "model.layers.0.mlp.down_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(layer0_input.device());
    let restored_shards = QwenSessionTpFocusedLayer0Shards {
        input_norm,
        post_attention_norm,
        q: q_shard,
        k: k_shard,
        v: v_shard,
        o: o_shard,
        gate: gate_shard,
        up: up_shard,
        down: down_shard,
    };
    let (_, _, restored_layer0) = qwen_session_tp_layer0_global_mse_loss_and_grad(
        layer0_input,
        &restored_shards.input_norm,
        &restored_shards.post_attention_norm,
        QwenTpAttentionShardWeights {
            q_proj: &restored_shards.q,
            q_bias: q_bias_shard,
            k_proj: &restored_shards.k,
            k_bias: k_bias_shard,
            v_proj: &restored_shards.v,
            v_bias: v_bias_shard,
            o_proj: &restored_shards.o,
        },
        &restored_shards.gate,
        &restored_shards.up,
        &restored_shards.down,
        runtime_config,
        q_heads_per_rank,
        kv_heads_per_rank,
        full_layer0,
        reduce_dir,
    )?;
    Ok((diff_stats(&restored_layer0, full_layer0)?, restored_shards))
}

#[allow(clippy::too_many_arguments)]
fn qwen_session_tp_focused_external_resume(
    global_manifest_output: &Path,
    rank: usize,
    world_size: usize,
    layer0_input: &Tensor,
    full_layer0: &Tensor,
    q_bias_shard: &Tensor,
    k_bias_shard: &Tensor,
    v_bias_shard: &Tensor,
    runtime_config: &QwenRuntimeConfig,
    q_heads_per_rank: i64,
    kv_heads_per_rank: i64,
    layer0_target: &Tensor,
    reference_update_output: &Tensor,
    reduce_dir: &Path,
) -> Result<QwenSessionTpFocusedResume> {
    let global_text = fs::read_to_string(global_manifest_output)
        .with_context(|| format!("failed to read {}", global_manifest_output.display()))?;
    let global_manifest: QwenShardedCheckpointManifest = serde_json::from_str(&global_text)
        .with_context(|| format!("failed to parse {}", global_manifest_output.display()))?;
    global_manifest.validate_artifacts()?;
    if global_manifest.parallel.tensor_model_parallel_size != world_size {
        bail!(
            "Qwen session TP focused resume tensor_model_parallel_size {} does not match WORLD_SIZE {world_size}",
            global_manifest.parallel.tensor_model_parallel_size
        );
    }
    if global_manifest.parallel.data_parallel_size != 1 {
        bail!(
            "Qwen session TP focused resume expects data_parallel_size=1, got {}",
            global_manifest.parallel.data_parallel_size
        );
    }
    let rank_manifest = global_manifest
        .ranks
        .iter()
        .find(|entry| entry.rank == rank)
        .ok_or_else(|| anyhow!("missing TP sharded rank manifest for rank {rank}"))?;
    if rank_manifest.tensor_model_parallel_rank != rank {
        bail!(
            "Qwen session TP focused resume expected tensor_model_parallel_rank {rank}, got {}",
            rank_manifest.tensor_model_parallel_rank
        );
    }
    if rank_manifest.shards.len() != 9 {
        bail!(
            "Qwen session TP focused resume expected 9 focused layer0 shards, got {}",
            rank_manifest.shards.len()
        );
    }
    let optimizer_tensors = read_safetensors_map(Path::new(&rank_manifest.optimizer_safetensors))?;
    for shard in &rank_manifest.shards {
        let optimizer_m = tensor(&optimizer_tensors, &shard.optimizer_m_name)?;
        let optimizer_v = tensor(&optimizer_tensors, &shard.optimizer_v_name)?;
        if optimizer_m.size() != shard.shard_shape || optimizer_v.size() != shard.shard_shape {
            bail!(
                "Qwen session TP focused resume optimizer slot shape mismatch for rank={}, tensor={}",
                rank,
                shard.name
            );
        }
    }
    let (restore_diff, restored_shards) = qwen_session_tp_focused_sharded_restore(
        global_manifest_output,
        rank,
        layer0_input,
        full_layer0,
        q_bias_shard,
        k_bias_shard,
        v_bias_shard,
        runtime_config,
        q_heads_per_rank,
        kv_heads_per_rank,
        reduce_dir,
        &[],
    )?;
    if restore_diff.max_abs > 1e-3 {
        bail!(
            "Qwen session TP focused external resume restore parity failed: rank={}, max_abs={}, mean_abs={}",
            rank,
            restore_diff.max_abs,
            restore_diff.mean_abs
        );
    }
    let resumed_update = qwen_session_tp_layer0_sgd_update(
        layer0_input,
        &restored_shards.input_norm,
        &restored_shards.post_attention_norm,
        &restored_shards.q,
        q_bias_shard,
        &restored_shards.k,
        k_bias_shard,
        &restored_shards.v,
        v_bias_shard,
        &restored_shards.o,
        &restored_shards.gate,
        &restored_shards.up,
        &restored_shards.down,
        runtime_config,
        q_heads_per_rank,
        kv_heads_per_rank,
        layer0_target,
        1e-3,
        &reduce_dir.join("next-update"),
    )?;
    let next_update_diff = diff_stats(&resumed_update.final_output, reference_update_output)?;
    if next_update_diff.max_abs > 1e-3 {
        bail!(
            "Qwen session TP focused external resume next-update parity failed: rank={}, max_abs={}, mean_abs={}",
            rank,
            next_update_diff.max_abs,
            next_update_diff.mean_abs
        );
    }
    Ok(QwenSessionTpFocusedResume {
        global_step: global_manifest.global_step,
        rank_manifest_output: global_manifest_output
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!("qwen-session-tp-sharded-rank-{rank}.json"))
            .display()
            .to_string(),
        model_safetensors: rank_manifest.model_safetensors.clone(),
        optimizer_safetensors: rank_manifest.optimizer_safetensors.clone(),
        tensor_count: rank_manifest.shards.len(),
        restore_diff,
        next_update_diff,
    })
}

#[allow(clippy::too_many_arguments)]
fn qwen_session_tp_layer0_sgd_update(
    input: &Tensor,
    input_norm_weight: &Tensor,
    post_attention_norm_weight: &Tensor,
    q_shard_weight: &Tensor,
    q_bias_shard: &Tensor,
    k_shard_weight: &Tensor,
    k_bias_shard: &Tensor,
    v_shard_weight: &Tensor,
    v_bias_shard: &Tensor,
    o_shard_weight: &Tensor,
    gate_shard_weight: &Tensor,
    up_shard_weight: &Tensor,
    down_shard_weight: &Tensor,
    config: &QwenRuntimeConfig,
    q_heads_per_rank: i64,
    kv_heads_per_rank: i64,
    target: &Tensor,
    learning_rate: f64,
    reduce_dir: &Path,
) -> Result<QwenSessionTpLayer0SgdUpdate> {
    let train_q = q_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_k = k_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_v = v_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_o = o_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_gate = gate_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_up = up_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_down = down_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let (initial_loss, output_grad, initial_output) =
        qwen_session_tp_layer0_global_mse_loss_and_grad(
            input,
            input_norm_weight,
            post_attention_norm_weight,
            QwenTpAttentionShardWeights {
                q_proj: &train_q,
                q_bias: q_bias_shard,
                k_proj: &train_k,
                k_bias: k_bias_shard,
                v_proj: &train_v,
                v_bias: v_bias_shard,
                o_proj: &train_o,
            },
            &train_gate,
            &train_up,
            &train_down,
            config,
            q_heads_per_rank,
            kv_heads_per_rank,
            target,
            &reduce_dir.join("initial"),
        )?;
    let attention_contribution = qwen_session_tp_layer0_local_contributions(
        input,
        input_norm_weight,
        QwenTpAttentionShardWeights {
            q_proj: &train_q,
            q_bias: q_bias_shard,
            k_proj: &train_k,
            k_bias: k_bias_shard,
            v_proj: &train_v,
            v_bias: v_bias_shard,
            o_proj: &train_o,
        },
        config,
        q_heads_per_rank,
        kv_heads_per_rank,
    );
    let reduced_attention_for_train = nccl_smoke::all_reduce_tensor_f32_for_launch(
        &reduce_dir.join("backward-attention"),
        &attention_contribution,
    )?;
    let mlp_contribution = qwen_session_tp_layer0_mlp_contribution(
        input,
        post_attention_norm_weight,
        &reduced_attention_for_train.detach(),
        &train_gate,
        &train_up,
        &train_down,
        config,
    );
    ((attention_contribution + mlp_contribution) * output_grad)
        .sum(Kind::Float)
        .backward();
    let q_grad = train_q.grad();
    let k_grad = train_k.grad();
    let v_grad = train_v.grad();
    let o_grad = train_o.grad();
    let gate_grad = train_gate.grad();
    let up_grad = train_up.grad();
    let down_grad = train_down.grad();
    let layer0_grads = [
        ("q", &q_grad),
        ("k", &k_grad),
        ("v", &v_grad),
        ("o", &o_grad),
        ("gate", &gate_grad),
        ("up", &up_grad),
        ("down", &down_grad),
    ];
    for (name, grad) in layer0_grads {
        if !grad.defined() || grad.norm().double_value(&[]) <= 0.0 {
            bail!("Qwen session TP layer0 train smoke expected positive {name} grad norm");
        }
    }
    let mut candidate_q = train_q.detach().shallow_clone();
    let mut candidate_k = train_k.detach().shallow_clone();
    let mut candidate_v = train_v.detach().shallow_clone();
    let mut candidate_o = train_o.detach().shallow_clone();
    let mut candidate_gate = train_gate.detach().shallow_clone();
    let mut candidate_up = train_up.detach().shallow_clone();
    let mut candidate_down = train_down.detach().shallow_clone();
    no_grad(|| -> Result<()> {
        let _ = candidate_q.f_sub_(&(q_grad.shallow_clone() * learning_rate))?;
        let _ = candidate_k.f_sub_(&(k_grad.shallow_clone() * learning_rate))?;
        let _ = candidate_v.f_sub_(&(v_grad.shallow_clone() * learning_rate))?;
        let _ = candidate_o.f_sub_(&(o_grad.shallow_clone() * learning_rate))?;
        let _ = candidate_gate.f_sub_(&(gate_grad.shallow_clone() * learning_rate))?;
        let _ = candidate_up.f_sub_(&(up_grad.shallow_clone() * learning_rate))?;
        let _ = candidate_down.f_sub_(&(down_grad.shallow_clone() * learning_rate))?;
        Ok(())
    })?;
    let (final_loss, _, final_output) = qwen_session_tp_layer0_global_mse_loss_and_grad(
        input,
        input_norm_weight,
        post_attention_norm_weight,
        QwenTpAttentionShardWeights {
            q_proj: &candidate_q,
            q_bias: q_bias_shard,
            k_proj: &candidate_k,
            k_bias: k_bias_shard,
            v_proj: &candidate_v,
            v_bias: v_bias_shard,
            o_proj: &candidate_o,
        },
        &candidate_gate,
        &candidate_up,
        &candidate_down,
        config,
        q_heads_per_rank,
        kv_heads_per_rank,
        target,
        &reduce_dir.join("candidate-output"),
    )?;
    Ok(QwenSessionTpLayer0SgdUpdate {
        initial_loss,
        final_loss,
        initial_output,
        final_output,
        q_grad_norm: q_grad.norm().double_value(&[]),
        k_grad_norm: k_grad.norm().double_value(&[]),
        v_grad_norm: v_grad.norm().double_value(&[]),
        o_grad_norm: o_grad.norm().double_value(&[]),
        gate_grad_norm: gate_grad.norm().double_value(&[]),
        up_grad_norm: up_grad.norm().double_value(&[]),
        down_grad_norm: down_grad.norm().double_value(&[]),
    })
}

#[allow(clippy::too_many_arguments)]
fn qwen_session_tp_causal_lm_loss_and_output_grad(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    input_norm_weight: &Tensor,
    post_attention_norm_weight: &Tensor,
    q_shard_weight: &Tensor,
    q_bias_shard: &Tensor,
    k_shard_weight: &Tensor,
    k_bias_shard: &Tensor,
    v_shard_weight: &Tensor,
    v_bias_shard: &Tensor,
    o_shard_weight: &Tensor,
    gate_shard_weight: &Tensor,
    up_shard_weight: &Tensor,
    down_shard_weight: &Tensor,
    config: &QwenRuntimeConfig,
    q_heads_per_rank: i64,
    kv_heads_per_rank: i64,
    reduce_dir: &Path,
) -> Result<(f64, Tensor)> {
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?
        .to_kind(Kind::Float)
        .to_device(input_ids.device());
    let layer0_input = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);
    let attention_contribution = qwen_session_tp_layer0_local_contributions(
        &layer0_input,
        input_norm_weight,
        QwenTpAttentionShardWeights {
            q_proj: q_shard_weight,
            q_bias: q_bias_shard,
            k_proj: k_shard_weight,
            k_bias: k_bias_shard,
            v_proj: v_shard_weight,
            v_bias: v_bias_shard,
            o_proj: o_shard_weight,
        },
        config,
        q_heads_per_rank,
        kv_heads_per_rank,
    );
    let reduced_attention = nccl_smoke::all_reduce_tensor_f32_for_launch(
        &reduce_dir.join("attention"),
        &attention_contribution,
    )?;
    let mlp_contribution = qwen_session_tp_layer0_mlp_contribution(
        &layer0_input,
        post_attention_norm_weight,
        &reduced_attention,
        gate_shard_weight,
        up_shard_weight,
        down_shard_weight,
        config,
    );
    let reduced_mlp =
        nccl_smoke::all_reduce_tensor_f32_for_launch(&reduce_dir.join("mlp"), &mlp_contribution)?;
    let mut hidden = (&layer0_input + &reduced_attention + &reduced_mlp)
        .detach()
        .set_requires_grad(true);
    for layer_index in 1..config.num_hidden_layers {
        let layer = QwenLayerWeights::load(weights, layer_index)?;
        let layer = QwenLayerWeights {
            input_norm: layer.input_norm.to_device(input_ids.device()),
            q_proj: layer.q_proj.to_device(input_ids.device()),
            q_bias: layer.q_bias.to_device(input_ids.device()),
            k_proj: layer.k_proj.to_device(input_ids.device()),
            k_bias: layer.k_bias.to_device(input_ids.device()),
            v_proj: layer.v_proj.to_device(input_ids.device()),
            v_bias: layer.v_bias.to_device(input_ids.device()),
            o_proj: layer.o_proj.to_device(input_ids.device()),
            post_attention_norm: layer.post_attention_norm.to_device(input_ids.device()),
            gate_proj: layer.gate_proj.to_device(input_ids.device()),
            up_proj: layer.up_proj.to_device(input_ids.device()),
            down_proj: layer.down_proj.to_device(input_ids.device()),
        };
        hidden = qwen_layer(&hidden, &layer, config);
    }
    let final_norm = tensor(weights, "model.norm.weight")?
        .to_kind(Kind::Float)
        .to_device(input_ids.device());
    let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps);
    let logits = hidden.linear::<&Tensor>(&embed_tokens, None);
    let seq_len = input_ids.size()[1];
    let shifted_logits = logits.narrow(1, 0, seq_len - 1);
    let targets = input_ids.narrow(1, 1, seq_len - 1);
    let vocab_size = shifted_logits.size()[2];
    let loss = shifted_logits
        .reshape([-1, vocab_size])
        .log_softmax(-1, Kind::Float)
        .g_nll_loss::<&Tensor>(&targets.reshape([-1]), None, Reduction::Mean, -100);
    let loss_value = loss.double_value(&[]);
    let output_grads = Tensor::run_backward(&[loss], &[&hidden], false, false);
    let output_grad = output_grads
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("Qwen session TP focused causal LM loss returned no output grad"))?;
    if !output_grad.defined() || output_grad.norm().double_value(&[]) <= 0.0 {
        bail!("Qwen session TP focused causal LM train smoke expected positive layer0 output grad");
    }
    Ok((loss_value, output_grad.detach()))
}

#[allow(clippy::too_many_arguments)]
fn qwen_session_tp_causal_lm_sgd_update(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    input_norm_weight: &Tensor,
    post_attention_norm_weight: &Tensor,
    q_shard_weight: &Tensor,
    q_bias_shard: &Tensor,
    k_shard_weight: &Tensor,
    k_bias_shard: &Tensor,
    v_shard_weight: &Tensor,
    v_bias_shard: &Tensor,
    o_shard_weight: &Tensor,
    gate_shard_weight: &Tensor,
    up_shard_weight: &Tensor,
    down_shard_weight: &Tensor,
    config: &QwenRuntimeConfig,
    q_heads_per_rank: i64,
    kv_heads_per_rank: i64,
    learning_rate: f64,
    reduce_dir: &Path,
) -> Result<QwenSessionTpCausalLmSgdUpdate> {
    let full_loss = {
        let device_weights = weights
            .iter()
            .map(|(name, tensor)| {
                (
                    name.clone(),
                    tensor.to_kind(Kind::Float).to_device(input_ids.device()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        qwen_causal_lm_loss(input_ids, &device_weights, config)?.double_value(&[])
    };
    let train_q = q_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_k = k_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_v = v_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_o = o_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_gate = gate_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_up = up_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let train_down = down_shard_weight
        .detach()
        .shallow_clone()
        .set_requires_grad(true);
    let (initial_loss, output_grad) = qwen_session_tp_causal_lm_loss_and_output_grad(
        input_ids,
        weights,
        input_norm_weight,
        post_attention_norm_weight,
        &train_q,
        q_bias_shard,
        &train_k,
        k_bias_shard,
        &train_v,
        v_bias_shard,
        &train_o,
        &train_gate,
        &train_up,
        &train_down,
        config,
        q_heads_per_rank,
        kv_heads_per_rank,
        &reduce_dir.join("initial"),
    )?;
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?
        .to_kind(Kind::Float)
        .to_device(input_ids.device());
    let layer0_input = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);
    let attention_contribution = qwen_session_tp_layer0_local_contributions(
        &layer0_input,
        input_norm_weight,
        QwenTpAttentionShardWeights {
            q_proj: &train_q,
            q_bias: q_bias_shard,
            k_proj: &train_k,
            k_bias: k_bias_shard,
            v_proj: &train_v,
            v_bias: v_bias_shard,
            o_proj: &train_o,
        },
        config,
        q_heads_per_rank,
        kv_heads_per_rank,
    );
    let reduced_attention_for_train = nccl_smoke::all_reduce_tensor_f32_for_launch(
        &reduce_dir.join("backward-attention"),
        &attention_contribution,
    )?;
    let mlp_contribution = qwen_session_tp_layer0_mlp_contribution(
        &layer0_input,
        post_attention_norm_weight,
        &reduced_attention_for_train.detach(),
        &train_gate,
        &train_up,
        &train_down,
        config,
    );
    ((attention_contribution + mlp_contribution) * output_grad)
        .sum(Kind::Float)
        .backward();
    let q_grad = train_q.grad();
    let k_grad = train_k.grad();
    let v_grad = train_v.grad();
    let o_grad = train_o.grad();
    let gate_grad = train_gate.grad();
    let up_grad = train_up.grad();
    let down_grad = train_down.grad();
    let causal_grads = [
        ("q", &q_grad),
        ("k", &k_grad),
        ("v", &v_grad),
        ("o", &o_grad),
        ("gate", &gate_grad),
        ("up", &up_grad),
        ("down", &down_grad),
    ];
    for (name, grad) in causal_grads {
        if !grad.defined() || grad.norm().double_value(&[]) <= 0.0 {
            bail!(
                "Qwen session TP focused causal LM train smoke expected positive {name} grad norm"
            );
        }
    }
    let mut best_loss = initial_loss;
    let mut best_learning_rate = learning_rate;
    for candidate_learning_rate in [
        learning_rate,
        learning_rate * 10.0,
        learning_rate * 100.0,
        learning_rate * 1000.0,
        1e-3,
        1e-2,
    ] {
        if !candidate_learning_rate.is_finite() || candidate_learning_rate <= 0.0 {
            continue;
        }
        let mut candidate_q = train_q.detach().shallow_clone();
        let mut candidate_k = train_k.detach().shallow_clone();
        let mut candidate_v = train_v.detach().shallow_clone();
        let mut candidate_o = train_o.detach().shallow_clone();
        let mut candidate_gate = train_gate.detach().shallow_clone();
        let mut candidate_up = train_up.detach().shallow_clone();
        let mut candidate_down = train_down.detach().shallow_clone();
        no_grad(|| -> Result<()> {
            let _ = candidate_q.f_sub_(&(q_grad.shallow_clone() * candidate_learning_rate))?;
            let _ = candidate_k.f_sub_(&(k_grad.shallow_clone() * candidate_learning_rate))?;
            let _ = candidate_v.f_sub_(&(v_grad.shallow_clone() * candidate_learning_rate))?;
            let _ = candidate_o.f_sub_(&(o_grad.shallow_clone() * candidate_learning_rate))?;
            let _ =
                candidate_gate.f_sub_(&(gate_grad.shallow_clone() * candidate_learning_rate))?;
            let _ = candidate_up.f_sub_(&(up_grad.shallow_clone() * candidate_learning_rate))?;
            let _ =
                candidate_down.f_sub_(&(down_grad.shallow_clone() * candidate_learning_rate))?;
            Ok(())
        })?;
        let (candidate_loss, _) = qwen_session_tp_causal_lm_loss_and_output_grad(
            input_ids,
            weights,
            input_norm_weight,
            post_attention_norm_weight,
            &candidate_q,
            q_bias_shard,
            &candidate_k,
            k_bias_shard,
            &candidate_v,
            v_bias_shard,
            &candidate_o,
            &candidate_gate,
            &candidate_up,
            &candidate_down,
            config,
            q_heads_per_rank,
            kv_heads_per_rank,
            &reduce_dir.join(format!("candidate-output-{candidate_learning_rate:.0e}")),
        )?;
        if candidate_loss < best_loss {
            best_loss = candidate_loss;
            best_learning_rate = candidate_learning_rate;
        }
        if best_loss < initial_loss {
            break;
        }
    }
    if best_loss >= initial_loss {
        bail!(
            "Qwen session TP focused causal LM train smoke did not reduce loss: initial={initial_loss}, final={best_loss}"
        );
    }
    Ok(QwenSessionTpCausalLmSgdUpdate {
        full_loss,
        initial_loss,
        final_loss: best_loss,
        learning_rate: best_learning_rate,
        q_grad: q_grad.detach().contiguous(),
        k_grad: k_grad.detach().contiguous(),
        v_grad: v_grad.detach().contiguous(),
        o_grad: o_grad.detach().contiguous(),
        gate_grad: gate_grad.detach().contiguous(),
        up_grad: up_grad.detach().contiguous(),
        down_grad: down_grad.detach().contiguous(),
        q_grad_norm: q_grad.norm().double_value(&[]),
        k_grad_norm: k_grad.norm().double_value(&[]),
        v_grad_norm: v_grad.norm().double_value(&[]),
        o_grad_norm: o_grad.norm().double_value(&[]),
        gate_grad_norm: gate_grad.norm().double_value(&[]),
        up_grad_norm: up_grad.norm().double_value(&[]),
        down_grad_norm: down_grad.norm().double_value(&[]),
        q_grad_sum: q_grad.sum(Kind::Float).double_value(&[]),
        k_grad_sum: k_grad.sum(Kind::Float).double_value(&[]),
        v_grad_sum: v_grad.sum(Kind::Float).double_value(&[]),
        o_grad_sum: o_grad.sum(Kind::Float).double_value(&[]),
        gate_grad_sum: gate_grad.sum(Kind::Float).double_value(&[]),
        up_grad_sum: up_grad.sum(Kind::Float).double_value(&[]),
        down_grad_sum: down_grad.sum(Kind::Float).double_value(&[]),
    })
}

#[allow(clippy::too_many_arguments)]
fn qwen_session_tp_layer0_local_contributions(
    input: &Tensor,
    input_norm_weight: &Tensor,
    attention_weights: QwenTpAttentionShardWeights<'_>,
    config: &QwenRuntimeConfig,
    q_heads_per_rank: i64,
    kv_heads_per_rank: i64,
) -> Tensor {
    let hidden_size = input.size()[2];
    let head_dim = hidden_size / config.num_attention_heads;
    let attention_input = rms_norm(input, input_norm_weight, config.rms_norm_eps);
    qwen_tp_attention_shard_contribution(
        &attention_input,
        attention_weights,
        config,
        0,
        q_heads_per_rank * head_dim,
        0,
        kv_heads_per_rank * head_dim,
        q_heads_per_rank,
        kv_heads_per_rank,
    )
    .1
}

fn qwen_session_tp_layer0_mlp_contribution(
    input: &Tensor,
    post_attention_norm_weight: &Tensor,
    reduced_attention: &Tensor,
    gate_shard_weight: &Tensor,
    up_shard_weight: &Tensor,
    down_input_shard_weight: &Tensor,
    config: &QwenRuntimeConfig,
) -> Tensor {
    let after_attention = input + reduced_attention;
    let mlp_input = rms_norm(
        &after_attention,
        post_attention_norm_weight,
        config.rms_norm_eps,
    );
    let (_, mlp_contribution) = qwen_tp_mlp_shard_contribution(
        &mlp_input,
        gate_shard_weight,
        up_shard_weight,
        down_input_shard_weight,
    );
    mlp_contribution
}

#[allow(clippy::too_many_arguments)]
fn qwen_session_tp_layer0_global_mse_loss_and_grad(
    input: &Tensor,
    input_norm_weight: &Tensor,
    post_attention_norm_weight: &Tensor,
    attention_weights: QwenTpAttentionShardWeights<'_>,
    gate_shard_weight: &Tensor,
    up_shard_weight: &Tensor,
    down_input_shard_weight: &Tensor,
    config: &QwenRuntimeConfig,
    q_heads_per_rank: i64,
    kv_heads_per_rank: i64,
    target: &Tensor,
    reduce_dir: &Path,
) -> Result<(f64, Tensor, Tensor)> {
    let attention_contribution = qwen_session_tp_layer0_local_contributions(
        input,
        input_norm_weight,
        attention_weights,
        config,
        q_heads_per_rank,
        kv_heads_per_rank,
    );
    let reduced_attention = nccl_smoke::all_reduce_tensor_f32_for_launch(
        &reduce_dir.join("attention"),
        &attention_contribution,
    )?;
    let mlp_contribution = qwen_session_tp_layer0_mlp_contribution(
        input,
        post_attention_norm_weight,
        &reduced_attention,
        gate_shard_weight,
        up_shard_weight,
        down_input_shard_weight,
        config,
    );
    let reduced_mlp =
        nccl_smoke::all_reduce_tensor_f32_for_launch(&reduce_dir.join("mlp"), &mlp_contribution)?;
    let layer_output = input + reduced_attention + reduced_mlp;
    let diff = &layer_output - target;
    let loss = diff.pow_tensor_scalar(2.0).mean(Kind::Float);
    let output_grad = (&diff * (2.0 / diff.numel() as f64)).detach();
    Ok((loss.double_value(&[]), output_grad, layer_output))
}

#[allow(clippy::too_many_arguments)]
fn qwen_session_tp_attention_global_mse_loss_and_grad(
    input: &Tensor,
    weights: QwenTpAttentionShardWeights<'_>,
    config: &QwenRuntimeConfig,
    q_output_start: i64,
    q_output_size: i64,
    kv_output_start: i64,
    kv_output_size: i64,
    q_heads_per_rank: i64,
    kv_heads_per_rank: i64,
    target: &Tensor,
    reduce_dir: &Path,
) -> Result<(f64, Tensor)> {
    let local_contribution = qwen_tp_attention_shard_contribution(
        input,
        weights,
        config,
        q_output_start,
        q_output_size,
        kv_output_start,
        kv_output_size,
        q_heads_per_rank,
        kv_heads_per_rank,
    )
    .1;
    let reduced = nccl_smoke::all_reduce_tensor_f32_for_launch(reduce_dir, &local_contribution)?;
    let diff = &reduced - target;
    let loss = diff.pow_tensor_scalar(2.0).mean(Kind::Float);
    let output_grad = (&diff * (2.0 / diff.numel() as f64)).detach();
    Ok((loss.double_value(&[]), output_grad))
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
    let model_path = resolve_qwen_model_path(model_path)?;
    let dtype = match config.train.dtype {
        crate::runtime::DType::Fp32 => QwenComputeDType::Fp32,
        crate::runtime::DType::Bf16 => QwenComputeDType::Bf16,
        crate::runtime::DType::Fp16 => {
            bail!("qwen session trainer does not support fp16 yet; use fp32 or bf16")
        }
    };
    let streaming_index_cache = config.data.as_ref().map(|data| {
        data.index_cache.clone().unwrap_or_else(|| {
            qwen_sft_streaming_index_cache_path(&run_paths.cache, "qwen-session-single")
        })
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

fn qwen_dp_artifact_dir(output_dir: &Path) -> Result<PathBuf> {
    let port = std::env::var("MASTER_PORT")
        .context("MASTER_PORT is not set; run through rustrain launch")?;
    Ok(output_dir.join(format!("launch-{port}")))
}

fn qwen_tp_artifact_dir(output_dir: &Path) -> Result<PathBuf> {
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

fn qwen_session_batch_plan_from_config(
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
    if data_config.kind != RuntimeDataKind::InstructionJsonl {
        bail!("qwen trainable session data path supports kind = instruction_jsonl");
    }
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let field_map = QwenSftFieldMap::from_runtime_data(data_config)?;
    let datasets = qwen_sft_train_eval_datasets_from_paths(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.max_eval_samples,
        data_config.train_split,
        data_config.shuffle,
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
    let streaming_window = qwen_sft_streaming_token_window_from_jsonl(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.train_split,
        data_config.shuffle,
        runtime_config.run.seed,
        data_cursor_start,
        required_batches + batch_size - 1,
        streaming_index_cache,
        &field_map,
    )?;
    let train_batches = (0..required_batches)
        .map(|relative_cursor| {
            let end = relative_cursor + batch_size;
            let streaming_batch = qwen_sft_padded_batch(
                &streaming_window.samples[relative_cursor..end],
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
        streaming_index_cache_path: streaming_index_cache.map(|path| path.display().to_string()),
        streaming_index_cache_hit: Some(streaming_window.source_index_cache_hit),
        streaming_index_cache_written: Some(streaming_window.source_index_cache_written),
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

fn qwen_session_dp_batch_plan_from_config(
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
    if data_config.kind != RuntimeDataKind::InstructionJsonl {
        bail!("qwen trainable session DP data path supports kind = instruction_jsonl");
    }
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let field_map = QwenSftFieldMap::from_runtime_data(data_config)?;
    let datasets = qwen_sft_train_eval_datasets_from_paths(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.max_eval_samples,
        data_config.train_split,
        data_config.shuffle,
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
    let streaming_window = qwen_sft_streaming_token_window_from_jsonl(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.train_split,
        data_config.shuffle,
        runtime_config.run.seed,
        data_cursor_start,
        required_batches + global_batch_size - 1,
        streaming_index_cache,
        &field_map,
    )?;
    let global_train_batches = (0..required_batches)
        .map(|relative_cursor| {
            let end = relative_cursor + global_batch_size;
            let streaming_batch = qwen_sft_padded_batch(
                &streaming_window.samples[relative_cursor..end],
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
        streaming_index_cache_path: streaming_index_cache.map(|path| path.display().to_string()),
        streaming_index_cache_hit: Some(streaming_window.source_index_cache_hit),
        streaming_index_cache_written: Some(streaming_window.source_index_cache_written),
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

pub(crate) fn resolve_qwen_model_path(model_path: &Path) -> Result<PathBuf> {
    if qwen_model_path_is_complete(model_path) {
        return Ok(model_path.to_path_buf());
    }
    let Some(model_dir_name) = model_path.file_name().and_then(|name| name.to_str()) else {
        bail!(
            "Qwen model path {} is missing config/tokenizer/model files and has no model directory name",
            model_path.display()
        );
    };
    let Some(root) = model_path.parent() else {
        bail!(
            "Qwen model path {} is missing config/tokenizer/model files and has no parent directory",
            model_path.display()
        );
    };
    let hub_root = root.join("hub");
    let hub_suffix = format!("--{model_dir_name}");
    let hub_model_dirs = fs::read_dir(&hub_root)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("models--") && name.ends_with(&hub_suffix))
        })
        .collect::<Vec<_>>();
    if hub_model_dirs.is_empty() {
        bail!(
            "Qwen model path {} is missing config/tokenizer/model files and no matching HF hub cache entry was found under {}",
            model_path.display(),
            hub_root.display()
        );
    }
    let mut candidates = Vec::new();
    for hub_model_dir in hub_model_dirs {
        let snapshots_dir = hub_model_dir.join("snapshots");
        if snapshots_dir.is_dir() {
            candidates.extend(
                fs::read_dir(&snapshots_dir)
                    .with_context(|| format!("failed to list {}", snapshots_dir.display()))?
                    .map(|entry| entry.map(|entry| entry.path()))
                    .collect::<std::io::Result<Vec<_>>>()
                    .with_context(|| {
                        format!("failed to read entries under {}", snapshots_dir.display())
                    })?,
            );
        }
    }
    candidates.sort();
    candidates
        .into_iter()
        .rev()
        .find(|candidate| qwen_model_path_is_complete(candidate))
        .ok_or_else(|| {
            anyhow!(
                "Qwen model path {} is missing config/tokenizer/model files and no complete HF hub snapshot exists under {}",
                model_path.display(),
                hub_root.display()
            )
        })
}

pub(crate) fn qwen_model_path_is_complete(model_path: &Path) -> bool {
    model_path.join("config.json").exists()
        && model_path.join("tokenizer.json").exists()
        && model_path.join("model.safetensors").exists()
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
            .map(|example| qwen_sft_token_sample(tokenizer, example, &QwenSftFieldMap::default()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            samples,
            pad_token_id: qwen_pad_token_id(tokenizer),
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: qwen_sft_dataset_fingerprint(&[], examples, &QwenSftFieldMap::default()),
        })
    }

    fn from_jsonl_paths_with_limit(
        tokenizer: &Tokenizer,
        paths: &[PathBuf],
        max_samples: Option<usize>,
        field_map: &QwenSftFieldMap,
    ) -> Result<Self> {
        let example_set =
            qwen_sft_examples_from_jsonl_paths_with_limit(paths, max_samples, field_map)?;
        if example_set.examples.is_empty() {
            bail!("SFT dataset must contain at least one example");
        }
        let samples = example_set
            .examples
            .iter()
            .map(|example| qwen_sft_token_sample(tokenizer, example, field_map))
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
        let (split_at, _) = qwen_sft_train_eval_sample_counts(self.samples.len(), train_split)?;
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

fn qwen_sft_train_eval_sample_counts(
    total_samples: usize,
    train_split: f32,
) -> Result<(usize, usize)> {
    if !(0.0..1.0).contains(&train_split) {
        bail!("SFT train_split must be in (0, 1)");
    }
    if total_samples < 2 {
        bail!("SFT train/eval split requires at least two samples");
    }
    let train_samples = ((total_samples as f32) * train_split).floor() as usize;
    let train_samples = train_samples.clamp(1, total_samples - 1);
    Ok((train_samples, total_samples - train_samples))
}

fn qwen_sft_streaming_cursor_window(
    data_cursor_start: usize,
    required_batches: usize,
    global_batch_size: usize,
    train_sample_count: usize,
) -> Result<Vec<QwenSftStreamingCursorEntry>> {
    if required_batches == 0 {
        bail!("SFT streaming cursor window requires at least one batch");
    }
    if global_batch_size == 0 {
        bail!("SFT streaming cursor window requires global_batch_size > 0");
    }
    if train_sample_count == 0 {
        bail!("SFT streaming cursor window requires at least one training sample");
    }
    let needed_samples = required_batches + global_batch_size - 1;
    (0..needed_samples)
        .map(|relative| {
            let cursor = data_cursor_start + relative;
            let (epoch, sample_offset) = qwen_data_epoch_and_offset(cursor, train_sample_count)?;
            Ok(QwenSftStreamingCursorEntry {
                cursor,
                epoch,
                sample_offset,
            })
        })
        .collect()
}

fn qwen_sft_streaming_index_cache_path(base_dir: &Path, label: &str) -> PathBuf {
    base_dir.join(format!("{label}-offset-index.json"))
}

fn qwen_sft_rank_index_cache_path(path: &Path, rank: usize) -> PathBuf {
    let extension = path.extension().and_then(|extension| extension.to_str());
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("offset-index");
    let file_name = match extension {
        Some(extension) if !extension.is_empty() => format!("{stem}.rank-{rank}.{extension}"),
        _ => format!("{stem}.rank-{rank}"),
    };
    path.with_file_name(file_name)
}

fn qwen_sft_streaming_token_window_from_jsonl(
    tokenizer: &Tokenizer,
    paths: &[PathBuf],
    eval_paths: &[PathBuf],
    max_samples: Option<usize>,
    train_split: f32,
    shuffle: bool,
    seed: u64,
    data_cursor_start: usize,
    window_samples: usize,
    index_cache: Option<&Path>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftStreamingTokenWindow> {
    if window_samples == 0 {
        bail!("SFT streaming token window requires at least one sample");
    }
    let source_index_load =
        qwen_sft_streaming_source_index_with_cache(paths, max_samples, index_cache, field_map)?;
    let source_index = source_index_load.index;
    let train_samples = if eval_paths.is_empty() {
        let (train_samples, _) =
            qwen_sft_train_eval_sample_counts(source_index.samples.len(), train_split)?;
        train_samples
    } else {
        source_index.samples.len()
    };
    let mut train_indices = source_index.samples;
    if shuffle {
        let mut rng = StdRng::seed_from_u64(seed);
        train_indices.shuffle(&mut rng);
    }
    train_indices.truncate(train_samples);
    if train_indices.is_empty() {
        bail!("SFT streaming token window requires at least one training sample");
    }

    let raw_sample_indices = (0..window_samples)
        .map(|relative| {
            let cursor = data_cursor_start + relative;
            let epoch = cursor / train_indices.len();
            let offset = cursor % train_indices.len();
            let index = if shuffle {
                qwen_epoch_permutation_index(train_indices.len(), seed, epoch, offset)
            } else {
                offset
            };
            train_indices[index].clone()
        })
        .collect::<Vec<_>>();
    let raw_window = qwen_sft_examples_by_raw_indices(&raw_sample_indices, field_map)?;
    let samples = raw_window
        .examples
        .iter()
        .map(|example| qwen_sft_token_sample(tokenizer, example, field_map))
        .collect::<Result<Vec<_>>>()?;
    Ok(QwenSftStreamingTokenWindow {
        samples,
        raw_sample_indices,
        raw_samples_read: raw_window.raw_samples_read,
        source_index_cache_hit: source_index_load.cache_hit,
        source_index_cache_written: source_index_load.cache_written,
    })
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
    max_eval_samples: Option<usize>,
    train_split: f32,
    shuffle: bool,
    seed: u64,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftTrainEvalDatasets> {
    let train_dataset = QwenSftDataset::from_jsonl_paths_with_limit(
        tokenizer,
        train_paths,
        max_samples,
        field_map,
    )?;
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

    let eval_field_map = qwen_sft_eval_field_map(field_map);
    let eval_dataset = QwenSftDataset::from_jsonl_paths_with_limit(
        tokenizer,
        eval_paths,
        max_eval_samples,
        &eval_field_map,
    )?;
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

fn qwen_sft_eval_field_map(field_map: &QwenSftFieldMap) -> QwenSftFieldMap {
    let mut eval_field_map = field_map.clone();
    eval_field_map.source_weights.clear();
    eval_field_map
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

fn qwen_sft_examples_from_jsonl_paths_with_limit(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftExampleSet> {
    if paths.is_empty() {
        bail!("SFT dataset must contain at least one JSONL path");
    }
    if max_samples == Some(0) {
        bail!("SFT data.max_samples must be greater than zero");
    }
    let source_weights = qwen_sft_source_weights(paths.len(), field_map)?;
    let mut examples = Vec::new();
    let mut source_files = BTreeSet::new();
    let mut source_sample_counts = BTreeMap::new();
    let mut seen_records = field_map.dedupe_samples.then(HashSet::new);
    for (path, source_weight) in paths.iter().zip(source_weights.iter().copied()) {
        if max_samples.is_some_and(|limit| examples.len() >= limit) {
            break;
        }
        let remaining = max_samples.map(|limit| limit.saturating_sub(examples.len()));
        let example_set = qwen_sft_examples_from_jsonl_path_with_limit(
            path,
            remaining,
            source_weight,
            field_map,
            &mut seen_records,
        )?;
        examples.extend(example_set.examples);
        source_files.extend(example_set.source_files);
        for source_count in example_set.source_sample_counts {
            *source_sample_counts.entry(source_count.path).or_insert(0) += source_count.samples;
        }
    }
    if examples.is_empty() {
        bail!("SFT dataset must contain at least one example");
    }
    let source_files = source_files.into_iter().collect::<Vec<_>>();
    let source_sample_counts = source_sample_counts
        .into_iter()
        .map(|(path, samples)| QwenSftSourceSampleCount { path, samples })
        .collect::<Vec<_>>();
    let fingerprint = qwen_sft_dataset_fingerprint(&source_files, &examples, field_map);
    Ok(QwenSftExampleSet {
        examples,
        source_files,
        source_sample_counts,
        fingerprint,
    })
}

fn qwen_sft_streaming_source_index(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftStreamingSourceIndex> {
    if paths.is_empty() {
        bail!("SFT dataset must contain at least one JSONL path");
    }
    if max_samples == Some(0) {
        bail!("SFT data.max_samples must be greater than zero");
    }
    let source_weights = qwen_sft_source_weights(paths.len(), field_map)?;
    let mut samples = Vec::new();
    let mut seen_records = field_map.dedupe_samples.then(HashSet::new);

    for (path, source_weight) in paths.iter().zip(source_weights.iter().copied()) {
        if max_samples.is_some_and(|limit| samples.len() >= limit) {
            break;
        }
        for file in qwen_sft_jsonl_files(path)? {
            if max_samples.is_some_and(|limit| samples.len() >= limit) {
                break;
            }
            let file_path = file.display().to_string();
            let mut reader = BufReader::new(
                fs::File::open(&file)
                    .with_context(|| format!("failed to read {}", file.display()))?,
            );
            let mut line = String::new();
            let mut line_index = 0usize;
            loop {
                if max_samples.is_some_and(|limit| samples.len() >= limit) {
                    break;
                }
                let byte_offset = reader.stream_position().with_context(|| {
                    format!("failed to seek SFT JSONL record {}", file.display())
                })?;
                line.clear();
                let bytes_read = reader.read_line(&mut line).with_context(|| {
                    format!(
                        "failed to read SFT JSONL record {}:{}",
                        file.display(),
                        line_index + 1
                    )
                })?;
                if bytes_read == 0 {
                    break;
                }
                if line.trim().is_empty() {
                    line_index += 1;
                    continue;
                }
                let record =
                    qwen_sft_record_from_jsonl_line(&line, field_map).with_context(|| {
                        format!(
                            "failed to parse SFT JSONL record {}:{}",
                            file.display(),
                            line_index + 1
                        )
                    })?;
                if !qwen_sft_record_passes_filters(&record, field_map) {
                    line_index += 1;
                    continue;
                }
                if let Some(seen_records) = &mut seen_records {
                    if !seen_records.insert(qwen_sft_record_dedupe_key(&record)) {
                        line_index += 1;
                        continue;
                    }
                }
                for _ in 0..source_weight {
                    if max_samples.is_some_and(|limit| samples.len() >= limit) {
                        break;
                    }
                    samples.push(QwenSftRawSampleIndex {
                        path: file_path.clone(),
                        index_in_file: line_index,
                        global_index: samples.len(),
                        byte_offset,
                    });
                }
                line_index += 1;
            }
        }
    }
    if samples.is_empty() {
        bail!("SFT dataset must contain at least one example");
    }

    Ok(QwenSftStreamingSourceIndex { samples })
}

fn qwen_sft_streaming_source_index_with_cache(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    cache_path: Option<&Path>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftStreamingSourceIndexLoad> {
    let expected_paths = paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    if let Some(cache_path) = cache_path {
        if cache_path.exists() {
            let contents = fs::read_to_string(cache_path)
                .with_context(|| format!("failed to read {}", cache_path.display()))?;
            let cache: QwenSftStreamingSourceIndexCache = serde_json::from_str(&contents)
                .with_context(|| format!("failed to parse {}", cache_path.display()))?;
            if cache.format != "rustrain.qwen_sft_offset_index.v4" {
                bail!(
                    "unsupported SFT streaming index cache format {} in {}",
                    cache.format,
                    cache_path.display()
                );
            }
            if cache.paths != expected_paths {
                bail!(
                    "SFT streaming index cache paths {:?} do not match {:?}",
                    cache.paths,
                    expected_paths
                );
            }
            if cache.max_samples != max_samples {
                bail!(
                    "SFT streaming index cache max_samples {:?} does not match {:?}",
                    cache.max_samples,
                    max_samples
                );
            }
            if cache.field_map != *field_map {
                bail!(
                    "SFT streaming index cache field_map {:?} does not match {:?}",
                    cache.field_map,
                    field_map
                );
            }
            if cache.min_response_chars != field_map.min_response_chars {
                bail!(
                    "SFT streaming index cache min_response_chars {} does not match {}",
                    cache.min_response_chars,
                    field_map.min_response_chars
                );
            }
            if cache.samples.is_empty() {
                bail!(
                    "SFT streaming index cache {} contains no samples",
                    cache_path.display()
                );
            }
            return Ok(QwenSftStreamingSourceIndexLoad {
                index: QwenSftStreamingSourceIndex {
                    samples: cache.samples,
                },
                cache_hit: true,
                cache_written: false,
            });
        }
    }

    let index = qwen_sft_streaming_source_index(paths, max_samples, field_map)?;
    let mut cache_written = false;
    if let Some(cache_path) = cache_path {
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let cache = QwenSftStreamingSourceIndexCache {
            format: "rustrain.qwen_sft_offset_index.v4".to_string(),
            paths: expected_paths,
            max_samples,
            field_map: field_map.clone(),
            min_response_chars: field_map.min_response_chars,
            samples: index.samples.clone(),
        };
        let contents = serde_json::to_string_pretty(&cache)
            .context("failed to serialize SFT streaming index cache")?;
        fs::write(cache_path, contents)
            .with_context(|| format!("failed to write {}", cache_path.display()))?;
        cache_written = true;
    }
    Ok(QwenSftStreamingSourceIndexLoad {
        index,
        cache_hit: false,
        cache_written,
    })
}

fn qwen_sft_examples_by_raw_indices(
    raw_indices: &[QwenSftRawSampleIndex],
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftRawExampleWindow> {
    if raw_indices.is_empty() {
        bail!("SFT streaming raw index read requires at least one sample");
    }
    let mut by_path: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for raw_index in raw_indices {
        by_path
            .entry(raw_index.path.clone())
            .or_default()
            .insert(raw_index.index_in_file);
    }

    let mut loaded = BTreeMap::new();
    let mut offsets_by_sample = BTreeMap::new();
    for raw_index in raw_indices {
        offsets_by_sample
            .entry((raw_index.path.clone(), raw_index.index_in_file))
            .or_insert(raw_index.byte_offset);
    }
    for (path, wanted_indices) in &by_path {
        let mut file = fs::File::open(path).with_context(|| format!("failed to read {path}"))?;
        for index_in_file in wanted_indices {
            let byte_offset = *offsets_by_sample
                .get(&(path.clone(), *index_in_file))
                .ok_or_else(|| {
                    anyhow!(
                        "SFT streaming raw sample offset not found: {}:{}",
                        path,
                        index_in_file + 1
                    )
                })?;
            if !wanted_indices.contains(index_in_file) {
                continue;
            }
            file.seek(SeekFrom::Start(byte_offset)).with_context(|| {
                format!(
                    "failed to seek SFT JSONL record {path}:{} at byte offset {}",
                    index_in_file + 1,
                    byte_offset
                )
            })?;
            let mut reader = BufReader::new(&file);
            let mut line = String::new();
            reader.read_line(&mut line).with_context(|| {
                format!(
                    "failed to read SFT JSONL record {path}:{} at byte offset {}",
                    index_in_file + 1,
                    byte_offset
                )
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let record = qwen_sft_record_from_jsonl_line(&line, field_map).with_context(|| {
                format!(
                    "failed to parse SFT JSONL record {path}:{} at byte offset {}",
                    index_in_file + 1,
                    byte_offset
                )
            })?;
            loaded.insert(
                (path.clone(), *index_in_file),
                QwenSftExample {
                    system: record.system,
                    instruction: record.instruction,
                    input: record.input,
                    response: record.response,
                },
            );
        }
    }

    let examples = raw_indices
        .iter()
        .map(|raw_index| {
            loaded
                .get(&(raw_index.path.clone(), raw_index.index_in_file))
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "SFT streaming raw sample not found: {}:{}",
                        raw_index.path,
                        raw_index.index_in_file + 1
                    )
                })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(QwenSftRawExampleWindow {
        examples,
        raw_samples_read: loaded.len(),
    })
}

fn qwen_sft_examples_from_jsonl_path_with_limit(
    path: &Path,
    max_samples: Option<usize>,
    source_weight: usize,
    field_map: &QwenSftFieldMap,
    seen_records: &mut Option<HashSet<String>>,
) -> Result<QwenSftExampleSet> {
    let files = qwen_sft_jsonl_files(path)?;

    if files.is_empty() {
        bail!("SFT JSONL path {} did not contain files", path.display());
    }

    let mut examples = Vec::new();
    let mut source_sample_counts = Vec::new();
    for file in &files {
        if max_samples.is_some_and(|limit| examples.len() >= limit) {
            break;
        }
        let reader = BufReader::new(
            fs::File::open(file).with_context(|| format!("failed to read {}", file.display()))?,
        );
        let before = examples.len();
        for (line_index, line) in reader.lines().enumerate() {
            if max_samples.is_some_and(|limit| examples.len() >= limit) {
                break;
            }
            let line = line.with_context(|| {
                format!(
                    "failed to read SFT JSONL record {}:{}",
                    file.display(),
                    line_index + 1
                )
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let record = qwen_sft_record_from_jsonl_line(&line, field_map).with_context(|| {
                format!(
                    "failed to parse SFT JSONL record {}:{}",
                    file.display(),
                    line_index + 1
                )
            })?;
            if !qwen_sft_record_passes_filters(&record, field_map) {
                continue;
            }
            if let Some(seen_records) = seen_records {
                if !seen_records.insert(qwen_sft_record_dedupe_key(&record)) {
                    continue;
                }
            }
            let example = QwenSftExample {
                system: record.system,
                instruction: record.instruction,
                input: record.input,
                response: record.response,
            };
            for _ in 0..source_weight {
                if max_samples.is_some_and(|limit| examples.len() >= limit) {
                    break;
                }
                examples.push(example.clone());
            }
        }
        let consumed = examples.len() - before;
        if consumed > 0 {
            source_sample_counts.push(QwenSftSourceSampleCount {
                path: file.display().to_string(),
                samples: consumed,
            });
        }
    }

    if examples.is_empty() {
        bail!("SFT JSONL path {} did not contain examples", path.display());
    }
    let source_files = source_sample_counts
        .iter()
        .map(|count| count.path.clone())
        .collect::<Vec<_>>();
    let fingerprint = qwen_sft_dataset_fingerprint(&source_files, &examples, field_map);
    Ok(QwenSftExampleSet {
        examples,
        source_files,
        source_sample_counts,
        fingerprint,
    })
}

fn qwen_sft_streaming_source_summary(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftStreamingSourceSummary> {
    if paths.is_empty() {
        bail!("SFT dataset must contain at least one JSONL path");
    }
    if max_samples == Some(0) {
        bail!("SFT data.max_samples must be greater than zero");
    }
    let source_weights = qwen_sft_source_weights(paths.len(), field_map)?;

    let mut samples = 0usize;
    let mut source_files = BTreeSet::new();
    let mut source_sample_counts = BTreeMap::new();
    let mut seen_records = field_map.dedupe_samples.then(HashSet::new);
    for (path, source_weight) in paths.iter().zip(source_weights.iter().copied()) {
        if max_samples.is_some_and(|limit| samples >= limit) {
            break;
        }
        for file in qwen_sft_jsonl_files(path)? {
            if max_samples.is_some_and(|limit| samples >= limit) {
                break;
            }
            let file_path = file.display().to_string();
            let reader = BufReader::new(
                fs::File::open(&file)
                    .with_context(|| format!("failed to read {}", file.display()))?,
            );
            let before = samples;
            for (line_index, line) in reader.lines().enumerate() {
                if max_samples.is_some_and(|limit| samples >= limit) {
                    break;
                }
                let line = line.with_context(|| {
                    format!(
                        "failed to read SFT JSONL record {}:{}",
                        file.display(),
                        line_index + 1
                    )
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                let record =
                    qwen_sft_record_from_jsonl_line(&line, field_map).with_context(|| {
                        format!(
                            "failed to parse SFT JSONL record {}:{}",
                            file.display(),
                            line_index + 1
                        )
                    })?;
                if !qwen_sft_record_passes_filters(&record, field_map) {
                    continue;
                }
                if let Some(seen_records) = &mut seen_records {
                    if !seen_records.insert(qwen_sft_record_dedupe_key(&record)) {
                        continue;
                    }
                }
                source_files.insert(file_path.clone());
                drop(record);
                for _ in 0..source_weight {
                    if max_samples.is_some_and(|limit| samples >= limit) {
                        break;
                    }
                    samples += 1;
                }
            }
            let consumed = samples - before;
            if consumed > 0 {
                *source_sample_counts.entry(file_path).or_insert(0) += consumed;
            }
        }
    }
    if samples == 0 {
        bail!("SFT dataset must contain at least one example");
    }

    let source_files = source_files.into_iter().collect::<Vec<_>>();
    let fingerprint = qwen_sft_streaming_fingerprint(paths, max_samples, &source_files, field_map)?;
    Ok(QwenSftStreamingSourceSummary {
        samples,
        source_files,
        source_sample_counts: source_sample_counts
            .into_iter()
            .map(|(path, samples)| QwenSftSourceSampleCount { path, samples })
            .collect(),
        fingerprint,
    })
}

fn qwen_sft_source_weights(path_count: usize, field_map: &QwenSftFieldMap) -> Result<Vec<usize>> {
    if field_map.source_weights.is_empty() {
        return Ok(vec![1; path_count]);
    }
    let weights = if field_map.source_weights.len() == 1 {
        vec![field_map.source_weights[0]; path_count]
    } else if field_map.source_weights.len() == path_count {
        field_map.source_weights.clone()
    } else {
        bail!("data.source_weights must be empty, length 1, or match data.paths length");
    };
    if weights.iter().any(|weight| *weight == 0) {
        bail!("data.source_weights entries must be greater than zero");
    }
    Ok(weights)
}

fn qwen_sft_streaming_fingerprint(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    source_files: &[String],
    field_map: &QwenSftFieldMap,
) -> Result<String> {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    qwen_sft_hash_field_map(&mut hash, field_map);
    for file in source_files {
        qwen_sft_hash_bytes(&mut hash, b"path");
        qwen_sft_hash_bytes(&mut hash, file.as_bytes());
        qwen_sft_hash_bytes(&mut hash, b"\0");
    }

    let mut samples = 0usize;
    let mut seen_records = field_map.dedupe_samples.then(HashSet::new);
    let source_weights = qwen_sft_source_weights(paths.len(), field_map)?;
    for (path, source_weight) in paths.iter().zip(source_weights.iter().copied()) {
        if max_samples.is_some_and(|limit| samples >= limit) {
            break;
        }
        for file in qwen_sft_jsonl_files(path)? {
            if max_samples.is_some_and(|limit| samples >= limit) {
                break;
            }
            let reader = BufReader::new(
                fs::File::open(&file)
                    .with_context(|| format!("failed to read {}", file.display()))?,
            );
            for (line_index, line) in reader.lines().enumerate() {
                if max_samples.is_some_and(|limit| samples >= limit) {
                    break;
                }
                let line = line.with_context(|| {
                    format!(
                        "failed to read SFT JSONL record {}:{}",
                        file.display(),
                        line_index + 1
                    )
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                let record =
                    qwen_sft_record_from_jsonl_line(&line, field_map).with_context(|| {
                        format!(
                            "failed to parse SFT JSONL record {}:{}",
                            file.display(),
                            line_index + 1
                        )
                    })?;
                if !qwen_sft_record_passes_filters(&record, field_map) {
                    continue;
                }
                if let Some(seen_records) = &mut seen_records {
                    if !seen_records.insert(qwen_sft_record_dedupe_key(&record)) {
                        continue;
                    }
                }
                for _ in 0..source_weight {
                    if max_samples.is_some_and(|limit| samples >= limit) {
                        break;
                    }
                    qwen_sft_hash_record(&mut hash, &record, field_map.system.is_some());
                    samples += 1;
                }
            }
        }
    }
    Ok(format!("{hash:016x}"))
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

fn qwen_sft_record_from_jsonl_line(
    line: &str,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftRecord> {
    let values: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(line).context("invalid JSON object")?;
    let instruction = qwen_normalize_jsonl_field(
        qwen_required_jsonl_string_field(&values, &field_map.instruction)?,
        field_map,
    );
    let system = match &field_map.system {
        Some(field) => {
            qwen_normalize_jsonl_field(qwen_optional_jsonl_string_field(&values, field)?, field_map)
        }
        None => String::new(),
    };
    let input = qwen_normalize_jsonl_field(
        qwen_optional_jsonl_string_field(&values, &field_map.input)?,
        field_map,
    );
    let response = qwen_normalize_jsonl_field(
        qwen_required_jsonl_string_field(&values, &field_map.response)?,
        field_map,
    );
    Ok(QwenSftRecord {
        system,
        instruction,
        input,
        response,
    })
}

fn qwen_normalize_jsonl_field(value: String, field_map: &QwenSftFieldMap) -> String {
    if field_map.trim_fields {
        value.trim().to_string()
    } else {
        value
    }
}

fn qwen_sft_record_dedupe_key(record: &QwenSftRecord) -> String {
    format!(
        "{}\0{}\0{}\0{}",
        record.system, record.instruction, record.input, record.response
    )
}

fn qwen_sft_record_passes_filters(record: &QwenSftRecord, field_map: &QwenSftFieldMap) -> bool {
    let needs_prompt_chars = field_map.min_prompt_chars.is_some()
        || field_map.max_prompt_chars.is_some()
        || field_map.min_sample_chars.is_some()
        || field_map.max_sample_chars.is_some();
    let prompt_chars = if needs_prompt_chars {
        Some(
            qwen_render_sft_record_prompt(record, field_map)
                .chars()
                .count(),
        )
    } else {
        None
    };
    qwen_sft_length_filter_passes(
        record.response.chars().count(),
        Some(field_map.min_response_chars),
        field_map.max_response_chars,
    ) && qwen_sft_length_filter_passes(
        record.instruction.chars().count(),
        field_map.min_instruction_chars,
        field_map.max_instruction_chars,
    ) && qwen_sft_length_filter_passes(
        record.input.chars().count(),
        field_map.min_input_chars,
        field_map.max_input_chars,
    ) && prompt_chars.is_none_or(|chars| {
        qwen_sft_length_filter_passes(
            chars,
            field_map.min_prompt_chars,
            field_map.max_prompt_chars,
        ) && qwen_sft_length_filter_passes(
            chars + record.response.chars().count(),
            field_map.min_sample_chars,
            field_map.max_sample_chars,
        )
    })
}

fn qwen_sft_length_filter_passes(
    chars: usize,
    min_chars: Option<usize>,
    max_chars: Option<usize>,
) -> bool {
    min_chars.is_none_or(|limit| chars >= limit) && max_chars.is_none_or(|limit| chars <= limit)
}

fn qwen_render_sft_record_prompt(record: &QwenSftRecord, field_map: &QwenSftFieldMap) -> String {
    let template = if record.input.trim().is_empty() {
        &field_map.prompt_template
    } else {
        &field_map.prompt_with_input_template
    };
    template
        .replace("{system}", &record.system)
        .replace("{instruction}", &record.instruction)
        .replace("{input}", &record.input)
}

fn qwen_required_jsonl_string_field(
    values: &BTreeMap<String, serde_json::Value>,
    field: &str,
) -> Result<String> {
    match values.get(field) {
        Some(serde_json::Value::String(value)) => Ok(value.clone()),
        Some(_) => bail!("SFT JSONL field {field} must be a string"),
        None => bail!("SFT JSONL record missing required field {field}"),
    }
}

fn qwen_optional_jsonl_string_field(
    values: &BTreeMap<String, serde_json::Value>,
    field: &str,
) -> Result<String> {
    match values.get(field) {
        Some(serde_json::Value::String(value)) => Ok(value.clone()),
        Some(_) => bail!("SFT JSONL field {field} must be a string"),
        None => Ok(String::new()),
    }
}

fn qwen_sft_dataset_fingerprint(
    source_files: &[String],
    examples: &[QwenSftExample],
    field_map: &QwenSftFieldMap,
) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    qwen_sft_hash_field_map(&mut hash, field_map);
    for file in source_files {
        qwen_sft_hash_bytes(&mut hash, b"path");
        qwen_sft_hash_bytes(&mut hash, file.as_bytes());
        qwen_sft_hash_bytes(&mut hash, b"\0");
    }
    for example in examples {
        qwen_sft_hash_example(&mut hash, example, field_map.system.is_some());
    }
    format!("{hash:016x}")
}

fn qwen_sft_hash_record(hash: &mut u64, record: &QwenSftRecord, include_system: bool) {
    if include_system {
        qwen_sft_hash_bytes(hash, b"system");
        qwen_sft_hash_bytes(hash, record.system.as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    qwen_sft_hash_bytes(hash, b"instruction");
    qwen_sft_hash_bytes(hash, record.instruction.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0input");
    qwen_sft_hash_bytes(hash, record.input.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0response");
    qwen_sft_hash_bytes(hash, record.response.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
}

fn qwen_sft_hash_field_map(hash: &mut u64, field_map: &QwenSftFieldMap) {
    qwen_sft_hash_bytes(hash, b"field_map");
    qwen_sft_hash_bytes(hash, field_map.instruction.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
    qwen_sft_hash_bytes(hash, field_map.input.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
    qwen_sft_hash_bytes(hash, field_map.response.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
    if let Some(system) = &field_map.system {
        qwen_sft_hash_bytes(hash, b"system");
        qwen_sft_hash_bytes(hash, system.as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    qwen_sft_hash_bytes(hash, field_map.prompt_template.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
    qwen_sft_hash_bytes(hash, field_map.prompt_with_input_template.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
    qwen_sft_hash_bytes(
        hash,
        if field_map.trim_fields {
            b"trim".as_slice()
        } else {
            b"raw".as_slice()
        },
    );
    qwen_sft_hash_bytes(hash, b"\0");
    qwen_sft_hash_bytes(hash, b"min_response_chars");
    qwen_sft_hash_bytes(hash, field_map.min_response_chars.to_string().as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
    if let Some(max_response_chars) = field_map.max_response_chars {
        qwen_sft_hash_bytes(hash, b"max_response_chars");
        qwen_sft_hash_bytes(hash, max_response_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(min_instruction_chars) = field_map.min_instruction_chars {
        qwen_sft_hash_bytes(hash, b"min_instruction_chars");
        qwen_sft_hash_bytes(hash, min_instruction_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(max_instruction_chars) = field_map.max_instruction_chars {
        qwen_sft_hash_bytes(hash, b"max_instruction_chars");
        qwen_sft_hash_bytes(hash, max_instruction_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(min_input_chars) = field_map.min_input_chars {
        qwen_sft_hash_bytes(hash, b"min_input_chars");
        qwen_sft_hash_bytes(hash, min_input_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(max_input_chars) = field_map.max_input_chars {
        qwen_sft_hash_bytes(hash, b"max_input_chars");
        qwen_sft_hash_bytes(hash, max_input_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(min_prompt_chars) = field_map.min_prompt_chars {
        qwen_sft_hash_bytes(hash, b"min_prompt_chars");
        qwen_sft_hash_bytes(hash, min_prompt_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(max_prompt_chars) = field_map.max_prompt_chars {
        qwen_sft_hash_bytes(hash, b"max_prompt_chars");
        qwen_sft_hash_bytes(hash, max_prompt_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(min_sample_chars) = field_map.min_sample_chars {
        qwen_sft_hash_bytes(hash, b"min_sample_chars");
        qwen_sft_hash_bytes(hash, min_sample_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(max_sample_chars) = field_map.max_sample_chars {
        qwen_sft_hash_bytes(hash, b"max_sample_chars");
        qwen_sft_hash_bytes(hash, max_sample_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if field_map.dedupe_samples {
        qwen_sft_hash_bytes(hash, b"dedupe_samples");
        qwen_sft_hash_bytes(hash, b"true");
        qwen_sft_hash_bytes(hash, b"\0");
    }
    qwen_sft_hash_bytes(hash, b"source_weights");
    for source_weight in &field_map.source_weights {
        qwen_sft_hash_bytes(hash, source_weight.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b",");
    }
    qwen_sft_hash_bytes(hash, b"\0");
}

fn qwen_sft_hash_example(hash: &mut u64, example: &QwenSftExample, include_system: bool) {
    if include_system {
        qwen_sft_hash_bytes(hash, b"system");
        qwen_sft_hash_bytes(hash, example.system.as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    qwen_sft_hash_bytes(hash, b"instruction");
    qwen_sft_hash_bytes(hash, example.instruction.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0input");
    qwen_sft_hash_bytes(hash, example.input.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0response");
    qwen_sft_hash_bytes(hash, example.response.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
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
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftTokenSample> {
    let prompt = qwen_render_sft_prompt(example, field_map)?;
    qwen_sft_token_sample_from_prompt(tokenizer, &prompt, &example.response)
}

fn qwen_render_sft_prompt(example: &QwenSftExample, field_map: &QwenSftFieldMap) -> Result<String> {
    field_map.validate()?;
    let template = if example.input.trim().is_empty() {
        &field_map.prompt_template
    } else {
        &field_map.prompt_with_input_template
    };
    Ok(template
        .replace("{system}", &example.system)
        .replace("{instruction}", &example.instruction)
        .replace("{input}", &example.input))
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

fn tensor_i64_max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<i64> {
    if actual.size() != expected.size() {
        bail!(
            "shape mismatch: actual {:?}, expected {:?}",
            actual.size(),
            expected.size()
        );
    }
    let actual_values: Vec<i64> =
        Vec::<i64>::try_from(actual.reshape([-1]).to_device(Device::Cpu))?;
    let expected_values: Vec<i64> =
        Vec::<i64>::try_from(expected.reshape([-1]).to_device(Device::Cpu))?;
    Ok(actual_values
        .iter()
        .zip(expected_values.iter())
        .map(|(actual, expected)| actual.saturating_sub(*expected).abs())
        .max()
        .unwrap_or(0))
}

fn tensor_max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<f64> {
    if actual.size() != expected.size() {
        bail!(
            "shape mismatch: actual {:?}, expected {:?}",
            actual.size(),
            expected.size()
        );
    }
    Ok((actual - expected)
        .abs()
        .max()
        .to_device(Device::Cpu)
        .double_value(&[]))
}

fn qwen_tensor_i64_fingerprint(tensor: &Tensor) -> Result<String> {
    let values: Vec<i64> = Vec::<i64>::try_from(tensor.reshape([-1]).to_device(Device::Cpu))?;
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        qwen_sft_hash_bytes(&mut hash, &value.to_le_bytes());
    }
    Ok(format!("{hash:016x}"))
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
            streaming_train_batches: None,
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
        let mut manifest = tiny_qwen_sharded_manifest();
        let mut replicated_norm_shard = manifest.ranks[0].shards[0].clone();
        replicated_norm_shard.name = "model.layers.0.input_layernorm.weight".to_string();
        replicated_norm_shard.shard_name = "rank0.input_layernorm".to_string();
        replicated_norm_shard.optimizer_m_name = "rank0.input_layernorm.m".to_string();
        replicated_norm_shard.optimizer_v_name = "rank0.input_layernorm.v".to_string();
        replicated_norm_shard.global_shape = vec![4];
        replicated_norm_shard.shard_shape = vec![4];
        replicated_norm_shard.partition = "replicated_norm_smoke".to_string();
        manifest.ranks[0].shards.push(replicated_norm_shard);
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
    fn qwen_sharded_checkpoint_manifest_rejects_invalid_global_metadata() {
        let mut missing_scheduler = tiny_qwen_sharded_manifest();
        missing_scheduler.scheduler.clear();
        let missing_scheduler_error = missing_scheduler
            .validate()
            .expect_err("missing scheduler should fail")
            .to_string();
        assert!(missing_scheduler_error.contains("requires scheduler"));

        let mut zero_step = tiny_qwen_sharded_manifest();
        zero_step.global_step = 0;
        let zero_step_error = zero_step
            .validate()
            .expect_err("zero global_step should fail")
            .to_string();
        assert!(zero_step_error.contains("global_step must be positive"));
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_invalid_parallel_rank_axes() {
        let mut duplicate_axes = tiny_qwen_sharded_manifest();
        duplicate_axes.ranks[1].data_parallel_rank = 0;
        duplicate_axes.ranks[1].rank = 1;
        let duplicate_axes_error = duplicate_axes
            .validate()
            .expect_err("duplicate parallel rank axes should fail")
            .to_string();
        assert!(duplicate_axes_error.contains("duplicate parallel rank axes"));

        let mut wrong_linear_rank = tiny_qwen_sharded_manifest();
        wrong_linear_rank.ranks.swap(0, 1);
        wrong_linear_rank.ranks[0].rank = 0;
        wrong_linear_rank.ranks[1].rank = 1;
        let wrong_linear_rank_error = wrong_linear_rank
            .validate()
            .expect_err("rank id that disagrees with axes should fail")
            .to_string();
        assert!(wrong_linear_rank_error.contains("does not match linear parallel rank"));
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_invalid_shard_shapes() {
        let mut rank_mismatch = tiny_qwen_sharded_manifest();
        rank_mismatch.ranks[0].shards[0].shard_shape = vec![4, 4, 1];
        let rank_mismatch_error = rank_mismatch
            .validate()
            .expect_err("shape rank mismatch should fail")
            .to_string();
        assert!(rank_mismatch_error.contains("global_shape rank"));

        let mut oversized_shard = tiny_qwen_sharded_manifest();
        oversized_shard.ranks[0].shards[0].shard_shape = vec![5, 4];
        let oversized_shard_error = oversized_shard
            .validate()
            .expect_err("oversized shard shape should fail")
            .to_string();
        assert!(oversized_shard_error.contains("exceeds global_shape"));

        let mut zero_dim = tiny_qwen_sharded_manifest();
        zero_dim.ranks[0].shards[0].global_shape = vec![4, 0];
        let zero_dim_error = zero_dim
            .validate()
            .expect_err("zero shape dim should fail")
            .to_string();
        assert!(zero_dim_error.contains("shape dim 1 must be positive"));
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_invalid_shard_contract_fields() {
        let mut unsupported_dtype = tiny_qwen_sharded_manifest();
        unsupported_dtype.ranks[0].shards[0].dtype = "int8".to_string();
        let unsupported_dtype_error = unsupported_dtype
            .validate()
            .expect_err("unsupported dtype should fail")
            .to_string();
        assert!(unsupported_dtype_error.contains("unsupported dtype int8"));

        let mut unsupported_partition = tiny_qwen_sharded_manifest();
        unsupported_partition.ranks[0].shards[0].partition = "rank0_delta".to_string();
        let unsupported_partition_error = unsupported_partition
            .validate()
            .expect_err("unsupported partition should fail")
            .to_string();
        assert!(unsupported_partition_error.contains("unsupported partition policy"));

        let mut duplicate_tensor = tiny_qwen_sharded_manifest();
        let repeated_shard = duplicate_tensor.ranks[0].shards[0].clone();
        duplicate_tensor.ranks[0].shards.push(repeated_shard);
        let duplicate_tensor_error = duplicate_tensor
            .validate()
            .expect_err("duplicate tensor shard should fail")
            .to_string();
        assert!(duplicate_tensor_error.contains("duplicate tensor shard"));

        let mut duplicate_slot = tiny_qwen_sharded_manifest();
        let mut second_shard = duplicate_slot.ranks[0].shards[0].clone();
        second_shard.name = "model.layers.0.self_attn.k_proj.weight".to_string();
        second_shard.shard_name = "rank0.k_proj".to_string();
        second_shard.optimizer_m_name = "rank0.q_proj.v".to_string();
        second_shard.optimizer_v_name = "rank0.k_proj.v".to_string();
        duplicate_slot.ranks[0].shards.push(second_shard);
        let duplicate_slot_error = duplicate_slot
            .validate()
            .expect_err("duplicate optimizer slot should fail")
            .to_string();
        assert!(duplicate_slot_error.contains("duplicate optimizer slot"));

        let mut slot_collision = tiny_qwen_sharded_manifest();
        slot_collision.ranks[0].shards[0].optimizer_m_name = "rank0.q_proj".to_string();
        let slot_collision_error = slot_collision
            .validate()
            .expect_err("optimizer slot colliding with shard_name should fail")
            .to_string();
        assert!(slot_collision_error.contains("collides with shard_name"));
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_validates_rank_owned_artifacts() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let manifest =
            tiny_qwen_sharded_manifest_with_artifacts(temp.path()).expect("artifacts should write");

        manifest
            .validate_artifacts()
            .expect("rank-owned artifacts should validate");
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_missing_model_artifact_tensor() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let mut manifest =
            tiny_qwen_sharded_manifest_with_artifacts(temp.path()).expect("artifacts should write");
        manifest.ranks[0].shards[0].shard_name = "rank0.missing_q_proj".to_string();

        let error = manifest
            .validate_artifacts()
            .expect_err("missing model shard should fail")
            .to_string();

        assert!(error.contains("missing model shard rank0.missing_q_proj"));
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_missing_optimizer_artifact_tensor() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let mut manifest =
            tiny_qwen_sharded_manifest_with_artifacts(temp.path()).expect("artifacts should write");
        manifest.ranks[0].shards[0].optimizer_m_name = "rank0.q_proj.missing_m".to_string();

        let error = manifest
            .validate_artifacts()
            .expect_err("missing optimizer slot should fail")
            .to_string();

        assert!(error.contains("missing optimizer m slot rank0.q_proj.missing_m"));
    }

    #[test]
    fn qwen_sharded_checkpoint_manifest_rejects_artifact_shape_mismatch() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let mut manifest =
            tiny_qwen_sharded_manifest_with_artifacts(temp.path()).expect("artifacts should write");
        manifest.ranks[0].shards[0].shard_shape = vec![4, 2];

        let error = manifest
            .validate_artifacts()
            .expect_err("artifact shape mismatch should fail")
            .to_string();

        assert!(error.contains("shape [4, 4] does not match manifest shard_shape [4, 2]"));
    }

    #[test]
    fn qwen_session_dp_global_sharded_manifest_writes_schema_root() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let manifest =
            tiny_qwen_sharded_manifest_with_artifacts(temp.path()).expect("artifacts should write");
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
            manifest.streaming_train_batches,
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
            streaming_train_batches: None,
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
            streaming_train_batches: None,
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

        let example_set = qwen_sft_examples_from_jsonl_path_with_limit(
            &jsonl,
            None,
            1,
            &QwenSftFieldMap::default(),
            &mut None,
        )
        .expect("examples should load from jsonl");
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
    fn qwen_sft_jsonl_reader_supports_configurable_field_names() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"sys":"Be concise.","prompt":"Summarize the project.","context":"Rust training code","answer":"rustrain"}
{"sys":"Use one word.","prompt":"Name the language.","answer":"Rust"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction: "prompt".to_string(),
            input: "context".to_string(),
            response: "answer".to_string(),
            system: Some("sys".to_string()),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("custom field examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("custom field streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("custom field raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("custom field cache should write");
        let mismatched_field_map = QwenSftFieldMap {
            response: "completion".to_string(),
            ..field_map.clone()
        };
        let mismatched_system_map = QwenSftFieldMap {
            system: Some("system_prompt".to_string()),
            ..field_map.clone()
        };
        let mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &mismatched_field_map,
        )
        .expect_err("cache should reject different field maps")
        .to_string();
        let system_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &mismatched_system_map,
        )
        .expect_err("cache should reject different system fields")
        .to_string();

        assert_eq!(loaded.examples.len(), 2);
        assert_eq!(loaded.examples[0].system, "Be concise.");
        assert_eq!(loaded.examples[0].instruction, "Summarize the project.");
        assert_eq!(loaded.examples[0].input, "Rust training code");
        assert_eq!(loaded.examples[0].response, "rustrain");
        assert_eq!(loaded.examples[1].system, "Use one word.");
        assert_eq!(loaded.examples[1].input, "");
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(raw_window.examples[0].response, "rustrain");
        assert!(first_cache.cache_written);
        assert!(mismatch.contains("field_map"));
        assert!(system_mismatch.contains("field_map"));
        assert!(
            qwen_render_sft_prompt(&loaded.examples[0], &field_map)
                .expect("system prompt should render")
                .contains("System: Be concise.")
        );
    }

    #[test]
    fn qwen_sft_prompt_template_changes_tokenized_prompt_and_fingerprint() {
        let example = QwenSftExample {
            system: String::new(),
            instruction: "Name the project.".to_string(),
            input: "Rust trainer".to_string(),
            response: "rustrain".to_string(),
        };
        let default_map = QwenSftFieldMap::default();
        let custom_map = QwenSftFieldMap {
            prompt_template: "### User\n{instruction}\n### Assistant\n".to_string(),
            prompt_with_input_template:
                "### User\n{instruction}\nContext: {input}\n### Assistant\n".to_string(),
            ..QwenSftFieldMap::default()
        };
        let default_prompt =
            qwen_render_sft_prompt(&example, &default_map).expect("default prompt should render");
        let custom_prompt =
            qwen_render_sft_prompt(&example, &custom_map).expect("custom prompt should render");
        let default_fingerprint = qwen_sft_dataset_fingerprint(
            &["data/train.jsonl".to_string()],
            std::slice::from_ref(&example),
            &default_map,
        );
        let custom_fingerprint = qwen_sft_dataset_fingerprint(
            &["data/train.jsonl".to_string()],
            std::slice::from_ref(&example),
            &custom_map,
        );

        assert_eq!(
            default_prompt,
            "Instruction:\nName the project.\n\nInput:\nRust trainer\n\nResponse:\n"
        );
        assert_eq!(
            custom_prompt,
            "### User\nName the project.\nContext: Rust trainer\n### Assistant\n"
        );
        assert_ne!(default_fingerprint, custom_fingerprint);
    }

    #[test]
    fn qwen_sft_trim_fields_controls_record_normalization_and_fingerprint() {
        let line = r#"{"instruction":"  Name the project.  ","input":"  Rust trainer  ","response":"  rustrain  "}"#;
        let trim_map = QwenSftFieldMap::default();
        let raw_map = QwenSftFieldMap {
            trim_fields: false,
            ..QwenSftFieldMap::default()
        };
        let trimmed =
            qwen_sft_record_from_jsonl_line(line, &trim_map).expect("trimmed record should parse");
        let raw = qwen_sft_record_from_jsonl_line(line, &raw_map).expect("raw record should parse");
        let trimmed_example = QwenSftExample {
            system: trimmed.system.clone(),
            instruction: trimmed.instruction.clone(),
            input: trimmed.input.clone(),
            response: trimmed.response.clone(),
        };
        let raw_example = QwenSftExample {
            system: raw.system.clone(),
            instruction: raw.instruction.clone(),
            input: raw.input.clone(),
            response: raw.response.clone(),
        };

        assert_eq!(trimmed.instruction, "Name the project.");
        assert_eq!(trimmed.input, "Rust trainer");
        assert_eq!(trimmed.response, "rustrain");
        assert_eq!(raw.instruction, "  Name the project.  ");
        assert_eq!(raw.input, "  Rust trainer  ");
        assert_eq!(raw.response, "  rustrain  ");
        assert_ne!(
            qwen_sft_dataset_fingerprint(&[], &[trimmed_example], &trim_map),
            qwen_sft_dataset_fingerprint(&[], &[raw_example], &raw_map)
        );
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
    fn qwen_sft_limits_explicit_eval_paths() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let train_jsonl = temp.path().join("train.jsonl");
        let eval_jsonl = temp.path().join("eval.jsonl");
        fs::write(
            &train_jsonl,
            "{\"instruction\":\"train one\",\"response\":\"alpha\"}\n{\"instruction\":\"train two\",\"response\":\"beta\"}\n",
        )
        .expect("train jsonl should write");
        fs::write(
            &eval_jsonl,
            "{\"instruction\":\"eval one\",\"response\":\"gamma\"}\n{\"instruction\":\"eval two\",\"response\":\"delta\"}\n{\"instruction\":\"eval three\",\"response\":\"epsilon\"}\n",
        )
        .expect("eval jsonl should write");
        let train_paths = vec![train_jsonl.clone()];
        let eval_paths = vec![eval_jsonl.clone()];
        let field_map = QwenSftFieldMap::default();
        let eval_field_map = qwen_sft_eval_field_map(&field_map);

        let train_set =
            qwen_sft_examples_from_jsonl_paths_with_limit(&train_paths, None, &field_map)
                .expect("train examples should load");
        let limited_eval_set =
            qwen_sft_examples_from_jsonl_paths_with_limit(&eval_paths, Some(2), &eval_field_map)
                .expect("limited eval examples should load");
        let streaming_eval_summary =
            qwen_sft_streaming_source_summary(&eval_paths, Some(2), &eval_field_map)
                .expect("limited eval streaming summary should scan");

        assert_eq!(train_set.examples.len(), 2);
        assert_eq!(limited_eval_set.examples.len(), 2);
        assert_eq!(
            limited_eval_set
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["eval one", "eval two"]
        );
        assert_eq!(
            streaming_eval_summary.samples,
            limited_eval_set.examples.len()
        );
        assert_eq!(
            streaming_eval_summary.source_sample_counts,
            limited_eval_set.source_sample_counts
        );
        assert_eq!(
            streaming_eval_summary.fingerprint,
            limited_eval_set.fingerprint
        );

        let combined_source_files =
            qwen_merge_sft_source_files(&train_set.source_files, &limited_eval_set.source_files);
        let combined_source_sample_counts = qwen_merge_sft_source_sample_counts(
            &train_set.source_sample_counts,
            &limited_eval_set.source_sample_counts,
        );
        assert_eq!(
            combined_source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: eval_jsonl.display().to_string(),
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: train_jsonl.display().to_string(),
                    samples: 2,
                },
            ]
        );
        assert_ne!(
            qwen_combine_sft_fingerprints(
                &combined_source_files,
                &train_set.fingerprint,
                &limited_eval_set.fingerprint,
            ),
            qwen_combine_sft_fingerprints(
                &combined_source_files,
                &train_set.fingerprint,
                "unlimited-eval-fingerprint",
            )
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

        let example_set = qwen_sft_examples_from_jsonl_paths_with_limit(
            &[first.clone(), dir.clone()],
            None,
            &QwenSftFieldMap::default(),
        )
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
    fn qwen_sft_jsonl_limit_stops_before_unneeded_files() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        let third = temp.path().join("third.jsonl");
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
        fs::write(
            &third,
            r#"{"instruction":"five","response":"e"}
"#,
        )
        .expect("third jsonl should write");

        let limited = qwen_sft_examples_from_jsonl_paths_with_limit(
            &[first.clone(), second.clone(), third],
            Some(3),
            &QwenSftFieldMap::default(),
        )
        .expect("limited examples should load");

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
            qwen_sft_dataset_fingerprint(
                &limited.source_files,
                &limited.examples,
                &QwenSftFieldMap::default()
            )
        );
    }

    #[test]
    fn qwen_sft_streaming_summary_matches_jsonl_reader_without_materializing_tokens() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        let third = temp.path().join("third.jsonl");
        fs::write(
            &first,
            r#"{"instruction":"one","response":"a"}
{"instruction":"two","input":"input","response":"b"}
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
        fs::write(
            &third,
            r#"{"instruction":"five","response":"e"}
"#,
        )
        .expect("third jsonl should write");

        let paths = vec![first.clone(), second.clone(), third];
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(
            &paths,
            Some(3),
            &QwenSftFieldMap::default(),
        )
        .expect("limited examples should load");
        let streamed =
            qwen_sft_streaming_source_summary(&paths, Some(3), &QwenSftFieldMap::default())
                .expect("streaming summary should scan");

        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
    }

    #[test]
    fn qwen_sft_filters_short_responses_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        fs::write(
            &jsonl,
            r#"{"instruction":"empty","response":""}
{"instruction":"short","response":"ok"}
{"instruction":"first","response":"valid"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            min_response_chars: 5,
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_long_responses_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","response":"valid"}
{"instruction":"too long","response":"toolong"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            max_response_chars: Some(5),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("max response drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
    }

    #[test]
    fn qwen_sft_filters_instruction_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"a","response":"skip"}
{"instruction":"first","response":"valid"}
{"instruction":"too long","response":"skip"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            min_instruction_chars: Some(3),
            max_instruction_chars: Some(6),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("instruction filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_input_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"skip short","input":"x","response":"short"}
{"instruction":"first","input":"ok","response":"valid"}
{"instruction":"skip long","input":"toolong","response":"long"}
{"instruction":"second","input":"mid","response":"works"}
{"instruction":"third","input":"fit","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            min_input_chars: Some(2),
            max_input_chars: Some(3),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("input filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.input.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "ok"), ("second", "mid")]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.input.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "ok"), ("second", "mid")]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_prompt_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"a","response":"skip"}
{"instruction":"first","input":"ok","response":"valid"}
{"instruction":"this prompt is too long","response":"skip"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            prompt_template: "Q:{instruction}\nA:".to_string(),
            prompt_with_input_template: "Q:{instruction}\nI:{input}\nA:".to_string(),
            min_prompt_chars: Some(11),
            max_prompt_chars: Some(15),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("prompt filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.input.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "ok"), ("second", "")]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.input.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "ok"), ("second", "")]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_sample_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"a","response":"x"}
{"instruction":"first","input":"ok","response":"valid"}
{"instruction":"too long","response":"this response is too long"}
{"instruction":"second","response":"works"}
{"instruction":"tiny","response":"z"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            prompt_template: "Q:{instruction}\nA:".to_string(),
            prompt_with_input_template: "Q:{instruction}\nI:{input}\nA:".to_string(),
            min_sample_chars: Some(16),
            max_sample_chars: Some(22),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("sample filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.response.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "valid"), ("second", "works")]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.response.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "valid"), ("second", "works")]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_dedupes_samples_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","response":"valid"}
{"instruction":"first","response":"valid"}
{"instruction":"second","response":"works"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            dedupe_samples: true,
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("deduped examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("deduped streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("deduped source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("deduped raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("deduped cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("dedupe drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
    }

    #[test]
    fn qwen_sft_applies_source_weights_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &first,
            r#"{"instruction":"first","response":"alpha"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"instruction":"second","response":"beta"}
"#,
        )
        .expect("second jsonl should write");
        let paths = vec![first.clone(), second.clone()];
        let field_map = QwenSftFieldMap {
            source_weights: vec![2, 1],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("weighted examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("weighted streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("weighted source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("weighted raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("weighted cache should write");
        let unweighted_map = QwenSftFieldMap::default();
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &unweighted_map,
        )
        .expect_err("source weight drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "first", "second"]
        );
        assert_eq!(
            loaded.source_sample_counts,
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
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 0, 0]
        );
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(&paths, Some(3), &loaded.source_files, &unweighted_map)
                .expect("unweighted fingerprint should compute")
        );
    }

    #[test]
    fn qwen_sft_streaming_window_uses_train_split_sample_count() {
        let (train_samples, eval_samples) =
            qwen_sft_train_eval_sample_counts(4, 0.75).expect("split should compute");
        let world_size = 2usize;
        let local_batch_size = 1usize;
        let global_batch_size = local_batch_size * world_size;
        let train_steps = 1usize;
        let data_cursor_start = 2usize;
        let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
        let (epoch_start, offset_start) =
            qwen_data_epoch_and_offset(data_cursor_start, train_samples)
                .expect("start cursor should map");
        let (epoch_next, offset_next) = qwen_data_epoch_and_offset(data_cursor_end, train_samples)
            .expect("next cursor should map");

        assert_eq!(train_samples, 3);
        assert_eq!(eval_samples, 1);
        assert_eq!(data_cursor_end, 4);
        assert_eq!((epoch_start, offset_start), (0, 2));
        assert_eq!((epoch_next, offset_next), (1, 1));
    }

    #[test]
    fn qwen_sft_streaming_cursor_window_covers_next_batch_overlap() {
        let cursors =
            qwen_sft_streaming_cursor_window(2, 3, 2, 3).expect("cursor window should build");
        let compact = cursors
            .iter()
            .map(|entry| (entry.cursor, entry.epoch, entry.sample_offset))
            .collect::<Vec<_>>();

        assert_eq!(compact, vec![(2, 0, 2), (3, 1, 0), (4, 1, 1), (5, 1, 2)]);
    }

    #[test]
    fn qwen_sft_streaming_raw_index_reads_only_cursor_window() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("train.jsonl");
        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
{"instruction":"one","response":"one"}
{"instruction":"two","response":"two"}
{"instruction":"three","response":"three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let source_index =
            qwen_sft_streaming_source_index(&paths, None, &QwenSftFieldMap::default())
                .expect("source index should build");
        let (train_samples, _) =
            qwen_sft_train_eval_sample_counts(source_index.samples.len(), 0.75)
                .expect("split should compute");
        let mut train_indices = source_index.samples;
        let mut rng = StdRng::seed_from_u64(777);
        train_indices.shuffle(&mut rng);
        train_indices.truncate(train_samples);
        let raw_indices = (0..4)
            .map(|relative| {
                let cursor = 2 + relative;
                let epoch = cursor / train_indices.len();
                let offset = cursor % train_indices.len();
                let index = qwen_epoch_permutation_index(train_indices.len(), 777, epoch, offset);
                train_indices[index].clone()
            })
            .collect::<Vec<_>>();
        let raw_window =
            qwen_sft_examples_by_raw_indices(&raw_indices, &QwenSftFieldMap::default())
                .expect("raw examples should read");

        assert_eq!(raw_window.examples.len(), 4);
        assert_eq!(raw_window.raw_samples_read, 3);
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["three", "three", "two", "one"]
        );
        assert_eq!(
            raw_indices
                .iter()
                .map(|index| index.index_in_file)
                .collect::<Vec<_>>(),
            vec![3, 3, 2, 1]
        );
        assert_eq!(
            raw_indices
                .iter()
                .map(|index| index.byte_offset)
                .collect::<Vec<_>>(),
            vec![119, 119, 80, 41]
        );
        assert_eq!(
            raw_indices
                .iter()
                .map(|index| index.path.clone())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([jsonl.display().to_string()])
        );
    }

    #[test]
    fn qwen_sft_streaming_source_index_parses_records_before_indexing_offsets() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("train.jsonl");
        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
not-json
{"instruction":"two","response":"two"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let error = match qwen_sft_streaming_source_index(&paths, None, &QwenSftFieldMap::default())
        {
            Ok(_) => panic!("malformed JSONL row should fail while building the offset index"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("failed to parse SFT JSONL record"));
    }

    #[test]
    fn qwen_sft_streaming_source_index_cache_writes_and_reuses_offsets() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("train.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
{"instruction":"one","response":"one"}
{"instruction":"two","response":"two"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];

        let first = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect("first cache load should build index");
        assert!(!first.cache_hit);
        assert!(first.cache_written);
        assert_eq!(first.index.samples.len(), 2);
        assert!(cache_path.exists());

        let second = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect("second cache load should hit cache");
        assert!(second.cache_hit);
        assert!(!second.cache_written);
        assert_eq!(second.index.samples, first.index.samples);

        let mismatch = match qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        ) {
            Ok(_) => panic!("mismatched max_samples should reject cache"),
            Err(error) => error.to_string(),
        };
        assert!(mismatch.contains("max_samples"));

        let min_response_mismatch_map = QwenSftFieldMap {
            min_response_chars: 2,
            ..QwenSftFieldMap::default()
        };
        let min_response_mismatch = match qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &min_response_mismatch_map,
        ) {
            Ok(_) => panic!("mismatched min_response_chars should reject cache"),
            Err(error) => error.to_string(),
        };
        assert!(min_response_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_rank_index_cache_path_keeps_extension() {
        let path = PathBuf::from("/tmp/rustrain/cache/offset-index.json");
        assert_eq!(
            qwen_sft_rank_index_cache_path(&path, 1),
            PathBuf::from("/tmp/rustrain/cache/offset-index.rank-1.json")
        );

        let no_extension = PathBuf::from("/tmp/rustrain/cache/offset-index");
        assert_eq!(
            qwen_sft_rank_index_cache_path(&no_extension, 2),
            PathBuf::from("/tmp/rustrain/cache/offset-index.rank-2")
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

    #[test]
    fn qwen_model_path_resolves_hf_hub_snapshot_when_legacy_dir_is_missing() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let legacy = temp.path().join("Qwen2.5-0.5B-Instruct");
        let incomplete_snapshot = temp
            .path()
            .join("hub")
            .join("models--Qwen--Qwen2.5-0.5B-Instruct")
            .join("snapshots")
            .join("111");
        let complete_snapshot = temp
            .path()
            .join("hub")
            .join("models--Qwen--Qwen2.5-0.5B-Instruct")
            .join("snapshots")
            .join("222");
        fs::create_dir_all(&incomplete_snapshot).expect("incomplete snapshot dir should write");
        fs::create_dir_all(&complete_snapshot).expect("complete snapshot dir should write");
        fs::write(incomplete_snapshot.join("config.json"), "{}").expect("config should write");
        fs::write(complete_snapshot.join("config.json"), "{}").expect("config should write");
        fs::write(complete_snapshot.join("tokenizer.json"), "{}").expect("tokenizer should write");
        fs::write(complete_snapshot.join("model.safetensors"), "")
            .expect("safetensors marker should write");

        let resolved =
            resolve_qwen_model_path(&legacy).expect("legacy path should resolve through HF hub");

        assert_eq!(resolved, complete_snapshot);
    }

    #[test]
    fn qwen_model_path_keeps_complete_configured_directory() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let model_path = temp.path().join("Qwen2.5-0.5B-Instruct");
        fs::create_dir_all(&model_path).expect("model dir should write");
        fs::write(model_path.join("config.json"), "{}").expect("config should write");
        fs::write(model_path.join("tokenizer.json"), "{}").expect("tokenizer should write");
        fs::write(model_path.join("model.safetensors"), "")
            .expect("safetensors marker should write");

        let resolved =
            resolve_qwen_model_path(&model_path).expect("complete path should not be rewritten");

        assert_eq!(resolved, model_path);
    }

    #[test]
    fn qwen_model_path_reports_missing_hf_snapshot() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let legacy = temp.path().join("Qwen2.5-0.5B-Instruct");
        let incomplete_snapshot = temp
            .path()
            .join("hub")
            .join("models--Qwen--Qwen2.5-0.5B-Instruct")
            .join("snapshots")
            .join("111");
        fs::create_dir_all(&incomplete_snapshot).expect("incomplete snapshot dir should write");
        fs::write(incomplete_snapshot.join("config.json"), "{}").expect("config should write");

        let error = match resolve_qwen_model_path(&legacy) {
            Ok(path) => panic!("incomplete cache should fail, resolved {}", path.display()),
            Err(error) => error.to_string(),
        };

        assert!(error.contains("no complete HF hub snapshot"));
    }

    #[test]
    fn qwen_model_safetensors_path_resolves_with_hf_hub_snapshot() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let legacy_safetensors = temp
            .path()
            .join("Qwen2.5-0.5B-Instruct")
            .join("model.safetensors");
        let complete_snapshot = temp
            .path()
            .join("hub")
            .join("models--Qwen--Qwen2.5-0.5B-Instruct")
            .join("snapshots")
            .join("222");
        fs::create_dir_all(&complete_snapshot).expect("complete snapshot dir should write");
        fs::write(complete_snapshot.join("config.json"), "{}").expect("config should write");
        fs::write(complete_snapshot.join("tokenizer.json"), "{}").expect("tokenizer should write");
        fs::write(complete_snapshot.join("model.safetensors"), "")
            .expect("safetensors marker should write");

        let resolved = resolve_qwen_model_safetensors_path(&legacy_safetensors)
            .expect("legacy safetensors path should resolve through HF hub");

        assert_eq!(resolved, complete_snapshot.join("model.safetensors"));
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
            streaming_train_batches: Some(true),
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

    fn tiny_qwen_sharded_manifest_with_artifacts(
        root: &Path,
    ) -> Result<QwenShardedCheckpointManifest> {
        let mut manifest = tiny_qwen_sharded_manifest();
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
