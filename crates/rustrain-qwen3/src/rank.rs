//! rank_smoke module - split from qwen_module.rs

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
use rustrain_checkpoint::safetensors::{read_safetensors_dir, read_safetensors_map, tensor};
use rustrain_core::runtime::{
    Config, DataConfig as RuntimeDataConfig, DataKind as RuntimeDataKind, Device as RuntimeDevice,
    FieldAffix, FieldCaseTransform, FieldCaseTransformKind, FieldDefault, FieldDefaultTarget,
    FieldRegexFilter, FieldRegexReplacement, FieldReplacement, FieldReplacementTarget, FieldSplit,
    FieldSplitSide, FieldStrip, FieldTransform, FieldTransformOp, FieldTruncation,
    LoraConfig as RuntimeLoraConfig, LrScheduler, RunPaths, load_config,
};
use rustrain_nccl::nccl as nccl_smoke;

use crate::generate::*;
use crate::lora::*;
use crate::model::*;
use crate::session::*;
use crate::sft::*;

#[derive(Debug, Serialize)]
pub(crate) struct QwenTpLinearRankSummary {
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
    pub(crate) local_rank: usize,
    pub(crate) model_path: String,
    pub(crate) input_shape: Vec<i64>,
    pub(crate) projections: Vec<QwenTpProjectionShardSummary>,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenTpProjectionShardSummary {
    pub(crate) name: String,
    pub(crate) tensor_name: String,
    pub(crate) full_output_shape: Vec<i64>,
    pub(crate) shard_output_shape: Vec<i64>,
    pub(crate) shard_start: i64,
    pub(crate) shard_end: i64,
    pub(crate) max_abs: Option<f64>,
    pub(crate) mean_abs: Option<f64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenTpAttentionRankSummary {
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
    pub(crate) local_rank: usize,
    pub(crate) model_path: String,
    pub(crate) input_shape: Vec<i64>,
    pub(crate) q_head_start: i64,
    pub(crate) q_head_end: i64,
    pub(crate) kv_head_start: i64,
    pub(crate) kv_head_end: i64,
    pub(crate) context_shard_shape: Vec<i64>,
    pub(crate) output_contribution_shape: Vec<i64>,
    pub(crate) full_output_shape: Vec<i64>,
    pub(crate) max_abs: Option<f64>,
    pub(crate) mean_abs: Option<f64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenTpAttentionNcclRankSummary {
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
    pub(crate) local_rank: usize,
    pub(crate) model_path: String,
    pub(crate) input_shape: Vec<i64>,
    pub(crate) q_head_start: i64,
    pub(crate) q_head_end: i64,
    pub(crate) kv_head_start: i64,
    pub(crate) kv_head_end: i64,
    pub(crate) context_shard_shape: Vec<i64>,
    pub(crate) output_contribution_shape: Vec<i64>,
    pub(crate) reduced_output_shape: Vec<i64>,
    pub(crate) full_output_shape: Vec<i64>,
    pub(crate) max_abs: f64,
    pub(crate) mean_abs: f64,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenTpMlpRankSummary {
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
    pub(crate) local_rank: usize,
    pub(crate) model_path: String,
    pub(crate) input_shape: Vec<i64>,
    pub(crate) intermediate_start: i64,
    pub(crate) intermediate_end: i64,
    pub(crate) activation_shard_shape: Vec<i64>,
    pub(crate) output_contribution_shape: Vec<i64>,
    pub(crate) full_output_shape: Vec<i64>,
    pub(crate) max_abs: Option<f64>,
    pub(crate) mean_abs: Option<f64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenTpMlpNcclRankSummary {
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
    pub(crate) local_rank: usize,
    pub(crate) model_path: String,
    pub(crate) input_shape: Vec<i64>,
    pub(crate) intermediate_start: i64,
    pub(crate) intermediate_end: i64,
    pub(crate) activation_shard_shape: Vec<i64>,
    pub(crate) output_contribution_shape: Vec<i64>,
    pub(crate) reduced_output_shape: Vec<i64>,
    pub(crate) full_output_shape: Vec<i64>,
    pub(crate) max_abs: f64,
    pub(crate) mean_abs: f64,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenSessionTpRankSummary {
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
    pub(crate) local_rank: usize,
    pub(crate) model_path: String,
    pub(crate) resume_from: Option<String>,
    pub(crate) resumed_sharded_checkpoint: bool,
    pub(crate) resume_global_step: Option<u64>,
    pub(crate) resume_rank_manifest_output: Option<String>,
    pub(crate) resume_model_safetensors: Option<String>,
    pub(crate) resume_optimizer_safetensors: Option<String>,
    pub(crate) resume_sharded_manifest_tensor_count: Option<usize>,
    pub(crate) resume_restore_max_abs: Option<f64>,
    pub(crate) resume_restore_mean_abs: Option<f64>,
    pub(crate) resume_next_update_max_abs: Option<f64>,
    pub(crate) resume_next_update_mean_abs: Option<f64>,
    pub(crate) tensor_model_parallel_size: usize,
    pub(crate) data_parallel_size: usize,
    pub(crate) attention_q_head_start: i64,
    pub(crate) attention_q_head_end: i64,
    pub(crate) attention_kv_head_start: i64,
    pub(crate) attention_kv_head_end: i64,
    pub(crate) attention_context_shard_shape: Vec<i64>,
    pub(crate) attention_reduced_output_shape: Vec<i64>,
    pub(crate) attention_max_abs: f64,
    pub(crate) attention_mean_abs: f64,
    pub(crate) attention_train_initial_loss: f64,
    pub(crate) attention_train_final_loss: f64,
    pub(crate) attention_train_loss_improved: bool,
    pub(crate) attention_train_learning_rate: f64,
    pub(crate) attention_train_q_grad_norm: f64,
    pub(crate) attention_train_k_grad_norm: f64,
    pub(crate) attention_train_v_grad_norm: f64,
    pub(crate) attention_train_o_grad_norm: f64,
    pub(crate) layer0_reduced_output_shape: Vec<i64>,
    pub(crate) layer0_max_abs: f64,
    pub(crate) layer0_mean_abs: f64,
    pub(crate) layer0_train_initial_loss: f64,
    pub(crate) layer0_train_final_loss: f64,
    pub(crate) layer0_train_loss_improved: bool,
    pub(crate) layer0_train_learning_rate: f64,
    pub(crate) layer0_train_q_grad_norm: f64,
    pub(crate) layer0_train_k_grad_norm: f64,
    pub(crate) layer0_train_v_grad_norm: f64,
    pub(crate) layer0_train_o_grad_norm: f64,
    pub(crate) layer0_train_gate_grad_norm: f64,
    pub(crate) layer0_train_up_grad_norm: f64,
    pub(crate) layer0_train_down_grad_norm: f64,
    pub(crate) causal_train_input_shape: Vec<i64>,
    pub(crate) causal_train_full_loss: f64,
    pub(crate) causal_train_initial_loss: f64,
    pub(crate) causal_train_initial_loss_delta: f64,
    pub(crate) causal_train_final_loss: f64,
    pub(crate) causal_train_loss_improved: bool,
    pub(crate) causal_train_learning_rate: f64,
    pub(crate) causal_train_q_grad_norm: f64,
    pub(crate) causal_train_k_grad_norm: f64,
    pub(crate) causal_train_v_grad_norm: f64,
    pub(crate) causal_train_o_grad_norm: f64,
    pub(crate) causal_train_gate_grad_norm: f64,
    pub(crate) causal_train_up_grad_norm: f64,
    pub(crate) causal_train_down_grad_norm: f64,
    pub(crate) causal_train_q_grad_sum: f64,
    pub(crate) causal_train_k_grad_sum: f64,
    pub(crate) causal_train_v_grad_sum: f64,
    pub(crate) causal_train_o_grad_sum: f64,
    pub(crate) causal_train_gate_grad_sum: f64,
    pub(crate) causal_train_up_grad_sum: f64,
    pub(crate) causal_train_down_grad_sum: f64,
    pub(crate) sharded_rank_manifest_output: String,
    pub(crate) sharded_global_manifest_output: String,
    pub(crate) sharded_manifest_tensor_count: usize,
    pub(crate) sharded_restore_max_abs: f64,
    pub(crate) sharded_restore_mean_abs: f64,
    pub(crate) sharded_next_update_max_abs: f64,
    pub(crate) sharded_next_update_mean_abs: f64,
    pub(crate) mlp_intermediate_start: i64,
    pub(crate) mlp_intermediate_end: i64,
    pub(crate) mlp_activation_shard_shape: Vec<i64>,
    pub(crate) mlp_reduced_output_shape: Vec<i64>,
    pub(crate) mlp_max_abs: f64,
    pub(crate) mlp_mean_abs: f64,
    pub(crate) mlp_train_initial_loss: f64,
    pub(crate) mlp_train_final_loss: f64,
    pub(crate) mlp_train_loss_improved: bool,
    pub(crate) mlp_train_learning_rate: f64,
    pub(crate) mlp_train_gate_grad_norm: f64,
    pub(crate) mlp_train_up_grad_norm: f64,
    pub(crate) mlp_train_down_grad_norm: f64,
}

pub(crate) struct QwenTpAttentionContribution {
    pub(crate) input: Tensor,
    pub(crate) context_shard: Tensor,
    pub(crate) output_contribution: Tensor,
    pub(crate) full_output: Tensor,
    pub(crate) q_head_start: i64,
    pub(crate) q_heads_per_rank: i64,
    pub(crate) kv_head_start: i64,
    pub(crate) kv_heads_per_rank: i64,
}

pub(crate) struct QwenTpAttentionShardWeights<'a> {
    pub(crate) q_proj: &'a Tensor,
    pub(crate) q_norm: &'a Tensor,
    pub(crate) k_proj: &'a Tensor,
    pub(crate) k_norm: &'a Tensor,
    pub(crate) v_proj: &'a Tensor,
    pub(crate) o_proj: &'a Tensor,
}

pub(crate) struct QwenSessionTpLayer0SgdUpdate {
    pub(crate) initial_loss: f64,
    pub(crate) final_loss: f64,
    pub(crate) initial_output: Tensor,
    pub(crate) final_output: Tensor,
    pub(crate) q_grad_norm: f64,
    pub(crate) k_grad_norm: f64,
    pub(crate) v_grad_norm: f64,
    pub(crate) o_grad_norm: f64,
    pub(crate) gate_grad_norm: f64,
    pub(crate) up_grad_norm: f64,
    pub(crate) down_grad_norm: f64,
}

pub(crate) struct QwenSessionTpCausalLmSgdUpdate {
    pub(crate) full_loss: f64,
    pub(crate) initial_loss: f64,
    pub(crate) final_loss: f64,
    pub(crate) learning_rate: f64,
    pub(crate) q_grad: Tensor,
    pub(crate) k_grad: Tensor,
    pub(crate) v_grad: Tensor,
    pub(crate) o_grad: Tensor,
    pub(crate) gate_grad: Tensor,
    pub(crate) up_grad: Tensor,
    pub(crate) down_grad: Tensor,
    pub(crate) q_grad_norm: f64,
    pub(crate) k_grad_norm: f64,
    pub(crate) v_grad_norm: f64,
    pub(crate) o_grad_norm: f64,
    pub(crate) gate_grad_norm: f64,
    pub(crate) up_grad_norm: f64,
    pub(crate) down_grad_norm: f64,
    pub(crate) q_grad_sum: f64,
    pub(crate) k_grad_sum: f64,
    pub(crate) v_grad_sum: f64,
    pub(crate) o_grad_sum: f64,
    pub(crate) gate_grad_sum: f64,
    pub(crate) up_grad_sum: f64,
    pub(crate) down_grad_sum: f64,
}

pub(crate) struct QwenSessionTpFocusedLayer0Shards {
    pub(crate) input_norm: Tensor,
    pub(crate) post_attention_norm: Tensor,
    pub(crate) q: Tensor,
    pub(crate) k: Tensor,
    pub(crate) v: Tensor,
    pub(crate) o: Tensor,
    pub(crate) gate: Tensor,
    pub(crate) up: Tensor,
    pub(crate) down: Tensor,
}

pub(crate) struct QwenSessionTpFocusedResume {
    pub(crate) global_step: u64,
    pub(crate) rank_manifest_output: String,
    pub(crate) model_safetensors: String,
    pub(crate) optimizer_safetensors: String,
    pub(crate) tensor_count: usize,
    pub(crate) restore_diff: DiffStats,
    pub(crate) next_update_diff: DiffStats,
}
#[derive(Debug, Serialize)]
pub(crate) struct QwenSessionDpRankSummary {
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
    pub(crate) dtype: String,
    pub(crate) resume_from: Option<String>,
    pub(crate) resumed_checkpoint: bool,
    pub(crate) data_kind: Option<String>,
    pub(crate) local_batch_size: usize,
    pub(crate) sequence_tokens: usize,
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
    pub(crate) tensor_count: usize,
    pub(crate) steps: usize,
    pub(crate) learning_rate: f64,
    pub(crate) tokens_per_second: f64,
    pub(crate) samples_per_second: f64,
    pub(crate) memory_rss_mb: Option<f64>,
    pub(crate) gpu_memory_allocated_mb: Option<f64>,
    pub(crate) max_grad_delta: f32,
    pub(crate) loss_delta: f64,
    pub(crate) local_loss: f64,
    pub(crate) post_update_loss: f64,
    pub(crate) global_loss: f64,
    pub(crate) global_post_update_loss: f64,
    pub(crate) global_loss_improved: bool,
    pub(crate) global_step_losses: Vec<f64>,
    pub(crate) expected_loss: f64,
    pub(crate) checkpoint_written: bool,
    pub(crate) checkpoint_path: String,
    pub(crate) delta_output: String,
    pub(crate) optimizer_output: String,
    pub(crate) manifest_output: String,
    pub(crate) sharded_rank_manifest_output: String,
    pub(crate) sharded_global_manifest_output: String,
    pub(crate) reloaded_loss: f64,
    pub(crate) reload_delta: f64,
    pub(crate) sharded_reloaded_loss: f64,
    pub(crate) sharded_reload_delta: f64,
    pub(crate) continuous_next_loss: f64,
    pub(crate) resumed_next_loss: f64,
    pub(crate) next_step_delta: f64,
    pub(crate) sharded_continuous_next_loss: f64,
    pub(crate) sharded_resumed_next_loss: f64,
    pub(crate) sharded_next_step_delta: f64,
    pub(crate) trainable_tensors: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenSessionDpDataPlanSummary {
    pub(crate) config_path: String,
    pub(crate) model_path: String,
    pub(crate) world_size: usize,
    pub(crate) local_batch_size: usize,
    pub(crate) global_batch_size: usize,
    pub(crate) train_steps: usize,
    pub(crate) required_batches: usize,
    pub(crate) data_cursor_start: usize,
    pub(crate) data_cursor_end: usize,
    pub(crate) data_cursor_next: usize,
    pub(crate) data_epoch_start: usize,
    pub(crate) data_epoch_end: usize,
    pub(crate) data_epoch_next: usize,
    pub(crate) data_sample_offset_start: usize,
    pub(crate) data_sample_offset_end: usize,
    pub(crate) data_sample_offset_next: usize,
    pub(crate) dataset_total_samples: usize,
    pub(crate) dataset_total_tokens: usize,
    pub(crate) dataset_train_samples: usize,
    pub(crate) dataset_eval_samples: usize,
    pub(crate) dataset_source_files: Vec<String>,
    pub(crate) dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) dataset_fingerprint: String,
    pub(crate) dataset_order_seed: u64,
    pub(crate) dataset_shuffle: bool,
    pub(crate) streaming_train_batches: bool,
}
pub fn qwen_tp_linear_rank(model_path: &Path, output_dir: PathBuf) -> Result<()> {
    let model_path = resolve_qwen3_model_path(model_path)?;
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("Qwen TP linear rank expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    let device = Device::Cuda(local_rank);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let weights = read_safetensors_dir(&model_path)?;
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
            Some("model.layers.0.self_attn.q_norm.weight"),
        ),
        (
            "k_proj",
            "model.layers.0.self_attn.k_proj.weight",
            Some("model.layers.0.self_attn.k_norm.weight"),
        ),
        ("v_proj", "model.layers.0.self_attn.v_proj.weight", None),
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
            bail!("Qwen TP linear rank requires {name} output size divisible by WORLD_SIZE");
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

pub fn qwen_tp_attention_rank(model_path: &Path, output_dir: PathBuf) -> Result<()> {
    let model_path = resolve_qwen3_model_path(model_path)?;
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("Qwen TP attention rank expects WORLD_SIZE=2");
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

pub fn qwen_tp_attention_nccl_rank(model_path: &Path, output_dir: PathBuf) -> Result<()> {
    let model_path = resolve_qwen3_model_path(model_path)?;
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("Qwen TP attention NCCL rank expects WORLD_SIZE=2");
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

pub(crate) fn qwen_tp_attention_contribution(
    model_path: &Path,
    device: Device,
    rank: usize,
    world_size: usize,
    context: &str,
) -> Result<QwenTpAttentionContribution> {
    let config = read_qwen3_runtime_config(&model_path.join("config.json"))?;
    if config.num_attention_heads % world_size as i64 != 0 {
        bail!("{context} requires attention heads divisible by WORLD_SIZE");
    }
    if config.num_key_value_heads % world_size as i64 != 0 {
        bail!("{context} requires KV heads divisible by WORLD_SIZE");
    }
    let weights = read_safetensors_dir(&model_path)?;
    let q_proj = tensor(&weights, "model.layers.0.self_attn.q_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let q_norm = tensor(&weights, "model.layers.0.self_attn.q_norm.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let k_proj = tensor(&weights, "model.layers.0.self_attn.k_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let k_norm = tensor(&weights, "model.layers.0.self_attn.k_norm.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let v_proj = tensor(&weights, "model.layers.0.self_attn.v_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let o_proj = tensor(&weights, "model.layers.0.self_attn.o_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);

    let hidden_size = q_proj.size()[1];
    let head_dim = config.head_dim;
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
            q_norm: &q_norm,
            k_proj: &k_proj,
            k_norm: &k_norm,
            v_proj: &v_proj,
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
    let full_output = qwen3_attention(
        &input, &q_proj, &q_norm, &k_proj, &k_norm, &v_proj, &o_proj, &config,
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
pub(crate) fn qwen_tp_attention_shard_contribution(
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
    let head_dim = config.head_dim;
    let q_shard = input
        .linear::<&Tensor>(
            &weights.q_proj.narrow(0, q_output_start, q_output_size),
            None,
        )
        .reshape([batch_size, seq_len, q_heads_per_rank, head_dim])
        .transpose(1, 2);
    let k_shard = input
        .linear::<&Tensor>(
            &weights.k_proj.narrow(0, kv_output_start, kv_output_size),
            None,
        )
        .reshape([batch_size, seq_len, kv_heads_per_rank, head_dim])
        .transpose(1, 2);
    let v_shard = input
        .linear::<&Tensor>(
            &weights.v_proj.narrow(0, kv_output_start, kv_output_size),
            None,
        )
        .reshape([batch_size, seq_len, kv_heads_per_rank, head_dim])
        .transpose(1, 2);
    let q_shard = rms_norm(&q_shard, weights.q_norm, config.rms_norm_eps);
    let k_shard = rms_norm(&k_shard, weights.k_norm, config.rms_norm_eps);
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

pub fn qwen_tp_mlp_rank(model_path: &Path, output_dir: PathBuf) -> Result<()> {
    let model_path = resolve_qwen3_model_path(model_path)?;
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("Qwen TP MLP rank expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    let device = Device::Cuda(local_rank);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let weights = read_safetensors_dir(&model_path)?;
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
        bail!("Qwen TP MLP requires intermediate size divisible by WORLD_SIZE");
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
            let full_output = qwen3_mlp(&input, &gate_proj, &up_proj, &down_proj);
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

pub fn qwen_tp_mlp_nccl_rank(model_path: &Path, output_dir: PathBuf) -> Result<()> {
    let model_path = resolve_qwen3_model_path(model_path)?;
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("Qwen TP MLP NCCL rank expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    let device = Device::Cuda(local_rank);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let weights = read_safetensors_dir(&model_path)?;
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
        bail!("Qwen TP MLP NCCL requires intermediate size divisible by WORLD_SIZE");
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
    let full_output = qwen3_mlp(&input, &gate_proj, &up_proj, &down_proj);
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

pub fn qwen_session_dp_rank(
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
        bail!("Qwen session DP expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    if steps == 0 {
        bail!("Qwen session DP requires at least one step");
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        bail!("Qwen session DP requires a positive finite learning rate");
    }

    let model_path = resolve_qwen3_model_path(model_path)?;
    let device = Device::Cuda(local_rank);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let output_dir = qwen_dp_artifact_dir(&output_dir)?;
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let config = read_qwen3_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_dir(&model_path)?;
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
    let dp_streaming_index_cache = runtime_data.and_then(|data| match data.kind {
        RuntimeDataKind::InstructionJsonl | RuntimeDataKind::InstructionArrow => Some(
            data.index_cache
                .as_ref()
                .map(|path| qwen_sft_rank_index_cache_path(path, rank))
                .unwrap_or_else(|| {
                    qwen_sft_streaming_index_cache_path(
                        &output_dir.join(format!("rank-{rank}-cache")),
                        "qwen-session-dp",
                    )
                }),
        ),
        _ => data.index_cache.clone(),
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
                read_safetensors_dir(&model_path)?,
                batch_plan.global_initial_input_ids.shallow_clone(),
                dtype.kind(),
                &manifest.to_delta_manifest()?,
                Some(device),
            )?
        } else {
            QwenTrainableSession::from_trainable_layers_on_device(
                config,
                read_safetensors_dir(&model_path)?,
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
    let train_started = Instant::now();
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
    let train_elapsed_secs = train_started.elapsed().as_secs_f64().max(1e-9);
    let trained_samples = batch_plan.local_batch_size * steps;
    let trained_tokens = trained_samples * batch_plan.sequence_tokens;
    let samples_per_second = trained_samples as f64 / train_elapsed_secs;
    let tokens_per_second = trained_tokens as f64 / train_elapsed_secs;
    let global_post_update_loss = *global_step_losses
        .last()
        .ok_or_else(|| anyhow!("missing Qwen session DP post-update loss"))?;
    let global_loss_improved = global_post_update_loss < global_loss;
    let require_loss_improvement = batch_plan.data_kind.as_deref() != Some("instruction_arrow");
    if require_loss_improvement && !global_loss_improved {
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
        read_safetensors_dir(&model_path)?,
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
    let sharded_global_error_output = output_dir.join("qwen-session-dp-sharded-global.error");
    if rank == 0 {
        if let Err(error) = write_qwen_session_dp_global_sharded_manifest(
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
        ) {
            fs::write(&sharded_global_error_output, format!("{error:#}\n")).with_context(|| {
                format!("failed to write {}", sharded_global_error_output.display())
            })?;
        }
    }
    wait_for_rank_barrier_or_error(
        &output_dir.join("qwen-session-dp-sharded-global-manifest-written"),
        rank,
        world_size,
        Duration::from_secs(300),
        &sharded_global_error_output,
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
        read_safetensors_dir(&model_path)?,
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
        read_safetensors_dir(&model_path)?,
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
        data_kind: batch_plan.data_kind.clone(),
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
        tokens_per_second,
        samples_per_second,
        memory_rss_mb: rustrain_train::metrics::memory_rss_mb(),
        gpu_memory_allocated_mb: rustrain_train::metrics::gpu_memory_allocated_mb(),
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

pub(crate) fn write_qwen_session_dp_rank_sharded_manifest(
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

    let metadata = serde_json::json!({
        "format": "rustrain.qwen_session_dp_shard.v1",
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
        rank_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata)? + "\n",
    )
    .with_context(|| format!("failed to write sharded metadata for rank {rank}"))?;

    Ok(rank_manifest_output)
}

pub(crate) fn write_qwen_session_dp_global_sharded_manifest(
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

pub(crate) fn qwen_sharded_rank_to_delta_manifest(
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

pub(crate) fn qwen_session_tp_rank(
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
        bail!("Qwen session TP expects WORLD_SIZE=2");
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

    let runtime_config = read_qwen3_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_dir(&model_path)?;
    let q_proj_full = tensor(&weights, "model.layers.0.self_attn.q_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let q_norm_full = tensor(&weights, "model.layers.0.self_attn.q_norm.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let k_proj_full = tensor(&weights, "model.layers.0.self_attn.k_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let k_norm_full = tensor(&weights, "model.layers.0.self_attn.k_norm.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let v_proj_full = tensor(&weights, "model.layers.0.self_attn.v_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let o_proj_full = tensor(&weights, "model.layers.0.self_attn.o_proj.weight")?
        .to_kind(Kind::Float)
        .to_device(device);
    let head_dim = runtime_config.head_dim;
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
                q_norm: &q_norm_full,
                k_proj: &train_k,
                k_norm: &k_norm_full,
                v_proj: &train_v,
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
            q_norm: &q_norm_full,
            k_proj: &train_k,
            k_norm: &k_norm_full,
            v_proj: &train_v,
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
        bail!("Qwen session TP attention train expected all shard gradients to be defined");
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
            "Qwen session TP attention train expected positive grad norms: q={attention_q_grad_norm}, k={attention_k_grad_norm}, v={attention_v_grad_norm}, o={attention_o_grad_norm}"
        );
    }
    let config_learning_rate = config.train.learning_rate as f64;
    if !config_learning_rate.is_finite() || config_learning_rate <= 0.0 {
        bail!("Qwen session TP train requires positive finite learning_rate");
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
                q_norm: &q_norm_full,
                k_proj: &candidate_k,
                k_norm: &k_norm_full,
                v_proj: &candidate_v,
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
            "Qwen session TP attention train did not reduce loss: initial={attention_train_initial_loss}, final={attention_train_final_loss}"
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
        bail!("Qwen session TP MLP requires intermediate size divisible by WORLD_SIZE");
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
    let mlp_full = qwen3_mlp(&mlp_input, &gate_proj_full, &up_proj_full, &down_proj_full);
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
        bail!("Qwen session TP MLP train expected all shard gradients to be defined");
    }
    let gate_grad_norm = gate_grad.norm().double_value(&[]);
    let up_grad_norm = up_grad.norm().double_value(&[]);
    let down_grad_norm = down_grad.norm().double_value(&[]);
    if gate_grad_norm <= 0.0 || up_grad_norm <= 0.0 || down_grad_norm <= 0.0 {
        bail!(
            "Qwen session TP MLP train expected positive grad norms: gate={gate_grad_norm}, up={up_grad_norm}, down={down_grad_norm}"
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
            "Qwen session TP MLP train did not reduce loss: initial={train_initial_loss}, final={train_final_loss}"
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
    let full_layer0 = qwen3_layer(
        &layer0_input,
        &QwenLayerWeights {
            input_norm: input_norm_full.shallow_clone(),
            q_proj: q_proj_full.shallow_clone(),
            q_norm: q_norm_full.shallow_clone(),
            k_proj: k_proj_full.shallow_clone(),
            k_norm: k_norm_full.shallow_clone(),
            v_proj: v_proj_full.shallow_clone(),
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
            q_norm: &q_norm_full,
            k_proj: &k_shard_base,
            k_norm: &k_norm_full,
            v_proj: &v_shard_base,
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
        &q_norm_full,
        &k_shard_base,
        &k_norm_full,
        &v_shard_base,
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
            "Qwen session TP layer0 train did not reduce loss: initial={}, final={}",
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
        &q_norm_full,
        &k_shard_base,
        &k_norm_full,
        &v_shard_base,
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
        &q_norm_full,
        &k_norm_full,
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
        &q_norm_full,
        &restored_shards.k,
        &k_norm_full,
        &restored_shards.v,
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
                &q_norm_full,
                &k_norm_full,
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

pub(crate) fn qwen_tp_mlp_shard_contribution(
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

pub(crate) fn qwen_session_tp_mlp_global_mse_loss_and_grad(
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
pub(crate) fn write_qwen_session_tp_focused_sharded_manifest(
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
            optimizer: "adamw_first_step_slots".to_string(),
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
pub(crate) fn qwen_session_tp_focused_sharded_restore(
    global_manifest_output: &Path,
    rank: usize,
    layer0_input: &Tensor,
    full_layer0: &Tensor,
    q_norm_shard: &Tensor,
    k_norm_shard: &Tensor,
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
            q_norm: q_norm_shard,
            k_proj: &restored_shards.k,
            k_norm: k_norm_shard,
            v_proj: &restored_shards.v,
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
pub(crate) fn qwen_session_tp_focused_external_resume(
    global_manifest_output: &Path,
    rank: usize,
    world_size: usize,
    layer0_input: &Tensor,
    full_layer0: &Tensor,
    q_norm_shard: &Tensor,
    k_norm_shard: &Tensor,
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
        q_norm_shard,
        k_norm_shard,
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
        q_norm_shard,
        &restored_shards.k,
        k_norm_shard,
        &restored_shards.v,
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
pub(crate) fn qwen_session_tp_layer0_sgd_update(
    input: &Tensor,
    input_norm_weight: &Tensor,
    post_attention_norm_weight: &Tensor,
    q_shard_weight: &Tensor,
    q_norm_shard: &Tensor,
    k_shard_weight: &Tensor,
    k_norm_shard: &Tensor,
    v_shard_weight: &Tensor,
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
                q_norm: q_norm_shard,
                k_proj: &train_k,
                k_norm: k_norm_shard,
                v_proj: &train_v,
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
            q_norm: q_norm_shard,
            k_proj: &train_k,
            k_norm: k_norm_shard,
            v_proj: &train_v,
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
            bail!("Qwen session TP layer0 train expected positive {name} grad norm");
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
            q_norm: q_norm_shard,
            k_proj: &candidate_k,
            k_norm: k_norm_shard,
            v_proj: &candidate_v,
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
pub(crate) fn qwen_session_tp_causal_lm_loss_and_output_grad(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    input_norm_weight: &Tensor,
    post_attention_norm_weight: &Tensor,
    q_shard_weight: &Tensor,
    q_norm_shard: &Tensor,
    k_shard_weight: &Tensor,
    k_norm_shard: &Tensor,
    v_shard_weight: &Tensor,
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
            q_norm: q_norm_shard,
            k_proj: k_shard_weight,
            k_norm: k_norm_shard,
            v_proj: v_shard_weight,
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
            q_norm: layer.q_norm.to_device(input_ids.device()),
            k_proj: layer.k_proj.to_device(input_ids.device()),
            k_norm: layer.k_norm.to_device(input_ids.device()),
            v_proj: layer.v_proj.to_device(input_ids.device()),
            o_proj: layer.o_proj.to_device(input_ids.device()),
            post_attention_norm: layer.post_attention_norm.to_device(input_ids.device()),
            gate_proj: layer.gate_proj.to_device(input_ids.device()),
            up_proj: layer.up_proj.to_device(input_ids.device()),
            down_proj: layer.down_proj.to_device(input_ids.device()),
        };
        hidden = qwen3_layer(&hidden, &layer, config);
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
        bail!("Qwen session TP focused causal LM train expected positive layer0 output grad");
    }
    Ok((loss_value, output_grad.detach()))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_session_tp_causal_lm_sgd_update(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    input_norm_weight: &Tensor,
    post_attention_norm_weight: &Tensor,
    q_shard_weight: &Tensor,
    q_norm_shard: &Tensor,
    k_shard_weight: &Tensor,
    k_norm_shard: &Tensor,
    v_shard_weight: &Tensor,
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
        qwen3_causal_lm_loss(input_ids, &device_weights, config)?.double_value(&[])
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
        q_norm_shard,
        &train_k,
        k_norm_shard,
        &train_v,
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
            q_norm: q_norm_shard,
            k_proj: &train_k,
            k_norm: k_norm_shard,
            v_proj: &train_v,
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
            bail!("Qwen session TP focused causal LM train expected positive {name} grad norm");
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
            q_norm_shard,
            &candidate_k,
            k_norm_shard,
            &candidate_v,
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
            "Qwen session TP focused causal LM train did not reduce loss: initial={initial_loss}, final={best_loss}"
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
pub(crate) fn qwen_session_tp_layer0_local_contributions(
    input: &Tensor,
    input_norm_weight: &Tensor,
    attention_weights: QwenTpAttentionShardWeights<'_>,
    config: &QwenRuntimeConfig,
    q_heads_per_rank: i64,
    kv_heads_per_rank: i64,
) -> Tensor {
    let hidden_size = input.size()[2];
    let head_dim = config.head_dim;
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

pub(crate) fn qwen_session_tp_layer0_mlp_contribution(
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
pub(crate) fn qwen_session_tp_layer0_global_mse_loss_and_grad(
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
pub(crate) fn qwen_session_tp_attention_global_mse_loss_and_grad(
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
