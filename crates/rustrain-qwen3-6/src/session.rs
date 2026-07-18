//! Qwen3.6 training session — single-GPU LoRA SFT + EP4 distributed training.

use std::collections::{BTreeMap, HashSet};
use std::env;

use anyhow::{Context, Result, anyhow, bail};
use tch::{Kind, Tensor};
use tracing::info;

use crate::checkpoint;
use crate::config::{
    LayerType, Qwen36RuntimeConfig, read_qwen36_runtime_config, resolve_qwen36_model_path,
};
use crate::lora::{
    Qwen36AdapterArtifact, Qwen36LoraConfig, Qwen36LoraTargetModule, native_lora_slots,
    validate_lora_targets,
};
use crate::pipeline::{PipelineStageLayout, stage_lora_slots, stage_needed_weights};
use crate::sft::SftDataset;
use rustrain_checkpoint::safetensors::read_safetensors_dir_filtered;
use rustrain_core::runtime::{Config, RunPaths};
use rustrain_parallel::topology::{DEFAULT_RANK_ORDER, ParallelTopology};

// ──────────────────────────────────────────────────────────────────────
// EP Shard
// ──────────────────────────────────────────────────────────────────────

pub struct EpShard {
    pub rank: usize,
    pub world_size: usize,
    pub experts_per_rank: usize,
    pub expert_start: usize,
    pub local_expert_indices: Vec<usize>,
}

impl EpShard {
    pub fn new(rank: usize, world_size: usize, num_experts: usize) -> Self {
        assert!(
            num_experts % world_size == 0,
            "num_experts {num_experts} not divisible by world_size {world_size}"
        );
        let epr = num_experts / world_size;
        let start = rank * epr;
        Self {
            rank,
            world_size,
            experts_per_rank: epr,
            expert_start: start,
            local_expert_indices: (start..start + epr).collect(),
        }
    }

    pub fn owns_expert(&self, global_idx: usize) -> bool {
        global_idx >= self.expert_start && global_idx < self.expert_start + self.experts_per_rank
    }
}

// ──────────────────────────────────────────────────────────────────────
// Summary
// ──────────────────────────────────────────────────────────────────────

pub struct Qwen36LoraSftSummary {
    pub adapter_output: String,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub trainable_params: usize,
}

// ──────────────────────────────────────────────────────────────────────
// Weight name helpers
// ──────────────────────────────────────────────────────────────────────

fn parse_env_usize(key: &str) -> Result<usize> {
    env::var(key)
        .with_context(|| format!("{key} not set"))
        .and_then(|v| {
            v.parse::<usize>()
                .with_context(|| format!("invalid {key}: {v}"))
        })
}

fn validate_lora_rank_for_tp(
    lora_rank: i64,
    tp_size: usize,
    layouts: &[checkpoint::LoraTpShardLayout],
) -> Result<()> {
    if tp_size > 1
        && layouts
            .iter()
            .any(|layout| *layout == checkpoint::LoraTpShardLayout::LatentRank)
        && lora_rank % tp_size as i64 != 0
    {
        bail!("latent-rank LoRA rank {lora_rank} must be divisible by TP_SIZE={tp_size}");
    }
    Ok(())
}

fn lora_config_from_config(config: &Config) -> Result<Qwen36LoraConfig> {
    let lora = config
        .lora
        .as_ref()
        .ok_or_else(|| anyhow!("[lora] section required"))?;
    let target_layers: Vec<usize> = lora.target_layers.clone();
    let target_modules: Vec<Qwen36LoraTargetModule> = lora
        .target_modules
        .iter()
        .map(|m| Qwen36LoraTargetModule::parse(m))
        .collect::<Result<Vec<_>>>()?;
    Ok(Qwen36LoraConfig {
        rank: lora.rank,
        alpha: lora.alpha,
        target_layers,
        target_modules,
    })
}

fn training_source_coordinate(
    dp_rank: usize,
    dp_size: usize,
    ep_rank: usize,
    ep_size: usize,
    ep_source_sharded: bool,
) -> (usize, usize) {
    if ep_source_sharded {
        (dp_rank * ep_size + ep_rank, dp_size * ep_size)
    } else if dp_size > 1 {
        (dp_rank, dp_size)
    } else {
        (0, 1)
    }
}

fn pipeline_1f1b_schedule(
    pp_rank: usize,
    num_microbatches: usize,
) -> Result<Vec<(Option<i64>, Option<i64>)>> {
    if pp_rank >= 2 {
        bail!("PP2 schedule received invalid pipeline rank {pp_rank}");
    }
    if num_microbatches == 0 {
        bail!("pipeline schedule requires at least one microbatch");
    }
    let mut schedule = Vec::with_capacity(num_microbatches + usize::from(pp_rank == 0));
    if pp_rank == 0 {
        for microbatch in 0..num_microbatches {
            let microbatch = i64::try_from(microbatch).context("microbatch id exceeds i64")?;
            schedule.push((Some(microbatch), (microbatch > 0).then_some(microbatch - 1)));
        }
        schedule.push((
            None,
            Some(i64::try_from(num_microbatches - 1).context("microbatch id exceeds i64")?),
        ));
    } else {
        for microbatch in 0..num_microbatches {
            let microbatch = i64::try_from(microbatch).context("microbatch id exceeds i64")?;
            schedule.push((Some(microbatch), Some(microbatch)));
        }
    }
    Ok(schedule)
}

// ──────────────────────────────────────────────────────────────────────
// Entry point: single-GPU LoRA SFT
// ──────────────────────────────────────────────────────────────────────

pub fn train_qwen3_6_lora_sft(
    config: &Config,
    run_paths: &RunPaths,
) -> Result<Qwen36LoraSftSummary> {
    train_impl(config, run_paths, None)
}

// ──────────────────────────────────────────────────────────────────────
// Entry point: EP4 distributed LoRA SFT
// ──────────────────────────────────────────────────────────────────────

pub fn train_qwen3_6_lora_sft_ep(
    config: &Config,
    run_paths: &RunPaths,
) -> Result<Qwen36LoraSftSummary> {
    let rank = parse_env_usize("RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    let model_path = config
        .model
        .model_path
        .as_ref()
        .ok_or_else(|| anyhow!("model.model_path required"))?;
    let model_path = std::fs::canonicalize(resolve_qwen36_model_path(model_path)?)
        .with_context(|| format!("canonicalize Qwen model path {}", model_path.display()))?;
    let runtime_config = read_qwen36_runtime_config(&model_path)?;
    let ep_shard = if runtime_config.is_moe {
        if config.parallel.context_parallel_size != 1 {
            bail!(
                "native MoE Qwen LoRA does not yet support context parallelism: TP={} PP={} DP={} EP={} CP={}",
                config.parallel.tensor_model_parallel_size,
                config.parallel.pipeline_model_parallel_size,
                config.parallel.data_parallel_size,
                config.parallel.expert_model_parallel_size,
                config.parallel.context_parallel_size,
            );
        }
        let rank_order = std::env::var("RUSTRAIN_PARALLEL_ORDER")
            .or_else(|_| std::env::var("PARALLEL_ORDER"))
            .unwrap_or_else(|_| DEFAULT_RANK_ORDER.to_string());
        let topology = ParallelTopology::with_order(
            config.parallel.tensor_model_parallel_size,
            config.parallel.pipeline_model_parallel_size,
            config.parallel.data_parallel_size,
            config.parallel.expert_model_parallel_size,
            1,
            &rank_order,
        )?;
        topology.validate_world_size(world_size)?;
        Some(EpShard::new(
            topology.expert_rank(rank)?,
            topology.expert_model_parallel_size(),
            runtime_config.num_experts,
        ))
    } else {
        None
    };
    train_impl(config, run_paths, ep_shard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tp_peers_share_expert_and_source_shards() {
        let topology = ParallelTopology::new(2, 1, 1, 2, 1).unwrap();
        let expected_ep_ranks = [0, 0, 1, 1];
        for (global_rank, expected_ep_rank) in expected_ep_ranks.into_iter().enumerate() {
            let ep_rank = topology.expert_rank(global_rank).unwrap();
            assert_eq!(ep_rank, expected_ep_rank);
            let shard = EpShard::new(ep_rank, 2, 8);
            assert_eq!(shard.expert_start, expected_ep_rank * 4);
            let (source_rank, source_count) = training_source_coordinate(0, 1, ep_rank, 2, true);
            let data_start = 3 * 2 * source_count + source_rank * 2;
            assert_eq!(data_start, 12 + expected_ep_rank * 2);
        }
    }

    #[test]
    fn tp_ep_dp_uses_every_expert_and_data_source_coordinate() {
        let expected = [(0, 4), (1, 4), (2, 4), (3, 4)];
        let observed = [(0, 0), (0, 1), (1, 0), (1, 1)]
            .map(|(dp_rank, ep_rank)| training_source_coordinate(dp_rank, 2, ep_rank, 2, true));
        assert_eq!(observed, expected);
        assert_eq!(training_source_coordinate(1, 2, 1, 2, false), (1, 2));
    }

    #[test]
    fn pp2_schedule_matches_non_interleaved_1f1b() {
        assert_eq!(
            pipeline_1f1b_schedule(0, 4).unwrap(),
            vec![
                (Some(0), None),
                (Some(1), Some(0)),
                (Some(2), Some(1)),
                (Some(3), Some(2)),
                (None, Some(3)),
            ]
        );
        assert_eq!(
            pipeline_1f1b_schedule(1, 4).unwrap(),
            vec![
                (Some(0), Some(0)),
                (Some(1), Some(1)),
                (Some(2), Some(2)),
                (Some(3), Some(3)),
            ]
        );
    }

    #[test]
    fn projection_sharded_lora_rank_does_not_require_tp_divisibility() {
        validate_lora_rank_for_tp(
            3,
            2,
            &[
                checkpoint::LoraTpShardLayout::ColumnParallel,
                checkpoint::LoraTpShardLayout::RoutedExpertFusedGateUp,
            ],
        )
        .unwrap();
        assert!(
            validate_lora_rank_for_tp(3, 2, &[checkpoint::LoraTpShardLayout::LatentRank],).is_err()
        );
    }
}

// ──────────────────────────────────────────────────────────────────────
// Core training implementation
// ──────────────────────────────────────────────────────────────────────

fn train_impl(
    config: &Config,
    run_paths: &RunPaths,
    ep_shard: Option<EpShard>,
) -> Result<Qwen36LoraSftSummary> {
    let model_path = config
        .model
        .model_path
        .as_ref()
        .ok_or_else(|| anyhow!("model.model_path required"))?;
    let model_path = std::fs::canonicalize(resolve_qwen36_model_path(model_path)?)
        .with_context(|| format!("canonicalize Qwen model path {}", model_path.display()))?;
    let runtime_config = read_qwen36_runtime_config(&model_path)?;
    let lora_config = lora_config_from_config(config)?;
    validate_lora_targets(&runtime_config, &lora_config)?;
    let device = match config.train.device {
        rustrain_core::runtime::Device::Cuda => {
            // EP mode: use LOCAL_RANK to select the correct GPU
            let local_rank = std::env::var("LOCAL_RANK")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            tch::Device::Cuda(local_rank)
        }
        rustrain_core::runtime::Device::Cpu => tch::Device::Cpu,
    };
    let compute_kind = match config.train.dtype {
        rustrain_core::runtime::DType::Fp16 => Kind::Half,
        rustrain_core::runtime::DType::Bf16 => Kind::BFloat16,
        rustrain_core::runtime::DType::Fp32 => Kind::Float,
    };
    if compute_kind != Kind::BFloat16 {
        bail!(
            "native Qwen3.5/3.6 LoRA currently supports bf16 only; {:?} would not be updated by the fused Adam kernel",
            config.train.dtype
        );
    }

    let shard_ref = ep_shard.as_ref();
    let is_ep = shard_ref.is_some();
    let env_enabled = |name: &str| {
        std::env::var(name)
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false)
    };
    let ep_a2a = env_enabled("QWEN36_EP_A2A");
    let ep_a2a_sharded = env_enabled("QWEN36_EP_A2A_SHARDED");
    if ep_a2a_sharded && !is_ep {
        bail!("QWEN36_EP_A2A_SHARDED=1 requires expert-parallel training");
    }
    if ep_a2a_sharded && !ep_a2a {
        bail!("QWEN36_EP_A2A_SHARDED=1 requires QWEN36_EP_A2A=1");
    }
    let env_world_size = std::env::var("WORLD_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    let env_rank = std::env::var("RANK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let world_size = env_world_size;
    let rank = env_rank;
    let distributed_generation = if world_size > 1 {
        let attempt = std::env::var("RUSTRAIN_ATTEMPT_ID")
            .context("distributed CLI training requires launcher-provided RUSTRAIN_ATTEMPT_ID")?;
        Some(format!(
            "{attempt}.adapter-final.{ordinal:020}",
            ordinal = 0
        ))
    } else {
        None
    };
    let tp_size = config.parallel.tensor_model_parallel_size;
    let pp_size = config.parallel.pipeline_model_parallel_size;
    let cp_size = config.parallel.context_parallel_size;
    let env_tp_size = std::env::var("TP_SIZE")
        .or_else(|_| std::env::var("RUSTRAIN_TP_SIZE"))
        .or_else(|_| std::env::var("TENSOR_MODEL_PARALLEL_SIZE"))
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    if env_tp_size > 1 && env_tp_size != tp_size {
        bail!(
            "TP_SIZE environment ({env_tp_size}) does not match config tensor_model_parallel_size ({tp_size})"
        );
    }
    let configured_dp_size = std::env::var("DP_SIZE")
        .or_else(|_| std::env::var("RUSTRAIN_DP_SIZE"))
        .or_else(|_| std::env::var("DATA_PARALLEL_SIZE"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(config.parallel.data_parallel_size);
    if is_ep && configured_dp_size != config.parallel.data_parallel_size {
        bail!(
            "DP_SIZE environment ({configured_dp_size}) does not match config data_parallel_size ({})",
            config.parallel.data_parallel_size
        );
    }
    let dense_model_parallel_size = tp_size
        .checked_mul(pp_size)
        .and_then(|size| size.checked_mul(cp_size))
        .context("Qwen model-parallel size overflow")?;
    let dp_size = if !is_ep && configured_dp_size == 1 && world_size > dense_model_parallel_size {
        world_size
            .checked_div(dense_model_parallel_size)
            .filter(|value| value * dense_model_parallel_size == world_size)
            .ok_or_else(|| {
                anyhow!(
                    "WORLD_SIZE={world_size} is not divisible by TPxPPxCP={dense_model_parallel_size}"
                )
            })?
    } else {
        configured_dp_size
    };
    let rank_order = std::env::var("RUSTRAIN_PARALLEL_ORDER")
        .or_else(|_| std::env::var("PARALLEL_ORDER"))
        .unwrap_or_else(|_| DEFAULT_RANK_ORDER.to_string());
    if cp_size != 1 {
        bail!("native Qwen LoRA does not yet support context parallelism (CP={cp_size})");
    }
    let parallel_topology = if let Some(shard) = shard_ref {
        Some(ParallelTopology::with_order(
            tp_size,
            pp_size,
            dp_size,
            shard.world_size,
            cp_size,
            &rank_order,
        )?)
    } else {
        if config.parallel.expert_model_parallel_size != 1 {
            bail!(
                "native dense Qwen LoRA requires EP=1: TP={} WORLD_SIZE={} PP={} DP={} EP={} CP={}",
                tp_size,
                world_size,
                pp_size,
                dp_size,
                config.parallel.expert_model_parallel_size,
                cp_size
            );
        }
        Some(ParallelTopology::with_order(
            tp_size,
            pp_size,
            dp_size,
            1,
            cp_size,
            &rank_order,
        )?)
    };
    if let Some(topology) = parallel_topology.as_ref() {
        topology.validate_world_size(world_size)?;
        topology.coordinates(rank)?;
    }
    let sequence_parallel = env_enabled("QWEN36_SEQUENCE_PARALLEL");
    if sequence_parallel
        && (runtime_config.is_moe
            || runtime_config.has_vision
            || tp_size != 2
            || pp_size != 1
            || cp_size != 1
            || dp_size != 1
            || runtime_config.mtp_num_hidden_layers > 0
            || runtime_config.router_aux_loss_coef != 0.0)
    {
        bail!(
            "QWEN36_SEQUENCE_PARALLEL=1 currently requires dense text-only fixed-LoRA TP2 with PP=CP=DP=1, MTP disabled, and router_aux_loss_coef=0"
        );
    }
    let is_data_parallel = dp_size > 1;
    unsafe {
        std::env::set_var("TP_SIZE", tp_size.to_string());
        std::env::set_var(
            "EP_SIZE",
            shard_ref
                .map(|shard| shard.world_size)
                .unwrap_or(1)
                .to_string(),
        );
        std::env::set_var("DP_SIZE", dp_size.to_string());
        std::env::set_var("CP_SIZE", cp_size.to_string());
        std::env::set_var("PP_SIZE", pp_size.to_string());
        std::env::set_var(
            "RUSTRAIN_DATA_PARALLEL",
            if is_data_parallel { "1" } else { "0" },
        );
    }
    let tp_rank = parallel_topology
        .as_ref()
        .map(|topology| topology.tensor_rank(rank))
        .transpose()?
        .unwrap_or(0);
    let dp_rank = parallel_topology
        .as_ref()
        .map(|topology| topology.data_rank(rank))
        .transpose()?
        .unwrap_or(0);
    let cp_rank = parallel_topology
        .as_ref()
        .map(|topology| topology.context_rank(rank))
        .transpose()?
        .unwrap_or(0);
    let pp_rank = parallel_topology
        .as_ref()
        .map(|topology| topology.pipeline_rank(rank))
        .transpose()?
        .unwrap_or(0);
    let ep_rank = parallel_topology
        .as_ref()
        .map(|topology| topology.expert_rank(rank))
        .transpose()?
        .unwrap_or(0);
    let ep_size = parallel_topology
        .as_ref()
        .map(ParallelTopology::expert_model_parallel_size)
        .unwrap_or(1);
    let stage = PipelineStageLayout::new(runtime_config.num_hidden_layers, pp_rank, pp_size)?;
    if pp_size > 1 && runtime_config.has_vision {
        bail!("pipeline-parallel Qwen training does not yet support the vision encoder");
    }
    if pp_size > 1 && runtime_config.mtp_num_hidden_layers > 0 {
        bail!("pipeline-parallel Qwen training does not yet support MTP layers");
    }
    let global_native_slots = native_lora_slots(&runtime_config, &lora_config);
    let native_slots = stage_lora_slots(&global_native_slots, &stage);
    let active_layouts = native_slots
        .iter()
        .filter(|slot| slot.active)
        .map(|slot| checkpoint::lora_tp_shard_layout(slot.module, &runtime_config))
        .collect::<Vec<_>>();
    validate_lora_rank_for_tp(lora_config.rank, tp_size, &active_layouts)?;
    let is_expert_parallel = ep_size > 1;
    unsafe {
        std::env::set_var("RUSTRAIN_TP_RANK", tp_rank.to_string());
        std::env::set_var("RUSTRAIN_CP_RANK", cp_rank.to_string());
        std::env::set_var("RUSTRAIN_EP_RANK", ep_rank.to_string());
        std::env::set_var("RUSTRAIN_DP_RANK", dp_rank.to_string());
        std::env::set_var("RUSTRAIN_PP_RANK", pp_rank.to_string());
    }
    if is_data_parallel || is_expert_parallel || tp_size > 1 {
        crate::kernel::CppTrainingContext::set_cuda_device(
            std::env::var("LOCAL_RANK")
                .ok()
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(0),
        );
    }

    // Full attention and GDN follow head-aligned ColumnParallel input
    // projections and RowParallel output projections. Dense MLP additionally
    // shards gate/up rows and down columns. Embeddings and the LM head shard
    // contiguous vocabulary rows and remain tied through the C++ context when
    // the model uses tied word embeddings.
    let base_tp_attention = tp_size > 1;
    let base_tp_mlp = tp_size > 1;
    let vocab_parallel = tp_size > 1;
    if vocab_parallel
        && (runtime_config.vocab_size <= 0 || runtime_config.vocab_size % tp_size as i64 != 0)
    {
        bail!(
            "vocab_size={} must be divisible by TP_SIZE={tp_size} for vocabulary parallelism",
            runtime_config.vocab_size
        );
    }
    if base_tp_attention {
        if runtime_config.mtp_num_hidden_layers > 0 {
            bail!("frozen base TP currently requires MTP to be disabled");
        }
        if runtime_config.num_attention_heads <= 0
            || runtime_config.num_attention_heads % tp_size as i64 != 0
            || runtime_config.num_key_value_heads <= 0
            || runtime_config.num_key_value_heads % tp_size as i64 != 0
            || runtime_config.num_attention_heads % runtime_config.num_key_value_heads != 0
        {
            bail!(
                "full-attention heads (q={}, kv={}) must preserve GQA groups and be divisible by TP_SIZE={tp_size}",
                runtime_config.num_attention_heads,
                runtime_config.num_key_value_heads
            );
        }
        let rotary_dim =
            (runtime_config.head_dim as f64 * runtime_config.partial_rotary_factor) as i64;
        if runtime_config.head_dim <= 0
            || rotary_dim < 0
            || rotary_dim > runtime_config.head_dim
            || rotary_dim % 2 != 0
        {
            bail!(
                "full-attention head_dim={} and partial_rotary_factor={} produce invalid rotary_dim={rotary_dim}",
                runtime_config.head_dim,
                runtime_config.partial_rotary_factor
            );
        }
        if runtime_config
            .layer_types
            .iter()
            .any(|layer| *layer == LayerType::LinearAttention)
        {
            if runtime_config.linear_num_key_heads <= 0
                || runtime_config.linear_num_value_heads <= 0
                || runtime_config.linear_num_value_heads % runtime_config.linear_num_key_heads != 0
                || runtime_config.linear_num_key_heads % tp_size as i64 != 0
                || runtime_config.linear_num_value_heads % tp_size as i64 != 0
                || runtime_config.linear_key_head_dim != 128
                || runtime_config.linear_value_head_dim != 128
                || runtime_config.linear_conv_kernel_dim <= 0
            {
                bail!(
                    "linear-attention TP requires k/v heads divisible by TP_SIZE with preserved value-head groups, 128-wide key/value heads, and a positive conv kernel: k_heads={}, v_heads={}, key_dim={}, value_dim={}, conv_kernel={}, tp={tp_size}",
                    runtime_config.linear_num_key_heads,
                    runtime_config.linear_num_value_heads,
                    runtime_config.linear_key_head_dim,
                    runtime_config.linear_value_head_dim,
                    runtime_config.linear_conv_kernel_dim,
                );
            }
        }
    }
    if base_tp_mlp {
        let intermediates = if runtime_config.is_moe {
            vec![
                ("routed expert", runtime_config.moe_intermediate_size),
                (
                    "shared expert",
                    runtime_config.shared_expert_intermediate_size,
                ),
            ]
        } else {
            vec![("dense", runtime_config.intermediate_size)]
        };
        for (kind, intermediate) in intermediates {
            if intermediate <= 0 || intermediate % tp_size as i64 != 0 {
                bail!(
                    "{kind} intermediate_size={intermediate} must be divisible by TP_SIZE={tp_size}"
                );
            }
        }
    }

    // Load only the frozen weights owned by this physical pipeline stage.
    let needed = stage_needed_weights(&runtime_config, &stage);

    // Stagger loading for EP to avoid OOM
    if is_ep {
        std::thread::sleep(std::time::Duration::from_secs(rank as u64 * 5));
    }

    info!(
        "loading {} weight tensors from {}",
        needed.len(),
        model_path.display()
    );
    let weights = read_safetensors_dir_filtered(&model_path, &needed)?;

    // Apply orthogonal EP and TP shards on CPU before moving weights to CUDA.
    let mut weights_gpu = BTreeMap::new();
    let num_experts = runtime_config.num_experts as i64;
    for (name, tensor) in &weights {
        let needs_expert_narrow = shard_ref.is_some()
            && (name.contains(".mlp.experts.gate_up_proj")
                || name.contains(".mlp.experts.down_proj"));
        let expert_shard = if needs_expert_narrow && tensor.size()[0] == num_experts {
            let shard = shard_ref.expect("expert shard checked above");
            Some(
                tensor
                    .narrow(0, shard.expert_start as i64, shard.experts_per_rank as i64)
                    .contiguous(),
            )
        } else {
            None
        };
        let expert_or_full = expert_shard.as_ref().unwrap_or(tensor);
        let moe_tp_shard = if runtime_config.is_moe && base_tp_mlp {
            crate::kernel::shard_moe_mlp_weight_for_tp(name, expert_or_full, tp_size, tp_rank)?
        } else {
            None
        };
        let vocab_shard = if expert_shard.is_none() && moe_tp_shard.is_none() && vocab_parallel {
            crate::kernel::shard_vocab_weight_for_tp(
                name,
                tensor,
                runtime_config.vocab_size,
                tp_size,
                tp_rank,
            )?
        } else {
            None
        };
        let local_shard = if moe_tp_shard.is_some() {
            moe_tp_shard
        } else if expert_shard.is_some() {
            expert_shard
        } else if vocab_shard.is_some() {
            vocab_shard
        } else if base_tp_attention {
            let full_attention_shard =
                crate::kernel::shard_full_attention_weight_for_tp(name, tensor, tp_size, tp_rank)?;
            let attention_shard = if full_attention_shard.is_some() {
                full_attention_shard
            } else {
                crate::kernel::shard_linear_attention_weight_for_tp(
                    name,
                    tensor,
                    tp_size,
                    tp_rank,
                    runtime_config.linear_num_key_heads,
                    runtime_config.linear_key_head_dim,
                    runtime_config.linear_num_value_heads,
                    runtime_config.linear_value_head_dim,
                )?
            };
            if attention_shard.is_some() || !base_tp_mlp {
                attention_shard
            } else {
                crate::kernel::shard_dense_mlp_weight_for_tp(name, tensor, tp_size, tp_rank)?
            }
        } else {
            None
        };
        let gpu_tensor = local_shard
            .as_ref()
            .unwrap_or(tensor)
            .to_device(device)
            .to_kind(compute_kind);
        weights_gpu.insert(name.clone(), gpu_tensor);
    }
    if let Some(shard) = shard_ref {
        info!(
            ep_size,
            ep_rank,
            experts_per_rank = shard.experts_per_rank,
            "frozen expert weights sharded across EP group"
        );
    }
    if base_tp_attention {
        info!(
            tp_size,
            tp_rank,
            base_tp_mlp,
            vocab_parallel,
            "frozen base TP enabled: attention/GDN, vocabulary, and dense/expert MLP shards"
        );
    }

    info!(
        "LoRA config: rank={}, alpha={}",
        lora_config.rank, lora_config.alpha
    );

    // Load SFT data
    let tokenizer_path = model_path.join("tokenizer.json");
    let data = if let Some(data_config) = &config.data {
        let path = &data_config.paths[0];
        SftDataset::from_jsonl(path, &tokenizer_path, config.model.seq_len)?
    } else {
        bail!("[data] section required for SFT training");
    };

    // Load MTP weights into weights_gpu if available (C++ uses them via set_mtp_weights)
    if runtime_config.mtp_num_hidden_layers > 0 {
        let mtp_names = crate::mtp::MtpWeights::weight_names(&runtime_config);
        let mtp_needed: HashSet<String> = mtp_names.into_iter().collect();
        let mtp_tensors = read_safetensors_dir_filtered(&model_path, &mtp_needed)?;
        let mut mtp_gpu = BTreeMap::new();
        for (name, tensor) in &mtp_tensors {
            mtp_gpu.insert(name.clone(), tensor.to_device(device).to_kind(compute_kind));
        }
        for (name, tensor) in mtp_gpu {
            weights_gpu.insert(name, tensor);
        }
    }

    let batch_size = config.train.micro_batch_size;
    let max_steps = config.train.max_steps as usize;
    let gradient_accumulation_steps = config.train.gradient_accumulation_steps;

    // ── C++ all-in-C++ training path (required) ──
    // LoRA A/B, Adam optimizer, forward, loss, backward all in C++.
    if !crate::kernel::kernels_available() {
        bail!(
            "C++ kernels (libqwen36_kernels.so) not found — required for training. Ensure the .so is in LD_LIBRARY_PATH."
        );
    }

    let expert_start = shard_ref.map(|s| s.expert_start).unwrap_or(0);
    let expert_count = shard_ref
        .map(|s| s.experts_per_rank)
        .unwrap_or(runtime_config.num_experts);
    let ctx = crate::kernel::CppTrainingContext::new_for_stage(
        &weights_gpu,
        &runtime_config,
        &stage,
        compute_kind,
        config.train.learning_rate as f64,
        config.train.adam_beta1 as f64,
        config.train.adam_beta2 as f64,
        config.train.adam_eps as f64,
        lora_config.alpha as f64 / lora_config.rank as f64, // lora scaling = alpha / rank
        lora_config.rank as i64,
        base_tp_attention,
        base_tp_mlp,
        vocab_parallel,
        is_data_parallel,
        is_expert_parallel,
        &lora_config.target_layers,
        &lora_config.target_modules,
        expert_start,
        expert_count,
    )?;

    if world_size > 1 {
        if let Some(topology) = parallel_topology.as_ref() {
            let tp_color = *topology
                .tensor_group(rank)?
                .iter()
                .min()
                .ok_or_else(|| anyhow!("empty TP process group"))?;
            let ep_color = *topology
                .expert_group(rank)?
                .iter()
                .min()
                .ok_or_else(|| anyhow!("empty EP process group"))?;
            let cp_color = *topology
                .context_group(rank)?
                .iter()
                .min()
                .ok_or_else(|| anyhow!("empty CP process group"))?;
            let dp_color = *topology
                .data_group(rank)?
                .iter()
                .min()
                .ok_or_else(|| anyhow!("empty DP process group"))?;
            let pp_color = *topology
                .pipeline_group(rank)?
                .iter()
                .min()
                .ok_or_else(|| anyhow!("empty PP process group"))?;
            ctx.init_parallel_nccl(
                rank, world_size, tp_rank, tp_size, tp_color, cp_rank, cp_size, cp_color, ep_rank,
                ep_size, ep_color, dp_rank, dp_size, dp_color, pp_rank, pp_size, pp_color,
            )?;
        } else {
            let ret = ctx.init_nccl();
            if ret != 0 {
                bail!("C++ NCCL init failed (code {})", ret);
            }
        }
        info!(
            rank,
            world_size, is_ep, "NCCL communicator created for Qwen LoRA parallel training"
        );
    }
    info!("C++ TrainingContext: {} LoRA params", ctx.lora_count());

    // Set MTP weights if available
    if runtime_config.mtp_num_hidden_layers > 0 {
        ctx.set_mtp_weights(&weights_gpu, &runtime_config, expert_start, expert_count)?;
        info!(
            "C++ TrainingContext: MTP weights set ({} layers)",
            runtime_config.mtp_num_hidden_layers
        );
    }

    // Enable gradient checkpointing if env var set
    if let Ok(gs) = std::env::var("QWEN36_CHECKPOINT_GROUP") {
        let group_size: i64 = gs.parse().unwrap_or(4);
        ctx.set_checkpoint(true, group_size);
        info!("C++ TrainingContext: gradient checkpointing ON (group_size={group_size})");
    }

    let native_count = ctx.lora_count() as usize;
    if native_slots.len() != native_count {
        bail!(
            "pipeline-local LoRA registry count {} does not match native slot count {native_count}",
            native_slots.len()
        );
    }
    let mut start_step = 0usize;
    let mut resumed_loss = 0.0_f64;
    if let Some(resume_path) = config.train.resume_from.as_ref() {
        let parallel = checkpoint::ParallelCheckpointManifest::from_topology(
            world_size,
            rank,
            parallel_topology
                .as_ref()
                .context("Qwen checkpoint restore is missing parallel topology")?,
        )?;
        let data = checkpoint::load_checkpoint_for_topology(resume_path, &parallel)?;
        let checkpoint_model_path =
            std::fs::canonicalize(&data.manifest.model_path).with_context(|| {
                format!(
                    "canonicalize checkpoint base model path {}",
                    data.manifest.model_path
                )
            })?;
        if checkpoint_model_path != model_path {
            bail!(
                "checkpoint base model {} does not match configured model {}",
                checkpoint_model_path.display(),
                model_path.display()
            );
        }
        if !data.dynamic_adapters.is_empty() {
            bail!("Qwen CLI fixed-LoRA training cannot resume dynamic adapter state");
        }
        if data.manifest.lora_rank != lora_config.rank
            || (data.manifest.lora_alpha - lora_config.alpha).abs() > 1e-12
        {
            bail!(
                "checkpoint LoRA rank/alpha {}/{} does not match config {}/{}",
                data.manifest.lora_rank,
                data.manifest.lora_alpha,
                lora_config.rank,
                lora_config.alpha
            );
        }
        let expected_identities = native_slots
            .iter()
            .filter(|slot| slot.active)
            .map(|slot| checkpoint::LoraSlotIdentity {
                index: slot.global_index,
                layer: slot.layer,
                module: slot.module.cpp_name().to_string(),
            })
            .collect::<Vec<_>>();
        if world_size > 1 {
            checkpoint::validate_fixed_tp_resume(
                &data.manifest,
                &active_layouts,
                &expected_identities,
            )?;
        }
        let active_slot_indices = native_slots
            .iter()
            .filter(|slot| slot.active)
            .map(|slot| slot.local_index)
            .collect::<Vec<_>>();
        let restore_slot_indices = checkpoint::fixed_restore_slot_indices(
            data.lora_a.len(),
            data.lora_b.len(),
            &active_slot_indices,
            native_count,
        )?;
        for ((a, b), &slot_index) in data
            .lora_a
            .iter()
            .zip(&data.lora_b)
            .zip(&restore_slot_indices)
        {
            ctx.set_lora_tensor(slot_index as i64, false, a)?;
            ctx.set_lora_tensor(slot_index as i64, true, b)?;
        }
        if data.adam_m.is_empty() != data.adam_v.is_empty() {
            bail!(
                "checkpoint fixed optimizer m/v count mismatch: {}/{}",
                data.adam_m.len(),
                data.adam_v.len()
            );
        }
        if data.adam_m.is_empty() && data.manifest.effective_fixed_optimizer_step() > 0 {
            bail!(
                "checkpoint fixed optimizer step {} has no Adam state",
                data.manifest.effective_fixed_optimizer_step()
            );
        }
        if !data.adam_m.is_empty() {
            let expected_saved_optimizer_count = restore_slot_indices.len().saturating_mul(2);
            if data.adam_m.len() != expected_saved_optimizer_count
                || data.adam_v.len() != expected_saved_optimizer_count
            {
                bail!(
                    "checkpoint fixed optimizer count mismatch: m={}, v={}, expected={expected_saved_optimizer_count}",
                    data.adam_m.len(),
                    data.adam_v.len()
                );
            }
            let (mut all_adam_m, mut all_adam_v) = ctx.export_optimizer_state()?;
            let expected_native_optimizer_count = native_count.saturating_mul(2);
            if all_adam_m.len() != expected_native_optimizer_count
                || all_adam_v.len() != expected_native_optimizer_count
            {
                bail!(
                    "native fixed optimizer count mismatch: m={}, v={}, expected={expected_native_optimizer_count}",
                    all_adam_m.len(),
                    all_adam_v.len()
                );
            }
            for (saved_slot, &native_slot) in restore_slot_indices.iter().enumerate() {
                let saved = saved_slot.saturating_mul(2);
                let native = native_slot.saturating_mul(2);
                all_adam_m[native] = data.adam_m[saved].shallow_clone();
                all_adam_m[native + 1] = data.adam_m[saved + 1].shallow_clone();
                all_adam_v[native] = data.adam_v[saved].shallow_clone();
                all_adam_v[native + 1] = data.adam_v[saved + 1].shallow_clone();
            }
            let imported = ctx.import_optimizer_state(&all_adam_m, &all_adam_v)?;
            if imported != expected_native_optimizer_count as i64 {
                bail!(
                    "native fixed optimizer import restored {imported} tensors, expected {expected_native_optimizer_count}"
                );
            }
        }
        let native_step = i64::try_from(data.manifest.effective_fixed_optimizer_step())
            .context("checkpoint step exceeds the native optimizer range")?;
        ctx.set_step_count(native_step)?;
        start_step = usize::try_from(data.manifest.step)
            .context("checkpoint step exceeds the CLI training range")?;
        if start_step > max_steps {
            bail!("checkpoint step {start_step} exceeds configured max_steps {max_steps}");
        }
        resumed_loss = data.manifest.loss;
        info!(
            step = start_step,
            loss = resumed_loss,
            path = %resume_path.display(),
            "Qwen fixed LoRA checkpoint restored"
        );
    }

    // Training loop
    let mut initial_loss = resumed_loss;
    let mut final_loss = resumed_loss;
    let mut observed_loss = false;

    for step in start_step..max_steps {
        let load_microbatch = |accumulation_index: usize| {
            let micro_step = step * gradient_accumulation_steps + accumulation_index;
            let (source_rank, source_count) =
                training_source_coordinate(dp_rank, dp_size, ep_rank, ep_size, ep_a2a_sharded);
            let data_start =
                (micro_step * batch_size * source_count + source_rank * batch_size) % data.len();
            let sft_batch = data.batch(data_start, batch_size);
            let (input_ids, target_mask) = sft_batch.to_tensors(device, compute_kind);

            // Build attention mask: 1 for real tokens, 0 for padding
            // target_mask > 0 means the token is either response (loss) or prompt (no loss but attend)
            // We need: 1 for all non-padding tokens, 0 for padding tokens
            // The SFT batch's target_mask is: 0=prompt, 1=response, but padding has mask=0 too.
            // We need attention_mask = (target_mask >= 0).to(float) but that's all 1s.
            // Actually: padding tokens have target_mask=0 AND are after EOS.
            // The SftBatch already has padding at the end with mask=0.
            // We need: attention_mask = 1 where token is NOT padding.
            // Since prompt tokens have mask=0 and response tokens have mask=1,
            // but padding also has mask=0, we can't distinguish prompt from padding using mask alone.
            // Solution: use the pad_token_id to build attention mask from input_ids.
            let pad_id = data.pad_token_id();
            let attention_mask = input_ids.ne(pad_id).to_kind(Kind::Float); // [batch, seq]

            (input_ids, target_mask, attention_mask)
        };

        let loss_value = if pp_size == 2 {
            let window_id = i64::try_from(step).context("pipeline window id exceeds i64")?;
            let num_microbatches = i64::try_from(gradient_accumulation_steps)
                .context("pipeline microbatch count exceeds i64")?;
            ctx.pipeline_begin_v1(window_id, num_microbatches)?;
            let schedule_result = (|| -> Result<f64> {
                let gradient_scale = 1.0 / gradient_accumulation_steps as f64;
                for (forward_mb, backward_mb) in
                    pipeline_1f1b_schedule(pp_rank, gradient_accumulation_steps)?
                {
                    if let Some(microbatch) = forward_mb {
                        let (input_ids, target_mask, attention_mask) =
                            load_microbatch(microbatch as usize);
                        ctx.pipeline_tick_v1(
                            window_id,
                            Some(microbatch),
                            backward_mb,
                            Some(&input_ids),
                            Some(&target_mask),
                            Some(&attention_mask),
                            gradient_scale,
                        )?;
                    } else {
                        ctx.pipeline_tick_v1(
                            window_id,
                            None,
                            backward_mb,
                            None,
                            None,
                            None,
                            gradient_scale,
                        )?;
                    }
                }
                Ok(ctx.pipeline_finish_v1()?.loss)
            })();
            match schedule_result {
                Ok(loss) => loss,
                Err(error) => {
                    let _ = ctx.pipeline_abort_v1();
                    return Err(error);
                }
            }
        } else {
            let mut loss = 0.0;
            for accumulation_index in 0..gradient_accumulation_steps {
                let (input_ids, target_mask, attention_mask) = load_microbatch(accumulation_index);

                loss += ctx.train_micro_step(
                    &input_ids,
                    &target_mask,
                    &attention_mask,
                    1.0 / gradient_accumulation_steps as f64,
                    accumulation_index + 1 == gradient_accumulation_steps,
                )? / gradient_accumulation_steps as f64;
            }
            loss
        };
        if !observed_loss {
            initial_loss = loss_value;
            observed_loss = true;
        }
        final_loss = loss_value;
        if step % 10 == 0 || step == max_steps - 1 {
            info!("step {step}/{max_steps} loss={loss_value:.6}");
        }
    }

    let (adapter_path, trainable_params) = if world_size > 1 {
        let generation = distributed_generation
            .as_deref()
            .expect("distributed save generation was validated before training");
        let parallel = checkpoint::ParallelCheckpointManifest::from_topology(
            world_size,
            rank,
            parallel_topology
                .as_ref()
                .context("distributed CLI export is missing parallel topology")?,
        )?;
        let (all_adam_m, all_adam_v) = ctx.export_optimizer_state()?;
        let expected_optimizer_count = native_count.saturating_mul(2);
        if all_adam_m.len() != expected_optimizer_count
            || all_adam_v.len() != expected_optimizer_count
        {
            bail!(
                "fixed optimizer state count mismatch: m={}, v={}, expected={expected_optimizer_count}",
                all_adam_m.len(),
                all_adam_v.len()
            );
        }
        let active_slots = native_slots
            .iter()
            .filter(|slot| slot.active)
            .collect::<Vec<_>>();
        let mut lora_a = Vec::with_capacity(active_slots.len());
        let mut lora_b = Vec::with_capacity(active_slots.len());
        let mut adam_m = Vec::with_capacity(active_slots.len().saturating_mul(2));
        let mut adam_v = Vec::with_capacity(active_slots.len().saturating_mul(2));
        let mut layouts = Vec::with_capacity(active_slots.len());
        let mut identities = Vec::with_capacity(active_slots.len());
        for slot in active_slots {
            lora_a.push(ctx.get_lora_a(slot.local_index as i64).with_context(|| {
                format!(
                    "native LoRA slot {} (global {}) is missing A",
                    slot.local_index, slot.global_index
                )
            })?);
            lora_b.push(ctx.get_lora_b(slot.local_index as i64).with_context(|| {
                format!(
                    "native LoRA slot {} (global {}) is missing B",
                    slot.local_index, slot.global_index
                )
            })?);
            let optimizer_index = slot.local_index.saturating_mul(2);
            adam_m.push(all_adam_m[optimizer_index].shallow_clone());
            adam_m.push(all_adam_m[optimizer_index + 1].shallow_clone());
            adam_v.push(all_adam_v[optimizer_index].shallow_clone());
            adam_v.push(all_adam_v[optimizer_index + 1].shallow_clone());
            layouts.push(checkpoint::lora_tp_shard_layout(
                slot.module,
                &runtime_config,
            ));
            identities.push(checkpoint::LoraSlotIdentity {
                index: slot.global_index,
                layer: slot.layer,
                module: slot.module.cpp_name().to_string(),
            });
        }
        let adapter_dir = run_paths.root.join("adapter");
        let step = u64::try_from(ctx.get_step_count())
            .context("native fixed adapter optimizer step is negative")?;
        let model_path_string = model_path
            .to_str()
            .context("Qwen model path is not UTF-8")?
            .to_string();
        let count = checkpoint::export_distributed_adapter_checkpoint(
            &adapter_dir,
            generation,
            &parallel,
            None,
            |staging| {
                checkpoint::save_checkpoint_with_dynamic_and_fixed_step_for_topology_generation(
                    staging,
                    step,
                    step,
                    final_loss,
                    &model_path_string,
                    lora_config.rank,
                    lora_config.alpha,
                    &lora_a,
                    &lora_b,
                    &adam_m,
                    &adam_v,
                    &[],
                    &layouts,
                    &identities,
                    &parallel,
                    Some(generation),
                )
            },
        )?;
        (adapter_dir.join("adapter_model.safetensors"), count)
    } else {
        // Export every positional native slot, then let the artifact mapper omit
        // inactive slots and assign stable projection-aware tensor names.
        let mut exported = Vec::with_capacity(native_count);
        for index in 0..ctx.lora_count() {
            let a = ctx
                .get_lora_a(index)
                .with_context(|| format!("native LoRA slot {index} is missing A"))?;
            let b = ctx
                .get_lora_b(index)
                .with_context(|| format!("native LoRA slot {index} is missing B"))?;
            exported.push((a, b));
        }
        let artifact = Qwen36AdapterArtifact::from_native_exports(
            &config.model.name,
            &config.model.architecture,
            Some(&model_path),
            &runtime_config,
            &lora_config,
            exported,
        )?;
        let count = artifact.tensors.len();
        let path = run_paths.root.join("adapter_model.safetensors");
        artifact.save(&run_paths.root)?;
        (path, count)
    };
    info!("saved adapter to {}", adapter_path.display());

    Ok(Qwen36LoraSftSummary {
        adapter_output: adapter_path.to_string_lossy().to_string(),
        initial_loss,
        final_loss,
        trainable_params,
    })
}
