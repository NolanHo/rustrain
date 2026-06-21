use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array2, Axis, array, concatenate, s};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor};

use crate::nccl_smoke;

const DP_WEIGHT: [f64; 2] = [0.2, -0.1];
const DP_DATASET: [([f64; 2], f64); 4] = [
    ([1.0, 0.0], 0.7),
    ([0.0, 1.0], -0.3),
    ([1.0, 1.0], 0.4),
    ([2.0, -1.0], 1.2),
];

#[derive(Debug, Serialize, Deserialize)]
struct DpRankStats {
    rank: usize,
    world_size: usize,
    sample_count: usize,
    loss_sum: f64,
    grad_sum: [f64; 2],
}

#[derive(Debug, Serialize)]
struct DpSmokeSummary {
    output_dir: String,
    world_size: usize,
    global_loss: f64,
    single_rank_loss: f64,
    loss_delta: f64,
    averaged_grad: [f64; 2],
    single_rank_grad: [f64; 2],
    grad_max_delta: f64,
    rank_logs: Vec<String>,
    rank0_checkpoint: String,
}

#[derive(Debug, Serialize)]
struct TensorParallelSmokeSummary {
    world_size: usize,
    column_max_delta: f64,
    row_max_delta: f64,
    column_rank_shapes: Vec<Vec<usize>>,
    row_rank_shapes: Vec<Vec<usize>>,
}

#[derive(Debug, Serialize)]
struct ExpertParallelSmokeSummary {
    world_size: usize,
    expert_count: usize,
    output_max_delta: f64,
    expert_load: Vec<usize>,
    rank_token_counts: Vec<usize>,
}

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
}

#[derive(Debug, Serialize, Deserialize)]
struct ExpertParallelCheckpointManifest {
    format: String,
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

pub fn run_data_parallel_smoke(output_dir: &Path, world_size: usize) -> Result<()> {
    if world_size != 2 {
        bail!("M12 DP smoke currently expects world_size = 2");
    }

    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let current_exe = std::env::current_exe().context("failed to locate current executable")?;
    let mut children = Vec::with_capacity(world_size);
    for rank in 0..world_size {
        children.push(
            Command::new(&current_exe)
                .arg("parallel-dp-rank-smoke")
                .arg("--output-dir")
                .arg(output_dir)
                .arg("--rank")
                .arg(rank.to_string())
                .arg("--world-size")
                .arg(world_size.to_string())
                .spawn()
                .with_context(|| format!("failed to spawn DP rank {rank}"))?,
        );
    }

    for (rank, mut child) in children.into_iter().enumerate() {
        let status = child
            .wait()
            .with_context(|| format!("failed to wait for DP rank {rank}"))?;
        if !status.success() {
            bail!("DP rank {rank} exited with status {status}");
        }
    }

    let mut rank_stats = Vec::with_capacity(world_size);
    let mut rank_logs = Vec::with_capacity(world_size);
    for rank in 0..world_size {
        let stats_path = output_dir.join(format!("rank-{rank}.json"));
        let log_path = output_dir.join(format!("rank-{rank}.log"));
        if !log_path.exists() {
            bail!("missing rank-local log {}", log_path.display());
        }
        let stats: DpRankStats = serde_json::from_str(
            &fs::read_to_string(&stats_path)
                .with_context(|| format!("failed to read {}", stats_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", stats_path.display()))?;
        rank_logs.push(log_path.display().to_string());
        rank_stats.push(stats);
    }

    let checkpoint_path = output_dir.join("rank0-checkpoint.json");
    if !checkpoint_path.exists() {
        bail!("rank0 checkpoint was not written");
    }
    for rank in 1..world_size {
        let unexpected = output_dir.join(format!("rank{rank}-checkpoint.json"));
        if unexpected.exists() {
            bail!(
                "non-rank0 checkpoint unexpectedly exists: {}",
                unexpected.display()
            );
        }
    }

    let total_samples = rank_stats
        .iter()
        .map(|stats| stats.sample_count)
        .sum::<usize>();
    let loss_sum = rank_stats.iter().map(|stats| stats.loss_sum).sum::<f64>();
    let grad_sum = rank_stats.iter().fold([0.0_f64; 2], |mut acc, stats| {
        acc[0] += stats.grad_sum[0];
        acc[1] += stats.grad_sum[1];
        acc
    });
    let global_loss = loss_sum / total_samples as f64;
    let averaged_grad = [
        grad_sum[0] / total_samples as f64,
        grad_sum[1] / total_samples as f64,
    ];
    let single = compute_dp_stats(0, 1);
    let single_rank_loss = single.loss_sum / single.sample_count as f64;
    let single_rank_grad = [
        single.grad_sum[0] / single.sample_count as f64,
        single.grad_sum[1] / single.sample_count as f64,
    ];
    let loss_delta = (global_loss - single_rank_loss).abs();
    let grad_max_delta = averaged_grad
        .into_iter()
        .zip(single_rank_grad)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);

    if loss_delta > 1e-12 || grad_max_delta > 1e-12 {
        bail!(
            "DP=2 did not match single-rank global batch: loss_delta={loss_delta}, grad_max_delta={grad_max_delta}"
        );
    }

    let summary = DpSmokeSummary {
        output_dir: output_dir.display().to_string(),
        world_size,
        global_loss,
        single_rank_loss,
        loss_delta,
        averaged_grad,
        single_rank_grad,
        grad_max_delta,
        rank_logs,
        rank0_checkpoint: checkpoint_path.display().to_string(),
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn run_data_parallel_rank(output_dir: PathBuf, rank: usize, world_size: usize) -> Result<()> {
    if rank >= world_size {
        return Err(anyhow!(
            "rank {rank} must be smaller than world_size {world_size}"
        ));
    }
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let stats = compute_dp_stats(rank, world_size);
    let log_path = output_dir.join(format!("rank-{rank}.log"));
    fs::write(
        &log_path,
        format!(
            "rank={rank}\nworld_size={world_size}\nsample_count={}\nloss_sum={:.12}\ngrad_sum={:?}\n",
            stats.sample_count, stats.loss_sum, stats.grad_sum
        ),
    )
    .with_context(|| format!("failed to write {}", log_path.display()))?;

    let stats_path = output_dir.join(format!("rank-{rank}.json"));
    fs::write(&stats_path, serde_json::to_string_pretty(&stats)?)
        .with_context(|| format!("failed to write {}", stats_path.display()))?;

    if rank == 0 {
        let checkpoint_path = output_dir.join("rank0-checkpoint.json");
        fs::write(
            &checkpoint_path,
            serde_json::json!({
                "rank": rank,
                "world_size": world_size,
                "weight": DP_WEIGHT,
            })
            .to_string(),
        )
        .with_context(|| format!("failed to write {}", checkpoint_path.display()))?;
    }

    Ok(())
}

pub fn run_data_parallel_rank_from_args(
    output_dir: PathBuf,
    rank: Option<usize>,
    world_size: Option<usize>,
) -> Result<()> {
    let rank = rank
        .map(Ok)
        .unwrap_or_else(|| parse_launcher_usize_env("RANK"))?;
    let world_size = world_size
        .map(Ok)
        .unwrap_or_else(|| parse_launcher_usize_env("WORLD_SIZE"))?;
    run_data_parallel_rank(output_dir, rank, world_size)
}

pub fn run_tensor_parallel_smoke(world_size: usize) -> Result<()> {
    if world_size != 2 {
        bail!("M13 TP smoke currently expects world_size = 2");
    }

    let input = array![[1.0_f64, 2.0, -1.0], [0.5, -0.25, 1.5]];
    let column_weight = array![
        [0.2_f64, -0.1, 0.4, 0.3],
        [0.5, 0.7, -0.2, 0.1],
        [-0.3, 0.6, 0.8, -0.4],
    ];
    let full_column = input.dot(&column_weight);
    let column_rank0 = input.dot(&column_weight.slice(s![.., 0..2]).to_owned());
    let column_rank1 = input.dot(&column_weight.slice(s![.., 2..4]).to_owned());
    let gathered_column = concatenate(Axis(1), &[column_rank0.view(), column_rank1.view()])
        .context("failed to gather column-parallel outputs")?;
    let column_max_delta = max_abs_diff(&full_column, &gathered_column);

    let row_weight = array![[0.3_f64, -0.5], [0.1, 0.2], [-0.4, 0.6], [0.8, -0.7],];
    let row_input = array![[1.0_f64, 2.0, -1.0, 0.25], [0.5, -0.5, 1.5, 2.0]];
    let full_row = row_input.dot(&row_weight);
    let row_rank0 = row_input
        .slice(s![.., 0..2])
        .to_owned()
        .dot(&row_weight.slice(s![0..2, ..]).to_owned());
    let row_rank1 = row_input
        .slice(s![.., 2..4])
        .to_owned()
        .dot(&row_weight.slice(s![2..4, ..]).to_owned());
    let reduced_row = row_rank0.clone() + row_rank1.clone();
    let row_max_delta = max_abs_diff(&full_row, &reduced_row);

    if column_max_delta > 1e-12 || row_max_delta > 1e-12 {
        bail!(
            "TP=2 parity failed: column_max_delta={column_max_delta}, row_max_delta={row_max_delta}"
        );
    }

    let summary = TensorParallelSmokeSummary {
        world_size,
        column_max_delta,
        row_max_delta,
        column_rank_shapes: vec![column_rank0.shape().to_vec(), column_rank1.shape().to_vec()],
        row_rank_shapes: vec![row_rank0.shape().to_vec(), row_rank1.shape().to_vec()],
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn run_expert_parallel_smoke(world_size: usize) -> Result<()> {
    if world_size != 2 {
        bail!("M17 EP smoke currently expects world_size = 2");
    }

    let tokens = ep_tokens();
    let router = ep_router();
    let expert_scales = ep_expert_scales();

    let assignments = route_top1(&tokens, &router);
    let reference = expert_outputs(&tokens, &assignments, &expert_scales);
    let mut expert_load = vec![0usize; expert_scales.len()];
    let mut rank_buckets = vec![Vec::<usize>::new(); world_size];
    for (token_index, expert_index) in assignments.iter().copied().enumerate() {
        expert_load[expert_index] += 1;
        rank_buckets[expert_index / (expert_scales.len() / world_size)].push(token_index);
    }

    let mut gathered = Array2::<f64>::zeros(tokens.dim());
    for bucket in &rank_buckets {
        for &token_index in bucket {
            let expert_index = assignments[token_index];
            for hidden_index in 0..tokens.ncols() {
                gathered[[token_index, hidden_index]] =
                    tokens[[token_index, hidden_index]] * expert_scales[expert_index][hidden_index];
            }
        }
    }

    let output_max_delta = max_abs_diff(&gathered, &reference);
    if output_max_delta > 1e-12 {
        bail!("EP=2 all-to-all parity failed: output_max_delta={output_max_delta}");
    }

    let summary = ExpertParallelSmokeSummary {
        world_size,
        expert_count: expert_scales.len(),
        output_max_delta,
        expert_load,
        rank_token_counts: rank_buckets.iter().map(Vec::len).collect(),
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn run_expert_parallel_rank_smoke(output_dir: PathBuf) -> Result<()> {
    let rank = parse_launcher_usize_env("RANK")?;
    let local_rank = parse_launcher_usize_env("LOCAL_RANK")?;
    let world_size = parse_launcher_usize_env("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("EP rank-local smoke currently expects WORLD_SIZE=2");
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

pub fn run_expert_parallel_nccl_rank_smoke(output_dir: PathBuf) -> Result<()> {
    let rank = parse_launcher_usize_env("RANK")?;
    let local_rank = parse_launcher_usize_env("LOCAL_RANK")?;
    let world_size = parse_launcher_usize_env("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("EP NCCL rank smoke currently expects WORLD_SIZE=2");
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
        bail!("EP NCCL smoke local expert scale gradient is missing or zero on rank {rank}");
    }

    let final_loss = first_step
        .post_loss
        .ok_or_else(|| anyhow!("EP NCCL first step did not compute post-update loss"))?;
    let train_loss_improved = final_loss < first_step.pre_loss;
    if !train_loss_improved {
        bail!(
            "EP NCCL train smoke did not lower loss on rank {rank}: initial={}, final={final_loss}",
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

pub fn run_expert_parallel_sparse_rank_smoke(output_dir: PathBuf) -> Result<()> {
    let rank = parse_launcher_usize_env("RANK")?;
    let local_rank = parse_launcher_usize_env("LOCAL_RANK")?;
    let world_size = parse_launcher_usize_env("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("EP sparse rank smoke currently expects WORLD_SIZE=2");
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
    let source_token_indices = ep_source_token_indices(rank, world_size, assignments.len());
    let global_expert_load = ep_global_expert_load(&assignments);
    let load_balance_loss = ep_load_balance_loss(&global_expert_load);
    let experts_per_rank = ep_expert_count() / world_size;
    let owned_expert_start = rank * experts_per_rank;
    let owned_expert_end = owned_expert_start + experts_per_rank;

    let dispatch_send_plan =
        ep_sparse_dispatch_send_plan(rank, world_size, &assignments, assignments.len())?;
    let dispatch_recv_plan = ep_sparse_dispatch_recv_plan(rank, world_size, &assignments)?;
    let dispatch_sends = dispatch_send_plan
        .iter()
        .map(|plan| {
            (
                plan.peer,
                ep_sparse_pack_token_rows(&tokens, &plan.token_indices),
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
        &output_dir.join("ep-sparse-dispatch"),
        &dispatch_sends,
        &dispatch_recvs,
    )?;

    let local_scales = ep_owned_expert_scales_tensor(owned_expert_start, owned_expert_end, device);
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
            &assignments,
            owned_expert_start,
            owned_expert_end,
            &local_scales,
        )?;
        combine_sends.push((peer, output));
    }
    owned_token_indices.sort_unstable();

    let combine_recv_plan = ep_sparse_combine_recv_plan(rank, world_size, &assignments)?;
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
        &output_dir.join("ep-sparse-combine"),
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
    let reference = ep_reference_output_tensor(&tokens, &assignments)?;
    let reference_rows = ep_sparse_pack_token_rows(&reference, &source_token_indices);
    let sparse_diff = tensor_diff_stats(&assembled_output, &reference_rows)?;
    if sparse_diff.0 > 1e-6 {
        bail!(
            "EP sparse dispatch/combine mismatch: rank={rank}, max_abs={}, mean_abs={}",
            sparse_diff.0,
            sparse_diff.1
        );
    }

    let summary = ExpertParallelSparseRankSummary {
        rank,
        world_size,
        local_rank,
        source_token_indices,
        owned_expert_start,
        owned_expert_end,
        owned_token_indices,
        global_expert_load,
        load_balance_loss,
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
        assembled_output_shape: assembled_output.size(),
        reference_output_shape: reference_rows.size(),
        sparse_output_max_abs: sparse_diff.0,
        sparse_output_mean_abs: sparse_diff.1,
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

fn compute_dp_stats(rank: usize, world_size: usize) -> DpRankStats {
    let mut sample_count = 0;
    let mut loss_sum = 0.0;
    let mut grad_sum = [0.0_f64; 2];

    for (sample_index, (features, target)) in DP_DATASET.iter().enumerate() {
        if sample_index % world_size != rank {
            continue;
        }
        let prediction = DP_WEIGHT[0] * features[0] + DP_WEIGHT[1] * features[1];
        let error = prediction - target;
        loss_sum += 0.5 * error * error;
        grad_sum[0] += error * features[0];
        grad_sum[1] += error * features[1];
        sample_count += 1;
    }

    DpRankStats {
        rank,
        world_size,
        sample_count,
        loss_sum,
        grad_sum,
    }
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

fn ep_reference_output_tensor(tokens: &Tensor, assignments: &[usize]) -> Result<Tensor> {
    ep_local_output_tensor(
        tokens,
        assignments,
        0,
        ep_expert_count(),
        &ep_all_expert_scales_tensor(tokens.device()),
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

fn max_abs_diff(actual: &Array2<f64>, expected: &Array2<f64>) -> f64 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max)
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

fn expert_outputs(
    tokens: &Array2<f64>,
    assignments: &[usize],
    expert_scales: &[[f64; 3]],
) -> Array2<f64> {
    let mut output = Array2::<f64>::zeros(tokens.dim());
    for (token_index, expert_index) in assignments.iter().copied().enumerate() {
        for hidden_index in 0..tokens.ncols() {
            output[[token_index, hidden_index]] =
                tokens[[token_index, hidden_index]] * expert_scales[expert_index][hidden_index];
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn dp_partition_matches_single_rank_global_batch() {
        let rank0 = compute_dp_stats(0, 2);
        let rank1 = compute_dp_stats(1, 2);
        let single = compute_dp_stats(0, 1);

        assert_eq!(rank0.sample_count + rank1.sample_count, single.sample_count);
        assert_eq!(rank0.loss_sum + rank1.loss_sum, single.loss_sum);
        assert_eq!(
            [
                rank0.grad_sum[0] + rank1.grad_sum[0],
                rank0.grad_sum[1] + rank1.grad_sum[1]
            ],
            single.grad_sum
        );
    }

    #[test]
    fn dp_rank_writes_rank_local_log_and_rank0_checkpoint() {
        let temp = TempDir::new().expect("temp dir should be created");

        run_data_parallel_rank(temp.path().to_path_buf(), 0, 2).expect("rank0 should run");
        run_data_parallel_rank(temp.path().to_path_buf(), 1, 2).expect("rank1 should run");

        assert!(temp.path().join("rank-0.log").exists());
        assert!(temp.path().join("rank-1.log").exists());
        assert!(temp.path().join("rank0-checkpoint.json").exists());
        assert!(!temp.path().join("rank1-checkpoint.json").exists());
    }

    #[test]
    fn tensor_parallel_smoke_runs_tp2_parity() {
        run_tensor_parallel_smoke(2).expect("TP=2 parity should pass");
    }

    #[test]
    fn expert_parallel_smoke_runs_ep2_all_to_all_parity() {
        run_expert_parallel_smoke(2).expect("EP=2 parity should pass");
    }
}
