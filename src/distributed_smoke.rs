use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array2, Axis, array, concatenate, s};
use serde::{Deserialize, Serialize};

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

fn array2_to_rows(array: &Array2<f64>) -> Vec<Vec<f64>> {
    array.rows().into_iter().map(|row| row.to_vec()).collect()
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
