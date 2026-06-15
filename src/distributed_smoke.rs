use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, anyhow, bail};
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
}
