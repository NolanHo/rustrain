use std::{fs, path::PathBuf};

use anyhow::{Context, Result, anyhow};
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

#[cfg(test)]
mod tests {
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
        let temp = tempfile::tempdir().expect("temp dir should be created");

        run_data_parallel_rank(temp.path().to_path_buf(), 0, 2).expect("rank0 should run");
        run_data_parallel_rank(temp.path().to_path_buf(), 1, 2).expect("rank1 should run");

        assert!(temp.path().join("rank-0.log").exists());
        assert!(temp.path().join("rank-1.log").exists());
        assert!(temp.path().join("rank0-checkpoint.json").exists());
        assert!(!temp.path().join("rank1-checkpoint.json").exists());
    }
}
