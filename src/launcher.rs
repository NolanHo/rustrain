use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    thread::sleep,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
struct LaunchSummary {
    nproc_per_node: usize,
    command: Vec<String>,
    output_dir: String,
    ranks: Vec<RankSummary>,
}

#[derive(Debug, Serialize)]
struct RankSummary {
    rank: usize,
    local_rank: usize,
    world_size: usize,
    status_code: Option<i32>,
    timed_out: bool,
    log_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LaunchEnvSummary {
    pub rank: usize,
    pub local_rank: usize,
    pub world_size: usize,
    pub local_world_size: usize,
    pub master_addr: String,
    pub master_port: u16,
    pub cuda_visible_devices: Option<String>,
}

pub fn launch(
    nproc_per_node: usize,
    output_dir: &Path,
    master_addr: &str,
    master_port: u16,
    command: &[String],
) -> Result<()> {
    if nproc_per_node == 0 {
        bail!("--nproc-per-node must be greater than zero");
    }
    if command.is_empty() {
        bail!(
            "launch requires a child command, for example: launch --nproc-per-node 2 tch-cuda-probe"
        );
    }

    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let current_exe = std::env::current_exe().context("failed to locate current executable")?;
    let timeout = launch_timeout()?;

    let mut children = Vec::with_capacity(nproc_per_node);
    for rank in 0..nproc_per_node {
        let log_path = output_dir.join(format!("rank-{rank}.log"));
        let log_file = fs::File::create(&log_path)
            .with_context(|| format!("failed to create {}", log_path.display()))?;
        let err_file = log_file
            .try_clone()
            .with_context(|| format!("failed to clone {}", log_path.display()))?;

        let mut child = Command::new(&current_exe);
        child
            .args(command)
            .env("RANK", rank.to_string())
            .env("LOCAL_RANK", rank.to_string())
            .env("WORLD_SIZE", nproc_per_node.to_string())
            .env("LOCAL_WORLD_SIZE", nproc_per_node.to_string())
            .env("MASTER_ADDR", master_addr)
            .env("MASTER_PORT", master_port.to_string())
            .env("RUSTRAIN_LAUNCH_OUTPUT_DIR", output_dir)
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(err_file));
        children.push((
            rank,
            log_path,
            child.spawn().with_context(|| {
                format!("failed to spawn rank {rank} for command {:?}", command)
            })?,
        ));
    }

    let wait_results = wait_for_ranks(children, timeout)?;
    let mut ranks = Vec::with_capacity(nproc_per_node);
    let mut failed = Vec::new();
    for wait_result in wait_results {
        if !wait_result.success() {
            failed.push(wait_result.rank);
        }
        ranks.push(RankSummary {
            rank: wait_result.rank,
            local_rank: wait_result.rank,
            world_size: nproc_per_node,
            status_code: wait_result.status.and_then(|status| status.code()),
            timed_out: wait_result.timed_out,
            log_path: wait_result.log_path.display().to_string(),
        });
    }

    let summary = LaunchSummary {
        nproc_per_node,
        command: command.to_vec(),
        output_dir: output_dir.display().to_string(),
        ranks,
    };
    let summary_json = serde_json::to_string_pretty(&summary)?;
    let summary_path = output_dir.join("launch-summary.json");
    fs::write(&summary_path, &summary_json)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{summary_json}");

    if !failed.is_empty() {
        let mut details = Vec::new();
        for rank in &failed {
            let log_path = output_dir.join(format!("rank-{rank}.log"));
            let log = fs::read_to_string(&log_path)
                .unwrap_or_else(|error| format!("failed to read {}: {error}", log_path.display()));
            details.push(format!("rank {rank} log {}:\n{log}", log_path.display()));
        }
        bail!("launch ranks failed: {failed:?}\n{}", details.join("\n"));
    }

    Ok(())
}

pub fn print_launch_env() -> Result<()> {
    let summary = read_launch_env()?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn read_launch_env() -> Result<LaunchEnvSummary> {
    Ok(LaunchEnvSummary {
        rank: parse_env_usize("RANK")?,
        local_rank: parse_env_usize("LOCAL_RANK")?,
        world_size: parse_env_usize("WORLD_SIZE")?,
        local_world_size: parse_env_usize("LOCAL_WORLD_SIZE")?,
        master_addr: std::env::var("MASTER_ADDR").context("MASTER_ADDR is not set")?,
        master_port: parse_env_u16("MASTER_PORT")?,
        cuda_visible_devices: std::env::var("CUDA_VISIBLE_DEVICES").ok(),
    })
}

#[derive(Debug)]
struct RankWaitResult {
    rank: usize,
    log_path: PathBuf,
    status: Option<ExitStatus>,
    timed_out: bool,
}

impl RankWaitResult {
    fn success(&self) -> bool {
        !self.timed_out && self.status.is_some_and(|status| status.success())
    }
}

fn wait_for_ranks(
    children: Vec<(usize, PathBuf, Child)>,
    timeout: Option<Duration>,
) -> Result<Vec<RankWaitResult>> {
    let start = Instant::now();
    let mut running: Vec<(usize, PathBuf, Child)> = children;
    let mut results = Vec::with_capacity(running.len());
    loop {
        let mut index = 0;
        while index < running.len() {
            let (rank, _, child) = &mut running[index];
            if let Some(status) = child
                .try_wait()
                .with_context(|| format!("failed to poll rank {rank}"))?
            {
                let (rank, log_path, _) = running.remove(index);
                results.push(RankWaitResult {
                    rank,
                    log_path,
                    status: Some(status),
                    timed_out: false,
                });
            } else {
                index += 1;
            }
        }
        if running.is_empty() {
            results.sort_by_key(|result| result.rank);
            return Ok(results);
        }
        if timeout.is_some_and(|timeout| start.elapsed() >= timeout) {
            for (rank, log_path, child) in &mut running {
                let _ = child.kill();
                let status = child.wait().ok();
                results.push(RankWaitResult {
                    rank: *rank,
                    log_path: log_path.clone(),
                    status,
                    timed_out: true,
                });
            }
            results.sort_by_key(|result| result.rank);
            return Ok(results);
        }
        sleep(Duration::from_millis(100));
    }
}

fn launch_timeout() -> Result<Option<Duration>> {
    let Some(raw) = std::env::var("RUSTRAIN_LAUNCH_TIMEOUT_SECS").ok() else {
        return Ok(None);
    };
    let seconds = raw
        .parse::<u64>()
        .with_context(|| "RUSTRAIN_LAUNCH_TIMEOUT_SECS must be an integer number of seconds")?;
    if seconds == 0 {
        return Ok(None);
    }
    Ok(Some(Duration::from_secs(seconds)))
}

fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

fn parse_env_u16(name: &str) -> Result<u16> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set"))?
        .parse::<u16>()
        .with_context(|| format!("{name} must be a u16"))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn launch_rejects_empty_child_command() {
        let temp = tempdir().expect("temp dir should be created");
        let error = launch(2, temp.path(), "127.0.0.1", 29500, &[])
            .expect_err("empty command should be rejected");
        assert!(error.to_string().contains("requires a child command"));
    }
}
