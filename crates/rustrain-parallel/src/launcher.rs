use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    thread::sleep,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::topology::ParallelTopology;

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
    assigned_cuda_visible_device: Option<String>,
    assigned_cuda_device_ordinal: Option<usize>,
    status_code: Option<i32>,
    timed_out: bool,
    log_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct LaunchNodeMarker {
    nnodes: usize,
    nproc_per_node: usize,
    node_rank: usize,
    master_addr: String,
    master_port: u16,
    run_id: String,
    attempt_id: String,
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
    pub assigned_cuda_visible_device: Option<String>,
    pub assigned_cuda_device_ordinal: Option<usize>,
    pub tensor_model_parallel_size: usize,
    pub pipeline_model_parallel_size: usize,
    pub data_parallel_size: usize,
    pub expert_model_parallel_size: usize,
    pub context_parallel_size: usize,
    pub parallel_rank_order: String,
}

pub fn launch(
    nproc_per_node: usize,
    output_dir: &Path,
    master_addr: &str,
    master_port: u16,
    command: &[String],
) -> Result<()> {
    launch_multi(
        nproc_per_node,
        1,
        0,
        output_dir,
        master_addr,
        master_port,
        command,
    )
}

pub fn launch_multi(
    nproc_per_node: usize,
    nnodes: usize,
    node_rank: usize,
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
    if nnodes == 0 {
        bail!("--nnodes must be greater than zero");
    }
    if node_rank >= nnodes {
        bail!("--node-rank ({node_rank}) must be less than --nnodes ({nnodes})");
    }

    let world_size = nproc_per_node * nnodes;
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let current_exe = std::env::current_exe().context("failed to locate current executable")?;
    let timeout = launch_timeout()?;
    let topology = ParallelTopology::from_env_with_world_size(world_size)?;
    let run_id = resolve_launch_run_id(nnodes, std::env::var("RUSTRAIN_RUN_ID").ok())?;
    let attempt_id = resolve_launch_attempt_id(nnodes, std::env::var("RUSTRAIN_ATTEMPT_ID").ok())?;
    let visible_cuda_devices =
        parse_visible_cuda_devices(std::env::var("CUDA_VISIBLE_DEVICES").ok());
    validate_visible_cuda_devices(nproc_per_node, visible_cuda_devices.as_deref())?;
    if nnodes > 1 {
        rendezvous_launch_nodes(
            output_dir,
            nnodes,
            nproc_per_node,
            node_rank,
            master_addr,
            master_port,
            &run_id,
            &attempt_id,
            timeout.unwrap_or(Duration::from_secs(120)),
        )?;
    }

    let mut children = Vec::with_capacity(nproc_per_node);
    for local_rank in 0..nproc_per_node {
        let global_rank = node_rank * nproc_per_node + local_rank;
        let log_path = output_dir.join(format!("rank-{global_rank}.log"));
        let log_file = fs::File::create(&log_path)
            .with_context(|| format!("failed to create {}", log_path.display()))?;
        let err_file = log_file
            .try_clone()
            .with_context(|| format!("failed to clone {}", log_path.display()))?;

        let mut child = Command::new(&current_exe);
        child
            .args(command)
            .env("RANK", global_rank.to_string())
            .env("LOCAL_RANK", local_rank.to_string())
            .env("WORLD_SIZE", world_size.to_string())
            .env("LOCAL_WORLD_SIZE", nproc_per_node.to_string())
            .env("MASTER_ADDR", master_addr)
            .env("MASTER_PORT", master_port.to_string())
            .env("RUSTRAIN_LAUNCH_OUTPUT_DIR", output_dir)
            .env("RUSTRAIN_RUN_ID", &run_id)
            .env("RUSTRAIN_ATTEMPT_ID", &attempt_id)
            .env("NNODES", nnodes.to_string())
            .env("NODE_RANK", node_rank.to_string())
            // Normalize topology variables for every child. Explicit
            // TP_SIZE/PP_SIZE/... values come from the parent environment;
            // unspecified axes default to replicated DP and are validated
            // against the launcher's world size above.
            .env("TP_SIZE", topology.tensor_model_parallel_size().to_string())
            .env(
                "PP_SIZE",
                topology.pipeline_model_parallel_size().to_string(),
            )
            .env("DP_SIZE", topology.data_parallel_size().to_string())
            .env("EP_SIZE", topology.expert_model_parallel_size().to_string())
            .env("CP_SIZE", topology.context_parallel_size().to_string())
            .env(
                "RUSTRAIN_TP_SIZE",
                topology.tensor_model_parallel_size().to_string(),
            )
            .env(
                "RUSTRAIN_PP_SIZE",
                topology.pipeline_model_parallel_size().to_string(),
            )
            .env(
                "RUSTRAIN_DP_SIZE",
                topology.data_parallel_size().to_string(),
            )
            .env(
                "RUSTRAIN_EP_SIZE",
                topology.expert_model_parallel_size().to_string(),
            )
            .env(
                "RUSTRAIN_CP_SIZE",
                topology.context_parallel_size().to_string(),
            )
            .env(
                "RUSTRAIN_PARALLEL_ORDER",
                topology
                    .order()
                    .into_iter()
                    .map(|axis| axis.name())
                    .collect::<Vec<_>>()
                    .join("-"),
            )
            .env(
                "PYTORCH_CUDA_ALLOC_CONF",
                "expandable_segments:True,max_split_size_mb:512",
            )
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(err_file));
        if let Some(assigned_device) = visible_cuda_devices
            .as_ref()
            .and_then(|devices| devices.get(local_rank))
        {
            child
                .env("RUSTRAIN_ASSIGNED_CUDA_VISIBLE_DEVICE", assigned_device)
                .env(
                    "RUSTRAIN_ASSIGNED_CUDA_DEVICE_ORDINAL",
                    local_rank.to_string(),
                );
        }
        children.push((
            global_rank,
            log_path,
            child.spawn().with_context(|| {
                format!(
                    "failed to spawn rank {global_rank} for command {:?}",
                    command
                )
            })?,
        ));
    }

    let wait_results = wait_for_ranks(children, timeout)?;
    let mut ranks = Vec::with_capacity(nproc_per_node);
    let mut failed = Vec::new();
    for wait_result in wait_results {
        let local_rank = wait_result.rank - node_rank * nproc_per_node;
        if !wait_result.success() {
            failed.push(wait_result.rank);
        }
        ranks.push(RankSummary {
            rank: wait_result.rank,
            local_rank,
            world_size,
            assigned_cuda_visible_device: visible_cuda_devices
                .as_ref()
                .and_then(|devices| devices.get(local_rank))
                .cloned(),
            assigned_cuda_device_ordinal: visible_cuda_devices
                .as_ref()
                .and_then(|devices| devices.get(local_rank))
                .map(|_| local_rank),
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
    let summary_path = if nnodes > 1 {
        output_dir.join(format!("launch-summary-node-{node_rank}.json"))
    } else {
        output_dir.join("launch-summary.json")
    };
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

fn resolve_launch_run_id(nnodes: usize, configured: Option<String>) -> Result<String> {
    if let Some(configured) = configured.filter(|value| !value.is_empty()) {
        validate_launch_id("RUSTRAIN_RUN_ID", &configured)?;
        return Ok(configured);
    }
    if nnodes > 1 {
        bail!("multi-node launch requires one shared RUSTRAIN_RUN_ID to be set on every launcher");
    }
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_nanos();
    Ok(format!("launch-{}-{nonce}", std::process::id()))
}

fn resolve_launch_attempt_id(nnodes: usize, configured: Option<String>) -> Result<String> {
    if let Some(configured) = configured.filter(|value| !value.is_empty()) {
        validate_launch_id("RUSTRAIN_ATTEMPT_ID", &configured)?;
        return Ok(configured);
    }
    if nnodes > 1 {
        bail!(
            "multi-node launch requires one shared RUSTRAIN_ATTEMPT_ID to be set on every launcher"
        );
    }
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_nanos();
    Ok(format!("attempt-{}-{nonce}", std::process::id()))
}

fn validate_launch_id(name: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        bail!("{name} may contain only ASCII letters, digits, '-', '_', and '.'");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rendezvous_launch_nodes(
    output_dir: &Path,
    nnodes: usize,
    nproc_per_node: usize,
    node_rank: usize,
    master_addr: &str,
    master_port: u16,
    run_id: &str,
    attempt_id: &str,
    timeout: Duration,
) -> Result<()> {
    let rendezvous = output_dir
        .join(".rustrain-launch")
        .join(run_id)
        .join(attempt_id);
    fs::create_dir_all(&rendezvous)
        .with_context(|| format!("failed to create {}", rendezvous.display()))?;
    let marker = LaunchNodeMarker {
        nnodes,
        nproc_per_node,
        node_rank,
        master_addr: master_addr.to_string(),
        master_port,
        run_id: run_id.to_string(),
        attempt_id: attempt_id.to_string(),
    };
    let marker_path = rendezvous.join(format!("node-{node_rank:05}.json"));
    let partial_path = rendezvous.join(format!(
        ".node-{node_rank:05}-{}.partial",
        std::process::id()
    ));
    fs::write(&partial_path, serde_json::to_vec_pretty(&marker)?)
        .with_context(|| format!("failed to write {}", partial_path.display()))?;
    fs::rename(&partial_path, &marker_path).with_context(|| {
        format!(
            "failed to publish launch rendezvous marker {}",
            marker_path.display()
        )
    })?;

    let deadline = Instant::now() + timeout;
    loop {
        let mut ready = true;
        for expected_rank in 0..nnodes {
            let path = rendezvous.join(format!("node-{expected_rank:05}.json"));
            let Some(contents) = fs::read(&path).ok() else {
                ready = false;
                break;
            };
            let observed: LaunchNodeMarker = serde_json::from_slice(&contents)
                .with_context(|| format!("failed to parse {}", path.display()))?;
            if observed.nnodes != nnodes
                || observed.nproc_per_node != nproc_per_node
                || observed.node_rank != expected_rank
                || observed.master_addr != master_addr
                || observed.master_port != master_port
                || observed.run_id != run_id
                || observed.attempt_id != attempt_id
            {
                bail!(
                    "multi-node launch rendezvous metadata differs at {}",
                    path.display()
                );
            }
        }
        if ready {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!(
                "timed out waiting for all launch nodes under {}; multi-node output_dir must be on shared storage",
                rendezvous.display()
            );
        }
        sleep(Duration::from_millis(100));
    }
}

pub fn print_launch_env() -> Result<()> {
    let summary = read_launch_env()?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn read_launch_env() -> Result<LaunchEnvSummary> {
    let topology = ParallelTopology::from_env()?;
    Ok(LaunchEnvSummary {
        rank: parse_env_usize("RANK")?,
        local_rank: parse_env_usize("LOCAL_RANK")?,
        world_size: parse_env_usize("WORLD_SIZE")?,
        local_world_size: parse_env_usize("LOCAL_WORLD_SIZE")?,
        master_addr: std::env::var("MASTER_ADDR").context("MASTER_ADDR is not set")?,
        master_port: parse_env_u16("MASTER_PORT")?,
        cuda_visible_devices: std::env::var("CUDA_VISIBLE_DEVICES").ok(),
        assigned_cuda_visible_device: std::env::var("RUSTRAIN_ASSIGNED_CUDA_VISIBLE_DEVICE").ok(),
        assigned_cuda_device_ordinal: std::env::var("RUSTRAIN_ASSIGNED_CUDA_DEVICE_ORDINAL")
            .ok()
            .map(|raw| {
                raw.parse::<usize>()
                    .with_context(|| "RUSTRAIN_ASSIGNED_CUDA_DEVICE_ORDINAL must be a usize")
            })
            .transpose()?,
        tensor_model_parallel_size: topology.tensor_model_parallel_size(),
        pipeline_model_parallel_size: topology.pipeline_model_parallel_size(),
        data_parallel_size: topology.data_parallel_size(),
        expert_model_parallel_size: topology.expert_model_parallel_size(),
        context_parallel_size: topology.context_parallel_size(),
        parallel_rank_order: topology
            .order()
            .into_iter()
            .map(|axis| axis.name())
            .collect::<Vec<_>>()
            .join("-"),
    })
}

fn parse_visible_cuda_devices(raw: Option<String>) -> Option<Vec<String>> {
    raw.map(|raw| {
        raw.split(',')
            .map(str::trim)
            .filter(|device| !device.is_empty())
            .map(ToOwned::to_owned)
            .collect()
    })
}

fn validate_visible_cuda_devices(
    nproc_per_node: usize,
    visible_cuda_devices: Option<&[String]>,
) -> Result<()> {
    let Some(visible_cuda_devices) = visible_cuda_devices else {
        return Ok(());
    };
    if visible_cuda_devices.len() < nproc_per_node {
        bail!(
            "launch requested {nproc_per_node} local ranks but CUDA_VISIBLE_DEVICES exposes only {} device(s): {}",
            visible_cuda_devices.len(),
            visible_cuda_devices.join(",")
        );
    }
    Ok(())
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

    #[test]
    fn launch_reuses_explicit_run_id_across_nodes() {
        assert_eq!(
            resolve_launch_run_id(2, Some("shared-run".into())).unwrap(),
            "shared-run"
        );
    }

    #[test]
    fn multi_node_launch_requires_explicit_run_id() {
        let error = resolve_launch_run_id(2, None).unwrap_err();
        assert!(error.to_string().contains("shared RUSTRAIN_RUN_ID"));
    }

    #[test]
    fn multi_node_launch_requires_explicit_attempt_id() {
        let error = resolve_launch_attempt_id(2, None).unwrap_err();
        assert!(error.to_string().contains("shared RUSTRAIN_ATTEMPT_ID"));
        assert_eq!(
            resolve_launch_attempt_id(2, Some("attempt-7".into())).unwrap(),
            "attempt-7"
        );
    }

    #[test]
    fn launch_ids_reject_path_components() {
        let error = resolve_launch_attempt_id(1, Some("../attempt".into())).unwrap_err();
        assert!(error.to_string().contains("RUSTRAIN_ATTEMPT_ID"));
    }

    #[test]
    fn multi_node_rendezvous_observes_every_node_on_shared_storage() {
        let temp = tempdir().unwrap();
        let root = temp.path();
        std::thread::scope(|scope| {
            let handles = (0..2)
                .map(|node_rank| {
                    scope.spawn(move || {
                        rendezvous_launch_nodes(
                            root,
                            2,
                            4,
                            node_rank,
                            "127.0.0.1",
                            29500,
                            "shared-run",
                            "attempt-7",
                            Duration::from_secs(1),
                        )
                    })
                })
                .collect::<Vec<_>>();
            for handle in handles {
                handle.join().unwrap().unwrap();
            }
        });
    }

    #[test]
    fn multi_node_rendezvous_reports_non_shared_storage() {
        let temp = tempdir().unwrap();
        let error = rendezvous_launch_nodes(
            temp.path(),
            2,
            4,
            0,
            "127.0.0.1",
            29500,
            "shared-run",
            "attempt-7",
            Duration::from_millis(20),
        )
        .unwrap_err();
        assert!(error.to_string().contains("shared storage"));
    }

    #[test]
    fn launch_parses_visible_cuda_devices() {
        let devices = parse_visible_cuda_devices(Some("0, 2,GPU-abc".to_string()))
            .expect("CUDA_VISIBLE_DEVICES should be parsed");
        assert_eq!(devices, vec!["0", "2", "GPU-abc"]);
    }

    #[test]
    fn launch_rejects_more_ranks_than_visible_cuda_devices() {
        let devices = parse_visible_cuda_devices(Some("0".to_string()))
            .expect("CUDA_VISIBLE_DEVICES should be parsed");
        let error = validate_visible_cuda_devices(2, Some(&devices))
            .expect_err("too many local ranks should be rejected");
        assert!(
            error
                .to_string()
                .contains("CUDA_VISIBLE_DEVICES exposes only 1 device")
        );
    }
}
