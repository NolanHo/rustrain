//! EP Coordinator: manages worker processes and dispatches commands via IPC.
//!
//! Parent process (HTTP server) uses `EpCoordinator` to:
//! 1. Fork N worker processes (one per GPU)
//! 2. Each worker creates NCCL communicator + Qwen36Session
//! 3. HTTP handlers call `coordinator.dispatch(cmd)` → signals all workers → waits
//!
//! Workers use `worker_main()` to enter the wait loop.

use std::io;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use rustrain_ipc::{EpChannel, EpCommand, EpResult, EpWorker};
use rustrain_parallel::topology::ParallelTopology;
use tch::{Device, Kind};

use crate::session::{
    AddLoRARequest, EvalOutput, InitLoRARequest, Qwen36Session, SessLoadDatasetRequest,
    SessLoadModelRequest, TrainInput, TrainOutput, TrainingSession,
};

/// Coordinator for EP workers. Lives in the HTTP server process.
/// Holds no GPU resources — only IPC state.
pub struct EpCoordinator {
    channel: Arc<EpChannel>,
    worker_pids: Vec<u32>,
}

impl EpCoordinator {
    /// Create coordinator by spawning `world_size` worker processes.
    /// Each worker is pinned to GPU `rank`.
    /// Uses exec (not fork) because CUDA + fork is incompatible — forked children
    /// inherit parent's CUDA context but can't use it.
    pub fn launch(
        world_size: usize,
        metrics_dir: PathBuf,
    ) -> io::Result<Self> {
        let channel = EpChannel::new(world_size)?;
        let shm_name = channel.shm_name().to_string();
        let channel = Arc::new(channel);

        let exe = std::env::current_exe()
            .map_err(|e| io::Error::other(format!("current_exe: {e}")))?;
        let mut worker_pids = Vec::with_capacity(world_size);

        for rank in 0..world_size {
            let metrics_path = metrics_dir.join(format!("ep_rank{}_metrics.jsonl", rank));
            let child = std::process::Command::new(&exe)
                .arg("ep-worker")
                .arg("--shm-name").arg(&shm_name)
                .arg("--rank").arg(rank.to_string())
                .arg("--world-size").arg(world_size.to_string())
                .arg("--metrics-path").arg(&metrics_path)
                .env("RANK", rank.to_string())
                .env("WORLD_SIZE", world_size.to_string())
                .env("LOCAL_RANK", rank.to_string())
                .env("RUSTRAIN_NCCL_RUN_ID", &shm_name)
                .env("QWEN36_LOSS_DIAG", std::env::var("QWEN36_LOSS_DIAG").unwrap_or_default())
                .env("QWEN36_GROUP_SIZE", std::env::var("QWEN36_GROUP_SIZE").unwrap_or_default())
                .env("QWEN36_FUSED_CE", std::env::var("QWEN36_FUSED_CE").unwrap_or_default())
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::inherit())
                .spawn()
                .map_err(|e| io::Error::other(format!("spawn worker {rank}: {e}")))?;
            let pid = child.id();
            // Keep child handle alive — store in a static to prevent early drop
            // (drop would kill the child)
            std::mem::forget(child);
            worker_pids.push(pid);
            tracing::info!("Launched EP worker rank {} (PID {})", rank, pid);
        }

        // Wait for workers to initialize
        std::thread::sleep(std::time::Duration::from_secs(2));

        Ok(Self { channel, worker_pids })
    }

    /// Dispatch a command to all workers, wait for completion, return rank 0's result.
    pub fn dispatch(&self, cmd: &EpCommand) -> EpResult {
        match self.channel.broadcast(cmd) {
            Ok(result) => result,
            Err(e) => EpResult::Error(format!("IPC error: {}", e)),
        }
    }

    /// Send shutdown command to all workers.
    pub fn shutdown(&self) {
        let _ = self.channel.broadcast(&EpCommand::Shutdown);
        // Wait for worker processes to exit
        for pid in &self.worker_pids {
            unsafe {
                libc::waitpid(*pid as i32, std::ptr::null_mut(), 0);
            }
        }
    }
}

impl Drop for EpCoordinator {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Worker main loop. Attaches to shared memory, creates NCCL communicator,
/// then enters wait loop: receive command → execute → signal done.
pub fn worker_main(
    shm_name: &str,
    rank: usize,
    world_size: usize,
    metrics_path: PathBuf,
) -> io::Result<()> {
    // Attach to shared memory
    let worker = EpWorker::attach(shm_name, rank, world_size)?;

    // CRITICAL: Set CUDA device BEFORE any PyTorch operation.
    // exec'd worker processes have no CUDA context — must initialize on
    // the correct device before load_model's to_device() call.
    rustrain_qwen3_6::kernel::CppTrainingContext::set_cuda_device(rank as i32);

    let device = Device::Cuda(rank);
    let compute_kind = Kind::BFloat16;

    // Create session (same as non-EP, but with LOCAL_RANK env for EP)
    // SAFETY: set_var is unsafe in Rust 2024 edition. We're in a forked child process
    // before any other threads exist, so this is safe.
    unsafe {
        std::env::set_var("RANK", rank.to_string());
        std::env::set_var("WORLD_SIZE", world_size.to_string());
        std::env::set_var("LOCAL_RANK", rank.to_string());
    }

    let mut session = Qwen36Session::new(device, compute_kind, metrics_path);

    // Pre-create NCCL communicator — all workers reach this point simultaneously
    // because EpCoordinator::launch forks all workers before they enter the loop.
    // NCCL communicator init is a collective operation requiring all ranks.
    // We use a CreateSession broadcast to synchronize all workers for NCCL init.
    // (The first CreateSession command triggers qwen36_init_nccl inside init_lora)
    tracing::info!("EP worker {} ready, entering command loop", rank);

    loop {
        let cmd = match worker.wait_command() {
            Ok(cmd) => cmd,
            Err(e) => {
                eprintln!("[ep-worker-{}] wait_command error: {}", rank, e);
                return Err(e);
            }
        };

        let result = execute_command(&mut session, &cmd);

        if let Err(e) = worker.signal_done(&result) {
            eprintln!("[ep-worker-{}] signal_done error: {}", rank, e);
            return Err(e);
        }

        if matches!(cmd, EpCommand::Shutdown) {
            tracing::info!("EP worker {} shutting down", rank);
            break;
        }
    }

    Ok(())
}

/// Execute a command on the local session, return result.
fn execute_command(session: &mut Qwen36Session, cmd: &EpCommand) -> EpResult {
    match cmd {
        EpCommand::CreateSession { session_id: _ } => {
            // Session already created in worker_main; just acknowledge
            EpResult::Ok
        }
        EpCommand::DeleteSession { .. } => {
            // In EP mode, we don't really delete the session — just reset state
            // A full implementation would manage session lifecycle per worker
            EpResult::Ok
        }
        EpCommand::LoadModel { model_path, config_toml, .. } => {
            match session.load_model(SessLoadModelRequest {
                model_path: model_path.clone(),
                config_toml: config_toml.clone(),
            }) {
                Ok(()) => EpResult::Ok,
                Err(e) => EpResult::Error(e.to_string()),
            }
        }
        EpCommand::LoadDataset { jsonl_path, seq_len, .. } => {
            match session.load_dataset(SessLoadDatasetRequest {
                jsonl_path: jsonl_path.clone(),
                seq_len: *seq_len,
            }) {
                Ok(n) => EpResult::Count(n),
                Err(e) => EpResult::Error(e.to_string()),
            }
        }
        EpCommand::InitLora { rank, alpha, target_layers, target_modules, lr, beta1, beta2, eps, .. } => {
            match session.init_lora(InitLoRARequest {
                rank: *rank,
                alpha: *alpha,
                target_layers: target_layers.clone(),
                target_modules: target_modules.clone(),
                lr: *lr,
                beta1: *beta1,
                beta2: *beta2,
                eps: *eps,
            }) {
                Ok(n) => EpResult::Count(n),
                Err(e) => EpResult::Error(e.to_string()),
            }
        }
        EpCommand::AddLora { rank, alpha, target_layers, target_modules, .. } => {
            match session.add_lora(AddLoRARequest {
                rank: *rank,
                alpha: *alpha,
                target_layers: target_layers.clone(),
                target_modules: target_modules.clone(),
            }) {
                Ok(id) => EpResult::AdapterId(id),
                Err(e) => EpResult::Error(e.to_string()),
            }
        }
        EpCommand::BatchAddLora { count, rank, alpha, target_layers, target_modules, .. } => {
            let mut ids = Vec::with_capacity(*count as usize);
            for _ in 0..*count {
                match session.add_lora(AddLoRARequest {
                    rank: *rank,
                    alpha: *alpha,
                    target_layers: target_layers.clone(),
                    target_modules: target_modules.clone(),
                }) {
                    Ok(id) => ids.push(id),
                    Err(e) => { return EpResult::Error(e.to_string()); }
                }
            }
            EpResult::Count(ids.len() as usize)
        }
        EpCommand::RemoveLora { adapter_id, .. } => {
            match session.remove_lora(*adapter_id) {
                Ok(b) => {
                    if b { EpResult::Ok } else { EpResult::Error("adapter not found".into()) }
                }
                Err(e) => EpResult::Error(e.to_string()),
            }
        }
        EpCommand::ListLora { .. } => {
            EpResult::AdapterIds(session.list_lora())
        }
        EpCommand::TrainStep {
            input_ids,
            target_mask,
            attention_mask,
            batch_size,
            seq_len,
            ..
        } => {
            let dp_shard = match dense_dp_shard_from_env() {
                Ok(shard) => shard,
                Err(error) => return EpResult::Error(error),
            };
            let rows = match train_step_rows(*batch_size, dp_shard) {
                Ok(rows) => rows,
                Err(error) => return EpResult::Error(error),
            };
            let elements = match tensor_element_range(*batch_size, *seq_len, rows.clone()) {
                Ok(elements) => elements,
                Err(error) => return EpResult::Error(error),
            };
            if let Err(error) = validate_flat_tensor_lengths(
                *batch_size,
                *seq_len,
                input_ids.len(),
                target_mask.len(),
                attention_mask.len(),
            ) {
                return EpResult::Error(error);
            }
            let sl = *seq_len as i64;
            let local_batch = rows.len() as i64;
            let input_ids_tensor = tch::Tensor::from_slice(&input_ids[elements.clone()])
                .reshape(&[local_batch, sl])
                .to_device(session.device());
            let target_mask_tensor = tch::Tensor::from_slice(&target_mask[elements.clone()])
                .reshape(&[local_batch, sl])
                .to_device(session.device());
            let attention_mask_tensor = tch::Tensor::from_slice(&attention_mask[elements])
                .reshape(&[local_batch, sl])
                .to_device(session.device());

            match session.train_step(TrainInput {
                input_ids: input_ids_tensor,
                target_mask: target_mask_tensor,
                attention_mask: attention_mask_tensor,
            }) {
                Ok(TrainOutput { loss, .. }) => EpResult::Loss(loss),
                Err(e) => EpResult::Error(e.to_string()),
            }
        }
        EpCommand::TrainMultiLora {
            input_ids,
            target_mask,
            attention_mask,
            batch_size,
            seq_len,
            n_total,
            lora_rank,
            adapter_ids,
            ..
        } => {
            if let Err(error) = validate_flat_tensor_lengths(
                *batch_size,
                *seq_len,
                input_ids.len(),
                target_mask.len(),
                attention_mask.len(),
            ) {
                return EpResult::Error(error);
            }
            let dp_shard = match dense_dp_shard_from_env() {
                Ok(shard) => shard,
                Err(error) => return EpResult::Error(error),
            };
            let rows = match multi_lora_rows(*batch_size, *n_total, dp_shard) {
                Ok(rows) => rows,
                Err(error) => return EpResult::Error(error),
            };
            let elements = match tensor_element_range(*batch_size, *seq_len, rows.clone()) {
                Ok(elements) => elements,
                Err(error) => return EpResult::Error(error),
            };
            let sl = *seq_len as i64;
            let local_batch = rows.len() as i64;
            let input_ids_tensor = tch::Tensor::from_slice(&input_ids[elements.clone()])
                .reshape(&[local_batch, sl])
                .to_device(session.device());
            let target_mask_tensor = tch::Tensor::from_slice(&target_mask[elements.clone()])
                .reshape(&[local_batch, sl])
                .to_device(session.device());
            let attention_mask_tensor = tch::Tensor::from_slice(&attention_mask[elements])
                .reshape(&[local_batch, sl])
                .to_device(session.device());

            match session.train_multi_lora(
                TrainInput {
                    input_ids: input_ids_tensor,
                    target_mask: target_mask_tensor,
                    attention_mask: attention_mask_tensor,
                },
                *n_total,
                *lora_rank,
                adapter_ids,
            ) {
                Ok(TrainOutput { loss, .. }) => EpResult::Loss(loss),
                Err(e) => EpResult::Error(e.to_string()),
            }
        }
        EpCommand::EvalStep { input_ids, target_mask, attention_mask, seq_len, .. } => {
            let sl = *seq_len as i64;
            let input_ids_tensor = tch::Tensor::from_slice(input_ids).reshape(&[1, sl]).to_device(session.device());
            let target_mask_tensor = tch::Tensor::from_slice(target_mask).reshape(&[1, sl]).to_device(session.device());
            let attention_mask_tensor = tch::Tensor::from_slice(attention_mask).reshape(&[1, sl]).to_device(session.device());

            match session.eval_step(TrainInput {
                input_ids: input_ids_tensor,
                target_mask: target_mask_tensor,
                attention_mask: attention_mask_tensor,
            }) {
                Ok(EvalOutput { loss }) => EpResult::Loss(loss),
                Err(e) => EpResult::Error(e.to_string()),
            }
        }
        EpCommand::ExportAdapter {
            path, adapter_id, ..
        } => {
            match session.export_adapter(path, *adapter_id) {
                Ok(n) => EpResult::Count(n),
                Err(e) => EpResult::Error(e.to_string()),
            }
        }
        EpCommand::Status { .. } => {
            let s = session.status();
            EpResult::Status {
                state: s.state,
                step: s.step,
                last_loss: s.last_loss,
                model_path: s.model_path,
            }
        }
        EpCommand::Shutdown => {
            EpResult::Ok
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DpShard {
    rank: usize,
    size: usize,
}

fn dense_dp_shard_from_env() -> Result<Option<DpShard>, String> {
    let data_parallel = std::env::var("RUSTRAIN_DATA_PARALLEL")
        .map(|value| !value.is_empty() && value != "0")
        .unwrap_or(false);
    if !data_parallel {
        return Ok(None);
    }

    let topology = ParallelTopology::from_env()
        .map_err(|error| format!("invalid dense DP topology: {error}"))?;
    let global_rank = std::env::var("RANK")
        .map_err(|_| "RANK is required for dense data parallel training".to_string())?
        .parse::<usize>()
        .map_err(|_| "RANK must be a non-negative integer".to_string())?;
    dp_shard_for_topology(&topology, global_rank)
}

fn dp_shard_for_topology(
    topology: &ParallelTopology,
    global_rank: usize,
) -> Result<Option<DpShard>, String> {
    let size = topology.data_parallel_size();
    if size <= 1 {
        return Ok(None);
    }
    let rank = topology
        .data_rank(global_rank)
        .map_err(|error| format!("invalid dense DP rank: {error}"))?;
    Ok(Some(DpShard { rank, size }))
}

fn validate_flat_tensor_lengths(
    batch_size: usize,
    seq_len: usize,
    input_len: usize,
    target_len: usize,
    attention_len: usize,
) -> Result<(), String> {
    if batch_size == 0 || seq_len == 0 {
        return Err(format!(
            "batch_size and seq_len must be positive, got batch_size={batch_size} seq_len={seq_len}"
        ));
    }
    let expected = batch_size
        .checked_mul(seq_len)
        .ok_or_else(|| "batch tensor element count overflowed usize".to_string())?;
    if input_len != expected || target_len != expected || attention_len != expected {
        return Err(format!(
            "tensor lengths must match batch_size={batch_size} seq_len={seq_len} (expected {expected}), got input={input_len} target={target_len} attention={attention_len}"
        ));
    }
    Ok(())
}

fn train_step_rows(batch_size: usize, shard: Option<DpShard>) -> Result<Range<usize>, String> {
    let Some(shard) = shard else {
        return Ok(0..batch_size);
    };
    if shard.size == 0 || shard.rank >= shard.size {
        return Err(format!(
            "invalid DP shard rank={}/size={}",
            shard.rank, shard.size
        ));
    }
    if batch_size % shard.size != 0 {
        return Err(format!(
            "global batch_size={batch_size} must be divisible by DP size {}",
            shard.size
        ));
    }
    let local_batch = batch_size / shard.size;
    let start = shard.rank * local_batch;
    Ok(start..start + local_batch)
}

fn multi_lora_rows(
    batch_size: usize,
    n_total: i32,
    shard: Option<DpShard>,
) -> Result<Range<usize>, String> {
    if n_total <= 0 {
        return Err(format!("n_total must be positive, got {n_total}"));
    }
    let n_total = n_total as usize;
    if batch_size == 1 || batch_size == n_total {
        return Ok(0..batch_size);
    }
    let Some(shard) = shard else {
        return Err(format!(
            "multi-LoRA batch_size={batch_size} must be 1 or n_total={n_total} when DP is disabled"
        ));
    };
    if shard.size == 0 || shard.rank >= shard.size {
        return Err(format!(
            "invalid DP shard rank={}/size={}",
            shard.rank, shard.size
        ));
    }
    let global_rows = n_total
        .checked_mul(shard.size)
        .ok_or_else(|| "multi-LoRA global row count overflowed usize".to_string())?;
    if batch_size != global_rows {
        return Err(format!(
            "multi-LoRA batch_size={batch_size} must be n_total={n_total} (replicated) or n_total*dp_size={global_rows} (global)"
        ));
    }
    let start = shard.rank * n_total;
    Ok(start..start + n_total)
}

fn tensor_element_range(
    batch_size: usize,
    seq_len: usize,
    rows: Range<usize>,
) -> Result<Range<usize>, String> {
    if rows.start > rows.end || rows.end > batch_size {
        return Err(format!(
            "row range {:?} is outside batch_size={batch_size}",
            rows
        ));
    }
    let start = rows
        .start
        .checked_mul(seq_len)
        .ok_or_else(|| "tensor slice start overflowed usize".to_string())?;
    let end = rows
        .end
        .checked_mul(seq_len)
        .ok_or_else(|| "tensor slice end overflowed usize".to_string())?;
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use super::{
        DpShard, dp_shard_for_topology, multi_lora_rows, tensor_element_range,
        train_step_rows, validate_flat_tensor_lengths,
    };
    use rustrain_parallel::topology::ParallelTopology;

    #[test]
    fn train_step_slices_global_batch_contiguously_by_dp_rank() {
        assert_eq!(
            train_step_rows(8, Some(DpShard { rank: 0, size: 2 })).unwrap(),
            0..4
        );
        assert_eq!(
            train_step_rows(8, Some(DpShard { rank: 1, size: 2 })).unwrap(),
            4..8
        );
        assert!(train_step_rows(7, Some(DpShard { rank: 0, size: 2 })).is_err());
        assert_eq!(train_step_rows(7, None).unwrap(), 0..7);
    }

    #[test]
    fn tp_peers_with_the_same_dp_rank_select_the_same_rows() {
        let topology = ParallelTopology::new(2, 1, 2, 1, 1).unwrap();
        let tp_rank_zero_rows = train_step_rows(
            12,
            dp_shard_for_topology(&topology, 2).unwrap(),
        )
        .unwrap();
        let tp_rank_one_rows = train_step_rows(
            12,
            dp_shard_for_topology(&topology, 3).unwrap(),
        )
        .unwrap();
        assert_eq!(tp_rank_zero_rows, tp_rank_one_rows);
        assert_eq!(tp_rank_zero_rows, 6..12);
    }

    #[test]
    fn custom_rank_order_keeps_tp_peers_on_the_same_dp_shard() {
        let topology = ParallelTopology::with_order(2, 1, 2, 1, 1, "dp-tp").unwrap();
        let first_tp_peer = train_step_rows(
            12,
            dp_shard_for_topology(&topology, 1).unwrap(),
        )
        .unwrap();
        let second_tp_peer = train_step_rows(
            12,
            dp_shard_for_topology(&topology, 3).unwrap(),
        )
        .unwrap();
        assert_eq!(first_tp_peer, second_tp_peer);
        assert_eq!(first_tp_peer, 6..12);
    }

    #[test]
    fn multi_lora_supports_replicated_and_global_dp_rows() {
        let shard = DpShard { rank: 1, size: 2 };
        assert_eq!(multi_lora_rows(3, 3, Some(shard)).unwrap(), 0..3);
        assert_eq!(multi_lora_rows(6, 3, Some(shard)).unwrap(), 3..6);
        assert_eq!(multi_lora_rows(1, 3, Some(shard)).unwrap(), 0..1);
        assert!(multi_lora_rows(9, 3, Some(shard)).is_err());
        assert!(multi_lora_rows(6, 3, None).is_err());
    }

    #[test]
    fn validates_flat_shapes_before_tensor_slicing() {
        validate_flat_tensor_lengths(4, 8, 32, 32, 32).unwrap();
        assert!(validate_flat_tensor_lengths(4, 8, 31, 32, 32).is_err());
        assert!(validate_flat_tensor_lengths(0, 8, 0, 0, 0).is_err());
        assert_eq!(tensor_element_range(4, 8, 1..3).unwrap(), 8..24);
    }
}
