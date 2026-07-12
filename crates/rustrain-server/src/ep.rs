//! EP Coordinator: manages worker processes and dispatches commands via IPC.
//!
//! Parent process (HTTP server) uses `EpCoordinator` to:
//! 1. Fork N worker processes (one per GPU)
//! 2. Each worker creates NCCL communicator + Qwen36Session
//! 3. HTTP handlers call `coordinator.dispatch(cmd)` → signals all workers → waits
//!
//! Workers use `worker_main()` to enter the wait loop.

use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use rustrain_ipc::{EpChannel, EpCommand, EpResult, EpWorker};
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
    /// Create coordinator by forking `world_size` worker processes.
    /// Each worker is pinned to GPU `rank`.
    /// Returns coordinator in parent process. Child processes never return (they enter worker loop).
    pub fn launch(
        world_size: usize,
        metrics_dir: PathBuf,
    ) -> io::Result<Self> {
        let channel = EpChannel::new(world_size)?;
        let shm_name = channel.shm_name().to_string();
        let channel = Arc::new(channel);

        let mut worker_pids = Vec::with_capacity(world_size);

        for rank in 0..world_size {
            let pid = unsafe { libc::fork() };
            if pid < 0 {
                return Err(io::Error::last_os_error());
            }
            if pid == 0 {
                // Child process — enter worker loop (never returns)
                let metrics_path = metrics_dir.join(format!("ep_rank{}_metrics.jsonl", rank));
                if let Err(e) = worker_main(&shm_name, rank, world_size, metrics_path) {
                    eprintln!("[ep-worker-{}] fatal: {}", rank, e);
                    std::process::exit(1);
                }
                std::process::exit(0);
            }
            // Parent — record child PID
            worker_pids.push(pid as u32);
            tracing::info!("Launched EP worker rank {} (PID {})", rank, pid);
        }

        // Wait for workers to initialize NCCL
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

    // Set CUDA device
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
        EpCommand::TrainStep { input_ids, target_mask, attention_mask, seq_len, .. } => {
            let sl = *seq_len as i64;
            let input_ids_tensor = tch::Tensor::from_slice(input_ids).reshape(&[1, sl]).to_device(session.device());
            let target_mask_tensor = tch::Tensor::from_slice(target_mask).reshape(&[1, sl]).to_device(session.device());
            let attention_mask_tensor = tch::Tensor::from_slice(attention_mask).reshape(&[1, sl]).to_device(session.device());

            match session.train_step(TrainInput {
                input_ids: input_ids_tensor,
                target_mask: target_mask_tensor,
                attention_mask: attention_mask_tensor,
            }) {
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
        EpCommand::ExportAdapter { path, .. } => {
            match session.export_adapter(path) {
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
