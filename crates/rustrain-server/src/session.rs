//! Training session trait + Qwen3.6 implementation.

use anyhow::{Context, Result, anyhow, bail};
use std::path::PathBuf;
use std::sync::Arc;
use tch::{Device, Kind, Tensor};
use tokio::sync::Mutex;

use crate::checkpoint;
use crate::metrics::{FileMetricsSink, MetricsSink, StepMetric};
use rustrain_parallel::topology::ParallelTopology;
use rustrain_qwen3_6::lora::{Qwen36AdapterArtifact, Qwen36LoraConfig, Qwen36LoraTargetModule};
use rustrain_qwen3_6::pipeline::{PipelineStageLayout, stage_lora_slots};

fn validate_qwen_parallel_features(
    is_moe: bool,
    tp_size: usize,
    ep_size: usize,
    ep_a2a: bool,
    ep_a2a_sharded: bool,
) -> Result<()> {
    let is_ep = is_moe && ep_size > 1;
    if ep_a2a_sharded && !is_ep {
        bail!("QWEN36_EP_A2A_SHARDED=1 requires expert-parallel training");
    }
    if ep_a2a_sharded && !ep_a2a {
        bail!("QWEN36_EP_A2A_SHARDED=1 requires QWEN36_EP_A2A=1");
    }
    if is_moe && tp_size > 1 && ep_size > 1 && !ep_a2a_sharded {
        bail!(
            "Qwen server MoE TPxEP requires QWEN36_EP_A2A_SHARDED=1; replicated expert TP is not supported by the native kernel"
        );
    }
    Ok(())
}

fn validate_tp_intermediate_sizes(tp_size: usize, sizes: &[(&str, i64)]) -> Result<()> {
    for (kind, intermediate) in sizes {
        if *intermediate <= 0 || *intermediate % tp_size as i64 != 0 {
            bail!("{kind} intermediate_size={intermediate} must be divisible by TP_SIZE={tp_size}");
        }
    }
    Ok(())
}

struct ResolvedQwenTopology {
    topology: ParallelTopology,
    global_rank: usize,
    world_size: usize,
    local_rank: usize,
    tp_rank: usize,
    cp_rank: usize,
    ep_rank: usize,
    dp_rank: usize,
    pp_rank: usize,
    tp_size: usize,
    cp_size: usize,
    ep_size: usize,
    dp_size: usize,
    pp_size: usize,
}

fn resolve_qwen_topology(
    runtime_config: &rustrain_qwen3_6::config::Qwen36RuntimeConfig,
) -> Result<ResolvedQwenTopology> {
    let global_rank = std::env::var("RANK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let world_size = std::env::var("WORLD_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let local_rank = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let tp_size = std::env::var("TP_SIZE")
        .or_else(|_| std::env::var("RUSTRAIN_TP_SIZE"))
        .or_else(|_| std::env::var("TENSOR_MODEL_PARALLEL_SIZE"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let has_explicit_non_tp_axis = [
        "EP_SIZE",
        "RUSTRAIN_EP_SIZE",
        "EXPERT_MODEL_PARALLEL_SIZE",
        "DP_SIZE",
        "RUSTRAIN_DP_SIZE",
        "DATA_PARALLEL_SIZE",
        "PP_SIZE",
        "RUSTRAIN_PP_SIZE",
        "PIPELINE_MODEL_PARALLEL_SIZE",
        "CP_SIZE",
        "RUSTRAIN_CP_SIZE",
        "CONTEXT_PARALLEL_SIZE",
    ]
    .iter()
    .any(|name| std::env::var_os(name).is_some());
    let topology = if runtime_config.is_moe && !has_explicit_non_tp_axis {
        let ep_size = world_size
            .checked_div(tp_size)
            .filter(|size| size * tp_size == world_size)
            .ok_or_else(|| {
                anyhow!("WORLD_SIZE={world_size} is not divisible by TP_SIZE={tp_size}")
            })?;
        let rank_order = std::env::var("RUSTRAIN_PARALLEL_ORDER")
            .or_else(|_| std::env::var("PARALLEL_ORDER"))
            .unwrap_or_else(|_| rustrain_parallel::topology::DEFAULT_RANK_ORDER.to_string());
        ParallelTopology::with_order(tp_size, 1, 1, ep_size, 1, &rank_order)?
    } else {
        ParallelTopology::from_env_with_world_size(world_size)?
    };
    if topology.context_parallel_size() != 1 {
        bail!(
            "native Qwen server does not yet support CP (tp={} pp={} dp={} ep={} cp={})",
            topology.tensor_model_parallel_size(),
            topology.pipeline_model_parallel_size(),
            topology.data_parallel_size(),
            topology.expert_model_parallel_size(),
            topology.context_parallel_size(),
        );
    }
    if topology.tensor_model_parallel_size() != tp_size {
        bail!(
            "Qwen server topology TP={} does not match TP_SIZE={tp_size}",
            topology.tensor_model_parallel_size()
        );
    }
    topology.coordinates(global_rank)?;
    if !runtime_config.is_moe && topology.expert_model_parallel_size() != 1 {
        bail!("native dense Qwen server requires expert_model_parallel_size=1");
    }

    Ok(ResolvedQwenTopology {
        tp_rank: topology.tensor_rank(global_rank)?,
        cp_rank: topology.context_rank(global_rank)?,
        ep_rank: topology.expert_rank(global_rank)?,
        dp_rank: topology.data_rank(global_rank)?,
        pp_rank: topology.pipeline_rank(global_rank)?,
        tp_size: topology.tensor_model_parallel_size(),
        cp_size: topology.context_parallel_size(),
        ep_size: topology.expert_model_parallel_size(),
        dp_size: topology.data_parallel_size(),
        pp_size: topology.pipeline_model_parallel_size(),
        topology,
        global_rank,
        world_size,
        local_rank,
    })
}

fn validate_dynamic_adapter_manifests<'a>(
    manifests: impl IntoIterator<Item = &'a checkpoint::DynamicAdapterManifest>,
) -> Result<()> {
    let mut adapter_ids = std::collections::BTreeSet::new();
    for manifest in manifests {
        if manifest.id <= 0 || !adapter_ids.insert(manifest.id) {
            bail!("checkpoint dynamic adapter IDs must be positive and unique");
        }
        if i64::try_from(manifest.optimizer_step).is_err() {
            bail!(
                "dynamic adapter {} optimizer step exceeds native range",
                manifest.id
            );
        }
        if manifest
            .optimizer_lr
            .is_some_and(|optimizer_lr| !optimizer_lr.is_finite() || optimizer_lr < 0.0)
        {
            bail!(
                "dynamic adapter {} optimizer learning rate must be finite and non-negative",
                manifest.id
            );
        }
    }
    Ok(())
}

/// Session states.
#[derive(Debug, Clone)]
pub enum SessionState {
    Unloaded,
    Loaded { model_path: String },
    Ready { model_path: String },
    Training { step: u64 },
    Paused { step: u64 },
    Error(String),
}

/// Request types (prefixed with Sess to avoid clash with gRPC generated types).
#[derive(Debug)]
pub struct SessLoadModelRequest {
    pub model_path: String,
    pub config_toml: String,
}

#[derive(Debug)]
pub struct SessLoadDatasetRequest {
    pub jsonl_path: String,
    pub seq_len: usize,
}

#[derive(Debug, Clone)]
pub struct InitLoRARequest {
    pub rank: i64,
    pub alpha: f64,
    pub target_layers: Vec<usize>,
    pub target_modules: Vec<String>,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

#[derive(Debug)]
pub struct TrainInput {
    pub input_ids: Tensor,
    pub target_mask: Tensor,
    pub attention_mask: Tensor,
}

#[derive(Debug)]
pub struct TrainOutput {
    pub loss: f64,
    pub step: u64,
}

#[derive(Debug)]
pub struct MultiLoraTrainOutput {
    pub loss: f64,
    pub adapter_losses: Vec<f64>,
    pub step: u64,
}

#[derive(Debug)]
pub struct EvalOutput {
    pub loss: f64,
}

#[derive(Debug)]
pub struct SessionStatus {
    pub state: String,
    pub step: u64,
    pub last_loss: f64,
    pub model_path: String,
}

/// Training session trait — each model implements this.
pub trait TrainingSession: Send {
    fn load_model(&mut self, req: SessLoadModelRequest) -> Result<()>;
    fn load_dataset(&mut self, req: SessLoadDatasetRequest) -> Result<usize>;
    fn init_lora(&mut self, req: InitLoRARequest) -> Result<usize>;
    fn train_step(&mut self, input: TrainInput) -> Result<TrainOutput>;
    fn train_multi_lora(
        &mut self,
        input: TrainInput,
        n_total: i32,
        rank: i32,
        adapter_ids: &[i64],
    ) -> Result<TrainOutput>;
    fn eval_step(&self, input: TrainInput) -> Result<EvalOutput>;
    fn save_checkpoint(&self, path: &str) -> Result<(u64, f64)>;
    fn save_checkpoint_with_generation(
        &self,
        path: &str,
        checkpoint_generation: Option<&str>,
    ) -> Result<(u64, f64)> {
        if checkpoint_generation.is_some() {
            bail!("this training session does not support coordinated checkpoint generations");
        }
        self.save_checkpoint(path)
    }
    fn load_checkpoint(&mut self, path: &str) -> Result<(u64, f64)>;
    #[doc(hidden)]
    fn load_checkpoint_in_place(&mut self, path: &str) -> Result<(u64, f64)>;
    fn export_adapter(&self, path: &str, adapter_id: Option<i64>) -> Result<usize>;
    /// Import one PEFT-style adapter as a new dynamic adapter. The fixed
    /// adapter remains reserved as ID 0 and is never overwritten by import.
    fn import_adapter(&mut self, path: &str) -> Result<i64>;
    fn status(&self) -> SessionStatus;
    fn get_metrics(&self) -> Vec<StepMetric>;

    /// Add a new LoRA adapter with independent rank/alpha/target layers/target modules.
    /// Returns adapter ID.
    fn add_lora(&mut self, req: AddLoRARequest) -> Result<i64>;

    /// Remove a LoRA adapter by ID.
    fn remove_lora(&mut self, adapter_id: i64) -> Result<bool>;

    /// List all active adapter IDs.
    fn list_lora(&self) -> Vec<i64>;
}

#[derive(Clone)]
struct QwenContextSpec {
    runtime_config: rustrain_qwen3_6::config::Qwen36RuntimeConfig,
    stage: PipelineStageLayout,
    init: InitLoRARequest,
    target_layers: Vec<usize>,
    target_modules: Vec<Qwen36LoraTargetModule>,
    base_tp_attention: bool,
    base_tp_mlp: bool,
    vocab_parallel: bool,
    data_parallel: bool,
    expert_parallel: bool,
    expert_start: usize,
    expert_count: usize,
    global_rank: usize,
    world_size: usize,
    tp_rank: usize,
    tp_size: usize,
    tp_color: usize,
    cp_rank: usize,
    cp_size: usize,
    cp_color: usize,
    ep_rank: usize,
    ep_size: usize,
    ep_color: usize,
    dp_rank: usize,
    dp_size: usize,
    dp_color: usize,
    pp_rank: usize,
    pp_size: usize,
    pp_color: usize,
}

struct PendingCheckpointLoad {
    transaction_id: String,
    source_path: String,
    ctx: rustrain_qwen3_6::kernel::CppTrainingContext,
    dynamic_lora_configs: std::collections::BTreeMap<i64, Qwen36LoraConfig>,
    dynamic_lora_optimizer_lrs: std::collections::BTreeMap<i64, f64>,
    state: SessionState,
    step: u64,
    last_loss: f64,
}

#[derive(Debug)]
pub struct AddLoRARequest {
    pub rank: i64,
    pub alpha: f64,
    pub target_layers: Vec<i64>,
    pub target_modules: String, // comma-separated, empty = all
    /// `None` inherits the training context learning rate.
    pub optimizer_lr: Option<f64>,
}

/// Qwen3.6 training session — wraps CppTrainingContext.
pub struct Qwen36Session {
    state: SessionState,
    model_path: Option<String>,
    config_toml: Option<String>,
    device: Device,
    compute_kind: Kind,
    ctx: Option<rustrain_qwen3_6::kernel::CppTrainingContext>,
    // Keep weights alive — C++ holds raw pointers to these tensors
    weights: Option<std::collections::BTreeMap<String, Tensor>>,
    dataset: Option<rustrain_qwen3_6::sft::SftDataset>,
    lora_rank: i64,
    lora_alpha: f64,
    lora_target_layers: Vec<usize>,
    lora_target_modules: Vec<Qwen36LoraTargetModule>,
    dynamic_lora_configs: std::collections::BTreeMap<i64, Qwen36LoraConfig>,
    dynamic_lora_optimizer_lrs: std::collections::BTreeMap<i64, f64>,
    lr: f64,
    metrics: Option<Arc<FileMetricsSink>>,
    last_loss: f64,
    step: u64,
    context_spec: Option<QwenContextSpec>,
    pending_checkpoint_load: Option<PendingCheckpointLoad>,
    // NCCL marker for EP (comm stored in C++ TrainingContext)
    _nccl_ep: bool,
}

// SAFETY: CppTrainingContext holds a raw pointer to C++ TrainingContext.
// The C++ context is only accessed from the thread that owns Qwen36Session.
// The Mutex in SessionManager ensures single-threaded access.
unsafe impl Send for Qwen36Session {}

impl Drop for Qwen36Session {
    fn drop(&mut self) {
        // Both native contexts borrow the frozen tensors through raw pointers.
        // Destroy every context explicitly before Rust releases those tensors.
        self.pending_checkpoint_load = None;
        self.ctx = None;
        self.weights = None;
    }
}

fn create_qwen_context(
    weights: &std::collections::BTreeMap<String, Tensor>,
    compute_kind: Kind,
    spec: &QwenContextSpec,
    synchronize_parameters: bool,
) -> Result<rustrain_qwen3_6::kernel::CppTrainingContext> {
    let lora_scaling = spec.init.alpha / spec.init.rank as f64;
    let ctx = rustrain_qwen3_6::kernel::CppTrainingContext::new_for_stage(
        weights,
        &spec.runtime_config,
        &spec.stage,
        compute_kind,
        spec.init.lr,
        spec.init.beta1,
        spec.init.beta2,
        spec.init.eps,
        lora_scaling,
        spec.init.rank,
        spec.base_tp_attention,
        spec.base_tp_mlp,
        spec.vocab_parallel,
        spec.data_parallel,
        spec.expert_parallel,
        &spec.target_layers,
        &spec.target_modules,
        spec.expert_start,
        spec.expert_count,
    )?;
    if spec.world_size > 1 {
        if synchronize_parameters {
            ctx.init_parallel_nccl(
                spec.global_rank,
                spec.world_size,
                spec.tp_rank,
                spec.tp_size,
                spec.tp_color,
                spec.cp_rank,
                spec.cp_size,
                spec.cp_color,
                spec.ep_rank,
                spec.ep_size,
                spec.ep_color,
                spec.dp_rank,
                spec.dp_size,
                spec.dp_color,
                spec.pp_rank,
                spec.pp_size,
                spec.pp_color,
            )?;
        } else {
            ctx.attach_parallel_nccl_no_sync(
                spec.global_rank,
                spec.world_size,
                spec.tp_rank,
                spec.tp_size,
                spec.tp_color,
                spec.cp_rank,
                spec.cp_size,
                spec.cp_color,
                spec.ep_rank,
                spec.ep_size,
                spec.ep_color,
                spec.dp_rank,
                spec.dp_size,
                spec.dp_color,
                spec.pp_rank,
                spec.pp_size,
                spec.pp_color,
            )?;
        }
    }
    Ok(ctx)
}

impl Qwen36Session {
    pub fn new(device: Device, compute_kind: Kind, metrics_path: PathBuf) -> Self {
        Self {
            state: SessionState::Unloaded,
            model_path: None,
            config_toml: None,
            device,
            compute_kind,
            ctx: None,
            weights: None,
            dataset: None,
            lora_rank: 0,
            lora_alpha: 0.0,
            lora_target_layers: Vec::new(),
            lora_target_modules: Vec::new(),
            dynamic_lora_configs: std::collections::BTreeMap::new(),
            dynamic_lora_optimizer_lrs: std::collections::BTreeMap::new(),
            lr: 1e-4,
            metrics: Some(Arc::new(FileMetricsSink::new(metrics_path))),
            last_loss: 0.0,
            step: 0,
            context_spec: None,
            pending_checkpoint_load: None,
            _nccl_ep: false,
        }
    }

    /// Get the device this session is bound to.
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn train_step_host_i64(
        &mut self,
        input_ids: &[i64],
        target_mask: &[i64],
        attention_mask: &[i64],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<TrainOutput> {
        let loss = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA not initialized"))?
            .train_step_host_i64(input_ids, target_mask, attention_mask, batch_size, seq_len)?;
        self.finish_train_step(loss, true)
    }

    pub fn train_multi_lora_host_i64(
        &mut self,
        input_ids: &[i64],
        target_mask: &[i64],
        attention_mask: &[i64],
        batch_size: usize,
        seq_len: usize,
        n_total: i32,
        lora_rank: i32,
        adapter_ids: &[i64],
    ) -> Result<TrainOutput> {
        let loss = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA not initialized"))?
            .train_multi_lora_host_i64(
                input_ids,
                target_mask,
                attention_mask,
                batch_size,
                seq_len,
                n_total,
                lora_rank,
                adapter_ids,
            )?;
        self.finish_train_step(loss, false)
    }

    pub fn train_multi_lora_host_i64_report(
        &mut self,
        input_ids: &[i64],
        target_mask: &[i64],
        attention_mask: &[i64],
        batch_size: usize,
        seq_len: usize,
        n_total: i32,
        lora_rank: i32,
        adapter_ids: &[i64],
    ) -> Result<MultiLoraTrainOutput> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA not initialized"))?;
        let report = ctx.train_multi_lora_host_i64_report(
            input_ids,
            target_mask,
            attention_mask,
            batch_size,
            seq_len,
            n_total,
            lora_rank,
            adapter_ids,
        )?;
        let loss = report.aggregate_loss;
        let output = self.finish_train_step(loss, false)?;
        Ok(MultiLoraTrainOutput {
            loss: output.loss,
            adapter_losses: report.adapter_losses,
            step: output.step,
        })
    }

    pub fn eval_step_host_i64(
        &self,
        input_ids: &[i64],
        target_mask: &[i64],
        attention_mask: &[i64],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<EvalOutput> {
        let loss = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA not initialized"))?
            .eval_step_host_i64(input_ids, target_mask, attention_mask, batch_size, seq_len)?;
        Ok(EvalOutput { loss })
    }

    fn finish_train_step(&mut self, loss: f64, record_metric: bool) -> Result<TrainOutput> {
        self.step = self
            .step
            .checked_add(1)
            .context("training step counter overflow")?;
        self.last_loss = loss;
        self.state = SessionState::Training { step: self.step };
        if record_metric {
            if let Some(ref metrics) = self.metrics {
                let mem_gb = rustrain_train::metrics::gpu_memory_allocated_mb()
                    .map(|m| m / 1024.0)
                    .unwrap_or(0.0);
                metrics.record_step(StepMetric {
                    step: self.step,
                    loss,
                    lr: self.lr,
                    mem_gb,
                    timestamp_unix: chrono::Utc::now().timestamp(),
                });
            }
        }
        Ok(TrainOutput {
            loss,
            step: self.step,
        })
    }

    pub fn prepare_checkpoint_load(
        &mut self,
        path: &str,
        transaction_id: &str,
    ) -> Result<(u64, f64)> {
        if transaction_id.is_empty()
            || !transaction_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            bail!("checkpoint transaction ID must be non-empty and path-safe");
        }
        if let Some(pending) = &self.pending_checkpoint_load {
            if pending.transaction_id == transaction_id {
                if pending.source_path != path {
                    bail!(
                        "checkpoint transaction {transaction_id} is already prepared from {}",
                        pending.source_path
                    );
                }
                return Ok((pending.step, pending.last_loss));
            }
            bail!(
                "checkpoint transaction {} is already prepared",
                pending.transaction_id
            );
        }
        let spec = self
            .context_spec
            .as_ref()
            .context("LoRA context specification is unavailable")?
            .clone();
        let weights = self
            .weights
            .as_ref()
            .context("base model weights are unavailable")?;
        let shadow_ctx = create_qwen_context(weights, self.compute_kind, &spec, false)?;
        let mut candidate = Self {
            state: self.state.clone(),
            model_path: self.model_path.clone(),
            config_toml: self.config_toml.clone(),
            device: self.device,
            compute_kind: self.compute_kind,
            ctx: Some(shadow_ctx),
            weights: None,
            dataset: None,
            lora_rank: self.lora_rank,
            lora_alpha: self.lora_alpha,
            lora_target_layers: self.lora_target_layers.clone(),
            lora_target_modules: self.lora_target_modules.clone(),
            dynamic_lora_configs: std::collections::BTreeMap::new(),
            dynamic_lora_optimizer_lrs: std::collections::BTreeMap::new(),
            lr: self.lr,
            metrics: None,
            last_loss: self.last_loss,
            step: self.step,
            context_spec: Some(spec),
            pending_checkpoint_load: None,
            _nccl_ep: self._nccl_ep,
        };
        let (step, loss) =
            <Self as TrainingSession>::load_checkpoint_in_place(&mut candidate, path)?;
        self.pending_checkpoint_load = Some(PendingCheckpointLoad {
            transaction_id: transaction_id.to_string(),
            source_path: path.to_string(),
            ctx: candidate
                .ctx
                .take()
                .context("checkpoint candidate lost its native context")?,
            dynamic_lora_configs: std::mem::take(&mut candidate.dynamic_lora_configs),
            dynamic_lora_optimizer_lrs: std::mem::take(&mut candidate.dynamic_lora_optimizer_lrs),
            state: candidate.state.clone(),
            step,
            last_loss: loss,
        });
        Ok((step, loss))
    }

    pub fn commit_checkpoint_load(&mut self, transaction_id: &str) -> Result<(u64, f64)> {
        let pending = self
            .pending_checkpoint_load
            .as_ref()
            .with_context(|| format!("checkpoint transaction {transaction_id} is not prepared"))?;
        if pending.transaction_id != transaction_id {
            bail!(
                "checkpoint transaction {} is prepared, not {transaction_id}",
                pending.transaction_id
            );
        }
        let pending = self
            .pending_checkpoint_load
            .take()
            .context("validated checkpoint transaction disappeared")?;
        self.ctx = Some(pending.ctx);
        self.dynamic_lora_configs = pending.dynamic_lora_configs;
        self.dynamic_lora_optimizer_lrs = pending.dynamic_lora_optimizer_lrs;
        self.state = pending.state;
        self.step = pending.step;
        self.last_loss = pending.last_loss;
        Ok((self.step, self.last_loss))
    }

    pub fn abort_checkpoint_load(&mut self, transaction_id: &str) -> Result<()> {
        match self.pending_checkpoint_load.as_ref() {
            Some(pending) if pending.transaction_id != transaction_id => bail!(
                "checkpoint transaction {} is prepared, not {transaction_id}",
                pending.transaction_id
            ),
            Some(_) => {
                self.pending_checkpoint_load = None;
                Ok(())
            }
            None => Ok(()),
        }
    }

    fn load_checkpoint_transactional(&mut self, path: &str) -> Result<(u64, f64)> {
        let transaction_id = format!(
            "direct-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        self.prepare_checkpoint_load(path, &transaction_id)?;
        self.commit_checkpoint_load(&transaction_id)
    }

    pub fn export_distributed_adapter(
        &self,
        path: &str,
        adapter_id: Option<i64>,
        generation: &str,
    ) -> Result<usize> {
        let parallel = checkpoint::ParallelCheckpointManifest::from_env()?;
        if parallel.world_size <= 1 {
            return <Self as TrainingSession>::export_adapter(self, path, adapter_id);
        }
        if parallel.pipeline_model_parallel_size > 1 {
            bail!(
                "pipeline-parallel adapter export requires cross-stage tensor-name remapping; checkpoint save/restore remains supported"
            );
        }
        let final_path = std::path::Path::new(path);
        checkpoint::export_distributed_adapter_checkpoint(
            final_path,
            generation,
            &parallel,
            adapter_id,
            |staging| {
                <Self as TrainingSession>::save_checkpoint_with_generation(
                    self,
                    staging
                        .to_str()
                        .context("distributed export staging path is not UTF-8")?,
                    Some(generation),
                )
                .map(|_| ())
            },
        )
    }
}

impl TrainingSession for Qwen36Session {
    fn load_model(&mut self, req: SessLoadModelRequest) -> Result<()> {
        let model_path = std::path::Path::new(&req.model_path);
        if !model_path.exists() {
            return Err(anyhow!("model path not found: {}", req.model_path));
        }
        if !rustrain_qwen3_6::kernel::kernels_available() {
            return Err(anyhow!("C++ kernels (libqwen36_kernels.so) not found"));
        }
        self.model_path = Some(req.model_path.clone());
        self.config_toml = Some(req.config_toml);
        self.state = SessionState::Loaded {
            model_path: req.model_path,
        };
        tracing::info!("model loaded");
        Ok(())
    }

    fn load_dataset(&mut self, req: SessLoadDatasetRequest) -> Result<usize> {
        if self.model_path.is_none() {
            return Err(anyhow!("model not loaded"));
        }
        let model_path = std::path::Path::new(self.model_path.as_ref().unwrap());
        let tokenizer_path = model_path.join("tokenizer.json");
        let dataset = rustrain_qwen3_6::sft::SftDataset::from_jsonl(
            std::path::Path::new(&req.jsonl_path),
            &tokenizer_path,
            req.seq_len,
        )?;
        let n = dataset.len();
        self.dataset = Some(dataset);
        tracing::info!(examples = n, "dataset loaded");
        Ok(n)
    }

    fn init_lora(&mut self, req: InitLoRARequest) -> Result<usize> {
        let model_path = self
            .model_path
            .as_ref()
            .ok_or_else(|| anyhow!("model not loaded"))?;

        // Load runtime config from model's config.json (no need to parse full TOML)
        let model_path_obj = std::path::Path::new(model_path);
        let runtime_config = rustrain_qwen3_6::config::read_qwen36_runtime_config(model_path_obj)?;

        let resolved = resolve_qwen_topology(&runtime_config)?;
        let ResolvedQwenTopology {
            ref topology,
            global_rank,
            world_size,
            local_rank,
            tp_rank,
            cp_rank,
            ep_rank: expert_rank,
            dp_rank,
            pp_rank,
            tp_size,
            cp_size,
            ep_size,
            dp_size,
            pp_size,
        } = resolved;
        let stage = PipelineStageLayout::new(runtime_config.num_hidden_layers, pp_rank, pp_size)?;
        if pp_size > 1 && runtime_config.mtp_num_hidden_layers > 0 {
            bail!("pipeline-parallel Qwen server does not yet support MTP layers");
        }

        // Resolve topology before disk IO so every worker reads only its
        // frozen stage ownership set.
        let wp = &runtime_config.weight_prefix;
        let mut needed =
            rustrain_qwen3_6::pipeline::stage_text_needed_weights(&runtime_config, &stage);
        // MTP weights
        if runtime_config.mtp_num_hidden_layers > 0 {
            for i in 0..runtime_config.mtp_num_hidden_layers {
                let p = format!("{wp}mtp.layers.{i}");
                needed.insert(format!("{p}.input_layernorm.weight"));
                needed.insert(format!("{p}.post_attention_layernorm.weight"));
                for w in &["q_proj", "q_norm", "k_proj", "k_norm", "v_proj", "o_proj"] {
                    needed.insert(format!("{p}.self_attn.{w}.weight"));
                }
                if runtime_config.is_moe {
                    needed.insert(format!("{p}.mlp.gate.weight"));
                    needed.insert(format!("{p}.mlp.shared_expert_gate.weight"));
                    needed.insert(format!("{p}.mlp.shared_expert.gate_proj.weight"));
                    needed.insert(format!("{p}.mlp.shared_expert.up_proj.weight"));
                    needed.insert(format!("{p}.mlp.shared_expert.down_proj.weight"));
                    needed.insert(format!("{p}.mlp.experts.gate_up_proj"));
                    needed.insert(format!("{p}.mlp.experts.down_proj"));
                } else {
                    needed.insert(format!("{p}.mlp.gate_proj.weight"));
                    needed.insert(format!("{p}.mlp.up_proj.weight"));
                    needed.insert(format!("{p}.mlp.down_proj.weight"));
                }
            }
            needed.insert(format!("{wp}mtp.fc.weight"));
            needed.insert(format!("{wp}mtp.pre_fc_norm_embedding.weight"));
            needed.insert(format!("{wp}mtp.pre_fc_norm_hidden.weight"));
            needed.insert(format!("{wp}mtp.norm.weight"));
        }

        // The safetensors keys already include the prefix (e.g. "model.language_model.layers.0...")
        // So we should NOT add the prefix again. Use the keys as-is.
        let raw_weights = rustrain_checkpoint::safetensors::read_safetensors_dir_filtered(
            model_path_obj,
            &needed,
        )?;
        let is_ep = runtime_config.is_moe && ep_size > 1;
        let is_data_parallel = dp_size > 1;
        let ep_a2a = std::env::var("QWEN36_EP_A2A")
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false);
        let ep_a2a_sharded = std::env::var("QWEN36_EP_A2A_SHARDED")
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false);
        validate_qwen_parallel_features(
            runtime_config.is_moe,
            tp_size,
            ep_size,
            ep_a2a,
            ep_a2a_sharded,
        )?;
        unsafe {
            std::env::set_var("TP_SIZE", tp_size.to_string());
            std::env::set_var("CP_SIZE", cp_size.to_string());
            std::env::set_var("EP_SIZE", ep_size.to_string());
            std::env::set_var("DP_SIZE", dp_size.to_string());
            std::env::set_var("PP_SIZE", pp_size.to_string());
            std::env::set_var("RUSTRAIN_TP_RANK", tp_rank.to_string());
            std::env::set_var("RUSTRAIN_CP_RANK", cp_rank.to_string());
            std::env::set_var("RUSTRAIN_EP_RANK", expert_rank.to_string());
            std::env::set_var("RUSTRAIN_DP_RANK", dp_rank.to_string());
            std::env::set_var("RUSTRAIN_PP_RANK", pp_rank.to_string());
            std::env::set_var(
                "RUSTRAIN_DATA_PARALLEL",
                if is_data_parallel { "1" } else { "0" },
            );
        }
        let base_tp_attention = tp_size > 1;
        let base_tp_mlp = tp_size > 1;
        let vocab_parallel = tp_size > 1;
        if vocab_parallel
            && (runtime_config.vocab_size <= 0 || runtime_config.vocab_size % tp_size as i64 != 0)
        {
            return Err(anyhow!(
                "vocab_size={} must be divisible by TP_SIZE={tp_size} for vocabulary parallelism",
                runtime_config.vocab_size
            ));
        }
        if base_tp_attention {
            if runtime_config.mtp_num_hidden_layers > 0 {
                return Err(anyhow!(
                    "frozen base TP currently requires MTP to be disabled"
                ));
            }
            if runtime_config
                .layer_types
                .iter()
                .any(|layer| *layer == rustrain_qwen3_6::config::LayerType::LinearAttention)
                && (runtime_config.linear_num_key_heads <= 0
                    || runtime_config.linear_num_value_heads <= 0
                    || runtime_config.linear_num_value_heads % runtime_config.linear_num_key_heads
                        != 0
                    || runtime_config.linear_num_key_heads % tp_size as i64 != 0
                    || runtime_config.linear_num_value_heads % tp_size as i64 != 0
                    || runtime_config.linear_key_head_dim != 128
                    || runtime_config.linear_value_head_dim != 128
                    || runtime_config.linear_conv_kernel_dim <= 0)
            {
                return Err(anyhow!(
                    "linear-attention TP requires k/v heads divisible by TP_SIZE with preserved value-head groups, 128-wide key/value heads, and a positive conv kernel: k_heads={}, v_heads={}, key_dim={}, value_dim={}, conv_kernel={}, tp={tp_size}",
                    runtime_config.linear_num_key_heads,
                    runtime_config.linear_num_value_heads,
                    runtime_config.linear_key_head_dim,
                    runtime_config.linear_value_head_dim,
                    runtime_config.linear_conv_kernel_dim,
                ));
            }
            if runtime_config.num_attention_heads <= 0
                || runtime_config.num_attention_heads % tp_size as i64 != 0
                || runtime_config.num_key_value_heads <= 0
                || runtime_config.num_key_value_heads % tp_size as i64 != 0
                || runtime_config.num_attention_heads % runtime_config.num_key_value_heads != 0
            {
                return Err(anyhow!(
                    "full-attention heads (q={}, kv={}) must preserve GQA groups and be divisible by TP_SIZE={tp_size}",
                    runtime_config.num_attention_heads,
                    runtime_config.num_key_value_heads
                ));
            }
            let rotary_dim =
                (runtime_config.head_dim as f64 * runtime_config.partial_rotary_factor) as i64;
            if runtime_config.head_dim <= 0
                || rotary_dim < 0
                || rotary_dim > runtime_config.head_dim
                || rotary_dim % 2 != 0
            {
                return Err(anyhow!(
                    "full-attention head_dim={} and partial_rotary_factor={} produce invalid rotary_dim={rotary_dim}",
                    runtime_config.head_dim,
                    runtime_config.partial_rotary_factor
                ));
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
            validate_tp_intermediate_sizes(tp_size, &intermediates)?;
        }

        // Compute expert shard
        let (expert_start, expert_count) = if is_ep {
            assert!(
                runtime_config.num_experts % ep_size == 0,
                "num_experts {} not divisible by ep_size {}",
                runtime_config.num_experts,
                ep_size
            );
            let epr = runtime_config.num_experts / ep_size;
            (expert_rank * epr, epr)
        } else {
            (0, runtime_config.num_experts)
        };

        // Set CUDA device for any torchrun worker. Dense Qwen workers use
        // replicated weights and NCCL gradient all-reduce (LoRA-only DP).
        if is_ep || is_data_parallel || tp_size > 1 {
            self.device = tch::Device::Cuda(local_rank);
        }

        // Apply orthogonal EP and TP shards on CPU before moving weights to CUDA.
        let num_experts = runtime_config.num_experts as i64;
        let mut weights: std::collections::BTreeMap<String, Tensor> =
            std::collections::BTreeMap::new();
        for (name, tensor) in raw_weights {
            let needs_expert_narrow = is_ep
                && (name.contains(".mlp.experts.gate_up_proj")
                    || name.contains(".mlp.experts.down_proj"));
            let expert_shard = if needs_expert_narrow && tensor.size()[0] == num_experts {
                Some(
                    tensor
                        .narrow(0, expert_start as i64, expert_count as i64)
                        .contiguous(),
                )
            } else {
                None
            };
            let expert_or_full = expert_shard.as_ref().unwrap_or(&tensor);
            let moe_tp_shard = if runtime_config.is_moe && base_tp_mlp {
                rustrain_qwen3_6::kernel::shard_moe_mlp_weight_for_tp(
                    &name,
                    expert_or_full,
                    tp_size,
                    tp_rank,
                )?
            } else {
                None
            };
            let vocab_shard = if expert_shard.is_none() && moe_tp_shard.is_none() && vocab_parallel
            {
                rustrain_qwen3_6::kernel::shard_vocab_weight_for_tp(
                    &name,
                    &tensor,
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
                    rustrain_qwen3_6::kernel::shard_full_attention_weight_for_tp(
                        &name, &tensor, tp_size, tp_rank,
                    )?;
                let attention_shard = if full_attention_shard.is_some() {
                    full_attention_shard
                } else {
                    rustrain_qwen3_6::kernel::shard_linear_attention_weight_for_tp(
                        &name,
                        &tensor,
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
                    rustrain_qwen3_6::kernel::shard_dense_mlp_weight_for_tp(
                        &name, &tensor, tp_size, tp_rank,
                    )?
                }
            } else {
                None
            };
            let processed = local_shard
                .as_ref()
                .unwrap_or(&tensor)
                .to_device(self.device)
                .to_kind(self.compute_kind);
            weights.insert(name, processed);
        }

        // Create C++ training context
        // If target_layers is empty, it means "all layers"
        let all_layers: Vec<usize> = if req.target_layers.is_empty() {
            (0..runtime_config.num_hidden_layers).collect()
        } else {
            req.target_layers.clone()
        };
        let target_modules = req
            .target_modules
            .iter()
            .map(|name| rustrain_qwen3_6::lora::Qwen36LoraTargetModule::parse(name))
            .collect::<Result<Vec<_>>>()?;
        rustrain_qwen3_6::lora::validate_lora_targets(
            &runtime_config,
            &Qwen36LoraConfig {
                rank: req.rank,
                alpha: req.alpha,
                target_layers: all_layers.clone(),
                target_modules: target_modules.clone(),
            },
        )?;
        let (tp_color, cp_color, ep_color, dp_color, pp_color) = if world_size > 1 {
            let tp_color = *topology
                .tensor_group(global_rank)?
                .iter()
                .min()
                .ok_or_else(|| anyhow!("empty TP process group"))?;
            let cp_color = *topology
                .context_group(global_rank)?
                .iter()
                .min()
                .ok_or_else(|| anyhow!("empty CP process group"))?;
            let ep_color = *topology
                .expert_group(global_rank)?
                .iter()
                .min()
                .ok_or_else(|| anyhow!("empty EP process group"))?;
            let dp_color = *topology
                .data_group(global_rank)?
                .iter()
                .min()
                .ok_or_else(|| anyhow!("empty DP process group"))?;
            let pp_color = *topology
                .pipeline_group(global_rank)?
                .iter()
                .min()
                .ok_or_else(|| anyhow!("empty PP process group"))?;
            (tp_color, cp_color, ep_color, dp_color, pp_color)
        } else {
            (0, 0, 0, 0, 0)
        };
        let context_spec = QwenContextSpec {
            runtime_config: runtime_config.clone(),
            stage,
            init: req.clone(),
            target_layers: all_layers.clone(),
            target_modules: target_modules.clone(),
            base_tp_attention,
            base_tp_mlp,
            vocab_parallel,
            data_parallel: is_data_parallel,
            expert_parallel: is_ep,
            expert_start,
            expert_count,
            global_rank,
            world_size,
            tp_rank,
            tp_size,
            tp_color,
            cp_rank,
            cp_size,
            cp_color,
            ep_rank: expert_rank,
            ep_size,
            ep_color,
            dp_rank,
            dp_size,
            dp_color,
            pp_rank,
            pp_size,
            pp_color,
        };
        let ctx = create_qwen_context(&weights, self.compute_kind, &context_spec, true)?;
        let nccl_ep = world_size > 1;
        if nccl_ep {
            tracing::info!(
                global_rank,
                world_size,
                data_parallel = is_data_parallel,
                expert_parallel = is_ep,
                tp_size,
                "NCCL communicator created in C++ for Qwen parallel training"
            );
        }

        let count = ctx.lora_count() as usize;
        self.ctx = Some(ctx);
        self.weights = Some(weights); // Keep alive — C++ holds raw pointers
        self.context_spec = Some(context_spec);
        self.pending_checkpoint_load = None;
        self._nccl_ep = nccl_ep;
        self.lora_rank = req.rank;
        self.lora_alpha = req.alpha;
        self.lora_target_layers = all_layers;
        self.lora_target_modules = target_modules;
        self.lr = req.lr;
        self.state = SessionState::Ready {
            model_path: model_path.clone(),
        };
        tracing::info!(lora_params = count, "LoRA initialized");
        Ok(count)
    }

    fn train_step(&mut self, input: TrainInput) -> Result<TrainOutput> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA not initialized"))?;

        let loss = ctx.train_step(&input.input_ids, &input.target_mask, &input.attention_mask)?;
        self.finish_train_step(loss, true)
    }

    fn train_multi_lora(
        &mut self,
        input: TrainInput,
        n_total: i32,
        rank: i32,
        adapter_ids: &[i64],
    ) -> Result<TrainOutput> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA not initialized"))?;

        if n_total <= 0 {
            return Err(anyhow!("n_total must be positive, got {n_total}"));
        }
        if !adapter_ids.is_empty() {
            if adapter_ids.len() != n_total as usize {
                return Err(anyhow!(
                    "selected adapter count {} does not match n_total={n_total}",
                    adapter_ids.len()
                ));
            }
            if adapter_ids.iter().any(|id| *id <= 0) {
                return Err(anyhow!("selected adapter IDs must be positive"));
            }
        }
        let loss = if adapter_ids.is_empty() {
            ctx.train_multi_lora(
                &input.input_ids,
                &input.target_mask,
                &input.attention_mask,
                n_total,
                rank,
            )?
        } else {
            ctx.train_multi_lora_selected(
                &input.input_ids,
                &input.target_mask,
                &input.attention_mask,
                adapter_ids,
                rank,
            )?
        };
        self.finish_train_step(loss, false)
    }

    fn eval_step(&self, input: TrainInput) -> Result<EvalOutput> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA not initialized"))?;
        let loss = ctx.eval_step(&input.input_ids, &input.target_mask, &input.attention_mask)?;
        Ok(EvalOutput { loss })
    }

    fn save_checkpoint(&self, path: &str) -> Result<(u64, f64)> {
        self.save_checkpoint_with_generation(path, None)
    }

    fn save_checkpoint_with_generation(
        &self,
        path: &str,
        checkpoint_generation: Option<&str>,
    ) -> Result<(u64, f64)> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA not initialized"))?;

        let model_path = self
            .model_path
            .as_ref()
            .ok_or_else(|| anyhow!("model path unavailable for checkpoint"))?;
        let runtime_config =
            rustrain_qwen3_6::config::read_qwen36_runtime_config(std::path::Path::new(model_path))?;
        let parallel = checkpoint::ParallelCheckpointManifest::from_env()?;
        let stage = self
            .context_spec
            .as_ref()
            .context("LoRA context specification is unavailable")?
            .stage
            .clone();

        let fixed_config = Qwen36LoraConfig {
            rank: self.lora_rank,
            alpha: self.lora_alpha,
            target_layers: self.lora_target_layers.clone(),
            target_modules: self.lora_target_modules.clone(),
        };
        let global_fixed_slots =
            rustrain_qwen3_6::lora::native_lora_slots(&runtime_config, &fixed_config);
        let fixed_slots = stage_lora_slots(&global_fixed_slots, &stage);
        let lora_count = ctx.lora_count() as usize;
        if fixed_slots.len() != lora_count {
            bail!(
                "fixed LoRA registry count {} does not match native slot count {lora_count}",
                fixed_slots.len()
            );
        }
        let (all_adam_m, all_adam_v) = ctx.export_optimizer_state()?;
        let expected_optimizer_count = lora_count.saturating_mul(2);
        if all_adam_m.len() != expected_optimizer_count
            || all_adam_v.len() != expected_optimizer_count
        {
            bail!(
                "fixed optimizer state count mismatch: m={}, v={}, expected={expected_optimizer_count}",
                all_adam_m.len(),
                all_adam_v.len()
            );
        }

        // Distributed manifests compact inactive stage-local slots while their
        // identities retain global layer and slot indices.
        let saved_fixed_slots = fixed_slots
            .iter()
            .filter(|slot| parallel.world_size <= 1 || slot.active)
            .collect::<Vec<_>>();
        let saved_fixed_count = saved_fixed_slots.len();
        let mut lora_a = Vec::with_capacity(saved_fixed_count);
        let mut lora_b = Vec::with_capacity(saved_fixed_count);
        let mut adam_m = Vec::with_capacity(saved_fixed_count.saturating_mul(2));
        let mut adam_v = Vec::with_capacity(saved_fixed_count.saturating_mul(2));
        let mut fixed_shard_layouts = Vec::with_capacity(saved_fixed_count);
        let mut fixed_slot_identities = Vec::with_capacity(saved_fixed_count);
        for slot in saved_fixed_slots {
            lora_a.push(ctx.get_lora_a(slot.local_index as i64).with_context(|| {
                format!(
                    "fixed LoRA A is missing for native slot {}",
                    slot.local_index
                )
            })?);
            lora_b.push(ctx.get_lora_b(slot.local_index as i64).with_context(|| {
                format!(
                    "fixed LoRA B is missing for native slot {}",
                    slot.local_index
                )
            })?);
            let optimizer_index = slot.local_index.saturating_mul(2);
            adam_m.push(all_adam_m[optimizer_index].shallow_clone());
            adam_m.push(all_adam_m[optimizer_index + 1].shallow_clone());
            adam_v.push(all_adam_v[optimizer_index].shallow_clone());
            adam_v.push(all_adam_v[optimizer_index + 1].shallow_clone());
            fixed_shard_layouts.push(checkpoint::lora_tp_shard_layout(
                slot.module,
                &runtime_config,
            ));
            fixed_slot_identities.push(checkpoint::LoraSlotIdentity {
                index: slot.global_index,
                layer: slot.layer,
                module: slot.module.cpp_name().to_string(),
            });
        }

        let mut dynamic_adapters = Vec::new();
        if !self.dynamic_lora_configs.is_empty() {
            for (&adapter_id, lora_config) in &self.dynamic_lora_configs {
                let global_slots =
                    rustrain_qwen3_6::lora::native_lora_slots(&runtime_config, lora_config);
                let slots = stage_lora_slots(&global_slots, &stage);
                let shard_layouts = slots
                    .iter()
                    .filter(|slot| slot.active)
                    .map(|slot| checkpoint::lora_tp_shard_layout(slot.module, &runtime_config))
                    .collect::<Vec<_>>();
                let mut dynamic_a = Vec::new();
                let mut dynamic_b = Vec::new();
                let mut dynamic_m = Vec::new();
                let mut dynamic_v = Vec::new();
                for slot in slots.iter().filter(|slot| slot.active) {
                    let module = slot.module.cpp_name();
                    dynamic_a.push(
                        ctx.get_adapter_lora_tensor(
                            adapter_id,
                            slot.layer as i64,
                            module,
                            false,
                        )
                        .with_context(|| {
                            format!(
                                "dynamic LoRA A is missing: adapter={adapter_id} layer={} module={module}",
                                slot.layer
                            )
                        })?,
                    );
                    dynamic_b.push(
                        ctx.get_adapter_lora_tensor(
                            adapter_id,
                            slot.layer as i64,
                            module,
                            true,
                        )
                        .with_context(|| {
                            format!(
                                "dynamic LoRA B is missing: adapter={adapter_id} layer={} module={module}",
                                slot.layer
                            )
                        })?,
                    );
                    // Keep one m/v entry for each A and B tensor, in slot order.
                    dynamic_m.push(
                        ctx.get_adapter_optimizer_tensor(
                            adapter_id,
                            slot.layer as i64,
                            module,
                            false,
                            false,
                        )
                        .with_context(|| {
                            format!(
                                "dynamic LoRA m_a is missing: adapter={adapter_id} layer={} module={module}",
                                slot.layer
                            )
                        })?,
                    );
                    dynamic_m.push(
                        ctx.get_adapter_optimizer_tensor(
                            adapter_id,
                            slot.layer as i64,
                            module,
                            true,
                            false,
                        )
                        .with_context(|| {
                            format!(
                                "dynamic LoRA m_b is missing: adapter={adapter_id} layer={} module={module}",
                                slot.layer
                            )
                        })?,
                    );
                    dynamic_v.push(
                        ctx.get_adapter_optimizer_tensor(
                            adapter_id,
                            slot.layer as i64,
                            module,
                            false,
                            true,
                        )
                        .with_context(|| {
                            format!(
                                "dynamic LoRA v_a is missing: adapter={adapter_id} layer={} module={module}",
                                slot.layer
                            )
                        })?,
                    );
                    dynamic_v.push(
                        ctx.get_adapter_optimizer_tensor(
                            adapter_id,
                            slot.layer as i64,
                            module,
                            true,
                            true,
                        )
                        .with_context(|| {
                            format!(
                                "dynamic LoRA v_b is missing: adapter={adapter_id} layer={} module={module}",
                                slot.layer
                            )
                        })?,
                    );
                }
                let optimizer_step = u64::try_from(ctx.get_adapter_step_count(adapter_id)?)
                    .context("native dynamic adapter optimizer step is negative")?;
                let optimizer_lr = self
                    .dynamic_lora_optimizer_lrs
                    .get(&adapter_id)
                    .copied()
                    .unwrap_or(self.lr);
                dynamic_adapters.push(checkpoint::DynamicAdapterCheckpoint {
                    manifest: checkpoint::DynamicAdapterManifest {
                        id: adapter_id,
                        rank: lora_config.rank,
                        alpha: lora_config.alpha,
                        optimizer_step,
                        optimizer_lr: Some(optimizer_lr),
                        target_layers: lora_config.target_layers.clone(),
                        target_modules: lora_config
                            .target_modules
                            .iter()
                            .map(|module| module.cpp_name().to_string())
                            .collect(),
                        shard_layouts,
                        slot_identities: slots
                            .iter()
                            .filter(|slot| slot.active)
                            .map(|slot| checkpoint::LoraSlotIdentity {
                                index: slot.global_index,
                                layer: slot.layer,
                                module: slot.module.cpp_name().to_string(),
                            })
                            .collect(),
                        parameter_count: dynamic_a.len(),
                        optimizer_count: dynamic_m.len(),
                    },
                    lora_a: dynamic_a,
                    lora_b: dynamic_b,
                    adam_m: dynamic_m,
                    adam_v: dynamic_v,
                });
            }
        }

        let fixed_optimizer_step = u64::try_from(ctx.get_step_count())
            .context("native fixed adapter optimizer step is negative")?;
        if stage.pipeline_size > 1 {
            let generation = checkpoint_generation
                .filter(|generation| !generation.is_empty())
                .context("pipeline-parallel checkpoint save requires a coordinated generation")?;
            let stage_union = checkpoint::StageUnionCheckpointMetadata {
                pipeline_stage: checkpoint::PipelineStageCheckpointManifest {
                    pipeline_rank: stage.pipeline_rank,
                    pipeline_size: stage.pipeline_size,
                    global_num_layers: stage.global_num_layers,
                    layer_start: stage.layer_range.start,
                    layer_end: stage.layer_range.end,
                },
                fixed_target_layers: self.lora_target_layers.clone(),
                fixed_target_modules: self
                    .lora_target_modules
                    .iter()
                    .map(|module| module.cpp_name().to_string())
                    .collect(),
            };
            checkpoint::save_stage_union_checkpoint_with_dynamic_and_fixed_step_for_topology_generation(
                std::path::Path::new(path),
                self.step,
                fixed_optimizer_step,
                self.last_loss,
                model_path,
                self.lora_rank,
                self.lora_alpha,
                &lora_a,
                &lora_b,
                &adam_m,
                &adam_v,
                &dynamic_adapters,
                &fixed_shard_layouts,
                &fixed_slot_identities,
                &parallel,
                generation,
                &stage_union,
            )?;
        } else {
            match checkpoint_generation {
                Some(generation) => {
                    checkpoint::save_checkpoint_with_dynamic_and_fixed_step_for_topology_generation(
                    std::path::Path::new(path),
                    self.step,
                    fixed_optimizer_step,
                    self.last_loss,
                    model_path,
                    self.lora_rank,
                    self.lora_alpha,
                    &lora_a,
                    &lora_b,
                    &adam_m,
                    &adam_v,
                    &dynamic_adapters,
                    &fixed_shard_layouts,
                    &fixed_slot_identities,
                    &parallel,
                    Some(generation),
                )?;
                }
                None => checkpoint::save_checkpoint_with_dynamic_and_fixed_step_for_topology(
                    std::path::Path::new(path),
                    self.step,
                    fixed_optimizer_step,
                    self.last_loss,
                    model_path,
                    self.lora_rank,
                    self.lora_alpha,
                    &lora_a,
                    &lora_b,
                    &adam_m,
                    &adam_v,
                    &dynamic_adapters,
                    &fixed_shard_layouts,
                    &fixed_slot_identities,
                    &parallel,
                )?,
            }
        }

        Ok((self.step, self.last_loss))
    }

    fn load_checkpoint(&mut self, path: &str) -> Result<(u64, f64)> {
        self.load_checkpoint_transactional(path)
    }

    fn load_checkpoint_in_place(&mut self, path: &str) -> Result<(u64, f64)> {
        let parallel = checkpoint::ParallelCheckpointManifest::from_env()?;
        let data = checkpoint::load_checkpoint_for_topology(std::path::Path::new(path), &parallel)?;
        let stage = self
            .context_spec
            .as_ref()
            .context("LoRA context specification is unavailable")?
            .stage
            .clone();
        let model_path = self
            .model_path
            .as_ref()
            .ok_or_else(|| anyhow!("model path unavailable for checkpoint restore"))?;
        let current_model_path = std::fs::canonicalize(model_path)
            .with_context(|| format!("canonicalize current base model path {model_path}"))?;
        let checkpoint_model_path =
            std::fs::canonicalize(&data.manifest.model_path).with_context(|| {
                format!(
                    "canonicalize checkpoint base model path {}",
                    data.manifest.model_path
                )
            })?;
        if checkpoint_model_path != current_model_path {
            bail!(
                "checkpoint base model {} does not match loaded model {}",
                checkpoint_model_path.display(),
                current_model_path.display()
            );
        }
        if data.manifest.lora_rank != self.lora_rank
            || data.manifest.lora_alpha.to_bits() != self.lora_alpha.to_bits()
        {
            bail!(
                "checkpoint fixed LoRA rank/alpha {}/{} does not match session {}/{}",
                data.manifest.lora_rank,
                data.manifest.lora_alpha,
                self.lora_rank,
                self.lora_alpha
            );
        }
        let runtime_config =
            rustrain_qwen3_6::config::read_qwen36_runtime_config(std::path::Path::new(model_path))?;
        let fixed_config = Qwen36LoraConfig {
            rank: self.lora_rank,
            alpha: self.lora_alpha,
            target_layers: self.lora_target_layers.clone(),
            target_modules: self.lora_target_modules.clone(),
        };
        if stage.pipeline_size > 1 {
            let expected_stage = checkpoint::PipelineStageCheckpointManifest {
                pipeline_rank: stage.pipeline_rank,
                pipeline_size: stage.pipeline_size,
                global_num_layers: stage.global_num_layers,
                layer_start: stage.layer_range.start,
                layer_end: stage.layer_range.end,
            };
            if data.manifest.pipeline_stage.as_ref() != Some(&expected_stage) {
                bail!(
                    "checkpoint pipeline stage metadata does not match the current runtime stage"
                );
            }
            let expected_modules = self
                .lora_target_modules
                .iter()
                .map(|module| module.cpp_name().to_string())
                .collect::<Vec<_>>();
            if data.manifest.fixed_target_layers != self.lora_target_layers
                || data.manifest.fixed_target_modules != expected_modules
            {
                bail!("checkpoint global fixed LoRA target signature does not match the session");
            }
        }
        let global_fixed_slots =
            rustrain_qwen3_6::lora::native_lora_slots(&runtime_config, &fixed_config);
        let fixed_slots = stage_lora_slots(&global_fixed_slots, &stage);
        let expected_fixed_layouts = fixed_slots
            .iter()
            .filter(|slot| slot.active)
            .map(|slot| checkpoint::lora_tp_shard_layout(slot.module, &runtime_config))
            .collect::<Vec<_>>();
        let expected_fixed_identities = fixed_slots
            .iter()
            .filter(|slot| slot.active)
            .map(|slot| checkpoint::LoraSlotIdentity {
                index: slot.global_index,
                layer: slot.layer,
                module: slot.module.cpp_name().to_string(),
            })
            .collect::<Vec<_>>();
        if parallel.world_size > 1
            || !data.manifest.fixed_shard_layouts.is_empty()
            || !data.manifest.fixed_slot_identities.is_empty()
        {
            checkpoint::validate_fixed_tp_resume(
                &data.manifest,
                &expected_fixed_layouts,
                &expected_fixed_identities,
            )?;
        }
        if parallel.world_size > 1
            || data.manifest.dynamic_adapters.iter().any(|adapter| {
                !adapter.shard_layouts.is_empty() || !adapter.slot_identities.is_empty()
            })
        {
            for dynamic in &data.dynamic_adapters {
                let target_modules = dynamic
                    .manifest
                    .target_modules
                    .iter()
                    .map(|name| Qwen36LoraTargetModule::parse(name))
                    .collect::<Result<Vec<_>>>()?;
                let config = Qwen36LoraConfig {
                    rank: dynamic.manifest.rank,
                    alpha: dynamic.manifest.alpha,
                    target_layers: dynamic.manifest.target_layers.clone(),
                    target_modules,
                };
                let global_expected_slots =
                    rustrain_qwen3_6::lora::native_lora_slots(&runtime_config, &config);
                let expected_slots = stage_lora_slots(&global_expected_slots, &stage);
                let expected_layouts = expected_slots
                    .iter()
                    .filter(|slot| slot.active)
                    .map(|slot| checkpoint::lora_tp_shard_layout(slot.module, &runtime_config))
                    .collect::<Vec<_>>();
                let expected_identities = expected_slots
                    .iter()
                    .filter(|slot| slot.active)
                    .map(|slot| checkpoint::LoraSlotIdentity {
                        index: slot.global_index,
                        layer: slot.layer,
                        module: slot.module.cpp_name().to_string(),
                    })
                    .collect::<Vec<_>>();
                checkpoint::validate_dynamic_tp_resume(
                    &data.manifest,
                    dynamic.manifest.id,
                    &dynamic.manifest.shard_layouts,
                    &expected_layouts,
                    &expected_identities,
                )?;
            }
        }
        validate_dynamic_adapter_manifests(
            data.dynamic_adapters
                .iter()
                .map(|dynamic| &dynamic.manifest),
        )?;
        for dynamic in &data.dynamic_adapters {
            let manifest = &dynamic.manifest;
            let target_modules = manifest
                .target_modules
                .iter()
                .map(|name| Qwen36LoraTargetModule::parse(name))
                .collect::<Result<Vec<_>>>()?;
            let config = Qwen36LoraConfig {
                rank: manifest.rank,
                alpha: manifest.alpha,
                target_layers: manifest.target_layers.clone(),
                target_modules,
            };
            rustrain_qwen3_6::lora::validate_lora_targets(&runtime_config, &config)?;
            let global_slots = rustrain_qwen3_6::lora::native_lora_slots(&runtime_config, &config);
            let active_slots = stage_lora_slots(&global_slots, &stage)
                .into_iter()
                .filter(|slot| slot.active)
                .count();
            let expected_optimizer_count = active_slots.saturating_mul(2);
            if dynamic.lora_a.len() != active_slots
                || dynamic.lora_b.len() != active_slots
                || dynamic.adam_m.len() != expected_optimizer_count
                || dynamic.adam_v.len() != expected_optimizer_count
                || manifest.parameter_count != active_slots
                || manifest.optimizer_count != expected_optimizer_count
            {
                bail!(
                    "dynamic adapter {} tensor count does not match its runtime slot signature",
                    manifest.id
                );
            }
        }
        if !data.dynamic_adapters.is_empty() {
            if !self.dynamic_lora_configs.is_empty() {
                bail!("cannot load dynamic LoRA checkpoint into a session with active adapters");
            }
            let ctx = self
                .ctx
                .as_ref()
                .ok_or_else(|| anyhow!("LoRA not initialized"))?;
            for dynamic in &data.dynamic_adapters {
                let target_modules = dynamic
                    .manifest
                    .target_modules
                    .iter()
                    .map(|name| Qwen36LoraTargetModule::parse(name))
                    .collect::<Result<Vec<_>>>()?;
                let lora_config = Qwen36LoraConfig {
                    rank: dynamic.manifest.rank,
                    alpha: dynamic.manifest.alpha,
                    target_layers: dynamic.manifest.target_layers.clone(),
                    target_modules,
                };
                rustrain_qwen3_6::lora::validate_lora_targets(&runtime_config, &lora_config)?;
                let layer_ids = lora_config
                    .target_layers
                    .iter()
                    .map(|&layer| layer as i64)
                    .collect::<Vec<_>>();
                let module_csv = lora_config
                    .target_modules
                    .iter()
                    .map(Qwen36LoraTargetModule::cpp_name)
                    .collect::<Vec<_>>()
                    .join(",");
                let optimizer_lr = dynamic.manifest.optimizer_lr.unwrap_or(self.lr);
                let allocated_id = match dynamic.manifest.optimizer_lr {
                    Some(saved_lr) if saved_lr.to_bits() != self.lr.to_bits() => ctx
                        .add_lora_for_restore_with_optimizer_lr(
                            lora_config.rank,
                            lora_config.alpha,
                            &layer_ids,
                            &module_csv,
                            saved_lr,
                        )?,
                    _ => ctx.add_lora_for_restore(
                        lora_config.rank,
                        lora_config.alpha,
                        &layer_ids,
                        &module_csv,
                    )?,
                };
                if allocated_id != dynamic.manifest.id {
                    if let Err(error) = ctx.set_adapter_id(allocated_id, dynamic.manifest.id) {
                        let _ = ctx.remove_lora(allocated_id);
                        return Err(error);
                    }
                }
                let adapter_id = dynamic.manifest.id;
                let load_result = (|| -> Result<()> {
                    let global_slots =
                        rustrain_qwen3_6::lora::native_lora_slots(&runtime_config, &lora_config);
                    let slots = stage_lora_slots(&global_slots, &stage);
                    let mut optimizer_index = 0usize;
                    for (slot_index, slot) in slots.iter().filter(|slot| slot.active).enumerate() {
                        let module = slot.module.cpp_name();
                        if slot_index >= dynamic.lora_a.len()
                            || slot_index >= dynamic.lora_b.len()
                            || optimizer_index + 1 >= dynamic.adam_m.len()
                            || optimizer_index + 1 >= dynamic.adam_v.len()
                        {
                            bail!(
                                "dynamic adapter {} tensor count mismatch",
                                dynamic.manifest.id
                            );
                        }
                        // A/B vectors are ordered exactly like active native slots.
                        ctx.set_adapter_lora_tensor(
                            adapter_id,
                            slot.layer as i64,
                            module,
                            false,
                            &dynamic.lora_a[slot_index],
                        )?;
                        ctx.set_adapter_lora_tensor(
                            adapter_id,
                            slot.layer as i64,
                            module,
                            true,
                            &dynamic.lora_b[slot_index],
                        )?;
                        ctx.set_adapter_optimizer_tensor(
                            adapter_id,
                            slot.layer as i64,
                            module,
                            false,
                            false,
                            &dynamic.adam_m[optimizer_index],
                        )?;
                        ctx.set_adapter_optimizer_tensor(
                            adapter_id,
                            slot.layer as i64,
                            module,
                            true,
                            false,
                            &dynamic.adam_m[optimizer_index + 1],
                        )?;
                        ctx.set_adapter_optimizer_tensor(
                            adapter_id,
                            slot.layer as i64,
                            module,
                            false,
                            true,
                            &dynamic.adam_v[optimizer_index],
                        )?;
                        ctx.set_adapter_optimizer_tensor(
                            adapter_id,
                            slot.layer as i64,
                            module,
                            true,
                            true,
                            &dynamic.adam_v[optimizer_index + 1],
                        )?;
                        optimizer_index += 2;
                    }
                    if optimizer_index != dynamic.manifest.optimizer_count {
                        bail!(
                            "dynamic adapter {} optimizer count mismatch",
                            dynamic.manifest.id
                        );
                    }
                    Ok(())
                })();
                if let Err(error) = load_result {
                    let _ = ctx.remove_lora(adapter_id);
                    return Err(error);
                }
                let optimizer_step = i64::try_from(dynamic.manifest.optimizer_step)
                    .context("dynamic adapter optimizer step exceeds native range")?;
                if let Err(error) = ctx.set_adapter_step_count(adapter_id, optimizer_step) {
                    let _ = ctx.remove_lora(adapter_id);
                    return Err(error);
                }
                self.dynamic_lora_configs.insert(adapter_id, lora_config);
                self.dynamic_lora_optimizer_lrs
                    .insert(adapter_id, optimizer_lr);
            }
        }
        // Import Adam optimizer state into C++ context
        if let Some(ctx) = &self.ctx {
            let native_slot_count = ctx.lora_count() as usize;
            if fixed_slots.len() != native_slot_count {
                bail!(
                    "fixed LoRA registry count {} does not match native slot count {native_slot_count}",
                    fixed_slots.len()
                );
            }
            let active_slot_indices = fixed_slots
                .iter()
                .filter(|slot| slot.active)
                .map(|slot| slot.local_index)
                .collect::<Vec<_>>();
            let restore_slot_indices = checkpoint::fixed_restore_slot_indices(
                data.lora_a.len(),
                data.lora_b.len(),
                &active_slot_indices,
                native_slot_count,
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
            if data.adam_m.is_empty()
                && !restore_slot_indices.is_empty()
                && data.manifest.effective_fixed_optimizer_step() > 0
            {
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
                let expected_native_optimizer_count = native_slot_count.saturating_mul(2);
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
                tracing::info!(imported, "optimizer state imported");
            }
            let native_step = i64::try_from(data.manifest.effective_fixed_optimizer_step())
                .context("checkpoint step exceeds the native optimizer range")?;
            ctx.set_step_count(native_step)?;
        }
        self.step = data.manifest.step;
        self.last_loss = data.manifest.loss;
        self.state = SessionState::Paused { step: self.step };
        Ok((data.manifest.step, data.manifest.loss))
    }

    fn export_adapter(&self, path: &str, adapter_id: Option<i64>) -> Result<usize> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA not initialized"))?;

        let model_path = self
            .model_path
            .as_ref()
            .ok_or_else(|| anyhow!("model not loaded"))?;
        let runtime_config =
            rustrain_qwen3_6::config::read_qwen36_runtime_config(std::path::Path::new(model_path))?;
        let adapter_id = adapter_id.unwrap_or(0);
        let lora_config = if adapter_id == 0 {
            Qwen36LoraConfig {
                rank: self.lora_rank,
                alpha: self.lora_alpha,
                target_layers: self.lora_target_layers.clone(),
                target_modules: self.lora_target_modules.clone(),
            }
        } else {
            self.dynamic_lora_configs
                .get(&adapter_id)
                .cloned()
                .ok_or_else(|| anyhow!("unknown dynamic LoRA adapter: {adapter_id}"))?
        };
        let slots = rustrain_qwen3_6::lora::native_lora_slots(&runtime_config, &lora_config);
        let mut exported = Vec::with_capacity(slots.len());
        for slot in slots {
            if adapter_id == 0 {
                let a = ctx
                    .get_lora_a(slot.index as i64)
                    .with_context(|| format!("native LoRA slot {} is missing A", slot.index))?;
                let b = ctx
                    .get_lora_b(slot.index as i64)
                    .with_context(|| format!("native LoRA slot {} is missing B", slot.index))?;
                exported.push((a, b));
            } else if slot.active {
                let module = slot.module.cpp_name();
                let a = ctx
                    .get_adapter_lora_tensor(adapter_id, slot.layer as i64, module, false)
                    .with_context(|| {
                        format!(
                            "dynamic LoRA A is missing: adapter={adapter_id} layer={} module={module}",
                            slot.layer
                        )
                    })?;
                let b = ctx
                    .get_adapter_lora_tensor(adapter_id, slot.layer as i64, module, true)
                    .with_context(|| {
                        format!(
                            "dynamic LoRA B is missing: adapter={adapter_id} layer={} module={module}",
                            slot.layer
                        )
                    })?;
                exported.push((a, b));
            } else {
                let placeholder = Tensor::zeros([], (Kind::Float, Device::Cpu));
                exported.push((placeholder.shallow_clone(), placeholder));
            }
        }
        let artifact = Qwen36AdapterArtifact::from_native_exports(
            model_path,
            "qwen3_hybrid_lora_sft",
            Some(std::path::Path::new(model_path)),
            &runtime_config,
            &lora_config,
            exported,
        )?;
        let count = artifact.tensors.len();
        artifact.save(std::path::Path::new(path))?;
        tracing::info!(params = count, path, "adapter exported");
        Ok(count)
    }

    fn import_adapter(&mut self, path: &str) -> Result<i64> {
        let model_path = self
            .model_path
            .as_ref()
            .ok_or_else(|| anyhow!("model not loaded"))?
            .clone();
        let runtime_config = rustrain_qwen3_6::config::read_qwen36_runtime_config(
            std::path::Path::new(&model_path),
        )?;
        let artifact = Qwen36AdapterArtifact::load(std::path::Path::new(path))?;
        let target_modules = artifact
            .config
            .target_modules
            .iter()
            .map(|name| Qwen36LoraTargetModule::parse(name))
            .collect::<Result<Vec<_>>>()?;
        let target_layers = artifact.config.target_layers.clone();
        let lora_config = Qwen36LoraConfig {
            rank: artifact.config.r,
            alpha: artifact.config.lora_alpha,
            target_layers: target_layers.clone(),
            target_modules: target_modules.clone(),
        };
        rustrain_qwen3_6::lora::validate_lora_targets(&runtime_config, &lora_config)?;

        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA not initialized"))?;
        let layer_ids = target_layers
            .iter()
            .map(|&layer| layer as i64)
            .collect::<Vec<_>>();
        let module_csv = target_modules
            .iter()
            .map(Qwen36LoraTargetModule::cpp_name)
            .collect::<Vec<_>>()
            .join(",");
        let adapter_id =
            ctx.add_lora(lora_config.rank, lora_config.alpha, &layer_ids, &module_csv)?;

        let load_result = (|| -> Result<()> {
            for (name, tensor) in &artifact.tensors {
                let (layer, module, is_b) =
                    rustrain_qwen3_6::lora::parse_adapter_tensor_name(&runtime_config, name)?;
                if layer >= runtime_config.num_hidden_layers {
                    bail!("adapter tensor layer {layer} is outside the model");
                }
                ctx.set_adapter_lora_tensor(
                    adapter_id,
                    layer as i64,
                    module.cpp_name(),
                    is_b,
                    tensor,
                )?;
            }
            Ok(())
        })();
        if let Err(error) = load_result {
            let _ = ctx.remove_lora(adapter_id);
            return Err(error);
        }
        self.dynamic_lora_configs.insert(adapter_id, lora_config);
        self.dynamic_lora_optimizer_lrs.insert(adapter_id, self.lr);
        tracing::info!(adapter_id, path, "LoRA adapter imported");
        Ok(adapter_id)
    }

    fn status(&self) -> SessionStatus {
        let state = match &self.state {
            SessionState::Unloaded => "unloaded",
            SessionState::Loaded { .. } => "loaded",
            SessionState::Ready { .. } => "ready",
            SessionState::Training { .. } => "training",
            SessionState::Paused { .. } => "paused",
            SessionState::Error(_) => "error",
        };
        SessionStatus {
            state: state.to_string(),
            step: self.step,
            last_loss: self.last_loss,
            model_path: self.model_path.clone().unwrap_or_default(),
        }
    }

    fn get_metrics(&self) -> Vec<StepMetric> {
        self.metrics
            .as_ref()
            .map(|m| m.read_metrics())
            .unwrap_or_default()
    }

    fn add_lora(&mut self, req: AddLoRARequest) -> Result<i64> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("model not loaded — call load_model + init_lora first"))?;
        let model_path = self
            .model_path
            .as_ref()
            .ok_or_else(|| anyhow!("model not loaded"))?;
        let runtime_config =
            rustrain_qwen3_6::config::read_qwen36_runtime_config(std::path::Path::new(model_path))?;
        let target_modules = if req.target_modules.trim().is_empty() {
            let mut modules = std::collections::BTreeSet::new();
            for layer_type in &runtime_config.layer_types {
                match layer_type {
                    rustrain_qwen3_6::config::LayerType::FullAttention => {
                        modules.extend([
                            Qwen36LoraTargetModule::QProj,
                            Qwen36LoraTargetModule::KProj,
                            Qwen36LoraTargetModule::VProj,
                            Qwen36LoraTargetModule::OProj,
                        ]);
                    }
                    rustrain_qwen3_6::config::LayerType::LinearAttention => {
                        modules.extend([
                            Qwen36LoraTargetModule::InProjQkv,
                            Qwen36LoraTargetModule::InProjZ,
                            Qwen36LoraTargetModule::InProjA,
                            Qwen36LoraTargetModule::InProjB,
                            Qwen36LoraTargetModule::OutProj,
                        ]);
                    }
                }
            }
            if runtime_config.is_moe {
                modules.extend([
                    Qwen36LoraTargetModule::SharedGateProj,
                    Qwen36LoraTargetModule::SharedUpProj,
                    Qwen36LoraTargetModule::SharedDownProj,
                    Qwen36LoraTargetModule::ExpertsGateUpProj,
                    Qwen36LoraTargetModule::ExpertsDownProj,
                ]);
            } else {
                modules.extend([
                    Qwen36LoraTargetModule::GateProj,
                    Qwen36LoraTargetModule::UpProj,
                    Qwen36LoraTargetModule::DownProj,
                ]);
            }
            modules.into_iter().collect()
        } else {
            req.target_modules
                .split(',')
                .filter(|name| !name.is_empty())
                .map(Qwen36LoraTargetModule::parse)
                .collect::<Result<Vec<_>>>()?
        };
        let config = Qwen36LoraConfig {
            rank: req.rank,
            alpha: req.alpha,
            target_layers: req
                .target_layers
                .iter()
                .map(|&layer| layer as usize)
                .collect(),
            target_modules: target_modules.clone(),
        };
        rustrain_qwen3_6::lora::validate_lora_targets(&runtime_config, &config)?;
        let native_module_names = target_modules
            .iter()
            .map(Qwen36LoraTargetModule::cpp_name)
            .collect::<Vec<_>>();
        let module_csv = native_module_names.join(",");
        let optimizer_lr = req.optimizer_lr.unwrap_or(self.lr);
        let id = match req.optimizer_lr {
            Some(optimizer_lr) => ctx.add_lora_with_optimizer_lr(
                req.rank,
                req.alpha,
                &req.target_layers,
                &module_csv,
                optimizer_lr,
            )?,
            None => ctx.add_lora(req.rank, req.alpha, &req.target_layers, &module_csv)?,
        };
        self.dynamic_lora_configs.insert(id, config);
        self.dynamic_lora_optimizer_lrs.insert(id, optimizer_lr);
        tracing::info!(adapter_id = id, rank = req.rank, "LoRA adapter added");
        Ok(id)
    }

    fn remove_lora(&mut self, adapter_id: i64) -> Result<bool> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("model not loaded"))?;
        let removed = ctx.remove_lora(adapter_id)?;
        if removed {
            self.dynamic_lora_configs.remove(&adapter_id);
            self.dynamic_lora_optimizer_lrs.remove(&adapter_id);
            tracing::info!(adapter_id, "LoRA adapter removed");
        }
        Ok(removed)
    }

    fn list_lora(&self) -> Vec<i64> {
        self.ctx
            .as_ref()
            .map(|ctx| ctx.list_lora())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        validate_dynamic_adapter_manifests, validate_qwen_parallel_features,
        validate_tp_intermediate_sizes,
    };
    use rustrain_qwen3_6::checkpoint::DynamicAdapterManifest;

    fn dynamic_manifest(
        id: i64,
        rank: i64,
        alpha: f64,
        target_modules: &[&str],
    ) -> DynamicAdapterManifest {
        DynamicAdapterManifest {
            id,
            rank,
            alpha,
            optimizer_step: id as u64,
            optimizer_lr: None,
            target_layers: vec![0],
            target_modules: target_modules
                .iter()
                .map(|module| (*module).to_string())
                .collect(),
            shard_layouts: Vec::new(),
            slot_identities: Vec::new(),
            parameter_count: 1,
            optimizer_count: 2,
        }
    }

    #[test]
    fn checkpoint_dynamic_manifests_allow_heterogeneous_signatures() {
        let manifests = [
            dynamic_manifest(1, 4, 8.0, &["q_proj", "v_proj"]),
            dynamic_manifest(2, 16, 32.0, &["down_proj"]),
        ];
        validate_dynamic_adapter_manifests(manifests.iter()).unwrap();
    }

    #[test]
    fn checkpoint_dynamic_manifests_reject_duplicate_ids() {
        let manifests = [
            dynamic_manifest(7, 4, 8.0, &["q_proj"]),
            dynamic_manifest(7, 8, 16.0, &["down_proj"]),
        ];
        let error = validate_dynamic_adapter_manifests(manifests.iter()).unwrap_err();
        assert!(error.to_string().contains("positive and unique"));
    }

    #[test]
    fn checkpoint_dynamic_manifests_reject_invalid_optimizer_lr() {
        let mut manifest = dynamic_manifest(1, 4, 8.0, &["q_proj"]);
        manifest.optimizer_lr = Some(f64::NAN);
        let error = validate_dynamic_adapter_manifests([&manifest]).unwrap_err();
        assert!(error.to_string().contains("optimizer learning rate"));
    }

    #[test]
    fn source_sharded_moe_tp_ep_is_supported() {
        validate_qwen_parallel_features(true, 2, 2, true, true).unwrap();
    }

    #[test]
    fn replicated_expert_tp_is_rejected() {
        let error = validate_qwen_parallel_features(true, 2, 2, true, false).unwrap_err();
        assert!(error.to_string().contains("replicated expert TP"));
    }

    #[test]
    fn source_sharded_ep_requires_a2a() {
        let error = validate_qwen_parallel_features(true, 2, 2, false, true).unwrap_err();
        assert!(error.to_string().contains("requires QWEN36_EP_A2A=1"));
    }

    #[test]
    fn source_sharding_requires_expert_parallelism() {
        let error = validate_qwen_parallel_features(true, 2, 1, true, true).unwrap_err();
        assert!(error.to_string().contains("expert-parallel training"));
    }

    #[test]
    fn tp_mlp_accepts_divisible_routed_and_shared_intermediates() {
        validate_tp_intermediate_sizes(2, &[("routed expert", 128), ("shared expert", 64)])
            .unwrap();
    }

    #[test]
    fn tp_mlp_rejects_non_divisible_shared_intermediate() {
        let error =
            validate_tp_intermediate_sizes(4, &[("routed expert", 128), ("shared expert", 66)])
                .unwrap_err();
        assert_eq!(
            error.to_string(),
            "shared expert intermediate_size=66 must be divisible by TP_SIZE=4"
        );
    }
}
