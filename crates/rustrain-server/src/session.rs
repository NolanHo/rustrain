//! Training session trait + Qwen3.6 implementation.

use anyhow::{anyhow, bail, Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tch::{Device, Kind, Tensor};
use tokio::sync::Mutex;

use crate::checkpoint;
use crate::metrics::{FileMetricsSink, MetricsSink, StepMetric};
use rustrain_qwen3_6::lora::{Qwen36AdapterArtifact, Qwen36LoraConfig, Qwen36LoraTargetModule};

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

#[derive(Debug)]
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
    ) -> Result<TrainOutput>;
    fn eval_step(&self, input: TrainInput) -> Result<EvalOutput>;
    fn save_checkpoint(&self, path: &str) -> Result<(u64, f64)>;
    fn load_checkpoint(&mut self, path: &str) -> Result<(u64, f64)>;
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

#[derive(Debug)]
pub struct AddLoRARequest {
    pub rank: i64,
    pub alpha: f64,
    pub target_layers: Vec<i64>,
    pub target_modules: String, // comma-separated, empty = all
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
    lr: f64,
    metrics: Option<Arc<FileMetricsSink>>,
    last_loss: f64,
    step: u64,
    // NCCL marker for EP (comm stored in C++ TrainingContext)
    _nccl_ep: bool,
}

// SAFETY: CppTrainingContext holds a raw pointer to C++ TrainingContext.
// The C++ context is only accessed from the thread that owns Qwen36Session.
// The Mutex in SessionManager ensures single-threaded access.
unsafe impl Send for Qwen36Session {}

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
            lr: 1e-4,
            metrics: Some(Arc::new(FileMetricsSink::new(metrics_path))),
            last_loss: 0.0,
            step: 0,
            _nccl_ep: false,
        }
    }

    /// Get the device this session is bound to.
    pub fn device(&self) -> Device {
        self.device
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

        // Build needed weight set — keys must match safetensors (with model prefix)
        let n_layers = runtime_config.num_hidden_layers;
        let wp = &runtime_config.weight_prefix;
        let mut needed: std::collections::HashSet<String> = std::collections::HashSet::new();
        needed.insert(format!("{wp}embed_tokens.weight"));
        needed.insert(format!("{wp}norm.weight"));
        if !runtime_config.tie_word_embeddings {
            // lm_head.weight is always at top level (no model prefix), even for multimodal models
            needed.insert("lm_head.weight".to_string());
        }
        for layer in 0..n_layers {
            let p = format!("{wp}layers.{layer}");
            needed.insert(format!("{p}.input_layernorm.weight"));
            needed.insert(format!("{p}.post_attention_layernorm.weight"));
            match runtime_config.layer_types[layer] {
                rustrain_qwen3_6::config::LayerType::FullAttention => {
                    for w in &["q_proj", "q_norm", "k_proj", "k_norm", "v_proj", "o_proj"] {
                        needed.insert(format!("{p}.self_attn.{w}.weight"));
                    }
                }
                rustrain_qwen3_6::config::LayerType::LinearAttention => {
                    needed.insert(format!("{p}.linear_attn.in_proj_qkv.weight"));
                    needed.insert(format!("{p}.linear_attn.in_proj_z.weight"));
                    needed.insert(format!("{p}.linear_attn.in_proj_a.weight"));
                    needed.insert(format!("{p}.linear_attn.in_proj_b.weight"));
                    needed.insert(format!("{p}.linear_attn.A_log"));
                    needed.insert(format!("{p}.linear_attn.dt_bias"));
                    needed.insert(format!("{p}.linear_attn.conv1d.weight"));
                    needed.insert(format!("{p}.linear_attn.norm.weight"));
                    needed.insert(format!("{p}.linear_attn.out_proj.weight"));
                }
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

        // ── Expert Parallel support ──
        // Read EP params from env vars (set by launcher script)
        let ep_rank = std::env::var("RANK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let ep_world_size = std::env::var("WORLD_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1);
        let local_rank = std::env::var("LOCAL_RANK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let is_ep = ep_world_size > 1 && runtime_config.is_moe;
        let is_data_parallel = ep_world_size > 1 && !runtime_config.is_moe;

        // Compute expert shard
        let (expert_start, expert_count) = if is_ep {
            assert!(
                runtime_config.num_experts % ep_world_size == 0,
                "num_experts {} not divisible by ep_world_size {}",
                runtime_config.num_experts,
                ep_world_size
            );
            let epr = runtime_config.num_experts / ep_world_size;
            (ep_rank * epr, epr)
        } else {
            (0, runtime_config.num_experts)
        };

        // Set CUDA device for any torchrun worker. Dense Qwen workers use
        // replicated weights and NCCL gradient all-reduce (LoRA-only DP).
        if is_ep || is_data_parallel {
            self.device = tch::Device::Cuda(local_rank);
        }

        // Move to device — for EP, narrow expert tensors before GPU transfer
        let num_experts = runtime_config.num_experts as i64;
        let mut weights: std::collections::BTreeMap<String, Tensor> =
            std::collections::BTreeMap::new();
        for (name, tensor) in raw_weights {
            let needs_narrow = is_ep
                && (name.contains(".mlp.experts.gate_up_proj")
                    || name.contains(".mlp.experts.down_proj"));
            if needs_narrow && tensor.size()[0] == num_experts {
                let narrowed = tensor
                    .narrow(0, expert_start as i64, expert_count as i64)
                    .contiguous()
                    .to_device(self.device)
                    .to_kind(self.compute_kind);
                weights.insert(name, narrowed);
            } else {
                let t = tensor.to_device(self.device);
                let processed = t.to_kind(self.compute_kind);
                weights.insert(name, processed);
            }
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
        let lora_scaling = req.alpha as f64 / req.rank as f64;
        let ctx = rustrain_qwen3_6::kernel::CppTrainingContext::new(
            &weights,
            &runtime_config,
            self.compute_kind,
            req.lr,
            req.beta1,
            req.beta2,
            req.eps,
            lora_scaling,
            req.rank,
            &all_layers,
            &target_modules,
            expert_start,
            expert_count,
        )?;

        // Initialize NCCL directly in C++. The same communicator handles EP
        // output collectives and replicated-weight LoRA gradient all-reduce.
        let nccl_ep = if is_ep || is_data_parallel {
            unsafe {
                std::env::set_var(
                    "RUSTRAIN_DATA_PARALLEL",
                    if is_data_parallel { "1" } else { "0" },
                );
            }
            let ret = ctx.init_nccl();
            if ret != 0 {
                return Err(anyhow!("C++ NCCL init failed (code {})", ret));
            }
            tracing::info!(
                ep_rank,
                ep_world_size,
                data_parallel = is_data_parallel,
                "NCCL communicator created in C++ for EP"
            );
            true
        } else {
            false
        };

        let count = ctx.lora_count() as usize;
        self.ctx = Some(ctx);
        self.weights = Some(weights); // Keep alive — C++ holds raw pointers
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
        self.step += 1;
        self.last_loss = loss;
        self.state = SessionState::Training { step: self.step };

        // Record metric
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

        Ok(TrainOutput {
            loss,
            step: self.step,
        })
    }

    fn train_multi_lora(
        &mut self,
        input: TrainInput,
        n_total: i32,
        rank: i32,
    ) -> Result<TrainOutput> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA not initialized"))?;

        let loss = ctx.train_multi_lora(
            &input.input_ids,
            &input.target_mask,
            &input.attention_mask,
            n_total,
            rank,
        )?;
        self.step += 1;
        self.last_loss = loss;
        self.state = SessionState::Training { step: self.step };

        Ok(TrainOutput {
            loss,
            step: self.step,
        })
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
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA not initialized"))?;

        let lora_count = ctx.lora_count();
        let mut lora_a = Vec::new();
        let mut lora_b = Vec::new();
        for i in 0..lora_count {
            if let (Some(a), Some(b)) = (ctx.get_lora_a(i as i64), ctx.get_lora_b(i as i64)) {
                lora_a.push(a);
                lora_b.push(b);
            }
        }

        // Export Adam optimizer state
        let (adam_m, adam_v) = ctx.export_optimizer_state()?;

        let mut dynamic_adapters = Vec::new();
        if !self.dynamic_lora_configs.is_empty() {
            let model_path = self
                .model_path
                .as_ref()
                .ok_or_else(|| anyhow!("model path unavailable for dynamic LoRA checkpoint"))?;
            let runtime_config = rustrain_qwen3_6::config::read_qwen36_runtime_config(
                std::path::Path::new(model_path),
            )?;
            for (&adapter_id, lora_config) in &self.dynamic_lora_configs {
                let slots = rustrain_qwen3_6::lora::native_lora_slots(&runtime_config, lora_config);
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
                dynamic_adapters.push(checkpoint::DynamicAdapterCheckpoint {
                    manifest: checkpoint::DynamicAdapterManifest {
                        id: adapter_id,
                        rank: lora_config.rank,
                        alpha: lora_config.alpha,
                        target_layers: lora_config.target_layers.clone(),
                        target_modules: lora_config
                            .target_modules
                            .iter()
                            .map(|module| module.cpp_name().to_string())
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

        checkpoint::save_checkpoint_with_dynamic(
            std::path::Path::new(path),
            self.step,
            self.last_loss,
            self.model_path.as_deref().unwrap_or(""),
            self.lora_rank,
            self.lora_alpha,
            &lora_a,
            &lora_b,
            &adam_m,
            &adam_v,
            &dynamic_adapters,
        )?;

        Ok((self.step, self.last_loss))
    }

    fn load_checkpoint(&mut self, path: &str) -> Result<(u64, f64)> {
        let data = checkpoint::load_checkpoint(std::path::Path::new(path))?;
        if !data.dynamic_adapters.is_empty() {
            let model_path = self
                .model_path
                .as_ref()
                .ok_or_else(|| anyhow!("model path unavailable for dynamic LoRA checkpoint"))?;
            let runtime_config = rustrain_qwen3_6::config::read_qwen36_runtime_config(
                std::path::Path::new(model_path),
            )?;
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
                let allocated_id =
                    ctx.add_lora(lora_config.rank, lora_config.alpha, &layer_ids, &module_csv)?;
                if allocated_id != dynamic.manifest.id {
                    if let Err(error) = ctx.set_adapter_id(allocated_id, dynamic.manifest.id) {
                        let _ = ctx.remove_lora(allocated_id);
                        return Err(error);
                    }
                }
                let adapter_id = dynamic.manifest.id;
                let load_result = (|| -> Result<()> {
                    let slots =
                        rustrain_qwen3_6::lora::native_lora_slots(&runtime_config, &lora_config);
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
                self.dynamic_lora_configs.insert(adapter_id, lora_config);
            }
        }
        // Import Adam optimizer state into C++ context
        if let Some(ctx) = &self.ctx {
            if data.lora_a.len() != data.lora_b.len()
                || data.lora_a.len() != ctx.lora_count() as usize
            {
                return Err(anyhow!(
                    "checkpoint LoRA slot count mismatch: checkpoint A/B={}/{}, context={}",
                    data.lora_a.len(),
                    data.lora_b.len(),
                    ctx.lora_count()
                ));
            }
            for (index, (a, b)) in data.lora_a.iter().zip(&data.lora_b).enumerate() {
                ctx.set_lora_tensor(index as i64, false, a)?;
                ctx.set_lora_tensor(index as i64, true, b)?;
            }
            if !data.adam_m.is_empty() && !data.adam_v.is_empty() {
                ctx.import_optimizer_state(&data.adam_m, &data.adam_v)?;
                tracing::info!(imported = data.adam_m.len(), "optimizer state imported");
            }
        }
        self.step = data.manifest.step;
        self.last_loss = data.manifest.loss;
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
        let id = ctx.add_lora(req.rank, req.alpha, &req.target_layers, &module_csv)?;
        self.dynamic_lora_configs.insert(id, config);
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
