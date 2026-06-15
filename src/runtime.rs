use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use chrono::Local;
use serde::{Deserialize, Serialize};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub run: RunConfig,
    pub model: ModelConfig,
    pub train: TrainConfig,
    #[serde(default)]
    pub data: Option<DataConfig>,
    pub parallel: ParallelConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RunConfig {
    pub name: String,
    pub base_dir: PathBuf,
    pub seed: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    pub name: String,
    pub architecture: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub seq_len: usize,
    pub norm: String,
    pub activation: String,
    pub rope: bool,
    pub rms_norm_eps: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainConfig {
    pub max_steps: u64,
    #[serde(default)]
    pub resume_from: Option<PathBuf>,
    pub micro_batch_size: usize,
    pub global_batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_eps: f32,
    pub dtype: DType,
    pub device: Device,
    pub checkpoint_every: u64,
    #[serde(default = "default_eval_every")]
    pub eval_every: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DataConfig {
    pub kind: DataKind,
    pub paths: Vec<PathBuf>,
    #[serde(default = "default_train_split")]
    pub train_split: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DataKind {
    Text,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Fp32,
    Fp16,
    Bf16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ParallelConfig {
    pub tensor_model_parallel_size: usize,
    pub pipeline_model_parallel_size: usize,
    pub data_parallel_size: usize,
    pub expert_model_parallel_size: usize,
    pub context_parallel_size: usize,
}

#[derive(Debug)]
pub struct RunPaths {
    pub root: PathBuf,
    pub checkpoints: PathBuf,
    pub logs: PathBuf,
    pub cache: PathBuf,
    pub resolved_config: PathBuf,
}

pub fn load_config(path: &Path) -> Result<Config> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read config {}", path.display()))?;
    toml::from_str(&contents).with_context(|| format!("failed to parse config {}", path.display()))
}

pub fn validate_config(config: &Config) -> Result<()> {
    if !matches!(config.train.device, Device::Cpu) {
        return Err(anyhow!("M1 toy backend only supports device = \"cpu\""));
    }

    let model = &config.model;
    if model.vocab_size < 2 {
        return Err(anyhow!("vocab_size must be at least 2"));
    }
    if model.hidden_size == 0 || model.intermediate_size == 0 {
        return Err(anyhow!(
            "hidden_size and intermediate_size must be greater than zero"
        ));
    }
    if model.seq_len < 2 {
        return Err(anyhow!("seq_len must be at least 2"));
    }
    if model.num_layers == 0 {
        return Err(anyhow!("num_layers must be greater than zero"));
    }
    if model.num_attention_heads == 0 || model.num_key_value_heads == 0 {
        return Err(anyhow!(
            "num_attention_heads and num_key_value_heads must be greater than zero"
        ));
    }
    if model.hidden_size % model.num_attention_heads != 0 {
        return Err(anyhow!(
            "hidden_size must be divisible by num_attention_heads"
        ));
    }
    if model.num_attention_heads % model.num_key_value_heads != 0 {
        return Err(anyhow!(
            "num_attention_heads must be divisible by num_key_value_heads"
        ));
    }
    let head_dim = model.hidden_size / model.num_attention_heads;
    if head_dim % 2 != 0 {
        return Err(anyhow!("head_dim must be even for RoPE"));
    }
    if model.norm != "rmsnorm" {
        return Err(anyhow!("M1 expects norm = \"rmsnorm\""));
    }
    if model.activation != "swiglu" {
        return Err(anyhow!("M1 expects activation = \"swiglu\""));
    }
    if !model.rope {
        return Err(anyhow!("M1 expects rope = true"));
    }

    let parallel = &config.parallel;
    let parallel_sizes = [
        (
            "tensor_model_parallel_size",
            parallel.tensor_model_parallel_size,
        ),
        (
            "pipeline_model_parallel_size",
            parallel.pipeline_model_parallel_size,
        ),
        ("data_parallel_size", parallel.data_parallel_size),
        (
            "expert_model_parallel_size",
            parallel.expert_model_parallel_size,
        ),
        ("context_parallel_size", parallel.context_parallel_size),
    ];

    for (name, value) in parallel_sizes {
        if value == 0 {
            return Err(anyhow!("{name} must be greater than zero"));
        }
        if value != 1 {
            return Err(anyhow!("M1 toy backend requires {name} = 1"));
        }
    }

    if config.train.micro_batch_size != 1 || config.train.global_batch_size != 1 {
        return Err(anyhow!(
            "M1 toy backend currently requires micro_batch_size = global_batch_size = 1"
        ));
    }
    if config.train.gradient_accumulation_steps == 0 {
        return Err(anyhow!(
            "gradient_accumulation_steps must be greater than zero"
        ));
    }
    if !(0.0..1.0).contains(&config.train.adam_beta1)
        || !(0.0..1.0).contains(&config.train.adam_beta2)
    {
        return Err(anyhow!("Adam beta values must be in [0, 1)"));
    }
    if config.train.adam_eps <= 0.0 {
        return Err(anyhow!("adam_eps must be greater than zero"));
    }
    if let Some(data) = &config.data {
        if data.paths.is_empty() {
            return Err(anyhow!("data.paths must not be empty"));
        }
        if !(0.0..1.0).contains(&data.train_split) {
            return Err(anyhow!("data.train_split must be in (0, 1)"));
        }
    }

    Ok(())
}

pub fn prepare_run_directory(run: &RunConfig) -> Result<RunPaths> {
    let timestamp = Local::now().format("%Y%m%d-%H%M%S");
    let root = run.base_dir.join(format!("{}-{timestamp}", run.name));
    let checkpoints = root.join("checkpoints");
    let logs = root.join("logs");
    let cache = root.join("cache");
    let resolved_config = root.join("resolved_config.toml");

    fs::create_dir_all(&checkpoints)
        .with_context(|| format!("failed to create {}", checkpoints.display()))?;
    fs::create_dir_all(&logs).with_context(|| format!("failed to create {}", logs.display()))?;
    fs::create_dir_all(&cache).with_context(|| format!("failed to create {}", cache.display()))?;

    Ok(RunPaths {
        root,
        checkpoints,
        logs,
        cache,
        resolved_config,
    })
}

pub fn init_logging(log_dir: &Path) -> Result<tracing_appender::non_blocking::WorkerGuard> {
    let file_appender = tracing_appender::rolling::never(log_dir, "train.log");
    let (file_writer, guard) = tracing_appender::non_blocking(file_appender);
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt::layer().with_target(false))
        .with(fmt::layer().with_writer(file_writer).with_ansi(false))
        .try_init()
        .context("failed to initialize tracing subscriber")?;

    Ok(guard)
}

pub fn write_resolved_config(config: &Config, path: &Path) -> Result<()> {
    let contents = toml::to_string_pretty(config).context("failed to serialize resolved config")?;
    fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))
}

fn default_train_split() -> f32 {
    0.8
}

fn default_eval_every() -> u64 {
    0
}
