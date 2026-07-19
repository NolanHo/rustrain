use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use chrono::Local;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

use crate::backend::BackendKind;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub run: RunConfig,
    pub model: ModelConfig,
    pub train: TrainConfig,
    #[serde(default)]
    pub data: Option<DataConfig>,
    #[serde(default)]
    pub lora: Option<LoraConfig>,
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
    #[serde(default)]
    pub model_path: Option<PathBuf>,
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
    #[serde(default)]
    pub trainable_layers: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainConfig {
    pub max_steps: u64,
    #[serde(default)]
    pub resume_from: Option<PathBuf>,
    #[serde(default = "default_backend")]
    pub backend: BackendKind,
    pub micro_batch_size: usize,
    pub global_batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub learning_rate: f32,
    /// Megatron's total auxiliary MTP loss weight. Defaults to 0.1.
    #[serde(default = "default_mtp_loss_scaling_factor")]
    pub mtp_loss_scaling_factor: f64,
    pub weight_decay: f32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_eps: f32,
    #[serde(default = "default_lr_scheduler")]
    pub lr_scheduler: LrScheduler,
    #[serde(default)]
    pub max_grad_norm: Option<f32>,
    pub dtype: DType,
    pub device: Device,
    /// Maximum fraction of a CUDA device available to PyTorch's caching allocator.
    #[serde(default = "default_cuda_memory_fraction")]
    pub cuda_memory_fraction: f64,
    /// Pre-dequantize cached FP8 expert weights at startup for faster linear kernels.
    #[serde(default = "default_predequant_expert_weights")]
    pub predequant_expert_weights: bool,
    pub checkpoint_every: u64,
    #[serde(default = "default_eval_every")]
    pub eval_every: u64,
}

pub const DEFAULT_MTP_LOSS_SCALING_FACTOR: f64 = 0.1;

fn default_mtp_loss_scaling_factor() -> f64 {
    DEFAULT_MTP_LOSS_SCALING_FACTOR
}

fn default_cuda_memory_fraction() -> f64 {
    0.95
}

fn default_predequant_expert_weights() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DataConfig {
    pub kind: DataKind,
    pub paths: Vec<PathBuf>,
    #[serde(default)]
    pub eval_paths: Vec<PathBuf>,
    #[serde(default = "default_train_split")]
    pub train_split: f32,
    #[serde(default)]
    pub max_samples: Option<usize>,
    #[serde(default)]
    pub max_eval_samples: Option<usize>,
    #[serde(default = "default_data_shuffle")]
    pub shuffle: bool,
    #[serde(default)]
    pub index_cache: Option<PathBuf>,
    #[serde(default = "default_instruction_field")]
    pub instruction_field: String,
    #[serde(default = "default_input_field")]
    pub input_field: String,
    #[serde(default = "default_response_field")]
    pub response_field: String,
    #[serde(default)]
    pub source_instruction_fields: Vec<String>,
    #[serde(default)]
    pub source_input_fields: Vec<String>,
    #[serde(default)]
    pub source_response_fields: Vec<String>,
    #[serde(default)]
    pub system_field: Option<String>,
    #[serde(default)]
    pub chat_messages_field: Option<String>,
    #[serde(default)]
    pub min_system_chars: Option<usize>,
    #[serde(default)]
    pub max_system_chars: Option<usize>,
    #[serde(default)]
    pub system_contains_any: Vec<String>,
    #[serde(default)]
    pub system_excludes_any: Vec<String>,
    #[serde(default = "default_prompt_template")]
    pub prompt_template: String,
    #[serde(default = "default_prompt_with_input_template")]
    pub prompt_with_input_template: String,
    #[serde(default = "default_trim_fields")]
    pub trim_fields: bool,
    #[serde(default = "default_min_response_chars")]
    pub min_response_chars: usize,
    #[serde(default)]
    pub max_response_chars: Option<usize>,
    #[serde(default)]
    pub instruction_contains_any: Vec<String>,
    #[serde(default)]
    pub instruction_excludes_any: Vec<String>,
    #[serde(default)]
    pub response_contains_any: Vec<String>,
    #[serde(default)]
    pub response_excludes_any: Vec<String>,
    #[serde(default)]
    pub input_contains_any: Vec<String>,
    #[serde(default)]
    pub input_excludes_any: Vec<String>,
    #[serde(default)]
    pub field_regex_contains_any: Vec<FieldRegexFilter>,
    #[serde(default)]
    pub field_regex_excludes_any: Vec<FieldRegexFilter>,
    #[serde(default)]
    pub min_instruction_chars: Option<usize>,
    #[serde(default)]
    pub max_instruction_chars: Option<usize>,
    #[serde(default)]
    pub min_input_chars: Option<usize>,
    #[serde(default)]
    pub max_input_chars: Option<usize>,
    #[serde(default)]
    pub min_prompt_chars: Option<usize>,
    #[serde(default)]
    pub max_prompt_chars: Option<usize>,
    #[serde(default)]
    pub min_sample_chars: Option<usize>,
    #[serde(default)]
    pub max_sample_chars: Option<usize>,
    #[serde(default)]
    pub dedupe_samples: bool,
    #[serde(default)]
    pub field_replacements: Vec<FieldReplacement>,
    #[serde(default)]
    pub field_regex_replacements: Vec<FieldRegexReplacement>,
    #[serde(default)]
    pub normalize_whitespace: bool,
    #[serde(default)]
    pub field_defaults: Vec<FieldDefault>,
    #[serde(default)]
    pub field_case_transforms: Vec<FieldCaseTransform>,
    #[serde(default)]
    pub field_affixes: Vec<FieldAffix>,
    #[serde(default)]
    pub field_strips: Vec<FieldStrip>,
    #[serde(default)]
    pub field_splits: Vec<FieldSplit>,
    #[serde(default)]
    pub field_truncations: Vec<FieldTruncation>,
    #[serde(default)]
    pub field_transforms: Vec<FieldTransform>,
    #[serde(default)]
    pub source_weights: Vec<usize>,
    #[serde(default)]
    pub source_max_samples: Vec<usize>,
    #[serde(default)]
    pub skip_invalid_records: bool,
    #[serde(default)]
    pub external_metadata_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FieldReplacement {
    pub field: FieldReplacementTarget,
    pub pattern: String,
    pub replacement: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FieldRegexReplacement {
    pub field: FieldReplacementTarget,
    pub pattern: String,
    pub replacement: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FieldRegexFilter {
    pub field: FieldReplacementTarget,
    pub pattern: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FieldReplacementTarget {
    System,
    Instruction,
    Input,
    Response,
    All,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FieldDefault {
    pub field: FieldDefaultTarget,
    pub value: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FieldDefaultTarget {
    System,
    Instruction,
    Input,
    Response,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FieldCaseTransform {
    pub field: FieldReplacementTarget,
    pub case: FieldCaseTransformKind,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FieldCaseTransformKind {
    Lowercase,
    Uppercase,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FieldAffix {
    pub field: FieldReplacementTarget,
    #[serde(default)]
    pub prefix: String,
    #[serde(default)]
    pub suffix: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FieldStrip {
    pub field: FieldReplacementTarget,
    #[serde(default)]
    pub prefix: String,
    #[serde(default)]
    pub suffix: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FieldTruncation {
    pub field: FieldReplacementTarget,
    pub max_chars: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FieldSplit {
    pub field: FieldReplacementTarget,
    pub delimiter: String,
    pub side: FieldSplitSide,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FieldSplitSide {
    Before,
    After,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FieldTransform {
    pub field: FieldReplacementTarget,
    pub op: FieldTransformOp,
    #[serde(default)]
    pub value: String,
    #[serde(default)]
    pub pattern: String,
    #[serde(default)]
    pub replacement: String,
    #[serde(default)]
    pub prefix: String,
    #[serde(default)]
    pub suffix: String,
    #[serde(default)]
    pub delimiter: String,
    #[serde(default)]
    pub side: Option<FieldSplitSide>,
    #[serde(default)]
    pub max_chars: Option<usize>,
    #[serde(default)]
    pub case: Option<FieldCaseTransformKind>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FieldTransformOp {
    Default,
    Replace,
    RegexReplace,
    Case,
    Affix,
    Strip,
    Split,
    Truncate,
}

impl Default for FieldTransform {
    fn default() -> Self {
        Self {
            field: FieldReplacementTarget::Instruction,
            op: FieldTransformOp::Replace,
            value: String::new(),
            pattern: String::new(),
            replacement: String::new(),
            prefix: String::new(),
            suffix: String::new(),
            delimiter: String::new(),
            side: None,
            max_chars: None,
            case: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoraConfig {
    pub rank: i64,
    pub alpha: f64,
    pub target_layers: Vec<usize>,
    pub target_modules: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DataKind {
    Text,
    InstructionJsonl,
    InstructionArrow,
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LrScheduler {
    Constant,
    LinearDecay,
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

fn qwen_source_parallel_factor(
    is_qwen_hybrid: bool,
    is_expert_parallel_architecture: bool,
    ep_source_sharded: bool,
    parallel: &ParallelConfig,
) -> usize {
    if !is_qwen_hybrid {
        return 1;
    }
    let ep_source_factor = if is_expert_parallel_architecture && ep_source_sharded {
        parallel.expert_model_parallel_size
    } else {
        1
    };
    parallel.data_parallel_size * ep_source_factor
}

pub fn validate_config(config: &Config) -> Result<()> {
    if matches!(config.train.backend, BackendKind::NdArray)
        && !matches!(config.train.device, Device::Cpu)
    {
        return Err(anyhow!("ndarray backend only supports device = \"cpu\""));
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
    if let Some(trainable_layers) = &model.trainable_layers {
        if trainable_layers.is_empty() {
            return Err(anyhow!("model.trainable_layers must not be empty when set"));
        }
        for layer in trainable_layers {
            if *layer >= model.num_layers {
                return Err(anyhow!(
                    "model.trainable_layers contains layer {} outside model.num_layers {}",
                    layer,
                    model.num_layers
                ));
            }
        }
    }
    if model.num_attention_heads == 0 || model.num_key_value_heads == 0 {
        return Err(anyhow!(
            "num_attention_heads and num_key_value_heads must be greater than zero"
        ));
    }
    // Qwen3.5/3.6 uses custom head_dim (not hidden_size / num_heads)
    // Skip divisibility check — head_dim is read from model config.json
    // and may not equal hidden_size / num_attention_heads (e.g. Qwen3.6-27B: 5120/24=213.33)
    if model.num_attention_heads % model.num_key_value_heads != 0 {
        return Err(anyhow!(
            "num_attention_heads must be divisible by num_key_value_heads"
        ));
    }
    // Qwen3.5/3.6 uses custom head_dim from config.json, not hidden_size/num_heads
    // Skip head_dim parity check here — actual head_dim is read in C++ kernel
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

    let is_tch_tiny_lm = matches!(config.train.backend, BackendKind::Tch)
        && config.model.architecture == "tch_tiny_lm";
    let is_qwen_trainable_session = matches!(config.train.backend, BackendKind::Tch)
        && config.model.architecture == "qwen_trainable_session";
    let is_qwen_lora_sft = matches!(config.train.backend, BackendKind::Tch)
        && config.model.architecture == "qwen_lora_sft";
    let is_qwen3_hybrid_lora_sft = matches!(config.train.backend, BackendKind::Tch)
        && matches!(
            config.model.architecture.as_str(),
            "qwen3_5_lora_sft" | "qwen3_5_lora_sft_ep" | "qwen3_6_lora_sft" | "qwen3_6_lora_sft_ep"
        );
    let is_tch_moe_ep_session = matches!(config.train.backend, BackendKind::Tch)
        && config.model.architecture == "tch_moe_ep_session";
    let is_v4_tp_rank = matches!(config.train.backend, BackendKind::Tch)
        && config.model.architecture == "deepseek_v4_tp_rank";
    let is_v4_tp_train = matches!(config.train.backend, BackendKind::Tch)
        && config.model.architecture == "deepseek_v4_tp_train";
    let is_v4_ep_rank = matches!(config.train.backend, BackendKind::Tch)
        && config.model.architecture == "deepseek_v4_ep_rank";
    let is_v4_ep_train = matches!(config.train.backend, BackendKind::Tch)
        && config.model.architecture == "deepseek_v4_ep_train";
    let is_v4_lora_sft_ep = matches!(config.train.backend, BackendKind::Tch)
        && config.model.architecture == "deepseek_v4_lora_sft_ep";
    let is_glm5_lora_sft_ep = matches!(config.train.backend, BackendKind::Tch)
        && config.model.architecture == "glm5_lora_sft_ep";
    let is_qwen3_hybrid_lora_sft_ep = matches!(config.train.backend, BackendKind::Tch)
        && matches!(
            config.model.architecture.as_str(),
            "qwen3_5_lora_sft_ep" | "qwen3_6_lora_sft_ep"
        );
    let is_v4_tp_ep_train = matches!(config.train.backend, BackendKind::Tch)
        && config.model.architecture == "deepseek_v4_tp_ep_train";
    let is_v3_tp_rank = matches!(config.train.backend, BackendKind::Tch)
        && config.model.architecture == "deepseek_tp_rank";
    let is_v3_ep_rank = matches!(config.train.backend, BackendKind::Tch)
        && config.model.architecture == "deepseek_ep_rank";
    if is_qwen3_hybrid_lora_sft_ep && parallel.context_parallel_size != 1 {
        return Err(anyhow!(
            "{} does not yet support context parallelism; context_parallel_size must be 1",
            config.model.architecture
        ));
    }
    for (name, value) in parallel_sizes {
        if value == 0 {
            return Err(anyhow!("{name} must be greater than zero"));
        }
        if value != 1
            && !((is_tch_tiny_lm || is_qwen_trainable_session)
                && name == "data_parallel_size"
                && value == 2)
            && !(is_qwen3_hybrid_lora_sft
                && name == "data_parallel_size"
                && value >= 2
                && parallel.context_parallel_size == 1)
            && !(is_qwen3_hybrid_lora_sft
                && name == "tensor_model_parallel_size"
                && parallel.context_parallel_size == 1)
            && !(is_qwen3_hybrid_lora_sft
                && name == "pipeline_model_parallel_size"
                && parallel.context_parallel_size == 1)
            && !(is_qwen_trainable_session
                && name == "tensor_model_parallel_size"
                && value == 2
                && parallel.data_parallel_size == 1)
            && !(is_qwen3_hybrid_lora_sft_ep
                && name == "expert_model_parallel_size"
                && parallel.context_parallel_size == 1)
            && !(is_tch_moe_ep_session
                && name == "expert_model_parallel_size"
                && value == 2
                && parallel.tensor_model_parallel_size == 1
                && parallel.data_parallel_size == 1)
            && !((is_v4_tp_rank || is_v3_tp_rank || is_v4_tp_train)
                && name == "tensor_model_parallel_size"
                && parallel.data_parallel_size == 1)
            && !((is_v4_ep_rank
                || is_v3_ep_rank
                || is_v4_ep_train
                || is_v4_lora_sft_ep
                || is_glm5_lora_sft_ep
                || is_qwen3_hybrid_lora_sft_ep)
                && name == "expert_model_parallel_size"
                && parallel.data_parallel_size == 1)
            && !(is_glm5_lora_sft_ep
                && (name == "tensor_model_parallel_size" || name == "context_parallel_size")
                && parallel.data_parallel_size == 1)
            && !(is_v4_tp_ep_train
                && (name == "tensor_model_parallel_size" || name == "expert_model_parallel_size")
                && parallel.data_parallel_size == 1)
        {
            return Err(anyhow!("M1 toy backend requires {name} = 1"));
        }
    }

    if is_qwen_trainable_session || is_qwen_lora_sft || is_qwen3_hybrid_lora_sft {
        if !matches!(config.train.device, Device::Cuda) {
            return Err(anyhow!(
                "{} requires device = \"cuda\"",
                config.model.architecture
            ));
        }
        if config.model.model_path.is_none() {
            return Err(anyhow!(
                "{} requires model.model_path",
                config.model.architecture
            ));
        }
    }

    if is_tch_moe_ep_session {
        if !matches!(config.train.device, Device::Cuda) {
            return Err(anyhow!("tch_moe_ep_session requires device = \"cuda\""));
        }
        if config.parallel.expert_model_parallel_size != 2 {
            return Err(anyhow!(
                "tch_moe_ep_session currently expects expert_model_parallel_size = 2"
            ));
        }
        if config.parallel.tensor_model_parallel_size != 1
            || config.parallel.pipeline_model_parallel_size != 1
            || config.parallel.data_parallel_size != 1
            || config.parallel.context_parallel_size != 1
        {
            return Err(anyhow!(
                "tch_moe_ep_session currently expects TP/PP/DP/CP sizes to remain 1"
            ));
        }
    }

    if is_qwen_lora_sft || is_qwen3_hybrid_lora_sft {
        let lora = config
            .lora
            .as_ref()
            .ok_or_else(|| anyhow!("qwen_lora_sft requires [lora] config"))?;
        if lora.rank <= 0 {
            return Err(anyhow!("lora.rank must be greater than zero"));
        }
        if lora.alpha <= 0.0 {
            return Err(anyhow!("lora.alpha must be greater than zero"));
        }
        if lora.target_layers.is_empty() {
            return Err(anyhow!("lora.target_layers must not be empty"));
        }
        for layer in &lora.target_layers {
            if *layer >= config.model.num_layers {
                return Err(anyhow!(
                    "lora.target_layers contains layer {} outside model.num_layers {}",
                    layer,
                    config.model.num_layers
                ));
            }
        }
        if lora.target_modules.is_empty() {
            return Err(anyhow!("lora.target_modules must not be empty"));
        }
        let supported_lora_modules: &[&str] = if is_qwen3_hybrid_lora_sft {
            &[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_a",
                "in_proj_b",
                "out_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "shared_gate_proj",
                "shared_up_proj",
                "shared_down_proj",
                "experts_gate_up_proj",
                "experts_down_proj",
            ]
        } else {
            &[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        };
        for module in &lora.target_modules {
            if !supported_lora_modules.contains(&module.as_str()) {
                return Err(anyhow!(
                    "{} unsupported lora.target_modules entry {}; supported: {}",
                    config.model.architecture,
                    module,
                    supported_lora_modules.join(", ")
                ));
            }
        }
        let mut unique_modules = std::collections::BTreeSet::new();
        if lora
            .target_modules
            .iter()
            .any(|module| !unique_modules.insert(module))
        {
            return Err(anyhow!(
                "{} lora.target_modules must not contain duplicates",
                config.model.architecture
            ));
        }
        if config.train.micro_batch_size == 0 {
            return Err(anyhow!(
                "{} requires micro_batch_size > 0",
                config.model.architecture
            ));
        }
        let ep_source_sharded = std::env::var("QWEN36_EP_A2A_SHARDED")
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false);
        let source_parallel_factor = qwen_source_parallel_factor(
            is_qwen3_hybrid_lora_sft,
            is_qwen3_hybrid_lora_sft_ep,
            ep_source_sharded,
            &config.parallel,
        );
        let expected_global_batch_size = config.train.micro_batch_size
            * config.train.gradient_accumulation_steps
            * source_parallel_factor;
        if config.train.global_batch_size != expected_global_batch_size {
            if source_parallel_factor > 1 {
                return Err(anyhow!(
                    "{} requires global_batch_size = micro_batch_size * gradient_accumulation_steps * effective source-parallel size ({source_parallel_factor}); this includes data_parallel_size and, for sharded EP sources, expert_model_parallel_size",
                    config.model.architecture,
                ));
            }
            return Err(anyhow!(
                "{} requires global_batch_size = micro_batch_size * gradient_accumulation_steps",
                config.model.architecture
            ));
        }
    } else if config.train.micro_batch_size != 1 || config.train.global_batch_size != 1 {
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
    if config.train.learning_rate <= 0.0 {
        return Err(anyhow!("learning_rate must be greater than zero"));
    }
    if !config.train.mtp_loss_scaling_factor.is_finite()
        || config.train.mtp_loss_scaling_factor < 0.0
    {
        return Err(anyhow!(
            "mtp_loss_scaling_factor must be finite and non-negative"
        ));
    }
    if !config.train.cuda_memory_fraction.is_finite()
        || config.train.cuda_memory_fraction <= 0.0
        || config.train.cuda_memory_fraction > 1.0
    {
        return Err(anyhow!("cuda_memory_fraction must be finite and in (0, 1]"));
    }
    if let Some(max_grad_norm) = config.train.max_grad_norm {
        if max_grad_norm <= 0.0 {
            return Err(anyhow!("max_grad_norm must be greater than zero"));
        }
    }
    if let Some(data) = &config.data {
        if data.paths.is_empty() {
            return Err(anyhow!("data.paths must not be empty"));
        }
        if !(0.0..1.0).contains(&data.train_split) {
            return Err(anyhow!("data.train_split must be in (0, 1)"));
        }
        if matches!(data.max_samples, Some(0)) {
            return Err(anyhow!(
                "data.max_samples must be greater than zero when set"
            ));
        }
        if matches!(data.max_eval_samples, Some(0)) {
            return Err(anyhow!(
                "data.max_eval_samples must be greater than zero when set"
            ));
        }
        if !(data.source_max_samples.is_empty()
            || data.source_max_samples.len() == 1
            || data.source_max_samples.len() == data.paths.len())
        {
            return Err(anyhow!(
                "data.source_max_samples must be empty, length 1, or match data.paths length"
            ));
        }
        if data.source_max_samples.iter().any(|limit| *limit == 0) {
            return Err(anyhow!(
                "data.source_max_samples entries must be greater than zero"
            ));
        }
        for (name, values, required) in [
            (
                "data.source_instruction_fields",
                &data.source_instruction_fields,
                true,
            ),
            ("data.source_input_fields", &data.source_input_fields, false),
            (
                "data.source_response_fields",
                &data.source_response_fields,
                true,
            ),
        ] {
            if !(values.is_empty() || values.len() == 1 || values.len() == data.paths.len()) {
                return Err(anyhow!(
                    "{name} must be empty, length 1, or match data.paths length"
                ));
            }
            if required && values.iter().any(|field| field.trim().is_empty()) {
                return Err(anyhow!("{name} entries must not be empty"));
            }
        }
        if let Some(max_response_chars) = data.max_response_chars {
            if max_response_chars == 0 {
                return Err(anyhow!(
                    "data.max_response_chars must be greater than zero when set"
                ));
            }
            if max_response_chars < data.min_response_chars {
                return Err(anyhow!(
                    "data.max_response_chars must be greater than or equal to data.min_response_chars"
                ));
            }
        }
        if data
            .instruction_contains_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            return Err(anyhow!(
                "data.instruction_contains_any entries must not be empty"
            ));
        }
        if data
            .instruction_excludes_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            return Err(anyhow!(
                "data.instruction_excludes_any entries must not be empty"
            ));
        }
        if data
            .response_contains_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            return Err(anyhow!(
                "data.response_contains_any entries must not be empty"
            ));
        }
        if data
            .response_excludes_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            return Err(anyhow!(
                "data.response_excludes_any entries must not be empty"
            ));
        }
        if data
            .input_contains_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            return Err(anyhow!("data.input_contains_any entries must not be empty"));
        }
        if data
            .input_excludes_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            return Err(anyhow!("data.input_excludes_any entries must not be empty"));
        }
        let has_system_source = data.system_field.is_some()
            || data.chat_messages_field.is_some()
            || data
                .field_defaults
                .iter()
                .any(|default| matches!(default.field, FieldDefaultTarget::System))
            || data.field_transforms.iter().any(|transform| {
                matches!(transform.field, FieldReplacementTarget::System)
                    && matches!(transform.op, FieldTransformOp::Default)
            });
        if !data.system_contains_any.is_empty() && !has_system_source {
            return Err(anyhow!(
                "data.system_contains_any requires data.system_field or data.chat_messages_field to be set"
            ));
        }
        if !data.system_excludes_any.is_empty() && !has_system_source {
            return Err(anyhow!(
                "data.system_excludes_any requires data.system_field or data.chat_messages_field to be set"
            ));
        }
        if data
            .system_contains_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            return Err(anyhow!(
                "data.system_contains_any entries must not be empty"
            ));
        }
        if data
            .system_excludes_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            return Err(anyhow!(
                "data.system_excludes_any entries must not be empty"
            ));
        }
        for filter in data
            .field_regex_contains_any
            .iter()
            .chain(data.field_regex_excludes_any.iter())
        {
            if filter.pattern.is_empty() {
                return Err(anyhow!(
                    "data field regex filter pattern entries must not be empty"
                ));
            }
            Regex::new(&filter.pattern).map_err(|error| {
                anyhow!(
                    "data field regex filter invalid regex pattern {:?}: {error}",
                    filter.pattern
                )
            })?;
            if matches!(filter.field, FieldReplacementTarget::System) && !has_system_source {
                return Err(anyhow!(
                    "data field regex filters targeting system require data.system_field, data.chat_messages_field, or a system field_default to be set"
                ));
            }
        }
        if data
            .chat_messages_field
            .as_ref()
            .is_some_and(|field| field.trim().is_empty())
        {
            return Err(anyhow!(
                "data.chat_messages_field must not be empty when set"
            ));
        }
        if let Some(min_instruction_chars) = data.min_instruction_chars {
            if min_instruction_chars == 0 {
                return Err(anyhow!(
                    "data.min_instruction_chars must be greater than zero when set"
                ));
            }
        }
        if let Some(max_instruction_chars) = data.max_instruction_chars {
            if max_instruction_chars == 0 {
                return Err(anyhow!(
                    "data.max_instruction_chars must be greater than zero when set"
                ));
            }
            if data
                .min_instruction_chars
                .is_some_and(|min_instruction_chars| max_instruction_chars < min_instruction_chars)
            {
                return Err(anyhow!(
                    "data.max_instruction_chars must be greater than or equal to data.min_instruction_chars"
                ));
            }
        }
        if let Some(min_input_chars) = data.min_input_chars {
            if min_input_chars == 0 {
                return Err(anyhow!(
                    "data.min_input_chars must be greater than zero when set"
                ));
            }
        }
        if let Some(max_input_chars) = data.max_input_chars {
            if max_input_chars == 0 {
                return Err(anyhow!(
                    "data.max_input_chars must be greater than zero when set"
                ));
            }
            if data
                .min_input_chars
                .is_some_and(|min_input_chars| max_input_chars < min_input_chars)
            {
                return Err(anyhow!(
                    "data.max_input_chars must be greater than or equal to data.min_input_chars"
                ));
            }
        }
        if let Some(min_system_chars) = data.min_system_chars {
            if !has_system_source {
                return Err(anyhow!(
                    "data.min_system_chars requires data.system_field or data.chat_messages_field to be set"
                ));
            }
            if min_system_chars == 0 {
                return Err(anyhow!(
                    "data.min_system_chars must be greater than zero when set"
                ));
            }
        }
        if let Some(max_system_chars) = data.max_system_chars {
            if !has_system_source {
                return Err(anyhow!(
                    "data.max_system_chars requires data.system_field or data.chat_messages_field to be set"
                ));
            }
            if max_system_chars == 0 {
                return Err(anyhow!(
                    "data.max_system_chars must be greater than zero when set"
                ));
            }
            if data
                .min_system_chars
                .is_some_and(|min_system_chars| max_system_chars < min_system_chars)
            {
                return Err(anyhow!(
                    "data.max_system_chars must be greater than or equal to data.min_system_chars"
                ));
            }
        }
        if let Some(min_prompt_chars) = data.min_prompt_chars {
            if min_prompt_chars == 0 {
                return Err(anyhow!(
                    "data.min_prompt_chars must be greater than zero when set"
                ));
            }
        }
        if let Some(max_prompt_chars) = data.max_prompt_chars {
            if max_prompt_chars == 0 {
                return Err(anyhow!(
                    "data.max_prompt_chars must be greater than zero when set"
                ));
            }
            if data
                .min_prompt_chars
                .is_some_and(|min_prompt_chars| max_prompt_chars < min_prompt_chars)
            {
                return Err(anyhow!(
                    "data.max_prompt_chars must be greater than or equal to data.min_prompt_chars"
                ));
            }
        }
        if let Some(min_sample_chars) = data.min_sample_chars {
            if min_sample_chars == 0 {
                return Err(anyhow!(
                    "data.min_sample_chars must be greater than zero when set"
                ));
            }
        }
        if let Some(max_sample_chars) = data.max_sample_chars {
            if max_sample_chars == 0 {
                return Err(anyhow!(
                    "data.max_sample_chars must be greater than zero when set"
                ));
            }
            if data
                .min_sample_chars
                .is_some_and(|min_sample_chars| max_sample_chars < min_sample_chars)
            {
                return Err(anyhow!(
                    "data.max_sample_chars must be greater than or equal to data.min_sample_chars"
                ));
            }
        }
        for replacement in &data.field_replacements {
            if replacement.pattern.is_empty() {
                return Err(anyhow!(
                    "data.field_replacements pattern entries must not be empty"
                ));
            }
            if matches!(replacement.field, FieldReplacementTarget::System) && !has_system_source {
                return Err(anyhow!(
                    "data.field_replacements targeting system requires data.system_field or data.chat_messages_field to be set"
                ));
            }
        }
        for replacement in &data.field_regex_replacements {
            if replacement.pattern.is_empty() {
                return Err(anyhow!(
                    "data.field_regex_replacements pattern entries must not be empty"
                ));
            }
            Regex::new(&replacement.pattern).map_err(|error| {
                anyhow!(
                    "data.field_regex_replacements invalid regex pattern {:?}: {error}",
                    replacement.pattern
                )
            })?;
            if matches!(replacement.field, FieldReplacementTarget::System) && !has_system_source {
                return Err(anyhow!(
                    "data.field_regex_replacements targeting system requires data.system_field, data.chat_messages_field, or a system field_default to be set"
                ));
            }
        }
        for default in &data.field_defaults {
            if default.value.is_empty() {
                return Err(anyhow!(
                    "data.field_defaults value entries must not be empty"
                ));
            }
        }
        for transform in &data.field_case_transforms {
            if matches!(transform.field, FieldReplacementTarget::System) && !has_system_source {
                return Err(anyhow!(
                    "data.field_case_transforms targeting system requires data.system_field, data.chat_messages_field, or a system field_default to be set"
                ));
            }
        }
        for affix in &data.field_affixes {
            if affix.prefix.is_empty() && affix.suffix.is_empty() {
                return Err(anyhow!(
                    "data.field_affixes entries must set prefix, suffix, or both"
                ));
            }
            if matches!(affix.field, FieldReplacementTarget::System) && !has_system_source {
                return Err(anyhow!(
                    "data.field_affixes targeting system requires data.system_field, data.chat_messages_field, or a system field_default to be set"
                ));
            }
        }
        for strip in &data.field_strips {
            if strip.prefix.is_empty() && strip.suffix.is_empty() {
                return Err(anyhow!(
                    "data.field_strips entries must set prefix, suffix, or both"
                ));
            }
            if matches!(strip.field, FieldReplacementTarget::System) && !has_system_source {
                return Err(anyhow!(
                    "data.field_strips targeting system requires data.system_field, data.chat_messages_field, or a system field_default to be set"
                ));
            }
        }
        for split in &data.field_splits {
            if split.delimiter.is_empty() {
                return Err(anyhow!(
                    "data.field_splits delimiter entries must not be empty"
                ));
            }
            if matches!(split.field, FieldReplacementTarget::System) && !has_system_source {
                return Err(anyhow!(
                    "data.field_splits targeting system requires data.system_field, data.chat_messages_field, or a system field_default to be set"
                ));
            }
        }
        for truncation in &data.field_truncations {
            if truncation.max_chars == 0 {
                return Err(anyhow!(
                    "data.field_truncations max_chars entries must be greater than zero"
                ));
            }
            if matches!(truncation.field, FieldReplacementTarget::System) && !has_system_source {
                return Err(anyhow!(
                    "data.field_truncations targeting system requires data.system_field, data.chat_messages_field, or a system field_default to be set"
                ));
            }
        }
        for transform in &data.field_transforms {
            match transform.op {
                FieldTransformOp::Default => {
                    if matches!(transform.field, FieldReplacementTarget::All) {
                        return Err(anyhow!(
                            "data.field_transforms default op cannot target all"
                        ));
                    }
                    if transform.value.is_empty() {
                        return Err(anyhow!(
                            "data.field_transforms default op requires non-empty value"
                        ));
                    }
                }
                FieldTransformOp::Replace => {
                    if transform.pattern.is_empty() {
                        return Err(anyhow!(
                            "data.field_transforms replace op requires non-empty pattern"
                        ));
                    }
                }
                FieldTransformOp::RegexReplace => {
                    if transform.pattern.is_empty() {
                        return Err(anyhow!(
                            "data.field_transforms regex_replace op requires non-empty pattern"
                        ));
                    }
                    Regex::new(&transform.pattern).map_err(|error| {
                        anyhow!(
                            "data.field_transforms invalid regex_replace pattern {:?}: {error}",
                            transform.pattern
                        )
                    })?;
                }
                FieldTransformOp::Case => {
                    if transform.case.is_none() {
                        return Err(anyhow!("data.field_transforms case op requires case"));
                    }
                }
                FieldTransformOp::Affix => {
                    if transform.prefix.is_empty() && transform.suffix.is_empty() {
                        return Err(anyhow!(
                            "data.field_transforms affix op requires prefix, suffix, or both"
                        ));
                    }
                }
                FieldTransformOp::Strip => {
                    if transform.prefix.is_empty() && transform.suffix.is_empty() {
                        return Err(anyhow!(
                            "data.field_transforms strip op requires prefix, suffix, or both"
                        ));
                    }
                }
                FieldTransformOp::Split => {
                    if transform.delimiter.is_empty() {
                        return Err(anyhow!(
                            "data.field_transforms split op requires non-empty delimiter"
                        ));
                    }
                    if transform.side.is_none() {
                        return Err(anyhow!("data.field_transforms split op requires side"));
                    }
                }
                FieldTransformOp::Truncate => {
                    if transform.max_chars.is_none_or(|max_chars| max_chars == 0) {
                        return Err(anyhow!(
                            "data.field_transforms truncate op requires max_chars greater than zero"
                        ));
                    }
                }
            }
            if matches!(transform.field, FieldReplacementTarget::System) && !has_system_source {
                return Err(anyhow!(
                    "data.field_transforms targeting system requires data.system_field, data.chat_messages_field, or a system field_default/field_transform default to be set"
                ));
            }
        }
        for path in &data.external_metadata_paths {
            if path.as_os_str().is_empty() {
                return Err(anyhow!(
                    "data.external_metadata_paths entries must not be empty"
                ));
            }
            if !path.exists() {
                return Err(anyhow!(
                    "data.external_metadata_paths entry does not exist: {}",
                    path.display()
                ));
            }
            if !path.is_file() {
                return Err(anyhow!(
                    "data.external_metadata_paths entry must be a file: {}",
                    path.display()
                ));
            }
        }
    }

    Ok(())
}

pub fn prepare_run_directory(run: &RunConfig) -> Result<RunPaths> {
    let run_id = std::env::var("RUSTRAIN_RUN_ID").ok();
    let root = resolve_run_root(run, run_id.as_deref())?;
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

pub fn prepare_rank_log_directory(
    run_paths: &RunPaths,
    rank: usize,
    world_size: usize,
) -> Result<PathBuf> {
    if world_size == 0 || rank >= world_size {
        return Err(anyhow!(
            "invalid launcher rank {rank} for WORLD_SIZE={world_size}"
        ));
    }
    let log_dir = if world_size > 1 {
        run_paths.logs.join(format!("rank-{rank:05}"))
    } else {
        run_paths.logs.clone()
    };
    fs::create_dir_all(&log_dir)
        .with_context(|| format!("failed to create {}", log_dir.display()))?;
    Ok(log_dir)
}

fn resolve_run_root(run: &RunConfig, run_id: Option<&str>) -> Result<PathBuf> {
    let suffix = match run_id {
        Some(run_id) => {
            if run_id.is_empty()
                || !run_id
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
            {
                return Err(anyhow!(
                    "RUSTRAIN_RUN_ID must contain only ASCII letters, digits, '.', '_', or '-'"
                ));
            }
            run_id.to_string()
        }
        None => Local::now().format("%Y%m%d-%H%M%S").to_string(),
    };
    Ok(run.base_dir.join(format!("{}-{suffix}", run.name)))
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

fn default_data_shuffle() -> bool {
    true
}

fn default_instruction_field() -> String {
    "instruction".to_string()
}

fn default_input_field() -> String {
    "input".to_string()
}

fn default_response_field() -> String {
    "response".to_string()
}

fn default_prompt_template() -> String {
    "Instruction:\n{instruction}\n\nResponse:\n".to_string()
}

fn default_prompt_with_input_template() -> String {
    "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string()
}

fn default_trim_fields() -> bool {
    true
}

fn default_min_response_chars() -> usize {
    1
}

fn default_eval_every() -> u64 {
    0
}

fn default_lr_scheduler() -> LrScheduler {
    LrScheduler::Constant
}

fn default_backend() -> BackendKind {
    BackendKind::NdArray
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_run_id_resolves_the_same_rank_output_directory() {
        let run = RunConfig {
            name: "distributed".into(),
            base_dir: "/tmp/rustrain-tests".into(),
            seed: 1,
        };

        let rank_zero = resolve_run_root(&run, Some("launch-123")).unwrap();
        let rank_one = resolve_run_root(&run, Some("launch-123")).unwrap();

        assert_eq!(rank_zero, rank_one);
        assert_eq!(
            rank_zero,
            PathBuf::from("/tmp/rustrain-tests/distributed-launch-123")
        );
    }

    #[test]
    fn shared_run_id_rejects_path_components() {
        let run = RunConfig {
            name: "distributed".into(),
            base_dir: "/tmp/rustrain-tests".into(),
            seed: 1,
        };

        let error = resolve_run_root(&run, Some("../other-run")).unwrap_err();
        assert!(error.to_string().contains("RUSTRAIN_RUN_ID"));
    }

    #[test]
    fn distributed_rank_logs_are_isolated_under_shared_run_root() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rustrain-rank-logs-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        let run_paths = RunPaths {
            root: root.clone(),
            checkpoints: root.join("checkpoints"),
            logs: root.join("logs"),
            cache: root.join("cache"),
            resolved_config: root.join("resolved_config.toml"),
        };

        let rank_zero = prepare_rank_log_directory(&run_paths, 0, 2).unwrap();
        let rank_one = prepare_rank_log_directory(&run_paths, 1, 2).unwrap();

        assert_ne!(rank_zero, rank_one);
        assert_eq!(rank_zero, root.join("logs/rank-00000"));
        assert_eq!(rank_one, root.join("logs/rank-00001"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn qwen_lora_sft_global_batch_matches_gradient_accumulation() {
        let mut config = qwen_lora_sft_config();

        validate_config(&config).expect("matching accumulated global batch should validate");

        config.train.global_batch_size = 2;
        let error = validate_config(&config).expect_err("mismatched global batch should fail");
        assert!(error.to_string().contains(
            "qwen_lora_sft requires global_batch_size = micro_batch_size * gradient_accumulation_steps"
        ));
    }

    #[test]
    fn cuda_memory_fraction_must_be_finite_and_bounded() {
        let mut config = qwen_lora_sft_config();
        config.train.cuda_memory_fraction = f64::NAN;
        assert!(validate_config(&config)
            .expect_err("NaN CUDA memory fraction must fail")
            .to_string()
            .contains("cuda_memory_fraction"));
        config.train.cuda_memory_fraction = 1.01;
        assert!(validate_config(&config)
            .expect_err("CUDA memory fraction above one must fail")
            .to_string()
            .contains("cuda_memory_fraction"));
    }

    #[test]
    fn qwen36_lora_sft_accepts_native_projection_targets() {
        let mut config = qwen_lora_sft_config();
        config.model.name = "qwen3_6_test".to_string();
        config.lora.as_mut().unwrap().target_modules = vec![
            "q_proj".to_string(),
            "in_proj_qkv".to_string(),
            "in_proj_z".to_string(),
            "out_proj".to_string(),
        ];

        config.model.architecture = "qwen3_6_lora_sft".to_string();
        validate_config(&config).expect("native Qwen3.6 LoRA targets should validate");
        config.model.architecture = "qwen3_5_lora_sft".to_string();
        validate_config(&config).expect("native Qwen3.5 LoRA targets should validate");
    }

    #[test]
    fn qwen_hybrid_lora_sft_accepts_replicated_data_parallelism() {
        let mut config = qwen_lora_sft_config();
        config.model.architecture = "qwen3_6_lora_sft".to_string();
        config.parallel.data_parallel_size = 2;
        config.train.global_batch_size = config.train.micro_batch_size
            * config.train.gradient_accumulation_steps
            * config.parallel.data_parallel_size;
        validate_config(&config).expect("Qwen3.6 LoRA DP should validate");

        config.model.architecture = "qwen3_5_lora_sft".to_string();
        validate_config(&config).expect("Qwen3.5 LoRA DP should validate");

        config.train.global_batch_size /= 2;
        let error = validate_config(&config).expect_err("DP global batch mismatch should fail");
        assert!(error.to_string().contains("data_parallel_size"));
    }

    #[test]
    fn qwen_hybrid_lora_sft_ep_accepts_tensor_expert_data_parallelism() {
        let mut config = qwen_lora_sft_config();
        config.model.architecture = "qwen3_6_lora_sft_ep".to_string();
        config.parallel.tensor_model_parallel_size = 2;
        config.parallel.expert_model_parallel_size = 2;
        config.parallel.data_parallel_size = 2;
        let ep_source_sharded = std::env::var("QWEN36_EP_A2A_SHARDED")
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false);
        config.train.global_batch_size = config.train.micro_batch_size
            * config.train.gradient_accumulation_steps
            * qwen_source_parallel_factor(true, true, ep_source_sharded, &config.parallel);
        assert_eq!(
            qwen_source_parallel_factor(true, true, true, &config.parallel),
            4
        );
        assert_eq!(
            qwen_source_parallel_factor(true, true, false, &config.parallel),
            2
        );
        validate_config(&config).expect("Qwen3.6 LoRA TPxEPxDP should validate");

        config.model.architecture = "qwen3_5_lora_sft_ep".to_string();
        validate_config(&config).expect("Qwen3.5 LoRA TPxEPxDP should validate");

        config.parallel.pipeline_model_parallel_size = 2;
        validate_config(&config).expect("Qwen3.5 LoRA TPxEPxDPxPP should validate");

        config.parallel.context_parallel_size = 2;
        let error = validate_config(&config).expect_err("Qwen TPxEPxDPxPPxCP should fail early");
        assert!(error.to_string().contains("context_parallel_size"));
    }

    #[test]
    fn data_max_samples_must_be_positive_when_set() {
        let mut config = qwen_lora_sft_config();
        config.data.as_mut().unwrap().max_samples = Some(0);

        let error = validate_config(&config).expect_err("zero max_samples should fail");

        assert!(
            error
                .to_string()
                .contains("data.max_samples must be greater than zero")
        );
    }

    #[test]
    fn data_field_regex_replacements_must_be_valid_regexes() {
        let mut config = qwen_lora_sft_config();
        config
            .data
            .as_mut()
            .unwrap()
            .field_regex_replacements
            .push(FieldRegexReplacement {
                field: FieldReplacementTarget::Instruction,
                pattern: "[".to_string(),
                replacement: "x".to_string(),
            });

        let error = validate_config(&config).expect_err("invalid regex should fail");

        assert!(
            error
                .to_string()
                .contains("data.field_regex_replacements invalid regex pattern")
        );
    }

    #[test]
    fn data_field_transforms_must_be_valid() {
        let mut config = qwen_lora_sft_config();
        config
            .data
            .as_mut()
            .unwrap()
            .field_transforms
            .push(FieldTransform {
                field: FieldReplacementTarget::Instruction,
                op: FieldTransformOp::RegexReplace,
                pattern: "[".to_string(),
                ..FieldTransform::default()
            });

        let error = validate_config(&config).expect_err("invalid regex should fail");

        assert!(
            error
                .to_string()
                .contains("data.field_transforms invalid regex_replace pattern")
        );
    }

    #[test]
    fn data_field_regex_filters_must_be_valid_regexes() {
        let mut config = qwen_lora_sft_config();
        config
            .data
            .as_mut()
            .unwrap()
            .field_regex_contains_any
            .push(FieldRegexFilter {
                field: FieldReplacementTarget::Instruction,
                pattern: "[".to_string(),
            });

        let error = validate_config(&config).expect_err("invalid regex should fail");

        assert!(
            error
                .to_string()
                .contains("data field regex filter invalid regex pattern")
        );
    }

    fn qwen_lora_sft_config() -> Config {
        Config {
            run: RunConfig {
                name: "qwen_lora_sft_test".to_string(),
                base_dir: "/tmp/rustrain-runs".into(),
                seed: 0,
            },
            model: ModelConfig {
                name: "qwen2_5_0_5b_lora_sft".to_string(),
                architecture: "qwen_lora_sft".to_string(),
                model_path: Some("/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct".into()),
                vocab_size: 151936,
                hidden_size: 896,
                num_layers: 24,
                num_attention_heads: 14,
                num_key_value_heads: 2,
                intermediate_size: 4864,
                seq_len: 32,
                norm: "rmsnorm".to_string(),
                activation: "swiglu".to_string(),
                rope: true,
                rms_norm_eps: 1e-6,
                trainable_layers: None,
            },
            train: TrainConfig {
                max_steps: 2,
                resume_from: None,
                backend: BackendKind::Tch,
                micro_batch_size: 2,
                global_batch_size: 4,
                gradient_accumulation_steps: 2,
                learning_rate: 100.0,
                mtp_loss_scaling_factor: 0.1,
                weight_decay: 0.0,
                adam_beta1: 0.9,
                adam_beta2: 0.999,
                adam_eps: 1e-8,
                lr_scheduler: LrScheduler::LinearDecay,
                max_grad_norm: Some(0.0001),
                dtype: DType::Fp32,
                device: Device::Cuda,
                cuda_memory_fraction: 0.95,
                predequant_expert_weights: true,
                checkpoint_every: 0,
                eval_every: 0,
            },
            data: Some(DataConfig {
                kind: DataKind::InstructionJsonl,
                paths: vec!["data/sft_toy/instructions.jsonl".into()],
                eval_paths: Vec::new(),
                train_split: 0.8,
                max_samples: None,
                max_eval_samples: None,
                shuffle: true,
                index_cache: None,
                instruction_field: default_instruction_field(),
                input_field: default_input_field(),
                response_field: default_response_field(),
                source_instruction_fields: Vec::new(),
                source_input_fields: Vec::new(),
                source_response_fields: Vec::new(),
                system_field: None,
                chat_messages_field: None,
                min_system_chars: None,
                max_system_chars: None,
                system_contains_any: Vec::new(),
                system_excludes_any: Vec::new(),
                prompt_template: default_prompt_template(),
                prompt_with_input_template: default_prompt_with_input_template(),
                trim_fields: default_trim_fields(),
                min_response_chars: default_min_response_chars(),
                max_response_chars: None,
                instruction_contains_any: Vec::new(),
                instruction_excludes_any: Vec::new(),
                response_contains_any: Vec::new(),
                response_excludes_any: Vec::new(),
                input_contains_any: Vec::new(),
                input_excludes_any: Vec::new(),
                field_regex_contains_any: Vec::new(),
                field_regex_excludes_any: Vec::new(),
                min_instruction_chars: None,
                max_instruction_chars: None,
                min_input_chars: None,
                max_input_chars: None,
                min_prompt_chars: None,
                max_prompt_chars: None,
                min_sample_chars: None,
                max_sample_chars: None,
                dedupe_samples: false,
                field_replacements: Vec::new(),
                field_regex_replacements: Vec::new(),
                normalize_whitespace: false,
                field_defaults: Vec::new(),
                field_case_transforms: Vec::new(),
                field_affixes: Vec::new(),
                field_strips: Vec::new(),
                field_splits: Vec::new(),
                field_truncations: Vec::new(),
                field_transforms: Vec::new(),
                source_weights: Vec::new(),
                source_max_samples: Vec::new(),
                skip_invalid_records: false,
                external_metadata_paths: Vec::new(),
            }),
            lora: Some(LoraConfig {
                rank: 4,
                alpha: 8.0,
                target_layers: vec![0, 1],
                target_modules: vec![
                    "q_proj".to_string(),
                    "k_proj".to_string(),
                    "v_proj".to_string(),
                    "o_proj".to_string(),
                    "gate_proj".to_string(),
                    "up_proj".to_string(),
                    "down_proj".to_string(),
                ],
            }),
            parallel: ParallelConfig {
                tensor_model_parallel_size: 1,
                pipeline_model_parallel_size: 1,
                data_parallel_size: 1,
                expert_model_parallel_size: 1,
                context_parallel_size: 1,
            },
        }
    }
}
