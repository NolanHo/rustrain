//! Qwen3.6 config parsing — reads `config.json` (multimodal) → `Qwen36RuntimeConfig`

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// Layer type enumeration for hybrid attention dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

impl std::fmt::Display for LayerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FullAttention => write!(f, "full_attention"),
            Self::LinearAttention => write!(f, "linear_attention"),
        }
    }
}

impl std::str::FromStr for LayerType {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "full_attention" => Ok(Self::FullAttention),
            "linear_attention" => Ok(Self::LinearAttention),
            other => bail!("unknown layer_type: {other}"),
        }
    }
}

/// Runtime config for Qwen3..6 text model (parsed from `config.json` text_config).
#[derive(Debug, Clone)]
pub struct Qwen36RuntimeConfig {
    // --- Core ---
    pub num_hidden_layers: usize,
    pub hidden_size: i64,
    pub vocab_size: i64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub hidden_act: String,

    // --- Layer types (hybrid) ---
    pub layer_types: Vec<LayerType>,
    pub full_attention_interval: usize,

    // --- Full attention ---
    pub num_attention_heads: i64,
    pub num_key_value_heads: i64,
    pub head_dim: i64,
    pub attention_bias: bool,
    pub attn_output_gate: bool,

    // --- RoPE ---
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub mrope_interleaved: bool,
    pub mrope_section: Vec<usize>,

    // --- Linear attention (Gated Delta Rule) ---
    pub linear_num_key_heads: i64,
    pub linear_key_head_dim: i64,
    pub linear_num_value_heads: i64,
    pub linear_value_head_dim: i64,
    pub linear_conv_kernel_dim: i64,
    pub mamba_ssm_dtype: String,

    // --- MoE (zero for dense models) ---
    pub is_moe: bool,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: i64,
    pub shared_expert_intermediate_size: i64,
    pub norm_topk_prob: bool,
    pub router_aux_loss_coef: f64,
    // --- Dense MLP (used when is_moe=false) ---
    pub intermediate_size: i64,

    // --- MTP ---
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,

    // --- Vision ---
    pub has_vision: bool,
    pub vision_depth: usize,
    pub vision_hidden_size: i64,
    pub vision_num_heads: i64,
    pub vision_patch_size: i64,
    pub vision_spatial_merge_size: i64,
    pub vision_temporal_patch_size: i64,
    pub vision_out_hidden_size: i64,

    // --- Weight prefix ---
    pub weight_prefix: String,
}

// --- Raw serde structs matching config.json ---

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    text_config: Option<TextConfig>,
    #[serde(default)]
    vision_config: Option<VisionConfig>,
    // Also allow direct (non-nested) config — if text_config is absent,
    // fields are at top level.
    #[serde(flatten)]
    flat: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    partial_rotary_factor: f64,
    #[serde(default)]
    mrope_interleaved: bool,
    #[serde(default)]
    mrope_section: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    #[serde(default)]
    model_type: Option<String>,
    num_hidden_layers: usize,
    hidden_size: i64,
    vocab_size: i64,
    #[serde(default = "default_one_e6")]
    rms_norm_eps: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default = "default_silu")]
    hidden_act: String,

    #[serde(default)]
    layer_types: Vec<String>,
    #[serde(default = "default_full_attention_interval")]
    full_attention_interval: usize,

    num_attention_heads: i64,
    #[serde(default)]
    num_key_value_heads: i64,
    #[serde(default)]
    head_dim: i64,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    attn_output_gate: bool,

    #[serde(default)]
    rope_parameters: Option<RopeParameters>,

    #[serde(default)]
    linear_num_key_heads: i64,
    #[serde(default)]
    linear_key_head_dim: i64,
    #[serde(default)]
    linear_num_value_heads: i64,
    #[serde(default)]
    linear_value_head_dim: i64,
    #[serde(default = "default_conv_kernel")]
    linear_conv_kernel_dim: i64,
    #[serde(default = "default_float32")]
    mamba_ssm_dtype: String,

    #[serde(default)]
    num_experts: usize,
    #[serde(default)]
    num_experts_per_tok: usize,
    #[serde(default)]
    moe_intermediate_size: i64,
    #[serde(default)]
    shared_expert_intermediate_size: i64,
    #[serde(default = "default_true")]
    norm_topk_prob: bool,
    #[serde(default = "default_router_aux")]
    router_aux_loss_coef: f64,
    #[serde(default)]
    intermediate_size: i64,

    #[serde(default)]
    mtp_num_hidden_layers: usize,
    #[serde(default)]
    mtp_use_dedicated_embeddings: bool,
}

#[derive(Debug, Deserialize)]
struct VisionConfig {
    #[serde(default = "default_vision_depth")]
    depth: usize,
    #[serde(default = "default_vision_hidden")]
    hidden_size: i64,
    #[serde(default = "default_vision_heads")]
    num_heads: i64,
    #[serde(default = "default_patch_size")]
    patch_size: i64,
    #[serde(default = "default_merge_size")]
    spatial_merge_size: i64,
    #[serde(default = "default_temporal_patch")]
    temporal_patch_size: i64,
    #[serde(default = "default_out_hidden")]
    out_hidden_size: i64,
}

// --- Defaults ---

fn default_true() -> bool { true }
fn default_one_e6() -> f64 { 1e-6 }
fn default_silu() -> String { "silu".to_string() }
fn default_full_attention_interval() -> usize { 4 }
fn default_rope_theta() -> f64 { 1_000_000.0 }
fn default_partial_rotary() -> f64 { 1.0 }
fn default_conv_kernel() -> i64 { 4 }
fn default_float32() -> String { "float32".to_string() }
fn default_router_aux() -> f64 { 0.001 }
fn default_vision_depth() -> usize { 27 }
fn default_vision_hidden() -> i64 { 1152 }
fn default_vision_heads() -> i64 { 16 }
fn default_patch_size() -> i64 { 16 }
fn default_merge_size() -> i64 { 2 }
fn default_temporal_patch() -> i64 { 2 }
fn default_out_hidden() -> i64 { 2048 }

/// Read `config.json` from a model directory and produce a `Qwen36RuntimeConfig`.
pub fn read_qwen36_runtime_config(model_path: &Path) -> Result<Qwen36RuntimeConfig> {
    let config_path = model_path.join("config.json");
    let contents = fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let raw: RawConfig = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;

    // Prefer nested text_config (multimodal), fall back to flat fields.
    let tc: TextConfig = if let Some(tc) = raw.text_config {
        tc
    } else {
        serde_json::from_value(raw.flat.clone())
            .with_context(|| "config.json has no text_config and flat parse failed")?
    };

    let rope = tc.rope_parameters.unwrap_or(RopeParameters {
        rope_theta: 1_000_000.0,
        partial_rotary_factor: 1.0,
        mrope_interleaved: false,
        mrope_section: vec![],
    });

    let layer_types: Vec<LayerType> = if tc.layer_types.is_empty() {
        // Default: all full attention
        vec![LayerType::FullAttention; tc.num_hidden_layers]
    } else {
        tc.layer_types
            .iter()
            .map(|s| s.parse::<LayerType>())
            .collect::<Result<Vec<_>>>()?
    };

    let has_vision = raw.vision_config.is_some();
    let vc = raw.vision_config.unwrap_or(VisionConfig {
        depth: 0,
        hidden_size: 0,
        num_heads: 0,
        patch_size: 0,
        spatial_merge_size: 0,
        temporal_patch_size: 0,
        out_hidden_size: 0,
    });

    // Weight prefix: "model.language_model." for multimodal, "model." for text-only.
    let weight_prefix = if has_vision {
        "model.language_model.".to_string()
    } else {
        "model.".to_string()
    };

    Ok(Qwen36RuntimeConfig {
        num_hidden_layers: tc.num_hidden_layers,
        hidden_size: tc.hidden_size,
        vocab_size: tc.vocab_size,
        rms_norm_eps: tc.rms_norm_eps,
        tie_word_embeddings: tc.tie_word_embeddings,
        hidden_act: tc.hidden_act,
        layer_types,
        full_attention_interval: tc.full_attention_interval,
        num_attention_heads: tc.num_attention_heads,
        num_key_value_heads: tc.num_key_value_heads,
        head_dim: tc.head_dim,
        attention_bias: tc.attention_bias,
        attn_output_gate: tc.attn_output_gate,
        rope_theta: rope.rope_theta,
        partial_rotary_factor: rope.partial_rotary_factor,
        mrope_interleaved: rope.mrope_interleaved,
        mrope_section: rope.mrope_section,
        linear_num_key_heads: tc.linear_num_key_heads,
        linear_key_head_dim: tc.linear_key_head_dim,
        linear_num_value_heads: tc.linear_num_value_heads,
        linear_value_head_dim: tc.linear_value_head_dim,
        linear_conv_kernel_dim: tc.linear_conv_kernel_dim,
        mamba_ssm_dtype: tc.mamba_ssm_dtype,
        num_experts: tc.num_experts,
        num_experts_per_tok: tc.num_experts_per_tok,
        moe_intermediate_size: tc.moe_intermediate_size,
        shared_expert_intermediate_size: tc.shared_expert_intermediate_size,
        is_moe: tc.model_type.as_deref() == Some("qwen3_5_moe") || tc.model_type.as_deref() == Some("qwen3_5_moe_text") || tc.num_experts > 0,
        norm_topk_prob: tc.norm_topk_prob,
        router_aux_loss_coef: tc.router_aux_loss_coef,
        intermediate_size: tc.intermediate_size,
        mtp_num_hidden_layers: tc.mtp_num_hidden_layers,
        mtp_use_dedicated_embeddings: tc.mtp_use_dedicated_embeddings,
        has_vision,
        vision_depth: vc.depth,
        vision_hidden_size: vc.hidden_size,
        vision_num_heads: vc.num_heads,
        vision_patch_size: vc.patch_size,
        vision_spatial_merge_size: vc.spatial_merge_size,
        vision_temporal_patch_size: vc.temporal_patch_size,
        vision_out_hidden_size: vc.out_hidden_size,
        weight_prefix,
    })
}

/// Resolve a model directory path — handles HF hub cache snapshots.
pub fn resolve_qwen36_model_path(model_path: &Path) -> Result<PathBuf> {
    if qwen36_model_path_is_complete(model_path) {
        return Ok(model_path.to_path_buf());
    }
    let Some(model_dir_name) = model_path.file_name().and_then(|name| name.to_str()) else {
        bail!(
            "Qwen3.6 model path {} is incomplete and has no directory name",
            model_path.display()
        );
    };
    let Some(root) = model_path.parent() else {
        bail!(
            "Qwen3.6 model path {} is incomplete and has no parent",
            model_path.display()
        );
    };
    let hub_root = root.join("hub");
    let hub_suffix = format!("--{model_dir_name}");
    let hub_model_dirs = fs::read_dir(&hub_root)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("models--") && name.ends_with(&hub_suffix))
        })
        .collect::<Vec<_>>();
    if hub_model_dirs.is_empty() {
        bail!(
            "Qwen3.6 model path {} is incomplete and no matching HF hub cache entry found under {}",
            model_path.display(),
            hub_root.display()
        );
    }
    let mut candidates = Vec::new();
    for hub_model_dir in hub_model_dirs {
        let snapshots_dir = hub_model_dir.join("snapshots");
        if snapshots_dir.is_dir() {
            candidates.extend(
                fs::read_dir(&snapshots_dir)
                    .with_context(|| format!("failed to list {}", snapshots_dir.display()))?
                    .map(|entry| entry.map(|entry| entry.path()))
                    .collect::<std::io::Result<Vec<_>>>()
                    .with_context(|| {
                        format!("failed to read entries under {}", snapshots_dir.display())
                    })?,
            );
        }
    }
    candidates.sort();
    candidates
        .into_iter()
        .rev()
        .find(|candidate| qwen36_model_path_is_complete(candidate))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Qwen3.6 model path {} is incomplete and no complete HF hub snapshot exists",
                model_path.display()
            )
        })
}

pub fn qwen36_model_path_is_complete(model_path: &Path) -> bool {
    model_path.join("config.json").exists()
        && model_path.join("tokenizer.json").exists()
        && (model_path.join("model.safetensors").exists()
            || model_path.join("model.safetensors.index.json").exists())
}
