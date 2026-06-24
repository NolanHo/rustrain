//! model module - split from qwen_module.rs

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    io::{BufRead, BufReader, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use arrow::{
    array::{Array, LargeStringArray, RecordBatch, StringArray},
    datatypes::{DataType, SchemaRef},
    ipc::reader::{FileReader as ArrowFileReader, StreamReader as ArrowStreamReader},
};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use tch::{Device, IndexOp, Kind, Reduction, Tensor, no_grad};
use tokenizers::Tokenizer;
use tracing::info;

use rustrain_checkpoint::io::{
    delta_manifest_path, optimizer_state_path, qwen_lora_sft_adapter_manifest_path,
    read_qwen_lora_sft_resume_manifest, write_qwen_delta_manifest,
    write_qwen_lora_sft_adapter_manifest,
};
use rustrain_checkpoint::manifest::*;
use rustrain_checkpoint::safetensors::{read_safetensors_map, tensor};
use rustrain_core::runtime::{
    Config, DataConfig as RuntimeDataConfig, DataKind as RuntimeDataKind, Device as RuntimeDevice,
    FieldAffix, FieldCaseTransform, FieldCaseTransformKind, FieldDefault, FieldDefaultTarget,
    FieldRegexFilter, FieldRegexReplacement, FieldReplacement, FieldReplacementTarget, FieldSplit,
    FieldSplitSide, FieldStrip, FieldTransform, FieldTransformOp, FieldTruncation,
    LoraConfig as RuntimeLoraConfig, LrScheduler, RunPaths, load_config,
};
use rustrain_nccl::nccl as nccl_smoke;

use crate::generate::*;
use crate::lora::*;
use crate::rank::*;
use crate::session::*;
use crate::sft::*;

#[derive(Debug, Serialize)]
pub struct DiffStats {
    pub(crate) max_abs: f64,
    pub(crate) mean_abs: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct QwenRuntimeConfig {
    pub num_hidden_layers: usize,
    pub num_attention_heads: i64,
    pub num_key_value_heads: i64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct QwenModelConfig {
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: i64,
    pub(crate) num_key_value_heads: i64,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
}

#[derive(Debug, Serialize)]
pub(crate) struct TopLogit {
    pub(crate) token_id: i64,
    pub(crate) logit: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenComputeDType {
    Fp32,
    Bf16,
}

impl QwenComputeDType {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "fp32" => Ok(Self::Fp32),
            "bf16" => Ok(Self::Bf16),
            other => bail!("unsupported Qwen compute dtype {other}; expected fp32 or bf16"),
        }
    }

    pub(crate) fn kind(self) -> Kind {
        match self {
            Self::Fp32 => Kind::Float,
            Self::Bf16 => Kind::BFloat16,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
        }
    }
}

pub(crate) fn resolve_qwen_model_safetensors_path(model_safetensors: &Path) -> Result<PathBuf> {
    if model_safetensors.exists() {
        return Ok(model_safetensors.to_path_buf());
    }
    if model_safetensors.file_name().and_then(|name| name.to_str()) != Some("model.safetensors") {
        return Ok(model_safetensors.to_path_buf());
    }
    let Some(model_path) = model_safetensors.parent() else {
        return Ok(model_safetensors.to_path_buf());
    };
    Ok(resolve_qwen_model_path(model_path)?.join("model.safetensors"))
}

pub(crate) fn read_runtime_config(path: &Path) -> Result<QwenRuntimeConfig> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let config: QwenModelConfig = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(QwenRuntimeConfig {
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
    })
}

pub fn resolve_qwen_model_path(model_path: &Path) -> Result<PathBuf> {
    if qwen_model_path_is_complete(model_path) {
        return Ok(model_path.to_path_buf());
    }
    let Some(model_dir_name) = model_path.file_name().and_then(|name| name.to_str()) else {
        bail!(
            "Qwen model path {} is missing config/tokenizer/model files and has no model directory name",
            model_path.display()
        );
    };
    let Some(root) = model_path.parent() else {
        bail!(
            "Qwen model path {} is missing config/tokenizer/model files and has no parent directory",
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
            "Qwen model path {} is missing config/tokenizer/model files and no matching HF hub cache entry was found under {}",
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
        .find(|candidate| qwen_model_path_is_complete(candidate))
        .ok_or_else(|| {
            anyhow!(
                "Qwen model path {} is missing config/tokenizer/model files and no complete HF hub snapshot exists under {}",
                model_path.display(),
                hub_root.display()
            )
        })
}

pub fn qwen_model_path_is_complete(model_path: &Path) -> bool {
    model_path.join("config.json").exists()
        && model_path.join("tokenizer.json").exists()
        && model_path.join("model.safetensors").exists()
}

pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let variance = input
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    input * (variance + eps).rsqrt() * weight
}

pub fn qwen_mlp(
    input: &Tensor,
    gate_proj: &Tensor,
    up_proj: &Tensor,
    down_proj: &Tensor,
) -> Tensor {
    let gate = input.linear::<&Tensor>(gate_proj, None);
    let up = input.linear::<&Tensor>(up_proj, None);
    (gate.silu() * up).linear::<&Tensor>(down_proj, None)
}

pub struct QwenLayerWeights {
    pub(crate) input_norm: Tensor,
    pub(crate) q_proj: Tensor,
    pub(crate) q_bias: Tensor,
    pub(crate) k_proj: Tensor,
    pub(crate) k_bias: Tensor,
    pub(crate) v_proj: Tensor,
    pub(crate) v_bias: Tensor,
    pub(crate) o_proj: Tensor,
    pub(crate) post_attention_norm: Tensor,
    pub(crate) gate_proj: Tensor,
    pub(crate) up_proj: Tensor,
    pub(crate) down_proj: Tensor,
}

pub(crate) struct QwenLayerCache {
    pub(crate) key: Tensor,
    pub(crate) value: Tensor,
}

impl QwenLayerWeights {
    pub fn load(weights: &BTreeMap<String, Tensor>, layer_index: usize) -> Result<Self> {
        Self::load_with_kind(weights, layer_index, Kind::Float)
    }

    pub(crate) fn load_with_kind(
        weights: &BTreeMap<String, Tensor>,
        layer_index: usize,
        kind: Kind,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_index}");
        Ok(Self {
            input_norm: tensor(weights, &format!("{prefix}.input_layernorm.weight"))?.to_kind(kind),
            q_proj: tensor(weights, &format!("{prefix}.self_attn.q_proj.weight"))?.to_kind(kind),
            q_bias: tensor(weights, &format!("{prefix}.self_attn.q_proj.bias"))?.to_kind(kind),
            k_proj: tensor(weights, &format!("{prefix}.self_attn.k_proj.weight"))?.to_kind(kind),
            k_bias: tensor(weights, &format!("{prefix}.self_attn.k_proj.bias"))?.to_kind(kind),
            v_proj: tensor(weights, &format!("{prefix}.self_attn.v_proj.weight"))?.to_kind(kind),
            v_bias: tensor(weights, &format!("{prefix}.self_attn.v_proj.bias"))?.to_kind(kind),
            o_proj: tensor(weights, &format!("{prefix}.self_attn.o_proj.weight"))?.to_kind(kind),
            post_attention_norm: tensor(
                weights,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?
            .to_kind(kind),
            gate_proj: tensor(weights, &format!("{prefix}.mlp.gate_proj.weight"))?.to_kind(kind),
            up_proj: tensor(weights, &format!("{prefix}.mlp.up_proj.weight"))?.to_kind(kind),
            down_proj: tensor(weights, &format!("{prefix}.mlp.down_proj.weight"))?.to_kind(kind),
        })
    }

    pub(crate) fn lora_target_weight(&self, module: QwenLoraTargetModule) -> &Tensor {
        match module {
            QwenLoraTargetModule::QProj => &self.q_proj,
            QwenLoraTargetModule::KProj => &self.k_proj,
            QwenLoraTargetModule::VProj => &self.v_proj,
            QwenLoraTargetModule::OProj => &self.o_proj,
            QwenLoraTargetModule::GateProj => &self.gate_proj,
            QwenLoraTargetModule::UpProj => &self.up_proj,
            QwenLoraTargetModule::DownProj => &self.down_proj,
        }
    }
}

pub fn qwen_layer(
    input: &Tensor,
    weights: &QwenLayerWeights,
    config: &QwenRuntimeConfig,
) -> Tensor {
    let compute_kind = weights.q_proj.kind();
    let input = input.to_kind(compute_kind);
    let attention_input =
        rms_norm(&input, &weights.input_norm, config.rms_norm_eps).to_kind(compute_kind);
    let attention_output = qwen_attention(
        &attention_input,
        &weights.q_proj,
        &weights.q_bias,
        &weights.k_proj,
        &weights.k_bias,
        &weights.v_proj,
        &weights.v_bias,
        &weights.o_proj,
        config,
    );
    let after_attention = input + attention_output;
    let mlp_input = rms_norm(
        &after_attention,
        &weights.post_attention_norm,
        config.rms_norm_eps,
    )
    .to_kind(compute_kind);
    let mlp_output = qwen_mlp(
        &mlp_input,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
    );
    (after_attention + mlp_output).to_kind(compute_kind)
}

pub fn qwen_forward_from_ids(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
) -> Result<Tensor> {
    qwen_forward_from_ids_with_kind(input_ids, weights, config, Kind::Float)
}

pub(crate) fn qwen_forward_from_ids_with_kind(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    compute_kind: Kind,
) -> Result<Tensor> {
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?.to_kind(compute_kind);
    let final_norm = tensor(weights, "model.norm.weight")?.to_kind(compute_kind);
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);
    for layer_index in 0..config.num_hidden_layers {
        let layer = QwenLayerWeights::load_with_kind(weights, layer_index, compute_kind)?;
        hidden = qwen_layer(&hidden, &layer, config);
    }
    let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps).to_kind(compute_kind);
    Ok(hidden.linear::<&Tensor>(&embed_tokens, None))
}

pub(crate) fn qwen_forward_from_ids_with_lora(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    registry: &QwenLoraRegistry,
    compute_kind: Kind,
) -> Result<Tensor> {
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?.to_kind(compute_kind);
    let final_norm = tensor(weights, "model.norm.weight")?.to_kind(compute_kind);
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);
    for layer_index in 0..config.num_hidden_layers {
        let layer = QwenLayerWeights::load_with_kind(weights, layer_index, compute_kind)?;
        hidden = if let Some(adapter) = registry.adapter_for_layer(layer_index) {
            qwen_layer_with_lora(&hidden, &layer, adapter, config)
        } else {
            qwen_layer(&hidden, &layer, config)
        };
    }
    let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps).to_kind(compute_kind);
    Ok(hidden.linear::<&Tensor>(&embed_tokens, None))
}

pub(crate) fn qwen_forward_with_cache(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    past_cache: Option<Vec<QwenLayerCache>>,
) -> Result<(Tensor, Vec<QwenLayerCache>)> {
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?.to_kind(Kind::Float);
    let final_norm = tensor(weights, "model.norm.weight")?.to_kind(Kind::Float);
    let position_offset = past_cache
        .as_ref()
        .and_then(|cache| cache.first())
        .map(|layer_cache| layer_cache.key.size()[2])
        .unwrap_or(0);
    let mut past_cache = past_cache.map(|cache| cache.into_iter());
    let mut next_cache = Vec::with_capacity(config.num_hidden_layers);
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);

    for layer_index in 0..config.num_hidden_layers {
        let layer = QwenLayerWeights::load(weights, layer_index)?;
        let past_layer_cache = past_cache.as_mut().and_then(|cache| cache.next());
        let (layer_hidden, layer_cache) =
            qwen_layer_with_cache(&hidden, &layer, config, past_layer_cache, position_offset);
        hidden = layer_hidden;
        next_cache.push(layer_cache);
    }
    let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps);
    Ok((hidden.linear::<&Tensor>(&embed_tokens, None), next_cache))
}

pub(crate) fn qwen_layer_with_cache(
    input: &Tensor,
    weights: &QwenLayerWeights,
    config: &QwenRuntimeConfig,
    past_cache: Option<QwenLayerCache>,
    position_offset: i64,
) -> (Tensor, QwenLayerCache) {
    let attention_input = rms_norm(input, &weights.input_norm, config.rms_norm_eps);
    let (attention_output, cache) = qwen_attention_with_cache(
        &attention_input,
        &weights.q_proj,
        &weights.q_bias,
        &weights.k_proj,
        &weights.k_bias,
        &weights.v_proj,
        &weights.v_bias,
        &weights.o_proj,
        config,
        past_cache,
        position_offset,
    );
    let after_attention = input + attention_output;
    let mlp_input = rms_norm(
        &after_attention,
        &weights.post_attention_norm,
        config.rms_norm_eps,
    );
    let mlp_output = qwen_mlp(
        &mlp_input,
        &weights.gate_proj,
        &weights.up_proj,
        &weights.down_proj,
    );
    (after_attention + mlp_output, cache)
}

pub(crate) fn qwen_causal_lm_loss(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
) -> Result<Tensor> {
    qwen_causal_lm_loss_with_kind(input_ids, weights, config, Kind::Float)
}

pub(crate) fn qwen_causal_lm_loss_with_kind(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    compute_kind: Kind,
) -> Result<Tensor> {
    let logits = qwen_forward_from_ids_with_kind(input_ids, weights, config, compute_kind)?;
    let seq_len = input_ids.size()[1];
    let shifted_logits = logits.narrow(1, 0, seq_len - 1);
    let targets = input_ids.narrow(1, 1, seq_len - 1);
    let vocab_size = shifted_logits.size()[2];
    Ok(shifted_logits
        .reshape([-1, vocab_size])
        .log_softmax(-1, Kind::Float)
        .g_nll_loss::<&Tensor>(&targets.reshape([-1]), None, Reduction::Mean, -100))
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_attention(
    input: &Tensor,
    q_proj: &Tensor,
    q_bias: &Tensor,
    k_proj: &Tensor,
    k_bias: &Tensor,
    v_proj: &Tensor,
    v_bias: &Tensor,
    o_proj: &Tensor,
    config: &QwenRuntimeConfig,
) -> Tensor {
    let shape = input.size();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let hidden_size = shape[2];
    let head_dim = hidden_size / config.num_attention_heads;
    let kv_repeat = config.num_attention_heads / config.num_key_value_heads;

    let q = input
        .linear(q_proj, Some(q_bias))
        .reshape([batch_size, seq_len, config.num_attention_heads, head_dim])
        .transpose(1, 2);
    let k = input
        .linear(k_proj, Some(k_bias))
        .reshape([batch_size, seq_len, config.num_key_value_heads, head_dim])
        .transpose(1, 2);
    let v = input
        .linear(v_proj, Some(v_bias))
        .reshape([batch_size, seq_len, config.num_key_value_heads, head_dim])
        .transpose(1, 2);
    let (cos, sin) = rope_cos_sin(seq_len, head_dim, config.rope_theta, input.device());
    let cos = cos.to_kind(input.kind());
    let sin = sin.to_kind(input.kind());
    let q = apply_rotary(&q, &cos, &sin);
    let k = apply_rotary(&k, &cos, &sin);
    let k = repeat_kv(&k, kv_repeat);
    let v = repeat_kv(&v, kv_repeat);
    let scores = q.matmul(&k.transpose(-2, -1)) / (head_dim as f64).sqrt();
    let causal_mask = Tensor::ones([seq_len, seq_len], (Kind::Bool, input.device())).triu(1);
    let scores = scores.masked_fill(&causal_mask, f64::NEG_INFINITY);
    let probs = scores.softmax(-1, Kind::Float).to_kind(v.kind());
    let context = probs
        .matmul(&v)
        .transpose(1, 2)
        .reshape([batch_size, seq_len, hidden_size]);

    context.linear::<&Tensor>(o_proj, None)
}

pub(crate) fn qwen_attention_with_lora(
    input: &Tensor,
    weights: &QwenLayerWeights,
    adapter: &QwenAttentionLoraAdapter,
    config: &QwenRuntimeConfig,
) -> Tensor {
    let q_proj = lora_weight_or_base(
        adapter,
        QwenLoraTargetModule::QProj,
        &weights.q_proj,
        input.device(),
    );
    let k_proj = lora_weight_or_base(
        adapter,
        QwenLoraTargetModule::KProj,
        &weights.k_proj,
        input.device(),
    );
    let v_proj = lora_weight_or_base(
        adapter,
        QwenLoraTargetModule::VProj,
        &weights.v_proj,
        input.device(),
    );
    let o_proj = lora_weight_or_base(
        adapter,
        QwenLoraTargetModule::OProj,
        &weights.o_proj,
        input.device(),
    );
    qwen_attention(
        input,
        &q_proj,
        &weights.q_bias,
        &k_proj,
        &weights.k_bias,
        &v_proj,
        &weights.v_bias,
        &o_proj,
        config,
    )
}

pub(crate) fn lora_weight_or_base(
    adapter: &QwenAttentionLoraAdapter,
    module: QwenLoraTargetModule,
    base: &Tensor,
    device: Device,
) -> Tensor {
    let base = base.to_device(device);
    if adapter.modules.contains_key(&module) {
        let delta = adapter
            .delta(module, device)
            .expect("LoRA module should have a delta")
            .to_kind(base.kind());
        base + delta
    } else {
        base
    }
}

pub(crate) fn qwen_layer_with_lora(
    input: &Tensor,
    weights: &QwenLayerWeights,
    adapter: &QwenAttentionLoraAdapter,
    config: &QwenRuntimeConfig,
) -> Tensor {
    let compute_kind = weights.q_proj.kind();
    let input = input.to_kind(compute_kind);
    let device = input.device();
    let attention_input =
        rms_norm(&input, &weights.input_norm, config.rms_norm_eps).to_kind(compute_kind);
    let attention_output = qwen_attention_with_lora(&attention_input, weights, adapter, config);
    let after_attention = input + attention_output;
    let mlp_input = rms_norm(
        &after_attention,
        &weights.post_attention_norm,
        config.rms_norm_eps,
    )
    .to_kind(compute_kind);
    let mlp_output = qwen_mlp(
        &mlp_input,
        &lora_weight_or_base(
            adapter,
            QwenLoraTargetModule::GateProj,
            &weights.gate_proj,
            device,
        ),
        &lora_weight_or_base(
            adapter,
            QwenLoraTargetModule::UpProj,
            &weights.up_proj,
            device,
        ),
        &lora_weight_or_base(
            adapter,
            QwenLoraTargetModule::DownProj,
            &weights.down_proj,
            device,
        ),
    );
    (after_attention + mlp_output).to_kind(compute_kind)
}

pub(crate) fn qwen_attention_lora_mse_loss(
    input: &Tensor,
    target: &Tensor,
    weights: &QwenLayerWeights,
    adapter: &QwenAttentionLoraAdapter,
    config: &QwenRuntimeConfig,
) -> Tensor {
    qwen_attention_with_lora(input, weights, adapter, config).mse_loss(target, Reduction::Mean)
}

pub(crate) fn tensor_i64_max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<i64> {
    if actual.size() != expected.size() {
        bail!(
            "shape mismatch: actual {:?}, expected {:?}",
            actual.size(),
            expected.size()
        );
    }
    let actual_values: Vec<i64> =
        Vec::<i64>::try_from(actual.reshape([-1]).to_device(Device::Cpu))?;
    let expected_values: Vec<i64> =
        Vec::<i64>::try_from(expected.reshape([-1]).to_device(Device::Cpu))?;
    Ok(actual_values
        .iter()
        .zip(expected_values.iter())
        .map(|(actual, expected)| actual.saturating_sub(*expected).abs())
        .max()
        .unwrap_or(0))
}

pub(crate) fn tensor_max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<f64> {
    if actual.size() != expected.size() {
        bail!(
            "shape mismatch: actual {:?}, expected {:?}",
            actual.size(),
            expected.size()
        );
    }
    Ok((actual - expected)
        .abs()
        .max()
        .to_device(Device::Cpu)
        .double_value(&[]))
}

pub(crate) fn qwen_tensor_i64_fingerprint(tensor: &Tensor) -> Result<String> {
    let values: Vec<i64> = Vec::<i64>::try_from(tensor.reshape([-1]).to_device(Device::Cpu))?;
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        qwen_sft_hash_bytes(&mut hash, &value.to_le_bytes());
    }
    Ok(format!("{hash:016x}"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_attention_with_cache(
    input: &Tensor,
    q_proj: &Tensor,
    q_bias: &Tensor,
    k_proj: &Tensor,
    k_bias: &Tensor,
    v_proj: &Tensor,
    v_bias: &Tensor,
    o_proj: &Tensor,
    config: &QwenRuntimeConfig,
    past_cache: Option<QwenLayerCache>,
    position_offset: i64,
) -> (Tensor, QwenLayerCache) {
    let shape = input.size();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let hidden_size = shape[2];
    let head_dim = hidden_size / config.num_attention_heads;
    let kv_repeat = config.num_attention_heads / config.num_key_value_heads;

    let q = input
        .linear(q_proj, Some(q_bias))
        .reshape([batch_size, seq_len, config.num_attention_heads, head_dim])
        .transpose(1, 2);
    let k = input
        .linear(k_proj, Some(k_bias))
        .reshape([batch_size, seq_len, config.num_key_value_heads, head_dim])
        .transpose(1, 2);
    let v = input
        .linear(v_proj, Some(v_bias))
        .reshape([batch_size, seq_len, config.num_key_value_heads, head_dim])
        .transpose(1, 2);
    let (cos, sin) = rope_cos_sin_with_offset(
        seq_len,
        head_dim,
        config.rope_theta,
        input.device(),
        position_offset,
    );
    let cos = cos.to_kind(input.kind());
    let sin = sin.to_kind(input.kind());
    let q = apply_rotary(&q, &cos, &sin);
    let k = apply_rotary(&k, &cos, &sin);
    let (k, v) = if let Some(cache) = past_cache {
        (
            Tensor::cat(&[&cache.key, &k], 2),
            Tensor::cat(&[&cache.value, &v], 2),
        )
    } else {
        (k, v)
    };
    let cache = QwenLayerCache {
        key: k.shallow_clone(),
        value: v.shallow_clone(),
    };
    let total_seq_len = k.size()[2];
    let k_for_attention = repeat_kv(&k, kv_repeat);
    let v_for_attention = repeat_kv(&v, kv_repeat);
    let scores = q.matmul(&k_for_attention.transpose(-2, -1)) / (head_dim as f64).sqrt();
    let scores = if position_offset == 0 {
        let causal_mask =
            Tensor::ones([seq_len, total_seq_len], (Kind::Bool, input.device())).triu(1);
        scores.masked_fill(&causal_mask, f64::NEG_INFINITY)
    } else {
        scores
    };
    let probs = scores
        .softmax(-1, Kind::Float)
        .to_kind(v_for_attention.kind());
    let context =
        probs
            .matmul(&v_for_attention)
            .transpose(1, 2)
            .reshape([batch_size, seq_len, hidden_size]);

    (context.linear::<&Tensor>(o_proj, None), cache)
}

pub(crate) fn rope_cos_sin(
    seq_len: i64,
    head_dim: i64,
    theta: f64,
    device: Device,
) -> (Tensor, Tensor) {
    rope_cos_sin_with_offset(seq_len, head_dim, theta, device, 0)
}

pub(crate) fn rope_cos_sin_with_offset(
    seq_len: i64,
    head_dim: i64,
    theta: f64,
    device: Device,
    position_offset: i64,
) -> (Tensor, Tensor) {
    let half = head_dim / 2;
    let inv_freq = Tensor::arange(half, (Kind::Float, device)) * 2.0;
    let inv_freq = (-(&inv_freq / head_dim as f64) * theta.ln()).exp();
    let positions =
        (Tensor::arange(seq_len, (Kind::Float, device)) + position_offset as f64).unsqueeze(1);
    let freqs = positions.matmul(&inv_freq.unsqueeze(0));
    let emb = Tensor::cat(&[&freqs, &freqs], -1).unsqueeze(0).unsqueeze(0);
    (emb.cos(), emb.sin())
}

pub(crate) fn apply_rotary(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    input * cos + rotate_half(input) * sin
}

pub(crate) fn rotate_half(input: &Tensor) -> Tensor {
    let last_dim = input.size()[input.dim() - 1];
    let half = last_dim / 2;
    let first = input.narrow(-1, 0, half);
    let second = input.narrow(-1, half, half);
    Tensor::cat(&[&(-second), &first], -1)
}

pub(crate) fn repeat_kv(input: &Tensor, repeats: i64) -> Tensor {
    if repeats == 1 {
        input.shallow_clone()
    } else {
        input.repeat_interleave_self_int(repeats, 1, None)
    }
}

pub fn diff_stats(actual: &Tensor, expected: &Tensor) -> Result<DiffStats> {
    if actual.size() != expected.size() {
        bail!(
            "shape mismatch: actual {:?}, expected {:?}",
            actual.size(),
            expected.size()
        );
    }
    let diff = (actual - expected).abs().to_device(Device::Cpu);
    Ok(DiffStats {
        max_abs: diff.max().double_value(&[]),
        mean_abs: diff.mean(Kind::Float).double_value(&[]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_module::test_utils::*;

    #[test]
    fn rms_norm_matches_manual_formula() {
        let input = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape([1, 2, 2]);
        let weight = Tensor::from_slice(&[0.5_f32, 2.0]);
        let output = rms_norm(&input, &weight, 1e-6);

        assert_eq!(output.size(), vec![1, 2, 2]);
        assert!(output.isfinite().all().int64_value(&[]) == 1);
    }

    #[test]
    fn rotate_half_splits_head_dimension_in_halves() {
        let input = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape([1, 1, 1, 4]);
        let output = rotate_half(&input);

        let values: Vec<f32> = Vec::<f32>::try_from(output.reshape([4])).unwrap();
        assert_eq!(values, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn qwen_causal_lm_loss_is_finite_for_tiny_weights() {
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let weights = tiny_qwen_weights();
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);

        let loss = qwen_causal_lm_loss(&input_ids, &weights, &config).expect("loss should run");

        assert_eq!(loss.size(), Vec::<i64>::new());
        assert!(loss.isfinite().int64_value(&[]) == 1);
    }

    #[test]
    fn trainable_tensor_names_expand_over_configured_layers() {
        let names = qwen_trainable_tensors_for_layers(&[0, 1], true);

        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(names.contains(&"model.norm.weight".to_string()));
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.1.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.1.mlp.down_proj.weight".to_string()));
        assert_eq!(names.len(), 26);

        let dp_names = qwen_trainable_tensors_for_layers(&[0, 1], false);
        assert!(!dp_names.contains(&"model.embed_tokens.weight".to_string()));
        assert_eq!(dp_names.len(), 25);
    }

    #[test]
    fn qwen_data_epoch_metadata_tracks_wrapping_cursor() {
        assert_eq!(qwen_data_epoch_and_offset(0, 6).unwrap(), (0, 0));
        assert_eq!(qwen_data_epoch_and_offset(5, 6).unwrap(), (0, 5));
        assert_eq!(qwen_data_epoch_and_offset(6, 6).unwrap(), (1, 0));
        assert_eq!(qwen_data_epoch_and_offset(16, 6).unwrap(), (2, 4));
        assert!(qwen_data_epoch_and_offset(0, 0).is_err());
    }

    #[test]
    fn qwen_model_path_resolves_hf_hub_snapshot_when_legacy_dir_is_missing() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let legacy = temp.path().join("Qwen2.5-0.5B-Instruct");
        let incomplete_snapshot = temp
            .path()
            .join("hub")
            .join("models--Qwen--Qwen2.5-0.5B-Instruct")
            .join("snapshots")
            .join("111");
        let complete_snapshot = temp
            .path()
            .join("hub")
            .join("models--Qwen--Qwen2.5-0.5B-Instruct")
            .join("snapshots")
            .join("222");
        fs::create_dir_all(&incomplete_snapshot).expect("incomplete snapshot dir should write");
        fs::create_dir_all(&complete_snapshot).expect("complete snapshot dir should write");
        fs::write(incomplete_snapshot.join("config.json"), "{}").expect("config should write");
        fs::write(complete_snapshot.join("config.json"), "{}").expect("config should write");
        fs::write(complete_snapshot.join("tokenizer.json"), "{}").expect("tokenizer should write");
        fs::write(complete_snapshot.join("model.safetensors"), "")
            .expect("safetensors marker should write");

        let resolved =
            resolve_qwen_model_path(&legacy).expect("legacy path should resolve through HF hub");

        assert_eq!(resolved, complete_snapshot);
    }

    #[test]
    fn qwen_model_path_keeps_complete_configured_directory() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let model_path = temp.path().join("Qwen2.5-0.5B-Instruct");
        fs::create_dir_all(&model_path).expect("model dir should write");
        fs::write(model_path.join("config.json"), "{}").expect("config should write");
        fs::write(model_path.join("tokenizer.json"), "{}").expect("tokenizer should write");
        fs::write(model_path.join("model.safetensors"), "")
            .expect("safetensors marker should write");

        let resolved =
            resolve_qwen_model_path(&model_path).expect("complete path should not be rewritten");

        assert_eq!(resolved, model_path);
    }

    #[test]
    fn qwen_model_path_reports_missing_hf_snapshot() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let legacy = temp.path().join("Qwen2.5-0.5B-Instruct");
        let incomplete_snapshot = temp
            .path()
            .join("hub")
            .join("models--Qwen--Qwen2.5-0.5B-Instruct")
            .join("snapshots")
            .join("111");
        fs::create_dir_all(&incomplete_snapshot).expect("incomplete snapshot dir should write");
        fs::write(incomplete_snapshot.join("config.json"), "{}").expect("config should write");

        let error = match resolve_qwen_model_path(&legacy) {
            Ok(path) => panic!("incomplete cache should fail, resolved {}", path.display()),
            Err(error) => error.to_string(),
        };

        assert!(error.contains("no complete HF hub snapshot"));
    }

    #[test]
    fn qwen_model_safetensors_path_resolves_with_hf_hub_snapshot() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let legacy_safetensors = temp
            .path()
            .join("Qwen2.5-0.5B-Instruct")
            .join("model.safetensors");
        let complete_snapshot = temp
            .path()
            .join("hub")
            .join("models--Qwen--Qwen2.5-0.5B-Instruct")
            .join("snapshots")
            .join("222");
        fs::create_dir_all(&complete_snapshot).expect("complete snapshot dir should write");
        fs::write(complete_snapshot.join("config.json"), "{}").expect("config should write");
        fs::write(complete_snapshot.join("tokenizer.json"), "{}").expect("tokenizer should write");
        fs::write(complete_snapshot.join("model.safetensors"), "")
            .expect("safetensors marker should write");

        let resolved = resolve_qwen_model_safetensors_path(&legacy_safetensors)
            .expect("legacy safetensors path should resolve through HF hub");

        assert_eq!(resolved, complete_snapshot.join("model.safetensors"));
    }
}
