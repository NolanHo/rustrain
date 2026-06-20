use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use tch::{Device, IndexOp, Kind, Reduction, Tensor, no_grad};
use tokenizers::Tokenizer;

use crate::nccl_smoke;

#[derive(Debug, Serialize)]
pub struct DiffStats {
    max_abs: f64,
    mean_abs: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct QwenRuntimeConfig {
    pub num_hidden_layers: usize,
    pub num_attention_heads: i64,
    pub num_key_value_heads: i64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

#[derive(Debug, Serialize)]
struct QwenModuleParitySummary {
    model_safetensors: String,
    fixture: String,
    attention_diff: DiffStats,
    rms_norm_diff: DiffStats,
    mlp_diff: DiffStats,
    layer0_diff: DiffStats,
    layer1_diff: DiffStats,
}

#[derive(Debug, Deserialize)]
struct QwenModelConfig {
    num_hidden_layers: usize,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    rms_norm_eps: f64,
    rope_theta: f64,
}

#[derive(Debug, Serialize)]
struct TopLogit {
    token_id: i64,
    logit: f64,
}

#[derive(Debug, Serialize)]
struct QwenLogitsParitySummary {
    model_path: String,
    reference_fixture: String,
    input_ids: Vec<i64>,
    logits_shape: Vec<i64>,
    logits_diff: DiffStats,
    last_token_topk: Vec<TopLogit>,
}

#[derive(Debug, Serialize)]
struct QwenGenerateParitySummary {
    model_path: String,
    reference_fixture: String,
    prompt_len: usize,
    max_new_tokens: usize,
    generated_ids: Vec<i64>,
    new_token_ids: Vec<i64>,
    reference_match: bool,
}

#[derive(Debug, Serialize)]
struct QwenSamplingSmokeSummary {
    model_path: String,
    reference_fixture: String,
    prompt_len: usize,
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
    generated_ids: Vec<i64>,
    cached_ids: Vec<i64>,
    new_token_ids: Vec<i64>,
    cache_match: bool,
}

#[derive(Debug, Serialize)]
struct QwenKvCacheParitySummary {
    model_path: String,
    reference_fixture: String,
    prompt_len: usize,
    max_new_tokens: usize,
    full_context_ids: Vec<i64>,
    cached_ids: Vec<i64>,
    new_token_ids: Vec<i64>,
    reference_match: bool,
}

#[derive(Debug, Serialize)]
struct QwenLoraSmokeSummary {
    model_path: String,
    fixture: String,
    adapter_output: String,
    target_layers: Vec<usize>,
    target_modules: Vec<String>,
    rank: i64,
    alpha: f64,
    zero_lora_max_delta: f64,
    nonzero_lora_max_delta: f64,
    reload_max_delta: f64,
    trainable_tensors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct QwenLoraTrainSmokeSummary {
    model_path: String,
    fixture: String,
    adapter_output: String,
    target_layers: Vec<usize>,
    target_modules: Vec<String>,
    rank: i64,
    alpha: f64,
    learning_rate: f64,
    initial_loss: f64,
    final_loss: f64,
    reloaded_loss: f64,
    reload_delta: f64,
    base_requires_grad: bool,
    trainable_tensors: Vec<TrainableTensorSummary>,
}

#[derive(Debug, Serialize)]
struct QwenLoraSftSmokeSummary {
    model_path: String,
    adapter_output: String,
    target_layers: Vec<usize>,
    target_modules: Vec<String>,
    batch_size: usize,
    prompt_tokens: Vec<usize>,
    response_tokens: Vec<usize>,
    sequence_tokens: usize,
    response_masked_positions: usize,
    padding_tokens: usize,
    rank: i64,
    alpha: f64,
    learning_rate: f64,
    initial_loss: f64,
    final_loss: f64,
    reloaded_loss: f64,
    reload_delta: f64,
    base_requires_grad: bool,
    trainable_tensors: Vec<TrainableTensorSummary>,
}

#[derive(Debug, Serialize)]
struct QwenTiedHeadTrainSummary {
    model_path: String,
    reference_fixture: String,
    delta_output: String,
    trainable_tensor: String,
    learning_rate: f64,
    initial_loss: f64,
    final_loss: f64,
    reloaded_loss: f64,
    reload_delta: f64,
    grad_defined: bool,
    grad_norm: f64,
}

#[derive(Debug, Serialize)]
struct TrainableTensorSummary {
    name: String,
    grad_defined: bool,
    grad_norm: f64,
    delta_norm: f64,
}

#[derive(Debug, Serialize)]
struct QwenFullTrainSmokeSummary {
    model_path: String,
    reference_fixture: String,
    delta_output: String,
    optimizer_output: String,
    manifest_output: String,
    compute_kind: String,
    learning_rate: f64,
    initial_loss: f64,
    final_loss: f64,
    reloaded_loss: f64,
    reload_delta: f64,
    resume_loss: f64,
    continuous_second_loss: f64,
    resumed_second_loss: f64,
    second_step_delta: f64,
    trainable_tensors: Vec<TrainableTensorSummary>,
}

#[derive(Debug, Serialize)]
struct QwenDpGradientRankSummary {
    rank: usize,
    world_size: usize,
    local_sequence_count: usize,
    tensor_count: usize,
    max_grad_delta: f32,
    loss_delta: f64,
    local_loss: f64,
    global_loss: f64,
    expected_loss: f64,
    checkpoint_written: bool,
    checkpoint_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QwenGradSignature {
    name: String,
    shape: Vec<i64>,
    samples: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct QwenDpCheckpointManifest {
    format: String,
    writer_rank: usize,
    world_size: usize,
    tensor_count: usize,
    max_grad_delta: f32,
    expected_loss: f64,
    dtype: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct QwenDeltaCheckpointManifest {
    format: String,
    base_model_path: String,
    reference_fixture: String,
    delta_safetensors: String,
    #[serde(default)]
    optimizer_safetensors: Option<String>,
    train_step: u64,
    learning_rate: f64,
    initial_loss: f64,
    final_loss: f64,
    tensors: Vec<QwenDeltaTensorManifestEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct QwenDeltaTensorManifestEntry {
    name: String,
    delta_name: String,
    #[serde(default)]
    adam_m_name: Option<String>,
    #[serde(default)]
    adam_v_name: Option<String>,
    shape: Vec<i64>,
    dtype: String,
    grad_norm: f64,
    delta_norm: f64,
}

struct AdamSlotNames {
    m: String,
    v: String,
}

struct AdamState {
    m: Tensor,
    v: Tensor,
}

struct QwenTrainableParameter {
    name: String,
    tensor: Tensor,
    base: Tensor,
    adam: Option<AdamState>,
}

struct QwenTrainStepArtifacts {
    tensor_summaries: Vec<TrainableTensorSummary>,
    manifest_tensors: Vec<QwenDeltaTensorManifestEntry>,
    delta_entries: Vec<(String, Tensor)>,
    optimizer_entries: Vec<(String, Tensor)>,
}

struct QwenTrainableRegistry {
    parameters: Vec<QwenTrainableParameter>,
}

struct QwenTrainStepResult {
    loss_before: f64,
    loss_after: f64,
    artifacts: QwenTrainStepArtifacts,
}

struct QwenTrainableSession {
    config: QwenRuntimeConfig,
    weights: BTreeMap<String, Tensor>,
    input_ids: Tensor,
    compute_kind: Kind,
    registry: QwenTrainableRegistry,
}

struct QwenAttentionDpSession {
    config: QwenRuntimeConfig,
    input: Tensor,
    target: Tensor,
    q_proj: Tensor,
    q_bias: Tensor,
    k_proj: Tensor,
    k_bias: Tensor,
    v_proj: Tensor,
    v_bias: Tensor,
    o_proj: Tensor,
    compute_kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenComputeDType {
    Fp32,
    Bf16,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct QwenLoraConfig {
    target_layers: Vec<usize>,
    target_modules: Vec<QwenLoraTargetModule>,
    rank: i64,
    alpha: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum QwenLoraTargetModule {
    QProj,
    VProj,
}

struct QwenLoraRegistry {
    config: QwenLoraConfig,
    adapters: BTreeMap<usize, QwenAttentionLoraAdapter>,
}

#[derive(Clone)]
struct QwenSftTokenSample {
    prompt_tokens: usize,
    response_tokens: usize,
    masked_positions: usize,
    token_ids: Vec<i64>,
    mask_values: Vec<f32>,
}

struct QwenSftExample {
    instruction: String,
    input: String,
    response: String,
}

#[derive(Deserialize)]
struct QwenSftRecord {
    instruction: String,
    #[serde(default)]
    input: String,
    response: String,
}

struct QwenSftDataset {
    samples: Vec<QwenSftTokenSample>,
    pad_token_id: i64,
}

struct QwenSftBatch {
    input_ids: Tensor,
    target_mask: Tensor,
    prompt_tokens: Vec<usize>,
    response_tokens: Vec<usize>,
    masked_positions: usize,
    padding_tokens: usize,
}

impl QwenComputeDType {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "fp32" => Ok(Self::Fp32),
            "bf16" => Ok(Self::Bf16),
            other => bail!("unsupported Qwen compute dtype {other}; expected fp32 or bf16"),
        }
    }

    fn kind(self) -> Kind {
        match self {
            Self::Fp32 => Kind::Float,
            Self::Bf16 => Kind::BFloat16,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
        }
    }
}

pub fn qwen_module_parity(model_safetensors: &Path, fixture: &Path) -> Result<()> {
    let weights = read_safetensors_map(model_safetensors)?;
    let fixture_tensors = read_safetensors_map(fixture)?;
    let input = tensor(&fixture_tensors, "embedded_hidden")?.to_kind(Kind::Float);
    let attention_input = tensor(&fixture_tensors, "input_attention_normed")?.to_kind(Kind::Float);
    let expected_attention = tensor(&fixture_tensors, "attention_output")?.to_kind(Kind::Float);
    let expected_norm = tensor(&fixture_tensors, "post_attention_normed")?.to_kind(Kind::Float);
    let expected_mlp = tensor(&fixture_tensors, "mlp_output")?.to_kind(Kind::Float);
    let expected_layer0 = tensor(&fixture_tensors, "layer0_output")?.to_kind(Kind::Float);
    let expected_layer1 = tensor(&fixture_tensors, "layer1_output")?.to_kind(Kind::Float);

    let config = QwenRuntimeConfig {
        num_hidden_layers: 24,
        num_attention_heads: 14,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };
    let layer0 = QwenLayerWeights::load(&weights, 0)?;

    let actual_attention = qwen_attention(
        &attention_input,
        &layer0.q_proj,
        &layer0.q_bias,
        &layer0.k_proj,
        &layer0.k_bias,
        &layer0.v_proj,
        &layer0.v_bias,
        &layer0.o_proj,
        &config,
    );
    let actual_norm = rms_norm(&input, &layer0.post_attention_norm, config.rms_norm_eps);
    let actual_mlp = qwen_mlp(
        &actual_norm,
        &layer0.gate_proj,
        &layer0.up_proj,
        &layer0.down_proj,
    );
    let actual_layer0 = qwen_layer(&input, &layer0, &config);
    let actual_layer1 = qwen_layer(
        &actual_layer0,
        &QwenLayerWeights::load(&weights, 1)?,
        &config,
    );
    let attention_diff = diff_stats(&actual_attention, &expected_attention)?;
    let rms_norm_diff = diff_stats(&actual_norm, &expected_norm)?;
    let mlp_diff = diff_stats(&actual_mlp, &expected_mlp)?;
    let layer0_diff = diff_stats(&actual_layer0, &expected_layer0)?;
    let layer1_diff = diff_stats(&actual_layer1, &expected_layer1)?;

    if attention_diff.max_abs > 1e-4 {
        bail!(
            "attention parity failed: max_abs={}",
            attention_diff.max_abs
        );
    }
    if rms_norm_diff.max_abs > 1e-5 {
        bail!("RMSNorm parity failed: max_abs={}", rms_norm_diff.max_abs);
    }
    if mlp_diff.max_abs > 1e-4 {
        bail!("MLP parity failed: max_abs={}", mlp_diff.max_abs);
    }
    if layer0_diff.max_abs > 1e-4 {
        bail!("layer0 parity failed: max_abs={}", layer0_diff.max_abs);
    }
    if layer1_diff.max_abs > 2e-4 {
        bail!("layer1 parity failed: max_abs={}", layer1_diff.max_abs);
    }

    let summary = QwenModuleParitySummary {
        model_safetensors: model_safetensors.display().to_string(),
        fixture: fixture.display().to_string(),
        attention_diff,
        rms_norm_diff,
        mlp_diff,
        layer0_diff,
        layer1_diff,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_logits_parity(model_path: &Path, reference_fixture: &Path) -> Result<()> {
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    let expected_logits = tensor(&reference, "logits")?.to_kind(Kind::Float);
    let actual_logits = qwen_forward_from_ids(&input_ids, &weights, &config)?;
    let logits_diff = diff_stats(&actual_logits, &expected_logits)?;

    if logits_diff.max_abs > 5e-3 {
        bail!("logits parity failed: max_abs={}", logits_diff.max_abs);
    }

    let last_logits = actual_logits.i((0, -1));
    let (values, indices) = last_logits.topk(8, -1, true, true);
    let values: Vec<f32> = Vec::<f32>::try_from(values.to_device(Device::Cpu))?;
    let indices: Vec<i64> = Vec::<i64>::try_from(indices.to_device(Device::Cpu))?;
    let last_token_topk = values
        .into_iter()
        .zip(indices)
        .map(|(logit, token_id)| TopLogit {
            token_id,
            logit: f64::from(logit),
        })
        .collect();
    let input_ids_flat: Vec<i64> =
        Vec::<i64>::try_from(input_ids.reshape([-1]).to_device(Device::Cpu))?;

    let summary = QwenLogitsParitySummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        input_ids: input_ids_flat,
        logits_shape: actual_logits.size(),
        logits_diff,
        last_token_topk,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_generate_parity(model_path: &Path, reference_fixture: &Path) -> Result<()> {
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    let expected_generated = tensor(&reference, "generated_ids")?.to_kind(Kind::Int64);
    let expected_ids: Vec<i64> =
        Vec::<i64>::try_from(expected_generated.reshape([-1]).to_device(Device::Cpu))?;
    let prompt_len = input_ids.size()[1] as usize;
    if expected_ids.len() < prompt_len {
        bail!(
            "reference generated ids shorter than prompt: generated={}, prompt={prompt_len}",
            expected_ids.len()
        );
    }
    let max_new_tokens = expected_ids.len() - prompt_len;
    let generated = qwen_greedy_generate(&input_ids, &weights, &config, max_new_tokens)?;
    let generated_ids: Vec<i64> =
        Vec::<i64>::try_from(generated.reshape([-1]).to_device(Device::Cpu))?;
    let reference_match = generated_ids == expected_ids;
    if !reference_match {
        bail!(
            "greedy generation parity failed: expected {:?}, got {:?}",
            expected_ids,
            generated_ids
        );
    }
    let summary = QwenGenerateParitySummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        prompt_len,
        max_new_tokens,
        new_token_ids: generated_ids[prompt_len..].to_vec(),
        generated_ids,
        reference_match,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_sampling_smoke(
    model_path: &Path,
    reference_fixture: &Path,
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
) -> Result<()> {
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    let prompt_len = input_ids.size()[1] as usize;
    let generated = qwen_sample_generate(
        &input_ids,
        &weights,
        &config,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        seed,
    )?;
    let cached = qwen_sample_generate_with_cache(
        &input_ids,
        &weights,
        &config,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        seed,
    )?;
    let generated_ids: Vec<i64> =
        Vec::<i64>::try_from(generated.reshape([-1]).to_device(Device::Cpu))?;
    let cached_ids: Vec<i64> = Vec::<i64>::try_from(cached.reshape([-1]).to_device(Device::Cpu))?;
    let cache_match = generated_ids == cached_ids;
    if !cache_match {
        bail!(
            "cached sampling diverged from full-context sampling: full={:?}, cached={:?}",
            generated_ids,
            cached_ids
        );
    }
    let new_token_ids = generated_ids[prompt_len..].to_vec();
    if new_token_ids.len() != max_new_tokens {
        bail!(
            "sampling smoke generated {} tokens, expected {max_new_tokens}",
            new_token_ids.len()
        );
    }

    let summary = QwenSamplingSmokeSummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        prompt_len,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        seed,
        generated_ids,
        cached_ids,
        new_token_ids,
        cache_match,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_kv_cache_parity(
    model_path: &Path,
    reference_fixture: &Path,
    max_new_tokens: usize,
) -> Result<()> {
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    let prompt_len = input_ids.size()[1] as usize;
    let full_context = qwen_greedy_generate(&input_ids, &weights, &config, max_new_tokens)?;
    let cached = qwen_greedy_generate_with_cache(&input_ids, &weights, &config, max_new_tokens)?;
    let full_context_ids: Vec<i64> =
        Vec::<i64>::try_from(full_context.reshape([-1]).to_device(Device::Cpu))?;
    let cached_ids: Vec<i64> = Vec::<i64>::try_from(cached.reshape([-1]).to_device(Device::Cpu))?;
    let reference_match = full_context_ids == cached_ids;
    if !reference_match {
        bail!(
            "KV-cache greedy parity failed: full_context={:?}, cached={:?}",
            full_context_ids,
            cached_ids
        );
    }

    let summary = QwenKvCacheParitySummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        prompt_len,
        max_new_tokens,
        new_token_ids: cached_ids[prompt_len..].to_vec(),
        full_context_ids,
        cached_ids,
        reference_match,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_lora_smoke(
    model_path: &Path,
    fixture: &Path,
    adapter_output: &Path,
    rank: i64,
    alpha: f64,
) -> Result<()> {
    if rank <= 0 {
        bail!("rank must be positive");
    }
    if alpha <= 0.0 {
        bail!("alpha must be positive");
    }

    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let fixture_tensors = read_safetensors_map(fixture)?;
    let attention_input = tensor(&fixture_tensors, "input_attention_normed")?.to_kind(Kind::Float);
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let layer0 = QwenLayerWeights::load(&weights, 0)?;
    let base = qwen_attention(
        &attention_input,
        &layer0.q_proj,
        &layer0.q_bias,
        &layer0.k_proj,
        &layer0.k_bias,
        &layer0.v_proj,
        &layer0.v_bias,
        &layer0.o_proj,
        &config,
    );
    let lora_config = QwenLoraConfig::layer0_qv(rank, alpha)?;
    let zero_registry = QwenLoraRegistry::zeros(&weights, &lora_config)?;
    let zero_output = qwen_attention_with_lora(
        &attention_input,
        &layer0,
        zero_registry.layer_adapter(0)?,
        &config,
    );
    let zero_lora_max_delta = diff_stats(&zero_output, &base)?.max_abs;
    if zero_lora_max_delta > 1e-7 {
        bail!("zero LoRA changed attention output: max_delta={zero_lora_max_delta}");
    }

    let registry = QwenLoraRegistry::deterministic(&weights, &lora_config, false)?;
    let adapted_output = qwen_attention_with_lora(
        &attention_input,
        &layer0,
        registry.layer_adapter(0)?,
        &config,
    );
    let nonzero_lora_max_delta = diff_stats(&adapted_output, &base)?.max_abs;
    if nonzero_lora_max_delta <= 0.0 {
        bail!("non-zero LoRA did not change attention output");
    }

    registry.save(adapter_output)?;
    let reloaded = QwenLoraRegistry::load(adapter_output)?;
    let reloaded_output = qwen_attention_with_lora(
        &attention_input,
        &layer0,
        reloaded.layer_adapter(0)?,
        &config,
    );
    let reload_max_delta = diff_stats(&reloaded_output, &adapted_output)?.max_abs;
    if reload_max_delta > 1e-7 {
        bail!("LoRA adapter reload changed output: max_delta={reload_max_delta}");
    }

    let summary = QwenLoraSmokeSummary {
        model_path: model_path.display().to_string(),
        fixture: fixture.display().to_string(),
        adapter_output: adapter_output.display().to_string(),
        target_layers: lora_config.target_layers.clone(),
        target_modules: lora_config.target_module_names(),
        rank,
        alpha,
        zero_lora_max_delta,
        nonzero_lora_max_delta,
        reload_max_delta,
        trainable_tensors: reloaded.trainable_tensor_names(),
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_lora_train_smoke(
    model_path: &Path,
    fixture: &Path,
    adapter_output: &Path,
    rank: i64,
    alpha: f64,
    learning_rate: f64,
) -> Result<()> {
    if rank <= 0 {
        bail!("rank must be positive");
    }
    if alpha <= 0.0 {
        bail!("alpha must be positive");
    }
    if learning_rate <= 0.0 {
        bail!("learning_rate must be positive");
    }

    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let fixture_tensors = read_safetensors_map(fixture)?;
    let attention_input = tensor(&fixture_tensors, "input_attention_normed")?.to_kind(Kind::Float);
    let base_target_output = tensor(&fixture_tensors, "attention_output")?.to_kind(Kind::Float);
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let layer0 = QwenLayerWeights::load(&weights, 0)?;
    let target_output = lora_train_target(&base_target_output);
    let lora_config = QwenLoraConfig::layer0_qv(rank, alpha)?;
    let registry = QwenLoraRegistry::deterministic(&weights, &lora_config, true)?;
    let base_requires_grad = layer0.q_proj.requires_grad()
        || layer0.k_proj.requires_grad()
        || layer0.v_proj.requires_grad()
        || layer0.o_proj.requires_grad();

    let initial_loss = qwen_attention_lora_mse_loss(
        &attention_input,
        &target_output,
        &layer0,
        registry.layer_adapter(0)?,
        &config,
    )
    .double_value(&[]);
    let loss = qwen_attention_lora_mse_loss(
        &attention_input,
        &target_output,
        &layer0,
        registry.layer_adapter(0)?,
        &config,
    );
    loss.backward();

    let base_tensors: BTreeMap<String, Tensor> = registry
        .trainable_tensors()
        .into_iter()
        .map(|(name, tensor)| (name, tensor_snapshot(&tensor)))
        .collect();
    let mut tensor_summaries = Vec::new();
    for (name, mut tensor) in registry.trainable_tensors() {
        let grad = tensor.grad();
        let grad_defined = grad.defined();
        let grad_norm = if grad_defined {
            grad.norm().double_value(&[])
        } else {
            0.0
        };
        if !grad_defined || grad_norm <= 0.0 {
            bail!("LoRA tensor {name} did not receive a gradient");
        }
        let _ = no_grad(|| tensor.f_sub_(&(&grad * learning_rate)))?;
        let delta_norm = (&tensor
            - base_tensors
                .get(&name)
                .ok_or_else(|| anyhow!("missing base LoRA tensor {name}"))?)
        .norm()
        .double_value(&[]);
        tensor_summaries.push(TrainableTensorSummary {
            name,
            grad_defined,
            grad_norm,
            delta_norm,
        });
    }

    let final_loss = qwen_attention_lora_mse_loss(
        &attention_input,
        &target_output,
        &layer0,
        registry.layer_adapter(0)?,
        &config,
    )
    .double_value(&[]);
    if final_loss >= initial_loss {
        bail!(
            "Qwen LoRA train smoke failed to reduce loss: initial_loss={initial_loss}, final_loss={final_loss}"
        );
    }

    registry.save(adapter_output)?;
    let reloaded = QwenLoraRegistry::load(adapter_output)?;
    let reloaded_loss = qwen_attention_lora_mse_loss(
        &attention_input,
        &target_output,
        &layer0,
        reloaded.layer_adapter(0)?,
        &config,
    )
    .double_value(&[]);
    let reload_delta = (final_loss - reloaded_loss).abs();
    if reload_delta > 1e-7 {
        bail!(
            "Qwen LoRA adapter reload loss parity failed: final_loss={final_loss}, reloaded_loss={reloaded_loss}, reload_delta={reload_delta}"
        );
    }

    let summary = QwenLoraTrainSmokeSummary {
        model_path: model_path.display().to_string(),
        fixture: fixture.display().to_string(),
        adapter_output: adapter_output.display().to_string(),
        target_layers: lora_config.target_layers.clone(),
        target_modules: lora_config.target_module_names(),
        rank,
        alpha,
        learning_rate,
        initial_loss,
        final_loss,
        reloaded_loss,
        reload_delta,
        base_requires_grad,
        trainable_tensors: tensor_summaries,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_lora_sft_smoke(
    model_path: &Path,
    adapter_output: &Path,
    sft_jsonl: Option<&Path>,
    sft_batch_size: usize,
    instruction: &str,
    response: &str,
    rank: i64,
    alpha: f64,
    learning_rate: f64,
) -> Result<()> {
    if rank <= 0 {
        bail!("rank must be positive");
    }
    if alpha <= 0.0 {
        bail!("alpha must be positive");
    }
    if learning_rate <= 0.0 {
        bail!("learning_rate must be positive");
    }
    if sft_batch_size == 0 {
        bail!("sft_batch_size must be positive");
    }

    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let dataset = if let Some(sft_jsonl) = sft_jsonl {
        QwenSftDataset::from_jsonl_path(&tokenizer, sft_jsonl)?
    } else {
        QwenSftDataset::from_instruction_pairs(
            &tokenizer,
            &[
                QwenSftExample {
                    instruction: instruction.to_string(),
                    input: String::new(),
                    response: response.to_string(),
                },
                QwenSftExample {
                    instruction: "Name the project.".to_string(),
                    input: String::new(),
                    response: "rustrain".to_string(),
                },
            ],
        )?
    };
    let batch = dataset.padded_batch(0, sft_batch_size.min(dataset.len()))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let config = read_runtime_config(&model_path.join("config.json"))?;
    let layer0 = QwenLayerWeights::load(&weights, 0)?;
    let lora_config = QwenLoraConfig::layer0_qv(rank, alpha)?;
    let registry = QwenLoraRegistry::deterministic(&weights, &lora_config, true)?;
    let base_requires_grad = layer0.q_proj.requires_grad()
        || layer0.k_proj.requires_grad()
        || layer0.v_proj.requires_grad()
        || layer0.o_proj.requires_grad();

    let initial_loss = qwen_attention_lora_sft_loss(
        &batch.input_ids,
        &batch.target_mask,
        &weights,
        registry.layer_adapter(0)?,
        &config,
    )?
    .double_value(&[]);
    let loss = qwen_attention_lora_sft_loss(
        &batch.input_ids,
        &batch.target_mask,
        &weights,
        registry.layer_adapter(0)?,
        &config,
    )?;
    loss.backward();

    let base_tensors: BTreeMap<String, Tensor> = registry
        .trainable_tensors()
        .into_iter()
        .map(|(name, tensor)| (name, tensor_snapshot(&tensor)))
        .collect();
    let mut tensor_summaries = Vec::new();
    for (name, mut tensor) in registry.trainable_tensors() {
        let grad = tensor.grad();
        let grad_defined = grad.defined();
        let grad_norm = if grad_defined {
            grad.norm().double_value(&[])
        } else {
            0.0
        };
        if !grad_defined || grad_norm <= 0.0 {
            bail!("LoRA tensor {name} did not receive a gradient");
        }
        let _ = no_grad(|| tensor.f_sub_(&(&grad * learning_rate)))?;
        let delta_norm = (&tensor
            - base_tensors
                .get(&name)
                .ok_or_else(|| anyhow!("missing base LoRA tensor {name}"))?)
        .norm()
        .double_value(&[]);
        tensor_summaries.push(TrainableTensorSummary {
            name,
            grad_defined,
            grad_norm,
            delta_norm,
        });
    }

    let final_loss = qwen_attention_lora_sft_loss(
        &batch.input_ids,
        &batch.target_mask,
        &weights,
        registry.layer_adapter(0)?,
        &config,
    )?
    .double_value(&[]);
    if final_loss >= initial_loss {
        bail!(
            "Qwen LoRA SFT smoke failed to reduce response-only loss: initial_loss={initial_loss}, final_loss={final_loss}"
        );
    }

    registry.save(adapter_output)?;
    let reloaded = QwenLoraRegistry::load(adapter_output)?;
    let reloaded_loss = qwen_attention_lora_sft_loss(
        &batch.input_ids,
        &batch.target_mask,
        &weights,
        reloaded.layer_adapter(0)?,
        &config,
    )?
    .double_value(&[]);
    let reload_delta = (final_loss - reloaded_loss).abs();
    if reload_delta > 1e-7 {
        bail!(
            "Qwen LoRA SFT adapter reload loss parity failed: final_loss={final_loss}, reloaded_loss={reloaded_loss}, reload_delta={reload_delta}"
        );
    }

    let summary = QwenLoraSftSmokeSummary {
        model_path: model_path.display().to_string(),
        adapter_output: adapter_output.display().to_string(),
        target_layers: lora_config.target_layers.clone(),
        target_modules: lora_config.target_module_names(),
        batch_size: batch.prompt_tokens.len(),
        prompt_tokens: batch.prompt_tokens,
        response_tokens: batch.response_tokens,
        sequence_tokens: batch.input_ids.size()[1] as usize,
        response_masked_positions: batch.masked_positions,
        padding_tokens: batch.padding_tokens,
        rank,
        alpha,
        learning_rate,
        initial_loss,
        final_loss,
        reloaded_loss,
        reload_delta,
        base_requires_grad,
        trainable_tensors: tensor_summaries,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_tied_head_train_smoke(
    model_path: &Path,
    reference_fixture: &Path,
    delta_output: &Path,
    learning_rate: f64,
) -> Result<()> {
    if learning_rate <= 0.0 {
        bail!("learning_rate must be positive");
    }

    let config = read_runtime_config(&model_path.join("config.json"))?;
    let mut weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);
    if input_ids.size()[1] < 2 {
        bail!("training fixture must contain at least two tokens");
    }

    let mut embed_tokens = tensor(&weights, "model.embed_tokens.weight")?
        .to_kind(Kind::Float)
        .set_requires_grad(true);
    weights.insert(
        "model.embed_tokens.weight".to_string(),
        embed_tokens.shallow_clone(),
    );

    let initial_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)?.double_value(&[]);
    let loss = qwen_causal_lm_loss(&input_ids, &weights, &config)?;
    loss.backward();
    let grad = embed_tokens.grad();
    let grad_defined = grad.defined();
    let grad_norm = if grad_defined {
        grad.norm().double_value(&[])
    } else {
        0.0
    };
    if !grad_defined || grad_norm <= 0.0 {
        bail!("tied embedding gradient was not populated");
    }

    let update = &grad * learning_rate;
    let _ = no_grad(|| embed_tokens.f_sub_(&update))?;

    let final_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)?.double_value(&[]);
    if final_loss >= initial_loss {
        bail!(
            "Qwen tied-head train smoke failed to reduce loss: initial_loss={initial_loss}, final_loss={final_loss}"
        );
    }
    let base_embed_tokens = tensor(
        &read_safetensors_map(&model_path.join("model.safetensors"))?,
        "model.embed_tokens.weight",
    )?
    .to_kind(Kind::Float);
    let delta = &embed_tokens - &base_embed_tokens;
    if let Some(parent) = delta_output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    Tensor::write_safetensors(
        &[(&"model.embed_tokens.weight.delta", &delta)],
        delta_output,
    )
    .with_context(|| format!("failed to write {}", delta_output.display()))?;

    let mut reloaded_weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let delta_tensors = read_safetensors_map(delta_output)?;
    let reloaded_embed = tensor(&reloaded_weights, "model.embed_tokens.weight")?
        .to_kind(Kind::Float)
        + tensor(&delta_tensors, "model.embed_tokens.weight.delta")?.to_kind(Kind::Float);
    reloaded_weights.insert("model.embed_tokens.weight".to_string(), reloaded_embed);
    let reloaded_loss =
        qwen_causal_lm_loss(&input_ids, &reloaded_weights, &config)?.double_value(&[]);
    let reload_delta = (final_loss - reloaded_loss).abs();
    if reload_delta > 1e-5 {
        bail!(
            "Qwen tied-head delta reload parity failed: final_loss={final_loss}, reloaded_loss={reloaded_loss}, reload_delta={reload_delta}"
        );
    }

    let summary = QwenTiedHeadTrainSummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        delta_output: delta_output.display().to_string(),
        trainable_tensor: "model.embed_tokens.weight".to_string(),
        learning_rate,
        initial_loss,
        final_loss,
        reloaded_loss,
        reload_delta,
        grad_defined,
        grad_norm,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_full_train_smoke(
    model_path: &Path,
    reference_fixture: &Path,
    delta_output: &Path,
    dtype: QwenComputeDType,
    learning_rate: f64,
) -> Result<()> {
    if learning_rate <= 0.0 {
        bail!("learning_rate must be positive");
    }

    let config = read_runtime_config(&model_path.join("config.json"))?;
    let weights = read_safetensors_map(&model_path.join("model.safetensors"))?;
    let reference = read_safetensors_map(reference_fixture)?;
    let input_ids = tensor(&reference, "input_ids")?.to_kind(Kind::Int64);

    let mut session = QwenTrainableSession::from_weights(config, weights, input_ids, dtype.kind())?;
    let first_step = session.train_step(learning_rate, 1)?;
    let initial_loss = first_step.loss_before;
    let final_loss = first_step.loss_after;
    if final_loss >= initial_loss {
        bail!(
            "Qwen full train smoke failed to reduce loss: initial_loss={initial_loss}, final_loss={final_loss}"
        );
    }

    if let Some(parent) = delta_output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let delta_refs: Vec<(&str, &Tensor)> = first_step
        .artifacts
        .delta_entries
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    Tensor::write_safetensors(&delta_refs, delta_output)
        .with_context(|| format!("failed to write {}", delta_output.display()))?;
    let optimizer_output = optimizer_state_path(delta_output);
    let optimizer_refs: Vec<(&str, &Tensor)> = first_step
        .artifacts
        .optimizer_entries
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    Tensor::write_safetensors(&optimizer_refs, &optimizer_output)
        .with_context(|| format!("failed to write {}", optimizer_output.display()))?;
    let manifest_output = delta_manifest_path(delta_output);
    let manifest = QwenDeltaCheckpointManifest {
        format: "rustrain.qwen_delta.v1".to_string(),
        base_model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        delta_safetensors: delta_output.display().to_string(),
        optimizer_safetensors: Some(optimizer_output.display().to_string()),
        train_step: 1,
        learning_rate,
        initial_loss,
        final_loss,
        tensors: first_step.artifacts.manifest_tensors,
    };
    write_qwen_delta_manifest(&manifest_output, &manifest)?;

    let mut resumed_session = QwenTrainableSession::from_manifest(
        session.config,
        read_safetensors_map(&model_path.join("model.safetensors"))?,
        session.input_ids.shallow_clone(),
        dtype.kind(),
        &manifest,
    )?;
    let reloaded_loss = resumed_session.loss_value()?;
    let reload_delta = (final_loss - reloaded_loss).abs();
    if reload_delta > 1e-5 {
        bail!(
            "Qwen full train delta reload parity failed: final_loss={final_loss}, reloaded_loss={reloaded_loss}, reload_delta={reload_delta}"
        );
    }

    let resumed_second_step = resumed_session.train_step(learning_rate, 2)?;
    let resume_loss_value = resumed_second_step.loss_before;
    let resumed_second_loss = resumed_second_step.loss_after;

    let continuous_second_step = session.train_step(learning_rate, 2)?;
    let continuous_second_loss = continuous_second_step.loss_after;
    let second_step_delta = (continuous_second_loss - resumed_second_loss).abs();
    if second_step_delta > 1e-5 {
        bail!(
            "Qwen full train manifest resume parity failed: continuous_second_loss={continuous_second_loss}, resumed_second_loss={resumed_second_loss}, second_step_delta={second_step_delta}"
        );
    }

    let summary = QwenFullTrainSmokeSummary {
        model_path: model_path.display().to_string(),
        reference_fixture: reference_fixture.display().to_string(),
        delta_output: delta_output.display().to_string(),
        optimizer_output: optimizer_output.display().to_string(),
        manifest_output: manifest_output.display().to_string(),
        compute_kind: dtype.label().to_string(),
        learning_rate,
        initial_loss,
        final_loss,
        reloaded_loss,
        reload_delta,
        resume_loss: resume_loss_value,
        continuous_second_loss,
        resumed_second_loss,
        second_step_delta,
        trainable_tensors: first_step.artifacts.tensor_summaries,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

pub fn qwen_dp_gradient_smoke(
    model_path: &Path,
    reference_fixture: &Path,
    output_dir: PathBuf,
    dtype: QwenComputeDType,
) -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("Qwen DP gradient smoke expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    let device = Device::Cuda(local_rank);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let output_dir = qwen_dp_artifact_dir(&output_dir)?;
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    println!("qwen attention DP rank {rank}: loading fixture");

    let config = read_runtime_config(&model_path.join("config.json"))?;
    let fixture = read_safetensors_map(reference_fixture)?;
    let attention_input = tensor(&fixture, "input_attention_normed")?.to_kind(dtype.kind());
    let attention_target = tensor(&fixture, "attention_output")?.to_kind(dtype.kind());
    let local_input = qwen_dp_attention_input_for_rank(&attention_input, rank, world_size)?;
    let local_target = qwen_dp_attention_target_for_rank(&attention_target, rank, world_size)?;

    println!("qwen attention DP rank {rank}: loading model weights");
    let mut local_session = QwenAttentionDpSession::from_weights(
        read_safetensors_map(&model_path.join("model.safetensors"))?,
        local_input,
        local_target,
        config,
        dtype.kind(),
        device,
    )?;
    println!("qwen attention DP rank {rank}: running local backward");
    let local_loss = local_session.loss_and_backward()?;
    let local_grads = local_session.grad_entries()?;

    let expected_path = output_dir.join("qwen-dp-expected-signatures.json");
    let (expected_loss, expected_signatures) = if rank == 0 {
        println!("qwen attention DP rank {rank}: running expected backward");
        let global_input = qwen_dp_attention_global(&attention_input)?;
        let global_target = qwen_dp_attention_global(&attention_target)?;
        let mut expected_session = QwenAttentionDpSession::from_weights(
            read_safetensors_map(&model_path.join("model.safetensors"))?,
            global_input,
            global_target,
            local_session.config,
            dtype.kind(),
            device,
        )?;
        let expected_loss = expected_session.loss_and_backward()?;
        let expected_signatures = grad_signatures(&expected_session.grad_entries()?)?;
        let encoded = serde_json::to_string_pretty(&(expected_loss, &expected_signatures))?;
        fs::write(&expected_path, encoded)
            .with_context(|| format!("failed to write {}", expected_path.display()))?;
        (expected_loss, expected_signatures)
    } else {
        println!("qwen attention DP rank {rank}: waiting for expected signatures");
        wait_for_expected_signatures(&expected_path, Duration::from_secs(300))?
    };

    println!("qwen attention DP rank {rank}: reducing gradient signatures");
    let mut local_signature_values = Vec::new();
    let mut expected_signature_values = Vec::new();
    for ((name, local_grad), expected) in local_grads.iter().zip(expected_signatures.iter()) {
        if name != &expected.name {
            bail!(
                "gradient tensor order mismatch: local {name} != expected {}",
                expected.name
            );
        }
        let local_signature = grad_signature(name, local_grad)?;
        local_signature_values.extend(local_signature.values());
        expected_signature_values.extend(expected.values());
    }
    wait_for_rank_barrier(
        &output_dir.join("qwen-dp-gradient-signatures-ready"),
        rank,
        world_size,
        Duration::from_secs(300),
    )?;
    let reduced_signatures = nccl_smoke::all_reduce_f32_for_launch(
        &output_dir.join("qwen-dp-gradient-signatures"),
        &local_signature_values,
    )?;
    let averaged_signatures: Vec<f32> = reduced_signatures
        .into_iter()
        .map(|value| value / world_size as f32)
        .collect();
    let max_grad_delta =
        signature_values_max_delta(&averaged_signatures, &expected_signature_values)?;

    let global_loss = expected_loss;
    let loss_delta = 0.0;

    if max_grad_delta > 5e-4 || loss_delta > 5e-4 {
        bail!(
            "Qwen DP gradient mismatch: rank={rank}, max_grad_delta={max_grad_delta}, loss_delta={loss_delta}"
        );
    }

    let checkpoint_path = output_dir.join("qwen-dp-rank0-checkpoint.json");
    let checkpoint_written = if rank == 0 {
        let manifest = QwenDpCheckpointManifest {
            format: "rustrain.qwen_dp_rank0.v1".to_string(),
            writer_rank: rank,
            world_size,
            tensor_count: local_grads.len(),
            max_grad_delta,
            expected_loss,
            dtype: dtype.label().to_string(),
        };
        fs::write(
            &checkpoint_path,
            serde_json::to_string_pretty(&manifest)? + "\n",
        )
        .with_context(|| format!("failed to write {}", checkpoint_path.display()))?;
        true
    } else {
        false
    };

    let summary = QwenDpGradientRankSummary {
        rank,
        world_size,
        local_sequence_count: local_session.input.size()[0] as usize,
        tensor_count: local_grads.len(),
        max_grad_delta,
        loss_delta,
        local_loss,
        global_loss,
        expected_loss,
        checkpoint_written,
        checkpoint_path: checkpoint_path.display().to_string(),
    };
    let summary_path = output_dir.join(format!("qwen-dp-gradient-rank-{rank}.json"));
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

fn qwen_dp_artifact_dir(output_dir: &Path) -> Result<PathBuf> {
    let port = std::env::var("MASTER_PORT")
        .context("MASTER_PORT is not set; run through rustrain launch")?;
    Ok(output_dir.join(format!("launch-{port}")))
}

fn delta_manifest_path(delta_output: &Path) -> std::path::PathBuf {
    let mut path = delta_output.as_os_str().to_os_string();
    path.push(".json");
    path.into()
}

fn optimizer_state_path(delta_output: &Path) -> std::path::PathBuf {
    let mut path = delta_output.as_os_str().to_os_string();
    path.push(".optimizer.safetensors");
    path.into()
}

fn adam_slot_names(name: &str) -> AdamSlotNames {
    AdamSlotNames {
        m: format!("{name}.adam_m"),
        v: format!("{name}.adam_v"),
    }
}

impl QwenTrainableRegistry {
    fn representative(weights: &mut BTreeMap<String, Tensor>) -> Result<Self> {
        Self::from_names(weights, representative_trainable_qwen_tensors())
    }

    fn from_names(
        weights: &mut BTreeMap<String, Tensor>,
        names: Vec<&'static str>,
    ) -> Result<Self> {
        let mut parameters = Vec::with_capacity(names.len());
        for name in names {
            let base = tensor(weights, name)?.to_kind(Kind::Float);
            let trainable = base.shallow_clone().set_requires_grad(true);
            weights.insert(name.to_string(), trainable.shallow_clone());
            parameters.push(QwenTrainableParameter {
                name: name.to_string(),
                tensor: trainable,
                base: tensor_snapshot(&base),
                adam: None,
            });
        }
        Ok(Self { parameters })
    }

    #[cfg(test)]
    fn parameter_names(&self) -> Vec<String> {
        self.parameters
            .iter()
            .map(|parameter| parameter.name.clone())
            .collect()
    }

    fn adamw_step(
        &mut self,
        weights: &mut BTreeMap<String, Tensor>,
        learning_rate: f64,
        step: i32,
    ) -> Result<QwenTrainStepArtifacts> {
        let mut tensor_summaries = Vec::with_capacity(self.parameters.len());
        let mut manifest_tensors = Vec::with_capacity(self.parameters.len());
        let mut delta_entries = Vec::with_capacity(self.parameters.len());
        let mut optimizer_entries = Vec::with_capacity(self.parameters.len() * 2);

        for parameter in &mut self.parameters {
            let grad = parameter.tensor.grad();
            let grad_defined = grad.defined();
            let grad_norm = if grad_defined {
                grad.norm().double_value(&[])
            } else {
                0.0
            };
            if !grad_defined || grad_norm <= 0.0 {
                bail!(
                    "trainable tensor {} did not receive a gradient",
                    parameter.name
                );
            }

            let adam_state = adamw_next_state(parameter.adam.as_ref(), &grad, 0.9, 0.999);
            let update = adamw_update(&adam_state, learning_rate, 0.9, 0.999, step, 1e-8);
            let _ = no_grad(|| parameter.tensor.f_sub_(&update))?;
            weights.insert(parameter.name.clone(), parameter.tensor.shallow_clone());

            let delta = &parameter.tensor - &parameter.base;
            let delta_norm = delta.norm().double_value(&[]);
            let delta_name = format!("{}.delta", parameter.name);
            let adam_names = adam_slot_names(&parameter.name);
            manifest_tensors.push(QwenDeltaTensorManifestEntry {
                name: parameter.name.clone(),
                delta_name: delta_name.clone(),
                adam_m_name: Some(adam_names.m.clone()),
                adam_v_name: Some(adam_names.v.clone()),
                shape: parameter.tensor.size(),
                dtype: "float32".to_string(),
                grad_norm,
                delta_norm,
            });
            delta_entries.push((delta_name, delta));
            optimizer_entries.push((adam_names.m, adam_state.m.shallow_clone()));
            optimizer_entries.push((adam_names.v, adam_state.v.shallow_clone()));
            tensor_summaries.push(TrainableTensorSummary {
                name: parameter.name.clone(),
                grad_defined,
                grad_norm,
                delta_norm,
            });
            parameter.adam = Some(adam_state);
        }

        Ok(QwenTrainStepArtifacts {
            tensor_summaries,
            manifest_tensors,
            delta_entries,
            optimizer_entries,
        })
    }

    fn zero_grad(&mut self) {
        for parameter in &mut self.parameters {
            parameter.tensor.zero_grad();
        }
    }

    fn apply_delta_checkpoint(
        weights: &mut BTreeMap<String, Tensor>,
        delta_tensors: &BTreeMap<String, Tensor>,
        manifest_tensors: &[QwenDeltaTensorManifestEntry],
    ) -> Result<()> {
        for entry in manifest_tensors {
            let reloaded = tensor(weights, &entry.name)?.to_kind(Kind::Float)
                + tensor(delta_tensors, &entry.delta_name)?.to_kind(Kind::Float);
            weights.insert(entry.name.clone(), reloaded);
        }
        Ok(())
    }

    fn load_from_manifest(
        weights: &mut BTreeMap<String, Tensor>,
        manifest: &QwenDeltaCheckpointManifest,
    ) -> Result<Self> {
        if manifest.format != "rustrain.qwen_delta.v1" {
            bail!(
                "unsupported Qwen delta checkpoint format {}",
                manifest.format
            );
        }
        let delta_tensors = read_safetensors_map(Path::new(&manifest.delta_safetensors))?;
        Self::apply_delta_checkpoint(weights, &delta_tensors, &manifest.tensors)?;
        let optimizer_tensors = if let Some(path) = &manifest.optimizer_safetensors {
            Some(read_safetensors_map(Path::new(path))?)
        } else {
            None
        };

        let mut parameters = Vec::with_capacity(manifest.tensors.len());
        for entry in &manifest.tensors {
            let reloaded = tensor(weights, &entry.name)?.to_kind(Kind::Float);
            let base = tensor_snapshot(
                &(reloaded.shallow_clone()
                    - tensor(&delta_tensors, &entry.delta_name)?.to_kind(Kind::Float)),
            );
            let trainable = reloaded.set_requires_grad(true);
            weights.insert(entry.name.clone(), trainable.shallow_clone());
            let adam = match (
                optimizer_tensors.as_ref(),
                entry.adam_m_name.as_ref(),
                entry.adam_v_name.as_ref(),
            ) {
                (Some(optimizer_tensors), Some(m_name), Some(v_name)) => Some(AdamState {
                    m: tensor(optimizer_tensors, m_name)?.to_kind(Kind::Float),
                    v: tensor(optimizer_tensors, v_name)?.to_kind(Kind::Float),
                }),
                (None, None, None) => None,
                _ => bail!(
                    "incomplete optimizer state for trainable tensor {}",
                    entry.name
                ),
            };
            parameters.push(QwenTrainableParameter {
                name: entry.name.clone(),
                tensor: trainable,
                base,
                adam,
            });
        }

        Ok(Self { parameters })
    }
}

impl QwenTrainableSession {
    fn from_weights(
        config: QwenRuntimeConfig,
        mut weights: BTreeMap<String, Tensor>,
        input_ids: Tensor,
        compute_kind: Kind,
    ) -> Result<Self> {
        if input_ids.size()[1] < 2 {
            bail!("training fixture must contain at least two tokens");
        }
        let registry = QwenTrainableRegistry::representative(&mut weights)?;
        Ok(Self {
            config,
            weights,
            input_ids,
            compute_kind,
            registry,
        })
    }

    fn from_manifest(
        config: QwenRuntimeConfig,
        mut weights: BTreeMap<String, Tensor>,
        input_ids: Tensor,
        compute_kind: Kind,
        manifest: &QwenDeltaCheckpointManifest,
    ) -> Result<Self> {
        if input_ids.size()[1] < 2 {
            bail!("training fixture must contain at least two tokens");
        }
        let registry = QwenTrainableRegistry::load_from_manifest(&mut weights, manifest)?;
        Ok(Self {
            config,
            weights,
            input_ids,
            compute_kind,
            registry,
        })
    }

    fn loss_value(&self) -> Result<f64> {
        Ok(qwen_causal_lm_loss_with_kind(
            &self.input_ids,
            &self.weights,
            &self.config,
            self.compute_kind,
        )?
        .double_value(&[]))
    }

    fn train_step(&mut self, learning_rate: f64, step: i32) -> Result<QwenTrainStepResult> {
        self.registry.zero_grad();
        let loss = qwen_causal_lm_loss_with_kind(
            &self.input_ids,
            &self.weights,
            &self.config,
            self.compute_kind,
        )?;
        let loss_before = loss.double_value(&[]);
        loss.backward();
        let artifacts = self
            .registry
            .adamw_step(&mut self.weights, learning_rate, step)?;
        let loss_after = self.loss_value()?;
        Ok(QwenTrainStepResult {
            loss_before,
            loss_after,
            artifacts,
        })
    }
}

impl QwenAttentionDpSession {
    fn from_weights(
        weights: BTreeMap<String, Tensor>,
        input: Tensor,
        target: Tensor,
        config: QwenRuntimeConfig,
        compute_kind: Kind,
        device: Device,
    ) -> Result<Self> {
        let q_proj = tensor(&weights, "model.layers.0.self_attn.q_proj.weight")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        let q_bias = tensor(&weights, "model.layers.0.self_attn.q_proj.bias")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        let k_proj = tensor(&weights, "model.layers.0.self_attn.k_proj.weight")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        let k_bias = tensor(&weights, "model.layers.0.self_attn.k_proj.bias")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        let v_proj = tensor(&weights, "model.layers.0.self_attn.v_proj.weight")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        let v_bias = tensor(&weights, "model.layers.0.self_attn.v_proj.bias")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        let o_proj = tensor(&weights, "model.layers.0.self_attn.o_proj.weight")?
            .to_kind(Kind::Float)
            .to_device(device)
            .set_requires_grad(true);
        Ok(Self {
            config,
            input: input.to_kind(compute_kind).to_device(device),
            target: target.to_kind(compute_kind).to_device(device),
            q_proj,
            q_bias,
            k_proj,
            k_bias,
            v_proj,
            v_bias,
            o_proj,
            compute_kind,
        })
    }

    fn loss_and_backward(&mut self) -> Result<f64> {
        for (_, parameter) in self.parameters_mut() {
            parameter.zero_grad();
        }
        let output = qwen_attention(
            &self.input,
            &self.q_proj.to_kind(self.compute_kind),
            &self.q_bias.to_kind(self.compute_kind),
            &self.k_proj.to_kind(self.compute_kind),
            &self.k_bias.to_kind(self.compute_kind),
            &self.v_proj.to_kind(self.compute_kind),
            &self.v_bias.to_kind(self.compute_kind),
            &self.o_proj.to_kind(self.compute_kind),
            &self.config,
        );
        let loss = output.mse_loss(&self.target, Reduction::Mean);
        let loss_value = loss.double_value(&[]);
        loss.backward();
        Ok(loss_value)
    }

    fn grad_entries(&self) -> Result<Vec<(String, Tensor)>> {
        let mut entries = Vec::new();
        for (name, parameter) in self.parameters() {
            let grad = parameter.grad();
            if !grad.defined() {
                bail!("trainable tensor {name} did not receive a gradient");
            }
            entries.push((name.to_string(), grad.to_kind(Kind::Float)));
        }
        Ok(entries)
    }

    fn parameters(&self) -> [(&'static str, &Tensor); 7] {
        [
            ("model.layers.0.self_attn.q_proj.weight", &self.q_proj),
            ("model.layers.0.self_attn.q_proj.bias", &self.q_bias),
            ("model.layers.0.self_attn.k_proj.weight", &self.k_proj),
            ("model.layers.0.self_attn.k_proj.bias", &self.k_bias),
            ("model.layers.0.self_attn.v_proj.weight", &self.v_proj),
            ("model.layers.0.self_attn.v_proj.bias", &self.v_bias),
            ("model.layers.0.self_attn.o_proj.weight", &self.o_proj),
        ]
    }

    fn parameters_mut(&mut self) -> [(&'static str, &mut Tensor); 7] {
        [
            ("model.layers.0.self_attn.q_proj.weight", &mut self.q_proj),
            ("model.layers.0.self_attn.q_proj.bias", &mut self.q_bias),
            ("model.layers.0.self_attn.k_proj.weight", &mut self.k_proj),
            ("model.layers.0.self_attn.k_proj.bias", &mut self.k_bias),
            ("model.layers.0.self_attn.v_proj.weight", &mut self.v_proj),
            ("model.layers.0.self_attn.v_proj.bias", &mut self.v_bias),
            ("model.layers.0.self_attn.o_proj.weight", &mut self.o_proj),
        ]
    }
}

fn adamw_next_state(
    previous: Option<&AdamState>,
    grad: &Tensor,
    beta1: f64,
    beta2: f64,
) -> AdamState {
    let m = if let Some(previous) = previous {
        &previous.m * beta1 + grad * (1.0 - beta1)
    } else {
        grad * (1.0 - beta1)
    };
    let grad_sq = grad.pow_tensor_scalar(2.0);
    let v = if let Some(previous) = previous {
        &previous.v * beta2 + grad_sq * (1.0 - beta2)
    } else {
        grad_sq * (1.0 - beta2)
    };
    AdamState { m, v }
}

fn qwen_dp_attention_global(input: &Tensor) -> Result<Tensor> {
    if input.size().len() != 3 || input.size()[0] != 1 || input.size()[1] < 2 {
        bail!("Qwen attention DP fixture expects shape [1, seq_len>=2, hidden]");
    }
    let reversed = input.flip([1]);
    Ok(Tensor::cat(&[input.shallow_clone(), reversed], 0))
}

fn qwen_dp_attention_input_for_rank(
    input: &Tensor,
    rank: usize,
    world_size: usize,
) -> Result<Tensor> {
    if world_size != 2 {
        bail!("Qwen attention DP fixture currently expects world_size=2");
    }
    let global = qwen_dp_attention_global(input)?;
    Ok(global.narrow(0, rank as i64, 1))
}

fn qwen_dp_attention_target_for_rank(
    target: &Tensor,
    rank: usize,
    world_size: usize,
) -> Result<Tensor> {
    qwen_dp_attention_input_for_rank(target, rank, world_size)
}

fn grad_signatures(grads: &[(String, Tensor)]) -> Result<Vec<QwenGradSignature>> {
    grads
        .iter()
        .map(|(name, grad)| grad_signature(name, grad))
        .collect()
}

fn grad_signature(name: &str, grad: &Tensor) -> Result<QwenGradSignature> {
    let shape = grad.size();
    let flat = grad.to_kind(Kind::Float).reshape([-1]);
    let numel = flat.numel();
    if numel == 0 {
        bail!("gradient tensor {name} is empty");
    }
    let sample_count = numel.min(16);
    let stride = (numel / sample_count).max(1);
    let samples = (0..sample_count)
        .map(|index| flat.double_value(&[((index * stride).min(numel - 1)) as i64]) as f32)
        .collect();
    Ok(QwenGradSignature {
        name: name.to_string(),
        shape,
        samples,
    })
}

impl QwenGradSignature {
    fn values(&self) -> Vec<f32> {
        self.samples.clone()
    }
}

fn signature_values_max_delta(actual: &[f32], expected: &[f32]) -> Result<f32> {
    if actual.len() != expected.len() {
        bail!(
            "gradient signature length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        );
    }
    Ok(actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max))
}

fn wait_for_expected_signatures(
    path: &Path,
    timeout: Duration,
) -> Result<(f64, Vec<QwenGradSignature>)> {
    let start = Instant::now();
    loop {
        match fs::read_to_string(path) {
            Ok(contents) => {
                return serde_json::from_str(&contents)
                    .with_context(|| format!("failed to parse {}", path.display()));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if start.elapsed() > timeout {
                    bail!("timed out waiting for {}", path.display());
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(error) => {
                return Err(error).with_context(|| format!("failed to read {}", path.display()));
            }
        }
    }
}

fn wait_for_rank_barrier(
    dir: &Path,
    rank: usize,
    world_size: usize,
    timeout: Duration,
) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let ready_path = dir.join(format!("rank-{rank}.ready"));
    fs::write(&ready_path, b"ready")
        .with_context(|| format!("failed to write {}", ready_path.display()))?;
    let start = Instant::now();
    loop {
        let all_ready = (0..world_size).all(|rank| dir.join(format!("rank-{rank}.ready")).exists());
        if all_ready {
            return Ok(());
        }
        if start.elapsed() > timeout {
            bail!("timed out waiting for barrier {}", dir.display());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set; run through rustrain launch"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

fn adamw_update(
    state: &AdamState,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    step: i32,
    eps: f64,
) -> Tensor {
    let m_hat = &state.m / (1.0 - beta1.powi(step));
    let v_hat = &state.v / (1.0 - beta2.powi(step));
    (m_hat / v_hat.sqrt().g_add_scalar(eps)) * learning_rate
}

fn write_qwen_delta_manifest(
    manifest_output: &Path,
    manifest: &QwenDeltaCheckpointManifest,
) -> Result<()> {
    if let Some(parent) = manifest_output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(
        manifest_output,
        serde_json::to_string_pretty(manifest).context("failed to serialize manifest")? + "\n",
    )
    .with_context(|| format!("failed to write {}", manifest_output.display()))
}

fn representative_trainable_qwen_tensors() -> Vec<&'static str> {
    vec![
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
    ]
}

fn read_runtime_config(path: &Path) -> Result<QwenRuntimeConfig> {
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

pub fn read_safetensors_map(path: &Path) -> Result<BTreeMap<String, Tensor>> {
    let tensors = Tensor::read_safetensors(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(tensors.into_iter().collect())
}

pub fn tensor<'a>(tensors: &'a BTreeMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| anyhow!("missing tensor {name}"))
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
    input_norm: Tensor,
    q_proj: Tensor,
    q_bias: Tensor,
    k_proj: Tensor,
    k_bias: Tensor,
    v_proj: Tensor,
    v_bias: Tensor,
    o_proj: Tensor,
    post_attention_norm: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

struct QwenLayerCache {
    key: Tensor,
    value: Tensor,
}

struct QwenAttentionLoraAdapter {
    q_a: Tensor,
    q_b: Tensor,
    v_a: Tensor,
    v_b: Tensor,
    rank: i64,
    alpha: f64,
}

impl QwenLoraTargetModule {
    fn as_str(self) -> &'static str {
        match self {
            Self::QProj => "q_proj",
            Self::VProj => "v_proj",
        }
    }
}

impl QwenLoraConfig {
    fn layer0_qv(rank: i64, alpha: f64) -> Result<Self> {
        if rank <= 0 {
            bail!("rank must be positive");
        }
        if alpha <= 0.0 {
            bail!("alpha must be positive");
        }
        if alpha.fract() != 0.0 || alpha > i64::MAX as f64 {
            bail!("alpha must be representable as an integer for safetensors metadata");
        }
        Ok(Self {
            target_layers: vec![0],
            target_modules: vec![QwenLoraTargetModule::QProj, QwenLoraTargetModule::VProj],
            rank,
            alpha: alpha as i64,
        })
    }

    fn alpha_f64(&self) -> f64 {
        self.alpha as f64
    }

    fn target_module_names(&self) -> Vec<String> {
        self.target_modules
            .iter()
            .map(|module| module.as_str().to_string())
            .collect()
    }

    fn includes(&self, module: QwenLoraTargetModule) -> bool {
        self.target_modules.contains(&module)
    }
}

impl QwenLoraRegistry {
    fn zeros(weights: &BTreeMap<String, Tensor>, config: &QwenLoraConfig) -> Result<Self> {
        Self::build(weights, config, false, false)
    }

    fn deterministic(
        weights: &BTreeMap<String, Tensor>,
        config: &QwenLoraConfig,
        trainable: bool,
    ) -> Result<Self> {
        Self::build(weights, config, true, trainable)
    }

    fn build(
        weights: &BTreeMap<String, Tensor>,
        config: &QwenLoraConfig,
        deterministic: bool,
        trainable: bool,
    ) -> Result<Self> {
        if !config.includes(QwenLoraTargetModule::QProj)
            || !config.includes(QwenLoraTargetModule::VProj)
        {
            bail!("current Qwen attention LoRA registry requires q_proj and v_proj targets");
        }
        let mut adapters = BTreeMap::new();
        for layer_index in &config.target_layers {
            let layer = QwenLayerWeights::load(weights, *layer_index)?;
            let adapter = if deterministic {
                if trainable {
                    QwenAttentionLoraAdapter::deterministic_trainable(
                        layer.q_proj.size()[1],
                        layer.q_proj.size()[0],
                        layer.v_proj.size()[0],
                        config.rank,
                        config.alpha_f64(),
                    )
                } else {
                    QwenAttentionLoraAdapter::deterministic(
                        layer.q_proj.size()[1],
                        layer.q_proj.size()[0],
                        layer.v_proj.size()[0],
                        config.rank,
                        config.alpha_f64(),
                    )
                }
            } else {
                QwenAttentionLoraAdapter::zeros(
                    layer.q_proj.size()[1],
                    layer.q_proj.size()[0],
                    layer.v_proj.size()[0],
                    config.rank,
                    config.alpha_f64(),
                )
            };
            adapters.insert(*layer_index, adapter);
        }
        Ok(Self {
            config: config.clone(),
            adapters,
        })
    }

    fn layer_adapter(&self, layer_index: usize) -> Result<&QwenAttentionLoraAdapter> {
        self.adapters
            .get(&layer_index)
            .ok_or_else(|| anyhow!("missing LoRA adapter for layer {layer_index}"))
    }

    fn trainable_tensor_names(&self) -> Vec<String> {
        self.trainable_tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    fn trainable_tensors(&self) -> Vec<(String, Tensor)> {
        self.adapters
            .iter()
            .flat_map(|(layer_index, adapter)| adapter.trainable_tensors(*layer_index))
            .collect()
    }

    fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let mut entries: Vec<(String, Tensor)> = Vec::new();
        entries.push((
            "config.rank".to_string(),
            Tensor::from_slice(&[self.config.rank]),
        ));
        entries.push((
            "config.alpha".to_string(),
            Tensor::from_slice(&[self.config.alpha]),
        ));
        let layers: Vec<i64> = self
            .config
            .target_layers
            .iter()
            .map(|layer| *layer as i64)
            .collect();
        entries.push((
            "config.target_layers".to_string(),
            Tensor::from_slice(&layers),
        ));
        let modules: Vec<i64> = self
            .config
            .target_modules
            .iter()
            .map(|module| match module {
                QwenLoraTargetModule::QProj => 0,
                QwenLoraTargetModule::VProj => 1,
            })
            .collect();
        entries.push((
            "config.target_modules".to_string(),
            Tensor::from_slice(&modules),
        ));
        for (layer_index, adapter) in &self.adapters {
            entries.extend(adapter.safetensor_entries(*layer_index));
        }
        let refs: Vec<(&str, &Tensor)> = entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&refs, path)
            .with_context(|| format!("failed to write {}", path.display()))
    }

    fn load(path: &Path) -> Result<Self> {
        let tensors = read_safetensors_map(path)?;
        let rank = tensor(&tensors, "config.rank")?.int64_value(&[0]);
        let alpha = tensor(&tensors, "config.alpha")?.int64_value(&[0]);
        let target_layers: Vec<usize> =
            Vec::<i64>::try_from(tensor(&tensors, "config.target_layers")?)?
                .into_iter()
                .map(|layer| layer as usize)
                .collect();
        let target_modules: Vec<QwenLoraTargetModule> =
            Vec::<i64>::try_from(tensor(&tensors, "config.target_modules")?)?
                .into_iter()
                .map(|module| match module {
                    0 => Ok(QwenLoraTargetModule::QProj),
                    1 => Ok(QwenLoraTargetModule::VProj),
                    other => Err(anyhow!("unknown LoRA target module id {other}")),
                })
                .collect::<Result<Vec<_>>>()?;
        let config = QwenLoraConfig {
            target_layers,
            target_modules,
            rank,
            alpha,
        };
        let mut adapters = BTreeMap::new();
        for layer_index in &config.target_layers {
            adapters.insert(
                *layer_index,
                QwenAttentionLoraAdapter::load_from_tensors(&tensors, *layer_index, &config)?,
            );
        }
        Ok(Self { config, adapters })
    }
}

impl QwenAttentionLoraAdapter {
    fn zeros(
        in_features: i64,
        q_out_features: i64,
        v_out_features: i64,
        rank: i64,
        alpha: f64,
    ) -> Self {
        Self {
            q_a: Tensor::zeros([rank, in_features], (Kind::Float, Device::Cpu)),
            q_b: Tensor::zeros([q_out_features, rank], (Kind::Float, Device::Cpu)),
            v_a: Tensor::zeros([rank, in_features], (Kind::Float, Device::Cpu)),
            v_b: Tensor::zeros([v_out_features, rank], (Kind::Float, Device::Cpu)),
            rank,
            alpha,
        }
    }

    fn deterministic(
        in_features: i64,
        q_out_features: i64,
        v_out_features: i64,
        rank: i64,
        alpha: f64,
    ) -> Self {
        let q_a = deterministic_lora_tensor([rank, in_features], 0.0005);
        let q_b = deterministic_lora_tensor([q_out_features, rank], -0.0003);
        let v_a = deterministic_lora_tensor([rank, in_features], -0.0004);
        let v_b = deterministic_lora_tensor([v_out_features, rank], 0.0002);
        Self {
            q_a,
            q_b,
            v_a,
            v_b,
            rank,
            alpha,
        }
    }

    fn deterministic_trainable(
        in_features: i64,
        q_out_features: i64,
        v_out_features: i64,
        rank: i64,
        alpha: f64,
    ) -> Self {
        let adapter = Self::deterministic(in_features, q_out_features, v_out_features, rank, alpha);
        let _ = adapter.q_a.set_requires_grad(true);
        let _ = adapter.q_b.set_requires_grad(true);
        let _ = adapter.v_a.set_requires_grad(true);
        let _ = adapter.v_b.set_requires_grad(true);
        adapter
    }

    #[cfg(test)]
    fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let rank = Tensor::from_slice(&[self.rank]);
        let alpha = Tensor::from_slice(&[self.alpha as f32]);
        Tensor::write_safetensors(
            &[
                (&"q_proj.lora_a", &self.q_a),
                (&"q_proj.lora_b", &self.q_b),
                (&"v_proj.lora_a", &self.v_a),
                (&"v_proj.lora_b", &self.v_b),
                (&"rank", &rank),
                (&"alpha", &alpha),
            ],
            path,
        )
        .with_context(|| format!("failed to write {}", path.display()))
    }

    #[cfg(test)]
    fn load(path: &Path) -> Result<Self> {
        let tensors = read_safetensors_map(path)?;
        let q_a = tensor(&tensors, "q_proj.lora_a")?.to_kind(Kind::Float);
        let q_b = tensor(&tensors, "q_proj.lora_b")?.to_kind(Kind::Float);
        let v_a = tensor(&tensors, "v_proj.lora_a")?.to_kind(Kind::Float);
        let v_b = tensor(&tensors, "v_proj.lora_b")?.to_kind(Kind::Float);
        let rank = tensor(&tensors, "rank")?.int64_value(&[0]);
        let alpha = tensor(&tensors, "alpha")?.double_value(&[0]);
        Ok(Self {
            q_a,
            q_b,
            v_a,
            v_b,
            rank,
            alpha,
        })
    }

    fn load_from_tensors(
        tensors: &BTreeMap<String, Tensor>,
        layer_index: usize,
        config: &QwenLoraConfig,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_index}.self_attn");
        let q_a = tensor(tensors, &format!("{prefix}.q_proj.lora_a"))?.to_kind(Kind::Float);
        let q_b = tensor(tensors, &format!("{prefix}.q_proj.lora_b"))?.to_kind(Kind::Float);
        let v_a = tensor(tensors, &format!("{prefix}.v_proj.lora_a"))?.to_kind(Kind::Float);
        let v_b = tensor(tensors, &format!("{prefix}.v_proj.lora_b"))?.to_kind(Kind::Float);
        Ok(Self {
            q_a,
            q_b,
            v_a,
            v_b,
            rank: config.rank,
            alpha: config.alpha_f64(),
        })
    }

    fn safetensor_entries(&self, layer_index: usize) -> Vec<(String, Tensor)> {
        let prefix = format!("model.layers.{layer_index}.self_attn");
        vec![
            (format!("{prefix}.q_proj.lora_a"), self.q_a.shallow_clone()),
            (format!("{prefix}.q_proj.lora_b"), self.q_b.shallow_clone()),
            (format!("{prefix}.v_proj.lora_a"), self.v_a.shallow_clone()),
            (format!("{prefix}.v_proj.lora_b"), self.v_b.shallow_clone()),
        ]
    }

    fn trainable_tensors(&self, layer_index: usize) -> Vec<(String, Tensor)> {
        let prefix = format!("model.layers.{layer_index}.self_attn");
        vec![
            (format!("{prefix}.q_proj.lora_a"), self.q_a.shallow_clone()),
            (format!("{prefix}.q_proj.lora_b"), self.q_b.shallow_clone()),
            (format!("{prefix}.v_proj.lora_a"), self.v_a.shallow_clone()),
            (format!("{prefix}.v_proj.lora_b"), self.v_b.shallow_clone()),
        ]
    }

    fn q_delta(&self, device: Device) -> Tensor {
        self.q_b
            .to_device(device)
            .matmul(&self.q_a.to_device(device))
            * (self.alpha / self.rank as f64)
    }

    fn v_delta(&self, device: Device) -> Tensor {
        self.v_b
            .to_device(device)
            .matmul(&self.v_a.to_device(device))
            * (self.alpha / self.rank as f64)
    }
}

fn deterministic_lora_tensor<const N: usize>(shape: [i64; N], scale: f64) -> Tensor {
    let len = shape.iter().product::<i64>() as usize;
    let values: Vec<f32> = (0..len)
        .map(|index| ((index % 17) as f64 - 8.0) as f32 * scale as f32)
        .collect();
    Tensor::from_slice(&values).reshape(shape)
}

fn tensor_snapshot(tensor: &Tensor) -> Tensor {
    let mut snapshot = Tensor::zeros_like(tensor);
    snapshot.copy_(tensor);
    snapshot
}

impl QwenLayerWeights {
    pub fn load(weights: &BTreeMap<String, Tensor>, layer_index: usize) -> Result<Self> {
        Self::load_with_kind(weights, layer_index, Kind::Float)
    }

    fn load_with_kind(
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
    after_attention + mlp_output
}

pub fn qwen_forward_from_ids(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
) -> Result<Tensor> {
    qwen_forward_from_ids_with_kind(input_ids, weights, config, Kind::Float)
}

fn qwen_forward_from_ids_with_kind(
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

fn qwen_forward_with_cache(
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

fn qwen_layer_with_cache(
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

fn qwen_causal_lm_loss(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
) -> Result<Tensor> {
    qwen_causal_lm_loss_with_kind(input_ids, weights, config, Kind::Float)
}

fn qwen_causal_lm_loss_with_kind(
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

pub fn qwen_greedy_generate(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    max_new_tokens: usize,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    for _ in 0..max_new_tokens {
        let logits = qwen_forward_from_ids(&generated, weights, config)?;
        let next_token = logits.i((0, -1)).argmax(-1, false).reshape([1, 1]);
        generated = Tensor::cat(&[&generated, &next_token], 1);
    }
    Ok(generated)
}

pub fn qwen_greedy_generate_with_cache(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    max_new_tokens: usize,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    let (logits, mut cache) = qwen_forward_with_cache(input_ids, weights, config, None)?;
    let mut next_token = logits.i((0, -1)).argmax(-1, false).reshape([1, 1]);

    for step in 0..max_new_tokens {
        generated = Tensor::cat(&[&generated, &next_token], 1);
        if step + 1 == max_new_tokens {
            break;
        }
        let (decode_logits, updated_cache) =
            qwen_forward_with_cache(&next_token, weights, config, Some(cache))?;
        cache = updated_cache;
        next_token = decode_logits.i((0, -1)).argmax(-1, false).reshape([1, 1]);
    }

    Ok(generated)
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_sample_generate(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..max_new_tokens {
        let logits = qwen_forward_from_ids(&generated, weights, config)?;
        let next_token =
            sample_token_from_logits(&logits.i((0, -1)), temperature, top_k, top_p, &mut rng)?
                .reshape([1, 1]);
        generated = Tensor::cat(&[&generated, &next_token], 1);
    }
    Ok(generated)
}

#[allow(clippy::too_many_arguments)]
pub fn qwen_sample_generate_with_cache(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    let mut rng = StdRng::seed_from_u64(seed);
    let (logits, mut cache) = qwen_forward_with_cache(input_ids, weights, config, None)?;
    let mut next_token =
        sample_token_from_logits(&logits.i((0, -1)), temperature, top_k, top_p, &mut rng)?
            .reshape([1, 1]);

    for step in 0..max_new_tokens {
        generated = Tensor::cat(&[&generated, &next_token], 1);
        if step + 1 == max_new_tokens {
            break;
        }
        let (decode_logits, updated_cache) =
            qwen_forward_with_cache(&next_token, weights, config, Some(cache))?;
        cache = updated_cache;
        next_token = sample_token_from_logits(
            &decode_logits.i((0, -1)),
            temperature,
            top_k,
            top_p,
            &mut rng,
        )?
        .reshape([1, 1]);
    }

    Ok(generated)
}

fn sample_token_from_logits(
    logits: &Tensor,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    rng: &mut StdRng,
) -> Result<Tensor> {
    if temperature <= 0.0 {
        bail!("temperature must be positive");
    }
    if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
        bail!("top_p must be in (0, 1]");
    }

    let logits: Vec<f32> =
        Vec::<f32>::try_from(logits.to_kind(Kind::Float).to_device(Device::Cpu))?;
    let mut candidates: Vec<(i64, f64)> = logits
        .into_iter()
        .enumerate()
        .filter_map(|(token_id, logit)| {
            let scaled = f64::from(logit) / temperature;
            scaled.is_finite().then_some((token_id as i64, scaled))
        })
        .collect();
    if candidates.is_empty() {
        bail!("no finite logits available for sampling");
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    if top_k > 0 && top_k < candidates.len() {
        candidates.truncate(top_k);
    }

    let max_logit = candidates[0].1;
    let mut probs: Vec<(i64, f64)> = candidates
        .into_iter()
        .map(|(token_id, logit)| (token_id, (logit - max_logit).exp()))
        .collect();
    let total: f64 = probs.iter().map(|(_, prob)| *prob).sum();
    if total <= 0.0 || !total.is_finite() {
        bail!("sampling probabilities are not finite");
    }
    for (_, prob) in &mut probs {
        *prob /= total;
    }

    if top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut keep = 0usize;
        for (_, prob) in &probs {
            keep += 1;
            cumulative += *prob;
            if cumulative >= top_p {
                break;
            }
        }
        probs.truncate(keep.max(1));
    }

    let renorm_total: f64 = probs.iter().map(|(_, prob)| *prob).sum();
    let mut draw = rng.gen_range(0.0..renorm_total);
    for (token_id, prob) in probs {
        if draw <= prob {
            return Ok(Tensor::from_slice(&[token_id]).to_kind(Kind::Int64));
        }
        draw -= prob;
    }

    bail!("sampling draw did not select a token")
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

fn qwen_attention_with_lora(
    input: &Tensor,
    weights: &QwenLayerWeights,
    adapter: &QwenAttentionLoraAdapter,
    config: &QwenRuntimeConfig,
) -> Tensor {
    let q_proj = &weights.q_proj + adapter.q_delta(input.device());
    let v_proj = &weights.v_proj + adapter.v_delta(input.device());
    qwen_attention(
        input,
        &q_proj,
        &weights.q_bias,
        &weights.k_proj,
        &weights.k_bias,
        &v_proj,
        &weights.v_bias,
        &weights.o_proj,
        config,
    )
}

fn qwen_attention_lora_mse_loss(
    input: &Tensor,
    target: &Tensor,
    weights: &QwenLayerWeights,
    adapter: &QwenAttentionLoraAdapter,
    config: &QwenRuntimeConfig,
) -> Tensor {
    qwen_attention_with_lora(input, weights, adapter, config).mse_loss(target, Reduction::Mean)
}

impl QwenSftDataset {
    fn from_instruction_pairs(tokenizer: &Tokenizer, examples: &[QwenSftExample]) -> Result<Self> {
        if examples.is_empty() {
            bail!("SFT dataset must contain at least one example");
        }
        let samples = examples
            .iter()
            .map(|example| qwen_sft_token_sample(tokenizer, example))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            samples,
            pad_token_id: qwen_pad_token_id(tokenizer),
        })
    }

    fn from_jsonl_path(tokenizer: &Tokenizer, path: &Path) -> Result<Self> {
        Self::from_instruction_pairs(tokenizer, &qwen_sft_examples_from_jsonl_path(path)?)
    }

    fn padded_batch(&self, start: usize, batch_size: usize) -> Result<QwenSftBatch> {
        if batch_size == 0 {
            bail!("SFT batch size must be positive");
        }
        if self.samples.is_empty() {
            bail!("SFT dataset must contain at least one sample");
        }
        let samples = (0..batch_size)
            .map(|offset| {
                let index = (start + offset) % self.samples.len();
                self.samples[index].clone()
            })
            .collect::<Vec<_>>();
        qwen_sft_padded_batch(&samples, self.pad_token_id)
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

fn qwen_sft_examples_from_jsonl_path(path: &Path) -> Result<Vec<QwenSftExample>> {
    let mut files = Vec::new();
    if path.is_dir() {
        let mut sorted = BTreeSet::new();
        for entry in
            fs::read_dir(path).with_context(|| format!("failed to list {}", path.display()))?
        {
            let entry =
                entry.with_context(|| format!("failed to read entry in {}", path.display()))?;
            let file_type = entry
                .file_type()
                .with_context(|| format!("failed to inspect {}", entry.path().display()))?;
            if file_type.is_file() {
                sorted.insert(entry.path());
            }
        }
        files.extend(sorted);
    } else {
        files.push(path.to_path_buf());
    }

    if files.is_empty() {
        bail!("SFT JSONL path {} did not contain files", path.display());
    }

    let mut examples = Vec::new();
    for file in files {
        let contents = fs::read_to_string(&file)
            .with_context(|| format!("failed to read {}", file.display()))?;
        for (line_index, line) in contents.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let record: QwenSftRecord = serde_json::from_str(line).with_context(|| {
                format!(
                    "failed to parse SFT JSONL record {}:{}",
                    file.display(),
                    line_index + 1
                )
            })?;
            examples.push(QwenSftExample {
                instruction: record.instruction,
                input: record.input,
                response: record.response,
            });
        }
    }

    if examples.is_empty() {
        bail!("SFT JSONL path {} did not contain examples", path.display());
    }
    Ok(examples)
}

fn qwen_sft_token_sample(
    tokenizer: &Tokenizer,
    example: &QwenSftExample,
) -> Result<QwenSftTokenSample> {
    let prompt = if example.input.trim().is_empty() {
        format!("Instruction:\n{}\n\nResponse:\n", example.instruction)
    } else {
        format!(
            "Instruction:\n{}\n\nInput:\n{}\n\nResponse:\n",
            example.instruction, example.input
        )
    };
    qwen_sft_token_sample_from_prompt(tokenizer, &prompt, &example.response)
}

fn qwen_sft_token_sample_from_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    response: &str,
) -> Result<QwenSftTokenSample> {
    let response = format!("{response}\n");
    let prompt_encoding = tokenizer
        .encode(prompt, false)
        .map_err(|error| anyhow!("failed to encode prompt: {error}"))?;
    let response_encoding = tokenizer
        .encode(response.as_str(), false)
        .map_err(|error| anyhow!("failed to encode response: {error}"))?;
    let prompt_tokens: Vec<i64> = prompt_encoding
        .get_ids()
        .iter()
        .map(|token| i64::from(*token))
        .collect();
    let response_tokens: Vec<i64> = response_encoding
        .get_ids()
        .iter()
        .map(|token| i64::from(*token))
        .collect();
    if prompt_tokens.is_empty() || response_tokens.is_empty() {
        bail!("SFT prompt and response must both tokenize to at least one token");
    }

    let mut token_ids = prompt_tokens.clone();
    token_ids.extend(response_tokens.iter().copied());
    if token_ids.len() < 2 {
        bail!("SFT sample must contain at least two tokens");
    }
    let target_len = token_ids.len() - 1;
    let prompt_len = prompt_tokens.len();
    let mask_values: Vec<f32> = (0..target_len)
        .map(|target_index| {
            if target_index + 1 >= prompt_len {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let masked_positions = mask_values.iter().filter(|value| **value > 0.0).count();
    if masked_positions == 0 {
        bail!("SFT response-only mask is empty");
    }

    Ok(QwenSftTokenSample {
        prompt_tokens: prompt_tokens.len(),
        response_tokens: response_tokens.len(),
        masked_positions,
        token_ids,
        mask_values,
    })
}

fn qwen_pad_token_id(tokenizer: &Tokenizer) -> i64 {
    tokenizer
        .get_padding()
        .map(|padding| i64::from(padding.pad_id))
        .or_else(|| tokenizer.token_to_id("<|endoftext|>").map(i64::from))
        .unwrap_or(0)
}

fn qwen_sft_padded_batch(
    samples: &[QwenSftTokenSample],
    pad_token_id: i64,
) -> Result<QwenSftBatch> {
    if samples.is_empty() {
        bail!("SFT batch must contain at least one sample");
    }
    let max_len = samples
        .iter()
        .map(|sample| sample.token_ids.len())
        .max()
        .ok_or_else(|| anyhow!("SFT batch must contain at least one sample"))?;
    if max_len < 2 {
        bail!("SFT batch sequence length must be at least two tokens");
    }

    let batch_size = samples.len();
    let mut input_values = Vec::with_capacity(batch_size * max_len);
    let mut mask_values = Vec::with_capacity(batch_size * (max_len - 1));
    let mut prompt_tokens = Vec::with_capacity(batch_size);
    let mut response_tokens = Vec::with_capacity(batch_size);
    let mut masked_positions = 0usize;
    let mut padding_tokens = 0usize;

    for sample in samples {
        prompt_tokens.push(sample.prompt_tokens);
        response_tokens.push(sample.response_tokens);
        input_values.extend(sample.token_ids.iter().copied());
        let pad_len = max_len - sample.token_ids.len();
        input_values.extend(std::iter::repeat(pad_token_id).take(pad_len));
        padding_tokens += pad_len;

        mask_values.extend(sample.mask_values.iter().copied());
        masked_positions += sample.masked_positions;
        mask_values.extend(std::iter::repeat(0.0).take(max_len - 1 - sample.mask_values.len()));
    }

    if masked_positions == 0 {
        bail!("SFT batch response-only mask is empty");
    }

    Ok(QwenSftBatch {
        input_ids: Tensor::from_slice(&input_values)
            .to_kind(Kind::Int64)
            .reshape([batch_size as i64, max_len as i64]),
        target_mask: Tensor::from_slice(&mask_values).reshape([
            batch_size as i64,
            (max_len - 1) as i64,
            1,
        ]),
        prompt_tokens,
        response_tokens,
        masked_positions,
        padding_tokens,
    })
}

fn qwen_attention_lora_sft_loss(
    input_ids: &Tensor,
    target_mask: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    adapter: &QwenAttentionLoraAdapter,
    config: &QwenRuntimeConfig,
) -> Result<Tensor> {
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?.to_kind(Kind::Float);
    let layer0 = QwenLayerWeights::load(weights, 0)?;
    let hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);
    let attention_input = rms_norm(&hidden, &layer0.input_norm, config.rms_norm_eps);
    let base_output = qwen_attention(
        &attention_input,
        &layer0.q_proj,
        &layer0.q_bias,
        &layer0.k_proj,
        &layer0.k_bias,
        &layer0.v_proj,
        &layer0.v_bias,
        &layer0.o_proj,
        config,
    );
    let target = lora_train_target(&base_output);
    let adapted = qwen_attention_with_lora(&attention_input, &layer0, adapter, config);
    let shifted_adapted = adapted.narrow(1, 0, input_ids.size()[1] - 1);
    let shifted_target = target.narrow(1, 0, input_ids.size()[1] - 1);
    let mask = target_mask.to_device(adapted.device());
    let squared = (shifted_adapted - shifted_target).pow_tensor_scalar(2.0) * &mask;
    Ok(squared.sum(Kind::Float) / mask.sum(Kind::Float))
}

fn lora_train_target(base_output: &Tensor) -> Tensor {
    let values = Tensor::arange(
        base_output.numel() as i64,
        (Kind::Float, base_output.device()),
    )
    .reshape(base_output.size())
    .fmod(11.0)
        / 10_000.0;
    base_output + values
}

#[allow(clippy::too_many_arguments)]
fn qwen_attention_with_cache(
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

fn rope_cos_sin(seq_len: i64, head_dim: i64, theta: f64, device: Device) -> (Tensor, Tensor) {
    rope_cos_sin_with_offset(seq_len, head_dim, theta, device, 0)
}

fn rope_cos_sin_with_offset(
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

fn apply_rotary(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    input * cos + rotate_half(input) * sin
}

fn rotate_half(input: &Tensor) -> Tensor {
    let last_dim = input.size()[input.dim() - 1];
    let half = last_dim / 2;
    let first = input.narrow(-1, 0, half);
    let second = input.narrow(-1, half, half);
    Tensor::cat(&[&(-second), &first], -1)
}

fn repeat_kv(input: &Tensor, repeats: i64) -> Tensor {
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
    fn representative_full_train_tensors_get_gradients_and_reload() {
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let mut weights = tiny_qwen_weights();
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let mut registry =
            QwenTrainableRegistry::representative(&mut weights).expect("registry should build");
        assert_eq!(
            registry.parameter_names(),
            representative_trainable_qwen_tensors()
        );

        let initial_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        let loss = qwen_causal_lm_loss(&input_ids, &weights, &config).expect("loss should run");
        loss.backward();
        let artifacts = registry
            .adamw_step(&mut weights, 1e-2, 1)
            .expect("optimizer step should apply");
        assert_eq!(
            artifacts.tensor_summaries.len(),
            representative_trainable_qwen_tensors().len()
        );
        assert_eq!(
            artifacts.manifest_tensors.len(),
            representative_trainable_qwen_tensors().len()
        );
        assert_eq!(
            artifacts.optimizer_entries.len(),
            representative_trainable_qwen_tensors().len() * 2
        );
        for summary in &artifacts.tensor_summaries {
            assert!(
                summary.grad_defined,
                "{} should receive a gradient",
                summary.name
            );
            assert!(
                summary.grad_norm > 0.0,
                "{} grad should be non-zero",
                summary.name
            );
            assert!(
                summary.delta_norm > 0.0,
                "{} delta should be non-zero",
                summary.name
            );
        }

        let final_loss = qwen_causal_lm_loss(&input_ids, &weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!(final_loss < initial_loss);

        let mut reloaded_weights = tiny_qwen_weights();
        let delta_tensors: BTreeMap<String, Tensor> = artifacts
            .delta_entries
            .into_iter()
            .map(|(name, tensor)| (name, tensor))
            .collect();
        QwenTrainableRegistry::apply_delta_checkpoint(
            &mut reloaded_weights,
            &delta_tensors,
            &artifacts.manifest_tensors,
        )
        .expect("delta reload should apply");
        let reloaded_loss = qwen_causal_lm_loss(&input_ids, &reloaded_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!((final_loss - reloaded_loss).abs() < 1e-6);
    }

    #[test]
    fn sampling_respects_top_k_and_top_p_filters() {
        let logits = Tensor::from_slice(&[0.0_f32, 1.0, 2.0, 3.0]);
        let mut rng = StdRng::seed_from_u64(7);

        let token =
            sample_token_from_logits(&logits, 0.8, 1, 0.5, &mut rng).expect("sample should run");

        assert_eq!(token.int64_value(&[0]), 3);
    }

    #[test]
    fn qwen_delta_manifest_roundtrips() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let delta_output = temp.path().join("delta.safetensors");
        let manifest_output = delta_manifest_path(&delta_output);
        let manifest = QwenDeltaCheckpointManifest {
            format: "rustrain.qwen_delta.v1".to_string(),
            base_model_path: "/models/qwen".to_string(),
            reference_fixture: "fixture.safetensors".to_string(),
            delta_safetensors: delta_output.display().to_string(),
            optimizer_safetensors: Some(optimizer_state_path(&delta_output).display().to_string()),
            train_step: 1,
            learning_rate: 1e-6,
            initial_loss: 2.0,
            final_loss: 1.5,
            tensors: vec![QwenDeltaTensorManifestEntry {
                name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                delta_name: "model.layers.0.self_attn.q_proj.weight.delta".to_string(),
                adam_m_name: Some("model.layers.0.self_attn.q_proj.weight.adam_m".to_string()),
                adam_v_name: Some("model.layers.0.self_attn.q_proj.weight.adam_v".to_string()),
                shape: vec![4, 4],
                dtype: "float32".to_string(),
                grad_norm: 3.0,
                delta_norm: 0.1,
            }],
        };

        write_qwen_delta_manifest(&manifest_output, &manifest).expect("manifest should write");
        let reloaded: QwenDeltaCheckpointManifest = serde_json::from_str(
            &fs::read_to_string(&manifest_output).expect("manifest should read"),
        )
        .expect("manifest should parse");

        assert_eq!(manifest_output, temp.path().join("delta.safetensors.json"));
        assert_eq!(
            optimizer_state_path(&delta_output),
            temp.path().join("delta.safetensors.optimizer.safetensors")
        );
        assert_eq!(reloaded.format, "rustrain.qwen_delta.v1");
        assert_eq!(
            reloaded.optimizer_safetensors,
            manifest.optimizer_safetensors
        );
        assert_eq!(
            reloaded.tensors[0].delta_name,
            manifest.tensors[0].delta_name
        );
        assert_eq!(
            reloaded.tensors[0].adam_m_name,
            manifest.tensors[0].adam_m_name
        );
    }

    #[test]
    fn qwen_optimizer_slots_reload_reproduces_next_adam_step() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let optimizer_output = temp.path().join("optimizer.safetensors");
        let tensor_name = "model.layers.0.self_attn.q_proj.weight";
        let slot_names = adam_slot_names(tensor_name);
        let first_grad = Tensor::from_slice(&[0.5_f32, -0.25, 0.125, -0.75]).reshape([2, 2]);
        let second_grad = Tensor::from_slice(&[-0.2_f32, 0.4, -0.6, 0.8]).reshape([2, 2]);
        let base_weight = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape([2, 2]);
        let learning_rate = 1e-3;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;

        let first_state = adamw_next_state(None, &first_grad, beta1, beta2);
        let first_update = adamw_update(&first_state, learning_rate, beta1, beta2, 1, eps);
        let after_first = &base_weight - first_update;
        Tensor::write_safetensors(
            &[
                (slot_names.m.as_str(), &first_state.m),
                (slot_names.v.as_str(), &first_state.v),
            ],
            &optimizer_output,
        )
        .expect("optimizer slots should write");

        let reloaded_slots = read_safetensors_map(&optimizer_output).expect("slots should reload");
        let reloaded_state = AdamState {
            m: tensor(&reloaded_slots, &slot_names.m)
                .expect("m slot should exist")
                .to_kind(Kind::Float),
            v: tensor(&reloaded_slots, &slot_names.v)
                .expect("v slot should exist")
                .to_kind(Kind::Float),
        };
        let continuous_second_state =
            adamw_next_state(Some(&first_state), &second_grad, beta1, beta2);
        let reloaded_second_state =
            adamw_next_state(Some(&reloaded_state), &second_grad, beta1, beta2);
        let continuous_after_second = &after_first
            - adamw_update(
                &continuous_second_state,
                learning_rate,
                beta1,
                beta2,
                2,
                eps,
            );
        let reloaded_after_second = &after_first
            - adamw_update(&reloaded_second_state, learning_rate, beta1, beta2, 2, eps);

        assert!(
            diff_stats(&continuous_second_state.m, &reloaded_second_state.m)
                .expect("m state diff should compute")
                .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(&continuous_second_state.v, &reloaded_second_state.v)
                .expect("v state diff should compute")
                .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(&continuous_after_second, &reloaded_after_second)
                .expect("weight diff should compute")
                .max_abs
                < 1e-8
        );
    }

    #[test]
    fn qwen_manifest_resume_reproduces_second_full_train_step() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let delta_output = temp.path().join("delta.safetensors");
        let optimizer_output = optimizer_state_path(&delta_output);
        let manifest_output = delta_manifest_path(&delta_output);
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let learning_rate = 1e-2;

        let mut continuous_weights = tiny_qwen_weights();
        let mut continuous_registry =
            QwenTrainableRegistry::representative(&mut continuous_weights)
                .expect("registry should build");
        let initial_loss = qwen_causal_lm_loss(&input_ids, &continuous_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        let first_loss =
            qwen_causal_lm_loss(&input_ids, &continuous_weights, &config).expect("loss should run");
        first_loss.backward();
        let first_artifacts = continuous_registry
            .adamw_step(&mut continuous_weights, learning_rate, 1)
            .expect("first optimizer step should apply");
        let final_loss = qwen_causal_lm_loss(&input_ids, &continuous_weights, &config)
            .expect("loss should run")
            .double_value(&[]);

        let delta_refs: Vec<(&str, &Tensor)> = first_artifacts
            .delta_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&delta_refs, &delta_output).expect("delta should write");
        let optimizer_refs: Vec<(&str, &Tensor)> = first_artifacts
            .optimizer_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&optimizer_refs, &optimizer_output)
            .expect("optimizer should write");
        let manifest = QwenDeltaCheckpointManifest {
            format: "rustrain.qwen_delta.v1".to_string(),
            base_model_path: "tiny-qwen".to_string(),
            reference_fixture: "inline".to_string(),
            delta_safetensors: delta_output.display().to_string(),
            optimizer_safetensors: Some(optimizer_output.display().to_string()),
            train_step: 1,
            learning_rate,
            initial_loss,
            final_loss,
            tensors: first_artifacts.manifest_tensors,
        };
        write_qwen_delta_manifest(&manifest_output, &manifest).expect("manifest should write");
        let reloaded_manifest: QwenDeltaCheckpointManifest = serde_json::from_str(
            &fs::read_to_string(&manifest_output).expect("manifest should read"),
        )
        .expect("manifest should parse");

        let mut resumed_weights = tiny_qwen_weights();
        let mut resumed_registry =
            QwenTrainableRegistry::load_from_manifest(&mut resumed_weights, &reloaded_manifest)
                .expect("registry should load from manifest");
        let resumed_loss = qwen_causal_lm_loss(&input_ids, &resumed_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!((final_loss - resumed_loss).abs() < 1e-6);

        continuous_registry.zero_grad();
        let continuous_second_loss =
            qwen_causal_lm_loss(&input_ids, &continuous_weights, &config).expect("loss should run");
        continuous_second_loss.backward();
        continuous_registry
            .adamw_step(&mut continuous_weights, learning_rate, 2)
            .expect("continuous second step should apply");

        let resumed_second_loss =
            qwen_causal_lm_loss(&input_ids, &resumed_weights, &config).expect("loss should run");
        resumed_second_loss.backward();
        resumed_registry
            .adamw_step(&mut resumed_weights, learning_rate, 2)
            .expect("resumed second step should apply");

        let continuous_after_second = qwen_causal_lm_loss(&input_ids, &continuous_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        let resumed_after_second = qwen_causal_lm_loss(&input_ids, &resumed_weights, &config)
            .expect("loss should run")
            .double_value(&[]);
        assert!((continuous_after_second - resumed_after_second).abs() < 1e-6);

        for name in representative_trainable_qwen_tensors() {
            let diff = diff_stats(
                tensor(&continuous_weights, name).expect("continuous tensor should exist"),
                tensor(&resumed_weights, name).expect("resumed tensor should exist"),
            )
            .expect("diff should compute");
            assert!(
                diff.max_abs < 1e-6,
                "{name} should match after manifest-resumed second step, max_abs={}",
                diff.max_abs
            );
        }
    }

    #[test]
    fn qwen_trainable_session_trains_and_resumes_from_manifest() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let delta_output = temp.path().join("session-delta.safetensors");
        let optimizer_output = optimizer_state_path(&delta_output);
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2, 3]).reshape([1, 4]);
        let learning_rate = 1e-2;

        let mut continuous_session = QwenTrainableSession::from_weights(
            config,
            tiny_qwen_weights(),
            input_ids.shallow_clone(),
            Kind::Float,
        )
        .expect("session should build");
        let first_step = continuous_session
            .train_step(learning_rate, 1)
            .expect("first step should train");
        assert!(first_step.loss_after < first_step.loss_before);

        let delta_refs: Vec<(&str, &Tensor)> = first_step
            .artifacts
            .delta_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&delta_refs, &delta_output).expect("delta should write");
        let optimizer_refs: Vec<(&str, &Tensor)> = first_step
            .artifacts
            .optimizer_entries
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&optimizer_refs, &optimizer_output)
            .expect("optimizer should write");
        let manifest = QwenDeltaCheckpointManifest {
            format: "rustrain.qwen_delta.v1".to_string(),
            base_model_path: "tiny-qwen".to_string(),
            reference_fixture: "inline".to_string(),
            delta_safetensors: delta_output.display().to_string(),
            optimizer_safetensors: Some(optimizer_output.display().to_string()),
            train_step: 1,
            learning_rate,
            initial_loss: first_step.loss_before,
            final_loss: first_step.loss_after,
            tensors: first_step.artifacts.manifest_tensors,
        };
        let mut resumed_session = QwenTrainableSession::from_manifest(
            config,
            tiny_qwen_weights(),
            input_ids,
            Kind::Float,
            &manifest,
        )
        .expect("session should resume");
        assert!((first_step.loss_after - resumed_session.loss_value().unwrap()).abs() < 1e-6);

        let continuous_second = continuous_session
            .train_step(learning_rate, 2)
            .expect("continuous second step should train");
        let resumed_second = resumed_session
            .train_step(learning_rate, 2)
            .expect("resumed second step should train");
        assert!((continuous_second.loss_after - resumed_second.loss_after).abs() < 1e-6);
    }

    #[test]
    fn qwen_attention_lora_adapter_roundtrips_mismatched_q_v_shapes() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let adapter_output = temp.path().join("adapter.safetensors");
        let adapter = QwenAttentionLoraAdapter::deterministic(4, 6, 2, 2, 8.0);

        assert_eq!(adapter.q_delta(Device::Cpu).size(), vec![6, 4]);
        assert_eq!(adapter.v_delta(Device::Cpu).size(), vec![2, 4]);

        adapter.save(&adapter_output).expect("adapter should write");
        let reloaded =
            QwenAttentionLoraAdapter::load(&adapter_output).expect("adapter should reload");

        assert_eq!(reloaded.q_delta(Device::Cpu).size(), vec![6, 4]);
        assert_eq!(reloaded.v_delta(Device::Cpu).size(), vec![2, 4]);
        assert!(
            diff_stats(
                &reloaded.q_delta(Device::Cpu),
                &adapter.q_delta(Device::Cpu)
            )
            .expect("q delta diff should compute")
            .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(
                &reloaded.v_delta(Device::Cpu),
                &adapter.v_delta(Device::Cpu)
            )
            .expect("v delta diff should compute")
            .max_abs
                < 1e-8
        );
    }

    #[test]
    fn qwen_attention_lora_train_step_reduces_tiny_mse_and_reloads() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let adapter_output = temp.path().join("adapter.safetensors");
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let weights = tiny_qwen_weights();
        let layer = QwenLayerWeights::load(&weights, 0).expect("layer should load");
        let input = Tensor::arange(12, (Kind::Float, Device::Cpu)).reshape([1, 3, 4]) / 12.0;
        let target = qwen_attention(
            &input,
            &layer.q_proj,
            &layer.q_bias,
            &layer.k_proj,
            &layer.k_bias,
            &layer.v_proj,
            &layer.v_bias,
            &layer.o_proj,
            &config,
        ) + Tensor::ones([1, 3, 4], (Kind::Float, Device::Cpu)) * 0.01;
        let adapter = QwenAttentionLoraAdapter::deterministic_trainable(4, 4, 2, 2, 8.0);

        let initial_loss = qwen_attention_lora_mse_loss(&input, &target, &layer, &adapter, &config)
            .double_value(&[]);
        let loss = qwen_attention_lora_mse_loss(&input, &target, &layer, &adapter, &config);
        loss.backward();
        for (_, mut tensor) in adapter.trainable_tensors(0) {
            let grad = tensor.grad();
            assert!(grad.defined());
            let _ = no_grad(|| tensor.f_sub_(&(&grad * 1.0))).expect("update should apply");
        }
        let final_loss = qwen_attention_lora_mse_loss(&input, &target, &layer, &adapter, &config)
            .double_value(&[]);
        assert!(final_loss < initial_loss);

        adapter.save(&adapter_output).expect("adapter should save");
        let reloaded =
            QwenAttentionLoraAdapter::load(&adapter_output).expect("adapter should reload");
        let reloaded_loss =
            qwen_attention_lora_mse_loss(&input, &target, &layer, &reloaded, &config)
                .double_value(&[]);
        assert!((final_loss - reloaded_loss).abs() < 1e-8);
    }

    #[test]
    fn qwen_lora_registry_roundtrips_configured_layer_targets() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let adapter_output = temp.path().join("adapter.safetensors");
        let weights = tiny_qwen_weights();
        let config = QwenLoraConfig::layer0_qv(2, 8.0).expect("config should build");
        let registry = QwenLoraRegistry::deterministic(&weights, &config, true)
            .expect("registry should build");

        assert_eq!(registry.config.target_layers, vec![0]);
        assert_eq!(
            registry.config.target_modules,
            vec![QwenLoraTargetModule::QProj, QwenLoraTargetModule::VProj]
        );
        assert_eq!(
            registry.trainable_tensor_names(),
            vec![
                "model.layers.0.self_attn.q_proj.lora_a".to_string(),
                "model.layers.0.self_attn.q_proj.lora_b".to_string(),
                "model.layers.0.self_attn.v_proj.lora_a".to_string(),
                "model.layers.0.self_attn.v_proj.lora_b".to_string(),
            ]
        );

        registry
            .save(&adapter_output)
            .expect("registry should save");
        let reloaded = QwenLoraRegistry::load(&adapter_output).expect("registry should reload");

        assert_eq!(reloaded.config, config);
        assert!(
            diff_stats(
                &reloaded
                    .layer_adapter(0)
                    .expect("layer adapter")
                    .q_delta(Device::Cpu),
                &registry
                    .layer_adapter(0)
                    .expect("layer adapter")
                    .q_delta(Device::Cpu),
            )
            .expect("q delta diff should compute")
            .max_abs
                < 1e-8
        );
        assert!(
            diff_stats(
                &reloaded
                    .layer_adapter(0)
                    .expect("layer adapter")
                    .v_delta(Device::Cpu),
                &registry
                    .layer_adapter(0)
                    .expect("layer adapter")
                    .v_delta(Device::Cpu),
            )
            .expect("v delta diff should compute")
            .max_abs
                < 1e-8
        );
    }

    #[test]
    fn qwen_sft_padded_batch_masks_padding_targets() {
        let samples = vec![
            QwenSftTokenSample {
                prompt_tokens: 2,
                response_tokens: 2,
                masked_positions: 2,
                token_ids: vec![10, 11, 12, 13],
                mask_values: vec![0.0, 1.0, 1.0],
            },
            QwenSftTokenSample {
                prompt_tokens: 1,
                response_tokens: 1,
                masked_positions: 1,
                token_ids: vec![20, 21],
                mask_values: vec![1.0],
            },
        ];

        let batch = qwen_sft_padded_batch(&samples, 0).expect("batch should build");
        let input_values: Vec<i64> = Vec::<i64>::try_from(batch.input_ids.reshape([-1])).unwrap();
        let mask_values: Vec<f32> = Vec::<f32>::try_from(batch.target_mask.reshape([-1])).unwrap();

        assert_eq!(batch.input_ids.size(), vec![2, 4]);
        assert_eq!(batch.target_mask.size(), vec![2, 3, 1]);
        assert_eq!(input_values, vec![10, 11, 12, 13, 20, 21, 0, 0]);
        assert_eq!(mask_values, vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(batch.masked_positions, 3);
        assert_eq!(batch.padding_tokens, 2);
    }

    #[test]
    fn qwen_sft_dataset_builds_wrapping_padded_batches() {
        let dataset = QwenSftDataset {
            samples: vec![
                QwenSftTokenSample {
                    prompt_tokens: 2,
                    response_tokens: 1,
                    masked_positions: 1,
                    token_ids: vec![1, 2, 3],
                    mask_values: vec![0.0, 1.0],
                },
                QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: 2,
                    masked_positions: 2,
                    token_ids: vec![4, 5, 6],
                    mask_values: vec![1.0, 1.0],
                },
            ],
            pad_token_id: 0,
        };

        let batch = dataset
            .padded_batch(1, 3)
            .expect("wrapping batch should build");
        let input_values: Vec<i64> = Vec::<i64>::try_from(batch.input_ids.reshape([-1])).unwrap();
        let mask_values: Vec<f32> = Vec::<f32>::try_from(batch.target_mask.reshape([-1])).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(batch.input_ids.size(), vec![3, 3]);
        assert_eq!(input_values, vec![4, 5, 6, 1, 2, 3, 4, 5, 6]);
        assert_eq!(mask_values, vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(batch.masked_positions, 5);
        assert_eq!(batch.padding_tokens, 0);
    }

    #[test]
    fn qwen_sft_jsonl_reader_loads_instruction_input_response_records() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        fs::write(
            &jsonl,
            r#"{"instruction":"Reply with the project name.","response":"rustrain"}
{"instruction":"Name the language.","input":"rustrain implementation","response":"Rust"}
"#,
        )
        .expect("jsonl should write");

        let examples =
            qwen_sft_examples_from_jsonl_path(&jsonl).expect("examples should load from jsonl");

        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0].instruction, "Reply with the project name.");
        assert_eq!(examples[0].input, "");
        assert_eq!(examples[0].response, "rustrain");
        assert_eq!(examples[1].instruction, "Name the language.");
        assert_eq!(examples[1].input, "rustrain implementation");
        assert_eq!(examples[1].response, "Rust");
    }

    #[test]
    fn cached_greedy_matches_full_context_greedy_for_tiny_weights() {
        let config = QwenRuntimeConfig {
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };
        let weights = tiny_qwen_weights();
        let input_ids = Tensor::from_slice(&[0_i64, 1, 2]).reshape([1, 3]);

        let full = qwen_greedy_generate(&input_ids, &weights, &config, 3)
            .expect("full-context generate should run");
        let cached = qwen_greedy_generate_with_cache(&input_ids, &weights, &config, 3)
            .expect("cached generate should run");
        let full_ids: Vec<i64> = Vec::<i64>::try_from(full.reshape([-1])).unwrap();
        let cached_ids: Vec<i64> = Vec::<i64>::try_from(cached.reshape([-1])).unwrap();

        assert_eq!(cached_ids, full_ids);
    }

    fn tiny_qwen_weights() -> BTreeMap<String, Tensor> {
        let mut weights = BTreeMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::arange(24, (Kind::Float, Device::Cpu)).reshape([6, 4]) / 24.0,
        );
        weights.insert(
            "model.norm.weight".to_string(),
            Tensor::ones([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            Tensor::ones([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            Tensor::ones([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Tensor::eye(4, (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.bias".to_string(),
            Tensor::zeros([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            Tensor::ones([2, 4], (Kind::Float, Device::Cpu)) * 0.05,
        );
        weights.insert(
            "model.layers.0.self_attn.k_proj.bias".to_string(),
            Tensor::zeros([2], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            Tensor::ones([2, 4], (Kind::Float, Device::Cpu)) * 0.03,
        );
        weights.insert(
            "model.layers.0.self_attn.v_proj.bias".to_string(),
            Tensor::zeros([2], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            Tensor::ones([4, 4], (Kind::Float, Device::Cpu)) * 0.02,
        );
        weights.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            Tensor::ones([8, 4], (Kind::Float, Device::Cpu)) * 0.01,
        );
        weights.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            Tensor::ones([8, 4], (Kind::Float, Device::Cpu)) * 0.02,
        );
        weights.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            Tensor::ones([4, 8], (Kind::Float, Device::Cpu)) * 0.03,
        );
        weights
    }
}
