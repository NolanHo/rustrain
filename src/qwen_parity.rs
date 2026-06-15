use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
struct QwenConfig {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    tie_word_embeddings: bool,
}

#[derive(Debug, Deserialize)]
struct PythonReferenceSummary {
    input_ids: Vec<u32>,
    logits_shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct TensorSpec {
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct QwenParitySmokeSummary {
    model_path: String,
    prompt_file: String,
    prompt_token_ids: Vec<u32>,
    reference_token_ids_match: bool,
    reference_logits_shape: Vec<usize>,
    safetensors_files: usize,
    checked_tensors: Vec<String>,
    tie_word_embeddings: bool,
    lm_head_weight_present: bool,
    head_dim: usize,
    kv_hidden_size: usize,
}

pub fn qwen_parity_smoke(
    model_path: &Path,
    prompt_file: &Path,
    reference_summary: &Path,
) -> Result<()> {
    let config = read_config(&model_path.join("config.json"))?;
    validate_config(&config)?;

    let prompt = fs::read_to_string(prompt_file)
        .with_context(|| format!("failed to read {}", prompt_file.display()))?;
    let prompt = prompt.trim();
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|error| anyhow!("failed to encode prompt: {error}"))?;
    let prompt_token_ids = encoding.get_ids().to_vec();

    let reference = read_reference_summary(reference_summary)?;
    let tensor_specs = read_tensor_specs(model_path)?;
    let checked_tensors = validate_qwen_weight_map(&config, &tensor_specs)?;
    let lm_head_weight_present = tensor_specs.contains_key("lm_head.weight");

    if reference.logits_shape.len() != 3 {
        bail!(
            "reference logits shape should be rank 3, got {:?}",
            reference.logits_shape
        );
    }
    if reference.logits_shape[0] != 1
        || reference.logits_shape[1] != prompt_token_ids.len()
        || reference.logits_shape[2] != config.vocab_size
    {
        bail!(
            "reference logits shape {:?} does not match batch=1 seq_len={} vocab={}",
            reference.logits_shape,
            prompt_token_ids.len(),
            config.vocab_size
        );
    }

    let summary = QwenParitySmokeSummary {
        model_path: model_path.display().to_string(),
        prompt_file: prompt_file.display().to_string(),
        reference_token_ids_match: reference.input_ids == prompt_token_ids,
        prompt_token_ids,
        reference_logits_shape: reference.logits_shape,
        safetensors_files: safetensor_files(model_path)?.len(),
        checked_tensors,
        tie_word_embeddings: config.tie_word_embeddings,
        lm_head_weight_present,
        head_dim: config.hidden_size / config.num_attention_heads,
        kv_hidden_size: config.num_key_value_heads
            * (config.hidden_size / config.num_attention_heads),
    };

    if !summary.reference_token_ids_match {
        bail!("Rust tokenizer ids differ from Python reference ids");
    }
    if summary.tie_word_embeddings && summary.lm_head_weight_present {
        bail!("Qwen tied-head checkpoint unexpectedly contains lm_head.weight");
    }

    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

fn read_config(path: &Path) -> Result<QwenConfig> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))
}

fn read_reference_summary(path: &Path) -> Result<PythonReferenceSummary> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))
}

fn validate_config(config: &QwenConfig) -> Result<()> {
    if config.model_type != "qwen2" {
        bail!("expected model_type=qwen2, got {}", config.model_type);
    }
    if config.hidden_size % config.num_attention_heads != 0 {
        bail!("hidden_size must be divisible by num_attention_heads");
    }
    if config.num_attention_heads % config.num_key_value_heads != 0 {
        bail!("num_attention_heads must be divisible by num_key_value_heads");
    }
    if config.rms_norm_eps <= 0.0 {
        bail!("rms_norm_eps must be positive");
    }
    if config.rope_theta <= 0.0 {
        bail!("rope_theta must be positive");
    }
    Ok(())
}

fn read_tensor_specs(model_path: &Path) -> Result<BTreeMap<String, TensorSpec>> {
    let mut specs = BTreeMap::new();
    for file in safetensor_files(model_path)? {
        let bytes =
            fs::read(&file).with_context(|| format!("failed to read {}", file.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("failed to parse {}", file.display()))?;
        for (name, view) in tensors.tensors() {
            specs.insert(
                name,
                TensorSpec {
                    dtype: format!("{:?}", view.dtype()),
                    shape: view.shape().to_vec(),
                },
            );
        }
    }
    if specs.is_empty() {
        bail!("no tensors found under {}", model_path.display());
    }
    Ok(specs)
}

fn validate_qwen_weight_map(
    config: &QwenConfig,
    specs: &BTreeMap<String, TensorSpec>,
) -> Result<Vec<String>> {
    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_hidden = config.num_key_value_heads * head_dim;
    let last_layer = config.num_hidden_layers - 1;
    let mut checked = Vec::new();

    let global = [
        (
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ),
        ("model.norm.weight".to_string(), vec![config.hidden_size]),
    ];
    for (name, shape) in global {
        require_shape(specs, &name, &shape)?;
        checked.push(name);
    }

    for layer in [0, last_layer] {
        let prefix = format!("model.layers.{layer}");
        let layer_specs = [
            (
                format!("{prefix}.input_layernorm.weight"),
                vec![config.hidden_size],
            ),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![config.hidden_size],
            ),
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![config.hidden_size, config.hidden_size],
            ),
            (
                format!("{prefix}.self_attn.q_proj.bias"),
                vec![config.hidden_size],
            ),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv_hidden, config.hidden_size],
            ),
            (format!("{prefix}.self_attn.k_proj.bias"), vec![kv_hidden]),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv_hidden, config.hidden_size],
            ),
            (format!("{prefix}.self_attn.v_proj.bias"), vec![kv_hidden]),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![config.hidden_size, config.hidden_size],
            ),
            (
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                format!("{prefix}.mlp.up_proj.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                format!("{prefix}.mlp.down_proj.weight"),
                vec![config.hidden_size, config.intermediate_size],
            ),
        ];
        for (name, shape) in layer_specs {
            require_shape(specs, &name, &shape)?;
            checked.push(name);
        }
    }

    Ok(checked)
}

fn require_shape(
    specs: &BTreeMap<String, TensorSpec>,
    name: &str,
    expected_shape: &[usize],
) -> Result<()> {
    let spec = specs
        .get(name)
        .ok_or_else(|| anyhow!("missing tensor {name}"))?;
    if spec.shape != expected_shape {
        bail!(
            "tensor {name} shape mismatch: expected {:?}, got {:?}",
            expected_shape,
            spec.shape
        );
    }
    Ok(())
}

fn safetensor_files(model_path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(model_path)
        .with_context(|| format!("failed to list {}", model_path.display()))?
    {
        let entry =
            entry.with_context(|| format!("failed to read entry in {}", model_path.display()))?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> QwenConfig {
        QwenConfig {
            model_type: "qwen2".to_string(),
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 12,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn qwen_weight_map_validates_first_and_last_layer_shapes() {
        let config = tiny_config();
        let mut specs = BTreeMap::new();
        let head_dim = config.hidden_size / config.num_attention_heads;
        let kv_hidden = config.num_key_value_heads * head_dim;
        let mut insert = |name: String, shape: Vec<usize>| {
            specs.insert(
                name,
                TensorSpec {
                    dtype: "BF16".to_string(),
                    shape,
                },
            );
        };
        insert(
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        );
        insert("model.norm.weight".to_string(), vec![config.hidden_size]);
        for layer in [0, 1] {
            let prefix = format!("model.layers.{layer}");
            insert(format!("{prefix}.input_layernorm.weight"), vec![8]);
            insert(format!("{prefix}.post_attention_layernorm.weight"), vec![8]);
            insert(format!("{prefix}.self_attn.q_proj.weight"), vec![8, 8]);
            insert(format!("{prefix}.self_attn.q_proj.bias"), vec![8]);
            insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv_hidden, 8],
            );
            insert(format!("{prefix}.self_attn.k_proj.bias"), vec![kv_hidden]);
            insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv_hidden, 8],
            );
            insert(format!("{prefix}.self_attn.v_proj.bias"), vec![kv_hidden]);
            insert(format!("{prefix}.self_attn.o_proj.weight"), vec![8, 8]);
            insert(format!("{prefix}.mlp.gate_proj.weight"), vec![12, 8]);
            insert(format!("{prefix}.mlp.up_proj.weight"), vec![12, 8]);
            insert(format!("{prefix}.mlp.down_proj.weight"), vec![8, 12]);
        }

        let checked =
            validate_qwen_weight_map(&config, &specs).expect("weight map should validate");

        assert_eq!(checked.len(), 26);
    }
}
