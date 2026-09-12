use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use safetensors::SafeTensors;
use serde::Deserialize;
use tokenizers::Tokenizer;

use rustrain_qwen::qwen_module::resolve_qwen_model_path;

#[derive(Debug, Deserialize)]
struct HfConfig {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    intermediate_size: Option<usize>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    torch_dtype: Option<String>,
}

pub fn inspect_model(model_path: &Path, prompt: &str, tensor_limit: usize) -> Result<()> {
    let model_path = resolve_qwen_model_path(model_path)?;
    reject_known_non_hf_safetensors(&model_path)?;

    let config_path = model_path.join("config.json");
    let tokenizer_path = model_path.join("tokenizer.json");

    let config = read_config(&config_path)?;
    println!("model_path: {}", model_path.display());
    println!("model_type: {}", display_opt(config.model_type.as_deref()));
    println!(
        "architectures: {}",
        config
            .architectures
            .as_ref()
            .map(|items| items.join(", "))
            .unwrap_or_else(|| "unknown".to_string())
    );
    println!("vocab_size: {}", display_opt(config.vocab_size));
    println!("hidden_size: {}", display_opt(config.hidden_size));
    println!(
        "num_hidden_layers: {}",
        display_opt(config.num_hidden_layers)
    );
    println!(
        "num_attention_heads: {}",
        display_opt(config.num_attention_heads)
    );
    println!(
        "num_key_value_heads: {}",
        display_opt(config.num_key_value_heads)
    );
    println!(
        "intermediate_size: {}",
        display_opt(config.intermediate_size)
    );
    println!("rms_norm_eps: {}", display_opt(config.rms_norm_eps));
    println!("rope_theta: {}", display_opt(config.rope_theta));
    println!(
        "torch_dtype: {}",
        display_opt(config.torch_dtype.as_deref())
    );

    inspect_tokenizer(&tokenizer_path, prompt)?;
    inspect_safetensors(&model_path, tensor_limit)?;

    Ok(())
}

fn read_config(path: &Path) -> Result<HfConfig> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))
}

fn inspect_tokenizer(path: &Path, prompt: &str) -> Result<()> {
    let tokenizer = Tokenizer::from_file(path)
        .map_err(|error| anyhow!("failed to load tokenizer {}: {error}", path.display()))?;
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|error| anyhow!("failed to encode prompt: {error}"))?;
    let decoded = tokenizer
        .decode(encoding.get_ids(), false)
        .map_err(|error| anyhow!("failed to decode prompt ids: {error}"))?;

    println!("tokenizer_path: {}", path.display());
    println!("prompt: {prompt}");
    println!("prompt_token_ids: {:?}", encoding.get_ids());
    println!("decoded_prompt: {decoded}");

    Ok(())
}

fn inspect_safetensors(model_path: &Path, tensor_limit: usize) -> Result<()> {
    let files = safetensor_files(model_path)?;
    if files.is_empty() {
        if model_path.join("release").exists()
            || model_path
                .join("latest_checkpointed_iteration.txt")
                .exists()
        {
            return Err(anyhow!(
                "{} looks like a torch distributed checkpoint, not an HF safetensors directory",
                model_path.display()
            ));
        }
        return Err(anyhow!(
            "no .safetensors files found under {}",
            model_path.display()
        ));
    }

    println!("safetensors_files: {}", files.len());
    let mut printed = 0usize;
    for file in files {
        let bytes =
            fs::read(&file).with_context(|| format!("failed to read {}", file.display()))?;
        let tensors = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("failed to parse {}", file.display()))?;
        println!("safetensors_file: {}", file.display());
        let mut tensor_views = tensors.tensors();
        tensor_views.sort_by(|(left, _), (right, _)| left.cmp(right));
        for (name, view) in tensor_views {
            if printed >= tensor_limit {
                println!("tensor_listing_truncated_at: {tensor_limit}");
                return Ok(());
            }
            println!(
                "tensor: {name} dtype={:?} shape={:?}",
                view.dtype(),
                view.shape()
            );
            printed += 1;
        }
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

fn reject_known_non_hf_safetensors(model_path: &Path) -> Result<()> {
    if model_path.join("release").exists()
        || model_path
            .join("latest_checkpointed_iteration.txt")
            .exists()
    {
        return Err(anyhow!(
            "{} looks like a torch distributed checkpoint, not an HF safetensors directory",
            model_path.display()
        ));
    }
    Ok(())
}

fn display_opt<T: std::fmt::Display>(value: Option<T>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
