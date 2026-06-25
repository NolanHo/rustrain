use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use tch::{Device, Kind, Tensor};
use tracing::info;

use crate::model::*;

/// A single SFT sample: token IDs + response-only loss mask.
#[derive(Clone, Debug)]
pub struct DeepSeekSftSample {
    pub tokens: Vec<i64>,
    pub target_mask: Vec<bool>,
}

/// A batch of SFT samples ready for training.
pub struct DeepSeekSftBatch {
    pub input_ids: Tensor,
    pub target_mask: Tensor,
    pub num_masked: usize,
}

/// Load JSONL instruction data, tokenize with HuggingFace tokenizer.
pub struct DeepSeekSftDataset {
    pub samples: Vec<DeepSeekSftSample>,
    pub pad_token_id: i64,
}

impl DeepSeekSftDataset {
    /// Load from JSONL files with fields {instruction, input, response}.
    pub fn from_jsonl(
        tokenizer: &tokenizers::Tokenizer,
        paths: &[std::path::PathBuf],
        instruction_field: &str,
        input_field: &str,
        response_field: &str,
        max_samples: Option<usize>,
        train_split: f32,
    ) -> Result<(Self, Self)> {
        let mut records = Vec::new();

        for path in paths {
            let file = std::fs::File::open(path)
                .with_context(|| format!("failed to open {}", path.display()))?;
            let reader = std::io::BufReader::new(file);
            for line in std::io::BufRead::lines(reader) {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let v: serde_json::Value = serde_json::from_str(&line)?;
                let instruction = v
                    .get(instruction_field)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let input = v
                    .get(input_field)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let response = v
                    .get(response_field)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                records.push((instruction, input, response));
            }
        }

        if let Some(max) = max_samples {
            records.truncate(max);
        }

        info!(records = records.len(), "loaded SFT records");

        // Tokenize each record
        let mut samples = Vec::with_capacity(records.len());
        for (instruction, input, response) in &records {
            // Build prompt
            let prompt = if input.is_empty() {
                format!("Instruction: {instruction}\nResponse: ")
            } else {
                format!("Instruction: {instruction}\nInput: {input}\nResponse: ")
            };

            let prompt_ids = tokenizer
                .encode(&prompt[..], true)
                .map_err(|e| anyhow::anyhow!("tokenizer encode prompt failed: {e}"))?
                .get_ids()
                .iter()
                .map(|&id| id as i64)
                .collect::<Vec<_>>();

            let response_ids = tokenizer
                .encode(&response[..], false)
                .map_err(|e| anyhow::anyhow!("tokenizer encode response failed: {e}"))?
                .get_ids()
                .iter()
                .map(|&id| id as i64)
                .collect::<Vec<_>>();

            // Concatenate: prompt_tokens + response_tokens
            let mut tokens = prompt_ids.clone();
            tokens.extend(&response_ids);

            // Target mask: 0 for prompt, 1 for response
            let mut target_mask = vec![false; prompt_ids.len()];
            target_mask.extend(vec![true; response_ids.len()]);

            samples.push(DeepSeekSftSample {
                tokens,
                target_mask,
            });
        }

        // Split into train / eval
        let split_point = ((records.len() as f32) * train_split).ceil() as usize;
        let split_point = split_point.max(1).min(samples.len().saturating_sub(1));
        let train_samples = samples[..split_point].to_vec();
        let eval_samples = samples[split_point..].to_vec();

        let pad_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<pad>"))
            .unwrap_or(0) as i64;

        info!(
            train_samples = train_samples.len(),
            eval_samples = eval_samples.len(),
            pad_token_id,
            "SFT dataset created"
        );

        Ok((
            Self {
                samples: train_samples,
                pad_token_id,
            },
            Self {
                samples: eval_samples,
                pad_token_id,
            },
        ))
    }

    /// Create a simple synthetic dataset for testing.
    pub fn synthetic(tokenizer: &tokenizers::Tokenizer) -> Result<Self> {
        let instruction = "Reply with the project name.";
        let response: String = "rustrain".to_string();

        let prompt = format!("Instruction: {instruction}\nResponse: ");
        let prompt_ids = tokenizer
            .encode(&prompt[..], true)
            .map_err(|e| anyhow::anyhow!("tokenizer failed: {e}"))?
            .get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect::<Vec<_>>();
        let response_ids = tokenizer
            .encode(&response[..], false)
            .map_err(|e| anyhow::anyhow!("tokenizer failed: {e}"))?
            .get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect::<Vec<_>>();

        let mut tokens = prompt_ids.clone();
        tokens.extend(&response_ids);
        let mut target_mask = vec![false; prompt_ids.len()];
        target_mask.extend(vec![true; response_ids.len()]);

        let pad_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<pad>"))
            .unwrap_or(0) as i64;

        Ok(Self {
            samples: vec![DeepSeekSftSample {
                tokens,
                target_mask,
            }],
            pad_token_id,
        })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Build a padded batch starting at `start` with `batch_size` samples.
    pub fn padded_batch(
        &self,
        start: usize,
        batch_size: usize,
        device: Device,
    ) -> DeepSeekSftBatch {
        let end = (start + batch_size).min(self.samples.len());
        let batch_samples = &self.samples[start..end];

        let max_len = batch_samples
            .iter()
            .map(|s| s.tokens.len())
            .max()
            .unwrap_or(1);
        let actual_batch = batch_samples.len();

        let mut input_ids = vec![self.pad_token_id; actual_batch * max_len];
        let mut target_mask = vec![0i64; actual_batch * max_len];
        let mut num_masked = 0;

        for (i, sample) in batch_samples.iter().enumerate() {
            for (j, &token) in sample.tokens.iter().enumerate() {
                input_ids[i * max_len + j] = token;
            }
            for (j, &mask) in sample.target_mask.iter().enumerate() {
                if mask {
                    target_mask[i * max_len + j] = 1;
                    num_masked += 1;
                }
            }
        }

        DeepSeekSftBatch {
            input_ids: Tensor::from_slice(&input_ids)
                .reshape([actual_batch as i64, max_len as i64])
                .to_device(device),
            target_mask: Tensor::from_slice(&target_mask)
                .reshape([actual_batch as i64, max_len as i64])
                .to_device(device),
            num_masked,
        }
    }
}

/// Compute response-only loss for a batch.
/// Loss = mean(cross_entropy(logits, targets) * target_mask) / sum(target_mask)
pub fn deepseek_lora_sft_loss(
    input_ids: &Tensor,
    target_mask: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &DeepSeekRuntimeConfig,
    trainable_layers: &[usize],
    lora_registry: &crate::lora::DeepSeekLoraRegistry,
) -> Result<Tensor> {
    // Forward
    let logits =
        deepseek_forward_lora(input_ids, weights, config, trainable_layers, lora_registry)?;

    // Shift: predict next token
    let shifted_logits = logits.narrow(1, 0, logits.size()[1] - 1);
    let shifted_targets = input_ids.narrow(1, 1, input_ids.size()[1] - 1);
    let shifted_mask = target_mask
        .narrow(1, 1, target_mask.size()[1] - 1)
        .to_kind(Kind::Float);

    // Cross entropy
    let vocab_size = config.vocab_size;
    let batch_size = shifted_logits.size()[0];
    let seq_len = shifted_logits.size()[1];

    let log_probs = shifted_logits
        .reshape([-1, vocab_size])
        .log_softmax(-1, Kind::Float);

    let targets = shifted_targets.reshape([-1]);
    let per_token_loss = log_probs
        .g_nll_loss::<&Tensor>(&targets, None, tch::Reduction::None, -100)
        .reshape([batch_size, seq_len]);

    // Apply mask: only count response tokens
    let masked_loss = &per_token_loss * &shifted_mask;
    let total_mask = shifted_mask.sum(Kind::Float);

    Ok(masked_loss.sum(Kind::Float) / total_mask.clamp_min(1.0))
}
