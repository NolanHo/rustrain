use anyhow::{Context, Result};
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};
use tracing::info;

use crate::lora::*;
use crate::model::*;

#[derive(Clone, Debug)]
pub struct V4SftSample {
    pub tokens: Vec<i64>,
    pub target_mask: Vec<bool>,
}

pub struct V4SftBatch {
    pub input_ids: Tensor,
    pub target_mask: Tensor,
    pub num_masked: usize,
}

pub struct V4SftDataset {
    pub samples: Vec<V4SftSample>,
    pub pad_token_id: i64,
}

impl V4SftDataset {
    pub fn synthetic(tokenizer: &tokenizers::Tokenizer) -> Result<Self> {
        let prompt: String = "Instruction: Reply with the project name.\nResponse: ".to_string();
        let response: String = "rustrain".to_string();
        Self::build_from_samples(vec![(prompt, response)], tokenizer)
    }

    /// Load SFT data from a JSONL file.
    /// Each line: {"instruction": "...", "input": "...", "response": "..."}
    pub fn from_jsonl_simple(
        path: &std::path::Path,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let mut samples = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let obj: serde_json::Value = serde_json::from_str(line)
                .with_context(|| format!("failed to parse JSONL line: {line}"))?;
            let instruction = obj["instruction"].as_str().unwrap_or("");
            let input = obj["input"].as_str().unwrap_or("");
            let response = obj["response"].as_str().unwrap_or("");
            let prompt = if input.is_empty() {
                format!("Instruction: {instruction}\nResponse: ")
            } else {
                format!("Instruction: {instruction}\nInput: {input}\nResponse: ")
            };
            samples.push((prompt, response.to_string()));
        }
        info!(samples = samples.len(), path = %path.display(), "loaded SFT JSONL");
        Self::build_from_samples(samples, tokenizer)
    }

    fn build_from_samples(
        samples: Vec<(String, String)>,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Result<Self> {
        let mut sft_samples = Vec::new();
        for (prompt, response) in &samples {
            let prompt_ids = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| anyhow::anyhow!("tokenizer failed: {e}"))?
                .get_ids()
                .iter()
                .map(|&id| id as i64)
                .collect::<Vec<_>>();
            let response_ids = tokenizer
                .encode(response.as_str(), false)
                .map_err(|e| anyhow::anyhow!("tokenizer failed: {e}"))?
                .get_ids()
                .iter()
                .map(|&id| id as i64)
                .collect::<Vec<_>>();
            let mut tokens = prompt_ids.clone();
            tokens.extend(&response_ids);
            let mut target_mask = vec![false; prompt_ids.len()];
            target_mask.extend(vec![true; response_ids.len()]);
            sft_samples.push(V4SftSample {
                tokens,
                target_mask,
            });
        }
        let pad_token_id = tokenizer.token_to_id("<pad>").unwrap_or(0) as i64;
        Ok(Self {
            samples: sft_samples,
            pad_token_id,
        })
    }

    pub fn padded_batch(&self, start: usize, batch_size: usize, device: Device) -> V4SftBatch {
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

        V4SftBatch {
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

pub fn v4_lora_sft_loss(
    input_ids: &Tensor,
    target_mask: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &V4RuntimeConfig,
    trainable_layers: &[usize],
    registry: &V4LoraRegistry,
) -> Result<Tensor> {
    let logits = v4_forward_lora(input_ids, weights, config, trainable_layers, registry)?;
    let shifted_logits = logits.narrow(1, 0, logits.size()[1] - 1);
    let shifted_targets = input_ids.narrow(1, 1, input_ids.size()[1] - 1);
    let shifted_mask = target_mask
        .narrow(1, 1, target_mask.size()[1] - 1)
        .to_kind(Kind::Float);
    let batch_size = shifted_logits.size()[0];
    let seq_len = shifted_logits.size()[1];

    let log_probs = shifted_logits
        .reshape([-1, config.vocab_size])
        .log_softmax(-1, Kind::Float);
    let per_token_loss = log_probs
        .g_nll_loss::<&Tensor>(
            &shifted_targets.reshape([-1]),
            None,
            tch::Reduction::None,
            -100,
        )
        .reshape([batch_size, seq_len]);

    let masked_loss = &per_token_loss * &shifted_mask;
    let total_mask = shifted_mask.sum(Kind::Float);
    Ok(masked_loss.sum(Kind::Float) / total_mask.clamp_min(1.0))
}

impl V4SftDataset {
    pub fn from_jsonl(
        tokenizer: &tokenizers::Tokenizer,
        paths: &[std::path::PathBuf],
        instruction_field: &str,
        input_field: &str,
        response_field: &str,
        max_samples: Option<usize>,
        train_split: f32,
    ) -> Result<(Self, Self)> {
        use std::io::BufRead;
        let mut records = Vec::new();
        for path in paths {
            let file = std::fs::File::open(path)
                .with_context(|| format!("failed to open {}", path.display()))?;
            for line in std::io::BufReader::new(file).lines() {
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
        info!(records = records.len(), "loaded V4 SFT records");

        let mut samples = Vec::with_capacity(records.len());
        for (instruction, input, response) in &records {
            let prompt = if input.is_empty() {
                format!("Instruction: {instruction}\nResponse: ")
            } else {
                format!("Instruction: {instruction}\nInput: {input}\nResponse: ")
            };
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
            samples.push(V4SftSample {
                tokens,
                target_mask,
            });
        }

        let split = ((records.len() as f32) * train_split).ceil() as usize;
        let split = split.max(1).min(samples.len().saturating_sub(1));
        let pad_token_id = tokenizer.token_to_id("<pad>").unwrap_or(0) as i64;
        Ok((
            Self {
                samples: samples[..split].to_vec(),
                pad_token_id,
            },
            Self {
                samples: samples[split..].to_vec(),
                pad_token_id,
            },
        ))
    }
}
