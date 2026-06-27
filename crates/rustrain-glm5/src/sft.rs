use anyhow::{Context, Result};
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};
use tracing::info;

use crate::lora::*;
use crate::model::*;
use rustrain_checkpoint::safetensors::tensor;
use crate::model::{rms_norm, glm5_mlp};

#[derive(Clone, Debug)]
pub struct Glm5SftSample {
    pub tokens: Vec<i64>,
    pub target_mask: Vec<bool>,
}

pub struct Glm5SftBatch {
    pub input_ids: Tensor,
    pub target_mask: Tensor,
    pub num_masked: usize,
}

pub struct Glm5SftDataset {
    pub samples: Vec<Glm5SftSample>,
    pub pad_token_id: i64,
}

impl Glm5SftDataset {
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
            sft_samples.push(Glm5SftSample { tokens, target_mask });
        }
        let pad_token_id = tokenizer.token_to_id("<pad>").unwrap_or(0) as i64;
        Ok(Self {
            samples: sft_samples,
            pad_token_id,
        })
    }

    pub fn padded_batch(&self, start: usize, batch_size: usize, device: Device) -> Glm5SftBatch {
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

        Glm5SftBatch {
            input_ids: Tensor::from_slice(&input_ids)
                .reshape([actual_batch as i64, max_len as i64])
                .to_device(device),
            target_mask: Tensor::from_slice(&target_mask)
                .reshape([actual_batch as i64, max_len as i64])
                .to_device(device),
            num_masked,
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }
}

/// GLM-5.2 LoRA SFT loss
pub fn glm5_lora_sft_loss(
    input_ids: &Tensor,
    target_mask: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &Glm5RuntimeConfig,
    trainable_layers: &[usize],
    registry: &Glm5LoraRegistry,
) -> Result<Tensor> {
    let logits = glm5_forward_lora(input_ids, weights, config, trainable_layers, registry)?;
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

/// GLM-5.2 forward with LoRA adapters applied
pub fn glm5_forward_lora(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &Glm5RuntimeConfig,
    trainable_layers: &[usize],
    registry: &Glm5LoraRegistry,
) -> Result<Tensor> {
    let embed_tokens = tensor(weights, "model.embed_tokens.weight")?;
    let final_norm = tensor(weights, "model.norm.weight")?;
    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);

    let mut index_share_state: Option<IndexShareState> = None;

    // Pre-load indexer weights for all "full" layers (for IndexShare)
    let mut indexer_weights_map: BTreeMap<usize, Glm5AttentionWeights> = BTreeMap::new();
    for layer in 0..config.num_hidden_layers {
        if layer < config.indexer_types.len() && config.indexer_types[layer] == "full" {
            let attn = Glm5AttentionWeights::load_raw(weights, layer)?;
            indexer_weights_map.insert(layer, attn);
        }
    }

    for layer in 0..config.num_hidden_layers {
        if !trainable_layers.contains(&layer) {
            continue;
        }
        if config.is_moe_layer(layer) {
            let lw = Glm5MoeLayerWeights::load_raw(weights, layer, config.n_routed_experts)?;
            let hidden_norm = rms_norm(&hidden, &lw.input_norm, config.rms_norm_eps);
            let attn = lora_attention_weights(&lw.attn, layer, registry);
            let source = config.indexer_source_layer(layer);
            let indexer_weights = indexer_weights_map.get(&source).unwrap_or(&attn);
            let attn_out = glm5_dsa_attention(
                &hidden_norm,
                &attn,
                indexer_weights,
                config,
                &mut index_share_state,
                layer,
            );
            let residual = &hidden + &attn_out;
            let mlp_input = rms_norm(&residual, &lw.post_attention_norm, config.rms_norm_eps);
            let mlp = glm5_moe_mlp(
                &mlp_input,
                &lw.gate,
                &lw.shared_gate_proj,
                &lw.shared_up_proj,
                &lw.shared_down_proj,
                &lw.experts,
                config.num_experts_per_tok,
                &config.scoring_func,
                config.n_group,
                config.topk_group,
                config.routed_scaling_factor,
            );
            hidden = residual + mlp;
        } else {
            let lw = Glm5DenseLayerWeights::load_raw(weights, layer)?;
            let hidden_norm = rms_norm(&hidden, &lw.input_norm, config.rms_norm_eps);
            let attn = lora_attention_weights(&lw.attn, layer, registry);
            let source = config.indexer_source_layer(layer);
            let indexer_weights = indexer_weights_map.get(&source).unwrap_or(&attn);
            let attn_out = glm5_dsa_attention(
                &hidden_norm,
                &attn,
                indexer_weights,
                config,
                &mut index_share_state,
                layer,
            );
            let residual = &hidden + &attn_out;
            let mlp_input = rms_norm(&residual, &lw.post_attention_norm, config.rms_norm_eps);
            let mlp = glm5_mlp(&mlp_input, &lw.gate_proj, &lw.up_proj, &lw.down_proj);
            hidden = residual + mlp;
        }
    }

    let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps);
    let lm_head = if config.tie_word_embeddings {
        embed_tokens.shallow_clone()
    } else {
        tensor(weights, "lm_head.weight")?.shallow_clone()
    };
    let logits = hidden.linear::<&Tensor>(&lm_head, None);
    Ok(logits)
}
