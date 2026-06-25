//! DeepSeek generate: greedy decode + sampling (temperature/top-k/top-p)

use std::collections::BTreeMap;

use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::Distribution;
use tch::{Device, Kind, Tensor};
use tracing::info;

use crate::lora::DeepSeekLoraRegistry;
use crate::model::*;

/// Greedy decode: pick argmax token at each step.
pub fn deepseek_greedy_generate(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &DeepSeekRuntimeConfig,
    trainable_layers: &[usize],
    max_new_tokens: usize,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    for step in 0..max_new_tokens {
        let logits = deepseek_forward_selective(&generated, weights, config, trainable_layers)?;
        let next_token = logits
            .select(0, 0)
            .select(1, logits.size()[1] - 1)
            .argmax(-1, false)
            .reshape([1, 1]);
        generated = Tensor::cat(&[&generated, &next_token], 1);
        if step == 0 {
            info!("generate: first token generated");
        }
    }
    Ok(generated)
}

/// Greedy decode with LoRA adapter applied.
pub fn deepseek_greedy_generate_lora(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &DeepSeekRuntimeConfig,
    trainable_layers: &[usize],
    registry: &DeepSeekLoraRegistry,
    max_new_tokens: usize,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    for _ in 0..max_new_tokens {
        let logits =
            deepseek_forward_lora(&generated, weights, config, trainable_layers, registry)?;
        let next_token = logits
            .select(0, 0)
            .select(1, logits.size()[1] - 1)
            .argmax(-1, false)
            .reshape([1, 1]);
        generated = Tensor::cat(&[&generated, &next_token], 1);
    }
    Ok(generated)
}

/// Sampling decode with temperature, top-k, top-p filtering.
#[allow(clippy::too_many_arguments)]
pub fn deepseek_sample_generate(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &DeepSeekRuntimeConfig,
    trainable_layers: &[usize],
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..max_new_tokens {
        let logits = deepseek_forward_selective(&generated, weights, config, trainable_layers)?;
        let next_token = sample_token_from_logits(
            &logits.select(0, 0).select(1, logits.size()[1] - 1),
            temperature,
            top_k,
            top_p,
            &mut rng,
        )?
        .reshape([1, 1]);
        generated = Tensor::cat(&[&generated, &next_token], 1);
    }
    Ok(generated)
}

/// Sampling decode with LoRA adapter.
#[allow(clippy::too_many_arguments)]
pub fn deepseek_sample_generate_lora(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &DeepSeekRuntimeConfig,
    trainable_layers: &[usize],
    registry: &DeepSeekLoraRegistry,
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..max_new_tokens {
        let logits =
            deepseek_forward_lora(&generated, weights, config, trainable_layers, registry)?;
        let next_token = sample_token_from_logits(
            &logits.select(0, 0).select(1, logits.size()[1] - 1),
            temperature,
            top_k,
            top_p,
            &mut rng,
        )?
        .reshape([1, 1]);
        generated = Tensor::cat(&[&generated, &next_token], 1);
    }
    Ok(generated)
}

/// Sample a single token from logits with temperature, top-k, top-p filtering.
pub fn sample_token_from_logits(
    logits: &Tensor,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    rng: &mut StdRng,
) -> Result<Tensor> {
    if temperature <= 0.0 {
        return Ok(logits.argmax(-1, false));
    }

    let logits = logits / temperature;
    let vocab_size = logits.size()[logits.size().len() - 1];
    let k = top_k.min(vocab_size as usize).max(1);
    let (topk_values, topk_indices) = logits.topk(k as i64, -1, true, true);
    let probs = topk_values.softmax(-1, Kind::Float);

    let final_probs = if top_p < 1.0 {
        let cumsum = probs.cumsum(-1, Kind::Float);
        let mask = cumsum.le(top_p).to_kind(Kind::Float);
        let masked = &probs * &mask;
        let total = masked.sum(Kind::Float).clamp_min(1e-9);
        masked / total
    } else {
        probs
    };

    let probs_vec: Vec<f64> = final_probs.try_into()?;
    let dist = rand_distr::WeightedIndex::new(&probs_vec)?;
    let idx = dist.sample(rng);
    Ok(topk_indices.select(0, idx as i64))
}
