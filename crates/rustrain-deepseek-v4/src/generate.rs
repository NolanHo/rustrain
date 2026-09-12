use anyhow::Result;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Distribution;
use std::collections::BTreeMap;
use tch::{Kind, Tensor};
use tracing::info;

use crate::lora::*;
use crate::model::*;

pub fn v4_greedy_generate(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &V4RuntimeConfig,
    trainable_layers: &[usize],
    max_new_tokens: usize,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    for _ in 0..max_new_tokens {
        let logits = v4_forward_selective(&generated, weights, config, trainable_layers)?;
        let next_token = logits
            .select(0, 0)
            .select(1, logits.size()[1] - 1)
            .argmax(-1, false)
            .reshape([1, 1]);
        generated = Tensor::cat(&[&generated, &next_token], 1);
    }
    Ok(generated)
}

pub fn v4_greedy_generate_lora(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &V4RuntimeConfig,
    trainable_layers: &[usize],
    registry: &V4LoraRegistry,
    max_new_tokens: usize,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    for _ in 0..max_new_tokens {
        let logits = v4_forward_lora(&generated, weights, config, trainable_layers, registry)?;
        let next_token = logits
            .select(0, 0)
            .select(1, logits.size()[1] - 1)
            .argmax(-1, false)
            .reshape([1, 1]);
        generated = Tensor::cat(&[&generated, &next_token], 1);
    }
    Ok(generated)
}

#[allow(clippy::too_many_arguments)]
pub fn v4_sample_generate(
    input_ids: &Tensor,
    weights: &std::collections::BTreeMap<String, Tensor>,
    config: &V4RuntimeConfig,
    trainable_layers: &[usize],
    max_new_tokens: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
) -> Result<Tensor> {
    use rand::{rngs::StdRng, SeedableRng};
    let mut generated = input_ids.shallow_clone();
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..max_new_tokens {
        let logits = v4_forward_selective(&generated, weights, config, trainable_layers)?;
        let next_token = v4_sample_token_from_logits(
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

pub fn v4_sample_token_from_logits(
    logits: &Tensor,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    rng: &mut rand::rngs::StdRng,
) -> Result<Tensor> {
    use rand_distr::Distribution;
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
