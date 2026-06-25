use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
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
