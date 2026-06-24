//! generate module - split from qwen_module.rs

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
use rustrain_nccl::nccl_smoke;

use crate::lora::*;
use crate::model::*;
use crate::rank_smoke::*;
use crate::session::*;
use crate::sft::*;

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

pub(crate) fn qwen_greedy_generate_with_kind(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    max_new_tokens: usize,
    compute_kind: Kind,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    for _ in 0..max_new_tokens {
        let logits = qwen_forward_from_ids_with_kind(&generated, weights, config, compute_kind)?;
        let next_token = logits.i((0, -1)).argmax(-1, false).reshape([1, 1]);
        generated = Tensor::cat(&[&generated, &next_token], 1);
    }
    Ok(generated)
}

pub(crate) fn qwen_greedy_generate_with_lora(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &QwenRuntimeConfig,
    registry: &QwenLoraRegistry,
    max_new_tokens: usize,
    compute_kind: Kind,
) -> Result<Tensor> {
    let mut generated = input_ids.shallow_clone();
    for _ in 0..max_new_tokens {
        let logits =
            qwen_forward_from_ids_with_lora(&generated, weights, config, registry, compute_kind)?;
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

pub(crate) fn sample_token_from_logits(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_module::test_utils::*;

    #[test]
    fn sampling_respects_top_k_and_top_p_filters() {
        let logits = Tensor::from_slice(&[0.0_f32, 1.0, 2.0, 3.0]);
        let mut rng = StdRng::seed_from_u64(7);

        let token =
            sample_token_from_logits(&logits, 0.8, 1, 0.5, &mut rng).expect("sample should run");

        assert_eq!(token.int64_value(&[0]), 3);
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
}
