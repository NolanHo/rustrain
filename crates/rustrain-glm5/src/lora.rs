use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use tch::{nn, Device, Kind, Tensor};

use crate::model::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize)]
pub enum Glm5LoraTargetModule {
    WqA,
    WqB,
    Wkv,
    Wo,
}

impl Glm5LoraTargetModule {
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "q_a_proj" => Ok(Self::WqA),
            "q_b_proj" => Ok(Self::WqB),
            "kv_a_proj_with_mqa" => Ok(Self::Wkv),
            "o_proj" => Ok(Self::Wo),
            other => bail!("unknown GLM-5 LoRA target module: {other}"),
        }
    }

    pub fn weight_suffix(&self) -> &'static str {
        match self {
            Self::WqA => "attn.q_a_proj",
            Self::WqB => "attn.q_b_proj",
            Self::Wkv => "attn.kv_a_proj_with_mqa",
            Self::Wo => "attn.o_proj",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Glm5LoraConfig {
    pub rank: i64,
    pub alpha: i64,
    pub target_layers: Vec<usize>,
    pub target_modules: Vec<Glm5LoraTargetModule>,
}

pub struct Glm5LoraRegistry {
    pub adapters: BTreeMap<(usize, Glm5LoraTargetModule), (Tensor, Tensor)>,
    pub config: Glm5LoraConfig,
    pub var_store: nn::VarStore,
}

impl Glm5LoraRegistry {
    pub fn new(
        weights: &BTreeMap<String, Tensor>,
        config: Glm5LoraConfig,
        device: Device,
    ) -> Result<Self> {
        let vs = nn::VarStore::new(device);
        let p = vs.root();
        let mut adapters = BTreeMap::new();

        for &layer in &config.target_layers {
            for &module in &config.target_modules {
                let suffix = match module {
                    Glm5LoraTargetModule::WqA => "q_a_proj",
                    Glm5LoraTargetModule::WqB => "q_b_proj",
                    Glm5LoraTargetModule::Wkv => "kv_a_proj_with_mqa",
                    Glm5LoraTargetModule::Wo => "o_proj",
                };
                let name = format!("model.layers.{layer}.self_attn.{suffix}.weight");
                let weight = weights
                    .get(&name)
                    .ok_or_else(|| anyhow::anyhow!("GLM-5 LoRA base weight not found: {name}"))?;
                let out_features = weight.size()[0];
                let in_features = weight.size()[1];

                let path = format!("layer{layer}/{:?}", module);
                let lora_a = p.randn(
                    &format!("{path}/lora_a"),
                    &[config.rank, in_features],
                    0.0,
                    1.0 / (config.rank as f64).sqrt(),
                );
                let lora_b = p.zeros(&format!("{path}/lora_b"), &[out_features, config.rank]);
                adapters.insert((layer, module), (lora_a, lora_b));
            }
        }

        Ok(Self {
            adapters,
            config,
            var_store: vs,
        })
    }

    pub fn param_count(&self) -> usize {
        self.adapters.len() * 2
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut tensors = Vec::new();
        for ((layer, module), (lora_a, lora_b)) in &self.adapters {
            let prefix = format!("layers.{layer}.{}", module.weight_suffix());
            tensors.push((format!("{prefix}.lora_a"), lora_a.shallow_clone()));
            tensors.push((format!("{prefix}.lora_b"), lora_b.shallow_clone()));
        }
        Tensor::write_safetensors(&tensors, path)?;
        Ok(())
    }

    pub fn load(path: &Path, config: Glm5LoraConfig) -> Result<Self> {
        let tensors = Tensor::read_safetensors(path)?;
        let map: BTreeMap<String, Tensor> = tensors.into_iter().collect();
        let vs = nn::VarStore::new(Device::Cpu);
        let p = vs.root();
        let mut adapters = BTreeMap::new();

        for &layer in &config.target_layers {
            for &module in &config.target_modules {
                let prefix = format!("layers.{layer}.{}", module.weight_suffix());
                let path_str = format!("layer{layer}/{:?}", module);
                let lora_a_stored = map
                    .get(&format!("{prefix}.lora_a"))
                    .ok_or_else(|| anyhow::anyhow!("missing lora_a for {prefix}"))?;
                let lora_b_stored = map
                    .get(&format!("{prefix}.lora_b"))
                    .ok_or_else(|| anyhow::anyhow!("missing lora_b for {prefix}"))?;

                let mut lora_a = p.zeros(&format!("{path_str}/lora_a"), &lora_a_stored.size());
                let mut lora_b = p.zeros(&format!("{path_str}/lora_b"), &lora_b_stored.size());
                let _ = tch::no_grad(|| {
                    lora_a.copy_(lora_a_stored);
                    lora_b.copy_(lora_b_stored);
                    Ok::<(), tch::TchError>(())
                });
                adapters.insert((layer, module), (lora_a, lora_b));
            }
        }

        Ok(Self {
            adapters,
            config,
            var_store: vs,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct Glm5LoraManifest {
    pub format: String,
    pub base_model_path: String,
    pub adapter_safetensors: String,
    pub rank: i64,
    pub alpha: i64,
    pub target_layers: Vec<usize>,
    pub target_modules: Vec<String>,
    pub steps: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
}

pub fn lora_manifest_path(adapter_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.json", adapter_path.display()))
}

pub fn write_lora_manifest(path: &Path, manifest: &Glm5LoraManifest) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(manifest)? + "\n")?;
    Ok(())
}

/// Compute W' = W + (B @ A) * scale for GLM-5 LoRA
pub fn glm5_lora_weight(
    base: &Tensor,
    layer: usize,
    module: Glm5LoraTargetModule,
    registry: &Glm5LoraRegistry,
) -> Tensor {
    if let Some((lora_a, lora_b)) = registry.adapters.get(&(layer, module)) {
        let scale = registry.config.alpha as f64 / lora_a.size()[0] as f64;
        let delta = lora_b.matmul(lora_a) * scale;
        base.shallow_clone() + delta.to_device(base.device()).to_kind(base.kind())
    } else {
        base.shallow_clone()
    }
}

/// Apply LoRA to attention weights for a specific layer
pub fn lora_attention_weights(
    base: &Glm5AttentionWeights,
    layer: usize,
    registry: &Glm5LoraRegistry,
) -> Glm5AttentionWeights {
    Glm5AttentionWeights {
        q_a_proj: glm5_lora_weight(&base.q_a_proj, layer, Glm5LoraTargetModule::WqA, registry),
        q_a_layernorm: base.q_a_layernorm.shallow_clone(),
        q_b_proj: glm5_lora_weight(&base.q_b_proj, layer, Glm5LoraTargetModule::WqB, registry),
        kv_a_proj_with_mqa: glm5_lora_weight(&base.kv_a_proj_with_mqa, layer, Glm5LoraTargetModule::Wkv, registry),
        kv_a_layernorm: base.kv_a_layernorm.shallow_clone(),
        kv_b_proj: base.kv_b_proj.shallow_clone(),
        o_proj: glm5_lora_weight(&base.o_proj, layer, Glm5LoraTargetModule::Wo, registry),
        indexer_k_norm_weight: base.indexer_k_norm_weight.as_ref().map(|t| t.shallow_clone()),
        indexer_k_norm_bias: base.indexer_k_norm_bias.as_ref().map(|t| t.shallow_clone()),
        indexer_weights_proj: base.indexer_weights_proj.as_ref().map(|t| t.shallow_clone()),
        indexer_wk: base.indexer_wk.as_ref().map(|t| t.shallow_clone()),
        indexer_wq_b: base.indexer_wq_b.as_ref().map(|t| t.shallow_clone()),
    }
}
