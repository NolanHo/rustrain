use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use tch::{nn, Device, Kind, Tensor};

use crate::model::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize)]
pub enum V4LoraTargetModule {
    WqA,
    WqB,
    Wkv,
    WoA,
    WoB,
}

impl V4LoraTargetModule {
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "wq_a" => Ok(Self::WqA),
            "wq_b" => Ok(Self::WqB),
            "wkv" => Ok(Self::Wkv),
            "wo_a" => Ok(Self::WoA),
            "wo_b" => Ok(Self::WoB),
            other => bail!("unknown V4 LoRA target module: {other}"),
        }
    }

    pub fn weight_suffix(&self) -> &'static str {
        match self {
            Self::WqA => "attn.wq_a",
            Self::WqB => "attn.wq_b",
            Self::Wkv => "attn.wkv",
            Self::WoA => "attn.wo_a",
            Self::WoB => "attn.wo_b",
        }
    }
}

#[derive(Clone, Debug)]
pub struct V4LoraConfig {
    pub rank: i64,
    pub alpha: i64,
    pub target_layers: Vec<usize>,
    pub target_modules: Vec<V4LoraTargetModule>,
}

pub struct V4LoraRegistry {
    pub adapters: BTreeMap<(usize, V4LoraTargetModule), (Tensor, Tensor)>,
    pub config: V4LoraConfig,
    pub var_store: nn::VarStore,
}

impl V4LoraRegistry {
    pub fn new(
        weights: &BTreeMap<String, Tensor>,
        config: V4LoraConfig,
        device: Device,
    ) -> Result<Self> {
        let vs = nn::VarStore::new(device);
        let p = vs.root();
        let mut adapters = BTreeMap::new();

        for &layer in &config.target_layers {
            for &module in &config.target_modules {
                let name = format!("layers.{layer}.{}.weight", module.weight_suffix());
                let weight = weights
                    .get(&name)
                    .ok_or_else(|| anyhow::anyhow!("V4 LoRA base weight not found: {name}"))?;
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

    pub fn load(path: &Path, config: V4LoraConfig) -> Result<Self> {
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
pub struct V4LoraManifest {
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

pub fn write_lora_manifest(path: &Path, manifest: &V4LoraManifest) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(manifest)? + "\n")?;
    Ok(())
}

/// Compute W' = W + (B @ A) * scale for V4 LoRA
pub fn v4_lora_weight(
    base: &Tensor,
    layer: usize,
    module: V4LoraTargetModule,
    registry: &V4LoraRegistry,
) -> Tensor {
    if let Some((lora_a, lora_b)) = registry.adapters.get(&(layer, module)) {
        let scale = registry.config.alpha as f64 / lora_a.size()[0] as f64;
        let delta = lora_b.matmul(lora_a) * scale;
        base.shallow_clone() + delta.to_device(base.device()).to_kind(base.kind())
    } else {
        base.shallow_clone()
    }
}
