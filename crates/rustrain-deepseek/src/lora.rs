use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor, no_grad};

use crate::model::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize)]
pub enum DeepSeekLoraTargetModule {
    QAProj,
    QBProj,
    KVAProj,
    KVBProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
}

impl DeepSeekLoraTargetModule {
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "q_a_proj" => Ok(Self::QAProj),
            "q_b_proj" => Ok(Self::QBProj),
            "kv_a_proj_with_mqa" => Ok(Self::KVAProj),
            "kv_b_proj" => Ok(Self::KVBProj),
            "o_proj" => Ok(Self::OProj),
            "gate_proj" => Ok(Self::GateProj),
            "up_proj" => Ok(Self::UpProj),
            "down_proj" => Ok(Self::DownProj),
            other => bail!("unknown LoRA target module: {other}"),
        }
    }

    pub fn weight_suffix(&self) -> &'static str {
        match self {
            Self::QAProj => "self_attn.q_a_proj",
            Self::QBProj => "self_attn.q_b_proj",
            Self::KVAProj => "self_attn.kv_a_proj_with_mqa",
            Self::KVBProj => "self_attn.kv_b_proj",
            Self::OProj => "self_attn.o_proj",
            Self::GateProj => "mlp.gate_proj",
            Self::UpProj => "mlp.up_proj",
            Self::DownProj => "mlp.down_proj",
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeepSeekLoraConfig {
    pub rank: i64,
    pub alpha: i64,
    pub target_layers: Vec<usize>,
    pub target_modules: Vec<DeepSeekLoraTargetModule>,
}

pub struct DeepSeekLoraAdapter {
    pub lora_a: Tensor,
    pub lora_b: Tensor,
    pub alpha: i64,
}

impl DeepSeekLoraAdapter {
    pub fn new(in_features: i64, out_features: i64, rank: i64, alpha: i64, device: Device) -> Self {
        let lora_a =
            Tensor::randn([rank, in_features], (Kind::Float, device)) / (rank as f64).sqrt();
        let lora_b = Tensor::zeros([out_features, rank], (Kind::Float, device));
        Self {
            lora_a,
            lora_b,
            alpha,
        }
    }

    pub fn delta_weight(&self) -> Tensor {
        let scale = self.alpha as f64 / self.lora_a.size()[0] as f64;
        self.lora_b.matmul(&self.lora_a) * scale
    }
}

pub struct DeepSeekLoraRegistry {
    pub adapters: BTreeMap<(usize, DeepSeekLoraTargetModule), DeepSeekLoraAdapter>,
    pub config: DeepSeekLoraConfig,
}

impl DeepSeekLoraRegistry {
    pub fn new(
        weights: &BTreeMap<String, Tensor>,
        config: DeepSeekLoraConfig,
        device: Device,
    ) -> Result<Self> {
        let mut adapters = BTreeMap::new();
        for &layer in &config.target_layers {
            for &module in &config.target_modules {
                let name = format!("model.layers.{layer}.{}.weight", module.weight_suffix());
                let weight = weights
                    .get(&name)
                    .ok_or_else(|| anyhow::anyhow!("LoRA base weight not found: {name}"))?;
                let out_features = weight.size()[0];
                let in_features = weight.size()[1];
                let adapter = DeepSeekLoraAdapter::new(
                    in_features,
                    out_features,
                    config.rank,
                    config.alpha,
                    device,
                );
                adapters.insert((layer, module), adapter);
            }
        }
        Ok(Self { adapters, config })
    }

    pub fn apply_to_weights(&self, weights: &mut BTreeMap<String, Tensor>) {
        for ((layer, module), adapter) in &self.adapters {
            let name = format!("model.layers.{layer}.{}.weight", module.weight_suffix());
            if let Some(base) = weights.get(&name) {
                let delta = adapter.delta_weight();
                let modified = base.shallow_clone() + &delta;
                weights.insert(name, modified);
            }
        }
    }

    pub fn trainable_parameters(&self) -> Vec<Tensor> {
        self.adapters
            .values()
            .flat_map(|a| [a.lora_a.shallow_clone(), a.lora_b.shallow_clone()])
            .collect()
    }

    pub fn param_count(&self) -> usize {
        self.adapters.len() * 2
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut tensors = Vec::new();
        for ((layer, module), adapter) in &self.adapters {
            let prefix = format!("model.layers.{layer}.{}", module.weight_suffix());
            tensors.push((format!("{prefix}.lora_a"), adapter.lora_a.shallow_clone()));
            tensors.push((format!("{prefix}.lora_b"), adapter.lora_b.shallow_clone()));
        }
        Tensor::write_safetensors(&tensors, path)
            .with_context(|| format!("failed to write LoRA adapter to {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: &Path, config: DeepSeekLoraConfig) -> Result<Self> {
        let tensors = Tensor::read_safetensors(path)
            .with_context(|| format!("failed to read LoRA adapter from {}", path.display()))?;
        let map: BTreeMap<String, Tensor> = tensors.into_iter().collect();
        let mut adapters = BTreeMap::new();
        for &layer in &config.target_layers {
            for &module in &config.target_modules {
                let prefix = format!("model.layers.{layer}.{}", module.weight_suffix());
                let lora_a = map
                    .get(&format!("{prefix}.lora_a"))
                    .ok_or_else(|| anyhow::anyhow!("missing lora_a for {prefix}"))?
                    .shallow_clone();
                let lora_b = map
                    .get(&format!("{prefix}.lora_b"))
                    .ok_or_else(|| anyhow::anyhow!("missing lora_b for {prefix}"))?
                    .shallow_clone();
                adapters.insert(
                    (layer, module),
                    DeepSeekLoraAdapter {
                        lora_a,
                        lora_b,
                        alpha: config.alpha,
                    },
                );
            }
        }
        Ok(Self { adapters, config })
    }
}

// ── Manifest ──────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct DeepSeekLoraManifest {
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

pub fn write_lora_manifest(path: &Path, manifest: &DeepSeekLoraManifest) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(manifest)? + "\n")?;
    Ok(())
}
