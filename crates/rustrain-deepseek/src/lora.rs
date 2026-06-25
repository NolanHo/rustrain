use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor, nn};

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

pub struct DeepSeekLoraRegistry {
    pub adapters: BTreeMap<(usize, DeepSeekLoraTargetModule), (Tensor, Tensor)>,
    pub config: DeepSeekLoraConfig,
    pub var_store: nn::VarStore,
}

impl DeepSeekLoraRegistry {
    pub fn new(
        weights: &BTreeMap<String, Tensor>,
        config: DeepSeekLoraConfig,
        device: Device,
    ) -> Result<Self> {
        let vs = nn::VarStore::new(device);
        let p = vs.root();
        let mut adapters = BTreeMap::new();

        for &layer in &config.target_layers {
            for &module in &config.target_modules {
                let name = format!("model.layers.{layer}.{}.weight", module.weight_suffix());
                let weight = weights
                    .get(&name)
                    .ok_or_else(|| anyhow::anyhow!("LoRA base weight not found: {name}"))?;
                let out_features = weight.size()[0];
                let in_features = weight.size()[1];

                let path = format!("layer{layer}/{:?}", module);
                // lora_a: [rank, in_features], normal init scaled by 1/sqrt(rank)
                let lora_a = p.randn(
                    &format!("{path}/lora_a"),
                    &[config.rank, in_features],
                    0.0,
                    1.0 / (config.rank as f64).sqrt(),
                );
                // lora_b: [out_features, rank], zero init
                let mut lora_b = p.zeros(&format!("{path}/lora_b"), &[out_features, config.rank]);

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
            let prefix = format!("model.layers.{layer}.{}", module.weight_suffix());
            tensors.push((format!("{prefix}.lora_a"), lora_a.shallow_clone()));
            tensors.push((format!("{prefix}.lora_b"), lora_b.shallow_clone()));
        }
        Tensor::write_safetensors(&tensors, path)?;
        Ok(())
    }

    pub fn load(path: &Path, config: DeepSeekLoraConfig) -> Result<Self> {
        let tensors = Tensor::read_safetensors(path)?;
        let map: BTreeMap<String, Tensor> = tensors.into_iter().collect();
        let vs = nn::VarStore::new(Device::Cpu);
        let p = vs.root();
        let mut adapters = BTreeMap::new();

        for &layer in &config.target_layers {
            for &module in &config.target_modules {
                let prefix = format!("model.layers.{layer}.{}", module.weight_suffix());
                let path_str = format!("layer{layer}/{:?}", module);
                let lora_a_stored = map
                    .get(&format!("{prefix}.lora_a"))
                    .ok_or_else(|| anyhow::anyhow!("missing lora_a for {prefix}"))?;
                let lora_b_stored = map
                    .get(&format!("{prefix}.lora_b"))
                    .ok_or_else(|| anyhow::anyhow!("missing lora_b for {prefix}"))?;

                // Create new VarStore variables and copy data
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
