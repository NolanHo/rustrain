use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use tch::{Device, Kind, Tensor, nn};

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
    /// Cached LoRA deltas: (layer, module) → delta tensor (B@A * scale)
    /// Invalidated after Adam step, recomputed on next lora_attention_weights call.
    pub delta_cache: BTreeMap<(usize, Glm5LoraTargetModule), Option<Tensor>>,
    /// Cached dequanted base weights (frozen, computed once)
    pub base_cache: BTreeMap<(usize, Glm5LoraTargetModule), Option<Tensor>>,
}

impl Glm5LoraRegistry {
    pub fn new(
        weights: &BTreeMap<String, Tensor>,
        config: Glm5LoraConfig,
        device: Device,
    ) -> Result<Self> {
        if config.rank <= 0 || config.alpha <= 0 {
            bail!("GLM-5 LoRA rank and alpha must be positive");
        }
        let mut targets = std::collections::BTreeSet::new();
        for &layer in &config.target_layers {
            for &module in &config.target_modules {
                if !targets.insert((layer, module)) {
                    bail!("duplicate GLM-5 LoRA target: layer {layer}, {module:?}");
                }
            }
        }
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
            delta_cache: BTreeMap::new(),
            base_cache: BTreeMap::new(),
        })
    }

    pub fn param_count(&self) -> usize {
        self.adapters
            .values()
            .map(|(lora_a, lora_b)| lora_a.numel() + lora_b.numel())
            .sum()
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
            delta_cache: BTreeMap::new(),
            base_cache: BTreeMap::new(),
        })
    }

    /// Invalidate delta cache (call after Adam step updates LoRA params)
    pub fn invalidate_delta_cache(&mut self) {
        self.delta_cache.clear();
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
/// Uses C++ dequant_fp8 for FP8 weights to avoid the tch-rs view bug.
pub fn glm5_lora_weight(
    base: &Tensor,
    weight_scale: Option<&Tensor>,
    layer: usize,
    module: Glm5LoraTargetModule,
    registry: &mut Glm5LoraRegistry,
) -> Result<Tensor> {
    if let Some((lora_a, lora_b)) = registry.adapters.get(&(layer, module)) {
        let lora_scale = registry.config.alpha as f64 / lora_a.size()[0] as f64;

        // Use cached delta (B@A * scale) if available, recompute otherwise
        let delta = if let Some(Some(cached)) = registry.delta_cache.get(&(layer, module)) {
            cached.shallow_clone()
        } else {
            let d = lora_b.matmul(lora_a) * lora_scale;
            registry
                .delta_cache
                .insert((layer, module), Some(d.shallow_clone()));
            d
        };

        // Use cached dequanted base weight if available
        let base_dequant = if let Some(Some(cached)) = registry.base_cache.get(&(layer, module)) {
            cached.shallow_clone()
        } else {
            let bd = if base.kind() == Kind::Float8e4m3fn {
                if let Some(s) = weight_scale {
                    rustrain_deepseek_v4::fp8_kernel::dequant_fp8_weight(base, s)
                        .context("failed to dequantize FP8 GLM-5 base weight for LoRA")?
                } else {
                    bail!("FP8 GLM-5 LoRA base weight is missing weight_scale_inv")
                }
            } else {
                base.shallow_clone()
            };
            registry
                .base_cache
                .insert((layer, module), Some(bd.shallow_clone()));
            bd
        };

        Ok(&base_dequant
            + &delta
                .to_device(base_dequant.device())
                .to_kind(base_dequant.kind()))
    } else {
        Ok(base.shallow_clone())
    }
}

/// Apply LoRA to attention weights for a specific layer
pub fn lora_attention_weights(
    base: &Glm5AttentionWeights,
    layer: usize,
    registry: &mut Glm5LoraRegistry,
) -> Result<Glm5AttentionWeights> {
    let q_a_modified = registry
        .adapters
        .contains_key(&(layer, Glm5LoraTargetModule::WqA));
    let q_b_modified = registry
        .adapters
        .contains_key(&(layer, Glm5LoraTargetModule::WqB));
    let kv_a_modified = registry
        .adapters
        .contains_key(&(layer, Glm5LoraTargetModule::Wkv));
    let o_modified = registry
        .adapters
        .contains_key(&(layer, Glm5LoraTargetModule::Wo));

    Ok(Glm5AttentionWeights {
        q_a_proj: glm5_lora_weight(
            &base.q_a_proj,
            base.q_a_proj_scale.as_ref(),
            layer,
            Glm5LoraTargetModule::WqA,
            registry,
        )?,
        q_a_layernorm: base.q_a_layernorm.shallow_clone(),
        q_b_proj: glm5_lora_weight(
            &base.q_b_proj,
            base.q_b_proj_scale.as_ref(),
            layer,
            Glm5LoraTargetModule::WqB,
            registry,
        )?,
        kv_a_proj_with_mqa: glm5_lora_weight(
            &base.kv_a_proj_with_mqa,
            base.kv_a_proj_scale.as_ref(),
            layer,
            Glm5LoraTargetModule::Wkv,
            registry,
        )?,
        kv_a_layernorm: base.kv_a_layernorm.shallow_clone(),
        kv_b_proj: base.kv_b_proj.shallow_clone(),
        o_proj: glm5_lora_weight(
            &base.o_proj,
            base.o_proj_scale.as_ref(),
            layer,
            Glm5LoraTargetModule::Wo,
            registry,
        )?,
        indexer_k_norm_weight: base
            .indexer_k_norm_weight
            .as_ref()
            .map(|t| t.shallow_clone()),
        indexer_k_norm_bias: base.indexer_k_norm_bias.as_ref().map(|t| t.shallow_clone()),
        indexer_weights_proj: base
            .indexer_weights_proj
            .as_ref()
            .map(|t| t.shallow_clone()),
        indexer_wk: base.indexer_wk.as_ref().map(|t| t.shallow_clone()),
        indexer_wq_b: base.indexer_wq_b.as_ref().map(|t| t.shallow_clone()),
        // A merged LoRA weight is already dequantized. Untargeted layers still
        // hold their original FP8 weight and must retain its block scale.
        q_a_proj_scale: (!q_a_modified)
            .then(|| base.q_a_proj_scale.as_ref().map(|t| t.shallow_clone()))
            .flatten(),
        q_b_proj_scale: (!q_b_modified)
            .then(|| base.q_b_proj_scale.as_ref().map(|t| t.shallow_clone()))
            .flatten(),
        kv_a_proj_scale: (!kv_a_modified)
            .then(|| base.kv_a_proj_scale.as_ref().map(|t| t.shallow_clone()))
            .flatten(),
        // kv_b_proj is NOT a LoRA target — keep its scale
        kv_b_proj_scale: base.kv_b_proj_scale.as_ref().map(|t| t.shallow_clone()),
        o_proj_scale: (!o_modified)
            .then(|| base.o_proj_scale.as_ref().map(|t| t.shallow_clone()))
            .flatten(),
        indexer_weights_proj_scale: base
            .indexer_weights_proj_scale
            .as_ref()
            .map(|t| t.shallow_clone()),
        indexer_wk_scale: base.indexer_wk_scale.as_ref().map(|t| t.shallow_clone()),
        indexer_wq_b_scale: base.indexer_wq_b_scale.as_ref().map(|t| t.shallow_clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attention_weights() -> Glm5AttentionWeights {
        let matrix = || Tensor::zeros([2, 2], (Kind::Float, Device::Cpu));
        let norm = || Tensor::ones([2], (Kind::Float, Device::Cpu));
        let scale = || Some(Tensor::ones([1, 1], (Kind::Float, Device::Cpu)));
        Glm5AttentionWeights {
            q_a_proj: matrix(),
            q_a_layernorm: norm(),
            q_b_proj: matrix(),
            kv_a_proj_with_mqa: matrix(),
            kv_a_layernorm: norm(),
            kv_b_proj: matrix(),
            o_proj: matrix(),
            indexer_k_norm_weight: None,
            indexer_k_norm_bias: None,
            indexer_weights_proj: None,
            indexer_wk: None,
            indexer_wq_b: None,
            q_a_proj_scale: scale(),
            q_b_proj_scale: scale(),
            kv_a_proj_scale: scale(),
            kv_b_proj_scale: scale(),
            o_proj_scale: scale(),
            indexer_weights_proj_scale: None,
            indexer_wk_scale: None,
            indexer_wq_b_scale: None,
        }
    }

    #[test]
    fn lora_only_clears_scales_for_weights_modified_on_that_layer() -> Result<()> {
        let mut weights = BTreeMap::new();
        for suffix in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"] {
            weights.insert(
                format!("model.layers.0.self_attn.{suffix}.weight"),
                Tensor::zeros([2, 2], (Kind::Float, Device::Cpu)),
            );
        }
        let config = Glm5LoraConfig {
            rank: 1,
            alpha: 1,
            target_layers: vec![0],
            target_modules: vec![
                Glm5LoraTargetModule::WqA,
                Glm5LoraTargetModule::WqB,
                Glm5LoraTargetModule::Wkv,
                Glm5LoraTargetModule::Wo,
            ],
        };
        let mut registry = Glm5LoraRegistry::new(&weights, config, Device::Cpu)?;

        let targeted = lora_attention_weights(&attention_weights(), 0, &mut registry)?;
        assert!(targeted.q_a_proj_scale.is_none());
        assert!(targeted.q_b_proj_scale.is_none());
        assert!(targeted.kv_a_proj_scale.is_none());
        assert!(targeted.o_proj_scale.is_none());
        assert!(targeted.kv_b_proj_scale.is_some());

        let untargeted = lora_attention_weights(&attention_weights(), 1, &mut registry)?;
        assert!(untargeted.q_a_proj_scale.is_some());
        assert!(untargeted.q_b_proj_scale.is_some());
        assert!(untargeted.kv_a_proj_scale.is_some());
        assert!(untargeted.o_proj_scale.is_some());
        assert!(untargeted.kv_b_proj_scale.is_some());
        Ok(())
    }
}
