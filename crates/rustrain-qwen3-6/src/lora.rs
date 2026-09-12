//! Qwen3.6 LoRA adapter registry — VarStore-backed, stores tensors directly.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use tch::{nn, Kind, Tensor};
use tracing::info;

use crate::config::Qwen36RuntimeConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Qwen36LoraTargetModule {
    QProj,
    KProj,
    VProj,
    OProj,
    InProjQkv,
    InProjZ,
    OutProj,
    SharedGateProj,
    SharedUpProj,
    SharedDownProj,
}

impl Qwen36LoraTargetModule {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "q_proj" => Ok(Self::QProj),
            "k_proj" => Ok(Self::KProj),
            "v_proj" => Ok(Self::VProj),
            "o_proj" => Ok(Self::OProj),
            "in_proj_qkv" => Ok(Self::InProjQkv),
            "in_proj_z" => Ok(Self::InProjZ),
            "out_proj" => Ok(Self::OutProj),
            "shared_gate_proj" => Ok(Self::SharedGateProj),
            "shared_up_proj" => Ok(Self::SharedUpProj),
            "shared_down_proj" => Ok(Self::SharedDownProj),
            other => bail!("unknown LoRA target module: {other}"),
        }
    }

    pub fn suffix(&self) -> &'static str {
        match self {
            Self::QProj => "self_attn.q_proj",
            Self::KProj => "self_attn.k_proj",
            Self::VProj => "self_attn.v_proj",
            Self::OProj => "self_attn.o_proj",
            Self::InProjQkv => "linear_attn.in_proj_qkv",
            Self::InProjZ => "linear_attn.in_proj_z",
            Self::OutProj => "linear_attn.out_proj",
            Self::SharedGateProj => "mlp.shared_expert.gate_proj",
            Self::SharedUpProj => "mlp.shared_expert.up_proj",
            Self::SharedDownProj => "mlp.shared_expert.down_proj",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen36LoraConfig {
    pub rank: i64,
    pub alpha: f64,
    pub target_layers: Vec<usize>,
    pub target_modules: Vec<Qwen36LoraTargetModule>,
}

pub struct Qwen36LoraRegistry {
    pub config: Qwen36LoraConfig,
    pub var_store: nn::VarStore,
    pub adapters: BTreeMap<(usize, Qwen36LoraTargetModule), (Tensor, Tensor)>,
}

impl Qwen36LoraRegistry {
    pub fn new(
        weights: &BTreeMap<String, Tensor>,
        config: &Qwen36RuntimeConfig,
        lora_config: Qwen36LoraConfig,
        device: tch::Device,
    ) -> Result<Self> {
        let mut var_store = nn::VarStore::new(device);
        let mut adapters = BTreeMap::new();
        let p = var_store.root();

        for &layer_idx in &lora_config.target_layers {
            let layer_prefix = format!("{}layers.{}", config.weight_prefix, layer_idx);
            for &module in &lora_config.target_modules {
                let weight_name = match module {
                    Qwen36LoraTargetModule::QProj => format!("{layer_prefix}.self_attn.q_proj.weight"),
                    Qwen36LoraTargetModule::KProj => format!("{layer_prefix}.self_attn.k_proj.weight"),
                    Qwen36LoraTargetModule::VProj => format!("{layer_prefix}.self_attn.v_proj.weight"),
                    Qwen36LoraTargetModule::OProj => format!("{layer_prefix}.self_attn.o_proj.weight"),
                    Qwen36LoraTargetModule::InProjQkv => format!("{layer_prefix}.linear_attn.in_proj_qkv.weight"),
                    Qwen36LoraTargetModule::InProjZ => format!("{layer_prefix}.linear_attn.in_proj_z.weight"),
                    Qwen36LoraTargetModule::OutProj => format!("{layer_prefix}.linear_attn.out_proj.weight"),
                    Qwen36LoraTargetModule::SharedGateProj => format!("{layer_prefix}.mlp.shared_expert.gate_proj.weight"),
                    Qwen36LoraTargetModule::SharedUpProj => format!("{layer_prefix}.mlp.shared_expert.up_proj.weight"),
                    Qwen36LoraTargetModule::SharedDownProj => format!("{layer_prefix}.mlp.shared_expert.down_proj.weight"),
                };

                let base_weight = match weights.get(&weight_name) {
                    Some(w) => w,
                    None => {
                        // Skip modules that don't exist for this layer type
                        // (e.g., q_proj doesn't exist for linear attention layers)
                        tracing::debug!("skipping LoRA target {weight_name} — not found (layer {layer_idx} may be different attention type)");
                        continue;
                    }
                };

                let (out_features, in_features) = (base_weight.size()[0], base_weight.size()[1]);
                let name = module.suffix().replace('.', "_");
                let scale = 1.0 / (lora_config.rank as f64).sqrt();
                let lora_a = p.randn(&format!("lora_a_{layer_idx}_{name}"), &[lora_config.rank, in_features], 0.0, scale);
                let lora_b = p.zeros(&format!("lora_b_{layer_idx}_{name}"), &[out_features, lora_config.rank]);
                adapters.insert((layer_idx, module), (lora_a, lora_b));
            }
        }

        Ok(Self { config: lora_config, var_store, adapters })
    }

    pub fn trainable_variables(&self) -> Vec<Tensor> {
        self.var_store.trainable_variables()
    }

    pub fn adapter_tensors(&self, layer: usize, module: Qwen36LoraTargetModule) -> Option<(Tensor, Tensor)> {
        let (a, b) = self.adapters.get(&(layer, module))?;
        Some((a.shallow_clone(), b.shallow_clone()))
    }

    /// Get references to the actual VarStore adapter tensors (for training backward).
    pub fn adapter_ref(&self, layer: usize, module: Qwen36LoraTargetModule) -> Option<(&Tensor, &Tensor)> {
        let (a, b) = self.adapters.get(&(layer, module))?;
        Some((a, b))
    }

    pub fn scaling(&self) -> f64 {
        self.config.alpha / self.config.rank as f64
    }

    pub fn trainable_param_count(&self) -> usize {
        self.trainable_variables().iter().map(|v| v.numel() as usize).sum()
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        // Save as PyTorch format for simplicity (safetensors requires tch feature)
        let vars = self.var_store.variables();
        let mut named: Vec<(String, &Tensor)> = Vec::new();
        for (name, tensor) in &vars {
            named.push((name.clone(), tensor));
        }
        named.sort_by(|a, b| a.0.cmp(&b.0));

        // Write safetensors manually: header JSON + raw data
        use std::io::Write;
        let mut tensors_data: Vec<Vec<u8>> = Vec::new();
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        for (name, tensor) in &named {
            let t = tensor.to_device(tch::Device::Cpu).contiguous().to_kind(Kind::Float);
            let shape: Vec<i64> = t.size().iter().copied().map(|d| d).collect();
            let t_flat = t.reshape([-1]);
            let data: Vec<f32> = Vec::<f32>::try_from(&t_flat)?;
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            header.insert(name.clone(), serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, offset + bytes.len() as u64],
            }));
            offset += bytes.len() as u64;
            tensors_data.push(bytes);
        }
        let header_str = serde_json::to_string(&serde_json::Value::Object(header))?;
        let header_bytes = header_str.into_bytes();
        let header_len = header_bytes.len() as u64;

        let file = std::fs::File::create(path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        let mut writer = std::io::BufWriter::new(file);
        writer.write_all(&header_len.to_le_bytes())?;
        writer.write_all(&header_bytes)?;
        for data in &tensors_data {
            writer.write_all(data)?;
        }
        info!("saved LoRA adapter to {}", path.display());
        Ok(())
    }
}
