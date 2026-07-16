//! Qwen3.6 LoRA adapter registry — VarStore-backed, stores tensors directly.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tch::{nn, Kind, Tensor};
use tracing::info;

use crate::config::{LayerType, Qwen36RuntimeConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen36LoraTargetModule {
    QProj,
    KProj,
    VProj,
    OProj,
    InProjQkv,
    InProjZ,
    InProjA,
    InProjB,
    OutProj,
    GateProj,
    UpProj,
    DownProj,
    SharedGateProj,
    SharedUpProj,
    SharedDownProj,
    ExpertsGateUpProj,
    ExpertsDownProj,
}

#[derive(Debug, Clone)]
pub struct Qwen36NativeLoraSlot {
    pub index: usize,
    pub layer: usize,
    pub module: Qwen36LoraTargetModule,
    pub active: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen36AdapterConfig {
    pub format_version: u32,
    pub peft_type: String,
    pub task_type: String,
    pub base_model_name_or_path: String,
    pub rustrain_architecture: String,
    pub model_family: String,
    pub r: i64,
    pub lora_alpha: f64,
    pub target_layers: Vec<usize>,
    pub target_modules: Vec<String>,
    pub adapter_dtype: String,
    pub bias: String,
    pub inference_mode: bool,
}

pub struct Qwen36AdapterArtifact {
    pub config: Qwen36AdapterConfig,
    pub tensors: BTreeMap<String, Tensor>,
}

#[derive(Debug, Serialize)]
struct PeftAdapterConfig<'a> {
    peft_type: &'static str,
    task_type: &'static str,
    base_model_name_or_path: &'a str,
    r: i64,
    lora_alpha: f64,
    lora_dropout: f64,
    target_modules: &'a [String],
    layers_to_transform: &'a [usize],
    layers_pattern: &'static str,
    bias: &'static str,
    fan_in_fan_out: bool,
    inference_mode: bool,
}

#[derive(Debug, Deserialize)]
struct PeftAdapterConfigOwned {
    #[serde(default = "default_lora_type")]
    peft_type: String,
    #[serde(default = "default_task_type")]
    task_type: String,
    #[serde(default)]
    base_model_name_or_path: String,
    r: i64,
    lora_alpha: f64,
    #[serde(default)]
    target_modules: Vec<String>,
    #[serde(default)]
    layers_to_transform: Vec<usize>,
    #[serde(default = "default_bias")]
    bias: String,
    #[serde(default)]
    inference_mode: bool,
}

fn default_lora_type() -> String {
    "LORA".to_string()
}

fn default_task_type() -> String {
    "CAUSAL_LM".to_string()
}

fn default_bias() -> String {
    "none".to_string()
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
            "in_proj_a" => Ok(Self::InProjA),
            "in_proj_b" => Ok(Self::InProjB),
            "out_proj" => Ok(Self::OutProj),
            "gate_proj" => Ok(Self::GateProj),
            "up_proj" => Ok(Self::UpProj),
            "down_proj" => Ok(Self::DownProj),
            "shared_gate_proj" => Ok(Self::SharedGateProj),
            "shared_up_proj" => Ok(Self::SharedUpProj),
            "shared_down_proj" => Ok(Self::SharedDownProj),
            "experts_gate_up_proj" => Ok(Self::ExpertsGateUpProj),
            "experts_down_proj" => Ok(Self::ExpertsDownProj),
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
            Self::InProjA => "linear_attn.in_proj_a",
            Self::InProjB => "linear_attn.in_proj_b",
            Self::OutProj => "linear_attn.out_proj",
            Self::GateProj => "mlp.gate_proj",
            Self::UpProj => "mlp.up_proj",
            Self::DownProj => "mlp.down_proj",
            Self::SharedGateProj => "mlp.shared_expert.gate_proj",
            Self::SharedUpProj => "mlp.shared_expert.up_proj",
            Self::SharedDownProj => "mlp.shared_expert.down_proj",
            Self::ExpertsGateUpProj => "mlp.experts.gate_up_proj",
            Self::ExpertsDownProj => "mlp.experts.down_proj",
        }
    }

    /// Stable C++ projection identifier used by the native training context.
    pub fn cpp_name(&self) -> &'static str {
        match self {
            Self::QProj => "q_proj",
            Self::KProj => "k_proj",
            Self::VProj => "v_proj",
            Self::OProj => "o_proj",
            Self::InProjQkv => "in_proj_qkv",
            Self::InProjZ => "in_proj_z",
            Self::InProjA => "in_proj_a",
            Self::InProjB => "in_proj_b",
            Self::OutProj => "out_proj",
            Self::GateProj => "gate_proj",
            Self::UpProj => "up_proj",
            Self::DownProj => "down_proj",
            Self::SharedGateProj => "shared_gate_proj",
            Self::SharedUpProj => "shared_up_proj",
            Self::SharedDownProj => "shared_down_proj",
            Self::ExpertsGateUpProj => "experts_gate_up_proj",
            Self::ExpertsDownProj => "experts_down_proj",
        }
    }
}

pub fn native_lora_slots(
    config: &Qwen36RuntimeConfig,
    lora_config: &Qwen36LoraConfig,
) -> Vec<Qwen36NativeLoraSlot> {
    let target_layers = lora_config
        .target_layers
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let target_modules = lora_config
        .target_modules
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let all_layers = target_layers.is_empty();
    let all_modules = target_modules.is_empty();
    let mut slots = Vec::new();

    for (layer, layer_type) in config.layer_types.iter().enumerate() {
        let modules: &[Qwen36LoraTargetModule] = match layer_type {
            LayerType::FullAttention => &[
                Qwen36LoraTargetModule::QProj,
                Qwen36LoraTargetModule::KProj,
                Qwen36LoraTargetModule::VProj,
                Qwen36LoraTargetModule::OProj,
            ],
            LayerType::LinearAttention => &[
                Qwen36LoraTargetModule::InProjQkv,
                Qwen36LoraTargetModule::InProjZ,
                Qwen36LoraTargetModule::InProjA,
                Qwen36LoraTargetModule::InProjB,
                Qwen36LoraTargetModule::OutProj,
            ],
        };
        for &module in modules {
            slots.push(Qwen36NativeLoraSlot {
                index: slots.len(),
                layer,
                module,
                active: (all_layers || target_layers.contains(&layer))
                    && (all_modules || target_modules.contains(&module)),
            });
        }
        let mlp_modules: &[Qwen36LoraTargetModule] = if config.is_moe {
            &[
                Qwen36LoraTargetModule::SharedGateProj,
                Qwen36LoraTargetModule::SharedUpProj,
                Qwen36LoraTargetModule::SharedDownProj,
                Qwen36LoraTargetModule::ExpertsGateUpProj,
                Qwen36LoraTargetModule::ExpertsDownProj,
            ]
        } else {
            &[
                Qwen36LoraTargetModule::GateProj,
                Qwen36LoraTargetModule::UpProj,
                Qwen36LoraTargetModule::DownProj,
            ]
        };
        for &module in mlp_modules {
            slots.push(Qwen36NativeLoraSlot {
                index: slots.len(),
                layer,
                module,
                active: (all_layers || target_layers.contains(&layer))
                    && (all_modules || target_modules.contains(&module)),
            });
        }
    }
    slots
}

pub fn validate_lora_targets(
    runtime_config: &Qwen36RuntimeConfig,
    lora_config: &Qwen36LoraConfig,
) -> Result<()> {
    let slots = native_lora_slots(runtime_config, lora_config);
    if !slots.iter().any(|slot| slot.active) {
        bail!("LoRA targets do not resolve to any projection in this model");
    }
    if !runtime_config.is_moe
        && lora_config.target_modules.iter().any(|module| {
            matches!(
                module,
                Qwen36LoraTargetModule::SharedGateProj
                    | Qwen36LoraTargetModule::SharedUpProj
                    | Qwen36LoraTargetModule::SharedDownProj
                    | Qwen36LoraTargetModule::ExpertsGateUpProj
                    | Qwen36LoraTargetModule::ExpertsDownProj
            )
        })
    {
        bail!("shared expert LoRA targets require a MoE Qwen model");
    }
    if runtime_config.is_moe
        && lora_config.target_modules.iter().any(|module| {
            matches!(
                module,
                Qwen36LoraTargetModule::GateProj
                    | Qwen36LoraTargetModule::UpProj
                    | Qwen36LoraTargetModule::DownProj
            )
        })
    {
        bail!("dense MLP LoRA targets require a dense Qwen model");
    }
    Ok(())
}

fn adapter_tensor_prefix(
    config: &Qwen36RuntimeConfig,
    layer: usize,
    module: Qwen36LoraTargetModule,
) -> String {
    format!(
        "base_model.model.{}layers.{layer}.{}",
        config.weight_prefix,
        module.suffix()
    )
}

/// Parse a canonical PEFT tensor name back to its layer, projection and side.
pub fn parse_adapter_tensor_name(
    config: &Qwen36RuntimeConfig,
    name: &str,
) -> Result<(usize, Qwen36LoraTargetModule, bool)> {
    let (base, is_b) = if let Some(base) = name.strip_suffix(".lora_A.weight") {
        (base, false)
    } else if let Some(base) = name.strip_suffix(".lora_B.weight") {
        (base, true)
    } else {
        bail!("invalid LoRA tensor name: {name}");
    };
    let prefix = format!("base_model.model.{}layers.", config.weight_prefix);
    let rest = base
        .strip_prefix(&prefix)
        .with_context(|| format!("LoRA tensor has unexpected module prefix: {name}"))?;
    let (layer, suffix) = rest
        .split_once('.')
        .with_context(|| format!("LoRA tensor is missing projection path: {name}"))?;
    let layer = layer
        .parse::<usize>()
        .with_context(|| format!("invalid LoRA layer in tensor name: {name}"))?;
    let module = [
        Qwen36LoraTargetModule::QProj,
        Qwen36LoraTargetModule::KProj,
        Qwen36LoraTargetModule::VProj,
        Qwen36LoraTargetModule::OProj,
        Qwen36LoraTargetModule::InProjQkv,
        Qwen36LoraTargetModule::InProjZ,
        Qwen36LoraTargetModule::InProjA,
        Qwen36LoraTargetModule::InProjB,
        Qwen36LoraTargetModule::OutProj,
        Qwen36LoraTargetModule::GateProj,
        Qwen36LoraTargetModule::UpProj,
        Qwen36LoraTargetModule::DownProj,
        Qwen36LoraTargetModule::SharedGateProj,
        Qwen36LoraTargetModule::SharedUpProj,
        Qwen36LoraTargetModule::SharedDownProj,
        Qwen36LoraTargetModule::ExpertsGateUpProj,
        Qwen36LoraTargetModule::ExpertsDownProj,
    ]
    .into_iter()
    .find(|module| module.suffix() == suffix)
    .with_context(|| format!("unknown Qwen LoRA projection suffix: {suffix}"))?;
    Ok((layer, module, is_b))
}

fn resolve_artifact_paths(path: &Path) -> (PathBuf, PathBuf) {
    if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors") {
        let config_path = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("adapter_config.json");
        (path.to_path_buf(), config_path)
    } else {
        (
            path.join("adapter_model.safetensors"),
            path.join("adapter_config.json"),
        )
    }
}

fn rustrain_config_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("rustrain_adapter.json")
}

impl Qwen36AdapterArtifact {
    pub fn from_native_exports(
        model_name: &str,
        architecture: &str,
        base_model_path: Option<&Path>,
        runtime_config: &Qwen36RuntimeConfig,
        lora_config: &Qwen36LoraConfig,
        exported: Vec<(Tensor, Tensor)>,
    ) -> Result<Self> {
        let slots = native_lora_slots(runtime_config, lora_config);
        if exported.len() != slots.len() {
            bail!(
                "native LoRA export returned {} slots, expected {} for this model",
                exported.len(),
                slots.len()
            );
        }

        let target_layers = slots
            .iter()
            .filter(|slot| slot.active)
            .map(|slot| slot.layer)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let target_modules = slots
            .iter()
            .filter(|slot| slot.active)
            .map(|slot| slot.module.cpp_name().to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        let mut tensors = BTreeMap::new();
        for (slot, (a, b)) in slots.into_iter().zip(exported) {
            if !slot.active {
                continue;
            }
            let prefix = adapter_tensor_prefix(runtime_config, slot.layer, slot.module);
            tensors.insert(
                format!("{prefix}.lora_A.weight"),
                a.to_device(tch::Device::Cpu).to_kind(Kind::Float),
            );
            tensors.insert(
                format!("{prefix}.lora_B.weight"),
                b.to_device(tch::Device::Cpu).to_kind(Kind::Float),
            );
        }
        if tensors.is_empty() {
            bail!("LoRA export contains no active target modules");
        }

        Ok(Self {
            config: Qwen36AdapterConfig {
                format_version: 1,
                peft_type: "LORA".to_string(),
                task_type: "CAUSAL_LM".to_string(),
                base_model_name_or_path: base_model_path
                    .map(|path| path.to_string_lossy().into_owned())
                    .unwrap_or_else(|| model_name.to_string()),
                rustrain_architecture: architecture.to_string(),
                model_family: "qwen3_hybrid_text".to_string(),
                r: lora_config.rank,
                lora_alpha: lora_config.alpha,
                target_layers,
                target_modules,
                adapter_dtype: "float32".to_string(),
                bias: "none".to_string(),
                inference_mode: true,
            },
            tensors,
        })
    }

    pub fn save(&self, path: &Path) -> Result<PathBuf> {
        let (tensor_path, config_path) = resolve_artifact_paths(path);
        if let Some(parent) = tensor_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let refs = self
            .tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        Tensor::write_safetensors(&refs, &tensor_path)
            .with_context(|| format!("failed to write {}", tensor_path.display()))?;
        let peft_config = PeftAdapterConfig {
            peft_type: "LORA",
            task_type: "CAUSAL_LM",
            base_model_name_or_path: &self.config.base_model_name_or_path,
            r: self.config.r,
            lora_alpha: self.config.lora_alpha,
            lora_dropout: 0.0,
            target_modules: &self.config.target_modules,
            layers_to_transform: &self.config.target_layers,
            layers_pattern: "layers",
            bias: "none",
            fan_in_fan_out: false,
            inference_mode: self.config.inference_mode,
        };
        fs::write(&config_path, serde_json::to_vec_pretty(&peft_config)?)
            .with_context(|| format!("failed to write {}", config_path.display()))?;
        let rustrain_path = rustrain_config_path(&config_path);
        fs::write(&rustrain_path, serde_json::to_vec_pretty(&self.config)?)
            .with_context(|| format!("failed to write {}", rustrain_path.display()))?;
        Ok(tensor_path)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let (tensor_path, config_path) = resolve_artifact_paths(path);
        let rustrain_path = rustrain_config_path(&config_path);
        let metadata_path = if rustrain_path.exists() {
            &rustrain_path
        } else {
            // Backward compatibility with the first named artifact draft,
            // which stored rustrain metadata directly in adapter_config.json.
            &config_path
        };
        let config: Qwen36AdapterConfig = if rustrain_path.exists() {
            serde_json::from_slice(
                &fs::read(metadata_path)
                    .with_context(|| format!("failed to read {}", metadata_path.display()))?,
            )
            .with_context(|| format!("failed to parse {}", metadata_path.display()))?
        } else {
            // Standard PEFT exports do not carry rustrain-specific fields.
            // Preserve their canonical metadata and fill only the native
            // runtime fields that are not part of the PEFT schema.
            let peft: PeftAdapterConfigOwned = serde_json::from_slice(
                &fs::read(&config_path)
                    .with_context(|| format!("failed to read {}", config_path.display()))?,
            )
            .with_context(|| format!("failed to parse {}", config_path.display()))?;
            if peft.target_modules.is_empty() {
                bail!("PEFT adapter_config.json has no target_modules");
            }
            Qwen36AdapterConfig {
                format_version: 1,
                peft_type: peft.peft_type,
                task_type: peft.task_type,
                base_model_name_or_path: peft.base_model_name_or_path,
                rustrain_architecture: "qwen3_hybrid_lora_sft".to_string(),
                model_family: "qwen3_hybrid_text".to_string(),
                r: peft.r,
                lora_alpha: peft.lora_alpha,
                target_layers: peft.layers_to_transform,
                target_modules: peft.target_modules,
                adapter_dtype: "float32".to_string(),
                bias: peft.bias,
                inference_mode: peft.inference_mode,
            }
        };
        if config.format_version != 1 || config.peft_type != "LORA" {
            bail!(
                "unsupported adapter format version/type: {}/{}",
                config.format_version,
                config.peft_type
            );
        }
        if config.r <= 0 || config.lora_alpha <= 0.0 {
            bail!("adapter rank and alpha must be positive");
        }

        let tensors = Tensor::read_safetensors(&tensor_path)
            .with_context(|| format!("failed to read {}", tensor_path.display()))?
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        validate_adapter_tensors(&config, &tensors)?;
        Ok(Self { config, tensors })
    }

    /// Convert the pre-registry positional export into the named v1 format.
    /// No metadata is inferred: callers must provide the runtime target
    /// contract that defined the positional slots.
    pub fn load_legacy(
        path: &Path,
        model_name: &str,
        architecture: &str,
        runtime_config: &Qwen36RuntimeConfig,
        lora_config: &Qwen36LoraConfig,
    ) -> Result<Self> {
        let positional = Tensor::read_safetensors(path)
            .with_context(|| format!("failed to read legacy adapter {}", path.display()))?
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let slots = native_lora_slots(runtime_config, lora_config);
        let active_slots = slots.iter().filter(|slot| slot.active).collect::<Vec<_>>();
        let a_count = positional
            .keys()
            .filter(|key| key.starts_with("lora_a_"))
            .count();
        let b_count = positional
            .keys()
            .filter(|key| key.starts_with("lora_b_"))
            .count();
        if a_count != b_count || (a_count != slots.len() && a_count != active_slots.len()) {
            bail!(
                "legacy adapter slot count {a_count}/{b_count} does not match fixed layout {} or compact target layout {}",
                slots.len(),
                active_slots.len()
            );
        }
        let compact = a_count == active_slots.len() && a_count != slots.len();
        let mut exported = Vec::with_capacity(slots.len());
        let mut compact_index = 0usize;
        for slot in &slots {
            let source_index = if compact {
                if !slot.active {
                    let placeholder = Tensor::zeros([], (Kind::Float, tch::Device::Cpu));
                    exported.push((placeholder.shallow_clone(), placeholder));
                    continue;
                }
                let index = compact_index;
                compact_index += 1;
                index
            } else {
                slot.index
            };
            let a_name = format!("lora_a_{source_index}");
            let b_name = format!("lora_b_{source_index}");
            let a = positional
                .get(&a_name)
                .with_context(|| format!("legacy adapter missing {a_name}"))?
                .shallow_clone();
            let b = positional
                .get(&b_name)
                .with_context(|| format!("legacy adapter missing {b_name}"))?
                .shallow_clone();
            exported.push((a, b));
        }
        Self::from_native_exports(
            model_name,
            architecture,
            None,
            runtime_config,
            lora_config,
            exported,
        )
    }
}

fn validate_adapter_tensors(
    config: &Qwen36AdapterConfig,
    tensors: &BTreeMap<String, Tensor>,
) -> Result<()> {
    if tensors.is_empty() || tensors.len() % 2 != 0 {
        bail!("adapter must contain paired LoRA A/B tensors");
    }
    let mut a_count = 0usize;
    for (name, a) in tensors {
        let Some(prefix) = name.strip_suffix(".lora_A.weight") else {
            if name.ends_with(".lora_B.weight") {
                let prefix = name.trim_end_matches(".lora_B.weight");
                if !tensors.contains_key(&format!("{prefix}.lora_A.weight")) {
                    bail!("missing paired tensor {prefix}.lora_A.weight");
                }
                continue;
            }
            bail!("unexpected adapter tensor name: {name}");
        };
        a_count += 1;
        let b_name = format!("{prefix}.lora_B.weight");
        let b = tensors
            .get(&b_name)
            .with_context(|| format!("missing paired tensor {b_name}"))?;
        let grouped_expert = prefix.contains(".mlp.experts.");
        if grouped_expert {
            if a.dim() != 3 || b.dim() != 3 {
                bail!("routed expert adapter tensors must be rank-3: {name}, {b_name}");
            }
            if a.size()[0] != b.size()[0] || a.size()[1] != config.r || b.size()[2] != config.r {
                bail!(
                    "routed expert adapter shape/rank mismatch for {prefix}: expected rank {}",
                    config.r
                );
            }
            if a.size()[2] <= 0 || b.size()[1] <= 0 {
                bail!("routed expert adapter has an empty feature dimension: {prefix}");
            }
        } else {
            if a.dim() != 2 || b.dim() != 2 {
                bail!("adapter tensors must be matrices: {name}, {b_name}");
            }
            if a.size()[0] != config.r || b.size()[1] != config.r {
                bail!("adapter rank mismatch for {prefix}: expected {}", config.r);
            }
            if a.size()[1] <= 0 || b.size()[0] <= 0 {
                bail!("adapter tensor has an empty feature dimension: {prefix}");
            }
        }
    }
    if a_count == 0 {
        bail!("adapter must contain at least one LoRA A tensor");
    }
    Ok(())
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
                    Qwen36LoraTargetModule::QProj => {
                        format!("{layer_prefix}.self_attn.q_proj.weight")
                    }
                    Qwen36LoraTargetModule::KProj => {
                        format!("{layer_prefix}.self_attn.k_proj.weight")
                    }
                    Qwen36LoraTargetModule::VProj => {
                        format!("{layer_prefix}.self_attn.v_proj.weight")
                    }
                    Qwen36LoraTargetModule::OProj => {
                        format!("{layer_prefix}.self_attn.o_proj.weight")
                    }
                    Qwen36LoraTargetModule::InProjQkv => {
                        format!("{layer_prefix}.linear_attn.in_proj_qkv.weight")
                    }
                    Qwen36LoraTargetModule::InProjZ => {
                        format!("{layer_prefix}.linear_attn.in_proj_z.weight")
                    }
                    Qwen36LoraTargetModule::InProjA => {
                        format!("{layer_prefix}.linear_attn.in_proj_a.weight")
                    }
                    Qwen36LoraTargetModule::InProjB => {
                        format!("{layer_prefix}.linear_attn.in_proj_b.weight")
                    }
                    Qwen36LoraTargetModule::OutProj => {
                        format!("{layer_prefix}.linear_attn.out_proj.weight")
                    }
                    Qwen36LoraTargetModule::GateProj => {
                        format!("{layer_prefix}.mlp.gate_proj.weight")
                    }
                    Qwen36LoraTargetModule::UpProj => {
                        format!("{layer_prefix}.mlp.up_proj.weight")
                    }
                    Qwen36LoraTargetModule::DownProj => {
                        format!("{layer_prefix}.mlp.down_proj.weight")
                    }
                    Qwen36LoraTargetModule::SharedGateProj => {
                        format!("{layer_prefix}.mlp.shared_expert.gate_proj.weight")
                    }
                    Qwen36LoraTargetModule::SharedUpProj => {
                        format!("{layer_prefix}.mlp.shared_expert.up_proj.weight")
                    }
                    Qwen36LoraTargetModule::SharedDownProj => {
                        format!("{layer_prefix}.mlp.shared_expert.down_proj.weight")
                    }
                    Qwen36LoraTargetModule::ExpertsGateUpProj => {
                        format!("{layer_prefix}.mlp.experts.gate_up_proj")
                    }
                    Qwen36LoraTargetModule::ExpertsDownProj => {
                        format!("{layer_prefix}.mlp.experts.down_proj")
                    }
                };

                let base_weight = match weights.get(&weight_name) {
                    Some(w) => w,
                    None => {
                        // Skip modules that don't exist for this layer type
                        // (e.g., q_proj doesn't exist for linear attention layers)
                        tracing::debug!(
                            "skipping LoRA target {weight_name} — not found (layer {layer_idx} may be different attention type)"
                        );
                        continue;
                    }
                };

                let name = module.suffix().replace('.', "_");
                let scale = 1.0 / (lora_config.rank as f64).sqrt();
                let (lora_a, lora_b) = if base_weight.dim() == 3 {
                    let experts = base_weight.size()[0];
                    let out_features = base_weight.size()[1];
                    let in_features = base_weight.size()[2];
                    (
                        p.randn(
                            &format!("lora_a_{layer_idx}_{name}"),
                            &[experts, lora_config.rank, in_features],
                            0.0,
                            scale,
                        ),
                        p.zeros(
                            &format!("lora_b_{layer_idx}_{name}"),
                            &[experts, out_features, lora_config.rank],
                        ),
                    )
                } else {
                    let out_features = base_weight.size()[0];
                    let in_features = base_weight.size()[1];
                    (
                        p.randn(
                            &format!("lora_a_{layer_idx}_{name}"),
                            &[lora_config.rank, in_features],
                            0.0,
                            scale,
                        ),
                        p.zeros(
                            &format!("lora_b_{layer_idx}_{name}"),
                            &[out_features, lora_config.rank],
                        ),
                    )
                };
                adapters.insert((layer_idx, module), (lora_a, lora_b));
            }
        }

        Ok(Self {
            config: lora_config,
            var_store,
            adapters,
        })
    }

    pub fn trainable_variables(&self) -> Vec<Tensor> {
        self.var_store.trainable_variables()
    }

    pub fn adapter_tensors(
        &self,
        layer: usize,
        module: Qwen36LoraTargetModule,
    ) -> Option<(Tensor, Tensor)> {
        let (a, b) = self.adapters.get(&(layer, module))?;
        Some((a.shallow_clone(), b.shallow_clone()))
    }

    /// Get references to the actual VarStore adapter tensors (for training backward).
    pub fn adapter_ref(
        &self,
        layer: usize,
        module: Qwen36LoraTargetModule,
    ) -> Option<(&Tensor, &Tensor)> {
        let (a, b) = self.adapters.get(&(layer, module))?;
        Some((a, b))
    }

    pub fn scaling(&self) -> f64 {
        self.config.alpha / self.config.rank as f64
    }

    pub fn trainable_param_count(&self) -> usize {
        self.trainable_variables()
            .iter()
            .map(|v| v.numel() as usize)
            .sum()
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
            let t = tensor
                .to_device(tch::Device::Cpu)
                .contiguous()
                .to_kind(Kind::Float);
            let shape: Vec<i64> = t.size().iter().copied().map(|d| d).collect();
            let t_flat = t.reshape([-1]);
            let data: Vec<f32> = Vec::<f32>::try_from(&t_flat)?;
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes.len() as u64],
                }),
            );
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

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime(prefix: &str, layer_types: Vec<LayerType>) -> Qwen36RuntimeConfig {
        Qwen36RuntimeConfig {
            num_hidden_layers: layer_types.len(),
            hidden_size: 16,
            vocab_size: 32,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            hidden_act: "silu".into(),
            layer_types,
            full_attention_interval: 4,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 8,
            attention_bias: false,
            attn_output_gate: false,
            rope_theta: 1e6,
            partial_rotary_factor: 1.0,
            mrope_interleaved: false,
            mrope_section: vec![],
            linear_num_key_heads: 2,
            linear_key_head_dim: 8,
            linear_num_value_heads: 2,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            mamba_ssm_dtype: "float32".into(),
            is_moe: false,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            router_aux_loss_coef: 0.0,
            intermediate_size: 32,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            has_vision: prefix.contains("language_model"),
            vision_depth: 0,
            vision_hidden_size: 0,
            vision_num_heads: 0,
            vision_patch_size: 0,
            vision_spatial_merge_size: 0,
            vision_temporal_patch_size: 0,
            vision_out_hidden_size: 0,
            weight_prefix: prefix.into(),
        }
    }

    #[test]
    fn native_slots_match_cpp_projection_order() {
        let config = runtime(
            "model.",
            vec![LayerType::FullAttention, LayerType::LinearAttention],
        );
        let lora = Qwen36LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_layers: vec![],
            target_modules: vec![],
        };
        let slots = native_lora_slots(&config, &lora);
        let modules = slots
            .iter()
            .map(|slot| slot.module.cpp_name())
            .collect::<Vec<_>>();
        assert_eq!(
            modules,
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_a",
                "in_proj_b",
                "out_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        );
    }

    #[test]
    fn parses_text_and_multimodal_peft_names() {
        let text = runtime("model.", vec![LayerType::FullAttention]);
        assert_eq!(
            parse_adapter_tensor_name(
                &text,
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            )
            .unwrap(),
            (0, Qwen36LoraTargetModule::QProj, false)
        );
        let multimodal = runtime("model.language_model.", vec![LayerType::LinearAttention]);
        assert_eq!(
            parse_adapter_tensor_name(
                &multimodal,
                "base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_B.weight"
            )
            .unwrap(),
            (0, Qwen36LoraTargetModule::InProjQkv, true)
        );
    }

    #[test]
    fn rejects_unpaired_adapter_tensor() {
        let config = Qwen36AdapterConfig {
            format_version: 1,
            peft_type: "LORA".into(),
            task_type: "CAUSAL_LM".into(),
            base_model_name_or_path: "Qwen/test".into(),
            rustrain_architecture: "qwen3.6".into(),
            model_family: "qwen3_hybrid_text".into(),
            r: 2,
            lora_alpha: 4.0,
            target_layers: vec![0],
            target_modules: vec!["q_proj".into()],
            adapter_dtype: "float32".into(),
            bias: "none".into(),
            inference_mode: true,
        };
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".into(),
            Tensor::zeros([4, 2], (Kind::Float, tch::Device::Cpu)),
        );
        assert!(validate_adapter_tensors(&config, &tensors).is_err());
    }
}
