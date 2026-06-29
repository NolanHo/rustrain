//! Qwen3.6 training session — single-GPU LoRA SFT + EP4 distributed training.

use std::collections::{BTreeMap, HashSet};
use std::env;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use tch::{nn, Kind, Reduction, Tensor, no_grad};
use tracing::info;

use crate::config::{read_qwen36_runtime_config, resolve_qwen36_model_path, Qwen36RuntimeConfig, LayerType};
use crate::lora::{Qwen36LoraConfig, Qwen36LoraRegistry, Qwen36LoraTargetModule};
use crate::model;
use crate::sft::SftDataset;
use rustrain_core::runtime::{Config, RunPaths, LoraConfig};
use rustrain_checkpoint::safetensors::{read_safetensors_dir_filtered};

// Global weight storage for checkpoint closures — uses thread_local to avoid Send+Sync requirements
// (PyTorch autograd::Function::backward runs on the same thread as forward)
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

thread_local! {
    static CHECKPOINT_FNS: RefCell<HashMap<usize, Box<dyn Fn(&Tensor) -> Tensor>>> = RefCell::new(HashMap::new());
}

// ──────────────────────────────────────────────────────────────────────
// EP Shard
// ──────────────────────────────────────────────────────────────────────

pub struct EpShard {
    pub rank: usize,
    pub world_size: usize,
    pub experts_per_rank: usize,
    pub expert_start: usize,
    pub local_expert_indices: Vec<usize>,
}

impl EpShard {
    pub fn new(rank: usize, world_size: usize, num_experts: usize) -> Self {
        assert!(num_experts % world_size == 0, "num_experts {num_experts} not divisible by world_size {world_size}");
        let epr = num_experts / world_size;
        let start = rank * epr;
        Self {
            rank,
            world_size,
            experts_per_rank: epr,
            expert_start: start,
            local_expert_indices: (start..start + epr).collect(),
        }
    }

    pub fn owns_expert(&self, global_idx: usize) -> bool {
        global_idx >= self.expert_start && global_idx < self.expert_start + self.experts_per_rank
    }
}

// ──────────────────────────────────────────────────────────────────────
// Summary
// ──────────────────────────────────────────────────────────────────────

pub struct Qwen36LoraSftSummary {
    pub adapter_output: String,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub trainable_params: usize,
}

// ──────────────────────────────────────────────────────────────────────
// Weight name helpers
// ──────────────────────────────────────────────────────────────────────

fn parse_env_usize(key: &str) -> Result<usize> {
    env::var(key)
        .with_context(|| format!("{key} not set"))
        .and_then(|v| v.parse::<usize>().with_context(|| format!("invalid {key}: {v}")))
}

fn lora_config_from_config(config: &Config) -> Result<Qwen36LoraConfig> {
    let lora = config.lora.as_ref().ok_or_else(|| anyhow!("[lora] section required"))?;
    let target_layers: Vec<usize> = lora.target_layers.clone();
    let target_modules: Vec<Qwen36LoraTargetModule> = lora
        .target_modules
        .iter()
        .map(|m| Qwen36LoraTargetModule::parse(m))
        .collect::<Result<Vec<_>>>()?;
    Ok(Qwen36LoraConfig {
        rank: lora.rank,
        alpha: lora.alpha,
        target_layers,
        target_modules,
    })
}

/// Build the set of weight names needed for training.
/// For EP mode, only load local expert slice.
fn build_needed_weights(
    config: &Qwen36RuntimeConfig,
    lora_config: &Qwen36LoraConfig,
    ep_shard: Option<&EpShard>,
) -> HashSet<String> {
    let p = &config.weight_prefix;
    let mut needed = HashSet::new();

    // Embed, norm, lm_head
    needed.insert(format!("{p}embed_tokens.weight"));
    needed.insert(format!("{p}norm.weight"));
    needed.insert("lm_head.weight".to_string());

    // Per-layer weights for ALL layers (not just LoRA targets)
    for layer in 0..config.num_hidden_layers {
        let lp = format!("{p}layers.{layer}");
        needed.insert(format!("{lp}.input_layernorm.weight"));
        needed.insert(format!("{lp}.post_attention_layernorm.weight"));

        // Attention weights (full or linear)
        match config.layer_types[layer] {
            LayerType::FullAttention => {
                for w in &["q_proj", "q_norm", "k_proj", "k_norm", "v_proj", "o_proj"] {
                    needed.insert(format!("{lp}.self_attn.{w}.weight"));
                }
            }
            LayerType::LinearAttention => {
                needed.insert(format!("{lp}.linear_attn.A_log"));
                needed.insert(format!("{lp}.linear_attn.conv1d.weight"));
                needed.insert(format!("{lp}.linear_attn.dt_bias"));
                needed.insert(format!("{lp}.linear_attn.norm.weight"));
                for w in &["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"] {
                    needed.insert(format!("{lp}.linear_attn.{w}.weight"));
                }
            }
        }

        // MoE weights
        needed.insert(format!("{lp}.mlp.gate.weight"));
        needed.insert(format!("{lp}.mlp.shared_expert_gate.weight"));
        needed.insert(format!("{lp}.mlp.shared_expert.gate_proj.weight"));
        needed.insert(format!("{lp}.mlp.shared_expert.up_proj.weight"));
        needed.insert(format!("{lp}.mlp.shared_expert.down_proj.weight"));

        // Routed experts — all for single-GPU, local slice for EP
        if let Some(shard) = ep_shard {
            // Fused expert tensors are 3D [num_experts, ...], loaded as a whole
            // In EP mode we still need the full tensor, then select local experts
            // (safetensors stores them as single fused tensor, can't shard at load time)
            needed.insert(format!("{lp}.mlp.experts.gate_up_proj"));
            needed.insert(format!("{lp}.mlp.experts.down_proj"));
        } else {
            needed.insert(format!("{lp}.mlp.experts.gate_up_proj"));
            needed.insert(format!("{lp}.mlp.experts.down_proj"));
        }
    }

    // Vision encoder (for multimodal)
    if config.has_vision {
        for name in crate::vision::VisionWeights::weight_names(config) {
            needed.insert(name);
        }
    }

    needed
}

// ──────────────────────────────────────────────────────────────────────
// Forward with LoRA
// ──────────────────────────────────────────────────────────────────────

/// Apply LoRA delta to a weight tensor: `W_new = W + scaling * B @ A`
/// 
/// Megatron mixed-precision pattern:
/// - LoRA A/B are FP32 master weights (VarStore default, for optimizer precision)
/// - Forward casts A/B to model dtype (BF16) for computation
/// - Backward: autograd flows gradient through the cast back to FP32 leaf
/// - Result: W + scaling * (B_bf16 @ A_bf16), all in BF16
fn lora_weight_delta(
    base_weight: &Tensor,
    lora_a: &Tensor,
    lora_b: &Tensor,
    scaling: f64,
) -> Tensor {
    let kind = base_weight.kind();
    // Cast FP32 master weights to compute dtype (BF16) for forward
    let a = lora_a.to_kind(kind);  // [rank, in_features] — BF16
    let b = lora_b.to_kind(kind);  // [out_features, rank] — BF16
    let delta = b.matmul(&a);       // [out_features, in_features] — BF16
    // Scalar multiply: keep BF16 (don't let it promote to FP32)
    let scaled_delta = (delta * scaling).to_kind(kind);
    base_weight + scaled_delta  // BF16 + BF16 = BF16
}

/// Forward pass with LoRA adapters applied to attention + shared expert.
/// Returns (logits, pre_lm_head_hidden) — hidden needed for MTP loss.
fn forward_with_lora(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &Qwen36RuntimeConfig,
    registry: &Qwen36LoraRegistry,
    ep_shard: Option<&EpShard>,
    compute_kind: Kind,
) -> (Tensor, Tensor) {
    let embed_prefix = format!("{}embed_tokens.weight", config.weight_prefix);
    let embed_tokens = weights.get(&embed_prefix).unwrap().to_kind(compute_kind);
    let final_norm = weights.get(&format!("{}norm.weight", config.weight_prefix)).unwrap().to_kind(compute_kind);

    let mut hidden = Tensor::embedding(&embed_tokens, &input_ids, -1, false, false);
    // Set requires_grad so C++ autograd::Function builds a graph through checkpoint groups
    hidden = hidden.detach().set_requires_grad(true);
    let scaling = registry.scaling();

    for layer_index in 0..config.num_hidden_layers {
        let is_target = registry.config.target_layers.contains(&layer_index);
        let prefix = format!("{}layers.{layer_index}", config.weight_prefix);

        // Norms
        let input_norm = weights.get(&format!("{prefix}.input_layernorm.weight")).unwrap().to_kind(compute_kind);
        let post_norm = weights.get(&format!("{prefix}.post_attention_layernorm.weight")).unwrap().to_kind(compute_kind);

        // Attention input
        let attn_input = model::rms_norm(&hidden, &input_norm, config.rms_norm_eps).to_kind(compute_kind);

        // Attention with LoRA
        let attn_output = match config.layer_types[layer_index] {
            LayerType::FullAttention => {
                let mut w = model::FullAttnWeights {
                    q_proj: weights.get(&format!("{prefix}.self_attn.q_proj.weight")).unwrap().to_kind(compute_kind),
                    q_norm: weights.get(&format!("{prefix}.self_attn.q_norm.weight")).unwrap().to_kind(compute_kind),
                    k_proj: weights.get(&format!("{prefix}.self_attn.k_proj.weight")).unwrap().to_kind(compute_kind),
                    k_norm: weights.get(&format!("{prefix}.self_attn.k_norm.weight")).unwrap().to_kind(compute_kind),
                    v_proj: weights.get(&format!("{prefix}.self_attn.v_proj.weight")).unwrap().to_kind(compute_kind),
                    o_proj: weights.get(&format!("{prefix}.self_attn.o_proj.weight")).unwrap().to_kind(compute_kind),
                };

                if is_target {
                    // Apply LoRA to all target modules (q_proj, k_proj, v_proj, o_proj)
                    for &module in &registry.config.target_modules {
                        let base = match module {
                            Qwen36LoraTargetModule::QProj => &w.q_proj,
                            Qwen36LoraTargetModule::KProj => &w.k_proj,
                            Qwen36LoraTargetModule::VProj => &w.v_proj,
                            Qwen36LoraTargetModule::OProj => &w.o_proj,
                            _ => continue, // non-full-attn modules handled below
                        };
                        let base_owned = base.shallow_clone();
                        if let Some((a, b)) = registry.adapter_ref(layer_index, module) {
                            let w_delta = lora_weight_delta(&base_owned, &a, &b, scaling);
                            match module {
                                Qwen36LoraTargetModule::QProj => w.q_proj = w_delta,
                                Qwen36LoraTargetModule::KProj => w.k_proj = w_delta,
                                Qwen36LoraTargetModule::VProj => w.v_proj = w_delta,
                                Qwen36LoraTargetModule::OProj => w.o_proj = w_delta,
                                _ => {}
                            }
                        }
                    }
                }

                model::full_attention(&attn_input, &w, config, compute_kind)
            }
            LayerType::LinearAttention => {
                let mut w = model::LinearAttnWeights {
                    in_proj_qkv: weights.get(&format!("{prefix}.linear_attn.in_proj_qkv.weight")).unwrap().to_kind(compute_kind),
                    in_proj_z: weights.get(&format!("{prefix}.linear_attn.in_proj_z.weight")).unwrap().to_kind(compute_kind),
                    in_proj_a: weights.get(&format!("{prefix}.linear_attn.in_proj_a.weight")).unwrap().to_kind(compute_kind),
                    in_proj_b: weights.get(&format!("{prefix}.linear_attn.in_proj_b.weight")).unwrap().to_kind(compute_kind),
                    a_log: weights.get(&format!("{prefix}.linear_attn.A_log")).unwrap().to_kind(compute_kind),
                    dt_bias: weights.get(&format!("{prefix}.linear_attn.dt_bias")).unwrap().to_kind(compute_kind),
                    conv1d_weight: weights.get(&format!("{prefix}.linear_attn.conv1d.weight")).unwrap().to_kind(compute_kind),
                    norm: weights.get(&format!("{prefix}.linear_attn.norm.weight")).unwrap().to_kind(compute_kind),
                    out_proj: weights.get(&format!("{prefix}.linear_attn.out_proj.weight")).unwrap().to_kind(compute_kind),
                };

                if is_target {
                    // Apply LoRA to linear attention projections
                    for &module in &registry.config.target_modules {
                        let base = match module {
                            Qwen36LoraTargetModule::InProjQkv => &w.in_proj_qkv,
                            Qwen36LoraTargetModule::InProjZ => &w.in_proj_z,
                            Qwen36LoraTargetModule::OutProj => &w.out_proj,
                            _ => continue,
                        };
                        let base_owned = base.shallow_clone();
                        if let Some((a, b)) = registry.adapter_ref(layer_index, module) {
                            let w_delta = lora_weight_delta(&base_owned, &a, &b, scaling);
                            match module {
                                Qwen36LoraTargetModule::InProjQkv => w.in_proj_qkv = w_delta,
                                Qwen36LoraTargetModule::InProjZ => w.in_proj_z = w_delta,
                                Qwen36LoraTargetModule::OutProj => w.out_proj = w_delta,
                                _ => {}
                            }
                        }
                    }
                }

                model::linear_attention(&attn_input, &w, config, compute_kind)
            }
        };

        let after_attention = &hidden + &attn_output;

        // MoE with optional EP sharding
        let moe_input = model::rms_norm(&after_attention, &post_norm, config.rms_norm_eps).to_kind(compute_kind);

        let moe_output = if let Some(shard) = ep_shard {
            // EP mode: load sharded experts, compute routed-only + shared, caller all-reduces routed
            let moe = model::MoeWeights::load_ep(weights, &prefix, compute_kind, shard.expert_start, shard.experts_per_rank).unwrap();
            let routed = model::moe_routed_only_ep(&moe_input, &moe, config, compute_kind);
            let shared = model::moe_shared_only(&moe_input, &moe, compute_kind);
            &routed + &shared
        } else {
            let moe = model::MoeWeights::load(weights, &prefix, compute_kind).unwrap();
            model::moe_forward(&moe_input, &moe, config, compute_kind)
        };

        hidden = (&after_attention + &moe_output).to_kind(compute_kind);
    }

    let hidden_raw = hidden.shallow_clone();  // pre-norm hidden for MTP
    let hidden_normed = model::rms_norm(&hidden, &final_norm, config.rms_norm_eps).to_kind(compute_kind);
    let lm_head = if config.tie_word_embeddings {
        embed_tokens.shallow_clone()
    } else {
        weights.get("lm_head.weight").unwrap().to_kind(compute_kind)
    };

    let logits = hidden_normed.linear::<&Tensor>(&lm_head, None);
    (logits, hidden_raw)  // MTP needs pre-norm hidden
}

/// Forward with gradient checkpointing via C++ autograd::Function.
fn forward_with_lora_checkpointed(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &Qwen36RuntimeConfig,
    registry: &Qwen36LoraRegistry,
    ep_shard: Option<&EpShard>,
    compute_kind: Kind,
    group_size: usize,
) -> (Tensor, Tensor) {
    let embed_prefix = format!("{}embed_tokens.weight", config.weight_prefix);
    let embed_tokens = weights.get(&embed_prefix).unwrap().to_kind(compute_kind);
    let final_norm = weights.get(&format!("{}norm.weight", config.weight_prefix)).unwrap().to_kind(compute_kind);

    let mut hidden = Tensor::embedding(&embed_tokens, &input_ids, -1, false, false);
    // Set requires_grad so C++ autograd::Function builds a graph through checkpoint groups
    hidden = hidden.detach().set_requires_grad(true);

    let ep_count = ep_shard.map(|s| s.experts_per_rank).unwrap_or(256);
    let scaling = registry.scaling();

    let adapter_map: BTreeMap<(usize, Qwen36LoraTargetModule), (Tensor, Tensor)> = registry
        .adapters.iter().map(|(k, v)| (*k, (v.0.shallow_clone(), v.1.shallow_clone()))).collect();

    for group_start in (0..config.num_hidden_layers).step_by(group_size) {
        let group_end = (group_start + group_size).min(config.num_hidden_layers);
        // Clone weights — Tensor doesn't impl Clone, so we manually shallow_clone each entry
        let w: BTreeMap<String, Tensor> = weights.iter().map(|(k, v)| (k.clone(), v.shallow_clone())).collect();
        let a: BTreeMap<(usize, Qwen36LoraTargetModule), (Tensor, Tensor)> = adapter_map.iter().map(|(k, v)| (*k, (v.0.shallow_clone(), v.1.shallow_clone()))).collect();
        let c = config.clone();
        let lc = registry.config.clone();
        let gs = group_start;
        let ge = group_end;
        let ek = compute_kind;
        let ep = ep_count;
        let sc = scaling;

        let checkpoint_fn = move |input: &Tensor| -> Tensor {
            let weights = &w;
            let adapters = &a;
            let cfg = &c;
            let lcfg = &lc;
            let mut h = input.shallow_clone();
            for layer_index in gs..ge {
                let is_target = lcfg.target_layers.contains(&layer_index);
                let prefix = format!("{}layers.{layer_index}", cfg.weight_prefix);
                let input_norm = weights.get(&format!("{prefix}.input_layernorm.weight")).unwrap().to_kind(ek);
                let post_norm = weights.get(&format!("{prefix}.post_attention_layernorm.weight")).unwrap().to_kind(ek);
                let attn_input = model::rms_norm(&h, &input_norm, cfg.rms_norm_eps).to_kind(ek);

                let attn_output = match cfg.layer_types[layer_index] {
                    LayerType::FullAttention => {
                        let mut fw = model::FullAttnWeights {
                            q_proj: weights.get(&format!("{prefix}.self_attn.q_proj.weight")).unwrap().to_kind(ek),
                            q_norm: weights.get(&format!("{prefix}.self_attn.q_norm.weight")).unwrap().to_kind(ek),
                            k_proj: weights.get(&format!("{prefix}.self_attn.k_proj.weight")).unwrap().to_kind(ek),
                            k_norm: weights.get(&format!("{prefix}.self_attn.k_norm.weight")).unwrap().to_kind(ek),
                            v_proj: weights.get(&format!("{prefix}.self_attn.v_proj.weight")).unwrap().to_kind(ek),
                            o_proj: weights.get(&format!("{prefix}.self_attn.o_proj.weight")).unwrap().to_kind(ek),
                        };
                        if is_target {
                            for &module in &lcfg.target_modules {
                                let base = match module {
                                    Qwen36LoraTargetModule::QProj => &fw.q_proj,
                                    Qwen36LoraTargetModule::KProj => &fw.k_proj,
                                    Qwen36LoraTargetModule::VProj => &fw.v_proj,
                                    Qwen36LoraTargetModule::OProj => &fw.o_proj,
                                    _ => continue,
                                };
                                let base_owned = base.shallow_clone();
                                if let Some((la, lb)) = adapters.get(&(layer_index, module)) {
                                    let delta = lb.to_kind(base_owned.kind()).matmul(&la.to_kind(base_owned.kind()));
                                    let w_new = &base_owned + (delta * sc).to_kind(base_owned.kind());
                                    match module {
                                        Qwen36LoraTargetModule::QProj => fw.q_proj = w_new,
                                        Qwen36LoraTargetModule::KProj => fw.k_proj = w_new,
                                        Qwen36LoraTargetModule::VProj => fw.v_proj = w_new,
                                        Qwen36LoraTargetModule::OProj => fw.o_proj = w_new,
                                        _ => {}
                                    }
                                }
                            }
                        }
                        model::full_attention(&attn_input, &fw, cfg, ek)
                    }
                    LayerType::LinearAttention => {
                        let lw = model::LinearAttnWeights {
                            in_proj_qkv: weights.get(&format!("{prefix}.linear_attn.in_proj_qkv.weight")).unwrap().to_kind(ek),
                            in_proj_z: weights.get(&format!("{prefix}.linear_attn.in_proj_z.weight")).unwrap().to_kind(ek),
                            in_proj_a: weights.get(&format!("{prefix}.linear_attn.in_proj_a.weight")).unwrap().to_kind(ek),
                            in_proj_b: weights.get(&format!("{prefix}.linear_attn.in_proj_b.weight")).unwrap().to_kind(ek),
                            a_log: weights.get(&format!("{prefix}.linear_attn.A_log")).unwrap().to_kind(ek),
                            dt_bias: weights.get(&format!("{prefix}.linear_attn.dt_bias")).unwrap().to_kind(ek),
                            conv1d_weight: weights.get(&format!("{prefix}.linear_attn.conv1d.weight")).unwrap().to_kind(ek),
                            norm: weights.get(&format!("{prefix}.linear_attn.norm.weight")).unwrap().to_kind(ek),
                            out_proj: weights.get(&format!("{prefix}.linear_attn.out_proj.weight")).unwrap().to_kind(ek),
                        };
                        model::linear_attention(&attn_input, &lw, cfg, ek)
                    }
                };

                let after_attention = &h + &attn_output;
                let moe_input = model::rms_norm(&after_attention, &post_norm, cfg.rms_norm_eps).to_kind(ek);
                let moe_output = if ep < 256 {
                    let moe = model::MoeWeights::load_ep(weights, &prefix, ek, 0, ep).unwrap();
                    &model::moe_routed_only_ep(&moe_input, &moe, cfg, ek) + &model::moe_shared_only(&moe_input, &moe, ek)
                } else {
                    let moe = model::MoeWeights::load(weights, &prefix, ek).unwrap();
                    model::moe_forward(&moe_input, &moe, cfg, ek)
                };
                h = (&after_attention + &moe_output).to_kind(ek);
            }
            h
        };

        // Use C++ checkpoint forward for this group, fallback to eager
        hidden = crate::kernel::checkpoint_forward(
            &hidden, &(group_start..group_end).collect::<Vec<_>>(),
            weights, config, registry,
            ep_shard.map(|s| s.expert_start).unwrap_or(0),
            ep_shard.map(|s| s.experts_per_rank).unwrap_or(256),
            compute_kind,
        ).unwrap_or_else(|e| {
            tracing::warn!("C++ checkpoint failed ({e}), using eager forward");
            let mut h = hidden.shallow_clone();
            for layer_index in group_start..group_end {
                h = forward_single_layer(&h, weights, config, registry, ep_shard, compute_kind, layer_index);
            }
            h
        });
    }

    let hidden_normed = model::rms_norm(&hidden, &final_norm, config.rms_norm_eps).to_kind(compute_kind);
    let lm_head = if config.tie_word_embeddings { embed_tokens.shallow_clone() }
        else { weights.get("lm_head.weight").unwrap().to_kind(compute_kind) };
    (hidden_normed.linear::<&Tensor>(&lm_head, None), hidden_normed)
}

/// Forward a single layer (used by checkpointed forward, both in no_grad and with grad).
fn forward_single_layer(
    hidden: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &Qwen36RuntimeConfig,
    registry: &Qwen36LoraRegistry,
    ep_shard: Option<&EpShard>,
    compute_kind: Kind,
    layer_index: usize,
) -> Tensor {
    let is_target = registry.config.target_layers.contains(&layer_index);
    let prefix = format!("{}layers.{layer_index}", config.weight_prefix);
    let scaling = registry.scaling();

    let input_norm = weights.get(&format!("{prefix}.input_layernorm.weight")).unwrap().to_kind(compute_kind);
    let post_norm = weights.get(&format!("{prefix}.post_attention_layernorm.weight")).unwrap().to_kind(compute_kind);

    let attn_input = model::rms_norm(hidden, &input_norm, config.rms_norm_eps).to_kind(compute_kind);

    let attn_output = match config.layer_types[layer_index] {
        LayerType::FullAttention => {
            let mut w = model::FullAttnWeights {
                q_proj: weights.get(&format!("{prefix}.self_attn.q_proj.weight")).unwrap().to_kind(compute_kind),
                q_norm: weights.get(&format!("{prefix}.self_attn.q_norm.weight")).unwrap().to_kind(compute_kind),
                k_proj: weights.get(&format!("{prefix}.self_attn.k_proj.weight")).unwrap().to_kind(compute_kind),
                k_norm: weights.get(&format!("{prefix}.self_attn.k_norm.weight")).unwrap().to_kind(compute_kind),
                v_proj: weights.get(&format!("{prefix}.self_attn.v_proj.weight")).unwrap().to_kind(compute_kind),
                o_proj: weights.get(&format!("{prefix}.self_attn.o_proj.weight")).unwrap().to_kind(compute_kind),
            };

            if is_target {
                for &module in &registry.config.target_modules {
                    let base = match module {
                        Qwen36LoraTargetModule::QProj => &w.q_proj,
                        Qwen36LoraTargetModule::KProj => &w.k_proj,
                        Qwen36LoraTargetModule::VProj => &w.v_proj,
                        Qwen36LoraTargetModule::OProj => &w.o_proj,
                        _ => continue,
                    };
                    let base_owned = base.shallow_clone();
                    if let Some((a, b)) = registry.adapter_ref(layer_index, module) {
                        let w_delta = lora_weight_delta(&base_owned, &a, &b, scaling);
                        match module {
                            Qwen36LoraTargetModule::QProj => w.q_proj = w_delta,
                            Qwen36LoraTargetModule::KProj => w.k_proj = w_delta,
                            Qwen36LoraTargetModule::VProj => w.v_proj = w_delta,
                            Qwen36LoraTargetModule::OProj => w.o_proj = w_delta,
                            _ => {}
                        }
                    }
                }
            }

            model::full_attention(&attn_input, &w, config, compute_kind)
        }
        LayerType::LinearAttention => {
            let mut w = model::LinearAttnWeights {
                in_proj_qkv: weights.get(&format!("{prefix}.linear_attn.in_proj_qkv.weight")).unwrap().to_kind(compute_kind),
                in_proj_z: weights.get(&format!("{prefix}.linear_attn.in_proj_z.weight")).unwrap().to_kind(compute_kind),
                in_proj_a: weights.get(&format!("{prefix}.linear_attn.in_proj_a.weight")).unwrap().to_kind(compute_kind),
                in_proj_b: weights.get(&format!("{prefix}.linear_attn.in_proj_b.weight")).unwrap().to_kind(compute_kind),
                a_log: weights.get(&format!("{prefix}.linear_attn.A_log")).unwrap().to_kind(compute_kind),
                dt_bias: weights.get(&format!("{prefix}.linear_attn.dt_bias")).unwrap().to_kind(compute_kind),
                conv1d_weight: weights.get(&format!("{prefix}.linear_attn.conv1d.weight")).unwrap().to_kind(compute_kind),
                norm: weights.get(&format!("{prefix}.linear_attn.norm.weight")).unwrap().to_kind(compute_kind),
                out_proj: weights.get(&format!("{prefix}.linear_attn.out_proj.weight")).unwrap().to_kind(compute_kind),
            };

            if is_target {
                for &module in &registry.config.target_modules {
                    let base = match module {
                        Qwen36LoraTargetModule::InProjQkv => &w.in_proj_qkv,
                        Qwen36LoraTargetModule::InProjZ => &w.in_proj_z,
                        Qwen36LoraTargetModule::OutProj => &w.out_proj,
                        _ => continue,
                    };
                    let base_owned = base.shallow_clone();
                    if let Some((a, b)) = registry.adapter_ref(layer_index, module) {
                        let w_delta = lora_weight_delta(&base_owned, &a, &b, scaling);
                        match module {
                            Qwen36LoraTargetModule::InProjQkv => w.in_proj_qkv = w_delta,
                            Qwen36LoraTargetModule::InProjZ => w.in_proj_z = w_delta,
                            Qwen36LoraTargetModule::OutProj => w.out_proj = w_delta,
                            _ => {}
                        }
                    }
                }
            }

            model::linear_attention(&attn_input, &w, config, compute_kind)
        }
    };

    let after_attention = hidden + &attn_output;

    let moe_input = model::rms_norm(&after_attention, &post_norm, config.rms_norm_eps).to_kind(compute_kind);

    let moe_output = if let Some(shard) = ep_shard {
        // EP mode: tensors already narrowed in BTreeMap, use expert_start=0
        let moe = model::MoeWeights::load_ep(weights, &prefix, compute_kind, 0, shard.experts_per_rank).unwrap();
        let routed = model::moe_routed_only_ep(&moe_input, &moe, config, compute_kind);
        let shared = model::moe_shared_only(&moe_input, &moe, compute_kind);
        &routed + &shared
    } else {
        let moe = model::MoeWeights::load(weights, &prefix, compute_kind).unwrap();
        model::moe_forward(&moe_input, &moe, config, compute_kind)
    };

    (after_attention + moe_output).to_kind(compute_kind)
}

/// Manual backward through checkpointed layer groups.
/// Recomputes each group's forward WITH grad, backprops, accumulates into LoRA params.
fn manual_backward_checkpointed(
    grad_output: &Tensor,
    group_inputs: &mut [Tensor],
    weights: &BTreeMap<String, Tensor>,
    config: &Qwen36RuntimeConfig,
    registry: &Qwen36LoraRegistry,
    ep_shard: Option<&EpShard>,
    compute_kind: Kind,
    group_size: usize,
) {
    let mut grad = grad_output.shallow_clone();

    for (group_idx, group_input) in group_inputs.iter_mut().enumerate().rev() {
        let group_start = group_idx * group_size;
        let group_end = (group_start + group_size).min(config.num_hidden_layers);

        // Recompute group forward WITH grad (builds graph for LoRA params)
        let group_output = {
            let mut h = group_input.shallow_clone();
            for layer_index in group_start..group_end {
                h = forward_single_layer(&h, weights, config, registry, ep_shard, compute_kind, layer_index);
            }
            h
        };

        // Backprop through this group (non-scalar: multiply by grad, sum, backward)
        let loss = (group_output * &grad).sum(Kind::Float);
        loss.backward();

        // Get gradient w.r.t. group input for the next group
        grad = group_input.grad();
        group_input.zero_grad();
    }
}

// ──────────────────────────────────────────────────────────────────────
// Cross-entropy loss
// ──────────────────────────────────────────────────────────────────────

fn cross_entropy_loss(
    logits: &Tensor,
    target_ids: &Tensor,
    target_mask: &Tensor,
    vocab_size: i64,
) -> Tensor {
    let seq_len = logits.size()[1];
    let shifted_logits = logits.narrow(1, 0, seq_len - 1).reshape([-1, vocab_size]);
    let shifted_targets = target_ids.narrow(1, 1, seq_len - 1).reshape([-1]);
    let shifted_mask = target_mask.narrow(1, 1, seq_len - 1).reshape([-1]);

    let log_probs = shifted_logits.log_softmax(-1, Kind::Float);
    let per_token_loss = log_probs.g_nll_loss::<&Tensor>(&shifted_targets, None, Reduction::None, -100);
    let masked_loss = &per_token_loss * &shifted_mask;
    let total = masked_loss.sum(Kind::Float);
    let count = shifted_mask.sum(Kind::Float).clamp_min(1.0);
    total / count
}

// ──────────────────────────────────────────────────────────────────────
// Entry point: single-GPU LoRA SFT
// ──────────────────────────────────────────────────────────────────────

pub fn train_qwen3_6_lora_sft(
    config: &Config,
    run_paths: &RunPaths,
) -> Result<Qwen36LoraSftSummary> {
    train_impl(config, run_paths, None)
}

// ──────────────────────────────────────────────────────────────────────
// Entry point: EP4 distributed LoRA SFT
// ──────────────────────────────────────────────────────────────────────

pub fn train_qwen3_6_lora_sft_ep(
    config: &Config,
    run_paths: &RunPaths,
) -> Result<Qwen36LoraSftSummary> {
    let rank = parse_env_usize("RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    let ep_shard = Some(EpShard::new(rank, world_size, 256));
    train_impl(config, run_paths, ep_shard)
}

// ──────────────────────────────────────────────────────────────────────
// Core training implementation
// ──────────────────────────────────────────────────────────────────────

fn train_impl(
    config: &Config,
    run_paths: &RunPaths,
    ep_shard: Option<EpShard>,
) -> Result<Qwen36LoraSftSummary> {
    let model_path = config.model.model_path.as_ref()
        .ok_or_else(|| anyhow!("model.model_path required"))?;
    let model_path = resolve_qwen36_model_path(model_path)?;
    let runtime_config = read_qwen36_runtime_config(&model_path)?;
    let lora_config = lora_config_from_config(config)?;
    let device = match config.train.device {
        rustrain_core::runtime::Device::Cuda => {
            // EP mode: use LOCAL_RANK to select the correct GPU
            let local_rank = std::env::var("LOCAL_RANK")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            tch::Device::Cuda(local_rank)
        }
        rustrain_core::runtime::Device::Cpu => tch::Device::Cpu,
    };
    let compute_kind = match config.train.dtype {
        rustrain_core::runtime::DType::Fp16 => Kind::Half,
        rustrain_core::runtime::DType::Bf16 => Kind::BFloat16,
        rustrain_core::runtime::DType::Fp32 => Kind::Float,
    };

    let shard_ref = ep_shard.as_ref();
    let is_ep = shard_ref.is_some();
    let world_size = shard_ref.map(|s| s.world_size).unwrap_or(1);
    let rank = shard_ref.map(|s| s.rank).unwrap_or(0);

    // Build needed weight set
    let needed = build_needed_weights(&runtime_config, &lora_config, shard_ref);

    // Stagger loading for EP to avoid OOM
    if is_ep {
        std::thread::sleep(std::time::Duration::from_secs(rank as u64 * 5));
    }

    info!("loading {} weight tensors from {}", needed.len(), model_path.display());
    let weights = read_safetensors_dir_filtered(&model_path, &needed)?;

    // Move to device — for EP, narrow expert tensors on CPU first to save GPU memory
    let mut weights_gpu = BTreeMap::new();
    if let Some(shard) = shard_ref {
        // EP mode: narrow expert tensors on CPU before transferring to GPU
        for (name, tensor) in &weights {
            // Check if this is an expert tensor that needs narrowing
            let needs_narrow = name.contains(".mlp.experts.gate_up_proj")
                || name.contains(".mlp.experts.down_proj");
            if needs_narrow && tensor.size()[0] == 256 {
                let narrowed = tensor
                    .narrow(0, shard.expert_start as i64, shard.experts_per_rank as i64)
                    .contiguous()
                    .to_device(device)
                    .to_kind(compute_kind);
                weights_gpu.insert(name.clone(), narrowed);
            } else {
                weights_gpu.insert(name.clone(), tensor.to_device(device).to_kind(compute_kind));
            }
        }
        info!("EP{}: narrowed expert tensors to {} experts per rank (on CPU before GPU transfer)", world_size, shard.experts_per_rank);
    } else {
        for (name, tensor) in &weights {
            weights_gpu.insert(name.clone(), tensor.to_device(device).to_kind(compute_kind));
        }
    }

    // Create LoRA registry
    let registry = Qwen36LoraRegistry::new(&weights_gpu, &runtime_config, lora_config, device)?;
    let mut trainable_vars = registry.trainable_variables();
    let trainable_params = registry.trainable_param_count();
    info!("LoRA: {} trainable tensors, {} params", trainable_vars.len(), trainable_params);

    // Load SFT data
    let tokenizer_path = model_path.join("tokenizer.json");
    let data = if let Some(data_config) = &config.data {
        let path = &data_config.paths[0];
        SftDataset::from_jsonl(path, &tokenizer_path, config.model.seq_len)?
    } else {
        bail!("[data] section required for SFT training");
    };

    // Load MTP weights if available
    let mtp_weights = if runtime_config.mtp_num_hidden_layers > 0 {
        let mtp_names = crate::mtp::MtpWeights::weight_names(&runtime_config);
        let mtp_needed: HashSet<String> = mtp_names.into_iter().collect();
        let mtp_tensors = read_safetensors_dir_filtered(&model_path, &mtp_needed)?;
        let mut mtp_gpu = BTreeMap::new();
        for (name, tensor) in &mtp_tensors {
            mtp_gpu.insert(name.clone(), tensor.to_device(device).to_kind(compute_kind));
        }
        // Merge MTP weights into the main weight map
        for (name, tensor) in mtp_gpu {
            weights_gpu.insert(name, tensor);
        }
        Some(crate::mtp::MtpWeights::load(&weights_gpu, &runtime_config, compute_kind)?)
    } else {
        None
    };

    // Get embed_tokens and lm_head for MTP
    let embed_tokens_ref = weights_gpu.get(&format!("{}embed_tokens.weight", runtime_config.weight_prefix))
        .map(|t| t.to_kind(compute_kind))
        .unwrap_or_else(|| Tensor::zeros([1, 1], (compute_kind, device)));
    let lm_head_ref = if runtime_config.tie_word_embeddings {
        embed_tokens_ref.shallow_clone()
    } else {
        weights_gpu.get("lm_head.weight").map(|t| t.to_kind(compute_kind)).unwrap_or_else(|| Tensor::zeros([1, 1], (compute_kind, device)))
    };

    // Adam optimizer state
    let lr = config.train.learning_rate as f64;
    let beta1 = config.train.adam_beta1 as f64;
    let beta2 = config.train.adam_beta2 as f64;
    let eps = config.train.adam_eps as f64;
    let mut adam_m: Vec<Tensor> = trainable_vars.iter().map(|v| Tensor::zeros_like(v)).collect();
    let mut adam_v: Vec<Tensor> = trainable_vars.iter().map(|v| Tensor::zeros_like(v)).collect();

    // NCCL init for EP
    let nccl_comm = if is_ep {
        let comm_dir = run_paths.root.join("nccl_comm");
        std::fs::create_dir_all(&comm_dir)?;
        Some(rustrain_nccl::nccl::NcclPersistentComm::new(&comm_dir)?)
    } else {
        None
    };

    let vocab_size = runtime_config.vocab_size;
    let batch_size = config.train.micro_batch_size;
    let max_steps = config.train.max_steps as usize;

    // Training loop
    let mut initial_loss = 0.0_f64;
    let mut final_loss = 0.0_f64;

    for step in 0..max_steps {
        let data_start = (step * batch_size) % data.len();
        let sft_batch = data.batch(data_start, batch_size);
        let (input_ids, target_mask) = sft_batch.to_tensors(device, compute_kind);

        // Forward — use Rust eager by default (verified working), C++ checkpoint optional
        // C++ checkpoint saves memory but has a gradient propagation issue to fix.
        // To enable: set use_checkpoint=true in config.
        let use_checkpoint = std::env::var("QWEN36_CHECKPOINT").is_ok();
        let (logits, hidden) = if use_checkpoint && crate::kernel::kernels_available() {
            let group_size = 4;
            // C++ checkpointed forward: groups of 4 layers, recompute during backward
            let embed_prefix = format!("{}embed_tokens.weight", runtime_config.weight_prefix);
            let embed_tokens = weights_gpu.get(&embed_prefix).unwrap().to_kind(compute_kind);
            let final_norm = weights_gpu.get(&format!("{}norm.weight", runtime_config.weight_prefix)).unwrap().to_kind(compute_kind);

            let mut hidden = Tensor::embedding(&embed_tokens, &input_ids, -1, false, false);
            hidden = hidden.detach().set_requires_grad(true);

            let ep_start = shard_ref.map(|s| s.expert_start).unwrap_or(0);
            let ep_count = shard_ref.map(|s| s.experts_per_rank).unwrap_or(256);

            for group_start in (0..runtime_config.num_hidden_layers).step_by(group_size) {
                let group_end = (group_start + group_size).min(runtime_config.num_hidden_layers);
                let layer_indices: Vec<usize> = (group_start..group_end).collect();
                hidden = crate::kernel::checkpoint_forward(
                    &hidden, &layer_indices, &weights_gpu, &runtime_config,
                    &registry, ep_start, ep_count, compute_kind,
                ).unwrap_or_else(|e| {
                    tracing::warn!("C++ checkpoint failed ({e}), using eager");
                    let mut h = hidden.shallow_clone();
                    for &li in &layer_indices {
                        h = forward_single_layer(&h, &weights_gpu, &runtime_config, &registry, shard_ref, compute_kind, li);
                    }
                    h
                });
            }

            let hidden_raw = hidden.shallow_clone();  // pre-norm hidden for MTP
            let hidden_normed = model::rms_norm(&hidden, &final_norm, runtime_config.rms_norm_eps).to_kind(compute_kind);
            let lm_head = if runtime_config.tie_word_embeddings { embed_tokens.shallow_clone() }
                else { weights_gpu.get("lm_head.weight").unwrap().to_kind(compute_kind) };
            let logits = hidden_normed.linear::<&Tensor>(&lm_head, None);
            (logits, hidden_raw)  // MTP needs pre-norm hidden
        } else {
            // Fallback: Rust eager forward (no checkpointing)
            forward_with_lora(&input_ids, &weights_gpu, &runtime_config, &registry, shard_ref, compute_kind)
        };

        // Loss: cross-entropy + optional MTP loss
        let lm_loss = cross_entropy_loss(&logits, &input_ids, &target_mask, vocab_size);
        let loss = if let Some(ref mtp_w) = mtp_weights {
            let mtp_aux = crate::mtp::mtp_loss(
                &logits, &hidden, &input_ids, &target_mask,
                mtp_w, &embed_tokens_ref, &lm_head_ref,
                &runtime_config, compute_kind,
            );
            &lm_loss + &mtp_aux
        } else {
            lm_loss
        };

        let loss_value = loss.double_value(&[]);
        if step == 0 {
            initial_loss = loss_value;
        }
        final_loss = loss_value;
        if step % 10 == 0 || step == max_steps - 1 {
            info!("step {step}/{max_steps} loss={loss_value:.6}");
        }

        // Backward — C++ autograd::Function handles recomputation automatically
        loss.backward();

        // Adam optimizer step (must be in no_grad context for in-place ops)
        tch::no_grad(|| {
            for (i, var) in trainable_vars.iter_mut().enumerate() {
                let grad = var.grad();
                if grad.defined() {
                    let g_synced = if let Some(ref comm) = nccl_comm {
                        comm.all_reduce(&grad).unwrap_or_else(|_| grad.shallow_clone())
                    } else {
                        grad.shallow_clone()
                    };
                    let g = if is_ep { &g_synced / world_size as f64 } else { g_synced };
                    adam_m[i] = &adam_m[i] * beta1 + &g * (1.0 - beta1);
                    adam_v[i] = &adam_v[i] * beta2 + (&g * &g) * (1.0 - beta2);

                    let step_f = (step + 1) as f64;
                    let mh = &adam_m[i] / (1.0 - beta1.powf(step_f));
                    let vh = &adam_v[i] / (1.0 - beta2.powf(step_f));
                    let update = &mh / (&vh.sqrt() + eps);
                    *var -= &(update * lr);
                    var.zero_grad();
                }
            }
        });
    }

    // Save adapter
    let adapter_path = run_paths.root.join("adapter.safetensors");
    registry.save(&adapter_path)?;

    Ok(Qwen36LoraSftSummary {
        adapter_output: adapter_path.to_string_lossy().to_string(),
        initial_loss,
        final_loss,
        trainable_params,
    })
}
