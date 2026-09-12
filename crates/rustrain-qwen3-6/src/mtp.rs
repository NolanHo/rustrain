//! Qwen3.6 MTP (Multi-Token Prediction) — forward pass and loss.
//!
//! MTP structure (from config + weight map):
//! - mtp.fc.weight: [hidden, 2*hidden] — project combined hidden+embed → hidden
//! - mtp.pre_fc_norm_embedding.weight: [hidden] — norm before embedding projection
//! - mtp.pre_fc_norm_hidden.weight: [hidden] — norm before hidden projection
//! - mtp.norm.weight: [hidden] — final norm before logits
//! - mtp.layers.{N}: full attention + MoE (same architecture as main layers)
//!
//! Forward:
//!   combined = cat(pre_fc_norm_embedding(embed[t]), pre_fc_norm_hidden(hidden[t]))
//!   projected = fc(combined)  → [hidden]
//!   layer_output = mtp_layer(projected)  → full attention + MoE
//!   logits = lm_head(layer_output)
//!   MTP predicts token t+1 from hidden[t] + embed[t]
//!
//! Loss: cross-entropy on shifted tokens, weight = 0.5

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use tch::{Kind, Reduction, Tensor};

use crate::config::{LayerType, Qwen36RuntimeConfig};
use crate::model::{self, FullAttnWeights, MoeWeights, Qwen36LayerWeights, rms_norm};
use rustrain_checkpoint::safetensors::tensor;

pub struct MtpWeights {
    pub fc: Tensor,                       // [hidden, 2*hidden]
    pub pre_fc_norm_embedding: Tensor,    // [hidden]
    pub pre_fc_norm_hidden: Tensor,       // [hidden]
    pub norm: Tensor,                      // [hidden] — final norm
    pub layers: Vec<MtpLayerWeights>,      // mtp_num_hidden_layers layers
}

pub struct MtpLayerWeights {
    pub input_norm: Tensor,
    pub post_attention_norm: Tensor,
    pub attn: FullAttnWeights,        // full attention only (no linear attn in MTP)
    pub moe: MoeWeights,              // MoE with shared expert + gate
}

impl MtpWeights {
    pub fn load(
        weights: &BTreeMap<String, Tensor>,
        config: &Qwen36RuntimeConfig,
        kind: Kind,
    ) -> Result<Self> {
        let fc = tensor(weights, "mtp.fc.weight")?.to_kind(kind);
        let pre_fc_norm_embedding = tensor(weights, "mtp.pre_fc_norm_embedding.weight")?.to_kind(kind);
        let pre_fc_norm_hidden = tensor(weights, "mtp.pre_fc_norm_hidden.weight")?.to_kind(kind);
        let norm = tensor(weights, "mtp.norm.weight")?.to_kind(kind);

        let mut layers = Vec::with_capacity(config.mtp_num_hidden_layers);
        for i in 0..config.mtp_num_hidden_layers {
            let prefix = format!("mtp.layers.{i}");
            let input_norm = tensor(weights, &format!("{prefix}.input_layernorm.weight"))?.to_kind(kind);
            let post_attention_norm = tensor(weights, &format!("{prefix}.post_attention_layernorm.weight"))?.to_kind(kind);
            let attn = FullAttnWeights::load(weights, &prefix, kind)?;
            let moe = MoeWeights::load(weights, &prefix, kind)?;
            layers.push(MtpLayerWeights {
                input_norm,
                post_attention_norm,
                attn,
                moe,
            });
        }

        Ok(Self { fc, pre_fc_norm_embedding, pre_fc_norm_hidden, norm, layers })
    }

    /// List all weight names needed for MTP.
    pub fn weight_names(config: &Qwen36RuntimeConfig) -> Vec<String> {
        let mut names = vec![
            "mtp.fc.weight".to_string(),
            "mtp.pre_fc_norm_embedding.weight".to_string(),
            "mtp.pre_fc_norm_hidden.weight".to_string(),
            "mtp.norm.weight".to_string(),
        ];
        for i in 0..config.mtp_num_hidden_layers {
            let p = format!("mtp.layers.{i}");
            names.extend([
                format!("{p}.input_layernorm.weight"),
                format!("{p}.post_attention_layernorm.weight"),
                format!("{p}.self_attn.q_proj.weight"),
                format!("{p}.self_attn.q_norm.weight"),
                format!("{p}.self_attn.k_proj.weight"),
                format!("{p}.self_attn.k_norm.weight"),
                format!("{p}.self_attn.v_proj.weight"),
                format!("{p}.self_attn.o_proj.weight"),
            ]);
            if config.is_moe {
                names.extend([
                    format!("{p}.mlp.gate.weight"),
                    format!("{p}.mlp.shared_expert_gate.weight"),
                    format!("{p}.mlp.shared_expert.gate_proj.weight"),
                    format!("{p}.mlp.shared_expert.up_proj.weight"),
                    format!("{p}.mlp.shared_expert.down_proj.weight"),
                    format!("{p}.mlp.experts.gate_up_proj"),
                    format!("{p}.mlp.experts.down_proj"),
                ]);
            } else {
                names.extend([
                    format!("{p}.mlp.gate_proj.weight"),
                    format!("{p}.mlp.up_proj.weight"),
                    format!("{p}.mlp.down_proj.weight"),
                ]);
            }
        }
        names
    }
}

/// MTP forward pass for a single MTP layer.
fn mtp_layer_forward(
    hidden: &Tensor,
    layer: &MtpLayerWeights,
    config: &Qwen36RuntimeConfig,
    compute_kind: Kind,
) -> Tensor {
    let input = hidden.to_kind(compute_kind);

    let attn_input = rms_norm(&input, &layer.input_norm, config.rms_norm_eps).to_kind(compute_kind);
    let attn_output = model::full_attention(&attn_input, &layer.attn, config, compute_kind);
    let after_attention = &input + &attn_output;

    let moe_input = rms_norm(&after_attention, &layer.post_attention_norm, config.rms_norm_eps).to_kind(compute_kind);
    let moe_output = model::moe_forward(&moe_input, &layer.moe, config, compute_kind);

    (after_attention + moe_output).to_kind(compute_kind)
}

/// Full MTP forward: produce logits for next-token prediction.
///
/// `hidden`: [batch, seq, hidden] — last hidden state from main model
/// `input_ids`: [batch, seq] — token IDs (for embedding lookup)
/// `embed_tokens`: [vocab, hidden] — embedding table
/// `lm_head`: [vocab, hidden] — output projection
/// Returns: [batch, seq-1, vocab] — logits for tokens 1..seq (shifted)
pub fn mtp_forward(
    hidden: &Tensor,
    input_ids: &Tensor,
    mtp_weights: &MtpWeights,
    embed_tokens: &Tensor,
    lm_head: &Tensor,
    config: &Qwen36RuntimeConfig,
    compute_kind: Kind,
) -> Tensor {
    let (batch, seq_len, _hidden) = hidden.size3().unwrap();
    let eps = config.rms_norm_eps;

    // hidden[t] + embed[t+1] → predict token t+2 (Megatron convention)
    let hidden_shifted = hidden.narrow(1, 0, seq_len - 1);  // [batch, seq-1, hidden]
    let embed_next = Tensor::embedding(embed_tokens, &input_ids.narrow(1, 1, seq_len - 1), -1, false, false);

    let h_normed = rms_norm(&hidden_shifted, &mtp_weights.pre_fc_norm_hidden, eps).to_kind(compute_kind);
    let e_normed = rms_norm(&embed_next, &mtp_weights.pre_fc_norm_embedding, eps).to_kind(compute_kind);

    // Combine: embed first, then hidden → fc projection
    let combined = Tensor::cat(&[&e_normed, &h_normed], -1);
    let projected = combined.linear::<&Tensor>(&mtp_weights.fc, None);

    // MTP layers
    let mut h = projected;
    for layer in &mtp_weights.layers {
        h = mtp_layer_forward(&h, layer, config, compute_kind);
    }

    // Final norm + logits
    let h = rms_norm(&h, &mtp_weights.norm, eps).to_kind(compute_kind);
    h.linear::<&Tensor>(lm_head, None)
}

/// MTP loss: cross-entropy on shifted tokens.
///
/// `main_logits`: [batch, seq, vocab] — from main model forward
/// `hidden`: [batch, seq, hidden] — last hidden state before lm_head
/// `input_ids`: [batch, seq]
/// `target_mask`: [batch, seq] — 1.0 for response tokens, 0.0 for prompt
/// Returns: scalar loss tensor
pub fn mtp_loss(
    main_logits: &Tensor,
    hidden: &Tensor,
    input_ids: &Tensor,
    target_mask: &Tensor,
    mtp_weights: &MtpWeights,
    embed_tokens: &Tensor,
    lm_head: &Tensor,
    config: &Qwen36RuntimeConfig,
    compute_kind: Kind,
) -> Tensor {
    let vocab_size = config.vocab_size;
    let seq_len = input_ids.size()[1];

    // MTP logits: [batch, seq-1, vocab]
    let mtp_logits = mtp_forward(hidden, input_ids, mtp_weights, embed_tokens, lm_head, config, compute_kind);

    // MTP predicts token t+2 from hidden[t] + embed[t+1] (Megatron convention)
    let mtp_targets = input_ids.narrow(1, 2, seq_len - 2).reshape([-1]);
    let mtp_mask = target_mask.narrow(1, 2, seq_len - 2).reshape([-1]);

    // Reshape to 2D for g_nll_loss: [batch*(seq-2), vocab]
    let log_probs = mtp_logits.narrow(1, 0, seq_len - 2).reshape([-1, vocab_size]).log_softmax(-1, Kind::Float);
    let per_token_loss = log_probs.g_nll_loss::<&Tensor>(&mtp_targets, None, Reduction::None, -100);
    let masked_loss = &per_token_loss * &mtp_mask;
    let total = masked_loss.sum(Kind::Float);
    let count = mtp_mask.sum(Kind::Float).clamp_min(1.0);

    // MTP loss weighted by 0.5 (matching V4/GLM5 convention)
    (total / count) * 0.5
}
