//! Qwen3.6 Vision encoder — ViT + patch merger
//! For text-only LoRA SFT, vision encoder uses frozen weights.

use std::collections::BTreeMap;

use anyhow::Result;
use tch::{Kind, Tensor};

use crate::config::Qwen36RuntimeConfig;
use rustrain_checkpoint::safetensors::tensor;

pub struct VisionWeights {
    pub patch_embed_proj_weight: Tensor, // [hidden, 3*P*P, ...]
    pub patch_embed_proj_bias: Tensor,
    pub pos_embed_weight: Tensor, // [num_pos, hidden]
    pub blocks: Vec<VisionBlockWeights>,
    pub merger_norm_weight: Tensor,
    pub merger_norm_bias: Tensor,
    pub merger_fc1_weight: Tensor,
    pub merger_fc1_bias: Tensor,
    pub merger_fc2_weight: Tensor,
    pub merger_fc2_bias: Tensor,
}

pub struct VisionBlockWeights {
    pub norm1_weight: Tensor,
    pub norm1_bias: Tensor,
    pub attn_qkv_weight: Tensor,
    pub attn_qkv_bias: Tensor,
    pub attn_proj_weight: Tensor,
    pub attn_proj_bias: Tensor,
    pub norm2_weight: Tensor,
    pub norm2_bias: Tensor,
    pub mlp_fc1_weight: Tensor,
    pub mlp_fc1_bias: Tensor,
    pub mlp_fc2_weight: Tensor,
    pub mlp_fc2_bias: Tensor,
}

impl VisionWeights {
    pub fn load(
        weights: &BTreeMap<String, Tensor>,
        config: &Qwen36RuntimeConfig,
        kind: Kind,
    ) -> Result<Self> {
        let p = "model.visual";
        let depth = config.vision_depth;

        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let bp = format!("{p}.blocks.{i}");
            blocks.push(VisionBlockWeights {
                norm1_weight: tensor(weights, &format!("{bp}.norm1.weight"))?.to_kind(kind),
                norm1_bias: tensor(weights, &format!("{bp}.norm1.bias"))?.to_kind(kind),
                attn_qkv_weight: tensor(weights, &format!("{bp}.attn.qkv.weight"))?.to_kind(kind),
                attn_qkv_bias: tensor(weights, &format!("{bp}.attn.qkv.bias"))?.to_kind(kind),
                attn_proj_weight: tensor(weights, &format!("{bp}.attn.proj.weight"))?.to_kind(kind),
                attn_proj_bias: tensor(weights, &format!("{bp}.attn.proj.bias"))?.to_kind(kind),
                norm2_weight: tensor(weights, &format!("{bp}.norm2.weight"))?.to_kind(kind),
                norm2_bias: tensor(weights, &format!("{bp}.norm2.bias"))?.to_kind(kind),
                mlp_fc1_weight: tensor(weights, &format!("{bp}.mlp.linear_fc1.weight"))?.to_kind(kind),
                mlp_fc1_bias: tensor(weights, &format!("{bp}.mlp.linear_fc1.bias"))?.to_kind(kind),
                mlp_fc2_weight: tensor(weights, &format!("{bp}.mlp.linear_fc2.weight"))?.to_kind(kind),
                mlp_fc2_bias: tensor(weights, &format!("{bp}.mlp.linear_fc2.bias"))?.to_kind(kind),
            });
        }

        Ok(Self {
            patch_embed_proj_weight: tensor(weights, &format!("{p}.patch_embed.proj.weight"))?.to_kind(kind),
            patch_embed_proj_bias: tensor(weights, &format!("{p}.patch_embed.proj.bias"))?.to_kind(kind),
            pos_embed_weight: tensor(weights, &format!("{p}.pos_embed.weight"))?.to_kind(kind),
            blocks,
            merger_norm_weight: tensor(weights, &format!("{p}.merger.norm.weight"))?.to_kind(kind),
            merger_norm_bias: tensor(weights, &format!("{p}.merger.norm.bias"))?.to_kind(kind),
            merger_fc1_weight: tensor(weights, &format!("{p}.merger.linear_fc1.weight"))?.to_kind(kind),
            merger_fc1_bias: tensor(weights, &format!("{p}.merger.linear_fc1.bias"))?.to_kind(kind),
            merger_fc2_weight: tensor(weights, &format!("{p}.merger.linear_fc2.weight"))?.to_kind(kind),
            merger_fc2_bias: tensor(weights, &format!("{p}.merger.linear_fc2.bias"))?.to_kind(kind),
        })
    }

    /// List all weight names needed for vision encoder.
    pub fn weight_names(config: &Qwen36RuntimeConfig) -> Vec<String> {
        let p = "model.visual";
        let depth = config.vision_depth;
        let mut names = vec![
            format!("{p}.patch_embed.proj.weight"),
            format!("{p}.patch_embed.proj.bias"),
            format!("{p}.pos_embed.weight"),
            format!("{p}.merger.norm.weight"),
            format!("{p}.merger.norm.bias"),
            format!("{p}.merger.linear_fc1.weight"),
            format!("{p}.merger.linear_fc1.bias"),
            format!("{p}.merger.linear_fc2.weight"),
            format!("{p}.merger.linear_fc2.bias"),
        ];
        for i in 0..depth {
            let bp = format!("{p}.blocks.{i}");
            names.extend([
                format!("{bp}.norm1.weight"),
                format!("{bp}.norm1.bias"),
                format!("{bp}.attn.qkv.weight"),
                format!("{bp}.attn.qkv.bias"),
                format!("{bp}.attn.proj.weight"),
                format!("{bp}.attn.proj.bias"),
                format!("{bp}.norm2.weight"),
                format!("{bp}.norm2.bias"),
                format!("{bp}.mlp.linear_fc1.weight"),
                format!("{bp}.mlp.linear_fc1.bias"),
                format!("{bp}.mlp.linear_fc2.weight"),
                format!("{bp}.mlp.linear_fc2.bias"),
            ]);
        }
        names
    }
}

/// Layer norm helper (with bias, unlike RMS norm).
fn layer_norm(input: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Tensor {
    input.layer_norm::<&Tensor>(vec![-1i64], Some(weight), Some(bias), eps, true)
}

/// ViT attention: QKV projection → multi-head attention → output projection.
fn vit_attention(
    hidden: &Tensor,
    w: &VisionBlockWeights,
    num_heads: i64,
    compute_kind: Kind,
) -> Tensor {
    let (batch, seq, _hidden_dim) = hidden.size3().unwrap();
    let head_dim = w.attn_proj_weight.size()[1] / num_heads;

    let qkv = hidden
        .linear::<&Tensor>(&w.attn_qkv_weight, Some(&w.attn_qkv_bias))
        .view([batch, seq, 3, num_heads, head_dim])
        .permute([2, 0, 3, 1, 4]); // [3, batch, heads, seq, head_dim]

    let q = qkv.select(0, 0).contiguous();
    let k = qkv.select(0, 1).contiguous();
    let v = qkv.select(0, 2).contiguous();

    let scale = 1.0 / (head_dim as f64).sqrt();
    let attn = (&q * scale).matmul(&k.transpose(-1, -2));
    let attn = attn.softmax(-1, compute_kind);
    let out = attn.matmul(&v); // [batch, heads, seq, head_dim]

    let out = out
        .permute([0, 2, 1, 3])
        .reshape([batch, seq, num_heads * head_dim]);

    out.linear::<&Tensor>(&w.attn_proj_weight, Some(&w.attn_proj_bias))
}

/// ViT MLP: GELU activation (gelu_pytorch_tanh).
fn vit_mlp(hidden: &Tensor, w: &VisionBlockWeights) -> Tensor {
    let h = hidden
        .linear::<&Tensor>(&w.mlp_fc1_weight, Some(&w.mlp_fc1_bias))
        .gelu("none");
    h.linear::<&Tensor>(&w.mlp_fc2_weight, Some(&w.mlp_fc2_bias))
}

/// Forward pass through the vision encoder.
///
/// `pixel_values`: [batch, channels, height, width] (preprocessed images)
/// Returns: [batch, num_visual_tokens, out_hidden_size]
pub fn vision_forward(
    pixel_values: &Tensor,
    weights: &VisionWeights,
    config: &Qwen36RuntimeConfig,
    compute_kind: Kind,
) -> Tensor {
    let num_heads = config.vision_num_heads;
    let eps = config.rms_norm_eps;

    // Patch embedding: Conv2d → [batch, hidden, h/P, w/P]
    let patch_embed = pixel_values.conv2d::<&Tensor>(
        &weights.patch_embed_proj_weight,
        Some(&weights.patch_embed_proj_bias),
        &[config.vision_patch_size as i64, config.vision_patch_size as i64],
        &[0, 0],
        &[1, 1],
        1,
    );

    // Flatten spatial dims: [batch, hidden, h', w'] → [batch, hidden, num_patches] → [batch, num_patches, hidden]
    let (batch, hidden_dim, h, w) = patch_embed.size4().unwrap();
    let patch_embed = patch_embed
        .reshape([batch, hidden_dim, h * w])
        .transpose(1, 2); // [batch, num_patches, hidden]

    // Add positional embeddings
    let pos = weights.pos_embed_weight.to_kind(compute_kind);
    let patch_embed = &patch_embed + &pos;

    // ViT blocks
    let mut x = patch_embed;
    for block in &weights.blocks {
        let normed = layer_norm(&x, &block.norm1_weight, &block.norm1_bias, eps);
        let attn_out = vit_attention(&normed, block, num_heads, compute_kind);
        x = &x + &attn_out;
        let normed = layer_norm(&x, &block.norm2_weight, &block.norm2_bias, eps);
        let mlp_out = vit_mlp(&normed, block);
        x = &x + &mlp_out;
    }

    // Merger: norm → linear_fc1 → GELU → linear_fc2
    let merged = layer_norm(&x, &weights.merger_norm_weight, &weights.merger_norm_bias, eps);
    let merged = merged
        .linear::<&Tensor>(&weights.merger_fc1_weight, Some(&weights.merger_fc1_bias))
        .gelu("none");
    let merged = merged.linear::<&Tensor>(&weights.merger_fc2_weight, Some(&weights.merger_fc2_bias));

    merged
}
