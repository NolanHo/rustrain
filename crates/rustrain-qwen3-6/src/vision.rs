//! Qwen3.6 Vision encoder — ViT (Conv3d patch embed) + M-RoPE attention + patch merger.
//!
//! Matches HF transformers `Qwen3_5MoeVisionModel` implementation:
//! - PatchEmbed: Conv3d with kernel [temporal_patch_size, P, P]
//! - Attention: M-RoPE (2D grid position → rotary embedding)
//! - PatchMerger: spatial_merge_size^2 concat → norm → linear_fc1 → GELU → linear_fc2
//! - Position embed: bilinear interpolation from learned pos_embed

use std::collections::BTreeMap;

use anyhow::Result;
use tch::{Kind, Tensor};

use crate::config::Qwen36RuntimeConfig;
use rustrain_checkpoint::safetensors::tensor;

pub struct VisionWeights {
    pub patch_embed_proj_weight: Tensor, // [hidden, in_ch, temporal, P, P] (5D, Conv3d)
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
/// Normalizes over the last dimension.
fn layer_norm(input: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Tensor {
    let last_dim = *input.size().last().unwrap();
    input.layer_norm::<&Tensor>(vec![last_dim], Some(weight), Some(bias), eps, true)
}

/// Rotate half (standard interleaved rotate_half used in rotary embeddings).
fn rotate_half(x: &Tensor) -> Tensor {
    let last_dim = x.size()[x.size().len() - 1];
    let half = last_dim / 2;
    let x1 = x.narrow(-1, 0, half);
    let x2 = x.narrow(-1, half, half);
    Tensor::cat(&[&x2.neg(), &x1], -1)
}

/// Vision rotary position embedding.
/// Computes inv_freq, then position_ids → cos/sin for 2D M-RoPE.
/// Returns (cos, sin) of shape [seq_len, head_dim].
fn vision_rope(
    head_dim: i64,
    grid_h: i64,
    grid_w: i64,
    spatial_merge_size: i64,
    device: tch::Device,
    compute_kind: Kind,
) -> (Tensor, Tensor) {
    // HF: dim = head_dim // 2, inv_freq = 1.0 / (theta ** (arange(0, dim, 2) / dim))
    // Then: rotary_pos_emb = position_ids.unsqueeze(-1) * inv_freq → flatten → [seq_len, head_dim]
    // Then: emb = cat(rotary_pos_emb, rotary_pos_emb, dim=-1) → cos, sin
    let dim = head_dim / 2;  // 36 for head_dim=72

    // inv_freq: [dim/2] = [18]
    // arange(0, dim, 2) → use arange(0, dim) then narrow/slice every other element
    let all_indices = Tensor::arange_start(0, dim, (Kind::Float, device)); // [0, 1, 2, ..., 35]
    let freq_indices = all_indices.slice(0, 0, dim, 2); // [0, 2, 4, ..., 34] = 18 elements
    let theta = Tensor::from(10000.0f64).to_device(device).to_kind(Kind::Float);
    let inv_freq: Tensor = 1.0 / ((&freq_indices / (dim as f64)) * theta.log()).exp(); // [18]

    // Position IDs: (h, w) grid reordered for spatial_merge_size.
    // HF: h_ids.reshape(H//sms, sms, W//sms, sms).transpose(1, 2).flatten()
    // This groups 2x2 patches together for the merger.
    let sms = spatial_merge_size;
    let merged_h = grid_h / sms;
    let merged_w = grid_w / sms;

    let h_full = Tensor::arange_start(0, grid_h, (Kind::Int64, device))
        .reshape([-1, 1]).expand([grid_h, grid_w], false);  // [H, W]
    let w_full = Tensor::arange_start(0, grid_w, (Kind::Int64, device))
        .reshape([1, -1]).expand([grid_h, grid_w], false);  // [H, W]

    let h_reordered = h_full
        .reshape([merged_h, sms, merged_w, sms])
        .permute([0, 2, 1, 3])  // [merged_h, merged_w, sms, sms]
        .reshape([-1]);  // [seq_len]
    let w_reordered = w_full
        .reshape([merged_h, sms, merged_w, sms])
        .permute([0, 2, 1, 3])
        .reshape([-1]);

    let position_ids = Tensor::stack(&[&h_reordered, &w_reordered], -1); // [seq_len, 2]

    // rotary_pos_emb = position_ids.unsqueeze(-1) * inv_freq → [seq_len, 2, 18] → flatten → [seq_len, 36]
    let pos_f = position_ids.to_kind(Kind::Float).unsqueeze(-1); // [seq_len, 2, 1]
    let rotary = pos_f.matmul(&inv_freq.reshape([1, dim / 2])); // [seq_len, 2, 18]
    let rotary: Tensor = rotary.reshape([grid_h * grid_w, dim]); // [seq_len, 36]

    // HF: emb = cat(rotary_pos_emb, rotary_pos_emb, dim=-1) → [seq_len, 72] = head_dim
    let emb = Tensor::cat(&[&rotary, &rotary], -1); // [seq_len, head_dim]

    let cos = emb.cos().to_kind(compute_kind);
    let sin = emb.sin().to_kind(compute_kind);
    (cos, sin)
}

/// ViT attention with M-RoPE: QKV projection → rotary embedding → softmax attention → output projection.
fn vit_attention(
    hidden: &Tensor,
    w: &VisionBlockWeights,
    num_heads: i64,
    cos: &Tensor,
    sin: &Tensor,
    compute_kind: Kind,
) -> Tensor {
    let seq = hidden.size()[0];
    let head_dim = w.attn_proj_weight.size()[1] / num_heads;

    // QKV: [seq, 3*hidden] → [3, seq, num_heads, head_dim]
    let qkv = hidden
        .linear::<&Tensor>(&w.attn_qkv_weight, Some(&w.attn_qkv_bias))
        .view([seq, 3, num_heads, head_dim])
        .permute([1, 0, 2, 3]); // [3, seq, heads, head_dim]

    let q = qkv.select(0, 0).contiguous(); // [seq, heads, head_dim]
    let k = qkv.select(0, 1).contiguous();
    let v = qkv.select(0, 2).contiguous();

    // Apply M-RoPE: cos/sin [seq, head_dim] → unsqueeze for heads
    let cos_h = cos.unsqueeze(-2); // [seq, 1, head_dim]
    let sin_h = sin.unsqueeze(-2);

    // Compute in FP32 for RoPE precision
    let q_f32 = q.to_kind(Kind::Float);
    let k_f32 = k.to_kind(Kind::Float);
    let cos_f = cos_h.to_kind(Kind::Float);
    let sin_f = sin_h.to_kind(Kind::Float);

    let q_rot = (&q_f32 * &cos_f + &rotate_half(&q_f32) * &sin_f).to_kind(compute_kind);
    let k_rot = (&k_f32 * &cos_f + &rotate_half(&k_f32) * &sin_f).to_kind(compute_kind);

    // Attention: [heads, seq, seq] = Q @ K^T / sqrt(d)
    let q_t = q_rot.transpose(0, 1).unsqueeze(0); // [1, heads, seq, head_dim]
    let k_t = k_rot.transpose(0, 1).unsqueeze(0);
    let v_t = v.transpose(0, 1).unsqueeze(0);

    let scale = 1.0 / (head_dim as f64).sqrt();
    let attn = (&q_t * scale).matmul(&k_t.transpose(-1, -2));
    let attn = attn.softmax(-1, compute_kind);
    let out = attn.matmul(&v_t); // [1, heads, seq, head_dim]

    // Reshape back: [seq, hidden]
    let out = out
        .permute([0, 2, 1, 3])
        .reshape([seq, num_heads * head_dim]);

    out.linear::<&Tensor>(&w.attn_proj_weight, Some(&w.attn_proj_bias))
}

/// ViT MLP: GELU activation (gelu_pytorch_tanh = "none" approximation in tch).
fn vit_mlp(hidden: &Tensor, w: &VisionBlockWeights) -> Tensor {
    let h = hidden
        .linear::<&Tensor>(&w.mlp_fc1_weight, Some(&w.mlp_fc1_bias))
        .gelu("none");
    h.linear::<&Tensor>(&w.mlp_fc2_weight, Some(&w.mlp_fc2_bias))
}

/// Forward pass through the vision encoder.
///
/// `pixel_values`: [seq_len, in_channels * temporal_patch_size * patch_size * patch_size]
///   (preprocessed, matching HF transformers format)
/// `grid_thw`: [t, h, w] grid dimensions (number of patches in each dimension)
/// Returns: [num_visual_tokens, out_hidden_size]
pub fn vision_forward(
    pixel_values: &Tensor,
    weights: &VisionWeights,
    config: &Qwen36RuntimeConfig,
    compute_kind: Kind,
) -> Tensor {
    let num_heads = config.vision_num_heads;
    let eps = 1e-6_f64; // ViT uses 1e-6, not the text model's rms_norm_eps
    let p = config.vision_patch_size;
    let t_p = config.vision_temporal_patch_size;
    let in_ch = 3_i64; // config.vision_in_channels (not stored, always 3)
    let hidden = config.vision_hidden_size;
    let sms = config.vision_spatial_merge_size; // spatial_merge_size

    // ── Patch embedding via Conv3d ──
    // Weight: [hidden, in_ch, temporal, P, P] → tch conv3d expects [out, in, kT, kH, kW]
    // Input: [seq_len, in_ch * t_p * P * P] → reshape to [1, in_ch, t, h, w] (3D volume)
    // But HF reshapes to [-1, in_ch, temporal, P, P] per patch, then Conv3d.
    // Actually HF does: hidden_states.view(-1, in_ch, temporal, P, P) then Conv3d.
    // The input is already preprocessed as [seq_len, in_ch*t_p*P*P] where seq_len = num_patches
    // Each row is a flattened [in_ch, temporal, P, P] patch.

    let seq_len = pixel_values.size()[0];
    // Reshape to [seq_len, in_ch, t_p, P, P] for per-patch Conv3d
    let x = pixel_values
        .view([seq_len, in_ch, t_p, p, p])
        .to_kind(compute_kind);

    // Conv3d: weight [hidden, in_ch, t_p, P, P], stride = kernel_size
    // tch-rs conv3d takes (weight, bias, stride, padding, dilation, groups) — 6 args
    let patch_embed = x.conv3d::<&Tensor>(
        &weights.patch_embed_proj_weight,
        Some(&weights.patch_embed_proj_bias),
        &[t_p, p, p],  // stride
        &[0, 0, 0],   // padding
        &[1, 1, 1],   // dilation
        1,            // groups
    );
    // Output: [1, hidden, 1, 1, 1] → flatten to [1, hidden]? No.
    // Actually Conv3d with stride=kernel on [seq_len, in_ch, t_p, P, P] gives [seq_len, hidden, 1, 1, 1]
    // But tch conv3d expects [N, C, D, H, W] → output [N, out, D', H', W']
    // Here N=seq_len, D=t_p, H=P, W=P, stride=(t_p,P,P) → D'=1, H'=1, W'=1
    let patch_embed = patch_embed.view([seq_len, hidden]); // [seq_len, hidden]

    // Grid dimensions (square grid for single image)
    let grid_h = (seq_len as f64).sqrt() as i64;
    let grid_w = grid_h;
    let device = x.device();

    // ── Position embeddings (bilinear interpolation, matching HF) ──
    // HF: bilinear_indices and bilinear_weights computed from get_vision_bilinear_indices_and_weights
    // then: pos_embeds = (self.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
    // The bilinear interpolation uses linspace(0, side-1, h) and linspace(0, side-1, w)
    // and also reorders for spatial_merge_size.
    let pos = &weights.pos_embed_weight;
    let num_pos = pos.size()[0];
    let pos_side = (num_pos as f64).sqrt() as i64;  // 48 for 2304
    let pos_h = pos_side;
    let pos_w = pos_side;

    // h_grid = linspace(0, pos_side-1, grid_h), w_grid = linspace(0, pos_side-1, grid_w)
    let h_grid = Tensor::linspace(0.0, (pos_side - 1) as f64, grid_h, (Kind::Float, device));
    let w_grid = Tensor::linspace(0.0, (pos_side - 1) as f64, grid_w, (Kind::Float, device));

    let h_floor = h_grid.floor().to_kind(Kind::Int64);
    let w_floor = w_grid.floor().to_kind(Kind::Int64);
    let h_ceil = (&h_floor + 1).clamp(0, pos_side - 1);
    let w_ceil = (&w_floor + 1).clamp(0, pos_side - 1);

    let h_frac = &h_grid - h_floor.to_kind(Kind::Float);
    let w_frac = &w_grid - w_floor.to_kind(Kind::Float);

    // Bilinear interpolation: 4 corner indices
    // h_floor_offset = h_floor * pos_side, h_ceil_offset = h_ceil * pos_side
    let h_floor_off = (&h_floor * pos_side).unsqueeze(-1); // [grid_h, 1]
    let h_ceil_off = (&h_ceil * pos_side).unsqueeze(-1);
    let w_floor_flat = w_floor.reshape([1, -1]); // [1, grid_w]
    let w_ceil_flat = w_ceil.reshape([1, -1]);

    // Corner indices: [grid_h*grid_w] each
    let idx_00 = (&h_floor_off + &w_floor_flat).reshape([-1]);
    let idx_01 = (&h_floor_off + &w_ceil_flat).reshape([-1]);
    let idx_10 = (&h_ceil_off + &w_floor_flat).reshape([-1]);
    let idx_11 = (&h_ceil_off + &w_ceil_flat).reshape([-1]);

    // Corner weights: [grid_h*grid_w]
    let h_frac_col = h_frac.unsqueeze(-1); // [grid_h, 1]
    let w_frac_row = w_frac.reshape([1, -1]); // [1, grid_w]
    let one = Tensor::from(1.0f64).to_device(device).to_kind(Kind::Float);
    let w_00: Tensor = ((&one - &h_frac_col) * (&one - &w_frac_row)).reshape([-1]);
    let w_01: Tensor = ((&one - &h_frac_col) * &w_frac_row).reshape([-1]);
    let w_10: Tensor = (&h_frac_col * (&one - &w_frac_row)).reshape([-1]);
    let w_11: Tensor = (&h_frac_col * &w_frac_row).reshape([-1]);

    // Reorder for spatial_merge_size (same as HF reorder)
    let merged_h = grid_h / sms;
    let merged_w = grid_w / sms;
    let h_idx = Tensor::arange_start(0, grid_h, (Kind::Int64, device)).reshape([merged_h, sms]);
    let w_idx = Tensor::arange_start(0, grid_w, (Kind::Int64, device)).reshape([merged_w, sms]);
    // reorder = (h_idx[:,:,None,None] * grid_w + w_idx[None,None,:,:]).transpose(1,2).flatten()
    let reorder = (&h_idx.unsqueeze(-1).unsqueeze(-1) * grid_w + &w_idx.unsqueeze(0).unsqueeze(0))
        .permute([0, 2, 1, 3])
        .reshape([-1]); // [seq_len]

    // Apply reorder to indices and weights
    let reorder_indices = |idx: &Tensor| -> Tensor { idx.index_select(0, &reorder) };
    let reorder_weights = |w: &Tensor| -> Tensor { w.index_select(0, &reorder) };

    let idx_00 = reorder_indices(&idx_00);
    let idx_01 = reorder_indices(&idx_01);
    let idx_10 = reorder_indices(&idx_10);
    let idx_11 = reorder_indices(&idx_11);
    let w_00 = reorder_weights(&w_00);
    let w_01 = reorder_weights(&w_01);
    let w_10 = reorder_weights(&w_10);
    let w_11 = reorder_weights(&w_11);

    // pos_embeds = sum_k(pos_embed[idx_k] * w_k[:, None])
    let pos_embeds = {
        let e00 = pos.index_select(0, &idx_00) * w_00.unsqueeze(-1).to_kind(pos.kind());
        let e01 = pos.index_select(0, &idx_01) * w_01.unsqueeze(-1).to_kind(pos.kind());
        let e10 = pos.index_select(0, &idx_10) * w_10.unsqueeze(-1).to_kind(pos.kind());
        let e11 = pos.index_select(0, &idx_11) * w_11.unsqueeze(-1).to_kind(pos.kind());
        (e00 + e01 + e10 + e11)
    }.to_kind(compute_kind); // [seq_len, hidden]

    let x = &patch_embed + &pos_embeds;

    // ── M-RoPE position embeddings ──
    let head_dim = hidden / num_heads;
    let (cos, sin) = vision_rope(head_dim, grid_h, grid_w, sms, x.device(), compute_kind);

    // ── ViT blocks ──
    let mut x = x; // [seq_len, hidden]
    for block in &weights.blocks {
        let normed = layer_norm(&x, &block.norm1_weight, &block.norm1_bias, eps);
        let attn_out = vit_attention(&normed, block, num_heads, &cos, &sin, compute_kind);
        x = &x + &attn_out;
        let normed = layer_norm(&x, &block.norm2_weight, &block.norm2_bias, eps);
        let mlp_out = vit_mlp(&normed, block);
        x = &x + &mlp_out;
    }

    // ── Patch merger ──
    // HF: use_postshuffle_norm=False, so norm is applied on [seq_len, hidden] BEFORE reshape.
    // Then: reshape to [num_tokens, hidden * sms^2] → fc1 → GELU → fc2
    let merged_h = grid_h / sms;
    let merged_w = grid_w / sms;
    let merged_hidden = hidden * sms * sms;

    // Norm on [seq_len, hidden] (matching HF: norm before post-shuffle reshape)
    let x_normed = layer_norm(&x, &weights.merger_norm_weight, &weights.merger_norm_bias, eps);

    // Reshape: [seq_len, hidden] → [merged_h, sms, merged_w, sms, hidden] → [num_tokens, hidden*sms^2]
    let x = x_normed
        .view([merged_h, sms, merged_w, sms, hidden])
        .permute([0, 2, 1, 3, 4])  // [merged_h, merged_w, sms, sms, hidden]
        .reshape([merged_h * merged_w, merged_hidden]); // [num_tokens, hidden * sms^2]

    let x = x
        .linear::<&Tensor>(&weights.merger_fc1_weight, Some(&weights.merger_fc1_bias))
        .gelu("none");
    let x = x.linear::<&Tensor>(&weights.merger_fc2_weight, Some(&weights.merger_fc2_bias));

    x // [num_tokens, out_hidden_size]
}
