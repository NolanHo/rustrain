//! Qwen3.6 model — hybrid linear/full attention, MoE with shared expert + gate, MTP

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use tch::{Device, IndexOp, Kind, Tensor, no_grad};

use crate::config::{LayerType, Qwen36RuntimeConfig};
use rustrain_checkpoint::safetensors::tensor;

// ──────────────────────────────────────────────────────────────────────
// RMS Norm
// ──────────────────────────────────────────────────────────────────────

/// RMSNorm — HF Qwen3.5-MoE uses `1.0 + weight` (not raw weight).
/// `output = (x * rsqrt(var+eps)) * (1.0 + weight)` in FP32, then cast back.
pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let input_f32 = input.to_kind(Kind::Float);
    let variance = input_f32
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    let inv_rms = (variance + eps).rsqrt();  // FP32
    // HF: output * (1.0 + self.weight.float())
    let normed = input.to_kind(Kind::Float) * inv_rms;
    let weight_adjusted = 1.0_f64 + weight.to_kind(Kind::Float);
    (normed * &weight_adjusted).to_kind(input.kind())
}

/// Gated RMSNorm — HF Qwen3_5MoeRMSNormGated uses raw weight (NOT 1+weight).
/// HF: hidden = weight * (hidden * rsqrt(var+eps)).to(dtype) * F.silu(gate.float())
pub fn rms_norm_gated(input: &Tensor, weight: &Tensor, gate: &Tensor, eps: f64) -> Tensor {
    let input_f32 = input.to_kind(Kind::Float);
    let variance = input_f32
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    let inv_rms = (variance + eps).rsqrt();
    let normed = input_f32 * inv_rms;
    // HF gated norm uses raw weight (not 1+weight)
    let normed = (normed * weight.to_kind(Kind::Float)).to_kind(input.kind());
    // HF applies SiLU to gate
    let gate_silu = gate.to_kind(Kind::Float).silu().to_kind(input.kind());
    normed * gate_silu
}

// ──────────────────────────────────────────────────────────────────────
// RoPE (partial rotary, split-half — matching HF Qwen3.5-MoE)
// ──────────────────────────────────────────────────────────────────────

/// Split-half rotate: x = [a, b] → [-b, a]
fn rotate_half(x: &Tensor) -> Tensor {
    let last_dim = x.size()[x.dim() - 1];
    let half = last_dim / 2;
    let x1 = x.narrow(-1, 0, half);
    let x2 = x.narrow(-1, half, half);
    Tensor::cat(&[&x2.neg(), &x1], -1)
}

/// Apply partial rotary RoPE (split-half, matching HF).
///
/// `q` / `k`: [batch, num_heads, seq, head_dim]
/// `position_ids`: [batch, seq] (text-only: 0..seq)
pub fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    position_ids: &Tensor,
    rotary_dim: i64,
    theta: f64,
) -> (Tensor, Tensor) {
    let head_dim = q.size()[3];
    let device = q.device();

    let pos = position_ids.to_kind(Kind::Float); // [batch, seq]
    let half = rotary_dim / 2;

    // inv_freq: [half]  — HF: 1 / (theta ^ (arange(0, half, 2) / half))
    // Actually HF computes: base = theta, dim = rotary_dim, inv_freq = 1 / (base ** (arange(0, dim, 2) / dim))
    // But cos/sin shape = [batch, seq, rotary_dim] (not [batch, seq, half])
    // HF: freqs = inv_freq @ position_ids → [batch, seq, half]
    //     emb = cat(freqs, freqs, dim=-1) → [batch, seq, rotary_dim]
    //     cos, sin = emb.cos(), emb.sin()
    let inv_freq = {
        let all = Tensor::arange_start(0, rotary_dim, (Kind::Float, device));
        let exponents = all.slice(0, 0, rotary_dim, 2) / (rotary_dim as f64);
        (exponents * theta.ln()).exp().reciprocal()
    };

    // freqs: [batch, seq, half]
    let freqs = pos.unsqueeze(-1) * inv_freq.unsqueeze(0);

    // HF: emb = cat(freqs, freqs, dim=-1) → [batch, seq, rotary_dim]
    let emb = Tensor::cat(&[&freqs, &freqs], -1);
    let cos = emb.cos().unsqueeze(1).to_kind(q.kind()); // [batch, 1, seq, rotary_dim]
    let sin = emb.sin().unsqueeze(1).to_kind(q.kind());

    let q_parts = q.split(rotary_dim, -1);
    let k_parts = k.split(rotary_dim, -1);
    let q_rot = &q_parts[0];
    let k_rot = &k_parts[0];
    let q_pass = q.narrow(-1, rotary_dim, head_dim - rotary_dim);
    let k_pass = k.narrow(-1, rotary_dim, head_dim - rotary_dim);

    // Split-half rotate: q_embed = q_rot * cos + rotate_half(q_rot) * sin
    let q_rotated = q_rot * &cos + &rotate_half(q_rot) * &sin;
    let k_rotated = k_rot * &cos + &rotate_half(k_rot) * &sin;

    let q_out = if head_dim > rotary_dim {
        Tensor::cat(&[&q_rotated, &q_pass], -1)
    } else {
        q_rotated
    };
    let k_out = if head_dim > rotary_dim {
        Tensor::cat(&[&k_rotated, &k_pass], -1)
    } else {
        k_rotated
    };
    (q_out, k_out)
}

// ──────────────────────────────────────────────────────────────────────
// Full Attention (with output gate)
// ──────────────────────────────────────────────────────────────────────

pub struct FullAttnWeights {
    pub q_proj: Tensor,      // [2 * num_heads * head_dim, hidden]
    pub q_norm: Tensor,      // [head_dim]
    pub k_proj: Tensor,      // [num_kv_heads * head_dim, hidden]
    pub k_norm: Tensor,      // [head_dim]
    pub v_proj: Tensor,      // [num_kv_heads * head_dim, hidden]
    pub o_proj: Tensor,      // [hidden, num_heads * head_dim]
}

impl FullAttnWeights {
    pub fn load(weights: &BTreeMap<String, Tensor>, prefix: &str, kind: Kind) -> Result<Self> {
        Ok(Self {
            q_proj: tensor(weights, &format!("{prefix}.self_attn.q_proj.weight"))?.to_kind(kind),
            q_norm: tensor(weights, &format!("{prefix}.self_attn.q_norm.weight"))?.to_kind(kind),
            k_proj: tensor(weights, &format!("{prefix}.self_attn.k_proj.weight"))?.to_kind(kind),
            k_norm: tensor(weights, &format!("{prefix}.self_attn.k_norm.weight"))?.to_kind(kind),
            v_proj: tensor(weights, &format!("{prefix}.self_attn.v_proj.weight"))?.to_kind(kind),
            o_proj: tensor(weights, &format!("{prefix}.self_attn.o_proj.weight"))?.to_kind(kind),
        })
    }
}

pub fn full_attention(
    hidden: &Tensor,
    w: &FullAttnWeights,
    config: &Qwen36RuntimeConfig,
    compute_kind: Kind,
) -> Tensor {
    let (batch, seq_len, _hidden) = hidden.size3().unwrap();
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let qkv_dim = num_heads * head_dim;
    let device = hidden.device();

    // Q projection → [batch, seq, num_heads, head_dim*2] → chunk into Q and gate (per-head!)
    // HF: q_proj(hidden).view(B, S, -1, head_dim*2) → chunk(2, dim=-1) → Q[B,S,H,D], gate[B,S,H,D]
    let q_out = hidden.linear::<&Tensor>(&w.q_proj, None);
    let q_out = q_out.view([batch, seq_len, num_heads, head_dim * 2]);
    let parts = q_out.chunk(2, -1);
    let q = parts[0].shallow_clone();   // [batch, seq, num_heads, head_dim]
    let gate = parts[1].shallow_clone(); // [batch, seq, num_heads, head_dim]

    let q = q.transpose(1, 2);   // [batch, heads, seq, head_dim]
    let gate = gate.transpose(1, 2);

    let k = hidden
        .linear::<&Tensor>(&w.k_proj, None)
        .view([batch, seq_len, num_kv_heads, head_dim])
        .transpose(1, 2);
    let v = hidden
        .linear::<&Tensor>(&w.v_proj, None)
        .view([batch, seq_len, num_kv_heads, head_dim])
        .transpose(1, 2);

    let q = rms_norm(&q, &w.q_norm, config.rms_norm_eps);
    let k = rms_norm(&k, &w.k_norm, config.rms_norm_eps);

    let rotary_dim = (head_dim as f64 * config.partial_rotary_factor) as i64;
    let position_ids = Tensor::arange(seq_len, (Kind::Int64, device))
        .unsqueeze(0)
        .expand([batch, seq_len], false);
    let (q, k) = if rotary_dim > 0 {
        apply_rope(&q, &k, &position_ids, rotary_dim, config.rope_theta)
    } else {
        (q, k)
    };

    // GQA: repeat K, V
    let n_rep = num_heads / num_kv_heads;
    let k = k.repeat_interleave_self_int(n_rep, 1, None);
    let v = v.repeat_interleave_self_int(n_rep, 1, None);

    // Scaled dot-product attention (eager — C++ handles internally in train_step)
    let scale = 1.0 / (head_dim as f64).sqrt();
    let attn_output = {
        let attn_weights = (&q * scale).matmul(&k.transpose(-1, -2));
        let causal_mask = Tensor::ones([seq_len, seq_len], (Kind::Bool, device))
            .triu(1)
            .eq(1);
        let attn_weights =
            attn_weights.masked_fill(&causal_mask.unsqueeze(0).unsqueeze(0), f64::NEG_INFINITY);
        attn_weights.softmax(-1, compute_kind).matmul(&v)
    };

    // Output gate (sigmoid)
    let attn_output = attn_output * gate.sigmoid().to_kind(compute_kind);

    let attn_output = attn_output
        .transpose(1, 2)
        .reshape([batch, seq_len, qkv_dim]);
    attn_output.linear::<&Tensor>(&w.o_proj, None)
}

// ──────────────────────────────────────────────────────────────────────
// Linear Attention (Gated Delta Rule)
// ──────────────────────────────────────────────────────────────────────

pub struct LinearAttnWeights {
    pub in_proj_qkv: Tensor,    // [qkv_dim, hidden]
    pub in_proj_z: Tensor,      // [v_dim, hidden]
    pub in_proj_a: Tensor,      // [num_v_heads, hidden]
    pub in_proj_b: Tensor,      // [num_v_heads, hidden]
    pub a_log: Tensor,           // [num_v_heads]
    pub dt_bias: Tensor,         // [num_v_heads]
    pub conv1d_weight: Tensor,   // [qkv_dim, 1, conv_kernel]
    pub norm: Tensor,            // [val_dim]
    pub out_proj: Tensor,       // [hidden, v_dim]
}

impl LinearAttnWeights {
    pub fn load(weights: &BTreeMap<String, Tensor>, prefix: &str, kind: Kind) -> Result<Self> {
        let a_log = tensor(weights, &format!("{prefix}.linear_attn.A_log"))?.to_kind(kind);
        let dt_bias = tensor(weights, &format!("{prefix}.linear_attn.dt_bias"))?.to_kind(kind);
        Ok(Self {
            in_proj_qkv: tensor(weights, &format!("{prefix}.linear_attn.in_proj_qkv.weight"))?.to_kind(kind),
            in_proj_z: tensor(weights, &format!("{prefix}.linear_attn.in_proj_z.weight"))?.to_kind(kind),
            in_proj_a: tensor(weights, &format!("{prefix}.linear_attn.in_proj_a.weight"))?.to_kind(kind),
            in_proj_b: tensor(weights, &format!("{prefix}.linear_attn.in_proj_b.weight"))?.to_kind(kind),
            a_log,
            dt_bias,
            conv1d_weight: tensor(weights, &format!("{prefix}.linear_attn.conv1d.weight"))?.to_kind(kind),
            norm: tensor(weights, &format!("{prefix}.linear_attn.norm.weight"))?.to_kind(kind),
            out_proj: tensor(weights, &format!("{prefix}.linear_attn.out_proj.weight"))?.to_kind(kind),
        })
    }
}

pub fn linear_attention(
    hidden: &Tensor,
    w: &LinearAttnWeights,
    config: &Qwen36RuntimeConfig,
    compute_kind: Kind,
) -> Tensor {
    let (batch, seq_len, _hidden) = hidden.size3().unwrap();
    let num_k_heads = config.linear_num_key_heads;
    let num_v_heads = config.linear_num_value_heads;
    let key_dim = config.linear_key_head_dim;
    let val_dim = config.linear_value_head_dim;
    let conv_kernel = config.linear_conv_kernel_dim;
    let device = hidden.device();

    let q_size = num_k_heads * key_dim;
    let k_size = num_k_heads * key_dim;
    let v_size = num_v_heads * val_dim;
    let qkv_dim = q_size + k_size + v_size;

    // QKV projection
    let qkv = hidden.linear::<&Tensor>(&w.in_proj_qkv, None); // [batch, seq, qkv_dim]

    // Depthwise conv1d (causal, SiLU)
    // conv1d expects [batch, channels, seq_in]
    let qkv_t = qkv.transpose(1, 2); // [batch, qkv_dim, seq]
    let pad = conv_kernel - 1;
    let padding = Tensor::zeros([batch, qkv_dim, pad], (compute_kind, device));
    let padded = Tensor::cat(&[&padding, &qkv_t], 2); // [batch, qkv_dim, seq+pad]
    // Functional conv1d: self.conv1d(weight, bias, stride, padding, dilation, groups)
    let conv_out = padded.conv1d::<&Tensor>(
        &w.conv1d_weight,
        None,
        &[1],
        &[0],
        &[1],
        qkv_dim, // depthwise: groups = channels
    );
    let conv_out = conv_out.narrow(2, 0, seq_len).silu(); // [batch, qkv_dim, seq]
    let qkv_conv = conv_out.transpose(1, 2); // [batch, seq, qkv_dim]

    // Split Q, K, V
    let parts = qkv_conv.split_sizes(&[q_size, k_size, v_size], -1);
    let q = parts[0].shallow_clone().reshape([batch, seq_len, num_k_heads, key_dim]);
    let k = parts[1].shallow_clone().reshape([batch, seq_len, num_k_heads, key_dim]);
    let v = parts[2].shallow_clone().reshape([batch, seq_len, num_v_heads, val_dim]);

    // Project A, B, Z
    let a = hidden.linear::<&Tensor>(&w.in_proj_a, None); // [batch, seq, num_v_heads]
    let b = hidden.linear::<&Tensor>(&w.in_proj_b, None);
    let z = hidden.linear::<&Tensor>(&w.in_proj_z, None); // [batch, seq, v_size]
    let z = z.reshape([batch, seq_len, num_v_heads, val_dim]);

    // g = -exp(A_log) * softplus(a + dt_bias)  — HF qwen3_next convention
    // g is NEGATIVE (decay factor), not exp(-A_log) which would be positive
    let a_log_f = w.a_log.to_kind(Kind::Float);
    let dt_bias_f = w.dt_bias.to_kind(Kind::Float);
    let a_f = a.to_kind(Kind::Float);
    let g = (a_log_f.unsqueeze(0).unsqueeze(0).exp().neg())
        * (a_f + dt_bias_f.unsqueeze(0).unsqueeze(0)).softplus();
    let g = g.to_kind(compute_kind);
    let beta = b.sigmoid().to_kind(compute_kind);

    // Repeat K heads to V heads (num_v_heads / num_k_heads)
    let n_rep = num_v_heads / num_k_heads;
    let q = q.repeat_interleave_self_int(n_rep, 2, None);
    let k = k.repeat_interleave_self_int(n_rep, 2, None);

    // L2 normalize Q, K (HF uses use_qk_l2norm_in_kernel=True)
    let q_norm = q.pow_tensor_scalar(2.0).sum_dim_intlist([-1].as_slice(), true, compute_kind).sqrt().clamp_min(1e-6);
    let k_norm = k.pow_tensor_scalar(2.0).sum_dim_intlist([-1].as_slice(), true, compute_kind).sqrt().clamp_min(1e-6);
    let q = (&q / &q_norm).to_kind(Kind::Float);
    let k = (&k / &k_norm).to_kind(Kind::Float);

    // Scale Q by 1/sqrt(key_dim) — matching HF: scale = 1 / (query.shape[-1] ** 0.5)
    let scale = 1.0 / (key_dim as f64).sqrt();
    let q = (&q * scale).to_kind(Kind::Float);
    let v = v.to_kind(Kind::Float);
    let beta = beta.to_kind(Kind::Float);
    let g = g.to_kind(Kind::Float);

    // Core Gated Delta Rule — recurrent formulation (matching HF torch_recurrent_gated_delta_rule)
    // Transpose from [B, S, H, D] → [B, H, S, D] for recurrent loop
    let q_t = q.transpose(1, 2);
    let k_t = k.transpose(1, 2);
    let v_t = v.transpose(1, 2);
    let g_t = g.transpose(1, 2);  // [B, H, S]
    let beta_t = beta.transpose(1, 2);
    let core_out = gated_delta_rule_recurrent(&q_t, &k_t, &v_t, &g_t, &beta_t);
    // core_out is [B, H, S, D_v] → transpose back to [B, S, H, D_v]
    let core_out = core_out.transpose(1, 2).to_kind(compute_kind);

    // Gated RMSNorm
    let core_flat = core_out.reshape([-1, val_dim]);
    let z_flat = z.reshape([-1, val_dim]).to_kind(compute_kind);
    let normed = rms_norm_gated(&core_flat, &w.norm, &z_flat, config.rms_norm_eps);
    let normed = normed.reshape([batch, seq_len, num_v_heads * val_dim]);

    let result = normed.linear::<&Tensor>(&w.out_proj, None);
    result
}

/// Gated Delta Rule — recurrent formulation matching HF torch_recurrent_gated_delta_rule.
///
/// All inputs are FP32, shapes: q,k: [B, H, S, D_k], v: [B, H, S, D_v], g,beta: [B, H, S]
/// Returns [B, H, S, D_v] in FP32.
fn gated_delta_rule_recurrent(
    q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor,
) -> Tensor {
    // All inputs already FP32, transposed to [B, H, S, dim]
    let (batch, num_heads, seq_len, key_dim) = q.size4().unwrap();
    let val_dim = v.size()[3];
    let device = q.device();

    // HF: g_t = g.exp() → decay factor
    let g_exp = g.exp(); // [B, H, S]

    // State S: [B, H, D_k, D_v] — recurrent state (key_dim × val_dim per head)
    let mut state = Tensor::zeros([batch, num_heads, key_dim, val_dim], (Kind::Float, device));

    // Output: [B, H, S, D_v]
    let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len as usize);

    for i in 0..seq_len {
        let q_i = q.select(2, i);           // [B, H, D_k]
        let k_i = k.select(2, i);           // [B, H, D_k]
        let v_i = v.select(2, i);           // [B, H, D_v]
        let g_i = g_exp.select(2, i).unsqueeze(-1).unsqueeze(-1); // [B, H, 1, 1]
        let beta_i = beta.select(2, i).unsqueeze(-1);            // [B, H, 1]

        // S = S * g_t  (decay)
        state = &state * &g_i;

        // kv_mem = sum(S * k, dim=-2) → [B, H, D_v]
        let kv_mem = (&state * k_i.unsqueeze(-1)).sum_dim_intlist([-2].as_slice(), false, Kind::Float);

        // delta = (v - kv_mem) * beta
        let delta = (&v_i - &kv_mem) * &beta_i;  // [B, H, D_v]

        // S += k ⊗ delta  (update)
        state = &state + k_i.unsqueeze(-1) * delta.unsqueeze(-2);  // [B, H, D_k, D_v]

        // output[i] = sum(S * q, dim=-2) → [B, H, D_v]
        let out_i = (&state * q_i.unsqueeze(-1)).sum_dim_intlist([-2].as_slice(), false, Kind::Float);
        outputs.push(out_i);
    }

    // Stack outputs: [B, H, S, D_v]
    Tensor::stack(&outputs.iter().collect::<Vec<_>>(), 2)
}

// ──────────────────────────────────────────────────────────────────────
// MoE with shared expert + gate + fused gate_up_proj
// ──────────────────────────────────────────────────────────────────────

pub struct MoeWeights {
    pub gate: Tensor,                // [num_experts, hidden]
    pub shared_expert_gate: Tensor,  // [1, hidden]
    pub shared_gate_proj: Tensor,    // [shared_inter, hidden]
    pub shared_up_proj: Tensor,      // [shared_inter, hidden]
    pub shared_down_proj: Tensor,    // [hidden, shared_inter]
    pub experts_gate_up: Tensor,     // [local_experts, 2*intermediate, hidden] (sliced for EP)
    pub experts_down: Tensor,        // [local_experts, hidden, intermediate]
    pub expert_start: usize,         // global index of first local expert (0 for single-GPU)
    pub expert_count: usize,          // number of local experts (256 for single-GPU)
}

impl MoeWeights {
    pub fn load(weights: &BTreeMap<String, Tensor>, prefix: &str, kind: Kind) -> Result<Self> {
        Self::load_ep(weights, prefix, kind, 0, 256)
    }

    /// Load MoE weights with expert sharding.
    /// `expert_start` and `expert_count` specify the local slice.
    pub fn load_ep(
        weights: &BTreeMap<String, Tensor>,
        prefix: &str,
        kind: Kind,
        expert_start: usize,
        expert_count: usize,
    ) -> Result<Self> {
        let full_gate_up = tensor(weights, &format!("{prefix}.mlp.experts.gate_up_proj"))?.to_kind(kind);
        let full_down = tensor(weights, &format!("{prefix}.mlp.experts.down_proj"))?.to_kind(kind);

        // If the tensor already has the right number of experts (pre-narrowed in BTreeMap),
        // or if it's full (256 experts = single GPU), use directly. Otherwise narrow.
        let (experts_gate_up, experts_down) = if full_gate_up.size()[0] == expert_count as i64
            || expert_count == 256
        {
            (full_gate_up, full_down)
        } else {
            (
                full_gate_up.narrow(0, expert_start as i64, expert_count as i64).contiguous(),
                full_down.narrow(0, expert_start as i64, expert_count as i64).contiguous(),
            )
        };

        Ok(Self {
            gate: tensor(weights, &format!("{prefix}.mlp.gate.weight"))?.to_kind(kind),
            shared_expert_gate: tensor(weights, &format!("{prefix}.mlp.shared_expert_gate.weight"))?.to_kind(kind),
            shared_gate_proj: tensor(weights, &format!("{prefix}.mlp.shared_expert.gate_proj.weight"))?.to_kind(kind),
            shared_up_proj: tensor(weights, &format!("{prefix}.mlp.shared_expert.up_proj.weight"))?.to_kind(kind),
            shared_down_proj: tensor(weights, &format!("{prefix}.mlp.shared_expert.down_proj.weight"))?.to_kind(kind),
            experts_gate_up,
            experts_down,
            expert_start,
            expert_count,
        })
    }
}

pub fn swiglu_mlp(input: &Tensor, gate_proj: &Tensor, up_proj: &Tensor, down_proj: &Tensor) -> Tensor {
    let gate = input.linear::<&Tensor>(gate_proj, None);
    let up = input.linear::<&Tensor>(up_proj, None);
    (gate.silu() * up).linear::<&Tensor>(down_proj, None)
}

pub fn moe_forward(
    hidden: &Tensor,
    w: &MoeWeights,
    config: &Qwen36RuntimeConfig,
    compute_kind: Kind,
) -> Tensor {
    let (batch, seq_len, hidden_dim) = hidden.size3().unwrap();
    let top_k = config.num_experts_per_tok as i64;
    let intermediate = config.moe_intermediate_size;
    let device = hidden.device();

    let flat = hidden.reshape([batch * seq_len, hidden_dim]);

    // Router — computed on all ranks (gate is replicated)
    let router_logits = flat.linear::<&Tensor>(&w.gate, None);
    let routing_weights = router_logits.softmax(-1, Kind::Float);
    let (topk_weights, topk_indices) = routing_weights.topk(top_k, -1, true, true);

    let topk_weights = if config.norm_topk_prob {
        let denom = topk_weights
            .sum_dim_intlist([-1].as_slice(), true, Kind::Float)
            .clamp_min(1e-9);
        &topk_weights / &denom
    } else {
        topk_weights
    };
    let topk_weights = topk_weights.to_kind(compute_kind);

    // Routed expert output — only compute for LOCAL experts
    let mut routed_output = Tensor::zeros(flat.size(), (compute_kind, device));

    for k in 0..top_k {
        let expert_indices = topk_indices.select(-1, k as i64);
        let expert_weights = topk_weights.select(-1, k as i64);

        for e_local in 0..w.expert_count as i64 {
            let e_global = w.expert_start as i64 + e_local;
            let mask = expert_indices.eq(e_global).to_kind(compute_kind);
            if mask.sum(Kind::Float).double_value(&[]) > 0.0 {
                let token_indices = mask.nonzero().squeeze_dim(-1);
                if token_indices.size()[0] == 0 {
                    continue;
                }
                let selected = flat.index_select(0, &token_indices);

                let expert_gate_up = w.experts_gate_up.select(0, e_local);
                let expert_down = w.experts_down.select(0, e_local);

                let gate_up = selected.linear::<&Tensor>(&expert_gate_up, None);
                let gu_parts = gate_up.split_sizes(&[intermediate, intermediate], -1);
                let expert_out =
                    (gu_parts[0].shallow_clone().silu() * &gu_parts[1]).linear::<&Tensor>(&expert_down, None);

                let weights = expert_weights
                    .index_select(0, &token_indices)
                    .unsqueeze(-1);
                let weighted = &expert_out * &weights;

                routed_output = routed_output.index_add_(0, &token_indices, &weighted);
            }
        }
    }

    // Shared expert (always computed, replicated on all ranks)
    let shared_out = swiglu_mlp(&flat, &w.shared_gate_proj, &w.shared_up_proj, &w.shared_down_proj);
    let gate_logit = flat.linear::<&Tensor>(&w.shared_expert_gate, None);
    let gate = gate_logit.sigmoid().to_kind(compute_kind);
    let shared_out = (&shared_out * &gate).to_kind(compute_kind);

    // For EP: caller must all-reduce routed_output (shared is already replicated)
    let total = &routed_output + &shared_out;
    total.reshape([batch, seq_len, hidden_dim])
}

/// EP MoE forward — returns only ROUTED expert partial sum (no shared expert).
/// Caller must all-reduce this, then add shared expert separately.
pub fn moe_routed_only_ep(
    hidden: &Tensor,
    w: &MoeWeights,
    config: &Qwen36RuntimeConfig,
    compute_kind: Kind,
) -> Tensor {
    let (batch, seq_len, hidden_dim) = hidden.size3().unwrap();
    let top_k = config.num_experts_per_tok as i64;
    let intermediate = config.moe_intermediate_size;
    let device = hidden.device();

    let flat = hidden.reshape([batch * seq_len, hidden_dim]);

    let router_logits = flat.linear::<&Tensor>(&w.gate, None);
    let routing_weights = router_logits.softmax(-1, Kind::Float);
    let (topk_weights, topk_indices) = routing_weights.topk(top_k, -1, true, true);

    let topk_weights = if config.norm_topk_prob {
        let denom = topk_weights
            .sum_dim_intlist([-1].as_slice(), true, Kind::Float)
            .clamp_min(1e-9);
        &topk_weights / &denom
    } else {
        topk_weights
    };
    let topk_weights = topk_weights.to_kind(compute_kind);

    let mut routed_output = Tensor::zeros(flat.size(), (compute_kind, device));

    for k in 0..top_k {
        let expert_indices = topk_indices.select(-1, k as i64);
        let expert_weights = topk_weights.select(-1, k as i64);

        for e_local in 0..w.expert_count as i64 {
            let e_global = w.expert_start as i64 + e_local;
            let mask = expert_indices.eq(e_global).to_kind(compute_kind);
            if mask.sum(Kind::Float).double_value(&[]) > 0.0 {
                let token_indices = mask.nonzero().squeeze_dim(-1);
                if token_indices.size()[0] == 0 {
                    continue;
                }
                let selected = flat.index_select(0, &token_indices);

                let expert_gate_up = w.experts_gate_up.select(0, e_local);
                let expert_down = w.experts_down.select(0, e_local);

                let gate_up = selected.linear::<&Tensor>(&expert_gate_up, None);
                let gu_parts = gate_up.split_sizes(&[intermediate, intermediate], -1);
                let expert_out =
                    (gu_parts[0].shallow_clone().silu() * &gu_parts[1]).linear::<&Tensor>(&expert_down, None);

                let weights = expert_weights
                    .index_select(0, &token_indices)
                    .unsqueeze(-1);
                let weighted = &expert_out * &weights;

                routed_output = routed_output.index_add_(0, &token_indices, &weighted);
            }
        }
    }

    routed_output.reshape([batch, seq_len, hidden_dim])
}

/// Shared expert forward only (for EP: computed on all ranks, no all-reduce needed).
pub fn moe_shared_only(hidden: &Tensor, w: &MoeWeights, compute_kind: Kind) -> Tensor {
    let (batch, seq_len, hidden_dim) = hidden.size3().unwrap();
    let flat = hidden.reshape([batch * seq_len, hidden_dim]);

    let shared_out = swiglu_mlp(&flat, &w.shared_gate_proj, &w.shared_up_proj, &w.shared_down_proj);
    let gate_logit = flat.linear::<&Tensor>(&w.shared_expert_gate, None);
    let gate = gate_logit.sigmoid().to_kind(compute_kind);
    let shared_out = (&shared_out * &gate).to_kind(compute_kind);

    shared_out.reshape([batch, seq_len, hidden_dim])
}

// ──────────────────────────────────────────────────────────────────────
// Layer dispatch
// ──────────────────────────────────────────────────────────────────────

pub struct Qwen36LayerWeights {
    pub input_norm: Tensor,
    pub post_attention_norm: Tensor,
    pub attn: LayerAttnWeights,
    pub moe: MoeWeights,
}

pub enum LayerAttnWeights {
    Full(FullAttnWeights),
    Linear(LinearAttnWeights),
}

impl Qwen36LayerWeights {
    pub fn load(
        weights: &BTreeMap<String, Tensor>,
        config: &Qwen36RuntimeConfig,
        layer_index: usize,
        kind: Kind,
    ) -> Result<Self> {
        let prefix = format!("{}layers.{layer_index}", config.weight_prefix);
        let input_norm =
            tensor(weights, &format!("{prefix}.input_layernorm.weight"))?.to_kind(kind);
        let post_attention_norm =
            tensor(weights, &format!("{prefix}.post_attention_layernorm.weight"))?.to_kind(kind);

        let attn = match config.layer_types[layer_index] {
            LayerType::FullAttention => {
                LayerAttnWeights::Full(FullAttnWeights::load(weights, &prefix, kind)?)
            }
            LayerType::LinearAttention => {
                LayerAttnWeights::Linear(LinearAttnWeights::load(weights, &prefix, kind)?)
            }
        };

        let moe = MoeWeights::load(weights, &prefix, kind)?;

        Ok(Self { input_norm, post_attention_norm, attn, moe })
    }
}

pub fn qwen36_layer(
    hidden: &Tensor,
    weights: &Qwen36LayerWeights,
    config: &Qwen36RuntimeConfig,
    compute_kind: Kind,
) -> Tensor {
    let input = hidden.to_kind(compute_kind);

    let attn_input =
        rms_norm(&input, &weights.input_norm, config.rms_norm_eps).to_kind(compute_kind);

    let attn_output = match &weights.attn {
        LayerAttnWeights::Full(w) => full_attention(&attn_input, w, config, compute_kind),
        LayerAttnWeights::Linear(w) => linear_attention(&attn_input, w, config, compute_kind),
    };

    let after_attention = &input + &attn_output;

    let moe_input = rms_norm(
        &after_attention,
        &weights.post_attention_norm,
        config.rms_norm_eps,
    )
    .to_kind(compute_kind);

    let moe_output = moe_forward(&moe_input, &weights.moe, config, compute_kind);

    (after_attention + moe_output).to_kind(compute_kind)
}

// ──────────────────────────────────────────────────────────────────────
// Full forward pass
// ──────────────────────────────────────────────────────────────────────

pub fn qwen36_forward_from_ids(
    input_ids: &Tensor,
    weights: &BTreeMap<String, Tensor>,
    config: &Qwen36RuntimeConfig,
    compute_kind: Kind,
) -> Result<Tensor> {
    let embed_prefix = format!("{}embed_tokens.weight", config.weight_prefix);
    let embed_tokens = tensor(weights, &embed_prefix)?.to_kind(compute_kind);
    let final_norm = tensor(weights, &format!("{}norm.weight", config.weight_prefix))?.to_kind(compute_kind);

    let mut hidden = Tensor::embedding(&embed_tokens, input_ids, -1, false, false);

    for layer_index in 0..config.num_hidden_layers {
        let layer = Qwen36LayerWeights::load(weights, config, layer_index, compute_kind)?;
        hidden = qwen36_layer(&hidden, &layer, config, compute_kind);
    }

    let hidden = rms_norm(&hidden, &final_norm, config.rms_norm_eps).to_kind(compute_kind);

    let lm_head = if config.tie_word_embeddings {
        embed_tokens.shallow_clone()
    } else {
        tensor(weights, "lm_head.weight")?.to_kind(compute_kind)
    };

    Ok(hidden.linear::<&Tensor>(&lm_head, None))
}
