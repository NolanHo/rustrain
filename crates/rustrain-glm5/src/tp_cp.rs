//! TP (Tensor Parallel) + CP (Context Parallel) support for GLM-5.2.
//!
//! Rank decomposition: world_size = tp_size × cp_size × ep_size
//!   tp_rank = rank % tp_size
//!   cp_rank = (rank / tp_size) % cp_size
//!   ep_rank = rank / (tp_size * cp_size)
//!
//! TP: attention heads sharded across tp_size ranks. All-reduce after o_proj.
//! CP: sequence split across cp_size ranks. Ring attention K/V exchange.
//! EP: experts sharded across ep_size ranks (unchanged from existing).

use tch::{Kind, Tensor};

/// TP shard info: which attention heads this rank handles.
pub struct Glm5TpShard {
    pub tp_rank: usize,
    pub tp_size: usize,
    pub heads_per_rank: i64,
    pub head_start: i64,
    pub idx_heads_per_rank: i64,
    pub idx_head_start: i64,
}

impl Glm5TpShard {
    pub fn new(tp_rank: usize, tp_size: usize, num_heads: i64, idx_n_heads: i64) -> Self {
        let heads_per_rank = num_heads / tp_size as i64;
        let head_start = tp_rank as i64 * heads_per_rank;
        let idx_heads_per_rank = (idx_n_heads / tp_size as i64).max(1);
        let idx_head_start = tp_rank as i64 * idx_heads_per_rank;
        Self {
            tp_rank,
            tp_size,
            heads_per_rank,
            head_start,
            idx_heads_per_rank,
            idx_head_start,
        }
    }
}

/// TP-sharded attention weights. Weights that are head-parallel are narrowed
/// to only this rank's head range. Low-rank projections (q_a, kv_a) are replicated.
pub struct Glm5TpAttentionWeights {
    // Replicated (full copy on each TP rank)
    pub q_a_proj: Tensor,
    pub q_a_layernorm: Tensor,
    pub kv_a_proj_with_mqa: Tensor,
    pub kv_a_layernorm: Tensor,
    // TP-sharded (only this rank's heads)
    pub q_b_proj: Tensor,       // narrowed by [head_start*(qk_nope+qk_rope), heads_per_rank*(qk_nope+qk_rope)]
    pub kv_b_proj: Tensor,     // narrowed by [head_start*(qk_nope+v_head), heads_per_rank*(qk_nope+v_head)]
    pub o_proj: Tensor,        // narrowed by [head_start*v_head, heads_per_rank*v_head] on dim 1
    // TP-sharded indexer
    pub indexer_wq_b: Option<Tensor>,
    // Replicated indexer
    pub indexer_wk: Option<Tensor>,
    pub indexer_k_norm_weight: Option<Tensor>,
    pub indexer_k_norm_bias: Option<Tensor>,
    pub indexer_weights_proj: Option<Tensor>,
    // FP8 scales (match the sharding of their weight)
    pub q_a_proj_scale: Option<Tensor>,
    pub q_b_proj_scale: Option<Tensor>,
    pub kv_a_proj_scale: Option<Tensor>,
    pub kv_b_proj_scale: Option<Tensor>,
    pub o_proj_scale: Option<Tensor>,
    pub indexer_wq_b_scale: Option<Tensor>,
    pub indexer_wk_scale: Option<Tensor>,
}

impl Clone for Glm5TpAttentionWeights {
    fn clone(&self) -> Self {
        macro_rules! clone_opt {
            ($t:expr) => { $t.as_ref().map(|t| t.shallow_clone()) };
        }
        Self {
            q_a_proj: self.q_a_proj.shallow_clone(),
            q_a_layernorm: self.q_a_layernorm.shallow_clone(),
            kv_a_proj_with_mqa: self.kv_a_proj_with_mqa.shallow_clone(),
            kv_a_layernorm: self.kv_a_layernorm.shallow_clone(),
            q_b_proj: self.q_b_proj.shallow_clone(),
            kv_b_proj: self.kv_b_proj.shallow_clone(),
            o_proj: self.o_proj.shallow_clone(),
            indexer_wq_b: clone_opt!(&self.indexer_wq_b),
            indexer_wk: clone_opt!(&self.indexer_wk),
            indexer_k_norm_weight: clone_opt!(&self.indexer_k_norm_weight),
            indexer_k_norm_bias: clone_opt!(&self.indexer_k_norm_bias),
            indexer_weights_proj: clone_opt!(&self.indexer_weights_proj),
            q_a_proj_scale: clone_opt!(&self.q_a_proj_scale),
            q_b_proj_scale: clone_opt!(&self.q_b_proj_scale),
            kv_a_proj_scale: clone_opt!(&self.kv_a_proj_scale),
            kv_b_proj_scale: clone_opt!(&self.kv_b_proj_scale),
            o_proj_scale: clone_opt!(&self.o_proj_scale),
            indexer_wq_b_scale: clone_opt!(&self.indexer_wq_b_scale),
            indexer_wk_scale: clone_opt!(&self.indexer_wk_scale),
        }
    }
}

impl Glm5TpAttentionWeights {
    /// Load TP-sharded attention weights from the global weight map.
    /// Only loads this rank's slice of head-parallel weights.
    pub fn load_sharded(
        weights: &std::collections::BTreeMap<String, Tensor>,
        layer: usize,
        kind: Kind,
        tp: &Glm5TpShard,
        config: &crate::model::Glm5RuntimeConfig,
    ) -> anyhow::Result<Self> {
        use rustrain_checkpoint::safetensors::tensor;
        use crate::model::KeepIfFp8;

        let p = format!("model.layers.{layer}.self_attn");
        let qk_nope = config.qk_nope_head_dim;
        let qk_rope = config.qk_rope_head_dim;
        let v_head = config.v_head_dim;
        let idx_hd = config.index_head_dim;
        let idx_nh = config.index_n_heads;

        // Replicated weights
        let q_a_proj = tensor(weights, &format!("{p}.q_a_proj.weight"))?.keep_if_fp8(kind);
        let q_a_layernorm = tensor(weights, &format!("{p}.q_a_layernorm.weight"))?.to_kind(kind);
        let kv_a = tensor(weights, &format!("{p}.kv_a_proj_with_mqa.weight"))?.keep_if_fp8(kind);
        let kv_a_ln = tensor(weights, &format!("{p}.kv_a_layernorm.weight"))?.to_kind(kind);

        // TP-sharded: q_b_proj [num_heads*(qk_nope+qk_rope), q_lora_rank]
        let q_b_full = tensor(weights, &format!("{p}.q_b_proj.weight"))?.keep_if_fp8(kind);
        let q_b_row_start = tp.head_start * (qk_nope + qk_rope);
        let q_b_row_len = tp.heads_per_rank * (qk_nope + qk_rope);
        let q_b_proj = q_b_full.narrow(0, q_b_row_start, q_b_row_len);

        // TP-sharded: kv_b_proj [num_heads*(qk_nope+v_head), kv_lora_rank]
        let kv_b_full = tensor(weights, &format!("{p}.kv_b_proj.weight"))?.keep_if_fp8(kind);
        let kv_b_row_start = tp.head_start * (qk_nope + v_head);
        let kv_b_row_len = tp.heads_per_rank * (qk_nope + v_head);
        let kv_b_proj = kv_b_full.narrow(0, kv_b_row_start, kv_b_row_len);

        // TP-sharded: o_proj [hidden_size, num_heads*v_head] — row parallel (narrow dim 1)
        let o_full = tensor(weights, &format!("{p}.o_proj.weight"))?.keep_if_fp8(kind);
        let o_col_start = tp.head_start * v_head;
        let o_col_len = tp.heads_per_rank * v_head;
        let o_proj = o_full.narrow(1, o_col_start, o_col_len);

        // TP-sharded indexer: wq_b [idx_n_heads*idx_head_dim, q_lora_rank]
        let indexer_type = config.indexer_types.get(layer).map(|s| s.as_str()).unwrap_or("full");
        let (indexer_wq_b, indexer_wq_b_scale) = if indexer_type == "full" {
            let wq_b_full = weights
                .get(&format!("{p}.indexer.wq_b.weight"))
                .map(|t| t.keep_if_fp8(kind));
            let wq_b_scale = weights
                .get(&format!("{p}.indexer.wq_b.weight_scale_inv"))
                .map(|t| t.shallow_clone());
            if let Some(wq_full) = wq_b_full {
                let row_start = tp.idx_head_start * idx_hd;
                let row_len = tp.idx_heads_per_rank * idx_hd;
                (Some(wq_full.narrow(0, row_start, row_len)), wq_b_scale)
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Replicated indexer weights
        let indexer_wk = weights.get(&format!("{p}.indexer.wk.weight")).map(|t| t.keep_if_fp8(kind));
        let indexer_k_norm_weight = weights.get(&format!("{p}.indexer.k_norm.weight")).map(|t| t.to_kind(kind));
        let indexer_k_norm_bias = weights.get(&format!("{p}.indexer.k_norm.bias")).map(|t| t.to_kind(kind));
        let indexer_weights_proj = weights.get(&format!("{p}.indexer.weights_proj.weight")).map(|t| {
            // TP-shard by idx_n_heads if needed
            if tp.tp_size > 1 {
                t.narrow(0, tp.idx_head_start as i64, tp.idx_heads_per_rank)
            } else {
                t.to_kind(kind)
            }
        });

        // FP8 scales
        let q_a_proj_scale = weights.get(&format!("{p}.q_a_proj.weight_scale_inv")).map(|t| t.shallow_clone());
        let kv_a_proj_scale = weights.get(&format!("{p}.kv_a_proj_with_mqa.weight_scale_inv")).map(|t| t.shallow_clone());
        let q_b_proj_scale = weights.get(&format!("{p}.q_b_proj.weight_scale_inv")).map(|t| t.shallow_clone());
        let kv_b_proj_scale = weights.get(&format!("{p}.kv_b_proj.weight_scale_inv")).map(|t| t.shallow_clone());
        let o_proj_scale = weights.get(&format!("{p}.o_proj.weight_scale_inv")).map(|t| t.shallow_clone());
        let indexer_wk_scale = weights.get(&format!("{p}.indexer.wk.weight_scale_inv")).map(|t| t.shallow_clone());

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            kv_a_proj_with_mqa: kv_a,
            kv_a_layernorm: kv_a_ln,
            q_b_proj,
            kv_b_proj,
            o_proj,
            indexer_wq_b,
            indexer_wk,
            indexer_k_norm_weight,
            indexer_k_norm_bias,
            indexer_weights_proj,
            q_a_proj_scale,
            q_b_proj_scale,
            kv_a_proj_scale,
            o_proj_scale,
            kv_b_proj_scale,
            indexer_wq_b_scale,
            indexer_wk_scale,
        })
    }
}

// ── TP+CP Attention Forward ──

/// TP+CP sharded DSA attention.
///
/// - `tp`: tensor parallel shard info
/// - `cp_rank`, `cp_size`: context parallel position
/// - `nccl_comm`: for ring K/V exchange (None if cp_size == 1)
/// - `s_local`: local sequence length (S / cp_size)
/// - `seq_global`: global sequence length (for causal mask + RoPE)
///
/// Returns attention output for local query positions [batch, s_local, heads_per_rank * v_head].
pub fn glm5_dsa_attention_tp_cp(
    input: &Tensor,
    attn: &Glm5TpAttentionWeights,
    indexer_weights: &Glm5TpAttentionWeights,
    config: &crate::model::Glm5RuntimeConfig,
    index_share_state: &mut Option<crate::model::IndexShareState>,
    layer: usize,
    tp: &Glm5TpShard,
    cp_rank: usize,
    cp_size: usize,
    ep_rank: usize,
    nccl_comm: Option<&rustrain_nccl::nccl::NcclPersistentComm>,
) -> Tensor {
    use crate::model::{glm5_safe_linear, rms_norm, rms_norm_with_bias, rope_cos_sin, apply_rotary_dispatch};

    let compute_kind = input.kind();
    let batch = input.size()[0];
    let s_local = input.size()[1];
    let seq_global = s_local * cp_size as i64;
    let nh_local = tp.heads_per_rank; // local attention heads
    let idx_nh_local = tp.idx_heads_per_rank; // local indexer heads
    let qk_nope = config.qk_nope_head_dim;
    let qk_rope = config.qk_rope_head_dim;
    let v_head = config.v_head_dim;
    let kv_lora = config.kv_lora_rank;
    let idx_head_dim = config.index_head_dim;
    let idx_topk = config.index_topk;

    // ── Q/K/V projections (TP-sharded) ──
    let q_a = glm5_safe_linear(input, &attn.q_a_proj, attn.q_a_proj_scale.as_ref());
    let q_a_normed = rms_norm(&q_a, &attn.q_a_layernorm.to_kind(compute_kind), config.rms_norm_eps);
    let q_b = glm5_safe_linear(&q_a_normed, &attn.q_b_proj, attn.q_b_proj_scale.as_ref());
    // q_b: [batch, s_local, heads_per_rank * (qk_nope+qk_rope)]
    let q = q_b
        .reshape([batch, s_local, nh_local, qk_nope + qk_rope])
        .transpose(1, 2);
    let q_nope = q.narrow(-1, 0, qk_nope);
    let q_rope = q.narrow(-1, qk_nope, qk_rope);

    let kv_a = glm5_safe_linear(input, &attn.kv_a_proj_with_mqa, attn.kv_a_proj_scale.as_ref());
    let kv_lora_raw = kv_a.narrow(-1, 0, kv_lora);
    let k_rope = kv_a.narrow(-1, kv_lora, qk_rope);
    let kv_lora_part = rms_norm(&kv_lora_raw, &attn.kv_a_layernorm.to_kind(compute_kind), config.rms_norm_eps);
    let kv_b = glm5_safe_linear(&kv_lora_part, &attn.kv_b_proj, attn.kv_b_proj_scale.as_ref());
    let kv_b = kv_b.reshape([batch, s_local, nh_local, qk_nope + v_head]);
    let k_nope = kv_b.narrow(-1, 0, qk_nope).transpose(1, 2);
    let v = kv_b.narrow(-1, qk_nope, v_head).transpose(1, 2);

    // RoPE — use global positions for CP correctness
    let k_rope_expanded = k_rope
        .unsqueeze(2)
        .transpose(1, 2)
        .expand([batch, nh_local, s_local, qk_rope], false);
    let rope_offset = cp_rank as i64 * s_local;
    let (cos, sin) = rope_cos_sin(seq_global as usize, qk_rope, config.rope_theta, input.device());
    let cos = cos.narrow(0, rope_offset, s_local as i64).to_kind(compute_kind);
    let sin = sin.narrow(0, rope_offset, s_local as i64).to_kind(compute_kind);
    let q_rope_rotated = apply_rotary_dispatch(&q_rope, &cos, &sin, config.rope_interleave);
    let k_rope_rotated = apply_rotary_dispatch(&k_rope_expanded, &cos, &sin, config.rope_interleave);

    let q_full = Tensor::cat(&[&q_nope, &q_rope_rotated], -1); // [B, H_local, S_local, d]
    let k_full = Tensor::cat(&[&k_nope, &k_rope_rotated], -1); // [B, H_local, S_local, d]
    let attn_scale = 1.0 / ((qk_nope + qk_rope) as f64).sqrt();

    // ── DSA Indexer (TP-sharded, CP-local) ──
    let should_compute_topk = !config.should_skip_topk(layer)
        && (index_share_state.is_none() || layer % (config.index_topk_freq as usize) == 0);

    if let (Some(wq_b), Some(wk), Some(k_norm_w), Some(k_norm_b), Some(weights_proj)) = (
        &indexer_weights.indexer_wq_b,
        &indexer_weights.indexer_wk,
        &indexer_weights.indexer_k_norm_weight,
        &indexer_weights.indexer_k_norm_bias,
        &indexer_weights.indexer_weights_proj,
    ) {
        if should_compute_topk {
            let idx_q = glm5_safe_linear(&q_a, wq_b, indexer_weights.indexer_wq_b_scale.as_ref());
            let idx_q = idx_q
                .reshape([batch, s_local, idx_nh_local, idx_head_dim])
                .transpose(1, 2);

            let idx_k_raw = glm5_safe_linear(input, wk, indexer_weights.indexer_wk_scale.as_ref());
            let idx_k = rms_norm_with_bias(&idx_k_raw, &k_norm_w.to_kind(compute_kind), &k_norm_b.to_kind(compute_kind), config.rms_norm_eps);
            let idx_k_expanded = idx_k
                .unsqueeze(1)
                .expand([batch, idx_nh_local, s_local, idx_head_dim], false);

            let (idx_q_rotated, idx_k_rotated) = if config.indexer_rope_interleave {
                let (cos_i, sin_i) = rope_cos_sin(seq_global as usize, idx_head_dim, config.rope_theta, input.device());
                let cos_i = cos_i.narrow(0, rope_offset, s_local as i64).to_kind(compute_kind);
                let sin_i = sin_i.narrow(0, rope_offset, s_local as i64).to_kind(compute_kind);
                let q_r = apply_rotary_dispatch(&idx_q, &cos_i, &sin_i, config.indexer_rope_interleave);
                let k_r = apply_rotary_dispatch(&idx_k_expanded, &cos_i, &sin_i, config.indexer_rope_interleave);
                (q_r, k_r)
            } else {
                (idx_q.shallow_clone(), idx_k_expanded.shallow_clone())
            };

            let idx_scale = 1.0 / (idx_head_dim as f64).sqrt();

            // For CP: topk only over local keys (s_local tokens).
            // The global topk would require cross-rank merge, but for correctness
            // we compute topk over local keys only — each CP rank attends to its local
            // KV window. This is an approximation; full cross-rank topk can be added later.
            let actual_topk = idx_topk.min(s_local);

            // Chunked score computation (reuse logic from base function)
            let score_chunk = 512_i64;
            let topk_indices = if s_local <= score_chunk {
                let idx_scores = idx_q_rotated.matmul(&idx_k_rotated.transpose(-2, -1)) * idx_scale;
                let idx_scores_expanded = if idx_nh_local != nh_local {
                    idx_scores
                        .mean_dim([1].as_slice(), true, compute_kind)
                        .expand([batch, nh_local, s_local, s_local], false)
                } else {
                    idx_scores
                };
                let (_, indices) = idx_scores_expanded.topk(actual_topk, -1, true, true);
                indices
            } else {
                let mut best_scores: Option<Tensor> = None;
                let mut best_indices: Option<Tensor> = None;
                for k_start in (0..s_local).step_by(score_chunk as usize) {
                    let k_end = (k_start + score_chunk).min(s_local);
                    let k_len = k_end - k_start;
                    let idx_k_chunk = idx_k_rotated.narrow(-2, k_start, k_len);
                    let scores_chunk = idx_q_rotated.matmul(&idx_k_chunk.transpose(-2, -1)) * idx_scale;
                    let scores_chunk = if idx_nh_local != nh_local {
                        scores_chunk
                            .mean_dim([1].as_slice(), true, compute_kind)
                            .expand([batch, nh_local, s_local, k_len], false)
                    } else {
                        scores_chunk
                    };
                    let local_topk = actual_topk.min(k_len);
                    let (ls, li) = scores_chunk.topk(local_topk, -1, true, true);
                    let offset = Tensor::full(li.size(), k_start as f64, (li.kind(), li.device()));
                    let li = &li.to_kind(Kind::Float) + &offset;
                    match (&best_scores, &best_indices) {
                        (Some(bs), Some(bi)) => {
                            let merged = Tensor::cat(&[bs, &ls], -1);
                            let merged_idx = Tensor::cat(&[bi, &li.to_kind(Kind::Int64)], -1);
                            let (s, pos) = merged.topk(actual_topk, -1, true, true);
                            best_scores = Some(s);
                            best_indices = Some(merged_idx.gather(-1, &pos, false));
                        }
                        _ => {
                            best_scores = Some(ls);
                            best_indices = Some(li.to_kind(Kind::Int64));
                        }
                    }
                }
                best_indices.unwrap()
            };

            let idx_bias_keys = glm5_safe_linear(input, weights_proj, None);
            let idx_bias_keys = idx_bias_keys
                .reshape([batch, s_local, idx_nh_local])
                .transpose(1, 2);

            *index_share_state = Some(crate::model::IndexShareState {
                topk_indices,
                idx_bias_keys,
                source_layer: layer,
            });
        }
    } else {
        *index_share_state = None;
    }

    // ── Attention with optional CP ring ──
    let context = if cp_size == 1 {
        // ── TP only (no CP): same as base but with local heads ──
        if let Some(state) = index_share_state {
            let actual_topk = state.topk_indices.size()[state.topk_indices.size().len() - 1];
            let bias_per_key = if idx_nh_local != nh_local {
                state
                    .idx_bias_keys
                    .mean_dim([1].as_slice(), true, compute_kind)
                    .expand([batch as i64, nh_local, s_local], false)
            } else {
                state.idx_bias_keys.shallow_clone()
            };

            let attn_chunk: i64 = if s_local > 2048 { 512 } else { s_local };
            if attn_chunk >= s_local {
                let sparse_mask = {
                    let mut m = Tensor::zeros(
                        [batch as i64, nh_local, s_local, s_local],
                        (compute_kind, input.device()),
                    );
                    let ones = Tensor::ones(
                        [batch as i64, nh_local, s_local, actual_topk],
                        (compute_kind, input.device()),
                    );
                    let _ = m.scatter_(-1, &state.topk_indices, &ones);
                    m
                };
                let causal_f = {
                    let cm = Tensor::ones([s_local, s_local], (Kind::Bool, input.device())).triu(1);
                    cm.unsqueeze(0).unsqueeze(0)
                        .expand([batch as i64, nh_local, s_local, s_local], false)
                        .to_kind(compute_kind)
                };
                let combined = &sparse_mask * (1.0 - &causal_f);
                drop(sparse_mask);
                drop(causal_f);
                let bias = bias_per_key
                    .unsqueeze(2)
                    .expand([batch as i64, nh_local, s_local, s_local], false)
                    .to_kind(compute_kind);
                let bias = &bias + (&combined - 1.0) * f64::NEG_INFINITY;
                drop(combined);
                Tensor::scaled_dot_product_attention(
                    &q_full, &k_full, &v,
                    Some(&bias), 0.0, false, Some(attn_scale), false,
                )
            } else {
                let n_chunks = (s_local + attn_chunk - 1) / attn_chunk;
                let mut outputs: Vec<Tensor> = Vec::with_capacity(n_chunks as usize);
                for q_start in (0..s_local).step_by(attn_chunk as usize) {
                    let q_end = (q_start + attn_chunk).min(s_local);
                    let q_len = q_end - q_start;
                    let q_chunk = q_full.narrow(2, q_start, q_len);
                    let sparse_mask = {
                        let chunk_topk = state.topk_indices.narrow(2, q_start, q_len);
                        let mut m = Tensor::zeros(
                            [batch as i64, nh_local, q_len, s_local],
                            (compute_kind, input.device()),
                        );
                        let ones = Tensor::ones(
                            [batch as i64, nh_local, q_len, actual_topk],
                            (compute_kind, input.device()),
                        );
                        let _ = m.scatter_(-1, &chunk_topk, &ones);
                        m
                    };
                    let causal_f = {
                        let q_pos = (Tensor::arange(q_len, (Kind::Int64, input.device())) + q_start).to_kind(compute_kind);
                        let k_pos = Tensor::arange(s_local, (Kind::Int64, input.device())).to_kind(compute_kind);
                        let diff = k_pos.unsqueeze(0) - q_pos.unsqueeze(1);
                        diff.gt(0.0).unsqueeze(0).unsqueeze(0)
                            .expand([batch as i64, nh_local, q_len, s_local], false)
                            .to_kind(compute_kind)
                    };
                    let combined = &sparse_mask * (1.0 - &causal_f);
                    drop(sparse_mask);
                    drop(causal_f);
                    let bias = bias_per_key
                        .unsqueeze(2)
                        .expand([batch as i64, nh_local, q_len, s_local], false)
                        .to_kind(compute_kind);
                    let bias = &bias + (&combined - 1.0) * f64::NEG_INFINITY;
                    drop(combined);
                    let chunk_out = Tensor::scaled_dot_product_attention(
                        &q_chunk, &k_full, &v,
                        Some(&bias), 0.0, false, Some(attn_scale), false,
                    );
                    drop(bias);
                    outputs.push(chunk_out);
                }
                let refs: Vec<&Tensor> = outputs.iter().collect();
                Tensor::cat(&refs, 2)
            }
        } else {
            Tensor::scaled_dot_product_attention::<&Tensor>(
                &q_full, &k_full, &v, None, 0.0, true, Some(attn_scale), false,
            )
        }
    } else {
        // ── Ring attention (CP > 1) ──
        // Each rank has local K/V [B, H, S_local, d]. We rotate K/V blocks through
        // CP ranks via ring_send_recv, computing partial attention for each K/V block.
        //
        // Online softmax (FlashAttention style): maintain running max and both
        // numerator (running_sum) and denominator (running_denom) across blocks.
        //
        // For DSA sparse mask: topk_indices refer to GLOBAL key positions.
        // Key position j belongs to CP rank (j / s_local). For a given K/V block
        // from rank `peer`, global key positions are [peer * s_local, (peer+1) * s_local).
        // We filter topk_indices to only keep indices within this range.

        let comm = nccl_comm.expect("CP requires NCCL communicator");
        let actual_topk = index_share_state
            .as_ref()
            .map(|s| s.topk_indices.size()[s.topk_indices.size().len() - 1])
            .unwrap_or(0);

        // Pre-compute per-key bias for local keys
        let bias_per_key = if let Some(state) = index_share_state {
            if idx_nh_local != nh_local {
                state
                    .idx_bias_keys
                    .mean_dim([1].as_slice(), true, compute_kind)
                    .expand([batch as i64, nh_local, s_local], false)
            } else {
                state.idx_bias_keys.shallow_clone()
            }
        } else {
            Tensor::zeros([], (compute_kind, input.device()))
        };

        // Online softmax state: running max, numerator, denominator
        let mut running_max: Option<Tensor> = None;    // [B, H, S_local, 1]
        let mut running_num: Option<Tensor> = None;     // [B, H, S_local, v_head]
        let mut running_denom: Option<Tensor> = None;   // [B, H, S_local, 1]

        // Current K/V block (start with local)
        let mut k_current = k_full.shallow_clone();
        let mut v_current = v.shallow_clone();

        for step in 0..cp_size {
            let peer = (cp_rank + step) % cp_size;
            let k_start_global = peer as i64 * s_local;

            // Build attention bias for this K/V block
            let bias = if let Some(state) = index_share_state {
                let global_topk = &state.topk_indices;
                let in_range = global_topk
                    .ge(k_start_global)
                    .logical_and(&global_topk.lt(k_start_global + s_local));
                let local_topk = (global_topk - k_start_global)
                    .masked_fill(&in_range.logical_not(), 0);
                let valid = in_range.to_kind(compute_kind);

                let mut sparse_mask = Tensor::zeros(
                    [batch as i64, nh_local, s_local, s_local],
                    (compute_kind, input.device()),
                );
                let weighted_ones = Tensor::ones(
                    [batch as i64, nh_local, s_local, actual_topk],
                    (compute_kind, input.device()),
                ) * &valid;
                let _ = sparse_mask.scatter_(-1, &local_topk, &weighted_ones);
                drop(local_topk);
                drop(valid);

                let causal_f = {
                    let q_pos = (Tensor::arange(s_local, (Kind::Int64, input.device())) + (cp_rank as i64 * s_local)).to_kind(compute_kind);
                    let k_pos = (Tensor::arange(s_local, (Kind::Int64, input.device())) + k_start_global).to_kind(compute_kind);
                    let diff = k_pos.unsqueeze(0) - q_pos.unsqueeze(1);
                    diff.gt(0.0).unsqueeze(0).unsqueeze(0)
                        .expand([batch as i64, nh_local, s_local, s_local], false)
                        .to_kind(compute_kind)
                };

                let combined: Tensor = &sparse_mask * (1.0 - &causal_f);
                let mask = combined.eq(0.0); // bool: true where masked
                drop(sparse_mask);
                drop(causal_f);

                if peer == cp_rank {
                    let bias_base = bias_per_key
                        .unsqueeze(2)
                        .expand([batch as i64, nh_local, s_local, s_local], false)
                        .to_kind(compute_kind);
                    Some(bias_base.masked_fill(&mask, f64::NEG_INFINITY))
                } else {
                    // Remote keys — only sparse+causal mask, no per-key bias
                    let bias_base = Tensor::zeros(
                        [batch as i64, nh_local, s_local, s_local],
                        (compute_kind, input.device()),
                    );
                    Some(bias_base.masked_fill(&mask, f64::NEG_INFINITY))
                }
            } else {
                let q_pos = (Tensor::arange(s_local, (Kind::Int64, input.device())) + (cp_rank as i64 * s_local)).to_kind(compute_kind);
                let k_pos = (Tensor::arange(s_local, (Kind::Int64, input.device())) + k_start_global).to_kind(compute_kind);
                let diff = k_pos.unsqueeze(0) - q_pos.unsqueeze(1);
                let causal_mask = diff.gt(0.0); // bool: true where masked (future)
                let bias_base = Tensor::zeros(
                    [s_local, s_local],
                    (compute_kind, input.device()),
                );
                let bias = bias_base.masked_fill(&causal_mask, f64::NEG_INFINITY);
                Some(bias.unsqueeze(0).unsqueeze(0)
                    .expand([batch as i64, nh_local, s_local, s_local], false)
                    .to_kind(compute_kind))
            };

            // Compute attention scores for this K/V block
            let scores = q_full.matmul(&k_current.transpose(-2, -1)) * attn_scale;
            let scores = if let Some(b) = &bias {
                scores + b
            } else {
                scores
            };
            drop(bias);

            // Online softmax update (FlashAttention-2 style)
            // For each query position, track:
            //   m = running max of scores
            //   num = sum(exp(scores - m) * v)
            //   denom = sum(exp(scores - m))
            // Use amax + unsqueeze to get [B, H, S_local, 1] max tensor
            let block_max = scores.amax([-1], true); // [B, H, S_local, 1]
            let exp_scores = (&scores - &block_max).exp();
            let block_num = exp_scores.matmul(&v_current);  // [B, H, S_local, v_head]
            let block_denom = exp_scores.sum_dim_intlist([-1].as_slice(), true, compute_kind); // [B, H, S_local, 1]

            match (&mut running_max, &mut running_num, &mut running_denom) {
                (Some(rm), Some(rn), Some(rd)) => {
                    // element-wise maximum via f_max_other
                    let new_max = rm.f_max_other(&block_max).unwrap();
                    let exp_old = (&*rm - &new_max).exp();
                    let exp_new = (&block_max - &new_max).exp();
                    *rn = &(&(*rn) * &exp_old) + &(&block_num * &exp_new);
                    *rd = &(&(*rd) * &exp_old) + &(&block_denom * &exp_new);
                    *rm = new_max;
                }
                (None, None, None) => {
                    running_max = Some(block_max);
                    running_num = Some(block_num);
                    running_denom = Some(block_denom);
                }
                _ => unreachable!(),
            }

            // Ring exchange K/V to next rank
            if step < cp_size - 1 {
                // Convert CP-local peers to global NCCL ranks:
                // global_rank = ep_rank * (tp_size * cp_size) + cp_rank * tp_size + tp_rank
                let tp_size = tp.tp_size;
                let next_cp = (cp_rank + 1) % cp_size;
                let prev_cp = (cp_rank + cp_size - 1) % cp_size;
                let send_peer = ep_rank * (tp_size * cp_size) + next_cp * tp_size + tp.tp_rank;
                let recv_peer = ep_rank * (tp_size * cp_size) + prev_cp * tp_size + tp.tp_rank;
                let k_recv = comm.ring_send_recv(&k_current, send_peer, recv_peer)
                    .expect("NCCL ring K exchange failed");
                let v_recv = comm.ring_send_recv(&v_current, send_peer, recv_peer)
                    .expect("NCCL ring V exchange failed");
                drop(k_current);
                drop(v_current);
                k_current = k_recv;
                v_current = v_recv;
            }
        }

        // Final: normalize numerator by denominator
        let running_num = running_num.unwrap();
        let running_denom = running_denom.unwrap();
        running_num / running_denom.clamp_min(1e-20)
    };

    let context = context
        .transpose(1, 2)
        .reshape([batch, s_local, nh_local * v_head]);

    glm5_safe_linear(&context, &attn.o_proj, attn.o_proj_scale.as_ref())
}
