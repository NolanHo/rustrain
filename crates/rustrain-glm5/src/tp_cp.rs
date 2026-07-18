//! TP (Tensor Parallel) + CP (Context Parallel) support for GLM-5.2.
//!
//! The TP+EP session is still gated until it uses the explicit Megatron rank
//! decomposition below. EP is not a Cartesian axis multiplied by TP: with
//! ETP=1 and CP=1, dense groups use `world = TP × DP`, while expert groups use
//! `world = EP × expert-DP` over the same stage-local ranks.
//!
//! TP: attention heads sharded across tp_size ranks. All-reduce after o_proj.
//! CP: sequence split across cp_size ranks. Ring attention K/V exchange.
//! The legacy TP/CP session still has a Cartesian EP decomposition for its
//! non-MTP path; native MTP rejects combined TP+EP until the rank contract
//! above is wired into its communicators.

use anyhow::{Context, Result, bail};
use tch::{Kind, Tensor};

use crate::lora::{Glm5LoraRegistry, Glm5LoraTargetModule};

/// Stage-local rank coordinates for Megatron's default `tp-cp-ep-dp-pp`
/// ordering with CP=1 and expert tensor parallelism fixed to one. These
/// coordinates are intentionally separate because dense DP and expert DP are
/// different groups when `TP > 1 && EP > 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm5MegatronRankCoordinates {
    pub rank: usize,
    pub tp_rank: usize,
    pub dense_dp_rank: usize,
    pub ep_rank: usize,
    pub expert_dp_rank: usize,
    pub dense_dp_size: usize,
    pub expert_dp_size: usize,
    pub tp_size: usize,
    pub ep_size: usize,
}

pub fn glm5_megatron_rank_coordinates(
    rank: usize,
    world_size: usize,
    tp_size: usize,
    ep_size: usize,
) -> Result<Glm5MegatronRankCoordinates> {
    if world_size == 0 || rank >= world_size {
        bail!("rank {rank} must be within positive world_size {world_size}");
    }
    if tp_size == 0 || ep_size == 0 {
        bail!("Megatron TP and EP sizes must be positive");
    }
    if world_size % tp_size != 0 || world_size % ep_size != 0 {
        bail!("world_size {world_size} must be divisible by both TP {tp_size} and EP {ep_size}");
    }
    Ok(Glm5MegatronRankCoordinates {
        rank,
        tp_rank: rank % tp_size,
        dense_dp_rank: rank / tp_size,
        ep_rank: rank % ep_size,
        expert_dp_rank: rank / ep_size,
        dense_dp_size: world_size / tp_size,
        expert_dp_size: world_size / ep_size,
        tp_size,
        ep_size,
    })
}

pub fn glm5_megatron_dense_dp_group(coords: Glm5MegatronRankCoordinates) -> Vec<usize> {
    (0..coords.dense_dp_size)
        .map(|dp| dp * coords.tp_size + coords.tp_rank)
        .collect()
}

pub fn glm5_megatron_expert_ep_group(coords: Glm5MegatronRankCoordinates) -> Vec<usize> {
    let start = coords.expert_dp_rank * coords.ep_size;
    (start..start + coords.ep_size).collect()
}

pub fn glm5_megatron_expert_dp_group(coords: Glm5MegatronRankCoordinates) -> Vec<usize> {
    (0..coords.expert_dp_size)
        .map(|dp| dp * coords.ep_size + coords.ep_rank)
        .collect()
}

fn cp_ring_send_recv_autograd(
    tensor: &Tensor,
    comm: &rustrain_nccl::nccl::NcclPersistentComm,
    send_peer: usize,
    recv_peer: usize,
) -> Result<Tensor> {
    rustrain_deepseek_v4::fp8_kernel::glm5_nccl_ring_autograd_cpp(
        tensor,
        comm.raw_comm_ptr(),
        send_peer,
        recv_peer,
    )
}

fn cp_ring_send_recv_kv_autograd(
    key: &Tensor,
    value: &Tensor,
    comm: &rustrain_nccl::nccl::NcclPersistentComm,
    send_peer: usize,
    recv_peer: usize,
) -> Result<(Tensor, Tensor)> {
    rustrain_deepseek_v4::fp8_kernel::glm5_nccl_kv_ring_autograd_cpp(
        key,
        value,
        comm.raw_comm_ptr(),
        send_peer,
        recv_peer,
    )
}

fn shard_block_scale(scale: &Tensor, start: i64, len: i64, axis: i64) -> Tensor {
    const BLOCK: i64 = 128;
    assert!(
        start % BLOCK == 0 && len % BLOCK == 0,
        "TP FP8 shard must align to 128-element scale blocks"
    );
    scale.narrow(axis, start / BLOCK, len / BLOCK)
}

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

fn validate_tp_coordinates(tp_rank: usize, tp_size: usize) -> Result<()> {
    if tp_size == 0 {
        bail!("tensor parallel size must be positive");
    }
    if tp_rank >= tp_size {
        bail!("tensor parallel rank {tp_rank} is outside world size {tp_size}");
    }
    Ok(())
}

fn tp_partition_range(
    total: i64,
    tp_rank: usize,
    tp_size: usize,
    name: &str,
) -> Result<(i64, i64)> {
    validate_tp_coordinates(tp_rank, tp_size)?;
    if total <= 0 {
        bail!("{name} must be positive, got {total}");
    }
    let tp_size_i64 = i64::try_from(tp_size).context("tensor parallel size does not fit i64")?;
    if total % tp_size_i64 != 0 {
        bail!("{name}={total} must be divisible by tensor parallel size {tp_size}");
    }
    let per_rank = total / tp_size_i64;
    let start = i64::try_from(tp_rank)
        .context("tensor parallel rank does not fit i64")?
        .checked_mul(per_rank)
        .context("tensor parallel partition offset overflow")?;
    Ok((start, per_rank))
}

fn checked_block_scale_shard(
    scale: &Tensor,
    start: i64,
    len: i64,
    axis: i64,
    name: &str,
) -> Result<Tensor> {
    if scale.numel() == 1 {
        return Ok(scale.shallow_clone());
    }
    const BLOCK: i64 = 128;
    if start % BLOCK != 0 || len % BLOCK != 0 {
        bail!(
            "{name} FP8 shard [{start}, {}) must align to {BLOCK}-element scale blocks",
            start + len
        );
    }
    let sizes = scale.size();
    let axis_usize = usize::try_from(axis).context("negative FP8 scale axis")?;
    if axis_usize >= sizes.len() {
        bail!(
            "{name} FP8 scale rank {} does not contain axis {axis}",
            sizes.len()
        );
    }
    let scale_start = start / BLOCK;
    let scale_len = len / BLOCK;
    if scale_start + scale_len > sizes[axis_usize] {
        bail!(
            "{name} FP8 scale axis {} is too short for shard ending at block {}",
            sizes[axis_usize],
            scale_start + scale_len
        );
    }
    Ok(scale.narrow(axis, scale_start, scale_len))
}

/// One Megatron-style tensor-parallel linear weight partition.
///
/// A column-parallel linear shards the output rows of a PyTorch `[out, in]`
/// weight. A row-parallel linear shards its input columns. The optional FP8
/// block scale is sliced on the same logical axis.
pub struct Glm5TpLinearShard {
    pub weight: Tensor,
    pub weight_scale: Option<Tensor>,
    pub start: i64,
    pub len: i64,
}

/// Shard a `[out_features, in_features]` weight over output features.
pub fn shard_column_parallel_linear(
    weight: &Tensor,
    weight_scale: Option<&Tensor>,
    tp_rank: usize,
    tp_size: usize,
    name: &str,
) -> Result<Glm5TpLinearShard> {
    let shape = weight.size();
    if shape.len() != 2 {
        bail!("{name} must be rank 2, got shape {shape:?}");
    }
    let (start, len) = tp_partition_range(shape[0], tp_rank, tp_size, name)?;
    let weight_scale = weight_scale
        .map(|scale| checked_block_scale_shard(scale, start, len, 0, name))
        .transpose()?;
    Ok(Glm5TpLinearShard {
        weight: weight.narrow(0, start, len),
        weight_scale,
        start,
        len,
    })
}

/// Shard a `[out_features, in_features]` weight over input features.
pub fn shard_row_parallel_linear(
    weight: &Tensor,
    weight_scale: Option<&Tensor>,
    tp_rank: usize,
    tp_size: usize,
    name: &str,
) -> Result<Glm5TpLinearShard> {
    let shape = weight.size();
    if shape.len() != 2 {
        bail!("{name} must be rank 2, got shape {shape:?}");
    }
    let (start, len) = tp_partition_range(shape[1], tp_rank, tp_size, name)?;
    let weight_scale = weight_scale
        .map(|scale| checked_block_scale_shard(scale, start, len, 1, name))
        .transpose()?;
    Ok(Glm5TpLinearShard {
        weight: weight.narrow(1, start, len),
        weight_scale,
        start,
        len,
    })
}

/// Vocabulary interval owned by one tensor-parallel rank. The interval is
/// computed from the padded vocabulary, exactly like Megatron's
/// `VocabUtility`; token IDs remain bounded by `global_vocab_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm5TpVocabRange {
    pub vocab_start: i64,
    pub vocab_end: i64,
    pub global_vocab_size: i64,
    pub padded_vocab_size: i64,
}

impl Glm5TpVocabRange {
    pub fn new(
        global_vocab_size: i64,
        checkpoint_vocab_rows: i64,
        tp_rank: usize,
        tp_size: usize,
    ) -> Result<Self> {
        validate_tp_coordinates(tp_rank, tp_size)?;
        if global_vocab_size <= 0 {
            bail!("global vocabulary size must be positive, got {global_vocab_size}");
        }
        if checkpoint_vocab_rows < global_vocab_size {
            bail!(
                "checkpoint vocabulary has {checkpoint_vocab_rows} rows, smaller than global vocabulary size {global_vocab_size}"
            );
        }
        let tp_size_i64 =
            i64::try_from(tp_size).context("tensor parallel size does not fit i64")?;
        let padded_vocab_size = checkpoint_vocab_rows
            .checked_add(tp_size_i64 - 1)
            .context("padded vocabulary size overflow")?
            / tp_size_i64
            * tp_size_i64;
        let (vocab_start, per_rank) = tp_partition_range(
            padded_vocab_size,
            tp_rank,
            tp_size,
            "padded vocabulary size",
        )?;
        Ok(Self {
            vocab_start,
            vocab_end: vocab_start + per_rank,
            global_vocab_size,
            padded_vocab_size,
        })
    }
}

pub struct Glm5TpVocabShard {
    pub weight: Tensor,
    pub weight_scale: Option<Tensor>,
    pub range: Glm5TpVocabRange,
}

fn pad_vocab_rows(weight: &Tensor, padded_vocab_size: i64, name: &str) -> Result<Tensor> {
    let shape = weight.size();
    if shape.len() != 2 {
        bail!("{name} must be rank 2, got shape {shape:?}");
    }
    if shape[0] > padded_vocab_size {
        bail!(
            "{name} has {} rows, larger than requested padded vocabulary {padded_vocab_size}",
            shape[0]
        );
    }
    if shape[0] == padded_vocab_size {
        return Ok(weight.shallow_clone());
    }
    let padding = Tensor::zeros(
        [padded_vocab_size - shape[0], shape[1]],
        (weight.kind(), weight.device()),
    );
    Ok(Tensor::cat(&[weight, &padding], 0))
}

/// Shard embedding or output-head rows over the padded vocabulary.
pub fn shard_vocab_parallel_weight(
    weight: &Tensor,
    weight_scale: Option<&Tensor>,
    global_vocab_size: i64,
    padded_vocab_size: i64,
    tp_rank: usize,
    tp_size: usize,
    name: &str,
) -> Result<Glm5TpVocabShard> {
    let shape = weight.size();
    if shape.len() != 2 {
        bail!("{name} must be rank 2, got shape {shape:?}");
    }
    if padded_vocab_size < shape[0] || padded_vocab_size < global_vocab_size {
        bail!(
            "padded vocabulary {padded_vocab_size} is smaller than {name} rows {} or global vocabulary {global_vocab_size}",
            shape[0]
        );
    }
    let range = Glm5TpVocabRange::new(global_vocab_size, padded_vocab_size, tp_rank, tp_size)?;
    if weight_scale.is_some() && shape[0] != range.padded_vocab_size {
        bail!("cannot synthesize padded FP8 block scales for {name}");
    }
    let padded = pad_vocab_rows(weight, range.padded_vocab_size, name)?;
    let len = range.vocab_end - range.vocab_start;
    let weight_scale = weight_scale
        .map(|scale| checked_block_scale_shard(scale, range.vocab_start, len, 0, name))
        .transpose()?;
    Ok(Glm5TpVocabShard {
        weight: padded.narrow(0, range.vocab_start, len),
        weight_scale,
        range,
    })
}

/// TP-sharded embedding and shared output-head weights. Both tensors use one
/// common padded vocabulary so embedding lookup and vocab-parallel CE agree on
/// rank ownership even when the checkpoint stores only the global vocabulary.
pub struct Glm5TpVocabWeights {
    pub embed_tokens: Glm5TpVocabShard,
    pub lm_head: Glm5TpVocabShard,
}

impl Glm5TpVocabWeights {
    pub fn load_sharded(
        weights: &std::collections::BTreeMap<String, Tensor>,
        kind: Kind,
        global_vocab_size: i64,
        tie_word_embeddings: bool,
        tp_rank: usize,
        tp_size: usize,
    ) -> Result<Self> {
        use crate::model::KeepIfFp8;
        use rustrain_checkpoint::safetensors::tensor;

        let embed = tensor(weights, "model.embed_tokens.weight")?.keep_if_fp8(kind);
        let embed_scale = weights.get("model.embed_tokens.weight_scale_inv");
        let (lm_head, lm_head_scale) = if tie_word_embeddings {
            (embed.shallow_clone(), embed_scale)
        } else {
            (
                tensor(weights, "lm_head.weight")?.keep_if_fp8(kind),
                weights.get("lm_head.weight_scale_inv"),
            )
        };
        if embed.size().len() != 2 || lm_head.size().len() != 2 {
            bail!("embedding and LM head weights must both be rank 2");
        }
        if embed.size()[1] != lm_head.size()[1] {
            bail!(
                "embedding hidden size {} does not match LM head hidden size {}",
                embed.size()[1],
                lm_head.size()[1]
            );
        }
        let checkpoint_rows = embed.size()[0].max(lm_head.size()[0]);
        let range = Glm5TpVocabRange::new(global_vocab_size, checkpoint_rows, tp_rank, tp_size)?;
        Ok(Self {
            embed_tokens: shard_vocab_parallel_weight(
                &embed,
                embed_scale,
                global_vocab_size,
                range.padded_vocab_size,
                tp_rank,
                tp_size,
                "model.embed_tokens.weight",
            )?,
            lm_head: shard_vocab_parallel_weight(
                &lm_head,
                lm_head_scale,
                global_vocab_size,
                range.padded_vocab_size,
                tp_rank,
                tp_size,
                "lm_head.weight",
            )?,
        })
    }
}

/// Megatron-compatible TP shards for a dense or shared-expert SwiGLU MLP.
/// Gate/up are column-parallel; down is row-parallel over the same intermediate
/// interval. Down-projection outputs must be all-reduced by the caller.
pub struct Glm5TpMlpWeights {
    pub gate_proj: Tensor,
    pub gate_proj_scale: Option<Tensor>,
    pub up_proj: Tensor,
    pub up_proj_scale: Option<Tensor>,
    pub down_proj: Tensor,
    pub down_proj_scale: Option<Tensor>,
    pub intermediate_start: i64,
    pub intermediate_end: i64,
}

impl Glm5TpMlpWeights {
    pub fn from_full(
        gate_proj: &Tensor,
        gate_proj_scale: Option<&Tensor>,
        up_proj: &Tensor,
        up_proj_scale: Option<&Tensor>,
        down_proj: &Tensor,
        down_proj_scale: Option<&Tensor>,
        tp_rank: usize,
        tp_size: usize,
        name: &str,
    ) -> Result<Self> {
        let gate_shape = gate_proj.size();
        let up_shape = up_proj.size();
        let down_shape = down_proj.size();
        if gate_shape.len() != 2 || up_shape.len() != 2 || down_shape.len() != 2 {
            bail!("{name} gate/up/down weights must all be rank 2");
        }
        if gate_shape != up_shape {
            bail!("{name} gate shape {gate_shape:?} does not match up shape {up_shape:?}");
        }
        if down_shape[0] != gate_shape[1] || down_shape[1] != gate_shape[0] {
            bail!(
                "{name} down shape {down_shape:?} must be [{}, {}]",
                gate_shape[1],
                gate_shape[0]
            );
        }
        let gate = shard_column_parallel_linear(
            gate_proj,
            gate_proj_scale,
            tp_rank,
            tp_size,
            &format!("{name}.gate_proj"),
        )?;
        let up = shard_column_parallel_linear(
            up_proj,
            up_proj_scale,
            tp_rank,
            tp_size,
            &format!("{name}.up_proj"),
        )?;
        let down = shard_row_parallel_linear(
            down_proj,
            down_proj_scale,
            tp_rank,
            tp_size,
            &format!("{name}.down_proj"),
        )?;
        if gate.start != up.start || gate.start != down.start || gate.len != down.len {
            bail!("{name} TP partitions disagree on the intermediate interval");
        }
        Ok(Self {
            gate_proj: gate.weight,
            gate_proj_scale: gate.weight_scale,
            up_proj: up.weight,
            up_proj_scale: up.weight_scale,
            down_proj: down.weight,
            down_proj_scale: down.weight_scale,
            intermediate_start: gate.start,
            intermediate_end: gate.start + gate.len,
        })
    }

    /// `prefix` may name a dense MLP (`model.layers.N.mlp`) or its shared
    /// expert (`model.layers.N.mlp.shared_experts`).
    pub fn load_sharded(
        weights: &std::collections::BTreeMap<String, Tensor>,
        prefix: &str,
        kind: Kind,
        tp_rank: usize,
        tp_size: usize,
    ) -> Result<Self> {
        use crate::model::KeepIfFp8;
        use rustrain_checkpoint::safetensors::tensor;

        let gate_name = format!("{prefix}.gate_proj.weight");
        let up_name = format!("{prefix}.up_proj.weight");
        let down_name = format!("{prefix}.down_proj.weight");
        let gate = tensor(weights, &gate_name)?.keep_if_fp8(kind);
        let up = tensor(weights, &up_name)?.keep_if_fp8(kind);
        let down = tensor(weights, &down_name)?.keep_if_fp8(kind);
        Self::from_full(
            &gate,
            weights.get(&format!("{prefix}.gate_proj.weight_scale_inv")),
            &up,
            weights.get(&format!("{prefix}.up_proj.weight_scale_inv")),
            &down,
            weights.get(&format!("{prefix}.down_proj.weight_scale_inv")),
            tp_rank,
            tp_size,
            prefix,
        )
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
    pub q_b_proj: Tensor, // narrowed by [head_start*(qk_nope+qk_rope), heads_per_rank*(qk_nope+qk_rope)]
    pub kv_b_proj: Tensor, // narrowed by [head_start*(qk_nope+v_head), heads_per_rank*(qk_nope+v_head)]
    pub o_proj: Tensor,    // narrowed by [head_start*v_head, heads_per_rank*v_head] on dim 1
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
    pub indexer_weights_proj_scale: Option<Tensor>,
    pub indexer_wq_b_scale: Option<Tensor>,
    pub indexer_wk_scale: Option<Tensor>,
}

impl Clone for Glm5TpAttentionWeights {
    fn clone(&self) -> Self {
        macro_rules! clone_opt {
            ($t:expr) => {
                $t.as_ref().map(|t| t.shallow_clone())
            };
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
            indexer_weights_proj_scale: clone_opt!(&self.indexer_weights_proj_scale),
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
        use crate::model::KeepIfFp8;
        use rustrain_checkpoint::safetensors::tensor;

        let p = format!("model.layers.{layer}.self_attn");
        let qk_nope = config.qk_nope_head_dim;
        let qk_rope = config.qk_rope_head_dim;
        let v_head = config.v_head_dim;
        let idx_hd = config.index_head_dim;

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
        let indexer_type = config
            .indexer_types
            .get(layer)
            .map(|s| s.as_str())
            .unwrap_or("full");
        let (indexer_wq_b, indexer_wq_b_scale) = if indexer_type == "full" {
            let row_start = tp.idx_head_start * idx_hd;
            let row_len = tp.idx_heads_per_rank * idx_hd;
            let wq_b_full = weights
                .get(&format!("{p}.indexer.wq_b.weight"))
                .map(|t| t.keep_if_fp8(kind));
            let wq_b_scale = weights
                .get(&format!("{p}.indexer.wq_b.weight_scale_inv"))
                .map(|t| shard_block_scale(t, row_start, row_len, 0));
            if let Some(wq_full) = wq_b_full {
                (Some(wq_full.narrow(0, row_start, row_len)), wq_b_scale)
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Replicated indexer weights
        let indexer_wk = weights
            .get(&format!("{p}.indexer.wk.weight"))
            .map(|t| t.keep_if_fp8(kind));
        let indexer_k_norm_weight = weights
            .get(&format!("{p}.indexer.k_norm.weight"))
            .map(|t| t.to_kind(kind));
        let indexer_k_norm_bias = weights
            .get(&format!("{p}.indexer.k_norm.bias"))
            .map(|t| t.to_kind(kind));
        let weights_proj_scale = weights.get(&format!(
            "{p}.indexer.weights_proj.weight_scale_inv"
        ));
        let (indexer_weights_proj, indexer_weights_proj_scale) = match weights
            .get(&format!("{p}.indexer.weights_proj.weight"))
        {
            Some(weight) if weight.kind() == Kind::Float8e4m3fn && tp.tp_size > 1 => {
                let scale = weights_proj_scale
                    .context("FP8 TP indexer weights_proj is missing weight_scale_inv")?;
                let full = rustrain_deepseek_v4::fp8_kernel::dequant_fp8_weight(weight, scale)
                    .context("failed to dequantize FP8 TP indexer weights_proj")?;
                (
                    Some(full.narrow(0, tp.idx_head_start, tp.idx_heads_per_rank)),
                    None,
                )
            }
            Some(weight) => {
                let weight = weight.keep_if_fp8(kind);
                let weight = if tp.tp_size > 1 {
                    weight.narrow(0, tp.idx_head_start, tp.idx_heads_per_rank)
                } else {
                    weight
                };
                (
                    Some(weight),
                    weights_proj_scale.map(|scale| scale.shallow_clone()),
                )
            }
            None => (None, None),
        };

        // FP8 scales
        let q_a_proj_scale = weights
            .get(&format!("{p}.q_a_proj.weight_scale_inv"))
            .map(|t| t.shallow_clone());
        let kv_a_proj_scale = weights
            .get(&format!("{p}.kv_a_proj_with_mqa.weight_scale_inv"))
            .map(|t| t.shallow_clone());
        let q_b_proj_scale = weights
            .get(&format!("{p}.q_b_proj.weight_scale_inv"))
            .map(|t| shard_block_scale(t, q_b_row_start, q_b_row_len, 0));
        let kv_b_proj_scale = weights
            .get(&format!("{p}.kv_b_proj.weight_scale_inv"))
            .map(|t| shard_block_scale(t, kv_b_row_start, kv_b_row_len, 0));
        let o_proj_scale = weights
            .get(&format!("{p}.o_proj.weight_scale_inv"))
            .map(|t| shard_block_scale(t, o_col_start, o_col_len, 1));
        let indexer_wk_scale = weights
            .get(&format!("{p}.indexer.wk.weight_scale_inv"))
            .map(|t| t.shallow_clone());

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
            indexer_weights_proj_scale,
            kv_b_proj_scale,
            indexer_wq_b_scale,
            indexer_wk_scale,
        })
    }

    /// Apply LoRA to the local TP shard.  The adapter tensors are created for
    /// the unsharded checkpoint dimensions, so the delta is sliced on the
    /// output dimension for column-parallel projections and on the input
    /// dimension for the row-parallel output projection.
    pub fn with_lora(
        &self,
        layer: usize,
        registry: &Glm5LoraRegistry,
        tp: &Glm5TpShard,
        config: &crate::model::Glm5RuntimeConfig,
    ) -> Result<Self> {
        fn apply(
            base: &Tensor,
            scale: Option<&Tensor>,
            layer: usize,
            module: Glm5LoraTargetModule,
            registry: &Glm5LoraRegistry,
            row_start: Option<i64>,
            row_len: Option<i64>,
            col_start: Option<i64>,
            col_len: Option<i64>,
        ) -> Result<(Tensor, bool)> {
            let Some((a, b)) = registry.adapters.get(&(layer, module)) else {
                return Ok((base.shallow_clone(), false));
            };
            let scale_lora = registry.config.alpha as f64 / a.size()[0] as f64;
            let mut delta = b.matmul(a) * scale_lora;
            if let (Some(start), Some(len)) = (row_start, row_len) {
                delta = delta.narrow(0, start, len);
            }
            if let (Some(start), Some(len)) = (col_start, col_len) {
                delta = delta.narrow(1, start, len);
            }
            let base = if base.kind() == Kind::Float8e4m3fn {
                let scale = scale.context("FP8 TP LoRA base weight is missing weight_scale_inv")?;
                rustrain_deepseek_v4::fp8_kernel::dequant_fp8_weight(base, scale)
                    .context("failed to dequantize FP8 TP LoRA base weight")?
            } else {
                base.shallow_clone()
            };
            let delta = delta.to_kind(base.kind());
            Ok((base + delta, true))
        }

        let q_a = apply(
            &self.q_a_proj,
            self.q_a_proj_scale.as_ref(),
            layer,
            Glm5LoraTargetModule::WqA,
            registry,
            None,
            None,
            None,
            None,
        )?;
        let q_b = apply(
            &self.q_b_proj,
            self.q_b_proj_scale.as_ref(),
            layer,
            Glm5LoraTargetModule::WqB,
            registry,
            Some(tp.head_start * (config.qk_nope_head_dim + config.qk_rope_head_dim)),
            Some(tp.heads_per_rank * (config.qk_nope_head_dim + config.qk_rope_head_dim)),
            None,
            None,
        )?;
        let kv_a = apply(
            &self.kv_a_proj_with_mqa,
            self.kv_a_proj_scale.as_ref(),
            layer,
            Glm5LoraTargetModule::Wkv,
            registry,
            None,
            None,
            None,
            None,
        )?;
        let o_proj = apply(
            &self.o_proj,
            self.o_proj_scale.as_ref(),
            layer,
            Glm5LoraTargetModule::Wo,
            registry,
            None,
            None,
            Some(tp.head_start * config.v_head_dim),
            Some(tp.heads_per_rank * config.v_head_dim),
        )?;

        Ok(Self {
            q_a_proj: q_a.0,
            q_a_layernorm: self.q_a_layernorm.shallow_clone(),
            kv_a_proj_with_mqa: kv_a.0,
            kv_a_layernorm: self.kv_a_layernorm.shallow_clone(),
            q_b_proj: q_b.0,
            kv_b_proj: self.kv_b_proj.shallow_clone(),
            o_proj: o_proj.0,
            indexer_wq_b: self.indexer_wq_b.as_ref().map(Tensor::shallow_clone),
            indexer_wk: self.indexer_wk.as_ref().map(Tensor::shallow_clone),
            indexer_k_norm_weight: self
                .indexer_k_norm_weight
                .as_ref()
                .map(Tensor::shallow_clone),
            indexer_k_norm_bias: self.indexer_k_norm_bias.as_ref().map(Tensor::shallow_clone),
            indexer_weights_proj: self
                .indexer_weights_proj
                .as_ref()
                .map(Tensor::shallow_clone),
            q_a_proj_scale: if q_a.1 {
                None
            } else {
                self.q_a_proj_scale.as_ref().map(Tensor::shallow_clone)
            },
            q_b_proj_scale: if q_b.1 {
                None
            } else {
                self.q_b_proj_scale.as_ref().map(Tensor::shallow_clone)
            },
            kv_a_proj_scale: if kv_a.1 {
                None
            } else {
                self.kv_a_proj_scale.as_ref().map(Tensor::shallow_clone)
            },
            kv_b_proj_scale: self.kv_b_proj_scale.as_ref().map(Tensor::shallow_clone),
            o_proj_scale: if o_proj.1 {
                None
            } else {
                self.o_proj_scale.as_ref().map(Tensor::shallow_clone)
            },
            indexer_weights_proj_scale: self
                .indexer_weights_proj_scale
                .as_ref()
                .map(Tensor::shallow_clone),
            indexer_wq_b_scale: self.indexer_wq_b_scale.as_ref().map(Tensor::shallow_clone),
            indexer_wk_scale: self.indexer_wk_scale.as_ref().map(Tensor::shallow_clone),
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
    tp_comm: Option<&rustrain_nccl::nccl::NcclPersistentComm>,
    cp_comm: Option<&rustrain_nccl::nccl::NcclPersistentComm>,
) -> Tensor {
    use crate::model::{
        apply_rotary, apply_rotary_dispatch, apply_rotary_interleave, glm5_safe_linear, rms_norm,
        rms_norm_with_bias, rope_cos_sin_for_config,
    };

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
    let q_a_normed = rms_norm(
        &q_a,
        &attn.q_a_layernorm.to_kind(compute_kind),
        config.rms_norm_eps,
    );
    let q_b = glm5_safe_linear(&q_a_normed, &attn.q_b_proj, attn.q_b_proj_scale.as_ref());
    // q_b: [batch, s_local, heads_per_rank * (qk_nope+qk_rope)]
    let q = q_b
        .reshape([batch, s_local, nh_local, qk_nope + qk_rope])
        .transpose(1, 2);
    let q_nope = q.narrow(-1, 0, qk_nope);
    let q_rope = q.narrow(-1, qk_nope, qk_rope);

    let kv_a = glm5_safe_linear(
        input,
        &attn.kv_a_proj_with_mqa,
        attn.kv_a_proj_scale.as_ref(),
    );
    let kv_lora_raw = kv_a.narrow(-1, 0, kv_lora);
    let k_rope = kv_a.narrow(-1, kv_lora, qk_rope);
    let kv_lora_part = rms_norm(
        &kv_lora_raw,
        &attn.kv_a_layernorm.to_kind(compute_kind),
        config.rms_norm_eps,
    );
    let kv_b = glm5_safe_linear(
        &kv_lora_part,
        &attn.kv_b_proj,
        attn.kv_b_proj_scale.as_ref(),
    );
    let kv_b = kv_b.reshape([batch, s_local, nh_local, qk_nope + v_head]);
    let k_nope = kv_b.narrow(-1, 0, qk_nope).transpose(1, 2);
    let v = kv_b.narrow(-1, qk_nope, v_head).transpose(1, 2);

    // RoPE — use global positions for CP correctness
    let k_rope_expanded = k_rope
        .unsqueeze(2)
        .transpose(1, 2)
        .expand([batch, nh_local, s_local, qk_rope], false);
    let rope_offset = cp_rank as i64 * s_local;
    let (cos, sin) = rope_cos_sin_for_config(
        seq_global as usize,
        qk_rope,
        config,
        config.rope_interleave,
        input.device(),
    )
    .expect("validated GLM-5 RoPE configuration");
    let cos = cos
        .narrow(0, rope_offset, s_local as i64)
        .to_kind(compute_kind);
    let sin = sin
        .narrow(0, rope_offset, s_local as i64)
        .to_kind(compute_kind);
    let q_rope_rotated = apply_rotary_dispatch(&q_rope, &cos, &sin, config.rope_interleave);
    let k_rope_rotated =
        apply_rotary_dispatch(&k_rope_expanded, &cos, &sin, config.rope_interleave);

    let q_full = Tensor::cat(&[&q_nope, &q_rope_rotated], -1); // [B, H_local, S_local, d]
    let k_full = Tensor::cat(&[&k_nope, &k_rope_rotated], -1); // [B, H_local, S_local, d]
    let attn_scale = 1.0 / ((qk_nope + qk_rope) as f64).sqrt();

    // ── DSA Indexer (TP-sharded, CP-local) ──
    let should_compute_topk = config.should_recompute_indexer(layer);

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
            let idx_k = rms_norm_with_bias(
                &idx_k_raw,
                &k_norm_w.to_kind(compute_kind),
                &k_norm_b.to_kind(compute_kind),
                config.rms_norm_eps,
            );
            let idx_k_expanded = idx_k
                .unsqueeze(1)
                .expand([batch, idx_nh_local, s_local, idx_head_dim], false);

            let (cos_i, sin_i) = rope_cos_sin_for_config(
                seq_global as usize,
                qk_rope,
                config,
                config.indexer_rope_interleave,
                input.device(),
            )
            .expect("validated GLM-5 indexer RoPE configuration");
            let cos_i = cos_i
                .narrow(0, rope_offset, s_local as i64)
                .to_kind(compute_kind);
            let sin_i = sin_i
                .narrow(0, rope_offset, s_local as i64)
                .to_kind(compute_kind);
            let rotate_indexer = |value: &Tensor| {
                let nope_dim = idx_head_dim - qk_rope;
                if config.indexer_rope_interleave {
                    let nope = value.narrow(-1, 0, nope_dim);
                    let rope = value.narrow(-1, nope_dim, qk_rope);
                    Tensor::cat(
                        &[&nope, &apply_rotary_interleave(&rope, &cos_i, &sin_i)],
                        -1,
                    )
                } else {
                    let rope = value.narrow(-1, 0, qk_rope);
                    let nope = value.narrow(-1, qk_rope, nope_dim);
                    Tensor::cat(&[&apply_rotary(&rope, &cos_i, &sin_i), &nope], -1)
                }
            };
            let idx_q_rotated = rotate_indexer(&idx_q);
            let idx_k_rotated = rotate_indexer(&idx_k_expanded);
            let head_weights = glm5_safe_linear(
                input,
                weights_proj,
                indexer_weights.indexer_weights_proj_scale.as_ref(),
            )
                .reshape([batch, s_local, idx_nh_local])
                .transpose(1, 2)
                .to_kind(Kind::Float)
                * ((config.index_n_heads * idx_head_dim) as f64)
                    .sqrt()
                    .recip();

            // Compute causal top-k over the complete CP key sequence.  Every
            // rank contributes one key block through the same ring used by the
            // attention pass; local-only top-k is numerically incorrect because
            // it can discard the highest-scoring key on another rank.
            let actual_topk = idx_topk.min(seq_global);
            let mut best_scores: Option<Tensor> = None;
            let mut best_indices: Option<Tensor> = None;
            let mut idx_k_current = idx_k_rotated.shallow_clone();
            for block in 0..cp_size {
                let peer = (cp_rank + cp_size - block) % cp_size;
                let key_offset = peer as i64 * s_local;
                let key_len = s_local;
                let per_head = idx_q_rotated
                    .matmul(&idx_k_current.transpose(-2, -1))
                    .relu()
                    .to_kind(Kind::Float);
                let local_scores = (per_head * head_weights.unsqueeze(-1)).sum_dim_intlist(
                    [1].as_slice(),
                    true,
                    Kind::Float,
                );
                let mut scores = if tp.tp_size > 1 {
                    tp_comm
                        .expect("TP indexer reduction requires TP communicator")
                        .all_reduce(&local_scores)
                        .expect("NCCL TP indexer score reduction failed")
                } else {
                    local_scores
                };
                // Indexer is causal in global coordinates, including across CP
                // boundaries.  Masking before top-k keeps future keys out of the
                // state rather than relying on a later attention mask.
                let q_pos = (Tensor::arange(s_local, (Kind::Int64, input.device())) + rope_offset)
                    .to_kind(compute_kind);
                let k_pos = (Tensor::arange(key_len, (Kind::Int64, input.device())) + key_offset)
                    .to_kind(compute_kind);
                let future = (&k_pos.unsqueeze(0) - &q_pos.unsqueeze(1)).gt(0.0);
                scores = scores.masked_fill(&future.unsqueeze(0).unsqueeze(0), f64::NEG_INFINITY);
                let (block_scores, block_local_indices) =
                    scores.topk(actual_topk.min(key_len), -1, true, true);
                let block_indices = block_local_indices + key_offset;
                match (&best_scores, &best_indices) {
                    (Some(prev_scores), Some(prev_indices)) => {
                        let merged_scores = Tensor::cat(&[prev_scores, &block_scores], -1);
                        let merged_indices = Tensor::cat(&[prev_indices, &block_indices], -1);
                        let merge_topk =
                            actual_topk.min(merged_scores.size()[merged_scores.size().len() - 1]);
                        let (scores, positions) = merged_scores.topk(merge_topk, -1, true, true);
                        best_indices = Some(merged_indices.gather(-1, &positions, false));
                        best_scores = Some(scores);
                    }
                    _ => {
                        best_scores = Some(block_scores);
                        best_indices = Some(block_indices);
                    }
                }
                if block + 1 < cp_size {
                    let comm = cp_comm.expect("CP global indexer requires CP communicator");
                    let next_cp = (cp_rank + 1) % cp_size;
                    let prev_cp = (cp_rank + cp_size - 1) % cp_size;
                    idx_k_current =
                        cp_ring_send_recv_autograd(&idx_k_current, comm, next_cp, prev_cp)
                            .expect("NCCL global indexer K exchange failed");
                }
            }
            let topk_indices = best_indices
                .expect("global top-k produced no candidates")
                .expand([batch, nh_local, s_local, actual_topk], false);
            let idx_bias_keys = Tensor::zeros([batch, 1, s_local], (compute_kind, input.device()));

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
                    let _ = m.scatter_add_(-1, &state.topk_indices, &ones);
                    m
                };
                let causal_f = {
                    let cm = Tensor::ones([s_local, s_local], (Kind::Bool, input.device())).triu(1);
                    cm.unsqueeze(0)
                        .unsqueeze(0)
                        .expand([batch as i64, nh_local, s_local, s_local], false)
                        .to_kind(compute_kind)
                };
                let combined = &sparse_mask * (1.0 - &causal_f);
                drop(sparse_mask);
                drop(causal_f);
                let bias =
                    Tensor::zeros_like(&combined).masked_fill(&combined.eq(0), f64::NEG_INFINITY);
                drop(combined);
                Tensor::scaled_dot_product_attention(
                    &q_full,
                    &k_full,
                    &v,
                    Some(&bias),
                    0.0,
                    false,
                    Some(attn_scale),
                    false,
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
                        let _ = m.scatter_add_(-1, &chunk_topk, &ones);
                        m
                    };
                    let causal_f = {
                        let q_pos = (Tensor::arange(q_len, (Kind::Int64, input.device()))
                            + q_start)
                            .to_kind(compute_kind);
                        let k_pos = Tensor::arange(s_local, (Kind::Int64, input.device()))
                            .to_kind(compute_kind);
                        let diff = k_pos.unsqueeze(0) - q_pos.unsqueeze(1);
                        diff.gt(0.0)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .expand([batch as i64, nh_local, q_len, s_local], false)
                            .to_kind(compute_kind)
                    };
                    let combined = &sparse_mask * (1.0 - &causal_f);
                    drop(sparse_mask);
                    drop(causal_f);
                    let bias = Tensor::zeros_like(&combined)
                        .masked_fill(&combined.eq(0), f64::NEG_INFINITY);
                    drop(combined);
                    let chunk_out = Tensor::scaled_dot_product_attention(
                        &q_chunk,
                        &k_full,
                        &v,
                        Some(&bias),
                        0.0,
                        false,
                        Some(attn_scale),
                        false,
                    );
                    drop(bias);
                    outputs.push(chunk_out);
                }
                let refs: Vec<&Tensor> = outputs.iter().collect();
                Tensor::cat(&refs, 2)
            }
        } else {
            Tensor::scaled_dot_product_attention::<&Tensor>(
                &q_full,
                &k_full,
                &v,
                None,
                0.0,
                true,
                Some(attn_scale),
                false,
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

        let comm = cp_comm.expect("CP requires NCCL communicator");
        let actual_topk = index_share_state
            .as_ref()
            .map(|s| s.topk_indices.size()[s.topk_indices.size().len() - 1])
            .unwrap_or(0);

        // Online softmax state: running max, numerator, denominator
        let mut running_max: Option<Tensor> = None; // [B, H, S_local, 1]
        let mut running_num: Option<Tensor> = None; // [B, H, S_local, v_head]
        let mut running_denom: Option<Tensor> = None; // [B, H, S_local, 1]

        // Current K/V block (start with local)
        let mut k_current = k_full.shallow_clone();
        let mut v_current = v.shallow_clone();

        for step in 0..cp_size {
            let peer = (cp_rank + cp_size - step) % cp_size;
            let k_start_global = peer as i64 * s_local;

            // Build attention bias for this K/V block
            let bias = if let Some(state) = index_share_state {
                let global_topk = &state.topk_indices;
                let in_range = global_topk
                    .ge(k_start_global)
                    .logical_and(&global_topk.lt(k_start_global + s_local));
                let local_topk =
                    (global_topk - k_start_global).masked_fill(&in_range.logical_not(), 0);
                let valid = in_range.to_kind(compute_kind);

                let mut sparse_mask = Tensor::zeros(
                    [batch as i64, nh_local, s_local, s_local],
                    (compute_kind, input.device()),
                );
                let weighted_ones = Tensor::ones(
                    [batch as i64, nh_local, s_local, actual_topk],
                    (compute_kind, input.device()),
                ) * &valid;
                let _ = sparse_mask.scatter_add_(-1, &local_topk, &weighted_ones);
                drop(local_topk);
                drop(valid);

                let causal_f = {
                    let q_pos = (Tensor::arange(s_local, (Kind::Int64, input.device()))
                        + (cp_rank as i64 * s_local))
                        .to_kind(compute_kind);
                    let k_pos = (Tensor::arange(s_local, (Kind::Int64, input.device()))
                        + k_start_global)
                        .to_kind(compute_kind);
                    let diff = k_pos.unsqueeze(0) - q_pos.unsqueeze(1);
                    diff.gt(0.0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand([batch as i64, nh_local, s_local, s_local], false)
                        .to_kind(compute_kind)
                };

                let combined: Tensor = &sparse_mask * (1.0 - &causal_f);
                let mask = combined.eq(0.0); // bool: true where masked
                drop(sparse_mask);
                drop(causal_f);

                let bias_base = Tensor::zeros(
                    [batch as i64, nh_local, s_local, s_local],
                    (compute_kind, input.device()),
                );
                Some(bias_base.masked_fill(&mask, f64::NEG_INFINITY))
            } else {
                let q_pos = (Tensor::arange(s_local, (Kind::Int64, input.device()))
                    + (cp_rank as i64 * s_local))
                    .to_kind(compute_kind);
                let k_pos = (Tensor::arange(s_local, (Kind::Int64, input.device()))
                    + k_start_global)
                    .to_kind(compute_kind);
                let diff = k_pos.unsqueeze(0) - q_pos.unsqueeze(1);
                let causal_mask = diff.gt(0.0); // bool: true where masked (future)
                let bias_base = Tensor::zeros([s_local, s_local], (compute_kind, input.device()));
                let bias = bias_base.masked_fill(&causal_mask, f64::NEG_INFINITY);
                Some(
                    bias.unsqueeze(0)
                        .unsqueeze(0)
                        .expand([batch as i64, nh_local, s_local, s_local], false)
                        .to_kind(compute_kind),
                )
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
            let finite = scores.isfinite();
            let has_valid = finite.any_dim(-1, true);
            let safe_scores = scores.masked_fill(&finite.logical_not(), 0.0);
            let block_max = safe_scores
                .amax([-1], true)
                .masked_fill(&has_valid.logical_not(), f64::NEG_INFINITY);
            let exp_scores = (&scores - &block_max)
                .exp()
                .masked_fill(&finite.logical_not(), 0.0)
                .masked_fill(&has_valid.logical_not(), 0.0);
            let block_num = exp_scores.matmul(&v_current); // [B, H, S_local, v_head]
            let block_denom = exp_scores.sum_dim_intlist([-1].as_slice(), true, compute_kind); // [B, H, S_local, 1]

            match (&mut running_max, &mut running_num, &mut running_denom) {
                (Some(rm), Some(rn), Some(rd)) => {
                    // element-wise maximum via f_max_other
                    let old_valid = rm.isfinite();
                    let new_valid = block_max.isfinite();
                    let new_max = rm.f_max_other(&block_max).unwrap();
                    let exp_old = (&*rm - &new_max)
                        .exp()
                        .masked_fill(&old_valid.logical_not(), 0.0);
                    let exp_new = (&block_max - &new_max)
                        .exp()
                        .masked_fill(&new_valid.logical_not(), 0.0);
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
                let next_cp = (cp_rank + 1) % cp_size;
                let prev_cp = (cp_rank + cp_size - 1) % cp_size;
                let (k_recv, v_recv) =
                    cp_ring_send_recv_kv_autograd(&k_current, &v_current, comm, next_cp, prev_cp)
                        .expect("NCCL ring K/V exchange failed");
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

#[cfg(test)]
mod tp_weight_tests {
    use super::*;

    fn assert_close(actual: &Tensor, expected: &Tensor) {
        let max_error = (actual - expected).abs().max().double_value(&[]);
        assert!(max_error < 1e-5, "max error {max_error} exceeded tolerance");
    }

    #[test]
    fn column_parallel_concat_matches_full_forward_and_dx() {
        let x = Tensor::from_slice(&[1.0_f32, -2.0, 0.5, 3.0]).reshape([2, 2]);
        let weight =
            Tensor::from_slice(&[0.5_f32, -1.0, 2.0, 1.5, -0.25, 0.75, 1.25, -0.5]).reshape([4, 2]);
        let grad =
            Tensor::from_slice(&[0.25_f32, -0.5, 1.0, 0.75, -1.25, 0.5, 0.1, -0.2]).reshape([2, 4]);
        let full = x.matmul(&weight.transpose(0, 1));
        let mut outputs = Vec::new();
        let mut dx = Tensor::zeros_like(&x);
        for rank in 0..2 {
            let shard = shard_column_parallel_linear(&weight, None, rank, 2, "test").unwrap();
            outputs.push(x.matmul(&shard.weight.transpose(0, 1)));
            let grad_shard = grad.narrow(1, shard.start, shard.len);
            dx += grad_shard.matmul(&shard.weight);
        }
        assert_close(&Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1), &full);
        assert_close(&dx, &grad.matmul(&weight));
    }

    #[test]
    fn row_parallel_sum_matches_full_forward_and_dx_shards() {
        let x = Tensor::from_slice(&[1.0_f32, -2.0, 0.5, 3.0]).reshape([2, 2]);
        let weight =
            Tensor::from_slice(&[0.5_f32, -1.0, 2.0, 1.5, -0.25, 0.75, 1.25, -0.5]).reshape([4, 2]);
        let grad =
            Tensor::from_slice(&[0.25_f32, -0.5, 1.0, 0.75, -1.25, 0.5, 0.1, -0.2]).reshape([2, 4]);
        let full = x.matmul(&weight.transpose(0, 1));
        let mut output = Tensor::zeros([2, 4], (Kind::Float, tch::Device::Cpu));
        let mut dx_shards = Vec::new();
        for rank in 0..2 {
            let shard = shard_row_parallel_linear(&weight, None, rank, 2, "test").unwrap();
            output += x
                .narrow(1, shard.start, shard.len)
                .matmul(&shard.weight.transpose(0, 1));
            dx_shards.push(grad.matmul(&shard.weight));
        }
        assert_close(&output, &full);
        assert_close(
            &Tensor::cat(&dx_shards.iter().collect::<Vec<_>>(), 1),
            &grad.matmul(&weight),
        );
    }

    #[test]
    fn mlp_column_and_row_partitions_match_swiglu_forward() -> anyhow::Result<()> {
        let x = Tensor::from_slice(&[1.0_f32, -2.0, 0.5, 3.0]).reshape([2, 2]);
        let gate =
            Tensor::from_slice(&[0.5_f32, -1.0, 2.0, 1.5, -0.25, 0.75, 1.25, -0.5]).reshape([4, 2]);
        let up = Tensor::from_slice(&[-0.5_f32, 1.0, 0.25, 0.75, 1.5, -0.25, -1.25, 0.5])
            .reshape([4, 2]);
        let down =
            Tensor::from_slice(&[0.5_f32, -1.0, 2.0, 1.5, -0.25, 0.75, 1.25, -0.5]).reshape([2, 4]);
        let full_hidden = x.matmul(&gate.transpose(0, 1)).silu() * x.matmul(&up.transpose(0, 1));
        let full_output = full_hidden.matmul(&down.transpose(0, 1));
        let mut output = Tensor::zeros([2, 2], (Kind::Float, tch::Device::Cpu));
        for rank in 0..2 {
            let shard = Glm5TpMlpWeights::from_full(
                &gate, None, &up, None, &down, None, rank, 2, "test.mlp",
            )?;
            let hidden = x.matmul(&shard.gate_proj.transpose(0, 1)).silu()
                * x.matmul(&shard.up_proj.transpose(0, 1));
            output += hidden.matmul(&shard.down_proj.transpose(0, 1));
        }
        assert_close(&output, &full_output);
        Ok(())
    }

    #[test]
    fn vocab_range_pads_and_shards_rows_like_megatron() {
        let full = Tensor::arange(20, (Kind::Float, tch::Device::Cpu)).reshape([10, 2]);
        let mut shards = Vec::new();
        for rank in 0..3 {
            let shard = shard_vocab_parallel_weight(&full, None, 10, 12, rank, 3, "embed").unwrap();
            assert_eq!(shard.range.vocab_end - shard.range.vocab_start, 4);
            shards.push(shard.weight);
        }
        let padded = Tensor::cat(&shards.iter().collect::<Vec<_>>(), 0);
        let expected = Tensor::cat(
            &[
                &full,
                &Tensor::zeros([2, 2], (Kind::Float, tch::Device::Cpu)),
            ],
            0,
        );
        assert_close(&padded, &expected);
        assert_eq!(Glm5TpVocabRange::new(10, 10, 2, 3).unwrap().vocab_start, 8);
    }

    #[test]
    fn non_divisible_linear_and_mlp_dimensions_are_rejected() {
        let weight = Tensor::zeros([5, 2], (Kind::Float, tch::Device::Cpu));
        assert!(shard_column_parallel_linear(&weight, None, 0, 2, "test").is_err());
        let gate = Tensor::zeros([5, 2], (Kind::Float, tch::Device::Cpu));
        let up = gate.shallow_clone();
        let down = Tensor::zeros([2, 5], (Kind::Float, tch::Device::Cpu));
        assert!(
            Glm5TpMlpWeights::from_full(&gate, None, &up, None, &down, None, 0, 2, "test.mlp")
                .is_err()
        );
    }
}
