//! Error types for topology resolution and layout conversion.
//!
//! Two types, one per layer of the crate: [`ParallelError`] for anything that
//! can go wrong while resolving a topology (bad config, rank out of range), and
//! [`ShardError`] for anything that can go wrong while inferring communication
//! from a pair of layouts.
//!
//! Every `ShardError` names both layouts: a propagation pass needs to report
//! *which edge* could not be converted, and it only has the two layouts.

use crate::config::ParallelDim;
use crate::layout::ParallelLayout;

/// A topology could not be resolved.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParallelError {
    #[error(
        "parallel dimension `{dim}` is 0; every dimension must be at least 1 \
         (a zero dimension makes the world size 0)"
    )]
    ZeroDimension { dim: ParallelDim },

    #[error(
        "parallel config product overflows usize: tensor={tensor} context={context} \
         expert={expert} data={data} pipeline={pipeline}"
    )]
    WorldSizeOverflow {
        tensor: usize,
        context: usize,
        expert: usize,
        data: usize,
        pipeline: usize,
    },

    #[error("rank {rank} is out of range for world size {world_size}")]
    RankOutOfRange { rank: usize, world_size: usize },
}

/// Two layouts could not be converted into one another.
///
/// Each variant explains *why* the conversion is not a layout conversion, and
/// what the caller should do instead. The reasons matter as much as the error:
/// the usual cause is a plan that is wrong about how a tensor is produced, not
/// a missing communication primitive.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ShardError {
    #[error(
        "cannot convert {from} to {to}: the two partials carry different reduce ops, and a \
         partial is only a reduction of itself; convert {from} to `replicate` first \
         (all_reduce with the source op) and produce {to} from the computation"
    )]
    PartialOpMismatch {
        from: ParallelLayout,
        to: ParallelLayout,
    },

    #[error(
        "cannot convert {from} to {to}: a complete replica is not a partial sum, and no \
         collective turns one into the other; a partial has to come from the computation \
         (e.g. a sharded matmul that accumulates only part of the reduction)"
    )]
    ReplicateToPartial {
        from: ParallelLayout,
        to: ParallelLayout,
    },

    #[error(
        "cannot convert {from} to {to}: reduce_scatter only produces a shard of a `sum` \
         partial; a `{from}` partial cannot be turned into a disjoint shard without \
         recomputation (all_reduce it to `replicate` first)"
    )]
    ReduceScatterRequiresSum {
        from: ParallelLayout,
        to: ParallelLayout,
    },

    #[error(
        "cannot convert {from} to {to}: a shard holds a disjoint piece of the tensor, so \
         its ranks cannot be reinterpreted as partial sums of one another; a partial is \
         produced by recomputation, or a replica is derived by all_gather"
    )]
    ShardToPartial {
        from: ParallelLayout,
        to: ParallelLayout,
    },

    #[error(
        "cannot convert {from} to {to}: the two layouts live on different process groups, \
         and a layout transition is only defined inside one group; crossing groups is a \
         redistribution (all-to-all) that no single collective here expresses. Convert one \
         side to `replicate` first — that is correct but costs a full gather"
    )]
    GroupMismatch {
        from: ParallelLayout,
        to: ParallelLayout,
    },

    #[error(
        "cannot convert {from} to {to}: no rule covers this pair; the conversion is not \
         derivable from the layouts alone. Converting {from} to `replicate` first is always \
         valid (and always costs a full gather), or express the source as a `shard`/`partial` \
         so the named layout has a defined meaning"
    )]
    UnsupportedTransition {
        from: ParallelLayout,
        to: ParallelLayout,
    },

    #[error("tensor rank {rank} is invalid: a tensor has a non-negative number of dimensions")]
    InvalidTensorRank { rank: i64 },

    #[error(
        "dim {dim} is out of range for a rank-{rank} tensor: after resolving a negative dim \
         against the rank, the axis must be in 0..{rank}"
    )]
    DimOutOfRange { dim: i64, rank: i64 },
}
