//! Error types for topology resolution and layout conversion.
//!
//! Two types, one per layer of the crate: [`ParallelError`] for anything that
//! can go wrong while resolving a topology (bad config, bad mesh, mask bit
//! outside the mesh, rank out of range), and [`ShardError`] for anything that
//! can go wrong while inferring communication from a pair of layouts or while
//! computing the local shape a layout implies.
//!
//! Every `ShardError` names both layouts: a propagation pass needs to report
//! *which edge* could not be converted, and it only has the two layouts.

use crate::config::ParallelDim;
use crate::layout::ParallelLayout;
use crate::mesh::GroupMask;

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

    #[error("a mesh must declare at least one axis (1..={max} allowed)")]
    NoAxes { max: usize },

    #[error("too many mesh axes: {count} declared, at most {max} allowed")]
    TooManyAxes { count: usize, max: usize },

    #[error(
        "mesh world size overflows usize: the degrees {degrees:?} do not multiply into a rank \
         number, so no rank can be enumerated and no group has members"
    )]
    MeshWorldSizeOverflow { degrees: Vec<usize> },

    #[error("mesh axis {index} has an empty name; axis names must be non-empty")]
    EmptyAxisName { index: usize },

    #[error("duplicate mesh axis name `{name}`; axis names must be unique")]
    DuplicateAxis { name: String },

    #[error("mesh axis `{name}` has degree 0; every degree must be at least 1")]
    ZeroDegree { name: String },

    #[error(
        "group mask bit {bit} is out of range: the mesh has {axes} axes, so mask bits \
         address axes 0..{axes}"
    )]
    GroupOutOfRange { bit: usize, axes: usize },

    #[error(
        "axis {axis} does not fit in a group mask: a mask carries {max} bits, so the axis \
         must be in 0..{max}"
    )]
    AxisOutOfRange { axis: usize, max: usize },
}

/// Two layouts could not be converted into one another, or a layout implies a
/// local shape that does not exist.
///
/// Each variant explains *why* the conversion is not a layout conversion, and
/// what the caller should do instead. The reasons matter as much as the error:
/// the usual cause is a plan that is wrong about how a tensor is produced, not
/// a missing communication primitive.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ShardError {
    #[error(
        "cannot convert {from} to {to}: the two partials are not the same reduction \
         (different reduce op or different group), and a partial is only a reduction of \
         itself; all_reduce {from} to `replicate` first, then produce {to} from the \
         computation"
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
        "cannot convert {from} to {to}: the layouts shard a common dimension over \
         different groups, and a layout transition is only defined inside one group; \
         crossing groups is a redistribution that no single collective here expresses. \
         Convert one side to `replicate` first — that is correct but costs a full gather"
    )]
    GroupMismatch {
        from: ParallelLayout,
        to: ParallelLayout,
    },

    #[error(
        "cannot convert {from} to {to}: a layout distributes one tensor over overlapping groups \
         {a} and {b}; the rank decomposition is ambiguous — the same ranks would both index \
         slices of one distribution and carry pieces of the other. Shard groups and the \
         partial's group must be pairwise disjoint"
    )]
    OverlappingGroups {
        from: ParallelLayout,
        to: ParallelLayout,
        a: GroupMask,
        b: GroupMask,
    },

    #[error(
        "cannot convert {from} to {to}: completing the partial and dropping a source shard in \
         the same step need two collectives whose order cannot be proven; express the \
         intermediate layout explicitly (complete the partial to the intermediate layout \
         first, then convert from there)"
    )]
    PartialCompletionDropsShard {
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

    #[error(
        "cannot shard dim {dim}: the global size {global} is not divisible by the divisor \
         {divisor} the layout imposes on that axis; no local shape exists, so this is a \
         compile-time error, not a runtime fallback"
    )]
    NotDivisible { dim: i64, global: i64, divisor: i64 },

    #[error(
        "group mask bit {bit} is out of range: the mesh has {axes} axes, so mask bits \
         address axes 0..{axes}"
    )]
    GroupOutOfRange { bit: usize, axes: usize },
}
