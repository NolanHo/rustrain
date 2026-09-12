//! Sharding specifications: which part of a tensor a rank holds.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::ShardError;
use crate::group::GroupKind;

/// Logical position of the expert axis in an expert-parallel weight tensor:
/// `[num_experts, ...]`, so the expert axis is dim 0.
///
/// Named here so that "gathering an expert shard" does not hard-code a 0 in the
/// middle of the transition rules.
pub const EXPERT_DIM: i64 = 0;

/// Default logical position of the sequence axis: dim 1 of the canonical
/// `[batch, sequence, ...]` activation layout.
///
/// Context parallelism splits the sequence axis, but "the sequence axis" is not
/// a property a layout can carry — `ParallelLayout::SequenceShard` names the
/// *kind* of sharding, not an index. This constant is the default assumption;
/// a plan whose activations are laid out differently (for example
/// `[batch, head, sequence, head_dim]`) overrides it with
/// [`DimNormalizer::with_sequence_dim`].
pub const DEFAULT_SEQUENCE_DIM: i64 = 1;

/// Reduction carried by a [`ParallelLayout::Partial`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReduceOp {
    /// Partials are summed.
    Sum,
    /// Partials are combined with a maximum (e.g. an amax).
    Max,
    /// Partials are combined with a minimum.
    Min,
}

impl ReduceOp {
    /// Stable lowercase name, used by `Display`.
    pub const fn as_str(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
        }
    }
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a tensor is distributed over the ranks of a process group.
///
/// A layout is *data*: it travels with a plan slot and the propagation pass
/// compares layouts to decide where communication is needed (invariant I-3).
/// No kernel has to know about it, and no training loop writes a collective by
/// hand.
///
/// `dim` is a *logical* dimension: it may be negative (Python-style) and is
/// resolved against a concrete tensor rank by [`DimNormalizer`] when a
/// conversion is emitted. Logical dims keep a plan reusable across tensors of
/// different ranks — the plan says "the last axis", not "axis 3".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelLayout {
    /// Every rank of the group holds an identical, complete copy.
    Replicate,

    /// The tensor is split evenly along `dim` across `group`; rank `i` of the
    /// group holds slice `i`.
    Shard { dim: i64, group: GroupKind },

    /// Every rank holds a partial reduction of the complete tensor. `op` is how
    /// the partials combine: the complete value is obtained by reducing them
    /// with `op` across `group`.
    Partial { op: ReduceOp, group: GroupKind },

    /// The expert axis is split across `group` (expert parallelism). Each rank
    /// owns a disjoint set of experts and holds those weights in full.
    ExpertShard { group: GroupKind },

    /// The sequence axis is split across `group` (context parallelism).
    /// `LocalAttention`-style kernels consume this layout directly; anything
    /// that needs whole sequences all-gathers it.
    SequenceShard { group: GroupKind },
}

impl ParallelLayout {
    /// Whether every rank of the group holds the complete tensor.
    pub fn is_replicated(&self) -> bool {
        matches!(self, ParallelLayout::Replicate)
    }

    /// The group this layout is distributed over, or `None` for
    /// [`ParallelLayout::Replicate`] (a replica has no communication group).
    pub fn group(&self) -> Option<GroupKind> {
        match self {
            ParallelLayout::Replicate => None,
            ParallelLayout::Shard { group, .. }
            | ParallelLayout::Partial { group, .. }
            | ParallelLayout::ExpertShard { group }
            | ParallelLayout::SequenceShard { group } => Some(*group),
        }
    }
}

impl fmt::Display for ParallelLayout {
    /// Compact form, e.g. `shard(-1, tp)` or `partial(sum, tp)`.
    ///
    /// Kept short because it appears inside error messages and plan dumps,
    /// where a reader compares two layouts at a glance.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParallelLayout::Replicate => f.write_str("replicate"),
            ParallelLayout::Shard { dim, group } => write!(f, "shard({dim}, {group})"),
            ParallelLayout::Partial { op, group } => write!(f, "partial({op}, {group})"),
            ParallelLayout::ExpertShard { group } => write!(f, "expert({group})"),
            ParallelLayout::SequenceShard { group } => write!(f, "seq({group})"),
        }
    }
}

/// Resolves logical dimension indices against the rank of a concrete tensor.
///
/// A plan is written once and instantiated on tensors whose rank is known only
/// at propagation time, so a layout stores a logical dim (`-1` = last axis) and
/// the axis becomes a concrete non-negative index here. This is also the only
/// place that decides whether a dim is legal at all: an axis outside
/// `0..rank` is a plan bug, and it is reported (`ShardError::DimOutOfRange`)
/// whether or not the conversion that mentioned it happens to need data
/// movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimNormalizer {
    rank: i64,
    sequence_dim: i64,
}

impl DimNormalizer {
    /// A normalizer for tensors of `rank` dimensions (scalars have rank 0).
    ///
    /// # Errors
    ///
    /// [`ShardError::InvalidTensorRank`] if `rank` is negative.
    pub fn new(rank: i64) -> Result<Self, ShardError> {
        if rank < 0 {
            return Err(ShardError::InvalidTensorRank { rank });
        }
        Ok(Self {
            rank,
            sequence_dim: DEFAULT_SEQUENCE_DIM,
        })
    }

    /// Overrides the logical axis [`ParallelLayout::SequenceShard`] refers to.
    ///
    /// The override is stored as a logical dim, so it is resolved (and range
    /// checked) by [`Self::sequence_dim`], not here.
    pub fn with_sequence_dim(mut self, dim: i64) -> Self {
        self.sequence_dim = dim;
        self
    }

    /// The tensor rank this normalizer was built for.
    pub fn rank(&self) -> i64 {
        self.rank
    }

    /// The resolved (non-negative) sequence axis.
    ///
    /// # Errors
    ///
    /// [`ShardError::DimOutOfRange`] if the configured sequence dim does not
    /// exist on a tensor of this rank.
    pub fn sequence_dim(&self) -> Result<i64, ShardError> {
        self.normalize(self.sequence_dim)
    }

    /// Resolves `dim` to a non-negative axis: a negative dim counts from the
    /// back (`-1` is the last axis), then the result must be within `0..rank`.
    ///
    /// # Errors
    ///
    /// [`ShardError::DimOutOfRange`] if the resolved axis does not exist.
    pub fn normalize(&self, dim: i64) -> Result<i64, ShardError> {
        let resolved = if dim < 0 { dim + self.rank } else { dim };
        if resolved < 0 || resolved >= self.rank {
            return Err(ShardError::DimOutOfRange {
                dim,
                rank: self.rank,
            });
        }
        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_dims_count_from_the_back() {
        let n = DimNormalizer::new(4).unwrap();
        assert_eq!(n.normalize(0).unwrap(), 0);
        assert_eq!(n.normalize(3).unwrap(), 3);
        assert_eq!(n.normalize(-1).unwrap(), 3);
        assert_eq!(n.normalize(-4).unwrap(), 0);
    }

    #[test]
    fn out_of_range_dims_are_rejected() {
        let n = DimNormalizer::new(4).unwrap();
        assert_eq!(
            n.normalize(4),
            Err(ShardError::DimOutOfRange { dim: 4, rank: 4 })
        );
        assert_eq!(
            n.normalize(-5),
            Err(ShardError::DimOutOfRange { dim: -5, rank: 4 })
        );
    }

    /// A scalar has no axes at all, so every dim is out of range rather than
    /// silently wrapping to axis 0.
    #[test]
    fn scalars_have_no_axes() {
        let n = DimNormalizer::new(0).unwrap();
        assert_eq!(n.rank(), 0);
        assert_eq!(
            n.normalize(0),
            Err(ShardError::DimOutOfRange { dim: 0, rank: 0 })
        );
        assert_eq!(
            n.sequence_dim(),
            Err(ShardError::DimOutOfRange { dim: 1, rank: 0 })
        );
    }

    #[test]
    fn sequence_dim_can_be_overridden() {
        let n = DimNormalizer::new(4).unwrap().with_sequence_dim(-2);
        assert_eq!(n.sequence_dim().unwrap(), 2);
        let bad = DimNormalizer::new(2).unwrap().with_sequence_dim(2);
        assert_eq!(
            bad.sequence_dim(),
            Err(ShardError::DimOutOfRange { dim: 2, rank: 2 })
        );
    }

    #[test]
    fn negative_tensor_rank_is_rejected() {
        assert_eq!(
            DimNormalizer::new(-1).unwrap_err(),
            ShardError::InvalidTensorRank { rank: -1 }
        );
    }
}
