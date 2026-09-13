//! Sharding specifications: which part of a tensor a rank holds.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{ParallelError, ShardError};
use crate::mesh::{GroupMask, Mesh};

/// Reduction carried by a [`ParallelLayout`]'s partial.
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

/// One independent shard of a tensor: dim `dim` is split evenly over the ranks
/// of `group`, rank `i` of the group holding slice `i`.
///
/// `dim` is a *logical* dimension: it may be negative (Python-style) and is
/// resolved against a concrete tensor rank when the layout is used. Logical
/// dims keep a plan reusable across tensors of different ranks — the plan says
/// "the last axis", not "axis 3".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShardSpec {
    /// The (logical) tensor axis that is split.
    pub dim: i64,
    /// The group whose ranks hold the slices.
    pub group: GroupMask,
}

/// A partial reduction: every rank of `group` holds a partial of the complete
/// tensor, and the complete value is obtained by reducing them with `op`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartialSpec {
    /// How the partials combine.
    pub op: ReduceOp,
    /// The group whose ranks hold one partial each.
    pub group: GroupMask,
}

/// How a tensor is distributed over the ranks of a mesh.
///
/// A layout is *data*: it travels with a plan slot and the propagation pass
/// compares layouts to decide where communication is needed (invariant I-3).
/// No kernel has to know about it, and no training loop writes a collective by
/// hand.
///
/// A layout is **several independent shards plus at most one partial**, because
/// one tensor can be split along several axes at once: with `tp=2, ep=4` the
/// MoE weight `[E, 2I, H]` is sharded dim 0 over `ep` *and* dim 1 over `tp`
/// (`docs/design/model-description.md` §2.1). `partial` stays "at most one":
/// no real layout needs two partial reductions, and the transition table has to
/// stay exhaustible.
///
/// Not `Copy`: it owns a `Vec`. Serializes deterministically as
/// `{"dims": [{"dim":..,"group":..}, ..], "partial": {"op":..,"group":..} | null}`
/// (a vec, never a map).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParallelLayout {
    /// The independent shards; the local size of dim `d` divides by the product
    /// of the degrees of every shard whose (normalized) dim is `d`.
    pub dims: Vec<ShardSpec>,
    /// The partial reduction, if any.
    pub partial: Option<PartialSpec>,
}

impl ParallelLayout {
    /// Every rank holds an identical, complete copy.
    pub fn replicate() -> Self {
        Self {
            dims: Vec::new(),
            partial: None,
        }
    }

    /// Split dim `dim` across `group`.
    pub fn shard(dim: i64, group: GroupMask) -> Self {
        Self {
            dims: vec![ShardSpec { dim, group }],
            partial: None,
        }
    }

    /// A partial reduction with `op` across `group`.
    pub fn partial(op: ReduceOp, group: GroupMask) -> Self {
        Self {
            dims: Vec::new(),
            partial: Some(PartialSpec { op, group }),
        }
    }

    /// Whether every rank holds the complete tensor.
    pub fn is_replicated(&self) -> bool {
        self.dims.is_empty() && self.partial.is_none()
    }

    /// Every distinct group this layout distributes over: each shard's group in
    /// declaration order, then the partial's group. Duplicates are dropped on
    /// first occurrence, so the result is deterministic.
    pub fn groups(&self) -> Vec<GroupMask> {
        let mut groups = Vec::new();
        for spec in &self.dims {
            if !groups.contains(&spec.group) {
                groups.push(spec.group);
            }
        }
        if let Some(partial) = &self.partial {
            if !groups.contains(&partial.group) {
                groups.push(partial.group);
            }
        }
        groups
    }

    /// The shard specs, in declaration order.
    pub fn shards(&self) -> &[ShardSpec] {
        &self.dims
    }

    /// The divisor this layout applies to logical dim `dim` (after
    /// normalization): the product of the degrees of every shard spec whose
    /// normalized dim is `dim`. 1 when the axis is unsharded.
    ///
    /// Every shard dim is validated first, so a plan that names an axis the
    /// tensor does not have is reported whether or not it affects `dim`.
    ///
    /// # Errors
    ///
    /// [`ShardError::InvalidTensorRank`] if `tensor_rank` is negative,
    /// [`ShardError::DimOutOfRange`] if `dim` or a shard dim does not exist on
    /// a rank-`tensor_rank` tensor, or [`ShardError::GroupOutOfRange`] if a
    /// shard's mask bit is outside `mesh`.
    pub fn divisor(&self, dim: i64, tensor_rank: i64, mesh: &Mesh) -> Result<i64, ShardError> {
        let norm = DimNormalizer::new(tensor_rank)?;
        let target = norm.normalize(dim)?;
        let mut divisor: i64 = 1;
        for spec in &self.dims {
            let resolved = norm.normalize(spec.dim)?;
            spec.group.validate(mesh).map_err(to_shard_error)?;
            if resolved == target {
                let degree = degree_as_i64(spec.group.degree(mesh).expect("group validated"));
                divisor = divisor.saturating_mul(degree);
            }
        }
        Ok(divisor)
    }

    /// The local shape a rank holds: `local[d] = global[d] / divisor(d)`.
    ///
    /// **Non-divisibility is a hard compile-time error**, never a fallback and
    /// never a runtime check (`docs/architecture.md` §1.5): the description
    /// declared the shard, so the framework owes it the local shape — and if
    /// the declared axis does not divide, the declaration is unsatisfiable
    /// (`tp=3` with `num_attention_heads=16`).
    ///
    /// Every shard dim and every group (including the partial's) is validated
    /// even on a scalar, so a broken layout is reported rather than skipped.
    ///
    /// # Errors
    ///
    /// [`ShardError::NotDivisible`] (naming the dim, the global size and the
    /// divisor) when an axis does not divide, plus the validation errors of
    /// [`Self::divisor`].
    pub fn local_shape(&self, global: &[i64], mesh: &Mesh) -> Result<Vec<i64>, ShardError> {
        let tensor_rank = global.len() as i64;
        let norm = DimNormalizer::new(tensor_rank)?;
        for spec in &self.dims {
            norm.normalize(spec.dim)?;
            spec.group.validate(mesh).map_err(to_shard_error)?;
        }
        if let Some(partial) = &self.partial {
            partial.group.validate(mesh).map_err(to_shard_error)?;
        }
        let mut local = Vec::with_capacity(global.len());
        for (d, &size) in global.iter().enumerate() {
            let divisor = self.divisor(d as i64, tensor_rank, mesh)?;
            if size % divisor != 0 {
                return Err(ShardError::NotDivisible {
                    dim: d as i64,
                    global: size,
                    divisor,
                });
            }
            local.push(size / divisor);
        }
        Ok(local)
    }

    /// Human form with axis names, e.g. `shard(-1, tp)`, `shard(0, ep) +
    /// shard(1, tp)`, `partial(sum, tp)`, `replicate`.
    ///
    /// A mask bit outside `mesh` has no name; it renders as its raw bit form
    /// (the form [`ParallelLayout`]'s `Display` always uses) rather than
    /// failing, because `describe` is for messages and dumps, not validation.
    pub fn describe(&self, mesh: &Mesh) -> String {
        if self.is_replicated() {
            return "replicate".to_string();
        }
        let mut parts: Vec<String> =
            Vec::with_capacity(self.dims.len() + usize::from(self.partial.is_some()));
        for spec in &self.dims {
            parts.push(format!("shard({}, {})", spec.dim, name(mesh, spec.group)));
        }
        if let Some(partial) = &self.partial {
            parts.push(format!(
                "partial({}, {})",
                partial.op,
                name(mesh, partial.group)
            ));
        }
        parts.join(" + ")
    }
}

fn name(mesh: &Mesh, mask: GroupMask) -> String {
    mesh.group_name(mask).unwrap_or_else(|_| mask.to_string())
}

fn degree_as_i64(degree: usize) -> i64 {
    i64::try_from(degree).unwrap_or(i64::MAX)
}

/// A [`ParallelError::GroupOutOfRange`] carried as a [`ShardError`], so the
/// shape arithmetic can report a mask that does not belong to the mesh.
fn to_shard_error(err: ParallelError) -> ShardError {
    match err {
        ParallelError::GroupOutOfRange { bit, axes } => ShardError::GroupOutOfRange { bit, axes },
        other => unreachable!("only GroupOutOfRange can arise here, got: {other}"),
    }
}

impl fmt::Display for ParallelLayout {
    /// Compact form without a mesh, so masks render as their raw bits:
    /// `replicate`, `shard(-1, mask(0b1))`, `partial(sum, mask(0b1))`.
    ///
    /// Kept short because it appears inside error messages and plan dumps,
    /// where a reader compares two layouts at a glance.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_replicated() {
            return f.write_str("replicate");
        }
        let mut parts: Vec<String> =
            Vec::with_capacity(self.dims.len() + usize::from(self.partial.is_some()));
        for spec in &self.dims {
            parts.push(format!("shard({}, {})", spec.dim, spec.group));
        }
        if let Some(partial) = &self.partial {
            parts.push(format!("partial({}, {})", partial.op, partial.group));
        }
        f.write_str(&parts.join(" + "))
    }
}

/// Resolves logical dimension indices against the rank of a concrete tensor.
///
/// A plan is written once and instantiated on tensors whose rank is known only
/// at propagation time, so a layout stores a logical dim (`-1` = last axis) and
/// the axis becomes a concrete non-negative index here. This is also the only
/// place that decides whether a dim is legal at all: an axis outside
/// `0..rank` is a plan bug, and it is reported ([`ShardError::DimOutOfRange`])
/// whether or not the conversion that mentioned it happens to need data
/// movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimNormalizer {
    rank: i64,
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
        Ok(Self { rank })
    }

    /// The tensor rank this normalizer was built for.
    pub fn rank(&self) -> i64 {
        self.rank
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
    }

    #[test]
    fn negative_tensor_rank_is_rejected() {
        assert_eq!(
            DimNormalizer::new(-1).unwrap_err(),
            ShardError::InvalidTensorRank { rank: -1 }
        );
    }
}
