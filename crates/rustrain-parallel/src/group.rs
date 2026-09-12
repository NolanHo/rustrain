//! Process groups: which ranks take part in which collective.
//!
//! A collective is always issued over a *group*, never over "the world" by
//! accident: a TP all-reduce and a DP all-reduce are different communications
//! even when the world size is the same. [`ProcessGroups`] derives every group
//! from the topology once, so the rest of the framework only ever handles
//! `(rank, GroupKind)` pairs.

use serde::{Deserialize, Serialize};

use crate::config::{ParallelConfig, ParallelDim};
use crate::error::ParallelError;
use std::fmt;

/// Which group a collective runs over.
///
/// [`GroupKind::Global`] is the whole world. It is a real group and not a
/// wildcard: a `Global` all-reduce and a `Tp` all-reduce are distinct, and a
/// layout on one group cannot be converted into a layout on another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupKind {
    /// Tensor parallel.
    Tp,
    /// Context/sequence parallel.
    Cp,
    /// Expert parallel.
    Ep,
    /// Data parallel.
    Dp,
    /// Pipeline parallel.
    Pp,
    /// Every rank in the job.
    Global,
}

impl GroupKind {
    /// Every kind, dimensions in topology order with `Global` last.
    ///
    /// Frozen so that any iteration over kinds (group construction, reporting,
    /// plan digests) is deterministic.
    pub const ALL: [GroupKind; 6] = [
        GroupKind::Tp,
        GroupKind::Cp,
        GroupKind::Ep,
        GroupKind::Dp,
        GroupKind::Pp,
        GroupKind::Global,
    ];

    /// Stable lowercase name, used by `Display`.
    pub const fn as_str(self) -> &'static str {
        match self {
            GroupKind::Tp => "tp",
            GroupKind::Cp => "cp",
            GroupKind::Ep => "ep",
            GroupKind::Dp => "dp",
            GroupKind::Pp => "pp",
            GroupKind::Global => "global",
        }
    }

    /// The topology dimension this group splits, or `None` for
    /// [`GroupKind::Global`].
    ///
    /// This is the single mapping between "which collective group" and "which
    /// parallel axis". Keeping it here (instead of matching on both enums at
    /// each call site) is what lets the plan compiler and the runtime agree on
    /// what a `GroupKind` means.
    pub const fn dim(self) -> Option<ParallelDim> {
        match self {
            GroupKind::Tp => Some(ParallelDim::Tp),
            GroupKind::Cp => Some(ParallelDim::Cp),
            GroupKind::Ep => Some(ParallelDim::Ep),
            GroupKind::Dp => Some(ParallelDim::Dp),
            GroupKind::Pp => Some(ParallelDim::Pp),
            GroupKind::Global => None,
        }
    }

    /// The group for a topology dimension; inverse of [`Self::dim`].
    pub const fn from_dim(dim: ParallelDim) -> GroupKind {
        dim.group_kind()
    }

    /// Index into the fixed-size group table of [`ProcessGroups`].
    pub(crate) const fn index(self) -> usize {
        match self {
            GroupKind::Tp => 0,
            GroupKind::Cp => 1,
            GroupKind::Ep => 2,
            GroupKind::Dp => 3,
            GroupKind::Pp => 4,
            GroupKind::Global => 5,
        }
    }
}

impl fmt::Display for GroupKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A concrete set of ranks that issue a collective together.
///
/// `ranks` is always ascending, which makes "the rank's index inside the group"
/// (the value a collective ultimately needs, e.g. as a broadcast source) a
/// stable property of the vector rather than of the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessGroup {
    /// The kind this group belongs to.
    pub kind: GroupKind,
    /// Member ranks, ascending.
    pub ranks: Vec<usize>,
}

impl ProcessGroup {
    /// Number of members.
    pub fn size(&self) -> usize {
        self.ranks.len()
    }

    /// Whether `rank` is a member.
    pub fn contains(&self, rank: usize) -> bool {
        self.ranks.contains(&rank)
    }
}

/// Every process group of a topology.
///
/// Built once per job and then queried by index; nothing here allocates on the
/// query path and nothing iterates a hash map, so results are deterministic and
/// cheap enough to consult from a scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessGroups {
    cfg: ParallelConfig,
    world_size: usize,
    /// `groups[kind.index()]`: every group of that kind, ordered by their first
    /// member. A fixed-size array keyed by [`GroupKind::index`] rather than a
    /// map, because a map's iteration order would leak into reports.
    groups: [Vec<ProcessGroup>; 6],
}

impl ProcessGroups {
    /// Builds every group of `cfg`.
    ///
    /// # Panics
    ///
    /// Panics if `cfg` is not a valid topology, because there is no group
    /// structure for e.g. a zero-sized dimension. Topology that comes from a
    /// config file must be validated at load time; when that is not possible,
    /// use [`ProcessGroups::try_new`] and report the error.
    pub fn new(cfg: ParallelConfig) -> Self {
        Self::try_new(cfg).unwrap_or_else(|err| panic!("invalid parallel config: {err}"))
    }

    /// Builds every group of `cfg`, reporting an invalid topology instead of
    /// panicking.
    ///
    /// # Errors
    ///
    /// [`ParallelError::ZeroDimension`] or [`ParallelError::WorldSizeOverflow`]
    /// if `cfg` is not a valid topology.
    pub fn try_new(cfg: ParallelConfig) -> Result<Self, ParallelError> {
        cfg.validate()?;
        let world_size = cfg.world_size();
        let mut groups: [Vec<ProcessGroup>; 6] = std::array::from_fn(|_| Vec::new());
        for kind in GroupKind::ALL {
            let (stride, extent) = stride_extent(&cfg, kind);
            let count = world_size / extent;
            let mut of_kind = Vec::with_capacity(count);
            for id in 0..count {
                // Inverse of `group_id` below. A group is the set of ranks
                // sharing every coordinate except the one this kind splits, so
                // its members are an arithmetic sequence with the dimension's
                // stride.
                let outer = id / stride;
                let fastest = id % stride;
                let base = outer * stride * extent + fastest;
                let ranks = (0..extent).map(|j| base + j * stride).collect();
                of_kind.push(ProcessGroup { kind, ranks });
            }
            groups[kind.index()] = of_kind;
        }
        tracing::debug!(
            world_size,
            tensor = cfg.tensor,
            context = cfg.context,
            expert = cfg.expert,
            data = cfg.data,
            pipeline = cfg.pipeline,
            "parallel topology resolved"
        );
        Ok(Self {
            cfg,
            world_size,
            groups,
        })
    }

    /// The topology these groups were derived from.
    pub fn config(&self) -> ParallelConfig {
        self.cfg
    }

    /// Number of ranks.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Every group of one kind, ordered by their first member.
    pub fn groups(&self, kind: GroupKind) -> &[ProcessGroup] {
        &self.groups[kind.index()]
    }

    /// How many groups of this kind exist.
    pub fn group_count(&self, kind: GroupKind) -> usize {
        let (_, extent) = stride_extent(&self.cfg, kind);
        self.world_size / extent
    }

    /// How many ranks are in one group of this kind; for
    /// [`GroupKind::Global`] this is the world size.
    pub fn group_size(&self, kind: GroupKind) -> usize {
        stride_extent(&self.cfg, kind).1
    }

    /// The group `rank` belongs to, for `kind`.
    ///
    /// # Errors
    ///
    /// [`ParallelError::RankOutOfRange`] if `rank >= world_size()`.
    pub fn group_of(&self, rank: usize, kind: GroupKind) -> Result<&ProcessGroup, ParallelError> {
        let id = self.group_id(rank, kind)?;
        Ok(&self.groups[kind.index()][id])
    }

    /// The position of `rank` inside its group of `kind`: `0..group_size()`.
    ///
    /// This is what a collective sometimes needs instead of the global rank —
    /// a broadcast source, or the slice index of a sharded tensor. For a
    /// dimension-backed kind it equals that rank's coordinate on the dimension,
    /// and for [`GroupKind::Global`] it equals the global rank.
    ///
    /// # Errors
    ///
    /// [`ParallelError::RankOutOfRange`] if `rank >= world_size()`.
    pub fn group_index(&self, rank: usize, kind: GroupKind) -> Result<usize, ParallelError> {
        if rank >= self.world_size {
            return Err(ParallelError::RankOutOfRange {
                rank,
                world_size: self.world_size,
            });
        }
        let (stride, extent) = stride_extent(&self.cfg, kind);
        Ok((rank / stride) % extent)
    }

    /// Index of `rank`'s group inside [`Self::groups`].
    fn group_id(&self, rank: usize, kind: GroupKind) -> Result<usize, ParallelError> {
        if rank >= self.world_size {
            return Err(ParallelError::RankOutOfRange {
                rank,
                world_size: self.world_size,
            });
        }
        let (stride, extent) = stride_extent(&self.cfg, kind);
        // The dimensions slower than this one select the group; the dimensions
        // faster than it are carried along as `rank % stride`. For `Global`
        // (stride 1, extent = world) this is always group 0.
        Ok((rank / (stride * extent)) * stride + (rank % stride))
    }
}

/// `(stride, extent)` of a group kind: members of a group are
/// `base + j * stride` for `j in 0..extent`.
///
/// The stride is the product of the dimensions *faster* than this one in the
/// rank order, and the extent is the size of the dimension itself. Both are
/// derived from the same rank formula as [`crate::RankLayout`], which is why
/// the strides read as a running product.
///
/// The caller must have validated the config: each partial product here is at
/// most the world size, so multiplication cannot overflow once the world size
/// itself fits in a `usize`.
fn stride_extent(cfg: &ParallelConfig, kind: GroupKind) -> (usize, usize) {
    let tp = cfg.tensor;
    let cp = cfg.context;
    let ep = cfg.expert;
    let dp = cfg.data;
    match kind {
        GroupKind::Tp => (1, tp),
        GroupKind::Cp => (tp, cp),
        GroupKind::Ep => (tp * cp, ep),
        GroupKind::Dp => (tp * cp * ep, dp),
        GroupKind::Pp => (tp * cp * ep * dp, cfg.pipeline),
        // The world is the group that splits nothing: stride 1, all ranks.
        GroupKind::Global => (1, cfg.world_size()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(tp: usize, cp: usize, ep: usize, dp: usize, pp: usize) -> ParallelConfig {
        ParallelConfig {
            tensor: tp,
            context: cp,
            expert: ep,
            data: dp,
            pipeline: pp,
        }
    }

    /// The closed-form group lookup must agree with the groups that were built:
    /// `group_of` picks the group containing the rank, and `group_index` is the
    /// rank's position inside it. This is the only non-obvious arithmetic in the
    /// crate, so it is checked exhaustively over a spread of topologies.
    #[test]
    fn lookup_matches_construction() {
        let configs = [
            cfg(1, 1, 1, 1, 1),
            cfg(2, 2, 2, 2, 1),
            cfg(2, 2, 1, 2, 1),
            cfg(2, 1, 1, 2, 2),
            cfg(3, 5, 1, 1, 1),
            cfg(1, 1, 1, 1, 4),
            cfg(4, 1, 1, 1, 1),
        ];
        for c in configs {
            let groups = ProcessGroups::new(c);
            for kind in GroupKind::ALL {
                assert_eq!(
                    groups.groups(kind).len(),
                    groups.group_count(kind),
                    "{c:?} {kind}"
                );
                for group in groups.groups(kind) {
                    assert_eq!(group.kind, kind);
                    assert_eq!(group.size(), groups.group_size(kind));
                    assert!(
                        group.ranks.windows(2).all(|w| w[0] < w[1]),
                        "{c:?} {kind}: ranks must be ascending: {:?}",
                        group.ranks
                    );
                }
                for (id, group) in groups.groups(kind).iter().enumerate() {
                    for (pos, rank) in group.ranks.iter().enumerate() {
                        assert_eq!(groups.group_id(*rank, kind).unwrap(), id, "{c:?} {kind}");
                        assert_eq!(
                            groups.group_index(*rank, kind).unwrap(),
                            pos,
                            "{c:?} {kind}"
                        );
                        assert_eq!(groups.group_of(*rank, kind).unwrap(), group);
                    }
                }
            }
            // Every rank of a kind appears in exactly one group of that kind.
            let mut seen = vec![0usize; c.world_size()];
            for kind in GroupKind::ALL {
                for group in groups.groups(kind) {
                    for rank in &group.ranks {
                        seen[*rank] += 1;
                    }
                }
                assert!(
                    seen.iter().all(|count| *count == kind.index() + 1),
                    "{c:?} {kind}: groups must partition the world"
                );
            }
        }
    }

    /// The position of a rank inside a dimension's group is that rank's
    /// coordinate on the dimension; the group-of-global position is the rank.
    #[test]
    fn group_index_is_the_dimension_coordinate() {
        let c = cfg(2, 2, 2, 2, 1);
        let groups = ProcessGroups::new(c);
        for rank in 0..c.world_size() {
            let layout = crate::RankLayout::from_rank(rank, c).unwrap();
            for dim in [
                ParallelDim::Tp,
                ParallelDim::Cp,
                ParallelDim::Ep,
                ParallelDim::Dp,
                ParallelDim::Pp,
            ] {
                let kind = GroupKind::from_dim(dim);
                assert_eq!(
                    groups.group_index(rank, kind).unwrap(),
                    layout.coordinate(dim),
                    "rank {rank} on {dim}"
                );
            }
            assert_eq!(groups.group_index(rank, GroupKind::Global).unwrap(), rank);
        }
    }
}
