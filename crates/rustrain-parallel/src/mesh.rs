//! The mesh: an ordered, named axis list, and the open mask vocabulary over it.
//!
//! A mesh is the *compile input* that turns a topology-independent description
//! into a concrete plan (invariant I-6): it never travels inside a plan — only
//! its fingerprint does. [`GroupMask`] is the open replacement for the old
//! closed `GroupKind`: bit `i` addresses mesh axis `i`, so any combination of
//! axes (`tp|ep`, all five, none) is a first-class group instead of six
//! hard-coded slots. A mask only means something next to the mesh that produced
//! it, which is why every mask-taking method here takes `&Mesh` and why the
//! plan stores a [`MeshFingerprint`] next to its masks.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::config::ParallelConfig;
use crate::error::ParallelError;

/// A bit set over mesh axes: bit `i` = the axis at index `i` of the mesh.
///
/// The mask carries no axis *names* and no *degrees*: those live in the mesh.
/// `Display` therefore renders the raw bits (`mask(0b101)`); a human name
/// (`tp|ep`) needs a mesh, via [`Mesh::group_name`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GroupMask(u32);

impl GroupMask {
    /// The empty mask: a size-1 group containing only the rank itself.
    ///
    /// Legal, not an error: a size-1 group is the no-op case, the same way a
    /// degree-1 axis is the no-op case. This is what lets `tp=1` and group
    /// mount/unmount share one code path.
    pub const NONE: GroupMask = GroupMask(0);

    /// Wraps a raw bit pattern.
    ///
    /// No validation: a mask may address axes a particular mesh does not have.
    /// The mesh is what rejects it ([`Mesh::group_ranks`] and friends), not the
    /// constructor — this is where a plan stops being portable across meshes.
    pub const fn from_bits(bits: u32) -> Self {
        GroupMask(bits)
    }

    /// The raw bit pattern.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// The mask containing only `axis`.
    ///
    /// # Errors
    ///
    /// [`ParallelError::AxisOutOfRange`] if `axis` does not fit in the mask's
    /// [`Mesh::MAX_AXES`] bits.
    pub fn single(axis: usize) -> Result<Self, ParallelError> {
        if axis >= Mesh::MAX_AXES {
            return Err(ParallelError::AxisOutOfRange {
                axis,
                max: Mesh::MAX_AXES,
            });
        }
        Ok(GroupMask(1u32 << axis))
    }

    /// Whether `axis` is in the mask. `axis >= MAX_AXES` is simply not in it.
    pub const fn contains(self, axis: usize) -> bool {
        axis < Mesh::MAX_AXES && (self.0 & (1u32 << axis)) != 0
    }

    /// Whether the mask addresses no axis at all.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Bitwise union: the group spanning both masks' axes.
    pub const fn union(self, other: Self) -> Self {
        GroupMask(self.0 | other.0)
    }

    /// Bitwise intersection: the axes both masks address.
    pub const fn intersect(self, other: Self) -> Self {
        GroupMask(self.0 & other.0)
    }

    /// The axes of `self` that are not in `other`.
    pub const fn without(self, other: Self) -> Self {
        GroupMask(self.0 & !other.0)
    }

    /// How many axes the mask addresses.
    pub const fn axis_count(self) -> u32 {
        self.0.count_ones()
    }

    /// The product of the degrees of the masked axes: the size of the group.
    ///
    /// 1 for the empty mask; saturating rather than wrapping on overflow.
    ///
    /// # Errors
    ///
    /// [`ParallelError::GroupOutOfRange`] when a bit is outside `mesh`.
    pub fn degree(self, mesh: &Mesh) -> Result<usize, ParallelError> {
        self.validate(mesh)?;
        let mut degree = 1usize;
        for (axis, (_, extent)) in mesh.axes.iter().enumerate() {
            if self.contains(axis) {
                degree = degree.saturating_mul(*extent);
            }
        }
        Ok(degree)
    }

    /// Checks that every bit addresses an axis of `mesh`.
    ///
    /// This is the portability gate: a mask is only meaningful with the mesh
    /// that produced it, and this is where a plan from another mesh is caught.
    ///
    /// # Errors
    ///
    /// [`ParallelError::GroupOutOfRange`] when a bit is outside `mesh`.
    pub fn validate(self, mesh: &Mesh) -> Result<(), ParallelError> {
        let axes = mesh.axis_count();
        let stray = (0..Mesh::MAX_AXES).find(|&axis| self.contains(axis) && axis >= axes);
        match stray {
            Some(bit) => Err(ParallelError::GroupOutOfRange { bit, axes }),
            None => Ok(()),
        }
    }
}

impl fmt::Display for GroupMask {
    /// The raw bits; axis names need a mesh ([`Mesh::group_name`]).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mask({:#b})", self.0)
    }
}

/// An ordered axis list. `axes[0]` varies fastest: the stride of axis `i` in
/// the rank number is the product of the degrees before it, and a rank's
/// coordinates are its mixed-radix digits in axis order.
///
/// The canonical five axes are `[tp, cp, ep, dp, pp]` (see
/// [`Mesh::from_config`]), but the mesh itself is open: any names, any order,
/// up to [`Mesh::MAX_AXES`] axes. Degrees may be 1 (a degree-1 axis is a legal,
/// no-op axis, not a special case); a mask with a bit *outside* the mesh is the
/// error case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mesh {
    axes: Vec<(String, usize)>,
}

impl Mesh {
    /// Most axes a mesh may declare; also the number of bits in a
    /// [`GroupMask`].
    pub const MAX_AXES: usize = 32;

    /// Builds a mesh, validating the axis list.
    ///
    /// # Errors
    ///
    /// - [`ParallelError::NoAxes`] if the list is empty;
    /// - [`ParallelError::TooManyAxes`] if it has more than
    ///   [`Mesh::MAX_AXES`] axes;
    /// - [`ParallelError::EmptyAxisName`] / [`ParallelError::DuplicateAxis`]
    ///   if a name is empty or appears twice;
    /// - [`ParallelError::ZeroDegree`] if a degree is 0.
    pub fn new(axes: Vec<(String, usize)>) -> Result<Self, ParallelError> {
        if axes.is_empty() {
            return Err(ParallelError::NoAxes {
                max: Self::MAX_AXES,
            });
        }
        if axes.len() > Self::MAX_AXES {
            return Err(ParallelError::TooManyAxes {
                count: axes.len(),
                max: Self::MAX_AXES,
            });
        }
        let mut seen: Vec<&str> = Vec::with_capacity(axes.len());
        for (index, (name, degree)) in axes.iter().enumerate() {
            if name.is_empty() {
                return Err(ParallelError::EmptyAxisName { index });
            }
            if *degree == 0 {
                return Err(ParallelError::ZeroDegree { name: name.clone() });
            }
            if seen.contains(&name.as_str()) {
                return Err(ParallelError::DuplicateAxis { name: name.clone() });
            }
            seen.push(name.as_str());
        }
        // The product must fit: every rank number, stride and group id below is
        // derived from it, and a saturated product would silently enumerate
        // ranks that do not exist instead of reporting a topology that cannot
        // be laid out.
        let mut world_size = 1usize;
        for (_, degree) in &axes {
            world_size = world_size.checked_mul(*degree).ok_or_else(|| {
                ParallelError::MeshWorldSizeOverflow {
                    degrees: axes.iter().map(|(_, degree)| *degree).collect(),
                }
            })?;
        }
        tracing::debug!(?axes, world_size, "mesh resolved");
        Ok(Self { axes })
    }

    /// The canonical five axes in rank order:
    /// `[("tp", tensor), ("cp", context), ("ep", expert), ("dp", data),
    /// ("pp", pipeline)]` — the same order [`crate::RankLayout`] packs ranks
    /// in, so `tp` varies fastest and `pp` slowest.
    ///
    /// Infallible: five valid axes, each degree at least 1. All five axes are
    /// kept **even when a degree is 1**, so a mask over such an axis is a
    /// size-1 group — legal, a no-op — rather than an out-of-range bit. The
    /// caller must have validated the config ([`ParallelConfig::validate`]
    /// rejects zero degrees); a zero degree here would make the mesh's rank
    /// arithmetic divide by zero.
    pub fn from_config(cfg: &ParallelConfig) -> Self {
        use crate::config::ParallelDim;
        Self {
            axes: vec![
                (ParallelDim::Tp.as_str().to_string(), cfg.tensor),
                (ParallelDim::Cp.as_str().to_string(), cfg.context),
                (ParallelDim::Ep.as_str().to_string(), cfg.expert),
                (ParallelDim::Dp.as_str().to_string(), cfg.data),
                (ParallelDim::Pp.as_str().to_string(), cfg.pipeline),
            ],
        }
    }

    /// The ordered `(name, degree)` axis list.
    pub fn axes(&self) -> &[(String, usize)] {
        &self.axes
    }

    /// Number of axes.
    pub fn axis_count(&self) -> usize {
        self.axes.len()
    }

    /// The degree of `axis`, or `None` if `axis` does not exist.
    pub fn degree(&self, axis: usize) -> Option<usize> {
        self.axes.get(axis).map(|(_, degree)| *degree)
    }

    /// The index of the axis named `name`, or `None` if the mesh has none.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.axes.iter().position(|(n, _)| n == name)
    }

    /// Total number of ranks: the product of the degrees, saturating rather
    /// than wrapping on overflow (a wrapped product could look like a
    /// plausible world size).
    pub fn world_size(&self) -> usize {
        self.axes
            .iter()
            .fold(1usize, |acc, (_, degree)| acc.saturating_mul(*degree))
    }

    /// The stride of `axis` in the rank number: the product of the degrees of
    /// the axes before it (`axes[0]` has stride 1). `None` if `axis` does not
    /// exist. Saturating on overflow.
    pub fn stride(&self, axis: usize) -> Option<usize> {
        if axis >= self.axis_count() {
            return None;
        }
        let mut stride = 1usize;
        for (_, degree) in &self.axes[..axis] {
            stride = stride.saturating_mul(*degree);
        }
        Some(stride)
    }

    /// The plan's copy of the mesh: the ordered `[(name, degree)]` list and
    /// nothing else — no rank lists, no traversable topology (invariant I-6).
    pub fn fingerprint(&self) -> MeshFingerprint {
        MeshFingerprint {
            axes: self.axes.clone(),
        }
    }

    /// Human name of a mask: the joined axis names (`"tp"`, `"tp|ep"`),
    /// `"global"` for the full mask, `"none"` for the empty one.
    ///
    /// # Errors
    ///
    /// [`ParallelError::GroupOutOfRange`] when a bit is outside this mesh.
    pub fn group_name(&self, mask: GroupMask) -> Result<String, ParallelError> {
        mask.validate(self)?;
        Ok(group_name_impl(self.axes(), mask))
    }

    /// The members of `mask`'s group for the group containing `rank`, in
    /// ascending order.
    ///
    /// Members are `rank` with every masked coordinate replaced by each value
    /// in `0..degree_i`, enumerated so that the slowest masked axis varies
    /// slowest — equivalently the arithmetic sequence of the group. Ascending
    /// order is a contract: the runtime reads the member list positionally.
    ///
    /// # Errors
    ///
    /// [`ParallelError::GroupOutOfRange`] if a bit is outside this mesh, or
    /// [`ParallelError::RankOutOfRange`] if `rank >= world_size()`.
    pub fn group_ranks(&self, mask: GroupMask, rank: usize) -> Result<Vec<usize>, ParallelError> {
        self.check(mask, rank)?;
        // The members of a group are `base + Σ c_i · stride_i` over every
        // combination of masked coordinates `c_i`. Decoding the mixed-radix
        // counter `j` (least-significant digit = first masked axis) enumerates
        // them ascending, because the strides grow with the axis index.
        let mut base = rank;
        let mut terms: Vec<(usize, usize)> = Vec::new(); // (stride_i, degree_i)
        for axis in 0..self.axis_count() {
            if !mask.contains(axis) {
                continue;
            }
            let degree = self.degree(axis).expect("mask validated");
            let stride = self.stride(axis).expect("axis in range");
            // Zero the masked coordinate in `base`.
            base -= ((rank / stride) % degree) * stride;
            terms.push((stride, degree));
        }
        let count = mask.degree(self).expect("mask validated");
        Ok((0..count)
            .map(|j| {
                let mut member = base;
                let mut weight = 1usize;
                for (stride, degree) in &terms {
                    let digit = (j / weight) % degree;
                    member = member.saturating_add(digit * stride);
                    weight = weight.saturating_mul(*degree);
                }
                member
            })
            .collect())
    }

    /// The index of `rank` inside its group of `mask`: mixed radix over the
    /// masked axes in axis order (`Σ_j c_{i_j} · Π_{l<j} degree_{i_l}`).
    ///
    /// For a single axis this is the coordinate on that axis — the value a
    /// collective needs as a broadcast source or a shard slice index. `0` for
    /// the empty mask.
    ///
    /// # Errors
    ///
    /// [`ParallelError::GroupOutOfRange`] if a bit is outside this mesh, or
    /// [`ParallelError::RankOutOfRange`] if `rank >= world_size()`.
    pub fn group_index(&self, mask: GroupMask, rank: usize) -> Result<usize, ParallelError> {
        self.check(mask, rank)?;
        // Mixed radix over the masked axes in axis order: the first masked axis
        // is the least significant digit (`Σ_j c_{i_j} · Π_{l<j} degree_{i_l}`).
        let mut index = 0usize;
        let mut weight = 1usize;
        for axis in 0..self.axis_count() {
            if !mask.contains(axis) {
                continue;
            }
            let degree = self.degree(axis).expect("axis in range");
            let stride = self.stride(axis).expect("axis in range");
            let digit = (rank / stride) % degree;
            index = index.saturating_add(digit.saturating_mul(weight));
            weight = weight.saturating_mul(degree);
        }
        Ok(index)
    }

    /// Which group of `mask` `rank` belongs to: mixed radix over the
    /// *unmasked* axes in axis order.
    ///
    /// For a single axis this reproduces the historical group id, so the
    /// "groups ordered by first member" contract carries over. `0` for the
    /// full mask (the whole world is one group).
    ///
    /// # Errors
    ///
    /// [`ParallelError::GroupOutOfRange`] if a bit is outside this mesh, or
    /// [`ParallelError::RankOutOfRange`] if `rank >= world_size()`.
    pub fn group_id(&self, mask: GroupMask, rank: usize) -> Result<usize, ParallelError> {
        self.check(mask, rank)?;
        // Mixed radix over the *unmasked* axes in axis order: the first
        // unmasked axis is the least significant digit.
        let mut id = 0usize;
        let mut weight = 1usize;
        for axis in 0..self.axis_count() {
            if mask.contains(axis) {
                continue;
            }
            let degree = self.degree(axis).expect("axis in range");
            let stride = self.stride(axis).expect("axis in range");
            let digit = (rank / stride) % degree;
            id = id.saturating_add(digit.saturating_mul(weight));
            weight = weight.saturating_mul(degree);
        }
        Ok(id)
    }

    /// Shared validation of `(mask, rank)` for the group queries.
    fn check(&self, mask: GroupMask, rank: usize) -> Result<(), ParallelError> {
        mask.validate(self)?;
        let world_size = self.world_size();
        if rank >= world_size {
            return Err(ParallelError::RankOutOfRange { rank, world_size });
        }
        Ok(())
    }
}

fn group_name_impl(axes: &[(String, usize)], mask: GroupMask) -> String {
    if mask.is_empty() {
        return "none".to_string();
    }
    let full = (0..axes.len()).all(|axis| mask.contains(axis));
    if full {
        return "global".to_string();
    }
    axes.iter()
        .enumerate()
        .filter(|(axis, _)| mask.contains(*axis))
        .map(|(_, (name, _))| name.as_str())
        .collect::<Vec<_>>()
        .join("|")
}

/// Ordered `[(name, degree)]` — what enters a plan.
///
/// A fingerprint is *results*, not topology: it has no rank lists and no
/// traversable mesh (invariant I-6). It serializes deterministically as
/// `{"axes": [[name, degree], ...]}` (a vec, never a map).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshFingerprint {
    axes: Vec<(String, usize)>,
}

impl MeshFingerprint {
    /// The ordered `(name, degree)` axis list.
    pub fn axes(&self) -> &[(String, usize)] {
        &self.axes
    }

    /// Total number of ranks: the product of the degrees, saturating.
    pub fn world_size(&self) -> usize {
        self.axes
            .iter()
            .fold(1usize, |acc, (_, degree)| acc.saturating_mul(*degree))
    }

    /// Human name of a mask, exactly as [`Mesh::group_name`] renders it.
    ///
    /// # Errors
    ///
    /// [`ParallelError::GroupOutOfRange`] when a bit is outside the axes.
    pub fn group_name(&self, mask: GroupMask) -> Result<String, ParallelError> {
        let axes = self.axes.len();
        let stray = (0..Mesh::MAX_AXES).find(|&axis| mask.contains(axis) && axis >= axes);
        match stray {
            Some(bit) => Err(ParallelError::GroupOutOfRange { bit, axes }),
            None => Ok(group_name_impl(self.axes(), mask)),
        }
    }

    /// The mesh this fingerprint describes, re-validated.
    ///
    /// This is the bridge a plan uses to check a mask against the topology it was compiled for: a
    /// fingerprint travels as data, so its axes are only trustworthy once [`Mesh::new`] has accepted
    /// them again (axis count, names, degrees, and a world size that fits in a `usize`).
    ///
    /// # Errors
    ///
    /// Whatever [`Mesh::new`] reports for this axis list.
    pub fn to_mesh(&self) -> Result<Mesh, ParallelError> {
        Mesh::new(self.axes.clone())
    }
}
