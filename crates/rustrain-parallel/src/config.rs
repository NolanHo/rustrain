//! The parallel topology: five dimensions and their sizes.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::ParallelError;
use crate::group::GroupKind;

/// One axis of the parallel topology.
///
/// The axes are independent: their sizes multiply into the world size, and a
/// rank's coordinate on each of them is packed into the global rank by
/// [`crate::RankLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelDim {
    /// Tensor parallel: weights/activations split inside a layer.
    Tp,
    /// Context (sequence) parallel: the sequence axis is split.
    Cp,
    /// Expert parallel: the MoE expert axis is split.
    Ep,
    /// Data parallel: independent replicas, one per batch shard.
    Dp,
    /// Pipeline parallel: the layer stack is split into stages.
    Pp,
}

impl ParallelDim {
    /// Stable lowercase name, used by `Display` and by [`GroupKind`].
    pub const fn as_str(self) -> &'static str {
        match self {
            ParallelDim::Tp => "tp",
            ParallelDim::Cp => "cp",
            ParallelDim::Ep => "ep",
            ParallelDim::Dp => "dp",
            ParallelDim::Pp => "pp",
        }
    }

    /// The process group this dimension indexes.
    ///
    /// Kept next to `Display` so that there is exactly one place mapping a
    /// topology axis to a collective group; [`GroupKind::dim`] is its inverse.
    pub const fn group_kind(self) -> GroupKind {
        match self {
            ParallelDim::Tp => GroupKind::Tp,
            ParallelDim::Cp => GroupKind::Cp,
            ParallelDim::Ep => GroupKind::Ep,
            ParallelDim::Dp => GroupKind::Dp,
            ParallelDim::Pp => GroupKind::Pp,
        }
    }
}

impl fmt::Display for ParallelDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Size of every parallel dimension.
///
/// Field names match the recipe's `[kernel.parallel]` table (`tensor`,
/// `context`, `expert`, `data`, `pipeline`); the *rank* order is fixed
/// separately, by [`crate::RankLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Tensor parallel size.
    pub tensor: usize,
    /// Context/sequence parallel size.
    pub context: usize,
    /// Expert parallel size.
    pub expert: usize,
    /// Data parallel size.
    pub data: usize,
    /// Pipeline parallel size.
    pub pipeline: usize,
}

impl Default for ParallelConfig {
    /// A single rank: every dimension is 1.
    ///
    /// Written out rather than derived, because `Default` on `usize` is 0 and a
    /// zero-sized dimension is not a topology — it is an error
    /// ([`ParallelError::ZeroDimension`]).
    fn default() -> Self {
        Self {
            tensor: 1,
            context: 1,
            expert: 1,
            data: 1,
            pipeline: 1,
        }
    }
}

impl ParallelConfig {
    /// Every dimension with its size, in a fixed order.
    ///
    /// Fixed order (tp, cp, ep, dp, pp) is what makes validation and tracing
    /// deterministic; it is not the rank order.
    pub const fn dimensions(&self) -> [(ParallelDim, usize); 5] {
        [
            (ParallelDim::Tp, self.tensor),
            (ParallelDim::Cp, self.context),
            (ParallelDim::Ep, self.expert),
            (ParallelDim::Dp, self.data),
            (ParallelDim::Pp, self.pipeline),
        ]
    }

    /// The size of one dimension.
    pub const fn dimension(&self, dim: ParallelDim) -> usize {
        match dim {
            ParallelDim::Tp => self.tensor,
            ParallelDim::Cp => self.context,
            ParallelDim::Ep => self.expert,
            ParallelDim::Dp => self.data,
            ParallelDim::Pp => self.pipeline,
        }
    }

    /// Total number of ranks: `tensor * context * expert * data * pipeline`.
    ///
    /// Saturating rather than wrapping on overflow: a wrapped product could
    /// look like a plausible world size, and this value drives rank range
    /// checks. [`Self::validate`] is what reports the overflow.
    pub const fn world_size(&self) -> usize {
        self.tensor
            .saturating_mul(self.context)
            .saturating_mul(self.expert)
            .saturating_mul(self.data)
            .saturating_mul(self.pipeline)
    }

    /// `world_size()`, or `None` if the product does not fit in a `usize`.
    pub const fn checked_world_size(&self) -> Option<usize> {
        let mut product = 1usize;
        let factors = [
            self.tensor,
            self.context,
            self.expert,
            self.data,
            self.pipeline,
        ];
        let mut i = 0;
        while i < factors.len() {
            product = match product.checked_mul(factors[i]) {
                Some(p) => p,
                None => return None,
            };
            i += 1;
        }
        Some(product)
    }

    /// Checks that this config describes a usable topology.
    ///
    /// Rejects a zero-sized dimension and a product that does not fit in a
    /// `usize`. The two overlap: for non-negative sizes a product of 0 is
    /// exactly the case where some dimension is 0, so the zero product is
    /// reported as [`ParallelError::ZeroDimension`] (naming the offending
    /// dimension rather than just the product).
    ///
    /// # Errors
    ///
    /// [`ParallelError::ZeroDimension`] if any dimension is 0,
    /// [`ParallelError::WorldSizeOverflow`] if the five sizes do not multiply
    /// into a `usize`.
    pub fn validate(&self) -> Result<(), ParallelError> {
        for (dim, size) in self.dimensions() {
            if size == 0 {
                return Err(ParallelError::ZeroDimension { dim });
            }
        }
        // Every dimension is >= 1 here, so the only remaining failure is that
        // the product does not fit.
        match self.checked_world_size() {
            Some(_) => Ok(()),
            None => Err(ParallelError::WorldSizeOverflow {
                tensor: self.tensor,
                context: self.context,
                expert: self.expert,
                data: self.data,
                pipeline: self.pipeline,
            }),
        }
    }
}
