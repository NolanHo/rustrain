//! Mapping between a global rank and its coordinate on every parallel axis.

use std::fmt;

use crate::config::{ParallelConfig, ParallelDim};
use crate::error::ParallelError;

/// The coordinate of one global rank on every parallel axis.
///
/// # Rank order (contract)
///
/// The global rank is a mixed-radix number whose digits are
/// `[tp, cp, ep, dp, pp]` with **tensor varying fastest and pipeline varying
/// slowest**. Written out, a rank is
///
/// ```text
/// rank = ((((pp_rank * dp + dp_rank) * ep + ep_rank) * cp + cp_rank) * tp + tp_rank)
/// ```
///
/// where `tp`/`cp`/`ep`/`dp` are the *sizes* of those dimensions and the outer
/// `pp_rank` is the pipeline *coordinate*. Expanded with the sizes, every digit
/// has its own stride:
///
/// ```text
/// rank = tp_rank + tp*cp_rank + tp*cp*ep_rank + tp*cp*ep*dp_rank + tp*cp*ep*dp*pp_rank
/// ```
///
/// This is Megatron's default dimension order, so an existing checkpoint's
/// `tp_rank`/`ep_rank` keep their meaning when the same topology is replayed
/// here. It is a contract, not an implementation detail: **anything that
/// enumerates ranks in a different order (checkpoint resharding, plugin
/// world-size assumptions, NCCL communicator creation) must follow it too.**
/// The hand-computed table in `tests/topology.rs` pins the order so that a
/// reordering cannot land silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankLayout {
    cfg: ParallelConfig,
    tp: usize,
    cp: usize,
    ep: usize,
    dp: usize,
    pp: usize,
}

impl RankLayout {
    /// Resolves the coordinate of `rank` under `cfg`.
    ///
    /// # Errors
    ///
    /// [`ParallelError::ZeroDimension`] / [`ParallelError::WorldSizeOverflow`]
    /// if `cfg` is not a valid topology (a zero-sized dimension would make the
    /// decomposition below divide by zero), or
    /// [`ParallelError::RankOutOfRange`] if `rank >= cfg.world_size()`.
    pub fn from_rank(rank: usize, cfg: ParallelConfig) -> Result<Self, ParallelError> {
        cfg.validate()?;
        let world_size = cfg.world_size();
        if rank >= world_size {
            return Err(ParallelError::RankOutOfRange { rank, world_size });
        }
        // Peel off the fastest dimension first: this is the exact inverse of
        // `rank()`, and the round-trip test pins the two together. The final
        // remainder is the pipeline coordinate, which the range check above
        // guarantees is < pipeline size.
        let tp = rank % cfg.tensor;
        let rest = rank / cfg.tensor;
        let cp = rest % cfg.context;
        let rest = rest / cfg.context;
        let ep = rest % cfg.expert;
        let rest = rest / cfg.expert;
        let dp = rest % cfg.data;
        let pp = rest / cfg.data;
        Ok(Self {
            cfg,
            tp,
            cp,
            ep,
            dp,
            pp,
        })
    }

    /// The global rank this coordinate maps back to.
    ///
    /// A literal transcription of the contract formula documented on this type;
    /// keep the two in sync (the compiler cannot check a comment).
    pub fn rank(&self) -> usize {
        ((((self.pp * self.cfg.data + self.dp) * self.cfg.expert + self.ep) * self.cfg.context
            + self.cp)
            * self.cfg.tensor)
            + self.tp
    }

    /// The topology this coordinate was resolved against.
    pub fn config(&self) -> ParallelConfig {
        self.cfg
    }

    /// Tensor-parallel coordinate.
    pub fn tp_rank(&self) -> usize {
        self.tp
    }

    /// Context/sequence-parallel coordinate.
    pub fn cp_rank(&self) -> usize {
        self.cp
    }

    /// Expert-parallel coordinate.
    pub fn ep_rank(&self) -> usize {
        self.ep
    }

    /// Data-parallel coordinate.
    pub fn dp_rank(&self) -> usize {
        self.dp
    }

    /// Pipeline-parallel coordinate.
    pub fn pp_rank(&self) -> usize {
        self.pp
    }

    /// The coordinate on `dim`.
    pub fn coordinate(&self, dim: ParallelDim) -> usize {
        match dim {
            ParallelDim::Tp => self.tp,
            ParallelDim::Cp => self.cp,
            ParallelDim::Ep => self.ep,
            ParallelDim::Dp => self.dp,
            ParallelDim::Pp => self.pp,
        }
    }
}

impl fmt::Display for RankLayout {
    /// Fastest dimension first, which is also the order of the digits in the
    /// rank formula.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tp={},cp={},ep={},dp={},pp={}",
            self.tp, self.cp, self.ep, self.dp, self.pp
        )
    }
}
