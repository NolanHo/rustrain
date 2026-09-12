//! Process groups, rank layout, parallel sharding specs and collective inference.
//!
//! Deliverable D3 of `docs/design/kernel-first/spec.md` (§2.5). This crate is
//! pure math over a topology: it knows nothing about the plugin ABI
//! (`rustrain-abi`), the operator registry (`rustrain-ops`) or the plan IR
//! (`rustrain-plan`), and it pulls in no GPU/torch dependency (invariant I-1).
//!
//! It answers three questions:
//!
//! 1. **Topology** — given a [`ParallelConfig`] and a global `rank`, which
//!    coordinate is that rank on each parallel axis ([`RankLayout`]), and which
//!    ranks does it share a process group with ([`ProcessGroups`])?
//! 2. **Sharding** — which part of a tensor does a rank hold? That is a
//!    [`ParallelLayout`]: a piece of *data* that travels with a plan slot
//!    (invariant I-3), not an implicit property of a kernel.
//! 3. **Communication** — what is the minimal ordered sequence of collectives
//!    that converts one layout into another ([`transitions`])?
//!
//! The third question is why this crate exists. The previous framework pasted
//! the same hand-written `all_reduce` calls into the training loop three times,
//! and the rules for when each was needed lived in the authors' heads. Here
//! they live in one function, with one test per rule.
//!
//! Everything is deterministic: every result is a pure function of its inputs
//! and no hash iteration order can leak into an output.

pub mod collective;
pub mod config;
pub mod error;
pub mod group;
pub mod layout;
pub mod rank;

pub use collective::{Collective, transitions};
pub use config::{ParallelConfig, ParallelDim};
pub use error::{ParallelError, ShardError};
pub use group::{GroupKind, ProcessGroup, ProcessGroups};
pub use layout::{DEFAULT_SEQUENCE_DIM, DimNormalizer, EXPERT_DIM, ParallelLayout, ReduceOp};
pub use rank::RankLayout;
