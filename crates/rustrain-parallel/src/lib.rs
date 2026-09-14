//! Mesh, rank layout, parallel sharding specs and collective inference.
//!
//! Deliverable D3 of `docs/design/kernel-first/spec.md` (§2.5). This crate is
//! pure math over a topology: it knows nothing about the plugin ABI
//! (`rustrain-abi`), the operator registry (`rustrain-ops`) or the plan IR
//! (`rustrain-plan`), and it pulls in no GPU/torch dependency (invariant I-1).
//!
//! It answers three questions:
//!
//! 1. **Topology** — given a [`Mesh`] (the compile input, built from a
//!    [`ParallelConfig`]) and a global `rank`, which coordinate is that rank on
//!    each axis ([`RankLayout`]), and which ranks does it share a group with
//!    for any [`GroupMask`] ([`Mesh::group_ranks`])?
//! 2. **Sharding** — which part of a tensor does a rank hold? That is a
//!    [`ParallelLayout`]: several independent `(dim, group)` shards plus at
//!    most one partial, a piece of *data* that travels with a plan slot
//!    (invariant I-3), not an implicit property of a kernel.
//! 3. **Communication** — what is the minimal ordered sequence of collectives
//!    that converts one layout into another ([`transitions`])?
//!
//! The third question is why this crate exists. The previous framework pasted
//! the same hand-written `all_reduce` calls into the training loop three times,
//! and the rules for when each was needed lived in the authors' heads. Here
//! they live in one function, with one test per rule.
//!
//! A mask only means something next to the mesh that produced it, and the plan
//! only ever carries the mesh's *fingerprint* ([`MeshFingerprint`], invariant
//! I-6) — never the traversable mesh itself.
//!
//! Everything is deterministic: every result is a pure function of its inputs
//! and no hash iteration order can leak into an output.

pub mod collective;
pub mod config;
pub mod error;
pub mod layout;
pub mod mesh;
pub mod rank;

pub use collective::{Collective, transitions};
pub use config::{ParallelConfig, ParallelDim};
pub use error::{ParallelError, ShardError};
pub use layout::{DimNormalizer, ParallelLayout, PartialSpec, ReduceOp, ShardSpec, ShardMode};
pub use mesh::{GroupMask, Mesh, MeshFingerprint};
pub use rank::RankLayout;
