//! Operator registry, capability filtering and recipe-driven resolution.
//!
//! This crate answers the only two questions the rest of the framework asks
//! about kernels (spec §2.3, `docs/design/kernel-first/spec.md`):
//!
//! * **what exists** — [`Registry`] is the single source of truth, and it is
//!   populated only by loading plugin `.so`s (invariant I-2: an implementation
//!   is never linked into the framework).
//! * **what should run here** — [`Registry::resolve`] applies a [`Recipe`] to a
//!   target environment and returns exactly one [`RegisteredOp`], or an error
//!   that lists every candidate for the operator together with the specific
//!   reason each one was rejected (contract R-1).
//!
//! Two rules shape every decision in this crate:
//!
//! * **Contract R-1 — no silent fallback.** A variant named by `prefer` either
//!   runs or the resolution fails; it never degrades to something else. An
//!   implementation that the recipe did not name is never selected implicitly,
//!   even when the resolver can see that it would have worked.
//! * **Contract R-2 — degradation is explicit and recorded.** Only variants
//!   listed in `fallback` may be substituted, and every skipped candidate is
//!   kept on the [`ResolvedOp`] so the plan digest, the log and
//!   `rustrain plan explain` can report what was passed over.
//!
//! Configuration comes from recipe files, never from the process environment
//! (contract CF-2), and every ordering that can reach a digest is sorted, never
//! hash-map order.
//!
//! ```
//! use rustrain_ops::{Phase, Recipe, TargetEnv};
//!
//! let recipe = Recipe::from_toml(r#"
//!     [kernel]
//!     default = "reference"
//!     [kernel.ops.rmsnorm]
//!     forward = "reference.f32"
//! "#).unwrap();
//!
//! assert_eq!(recipe.variant_for("rmsnorm", Phase::Forward), Some("reference.f32"));
//! assert_eq!(recipe.variant_for("rmsnorm", Phase::Backward), None);
//! let _ = TargetEnv::default();
//! ```

pub mod capability;
pub mod recipe;
pub mod registered;
pub mod registry;
pub mod vocab;

pub use capability::{
    GROUP_ORDER, Phase, RejectReason, TargetEnv, backward_name, declared_device, device_name,
    group_name, phase_reject_reason, reject_for, reject_reason,
};
pub use recipe::{
    AUTODIFF, ActivationPolicy, BackwardPlan, DtypeName, MemoryPool, MemoryRecipe, OpMemoryRecipe,
    OpRecipe, OptimizerState, ParallelRecipe, PrecisionRecipe, QuantName, Recipe, RecipeError,
    ScaleName,
};
pub use registered::{OpSummary, RegisteredOp};
pub use registry::{
    CandidateRejection, Registry, RegistryError, ResolutionFailure, ResolveError, ResolveRequest,
    ResolvedOp,
};
pub use vocab::{COMPOSITE_OPS, is_composite_op};

#[cfg(test)]
mod tests;
