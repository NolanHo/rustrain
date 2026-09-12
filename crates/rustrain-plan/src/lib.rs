//! Plan IR, compilation and validation.
//!
//! The compiler turns a [`Plan`] — a pure description of an operator graph —
//! into a [`CompiledPlan`]: every node bound to a concrete implementation, every
//! tensor's sharding reconciled by inserted collectives, and a digest that
//! covers the whole decision.
//!
//! Nothing in this crate computes. Nothing in this crate reads the environment.

// `PlanError` carries structured diagnostics (a layout, a sharding failure, the
// candidate table from resolution) because those are what make the message
// actionable. Boxing it to shrink the `Err` variant would push a deref onto
// every caller for no benefit at the scale a compile runs at.
#![allow(clippy::result_large_err)]

pub mod attrs;
pub mod compile;
pub mod ir;
pub mod shard;

pub use attrs::{AbiAttrs, AttrValue, Attrs};
pub use compile::{CompiledPlan, CompiledStep, Compiler, ResolvedNode, StreamId};
pub use ir::{
    CheckpointPolicy, NodeId, OpRef, Phase, Plan, PlanBuilder, PlanMeta, PlanNode,
    PrecisionOverride, Slot, SlotId, SlotKind, StreamPolicy, Trace, intrinsic,
};

use rustrain_parallel::{GroupKind, ShardError};

/// Everything that can make a plan invalid or unresolvable.
///
/// The distinction that matters: [`PlanError::Resolve`] means "no implementation
/// could run here", the rest mean "the graph itself is wrong". Both are hard
/// errors — there is no fallback path (spec contract R-1).
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("node {node:?} references slot {slot:?}, which does not exist")]
    UnknownSlot { node: NodeId, slot: SlotId },

    #[error(
        "node {node:?} consumes slot {slot:?} which is produced by later node {produced_by:?}; \
         plan nodes must be emitted in topological order"
    )]
    NotTopological {
        node: NodeId,
        slot: SlotId,
        produced_by: NodeId,
    },

    #[error("slot {slot:?} is written by both node {first:?} and node {second:?}")]
    SlotWrittenTwice {
        slot: SlotId,
        first: NodeId,
        second: NodeId,
    },

    #[error("node {node:?} declares no outputs; every node must produce something")]
    NodeWithoutOutput { node: NodeId },

    #[error("the plan has no nodes")]
    EmptyPlan,

    #[error("node {node:?} ({op}) failed to resolve: {source}")]
    Resolve {
        node: NodeId,
        op: String,
        #[source]
        source: rustrain_ops::ResolveError,
    },

    #[error("node {node:?} ({op}) cannot take part in validation: {reason}")]
    NotValidatable {
        node: NodeId,
        op: String,
        reason: String,
    },

    #[error("node {node:?} ({op}) has {found} inputs but the implementation expects {expected}")]
    ArityMismatch {
        node: NodeId,
        op: String,
        expected: usize,
        found: usize,
    },

    #[error("node {node:?} ({op}) shape inference failed with status {status}: {message}")]
    InferFailed {
        node: NodeId,
        op: String,
        status: i32,
        message: String,
    },

    #[error("node {node:?} ({op}) has no shape inference, so the plan cannot be validated")]
    InferMissing { node: NodeId, op: String },

    #[error("node {node:?} ({op}) inferred output {index} shape {inferred:?} but slot {slot:?} declares {declared:?}")]
    InferredShapeMismatch {
        node: NodeId,
        op: String,
        index: usize,
        slot: SlotId,
        inferred: Vec<i64>,
        declared: Vec<i64>,
    },

    #[error("node {node:?} input {index} needs layout {needed} but slot {slot:?} holds {held}")]
    LayoutConflict {
        node: NodeId,
        slot: SlotId,
        index: usize,
        needed: String,
        held: String,
    },

    #[error("cannot reconcile sharding for node {node:?}: {source}")]
    Shard {
        node: NodeId,
        #[source]
        source: ShardError,
    },

    #[error("no sharding rule applies to node {node:?}: {source}")]
    ShardDerivation {
        node: NodeId,
        #[source]
        source: shard::DeriveError,
    },

    #[error(
        "node {node:?} ({op}) is not deterministic but the plan requires determinism; \
         set deterministic = false in the recipe to allow it"
    )]
    Nondeterministic { node: NodeId, op: String },

    #[error("node {node:?} names the unknown intrinsic `{op}`")]
    UnknownIntrinsic { node: NodeId, op: String },

    #[error("intrinsic {op} is missing required attribute `{attr}`")]
    IntrinsicMissingAttr { op: String, attr: String },

    #[error("intrinsic {op} has an unparseable `{attr}` value `{value}`")]
    IntrinsicBadAttr {
        op: String,
        attr: String,
        value: String,
    },

    #[error(
        "node {node:?} ({op}) requires group {group:?} but the parallel topology does not provide it"
    )]
    GroupUnavailable {
        node: NodeId,
        op: String,
        group: GroupKind,
    },

    #[error("plan digest computation failed: {0}")]
    Digest(String),
}

pub type Result<T> = std::result::Result<T, PlanError>;
