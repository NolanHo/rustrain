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
pub mod instantiate;
pub mod ir;
pub mod memory;
pub mod shard;

pub use attrs::{AbiAttrs, AttrValue, Attrs};
pub use compile::{CompiledPlan, CompiledStep, Compiler, ResolvedNode, StreamId};
pub use instantiate::{DeclaredAxes, InstanceStage, instantiate, instantiate_stages};
pub use ir::{
    CheckpointPolicy, NodeId, OpRef, Phase, Plan, PlanBuilder, PlanMeta, PlanNode,
    PrecisionOverride, Slot, SlotId, SlotKind, StreamPolicy, Trace, intrinsic,
};
pub use memory::{
    Lifetime, MemoryPlan, Placement, PolicyDecision, RuntimeCapabilities, SlotAllocation,
};

use rustrain_parallel::{GroupMask, ParallelError, ShardError};

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

    #[error(
        "node {node:?} ({op}) inferred output {index} shape {inferred:?} but slot {slot:?} declares {declared:?}"
    )]
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
        "node {node:?} ({op}) uses group {group}, which the plan's mesh does not provide: \
         a mask bit addresses an axis the mesh does not have"
    )]
    GroupUnavailable {
        node: NodeId,
        op: String,
        group: GroupMask,
    },

    #[error(
        "slot `{slot}` declares axis `{axis}` on dim {dim}, which the mesh does not provide \
         (mesh axes: {axes}); a declared axis must name an axis of the mesh — the name-level \
         counterpart of a group mask bit outside the mesh"
    )]
    UnknownAxis {
        slot: String,
        dim: String,
        axis: String,
        axes: String,
    },

    #[error("declaration names slot `{slot}`, which the plan does not have")]
    UnknownDeclaredSlot { slot: String },

    #[error("declaration for slot `{slot}` names dim `{dim}`, which is not an integer")]
    BadDeclaredDim { slot: String, dim: String },

    #[error("slot `{slot}` cannot be instantiated: {source}")]
    Instantiate {
        slot: String,
        #[source]
        source: ShardError,
    },

    #[error(
        "the mesh's `pp` degree is {pp}, but instance `{prefix}` declares no stage; an \
         absent stage means stage 0 only when `pp` is 1"
    )]
    MissingStage { prefix: String, pp: usize },

    #[error(
        "instance `{prefix}` declares stage {stage}, but the mesh's `pp` degree is {pp}; \
         every stage must be in 0..{pp}"
    )]
    StageOutOfRange {
        prefix: String,
        stage: i64,
        pp: usize,
    },

    #[error("rank {rank} is out of range for the mesh's world size {world_size}")]
    RankOutOfRange { rank: usize, world_size: usize },

    #[error("the plan's mesh fingerprint does not describe a valid mesh: {source}")]
    Mesh {
        #[source]
        source: ParallelError,
    },

    #[error("plan digest computation failed: {0}")]
    Digest(String),

    #[error(
        "projected peak memory {peak} B exceeds the {budget} B budget; the peak is reached at node \
         {hottest_node} ({hottest_op}) with {hottest_bytes} B live.\n  - {}",
        .suggestions.join("\n  - ")
    )]
    MemoryBudgetExceeded {
        peak: u64,
        budget: u64,
        hottest_node: usize,
        hottest_op: String,
        hottest_bytes: u64,
        suggestions: Vec<String>,
    },
}

pub type Result<T> = std::result::Result<T, PlanError>;
