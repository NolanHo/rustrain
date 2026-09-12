//! The plan IR: slots, nodes, and the builder that produces them.
//!
//! A plan is *data*. Nothing here computes; the compiler resolves each node to a
//! concrete operator implementation and the runtime drives the result.
//!
//! Shapes are required to be concrete at build time. Symbolic dimensions would
//! push conditionals into the compiler and the digest for very little payoff —
//! the model builder already knows the configuration.

use serde::{Deserialize, Serialize};

use rustrain_parallel::{ParallelConfig, ParallelLayout};

use crate::attrs::Attrs;

pub use rustrain_ops::Phase;

/// Index into [`Plan::slots`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct SlotId(pub usize);

/// Index into [`Plan::nodes`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct NodeId(pub usize);

/// What a slot is for. Drives memory planning and the run manifest, and lets
/// the validator catch obviously wrong wiring (e.g. a weight as a loss).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum SlotKind {
    /// Model weight, usually read-only during a step.
    Weight,
    /// Gradient accumulator.
    Gradient,
    /// Optimizer state (m, v).
    State,
    /// Forward/backward activation.
    Activation,
    /// Short-lived scratch.
    Temp,
    /// Externally supplied input (tokens, masks).
    Input,
    /// Produced output consumed by the caller (loss, logits).
    Output,
}

/// A named tensor in the plan.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Slot {
    pub name: String,
    pub dtype: rustrain_abi::ffi::RsDtype,
    pub shape: Vec<i64>,
    /// How the tensor is distributed. See spec invariant I-3.
    pub layout: ParallelLayout,
    pub kind: SlotKind,
}

impl Slot {
    pub fn numel(&self) -> i64 {
        self.shape.iter().product()
    }

    /// Byte size of the element buffer, or `None` for a dtype with no
    /// whole-byte width (sub-byte packing is not implemented).
    pub fn element_bytes(&self) -> Option<u64> {
        let width = self.dtype.byte_width()?;
        Some(self.numel().max(0) as u64 * width as u64)
    }

    /// Element count along one (possibly negative) dimension.
    pub fn dim(&self, axis: i64) -> Option<i64> {
        let r = self.shape.len() as i64;
        let a = if axis < 0 { axis + r } else { axis };
        if a < 0 || a >= r {
            None
        } else {
            Some(self.shape[a as usize])
        }
    }
}

/// Where a node came from. Purely diagnostic; never affects semantics, but it is
/// what makes `plan explain` readable when a 60-layer model has 4000 nodes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trace {
    pub path: String,
    /// Set when the compiler inserted the node rather than the model builder.
    pub inserted_by: Option<String>,
}

impl Trace {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            inserted_by: None,
        }
    }

    pub fn inserted(path: impl Into<String>, by: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            inserted_by: Some(by.into()),
        }
    }

    pub fn is_inserted(&self) -> bool {
        self.inserted_by.is_some()
    }
}

/// Whether the activation this node produces is kept, recomputed on the
/// backward pass, or parked in host memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum CheckpointPolicy {
    #[default]
    None,
    /// Drop the activations, recompute them during backward.
    Recompute,
    /// Keep the activations but move them to host memory.
    Offload,
}

/// Which CUDA stream the node's work is issued on.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum StreamPolicy {
    /// The step's main stream; ordered with everything else on it.
    #[default]
    Default,
    /// A side stream, used so a collective can overlap with compute.
    /// The scheduler is responsible for inserting the synchronization.
    Side,
}

/// Reference to an operator, optionally pinning the implementation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpRef {
    pub name: String,
    /// `None` means "let the recipe decide".
    pub variant: Option<String>,
}

impl OpRef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            variant: None,
        }
    }

    pub fn variant(name: impl Into<String>, variant: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            variant: Some(variant.into()),
        }
    }

    pub fn display(&self) -> String {
        match &self.variant {
            Some(v) => format!("{}@{v}", self.name),
            None => self.name.clone(),
        }
    }
}

/// Per-node precision override, taking precedence over the recipe.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PrecisionOverride {
    pub in_dtype: Option<rustrain_abi::ffi::RsDtype>,
    pub out_dtype: Option<rustrain_abi::ffi::RsDtype>,
    pub accum_dtype: Option<rustrain_abi::ffi::RsDtype>,
    pub grad_dtype: Option<rustrain_abi::ffi::RsDtype>,
}

impl PrecisionOverride {
    pub fn is_empty(&self) -> bool {
        self.in_dtype.is_none()
            && self.out_dtype.is_none()
            && self.accum_dtype.is_none()
            && self.grad_dtype.is_none()
    }
}

/// One operator application.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanNode {
    pub op: OpRef,
    pub inputs: Vec<SlotId>,
    pub outputs: Vec<SlotId>,
    pub attrs: Attrs,
    pub phase: Phase,
    #[serde(default, skip_serializing_if = "PrecisionOverride::is_empty")]
    pub precision: PrecisionOverride,
    #[serde(default)]
    pub checkpoint: CheckpointPolicy,
    #[serde(default)]
    pub stream: StreamPolicy,
    pub source: Trace,
}

impl PlanNode {
    pub fn display_op(&self) -> String {
        self.op.display()
    }
}

/// Plan-level metadata. Everything here feeds the digest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanMeta {
    pub name: String,
    /// The phase this whole plan belongs to. Individual nodes may differ while
    /// a model is being staged, but a compiled plan is normally single-phase.
    pub phase: Phase,
    pub parallel: ParallelConfig,
    pub seed: u64,
}

/// A complete operator graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    pub meta: PlanMeta,
    pub slots: Vec<Slot>,
    pub nodes: Vec<PlanNode>,
}

impl Plan {
    pub fn slot(&self, id: SlotId) -> &Slot {
        &self.slots[id.0]
    }

    pub fn slot_mut(&mut self, id: SlotId) -> &mut Slot {
        &mut self.slots[id.0]
    }

    pub fn node(&self, id: NodeId) -> &PlanNode {
        &self.nodes[id.0]
    }

    pub fn slot_id(&self, name: &str) -> Option<SlotId> {
        self.slots.iter().position(|s| s.name == name).map(SlotId)
    }

    /// Node indices that produce each slot.
    pub fn producers(&self) -> Vec<Option<NodeId>> {
        let mut out = vec![None; self.slots.len()];
        for (i, n) in self.nodes.iter().enumerate() {
            for o in &n.outputs {
                out[o.0] = Some(NodeId(i));
            }
        }
        out
    }

    /// Structural checks that must hold before any semantic validation.
    ///
    /// Nodes are required to be in topological order: a node may only consume
    /// slots produced by an earlier node or declared as inputs/weights. That
    /// keeps the executor a single forward walk, with no scheduling search.
    pub fn check_structure(&self) -> Result<(), crate::PlanError> {
        let producers = self.producers();

        for (i, n) in self.nodes.iter().enumerate() {
            for inp in &n.inputs {
                if inp.0 >= self.slots.len() {
                    return Err(crate::PlanError::UnknownSlot {
                        node: NodeId(i),
                        slot: *inp,
                    });
                }
                if let Some(p) = producers[inp.0]
                    && p.0 >= i
                {
                    return Err(crate::PlanError::NotTopological {
                        node: NodeId(i),
                        slot: *inp,
                        produced_by: p,
                    });
                }
            }
            for out in &n.outputs {
                if out.0 >= self.slots.len() {
                    return Err(crate::PlanError::UnknownSlot {
                        node: NodeId(i),
                        slot: *out,
                    });
                }
                if let Some(p) = producers[out.0]
                    && p.0 != i
                {
                    return Err(crate::PlanError::SlotWrittenTwice {
                        slot: *out,
                        first: p,
                        second: NodeId(i),
                    });
                }
            }
            if n.outputs.is_empty() {
                return Err(crate::PlanError::NodeWithoutOutput { node: NodeId(i) });
            }
        }
        Ok(())
    }

    /// Slot ids that are never produced by a node — these are the plan's inputs.
    pub fn input_slots(&self) -> Vec<SlotId> {
        self.producers()
            .into_iter()
            .enumerate()
            .filter_map(|(i, p)| p.is_none().then_some(SlotId(i)))
            .collect()
    }

    /// Slot ids produced but never consumed.
    pub fn dangling_slots(&self) -> Vec<SlotId> {
        let mut consumed = vec![false; self.slots.len()];
        for n in &self.nodes {
            for i in &n.inputs {
                consumed[i.0] = true;
            }
        }
        (0..self.slots.len())
            .filter(|i| !consumed[*i])
            .map(SlotId)
            .collect()
    }
}

/// Ergonomic plan construction.
///
/// The builder is intentionally dumb: it appends nodes in call order, so the
/// caller is responsible for emitting them in topological order. That matches
/// how a model is written (layer 0, layer 1, ...) and avoids a scheduling pass.
pub struct PlanBuilder {
    meta: PlanMeta,
    slots: Vec<Slot>,
    nodes: Vec<PlanNode>,
    trace_prefix: String,
}

impl PlanBuilder {
    pub fn new(name: impl Into<String>, phase: Phase, parallel: ParallelConfig) -> Self {
        Self {
            meta: PlanMeta {
                name: name.into(),
                phase,
                parallel,
                seed: 0,
            },
            slots: Vec::new(),
            nodes: Vec::new(),
            trace_prefix: String::new(),
        }
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.meta.seed = seed;
        self
    }

    /// Sets a prefix applied to every subsequent [`Trace`] path.
    pub fn scope(&mut self, prefix: impl Into<String>) {
        self.trace_prefix = prefix.into();
    }

    pub fn slot(
        &mut self,
        name: impl Into<String>,
        dtype: rustrain_abi::ffi::RsDtype,
        shape: Vec<i64>,
        kind: SlotKind,
    ) -> SlotId {
        self.slot_with_layout(name, dtype, shape, kind, ParallelLayout::Replicate)
    }

    pub fn slot_with_layout(
        &mut self,
        name: impl Into<String>,
        dtype: rustrain_abi::ffi::RsDtype,
        shape: Vec<i64>,
        kind: SlotKind,
        layout: ParallelLayout,
    ) -> SlotId {
        let id = SlotId(self.slots.len());
        self.slots.push(Slot {
            name: name.into(),
            dtype,
            shape,
            layout,
            kind,
        });
        id
    }

    pub fn node(
        &mut self,
        op: OpRef,
        inputs: Vec<SlotId>,
        outputs: Vec<SlotId>,
        attrs: Attrs,
        source: impl AsRef<str>,
    ) -> NodeId {
        self.node_in_phase(op, inputs, outputs, attrs, source, self.meta.phase)
    }

    pub fn node_in_phase(
        &mut self,
        op: OpRef,
        inputs: Vec<SlotId>,
        outputs: Vec<SlotId>,
        attrs: Attrs,
        source: impl AsRef<str>,
        phase: Phase,
    ) -> NodeId {
        let path = if self.trace_prefix.is_empty() {
            source.as_ref().to_string()
        } else {
            format!("{}.{}", self.trace_prefix, source.as_ref())
        };
        let id = NodeId(self.nodes.len());
        self.nodes.push(PlanNode {
            op,
            inputs,
            outputs,
            attrs,
            phase,
            precision: PrecisionOverride::default(),
            checkpoint: CheckpointPolicy::None,
            stream: StreamPolicy::Default,
            source: Trace::new(path),
        });
        id
    }

    pub fn current_node(&mut self, id: NodeId) -> &mut PlanNode {
        &mut self.nodes[id.0]
    }

    pub fn compare_nodes(&self) -> &[PlanNode] {
        &self.nodes
    }

    pub fn build(self) -> Result<Plan, crate::PlanError> {
        let plan = Plan {
            meta: self.meta,
            slots: self.slots,
            nodes: self.nodes,
        };
        plan.check_structure()?;
        Ok(plan)
    }
}

/// Operators the runtime implements itself rather than looking up in the
/// registry: the collectives that sharding propagation inserts, and
/// synchronization.
///
/// They live in the same IR as plugin operators so that the plan is a complete
/// description of what happens, and so `plan explain` can show where
/// communication lands (spec contract S-2).
pub mod intrinsic {
    /// Reserved prefix. The primitive vocabulary has no dots, so a name that
    /// starts with this cannot collide with an operator.
    ///
    /// The prefix is not cosmetic: `broadcast` is both a primitive (a view that
    /// stretches a size-1 dim) and, before this, an intrinsic. The compiler
    /// checks for intrinsics first, so a plan calling the primitive was handed
    /// to the collective path and rejected for a missing `group` attribute. A
    /// reserved namespace is what makes the two sets disjoint by construction
    /// rather than by nobody happening to pick the same word twice.
    pub const PREFIX: &str = "intrinsic.";

    pub const ALL_REDUCE: &str = "intrinsic.all_reduce";
    pub const ALL_GATHER: &str = "intrinsic.all_gather";
    pub const REDUCE_SCATTER: &str = "intrinsic.reduce_scatter";
    pub const BROADCAST: &str = "intrinsic.broadcast";
    pub const SYNC: &str = "intrinsic.sync";

    pub fn is_intrinsic(name: &str) -> bool {
        name.starts_with(PREFIX)
            && matches!(
                name,
                ALL_REDUCE | ALL_GATHER | REDUCE_SCATTER | BROADCAST | SYNC
            )
    }

    /// Attribute key carrying the [`super::ParallelLayout`]-ish group for an
    /// inserted collective, as a string (`"tp"`, `"ep"`, ...).
    pub const ATTR_GROUP: &str = "group";
    /// Attribute key carrying the reduction for `all_reduce`.
    pub const ATTR_REDUCE: &str = "reduce";
    /// Attribute key carrying the dimension for gather/scatter.
    pub const ATTR_DIM: &str = "dim";

    pub fn group_name(g: rustrain_parallel::GroupKind) -> &'static str {
        match g {
            rustrain_parallel::GroupKind::Tp => "tp",
            rustrain_parallel::GroupKind::Cp => "cp",
            rustrain_parallel::GroupKind::Ep => "ep",
            rustrain_parallel::GroupKind::Dp => "dp",
            rustrain_parallel::GroupKind::Pp => "pp",
            rustrain_parallel::GroupKind::Global => "global",
        }
    }

    pub fn parse_group(s: &str) -> Option<rustrain_parallel::GroupKind> {
        use rustrain_parallel::GroupKind::*;
        match s {
            "tp" => Some(Tp),
            "cp" => Some(Cp),
            "ep" => Some(Ep),
            "dp" => Some(Dp),
            "pp" => Some(Pp),
            "global" => Some(Global),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustrain_abi::ffi::RsDtype;
    use rustrain_parallel::GroupKind;

    fn builder() -> PlanBuilder {
        PlanBuilder::new("t", Phase::Forward, ParallelConfig::default())
    }

    #[test]
    fn builder_produces_topological_plan() {
        let mut b = builder();
        let x = b.slot("x", RsDtype::F32, vec![2, 4], SlotKind::Activation);
        let w = b.slot("w", RsDtype::F32, vec![4, 4], SlotKind::Weight);
        let y = b.slot("y", RsDtype::F32, vec![2, 4], SlotKind::Activation);
        b.node(OpRef::new("matmul"), vec![x, w], vec![y], Attrs::new(), "l0");
        let p = b.build().unwrap();
        assert_eq!(p.nodes.len(), 1);
        assert_eq!(p.input_slots(), vec![x, w]);
        assert_eq!(p.dangling_slots(), vec![y]);
    }

    #[test]
    fn forward_reference_is_rejected() {
        let mut b = builder();
        let a = b.slot("a", RsDtype::F32, vec![2], SlotKind::Activation);
        let mid = b.slot("mid", RsDtype::F32, vec![2], SlotKind::Activation);
        let late = b.slot("late", RsDtype::F32, vec![2], SlotKind::Activation);
        // node 0 consumes `late`, which only node 1 produces.
        b.node(OpRef::new("op0"), vec![late], vec![mid], Attrs::new(), "0");
        b.node(OpRef::new("op1"), vec![mid], vec![late], Attrs::new(), "1");
        let err = b.build().unwrap_err();
        assert!(
            matches!(err, crate::PlanError::NotTopological { .. }),
            "got {err:?}"
        );
        let _ = a;
    }

    #[test]
    fn writing_a_slot_twice_is_rejected() {
        let mut b = builder();
        let x = b.slot("x", RsDtype::F32, vec![2], SlotKind::Activation);
        let y = b.slot("y", RsDtype::F32, vec![2], SlotKind::Activation);
        let z = b.slot("z", RsDtype::F32, vec![2], SlotKind::Activation);
        b.node(OpRef::new("a"), vec![x], vec![z], Attrs::new(), "a");
        b.node(OpRef::new("b"), vec![y], vec![z], Attrs::new(), "b");
        let err = b.build().unwrap_err();
        assert!(
            matches!(err, crate::PlanError::SlotWrittenTwice { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn negative_dim_lookup() {
        let s = Slot {
            name: "s".into(),
            dtype: RsDtype::F32,
            shape: vec![2, 3, 4],
            layout: ParallelLayout::Replicate,
            kind: SlotKind::Activation,
        };
        assert_eq!(s.dim(-1), Some(4));
        assert_eq!(s.dim(-3), Some(2));
        assert_eq!(s.dim(0), Some(2));
        assert_eq!(s.dim(3), None);
        assert_eq!(s.numel(), 24);
    }

    #[test]
    fn intrinsic_names_round_trip() {
        use intrinsic::*;
        assert!(is_intrinsic(ALL_REDUCE));
        assert!(!is_intrinsic("matmul"));
        for g in [
            GroupKind::Tp,
            GroupKind::Cp,
            GroupKind::Ep,
            GroupKind::Dp,
            GroupKind::Pp,
            GroupKind::Global,
        ] {
            assert_eq!(parse_group(group_name(g)), Some(g));
        }
    }
}
