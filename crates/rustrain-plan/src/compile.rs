//! Compiling a plan: resolve every node, reconcile sharding, validate, flatten.
//!
//! The order matters. Sharding propagation runs first because it *changes the
//! graph* (it splices in collectives); validating before that would validate a
//! graph that is not the one that runs. Resolution runs next so that every later
//! check has an implementation to ask. Shape inference runs last, because it is
//! the only check that needs the resolved descriptors.

use serde::Serialize;

use rustrain_abi::ffi::{RsNumerics, RsTensor};
use rustrain_ops::{Phase, Recipe, RegisteredOp, Registry, ResolveRequest, TargetEnv};
use rustrain_parallel::{GroupMask, MeshFingerprint, ParallelLayout, ReduceOp};

use crate::PlanError;
use crate::attrs::{AbiAttrs, AttrValue};
use crate::ir::{NodeId, Plan, Slot, SlotId, StreamPolicy, Trace, intrinsic};
use crate::memory;
use crate::shard::{self, InsertedCollective};

/// A CUDA stream. The scaffold has exactly two; the scheduler assigns them.
pub type StreamId = u32;

/// The step's main stream: everything is ordered with respect to it.
pub const MAIN_STREAM: StreamId = 0;
/// A side stream used for collectives that are meant to overlap with compute.
pub const SIDE_STREAM: StreamId = 1;

/// One executable step. Either a plugin operator or a framework intrinsic.
pub enum CompiledStep {
    Op {
        node: NodeId,
        op: RegisteredOp,
        numerics: RsNumerics,
        /// Owns the C strings and slices the plugin will read.
        attrs: AbiAttrs,
        inputs: Vec<SlotId>,
        outputs: Vec<SlotId>,
        stream: StreamId,
        phase: Phase,
        source: Trace,
    },
    /// A collective the compiler inserted, or the plan declared explicitly.
    /// Executed by the runtime through `rs_services::collective` — it is never
    /// looked up in the registry (spec contract S-2).
    Intrinsic {
        node: NodeId,
        op: String,
        group: GroupMask,
        reduce: Option<ReduceOp>,
        dim: Option<i64>,
        input: SlotId,
        output: SlotId,
        stream: StreamId,
        source: Trace,
    },
}

impl CompiledStep {
    pub fn node(&self) -> NodeId {
        match self {
            CompiledStep::Op { node, .. } | CompiledStep::Intrinsic { node, .. } => *node,
        }
    }

    pub fn source(&self) -> &Trace {
        match self {
            CompiledStep::Op { source, .. } | CompiledStep::Intrinsic { source, .. } => source,
        }
    }

    pub fn stream(&self) -> StreamId {
        match self {
            CompiledStep::Op { stream, .. } | CompiledStep::Intrinsic { stream, .. } => *stream,
        }
    }

    /// Short label used by `plan explain` and by runtime logs.
    ///
    /// The group renders as its raw mask bits: a name needs the mesh, which a
    /// bare step does not carry (`CompiledPlan::explain` renders names through
    /// the private `label_with`).
    pub fn label(&self) -> String {
        match self {
            CompiledStep::Op { op, .. } => op.spec_name(),
            CompiledStep::Intrinsic { op, group, .. } => format!("{op}[{group}]"),
        }
    }

    /// Like [`Self::label`], with the group rendered by axis name against the
    /// mesh the plan was compiled for.
    fn label_with(&self, mesh: &MeshFingerprint) -> String {
        match self {
            CompiledStep::Op { op, .. } => op.spec_name(),
            CompiledStep::Intrinsic { op, group, .. } => {
                format!(
                    "{op}[{}]",
                    mesh.group_name(*group)
                        .unwrap_or_else(|_| group.to_string())
                )
            }
        }
    }
}

/// `(implementation, skipped candidates with their reasons)`. Named because the
/// tuple appears in signatures and clippy is right that spelling it out is worse.
type ResolvedEntry = (RegisteredOp, Vec<(String, String)>);

/// A node paired with the implementation it resolved to.
#[derive(Clone, Debug)]
pub struct ResolvedNode {
    pub node: NodeId,
    pub spec_name: String,
    pub plugin: String,
    pub rejected: Vec<(String, String)>,
}

/// The result of compilation. This is what the runtime executes.
pub struct CompiledPlan {
    /// The graph *after* sharding propagation — the one that actually runs.
    pub plan: Plan,
    pub steps: Vec<CompiledStep>,
    pub digest: String,
    /// The mesh this plan was compiled for, copied from `plan.meta.mesh` so
    /// the topology fact has exactly one source (decision 4): the plan's
    /// fingerprint, never a traversable mesh (invariant I-6).
    pub mesh: MeshFingerprint,
    pub resolved: Vec<ResolvedNode>,
    pub inserted: Vec<InsertedCollective>,
    /// Lifetimes, storage placement and the projected peak. Computed here, not
    /// discovered at run time.
    pub memory: crate::memory::MemoryPlan,
}

impl std::fmt::Debug for CompiledPlan {
    /// Hand-written because the steps own raw ABI attribute storage; the useful
    /// debugging surface is the digest, the step count and the graph, all of
    /// which are printable.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledPlan")
            .field("name", &self.plan.meta.name)
            .field("digest", &self.digest)
            .field("slots", &self.plan.slots.len())
            .field("steps", &self.steps.len())
            .field("inserted_collectives", &self.inserted.len())
            .field("peak_bytes", &self.memory.peak_bytes)
            .finish()
    }
}

impl CompiledPlan {
    /// Human-readable rendering, intended for `rustrain plan explain`.
    pub fn explain(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "plan {}  digest {}  world {}\n",
            self.plan.meta.name,
            &self.digest[..12.min(self.digest.len())],
            self.mesh.world_size()
        ));
        out.push_str(&format!(
            "  slots {}  nodes {}  inserted collectives {}\n",
            self.plan.slots.len(),
            self.steps.len(),
            self.inserted.len()
        ));

        for (i, step) in self.steps.iter().enumerate() {
            let mark = if step.source().is_inserted() {
                "*"
            } else {
                " "
            };
            out.push_str(&format!(
                "{mark}[{i:>4}] {:>28}  <- {}  -> {}  @{}\n",
                step.label_with(&self.mesh),
                self.slot_list(step_inputs(step)),
                self.slot_list(step_outputs(step)),
                step.source().path,
            ));
        }
        out.push_str(&self.memory.explain());
        if !self.inserted.is_empty() {
            out.push_str("\ninserted communication:\n");
            for ins in &self.inserted {
                out.push_str(&format!(
                    "  {:<14} {}  ({})\n",
                    ins.op, ins.reason, ins.source
                ));
            }
        }
        out
    }

    fn slot_list(&self, ids: &[SlotId]) -> String {
        ids.iter()
            .map(|s| self.plan.slot(*s).name.clone())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Slots a caller must provide before executing.
    pub fn inputs(&self) -> Vec<SlotId> {
        self.plan.input_slots()
    }
}

fn step_inputs(s: &CompiledStep) -> &[SlotId] {
    match s {
        CompiledStep::Op { inputs, .. } => inputs,
        CompiledStep::Intrinsic { input, .. } => std::slice::from_ref(input),
    }
}

fn step_outputs(s: &CompiledStep) -> &[SlotId] {
    match s {
        CompiledStep::Op { outputs, .. } => outputs,
        CompiledStep::Intrinsic { output, .. } => std::slice::from_ref(output),
    }
}

/// Turns a plan into something the runtime can execute.
///
/// The compiler carries no copy of the topology: the mesh lives in the plan's
/// fingerprint (`plan.meta.mesh`, decision 4), which every pass resolves when
/// it needs one.
pub struct Compiler<'a> {
    registry: &'a Registry,
    recipe: &'a Recipe,
    env: TargetEnv,
    deterministic: bool,
    /// What the runtime can execute. A memory policy in this set may be planned
    /// for; anything outside it is refused rather than projected as a saving.
    caps: crate::memory::RuntimeCapabilities,
}

impl<'a> Compiler<'a> {
    pub fn new(registry: &'a Registry, recipe: &'a Recipe, env: TargetEnv) -> Self {
        Self {
            registry,
            recipe,
            env,
            deterministic: true,
            caps: crate::memory::RuntimeCapabilities::default(),
        }
    }

    /// Declares which memory strategies the runtime can actually execute.
    pub fn capabilities(mut self, caps: crate::memory::RuntimeCapabilities) -> Self {
        self.caps = caps;
        self
    }

    /// When false, operators that declare themselves non-deterministic are
    /// allowed (spec contract PL-1 item 5).
    pub fn deterministic(mut self, yes: bool) -> Self {
        self.deterministic = yes;
        self
    }

    pub fn compile(&self, plan: &Plan) -> Result<CompiledPlan, PlanError> {
        if plan.nodes.is_empty() {
            return Err(PlanError::EmptyPlan);
        }
        plan.check_structure()?;

        // The mesh intrinsic attrs are checked against, resolved once here.
        // (Propagation already revalidates the fingerprint for slot layouts.)
        let mesh_fingerprint = plan.meta.mesh.clone();
        let mesh = mesh_fingerprint
            .to_mesh()
            .map_err(|source| PlanError::Mesh { source })?;

        let propagation = shard::propagate(plan)?;
        let plan = propagation.plan;

        // Pass 1: resolve everything first. The memory pass has to ask each
        // implementation for its workspace before it can project a peak, and it
        // must do that before any step is emitted.
        let mut resolution: Vec<Option<ResolvedEntry>> = vec![None; plan.nodes.len()];
        for (i, node) in plan.nodes.iter().enumerate() {
            if intrinsic::is_intrinsic(&node.op.name) {
                continue;
            }
            resolution[i] = Some(self.resolve_node(NodeId(i), &plan, node)?);
        }

        let resolved_ops: Vec<Option<RegisteredOp>> = resolution
            .iter()
            .map(|r| r.as_ref().map(|(op, _)| op.clone()))
            .collect();

        // Project the peak and refuse a plan that cannot fit. Doing this here,
        // rather than letting the allocator discover it, is the whole point of
        // having the graph: the failure names the node and the strategy.
        let memory = memory::plan(&plan, &resolved_ops, &self.recipe.memory, self.caps)?;
        memory::enforce_budget(&memory, &plan)?;

        // Pass 2: validate against the implementations and flatten.
        let mut steps = Vec::with_capacity(plan.nodes.len());
        let mut resolved = Vec::new();

        for (i, node) in plan.nodes.iter().enumerate() {
            let id = NodeId(i);
            if intrinsic::is_intrinsic(&node.op.name) {
                steps.push(self.compile_intrinsic(id, node, &mesh)?);
                continue;
            }
            let (op, rejections) = resolution[i]
                .take()
                .expect("every non-intrinsic node was resolved in pass 1");
            self.check_arity(id, node, &op)?;

            let attrs = node.attrs.to_abi();
            let numerics = self.numerics_for(node);

            self.validate_shapes(id, &plan, node, &op, &attrs, node.outputs.len())?;

            resolved.push(ResolvedNode {
                node: id,
                spec_name: op.spec_name(),
                plugin: op.plugin_identity(),
                rejected: rejections,
            });

            steps.push(CompiledStep::Op {
                node: id,
                op,
                numerics,
                attrs,
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
                stream: stream_of(node.stream),
                phase: node.phase,
                source: node.source.clone(),
            });
        }

        let digest = compute_digest(&plan, &steps, self.recipe)?;

        Ok(CompiledPlan {
            plan,
            steps,
            digest,
            mesh: mesh_fingerprint,
            resolved,
            inserted: propagation.inserted,
            memory,
        })
    }

    /// Picks the implementation that will run for a node.
    ///
    /// A backward node names the *forward* operator; the recipe decides how its
    /// gradient is produced. So the backward case resolves the forward variant
    /// and then follows that variant's declared `backward_op` — resolving the
    /// name directly for `Phase::Backward` would hand back the forward kernel
    /// and silently run the wrong code.
    fn resolve_node(
        &self,
        id: NodeId,
        plan: &Plan,
        node: &crate::ir::PlanNode,
    ) -> Result<(RegisteredOp, Vec<(String, String)>), PlanError> {
        let dtypes: Vec<_> = node.inputs.iter().map(|s| plan.slot(*s).dtype).collect();

        // The variant that implements the *forward* direction of this node.
        let selection_phase = if node.phase == Phase::Backward {
            Phase::Forward
        } else {
            node.phase
        };

        let resolved = match &node.op.variant {
            // An explicit per-node variant is absolute: no recipe, no default
            // provider, no fallback (contracts R-1 and R-2).
            Some(v) => self.registry.resolve(&ResolveRequest {
                name: node.op.name.clone(),
                phase: selection_phase,
                prefer: Some(v.clone()),
                fallback: Vec::new(),
                dtypes: dtypes.clone(),
                env: self.env.clone(),
            }),
            None => self.recipe.resolve(
                self.registry,
                &node.op.name,
                selection_phase,
                &dtypes,
                &self.env,
            ),
        }
        .map_err(|source| PlanError::Resolve {
            node: id,
            op: node.op.name.clone(),
            source,
        })?;

        let rejections = resolved
            .rejected
            .iter()
            .map(|(v, why)| (v.clone(), why.to_string()))
            .collect();

        if node.phase != Phase::Backward {
            return Ok((resolved.op, rejections));
        }

        if matches!(
            self.recipe.backward_plan(&node.op.name),
            Some(rustrain_ops::recipe::BackwardPlan::Autodiff)
        ) {
            return Err(PlanError::NotValidatable {
                node: id,
                op: node.op.name.clone(),
                reason: "backward = \"autodiff\" needs backward nodes derived from the forward \
                         variant's declared expansion, which this compiler does not generate \
                         yet; configure an explicit backward implementation instead"
                    .to_string(),
            });
        }

        let back = self
            .registry
            .backward_of(&resolved.op, &dtypes, &self.env)
            .map_err(|source| PlanError::Resolve {
                node: id,
                op: node.op.name.clone(),
                source,
            })?;
        Ok((back, rejections))
    }

    fn check_arity(
        &self,
        id: NodeId,
        node: &crate::ir::PlanNode,
        op: &RegisteredOp,
    ) -> Result<(), PlanError> {
        // Arity is not in the ABI, so the only structural check available is
        // that the implementation has not declared more inputs than exist.
        if node.inputs.is_empty() && node.outputs.is_empty() {
            return Err(PlanError::NotValidatable {
                node: id,
                op: op.spec_name(),
                reason: "node has no inputs and no outputs".to_string(),
            });
        }
        Ok(())
    }

    /// Runs the implementation's own shape inference and compares the result to
    /// what the plan declared. A disagreement means the model builder and the
    /// kernel disagree about the math — better to find out here than at step 400.
    fn validate_shapes(
        &self,
        id: NodeId,
        plan: &Plan,
        node: &crate::ir::PlanNode,
        op: &RegisteredOp,
        attrs: &AbiAttrs,
        n_out: usize,
    ) -> Result<(), PlanError> {
        let Some(infer) = op.desc().infer else {
            return Err(PlanError::InferMissing {
                node: id,
                op: op.spec_name(),
            });
        };

        let in_tensors: Vec<RsTensor> = node
            .inputs
            .iter()
            .map(|s| tensor_for(plan.slot(*s)))
            .collect();
        let mut out_tensors: Vec<RsTensor> = node
            .outputs
            .iter()
            .map(|s| tensor_for(plan.slot(*s)))
            .collect();

        let in_ptrs: Vec<*const RsTensor> = in_tensors.iter().map(std::ptr::from_ref).collect();
        let mut out_ptrs: Vec<*mut RsTensor> =
            out_tensors.iter_mut().map(std::ptr::from_mut).collect();

        // SAFETY: pointers are into the vectors above, which outlive the call,
        // and the descriptors are plain data the implementation only reads
        // (outputs: writes shape/dtype, never `data`).
        let status = unsafe {
            infer(
                in_ptrs.as_ptr(),
                in_ptrs.len() as u32,
                out_ptrs.as_mut_ptr(),
                n_out as u32,
                attrs.as_ptr(),
            )
        };
        if status != 0 {
            return Err(PlanError::InferFailed {
                node: id,
                op: op.spec_name(),
                status,
                message: format!("{} returned {status}", op.spec_name()),
            });
        }

        for (j, slot_id) in node.outputs.iter().enumerate() {
            let inferred = out_tensors[j].dims().to_vec();
            let declared = plan.slot(*slot_id).shape.clone();
            if !inferred.is_empty() && inferred != declared {
                return Err(PlanError::InferredShapeMismatch {
                    node: id,
                    op: op.spec_name(),
                    index: j,
                    slot: *slot_id,
                    inferred,
                    declared,
                });
            }
        }
        Ok(())
    }

    /// Recipe precision, with the recipe's per-node override applied.
    fn numerics_for(&self, node: &crate::ir::PlanNode) -> RsNumerics {
        let mut n = self.recipe.precision.numerics(node.phase);
        if let Some(d) = node.precision.in_dtype {
            n.in_dtype = d;
        }
        if let Some(d) = node.precision.out_dtype {
            n.out_dtype = d;
        }
        if let Some(d) = node.precision.accum_dtype {
            n.accum_dtype = d;
        }
        if let Some(d) = node.precision.grad_dtype {
            n.grad_dtype = d;
        }
        n
    }

    fn compile_intrinsic(
        &self,
        id: NodeId,
        node: &crate::ir::PlanNode,
        mesh: &rustrain_parallel::Mesh,
    ) -> Result<CompiledStep, PlanError> {
        let op = node.op.name.clone();
        if !intrinsic::is_intrinsic(&op) {
            return Err(PlanError::UnknownIntrinsic { node: id, op });
        }
        // The group attribute is the mask's integer bits (decision 2): a mask
        // has no name without the mesh, and a node attribute is not the place
        // for one. The wrong type is reported as a bad attribute, never
        // silently coerced.
        let group_bits = match node.attrs.get(intrinsic::ATTR_GROUP) {
            Some(AttrValue::I64(bits)) => *bits,
            Some(other) => {
                return Err(PlanError::IntrinsicBadAttr {
                    op: op.clone(),
                    attr: intrinsic::ATTR_GROUP.to_string(),
                    value: format!("{other:?}"),
                });
            }
            None => {
                return Err(PlanError::IntrinsicMissingAttr {
                    op: op.clone(),
                    attr: intrinsic::ATTR_GROUP.to_string(),
                });
            }
        };
        let group = match u32::try_from(group_bits) {
            Ok(bits) => GroupMask::from_bits(bits),
            // A mask is 32 bits; a plan attribute outside `0..=u32::MAX`
            // (including a negative) is rejected here instead of being
            // silently truncated into a *different* group than declared.
            Err(_) => {
                return Err(PlanError::IntrinsicBadAttr {
                    op: op.clone(),
                    attr: intrinsic::ATTR_GROUP.to_string(),
                    value: group_bits.to_string(),
                });
            }
        };
        // Revalidate the mask against the mesh the plan was compiled for: the
        // attribute is plan data, so a plan from another topology (or a
        // hand-edited one) is reported here, never handed to the runtime.
        if group.validate(mesh).is_err() {
            return Err(PlanError::GroupUnavailable {
                node: id,
                op,
                group,
            });
        }

        let reduce = match node.attrs.str(intrinsic::ATTR_REDUCE) {
            Some("sum") => Some(ReduceOp::Sum),
            Some("max") => Some(ReduceOp::Max),
            Some("min") => Some(ReduceOp::Min),
            Some(other) => {
                return Err(PlanError::IntrinsicBadAttr {
                    op: op.clone(),
                    attr: intrinsic::ATTR_REDUCE.to_string(),
                    value: other.to_string(),
                });
            }
            None => None,
        };

        let input = *node
            .inputs
            .first()
            .ok_or_else(|| PlanError::NotValidatable {
                node: id,
                op: op.clone(),
                reason: "intrinsic needs exactly one input".to_string(),
            })?;
        let output = *node
            .outputs
            .first()
            .ok_or_else(|| PlanError::NotValidatable {
                node: id,
                op: op.clone(),
                reason: "intrinsic needs exactly one output".to_string(),
            })?;

        Ok(CompiledStep::Intrinsic {
            node: id,
            op,
            group,
            reduce,
            dim: node.attrs.i64(intrinsic::ATTR_DIM),
            input,
            output,
            stream: stream_of(node.stream),
            source: node.source.clone(),
        })
    }
}

fn stream_of(p: StreamPolicy) -> StreamId {
    match p {
        StreamPolicy::Default => MAIN_STREAM,
        StreamPolicy::Side => SIDE_STREAM,
    }
}

/// Builds a shape-only descriptor for inference: no data pointer, no backend.
fn tensor_for(slot: &Slot) -> RsTensor {
    let mut t = RsTensor::new(slot.dtype, &slot.shape);
    // Inference must not need a buffer; keeping it null also makes an
    // implementation that dereferences it fail loudly and immediately.
    t.data = std::ptr::null_mut();
    t
}

/// Everything that determines what will run, hashed.
///
/// Deliberately coarse: it includes the whole plan and the whole recipe, not a
/// hand-picked subset, so that adding a field cannot silently escape the digest.
fn compute_digest(
    plan: &Plan,
    steps: &[CompiledStep],
    recipe: &Recipe,
) -> Result<String, PlanError> {
    #[derive(Serialize)]
    struct NumericsKey {
        in_dtype: i32,
        out_dtype: i32,
        accum_dtype: i32,
        grad_dtype: i32,
        quant: i32,
        block: [u32; 2],
        scale_dtype: i32,
        scale_mode: i32,
        amax_history: u32,
    }

    #[derive(Serialize)]
    struct Decision {
        node: usize,
        op: String,
        implementation: String,
        numerics: Option<NumericsKey>,
        stream: StreamId,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
    }

    let decisions: Vec<Decision> = steps
        .iter()
        .enumerate()
        .map(|(i, s)| match s {
            CompiledStep::Op {
                op,
                numerics,
                inputs,
                outputs,
                stream,
                ..
            } => Decision {
                node: i,
                op: op.name().to_string(),
                implementation: format!("{}:{}", op.plugin_identity(), op.spec_name()),
                numerics: Some(NumericsKey {
                    in_dtype: numerics.in_dtype.0,
                    out_dtype: numerics.out_dtype.0,
                    accum_dtype: numerics.accum_dtype.0,
                    grad_dtype: numerics.grad_dtype.0,
                    quant: numerics.quant.0,
                    block: [numerics.block_m, numerics.block_n],
                    scale_dtype: numerics.scale_dtype.0,
                    scale_mode: numerics.scale_mode.0,
                    amax_history: numerics.amax_history,
                }),
                stream: *stream,
                inputs: inputs.iter().map(|s| s.0).collect(),
                outputs: outputs.iter().map(|s| s.0).collect(),
            },
            CompiledStep::Intrinsic {
                op,
                group,
                reduce,
                dim,
                input,
                output,
                stream,
                ..
            } => Decision {
                node: i,
                op: op.clone(),
                // The mask renders as its bits (`mask(0b1)`): deterministic,
                // and distinct per group without needing the mesh's names.
                implementation: format!("intrinsic:{group}:{reduce:?}:{dim:?}"),
                numerics: None,
                stream: *stream,
                inputs: vec![input.0],
                outputs: vec![output.0],
            },
        })
        .collect();

    #[derive(Serialize)]
    struct DigestInput<'a> {
        plan: &'a Plan,
        decisions: &'a [Decision],
    }

    // The recipe enters through the decisions it produced, not as text. Two
    // recipes that resolve to the same operators, variants and numerics describe
    // the same run and must digest identically — otherwise a cosmetic recipe
    // edit would make two identical runs look different, which defeats the point
    // of recording the digest at all.
    //
    // The plan carries the mesh fingerprint and the mask-based layouts, so the
    // preimage keeps both without any map iteration that could leak order into
    // the bytes (the plan's attrs are a `BTreeMap`; layouts are vecs).
    let _ = recipe;
    let input = DigestInput {
        plan,
        decisions: &decisions,
    };
    let json = serde_json::to_vec(&input).map_err(|e| PlanError::Digest(e.to_string()))?;
    Ok(blake3::hash(&json).to_hex().to_string())
}

/// The layout a slot ended up with, for diagnostics and tests.
pub fn slot_layout(plan: &Plan, id: SlotId) -> &ParallelLayout {
    &plan.slot(id).layout
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{OpRef, SlotKind};
    use crate::{Attrs, PlanBuilder};
    use rustrain_abi::ffi::RsDtype;
    use rustrain_parallel::{Mesh, ParallelConfig};

    fn default_mesh() -> Mesh {
        Mesh::from_config(&ParallelConfig::default())
    }

    /// Compiles `plan` with an empty registry and recipe: the plans below
    /// contain only intrinsics, which are compiled without a lookup.
    fn compile_plan(plan: &Plan) -> CompiledPlan {
        let registry = Registry::new();
        let recipe = Recipe::default();
        Compiler::new(&registry, &recipe, TargetEnv::default())
            .compile(plan)
            .unwrap()
    }

    /// One explicit `all_reduce` node whose group attribute carries the mask
    /// as integer bits.
    fn all_reduce_plan(mask: GroupMask) -> Plan {
        let mut b = PlanBuilder::new("attr", Phase::Forward, default_mesh().fingerprint());
        let x = b.slot("x", RsDtype::F32, vec![4], SlotKind::Activation);
        let y = b.slot("y", RsDtype::F32, vec![4], SlotKind::Activation);
        b.node(
            OpRef::new(intrinsic::ALL_REDUCE),
            vec![x],
            vec![y],
            // The write side, exactly as `shard::propagate` does it for
            // inserted collectives (decision 2).
            Attrs::new()
                .set(intrinsic::ATTR_GROUP, mask.bits() as i64)
                .set(intrinsic::ATTR_REDUCE, "sum"),
            "redu",
        );
        b.build().unwrap()
    }

    /// The group attribute round-trips as an integer: written as
    /// `GroupMask::bits() as i64`, read back as `GroupMask::from_bits` and
    /// equal to the original mask (decision 2).
    #[test]
    fn group_attr_round_trips_as_integer() {
        let mask = GroupMask::from_bits(0b101); // tp | ep: both axes exist
        let plan = all_reduce_plan(mask);

        let compiled = compile_plan(&plan);
        let step = compiled
            .steps
            .iter()
            .find(|s| matches!(s, CompiledStep::Intrinsic { .. }))
            .expect("the plan has one intrinsic step");
        match step {
            CompiledStep::Intrinsic { group, reduce, .. } => {
                assert_eq!(*group, mask);
                assert_eq!(*reduce, Some(ReduceOp::Sum));
            }
            CompiledStep::Op { .. } => unreachable!("intrinsic-only plan"),
        }
    }

    /// An out-of-range mask in the attribute is a *reported* error naming
    /// node, op and mask — never a panic and never a silent truncation
    /// (decision 2 / 3).
    #[test]
    fn out_of_range_group_attr_is_reported() {
        let stray = GroupMask::from_bits(1 << 10); // bit 10: the mesh has 5 axes
        let plan = all_reduce_plan(stray);

        let registry = Registry::new();
        let recipe = Recipe::default();
        let err = Compiler::new(&registry, &recipe, TargetEnv::default())
            .compile(&plan)
            .unwrap_err();
        match err {
            PlanError::GroupUnavailable { node, op, group } => {
                assert_eq!(node, NodeId(0));
                assert_eq!(op, intrinsic::ALL_REDUCE);
                assert_eq!(group, stray);
            }
            other => panic!("expected GroupUnavailable, got {other:?}"),
        }
    }

    /// **Case F3 (reviewer finding, MEDIUM).** A group mask has 32 bits, so an
    /// `ATTR_GROUP` value outside `0..=u32::MAX` (including negatives) must be
    /// rejected naming the raw value — `2^32` used to be cast to
    /// `GroupMask::NONE` and `2^32 + 1` to `tp`, silently compiling a
    /// *different group than declared*.
    #[test]
    fn group_attr_outside_u32_is_reported_not_truncated() {
        for raw in [1i64 << 32, (1i64 << 32) + 1, -1i64, i64::MAX] {
            let mut b = PlanBuilder::new("big", Phase::Forward, default_mesh().fingerprint());
            let x = b.slot("x", RsDtype::F32, vec![4], SlotKind::Activation);
            let y = b.slot("y", RsDtype::F32, vec![4], SlotKind::Activation);
            b.node(
                OpRef::new(intrinsic::ALL_REDUCE),
                vec![x],
                vec![y],
                Attrs::new()
                    .set(intrinsic::ATTR_GROUP, raw)
                    .set(intrinsic::ATTR_REDUCE, "sum"),
                "redu",
            );
            let plan = b.build().unwrap();

            let registry = Registry::new();
            let recipe = Recipe::default();
            let err = Compiler::new(&registry, &recipe, TargetEnv::default())
                .compile(&plan)
                .unwrap_err();
            match err {
                PlanError::IntrinsicBadAttr { op, attr, value } => {
                    assert_eq!(op, intrinsic::ALL_REDUCE);
                    assert_eq!(attr, intrinsic::ATTR_GROUP);
                    assert_eq!(
                        value,
                        raw.to_string(),
                        "the error must name the raw value, not a truncated mask"
                    );
                }
                other => panic!("expected IntrinsicBadAttr for {raw}, got {other:?}"),
            }
        }
    }

    /// The digest is a deterministic function of the plan: compiling the same
    /// plan twice yields the same bytes. The preimage keeps the mesh
    /// fingerprint and the mask-based layouts (decision 8), and nothing in
    /// the encoding iterates a hash map.
    #[test]
    fn digest_is_deterministic() {
        let plan = all_reduce_plan(GroupMask::from_bits(0b1));
        let a = compile_plan(&plan);
        let b = compile_plan(&plan);
        assert_eq!(a.digest, b.digest);
    }
}
