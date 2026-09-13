//! Memory planning: lifetimes, reuse, peak projection, budget gate.
//!
//! Having a plan IR is what makes this tractable. A slot's live range follows
//! from its producer and its last consumer, both known before anything runs, so
//! **peak device memory is computable at compile time** — a plan that cannot fit
//! is refused before a single GPU is allocated, rather than discovered as an OOM
//! four hundred steps in.
//!
//! What this pass does *not* do is perform offloading or recomputation. It plans
//! for them and refuses a plan whose budget depends on a strategy the runtime
//! cannot execute ([`RuntimeCapabilities`]). Projecting savings from a strategy
//! nobody implements would be the same silent lie as a fallback that never runs.

use rustrain_abi::ffi::{RsMemReq, RsTensor};
use rustrain_ops::{ActivationPolicy, MemoryPool, MemoryRecipe, RegisteredOp};

use crate::PlanError;
use crate::attrs::AbiAttrs;
use crate::ir::{NodeId, Plan, SlotId, SlotKind, intrinsic};

/// What the runtime can actually do, as opposed to what the recipe may ask for.
///
/// Both default to `false` on purpose: until an offload queue and a backward
/// recompute pass exist, a plan that needs one must fail loudly rather than
/// assume memory it will not get back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeCapabilities {
    /// Can move an activation to host memory and bring it back.
    pub offload: bool,
    /// Can regenerate an activation during the backward pass.
    pub recompute: bool,
}

impl RuntimeCapabilities {
    pub fn supports(&self, policy: ActivationPolicy) -> bool {
        match policy {
            ActivationPolicy::Keep => true,
            ActivationPolicy::Offload => self.offload,
            ActivationPolicy::Recompute => self.recompute,
        }
    }

    /// Everything implemented so far. Named so call sites read as intent.
    pub fn none() -> Self {
        Self::default()
    }
}

/// A slot's live range, in step order.
///
/// `born` is the index of the step that writes it (0 for a plan input); `dies`
/// is the index of the last step that reads it, or the step count when nothing
/// ever does (a plan output).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lifetime {
    pub slot: SlotId,
    pub born: usize,
    pub dies: usize,
}

impl Lifetime {
    pub fn live_at(&self, step: usize) -> bool {
        step >= self.born && step <= self.dies
    }

    pub fn overlaps(&self, other: &Lifetime) -> bool {
        self.born <= other.dies && other.born <= self.dies
    }
}

/// Where a slot's bytes come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Lives for the whole run and is never reused (weights, gradients, state).
    Persistent { offset: u64 },
    /// Reused from the transient pool. Two slots with the same offset share
    /// storage, which is only sound because their lifetimes do not overlap.
    Pool { offset: u64 },
    /// Shares another slot's storage outright — the output of a spliced
    /// collective, which reduces its input in place.
    Aliased(SlotId),
    /// Counted as not resident: offloaded or regenerated.
    NonResident,
}

/// One slot's allocation, with the policy that produced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotAllocation {
    pub slot: SlotId,
    /// Bytes the slot needs.
    pub bytes: u64,
    /// Bytes this slot contributes to the projected peak. Zero for a slot the
    /// plan assumes is offloaded or regenerated.
    pub counted_bytes: u64,
    pub placement: Placement,
    /// The policy of the node that produces it; `Keep` for plan inputs, weights
    /// and state.
    pub policy: ActivationPolicy,
}

/// A policy choice the planner made, kept so it can be reported.
///
/// Contract MEM-3: relaxation must never be silent, so every step of it lands
/// here and in the plan explanation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyDecision {
    pub node: NodeId,
    pub op: String,
    pub from: ActivationPolicy,
    pub to: ActivationPolicy,
    pub reason: String,
}

/// The planner's result.
#[derive(Clone, Debug)]
pub struct MemoryPlan {
    pub lifetimes: Vec<Lifetime>,
    pub allocations: Vec<SlotAllocation>,
    pub decisions: Vec<PolicyDecision>,
    /// Weights, gradients and optimizer state: live throughout.
    pub persistent_bytes: u64,
    /// Size of the transient pool, i.e. the reuse-aware peak of the activations.
    pub transient_pool_bytes: u64,
    /// Largest single-step workspace any operator asked for.
    pub max_workspace_bytes: u64,
    /// `persistent_bytes + transient_pool_bytes + max_workspace_bytes`.
    pub peak_bytes: u64,
    pub budget_bytes: Option<u64>,
    /// Byte alignment every offset in the persistent region and the activation
    /// pool was rounded up to (CPU 1, CUDA 256; overridable in the recipe). The
    /// region sizes already include the padding, so the peak changes with it.
    pub align_bytes: u64,
    /// Policies a node wanted that the runtime cannot execute.
    pub unsupported: Vec<(NodeId, String, ActivationPolicy)>,
    /// `(slot, earlier_slot_it_reuses, bytes)` for every pool slot that took over
    /// storage from an earlier one. Empty when pooling is off.
    pub reuse: Vec<(SlotId, SlotId, u64)>,
}

impl MemoryPlan {
    pub fn allocation(&self, slot: SlotId) -> Option<&SlotAllocation> {
        self.allocations.iter().find(|a| a.slot == slot)
    }

    /// One-line summary for logs.
    pub fn summary(&self) -> String {
        let budget = match self.budget_bytes {
            Some(b) => format!("{b}"),
            None => "unset".to_string(),
        };
        format!(
            "peak {} B (persistent {} + activations {} + workspace {}) / budget {} / align {} B",
            self.peak_bytes,
            self.persistent_bytes,
            self.transient_pool_bytes,
            self.max_workspace_bytes,
            budget,
            self.align_bytes,
        )
    }

    /// Multi-line rendering for `rustrain plan explain`.
    pub fn explain(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("memory: {}\n", self.summary()));

        if !self.decisions.is_empty() {
            out.push_str("  policy relaxations:\n");
            for d in &self.decisions {
                out.push_str(&format!(
                    "    [{}] {} -> {}: {}\n",
                    d.node.0,
                    name_of(d.to),
                    name_of(d.from),
                    d.reason
                ));
            }
        }
        if !self.unsupported.is_empty() {
            out.push_str("  policies the runtime cannot execute:\n");
            for (node, op, policy) in &self.unsupported {
                out.push_str(&format!("    [{}] {op} wants {policy:?}\n", node.0));
            }
        }

        if !self.reuse.is_empty() {
            out.push_str("  storage reuse:\n");
            for (slot, with, bytes) in &self.reuse {
                out.push_str(&format!(
                    "    slot {} takes over {} B from slot {}\n",
                    slot.0, bytes, with.0
                ));
            }
        }
        out
    }
}

fn name_of(p: ActivationPolicy) -> &'static str {
    match p {
        ActivationPolicy::Keep => "keep",
        ActivationPolicy::Offload => "offload",
        ActivationPolicy::Recompute => "recompute",
    }
}

/// A resolved operator per node; `None` for intrinsics, which have no
/// implementation to ask.
pub type ResolvedOps = [Option<RegisteredOp>];

/// Runs the memory pass.
///
/// `align_bytes` is the byte alignment every offset into the persistent region
/// and the activation pool is rounded up to. It is resolved by the caller (the
/// compiler): the recipe's override, or the target device's default (CPU 1,
/// CUDA 256). The alignment is passed in rather than derived here, so this
/// pass stays a pure function of its inputs.
///
/// Returns the plan and the selected policy per node. The caller decides what to
/// do about the budget; [`enforce_budget`] is the policy for that.
pub fn plan(
    plan: &Plan,
    ops: &ResolvedOps,
    recipe: &MemoryRecipe,
    caps: RuntimeCapabilities,
    align_bytes: u64,
) -> Result<MemoryPlan, PlanError> {
    if ops.len() != plan.nodes.len() {
        return Err(PlanError::Digest(format!(
            "memory pass got {} resolved ops for {} nodes",
            ops.len(),
            plan.nodes.len()
        )));
    }

    let sizes = slot_bytes(plan);
    let lifetimes = compute_lifetimes(plan, &sizes);
    let aliases = compute_aliases(plan, &sizes);

    // The requested policy per node. Intrinsics and plan inputs keep everything.
    let mut policy: Vec<ActivationPolicy> = plan
        .nodes
        .iter()
        .map(|n| {
            if intrinsic::is_intrinsic(&n.op.name) {
                ActivationPolicy::Keep
            } else {
                recipe.policy_for(&n.op.name)
            }
        })
        .collect();

    let mut decisions = Vec::new();
    let mut unsupported: Vec<(NodeId, String, ActivationPolicy)> = Vec::new();

    // A policy the runtime cannot execute is refused here rather than projected
    // as a saving that will never materialise.
    for (i, p) in policy.iter().enumerate() {
        if *p != ActivationPolicy::Keep && !caps.supports(*p) {
            unsupported.push((NodeId(i), plan.nodes[i].op.name.clone(), *p));
        }
    }

    let workspace = query_workspace(plan, ops, &sizes)?;

    // The hard budget is what a plan is allowed to spend; the target is where
    // relaxation aims, leaving room for allocator fragmentation. Conflating the
    // two made a tight budget impossible to satisfy by construction.
    let hard_budget = recipe.budget_bytes;
    let target = recipe.target_bytes();
    let mut mem = build(
        plan,
        &lifetimes,
        &aliases,
        &sizes,
        &policy,
        &workspace,
        recipe,
        hard_budget,
        align_bytes,
    );

    // Relaxation, in a deterministic order: steps in index order, and for each,
    // Keep -> Offload -> Recompute. Stop as soon as the target is met.
    if let Some(target) = target
        && mem.peak_bytes > target
    {
        let ladder = [ActivationPolicy::Offload, ActivationPolicy::Recompute];
        'outer: for step in 0..plan.nodes.len() {
            if intrinsic::is_intrinsic(&plan.nodes[step].op.name)
                || policy[step] != ActivationPolicy::Keep
            {
                continue;
            }
            for next in ladder {
                if next == ActivationPolicy::Recompute && !caps.recompute {
                    continue;
                }
                if next == ActivationPolicy::Offload && !caps.offload {
                    continue;
                }
                let from = policy[step];
                policy[step] = next;
                decisions.push(PolicyDecision {
                    node: NodeId(step),
                    op: plan.nodes[step].op.name.clone(),
                    from,
                    to: next,
                    reason: format!(
                        "peak {} B is over the {} B target; {} on this node is the next rung of \
                         the ladder (keep -> offload -> recompute) for the lowest step index",
                        mem.peak_bytes,
                        target,
                        name_of(next)
                    ),
                });
                mem = build(
                    plan,
                    &lifetimes,
                    &aliases,
                    &sizes,
                    &policy,
                    &workspace,
                    recipe,
                    hard_budget,
                    align_bytes,
                );
                if mem.peak_bytes <= target {
                    break 'outer;
                }
            }
        }
    }

    Ok(MemoryPlan {
        decisions,
        unsupported,
        budget_bytes: hard_budget,
        ..mem
    })
}

/// Reports that a plan's projected peak exceeds its budget.
///
/// Called separately from [`plan`] so the caller can inspect the projection
/// before deciding, and so tests can exercise the arithmetic without the gate.
/// **The verdict is advisory** (D12, `docs/architecture.md` §8): the compiler
/// carries the error as a warning on the compiled plan instead of failing, and
/// this function remains the single place that names the hottest step.
pub fn enforce_budget(mem: &MemoryPlan, plan: &Plan) -> Result<(), PlanError> {
    let Some(budget) = mem.budget_bytes else {
        return Ok(());
    };
    if mem.peak_bytes <= budget {
        return Ok(());
    }

    // Name the step where the peak is reached: that is the actionable part.
    let mut hottest = 0usize;
    let mut hottest_bytes = 0u64;
    for step in 0..plan.nodes.len() {
        let live: u64 = mem
            .allocations
            .iter()
            .filter(|a| {
                mem.lifetimes
                    .iter()
                    .find(|l| l.slot == a.slot)
                    .is_some_and(|l| l.live_at(step))
            })
            .map(|a| a.counted_bytes)
            .sum();
        let total = mem.persistent_bytes + live;
        if total > hottest_bytes {
            hottest_bytes = total;
            hottest = step;
        }
    }

    let mut suggestions: Vec<String> = mem
        .unsupported
        .iter()
        .map(|(node, op, policy)| {
            format!(
                "node {} ({op}) wants {} but the runtime does not implement it",
                node.0,
                name_of(*policy)
            )
        })
        .collect();
    if suggestions.is_empty() {
        suggestions.push(
            "raise budget_bytes, set [kernel.memory.ops.<op>].activation_policy, or reduce \
             recompute_groups so more layers can be offloaded"
                .to_string(),
        );
    }

    Err(PlanError::MemoryBudgetExceeded {
        peak: mem.peak_bytes,
        budget,
        hottest_node: hottest,
        hottest_op: plan.nodes[hottest].op.name.clone(),
        hottest_bytes,
        suggestions,
    })
}

/// Rounds `x` up to a multiple of `align`. `align` is validated non-zero by
/// the recipe, and `1` is the identity — which is what keeps the CPU layout
/// byte-identical to the pre-alignment planner.
fn round_up(x: u64, align: u64) -> u64 {
    x.div_ceil(align) * align
}

#[allow(clippy::too_many_arguments)]
fn build(
    plan: &Plan,
    lifetimes: &[Lifetime],
    aliases: &[Option<SlotId>],
    sizes: &[u64],
    policy: &[ActivationPolicy],
    workspace: &[u64],
    recipe: &MemoryRecipe,
    budget: Option<u64>,
    align: u64,
) -> MemoryPlan {
    let producer_policy: Vec<ActivationPolicy> = {
        let mut v = vec![ActivationPolicy::Keep; plan.slots.len()];
        for (i, n) in plan.nodes.iter().enumerate() {
            for o in &n.outputs {
                v[o.0] = policy[i];
            }
        }
        v
    };

    let mut allocations = Vec::with_capacity(plan.slots.len());

    // Persistent first: they are never reused and their sum is a floor.
    // Every offset is rounded up to `align`: a `float` slot at offset 12 is
    // harmless on the scalar CPU provider and a misaligned-device-pointer bug
    // for an ATen kernel using 16-byte vector loads.
    let mut persistent_offset = 0u64;
    for (i, slot) in plan.slots.iter().enumerate() {
        if is_persistent(slot.kind) {
            let bytes = sizes[i];
            allocations.push(SlotAllocation {
                slot: SlotId(i),
                bytes,
                counted_bytes: bytes,
                placement: Placement::Persistent {
                    offset: persistent_offset,
                },
                policy: ActivationPolicy::Keep,
            });
            persistent_offset = round_up(persistent_offset + bytes, align);
        }
    }
    // The padding between slots is part of the region the executor allocates.
    let persistent_bytes = persistent_offset;

    // Aliases share storage outright.
    for (i, alias) in aliases.iter().enumerate() {
        if let Some(root) = alias {
            let root_bytes = allocations
                .iter()
                .find(|a| a.slot == *root)
                .map(|a| a.bytes)
                .unwrap_or(sizes[i]);
            allocations.push(SlotAllocation {
                slot: SlotId(i),
                bytes: sizes[i],
                counted_bytes: 0,
                placement: Placement::Aliased(*root),
                policy: ActivationPolicy::Keep,
            });
            let _ = root_bytes;
        }
    }

    // Transients: interval allocation into one pool.
    let mut transient: Vec<usize> = (0..plan.slots.len())
        .filter(|i| !is_persistent(plan.slots[*i].kind) && aliases[*i].is_none())
        .collect();
    transient.sort_by_key(|i| (lifetimes[*i].born, *i));

    let mut pool_bytes = 0u64;
    // (offset, bytes, dies) of blocks currently handed out.
    let mut handed: Vec<(u64, u64, usize)> = Vec::new();

    for i in &transient {
        let lt = lifetimes[*i];
        let bytes = sizes[*i];
        let pol = producer_policy[*i];
        let counted = if pol == ActivationPolicy::Keep {
            bytes
        } else {
            0
        };

        // Release blocks whose slot has died.
        handed.retain(|(_, _, dies)| *dies >= lt.born);

        let mut chosen: Option<u64> = None;
        if recipe.pool == MemoryPool::Slab && counted > 0 {
            // First fit among blocks that are free and large enough. Simplest
            // correct policy; the point is that reuse happens at all, and that
            // it is a deterministic function of the plan.
            let mut free: Vec<(u64, u64)> = vec![(0, pool_bytes)];
            for (off, len, _) in &handed {
                let mut next = Vec::with_capacity(free.len() + 1);
                for (foff, flen) in free {
                    if *off >= foff + flen || off + len <= foff {
                        next.push((foff, flen));
                        continue;
                    }
                    if *off > foff {
                        next.push((foff, off - foff));
                    }
                    let end = off + len;
                    if end < foff + flen {
                        next.push((end, foff + flen - end));
                    }
                }
                free = next;
            }
            free.sort_by_key(|(off, _)| *off);
            // A free block's start may sit in the padding after an earlier
            // placement ended unaligned; only its aligned start is usable, and
            // the fit must hold for the block that remains.
            for (off, len) in &free {
                let start = round_up(*off, align);
                let pad = start - *off;
                if *len >= pad && *len - pad >= bytes {
                    chosen = Some(start);
                    break;
                }
            }
        }

        let (offset, placement) = match chosen {
            Some(off) => (off, Placement::Pool { offset: off }),
            None if counted == 0 => (0, Placement::NonResident),
            None => {
                let off = pool_bytes;
                pool_bytes = round_up(off + bytes, align);
                (off, Placement::Pool { offset: off })
            }
        };

        if counted > 0 {
            handed.push((offset, bytes, lt.dies));
        }

        allocations.push(SlotAllocation {
            slot: SlotId(*i),
            bytes,
            counted_bytes: counted,
            placement,
            policy: pol,
        });
    }

    // Two pool slots sharing an offset share storage; record the pairing so the
    // decision is reportable rather than implicit in the arithmetic.
    let mut by_offset: std::collections::BTreeMap<u64, Vec<SlotId>> =
        std::collections::BTreeMap::new();
    for a in &allocations {
        if let Placement::Pool { offset } = a.placement
            && a.counted_bytes > 0
        {
            by_offset.entry(offset).or_default().push(a.slot);
        }
    }
    let mut reuse = Vec::new();
    for slots in by_offset.values() {
        for w in slots.windows(2) {
            reuse.push((w[1], w[0], sizes[w[1].0]));
        }
    }

    let max_workspace_bytes = workspace.iter().copied().max().unwrap_or(0);
    let peak_bytes = persistent_bytes + pool_bytes + max_workspace_bytes;

    MemoryPlan {
        lifetimes: lifetimes.to_vec(),
        allocations,
        decisions: Vec::new(),
        persistent_bytes,
        transient_pool_bytes: pool_bytes,
        max_workspace_bytes,
        peak_bytes,
        budget_bytes: budget,
        align_bytes: align,
        unsupported: Vec::new(),
        reuse,
    }
}

fn is_persistent(kind: SlotKind) -> bool {
    matches!(
        kind,
        SlotKind::Weight | SlotKind::Gradient | SlotKind::State
    )
}

/// Byte size of every slot's element buffer.
fn slot_bytes(plan: &Plan) -> Vec<u64> {
    plan.slots
        .iter()
        .map(|s| s.element_bytes().unwrap_or(0))
        .collect()
}

fn compute_lifetimes(plan: &Plan, sizes: &[u64]) -> Vec<Lifetime> {
    let n_steps = plan.nodes.len();
    let mut born = vec![0usize; plan.slots.len()];
    let mut dies: Vec<Option<usize>> = vec![None; plan.slots.len()];
    let mut produced = vec![false; plan.slots.len()];

    for (i, node) in plan.nodes.iter().enumerate() {
        for o in &node.outputs {
            born[o.0] = i;
            produced[o.0] = true;
        }
        for inp in &node.inputs {
            // `None` means "never consumed so far". Folding the step count into
            // this instead of keeping it apart made every slot look as though it
            // died at the end, which silently disabled all reuse.
            dies[inp.0] = Some(dies[inp.0].map_or(i, |d: usize| d.max(i)));
        }
    }

    // A `view` output aliases its input's storage — the operator hands back
    // the input's own descriptor and the executor adopts it, no copy — so the
    // input must stay live exactly as long as the output. Without this
    // extension a view whose output is kept to the end of the plan (the
    // runner's hidden-state keeper) would "keep alive" a buffer the pool has
    // already handed to a later activation, and reading the input afterwards
    // reads that later activation instead.
    for node in &plan.nodes {
        if node.op.name == "view" && node.inputs.len() == 1 && node.outputs.len() == 1 {
            let input = node.inputs[0];
            let output = node.outputs[0];
            let out_dies = dies[output.0].unwrap_or(n_steps);
            dies[input.0] = Some(dies[input.0].map_or(out_dies, |d: usize| d.max(out_dies)));
        }
    }

    // An in-place collective shares its input's storage: `compute_aliases` marks the output
    // `Aliased(input)` whenever the result fits, and the executor then writes through the input's
    // pointer. The input must therefore stay reserved exactly as long as the output does —
    // otherwise the pool hands those bytes to a later activation and *both* slots read that later
    // activation's values. The first D6 device run died here in a way a shape check cannot see:
    // the embedding's `all_reduce` (degree-1, tp) aliased the embedding output, whose own last
    // reader was that collective, so the pool gave its bytes to the next normalisation and every
    // hidden-state summary after `embed.y` was computed from the wrong tensor.
    for node in &plan.nodes {
        if !intrinsic::is_intrinsic(&node.op.name) {
            continue;
        }
        let (Some(input), Some(output)) = (node.inputs.first(), node.outputs.first()) else {
            continue;
        };
        // Only when the output fits the input is the output aliased; otherwise it gets a buffer
        // of its own and the input may die on schedule.
        if sizes[output.0] > sizes[input.0] {
            continue;
        }
        let out_dies = dies[output.0].unwrap_or(n_steps);
        dies[input.0] = Some(dies[input.0].map_or(out_dies, |d: usize| d.max(out_dies)));
    }

    (0..plan.slots.len())
        .map(|i| Lifetime {
            slot: SlotId(i),
            // A plan input is available from the very first step.
            born: if produced[i] { born[i] } else { 0 },
            // Nothing reads it: it is a plan output and must survive the run.
            dies: dies[i].unwrap_or(n_steps),
        })
        .collect()
}

/// Spliced collectives reduce in place, so their output shares the input's
/// storage. Resolved here so peak accounting counts it once.
/// A spliced collective's output shares its input's storage — but only while
/// the output fits: an `all_gather` (or an uneven `all_to_all`) produces more
/// bytes than it consumes, so aliasing it onto the input would hand the runtime
/// a buffer the result cannot fit in. Such outputs get a real pool placement
/// instead; the runtime then hands the backend two descriptors and it copies
/// input → output. A `reduce_scatter` (smaller output) still aliases.
fn compute_aliases(plan: &Plan, sizes: &[u64]) -> Vec<Option<SlotId>> {
    let mut aliases: Vec<Option<SlotId>> = vec![None; plan.slots.len()];
    for node in &plan.nodes {
        if !intrinsic::is_intrinsic(&node.op.name) {
            continue;
        }
        let (Some(input), Some(output)) = (node.inputs.first(), node.outputs.first()) else {
            continue;
        };
        let root = aliases[input.0].unwrap_or(*input);
        if sizes[output.0] <= sizes[root.0] {
            aliases[output.0] = Some(root);
        }
    }
    aliases
}

/// Asks every operator how much scratch it needs.
///
/// An implementation that does not implement `memory` reports zero, which is the
/// ABI's convention for "no workspace" — a provider that needs scratch and does
/// not say so is a provider bug the conformance gate has to catch, not something
/// the planner can infer.
fn query_workspace(plan: &Plan, ops: &ResolvedOps, sizes: &[u64]) -> Result<Vec<u64>, PlanError> {
    let mut out = vec![0u64; plan.nodes.len()];

    for (i, node) in plan.nodes.iter().enumerate() {
        let Some(op) = ops[i].as_ref() else {
            continue;
        };
        let Some(query) = op.desc().memory else {
            continue;
        };

        let attrs: AbiAttrs = node.attrs.to_abi();
        let io: Vec<RsTensor> = node
            .inputs
            .iter()
            .chain(node.outputs.iter())
            .map(|s| {
                let slot = plan.slot(*s);
                RsTensor::new(slot.dtype, &slot.shape)
            })
            .collect();
        let io_ptrs: Vec<*const RsTensor> = io.iter().map(std::ptr::from_ref).collect();
        let _ = sizes;

        let mut req = RsMemReq::default();
        // SAFETY: `io_ptrs` points at descriptors that outlive the call, and the
        // attribute array is owned by `attrs`. The implementation only reads
        // inputs and writes `req`.
        let status = unsafe {
            query(
                io_ptrs.as_ptr(),
                io_ptrs.len() as u32,
                attrs.as_ptr(),
                &mut req,
            )
        };
        if status == 0 {
            out[i] = req.workspace_bytes;
        }
        // A non-zero status means "cannot tell". The ABI has no per-op error
        // channel here, so it is treated as zero workspace and shows up as a
        // conformance failure rather than a planning failure.
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    // The tests bind a local `plan`, which would shadow the pass itself.
    use super::plan as run_memory_pass;
    use crate::ir::{OpRef, Phase, SlotId};
    use crate::{Attrs, PlanBuilder};
    use rustrain_abi::ffi::RsDtype;
    use rustrain_parallel::{Mesh, ParallelConfig};

    fn plan_with(n_layers: usize) -> Plan {
        let mut b = PlanBuilder::new(
            "mem",
            Phase::Forward,
            Mesh::from_config(&ParallelConfig::default()).fingerprint(),
        );
        let w = b.slot("w", RsDtype::F32, vec![64, 64], SlotKind::Weight);
        let mut prev = b.slot("x", RsDtype::F32, vec![8, 64], SlotKind::Input);
        for l in 0..n_layers {
            let out = b.slot(
                format!("a{l}"),
                RsDtype::F32,
                vec![8, 64],
                SlotKind::Activation,
            );
            b.node(
                OpRef::new("elementwise_unary"),
                vec![prev, w],
                vec![out],
                Attrs::new().set("kind", "silu"),
                format!("layer{l}"),
            );
            prev = out;
        }
        let loss = b.slot("loss", RsDtype::F32, vec![8, 64], SlotKind::Output);
        b.node(
            OpRef::new("elementwise_unary"),
            vec![prev],
            vec![loss],
            Attrs::new().set("kind", "relu"),
            "loss",
        );
        b.build().unwrap()
    }

    fn no_ops(plan: &Plan) -> Vec<Option<RegisteredOp>> {
        vec![None; plan.nodes.len()]
    }

    /// A plan with mixed-size persistent and transient slots, so the offsets
    /// are a non-trivial function of the packing order. The exact numbers the
    /// alignment tests pin were produced by the pre-alignment planner and are
    /// reproduced here from the slot sizes alone.
    fn mixed_plan() -> Plan {
        let mut b = PlanBuilder::new(
            "mixed",
            Phase::Forward,
            Mesh::from_config(&ParallelConfig::default()).fingerprint(),
        );
        let w0 = b.slot("w0", RsDtype::F32, vec![64, 64], SlotKind::Weight); // 16384 B
        let w1 = b.slot("w1", RsDtype::F32, vec![3, 5], SlotKind::Weight); // 60 B
        let x = b.slot("x", RsDtype::F32, vec![3, 5], SlotKind::Input); // 60 B
        let a = b.slot("a", RsDtype::F32, vec![8, 64], SlotKind::Activation); // 2048 B
        b.node(
            OpRef::new("elementwise_unary"),
            vec![x, w0],
            vec![a],
            Attrs::new(),
            "n0",
        );
        let y = b.slot("y", RsDtype::F32, vec![2, 7], SlotKind::Activation); // 56 B
        b.node(
            OpRef::new("elementwise_unary"),
            vec![a, w1],
            vec![y],
            Attrs::new(),
            "n1",
        );
        let out = b.slot("out", RsDtype::F32, vec![2, 7], SlotKind::Output); // 56 B
        b.node(
            OpRef::new("elementwise_unary"),
            vec![y],
            vec![out],
            Attrs::new(),
            "n2",
        );
        b.build().unwrap()
    }

    fn offset_of(mem: &MemoryPlan, slot: SlotId) -> u64 {
        match mem.allocation(slot).unwrap().placement {
            Placement::Persistent { offset } | Placement::Pool { offset } => offset,
            other => panic!("slot {slot:?} is {other:?}, expected a real offset"),
        }
    }

    /// An in-place collective's input must stay reserved as long as its output.
    ///
    /// The executor writes the collective's result through the *input's* pointer when the planner
    /// aliased them, so the input's live range has to cover the output's: otherwise the pool hands
    /// those bytes to an activation born after the collective, and the input, the output and that
    /// activation all describe the same memory. On the first D6 device run this is what made every
    /// hidden-state summary after `embed.y` report a tensor that had been overwritten — right
    /// shape, right name, wrong numbers, and invisible to every shape check.
    #[test]
    fn an_in_place_collectives_input_outlives_its_output() {
        let mut b = PlanBuilder::new(
            "collective-alias",
            Phase::Forward,
            Mesh::from_config(&ParallelConfig::default()).fingerprint(),
        );
        let x = b.slot("x", RsDtype::F32, vec![8], SlotKind::Input);
        // `a` is read only by the collective; `done` is read twice afterwards, so without the
        // extension `a` dies at the collective while its storage is still the collective's output.
        let a = b.slot("a", RsDtype::F32, vec![2, 8], SlotKind::Activation);
        let done = b.slot("done", RsDtype::F32, vec![2, 8], SlotKind::Activation);
        let c = b.slot("c", RsDtype::F32, vec![2, 8], SlotKind::Activation);
        let d = b.slot("d", RsDtype::F32, vec![2, 8], SlotKind::Activation);
        b.node(
            OpRef::new("elementwise_unary"),
            vec![x],
            vec![a],
            Attrs::new(),
            "n0",
        );
        b.node(
            OpRef::new(crate::ir::intrinsic::ALL_REDUCE),
            vec![a],
            vec![done],
            Attrs::new(),
            "n1",
        );
        b.node(
            OpRef::new("elementwise_unary"),
            vec![done],
            vec![c],
            Attrs::new(),
            "n2",
        );
        b.node(
            OpRef::new("elementwise_unary"),
            vec![done],
            vec![d],
            Attrs::new(),
            "n3",
        );

        let plan = b.build().unwrap();
        let sizes = slot_bytes(&plan);
        let lt = compute_lifetimes(&plan, &sizes);

        // The planner aliases the collective's output onto its input…
        let aliases = compute_aliases(&plan, &sizes);
        assert_eq!(
            aliases[done.0],
            Some(a),
            "an in-place collective aliases its input"
        );
        // …so the input may not die before the output's last reader, nor before an activation
        // born after the collective that could take those bytes.
        assert_eq!(
            lt[a.0].dies,
            lt[done.0].dies,
            "the collective's input must live exactly as long as its output"
        );
        assert!(
            lt[a.0].dies >= lt[c.0].born,
            "an activation born after the collective may not reuse its input's bytes"
        );
        assert!(
            lt[a.0].dies > lt[a.0].born,
            "the naive range (producer to the collective) is what the extension replaces"
        );
        // And the pool gives `c` bytes of its own, since `a`'s are still spoken for.
        let mem = run_memory_pass(
            &plan,
            &no_ops(&plan),
            &MemoryRecipe::default(),
            RuntimeCapabilities::none(),
            1,
        )
        .unwrap();
        assert_ne!(
            offset_of(&mem, a),
            offset_of(&mem, c),
            "the collective's input bytes were handed to a later activation"
        );
    }

    /// CPU alignment (1) must leave the pre-alignment layout byte-identical:
    /// every offset below is today's number, and any packing change that moves
    /// one of them fails this test.
    #[test]
    fn cpu_alignment_1_keeps_the_exact_mixed_size_offsets() {
        let plan = mixed_plan();
        let recipe = MemoryRecipe::default();
        let mem = run_memory_pass(
            &plan,
            &no_ops(&plan),
            &recipe,
            RuntimeCapabilities::none(),
            1,
        )
        .unwrap();

        // Persistent: w0 at 0, w1 immediately after w0's 16384 bytes.
        assert_eq!(offset_of(&mem, SlotId(0)), 0);
        assert_eq!(offset_of(&mem, SlotId(1)), 16384);
        assert_eq!(mem.persistent_bytes, 16444);
        // Pool: x appends at 0, a appends at 60, y reuses x's block at 0,
        // and out reuses the gap after y at 56.
        assert_eq!(offset_of(&mem, SlotId(2)), 0);
        assert_eq!(offset_of(&mem, SlotId(3)), 60);
        assert_eq!(offset_of(&mem, SlotId(4)), 0);
        assert_eq!(offset_of(&mem, SlotId(5)), 56);
        assert_eq!(mem.transient_pool_bytes, 2108);
        assert_eq!(mem.peak_bytes, 18552);
        assert_eq!(mem.align_bytes, 1);
    }

    /// CUDA alignment (256) rounds every offset up, and the padding is part of
    /// the region sizes — the peak moves. Removing the rounding makes every
    /// assertion here fail (the offsets collapse to the alignment-1 layout).
    #[test]
    fn cuda_alignment_256_rounds_offsets_and_moves_the_peak() {
        let plan = mixed_plan();
        let recipe = MemoryRecipe::default();
        let mem = run_memory_pass(
            &plan,
            &no_ops(&plan),
            &recipe,
            RuntimeCapabilities::none(),
            256,
        )
        .unwrap();

        // Persistent: w0 at 0 (16384 is already 256-aligned), w1 still at
        // 16384, but the region pads w1's 60 B up to the next multiple of 256.
        assert_eq!(offset_of(&mem, SlotId(0)), 0);
        assert_eq!(offset_of(&mem, SlotId(1)), 16384);
        assert_eq!(mem.persistent_bytes, 16640);
        // Pool: x at 0 pads its block up to 256, a lands at 256, y reuses 0,
        // and out lands at 256 — not at 56, which is not 256-aligned.
        assert_eq!(offset_of(&mem, SlotId(2)), 0);
        assert_eq!(offset_of(&mem, SlotId(3)), 256);
        assert_eq!(offset_of(&mem, SlotId(4)), 0);
        assert_eq!(offset_of(&mem, SlotId(5)), 256);
        assert_eq!(mem.transient_pool_bytes, 2304);
        assert_eq!(mem.peak_bytes, 18944);
        assert_eq!(mem.align_bytes, 256);
    }

    #[test]
    fn lifetimes_follow_producer_and_last_consumer() {
        let plan = plan_with(3);
        let lt = compute_lifetimes(&plan, &slot_bytes(&plan));
        // Slot 0 is the weight. Its live *range* ends at its last use (node 2,
        // the third layer) — but it is still persistent, because reuse decisions
        // key off the slot kind, not the range. The two are deliberately
        // separate: a weight's last reader says nothing about whether its
        // storage may be handed to an activation.
        assert_eq!(lt[0].born, 0);
        assert_eq!(lt[0].dies, 2);
        assert!(is_persistent(plan.slot(SlotId(0)).kind));
        // Slot 2 is the first activation; produced by node 0, last read by node 1.
        assert_eq!(lt[2].born, 0);
        assert_eq!(lt[2].dies, 1);
        assert!(lt[2].overlaps(&lt[3]));
        assert!(!lt[2].overlaps(&lt[4]));
    }

    #[test]
    fn peak_accounts_persistent_plus_pool_plus_workspace() {
        let plan = plan_with(2);
        let recipe = MemoryRecipe::default();
        let mem = run_memory_pass(
            &plan,
            &no_ops(&plan),
            &recipe,
            RuntimeCapabilities::none(),
            1,
        )
        .unwrap();
        // One 64x64 f32 weight = 16 KiB.
        assert_eq!(mem.persistent_bytes, 64 * 64 * 4);
        assert!(mem.transient_pool_bytes > 0);
        assert_eq!(
            mem.peak_bytes,
            mem.persistent_bytes + mem.transient_pool_bytes + mem.max_workspace_bytes
        );
    }

    #[test]
    fn slab_pool_reuses_storage_for_disjoint_lifetimes() {
        let plan = plan_with(4);
        let recipe = MemoryRecipe {
            pool: MemoryPool::Slab,
            ..Default::default()
        };
        let mem = run_memory_pass(
            &plan,
            &no_ops(&plan),
            &recipe,
            RuntimeCapabilities::none(),
            1,
        )
        .unwrap();

        let activation_bytes = 8 * 64 * 4;
        // Four layers, each 8x64 f32, but layer N's activation dies when layer
        // N+1 consumes it — so a pool of one slot plus the live set is enough,
        // nowhere near 4x.
        assert!(
            mem.transient_pool_bytes < 4 * activation_bytes,
            "pool {} B should be well under {} B",
            mem.transient_pool_bytes,
            4 * activation_bytes
        );
    }

    #[test]
    fn no_pool_allocates_every_slot_separately() {
        let plan = plan_with(4);
        let recipe = MemoryRecipe {
            pool: MemoryPool::None,
            ..Default::default()
        };
        let mem = run_memory_pass(
            &plan,
            &no_ops(&plan),
            &recipe,
            RuntimeCapabilities::none(),
            1,
        )
        .unwrap();
        let transient: u64 = mem
            .allocations
            .iter()
            .filter(|a| {
                a.policy == ActivationPolicy::Keep
                    && !matches!(
                        a.placement,
                        Placement::Persistent { .. } | Placement::Aliased(_)
                    )
            })
            .map(|a| a.bytes)
            .sum();
        assert_eq!(mem.transient_pool_bytes, transient);
    }

    #[test]
    fn budget_below_peak_is_refused_and_names_the_hottest_step() {
        let plan = plan_with(3);
        let recipe = MemoryRecipe {
            budget_bytes: Some(1), // impossible
            ..Default::default()
        };
        let mem = run_memory_pass(
            &plan,
            &no_ops(&plan),
            &recipe,
            RuntimeCapabilities::none(),
            1,
        )
        .unwrap();
        let err = enforce_budget(&mem, &plan).unwrap_err();
        match err {
            PlanError::MemoryBudgetExceeded {
                peak,
                budget,
                hottest_op,
                suggestions,
                ..
            } => {
                assert_eq!(budget, 1);
                assert!(peak > 1);
                assert!(!hottest_op.is_empty());
                assert!(!suggestions.is_empty());
            }
            other => panic!("expected a budget failure, got {other:?}"),
        }
    }

    #[test]
    fn auto_cannot_relax_without_runtime_support() {
        let plan = plan_with(3);
        let recipe = MemoryRecipe {
            budget_bytes: Some(1),
            ..Default::default()
        };
        // No offload, no recompute: the planner must not pretend it can help.
        let mem = run_memory_pass(
            &plan,
            &no_ops(&plan),
            &recipe,
            RuntimeCapabilities::none(),
            1,
        )
        .unwrap();
        assert!(mem.decisions.is_empty(), "nothing was executable to try");
        assert!(enforce_budget(&mem, &plan).is_err());
    }

    #[test]
    fn auto_relaxes_deterministically_when_the_runtime_supports_it() {
        let plan = plan_with(3);
        let recipe = MemoryRecipe {
            budget_bytes: Some(1),
            pool: MemoryPool::Slab,
            ..Default::default()
        };

        let caps = RuntimeCapabilities {
            offload: true,
            recompute: true,
        };
        let a = run_memory_pass(&plan, &no_ops(&plan), &recipe, caps, 1).unwrap();
        let b = run_memory_pass(&plan, &no_ops(&plan), &recipe, caps, 1).unwrap();
        assert_eq!(
            a.decisions, b.decisions,
            "relaxation must be a deterministic function of the plan"
        );
        assert!(
            !a.decisions.is_empty(),
            "with support available the planner should have used it"
        );
        // Every decision is recorded with a reason naming the pressure.
        for d in &a.decisions {
            assert!(d.reason.contains("peak"), "reason: {}", d.reason);
        }
    }

    #[test]
    fn unsupported_policy_is_reported_not_projected() {
        let plan = plan_with(2);
        let mut recipe = MemoryRecipe::default();
        recipe.ops.insert(
            "elementwise_unary".to_string(),
            rustrain_ops::OpMemoryRecipe {
                activation_policy: Some(ActivationPolicy::Offload),
            },
        );
        let mem = run_memory_pass(
            &plan,
            &no_ops(&plan),
            &recipe,
            RuntimeCapabilities::none(),
            1,
        )
        .unwrap();
        assert_eq!(mem.unsupported.len(), plan.nodes.len());
        assert!(
            mem.explain().contains("cannot execute"),
            "explain must surface it: {}",
            mem.explain()
        );
    }
}
