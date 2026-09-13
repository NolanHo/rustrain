//! Sharding rules and the propagation pass that inserts communication.
//!
//! This module is where invariant I-3 is enforced: a model declares *where*
//! tensors live, and the framework derives *what communication* that implies.
//! Nothing here is model-specific — it is a table over the fixed primitive
//! vocabulary.
//!
//! The pass is a rewrite: it reads a plan, computes the layout each node needs
//! and produces, and — where the declared layout disagrees with what a consumer
//! needs — splices in an intrinsic collective node and repoints the consumers.

use rustrain_parallel::{
    Collective, DimNormalizer, GroupMask, Mesh, ParallelLayout, ReduceOp, ShardError, ShardSpec,
    transitions,
};

use crate::PlanError;
use crate::attrs::Attrs;
use crate::ir::{NodeId, OpRef, Plan, PlanNode, Slot, SlotId, SlotKind, Trace, intrinsic};

/// Why a sharding rule could not produce an answer.
///
/// Distinct from `ShardError`: that one describes a *conversion* the collective
/// rules cannot express, this one describes an operator whose distribution the
/// framework has no rule for. Both are hard errors.
#[derive(Debug, thiserror::Error)]
pub enum DeriveError {
    #[error(
        "cannot derive sharding for `{op}`: weight layout {layout} has no rule; \
         expected replicate, a single shard on the output dim (1 / -1, column parallel) \
         or a single shard on the contraction dim (0 / -2, row parallel)"
    )]
    UnsupportedWeightLayout { op: String, layout: String },
}

/// How an operator relates the sharding of its operands to its results.
///
/// The vocabulary is fixed (spec §2.4), so the rules live in the framework
/// rather than in the plugin: a plugin implements *math*, not distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShardRule {
    /// Every operand and result must share one layout. Covers elementwise ops,
    /// norms, softmax, RoPE, quantization and shape manipulation.
    Elementwise,
    /// `y = x @ w`, with `w` as `[K, N]` — the contraction last-but-one, the
    /// output features last.
    ///
    /// Sharding `w` on its output dim (1 or -1) splits the output features:
    /// column parallel, no collective. Sharding it on the contraction dim
    /// (0 or -2) leaves each rank holding a partial sum: row parallel, which
    /// is what forces an all-reduce downstream.
    ///
    /// The convention is stated here because it is not inferable from the rule
    /// table, and getting it backwards shards silently. Before a second
    /// provider with a different weight layout is added, the layout must become
    /// a declared property of the operator rather than a framework assumption.
    Linear,
    /// `a @ b` with the contraction on `a`'s last and `b`'s second-to-last dim.
    MatMul,
    /// The operator's distribution is not inferable; the plan's declared
    /// layouts are authoritative and only explicit conversions are inserted.
    Declared,
}

/// Classifies a primitive by name. Unknown names are [`ShardRule::Declared`],
/// which is the conservative choice: the framework will not invent distribution
/// semantics for an operator it does not understand.
///
/// An operator added to the vocabulary without a rule here does not fail — it
/// silently falls to `Declared`, which means its distribution is taken from the
/// plan rather than derived. That is safe but means no collective is ever
/// inserted around it, so the vocabulary and this table have to move together.
pub fn rule_for(op: &str) -> ShardRule {
    match op {
        "elementwise_unary" | "elementwise_binary" | "compare" | "softmax" | "rmsnorm"
        | "layernorm" | "rope" | "quantize" | "dequantize" | "amax_update" | "view" | "reshape"
        | "transpose" | "narrow" | "cat" | "broadcast" | "gather" | "scatter" | "cross_entropy" => {
            ShardRule::Elementwise
        }
        // A lookup **is** a linear over its table: the ids select rows of `w`, so a table sharded on
        // its vocabulary axis owes exactly what a row-parallel linear owes (a partial sum over the
        // group, materialized by an all-reduce) and one sharded on its embedding axis owes a
        // column-parallel output. Classifying it as `Elementwise` asked for the *activation's*
        // layout instead, which turns the declared shard on the table into a conflict.
        "linear" | "embedding" => ShardRule::Linear,
        "matmul" | "bmm" => ShardRule::MatMul,
        _ => ShardRule::Declared,
    }
}

/// The distribution a node's inputs must have and its outputs will have.
#[derive(Clone, Debug, PartialEq)]
pub struct DerivedShards {
    pub required_inputs: Vec<ParallelLayout>,
    pub outputs: Vec<ParallelLayout>,
}

/// Computes required input layouts and produced output layouts for one node.
///
/// `declared_inputs` are the layouts the plan assigned to the input slots; they
/// are the starting point for the derivation, not a constraint to satisfy.
/// `input_ranks` / `output_ranks` carry the tensor rank of each corresponding
/// input and output (one entry per layout).
///
/// Every layout is resolved against its own rank on the way in **and** out, so
/// the rules — and every layout the caller stores — see exactly one spelling
/// of a dim. `shard(0, g)` on a rank-1 `[H]` and `shard(-1, g)` on a rank-2
/// `[S, H]` are the same distribution there but *different axes* once the
/// spellings are mixed: a rule that copied one onto the other renamed the axis,
/// and the transition table then refused a conversion nobody should have asked
/// for. Resolution is what makes the two spellings provably one fact.
pub fn derive(
    rule: ShardRule,
    op: &str,
    declared_inputs: &[ParallelLayout],
    declared_outputs: &[ParallelLayout],
    input_ranks: &[i64],
    output_ranks: &[i64],
) -> Result<DerivedShards, DeriveError> {
    let inputs: Vec<ParallelLayout> = declared_inputs
        .iter()
        .enumerate()
        .map(|(i, layout)| canonicalize(layout, input_ranks.get(i).copied().unwrap_or(0)))
        .collect();
    let first = || {
        inputs
            .first()
            .cloned()
            .unwrap_or_else(ParallelLayout::replicate)
    };

    let mut derived = match rule {
        ShardRule::Elementwise => {
            // An elementwise op maps values pointwise, so it cannot consume an *incomplete* sum:
            // `f(Σ x)` is not `Σ f(x)`, and the next layer is defined on the complete value. A
            // partial operand therefore forces the node back to `Replicate` — the conversion is the
            // all-reduce the transition table already emits — instead of being copied into every
            // other operand (which is how a vocabulary-sharded embedding used to make its
            // downstream norm demand a partial *weight*).
            //
            // The node's distribution is anchored on an operand whose rank is the output's. A
            // broadcast operand's layout is spelled on fewer axes, and copying it onto the output
            // (or a higher-rank operand) renames the axis — `shard(0, g)` on a rank-1 `[H]` is the
            // feature axis, while on a rank-2 `[S, H]` it is the sequence axis, and the walk then
            // demands a conversion between two different distributions. The one exception is a
            // single-operand view (`reshape`/`narrow`): its operand *is* the distribution, carried
            // onto the output as-is — the table has no shape algebra to remap its axis, and the
            // view's consumers are the description's responsibility.
            let out_rank = output_ranks.first().copied().unwrap_or(0);
            let anchor = inputs
                .iter()
                .zip(input_ranks)
                .find(|(_, rank)| **rank == out_rank)
                .map(|(layout, _)| layout.clone())
                .or_else(|| inputs.first().cloned())
                .unwrap_or_else(ParallelLayout::replicate);
            let out = if anchor.partial.is_some() {
                ParallelLayout::replicate()
            } else {
                anchor
            };

            let required_inputs: Vec<ParallelLayout> = inputs
                .iter()
                .zip(input_ranks)
                .map(|(declared, rank)| {
                    // Operands that are replicated *by declaration* stay replicated. A replica is
                    // broadcast locally by the kernel, while narrowing one to the activation's shard
                    // is a local view the compiler refuses to materialize — so requiring `out` of a
                    // replica would ask for a conversion that has no owner. A *partial* operand is
                    // completed instead (the same fix-2 argument: the op cannot apply pointwise to
                    // an incomplete sum).
                    if declared.is_replicated() || declared.partial.is_some() {
                        return ParallelLayout::replicate();
                    }
                    // A single-operand view carries its own distribution.
                    if declared_inputs.len() == 1 {
                        return declared.clone();
                    }
                    if *rank == out_rank {
                        return out.clone();
                    }
                    // A broadcast operand (rank < output rank): require the output's shards on the
                    // axes the operand has. Broadcasting aligns trailing axes, so an output shard
                    // on axis `d` is the operand's axis `d - (out_rank - rank)`; output shards on
                    // axes the operand lacks are broadcast from the group and need nothing here.
                    let offset = out_rank.saturating_sub(*rank);
                    ParallelLayout {
                        dims: out
                            .dims
                            .iter()
                            .filter(|spec| spec.dim >= offset)
                            .map(|spec| ShardSpec {
                                dim: spec.dim - offset,
                                group: spec.group,
                            })
                            .collect(),
                        partial: None,
                    }
                })
                .collect();
            DerivedShards {
                required_inputs,
                outputs: vec![out; declared_outputs.len()],
            }
        }

        ShardRule::Linear => {
            // y = x @ w ; declared_inputs = [x, w], with w as [K, N]: the
            // contraction last-but-one, the output features last. Dims arrive
            // resolved against the weight's own rank, so on a rank-2 `[K, N]`
            // weight the contraction is axis 0 and the output features axis 1
            // (the spellings `0`/`-2` and `1`/`-1` name those axes before
            // resolution).
            let x = first();
            let w = inputs
                .get(1)
                .cloned()
                .unwrap_or_else(ParallelLayout::replicate);

            let out = match (w.is_replicated(), w.dims.as_slice(), w.partial.as_ref()) {
                // An unsharded weight: the output keeps whatever layout the
                // activation has.
                (true, _, _) => x.clone(),
                // Weight sharded on its output dim as a *single* spec: output
                // features split across ranks (column parallel), no collective
                // owed.
                (false, [ShardSpec { dim, group }], None) if *dim == 1 => {
                    ParallelLayout::shard(-1, *group)
                }
                // Contraction split across ranks (row parallel) as a single
                // spec: every rank holds a partial sum, which forces the
                // all-reduce downstream.
                (false, [ShardSpec { dim, group }], None) if *dim == 0 => {
                    ParallelLayout::partial(ReduceOp::Sum, *group)
                }
                // Anything else — two shards on the weight, an existing
                // partial, a partial weight — has no rule in the table, and
                // guessing here would silently pick a distribution nobody
                // declared. Refuse and name the layout.
                _ => {
                    return Err(DeriveError::UnsupportedWeightLayout {
                        op: op.to_string(),
                        layout: format!("{w}"),
                    });
                }
            };
            DerivedShards {
                required_inputs: vec![x, w],
                outputs: vec![out; declared_outputs.len()],
            }
        }

        ShardRule::MatMul => {
            // a @ b ; contraction on a's last and b's second-to-last dim. Dims
            // arrive resolved, so the contraction is `a`'s axis `rank_a - 1`
            // and `b`'s `rank_b - 2`, and b's output features are `rank_b - 1`
            // (exactly what the `-1`/`-2` spellings meant before resolution).
            let a = first();
            let b = inputs
                .get(1)
                .cloned()
                .unwrap_or_else(ParallelLayout::replicate);
            let rank_a = input_ranks.first().copied().unwrap_or(0);
            let rank_b = input_ranks.get(1).copied().unwrap_or(0);

            let out = match (single_shard(&a), single_shard(&b)) {
                // The contraction is split: every rank holds a partial sum.
                (Some(sa), _) if sa.dim == rank_a.saturating_sub(1) => {
                    ParallelLayout::partial(ReduceOp::Sum, sa.group)
                }
                (_, Some(sb)) if sb.dim == rank_b.saturating_sub(2) => {
                    ParallelLayout::partial(ReduceOp::Sum, sb.group)
                }
                // b's output dim is split: the result inherits that shard.
                (_, Some(sb)) if sb.dim == rank_b.saturating_sub(1) => {
                    ParallelLayout::shard(-1, sb.group)
                }
                // Any other single shard propagates to the output.
                (Some(sa), _) => ParallelLayout::shard(sa.dim, sa.group),
                // Anything multi-shard or partial: no rule, so no
                // distribution is invented (as before the mask vocabulary).
                _ => ParallelLayout::replicate(),
            };
            DerivedShards {
                required_inputs: vec![a, b],
                outputs: vec![out; declared_outputs.len()],
            }
        }

        ShardRule::Declared => DerivedShards {
            required_inputs: inputs.clone(),
            outputs: declared_outputs
                .iter()
                .zip(output_ranks)
                .map(|(layout, &rank)| canonicalize(layout, rank))
                .collect(),
        },
    };

    // One spelling leaves the function too: every answer is resolved against
    // the rank it belongs to, so callers can store it verbatim.
    derived.required_inputs = derived
        .required_inputs
        .iter()
        .enumerate()
        .map(|(i, layout)| canonicalize(layout, input_ranks.get(i).copied().unwrap_or(0)))
        .collect();
    derived.outputs = derived
        .outputs
        .iter()
        .enumerate()
        .map(|(i, layout)| canonicalize(layout, output_ranks.get(i).copied().unwrap_or(0)))
        .collect();
    Ok(derived)
}

/// The single shard of a layout, when it has exactly one shard and no
/// partial — the only shape the rule table reasons about.
fn single_shard(l: &ParallelLayout) -> Option<ShardSpec> {
    match (l.dims.as_slice(), l.partial.as_ref()) {
        ([spec], None) => Some(*spec),
        _ => None,
    }
}

/// One spliced-in collective, reported so `plan explain` and the tests can show
/// what the compiler decided and why.
#[derive(Clone, Debug, PartialEq)]
pub struct InsertedCollective {
    pub reason: String,
    pub op: &'static str,
    pub group: GroupMask,
    pub reduce: Option<ReduceOp>,
    pub dim: Option<i64>,
    pub source: String,
    pub produced_slot: SlotId,
    pub consumed_slot: SlotId,
}

/// Result of the propagation pass.
#[derive(Clone, Debug)]
pub struct ShardPropagation {
    pub plan: Plan,
    pub inserted: Vec<InsertedCollective>,
}

/// Maps a layout transition to the intrinsic operator that performs it.
fn intrinsic_for(c: &Collective) -> (&'static str, GroupMask, Option<ReduceOp>, Option<i64>) {
    match c {
        Collective::AllReduce { group, op } => (intrinsic::ALL_REDUCE, *group, Some(*op), None),
        Collective::AllGather { group, dim } => (intrinsic::ALL_GATHER, *group, None, Some(*dim)),
        Collective::ReduceScatter { group, dim } => {
            (intrinsic::REDUCE_SCATTER, *group, None, Some(*dim))
        }
        Collective::Broadcast { group, .. } => (intrinsic::BROADCAST, *group, None, None),
    }
}

/// Resolves a layout's shard dims, mirroring the normalization `transitions`
/// performs; used to check that the single emitted collective materializes
/// every target shard.
fn resolve_shards(
    layout: &ParallelLayout,
    norm: &DimNormalizer,
) -> Result<Vec<(i64, GroupMask)>, ShardError> {
    layout
        .dims
        .iter()
        .map(|spec| Ok((norm.normalize(spec.dim)?, spec.group)))
        .collect()
}

/// Validates every slot's declared layout against the mesh and the tensor's
/// rank: each shard dim must exist on the tensor, and every mask — the
/// shards' and the partial's — must address axes of the mesh. A mask over
/// axes whose degrees are all 1 is a legal size-1 group, never an error.
///
/// Reported up front, before the walk, so a broken declaration cannot flow
/// into a derived layout and be smuggled into an inserted collective. Each
/// failure is attributed to the node that owns the layout: the producer of
/// the slot, or — for a plan input such as a weight — its first consumer.
fn validate_layouts(plan: &Plan, mesh: &Mesh) -> Result<(), PlanError> {
    let producer_of = plan.producers();
    let mut first_consumer_of: Vec<Option<usize>> = vec![None; plan.slots.len()];
    for (i, node) in plan.nodes.iter().enumerate() {
        for input in &node.inputs {
            first_consumer_of[input.0].get_or_insert(i);
        }
    }

    for (idx, slot) in plan.slots.iter().enumerate() {
        // A slot no node ever touches is inert: its layout participates in
        // nothing that runs, and there is no node to blame it on.
        let owner = producer_of[idx].map(|p| p.0).or(first_consumer_of[idx]);
        let Some(owner) = owner else { continue };
        let node = NodeId(owner);
        let op = plan.nodes[owner].op.name.clone();

        let norm = DimNormalizer::new(slot.shape.len() as i64)
            .map_err(|source| PlanError::Shard { node, source })?;
        for spec in &slot.layout.dims {
            norm.normalize(spec.dim)
                .map_err(|source| PlanError::Shard { node, source })?;
            if spec.group.validate(mesh).is_err() {
                return Err(PlanError::GroupUnavailable {
                    node,
                    op,
                    group: spec.group,
                });
            }
        }
        if let Some(partial) = &slot.layout.partial
            && partial.group.validate(mesh).is_err()
        {
            return Err(PlanError::GroupUnavailable {
                node,
                op,
                group: partial.group,
            });
        }
    }
    Ok(())
}

/// Runs the propagation pass over a plan.
///
/// The mesh comes from the plan itself (`plan.meta.mesh`, design §1.3): the
/// plan carries the fingerprint, and this is where it is resolved and where
/// every layout's masks are checked against it. A mask bit outside the mesh
/// is [`PlanError::GroupUnavailable`]; a mask over axes whose degrees are all
/// 1 is a legal size-1 group and is never an error.
///
/// Only conversions the framework can express as a single collective are
/// inserted. If two consumers of the same slot demand different layouts, that is
/// reported rather than silently resolved — the scaffold deliberately does not
/// insert fan-out conversions, because doing so is a scheduling decision with
/// real cost and it should be visible in the source plan.
///
/// Resolves a `layout`'s shard dims against a tensor of `rank` dimensions, in
/// place of the caller's spelling.
///
/// Two layouts that differ only in how a dim is *spelled* are the same distribution: `shard(0, g)`
/// and `shard(-1, g)` on a rank-1 tensor, or `shard(-1, g)` and `shard(1, g)` on a rank-2 one. The
/// walks below compare layouts to decide whether a conversion is owed, and they **store** what they
/// derived — so both the comparison and the stored layouts go through this: a plan must carry one
/// spelling of each axis, otherwise a declared weight and the rule that reads it disagree about
/// nothing, and the plan is reported as a conflict. (A 1-D norm weight sharded on its only axis is
/// exactly that case.)
///
/// A dim that does not resolve is left as written: the validator reports it
/// with its node context, and resolving it here silently would bury that.
pub(crate) fn canonicalize(layout: &ParallelLayout, rank: i64) -> ParallelLayout {
    let Ok(norm) = DimNormalizer::new(rank) else {
        return layout.clone();
    };
    let mut resolved = layout.clone();
    for spec in &mut resolved.dims {
        if let Ok(dim) = norm.normalize(spec.dim) {
            spec.dim = dim;
        }
    }
    resolved
}

pub fn propagate(plan: &Plan) -> Result<ShardPropagation, PlanError> {
    // The mask vocabulary is only meaningful next to the mesh that produced
    // it, and a plan stores the fingerprint, not the mesh (invariant I-6):
    // resolve it here, and let a fingerprint that does not describe a valid
    // mesh be reported rather than guessed at.
    let mesh = plan
        .meta
        .mesh
        .to_mesh()
        .map_err(|source| PlanError::Mesh { source })?;
    validate_layouts(plan, &mesh)?;

    // Layouts are rendered through the mesh wherever a human reads them: a mask is bit positions,
    // and `tp` is what the reader is thinking in (design §1.2). `ParallelLayout`'s `Display` (the
    // bit form) stays for the places that have no mesh in hand.
    let show = |layout: &ParallelLayout| layout.describe(&mesh);

    let mut out = plan.clone();
    let mut inserted: Vec<InsertedCollective> = Vec::new();

    let n_slots = plan.slots.len();
    // The layout a slot *actually* holds at this point in the walk. It starts as
    // the declared layout and is replaced whenever a producer turns out to
    // deliver something else, or a consumer forces a conversion.
    let mut effective: Vec<ParallelLayout> = plan.slots.iter().map(|s| s.layout.clone()).collect();
    let mut claimed: Vec<bool> = vec![false; n_slots];
    let mut producer_of: Vec<Option<usize>> = vec![None; n_slots];
    for (i, n) in plan.nodes.iter().enumerate() {
        for o in &n.outputs {
            producer_of[o.0] = Some(i);
        }
    }

    /// A conversion the walk decided on: it is spliced in after the walk so the
    /// emitted nodes land directly after their producer.
    struct Pending {
        slot: SlotId,
        from: ParallelLayout,
        to: ParallelLayout,
        producer: NodeId,
        reason: String,
    }
    let mut pending: Vec<Pending> = Vec::new();

    for (i, n) in plan.nodes.iter().enumerate() {
        let id = NodeId(i);
        if intrinsic::is_intrinsic(&n.op.name) {
            continue; // already explicit; its declared layout is authoritative
        }
        let rule = rule_for(&n.op.name);

        // Feed the rule what the operands actually hold, not what the plan
        // promised. Using the declared layouts here was the bug that made a
        // partial sum never trigger an all-reduce.
        let eff_in: Vec<ParallelLayout> = n.inputs.iter().map(|s| effective[s.0].clone()).collect();
        let input_ranks: Vec<i64> = n
            .inputs
            .iter()
            .map(|s| plan.slot(*s).shape.len() as i64)
            .collect();
        let declared_out: Vec<ParallelLayout> = n
            .outputs
            .iter()
            .map(|s| plan.slot(*s).layout.clone())
            .collect();
        let output_ranks: Vec<i64> = n
            .outputs
            .iter()
            .map(|s| plan.slot(*s).shape.len() as i64)
            .collect();

        let derived = derive(
            rule,
            &n.op.name,
            &eff_in,
            &declared_out,
            &input_ranks,
            &output_ranks,
        )
        .map_err(|source| PlanError::ShardDerivation { node: id, source })?;

        // Input side: a rule may demand a layout the operand does not have.
        for (k, inp) in n.inputs.iter().enumerate() {
            let Some(need) = derived.required_inputs.get(k).cloned() else {
                continue;
            };
            let rank = plan.slot(SlotId(inp.0)).shape.len() as i64;
            let need = canonicalize(&need, rank);
            let held = canonicalize(&effective[inp.0], rank);
            if need == held {
                continue;
            }
            if claimed[inp.0] {
                return Err(PlanError::LayoutConflict {
                    node: id,
                    slot: *inp,
                    index: k,
                    needed: show(&need),
                    held: format!("{held} (a conversion for this slot was already decided)"),
                });
            }
            let producer = producer_of[inp.0].ok_or_else(|| PlanError::LayoutConflict {
                node: id,
                slot: *inp,
                index: k,
                needed: show(&need),
                held: format!(
                    "{} (on a model input, which no node can convert)",
                    show(&held)
                ),
            })?;
            let reason = format!(
                "node {i} ({}) input {k} needs {} but slot {} holds {}",
                n.op.name,
                show(&need),
                inp.0,
                show(&held)
            );
            claimed[inp.0] = true;
            effective[inp.0] = need.clone();
            pending.push(Pending {
                slot: *inp,
                from: held,
                to: need,
                producer: NodeId(producer),
                reason,
            });
        }

        // Output side: a producer must deliver the layout its slot promises.
        for (j, o) in n.outputs.iter().enumerate() {
            let produced = derived
                .outputs
                .get(j)
                .cloned()
                .unwrap_or_else(ParallelLayout::replicate);
            let promised = declared_out[j].clone();
            let rank = plan.slot(*o).shape.len() as i64;
            let produced = canonicalize(&produced, rank);
            let promised = canonicalize(&promised, rank);
            if produced == promised {
                effective[o.0] = produced;
                continue;
            }
            if claimed[o.0] {
                return Err(PlanError::LayoutConflict {
                    node: id,
                    slot: *o,
                    index: j,
                    needed: show(&promised),
                    held: format!(
                        "{} (a conversion for this slot was already decided)",
                        show(&produced)
                    ),
                });
            }
            let reason = format!(
                "node {i} ({}) produces {} on slot {} but the plan declared {}",
                n.op.name,
                show(&produced),
                o.0,
                show(&promised)
            );
            claimed[o.0] = true;
            effective[o.0] = promised.clone();
            pending.push(Pending {
                slot: *o,
                from: produced,
                to: promised,
                producer: id,
                reason,
            });
        }
    }

    // Splice the decided conversions in.
    let mut rewrites: Vec<Option<SlotId>> = vec![None; n_slots];
    let mut spliced_nodes: Vec<usize> = Vec::new();

    for c in pending {
        let norm = DimNormalizer::new(plan.slot(c.slot).shape.len() as i64).map_err(|source| {
            PlanError::Shard {
                node: c.producer,
                source,
            }
        })?;
        let cs = transitions(&c.from, &c.to, &norm).map_err(|source| PlanError::Shard {
            node: c.producer,
            source,
        })?;
        if cs.is_empty() {
            if c.from == c.to {
                continue;
            }
            // The rules say some pairs need no collective because the change is
            // a *local view* (replicate -> shard: each rank keeps its own
            // slice). That is not free — it is a narrow on every rank — and the
            // planner has no view node to insert, so the honest answer is to
            // refuse rather than emit a plan whose declared layout is not what
            // the producer actually wrote.
            return Err(PlanError::LayoutConflict {
                node: c.producer,
                slot: c.slot,
                index: 0,
                needed: format!("{}", c.to),
                held: format!(
                    "{} — no collective performs this conversion; it is a local view, so either \
                     declare the slot's layout as what the producer yields or make the view an \
                     explicit node",
                    c.from
                ),
            });
        }
        if cs.len() > 1 {
            return Err(PlanError::LayoutConflict {
                node: c.producer,
                slot: c.slot,
                index: 0,
                needed: format!("{}", c.to),
                held: format!(
                    "{} requires a {}-step conversion; express the intermediate layout explicitly",
                    c.from,
                    cs.len()
                ),
            });
        }

        // A single emitted collective still has to materialize every target
        // shard: a shard already held by the source survives the collective,
        // and a reduce_scatter produces exactly the shard it scatters onto.
        // Anything else (a partial completed by all_reduce, then a target shard
        // that is only a local narrow; a reduce_scatter that materializes one
        // target shard while another is still a local narrow) leaves the slot's
        // declared layout different from what the producer actually wrote, and
        // the planner has no view node to insert — refuse it the same way the
        // empty-sequence local view is refused.
        let from_shards = resolve_shards(&c.from, &norm).map_err(|source| PlanError::Shard {
            node: c.producer,
            source,
        })?;
        let to_shards = resolve_shards(&c.to, &norm).map_err(|source| PlanError::Shard {
            node: c.producer,
            source,
        })?;
        let produced = match &cs[0] {
            Collective::ReduceScatter { dim, group } => Some((*dim, *group)),
            _ => None,
        };
        if to_shards
            .iter()
            .any(|shard| !from_shards.contains(shard) && produced != Some(*shard))
        {
            return Err(PlanError::LayoutConflict {
                node: c.producer,
                slot: c.slot,
                index: 0,
                needed: format!("{}", c.to),
                held: format!(
                    "{} — {} does not materialize every shard the target declares; the \
                     plan's declared layout is not what the producer actually wrote, so \
                     either declare the slot's layout as what the producer yields or make \
                     the re-slicing an explicit node",
                    c.from, cs[0]
                ),
            });
        }

        let (op, group, reduce, dim) = intrinsic_for(&cs[0]);
        let new_slot = SlotId(out.slots.len());
        rewrites[c.slot.0] = Some(new_slot);

        let mut converted = plan.slot(c.slot).clone();
        converted.name = format!("{}__{}", converted.name, op);
        converted.layout = c.to;

        // The group travels as the mask's integer bits (decision 2): a mask
        // has no name without the mesh, and a node attribute is not the place
        // for one. The compiler reads the bits back and revalidates them.
        let mut attrs = Attrs::new().set(intrinsic::ATTR_GROUP, group.bits() as i64);
        if let Some(r) = reduce {
            attrs.insert(
                intrinsic::ATTR_REDUCE,
                match r {
                    ReduceOp::Sum => "sum",
                    ReduceOp::Max => "max",
                    ReduceOp::Min => "min",
                },
            );
        }
        if let Some(d) = dim {
            attrs.insert(intrinsic::ATTR_DIM, d);
        }

        inserted.push(InsertedCollective {
            reason: c.reason,
            op,
            group,
            reduce,
            dim,
            source: plan.nodes[c.producer.0].source.path.clone(),
            produced_slot: new_slot,
            consumed_slot: c.slot,
        });

        out.slots.push(converted);
        spliced_nodes.push(out.nodes.len());
        let node = PlanNode {
            op: OpRef::new(op),
            inputs: vec![c.slot],
            outputs: vec![new_slot],
            attrs,
            phase: plan.nodes[c.producer.0].phase,
            precision: Default::default(),
            checkpoint: Default::default(),
            stream: Default::default(),
            source: Trace::inserted(
                plan.nodes[c.producer.0].source.path.clone(),
                "shard-propagation",
            ),
        };
        out.nodes.push(node);
    }

    // Repoint every consumer of a rewritten slot at its converted twin. The
    // spliced collectives are skipped: their own input *is* the original slot,
    // and repointing it would make each node read its own output.
    for (i, n) in out.nodes.iter_mut().enumerate() {
        if spliced_nodes.contains(&i) {
            continue;
        }
        for inp in n.inputs.iter_mut() {
            if let Some(new) = rewrites[inp.0] {
                *inp = new;
            }
        }
    }

    out = reorder_topologically(&out)?;

    let _ = &inserted;
    Ok(ShardPropagation {
        plan: out,
        inserted,
    })
}

/// Emits nodes so that every node follows the producers of its inputs.
///
/// The propagation pass appends its inserted collectives at the end, which
/// keeps the graph valid but makes the order unreadable. This restores emission
/// order (Kahn) and is stable for independent nodes.
fn reorder_topologically(plan: &Plan) -> Result<Plan, PlanError> {
    use std::collections::BinaryHeap;

    let producers = plan.producers();
    let mut deps: Vec<usize> = vec![0; plan.nodes.len()];
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); plan.nodes.len()];

    for (i, n) in plan.nodes.iter().enumerate() {
        let mut seen = Vec::new();
        for inp in &n.inputs {
            if let Some(p) = producers[inp.0]
                && p.0 != i
                && !seen.contains(&p.0)
            {
                seen.push(p.0);
            }
        }
        deps[i] = seen.len();
        for d in seen {
            consumers[d].push(i);
        }
    }

    // Min-heap on index keeps the original relative order of independent nodes.
    let mut ready: BinaryHeap<std::cmp::Reverse<usize>> = (0..plan.nodes.len())
        .filter(|i| deps[*i] == 0)
        .map(std::cmp::Reverse)
        .collect();

    let mut order = Vec::with_capacity(plan.nodes.len());
    while let Some(std::cmp::Reverse(i)) = ready.pop() {
        order.push(i);
        for &c in &consumers[i] {
            deps[c] -= 1;
            if deps[c] == 0 {
                ready.push(std::cmp::Reverse(c));
            }
        }
    }

    if order.len() != plan.nodes.len() {
        return Err(PlanError::Digest(
            "sharding propagation produced a cyclic plan".to_string(),
        ));
    }

    let mut out = plan.clone();
    out.nodes = order.iter().map(|&i| plan.nodes[i].clone()).collect();
    out.check_structure()?;
    Ok(out)
}

/// A convenience accessor for tests and reports.
pub fn layout_of(slot: &Slot) -> String {
    format!("{}", slot.layout)
}

/// True when the slot's kind suggests it should never be split, used by the
/// validator to catch obviously wrong declarations.
pub fn is_distributable(slot: &Slot) -> bool {
    !matches!(slot.kind, SlotKind::State)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlanBuilder;
    use rustrain_abi::ffi::RsDtype;
    use rustrain_ops::Phase;
    use rustrain_parallel::{Mesh, ParallelConfig, PartialSpec};

    /// The canonical five axes with `tp = 2`; `tp` is the first axis of
    /// [`Mesh::from_config`], so its mask is bit 0.
    fn tp_mesh() -> Mesh {
        Mesh::from_config(&ParallelConfig {
            tensor: 2,
            ..Default::default()
        })
    }

    fn tp_mask() -> GroupMask {
        GroupMask::single(0).expect("bit 0 always fits")
    }

    /// `w` is `[K, N]`; sharding the output dim (1 / -1) is column parallel and
    /// owes nothing. Both spellings name the same axis, and `derive` returns the
    /// canonical (resolved) one — one spelling leaves the function.
    #[test]
    fn column_parallel_linear_needs_no_collective() {
        for dim in [1, -1] {
            let d = derive(
                ShardRule::Linear,
                "linear",
                &[
                    ParallelLayout::replicate(),
                    ParallelLayout::shard(dim, tp_mask()),
                ],
                &[ParallelLayout::replicate()],
                &[2, 2],
                &[2],
            )
            .unwrap();
            assert_eq!(
                d.outputs[0],
                ParallelLayout::shard(1, tp_mask()),
                "sharding the weight's output dim {dim} must stay column parallel, spelled \
                 canonically as axis 1"
            );
        }
    }

    /// Sharding the contraction dim (0 / -2) leaves a partial sum in every rank.
    #[test]
    fn row_parallel_linear_yields_partial_sum() {
        for dim in [0, -2] {
            let d = derive(
                ShardRule::Linear,
                "linear",
                &[
                    ParallelLayout::replicate(),
                    ParallelLayout::shard(dim, tp_mask()),
                ],
                &[ParallelLayout::replicate()],
                &[2, 2],
                &[2],
            )
            .unwrap();
            assert_eq!(
                d.outputs[0],
                ParallelLayout::partial(ReduceOp::Sum, tp_mask()),
                "sharding the weight's contraction dim {dim} must produce a partial sum"
            );
        }
    }

    /// The broadcast case that first exposed the cross-rank copy: `g = aneg * sp`
    /// with `aneg` a rank-1 `[H]` sharded on its only axis and `sp` a rank-2
    /// `[S, H]` sharded on its last axis. The rule must anchor the output on the
    /// operand that has the output's rank (`sp`) and translate the requirement
    /// for the rank-1 operand back onto *its* axis — copying the first operand's
    /// layout onto the rank-2 ones renamed `shard(0, g)` into the sequence axis
    /// and made the walk demand an all-gather that no target layout owns.
    #[test]
    fn a_broadcast_elementwise_anchors_on_the_output_ranked_operand() {
        let g = tp_mask();
        let d = derive(
            ShardRule::Elementwise,
            "elementwise_binary",
            &[ParallelLayout::shard(0, g), ParallelLayout::shard(-1, g)],
            &[ParallelLayout::replicate()],
            &[1, 2],
            &[2],
        )
        .unwrap();
        assert_eq!(
            d.outputs[0],
            ParallelLayout::shard(1, g),
            "the output follows the operand that has the output's rank, spelled canonically"
        );
        assert_eq!(
            d.required_inputs,
            vec![ParallelLayout::shard(0, g), ParallelLayout::shard(1, g)],
            "each operand is required on its *own* axes: the rank-1 operand keeps shard(0, g) on \
             its only axis, the rank-2 operand is required on the feature axis"
        );
    }

    /// A single-operand view op carries its operand's distribution as-is, even
    /// across a rank change: `reshape`/`narrow` have no shape algebra to remap
    /// the axis, and demanding a conversion on the view's own input would insert
    /// a collective the description never asked for.
    #[test]
    fn a_single_operand_view_carries_its_operands_distribution() {
        let g = tp_mask();
        let d = derive(
            ShardRule::Elementwise,
            "reshape",
            &[ParallelLayout::shard(-1, g)],
            &[ParallelLayout::replicate()],
            &[2],
            &[4],
        )
        .unwrap();
        assert_eq!(
            d.outputs[0],
            ParallelLayout::shard(1, g),
            "the view copies the operand's resolved layout"
        );
        assert_eq!(
            d.required_inputs,
            vec![ParallelLayout::shard(1, g)],
            "the view's operand keeps its own layout — no conversion is owed at a view"
        );
    }

    /// A broadcast operand that disagrees with the anchored output on its own
    /// axis is a real requirement, not a copy: `aneg` sharded over a *different*
    /// group than the output's feature shard must be demanded, and the walk
    /// turns that into the group-mismatch refusal.
    #[test]
    fn a_broadcast_operand_is_required_on_its_own_axis() {
        let tp = tp_mask();
        let ep = GroupMask::single(1).expect("bit 1 always fits");
        let d = derive(
            ShardRule::Elementwise,
            "elementwise_binary",
            &[ParallelLayout::shard(0, ep), ParallelLayout::shard(-1, tp)],
            &[ParallelLayout::replicate()],
            &[1, 2],
            &[2],
        )
        .unwrap();
        assert_eq!(
            d.required_inputs,
            vec![ParallelLayout::shard(0, tp), ParallelLayout::shard(1, tp)],
            "the rank-1 operand is required on its own axis with the *output's* group, so the \
             walk can refuse the mismatch instead of computing with misaligned slices"
        );
    }

    /// Decision 6: a weight with several shards — or any partial — has no rule
    /// in the table. Guessing (pick one shard? fold the groups?) would silently
    /// choose distribution semantics nobody declared, so the honest answer is a
    /// refusal that names the layout. This pins the refusal; if a real model
    /// needs such a weight, the rule table grows a case and this test moves
    /// with it.
    #[test]
    fn multi_shard_or_partial_weight_layout_is_refused() {
        let g = tp_mask();
        let weights = [
            // Two independent shards: no single collective turns this into a
            // `Partial` or a single `Shard`.
            ParallelLayout {
                dims: vec![
                    ShardSpec { dim: 0, group: g },
                    ShardSpec { dim: 1, group: g },
                ],
                partial: None,
            },
            // A partial weight is not a shard at all.
            ParallelLayout::partial(ReduceOp::Sum, g),
            // One shard plus a partial: two facts, no rule.
            ParallelLayout {
                dims: vec![ShardSpec { dim: 0, group: g }],
                partial: Some(PartialSpec {
                    op: ReduceOp::Sum,
                    group: g,
                }),
            },
        ];
        for weight in weights {
            let err = derive(
                ShardRule::Linear,
                "linear",
                &[ParallelLayout::replicate(), weight],
                &[ParallelLayout::replicate()],
                &[2, 2],
                &[2],
            )
            .unwrap_err();
            match err {
                DeriveError::UnsupportedWeightLayout { ref layout, .. } => {
                    assert!(
                        layout.contains("shard") || layout.contains("partial"),
                        "the refusal must name the layout: {layout}"
                    );
                }
            }
        }
    }

    /// The headline behaviour: a row-parallel linear feeding a replicate slot
    /// must make the compiler insert an all-reduce, and it must be visible in
    /// the plan (contract S-2).
    #[test]
    fn row_parallel_linear_inserts_all_reduce() {
        let mut b = PlanBuilder::new("tp", Phase::Forward, tp_mesh().fingerprint());
        let x = b.slot("x", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        let w = b.slot_with_layout(
            "w",
            RsDtype::F32,
            vec![8, 8],
            SlotKind::Weight,
            // Contraction dim => every rank holds a partial sum.
            ParallelLayout::shard(0, tp_mask()),
        );
        let y = b.slot("y", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        let z = b.slot("z", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        b.node(
            OpRef::new("linear"),
            vec![x, w],
            vec![y],
            Attrs::new(),
            "layer0.linear",
        );
        b.node(
            OpRef::new("elementwise_unary"),
            vec![y],
            vec![z],
            Attrs::new().set("kind", "silu"),
            "layer0.act",
        );
        let plan = b.build().unwrap();

        let prop = propagate(&plan).unwrap();
        assert_eq!(prop.inserted.len(), 1, "expected exactly one all-reduce");
        let ins = &prop.inserted[0];
        assert_eq!(ins.op, intrinsic::ALL_REDUCE);
        assert_eq!(ins.group, tp_mask());
        assert_eq!(ins.reduce, Some(ReduceOp::Sum));
        assert!(
            ins.reason.contains("partial") && ins.reason.contains("declared"),
            "reason should name the produced layout and the promise: {}",
            ins.reason
        );

        // The consumer now reads the converted slot.
        let act = prop
            .plan
            .nodes
            .iter()
            .find(|n| n.op.name == "elementwise_unary")
            .unwrap();
        assert_eq!(act.inputs[0], ins.produced_slot);
        assert_eq!(
            prop.plan.slot(ins.produced_slot).layout,
            ParallelLayout::replicate()
        );
    }

    /// A declared layout the producer cannot deliver by any collective is a
    /// contradiction, not a thing to paper over with a view nobody inserted.
    #[test]
    fn a_view_only_conversion_is_refused() {
        let mut b = PlanBuilder::new(
            "views",
            Phase::Forward,
            Mesh::from_config(&ParallelConfig::default()).fingerprint(),
        );
        let x = b.slot("x", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        // Produces a replicate layout...
        let r = b.slot("r", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        // ...but the consumer's slot claims to be sharded. The mesh's tp
        // degree is 1, which is a legal size-1 group — the refusal below is
        // about the missing view, not about the mask.
        let s = b.slot_with_layout(
            "s",
            RsDtype::F32,
            vec![4, 8],
            SlotKind::Activation,
            ParallelLayout::shard(-1, tp_mask()),
        );
        b.node(
            OpRef::new("elementwise_unary"),
            vec![x],
            vec![r],
            Attrs::new(),
            "a",
        );
        b.node(
            OpRef::new("elementwise_unary"),
            vec![r],
            vec![s],
            Attrs::new(),
            "b",
        );
        let plan = b.build().unwrap();

        let err = propagate(&plan).unwrap_err();
        match err {
            PlanError::LayoutConflict { held, .. } => {
                assert!(
                    held.contains("local view"),
                    "the error must say why no collective applies: {held}"
                );
            }
            other => panic!("expected a layout conflict, got {other:?}"),
        }
    }

    /// A two-axis mesh (tp = bit 0, ep = bit 1) for the reviewer's attack
    /// pairs: `tp` has degree 2, `ep` degree 3.
    fn tp_ep_mesh() -> Mesh {
        Mesh::new(vec![("tp".to_string(), 2), ("ep".to_string(), 3)]).unwrap()
    }

    fn tp_ep_masks() -> (GroupMask, GroupMask) {
        (
            GroupMask::single(0).expect("bit 0 always fits"),
            GroupMask::single(1).expect("bit 1 always fits"),
        )
    }

    /// A plan whose walk must decide the conversion `from -> to` on a slot that
    /// **has a producer**, so the conversion reaches the transition table — the
    /// hand-declared pair the reviewer drove through `propagate`.
    ///
    /// Since the lead's fix 2, an elementwise node can no longer consume a
    /// partial at all (it demands `replicate` of such an operand), so a
    /// one-input elementwise pair — input declared `from`, output declared `to`
    /// — refused on the *input* whenever `from` carried a partial, and the
    /// transition table never saw the pair. A `linear` keeps the pair alive: its
    /// rule demands the operands as they are (a linear legitimately consumes a
    /// partial activation and produces a partial output), and with a replicated
    /// weight its produced layout is exactly `from` — so the walk decides
    /// `from -> to` on the **output** slot, the same shape the pair always had.
    fn declared_pair_plan(from: ParallelLayout, to: ParallelLayout) -> Plan {
        let mesh = tp_ep_mesh();
        let mut b = PlanBuilder::new("pair", Phase::Forward, mesh.fingerprint());
        let x = b.slot_with_layout("x", RsDtype::F32, vec![6, 6], SlotKind::Activation, from);
        let w = b.slot("w", RsDtype::F32, vec![6, 6], SlotKind::Weight);
        let y = b.slot_with_layout("y", RsDtype::F32, vec![6, 6], SlotKind::Activation, to);
        b.node(OpRef::new("linear"), vec![x, w], vec![y], Attrs::new(), "a");
        b.build().unwrap()
    }

    /// The lead's fix 2, pinned at the rule level: an elementwise op cannot
    /// consume a partial (`f(Σx)` is not `Σf(x)`), so a partial operand is
    /// demanded as `replicate` — the completion is the all-reduce the walk
    /// inserts — and the output is `replicate` too. This is what moved the pair
    /// tests below onto the two-input shape: with a partial `from`, the old
    /// one-input pair refused on the model input before the transition table
    /// could see the conversion.
    #[test]
    fn an_elementwise_partial_operand_is_demanded_as_replicate() {
        let d = derive(
            ShardRule::Elementwise,
            "elementwise_unary",
            &[ParallelLayout::partial(ReduceOp::Sum, tp_mask())],
            &[ParallelLayout::replicate()],
            &[2],
            &[2],
        )
        .unwrap();
        assert_eq!(
            d.outputs,
            vec![ParallelLayout::replicate()],
            "a partial operand forces the node back to replicate"
        );
        assert_eq!(
            d.required_inputs,
            vec![ParallelLayout::replicate()],
            "the partial operand itself is demanded complete — the conversion is the all-reduce"
        );
    }

    /// **Case F2 (reviewer finding, MEDIUM).** `partial(Sum, tp) -> shard(0,
    /// ep)`: `transitions` emits a single `all_reduce`, but that collective
    /// produces a full-extent tensor — the target's ep shard is a local narrow
    /// nobody materializes. The compiler used to accept the plan; it must
    /// refuse it with the same "declared layout is not what the producer
    /// actually wrote" wording as the empty-sequence refusal.
    #[test]
    fn an_unmaterialized_shard_after_all_reduce_is_refused() {
        let (tp, ep) = tp_ep_masks();
        let plan = declared_pair_plan(
            ParallelLayout::partial(ReduceOp::Sum, tp),
            ParallelLayout::shard(0, ep),
        );

        let err = propagate(&plan).unwrap_err();
        match err {
            PlanError::LayoutConflict { held, .. } => {
                assert!(
                    held.contains("not what the producer actually wrote"),
                    "the refusal must carry the local-view wording: {held}"
                );
            }
            other => panic!("expected a layout conflict, got {other:?}"),
        }
    }

    /// **Case F2, second shape.** `partial(Sum, tp) -> {shard(0, tp),
    /// shard(1, tp)}` used to be accepted after inserting only
    /// `reduce_scatter`: the dim-1 narrow is never materialized. The target
    /// layout's two shards overlap on `tp`, so it is now refused up front by
    /// the table's overlap rule — the compiler never sees a plan to accept.
    #[test]
    fn a_two_shard_target_over_the_same_group_is_refused() {
        let (tp, _) = tp_ep_masks();
        let plan = declared_pair_plan(
            ParallelLayout::partial(ReduceOp::Sum, tp),
            ParallelLayout {
                dims: vec![
                    ShardSpec { dim: 0, group: tp },
                    ShardSpec { dim: 1, group: tp },
                ],
                partial: None,
            },
        );

        let err = propagate(&plan).unwrap_err();
        match err {
            PlanError::Shard {
                source: ShardError::OverlappingGroups { a, b, .. },
                ..
            } => assert_eq!((a, b), (tp, tp)),
            other => panic!("expected an overlapping-groups refusal, got {other:?}"),
        }
    }

    /// The same F2 shape with *disjoint* groups: `partial(Sum, tp) ->
    /// {shard(0, tp), shard(1, ep)}`. `transitions` emits a single
    /// `reduce_scatter(0, tp)` that materializes only the dim-0 shard; the
    /// dim-1 ep shard is still a local narrow, so the compiler refuses it.
    #[test]
    fn a_reduce_scatter_must_materialize_every_target_shard() {
        let (tp, ep) = tp_ep_masks();
        let plan = declared_pair_plan(
            ParallelLayout::partial(ReduceOp::Sum, tp),
            ParallelLayout {
                dims: vec![
                    ShardSpec { dim: 0, group: tp },
                    ShardSpec { dim: 1, group: ep },
                ],
                partial: None,
            },
        );

        let err = propagate(&plan).unwrap_err();
        match err {
            PlanError::LayoutConflict { held, .. } => {
                assert!(
                    held.contains("not what the producer actually wrote"),
                    "the refusal must carry the local-view wording: {held}"
                );
            }
            other => panic!("expected a layout conflict, got {other:?}"),
        }
    }

    /// **Case F1b (reviewer finding, HIGH), end to end.** `{shard(0, tp),
    /// partial(Sum, tp)} -> {shard(0, tp)}` used to be accepted after
    /// inserting only `all_reduce` — a plan whose splice corrupts data. The
    /// overlap refusal must reach `propagate` as a `Shard` error, never a
    /// compiled plan.
    #[test]
    fn overlapping_shard_partial_pair_is_refused_end_to_end() {
        let (tp, _) = tp_ep_masks();
        let plan = declared_pair_plan(
            ParallelLayout {
                dims: vec![ShardSpec { dim: 0, group: tp }],
                partial: Some(PartialSpec {
                    op: ReduceOp::Sum,
                    group: tp,
                }),
            },
            ParallelLayout::shard(0, tp),
        );

        let err = propagate(&plan).unwrap_err();
        match err {
            PlanError::Shard {
                source: ShardError::OverlappingGroups { a, b, .. },
                ..
            } => assert_eq!((a, b), (tp, tp)),
            other => panic!("expected an overlapping-groups refusal, got {other:?}"),
        }
    }

    /// **Case F1a (reviewer finding, MEDIUM), end to end.** `{shard(0, ep),
    /// partial(Sum, tp)} -> {shard(0, tp)}` used to compile a two-collective
    /// splice in the wrong order; the table now refuses the conversion and the
    /// compiler reports it, telling the writer to make the intermediate layout
    /// explicit.
    #[test]
    fn partial_completion_plus_dropped_shard_is_refused_end_to_end() {
        let (tp, ep) = tp_ep_masks();
        let plan = declared_pair_plan(
            ParallelLayout {
                dims: vec![ShardSpec { dim: 0, group: ep }],
                partial: Some(PartialSpec {
                    op: ReduceOp::Sum,
                    group: tp,
                }),
            },
            ParallelLayout::shard(0, tp),
        );

        let err = propagate(&plan).unwrap_err();
        match err {
            PlanError::Shard {
                source: ShardError::PartialCompletionDropsShard { .. },
                ..
            } => {}
            other => panic!("expected a partial-completion refusal, got {other:?}"),
        }
    }

    /// The disjoint single-scatter pair stays accepted: `partial(Sum, tp) ->
    /// shard(0, tp)` is one reduce_scatter that materializes the target shard,
    /// so rule 3 must not over-refuse it.
    #[test]
    fn a_materialized_reduce_scatter_shard_is_accepted() {
        let (tp, _) = tp_ep_masks();
        let plan = declared_pair_plan(
            ParallelLayout::partial(ReduceOp::Sum, tp),
            ParallelLayout::shard(0, tp),
        );

        let prop = propagate(&plan).unwrap();
        assert_eq!(prop.inserted.len(), 1);
        assert_eq!(prop.inserted[0].op, intrinsic::REDUCE_SCATTER);
        assert_eq!(
            prop.plan.slot(prop.inserted[0].produced_slot).layout,
            ParallelLayout::shard(0, tp)
        );
    }

    #[test]
    fn each_consumer_reads_the_converted_slot() {
        let mut b = PlanBuilder::new("conflict", Phase::Forward, tp_mesh().fingerprint());
        let x = b.slot("x", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        let w = b.slot_with_layout(
            "w",
            RsDtype::F32,
            vec![8, 8],
            SlotKind::Weight,
            // `w` is [K, N]; sharding dim 0 splits the contraction, so every
            // rank ends up with a partial sum that has to be reduced.
            ParallelLayout::shard(0, tp_mask()),
        );
        // The linear produces Partial(Sum, tp).
        let p = b.slot("p", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        let r = b.slot("r", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        let s = b.slot("s", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        b.node(
            OpRef::new("linear"),
            vec![x, w],
            vec![p],
            Attrs::new(),
            "lin",
        );
        b.node(
            OpRef::new("elementwise_unary"),
            vec![p],
            vec![r],
            Attrs::new(),
            "a",
        );
        b.node(
            OpRef::new("elementwise_unary"),
            vec![p],
            vec![s],
            Attrs::new(),
            "b",
        );
        let plan = b.build().unwrap();

        let prop = propagate(&plan).unwrap();
        assert_eq!(prop.inserted.len(), 1, "one conversion, materialised once");
        let ins = &prop.inserted[0];
        assert_eq!(ins.op, intrinsic::ALL_REDUCE);

        let consumers: Vec<_> = prop
            .plan
            .nodes
            .iter()
            .filter(|n| n.op.name == "elementwise_unary")
            .collect();
        assert_eq!(consumers.len(), 2);
        for c in consumers {
            assert_eq!(
                c.inputs[0], ins.produced_slot,
                "every consumer must read the converted slot, not the raw partial"
            );
        }
        prop.plan.check_structure().unwrap();
    }

    /// Decision 3: a mask bit outside the mesh is a reported error —
    /// `GroupUnavailable` naming the node whose layout is unusable, the
    /// operator, and the mask — never a panic and never a silent drop.
    #[test]
    fn group_outside_the_mesh_is_reported() {
        // A one-axis mesh: only bit 0 (tp) exists.
        let mesh = Mesh::new(vec![("tp".to_string(), 2)]).unwrap();
        let stray = GroupMask::from_bits(0b100); // bit 2: axis 2 does not exist

        let mut b = PlanBuilder::new("stray", Phase::Forward, mesh.fingerprint());
        let x = b.slot("x", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        let w = b.slot_with_layout(
            "w",
            RsDtype::F32,
            vec![8, 8],
            SlotKind::Weight,
            ParallelLayout::shard(0, stray),
        );
        let y = b.slot("y", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        b.node(
            OpRef::new("linear"),
            vec![x, w],
            vec![y],
            Attrs::new(),
            "layer0.linear",
        );
        let plan = b.build().unwrap();

        let err = propagate(&plan).unwrap_err();
        match err {
            PlanError::GroupUnavailable { node, op, group } => {
                assert_eq!(
                    node,
                    NodeId(0),
                    "the consuming node owns the weight's layout"
                );
                assert_eq!(op, "linear");
                assert_eq!(group, stray);
            }
            other => panic!("expected GroupUnavailable, got {other:?}"),
        }
    }

    /// A mask over axes whose degrees are all 1 is a legal size-1 group — the
    /// no-op case, never an error (decision 3). `propagate` must accept it and
    /// still insert the conversion the layouts ask for.
    #[test]
    fn mask_over_degree_one_axes_is_legal() {
        let mut b = PlanBuilder::new(
            "deg1",
            Phase::Forward,
            Mesh::from_config(&ParallelConfig::default()).fingerprint(),
        );
        let x = b.slot("x", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        let w = b.slot_with_layout(
            "w",
            RsDtype::F32,
            vec![8, 8],
            SlotKind::Weight,
            ParallelLayout::shard(0, tp_mask()),
        );
        let y = b.slot("y", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        b.node(
            OpRef::new("linear"),
            vec![x, w],
            vec![y],
            Attrs::new(),
            "lin",
        );
        let plan = b.build().unwrap();

        // tp has degree 1, so the mask names a size-1 group: the row-parallel
        // partial still collapses through an all-reduce (over that size-1
        // group) rather than being rejected.
        let prop = propagate(&plan).unwrap();
        assert_eq!(prop.inserted.len(), 1);
        assert_eq!(prop.inserted[0].op, intrinsic::ALL_REDUCE);
    }
}
