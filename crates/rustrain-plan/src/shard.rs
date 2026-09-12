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
    Collective, DimNormalizer, GroupKind, ParallelLayout, ProcessGroups, ReduceOp, transitions,
};

use crate::attrs::Attrs;
use crate::ir::{NodeId, OpRef, Plan, PlanNode, Slot, SlotId, SlotKind, Trace, intrinsic};
use crate::PlanError;

/// Why a sharding rule could not produce an answer.
///
/// Distinct from `ShardError`: that one describes a *conversion* the collective
/// rules cannot express, this one describes an operator whose distribution the
/// framework has no rule for. Both are hard errors.
#[derive(Debug, thiserror::Error)]
pub enum DeriveError {
    #[error(
        "cannot derive sharding for `{op}`: weight layout {layout} has no rule; \
         expected replicate, shard(0) (column parallel) or shard(-1) (row parallel)"
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
    /// `y = x @ w^T` (PyTorch `linear`). `w` is `[out, in]`, so sharding `w` on
    /// dim 0 splits the output features (column parallel, no collective) and
    /// sharding it on dim 1 splits the contraction (row parallel, all-reduce).
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
pub fn rule_for(op: &str) -> ShardRule {
    match op {
        "elementwise_unary" | "elementwise_binary" | "softmax" | "rmsnorm" | "layernorm"
        | "rope" | "quantize" | "dequantize" | "amax_update" | "view" | "reshape"
        | "transpose" | "narrow" | "cat" | "broadcast" | "gather" | "scatter"
        | "embedding" | "cross_entropy" => ShardRule::Elementwise,
        "linear" => ShardRule::Linear,
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
pub fn derive(
    rule: ShardRule,
    op: &str,
    declared_inputs: &[ParallelLayout],
    declared_outputs: &[ParallelLayout],
) -> Result<DerivedShards, DeriveError> {
    let first = || {
        declared_inputs
            .first()
            .cloned()
            .unwrap_or(ParallelLayout::Replicate)
    };

    match rule {
        ShardRule::Elementwise => {
            let l = first();
            Ok(DerivedShards {
                required_inputs: vec![l.clone(); declared_inputs.len()],
                outputs: vec![l; declared_outputs.len()],
            })
        }

        ShardRule::Linear => {
            // y = x @ w^T ; declared_inputs = [x, w]
            let x = first();
            let w = declared_inputs
                .get(1)
                .cloned()
                .unwrap_or(ParallelLayout::Replicate);
            let out = match w {
                ParallelLayout::Replicate => x.clone(),
                // Column parallel: each rank owns a slice of the output features.
                ParallelLayout::Shard {
                    dim: 0,
                    group,
                } => ParallelLayout::Shard { dim: -1, group },
                // Row parallel: each rank holds a partial sum. This is the case
                // that forces an all-reduce downstream — the classic Megatron
                // pattern the old code hand-wrote as a detach trick.
                ParallelLayout::Shard {
                    dim: -1,
                    group,
                } => ParallelLayout::Partial {
                    op: ReduceOp::Sum,
                    group,
                },
                other => {
                    return Err(DeriveError::UnsupportedWeightLayout {
                        op: op.to_string(),
                        layout: format!("{other}"),
                    });
                }
            };
            Ok(DerivedShards {
                required_inputs: vec![x, w],
                outputs: vec![out; declared_outputs.len()],
            })
        }

        ShardRule::MatMul => {
            // a @ b ; contraction on a.dim(-1) and b.dim(-2)
            let a = first();
            let b = declared_inputs
                .get(1)
                .cloned()
                .unwrap_or(ParallelLayout::Replicate);

            let out = match (&a, &b) {
                (ParallelLayout::Shard { dim: -1, group }, _) => ParallelLayout::Partial {
                    op: ReduceOp::Sum,
                    group: *group,
                },
                (_, ParallelLayout::Shard { dim: -2, group }) => ParallelLayout::Partial {
                    op: ReduceOp::Sum,
                    group: *group,
                },
                (_, ParallelLayout::Shard { dim: -1, group }) => ParallelLayout::Shard {
                    dim: -1,
                    group: *group,
                },
                (ParallelLayout::Shard { dim, group }, _) => ParallelLayout::Shard {
                    dim: *dim,
                    group: *group,
                },
                _ => ParallelLayout::Replicate,
            };
            let _ = op;
            Ok(DerivedShards {
                required_inputs: vec![a, b],
                outputs: vec![out; declared_outputs.len()],
            })
        }

        ShardRule::Declared => Ok(DerivedShards {
            required_inputs: declared_inputs.to_vec(),
            outputs: declared_outputs.to_vec(),
        }),
    }
}

/// One spliced-in collective, reported so `plan explain` and the tests can show
/// what the compiler decided and why.
#[derive(Clone, Debug, PartialEq)]
pub struct InsertedCollective {
    pub reason: String,
    pub op: &'static str,
    pub group: GroupKind,
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
fn intrinsic_for(c: &Collective) -> (&'static str, GroupKind, Option<ReduceOp>, Option<i64>) {
    match c {
        Collective::AllReduce { group, op } => (intrinsic::ALL_REDUCE, *group, Some(*op), None),
        Collective::AllGather { group, dim } => (intrinsic::ALL_GATHER, *group, None, Some(*dim)),
        Collective::ReduceScatter { group, dim } => {
            (intrinsic::REDUCE_SCATTER, *group, None, Some(*dim))
        }
        Collective::Broadcast { group, .. } => (intrinsic::BROADCAST, *group, None, None),
    }
}

/// Runs the propagation pass over a plan.
///
/// Only conversions the framework can express as a single collective are
/// inserted. If two consumers of the same slot demand different layouts, that is
/// reported rather than silently resolved — the scaffold deliberately does not
/// insert fan-out conversions, because doing so is a scheduling decision with
/// real cost and it should be visible in the source plan.
pub fn propagate(plan: &Plan, groups: &ProcessGroups) -> Result<ShardPropagation, PlanError> {
    let _ = groups;
    let mut out = plan.clone();
    let mut inserted: Vec<InsertedCollective> = Vec::new();

    let n_slots = plan.slots.len();
    // The layout a slot *actually* holds at this point in the walk. It starts as
    // the declared layout and is replaced whenever a producer turns out to
    // deliver something else, or a consumer forces a conversion.
    let mut effective: Vec<ParallelLayout> =
        plan.slots.iter().map(|s| s.layout.clone()).collect();
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
        let eff_in: Vec<ParallelLayout> =
            n.inputs.iter().map(|s| effective[s.0].clone()).collect();
        let declared_out: Vec<ParallelLayout> = n
            .outputs
            .iter()
            .map(|s| plan.slot(*s).layout.clone())
            .collect();

        let derived = derive(rule, &n.op.name, &eff_in, &declared_out)
            .map_err(|source| PlanError::ShardDerivation { node: id, source })?;

        // Input side: a rule may demand a layout the operand does not have.
        for (k, inp) in n.inputs.iter().enumerate() {
            let Some(need) = derived.required_inputs.get(k).cloned() else {
                continue;
            };
            let held = effective[inp.0].clone();
            if need == held {
                continue;
            }
            if claimed[inp.0] {
                return Err(PlanError::LayoutConflict {
                    node: id,
                    slot: *inp,
                    index: k,
                    needed: format!("{need}"),
                    held: format!("{held} (a conversion for this slot was already decided)"),
                });
            }
            let producer = producer_of[inp.0].ok_or_else(|| PlanError::LayoutConflict {
                node: id,
                slot: *inp,
                index: k,
                needed: format!("{need}"),
                held: format!("{held} (on a model input, which no node can convert)"),
            })?;
            pending.push(Pending {
                slot: *inp,
                from: held,
                to: need.clone(),
                producer: NodeId(producer),
                reason: format!(
                    "node {i} ({}) input {k} needs {need} but slot {} holds {held}",
                    n.op.name, inp.0
                ),
            });
            claimed[inp.0] = true;
            effective[inp.0] = need;
        }

        // Output side: a producer must deliver the layout its slot promises.
        for (j, o) in n.outputs.iter().enumerate() {
            let produced = derived
                .outputs
                .get(j)
                .cloned()
                .unwrap_or(ParallelLayout::Replicate);
            let promised = declared_out[j].clone();
            if produced == promised {
                effective[o.0] = produced;
                continue;
            }
            if claimed[o.0] {
                return Err(PlanError::LayoutConflict {
                    node: id,
                    slot: *o,
                    index: j,
                    needed: format!("{promised}"),
                    held: format!("{produced} (a conversion for this slot was already decided)"),
                });
            }
            pending.push(Pending {
                slot: *o,
                from: produced.clone(),
                to: promised.clone(),
                producer: id,
                reason: format!(
                    "node {i} ({}) produces {produced} on slot {} but the plan declared {promised}",
                    n.op.name, o.0
                ),
            });
            claimed[o.0] = true;
            effective[o.0] = promised;
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
            // The transition is local (e.g. replicate -> shard): the producer's
            // output is already in the right place, so nothing is spliced.
            continue;
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

        let (op, group, reduce, dim) = intrinsic_for(&cs[0]);
        let new_slot = SlotId(out.slots.len());
        rewrites[c.slot.0] = Some(new_slot);

        let mut converted = plan.slot(c.slot).clone();
        converted.name = format!("{}__{}", converted.name, op);
        converted.layout = c.to.clone();

        let mut attrs = Attrs::new().set(intrinsic::ATTR_GROUP, intrinsic::group_name(group));
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
            source: Trace::inserted(plan.nodes[c.producer.0].source.path.clone(), "shard-propagation"),
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
    Ok(ShardPropagation { plan: out, inserted })
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
    use rustrain_ops::Phase;
    use rustrain_abi::ffi::RsDtype;
    use rustrain_parallel::ParallelConfig;

    fn groups() -> ProcessGroups {
        ProcessGroups::new(ParallelConfig {
            tensor: 2,
            ..Default::default()
        })
    }

    #[test]
    fn column_parallel_linear_needs_no_collective() {
        let d = derive(
            ShardRule::Linear,
            "linear",
            &[
                ParallelLayout::Replicate,
                ParallelLayout::Shard {
                    dim: 0,
                    group: GroupKind::Tp,
                },
            ],
            &[ParallelLayout::Replicate],
        )
        .unwrap();
        assert_eq!(
            d.outputs[0],
            ParallelLayout::Shard {
                dim: -1,
                group: GroupKind::Tp
            }
        );
    }

    #[test]
    fn row_parallel_linear_yields_partial_sum() {
        let d = derive(
            ShardRule::Linear,
            "linear",
            &[
                ParallelLayout::Replicate,
                ParallelLayout::Shard {
                    dim: -1,
                    group: GroupKind::Tp,
                },
            ],
            &[ParallelLayout::Replicate],
        )
        .unwrap();
        assert_eq!(
            d.outputs[0],
            ParallelLayout::Partial {
                op: ReduceOp::Sum,
                group: GroupKind::Tp
            }
        );
    }

    /// The headline behaviour: a row-parallel linear feeding a replicate slot
    /// must make the compiler insert an all-reduce, and it must be visible in
    /// the plan (contract S-2).
    #[test]
    fn row_parallel_linear_inserts_all_reduce() {
        let mut b = PlanBuilder::new(
            "tp",
            Phase::Forward,
            ParallelConfig {
                tensor: 2,
                ..Default::default()
            },
        );
        let x = b.slot("x", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        let w = b.slot_with_layout(
            "w",
            RsDtype::F32,
            vec![8, 8],
            SlotKind::Weight,
            ParallelLayout::Shard {
                dim: -1,
                group: GroupKind::Tp,
            },
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

        let prop = propagate(&plan, &groups()).unwrap();
        assert_eq!(prop.inserted.len(), 1, "expected exactly one all-reduce");
        let ins = &prop.inserted[0];
        assert_eq!(ins.op, intrinsic::ALL_REDUCE);
        assert_eq!(ins.group, GroupKind::Tp);
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
        assert_eq!(prop.plan.slot(ins.produced_slot).layout, ParallelLayout::Replicate);
    }

    /// Two consumers of a partial sum, each declaring a different result
    /// layout: the compiler materialises the conversion once, and both
    /// consumers read the converted slot. Nothing is guessed silently — the
    /// collective appears in the plan and in `inserted`.
    #[test]
    fn each_consumer_reads_the_converted_slot() {
        let mut b = PlanBuilder::new("conflict", Phase::Forward, ParallelConfig::default());
        let x = b.slot("x", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        let w = b.slot_with_layout(
            "w",
            RsDtype::F32,
            vec![8, 8],
            SlotKind::Weight,
            ParallelLayout::Shard {
                dim: -1,
                group: GroupKind::Tp,
            },
        );
        // The linear produces Partial(Sum, tp).
        let p = b.slot("p", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        let r = b.slot("r", RsDtype::F32, vec![4, 8], SlotKind::Activation);
        let s = b.slot_with_layout(
            "s",
            RsDtype::F32,
            vec![4, 8],
            SlotKind::Activation,
            ParallelLayout::Shard {
                dim: -1,
                group: GroupKind::Tp,
            },
        );
        b.node(OpRef::new("linear"), vec![x, w], vec![p], Attrs::new(), "lin");
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

        let prop = propagate(&plan, &groups()).unwrap();
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
}
