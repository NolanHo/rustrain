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
    Collective, DimNormalizer, GroupMask, Mesh, ParallelLayout, PartialSpec, ReduceOp, ShardError,
    ShardMode, ShardSpec, transitions,
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

    #[error(
        "cannot derive sharding for `{op}`: it declares the pass-through rule, whose outputs \
         follow input 0, but the node has no inputs"
    )]
    PassThroughWithoutInput { op: String },

    #[error(
        "cannot derive sharding for `{op}`: operand {index} layout {layout} has no rule; \
         expected replicate or a single shard, never a partial or several shards"
    )]
    UnsupportedOperandLayout {
        op: String,
        index: usize,
        layout: String,
    },

    #[error(
        "cannot derive sharding for `{op}`: both operands shard the contraction dim over \
         different groups ({a} vs {b}); the output would be a partial over a union of groups \
         the table does not document — refuse rather than drop one operand's shard"
    )]
    ContractionGroupsDiffer {
        op: String,
        a: GroupMask,
        b: GroupMask,
    },

    #[error(
        "cannot derive sharding for `{op}`: both operands' surviving shards land on output \
         dim {dim} over different groups ({a} vs {b}); the dim is shared, so the two splits \
         contradict each other — refuse rather than drop one"
    )]
    OverlappingOutputShard {
        op: String,
        dim: i64,
        a: GroupMask,
        b: GroupMask,
    },

    #[error(
        "cannot derive sharding for `{op}`: operand {operand} carries {layout} on rank {rank} \
         but the output has rank {out_rank}; a rank-changing view has no shape algebra to \
         remap the shard, so carrying it verbatim would rename the axis — declare the output \
         layout explicitly instead"
    )]
    UnmappableViewShard {
        op: String,
        operand: usize,
        layout: String,
        rank: i64,
        out_rank: i64,
    },
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
    /// A lookup `out = w[ids]` is a linear over its table with the operands in
    /// the *kernel's* order `(w, ids)`: the ids select rows of the table, so a
    /// table sharded on its vocabulary axis owes exactly what a row-parallel
    /// linear owes (a partial sum over the group, materialized by an
    /// all-reduce) and one sharded on its embedding axis owes a column-parallel
    /// output. The distribution arithmetic is [`ShardRule::Linear`]'s with the
    /// activation read from input 1 and the weight from input 0 — a separate
    /// rule because the positions are reversed relative to `linear`'s `(x, w)`.
    Embedding,
    /// `a @ b` with the contraction on `a`'s last and `b`'s second-to-last dim.
    MatMul,
    /// The operator's distribution is not inferable; the plan's declared
    /// layouts are authoritative and only explicit conversions are inserted.
    Declared,
    /// Every output inherits **input 0**'s distribution; the other inputs keep
    /// theirs.
    ///
    /// For operators that act per element along the sharded axis without
    /// requiring their other operands to agree with it: a channel-wise causal
    /// convolution whose declared weight splits on the channel axis, an
    /// attention whose queries are split while keys/values may be replicated.
    /// [`ShardRule::Elementwise`] cannot express either — it demands one shared
    /// distribution across every operand, which is false for the weight next to
    /// a split activation.
    PassThrough,
}

/// Where a node's shard rule comes from.
///
/// The framework owns the derivation *algebra* (the [`ShardRule`] kinds and
/// [`derive`]); **which kind an operator uses is the operator's own
/// declaration** (`RsOpDesc::shard`, ABI v2). Production code answers from the
/// operator registry — `impl ShardRules for Registry` below — while tests and
/// synthetic plans answer from [`RuleTable`].
///
/// There is deliberately no fallback: an operator whose providers declare
/// nothing, or two variants that disagree, is an error. Classifying by name
/// (`rule_for`, deleted in ABI v2) is invariant I-5's forbidden pattern — it
/// turned "support an operator the framework has not heard of" into "recompile
/// the framework", and its silent `Declared` default is what let `tp > 1` fall
/// apart on the first operator the table did not list.
pub trait ShardRules {
    /// The rule the operator declares. `Err` names what is missing, so the
    /// caller reports it against the node it was asked about.
    fn rule(&self, op: &str) -> Result<ShardRule, String>;
}

/// Maps the ABI's declared rule onto the algebra, refusing discriminants the
/// framework does not know rather than guessing [`ShardRule::Declared`].
fn rule_from_abi(
    raw: rustrain_abi::ffi::RsShardRule,
    op: &str,
    who: &str,
) -> Result<ShardRule, String> {
    match raw.raw() {
        0 => Ok(ShardRule::Declared),
        1 => Ok(ShardRule::Elementwise),
        2 => Ok(ShardRule::Linear),
        3 => Ok(ShardRule::Embedding),
        4 => Ok(ShardRule::MatMul),
        5 => Ok(ShardRule::PassThrough),
        other => Err(format!(
            "{who} declares sharding rule {other} for `{op}`, which this framework does not know; \
             a rule the planner cannot evaluate is an error, never a silent `Declared`"
        )),
    }
}

/// The shard rules of every registered operator, read from the descriptors.
///
/// Two implementations of one operator must declare the same rule: the rule is
/// a property of the operator's math, not of the code that runs it. A
/// disagreement would make the plan depend on which variant the recipe picked,
/// so it is reported and never resolved by choosing one.
impl ShardRules for rustrain_ops::Registry {
    fn rule(&self, op: &str) -> Result<ShardRule, String> {
        let candidates = self.candidates(op);
        if candidates.is_empty() {
            return Err(format!(
                "no registered implementation of `{op}`; an operator's sharding rule comes from \
                 the plugin that provides it, so a plan cannot be derived without one"
            ));
        }
        let mut decided: Option<(ShardRule, String)> = None;
        for candidate in candidates {
            let who = candidate.plugin_identity();
            let rule = rule_from_abi(candidate.shard(), op, &who)?;
            match &decided {
                Some((first, holder)) if *first != rule => {
                    return Err(format!(
                        "implementations of `{op}` disagree about its sharding rule: {holder} \
                         declares {first:?}, {who} declares {rule:?}; the rule belongs to the \
                         operator, not to the variant"
                    ));
                }
                Some(_) => {}
                None => decided = Some((rule, who)),
            }
        }
        Ok(decided.expect("the candidate list is not empty").0)
    }
}

/// A rule table built by hand: for plans whose operators are synthetic (tests,
/// tooling) and for plans that resolve no registry at all.
#[derive(Clone, Debug, Default)]
pub struct RuleTable {
    rules: std::collections::BTreeMap<String, ShardRule>,
    fallback: Option<ShardRule>,
}

impl RuleTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares `op`'s rule.
    pub fn set(mut self, op: impl Into<String>, rule: ShardRule) -> Self {
        self.rules.insert(op.into(), rule);
        self
    }

    /// Answers every name with `rule` — for plans whose nodes are placeholders.
    /// A test that wants the production behaviour must build the registry
    /// instead: this never sees a descriptor.
    pub fn every(rule: ShardRule) -> Self {
        Self {
            rules: Default::default(),
            fallback: Some(rule),
        }
    }
}

impl ShardRules for RuleTable {
    fn rule(&self, op: &str) -> Result<ShardRule, String> {
        self.rules
            .get(op)
            .copied()
            .or(self.fallback)
            .ok_or_else(|| format!("no sharding rule is declared for `{op}` in this table"))
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
// The rule table needs each operand's rank *and* its shape, one entry per layout: eight values
// that belong together, and bundling them into a struct would only rebuild the same list at
// every call site.
#[allow(clippy::too_many_arguments)]
pub fn derive(
    rule: ShardRule,
    op: &str,
    declared_inputs: &[ParallelLayout],
    declared_outputs: &[ParallelLayout],
    input_ranks: &[i64],
    output_ranks: &[i64],
    input_shapes: &[Vec<i64>],
    output_shapes: &[Vec<i64>],
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
            // demands a conversion between two different distributions. When no operand has the
            // output's rank (a rank-changing unary view, or a binary op broadcasting *both*
            // operands), the first operand is still the distribution — but the mapping across the
            // rank change is per-primitive, because the primitives' axis semantics differ.
            // Broadcasting aligns trailing axes, so a unary `broadcast` — and a binary op
            // broadcasting both operands — carries the shard right-aligned, `axis + (R - r)`. A
            // rank-changing `reshape`/`narrow`/`view` refolds in row-major order, so the shard
            // keeps its axis index — the explicit mapping the table commits to for those
            // primitives. Any axis the output cannot hold is refused rather than renamed.
            let out_rank = output_ranks.first().copied().unwrap_or(0);
            let anchor = inputs
                .iter()
                .zip(input_ranks)
                .find(|(_, rank)| **rank == out_rank)
                .map(|(layout, rank)| (layout.clone(), *rank))
                .or_else(|| {
                    inputs
                        .first()
                        .cloned()
                        .map(|layout| (layout, input_ranks.first().copied().unwrap_or(0)))
                });
            let (anchor, anchor_rank) =
                anchor.unwrap_or_else(|| (ParallelLayout::replicate(), out_rank));
            let trailing = op == "broadcast" || inputs.len() > 1;
            let out = if anchor.partial.is_some() {
                ParallelLayout::replicate()
            } else {
                carry_to_output_rank(
                    &anchor,
                    anchor_rank,
                    out_rank,
                    op,
                    0,
                    trailing,
                    input_shapes.first().map(Vec::as_slice),
                    output_shapes.first().map(Vec::as_slice),
                )?
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
                                mode: spec.mode,
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
                (false, [ShardSpec { dim, group, mode }], None) if *dim == 1 => ParallelLayout {
                    // The output features are exactly as sharded as the weight's — including a
                    // replicating weight, whose slabs overlap so that every rank owns the
                    // features it needs.
                    dims: vec![ShardSpec {
                        dim: -1,
                        group: *group,
                        mode: *mode,
                    }],
                    partial: None,
                },
                // Contraction split across ranks (row parallel) as a single
                // spec: every rank holds a partial sum, which forces the
                // all-reduce downstream.
                (false, [ShardSpec { dim, group, mode }], None) if *dim == 0 => {
                    // A *replicating* weight on the contraction axis has no rule: every rank
                    // would hold an overlapping slab and the all-reduce would count the overlap
                    // twice. Refused rather than guessed.
                    if !mode.is_divide() {
                        return Err(DeriveError::UnsupportedWeightLayout {
                            op: op.to_string(),
                            layout: format!("{w}"),
                        });
                    }
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

        ShardRule::Embedding => {
            // out = w[ids]; declared_inputs = [w, ids] (the kernel's operand
            // order — the weight first, unlike `linear`'s [x, w]). The lookup
            // is still a linear over the table, so the arithmetic is `Linear`'s
            // with the activation read from input 1 and the weight from
            // input 0.
            let x = inputs
                .get(1)
                .cloned()
                .unwrap_or_else(ParallelLayout::replicate);
            let w = first();

            let out = match (w.is_replicated(), w.dims.as_slice(), w.partial.as_ref()) {
                (true, _, _) => x.clone(),
                (false, [ShardSpec { dim, group, mode }], None) if *dim == 1 => ParallelLayout {
                    // The output features are exactly as sharded as the weight's — including a
                    // replicating weight, whose slabs overlap so that every rank owns the
                    // features it needs.
                    dims: vec![ShardSpec {
                        dim: -1,
                        group: *group,
                        mode: *mode,
                    }],
                    partial: None,
                },
                (false, [ShardSpec { dim, group, mode }], None) if *dim == 0 => {
                    // A *replicating* weight on the contraction axis has no rule: every rank
                    // would hold an overlapping slab and the all-reduce would count the overlap
                    // twice. Refused rather than guessed.
                    if !mode.is_divide() {
                        return Err(DeriveError::UnsupportedWeightLayout {
                            op: op.to_string(),
                            layout: format!("{w}"),
                        });
                    }
                    ParallelLayout::partial(ReduceOp::Sum, *group)
                }
                _ => {
                    return Err(DeriveError::UnsupportedWeightLayout {
                        op: op.to_string(),
                        layout: format!("{w}"),
                    });
                }
            };
            DerivedShards {
                required_inputs: vec![w, x],
                outputs: vec![out; declared_outputs.len()],
            }
        }

        ShardRule::MatMul => {
            // a @ b ; contraction on a's last and b's second-to-last dim. Dims
            // arrive resolved, so the contraction is `a`'s axis `rank_a - 1`
            // and `b`'s `rank_b - 2`, and b's output features are `rank_b - 1`
            // (exactly what the `-1`/`-2` spellings meant before resolution).
            //
            // Every distribution that can survive the contraction survives: a shard on either
            // contraction dim becomes the documented partial, and a shard on any other dim
            // rides the output (batch dims are left-aligned, `b`'s output dim is the output's
            // last). The old arms answered with *one* layout, so `a`'s batch shard was dropped
            // the moment `b` carried a shard — `instantiate` then stored a truncated layout and
            // over-claimed the local batch extent. Anything the table cannot express is
            // refused, never dropped.
            let a = first();
            let b = inputs
                .get(1)
                .cloned()
                .unwrap_or_else(ParallelLayout::replicate);
            let rank_a = input_ranks.first().copied().unwrap_or(0);
            let rank_b = input_ranks.get(1).copied().unwrap_or(0);
            let out_rank = output_ranks.first().copied().unwrap_or(0);

            // A partial or multi-shard operand has no rule: the old arm silently answered
            // `replicate` for it, which is a dropped distribution, not a derived one.
            for (index, operand) in [&a, &b].into_iter().enumerate() {
                if !(operand.is_replicated() || single_shard(operand).is_some()) {
                    return Err(DeriveError::UnsupportedOperandLayout {
                        op: op.to_string(),
                        index,
                        layout: format!("{operand}"),
                    });
                }
            }

            let contract_a = single_shard(&a).filter(|s| s.dim == rank_a.saturating_sub(1));
            let contract_b = single_shard(&b).filter(|s| s.dim == rank_b.saturating_sub(2));

            // A shard on either contraction dim leaves every rank holding a partial sum. Two
            // of them over the *same* group are one partial; over *different* groups the
            // result would be a partial over a union of groups the table does not document —
            // refuse rather than drop one.
            // A *replicating* shard on a contraction dim has no rule either: the sum would count
            // every overlapped element twice.
            for contract in [contract_a, contract_b].into_iter().flatten() {
                if !contract.mode.is_divide() {
                    return Err(DeriveError::UnsupportedWeightLayout {
                        op: op.to_string(),
                        layout: format!("{contract:?}"),
                    });
                }
            }
            let partial = match (contract_a, contract_b) {
                (Some(sa), Some(sb)) if sa.group != sb.group => {
                    return Err(DeriveError::ContractionGroupsDiffer {
                        op: op.to_string(),
                        a: sa.group,
                        b: sb.group,
                    });
                }
                (Some(sa), _) => Some(PartialSpec {
                    op: ReduceOp::Sum,
                    group: sa.group,
                }),
                (_, Some(sb)) => Some(PartialSpec {
                    op: ReduceOp::Sum,
                    group: sb.group,
                }),
                (None, None) => None,
            };

            // The surviving shards. Batch dims are left-aligned (torch broadcasts the leading
            // dims of `a` against the leading dims of `b`), so a shard keeps its own index on
            // both sides; `b`'s output dim is the output's last. Two shards landing on the
            // same output dim over different groups contradict each other (the shared dim
            // cannot be split two ways); over the same group they are one fact.
            let mut dims: Vec<ShardSpec> = Vec::new();
            let mut push = |spec: ShardSpec, dim: i64| -> Result<(), DeriveError> {
                if let Some(existing) = dims.iter_mut().find(|d| d.dim == dim) {
                    if existing.group != spec.group {
                        return Err(DeriveError::OverlappingOutputShard {
                            op: op.to_string(),
                            dim,
                            a: existing.group,
                            b: spec.group,
                        });
                    }
                    // Two shards on one dim compose to the weaker promise, as they do when they
                    // are declared (`ParallelLayout::axis_shard`).
                    if let ShardMode::Replicate { unit } = spec.mode {
                        existing.mode = ShardMode::Replicate { unit };
                    }
                } else {
                    dims.push(ShardSpec {
                        dim,
                        group: spec.group,
                        mode: spec.mode,
                    });
                }
                Ok(())
            };
            if let Some(spec) = single_shard(&a).filter(|s| s.dim != rank_a.saturating_sub(1)) {
                push(spec, spec.dim)?;
            }
            if let Some(spec) = single_shard(&b).filter(|s| s.dim != rank_b.saturating_sub(2)) {
                let dim = if spec.dim == rank_b.saturating_sub(1) {
                    out_rank.saturating_sub(1)
                } else {
                    spec.dim
                };
                push(spec, dim)?;
            }

            let out = ParallelLayout { dims, partial };
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

        ShardRule::PassThrough => {
            // The operands keep exactly what they hold (no conversion is ever
            // asked of them); the outputs are whatever input 0 holds. The
            // extents differ between input and output by design — a layout
            // records (dim, group), and the local shape comes from the plan —
            // so "follows" is a statement about the distribution, not the size.
            let first = inputs
                .first()
                .ok_or_else(|| DeriveError::PassThroughWithoutInput { op: op.to_string() })?;
            // The shards follow input 0; a **partial** declared on an output
            // stays, because it states something the operands cannot: an
            // operator whose math sums over a split dim (a fused MoE) declares
            // it, and that declaration is what makes its completion land at the
            // consumer instead of nowhere.
            let outputs = declared_outputs
                .iter()
                .zip(output_ranks)
                .map(|(declared, &rank)| {
                    let mut out = canonicalize(first, rank);
                    if declared.partial.is_some() {
                        out.partial = declared.partial;
                    }
                    out
                })
                .collect();
            DerivedShards {
                required_inputs: inputs.clone(),
                outputs,
            }
        }
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

/// Carries a layout from an operand of rank `rank` onto the output of rank `out_rank`.
///
/// Same-rank: verbatim. Rank-changing with `trailing` (broadcasting semantics): the operand's
/// axes are the output's *trailing* axes, so every shard moves up by `out_rank - rank`; a
/// shrink is impossible and refused. Rank-changing, axis-preserving (`reshape`/`narrow`/
/// `view`, which refold in row-major order): the shard keeps its axis index, which is
/// expressible exactly while the output still has that axis. In every case a shard whose
/// mapped axis does not exist — a scalar operand, a dim that never resolved against its own
/// rank, or a shrink past the shard's axis — is refused rather than renamed.
#[allow(clippy::too_many_arguments)]
fn carry_to_output_rank(
    layout: &ParallelLayout,
    rank: i64,
    out_rank: i64,
    op: &str,
    operand: usize,
    trailing: bool,
    in_shape: Option<&[i64]>,
    out_shape: Option<&[i64]>,
) -> Result<ParallelLayout, DeriveError> {
    if rank == out_rank {
        return Ok(layout.clone());
    }
    if layout.dims.is_empty() {
        return Ok(ParallelLayout::replicate());
    }
    if rank > out_rank && trailing {
        return Err(DeriveError::UnmappableViewShard {
            op: op.to_string(),
            operand,
            layout: format!("{layout}"),
            rank,
            out_rank,
        });
    }
    let offset = if trailing { out_rank - rank } else { 0 };
    let unmappable = || DeriveError::UnmappableViewShard {
        op: op.to_string(),
        operand,
        layout: format!("{layout}"),
        rank,
        out_rank,
    };
    // An axis-preserving rank change refolds in row-major order, and there are two directions:
    //
    //   * a split (`rank < out_rank`) turns the operand's last axis into axes `rank-1 .. out_rank`,
    //     so a replicating shard *on that axis* stays on the outer piece while its unit — counted
    //     in elements of the piece — shrinks by the inner piece. Folding 512 features into
    //     `[.., 2, 256]` turns a unit of 512 into a unit of 256.
    //   * a merge (`rank > out_rank`) folds the operand's axes `out_rank-1 ..` into the operand's
    //     own last axis, so a shard on any of those axes is no longer an axis slab of the output.
    //
    // A split that does not line up with the operand's last axis, and a merge of a replicating
    // shard whose unit would have to be re-sliced, are refused: renaming a distribution is how a
    // wrong slab reaches a kernel without anyone noticing. Shards on axes the refold does not
    // touch carry over verbatim in both directions.
    let split_inner = if !trailing && rank < out_rank {
        match (in_shape, out_shape) {
            (Some(input), Some(output)) if !input.is_empty() => {
                let extra = (out_rank - rank) as usize;
                let folded: i64 = output
                    .get(output.len().saturating_sub(extra + 1)..)
                    .unwrap_or(&[])
                    .iter()
                    .product();
                if folded != *input.last().unwrap() {
                    // The refold does not put the operand's last axis into the output's innermost
                    // axes, so no unit conversion below can be trusted.
                    return Err(unmappable());
                }
                Some(
                    output[output.len() - extra..]
                        .iter()
                        .product::<i64>()
                        .max(1),
                )
            }
            _ => None,
        }
    } else {
        None
    };
    let mut dims = Vec::with_capacity(layout.dims.len());
    for spec in &layout.dims {
        let mut mapped = ShardSpec {
            dim: spec.dim + offset,
            group: spec.group,
            mode: spec.mode,
        };
        if let (Some(inner), ShardMode::Replicate { unit }) = (split_inner, spec.mode) {
            if spec.dim == rank - 1 {
                if inner == 0 || unit % inner != 0 {
                    // The unit is finer than the elements the refold puts inside one output
                    // element: a whole number of output elements per slab cannot be expressed, so
                    // this is refused rather than rounded.
                    return Err(unmappable());
                }
                mapped.mode = ShardMode::Replicate { unit: unit / inner };
            }
        }
        if !trailing && rank > out_rank {
            if let ShardMode::Replicate { .. } = spec.mode {
                if spec.dim >= out_rank - 1 {
                    return Err(unmappable());
                }
            }
        }
        dims.push(mapped);
    }
    if dims.iter().any(|spec| spec.dim < 0 || spec.dim >= out_rank) {
        return Err(unmappable());
    }
    Ok(ParallelLayout {
        dims,
        partial: None,
    })
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

pub fn propagate(plan: &Plan, rules: &dyn ShardRules) -> Result<ShardPropagation, PlanError> {
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
        let rule = rules
            .rule(&n.op.name)
            .map_err(|reason| PlanError::ShardRuleUndeclared {
                node: id,
                op: n.op.name.clone(),
                reason,
            })?;

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

        let input_shapes: Vec<Vec<i64>> = n
            .inputs
            .iter()
            .map(|s| plan.slot(*s).shape.clone())
            .collect();
        let output_shapes: Vec<Vec<i64>> = n
            .outputs
            .iter()
            .map(|s| plan.slot(*s).shape.clone())
            .collect();
        let derived = derive(
            rule,
            &n.op.name,
            &eff_in,
            &declared_out,
            &input_ranks,
            &output_ranks,
            &input_shapes,
            &output_shapes,
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
        // A shape-changing collective must carry its new **local** shape: the
        // plan's slots are already local, so the converted slot starts from the
        // source's local shape, inflates it along every shard dim the source
        // holds (an all-gather produces the global extent there) and deflates
        // it along every shard dim the target declares (a reduce-scatter keeps
        // one slice). All-reduce completes a partial, whose shape was already
        // the full local one, so its dims are untouched. The divisibility is
        // guaranteed by instantiate's local-shape checks; an inflated extent
        // that does not divide is reported rather than truncated.
        for spec in &c.from.dims {
            let degree = spec
                .group
                .degree(&mesh)
                .map_err(|source| PlanError::Mesh { source })?;
            converted.shape[spec.dim as usize] = converted.shape[spec.dim as usize]
                .checked_mul(degree as i64)
                .ok_or_else(|| {
                    PlanError::Digest(format!(
                        "the converted slot `{}` overflows along dim {} when gathered",
                        plan.slot(c.slot).name,
                        spec.dim
                    ))
                })?;
        }
        for spec in &c.to.dims {
            let degree = spec
                .group
                .degree(&mesh)
                .map_err(|source| PlanError::Mesh { source })?;
            if converted.shape[spec.dim as usize] % degree as i64 != 0 {
                return Err(PlanError::Shard {
                    node: c.producer,
                    source: ShardError::NotDivisible {
                        dim: spec.dim,
                        global: converted.shape[spec.dim as usize],
                        divisor: degree as i64,
                    },
                });
            }
            converted.shape[spec.dim as usize] /= degree as i64;
        }
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
    use rustrain_ops::{Phase, Registry};
    use rustrain_parallel::{Mesh, ParallelConfig, PartialSpec};

    /// The rules every test walks with: the ones the built-in reference
    /// provider declares, exactly as production reads them.
    fn rules() -> Registry {
        let mut registry = Registry::new();
        // SAFETY: the built-in descriptors are leaked by `PluginBuilder`, so
        // they live as long as the process.
        let builtin =
            unsafe { rustrain_abi::Plugin::from_static(rustrain_kernels::plugin(), "<built-in>") }
                .expect("the built-in reference provider is ABI-valid");
        registry
            .add_plugin(builtin)
            .expect("registering the built-in provider");
        registry
    }

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
                &[],
                &[],
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
                &[],
                &[],
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
            &[],
            &[],
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

    /// A single-operand `reshape` carries its operand's distribution on the **same axis
    /// index** across a rank change: a row-major refold keeps the shard's axis where it was,
    /// and a reshape whose output rank cannot hold that axis is refused. The operand's own
    /// requirement is unchanged — no conversion is owed at a view. (`broadcast` is the
    /// opposite: its axes are the output's trailing axes, covered by the tests below.)
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
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            d.outputs[0],
            ParallelLayout::shard(1, g),
            "the reshape keeps the shard's axis index (the rank-2 last axis is the rank-4 \
             axis 1), never renames it"
        );
        assert_eq!(
            d.required_inputs,
            vec![ParallelLayout::shard(1, g)],
            "the view's operand keeps its own layout — no conversion is owed at a view"
        );
    }

    /// The real Qwen3.6 reshape chain, both directions: the column-parallel QK weight shards
    /// `qgw [512, 8192]` on its last axis, and rank `k` holds the flat columns
    /// `[4096k, 4096k+4096)` — which row-major refolding turns into heads `8k..8k+8` of
    /// `qgh [512, 16, 2, 256]` (axis 1, kept by index) and, after the rank-preserving narrow
    /// to `qs [512, 16, 1, 256]`, back into the flat half of `q [512, 4096]` (axis 1 again).
    /// Right-aligning these reshapes would silently move the shard onto the wrong axis.
    #[test]
    fn the_real_qwen36_reshape_chain_keeps_the_shard_axis_index() {
        let g = tp_mask();
        let grow = derive(
            ShardRule::Elementwise,
            "reshape",
            &[ParallelLayout::shard(1, g)],
            &[ParallelLayout::replicate()],
            &[2],
            &[4],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            grow.outputs[0],
            ParallelLayout::shard(1, g),
            "`qgw`'s flat half is the heads axis of `qgh` — axis 1, not the head_dim axis"
        );
        let shrink = derive(
            ShardRule::Elementwise,
            "reshape",
            &[ParallelLayout::shard(1, g)],
            &[ParallelLayout::replicate()],
            &[4],
            &[2],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            shrink.outputs[0],
            ParallelLayout::shard(1, g),
            "`qs`'s heads shard folds back onto `q`'s flat axis — axis 1 again"
        );
    }

    /// **Reviewer C2 (HIGH).** A rank-growing unary view (`broadcast` of a rank-1 `[H]`
    /// sharded `shard(0, tp)` to `[S, H]`) used to copy the operand's layout onto the output
    /// verbatim, renaming the feature axis into the sequence axis. The right-aligned broadcast
    /// mapping applies to the output too: the shard rides the output's trailing axis
    /// (`shard(1, tp)`, local shape `[S, H/2]`), and the operand keeps its own axis.
    #[test]
    fn a_rank_growing_unary_view_maps_the_shard_to_the_output_axis() {
        let g = tp_mask();
        let d = derive(
            ShardRule::Elementwise,
            "broadcast",
            &[ParallelLayout::shard(0, g)],
            &[ParallelLayout::replicate()],
            &[1],
            &[2],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            d.outputs[0],
            ParallelLayout::shard(1, g),
            "the shard rides the output's trailing axis: dim 0 of `[H]` is dim 1 of `[S, H]`"
        );
        assert_eq!(
            d.required_inputs,
            vec![ParallelLayout::shard(0, g)],
            "the operand keeps its own axis — no conversion is owed at the view"
        );
    }

    /// A rank-**shrinking** view with a distributed operand has no expressible output: the
    /// table has no shape algebra to remap a shard onto fewer axes, so it is refused rather
    /// than renamed (the same refusal the walk already produced downstream, now at the rule).
    #[test]
    fn a_rank_shrinking_view_with_a_shard_is_refused() {
        let g = tp_mask();
        let err = derive(
            ShardRule::Elementwise,
            "reshape",
            &[ParallelLayout::shard(1, g)],
            &[ParallelLayout::replicate()],
            &[2],
            &[1],
            &[],
            &[],
        )
        .unwrap_err();
        match err {
            DeriveError::UnmappableViewShard { rank, out_rank, .. } => {
                assert_eq!((rank, out_rank), (2, 1));
            }
            other => panic!("expected an unmappable-view refusal, got {other:?}"),
        }
    }

    /// A shard whose mapped axis does not exist — a scalar operand (rank 0) that somehow
    /// carries a shard — must be refused rather than renamed onto an axis it never named.
    #[test]
    fn a_view_shard_that_cannot_map_to_the_output_is_refused() {
        let g = tp_mask();
        let err = derive(
            ShardRule::Elementwise,
            "broadcast",
            &[ParallelLayout::shard(0, g)],
            &[ParallelLayout::replicate()],
            &[0],
            &[2],
            &[],
            &[],
        )
        .unwrap_err();
        match err {
            DeriveError::UnmappableViewShard { .. } => {}
            other => panic!("expected an unmappable-view refusal, got {other:?}"),
        }
    }

    /// **Reviewer C1 (HIGH).** `a = [B,S,K] shard(0, tp)` (batch axis), `b = [K,N] shard(1,
    /// ep)` (output axis) must produce `{shard(0, tp), shard(2, ep)}`: the contraction axis is
    /// complete on both operands, so nothing is partial, and **both** surviving distributions
    /// belong to the output. The old arms answered with `b`'s shard alone, silently dropping
    /// `a`'s batch shard — `instantiate` then stored a truncated layout and over-claimed the
    /// local batch extent.
    #[test]
    fn a_matmul_keeps_every_surviving_shard_of_both_operands() {
        let tp = tp_mask();
        let ep = GroupMask::single(1).expect("bit 1 always fits");
        let d = derive(
            ShardRule::MatMul,
            "matmul",
            &[ParallelLayout::shard(0, tp), ParallelLayout::shard(1, ep)],
            &[ParallelLayout::replicate()],
            &[3, 2],
            &[3],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            d.outputs[0],
            ParallelLayout {
                dims: vec![ShardSpec::shard(0, tp), ShardSpec::shard(2, ep),],
                partial: None,
            },
            "`a`'s batch shard and `b`'s output shard both survive into the output"
        );
    }

    /// A shard on either contraction dim is the documented partial. When **both** operands
    /// shard the contraction over the *same* group it is one partial; over *different* groups
    /// the output would be a partial over a union of groups the table does not document —
    /// refused, never silently reduced to one group.
    #[test]
    fn a_matmul_contraction_shard_is_a_partial_and_differing_groups_are_refused() {
        let tp = tp_mask();
        let ep = GroupMask::single(1).expect("bit 1 always fits");
        let d = derive(
            ShardRule::MatMul,
            "matmul",
            &[ParallelLayout::shard(1, tp), ParallelLayout::replicate()],
            &[ParallelLayout::replicate()],
            &[2, 2],
            &[1],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            d.outputs[0],
            ParallelLayout::partial(ReduceOp::Sum, tp),
            "a's contraction shard is the documented partial"
        );
        let d = derive(
            ShardRule::MatMul,
            "matmul",
            &[ParallelLayout::shard(1, tp), ParallelLayout::shard(0, tp)],
            &[ParallelLayout::replicate()],
            &[2, 2],
            &[1],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            d.outputs[0],
            ParallelLayout::partial(ReduceOp::Sum, tp),
            "the same group on both contraction dims is one partial"
        );
        let err = derive(
            ShardRule::MatMul,
            "matmul",
            &[ParallelLayout::shard(1, tp), ParallelLayout::shard(0, ep)],
            &[ParallelLayout::replicate()],
            &[2, 2],
            &[1],
            &[],
            &[],
        )
        .unwrap_err();
        match err {
            DeriveError::ContractionGroupsDiffer { .. } => {}
            other => panic!("expected a contraction-groups refusal, got {other:?}"),
        }
    }

    /// A partial or multi-shard operand has no rule in the table; the old arm silently
    /// answered `replicate` for it, dropping the distribution. It is refused now — never
    /// dropped.
    #[test]
    fn a_matmul_partial_or_multi_shard_operand_is_refused() {
        let tp = tp_mask();
        let ep = GroupMask::single(1).expect("bit 1 always fits");
        for bad in [
            ParallelLayout::partial(ReduceOp::Sum, tp),
            ParallelLayout {
                dims: vec![ShardSpec::shard(0, tp), ShardSpec::shard(1, ep)],
                partial: None,
            },
        ] {
            for index in [0, 1] {
                let inputs = if index == 0 {
                    vec![bad.clone(), ParallelLayout::replicate()]
                } else {
                    vec![ParallelLayout::replicate(), bad.clone()]
                };
                let err = derive(
                    ShardRule::MatMul,
                    "matmul",
                    &inputs,
                    &[ParallelLayout::replicate()],
                    &[2, 2],
                    &[1],
                    &[],
                    &[],
                )
                .unwrap_err();
                match err {
                    DeriveError::UnsupportedOperandLayout { index: i, .. } => {
                        assert_eq!(i, index, "the refusal names the operand that carries it");
                    }
                    other => panic!("expected an operand refusal, got {other:?}"),
                }
            }
        }
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
            &[],
            &[],
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
                dims: vec![ShardSpec::shard(0, g), ShardSpec::shard(1, g)],
                partial: None,
            },
            // A partial weight is not a shard at all.
            ParallelLayout::partial(ReduceOp::Sum, g),
            // One shard plus a partial: two facts, no rule.
            ParallelLayout {
                dims: vec![ShardSpec::shard(0, g)],
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
                &[],
                &[],
            )
            .unwrap_err();
            match err {
                DeriveError::UnsupportedWeightLayout { ref layout, .. } => {
                    assert!(
                        layout.contains("shard") || layout.contains("partial"),
                        "the refusal must name the layout: {layout}"
                    );
                }
                other => panic!("expected the weight-layout refusal, got {other:?}"),
            }
        }
    }

    /// A *replicating* weight on a contraction dim has no rule anywhere in the table: the
    /// all-reduce downstream would count every overlapped element twice. `docs/design/qwen36-text/
    /// spec.md` §D6.6 legalizes replication on the key/value heads, which are not contracted — so
    /// the refusal has to hold on every rule that contracts, and the legal case has to stay legal.
    #[test]
    fn a_replicating_weight_on_a_contraction_axis_is_refused() {
        let g = tp_mask();
        let replicating = || ParallelLayout {
            dims: vec![ShardSpec::replicating(0, g, 4)],
            partial: None,
        };
        let column = || ParallelLayout {
            dims: vec![ShardSpec::replicating(1, g, 4)],
            partial: None,
        };
        // Row-parallel linear: the weight's dim 0 is the contraction.
        assert!(matches!(
            derive(
                ShardRule::Linear,
                "linear",
                &[ParallelLayout::replicate(), replicating()],
                &[ParallelLayout::replicate()],
                &[2, 2],
                &[2],
                &[],
                &[],
            ),
            Err(DeriveError::UnsupportedWeightLayout { .. })
        ));
        // The same weight along the *output* dim is the legal column-parallel case.
        let out = derive(
            ShardRule::Linear,
            "linear",
            &[ParallelLayout::replicate(), column()],
            &[ParallelLayout::replicate()],
            &[2, 2],
            &[2],
            &[],
            &[],
        )
        .expect("a replicating output dim needs no collective");
        assert_eq!(
            out.outputs[0].dims,
            vec![ShardSpec::replicating(1, g, 4)],
            "the linear rule carries the weight's output dim verbatim"
        );

        // The embedding table's dim 0 is a contraction too (the vocabulary sum).
        assert!(matches!(
            derive(
                ShardRule::Embedding,
                "embedding",
                &[replicating(), ParallelLayout::replicate()],
                &[ParallelLayout::replicate()],
                &[2, 2],
                &[2],
                &[],
                &[],
            ),
            Err(DeriveError::UnsupportedWeightLayout { .. })
        ));
        assert_eq!(
            derive(
                ShardRule::Embedding,
                "embedding",
                &[column(), ParallelLayout::replicate()],
                &[ParallelLayout::replicate()],
                &[2, 2],
                &[2],
                &[],
                &[],
            )
            .expect("a replicating vocabulary dim is a feature shard")
            .outputs[0]
                .dims,
            // The rule writes `-1`; `canonicalize` stores one spelling per axis.
            vec![ShardSpec::replicating(1, g, 4)]
        );

        // MatMul contracts the first operand's last axis and the second's second-to-last; a
        // replicating shard on either is the same double-count, refused the same way.
        // Operand A contracts its last axis; operand B contracts its second-to-last.
        for (a, b) in [
            (column(), ParallelLayout::replicate()),
            (ParallelLayout::replicate(), replicating()),
        ] {
            assert!(
                matches!(
                    derive(
                        ShardRule::MatMul,
                        "matmul",
                        &[a.clone(), b.clone()],
                        &[ParallelLayout::replicate()],
                        &[2, 2],
                        &[2],
                        &[],
                        &[],
                    ),
                    Err(DeriveError::UnsupportedWeightLayout { .. })
                ),
                "a replicating contraction operand must be refused: {a} / {b}"
            );
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

        let prop = propagate(&plan, &rules()).unwrap();
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

        let err = propagate(&plan, &rules()).unwrap_err();
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
            &[],
            &[],
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

        let err = propagate(&plan, &rules()).unwrap_err();
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
                dims: vec![ShardSpec::shard(0, tp), ShardSpec::shard(1, tp)],
                partial: None,
            },
        );

        let err = propagate(&plan, &rules()).unwrap_err();
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
                dims: vec![ShardSpec::shard(0, tp), ShardSpec::shard(1, ep)],
                partial: None,
            },
        );

        let err = propagate(&plan, &rules()).unwrap_err();
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
                dims: vec![ShardSpec::shard(0, tp)],
                partial: Some(PartialSpec {
                    op: ReduceOp::Sum,
                    group: tp,
                }),
            },
            ParallelLayout::shard(0, tp),
        );

        let err = propagate(&plan, &rules()).unwrap_err();
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
                dims: vec![ShardSpec::shard(0, ep)],
                partial: Some(PartialSpec {
                    op: ReduceOp::Sum,
                    group: tp,
                }),
            },
            ParallelLayout::shard(0, tp),
        );

        let err = propagate(&plan, &rules()).unwrap_err();
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

        let prop = propagate(&plan, &rules()).unwrap();
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

        let prop = propagate(&plan, &rules()).unwrap();
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

        let err = propagate(&plan, &rules()).unwrap_err();
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
        let prop = propagate(&plan, &rules()).unwrap();
        assert_eq!(prop.inserted.len(), 1);
        assert_eq!(prop.inserted[0].op, intrinsic::ALL_REDUCE);
    }

    /// A refold that splits the sharded axis rescales a replicating unit: a shard in units of
    /// 256 elements on `[…, 512]` is a shard in units of one head on `[…, 2, 256]`
    /// (`docs/design/qwen36-text/spec.md` §D6.6) — and a unit the refold would cut in half is
    /// refused rather than rounded.
    #[test]
    fn a_refold_rescales_a_replicating_unit() {
        let tp = tp_mask();
        let heads = ParallelLayout {
            dims: vec![ShardSpec::replicating(1, tp, 256)],
            partial: None,
        };
        let out = carry_to_output_rank(
            &heads,
            2,
            3,
            "reshape",
            0,
            false,
            Some(&[8, 512]),
            Some(&[8, 2, 256]),
        )
        .unwrap();
        assert_eq!(out.dims, vec![ShardSpec::replicating(1, tp, 1)]);

        // A unit of 128 would straddle the refold: refused, never silently rounded to half a head.
        let half = ParallelLayout {
            dims: vec![ShardSpec::replicating(1, tp, 128)],
            partial: None,
        };
        assert!(
            carry_to_output_rank(
                &half,
                2,
                3,
                "reshape",
                0,
                false,
                Some(&[8, 512]),
                Some(&[8, 2, 256]),
            )
            .is_err()
        );

        // A strict shard is untouched either way.
        let strict = ParallelLayout {
            dims: vec![ShardSpec::shard(1, tp)],
            partial: None,
        };
        let out = carry_to_output_rank(
            &strict,
            2,
            3,
            "reshape",
            0,
            false,
            Some(&[8, 512]),
            Some(&[8, 2, 256]),
        )
        .unwrap();
        assert_eq!(out.dims, vec![ShardSpec::shard(1, tp)]);

        // A merge is the other direction: `[…, 2, 256]` into `[…, 512]` folds the operand's axes
        // 2 and 3 into the output's last axis, so a shard on either is no longer an axis slab.
        // Refused, not renamed — a replicating unit of 256 elements of the inner axis is not
        // 256 elements of the merged one.
        let inner = ParallelLayout {
            dims: vec![ShardSpec::replicating(2, tp, 256)],
            partial: None,
        };
        assert!(
            carry_to_output_rank(
                &inner,
                3,
                2,
                "reshape",
                0,
                false,
                Some(&[8, 2, 256]),
                Some(&[8, 512]),
            )
            .is_err(),
            "a unit on a merged axis cannot be re-sliced"
        );
        let outer = ParallelLayout {
            dims: vec![ShardSpec::replicating(0, tp, 8)],
            partial: None,
        };
        let out = carry_to_output_rank(
            &outer,
            3,
            2,
            "reshape",
            0,
            false,
            Some(&[8, 2, 256]),
            Some(&[8, 512]),
        )
        .unwrap();
        assert_eq!(
            out.dims,
            vec![ShardSpec::replicating(0, tp, 8)],
            "a shard the refold does not touch carries over verbatim"
        );

        // A refold whose last piece is not the operand's last axis does not line up: `512 * 128`
        // into `64 * 4 * 4 * 64` splits axis 0 by a hundred, so neither the axis index nor the
        // unit of the shard below survives the rename. Refused for every mode, not just a
        // replicating one — `[512, 128]` into `[256, 256]` is the same lie with `Divide`.
        let misaligned = ParallelLayout {
            dims: vec![ShardSpec::replicating(1, tp, 64)],
            partial: None,
        };
        assert!(
            carry_to_output_rank(
                &misaligned,
                2,
                4,
                "reshape",
                0,
                false,
                Some(&[512, 128]),
                Some(&[64, 4, 4, 64]),
            )
            .is_err(),
            "a refold that does not put the operand's last axis innermost is refused"
        );
        let misaligned_strict = ParallelLayout {
            dims: vec![ShardSpec::shard(1, tp)],
            partial: None,
        };
        assert!(
            carry_to_output_rank(
                &misaligned_strict,
                2,
                4,
                "reshape",
                0,
                false,
                Some(&[512, 128]),
                Some(&[64, 4, 4, 64]),
            )
            .is_err()
        );
    }
}
