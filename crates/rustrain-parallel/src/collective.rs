//! The collective vocabulary and the layout transition rules.
//!
//! [`transitions`] is the heart of this crate: given the layout a tensor
//! currently has and the layout the consumer needs, it returns the minimal
//! ordered sequence of collectives in between. The plan compiler inserts those
//! nodes; nothing else in the framework decides where communication happens
//! (invariant I-3).

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::ShardError;
use crate::group::GroupKind;
use crate::layout::{DimNormalizer, EXPERT_DIM, ParallelLayout, ReduceOp};

/// One communication step between two layouts.
///
/// Dims carried here are always resolved (non-negative) axes: the runtime
/// indexes a real tensor with them, and [`DimNormalizer`] is the only place a
/// negative dim exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Collective {
    /// Reduce the same-shaped tensors of every rank in `group` and leave the
    /// result on all of them. Turns a `Partial(op)` into a `Replicate`.
    AllReduce { group: GroupKind, op: ReduceOp },

    /// Concatenate the group's pieces along `dim`. Turns a shard into a
    /// `Replicate`.
    AllGather { group: GroupKind, dim: i64 },

    /// Reduce and scatter: the group's partial tensors are reduced with `sum`
    /// and the result is split into `group_size()` pieces along `dim`, one per
    /// rank. Turns a `Partial(Sum)` into a `Shard`.
    ReduceScatter { group: GroupKind, dim: i64 },

    /// Copy one rank's tensor to the rest of `group`.
    ///
    /// Part of the vocabulary because the runtime needs it (weight sync after
    /// an update, pipeline stage handoff), but the transition rules below never
    /// emit it: every conversion they express is an all-reduce, an all-gather
    /// or a reduce-scatter. It is declared here anyway so that the enum the
    /// scheduler matches on is complete from the start rather than growing a
    /// variant later.
    Broadcast {
        group: GroupKind,
        src_group_index: usize,
    },
}

impl Collective {
    /// The process group this collective runs over.
    pub fn group(&self) -> GroupKind {
        match self {
            Collective::AllReduce { group, .. }
            | Collective::AllGather { group, .. }
            | Collective::ReduceScatter { group, .. }
            | Collective::Broadcast { group, .. } => *group,
        }
    }
}

impl fmt::Display for Collective {
    /// Compact form for plan dumps and error messages, e.g.
    /// `reduce_scatter(tp, dim=0)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Collective::AllReduce { group, op } => write!(f, "all_reduce({op}, {group})"),
            Collective::AllGather { group, dim } => write!(f, "all_gather({group}, dim={dim})"),
            Collective::ReduceScatter { group, dim } => {
                write!(f, "reduce_scatter({group}, dim={dim})")
            }
            Collective::Broadcast {
                group,
                src_group_index,
            } => write!(f, "broadcast({group}, src={src_group_index})"),
        }
    }
}

/// The minimal ordered sequence of collectives that converts a tensor held in
/// layout `from` into layout `to`.
///
/// `norm` resolves the logical (possibly negative) dims of both layouts against
/// the rank of the tensor being converted.
///
/// # Rules
///
/// Each rule states its reasoning; each one has a test in
/// `tests/transitions.rs`.
///
/// - `Replicate -> Replicate`: empty. The data is already identical everywhere;
///   any collective here is pure overhead.
/// - `Partial(op, g) -> Replicate`: `all_reduce(op, g)`. The partials are
///   exactly the pieces of a reduction that has not been completed yet, so
///   completing it across the group is the conversion.
/// - `Partial(a, g) -> Partial(b, g)`: empty when `a == b`; otherwise an error,
///   because a partial is only a reduction of itself. `Max` partials cannot
///   become `Sum` partials by moving data: the caller must all-reduce to
///   `replicate` (the error says so) and produce the target partial by
///   recomputation.
/// - `Partial(Sum, g) -> Shard(d, g)`: `reduce_scatter(d, g)`. Reduce-scatter is
///   precisely "reduce the partials, then hand every rank its slice", so one
///   collective does both halves. Only `Sum` qualifies: a `Max`/`Min` partial
///   carries no additively-splittable value, so it is an error rather than a
///   guess.
/// - `Shard(d, g) -> Replicate`: `all_gather(d, g)`. Every rank is missing
///   exactly the other ranks' slices.
/// - `Replicate -> Shard(d, g)`: **empty**. This is the rule people get wrong.
///   Every rank already holds a complete copy, so the conversion is a *local*
///   one: each rank keeps the slice its group index selects and drops the rest.
///   Inserting an all-gather (or a reduce-scatter) here costs a full tensor of
///   bandwidth and changes nothing about the result. The framework never needs
///   to communicate to *narrow* a tensor, only to widen one.
/// - `Shard(d1, g) -> Shard(d2, g)`: empty when `d1 == d2` after normalization;
///   otherwise `all_gather(d1, g)` followed by a local slice along `d2`. The
///   pieces to reassemble live along `d1`, so that is the axis to gather; once
///   the tensor is complete, re-slicing along `d2` is local — the same
///   `Replicate -> Shard` argument one step later.
/// - `Shard(d, g) -> Partial(Sum, g)`: error. A shard is a *disjoint* piece, so
///   the ranks cannot reinterpret their pieces as partial sums of one another;
///   no collective produces a partial from a shard. It requires recomputation
///   (or an all-gather to `replicate` first, which is a different tensor).
/// - `ExpertShard(g) -> Replicate`: `all_gather(dim 0, g)`. An expert-parallel
///   weight tensor is `[num_experts, ...]` per rank, so gathering the full
///   expert set means concatenating along dim 0; the reverse
///   (`Replicate -> ExpertShard`) is empty for the same reason as
///   `Replicate -> Shard`.
/// - `SequenceShard(g) -> Replicate`: `all_gather(seq_dim, g)`, the reverse is
///   empty. The sequence axis comes from [`DimNormalizer::sequence_dim`], since
///   the layout names the kind of sharding, not an index.
/// - Different groups (`g1 != g2`), with neither side `Replicate`: error. A
///   layout transition is defined inside one process group; crossing groups is
///   a redistribution this crate cannot express as a single collective, and
///   guessing one would silently produce a wrong tensor. The route through
///   `Replicate` always works (the error says so) and is explicit about its
///   cost.
/// - `Replicate -> Partial(..)`: error. A complete copy is not a partial sum;
///   the partial has to come from the computation.
///
/// # Summary
///
/// ```text
/// from \ to          Replicate        Shard(d2, g)    Partial(o2, g)   Expert/Sequence
/// Replicate          –                local           error            local
/// Shard(d1, g)       all_gather(d1)   gather if d1≠d2 error            error
/// Partial(o1, g)     all_reduce(o1)   reduce_scatter¹ error if o1≠o2   error
/// ExpertShard(g)     all_gather(0)    error           error            error
/// SequenceShard(g)   all_gather(seq)  error           error            error
/// ```
///
/// ¹ only for `Sum`; `Max`/`Min` partials are rejected.
///
/// Pairs whose groups differ are rejected wherever neither side is
/// `Replicate`, and every pair without a rule above is rejected as
/// [`ShardError::UnsupportedTransition`] rather than approximated — in
/// particular the `ExpertShard`/`SequenceShard` ↔ `Shard` spellings are *not*
/// assumed to be the same layout, even when the dims happen to match. Going
/// through `Replicate` is always available and always explicit.
///
/// # Errors
///
/// [`ShardError`], naming both layouts: an out-of-range dim
/// ([`ShardError::DimOutOfRange`]), a pair of groups that do not match
/// ([`ShardError::GroupMismatch`]), or one of the conversions that no
/// collective can perform.
pub fn transitions(
    from: &ParallelLayout,
    to: &ParallelLayout,
    norm: &DimNormalizer,
) -> Result<Vec<Collective>, ShardError> {
    // Resolve every logical dim in both operands before matching on the rules.
    // Doing it up front (rather than only where a dim is used by an emitted
    // collective) means a plan that names an axis the tensor does not have is
    // reported even when the conversion turns out to be local — `Replicate ->
    // Shard(9, tp)` on a rank-4 tensor is a bug regardless of the fact that it
    // needs no communication.
    let from_n = normalize_layout(from, norm)?;
    let to_n = normalize_layout(to, norm)?;

    let collectives = match (from_n, to_n) {
        // Replicated already.
        (ParallelLayout::Replicate, ParallelLayout::Replicate) => Vec::new(),

        // A replica is not a partial sum.
        (ParallelLayout::Replicate, ParallelLayout::Partial { .. }) => {
            return Err(ShardError::ReplicateToPartial {
                from: *from,
                to: *to,
            });
        }

        // `Replicate -> Shard/ExpertShard/SequenceShard` is a local narrow:
        // every rank holds the whole tensor and keeps the slice its group index
        // selects. Communicating here would be pure overhead.
        (ParallelLayout::Replicate, _) => Vec::new(),

        // Anything -> Replicate widens: gather what the rank is missing.
        (_, ParallelLayout::Replicate) => into_replicate(&from_n, norm)?,

        (_, _) => two_sided(*from, *to, from_n, to_n)?,
    };

    if !collectives.is_empty() {
        tracing::trace!(%from, %to, ?collectives, "layout transition");
    }
    Ok(collectives)
}

/// Rewrites a layout with its logical dims resolved.
///
/// Errors (rather than passing the raw dim through) so that every rule below
/// works on non-negative axes and cannot emit a collective with a negative dim.
///
/// This also checks the sequence axis of a [`ParallelLayout::SequenceShard`],
/// which carries no dim of its own: the layout still *claims* the tensor has a
/// sequence axis, so a rank-1 tensor cannot be sequence-sharded. Validating it
/// here keeps the rule uniform — an axis a layout names is checked whether or
/// not the conversion happens to move data.
fn normalize_layout(
    layout: &ParallelLayout,
    norm: &DimNormalizer,
) -> Result<ParallelLayout, ShardError> {
    Ok(match layout {
        ParallelLayout::Shard { dim, group } => ParallelLayout::Shard {
            dim: norm.normalize(*dim)?,
            group: *group,
        },
        ParallelLayout::SequenceShard { .. } => {
            norm.sequence_dim()?;
            *layout
        }
        other => *other,
    })
}

/// `X -> Replicate` for an `X` whose dims are already resolved.
fn into_replicate(
    from: &ParallelLayout,
    norm: &DimNormalizer,
) -> Result<Vec<Collective>, ShardError> {
    Ok(match from {
        // Unreachable through `transitions` (the caller matches Replicate
        // first); kept so that this function is total.
        ParallelLayout::Replicate => Vec::new(),
        // Each rank is missing exactly the other ranks' slices.
        ParallelLayout::Shard { dim, group } => vec![Collective::AllGather {
            group: *group,
            dim: *dim,
        }],
        // The partials are an unfinished reduction; complete it.
        ParallelLayout::Partial { op, group } => vec![Collective::AllReduce {
            group: *group,
            op: *op,
        }],
        // Expert-parallel weights are `[num_experts, ...]` per rank.
        ParallelLayout::ExpertShard { group } => vec![Collective::AllGather {
            group: *group,
            dim: EXPERT_DIM,
        }],
        // The sequence axis is a convention, so it comes from the normalizer.
        ParallelLayout::SequenceShard { group } => vec![Collective::AllGather {
            group: *group,
            dim: norm.sequence_dim()?,
        }],
    })
}

/// Conversions between two non-replicated layouts.
///
/// `from`/`to` are the caller's originals (for error messages); `from_n`/`to_n`
/// are the same layouts with dims resolved (for the rules).
fn two_sided(
    from: ParallelLayout,
    to: ParallelLayout,
    from_n: ParallelLayout,
    to_n: ParallelLayout,
) -> Result<Vec<Collective>, ShardError> {
    // Neither side is Replicate by the time we get here, so both name a group.
    // The conversion has to happen inside one group: `all_gather(tp)` cannot
    // produce a tensor that is sharded over `cp`.
    let group = match (from_n.group(), to_n.group()) {
        (Some(a), Some(b)) if a == b => a,
        _ => return Err(ShardError::GroupMismatch { from, to }),
    };

    match (from_n, to_n) {
        // Same group, same layout (Shard dims are resolved by now, so `-1` and
        // `3` are the same axis on a rank-4 tensor): nothing to move.
        (a, b) if a == b => Ok(Vec::new()),

        // Re-slicing along another axis: gather along the axis the pieces
        // currently live on, then slice locally.
        (ParallelLayout::Shard { dim, .. }, ParallelLayout::Shard { .. }) => {
            Ok(vec![Collective::AllGather { group, dim }])
        }

        // Reduce and scatter in one step: exactly what a partial-sum -> shard
        // conversion is.
        (
            ParallelLayout::Partial {
                op: ReduceOp::Sum, ..
            },
            ParallelLayout::Shard { dim, .. },
        ) => Ok(vec![Collective::ReduceScatter { group, dim }]),

        // Max/Min partials are not additively splittable.
        (ParallelLayout::Partial { .. }, ParallelLayout::Shard { .. }) => {
            Err(ShardError::ReduceScatterRequiresSum { from, to })
        }

        // Reached only when the ops differ: equal layouts returned above.
        (ParallelLayout::Partial { .. }, ParallelLayout::Partial { .. }) => {
            Err(ShardError::PartialOpMismatch { from, to })
        }

        // A disjoint piece is not a partial sum.
        (ParallelLayout::Shard { .. }, ParallelLayout::Partial { .. }) => {
            Err(ShardError::ShardToPartial { from, to })
        }

        // No rule: refuse instead of guessing.
        _ => Err(ShardError::UnsupportedTransition { from, to }),
    }
}
