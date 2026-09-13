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
use crate::layout::{DimNormalizer, ParallelLayout, ReduceOp, ShardSpec};
use crate::mesh::GroupMask;

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
    AllReduce { group: GroupMask, op: ReduceOp },

    /// Concatenate the group's pieces along `dim`. Turns a shard into a
    /// `Replicate`.
    AllGather { group: GroupMask, dim: i64 },

    /// Reduce and scatter: the group's partial tensors are reduced with `sum`
    /// and the result is split into `degree(group)` pieces along `dim`, one per
    /// rank. Turns a `Partial(Sum)` into a `Shard`.
    ReduceScatter { group: GroupMask, dim: i64 },

    /// Copy one rank's tensor to the rest of `group`.
    ///
    /// Part of the vocabulary because the runtime needs it (weight sync after
    /// an update, pipeline stage handoff), but the transition rules below never
    /// emit it: every conversion they express is an all-reduce, an all-gather
    /// or a reduce-scatter. It is declared here anyway so that the enum the
    /// scheduler matches on is complete from the start rather than growing a
    /// variant later.
    Broadcast {
        group: GroupMask,
        src_group_index: usize,
    },
}

impl Collective {
    /// The process group this collective runs over.
    pub fn group(&self) -> GroupMask {
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
    /// `reduce_scatter(mask(0b1), dim=0)`. Masks render as raw bits; a name
    /// needs the mesh, which a collective does not carry.
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
/// `tests/transitions.rs`. A layout is a *set* of `(dim, group)` shards plus at
/// most one partial, so every rule below generalizes the single-shard table by
/// matching shards pairwise on their normalized dim and their group.
///
/// - `Replicate -> Replicate`: empty. The data is already identical everywhere;
///   any collective here is pure overhead.
/// - `Partial(op, g) -> Replicate`: `all_reduce(op, g)`. The partials are
///   exactly the pieces of a reduction that has not been completed yet, so
///   completing it across the group is the conversion.
/// - `Partial(a, g) -> Partial(b, g)`: empty when the partials are equal;
///   otherwise an error, because a partial is only a reduction of itself. A
///   different op *or* a different group cannot be reached by moving data: the
///   caller must all-reduce to `replicate` (the error says so) and produce the
///   target partial by recomputation.
/// - `Partial(Sum, g) -> Shard(d, g)`: `reduce_scatter(d, g)`. Reduce-scatter
///   is precisely "reduce the partials, then hand every rank its slice", so one
///   collective does both halves. Only `Sum` qualifies: a `Max`/`Min` partial
///   carries no additively-splittable value, so it is an error rather than a
///   guess (the route is `all_reduce` to `replicate`, then a local narrow).
///   With several shards, the scatter runs over the *first* target shard whose
///   group is the partial's group; the others are local narrows of the reduced
///   tensor.
/// - `Shard(d, g) -> Replicate`: `all_gather(d, g)`. Every rank is missing
///   exactly the other ranks' slices. Several shards mean several gathers — one
///   per shard, in declaration order.
/// - `Replicate -> Shard(d, g)`: **empty**. This is the rule people get wrong.
///   Every rank already holds a complete copy, so the conversion is a *local*
///   one: each rank keeps the slice its group index selects and drops the rest.
///   Inserting an all-gather (or a reduce-scatter) here costs a full tensor of
///   bandwidth and changes nothing about the result. The framework never needs
///   to communicate to *narrow* a tensor, only to widen one.
/// - `Shard(d1, g) -> Shard(d2, g)`: empty when the `(dim, group)` sets agree;
///   otherwise `all_gather(d1, g)` for every source shard the target does not
///   have. The pieces to reassemble live along the source dim, so that is the
///   axis to gather; once the tensor is complete along it, re-slicing along the
///   target dims is local — the same `Replicate -> Shard` argument one step
///   later.
/// - `Shard -> Partial`: error. A shard is a *disjoint* piece, so the ranks
///   cannot reinterpret their pieces as partial sums of one another; no
///   collective produces a partial from a shard. It requires recomputation (or
///   an all-gather to `replicate` first, which is a different tensor).
/// - **A group change is not a blanket error anymore** — but `Shard(d, g1) ->
///   Shard(d, g2)` with `g1 != g2` still is one, because the route through
///   `Replicate` is the legal one: `Shard(d, g1) -> Replicate -> Shard(d, g2)`
///   is two steps, written explicitly by the caller (the plan compiler refuses
///   multi-step conversions, so the intermediate `replicate` layout has to
///   appear in the plan). The only exception is a shard produced by a
///   `Partial(Sum, g)` conversion (the reduce-scatter rule above), which never
///   crosses an existing shard's group.
/// - `Replicate -> Partial(..)`: error. A complete copy is not a partial sum;
///   the partial has to come from the computation.
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
    let from_shards = normalize_shards(&from.dims, norm)?;
    let to_shards = normalize_shards(&to.dims, norm)?;

    let collectives = match (
        from.is_replicated(),
        to.is_replicated(),
        from.partial.as_ref(),
        to.partial.as_ref(),
    ) {
        // Replicated already.
        (true, true, None, None) => Vec::new(),

        // A replica is not a partial sum.
        (true, _, _, Some(_)) => {
            return Err(ShardError::ReplicateToPartial {
                from: from.clone(),
                to: to.clone(),
            });
        }

        // `Replicate -> Shard(..)` is a local narrow: every rank holds the
        // whole tensor and keeps the slice its group index selects.
        // Communicating here would be pure overhead.
        (true, false, None, None) => Vec::new(),

        // Anything -> Replicate widens: gather what the rank is missing.
        (false, true, _, None) => into_replicate(from, &from_shards),

        (_, _, _, _) => two_sided(from, to, &from_shards, &to_shards)?,
    };

    if !collectives.is_empty() {
        tracing::trace!(%from, %to, ?collectives, "layout transition");
    }
    Ok(collectives)
}

/// Resolves the dims of every shard. Errors (rather than passing the raw dim
/// through) so that every rule below works on non-negative axes and cannot
/// emit a collective with a negative dim.
fn normalize_shards(
    shards: &[ShardSpec],
    norm: &DimNormalizer,
) -> Result<Vec<(i64, GroupMask)>, ShardError> {
    shards
        .iter()
        .map(|spec| Ok((norm.normalize(spec.dim)?, spec.group)))
        .collect()
}

/// `X -> Replicate` for an `X` whose dims are already resolved. The partial is
/// completed first (its pieces are the unfinished reduction), then every shard
/// is gathered, in declaration order.
fn into_replicate(from: &ParallelLayout, from_shards: &[(i64, GroupMask)]) -> Vec<Collective> {
    let mut collectives =
        Vec::with_capacity(from_shards.len() + usize::from(from.partial.is_some()));
    if let Some(partial) = &from.partial {
        collectives.push(Collective::AllReduce {
            group: partial.group,
            op: partial.op,
        });
    }
    for &(dim, group) in from_shards {
        collectives.push(Collective::AllGather { group, dim });
    }
    collectives
}

/// Conversions between two non-replicated layouts.
///
/// `from`/`to` are the caller's originals (for error messages); the shard
/// lists are the same layouts with dims resolved (for the rules).
fn two_sided(
    from: &ParallelLayout,
    to: &ParallelLayout,
    from_shards: &[(i64, GroupMask)],
    to_shards: &[(i64, GroupMask)],
) -> Result<Vec<Collective>, ShardError> {
    // Neither side is Replicate by the time we get here. A partial is only a
    // reduction of itself: same op and same group, or an error.
    match (&from.partial, &to.partial) {
        (Some(a), Some(b)) if a != b => {
            return Err(ShardError::PartialOpMismatch {
                from: from.clone(),
                to: to.clone(),
            });
        }
        // A disjoint piece is not a partial sum.
        (None, Some(_)) => {
            return Err(ShardError::ShardToPartial {
                from: from.clone(),
                to: to.clone(),
            });
        }
        _ => {}
    }

    // Every target shard the source does not already have must be reachable by
    // a local narrow — the source must not shard that dim over any other
    // group, or the change is a group crossing, which the caller has to make
    // explicit through a `replicate` layout. The one exception is a target
    // shard whose group is the source partial's group: that shard is produced
    // by the partial conversion (reduce_scatter), not by re-interpreting an
    // existing shard.
    let partial_group = from.partial.as_ref().map(|p| p.group);
    for &(dim, group) in to_shards {
        if from_shards.contains(&(dim, group)) {
            continue;
        }
        let crosses = from_shards.iter().any(|&(d, g)| d == dim && g != group);
        if !crosses {
            continue;
        }
        if partial_group == Some(group) && to.partial.is_none() {
            if from.partial.as_ref().map(|p| p.op) == Some(ReduceOp::Sum) {
                // The reduce_scatter below handles it.
                continue;
            }
            return Err(ShardError::ReduceScatterRequiresSum {
                from: from.clone(),
                to: to.clone(),
            });
        }
        return Err(ShardError::GroupMismatch {
            from: from.clone(),
            to: to.clone(),
        });
    }

    let mut collectives = Vec::new();

    // Complete the partial. If a target shard carries the partial's group, the
    // completion and the sharding are one reduce_scatter (Sum only — Max/Min
    // partials carry no additively-splittable value); otherwise the completion
    // is a plain all_reduce and the target shards are local narrows of the
    // reduced tensor.
    if let Some(partial) = &from.partial {
        if to.partial.is_none() {
            let scatter = to_shards.iter().find(|&&(dim, group)| {
                group == partial.group && !from_shards.contains(&(dim, group))
            });
            match scatter {
                Some(&(dim, _)) => {
                    if partial.op != ReduceOp::Sum {
                        return Err(ShardError::ReduceScatterRequiresSum {
                            from: from.clone(),
                            to: to.clone(),
                        });
                    }
                    collectives.push(Collective::ReduceScatter {
                        group: partial.group,
                        dim,
                    });
                }
                None => collectives.push(Collective::AllReduce {
                    group: partial.group,
                    op: partial.op,
                }),
            }
        }
    }

    // Re-slicing onto a target the source shards: gather along the axis the
    // pieces currently live on, in declaration order; the local re-slice along
    // the target dims is not a collective.
    for &(dim, group) in from_shards {
        if !to_shards.contains(&(dim, group)) {
            collectives.push(Collective::AllGather { group, dim });
        }
    }

    Ok(collectives)
}
