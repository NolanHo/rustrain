//! One test per transition rule, plus the error paths and a property check.
//!
//! The rule table is documented on `rustrain_parallel::transitions`; the test
//! names here mirror the rule order so that a missing rule is easy to spot.
//! Masks are written as raw bits over the canonical five-axis mesh
//! `[tp, cp, ep, dp, pp]`, so `0b1` = tp, `0b10` = cp, `0b100` = ep,
//! `0b1000` = dp, `0b10000` = pp.

use rustrain_parallel::{
    Collective, DimNormalizer, GroupMask, Mesh, ParallelConfig, ParallelLayout, ReduceOp,
    ShardError, transitions,
};

const TP: GroupMask = GroupMask::from_bits(0b00001);
const CP: GroupMask = GroupMask::from_bits(0b00010);
const EP: GroupMask = GroupMask::from_bits(0b00100);
const DP: GroupMask = GroupMask::from_bits(0b01000);
const PP: GroupMask = GroupMask::from_bits(0b10000);
const ALL: GroupMask = GroupMask::from_bits(0b11111);
const NONE: GroupMask = GroupMask::NONE;

fn shard(dim: i64, group: GroupMask) -> ParallelLayout {
    ParallelLayout::shard(dim, group)
}

fn partial(op: ReduceOp, group: GroupMask) -> ParallelLayout {
    ParallelLayout::partial(op, group)
}

fn layout(dims: &[(i64, GroupMask)], partial: Option<(ReduceOp, GroupMask)>) -> ParallelLayout {
    ParallelLayout {
        dims: dims
            .iter()
            .map(|&(dim, group)| rustrain_parallel::ShardSpec { dim, group })
            .collect(),
        partial: partial.map(|(op, group)| rustrain_parallel::PartialSpec { op, group }),
    }
}

fn replicate() -> ParallelLayout {
    ParallelLayout::replicate()
}

/// Rank-4 tensors (`[batch, seq, head, dim]`) unless a test says otherwise.
fn norm() -> DimNormalizer {
    DimNormalizer::new(4).unwrap()
}

fn all_reduce(op: ReduceOp, group: GroupMask) -> Collective {
    Collective::AllReduce { group, op }
}

fn all_gather(dim: i64, group: GroupMask) -> Collective {
    Collective::AllGather { group, dim }
}

fn reduce_scatter(dim: i64, group: GroupMask) -> Collective {
    Collective::ReduceScatter { group, dim }
}

// ---------------------------------------------------------------- happy rules

#[test]
fn replicate_to_replicate_is_empty() {
    assert_eq!(
        transitions(&replicate(), &replicate(), &norm()),
        Ok(Vec::new())
    );
    // Even for a scalar: there is nothing to move.
    let scalar = DimNormalizer::new(0).unwrap();
    assert_eq!(
        transitions(&replicate(), &replicate(), &scalar),
        Ok(Vec::new())
    );
}

#[test]
fn partial_to_replicate_all_reduces() {
    for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
        assert_eq!(
            transitions(&partial(op, TP), &replicate(), &norm()),
            Ok(vec![all_reduce(op, TP)]),
            "partial({op}, tp)"
        );
    }
    // A combined mask is a real group, not a wildcard.
    assert_eq!(
        transitions(&partial(ReduceOp::Sum, TP.union(EP)), &replicate(), &norm()),
        Ok(vec![all_reduce(ReduceOp::Sum, TP.union(EP))])
    );
}

#[test]
fn partial_to_partial_with_the_same_reduction_is_empty() {
    assert_eq!(
        transitions(
            &partial(ReduceOp::Sum, TP),
            &partial(ReduceOp::Sum, TP),
            &norm()
        ),
        Ok(Vec::new())
    );
    assert_eq!(
        transitions(
            &partial(ReduceOp::Max, ALL),
            &partial(ReduceOp::Max, ALL),
            &norm()
        ),
        Ok(Vec::new())
    );
}

#[test]
fn partial_to_partial_with_a_different_reduction_is_an_error() {
    let from = partial(ReduceOp::Max, TP);
    let to = partial(ReduceOp::Sum, TP);
    let err = transitions(&from, &to, &norm()).unwrap_err();
    assert_eq!(err, ShardError::PartialOpMismatch { from, to });
    // The reason (a partial is only a reduction of itself) and the way out
    // (round trip through Replicate) are both in the message.
    let message = err.to_string();
    assert!(message.contains("partial(max"), "{message}");
    assert!(message.contains("partial(sum"), "{message}");
    assert!(message.contains("replicate"), "{message}");

    // Same op on a different group is also "not the same reduction": the
    // source partial cannot be re-targeted by moving data.
    let from = partial(ReduceOp::Sum, TP);
    let to = partial(ReduceOp::Sum, EP);
    let err = transitions(&from, &to, &norm()).unwrap_err();
    assert_eq!(err, ShardError::PartialOpMismatch { from, to });
    assert!(err.to_string().contains("replicate"), "{err}");
}

#[test]
fn partial_sum_to_shard_reduce_scatters() {
    // The dim in the emitted collective is the resolved one: -1 on a rank-4
    // tensor is axis 3.
    assert_eq!(
        transitions(&partial(ReduceOp::Sum, TP), &shard(-1, TP), &norm()),
        Ok(vec![reduce_scatter(3, TP)])
    );
    assert_eq!(
        transitions(&partial(ReduceOp::Sum, DP), &shard(1, DP), &norm()),
        Ok(vec![reduce_scatter(1, DP)])
    );
}

#[test]
fn non_sum_partial_to_shard_is_an_error() {
    let from = partial(ReduceOp::Max, TP);
    let to = shard(-1, TP);
    let err = transitions(&from, &to, &norm()).unwrap_err();
    assert_eq!(err, ShardError::ReduceScatterRequiresSum { from, to });
    let message = err.to_string();
    assert!(message.contains("reduce_scatter"), "{message}");
    assert!(message.contains("sum"), "{message}");
}

#[test]
fn shard_to_replicate_all_gathers() {
    assert_eq!(
        transitions(&shard(-1, TP), &replicate(), &norm()),
        Ok(vec![all_gather(3, TP)])
    );
    assert_eq!(
        transitions(&shard(0, CP), &replicate(), &norm()),
        Ok(vec![all_gather(0, CP)])
    );
    // Gathering the world's shards is a different collective from gathering a
    // TP group's shards.
    assert_eq!(
        transitions(&shard(2, ALL), &replicate(), &norm()),
        Ok(vec![all_gather(2, ALL)])
    );
}

/// The rule people get wrong: a replica already holds everything, so narrowing
/// it to a shard is a *local* slice. No collective, whatever the dim or group.
#[test]
fn replicate_to_shard_is_local() {
    for to in [
        shard(0, TP),
        shard(-1, TP),
        shard(2, CP),
        shard(0, ALL),
        shard(1, NONE),
    ] {
        assert_eq!(
            transitions(&replicate(), &to, &norm()),
            Ok(Vec::new()),
            "replicate -> {to} must not communicate"
        );
    }
}

#[test]
fn shard_to_same_shard_is_empty() {
    assert_eq!(
        transitions(&shard(0, TP), &shard(0, TP), &norm()),
        Ok(Vec::new())
    );
    // -1 and 3 are the same axis on a rank-4 tensor, so this is the same
    // conversion spelled two ways.
    assert_eq!(
        transitions(&shard(-1, TP), &shard(3, TP), &norm()),
        Ok(Vec::new())
    );
}

#[test]
fn shard_to_other_shard_gathers_the_source_axis() {
    // The pieces live along the source dim, so that is what gets gathered; the
    // local re-slice along the target dim is not a collective.
    assert_eq!(
        transitions(&shard(3, TP), &shard(0, TP), &norm()),
        Ok(vec![all_gather(3, TP)])
    );
    assert_eq!(
        transitions(&shard(-2, CP), &shard(1, CP), &norm()),
        Ok(vec![all_gather(2, CP)])
    );
}

#[test]
fn shard_to_partial_is_an_error() {
    let cases = [
        (shard(0, TP), partial(ReduceOp::Sum, TP)),
        (shard(-1, DP), partial(ReduceOp::Max, DP)),
        // A shard plus a partial on the target side is still "shard -> partial".
        (shard(0, TP), layout(&[(0, TP)], Some((ReduceOp::Sum, EP)))),
    ];
    for (from, to) in cases {
        let err = transitions(&from, &to, &norm()).unwrap_err();
        assert_eq!(err, ShardError::ShardToPartial { from, to });
        let message = err.to_string();
        assert!(message.contains("recomputation"), "{message}");
    }
}

// ------------------------------------------------------------ multi-shard rules

/// The D3 example: `{shard(0, ep), shard(1, tp)} -> Replicate` is two
/// all-gathers, one per shard, in declaration order.
#[test]
fn multi_shard_to_replicate_gathers_every_shard() {
    let from = layout(&[(0, EP), (1, TP)], None);
    assert_eq!(
        transitions(&from, &replicate(), &norm()),
        Ok(vec![all_gather(0, EP), all_gather(1, TP)])
    );
}

#[test]
fn multi_shard_to_a_subset_gathers_only_the_missing_shard() {
    let from = layout(&[(0, EP), (1, TP)], None);
    let to = layout(&[(0, EP)], None);
    assert_eq!(
        transitions(&from, &to, &norm()),
        Ok(vec![all_gather(1, TP)])
    );
}

#[test]
fn adding_a_shard_is_local() {
    // Replicate -> shard is a local narrow; so is "add another independent
    // shard on an axis the source does not shard".
    let from = layout(&[(0, EP)], None);
    let to = layout(&[(0, EP), (1, TP)], None);
    assert_eq!(transitions(&from, &to, &norm()), Ok(Vec::new()));
}

#[test]
fn a_sharded_partial_to_replicate_reduces_then_gathers() {
    let from = layout(&[(0, EP)], Some((ReduceOp::Sum, TP)));
    assert_eq!(
        transitions(&from, &replicate(), &norm()),
        Ok(vec![all_reduce(ReduceOp::Sum, TP), all_gather(0, EP)])
    );
}

#[test]
fn partial_sum_to_multi_shard_scatters_on_the_partial_group() {
    // The reduce_scatter rides the target shard whose group is the partial's
    // group; the other shards are local narrows of the reduced tensor.
    let from = partial(ReduceOp::Sum, TP);
    let to = layout(&[(1, EP), (0, TP)], None);
    assert_eq!(
        transitions(&from, &to, &norm()),
        Ok(vec![reduce_scatter(0, TP)])
    );
}

/// The interesting combined case: a tensor that is `ep`-sharded and `tp`-
/// partial becomes `ep`-sharded *and* `tp`-sharded. The partial's completion
/// and the new shard are one reduce_scatter; the ep shard is untouched.
#[test]
fn partial_shard_combo_completes_with_reduce_scatter() {
    let from = layout(&[(0, EP)], Some((ReduceOp::Sum, TP)));
    let to = layout(&[(0, EP), (0, TP)], None);
    assert_eq!(
        transitions(&from, &to, &norm()),
        Ok(vec![reduce_scatter(0, TP)])
    );
}

#[test]
fn a_partial_completed_without_a_scatter_all_reduces() {
    // The target keeps the shard the source already has; only the partial
    // disappears, so it is a plain all_reduce (any op qualifies).
    let from = layout(&[(0, TP)], Some((ReduceOp::Max, TP)));
    let to = layout(&[(0, TP)], None);
    assert_eq!(
        transitions(&from, &to, &norm()),
        Ok(vec![all_reduce(ReduceOp::Max, TP)])
    );
    // Target sharded over a *different* group than the partial: the shards are
    // local narrows of the reduced tensor, so an all_reduce suffices.
    let from = partial(ReduceOp::Max, TP);
    let to = layout(&[(0, EP)], None);
    assert_eq!(
        transitions(&from, &to, &norm()),
        Ok(vec![all_reduce(ReduceOp::Max, TP)])
    );
}

#[test]
fn a_partial_kept_plus_an_extra_shard_is_local() {
    let from = partial(ReduceOp::Sum, TP);
    let to = layout(&[(0, TP)], Some((ReduceOp::Sum, TP)));
    assert_eq!(transitions(&from, &to, &norm()), Ok(Vec::new()));
}

// --------------------------------------------------------------- error paths

#[test]
fn shard_group_changes_are_an_error() {
    let cases = [
        (shard(0, TP), shard(0, CP)),
        (shard(0, CP), shard(0, TP)),
        (shard(0, TP), shard(0, ALL)),
        // Nested groups count as a change too: the route through `replicate`
        // is the legal one for *every* group crossing.
        (shard(0, TP), shard(0, TP.union(EP))),
        (shard(0, TP.union(EP)), shard(0, TP)),
        // A multi-shard layout crosses a group on one of its dims.
        (shard(0, TP), layout(&[(0, CP), (1, DP)], None)),
        (
            layout(&[(0, EP)], Some((ReduceOp::Sum, DP))),
            layout(&[(0, TP)], None),
        ),
    ];
    for (from, to) in cases {
        let err = transitions(&from, &to, &norm()).unwrap_err();
        assert_eq!(
            err,
            ShardError::GroupMismatch {
                from: from.clone(),
                to: to.clone()
            },
            "{from} -> {to}"
        );
        // The error has to say that the Replicate route exists, because that is
        // the only correct way forward.
        assert!(err.to_string().contains("replicate"), "{err}");
    }
}

/// The group change is *not* an error when the caller writes the intermediate
/// `replicate` layout explicitly — that is the legal route, two steps, each a
/// single-step conversion.
#[test]
fn a_group_change_through_replicate_is_two_explicit_steps() {
    assert_eq!(
        transitions(&shard(0, TP), &replicate(), &norm()),
        Ok(vec![all_gather(0, TP)])
    );
    assert_eq!(
        transitions(&replicate(), &shard(0, EP), &norm()),
        Ok(Vec::new())
    );
}

/// The same group crossing is *not* an error when one side is a replica: the
/// single-sided rules already cover it, and the group of the other side decides
/// the collective.
#[test]
fn a_replica_makes_the_other_side_s_group_irrelevant() {
    assert_eq!(
        transitions(&shard(0, TP), &replicate(), &norm()),
        Ok(vec![all_gather(0, TP)])
    );
    assert_eq!(
        transitions(&replicate(), &shard(0, CP), &norm()),
        Ok(Vec::new())
    );
}

#[test]
fn replicate_to_partial_is_an_error() {
    for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
        let to = partial(op, TP);
        let err = transitions(&replicate(), &to, &norm()).unwrap_err();
        assert_eq!(
            err,
            ShardError::ReplicateToPartial {
                from: replicate(),
                to
            }
        );
        assert!(err.to_string().contains("computation"), "{err}");
    }
    // Replicate -> (shards + partial) is refused for the same reason: the
    // shards would be local narrows, but the partial cannot appear.
    let to = layout(&[(0, TP)], Some((ReduceOp::Sum, TP)));
    let err = transitions(&replicate(), &to, &norm()).unwrap_err();
    assert_eq!(
        err,
        ShardError::ReplicateToPartial {
            from: replicate(),
            to
        }
    );
}

#[test]
fn dims_are_validated_even_when_the_conversion_is_local() {
    // Replicate -> Shard needs no communication, but the axis still has to
    // exist: a plan that names axis 9 of a rank-4 tensor is broken either way.
    assert_eq!(
        transitions(&replicate(), &shard(9, TP), &norm()),
        Err(ShardError::DimOutOfRange { dim: 9, rank: 4 })
    );
    assert_eq!(
        transitions(&replicate(), &shard(-5, TP), &norm()),
        Err(ShardError::DimOutOfRange { dim: -5, rank: 4 })
    );
    // Same on the source side, and on the paths that do emit a collective.
    assert_eq!(
        transitions(&shard(4, TP), &replicate(), &norm()),
        Err(ShardError::DimOutOfRange { dim: 4, rank: 4 })
    );
    assert_eq!(
        transitions(&partial(ReduceOp::Sum, TP), &shard(4, TP), &norm()),
        Err(ShardError::DimOutOfRange { dim: 4, rank: 4 })
    );
    // An identical pair does not get a free pass either.
    assert_eq!(
        transitions(&shard(9, TP), &shard(9, TP), &norm()),
        Err(ShardError::DimOutOfRange { dim: 9, rank: 4 })
    );
    // One broken dim in a multi-shard layout fails the whole conversion.
    let from = layout(&[(0, EP), (9, TP)], None);
    assert_eq!(
        transitions(&from, &replicate(), &norm()),
        Err(ShardError::DimOutOfRange { dim: 9, rank: 4 })
    );
}

// ------------------------------------------------------------------ property

fn cfg(tp: usize, cp: usize, ep: usize, dp: usize, pp: usize) -> ParallelConfig {
    ParallelConfig {
        tensor: tp,
        context: cp,
        expert: ep,
        data: dp,
        pipeline: pp,
    }
}

/// For every conversion the rules actually return, each collective must address
/// a group that contains the rank whose tensor is being converted: a rank can
/// only take part in collectives of its own groups.
///
/// "Contains the rank" alone is weak — every group of a mask covers every rank
/// by construction — so the sweep also checks the stronger property that a
/// collective runs over a group one of the two *layouts* named (a rule may not
/// invent a group), that emitted dims are resolved, that no rule needs
/// `Broadcast`, and that the success paths are not vacuous.
#[test]
fn every_returned_collective_targets_a_group_of_the_source_rank() {
    let configs = [
        cfg(1, 1, 1, 1, 1),
        cfg(4, 1, 1, 1, 1),
        cfg(2, 2, 2, 2, 1),
        cfg(2, 2, 1, 2, 1),
        cfg(1, 1, 2, 1, 2),
    ];
    let layouts = [
        replicate(),
        shard(-1, TP),
        shard(0, CP),
        shard(2, EP),
        shard(1, TP.union(DP)),
        shard(1, PP),
        partial(ReduceOp::Sum, TP),
        partial(ReduceOp::Max, ALL),
        layout(&[(0, EP), (1, TP)], None),
        layout(&[(0, EP)], Some((ReduceOp::Sum, TP))),
    ];
    let tensor_rank = 4;
    let normalizer = DimNormalizer::new(tensor_rank).unwrap();

    let mut conversions = 0usize;
    let mut collectives = 0usize;
    for config in configs {
        let mesh = Mesh::from_config(&config);
        for rank in 0..config.world_size() {
            for from in &layouts {
                for to in &layouts {
                    let Ok(emitted) = transitions(from, to, &normalizer) else {
                        continue;
                    };
                    conversions += 1;
                    for collective in emitted {
                        collectives += 1;
                        let group = collective.group();
                        let members = mesh
                            .group_ranks(group, rank)
                            .expect("every mask used by the rules addresses the canonical axes");
                        assert!(
                            members.contains(&rank),
                            "rank {rank} cannot issue {collective}: not a member of {group}"
                        );
                        // The rank's index inside that group is well defined.
                        let index = mesh.group_index(group, rank).unwrap();
                        assert_eq!(members[index], rank);
                        // The collective must run over a group the layouts named.
                        assert!(
                            from.groups().contains(&group) || to.groups().contains(&group),
                            "{from} -> {to} emitted {collective} on a group neither \
                             layout mentions"
                        );
                        // Dims in a collective are ready for the runtime.
                        if let Collective::AllGather { dim, .. }
                        | Collective::ReduceScatter { dim, .. } = collective
                        {
                            assert!(
                                (0..tensor_rank).contains(&dim),
                                "{collective} carries an unresolved dim"
                            );
                        }
                        assert!(
                            !matches!(collective, Collective::Broadcast { .. }),
                            "no transition rule needs a broadcast: {collective}"
                        );
                    }
                }
            }
        }
    }
    assert!(
        conversions > 100,
        "the sweep must exercise real conversions"
    );
    assert!(collectives > 100, "the sweep must emit real collectives");
}

/// The whole rule table, written out: `from -> to = collectives`.
///
/// Pinning the table *as a table* (rather than only one test per row) makes a
/// silent reordering of emitted collectives fail loudly, because the order of
/// the vector is part of the contract — the plan compiler splices them in
/// order.
#[test]
fn the_transition_table() {
    type Case = (
        &'static str,
        ParallelLayout,
        ParallelLayout,
        Result<Vec<Collective>, ShardError>,
    );
    let rank4 = norm();
    let table: Vec<Case> = vec![
        (
            "replicate -> replicate",
            replicate(),
            replicate(),
            Ok(vec![]),
        ),
        (
            "partial -> replicate",
            partial(ReduceOp::Sum, TP),
            replicate(),
            Ok(vec![all_reduce(ReduceOp::Sum, TP)]),
        ),
        (
            "partial -> same partial",
            partial(ReduceOp::Max, TP),
            partial(ReduceOp::Max, TP),
            Ok(vec![]),
        ),
        (
            "partial(sum) -> shard",
            partial(ReduceOp::Sum, TP),
            shard(-1, TP),
            Ok(vec![reduce_scatter(3, TP)]),
        ),
        (
            "shard -> replicate",
            shard(-1, TP),
            replicate(),
            Ok(vec![all_gather(3, TP)]),
        ),
        ("replicate -> shard", replicate(), shard(0, CP), Ok(vec![])),
        (
            "shard -> other shard",
            shard(3, TP),
            shard(0, TP),
            Ok(vec![all_gather(3, TP)]),
        ),
        (
            "shard -> partial",
            shard(0, TP),
            partial(ReduceOp::Sum, TP),
            Err(ShardError::ShardToPartial {
                from: shard(0, TP),
                to: partial(ReduceOp::Sum, TP),
            }),
        ),
        (
            "shard group change",
            shard(0, TP),
            shard(0, CP),
            Err(ShardError::GroupMismatch {
                from: shard(0, TP),
                to: shard(0, CP),
            }),
        ),
        (
            "multi shard -> replicate",
            layout(&[(0, EP), (1, TP)], None),
            replicate(),
            Ok(vec![all_gather(0, EP), all_gather(1, TP)]),
        ),
        (
            "partial + shard -> replicate",
            layout(&[(0, EP)], Some((ReduceOp::Sum, TP))),
            replicate(),
            Ok(vec![all_reduce(ReduceOp::Sum, TP), all_gather(0, EP)]),
        ),
    ];
    for (name, from, to, expected) in table {
        assert_eq!(transitions(&from, &to, &rank4), expected, "{name}");
    }
}
