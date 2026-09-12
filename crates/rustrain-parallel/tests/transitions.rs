//! One test per transition rule, plus the error paths and a property check.
//!
//! The rule table is documented on `rustrain_parallel::transitions`; the test
//! names here mirror the rule order so that a missing rule is easy to spot.

use rustrain_parallel::{
    Collective, DimNormalizer, GroupKind, ParallelConfig, ParallelLayout, ProcessGroups, ReduceOp,
    ShardError, transitions,
};

fn shard(dim: i64, group: GroupKind) -> ParallelLayout {
    ParallelLayout::Shard { dim, group }
}

fn partial(op: ReduceOp, group: GroupKind) -> ParallelLayout {
    ParallelLayout::Partial { op, group }
}

fn expert(group: GroupKind) -> ParallelLayout {
    ParallelLayout::ExpertShard { group }
}

fn seq(group: GroupKind) -> ParallelLayout {
    ParallelLayout::SequenceShard { group }
}

const REPLICATE: ParallelLayout = ParallelLayout::Replicate;

/// Rank-4 tensors (`[batch, seq, head, dim]`) unless a test says otherwise.
fn norm() -> DimNormalizer {
    DimNormalizer::new(4).unwrap()
}

fn all_reduce(op: ReduceOp, group: GroupKind) -> Collective {
    Collective::AllReduce { group, op }
}

fn all_gather(dim: i64, group: GroupKind) -> Collective {
    Collective::AllGather { group, dim }
}

fn reduce_scatter(dim: i64, group: GroupKind) -> Collective {
    Collective::ReduceScatter { group, dim }
}

// ---------------------------------------------------------------- happy rules

#[test]
fn replicate_to_replicate_is_empty() {
    assert_eq!(transitions(&REPLICATE, &REPLICATE, &norm()), Ok(Vec::new()));
    // Even for a scalar: there is nothing to move.
    let scalar = DimNormalizer::new(0).unwrap();
    assert_eq!(transitions(&REPLICATE, &REPLICATE, &scalar), Ok(Vec::new()));
}

#[test]
fn partial_to_replicate_all_reduces() {
    for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
        assert_eq!(
            transitions(&partial(op, GroupKind::Tp), &REPLICATE, &norm()),
            Ok(vec![all_reduce(op, GroupKind::Tp)]),
            "partial({op}, tp)"
        );
    }
    // The group of the partial is the group of the all-reduce; `Global` is a
    // real group, not a wildcard.
    assert_eq!(
        transitions(
            &partial(ReduceOp::Sum, GroupKind::Global),
            &REPLICATE,
            &norm()
        ),
        Ok(vec![all_reduce(ReduceOp::Sum, GroupKind::Global)])
    );
}

#[test]
fn partial_to_partial_with_the_same_op_is_empty() {
    assert_eq!(
        transitions(
            &partial(ReduceOp::Sum, GroupKind::Tp),
            &partial(ReduceOp::Sum, GroupKind::Tp),
            &norm()
        ),
        Ok(Vec::new())
    );
    assert_eq!(
        transitions(
            &partial(ReduceOp::Max, GroupKind::Global),
            &partial(ReduceOp::Max, GroupKind::Global),
            &norm()
        ),
        Ok(Vec::new())
    );
}

#[test]
fn partial_to_partial_with_a_different_op_is_an_error() {
    let from = partial(ReduceOp::Max, GroupKind::Tp);
    let to = partial(ReduceOp::Sum, GroupKind::Tp);
    let err = transitions(&from, &to, &norm()).unwrap_err();
    assert_eq!(err, ShardError::PartialOpMismatch { from, to });
    // The reason (a partial is only a reduction of itself) and the way out
    // (round trip through Replicate) are both in the message.
    let message = err.to_string();
    assert!(message.contains("partial(max, tp)"), "{message}");
    assert!(message.contains("partial(sum, tp)"), "{message}");
    assert!(message.contains("replicate"), "{message}");
}

#[test]
fn partial_sum_to_shard_reduce_scatters() {
    // The dim in the emitted collective is the resolved one: -1 on a rank-4
    // tensor is axis 3.
    assert_eq!(
        transitions(
            &partial(ReduceOp::Sum, GroupKind::Tp),
            &shard(-1, GroupKind::Tp),
            &norm()
        ),
        Ok(vec![reduce_scatter(3, GroupKind::Tp)])
    );
    assert_eq!(
        transitions(
            &partial(ReduceOp::Sum, GroupKind::Dp),
            &shard(1, GroupKind::Dp),
            &norm()
        ),
        Ok(vec![reduce_scatter(1, GroupKind::Dp)])
    );
}

#[test]
fn non_sum_partial_to_shard_is_an_error() {
    let from = partial(ReduceOp::Max, GroupKind::Tp);
    let to = shard(-1, GroupKind::Tp);
    let err = transitions(&from, &to, &norm()).unwrap_err();
    assert_eq!(err, ShardError::ReduceScatterRequiresSum { from, to });
    let message = err.to_string();
    assert!(message.contains("reduce_scatter"), "{message}");
    assert!(message.contains("sum"), "{message}");
}

#[test]
fn shard_to_replicate_all_gathers() {
    assert_eq!(
        transitions(&shard(-1, GroupKind::Tp), &REPLICATE, &norm()),
        Ok(vec![all_gather(3, GroupKind::Tp)])
    );
    assert_eq!(
        transitions(&shard(0, GroupKind::Cp), &REPLICATE, &norm()),
        Ok(vec![all_gather(0, GroupKind::Cp)])
    );
    // Gathering the world's shards is a different collective from gathering a
    // TP group's shards.
    assert_eq!(
        transitions(&shard(2, GroupKind::Global), &REPLICATE, &norm()),
        Ok(vec![all_gather(2, GroupKind::Global)])
    );
}

/// The rule people get wrong: a replica already holds everything, so narrowing
/// it to a shard is a *local* slice. No collective, whatever the dim or group.
#[test]
fn replicate_to_shard_is_local() {
    for to in [
        shard(0, GroupKind::Tp),
        shard(-1, GroupKind::Tp),
        shard(2, GroupKind::Cp),
        shard(0, GroupKind::Global),
    ] {
        assert_eq!(
            transitions(&REPLICATE, &to, &norm()),
            Ok(Vec::new()),
            "replicate -> {to} must not communicate"
        );
    }
}

#[test]
fn shard_to_same_shard_is_empty() {
    assert_eq!(
        transitions(&shard(0, GroupKind::Tp), &shard(0, GroupKind::Tp), &norm()),
        Ok(Vec::new())
    );
    // -1 and 3 are the same axis on a rank-4 tensor, so this is the same
    // conversion spelled two ways.
    assert_eq!(
        transitions(&shard(-1, GroupKind::Tp), &shard(3, GroupKind::Tp), &norm()),
        Ok(Vec::new())
    );
}

#[test]
fn shard_to_other_shard_gathers_the_source_axis() {
    // The pieces live along the source dim, so that is what gets gathered; the
    // local re-slice along the target dim is not a collective.
    assert_eq!(
        transitions(&shard(3, GroupKind::Tp), &shard(0, GroupKind::Tp), &norm()),
        Ok(vec![all_gather(3, GroupKind::Tp)])
    );
    assert_eq!(
        transitions(&shard(-2, GroupKind::Cp), &shard(1, GroupKind::Cp), &norm()),
        Ok(vec![all_gather(2, GroupKind::Cp)])
    );
}

#[test]
fn shard_to_partial_is_an_error() {
    let cases = [
        (
            shard(0, GroupKind::Tp),
            partial(ReduceOp::Sum, GroupKind::Tp),
        ),
        (
            shard(-1, GroupKind::Dp),
            partial(ReduceOp::Max, GroupKind::Dp),
        ),
    ];
    for (from, to) in cases {
        let err = transitions(&from, &to, &norm()).unwrap_err();
        assert_eq!(err, ShardError::ShardToPartial { from, to });
        let message = err.to_string();
        assert!(message.contains("recomputation"), "{message}");
    }
}

#[test]
fn expert_shard_round_trip() {
    // Expert-parallel weights are `[num_experts, ...]`, so the gather is dim 0.
    assert_eq!(
        transitions(&expert(GroupKind::Ep), &REPLICATE, &norm()),
        Ok(vec![all_gather(0, GroupKind::Ep)])
    );
    assert_eq!(
        transitions(&REPLICATE, &expert(GroupKind::Ep), &norm()),
        Ok(Vec::new())
    );
    assert_eq!(
        transitions(&expert(GroupKind::Ep), &expert(GroupKind::Ep), &norm()),
        Ok(Vec::new())
    );
}

#[test]
fn sequence_shard_round_trip() {
    assert_eq!(
        transitions(&seq(GroupKind::Cp), &REPLICATE, &norm()),
        Ok(vec![all_gather(1, GroupKind::Cp)])
    );
    assert_eq!(
        transitions(&REPLICATE, &seq(GroupKind::Cp), &norm()),
        Ok(Vec::new())
    );
    // A plan whose activations are `[batch, head, seq, dim]` overrides the
    // sequence axis; the emitted collective follows the override.
    let heads_first = norm().with_sequence_dim(-2);
    assert_eq!(
        transitions(&seq(GroupKind::Cp), &REPLICATE, &heads_first),
        Ok(vec![all_gather(2, GroupKind::Cp)])
    );
}

// --------------------------------------------------------------- error paths

#[test]
fn different_groups_are_an_error() {
    let cases = [
        (shard(0, GroupKind::Tp), shard(0, GroupKind::Cp)),
        (shard(0, GroupKind::Cp), shard(0, GroupKind::Tp)),
        (
            partial(ReduceOp::Sum, GroupKind::Tp),
            partial(ReduceOp::Sum, GroupKind::Ep),
        ),
        (expert(GroupKind::Ep), expert(GroupKind::Dp)),
        (seq(GroupKind::Cp), seq(GroupKind::Tp)),
        // `Global` is a group, not a wildcard: a world-wide shard is not a
        // TP shard.
        (shard(0, GroupKind::Tp), shard(0, GroupKind::Global)),
    ];
    for (from, to) in cases {
        let err = transitions(&from, &to, &norm()).unwrap_err();
        assert_eq!(
            err,
            ShardError::GroupMismatch { from, to },
            "{from} -> {to}"
        );
        // The error has to say that the Replicate route exists, because that is
        // the only correct way forward.
        assert!(err.to_string().contains("replicate"), "{err}");
    }
}

/// The same group mismatch is *not* an error when one side is a replica: the
/// single-sided rules already cover it, and the group of the other side decides
/// the collective.
#[test]
fn a_replica_makes_the_other_side_s_group_irrelevant() {
    assert_eq!(
        transitions(&shard(0, GroupKind::Tp), &REPLICATE, &norm()),
        Ok(vec![all_gather(0, GroupKind::Tp)])
    );
    assert_eq!(
        transitions(&REPLICATE, &shard(0, GroupKind::Cp), &norm()),
        Ok(Vec::new())
    );
}

#[test]
fn replicate_to_partial_is_an_error() {
    for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
        let to = partial(op, GroupKind::Tp);
        let err = transitions(&REPLICATE, &to, &norm()).unwrap_err();
        assert_eq!(
            err,
            ShardError::ReplicateToPartial {
                from: REPLICATE,
                to
            }
        );
        assert!(err.to_string().contains("computation"), "{err}");
    }
}

#[test]
fn unsupported_pairs_are_an_error() {
    let cases = [
        (expert(GroupKind::Tp), shard(0, GroupKind::Tp)),
        (shard(0, GroupKind::Tp), expert(GroupKind::Tp)),
        (seq(GroupKind::Cp), partial(ReduceOp::Sum, GroupKind::Cp)),
        (partial(ReduceOp::Sum, GroupKind::Ep), expert(GroupKind::Ep)),
        (expert(GroupKind::Ep), seq(GroupKind::Ep)),
    ];
    for (from, to) in cases {
        let err = transitions(&from, &to, &norm()).unwrap_err();
        assert_eq!(
            err,
            ShardError::UnsupportedTransition { from, to },
            "{from} -> {to}"
        );
    }
}

#[test]
fn dims_are_validated_even_when_the_conversion_is_local() {
    // Replicate -> Shard needs no communication, but the axis still has to
    // exist: a plan that names axis 9 of a rank-4 tensor is broken either way.
    assert_eq!(
        transitions(&REPLICATE, &shard(9, GroupKind::Tp), &norm()),
        Err(ShardError::DimOutOfRange { dim: 9, rank: 4 })
    );
    assert_eq!(
        transitions(&REPLICATE, &shard(-5, GroupKind::Tp), &norm()),
        Err(ShardError::DimOutOfRange { dim: -5, rank: 4 })
    );
    // Same on the source side, and on the paths that do emit a collective.
    assert_eq!(
        transitions(&shard(4, GroupKind::Tp), &REPLICATE, &norm()),
        Err(ShardError::DimOutOfRange { dim: 4, rank: 4 })
    );
    assert_eq!(
        transitions(
            &partial(ReduceOp::Sum, GroupKind::Tp),
            &shard(4, GroupKind::Tp),
            &norm()
        ),
        Err(ShardError::DimOutOfRange { dim: 4, rank: 4 })
    );
    // An identical pair does not get a free pass either.
    assert_eq!(
        transitions(&shard(9, GroupKind::Tp), &shard(9, GroupKind::Tp), &norm()),
        Err(ShardError::DimOutOfRange { dim: 9, rank: 4 })
    );
    // A SequenceShard carries no dim of its own, but it still claims the tensor
    // has a sequence axis — checked in both directions, and for an override
    // that does not exist on the tensor.
    let scalar = DimNormalizer::new(0).unwrap();
    assert_eq!(
        transitions(&REPLICATE, &seq(GroupKind::Cp), &scalar),
        Err(ShardError::DimOutOfRange { dim: 1, rank: 0 })
    );
    assert_eq!(
        transitions(&seq(GroupKind::Cp), &REPLICATE, &scalar),
        Err(ShardError::DimOutOfRange { dim: 1, rank: 0 })
    );
    let missing_axis = norm().with_sequence_dim(9);
    assert_eq!(
        transitions(&REPLICATE, &seq(GroupKind::Cp), &missing_axis),
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
/// "Contains the rank" alone is weak — every group of a kind covers every rank
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
        REPLICATE,
        shard(-1, GroupKind::Tp),
        shard(0, GroupKind::Cp),
        shard(2, GroupKind::Ep),
        shard(1, GroupKind::Global),
        partial(ReduceOp::Sum, GroupKind::Tp),
        partial(ReduceOp::Max, GroupKind::Global),
        expert(GroupKind::Ep),
        seq(GroupKind::Cp),
    ];
    let tensor_rank = 4;
    let normalizer = DimNormalizer::new(tensor_rank).unwrap();

    let mut conversions = 0usize;
    let mut collectives = 0usize;
    for config in configs {
        let groups = ProcessGroups::new(config);
        for rank in 0..config.world_size() {
            for from in layouts {
                for to in layouts {
                    let Ok(emitted) = transitions(&from, &to, &normalizer) else {
                        continue;
                    };
                    conversions += 1;
                    for collective in emitted {
                        collectives += 1;
                        let kind = collective.group();
                        let group = groups
                            .group_of(rank, kind)
                            .expect("every group kind covers every rank of the world");
                        assert!(
                            group.contains(rank),
                            "rank {rank} cannot issue {collective}: not a member of {kind}"
                        );
                        // The rank's index inside that group is well defined.
                        let index = groups.group_index(rank, kind).unwrap();
                        assert_eq!(group.ranks[index], rank);
                        // The collective must run over a group the layouts named.
                        assert!(
                            from.group() == Some(kind) || to.group() == Some(kind),
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
