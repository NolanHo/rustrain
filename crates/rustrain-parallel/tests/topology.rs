//! Topology tests: the rank order contract, the hand-computed rank tables, and
//! process group membership.
//!
//! The tables below are the point of this file. They are written out by hand
//! from the contract in `src/rank.rs`, so a change to the rank order (a
//! different dimension order, a different stride) fails here instead of
//! silently reshuffling which rank holds which slice.
//!
//! Pinning the *whole* order needs a config in which all four leading strides
//! are visible at once, which is why one table is `tp=2, cp=2, ep=2, dp=2`
//! (world 16): a table that leaves `ep` or `dp` at size 1 cannot tell those two
//! dimensions apart.

use rustrain_parallel::{
    GroupKind, ParallelConfig, ParallelDim, ParallelError, ProcessGroups, RankLayout,
};

fn cfg(tp: usize, cp: usize, ep: usize, dp: usize, pp: usize) -> ParallelConfig {
    // Named fields on purpose: a positional constructor would make it easy to
    // swap `context` and `expert` in a test and never notice.
    ParallelConfig {
        tensor: tp,
        context: cp,
        expert: ep,
        data: dp,
        pipeline: pp,
    }
}

#[test]
fn default_config_is_a_single_rank() {
    let c = ParallelConfig::default();
    assert_eq!(c, cfg(1, 1, 1, 1, 1));
    assert_eq!(c.world_size(), 1);
    assert_eq!(c.checked_world_size(), Some(1));
    assert_eq!(c.validate(), Ok(()));
    for (dim, size) in c.dimensions() {
        assert_eq!(size, 1, "{dim} defaults to 1");
        assert_eq!(c.dimension(dim), 1);
    }
}

#[test]
fn world_size_is_the_product_of_the_dimensions() {
    assert_eq!(cfg(1, 1, 1, 1, 1).world_size(), 1);
    assert_eq!(cfg(8, 1, 1, 1, 1).world_size(), 8);
    assert_eq!(cfg(2, 2, 2, 2, 1).world_size(), 16);
    assert_eq!(cfg(2, 2, 2, 2, 2).world_size(), 32);
    assert_eq!(cfg(3, 5, 7, 1, 1).world_size(), 105);
    assert_eq!(cfg(3, 5, 7, 1, 1).checked_world_size(), Some(105));
}

#[test]
fn zero_dimensions_are_rejected() {
    let cases = [
        (cfg(0, 1, 1, 1, 1), ParallelDim::Tp),
        (cfg(1, 0, 1, 1, 1), ParallelDim::Cp),
        (cfg(1, 1, 0, 1, 1), ParallelDim::Ep),
        (cfg(1, 1, 1, 0, 1), ParallelDim::Dp),
        (cfg(1, 1, 1, 1, 0), ParallelDim::Pp),
    ];
    for (c, dim) in cases {
        assert_eq!(
            c.validate(),
            Err(ParallelError::ZeroDimension { dim }),
            "config {c:?}"
        );
        // A zero dimension is also the case where the product is 0: there is no
        // world size that describes "no ranks on this axis".
        assert_eq!(c.world_size(), 0, "config {c:?}");
    }
}

#[test]
fn overflowing_world_size_is_rejected() {
    let c = cfg(usize::MAX, 2, 1, 1, 1);
    assert_eq!(c.checked_world_size(), None);
    assert!(
        matches!(c.validate(), Err(ParallelError::WorldSizeOverflow { .. })),
        "an overflowing product must not be usable as a topology: {:?}",
        c.validate()
    );
    // `world_size` saturates rather than wrapping: a wrapped product could look
    // like a plausible world size and silently pass a rank range check.
    assert_eq!(c.world_size(), usize::MAX);
}

#[test]
fn rank_round_trips_for_world_16() {
    let c = cfg(2, 2, 2, 2, 1);
    assert_eq!(c.world_size(), 16);
    let mut seen = Vec::new();
    for rank in 0..c.world_size() {
        let layout = RankLayout::from_rank(rank, c).unwrap();
        assert_eq!(layout.rank(), rank, "round trip for rank {rank}");
        assert_eq!(layout.config(), c);

        // Coordinates stay inside their dimension.
        assert!(layout.tp_rank() < c.tensor);
        assert!(layout.cp_rank() < c.context);
        assert!(layout.ep_rank() < c.expert);
        assert!(layout.dp_rank() < c.data);
        assert!(layout.pp_rank() < c.pipeline);
        // `coordinate` agrees with the named accessors.
        assert_eq!(layout.coordinate(ParallelDim::Tp), layout.tp_rank());
        assert_eq!(layout.coordinate(ParallelDim::Cp), layout.cp_rank());
        assert_eq!(layout.coordinate(ParallelDim::Ep), layout.ep_rank());
        assert_eq!(layout.coordinate(ParallelDim::Dp), layout.dp_rank());
        assert_eq!(layout.coordinate(ParallelDim::Pp), layout.pp_rank());

        seen.push((
            layout.tp_rank(),
            layout.cp_rank(),
            layout.ep_rank(),
            layout.dp_rank(),
            layout.pp_rank(),
        ));
    }
    // 16 ranks, 16 distinct coordinates: the decomposition is a bijection.
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 16);
}

#[test]
fn rank_round_trips_for_degenerate_configs() {
    let configs = [
        cfg(1, 1, 1, 1, 1), // single rank
        cfg(8, 1, 1, 1, 1), // tp only
        cfg(1, 4, 1, 1, 1), // cp only
        cfg(1, 1, 3, 1, 1), // ep only
        cfg(1, 1, 1, 5, 1), // dp only
        cfg(1, 1, 1, 1, 7), // pp only
        cfg(3, 5, 1, 1, 1), // non-power-of-two
        cfg(1, 1, 1, 2, 2), // dp and pp, nothing else
        cfg(2, 2, 2, 2, 2), // world 32
    ];
    for c in configs {
        c.validate().unwrap();
        for rank in 0..c.world_size() {
            let layout = RankLayout::from_rank(rank, c).unwrap();
            assert_eq!(layout.rank(), rank, "{c:?} rank {rank}");
        }
    }
}

/// Hand-computed table for `tp=2, cp=2, ep=2, dp=1, pp=1` (world 8) — the
/// acceptance configuration of deliverable D3.
///
/// TP varies fastest, then CP, then EP, then DP, then PP:
///
/// ```text
/// rank = ((((pp_rank * dp + dp_rank) * ep + ep_rank) * cp + cp_rank) * tp + tp_rank)
/// ```
#[test]
fn hand_computed_rank_table_tp2_cp2_ep2() {
    let c = cfg(2, 2, 2, 1, 1);
    // (rank, tp, cp, ep, dp, pp)
    let expected: [(usize, usize, usize, usize, usize, usize); 8] = [
        (0, 0, 0, 0, 0, 0),
        (1, 1, 0, 0, 0, 0),
        (2, 0, 1, 0, 0, 0),
        (3, 1, 1, 0, 0, 0),
        (4, 0, 0, 1, 0, 0),
        (5, 1, 0, 1, 0, 0),
        (6, 0, 1, 1, 0, 0),
        (7, 1, 1, 1, 0, 0),
    ];
    assert_eq!(expected.len(), c.world_size());

    for (rank, tp, cp, ep, dp, pp) in expected {
        let layout = RankLayout::from_rank(rank, c).unwrap();
        assert_eq!(
            (
                layout.tp_rank(),
                layout.cp_rank(),
                layout.ep_rank(),
                layout.dp_rank(),
                layout.pp_rank(),
            ),
            (tp, cp, ep, dp, pp),
            "rank {rank}"
        );
        // The same formula the table was computed from, transcribed literally:
        // this is what makes the table a check on the contract rather than on
        // itself.
        assert_eq!(
            ((((pp * c.data + dp) * c.expert + ep) * c.context + cp) * c.tensor) + tp,
            rank
        );
    }
}

/// Hand-computed table for `tp=2, dp=2, pp=2` (world 8), which pins the DP and
/// PP strides that the table above leaves constant at 1.
#[test]
fn hand_computed_rank_table_tp2_dp2_pp2() {
    let c = cfg(2, 1, 1, 2, 2);
    assert_eq!(c.world_size(), 8);
    // (rank, tp, cp, ep, dp, pp)
    let expected: [(usize, usize, usize, usize, usize, usize); 8] = [
        (0, 0, 0, 0, 0, 0),
        (1, 1, 0, 0, 0, 0),
        (2, 0, 0, 0, 1, 0),
        (3, 1, 0, 0, 1, 0),
        (4, 0, 0, 0, 0, 1),
        (5, 1, 0, 0, 0, 1),
        (6, 0, 0, 0, 1, 1),
        (7, 1, 0, 0, 1, 1),
    ];

    for (rank, tp, cp, ep, dp, pp) in expected {
        let layout = RankLayout::from_rank(rank, c).unwrap();
        assert_eq!(
            (
                layout.tp_rank(),
                layout.cp_rank(),
                layout.ep_rank(),
                layout.dp_rank(),
                layout.pp_rank(),
            ),
            (tp, cp, ep, dp, pp),
            "rank {rank}"
        );
        assert_eq!(
            ((((pp * c.data + dp) * c.expert + ep) * c.context + cp) * c.tensor) + tp,
            rank
        );
    }
}

/// Hand-computed table for `tp=2, cp=2, ep=2, dp=2` (world 16): the config that
/// pins the *whole* order at once.
///
/// The tables above each leave one of `ep`/`dp` at size 1, so neither can tell
/// those two dimensions apart — an `ep`↔`dp` swap would satisfy both of them.
/// Here every stride is visible: `rank = 8*dp + 4*ep + 2*cp + tp`.
#[test]
fn hand_computed_rank_table_tp2_cp2_ep2_dp2() {
    let c = cfg(2, 2, 2, 2, 1);
    assert_eq!(c.world_size(), 16);
    // (rank, tp, cp, ep, dp, pp)
    let expected: [(usize, usize, usize, usize, usize, usize); 16] = [
        (0, 0, 0, 0, 0, 0),
        (1, 1, 0, 0, 0, 0),
        (2, 0, 1, 0, 0, 0),
        (3, 1, 1, 0, 0, 0),
        (4, 0, 0, 1, 0, 0),
        (5, 1, 0, 1, 0, 0),
        (6, 0, 1, 1, 0, 0),
        (7, 1, 1, 1, 0, 0),
        (8, 0, 0, 0, 1, 0),
        (9, 1, 0, 0, 1, 0),
        (10, 0, 1, 0, 1, 0),
        (11, 1, 1, 0, 1, 0),
        (12, 0, 0, 1, 1, 0),
        (13, 1, 0, 1, 1, 0),
        (14, 0, 1, 1, 1, 0),
        (15, 1, 1, 1, 1, 0),
    ];

    for (rank, tp, cp, ep, dp, pp) in expected {
        let layout = RankLayout::from_rank(rank, c).unwrap();
        assert_eq!(
            (
                layout.tp_rank(),
                layout.cp_rank(),
                layout.ep_rank(),
                layout.dp_rank(),
                layout.pp_rank(),
            ),
            (tp, cp, ep, dp, pp),
            "rank {rank}"
        );
        assert_eq!(
            ((((pp * c.data + dp) * c.expert + ep) * c.context + cp) * c.tensor) + tp,
            rank
        );
        // The strides spelled out, so that a swapped pair is reported as such
        // rather than as an anonymous table mismatch.
        assert_eq!(rank, 8 * dp + 4 * ep + 2 * cp + tp, "rank {rank} strides");
    }
}

/// `tp=2, cp=2, ep=2, dp=2` (world 16), every group written out in full.
///
/// This is the group-level counterpart of the table above: it pins the group
/// strides (`tp`=1, `cp`=2, `ep`=4, `dp`=8) with `ep` and `dp` both > 1.
#[test]
fn groups_for_tp2_cp2_ep2_dp2_world_16() {
    let c = cfg(2, 2, 2, 2, 1);
    let groups = ProcessGroups::new(c);
    let ranks = |kind: GroupKind| -> Vec<Vec<usize>> {
        groups
            .groups(kind)
            .iter()
            .map(|group| group.ranks.clone())
            .collect()
    };

    assert_eq!(
        ranks(GroupKind::Tp),
        vec![
            vec![0, 1],
            vec![2, 3],
            vec![4, 5],
            vec![6, 7],
            vec![8, 9],
            vec![10, 11],
            vec![12, 13],
            vec![14, 15]
        ]
    );
    assert_eq!(
        ranks(GroupKind::Cp),
        vec![
            vec![0, 2],
            vec![1, 3],
            vec![4, 6],
            vec![5, 7],
            vec![8, 10],
            vec![9, 11],
            vec![12, 14],
            vec![13, 15]
        ]
    );
    // EP's stride is tp * cp = 4.
    assert_eq!(
        ranks(GroupKind::Ep),
        vec![
            vec![0, 4],
            vec![1, 5],
            vec![2, 6],
            vec![3, 7],
            vec![8, 12],
            vec![9, 13],
            vec![10, 14],
            vec![11, 15]
        ]
    );
    // DP's stride is tp * cp * ep = 8.
    assert_eq!(
        ranks(GroupKind::Dp),
        vec![
            vec![0, 8],
            vec![1, 9],
            vec![2, 10],
            vec![3, 11],
            vec![4, 12],
            vec![5, 13],
            vec![6, 14],
            vec![7, 15]
        ]
    );
    assert_eq!(ranks(GroupKind::Pp).len(), 16);
    assert_eq!(
        ranks(GroupKind::Global),
        vec![(0..16).collect::<Vec<usize>>()]
    );

    assert_eq!(groups.group_size(GroupKind::Ep), 2);
    assert_eq!(groups.group_size(GroupKind::Dp), 2);
    assert_eq!(groups.group_count(GroupKind::Ep), 8);
    assert_eq!(groups.group_count(GroupKind::Dp), 8);
}

#[test]
fn rank_out_of_range_is_rejected() {
    let c = cfg(2, 2, 2, 2, 1);
    assert_eq!(
        RankLayout::from_rank(16, c),
        Err(ParallelError::RankOutOfRange {
            rank: 16,
            world_size: 16
        })
    );
    // An invalid topology is reported before the rank is even considered: the
    // decomposition would divide by zero.
    assert_eq!(
        RankLayout::from_rank(0, cfg(0, 1, 1, 1, 1)),
        Err(ParallelError::ZeroDimension {
            dim: ParallelDim::Tp
        })
    );
}

/// `tp=2, cp=2, dp=2` (world 8). Every group is written out in full, because
/// "which ranks share a TP group" is the input to every collective decision.
#[test]
fn groups_for_tp2_cp2_dp2_world_8() {
    let c = cfg(2, 2, 1, 2, 1);
    assert_eq!(c.world_size(), 8);
    let groups = ProcessGroups::new(c);
    assert_eq!(groups.world_size(), 8);
    assert_eq!(groups.config(), c);

    let ranks = |kind: GroupKind| -> Vec<Vec<usize>> {
        groups
            .groups(kind)
            .iter()
            .map(|group| group.ranks.clone())
            .collect()
    };

    // TP: the fastest dimension, so its groups are adjacent ranks.
    assert_eq!(
        ranks(GroupKind::Tp),
        vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]
    );
    // CP: stride tp = 2.
    assert_eq!(
        ranks(GroupKind::Cp),
        vec![vec![0, 2], vec![1, 3], vec![4, 6], vec![5, 7]]
    );
    // EP: size 1, so every rank is its own group.
    assert_eq!(
        ranks(GroupKind::Ep),
        vec![
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![5],
            vec![6],
            vec![7]
        ]
    );
    // DP: stride tp * cp = 4.
    assert_eq!(
        ranks(GroupKind::Dp),
        vec![vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]]
    );
    // PP: size 1 as well.
    assert_eq!(ranks(GroupKind::Pp).len(), 8);
    for group in ranks(GroupKind::Pp) {
        assert_eq!(group.len(), 1);
    }
    // Global: everything.
    assert_eq!(ranks(GroupKind::Global), vec![vec![0, 1, 2, 3, 4, 5, 6, 7]]);

    assert_eq!(groups.group_size(GroupKind::Tp), 2);
    assert_eq!(groups.group_size(GroupKind::Cp), 2);
    assert_eq!(groups.group_size(GroupKind::Dp), 2);
    assert_eq!(groups.group_size(GroupKind::Ep), 1);
    assert_eq!(groups.group_size(GroupKind::Global), 8);
    assert_eq!(groups.group_count(GroupKind::Tp), 4);
    assert_eq!(groups.group_count(GroupKind::Dp), 4);
    assert_eq!(groups.group_count(GroupKind::Global), 1);

    for rank in 0..8 {
        let layout = RankLayout::from_rank(rank, c).unwrap();
        // The rank's position inside a dimension's group is its coordinate on
        // that dimension; inside the Global group it is the rank itself.
        for (kind, coordinate) in [
            (GroupKind::Tp, layout.tp_rank()),
            (GroupKind::Cp, layout.cp_rank()),
            (GroupKind::Ep, layout.ep_rank()),
            (GroupKind::Dp, layout.dp_rank()),
            (GroupKind::Pp, layout.pp_rank()),
            (GroupKind::Global, rank),
        ] {
            assert_eq!(
                groups.group_index(rank, kind).unwrap(),
                coordinate,
                "rank {rank} in {kind}"
            );
            let group = groups.group_of(rank, kind).unwrap();
            assert!(group.contains(rank), "rank {rank} must be in its own group");
            assert_eq!(group.kind, kind);
            assert_eq!(group.size(), groups.group_size(kind));
            // The index really does address the rank inside the group.
            assert_eq!(group.ranks[groups.group_index(rank, kind).unwrap()], rank);
        }
    }

    // Ranks are ascending inside every group of every kind (a property the
    // runtime relies on: the index is the local collective rank).
    for kind in GroupKind::ALL {
        for group in groups.groups(kind) {
            assert!(
                group.ranks.windows(2).all(|w| w[0] < w[1]),
                "{kind} group is not ascending: {:?}",
                group.ranks
            );
        }
    }
}

/// `tp=2, cp=2, ep=2` (world 8): the acceptance configuration of D3 in the
/// spec. Same rank table as `hand_computed_rank_table_tp2_cp2_ep2`, whose
/// groups are written out in full here.
#[test]
fn groups_for_tp2_cp2_ep2_world_8() {
    let c = cfg(2, 2, 2, 1, 1);
    assert_eq!(c.world_size(), 8);
    let groups = ProcessGroups::new(c);
    let ranks = |kind: GroupKind| -> Vec<Vec<usize>> {
        groups
            .groups(kind)
            .iter()
            .map(|group| group.ranks.clone())
            .collect()
    };

    assert_eq!(
        ranks(GroupKind::Tp),
        vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]
    );
    assert_eq!(
        ranks(GroupKind::Cp),
        vec![vec![0, 2], vec![1, 3], vec![4, 6], vec![5, 7]]
    );
    // EP sits above CP in the rank order, so its stride is tp * cp = 4.
    assert_eq!(
        ranks(GroupKind::Ep),
        vec![vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]]
    );
    assert_eq!(ranks(GroupKind::Global), vec![vec![0, 1, 2, 3, 4, 5, 6, 7]]);
    assert_eq!(groups.group_size(GroupKind::Ep), 2);
    assert_eq!(groups.group_count(GroupKind::Ep), 4);

    // Coordinates and group positions agree, rank by rank.
    // (rank, tp, cp, ep, dp, pp)
    let expected = [
        (0, 0, 0, 0, 0, 0),
        (1, 1, 0, 0, 0, 0),
        (2, 0, 1, 0, 0, 0),
        (3, 1, 1, 0, 0, 0),
        (4, 0, 0, 1, 0, 0),
        (5, 1, 0, 1, 0, 0),
        (6, 0, 1, 1, 0, 0),
        (7, 1, 1, 1, 0, 0),
    ];
    for (rank, tp, cp, ep, dp, pp) in expected {
        let layout = RankLayout::from_rank(rank, c).unwrap();
        assert_eq!(
            (
                layout.tp_rank(),
                layout.cp_rank(),
                layout.ep_rank(),
                layout.dp_rank(),
                layout.pp_rank(),
            ),
            (tp, cp, ep, dp, pp),
            "rank {rank}"
        );
        assert_eq!(
            (
                groups.group_index(rank, GroupKind::Tp).unwrap(),
                groups.group_index(rank, GroupKind::Cp).unwrap(),
                groups.group_index(rank, GroupKind::Ep).unwrap(),
                groups.group_index(rank, GroupKind::Dp).unwrap(),
                groups.group_index(rank, GroupKind::Pp).unwrap(),
            ),
            (tp, cp, ep, dp, pp),
            "group index of rank {rank}"
        );
    }
}

#[test]
fn group_lookup_rejects_ranks_outside_the_world() {
    let groups = ProcessGroups::new(cfg(2, 2, 1, 2, 1));
    let err = ParallelError::RankOutOfRange {
        rank: 8,
        world_size: 8,
    };
    assert_eq!(groups.group_of(8, GroupKind::Tp), Err(err.clone()));
    assert_eq!(groups.group_index(8, GroupKind::Global), Err(err));
}

#[test]
fn invalid_configs_do_not_build_process_groups() {
    let bad = cfg(2, 0, 1, 1, 1);
    assert_eq!(
        ProcessGroups::try_new(bad),
        Err(ParallelError::ZeroDimension {
            dim: ParallelDim::Cp
        })
    );
}

#[test]
#[should_panic(expected = "invalid parallel config")]
fn process_groups_new_panics_on_an_invalid_config() {
    ProcessGroups::new(cfg(1, 1, 0, 1, 1));
}

/// The group kind ↔ dimension mapping is a bijection on the five topology
/// dimensions, and `GroupKind::ALL` is the frozen iteration order.
#[test]
fn group_kind_maps_to_parallel_dim() {
    let dims = [
        ParallelDim::Tp,
        ParallelDim::Cp,
        ParallelDim::Ep,
        ParallelDim::Dp,
        ParallelDim::Pp,
    ];
    for (index, dim) in dims.into_iter().enumerate() {
        let kind = GroupKind::from_dim(dim);
        assert_eq!(kind.dim(), Some(dim));
        assert_eq!(kind, GroupKind::ALL[index]);
        assert_eq!(kind.to_string(), dim.to_string());
        assert_eq!(dim.group_kind(), kind);
    }
    assert_eq!(GroupKind::Global.dim(), None);
    assert_eq!(GroupKind::ALL[5], GroupKind::Global);
    assert_eq!(GroupKind::ALL.len(), 6);
}
