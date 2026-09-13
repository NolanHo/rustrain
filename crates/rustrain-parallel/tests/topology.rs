//! Topology tests: the rank order contract, the hand-computed rank tables, and
//! group membership over masks.
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

use rustrain_parallel::{GroupMask, Mesh, ParallelConfig, ParallelDim, ParallelError, RankLayout};

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

/// The canonical five-axis mesh of a config: `tp` is axis 0, `pp` is axis 4.
fn mesh(c: ParallelConfig) -> Mesh {
    Mesh::from_config(&c)
}

fn single(axis: usize) -> GroupMask {
    GroupMask::single(axis).unwrap()
}

/// Enumerates every group of `mask`, ordered by first member, as `(id, members)`.
///
/// The first member of group `i` is at most the first member of group `i + 1`
/// — the "groups ordered by first member" contract the runtime's communicator
/// creation relies on.
fn groups_by_first_member(mesh: &Mesh, mask: GroupMask) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for rank in 0..mesh.world_size() {
        let id = mesh.group_id(mask, rank).unwrap();
        let members = mesh.group_ranks(mask, rank).unwrap();
        if id == groups.len() {
            groups.push(members);
        } else {
            assert_eq!(
                groups[id], members,
                "rank {rank} disagrees with the group of the first member"
            );
        }
    }
    groups
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

/// The mesh of a config is the five canonical axes in rank order, so a mask
/// bit is exactly a `ParallelDim` in axis order.
#[test]
fn from_config_produces_the_canonical_five_axes() {
    let c = cfg(4, 3, 2, 5, 2);
    let m = mesh(c);
    assert_eq!(
        m.axes(),
        &[
            ("tp".to_string(), 4),
            ("cp".to_string(), 3),
            ("ep".to_string(), 2),
            ("dp".to_string(), 5),
            ("pp".to_string(), 2),
        ]
    );
    assert_eq!(m.axis_count(), 5);
    assert_eq!(m.world_size(), 4 * 3 * 2 * 5 * 2);
    for (axis, dim) in [
        ParallelDim::Tp,
        ParallelDim::Cp,
        ParallelDim::Ep,
        ParallelDim::Dp,
        ParallelDim::Pp,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(m.index_of(dim.as_str()), Some(axis));
        assert_eq!(m.degree(axis), Some(c.dimension(dim)));
    }
    assert_eq!(m.index_of("nope"), None);
    assert_eq!(m.degree(5), None);
    assert_eq!(m.stride(5), None);
    assert_eq!(m.stride(0), Some(1));
    assert_eq!(m.stride(1), Some(4));
    assert_eq!(m.stride(2), Some(12));
    assert_eq!(m.stride(3), Some(24));
    assert_eq!(m.stride(4), Some(120));

    // All five axes are kept even at degree 1: a mask over such an axis is a
    // size-1 group, legal, a no-op — not an out-of-range bit.
    let degenerate = mesh(cfg(1, 1, 1, 1, 1));
    assert_eq!(degenerate.axis_count(), 5);
    for rank in 0..1 {
        for axis in 0..5 {
            assert_eq!(degenerate.group_ranks(single(axis), rank), Ok(vec![0]));
            assert_eq!(degenerate.group_index(single(axis), rank), Ok(0));
            assert_eq!(degenerate.group_id(single(axis), rank), Ok(0));
        }
    }
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
    let m = mesh(c);
    let ranks = |axis: usize| -> Vec<Vec<usize>> { groups_by_first_member(&m, single(axis)) };

    assert_eq!(
        ranks(0),
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
        ranks(1),
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
        ranks(2),
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
        ranks(3),
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
    assert_eq!(ranks(4).len(), 16);
    // The full mask is the whole world: one group of all 16 ranks.
    let full = GroupMask::from_bits(0b11111);
    assert_eq!(
        groups_by_first_member(&m, full),
        vec![(0..16).collect::<Vec<usize>>()]
    );

    assert_eq!(single(2).degree(&m).unwrap(), 2);
    assert_eq!(single(3).degree(&m).unwrap(), 2);
    assert_eq!(ranks(2).len(), 8);
    assert_eq!(ranks(3).len(), 8);
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
    let m = mesh(c);
    assert_eq!(m.world_size(), 8);
    assert_eq!(m.axes(), Mesh::from_config(&c).axes());

    let ranks = |axis: usize| -> Vec<Vec<usize>> { groups_by_first_member(&m, single(axis)) };

    // TP: the fastest dimension, so its groups are adjacent ranks.
    assert_eq!(
        ranks(0),
        vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]
    );
    // CP: stride tp = 2.
    assert_eq!(
        ranks(1),
        vec![vec![0, 2], vec![1, 3], vec![4, 6], vec![5, 7]]
    );
    // EP: size 1, so every rank is its own group — a legal, no-op axis.
    assert_eq!(
        ranks(2),
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
        ranks(3),
        vec![vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]]
    );
    // PP: size 1 as well.
    assert_eq!(ranks(4).len(), 8);
    for group in ranks(4) {
        assert_eq!(group.len(), 1);
    }
    // Full mask: everything.
    let full = GroupMask::from_bits(0b11111);
    assert_eq!(
        groups_by_first_member(&m, full),
        vec![vec![0, 1, 2, 3, 4, 5, 6, 7]]
    );

    for rank in 0..8 {
        let layout = RankLayout::from_rank(rank, c).unwrap();
        // The rank's position inside a dimension's group is its coordinate on
        // that dimension; inside the world group it is the rank itself.
        for (axis, coordinate) in [
            (0, layout.tp_rank()),
            (1, layout.cp_rank()),
            (2, layout.ep_rank()),
            (3, layout.dp_rank()),
            (4, layout.pp_rank()),
        ] {
            let mask = single(axis);
            assert_eq!(
                m.group_index(mask, rank).unwrap(),
                coordinate,
                "rank {rank} in axis {axis}"
            );
            let members = m.group_ranks(mask, rank).unwrap();
            assert!(
                members.contains(&rank),
                "rank {rank} must be in its own group"
            );
            // The index really does address the rank inside the group.
            assert_eq!(members[m.group_index(mask, rank).unwrap()], rank);
        }
        assert_eq!(m.group_index(full, rank).unwrap(), rank);
        assert_eq!(m.group_id(full, rank).unwrap(), 0);
    }

    // Ranks are ascending inside every group of every axis (a property the
    // runtime relies on: the index is the local collective rank).
    for axis in 0..5 {
        for group in ranks(axis) {
            assert!(
                group.windows(2).all(|w| w[0] < w[1]),
                "axis {axis} group is not ascending: {:?}",
                group
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
    let m = mesh(c);
    let ranks = |axis: usize| -> Vec<Vec<usize>> { groups_by_first_member(&m, single(axis)) };

    assert_eq!(
        ranks(0),
        vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]
    );
    assert_eq!(
        ranks(1),
        vec![vec![0, 2], vec![1, 3], vec![4, 6], vec![5, 7]]
    );
    // EP sits above CP in the rank order, so its stride is tp * cp = 4.
    assert_eq!(
        ranks(2),
        vec![vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]]
    );
    let full = GroupMask::from_bits(0b11111);
    assert_eq!(
        groups_by_first_member(&m, full),
        vec![vec![0, 1, 2, 3, 4, 5, 6, 7]]
    );

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
                m.group_index(single(0), rank).unwrap(),
                m.group_index(single(1), rank).unwrap(),
                m.group_index(single(2), rank).unwrap(),
                m.group_index(single(3), rank).unwrap(),
                m.group_index(single(4), rank).unwrap(),
            ),
            (tp, cp, ep, dp, pp),
            "group index of rank {rank}"
        );
    }
}

#[test]
fn group_lookup_rejects_ranks_outside_the_world() {
    let m = mesh(cfg(2, 2, 1, 2, 1));
    let err = ParallelError::RankOutOfRange {
        rank: 8,
        world_size: 8,
    };
    assert_eq!(m.group_ranks(single(0), 8), Err(err.clone()));
    assert_eq!(m.group_index(single(0), 8), Err(err.clone()));
    assert_eq!(m.group_id(single(0), 8), Err(err));
}

/// A mask bit outside the mesh is the error case, on every group query and on
/// the mask's own degree.
#[test]
fn mask_bits_outside_the_mesh_are_rejected() {
    let m = mesh(cfg(2, 2, 1, 2, 1));
    let err = ParallelError::GroupOutOfRange { bit: 5, axes: 5 };
    let stray = GroupMask::from_bits(0b100000);
    assert_eq!(stray.validate(&m), Err(err.clone()));
    assert_eq!(stray.degree(&m), Err(err.clone()));
    assert_eq!(m.group_ranks(stray, 0), Err(err.clone()));
    assert_eq!(m.group_index(stray, 0), Err(err.clone()));
    assert_eq!(m.group_id(stray, 0), Err(err.clone()));
    assert_eq!(m.group_name(stray), Err(err.clone()));
    assert_eq!(m.fingerprint().group_name(stray), Err(err));
}

/// The mask arithmetic must be **numerically identical** to the historical
/// `stride_extent`-based arithmetic the closed `GroupKind` used
/// (`(rank / stride) % extent` group index, mixed-radix group id, members as
/// `base + j*stride`). The old formulas are transcribed here verbatim as the
/// reference implementation; this is the proof that the open vocabulary did
/// not change any number for the single-axis groups.
#[test]
fn single_axis_masks_match_the_old_stride_extent_arithmetic() {
    let configs = [
        cfg(4, 3, 2, 5, 2),
        cfg(2, 2, 2, 2, 2),
        cfg(1, 1, 1, 1, 1),
        cfg(8, 1, 1, 1, 1),
        cfg(1, 4, 1, 1, 1),
        cfg(1, 1, 3, 1, 1),
        cfg(1, 1, 1, 5, 1),
        cfg(1, 1, 1, 1, 7),
        cfg(3, 5, 1, 1, 1),
        cfg(1, 1, 1, 2, 2),
        cfg(2, 2, 1, 2, 1),
    ];
    for c in configs {
        c.validate().unwrap();
        let m = mesh(c);
        let world = c.world_size();
        assert_eq!(m.world_size(), world);
        // The old closed-form stride/extent per kind (group.rs before D3).
        let old = |axis: Option<usize>| -> (usize, usize) {
            let tp = c.tensor;
            let cp = c.context;
            let ep = c.expert;
            let dp = c.data;
            match axis {
                Some(0) => (1, tp),
                Some(1) => (tp, cp),
                Some(2) => (tp * cp, ep),
                Some(3) => (tp * cp * ep, dp),
                Some(4) => (tp * cp * ep * dp, c.pipeline),
                // `None` is the old `Global`: stride 1, extent = world.
                None => (1, world),
                _ => unreachable!(),
            }
        };
        for axis in [None, Some(0), Some(1), Some(2), Some(3), Some(4)] {
            let mask = match axis {
                None => GroupMask::from_bits(0b11111),
                Some(a) => single(a),
            };
            let (stride, extent) = old(axis);
            // The mesh strides agree with the old closed-form ones.
            if let Some(a) = axis {
                assert_eq!(m.stride(a), Some(stride), "{c:?} axis {a}");
                assert_eq!(m.degree(a), Some(extent), "{c:?} axis {a}");
            }
            assert_eq!(mask.degree(&m).unwrap(), extent, "{c:?} {axis:?}");

            let mut seen_ids = Vec::new();
            for rank in 0..world {
                // Old arithmetic, transcribed:
                let old_index = (rank / stride) % extent;
                let old_id = (rank / (stride * extent)) * stride + (rank % stride);
                let old_members: Vec<usize> = (0..extent)
                    .map(|j| {
                        let outer = old_id / stride;
                        let fastest = old_id % stride;
                        outer * stride * extent + fastest + j * stride
                    })
                    .collect();

                assert_eq!(
                    m.group_index(mask, rank).unwrap(),
                    old_index,
                    "{c:?} {axis:?} rank {rank}"
                );
                assert_eq!(
                    m.group_id(mask, rank).unwrap(),
                    old_id,
                    "{c:?} {axis:?} rank {rank}"
                );
                assert_eq!(
                    m.group_ranks(mask, rank).unwrap(),
                    old_members,
                    "{c:?} {axis:?} rank {rank}"
                );
                if !seen_ids.contains(&old_id) {
                    seen_ids.push(old_id);
                }
            }
            // Group ids appear in ascending order when ranks are walked
            // ascending: groups are ordered by their first member, which is the
            // ordering contract the old `ProcessGroups::groups(kind)` had.
            assert_eq!(
                seen_ids,
                (0..world / extent).collect::<Vec<usize>>(),
                "{c:?} {axis:?}"
            );
        }
    }
}
