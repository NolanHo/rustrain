//! `Display` forms: these strings appear in error messages and in plan dumps,
//! so they are part of the crate's surface even though nothing parses them.

use rustrain_parallel::{
    Collective, GroupKind, ParallelConfig, ParallelDim, ParallelLayout, RankLayout, ReduceOp,
};

#[test]
fn layouts_display_compactly() {
    assert_eq!(ParallelLayout::Replicate.to_string(), "replicate");
    assert_eq!(
        ParallelLayout::Shard {
            dim: -1,
            group: GroupKind::Tp
        }
        .to_string(),
        "shard(-1, tp)"
    );
    assert_eq!(
        ParallelLayout::Partial {
            op: ReduceOp::Sum,
            group: GroupKind::Tp
        }
        .to_string(),
        "partial(sum, tp)"
    );
    assert_eq!(
        ParallelLayout::ExpertShard {
            group: GroupKind::Ep
        }
        .to_string(),
        "expert(ep)"
    );
    assert_eq!(
        ParallelLayout::SequenceShard {
            group: GroupKind::Cp
        }
        .to_string(),
        "seq(cp)"
    );
}

#[test]
fn layouts_expose_their_group_and_replica_flag() {
    assert!(ParallelLayout::Replicate.is_replicated());
    assert_eq!(ParallelLayout::Replicate.group(), None);
    for layout in [
        ParallelLayout::Shard {
            dim: 0,
            group: GroupKind::Dp,
        },
        ParallelLayout::Partial {
            op: ReduceOp::Max,
            group: GroupKind::Dp,
        },
        ParallelLayout::ExpertShard {
            group: GroupKind::Dp,
        },
        ParallelLayout::SequenceShard {
            group: GroupKind::Dp,
        },
    ] {
        assert!(!layout.is_replicated());
        assert_eq!(layout.group(), Some(GroupKind::Dp));
    }
}

#[test]
fn collectives_display_compactly() {
    assert_eq!(
        Collective::AllReduce {
            group: GroupKind::Tp,
            op: ReduceOp::Sum
        }
        .to_string(),
        "all_reduce(sum, tp)"
    );
    assert_eq!(
        Collective::AllGather {
            group: GroupKind::Tp,
            dim: 3
        }
        .to_string(),
        "all_gather(tp, dim=3)"
    );
    assert_eq!(
        Collective::ReduceScatter {
            group: GroupKind::Cp,
            dim: 0
        }
        .to_string(),
        "reduce_scatter(cp, dim=0)"
    );
    assert_eq!(
        Collective::Broadcast {
            group: GroupKind::Dp,
            src_group_index: 0
        }
        .to_string(),
        "broadcast(dp, src=0)"
    );
}

#[test]
fn dims_ops_and_groups_display_lowercase() {
    assert_eq!(ParallelDim::Tp.to_string(), "tp");
    assert_eq!(ParallelDim::Ep.to_string(), "ep");
    assert_eq!(ReduceOp::Sum.to_string(), "sum");
    assert_eq!(ReduceOp::Min.to_string(), "min");
    assert_eq!(GroupKind::Global.to_string(), "global");
}

/// The digits are printed fastest-first. The config has `tp, cp, ep, dp` all
/// larger than 1 so that a reordered decomposition would print different
/// numbers, not the same ones in a different order.
#[test]
fn rank_layouts_display_fastest_dimension_first() {
    let c = ParallelConfig {
        tensor: 2,
        context: 2,
        expert: 2,
        data: 2,
        pipeline: 1,
    };
    // rank = 8*dp + 4*ep + 2*cp + tp
    assert_eq!(
        RankLayout::from_rank(4, c).unwrap().to_string(),
        "tp=0,cp=0,ep=1,dp=0,pp=0"
    );
    assert_eq!(
        RankLayout::from_rank(11, c).unwrap().to_string(),
        "tp=1,cp=1,ep=0,dp=1,pp=0"
    );
}
