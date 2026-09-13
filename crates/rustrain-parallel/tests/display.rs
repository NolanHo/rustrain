//! `Display` forms: these strings appear in error messages and in plan dumps,
//! so they are part of the crate's surface even though nothing parses them.

use rustrain_parallel::{
    Collective, GroupMask, Mesh, ParallelConfig, ParallelDim, ParallelLayout, RankLayout, ReduceOp,
};

#[test]
fn layouts_display_compactly() {
    // Without a mesh, masks render as their raw bits — the name lives in the
    // mesh, and `Display` has no mesh to ask.
    assert_eq!(ParallelLayout::replicate().to_string(), "replicate");
    assert_eq!(
        ParallelLayout::shard(-1, GroupMask::from_bits(0b1)).to_string(),
        "shard(-1, mask(0b1))"
    );
    assert_eq!(
        ParallelLayout::partial(ReduceOp::Sum, GroupMask::from_bits(0b1)).to_string(),
        "partial(sum, mask(0b1))"
    );
    let multi = ParallelLayout {
        dims: vec![
            rustrain_parallel::ShardSpec {
                dim: 0,
                group: GroupMask::from_bits(0b100),
            },
            rustrain_parallel::ShardSpec {
                dim: 1,
                group: GroupMask::from_bits(0b1),
            },
        ],
        partial: None,
    };
    assert_eq!(
        multi.to_string(),
        "shard(0, mask(0b100)) + shard(1, mask(0b1))"
    );
}

#[test]
fn describe_renders_names_from_the_mesh() {
    let mesh = Mesh::from_config(&ParallelConfig {
        tensor: 2,
        context: 2,
        expert: 2,
        data: 2,
        pipeline: 2,
    });
    let tp = GroupMask::from_bits(0b1);
    let ep = GroupMask::from_bits(0b100);
    assert_eq!(
        ParallelLayout::shard(-1, tp).describe(&mesh),
        "shard(-1, tp)"
    );
    assert_eq!(
        ParallelLayout::partial(ReduceOp::Sum, tp).describe(&mesh),
        "partial(sum, tp)"
    );
    let multi = ParallelLayout {
        dims: vec![
            rustrain_parallel::ShardSpec { dim: 0, group: ep },
            rustrain_parallel::ShardSpec { dim: 1, group: tp },
        ],
        partial: None,
    };
    assert_eq!(multi.describe(&mesh), "shard(0, ep) + shard(1, tp)");
    assert_eq!(ParallelLayout::replicate().describe(&mesh), "replicate");
}

#[test]
fn layouts_expose_their_groups_and_replica_flag() {
    assert!(ParallelLayout::replicate().is_replicated());
    assert_eq!(ParallelLayout::replicate().groups(), Vec::new());

    let shard = ParallelLayout::shard(0, GroupMask::from_bits(0b1000));
    assert!(!shard.is_replicated());
    assert_eq!(shard.groups(), vec![GroupMask::from_bits(0b1000)]);

    let partial = ParallelLayout::partial(ReduceOp::Max, GroupMask::from_bits(0b1000));
    assert!(!partial.is_replicated());
    assert_eq!(partial.groups(), vec![GroupMask::from_bits(0b1000)]);

    // Distinct groups, in declaration order, duplicates dropped.
    let combined = ParallelLayout {
        dims: vec![
            rustrain_parallel::ShardSpec {
                dim: 0,
                group: GroupMask::from_bits(0b1),
            },
            rustrain_parallel::ShardSpec {
                dim: 1,
                group: GroupMask::from_bits(0b100),
            },
            rustrain_parallel::ShardSpec {
                dim: 2,
                group: GroupMask::from_bits(0b1),
            },
        ],
        partial: Some(rustrain_parallel::PartialSpec {
            op: ReduceOp::Sum,
            group: GroupMask::from_bits(0b100),
        }),
    };
    assert_eq!(
        combined.groups(),
        vec![GroupMask::from_bits(0b1), GroupMask::from_bits(0b100),]
    );
}

#[test]
fn collectives_display_compactly() {
    assert_eq!(
        Collective::AllReduce {
            group: GroupMask::from_bits(0b1),
            op: ReduceOp::Sum
        }
        .to_string(),
        "all_reduce(sum, mask(0b1))"
    );
    assert_eq!(
        Collective::AllGather {
            group: GroupMask::from_bits(0b1),
            dim: 3
        }
        .to_string(),
        "all_gather(mask(0b1), dim=3)"
    );
    assert_eq!(
        Collective::ReduceScatter {
            group: GroupMask::from_bits(0b10),
            dim: 0
        }
        .to_string(),
        "reduce_scatter(mask(0b10), dim=0)"
    );
    assert_eq!(
        Collective::Broadcast {
            group: GroupMask::from_bits(0b1000),
            src_group_index: 0
        }
        .to_string(),
        "broadcast(mask(0b1000), src=0)"
    );
    // The group accessor returns the mask the collective runs over.
    assert_eq!(
        Collective::AllGather {
            group: GroupMask::from_bits(0b100),
            dim: 0
        }
        .group(),
        GroupMask::from_bits(0b100)
    );
}

#[test]
fn dims_ops_and_masks_display_lowercase() {
    assert_eq!(ParallelDim::Tp.to_string(), "tp");
    assert_eq!(ParallelDim::Ep.to_string(), "ep");
    assert_eq!(ReduceOp::Sum.to_string(), "sum");
    assert_eq!(ReduceOp::Min.to_string(), "min");
    assert_eq!(GroupMask::from_bits(0b101).to_string(), "mask(0b101)");
    assert_eq!(GroupMask::NONE.to_string(), "mask(0b0)");
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
