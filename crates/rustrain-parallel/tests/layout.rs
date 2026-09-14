//! Shard-spec arithmetic: `divisor`, `local_shape`, multi-shard layouts, and
//! the D3 acceptance tests with the real Qwen3.6 shapes from
//! `docs/design/qwen36-5d-example.md`.

use rustrain_parallel::{GroupMask, Mesh, ParallelConfig, ParallelLayout, ReduceOp, ShardError};

fn cfg(tp: usize, cp: usize, ep: usize, dp: usize, pp: usize) -> ParallelConfig {
    ParallelConfig {
        tensor: tp,
        context: cp,
        expert: ep,
        data: dp,
        pipeline: pp,
    }
}

fn mesh(c: ParallelConfig) -> Mesh {
    Mesh::from_config(&c)
}

fn axis(mesh: &Mesh, name: &str) -> GroupMask {
    GroupMask::single(mesh.index_of(name).unwrap()).unwrap()
}

#[test]
fn constructors_and_accessors() {
    let rep = ParallelLayout::replicate();
    assert!(rep.is_replicated());
    assert_eq!(rep.dims, Vec::new());
    assert_eq!(rep.partial, None);
    assert_eq!(rep.shards(), &[] as &[rustrain_parallel::ShardSpec]);
    assert_eq!(rep.groups(), Vec::new());

    let tp = GroupMask::from_bits(0b1);
    let shard = ParallelLayout::shard(-1, tp);
    assert!(!shard.is_replicated());
    assert_eq!(
        shard.dims,
        vec![rustrain_parallel::ShardSpec::shard(-1, tp)]
    );
    assert_eq!(shard.groups(), vec![tp]);

    let partial = ParallelLayout::partial(ReduceOp::Sum, tp);
    assert!(!partial.is_replicated());
    assert_eq!(partial.groups(), vec![tp]);
}

/// Test 2 of D3 step 1: one tensor sharded dim 0 by `ep` and dim 1 by `tp`
/// simultaneously; `groups()` returns both masks; `local_shape` divides both.
#[test]
fn multi_shard_layout_shards_two_dims() {
    let m = mesh(cfg(2, 1, 4, 1, 1));
    let ep = axis(&m, "ep");
    let tp = axis(&m, "tp");

    let layout = ParallelLayout {
        dims: vec![
            rustrain_parallel::ShardSpec::shard(0, ep),
            rustrain_parallel::ShardSpec::shard(1, tp),
        ],
        partial: None,
    };
    assert!(!layout.is_replicated());
    assert_eq!(layout.groups(), vec![ep, tp]);
    assert_eq!(layout.shards().len(), 2);

    assert_eq!(layout.divisor(0, 2, &m).unwrap(), 4);
    assert_eq!(layout.divisor(1, 2, &m).unwrap(), 2);
    assert_eq!(layout.divisor(-2, 2, &m).unwrap(), 4); // -2 is dim 0
    assert_eq!(layout.divisor(-1, 2, &m).unwrap(), 2); // -1 is dim 1

    assert_eq!(layout.local_shape(&[256, 512], &m).unwrap(), vec![64, 256]);
    // The two shards are independent: each axis divides exactly once.
    assert_eq!(layout.local_shape(&[8, 6], &m).unwrap(), vec![2, 3]);
}

/// Test 3 of D3 step 1: the real Qwen3.6 MoE shape.
///
/// Numbers from `docs/design/qwen36-5d-example.md`: §2 declares the global
/// shape `experts.gate_up_proj [256, 1024, 2048]` = `[E, 2*512, H]`, §3 splits
/// it into a gate segment `[E, 512, H]` = `[256, 512, 2048]`, and §5's
/// walkthrough (`tp=2, cp=2, ep=4, dp=2, pp=2`) shards dim 0 over `{ep}` and
/// the gate/intermediate dim over `{tp}` (slot orientation in the §5.1 table,
/// checkpoint orientation here — the transposed spelling of the same fact).
/// Local: `[256/4, 512/2, 2048]` = `[64, 256, 2048]`.
#[test]
fn d3_acceptance_qwen36_gate_up_proj_local_shape() {
    let m = mesh(cfg(2, 2, 4, 2, 2));
    let ep = axis(&m, "ep");
    let tp = axis(&m, "tp");

    let gate_segment = ParallelLayout {
        dims: vec![
            rustrain_parallel::ShardSpec::shard(0, ep),
            rustrain_parallel::ShardSpec::shard(1, tp),
        ],
        partial: None,
    };

    // Global gate segment [E=256, I=512, H=2048] (doc §2 + §3 split).
    let global = [256i64, 512, 2048];
    assert_eq!(
        gate_segment.local_shape(&global, &m).unwrap(),
        [64, 256, 2048]
    );
    assert_eq!(gate_segment.divisor(0, 3, &m).unwrap(), 4);
    assert_eq!(gate_segment.divisor(1, 3, &m).unwrap(), 2);
    assert_eq!(gate_segment.divisor(2, 3, &m).unwrap(), 1);
}

/// Test 4 of D3 step 1: non-divisibility is a compile-time error, not a
/// panic, not a fallback, not a runtime check.
///
/// `docs/design/qwen36-5d-example.md` §4: `num_attention_heads % tp == 0` is a
/// mechanical L1 constraint — with `tp=3` and `num_attention_heads=16` it is
/// violated, and `local_shape`/`divisor` must say so, naming the dim, the
/// global size and the divisor.
#[test]
fn d3_acceptance_non_divisible_shard_is_a_compile_time_error() {
    let m = mesh(cfg(3, 1, 1, 1, 1));
    let tp = axis(&m, "tp");

    // 16 attention heads sharded over tp=3: 16 % 3 != 0.
    let heads = ParallelLayout::shard(0, tp);
    assert_eq!(heads.divisor(0, 1, &m).unwrap(), 3);
    let err = heads.local_shape(&[16], &m).unwrap_err();
    assert_eq!(
        err,
        ShardError::NotDivisible {
            dim: 0,
            global: 16,
            divisor: 3
        }
    );
    // The message names the dim, the global size and the divisor.
    let message = err.to_string();
    assert!(message.contains("dim 0"), "{message}");
    assert!(message.contains("16"), "{message}");
    assert!(message.contains("3"), "{message}");
    assert!(message.contains("compile-time"), "{message}");
}

#[test]
fn shards_on_the_same_dim_compound() {
    // HSDP-style: dim 0 sharded over tp *and* ep — the divisor is the product.
    let m = mesh(cfg(2, 1, 4, 1, 1));
    let ep = axis(&m, "ep");
    let tp = axis(&m, "tp");
    let layout = ParallelLayout {
        dims: vec![
            rustrain_parallel::ShardSpec::shard(0, ep),
            rustrain_parallel::ShardSpec::shard(0, tp),
        ],
        partial: None,
    };
    assert_eq!(layout.divisor(0, 1, &m).unwrap(), 8);
    assert_eq!(layout.local_shape(&[64], &m).unwrap(), vec![8]);
    // Non-divisible by the compound divisor: same hard error.
    assert_eq!(
        layout.local_shape(&[9], &m).unwrap_err(),
        ShardError::NotDivisible {
            dim: 0,
            global: 9,
            divisor: 8
        }
    );
}

#[test]
fn negative_dims_are_resolved_by_local_shape() {
    let m = mesh(cfg(2, 1, 1, 1, 1));
    let tp = axis(&m, "tp");
    let layout = ParallelLayout::shard(-1, tp);
    assert_eq!(layout.local_shape(&[8, 4], &m).unwrap(), vec![8, 2]);
}

#[test]
fn a_broken_shard_is_reported_even_when_it_is_not_the_dim_queried() {
    let m = mesh(cfg(2, 1, 1, 1, 1));
    let tp = axis(&m, "tp");
    // Axis 5 of a rank-2 tensor is a plan bug regardless of which dim is
    // queried — same discipline as the transition rules.
    let layout = ParallelLayout::shard(5, tp);
    assert_eq!(
        layout.divisor(0, 2, &m),
        Err(ShardError::DimOutOfRange { dim: 5, rank: 2 })
    );
    // ... and it is reported even for a scalar, where no divisor is computed.
    assert_eq!(
        layout.local_shape(&[], &m),
        Err(ShardError::DimOutOfRange { dim: 5, rank: 0 })
    );
    // An invalid tensor rank is reported by `divisor` itself.
    assert_eq!(
        ParallelLayout::replicate().divisor(0, -1, &m),
        Err(ShardError::InvalidTensorRank { rank: -1 })
    );
}

#[test]
fn a_mask_from_another_mesh_is_rejected() {
    let m = Mesh::new(vec![("a".to_string(), 2), ("b".to_string(), 2)]).unwrap(); // two axes
    let stray = GroupMask::from_bits(0b100); // bit 2: no such axis
    let layout = ParallelLayout::shard(0, stray);
    assert_eq!(
        layout.divisor(0, 1, &m),
        Err(ShardError::GroupOutOfRange { bit: 2, axes: 2 })
    );
    assert_eq!(
        layout.local_shape(&[4], &m),
        Err(ShardError::GroupOutOfRange { bit: 2, axes: 2 })
    );
    // The partial's group is validated too, even though it has no dim.
    let partial = ParallelLayout::partial(ReduceOp::Sum, stray);
    assert_eq!(
        partial.local_shape(&[4], &m),
        Err(ShardError::GroupOutOfRange { bit: 2, axes: 2 })
    );
}

#[test]
fn describe_renders_names_via_the_mesh() {
    let m = mesh(cfg(2, 2, 4, 2, 2));
    let ep = axis(&m, "ep");
    let tp = axis(&m, "tp");
    let all = tp
        .union(axis(&m, "cp"))
        .union(ep)
        .union(axis(&m, "dp"))
        .union(axis(&m, "pp"));

    assert_eq!(ParallelLayout::replicate().describe(&m), "replicate");
    assert_eq!(ParallelLayout::shard(-1, tp).describe(&m), "shard(-1, tp)");
    assert_eq!(
        ParallelLayout::partial(ReduceOp::Sum, tp).describe(&m),
        "partial(sum, tp)"
    );
    let multi = ParallelLayout {
        dims: vec![
            rustrain_parallel::ShardSpec::shard(0, ep),
            rustrain_parallel::ShardSpec::shard(1, tp),
        ],
        partial: None,
    };
    assert_eq!(multi.describe(&m), "shard(0, ep) + shard(1, tp)");
    let with_partial = ParallelLayout {
        dims: vec![rustrain_parallel::ShardSpec::shard(0, ep)],
        partial: Some(rustrain_parallel::PartialSpec {
            op: ReduceOp::Sum,
            group: tp,
        }),
    };
    assert_eq!(with_partial.describe(&m), "shard(0, ep) + partial(sum, tp)");
    assert_eq!(
        ParallelLayout::shard(0, all).describe(&m),
        "shard(0, global)"
    );
    assert_eq!(
        ParallelLayout::shard(0, GroupMask::NONE).describe(&m),
        "shard(0, none)"
    );

    // Without names (no mesh) the `Display` form renders raw bits.
    assert_eq!(
        multi.to_string(),
        "shard(0, mask(0b100)) + shard(1, mask(0b1))"
    );
    assert_eq!(ParallelLayout::replicate().to_string(), "replicate");
    // A bit outside the mesh has no name: describe falls back to the bit form.
    let stray = GroupMask::from_bits(1 << 5);
    assert_eq!(
        ParallelLayout::shard(0, stray).describe(&m),
        "shard(0, mask(0b100000))"
    );
}

/// Serialization forms, pinned by ruling 4: masks as `u32` bits, layouts as
/// `{"dims":[..],"partial":..|null}` — deterministic, never a map.
#[test]
fn serialization_forms_are_pinned() {
    let m = mesh(cfg(2, 2, 1, 1, 1));
    let ep = axis(&m, "ep");
    let tp = axis(&m, "tp");

    let layout = ParallelLayout {
        dims: vec![
            rustrain_parallel::ShardSpec::shard(0, ep),
            rustrain_parallel::ShardSpec::shard(-1, tp),
        ],
        partial: Some(rustrain_parallel::PartialSpec {
            op: ReduceOp::Sum,
            group: GroupMask::NONE,
        }),
    };
    // ep is axis 2 (bit 0b100 = 4), tp axis 0 (1).
    let json = serde_json::to_string(&layout).unwrap();
    assert_eq!(
        json,
        r#"{"dims":[{"dim":0,"group":4},{"dim":-1,"group":1}],"partial":{"op":"sum","group":0}}"#
    );
    let back: ParallelLayout = serde_json::from_str(&json).unwrap();
    assert_eq!(back, layout);

    let rep = ParallelLayout::replicate();
    assert_eq!(
        serde_json::to_string(&rep).unwrap(),
        r#"{"dims":[],"partial":null}"#
    );
    let back_rep: ParallelLayout = serde_json::from_str(r#"{"dims":[],"partial":null}"#).unwrap();
    assert_eq!(back_rep, rep);
}

/// A declared replicating shard: `docs/design/qwen36-text/spec.md` §D6.6. The axis is sharded in
/// *units*, a rank owns the units it needs, and two ranks may hold the same one — which is how a
/// tensor-parallel attention keeps its key/value heads when there are fewer heads than ranks.
#[test]
fn a_replicating_shard_hands_each_rank_the_units_it_needs() {
    use rustrain_parallel::{ShardMode, ShardSpec};

    // The head case: 512 features of 256 (two heads) over four ranks. A single-element unit would
    // give a rank half a head; the unit is why it does not.
    let mode = ShardMode::Replicate { unit: 256 };
    let slabs: Vec<(i64, i64)> = (0..4).map(|c| mode.slab(512, c, 4).unwrap()).collect();
    assert_eq!(
        slabs,
        vec![(0, 256), (0, 256), (256, 256), (256, 256)],
        "two heads, four ranks: each rank gets one, the last two share"
    );

    // Fewer units than ranks (4 units, 8 ranks): one unit each, overlapping.
    let slabs: Vec<(i64, i64)> = (0..8).map(|c| mode.slab(1024, c, 8).unwrap()).collect();
    assert_eq!(
        slabs,
        vec![
            (0, 256),
            (0, 256),
            (256, 256),
            (256, 256),
            (512, 256),
            (512, 256),
            (768, 256),
            (768, 256)
        ]
    );

    // Every element belongs to some rank, and no slab cuts a unit in half.
    let covered: std::collections::BTreeSet<i64> =
        slabs.iter().flat_map(|(off, len)| *off..*off + *len).collect();
    assert_eq!(covered.len(), 1024, "the slabs must cover the axis");

    // A unit that does not divide the axis is refused, not rounded: 512 with units of 3.
    assert_eq!(ShardMode::Replicate { unit: 3 }.slab(512, 0, 2), None);
    // …and a strict shard is unchanged: 16 heads over 4 ranks.
    assert_eq!(ShardMode::Divide.slab(16, 1, 4), Some((4, 4)));
    assert_eq!(
        ShardMode::Divide.slab(2, 1, 4),
        None,
        "an undeclared undersized axis stays a hard error"
    );

    // Through a layout: the local shape follows the mode, and the slab is what the loader takes.
    let m = mesh(cfg(4, 1, 1, 1, 1));
    let tp = axis(&m, "tp");
    let strict = ParallelLayout {
        dims: vec![ShardSpec::shard(1, tp)],
        partial: None,
    };
    assert!(matches!(
        strict.local_shape(&[4, 2], &m),
        Err(ShardError::NotDivisible { dim: 1, global: 2, divisor: 4 })
    ));
    let replicating = ParallelLayout {
        dims: vec![ShardSpec::replicating(1, tp, 1)],
        partial: None,
    };
    assert_eq!(replicating.local_shape(&[4, 2], &m).unwrap(), vec![4, 1]);
    assert_eq!(replicating.slab(2, 1, 2, 4, 2).unwrap(), (1, 1));
    assert_eq!(replicating.slab(2, 1, 1, 4, 2).unwrap(), (0, 1));
    assert_eq!(replicating.slab(2, 1, 0, 4, 2).unwrap(), (0, 1));
}
