//! D4 acceptance on the plan side: `instantiate` on a hand-built global plan, then the
//! existing compile path — a row-parallel weight's declared shard makes its output a
//! `partial(sum, tp)`, and `Compiler::compile` on the instantiated plan inserts an
//! `all_reduce` whose mask is `tp` (the existing behaviour, re-proved on an instantiated plan).

use std::collections::BTreeMap;

use rustrain_abi::Plugin;
use rustrain_abi::ffi::RsDtype;
use rustrain_ops::{Phase, Recipe, Registry, TargetEnv};
use rustrain_parallel::{GroupMask, Mesh, ParallelConfig, ParallelLayout, ReduceOp, ShardSpec};
use rustrain_plan::ir::intrinsic;
use rustrain_plan::{
    Attrs, DeclaredAxes, InstanceStage, OpRef, Plan, PlanBuilder, PlanError, SlotKind, instantiate,
    instantiate_stages,
};

/// The canonical five-axis mesh with `tp = 2`; `tp` is the first axis, so its mask is bit 0.
fn tp_mesh() -> Mesh {
    Mesh::from_config(&ParallelConfig {
        tensor: 2,
        ..Default::default()
    })
}

fn tp_mask(mesh: &Mesh) -> GroupMask {
    GroupMask::single(mesh.index_of("tp").unwrap()).unwrap()
}

/// The registry the CLI builds: the built-in reference provider only.
fn reference_registry() -> Registry {
    let mut registry = Registry::new();
    // SAFETY: the built-in descriptors are leaked by `PluginBuilder`, so they live as long as
    // the process.
    let builtin = unsafe { Plugin::from_static(rustrain_kernels::plugin(), "<built-in>") }
        .expect("the built-in reference provider is ABI-valid");
    registry
        .add_plugin(builtin)
        .expect("the built-in provider registers");
    registry
}

/// The real MLP shape around every row-parallel projection: a column-parallel up-projection
/// shards the activation's feature dim (`h`), which is exactly the contraction dim of the
/// row-parallel down-projection — so the local shapes pair up ([4, 4] @ [4, 8]), the down
/// projection's output is a `partial(sum, tp)`, and the replicated residual path (`x`) forces
/// the all-reduce at the final add.
#[test]
fn a_row_parallel_declared_shard_yields_partial_then_all_reduce() {
    // The global plan, exactly as `expand` produces one: every layout replicated.
    let mut b = PlanBuilder::new(
        "mlp",
        Phase::Forward,
        Mesh::from_config(&ParallelConfig::default()).fingerprint(),
    );
    let x = b.slot("x", RsDtype::F32, vec![4, 8], SlotKind::Input);
    let w1 = b.slot("w1", RsDtype::F32, vec![8, 8], SlotKind::Weight);
    let h = b.slot("h", RsDtype::F32, vec![4, 8], SlotKind::Activation);
    let w2 = b.slot("w2", RsDtype::F32, vec![8, 8], SlotKind::Weight);
    let y = b.slot("y", RsDtype::F32, vec![4, 8], SlotKind::Activation);
    let out = b.slot("out", RsDtype::F32, vec![4, 8], SlotKind::Activation);
    b.node(
        OpRef::new("linear"),
        vec![x, w1],
        vec![h],
        Attrs::new(),
        "l0.up",
    );
    b.node(
        OpRef::new("linear"),
        vec![h, w2],
        vec![y],
        Attrs::new(),
        "l0.down",
    );
    b.node(
        OpRef::new("elementwise_binary"),
        vec![x, y],
        vec![out],
        Attrs::new().set("kind", "add"),
        "l0.residual",
    );
    let global = b.build().unwrap();

    // The description declares `w1` column-parallel (output dim over tp) and `w2` row-parallel
    // (contraction dim over tp). `pp` is 1, so no stage declarations are needed.
    let declared = DeclaredAxes {
        slots: BTreeMap::from([
            (
                "w1".to_string(),
                BTreeMap::from([("1".to_string(), vec!["tp".to_string()])]),
            ),
            (
                "w2".to_string(),
                BTreeMap::from([("0".to_string(), vec!["tp".to_string()])]),
            ),
        ]),
        instances: Vec::new(),
    };
    let mesh = tp_mesh();
    let tp = tp_mask(&mesh);

    let instantiated =
        instantiate(&global, &declared, &mesh, 0, &reference_registry()).expect("instantiate");

    // The column-parallel shard propagates to the up-projection's output: sharded features.
    let h_id = instantiated.slot_id("h").expect("h survives");
    assert_eq!(
        instantiated.slot(h_id).layout,
        ParallelLayout::shard(1, tp),
        "the column-parallel weight shards the output's feature dim (canonical spelling: \
         instantiate resolves every dim against the slot's rank)"
    );
    assert_eq!(
        instantiated.slot(h_id).shape,
        vec![4, 4],
        "local features = 8 / tp"
    );
    // The declared row shard makes the down-projection's output a partial sum over tp.
    let y_id = instantiated.slot_id("y").expect("y survives");
    assert_eq!(
        instantiated.slot(y_id).layout,
        ParallelLayout::partial(ReduceOp::Sum, tp),
        "the row-parallel weight's shard must make the output a partial(sum, tp)"
    );
    // A partial does not divide any dim: the local shape is the global one.
    assert_eq!(instantiated.slot(y_id).shape, vec![4, 8]);

    // The existing compile path: the residual add's first input is replicated, so its second
    // input (the partial) is converted by an all_reduce over tp.
    let registry = reference_registry();
    let recipe = Recipe::from_toml("[kernel]\ndefault = \"reference\"\n").unwrap();
    let compiled = rustrain_plan::Compiler::new(&registry, &recipe, TargetEnv::default())
        .compile(&instantiated)
        .expect("the instantiated plan compiles");

    let all_reduces: Vec<_> = compiled
        .inserted
        .iter()
        .filter(|inserted| inserted.op == intrinsic::ALL_REDUCE)
        .collect();
    assert_eq!(all_reduces.len(), 1, "exactly one all_reduce is inserted");
    assert_eq!(all_reduces[0].group, tp, "the all_reduce is over tp");
    assert_eq!(all_reduces[0].reduce, Some(ReduceOp::Sum));
}

/// A rank outside the mesh is reported, not wrapped.
#[test]
fn a_rank_outside_the_mesh_is_reported() {
    let mut b = PlanBuilder::new(
        "tiny",
        Phase::Forward,
        Mesh::from_config(&ParallelConfig::default()).fingerprint(),
    );
    let x = b.slot("x", RsDtype::F32, vec![4], SlotKind::Input);
    let y = b.slot("y", RsDtype::F32, vec![4], SlotKind::Activation);
    b.node(
        OpRef::new("elementwise_unary"),
        vec![x],
        vec![y],
        Attrs::new().set("kind", "silu"),
        "act",
    );
    let global = b.build().unwrap();

    let mesh = tp_mesh(); // world size 2
    let declared = DeclaredAxes {
        slots: BTreeMap::new(),
        instances: Vec::new(),
    };
    match instantiate(&global, &declared, &mesh, 2, &reference_registry()).unwrap_err() {
        PlanError::RankOutOfRange { rank, world_size } => {
            assert_eq!(rank, 2);
            assert_eq!(world_size, 2);
        }
        other => panic!("expected RankOutOfRange, got {other:?}"),
    }
}

/// A declaration naming a slot the plan does not have is reported, not skipped.
#[test]
fn a_declaration_naming_an_unknown_slot_is_reported() {
    let mut b = PlanBuilder::new(
        "tiny",
        Phase::Forward,
        Mesh::from_config(&ParallelConfig::default()).fingerprint(),
    );
    let x = b.slot("x", RsDtype::F32, vec![4], SlotKind::Input);
    let y = b.slot("y", RsDtype::F32, vec![4], SlotKind::Activation);
    b.node(
        OpRef::new("elementwise_unary"),
        vec![x],
        vec![y],
        Attrs::new().set("kind", "silu"),
        "act",
    );
    let global = b.build().unwrap();

    let declared = DeclaredAxes {
        slots: BTreeMap::from([(
            "no.such.slot".to_string(),
            BTreeMap::from([("0".to_string(), vec!["tp".to_string()])]),
        )]),
        instances: Vec::new(),
    };
    match instantiate(&global, &declared, &tp_mesh(), 0, &reference_registry()).unwrap_err() {
        PlanError::UnknownDeclaredSlot { slot } => assert_eq!(slot, "no.such.slot"),
        other => panic!("expected UnknownDeclaredSlot, got {other:?}"),
    }
}

/// With `pp = 2` a stage outside `0..2` is reported naming the entry and the degree.
#[test]
fn a_stage_outside_the_pipeline_degree_is_reported() {
    let mut b = PlanBuilder::new(
        "tiny",
        Phase::Forward,
        Mesh::from_config(&ParallelConfig::default()).fingerprint(),
    );
    let x = b.slot("x", RsDtype::F32, vec![4], SlotKind::Input);
    let y = b.slot("y", RsDtype::F32, vec![4], SlotKind::Activation);
    b.node(
        OpRef::new("elementwise_unary"),
        vec![x],
        vec![y],
        Attrs::new().set("kind", "silu"),
        "act",
    );
    let global = b.build().unwrap();

    let declared = DeclaredAxes {
        slots: BTreeMap::new(),
        instances: vec![rustrain_plan::InstanceStage {
            prefix: "act".to_string(),
            stage: Some(5),
        }],
    };
    let mesh = Mesh::from_config(&ParallelConfig {
        pipeline: 2,
        ..Default::default()
    });
    match instantiate(&global, &declared, &mesh, 0, &reference_registry()).unwrap_err() {
        PlanError::StageOutOfRange { prefix, stage, pp } => {
            assert_eq!(prefix, "act");
            assert_eq!(stage, 5);
            assert_eq!(pp, 2);
        }
        other => panic!("expected StageOutOfRange, got {other:?}"),
    }
}

/// With `pp = 2` a node whose instance declares no stage is refused — the same missing-stage
/// contract, attributed to the node's trace path.
#[test]
fn pp_above_one_without_an_instance_stage_is_reported() {
    let mut b = PlanBuilder::new(
        "tiny",
        Phase::Forward,
        Mesh::from_config(&ParallelConfig::default()).fingerprint(),
    );
    let x = b.slot("x", RsDtype::F32, vec![4], SlotKind::Input);
    let y = b.slot("y", RsDtype::F32, vec![4], SlotKind::Activation);
    b.node(
        OpRef::new("elementwise_unary"),
        vec![x],
        vec![y],
        Attrs::new().set("kind", "silu"),
        "act",
    );
    let global = b.build().unwrap();

    // No instances declared at all: every node's instance is unknown, and with pp > 1 that is
    // the "declares no stage" error (R1: absent stage means stage 0 only when pp == 1).
    let declared = DeclaredAxes {
        slots: BTreeMap::new(),
        instances: Vec::new(),
    };
    let mesh = Mesh::from_config(&ParallelConfig {
        pipeline: 2,
        ..Default::default()
    });
    match instantiate(&global, &declared, &mesh, 0, &reference_registry()).unwrap_err() {
        PlanError::MissingStage { pp, .. } => assert_eq!(pp, 2),
        other => panic!("expected MissingStage, got {other:?}"),
    }
}

/// **Reviewer C1 (HIGH), instantiate level.** `a = [B,S,K] shard(0, tp)` and `b = [K,N]
/// shard(1, ep)` both survive the matmul: the output carries `{shard(0, tp), shard(2, ep)}`
/// and the local shape divides the batch axis too — `[1, 8, 2]`, not the over-claimed
/// `[2, 8, 2]` the dropped shard used to produce.
#[test]
fn a_matmul_keeps_the_batch_shard_and_the_output_shard_in_the_local_shape() {
    let mesh = Mesh::new(vec![("tp".to_string(), 2), ("ep".to_string(), 2)]).unwrap();
    let mut b = PlanBuilder::new("mm", Phase::Forward, mesh.fingerprint());
    let a = b.slot("a", RsDtype::F32, vec![2, 8, 16], SlotKind::Activation);
    let w = b.slot("b", RsDtype::F32, vec![16, 4], SlotKind::Weight);
    let y = b.slot("y", RsDtype::F32, vec![2, 8, 4], SlotKind::Activation);
    b.node(
        OpRef::new("matmul"),
        vec![a, w],
        vec![y],
        Attrs::new(),
        "mm",
    );
    let global = b.build().unwrap();

    let declared = DeclaredAxes {
        slots: BTreeMap::from([
            (
                "a".to_string(),
                BTreeMap::from([("0".to_string(), vec!["tp".to_string()])]),
            ),
            (
                "b".to_string(),
                BTreeMap::from([("1".to_string(), vec!["ep".to_string()])]),
            ),
        ]),
        instances: Vec::new(),
    };
    let tp = GroupMask::single(mesh.index_of("tp").unwrap()).unwrap();
    let ep = GroupMask::single(mesh.index_of("ep").unwrap()).unwrap();
    let instantiated =
        instantiate(&global, &declared, &mesh, 0, &reference_registry()).expect("instantiate");

    let y = instantiated.slot_id("y").expect("y survives");
    assert_eq!(
        instantiated.slot(y).layout,
        ParallelLayout {
            dims: vec![
                ShardSpec { dim: 0, group: tp },
                ShardSpec { dim: 2, group: ep },
            ],
            partial: None,
        },
        "both surviving shards ride the output"
    );
    assert_eq!(
        instantiated.slot(y).shape,
        vec![1, 8, 2],
        "the local shape divides the batch axis by tp and the output axis by ep"
    );
}

/// **Reviewer C2 (HIGH), instantiate level.** `broadcast` of a rank-1 `[H]` sharded
/// `shard(0, tp)` to `[S, H]` must shard the output's *feature* axis: `shard(1, tp)`, local
/// shape `[S, H/2]`. The old rule renamed the shard onto the sequence axis.
#[test]
fn a_rank_growing_broadcast_shards_the_output_feature_axis() {
    let mesh = Mesh::from_config(&ParallelConfig {
        tensor: 2,
        ..Default::default()
    });
    let mut b = PlanBuilder::new("bc", Phase::Forward, mesh.fingerprint());
    let h = b.slot("h", RsDtype::F32, vec![8], SlotKind::Activation);
    let y = b.slot("y", RsDtype::F32, vec![4, 8], SlotKind::Activation);
    b.node(
        OpRef::new("broadcast"),
        vec![h],
        vec![y],
        Attrs::new(),
        "bc",
    );
    let global = b.build().unwrap();

    let declared = DeclaredAxes {
        slots: BTreeMap::from([(
            "h".to_string(),
            BTreeMap::from([("0".to_string(), vec!["tp".to_string()])]),
        )]),
        instances: Vec::new(),
    };
    let tp = GroupMask::single(mesh.index_of("tp").unwrap()).unwrap();
    let instantiated =
        instantiate(&global, &declared, &mesh, 0, &reference_registry()).expect("instantiate");
    let y = instantiated.slot_id("y").expect("y survives");
    assert_eq!(
        instantiated.slot(y).layout,
        ParallelLayout::shard(1, tp),
        "the shard rides the output's trailing (feature) axis, not the sequence axis"
    );
    assert_eq!(
        instantiated.slot(y).shape,
        vec![4, 4],
        "the local feature extent is H/2; the sequence axis is untouched"
    );
}

/// A two-stage global plan — `pre` and `post`, one linear each — whose `post.w` weight is
/// declared `{0: tp}` by [`staged_declarations`].
fn staged_plan(pre_w: Vec<i64>, post_w: Vec<i64>) -> Plan {
    let mesh = Mesh::from_config(&ParallelConfig {
        tensor: 2,
        pipeline: 2,
        ..Default::default()
    });
    let mut b = PlanBuilder::new("staged", Phase::Forward, mesh.fingerprint());
    let x = b.slot("x", RsDtype::F32, vec![8, 8], SlotKind::Input);
    let pre_w = b.slot("pre.w", RsDtype::F32, pre_w, SlotKind::Weight);
    let pre_y = b.slot("pre.y", RsDtype::F32, vec![8, 8], SlotKind::Activation);
    let post_w = b.slot("post.w", RsDtype::F32, post_w, SlotKind::Weight);
    let post_y = b.slot("post.y", RsDtype::F32, vec![8, 8], SlotKind::Activation);
    b.node(
        OpRef::new("linear"),
        vec![x, pre_w],
        vec![pre_y],
        Attrs::new(),
        "pre",
    );
    b.node(
        OpRef::new("linear"),
        vec![pre_y, post_w],
        vec![post_y],
        Attrs::new(),
        "post",
    );
    b.build().unwrap()
}

fn staged_declarations(pre_stage: i64, post_stage: i64) -> DeclaredAxes {
    DeclaredAxes {
        slots: BTreeMap::from([(
            "post.w".to_string(),
            BTreeMap::from([("0".to_string(), vec!["tp".to_string()])]),
        )]),
        instances: vec![
            InstanceStage {
                prefix: "pre".to_string(),
                stage: Some(pre_stage),
            },
            InstanceStage {
                prefix: "post".to_string(),
                stage: Some(post_stage),
            },
        ],
    }
}

/// **Reviewer C3 (MEDIUM), plan level.** A stage-1-only non-divisible shard must be reported:
/// stage 0 instantiates clean while stage 1's `post.w [9, 8]` sharded `{0: tp}` does not
/// divide by 2. Checking only rank 0 (stage 0) masked exactly this.
#[test]
fn a_stage_one_only_non_divisible_shard_is_reported_per_stage() {
    let global = staged_plan(vec![8, 8], vec![9, 8]);
    let declared = staged_declarations(0, 1);
    let mesh = Mesh::from_config(&ParallelConfig {
        tensor: 2,
        pipeline: 2,
        ..Default::default()
    });

    let results = instantiate_stages(&global, &declared, &mesh, &reference_registry());
    assert_eq!(results.len(), 2, "one representative rank per pp stage");
    let stage0 = &results[0];
    assert_eq!(stage0.stage, 0);
    assert_eq!(stage0.rank, 0);
    assert!(
        stage0.result.is_ok(),
        "stage 0 is clean; its plan must not carry the stage-1 failure: {:?}",
        stage0.result
    );
    let stage1 = &results[1];
    assert_eq!(stage1.stage, 1);
    assert_eq!(stage1.rank, 2, "rank 2 = stage 1 * stride(tp=2)");
    match &stage1.result {
        Err(PlanError::Instantiate { slot, source }) => {
            assert_eq!(slot, "post.w", "the failure names the stage-1 slot");
            let text = format!("{source}");
            assert!(
                text.contains("dim 0") && text.contains("9") && text.contains("divisor 2"),
                "the failure names the constraint: {text}"
            );
        }
        other => panic!("stage 1 must carry the divisibility failure, got {other:?}"),
    }
}

/// **Reviewer C4 (MEDIUM), plan level.** A stage that owns no work: with every instance on
/// stage 0 and `pp = 2`, stage 1 instantiates to a plan with **no nodes** — and
/// `check_structure` accepts it, so the L1 item must detect the empty stage itself rather
/// than report a clean bill of health for it.
#[test]
fn a_stage_with_no_instances_instantiates_to_an_empty_plan() {
    let global = staged_plan(vec![8, 8], vec![8, 8]);
    let declared = staged_declarations(0, 0);
    let mesh = Mesh::from_config(&ParallelConfig {
        tensor: 2,
        pipeline: 2,
        ..Default::default()
    });

    let results = instantiate_stages(&global, &declared, &mesh, &reference_registry());
    assert_eq!(results.len(), 2);
    assert!(
        results[0]
            .result
            .as_ref()
            .is_ok_and(|p| !p.nodes.is_empty()),
        "stage 0 carries the plan"
    );
    let stage1 = &results[1];
    assert_eq!(stage1.stage, 1);
    match &stage1.result {
        Ok(plan) => assert!(
            plan.nodes.is_empty(),
            "stage 1 owns no nodes; instantiate returns the empty stage for the caller to \
             refuse"
        ),
        Err(e) => panic!("instantiate itself must not guess about the empty stage: {e}"),
    }
}

/// **Reviewer C2, instantiate level — the real Qwen3.6 reshape chain.** The column-parallel
/// QK weight shards `qgw [512, 8192]` on its last axis; rank `k` holds flat columns
/// `[4096k, 4096k+4096)`, which row-major refolding turns into heads `8k..8k+8` of
/// `qgh [512, 16, 2, 256]` — axis 1, kept by index. After the rank-preserving narrow, the
/// shrink reshape back to `q [512, 4096]` keeps axis 1 too. The local shapes must show the
/// heads split (8 of 16), never a renamed axis.
#[test]
fn the_real_qwen36_reshape_chain_instantiates_with_heads_split() {
    let mesh = Mesh::from_config(&ParallelConfig {
        tensor: 2,
        ..Default::default()
    });
    let mut b = PlanBuilder::new("qk", Phase::Forward, mesh.fingerprint());
    let qgw = b.slot("qgw", RsDtype::F32, vec![512, 8192], SlotKind::Activation);
    let qgh = b.slot(
        "qgh",
        RsDtype::F32,
        vec![512, 16, 2, 256],
        SlotKind::Activation,
    );
    let qs = b.slot(
        "qs",
        RsDtype::F32,
        vec![512, 16, 1, 256],
        SlotKind::Activation,
    );
    let q = b.slot("q", RsDtype::F32, vec![512, 4096], SlotKind::Activation);
    b.node(
        OpRef::new("reshape"),
        vec![qgw],
        vec![qgh],
        Attrs::new().set("shape", vec![512i64, 16, 2, 256]),
        "qk.heads",
    );
    b.node(
        OpRef::new("narrow"),
        vec![qgh],
        vec![qs],
        Attrs::new()
            .set("dim", 2i64)
            .set("start", 0i64)
            .set("length", 1i64),
        "qk.narrow",
    );
    b.node(
        OpRef::new("reshape"),
        vec![qs],
        vec![q],
        Attrs::new().set("shape", vec![512i64, 4096]),
        "qk.flat",
    );
    let global = b.build().unwrap();

    let declared = DeclaredAxes {
        slots: BTreeMap::from([(
            "qgw".to_string(),
            BTreeMap::from([("1".to_string(), vec!["tp".to_string()])]),
        )]),
        instances: Vec::new(),
    };
    let tp = GroupMask::single(mesh.index_of("tp").unwrap()).unwrap();
    let instantiated =
        instantiate(&global, &declared, &mesh, 0, &reference_registry()).expect("instantiate");

    let qgh = instantiated.slot_id("qgh").expect("qgh survives");
    assert_eq!(
        instantiated.slot(qgh).layout,
        ParallelLayout::shard(1, tp),
        "the flat half is the heads axis of `qgh` — axis 1, not the head_dim axis"
    );
    assert_eq!(
        instantiated.slot(qgh).shape,
        vec![512, 8, 2, 256],
        "the local heads extent is 8 of 16; the head_dim axis stays complete"
    );
    let q = instantiated.slot_id("q").expect("q survives");
    assert_eq!(
        instantiated.slot(q).layout,
        ParallelLayout::shard(1, tp),
        "the heads shard folds back onto `q`'s flat axis — axis 1 again"
    );
    assert_eq!(
        instantiated.slot(q).shape,
        vec![512, 2048],
        "the local flat extent is 4096 / tp"
    );
}
