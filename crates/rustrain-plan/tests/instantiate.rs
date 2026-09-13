//! D4 acceptance on the plan side: `instantiate` on a hand-built global plan, then the
//! existing compile path — a row-parallel weight's declared shard makes its output a
//! `partial(sum, tp)`, and `Compiler::compile` on the instantiated plan inserts an
//! `all_reduce` whose mask is `tp` (the existing behaviour, re-proved on an instantiated plan).

use std::collections::BTreeMap;

use rustrain_abi::Plugin;
use rustrain_abi::ffi::RsDtype;
use rustrain_ops::{Phase, Recipe, Registry, TargetEnv};
use rustrain_parallel::{GroupMask, Mesh, ParallelConfig, ParallelLayout, ReduceOp};
use rustrain_plan::ir::intrinsic;
use rustrain_plan::{Attrs, DeclaredAxes, OpRef, PlanBuilder, PlanError, SlotKind, instantiate};

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

    let instantiated = instantiate(&global, &declared, &mesh, 0).expect("instantiate");

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
    match instantiate(&global, &declared, &mesh, 2).unwrap_err() {
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
    match instantiate(&global, &declared, &tp_mesh(), 0).unwrap_err() {
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
    match instantiate(&global, &declared, &mesh, 0).unwrap_err() {
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
    match instantiate(&global, &declared, &mesh, 0).unwrap_err() {
        PlanError::MissingStage { pp, .. } => assert_eq!(pp, 2),
        other => panic!("expected MissingStage, got {other:?}"),
    }
}
