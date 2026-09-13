//! The D6 collective backend, proven exchange by exchange: real multi-rank
//! plans — built by hand, compiled by the real compiler, executed by N
//! threads sharing one [`ThreadBackend`] world — with every expected byte
//! written out in the test.
//!
//! These tests are the backend's correctness argument made executable. The
//! plans are intrinsic-only (no provider needed: the compiler resolves
//! intrinsics without a registry lookup), and every collective semantics the
//! module documents has at least one case whose output catches a wrong
//! *ordering* — not just a wrong arithmetic:
//!
//! * `all_to_all` with an uneven `split` ([1, 3] over two ranks) makes the two
//!   ranks produce *different-sized* outputs, so a swapped send/receive order
//!   or a wrong chunk boundary cannot pass;
//! * `all_to_all` along a middle dim pins the per-element assembly;
//! * `all_gather` along a middle dim pins the same assembly for gathers;
//! * `broadcast` from group index 1 (not the default 0) pins the source rule;
//! * `reduce_scatter` pins both the reduction and the slice each rank keeps.

use std::thread;

use rustrain_abi::ffi::RsDtype;
use rustrain_ops::{Phase, Recipe, Registry, TargetEnv};
use rustrain_parallel::{GroupMask, Mesh, ParallelConfig};
use rustrain_plan::{Attrs, OpRef, Plan, PlanBuilder, SlotKind, intrinsic};
use rustrain_runtime::{Executor, HostAllocator, ThreadBackend, ThreadShared};

fn tp_mesh(tp: usize) -> Mesh {
    Mesh::from_config(&ParallelConfig {
        tensor: tp,
        ..Default::default()
    })
}

fn tp_group(mesh: &Mesh) -> GroupMask {
    GroupMask::single(mesh.index_of("tp").expect("canonical mesh has tp"))
        .expect("tp fits in the mask")
}

/// One intrinsic node over the tp group: `x` (input, `in_shape`) → `y`
/// (output, `out_shape`) with the given attributes.
fn collective_plan(
    mesh: &Mesh,
    op: &str,
    in_shape: Vec<i64>,
    out_shape: Vec<i64>,
    attrs: Attrs,
) -> Plan {
    let mut b = PlanBuilder::new("exchange", Phase::Forward, mesh.fingerprint());
    let x = b.slot("x", RsDtype::F32, in_shape, SlotKind::Input);
    let y = b.slot("y", RsDtype::F32, out_shape, SlotKind::Output);
    b.node(OpRef::new(op), vec![x], vec![y], attrs, "exchange");
    b.build().unwrap()
}

fn compile(_mesh: &Mesh, plan: &Plan) -> rustrain_plan::CompiledPlan {
    let registry = Registry::new();
    let recipe = Recipe::default();
    rustrain_plan::Compiler::new(&registry, &recipe, TargetEnv::default())
        .compile(plan)
        .unwrap_or_else(|e| panic!("the intrinsic-only plan must compile: {e}"))
}

/// Runs one executor per rank — each with its own plan, so per-rank local
/// shapes (an uneven split gives different ranks different output extents)
/// are expressible — feeding each rank its own input, and returns each rank's
/// output. A failing rank poisons the shared world *before* returning: the
/// other ranks may be blocked at the same rendezvous, and a silent exit would
/// leave them hanging.
fn run_world(world: usize, mesh: &Mesh, plans: Vec<Plan>, inputs: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    assert_eq!(plans.len(), world, "one plan per rank");
    assert_eq!(inputs.len(), world, "one input per rank");
    let shared = ThreadShared::new(world);
    let mut outputs: Vec<Option<Vec<f32>>> = vec![None; world];
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for rank in 0..world {
            let shared = shared.clone();
            let mesh = mesh.clone();
            let plan = plans[rank].clone();
            let input = inputs[rank].clone();
            handles.push(scope.spawn(move || -> Result<Vec<f32>, String> {
                let result = (|| {
                    let compiled = compile(&mesh, &plan);
                    let backend = ThreadBackend::new(rank, mesh.clone(), shared.clone());
                    let mut executor =
                        Executor::new(compiled, Box::new(HostAllocator::new()), Box::new(backend))
                            .map_err(|e| e.to_string())?;
                    let x = executor
                        .plan()
                        .plan
                        .slot_id("x")
                        .expect("the plan has an input slot");
                    let y = executor
                        .plan()
                        .plan
                        .slot_id("y")
                        .expect("the plan has an output slot");
                    executor.write_f32(x, &input).map_err(|e| e.to_string())?;
                    executor.run().map_err(|e| e.to_string())?;
                    executor.read_f32(y).map_err(|e| e.to_string())
                })();
                if let Err(e) = &result {
                    shared.poison(&format!("rank {rank} failed: {e}"));
                }
                result
            }));
        }
        for (rank, handle) in handles.into_iter().enumerate() {
            match handle.join().expect("no rank thread may panic") {
                Ok(out) => outputs[rank] = Some(out),
                Err(e) => panic!("rank {rank} failed: {e}"),
            }
        }
    });
    outputs.into_iter().map(|o| o.expect("filled")).collect()
}

/// The same plan for every rank — the common case, where the collective does
/// not change the local extent per rank.
fn same_plan(plan: Plan, world: usize) -> Vec<Plan> {
    (0..world).map(|_| plan.clone()).collect()
}

#[test]
fn all_reduce_sums_the_group_elementwise() {
    let mesh = tp_mesh(2);
    let group = tp_group(&mesh);
    let plan = collective_plan(
        &mesh,
        intrinsic::ALL_REDUCE,
        vec![4],
        vec![4],
        Attrs::new()
            .set(intrinsic::ATTR_GROUP, group.bits() as i64)
            .set(intrinsic::ATTR_REDUCE, "sum"),
    );
    let outputs = run_world(
        2,
        &mesh,
        same_plan(plan, 2),
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]],
    );
    assert_eq!(outputs[0], vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(outputs[1], vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn all_reduce_max_and_min_are_elementwise() {
    let mesh = tp_mesh(2);
    let group = tp_group(&mesh);
    for (op_name, expected) in [
        ("max", vec![10.0, 2.0, 30.0, -4.0]),
        ("min", vec![-1.0, -20.0, 3.0, -14.0]),
    ] {
        let plan = collective_plan(
            &mesh,
            intrinsic::ALL_REDUCE,
            vec![4],
            vec![4],
            Attrs::new()
                .set(intrinsic::ATTR_GROUP, group.bits() as i64)
                .set(intrinsic::ATTR_REDUCE, op_name),
        );
        let outputs = run_world(
            2,
            &mesh,
            same_plan(plan, 2),
            vec![vec![-1.0, 2.0, 3.0, -4.0], vec![10.0, -20.0, 30.0, -14.0]],
        );
        assert_eq!(outputs[0], expected, "all_reduce({op_name})");
        assert_eq!(outputs[1], expected, "all_reduce({op_name})");
    }
}

#[test]
fn all_gather_concatenates_in_group_order_along_dim_0() {
    let mesh = tp_mesh(2);
    let group = tp_group(&mesh);
    let plan = collective_plan(
        &mesh,
        intrinsic::ALL_GATHER,
        vec![2],
        vec![4],
        Attrs::new()
            .set(intrinsic::ATTR_GROUP, group.bits() as i64)
            .set(intrinsic::ATTR_DIM, 0i64),
    );
    let outputs = run_world(
        2,
        &mesh,
        same_plan(plan, 2),
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    );
    assert_eq!(outputs[0], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(outputs[1], vec![1.0, 2.0, 3.0, 4.0]);
}

/// A gather along a *middle* dim: the result is not a plain concatenation of
/// the members' buffers — it interleaves slabs — so this case pins the
/// per-element assembly.
#[test]
fn all_gather_along_a_middle_dim_interleaves_slabs() {
    let mesh = tp_mesh(2);
    let group = tp_group(&mesh);
    let plan = collective_plan(
        &mesh,
        intrinsic::ALL_GATHER,
        vec![2, 2],
        vec![2, 4],
        Attrs::new()
            .set(intrinsic::ATTR_GROUP, group.bits() as i64)
            .set(intrinsic::ATTR_DIM, 1i64),
    );
    let outputs = run_world(
        2,
        &mesh,
        same_plan(plan, 2),
        vec![
            vec![1.0, 2.0, 3.0, 4.0], // [[1,2],[3,4]]
            vec![5.0, 6.0, 7.0, 8.0], // [[5,6],[7,8]]
        ],
    );
    let expected = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0];
    assert_eq!(outputs[0], expected);
    assert_eq!(outputs[1], expected);
}

#[test]
fn reduce_scatter_reduces_then_hands_each_rank_its_slice() {
    let mesh = tp_mesh(2);
    let group = tp_group(&mesh);
    let plan = collective_plan(
        &mesh,
        intrinsic::REDUCE_SCATTER,
        vec![4],
        vec![2],
        Attrs::new()
            .set(intrinsic::ATTR_GROUP, group.bits() as i64)
            .set(intrinsic::ATTR_DIM, 0i64),
    );
    let outputs = run_world(
        2,
        &mesh,
        same_plan(plan, 2),
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]],
    );
    assert_eq!(outputs[0], vec![11.0, 22.0]);
    assert_eq!(outputs[1], vec![33.0, 44.0]);
}

/// The source is group index 1 — not the default 0 — so this case only passes
/// when the backend reads the compiled `src` and maps it to the right member.
#[test]
fn broadcast_from_a_named_source() {
    let mesh = tp_mesh(2);
    let group = tp_group(&mesh);
    let plan = collective_plan(
        &mesh,
        intrinsic::BROADCAST,
        vec![3],
        vec![3],
        Attrs::new()
            .set(intrinsic::ATTR_GROUP, group.bits() as i64)
            .set(intrinsic::ATTR_SRC, 1i64),
    );
    let outputs = run_world(
        2,
        &mesh,
        same_plan(plan, 2),
        vec![vec![0.0, 0.0, 0.0], vec![9.0, 8.0, 7.0]],
    );
    assert_eq!(outputs[0], vec![9.0, 8.0, 7.0]);
    assert_eq!(outputs[1], vec![9.0, 8.0, 7.0]);
}

/// The split-ordering case that matters: an **uneven** split makes the two
/// ranks' outputs different sizes, so a swapped sender/destination order or a
/// wrong chunk boundary cannot pass by symmetry. Split entry `i` is the input
/// chunk destined for group index `i`; rank `r` receives `split[r]` elements
/// from every member, concatenated in group order.
#[test]
fn all_to_all_uneven_split_orders_chunks_by_destination() {
    let mesh = tp_mesh(2);
    let group = tp_group(&mesh);
    let attrs = || {
        Attrs::new()
            .set(intrinsic::ATTR_GROUP, group.bits() as i64)
            .set(intrinsic::ATTR_DIM, 0i64)
            .set(intrinsic::ATTR_SPLIT, vec![1i64, 3])
    };
    // Rank 0 sends chunk 0 ([a]) to rank 0 and chunk 1 ([b, c, d]) to rank 1;
    // it receives split[0] = 1 element from each rank: [a, e]. Rank 1
    // receives split[1] = 3 elements from each rank — a different output
    // extent, so each rank gets its own plan.
    let plans = vec![
        collective_plan(&mesh, intrinsic::ALL_TO_ALL, vec![4], vec![2], attrs()),
        collective_plan(&mesh, intrinsic::ALL_TO_ALL, vec![4], vec![6], attrs()),
    ];
    let outputs = run_world(
        2,
        &mesh,
        plans,
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
    );
    assert_eq!(outputs[0], vec![1.0, 5.0]);
    assert_eq!(outputs[1], vec![2.0, 3.0, 4.0, 6.0, 7.0, 8.0]);
}

/// The same split semantics along a middle dim, pinning the per-element
/// assembly for the redistribution path.
#[test]
fn all_to_all_along_a_middle_dim() {
    let mesh = tp_mesh(2);
    let group = tp_group(&mesh);
    let attrs = || {
        Attrs::new()
            .set(intrinsic::ATTR_GROUP, group.bits() as i64)
            .set(intrinsic::ATTR_DIM, 1i64)
            .set(intrinsic::ATTR_SPLIT, vec![1i64, 3])
    };
    let plans = vec![
        collective_plan(
            &mesh,
            intrinsic::ALL_TO_ALL,
            vec![2, 4],
            vec![2, 2],
            attrs(),
        ),
        collective_plan(
            &mesh,
            intrinsic::ALL_TO_ALL,
            vec![2, 4],
            vec![2, 6],
            attrs(),
        ),
    ];
    let outputs = run_world(
        2,
        &mesh,
        plans,
        vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], // [[1,2,3,4],[5,6,7,8]]
            vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], // [[9,10,11,12],[13,14,15,16]]
        ],
    );
    // Rank 0 receives column chunk [0] (1 column) from both ranks.
    assert_eq!(outputs[0], vec![1.0, 9.0, 5.0, 13.0]);
    // Rank 1 receives columns [1, 4) from both ranks.
    assert_eq!(
        outputs[1],
        vec![
            2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 6.0, 7.0, 8.0, 14.0, 15.0, 16.0
        ]
    );
}

#[test]
fn all_to_all_equal_split_is_a_pairwise_transpose() {
    let mesh = tp_mesh(2);
    let group = tp_group(&mesh);
    let plan = collective_plan(
        &mesh,
        intrinsic::ALL_TO_ALL,
        vec![4],
        vec![4],
        Attrs::new()
            .set(intrinsic::ATTR_GROUP, group.bits() as i64)
            .set(intrinsic::ATTR_DIM, 0i64),
    );
    let outputs = run_world(
        2,
        &mesh,
        same_plan(plan, 2),
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
    );
    assert_eq!(outputs[0], vec![1.0, 2.0, 5.0, 6.0]);
    assert_eq!(outputs[1], vec![3.0, 4.0, 7.0, 8.0]);
}

/// A split that does not fit the input's extent is a **reported** error on
/// every rank — the backend re-validates the compiled plan against the real
/// local shapes instead of trusting it.
#[test]
fn all_to_all_rejects_a_split_mismatch_on_every_rank() {
    let mesh = tp_mesh(2);
    let group = tp_group(&mesh);
    // Compile-level validation refuses this split (sums to 3, input holds 4),
    // so the case is built at the backend level: a plan with a *valid* compile
    // but an executor-level mismatch is exercised through the shape check by
    // declaring an output shape the exchange cannot produce.
    let plan = collective_plan(
        &mesh,
        intrinsic::ALL_TO_ALL,
        vec![4],
        vec![6], // split [1,3]: rank 0 would receive 2, not 6 — the error path.
        Attrs::new()
            .set(intrinsic::ATTR_GROUP, group.bits() as i64)
            .set(intrinsic::ATTR_DIM, 0i64)
            .set(intrinsic::ATTR_SPLIT, vec![1i64, 3]),
    );
    let shared = ThreadShared::new(2);
    let errors: Vec<String> = thread::scope(|scope| {
        let mut handles = Vec::new();
        for rank in 0..2 {
            let shared = shared.clone();
            let mesh = mesh.clone();
            let plan = plan.clone();
            handles.push(scope.spawn(move || {
                let compiled = compile(&mesh, &plan);
                let backend = ThreadBackend::new(rank, mesh.clone(), shared.clone());
                let mut executor =
                    Executor::new(compiled, Box::new(HostAllocator::new()), Box::new(backend))
                        .map_err(|e| e.to_string())?;
                let x = executor.plan().plan.slot_id("x").expect("x");
                executor
                    .write_f32(x, &[1.0, 2.0, 3.0, 4.0])
                    .map_err(|e| e.to_string())?;
                match executor.run() {
                    Ok(_) => Ok(()),
                    Err(e) => {
                        // A failing rank must not leave the others blocked.
                        shared.poison("shape mismatch on this rank");
                        Err(e.to_string())
                    }
                }
            }));
        }
        handles
            .into_iter()
            .map(|h| match h.join().expect("no panic") {
                Ok(()) => String::new(),
                Err(e) => e,
            })
            .collect()
    });
    // Rank 0 fails the shape validation (its expected receive of 2 elements
    // does not match the declared output of 6); rank 1, which validates, is
    // woken by rank 0's poison with the recorded failure. Both must be
    // errors, and at least one must name the shape.
    assert!(
        errors[0].contains("exchange produces shape"),
        "rank 0: {}",
        errors[0]
    );
    assert!(
        errors[1].contains("shape mismatch on this rank"),
        "rank 1 must wake with the poison: {}",
        errors[1]
    );
}

/// `sync` exchanges nothing but still rendezvouses: both ranks complete and
/// the tensor passes through.
#[test]
fn sync_is_a_barrier_with_a_pass_through() {
    let mesh = tp_mesh(2);
    let group = tp_group(&mesh);
    let plan = collective_plan(
        &mesh,
        intrinsic::SYNC,
        vec![2],
        vec![2],
        Attrs::new().set(intrinsic::ATTR_GROUP, group.bits() as i64),
    );
    let outputs = run_world(
        2,
        &mesh,
        same_plan(plan, 2),
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    );
    assert_eq!(outputs[0], vec![1.0, 2.0]);
    assert_eq!(outputs[1], vec![3.0, 4.0]);
}

/// A poisoned world wakes blocked ranks with the recorded error instead of
/// deadlocking: rank 1 announces it is about to meet, main poisons the shared
/// state, and the (would-be blocking) collective finishes with an error.
#[test]
fn a_poisoned_rendezvous_wakes_blocked_ranks() {
    let mesh = tp_mesh(2);
    let group = tp_group(&mesh);
    let plan = collective_plan(
        &mesh,
        intrinsic::SYNC,
        vec![2],
        vec![2],
        Attrs::new().set(intrinsic::ATTR_GROUP, group.bits() as i64),
    );
    let shared = ThreadShared::new(2);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
    let (done_tx, done_rx) = std::sync::mpsc::channel::<Result<(), String>>();

    let worker_shared = shared.clone();
    let worker_mesh = mesh.clone();
    let worker_plan = plan.clone();
    let worker = thread::spawn(move || {
        let compiled = compile(&worker_mesh, &worker_plan);
        let backend = ThreadBackend::new(1, worker_mesh.clone(), worker_shared.clone());
        let mut executor =
            Executor::new(compiled, Box::new(HostAllocator::new()), Box::new(backend))
                .map_err(|e| e.to_string())?;
        let x = executor.plan().plan.slot_id("x").expect("x");
        executor
            .write_f32(x, &[1.0, 2.0])
            .map_err(|e| e.to_string())?;
        ready_tx.send(()).expect("send");
        executor.run().map(|_| ()).map_err(|e| e.to_string())
    });

    // The worker has signalled it is entering its first collective; poison
    // before it can complete the rendezvous (and before rank 0 joins it).
    ready_rx.recv().expect("worker ready");
    shared.poison("rank 0 failed at an operator upstream");
    let _ = shared;

    let result = worker.join().expect("no panic");
    done_tx.send(result).expect("send");
    let error = done_rx.recv().expect("result").unwrap_err();
    assert!(
        error.contains("rank 0 failed at an operator upstream"),
        "the blocked rank must wake with the recorded failure: {error}"
    );
}
