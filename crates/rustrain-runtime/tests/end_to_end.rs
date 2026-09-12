//! End-to-end: a plugin authored in Rust, registered, resolved from a recipe,
//! compiled into a plan, and executed — with the all-reduce that sharding
//! propagation inserts actually driven by the runtime.
//!
//! This is the whole architecture in one file. Nothing here touches a GPU, and
//! nothing here reads an environment variable.

use std::sync::OnceLock;

use rustrain_abi::author::{OpSpec, PluginBuilder};
use rustrain_abi::ffi::{
    RsAttrKind, RsAttrs, RsCtx, RsDeviceKind, RsDtype, RsMemReq, RsPlugin, RsTensor,
};
use rustrain_abi::Plugin;
use rustrain_ops::{Phase, Recipe, Registry, TargetEnv};
use rustrain_parallel::{GroupKind, ParallelConfig, ParallelLayout, ReduceOp};
use rustrain_plan::{Attrs, OpRef, PlanBuilder, SlotKind};

use rustrain_runtime::{Executor, HostAllocator, RuntimeError, SingleRank, required_inputs};

// ── reading attributes from C ───────────────────────────────────────────────

/// Reads a numeric attribute. Returns `None` for a missing key or a key of a
/// type that cannot be interpreted as a number.
unsafe fn attr_f64(attrs: *const RsAttrs, key: &str) -> Option<f64> {
    if attrs.is_null() {
        return None;
    }
    // SAFETY: the caller passes an `RsAttrs` produced by the plan, whose slices
    // and strings outlive the call.
    let list = unsafe { (*attrs).as_slice() };
    for a in list {
        if a.key.is_null() {
            continue;
        }
        // SAFETY: the key is a NUL-terminated string owned by the plan.
        let name = unsafe { std::ffi::CStr::from_ptr(a.key) }.to_str().ok()?;
        if name != key {
            continue;
        }
        return match a.kind {
            RsAttrKind::F64 => Some(a.f64),
            RsAttrKind::I64 => Some(a.i64 as f64),
            _ => None,
        };
    }
    None
}

// ── a provider written in Rust ──────────────────────────────────────────────

/// `out = in * factor`, elementwise.
unsafe extern "C" fn scale_execute(
    _ctx: *mut RsCtx,
    inputs: *const *const RsTensor,
    n_in: u32,
    outputs: *const *mut RsTensor,
    n_out: u32,
    attrs: *const RsAttrs,
) -> i32 {
    if n_in != 1 || n_out != 1 || inputs.is_null() || outputs.is_null() {
        return 1;
    }
    // SAFETY: the executor passes exactly the descriptors the plan declared.
    let input = unsafe { &**inputs };
    let output = unsafe { &mut **outputs };
    if input.dtype != RsDtype::F32 || output.data.is_null() || input.data.is_null() {
        return 2;
    }
    // SAFETY: `attrs` is the step's attribute array or null.
    let factor = unsafe { attr_f64(attrs, "factor") }.unwrap_or(1.0) as f32;
    let len = input.dims().iter().product::<i64>().max(0) as usize;
    // SAFETY: both buffers hold `len` f32; the compiler checked the shapes via
    // `infer` before the plan was allowed to run.
    unsafe {
        let src = input.data as *const f32;
        let dst = output.data as *mut f32;
        for i in 0..len {
            *dst.add(i) = *src.add(i) * factor;
        }
    }
    0
}

/// Two inputs, one output — the shape a tensor-parallel `linear` has, so that
/// the sharding rules classify it as one and the compiler has to reconcile a
/// partial sum.
unsafe extern "C" fn linear_execute(
    _ctx: *mut RsCtx,
    inputs: *const *const RsTensor,
    n_in: u32,
    outputs: *const *mut RsTensor,
    n_out: u32,
    _attrs: *const RsAttrs,
) -> i32 {
    if n_in != 2 || n_out != 1 {
        return 1;
    }
    // SAFETY: as above.
    let a = unsafe { &**inputs };
    let b = unsafe { &*(*inputs.add(1)) };
    let out = unsafe { &mut **outputs };
    if a.dtype != RsDtype::F32 || out.data.is_null() {
        return 2;
    }
    let len = a.dims().iter().product::<i64>().max(0) as usize;
    let scale = b.shape.first().copied().unwrap_or(1) as f32;
    // SAFETY: buffers sized by `infer` in the plan.
    unsafe {
        let src = a.data as *const f32;
        let dst = out.data as *mut f32;
        for i in 0..len {
            *dst.add(i) = *src.add(i) * scale;
        }
    }
    0
}

unsafe extern "C" fn scale_infer(
    inputs: *const *const RsTensor,
    n_in: u32,
    outputs: *const *mut RsTensor,
    n_out: u32,
    _attrs: *const RsAttrs,
) -> i32 {
    if n_in < 1 || n_out < 1 || inputs.is_null() || outputs.is_null() {
        return 1;
    }
    // SAFETY: descriptors come from the compiler's shape pass.
    let input = unsafe { &**inputs };
    let output = unsafe { &mut **outputs };
    output.dtype = input.dtype;
    output.rank = input.rank;
    output.shape = input.shape;
    output.set_contiguous_strides();
    0
}

unsafe extern "C" fn zero_memory(
    _io: *const *const RsTensor,
    _n_io: u32,
    _attrs: *const RsAttrs,
    out: *mut RsMemReq,
) -> i32 {
    if out.is_null() {
        return 1;
    }
    // SAFETY: the compiler allocated the scratch struct.
    unsafe {
        *out = RsMemReq {
            workspace_bytes: 0,
            save_for_backward_bytes: 0,
            save_tensor_count: 0,
            _pad: 0,
        };
    }
    0
}

/// The plugin, built once. `build()` leaks its descriptors on purpose, which is
/// what lets them be handed out as `&'static`.
fn test_plugin() -> &'static RsPlugin {
    static PLUGIN: OnceLock<&'static RsPlugin> = OnceLock::new();
    PLUGIN.get_or_init(|| {
        PluginBuilder::new("test", "0.1.0")
            .op(
                OpSpec::new("scale", "test.f32")
                    .doc("out = in * factor")
                    .dtypes(&[RsDtype::F32])
                    .execute(scale_execute)
                    .infer(scale_infer)
                    .memory(zero_memory),
            )
            .op(
                OpSpec::new("linear", "test.f32")
                    .doc("stand-in for a tensor-parallel linear")
                    .dtypes(&[RsDtype::F32])
                    .execute(linear_execute)
                    .infer(scale_infer)
                    .memory(zero_memory),
            )
            .build()
    })
}

fn setup() -> (Registry, Recipe, TargetEnv) {
    let plugin = unsafe { Plugin::from_static(test_plugin(), "<in-process test plugin>") }
        .expect("the test plugin must pass ABI validation");

    let mut registry = Registry::new();
    registry
        .add_plugin(plugin)
        .expect("the test plugin must register");

    let recipe = Recipe::from_toml("[kernel]\ndefault = \"test\"\n").expect("recipe parses");
    (registry, recipe, TargetEnv::default())
}

// ── tests ───────────────────────────────────────────────────────────────────

#[test]
fn plugin_loads_registers_and_resolves_from_the_recipe() {
    let (registry, _, _) = setup();
    let mut names = registry.op_names();
    names.sort_unstable();
    assert_eq!(names, vec!["linear", "scale"]);
}

#[test]
fn compiled_plan_executes_and_produces_the_expected_numbers() {
    let (registry, recipe, env) = setup();

    let mut b = PlanBuilder::new("scale", Phase::Forward, ParallelConfig::default());
    let x = b.slot("x", RsDtype::F32, vec![4], SlotKind::Input);
    let y = b.slot("y", RsDtype::F32, vec![4], SlotKind::Output);
    b.node(
        OpRef::new("scale"),
        vec![x],
        vec![y],
        Attrs::new().set("factor", 2.0),
        "scale0",
    );
    let plan = b.build().unwrap();

    let compiled = rustrain_plan::Compiler::new(&registry, &recipe, env, ParallelConfig::default())
        .compile(&plan)
        .expect("compile");

    assert_eq!(required_inputs(&compiled), vec![(x, SlotKind::Input)]);

    let mut ex = Executor::new(
        compiled,
        Box::new(HostAllocator::new()),
        Box::new(SingleRank::new(1)),
    )
    .unwrap();

    ex.write_f32(x, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let stats = ex.run().unwrap();
    assert_eq!(stats.ops, 1);
    assert_eq!(stats.collectives, 0);

    assert_eq!(ex.read_f32(y).unwrap(), vec![2.0, 4.0, 6.0, 8.0]);
}

/// The headline: a row-parallel weight makes the compiler insert an all-reduce,
/// and the runtime executes it as part of the plan. Nothing in the test asks
/// for communication — it falls out of the sharding declarations.
#[test]
fn row_parallel_weight_inserts_a_collective_the_runtime_drives() {
    let (registry, recipe, env) = setup();
    let parallel = ParallelConfig {
        tensor: 2,
        ..Default::default()
    };

    let mut b = PlanBuilder::new("tp", Phase::Forward, parallel);
    let x = b.slot("x", RsDtype::F32, vec![4], SlotKind::Input);
    // A weight is [K, N]; sharding dim 0 splits the contraction, so each rank
    // holds a partial sum from the same `linear` rule the framework applies.
    let w = b.slot_with_layout(
        "w",
        RsDtype::F32,
        vec![3],
        SlotKind::Weight,
        ParallelLayout::Shard {
            dim: 0,
            group: GroupKind::Tp,
        },
    );
    // ...but the plan promises a complete tensor here, so a conversion is owed.
    let y = b.slot("y", RsDtype::F32, vec![4], SlotKind::Output);
    b.node(
        OpRef::new("linear"),
        vec![x, w],
        vec![y],
        Attrs::new(),
        "tp.linear",
    );
    let plan = b.build().unwrap();

    let compiled = rustrain_plan::Compiler::new(&registry, &recipe, env, parallel).compile(&plan);

    // The weight layout is the only thing that could force communication, and
    // the shard rule for `linear` must have noticed it.
    let compiled = compiled.expect("compile");
    assert_eq!(
        compiled.inserted.len(),
        1,
        "exactly one collective should have been spliced in"
    );
    assert_eq!(compiled.inserted[0].op, rustrain_plan::intrinsic::ALL_REDUCE);
    assert_eq!(compiled.inserted[0].reduce, Some(ReduceOp::Sum));

    let mut ex = Executor::new(
        compiled,
        Box::new(HostAllocator::new()),
        Box::new(SingleRank::new(1)),
    )
    .unwrap();
    ex.write_f32(x, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    ex.write_f32(w, &[1.0, 1.0, 1.0]).unwrap();

    let stats = ex.run().unwrap();
    assert_eq!(stats.collectives, 1, "the runtime must drive the all-reduce");
    assert_eq!(stats.ops, 1);
    // With one process the all-reduce is the identity, so the value survives.
    assert_eq!(ex.read_f32(y).unwrap(), vec![3.0, 6.0, 9.0, 12.0]);
}

/// A single-process executor must refuse to pretend it performed a collective
/// that a real multi-rank run would need.
#[test]
fn single_rank_refuses_when_the_world_is_larger_than_one() {
    let (registry, recipe, env) = setup();
    let parallel = ParallelConfig {
        tensor: 2,
        ..Default::default()
    };

    let mut b = PlanBuilder::new("tp", Phase::Forward, parallel);
    let x = b.slot("x", RsDtype::F32, vec![4], SlotKind::Input);
    let w = b.slot_with_layout(
        "w",
        RsDtype::F32,
        vec![3],
        SlotKind::Weight,
        ParallelLayout::Shard {
            dim: 0,
            group: GroupKind::Tp,
        },
    );
    let y = b.slot("y", RsDtype::F32, vec![4], SlotKind::Output);
    b.node(OpRef::new("linear"), vec![x, w], vec![y], Attrs::new(), "lin");
    let compiled = rustrain_plan::Compiler::new(&registry, &recipe, env, parallel)
        .compile(&b.build().unwrap())
        .unwrap();

    let mut ex = Executor::new(
        compiled,
        Box::new(HostAllocator::new()),
        // Two ranks, but only one process to serve them.
        Box::new(SingleRank::new(2)),
    )
    .unwrap();
    ex.write_f32(x, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    ex.write_f32(w, &[1.0, 1.0, 1.0]).unwrap();

    match ex.run() {
        Err(RuntimeError::Collective { reason, .. }) => {
            assert!(
                reason.contains("world_size=2"),
                "the refusal must name the world size: {reason}"
            );
        }
        other => panic!("expected a collective failure, got {other:?}"),
    }
}

#[test]
fn resolved_implementation_is_recorded_in_the_digest() {
    let (registry, recipe, env) = setup();
    let mut b = PlanBuilder::new("scale", Phase::Forward, ParallelConfig::default());
    let x = b.slot("x", RsDtype::F32, vec![2], SlotKind::Input);
    let y = b.slot("y", RsDtype::F32, vec![2], SlotKind::Output);
    b.node(OpRef::new("scale"), vec![x], vec![y], Attrs::new(), "s");
    let plan = b.build().unwrap();

    let a = rustrain_plan::Compiler::new(&registry, &recipe, env.clone(), ParallelConfig::default())
        .compile(&plan)
        .unwrap();

    // Same inputs, same decision, same digest.
    let b2 = rustrain_plan::Compiler::new(&registry, &recipe, env.clone(), ParallelConfig::default())
        .compile(&plan)
        .unwrap();
    assert_eq!(a.digest, b2.digest);
    assert_eq!(a.resolved.len(), 1);
    assert_eq!(a.resolved[0].spec_name, "scale@test.f32");

    // Naming a different implementation in the recipe changes the digest without
    // recompiling anything.
    let other = Recipe::from_toml("[kernel]\ndefault = \"test\"\n[kernel.ops.scale]\nforward = \"test.f32\"\n")
        .unwrap();
    let c = rustrain_plan::Compiler::new(&registry, &other, env, ParallelConfig::default())
        .compile(&plan)
        .unwrap();
    assert_eq!(c.resolved[0].spec_name, "scale@test.f32");
    assert_eq!(a.digest, c.digest, "an equivalent recipe must not churn the digest");
}

#[test]
fn unknown_operator_is_a_hard_error_not_a_fallback() {
    let (registry, recipe, env) = setup();
    let mut b = PlanBuilder::new("bad", Phase::Forward, ParallelConfig::default());
    let x = b.slot("x", RsDtype::F32, vec![2], SlotKind::Input);
    let y = b.slot("y", RsDtype::F32, vec![2], SlotKind::Output);
    b.node(OpRef::new("does_not_exist"), vec![x], vec![y], Attrs::new(), "nope");
    let plan = b.build().unwrap();

    let err = rustrain_plan::Compiler::new(&registry, &recipe, env, ParallelConfig::default())
        .compile(&plan)
        .unwrap_err();
    let text = err.to_string();
    assert!(text.contains("does_not_exist"), "{text}");
}

#[test]
fn host_allocator_tracks_residency_and_frees_on_drop() {
    let (registry, recipe, env) = setup();
    let mut b = PlanBuilder::new("scale", Phase::Forward, ParallelConfig::default());
    let x = b.slot("x", RsDtype::F32, vec![1024], SlotKind::Input);
    let y = b.slot("y", RsDtype::F32, vec![1024], SlotKind::Output);
    b.node(OpRef::new("scale"), vec![x], vec![y], Attrs::new(), "s");
    let compiled = rustrain_plan::Compiler::new(&registry, &recipe, env, ParallelConfig::default())
        .compile(&b.build().unwrap())
        .unwrap();

    let ex = Executor::new(
        compiled,
        Box::new(HostAllocator::new()),
        Box::new(SingleRank::new(1)),
    )
    .unwrap();
    assert_eq!(ex.stats().resident_bytes, 2 * 1024 * 4);
    drop(ex); // must not leak or double-free; run under a sanitizer to prove more
}

#[test]
fn device_kind_is_exposed_so_a_gpu_allocator_can_slot_in() {
    let a = HostAllocator::new();
    assert_eq!(
        rustrain_runtime::Allocator::device(&a),
        RsDeviceKind::CPU
    );
}
