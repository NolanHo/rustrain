//! Does the conformance gate actually catch anything?
//!
//! A gate nobody has seen fail is decoration. These tests register deliberately
//! broken implementations next to the reference one and assert that the harness
//! reports them — and that it reports the *right* thing, since "the reference is
//! the reference" is a skip and not a pass.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use rustrain_abi::Plugin;
use rustrain_abi::author::{OpSpec, PluginBuilder};
use rustrain_abi::ffi::{
    RsAttrs, RsCtx, RsDtype, RsMemReq, RsNumerics, RsPlugin, RsShardRule, RsTensor,
};
use rustrain_ops::{Phase, Recipe, Registry, TargetEnv};
use rustrain_runtime::conformance::{Check, Harness, REFERENCE_VARIANT, default_cases};

// ── a deliberately wrong provider ───────────────────────────────────────────

const WRONG_VARIANT: &str = "synthetic.wrong";
const NONDETERMINISTIC_VARIANT: &str = "synthetic.drift";

static DRIFT: AtomicU64 = AtomicU64::new(0);

/// Computes `a - b` where the case asked for `add`. Wrong, and wrong in a way
/// that is invisible without a second implementation to compare against.
unsafe extern "C" fn subtract_execute(
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
    // SAFETY: the executor passes descriptors sized by the same plan `infer` ran on.
    unsafe {
        let a = &**inputs;
        let b = &*(*inputs.add(1));
        let out = &mut **outputs;
        let n = a.dims().iter().product::<i64>().max(0) as usize;
        let (pa, pb, po) = (
            a.data as *const f32,
            b.data as *const f32,
            out.data as *mut f32,
        );
        for i in 0..n {
            *po.add(i) = *pa.add(i) - *pb.add(i);
        }
    }
    0
}

/// Adds a different constant on every call, so two runs cannot agree.
unsafe extern "C" fn drifting_execute(
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
    let drift = DRIFT.fetch_add(1, Ordering::SeqCst) as f32;
    // SAFETY: as above.
    unsafe {
        let a = &**inputs;
        let b = &*(*inputs.add(1));
        let out = &mut **outputs;
        let n = a.dims().iter().product::<i64>().max(0) as usize;
        let (pa, pb, po) = (
            a.data as *const f32,
            b.data as *const f32,
            out.data as *mut f32,
        );
        for i in 0..n {
            *po.add(i) = *pa.add(i) + *pb.add(i) + drift;
        }
    }
    0
}

unsafe extern "C" fn binary_infer(
    inputs: *const *const RsTensor,
    n_in: u32,
    outputs: *const *mut RsTensor,
    n_out: u32,
    _attrs: *const RsAttrs,
) -> i32 {
    if n_in != 2 || n_out != 1 {
        return 1;
    }
    // SAFETY: descriptors come from the compiler's shape pass.
    unsafe {
        let a = &**inputs;
        let out = &mut **outputs;
        out.dtype = RsDtype::F32;
        out.rank = a.rank;
        out.shape = a.shape;
        out.set_contiguous_strides();
    }
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
    unsafe { *out = RsMemReq::default() };
    0
}

fn numerics() -> RsNumerics {
    RsNumerics {
        in_dtype: RsDtype::F32,
        out_dtype: RsDtype::F32,
        accum_dtype: RsDtype::F32,
        grad_dtype: RsDtype::F32,
        ..Default::default()
    }
}

fn broken_plugin() -> &'static RsPlugin {
    static PLUGIN: OnceLock<&'static RsPlugin> = OnceLock::new();
    PLUGIN.get_or_init(|| {
        PluginBuilder::new("synthetic", "0.1.0")
            .op(OpSpec::new("elementwise_binary", WRONG_VARIANT)
                .shard(RsShardRule::ELEMENTWISE)
                .doc("computes a - b regardless of the requested kind: deliberately wrong")
                .dtypes(&[RsDtype::F32])
                .numerics(numerics())
                .execute(subtract_execute)
                .infer(binary_infer)
                .memory(zero_memory))
            .op(OpSpec::new("elementwise_binary", NONDETERMINISTIC_VARIANT)
                .shard(RsShardRule::ELEMENTWISE)
                .doc("correct, but adds a different constant on every call")
                .dtypes(&[RsDtype::F32])
                .numerics(numerics())
                .execute(drifting_execute)
                .infer(binary_infer)
                .memory(zero_memory))
            .build()
    })
}

fn harness_registry() -> Registry {
    let mut registry = Registry::new();
    // SAFETY: both descriptors are leaked by `PluginBuilder` and live for the process.
    let reference = unsafe { Plugin::from_static(rustrain_kernels::plugin(), "<built-in>") }
        .expect("the built-in provider passes ABI validation");
    registry.add_plugin(reference).expect("registering it");
    let broken = unsafe { Plugin::from_static(broken_plugin(), "<synthetic>") }
        .expect("the synthetic provider passes ABI validation");
    registry.add_plugin(broken).expect("registering it");
    registry
}

fn binary_case() -> rustrain_runtime::conformance::Case {
    default_cases()
        .into_iter()
        .find(|c| c.op == "elementwise_binary")
        .expect("the default table has a case for elementwise_binary")
}

// ── the tests ───────────────────────────────────────────────────────────────

/// The acceptance criterion from the spec: injecting a wrong kernel must be
/// caught, and caught by the numeric check rather than by luck.
#[test]
fn a_wrong_implementation_is_caught_by_the_numeric_check() {
    let registry = harness_registry();
    let recipe = Recipe::from_toml("[kernel]\ndefault = \"reference\"\n").unwrap();
    let harness = Harness::new(&registry, &recipe);

    let results = harness.run(&binary_case());
    let wrong = results
        .iter()
        .find(|r| r.variant == WRONG_VARIANT)
        .expect("the synthetic variant must be checked");

    match &wrong.numeric {
        Check::Fail { detail } => {
            assert!(
                detail.contains("exceeds the budget"),
                "the failure must be a measured numeric difference, got: {detail}"
            );
        }
        other => panic!("a kernel that computes a - b must fail the numeric check, got {other:?}"),
    }
    assert!(!wrong.ok());
}

/// Determinism is a separate axis: an implementation can be numerically close on
/// one call and still be unusable.
#[test]
fn a_drifting_implementation_is_caught_by_the_determinism_check() {
    let registry = harness_registry();
    let recipe = Recipe::from_toml("[kernel]\ndefault = \"reference\"\n").unwrap();
    let harness = Harness::new(&registry, &recipe);

    let results = harness.run(&binary_case());
    let drifting = results
        .iter()
        .find(|r| r.variant == NONDETERMINISTIC_VARIANT)
        .expect("the drifting variant must be checked");

    match &drifting.determinism {
        Check::Fail { detail } => {
            assert!(
                detail.contains("different bytes"),
                "the failure must name what differed, got: {detail}"
            );
        }
        other => panic!("a drifting kernel must fail the determinism check, got {other:?}"),
    }
}

/// The reference cannot be compared against itself, and the harness says so
/// rather than reporting a pass it did not earn.
#[test]
fn the_reference_is_skipped_not_passed() {
    let registry = harness_registry();
    let recipe = Recipe::from_toml("[kernel]\ndefault = \"reference\"\n").unwrap();
    let harness = Harness::new(&registry, &recipe);

    let results = harness.run(&binary_case());
    let reference = results
        .iter()
        .find(|r| r.variant == REFERENCE_VARIANT)
        .expect("the reference variant must be checked");

    match &reference.numeric {
        Check::Skipped { reason } => {
            assert!(
                reason.contains("reference implementation"),
                "the skip must explain itself, got: {reason}"
            );
        }
        other => panic!("comparing the reference to itself is not a check; got {other:?}"),
    }
    // Determinism, on the other hand, *is* checkable, and must still run.
    assert!(
        matches!(reference.determinism, Check::Pass { .. }),
        "determinism is checkable for the reference too, got {:?}",
        reference.determinism
    );
}

/// The gate is only meaningful if it is green when nothing is broken.
#[test]
fn the_reference_alone_passes_every_implemented_check() {
    let mut registry = Registry::new();
    let reference = unsafe { Plugin::from_static(rustrain_kernels::plugin(), "<built-in>") }
        .expect("ABI validation");
    registry.add_plugin(reference).expect("registering it");

    let recipe = Recipe::from_toml("[kernel]\ndefault = \"reference\"\n").unwrap();
    let harness = Harness::new(&registry, &recipe);

    let mut report = rustrain_runtime::conformance::Report::default();
    for case in default_cases() {
        report.results.extend(harness.run(&case));
    }

    assert!(
        report.passed(),
        "the reference implementation must satisfy every implemented check:\n{}",
        report.explain()
    );
    assert!(
        !report.results.is_empty(),
        "the default table must cover something"
    );
    // Gradient is not implemented yet, so it must be *skipped*, and the skip has
    // to say why rather than looking like a pass.
    for r in &report.results {
        match &r.gradient {
            Check::Skipped { reason } => assert!(
                reason.contains("backward derivation"),
                "the gradient skip must point at what is missing: {reason}"
            ),
            other => panic!("gradient checking is not implemented; got {other:?}"),
        }
    }
}

/// A plan pinned to a variant that does not exist is an error, not a fallback.
#[test]
fn an_unregistered_variant_is_reported_rather_than_substituted() {
    let registry = harness_registry();
    let recipe = Recipe::from_toml("[kernel]\ndefault = \"reference\"\n").unwrap();
    let harness = Harness::new(&registry, &recipe);

    let result = harness.check_variant(&binary_case(), "no.such.variant");
    match &result.numeric {
        Check::Fail { detail } => assert!(
            detail.contains("not registered"),
            "the error must say the variant is missing, got: {detail}"
        ),
        other => panic!("expected a failure, got {other:?}"),
    }
}

#[test]
fn the_registry_used_by_these_tests_is_the_real_one() {
    let registry = harness_registry();
    let names = registry.op_names();
    assert!(names.contains(&"elementwise_binary"));
    let variants: Vec<&str> = registry
        .candidates("elementwise_binary")
        .into_iter()
        .map(|c| c.variant())
        .collect();
    assert!(variants.contains(&REFERENCE_VARIANT));
    assert!(variants.contains(&WRONG_VARIANT));
    let _ = Phase::Forward;
    let _ = TargetEnv::default();
}
