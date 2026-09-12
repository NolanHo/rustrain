//! Integration tests: only the public API, exercised the way another crate
//! (the plan compiler, the CLI) would use it.
//!
//! Descriptor registration is deliberately absent here — a `RegisteredOp` can
//! only come from a loaded plugin through `Registry::add_plugin`, which needs a
//! compiled `.so`. The in-crate unit tests cover registration and resolution
//! through the crate-internal test path; this file pins the surface that is
//! visible from outside, so a signature change that breaks downstream crates
//! fails here first.

use rustrain_ops::{
    BackwardPlan, COMPOSITE_OPS, DtypeName, OpSummary, Phase, QuantName, Recipe, Registry,
    ResolveError, ScaleName, TargetEnv, is_composite_op,
};

/// The example from spec §2.7, verbatim.
const SPEC_RECIPE: &str = r#"
[kernel]
default = "aten"
strict  = true

[kernel.ops.rmsnorm]
forward  = "cuda.fused_bf16"
backward = "cuda.fused_bf16"

[kernel.ops.mlp_swiglu]
forward         = "cuda.fp8_block128"
backward        = "autodiff"
check_expansion = true

[kernel.precision]
compute        = "bf16"
accumulate     = "fp32"
master_weights = "fp32"
grad           = "bf16"
weights        = "fp8_e4m3"
quant_scheme   = "per_block"
block          = [128, 128]
scale_mode     = "delayed"
amax_history   = 8

[kernel.parallel]
tensor   = 8
expert   = 1
context  = 1
data     = 1
overlap_collectives = true
"#;

#[test]
fn spec_example_recipe_parses_as_written() {
    // The spelling in the spec (`fp32`, `fp8_e4m3`) must be accepted, or the
    // documented example would not run.
    let recipe = Recipe::from_toml(SPEC_RECIPE).expect("spec §2.7 example parses");

    assert_eq!(recipe.default_provider(), Some("aten"));
    assert!(recipe.strict);
    assert_eq!(
        recipe.variant_for("rmsnorm", Phase::Forward),
        Some("cuda.fused_bf16")
    );
    // `backward = "autodiff"` is the strategy spelling from spec §2.7, not a
    // variant name: no plugin publishes a variant called `autodiff`, so it must
    // not become a lookup, and `backward_plan` is how a caller learns to
    // differentiate the expansion instead.
    assert_eq!(recipe.variant_for("mlp_swiglu", Phase::Backward), None);
    assert_eq!(
        recipe.backward_plan("mlp_swiglu"),
        Some(BackwardPlan::Autodiff)
    );
    assert_eq!(
        recipe.backward_plan("rmsnorm"),
        Some(BackwardPlan::Variant("cuda.fused_bf16".to_string()))
    );
    assert_eq!(recipe.variant_for("mlp_swiglu", Phase::Update), None);
    assert!(recipe.check_expansion_for("mlp_swiglu"));

    let numerics = recipe.precision.numerics(Phase::Forward);
    assert_eq!(numerics.in_dtype.name(), "bf16");
    assert_eq!(numerics.accum_dtype.name(), "f32");
    assert_eq!(numerics.grad_dtype.name(), "bf16");
    assert_eq!((numerics.block_m, numerics.block_n), (128, 128));
    assert_eq!(numerics.amax_history, 8);

    // Serialisation always writes the canonical ABI spelling, so an alias can
    // never leak into a plan digest.
    let encoded = recipe.to_toml().expect("recipe encodes");
    assert!(encoded.contains("accumulate = \"f32\""), "{encoded}");
    assert!(encoded.contains("weights = \"f8e4m3\""), "{encoded}");
    assert_eq!(Recipe::from_toml(&encoded).unwrap(), recipe);

    assert_eq!(recipe.parallel.world_size(), 8);
    assert!(recipe.parallel.overlap_collectives);
}

#[test]
fn every_documented_name_spelling_parses() {
    for (text, expected) in [
        ("f32", "f32"),
        ("fp32", "f32"),
        ("f16", "f16"),
        ("bf16", "bf16"),
        ("f8e4m3", "f8e4m3"),
        ("fp8e4m3", "f8e4m3"),
        ("fp8_e4m3", "f8e4m3"),
        ("fp4e2m1", "fp4e2m1"),
        ("i64", "i64"),
    ] {
        assert_eq!(DtypeName::parse(text).unwrap().name(), expected, "{text}");
    }
    assert_eq!(QuantName::parse("per_block").unwrap().name(), "per_block");
    assert_eq!(ScaleName::parse("delayed").unwrap().name(), "delayed");

    // Unknown spellings stay hard errors that say what would work.
    let error = DtypeName::parse("bf19").unwrap_err().to_string();
    assert!(error.contains("bf19"), "{error}");
    assert!(error.contains("bf16"), "{error}");
}

#[test]
fn recipe_documents_reject_unknown_keys() {
    // A typo must fail, and toml renders the offending key.
    let error = Recipe::from_toml(
        r#"
        [kernel]
        stict = true
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("stict"), "{error}");
    assert!(error.contains("unknown field"), "{error}");
}

#[test]
fn phase_names_are_stable_for_digests() {
    // These strings end up in plan digests; renaming one changes every digest.
    assert_eq!(
        serde_json::to_string(&Phase::Forward).unwrap(),
        "\"forward\""
    );
    assert_eq!(
        serde_json::to_string(&Phase::Backward).unwrap(),
        "\"backward\""
    );
    assert_eq!(serde_json::to_string(&Phase::Update).unwrap(), "\"update\"");
    assert_eq!(
        serde_json::from_str::<Phase>("\"backward\"").unwrap(),
        Phase::Backward
    );
}

#[test]
fn op_summary_json_is_stable() {
    let summary = OpSummary {
        op: "mlp_swiglu".to_string(),
        variant: "cuda.fp8_block128".to_string(),
        plugin: "cuda_kernels@0.2.0".to_string(),
        plugin_origin: "/opt/rustrain/libcuda_kernels.so".to_string(),
        dtypes: vec!["bf16".to_string(), "f8e4m3".to_string()],
        backward: "EXPLICIT".to_string(),
        has_expansion: true,
        is_composite: true,
    };
    let json = serde_json::to_string(&summary).unwrap();
    assert!(json.contains("\"variant\":\"cuda.fp8_block128\""), "{json}");
    assert_eq!(serde_json::from_str::<OpSummary>(&json).unwrap(), summary);
}

#[test]
fn vocabulary_marks_model_blocks_as_composites() {
    assert!(is_composite_op("mlp_swiglu"));
    assert!(is_composite_op("flash_attn"));
    // Primitives may declare an expansion, but are not required to.
    assert!(!is_composite_op("rmsnorm"));
    assert!(!is_composite_op("all_reduce"));
    assert!(COMPOSITE_OPS.contains(&"transformer_layer"));
}

#[test]
fn empty_registry_reports_that_nothing_is_loaded() {
    let registry = Registry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
    assert!(registry.describe().is_empty());
    assert!(registry.op_names().is_empty());
    assert!(registry.candidates("rmsnorm").is_empty());

    let mut request = rustrain_ops::ResolveRequest {
        name: "rmsnorm".to_string(),
        ..Default::default()
    };
    request.dtypes = vec![];
    request.env = TargetEnv::default();
    let error = registry.resolve(&request).unwrap_err();
    assert!(matches!(error, ResolveError::UnknownOp { .. }));
    assert!(error.to_string().contains("no plugin has been loaded"));
}
