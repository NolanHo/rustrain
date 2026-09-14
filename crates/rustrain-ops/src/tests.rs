//! Unit tests for the registry, the capability model and the recipe.
//!
//! Descriptors are built here rather than `dlopen`ed: `Registry::add_detached`
//! exists exactly so that the *registration and resolution* logic can be tested
//! without a compiled plugin, and it feeds the same private `register` path that
//! [`Registry::add_plugin`] uses — duplicate detection, all-or-nothing
//! registration and ordering are therefore exercised through the production
//! code, not a copy of it.
//!
//! What stays uncovered here is the two lines of `add_plugin` that turn the
//! loader's `LoadedOp`s into handles (`RegisteredOp::from_loaded`) plus the
//! `Plugin` → `.so` lifetime handoff; both need a real plugin binary.
//! `cargo test -p rustrain-abi` covers *loading* a C plugin, but nothing in the
//! workspace yet loads one into a `Registry`. Adding that needs a fixture `.so`
//! compiled from this crate's test build, which the descriptor path makes
//! unnecessary for everything except those two lines.

use std::ffi::c_char;
use std::path::PathBuf;
use std::ptr;

use rustrain_abi::{
    ABI_VERSION, RsAttrs, RsBackwardKind, RsCtx, RsDeviceKind, RsDtype, RsExpansion,
    RsExpansionNode, RsGroupKind, RsNumerics, RsOpDesc, RsOpId, RsQuantKind, RsRequires,
    RsScaleMode, RsTensor,
};

use crate::capability::*;
use crate::recipe::*;
use crate::registered::*;
use crate::registry::*;

// ── fixtures ────────────────────────────────────────────────────────────────

/// Leaks a NUL-terminated copy of `text`.
///
/// Descriptors are process-lifetime by construction (that is the ABI's model),
/// so leaking in a test is faithful rather than sloppy.
fn cptr(text: &str) -> *const c_char {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    Box::leak(bytes.into_boxed_slice()).as_ptr() as *const c_char
}

unsafe extern "C" fn fake_execute(
    _ctx: *mut RsCtx,
    _in: *const *const RsTensor,
    _n_in: u32,
    _out: *const *mut RsTensor,
    _n_out: u32,
    _attrs: *const RsAttrs,
) -> i32 {
    0
}

/// Declarative description of a synthetic operator.
struct Spec {
    name: String,
    variant: String,
    doc: String,
    /// `None` = publish no `requires` at all. `Some(vec![])` = publish an
    /// all-zero `requires` ("declares nothing"). `Some(list)` = that dtype set.
    dtypes: Option<Vec<RsDtype>>,
    min_sm: u32,
    min_world_size: i64,
    groups: Vec<RsGroupKind>,
    backward: RsBackwardKind,
    backward_op: Option<(String, String)>,
    expansion: bool,
}

impl Spec {
    fn new(name: &str, variant: &str) -> Self {
        Self {
            name: name.to_string(),
            variant: variant.to_string(),
            doc: format!("{name} implementation {variant}"),
            dtypes: Some(vec![RsDtype::F32, RsDtype::BF16, RsDtype::F16]),
            min_sm: 0,
            min_world_size: 0,
            groups: Vec::new(),
            backward: RsBackwardKind::AUTODIFF,
            backward_op: None,
            expansion: false,
        }
    }

    fn dtypes(mut self, dtypes: &[RsDtype]) -> Self {
        self.dtypes = Some(dtypes.to_vec());
        self
    }

    fn no_requires(mut self) -> Self {
        self.dtypes = None;
        self
    }

    fn zero_requires(mut self) -> Self {
        self.dtypes = Some(Vec::new());
        self.min_sm = 0;
        self.min_world_size = 0;
        self.groups.clear();
        self
    }

    fn min_sm(mut self, sm: u32) -> Self {
        self.min_sm = sm;
        self
    }

    fn min_world_size(mut self, world: i64) -> Self {
        self.min_world_size = world;
        self
    }

    fn groups(mut self, groups: &[RsGroupKind]) -> Self {
        self.groups = groups.to_vec();
        self
    }

    fn backward(mut self, kind: RsBackwardKind) -> Self {
        self.backward = kind;
        if kind == RsBackwardKind::EXPLICIT {
            self.backward_op = Some((format!("{}_backward", self.name), self.variant.clone()));
        }
        self
    }

    fn expansion(mut self) -> Self {
        self.expansion = true;
        self
    }

    fn build(self) -> &'static RsOpDesc {
        let requires = self.dtypes.map(|dtypes| {
            Box::leak(Box::new(RsRequires {
                dtype_mask: RsRequires::dtype_mask_for(&dtypes),
                min_sm: self.min_sm,
                min_world_size: self.min_world_size,
                needs_groups: self.groups.iter().fold(0u32, |acc, g| acc | g.0),
                _pad: 0,
            })) as *const RsRequires
        });

        let expansion = if self.expansion {
            // One node with no operands: enough to declare "this operator
            // publishes an expansion" without inventing an io mapping.
            let nodes = Box::leak(
                vec![RsExpansionNode {
                    op: cptr("matmul"),
                    attrs: ptr::null(),
                    inputs: ptr::null(),
                    n_inputs: 0,
                    outputs: ptr::null(),
                    n_outputs: 0,
                }]
                .into_boxed_slice(),
            );
            Some(Box::leak(Box::new(RsExpansion {
                n_nodes: 1,
                _pad: 0,
                nodes: nodes.as_ptr(),
                n_tensors: 2,
                n_inputs: 1,
                n_outputs: 1,
                _pad2: 0,
            })) as *const RsExpansion)
        } else {
            None
        };

        let (backward_op_name, backward_op_variant) = match &self.backward_op {
            Some((name, variant)) => (cptr(name), cptr(variant)),
            None => (ptr::null(), ptr::null()),
        };

        Box::leak(Box::new(RsOpDesc {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<RsOpDesc>() as u32,
            id: RsOpId {
                name: cptr(&self.name),
                variant: cptr(&self.variant),
                version: 1,
            },
            doc: cptr(&self.doc),
            requires: requires.unwrap_or(ptr::null()),
            numerics: RsNumerics::default(),
            infer: None,
            memory: None,
            expansion: expansion.unwrap_or(ptr::null()),
            backward: self.backward,
            backward_op: RsOpId {
                name: backward_op_name,
                variant: backward_op_variant,
                version: 1,
            },
            collectives: ptr::null(),
            n_collectives: 0,
            execute: Some(fake_execute),
            last_error: None,
            shard: rustrain_abi::ffi::RsShardRule::DECLARED,
        }))
    }
}

/// Registers one plugin per `(name, descs)` pair.
fn registry(plugins: &[(&str, &[&'static RsOpDesc])]) -> Registry {
    let mut registry = Registry::new();
    for (name, descs) in plugins {
        let added = registry
            .add_detached(name, "0.1.0", format!("/plugins/{name}.so"), descs)
            .expect("test plugin registration");
        assert_eq!(added, descs.len());
    }
    registry
}

/// `RsNumerics` has no `PartialEq`/`Debug` in the ABI crate (it is a POD
/// mirror), so compare it field by field through a comparable projection.
#[allow(clippy::type_complexity)]
fn numerics_fields(
    n: &RsNumerics,
) -> (
    RsDtype,
    RsDtype,
    RsDtype,
    RsDtype,
    RsQuantKind,
    u32,
    u32,
    RsDtype,
    RsScaleMode,
    u32,
) {
    (
        n.in_dtype,
        n.out_dtype,
        n.accum_dtype,
        n.grad_dtype,
        n.quant,
        n.block_m,
        n.block_n,
        n.scale_dtype,
        n.scale_mode,
        n.amax_history,
    )
}

fn cpu_env() -> TargetEnv {
    TargetEnv::default()
}

fn cuda_env(sm: u32, world_size: usize) -> TargetEnv {
    TargetEnv {
        world_size,
        sm: Some(sm),
        groups: Vec::new(),
        device: RsDeviceKind::CUDA,
    }
}

fn request(name: &str, phase: Phase, dtypes: &[RsDtype]) -> ResolveRequest {
    ResolveRequest {
        name: name.to_string(),
        phase,
        dtypes: dtypes.to_vec(),
        ..ResolveRequest::default()
    }
}

// ── registration ────────────────────────────────────────────────────────────

#[test]
fn duplicate_op_variant_is_rejected_and_names_both_plugins() {
    let first = [Spec::new("rmsnorm", "reference.f32").build()];
    let second = [Spec::new("rmsnorm", "reference.f32").build()];

    let mut registry = Registry::new();
    assert_eq!(
        registry
            .add_detached("reference", "0.1.0", "/plugins/reference.so", &first)
            .unwrap(),
        1
    );

    let error = registry
        .add_detached("aten", "2.0.0", "/plugins/aten.so", &second)
        .unwrap_err();
    let text = error.to_string();
    for needle in [
        "rmsnorm@reference.f32",
        "reference@0.1.0",
        "/plugins/reference.so",
        "aten@2.0.0",
        "/plugins/aten.so",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
    }

    // All-or-nothing: the rejected plugin contributed nothing.
    assert_eq!(registry.len(), 1);
}

#[test]
fn duplicate_inside_one_plugin_is_rejected_and_adds_nothing() {
    let descs = [
        Spec::new("rmsnorm", "reference.f32").build(),
        Spec::new("rmsnorm", "reference.f32").build(),
    ];
    let mut registry = Registry::new();
    let error = registry
        .add_detached("reference", "0.1.0", "/plugins/reference.so", &descs)
        .unwrap_err();
    assert!(error.to_string().contains("rmsnorm@reference.f32"));
    assert!(registry.is_empty());
}

#[test]
fn candidates_are_sorted_by_variant() {
    // Registered out of order on purpose: candidate order must come from the
    // registry, not from plugin iteration order, or digests would wobble.
    let descs = [
        Spec::new("rmsnorm", "zulu.f32").build(),
        Spec::new("rmsnorm", "alpha.bf16").build(),
        Spec::new("matmul", "aten.f32").build(),
        Spec::new("rmsnorm", "mike.f16").build(),
    ];
    let registry = registry(&[("reference", &descs)]);

    let variants: Vec<&str> = registry
        .candidates("rmsnorm")
        .iter()
        .map(|op| op.variant())
        .collect();
    assert_eq!(variants, ["alpha.bf16", "mike.f16", "zulu.f32"]);

    assert_eq!(registry.op_names(), ["matmul", "rmsnorm"]);

    let described: Vec<(String, String)> = registry
        .describe()
        .into_iter()
        .map(|summary| (summary.op, summary.variant))
        .collect();
    assert_eq!(
        described,
        [
            ("matmul".to_string(), "aten.f32".to_string()),
            ("rmsnorm".to_string(), "alpha.bf16".to_string()),
            ("rmsnorm".to_string(), "mike.f16".to_string()),
            ("rmsnorm".to_string(), "zulu.f32".to_string()),
        ]
    );
}

#[test]
fn describe_reports_plugin_identity_dtypes_and_expansion() {
    let primitives = [Spec::new("rmsnorm", "reference.f32")
        .dtypes(&[RsDtype::F32, RsDtype::BF16])
        .build()];
    let fused = [Spec::new("mlp_swiglu", "cuda.fp8_block128")
        .expansion()
        .backward(RsBackwardKind::EXPLICIT)
        .build()];
    let registry = registry(&[("reference", &primitives), ("cuda_kernels", &fused)]);
    let summaries = registry.describe();

    let rmsnorm = summaries
        .iter()
        .find(|s| s.op == "rmsnorm")
        .expect("rmsnorm summary");
    assert_eq!(rmsnorm.plugin, "reference@0.1.0");
    assert_eq!(rmsnorm.plugin_origin, "/plugins/reference.so");
    assert_eq!(rmsnorm.dtypes, ["f32", "bf16"]);
    assert_eq!(rmsnorm.backward, "AUTODIFF");
    assert!(!rmsnorm.has_expansion);
    // `rmsnorm` is a primitive: an expansion is allowed but not required.
    assert!(!rmsnorm.is_composite);

    let fused = summaries
        .iter()
        .find(|s| s.op == "mlp_swiglu")
        .expect("mlp_swiglu summary");
    assert!(fused.has_expansion);
    // Spec §2.4: `mlp_swiglu` is a model block, so R-4 requires an expansion.
    assert!(fused.is_composite);
    assert_eq!(fused.backward, "EXPLICIT");
}

// ── capability model ────────────────────────────────────────────────────────

#[test]
fn dtype_mismatch_names_requested_and_accepted() {
    let descs = [Spec::new("linear", "reference.f32")
        .dtypes(&[RsDtype::F32, RsDtype::BF16])
        .build()];
    let registry = registry(&[("reference", &descs)]);
    let op = registry.candidates("linear")[0];

    assert_eq!(reject_reason(op, &[RsDtype::BF16], &cpu_env()), None);

    let reason = reject_reason(op, &[RsDtype::BF16, RsDtype::F8E4M3], &cpu_env())
        .expect("f8e4m3 must be rejected");
    assert_eq!(
        reason,
        RejectReason::DtypeUnsupported {
            requested: RsDtype::F8E4M3,
            accepted: vec![RsDtype::F32, RsDtype::BF16],
        }
    );
    let text = reason.to_string();
    assert!(text.contains("f8e4m3"), "{text}");
    assert!(text.contains("f32, bf16"), "{text}");
}

#[test]
fn sm_world_size_and_group_requirements_are_enforced() {
    let descs = [
        Spec::new("flash_attn", "cuda.sm90").min_sm(90).build(),
        Spec::new("all_reduce", "cuda.tp").min_world_size(2).build(),
        Spec::new("all_reduce", "cuda.dp2")
            .groups(&[RsGroupKind::DP])
            .build(),
    ];
    let registry = registry(&[("cuda_kernels", &descs)]);

    let sm90 = registry
        .candidates("flash_attn")
        .into_iter()
        .next()
        .unwrap();
    let unknown_sm = TargetEnv {
        sm: None,
        ..cuda_env(0, 1)
    };
    // Unknown capability is not "good enough".
    assert_eq!(
        reject_reason(sm90, &[], &unknown_sm),
        Some(RejectReason::SmUnknown { required: 90 })
    );
    assert_eq!(
        reject_reason(sm90, &[], &cuda_env(89, 1)),
        Some(RejectReason::SmTooLow {
            required: 90,
            available: 89,
        })
    );
    assert_eq!(reject_reason(sm90, &[], &cuda_env(90, 1)), None);

    // Candidates are sorted by variant: cuda.dp2 comes before cuda.tp.
    let dp2 = registry.candidates("all_reduce")[0].clone();
    let tp = registry.candidates("all_reduce")[1].clone();
    assert_eq!(
        reject_reason(&tp, &[], &cuda_env(90, 1)),
        Some(RejectReason::WorldSizeTooSmall {
            required: 2,
            available: 1,
        })
    );

    let mut env = cuda_env(90, 2);
    assert_eq!(
        reject_reason(&dp2, &[], &env),
        Some(RejectReason::MissingGroup {
            group: RsGroupKind::DP
        })
    );
    env.groups.push(RsGroupKind::DP);
    assert_eq!(reject_reason(&dp2, &[], &env), None);
}

#[test]
fn unconstrained_variants_accept_everything() {
    let descs = [
        Spec::new("linear", "no.declaration").no_requires().build(),
        Spec::new("linear", "zeroed.declaration")
            .zero_requires()
            .build(),
    ];
    let registry = registry(&[("reference", &descs)]);

    for op in registry.candidates("linear") {
        // No `requires` pointer, or an all-zero one: nothing is declared, so
        // nothing is rejected — including for a dtype the ABI hardly knows.
        assert_eq!(reject_reason(op, &[RsDtype::FP4E2M1], &cpu_env()), None);
        assert!(op.requires().map(|r| r.dtype_mask).unwrap_or(0) == 0);
    }

    // The summary must not report "accepts nothing" for them either.
    for summary in registry.describe() {
        assert_eq!(summary.dtypes.len(), RsDtype::ALL.len(), "{summary:?}");
    }
}

#[test]
fn partial_requires_with_zero_dtype_mask_accepts_no_dtype() {
    // The header documents `dtype_mask` as "bit i set => dtype i accepted", so
    // a partially filled `requires` that leaves the mask at 0 accepts nothing.
    // Reading it as "no constraint" would silently hand a kernel a dtype it
    // never declared.
    let descs = [Spec::new("linear", "cuda.sm90")
        .zero_requires()
        .min_sm(90)
        .build()];
    let registry = registry(&[("cuda_kernels", &descs)]);
    let op = registry.candidates("linear")[0];

    let reason = reject_reason(op, &[RsDtype::BF16], &cuda_env(90, 1)).expect("must be rejected");
    assert_eq!(
        reason,
        RejectReason::DtypeUnsupported {
            requested: RsDtype::BF16,
            accepted: Vec::new(),
        }
    );
    assert!(reason.to_string().contains("dtype_mask is 0"));
    // No dtype asked for => no dtype constraint to violate.
    assert_eq!(reject_reason(op, &[], &cuda_env(90, 1)), None);
}

#[test]
fn device_namespace_declares_a_device() {
    assert_eq!(
        declared_device("cuda.fp8_block128"),
        Some(RsDeviceKind::CUDA)
    );
    assert_eq!(declared_device("cpu.avx512"), Some(RsDeviceKind::CPU));
    // Unfamiliar namespaces impose nothing: rejecting unknown prefixes would
    // make a third-party naming scheme unusable.
    assert_eq!(declared_device("reference.f32"), None);
    assert_eq!(declared_device("autodiff"), None);

    let descs = [
        Spec::new("rmsnorm", "cuda.fused_bf16").build(),
        Spec::new("rmsnorm", "cpu.avx512").build(),
        Spec::new("rmsnorm", "reference.f32").build(),
    ];
    let registry = registry(&[("kernels", &descs)]);

    let reasons: Vec<Option<RejectReason>> = registry
        .candidates("rmsnorm")
        .iter()
        .map(|op| reject_reason(op, &[], &cpu_env()))
        .collect();
    // Candidates are sorted by variant: cpu.avx512, cuda.fused_bf16, reference.f32.
    assert_eq!(reasons[0], None);
    assert_eq!(
        reasons[1],
        Some(RejectReason::DeviceUnsupported {
            requested: RsDeviceKind::CPU,
            supported: RsDeviceKind::CUDA,
        })
    );
    assert_eq!(reasons[2], None);
}

#[test]
fn backward_selection_needs_an_explicit_backward() {
    let descs = [
        Spec::new("rmsnorm", "reference.f32").build(),
        Spec::new("rmsnorm", "cuda.fused_bf16")
            .backward(RsBackwardKind::EXPLICIT)
            .build(),
        Spec::new("quantize", "reference.f32")
            .backward(RsBackwardKind::NONDIFF)
            .build(),
    ];
    let registry = registry(&[("reference", &descs)]);

    // Sorted by variant: cuda.fused_bf16 (EXPLICIT) first, reference.f32 second.
    let explicit = registry.candidates("rmsnorm")[0].clone();
    let autodiff = registry.candidates("rmsnorm")[1].clone();
    assert_eq!(explicit.variant(), "cuda.fused_bf16");
    assert_eq!(autodiff.variant(), "reference.f32");

    assert_eq!(phase_reject_reason(&autodiff, Phase::Forward), None);
    assert_eq!(
        phase_reject_reason(&autodiff, Phase::Backward),
        Some(RejectReason::PhaseUnsupported {
            phase: Phase::Backward,
            backward_kind: RsBackwardKind::AUTODIFF,
        })
    );
    assert!(autodiff.backward_op_id().is_none());

    assert_eq!(phase_reject_reason(&explicit, Phase::Backward), None);
    assert_eq!(
        explicit.backward_op_id(),
        Some((
            "rmsnorm_backward".to_string(),
            "cuda.fused_bf16".to_string()
        ))
    );

    let nondiff = registry.candidates("quantize")[0].clone();
    assert!(matches!(
        phase_reject_reason(&nondiff, Phase::Backward),
        Some(RejectReason::PhaseUnsupported {
            backward_kind: RsBackwardKind::NONDIFF,
            ..
        })
    ));
    // Update is never phase-rejected: an optimizer has no "backward kind".
    assert_eq!(phase_reject_reason(&nondiff, Phase::Update), None);
}

// ── resolution ──────────────────────────────────────────────────────────────

#[test]
fn prefer_missing_variant_errors_without_falling_back() {
    let descs = [Spec::new("rmsnorm", "reference.f32").build()];
    let registry = registry(&[("reference", &descs)]);

    let mut req = request("rmsnorm", Phase::Forward, &[RsDtype::F32]);
    req.prefer = Some("cuda.fused_bf16".to_string());
    // A perfectly good fallback is named: it must NOT be used (contract R-1).
    req.fallback = vec!["reference.f32".to_string()];

    let error = registry.resolve(&req).unwrap_err();
    assert!(
        matches!(error, ResolveError::PreferredNotPublished { .. }),
        "unexpected error: {error:?}"
    );
    let text = error.to_string();
    assert!(text.contains("rmsnorm@cuda.fused_bf16"), "{text}");
    assert!(text.contains("reference.f32"), "{text}");
    assert!(text.contains("never falls through"), "{text}");
}

#[test]
fn prefer_rejected_variant_errors_without_falling_back() {
    let descs = [
        Spec::new("rmsnorm", "cuda.fused_bf16")
            .dtypes(&[RsDtype::BF16])
            .min_sm(90)
            .build(),
        Spec::new("rmsnorm", "reference.f32")
            .dtypes(&[RsDtype::BF16])
            .build(),
    ];
    let registry = registry(&[("kernels", &descs)]);

    let mut req = request("rmsnorm", Phase::Forward, &[RsDtype::BF16]);
    req.prefer = Some("cuda.fused_bf16".to_string());
    req.fallback = vec!["reference.f32".to_string()];
    req.env = cuda_env(89, 1);

    let error = registry.resolve(&req).unwrap_err();
    match &error {
        ResolveError::PreferredRejected {
            variant, reason, ..
        } => {
            assert_eq!(variant, "cuda.fused_bf16");
            assert_eq!(
                reason,
                &RejectReason::SmTooLow {
                    required: 90,
                    available: 89
                }
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
    let text = error.to_string();
    assert!(text.contains("sm 89 < required 90"), "{text}");
    assert!(text.contains("no fallback was attempted"), "{text}");
    // The candidate table still shows the variant that would have worked.
    assert!(text.contains("reference.f32"), "{text}");
}

#[test]
fn fallback_order_is_respected_and_skips_are_recorded() {
    let descs = [
        Spec::new("rmsnorm", "a.f32").min_world_size(4).build(),
        Spec::new("rmsnorm", "b.f32").build(),
        Spec::new("rmsnorm", "c.f32").build(),
    ];
    let registry = registry(&[("kernels", &descs)]);

    let mut req = request("rmsnorm", Phase::Forward, &[RsDtype::F32]);
    req.fallback = vec![
        "missing.variant".to_string(),
        "a.f32".to_string(),
        "b.f32".to_string(),
        "c.f32".to_string(),
    ];
    req.env = TargetEnv {
        world_size: 1,
        ..cpu_env()
    };

    let resolved = registry.resolve(&req).expect("b.f32 must win");
    assert_eq!(resolved.variant(), "b.f32");
    assert!(resolved.degraded());

    let skipped: Vec<(String, RejectReason)> = resolved.rejected.clone();
    assert_eq!(
        skipped.len(),
        2,
        "c.f32 must not even be tried: {skipped:?}"
    );
    assert_eq!(
        skipped[0],
        (
            "missing.variant".to_string(),
            RejectReason::NotPublished {
                variant: "missing.variant".to_string()
            }
        )
    );
    assert_eq!(
        skipped[1],
        (
            "a.f32".to_string(),
            RejectReason::WorldSizeTooSmall {
                required: 4,
                available: 1
            }
        )
    );

    // Contract R-2: the degradation is visible in the one-line report.
    let report = resolved.report();
    assert!(report.contains("rmsnorm@b.f32"), "{report}");
    assert!(report.contains("degraded"), "{report}");
    assert!(report.contains("missing.variant"), "{report}");
}

#[test]
fn unresolvable_op_lists_every_candidate_with_its_reason() {
    let descs = [
        Spec::new("mlp_swiglu", "autodiff")
            .dtypes(&[RsDtype::BF16, RsDtype::F16])
            .build(),
        Spec::new("mlp_swiglu", "cuda.fp8_block128")
            .dtypes(&[RsDtype::BF16, RsDtype::F8E4M3])
            .min_sm(90)
            .build(),
        Spec::new("mlp_swiglu", "cpu.avx512")
            .min_world_size(8)
            .build(),
    ];
    let registry = registry(&[("kernels", &descs)]);

    let mut req = request(
        "mlp_swiglu",
        Phase::Forward,
        &[RsDtype::BF16, RsDtype::F8E4M3],
    );
    req.fallback = vec!["aten.fp8".to_string()];
    req.env = cuda_env(89, 2);

    let error = registry.resolve(&req).unwrap_err();
    assert!(
        matches!(error, ResolveError::Unresolved(_)),
        "unexpected error: {error:?}"
    );
    let text = error.to_string();

    // Every candidate, with the specific reason it was rejected.
    assert!(text.contains("mlp_swiglu@autodiff"), "{text}");
    assert!(
        text.contains("dtype f8e4m3 is not accepted (variant accepts: f16, bf16)"),
        "{text}"
    );
    assert!(text.contains("mlp_swiglu@cuda.fp8_block128"), "{text}");
    assert!(text.contains("sm 89 < required 90"), "{text}");
    assert!(text.contains("mlp_swiglu@cpu.avx512"), "{text}");
    assert!(
        text.contains("device cuda requested but this variant is a cpu implementation"),
        "{text}"
    );
    assert!(
        text.contains("named but not published: mlp_swiglu@aten.fp8"),
        "{text}"
    );

    // …and the request it failed to satisfy.
    assert!(text.contains("dtypes [bf16, f8e4m3]"), "{text}");
    assert!(text.contains("phase forward"), "{text}");
    assert!(text.contains("world_size=2"), "{text}");
    assert!(text.contains("sm=89"), "{text}");
    assert!(text.contains("device=cuda"), "{text}");
    assert!(text.contains("contract R-1"), "{text}");

    // The structured form carries the same table for programmatic use.
    let failure = error.failure().expect("failure table");
    assert_eq!(failure.candidates.len(), 3);
    assert!(failure.eligible.is_empty());
    assert_eq!(failure.unpublished, ["aten.fp8"]);
}

#[test]
fn unknown_op_lists_what_is_registered() {
    let descs = [
        Spec::new("rmsnorm", "reference.f32").build(),
        Spec::new("matmul", "reference.f32").build(),
    ];
    let registry = registry(&[("reference", &descs)]);

    let error = registry
        .resolve(&request("layernorm", Phase::Forward, &[]))
        .unwrap_err();
    assert!(matches!(error, ResolveError::UnknownOp { .. }));
    let text = error.to_string();
    assert!(text.contains("layernorm"), "{text}");
    assert!(text.contains("matmul, rmsnorm"), "{text}");

    let empty = Registry::new()
        .resolve(&request("rmsnorm", Phase::Forward, &[]))
        .unwrap_err()
        .to_string();
    assert!(empty.contains("nothing is registered at all"), "{empty}");
}

#[test]
fn default_provider_decides_when_nothing_is_named() {
    let reference = [Spec::new("rmsnorm", "reference.f32").build()];
    let aten = [Spec::new("rmsnorm", "aten.bf16")
        .dtypes(&[RsDtype::BF16])
        .build()];
    let cuda = [Spec::new("rmsnorm", "cuda.sm90").min_sm(90).build()];
    let registry = registry(&[
        ("reference", &reference),
        ("aten", &aten),
        ("cuda_kernels", &cuda),
    ]);

    let req = request("rmsnorm", Phase::Forward, &[RsDtype::BF16]);
    let resolved = registry
        .resolve_with_default(&req, Some("reference"))
        .expect("reference.f32 is the only runnable variant of that provider");
    assert_eq!(resolved.variant(), "reference.f32");
    assert!(
        !resolved.degraded(),
        "an unambiguous pick is not a degradation"
    );
}

#[test]
fn ambiguous_when_the_default_provider_publishes_two_runnable_variants() {
    let descs = [
        Spec::new("rmsnorm", "aten.f32").build(),
        Spec::new("rmsnorm", "aten.bf16").build(),
    ];
    let registry = registry(&[("aten", &descs)]);

    let req = request("rmsnorm", Phase::Forward, &[RsDtype::F32]);
    let error = registry
        .resolve_with_default(&req, Some("aten"))
        .unwrap_err();
    assert!(matches!(error, ResolveError::Ambiguous(_)));
    let text = error.to_string();
    assert!(text.contains("publishes 2 runnable variants"), "{text}");
    assert!(text.contains("aten.bf16, aten.f32"), "{text}");
}

#[test]
fn ambiguous_without_a_default_provider() {
    let descs = [Spec::new("rmsnorm", "reference.f32").build()];
    let registry = registry(&[("reference", &descs)]);

    // One candidate exists, but the request names nobody to prefer it: picking
    // it would be exactly the implicit selection R-1 forbids.
    let error = registry
        .resolve(&request("rmsnorm", Phase::Forward, &[RsDtype::F32]))
        .unwrap_err();
    assert!(matches!(error, ResolveError::Ambiguous(_)));
    let text = error.to_string();
    assert!(text.contains("[kernel].default"), "{text}");
    assert!(text.contains("runnable here but never named"), "{text}");
}

#[test]
fn empty_fallback_chain_assigns_no_blame_and_reports_zero_eligible() {
    let descs = [Spec::new("rmsnorm", "cuda.sm90").min_sm(90).build()];
    let registry = registry(&[("kernels", &descs)]);

    let mut req = request("rmsnorm", Phase::Forward, &[]);
    req.env = cuda_env(80, 1);

    let error = registry
        .resolve_with_default(&req, Some("kernels"))
        .unwrap_err();
    let text = error.to_string();
    assert!(
        text.contains(
            "provider `kernels` publishes 1 variant of `rmsnorm`, none of which can run here"
        ),
        "{text}"
    );
    assert!(text.contains("sm 80 < required 90"), "{text}");
}

#[test]
fn backward_phase_resolution_requires_explicit() {
    let descs = [
        Spec::new("rmsnorm", "reference.f32").build(),
        Spec::new("rmsnorm", "cuda.fused_bf16")
            .backward(RsBackwardKind::EXPLICIT)
            .build(),
    ];
    let registry = registry(&[("kernels", &descs)]);

    let mut req = request("rmsnorm", Phase::Backward, &[RsDtype::BF16]);
    req.prefer = Some("reference.f32".to_string());
    let error = registry.resolve(&req).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("phase backward is not selectable"), "{text}");
    assert!(text.contains("backward = AUTODIFF"), "{text}");

    let mut req = request("rmsnorm", Phase::Backward, &[RsDtype::BF16]);
    req.prefer = Some("cuda.fused_bf16".to_string());
    req.env = cuda_env(90, 1);
    let resolved = registry
        .resolve(&req)
        .expect("EXPLICIT variant is selectable");
    assert_eq!(resolved.variant(), "cuda.fused_bf16");
}

// ── recipe ──────────────────────────────────────────────────────────────────

const SPEC_RECIPE: &str = r#"
[kernel]
default = "aten"
strict  = false

[kernel.ops.rmsnorm]
forward  = "cuda.fused_bf16"
backward = "cuda.fused_bf16"

[kernel.ops.mlp_swiglu]
forward         = "cuda.fp8_block128"
fallback        = ["aten.bf16", "reference.f32"]
check_expansion = true

[kernel.precision]
compute        = "bf16"
accumulate     = "fp32"
master_weights = "fp32"
grad           = "bf16"
weights        = "fp8e4m3"
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
fn spec_2_7_recipe_parses_and_maps_to_the_abi() {
    let recipe = Recipe::from_toml(SPEC_RECIPE).expect("spec §2.7 example must parse");

    assert_eq!(recipe.default, "aten");
    assert_eq!(recipe.default_provider(), Some("aten"));
    assert!(!recipe.strict);
    assert!(recipe.check_expansion_for("mlp_swiglu"));
    assert!(!recipe.check_expansion_for("rmsnorm"));

    // Precedence: op-level forward/backward, then the resolver's fallback list.
    assert_eq!(
        recipe.variant_for("rmsnorm", Phase::Forward),
        Some("cuda.fused_bf16")
    );
    assert_eq!(
        recipe.variant_for("rmsnorm", Phase::Backward),
        Some("cuda.fused_bf16")
    );
    assert_eq!(
        recipe.variant_for("mlp_swiglu", Phase::Forward),
        Some("cuda.fp8_block128")
    );
    assert_eq!(recipe.variant_for("mlp_swiglu", Phase::Backward), None);
    assert_eq!(recipe.variant_for("mlp_swiglu", Phase::Update), None);
    assert_eq!(recipe.variant_for("unknown_op", Phase::Forward), None);

    let forward = recipe.precision.numerics(Phase::Forward);
    assert_eq!(forward.in_dtype, RsDtype::BF16);
    assert_eq!(forward.out_dtype, RsDtype::BF16);
    assert_eq!(forward.accum_dtype, RsDtype::F32);
    assert_eq!(forward.grad_dtype, RsDtype::BF16);
    assert_eq!(forward.quant, RsQuantKind::PER_BLOCK);
    assert_eq!((forward.block_m, forward.block_n), (128, 128));
    assert_eq!(forward.scale_mode, RsScaleMode::DELAYED);
    assert_eq!(forward.amax_history, 8);

    let backward = recipe.precision.numerics(Phase::Backward);
    assert_eq!(backward.in_dtype, RsDtype::BF16);
    assert_eq!(backward.out_dtype, RsDtype::BF16);
    assert_eq!(backward.accum_dtype, RsDtype::F32);

    let update = recipe.precision.numerics(Phase::Update);
    assert_eq!(update.in_dtype, RsDtype::F32);
    assert_eq!(update.out_dtype, RsDtype::F8E4M3);

    assert_eq!(recipe.parallel.world_size(), 8);
    assert!(recipe.parallel.overlap_collectives);
}

#[test]
fn precision_recipe_defaults_match_the_abi_zero_state() {
    let recipe = Recipe::default();
    for phase in Phase::ALL {
        assert_eq!(
            numerics_fields(&recipe.precision.numerics(phase)),
            numerics_fields(&RsNumerics::default())
        );
    }
    assert_eq!(recipe.parallel, ParallelRecipe::default());
    assert_eq!(recipe.parallel.world_size(), 1);
}

#[test]
fn recipe_round_trips_through_toml() {
    let recipe = Recipe::from_toml(SPEC_RECIPE).unwrap();
    let encoded = recipe.to_toml().unwrap();
    let reparsed = Recipe::from_toml(&encoded).unwrap();
    assert_eq!(reparsed, recipe);
}

#[test]
fn unknown_key_is_a_hard_error() {
    // `accum` is a typo for `accumulate`. The legacy behaviour was to ignore
    // it; now it must fail and say which key was not understood.
    let error = Recipe::from_toml(
        r#"
        [kernel.precision]
        compute = "bf16"
        accum  = "fp32"
        "#,
    )
    .unwrap_err();
    let text = error.to_string();
    assert!(text.contains("accum"), "{text}");
    assert!(text.contains("unknown field"), "{text}");

    // Same rule for an op entry and for the top-level document.
    let op_typo = Recipe::from_toml("[kernel.ops.rmsnorm]\nfoward = \"reference.f32\"\n")
        .unwrap_err()
        .to_string();
    assert!(op_typo.contains("foward"), "{op_typo}");

    let section_typo = Recipe::from_toml("[kernels]\ndefault = \"aten\"\n")
        .unwrap_err()
        .to_string();
    assert!(section_typo.contains("kernels"), "{section_typo}");
}

#[test]
fn unknown_dtype_name_is_a_hard_error_that_names_the_field() {
    let error = Recipe::from_toml(
        r#"
        [kernel.precision]
        compute = "bf19"
        "#,
    )
    .unwrap_err();
    let text = error.to_string();
    // The offending key must appear (toml renders the span), and the message
    // must list the accepted spellings.
    assert!(text.contains("compute"), "{text}");
    assert!(text.contains("bf19"), "{text}");
    assert!(text.contains("f32"), "{text}");
    assert!(text.contains("fp4e2m1"), "{text}");
}

#[test]
fn name_types_parse_and_reject_with_the_accepted_list() {
    assert_eq!(DtypeName::parse("bf16").unwrap().get(), RsDtype::BF16);
    assert_eq!(
        QuantName::parse("per_block").unwrap().get(),
        RsQuantKind::PER_BLOCK
    );
    assert_eq!(
        ScaleName::parse("delayed").unwrap().get(),
        RsScaleMode::DELAYED
    );

    let error = QuantName::parse("per_blk").unwrap_err().to_string();
    assert!(error.contains("per_blk"), "{error}");
    assert!(
        error.contains("per_tensor, per_token, per_block"),
        "{error}"
    );

    let error = ScaleName::parse("dynamic").unwrap_err().to_string();
    assert!(error.contains("dynamic_amax, delayed"), "{error}");
}

#[test]
fn dtype_names_cover_the_whole_abi_dtype_set() {
    // Pins the recipe vocabulary to the ABI: an added `rs_dtype` must be named
    // here too, or a recipe could not express it.
    let abi: Vec<&str> = RsDtype::ALL.iter().map(|d| d.name()).collect();
    let recipe: Vec<&str> = DtypeName::ALL.iter().map(|d| d.name()).collect();
    assert_eq!(recipe, abi);
    for dtype in RsDtype::ALL {
        assert_eq!(DtypeName::parse(dtype.name()).unwrap().get(), dtype);
    }
}

#[test]
fn contradictory_precision_settings_are_rejected() {
    let error = Recipe::from_toml(
        r#"
        [kernel.precision]
        quant_scheme = "per_block"
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("per_block"), "{error}");
    assert!(error.contains("block"), "{error}");

    let error = Recipe::from_toml(
        r#"
        [kernel.precision]
        scale_mode = "delayed"
        "#,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("amax_history"), "{error}");
}

#[test]
fn recipe_builds_a_resolve_request_from_the_file() {
    let recipe = Recipe::from_toml(SPEC_RECIPE).unwrap();
    let req = recipe.resolve_request("mlp_swiglu", Phase::Forward, &[RsDtype::BF16], &cpu_env());
    assert_eq!(req.prefer.as_deref(), Some("cuda.fp8_block128"));
    assert_eq!(req.fallback, ["aten.bf16", "reference.f32"]);
    assert_eq!(req.dtypes, [RsDtype::BF16]);

    // A phase with no op-level entry falls through to the fallback chain.
    let req = recipe.resolve_request("mlp_swiglu", Phase::Update, &[], &cpu_env());
    assert_eq!(req.prefer, None);
    assert_eq!(req.fallback, ["aten.bf16", "reference.f32"]);
}

#[test]
fn strict_recipe_refuses_a_degraded_resolution() {
    let descs = [
        Spec::new("mlp_swiglu", "aten.bf16")
            .min_world_size(4)
            .build(),
        Spec::new("mlp_swiglu", "reference.f32").build(),
    ];
    let registry = registry(&[("kernels", &descs)]);

    let recipe = Recipe::from_toml(
        r#"
        [kernel]
        strict = true

        [kernel.ops.mlp_swiglu]
        fallback = ["aten.bf16", "reference.f32"]
        "#,
    )
    .unwrap();

    let env = TargetEnv {
        world_size: 1,
        ..cpu_env()
    };
    let error = recipe
        .resolve(&registry, "mlp_swiglu", Phase::Forward, &[], &env)
        .unwrap_err();
    match &error {
        ResolveError::StrictDegraded { op, skipped } => {
            assert_eq!(op, "mlp_swiglu");
            assert_eq!(skipped.len(), 1);
            assert_eq!(skipped[0].0, "aten.bf16");
        }
        other => panic!("unexpected error: {other:?}"),
    }
    let text = error.to_string();
    assert!(text.contains("strict"), "{text}");
    assert!(text.contains("aten.bf16"), "{text}");

    // With strict off the same recipe degrades — and says so.
    let lenient = Recipe::from_toml(
        r#"
        [kernel.ops.mlp_swiglu]
        fallback = ["aten.bf16", "reference.f32"]
        "#,
    )
    .unwrap();
    let resolved = lenient
        .resolve(&registry, "mlp_swiglu", Phase::Forward, &[], &env)
        .expect("fallback must resolve");
    assert_eq!(resolved.variant(), "reference.f32");
    assert!(resolved.degraded());
}

#[test]
fn recipe_resolution_uses_the_op_variant_and_the_default_provider() {
    let cuda = [Spec::new("rmsnorm", "cuda.fused_bf16")
        .dtypes(&[RsDtype::BF16])
        .min_sm(90)
        .build()];
    let aten = [Spec::new("rmsnorm", "aten.bf16")
        .dtypes(&[RsDtype::BF16])
        .build()];
    let reg = registry(&[("cuda_kernels", &cuda), ("aten", &aten)]);
    let env = cuda_env(89, 1);

    // `forward` is `prefer`: published but unusable here, so the run fails
    // instead of silently using `aten.bf16`.
    let strict_forward = Recipe::from_toml(
        r#"
        [kernel]
        default = "kernels"

        [kernel.ops.rmsnorm]
        forward = "cuda.fused_bf16"
        "#,
    )
    .unwrap();
    let error = strict_forward
        .resolve(&reg, "rmsnorm", Phase::Forward, &[RsDtype::BF16], &env)
        .unwrap_err();
    assert!(matches!(error, ResolveError::PreferredRejected { .. }));

    // No op-level variant: the default provider decides. It publishes exactly
    // one variant of `rmsnorm`, and that variant cannot run here, so the run
    // fails with the candidate table rather than quietly picking `aten`.
    let no_variant = Recipe::from_toml("[kernel]\ndefault = \"cuda_kernels\"\n").unwrap();
    let error = no_variant
        .resolve(&reg, "rmsnorm", Phase::Forward, &[RsDtype::BF16], &env)
        .unwrap_err();
    let text = error.to_string();
    assert!(matches!(error, ResolveError::Ambiguous(_)));
    assert!(
        text.contains(
            "provider `cuda_kernels` publishes 1 variant of `rmsnorm`, none of which can run here"
        ),
        "{text}"
    );
    // The variant that *would* have worked is named, so the fix is obvious.
    assert!(text.contains("aten.bf16"), "{text}");

    // A default provider that publishes two runnable variants is ambiguous:
    // the recipe has to name one.
    let both = [
        Spec::new("matmul", "aten.f32").build(),
        Spec::new("matmul", "aten.bf16").build(),
    ];
    let two_providers = self::registry(&[("aten", &both)]);
    let ambiguous = Recipe::from_toml("[kernel]\ndefault = \"aten\"\n").unwrap();
    let error = ambiguous
        .resolve(
            &two_providers,
            "matmul",
            Phase::Forward,
            &[RsDtype::BF16],
            &cpu_env(),
        )
        .unwrap_err();
    assert!(matches!(error, ResolveError::Ambiguous(_)));

    // …and naming it in the fallback chain works, with the skip recorded.
    let explicit_fallback = Recipe::from_toml(
        r#"
        [kernel.ops.rmsnorm]
        fallback = ["cuda.fused_bf16", "aten.bf16"]
        "#,
    )
    .unwrap();
    let resolved = explicit_fallback
        .resolve(&reg, "rmsnorm", Phase::Forward, &[RsDtype::BF16], &env)
        .expect("aten.bf16 is an explicitly listed fallback");
    assert_eq!(resolved.variant(), "aten.bf16");
    assert_eq!(resolved.rejected.len(), 1);
    assert_eq!(resolved.rejected[0].0, "cuda.fused_bf16");
}

#[test]
fn registry_handles_outlive_the_registry() {
    // The plan compiler collects handles and drops the registry; a handle that
    // borrowed from it would not survive that.
    let handles: Vec<RegisteredOp> = {
        let descs = [Spec::new("rmsnorm", "reference.f32").build()];
        let registry = registry(&[("reference", &descs)]);
        let req = request("rmsnorm", Phase::Forward, &[RsDtype::F32]);
        let resolved = registry
            .resolve_with_default(&req, Some("reference"))
            .unwrap();
        vec![resolved.op.clone()]
    };
    assert_eq!(handles[0].spec_name(), "rmsnorm@reference.f32");
    assert_eq!(handles[0].plugin_identity(), "reference@0.1.0");
    assert_eq!(handles[0].origin(), PathBuf::from("/plugins/reference.so"));
    assert_eq!(handles[0].doc(), "rmsnorm implementation reference.f32");
    assert_eq!(handles[0].version(), 1);
    assert_eq!(handles[0].plugin_name(), "reference");
    assert_eq!(handles[0].plugin_version(), "0.1.0");
    assert!(handles[0].collectives().is_empty());
    assert!(handles[0].expansion().is_none());
    assert_eq!(handles[0], handles[0].clone());
}

#[test]
fn composite_without_an_expansion_is_visible_in_the_summary() {
    // Contract R-4 says a fused/model-block operator must publish an
    // expansion; `describe()` is where a missing one has to be noticeable, so
    // the two flags must be independent.
    let descs = [Spec::new("mlp_swiglu", "cuda.fp8_block128").build()];
    let registry = registry(&[("cuda_kernels", &descs)]);
    let summary = &registry.describe()[0];
    assert!(summary.is_composite);
    assert!(!summary.has_expansion);
}

#[test]
fn recipe_loads_from_a_file_and_reports_a_missing_one() {
    let path =
        std::env::temp_dir().join(format!("rustrain-ops-recipe-{}.toml", std::process::id()));
    std::fs::write(&path, SPEC_RECIPE).expect("write temp recipe");

    let recipe = Recipe::load(&path).expect("load from file");
    assert_eq!(recipe.default, "aten");
    assert_eq!(
        recipe.variant_for("rmsnorm", Phase::Forward),
        Some("cuda.fused_bf16")
    );
    std::fs::remove_file(&path).ok();

    let missing = path.with_extension("does-not-exist");
    let error = Recipe::load(&missing).unwrap_err().to_string();
    assert!(error.contains("cannot read recipe"), "{error}");
    assert!(error.contains("does-not-exist"), "{error}");
}

#[test]
fn an_empty_document_is_an_all_default_recipe() {
    // Every field is `#[serde(default)]`, so a recipe file may set only what it
    // changes — and "nothing set" must mean the ABI zero state, not an error.
    let recipe = Recipe::from_toml("").expect("an empty recipe is valid");
    assert_eq!(recipe, Recipe::default());
    assert_eq!(recipe.default_provider(), None);

    let partial = Recipe::from_toml("[kernel]\ndefault = \"aten\"\n").unwrap();
    assert_eq!(partial.precision, PrecisionRecipe::default());
    assert_eq!(partial.parallel, ParallelRecipe::default());
    assert!(partial.ops.is_empty());
    assert!(!partial.strict);
}

// ── backward wiring, reserved spellings, and review follow-ups ──────────────

#[test]
fn backward_of_follows_the_declared_pair() {
    let descs = [
        Spec::new("rmsnorm", "cuda.fused_bf16")
            .backward(RsBackwardKind::EXPLICIT)
            .build(),
        Spec::new("rmsnorm_backward", "cuda.fused_bf16").build(),
    ];
    let registry = registry(&[("kernels", &descs)]);
    let env = cuda_env(90, 1);

    let forward = registry.candidates("rmsnorm")[0].clone();
    let backward = registry
        .backward_of(&forward, &[RsDtype::BF16], &env)
        .expect("the declared backward variant is registered and runnable");
    assert_eq!(backward.name(), "rmsnorm_backward");
    assert_eq!(backward.variant(), "cuda.fused_bf16");
}

#[test]
fn backward_of_reports_a_missing_or_unrunnable_backward() {
    // (a) the declared backward operator is not registered at all.
    let forward_only = [Spec::new("rmsnorm", "cuda.fused_bf16")
        .backward(RsBackwardKind::EXPLICIT)
        .build()];
    let reg = registry(&[("kernels", &forward_only)]);
    let forward = reg.candidates("rmsnorm")[0].clone();
    let error = reg
        .backward_of(&forward, &[RsDtype::BF16], &cuda_env(90, 1))
        .unwrap_err();
    assert!(matches!(error, ResolveError::BackwardOpMissing { .. }));
    let text = error.to_string();
    assert!(text.contains("rmsnorm_backward@cuda.fused_bf16"), "{text}");
    assert!(text.contains("not registered"), "{text}");

    // (b) registered, but it cannot run on this host.
    let descs = [
        Spec::new("rmsnorm", "cuda.fused_bf16")
            .backward(RsBackwardKind::EXPLICIT)
            .build(),
        Spec::new("rmsnorm_backward", "cuda.fused_bf16")
            .min_sm(90)
            .build(),
    ];
    let reg = registry(&[("kernels", &descs)]);
    let forward = reg.candidates("rmsnorm")[0].clone();
    let error = reg
        .backward_of(&forward, &[RsDtype::BF16], &cuda_env(80, 1))
        .unwrap_err();
    assert!(matches!(error, ResolveError::BackwardOpRejected { .. }));
    assert!(error.to_string().contains("sm 80 < required 90"));

    // (c) a variant that declares no backward at all.
    let autodiff = [Spec::new("rmsnorm", "reference.f32").build()];
    let reg = registry(&[("reference", &autodiff)]);
    let forward = reg.candidates("rmsnorm")[0].clone();
    let error = reg
        .backward_of(&forward, &[RsDtype::F32], &cpu_env())
        .unwrap_err();
    assert!(matches!(error, ResolveError::BackwardNotDeclared { .. }));
    assert!(error.to_string().contains("backward = AUTODIFF"));
}

#[test]
fn autodiff_is_a_strategy_not_a_variant() {
    // Spec §2.7 writes `backward = "autodiff"` for `mlp_swiglu`. Nothing
    // publishes a variant by that name, so it must never become `prefer`.
    let recipe = Recipe::from_toml(
        r#"
        [kernel.ops.mlp_swiglu]
        forward  = "cuda.fp8_block128"
        backward = "autodiff"
        "#,
    )
    .unwrap();

    assert_eq!(recipe.variant_for("mlp_swiglu", Phase::Backward), None);
    assert_eq!(
        recipe.backward_plan("mlp_swiglu"),
        Some(BackwardPlan::Autodiff)
    );
    assert_eq!(
        recipe
            .resolve_request("mlp_swiglu", Phase::Backward, &[], &cpu_env())
            .prefer,
        None
    );

    // A real variant name still goes through the normal `prefer` path.
    let explicit = Recipe::from_toml(
        r#"
        [kernel.ops.rmsnorm]
        backward = "cuda.fused_bf16"
        "#,
    )
    .unwrap();
    assert_eq!(
        explicit.variant_for("rmsnorm", Phase::Backward),
        Some("cuda.fused_bf16")
    );
    assert_eq!(
        explicit.backward_plan("rmsnorm"),
        Some(BackwardPlan::Variant("cuda.fused_bf16".to_string()))
    );
}

#[test]
fn empty_or_padded_variant_names_are_hard_errors() {
    for (toml, needle) in [
        (
            "[kernel.ops.rmsnorm]\nforward = \"\"\n",
            "kernel.ops.rmsnorm.forward",
        ),
        (
            "[kernel.ops.rmsnorm]\nforward = \" reference.f32\"\n",
            "not a usable variant name",
        ),
        (
            "[kernel.ops.rmsnorm]\nfallback = [\"aten.bf16\", \" \"]\n",
            "kernel.ops.rmsnorm.fallback[1]",
        ),
    ] {
        let error = Recipe::from_toml(toml).unwrap_err().to_string();
        assert!(error.contains(needle), "expected {needle:?} in:\n{error}");
    }

    // The reserved strategy spelling is not a variant, so it is exempt.
    let reserved = Recipe::from_toml("[kernel.ops.mlp_swiglu]\nbackward = \"autodiff\"\n")
        .expect("`autodiff` is a strategy, not a padded variant name");
    assert_eq!(
        reserved.backward_plan("mlp_swiglu"),
        Some(BackwardPlan::Autodiff)
    );
}

#[test]
fn a_parallelism_degree_of_zero_is_rejected() {
    let error = Recipe::from_toml("[kernel.parallel]\ntensor = 0\n")
        .unwrap_err()
        .to_string();
    assert!(error.contains("kernel.parallel.tensor = 0"), "{error}");

    // Omitted degrees are the identity, so the world size is always defined.
    let partial = Recipe::from_toml("[kernel.parallel]\ntensor = 8\n").unwrap();
    assert_eq!(partial.parallel.world_size(), 8);
    assert_eq!(partial.parallel.data, 1);
}

#[test]
fn device_namespaces_are_exact_words() {
    // `cpu_offload.f32` is not a CPU-only kernel; splitting on `_` would have
    // rejected it on a CUDA host for no reason.
    assert_eq!(declared_device("cpu_offload.f32"), None);
    assert_eq!(declared_device("cuda_sm90.f32"), None);
    assert_eq!(declared_device("cpu.f32"), Some(RsDeviceKind::CPU));
}

#[test]
fn an_out_of_range_min_world_size_does_not_flip_the_dtype_reading() {
    // `min_world_size = -1` is not a declaration (the header says "0 = no
    // constraint"); it must not turn an all-zero `requires` into "accepts no
    // dtype", which would blame a field the author never touched.
    let descs = [Spec::new("linear", "reference.f32")
        .zero_requires()
        .min_world_size(-1)
        .build()];
    let registry = registry(&[("reference", &descs)]);
    let op = registry.candidates("linear")[0];
    assert_eq!(reject_reason(op, &[RsDtype::FP4E2M1], &cpu_env()), None);
}

#[test]
fn one_plugin_publishing_the_same_variant_twice_is_reported_as_such() {
    let descs = [
        Spec::new("rmsnorm", "reference.f32").build(),
        Spec::new("rmsnorm", "reference.f32").build(),
    ];
    let mut registry = Registry::new();
    let error = registry
        .add_detached("reference", "0.1.0", "/plugins/reference.so", &descs)
        .unwrap_err();
    assert!(matches!(error, RegistryError::DuplicateWithinPlugin { .. }));
    let text = error.to_string();
    assert!(text.contains("more than once"), "{text}");
    assert!(text.contains("reference@0.1.0"), "{text}");
}

#[test]
fn loaded_plugins_are_reported_even_without_operators() {
    let mut registry = Registry::new();
    assert_eq!(
        registry
            .add_detached("empty_plugin", "1.0.0", "/plugins/empty.so", &[])
            .unwrap(),
        0
    );
    assert_eq!(registry.plugin_names(), ["empty_plugin@1.0.0"]);
    assert!(registry.is_empty());

    // A misspelled operator name says which plugins *were* loaded instead of
    // claiming that nothing was.
    let error = registry
        .resolve(&request("rmsnorm", Phase::Forward, &[]))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("1 plugin loaded (empty_plugin@1.0.0) but none published an operator"),
        "{error}"
    );
}

#[test]
fn a_named_but_unloaded_default_provider_says_so() {
    let descs = [Spec::new("rmsnorm", "reference.f32").build()];
    let registry = registry(&[("reference", &descs)]);

    let error = registry
        .resolve_with_default(
            &request("rmsnorm", Phase::Forward, &[RsDtype::F32]),
            Some("aten"),
        )
        .unwrap_err();
    let text = error.to_string();
    assert!(text.contains("no loaded plugin is named `aten`"), "{text}");
    assert!(text.contains("reference@0.1.0"), "{text}");
}

#[test]
fn a_provider_with_several_variants_wins_when_only_one_can_run() {
    // The default-provider rule is about *runnable* variants: a provider that
    // publishes two, one of which this host cannot run, is still unambiguous.
    let descs = [
        Spec::new("rmsnorm", "aten.f32").build(),
        Spec::new("rmsnorm", "aten.sm90").min_sm(90).build(),
    ];
    let registry = registry(&[("aten", &descs)]);

    let resolved = registry
        .resolve_with_default(
            &request("rmsnorm", Phase::Forward, &[RsDtype::F32]),
            Some("aten"),
        )
        .expect("aten.f32 is the only runnable variant");
    assert_eq!(resolved.variant(), "aten.f32");
    assert!(
        !resolved.degraded(),
        "the policy picked the only runnable variant; nothing was skipped in a chain"
    );
}

#[test]
fn prefer_not_published_still_carries_the_candidate_table() {
    // R-1 asks for every candidate and its reason even when the named variant
    // simply does not exist — that is exactly when the table is most useful.
    let descs = [
        Spec::new("rmsnorm", "a.f32")
            .dtypes(&[RsDtype::F32])
            .build(),
        Spec::new("rmsnorm", "c.sm90")
            .dtypes(&[RsDtype::BF16])
            .min_sm(90)
            .build(),
    ];
    let registry = registry(&[("kernels", &descs)]);

    let mut req = request("rmsnorm", Phase::Forward, &[RsDtype::BF16]);
    req.prefer = Some("typo.bf16".to_string());
    req.env = cuda_env(80, 1);

    let error = registry.resolve(&req).unwrap_err();
    assert!(matches!(error, ResolveError::PreferredNotPublished { .. }));
    let failure = error.failure().expect("candidate table");
    assert_eq!(failure.phase, Phase::Forward);
    assert_eq!(failure.candidates.len(), 2);
    assert_eq!(failure.candidates[0].variant, "a.f32");
    assert_eq!(failure.candidates[0].plugin_origin, "/plugins/kernels.so");
    assert_eq!(failure.unpublished, ["typo.bf16"]);

    let text = error.to_string();
    // Every candidate with its own reason, plus the request context.
    assert!(
        text.contains("dtype bf16 is not accepted (variant accepts: f32)"),
        "{text}"
    );
    assert!(text.contains("sm 80 < required 90"), "{text}");
    assert!(text.contains("phase forward"), "{text}");
    assert!(
        text.contains("named but not published: rmsnorm@typo.bf16"),
        "{text}"
    );
}
