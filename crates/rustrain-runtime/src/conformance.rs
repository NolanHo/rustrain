//! Conformance checking: is every implementation of an operator the same
//! operator?
//!
//! The gate that makes "swap a kernel from a configuration file" safe. For each
//! `(operator, variant)` it runs the variant and the reference implementation on
//! identical inputs and compares the results, then runs the variant twice to
//! check it is reproducible.
//!
//! Every check returns [`Check::Skipped`] with a reason rather than passing
//! quietly when it could not be performed. A check that did not run must not
//! read as a check that succeeded — that is the exact failure mode this whole
//! architecture exists to remove, and it would be ironic to reintroduce it in
//! the gate itself.

use rustrain_abi::ffi::{RsAttrs, RsDtype, RsTensor};
use rustrain_ops::{Phase, Recipe, RegisteredOp, Registry, TargetEnv};
use rustrain_parallel::{Mesh, ParallelConfig};
use rustrain_plan::attrs::AbiAttrs;
use rustrain_plan::{Attrs, Compiler, OpRef, Plan, PlanBuilder, SlotId, SlotKind};

use crate::{Executor, HostAllocator, SingleRank};

/// The provider every other provider is measured against.
pub const REFERENCE_VARIANT: &str = "reference.f32";

/// A numeric comparison budget.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    /// Relative error allowed when `|expected|` is large.
    pub rel: f64,
    /// Absolute error allowed near zero, where relative error is meaningless.
    pub abs: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        // Loose enough for a different summation order in f32, tight enough that
        // a wrong formula or a transposed operand cannot slip through.
        Self {
            rel: 1e-4,
            abs: 1e-5,
        }
    }
}

/// The outcome of one check.
#[derive(Clone, Debug, PartialEq)]
pub enum Check {
    Pass {
        detail: String,
    },
    Fail {
        detail: String,
    },
    /// Not performed. The reason is mandatory and is printed.
    Skipped {
        reason: String,
    },
}

impl Check {
    fn pass(detail: impl Into<String>) -> Self {
        Self::Pass {
            detail: detail.into(),
        }
    }

    fn fail(detail: impl Into<String>) -> Self {
        Self::Fail {
            detail: detail.into(),
        }
    }

    fn skipped(reason: impl Into<String>) -> Self {
        Self::Skipped {
            reason: reason.into(),
        }
    }

    pub fn is_fail(&self) -> bool {
        matches!(self, Check::Fail { .. })
    }

    pub fn is_skipped(&self) -> bool {
        matches!(self, Check::Skipped { .. })
    }

    pub fn label(&self) -> &'static str {
        match self {
            Check::Pass { .. } => "pass",
            Check::Fail { .. } => "FAIL",
            Check::Skipped { .. } => "skip",
        }
    }

    pub fn detail(&self) -> &str {
        match self {
            Check::Pass { detail } | Check::Fail { detail } => detail,
            Check::Skipped { reason } => reason,
        }
    }
}

/// Everything checked for one `(operator, variant)`.
#[derive(Clone, Debug)]
pub struct CaseResult {
    pub op: String,
    pub variant: String,
    pub numeric: Check,
    pub expansion: Check,
    pub gradient: Check,
    pub determinism: Check,
}

impl CaseResult {
    /// A case is acceptable when nothing failed. Skips are visible, not fatal.
    pub fn ok(&self) -> bool {
        !self.numeric.is_fail()
            && !self.expansion.is_fail()
            && !self.gradient.is_fail()
            && !self.determinism.is_fail()
    }

    pub fn skips(&self) -> usize {
        [
            &self.numeric,
            &self.expansion,
            &self.gradient,
            &self.determinism,
        ]
        .iter()
        .filter(|c| c.is_skipped())
        .count()
    }
}

/// The whole run.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub results: Vec<CaseResult>,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.results.iter().all(CaseResult::ok)
    }

    pub fn failures(&self) -> usize {
        self.results.iter().filter(|r| !r.ok()).count()
    }

    pub fn skips(&self) -> usize {
        self.results.iter().map(CaseResult::skips).sum()
    }

    /// One line per case, then the failures in full.
    pub fn explain(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:<34} {:>5} {:>5} {:>5} {:>5}\n",
            "OP@VARIANT", "NUM", "EXP", "GRAD", "DET"
        ));
        for r in &self.results {
            out.push_str(&format!(
                "{:<34} {:>5} {:>5} {:>5} {:>5}\n",
                format!("{}@{}", r.op, r.variant),
                r.numeric.label(),
                r.expansion.label(),
                r.gradient.label(),
                r.determinism.label(),
            ));
        }

        let detail: Vec<&CaseResult> = self.results.iter().filter(|r| r.skips() > 0).collect();
        if !detail.is_empty() {
            out.push_str("\nskipped, and why:\n");
            for r in &detail {
                for (name, c) in [
                    ("numeric", &r.numeric),
                    ("expansion", &r.expansion),
                    ("gradient", &r.gradient),
                    ("determinism", &r.determinism),
                ] {
                    if let Check::Skipped { reason } = c {
                        out.push_str(&format!("  {}@{} {name}: {reason}\n", r.op, r.variant));
                    }
                }
            }
        }

        let failed: Vec<&CaseResult> = self.results.iter().filter(|r| !r.ok()).collect();
        if !failed.is_empty() {
            out.push_str("\nfailures:\n");
            for r in &failed {
                for (name, c) in [
                    ("numeric", &r.numeric),
                    ("expansion", &r.expansion),
                    ("gradient", &r.gradient),
                    ("determinism", &r.determinism),
                ] {
                    if c.is_fail() {
                        out.push_str(&format!(
                            "  {}@{} {name}: {}\n",
                            r.op,
                            r.variant,
                            c.detail()
                        ));
                    }
                }
            }
        }

        out.push_str(&format!(
            "\n{} case(s): {} failing, {} skipped check(s)\n",
            self.results.len(),
            self.failures(),
            self.skips()
        ));
        out
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "passed": self.passed(),
            "cases": self.results.iter().map(|r| serde_json::json!({
                "op": r.op,
                "variant": r.variant,
                "numeric": check_json(&r.numeric),
                "expansion": check_json(&r.expansion),
                "gradient": check_json(&r.gradient),
                "determinism": check_json(&r.determinism),
            })).collect::<Vec<_>>(),
            "failures": self.failures(),
            "skips": self.skips(),
        })
    }
}

fn check_json(c: &Check) -> serde_json::Value {
    serde_json::json!({
        "status": c.label(),
        "detail": c.detail(),
    })
}

/// How to fill an input buffer.
#[derive(Clone, Copy, Debug)]
pub enum Fill {
    Zeros,
    Ones,
    /// `(i % 17) * 0.25 - 2.0`: deterministic, signed, fractional, so a sign
    /// error or a missing scale shows up in the comparison.
    Ramp,
    /// A small LCG. Reproducible from the seed without a dependency.
    Pseudo {
        seed: u64,
    },
    /// Indices in `[0, modulo)`, for i32/i64 inputs.
    Indices {
        modulo: i64,
    },
}

impl Fill {
    fn f32_at(&self, i: usize) -> f32 {
        match self {
            Fill::Zeros => 0.0,
            Fill::Ones => 1.0,
            Fill::Ramp => (i % 17) as f32 * 0.25 - 2.0,
            Fill::Indices { .. } => 0.0,
            Fill::Pseudo { seed } => {
                // xorshift64*: tiny, deterministic, no dependency.
                let mut x = seed
                    .wrapping_add(i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15);
                x ^= x >> 30;
                x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                x ^= x >> 27;
                ((x >> 40) as f32 / 16_777_216.0) - 0.5
            }
        }
    }

    fn index_at(&self, i: usize, modulo: i64) -> i64 {
        match self {
            Fill::Indices { modulo: m } => (i as i64) % (*m).max(1),
            _ => (i as i64) % modulo.max(1),
        }
    }
}

/// One input of a case.
#[derive(Clone, Debug)]
pub struct InputSpec {
    pub name: String,
    pub dtype: RsDtype,
    pub shape: Vec<i64>,
    pub fill: Fill,
}

impl InputSpec {
    pub fn f32(name: impl Into<String>, shape: Vec<i64>, fill: Fill) -> Self {
        Self {
            name: name.into(),
            dtype: RsDtype::F32,
            shape,
            fill,
        }
    }

    pub fn indices(name: impl Into<String>, shape: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            dtype: RsDtype::I32,
            shape,
            fill: Fill::Indices { modulo: 4 },
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product::<i64>().max(0) as usize
    }

    fn bytes(&self) -> Vec<u8> {
        let n = self.numel();
        match self.dtype {
            d if d == RsDtype::F32 => {
                let mut v = Vec::with_capacity(n * 4);
                for i in 0..n {
                    v.extend_from_slice(&self.fill.f32_at(i).to_ne_bytes());
                }
                v
            }
            d if d == RsDtype::I32 => {
                let mut v = Vec::with_capacity(n * 4);
                for i in 0..n {
                    v.extend_from_slice(&(self.fill.index_at(i, 4) as i32).to_ne_bytes());
                }
                v
            }
            d if d == RsDtype::I64 => {
                let mut v = Vec::with_capacity(n * 8);
                for i in 0..n {
                    v.extend_from_slice(&self.fill.index_at(i, 4).to_ne_bytes());
                }
                v
            }
            _ => Vec::new(),
        }
    }
}

/// One operator invocation to check.
#[derive(Clone, Debug)]
pub struct Case {
    pub op: String,
    pub inputs: Vec<InputSpec>,
    /// How many results the operator produces. The ABI does not declare arity,
    /// so the case states it.
    pub n_outputs: usize,
    pub attrs: Attrs,
}

impl Case {
    pub fn new(op: impl Into<String>, inputs: Vec<InputSpec>) -> Self {
        Self {
            op: op.into(),
            inputs,
            n_outputs: 1,
            attrs: Attrs::new(),
        }
    }

    pub fn outputs(mut self, n: usize) -> Self {
        self.n_outputs = n;
        self
    }

    pub fn attrs(mut self, attrs: Attrs) -> Self {
        self.attrs = attrs;
        self
    }

    pub fn label(&self) -> String {
        format!("{}({})", self.op, self.inputs.len())
    }
}

/// One operator's output, as raw bytes plus its shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub name: String,
    pub dtype: RsDtype,
    pub shape: Vec<i64>,
    pub bytes: Vec<u8>,
}

impl Output {
    pub fn as_f32(&self) -> Option<Vec<f32>> {
        if self.dtype != RsDtype::F32 {
            return None;
        }
        Some(
            self.bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }
}

/// Runs the checks.
pub struct Harness<'a> {
    registry: &'a Registry,
    recipe: &'a Recipe,
    tolerance: Tolerance,
}

impl<'a> Harness<'a> {
    pub fn new(registry: &'a Registry, recipe: &'a Recipe) -> Self {
        Self {
            registry,
            recipe,
            tolerance: Tolerance::default(),
        }
    }

    pub fn tolerance(mut self, t: Tolerance) -> Self {
        self.tolerance = t;
        self
    }

    /// Check every variant published for the case's operator.
    pub fn run(&self, case: &Case) -> Vec<CaseResult> {
        let mut variants: Vec<String> = self
            .registry
            .candidates(&case.op)
            .into_iter()
            .map(|c| c.variant().to_string())
            .collect();
        variants.sort();
        variants.dedup();

        if variants.is_empty() {
            return vec![CaseResult {
                op: case.op.clone(),
                variant: "<none>".to_string(),
                numeric: Check::fail(format!("no implementation of `{}` is registered", case.op)),
                expansion: Check::skipped("no implementation to check"),
                gradient: Check::skipped("no implementation to check"),
                determinism: Check::skipped("no implementation to check"),
            }];
        }

        variants
            .iter()
            .map(|v| self.check_variant(case, v))
            .collect()
    }

    /// Check one variant: numeric, expansion, gradient, determinism.
    pub fn check_variant(&self, case: &Case, variant: &str) -> CaseResult {
        let mut result = CaseResult {
            op: case.op.clone(),
            variant: variant.to_string(),
            numeric: Check::skipped("not attempted"),
            expansion: Check::skipped("not attempted"),
            gradient: Check::skipped(
                "backward derivation (spec §2.11 / D17) is not implemented, so there is no \
                 analytic gradient to compare against a finite difference",
            ),
            determinism: Check::skipped("not attempted"),
        };

        let first = match self.execute(case, variant) {
            Ok(out) => out,
            Err(e) => {
                result.numeric = Check::fail(format!("the implementation did not run: {e}"));
                result.determinism = Check::skipped("the implementation did not run");
                result.expansion = Check::skipped("the implementation did not run");
                return result;
            }
        };

        // Numeric: against the reference, unless this *is* the reference.
        if variant == REFERENCE_VARIANT {
            result.numeric = Check::skipped(format!(
                "this is the reference implementation ({REFERENCE_VARIANT}); a comparison needs \
                 a second implementation, which is what a provider plugin supplies"
            ));
        } else {
            match self.execute(case, REFERENCE_VARIANT) {
                Ok(expected) => {
                    result.numeric = compare(&first, &expected, self.tolerance);
                }
                Err(e) => {
                    result.numeric = Check::skipped(format!(
                        "the reference implementation could not run on this case: {e}"
                    ));
                }
            }
        }

        // Determinism: the same inputs twice, bit for bit.
        match self.execute(case, variant) {
            Ok(second) => {
                result.determinism = if first == second {
                    Check::pass("two runs produced identical bytes")
                } else {
                    Check::fail(format!(
                        "two runs on identical inputs produced different bytes:\n    first:  {}\n    \
                         second: {}",
                        summarise(&first),
                        summarise(&second)
                    ))
                };
            }
            Err(e) => {
                result.determinism = Check::skipped(format!("the second run failed: {e}"));
            }
        }

        // Expansion: replay the declared composition and compare.
        result.expansion = self.check_expansion(case, variant, &first);

        result
    }

    fn check_expansion(&self, case: &Case, variant: &str, fused: &[Output]) -> Check {
        let Some(op) = self.lookup(&case.op, variant) else {
            return Check::skipped("the implementation is not registered");
        };
        let Some(expansion) = op.expansion() else {
            return Check::skipped(
                "the implementation declares no expansion; per contract R-4 a composite operator \
                 must declare one for this check to exist",
            );
        };
        // An expansion whose nodes carry no attributes cannot be replayed: a
        // `reduce` without its `kind`, or a `matmul` without `transpose_b`, is a
        // different computation. Say so rather than running something else.
        // SAFETY: the descriptors are process-lifetime data owned by the plugin.
        let nodes = unsafe { expansion.as_slice() };
        if nodes.iter().any(|n| n.attrs.is_null()) {
            return Check::skipped(
                "at least one expansion node carries no attributes; the provider must declare \
                 them with ExpansionSpec::node_with_attrs, otherwise a replay would run a \
                 different computation (spec §7.5)",
            );
        }
        match self.replay(case, nodes) {
            Ok(replayed) => compare(fused, &replayed, self.tolerance),
            Err(e) => Check::skipped(format!("the declared expansion could not be replayed: {e}")),
        }
    }

    /// Runs the declared expansion node by node and returns the parent outputs.
    fn replay(
        &self,
        case: &Case,
        nodes: &[rustrain_abi::ffi::RsExpansionNode],
    ) -> Result<Vec<Output>, String> {
        let n_parent_in = case.inputs.len();
        let n_parent_out = case.n_outputs;

        // Local tensor table: parent inputs, then parent outputs, then
        // temporaries discovered by inference.
        let mut shapes: Vec<Option<(RsDtype, Vec<i64>)>> = Vec::new();
        for spec in &case.inputs {
            shapes.push(Some((spec.dtype, spec.shape.clone())));
        }
        for _ in 0..n_parent_out {
            shapes.push(None);
        }

        let mut plan_nodes: Vec<(String, Vec<i32>, Vec<i32>, Attrs)> = Vec::new();

        for node in nodes {
            // SAFETY: the descriptor arrays are owned by the plugin.
            let (name, ins, outs, attrs) = unsafe {
                let name = std::ffi::CStr::from_ptr(node.op)
                    .to_str()
                    .map_err(|e| e.to_string())?
                    .to_string();
                let ins: Vec<i32> = node.input_ids().to_vec();
                let outs: Vec<i32> = node.output_ids().to_vec();
                (name, ins, outs, read_attrs(node.attrs)?)
            };

            // Ask the primitive for the outputs' shapes.
            let op = self
                .lookup(&name, REFERENCE_VARIANT)
                .ok_or_else(|| format!("`{name}` has no {REFERENCE_VARIANT} implementation"))?;
            let inferred = infer_for(&op, &shapes, &ins, outs.len(), &attrs)?;
            for (id, shape) in outs.iter().zip(inferred) {
                let slot = *id as usize;
                if shapes.len() <= slot {
                    shapes.resize(slot + 1, None);
                }
                shapes[slot] = Some(shape);
            }
            plan_nodes.push((name, ins, outs, attrs));
        }

        // Build and run a plan over the expansion.
        let mut b = PlanBuilder::new(
            "expansion",
            Phase::Forward,
            Mesh::from_config(&ParallelConfig::default()).fingerprint(),
        );
        let mut slots: Vec<SlotId> = Vec::new();
        for (i, shape) in shapes.iter().enumerate() {
            let (dtype, dims) = shape.clone().ok_or_else(|| {
                format!("local tensor {i} never got a shape; the expansion is incomplete")
            })?;
            let kind = if i < n_parent_in {
                SlotKind::Input
            } else if i < n_parent_in + n_parent_out {
                SlotKind::Output
            } else {
                SlotKind::Temp
            };
            slots.push(b.slot(format!("t{i}"), dtype, dims, kind));
        }

        for (name, ins, outs, attrs) in plan_nodes {
            let inputs: Vec<SlotId> = ins.iter().map(|i| slots[*i as usize]).collect();
            let outputs: Vec<SlotId> = outs.iter().map(|i| slots[*i as usize]).collect();
            b.node(
                OpRef::new(&name),
                inputs,
                outputs,
                attrs,
                format!("expansion.{name}"),
            );
        }

        let plan = b.build().map_err(|e| e.to_string())?;
        let mut ex = self.make_executor(&plan)?;
        for (i, spec) in case.inputs.iter().enumerate() {
            ex.write_raw(slots[i], &spec.bytes())
                .map_err(|e| e.to_string())?;
        }
        ex.run().map_err(|e| e.to_string())?;

        (0..n_parent_out)
            .map(|k| {
                let slot = slots[n_parent_in + k];
                let meta = ex.descriptor(slot).map_err(|e| e.to_string())?;
                Ok(Output {
                    name: format!("out{k}"),
                    dtype: meta.dtype,
                    shape: meta.dims().to_vec(),
                    bytes: ex.read_raw(slot).map_err(|e| e.to_string())?,
                })
            })
            .collect()
    }

    /// Runs one variant on the case's inputs and returns its outputs.
    fn execute(&self, case: &Case, variant: &str) -> Result<Vec<Output>, String> {
        let op = self
            .lookup(&case.op, variant)
            .ok_or_else(|| format!("`{}@{variant}` is not registered", case.op))?;

        // The ABI does not declare arity, so the case provides the count.
        let attrs = case.attrs.to_abi();
        let in_tensors: Vec<RsTensor> = case
            .inputs
            .iter()
            .map(|s| RsTensor::new(s.dtype, &s.shape))
            .collect();
        let in_ptrs: Vec<*const RsTensor> = in_tensors.iter().map(std::ptr::from_ref).collect();
        let mut out_tensors: Vec<RsTensor> = (0..case.n_outputs)
            .map(|_| RsTensor::new(RsDtype::F32, &[]))
            .collect();
        let mut out_ptrs: Vec<*mut RsTensor> =
            out_tensors.iter_mut().map(std::ptr::from_mut).collect();

        let Some(infer) = op.desc().infer else {
            return Err("the implementation has no shape inference".to_string());
        };
        // SAFETY: descriptors live in this function; `infer` only reads inputs
        // and writes the outputs' shapes.
        let status = unsafe {
            infer(
                in_ptrs.as_ptr(),
                in_ptrs.len() as u32,
                out_ptrs.as_mut_ptr(),
                case.n_outputs as u32,
                attrs.as_ptr(),
            )
        };
        if status != 0 {
            return Err(format!(
                "shape inference returned {status}: {}",
                op_last_error(&op)
            ));
        }

        // Now a real plan with the inferred shapes.
        let mut b = PlanBuilder::new(
            "case",
            Phase::Forward,
            Mesh::from_config(&ParallelConfig::default()).fingerprint(),
        );
        let mut in_slots = Vec::new();
        for s in &case.inputs {
            in_slots.push(b.slot(s.name.clone(), s.dtype, s.shape.clone(), SlotKind::Input));
        }
        let out_slots: Vec<SlotId> = out_tensors
            .iter()
            .enumerate()
            .map(|(i, t)| {
                b.slot(
                    format!("out{i}"),
                    t.dtype,
                    t.dims().to_vec(),
                    SlotKind::Output,
                )
            })
            .collect();
        b.node(
            OpRef::variant(&case.op, variant),
            in_slots.clone(),
            out_slots.clone(),
            case.attrs.clone(),
            "case",
        );

        let plan = b.build().map_err(|e| e.to_string())?;
        let mut ex = self.make_executor(&plan)?;
        for (slot, spec) in in_slots.iter().zip(&case.inputs) {
            ex.write_raw(*slot, &spec.bytes())
                .map_err(|e| e.to_string())?;
        }
        ex.run().map_err(|e| e.to_string())?;

        out_slots
            .iter()
            .enumerate()
            .map(|(i, slot)| {
                let meta = ex.descriptor(*slot).map_err(|e| e.to_string())?;
                Ok(Output {
                    name: format!("out{i}"),
                    dtype: meta.dtype,
                    shape: meta.dims().to_vec(),
                    bytes: ex.read_raw(*slot).map_err(|e| e.to_string())?,
                })
            })
            .collect()
    }

    fn make_executor(&self, plan: &Plan) -> Result<Executor, String> {
        // Single-process, so no memory strategy beyond `keep` is available; the
        // compiler is told that rather than left to assume one.
        let compiled = Compiler::new(self.registry, self.recipe, TargetEnv::default())
            .capabilities(rustrain_plan::RuntimeCapabilities::default())
            .compile(plan)
            .map_err(|e| e.to_string())?;
        Executor::new(
            compiled,
            Box::new(HostAllocator::new()),
            Box::new(SingleRank::new(1)),
        )
        .map_err(|e| e.to_string())
    }

    fn lookup(&self, op: &str, variant: &str) -> Option<RegisteredOp> {
        self.registry
            .candidates(op)
            .into_iter()
            .find(|c| c.variant() == variant)
            .cloned()
    }
}

/// Runs one primitive's inference over the local tensor table.
fn infer_for(
    op: &RegisteredOp,
    shapes: &[Option<(RsDtype, Vec<i64>)>],
    inputs: &[i32],
    n_out: usize,
    attrs: &Attrs,
) -> Result<Vec<(RsDtype, Vec<i64>)>, String> {
    let Some(infer) = op.desc().infer else {
        return Err(format!("`{}` has no shape inference", op.spec_name()));
    };
    let abi: AbiAttrs = attrs.to_abi();

    let in_tensors: Vec<RsTensor> = inputs
        .iter()
        .map(|i| {
            shapes
                .get(*i as usize)
                .and_then(Option::as_ref)
                .map(|(d, s)| RsTensor::new(*d, s))
                .ok_or_else(|| {
                    format!(
                        "`{}` reads local tensor {i}, which no earlier node produced",
                        op.spec_name()
                    )
                })
        })
        .collect::<Result<_, _>>()?;
    let in_ptrs: Vec<*const RsTensor> = in_tensors.iter().map(std::ptr::from_ref).collect();

    let mut out_tensors: Vec<RsTensor> = (0..n_out)
        .map(|_| RsTensor::new(RsDtype::F32, &[]))
        .collect();
    let mut out_ptrs: Vec<*mut RsTensor> = out_tensors.iter_mut().map(std::ptr::from_mut).collect();

    // SAFETY: as in `Harness::execute`.
    let status = unsafe {
        infer(
            in_ptrs.as_ptr(),
            in_ptrs.len() as u32,
            out_ptrs.as_mut_ptr(),
            n_out as u32,
            abi.as_ptr(),
        )
    };
    if status != 0 {
        return Err(format!(
            "`{}` shape inference returned {status}: {}",
            op.spec_name(),
            op_last_error(op)
        ));
    }
    Ok(out_tensors
        .iter()
        .map(|t| (t.dtype, t.dims().to_vec()))
        .collect())
}

/// The implementation's own explanation for its last failure.
///
/// Without this the harness reports "inference returned 1" and the operator's
/// carefully worded message — which names the offending input and its expected
/// shape — is thrown away.
fn op_last_error(op: &RegisteredOp) -> String {
    let Some(f) = op.desc().last_error else {
        return "(the implementation reports no detail)".to_string();
    };
    // SAFETY: the plugin owns this function pointer. A null context is the
    // documented way to ask for the thread-local message.
    let p = unsafe { f(std::ptr::null_mut()) };
    if p.is_null() {
        "(no message)".to_string()
    } else {
        // SAFETY: non-null and NUL-terminated, per the ABI.
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    }
}

/// Reads an `RsAttrs` back into the typed map, so an expansion node's
/// attributes can be carried into a replayed plan.
///
/// # Safety
/// `p` must be null or point to a live `RsAttrs`.
unsafe fn read_attrs(p: *const RsAttrs) -> Result<Attrs, String> {
    if p.is_null() {
        return Ok(Attrs::new());
    }
    // SAFETY: the caller guarantees liveness.
    let list = unsafe { (*p).as_slice() };
    let mut out = Attrs::new();
    for a in list {
        let Some(key) = (unsafe { rustrain_plan::attrs::read_key(a.key) }) else {
            continue;
        };
        match a.kind {
            k if k == rustrain_abi::ffi::RsAttrKind::I64 => {
                out.insert(key, a.i64);
            }
            k if k == rustrain_abi::ffi::RsAttrKind::F64 => {
                out.insert(key, a.f64);
            }
            k if k == rustrain_abi::ffi::RsAttrKind::BOOL => {
                out.insert(key, a.boolean != 0);
            }
            k if k == rustrain_abi::ffi::RsAttrKind::STR => {
                if !a.str.is_null() {
                    let s = unsafe { std::ffi::CStr::from_ptr(a.str) }
                        .to_string_lossy()
                        .into_owned();
                    out.insert(key, s);
                }
            }
            k if k == rustrain_abi::ffi::RsAttrKind::I64S && !a.i64s.is_null() => {
                let xs = unsafe { std::slice::from_raw_parts(a.i64s, a.n_i64s as usize) };
                out.insert(key, xs.to_vec());
            }
            _ => {}
        }
    }
    Ok(out)
}

/// Compares two sets of outputs within tolerance.
pub fn compare(actual: &[Output], expected: &[Output], tol: Tolerance) -> Check {
    if actual.len() != expected.len() {
        return Check::fail(format!(
            "produced {} output(s), the reference produced {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    let mut compared = 0usize;

    for (a, e) in actual.iter().zip(expected) {
        if a.shape != e.shape {
            return Check::fail(format!(
                "output {} has shape {:?}, the reference has {:?}",
                a.name, a.shape, e.shape
            ));
        }
        if a.dtype != e.dtype {
            return Check::fail(format!(
                "output {} has dtype {}, the reference has {}",
                a.name, a.dtype, e.dtype
            ));
        }

        let (Some(xs), Some(ys)) = (a.as_f32(), e.as_f32()) else {
            // Non-f32 outputs are compared byte for byte.
            if a.bytes == e.bytes {
                compared += 1;
                continue;
            }
            return Check::fail(format!(
                "output {} differs byte for byte (dtype {})",
                a.name, a.dtype
            ));
        };
        if xs.len() != ys.len() {
            return Check::fail(format!(
                "output {} has {} elements, the reference has {}",
                a.name,
                xs.len(),
                ys.len()
            ));
        }
        for (i, (x, y)) in xs.iter().zip(&ys).enumerate() {
            let (x, y) = (*x as f64, *y as f64);
            let err = (x - y).abs();
            let scale = tol.abs + tol.rel * y.abs();
            let ratio = if scale > 0.0 { err / scale } else { err };
            if ratio > worst {
                worst = ratio;
                worst_at = i;
            }
            compared += 1;
        }
    }

    if worst <= 1.0 {
        Check::pass(format!(
            "{compared} element(s) within tolerance (worst {:.3} of budget, rel {:.1e} / abs {:.1e})",
            worst, tol.rel, tol.abs
        ))
    } else {
        Check::fail(format!(
            "{compared} element(s) compared; element {worst_at} exceeds the budget by {worst:.1}x \
             (rel {:.1e} / abs {:.1e})",
            tol.rel, tol.abs
        ))
    }
}

fn summarise(outputs: &[Output]) -> String {
    outputs
        .iter()
        .map(|o| match o.as_f32() {
            Some(v) => format!(
                "[{}]",
                v.iter()
                    .take(4)
                    .map(|x| format!("{x:.6}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            None => format!("{} byte(s)", o.bytes.len()),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The cases `rustrain ops check` runs by default.
///
/// Deliberately conservative: only operators whose arity and attributes are
/// settled appear here. An operator that is missing is reported by the CLI as
/// "no case", which is honest; inventing a case with guessed arity would produce
/// a failure that looks like a kernel bug and is really a harness bug.
// A table is clearer written out as a sequence of pushes than as one enormous
// `vec![...]`, and the length is what makes it reviewable.
#[allow(clippy::vec_init_then_push)]
pub fn default_cases() -> Vec<Case> {
    use Fill::*;

    let mut cases = Vec::new();

    cases.push(
        Case::new(
            "elementwise_unary",
            vec![InputSpec::f32("x", vec![4, 8], Ramp)],
        )
        .attrs(Attrs::new().set("kind", "silu")),
    );
    cases.push(
        Case::new(
            "elementwise_binary",
            vec![
                InputSpec::f32("a", vec![4, 8], Ramp),
                InputSpec::f32("b", vec![4, 8], Pseudo { seed: 7 }),
            ],
        )
        .attrs(Attrs::new().set("kind", "add")),
    );
    cases.push(
        Case::new("reduce", vec![InputSpec::f32("x", vec![4, 8], Ramp)])
            .attrs(Attrs::new().set("kind", "sum").set("axis", -1i64)),
    );
    cases.push(
        Case::new("softmax", vec![InputSpec::f32("x", vec![4, 8], Ramp)])
            .attrs(Attrs::new().set("axis", -1i64)),
    );
    cases.push(
        Case::new(
            "rmsnorm",
            vec![
                InputSpec::f32("x", vec![4, 8], Ramp),
                InputSpec::f32("w", vec![8], Ones),
            ],
        )
        .attrs(Attrs::new().set("eps", 1e-6f64)),
    );
    // The declared weight convention: the trunk's 1 + weight (offset 1.0).
    cases.push(
        Case::new(
            "rmsnorm",
            vec![
                InputSpec::f32("x", vec![4, 8], Ramp),
                InputSpec::f32("w", vec![8], Ones),
            ],
        )
        .attrs(
            Attrs::new()
                .set("eps", 1e-6f64)
                .set("weight_offset", 1.0f64),
        ),
    );
    cases.push(
        Case::new(
            "layernorm",
            vec![
                InputSpec::f32("x", vec![4, 8], Ramp),
                InputSpec::f32("w", vec![8], Ones),
                InputSpec::f32("b", vec![8], Zeros),
            ],
        )
        .attrs(Attrs::new().set("eps", 1e-5f64)),
    );
    // The vocabulary the backward pass needs (spec §2.11). These exercise the
    // implementations' shape inference and determinism; the numeric comparison
    // activates as soon as a second provider exists.
    cases.push(
        Case::new(
            "elementwise_unary",
            vec![InputSpec::f32("x", vec![4, 8], Ones)],
        )
        .attrs(Attrs::new().set("kind", "rsqrt")),
    );
    cases.push(
        Case::new(
            "elementwise_unary",
            vec![InputSpec::f32("x", vec![4, 8], Ramp)],
        )
        .attrs(Attrs::new().set("kind", "silu_grad")),
    );
    cases.push(
        Case::new(
            "elementwise_binary",
            vec![
                InputSpec::f32("a", vec![4, 8], Ones),
                InputSpec::f32("b", vec![4, 8], Ones),
            ],
        )
        .attrs(Attrs::new().set("kind", "pow")),
    );
    cases.push(
        Case::new("reduce", vec![InputSpec::f32("x", vec![4, 8], Ramp)]).attrs(
            Attrs::new()
                .set("kind", "max")
                .set("axis", -1i64)
                .set("keepdim", true),
        ),
    );
    cases.push(
        Case::new(
            "compare",
            vec![
                InputSpec::f32("a", vec![4, 8], Ramp),
                InputSpec::f32("b", vec![4, 8], Pseudo { seed: 31 }),
            ],
        )
        .attrs(Attrs::new().set("kind", "ge")),
    );
    cases.push(Case::new(
        "matmul",
        vec![
            InputSpec::f32("a", vec![4, 8], Ramp),
            InputSpec::f32("b", vec![8, 16], Pseudo { seed: 11 }),
        ],
    ));
    cases.push(Case::new(
        "linear",
        vec![
            InputSpec::f32("x", vec![4, 8], Ramp),
            InputSpec::f32("w", vec![8, 16], Pseudo { seed: 13 }),
        ],
    ));
    cases.push(Case::new(
        "bmm",
        vec![
            InputSpec::f32("a", vec![2, 4, 8], Ramp),
            InputSpec::f32("b", vec![2, 8, 16], Pseudo { seed: 17 }),
        ],
    ));
    cases.push(Case::new(
        "transpose",
        vec![InputSpec::f32("x", vec![4, 8], Ramp)],
    ));
    cases.push(
        Case::new("reshape", vec![InputSpec::f32("x", vec![4, 8], Ramp)])
            .attrs(Attrs::new().set("shape", vec![2i64, 16])),
    );
    cases.push(
        Case::new("narrow", vec![InputSpec::f32("x", vec![4, 8], Ramp)]).attrs(
            Attrs::new()
                .set("dim", -1i64)
                .set("start", 2i64)
                .set("length", 4i64),
        ),
    );
    cases.push(
        Case::new(
            "cat",
            vec![
                InputSpec::f32("a", vec![4, 4], Ramp),
                InputSpec::f32("b", vec![4, 4], Pseudo { seed: 19 }),
            ],
        )
        .attrs(Attrs::new().set("dim", -1i64)),
    );
    cases.push(
        Case::new("broadcast", vec![InputSpec::f32("x", vec![4, 1], Ramp)])
            .attrs(Attrs::new().set("shape", vec![4i64, 8])),
    );
    cases.push(
        Case::new(
            "cross_entropy",
            vec![
                InputSpec::f32("logits", vec![4, 8], Ramp),
                InputSpec::indices("targets", vec![4]),
            ],
        )
        .attrs(Attrs::new().set("axis", -1i64)),
    );
    cases.push(Case::new(
        "sdpa",
        vec![
            InputSpec::f32("q", vec![2, 4, 8], Ramp),
            InputSpec::f32("k", vec![2, 4, 8], Pseudo { seed: 23 }),
            InputSpec::f32("v", vec![2, 4, 8], Pseudo { seed: 29 }),
        ],
    ));
    cases.push(
        // Two outputs: the payload and its scale tensor.
        Case::new("quantize", vec![InputSpec::f32("x", vec![4, 8], Ramp)])
            .outputs(2)
            .attrs(
                Attrs::new()
                    .set("scheme", "per_tensor")
                    .set("format", "f8e4m3"),
            ),
    );
    // D5's five new primitives (all_to_all is an intrinsic — see the runtime
    // end-to-end test — so only the four compute ops get gate cases here).
    cases.push(
        Case::new("l2norm", vec![InputSpec::f32("x", vec![4, 8], Ramp)])
            .attrs(Attrs::new().set("dim", -1i64).set("eps", 1e-6f64)),
    );
    cases.push(
        Case::new(
            "rmsnorm_gated",
            vec![
                InputSpec::f32("x", vec![4, 8], Ramp),
                InputSpec::f32("w", vec![8], Ones),
                InputSpec::f32("gate", vec![4, 8], Pseudo { seed: 43 }),
            ],
        )
        .attrs(Attrs::new().set("eps", 1e-6f64).set("gate_act", "silu")),
    );
    cases.push(
        Case::new(
            "causal_conv1d",
            vec![
                InputSpec::f32("x", vec![4, 3], Ramp),
                InputSpec::f32("w", vec![3, 1, 4], Pseudo { seed: 47 }),
            ],
        )
        .attrs(
            Attrs::new()
                .set("kernel", 4i64)
                .set("groups", "channels")
                .set("activation", "silu"),
        ),
    );
    cases.push(
        Case::new(
            "gated_delta_rule",
            vec![
                InputSpec::f32("q", vec![1, 4, 2], Ramp),
                InputSpec::f32("k", vec![1, 4, 2], Pseudo { seed: 53 }),
                InputSpec::f32("v", vec![1, 4, 2], Pseudo { seed: 59 }),
                InputSpec::f32("g", vec![1, 4, 1], Pseudo { seed: 61 }),
                InputSpec::f32("beta", vec![1, 4, 1], Pseudo { seed: 67 }),
            ],
        )
        .attrs(
            Attrs::new()
                .set("state_dtype", "f32")
                .set("chunk_size", 2i64),
        ),
    );
    // moe_layer: the one-operator sparse layer with the static [.., H] in/out
    // contract (op-vocabulary §3.3) — one token (rows=1), E=4, I=2, K=2. The
    // routing indices stay inside [0, E) (the Indices fill is modulo 4, which
    // is exactly the expert count here), so the dropless routing runs with
    // both selected experts firing. No attributes: top_k / norm_topk_prob
    // belong to the router node, not to this op.
    cases.push(Case::new(
        "moe_layer",
        vec![
            InputSpec::f32("h", vec![1, 2], Ramp),
            InputSpec::f32("routing_weights", vec![1, 2], Pseudo { seed: 83 }),
            InputSpec::indices("routing_indices", vec![1, 2]),
            InputSpec::f32("experts_gate_proj", vec![4, 2, 2], Pseudo { seed: 89 }),
            InputSpec::f32("experts_up_proj", vec![4, 2, 2], Pseudo { seed: 97 }),
            InputSpec::f32("experts_down_proj", vec![4, 2, 2], Pseudo { seed: 101 }),
            InputSpec::f32("shared_gate_proj", vec![2, 2], Pseudo { seed: 103 }),
            InputSpec::f32("shared_up_proj", vec![2, 2], Pseudo { seed: 107 }),
            InputSpec::f32("shared_down_proj", vec![2, 2], Pseudo { seed: 109 }),
            InputSpec::f32("shared_expert_gate", vec![1, 2], Pseudo { seed: 113 }),
        ],
    ));
    // rope's T2 completion: partial rotary + theta + position defaults, two
    // outputs — the case the gate used to skip for lack of a convention.
    cases.push(
        Case::new(
            "rope",
            vec![
                InputSpec::f32("q", vec![2, 6], Ramp),
                InputSpec::f32("k", vec![2, 6], Pseudo { seed: 71 }),
            ],
        )
        .outputs(2)
        .attrs(
            Attrs::new()
                .set("rotary_dim", 4i64)
                .set("theta", 1e7f64)
                .set("partial_rotary", true),
        ),
    );
    // sdpa's T2 completion: GQA + causal (the declared expansion cannot
    // replay GQA, so its expansion check skips with that reason — the fused
    // body and determinism are still exercised).
    cases.push(
        Case::new(
            "sdpa",
            vec![
                InputSpec::f32("q", vec![1, 2, 2, 4], Ramp),
                InputSpec::f32("k", vec![1, 2, 1, 4], Pseudo { seed: 73 }),
                InputSpec::f32("v", vec![1, 2, 1, 4], Pseudo { seed: 79 }),
            ],
        )
        .attrs(
            Attrs::new()
                .set("num_heads", 2i64)
                .set("num_kv_heads", 1i64)
                .set("causal", true),
        ),
    );
    cases
}

/// Operators this harness knows exist but has no case for yet, with the reason.
///
/// Printed by the CLI so the coverage gap is visible instead of being an
/// absence nobody notices.
pub fn uncovered_operators() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "adamw",
            "needs optimizer state tensors (m, v, master weights) and step counters; a case has \
             to model a parameter, not a buffer",
        ),
        (
            "topk_router",
            "its top-k selection step has no primitive in the vocabulary (spec §7.5), so there is \
             no independent second implementation to compare against",
        ),
        (
            "embedding",
            "index inputs need a synthesizer that produces in-range indices for each shape",
        ),
        ("gather", "as embedding"),
        ("scatter", "as embedding, plus duplicate-index semantics"),
        (
            "amax_update",
            "maintains state across calls; a single-invocation case would not exercise it",
        ),
        (
            "view",
            "aliases its input's buffer; a case has to compare the aliased pointer, not a buffer",
        ),
        (
            "dequantize",
            "takes a quantized payload and its scale; the harness cannot yet feed one operator's              output into another's input",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    use rustrain_abi::Plugin;

    fn moe_case() -> Case {
        default_cases()
            .into_iter()
            .find(|c| c.op == "moe_layer")
            .expect("the default table has a case for moe_layer")
    }

    /// The case must line up with the operator's declared contract
    /// (`crates/rustrain-kernels/src/op/moe.rs`), inputs in the declared order:
    /// a case that does not match the contract would run a different
    /// computation than every provider implements.
    #[test]
    fn moe_layer_case_matches_the_declared_contract() {
        let case = moe_case();
        assert_eq!(case.n_outputs, 1, "one output, h's exact shape");
        assert!(case.attrs.is_empty(), "moe_layer takes no attributes");

        let expected: Vec<(&str, RsDtype, Vec<i64>)> = vec![
            ("h", RsDtype::F32, vec![1, 2]),
            ("routing_weights", RsDtype::F32, vec![1, 2]),
            ("routing_indices", RsDtype::I32, vec![1, 2]),
            ("experts_gate_proj", RsDtype::F32, vec![4, 2, 2]),
            ("experts_up_proj", RsDtype::F32, vec![4, 2, 2]),
            ("experts_down_proj", RsDtype::F32, vec![4, 2, 2]),
            ("shared_gate_proj", RsDtype::F32, vec![2, 2]),
            ("shared_up_proj", RsDtype::F32, vec![2, 2]),
            ("shared_down_proj", RsDtype::F32, vec![2, 2]),
            ("shared_expert_gate", RsDtype::F32, vec![1, 2]),
        ];
        assert_eq!(case.inputs.len(), expected.len(), "exactly ten inputs");
        for (i, (spec, (name, dtype, shape))) in case.inputs.iter().zip(expected.iter()).enumerate()
        {
            assert_eq!(spec.name, *name, "input {i} name");
            assert_eq!(spec.dtype, *dtype, "input {i} dtype");
            assert_eq!(spec.shape, *shape, "input {i} shape");
        }
    }

    /// The fused body uses routing indices as-is with no wrap-around and
    /// rejects values outside [0, E), so the case's synthesized indices must
    /// all land inside the expert range.
    #[test]
    fn moe_layer_case_routing_indices_are_in_range() {
        let case = moe_case();
        let rows = case.inputs[0].shape[0];
        let e = case.inputs[3].shape[0];
        let k = case.inputs[1].shape[1];
        let spec = &case.inputs[2];
        assert_eq!(spec.dtype, RsDtype::I32);
        assert_eq!(spec.shape, vec![rows, k], "indices mirror [rows, K]");

        let bytes = spec.bytes();
        assert_eq!(bytes.len(), 4 * spec.numel());
        for chunk in bytes.chunks_exact(4) {
            let idx = i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            assert!(
                (0..e).contains(&(idx as i64)),
                "synthesized expert index {idx} is outside [0, {e})"
            );
        }
    }

    /// The case must run green against the real reference provider: the output
    /// keeps h's exact shape and two runs agree bitwise. Numeric stays an
    /// honest skip while the reference is the only provider.
    #[test]
    fn moe_layer_case_runs_green_against_the_reference() {
        let mut registry = Registry::new();
        // SAFETY: the built-in provider descriptor is leaked by `build_plugin`
        // and lives for the process.
        let reference = unsafe { Plugin::from_static(rustrain_kernels::plugin(), "<built-in>") }
            .expect("the built-in provider passes ABI validation");
        registry.add_plugin(reference).expect("registering it");

        let recipe = Recipe::from_toml("[kernel]\ndefault = \"reference\"\n")
            .expect("a recipe with the reference as default");
        let harness = Harness::new(&registry, &recipe);
        let case = moe_case();

        // The output keeps h's exact shape — the option-A static in/out.
        let outputs = harness
            .execute(&case, REFERENCE_VARIANT)
            .expect("the reference implements moe_layer for this case");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].dtype, RsDtype::F32);
        assert_eq!(outputs[0].shape, vec![1, 2]);

        for result in harness.run(&case) {
            assert!(
                result.ok(),
                "moe_layer@{} must be green: {:?}",
                result.variant,
                result.determinism
            );
            assert!(
                matches!(result.determinism, Check::Pass { .. }),
                "the fused body is deterministic, got {:?}",
                result.determinism
            );
        }
    }
}
