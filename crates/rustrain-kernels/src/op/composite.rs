//! Composite operators with their declared primitive expansions (contract
//! R-4): `sdpa`, `cross_entropy`, `adamw`, `topk_router`.
//!
//! The fused bodies below are executed directly; the expansions describe the
//! equivalent primitive composition over the local-tensor-id convention in
//! `rustrain_op.h` (`[0, n_inputs)` = parent inputs, then parent outputs,
//! then temporaries).
//!
//! **Expansion attribute caveat** (ABI limitation, reported rather than
//! patched — `rustrain-abi` is out of this crate's write scope): the current
//! `ExpansionSpec::node` API cannot attach per-node attributes, so nodes that
//! need one (`elementwise_unary` `kind`, `reduce` `kind`, scalar `rhs`) are
//! declared with their required attributes documented here in the operator
//! doc strings. A conformance checker must supply those per-node attributes
//! (e.g. by extending `ExpansionSpec` with `node_with_attrs`).

use rustrain_abi::author::ExpansionSpec;
use rustrain_abi::ffi::{MAX_RANK, RsAttrs, RsDtype, RsTensor};

use crate::attrs::{attr_f64, attr_i64};
use crate::dispatch::{Call, run};
use crate::error::{OpResult, err, fail};
use crate::tensor::{SmallShape, expect_out, set_output_desc};

macro_rules! infer_entry {
    ($name:ident, $op:literal, $body:path) => {
        pub(crate) unsafe extern "C" fn $name(
            in_: *const *const RsTensor,
            n_in: u32,
            out: *const *mut RsTensor,
            n_out: u32,
            attrs: *const RsAttrs,
        ) -> i32 {
            // SAFETY: ABI contract; pointers are the framework's.
            unsafe { run($op, in_, n_in, out, n_out, attrs, $body) }
        }
    };
}
macro_rules! exec_entry {
    ($name:ident, $op:literal, $body:path) => {
        pub(crate) unsafe extern "C" fn $name(
            ctx: *mut rustrain_abi::ffi::RsCtx,
            in_: *const *const RsTensor,
            n_in: u32,
            out: *const *mut RsTensor,
            n_out: u32,
            attrs: *const RsAttrs,
        ) -> i32 {
            let _ = ctx;
            // SAFETY: ABI contract; pointers are the framework's.
            unsafe { run($op, in_, n_in, out, n_out, attrs, $body) }
        }
    };
}

// ── sdpa ────────────────────────────────────────────────────────────────────

fn sdpa_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((3, 3))?;
    c.expect_out_count(1)?;
    let q = c.in_t(0);
    let k = c.in_t(1);
    let v = c.in_t(2);
    for (i, t) in [(0usize, q), (1, k), (2, v)] {
        if t.dtype != RsDtype::F32 {
            return Err(err(c.op, format!("input {i} has dtype {}, expected f32", t.dtype)));
        }
    }
    let rank = q.rank as usize;
    if rank < 3 || k.rank as usize != rank || v.rank as usize != rank {
        return Err(fail!(
            c.op,
            "sdpa expects q, k, v of equal rank >= 3, got ranks {}, {}, {}",
            q.rank,
            k.rank,
            v.rank
        ));
    }
    for d in 0..rank - 2 {
        if q.shape[d] != k.shape[d] || q.shape[d] != v.shape[d] {
            return Err(fail!(
                c.op,
                "sdpa batch dim {d} mismatch: q={}, k={}, v={} (batch dims must be identical)",
                q.shape[d],
                k.shape[d],
                v.shape[d]
            ));
        }
    }
    let (s, dv) = (q.shape[rank - 2], q.shape[rank - 1]);
    let (t, dk) = (k.shape[rank - 2], k.shape[rank - 1]);
    if dk != dv || k.shape[rank - 2] != v.shape[rank - 2] {
        return Err(fail!(
            c.op,
            "sdpa dim mismatch: q {:?}, k {:?}, v {:?}",
            q.dims(),
            k.dims(),
            v.dims()
        ));
    }
    let _ = attr_f64(a, "scale");
    let mut shape = SmallShape {
        len: rank,
        dims: [0; MAX_RANK],
    };
    shape.dims[..rank].copy_from_slice(q.dims());
    shape.dims[rank - 1] = v.shape[rank - 1];
    let o = c.out_t(0);
    set_output_desc(o, RsDtype::F32, shape.as_slice());
    let _ = (s, t);
    Ok(())
}

fn sdpa_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((3, 3))?;
    c.expect_out_count(1)?;
    let q = c.in_t(0);
    let k = c.in_t(1);
    let v = c.in_t(2);
    let rank = q.rank as usize;
    let (s, d) = (q.dims()[rank - 2] as usize, q.dims()[rank - 1] as usize);
    let t = k.dims()[rank - 2] as usize;
    let dv = v.dims()[rank - 1] as usize;
    let batches: usize = q.dims()[..rank - 2].iter().map(|&x| x as usize).product();
    // Convention (documented): when `scale` is absent the fused body uses
    // scale = 1.0, exactly like the declared expansion's softmax node; the
    // caller passes 1/sqrt(D) explicitly if it wants scaled attention. This
    // keeps fused and expansion bitwise comparable, which is the point of a
    // reference backend.
    let scale = attr_f64(a, "scale").unwrap_or(1.0) as f32;
    let mut shape = SmallShape {
        len: rank,
        dims: [0; MAX_RANK],
    };
    shape.dims[..rank].copy_from_slice(q.dims());
    shape.dims[rank - 1] = v.dims()[rank - 1];
    expect_out(c.out_t(0), c.op, RsDtype::F32, shape.as_slice())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let qv = unsafe { crate::tensor::f32_in(c.op, "q", q) }?;
    let kv = unsafe { crate::tensor::f32_in(c.op, "k", k) }?;
    let vv = unsafe { crate::tensor::f32_in(c.op, "v", v) }?;
    let mut ov = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    // Fused body = the declared expansion, with stable softmax: per batch,
    // scores = q @ k^T (d ascending), row-softmax over t, then @ v (t
    // ascending). All reductions in fixed index order — deterministic.
    let mut scores = vec![0.0f32; s * t];
    let mut probs = vec![0.0f32; s * t];
    for b in 0..batches {
        let qb = &qv.as_slice().expect("contiguous")[b * s * d..(b + 1) * s * d];
        let kb = &kv.as_slice().expect("contiguous")[b * t * d..(b + 1) * t * d];
        let vb = &vv.as_slice().expect("contiguous")[b * t * dv..(b + 1) * t * dv];
        for i in 0..s {
            for j in 0..t {
                let mut acc = 0.0f32;
                for dd in 0..d {
                    acc += qb[i * d + dd] * kb[j * d + dd];
                }
                scores[i * t + j] = acc;
            }
        }
        for i in 0..s {
            let row = &scores[i * t..(i + 1) * t];
            let m = row.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
            let sum = row.iter().fold(0.0f32, |acc, &x| acc + ((x - m) * scale).exp());
            for j in 0..t {
                probs[i * t + j] = ((row[j] - m) * scale).exp() / sum;
            }
        }
        let ob = &mut ov.as_slice_mut().expect("contiguous")[b * s * dv..(b + 1) * s * dv];
        for i in 0..s {
            for jj in 0..dv {
                let mut acc = 0.0f32;
                for j in 0..t {
                    acc += probs[i * t + j] * vb[j * dv + jj];
                }
                ob[i * dv + jj] = acc;
            }
        }
    }
    Ok(())
}

/// Declared expansion of `sdpa` (unscaled form; see the op doc).
pub(crate) fn sdpa_expansion() -> ExpansionSpec {
    let mut e = ExpansionSpec::new(3, 1);
    let kt = e.temp();
    let s = e.temp();
    let p = e.temp();
    e.node("transpose", &[1], &[kt])
        .node("matmul", &[0, kt], &[s])
        .node("softmax", &[s], &[p])
        .node("matmul", &[p, 2], &[3])
}

// ── cross_entropy ───────────────────────────────────────────────────────────

fn cross_entropy_infer_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let logits = c.in_t(0);
    let targets = c.in_t(1);
    if logits.dtype != RsDtype::F32 {
        return Err(err(
            c.op,
            format!("input 'logits' has dtype {}, expected f32", logits.dtype),
        ));
    }
    match targets.dtype {
        RsDtype::I32 | RsDtype::I64 => {}
        other => {
            return Err(err(
                c.op,
                format!("input 'targets' has dtype {other}, expected i32 or i64"),
            ));
        }
    }
    if logits.rank != 2 {
        return Err(err(
            c.op,
            format!("cross_entropy expects logits [N, C], got rank {}", logits.rank),
        ));
    }
    if targets.rank != 1 || targets.shape[0] != logits.shape[0] {
        return Err(err(
            c.op,
            format!(
                "cross_entropy targets must be [N={}], got {:?}",
                logits.shape[0],
                targets.dims()
            ),
        ));
    }
    // Loss is a rank-0 scalar (mean reduction, documented).
    set_output_desc(c.out_t(0), RsDtype::F32, &[]);
    Ok(())
}

fn cross_entropy_exec_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let logits = c.in_t(0);
    let targets = c.in_t(1);
    expect_out(c.out_t(0), c.op, RsDtype::F32, &[])?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let lv = unsafe { crate::tensor::f32_in(c.op, "logits", logits) }?;
    let ids = unsafe { crate::tensor::indices_i64(c.op, "targets", targets) }?;
    let mut ov = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let (n, ch) = (lv.shape()[0], lv.shape()[1]);
    // Stable log-sum-exp form (documented): loss = mean_n(lse_n - x[n, t_n]).
    // This stays finite for extreme logits (e.g. [1000, -1000, 0] → 1000)
    // where the naive log(softmax) expansion underflows to -inf; the two
    // agree within tolerance wherever the naive form is well-conditioned.
    let mut total = 0.0f64;
    for i in 0..n {
        let row = &lv.as_slice().expect("contiguous")[i * ch..(i + 1) * ch];
        let m = row.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let sum = row.iter().fold(0.0f32, |acc, &x| acc + (x - m).exp());
        let lse = m as f64 + (sum as f64).ln();
        let t = ids[i];
        if t < 0 || t >= ch as i64 {
            return Err(fail!(
                c.op,
                "target {t} out of range [0, {ch}) (negative targets are not wrapped)"
            ));
        }
        total += lse - row[t as usize] as f64;
    }
    if let Some(first) = ov.iter_mut().next() {
        *first = (total / n as f64) as f32;
    }
    Ok(())
}

/// Declared expansion of `cross_entropy`.
///
/// Node attributes required by the primitives (which the current
/// `ExpansionSpec` API cannot attach — see the module doc):
/// node 1: `elementwise_unary` kind=log; node 3: `elementwise_unary`
/// kind=neg; node 4: `reduce` kind=mean (no axis → scalar).
pub(crate) fn cross_entropy_expansion() -> ExpansionSpec {
    let mut e = ExpansionSpec::new(2, 1);
    let p = e.temp();
    let lp = e.temp();
    let per = e.temp();
    let neg = e.temp();
    e.node("softmax", &[0], &[p])
        .node("elementwise_unary", &[p], &[lp])
        .node("gather", &[lp, 1], &[per])
        .node("elementwise_unary", &[per], &[neg])
        .node("reduce", &[neg], &[2])
}

// ── adamw ───────────────────────────────────────────────────────────────────

fn adamw_params(a: &RsAttrs, op: &'static str) -> OpResult<(f32, f32, f32, f32, f32, i64)> {
    let lr = attr_f64(a, "lr").unwrap_or(1e-3) as f32;
    let b1 = attr_f64(a, "beta1").unwrap_or(0.9) as f32;
    let b2 = attr_f64(a, "beta2").unwrap_or(0.999) as f32;
    let eps = attr_f64(a, "eps").unwrap_or(1e-8) as f32;
    let wd = attr_f64(a, "weight_decay").unwrap_or(0.0) as f32;
    let step = attr_i64(a, "step").unwrap_or(1);
    if step < 1 {
        return Err(err(op, format!("adamw 'step' must be >= 1, got {step}")));
    }
    if !(0.0..1.0).contains(&b1) || !(0.0..1.0).contains(&b2) {
        return Err(err(
            op,
            format!("adamw beta1/beta2 must be in [0, 1), got {b1} and {b2}"),
        ));
    }
    Ok((lr, b1, b2, eps, wd, step))
}

fn adamw_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((4, 4))?;
    c.expect_out_count(3)?;
    let p = c.in_t(0);
    for (i, who) in [(1usize, "grad"), (2, "exp_avg"), (3, "exp_avg_sq")] {
        let t = c.in_t(i);
        if t.dtype != RsDtype::F32 {
            return Err(err(c.op, format!("input '{who}' has dtype {}, expected f32", t.dtype)));
        }
        if t.dims() != p.dims() {
            return Err(err(
                c.op,
                format!(
                    "adamw input '{who}' shape {:?} must match param shape {:?}",
                    t.dims(),
                    p.dims()
                ),
            ));
        }
    }
    adamw_params(a, c.op)?;
    for i in 0..3 {
        set_output_desc(c.out_t(i), RsDtype::F32, p.dims());
    }
    Ok(())
}

fn adamw_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((4, 4))?;
    c.expect_out_count(3)?;
    let p = c.in_t(0);
    let (lr, b1, b2, eps, wd, step) = adamw_params(a, c.op)?;
    // Bias corrections are integer powers: exact and deterministic.
    let c1 = 1.0 - b1.powi(step as i32);
    let c2 = 1.0 - b2.powi(step as i32);
    for i in 0..3 {
        expect_out(c.out_t(i), c.op, RsDtype::F32, p.dims())?;
    }
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let pv = unsafe { crate::tensor::f32_in(c.op, "param", p) }?;
    let gv = unsafe { crate::tensor::f32_in(c.op, "grad", c.in_t(1)) }?;
    let mv = unsafe { crate::tensor::f32_in(c.op, "exp_avg", c.in_t(2)) }?;
    let vv = unsafe { crate::tensor::f32_in(c.op, "exp_avg_sq", c.in_t(3)) }?;
    let mut po = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let mut mo = unsafe { crate::tensor::f32_out(c.op, c.out_t(1)) }?;
    let mut vo = unsafe { crate::tensor::f32_out(c.op, c.out_t(2)) }?;
    // Pointwise in ascending index order (decoupled weight decay, standard
    // bias correction). exp_avg' and exp_avg_sq' are written back so the
    // caller can persist them.
    for i in 0..pv.len() {
        let m = b1 * mv[i] + (1.0 - b1) * gv[i];
        let v = b2 * vv[i] + (1.0 - b2) * gv[i] * gv[i];
        let mh = m / c1;
        let vh = v / c2;
        mo[i] = m;
        vo[i] = v;
        po[i] = pv[i] - lr * (mh / (vh.sqrt() + eps) + wd * pv[i]);
    }
    Ok(())
}

/// Declared expansion of `adamw` at the default hyperparameters
/// (lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0, step=1).
///
/// Node attributes required (not attachable via the current `ExpansionSpec`
/// API — see the module doc), using `elementwise_binary` (`kind` + scalar
/// `rhs` for the single-input nodes) and `elementwise_unary` (`kind`):
/// 1 mul g*(1-b1); 2 mul m*b1; 3 add → m'; 4 mul g*g; 5 mul g^2*(1-b2);
/// 6 mul v*b2; 7 add → v'; 8 mul m'/(1-b1^t); 9 mul v'/(1-b2^t);
/// 10 sqrt; 11 add eps; 12 div mhat/den; 13 mul lr; 14 mul p*(lr*wd);
/// 15 sub p - step13; 16 sub - step14 → p'.
pub(crate) fn adamw_expansion() -> ExpansionSpec {
    let mut e = ExpansionSpec::new(4, 3);
    let t7 = e.temp();
    let t8 = e.temp();
    let t9 = e.temp();
    let t10 = e.temp();
    let t11 = e.temp();
    let t12 = e.temp();
    let t13 = e.temp();
    let t14 = e.temp();
    let t15 = e.temp();
    let t16 = e.temp();
    let t17 = e.temp();
    let t18 = e.temp();
    let t19 = e.temp();
    e.node("elementwise_binary", &[1], &[t7])
        .node("elementwise_binary", &[2], &[t8])
        .node("elementwise_binary", &[t8, t7], &[5])
        .node("elementwise_binary", &[1, 1], &[t9])
        .node("elementwise_binary", &[t9], &[t10])
        .node("elementwise_binary", &[3], &[t11])
        .node("elementwise_binary", &[t11, t10], &[6])
        .node("elementwise_binary", &[5], &[t12])
        .node("elementwise_binary", &[6], &[t13])
        .node("elementwise_unary", &[t13], &[t14])
        .node("elementwise_binary", &[t14], &[t15])
        .node("elementwise_binary", &[t12, t15], &[t16])
        .node("elementwise_binary", &[t16], &[t17])
        .node("elementwise_binary", &[0], &[t18])
        .node("elementwise_binary", &[0, t17], &[t19])
        .node("elementwise_binary", &[t19, t18], &[4])
}

// ── topk_router ─────────────────────────────────────────────────────────────

fn topk_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(2)?;
    let logits = c.in_t(0);
    if logits.dtype != RsDtype::F32 {
        return Err(err(
            c.op,
            format!("input 'logits' has dtype {}, expected f32", logits.dtype),
        ));
    }
    if logits.rank != 2 {
        return Err(err(
            c.op,
            format!("topk_router expects logits [N, E], got rank {}", logits.rank),
        ));
    }
    let e = logits.shape[1];
    let k = attr_i64(a, "top_k").unwrap_or(2);
    if k < 1 || k > e {
        return Err(err(
            c.op,
            format!("topk_router 'top_k' must be in [1, E={e}], got {k}"),
        ));
    }
    let n = logits.shape[0];
    // Out 0: routing weights (raw softmax probabilities of the selected
    // experts); out 1: expert indices.
    set_output_desc(c.out_t(0), RsDtype::F32, &[n, k]);
    set_output_desc(c.out_t(1), RsDtype::I32, &[n, k]);
    Ok(())
}

fn topk_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(2)?;
    let logits = c.in_t(0);
    let n = logits.shape[0] as usize;
    let e = logits.shape[1] as usize;
    let k = attr_i64(a, "top_k").unwrap_or(2) as usize;
    expect_out(c.out_t(0), c.op, RsDtype::F32, &[logits.shape[0], k as i64])?;
    expect_out(c.out_t(1), c.op, RsDtype::I32, &[logits.shape[0], k as i64])?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let lv = unsafe { crate::tensor::f32_in(c.op, "logits", logits) }?;
    let mut wo = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let mut io = unsafe { crate::tensor::i32_out(c.op, c.out_t(1)) }?;
    // Per-row stable softmax, then a deterministic insertion selection: the
    // comparison is strict `>`, so on ties the lower expert index wins.
    for i in 0..n {
        let row = &lv.as_slice().expect("contiguous")[i * e..(i + 1) * e];
        let m = row.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let sum = row.iter().fold(0.0f32, |acc, &x| acc + (x - m).exp());
        let mut best: Vec<(f32, usize)> = Vec::with_capacity(k);
        for (ex, &x) in row.iter().enumerate() {
            let p = (x - m).exp() / sum;
            let mut pos = best.len();
            while pos > 0 && p > best[pos - 1].0 {
                pos -= 1;
            }
            if best.len() < k {
                best.insert(pos, (p, ex));
            } else if pos < k {
                best.insert(pos, (p, ex));
                best.truncate(k);
            }
        }
        for (j, &(p, ex)) in best.iter().enumerate() {
            wo[i * k + j] = p;
            io[i * k + j] = ex as i32;
        }
    }
    Ok(())
}

/// Declared expansion of `topk_router`.
///
/// Only the gating-probability path is expressible in the fixed primitive
/// vocabulary: `softmax(logits) -> probs`. The top-k *selection* (weights =
/// top-k subset of the probabilities, indices = their positions, ties broken
/// by the lower index) has no primitive equivalent in spec §2.4 (there is no
/// top-k / argmax operator), so that part is documented in the op doc rather
/// than faked with a non-equivalent node.
pub(crate) fn topk_expansion() -> ExpansionSpec {
    let mut e = ExpansionSpec::new(1, 2);
    let p = e.temp();
    e.node("softmax", &[0], &[p])
}

infer_entry!(sdpa_infer, "sdpa", sdpa_infer_body);
exec_entry!(sdpa_exec, "sdpa", sdpa_exec_body);
infer_entry!(ce_infer, "cross_entropy", cross_entropy_infer_body);
exec_entry!(ce_exec, "cross_entropy", cross_entropy_exec_body);
infer_entry!(adamw_infer, "adamw", adamw_infer_body);
exec_entry!(adamw_exec, "adamw", adamw_exec_body);
infer_entry!(topk_infer, "topk_router", topk_infer_body);
exec_entry!(topk_exec, "topk_router", topk_exec_body);
