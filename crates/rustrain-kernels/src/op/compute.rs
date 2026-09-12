//! L0 compute operators: `matmul`, `linear`, `bmm`, `elementwise_unary`,
//! `elementwise_binary`, `reduce`, `softmax`, `rmsnorm`, `layernorm`, `rope`.
//!
//! Determinism rules used throughout (and why): every reduction accumulates
//! in ascending index order with plain f32 adds — no threads, no trees, no
//! `HashMap` iteration — so the same inputs twice produce bitwise-identical
//! outputs. This is the reference provider's whole reason to exist.

use ndarray::{ArrayView1, ArrayViewD, Axis, Dimension, IxDyn, Zip};
use rustrain_abi::ffi::{RsAttrs, RsTensor};

use crate::attrs::{attr_f64, attr_i64, require_str_of};
use crate::dispatch::{Call, run};
use crate::error::{OpResult, err, fail};
use crate::tensor::{
    SmallShape, broadcast_shape_small, expect_out, expect_dtype, resolve_axis, set_output_desc,
};

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

/// All compute ops take f32 inputs (declared in `requires`); this validates
/// one descriptor at infer time without touching its data.
fn check_f32_desc(t: &RsTensor, op: &'static str, who: &str) -> OpResult<()> {
    expect_dtype(t, rustrain_abi::ffi::RsDtype::F32, op, who)
}

// ── matmul ──────────────────────────────────────────────────────────────────

/// Shared matmul shape plan (used by infer and execute, so direct execute
/// calls get the same validation): returns (m, k, n, transpose_b).
fn matmul_plan(a: &RsTensor, b: &RsTensor, attrs: &RsAttrs, op: &'static str) -> OpResult<(i64, i64, i64, bool)> {
    check_f32_desc(a, op, "a")?;
    check_f32_desc(b, op, "b")?;
    if a.rank != 2 || b.rank != 2 {
        return Err(fail!(
            op,
            "matmul expects rank-2 inputs, got ranks {} and {}",
            a.rank,
            b.rank
        ));
    }
    let b_t = crate::attrs::attr_bool(attrs, "transpose_b").unwrap_or(false);
    // With transpose_b, b is [N, K] and the product is a @ b^T = [M, N];
    // without it b is [K, N]. The attribute must be declared — the shapes
    // are never used to guess it.
    let (k, n) = if b_t {
        if a.shape[1] != b.shape[1] {
            return Err(fail!(
                op,
                "matmul inner dims mismatch with transpose_b: a is {:?}, b is {:?} \
                 (b is [N, K] and must satisfy a.K == b.K)",
                a.dims(),
                b.dims()
            ));
        }
        (a.shape[1], b.shape[0])
    } else {
        if a.shape[1] != b.shape[0] {
            return Err(fail!(
                op,
                "matmul inner dims mismatch: a is {:?}, b is {:?}",
                a.dims(),
                b.dims()
            ));
        }
        (a.shape[1], b.shape[1])
    };
    Ok((a.shape[0], k, n, b_t))
}

fn matmul_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let (m, _k, n, _bt) = matmul_plan(c.in_t(0), c.in_t(1), a, c.op)?;
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, &[m, n]);
    Ok(())
}

fn matmul_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let (m, k, n, b_t) = matmul_plan(c.in_t(0), c.in_t(1), a, c.op)?;
    let shape = [m, n];
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, &shape)?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let av = unsafe { crate::tensor::f32_in(c.op, "a", c.in_t(0)) }?;
    let bv = unsafe { crate::tensor::f32_in(c.op, "b", c.in_t(1)) }?;
    let mut cv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let (m, k, n) = (m as usize, k as usize, n as usize);
    // Naive triple loop with k accumulated in ascending order: the fixed
    // summation order is what makes the result bitwise reproducible.
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += if b_t {
                    av[[i, kk]] * bv[[j, kk]]
                } else {
                    av[[i, kk]] * bv[[kk, j]]
                };
            }
            cv[[i, j]] = acc;
        }
    }
    Ok(())
}

// ── linear ──────────────────────────────────────────────────────────────────

/// `y = x @ w (+ b)`. Weight convention (documented in the op doc): `w` is
/// `[K, N]` and the input is `[..., K]`, so `linear` is literally a batched
/// `matmul` plus an optional bias broadcast over the last dim.
fn linear_infer_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 3))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let w = c.in_t(1);
    check_f32_desc(x, c.op, "x")?;
    check_f32_desc(w, c.op, "w")?;
    if x.rank < 1 || w.rank != 2 {
        return Err(fail!(
            c.op,
            "linear expects x with rank >= 1 and w with rank 2, got ranks {} and {}",
            x.rank,
            w.rank
        ));
    }
    let k = x.dims()[x.rank as usize - 1];
    if w.shape[0] != k {
        return Err(fail!(
            c.op,
            "linear inner dim mismatch: x.last = {k}, w is {:?} (expected [K={k}, N])",
            w.dims()
        ));
    }
    if c.n_in() == 3 {
        let b = c.in_t(2);
        check_f32_desc(b, c.op, "b")?;
        if b.rank != 1 || b.shape[0] != w.shape[1] {
            return Err(fail!(
                c.op,
                "linear bias must be [N={}], got {:?}",
                w.shape[1],
                b.dims()
            ));
        }
    }
    let mut shape = SmallShape {
        len: x.rank as usize,
        dims: [0; rustrain_abi::ffi::MAX_RANK],
    };
    shape.dims[..shape.len].copy_from_slice(x.dims());
    shape.dims[shape.len - 1] = w.shape[1];
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, shape.as_slice());
    Ok(())
}

fn linear_exec_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 3))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let w = c.in_t(1);
    check_f32_desc(x, c.op, "x")?;
    check_f32_desc(w, c.op, "w")?;
    if x.rank < 1 || x.rank as usize > rustrain_abi::ffi::MAX_RANK || w.rank != 2 {
        return Err(fail!(
            c.op,
            "linear expects x with rank 1..=8 and w with rank 2, got ranks {} and {}",
            x.rank,
            w.rank
        ));
    }
    let k = x.dims()[x.rank as usize - 1] as usize;
    if w.shape[0] != k as i64 {
        return Err(fail!(
            c.op,
            "linear inner dim mismatch: x.last = {k}, w is {:?} (expected [K={k}, N])",
            w.dims()
        ));
    }
    let n = w.shape[1] as usize;
    if c.n_in() == 3 {
        let b = c.in_t(2);
        check_f32_desc(b, c.op, "b")?;
        if b.rank != 1 || b.shape[0] != w.shape[1] {
            return Err(fail!(
                c.op,
                "linear bias must be [N={}], got {:?}",
                w.shape[1],
                b.dims()
            ));
        }
    }
    let mut shape = SmallShape {
        len: x.rank as usize,
        dims: [0; rustrain_abi::ffi::MAX_RANK],
    };
    shape.dims[..shape.len].copy_from_slice(x.dims());
    shape.dims[shape.len - 1] = w.shape[1];
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, shape.as_slice())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let wv = unsafe { crate::tensor::f32_in(c.op, "w", w) }?;
    let bias = if c.n_in() == 3 {
        // SAFETY: descriptor liveness is the ABI caller's contract.
        Some(unsafe { crate::tensor::f32_in(c.op, "b", c.in_t(2)) }?)
    } else {
        None
    };
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    // Contiguous flat slices: indexing is explicit, ordering is fixed.
    let xs = xv.as_slice().expect("contiguous");
    let ws = wv.as_slice().expect("contiguous");
    let bs = bias.as_ref().map(|b| b.as_slice().expect("contiguous"));
    let ys = yv.as_slice_mut().expect("contiguous");
    let batches = xs.len() / k;
    // Deterministic per-batch triple loop, k ascending.
    for b in 0..batches {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += xs[b * k + kk] * ws[kk * n + j];
            }
            ys[b * n + j] = match bs {
                Some(bv) => acc + bv[j],
                None => acc,
            };
        }
    }
    Ok(())
}

// ── bmm ─────────────────────────────────────────────────────────────────────

/// Shared bmm shape plan for infer and execute: returns (rank, m, k, n, b_t).
fn bmm_plan(
    a: &RsTensor,
    b: &RsTensor,
    attrs: &RsAttrs,
    op: &'static str,
) -> OpResult<(usize, i64, i64, i64, bool)> {
    check_f32_desc(a, op, "a")?;
    check_f32_desc(b, op, "b")?;
    if a.rank != b.rank || a.rank < 3 {
        return Err(fail!(
            op,
            "bmm expects equal ranks >= 3, got ranks {} and {}",
            a.rank,
            b.rank
        ));
    }
    let r = a.rank as usize;
    for d in 0..r - 2 {
        if a.shape[d] != b.shape[d] {
            return Err(fail!(
                op,
                "bmm batch dim {d} mismatch: {} vs {} (batch dims must be identical, \
                 broadcasting is not supported)",
                a.shape[d],
                b.shape[d]
            ));
        }
    }
    let b_t = crate::attrs::attr_bool(attrs, "transpose_b").unwrap_or(false);
    let (m, k) = (a.shape[r - 2], a.shape[r - 1]);
    let (k2, n) = if b_t {
        // b is [.., N, K] and the product is a @ b^T.
        (b.shape[r - 1], b.shape[r - 2])
    } else {
        (b.shape[r - 2], b.shape[r - 1])
    };
    if k != k2 {
        return Err(fail!(
            op,
            "bmm inner dims mismatch: a is {:?}, b is {:?} (transpose_b = {b_t})",
            a.dims(),
            b.dims()
        ));
    }
    Ok((r, m, k, n, b_t))
}

fn bmm_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let (r, _m, _k, n, _bt) = bmm_plan(c.in_t(0), c.in_t(1), a, c.op)?;
    let mut shape = SmallShape {
        len: r,
        dims: [0; rustrain_abi::ffi::MAX_RANK],
    };
    shape.dims[..r].copy_from_slice(c.in_t(0).dims());
    shape.dims[r - 1] = n;
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, shape.as_slice());
    Ok(())
}

fn bmm_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let (r, m, k, n, b_t) = bmm_plan(c.in_t(0), c.in_t(1), a, c.op)?;
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let batches: usize = c.in_t(0).dims()[..r - 2].iter().map(|&d| d as usize).product();
    let mut shape = SmallShape {
        len: r,
        dims: [0; rustrain_abi::ffi::MAX_RANK],
    };
    shape.dims[..r].copy_from_slice(c.in_t(0).dims());
    shape.dims[r - 1] = n as i64;
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, shape.as_slice())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let av = unsafe { crate::tensor::f32_in(c.op, "a", c.in_t(0)) }?;
    let bv = unsafe { crate::tensor::f32_in(c.op, "b", c.in_t(1)) }?;
    let mut cv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let xs = av.as_slice().expect("contiguous");
    let bs = bv.as_slice().expect("contiguous");
    let ys = cv.as_slice_mut().expect("contiguous");
    for batch in 0..batches {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += if b_t {
                        xs[batch * (m * k) + i * k + kk] * bs[batch * (n * k) + j * k + kk]
                    } else {
                        xs[batch * (m * k) + i * k + kk] * bs[batch * (k * n) + kk * n + j]
                    };
                }
                ys[batch * (m * n) + i * n + j] = acc;
            }
        }
    }
    Ok(())
}

// ── elementwise_unary ───────────────────────────────────────────────────────

/// Accepted `kind` values. `neg` and `sqrt` extend the core list because the
/// declared expansions of `cross_entropy` (negation of the gathered
/// log-probability) and `adamw` (sqrt of the second moment) must be
/// expressible in the fixed primitive vocabulary; see the op doc.
pub(crate) const UNARY_KINDS: &[&str] =
    &["silu", "gelu", "sigmoid", "tanh", "relu", "exp", "log", "neg", "sqrt"];

fn unary_fn(op: &'static str, kind: &str) -> OpResult<fn(f32) -> f32> {
    Ok(match kind {
        // x * sigmoid(x): the x/(1+e^-x) form is exact for both tails
        // (e^-x saturates to 0 or inf, never NaN).
        "silu" => |x: f32| x / (1.0 + (-x).exp()),
        // GELU via the tanh approximation — the convention most kernels use.
        // Chosen and documented because the primitive vocabulary has no erf.
        "gelu" => |x: f32| {
            let c = std::f32::consts::FRAC_2_PI.sqrt() * (x + 0.044_715 * x.powi(3));
            0.5 * x * (1.0 + c.tanh())
        },
        "sigmoid" => |x: f32| 1.0 / (1.0 + (-x).exp()),
        "tanh" => f32::tanh,
        "relu" => |x: f32| x.max(0.0),
        "exp" => f32::exp,
        // Natural logarithm; ln(0) = -inf per IEEE.
        "log" => f32::ln,
        "neg" => |x: f32| -x,
        "sqrt" => f32::sqrt,
        other => {
            return Err(err(
                op,
                format!(
                    "unknown kind '{other}' for elementwise_unary; accepted values: {}",
                    UNARY_KINDS.join(", ")
                ),
            ));
        }
    })
}

fn unary_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "x")?;
    require_str_of(a, "kind", UNARY_KINDS, c.op)?;
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, x.dims());
    Ok(())
}

fn unary_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let kind = require_str_of(a, "kind", UNARY_KINDS, c.op)?;
    let f = unary_fn(c.op, kind)?;
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, x.dims())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    for (o, &v) in yv.iter_mut().zip(xv.iter()) {
        *o = f(v);
    }
    Ok(())
}

// ── elementwise_binary ──────────────────────────────────────────────────────

pub(crate) const BINARY_KINDS: &[&str] = &["add", "sub", "mul", "div", "maximum"];

fn binary_fn(op: &'static str, kind: &str) -> OpResult<fn(f32, f32) -> f32> {
    Ok(match kind {
        "add" => |a, b| a + b,
        "sub" => |a, b| a - b,
        "mul" => |a, b| a * b,
        // IEEE division: x/0 -> +/-inf, 0/0 -> NaN.
        "div" => |a, b| a / b,
        // f32::max ignores NaN like IEEE maximumNumber; NaN inputs are
        // outside the reference provider's contract anyway.
        "maximum" => f32::max,
        other => {
            return Err(err(
                op,
                format!(
                    "unknown kind '{other}' for elementwise_binary; accepted values: {}",
                    BINARY_KINDS.join(", ")
                ),
            ));
        }
    })
}

fn binary_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 2))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "a")?;
    require_str_of(a, "kind", BINARY_KINDS, c.op)?;
    let shape = if c.n_in() == 2 {
        let b = c.in_t(1);
        check_f32_desc(b, c.op, "b")?;
        if attr_f64(a, "rhs").is_some() {
            return Err(err(
                c.op,
                "attribute 'rhs' is only valid when a single input is given \
                 (scalar-constant form)",
            ));
        }
        broadcast_shape_small(x.dims(), b.dims(), c.op)?
    } else {
        if attr_f64(a, "rhs").is_none() {
            return Err(err(
                c.op,
                "attribute 'rhs' (f64 scalar) is required when a single input is given",
            ));
        }
        SmallShape::of(x)
    };
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, shape.as_slice());
    Ok(())
}

fn binary_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 2))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let kind = require_str_of(a, "kind", BINARY_KINDS, c.op)?;
    let f = binary_fn(c.op, kind)?;
    let out_shape = if c.n_in() == 2 {
        let b = c.in_t(1);
        broadcast_shape_small(x.dims(), b.dims(), c.op)?
    } else {
        SmallShape::of(x)
    };
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, out_shape.as_slice())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "a", x) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    if c.n_in() == 1 {
        let rhs = attr_f64(a, "rhs").expect("validated in infer") as f32;
        for (o, &v) in yv.iter_mut().zip(xv.iter()) {
            *o = f(v, rhs);
        }
    } else {
        // SAFETY: descriptor liveness is the ABI caller's contract.
        let bv = unsafe { crate::tensor::f32_in(c.op, "b", c.in_t(1)) }?;
        let dims: Vec<usize> = out_shape
            .as_slice()
            .iter()
            .map(|&d| d as usize)
            .collect();
        let ab = xv
            .broadcast(IxDyn(&dims))
            .ok_or_else(|| err(c.op, "broadcast of 'a' failed unexpectedly"))?;
        let bb = bv
            .broadcast(IxDyn(&dims))
            .ok_or_else(|| err(c.op, "broadcast of 'b' failed unexpectedly"))?;
        // Zip iterates in the fixed logical order — deterministic.
        Zip::from(&ab).and(&bb).and(&mut yv).for_each(|&a, &b, y| *y = f(a, b));
    }
    Ok(())
}

// ── reduce ──────────────────────────────────────────────────────────────────

pub(crate) const REDUCE_KINDS: &[&str] = &["sum", "mean", "max", "amax"];

/// One lane reduction; `lane` is a 1-D view over the reduced axis. Folds run
/// left to right (ascending) so summation order is fixed.
fn reduce_lane(op: &'static str, kind: &str, lane: ArrayView1<f32>) -> OpResult<f32> {
    Ok(match kind {
        "sum" => lane.iter().fold(0.0f32, |acc, &v| acc + v),
        "mean" => {
            let s = lane.iter().fold(0.0f32, |acc, &v| acc + v);
            if lane.is_empty() {
                0.0
            } else {
                s / lane.len() as f32
            }
        }
        // Start at -inf so all-negative lanes still produce their max.
        "max" => lane.iter().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v)),
        "amax" => lane.iter().fold(0.0f32, |acc, &v| acc.max(v.abs())),
        other => {
            return Err(err(
                op,
                format!(
                    "unknown kind '{other}' for reduce; accepted values: {}",
                    REDUCE_KINDS.join(", ")
                ),
            ));
        }
    })
}

fn reduce_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "x")?;
    require_str_of(a, "kind", REDUCE_KINDS, c.op)?;
    let o = c.out_t(0);
    match attr_i64(a, "axis") {
        None => {
            // No axis: reduce everything into a rank-0 scalar.
            set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, &[]);
        }
        Some(axis) => {
            let rank = x.rank as usize;
            let ax = resolve_axis(axis, rank, c.op)?;
            let mut shape = SmallShape {
                len: rank - 1,
                dims: [0; rustrain_abi::ffi::MAX_RANK],
            };
            let mut it = 0usize;
            for d in 0..rank {
                if d != ax {
                    shape.dims[it] = x.shape[d];
                    it += 1;
                }
            }
            let o = c.out_t(0);
            set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, shape.as_slice());
        }
    }
    Ok(())
}

fn reduce_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let kind = require_str_of(a, "kind", REDUCE_KINDS, c.op)?;
    let out_shape: SmallShape = match attr_i64(a, "axis") {
        None => SmallShape {
            len: 0,
            dims: [0; rustrain_abi::ffi::MAX_RANK],
        },
        Some(axis) => {
            let rank = x.rank as usize;
            let ax = resolve_axis(axis, rank, c.op)?;
            let mut s = SmallShape {
                len: rank - 1,
                dims: [0; rustrain_abi::ffi::MAX_RANK],
            };
            let mut it = 0usize;
            for d in 0..rank {
                if d != ax {
                    s.dims[it] = x.shape[d];
                    it += 1;
                }
            }
            s
        }
    };
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, out_shape.as_slice())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    match attr_i64(a, "axis") {
        None => {
            // Reduce-all: iterate the whole buffer in order.
            let flat = xv.as_slice().ok_or_else(|| {
                err(c.op, "input is not contiguous in memory (unexpected)")
            })?;
            let lane = ArrayView1::from(flat);
            let v = reduce_lane(c.op, kind, lane)?;
            if let Some(first) = yv.iter_mut().next() {
                *first = v;
            }
        }
        Some(axis) => {
            let ax = Axis(resolve_axis(axis, x.rank as usize, c.op)?);
            // Lanes over the reduced axis come out in exactly the order of
            // y's elements (y is x with that axis removed), so a positional
            // zip is the pairing — and it fixes the iteration order.
            for (lane, o) in xv.lanes(ax).into_iter().zip(yv.iter_mut()) {
                *o = reduce_lane(c.op, kind, lane)?;
            }
        }
    }
    Ok(())
}

// ── softmax ─────────────────────────────────────────────────────────────────

fn softmax_scale(a: &RsAttrs) -> f64 {
    attr_f64(a, "scale").unwrap_or(1.0)
}

fn softmax_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "x")?;
    let rank = x.rank as usize;
    if rank == 0 {
        return Err(err(c.op, "softmax expects rank >= 1"));
    }
    resolve_axis(attr_i64(a, "axis").unwrap_or(-1), rank, c.op)?;
    let _ = softmax_scale(a);
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, x.dims());
    Ok(())
}

fn softmax_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let rank = x.rank as usize;
    let ax = Axis(resolve_axis(attr_i64(a, "axis").unwrap_or(-1), rank, c.op)?);
    let scale = softmax_scale(a) as f32;
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, x.dims())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    // Stable softmax: subtract the lane max before exponentiating, so a
    // large-magnitude lane (e.g. [1000, 1000, 0]) cannot overflow to inf/NaN.
    for (lane, mut out) in xv.lanes(ax).into_iter().zip(yv.lanes_mut(ax)) {
        let m = lane.iter().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
        let s = lane.iter().fold(0.0f32, |acc, &v| acc + ((v - m) * scale).exp());
        for (o, &v) in out.iter_mut().zip(lane.iter()) {
            *o = ((v - m) * scale).exp() / s;
        }
    }
    Ok(())
}

// ── rmsnorm ─────────────────────────────────────────────────────────────────

fn norm_eps(a: &RsAttrs) -> f64 {
    attr_f64(a, "eps").unwrap_or(1e-5)
}

fn rmsnorm_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 2))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "x")?;
    let rank = x.rank as usize;
    if rank < 1 {
        return Err(err(c.op, "rmsnorm expects rank >= 1"));
    }
    let d = x.dims()[rank - 1];
    if c.n_in() == 2 {
        let w = c.in_t(1);
        check_f32_desc(w, c.op, "w")?;
        if w.rank != 1 || w.shape[0] != d {
            return Err(fail!(
                c.op,
                "rmsnorm weight must be [D={d}], got {:?}",
                w.dims()
            ));
        }
    }
    let _ = norm_eps(a);
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, x.dims());
    Ok(())
}

fn rmsnorm_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 2))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    if x.rank < 1 {
        return Err(err(c.op, "rmsnorm expects rank >= 1"));
    }
    let rank = x.rank as usize;
    let eps = norm_eps(a) as f32;
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, x.dims())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let wv = if c.n_in() == 2 {
        // SAFETY: descriptor liveness is the ABI caller's contract.
        Some(unsafe { crate::tensor::f32_in(c.op, "w", c.in_t(1)) }?)
    } else {
        None
    };
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    // Convention (documented): y = x / sqrt(mean(x^2) + eps) * w, eps inside
    // the sqrt. mean(x^2) is accumulated in ascending order.
    let ax = Axis(rank - 1);
    for (lane, mut out) in xv.lanes(ax).into_iter().zip(yv.lanes_mut(ax)) {
        let ss = lane.iter().fold(0.0f32, |acc, &v| acc + v * v) / lane.len() as f32;
        let r = (ss + eps).sqrt();
        for (i, (o, &v)) in out.iter_mut().zip(lane.iter()).enumerate() {
            let n = v / r;
            *o = match &wv {
                Some(w) => n * w[i],
                None => n,
            };
        }
    }
    Ok(())
}

// ── layernorm ───────────────────────────────────────────────────────────────

fn layernorm_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 3))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "x")?;
    let rank = x.rank as usize;
    if rank < 1 {
        return Err(err(c.op, "layernorm expects rank >= 1"));
    }
    let d = x.dims()[rank - 1];
    if c.n_in() >= 2 {
        let w = c.in_t(1);
        check_f32_desc(w, c.op, "w")?;
        if w.rank != 1 || w.shape[0] != d {
            return Err(fail!(
                c.op,
                "layernorm weight must be [D={d}], got {:?}",
                w.dims()
            ));
        }
    }
    if c.n_in() == 3 {
        let b = c.in_t(2);
        check_f32_desc(b, c.op, "b")?;
        if b.rank != 1 || b.shape[0] != d {
            return Err(fail!(
                c.op,
                "layernorm bias must be [D={d}], got {:?}",
                b.dims()
            ));
        }
    }
    let _ = norm_eps(a);
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, x.dims());
    Ok(())
}

fn layernorm_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 3))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    if x.rank < 1 {
        return Err(err(c.op, "layernorm expects rank >= 1"));
    }
    let rank = x.rank as usize;
    let eps = norm_eps(a) as f32;
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, x.dims())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let wv = if c.n_in() >= 2 {
        Some(unsafe { crate::tensor::f32_in(c.op, "w", c.in_t(1)) }?)
    } else {
        None
    };
    let bv = if c.n_in() == 3 {
        Some(unsafe { crate::tensor::f32_in(c.op, "b", c.in_t(2)) }?)
    } else {
        None
    };
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    // Convention (documented): y = (x - mean) / sqrt(var + eps) * w + b with
    // the biased variance 1/D * sum((x - mean)^2), eps outside the sqrt.
    let ax = Axis(rank - 1);
    for (lane, mut out) in xv.lanes(ax).into_iter().zip(yv.lanes_mut(ax)) {
        let n = lane.len() as f32;
        let mean = lane.iter().fold(0.0f32, |acc, &v| acc + v) / n;
        let var = lane.iter().fold(0.0f32, |acc, &v| {
            let d = v - mean;
            acc + d * d
        }) / n;
        let r = (var + eps).sqrt();
        for (i, (o, &v)) in out.iter_mut().zip(lane.iter()).enumerate() {
            let mut y = (v - mean) / r;
            if let Some(w) = &wv {
                y *= w[i];
            }
            if let Some(b) = &bv {
                y += b[i];
            }
            *o = y;
        }
    }
    Ok(())
}

// ── rope ────────────────────────────────────────────────────────────────────

fn rope_infer_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((3, 3))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "x")?;
    let rank = x.rank as usize;
    if rank < 1 {
        return Err(err(c.op, "rope expects rank >= 1"));
    }
    let d = x.dims()[rank - 1];
    if d % 2 != 0 {
        return Err(err(
            c.op,
            format!("rope requires an even last dim (D={d}); the half-rotation pairs (2i, 2i+1)"),
        ));
    }
    let half = d / 2;
    let mut want = SmallShape {
        len: rank,
        dims: [0; rustrain_abi::ffi::MAX_RANK],
    };
    want.dims[..rank].copy_from_slice(x.dims());
    want.dims[rank - 1] = half;
    for (i, who) in [(1usize, "cos"), (2, "sin")] {
        let t = c.in_t(i);
        check_f32_desc(t, c.op, who)?;
        broadcast_shape_small(t.dims(), want.as_slice(), c.op).map_err(|e| {
            err(
                c.op,
                format!("input '{who}' must broadcast to {:?}: {e}", want.as_slice()),
            )
        })?;
    }
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, x.dims());
    Ok(())
}

fn rope_exec_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((3, 3))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    if x.rank < 1 {
        return Err(err(c.op, "rope expects rank >= 1"));
    }
    let rank = x.rank as usize;
    let d = x.dims()[rank - 1] as usize;
    if d % 2 != 0 {
        return Err(err(
            c.op,
            format!("rope requires an even last dim (D={d})"),
        ));
    }
    let half = d / 2;
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, x.dims())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let cos = unsafe { crate::tensor::f32_in(c.op, "cos", c.in_t(1)) }?;
    let sin = unsafe { crate::tensor::f32_in(c.op, "sin", c.in_t(2)) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    // Target shape for cos/sin: x.shape with the last dim halved.
    let mut want: Vec<usize> = xv.shape().to_vec();
    want[rank - 1] = half;
    let cb = cos
        .broadcast(IxDyn(&want))
        .ok_or_else(|| err(c.op, "cos does not broadcast to the halved shape"))?;
    let sb = sin
        .broadcast(IxDyn(&want))
        .ok_or_else(|| err(c.op, "sin does not broadcast to the halved shape"))?;
    let cd: ArrayViewD<f32> = cb;
    let sd: ArrayViewD<f32> = sb;
    // NeoX/GPT-J half rotation (documented): for each pair (2i, 2i+1),
    //   y[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
    //   y[2i+1] = x[2i+1] * cos[i] + x[2i] * sin[i]
    // cos/sin are caller-computed tables; the op only rotates, which keeps it
    // convention-free about theta bases and position encodings.
    // Fixed-size index buffers: the hot loop must not allocate (the memory
    // reporter would have to account for it, and per-element allocs are
    // exactly the kind of hidden workspace the planning rule forbids).
    for ((idx, o), &v) in yv.indexed_iter_mut().zip(xv.iter()) {
        let idx = idx.slice();
        let rank = idx.len();
        let mut cidx = [0usize; rustrain_abi::ffi::MAX_RANK];
        cidx[..rank].copy_from_slice(idx);
        let last = cidx[rank - 1];
        let i = last / 2;
        cidx[rank - 1] = i;
        let c = cd[IxDyn(&cidx[..rank])];
        let s = sd[IxDyn(&cidx[..rank])];
        let (other, sign_even) = if last % 2 == 0 {
            // even element: paired with the odd element to its right
            let mut oidx = cidx;
            oidx[rank - 1] = last + 1;
            (xv[IxDyn(&oidx[..rank])], true)
        } else {
            let mut oidx = cidx;
            oidx[rank - 1] = last - 1;
            (xv[IxDyn(&oidx[..rank])], false)
        };
        *o = if sign_even { v * c - other * s } else { v * c + other * s };
    }
    Ok(())
}

infer_entry!(matmul_infer, "matmul", matmul_infer_body);
exec_entry!(matmul_exec, "matmul", matmul_exec_body);
infer_entry!(linear_infer, "linear", linear_infer_body);
exec_entry!(linear_exec, "linear", linear_exec_body);
infer_entry!(bmm_infer, "bmm", bmm_infer_body);
exec_entry!(bmm_exec, "bmm", bmm_exec_body);
infer_entry!(unary_infer, "elementwise_unary", unary_infer_body);
exec_entry!(unary_exec, "elementwise_unary", unary_exec_body);
infer_entry!(binary_infer, "elementwise_binary", binary_infer_body);
exec_entry!(binary_exec, "elementwise_binary", binary_exec_body);
infer_entry!(reduce_infer, "reduce", reduce_infer_body);
exec_entry!(reduce_exec, "reduce", reduce_exec_body);
infer_entry!(softmax_infer, "softmax", softmax_infer_body);
exec_entry!(softmax_exec, "softmax", softmax_exec_body);
infer_entry!(rmsnorm_infer, "rmsnorm", rmsnorm_infer_body);
exec_entry!(rmsnorm_exec, "rmsnorm", rmsnorm_exec_body);
infer_entry!(layernorm_infer, "layernorm", layernorm_infer_body);
exec_entry!(layernorm_exec, "layernorm", layernorm_exec_body);
infer_entry!(rope_infer, "rope", rope_infer_body);
exec_entry!(rope_exec, "rope", rope_exec_body);
