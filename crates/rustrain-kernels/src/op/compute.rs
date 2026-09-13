//! L0 compute operators: `matmul`, `linear`, `bmm`, `elementwise_unary`,
//! `elementwise_binary`, `compare`, `reduce`, `softmax`, `rmsnorm`,
//! `layernorm`, `rope`.
//!
//! Determinism rules used throughout (and why): every reduction accumulates
//! in ascending index order with plain f32 adds — no threads, no trees, no
//! `HashMap` iteration — so the same inputs twice produce bitwise-identical
//! outputs. This is the reference provider's whole reason to exist.

use ndarray::{ArrayView1, Axis, IxDyn, Zip};
use rustrain_abi::ffi::{RsAttrs, RsTensor};

use crate::attrs::{attr_bool, attr_f64, attr_i64, require_str_of, str_or};
use crate::dispatch::{Call, run};
use crate::error::{OpResult, err, fail};
use crate::tensor::{
    SmallShape, broadcast_shape_small, expect_dtype, expect_out, resolve_axis, set_output_desc,
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
fn matmul_plan(
    a: &RsTensor,
    b: &RsTensor,
    attrs: &RsAttrs,
    op: &'static str,
) -> OpResult<(i64, i64, i64, bool)> {
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
    expect_out(
        c.out_t(0),
        c.op,
        rustrain_abi::ffi::RsDtype::F32,
        shape.as_slice(),
    )?;
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
    let batches: usize = c.in_t(0).dims()[..r - 2]
        .iter()
        .map(|&d| d as usize)
        .product();
    let mut shape = SmallShape {
        len: r,
        dims: [0; rustrain_abi::ffi::MAX_RANK],
    };
    shape.dims[..r].copy_from_slice(c.in_t(0).dims());
    shape.dims[r - 1] = n as i64;
    expect_out(
        c.out_t(0),
        c.op,
        rustrain_abi::ffi::RsDtype::F32,
        shape.as_slice(),
    )?;
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
/// expressible in the fixed primitive vocabulary; `rsqrt` and the `*_grad`
/// kinds close the §2.11 vocabulary gaps for the backward pass; see the op
/// doc.
pub(crate) const UNARY_KINDS: &[&str] = &[
    "silu",
    "gelu",
    "sigmoid",
    "tanh",
    "relu",
    "exp",
    "log",
    "neg",
    "sqrt",
    "rsqrt",
    "softplus",
    "negative_exp",
    "silu_grad",
    "gelu_grad",
    "sigmoid_grad",
    "tanh_grad",
    "relu_grad",
];

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
        // 1/sqrt(x), written as the literal composition the doc promises
        // (IEEE: rsqrt(0) = +inf, rsqrt of a negative is NaN). Not the
        // separately-rounded rsqrt intrinsic, so the doc's "1/sqrt(x)" is
        // exactly what runs.
        "rsqrt" => |x: f32| 1.0 / x.sqrt(),
        // torch F.softplus with the default beta=1, threshold=20: the exact
        // identity for x > 20 (ln(1+e^x) would overflow its argument's
        // exponent), ln(1+e^x) otherwise. Qwen3.6's dt gate:
        // softplus(a + dt_bias).
        "softplus" => |x: f32| {
            if x > 20.0 { x } else { x.exp().ln_1p() }
        },
        // -exp(x), spelled the way the Qwen3.6 description declares it
        // (`elementwise_unary(kind: negative_exp)` on A_log); the composition
        // neg(exp(x)) would need two nodes for the same math.
        "negative_exp" => |x: f32| -x.exp(),
        // silu(x) = x*σ(x) ⇒ silu' = σ(x)*(1 + x*(1 - σ(x))). σ comes from
        // the same 1/(1+e^-x) as the forward silu, so a central finite
        // difference of the forward op matches to rounding error.
        "silu_grad" => |x: f32| {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 + x * (1.0 - s))
        },
        // Derivative of the tanh-approximation gelu above (NOT the erf form):
        // with c = s*(x + a*x^3), f = x/2*(1 + tanh c),
        // f' = (1 + tanh c)/2 + x/2*(1 - tanh^2 c)*dc/dx. Same constants s
        // and a as the forward, so the finite-difference check is exact by
        // construction.
        "gelu_grad" => |x: f32| {
            let s = std::f32::consts::FRAC_2_PI.sqrt();
            let c = s * (x + 0.044_715 * x.powi(3));
            let t = c.tanh();
            let dcdx = s * (1.0 + 3.0 * 0.044_715 * x * x);
            0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dcdx
        },
        // sigmoid' = σ(1-σ).
        "sigmoid_grad" => |x: f32| {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 - s)
        },
        // tanh' = 1 - tanh^2.
        "tanh_grad" => |x: f32| {
            let t = x.tanh();
            1.0 - t * t
        },
        // Subgradient convention at x = 0 (the derivative is undefined
        // there): 0. The strict '>' follows IEEE — NaN compares false, so
        // relu_grad(NaN) = 0, matching relu(NaN) = 0 via f32::max.
        "relu_grad" => |x: f32| {
            if x > 0.0 { 1.0 } else { 0.0 }
        },
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

pub(crate) const BINARY_KINDS: &[&str] = &["add", "sub", "mul", "div", "maximum", "pow"];

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
        // IEEE powf: 0^0 = 1, a negative base with a fractional exponent is
        // NaN. libm's powf is deterministic on the reference target (no
        // fast-math variance to worry about).
        "pow" => f32::powf,
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
    expect_out(
        c.out_t(0),
        c.op,
        rustrain_abi::ffi::RsDtype::F32,
        out_shape.as_slice(),
    )?;
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
        let dims: Vec<usize> = out_shape.as_slice().iter().map(|&d| d as usize).collect();
        let ab = xv
            .broadcast(IxDyn(&dims))
            .ok_or_else(|| err(c.op, "broadcast of 'a' failed unexpectedly"))?;
        let bb = bv
            .broadcast(IxDyn(&dims))
            .ok_or_else(|| err(c.op, "broadcast of 'b' failed unexpectedly"))?;
        // Zip iterates in the fixed logical order — deterministic.
        Zip::from(&ab)
            .and(&bb)
            .and(&mut yv)
            .for_each(|&a, &b, y| *y = f(a, b));
    }
    Ok(())
}

// ── compare ─────────────────────────────────────────────────────────────────

pub(crate) const COMPARE_KINDS: &[&str] = &["eq", "ne", "lt", "le", "gt", "ge"];

/// The comparison as an f32 mask: exactly 1.0 where it holds, 0.0 elsewhere.
/// NaN policy (documented in the op doc): any NaN operand yields 0.0 for
/// every kind. For eq/lt/le/gt/ge that is IEEE itself (comparisons with NaN
/// are false); 'ne' deliberately follows suit — C's `NaN != x` would be true,
/// but a true result would smuggle a 1 into a mask through a NaN, so 'ne' is
/// the logical negation of 'eq' instead.
fn compare_fn(op: &'static str, kind: &str) -> OpResult<fn(f32, f32) -> f32> {
    Ok(match kind {
        "eq" => |a, b| (a == b) as u32 as f32,
        "ne" => |a, b| (a != b && !a.is_nan() && !b.is_nan()) as u32 as f32,
        "lt" => |a, b| (a < b) as u32 as f32,
        "le" => |a, b| (a <= b) as u32 as f32,
        "gt" => |a, b| (a > b) as u32 as f32,
        "ge" => |a, b| (a >= b) as u32 as f32,
        other => {
            return Err(err(
                op,
                format!(
                    "unknown kind '{other}' for compare; accepted values: {}",
                    COMPARE_KINDS.join(", ")
                ),
            ));
        }
    })
}

/// Shared compare validation: both inputs must be f32 with identical shapes.
fn compare_plan(x: &RsTensor, b: &RsTensor, op: &'static str) -> OpResult<()> {
    check_f32_desc(x, op, "a")?;
    check_f32_desc(b, op, "b")?;
    if x.dims() != b.dims() {
        return Err(fail!(
            op,
            "compare expects both inputs to have the same shape and dtype (f32), \
             got a {:?} and b {:?}",
            x.dims(),
            b.dims()
        ));
    }
    Ok(())
}

fn compare_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    require_str_of(a, "kind", COMPARE_KINDS, c.op)?;
    compare_plan(x, c.in_t(1), c.op)?;
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, x.dims());
    Ok(())
}

fn compare_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let kind = require_str_of(a, "kind", COMPARE_KINDS, c.op)?;
    let f = compare_fn(c.op, kind)?;
    compare_plan(x, c.in_t(1), c.op)?;
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, x.dims())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let av = unsafe { crate::tensor::f32_in(c.op, "a", x) }?;
    let bv = unsafe { crate::tensor::f32_in(c.op, "b", c.in_t(1)) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    // Equal shapes means the plain iterators stay paired in the same fixed
    // logical order — no broadcast machinery, no allocation.
    for ((o, &a), &b) in yv.iter_mut().zip(av.iter()).zip(bv.iter()) {
        *o = f(a, b);
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

/// Shared reduce output shape for infer and execute. With an 'axis', the
/// reduced axis is removed — unless 'keepdim' (default false) keeps it as a
/// size-1 dim, which the softmax/layernorm VJPs need. With no 'axis', the
/// output is a rank-0 scalar, or all-ones when keepdim is set (torch
/// convention). Pure and allocation-free.
fn reduce_out_shape(x: &RsTensor, attrs: &RsAttrs, op: &'static str) -> OpResult<SmallShape> {
    let keepdim = attr_bool(attrs, "keepdim").unwrap_or(false);
    let rank = x.rank as usize;
    match attr_i64(attrs, "axis") {
        None => {
            let mut s = SmallShape {
                len: if keepdim { rank } else { 0 },
                dims: [0; rustrain_abi::ffi::MAX_RANK],
            };
            if keepdim {
                // Every dim survives as size 1.
                s.dims[..rank].fill(1);
            }
            Ok(s)
        }
        Some(axis) => {
            let ax = resolve_axis(axis, rank, op)?;
            let mut s = SmallShape {
                len: if keepdim { rank } else { rank - 1 },
                dims: [0; rustrain_abi::ffi::MAX_RANK],
            };
            if keepdim {
                s.dims[..rank].copy_from_slice(x.dims());
                s.dims[ax] = 1;
            } else {
                let mut it = 0usize;
                for d in 0..rank {
                    if d != ax {
                        s.dims[it] = x.shape[d];
                        it += 1;
                    }
                }
            }
            Ok(s)
        }
    }
}

fn reduce_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "x")?;
    require_str_of(a, "kind", REDUCE_KINDS, c.op)?;
    let shape = reduce_out_shape(x, a, c.op)?;
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, shape.as_slice());
    Ok(())
}

fn reduce_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let kind = require_str_of(a, "kind", REDUCE_KINDS, c.op)?;
    let out_shape = reduce_out_shape(x, a, c.op)?;
    expect_out(
        c.out_t(0),
        c.op,
        rustrain_abi::ffi::RsDtype::F32,
        out_shape.as_slice(),
    )?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    match attr_i64(a, "axis") {
        None => {
            // Reduce-all: iterate the whole buffer in order. Works for both
            // the rank-0 scalar shape and the all-ones keepdim shape (both
            // hold exactly one element).
            let flat = xv
                .as_slice()
                .ok_or_else(|| err(c.op, "input is not contiguous in memory (unexpected)"))?;
            let lane = ArrayView1::from(flat);
            let v = reduce_lane(c.op, kind, lane)?;
            if let Some(first) = yv.iter_mut().next() {
                *first = v;
            }
        }
        Some(axis) => {
            let ax = Axis(resolve_axis(axis, x.rank as usize, c.op)?);
            // Lanes over the reduced axis come out in exactly the order of
            // y's elements (y is x with that axis removed, or kept as size 1
            // under keepdim), so a positional zip is the pairing — and it
            // fixes the iteration order.
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
        let s = lane
            .iter()
            .fold(0.0f32, |acc, &v| acc + ((v - m) * scale).exp());
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
    // Weight convention (declared, D5): y = n * (w + weight_offset). The
    // trunk uses offset 1.0 (1 + weight); GDN's gated normalisation uses the
    // raw weight (offset 0.0, the default — the pre-D5 behaviour). Without a
    // weight input the offset has nothing to add to and is ignored.
    let offset = attr_f64(a, "weight_offset").unwrap_or(0.0) as f32;
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
    // Convention (documented): y = x / sqrt(mean(x^2) + eps) * (w + offset),
    // eps inside the sqrt. mean(x^2) is accumulated in ascending order.
    let ax = Axis(rank - 1);
    for (lane, mut out) in xv.lanes(ax).into_iter().zip(yv.lanes_mut(ax)) {
        let ss = lane.iter().fold(0.0f32, |acc, &v| acc + v * v) / lane.len() as f32;
        let r = (ss + eps).sqrt();
        for (i, (o, &v)) in out.iter_mut().zip(lane.iter()).enumerate() {
            let n = v / r;
            *o = match &wv {
                Some(w) => n * (w[i] + offset),
                None => n,
            };
        }
    }
    Ok(())
}

// ── l2norm ──────────────────────────────────────────────────────────────────

/// `y = x * rsqrt(sum(x^2, dim) + eps)`: the L2 normalisation GDN applies to
/// q and k (HF `l2norm` in `modeling_qwen3_5_moe.py`, aligned with FLA: the
/// SUM of squares — not the mean — with eps added *inside* the rsqrt).
/// No learnable parameter; the delta rule's 1/sqrt(D) query scaling is a
/// separate, always-applied convention of `gated_delta_rule` itself.
fn l2norm_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "x")?;
    let rank = x.rank as usize;
    if rank < 1 {
        return Err(err(c.op, "l2norm expects rank >= 1"));
    }
    resolve_axis(attr_i64(a, "dim").unwrap_or(-1), rank, c.op)?;
    let _ = attr_f64(a, "eps").unwrap_or(1e-6);
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, x.dims());
    Ok(())
}

fn l2norm_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "x")?;
    let rank = x.rank as usize;
    if rank < 1 {
        return Err(err(c.op, "l2norm expects rank >= 1"));
    }
    let eps = attr_f64(a, "eps").unwrap_or(1e-6) as f32;
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, x.dims())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    // Convention (documented): y = x / sqrt(sum(x^2) + eps), the sum (not the
    // mean) accumulated in ascending order, eps inside the sqrt.
    let ax = Axis(resolve_axis(attr_i64(a, "dim").unwrap_or(-1), rank, c.op)?);
    for (lane, mut out) in xv.lanes(ax).into_iter().zip(yv.lanes_mut(ax)) {
        let ss = lane.iter().fold(0.0f32, |acc, &v| acc + v * v);
        let r = (ss + eps).sqrt();
        for (o, &v) in out.iter_mut().zip(lane.iter()) {
            *o = v / r;
        }
    }
    Ok(())
}

// ── rmsnorm_gated ───────────────────────────────────────────────────────────

/// GDN's output normalisation (HF `Qwen3_5MoeRMSNormGated`): the reduce and
/// the gating run in ONE pass over the data. `x` is treated as `[rows, D]`
/// rows (D = w.len(), any leading layout whose numel is a multiple of D —
/// HF reshapes the v-heads flat before the norm), and
/// `y = (x / sqrt(mean(x^2) + eps)) * (w + weight_offset) * silu(gate)`.
/// Weight convention: RAW weight (offset 0.0), unlike the trunk's 1 + weight.
fn rmsnorm_gated_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((3, 3))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "x")?;
    let rank = x.rank as usize;
    if rank < 1 {
        return Err(err(c.op, "rmsnorm_gated expects rank >= 1"));
    }
    let d = x.dims()[rank - 1];
    if d <= 0 {
        return Err(err(c.op, "rmsnorm_gated expects a positive last dim"));
    }
    let w = c.in_t(1);
    check_f32_desc(w, c.op, "w")?;
    if w.rank != 1 || w.shape[0] != d {
        return Err(fail!(
            c.op,
            "rmsnorm_gated weight must be [D={d}], got {:?}",
            w.dims()
        ));
    }
    let gate = c.in_t(2);
    check_f32_desc(gate, c.op, "gate")?;
    if gate.numel() != x.numel() {
        return Err(fail!(
            c.op,
            "rmsnorm_gated gate must have the same element count as x ({}), got {} \
             — the gate is applied elementwise after the row normalisation",
            x.numel(),
            gate.numel()
        ));
    }
    str_or(a, "gate_act", "silu", &["silu"], c.op)?;
    let _ = attr_f64(a, "eps").unwrap_or(1e-6);
    let _ = attr_f64(a, "weight_offset").unwrap_or(0.0);
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, x.dims());
    Ok(())
}

fn rmsnorm_gated_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((3, 3))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "x")?;
    let rank = x.rank as usize;
    if rank < 1 {
        return Err(err(c.op, "rmsnorm_gated expects rank >= 1"));
    }
    let d = x.dims()[rank - 1] as usize;
    let eps = attr_f64(a, "eps").unwrap_or(1e-6) as f32;
    let offset = attr_f64(a, "weight_offset").unwrap_or(0.0) as f32;
    str_or(a, "gate_act", "silu", &["silu"], c.op)?;
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, x.dims())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let wv = unsafe { crate::tensor::f32_in(c.op, "w", c.in_t(1)) }?;
    let gv = unsafe { crate::tensor::f32_in(c.op, "gate", c.in_t(2)) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let xs = xv.as_slice().expect("contiguous");
    let gs = gv.as_slice().expect("contiguous");
    let ys = yv.as_slice_mut().expect("contiguous");
    // One reduce pass (documented): rows of D, mean(x^2) ascending, then
    // weight (raw + offset) and the silu gate in the same element sweep.
    // silu(t) = t / (1 + e^-t): the x/(1+e^-x) form is exact for both tails.
    let rows = xs.len() / d;
    for r in 0..rows {
        let row = &xs[r * d..(r + 1) * d];
        let ss = row.iter().fold(0.0f32, |acc, &v| acc + v * v) / d as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        for i in 0..d {
            let n = row[i] * inv;
            let g = gs[r * d + i];
            ys[r * d + i] = n * (wv[i] + offset) * (g / (1.0 + (-g).exp()));
        }
    }
    Ok(())
}

// ── causal_conv1d ───────────────────────────────────────────────────────────

/// Qwen3.6's depthwise causal sequence convolution (HF `causal_conv1d_fn`):
/// out[t, c] = sum_k w[c, 0, k] * x[t + k - pad, c], with x[j < 0] = 0 and
/// an optional fused silu. `pad` defaults to kernel - 1, which makes the
/// window strictly causal (taps t-kernel+1..t only). Input layout is
/// [..., L, C] (sequence first, channels last); the weight is [C, 1, K].
fn causal_conv1d_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "x")?;
    let rank = x.rank as usize;
    if rank < 2 {
        return Err(err(c.op, "causal_conv1d expects rank >= 2 ([..., L, C])"));
    }
    let channels = x.dims()[rank - 1];
    let w = c.in_t(1);
    check_f32_desc(w, c.op, "w")?;
    if w.rank != 3 || w.shape[1] != 1 || w.shape[0] != channels {
        return Err(fail!(
            c.op,
            "causal_conv1d weight must be [C={channels}, 1, K] (depthwise), got {:?}",
            w.dims()
        ));
    }
    let k = w.shape[2];
    let kernel = attr_i64(a, "kernel").unwrap_or(k);
    if kernel != k {
        return Err(fail!(
            c.op,
            "attribute 'kernel' ({kernel}) must equal the weight's tap count K={k} — \
             the declared kernel size is the checkpoint's own, never a silent truncation"
        ));
    }
    str_or(a, "groups", "channels", &["channels"], c.op)?;
    let act = str_or(a, "activation", "", &["", "silu"], c.op)?;
    let pad = attr_i64(a, "pad").unwrap_or(k - 1);
    if pad < 0 || pad >= k {
        return Err(fail!(
            c.op,
            "attribute 'pad' ({pad}) must be in [0, K={k}); the default K-1 makes the \
             window strictly causal"
        ));
    }
    let _ = act;
    let o = c.out_t(0);
    set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, x.dims());
    Ok(())
}

fn causal_conv1d_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_f32_desc(x, c.op, "x")?;
    let rank = x.rank as usize;
    if rank < 2 {
        return Err(err(c.op, "causal_conv1d expects rank >= 2 ([..., L, C])"));
    }
    let (l, channels) = (x.dims()[rank - 2] as usize, x.dims()[rank - 1] as usize);
    let w = c.in_t(1);
    if w.rank != 3 || w.shape[1] != 1 || w.shape[0] != channels as i64 {
        return Err(fail!(
            c.op,
            "causal_conv1d weight must be [C={channels}, 1, K] (depthwise), got {:?}",
            w.dims()
        ));
    }
    let k = w.shape[2] as usize;
    let kernel = attr_i64(a, "kernel").unwrap_or(k as i64);
    if kernel != k as i64 {
        return Err(fail!(
            c.op,
            "attribute 'kernel' ({kernel}) must equal the weight's tap count K={k}"
        ));
    }
    str_or(a, "groups", "channels", &["channels"], c.op)?;
    let act = str_or(a, "activation", "", &["", "silu"], c.op)?;
    let pad = attr_i64(a, "pad").unwrap_or(k as i64 - 1);
    if pad < 0 || pad >= k as i64 {
        return Err(fail!(
            c.op,
            "attribute 'pad' ({pad}) must be in [0, K={k}); the default K-1 makes the \
             window strictly causal"
        ));
    }
    expect_out(c.out_t(0), c.op, rustrain_abi::ffi::RsDtype::F32, x.dims())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let wv = unsafe { crate::tensor::f32_in(c.op, "w", w) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let xs = xv.as_slice().expect("contiguous");
    let ws = wv.as_slice().expect("contiguous");
    let ys = yv.as_slice_mut().expect("contiguous");
    let batches = xs.len() / (l * channels);
    // Fixed ascending order (documented): batch, channel, time, tap k.
    // out[t] = sum_k w[c, 0, k] * x[t + k - pad], x[j < 0] = 0 — the
    // cross-correlation of F.conv1d with `pad` zeros on the left, cropped to
    // the original length (HF causal_conv1d_fn, bias-less).
    for b in 0..batches {
        let xb = &xs[b * l * channels..(b + 1) * l * channels];
        let yb = &mut ys[b * l * channels..(b + 1) * l * channels];
        for c in 0..channels {
            let taps = &ws[c * k..(c + 1) * k];
            for t in 0..l {
                let mut acc = 0.0f32;
                for (kk, &wk) in taps.iter().enumerate() {
                    let src = t as i64 + kk as i64 - pad;
                    if (0..l as i64).contains(&src) {
                        acc += wk * xb[src as usize * channels + c];
                    }
                }
                yb[t * channels + c] = if act == "silu" {
                    acc / (1.0 + (-acc).exp())
                } else {
                    acc
                };
            }
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

/// Shared rope shape plan for infer and execute. Returns
/// (rank, s, d, rotary_dim) and validates both operands and the optional
/// position tensor.
fn rope_plan(
    x: &RsTensor,
    y: &RsTensor,
    pos: Option<&RsTensor>,
    attrs: &RsAttrs,
    op: &'static str,
) -> OpResult<(usize, i64, i64, i64)> {
    check_f32_desc(x, op, "x")?;
    check_f32_desc(y, op, "y")?;
    let rank = x.rank as usize;
    if rank < 2 {
        return Err(err(
            op,
            "rope expects rank >= 2 inputs; positions vary along the FIRST axis (the \
             plan's sequence-first layout: [seq, ...])",
        ));
    }
    if y.rank as usize != rank || x.dims() != y.dims() {
        return Err(fail!(
            op,
            "rope expects x and y with identical shape, got {:?} and {:?}",
            x.dims(),
            y.dims()
        ));
    }
    // Sequence-first convention (the plan's squeezed-batch layout): S is the
    // first axis; the rotary block is the prefix of the last axis (per head).
    let (s, d) = (x.dims()[0], x.dims()[rank - 1]);
    if let Some(p) = pos {
        if p.numel() != s {
            return Err(fail!(
                op,
                "rope position tensor must hold one entry per position (S={s}), got {}",
                p.numel()
            ));
        }
        match p.dtype {
            rustrain_abi::ffi::RsDtype::F32
            | rustrain_abi::ffi::RsDtype::I32
            | rustrain_abi::ffi::RsDtype::I64 => {}
            other => {
                return Err(err(
                    op,
                    format!("rope position tensor has dtype {other}, expected f32, i32 or i64"),
                ));
            }
        }
    }
    let rotary = attr_i64(attrs, "rotary_dim").unwrap_or(d);
    if rotary <= 0 || rotary > d || rotary % 2 != 0 {
        return Err(fail!(
            op,
            "attribute 'rotary_dim' ({rotary}) must be a positive even number <= D={d}"
        ));
    }
    let partial = attr_bool(attrs, "partial_rotary").unwrap_or(false);
    if !partial && rotary != d {
        return Err(fail!(
            op,
            "attribute 'rotary_dim' ({rotary}) is smaller than D={d} but 'partial_rotary' \
             is not set — declare partial_rotary=true to rotate only the first rotary_dim \
             dims and pass the rest through"
        ));
    }
    let _ = attr_f64(attrs, "theta").unwrap_or(1e7);
    Ok((rank, s, d, rotary))
}

fn rope_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 3))?;
    c.expect_out_count(2)?;
    let pos = (c.n_in() == 3).then(|| c.in_t(2));
    let (rank, _s, _d, _rotary) = rope_plan(c.in_t(0), c.in_t(1), pos, a, c.op)?;
    let shape = SmallShape::of(c.in_t(0));
    for i in 0..2 {
        let o = c.out_t(i);
        set_output_desc(o, rustrain_abi::ffi::RsDtype::F32, shape.as_slice());
        let _ = rank;
    }
    Ok(())
}

/// Reads the optional position tensor as f64 positions, validating length S.
/// Absent → arange(S) (causal positions from 0 — the documented default).
fn rope_positions(p: Option<&RsTensor>, s: usize, op: &'static str) -> OpResult<Vec<f64>> {
    let Some(t) = p else {
        return Ok((0..s).map(|i| i as f64).collect());
    };
    crate::tensor::expect_contiguous(t, op, "pos")?;
    if t.numel() as usize != s {
        return Err(fail!(
            op,
            "rope position tensor must hold one entry per position (S={s}), got {}",
            t.numel()
        ));
    }
    match t.dtype {
        rustrain_abi::ffi::RsDtype::F32 => {
            // SAFETY: descriptor liveness is the ABI caller's contract.
            let v = unsafe { crate::tensor::f32_in(op, "pos", t) }?;
            Ok(v.as_slice()
                .expect("contiguous")
                .iter()
                .map(|&f| f as f64)
                .collect())
        }
        rustrain_abi::ffi::RsDtype::I32 | rustrain_abi::ffi::RsDtype::I64 => {
            // SAFETY: descriptor liveness is the ABI caller's contract.
            let v = unsafe { crate::tensor::indices_i64(op, "pos", t) }?;
            Ok(v.iter().map(|&i| i as f64).collect())
        }
        other => Err(err(
            op,
            format!("rope position tensor has dtype {other}, expected f32, i32 or i64"),
        )),
    }
}

fn rope_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 3))?;
    c.expect_out_count(2)?;
    let pos = (c.n_in() == 3).then(|| c.in_t(2));
    let (rank, s, d, rotary) = rope_plan(c.in_t(0), c.in_t(1), pos, a, c.op)?;
    let theta = attr_f64(a, "theta").unwrap_or(1e7);
    let shape = SmallShape::of(c.in_t(0));
    for i in 0..2 {
        expect_out(
            c.out_t(i),
            c.op,
            rustrain_abi::ffi::RsDtype::F32,
            shape.as_slice(),
        )?;
    }
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", c.in_t(0)) }?;
    let yv = unsafe { crate::tensor::f32_in(c.op, "y", c.in_t(1)) }?;
    let mut out0 = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let mut out1 = unsafe { crate::tensor::f32_out(c.op, c.out_t(1)) }?;
    let positions = rope_positions(pos, s as usize, c.op)?;
    let (s, d, rotary) = (s as usize, d as usize, rotary as usize);
    let h = rotary / 2;
    // inv_freq[j] = theta^(-2j / rotary_dim), the Qwen3.6 text convention
    // (compute_default_rope_parameters, float32 like the HF source).
    let mut inv_freq = vec![0.0f32; h];
    for (j, f) in inv_freq.iter_mut().enumerate() {
        *f = (theta as f32).powf(-((2 * j) as f32) / rotary as f32);
    }
    // cos/sin per (position t, pair j), computed once and reused for both
    // operands (the two tensors share the position table).
    let mut cos = vec![0.0f32; s * h];
    let mut sin = vec![0.0f32; s * h];
    for (t, &p) in positions.iter().enumerate() {
        for (j, &f) in inv_freq.iter().enumerate() {
            cos[t * h + j] = (p * f as f64).cos() as f32;
            sin[t * h + j] = (p * f as f64).sin() as f32;
        }
    }
    let xs = xv.as_slice().expect("contiguous");
    let ys = yv.as_slice().expect("contiguous");
    let o0 = out0.as_slice_mut().expect("contiguous");
    let o1 = out1.as_slice_mut().expect("contiguous");
    let batches: usize = xs.len() / (s * d);
    // Half-split rotation over the first `rotary` dims of the last axis
    // (HF rotate_half: pairs (i, i+h) with h = rotary/2), pass-through for
    // the remaining dims. Deterministic ascending loops.
    for b in 0..batches {
        for t in 0..s {
            for j in 0..h {
                let c = cos[t * h + j];
                let sn = sin[t * h + j];
                let i = j; // the first half of the rotary block
                let base = (b * s + t) * d;
                let (x_a, x_b) = (xs[base + i], xs[base + i + h]);
                o0[base + i] = x_a * c - x_b * sn;
                o0[base + i + h] = x_b * c + x_a * sn;
                let (y_a, y_b) = (ys[base + i], ys[base + i + h]);
                o1[base + i] = y_a * c - y_b * sn;
                o1[base + i + h] = y_b * c + y_a * sn;
            }
            let base = (b * s + t) * d;
            o0[base + rotary..base + d].copy_from_slice(&xs[base + rotary..base + d]);
            o1[base + rotary..base + d].copy_from_slice(&ys[base + rotary..base + d]);
        }
    }
    let _ = rank;
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
infer_entry!(compare_infer, "compare", compare_infer_body);
exec_entry!(compare_exec, "compare", compare_exec_body);
infer_entry!(reduce_infer, "reduce", reduce_infer_body);
exec_entry!(reduce_exec, "reduce", reduce_exec_body);
infer_entry!(softmax_infer, "softmax", softmax_infer_body);
exec_entry!(softmax_exec, "softmax", softmax_exec_body);
infer_entry!(rmsnorm_infer, "rmsnorm", rmsnorm_infer_body);
exec_entry!(rmsnorm_exec, "rmsnorm", rmsnorm_exec_body);
infer_entry!(l2norm_infer, "l2norm", l2norm_infer_body);
exec_entry!(l2norm_exec, "l2norm", l2norm_exec_body);
infer_entry!(
    rmsnorm_gated_infer,
    "rmsnorm_gated",
    rmsnorm_gated_infer_body
);
exec_entry!(rmsnorm_gated_exec, "rmsnorm_gated", rmsnorm_gated_exec_body);
infer_entry!(
    causal_conv1d_infer,
    "causal_conv1d",
    causal_conv1d_infer_body
);
exec_entry!(causal_conv1d_exec, "causal_conv1d", causal_conv1d_exec_body);
infer_entry!(layernorm_infer, "layernorm", layernorm_infer_body);
exec_entry!(layernorm_exec, "layernorm", layernorm_exec_body);
infer_entry!(rope_infer, "rope", rope_infer_body);
exec_entry!(rope_exec, "rope", rope_exec_body);
