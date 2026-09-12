//! Data-movement operators: `embedding`, `gather`, `scatter`.
//!
//! Conventions chosen here (documented in the op docs too):
//!
//! * Index tensors are accepted as `i32` or `i64`; values are validated
//!   against the indexed dimension, and **negative indices are rejected**
//!   (no wrap-around — a reference backend must not guess).
//! * `gather` reads rank-1 indices `[K]` and replaces the `axis` dim (default
//!   `-1`, the last) of the input with `K`.
//! * `scatter` copies the input and then writes, in ascending `k` order, so
//!   on duplicate indices the **last writer wins**, deterministically.

use rustrain_abi::ffi::{MAX_RANK, RsAttrs, RsDtype, RsTensor};

use crate::attrs::attr_i64;
use crate::dispatch::{Call, run};
use crate::error::{OpResult, err, fail};
use crate::tensor::{SmallShape, expect_out, resolve_axis, set_output_desc};

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

fn check_index_dtype(t: &RsTensor, op: &'static str, who: &str) -> OpResult<()> {
    match t.dtype {
        RsDtype::I32 | RsDtype::I64 => Ok(()),
        other => Err(err(
            op,
            format!("input '{who}' has dtype {other}, expected i32 or i64"),
        )),
    }
}

// ── embedding ───────────────────────────────────────────────────────────────

fn embedding_infer_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let w = c.in_t(0);
    let idx = c.in_t(1);
    if w.dtype != RsDtype::F32 {
        return Err(err(c.op, format!("input 'w' has dtype {}, expected f32", w.dtype)));
    }
    check_index_dtype(idx, c.op, "indices")?;
    if w.rank != 2 {
        return Err(err(
            c.op,
            format!("embedding expects weight [V, D], got rank {}", w.rank),
        ));
    }
    let r = idx.rank as usize;
    if r == 0 || r + 1 > MAX_RANK {
        return Err(err(
            c.op,
            format!("embedding indices rank {} produces an out-of-range output rank", idx.rank),
        ));
    }
    let mut shape = SmallShape {
        len: r + 1,
        dims: [0; MAX_RANK],
    };
    shape.dims[..r].copy_from_slice(idx.dims());
    shape.dims[r] = w.shape[1];
    let o = c.out_t(0);
    set_output_desc(o, RsDtype::F32, shape.as_slice());
    Ok(())
}

fn embedding_exec_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let w = c.in_t(0);
    let idx = c.in_t(1);
    let r = idx.rank as usize;
    let mut shape = SmallShape {
        len: r + 1,
        dims: [0; MAX_RANK],
    };
    shape.dims[..r].copy_from_slice(idx.dims());
    shape.dims[r] = w.shape[1];
    expect_out(c.out_t(0), c.op, RsDtype::F32, shape.as_slice())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let wv = unsafe { crate::tensor::f32_in(c.op, "w", w) }?;
    let ids = unsafe { crate::tensor::indices_i64(c.op, "indices", idx) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let (v, d) = (wv.shape()[0], wv.shape()[1]);
    for (n, &id) in ids.iter().enumerate() {
        if id < 0 || id >= v as i64 {
            return Err(fail!(
                c.op,
                "index {id} out of range [0, {v}) (negative indices are not wrapped)"
            ));
        }
        let row = id as usize;
        for j in 0..d {
            yv[n * d + j] = wv[row * d + j];
        }
    }
    Ok(())
}

// ── gather ──────────────────────────────────────────────────────────────────

fn gather_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let idx = c.in_t(1);
    if x.dtype != RsDtype::F32 {
        return Err(err(c.op, format!("input 'x' has dtype {}, expected f32", x.dtype)));
    }
    check_index_dtype(idx, c.op, "indices")?;
    let rank = x.rank as usize;
    if rank < 1 {
        return Err(err(c.op, "gather expects x with rank >= 1"));
    }
    if idx.rank != 1 {
        return Err(err(
            c.op,
            format!("gather indices must be rank 1 [K], got rank {}", idx.rank),
        ));
    }
    let ax = resolve_axis(attr_i64(a, "axis").unwrap_or(-1), rank, c.op)?;
    let k = idx.shape[0];
    let mut shape = SmallShape {
        len: rank,
        dims: [0; MAX_RANK],
    };
    shape.dims[..rank].copy_from_slice(x.dims());
    shape.dims[ax] = k;
    let o = c.out_t(0);
    set_output_desc(o, RsDtype::F32, shape.as_slice());
    Ok(())
}

fn gather_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let idx = c.in_t(1);
    let rank = x.rank as usize;
    let ax = resolve_axis(attr_i64(a, "axis").unwrap_or(-1), rank, c.op)?;
    let k = idx.shape[0] as usize;
    let mut shape = SmallShape {
        len: rank,
        dims: [0; MAX_RANK],
    };
    shape.dims[..rank].copy_from_slice(x.dims());
    shape.dims[ax] = idx.shape[0];
    expect_out(c.out_t(0), c.op, RsDtype::F32, shape.as_slice())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let ids = unsafe { crate::tensor::indices_i64(c.op, "indices", idx) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let dim_size = x.dims()[ax];
    // Linear-index arithmetic with the axis removed; loops ascend so the
    // order — and therefore the result — is fixed.
    let (outer, inner) = {
        let mut o = 1usize;
        for &d in &x.dims()[..ax] {
            o *= d as usize;
        }
        let mut i = 1usize;
        for &d in &x.dims()[ax + 1..] {
            i *= d as usize;
        }
        (o, i)
    };
    for oi in 0..outer {
        for kk in 0..k {
            let id = ids[kk];
            if id < 0 || id >= dim_size {
                return Err(fail!(
                    c.op,
                    "index {id} out of range [0, {dim_size}) along axis {ax} \
                     (negative indices are not wrapped)"
                ));
            }
            let src = (oi * dim_size as usize + id as usize) * inner;
            let dst = (oi * k + kk) * inner;
            for t in 0..inner {
                yv[dst + t] = xv[src + t];
            }
        }
    }
    Ok(())
}

// ── scatter ─────────────────────────────────────────────────────────────────

fn scatter_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((3, 3))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let idx = c.in_t(1);
    let values = c.in_t(2);
    if x.dtype != RsDtype::F32 || values.dtype != RsDtype::F32 {
        return Err(err(c.op, "scatter expects f32 'x' and 'values'"));
    }
    check_index_dtype(idx, c.op, "indices")?;
    let rank = x.rank as usize;
    if rank < 1 {
        return Err(err(c.op, "scatter expects x with rank >= 1"));
    }
    if values.dims() != x.dims() {
        return Err(err(
            c.op,
            format!(
                "scatter 'values' must have the same shape as 'x': {:?} vs {:?}",
                values.dims(),
                x.dims()
            ),
        ));
    }
    if idx.rank != 1 {
        return Err(err(
            c.op,
            format!("scatter indices must be rank 1 [K], got rank {}", idx.rank),
        ));
    }
    let ax = resolve_axis(attr_i64(a, "axis").unwrap_or(-1), rank, c.op)?;
    if idx.shape[0] != values.dims()[ax] {
        return Err(err(
            c.op,
            format!(
                "scatter indices length {} must equal values dim {ax} = {}",
                idx.shape[0],
                values.dims()[ax]
            ),
        ));
    }
    let o = c.out_t(0);
    set_output_desc(o, RsDtype::F32, x.dims());
    Ok(())
}

fn scatter_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((3, 3))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let idx = c.in_t(1);
    let values = c.in_t(2);
    let rank = x.rank as usize;
    let ax = resolve_axis(attr_i64(a, "axis").unwrap_or(-1), rank, c.op)?;
    expect_out(c.out_t(0), c.op, RsDtype::F32, x.dims())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let xv = unsafe { crate::tensor::f32_in(c.op, "x", x) }?;
    let ids = unsafe { crate::tensor::indices_i64(c.op, "indices", idx) }?;
    let vv = unsafe { crate::tensor::f32_in(c.op, "values", values) }?;
    let mut yv = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    // y starts as a copy of x, then k ascends: on duplicate indices the last
    // writer (largest k) wins — deterministic and documented.
    for (o, &v) in yv.iter_mut().zip(xv.iter()) {
        *o = v;
    }
    let dim_size = x.dims()[ax];
    let (outer, inner) = {
        let mut o = 1usize;
        for &d in &x.dims()[..ax] {
            o *= d as usize;
        }
        let mut i = 1usize;
        for &d in &x.dims()[ax + 1..] {
            i *= d as usize;
        }
        (o, i)
    };
    for oi in 0..outer {
        for (kk, &id) in ids.iter().enumerate() {
            if id < 0 || id >= dim_size {
                return Err(fail!(
                    c.op,
                    "index {id} out of range [0, {dim_size}) along axis {ax} \
                     (negative indices are not wrapped)"
                ));
            }
            let src = (oi * ids.len() + kk) * inner;
            let dst = (oi * dim_size as usize + id as usize) * inner;
            for t in 0..inner {
                yv[dst + t] = vv[src + t];
            }
        }
    }
    Ok(())
}

infer_entry!(embedding_infer, "embedding", embedding_infer_body);
exec_entry!(embedding_exec, "embedding", embedding_exec_body);
infer_entry!(gather_infer, "gather", gather_infer_body);
exec_entry!(gather_exec, "gather", gather_exec_body);
infer_entry!(scatter_infer, "scatter", scatter_infer_body);
exec_entry!(scatter_exec, "scatter", scatter_exec_body);
