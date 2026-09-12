//! Metadata / zero-copy view operators: `view`, `reshape`, `transpose`,
//! `narrow`, `cat`, `broadcast`.
//!
//! These do not compute. Except for `cat` (which concatenates and therefore
//! copies), `execute` only writes the output descriptor — `shape`, `stride`
//! and `data` — while aliasing the input buffer: `out.data = in.data`, no
//! allocation, no copy. `infer` declares the same shape/stride so plan
//! validation sees the truth before any execution.
//!
//! The reference provider is f32-only (declared in `requires`); the view ops
//! themselves are dtype-agnostic and copy the input dtype through, so they
//! remain valid if the provider is widened later.

use std::ptr;

use rustrain_abi::ffi::{MAX_RANK, RsAttrs, RsTensor};

use crate::attrs::{attr_i64, attr_i64s};
use crate::dispatch::{Call, run};
use crate::error::{OpResult, err, fail};
use crate::tensor::{SmallShape, expect_contiguous, resolve_axis, set_output_desc, set_view_desc};

/// Declares an `infer` trampoline for a body `fn(&mut Call, &RsAttrs)`.
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

fn check_rank(t: &RsTensor, op: &'static str, who: &str) -> OpResult<()> {
    if t.rank as usize > MAX_RANK {
        return Err(err(
            op,
            format!("input '{who}' has rank {}, exceeding MAX_RANK", t.rank),
        ));
    }
    Ok(())
}

// ── view ────────────────────────────────────────────────────────────────────

fn view_infer_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_rank(x, c.op, "x")?;
    let (dtype, rank, shape, stride) = (x.dtype, x.rank as usize, x.shape, x.stride);
    let o = c.out_t(0);
    set_view_desc(o, dtype, rank, &shape, &stride, ptr::null_mut());
    Ok(())
}

fn view_exec_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = *c.in_t(0);
    check_rank(&x, c.op, "x")?;
    // The whole point of a view: the output aliases the input buffer. No
    // allocation, no copy — the executor must not pre-allocate this output.
    *c.out_t(0) = x;
    Ok(())
}

// ── reshape ─────────────────────────────────────────────────────────────────

/// Resolves the `shape` attribute against the input: at most one `-1` is
/// allowed and it is filled so the element count is preserved.
fn resolve_reshape(x: &RsTensor, a: &RsAttrs, op: &'static str) -> OpResult<SmallShape> {
    let shape = attr_i64s(a, "shape").ok_or_else(|| {
        err(
            op,
            "attribute 'shape' (list of i64) is required for reshape",
        )
    })?;
    if shape.is_empty() || shape.len() > MAX_RANK {
        return Err(err(
            op,
            format!(
                "attribute 'shape' must have 1..={MAX_RANK} entries, got {}",
                shape.len()
            ),
        ));
    }
    let mut out = SmallShape {
        len: shape.len(),
        dims: [0; MAX_RANK],
    };
    out.dims[..shape.len()].copy_from_slice(shape);

    let given: i64 = shape.iter().filter(|&&d| d != -1).product();
    let negs = shape.iter().filter(|&&d| d == -1).count();
    if negs > 1 {
        return Err(err(op, "reshape allows at most one -1 dimension"));
    }
    if negs == 0 {
        if given != x.numel() {
            return Err(err(
                op,
                format!(
                    "reshape of {:?} ({} elements) to {:?} ({} elements) changes numel",
                    x.dims(),
                    x.numel(),
                    shape,
                    given
                ),
            ));
        }
        return Ok(out);
    }
    // One -1: derive it from the other dims. Zero-size reshape is ambiguous
    // when the unknown dim remains, so require a non-zero fixed product.
    if given == 0 {
        return Err(err(
            op,
            "reshape cannot infer the -1 dimension when the other dimensions multiply to zero",
        ));
    }
    let n = x.numel();
    if n % given != 0 {
        return Err(err(
            op,
            format!(
                "reshape of {:?} ({} elements) cannot fill -1: {} not divisible by {}",
                x.dims(),
                n,
                n,
                given
            ),
        ));
    }
    for d in out.dims.iter_mut().take(shape.len()) {
        if *d == -1 {
            *d = n / given;
        }
    }
    Ok(out)
}

fn reshape_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    check_rank(x, c.op, "x")?;
    let shape = resolve_reshape(x, a, c.op)?;
    let o = c.out_t(0);
    set_output_desc(o, x.dtype, shape.as_slice());
    Ok(())
}

fn reshape_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    // A reshape of a non-contiguous input would need a copy; the reference
    // provider is a view-only reshape, so it must be contiguous to alias.
    expect_contiguous(x, c.op, "x")?;
    let shape = resolve_reshape(x, a, c.op)?;
    let data = x.data;
    let dtype = x.dtype;
    let o = c.out_t(0);
    set_output_desc(o, dtype, shape.as_slice());
    o.data = data;
    Ok(())
}

// ── transpose ───────────────────────────────────────────────────────────────

fn transpose_plan(
    x: &RsTensor,
    a: &RsAttrs,
    op: &'static str,
) -> OpResult<(usize, [i64; MAX_RANK], [i64; MAX_RANK])> {
    let rank = x.rank as usize;
    check_rank(x, op, "x")?;
    let d0 = attr_i64(a, "dim0").unwrap_or(-2);
    let d1 = attr_i64(a, "dim1").unwrap_or(-1);
    let a0 = resolve_axis(d0, rank, op)?;
    let a1 = resolve_axis(d1, rank, op)?;
    if a0 == a1 {
        return Err(err(op, format!("transpose dims {d0} and {d1} are the same axis")));
    }
    let mut shape = x.shape;
    let mut stride = x.stride;
    shape.swap(a0, a1);
    stride.swap(a0, a1);
    Ok((rank, shape, stride))
}

fn transpose_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let (rank, shape, stride) = transpose_plan(x, a, c.op)?;
    let dtype = x.dtype;
    let o = c.out_t(0);
    set_view_desc(o, dtype, rank, &shape, &stride, ptr::null_mut());
    Ok(())
}

fn transpose_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let (rank, shape, stride) = transpose_plan(x, a, c.op)?;
    let (dtype, data) = (x.dtype, x.data);
    let o = c.out_t(0);
    set_view_desc(o, dtype, rank, &shape, &stride, data);
    Ok(())
}

// ── narrow ──────────────────────────────────────────────────────────────────

fn narrow_plan(
    x: &RsTensor,
    a: &RsAttrs,
    op: &'static str,
) -> OpResult<(usize, [i64; MAX_RANK], [i64; MAX_RANK], i64)> {
    let rank = x.rank as usize;
    check_rank(x, op, "x")?;
    let dim = resolve_axis(attr_i64(a, "dim").unwrap_or(-1), rank, op)?;
    let dsize = x.shape[dim];
    let start = attr_i64(a, "start").unwrap_or(0);
    let length = attr_i64(a, "length").unwrap_or(dsize - start);
    if start < 0 || start > dsize || length < 0 || start + length > dsize {
        return Err(fail!(
            op,
            "narrow dim {dim} of size {dsize} cannot hold [start={start}, length={length}]"
        ));
    }
    let mut shape = x.shape;
    shape[dim] = length;
    let offset = start * x.stride[dim];
    Ok((rank, shape, x.stride, offset))
}

fn narrow_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let (rank, shape, stride, _) = narrow_plan(x, a, c.op)?;
    let dtype = x.dtype;
    let o = c.out_t(0);
    set_view_desc(o, dtype, rank, &shape, &stride, ptr::null_mut());
    Ok(())
}

fn narrow_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let (rank, shape, stride, offset) = narrow_plan(x, a, c.op)?;
    let (dtype, data) = (x.dtype, x.data);
    // Element size: reference.f32 is f32-only, so offsets are in units of 4
    // bytes. If other dtypes are added, switch on byte_width().
    let byte_offset = offset * 4;
    let o = c.out_t(0);
    set_view_desc(o, dtype, rank, &shape, &stride, unsafe { data.add(byte_offset as usize) });
    Ok(())
}

// ── cat ─────────────────────────────────────────────────────────────────────

fn cat_dim(a: &RsAttrs, rank: usize, op: &'static str) -> OpResult<usize> {
    resolve_axis(attr_i64(a, "dim").unwrap_or(-1), rank, op)
}

fn cat_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, usize::MAX))?;
    c.expect_out_count(1)?;
    let first = c.in_t(0);
    check_rank(first, c.op, "in[0]")?;
    let rank = first.rank as usize;
    let dim = cat_dim(a, rank, c.op)?;
    let dtype = first.dtype;
    let mut shape = first.shape;
    shape[dim] = 0;
    for i in 0..c.n_in() {
        let t = c.in_t(i);
        if t.rank as usize != rank {
            return Err(fail!(
                c.op,
                "cat input {i} has rank {}, expected {rank}",
                t.rank
            ));
        }
        if t.dtype != dtype {
            return Err(fail!(
                c.op,
                "cat input {i} has dtype {}, expected {}",
                t.dtype,
                dtype
            ));
        }
        for d in 0..rank {
            if d == dim {
                continue;
            }
            if t.shape[d] != first.shape[d] {
                return Err(fail!(
                    c.op,
                    "cat input {i} has dim {d} = {}, expected {}",
                    t.shape[d],
                    first.shape[d]
                ));
            }
        }
        shape[dim] += t.shape[dim];
    }
    let o = c.out_t(0);
    set_output_desc(o, dtype, &shape[..rank]);
    Ok(())
}

fn cat_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, usize::MAX))?;
    c.expect_out_count(1)?;
    let first = c.in_t(0);
    let rank = first.rank as usize;
    let dim = cat_dim(a, rank, c.op)?;
    let mut out_shape = first.shape;
    out_shape[dim] = 0;
    for i in 0..c.n_in() {
        let t = c.in_t(i);
        if t.rank as usize != rank || t.dtype != first.dtype {
            return Err(fail!(c.op, "cat input {i} is inconsistent with input 0"));
        }
        for d in 0..rank {
            if d != dim && t.shape[d] != first.shape[d] {
                return Err(fail!(c.op, "cat input {i} is inconsistent with input 0"));
            }
        }
        out_shape[dim] += t.shape[dim];
    }
    // cat is the one meta op that copies: the concatenated buffer is new.
    crate::tensor::expect_out(
        c.out_t(0),
        c.op,
        first.dtype,
        &out_shape[..rank],
    )?;
    let mut out = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let mut offset = 0usize;
    for i in 0..c.n_in() {
        let t = c.in_t(i);
        // SAFETY: descriptor liveness is the ABI caller's contract.
        let xv = unsafe { crate::tensor::f32_in(c.op, "in", t) }?;
        let n = xv.len();
        let mut dst = out.slice_mut(ndarray::s![offset..offset + n]);
        for (d, &v) in dst.iter_mut().zip(xv.iter()) {
            *d = v;
        }
        offset += n;
    }
    Ok(())
}

// ── broadcast ───────────────────────────────────────────────────────────────

fn broadcast_plan(
    x: &RsTensor,
    a: &RsAttrs,
    op: &'static str,
) -> OpResult<(usize, [i64; MAX_RANK], [i64; MAX_RANK])> {
    let shape = attr_i64s(a, "shape").ok_or_else(|| {
        err(
            op,
            "attribute 'shape' (list of i64) is required for broadcast",
        )
    })?;
    if shape.is_empty() || shape.len() > MAX_RANK {
        return Err(err(
            op,
            format!(
                "attribute 'shape' must have 1..={MAX_RANK} entries, got {}",
                shape.len()
            ),
        ));
    }
    check_rank(x, op, "x")?;
    let r = shape.len();
    let xr = x.rank as usize;
    if xr > r {
        return Err(err(
            op,
            format!("cannot broadcast rank {xr} into rank {r} (target rank must be >= input rank)"),
        ));
    }
    let mut out = [0i64; MAX_RANK];
    let mut stride = [0i64; MAX_RANK];
    for i in 0..r {
        out[i] = shape[i];
        if out[i] < 0 {
            return Err(err(op, format!("broadcast shape contains a negative dim {}", out[i])));
        }
        // Right-aligned: input dim j maps to output dim i.
        let j = i as isize - (r as isize - xr as isize);
        if j < 0 {
            // A new leading dimension: the input implicitly repeats.
            stride[i] = 0;
        } else {
            let j = j as usize;
            let xd = x.shape[j];
            if xd == out[i] {
                stride[i] = x.stride[j];
            } else if xd == 1 {
                // Stretched dim: zero stride reads the same element repeatedly.
                stride[i] = 0;
            } else {
                return Err(fail!(
                    op,
                    "cannot broadcast dim {} of size {} to size {}",
                    j,
                    xd,
                    out[i]
                ));
            }
        }
    }
    Ok((r, out, stride))
}

fn broadcast_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let (rank, shape, stride) = broadcast_plan(x, a, c.op)?;
    let dtype = x.dtype;
    let o = c.out_t(0);
    set_view_desc(o, dtype, rank, &shape, &stride, ptr::null_mut());
    Ok(())
}

fn broadcast_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(1)?;
    let x = c.in_t(0);
    let (rank, shape, stride) = broadcast_plan(x, a, c.op)?;
    let (dtype, data) = (x.dtype, x.data);
    let o = c.out_t(0);
    set_view_desc(o, dtype, rank, &shape, &stride, data);
    Ok(())
}

infer_entry!(view_infer, "view", view_infer_body);
exec_entry!(view_exec, "view", view_exec_body);
infer_entry!(reshape_infer, "reshape", reshape_infer_body);
exec_entry!(reshape_exec, "reshape", reshape_exec_body);
infer_entry!(transpose_infer, "transpose", transpose_infer_body);
exec_entry!(transpose_exec, "transpose", transpose_exec_body);
infer_entry!(narrow_infer, "narrow", narrow_infer_body);
exec_entry!(narrow_exec, "narrow", narrow_exec_body);
infer_entry!(cat_infer, "cat", cat_infer_body);
exec_entry!(cat_exec, "cat", cat_exec_body);
infer_entry!(broadcast_infer, "broadcast", broadcast_infer_body);
exec_entry!(broadcast_exec, "broadcast", broadcast_exec_body);
