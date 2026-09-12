//! Bridging between the POD `RsTensor` descriptors and `ndarray` views.
//!
//! House rules encoded here:
//!
//! * `reference.f32` only reads and writes f32 buffers (plus i32/i64 index
//!   buffers and the u8 byte buffers that carry the emulated fp8 payloads).
//! * Non-contiguous *inputs* are rejected with a clear message. Output views
//!   are always written through a caller-provided contiguous buffer whose
//!   shape/dtype the executor set from `infer()`; anything else is an error.
//!   General strided-input handling is a later improvement — the reference
//!   provider's job is ground-truth numerics, not layout gymnastics.
//! * `infer()` never allocates (success paths build shapes into fixed-size
//!   arrays via [`SmallShape`]) and never reads element data.

use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn};
use rustrain_abi::ffi::{RsDtype, RsTensor, MAX_RANK};

use crate::error::{OpResult, err};

/// A shape that fits on the stack; `infer` computes into these so it stays
/// allocation-free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmallShape {
    pub len: usize,
    pub dims: [i64; MAX_RANK],
}

impl SmallShape {
    pub fn of(t: &RsTensor) -> Self {
        let mut s = SmallShape {
            len: (t.rank as usize).min(MAX_RANK),
            dims: [0; MAX_RANK],
        };
        s.dims[..s.len].copy_from_slice(&t.shape[..s.len]);
        s
    }

    pub fn as_slice(&self) -> &[i64] {
        &self.dims[..self.len]
    }
}

/// Checks the declared dtype of a descriptor (no data access — safe in infer).
pub fn expect_dtype(
    t: &RsTensor,
    want: RsDtype,
    op: &'static str,
    who: &str,
) -> OpResult<()> {
    if t.dtype == want {
        Ok(())
    } else {
        Err(err(
            op,
            format!("input '{who}' has dtype {}, expected {}", t.dtype, want),
        ))
    }
}

/// Rejects a non-contiguous input. Documented limitation: contiguity handling
/// is a later improvement; for now the reference provider requires it.
pub fn expect_contiguous(t: &RsTensor, op: &'static str, who: &str) -> OpResult<()> {
    if t.is_contiguous() {
        Ok(())
    } else {
        Err(err(
            op,
            format!(
                "input '{who}' is non-contiguous (strides {:?}); reference.f32 requires \
                 contiguous inputs — general strided-input handling is a later improvement",
                t.strides()
            ),
        ))
    }
}

/// Reads `dims` as `usize`, rejecting ranks above the ABI maximum and negative
/// dimensions. Used by the execution paths; `infer` works on i64 shapes
/// directly and avoids this allocation.
pub fn dims_usize(t: &RsTensor, op: &'static str, who: &str) -> OpResult<Vec<usize>> {
    if t.rank as usize > MAX_RANK {
        return Err(err(
            op,
            format!("input '{who}' has rank {}, exceeding MAX_RANK {MAX_RANK}", t.rank),
        ));
    }
    let mut v = Vec::with_capacity(t.rank as usize);
    for &d in t.dims() {
        if d < 0 {
            return Err(err(
                op,
                format!("input '{who}' has a negative dimension {d}"),
            ));
        }
        v.push(d as usize);
    }
    Ok(v)
}

/// Borrows an input as a contiguous f32 array view.
///
/// # Safety
/// `t.data` must point to a live, caller-owned f32 buffer of `t.numel()`
/// elements for the duration of the returned borrow.
pub unsafe fn f32_in<'a>(
    op: &'static str,
    who: &str,
    t: &'a RsTensor,
) -> OpResult<ArrayViewD<'a, f32>> {
    expect_dtype(t, RsDtype::F32, op, who)?;
    expect_contiguous(t, op, who)?;
    let dims = dims_usize(t, op, who)?;
    let n: usize = dims.iter().product();
    if n == 0 {
        // Zero-element tensors are legal; alias a static empty buffer so the
        // view still carries the right shape.
        return ArrayViewD::from_shape(IxDyn(&dims), &[] as &[f32])
            .map_err(|e| err(op, format!("input '{who}': {e}")));
    }
    if t.data.is_null() {
        return Err(err(op, format!("input '{who}' has null data")));
    }
    let p = t.data as *const f32;
    // SAFETY: validated non-null with n live elements above.
    let buf = unsafe { std::slice::from_raw_parts(p, n) };
    ArrayViewD::from_shape(IxDyn(&dims), buf).map_err(|e| err(op, format!("input '{who}': {e}")))
}

/// Borrows an output as a contiguous mutable f32 array view.
///
/// # Safety
/// See [`f32_in`]; the buffer must additionally be writable.
pub unsafe fn f32_out<'a>(
    op: &'static str,
    t: &'a mut RsTensor,
) -> OpResult<ArrayViewMutD<'a, f32>> {
    expect_dtype(t, RsDtype::F32, op, "output")?;
    expect_contiguous(t, op, "output")?;
    let dims = dims_usize(t, op, "output")?;
    let n: usize = dims.iter().product();
    if n == 0 {
        return ArrayViewMutD::from_shape(IxDyn(&dims), &mut [] as &mut [f32])
            .map_err(|e| err(op, format!("output: {e}")));
    }
    if t.data.is_null() {
        return Err(err(op, "output has null data"));
    }
    let p = t.data as *mut f32;
    // SAFETY: validated non-null with n writable elements above.
    let buf = unsafe { std::slice::from_raw_parts_mut(p, n) };
    ArrayViewMutD::from_shape(IxDyn(&dims), buf).map_err(|e| err(op, format!("output: {e}")))
}

/// Borrows an input as a contiguous byte view; used for the emulated fp8
/// payloads (`f8e4m3` / `f8e5m2` are stored one byte per element).
///
/// # Safety
/// `t.data` must point to a live caller-owned byte buffer of `t.numel()`
/// elements.
pub unsafe fn u8_in<'a>(
    op: &'static str,
    who: &str,
    t: &'a RsTensor,
) -> OpResult<ArrayViewD<'a, u8>> {
    expect_contiguous(t, op, who)?;
    let dims = dims_usize(t, op, who)?;
    let n: usize = dims.iter().product();
    if n == 0 {
        return ArrayViewD::from_shape(IxDyn(&dims), &[] as &[u8])
            .map_err(|e| err(op, format!("input '{who}': {e}")));
    }
    if t.data.is_null() {
        return Err(err(op, format!("input '{who}' has null data")));
    }
    let p = t.data as *const u8;
    // SAFETY: validated non-null with n live bytes above.
    let buf = unsafe { std::slice::from_raw_parts(p, n) };
    ArrayViewD::from_shape(IxDyn(&dims), buf).map_err(|e| err(op, format!("input '{who}': {e}")))
}

/// Borrows an output as a mutable byte view; used for the emulated fp8
/// payloads (`f8e4m3` / `f8e5m2` are stored one byte per element).
///
/// # Safety
/// See [`f32_out`].
pub unsafe fn u8_out<'a>(
    op: &'static str,
    t: &'a mut RsTensor,
) -> OpResult<ArrayViewMutD<'a, u8>> {
    let dims = dims_usize(t, op, "output")?;
    let n: usize = dims.iter().product();
    if n == 0 {
        return ArrayViewMutD::from_shape(IxDyn(&dims), &mut [] as &mut [u8])
            .map_err(|e| err(op, format!("output: {e}")));
    }
    if t.data.is_null() {
        return Err(err(op, "output has null data"));
    }
    let p = t.data as *mut u8;
    // SAFETY: validated non-null with n writable bytes above.
    let buf = unsafe { std::slice::from_raw_parts_mut(p, n) };
    ArrayViewMutD::from_shape(IxDyn(&dims), buf).map_err(|e| err(op, format!("output: {e}")))
}

/// Borrows an output as a mutable i32 view (expert indices etc.).
///
/// # Safety
/// See [`f32_out`].
pub unsafe fn i32_out<'a>(
    op: &'static str,
    t: &'a mut RsTensor,
) -> OpResult<ArrayViewMutD<'a, i32>> {
    let dims = dims_usize(t, op, "output")?;
    let n: usize = dims.iter().product();
    if n == 0 {
        return ArrayViewMutD::from_shape(IxDyn(&dims), &mut [] as &mut [i32])
            .map_err(|e| err(op, format!("output: {e}")));
    }
    if t.data.is_null() {
        return Err(err(op, "output has null data"));
    }
    let p = t.data as *mut i32;
    // SAFETY: validated non-null with n writable elements above.
    let buf = unsafe { std::slice::from_raw_parts_mut(p, n) };
    ArrayViewMutD::from_shape(IxDyn(&dims), buf).map_err(|e| err(op, format!("output: {e}")))
}

/// Reads an index tensor (i32 or i64) into an owned `Vec<i64>`. Executions
/// paths only; `infer` checks the dtype without touching data.
///
/// # Safety
/// `t.data` must point to a live caller-owned index buffer.
pub unsafe fn indices_i64(
    op: &'static str,
    who: &str,
    t: &RsTensor,
) -> OpResult<Vec<i64>> {
    expect_contiguous(t, op, who)?;
    let n = t.numel();
    if n < 0 {
        return Err(err(op, format!("input '{who}' has a negative dimension")));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if t.data.is_null() {
        return Err(err(op, format!("input '{who}' has null data")));
    }
    let mut out = Vec::with_capacity(n as usize);
    match t.dtype {
        RsDtype::I32 => {
            // SAFETY: validated non-null with n live elements.
            let buf = unsafe { std::slice::from_raw_parts(t.data as *const i32, n as usize) };
            out.extend(buf.iter().map(|&v| v as i64));
        }
        RsDtype::I64 => {
            // SAFETY: validated non-null with n live elements.
            let buf = unsafe { std::slice::from_raw_parts(t.data as *const i64, n as usize) };
            out.extend_from_slice(buf);
        }
        other => {
            return Err(err(
                op,
                format!("input '{who}' has dtype {other}, expected i32 or i64"),
            ));
        }
    }
    Ok(out)
}

/// Resolves an axis attribute: negative values count from the end, so `-1`
/// is the last axis. Errors when out of range.
pub fn resolve_axis(axis: i64, rank: usize, op: &'static str) -> OpResult<usize> {
    let ax = if axis < 0 {
        rank as i64 + axis
    } else {
        axis
    };
    if ax < 0 || ax >= rank as i64 {
        return Err(err(
            op,
            format!(
                "axis {axis} is out of range for rank {rank} \
                 (negative axes count from the end; -1 = last)"
            ),
        ));
    }
    Ok(ax as usize)
}

/// Fills an output descriptor with a contiguous layout (used by `infer`).
pub fn set_output_desc(t: &mut RsTensor, dtype: RsDtype, shape: &[i64]) {
    t.dtype = dtype;
    t.rank = shape.len() as u32;
    t.shape[..shape.len()].copy_from_slice(shape);
    t.set_contiguous_strides();
}

/// Fills an output descriptor with an arbitrary view layout (used by the
/// zero-copy meta ops, where the stride may contain zeros or swaps).
pub fn set_view_desc(
    t: &mut RsTensor,
    dtype: RsDtype,
    rank: usize,
    shape: &[i64; MAX_RANK],
    stride: &[i64; MAX_RANK],
    data: *mut std::ffi::c_void,
) {
    t.dtype = dtype;
    t.rank = rank as u32;
    t.shape = *shape;
    t.stride = *stride;
    t.data = data;
}

/// Validates that a caller-provided output matches what the op requires:
/// same dtype and exactly the shape `infer()` declared. The executor
/// pre-sets these fields; a mismatch here is a plan/executor bug, and the
/// reference provider refuses rather than writing out of bounds.
pub fn expect_out(
    t: &RsTensor,
    op: &'static str,
    want_dtype: RsDtype,
    want_shape: &[i64],
) -> OpResult<()> {
    if t.dtype != want_dtype {
        return Err(err(
            op,
            format!(
                "output has dtype {}, expected {} — the executor must allocate outputs \
                 with the dtype infer() declared",
                t.dtype, want_dtype
            ),
        ));
    }
    if t.dims() != want_shape {
        return Err(err(
            op,
            format!(
                "output has shape {:?}, expected {:?} — the executor must allocate outputs \
                 with the shape infer() declared",
                t.dims(),
                want_shape
            ),
        ));
    }
    Ok(())
}

/// Broadcast shape of two inputs (right-aligned, size-1 dims stretch), as a
/// stack shape for `infer`. Pure — no allocation.
pub fn broadcast_shape_small(
    a: &[i64],
    b: &[i64],
    op: &'static str,
) -> OpResult<SmallShape> {
    let r = a.len().max(b.len());
    if r > MAX_RANK {
        return Err(err(op, format!("broadcast rank {r} exceeds MAX_RANK")));
    }
    let mut dims = [0i64; MAX_RANK];
    for i in 0..r {
        let da = if i < r - a.len() { 1 } else { a[i - (r - a.len())] };
        let db = if i < r - b.len() { 1 } else { b[i - (r - b.len())] };
        dims[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            (x, y) => {
                return Err(err(
                    op,
                    format!("cannot broadcast shapes {:?} and {:?} (dims {x} vs {y})", a, b),
                ));
            }
        };
    }
    Ok(SmallShape { len: r, dims })
}

/// Casts the helper errors used with `?` inside functions that name the op
/// differently at call time.
pub type ShapeResult = OpResult<SmallShape>;
