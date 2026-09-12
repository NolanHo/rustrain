//! Adapters between the raw C entry-point signatures and the safe Rust
//! operator bodies.
//!
//! Each published op is a pair of `extern "C"` trampolines (`infer` /
//! `execute`) plus the shared `memory` trampoline. The trampolines:
//!
//! * slice the raw pointer arrays without allocating,
//! * catch panics so that nothing ever unwinds across the FFI boundary,
//! * record the error message in the thread-local `last_error` slot.
//!
//! The bodies themselves are plain `fn(&mut Call, &RsAttrs) -> OpResult<()>`,
//! which keeps them unit-testable without dlopen.

use std::panic::AssertUnwindSafe;
use std::ptr;

use rustrain_abi::ffi::{RsAttrs, RsMemReq, RsTensor};

use crate::error::{OpError, OpResult, err, set_last_error};

/// A borrowed view of one operator call: raw descriptor pointers plus the
/// operator name used in every error message.
pub struct Call<'a> {
    pub op: &'static str,
    pub ins: &'a [*const RsTensor],
    pub outs: &'a [*mut RsTensor],
}

impl<'a> Call<'a> {
    /// # Safety
    /// `in_`/`out` must be null or point to arrays of `n_in`/`n_out` live
    /// descriptor pointers.
    pub unsafe fn from_raw(
        op: &'static str,
        in_: *const *const RsTensor,
        n_in: u32,
        out: *const *mut RsTensor,
        n_out: u32,
    ) -> Self {
        // SAFETY: forwarded from the C caller; null + zero length is legal.
        let ins = if n_in == 0 || in_.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(in_, n_in as usize) }
        };
        let outs = if n_out == 0 || out.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(out, n_out as usize) }
        };
        Call { op, ins, outs }
    }

    pub fn n_in(&self) -> usize {
        self.ins.len()
    }

    pub fn n_out(&self) -> usize {
        self.outs.len()
    }

    /// Borrows input `i`. The returned lifetime is `'a` (the lifetime of the
    /// descriptor array), deliberately *not* tied to `&self`, so operator
    /// bodies can hold input borrows across `out_t` calls.
    ///
    /// # Safety
    /// `i < n_in`; descriptor liveness is the ABI caller's contract.
    pub fn in_t(&self, i: usize) -> &'a RsTensor {
        // SAFETY: index checked by the caller; liveness by the ABI contract.
        unsafe { &*self.ins[i] }
    }

    /// Borrows output `i`, likewise decoupled from `&mut self`.
    ///
    /// # Safety
    /// `i < n_out`; the caller must supply distinct live output descriptors
    /// (two `&mut` to the same tensor through this method would alias).
    pub fn out_t(&mut self, i: usize) -> &'a mut RsTensor {
        // SAFETY: index checked by the caller; liveness by the ABI contract.
        unsafe { &mut *self.outs[i] }
    }

    /// Checks the input arity. `range` is the accepted range as an inclusive
    /// `(min, max)` pair.
    pub fn expect_arity(&self, range: (usize, usize)) -> OpResult<()> {
        let (min, max) = range;
        if !(min..=max).contains(&self.n_in()) {
            return Err(err(
                self.op,
                format!(
                    "expected {min}..={max} input tensor(s), got {}",
                    self.n_in()
                ),
            ));
        }
        Ok(())
    }

    pub fn expect_out_count(&self, want: usize) -> OpResult<()> {
        if self.n_out() != want {
            return Err(err(
                self.op,
                format!("expected {want} output tensor(s), got {}", self.n_out()),
            ));
        }
        Ok(())
    }
}

/// Static empty attribute list, used when the framework passes a null
/// `attrs` pointer (which the reference provider treats as "no attributes").
/// Built as a local because `RsAttrs` contains raw pointers and is therefore
/// `!Sync` — a `static` of it would not type-check.
fn empty_attrs() -> RsAttrs {
    RsAttrs {
        items: ptr::null(),
        len: 0,
        _pad: 0,
    }
}

fn attrs_of(p: *const RsAttrs) -> RsAttrs {
    if p.is_null() {
        empty_attrs()
    } else {
        // SAFETY: the ABI contract says `attrs` is null or a live RsAttrs.
        unsafe { *p }
    }
}

/// Shared trampoline for `infer` and `execute`.
///
/// # Safety
/// See the `RsInferFn` / `RsExecuteFn` contracts in `rustrain-abi`.
pub unsafe fn run(
    op: &'static str,
    in_: *const *const RsTensor,
    n_in: u32,
    out: *const *mut RsTensor,
    n_out: u32,
    attrs: *const RsAttrs,
    body: impl FnOnce(&mut Call<'_>, &RsAttrs) -> OpResult<()>,
) -> i32 {
    // Panics must not unwind into C code (UB); report them like any failure.
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: forwarded from the C caller under the ABI contract.
        let mut call = unsafe { Call::from_raw(op, in_, n_in, out, n_out) };
        let attrs = attrs_of(attrs);
        body(&mut call, &attrs)
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(&e);
            e.status()
        }
        Err(_) => {
            let e: OpError = err(op, "internal panic in operator body");
            set_last_error(&e);
            e.status()
        }
    }
}

/// The shared `memory` trampoline: the reference provider uses no workspace,
/// so every op reports zeros.
///
/// Why a real function instead of a null `memory` slot: to the compiler a
/// null `memory` means "cannot plan", so every op registers this zero
/// reporter. If an op ever needs a workspace it must report the bytes here —
/// outputs stay caller-provided either way (contract C-3: no allocation on
/// the output side).
///
/// # Safety
/// `out` must be null or a writable `RsMemReq`.
pub unsafe extern "C" fn memory_zero(
    _io: *const *const RsTensor,
    _n_io: u32,
    _attrs: *const RsAttrs,
    out: *mut RsMemReq,
) -> i32 {
    if out.is_null() {
        let e = err("memory", "null RsMemReq output pointer");
        set_last_error(&e);
        return e.status();
    }
    // SAFETY: checked non-null above.
    unsafe { *out = RsMemReq::default() };
    0
}
