//! Error reporting for the `reference` provider.
//!
//! Every operator failure is an [`OpError`], which carries a human-readable
//! message naming the operator, the offending input/attribute and — for
//! rejected attribute values — the accepted values. The message is what the
//! framework surfaces through the ABI `last_error` hook, so it is written to
//! be actionable without reading plugin source.

use std::cell::RefCell;
use std::ffi::{CString, c_char};

use rustrain_abi::ffi::RsCtx;

/// One failed operator call.
#[derive(Debug, thiserror::Error)]
#[error("{op}: {msg}")]
pub struct OpError {
    pub op: &'static str,
    pub msg: String,
}

impl OpError {
    /// Every failure reports the same non-zero status; the message is the
    /// diagnostic. Distinct codes would buy nothing at the ABI boundary
    /// because the framework already pairs a failing status with
    /// `last_error`.
    pub const fn status(&self) -> i32 {
        1
    }
}

/// Constructs an [`OpError`].
pub fn err(op: &'static str, msg: impl Into<String>) -> OpError {
    OpError {
        op,
        msg: msg.into(),
    }
}

/// `format!`-style constructor used throughout the operator bodies.
macro_rules! fail {
    ($op:expr, $($arg:tt)*) => {
        $crate::error::err($op, format!($($arg)*))
    };
}
pub(crate) use fail;

pub type OpResult<T> = Result<T, OpError>;

thread_local! {
    /// The most recent failure on this thread. Kept as a `CString` so that
    /// [`plugin_last_error`] can hand the framework a stable pointer without
    /// an allocation on the error path.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Records the message served by [`plugin_last_error`].
///
/// The pointer handed out by `plugin_last_error` stays valid until the next
/// error is recorded on the same thread. That is safe under the ABI usage
/// pattern: the framework calls `last_error` immediately after a failing
/// `execute`/`infer` on the same thread.
pub fn set_last_error(e: &OpError) {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(e.to_string()).ok();
    });
}

/// Plugin-wide `last_error` callback, registered on every operator descriptor.
///
/// # Safety
/// Returns null when no error is recorded, otherwise a pointer to a
/// NUL-terminated string owned by the thread-local above.
pub unsafe extern "C" fn plugin_last_error(_ctx: *mut RsCtx) -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(s) => s.as_ptr(),
        None => std::ptr::null(),
    })
}
