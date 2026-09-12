//! Borrowing readers for the C attribute list (`rs_attrs`).
//!
//! Attribute lookup is linear over the (tiny) list and never allocates, so it
//! is safe to use inside `infer`, which must stay pure. A key present with the
//! wrong [`RsAttrKind`] is treated as absent: the required-attribute checks in
//! the operator bodies then produce a clear "missing attribute" error instead
//! of silently reading a value of the wrong type.

use std::ffi::CStr;

use rustrain_abi::ffi::{RsAttr, RsAttrKind, RsAttrs};

/// Finds the attribute named `key`, if any.
pub fn get<'a>(attrs: &'a RsAttrs, key: &str) -> Option<&'a RsAttr> {
    // SAFETY: the caller guarantees `items` points to `len` live `RsAttr`s.
    unsafe { attrs.as_slice() }.iter().find(|a| {
        if a.key.is_null() {
            return false;
        }
        // SAFETY: caller-owned NUL-terminated string, valid for the call.
        unsafe { CStr::from_ptr(a.key) }.to_bytes() == key.as_bytes()
    })
}

/// `key` as an f64 attribute.
pub fn attr_f64(attrs: &RsAttrs, key: &str) -> Option<f64> {
    get(attrs, key)
        .filter(|a| a.kind == RsAttrKind::F64)
        .map(|a| a.f64)
}

/// `key` as an i64 attribute.
pub fn attr_i64(attrs: &RsAttrs, key: &str) -> Option<i64> {
    get(attrs, key)
        .filter(|a| a.kind == RsAttrKind::I64)
        .map(|a| a.i64)
}

/// `key` as a bool attribute.
pub fn attr_bool(attrs: &RsAttrs, key: &str) -> Option<bool> {
    get(attrs, key)
        .filter(|a| a.kind == RsAttrKind::BOOL)
        .map(|a| a.boolean != 0)
}

/// `key` as a string attribute, borrowing the C string.
pub fn attr_str<'a>(attrs: &'a RsAttrs, key: &str) -> Option<&'a str> {
    let a = get(attrs, key)?;
    if a.kind != RsAttrKind::STR || a.str.is_null() {
        return None;
    }
    // SAFETY: caller-owned NUL-terminated string, valid for the call.
    unsafe { CStr::from_ptr(a.str) }.to_str().ok()
}

/// `key` as a list-of-i64 attribute, borrowing the C array.
pub fn attr_i64s<'a>(attrs: &'a RsAttrs, key: &str) -> Option<&'a [i64]> {
    let a = get(attrs, key)?;
    if a.kind != RsAttrKind::I64S || a.i64s.is_null() {
        return None;
    }
    // SAFETY: caller-owned array of `n_i64s` values, valid for the call.
    Some(unsafe { std::slice::from_raw_parts(a.i64s, a.n_i64s as usize) })
}

/// Parses a required string attribute that must be one of `accepted`.
/// The error message lists the accepted values, which is part of the
/// reference provider's contract: an unknown `kind`/`scheme`/`format` is a
/// hard error, never a silent fallback.
pub fn require_str_of<'a>(
    attrs: &'a RsAttrs,
    key: &str,
    accepted: &[&str],
    op: &'static str,
) -> OpResultAttr<'a> {
    match attr_str(attrs, key) {
        None => Err(crate::error::err(
            op,
            format!(
                "attribute '{key}' is required and must be one of: {}",
                accepted.join(", ")
            ),
        )),
        Some(v) if accepted.contains(&v) => Ok(v),
        Some(v) => Err(crate::error::err(
            op,
            format!(
                "unknown {key} '{v}'; accepted values: {}",
                accepted.join(", ")
            ),
        )),
    }
}

pub type OpResultAttr<'a> = crate::error::OpResult<&'a str>;
