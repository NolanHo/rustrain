//! Operator attributes: a typed map on the Rust side, and an owned array of
//! C structs ready to hand to a plugin across the ABI.
//!
//! Two representations exist because the two sides want opposite things. The
//! plan wants a typed, serializable, comparable map (it goes into the digest);
//! the ABI wants a flat POD array with process-lifetime lifetimes. [`Attrs`] is
//! the former, [`AbiAttrs`] the latter, and converting is explicit so nobody
//! accidentally holds a pointer into a dropped temporary.

use std::collections::BTreeMap;
use std::ffi::{CString, c_char};

use serde::{Deserialize, Serialize};

use rustrain_abi::ffi::{RsAttr, RsAttrKind, RsAttrs};

/// A typed attribute value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AttrValue {
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    I64s(Vec<i64>),
}

impl AttrValue {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            AttrValue::I64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            AttrValue::F64(v) => Some(*v),
            AttrValue::I64(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            AttrValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            AttrValue::Str(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i64s(&self) -> Option<&[i64]> {
        match self {
            AttrValue::I64s(v) => Some(v),
            _ => None,
        }
    }

    fn kind(&self) -> RsAttrKind {
        match self {
            AttrValue::I64(_) => RsAttrKind::I64,
            AttrValue::F64(_) => RsAttrKind::F64,
            AttrValue::Bool(_) => RsAttrKind::BOOL,
            AttrValue::Str(_) => RsAttrKind::STR,
            AttrValue::I64s(_) => RsAttrKind::I64S,
        }
    }
}

/// An ordered, typed attribute map.
///
/// Ordered (`BTreeMap`) rather than hashed so that two plans built with the same
/// attributes always serialize and digest identically.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Attrs(BTreeMap<String, AttrValue>);

impl Attrs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(mut self, key: impl Into<String>, value: impl Into<AttrValue>) -> Self {
        self.0.insert(key.into(), value.into());
        self
    }

    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<AttrValue>) {
        self.0.insert(key.into(), value.into());
    }

    pub fn get(&self, key: &str) -> Option<&AttrValue> {
        self.0.get(key)
    }

    pub fn i64(&self, key: &str) -> Option<i64> {
        self.0.get(key).and_then(AttrValue::as_i64)
    }

    pub fn f64(&self, key: &str) -> Option<f64> {
        self.0.get(key).and_then(AttrValue::as_f64)
    }

    pub fn bool(&self, key: &str) -> Option<bool> {
        self.0.get(key).and_then(AttrValue::as_bool)
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(AttrValue::as_str)
    }

    pub fn i64s(&self, key: &str) -> Option<&[i64]> {
        self.0.get(key).and_then(AttrValue::as_i64s)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &AttrValue)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Renders as `key=value` pairs, sorted; used in `plan explain` and errors.
    pub fn describe(&self) -> String {
        if self.0.is_empty() {
            return String::new();
        }
        self.0
            .iter()
            .map(|(k, v)| format!("{k}={}", render(v)))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Produces an owned, ABI-ready attribute array.
    ///
    /// The returned value owns every C string and slice it points at, so it can
    /// be moved (but must not be dropped while a plugin is still using it).
    pub fn to_abi(&self) -> AbiAttrs {
        AbiAttrs::from_attrs(self)
    }
}

fn render(v: &AttrValue) -> String {
    match v {
        AttrValue::I64(i) => i.to_string(),
        AttrValue::F64(f) => format!("{f}"),
        AttrValue::Bool(b) => b.to_string(),
        AttrValue::Str(s) => format!("\"{s}\""),
        AttrValue::I64s(v) => format!(
            "[{}]",
            v.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
        ),
    }
}

impl From<i64> for AttrValue {
    fn from(v: i64) -> Self {
        AttrValue::I64(v)
    }
}
impl From<i32> for AttrValue {
    fn from(v: i32) -> Self {
        AttrValue::I64(v as i64)
    }
}
impl From<usize> for AttrValue {
    fn from(v: usize) -> Self {
        AttrValue::I64(v as i64)
    }
}
impl From<f64> for AttrValue {
    fn from(v: f64) -> Self {
        AttrValue::F64(v)
    }
}
impl From<bool> for AttrValue {
    fn from(v: bool) -> Self {
        AttrValue::Bool(v)
    }
}
impl From<&str> for AttrValue {
    fn from(v: &str) -> Self {
        AttrValue::Str(v.to_string())
    }
}
impl From<String> for AttrValue {
    fn from(v: String) -> Self {
        AttrValue::Str(v)
    }
}
impl From<Vec<i64>> for AttrValue {
    fn from(v: Vec<i64>) -> Self {
        AttrValue::I64s(v)
    }
}

/// Owned storage backing an `RsAttrs` view.
///
/// Keeping the owners in the same struct as the pointers is what makes this
/// safe to move: the strings and slices are `Box`ed, so their addresses are
/// stable even when `AbiAttrs` itself moves.
pub struct AbiAttrs {
    keys: Vec<CString>,
    strs: Vec<Option<CString>>,
    slices: Vec<Option<Box<[i64]>>>,
    items: Vec<RsAttr>,
    view: RsAttrs,
}

impl Default for AbiAttrs {
    fn default() -> Self {
        Self::empty()
    }
}

impl AbiAttrs {
    /// An attribute view with no entries.
    ///
    /// Written out field by field on purpose: `..Default::default()` here would
    /// call `Default::default()`, which is defined in terms of this function.
    pub fn empty() -> Self {
        Self {
            keys: Vec::new(),
            strs: Vec::new(),
            slices: Vec::new(),
            items: Vec::new(),
            view: RsAttrs {
                items: std::ptr::null(),
                len: 0,
                _pad: 0,
            },
        }
    }

    fn from_attrs(attrs: &Attrs) -> Self {
        let mut out = Self::default();

        for (k, v) in attrs.iter() {
            let key = CString::new(k).expect("attribute key must not contain NUL");
            out.keys.push(key);
            let key_ptr = out.keys.last().unwrap().as_ptr();

            let mut item = RsAttr {
                key: key_ptr,
                kind: v.kind(),
                _pad0: 0,
                i64: 0,
                f64: 0.0,
                boolean: 0,
                _pad1: 0,
                str: std::ptr::null(),
                i64s: std::ptr::null(),
                n_i64s: 0,
                _pad2: 0,
            };

            match v {
                AttrValue::I64(i) => item.i64 = *i,
                AttrValue::F64(f) => item.f64 = *f,
                AttrValue::Bool(b) => item.boolean = i32::from(*b),
                AttrValue::Str(s) => {
                    let c = CString::new(s.as_str()).expect("attribute value must not contain NUL");
                    out.strs.push(Some(c));
                    item.str = out.strs.last().unwrap().as_ref().unwrap().as_ptr();
                }
                AttrValue::I64s(xs) => {
                    let boxed = xs.clone().into_boxed_slice();
                    out.slices.push(Some(boxed));
                    let s = out.slices.last().unwrap().as_ref().unwrap();
                    item.i64s = s.as_ptr();
                    item.n_i64s = s.len() as u32;
                }
            }
            out.items.push(item);
        }

        out.view = RsAttrs {
            items: out.items.as_ptr(),
            len: out.items.len() as u32,
            _pad: 0,
        };
        out
    }

    /// The view to pass across the ABI. Valid as long as `self` is alive.
    pub fn as_rs(&self) -> &RsAttrs {
        &self.view
    }

    pub fn as_ptr(&self) -> *const RsAttrs {
        &self.view
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Reads a `*const c_char` attribute key.
///
/// # Safety
/// `p` must be null or a NUL-terminated string.
pub unsafe fn read_key(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    unsafe { std::ffi::CStr::from_ptr(p) }
        .to_str()
        .ok()
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_roundtrip_all_kinds() {
        let attrs = Attrs::new()
            .set("kind", "silu")
            .set("eps", 1e-6)
            .set("axis", -1i64)
            .set("keepdim", true)
            .set("block", vec![128i64, 128]);

        let abi = attrs.to_abi();
        assert_eq!(abi.len(), 5);

        let items = unsafe { abi.as_rs().as_slice() };
        let find = |name: &str| {
            items
                .iter()
                .find(|a| unsafe { read_key(a.key) }.as_deref() == Some(name))
                .copied()
                .unwrap_or_else(|| panic!("missing attr {name}"))
        };

        assert_eq!(find("kind").kind, RsAttrKind::STR);
        assert_eq!(
            unsafe { std::ffi::CStr::from_ptr(find("kind").str) }
                .to_str()
                .unwrap(),
            "silu"
        );
        assert_eq!(find("axis").i64, -1);
        assert_eq!(find("keepdim").boolean, 1);
        assert_eq!(find("eps").f64, 1e-6);

        let block = find("block");
        assert_eq!(block.n_i64s, 2);
        let xs = unsafe { std::slice::from_raw_parts(block.i64s, 2) };
        assert_eq!(xs, &[128, 128]);
    }

    /// Moving the owner must not invalidate what the view points at.
    ///
    /// The address of the `RsAttrs` view itself does move (it is a field of the
    /// owner), and that is fine: the plugin receives the address at call time.
    /// What must not move is the data behind it — keys, strings and slices are
    /// boxed precisely so that this holds.
    #[test]
    fn abi_attrs_survive_move() {
        let attrs = Attrs::new().set("kind", "gelu").set("block", vec![1i64, 2, 3]);
        let abi = attrs.to_abi();
        let moved = abi;

        let items = unsafe { moved.as_rs().as_slice() };
        assert_eq!(items.len(), 2);

        let find = |name: &str| {
            *items
                .iter()
                .find(|a| unsafe { read_key(a.key) }.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("missing attr {name}"))
        };

        let block = find("block");
        let xs = unsafe { std::slice::from_raw_parts(block.i64s, block.n_i64s as usize) };
        assert_eq!(xs, &[1, 2, 3]);

        let kind = find("kind");
        assert_eq!(
            unsafe { std::ffi::CStr::from_ptr(kind.str) }.to_str().unwrap(),
            "gelu"
        );
    }

    #[test]
    fn empty_attrs_is_null_view() {
        let abi = Attrs::new().to_abi();
        assert!(abi.is_empty());
        assert_eq!(unsafe { abi.as_rs().as_slice() }.len(), 0);
    }

    #[test]
    fn describe_is_sorted_and_stable() {
        let a = Attrs::new().set("b", 2i64).set("a", 1i64);
        assert_eq!(a.describe(), "a=1 b=2");
    }
}
