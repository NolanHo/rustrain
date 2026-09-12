//! `#[repr(C)]` mirrors of `include/rustrain_op.h`.
//!
//! Layout is a hard contract with plugins compiled from the C header. Rust
//! `#[repr(C)]` reproduces the C layout rule for the same field order, so the
//! field order here must match the header byte for byte. `test::layout` pins
//! the resulting sizes.

use std::ffi::{c_char, c_void};
use std::fmt;

pub const MAX_RANK: usize = 8;
pub const MAX_RESERVED: usize = 4;
pub const DTYPE_COUNT: usize = 9;

/// Newtype over the C enum rather than a Rust `enum`, so that a value written
/// by a plugin built against a newer header cannot be undefined behaviour here.
macro_rules! c_enum {
    // The optional visibility and the variant doc comments are accepted because
    // the call sites below spell them out; the generated struct is public
    // either way, and the docs belong on the variant constants. Without both,
    // `$name:ident` swallows the `pub` keyword and the macro fails to match.
    ($(#[$m:meta])* $vis:vis $name:ident : $repr:ty {
        $($(#[$vm:meta])* $variant:ident = $value:expr;)*
    }) => {
        $(#[$m])*
        #[repr(transparent)]
        #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
        #[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
        #[cfg_attr(feature = "serde", serde(transparent))]
        pub struct $name(pub $repr);

        impl $name {
            $($(#[$vm])* pub const $variant: Self = Self($value);)*
            #[inline]
            pub const fn raw(self) -> $repr { self.0 }
            #[inline]
            pub const fn from_raw(raw: $repr) -> Self { Self(raw) }
        }
    };
}

c_enum! {
    /// Element type of a tensor buffer.
    RsDtype: i32 {
        F32 = 0; F16 = 1; BF16 = 2;
        F8E4M3 = 3; F8E5M2 = 4; FP4E2M1 = 5;
        I32 = 6; I64 = 7; U8 = 8;
    }
}

impl RsDtype {
    pub const ALL: [RsDtype; DTYPE_COUNT] = [
        Self::F32,
        Self::F16,
        Self::BF16,
        Self::F8E4M3,
        Self::F8E5M2,
        Self::FP4E2M1,
        Self::I32,
        Self::I64,
        Self::U8,
    ];

    pub const fn name(self) -> &'static str {
        match self.0 {
            0 => "f32",
            1 => "f16",
            2 => "bf16",
            3 => "f8e4m3",
            4 => "f8e5m2",
            5 => "fp4e2m1",
            6 => "i32",
            7 => "i64",
            8 => "u8",
            _ => "unknown",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|d| d.name() == s)
    }

    /// Size in bytes of one element, when it is a whole number of bytes.
    pub const fn byte_width(self) -> Option<u32> {
        match self.0 {
            0 => Some(4),
            1 | 2 => Some(2),
            3 | 4 | 8 => Some(1),
            6 => Some(4),
            7 => Some(8),
            _ => None, // fp4 is sub-byte
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self.0, 0..=5)
    }

    pub const fn is_quantized(self) -> bool {
        matches!(self.0, 3..=5)
    }
}

impl fmt::Display for RsDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

c_enum! {
    pub RsDeviceKind: i32 { CPU = 0; CUDA = 1; }
}

c_enum! {
    /// How the framework obtains gradients for an operator.
    RsBackwardKind: i32 {
        /// Differentiable by composing the declared expansion.
        AUTODIFF = 0;
        /// A separate operator implements the backward pass.
        EXPLICIT = 1;
        /// No gradients flow through this operator.
        NONDIFF = 2;
    }
}

c_enum! {
    /// Granularity of a quantization scheme. This is data, not something a
    /// kernel may infer from the shape of a scale tensor.
    RsQuantKind: i32 { NONE = 0; PER_TENSOR = 1; PER_TOKEN = 2; PER_BLOCK = 3; }
}

c_enum! {
    pub RsScaleMode: i32 { STATIC = 0; DYNAMIC_AMAX = 1; DELAYED = 2; }
}

c_enum! {
    pub RsAttrKind: i32 { I64 = 0; F64 = 1; BOOL = 2; STR = 3; I64S = 4; }
}

c_enum! {
    pub RsGroupKind: u32 { TP = 1; EP = 2; CP = 4; DP = 8; }
}

c_enum! {
    pub RsCollectiveKind: i32 {
        ALL_REDUCE = 0; ALL_GATHER = 1; REDUCE_SCATTER = 2; SEND_RECV = 3;
    }
}

/// Backend-agnostic tensor descriptor. Mirrors `rs_tensor`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RsTensor {
    pub dtype: RsDtype,
    pub rank: u32,
    pub shape: [i64; MAX_RANK],
    pub stride: [i64; MAX_RANK],
    /// Device (or host) pointer to the element buffer.
    pub data: *mut c_void,
    /// Optional `RsTensor*` with quantization scales.
    pub scale: *mut c_void,
    /// Optional `RsTensor*` with dynamic-scaling history.
    pub amax: *mut c_void,
    /// Backend-private. The aten backend stores an `at::Tensor*` in slot 0.
    pub reserved: [*mut c_void; MAX_RESERVED],
}

impl Default for RsTensor {
    fn default() -> Self {
        Self {
            dtype: RsDtype::F32,
            rank: 0,
            shape: [0; MAX_RANK],
            stride: [0; MAX_RANK],
            data: std::ptr::null_mut(),
            scale: std::ptr::null_mut(),
            amax: std::ptr::null_mut(),
            reserved: [std::ptr::null_mut(); MAX_RESERVED],
        }
    }
}

impl RsTensor {
    pub fn new(dtype: RsDtype, shape: &[i64]) -> Self {
        let mut t = Self {
            dtype,
            rank: shape.len() as u32,
            ..Default::default()
        };
        for (i, d) in shape.iter().take(MAX_RANK).enumerate() {
            t.shape[i] = *d;
        }
        t.set_contiguous_strides();
        t
    }

    /// Row-major (contiguous) strides for the current shape.
    pub fn set_contiguous_strides(&mut self) {
        let r = (self.rank as usize).min(MAX_RANK);
        let mut acc = 1i64;
        for i in (0..r).rev() {
            self.stride[i] = acc;
            acc *= self.shape[i].max(1);
        }
        for i in r..MAX_RANK {
            self.stride[i] = 0;
        }
    }

    pub fn dims(&self) -> &[i64] {
        &self.shape[..(self.rank as usize).min(MAX_RANK)]
    }

    pub fn strides(&self) -> &[i64] {
        &self.stride[..(self.rank as usize).min(MAX_RANK)]
    }

    pub fn numel(&self) -> i64 {
        self.dims().iter().product()
    }

    pub fn is_contiguous(&self) -> bool {
        let r = (self.rank as usize).min(MAX_RANK);
        let mut acc = 1i64;
        for i in (0..r).rev() {
            if self.shape[i] != 1 && self.stride[i] != acc {
                return false;
            }
            acc *= self.shape[i].max(1);
        }
        true
    }

    /// Byte footprint of the element buffer, or `None` for sub-byte dtypes.
    pub fn byte_len(&self) -> Option<u64> {
        self.dtype.byte_width().map(|w| self.numel() as u64 * w as u64)
    }

    pub fn is_null(&self) -> bool {
        self.data.is_null()
    }

    /// Reads the optional scale descriptor. Returns `None` when absent.
    ///
    /// # Safety
    /// If `scale` is non-null it must point to a live `RsTensor` owned by the
    /// caller for the duration of the call.
    pub unsafe fn scale_tensor(&self) -> Option<&RsTensor> {
        unsafe { (self.scale as *const RsTensor).as_ref() }
    }

    /// Reads the optional amax descriptor. Returns `None` when absent.
    ///
    /// # Safety
    /// See [`RsTensor::scale_tensor`].
    pub unsafe fn amax_tensor(&self) -> Option<&RsTensor> {
        unsafe { (self.amax as *const RsTensor).as_ref() }
    }
}

impl fmt::Debug for RsTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RsTensor({} {:?} strides={:?} data={:p} scale={}{})",
            self.dtype,
            self.dims(),
            self.strides(),
            self.data,
            if self.scale.is_null() { "none" } else { "yes" },
            if self.amax.is_null() { "" } else { " +amax" }
        )
    }
}

/// `{name, variant, version}`.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct RsOpId {
    pub name: *const c_char,
    pub variant: *const c_char,
    pub version: u32,
}

/// Numerics contract of an operator. Mirrors `rs_numerics`.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct RsNumerics {
    pub in_dtype: RsDtype,
    pub out_dtype: RsDtype,
    pub accum_dtype: RsDtype,
    pub grad_dtype: RsDtype,
    pub quant: RsQuantKind,
    pub block_m: u32,
    pub block_n: u32,
    pub scale_dtype: RsDtype,
    pub scale_mode: RsScaleMode,
    pub amax_history: u32,
    pub _pad: u32,
}

impl RsNumerics {
    /// True when the two contracts are compatible without an explicit
    /// convert/quantize step between them.
    pub fn compatible_with(&self, next: &RsNumerics) -> bool {
        self.out_dtype == next.in_dtype && self.quant == next.quant
    }

    pub fn describe(&self) -> String {
        let mut s = format!(
            "{}->{} (accum {})",
            self.in_dtype, self.out_dtype, self.accum_dtype
        );
        match self.quant.0 {
            0 => {}
            1 => s.push_str(" per-tensor"),
            2 => s.push_str(" per-token"),
            3 => s.push_str(&format!(" per-block {}x{}", self.block_m, self.block_n)),
            other => s.push_str(&format!(" quant#{other}")),
        }
        s
    }
}

/// One attribute. Mirrors `rs_attr`. A union would be tighter but this flat
/// form is layout-safe from both languages.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RsAttr {
    pub key: *const c_char,
    pub kind: RsAttrKind,
    pub _pad0: u32,
    pub i64: i64,
    pub f64: f64,
    pub boolean: i32,
    pub _pad1: u32,
    pub str: *const c_char,
    pub i64s: *const i64,
    pub n_i64s: u32,
    pub _pad2: u32,
}

/// Attribute list. Mirrors `rs_attrs`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RsAttrs {
    pub items: *const RsAttr,
    pub len: u32,
    pub _pad: u32,
}

impl RsAttrs {
    /// Borrow the attribute slice, or an empty slice when `items` is null.
    ///
    /// # Safety
    /// `items` must point to `len` live `RsAttr` values.
    pub unsafe fn as_slice(&self) -> &[RsAttr] {
        if self.items.is_null() || self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.items, self.len as usize) }
        }
    }
}

/// Declared environment requirements. Mirrors `rs_requires`.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct RsRequires {
    pub dtype_mask: u32,
    pub min_sm: u32,
    pub min_world_size: i64,
    pub needs_groups: u32,
    pub _pad: u32,
}

impl RsRequires {
    pub fn dtype_mask_for(dtypes: &[RsDtype]) -> u32 {
        dtypes
            .iter()
            .filter(|d| (0..32).contains(&d.0))
            .fold(0u32, |acc, d| acc | (1u32 << d.0))
    }

    pub fn accepts(&self, dtype: RsDtype) -> bool {
        (0..32).contains(&dtype.0) && self.dtype_mask & (1u32 << dtype.0) != 0
    }

    /// Human-readable form of the accepted dtype set.
    pub fn dtypes(&self) -> Vec<RsDtype> {
        RsDtype::ALL
            .into_iter()
            .filter(|d| self.accepts(*d))
            .collect()
    }

    pub fn needs_group(&self, group: RsGroupKind) -> bool {
        self.needs_groups & group.0 != 0
    }
}

/// A collective an operator performs internally. Mirrors `rs_collective`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RsCollective {
    pub kind: RsCollectiveKind,
    pub group: RsGroupKind,
    /// Index into the operator's io list (inputs first, then outputs).
    pub tensor_index: u32,
    pub on_side_stream: u32,
}

/// Declared memory footprint. Mirrors `rs_mem_req`.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct RsMemReq {
    pub workspace_bytes: u64,
    pub save_for_backward_bytes: u64,
    pub save_tensor_count: u32,
    pub _pad: u32,
}

/// One node of a declared expansion. Mirrors `rs_expansion_node`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RsExpansionNode {
    pub op: *const c_char,
    pub attrs: *const RsAttrs,
    pub inputs: *const i32,
    pub n_inputs: u32,
    pub outputs: *const i32,
    pub n_outputs: u32,
}

impl RsExpansionNode {
    /// # Safety
    /// Pointers must refer to live arrays for the duration of the borrow.
    pub unsafe fn input_ids(&self) -> &[i32] {
        if self.inputs.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.inputs, self.n_inputs as usize) }
        }
    }

    /// # Safety
    /// See [`RsExpansionNode::input_ids`].
    pub unsafe fn output_ids(&self) -> &[i32] {
        if self.outputs.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.outputs, self.n_outputs as usize) }
        }
    }
}

/// Declared primitive expansion of a composite operator. Mirrors `rs_expansion`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RsExpansion {
    pub n_nodes: u32,
    pub _pad: u32,
    pub nodes: *const RsExpansionNode,
    pub n_tensors: u32,
    pub n_inputs: u32,
    pub n_outputs: u32,
    pub _pad2: u32,
}

impl RsExpansion {
    /// # Safety
    /// `nodes` must point to `n_nodes` live descriptors.
    pub unsafe fn as_slice(&self) -> &[RsExpansionNode] {
        if self.nodes.is_null() || self.n_nodes == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.nodes, self.n_nodes as usize) }
        }
    }
}

pub type RsInferFn = unsafe extern "C" fn(
    in_: *const *const RsTensor,
    n_in: u32,
    out: *const *mut RsTensor,
    n_out: u32,
    attrs: *const RsAttrs,
) -> i32;

pub type RsMemoryFn = unsafe extern "C" fn(
    io: *const *const RsTensor,
    n_io: u32,
    attrs: *const RsAttrs,
    out: *mut RsMemReq,
) -> i32;

pub type RsExecuteFn = unsafe extern "C" fn(
    ctx: *mut RsCtx,
    in_: *const *const RsTensor,
    n_in: u32,
    out: *const *mut RsTensor,
    n_out: u32,
    attrs: *const RsAttrs,
) -> i32;

pub type RsLastErrorFn = unsafe extern "C" fn(ctx: *mut RsCtx) -> *const c_char;

/// Service table the framework injects into a plugin (contract C-3).
/// Mirrors `rs_services`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RsServices {
    pub abi_version: u32,
    pub struct_size: u32,
    pub user: *mut c_void,
    pub alloc: Option<unsafe extern "C" fn(*mut c_void, u64, i32) -> *mut c_void>,
    pub free: Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>,
    pub current_stream: Option<unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void>,
    pub collective: Option<
        unsafe extern "C" fn(
            *mut c_void,
            RsCollectiveKind,
            RsGroupKind,
            *mut c_void,
            *mut c_void,
        ) -> i32,
    >,
    pub log: Option<unsafe extern "C" fn(*mut c_void, i32, *const c_char)>,
}

/// Per-call context handed to `execute`. Mirrors `rs_ctx`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RsCtx {
    pub user: *mut c_void,
    pub svc: *const RsServices,
}

/// The operator descriptor a plugin publishes. Mirrors `rs_op_desc`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RsOpDesc {
    pub abi_version: u32,
    pub struct_size: u32,
    pub id: RsOpId,
    pub doc: *const c_char,
    pub requires: *const RsRequires,
    pub numerics: RsNumerics,
    pub infer: Option<RsInferFn>,
    pub memory: Option<RsMemoryFn>,
    pub expansion: *const RsExpansion,
    pub backward: RsBackwardKind,
    pub backward_op: RsOpId,
    pub collectives: *const RsCollective,
    pub n_collectives: u32,
    pub execute: Option<RsExecuteFn>,
    pub last_error: Option<RsLastErrorFn>,
}

/// The plugin header. Mirrors `rs_plugin`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RsPlugin {
    pub abi_version: u32,
    pub struct_size: u32,
    pub plugin_name: *const c_char,
    pub plugin_version: *const c_char,
    pub n_ops: u32,
    pub _pad: u32,
    pub ops: *const *const RsOpDesc,
    pub init: Option<unsafe extern "C" fn(*mut RsServices) -> i32>,
}

pub type RustrainPluginV1Fn = unsafe extern "C" fn() -> *const RsPlugin;

/// Reads a NUL-terminated C string as `&str`, returning `None` for null.
///
/// # Safety
/// `p` must be null or point to a NUL-terminated string that stays alive for
/// the duration of the returned borrow.
pub(crate) unsafe fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    unsafe { std::ffi::CStr::from_ptr(p) }.to_str().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    /// These sizes are the wire format. They are asserted here so that an
    /// accidental field reorder in either language is caught immediately;
    /// the C plugin test then proves the two agree at run time.
    #[test]
    fn layout() {
        assert_eq!(size_of::<RsTensor>(), 192);
        assert_eq!(size_of::<RsOpId>(), 24);
        assert_eq!(size_of::<RsNumerics>(), 44);
        assert_eq!(size_of::<RsAttr>(), 64);
        assert_eq!(size_of::<RsAttrs>(), 16);
        assert_eq!(size_of::<RsRequires>(), 24);
        assert_eq!(size_of::<RsCollective>(), 16);
        assert_eq!(size_of::<RsMemReq>(), 24);
        assert_eq!(size_of::<RsExpansionNode>(), 48);
        assert_eq!(size_of::<RsExpansion>(), 32);
        assert_eq!(size_of::<RsServices>(), 56);
        assert_eq!(size_of::<RsCtx>(), 16);
        assert_eq!(size_of::<RsOpDesc>(), 184);
        // 8 fields: 4 + 4 + 8 + 8 + 4 + 4 + 8 + 8. (56 is impossible for the
        // frozen header; `rs_plugin` has no field that could pad it to 56.)
        assert_eq!(size_of::<RsPlugin>(), 48);

        assert_eq!(align_of::<RsTensor>(), 8);
        assert_eq!(align_of::<RsOpDesc>(), 8);
    }

    #[test]
    fn field_offsets_match_c() {
        // Offsets that depend on padding rules; pinned once so a reorder is loud.
        assert_eq!(std::mem::offset_of!(RsOpDesc, doc), 32);
        assert_eq!(std::mem::offset_of!(RsOpDesc, numerics), 48);
        assert_eq!(std::mem::offset_of!(RsOpDesc, infer), 96);
        assert_eq!(std::mem::offset_of!(RsOpDesc, execute), 168);
        assert_eq!(std::mem::offset_of!(RsExpansionNode, outputs), 32);
        assert_eq!(std::mem::offset_of!(RsServices, log), 48);
    }

    #[test]
    fn contiguous_strides_and_numel() {
        let t = RsTensor::new(RsDtype::BF16, &[2, 3, 4]);
        assert_eq!(t.strides(), &[12, 4, 1]);
        assert_eq!(t.numel(), 24);
        assert!(t.is_contiguous());
        assert_eq!(t.byte_len(), Some(48));

        let mut s = t;
        s.stride = [24, 4, 1, 0, 0, 0, 0, 0];
        assert!(!s.is_contiguous());
    }

    #[test]
    fn dtype_roundtrip() {
        for d in RsDtype::ALL {
            assert_eq!(RsDtype::parse(d.name()), Some(d));
        }
        assert_eq!(RsDtype::parse("nope"), None);
    }

    #[test]
    fn requires_mask() {
        let r = RsRequires {
            dtype_mask: RsRequires::dtype_mask_for(&[RsDtype::BF16, RsDtype::F32]),
            ..Default::default()
        };
        assert!(r.accepts(RsDtype::BF16));
        assert!(r.accepts(RsDtype::F32));
        assert!(!r.accepts(RsDtype::F8E4M3));
        assert_eq!(r.dtypes(), vec![RsDtype::F32, RsDtype::BF16]);
    }
}
