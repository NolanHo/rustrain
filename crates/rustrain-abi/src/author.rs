//! Helpers for authoring a plugin in Rust.
//!
//! A plugin is a process-lifetime object: it publishes a `&'static RsPlugin`
//! whose descriptors point into memory that is never freed. [`PluginBuilder`]
//! takes care of the pointer stability and the C-string ownership so that
//! plugin authors only write operator bodies.
//!
//! ```ignore
//! #[unsafe(no_mangle)]
//! pub extern "C" fn rustrain_plugin_v1() -> *const RsPlugin {
//!     static PLUGIN: OnceLock<&'static RsPlugin> = OnceLock::new();
//!     *PLUGIN.get_or_init(|| {
//!         PluginBuilder::new("reference", env!("CARGO_PKG_VERSION"))
//!             .op(OpSpec::new("rmsnorm", "reference.f32")
//!                 .dtypes(&[RsDtype::F32])
//!                 .execute(rmsnorm_exec))
//!             .build()
//!     }) as *const RsPlugin
//! }
//! ```

use std::ffi::{CString, c_char, c_void};

use crate::ffi::*;

/// Builds a NUL-terminated `*const c_char` from a string literal.
///
/// Rust string literals are not NUL-terminated, so the naive
/// `s.as_ptr() as *const c_char` is unsound. This macro concatenates a NUL at
/// compile time and is the only sanctioned way to pass literals across the ABI.
#[macro_export]
macro_rules! rs_cstr {
    ($s:literal) => {
        ::core::concat!($s, "\0").as_ptr() as *const ::core::ffi::c_char
    };
}

/// Owns every C string and descriptor a plugin publishes.
///
/// Nothing is ever freed: the plugin lives as long as the process, and the
/// framework may hold descriptors for the whole run. The descriptors are boxed
/// rather than stored inline because their addresses are published as raw
/// pointers — a `Vec` reallocation must not be able to move them.
#[allow(clippy::vec_box)]
#[derive(Default)]
pub struct PluginBuilder {
    name: String,
    version: String,
    docs: Vec<CString>,
    ops: Vec<Box<RsOpDesc>>,
    op_ptrs: Vec<*const RsOpDesc>,
    requires: Vec<Box<RsRequires>>,
    collectives: Vec<Box<[RsCollective]>>,
    expansions: Vec<Box<RsExpansion>>,
    expansion_nodes: Vec<Box<[RsExpansionNode]>>,
    init: Option<unsafe extern "C" fn(*mut RsServices) -> i32>,
}

impl PluginBuilder {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            ..Default::default()
        }
    }

    /// Registers an operator. The builder takes ownership of the descriptor.
    pub fn op(mut self, mut spec: OpSpec) -> Self {
        if let Some(r) = spec.requires.take() {
            let boxed = Box::new(r);
            spec.desc.requires = &*boxed as *const RsRequires;
            self.requires.push(boxed);
        }
        if !spec.collectives.is_empty() {
            let boxed = spec.collectives.into_boxed_slice();
            spec.desc.collectives = boxed.as_ptr();
            spec.desc.n_collectives = boxed.len() as u32;
            self.collectives.push(boxed);
        }
        if let Some(exp) = spec.expansion.take() {
            let (expansion, nodes) = exp.into_raw();
            let nodes = nodes.into_boxed_slice();
            let mut expansion = Box::new(expansion);
            expansion.nodes = nodes.as_ptr();
            expansion.n_nodes = nodes.len() as u32;
            self.expansion_nodes.push(nodes);
            spec.desc.expansion = &*expansion as *const RsExpansion;
            self.expansions.push(expansion);
        }
        let mut boxed = Box::new(spec.desc);
        boxed.abi_version = crate::ABI_VERSION;
        boxed.struct_size = std::mem::size_of::<RsOpDesc>() as u32;
        self.op_ptrs.push(&*boxed as *const RsOpDesc);
        self.ops.push(boxed);
        self
    }

    pub fn init(mut self, f: unsafe extern "C" fn(*mut RsServices) -> i32) -> Self {
        self.init = Some(f);
        self
    }

    /// Leaks the plugin so the returned reference is `'static`.
    ///
    /// # Panics
    /// Panics if a descriptor is missing its name, variant or execute function.
    pub fn build(self) -> &'static RsPlugin {
        for (i, op) in self.ops.iter().enumerate() {
            assert!(
                !op.id.name.is_null(),
                "op #{i} has no name; use OpSpec::new(name, variant)"
            );
            assert!(
                !op.id.variant.is_null(),
                "op #{i} has no variant; use OpSpec::new(name, variant)"
            );
            assert!(
                op.execute.is_some(),
                "op #{i} has no execute function; call OpSpec::execute()"
            );
            assert!(
                op.expansion.is_null() || unsafe { (*op.expansion).n_nodes } > 0,
                "op #{i} declares an empty expansion; omit it or fill it in"
            );
            if op.backward == RsBackwardKind::EXPLICIT {
                assert!(
                    !op.backward_op.name.is_null(),
                    "op #{i} declares EXPLICIT backward without a backward op name"
                );
            }
        }

        let name = c_string(&self.name);
        let version = c_string(&self.version);
        let mut op_ptrs = self.op_ptrs;
        op_ptrs.shrink_to_fit();

        let plugin = Box::new(RsPlugin {
            abi_version: crate::ABI_VERSION,
            struct_size: std::mem::size_of::<RsPlugin>() as u32,
            plugin_name: name,
            plugin_version: version,
            n_ops: op_ptrs.len() as u32,
            _pad: 0,
            ops: op_ptrs.as_ptr(),
            init: self.init,
        });

        // Everything from here on out is intentionally leaked: the descriptors
        // must stay valid for the whole process, and a plugin is loaded once.
        for d in self.docs {
            std::mem::forget(d);
        }
        std::mem::forget(op_ptrs);
        std::mem::forget(self.ops);
        std::mem::forget(self.requires);
        std::mem::forget(self.collectives);
        std::mem::forget(self.expansions);
        std::mem::forget(self.expansion_nodes);
        Box::leak(plugin)
    }
}

fn c_string(s: &str) -> *const c_char {
    let owned = CString::new(s).expect("plugin strings must not contain NUL");
    let ptr = owned.as_ptr();
    std::mem::forget(owned);
    ptr
}

/// Declarative description of one operator, consumed by [`PluginBuilder::op`].
pub struct OpSpec {
    desc: RsOpDesc,
    requires: Option<RsRequires>,
    collectives: Vec<RsCollective>,
    expansion: Option<ExpansionSpec>,
}

impl OpSpec {
    /// `name` / `variant` may be any `&'static str`, not only a literal, so they
    /// go through the runtime `c_string` helper rather than `rs_cstr!`. The copy
    /// is leaked: a descriptor is a process-lifetime object.
    pub fn new(name: &'static str, variant: &'static str) -> Self {
        Self {
            desc: RsOpDesc {
                abi_version: crate::ABI_VERSION,
                struct_size: std::mem::size_of::<RsOpDesc>() as u32,
                id: RsOpId {
                    name: c_string(name),
                    variant: c_string(variant),
                    version: 1,
                },
                doc: rs_cstr!(""),
                requires: std::ptr::null(),
                numerics: RsNumerics::default(),
                infer: None,
                memory: None,
                expansion: std::ptr::null(),
                backward: RsBackwardKind::NONDIFF,
                backward_op: RsOpId::default(),
                collectives: std::ptr::null(),
                n_collectives: 0,
                execute: None,
                last_error: None,
            },
            requires: None,
            collectives: Vec::new(),
            expansion: None,
        }
    }

    pub fn doc(mut self, doc: &'static str) -> Self {
        self.desc.doc = c_string(doc);
        self
    }

    pub fn version(mut self, v: u32) -> Self {
        self.desc.id.version = v;
        self
    }

    /// Declares which element types this implementation accepts.
    pub fn dtypes(mut self, dtypes: &[RsDtype]) -> Self {
        let mut r = self.requires.take().unwrap_or_default();
        r.dtype_mask = RsRequires::dtype_mask_for(dtypes);
        self.requires = Some(r);
        self
    }

    pub fn min_sm(mut self, sm: u32) -> Self {
        let mut r = self.requires.take().unwrap_or_default();
        r.min_sm = sm;
        self.requires = Some(r);
        self
    }

    pub fn min_world_size(mut self, n: i64) -> Self {
        let mut r = self.requires.take().unwrap_or_default();
        r.min_world_size = n;
        self.requires = Some(r);
        self
    }

    pub fn needs_groups(mut self, groups: &[RsGroupKind]) -> Self {
        let mut r = self.requires.take().unwrap_or_default();
        r.needs_groups = groups.iter().fold(0, |acc, g| acc | g.0);
        self.requires = Some(r);
        self
    }

    pub fn numerics(mut self, n: RsNumerics) -> Self {
        self.desc.numerics = n;
        self
    }

    pub fn backward(mut self, kind: RsBackwardKind) -> Self {
        self.desc.backward = kind;
        self
    }

    pub fn backward_op(mut self, name: &'static str, variant: &'static str) -> Self {
        self.desc.backward = RsBackwardKind::EXPLICIT;
        self.desc.backward_op = RsOpId {
            name: c_string(name),
            variant: c_string(variant),
            version: 1,
        };
        self
    }

    pub fn execute(mut self, f: RsExecuteFn) -> Self {
        self.desc.execute = Some(f);
        self
    }

    pub fn infer(mut self, f: RsInferFn) -> Self {
        self.desc.infer = Some(f);
        self
    }

    pub fn memory(mut self, f: RsMemoryFn) -> Self {
        self.desc.memory = Some(f);
        self
    }

    pub fn last_error(mut self, f: RsLastErrorFn) -> Self {
        self.desc.last_error = Some(f);
        self
    }

    pub fn collective(mut self, c: RsCollective) -> Self {
        self.collectives.push(c);
        self
    }

    /// Declares the primitive composition this operator is equivalent to.
    pub fn expansion(mut self, e: ExpansionSpec) -> Self {
        self.expansion = Some(e);
        self
    }
}

/// Owned attributes for ONE expansion node.
///
/// A composite operator's expansion only describes the fused body if each node
/// carries the attributes that select its behaviour — a `reduce` without its
/// `kind`, or a `matmul` without `transpose_b`, does not describe the same
/// computation. Keys, strings and slices are boxed so their addresses survive
/// moving the owning [`ExpansionSpec`].
#[derive(Default)]
pub struct NodeAttrs {
    keys: Vec<CString>,
    strs: Vec<Option<CString>>,
    slices: Vec<Option<Box<[i64]>>>,
    items: Vec<RsAttr>,
}

impl NodeAttrs {
    pub fn new() -> Self {
        Self::default()
    }

    fn blank(kind: RsAttrKind) -> RsAttr {
        RsAttr {
            key: std::ptr::null(),
            kind,
            _pad0: 0,
            i64: 0,
            f64: 0.0,
            boolean: 0,
            _pad1: 0,
            str: std::ptr::null(),
            i64s: std::ptr::null(),
            n_i64s: 0,
            _pad2: 0,
        }
    }

    fn push(mut self, key: &str, mut item: RsAttr) -> Self {
        let k = CString::new(key).expect("attribute key must not contain NUL");
        self.keys.push(k);
        item.key = self.keys.last().unwrap().as_ptr();
        self.items.push(item);
        self
    }

    pub fn i64(self, key: &str, v: i64) -> Self {
        let mut a = Self::blank(RsAttrKind::I64);
        a.i64 = v;
        self.push(key, a)
    }

    pub fn f64(self, key: &str, v: f64) -> Self {
        let mut a = Self::blank(RsAttrKind::F64);
        a.f64 = v;
        self.push(key, a)
    }

    pub fn bool(self, key: &str, v: bool) -> Self {
        let mut a = Self::blank(RsAttrKind::BOOL);
        a.boolean = i32::from(v);
        self.push(key, a)
    }

    pub fn str(self, key: &str, v: &str) -> Self {
        let mut a = Self::blank(RsAttrKind::STR);
        let c = CString::new(v).expect("attribute value must not contain NUL");
        let mut me = self;
        me.strs.push(Some(c));
        a.str = me.strs.last().unwrap().as_ref().unwrap().as_ptr();
        me.push(key, a)
    }

    pub fn i64s(self, key: &str, v: &[i64]) -> Self {
        let mut a = Self::blank(RsAttrKind::I64S);
        let mut me = self;
        me.slices.push(Some(v.to_vec().into_boxed_slice()));
        let s = me.slices.last().unwrap().as_ref().unwrap();
        a.i64s = s.as_ptr();
        a.n_i64s = s.len() as u32;
        me.push(key, a)
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// Builder for a declared primitive expansion (contract R-4).
///
/// Like [`PluginBuilder`], it boxes what it publishes because the nodes carry
/// raw pointers into the owned arrays.
#[allow(clippy::vec_box)]
pub struct ExpansionSpec {
    nodes: Vec<RsExpansionNode>,
    /// C strings referenced by the nodes; kept alive until `into_raw`.
    names: Vec<CString>,
    /// Attribute arrays referenced by the nodes.
    attr_lists: Vec<Box<RsAttrs>>,
    attr_items: Vec<Box<[RsAttr]>>,
    /// Keeps each node's keys, strings and slices alive behind the pointers in
    /// the matching `attr_lists` entry.
    attr_owners: Vec<NodeAttrs>,
    inputs: Vec<Box<[i32]>>,
    outputs: Vec<Box<[i32]>>,
    pub n_tensors: u32,
    pub n_inputs: u32,
    pub n_outputs: u32,
}

impl ExpansionSpec {
    /// `n_inputs` / `n_outputs` describe the *parent* operator's arity.
    /// Local tensor ids follow the convention in `rustrain_op.h`.
    pub fn new(n_inputs: u32, n_outputs: u32) -> Self {
        Self {
            nodes: Vec::new(),
            names: Vec::new(),
            attr_lists: Vec::new(),
            attr_items: Vec::new(),
            attr_owners: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            n_tensors: n_inputs + n_outputs,
            n_inputs,
            n_outputs,
        }
    }

    /// Reserves a temporary local tensor id.
    pub fn temp(&mut self) -> i32 {
        let id = self.n_tensors as i32;
        self.n_tensors += 1;
        id
    }

    /// Appends a primitive node that carries attributes.
    ///
    /// Use this whenever the node's identity depends on an attribute: a
    /// conformance checker that replays an expansion has to receive the same
    /// `kind`, `axis`, `transpose_b` and so on that the fused body used, or it
    /// compares two different computations.
    /// Appends a primitive node that carries attributes.
    ///
    /// Use this whenever the node's identity depends on an attribute: a
    /// conformance checker replaying an expansion has to receive the same
    /// `kind`, `axis`, `transpose_b` and so on that the fused body used, or it
    /// compares two different computations.
    pub fn node_with_attrs(
        mut self,
        op: &str,
        inputs: &[i32],
        outputs: &[i32],
        attrs: NodeAttrs,
    ) -> Self {
        let name = CString::new(op).expect("op name must not contain NUL");
        self.names.push(name);
        let op_ptr = self.names.last().unwrap().as_ptr();

        // The view points into the owner's boxed items; the owner itself is kept
        // in `attr_owners` so the box outlives every use of the pointer.
        let items = attrs.items.clone().into_boxed_slice();
        let view = Box::new(RsAttrs {
            items: items.as_ptr(),
            len: attrs.len() as u32,
            _pad: 0,
        });
        let attrs_ptr = view.as_ref() as *const RsAttrs;
        self.attr_items.push(items);
        self.attr_lists.push(view);
        self.attr_owners.push(attrs);

        let inputs = inputs.to_vec().into_boxed_slice();
        let outputs = outputs.to_vec().into_boxed_slice();
        self.nodes.push(RsExpansionNode {
            op: op_ptr,
            attrs: attrs_ptr,
            inputs: inputs.as_ptr(),
            n_inputs: inputs.len() as u32,
            outputs: outputs.as_ptr(),
            n_outputs: outputs.len() as u32,
        });
        self.inputs.push(inputs);
        self.outputs.push(outputs);
        self
    }

    /// Appends a primitive node over local tensor ids, with no attributes.
    pub fn node(mut self, op: &str, inputs: &[i32], outputs: &[i32]) -> Self {
        let name = CString::new(op).expect("op name must not contain NUL");
        self.names.push(name);
        let op_ptr = self.names.last().unwrap().as_ptr();

        let inputs = inputs.to_vec().into_boxed_slice();
        let outputs = outputs.to_vec().into_boxed_slice();
        self.nodes.push(RsExpansionNode {
            op: op_ptr,
            attrs: std::ptr::null(),
            inputs: inputs.as_ptr(),
            n_inputs: inputs.len() as u32,
            outputs: outputs.as_ptr(),
            n_outputs: outputs.len() as u32,
        });
        self.inputs.push(inputs);
        self.outputs.push(outputs);
        self
    }

    fn into_raw(self) -> (RsExpansion, Vec<RsExpansionNode>) {
        let ExpansionSpec {
            nodes,
            names,
            attr_lists,
            attr_items,
            attr_owners,
            inputs,
            outputs,
            n_tensors,
            n_inputs,
            n_outputs,
        } = self;

        let expansion = RsExpansion {
            n_nodes: nodes.len() as u32,
            _pad: 0,
            nodes: std::ptr::null(),
            n_tensors,
            n_inputs,
            n_outputs,
            _pad2: 0,
        };

        // The caller (PluginBuilder) stores these boxes to keep the pointers
        // live; forgetting them here is handled by the leak in `build`.
        for b in names {
            std::mem::forget(b);
        }
        for b in attr_lists {
            std::mem::forget(b);
        }
        for b in attr_items {
            std::mem::forget(b);
        }
        for b in attr_owners {
            std::mem::forget(b);
        }
        for b in inputs {
            std::mem::forget(b);
        }
        for b in outputs {
            std::mem::forget(b);
        }

        (expansion, nodes)
    }
}

/// Marker so `#[unsafe(no_mangle)]` plugins can name the export type.
pub type PluginEntryFn = unsafe extern "C" fn() -> *const RsPlugin;

/// Convenience: builds an `RsAttrs` view over a borrowed slice.
pub fn attrs_view(items: &[RsAttr]) -> RsAttrs {
    RsAttrs {
        items: items.as_ptr(),
        len: items.len() as u32,
        _pad: 0,
    }
}

/// Reads a `*const c_void` reserved slot as a typed pointer.
///
/// # Safety
/// The slot must actually hold a `T*` written by the same backend.
pub unsafe fn reserved_as<T>(t: &RsTensor, slot: usize) -> Option<&T> {
    let p = *t.reserved.get(slot)?;
    if p.is_null() {
        None
    } else {
        Some(unsafe { &*(p as *const T) })
    }
}

/// Writes a typed pointer into a reserved slot.
pub fn set_reserved<T>(t: &mut RsTensor, slot: usize, value: *mut T) {
    if slot < MAX_RESERVED {
        t.reserved[slot] = value as *mut c_void;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn noop_execute(
        _ctx: *mut RsCtx,
        _in: *const *const RsTensor,
        _n_in: u32,
        _out: *const *mut RsTensor,
        _n_out: u32,
        _attrs: *const RsAttrs,
    ) -> i32 {
        0
    }

    /// `rs_cstr!` exists so that a literal never crosses the boundary without
    /// its NUL; if this regresses, every plugin string is silently over-read.
    #[test]
    fn rs_cstr_is_nul_terminated() {
        let p = rs_cstr!("add");
        // SAFETY: the macro appends the NUL at compile time.
        assert_eq!(unsafe { std::ffi::CStr::from_ptr(p) }.to_str(), Ok("add"));
    }

    #[test]
    fn build_publishes_the_declared_op() {
        let plugin = PluginBuilder::new("reference", "1.2.3")
            .op(
                OpSpec::new("add", "reference.f32")
                    .doc("elementwise sum")
                    .dtypes(&[RsDtype::F32])
                    .execute(noop_execute),
            )
            .build();

        assert_eq!(plugin.abi_version, crate::ABI_VERSION);
        assert_eq!(plugin.struct_size, std::mem::size_of::<RsPlugin>() as u32);
        assert_eq!(plugin.n_ops, 1);
        assert_eq!(unsafe { cstr(plugin.plugin_name) }, Some("reference"));
        assert_eq!(unsafe { cstr(plugin.plugin_version) }, Some("1.2.3"));

        // SAFETY: `build()` published one descriptor and leaked it on purpose.
        let desc = unsafe { &**plugin.ops };
        assert_eq!(desc.abi_version, crate::ABI_VERSION);
        assert_eq!(desc.struct_size, std::mem::size_of::<RsOpDesc>() as u32);
        assert_eq!(unsafe { cstr(desc.id.name) }, Some("add"));
        assert_eq!(unsafe { cstr(desc.id.variant) }, Some("reference.f32"));
        assert_eq!(unsafe { cstr(desc.doc) }, Some("elementwise sum"));
        assert!(desc.execute.is_some());

        let requires = unsafe { desc.requires.as_ref() }.expect("dtypes() declares requires");
        assert!(requires.accepts(RsDtype::F32));
        assert!(!requires.accepts(RsDtype::F16));
    }

    /// `OpSpec::new` takes `&'static str`, not only literals: a plugin may name
    /// its ops from constants, which is why the strings are copied at runtime.
    #[test]
    fn build_accepts_non_literal_op_names() {
        const NAME: &str = "runtime_name";

        let plugin = PluginBuilder::new("t", "0")
            .op(OpSpec::new(NAME, "c").execute(noop_execute))
            .build();

        // SAFETY: `build()` published one descriptor and leaked it on purpose.
        let desc = unsafe { &**plugin.ops };
        assert_eq!(unsafe { cstr(desc.id.name) }, Some("runtime_name"));
    }

    #[test]
    fn build_publishes_a_declared_expansion() {
        let spec = ExpansionSpec::new(2, 1).node("mul", &[0, 1], &[3]);
        let plugin = PluginBuilder::new("t", "0")
            .op(
                OpSpec::new("fused", "c")
                    .execute(noop_execute)
                    .expansion(spec),
            )
            .build();

        // SAFETY: `build()` published one descriptor and leaked it on purpose.
        let desc = unsafe { &**plugin.ops };
        let expansion = unsafe { desc.expansion.as_ref() }.expect("expansion is published");
        assert_eq!(expansion.n_nodes, 1);
        assert_eq!(expansion.n_inputs, 2);
        assert_eq!(expansion.n_outputs, 1);
        assert_eq!(expansion.n_tensors, 3);

        let nodes = unsafe { expansion.as_slice() };
        assert_eq!(nodes.len(), 1);
        assert_eq!(unsafe { cstr(nodes[0].op) }, Some("mul"));
        assert_eq!(unsafe { nodes[0].input_ids() }, &[0, 1]);
        assert_eq!(unsafe { nodes[0].output_ids() }, &[3]);
    }

    #[test]
    #[should_panic(expected = "has no execute function")]
    fn build_rejects_an_op_without_execute() {
        let _ = PluginBuilder::new("t", "0")
            .op(OpSpec::new("add", "c"))
            .build();
    }

    #[test]
    #[should_panic(expected = "declares an empty expansion")]
    fn build_rejects_an_empty_expansion() {
        let _ = PluginBuilder::new("t", "0")
            .op(
                OpSpec::new("add", "c")
                    .execute(noop_execute)
                    .expansion(ExpansionSpec::new(2, 1)),
            )
            .build();
    }

    #[test]
    #[should_panic(expected = "EXPLICIT backward without a backward op name")]
    fn build_rejects_explicit_backward_without_an_op_name() {
        let _ = PluginBuilder::new("t", "0")
            .op(
                OpSpec::new("add", "c")
                    .execute(noop_execute)
                    .backward(RsBackwardKind::EXPLICIT),
            )
            .build();
    }

    /// An expansion node's attributes must reach the plugin side intact, and
    /// must still be readable after the owning spec has been moved into a plugin.
    #[test]
    fn expansion_node_attributes_survive_publication() {
        let expansion = ExpansionSpec::new(2, 1)
            .node_with_attrs(
                "reduce",
                &[0],
                &[2],
                NodeAttrs::new().str("kind", "sum").i64("axis", -1),
            )
            .node_with_attrs(
                "matmul",
                &[2, 1],
                &[3],
                NodeAttrs::new().bool("transpose_b", true).f64("beta", 0.5),
            );

        let plugin = PluginBuilder::new("attrs", "0.1.0")
            .op(
                OpSpec::new("fused", "test.f32")
                    .dtypes(&[RsDtype::F32])
                    .execute(noop_execute)
                    .expansion(expansion),
            )
            .build();

        let desc = unsafe { &**plugin.ops };
        let expansion = unsafe { &*desc.expansion };
        let nodes = unsafe { expansion.as_slice() };
        assert_eq!(nodes.len(), 2);

        let read = |node: &RsExpansionNode, key: &str| -> Option<RsAttr> {
            assert!(!node.attrs.is_null(), "node must carry an attribute list");
            let list = unsafe { (*node.attrs).as_slice() };
            list.iter()
                .find(|a| {
                    !a.key.is_null()
                        && unsafe { std::ffi::CStr::from_ptr(a.key) }.to_str().ok() == Some(key)
                })
                .copied()
        };

        let kind = read(&nodes[0], "kind").expect("kind on the reduce node");
        assert_eq!(kind.kind, RsAttrKind::STR);
        assert_eq!(
            unsafe { std::ffi::CStr::from_ptr(kind.str) }.to_str().unwrap(),
            "sum"
        );

        let axis = read(&nodes[0], "axis").expect("axis on the reduce node");
        assert_eq!(axis.i64, -1);

        let tb = read(&nodes[1], "transpose_b").expect("transpose_b on the matmul node");
        assert_eq!(tb.boolean, 1);

        let beta = read(&nodes[1], "beta").expect("beta on the matmul node");
        assert_eq!(beta.f64, 0.5);
    }
}
