//! Conformance tests for the `reference` provider.
//!
//! The descriptors are plain Rust, so the operator bodies are driven
//! directly through the published function pointers instead of dlopen: every
//! test runs `infer` to size the outputs, allocates caller-provided buffers,
//! and then calls `execute`.

use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;
use std::slice;

use rustrain_abi::author::attrs_view;
use rustrain_abi::ffi::*;
use rustrain_kernels::plugin;

// ── harness ─────────────────────────────────────────────────────────────────

/// One tensor with its live buffer. Exactly one buffer is populated,
/// depending on the descriptor's dtype.
struct Owned {
    t: RsTensor,
    f: Vec<f32>,
    b: Vec<u8>,
    i32s: Vec<i32>,
    i64s: Vec<i64>,
}

impl Owned {
    fn f32(shape: &[i64], v: Vec<f32>) -> Self {
        let mut t = RsTensor::new(RsDtype::F32, shape);
        assert_eq!(v.len() as i64, t.numel(), "bad test data for {shape:?}");
        t.data = v.as_ptr() as *mut c_void;
        Self {
            t,
            f: v,
            b: vec![],
            i32s: vec![],
            i64s: vec![],
        }
    }

    fn u8(dtype: RsDtype, shape: &[i64], v: Vec<u8>) -> Self {
        let mut t = RsTensor::new(dtype, shape);
        assert_eq!(v.len() as i64, t.numel());
        t.data = v.as_ptr() as *mut c_void;
        Self {
            t,
            f: vec![],
            b: v,
            i32s: vec![],
            i64s: vec![],
        }
    }

    fn i32(shape: &[i64], v: Vec<i32>) -> Self {
        let mut t = RsTensor::new(RsDtype::I32, shape);
        assert_eq!(v.len() as i64, t.numel());
        t.data = v.as_ptr() as *mut c_void;
        Self {
            t,
            f: vec![],
            b: vec![],
            i32s: v,
            i64s: vec![],
        }
    }

    fn i64(shape: &[i64], v: Vec<i64>) -> Self {
        let mut t = RsTensor::new(RsDtype::I64, shape);
        assert_eq!(v.len() as i64, t.numel());
        t.data = v.as_ptr() as *mut c_void;
        Self {
            t,
            f: vec![],
            b: vec![],
            i32s: vec![],
            i64s: v,
        }
    }

    /// Allocates a zeroed buffer shaped by an `infer`-produced descriptor.
    fn zeros_for(d: &RsTensor) -> Self {
        let n = d.numel().max(0) as usize;
        match d.dtype {
            RsDtype::F32 => {
                let mut t = *d;
                let v = vec![0.0f32; n];
                t.data = v.as_ptr() as *mut c_void;
                Self {
                    t,
                    f: v,
                    b: vec![],
                    i32s: vec![],
                    i64s: vec![],
                }
            }
            RsDtype::F8E4M3 | RsDtype::F8E5M2 | RsDtype::U8 => {
                let mut t = *d;
                let v = vec![0u8; n];
                t.data = v.as_ptr() as *mut c_void;
                Self {
                    t,
                    f: vec![],
                    b: v,
                    i32s: vec![],
                    i64s: vec![],
                }
            }
            RsDtype::I32 => {
                let mut t = *d;
                let v = vec![0i32; n];
                t.data = v.as_ptr() as *mut c_void;
                Self {
                    t,
                    f: vec![],
                    b: vec![],
                    i32s: v,
                    i64s: vec![],
                }
            }
            RsDtype::I64 => {
                let mut t = *d;
                let v = vec![0i64; n];
                t.data = v.as_ptr() as *mut c_void;
                Self {
                    t,
                    f: vec![],
                    b: vec![],
                    i32s: vec![],
                    i64s: v,
                }
            }
            other => panic!("no zero buffer for {other}"),
        }
    }

    fn fdata(&self) -> &[f32] {
        &self.f
    }

    /// The live buffer as bytes.
    fn bytes(&self) -> &[u8] {
        match self.t.dtype {
            RsDtype::F32 => unsafe {
                slice::from_raw_parts(self.f.as_ptr() as *const u8, self.f.len() * 4)
            },
            RsDtype::F8E4M3 | RsDtype::F8E5M2 | RsDtype::U8 => &self.b,
            RsDtype::I32 => unsafe {
                slice::from_raw_parts(self.i32s.as_ptr() as *const u8, self.i32s.len() * 4)
            },
            RsDtype::I64 => unsafe {
                slice::from_raw_parts(self.i64s.as_ptr() as *const u8, self.i64s.len() * 8)
            },
            _ => &[],
        }
    }
}

/// A NUL-terminated C string, leaked (tests are short-lived).
fn cstr(s: &str) -> *const c_char {
    CString::new(s).unwrap().into_raw()
}

fn ai64(key: &str, v: i64) -> RsAttr {
    RsAttr {
        key: cstr(key),
        kind: RsAttrKind::I64,
        _pad0: 0,
        i64: v,
        f64: 0.0,
        boolean: 0,
        _pad1: 0,
        str: ptr::null(),
        i64s: ptr::null(),
        n_i64s: 0,
        _pad2: 0,
    }
}

fn af64(key: &str, v: f64) -> RsAttr {
    RsAttr {
        key: cstr(key),
        kind: RsAttrKind::F64,
        _pad0: 0,
        i64: 0,
        f64: v,
        boolean: 0,
        _pad1: 0,
        str: ptr::null(),
        i64s: ptr::null(),
        n_i64s: 0,
        _pad2: 0,
    }
}

fn astr(key: &str, v: &str) -> RsAttr {
    RsAttr {
        key: cstr(key),
        kind: RsAttrKind::STR,
        _pad0: 0,
        i64: 0,
        f64: 0.0,
        boolean: 0,
        _pad1: 0,
        str: cstr(v),
        i64s: ptr::null(),
        n_i64s: 0,
        _pad2: 0,
    }
}

fn abool(key: &str, v: bool) -> RsAttr {
    RsAttr {
        key: cstr(key),
        kind: RsAttrKind::BOOL,
        _pad0: 0,
        i64: 0,
        f64: 0.0,
        boolean: v as i32,
        _pad1: 0,
        str: ptr::null(),
        i64s: ptr::null(),
        n_i64s: 0,
        _pad2: 0,
    }
}

fn ai64s(key: &str, v: &[i64]) -> RsAttr {
    let v: &'static [i64] = Box::leak(v.to_vec().into_boxed_slice());
    RsAttr {
        key: cstr(key),
        kind: RsAttrKind::I64S,
        _pad0: 0,
        i64: 0,
        f64: 0.0,
        boolean: 0,
        _pad1: 0,
        str: ptr::null(),
        i64s: v.as_ptr(),
        n_i64s: v.len() as u32,
        _pad2: 0,
    }
}

fn op(name: &str) -> &'static RsOpDesc {
    let p = plugin();
    let ops = unsafe { slice::from_raw_parts(p.ops, p.n_ops as usize) };
    ops.iter()
        .find(|d| {
            unsafe { CStr::from_ptr((&*(**d)).id.name) }
                .to_str()
                .unwrap()
                == name
        })
        .map(|d| unsafe { &**d })
        .unwrap_or_else(|| panic!("op '{name}' not registered"))
}

unsafe fn last_err(o: &'static RsOpDesc) -> String {
    unsafe {
        let f = o.last_error.unwrap();
        let p = f(ptr::null_mut());
        if p.is_null() {
            "(no message)".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

unsafe fn call_infer(
    o: &'static RsOpDesc,
    ins: &[&RsTensor],
    outs: &mut [RsTensor],
    attrs: &[RsAttr],
) -> Result<(), String> {
    unsafe {
        let a = attrs_view(attrs);
        let inptrs: Vec<*const RsTensor> = ins.iter().map(|t| *t as *const RsTensor).collect();
        let mut outptrs: Vec<*mut RsTensor> = outs.iter_mut().map(|t| t as *mut RsTensor).collect();
        let st = (o.infer.unwrap())(
            inptrs.as_ptr(),
            ins.len() as u32,
            outptrs.as_mut_ptr(),
            outs.len() as u32,
            &a,
        );
        if st == 0 { Ok(()) } else { Err(last_err(o)) }
    }
}

unsafe fn call_exec(
    o: &'static RsOpDesc,
    ins: &[&RsTensor],
    outs: &mut [&mut RsTensor],
    attrs: &[RsAttr],
) -> Result<(), String> {
    unsafe {
        let a = attrs_view(attrs);
        let inptrs: Vec<*const RsTensor> = ins.iter().map(|t| *t as *const RsTensor).collect();
        let mut outptrs: Vec<*mut RsTensor> =
            outs.iter_mut().map(|t| *t as *mut RsTensor).collect();
        let st = (o.execute.unwrap())(
            ptr::null_mut(),
            inptrs.as_ptr(),
            ins.len() as u32,
            outptrs.as_mut_ptr(),
            outs.len() as u32,
            &a,
        );
        if st == 0 { Ok(()) } else { Err(last_err(o)) }
    }
}

/// infer -> allocate -> execute, returning the filled outputs.
unsafe fn run_op(
    o: &'static RsOpDesc,
    ins: &[&RsTensor],
    attrs: &[RsAttr],
    n_out: usize,
) -> Result<Vec<Owned>, String> {
    let mut descs: Vec<RsTensor> = (0..n_out).map(|_| RsTensor::default()).collect();
    unsafe { call_infer(o, ins, &mut descs, attrs) }?;
    let mut outs: Vec<Owned> = descs.iter().map(Owned::zeros_for).collect();
    let mut out_refs: Vec<&mut RsTensor> = outs.iter_mut().map(|x| &mut x.t).collect();
    unsafe { call_exec(o, ins, &mut out_refs, attrs) }?;
    Ok(outs)
}

fn assert_close(a: f32, b: f32, tol: f32, what: &str) {
    assert!(
        (a - b).abs() <= tol,
        "{what}: got {a}, expected {b} (tol {tol})"
    );
}

/// Reads a (possibly strided) view descriptor against the base buffer it
/// aliases, walking indices by the declared strides.
fn collect_view(t: &RsTensor, base: &[f32]) -> Vec<f32> {
    let dims: Vec<usize> = t.dims().iter().map(|&d| d as usize).collect();
    let strides: Vec<usize> = t.strides().iter().map(|&x| x as usize).collect();
    let mut out = Vec::new();
    fn walk(
        d: usize,
        acc: usize,
        dims: &[usize],
        strides: &[usize],
        base: &[f32],
        out: &mut Vec<f32>,
    ) {
        if d == dims.len() {
            out.push(base[acc]);
            return;
        }
        for i in 0..dims[d] {
            walk(d + 1, acc + i * strides[d], dims, strides, base, out);
        }
    }
    walk(0, 0, &dims, &strides, base, &mut out);
    out
}

// ── plugin metadata ─────────────────────────────────────────────────────────

#[test]
fn plugin_loads_through_the_abi_loader() {
    // The crate publishes a cdylib next to the rlib; drive it through
    // rustrain-abi's real dlopen loader (contract C-1) to prove the plugin
    // is actually loadable, not just in-process callable.
    let exe = std::env::current_exe().unwrap();
    let dir = exe.parent().unwrap(); // the test binary lives in deps/, next to the cdylib
    let so = dir.join(format!(
        "{}rustrain_kernels{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ));
    assert!(so.exists(), "cdylib not built at {so:?}");

    // SAFETY: the library is our own cdylib built from this crate; the
    // loader keeps it mapped for the handle's lifetime.
    let plugin = unsafe { rustrain_abi::Plugin::load(&so, None) }
        .unwrap_or_else(|e| panic!("loader rejected {so:?}: {e}"));
    assert_eq!(plugin.name(), "reference");
    let ops = plugin.ops();
    assert_eq!(ops.len(), 32);
    assert_eq!(ops[0].spec_name(), "view@reference.f32");
    assert_eq!(ops[11].spec_name(), "compare@reference.f32");
    assert_eq!(ops[26].spec_name(), "topk_router@reference.f32");
    assert_eq!(ops[27].spec_name(), "l2norm@reference.f32");
    assert_eq!(ops[28].spec_name(), "rmsnorm_gated@reference.f32");
    assert_eq!(ops[29].spec_name(), "causal_conv1d@reference.f32");
    assert_eq!(ops[30].spec_name(), "gated_delta_rule@reference.f32");
    assert_eq!(ops[31].spec_name(), "moe_layer@reference.f32");
    for o in &ops {
        assert!(o.variant() == "reference.f32", "{}: variant", o.name());
    }
}

#[test]
fn plugin_publishes_the_expected_ops() {
    let p = plugin();
    assert_eq!(p.abi_version, rustrain_abi::ABI_VERSION);
    assert_eq!(
        unsafe { CStr::from_ptr(p.plugin_name) }.to_str().unwrap(),
        "reference"
    );
    assert_eq!(
        unsafe { CStr::from_ptr(p.plugin_version) }
            .to_str()
            .unwrap(),
        env!("CARGO_PKG_VERSION")
    );
    let ops = unsafe { slice::from_raw_parts(p.ops, p.n_ops as usize) };
    assert_eq!(p.n_ops, 32, "one variant per vocabulary op");
    let names: Vec<&str> = ops
        .iter()
        .map(|d| unsafe { CStr::from_ptr((**d).id.name) }.to_str().unwrap())
        .collect();
    assert_eq!(
        names,
        vec![
            "view",
            "reshape",
            "transpose",
            "narrow",
            "cat",
            "broadcast",
            "matmul",
            "linear",
            "bmm",
            "elementwise_unary",
            "elementwise_binary",
            "compare",
            "reduce",
            "softmax",
            "rmsnorm",
            "layernorm",
            "rope",
            "quantize",
            "dequantize",
            "amax_update",
            "embedding",
            "gather",
            "scatter",
            "sdpa",
            "cross_entropy",
            "adamw",
            "topk_router",
            "l2norm",
            "rmsnorm_gated",
            "causal_conv1d",
            "gated_delta_rule",
            "moe_layer",
        ]
    );
    for d in ops {
        let desc = unsafe { &**d };
        let name = unsafe { CStr::from_ptr(desc.id.name) }.to_str().unwrap();
        let variant = unsafe { CStr::from_ptr(desc.id.variant) }.to_str().unwrap();
        // Every op must carry the full planning surface: infer, memory,
        // execute, last_error.
        assert!(desc.infer.is_some(), "{name}: infer");
        assert!(desc.memory.is_some(), "{name}: memory");
        assert!(desc.execute.is_some(), "{name}: execute");
        assert!(desc.last_error.is_some(), "{name}: last_error");
        assert_eq!(desc.abi_version, rustrain_abi::ABI_VERSION);
        // The expected spec_name (name@variant).
        assert_eq!(variant, "reference.f32", "{name}: variant");
        // Composites must declare an expansion (contract R-4); primitives
        // must not.
        let composite = matches!(name, "sdpa" | "cross_entropy" | "adamw" | "topk_router");
        assert_eq!(
            !desc.expansion.is_null(),
            composite,
            "{name}: expansion presence"
        );
    }
}

#[test]
fn sdpa_memory_reports_its_scratch() {
    let o = op("sdpa");
    let q = Owned::f32(&[1, 3, 4], vec![0.0; 12]);
    let k = Owned::f32(&[1, 5, 4], vec![0.0; 20]);
    let v = Owned::f32(&[1, 5, 2], vec![0.0; 10]);
    let mut req = RsMemReq::default();
    let io: Vec<*const RsTensor> = vec![&q.t, &k.t, &v.t];
    let st = unsafe { (o.memory.unwrap())(io.as_ptr(), io.len() as u32, ptr::null(), &mut req) };
    assert_eq!(st, 0);
    // Two f32 scratch buffers of S*T = 3*5 elements each.
    assert_eq!(req.workspace_bytes, 8 * 3 * 5);
    assert_eq!(req.save_for_backward_bytes, 0);
}

#[test]
fn gated_delta_rule_memory_reports_its_scratch() {
    // vh=1, d=dv=2, S=4, chunk_size=2 -> s2=4, nc=2. The reported f32 count
    // is vh*(s2*(3d+3dv+3) + nc*c*(3c+d+dv) + d*dv) + c*dv + d*dv
    // = 1*(4*15 + 2*2*10 + 4) + 4 + 4 = 112 f32 = 448 bytes — the exact
    // single buffer the chunked body carves.
    let o = op("gated_delta_rule");
    let q = Owned::f32(&[1, 4, 2], vec![0.0; 8]);
    let v = Owned::f32(&[1, 4, 2], vec![0.0; 8]);
    let g = Owned::f32(&[1, 4, 1], vec![0.0; 4]);
    let mut req = RsMemReq::default();
    let io: Vec<*const RsTensor> = vec![&q.t, &q.t, &v.t, &g.t, &g.t];
    let attrs = attrs_view(&[ai64("chunk_size", 2)]);
    let st = unsafe { (o.memory.unwrap())(io.as_ptr(), io.len() as u32, &attrs, &mut req) };
    assert_eq!(st, 0);
    assert_eq!(req.workspace_bytes, 448);
    assert_eq!(req.save_for_backward_bytes, 0);
}

#[test]
fn memory_reports_zero_workspace_for_every_op() {
    let p = plugin();
    let ops = unsafe { slice::from_raw_parts(p.ops, p.n_ops as usize) };
    for d in ops {
        let desc = unsafe { &**d };
        let mut req = RsMemReq {
            workspace_bytes: u64::MAX,
            save_for_backward_bytes: u64::MAX,
            save_tensor_count: 7,
            _pad: 0,
        };
        let st = unsafe { (desc.memory.unwrap())(ptr::null(), 0, ptr::null(), &mut req) };
        assert_eq!(st, 0);
        assert_eq!(req.workspace_bytes, 0);
        assert_eq!(req.save_for_backward_bytes, 0);
        assert_eq!(req.save_tensor_count, 0);
    }
}

#[test]
fn expansions_are_structurally_sound() {
    let p = plugin();
    let ops = unsafe { slice::from_raw_parts(p.ops, p.n_ops as usize) };
    for d in ops {
        let desc = unsafe { &**d };
        let name = unsafe { CStr::from_ptr(desc.id.name) }.to_str().unwrap();
        if desc.expansion.is_null() {
            continue;
        }
        let e = unsafe { &*desc.expansion };
        assert!(e.n_nodes > 0, "{name}: non-empty expansion");
        let nodes = unsafe { e.as_slice() };
        // Local tensor ids must live in [0, n_tensors); inputs are the
        // parent's [0, n_inputs), outputs [n_inputs, n_inputs+n_outputs).
        for n in nodes {
            for &i in unsafe { n.input_ids() } {
                assert!(i >= 0 && (i as u32) < e.n_tensors, "{name}: input id {i}");
            }
            for &o in unsafe { n.output_ids() } {
                assert!(o >= 0 && (o as u32) < e.n_tensors, "{name}: output id {o}");
            }
        }
        // Every parent output must be produced by at least one node (R-4:
        // a truthful expansion describes where each output comes from). The
        // documented exception is topk_router: the top-k selection itself has
        // no primitive in spec §2.4, so its expansion covers only the
        // gating-probability path and its two outputs are uncovered.
        if name != "topk_router" {
            let mut produced = vec![false; e.n_outputs as usize];
            for n in nodes {
                for &o in unsafe { n.output_ids() } {
                    let o = o as u32;
                    if o >= e.n_inputs && o < e.n_inputs + e.n_outputs {
                        produced[(o - e.n_inputs) as usize] = true;
                    }
                }
            }
            assert!(
                produced.iter().all(|&p| p),
                "{name}: parent outputs not all produced by the expansion"
            );
        }
        match name {
            "sdpa" => {
                assert_eq!((e.n_inputs, e.n_outputs, e.n_nodes), (3, 1, 3));
                let ops_of: Vec<&str> = nodes
                    .iter()
                    .map(|n| unsafe { CStr::from_ptr(n.op) }.to_str().unwrap())
                    .collect();
                assert_eq!(ops_of, ["bmm", "softmax", "bmm"]);
                assert_eq!(unsafe { nodes[2].output_ids() }, &[3]); // parent output 0
            }
            "cross_entropy" => {
                assert_eq!((e.n_inputs, e.n_outputs, e.n_nodes), (2, 1, 6));
                let ops_of: Vec<&str> = nodes
                    .iter()
                    .map(|n| unsafe { CStr::from_ptr(n.op) }.to_str().unwrap())
                    .collect();
                assert_eq!(
                    ops_of,
                    [
                        "softmax",
                        "elementwise_unary",
                        "reshape",
                        "gather",
                        "elementwise_unary",
                        "reduce"
                    ]
                );
                assert_eq!(unsafe { nodes[5].output_ids() }, &[2]); // parent output 0
            }
            "adamw" => {
                assert_eq!((e.n_inputs, e.n_outputs), (4, 3));
                assert_eq!(e.n_nodes, 16);
                assert_eq!(unsafe { nodes[2].output_ids() }, &[5]); // m'
                assert_eq!(unsafe { nodes[6].output_ids() }, &[6]); // v'
                assert_eq!(unsafe { nodes[15].output_ids() }, &[4]); // p'
            }
            "topk_router" => {
                assert_eq!((e.n_inputs, e.n_outputs, e.n_nodes), (1, 2, 1));
                assert_eq!(
                    unsafe { CStr::from_ptr(nodes[0].op) }.to_str().unwrap(),
                    "softmax"
                );
            }
            other => panic!("unexpected expansion on {other}"),
        }
    }
}

// ── fused ≈ declared expansion (contract R-4) ──────────────────────────────
//
// These execute the declared expansion node-by-node with the per-node
// attributes documented in the expansion builders, and compare against the
// fused body. The current ExpansionSpec API cannot attach those attributes
// to the nodes themselves (an rustrain-abi gap reported to the parent), so
// the checker of the future will need them supplied the same way.

#[test]
fn sdpa_fused_matches_its_declared_expansion() {
    let q = Owned::f32(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let k = Owned::f32(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let v = Owned::f32(&[1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let fused = unsafe { run_op(op("sdpa"), &[&q.t, &k.t, &v.t], &[], 1) }.unwrap();
    // bmm(q, k, transpose_b=true) -> softmax -> bmm(p, v)
    let s0 = unsafe { run_op(op("bmm"), &[&q.t, &k.t], &[abool("transpose_b", true)], 1) }.unwrap();
    let p = unsafe { run_op(op("softmax"), &[&s0[0].t], &[], 1) }.unwrap();
    let o = unsafe { run_op(op("bmm"), &[&p[0].t, &v.t], &[], 1) }.unwrap();
    assert_close(
        o[0].fdata()[0],
        fused[0].fdata()[0],
        1e-6,
        "sdpa fused vs expansion [0]",
    );
    assert_close(
        o[0].fdata()[3],
        fused[0].fdata()[3],
        1e-6,
        "sdpa fused vs expansion [3]",
    );
}

#[test]
fn cross_entropy_fused_matches_its_declared_expansion() {
    let logits = Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    let targets = Owned::i32(&[2], vec![2, 0]);
    let fused = unsafe { run_op(op("cross_entropy"), &[&logits.t, &targets.t], &[], 1) }.unwrap();
    // softmax -> log -> reshape(targets, [-1,1]) -> gather(axis=-1) -> neg ->
    // reduce(mean, all axes)
    let p = unsafe { run_op(op("softmax"), &[&logits.t], &[ai64("axis", -1)], 1) }.unwrap();
    let lp = unsafe {
        run_op(
            op("elementwise_unary"),
            &[&p[0].t],
            &[astr("kind", "log")],
            1,
        )
    }
    .unwrap();
    let t2 =
        unsafe { run_op(op("reshape"), &[&targets.t], &[ai64s("shape", &[-1, 1])], 1) }.unwrap();
    let per =
        unsafe { run_op(op("gather"), &[&lp[0].t, &t2[0].t], &[ai64("axis", -1)], 1) }.unwrap();
    let neg = unsafe {
        run_op(
            op("elementwise_unary"),
            &[&per[0].t],
            &[astr("kind", "neg")],
            1,
        )
    }
    .unwrap();
    let loss = unsafe { run_op(op("reduce"), &[&neg[0].t], &[astr("kind", "mean")], 1) }.unwrap();
    assert_close(
        loss[0].fdata()[0],
        fused[0].fdata()[0],
        1e-6,
        "cross_entropy fused vs expansion",
    );
}

#[test]
fn adamw_fused_matches_its_declared_expansion() {
    // Defaults: lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0, step=1.
    let p0 = Owned::f32(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let g0 = Owned::f32(&[4], vec![0.5, -0.5, 0.25, -0.25]);
    let m0 = Owned::f32(&[4], vec![0.0; 4]);
    let v0 = Owned::f32(&[4], vec![0.0; 4]);
    let fused = unsafe { run_op(op("adamw"), &[&p0.t, &g0.t, &m0.t, &v0.t], &[], 3) }.unwrap();
    // The 16 documented expansion steps.
    let eb = |ins: &[&RsTensor], attrs: &[RsAttr]| -> Owned {
        let o = unsafe { run_op(op("elementwise_binary"), ins, attrs, 1) }.unwrap();
        o.into_iter().next().unwrap()
    };
    let eu = |ins: &[&RsTensor], attrs: &[RsAttr]| -> Owned {
        let o = unsafe { run_op(op("elementwise_unary"), ins, attrs, 1) }.unwrap();
        o.into_iter().next().unwrap()
    };
    let t7 = eb(&[&g0.t], &[astr("kind", "mul"), af64("rhs", 1.0 - 0.9)]); // g*(1-b1)
    let t8 = eb(&[&m0.t], &[astr("kind", "mul"), af64("rhs", 0.9)]); // m*b1
    let mp = eb(&[&t8.t, &t7.t], &[astr("kind", "add")]); // m'
    let t9 = eb(&[&g0.t, &g0.t], &[astr("kind", "mul")]); // g*g
    let t10 = eb(&[&t9.t], &[astr("kind", "mul"), af64("rhs", 1.0 - 0.999)]); // g^2*(1-b2)
    let t11 = eb(&[&v0.t], &[astr("kind", "mul"), af64("rhs", 0.999)]); // v*b2
    let vp = eb(&[&t11.t, &t10.t], &[astr("kind", "add")]); // v'
    let t12 = eb(&[&mp.t], &[astr("kind", "mul"), af64("rhs", 10.0)]); // mhat
    let t13 = eb(&[&vp.t], &[astr("kind", "mul"), af64("rhs", 1000.0)]); // vhat
    let t14 = eu(&[&t13.t], &[astr("kind", "sqrt")]); // sqrt(vhat)
    let t15 = eb(&[&t14.t], &[astr("kind", "add"), af64("rhs", 1e-8)]); // + eps
    let t16 = eb(&[&t12.t, &t15.t], &[astr("kind", "div")]); // mhat/den
    let t17 = eb(&[&t16.t], &[astr("kind", "mul"), af64("rhs", 1e-3)]); // * lr
    let t18 = eb(&[&p0.t], &[astr("kind", "mul"), af64("rhs", 0.0)]); // p*lr*wd
    let t19 = eb(&[&p0.t, &t17.t], &[astr("kind", "sub")]); // p - step
    let pp = eb(&[&t19.t, &t18.t], &[astr("kind", "sub")]); // p'
    for i in 0..4 {
        assert_close(pp.fdata()[i], fused[0].fdata()[i], 1e-5, "adamw param");
        assert_close(mp.fdata()[i], fused[1].fdata()[i], 1e-7, "adamw exp_avg");
        assert_close(vp.fdata()[i], fused[2].fdata()[i], 1e-7, "adamw exp_avg_sq");
    }
}

// ── hand-computed values, op by op ──────────────────────────────────────────

#[test]
fn view_aliases_without_copy() {
    let x = Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let outs = unsafe { run_op(op("view"), &[&x.t], &[], 1) }.unwrap();
    let o = &outs[0];
    assert_eq!(o.t.dims(), &[2, 3]);
    assert_eq!(o.t.strides(), &[3, 1]);
    assert_eq!(o.t.data, x.t.data, "view must alias the input buffer");
    assert_eq!(
        collect_view(&o.t, x.fdata()),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn reshape_reinterprets_and_aliases() {
    let x = Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let outs = unsafe { run_op(op("reshape"), &[&x.t], &[ai64s("shape", &[3, 2])], 1) }.unwrap();
    let o = &outs[0];
    assert_eq!(o.t.dims(), &[3, 2]);
    assert_eq!(o.t.strides(), &[2, 1]);
    assert_eq!(o.t.data, x.t.data);
    assert_eq!(
        collect_view(&o.t, x.fdata()),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    // One -1 is inferred from numel.
    let outs = unsafe { run_op(op("reshape"), &[&x.t], &[ai64s("shape", &[6, -1])], 1) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[6, 1]);
}

#[test]
fn transpose_swaps_shape_and_strides() {
    let x = Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let outs = unsafe {
        run_op(
            op("transpose"),
            &[&x.t],
            &[ai64("dim0", 0), ai64("dim1", 1)],
            1,
        )
    }
    .unwrap();
    let o = &outs[0];
    assert_eq!(o.t.dims(), &[3, 2]);
    assert_eq!(o.t.strides(), &[1, 3]);
    assert_eq!(o.t.data, x.t.data);
    // Read through the aliased descriptor: out[i][j] = data[i*stride0 + j*stride1].
    assert_eq!(
        collect_view(&o.t, x.fdata()),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

/// `reshape` of a strided view copies the LOGICAL order, not the memory order.
///
/// A `narrow` produces exactly this shape of descriptor (a slice of a flat
/// projection: [rows, width] with the row stride of the whole projection), and
/// `reshape` then groups it into per-head blocks. Reinterpreting memory instead
/// of the logical order would read the neighbouring segments instead — values
/// that are plausible, wrong, and invisible to a shape check. Before this, the
/// reference provider refused the input outright, so the plan that needs it
/// could not run at all.
#[test]
fn reshape_materialises_a_strided_view_in_logical_order() {
    let base = Owned::f32(&[2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let mut strided = base.t;
    strided.shape = [2, 2, 0, 0, 0, 0, 0, 0];
    strided.stride = [4, 1, 0, 0, 0, 0, 0, 0];
    assert!(!strided.is_contiguous());
    let outs = unsafe { run_op(op("reshape"), &[&strided], &[ai64s("shape", &[4])], 1) }.unwrap();
    let o = &outs[0];
    assert_eq!(o.t.dims(), &[4]);
    // The view reads (0,0)=1, (0,1)=2, (1,0)=5, (1,1)=6 — not 1,2,3,4.
    assert_eq!(o.fdata(), &[1.0, 2.0, 5.0, 6.0]);
}

/// Rope's positions live on the FIRST axis, so a `[seq, heads, head_dim]` input
/// must turn each head of a position with that position's angle.
///
/// The pre-D5 loop indexed `(batch * seq + position)`, which puts the position on
/// the *middle* axis: for the plan's `[512, 16, 256]` q/k it rotated a scrambled
/// mix of positions and heads, and every full-attention layer inherited the wrong
/// angles. The conformance case is rank-2 (`R = 1`), where the two index orders
/// agree — which is why the gate never saw it.
#[test]
fn rope_uses_the_position_of_the_first_axis_for_every_head() {
    let theta = 1e7f64;
    let (s, heads, d, rotary) = (3usize, 2usize, 4usize, 4usize);
    let mut data = vec![0.0f32; s * heads * d];
    data[heads * d] = 1.0; // [seq = 1, head = 0, dim = 0]
    let x = Owned::f32(&[s as i64, heads as i64, d as i64], data);
    let outs = unsafe {
        run_op(
            op("rope"),
            &[&x.t, &x.t],
            &[ai64("rotary_dim", rotary as i64)],
            2,
        )
    }
    .unwrap();
    let o = &outs[0];
    // Row `t * heads + head` of the flattened buffer carries position `t`, so
    // [seq = 1, head = 0] is row 2 and its first element sits at 2 * d. Only
    // dim 0 is non-zero, so the pair it rotates with (dim h) is the only other
    // element that moves.
    let base = heads * d; // row (seq=1, head=0)
    let h = rotary / 2;
    let angle = 1.0 * (theta as f32).powf(0.0) as f64; // j = 0: inv_freq = theta^0 = 1
    let (want_cos, want_sin) = (angle.cos() as f32, angle.sin() as f32);
    assert!(
        (o.fdata()[base] - want_cos).abs() < 1e-6,
        "dim 0 rotated by {} but position 1's cos is {want_cos}",
        o.fdata()[base]
    );
    assert!(
        (o.fdata()[base + h] - want_sin).abs() < 1e-6,
        "dim h rotated by {} but position 1's sin is {want_sin}",
        o.fdata()[base + h]
    );
    for j in 1..h {
        assert_eq!(o.fdata()[base + j], 0.0, "j={j}: a zero dim moved");
        assert_eq!(o.fdata()[base + h + j], 0.0, "j={j}: a zero dim moved");
    }
    // Position 0's angle is cos = 1, which is what a wrong index order applies
    // here (it reads the position off the middle axis instead of the first).
    assert!(
        (o.fdata()[base] - 1.0).abs() > 1e-3,
        "the element was rotated as if it belonged to position 0"
    );
    // Nothing else in position 1's head moved.
    for d in 0..d {
        if d != 0 && d != h {
            assert_eq!(o.fdata()[base + d], 0.0, "dim {d} should be untouched");
        }
    }
    let _ = h;
}

#[test]
fn narrow_offsets_the_data_pointer() {
    let x = Owned::f32(&[3, 4], (1..=12).map(|v| v as f32).collect());
    let outs = unsafe {
        run_op(
            op("narrow"),
            &[&x.t],
            &[ai64("dim", 1), ai64("start", 1), ai64("length", 2)],
            1,
        )
    }
    .unwrap();
    let o = &outs[0];
    assert_eq!(o.t.dims(), &[3, 2]);
    assert_eq!(o.t.strides(), &[4, 1]);
    assert_eq!(o.t.data, unsafe { x.t.data.add(4) });
    assert_eq!(
        collect_view(&o.t, &x.fdata()[1..]),
        vec![2.0, 3.0, 6.0, 7.0, 10.0, 11.0]
    );
}

#[test]
fn cat_copies_in_input_order() {
    let a = Owned::f32(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Owned::f32(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    // dim 0
    let outs = unsafe { run_op(op("cat"), &[&a.t, &b.t], &[ai64("dim", 0)], 1) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[4, 2]);
    assert_eq!(outs[0].fdata(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    // dim -1 (defaults to the last axis)
    let outs = unsafe { run_op(op("cat"), &[&a.t, &b.t], &[ai64("dim", -1)], 1) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[2, 4]);
    assert_eq!(outs[0].fdata(), &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    // default dim is -1
    let outs = unsafe { run_op(op("cat"), &[&a.t, &b.t], &[], 1) }.unwrap();
    assert_eq!(outs[0].fdata(), &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn broadcast_uses_zero_strides() {
    let x = Owned::f32(&[3, 1], vec![1.0, 2.0, 3.0]);
    let outs = unsafe { run_op(op("broadcast"), &[&x.t], &[ai64s("shape", &[3, 4])], 1) }.unwrap();
    let o = &outs[0];
    assert_eq!(o.t.dims(), &[3, 4]);
    assert_eq!(o.t.strides(), &[1, 0]);
    assert_eq!(o.t.data, x.t.data);
    // Read through the aliased descriptor (stride 0 repeats the element).
    assert_eq!(
        collect_view(&o.t, x.fdata()),
        vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    );
}

#[test]
fn matmul_hand_values() {
    let a = Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Owned::f32(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let outs = unsafe { run_op(op("matmul"), &[&a.t, &b.t], &[], 1) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[2, 2]);
    assert_eq!(outs[0].fdata(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn linear_hand_values_with_and_without_bias() {
    let x = Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let w = Owned::f32(&[3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let b = Owned::f32(&[2], vec![10.0, 20.0]);
    let outs = unsafe { run_op(op("linear"), &[&x.t, &w.t, &b.t], &[], 1) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[2, 2]);
    assert_eq!(outs[0].fdata(), &[14.0, 25.0, 20.0, 31.0]);
    let outs = unsafe { run_op(op("linear"), &[&x.t, &w.t], &[], 1) }.unwrap();
    assert_eq!(outs[0].fdata(), &[4.0, 5.0, 10.0, 11.0]);
}

#[test]
fn matmul_and_bmm_transpose_b_hand_values() {
    // a @ b^T with a [2,2] = [[1,2],[3,4]], b [2,2] = [[5,6],[7,8]]:
    // [[1*5+2*6, 1*7+2*8],[3*5+4*6, 3*7+4*8]] = [[17,23],[39,53]].
    let a = Owned::f32(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Owned::f32(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let outs = unsafe {
        run_op(
            op("matmul"),
            &[&a.t, &b.t],
            &[abool("transpose_b", true)],
            1,
        )
    }
    .unwrap();
    assert_eq!(outs[0].fdata(), &[17.0, 23.0, 39.0, 53.0]);

    // Batched form: batch 0 as above; batch 1 = 2a @ b^T = [[34,46],[78,106]].
    let a = Owned::f32(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0]);
    let b = Owned::f32(&[2, 2, 2], vec![5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0]);
    let outs =
        unsafe { run_op(op("bmm"), &[&a.t, &b.t], &[abool("transpose_b", true)], 1) }.unwrap();
    assert_eq!(
        outs[0].fdata(),
        &[17.0, 23.0, 39.0, 53.0, 34.0, 46.0, 78.0, 106.0]
    );
}

#[test]
fn bmm_hand_values() {
    let a = Owned::f32(&[2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
    let b = Owned::f32(&[2, 2, 2], vec![1.0, 1.0, 1.0, 1.0, 3.0, 0.0, 0.0, 3.0]);
    let outs = unsafe { run_op(op("bmm"), &[&a.t, &b.t], &[], 1) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[2, 2, 2]);
    assert_eq!(outs[0].fdata(), &[1.0, 1.0, 1.0, 1.0, 6.0, 0.0, 0.0, 6.0]);
}

#[test]
fn elementwise_unary_hand_values() {
    let x = Owned::f32(&[4], vec![-2.0, -1.0, 0.0, 1.0]);
    let run = |kind: &str| {
        let outs =
            unsafe { run_op(op("elementwise_unary"), &[&x.t], &[astr("kind", kind)], 1) }.unwrap();
        outs[0].fdata().to_vec()
    };
    assert_eq!(run("relu"), vec![0.0, 0.0, 0.0, 1.0]);
    assert_eq!(run("neg"), vec![2.0, 1.0, 0.0, -1.0]);
    let sqrt = run("sqrt");
    assert!(
        sqrt[0].is_nan() && sqrt[1].is_nan(),
        "sqrt of negatives is NaN"
    );
    assert_eq!(&sqrt[2..], &[0.0, 1.0]);
    let sig = run("sigmoid");
    assert_close(sig[2], 0.5, 1e-7, "sigmoid(0)");
    assert_close(sig[3], 0.731_058_6, 1e-6, "sigmoid(1)");
    let silu = run("silu");
    assert_close(silu[2], 0.0, 1e-7, "silu(0)");
    assert_close(silu[3], 0.731_058_6, 1e-6, "silu(1) = sigmoid(1)");
    let tanh = run("tanh");
    assert_close(tanh[3], 0.761_594_2, 1e-6, "tanh(1)");
    let gelu = run("gelu");
    assert_close(gelu[2], 0.0, 1e-7, "gelu(0)");
    let exp = run("exp");
    assert_close(exp[3], std::f32::consts::E, 1e-6, "exp(1)");
    let log = run("log");
    assert_close(log[3], 0.0, 1e-7, "ln(1)");

    // rsqrt = 1/sqrt(x): IEEE at the edges.
    let rsqrt = run("rsqrt");
    assert!(
        rsqrt[0].is_nan() && rsqrt[1].is_nan(),
        "rsqrt of negatives is NaN"
    );
    assert_eq!(rsqrt[2], f32::INFINITY, "rsqrt(0) = +inf");
    assert_eq!(rsqrt[3], 1.0, "rsqrt(1)");
    let x4 = Owned::f32(&[1], vec![4.0]);
    let outs = unsafe {
        run_op(
            op("elementwise_unary"),
            &[&x4.t],
            &[astr("kind", "rsqrt")],
            1,
        )
    }
    .unwrap();
    assert_eq!(outs[0].fdata(), &[0.5], "rsqrt(4) = 1/2");

    // Hand-computed *_grad values at the sample points (the central finite
    // difference check below is the real contract).
    let sig_g = run("sigmoid_grad");
    assert_close(sig_g[2], 0.25, 1e-7, "sigmoid_grad(0) = σ(1-σ)");
    assert_close(sig_g[3], 0.196_611_94, 1e-6, "sigmoid_grad(1)");
    let tanh_g = run("tanh_grad");
    assert_close(tanh_g[2], 1.0, 1e-7, "tanh_grad(0)");
    assert_close(
        tanh_g[3],
        0.419_974_34,
        1e-6,
        "tanh_grad(1) = 1 - tanh(1)^2",
    );
    let relu_g = run("relu_grad");
    assert_eq!(
        relu_g,
        vec![0.0, 0.0, 0.0, 1.0],
        "relu_grad: 0 at 0 by convention"
    );
    let silu_g = run("silu_grad");
    assert_close(silu_g[2], 0.5, 1e-7, "silu_grad(0)");
    assert_close(silu_g[3], 0.927_670_5, 1e-6, "silu_grad(1)");
    let gelu_g = run("gelu_grad");
    assert_close(
        gelu_g[2],
        0.5,
        1e-7,
        "gelu_grad(0) of the tanh approximation",
    );
}

/// The `*_grad` kinds exist so VJPs are writable; their entire contract is
/// that they equal the derivative of the provider's own forward function. So
/// they are checked against a central finite difference of that forward op —
/// not against a separately hardcoded derivative.
#[test]
fn unary_grad_kinds_match_central_finite_difference() {
    let h = 1e-3f32;
    let xs = [-4.0, -2.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.0, 4.0];
    for (fwd, grad) in [
        ("silu", "silu_grad"),
        ("gelu", "gelu_grad"),
        ("sigmoid", "sigmoid_grad"),
        ("tanh", "tanh_grad"),
        ("relu", "relu_grad"),
    ] {
        for &x in &xs {
            // relu is not differentiable at 0: the subgradient convention is
            // 0 while a central difference reads 0.5, so skip that point.
            if fwd == "relu" && x == 0.0 {
                continue;
            }
            let xp = Owned::f32(&[1], vec![x + h]);
            let xm = Owned::f32(&[1], vec![x - h]);
            let fp = unsafe { run_op(op("elementwise_unary"), &[&xp.t], &[astr("kind", fwd)], 1) }
                .unwrap();
            let fm = unsafe { run_op(op("elementwise_unary"), &[&xm.t], &[astr("kind", fwd)], 1) }
                .unwrap();
            let x0 = Owned::f32(&[1], vec![x]);
            let g = unsafe { run_op(op("elementwise_unary"), &[&x0.t], &[astr("kind", grad)], 1) }
                .unwrap();
            // Central difference in f64 so the FD itself is not the limiting
            // error; the tolerance covers f32 rounding of the two forward
            // calls (two ~1e-7 absolute roundings over a 2h = 2e-3 step).
            let fd = (fp[0].fdata()[0] as f64 - fm[0].fdata()[0] as f64) / (2.0 * h as f64);
            assert_close(
                g[0].fdata()[0],
                fd as f32,
                2e-3,
                &format!("{grad} vs central difference of {fwd} at x={x}"),
            );
        }
    }
}

#[test]
fn elementwise_binary_hand_values() {
    let a = Owned::f32(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Owned::f32(&[2, 2], vec![10.0, 20.0, 30.0, 40.0]);
    let run = |kind: &str| {
        let outs = unsafe {
            run_op(
                op("elementwise_binary"),
                &[&a.t, &b.t],
                &[astr("kind", kind)],
                1,
            )
        }
        .unwrap();
        outs[0].fdata().to_vec()
    };
    assert_eq!(run("add"), vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(run("sub"), vec![-9.0, -18.0, -27.0, -36.0]);
    assert_eq!(run("mul"), vec![10.0, 40.0, 90.0, 160.0]);
    assert_eq!(run("div"), vec![0.1, 0.1, 0.1, 0.1]);
    assert_eq!(run("maximum"), vec![10.0, 20.0, 30.0, 40.0]);

    // pow: x^y elementwise; 0^0 = 1 per IEEE.
    let base = Owned::f32(&[4], vec![2.0, 3.0, 4.0, 0.0]);
    let exp = Owned::f32(&[4], vec![3.0, 2.0, 0.5, 0.0]);
    let outs = unsafe {
        run_op(
            op("elementwise_binary"),
            &[&base.t, &exp.t],
            &[astr("kind", "pow")],
            1,
        )
    }
    .unwrap();
    let got = outs[0].fdata();
    for (i, want) in [8.0f32, 9.0, 2.0, 1.0].iter().enumerate() {
        assert_close(got[i], *want, 1e-6, "pow hand value");
    }

    // Broadcasting: [2,1] + [1,3].
    let a = Owned::f32(&[2, 1], vec![1.0, 2.0]);
    let b = Owned::f32(&[1, 3], vec![10.0, 20.0, 30.0]);
    let outs = unsafe {
        run_op(
            op("elementwise_binary"),
            &[&a.t, &b.t],
            &[astr("kind", "add")],
            1,
        )
    }
    .unwrap();
    assert_eq!(outs[0].t.dims(), &[2, 3]);
    assert_eq!(outs[0].fdata(), &[11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);

    // Scalar form: y = x * rhs.
    let x = Owned::f32(&[3], vec![1.0, 2.0, 3.0]);
    let outs = unsafe {
        run_op(
            op("elementwise_binary"),
            &[&x.t],
            &[astr("kind", "mul"), af64("rhs", 2.0)],
            1,
        )
    }
    .unwrap();
    assert_eq!(outs[0].fdata(), &[2.0, 4.0, 6.0]);
}

#[test]
fn compare_all_kinds_and_nan_policy() {
    // Mask convention: exactly 1.0 where the comparison holds, 0.0 elsewhere.
    let a = Owned::f32(&[4], vec![1.0, 2.0, 2.0, 0.0]);
    let b = Owned::f32(&[4], vec![1.0, 2.0, 3.0, -1.0]);
    let run = |kind: &str| {
        let outs =
            unsafe { run_op(op("compare"), &[&a.t, &b.t], &[astr("kind", kind)], 1) }.unwrap();
        assert_eq!(outs[0].t.dims(), &[4]);
        assert_eq!(outs[0].t.dtype, RsDtype::F32);
        outs[0].fdata().to_vec()
    };
    assert_eq!(run("eq"), vec![1.0, 1.0, 0.0, 0.0]);
    assert_eq!(run("ne"), vec![0.0, 0.0, 1.0, 1.0]);
    assert_eq!(run("lt"), vec![0.0, 0.0, 1.0, 0.0]);
    assert_eq!(run("le"), vec![1.0, 1.0, 1.0, 0.0]);
    assert_eq!(run("gt"), vec![0.0, 0.0, 0.0, 1.0]);
    assert_eq!(run("ge"), vec![1.0, 1.0, 0.0, 1.0]);

    // NaN policy (documented): any NaN operand yields 0.0 for EVERY kind —
    // a NaN never smuggles a 1 into a mask, not even through 'ne'.
    let nan = Owned::f32(&[1], vec![f32::NAN]);
    let one = Owned::f32(&[1], vec![1.0]);
    for kind in ["eq", "ne", "lt", "le", "gt", "ge"] {
        let outs =
            unsafe { run_op(op("compare"), &[&nan.t, &one.t], &[astr("kind", kind)], 1) }.unwrap();
        assert_eq!(outs[0].fdata(), &[0.0], "NaN on the left, kind {kind}");
        let outs =
            unsafe { run_op(op("compare"), &[&one.t, &nan.t], &[astr("kind", kind)], 1) }.unwrap();
        assert_eq!(outs[0].fdata(), &[0.0], "NaN on the right, kind {kind}");
    }
}

#[test]
fn compare_shape_mismatch_is_a_hard_error_naming_both_shapes() {
    unsafe {
        let a = Owned::f32(&[4], vec![1.0; 4]);
        let b = Owned::f32(&[2, 2], vec![1.0; 4]);
        let mut out = RsTensor::default();
        let err = call_exec(
            op("compare"),
            &[&a.t, &b.t],
            &mut [&mut out],
            &[astr("kind", "eq")],
        )
        .unwrap_err();
        assert!(
            err.contains("[4]") && err.contains("[2, 2]"),
            "shape mismatch message must name both shapes: {err}"
        );
        // infer must reject the same pair.
        let mut outs = [RsTensor::default()];
        let err = call_infer(
            op("compare"),
            &[&a.t, &b.t],
            &mut outs,
            &[astr("kind", "eq")],
        )
        .unwrap_err();
        assert!(
            err.contains("[4]") && err.contains("[2, 2]"),
            "infer message: {err}"
        );
    }
}

#[test]
fn reduce_hand_values() {
    let x = Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let run = |kind: &str, axis: Option<i64>| {
        let mut attrs = vec![astr("kind", kind)];
        if let Some(ax) = axis {
            attrs.push(ai64("axis", ax));
        }
        let mut outs = unsafe { run_op(op("reduce"), &[&x.t], &attrs, 1) }.unwrap();
        outs.remove(0)
    };
    assert_eq!(run("sum", Some(0)).fdata(), &[5.0, 7.0, 9.0]);
    assert_eq!(run("sum", Some(1)).fdata(), &[6.0, 15.0]);
    assert_eq!(run("sum", Some(-1)).fdata(), &[6.0, 15.0]);
    assert_eq!(run("mean", Some(-1)).fdata(), &[2.0, 5.0]);
    assert_eq!(run("max", Some(-1)).fdata(), &[3.0, 6.0]);
    assert_eq!(run("amax", Some(0)).fdata(), &[4.0, 5.0, 6.0]);
    // No axis: reduce everything to a rank-0 scalar.
    let all = run("sum", None);
    assert_eq!(all.t.rank, 0);
    assert_eq!(all.t.numel(), 1);
    assert_eq!(all.fdata(), &[21.0]);
    assert_eq!(run("mean", None).fdata(), &[3.5]);
    assert_eq!(run("max", None).fdata(), &[6.0]);
    // All-negative max must not inherit the +0 start.
    let neg = Owned::f32(&[2], vec![-5.0, -2.0]);
    let outs = unsafe { run_op(op("reduce"), &[&neg.t], &[astr("kind", "max")], 1) }.unwrap();
    assert_eq!(outs[0].fdata(), &[-2.0]);

    // keepdim = true: the reduced axis survives as size 1 (the softmax/
    // layernorm VJPs need the rank preserved); values agree with the
    // axis-removed run after squeezing.
    let run_kd = |kind: &str, axis: i64| {
        let outs = unsafe {
            run_op(
                op("reduce"),
                &[&x.t],
                &[
                    astr("kind", kind),
                    ai64("axis", axis),
                    abool("keepdim", true),
                ],
                1,
            )
        }
        .unwrap();
        outs.into_iter().next().unwrap()
    };
    let kd0 = run_kd("sum", 0);
    assert_eq!(kd0.t.dims(), &[1, 3]);
    assert_eq!(kd0.fdata(), &[5.0, 7.0, 9.0]);
    let kd1 = run_kd("sum", 1);
    assert_eq!(kd1.t.dims(), &[2, 1]);
    assert_eq!(kd1.fdata(), &[6.0, 15.0]);
    assert_eq!(
        run("sum", Some(0)).fdata(),
        kd0.fdata(),
        "squeezed values agree"
    );
    assert_eq!(
        run("sum", Some(1)).fdata(),
        kd1.fdata(),
        "squeezed values agree"
    );
    // Negative axis with keepdim.
    let kdm1 = run_kd("mean", -1);
    assert_eq!(kdm1.t.dims(), &[2, 1]);
    assert_eq!(kdm1.fdata(), &[2.0, 5.0]);
    // No axis + keepdim: every dim becomes 1 (torch convention), value the
    // same as the rank-0 scalar.
    let all_kd = unsafe {
        run_op(
            op("reduce"),
            &[&x.t],
            &[astr("kind", "sum"), abool("keepdim", true)],
            1,
        )
    }
    .unwrap();
    assert_eq!(all_kd[0].t.dims(), &[1, 1]);
    assert_eq!(all_kd[0].fdata(), &[21.0]);
}

#[test]
fn softmax_hand_values() {
    let x = Owned::f32(&[3], vec![1.0, 2.0, 3.0]);
    let outs = unsafe { run_op(op("softmax"), &[&x.t], &[], 1) }.unwrap();
    // Reference computed in the same f32 order: e1+e2+e3 ascending.
    let (e1, e2, e3) = (1f32.exp(), 2f32.exp(), 3f32.exp());
    let sum = e1 + e2 + e3;
    assert_close(outs[0].fdata()[0], e1 / sum, 1e-6, "softmax[0]");
    assert_close(outs[0].fdata()[1], e2 / sum, 1e-6, "softmax[1]");
    assert_close(outs[0].fdata()[2], e3 / sum, 1e-6, "softmax[2]");

    // Axis 0 over a [2, 3] input: per column.
    let x2 = Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let outs = unsafe { run_op(op("softmax"), &[&x2.t], &[ai64("axis", 0)], 1) }.unwrap();
    let r = outs[0].fdata();
    for col in 0..3 {
        let (a, b) = (x2.fdata()[col], x2.fdata()[3 + col]);
        let (ea, eb) = (a.exp(), b.exp());
        assert_close(r[col], ea / (ea + eb), 1e-6, "axis0 col");
        assert_close(r[3 + col], eb / (ea + eb), 1e-6, "axis0 col");
    }

    // Explicit scale attribute.
    let outs = unsafe { run_op(op("softmax"), &[&x.t], &[af64("scale", 0.5)], 1) }.unwrap();
    let (e1, e2, e3) = (0.5f32.exp(), 1f32.exp(), 1.5f32.exp());
    let sum = e1 + e2 + e3;
    assert_close(outs[0].fdata()[0], e1 / sum, 1e-6, "scaled softmax");
}

#[test]
fn rmsnorm_hand_values() {
    let x = Owned::f32(&[2], vec![3.0, 4.0]);
    let outs = unsafe { run_op(op("rmsnorm"), &[&x.t], &[], 1) }.unwrap();
    let r = ((9.0f32 + 16.0) / 2.0 + 1e-5).sqrt();
    assert_close(outs[0].fdata()[0], 3.0 / r, 1e-6, "rmsnorm[0]");
    assert_close(outs[0].fdata()[1], 4.0 / r, 1e-6, "rmsnorm[1]");

    // With weight.
    let w = Owned::f32(&[2], vec![2.0, 3.0]);
    let outs = unsafe { run_op(op("rmsnorm"), &[&x.t, &w.t], &[], 1) }.unwrap();
    assert_close(outs[0].fdata()[0], 6.0 / r, 1e-6, "weighted rmsnorm[0]");
    assert_close(outs[0].fdata()[1], 12.0 / r, 1e-6, "weighted rmsnorm[1]");
}

#[test]
fn rmsnorm_weight_offset_is_the_declared_weight_convention() {
    // The trunk convention: y = x/r * (1 + w). weight_offset=1.0 turns the
    // raw weight [2, 3] into the effective weight [3, 4], so with
    // x = [3, 4] and r = sqrt((9+16)/2 + 1e-5):
    //   y[0] = (3/r) * (2+1) = 9/r,  y[1] = (4/r) * (3+1) = 16/r.
    let x = Owned::f32(&[2], vec![3.0, 4.0]);
    let w = Owned::f32(&[2], vec![2.0, 3.0]);
    let outs = unsafe {
        run_op(
            op("rmsnorm"),
            &[&x.t, &w.t],
            &[af64("weight_offset", 1.0)],
            1,
        )
    }
    .unwrap();
    let r = ((9.0f32 + 16.0) / 2.0 + 1e-5).sqrt();
    assert_close(outs[0].fdata()[0], 9.0 / r, 1e-6, "offset rmsnorm[0]");
    assert_close(outs[0].fdata()[1], 16.0 / r, 1e-6, "offset rmsnorm[1]");
}

#[test]
fn l2norm_hand_values() {
    // y = x / sqrt(sum(x^2) + eps), SUM not mean, eps inside the sqrt.
    // x = [3, 4, 0]: sum(x^2) = 25, r = sqrt(25 + 1e-6) ~ 5.
    let x = Owned::f32(&[3], vec![3.0, 4.0, 0.0]);
    let outs = unsafe { run_op(op("l2norm"), &[&x.t], &[], 1) }.unwrap();
    let r = (25.0f32 + 1e-6).sqrt();
    assert_close(outs[0].fdata()[0], 3.0 / r, 1e-6, "l2norm[0]");
    assert_close(outs[0].fdata()[1], 4.0 / r, 1e-6, "l2norm[1]");
    assert_close(outs[0].fdata()[2], 0.0, 1e-6, "l2norm[2]");

    // dim=0 over [[3,4],[4,3]]: column 0 = (3,4) has norm 5, column 1 =
    // (4,3) has norm 5, so both columns normalise to (0.6, 0.8).
    let x = Owned::f32(&[2, 2], vec![3.0, 4.0, 4.0, 3.0]);
    let outs = unsafe { run_op(op("l2norm"), &[&x.t], &[ai64("dim", 0)], 1) }.unwrap();
    let expected = [3.0 / 5.0, 4.0 / 5.0, 4.0 / 5.0, 3.0 / 5.0];
    for (i, (got, want)) in outs[0].fdata().iter().zip(&expected).enumerate() {
        assert_close(*got, *want, 1e-6, &format!("l2norm dim0[{i}]"));
    }
}

#[test]
fn rmsnorm_gated_hand_values() {
    // y = (x / sqrt(mean(x^2) + eps)) * w * silu(gate), the RAW weight
    // convention (offset 0.0). x = [3, 4], w = [2, 3], gate = [0, 1]:
    //   r  = sqrt((9+16)/2 + 1e-6) = sqrt(12.5 + 1e-6)
    //   y0 = (3/r) * 2 * silu(0) = 0                       (silu(0) = 0)
    //   y1 = (4/r) * 3 * silu(1), silu(1) = 1/(1+e^-1) = 0.7310586
    let x = Owned::f32(&[2], vec![3.0, 4.0]);
    let w = Owned::f32(&[2], vec![2.0, 3.0]);
    let gate = Owned::f32(&[2], vec![0.0, 1.0]);
    let outs = unsafe {
        run_op(
            op("rmsnorm_gated"),
            &[&x.t, &w.t, &gate.t],
            &[af64("eps", 1e-6)],
            1,
        )
    }
    .unwrap();
    let r = ((9.0f32 + 16.0) / 2.0 + 1e-6).sqrt();
    let silu1 = 1.0 / (1.0 + (-1.0f32).exp());
    assert_close(outs[0].fdata()[0], 0.0, 1e-6, "gated[0]");
    assert_close(
        outs[0].fdata()[1],
        (4.0 / r) * 3.0 * silu1,
        1e-5,
        "gated[1]",
    );
}

#[test]
fn causal_conv1d_hand_values() {
    // Depthwise causal conv, kernel K=3, pad defaults to K-1=2 (left-only).
    // x = [1, 2, 3] (L=3, C=1), w = [1, 2, 4]:
    //   out[0] = w0*x[-2] + w1*x[-1] + w2*x[0] = 0 + 0 + 4*1 = 4
    //   out[1] = w0*x[-1] + w1*x[0]  + w2*x[1] = 0 + 2*1 + 4*2 = 10
    //   out[2] = w0*x[0]  + w1*x[1]  + w2*x[2] = 1*1 + 2*2 + 4*3 = 17
    let x = Owned::f32(&[3, 1], vec![1.0, 2.0, 3.0]);
    let w = Owned::f32(&[1, 1, 3], vec![1.0, 2.0, 4.0]);
    let attrs = [ai64("kernel", 3)];
    let outs = unsafe { run_op(op("causal_conv1d"), &[&x.t, &w.t], &attrs, 1) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[3, 1]);
    assert_eq!(outs[0].fdata(), &[4.0, 10.0, 17.0]);

    // The fused silu runs in the same sweep: silu([4, 10, 17]).
    let attrs = [ai64("kernel", 3), astr("activation", "silu")];
    let outs = unsafe { run_op(op("causal_conv1d"), &[&x.t, &w.t], &attrs, 1) }.unwrap();
    let silu = |v: f32| v / (1.0 + (-v).exp());
    assert_eq!(outs[0].fdata(), &[silu(4.0), silu(10.0), silu(17.0)]);
}

#[test]
fn gated_delta_rule_hand_values_single_chunk() {
    // One chunk (S=2, chunk_size=2), vh=1, D=Dv=2. q = k = identity,
    // v = [[1,0],[2,0]], g = 0 (decay 1), beta = 1.
    // The query scale is D^-0.5 = 1/sqrt(2), so q' = I/sqrt(2).
    // cum = [0,0] -> pairwise = [[1,0],[1,1]]; ut[i][j] = (k_i . k_j) * pw:
    // k is the identity, so ut = [[1,0],[0,1]] (the strictly-lower part is 0);
    // intra = (q' @ k^T)∘pw = [[1,0],[0,1]]/sqrt(2).
    // Solve (I + strictly_lower(ut)) x = v_beta:
    //   x0 = v[0] = [1,0]; x1 = v[1] - 0 = [2,0]  -> new_values = [[1,0],[2,0]].
    // Scan with S = 0: out = intra @ new_values =
    //   row0 = [1/sqrt(2), 0]; row1 = [0, 2/sqrt(2)].
    // Per-token check (the defining order, o_t = q_t^T S_t after the update):
    //   S_0 = k0 ⊗ (v0 * 1) = [[1,0],[0,0]]; o_0 = q0·S_0 = [1/sqrt(2), 0]
    //   S_1 = S_0 + k1 ⊗ ((v1 - S_0 k1) * 1) = [[1,0],[0,2]];
    //   o_1 = q1·S_1 = [0, 2/sqrt(2)]. Same numbers.
    let q = Owned::f32(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let k = Owned::f32(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let v = Owned::f32(&[1, 2, 2], vec![1.0, 0.0, 2.0, 0.0]);
    let g = Owned::f32(&[1, 2, 1], vec![0.0, 0.0]);
    let beta = Owned::f32(&[1, 2, 1], vec![1.0, 1.0]);
    let outs = unsafe {
        run_op(
            op("gated_delta_rule"),
            &[&q.t, &k.t, &v.t, &g.t, &beta.t],
            &[ai64("chunk_size", 2), astr("state_dtype", "f32")],
            1,
        )
    }
    .unwrap();
    assert_eq!(outs[0].t.dims(), &[1, 2, 2]);
    let inv2 = 2.0f32.powf(-0.5);
    // Output rows are [vh*Dv] = 2 per position: t0 = [1/sqrt(2), 0],
    // t1 = [2/sqrt(2), 0].
    assert_close(outs[0].fdata()[0], 1.0 * inv2, 1e-6, "gdn[0]");
    assert_close(outs[0].fdata()[1], 0.0, 1e-6, "gdn[1]");
    assert_close(outs[0].fdata()[2], 2.0 * inv2, 1e-6, "gdn[2]");
    assert_close(outs[0].fdata()[3], 0.0, 1e-6, "gdn[3]");
}

#[test]
fn gated_delta_rule_hand_values_two_chunks_with_decay_and_gating() {
    // Two chunks (S=4, chunk_size=2), vh=1, D=1, Dv=1, q = k = 1,
    // v = [8,8,8,8], g = ln(0.5) everywhere (decay 0.5), beta = 0.5.
    // Per-token reference (S_t = S_{t-1}*0.5 + ((v_t - S_{t-1}*0.5) * 0.5),
    // o_t = S_t — read AFTER the update):
    //   t0: S = 0*0.5 = 0;        delta = (8-0)*0.5 = 4;   S = 4;     o0 = 4
    //   t1: S = 4*0.5 = 2;        delta = (8-2)*0.5 = 3;   S = 5;     o1 = 5
    //   t2: S = 5*0.5 = 2.5;      delta = (8-2.5)*0.5 = 2.75; S = 5.25; o2 = 5.25
    //   t3: S = 5.25*0.5 = 2.625; delta = (8-2.625)*0.5 = 2.6875; S = 5.3125; o3 = 5.3125
    // The chunked body must produce the same numbers (the chunk boundary
    // carries the state through the fp32 scan; all values are exact dyadics).
    let q = Owned::f32(&[1, 4, 1], vec![1.0, 1.0, 1.0, 1.0]);
    let k = Owned::f32(&[1, 4, 1], vec![1.0, 1.0, 1.0, 1.0]);
    let v = Owned::f32(&[1, 4, 1], vec![8.0, 8.0, 8.0, 8.0]);
    let ln05 = 0.5f32.ln();
    let g = Owned::f32(&[1, 4, 1], vec![ln05, ln05, ln05, ln05]);
    let beta = Owned::f32(&[1, 4, 1], vec![0.5, 0.5, 0.5, 0.5]);
    let outs = unsafe {
        run_op(
            op("gated_delta_rule"),
            &[&q.t, &k.t, &v.t, &g.t, &beta.t],
            &[ai64("chunk_size", 2)],
            1,
        )
    }
    .unwrap();
    assert_eq!(outs[0].t.dims(), &[1, 4, 1]);
    let expected = [4.0, 5.0, 5.25, 5.3125];
    for (i, (got, want)) in outs[0].fdata().iter().zip(&expected).enumerate() {
        assert_close(*got, *want, 1e-5, &format!("gdn decay[{i}]"));
    }
}

#[test]
fn gated_delta_rule_repeats_qk_heads_for_gqa() {
    // vh=2, D=Dv=2, q/k last dim 2 -> kh=1: the single q/k head is
    // repeat_interleaved by 2. With g=0, beta=1 and k = identity, each value
    // head sees the same key: head 0 reads v0 = [1,0],[2,0], head 1 reads
    // v1 = [3,0],[4,0]. Per the single-chunk arithmetic, head h's output is
    // [[v_h[0] / sqrt(2), 0], [0, v_h[1] / sqrt(2)]].
    let q = Owned::f32(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let k = Owned::f32(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    // v[t][h][j] = v[t*4 + h*2 + j]: head0 = (1,0),(2,0); head1 = (3,0),(4,0).
    let v = Owned::f32(&[1, 2, 4], vec![1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0]);
    let g = Owned::f32(&[1, 2, 2], vec![0.0, 0.0, 0.0, 0.0]);
    let beta = Owned::f32(&[1, 2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let outs = unsafe {
        run_op(
            op("gated_delta_rule"),
            &[&q.t, &k.t, &v.t, &g.t, &beta.t],
            &[],
            1,
        )
    }
    .unwrap();
    assert_eq!(outs[0].t.dims(), &[1, 2, 4]);
    let inv2 = 2.0f32.powf(-0.5);
    // Output rows are [vh*Dv] = 4 per position: t0 = [h0, h1] =
    // [1/sqrt(2), 0, 3/sqrt(2), 0], t1 = [2/sqrt(2), 0, 4/sqrt(2), 0].
    let expected = [
        1.0 * inv2, // t0, head0
        0.0,
        3.0 * inv2, // t0, head1
        0.0,
        2.0 * inv2, // t1, head0
        0.0,
        4.0 * inv2, // t1, head1
        0.0,
    ];
    for (i, (got, want)) in outs[0].fdata().iter().zip(&expected).enumerate() {
        assert_close(*got, *want, 1e-6, &format!("gdn gqa[{i}]"));
    }
}

#[test]
fn layernorm_hand_values() {
    let x = Owned::f32(&[3], vec![1.0, 2.0, 3.0]);
    let outs = unsafe { run_op(op("layernorm"), &[&x.t], &[], 1) }.unwrap();
    // mean 2, biased var (1+0+1)/3 = 2/3.
    let r = (2.0f32 / 3.0 + 1e-5).sqrt();
    assert_close(outs[0].fdata()[0], -1.0 / r, 1e-6, "layernorm[0]");
    assert_close(outs[0].fdata()[1], 0.0, 1e-6, "layernorm[1]");
    assert_close(outs[0].fdata()[2], 1.0 / r, 1e-6, "layernorm[2]");

    // With weight and bias.
    let w = Owned::f32(&[3], vec![1.0, 2.0, 3.0]);
    let b = Owned::f32(&[3], vec![0.0, 1.0, 0.0]);
    let outs = unsafe { run_op(op("layernorm"), &[&x.t, &w.t, &b.t], &[], 1) }.unwrap();
    assert_close(outs[0].fdata()[0], -1.0 / r, 1e-6, "affine layernorm[0]");
    assert_close(outs[0].fdata()[1], 1.0, 1e-6, "affine layernorm[1]");
    assert_close(outs[0].fdata()[2], 3.0 / r, 1e-6, "affine layernorm[2]");
}

#[test]
fn rope_hand_values() {
    // Half-split rotation over the first rotary_dim dims of the last axis.
    // theta=1 makes every inv_freq[j] = 1^(-2j/4) = 1, so angle_j = pos * 1:
    // for pos = 1, cos = cos(1) = 0.5403023, sin = sin(1) = 0.8414710.
    // Pairs are (i, i+h) with h = rotary_dim/2 = 2:
    //   out[i]   = x[i]*cos - x[i+2]*sin
    //   out[i+2] = x[i+2]*cos + x[i]*sin
    let x = Owned::f32(&[1, 4], vec![1.0, 2.0, 3.0, 4.0]);
    let y = Owned::f32(&[1, 4], vec![5.0, 6.0, 7.0, 8.0]);
    let pos = Owned::i64(&[1], vec![1]);
    let attrs = [
        ai64("rotary_dim", 4),
        af64("theta", 1.0),
        abool("partial_rotary", true),
    ];
    let outs = unsafe { run_op(op("rope"), &[&x.t, &y.t, &pos.t], &attrs, 2) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[1, 4]);
    assert_eq!(outs[1].t.dims(), &[1, 4]);
    let (c, s) = (1.0f32.cos(), 1.0f32.sin());
    let expected = [
        1.0 * c - 3.0 * s,
        2.0 * c - 4.0 * s,
        3.0 * c + 1.0 * s,
        4.0 * c + 2.0 * s,
    ];
    for (i, (got, want)) in outs[0].fdata().iter().zip(&expected).enumerate() {
        assert_close(*got, *want, 1e-6, &format!("rope x[{i}]"));
    }
    let expected = [
        5.0 * c - 7.0 * s,
        6.0 * c - 8.0 * s,
        7.0 * c + 5.0 * s,
        8.0 * c + 6.0 * s,
    ];
    for (i, (got, want)) in outs[1].fdata().iter().zip(&expected).enumerate() {
        assert_close(*got, *want, 1e-6, &format!("rope y[{i}]"));
    }
}

#[test]
fn rope_partial_rotary_and_default_positions() {
    // D=6, rotary_dim=4: dims 4..5 pass through untouched. Positions default
    // to arange(S) = [0, 1] along the FIRST axis (the sequence-first plan
    // convention). Position 0 has
    // angle 0 => cos=1, sin=0 => identity on the rotary block.
    // theta=1e7: inv_freq[0] = 1e7^0 = 1 (angle_0 = pos),
    // inv_freq[1] = 1e7^(-2/4) = 1/sqrt(1e7) (angle_1 = pos * 1/sqrt(1e7)).
    // Row t=1, pairs (0,2) and (1,3):
    //   out[0] = 7*cos(1) - 9*sin(1);        out[2] = 9*cos(1) + 7*sin(1)
    //   out[1] = 8*cos(f1) - 10*sin(f1);     out[3] = 10*cos(f1) + 8*sin(f1)
    //   out[4] = 11, out[5] = 12 (pass-through)
    let x = Owned::f32(
        &[2, 6],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let attrs = [
        ai64("rotary_dim", 4),
        af64("theta", 1e7),
        abool("partial_rotary", true),
    ];
    let outs = unsafe { run_op(op("rope"), &[&x.t, &x.t], &attrs, 2) }.unwrap();
    let f1 = 1e7f32.powf(-0.5);
    let (c1, s1) = (1.0f32.cos(), 1.0f32.sin());
    let (c2, s2) = (f1.cos(), f1.sin());
    let expected = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0, // t=0: angle 0, identity
        7.0 * c1 - 9.0 * s1,
        8.0 * c2 - 10.0 * s2,
        9.0 * c1 + 7.0 * s1,
        10.0 * c2 + 8.0 * s2,
        11.0,
        12.0,
    ];
    for (i, (got, want)) in outs[0].fdata().iter().zip(&expected).enumerate() {
        assert_close(*got, *want, 1e-5, &format!("partial rope[{i}]"));
    }
}

#[test]
fn quantize_dequantize_hand_values() {
    // per_tensor e4m3: amax = 1 -> scale = 1/448.
    let x = Owned::f32(&[3], vec![0.5, -0.5, 1.0]);
    let attrs = [astr("scheme", "per_tensor"), astr("format", "f8e4m3")];
    let outs = unsafe { run_op(op("quantize"), &[&x.t], &attrs, 2) }.unwrap();
    let q = &outs[0];
    let s = &outs[1];
    assert_eq!(q.t.dtype, RsDtype::F8E4M3);
    assert_eq!(s.t.rank, 0);
    assert_eq!(s.fdata(), &[1.0 / 448.0]);
    // 224 = 1.75 * 2^7 -> 0_1110_110 = 0x76; -224 -> 0xF6; 448 -> 0_1111_110 = 0x7E.
    assert_eq!(q.bytes(), &[0x76, 0xF6, 0x7E]);

    // Round trip through dequantize.
    let outs = unsafe { run_op(op("dequantize"), &[&q.t, &s.t], &attrs, 1) }.unwrap();
    let dq = outs[0].fdata();
    assert_close(dq[0], 0.5, 1e-6, "dq 0.5");
    assert_close(dq[1], -0.5, 1e-6, "dq -0.5");
    assert_close(dq[2], 1.0, 1e-6, "dq 1.0");
}

#[test]
fn quantize_per_token_hand_values() {
    let x = Owned::f32(&[2, 2], vec![0.5, -0.5, 1.0, 2.0]);
    let attrs = [astr("scheme", "per_token"), astr("format", "f8e4m3")];
    let outs = unsafe { run_op(op("quantize"), &[&x.t], &attrs, 2) }.unwrap();
    let s = &outs[1];
    assert_eq!(s.t.dims(), &[2]);
    assert_eq!(s.fdata(), &[0.5 / 448.0, 2.0 / 448.0]);
    // Row 0: ±448 -> 0x7E / 0xFE. Row 1: 1/scale = 224 -> 0x76; 2/scale = 448 -> 0x7E.
    assert_eq!(outs[0].bytes(), &[0x7E, 0xFE, 0x76, 0x7E]);
}

#[test]
fn quantize_per_block_hand_values() {
    let x = Owned::f32(&[2, 4], vec![0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, 0.5]);
    let attrs = [
        astr("scheme", "per_block"),
        astr("format", "f8e4m3"),
        ai64s("block", &[1, 2]),
    ];
    let outs = unsafe { run_op(op("quantize"), &[&x.t], &attrs, 2) }.unwrap();
    let s = &outs[1];
    assert_eq!(s.t.dims(), &[2, 2]);
    assert_eq!(
        s.fdata(),
        &[0.5 / 448.0, 1.0 / 448.0, 2.0 / 448.0, 4.0 / 448.0]
    );
    // Block (0,0): ±448 -> 0x7E/0xFE; (0,1): same; (1,0): same;
    // (1,1): 448 -> 0x7E, 0.5/(4/448) = 56 = 1.75*2^5 -> 0_1100_110 = 0x66.
    assert_eq!(
        outs[0].bytes(),
        &[0x7E, 0xFE, 0x7E, 0xFE, 0x7E, 0xFE, 0x7E, 0x66]
    );
}

#[test]
fn amax_update_hand_values() {
    let x = Owned::f32(&[3], vec![1.0, -5.0, 2.0]);
    let amax = Owned::f32(&[], vec![3.0]);
    let outs = unsafe {
        run_op(
            op("amax_update"),
            &[&x.t, &amax.t],
            &[astr("scheme", "per_tensor")],
            1,
        )
    }
    .unwrap();
    assert_eq!(outs[0].fdata(), &[5.0]);

    let x = Owned::f32(&[2, 3], vec![1.0, -5.0, 2.0, 0.0, -1.0, 3.0]);
    let amax = Owned::f32(&[2], vec![2.0, 2.0]);
    let outs = unsafe {
        run_op(
            op("amax_update"),
            &[&x.t, &amax.t],
            &[astr("scheme", "per_token")],
            1,
        )
    }
    .unwrap();
    assert_eq!(outs[0].fdata(), &[5.0, 3.0]);
}

#[test]
fn embedding_hand_values() {
    let w = Owned::f32(
        &[4, 3],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let idx = Owned::i32(&[3], vec![3, 0, 2]);
    let outs = unsafe { run_op(op("embedding"), &[&w.t, &idx.t], &[], 1) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[3, 3]);
    assert_eq!(
        outs[0].fdata(),
        &[10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn gather_hand_values() {
    let x = Owned::f32(&[3, 4], (1..=12).map(|v| v as f32).collect());
    // Torch-gather convention: indices share x's rank. Along axis -1
    // (default), each row selects columns 1 and 2.
    let idx = Owned::i32(&[3, 2], vec![1, 2, 1, 2, 1, 2]);
    let outs = unsafe { run_op(op("gather"), &[&x.t, &idx.t], &[ai64("axis", -1)], 1) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[3, 2]);
    assert_eq!(outs[0].fdata(), &[2.0, 3.0, 6.0, 7.0, 10.0, 11.0]);
    // Along axis 0, each column selects rows 2 and 0.
    let idx = Owned::i64(&[2, 4], vec![2, 2, 2, 2, 0, 0, 0, 0]);
    let outs = unsafe { run_op(op("gather"), &[&x.t, &idx.t], &[ai64("axis", 0)], 1) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[2, 4]);
    assert_eq!(
        outs[0].fdata(),
        &[9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn scatter_hand_values_and_last_writer_wins() {
    let x = Owned::f32(&[3, 4], vec![0.0; 12]);
    // values[i, k] (row-major [3, 2]): k=0 column is [1,1,2], k=1 is [1,2,2].
    let idx = Owned::i32(&[2], vec![3, 1]);
    let values = Owned::f32(&[3, 2], vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    let outs = unsafe {
        run_op(
            op("scatter"),
            &[&x.t, &idx.t, &values.t],
            &[ai64("axis", -1)],
            1,
        )
    }
    .unwrap();
    assert_eq!(outs[0].t.dims(), &[3, 4]);
    // out[:,3] = values[:,0] = [1,1,2]; out[:,1] = values[:,1] = [1,2,2].
    let expected = [
        0.0, 1.0, 0.0, 1.0, // row 0
        0.0, 2.0, 0.0, 1.0, // row 1
        0.0, 2.0, 0.0, 2.0, // row 2
    ];
    assert_eq!(outs[0].fdata(), &expected);

    // Duplicate indices: the last writer wins (k=1 overwrites k=0 per row).
    let idx = Owned::i32(&[2], vec![1, 1]);
    // k=0 column [5,5,7], k=1 column [5,7,7]; out[:,1] ends as [5,7,7].
    let values = Owned::f32(&[3, 2], vec![5.0, 5.0, 5.0, 7.0, 7.0, 7.0]);
    let outs = unsafe {
        run_op(
            op("scatter"),
            &[&x.t, &idx.t, &values.t],
            &[ai64("axis", -1)],
            1,
        )
    }
    .unwrap();
    assert_eq!(
        outs[0].fdata(),
        &[0.0, 5.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0]
    );
}

#[test]
fn scatter_reduce_add_accumulates_duplicates_bitwise_deterministically() {
    let x = Owned::f32(&[3, 4], vec![0.0; 12]);
    // Duplicate indices [1, 1]: values is [3, 2] row-major, so the k=0
    // column is [1,2,3] and the k=1 column is [4,5,6]; with reduce = "add"
    // out[:,1] accumulates to [5,7,9].
    let idx = Owned::i32(&[2], vec![1, 1]);
    let values = Owned::f32(&[3, 2], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let attrs = [astr("reduce", "add")];
    let r1 = unsafe { run_op(op("scatter"), &[&x.t, &idx.t, &values.t], &attrs, 1) }.unwrap();
    assert_eq!(r1[0].t.dims(), &[3, 4]);
    assert_eq!(
        r1[0].fdata(),
        &[0.0, 5.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0]
    );
    // Fixed ascending-k accumulation order: two runs are bitwise identical.
    let r2 = unsafe { run_op(op("scatter"), &[&x.t, &idx.t, &values.t], &attrs, 1) }.unwrap();
    assert_eq!(
        r1[0].bytes(),
        r2[0].bytes(),
        "scatter add must be bitwise reproducible"
    );
    // The default (assign) keeps last-writer-wins: out[:,1] ends as [4,5,6].
    let r3 = unsafe {
        run_op(
            op("scatter"),
            &[&x.t, &idx.t, &values.t],
            &[ai64("axis", -1)],
            1,
        )
    }
    .unwrap();
    assert_eq!(
        r3[0].fdata(),
        &[0.0, 4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0]
    );
    // assign is also the behaviour when 'reduce' is passed explicitly.
    let r4 = unsafe {
        run_op(
            op("scatter"),
            &[&x.t, &idx.t, &values.t],
            &[ai64("axis", -1), astr("reduce", "assign")],
            1,
        )
    }
    .unwrap();
    assert_eq!(r4[0].fdata(), r3[0].fdata());
}

#[test]
fn sdpa_hand_values() {
    // q = k = identity (S=T=2, D=2), v = [[1,2],[3,4]]. Scores are I, so
    // row 0 of the probs is softmax([1,0]) = [e/(e+1), 1/(e+1)] and row 1 is
    // its mirror. Hand-computed in f32 below.
    let q = Owned::f32(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let k = Owned::f32(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let v = Owned::f32(&[1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let outs = unsafe { run_op(op("sdpa"), &[&q.t, &k.t, &v.t], &[], 1) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[1, 2, 2]);
    let e = 1f32.exp();
    let p0 = e / (e + 1.0);
    let p1 = 1.0 / (e + 1.0);
    let expected = [
        p0 * 1.0 + p1 * 3.0,
        p0 * 2.0 + p1 * 4.0,
        p1 * 1.0 + p0 * 3.0,
        p1 * 2.0 + p0 * 4.0,
    ];
    assert_eq!(outs[0].fdata(), &expected);

    // S != T: q [1,1,2], k [1,3,2], v [1,3,1] with uniform keys.
    let q = Owned::f32(&[1, 1, 2], vec![1.0, 1.0]);
    let k = Owned::f32(&[1, 3, 2], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
    let v = Owned::f32(&[1, 3, 1], vec![2.0, 4.0, 6.0]);
    let outs = unsafe { run_op(op("sdpa"), &[&q.t, &k.t, &v.t], &[], 1) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[1, 1, 1]);
    // scores = [1, 1, 1] -> uniform probs 1/3 each.
    assert_close(outs[0].fdata()[0], 12.0 / 3.0, 1e-6, "sdpa uniform");
}

#[test]
fn sdpa_gqa_causal_hand_values() {
    // GQA, per-head form [.., S, H, D]: num_heads=2, num_kv_heads=1, so both
    // query heads share one kv head. q [1,2,2,2] (S=2, D=2):
    // head0 = [[1,0],[0,1]], head1 = [[0,1],[1,0]]; k = identity [1,2,1,2];
    // v = [[1,2],[3,4]]; scale 1.0 (explicit, for hand math), causal.
    // head0 scores = q0 @ k^T = [[1,0],[0,1]]; causal masks j > i:
    //   row0: [1, -inf]  -> probs [1, 0]      -> o0 = v[0] = [1, 2]
    //   row1: [0, 1]     -> probs [1/(1+e), e/(1+e)]
    // head1 scores = q1 @ k^T = [[0,1],[1,0]]:
    //   row0: [0, -inf]  -> probs [1, 0]      -> o1 = v[0] = [1, 2]
    //   row1: [1, 0]     -> probs [e/(1+e), 1/(1+e)]
    // q[i][h][dd] = q[i*4 + h*2 + dd].
    let q = Owned::f32(&[1, 2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    let k = Owned::f32(&[1, 2, 1, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let v = Owned::f32(&[1, 2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]);
    // The per-head form is declared; the head counts are read from the tensors
    // (q has 2 heads, k/v 1 — the GQA repeat is 2).
    let attrs = [
        abool("per_head", true),
        abool("causal", true),
        af64("scale", 1.0),
    ];
    let outs = unsafe { run_op(op("sdpa"), &[&q.t, &k.t, &v.t], &attrs, 1) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[1, 2, 2, 2]);
    let e = 1.0f32.exp();
    let (p0, p1) = (1.0 / (1.0 + e), e / (1.0 + e));
    // Output rows interleave heads: row0 = [o0[0], o0[1], o1[0], o1[1]].
    let expected = [
        1.0,
        2.0,
        1.0,
        2.0, // row 0: both heads attend only j=0 -> v[0]
        p0 * 1.0 + p1 * 3.0,
        p0 * 2.0 + p1 * 4.0,
        p1 * 1.0 + p0 * 3.0,
        p1 * 2.0 + p0 * 4.0,
    ];
    for (i, (got, want)) in outs[0].fdata().iter().zip(&expected).enumerate() {
        assert_close(*got, *want, 1e-6, &format!("sdpa gqa[{i}]"));
    }
}

#[test]
fn sdpa_additive_mask_hand_values() {
    // The legacy flat form with an additive mask: q = k = identity,
    // v = [[1,2],[3,4]], mask [0, -inf] (broadcast over S): every row may
    // attend only j=0, so the output is v[0] = [1, 2] for both rows.
    let q = Owned::f32(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let k = Owned::f32(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let v = Owned::f32(&[1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mask = Owned::f32(&[2], vec![0.0, f32::NEG_INFINITY]);
    let outs = unsafe { run_op(op("sdpa"), &[&q.t, &k.t, &v.t, &mask.t], &[], 1) }.unwrap();
    assert_eq!(outs[0].fdata(), &[1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn cross_entropy_hand_values() {
    let logits = Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    let targets = Owned::i32(&[2], vec![2, 0]);
    let outs = unsafe { run_op(op("cross_entropy"), &[&logits.t, &targets.t], &[], 1) }.unwrap();
    assert_eq!(outs[0].t.rank, 0);
    // row 0: ln(e^-2 + e^-1 + 1); row 1: ln(3); mean of the two.
    let expected = (((-2.0f64).exp() + (-1.0f64).exp() + 1.0).ln() + 3.0f64.ln()) / 2.0;
    assert_close(outs[0].fdata()[0], expected as f32, 1e-6, "cross_entropy");
}

#[test]
fn adamw_hand_values() {
    let p = Owned::f32(&[1], vec![1.0]);
    let g = Owned::f32(&[1], vec![2.0]);
    let m = Owned::f32(&[1], vec![0.0]);
    let v = Owned::f32(&[1], vec![0.0]);
    let outs = unsafe { run_op(op("adamw"), &[&p.t, &g.t, &m.t, &v.t], &[], 3) }.unwrap();
    assert_eq!(outs.len(), 3);
    // defaults: lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0, step=1.
    // m = 0.2, v = 0.004; bias-corrected: mh = 0.2/(1-0.9) = 2, vh = 4.
    let mh = 0.2f32 / (1.0 - 0.9f32);
    let vh = 0.004f32 / (1.0 - 0.999f32);
    let expected = 1.0 - 1e-3 * (mh / (vh.sqrt() + 1e-8));
    assert_close(outs[0].fdata()[0], expected, 1e-7, "adamw param");
    assert_close(outs[1].fdata()[0], 0.2, 1e-7, "adamw exp_avg");
    assert_close(outs[2].fdata()[0], 0.004, 1e-7, "adamw exp_avg_sq");
}

#[test]
fn topk_router_hand_values_and_tie_break() {
    let logits = Owned::f32(&[1, 4], vec![1.0, 3.0, 2.0, 4.0]);
    let outs = unsafe { run_op(op("topk_router"), &[&logits.t], &[ai64("top_k", 2)], 2) }.unwrap();
    assert_eq!(outs[0].t.dims(), &[1, 2]);
    assert_eq!(outs[0].t.dtype, RsDtype::F32);
    assert_eq!(outs[1].t.dtype, RsDtype::I32);
    // softmax([1,3,2,4]): m=4, sum = e^-3 + e^-1 + e^-2 + 1.
    let m = 4.0f32;
    let e = |v: f32| (v - m).exp();
    let sum = e(1.0) + e(3.0) + e(2.0) + e(4.0);
    let p3 = e(4.0) / sum;
    let p1 = e(3.0) / sum;
    assert_eq!(outs[1].i32s, vec![3, 1]);
    assert_close(outs[0].fdata()[0], p3, 1e-6, "topk weight 0");
    assert_close(outs[0].fdata()[1], p1, 1e-6, "topk weight 1");

    // Ties break toward the lower expert index.
    let logits = Owned::f32(&[1, 2], vec![0.0, 0.0]);
    let outs = unsafe { run_op(op("topk_router"), &[&logits.t], &[ai64("top_k", 1)], 2) }.unwrap();
    assert_eq!(outs[1].i32s, vec![0]);

    // norm_topk_prob: the selected weights are renormalised to sum to 1 —
    // HF's Qwen3_5MoeTopKRouter does this unconditionally, the declaration
    // asks for it via the attribute. Indices unchanged.
    let logits = Owned::f32(&[1, 4], vec![1.0, 3.0, 2.0, 4.0]);
    let outs = unsafe {
        run_op(
            op("topk_router"),
            &[&logits.t],
            &[ai64("top_k", 2), abool("norm_topk_prob", true)],
            2,
        )
    }
    .unwrap();
    assert_eq!(outs[1].i32s, vec![3, 1]);
    let total = p3 + p1;
    assert_close(
        outs[0].fdata()[0],
        p3 / total,
        1e-6,
        "renormalised weight 0",
    );
    assert_close(
        outs[0].fdata()[1],
        p1 / total,
        1e-6,
        "renormalised weight 1",
    );
}

// ── moe_layer ────────────────────────────────────────────────────────────────

#[test]
fn moe_layer_hand_values() {
    // One token (rows=1), H=2, E=3, I=2, K=2. x = [1, 2], routing weights
    // [0.25, 0.75], indices [2, 0] — experts 2 and 0 BOTH run (dropless:
    // every selected expert runs), expert 1 does not (its weights are the 7.0
    // sentinel in gate_proj, up_proj AND down_proj: a single read of it would
    // blow the exact expected values).
    //
    // gate_proj/up_proj are DE-FUSED [E, H, I] ([out, in] per expert:
    // g[j] = x @ gate[e][:, j]), the halves the description's binding splits
    // off the checkpoint's gate_up_proj:
    //   expert 0: gate cols (0, .5) -> g = 0*1 + .5*2 = 1; (2, -1) -> g = 0.
    //             up cols (1, 0) -> u = 1; (0, 1) -> u = 2.
    //             a = [silu(1)*1, silu(0)*2] = [0.7310586, 0]   (silu(0) = 0)
    //   expert 2: gate cols (2, -1) -> g = 0; (0, .5) -> g = 1.
    //             up cols (0, 1) -> u = 2; (1, 0) -> u = 1.
    //             a = [silu(0)*2, silu(1)*1] = [0, 0.7310586]
    // down_proj is [E, H, I] (out[h] = sum_j a[j] * down[e][h*I + j]):
    //   expert 0: [[1, 0], [0, 1]] -> [0.7310586, 0]
    //   expert 2: [[0, 3], [2, 0]] -> [3*0.7310586, 0]
    // Expert sum (k ascending), weighted by the routing probabilities:
    //   0.25 * [3*s, 0] + 0.75 * [s, 0] = [0.75*s + 0.75*s, 0] = [1.5*s, 0]
    // Shared expert (same gate/up/down trick): a = [s, 0] -> s_out = [s, 0];
    // shared_expert_gate [1, 2] = (1, -0.5) -> gs = 1*1 - 0.5*2 = 0, and
    // sigmoid(0) = 0.5 exactly -> shared contribution [0.5*s, 0].
    // Total: [1.5*s + 0.5*s, 0] = [2*s, 0], where s = silu(1) = 1/(1 + e^-1).
    let h = Owned::f32(&[1, 2], vec![1.0, 2.0]);
    let w = Owned::f32(&[1, 2], vec![0.25, 0.75]);
    let idx = Owned::i32(&[1, 2], vec![2, 0]);
    let gp = Owned::f32(
        &[3, 2, 2],
        vec![
            0.0, 2.0, 0.5, -1.0, // expert 0: gate cols (0,.5),(2,-1)
            7.0, 7.0, 7.0, 7.0, // expert 1: sentinel, never read
            2.0, 0.0, -1.0, 0.5, // expert 2: gate cols (2,-1),(0,.5)
        ],
    );
    let up = Owned::f32(
        &[3, 2, 2],
        vec![
            1.0, 0.0, 0.0, 1.0, // expert 0: up cols (1,0),(0,1)
            7.0, 7.0, 7.0, 7.0, // expert 1: sentinel, never read
            0.0, 1.0, 1.0, 0.0, // expert 2: up cols (0,1),(1,0)
        ],
    );
    let dn = Owned::f32(
        &[3, 2, 2],
        vec![
            1.0, 0.0, 0.0, 1.0, // expert 0
            7.0, 7.0, 7.0, 7.0, // expert 1: sentinel
            0.0, 3.0, 2.0, 0.0, // expert 2
        ],
    );
    let sg = Owned::f32(&[2, 2], vec![0.0, 0.5, 2.0, -1.0]);
    let su = Owned::f32(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let sd = Owned::f32(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let sgg = Owned::f32(&[1, 2], vec![1.0, -0.5]);
    let outs = unsafe {
        run_op(
            op("moe_layer"),
            &[
                &h.t, &w.t, &idx.t, &gp.t, &up.t, &dn.t, &sg.t, &su.t, &sd.t, &sgg.t,
            ],
            &[],
            1,
        )
    }
    .unwrap();
    assert_eq!(outs[0].t.dims(), &[1, 2]);
    assert_eq!(outs[0].t.dtype, RsDtype::F32);
    let s = 1.0 / (1.0 + (-1.0f32).exp()); // silu(1)
    assert_close(outs[0].fdata()[0], 2.0 * s, 1e-6, "moe[0]");
    assert_close(outs[0].fdata()[1], 0.0, 1e-6, "moe[1]");

    // The 3D form [b, s, H] is the declared static in/out: the same math with
    // the batch dim kept, output keeps h's shape.
    let h3 = Owned::f32(&[1, 1, 2], vec![1.0, 2.0]);
    let outs = unsafe {
        run_op(
            op("moe_layer"),
            &[
                &h3.t, &w.t, &idx.t, &gp.t, &up.t, &dn.t, &sg.t, &su.t, &sd.t, &sgg.t,
            ],
            &[],
            1,
        )
    }
    .unwrap();
    assert_eq!(outs[0].t.dims(), &[1, 1, 2]);
    assert_close(outs[0].fdata()[0], 2.0 * s, 1e-6, "moe3d[0]");
    assert_close(outs[0].fdata()[1], 0.0, 1e-6, "moe3d[1]");
}

#[test]
fn moe_layer_declares_the_dispatch_and_combine_all_to_alls() {
    // The two all_to_all({tp, ep}) collectives are declared, not derivable:
    // the routing is data (op-vocabulary §4), so the planner learns them only
    // from this descriptor. dispatch = the h input (io index 0), combine =
    // the output (io index 10, at offset n_inputs).
    let o = op("moe_layer");
    assert_eq!(o.n_collectives, 2);
    assert!(!o.collectives.is_null());
    let cols = unsafe { slice::from_raw_parts(o.collectives, o.n_collectives as usize) };
    let tp_ep = RsGroupKind::from_raw(RsGroupKind::TP.raw() | RsGroupKind::EP.raw());
    for (i, c) in cols.iter().enumerate() {
        assert_eq!(c.kind, RsCollectiveKind::ALL_TO_ALL, "collective {i} kind");
        assert_eq!(c.group, tp_ep, "collective {i} group must be {{tp, ep}}");
    }
    assert_eq!(cols[0].tensor_index, 0, "dispatch sends the h input");
    assert_eq!(cols[1].tensor_index, 10, "combine assembles the output");
}

#[test]
fn moe_layer_memory_reports_its_scratch() {
    // The fused body allocates one H-element f32 accumulation buffer per call.
    let o = op("moe_layer");
    let h = Owned::f32(&[1, 2], vec![0.0; 2]);
    let w = Owned::f32(&[1, 1], vec![0.0]);
    let idx = Owned::i32(&[1, 1], vec![0]);
    let gp = Owned::f32(&[1, 2, 1], vec![0.0; 2]);
    let up = Owned::f32(&[1, 2, 1], vec![0.0; 2]);
    let dn = Owned::f32(&[1, 2, 1], vec![0.0; 2]);
    let sg = Owned::f32(&[1, 2], vec![0.0; 2]);
    let su = Owned::f32(&[1, 2], vec![0.0; 2]);
    let sd = Owned::f32(&[2, 1], vec![0.0; 2]);
    let sgg = Owned::f32(&[1, 2], vec![0.0; 2]);
    let mut req = RsMemReq::default();
    let io: Vec<*const RsTensor> = vec![
        &h.t, &w.t, &idx.t, &gp.t, &up.t, &dn.t, &sg.t, &su.t, &sd.t, &sgg.t,
    ];
    let st = unsafe { (o.memory.unwrap())(io.as_ptr(), io.len() as u32, ptr::null(), &mut req) };
    assert_eq!(st, 0);
    assert_eq!(req.workspace_bytes, 4 * 2); // one H=2 f32 buffer
    assert_eq!(req.save_for_backward_bytes, 0);
}

#[test]
fn moe_layer_rejects_out_of_range_expert_indices() {
    // Indices are used as-is (no wrap-around): an expert id outside [0, E) is
    // a hard error naming the token, the routing slot and the id.
    let h = Owned::f32(&[1, 2], vec![1.0, 2.0]);
    let w = Owned::f32(&[1, 1], vec![1.0]);
    let idx = Owned::i32(&[1, 1], vec![3]); // E = 3 -> 3 is out of range
    let gp = Owned::f32(&[3, 2, 1], vec![0.0; 6]);
    let up = Owned::f32(&[3, 2, 1], vec![0.0; 6]);
    let dn = Owned::f32(&[3, 2, 1], vec![0.0; 6]);
    let sg = Owned::f32(&[1, 2], vec![0.0; 2]);
    let su = Owned::f32(&[1, 2], vec![0.0; 2]);
    let sd = Owned::f32(&[2, 1], vec![0.0; 2]);
    let sgg = Owned::f32(&[1, 2], vec![0.0; 2]);
    let o = op("moe_layer");
    let mut descs: Vec<RsTensor> = (0..1).map(|_| RsTensor::default()).collect();
    unsafe {
        call_infer(
            o,
            &[
                &h.t, &w.t, &idx.t, &gp.t, &up.t, &dn.t, &sg.t, &su.t, &sd.t, &sgg.t,
            ],
            &mut descs,
            &[],
        )
    }
    .unwrap();
    let mut out = Owned::zeros_for(&descs[0]);
    let err = unsafe {
        call_exec(
            o,
            &[
                &h.t, &w.t, &idx.t, &gp.t, &up.t, &dn.t, &sg.t, &su.t, &sd.t, &sgg.t,
            ],
            &mut [&mut out.t],
            &[],
        )
    }
    .unwrap_err();
    assert!(
        err.contains("expert 3") && err.contains("outside [0, 3)"),
        "the error must name the offending index and the range: {err}"
    );
}

// ── matmul vs naive reference on pseudo-random data ─────────────────────────

/// Deterministic LCG; same stream every test run.
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 24) as f32
    }
}

#[test]
fn matmul_matches_naive_triple_loop_on_random_data() {
    let (m, k, n) = (7usize, 5usize, 3usize);
    let mut rng = Lcg(0x12345678);
    let a: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.f32()).collect();
    let ta = Owned::f32(&[m as i64, k as i64], a.clone());
    let tb = Owned::f32(&[k as i64, n as i64], b.clone());
    let outs = unsafe { run_op(op("matmul"), &[&ta.t, &tb.t], &[], 1) }.unwrap();
    let got = outs[0].fdata();

    // Naive reference with the same k-ascending accumulation order; identical
    // order means bitwise-identical results, which is the whole point of the
    // determinism contract.
    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            want[i * n + j] = acc;
        }
    }
    assert_eq!(got, want.as_slice(), "bitwise equality with the naive loop");
}

// ── numerical-stability cases ───────────────────────────────────────────────

#[test]
fn softmax_is_stable_for_large_magnitude_inputs() {
    let x = Owned::f32(&[3], vec![1000.0, 1000.0, 0.0]);
    let outs = unsafe { run_op(op("softmax"), &[&x.t], &[], 1) }.unwrap();
    let r = outs[0].fdata();
    assert_eq!(
        r[0], 0.5,
        "exp(0)/(exp(0)+exp(0)+exp(-1000)) is exactly 0.5"
    );
    assert_eq!(r[1], 0.5);
    assert_eq!(r[2], 0.0);
    assert!(r.iter().all(|v| v.is_finite()));

    let x = Owned::f32(&[2], vec![1000.0, -1000.0]);
    let outs = unsafe { run_op(op("softmax"), &[&x.t], &[], 1) }.unwrap();
    assert_eq!(outs[0].fdata(), &[1.0, 0.0]);
}

#[test]
fn cross_entropy_is_stable_for_extreme_logits() {
    let logits = Owned::f32(&[1, 3], vec![1000.0, -1000.0, 0.0]);
    let targets = Owned::i32(&[1], vec![2]);
    let outs = unsafe { run_op(op("cross_entropy"), &[&logits.t, &targets.t], &[], 1) }.unwrap();
    // m = 1000, sum = e^0 + e^-2000 + e^-1000 = 1 (both underflow), so the
    // loss is exactly 1000 - 0 = 1000. The naive log(softmax) path would
    // give -inf here.
    assert_eq!(outs[0].fdata()[0], 1000.0);
}

#[test]
fn rmsnorm_is_stable_for_near_zero_inputs() {
    let x = Owned::f32(&[2], vec![1e-30, -1e-30]);
    let outs = unsafe { run_op(op("rmsnorm"), &[&x.t], &[], 1) }.unwrap();
    // x^2 underflows to 0, so r = sqrt(0 + 1e-5); the result is finite and
    // equal to x / r.
    let r = 1e-5f32.sqrt();
    let r0 = outs[0].fdata()[0];
    let r1 = outs[0].fdata()[1];
    assert!(
        r0.is_finite() && r1.is_finite(),
        "near-zero input must not produce NaN"
    );
    assert_close(r0, 1e-30 / r, 1e-34, "rmsnorm near-zero [0]");
    assert_close(r1, -1e-30 / r, 1e-34, "rmsnorm near-zero [1]");
}

// ── quantization round trip ─────────────────────────────────────────────────

#[test]
fn quantize_dequantize_round_trip_stays_within_format_resolution() {
    let mut rng = Lcg(0xdeadbeef);
    let values: Vec<f32> = (0..64).map(|_| 0.01 + 99.99 * rng.f32()).collect();
    let x = Owned::f32(&[64], values.clone());
    for format in ["f8e4m3", "f8e5m2"] {
        let half_ulp = if format == "f8e4m3" { 0.0625 } else { 0.125 };
        let attrs = [astr("scheme", "per_tensor"), astr("format", format)];
        let q = unsafe { run_op(op("quantize"), &[&x.t], &attrs, 2) }.unwrap();
        let dq = unsafe { run_op(op("dequantize"), &[&q[0].t, &q[1].t], &attrs, 1) }.unwrap();
        for (i, &v) in values.iter().enumerate() {
            let err = (dq[0].fdata()[i] - v).abs();
            assert!(
                err <= v * half_ulp * 1.01 + 1e-5,
                "{format} value {v}: round-trip error {err} exceeds half-ulp bound"
            );
        }
    }
}

#[test]
fn quantize_dequantize_round_trips_grid_values_exactly() {
    // With amax = 448 the scale is exactly 1, and every grid value must
    // survive the round trip bitwise.
    let x = Owned::f32(&[5], vec![0.0, 0.5, 1.0, 2.0, 448.0]);
    let attrs = [astr("scheme", "per_tensor"), astr("format", "f8e4m3")];
    let q = unsafe { run_op(op("quantize"), &[&x.t], &attrs, 2) }.unwrap();
    assert_eq!(q[1].fdata(), &[1.0]);
    let dq = unsafe { run_op(op("dequantize"), &[&q[0].t, &q[1].t], &attrs, 1) }.unwrap();
    assert_eq!(dq[0].fdata(), &[0.0, 0.5, 1.0, 2.0, 448.0]);
}

#[test]
fn fp8_emulation_grid_is_correct() {
    // Spot-check the documented grid: RNE onto exponent/mantissa pairs.
    // e4m3: 0.5 = 2^-1 -> 0_0110_000 = 0x30.
    let v = |x: f32| {
        rustrain_kernels::op::quant::f32_to_fp8(x, rustrain_kernels::op::quant::Fp8Format::E4M3)
    };
    assert_eq!(v(0.0), 0x00);
    assert_eq!(v(-0.0), 0x80);
    assert_eq!(v(0.5), 0x30);
    assert_eq!(v(1.0), 0x38);
    assert_eq!(v(448.0), 0x7E);
    assert_eq!(v(449.0), 0x7E, "449 still rounds down to 448 (RNE)");
    assert_eq!(
        v(465.0),
        0x7F,
        "past the RNE boundary, e4m3 has no infinity: NaN"
    );
    // RNE tie: 1.5 + 1/16 ulp boundary. e4m3 ulp at 1.5 is 2^-4 = 0.0625;
    // 1.5625 is halfway between 1.5 (mant 100) and 1.625 (mant 101).
    assert_eq!(v(1.5625), 0x3C, "1.5 -> 0_0111_100");
    let f = |b: u8| {
        rustrain_kernels::op::quant::fp8_to_f32(b, rustrain_kernels::op::quant::Fp8Format::E4M3)
    };
    assert_eq!(f(0x30), 0.5);
    assert_eq!(f(0x7E), 448.0);
    assert!(f(0x7F).is_nan());
    // Subnormal: min subnormal unit is 2^-9.
    assert_eq!(f(0x01), 2f32.powi(-9));

    let v5 = |x: f32| {
        rustrain_kernels::op::quant::f32_to_fp8(x, rustrain_kernels::op::quant::Fp8Format::E5M2)
    };
    let f5 = |b: u8| {
        rustrain_kernels::op::quant::fp8_to_f32(b, rustrain_kernels::op::quant::Fp8Format::E5M2)
    };
    assert_eq!(v5(1.0), 0x3C); // 0_01111_00
    assert_eq!(v5(57344.0), 0x7B); // 0_11110_11
    assert_eq!(v5(1e10), 0x7C, "e5m2 overflow is +inf");
    assert_eq!(f5(0x7C), f32::INFINITY);
    assert_eq!(f5(0x01), 2f32.powi(-16));
}

// ── determinism: every op twice, bitwise ────────────────────────────────────

/// Builds the inputs, attributes and output arity for one determinism case
/// per registered op.
#[allow(clippy::vec_init_then_push)] // 26 tuple pushes read as a table; a vec! literal is noise
fn determinism_cases() -> Vec<(&'static str, Vec<Owned>, Vec<RsAttr>, usize)> {
    let mut cases: Vec<(&'static str, Vec<Owned>, Vec<RsAttr>, usize)> = Vec::new();
    cases.push((
        "view",
        vec![Owned::f32(&[2, 3], (1..=6).map(|v| v as f32).collect())],
        vec![],
        1,
    ));
    cases.push((
        "reshape",
        vec![Owned::f32(&[2, 3], (1..=6).map(|v| v as f32).collect())],
        vec![ai64s("shape", &[3, 2])],
        1,
    ));
    cases.push((
        "transpose",
        vec![Owned::f32(&[2, 3], (1..=6).map(|v| v as f32).collect())],
        vec![ai64("dim0", 0), ai64("dim1", 1)],
        1,
    ));
    cases.push((
        "narrow",
        vec![Owned::f32(&[3, 4], (1..=12).map(|v| v as f32).collect())],
        vec![ai64("dim", -1), ai64("start", 1), ai64("length", 2)],
        1,
    ));
    cases.push((
        "cat",
        vec![
            Owned::f32(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            Owned::f32(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]),
        ],
        vec![ai64("dim", -1)],
        1,
    ));
    cases.push((
        "broadcast",
        vec![Owned::f32(&[3, 1], vec![1.0, 2.0, 3.0])],
        vec![ai64s("shape", &[3, 4])],
        1,
    ));
    cases.push((
        "matmul",
        vec![
            Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Owned::f32(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ],
        vec![],
        1,
    ));
    cases.push((
        "linear",
        vec![
            Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Owned::f32(&[3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            Owned::f32(&[2], vec![10.0, 20.0]),
        ],
        vec![],
        1,
    ));
    cases.push((
        "bmm",
        vec![
            Owned::f32(&[2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]),
            Owned::f32(&[2, 2, 2], vec![1.0, 1.0, 1.0, 1.0, 3.0, 0.0, 0.0, 3.0]),
        ],
        vec![],
        1,
    ));
    cases.push((
        "elementwise_unary",
        vec![Owned::f32(&[4], vec![-2.0, -1.0, 0.0, 1.0])],
        vec![astr("kind", "silu")],
        1,
    ));
    cases.push((
        "elementwise_binary",
        vec![
            Owned::f32(&[2, 1], vec![1.0, 2.0]),
            Owned::f32(&[1, 3], vec![10.0, 20.0, 30.0]),
        ],
        vec![astr("kind", "add")],
        1,
    ));
    cases.push((
        "compare",
        vec![
            Owned::f32(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            Owned::f32(&[2, 2], vec![1.0, 9.0, 3.0, 4.0]),
        ],
        vec![astr("kind", "ge")],
        1,
    ));
    cases.push((
        "reduce",
        vec![Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
        vec![astr("kind", "mean"), ai64("axis", -1)],
        1,
    ));
    cases.push((
        "softmax",
        vec![Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
        vec![],
        1,
    ));
    cases.push((
        "rmsnorm",
        vec![
            Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Owned::f32(&[3], vec![1.0, 2.0, 3.0]),
        ],
        vec![],
        1,
    ));
    cases.push((
        "layernorm",
        vec![
            Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            Owned::f32(&[3], vec![1.0, 2.0, 3.0]),
            Owned::f32(&[3], vec![0.5, -0.5, 1.0]),
        ],
        vec![],
        1,
    ));
    cases.push((
        "quantize",
        vec![Owned::f32(
            &[2, 4],
            vec![0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, 0.5],
        )],
        vec![astr("scheme", "per_token"), astr("format", "f8e4m3")],
        2,
    ));
    cases.push((
        "dequantize",
        vec![
            Owned::u8(
                RsDtype::F8E4M3,
                &[2, 4],
                vec![0x7E, 0xFE, 0x76, 0x7E, 0x30, 0x38, 0x40, 0x66],
            ),
            Owned::f32(&[2], vec![1.0 / 448.0, 2.0 / 448.0]),
        ],
        vec![astr("scheme", "per_token"), astr("format", "f8e4m3")],
        1,
    ));
    cases.push((
        "amax_update",
        vec![
            Owned::f32(&[2, 4], vec![1.0, -5.0, 2.0, 0.5, -1.0, 3.0, 0.0, 2.0]),
            Owned::f32(&[2], vec![2.0, 2.0]),
        ],
        vec![astr("scheme", "per_token")],
        1,
    ));
    cases.push((
        "embedding",
        vec![
            Owned::f32(
                &[5, 2],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ),
            Owned::i32(&[3], vec![4, 0, 2]),
        ],
        vec![],
        1,
    ));
    cases.push((
        "gather",
        vec![
            Owned::f32(&[3, 4], (1..=12).map(|v| v as f32).collect()),
            Owned::i32(&[3, 2], vec![1, 2, 1, 2, 1, 2]),
        ],
        vec![ai64("axis", -1)],
        1,
    ));
    cases.push((
        "scatter",
        vec![
            Owned::f32(&[3, 4], vec![0.0; 12]),
            Owned::i32(&[2], vec![3, 1]),
            Owned::f32(&[3, 2], vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
        ],
        vec![ai64("axis", -1)],
        1,
    ));
    cases.push((
        "scatter",
        vec![
            Owned::f32(&[3, 4], vec![0.0; 12]),
            Owned::i32(&[3], vec![1, 1, 2]),
            Owned::f32(
                &[3, 3],
                vec![0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.25, 0.5, 1.0],
            ),
        ],
        vec![astr("reduce", "add")],
        1,
    ));
    cases.push((
        "sdpa",
        vec![
            Owned::f32(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            Owned::f32(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            Owned::f32(&[1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        ],
        vec![],
        1,
    ));
    cases.push((
        "cross_entropy",
        vec![
            Owned::f32(&[2, 3], vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]),
            Owned::i32(&[2], vec![2, 0]),
        ],
        vec![],
        1,
    ));
    cases.push((
        "adamw",
        vec![
            Owned::f32(&[4], vec![1.0, 2.0, 3.0, 4.0]),
            Owned::f32(&[4], vec![0.5, -0.5, 0.25, -0.25]),
            Owned::f32(&[4], vec![0.0; 4]),
            Owned::f32(&[4], vec![0.0; 4]),
        ],
        vec![],
        3,
    ));
    cases.push((
        "topk_router",
        vec![Owned::f32(
            &[2, 4],
            vec![1.0, 3.0, 2.0, 4.0, 4.0, 1.0, 3.0, 2.0],
        )],
        vec![ai64("top_k", 2)],
        2,
    ));
    cases.push((
        "l2norm",
        vec![Owned::f32(&[2, 4], (1..=8).map(|v| v as f32).collect())],
        vec![ai64("dim", -1), af64("eps", 1e-6)],
        1,
    ));
    cases.push((
        "rmsnorm_gated",
        vec![
            Owned::f32(&[2, 4], (1..=8).map(|v| v as f32).collect()),
            Owned::f32(&[4], vec![1.0, 2.0, 1.0, 2.0]),
            Owned::f32(&[2, 4], (1..=8).map(|v| v as f32 * 0.5).collect()),
        ],
        vec![af64("eps", 1e-6), astr("gate_act", "silu")],
        1,
    ));
    cases.push((
        "causal_conv1d",
        vec![
            Owned::f32(&[3, 2], (1..=6).map(|v| v as f32).collect()),
            Owned::f32(&[2, 1, 3], vec![0.5, 1.0, 1.5, 0.25, 0.75, 1.25]),
        ],
        vec![ai64("kernel", 3), astr("activation", "silu")],
        1,
    ));
    cases.push((
        "gated_delta_rule",
        vec![
            Owned::f32(&[1, 4, 2], (1..=8).map(|v| v as f32).collect()),
            Owned::f32(&[1, 4, 2], (1..=8).map(|v| v as f32 * 0.5).collect()),
            Owned::f32(&[1, 4, 2], (1..=8).map(|v| v as f32 * 0.25).collect()),
            Owned::f32(&[1, 4, 1], vec![-0.5, -0.5, -0.5, -0.5]),
            Owned::f32(&[1, 4, 1], vec![0.25, 0.5, 0.75, 1.0]),
        ],
        vec![ai64("chunk_size", 2)],
        1,
    ));
    cases.push((
        "rope",
        vec![
            Owned::f32(&[2, 6], (1..=12).map(|v| v as f32).collect()),
            Owned::f32(&[2, 6], (1..=12).map(|v| v as f32 * 0.5).collect()),
        ],
        vec![
            ai64("rotary_dim", 4),
            af64("theta", 1e7),
            abool("partial_rotary", true),
        ],
        2,
    ));
    cases.push((
        "moe_layer",
        vec![
            Owned::f32(&[1, 2], vec![1.0, 2.0]),
            Owned::f32(&[1, 2], vec![0.25, 0.75]),
            Owned::i32(&[1, 2], vec![2, 0]),
            Owned::f32(&[3, 2, 2], (1..=12).map(|v| v as f32 * 0.25).collect()),
            Owned::f32(&[3, 2, 2], (1..=12).map(|v| v as f32 * 0.5).collect()),
            Owned::f32(&[3, 2, 2], (1..=12).map(|v| v as f32 * 0.5).collect()),
            Owned::f32(&[2, 2], vec![0.5, -0.5, 0.25, -0.25]),
            Owned::f32(&[2, 2], vec![0.25, -0.25, 0.5, -0.5]),
            Owned::f32(&[2, 2], vec![0.5, 0.25, -0.25, 0.5]),
            Owned::f32(&[1, 2], vec![0.1, -0.2]),
        ],
        vec![],
        1,
    ));
    cases
}

#[test]
fn every_op_is_bitwise_deterministic() {
    // The view ops alias their input: their output descriptors must also
    // alias the same buffer both times.
    let view_ops = ["view", "reshape", "transpose", "broadcast"];
    for (name, ins, attrs, n_out) in determinism_cases() {
        let o = op(name);
        let in_refs: Vec<&RsTensor> = ins.iter().map(|t| &t.t).collect();
        let r1 = unsafe { run_op(o, &in_refs, &attrs, n_out) }
            .unwrap_or_else(|e| panic!("{name}: first run failed: {e}"));
        let r2 = unsafe { run_op(o, &in_refs, &attrs, n_out) }
            .unwrap_or_else(|e| panic!("{name}: second run failed: {e}"));
        assert_eq!(r1.len(), r2.len(), "{name}: output count");
        for (a, b) in r1.iter().zip(&r2) {
            assert_eq!(a.t.dtype, b.t.dtype, "{name}: dtype");
            assert_eq!(a.t.dims(), b.t.dims(), "{name}: shape");
            assert_eq!(a.t.strides(), b.t.strides(), "{name}: strides");
            let n = a.t.numel().max(0) as usize * a.t.dtype.byte_width().unwrap_or(1) as usize;
            assert_eq!(
                &a.bytes()[..n],
                &b.bytes()[..n],
                "{name}: buffer bytes differ between runs"
            );
        }
        if view_ops.contains(&name) {
            assert_eq!(
                r1[0].t.data, ins[0].t.data,
                "{name}: view output must alias the input both times"
            );
            // The aliased *contents* must also be identical across runs:
            // walk the output descriptor's strides against the input buffer
            // it aliases (narrow starts at an offset).
            let base = &ins[0].fdata();
            let off = (r1[0].t.data as usize - ins[0].t.data as usize) / 4;
            let v1 = collect_view(&r1[0].t, &base[off..]);
            let v2 = collect_view(&r2[0].t, &base[off..]);
            assert_eq!(v1, v2, "{name}: aliased view contents differ");
        }
    }
}

// ── infer: documented shapes and dtypes ─────────────────────────────────────

#[test]
fn infer_produces_documented_shapes_and_dtypes() {
    unsafe {
        let infer_ok = |o: &'static RsOpDesc,
                        ins: &[&RsTensor],
                        attrs: &[RsAttr],
                        n_out: usize|
         -> Vec<RsTensor> {
            let mut descs: Vec<RsTensor> = (0..n_out).map(|_| RsTensor::default()).collect();
            call_infer(o, ins, &mut descs, attrs).unwrap();
            descs
        };

        // reduce: negative axis and the no-axis scalar case.
        let x = Owned::f32(&[2, 3, 4], vec![0.0; 24]);
        let d = infer_ok(
            op("reduce"),
            &[&x.t],
            &[astr("kind", "sum"), ai64("axis", -1)],
            1,
        );
        assert_eq!(d[0].dims(), &[2, 3]);
        let d = infer_ok(op("reduce"), &[&x.t], &[astr("kind", "sum")], 1);
        assert_eq!(d[0].rank, 0);
        assert_eq!(d[0].numel(), 1);
        assert_eq!(d[0].dtype, RsDtype::F32);

        // reduce keepdim: the reduced axis survives as size 1.
        let d = infer_ok(
            op("reduce"),
            &[&x.t],
            &[
                astr("kind", "sum"),
                ai64("axis", -1),
                abool("keepdim", true),
            ],
            1,
        );
        assert_eq!(d[0].dims(), &[2, 3, 1]);
        assert_eq!(d[0].dtype, RsDtype::F32);

        // compare: equal-shape f32 inputs -> f32 of the same shape.
        let d = infer_ok(op("compare"), &[&x.t, &x.t], &[astr("kind", "eq")], 1);
        assert_eq!(d[0].dims(), &[2, 3, 4]);
        assert_eq!(d[0].dtype, RsDtype::F32);

        // scatter's 'reduce' attribute is validated at infer time.
        let sx = Owned::f32(&[4], vec![0.0; 4]);
        let si = Owned::i32(&[2], vec![0, 1]);
        let sv = Owned::f32(&[2], vec![1.0, 2.0]);
        let d = infer_ok(
            op("scatter"),
            &[&sx.t, &si.t, &sv.t],
            &[astr("reduce", "add")],
            1,
        );
        assert_eq!(d[0].dims(), &[4]);

        // gather: torch convention — indices share x's rank; the gathered
        // axis takes the indices' dim.
        let xi = Owned::i32(&[2, 3, 2], vec![0; 12]);
        let d = infer_ok(op("gather"), &[&x.t, &xi.t], &[ai64("axis", -1)], 1);
        assert_eq!(d[0].dims(), &[2, 3, 2]);

        // cat along -1.
        let a = Owned::f32(&[2, 2], vec![0.0; 4]);
        let b = Owned::f32(&[2, 2], vec![0.0; 4]);
        let d = infer_ok(op("cat"), &[&a.t, &b.t], &[ai64("dim", -1)], 1);
        assert_eq!(d[0].dims(), &[2, 4]);

        // narrow along -1.
        let d = infer_ok(
            op("narrow"),
            &[&x.t],
            &[ai64("dim", -1), ai64("start", 1), ai64("length", 2)],
            1,
        );
        assert_eq!(d[0].dims(), &[2, 3, 2]);

        // transpose defaults to swapping the last two dims.
        let d = infer_ok(op("transpose"), &[&x.t], &[], 1);
        assert_eq!(d[0].dims(), &[2, 4, 3]);
        assert_eq!(d[0].strides(), &[12, 1, 4]);

        // quantize scale shapes per scheme.
        let xq = Owned::f32(&[4, 6], vec![0.0; 24]);
        let d = infer_ok(
            op("quantize"),
            &[&xq.t],
            &[astr("scheme", "per_tensor"), astr("format", "f8e4m3")],
            2,
        );
        assert_eq!(d[0].dtype, RsDtype::F8E4M3);
        assert_eq!(d[0].dims(), &[4, 6]);
        assert_eq!(d[1].dtype, RsDtype::F32);
        assert_eq!(d[1].rank, 0);
        let d = infer_ok(
            op("quantize"),
            &[&xq.t],
            &[astr("scheme", "per_token"), astr("format", "f8e5m2")],
            2,
        );
        assert_eq!(d[0].dtype, RsDtype::F8E5M2);
        assert_eq!(d[1].dims(), &[4]);
        let d = infer_ok(
            op("quantize"),
            &[&xq.t],
            &[
                astr("scheme", "per_block"),
                astr("format", "f8e4m3"),
                ai64s("block", &[2, 3]),
            ],
            2,
        );
        assert_eq!(d[1].dims(), &[2, 2]);
        // Non-divisible blocks are rejected at infer time.
        let err = call_infer(
            op("quantize"),
            &[&xq.t],
            &mut [RsTensor::default(), RsTensor::default()],
            &[
                astr("scheme", "per_block"),
                astr("format", "f8e4m3"),
                ai64s("block", &[2, 4]),
            ],
        )
        .unwrap_err();
        assert!(err.contains("divide"), "unexpected message: {err}");

        // topk_router: weights f32, indices i32.
        let logits = Owned::f32(&[2, 4], vec![0.0; 8]);
        let d = infer_ok(op("topk_router"), &[&logits.t], &[ai64("top_k", 2)], 2);
        assert_eq!(d[0].dims(), &[2, 2]);
        assert_eq!(d[0].dtype, RsDtype::F32);
        assert_eq!(d[1].dims(), &[2, 2]);
        assert_eq!(d[1].dtype, RsDtype::I32);

        // sdpa: [2,3,4] x [2,5,4] x [2,5,6] -> [2,3,6].
        let q = Owned::f32(&[2, 3, 4], vec![0.0; 24]);
        let k = Owned::f32(&[2, 5, 4], vec![0.0; 40]);
        let v = Owned::f32(&[2, 5, 6], vec![0.0; 60]);
        let d = infer_ok(op("sdpa"), &[&q.t, &k.t, &v.t], &[], 1);
        assert_eq!(d[0].dims(), &[2, 3, 6]);

        // cross_entropy: rank-0 scalar loss.
        let logits = Owned::f32(&[2, 3], vec![0.0; 6]);
        let t = Owned::i64(&[2], vec![0, 1]);
        let d = infer_ok(op("cross_entropy"), &[&logits.t, &t.t], &[], 1);
        assert_eq!(d[0].rank, 0);

        // softmax with negative axis; elementwise_binary broadcast shape.
        let d = infer_ok(op("softmax"), &[&x.t], &[ai64("axis", -2)], 1);
        assert_eq!(d[0].dims(), &[2, 3, 4]);
        let s = Owned::f32(&[2, 1], vec![0.0; 2]);
        let o = Owned::f32(&[1, 3], vec![0.0; 3]);
        let d = infer_ok(
            op("elementwise_binary"),
            &[&s.t, &o.t],
            &[astr("kind", "add")],
            1,
        );
        assert_eq!(d[0].dims(), &[2, 3]);

        // adamw: three outputs, all the input shape.
        let p = Owned::f32(&[2, 2], vec![0.0; 4]);
        let g = Owned::f32(&[2, 2], vec![0.0; 4]);
        let m = Owned::f32(&[2, 2], vec![0.0; 4]);
        let vv = Owned::f32(&[2, 2], vec![0.0; 4]);
        let d = infer_ok(op("adamw"), &[&p.t, &g.t, &m.t, &vv.t], &[], 3);
        assert!(d.iter().all(|o| o.dims() == [2, 2]));

        // l2norm: shape preserved for any dim.
        let d = infer_ok(op("l2norm"), &[&x.t], &[ai64("dim", -1)], 1);
        assert_eq!(d[0].dims(), &[2, 3, 4]);

        // rmsnorm_gated: shape preserved; the weight fixes the row width.
        let w = Owned::f32(&[4], vec![0.0; 4]);
        let gate = Owned::f32(&[2, 3, 4], vec![0.0; 24]);
        let d = infer_ok(op("rmsnorm_gated"), &[&x.t, &w.t, &gate.t], &[], 1);
        assert_eq!(d[0].dims(), &[2, 3, 4]);

        // causal_conv1d: shape preserved.
        let cx = Owned::f32(&[5, 3], vec![0.0; 15]);
        let cw = Owned::f32(&[3, 1, 4], vec![0.0; 12]);
        let d = infer_ok(op("causal_conv1d"), &[&cx.t, &cw.t], &[], 1);
        assert_eq!(d[0].dims(), &[5, 3]);

        // gated_delta_rule: [.., S, kh*D] -> [.., S, vh*Dv] (D = Dv = 2).
        let qq = Owned::f32(&[1, 4, 2], vec![0.0; 8]);
        let kk = Owned::f32(&[1, 4, 2], vec![0.0; 8]);
        let vv3 = Owned::f32(&[1, 4, 2], vec![0.0; 8]);
        let gg = Owned::f32(&[1, 4, 1], vec![0.0; 4]);
        let d = infer_ok(
            op("gated_delta_rule"),
            &[&qq.t, &kk.t, &vv3.t, &gg.t, &gg.t],
            &[ai64("chunk_size", 2)],
            1,
        );
        assert_eq!(d[0].dims(), &[1, 4, 2]);

        // rope: two outputs, both the input shape.
        let r = Owned::f32(&[2, 6], vec![0.0; 12]);
        let d = infer_ok(
            op("rope"),
            &[&r.t, &r.t],
            &[ai64("rotary_dim", 4), abool("partial_rotary", true)],
            2,
        );
        assert_eq!(d[0].dims(), &[2, 6]);
        assert_eq!(d[1].dims(), &[2, 6]);

        // sdpa GQA, per-head form [.., S, H, D]: the form is declared, the
        // head counts come from the tensors (q 2 heads, k/v 1).
        let qg = Owned::f32(&[1, 2, 2, 4], vec![0.0; 16]);
        let kg = Owned::f32(&[1, 2, 1, 4], vec![0.0; 8]);
        let vg = Owned::f32(&[1, 2, 1, 4], vec![0.0; 8]);
        let d = infer_ok(
            op("sdpa"),
            &[&qg.t, &kg.t, &vg.t],
            &[abool("per_head", true), abool("causal", true)],
            1,
        );
        assert_eq!(d[0].dims(), &[1, 2, 2, 4]);
    }
}

// ── error handling ──────────────────────────────────────────────────────────

#[test]
fn unknown_attribute_values_are_hard_errors_naming_the_accepted_values() {
    unsafe {
        let x = Owned::f32(&[4], vec![1.0, 2.0, 3.0, 4.0]);
        let mut out = RsTensor::default();
        let err = call_exec(
            op("elementwise_unary"),
            &[&x.t],
            &mut [&mut out],
            &[astr("kind", "bogus")],
        )
        .unwrap_err();
        assert!(
            err.contains("accepted values") && err.contains("silu") && err.contains("bogus"),
            "unary message: {err}"
        );

        let err = call_exec(
            op("quantize"),
            &[&x.t],
            &mut [&mut RsTensor::default(), &mut RsTensor::default()],
            &[astr("scheme", "bogus"), astr("format", "f8e4m3")],
        )
        .unwrap_err();
        assert!(
            err.contains("per_tensor") && err.contains("per_block") && err.contains("bogus"),
            "quantize scheme message: {err}"
        );

        let err = call_exec(
            op("reduce"),
            &[&x.t],
            &mut [&mut out],
            &[astr("kind", "product")],
        )
        .unwrap_err();
        assert!(
            err.contains("sum") && err.contains("amax") && err.contains("product"),
            "reduce message: {err}"
        );

        // compare: an unknown kind names the accepted set.
        let err = call_exec(
            op("compare"),
            &[&x.t, &x.t],
            &mut [&mut out],
            &[astr("kind", "xor")],
        )
        .unwrap_err();
        assert!(
            err.contains("accepted values") && err.contains("eq") && err.contains("xor"),
            "compare message: {err}"
        );

        // scatter: an unknown 'reduce' names the accepted set.
        let sx = Owned::f32(&[4], vec![0.0; 4]);
        let si = Owned::i32(&[2], vec![0, 1]);
        let sv = Owned::f32(&[2], vec![1.0, 2.0]);
        let err = call_exec(
            op("scatter"),
            &[&sx.t, &si.t, &sv.t],
            &mut [&mut out],
            &[astr("reduce", "mul")],
        )
        .unwrap_err();
        assert!(
            err.contains("assign") && err.contains("add") && err.contains("mul"),
            "scatter reduce message: {err}"
        );

        // Missing required attribute is also a non-zero status with a
        // message naming what is required.
        let err = call_exec(op("reduce"), &[&x.t], &mut [&mut out], &[]).unwrap_err();
        assert!(err.contains("required"), "missing kind message: {err}");

        // The new primitives' string attributes: a value the description
        // wrote but the vocabulary does not know is a hard error naming the
        // accepted set — never a silent fallback to the default.
        let g = Owned::f32(&[4], vec![0.0; 4]);
        let err = call_exec(
            op("rmsnorm_gated"),
            &[&x.t, &Owned::f32(&[4], vec![1.0; 4]).t, &g.t],
            &mut [&mut out],
            &[astr("gate_act", "gelu")],
        )
        .unwrap_err();
        assert!(
            err.contains("gate_act") && err.contains("silu") && err.contains("gelu"),
            "rmsnorm_gated gate_act message: {err}"
        );

        let err = call_exec(
            op("causal_conv1d"),
            &[
                &Owned::f32(&[3, 1], vec![0.0; 3]).t,
                &Owned::f32(&[1, 1, 3], vec![0.0; 3]).t,
            ],
            &mut [&mut out],
            &[astr("groups", "dense")],
        )
        .unwrap_err();
        assert!(
            err.contains("groups") && err.contains("channels") && err.contains("dense"),
            "causal_conv1d groups message: {err}"
        );

        let err = call_exec(
            op("gated_delta_rule"),
            &[
                &Owned::f32(&[1, 2, 2], vec![0.0; 4]).t,
                &Owned::f32(&[1, 2, 2], vec![0.0; 4]).t,
                &Owned::f32(&[1, 2, 2], vec![0.0; 4]).t,
                &Owned::f32(&[1, 2, 1], vec![0.0; 2]).t,
                &Owned::f32(&[1, 2, 1], vec![0.0; 2]).t,
            ],
            &mut [&mut out],
            &[astr("state_dtype", "bf16")],
        )
        .unwrap_err();
        assert!(
            err.contains("state_dtype") && err.contains("f32") && err.contains("bf16"),
            "gated_delta_rule state_dtype message: {err}"
        );

        // rope: an odd rotary_dim is refused (pairs cannot be built), and a
        // rotary_dim below D without partial_rotary is refused rather than
        // guessed.
        let r6 = Owned::f32(&[1, 6], vec![0.0; 6]);
        let err = call_exec(
            op("rope"),
            &[&r6.t, &r6.t],
            &mut [&mut RsTensor::default(), &mut RsTensor::default()],
            &[ai64("rotary_dim", 3)],
        )
        .unwrap_err();
        assert!(err.contains("rotary_dim"), "rope odd rotary_dim: {err}");
        let err = call_exec(
            op("rope"),
            &[&r6.t, &r6.t],
            &mut [&mut RsTensor::default(), &mut RsTensor::default()],
            &[ai64("rotary_dim", 4)],
        )
        .unwrap_err();
        assert!(
            err.contains("partial_rotary"),
            "rope partial declaration missing: {err}"
        );
    }
}

#[test]
fn narrow_rejects_null_data_before_pointer_arithmetic() {
    unsafe {
        let x = RsTensor::new(RsDtype::F32, &[3, 4]); // data stays null
        let mut o = RsTensor::default();
        let err = call_exec(
            op("narrow"),
            &[&x],
            &mut [&mut o],
            &[ai64("dim", 1), ai64("start", 2), ai64("length", 1)],
        )
        .unwrap_err();
        assert!(err.contains("null data"), "message: {err}");
    }
}

#[test]
fn execute_revalidates_input_shapes() {
    unsafe {
        // infer rejects this pair (inner dims 2 vs 3); execute must too, and
        // must not silently compute with the wrong k.
        let a = Owned::f32(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Owned::f32(&[3, 2], vec![1.0; 6]);
        let out = Owned::f32(&[2, 2], vec![0.0; 4]);
        let mut o = out.t;
        let err = call_exec(op("matmul"), &[&a.t, &b.t], &mut [&mut o], &[]).unwrap_err();
        assert!(err.contains("inner dims mismatch"), "message: {err}");
    }
}

#[test]
fn dequantize_rejects_a_scale_shape_inconsistent_with_the_declared_scheme() {
    unsafe {
        // Declared per_token wants scale [2]; give it [4].
        let q = Owned::u8(RsDtype::F8E4M3, &[2, 4], vec![0x7E; 8]);
        let scale = Owned::f32(&[4], vec![0.0; 4]);
        let err = call_exec(
            op("dequantize"),
            &[&q.t, &scale.t],
            &mut [&mut RsTensor::default()],
            &[astr("scheme", "per_token"), astr("format", "f8e4m3")],
        )
        .unwrap_err();
        assert!(
            err.contains("inconsistent") && err.contains("declared scheme"),
            "scale shape message: {err}"
        );
    }
}

#[test]
fn non_contiguous_inputs_are_rejected_with_a_clear_message() {
    unsafe {
        let x = Owned::f32(&[3, 3], vec![1.0; 9]);
        // Transpose produces a non-contiguous [3, 3] view (strides [1, 3]),
        // whose shape still passes matmul's dim checks — so the contiguity
        // rule is what must fire.
        let t = run_op(
            op("transpose"),
            &[&x.t],
            &[ai64("dim0", 0), ai64("dim1", 1)],
            1,
        )
        .unwrap();
        let y = Owned::f32(&[3, 3], vec![0.0; 9]);
        let out = Owned::f32(&[3, 3], vec![0.0; 9]);
        let mut o = out.t;
        let err = call_exec(op("matmul"), &[&t[0].t, &y.t], &mut [&mut o], &[]).unwrap_err();
        assert!(
            err.contains("non-contiguous") && err.contains("later improvement"),
            "contiguity message: {err}"
        );
    }
}

#[test]
fn wrong_output_shape_or_dtype_is_rejected_not_corrupted() {
    unsafe {
        let a = Owned::f32(&[2, 3], vec![1.0; 6]);
        let b = Owned::f32(&[3, 2], vec![1.0; 6]);
        // matmul infers [2, 2]; hand the executor a [2, 3] output instead.
        let wrong = Owned::f32(&[2, 3], vec![0.0; 6]);
        let mut w = wrong.t;
        let err = call_exec(op("matmul"), &[&a.t, &b.t], &mut [&mut w], &[]).unwrap_err();
        assert!(
            err.contains("shape") && err.contains("infer()"),
            "message: {err}"
        );
        // Null data is rejected.
        let mut null_out = RsTensor::new(RsDtype::F32, &[2, 2]);
        let err = call_exec(op("matmul"), &[&a.t, &b.t], &mut [&mut null_out], &[]).unwrap_err();
        assert!(err.contains("null data"), "message: {err}");
        // Wrong input dtype is rejected.
        let bad = Owned::u8(RsDtype::U8, &[2, 3], vec![1; 6]);
        let good_out = Owned::f32(&[2, 2], vec![0.0; 4]);
        let mut go = good_out.t;
        let err = call_exec(op("matmul"), &[&bad.t, &b.t], &mut [&mut go], &[]).unwrap_err();
        assert!(err.contains("dtype"), "message: {err}");
    }
}
