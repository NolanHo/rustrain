//! `moe_layer` — Qwen3.6's sparse MoE layer as ONE explicit operator
//! (option A of `docs/design/op-vocabulary.md` §5).
//!
//! The router (`topk_router`) runs upstream; dispatch and combine are
//! `all_to_all({tp, ep})` exchanges whose per-rank token counts are only
//! known at run time, so the whole layer is a single node with static
//! `[.., H]` in/out shapes — the reference implementation runs single-rank
//! and never sees the exchange; the collectives are declared on the
//! descriptor because layout arithmetic cannot derive them (the routing is
//! data, `op-vocabulary.md` §4).
//!
//! Numerics are HF's `Qwen3_5MoeSparseMoeBlock` (transformers
//! `modeling_qwen3_5_moe.py`): per token, **every selected expert runs**
//! (dropless — no capacity truncation, which is exactly why the fused form
//! exists), each expert computes `down(silu(gate(x)) * up(x))`, the result is
//! scaled by the (already normalised) routing probability and summed, and the
//! shared expert's output is added gated by `sigmoid(shared_expert_gate(x))`.
//! All accumulations run in fixed ascending orders (k, then j, then the
//! hidden axis), so two runs are bitwise identical.

use rustrain_abi::ffi::{RsAttrs, RsDtype, RsMemReq, RsTensor};

use crate::dispatch::{Call, run};
use crate::error::{OpResult, err, fail};
use crate::tensor::{expect_dtype, expect_out, set_output_desc};

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

/// One validated moe_layer call's plan, shared by infer and execute so direct
/// execute calls get the same validation.
pub(crate) struct MoePlan {
    /// h's rank; the output keeps h's whole shape.
    rank: usize,
    /// Tokens: the product of h's leading dims (the `..` of `[.., H]`).
    rows: usize,
    /// Hidden size (h's last dim, also the weight convention's H).
    h: usize,
    /// Top-k: the routing tensors' last dim (the router's declaration, not
    /// an attribute of this op).
    k: usize,
    /// Number of experts (gate_up_proj dim 0).
    e: usize,
    /// Per-expert intermediate size (gate_up_proj's middle axis is 2*I).
    i: usize,
}

/// silu(x) = x * sigmoid(x), the `x / (1 + e^-x)` form `elementwise_unary`
/// uses — exact for both tails, and identical here so the fused layer agrees
/// with the vocabulary's own activation bitwise.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// sigmoid(x) = 1 / (1 + e^-x), the `elementwise_unary` form.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Validates the nine declared inputs and derives the sizes. The input
/// contract (see the operator doc in `lib.rs`):
///
/// 0. `h` f32 [.., H], rank >= 2 — the hidden states; rows = prod(leading).
/// 1. `routing_weights` f32 [rows, K] — topk_router output 0, the softmax
///    probabilities of the selected experts, already renormalised when the
///    router runs norm_topk_prob. Used as-is, never renormalised here.
/// 2. `routing_indices` i32/i64 [rows, K] — topk_router output 1.
/// 3. `experts_gate_up_proj` f32 [E, 2*I, H] — checkpoint orientation
///    ([out, in] per expert): gate = rows [0, I), up = rows [I, 2*I) — the
///    halves `F.linear(x, W).chunk(2, dim=-1)` splits.
/// 4. `experts_down_proj` f32 [E, H, I] ([out, in] per expert).
/// 5. `shared_gate_proj` f32 [I, H].
/// 6. `shared_up_proj` f32 [I, H].
/// 7. `shared_down_proj` f32 [H, I].
/// 8. `shared_expert_gate` f32 [1, H].
fn moe_plan(c: &mut Call) -> OpResult<MoePlan> {
    c.expect_arity((9, 9))?;
    c.expect_out_count(1)?;

    let ht = c.in_t(0);
    expect_dtype(ht, RsDtype::F32, c.op, "h")?;
    if ht.rank < 2 {
        return Err(fail!(
            c.op,
            "moe_layer 'h' must be [.., H] of rank >= 2 (tokens x hidden), got rank {}",
            ht.rank
        ));
    }
    let rank = ht.rank as usize;
    let h = ht.shape[rank - 1];
    if h < 1 {
        return Err(fail!(
            c.op,
            "moe_layer 'h' last dim (hidden) must be >= 1, got {h}"
        ));
    }
    let mut rows: i64 = 1;
    for &d in &ht.shape[..rank - 1] {
        if d < 0 {
            return Err(err(c.op, format!("input 'h' has a negative dimension {d}")));
        }
        rows = rows.checked_mul(d).ok_or_else(|| {
            err(
                c.op,
                "input 'h' leading-dims product overflows i64".to_string(),
            )
        })?;
    }

    let wt = c.in_t(1);
    expect_dtype(wt, RsDtype::F32, c.op, "routing_weights")?;
    if wt.rank != 2 || wt.shape[0] != rows {
        return Err(fail!(
            c.op,
            "routing_weights must be [rows={rows}, K] to match h's {:?}, got {:?}",
            ht.dims(),
            wt.dims()
        ));
    }
    let k = wt.shape[1];
    if k < 1 {
        return Err(fail!(
            c.op,
            "routing_weights last dim (top-k K) must be >= 1, got {k}"
        ));
    }

    let it = c.in_t(2);
    match it.dtype {
        RsDtype::I32 | RsDtype::I64 => {}
        other => {
            return Err(err(
                c.op,
                format!("input 'routing_indices' has dtype {other}, expected i32 or i64"),
            ));
        }
    }
    if it.rank != 2 || it.shape[0] != rows || it.shape[1] != k {
        return Err(fail!(
            c.op,
            "routing_indices must be [rows={rows}, K={k}] like routing_weights, got {:?}",
            it.dims()
        ));
    }

    let gut = c.in_t(3);
    expect_dtype(gut, RsDtype::F32, c.op, "experts_gate_up_proj")?;
    if gut.rank != 3 {
        return Err(fail!(
            c.op,
            "experts_gate_up_proj must be [E, 2*I, H], got rank {}",
            gut.rank
        ));
    }
    let (e, i2, gh) = (gut.shape[0], gut.shape[1], gut.shape[2]);
    if e < 1 || i2 < 2 || i2 % 2 != 0 {
        return Err(fail!(
            c.op,
            "experts_gate_up_proj must be [E, 2*I, H] with E >= 1, I >= 1, got {:?}",
            gut.dims()
        ));
    }
    if gh != h {
        return Err(fail!(
            c.op,
            "experts_gate_up_proj hidden dim {gh} must equal h's hidden dim {h}"
        ));
    }
    let i = i2 / 2;

    let dnt = c.in_t(4);
    expect_dtype(dnt, RsDtype::F32, c.op, "experts_down_proj")?;
    if dnt.rank != 3 || dnt.shape[0] != e || dnt.shape[1] != h || dnt.shape[2] != i {
        return Err(fail!(
            c.op,
            "experts_down_proj must be [E={e}, H={h}, I={i}], got {:?}",
            dnt.dims()
        ));
    }

    for (idx, who) in [(5usize, "shared_gate_proj"), (6, "shared_up_proj")] {
        let t = c.in_t(idx);
        expect_dtype(t, RsDtype::F32, c.op, who)?;
        if t.rank != 2 || t.shape[0] != i || t.shape[1] != h {
            return Err(fail!(
                c.op,
                "{who} must be [I={i}, H={h}], got {:?}",
                t.dims()
            ));
        }
    }

    let sdt = c.in_t(7);
    expect_dtype(sdt, RsDtype::F32, c.op, "shared_down_proj")?;
    if sdt.rank != 2 || sdt.shape[0] != h || sdt.shape[1] != i {
        return Err(fail!(
            c.op,
            "shared_down_proj must be [H={h}, I={i}], got {:?}",
            sdt.dims()
        ));
    }

    let sgt = c.in_t(8);
    expect_dtype(sgt, RsDtype::F32, c.op, "shared_expert_gate")?;
    if sgt.rank != 2 || sgt.shape[0] != 1 || sgt.shape[1] != h {
        return Err(fail!(
            c.op,
            "shared_expert_gate must be [1, H={h}], got {:?}",
            sgt.dims()
        ));
    }

    if k > e {
        return Err(fail!(
            c.op,
            "routing selects K={k} of E={e} experts; top-k must be <= the expert count"
        ));
    }

    Ok(MoePlan {
        rank,
        rows: rows as usize,
        h: h as usize,
        k: k as usize,
        e: e as usize,
        i: i as usize,
    })
}

fn moe_infer_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    let plan = moe_plan(c)?;
    let ht = c.in_t(0);
    let o = c.out_t(0);
    // Static in/out: the output keeps h's exact shape ([b, s, H], [s, H],
    // whatever the plan declares) — the whole point of option A.
    set_output_desc(o, RsDtype::F32, &ht.shape[..plan.rank]);
    Ok(())
}

fn moe_exec_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    let plan = moe_plan(c)?;
    let ht = c.in_t(0);
    expect_out(c.out_t(0), c.op, RsDtype::F32, &ht.shape[..plan.rank])?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let hv = unsafe { crate::tensor::f32_in(c.op, "h", ht) }?;
    let wv = unsafe { crate::tensor::f32_in(c.op, "routing_weights", c.in_t(1)) }?;
    let idx = unsafe { crate::tensor::indices_i64(c.op, "routing_indices", c.in_t(2)) }?;
    let gu = unsafe { crate::tensor::f32_in(c.op, "experts_gate_up_proj", c.in_t(3)) }?;
    let dn = unsafe { crate::tensor::f32_in(c.op, "experts_down_proj", c.in_t(4)) }?;
    let sg = unsafe { crate::tensor::f32_in(c.op, "shared_gate_proj", c.in_t(5)) }?;
    let su = unsafe { crate::tensor::f32_in(c.op, "shared_up_proj", c.in_t(6)) }?;
    let sd = unsafe { crate::tensor::f32_in(c.op, "shared_down_proj", c.in_t(7)) }?;
    let sgg = unsafe { crate::tensor::f32_in(c.op, "shared_expert_gate", c.in_t(8)) }?;
    let mut ov = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;

    let hs = hv.as_slice().expect("contiguous");
    let ws = wv.as_slice().expect("contiguous");
    let gus = gu.as_slice().expect("contiguous");
    let dns = dn.as_slice().expect("contiguous");
    let sgs = sg.as_slice().expect("contiguous");
    let sus = su.as_slice().expect("contiguous");
    let sds = sd.as_slice().expect("contiguous");
    let sggs = sgg.as_slice().expect("contiguous");
    let os = ov.as_slice_mut().expect("contiguous");

    let (rows, h, k, e, i) = (plan.rows, plan.h, plan.k, plan.e, plan.i);

    os.fill(0.0);
    // One H-element accumulation buffer per call (the memory reporter's
    // 4*H bytes): each expert's output is built fully, scaled by its routing
    // weight, and added — HF's `index_add_` structure, with the expert order
    // fixed to ascending k (deterministic; HF's `nonzero()` order is not).
    let mut acc = vec![0.0f32; h];
    for r in 0..rows {
        let x = &hs[r * h..(r + 1) * h];

        // Dropless: every selected expert runs.
        for kk in 0..k {
            let ei = idx[r * k + kk];
            if ei < 0 || ei >= e as i64 {
                return Err(fail!(
                    c.op,
                    "token {r}: routing slot {kk} selects expert {ei}, outside [0, {e}) — \
                     indices are used as-is, no wrap-around"
                ));
            }
            let ei = ei as usize;
            let w = ws[r * k + kk];
            // Per-expert weight windows: gate_up_proj[e] is [2*I, H] (gate
            // rows [0, I), up rows [I, 2*I)), down_proj[e] is [H, I].
            let ge = &gus[ei * 2 * i * h..(ei + 1) * 2 * i * h];
            let de = &dns[ei * h * i..(ei + 1) * h * i];
            acc.fill(0.0);
            for j in 0..i {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for hh in 0..h {
                    g += x[hh] * ge[j * h + hh];
                    u += x[hh] * ge[(i + j) * h + hh];
                }
                let a = silu(g) * u;
                for hh in 0..h {
                    acc[hh] += a * de[hh * i + j];
                }
            }
            for hh in 0..h {
                os[r * h + hh] += w * acc[hh];
            }
        }

        // Shared expert, gated: sigmoid(shared_expert_gate @ x) times the
        // same silu-gated MLP, added once per token.
        let mut gs = 0.0f32;
        for hh in 0..h {
            gs += x[hh] * sggs[hh];
        }
        let gate = sigmoid(gs);
        acc.fill(0.0);
        for j in 0..i {
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            for hh in 0..h {
                g += x[hh] * sgs[j * h + hh];
                u += x[hh] * sus[j * h + hh];
            }
            let a = silu(g) * u;
            for hh in 0..h {
                acc[hh] += a * sds[hh * i + j];
            }
        }
        for hh in 0..h {
            os[r * h + hh] += gate * acc[hh];
        }
    }
    Ok(())
}

/// `moe_layer`'s memory reporter: the fused body allocates one H-element f32
/// accumulation buffer per call (each expert's output is built there before
/// being scaled into the output row). The workspace rule (lib.rs): an op that
/// needs scratch must report it here — a NULL memory slot means "cannot plan"
/// to the compiler. Like sdpa's, the reporter returns zeros when given no
/// live tensors (the shared zero-reporter contract the null-io test pins).
///
/// # Safety
/// `io` must be null or point to `n_io` live descriptors; `out` must be null
/// or a writable `RsMemReq`.
pub(crate) unsafe extern "C" fn moe_memory(
    io: *const *const RsTensor,
    n_io: u32,
    _attrs: *const RsAttrs,
    out: *mut RsMemReq,
) -> i32 {
    if out.is_null() {
        let e = err("moe_layer", "null RsMemReq output pointer");
        crate::error::set_last_error(&e);
        return e.status();
    }
    // SAFETY: checked non-null above.
    unsafe { *out = RsMemReq::default() };
    if io.is_null() || n_io < 9 {
        return 0;
    }
    // SAFETY: the ABI contract says `io` points to live descriptors.
    let ht = unsafe { &**io };
    if ht.rank >= 2 {
        let h = ht.shape[ht.rank as usize - 1].max(0) as u64;
        // One f32 buffer of H elements.
        unsafe { (*out).workspace_bytes = 4 * h };
    }
    0
}

infer_entry!(moe_infer, "moe_layer", moe_infer_body);
exec_entry!(moe_exec, "moe_layer", moe_exec_body);
