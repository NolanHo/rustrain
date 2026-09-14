//! Composite operators with their declared primitive expansions (contract
//! R-4): `sdpa`, `cross_entropy`, `adamw`, `topk_router`.
//!
//! The fused bodies below are executed directly; the expansions describe the
//! equivalent primitive composition over the local-tensor-id convention in
//! `rustrain_op.h` (`[0, n_inputs)` = parent inputs, then parent outputs,
//! then temporaries).
//!
//! **Expansion attribute caveat** (ABI limitation, reported rather than
//! patched — `rustrain-abi` is out of this crate's write scope): the current
//! `ExpansionSpec::node` API cannot attach per-node attributes, so nodes that
//! need one (`elementwise_unary` `kind`, `reduce` `kind`, scalar `rhs`) are
//! declared with their required attributes documented here in the operator
//! doc strings. A conformance checker must supply those per-node attributes
//! (e.g. by extending `ExpansionSpec` with `node_with_attrs`).

use rustrain_abi::author::ExpansionSpec;
use rustrain_abi::ffi::{MAX_RANK, RsAttrs, RsDtype, RsMemReq, RsTensor};

use crate::attrs::{attr_bool, attr_f64, attr_i64};
use crate::dispatch::{Call, run};
use crate::error::{OpResult, err, fail};
use crate::tensor::{SmallShape, expect_out, set_output_desc};

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

// ── sdpa ────────────────────────────────────────────────────────────────────

/// One validated sdpa call's plan: the head decomposition, the scale and the
/// mask/causal switches. Shared by infer and execute so direct execute calls
/// get the same validation. `headed` distinguishes the two declared forms:
/// with `num_heads` declared the inputs are per-head `[.., S, H, D]` (the
/// plan's sequence-first layout, S at -3, H at -2, D at -1); without it the
/// legacy flat form `[.., S, D]` (S at -2, one head).
struct SdpaPlan {
    rank: usize,
    s: usize,
    t: usize,
    num_heads: usize,
    num_kv: usize,
    d: usize,
    dv: usize,
    scale: f32,
    causal: bool,
    /// num_heads was declared: inputs are per-head [.., S, H, D].
    headed: bool,
}

fn sdpa_plan(
    q: &RsTensor,
    k: &RsTensor,
    v: &RsTensor,
    a: &RsAttrs,
    op: &'static str,
) -> OpResult<SdpaPlan> {
    for (i, t) in [(0usize, q), (1, k), (2, v)] {
        if t.dtype != RsDtype::F32 {
            return Err(err(
                op,
                format!("input {i} has dtype {}, expected f32", t.dtype),
            ));
        }
    }
    let rank = q.rank as usize;
    if rank < 2 || k.rank as usize != rank || v.rank as usize != rank {
        return Err(fail!(
            op,
            "sdpa expects q, k, v of equal rank >= 2, got ranks {}, {}, {}",
            q.rank,
            k.rank,
            v.rank
        ));
    }
    let headed = attr_bool(a, "per_head").unwrap_or(false);
    // The per-head *form* is declared; the head *counts* are read from the
    // tensors' own axes. An absolute count cannot survive sharding: the plan
    // hands this node the rank's local slice, so a declared global head count
    // would contradict the tensor it is supposed to describe. GQA grouping is
    // therefore q_heads / kv_heads (validated divisible), which is exactly the
    // model's grouping whenever the shard splits both head axes proportionally
    // — and is the reason tp > 2 against 2 kv heads needs KV replication
    // (qwen36-5d-example.md §4).
    let (batch_axes, s, hq, d) = if headed {
        if rank < 3 {
            return Err(fail!(
                op,
                "sdpa with 'num_heads' expects per-head inputs [.., S, H, D] of rank >= 3,                  got rank {rank}"
            ));
        }
        (
            rank - 3,
            q.shape[rank - 3] as usize,
            q.shape[rank - 2] as usize,
            q.shape[rank - 1] as usize,
        )
    } else {
        (
            rank - 2,
            q.shape[rank - 2] as usize,
            1,
            q.shape[rank - 1] as usize,
        )
    };
    let num_heads = hq as i64;
    let num_kv = if headed { k.shape[rank - 2] } else { 1 };
    if num_heads < 1 || num_kv < 1 || num_heads % num_kv != 0 {
        return Err(fail!(
            op,
            "sdpa head axes: q has {num_heads} head(s), k has {num_kv}; the query heads must be a \
             positive multiple of the key/value heads (GQA) — the counts come from the input \
             tensors, never from an attribute"
        ));
    }
    for dd in 0..batch_axes {
        if q.shape[dd] != k.shape[dd] || q.shape[dd] != v.shape[dd] {
            return Err(fail!(
                op,
                "sdpa batch dim {dd} mismatch: q={}, k={}, v={} (batch dims must be identical)",
                q.shape[dd],
                k.shape[dd],
                v.shape[dd]
            ));
        }
    }
    let (t, kd) = (
        k.shape[rank - 2 - headed as usize] as usize,
        k.shape[rank - 1] as usize,
    );
    if v.shape[rank - 2 - headed as usize] as usize != t {
        return Err(fail!(
            op,
            "sdpa dim mismatch: q {:?}, k {:?}, v {:?} (k and v share the sequence length T)",
            q.dims(),
            k.dims(),
            v.dims()
        ));
    }

    if kd != d {
        return Err(fail!(op, "sdpa k head dim ({kd}) must equal q's ({d})"));
    }
    let (v_heads, vd) = if headed {
        (v.shape[rank - 2] as usize, v.shape[rank - 1] as usize)
    } else {
        (1, v.shape[rank - 1] as usize)
    };
    if v_heads != num_kv as usize {
        return Err(fail!(
            op,
            "sdpa v head axis is {v_heads} but k's is {num_kv}: the per-head form requires k and \
             v to carry the same key/value head count"
        ));
    }
    // The declared scale convention: explicit `scale` wins; absent, the
    // per-head form defaults to 1/sqrt(head_dim) and the legacy flat form
    // keeps its pre-D5 default of 1.0 — additive with the old behaviour as the
    // default, and the declared expansion's softmax node still matches the
    // legacy path.
    let scale = match attr_f64(a, "scale") {
        Some(s) => s as f32,
        None if headed => 1.0 / (d as f32).sqrt(),
        None => 1.0,
    };
    let causal = crate::attrs::attr_bool(a, "causal").unwrap_or(false);
    Ok(SdpaPlan {
        rank,
        s,
        t,
        num_heads: num_heads as usize,
        num_kv: num_kv as usize,
        d,
        dv: vd,
        scale,
        causal,
        headed,
    })
}

/// The sdpa output shape: the input shape with the head dim replaced by Dv —
/// each of the num_heads output heads carries the value head width Dv (which
/// equals D in Qwen3.6, where the output is q-shaped).
fn sdpa_out_shape(q: &RsTensor, headed: bool, dv: usize) -> SmallShape {
    let rank = q.rank as usize;
    let mut shape = SmallShape {
        len: rank,
        dims: [0; MAX_RANK],
    };
    shape.dims[..rank].copy_from_slice(q.dims());
    shape.dims[rank - 1] = dv as i64;
    let _ = headed;
    shape
}

fn sdpa_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((3, 4))?;
    c.expect_out_count(1)?;
    let plan = sdpa_plan(c.in_t(0), c.in_t(1), c.in_t(2), a, c.op)?;
    if c.n_in() == 4 {
        // The additive padding/causal mask must broadcast (right-aligned) to
        // [.., S, T]; 0.0 attends, -inf masks (the HF attention_mask form).
        let m = c.in_t(3);
        if m.dtype != RsDtype::F32 {
            return Err(err(
                c.op,
                format!("sdpa mask has dtype {}, expected f32", m.dtype),
            ));
        }
        let batch_axes = plan.rank - if plan.headed { 3 } else { 2 };
        let mut want = SmallShape {
            len: plan.rank,
            dims: [0; MAX_RANK],
        };
        want.dims[..batch_axes].copy_from_slice(&c.in_t(0).dims()[..batch_axes]);
        want.dims[batch_axes] = plan.s as i64;
        want.dims[batch_axes + 1] = plan.t as i64;
        want.len = batch_axes + 2;
        crate::tensor::broadcast_shape_small(m.dims(), want.as_slice(), c.op).map_err(|e| {
            err(
                c.op,
                format!("sdpa mask must broadcast to {:?}: {e}", want.as_slice()),
            )
        })?;
    }
    let shape = sdpa_out_shape(c.in_t(0), plan.headed, plan.dv);
    let o = c.out_t(0);
    set_output_desc(o, RsDtype::F32, shape.as_slice());
    Ok(())
}

fn sdpa_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((3, 4))?;
    c.expect_out_count(1)?;
    let q = c.in_t(0);
    let plan = sdpa_plan(q, c.in_t(1), c.in_t(2), a, c.op)?;
    let SdpaPlan {
        rank,
        s,
        t,
        num_heads,
        num_kv,
        d,
        dv,
        scale,
        causal,
        headed,
    } = plan;
    let out_shape = sdpa_out_shape(q, headed, dv);
    expect_out(c.out_t(0), c.op, RsDtype::F32, out_shape.as_slice())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let qv = unsafe { crate::tensor::f32_in(c.op, "q", q) }?;
    let kv = unsafe { crate::tensor::f32_in(c.op, "k", c.in_t(1)) }?;
    let vv = unsafe { crate::tensor::f32_in(c.op, "v", c.in_t(2)) }?;
    let mut ov = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let qs = qv.as_slice().expect("contiguous");
    let ks = kv.as_slice().expect("contiguous");
    let vs = vv.as_slice().expect("contiguous");
    let os = ov.as_slice_mut().expect("contiguous");
    let batch_axes = rank - if headed { 3 } else { 2 };
    let batch_dims: Vec<usize> = q.dims()[..batch_axes].iter().map(|&x| x as usize).collect();
    let batches: usize = batch_dims.iter().product();

    // The additive mask, right-aligned broadcast to [.., S, T]. Read with
    // plain index math (size-1 dims stretch, missing leading dims stretch)
    // straight from the caller's buffer — no copy, no extra scratch, and the
    // memory reporter stays exact.
    let mask_raw: Option<(*const f32, usize, Vec<usize>)> = if c.n_in() == 4 {
        let m = c.in_t(3);
        // Validates dtype/contiguity; the descriptor and its buffer are
        // owned by the caller and outlive this call.
        let mv = unsafe { crate::tensor::f32_in(c.op, "mask", m) }?;
        let dims: Vec<usize> = m.dims().iter().map(|&x| x.max(1) as usize).collect();
        Some((mv.as_ptr(), mv.len(), dims))
    } else {
        None
    };

    // Two scratch buffers (scores + probs), reused per (batch, head) — the
    // memory reporter accounts for exactly these S*T elements each.
    let mut scores = vec![0.0f32; s * t];
    let mut probs = vec![0.0f32; s * t];
    // SAFETY: the mask buffer is the caller's, live for the whole call, and
    // `f32_in` above validated dtype and contiguity.
    let mask_slice = mask_raw
        .as_ref()
        .map(|(p, n, _)| unsafe { std::slice::from_raw_parts(*p, *n) });
    let mut prefix = vec![0usize; batch_axes];
    // Strides of the last (S/H, head, D) axes for the per-batch head reads.
    let r = num_heads / num_kv;
    for b in 0..batches {
        let mut rem = b;
        for dd in (0..batch_axes).rev() {
            prefix[dd] = rem % batch_dims[dd];
            rem /= batch_dims[dd];
        }
        for h in 0..num_heads {
            let kv_h = h / r;
            // Per-head offsets: headed [.., S, H, D] with strides S: H*D,
            // heads: D; legacy [.., S, D] with strides S: D. Explicit math.
            let (q_base, q_step, k_base, k_step, v_base, v_step, o_base, o_step) = if headed {
                (
                    b * s * num_heads * d + h * d,
                    num_heads * d,
                    b * t * num_kv * d + kv_h * d,
                    num_kv * d,
                    b * t * num_kv * dv + kv_h * dv,
                    num_kv * dv,
                    b * s * num_heads * dv + h * dv,
                    num_heads * dv,
                )
            } else {
                (b * s * d, d, b * t * d, d, b * t * dv, dv, b * s * dv, dv)
            };
            // scores = q @ k^T * scale, then the declared mask and the causal
            // triangle. All reductions in ascending order — deterministic.
            for i in 0..s {
                for j in 0..t {
                    let mut acc = 0.0f32;
                    for dd in 0..d {
                        acc += qs[q_base + i * q_step + dd] * ks[k_base + j * k_step + dd];
                    }
                    acc *= scale;
                    if let (Some(mask), Some((_, _, mdims))) = (&mask_slice, &mask_raw) {
                        acc += mask_value(mask, mdims, &prefix, i, j);
                    }
                    if causal && j > i {
                        acc = f32::NEG_INFINITY;
                    }
                    scores[i * t + j] = acc;
                }
            }
            // Stable row softmax over T; -inf entries (masked positions)
            // exponentiate to 0 exactly.
            for i in 0..s {
                let row = &scores[i * t..(i + 1) * t];
                let m = row.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                let sum = row.iter().fold(0.0f32, |acc, &x| acc + (x - m).exp());
                for j in 0..t {
                    probs[i * t + j] = (row[j] - m).exp() / sum;
                }
            }
            for i in 0..s {
                for jj in 0..dv {
                    let mut acc = 0.0f32;
                    for j in 0..t {
                        acc += probs[i * t + j] * vs[v_base + j * v_step + jj];
                    }
                    os[o_base + i * o_step + jj] = acc;
                }
            }
        }
    }
    Ok(())
}

fn mask_value(mask: &[f32], mdims: &[usize], prefix: &[usize], i: usize, j: usize) -> f32 {
    let rank = prefix.len() + 2;
    let mut offset = 0usize;
    let mut stride = 1usize;
    for d in (0..rank).rev() {
        let coord = if d < prefix.len() {
            prefix[d]
        } else if d == prefix.len() {
            i
        } else {
            j
        };
        let md = d as i64 + mdims.len() as i64 - rank as i64;
        let dim = if md >= 0 && (md as usize) < mdims.len() {
            mdims[md as usize]
        } else {
            1
        };
        let local = if dim == 1 { 0 } else { coord % dim };
        offset += local * stride;
        stride *= dim.max(1);
    }
    mask[offset]
}

/// Declared expansion of `sdpa`: `bmm(q, k, transpose_b=true)` -> `softmax`
/// -> `bmm(p, v)`. The transposed-B form is chosen because this provider's
/// matmul family requires contiguous inputs (documented policy), and a
/// `transpose` view would be strided and therefore unusable downstream.
///
/// The expansion describes the **legacy flat form** (num_heads =
/// num_kv_heads = 1, scale 1.0, no causal/mask) — the only form the
/// primitive vocabulary can express, which is precisely why the fused body
/// exists. GQA/causal cases cannot be replayed and the conformance gate
/// reports that as a skip, never as a mismatch.
///
/// Node attribute table (the current `ExpansionSpec` API cannot attach
/// per-node attrs — see the module doc):
/// node 0: bmm, attrs {transpose_b: true};
/// node 1: softmax, attrs {axis: -1, scale: 1.0} (defaults);
/// node 2: bmm, no attrs.
pub(crate) fn sdpa_expansion() -> ExpansionSpec {
    let mut e = ExpansionSpec::new(3, 1);
    let s = e.temp();
    let p = e.temp();
    e.node("bmm", &[0, 1], &[s])
        .node("softmax", &[s], &[p])
        .node("bmm", &[p, 2], &[3])
}

/// `sdpa`'s memory reporter. The fused body needs two f32 scratch buffers of
/// S*T elements (scores and probabilities), allocated per call. The
/// workspace rule (lib.rs): an op that needs scratch must report it here —
/// a NULL memory slot means "cannot plan" to the compiler, and silently
/// reporting zero while allocating would break planning. Every other op
/// registers the shared zero reporter.
///
/// # Safety
/// `io` must be null or point to `n_io` live descriptors; `out` must be
/// null or a writable `RsMemReq`.
pub(crate) unsafe extern "C" fn sdpa_memory(
    io: *const *const RsTensor,
    n_io: u32,
    _attrs: *const RsAttrs,
    out: *mut RsMemReq,
) -> i32 {
    if out.is_null() {
        let e = err("sdpa", "null RsMemReq output pointer");
        crate::error::set_last_error(&e);
        return e.status();
    }
    // SAFETY: checked non-null above.
    unsafe { *out = RsMemReq::default() };
    if io.is_null() || n_io < 2 {
        return 0;
    }
    // SAFETY: the ABI contract says `io` points to live descriptors.
    let q = unsafe { &**io };
    let k = unsafe { &**io.add(1) };
    if q.rank >= 3 && k.rank >= 3 {
        let s = q.shape[q.rank as usize - 2].max(0) as u64;
        let t = k.shape[k.rank as usize - 2].max(0) as u64;
        // Two f32 buffers of S*T elements each.
        unsafe { (*out).workspace_bytes = 8 * s * t };
    }
    0
}

// ── cross_entropy ───────────────────────────────────────────────────────────

fn cross_entropy_infer_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let logits = c.in_t(0);
    let targets = c.in_t(1);
    if logits.dtype != RsDtype::F32 {
        return Err(err(
            c.op,
            format!("input 'logits' has dtype {}, expected f32", logits.dtype),
        ));
    }
    match targets.dtype {
        RsDtype::I32 | RsDtype::I64 => {}
        other => {
            return Err(err(
                c.op,
                format!("input 'targets' has dtype {other}, expected i32 or i64"),
            ));
        }
    }
    if logits.rank != 2 {
        return Err(err(
            c.op,
            format!(
                "cross_entropy expects logits [N, C], got rank {}",
                logits.rank
            ),
        ));
    }
    if targets.rank != 1 || targets.shape[0] != logits.shape[0] {
        return Err(err(
            c.op,
            format!(
                "cross_entropy targets must be [N={}], got {:?}",
                logits.shape[0],
                targets.dims()
            ),
        ));
    }
    // Loss is a rank-0 scalar (mean reduction, documented).
    set_output_desc(c.out_t(0), RsDtype::F32, &[]);
    Ok(())
}

fn cross_entropy_exec_body(c: &mut Call, _a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((2, 2))?;
    c.expect_out_count(1)?;
    let logits = c.in_t(0);
    let targets = c.in_t(1);
    if logits.dtype != RsDtype::F32 {
        return Err(err(
            c.op,
            format!("input 'logits' has dtype {}, expected f32", logits.dtype),
        ));
    }
    if logits.rank != 2 || targets.rank != 1 || targets.shape[0] != logits.shape[0] {
        return Err(err(
            c.op,
            format!(
                "cross_entropy expects logits [N, C] and targets [N], got {:?} and {:?}",
                logits.dims(),
                targets.dims()
            ),
        ));
    }
    match targets.dtype {
        RsDtype::I32 | RsDtype::I64 => {}
        other => {
            return Err(err(
                c.op,
                format!("input 'targets' has dtype {other}, expected i32 or i64"),
            ));
        }
    }
    expect_out(c.out_t(0), c.op, RsDtype::F32, &[])?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let lv = unsafe { crate::tensor::f32_in(c.op, "logits", logits) }?;
    let ids = unsafe { crate::tensor::indices_i64(c.op, "targets", targets) }?;
    let mut ov = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let (n, ch) = (lv.shape()[0], lv.shape()[1]);
    // Stable log-sum-exp form (documented): loss = mean_n(lse_n - x[n, t_n]).
    // This stays finite for extreme logits (e.g. [1000, -1000, 0] → 1000)
    // where the naive log(softmax) expansion underflows to -inf; the two
    // agree within tolerance wherever the naive form is well-conditioned.
    let mut total = 0.0f64;
    for (i, &t) in ids.iter().enumerate() {
        let row = &lv.as_slice().expect("contiguous")[i * ch..(i + 1) * ch];
        let m = row.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let sum = row.iter().fold(0.0f32, |acc, &x| acc + (x - m).exp());
        let lse = m as f64 + (sum as f64).ln();
        if t < 0 || t >= ch as i64 {
            return Err(fail!(
                c.op,
                "target {t} out of range [0, {ch}) (negative targets are not wrapped)"
            ));
        }
        total += lse - row[t as usize] as f64;
    }
    if let Some(first) = ov.iter_mut().next() {
        *first = (total / n as f64) as f32;
    }
    Ok(())
}

/// Declared expansion of `cross_entropy`:
/// `softmax(logits)` -> `log` -> `reshape(targets, [N, 1])` ->
/// `gather(axis=-1)` -> `neg` -> `reduce(mean, all axes)`.
///
/// `gather` follows the torch convention (indices have the same rank as the
/// input), so the targets must be reshaped to `[N, 1]` before gathering the
/// per-row log-probability; the reshape node's `shape` attribute is
/// `[-1, 1]`, which is shape-independent (the `-1` is filled from numel).
///
/// Node attribute table (the current `ExpansionSpec` API cannot attach
/// per-node attrs — see the module doc):
/// node 0: softmax, attrs {axis: -1} (default);
/// node 1: elementwise_unary, attrs {kind: "log"};
/// node 2: reshape, attrs {shape: [-1, 1]};
/// node 3: gather, attrs {axis: -1} (default);
/// node 4: elementwise_unary, attrs {kind: "neg"};
/// node 5: reduce, attrs {kind: "mean"} (no axis → scalar).
///
/// The fused body uses the stable log-sum-exp form of the same loss; the
/// two agree within tolerance wherever the naive log(softmax) path is
/// well-conditioned (documented in the op doc).
pub(crate) fn cross_entropy_expansion() -> ExpansionSpec {
    let mut e = ExpansionSpec::new(2, 1);
    let p = e.temp();
    let lp = e.temp();
    let t2 = e.temp();
    let per = e.temp();
    let neg = e.temp();
    e.node("softmax", &[0], &[p])
        .node("elementwise_unary", &[p], &[lp])
        .node("reshape", &[1], &[t2])
        .node("gather", &[lp, t2], &[per])
        .node("elementwise_unary", &[per], &[neg])
        .node("reduce", &[neg], &[2])
}

// ── adamw ───────────────────────────────────────────────────────────────────

fn adamw_params(a: &RsAttrs, op: &'static str) -> OpResult<(f32, f32, f32, f32, f32, i64)> {
    let lr = attr_f64(a, "lr").unwrap_or(1e-3) as f32;
    let b1 = attr_f64(a, "beta1").unwrap_or(0.9) as f32;
    let b2 = attr_f64(a, "beta2").unwrap_or(0.999) as f32;
    let eps = attr_f64(a, "eps").unwrap_or(1e-8) as f32;
    let wd = attr_f64(a, "weight_decay").unwrap_or(0.0) as f32;
    let step = attr_i64(a, "step").unwrap_or(1);
    if step < 1 {
        return Err(err(op, format!("adamw 'step' must be >= 1, got {step}")));
    }
    if !(0.0..1.0).contains(&b1) || !(0.0..1.0).contains(&b2) {
        return Err(err(
            op,
            format!("adamw beta1/beta2 must be in [0, 1), got {b1} and {b2}"),
        ));
    }
    Ok((lr, b1, b2, eps, wd, step))
}

fn adamw_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((4, 4))?;
    c.expect_out_count(3)?;
    let p = c.in_t(0);
    for (i, who) in [(1usize, "grad"), (2, "exp_avg"), (3, "exp_avg_sq")] {
        let t = c.in_t(i);
        if t.dtype != RsDtype::F32 {
            return Err(err(
                c.op,
                format!("input '{who}' has dtype {}, expected f32", t.dtype),
            ));
        }
        if t.dims() != p.dims() {
            return Err(err(
                c.op,
                format!(
                    "adamw input '{who}' shape {:?} must match param shape {:?}",
                    t.dims(),
                    p.dims()
                ),
            ));
        }
    }
    adamw_params(a, c.op)?;
    for i in 0..3 {
        set_output_desc(c.out_t(i), RsDtype::F32, p.dims());
    }
    Ok(())
}

fn adamw_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((4, 4))?;
    c.expect_out_count(3)?;
    let p = c.in_t(0);
    for (i, who) in [(1usize, "grad"), (2, "exp_avg"), (3, "exp_avg_sq")] {
        let t = c.in_t(i);
        if t.dtype != RsDtype::F32 {
            return Err(err(
                c.op,
                format!("input '{who}' has dtype {}, expected f32", t.dtype),
            ));
        }
        if t.dims() != p.dims() {
            return Err(err(
                c.op,
                format!(
                    "adamw input '{who}' shape {:?} must match param shape {:?}",
                    t.dims(),
                    p.dims()
                ),
            ));
        }
    }
    let (lr, b1, b2, eps, wd, step) = adamw_params(a, c.op)?;
    // Bias corrections are integer powers: exact and deterministic.
    let c1 = 1.0 - b1.powi(step as i32);
    let c2 = 1.0 - b2.powi(step as i32);
    for i in 0..3 {
        expect_out(c.out_t(i), c.op, RsDtype::F32, p.dims())?;
    }
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let pv = unsafe { crate::tensor::f32_in(c.op, "param", p) }?;
    let gv = unsafe { crate::tensor::f32_in(c.op, "grad", c.in_t(1)) }?;
    let mv = unsafe { crate::tensor::f32_in(c.op, "exp_avg", c.in_t(2)) }?;
    let vv = unsafe { crate::tensor::f32_in(c.op, "exp_avg_sq", c.in_t(3)) }?;
    let mut po = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let mut mo = unsafe { crate::tensor::f32_out(c.op, c.out_t(1)) }?;
    let mut vo = unsafe { crate::tensor::f32_out(c.op, c.out_t(2)) }?;
    // Pointwise in ascending index order (decoupled weight decay, standard
    // bias correction). exp_avg' and exp_avg_sq' are written back so the
    // caller can persist them.
    let ps = pv.as_slice().expect("contiguous");
    let gs = gv.as_slice().expect("contiguous");
    let ms = mv.as_slice().expect("contiguous");
    let vs = vv.as_slice().expect("contiguous");
    let pos = po.as_slice_mut().expect("contiguous");
    let mos = mo.as_slice_mut().expect("contiguous");
    let vos = vo.as_slice_mut().expect("contiguous");
    for i in 0..ps.len() {
        let m = b1 * ms[i] + (1.0 - b1) * gs[i];
        let v = b2 * vs[i] + (1.0 - b2) * gs[i] * gs[i];
        let mh = m / c1;
        let vh = v / c2;
        mos[i] = m;
        vos[i] = v;
        pos[i] = ps[i] - lr * (mh / (vh.sqrt() + eps) + wd * ps[i]);
    }
    Ok(())
}

/// Declared expansion of `adamw` at the default hyperparameters
/// (lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0, step=1).
///
/// Node attributes required (not attachable via the current `ExpansionSpec`
/// API — see the module doc), using `elementwise_binary` (`kind` + scalar
/// `rhs` for the single-input nodes) and `elementwise_unary` (`kind`):
/// 1 mul g*(1-b1); 2 mul m*b1; 3 add → m'; 4 mul g*g; 5 mul g^2*(1-b2);
/// 6 mul v*b2; 7 add → v'; 8 mul m'/(1-b1^t); 9 mul v'/(1-b2^t);
/// 10 sqrt; 11 add eps; 12 div mhat/den; 13 mul lr; 14 mul p*(lr*wd);
/// 15 sub p - step13; 16 sub - step14 → p'.
pub(crate) fn adamw_expansion() -> ExpansionSpec {
    let mut e = ExpansionSpec::new(4, 3);
    let t7 = e.temp();
    let t8 = e.temp();
    let t9 = e.temp();
    let t10 = e.temp();
    let t11 = e.temp();
    let t12 = e.temp();
    let t13 = e.temp();
    let t14 = e.temp();
    let t15 = e.temp();
    let t16 = e.temp();
    let t17 = e.temp();
    let t18 = e.temp();
    let t19 = e.temp();
    e.node("elementwise_binary", &[1], &[t7])
        .node("elementwise_binary", &[2], &[t8])
        .node("elementwise_binary", &[t8, t7], &[5])
        .node("elementwise_binary", &[1, 1], &[t9])
        .node("elementwise_binary", &[t9], &[t10])
        .node("elementwise_binary", &[3], &[t11])
        .node("elementwise_binary", &[t11, t10], &[6])
        .node("elementwise_binary", &[5], &[t12])
        .node("elementwise_binary", &[6], &[t13])
        .node("elementwise_unary", &[t13], &[t14])
        .node("elementwise_binary", &[t14], &[t15])
        .node("elementwise_binary", &[t12, t15], &[t16])
        .node("elementwise_binary", &[t16], &[t17])
        .node("elementwise_binary", &[0], &[t18])
        .node("elementwise_binary", &[0, t17], &[t19])
        .node("elementwise_binary", &[t19, t18], &[4])
}

// ── topk_router ─────────────────────────────────────────────────────────────

fn topk_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(2)?;
    let logits = c.in_t(0);
    if logits.dtype != RsDtype::F32 {
        return Err(err(
            c.op,
            format!("input 'logits' has dtype {}, expected f32", logits.dtype),
        ));
    }
    if logits.rank != 2 {
        return Err(err(
            c.op,
            format!(
                "topk_router expects logits [N, E], got rank {}",
                logits.rank
            ),
        ));
    }
    let e = logits.shape[1];
    let k = attr_i64(a, "top_k").unwrap_or(2);
    if k < 1 || k > e {
        return Err(err(
            c.op,
            format!("topk_router 'top_k' must be in [1, E={e}], got {k}"),
        ));
    }
    let n = logits.shape[0];
    // Out 0: routing weights (raw softmax probabilities of the selected
    // experts); out 1: expert indices.
    set_output_desc(c.out_t(0), RsDtype::F32, &[n, k]);
    set_output_desc(c.out_t(1), RsDtype::I32, &[n, k]);
    Ok(())
}

fn topk_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((1, 1))?;
    c.expect_out_count(2)?;
    let logits = c.in_t(0);
    if logits.dtype != RsDtype::F32 {
        return Err(err(
            c.op,
            format!("input 'logits' has dtype {}, expected f32", logits.dtype),
        ));
    }
    if logits.rank != 2 {
        return Err(err(
            c.op,
            format!(
                "topk_router expects logits [N, E], got rank {}",
                logits.rank
            ),
        ));
    }
    let n = logits.shape[0] as usize;
    let e = logits.shape[1] as usize;
    let k = attr_i64(a, "top_k").unwrap_or(2) as usize;
    if k < 1 || k > e {
        return Err(err(
            c.op,
            format!("topk_router 'top_k' must be in [1, E={e}], got {k}"),
        ));
    }
    let norm = attr_bool(a, "norm_topk_prob").unwrap_or(false);
    expect_out(c.out_t(0), c.op, RsDtype::F32, &[logits.shape[0], k as i64])?;
    expect_out(c.out_t(1), c.op, RsDtype::I32, &[logits.shape[0], k as i64])?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let lv = unsafe { crate::tensor::f32_in(c.op, "logits", logits) }?;
    let mut wo = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let mut io = unsafe { crate::tensor::i32_out(c.op, c.out_t(1)) }?;
    // Per-row stable softmax, then a deterministic insertion selection: the
    // comparison is strict `>`, so on ties the lower expert index wins.
    let ls = lv.as_slice().expect("contiguous");
    let ws = wo.as_slice_mut().expect("contiguous");
    let is = io.as_slice_mut().expect("contiguous");
    for i in 0..n {
        let row = &ls[i * e..(i + 1) * e];
        let m = row.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let sum = row.iter().fold(0.0f32, |acc, &x| acc + (x - m).exp());
        let mut best: Vec<(f32, usize)> = Vec::with_capacity(k);
        for (ex, &x) in row.iter().enumerate() {
            let p = (x - m).exp() / sum;
            let mut pos = best.len();
            while pos > 0 && p > best[pos - 1].0 {
                pos -= 1;
            }
            if best.len() < k {
                best.insert(pos, (p, ex));
            } else if pos < k {
                best.insert(pos, (p, ex));
                best.truncate(k);
            }
        }
        for (j, &(p, ex)) in best.iter().enumerate() {
            ws[i * k + j] = p;
            is[i * k + j] = ex as i32;
        }
        // HF's Qwen3_5MoeTopKRouter renormalises the selected weights to sum
        // to 1 (unconditionally, `router_top_value /= router_top_value.sum(-1,
        // keepdim=True)`); the declaration asks for it via `norm_topk_prob`.
        // The sum runs in ascending k order (deterministic). A zero sum cannot
        // occur: the top-k of a softmax row is positive.
        if norm {
            let s = ws[i * k..(i + 1) * k].iter().sum::<f32>();
            for w in &mut ws[i * k..(i + 1) * k] {
                *w /= s;
            }
        }
    }
    Ok(())
}

/// Declared expansion of `topk_router`.
///
/// Only the gating-probability path is expressible in the fixed primitive
/// vocabulary: `softmax(logits) -> probs`. The top-k *selection* (weights =
/// top-k subset of the probabilities, indices = their positions, ties broken
/// by the lower index) has no primitive equivalent in spec §2.4 (there is no
/// top-k / argmax operator), so that part is documented in the op doc rather
/// than faked with a non-equivalent node.
pub(crate) fn topk_expansion() -> ExpansionSpec {
    let mut e = ExpansionSpec::new(1, 2);
    let p = e.temp();
    e.node("softmax", &[0], &[p])
}

infer_entry!(sdpa_infer, "sdpa", sdpa_infer_body);
exec_entry!(sdpa_exec, "sdpa", sdpa_exec_body);
infer_entry!(ce_infer, "cross_entropy", cross_entropy_infer_body);
exec_entry!(ce_exec, "cross_entropy", cross_entropy_exec_body);
infer_entry!(adamw_infer, "adamw", adamw_infer_body);
exec_entry!(adamw_exec, "adamw", adamw_exec_body);
infer_entry!(topk_infer, "topk_router", topk_infer_body);
exec_entry!(topk_exec, "topk_router", topk_exec_body);
