//! The GDN recurrence itself: `gated_delta_rule`.
//!
//! This is the reference implementation of the chunked gated delta rule,
//! transcribed from the HF source (`torch_chunk_gated_delta_rule` in
//! `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`) with the same
//! structural choices: fp32 state, the strictly-lower-triangular
//! "UT transform" condensed into one solve per chunk, and the sequential
//! scan over chunks. Every loop runs in ascending index order, so the result
//! is bitwise reproducible.
//!
//! Declared conventions (the plan data cannot express them, so they live in
//! this doc and are enforced as hard errors):
//!
//! * `k_head_dim == v_head_dim` (both 128 in Qwen3.6) — the head count is
//!   derived as `k_last / v_head_dim`; a mismatch is a hard error, never a
//!   guess.
//! * The query is scaled by `k_head_dim^-0.5` inside the recurrence (the HF
//!   "always normalize queries by the head dimension" step); the L2
//!   normalisation itself is the caller's (`l2norm` nodes), matching the
//!   decomposition in op-vocabulary §3.2.
//! * `num_k_heads` may be smaller than `num_v_heads`: q and k heads are
//!   `repeat_interleave`d by the ratio (HF does this before the kernel, the
//!   plan folds it in — GQA for the delta rule).
//! * The state is read AFTER the update (`o_t = q_t^T S_t` with S_t already
//!   containing token t), the gated delta rule's defining order.
//! * The CP>1 dependency is *declared*, not derived: the descriptor carries
//!   `collectives: all_gather {cp}` for the affine-map merge (op-vocabulary
//!   §4); the reference provider itself runs single-rank.

use rustrain_abi::ffi::{MAX_RANK, RsAttrs, RsDtype, RsMemReq, RsTensor};

use crate::attrs::{attr_i64, str_or};
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

/// The shape plan shared by infer, execute and the memory reporter.
/// Returns `(batches, s, vh, kh, d, dv, repeat, chunk)`.
#[allow(clippy::type_complexity)]
fn delta_rule_plan(
    q: &RsTensor,
    k: &RsTensor,
    v: &RsTensor,
    g: &RsTensor,
    attrs: &RsAttrs,
    op: &'static str,
) -> OpResult<(usize, usize, usize, usize, usize, usize, usize, usize)> {
    for (i, t) in [(0usize, q), (1, k), (2, v), (3, g)] {
        if t.dtype != RsDtype::F32 {
            return Err(err(
                op,
                format!("input {i} has dtype {}, expected f32", t.dtype),
            ));
        }
        if t.rank < 2 {
            return Err(err(
                op,
                format!(
                    "input {i} needs rank >= 2 ([..., S, C]), got rank {}",
                    t.rank
                ),
            ));
        }
    }
    // Batch dims (all but the last two) must be identical across all inputs.
    let batches = q.dims()[..q.rank as usize - 2]
        .iter()
        .map(|&d| d as usize)
        .product::<usize>();
    for t in [q, k, v, g] {
        if t.rank != q.rank || t.dims()[..t.rank as usize - 2] != q.dims()[..q.rank as usize - 2] {
            return Err(fail!(
                op,
                "gated_delta_rule expects identical batch dims on all inputs, got q {:?} and {:?}",
                q.dims(),
                t.dims()
            ));
        }
        if t.shape[t.rank as usize - 2] != q.shape[q.rank as usize - 2] {
            return Err(fail!(
                op,
                "gated_delta_rule expects the same sequence length S on all inputs, got q {:?} \
                 and {:?}",
                q.dims(),
                t.dims()
            ));
        }
    }
    let vh = g.shape[g.rank as usize - 1] as usize;
    if vh == 0 {
        return Err(err(op, "gated_delta_rule expects at least one value head"));
    }
    let dv = v.shape[v.rank as usize - 1] as usize / vh;
    if dv == 0 || v.shape[v.rank as usize - 1] as usize % vh != 0 {
        return Err(fail!(
            op,
            "gated_delta_rule value last dim ({}) must be a multiple of the value-head count \
             {vh} (g's last dim)",
            v.shape[v.rank as usize - 1]
        ));
    }
    // Declared convention: k_head_dim == v_head_dim (see the module doc).
    let d = dv;
    let kd = k.shape[k.rank as usize - 1] as usize;
    let qd = q.shape[q.rank as usize - 1] as usize;
    if kd % d != 0 || qd % d != 0 {
        return Err(fail!(
            op,
            "gated_delta_rule q/k last dims ({qd}, {kd}) must be multiples of the head dim \
             {d} (k_head_dim == v_head_dim is the declared convention)"
        ));
    }
    let (kh, qh) = (kd / d, qd / d);
    if qh != kh {
        return Err(fail!(
            op,
            "gated_delta_rule q and k must have the same head count, got {qh} vs {kh}"
        ));
    }
    if vh % kh != 0 {
        return Err(fail!(
            op,
            "gated_delta_rule value heads ({vh}) must be a multiple of the q/k heads ({kh}) \
             so the q/k heads can be repeat_interleaved (GQA)"
        ));
    }
    let repeat = vh / kh;
    let chunk = attr_i64(attrs, "chunk_size").unwrap_or(64);
    if chunk < 1 {
        return Err(fail!(op, "attribute 'chunk_size' ({chunk}) must be >= 1"));
    }
    str_or(attrs, "state_dtype", "f32", &["f32"], op)?;
    let s = q.shape[q.rank as usize - 2] as usize;
    Ok((batches, s, vh, kh, d, dv, repeat, chunk as usize))
}

fn gated_delta_rule_infer_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((5, 5))?;
    c.expect_out_count(1)?;
    let q = c.in_t(0);
    let (_b, _s, vh, _kh, _d, dv, _rep, _chunk) =
        delta_rule_plan(q, c.in_t(1), c.in_t(2), c.in_t(3), a, c.op)?;
    // beta (input 4) shares g's [.., S, vh] shape; checked in execute too.
    let rank = q.rank as usize;
    let mut shape = SmallShape {
        len: rank,
        dims: [0; MAX_RANK],
    };
    shape.dims[..rank].copy_from_slice(q.dims());
    shape.dims[rank - 1] = (vh * dv) as i64;
    let o = c.out_t(0);
    set_output_desc(o, RsDtype::F32, shape.as_slice());
    Ok(())
}

/// Total f32 scratch the chunked body allocates, from the same formula the
/// executor carves its single buffer with. The memory reporter and the
/// executor share this, so the reported workspace is exact by construction.
fn scratch_counts(vh: usize, s2: usize, nc: usize, c: usize, d: usize, dv: usize) -> usize {
    // q' + k' + k_beta(=decayed in place): 3 * vh*s2*d
    // v' + v_beta + out:                   3 * vh*s2*dv
    // beta' + g' + cum:                    3 * vh*s2
    // pairwise + ut + intra:               3 * vh*nc*c*c
    // k_cumdecay (d cols) + new_values (dv cols):
    //                                       vh*nc*c*(d + dv)
    // S_state:                              vh*d*dv
    // per-chunk working rows v_new + new_ss (reused per chunk/head):
    //                                       c*dv + d*dv
    vh * (s2 * (3 * d + 3 * dv + 3) + nc * c * (3 * c + d + dv) + d * dv) + c * dv + d * dv
}

/// The workspace the chunked body needs, or `None` when the io descriptors do
/// not yet describe a call it can serve (the reporter then reports zero, like
/// the other ops' null-io behaviour).
pub(crate) fn delta_rule_workspace(
    q: &RsTensor,
    k: &RsTensor,
    v: &RsTensor,
    g: &RsTensor,
    chunk: usize,
) -> Option<u64> {
    let empty = RsAttrs {
        items: std::ptr::null(),
        len: 0,
        _pad: 0,
    };
    let plan = delta_rule_plan(q, k, v, g, &empty, "gated_delta_rule").ok()?;
    let (batches, s, vh, _kh, d, dv, _rep, _chunk) = plan;
    let c = chunk.max(1);
    let s2 = s.div_ceil(c) * c;
    let nc = s2 / c;
    Some(4 * batches as u64 * scratch_counts(vh, s2, nc, c, d, dv) as u64)
}

/// `gated_delta_rule`'s memory reporter: the chunked body allocates one flat
/// f32 scratch buffer whose size this reports exactly. Outputs stay
/// caller-provided (contract C-3).
///
/// # Safety
/// `io` must be null or point to `n_io` live descriptors; `out` must be null
/// or a writable `RsMemReq`.
pub(crate) unsafe extern "C" fn gated_delta_rule_memory(
    io: *const *const RsTensor,
    n_io: u32,
    attrs: *const RsAttrs,
    out: *mut RsMemReq,
) -> i32 {
    if out.is_null() {
        let e = err("gated_delta_rule", "null RsMemReq output pointer");
        crate::error::set_last_error(&e);
        return e.status();
    }
    // SAFETY: checked non-null above.
    unsafe { *out = RsMemReq::default() };
    if io.is_null() || n_io < 4 || attrs.is_null() {
        return 0;
    }
    // SAFETY: the ABI contract says `io` points to live descriptors.
    let q = unsafe { &**io };
    let k = unsafe { &**io.add(1) };
    let v = unsafe { &**io.add(2) };
    let g = unsafe { &**io.add(3) };
    // SAFETY: caller-owned attribute list, valid for the call.
    let chunk = unsafe { crate::attrs::attr_i64(&*attrs, "chunk_size") }
        .unwrap_or(64)
        .max(1) as usize;
    if let Some(bytes) = delta_rule_workspace(q, k, v, g, chunk) {
        // SAFETY: `out` was checked non-null above.
        unsafe { (*out).workspace_bytes = bytes };
    }
    0
}

// ── the chunked body ────────────────────────────────────────────────────────

fn gated_delta_rule_exec_body(c: &mut Call, a: &RsAttrs) -> OpResult<()> {
    c.expect_arity((5, 5))?;
    c.expect_out_count(1)?;
    let q = c.in_t(0);
    let (batches, s, vh, kh, d, dv, repeat, chunk) =
        delta_rule_plan(q, c.in_t(1), c.in_t(2), c.in_t(3), a, c.op)?;
    // beta (input 4): same validation as g, then the dtype/shape check.
    let beta = c.in_t(4);
    if beta.dtype != RsDtype::F32 || beta.rank != c.in_t(3).rank || beta.dims() != c.in_t(3).dims()
    {
        return Err(fail!(
            c.op,
            "gated_delta_rule beta must match g's shape {:?}, got {:?} (dtype {})",
            c.in_t(3).dims(),
            beta.dims(),
            beta.dtype
        ));
    }
    let rank = q.rank as usize;
    let mut shape = SmallShape {
        len: rank,
        dims: [0; MAX_RANK],
    };
    shape.dims[..rank].copy_from_slice(q.dims());
    shape.dims[rank - 1] = (vh * dv) as i64;
    expect_out(c.out_t(0), c.op, RsDtype::F32, shape.as_slice())?;
    // SAFETY: descriptor liveness is the ABI caller's contract.
    let qv = unsafe { crate::tensor::f32_in(c.op, "q", q) }?;
    let kv = unsafe { crate::tensor::f32_in(c.op, "k", c.in_t(1)) }?;
    let vv = unsafe { crate::tensor::f32_in(c.op, "v", c.in_t(2)) }?;
    let gv = unsafe { crate::tensor::f32_in(c.op, "g", c.in_t(3)) }?;
    let bv = unsafe { crate::tensor::f32_in(c.op, "beta", beta) }?;
    let mut ov = unsafe { crate::tensor::f32_out(c.op, c.out_t(0)) }?;
    let qs = qv.as_slice().expect("contiguous");
    let ks = kv.as_slice().expect("contiguous");
    let vs = vv.as_slice().expect("contiguous");
    let gs = gv.as_slice().expect("contiguous");
    let bs = bv.as_slice().expect("contiguous");
    let os = ov.as_slice_mut().expect("contiguous");

    let c = chunk;
    let s2 = s.div_ceil(c) * c;
    let nc = s2 / c;
    let total = scratch_counts(vh, s2, nc, c, d, dv);
    let mut scratch = vec![0.0f32; total];

    // Slice sizes: the f32 counts of the scratch_counts formula, in the same
    // order the comment there lists them.
    let qn = vh * s2 * d;
    let kn = vh * s2 * d;
    let kbn = vh * s2 * d;
    let vn = vh * s2 * dv;
    let vbn = vh * s2 * dv;
    let on = vh * s2 * dv;
    let betan = vh * s2;
    let gn = vh * s2;
    let cumn = vh * s2;
    let pwn = vh * nc * c * c;
    let utn = vh * nc * c * c;
    let intrn = vh * nc * c * c;
    let kcn = vh * nc * c * d;
    let nvn = vh * nc * c * dv;
    let ssn = vh * d * dv;

    // The HF query scaling: q *= k_head_dim^-0.5 (always applied, the l2norm
    // itself is upstream).
    let scale = (d as f32).powf(-0.5);
    for b in 0..batches {
        let qb = &qs[b * s * kh * d..(b + 1) * s * kh * d];
        let kb = &ks[b * s * kh * d..(b + 1) * s * kh * d];
        let vb = &vs[b * s * vh * dv..(b + 1) * s * vh * dv];
        let gb = &gs[b * s * vh..(b + 1) * s * vh];
        let bb = &bs[b * s * vh..(b + 1) * s * vh];
        let ob = &mut os[b * s * vh * dv..(b + 1) * s * vh * dv];
        let (q_p, rest) = scratch.split_at_mut(qn);
        let (k_p, rest) = rest.split_at_mut(kn);
        let (kb_p, rest) = rest.split_at_mut(kbn);
        let (v_p, rest) = rest.split_at_mut(vn);
        let (vb_p, rest) = rest.split_at_mut(vbn);
        let (o_p, rest) = rest.split_at_mut(on);
        let (beta_p, rest) = rest.split_at_mut(betan);
        let (g_p, rest) = rest.split_at_mut(gn);
        let (cum_p, rest) = rest.split_at_mut(cumn);
        let (pw_p, rest) = rest.split_at_mut(pwn);
        let (ut_p, rest) = rest.split_at_mut(utn);
        let (intra_p, rest) = rest.split_at_mut(intrn);
        let (kc_p, rest) = rest.split_at_mut(kcn);
        let (nv_p, rest) = rest.split_at_mut(nvn);
        let (ss_p, rest) = rest.split_at_mut(ssn);
        let (v_new_p, rest) = rest.split_at_mut(c * dv);
        let (new_ss_p, tail) = rest.split_at_mut(d * dv);
        debug_assert!(tail.is_empty());

        for h in 0..vh {
            let src_h = h / repeat;
            let q_batch = &mut q_p[h * s2 * d..(h + 1) * s2 * d];
            let k_batch = &mut k_p[h * s2 * d..(h + 1) * s2 * d];
            let v_batch = &mut v_p[h * s2 * dv..(h + 1) * s2 * dv];
            let beta_batch = &mut beta_p[h * s2..(h + 1) * s2];
            let g_batch = &mut g_p[h * s2..(h + 1) * s2];
            for t in 0..s {
                for i in 0..d {
                    q_batch[t * d + i] = qb[t * kh * d + src_h * d + i] * scale;
                    k_batch[t * d + i] = kb[t * kh * d + src_h * d + i];
                }
                for j in 0..dv {
                    v_batch[t * dv + j] = vb[t * vh * dv + h * dv + j];
                }
                beta_batch[t] = bb[t * vh + h];
                g_batch[t] = gb[t * vh + h];
            }
            for t in s..s2 {
                for i in 0..d {
                    q_batch[t * d + i] = 0.0;
                    k_batch[t * d + i] = 0.0;
                }
                for j in 0..dv {
                    v_batch[t * dv + j] = 0.0;
                }
                beta_batch[t] = 0.0;
                g_batch[t] = 0.0;
            }
            // v_beta = v * beta, k_beta = k * beta (the gated update's
            // "learning rate").
            let vb_batch = &mut vb_p[h * s2 * dv..(h + 1) * s2 * dv];
            let k_beta = &mut kb_p[h * s2 * d..(h + 1) * s2 * d];
            for t in 0..s2 {
                for j in 0..dv {
                    vb_batch[t * dv + j] = v_batch[t * dv + j] * beta_batch[t];
                }
                for i in 0..d {
                    k_beta[t * d + i] = k_batch[t * d + i] * beta_batch[t];
                }
            }
            // Cumulative decay in log space, PER CHUNK (the HF source
            // reshapes to [.., nc, c] before cumsum): cum[t] = sum of g over
            // the chunk's positions up to t. Restarting each chunk is what
            // makes the chunked scan equal the per-token recurrence.
            let cum = &mut cum_p[h * s2..(h + 1) * s2];
            for ch in 0..nc {
                let mut acc = 0.0f32;
                for i in 0..c {
                    acc += g_batch[ch * c + i];
                    cum[ch * c + i] = acc;
                }
            }
            // pairwise[i, j] = exp(cum_i - cum_j) for i >= j, else 0.
            let pw = &mut pw_p[h * nc * c * c..(h + 1) * nc * c * c];
            for ch in 0..nc {
                for i in 0..c {
                    for j in 0..c {
                        let v = if i >= j {
                            (cum[ch * c + i] - cum[ch * c + j]).exp()
                        } else {
                            0.0
                        };
                        pw[(ch * c + i) * c + j] = v;
                    }
                }
            }
            // ut = (k_beta @ k^T) * pairwise; intra = (q @ k^T) * pairwise.
            let ut = &mut ut_p[h * nc * c * c..(h + 1) * nc * c * c];
            let intra = &mut intra_p[h * nc * c * c..(h + 1) * nc * c * c];
            for ch in 0..nc {
                let kc = &k_batch[ch * c * d..(ch + 1) * c * d];
                let qc = &q_batch[ch * c * d..(ch + 1) * c * d];
                for i in 0..c {
                    for j in 0..c {
                        let mut accu = 0.0f32;
                        let mut accq = 0.0f32;
                        for dd in 0..d {
                            accu += k_beta[(ch * c + i) * d + dd] * kc[j * d + dd];
                            accq += qc[i * d + dd] * kc[j * d + dd];
                        }
                        let p = pw[(ch * c + i) * c + j];
                        ut[(ch * c + i) * c + j] = accu * p;
                        intra[(ch * c + i) * c + j] = accq * p;
                    }
                }
            }
            // decayed_k_beta = k_beta * cum.exp() — k_beta itself is dead
            // after `ut`, so it is overwritten in place.
            for t in 0..s2 {
                let e = cum[t].exp();
                for i in 0..d {
                    k_beta[t * d + i] *= e;
                }
            }
            // Forward substitution of (I + strictly_lower(ut)) x = b — the
            // unitriangular triangular solve the HF source calls. The
            // diagonal is 1 by convention and never read.
            let nv = &mut nv_p[h * nc * c * dv..(h + 1) * nc * c * dv];
            let kc2 = &mut kc_p[h * nc * c * d..(h + 1) * nc * c * d];
            for ch in 0..nc {
                for i in 0..c {
                    for j in 0..dv {
                        let mut acc = vb_batch[(ch * c + i) * dv + j];
                        for jj in 0..i {
                            acc -= ut[(ch * c + i) * c + jj] * nv[(ch * c + jj) * dv + j];
                        }
                        nv[(ch * c + i) * dv + j] = acc;
                    }
                    for dd in 0..d {
                        let mut acc = k_beta[(ch * c + i) * d + dd];
                        for jj in 0..i {
                            acc -= ut[(ch * c + i) * c + jj] * kc2[(ch * c + jj) * d + dd];
                        }
                        kc2[(ch * c + i) * d + dd] = acc;
                    }
                }
            }
            // The scan's per-token adjustments: query gains the within-chunk
            // decay; key the complement (HF: apply decay once per chunk).
            for ch in 0..nc {
                let last = cum[(ch + 1) * c - 1];
                for t in ch * c..(ch + 1) * c {
                    let eq = cum[t].exp();
                    for i in 0..d {
                        q_batch[t * d + i] *= eq;
                        k_batch[t * d + i] *= (last - cum[t]).exp();
                    }
                }
            }
        }

        // The sequential scan over chunks, on the shared fp32 state (the
        // ss_p slice, zeroed per batch — the scratch buffer is reused).
        for v in ss_p.iter_mut() {
            *v = 0.0;
        }
        for ch in 0..nc {
            for h in 0..vh {
                let q_ch = &q_p[h * s2 * d + ch * c * d..h * s2 * d + (ch + 1) * c * d];
                let k_ch = &k_p[h * s2 * d + ch * c * d..h * s2 * d + (ch + 1) * c * d];
                let nv_ch =
                    &nv_p[h * nc * c * dv + ch * c * dv..h * nc * c * dv + (ch + 1) * c * dv];
                let kc_ch = &kc_p[h * nc * c * d + ch * c * d..h * nc * c * d + (ch + 1) * c * d];
                let intra_ch =
                    &intra_p[h * nc * c * c + ch * c * c..h * nc * c * c + (ch + 1) * c * c];
                let ss = &mut ss_p[h * d * dv..(h + 1) * d * dv];
                let o_ch = &mut o_p[h * s2 * dv + ch * c * dv..h * s2 * dv + (ch + 1) * c * dv];
                // v_new = new_values - k_cumdecay @ S
                for i in 0..c {
                    for j in 0..dv {
                        let mut acc = nv_ch[i * dv + j];
                        for dd in 0..d {
                            acc -= kc_ch[i * d + dd] * ss[dd * dv + j];
                        }
                        v_new_p[i * dv + j] = acc;
                    }
                }
                // inter = q_ch @ S; out = inter + intra @ v_new
                for i in 0..c {
                    for j in 0..dv {
                        let mut acc = 0.0f32;
                        for dd in 0..d {
                            acc += q_ch[i * d + dd] * ss[dd * dv + j];
                        }
                        for jj in 0..c {
                            acc += intra_ch[i * c + jj] * v_new_p[jj * dv + j];
                        }
                        o_ch[i * dv + j] = acc;
                    }
                }
                // S = S * chunk_decay + k^T @ v_new
                let cum = &cum_p[h * s2..(h + 1) * s2];
                let chunk_decay = cum[(ch + 1) * c - 1].exp();
                for i in 0..d {
                    for j in 0..dv {
                        let mut acc = 0.0f32;
                        for jj in 0..c {
                            acc += k_ch[jj * d + i] * v_new_p[jj * dv + j];
                        }
                        new_ss_p[i * dv + j] = ss[i * dv + j] * chunk_decay + acc;
                    }
                }
                ss.copy_from_slice(new_ss_p);
            }
        }

        // Crop the padding and write the output rows (vh*Dv per position).
        for t in 0..s {
            for h in 0..vh {
                for j in 0..dv {
                    ob[t * vh * dv + h * dv + j] = o_p[h * s2 * dv + t * dv + j];
                }
            }
        }
    }
    Ok(())
}

infer_entry!(
    gated_delta_rule_infer,
    "gated_delta_rule",
    gated_delta_rule_infer_body
);
exec_entry!(
    gated_delta_rule_exec,
    "gated_delta_rule",
    gated_delta_rule_exec_body
);
