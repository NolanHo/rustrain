// The two sequence operators of the GDN (linear attention) layers: the depthwise
// causal convolution and the delta-rule recurrence.
//
// Neither is a kernel written here:
//
//   * `causal_conv1d` maps onto cuDNN's 1-D convolution through `at::conv1d`
//     with a depthwise weight, then crops the right-hand tail the padding
//     produced. HF reaches the same maths through Dao-AILab's fused kernel; the
//     reference provider reaches it with a scalar sweep. All three compute
//     `out[t, c] = sum_k w[c, 0, k] * x[t + k - pad, c]`.
//
//   * `gated_delta_rule` runs the recurrence the ABI's documentation defines for
//     it, in the order that documentation now spells out (`S_t = S_{t-1}
//     exp(g_t)`; `delta_t = beta_t (v_t - k_t^T S_t)`; `S_t += k_t delta_t^T`;
//     `o_t = q_t^T S_t` read after the update — the decay lands before the read).
//     Two schedules run it: `chunk_size == 1` is the token-by-token scan, and
//     `chunk_size >= 2` is the chunked scan the attribute is named for. Both are
//     the same function, so the attribute chooses the schedule and never the
//     value.
#include "common.h"

namespace rsaten {
namespace {

// ── causal_conv1d ───────────────────────────────────────────────────────────

int32_t conv_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                   uint32_t n_out, const rs_attrs* attrs) {
    return guard("causal_conv1d", [&] {
        if (n_in != 2 || n_out != 1) {
            return fail("causal_conv1d expects two inputs and one output");
        }
        int rc = check_float(in[0], "causal_conv1d", "x");
        if (rc != 0) {
            return rc;
        }
        rc = check_float(in[1], "causal_conv1d", "w");
        if (rc != 0) {
            return rc;
        }
        if (in[0]->rank < 2) {
            return fail("causal_conv1d expects x of rank >= 2 ([..., L, C])");
        }
        int64_t channels = in[0]->shape[in[0]->rank - 1];
        if (in[1]->rank != 3 || in[1]->shape[1] != 1 || in[1]->shape[0] != channels) {
            return fail("causal_conv1d weight must be [C, 1, K] (depthwise)");
        }
        int64_t kernel = in[1]->shape[2];
        if (i64_or(attrs, "kernel", kernel) != kernel) {
            return fail("causal_conv1d: the declared 'kernel' must equal the weight's tap count");
        }
        std::string groups;
        static const char* const GROUP_KINDS[] = {"channels"};
        if (!require_kind(attrs, "groups", GROUP_KINDS, 1, "causal_conv1d", &groups)) {
            return 1;
        }
        std::string activation;
        static const char* const ACTIVATIONS[] = {"", "silu", "none"};
        if (!require_kind(attrs, "activation", ACTIVATIONS, 3, "causal_conv1d", &activation)) {
            return 1;
        }
        set_shape(out[0], float_dtype_of(in[0]), dims_of(in[0]));
        return 0;
    });
}

int32_t conv_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                     uint32_t n_out, const rs_attrs* attrs) {
    return guard("causal_conv1d", [&] {
        if (n_in != 2 || n_out != 1) {
            return fail("causal_conv1d expects two inputs and one output");
        }
        at::Tensor x = view(in[0]);
        at::Tensor w = view(in[1]);
        int64_t L = x.size(-2);
        int64_t C = x.size(-1);
        int64_t K = w.size(2);
        int64_t pad = i64_or(attrs, "pad", K - 1);
        int64_t batch = 1;
        for (int64_t i = 0; i + 2 < x.dim(); ++i) {
            batch *= x.size(i);
        }
        // [.., L, C] -> [B, C, L]: cuDNN's convolution layout.
        at::Tensor xc = x.reshape({batch, L, C}).permute({0, 2, 1}).contiguous();
        at::Tensor yc = at::conv1d(xc, w, /*bias=*/std::nullopt, /*stride=*/{1}, /*padding=*/{pad},
                                   /*dilation=*/{1}, /*groups=*/C);
        // The padding put `pad` zeros on the left (which is what causality
        // needs) and `pad` behind the last element, which the recurrence never
        // reads: crop back to L.
        yc = yc.narrow(/*dim=*/2, /*start=*/0, L);
        std::string activation;
        static const char* const ACTIVATIONS[] = {"", "silu", "none"};
        if (!require_kind(attrs, "activation", ACTIVATIONS, 3, "causal_conv1d", &activation)) {
            return 1;
        }
        if (activation == "silu") {
            yc = yc / (1.0 + (-yc).exp());
        }
        at::Tensor y = yc.permute({0, 2, 1}).reshape(dims_of(out[0]));
        return write_out(out[0], y, "causal_conv1d");
    });
}

// ── gated_delta_rule ────────────────────────────────────────────────────────

struct DeltaPlan {
    int64_t batches = 1;
    int64_t s = 0;
    int64_t vh = 0;
    int64_t kh = 0;
    int64_t d = 0;
    int64_t dv = 0;
    int64_t repeat = 1;
    /// `chunk_size` (i64, default 64): the scan's chunk length. The declaration
    /// says it "sizes the chunked scan (padding to a multiple is internal, the
    /// output keeps S)" and that it does not change the value — so the body reads
    /// it to pick the scan's granularity, and both granularities compute the same
    /// recurrence (the reference provider reads it too).
    int64_t chunk = 64;
};

bool delta_plan(const rs_tensor* const* in, const rs_attrs* attrs, const char* op,
                DeltaPlan* plan) {
    const rs_tensor* q = in[0];
    const rs_tensor* k = in[1];
    const rs_tensor* v = in[2];
    const rs_tensor* g = in[3];
    const rs_tensor* beta = in[4];
    for (int i = 0; i < 5; ++i) {
        int rc = check_float(in[i], op, "q/k/v/g/beta");
        if (rc != 0) {
            return false;
        }
    }
    if (q->rank < 2 || q->rank != k->rank || q->rank != v->rank || q->rank != g->rank ||
        g->rank != beta->rank) {
        fail(std::string(op) + ": q, k, v, g and beta must share a rank >= 2");
        return false;
    }
    for (uint32_t d = 0; d + 2 < q->rank; ++d) {
        if (k->shape[d] != q->shape[d] || v->shape[d] != q->shape[d] ||
            g->shape[d] != q->shape[d] || beta->shape[d] != q->shape[d]) {
            fail(std::string(op) + ": q, k, v, g and beta must share their batch dims");
            return false;
        }
    }
    plan->batches = 1;
    for (uint32_t d = 0; d + 2 < q->rank; ++d) {
        plan->batches *= q->shape[d];
    }
    plan->s = q->shape[q->rank - 2];
    plan->vh = g->shape[g->rank - 1];
    if (plan->vh <= 0) {
        fail(std::string(op) + ": at least one value head is required");
        return false;
    }
    if (v->shape[v->rank - 1] % plan->vh != 0) {
        fail(std::string(op) + ": the value last dim must be a multiple of the value heads");
        return false;
    }
    plan->dv = v->shape[v->rank - 1] / plan->vh;
    plan->d = plan->dv;  // k_head_dim == v_head_dim is the declared convention
    if (k->shape[k->rank - 1] % plan->d != 0 || q->shape[q->rank - 1] % plan->d != 0) {
        fail(std::string(op) + ": q/k last dims must be multiples of the head dim");
        return false;
    }
    plan->kh = k->shape[k->rank - 1] / plan->d;
    if (q->shape[q->rank - 1] / plan->d != plan->kh) {
        fail(std::string(op) + ": q and k must have the same head count");
        return false;
    }
    if (plan->vh % plan->kh != 0) {
        fail(std::string(op) + ": value heads must be a multiple of the q/k heads (GQA)");
        return false;
    }
    plan->repeat = plan->vh / plan->kh;
    if (g->shape[g->rank - 2] != plan->s || beta->shape[beta->rank - 2] != plan->s ||
        beta->shape[beta->rank - 1] != plan->vh) {
        fail(std::string(op) + ": g and beta must be [.., S, vh]");
        return false;
    }
    std::string state_dtype;
    static const char* const STATE_DTYPES[] = {"f32"};
    if (!require_kind(attrs, "state_dtype", STATE_DTYPES, 1, op, &state_dtype)) {
        return false;
    }
    // The declaration's default, and the same lower bound the reference provider
    // enforces (`crates/rustrain-kernels/src/op/recurrent.rs`): a chunk of zero
    // tokens would not be a scan.
    plan->chunk = i64_or(attrs, "chunk_size", 64);
    if (plan->chunk < 1) {
        std::string msg = std::string(op) + ": attribute 'chunk_size' (" +
                          std::to_string(plan->chunk) + ") must be >= 1";
        fail(msg);
        return false;
    }
    return true;
}

int32_t delta_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                    uint32_t n_out, const rs_attrs* attrs) {
    return guard("gated_delta_rule", [&] {
        if (n_in != 5 || n_out != 1) {
            return fail("gated_delta_rule expects five inputs and one output");
        }
        DeltaPlan plan;
        if (!delta_plan(in, attrs, "gated_delta_rule", &plan)) {
            return 1;
        }
        std::vector<int64_t> shape = dims_of(in[0]);
        shape[shape.size() - 1] = plan.vh * plan.dv;
        set_shape(out[0], float_dtype_of(in[0]), shape);
        return 0;
    });
}

/// One operand reshaped to [B, S, heads, D], with the q/k heads repeated
/// interleaved when the value heads are more numerous (the declared GQA rule).
at::Tensor delta_heads(const at::Tensor& t, int64_t batch, int64_t S, int64_t heads, int64_t d,
                       int64_t repeat) {
    at::Tensor h = t.reshape({batch, S, heads, d});
    if (repeat > 1) {
        h = h.repeat_interleave(repeat, /*dim=*/2);
    }
    return h;
}

/// The declared recurrence, one token at a time: `S_t = d_t S_{t-1} + k_t δ_tᵀ`
/// with `δ_t = β_t (v_t - k_tᵀ S_{t-1})` and `o_t = q_tᵀ S_t` read after the
/// update — the operator's own words, executed literally.
///
/// This is the `chunk_size = 1` schedule: the scan with one token per chunk.
/// It costs `S` iterations of a handful of small kernels, which at S = 512 is
/// ~40 ms per layer and is why the chunked schedule exists (`delta_chunked`).
int32_t delta_recurrence(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                         const at::Tensor& g, const at::Tensor& beta, const DeltaPlan& plan,
                         at::Tensor& result, rs_tensor* const* out, at::ScalarType io_dtype) {
    const int64_t B = plan.batches;
    const int64_t S = plan.s;
    const int64_t H = plan.vh;
    const int64_t D = plan.d;
    const int64_t Dv = plan.dv;
    at::Tensor state = at::zeros({B, H, D, Dv}, q.options());
    for (int64_t t = 0; t < S; ++t) {
        at::Tensor decay = g.select(1, t).exp().reshape({B, H, 1, 1});
        state = state * decay;
        at::Tensor k_t = k.select(1, t).reshape({B, H, 1, D});
        at::Tensor v_t = v.select(1, t).reshape({B, H, 1, Dv});
        at::Tensor beta_t = beta.select(1, t).reshape({B, H, 1, 1});
        at::Tensor delta = (v_t - at::matmul(k_t, state)) * beta_t;
        state = state + at::matmul(k_t.transpose(-1, -2), delta);
        at::Tensor q_t = q.select(1, t).reshape({B, H, 1, D});
        result.select(1, t).copy_(at::matmul(q_t, state).reshape({B, H, Dv}));
    }
    return write_out(out[0], result.reshape(dims_of(out[0])).to(io_dtype), "gated_delta_rule");
}

/// The same scan, chunked: the algebra the operator declares, rearranged so the
/// sequential dependency runs over `ceil(S / chunk_size)` chunks instead of over
/// tokens.
///
/// Write `c_t = Σ_{j≤t} g_j` (the chunk's cumulative log-decay) and `δ_t` the
/// token's effective write. Two facts follow, both exact:
///
///   S_t = e^{c_t} S_prev + Σ_{s≤t} e^{c_t - c_s} k_s δ_sᵀ                (state)
///   o_t = e^{c_t} (q_tᵀ S_prev) + Σ_{s≤t} e^{c_t - c_s}(q_t·k_s) δ_sᵀ    (output)
///
/// and substituting the first into `δ_t = β_t (v_t - e^{g_t} k_tᵀ S_{t-1})` —
/// the decay is applied *before* the read, so the coefficient is `c_t`, not
/// `c_{t-1}`; getting that wrong costs 1e-2 relative, which is how it was found —
/// turns the chunk's deltas into one unit-lower-triangular system:
///
///   (I + tril(diag(β) · [e^{c_t - c_s}(k_t·k_s)], -1)) δ = diag(β)(v - e^{c}(k S_prev))
///
/// So a chunk is one triangular solve plus a handful of batched matmuls, and the
/// state crosses chunks once. Every decay is written as the **ratio**
/// `e^{c_t - c_s}` rather than as `e^{c_t} · e^{-c_s}`. The operands are the same
/// value, but the second form needs both factors to be representable at once: one
/// overflows to infinity past `ln(FLT_MAX) ≈ 88.7` while the other is by then
/// subnormal and soon zero, so the products that bring them back together
/// evaluate to `inf - inf` or `0 * inf` — NaN, which is what this body did until
/// this model's own decay magnitudes found it. Only the lower triangle is used
/// here, so the exponent never exceeds zero and nothing overflows. The reference
/// provider builds the same pairwise matrix for the same reason.
///
/// The algebra is exact; what differs is the order of floating-point summation,
/// measured at ~5e-7 relative against the token-by-token scan on this model's
/// shapes (the conformance case holds it to the declared tolerance).
int32_t delta_chunked(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                      const at::Tensor& g, const at::Tensor& beta, const DeltaPlan& plan,
                      at::Tensor& result, rs_tensor* const* out, at::ScalarType io_dtype) {
    const int64_t B = plan.batches;
    const int64_t S = plan.s;
    const int64_t H = plan.vh;
    const int64_t D = plan.d;
    const int64_t Dv = plan.dv;
    const int64_t C = plan.chunk;
    const at::TensorOptions fopts = q.options();
    at::Tensor state = at::zeros({B, H, D, Dv}, fopts);

    for (int64_t c0 = 0; c0 < S; c0 += C) {
        // The last chunk may be short: the algebra above never needed a chunk of
        // exactly `chunk_size`, and a short one avoids padding the operands (the
        // reference provider pads instead; both are exact, the values match).
        const int64_t n = std::min(C, S - c0);
        // [B, n, H, ·] → [B, H, n, ·]: one batch dim for the chunk's matmuls.
        at::Tensor qc = q.narrow(1, c0, n).transpose(1, 2);
        at::Tensor kc = k.narrow(1, c0, n).transpose(1, 2);
        at::Tensor vc = v.narrow(1, c0, n).transpose(1, 2);
        at::Tensor gc = g.narrow(1, c0, n).transpose(1, 2);
        at::Tensor bc = beta.narrow(1, c0, n).transpose(1, 2);

        at::Tensor cum = at::cumsum(gc, /*dim=*/-1);   // cum_t, inclusive, <= 0 and falling
        // Every decay appears as a **ratio** `exp(cum_t - cum_s)`, never as a pair of
        // exponentials multiplied back together. That is not cosmetic: `exp(cum)` underflows
        // to zero and `exp(-cum)` overflows to infinity once a chunk's cumulative decay
        // passes ~88, and their product is then `0 * inf = NaN`. The reference provider
        // builds the same pairwise matrix for the same reason. Only the lower triangle is
        // used, so the exponent never exceeds zero and cannot overflow.
        at::Tensor dd = cum.unsqueeze(-1) - cum.unsqueeze(-2);   // dd[t, s] = cum_t - cum_s
        at::Tensor pair = at::tril(dd, /*diagonal=*/0).exp();
        at::Tensor decay_t = cum.exp();                          // may underflow to 0: fine
        at::Tensor k_raw = at::matmul(kc, state);                // k_t^T S_prev

        // The chunk's deltas solve `(I + L) δ = diag(β)(v - exp(cum) (k S_prev))`.
        at::Tensor kk = at::matmul(kc, kc.transpose(-1, -2));
        at::Tensor lower = at::tril(pair * kk, /*diagonal=*/-1) * bc.unsqueeze(-1);
        at::Tensor tri = lower + at::eye(n, fopts);
        at::Tensor rhs = bc.unsqueeze(-1) * (vc - decay_t.unsqueeze(-1) * k_raw);
        at::Tensor delta = at::linalg_solve_triangular(tri, rhs, /*upper=*/false);

        at::Tensor mask = at::tril(pair * at::matmul(qc, kc.transpose(-1, -2)));
        at::Tensor o = at::matmul(mask, delta) + decay_t.unsqueeze(-1) * at::matmul(qc, state);
        result.narrow(1, c0, n).copy_(o.transpose(1, 2));
        // The state crosses the chunk boundary exactly once, again as a ratio:
        // `S ← exp(cum_last) S_prev + Σ_s exp(cum_last - cum_s) k_s δ_sᵀ`.
        at::Tensor last = cum.select(-1, n - 1);
        at::Tensor carry = (last.unsqueeze(-1) - cum).exp();
        state = last.exp().reshape({B, H, 1, 1}) * state +
                at::matmul((kc * carry.unsqueeze(-1)).transpose(-1, -2), delta);
    }
    return write_out(out[0], result.reshape(dims_of(out[0])).to(io_dtype), "gated_delta_rule");
}

int32_t delta_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                      uint32_t n_out, const rs_attrs* attrs) {
    return guard("gated_delta_rule", [&] {
        if (n_in != 5 || n_out != 1) {
            return fail("gated_delta_rule expects five inputs and one output");
        }
        DeltaPlan plan;
        if (!delta_plan(in, attrs, "gated_delta_rule", &plan)) {
            return 1;
        }
        const int64_t B = plan.batches;
        const int64_t S = plan.s;
        const int64_t H = plan.vh;
        const int64_t D = plan.d;
        const int64_t Dv = plan.dv;
        const double scale = 1.0 / std::sqrt(static_cast<double>(D));

        // The operator's contract keeps the recurrence state fp32 (the
        // 'state_dtype' attribute accepts only "f32"): every operand is cast
        // to f32 for the recurrence and the result is cast back to the
        // caller's float dtype, so the value matches the documented contract
        // whether the variant runs f32 or bf16.
        const at::ScalarType io_dtype = view(in[0]).scalar_type();
        const at::TensorOptions fopts = view(in[0]).options().dtype(at::kFloat);
        at::Tensor q = delta_heads(view(in[0]).to(at::kFloat), B, S, plan.kh, D, plan.repeat) * scale;
        at::Tensor k = delta_heads(view(in[1]).to(at::kFloat), B, S, plan.kh, D, plan.repeat);
        at::Tensor v = delta_heads(view(in[2]).to(at::kFloat), B, S, H, Dv, 1);
        at::Tensor g = view(in[3]).to(at::kFloat).reshape({B, S, H});
        at::Tensor beta = view(in[4]).to(at::kFloat).reshape({B, S, H});
        at::Tensor result = at::empty({B, S, H, Dv}, fopts);

        if (plan.chunk <= 1) {
            return delta_recurrence(q, k, v, g, beta, plan, result, out, io_dtype);
        }
        return delta_chunked(q, k, v, g, beta, plan, result, out, io_dtype);
    });
}

}  // namespace

void add_recurrent_ops(std::vector<OpDef>& ops) {
    ops.push_back(OpDef{"causal_conv1d", RS_SHARD_PASS_THROUGH,
                        "Depthwise causal convolution: out[t, c] = sum_k w[c, 0, k] * "
                        "x[t + k - pad, c] with x[j < 0] = 0 and an optional fused silu. x is "
                        "[.., L, C], the weight is [C, 1, K]. Executed by cuDNN's 1-D "
                        "convolution and cropped back to L.",
                        f32_mask(), RS_AUTODIFF, conv_infer, conv_execute});
    ops.push_back(OpDef{"gated_delta_rule", RS_SHARD_PASS_THROUGH,
                        "The GDN recurrence (q, k, v, g, beta) -> [.., S, vh*Dv], state fp32, "
                        "the query scaled by D^-0.5 inside, k_head_dim == v_head_dim and GQA "
                        "by repeat_interleave. This body runs the declared recurrence as the "
                        "chunked scan its own declaration names: 'chunk_size' (default 64) is "
                        "the chunk length, a chunk is one triangular solve plus a few batched "
                        "matmuls instead of one token per step, and the value is the same "
                        "(chunk_size = 1 runs the token-by-token scan).",
                        f32_mask(), RS_AUTODIFF, delta_infer, delta_execute});
}

}  // namespace rsaten
