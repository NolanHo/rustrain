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
//     it (`S_t = S_{t-1} exp(g_t) + k_t ((v_t - S_{t-1} k_t) beta_t)`, read after
//     the update), expressed as ATen matrix products over the state. The
//     reference's body is the chunked form of the same function; the chunked
//     form exists for speed, and the declared `chunk_size` sizes it rather than
//     changing the value — which is why this body does not read the attribute
//     and says so in its doc string.
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
        int rc = check_f32(in[0], "causal_conv1d", "x");
        if (rc != 0) {
            return rc;
        }
        rc = check_f32(in[1], "causal_conv1d", "w");
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
        set_shape(out[0], dims_of(in[0]));
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
};

bool delta_plan(const rs_tensor* const* in, const rs_attrs* attrs, const char* op,
                DeltaPlan* plan) {
    const rs_tensor* q = in[0];
    const rs_tensor* k = in[1];
    const rs_tensor* v = in[2];
    const rs_tensor* g = in[3];
    const rs_tensor* beta = in[4];
    for (int i = 0; i < 5; ++i) {
        int rc = check_f32(in[i], op, "q/k/v/g/beta");
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
        set_shape(out[0], shape);
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

        at::Tensor q = delta_heads(view(in[0]), B, S, plan.kh, D, plan.repeat) * scale;
        at::Tensor k = delta_heads(view(in[1]), B, S, plan.kh, D, plan.repeat);
        at::Tensor v = delta_heads(view(in[2]), B, S, H, Dv, 1);
        at::Tensor g = view(in[3]).reshape({B, S, H});
        at::Tensor beta = view(in[4]).reshape({B, S, H});

        at::Tensor state = at::zeros({B, H, D, Dv}, q.options());
        at::Tensor result = at::empty({B, S, H, Dv}, q.options());

        // The recurrence is sequential in t by definition; everything else is
        // batched over (batch, head), so the loop runs S times and each step is
        // a handful of small matrix products.
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
        return write_out(out[0], result.reshape(dims_of(out[0])), "gated_delta_rule");
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
                        "by repeat_interleave. This body runs the declared recurrence "
                        "directly; 'chunk_size' sizes the chunked fast path and does not "
                        "change the value, so it is not read here.",
                        f32_mask(), RS_AUTODIFF, delta_infer, delta_execute});
}

}  // namespace rsaten
