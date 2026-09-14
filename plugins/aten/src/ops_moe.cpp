// The sparse MoE layer: one operator, ten inputs, the router resolved upstream.
//
// The reference's semantics (op-vocabulary §5) are mirrored exactly: per token
// EVERY selected expert runs — dropless, no capacity truncation — and the routed
// contributions are summed in ascending expert index, with the shared expert's
// sigmoid-gated output added afterwards. The gate/up halves are separate inputs
// because the checkpoint's fused [E, 2I, H] form cannot be sharded across the
// gate/up boundary (qwen36-5d-example §3).
//
// The body is ATen composition, in two schedules: the per-(expert, slot) loop
// (`index_select`, three matmuls, `index_copy_`) and, where `torch._grouped_mm`
// accepts the operands, one grouped GEMM per projection over the expert-ordered
// pairs. The operator's contract is the arithmetic, not the schedule; both
// schedules compute it, and the conformance gate carries a case for each.
#include "common.h"

namespace rsaten {
namespace {

constexpr int I_H = 0;
constexpr int I_ROUTING_WEIGHTS = 1;
constexpr int I_ROUTING_INDICES = 2;
constexpr int I_EXPERTS_GATE = 3;
constexpr int I_EXPERTS_UP = 4;
constexpr int I_EXPERTS_DOWN = 5;
constexpr int I_SHARED_GATE = 6;
constexpr int I_SHARED_UP = 7;
constexpr int I_SHARED_DOWN = 8;
constexpr int I_SHARED_EXPERT_GATE = 9;

/// Validates the ten-input contract and reports the shapes the body needs.
struct MoePlan {
    int64_t rows = 0;
    int64_t hidden = 0;
    int64_t inter = 0;
    int64_t experts = 0;
    int64_t top_k = 0;
};

bool moe_plan(const rs_tensor* const* in, const char* op, MoePlan* plan) {
    const rs_tensor* h = in[I_H];
    if (h->rank < 2) {
        fail(std::string(op) + ": h must have rank >= 2 ([.., H])");
        return false;
    }
    plan->hidden = h->shape[h->rank - 1];
    plan->rows = 1;
    for (uint32_t d = 0; d + 1 < h->rank; ++d) {
        plan->rows *= h->shape[d];
    }
    const rs_tensor* weights = in[I_ROUTING_WEIGHTS];
    const rs_tensor* indices = in[I_ROUTING_INDICES];
    if (weights->rank != 2 || weights->shape[0] != plan->rows) {
        fail(std::string(op) + ": routing_weights must be [rows, K]");
        return false;
    }
    plan->top_k = weights->shape[1];
    if (indices->rank != 2 || indices->shape[0] != plan->rows || indices->shape[1] != plan->top_k) {
        fail(std::string(op) + ": routing_indices must match routing_weights");
        return false;
    }
    const rs_tensor* gate = in[I_EXPERTS_GATE];
    const rs_tensor* up = in[I_EXPERTS_UP];
    const rs_tensor* down = in[I_EXPERTS_DOWN];
    if (gate->rank != 3 || up->rank != 3 || down->rank != 3) {
        fail(std::string(op) + ": expert projections must be [E, H, I]");
        return false;
    }
    plan->experts = gate->shape[0];
    plan->inter = gate->shape[2];
    if (gate->shape[1] != plan->hidden || up->shape[0] != plan->experts ||
        up->shape[1] != plan->hidden || up->shape[2] != plan->inter || down->shape[0] != plan->experts ||
        down->shape[1] != plan->hidden || down->shape[2] != plan->inter) {
        fail(std::string(op) + ": expert projections disagree on [E, H, I]");
        return false;
    }
    if (plan->top_k < 1 || plan->top_k > plan->experts) {
        fail(std::string(op) + ": K must be in [1, E]");
        return false;
    }
    const rs_tensor* shared_gate = in[I_SHARED_GATE];
    const rs_tensor* shared_up = in[I_SHARED_UP];
    const rs_tensor* shared_down = in[I_SHARED_DOWN];
    if (shared_gate->rank != 2 || shared_gate->shape[0] != plan->inter ||
        shared_gate->shape[1] != plan->hidden || shared_up->rank != 2 ||
        shared_up->shape[0] != plan->inter || shared_up->shape[1] != plan->hidden ||
        shared_down->rank != 2 || shared_down->shape[0] != plan->hidden ||
        shared_down->shape[1] != plan->inter) {
        fail(std::string(op) + ": the shared expert projections must be [I, H] / [H, I]");
        return false;
    }
    const rs_tensor* shared_expert_gate = in[I_SHARED_EXPERT_GATE];
    if (shared_expert_gate->rank != 2 || shared_expert_gate->shape[0] != 1 ||
        shared_expert_gate->shape[1] != plan->hidden) {
        fail(std::string(op) + ": shared_expert_gate must be [1, H]");
        return false;
    }
    return true;
}

int32_t moe_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                  uint32_t n_out, const rs_attrs*) {
    return guard("moe_layer", [&] {
        if (n_in != 10 || n_out != 1) {
            return fail("moe_layer expects ten inputs and one output");
        }
        for (int i = 0; i < 10; ++i) {
            if (i == I_ROUTING_INDICES) {
                continue;  // indices are i32/i64, checked below
            }
            int rc = check_float(in[i], "moe_layer", "input");
            if (rc != 0) {
                return rc;
            }
        }
        if (in[I_ROUTING_INDICES]->dtype != RS_I32 && in[I_ROUTING_INDICES]->dtype != RS_I64) {
            return fail("moe_layer: routing_indices must be i32 or i64");
        }
        MoePlan plan;
        if (!moe_plan(in, "moe_layer", &plan)) {
            return 1;
        }
        set_shape(out[0], float_dtype_of(in[0]), dims_of(in[I_H]));
        return 0;
    });
}

/// Which (row, slot) pairs selected which expert — a fact about *data*, resolved
/// on the host once and shared by both schedules.
///
/// Asking the device for it inside the loop is what the per-expert schedule used
/// to do: `at::nonzero` returns a data-dependent shape, so every call
/// synchronises, and E * K of them measured ~92 ms of the ~98 ms that schedule
/// took. The indices tensor is tiny by construction ([rows, K]), so one copy to
/// the host answers the same question with one synchronisation. `rows_by[e][k]`
/// is ascending in `r`, which is the order the reference contract accumulates in.
using RowsByExpert = std::vector<std::vector<std::vector<int64_t>>>;

int32_t moe_routing(const at::Tensor& indices, int64_t rows, int64_t K, int64_t E,
                    RowsByExpert* rows_by) {
    at::Tensor indices_cpu = indices.to(at::kCPU).contiguous();
    const bool wide = indices_cpu.scalar_type() == at::kLong;
    const void* index_data = indices_cpu.data_ptr();
    rows_by->assign(static_cast<size_t>(E),
                    std::vector<std::vector<int64_t>>(static_cast<size_t>(K)));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t k = 0; k < K; ++k) {
            const int64_t at = r * K + k;
            const int64_t e =
                wide ? static_cast<const int64_t*>(index_data)[at]
                     : static_cast<int64_t>(static_cast<const int32_t*>(index_data)[at]);
            if (e < 0 || e >= E) {
                return fail("moe_layer: routing index " + std::to_string(e) + " at row " +
                            std::to_string(r) + ", slot " + std::to_string(k) +
                            " is outside [0, " + std::to_string(E) +
                            "); out-of-range indices are a hard error, never a drop");
            }
            (*rows_by)[static_cast<size_t>(e)][static_cast<size_t>(k)].push_back(r);
        }
    }
    return 0;
}

/// The shared expert's contribution: `sigmoid(gate @ x) * down(silu(x @ Wg^T) * (x @ Wu^T))`.
///
/// Both schedules run this identical tail — it is not part of what a schedule
/// chooses, and keeping it in one place is what stops the two bodies from
/// drifting apart on the half of the operator nobody is optimizing.
at::Tensor moe_shared_expert(const rs_tensor* const* in, const at::Tensor& h) {
    at::Tensor sg = at::sigmoid(at::matmul(h, view(in[I_SHARED_EXPERT_GATE]).t()));
    at::Tensor g = at::matmul(h, view(in[I_SHARED_GATE]).t());
    at::Tensor u = at::matmul(h, view(in[I_SHARED_UP]).t());
    at::Tensor hidden = (g / (1.0 + (-g).exp())) * u;
    return sg * at::matmul(hidden, view(in[I_SHARED_DOWN]).t());
}

/// Whether `torch._grouped_mm` can run these operands at all.
///
/// It requires the contraction extent of every matrix operand to be a multiple of
/// 16 bytes (`GroupedMMUtils.h: check_valid_strides_and_return_transposed`) — the
/// alignment cuBLAS's grouped tensor-core path needs. The model's geometry meets it
/// (H = 2048, I = 512, so at 2 bytes per element both are multiples of 8); the
/// conformance gate carries one case per schedule for exactly that reason: H = I = 2
/// is unaligned and reaches the loop, H = I = 8 is aligned and reaches this one.
///
/// When the rule does not hold the reference loop runs instead of the operator
/// failing: the shapes are legal, and the loop is right here — the schedule every
/// earlier numeric claim was made with. That is a **schedule** choice inside a body,
/// the same kind of choice ATen makes when it picks a kernel by shape; it is not a
/// resolution change (the variant was already chosen) and it is not a contract change
/// (same rows, same routing weights, same ascending-slot sum). Which schedule ran is
/// deliberately not observable downstream — no ABI channel carries it and the plan
/// digest names the variant, not the body's internal branch — so the gate covers both
/// by running both.
bool grouped_mm_fits(const at::Tensor& h, int64_t inter, int64_t elem_size) {
    const int64_t alignment = 16 / elem_size;
    return h.size(-1) % alignment == 0 && inter % alignment == 0;
}

/// The grouped schedule — declared here and defined below, so the dispatch in
/// [`moe_execute`] reads next to the rule that chooses it.
int32_t moe_grouped(const rs_tensor* const* in, const MoePlan& plan, rs_tensor* const* out);

int32_t moe_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                    uint32_t n_out, const rs_attrs*) {
    return guard("moe_layer", [&] {
        if (n_in != 10 || n_out != 1) {
            return fail("moe_layer expects ten inputs and one output");
        }
        MoePlan plan;
        if (!moe_plan(in, "moe_layer", &plan)) {
            return 1;
        }
        const int64_t rows = plan.rows;
        const int64_t K = plan.top_k;
        const int64_t E = plan.experts;

        // The schedule. The loop's cost is per-call overhead paid `E * K` times —
        // 2048 at this model's geometry, measured 59.3 ms per layer at 8 tokens and
        // 135 ms at 512, i.e. it is not doing arithmetic, it is dispatching — so the
        // grouped path runs wherever `torch._grouped_mm` accepts the operands, at any
        // dtype. The loop stays right below as the fallback for geometries the grouped
        // kernel cannot take, and the conformance gate carries one case per schedule
        // (`crates/rustrain-runtime/src/conformance.rs`): H = I = 2 takes the loop,
        // H = I = 8 is aligned and takes the grouped one.
        at::Tensor h = view(in[I_H]).reshape({rows, plan.hidden});
        if (grouped_mm_fits(h, plan.inter, h.element_size())) {
            return moe_grouped(in, plan, out);
        }
        at::Tensor weights = view(in[I_ROUTING_WEIGHTS]).reshape({rows, K});
        at::Tensor indices = as_indices(in[I_ROUTING_INDICES]).reshape({rows, K});
        at::Tensor gate = view(in[I_EXPERTS_GATE]);
        at::Tensor up = view(in[I_EXPERTS_UP]);
        at::Tensor down = view(in[I_EXPERTS_DOWN]);

        // Which rows selected which expert is a fact about *data*, and the loop below needs it
        // per (expert, k). Asking the device for it inside the loop — `at::nonzero` returns a
        // data-dependent shape, so each call synchronises — costs E * K of those per call: 2048
        // synchronisations, measured at ~92 ms of the ~98 ms this layer used to take.
        RowsByExpert rows_by;
        if (moe_routing(indices, rows, K, E, &rows_by) != 0) {
            return 1;
        }

        // One contribution per (row, selected expert) pair, then a reduction
        // over k. Accumulating with `index_add_` would be shorter but it uses
        // atomics on CUDA: two runs on the same input can produce different
        // bytes, and the conformance gate's determinism check compares exactly
        // that. `index_copy_` writes a unique location per (row, k) — each row
        // picks one expert per rank-of-selection — so the accumulation is
        // deterministic and the final sum runs in ascending k, which is the
        // order the operator's contract declares.
        at::Tensor contributions = at::zeros({rows, K, plan.hidden}, h.options());
        for (int64_t e = 0; e < E; ++e) {
            for (int64_t k = 0; k < K; ++k) {
                const std::vector<int64_t>& rows_here =
                    rows_by[static_cast<size_t>(e)][static_cast<size_t>(k)];
                if (rows_here.empty()) {
                    continue;
                }
                at::Tensor selected =
                    at::tensor(rows_here, at::dtype(at::kLong)).to(h.device());
                at::Tensor x = h.index_select(0, selected);
                at::Tensor g = at::matmul(x, gate.select(0, e));
                at::Tensor u = at::matmul(x, up.select(0, e));
                at::Tensor hidden = (g / (1.0 + (-g).exp())) * u;
                // `experts_down_proj` is [E, H, I] with H the output: the
                // reference computes `acc[hh] += a[j] * down[e][hh][j]`, which is
                // `hidden @ down[e]^T`, not `hidden @ down[e]`. The transpose is
                // the whole difference between the right answer and a shape error
                // (or, when H == I, a silently wrong one).
                at::Tensor y = at::matmul(hidden, down.select(0, e).t());
                at::Tensor w = weights.select(1, k).index_select(0, selected).reshape({-1, 1});
                contributions.select(1, k).index_copy_(0, selected, y * w);
            }
        }
        at::Tensor result = contributions.sum(/*dim=*/1);
        result = result + moe_shared_expert(in, h);

        return write_out(out[0], result.reshape(dims_of(out[0])), "moe_layer");
    });
}

/// The grouped schedule: one grouped GEMM per projection instead of `E * K` small ones.
///
/// The loop pays ATen's dispatch and kernel-launch cost `E * K` times per call — 2048
/// at this model's geometry — whether or not an expert has rows, which is why the
/// layer's cost barely moves with token count. This schedule makes the loop go away:
/// the (row, slot) pairs are sorted into expert order *on the host* (the same pass the
/// loop's `rows_by` needs, so the two schedules cannot disagree about who sees what),
/// which turns the three projections into three `torch._grouped_mm` calls whose
/// segment sizes come from that ordering.
///
/// What it does **not** change: which rows each expert sees, the routing weight each
/// contribution is scaled by, and the ascending-slot sum that produces the output. The
/// arithmetic is the same; the GEMM's internal reduction order is not, so the result
/// is compared against the loop rather than assumed identical to it (host measurements
/// in `docs/design/qwen36-text/spec.md` D6.12).
///
/// The plan is validated and `guard`ed by the caller; this function only computes.
int32_t moe_grouped(const rs_tensor* const* in, const MoePlan& plan, rs_tensor* const* out) {
    const int64_t rows = plan.rows;
    const int64_t K = plan.top_k;
    const int64_t E = plan.experts;
    const int64_t pairs = rows * K;
    // `_grouped_mm` takes int32 offsets, so a pair count that does not fit one
    // cannot be expressed — and a silent `static_cast` here would wrap into a
    // schedule that runs and computes the wrong rows. Refuse instead.
    if (pairs > std::numeric_limits<int32_t>::max()) {
        return fail("moe_layer: " + std::to_string(pairs) +
                    " (row, slot) pairs do not fit the int32 offsets `torch._grouped_mm` takes");
    }

    at::Tensor h = view(in[I_H]).reshape({rows, plan.hidden});
    at::Tensor weights = view(in[I_ROUTING_WEIGHTS]).reshape({rows, K});
    at::Tensor indices = as_indices(in[I_ROUTING_INDICES]).reshape({rows, K});

    RowsByExpert rows_by;
    if (moe_routing(indices, rows, K, E, &rows_by) != 0) {
        return 1;
    }

    // The expert-major pair order, and the group boundaries `_grouped_mm`
    // takes: `offsets[e]` is the row where expert `e`'s group **ends**
    // (exclusive), the convention the op's own fallback slices with —
    // `GroupedMMUtils.h: _grouped_mm_fallback` runs `mat_a.slice(0,
    // group_start, offs[group_idx])` and carries `group_start = offs[...]`
    // forward — so the last entry is the gathered row count, not its
    // predecessor. Reading it as the group *start* shifts every expert onto
    // its neighbour's rows and drops the last one: the first host A/B of this
    // schedule measured 8.8e-1 relative error on the logits for exactly that
    // reason. An expert nobody routed to contributes an empty group, which the
    // kernel accepts — padding it to one row would multiply a row that must
    // not exist.
    std::vector<int64_t> order;
    std::vector<int32_t> offsets;
    order.reserve(static_cast<size_t>(pairs));
    offsets.reserve(static_cast<size_t>(E));
    for (int64_t e = 0; e < E; ++e) {
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t r : rows_by[static_cast<size_t>(e)][static_cast<size_t>(k)]) {
                order.push_back(r * K + k);
            }
        }
        offsets.push_back(static_cast<int32_t>(order.size()));
    }
    if (static_cast<int64_t>(order.size()) != pairs) {
        return fail("moe_layer: the expert-major order covers " + std::to_string(order.size()) +
                    " of " + std::to_string(pairs) +
                    " (row, slot) pairs; the routing pass lost one");
    }

    at::Tensor order_t = at::tensor(order, at::dtype(at::kLong)).to(h.device());
    at::Tensor offsets_t = at::tensor(offsets, at::dtype(at::kInt)).to(h.device());
    // The input row of each pair: pair `r * K + k` is row `r`, so the gather
    // is `order / K`. Integer division on a host vector, not a device op.
    std::vector<int64_t> pair_rows;
    pair_rows.reserve(order.size());
    for (int64_t pair : order) {
        pair_rows.push_back(pair / K);
    }
    at::Tensor x =
        h.index_select(0, at::tensor(pair_rows, at::dtype(at::kLong)).to(h.device()));

    at::Tensor gate = view(in[I_EXPERTS_GATE]);
    at::Tensor up = view(in[I_EXPERTS_UP]);
    at::Tensor down = view(in[I_EXPERTS_DOWN]);
    at::Tensor g = at::_grouped_mm(x, gate, offsets_t);
    at::Tensor u = at::_grouped_mm(x, up, offsets_t);
    at::Tensor hidden = (g / (1.0 + (-g).exp())) * u;
    // `experts_down_proj` is [E, H, I] and the projection contracts I, so the
    // grouped GEMM wants it transposed. The transpose is a *view* —
    // `_grouped_mm` reads the strides of `mat2` and accepts the transposed
    // layout, so no (E*H*I) copy happens per call.
    at::Tensor y = at::_grouped_mm(hidden, down.transpose(1, 2), offsets_t);
    at::Tensor scaled =
        y * weights.reshape({pairs}).index_select(0, order_t).reshape({-1, 1});

    // Scatter the pairs back to their (row, slot) positions and sum in
    // ascending slot order, exactly as the loop does. `index_copy_` rather
    // than `index_add_`: every position is written exactly once, so there is
    // no atomics nondeterminism to inherit.
    at::Tensor contributions = at::empty({pairs, plan.hidden}, h.options());
    contributions.index_copy_(0, order_t, scaled);
    at::Tensor result = contributions.reshape({rows, K, plan.hidden}).sum(/*dim=*/1);
    result = result + moe_shared_expert(in, h);

    return write_out(out[0], result.reshape(dims_of(out[0])), "moe_layer");
}

}  // namespace

namespace {

/// The communication this operator's math owes, declared because the layouts
/// cannot show it: the routing is data, and the intermediate dim a tp split
/// cuts is the contraction dim of the down projections, so the output is a
/// partial sum over tp. The planner turns the ALL_REDUCE into a
/// `partial(sum, tp)` and places the reduction at the consumer that needs a
/// replicated value; the two ALL_TO_ALL are the expert dispatch/combine.
const rs_collective kMoeCollectives[] = {
    {RS_C_ALL_TO_ALL, static_cast<rs_group_kind>(RS_G_TP | RS_G_EP), 0, 0},
    {RS_C_ALL_TO_ALL, static_cast<rs_group_kind>(RS_G_TP | RS_G_EP), 10, 0},
    {RS_C_ALL_REDUCE, RS_G_TP, 10, 0},
};

}  // namespace

void add_moe_ops(std::vector<OpDef>& ops) {
    ops.push_back(OpDef{"moe_layer", RS_SHARD_PASS_THROUGH,
                        "Qwen3.6's sparse MoE layer as one operator: ten inputs (h, routing "
                        "weights and indices from topk_router, the DE-FUSED expert gate/up/down "
                        "projections and the shared expert), dropless routing, per-token sum "
                        "over the selected experts plus the sigmoid-gated shared expert.",
                        (1u << RS_F32) | (1u << RS_I32) | (1u << RS_I64), RS_AUTODIFF, moe_infer,
                        moe_execute,
                        /* expansion = */ nullptr, kMoeCollectives, 3});
}

}  // namespace rsaten
