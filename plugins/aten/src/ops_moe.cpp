// The sparse MoE layer: one operator, ten inputs, the router resolved upstream.
//
// The reference's semantics (op-vocabulary §5) are mirrored exactly: per token
// EVERY selected expert runs — dropless, no capacity truncation — and the routed
// contributions are summed in ascending expert index, with the shared expert's
// sigmoid-gated output added afterwards. The gate/up halves are separate inputs
// because the checkpoint's fused [E, 2I, H] form cannot be sharded across the
// gate/up boundary (qwen36-5d-example §3).
//
// The body is ATen composition: tokens are selected per expert, three matmuls
// run, and the results are accumulated with `index_add_`. A grouped GEMM
// (`torch._grouped_mm`, or an upstream kernel) is the fast path this body
// deliberately leaves for later — the operator's contract is the arithmetic, not
// the schedule.
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
            int rc = check_f32(in[i], "moe_layer", "input");
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
        set_shape(out[0], dims_of(in[I_H]));
        return 0;
    });
}

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

        at::Tensor h = view(in[I_H]).reshape({rows, plan.hidden});
        at::Tensor weights = view(in[I_ROUTING_WEIGHTS]).reshape({rows, K});
        at::Tensor indices = as_indices(in[I_ROUTING_INDICES]).reshape({rows, K});
        at::Tensor gate = view(in[I_EXPERTS_GATE]);
        at::Tensor up = view(in[I_EXPERTS_UP]);
        at::Tensor down = view(in[I_EXPERTS_DOWN]);

        // Which rows selected which expert is a fact about *data*, and the loop below needs it
        // per (expert, k). Asking the device for it inside the loop — `at::nonzero` returns a
        // data-dependent shape, so each call synchronises — costs E * K of those per call: 2048
        // synchronisations, measured at ~92 ms of the ~98 ms this layer used to take. They are
        // computed once, on the host, from a tensor that is tiny by construction ([rows, K]);
        // nothing else changes, so the same rows are selected in the same (ascending) order and
        // the same expert sub-matrices are multiplied with the same shapes.
        auto indices_cpu = indices.to(at::kCPU).contiguous();
        const bool wide = indices_cpu.scalar_type() == at::kLong;
        const void* index_data = indices_cpu.data_ptr();
        std::vector<std::vector<std::vector<int64_t>>> rows_by(
            static_cast<size_t>(E), std::vector<std::vector<int64_t>>(static_cast<size_t>(K)));
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t k = 0; k < K; ++k) {
                const int64_t at = r * K + k;
                const int64_t e = wide ? static_cast<const int64_t*>(index_data)[at]
                                       : static_cast<int64_t>(static_cast<const int32_t*>(index_data)[at]);
                if (e < 0 || e >= E) {
                    return fail("moe_layer: routing index " + std::to_string(e) + " at row " +
                                std::to_string(r) + ", slot " + std::to_string(k) +
                                " is outside [0, " + std::to_string(E) +
                                "); out-of-range indices are a hard error, never a drop");
                }
                rows_by[static_cast<size_t>(e)][static_cast<size_t>(k)].push_back(r);
            }
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

        // The shared expert: sigmoid(shared_expert_gate @ x) * down(silu(x @ Wg^T) * (x @ Wu^T)).
        at::Tensor shared_gate = view(in[I_SHARED_GATE]);
        at::Tensor shared_up = view(in[I_SHARED_UP]);
        at::Tensor shared_down = view(in[I_SHARED_DOWN]);
        at::Tensor sg = at::sigmoid(at::matmul(h, view(in[I_SHARED_EXPERT_GATE]).t()));
        at::Tensor g = at::matmul(h, shared_gate.t());
        at::Tensor u = at::matmul(h, shared_up.t());
        at::Tensor hidden = (g / (1.0 + (-g).exp())) * u;
        result = result + sg * at::matmul(hidden, shared_down.t());

        return write_out(out[0], result.reshape(dims_of(out[0])), "moe_layer");
    });
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
