// The normalisation family: the two trunk normalisations, GDN's L2
// normalisation and GDN's gated output normalisation.
//
// Each is a short ATen composition. Two conventions are load bearing and are
// the reason these are not `at::layer_norm` one-liners:
//
//   * `rmsnorm` multiplies by the RAW weight plus a declared offset — the trunk
//     uses 1.0 (HF's `1 + weight`), GDN's gated normalisation uses 0.0. The
//     offset is data, so it is read from the attributes, never inferred from
//     the weight's values;
//   * `rmsnorm_gated` fuses the gating into the same pass: the gate is applied
//     AFTER the row normalisation, which is what the HF implementation does.
#include "common.h"

namespace rsaten {
namespace {

/// The rows of `x` treated as [rows, D] with D the last dim; the reduction runs
/// over D only. Written as `1 / sqrt(mean + eps)` rather than `rsqrt`, so the
/// arithmetic is the reference's expression and not merely equal to it.
at::Tensor rms_scale(const at::Tensor& x, double eps) {
    at::Tensor sq = x * x;
    at::Tensor mean = sq.mean(/*dim=*/-1, /*keepdim=*/true);
    return 1.0 / at::sqrt(mean + eps);
}

int32_t rmsnorm_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                      uint32_t n_out, const rs_attrs*) {
    return guard("rmsnorm", [&] {
        if (n_in < 1 || n_in > 2 || n_out != 1) {
            return fail("rmsnorm expects one or two inputs and one output");
        }
        int rc = check_f32(in[0], "rmsnorm", "x");
        if (rc != 0) {
            return rc;
        }
        if (in[0]->rank < 1) {
            return fail("rmsnorm expects rank >= 1");
        }
        if (n_in == 2) {
            rc = check_f32(in[1], "rmsnorm", "w");
            if (rc != 0) {
                return rc;
            }
            if (in[1]->rank != 1 || in[1]->shape[0] != in[0]->shape[in[0]->rank - 1]) {
                return fail("rmsnorm weight must be [D] with D the last dim");
            }
        }
        set_shape(out[0], dims_of(in[0]));
        return 0;
    });
}

int32_t rmsnorm_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                        uint32_t n_out, const rs_attrs* attrs) {
    return guard("rmsnorm", [&] {
        if (n_in < 1 || n_in > 2 || n_out != 1) {
            return fail("rmsnorm expects one or two inputs and one output");
        }
        at::Tensor y = view(in[0]) * rms_scale(view(in[0]), f64_or(attrs, "eps", 1e-5));
        if (n_in == 2) {
            double offset = f64_or(attrs, "weight_offset", 0.0);
            at::Tensor w = view(in[1]);
            y = y * (offset == 0.0 ? w : w + offset);
        }
        return write_out(out[0], y, "rmsnorm");
    });
}

int32_t layernorm_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                        uint32_t n_out, const rs_attrs*) {
    return guard("layernorm", [&] {
        if (n_in < 1 || n_in > 3 || n_out != 1) {
            return fail("layernorm expects one to three inputs and one output");
        }
        int rc = check_f32(in[0], "layernorm", "x");
        if (rc != 0) {
            return rc;
        }
        if (in[0]->rank < 1) {
            return fail("layernorm expects rank >= 1");
        }
        int64_t d = in[0]->shape[in[0]->rank - 1];
        for (uint32_t i = 1; i < n_in; ++i) {
            rc = check_f32(in[i], "layernorm", "w/b");
            if (rc != 0) {
                return rc;
            }
            if (in[i]->rank != 1 || in[i]->shape[0] != d) {
                return fail("layernorm parameters must be [D] with D the last dim");
            }
        }
        set_shape(out[0], dims_of(in[0]));
        return 0;
    });
}

int32_t layernorm_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in,
                          rs_tensor* const* out, uint32_t n_out, const rs_attrs* attrs) {
    return guard("layernorm", [&] {
        if (n_in < 1 || n_in > 3 || n_out != 1) {
            return fail("layernorm expects one to three inputs and one output");
        }
        at::Tensor x = view(in[0]);
        double eps = f64_or(attrs, "eps", 1e-5);
        at::Tensor mean = x.mean(/*dim=*/-1, /*keepdim=*/true);
        at::Tensor centered = x - mean;
        // Biased variance, the reference's documented convention.
        at::Tensor var = (centered * centered).mean(/*dim=*/-1, /*keepdim=*/true);
        at::Tensor y = centered * (1.0 / at::sqrt(var + eps));
        if (n_in >= 2) {
            y = y * view(in[1]);
        }
        if (n_in == 3) {
            y = y + view(in[2]);
        }
        return write_out(out[0], y, "layernorm");
    });
}

int32_t l2norm_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                     uint32_t n_out, const rs_attrs* attrs) {
    return guard("l2norm", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("l2norm expects one input and one output");
        }
        int rc = check_f32(in[0], "l2norm", "x");
        if (rc != 0) {
            return rc;
        }
        int64_t dim = 0;
        if (!resolve_dim(i64_or(attrs, "dim", -1), in[0]->rank, "l2norm", &dim)) {
            return 1;
        }
        set_shape(out[0], dims_of(in[0]));
        return 0;
    });
}

int32_t l2norm_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                       uint32_t n_out, const rs_attrs* attrs) {
    return guard("l2norm", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("l2norm expects one input and one output");
        }
        int64_t dim = 0;
        if (!resolve_dim(i64_or(attrs, "dim", -1), in[0]->rank, "l2norm", &dim)) {
            return 1;
        }
        at::Tensor x = view(in[0]);
        // The SUM of squares (not the mean) with eps inside the sqrt — GDN's
        // convention, aligned with the HF l2norm.
        at::Tensor sq = x * x;
        at::Tensor norm = at::sqrt(sq.sum(dim, /*keepdim=*/true) + f64_or(attrs, "eps", 1e-6));
        return write_out(out[0], x / norm, "l2norm");
    });
}

int32_t rmsnorm_gated_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                            uint32_t n_out, const rs_attrs* attrs) {
    return guard("rmsnorm_gated", [&] {
        if (n_in != 3 || n_out != 1) {
            return fail("rmsnorm_gated expects three inputs and one output");
        }
        int rc = check_f32(in[0], "rmsnorm_gated", "x");
        if (rc != 0) {
            return rc;
        }
        if (in[0]->rank < 1) {
            return fail("rmsnorm_gated expects rank >= 1");
        }
        int64_t d = in[0]->shape[in[0]->rank - 1];
        rc = check_f32(in[1], "rmsnorm_gated", "w");
        if (rc != 0) {
            return rc;
        }
        if (in[1]->rank != 1 || in[1]->shape[0] != d) {
            return fail("rmsnorm_gated weight must be [D] with D the last dim");
        }
        rc = check_f32(in[2], "rmsnorm_gated", "gate");
        if (rc != 0) {
            return rc;
        }
        if (numel_of(in[2]) != numel_of(in[0])) {
            return fail("rmsnorm_gated gate must have the same element count as x");
        }
        std::string act;
        static const char* const GATE_ACTS[] = {"silu"};
        if (!require_kind(attrs, "gate_act", GATE_ACTS, 1, "rmsnorm_gated", &act)) {
            return 1;
        }
        set_shape(out[0], dims_of(in[0]));
        return 0;
    });
}

int32_t rmsnorm_gated_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in,
                              rs_tensor* const* out, uint32_t n_out, const rs_attrs* attrs) {
    return guard("rmsnorm_gated", [&] {
        if (n_in != 3 || n_out != 1) {
            return fail("rmsnorm_gated expects three inputs and one output");
        }
        at::Tensor x = view(in[0]);
        double offset = f64_or(attrs, "weight_offset", 0.0);
        at::Tensor w = view(in[1]);
        at::Tensor y = x * rms_scale(x, f64_or(attrs, "eps", 1e-6));
        y = y * (offset == 0.0 ? w : w + offset);
        at::Tensor gate = view(in[2]).reshape(x.sizes());
        return write_out(out[0], y * (gate / (1.0 + (-gate).exp())), "rmsnorm_gated");
    });
}

}  // namespace

void add_norm_ops(std::vector<OpDef>& ops) {
    ops.push_back(OpDef{"rmsnorm", RS_SHARD_ELEMENTWISE,
                        "y = x / sqrt(mean(x^2) + eps) * (w + weight_offset), normalised over "
                        "the last dim; eps (default 1e-5) sits inside the sqrt and the weight "
                        "offset is declared data (the trunk uses 1.0).",
                        f32_mask(), RS_AUTODIFF, rmsnorm_infer, rmsnorm_execute});
    ops.push_back(OpDef{"layernorm", RS_SHARD_ELEMENTWISE,
                        "y = (x - mean) / sqrt(var + eps) * w + b over the last dim, biased "
                        "variance, eps (default 1e-5) outside the sqrt.",
                        f32_mask(), RS_AUTODIFF, layernorm_infer, layernorm_execute});
    ops.push_back(OpDef{"l2norm", RS_SHARD_PASS_THROUGH,
                        "y = x / sqrt(sum(x^2, dim) + eps): the SUM of squares with eps inside "
                        "the sqrt, dim (default -1) and eps (default 1e-6).",
                        f32_mask(), RS_AUTODIFF, l2norm_infer, l2norm_execute});
    ops.push_back(OpDef{"rmsnorm_gated", RS_SHARD_PASS_THROUGH,
                        "GDN's output normalisation: the row normalisation, the raw weight and "
                        "silu(gate) in one pass, gate applied after the normalisation.",
                        f32_mask(), RS_AUTODIFF, rmsnorm_gated_infer, rmsnorm_gated_execute});
}

}  // namespace rsaten
