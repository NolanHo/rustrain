// Compute operators: the matrix products, the elementwise family, the
// reductions, attention and the router.
//
// Every body here is a composition of ATen operators. Nothing in this file is a
// kernel: cuBLAS runs the GEMMs, ATen's CUDA kernels run the elementwise and
// index work, and `at::scaled_dot_product_attention` picks FlashAttention-2 /
// memory-efficient attention for the attention node. The mapping is what this
// plugin contributes; the arithmetic is upstream's.
//
// The semantics mirrored are the reference provider's contracts
// (crates/rustrain-kernels/src/op/*.rs and docs/design/op-vocabulary.md): the
// conformance gate compares this implementation against that oracle op by op,
// so a deviation here has to be a deviation the gate can see.
#include "common.h"

namespace rsaten {
namespace {

const char* const UNARY_KINDS[] = {
    "silu",          "gelu",        "sigmoid", "tanh",      "relu",        "exp",
    "log",           "neg",         "sqrt",    "rsqrt",     "softplus",    "negative_exp",
    "silu_grad",     "gelu_grad",   "sigmoid_grad", "tanh_grad", "relu_grad",
};
const int N_UNARY_KINDS = static_cast<int>(sizeof(UNARY_KINDS) / sizeof(UNARY_KINDS[0]));

const char* const BINARY_KINDS[] = {"add", "sub", "mul", "div", "maximum", "pow"};
const int N_BINARY_KINDS = static_cast<int>(sizeof(BINARY_KINDS) / sizeof(BINARY_KINDS[0]));

const char* const COMPARE_KINDS[] = {"eq", "ne", "lt", "le", "gt", "ge"};
const int N_COMPARE_KINDS = static_cast<int>(sizeof(COMPARE_KINDS) / sizeof(COMPARE_KINDS[0]));

const char* const REDUCE_KINDS[] = {"sum", "mean", "max", "amax"};
const int N_REDUCE_KINDS = static_cast<int>(sizeof(REDUCE_KINDS) / sizeof(REDUCE_KINDS[0]));

const char* const SCATTER_REDUCE_KINDS[] = {"assign", "add"};
const int N_SCATTER_REDUCE_KINDS = 2;

// ── matmul / linear / bmm ───────────────────────────────────────────────────

int32_t matmul_shape(const rs_tensor* a, const rs_tensor* b, const rs_attrs* attrs,
                     std::vector<int64_t>* shape) {
    int rc = check_f32(a, "matmul", "a");
    if (rc != 0) {
        return rc;
    }
    rc = check_f32(b, "matmul", "b");
    if (rc != 0) {
        return rc;
    }
    if (a->rank != 2 || b->rank != 2) {
        return fail("matmul expects rank-2 inputs");
    }
    bool transpose_b = bool_or(attrs, "transpose_b", false);
    if (transpose_b) {
        if (a->shape[1] != b->shape[1]) {
            return fail("matmul: inner dims mismatch with transpose_b (b is [N, K])");
        }
        *shape = {a->shape[0], b->shape[0]};
    } else {
        if (a->shape[1] != b->shape[0]) {
            return fail("matmul: inner dims mismatch (b must be [K, N])");
        }
        *shape = {a->shape[0], b->shape[1]};
    }
    return 0;
}

int32_t matmul_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                     uint32_t n_out, const rs_attrs* attrs) {
    return guard("matmul", [&] {
        if (n_in != 2 || n_out != 1) {
            return fail("matmul expects two inputs and one output");
        }
        std::vector<int64_t> shape;
        int rc = matmul_shape(in[0], in[1], attrs, &shape);
        if (rc != 0) {
            return rc;
        }
        set_shape(out[0], shape);
        return 0;
    });
}

int32_t matmul_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                       uint32_t n_out, const rs_attrs* attrs) {
    return guard("matmul", [&] {
        if (n_in != 2 || n_out != 1) {
            return fail("matmul expects two inputs and one output");
        }
        std::vector<int64_t> shape;
        int rc = matmul_shape(in[0], in[1], attrs, &shape);
        if (rc != 0) {
            return rc;
        }
        at::Tensor a = view(in[0]);
        at::Tensor b = view(in[1]);
        at::Tensor c = bool_or(attrs, "transpose_b", false) ? at::matmul(a, b.t()) : at::matmul(a, b);
        return write_out(out[0], c, "matmul");
    });
}

/// `linear`'s weight convention: `w` is `[K, N]`, so the product is a plain
/// `x @ w` — torch's `nn.Linear` stores `[N, K]` and is NOT this contract.
int32_t linear_shape(const rs_tensor* x, const rs_tensor* w, const rs_tensor* b,
                     std::vector<int64_t>* shape) {
    int rc = check_f32(x, "linear", "x");
    if (rc != 0) {
        return rc;
    }
    rc = check_f32(w, "linear", "w");
    if (rc != 0) {
        return rc;
    }
    if (x->rank < 1 || w->rank != 2) {
        return fail("linear expects x with rank >= 1 and w with rank 2");
    }
    int64_t k = x->shape[x->rank - 1];
    if (w->shape[0] != k) {
        return fail("linear: inner dim mismatch (w must be [K= " + std::to_string(k) + ", N])");
    }
    if (b != nullptr) {
        rc = check_f32(b, "linear", "b");
        if (rc != 0) {
            return rc;
        }
        if (b->rank != 1 || b->shape[0] != w->shape[1]) {
            return fail("linear: bias must be [N]");
        }
    }
    shape->assign(x->shape, x->shape + x->rank);
    (*shape)[x->rank - 1] = w->shape[1];
    return 0;
}

int32_t linear_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                     uint32_t n_out, const rs_attrs*) {
    return guard("linear", [&] {
        if ((n_in != 2 && n_in != 3) || n_out != 1) {
            return fail("linear expects two or three inputs and one output");
        }
        std::vector<int64_t> shape;
        int rc = linear_shape(in[0], in[1], n_in == 3 ? in[2] : nullptr, &shape);
        if (rc != 0) {
            return rc;
        }
        set_shape(out[0], shape);
        return 0;
    });
}

int32_t linear_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                       uint32_t n_out, const rs_attrs*) {
    return guard("linear", [&] {
        if ((n_in != 2 && n_in != 3) || n_out != 1) {
            return fail("linear expects two or three inputs and one output");
        }
        std::vector<int64_t> shape;
        int rc = linear_shape(in[0], in[1], n_in == 3 ? in[2] : nullptr, &shape);
        if (rc != 0) {
            return rc;
        }
        at::Tensor y = at::matmul(view(in[0]), view(in[1]));
        if (n_in == 3) {
            y = y + view(in[2]);
        }
        return write_out(out[0], y, "linear");
    });
}

int32_t bmm_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                  uint32_t n_out, const rs_attrs* attrs) {
    return guard("bmm", [&] {
        if (n_in != 2 || n_out != 1) {
            return fail("bmm expects two inputs and one output");
        }
        int rc = check_f32(in[0], "bmm", "a");
        if (rc != 0) {
            return rc;
        }
        rc = check_f32(in[1], "bmm", "b");
        if (rc != 0) {
            return rc;
        }
        at::Tensor a = view(in[0]);
        at::Tensor b = view(in[1]);
        at::Tensor c =
            bool_or(attrs, "transpose_b", false) ? at::matmul(a, b.transpose(-1, -2)) : at::matmul(a, b);
        set_shape(out[0], c.sizes());
        return 0;
    });
}

int32_t bmm_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                    uint32_t n_out, const rs_attrs* attrs) {
    return guard("bmm", [&] {
        if (n_in != 2 || n_out != 1) {
            return fail("bmm expects two inputs and one output");
        }
        at::Tensor a = view(in[0]);
        at::Tensor b = view(in[1]);
        at::Tensor c =
            bool_or(attrs, "transpose_b", false) ? at::matmul(a, b.transpose(-1, -2)) : at::matmul(a, b);
        return write_out(out[0], c, "bmm");
    });
}

// ── elementwise_unary ───────────────────────────────────────────────────────

/// The kinds, spelled with the reference's own formulas: `silu` is
/// `x / (1 + e^-x)` (the form that does not produce NaN in either tail) and the
/// `*_grad` kinds are the derivatives of those exact forward forms — an
/// ATen convenience function that computes an algebraically equal but
/// differently rounded expression would still pass a tolerance gate, and would
/// still be a different function to explain.
at::Tensor unary_kind(const std::string& kind, const at::Tensor& x, const char* op,
                      bool* ok) {
    *ok = true;
    if (kind == "silu") {
        return x / (1.0 + (-x).exp());
    }
    if (kind == "gelu") {
        auto c = std::sqrt(2.0 / M_PI) * (x + 0.044715 * x.pow(3));
        return 0.5 * x * (1.0 + c.tanh());
    }
    if (kind == "sigmoid") {
        return 1.0 / (1.0 + (-x).exp());
    }
    if (kind == "tanh") {
        return x.tanh();
    }
    if (kind == "relu") {
        return at::relu(x);
    }
    if (kind == "exp") {
        return x.exp();
    }
    if (kind == "log") {
        return x.log();
    }
    if (kind == "neg") {
        return -x;
    }
    if (kind == "sqrt") {
        return x.sqrt();
    }
    if (kind == "rsqrt") {
        return 1.0 / x.sqrt();
    }
    if (kind == "softplus") {
        // torch's F.softplus(beta=1, threshold=20): the exact identity above
        // the threshold, where ln(1+e^x) would overflow its own argument.
        return at::where(x > 20.0, x, x.exp().log1p());
    }
    if (kind == "negative_exp") {
        return -x.exp();
    }
    if (kind == "silu_grad") {
        auto s = 1.0 / (1.0 + (-x).exp());
        return s * (1.0 + x * (1.0 - s));
    }
    if (kind == "gelu_grad") {
        auto s = std::sqrt(2.0 / M_PI);
        auto c = s * (x + 0.044715 * x.pow(3));
        auto t = c.tanh();
        auto dcdx = s * (1.0 + 3.0 * 0.044715 * x * x);
        return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dcdx;
    }
    if (kind == "sigmoid_grad") {
        auto s = 1.0 / (1.0 + (-x).exp());
        return s * (1.0 - s);
    }
    if (kind == "tanh_grad") {
        auto t = x.tanh();
        return 1.0 - t * t;
    }
    if (kind == "relu_grad") {
        return (x > 0.0).to(x.scalar_type());
    }
    *ok = false;
    fail(std::string(op) + ": unknown kind '" + kind + "'");
    return x;
}

int32_t unary_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                    uint32_t n_out, const rs_attrs* attrs) {
    return guard("elementwise_unary", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("elementwise_unary expects one input and one output");
        }
        int rc = check_f32(in[0], "elementwise_unary", "x");
        if (rc != 0) {
            return rc;
        }
        std::string kind;
        if (!require_kind(attrs, "kind", UNARY_KINDS, N_UNARY_KINDS, "elementwise_unary", &kind)) {
            return 1;
        }
        set_shape(out[0], dims_of(in[0]));
        return 0;
    });
}

int32_t unary_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                      uint32_t n_out, const rs_attrs* attrs) {
    return guard("elementwise_unary", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("elementwise_unary expects one input and one output");
        }
        std::string kind;
        if (!require_kind(attrs, "kind", UNARY_KINDS, N_UNARY_KINDS, "elementwise_unary", &kind)) {
            return 1;
        }
        bool ok = false;
        at::Tensor y = unary_kind(kind, view(in[0]), "elementwise_unary", &ok);
        if (!ok) {
            return 1;
        }
        return write_out(out[0], y, "elementwise_unary");
    });
}

// ── elementwise_binary ──────────────────────────────────────────────────────

int32_t binary_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                     uint32_t n_out, const rs_attrs* attrs) {
    return guard("elementwise_binary", [&] {
        if (n_in < 1 || n_in > 2 || n_out != 1) {
            return fail("elementwise_binary expects one or two inputs and one output");
        }
        int rc = check_f32(in[0], "elementwise_binary", "a");
        if (rc != 0) {
            return rc;
        }
        std::string kind;
        if (!require_kind(attrs, "kind", BINARY_KINDS, N_BINARY_KINDS, "elementwise_binary", &kind)) {
            return 1;
        }
        if (n_in == 1) {
            double rhs = 0.0;
            if (!attr_f64(attrs, "rhs", &rhs)) {
                return fail("elementwise_binary: one input needs the scalar attribute 'rhs'");
            }
            set_shape(out[0], dims_of(in[0]));
            return 0;
        }
        rc = check_f32(in[1], "elementwise_binary", "b");
        if (rc != 0) {
            return rc;
        }
        at::Tensor a = view(in[0]);
        at::Tensor b = view(in[1]);
        set_shape(out[0], at::broadcast_tensors({a, b})[0].sizes());
        return 0;
    });
}

int32_t binary_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                       uint32_t n_out, const rs_attrs* attrs) {
    return guard("elementwise_binary", [&] {
        if (n_in < 1 || n_in > 2 || n_out != 1) {
            return fail("elementwise_binary expects one or two inputs and one output");
        }
        std::string kind;
        if (!require_kind(attrs, "kind", BINARY_KINDS, N_BINARY_KINDS, "elementwise_binary", &kind)) {
            return 1;
        }
        at::Tensor a = view(in[0]);
        at::Tensor b;
        if (n_in == 1) {
            double rhs = 0.0;
            if (!attr_f64(attrs, "rhs", &rhs)) {
                return fail("elementwise_binary: one input needs the scalar attribute 'rhs'");
            }
            b = at::full({}, rhs, a.options());
        } else {
            b = view(in[1]);
        }
        at::Tensor y;
        if (kind == "add") {
            y = a + b;
        } else if (kind == "sub") {
            y = a - b;
        } else if (kind == "mul") {
            y = a * b;
        } else if (kind == "div") {
            y = a / b;
        } else if (kind == "maximum") {
            y = at::maximum(a, b);
        } else {
            y = at::pow(a, b);
        }
        return write_out(out[0], y, "elementwise_binary");
    });
}

// ── compare ─────────────────────────────────────────────────────────────────

/// The vocabulary's mask primitive: 1.0 where the comparison holds, 0.0
/// elsewhere. Every comparison involving NaN is 0.0 — including `ne`, so a NaN
/// can never smuggle a 1 into a mask (the reference's documented policy).
int32_t compare_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                        uint32_t n_out, const rs_attrs* attrs) {
    return guard("compare", [&] {
        if (n_in != 2 || n_out != 1) {
            return fail("compare expects two inputs and one output");
        }
        int rc = check_f32(in[0], "compare", "a");
        if (rc != 0) {
            return rc;
        }
        rc = check_f32(in[1], "compare", "b");
        if (rc != 0) {
            return rc;
        }
        std::string kind;
        if (!require_kind(attrs, "kind", COMPARE_KINDS, N_COMPARE_KINDS, "compare", &kind)) {
            return 1;
        }
        at::Tensor a = view(in[0]);
        at::Tensor b = view(in[1]);
        at::Tensor m;
        if (kind == "eq") {
            m = a == b;
        } else if (kind == "ne") {
            m = a != b;
        } else if (kind == "lt") {
            m = a < b;
        } else if (kind == "le") {
            m = a <= b;
        } else if (kind == "gt") {
            m = a > b;
        } else {
            m = a >= b;
        }
        at::Tensor mask = m.to(a.scalar_type());
        mask = at::where(a.isnan() | b.isnan(), at::zeros_like(mask), mask);
        return write_out(out[0], mask, "compare");
    });
}

// ── reduce ──────────────────────────────────────────────────────────────────

int32_t reduce_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                     uint32_t n_out, const rs_attrs* attrs) {
    return guard("reduce", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("reduce expects one input and one output");
        }
        int rc = check_f32(in[0], "reduce", "x");
        if (rc != 0) {
            return rc;
        }
        std::string kind;
        if (!require_kind(attrs, "kind", REDUCE_KINDS, N_REDUCE_KINDS, "reduce", &kind)) {
            return 1;
        }
        at::Tensor x = view(in[0]);
        bool keepdim = bool_or(attrs, "keepdim", false);
        int64_t axis = 0;
        at::Tensor y;
        if (!attr_i64(attrs, "axis", &axis)) {
            // No axis: reduce everything, and keepdim makes every dim size 1
            // (torch's convention).
            std::vector<int64_t> shape(static_cast<size_t>(x.dim()), 1);
            set_shape(out[0], keepdim ? at::IntArrayRef(shape) : at::IntArrayRef({}));
            return 0;
        }
        int64_t dim = 0;
        if (!resolve_dim(axis, in[0]->rank, "reduce", &dim)) {
            return 1;
        }
        if (kind == "max") {
            y = std::get<0>(at::max(x, dim, keepdim));
        } else if (kind == "amax") {
            y = x.abs().amax(dim, keepdim);
        } else if (kind == "mean") {
            y = x.mean(dim, keepdim);
        } else {
            y = x.sum(dim, keepdim);
        }
        set_shape(out[0], y.sizes());
        return 0;
    });
}

int32_t reduce_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                       uint32_t n_out, const rs_attrs* attrs) {
    return guard("reduce", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("reduce expects one input and one output");
        }
        std::string kind;
        if (!require_kind(attrs, "kind", REDUCE_KINDS, N_REDUCE_KINDS, "reduce", &kind)) {
            return 1;
        }
        at::Tensor x = view(in[0]);
        bool keepdim = bool_or(attrs, "keepdim", false);
        int64_t axis = 0;
        at::Tensor y;
        if (!attr_i64(attrs, "axis", &axis)) {
            // No axis: reduce everything. keepdim = true makes every dim size 1
            // (torch's convention, which the softmax/layernorm VJPs rely on).
            if (kind == "max") {
                y = at::max(x);
            } else if (kind == "amax") {
                y = x.abs().max();
            } else if (kind == "mean") {
                y = x.mean();
            } else {
                y = x.sum();
            }
            if (keepdim) {
                y = y.reshape(std::vector<int64_t>(x.dim(), 1));
            }
        } else {
            int64_t dim = 0;
            if (!resolve_dim(axis, in[0]->rank, "reduce", &dim)) {
                return 1;
            }
            if (kind == "max") {
                y = std::get<0>(at::max(x, dim, keepdim));
            } else if (kind == "amax") {
                y = x.abs().amax(dim, keepdim);
            } else if (kind == "mean") {
                y = x.mean(dim, keepdim);
            } else {
                y = x.sum(dim, keepdim);
            }
        }
        return write_out(out[0], y, "reduce");
    });
}

// ── softmax ─────────────────────────────────────────────────────────────────

int32_t softmax_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                        uint32_t n_out, const rs_attrs* attrs) {
    return guard("softmax", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("softmax expects one input and one output");
        }
        int rc = check_f32(in[0], "softmax", "x");
        if (rc != 0) {
            return rc;
        }
        int64_t dim = 0;
        if (!resolve_dim(i64_or(attrs, "axis", -1), in[0]->rank, "softmax", &dim)) {
            return 1;
        }
        double scale = f64_or(attrs, "scale", 1.0);
        at::Tensor x = view(in[0]);
        if (scale != 1.0) {
            x = x * scale;
        }
        return write_out(out[0], at::softmax(x, dim), "softmax");
    });
}

// ── embedding / gather / scatter ────────────────────────────────────────────

bool index_dtype_ok(const rs_tensor* t, const char* op, const char* who) {
    if (t->dtype == RS_I32 || t->dtype == RS_I64) {
        return true;
    }
    fail(std::string(op) + ": " + who + " must be i32 or i64");
    return false;
}

int32_t embedding_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                        uint32_t n_out, const rs_attrs*) {
    return guard("embedding", [&] {
        if (n_in != 2 || n_out != 1) {
            return fail("embedding expects two inputs and one output");
        }
        int rc = check_f32(in[0], "embedding", "w");
        if (rc != 0) {
            return rc;
        }
        if (!index_dtype_ok(in[1], "embedding", "indices")) {
            return 1;
        }
        if (in[0]->rank != 2) {
            return fail("embedding expects weight [V, D]");
        }
        std::vector<int64_t> shape = dims_of(in[1]);
        shape.push_back(in[0]->shape[1]);
        set_shape(out[0], shape);
        return 0;
    });
}

int32_t embedding_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in,
                          rs_tensor* const* out, uint32_t n_out, const rs_attrs*) {
    return guard("embedding", [&] {
        if (n_in != 2 || n_out != 1) {
            return fail("embedding expects two inputs and one output");
        }
        at::Tensor idx = as_indices(in[1]);
        // No wrap-around: a negative index is a description bug, not a request
        // to read the last row.
        if (idx.numel() > 0 && at::min(idx).item<int64_t>() < 0) {
            return fail("embedding: negative index (no wrap-around)");
        }
        return write_out(out[0], at::embedding(view(in[0]), idx), "embedding");
    });
}

int32_t gather_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                     uint32_t n_out, const rs_attrs*) {
    return guard("gather", [&] {
        if (n_in != 2 || n_out != 1) {
            return fail("gather expects two inputs and one output");
        }
        int rc = check_f32(in[0], "gather", "x");
        if (rc != 0) {
            return rc;
        }
        if (!index_dtype_ok(in[1], "gather", "indices")) {
            return 1;
        }
        set_shape(out[0], dims_of(in[1]));
        return 0;
    });
}

int32_t gather_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in,
                       rs_tensor* const* out, uint32_t n_out, const rs_attrs* attrs) {
    return guard("gather", [&] {
        if (n_in != 2 || n_out != 1) {
            return fail("gather expects two inputs and one output");
        }
        int64_t dim = 0;
        if (!resolve_dim(i64_or(attrs, "axis", -1), in[0]->rank, "gather", &dim)) {
            return 1;
        }
        at::Tensor idx = as_indices(in[1]);
        if (idx.numel() > 0 && at::min(idx).item<int64_t>() < 0) {
            return fail("gather: negative index (no wrap-around)");
        }
        return write_out(out[0], at::gather(view(in[0]), dim, idx), "gather");
    });
}

int32_t scatter_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                      uint32_t n_out, const rs_attrs* attrs) {
    return guard("scatter", [&] {
        if (n_in != 3 || n_out != 1) {
            return fail("scatter expects three inputs and one output");
        }
        int rc = check_f32(in[0], "scatter", "x");
        if (rc != 0) {
            return rc;
        }
        if (!index_dtype_ok(in[1], "scatter", "indices")) {
            return 1;
        }
        rc = check_f32(in[2], "scatter", "values");
        if (rc != 0) {
            return rc;
        }
        std::string reduce;
        if (!require_kind(attrs, "reduce", SCATTER_REDUCE_KINDS, N_SCATTER_REDUCE_KINDS,
                          "scatter", &reduce)) {
            return 1;
        }
        set_shape(out[0], dims_of(in[0]));
        return 0;
    });
}

int32_t scatter_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                        uint32_t n_out, const rs_attrs* attrs) {
    return guard("scatter", [&] {
        if (n_in != 3 || n_out != 1) {
            return fail("scatter expects three inputs and one output");
        }
        int64_t dim = 0;
        if (!resolve_dim(i64_or(attrs, "axis", -1), in[0]->rank, "scatter", &dim)) {
            return 1;
        }
        std::string reduce;
        if (!require_kind(attrs, "reduce", SCATTER_REDUCE_KINDS, N_SCATTER_REDUCE_KINDS,
                          "scatter", &reduce)) {
            return 1;
        }
        at::Tensor x = view(in[0]);
        at::Tensor idx = as_indices(in[1]);
        at::Tensor values = view(in[2]);
        at::Tensor y = reduce == "add" ? at::scatter_add(x, dim, idx, values)
                                       : at::scatter(x, dim, idx, values);
        return write_out(out[0], y, "scatter");
    });
}

// ── sdpa ────────────────────────────────────────────────────────────────────

struct SdpaPlan {
    int64_t s = 0;
    int64_t t = 0;
    int64_t heads = 1;
    int64_t kv_heads = 1;
    int64_t head_dim = 0;
    int64_t value_dim = 0;
    bool headed = false;
    double scale = 1.0;
};

bool sdpa_plan(const rs_tensor* q, const rs_tensor* k, const rs_tensor* v, const rs_attrs* attrs,
               const char* op, SdpaPlan* plan) {
    if (q->rank != k->rank || q->rank != v->rank || q->rank < 2) {
        fail(std::string(op) + ": q, k and v must have the same rank >= 2");
        return false;
    }
    bool headed = find_attr(attrs, "num_heads", RS_ATTR_I64) != nullptr;
    plan->headed = headed;
    if (headed) {
        if (q->rank < 3) {
            fail(std::string(op) + ": the per-head form needs rank >= 3");
            return false;
        }
        plan->heads = 1;
        attr_i64(attrs, "num_heads", &plan->heads);
        plan->kv_heads = plan->heads;
        attr_i64(attrs, "num_kv_heads", &plan->kv_heads);
        plan->s = q->shape[q->rank - 3];
        plan->head_dim = q->shape[q->rank - 1];
        plan->t = k->shape[k->rank - 3];
        plan->value_dim = v->shape[v->rank - 1];
        if (plan->heads % plan->kv_heads != 0) {
            fail(std::string(op) + ": num_heads must be a multiple of num_kv_heads");
            return false;
        }
        plan->scale = f64_or(attrs, "scale", 1.0 / std::sqrt(static_cast<double>(plan->head_dim)));
    } else {
        plan->s = q->shape[q->rank - 2];
        plan->t = k->shape[k->rank - 2];
        plan->head_dim = q->shape[q->rank - 1];
        plan->value_dim = v->shape[v->rank - 1];
        plan->scale = f64_or(attrs, "scale", 1.0);
    }
    return true;
}

int32_t sdpa_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                   uint32_t n_out, const rs_attrs* attrs) {
    return guard("sdpa", [&] {
        if (n_in < 3 || n_in > 4 || n_out != 1) {
            return fail("sdpa expects three or four inputs and one output");
        }
        for (uint32_t i = 0; i < 3; ++i) {
            int rc = check_f32(in[i], "sdpa", "q/k/v");
            if (rc != 0) {
                return rc;
            }
        }
        SdpaPlan plan;
        if (!sdpa_plan(in[0], in[1], in[2], attrs, "sdpa", &plan)) {
            return 1;
        }
        std::vector<int64_t> shape = dims_of(in[0]);
        if (plan.headed) {
            shape[shape.size() - 2] = plan.heads;
            shape[shape.size() - 1] = plan.value_dim;
        } else {
            shape[shape.size() - 1] = plan.value_dim;
        }
        set_shape(out[0], shape);
        return 0;
    });
}

int32_t sdpa_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                     uint32_t n_out, const rs_attrs* attrs) {
    return guard("sdpa", [&] {
        if (n_in < 3 || n_in > 4 || n_out != 1) {
            return fail("sdpa expects three or four inputs and one output");
        }
        SdpaPlan plan;
        if (!sdpa_plan(in[0], in[1], in[2], attrs, "sdpa", &plan)) {
            return 1;
        }
        at::Tensor q = view(in[0]);
        at::Tensor k = view(in[1]);
        at::Tensor v = view(in[2]);
        const int64_t S = plan.s;
        const int64_t T = plan.t;
        int64_t batch = 1;
        for (int64_t i = 0; i + (plan.headed ? 3 : 2) < q.dim(); ++i) {
            batch *= q.size(i);
        }
        at::Tensor qh, kh, vh;
        if (plan.headed) {
            // [.., S, H, D] -> [B, H, S, D]: the plan's layout is
            // sequence-first with the head axis inside it, torch's is
            // batch-first with the head axis outside.
            qh = q.reshape({batch, S, plan.heads, plan.head_dim}).permute({0, 2, 1, 3});
            kh = k.reshape({batch, T, plan.kv_heads, plan.head_dim}).permute({0, 2, 1, 3});
            vh = v.reshape({batch, T, plan.kv_heads, plan.value_dim}).permute({0, 2, 1, 3});
            if (plan.kv_heads != plan.heads) {
                int64_t repeat = plan.heads / plan.kv_heads;
                kh = kh.repeat_interleave(repeat, /*dim=*/1);
                vh = vh.repeat_interleave(repeat, /*dim=*/1);
            }
        } else {
            qh = q.reshape({batch, S, 1, plan.head_dim}).permute({0, 2, 1, 3});
            kh = k.reshape({batch, T, 1, plan.head_dim}).permute({0, 2, 1, 3});
            vh = v.reshape({batch, T, 1, plan.value_dim}).permute({0, 2, 1, 3});
        }

        // The additive mask is applied here rather than handed to the fused
        // kernel: the causal triangle and the declared mask are combined once,
        // so the same code path runs whatever backend SDPA picks.
        bool causal = bool_or(attrs, "causal", false);
        std::optional<at::Tensor> mask;
        if (n_in == 4 || causal) {
            at::Tensor additive =
                n_in == 4 ? view(in[3]).expand({batch, S, T}).reshape({batch, 1, S, T})
                          : at::zeros({batch, 1, S, T}, q.options());
            if (causal) {
                additive = additive +
                           at::full({S, T}, -std::numeric_limits<double>::infinity(), q.options())
                               .triu(/*diagonal=*/1);
            }
            mask = additive;
        }

        at::Tensor o =
            at::scaled_dot_product_attention(qh, kh, vh, mask, 0.0, false, plan.scale);
        o = o.permute({0, 2, 1, 3});  // [B, H, S, Dv] -> [B, S, H, Dv]
        return write_out(out[0], o.reshape(dims_of(out[0])), "sdpa");
    });
}

// ── rope ────────────────────────────────────────────────────────────────────

/// Builds the [S, h] cos/sin tables: inv_freq[j] = theta^(-2j/rotary_dim), so
/// the rotation is the `compute_default_rope_parameters` convention (f32, the
/// same as the HF source).
bool rope_tables(const rs_tensor* pos_tensor, double theta, int64_t S, int64_t h, int64_t rotary,
                 const at::TensorOptions& options, at::Tensor* cos, at::Tensor* sin,
                 const char* op) {
    at::Tensor inv_freq = at::empty({h}, options);
    {
        std::vector<float> host(static_cast<size_t>(h));
        for (int64_t j = 0; j < h; ++j) {
            host[static_cast<size_t>(j)] = static_cast<float>(
                std::pow(theta, -static_cast<double>(2 * j) / static_cast<double>(rotary)));
        }
        inv_freq = at::from_blob(host.data(), {h}, at::TensorOptions().dtype(at::kFloat)).clone();
        inv_freq = inv_freq.to(options.device());
    }
    at::Tensor positions;
    if (pos_tensor != nullptr) {
        if (numel_of(pos_tensor) != S) {
            fail(std::string(op) + ": the position tensor must hold one entry per position");
            return false;
        }
        positions = view(pos_tensor).to(at::kFloat).reshape({S});
    } else {
        positions = at::arange(static_cast<double>(S), options).to(at::kFloat);
    }
    at::Tensor angles = positions.reshape({S, 1}) * inv_freq.reshape({1, h});
    *cos = angles.cos();
    *sin = angles.sin();
    return true;
}

int32_t rope_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                   uint32_t n_out, const rs_attrs* attrs) {
    return guard("rope", [&] {
        if (n_in < 2 || n_in > 3 || n_out != 2) {
            return fail("rope expects two or three inputs and two outputs");
        }
        int rc = check_f32(in[0], "rope", "x");
        if (rc != 0) {
            return rc;
        }
        rc = check_f32(in[1], "rope", "y");
        if (rc != 0) {
            return rc;
        }
        if (in[0]->rank < 2 || in[0]->rank != in[1]->rank) {
            return fail("rope expects rank >= 2 inputs of identical rank");
        }
        for (uint32_t d = 0; d < in[0]->rank; ++d) {
            if (in[0]->shape[d] != in[1]->shape[d]) {
                return fail("rope expects x and y with identical shape");
            }
        }
        set_shape(out[0], dims_of(in[0]));
        set_shape(out[1], dims_of(in[1]));
        return 0;
    });
}

bool rope_plan(const rs_tensor* x, const rs_attrs* attrs, int64_t* s, int64_t* d, int64_t* rotary,
               const char* op) {
    *s = x->shape[0];
    *d = x->shape[x->rank - 1];
    *rotary = i64_or(attrs, "rotary_dim", *d);
    if (*rotary <= 0 || *rotary > *d || *rotary % 2 != 0) {
        fail(std::string(op) + ": 'rotary_dim' must be a positive even number <= D");
        return false;
    }
    bool partial = bool_or(attrs, "partial_rotary", false);
    if (!partial && *rotary != *d) {
        fail(std::string(op) +
             ": 'rotary_dim' is smaller than D but 'partial_rotary' is not declared");
        return false;
    }
    return true;
}

/// One operand's rotation: half-split pairs (i, i+h) over the first `rotary`
/// dims of the last axis, the remaining dims copied through.
at::Tensor rope_apply(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin, int64_t S,
                      int64_t D, int64_t rotary) {
    int64_t h = rotary / 2;
    at::Tensor flat = x.reshape({-1, S, D});
    if (!flat.is_contiguous()) {
        flat = flat.contiguous();
    }
    at::Tensor cos_b = cos.reshape({1, S, h});
    at::Tensor sin_b = sin.reshape({1, S, h});
    at::Tensor first = flat.narrow(2, 0, h);
    at::Tensor second = flat.narrow(2, h, h);
    at::Tensor rotated =
        at::cat({first * cos_b - second * sin_b, second * cos_b + first * sin_b}, 2);
    if (rotary < D) {
        return at::cat({rotated, flat.narrow(2, rotary, D - rotary)}, 2).reshape(x.sizes());
    }
    return rotated.reshape(x.sizes());
}

int32_t rope_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                     uint32_t n_out, const rs_attrs* attrs) {
    return guard("rope", [&] {
        if (n_in < 2 || n_in > 3 || n_out != 2) {
            return fail("rope expects two or three inputs and two outputs");
        }
        int64_t S = 0, D = 0, rotary = 0;
        if (!rope_plan(in[0], attrs, &S, &D, &rotary, "rope")) {
            return 1;
        }
        at::Tensor x = view(in[0]);
        at::Tensor cos, sin;
        if (!rope_tables(n_in == 3 ? in[2] : nullptr, f64_or(attrs, "theta", 1e7), S, rotary / 2,
                         rotary, x.options(), &cos, &sin, "rope")) {
            return 1;
        }
        int rc = write_out(out[0], rope_apply(x, cos, sin, S, D, rotary), "rope");
        if (rc != 0) {
            return rc;
        }
        return write_out(out[1], rope_apply(view(in[1]), cos, sin, S, D, rotary), "rope");
    });
}

// ── topk_router ─────────────────────────────────────────────────────────────

int32_t topk_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                   uint32_t n_out, const rs_attrs* attrs) {
    return guard("topk_router", [&] {
        if (n_in != 1 || n_out != 2) {
            return fail("topk_router expects one input and two outputs");
        }
        int rc = check_f32(in[0], "topk_router", "logits");
        if (rc != 0) {
            return rc;
        }
        if (in[0]->rank != 2) {
            return fail("topk_router expects logits [N, E]");
        }
        int64_t e = in[0]->shape[1];
        int64_t k = i64_or(attrs, "top_k", 2);
        if (k < 1 || k > e) {
            return fail("topk_router: 'top_k' must be in [1, E]");
        }
        set_shape(out[0], {in[0]->shape[0], k});
        set_shape(out[1], {in[0]->shape[0], k});
        out[1]->dtype = RS_I32;
        return 0;
    });
}

int32_t topk_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                     uint32_t n_out, const rs_attrs* attrs) {
    return guard("topk_router", [&] {
        if (n_in != 1 || n_out != 2) {
            return fail("topk_router expects one input and two outputs");
        }
        at::Tensor logits = view(in[0]);
        int64_t k = i64_or(attrs, "top_k", 2);
        int64_t e = logits.size(1);
        if (k < 1 || k > e) {
            return fail("topk_router: 'top_k' must be in [1, E]");
        }
        at::Tensor probs = at::softmax(logits, /*dim=*/-1);
        // Ties break toward the lower expert index: a stable descending sort
        // keeps the original order among equal probabilities, and ascending
        // selection on top of it is exactly that rule. `at::topk`'s tie
        // behaviour is not part of its contract.
        auto sorted = at::sort(probs, /*dim=*/-1, /*descending=*/true, /*stable=*/true);
        at::Tensor values = std::get<0>(sorted).narrow(1, 0, k);
        at::Tensor indices = std::get<1>(sorted).narrow(1, 0, k);
        if (bool_or(attrs, "norm_topk_prob", false)) {
            values = values / values.sum(/*dim=*/-1, /*keepdim=*/true);
        }
        int rc = write_out(out[1], indices.to(at::kInt), "topk_router");
        if (rc != 0) {
            return rc;
        }
        return write_out(out[0], values, "topk_router");
    });
}

// ── cross_entropy ───────────────────────────────────────────────────────────

int32_t ce_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                   uint32_t n_out, const rs_attrs*) {
    return guard("cross_entropy", [&] {
        if (n_in != 2 || n_out != 1) {
            return fail("cross_entropy expects two inputs and one output");
        }
        at::Tensor logits = view(in[0]);
        at::Tensor targets = as_indices(in[1]);
        at::Tensor picked = logits.gather(1, targets.reshape({-1, 1})).reshape({-1});
        at::Tensor loss = (at::logsumexp(logits, /*dim=*/1) - picked).mean();
        return write_out(out[0], loss, "cross_entropy");
    });
}

int32_t ce_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out, uint32_t n_out,
                 const rs_attrs*) {
    return guard("cross_entropy", [&] {
        if (n_in != 2 || n_out != 1) {
            return fail("cross_entropy expects two inputs and one output");
        }
        int rc = check_f32(in[0], "cross_entropy", "logits");
        if (rc != 0) {
            return rc;
        }
        if (!index_dtype_ok(in[1], "cross_entropy", "targets")) {
            return 1;
        }
        if (in[0]->rank != 2 || in[1]->rank != 1 || in[1]->shape[0] != in[0]->shape[0]) {
            return fail("cross_entropy expects logits [N, C] and targets [N]");
        }
        set_shape(out[0], {});
        return 0;
    });
}

}  // namespace

void add_compute_ops(std::vector<OpDef>& ops) {
    ops.push_back(OpDef{"matmul",
                        "C = A@B, f32 accumulate; 'transpose_b' (default false) computes A @ "
                        "B^T with B as [N, K]. The attribute is declared, never inferred.",
                        f32_mask(), RS_AUTODIFF, matmul_infer, matmul_execute});
    ops.push_back(OpDef{"linear",
                        "y = x @ w (+ b): w is [K, N] (output features last), x is [..., K], "
                        "bias is [N], y is [..., N].",
                        f32_mask(), RS_AUTODIFF, linear_infer, linear_execute});
    ops.push_back(OpDef{"bmm",
                        "Batched matmul over identical batch dims, with the same optional "
                        "'transpose_b' convention as matmul.",
                        f32_mask(), RS_AUTODIFF, bmm_infer, bmm_execute});
    ops.push_back(OpDef{"elementwise_unary",
                        "Applies the required 'kind' attribute elementwise (silu, gelu, "
                        "sigmoid, tanh, relu, exp, log, neg, sqrt, rsqrt, softplus, "
                        "negative_exp and the five *_grad kinds), with the reference's exact "
                        "formulas: silu = x/(1+e^-x), softplus = ln(1+e^x) with the identity "
                        "above 20, gelu = the tanh approximation.",
                        f32_mask(), RS_AUTODIFF, unary_infer, unary_execute});
    ops.push_back(OpDef{"elementwise_binary",
                        "Applies the required 'kind' attribute (add, sub, mul, div, maximum, "
                        "pow) with right-aligned broadcasting; a single input plus the scalar "
                        "'rhs' attribute is the same operation against a constant.",
                        f32_mask(), RS_AUTODIFF, binary_infer, binary_execute});
    ops.push_back(OpDef{"compare",
                        "Elementwise comparison to an f32 mask (1.0 where it holds, 0.0 "
                        "elsewhere). Every comparison involving NaN yields 0.0, 'ne' "
                        "included.",
                        f32_mask(), RS_AUTODIFF, [](const rs_tensor* const* in, uint32_t n_in,
                                                     rs_tensor* const* out, uint32_t n_out,
                                                     const rs_attrs* attrs) {
                            return guard("compare", [&] {
                                if (n_in != 2 || n_out != 1) {
                                    return fail("compare expects two inputs and one output");
                                }
                                int rc = check_f32(in[0], "compare", "a");
                                if (rc != 0) {
                                    return rc;
                                }
                                rc = check_f32(in[1], "compare", "b");
                                if (rc != 0) {
                                    return rc;
                                }
                                std::string kind;
                                if (!require_kind(attrs, "kind", COMPARE_KINDS, N_COMPARE_KINDS,
                                                  "compare", &kind)) {
                                    return 1;
                                }
                                set_shape(out[0], dims_of(in[0]));
                                return 0;
                            });
                        },
                        compare_execute});
    ops.push_back(OpDef{"reduce",
                        "Reduces along 'axis' with 'kind' (sum, mean, max, amax); no axis "
                        "reduces everything to a scalar, and 'keepdim' keeps the reduced "
                        "dims as size 1.",
                        f32_mask(), RS_AUTODIFF, reduce_infer, reduce_execute});
    ops.push_back(OpDef{"softmax",
                        "Stable softmax of x*'scale' over 'axis' (default -1), shape "
                        "preserved.",
                        f32_mask(), RS_AUTODIFF,
                        [](const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                           uint32_t n_out, const rs_attrs* attrs) {
                            return guard("softmax", [&] {
                                if (n_in != 1 || n_out != 1) {
                                    return fail("softmax expects one input and one output");
                                }
                                int rc = check_f32(in[0], "softmax", "x");
                                if (rc != 0) {
                                    return rc;
                                }
                                int64_t dim = 0;
                                if (!resolve_dim(i64_or(attrs, "axis", -1), in[0]->rank, "softmax",
                                                 &dim)) {
                                    return 1;
                                }
                                (void)f64_or(attrs, "scale", 1.0);
                                set_shape(out[0], dims_of(in[0]));
                                return 0;
                            });
                        },
                        softmax_execute});
    ops.push_back(OpDef{"embedding",
                        "out = w[indices]: weight [V, D], indices i32/i64 of any rank, output "
                        "indices.shape + [D]. Negative indices are rejected.",
                        (1u << RS_F32) | (1u << RS_I32) | (1u << RS_I64), RS_AUTODIFF,
                        embedding_infer, embedding_execute});
    ops.push_back(OpDef{"gather",
                        "Torch-gather convention along 'axis' (default -1); the output takes "
                        "the indices' shape and negative indices are rejected.",
                        (1u << RS_F32) | (1u << RS_I32) | (1u << RS_I64), RS_AUTODIFF,
                        gather_infer, gather_execute});
    ops.push_back(OpDef{"scatter",
                        "Copy of x with values written at the indices along 'axis' (default "
                        "-1); 'reduce' selects assign or add.",
                        (1u << RS_F32) | (1u << RS_I32) | (1u << RS_I64), RS_AUTODIFF,
                        scatter_infer, scatter_execute});
    ops.push_back(OpDef{"sdpa",
                        "o = softmax(q @ k^T * scale + mask) @ v, per-head form with "
                        "'num_heads'/'num_kv_heads' (GQA) or the legacy flat form; 'causal' "
                        "masks j > i and an optional additive mask is added before the "
                        "softmax. Executed by ATen's fused kernels.",
                        (1u << RS_F32), RS_AUTODIFF, sdpa_infer, sdpa_execute});
    ops.push_back(OpDef{"rope",
                        "Rotary embedding of two operands at once: half-split pairs (i, i+h) "
                        "over the first 'rotary_dim' dims of the last axis, positions along "
                        "axis 0, inv_freq[j] = theta^(-2j/rotary_dim).",
                        (1u << RS_F32) | (1u << RS_I32) | (1u << RS_I64), RS_AUTODIFF,
                        rope_infer, rope_execute});
    ops.push_back(OpDef{"topk_router",
                        "(weights, indices) = top-k of softmax(logits [N, E]) per row, with "
                        "'top_k' and 'norm_topk_prob'. Ties break toward the lower expert "
                        "index; indices are i32.",
                        f32_mask(), RS_AUTODIFF, topk_infer, topk_execute});
    ops.push_back(OpDef{"cross_entropy",
                        "Mean cross-entropy of logits [N, C] against targets [N], computed in "
                        "the stable log-sum-exp form.",
                        (1u << RS_F32) | (1u << RS_I32) | (1u << RS_I64), RS_NONDIFF, ce_infer,
                        ce_execute});
}

}  // namespace rsaten
