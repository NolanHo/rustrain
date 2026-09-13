// Movement operators: the zero-copy views and the one copying concatenation.
//
// A view operator's output memory IS its input's memory. This plugin reproduces
// that by returning the ATen view it computed and letting `adopt` hand the
// result's pointer, shape and strides back to the executor — an executor that
// assumed the output lives in the buffer it allocated for the slot would read
// uninitialised memory (docs/architecture.md's view-alias rule).
#include "common.h"

namespace rsaten {
namespace {

// ── view ────────────────────────────────────────────────────────────────────

int32_t view_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                   uint32_t n_out, const rs_attrs*) {
    return guard("view", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("view expects one input and one output");
        }
        int rc = check_f32(in[0], "view", "input");
        if (rc != 0) {
            return rc;
        }
        set_shape(out[0], dims_of(in[0]));
        return 0;
    });
}

int32_t view_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                     uint32_t n_out, const rs_attrs*) {
    return guard("view", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("view expects one input and one output");
        }
        adopt(out[0], view(in[0]));
        return 0;
    });
}

// ── reshape ─────────────────────────────────────────────────────────────────

/// Resolves the `shape` attribute against the input's element count. One `-1`
/// is allowed and filled; anything else must multiply out exactly.
bool reshape_target(const rs_tensor* x, const rs_attrs* attrs, std::vector<int64_t>* target) {
    if (!attr_i64s(attrs, "shape", target)) {
        fail("reshape: attribute 'shape' is required");
        return false;
    }
    int64_t numel = 1;
    for (uint32_t i = 0; i < x->rank; ++i) {
        numel *= x->shape[i];
    }
    int64_t wildcard = -1;
    int64_t known = 1;
    for (size_t i = 0; i < target->size(); ++i) {
        int64_t d = (*target)[i];
        if (d == -1) {
            if (wildcard >= 0) {
                fail("reshape: at most one -1 is allowed in 'shape'");
                return false;
            }
            wildcard = static_cast<int64_t>(i);
        } else if (d < 0) {
            fail("reshape: negative dimension " + std::to_string(d) + " in 'shape'");
            return false;
        } else {
            known *= d;
        }
    }
    if (wildcard >= 0) {
        if (known == 0 || numel % known != 0) {
            fail("reshape: cannot infer the -1 dimension: " + std::to_string(numel) +
                 " elements do not divide " + std::to_string(known));
            return false;
        }
        (*target)[static_cast<size_t>(wildcard)] = numel / known;
    } else if (known != numel) {
        fail("reshape: 'shape' holds " + std::to_string(known) + " elements, the input has " +
             std::to_string(numel));
        return false;
    }
    return true;
}

int32_t reshape_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                      uint32_t n_out, const rs_attrs* attrs) {
    return guard("reshape", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("reshape expects one input and one output");
        }
        int rc = check_f32(in[0], "reshape", "input");
        if (rc != 0) {
            return rc;
        }
        std::vector<int64_t> target;
        if (!reshape_target(in[0], attrs, &target)) {
            return 1;
        }
        set_shape(out[0], target);
        return 0;
    });
}

int32_t reshape_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                        uint32_t n_out, const rs_attrs* attrs) {
    return guard("reshape", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("reshape expects one input and one output");
        }
        at::Tensor x = view(in[0]);
        std::vector<int64_t> target;
        if (!reshape_target(in[0], attrs, &target)) {
            return 1;
        }
        at::Tensor reshaped = x.reshape(target);
        // `reshape` means the values in row-major LOGICAL order, read as
        // `target`. A contiguous input is a view of the framework's buffer and
        // the executor adopts the pointer; a strided input (a `narrow` of a
        // flat projection, say) makes ATen allocate, and that tensor dies with
        // this call — so its values are copied into the plan's output slot
        // instead of a dangling pointer being handed back.
        if (x.is_contiguous()) {
            adopt(out[0], reshaped);
            return 0;
        }
        return write_out(out[0], reshaped, "reshape");
    });
}

// ── transpose ───────────────────────────────────────────────────────────────

int32_t transpose_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                        uint32_t n_out, const rs_attrs* attrs) {
    return guard("transpose", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("transpose expects one input and one output");
        }
        int rc = check_f32(in[0], "transpose", "input");
        if (rc != 0) {
            return rc;
        }
        int64_t a = 0, b = 0;
        if (!resolve_dim(i64_or(attrs, "dim0", -2), in[0]->rank, "transpose", &a) ||
            !resolve_dim(i64_or(attrs, "dim1", -1), in[0]->rank, "transpose", &b)) {
            return 1;
        }
        std::vector<int64_t> shape = dims_of(in[0]);
        std::swap(shape[static_cast<size_t>(a)], shape[static_cast<size_t>(b)]);
        set_shape(out[0], shape);
        return 0;
    });
}

int32_t transpose_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in,
                          rs_tensor* const* out, uint32_t n_out, const rs_attrs* attrs) {
    return guard("transpose", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("transpose expects one input and one output");
        }
        int64_t a = 0, b = 0;
        if (!resolve_dim(i64_or(attrs, "dim0", -2), in[0]->rank, "transpose", &a) ||
            !resolve_dim(i64_or(attrs, "dim1", -1), in[0]->rank, "transpose", &b)) {
            return 1;
        }
        adopt(out[0], view(in[0]).transpose(a, b));
        return 0;
    });
}

// ── narrow ──────────────────────────────────────────────────────────────────

bool narrow_window(const rs_tensor* x, const rs_attrs* attrs, int64_t* dim, int64_t* start,
                   int64_t* length, const char* op) {
    if (!resolve_dim(i64_or(attrs, "dim", -1), x->rank, op, dim)) {
        return false;
    }
    int64_t total = x->shape[*dim];
    *start = i64_or(attrs, "start", 0);
    *length = i64_or(attrs, "length", total - *start);
    if (*start < 0 || *length < 0 || *start + *length > total) {
        fail(std::string(op) + ": window [" + std::to_string(*start) + ", " +
             std::to_string(*start + *length) + ") does not fit axis " + std::to_string(*dim) +
             " of size " + std::to_string(total));
        return false;
    }
    return true;
}

int32_t narrow_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                     uint32_t n_out, const rs_attrs* attrs) {
    return guard("narrow", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("narrow expects one input and one output");
        }
        int rc = check_f32(in[0], "narrow", "input");
        if (rc != 0) {
            return rc;
        }
        int64_t dim = 0, start = 0, length = 0;
        if (!narrow_window(in[0], attrs, &dim, &start, &length, "narrow")) {
            return 1;
        }
        std::vector<int64_t> shape = dims_of(in[0]);
        shape[static_cast<size_t>(dim)] = length;
        set_shape(out[0], shape);
        return 0;
    });
}

int32_t narrow_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                       uint32_t n_out, const rs_attrs* attrs) {
    return guard("narrow", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("narrow expects one input and one output");
        }
        int64_t dim = 0, start = 0, length = 0;
        if (!narrow_window(in[0], attrs, &dim, &start, &length, "narrow")) {
            return 1;
        }
        adopt(out[0], view(in[0]).narrow(dim, start, length));
        return 0;
    });
}

// ── cat ─────────────────────────────────────────────────────────────────────

int32_t cat_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                  uint32_t n_out, const rs_attrs* attrs) {
    return guard("cat", [&] {
        if (n_in < 1 || n_out != 1) {
            return fail("cat expects at least one input and one output");
        }
        int64_t dim = 0;
        if (!resolve_dim(i64_or(attrs, "dim", -1), in[0]->rank, "cat", &dim)) {
            return 1;
        }
        std::vector<int64_t> shape = dims_of(in[0]);
        int64_t total = 0;
        for (uint32_t i = 0; i < n_in; ++i) {
            int rc = check_f32(in[i], "cat", "input");
            if (rc != 0) {
                return rc;
            }
            if (in[i]->rank != in[0]->rank) {
                return fail("cat: inputs have different ranks");
            }
            for (uint32_t d = 0; d < in[0]->rank; ++d) {
                if (static_cast<int64_t>(d) != dim && in[i]->shape[d] != in[0]->shape[d]) {
                    return fail("cat: inputs disagree on axis " + std::to_string(d) +
                                ", which is not the concatenation axis");
                }
            }
            total += in[i]->shape[dim];
        }
        shape[static_cast<size_t>(dim)] = total;
        set_shape(out[0], shape);
        return 0;
    });
}

int32_t cat_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                    uint32_t n_out, const rs_attrs* attrs) {
    return guard("cat", [&] {
        if (n_in < 1 || n_out != 1) {
            return fail("cat expects at least one input and one output");
        }
        int64_t dim = 0;
        if (!resolve_dim(i64_or(attrs, "dim", -1), in[0]->rank, "cat", &dim)) {
            return 1;
        }
        std::vector<at::Tensor> parts;
        parts.reserve(n_in);
        for (uint32_t i = 0; i < n_in; ++i) {
            parts.push_back(view(in[i]));
        }
        return write_out(out[0], at::cat(parts, dim), "cat");
    });
}

// ── broadcast ───────────────────────────────────────────────────────────────

int32_t broadcast_infer(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                        uint32_t n_out, const rs_attrs* attrs) {
    return guard("broadcast", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("broadcast expects one input and one output");
        }
        int rc = check_f32(in[0], "broadcast", "input");
        if (rc != 0) {
            return rc;
        }
        std::vector<int64_t> shape;
        if (!attr_i64s(attrs, "shape", &shape)) {
            return fail("broadcast: attribute 'shape' is required");
        }
        set_shape(out[0], shape);
        return 0;
    });
}

int32_t broadcast_execute(rs_ctx*, const rs_tensor* const* in, uint32_t n_in,
                          rs_tensor* const* out, uint32_t n_out, const rs_attrs* attrs) {
    return guard("broadcast", [&] {
        if (n_in != 1 || n_out != 1) {
            return fail("broadcast expects one input and one output");
        }
        std::vector<int64_t> shape;
        if (!attr_i64s(attrs, "shape", &shape)) {
            return fail("broadcast: attribute 'shape' is required");
        }
        adopt(out[0], view(in[0]).expand(shape));
        return 0;
    });
}

}  // namespace

void add_meta_ops(std::vector<OpDef>& ops) {
    ops.push_back(OpDef{"view",
                        "Zero-copy alias: the output descriptor (shape/stride/data) is an exact "
                        "copy of the input and aliases its buffer. No allocation, no copy.",
                        f32_mask(), RS_AUTODIFF, view_infer, view_execute});
    ops.push_back(OpDef{"reshape",
                        "Reinterprets the input's row-major LOGICAL order as 'shape' (one -1 "
                        "allowed). A contiguous input aliases its buffer; a strided one is "
                        "copied into the output slot in logical order.",
                        f32_mask(), RS_AUTODIFF, reshape_infer, reshape_execute});
    ops.push_back(OpDef{"transpose",
                        "Zero-copy view: swaps axes dim0/dim1 (defaults -2/-1), swapping shape "
                        "and strides while aliasing the input buffer.",
                        f32_mask(), RS_AUTODIFF, transpose_infer, transpose_execute});
    ops.push_back(OpDef{"narrow",
                        "Zero-copy view: selects [start, start+length) along 'dim' (default -1) "
                        "with identical strides and an offset data pointer. No copy.",
                        f32_mask(), RS_AUTODIFF, narrow_infer, narrow_execute});
    ops.push_back(OpDef{"cat",
                        "Concatenates inputs along 'dim' (default -1). The exception among the "
                        "movement operators: cat copies, because the concatenated buffer is new.",
                        f32_mask(), RS_AUTODIFF, cat_infer, cat_execute});
    ops.push_back(OpDef{"broadcast",
                        "Zero-copy view: right-aligns the input against the target 'shape' "
                        "attribute; size-1 and new leading dims get stride 0.",
                        f32_mask(), RS_AUTODIFF, broadcast_infer, broadcast_execute});
}

}  // namespace rsaten
