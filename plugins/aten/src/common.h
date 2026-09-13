// rustrain ATen plugin — shared helpers.
//
// This plugin is T1: an implementation body. It publishes the same operator
// contract the reference provider publishes (same op names, same operand order,
// same attributes, same numerics) under a different variant, and maps each
// operator onto ATen (cuBLAS / cuDNN / torch CUDA kernels). No kernel is hand
// written here.
//
// Two conventions are load bearing:
//   * the variant name starts with `cuda`, which is how the framework knows
//     this implementation needs a device (rustrain-ops::capability) — and the
//     reason `run --device cuda` is not optional;
//   * a tensor descriptor is a *view*: shape and element strides name a window
//     into a buffer the framework owns. `at::from_blob` reproduces exactly that
//     window, so a strided input needs no copy and a view operator can hand its
//     result's pointer back to the executor.
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "rustrain_op.h"

namespace rsaten {

/// The variant every operator in this plugin is published under.
///
/// `cuda` is the device namespace a variant name is read for
/// (`rustrain-ops::capability::declared_device`), `aten` names the
/// implementation family, `f32` the numerics it accepts today.
inline constexpr const char* VARIANT = "cuda.aten.f32";

// ── errors ──────────────────────────────────────────────────────────────────

/// The message `last_error` hands back to the framework.
///
/// Thread local rather than global: the framework may execute several ranks on
/// several threads, and a shared string would let one rank's failure overwrite
/// another's message before it is read.
inline std::string& error_slot() {
    static thread_local std::string slot;
    return slot;
}

/// Records the message `last_error` will hand back and returns the ABI's failure
/// status.
///
/// Deliberately not `[[nodiscard]]`: several validation helpers report by writing
/// the message and returning `false` to their caller, and forcing every one of
/// those into `return fail(...)` would make the control flow harder to read than
/// the warning is worth.
inline int fail(std::string message) {
    error_slot() = std::move(message);
    return 1;
}

inline const char* last_error(rs_ctx*) { return error_slot().c_str(); }

/// Runs `body`, turning any C++ exception into the plugin's error channel.
///
/// ATen reports every contract violation (bad shape, bad device, bad dtype) by
/// throwing; letting one escape across the C ABI boundary would terminate the
/// host process instead of failing the operator.
template <typename Body>
inline int guard(const char* op, Body&& body) {
    try {
        return body();
    } catch (const std::exception& e) {
        return fail(std::string(op) + ": " + e.what());
    } catch (...) {
        return fail(std::string(op) + ": unknown C++ exception");
    }
}

// ── attributes ──────────────────────────────────────────────────────────────

/// Finds `key`, requiring the declared kind: an attribute present with the
/// wrong kind reads as absent, so a mistyped description fails where the
/// required-attribute check is written rather than silently reading a zero.
inline const rs_attr* find_attr(const rs_attrs* attrs, const char* key, rs_attr_kind kind) {
    if (attrs == nullptr || attrs->items == nullptr) {
        return nullptr;
    }
    for (uint32_t i = 0; i < attrs->len; ++i) {
        const rs_attr& a = attrs->items[i];
        if (a.key != nullptr && a.kind == kind && std::strcmp(a.key, key) == 0) {
            return &a;
        }
    }
    return nullptr;
}

inline bool attr_i64(const rs_attrs* attrs, const char* key, int64_t* out) {
    const rs_attr* a = find_attr(attrs, key, RS_ATTR_I64);
    if (a == nullptr) {
        return false;
    }
    *out = a->i64;
    return true;
}

inline bool attr_f64(const rs_attrs* attrs, const char* key, double* out) {
    const rs_attr* a = find_attr(attrs, key, RS_ATTR_F64);
    if (a == nullptr) {
        return false;
    }
    *out = a->f64;
    return true;
}

inline bool attr_bool(const rs_attrs* attrs, const char* key, bool* out) {
    const rs_attr* a = find_attr(attrs, key, RS_ATTR_BOOL);
    if (a == nullptr) {
        return false;
    }
    *out = a->boolean != 0;
    return true;
}

inline bool attr_str(const rs_attrs* attrs, const char* key, std::string* out) {
    const rs_attr* a = find_attr(attrs, key, RS_ATTR_STR);
    if (a == nullptr || a->str == nullptr) {
        return false;
    }
    *out = a->str;
    return true;
}

inline bool attr_i64s(const rs_attrs* attrs, const char* key, std::vector<int64_t>* out) {
    const rs_attr* a = find_attr(attrs, key, RS_ATTR_I64S);
    if (a == nullptr || a->i64s == nullptr) {
        return false;
    }
    out->assign(a->i64s, a->i64s + a->n_i64s);
    return true;
}

inline int64_t i64_or(const rs_attrs* attrs, const char* key, int64_t fallback) {
    int64_t v = 0;
    return attr_i64(attrs, key, &v) ? v : fallback;
}

inline double f64_or(const rs_attrs* attrs, const char* key, double fallback) {
    double v = 0;
    return attr_f64(attrs, key, &v) ? v : fallback;
}

inline bool bool_or(const rs_attrs* attrs, const char* key, bool fallback) {
    bool v = false;
    return attr_bool(attrs, key, &v) ? v : fallback;
}

/// A required string attribute that must be one of `accepted` (null-terminated).
/// A missing or unknown value is a hard error naming the accepted set — the
/// reference provider's contract, and the reason an unknown `kind` never
/// silently becomes a default.
inline bool require_kind(const rs_attrs* attrs, const char* key, const char* const* accepted,
                         int n_accepted, const char* op, std::string* out) {
    std::string value;
    if (!attr_str(attrs, key, &value)) {
        std::string list;
        for (int i = 0; i < n_accepted; ++i) {
            list += (i == 0 ? "" : ", ");
            list += accepted[i];
        }
        (void)fail(std::string(op) + ": attribute '" + key + "' is required and must be one of: " + list);
        return false;
    }
    for (int i = 0; i < n_accepted; ++i) {
        if (value == accepted[i]) {
            *out = value;
            return true;
        }
    }
    std::string list;
    for (int i = 0; i < n_accepted; ++i) {
        list += (i == 0 ? "" : ", ");
        list += accepted[i];
    }
    (void)fail(std::string(op) + ": unknown " + key + " '" + value + "'; accepted values: " + list);
    return false;
}

// ── tensor descriptors ──────────────────────────────────────────────────────

inline at::ScalarType scalar_of(rs_dtype d) {
    switch (d) {
        case RS_F32: return at::kFloat;
        case RS_F16: return at::kHalf;
        case RS_BF16: return at::kBFloat16;
        case RS_F8E4M3: return at::kFloat8_e4m3fn;
        case RS_F8E5M2: return at::kFloat8_e5m2;
        case RS_I32: return at::kInt;
        case RS_I64: return at::kLong;
        case RS_U8: return at::kByte;
        default: return at::kFloat;
    }
}

inline rs_dtype dtype_of(at::ScalarType s) {
    switch (s) {
        case at::kFloat: return RS_F32;
        case at::kHalf: return RS_F16;
        case at::kBFloat16: return RS_BF16;
        case at::kFloat8_e4m3fn: return RS_F8E4M3;
        case at::kFloat8_e5m2: return RS_F8E5M2;
        case at::kInt: return RS_I32;
        case at::kLong: return RS_I64;
        case at::kByte: return RS_U8;
        default: return RS_F32;
    }
}

/// The device every buffer in this plugin lives on: the thread's current CUDA
/// device. The framework sets it (one rank per process/thread); reading it here
/// rather than hard-coding 0 is what keeps `--device cuda:3` honest.
inline int64_t device_index() { return c10::cuda::current_device(); }

inline std::vector<int64_t> dims_of(const rs_tensor* t) {
    return std::vector<int64_t>(t->shape, t->shape + t->rank);
}

inline std::vector<int64_t> strides_of(const rs_tensor* t) {
    return std::vector<int64_t>(t->stride, t->stride + t->rank);
}

/// A non-owning ATen view of a framework buffer.
///
/// The strides are element strides in the descriptor and in ATen alike, so the
/// window is reproduced exactly; `from_blob` keeps the memory external, which
/// is the only correct choice here — the framework owns every byte this plugin
/// touches and frees them when the plan says so.
inline at::Tensor view(const rs_tensor* t) {
    return at::from_blob(t->data, dims_of(t), strides_of(t),
                         at::TensorOptions().dtype(scalar_of(t->dtype)).device(at::kCUDA,
                                                                                device_index()));
}

/// Element count of a descriptor's declared shape.
inline int64_t numel_of(const rs_tensor* t) {
    int64_t n = 1;
    for (uint32_t i = 0; i < t->rank && i < RS_MAX_RANK; ++i) {
        n *= t->shape[i];
    }
    return n;
}

/// The same window, meant to be written.
inline at::Tensor out_view(rs_tensor* t) { return view(t); }

/// Reads an index tensor as 64-bit: ATen's indexing kernels want `kLong`, and
/// the ABI's i32 form is widened explicitly rather than reinterpreted.
inline at::Tensor as_indices(const rs_tensor* t) {
    at::Tensor idx = view(t);
    return idx.scalar_type() == at::kLong ? idx : idx.to(at::kLong);
}

/// Copies `value` into the descriptor's buffer, refusing a shape the plan did
/// not declare (the executor sized that buffer from the slot's shape, so a
/// mismatch means someone is about to write outside it).
inline int write_out(rs_tensor* out, const at::Tensor& value, const char* op) {
    if (out == nullptr || out->data == nullptr) {
        return fail(std::string(op) + ": output descriptor has no buffer");
    }
    at::Tensor dst = out_view(out);
    if (dst.sizes() != value.sizes()) {
        return fail(std::string(op) + ": shape mismatch: the plan declares " +
                    std::to_string(dst.numel()) + " elements, the result has " +
                    std::to_string(value.numel()));
    }
    dst.copy_(value);
    return 0;
}

/// Hands a *view* result back to the executor: the output descriptor adopts the
/// result's pointer, shape and strides, and its storage stays the input's.
inline void adopt(rs_tensor* out, const at::Tensor& value) {
    out->dtype = dtype_of(value.scalar_type());
    out->rank = static_cast<uint32_t>(value.dim());
    for (uint32_t i = 0; i < out->rank && i < RS_MAX_RANK; ++i) {
        out->shape[i] = value.size(static_cast<int64_t>(i));
        out->stride[i] = value.stride(static_cast<int64_t>(i));
    }
    for (uint32_t i = out->rank; i < RS_MAX_RANK; ++i) {
        out->shape[i] = 0;
        out->stride[i] = 0;
    }
    out->data = value.data_ptr();
    out->scale = nullptr;
    out->amax = nullptr;
}

/// Fills an output descriptor's shape from `sizes`, leaving the buffer to the
/// executor (`infer` runs with null data pointers).
inline void set_shape(rs_tensor* out, at::IntArrayRef sizes) {
    out->dtype = RS_F32;
    out->rank = static_cast<uint32_t>(sizes.size());
    for (uint32_t i = 0; i < out->rank && i < RS_MAX_RANK; ++i) {
        out->shape[i] = sizes[static_cast<int64_t>(i)];
        out->stride[i] = 0;
    }
    for (uint32_t i = out->rank; i < RS_MAX_RANK; ++i) {
        out->shape[i] = 0;
        out->stride[i] = 0;
    }
}

/// Every operator here is f32 in and f32 out; a descriptor that is not f32 is a
/// framework bug (the variant declares its dtype set), so it is refused loudly.
inline int check_f32(const rs_tensor* t, const char* op, const char* who) {
    if (t == nullptr) {
        return fail(std::string(op) + ": " + who + " descriptor is null");
    }
    if (t->dtype != RS_F32) {
        return fail(std::string(op) + ": " + who + " is not f32 (this variant is f32 only)");
    }
    return 0;
}

/// `dim` with Python's negative-axis convention, validated against `rank`.
inline bool resolve_dim(int64_t dim, uint32_t rank, const char* op, int64_t* out) {
    int64_t r = static_cast<int64_t>(rank);
    if (dim < 0) {
        dim += r;
    }
    if (dim < 0 || dim >= r) {
        (void)fail(std::string(op) + ": axis " + std::to_string(dim) + " is out of range for rank " +
                   std::to_string(rank));
        return false;
    }
    *out = dim;
    return true;
}

// ── the operator table ──────────────────────────────────────────────────────

/// One operator as this plugin publishes it. The ABI descriptor is built from
/// this at load time; keeping the authoring form separate means the repeating
/// fields (variant, numerics, memory hook, last_error) are written once.
///
/// The function-pointer types are spelled out rather than typedef'd because
/// `rustrain_op.h` declares them inline in `rs_op_desc` — the header is the
/// contract, and these are its types.
using InferFn = int32_t (*)(const rs_tensor* const* in, uint32_t n_in, rs_tensor* const* out,
                            uint32_t n_out, const rs_attrs* attrs);
using ExecuteFn = int32_t (*)(rs_ctx* ctx, const rs_tensor* const* in, uint32_t n_in,
                              rs_tensor* const* out, uint32_t n_out, const rs_attrs* attrs);
using MemoryFn = int32_t (*)(const rs_tensor* const* io, uint32_t n_io, const rs_attrs* attrs,
                             rs_mem_req* out);

struct OpDef {
    const char* name;
    const char* doc;
    uint32_t dtype_mask;
    rs_backward_kind backward;
    InferFn infer;
    ExecuteFn execute;
    const rs_expansion* expansion = nullptr;
    const rs_collective* collectives = nullptr;
    uint32_t n_collectives = 0;
};

inline uint32_t f32_mask() { return 1u << RS_F32; }

/// Zero scratch: ATen allocates what it needs through its own caching
/// allocator, and this plugin's outputs are the framework's buffers.
inline int32_t memory_zero(const rs_tensor* const*, uint32_t, const rs_attrs*, rs_mem_req* out) {
    *out = rs_mem_req{0, 0, 0, 0};
    return 0;
}

void add_meta_ops(std::vector<OpDef>& ops);
void add_compute_ops(std::vector<OpDef>& ops);
void add_norm_ops(std::vector<OpDef>& ops);
void add_recurrent_ops(std::vector<OpDef>& ops);
void add_moe_ops(std::vector<OpDef>& ops);

}  // namespace rsaten
