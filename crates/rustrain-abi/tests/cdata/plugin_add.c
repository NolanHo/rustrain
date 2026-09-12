/*
 * rustrain ABI test fixture: the happy path.
 *
 * This plugin exists to prove what the Rust side cannot prove on its own:
 *   1. the `_Static_assert`s below pin the wire layout from the C side, so the
 *      numbers asserted in `ffi::tests::layout` are an agreement, not a guess;
 *   2. a library compiled by the system C compiler can be dlopen'ed, enumerated
 *      and called back with rs_tensor descriptors built in Rust;
 *   3. a failing execute() reaches the host through the last_error() hook.
 *
 * It exports exactly one symbol (contract C-1) and has no init hook, so the
 * whole plugin is process-lifetime constant data.
 */
#include <stddef.h>

#include "rustrain_op.h"

/* ── cross-language layout agreement ─────────────────────────────────────── */

_Static_assert(sizeof(rs_tensor) == 192, "rs_tensor layout");
_Static_assert(sizeof(rs_op_id) == 24, "rs_op_id layout");
_Static_assert(sizeof(rs_numerics) == 44, "rs_numerics layout");
_Static_assert(sizeof(rs_attr) == 64, "rs_attr layout");
_Static_assert(sizeof(rs_attrs) == 16, "rs_attrs layout");
_Static_assert(sizeof(rs_requires) == 24, "rs_requires layout");
_Static_assert(sizeof(rs_collective) == 16, "rs_collective layout");
_Static_assert(sizeof(rs_mem_req) == 24, "rs_mem_req layout");
_Static_assert(sizeof(rs_expansion_node) == 48, "rs_expansion_node layout");
_Static_assert(sizeof(rs_expansion) == 32, "rs_expansion layout");
_Static_assert(sizeof(rs_services) == 56, "rs_services layout");
_Static_assert(sizeof(rs_ctx) == 16, "rs_ctx layout");
_Static_assert(sizeof(rs_op_desc) == 184, "rs_op_desc layout");
/*
 * 48, not 56: rs_plugin is 4 + 4 + 8 + 8 + 4 + 4 + 8 + 8 with every member in
 * its natural slot. There is no padding hole at any offset, and the header has
 * no field that could take the size to 56 — see the D1 report.
 */
_Static_assert(sizeof(rs_plugin) == 48, "rs_plugin layout");

/* Offsets of the fields the loader reads by name, pinned the same way the Rust
 * mirror pins them: a field reorder has to break both compilers. */
_Static_assert(offsetof(rs_op_desc, doc) == 32, "rs_op_desc.doc offset");
_Static_assert(offsetof(rs_op_desc, numerics) == 48, "rs_op_desc.numerics offset");
_Static_assert(offsetof(rs_op_desc, infer) == 96, "rs_op_desc.infer offset");
_Static_assert(offsetof(rs_op_desc, execute) == 168, "rs_op_desc.execute offset");
_Static_assert(offsetof(rs_expansion_node, outputs) == 32, "rs_expansion_node.outputs offset");
_Static_assert(offsetof(rs_services, log) == 48, "rs_services.log offset");

/* ── the operator ────────────────────────────────────────────────────────── */

static const char* g_last_error = "no error";

static int32_t add_execute(rs_ctx* ctx, const rs_tensor* const* in, uint32_t n_in,
                           rs_tensor* const* out, uint32_t n_out, const rs_attrs* attrs) {
    (void)ctx;
    (void)attrs;

    if (n_in != 2 || n_out != 1) {
        g_last_error = "add expects two inputs and one output";
        return 1;
    }
    const rs_tensor* a = in[0];
    const rs_tensor* b = in[1];
    rs_tensor* o = out[0];
    if (a == NULL || b == NULL || o == NULL) {
        g_last_error = "add received a null tensor descriptor";
        return 2;
    }
    if (a->dtype != RS_F32 || b->dtype != RS_F32 || o->dtype != RS_F32) {
        g_last_error = "add only implements f32";
        return 3;
    }
    if (a->rank != 1 || b->rank != 1 || o->rank != 1) {
        g_last_error = "add only implements rank-1 tensors";
        return 4;
    }
    if (a->shape[0] != o->shape[0] || b->shape[0] != o->shape[0]) {
        g_last_error = "add: input shapes differ";
        return 5;
    }
    /* The host promises contiguous tensors, so element stride is 1. A real
     * plugin would walk shape/stride; the fixture asserts the assumption. */
    if (a->stride[0] != 1 || b->stride[0] != 1 || o->stride[0] != 1) {
        g_last_error = "add assumes contiguous inputs";
        return 6;
    }
    if (a->data == NULL || b->data == NULL || o->data == NULL) {
        g_last_error = "add received a null element buffer";
        return 7;
    }

    const float* pa = (const float*)a->data;
    const float* pb = (const float*)b->data;
    float* po = (float*)o->data;
    for (int64_t i = 0; i < o->shape[0]; ++i) {
        po[i] = pa[i] + pb[i];
    }

    g_last_error = "no error";
    return 0;
}

static const char* add_last_error(rs_ctx* ctx) {
    (void)ctx;
    return g_last_error;
}

static const rs_requires add_requires = {
    .dtype_mask = 1u << RS_F32,
    .min_sm = 0,
    .min_world_size = 0,
    .needs_groups = 0,
    ._pad = 0,
};

static const rs_op_desc add_desc = {
    .abi_version = RUSTRAIN_ABI_VERSION,
    .struct_size = sizeof(rs_op_desc),
    .id = {.name = "add", .variant = "c", .version = 1},
    .doc = "elementwise sum of two contiguous f32 buffers",
    .requires = &add_requires,
    .numerics =
        {
            .in_dtype = RS_F32,
            .out_dtype = RS_F32,
            .accum_dtype = RS_F32,
            .grad_dtype = RS_F32,
            .quant = RS_Q_NONE,
            .block_m = 0,
            .block_n = 0,
            .scale_dtype = RS_F32,
            .scale_mode = RS_SCALE_STATIC,
            .amax_history = 0,
            ._pad = 0,
        },
    .infer = NULL,
    .memory = NULL,
    .expansion = NULL,
    .backward = RS_NONDIFF,
    .backward_op = {.name = NULL, .variant = NULL, .version = 0},
    .collectives = NULL,
    .n_collectives = 0,
    .execute = add_execute,
    .last_error = add_last_error,
};

static const rs_op_desc* const add_ops[] = {&add_desc};

static const rs_plugin cdata_add_plugin = {
    .abi_version = RUSTRAIN_ABI_VERSION,
    .struct_size = sizeof(rs_plugin),
    .plugin_name = "cdata_add",
    .plugin_version = "0.1.0",
    .n_ops = 1,
    ._pad = 0,
    .ops = add_ops,
    .init = NULL,
};

/* Contract C-1: the only exported symbol. */
const rs_plugin* rustrain_plugin_v1(void) {
    return &cdata_add_plugin;
}
