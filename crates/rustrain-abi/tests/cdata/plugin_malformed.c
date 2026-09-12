/*
 * Malformed op tables, selected at compile time.
 *
 *   -DNULL_OP_SLOT   n_ops = 2, ops[1] = NULL. Dropping the null slot would
 *                    silently renumber the ops, so the loader must reject it.
 *   -DNULL_OP_TABLE  n_ops = 2, ops = NULL. Treating this as "no ops" would
 *                    let a plugin declare operators it does not publish.
 */
#include "rustrain_op.h"

#if !defined(NULL_OP_SLOT) && !defined(NULL_OP_TABLE)
#error "build plugin_malformed.c with -DNULL_OP_SLOT or -DNULL_OP_TABLE"
#endif

#if defined(NULL_OP_SLOT)
static int32_t noop_execute(rs_ctx* ctx, const rs_tensor* const* in, uint32_t n_in,
                            rs_tensor* const* out, uint32_t n_out, const rs_attrs* attrs) {
    (void)ctx;
    (void)in;
    (void)n_in;
    (void)out;
    (void)n_out;
    (void)attrs;
    return 0;
}

static const rs_op_desc desc = {
    .abi_version = RUSTRAIN_ABI_VERSION,
    .struct_size = sizeof(rs_op_desc),
    .id = {.name = "add", .variant = "c", .version = 1},
    .doc = "a well-formed op next to a malformed table",
    .requires = NULL,
    .numerics = {0},
    .infer = NULL,
    .memory = NULL,
    .expansion = NULL,
    .backward = RS_NONDIFF,
    .backward_op = {.name = NULL, .variant = NULL, .version = 0},
    .collectives = NULL,
    .n_collectives = 0,
    .execute = noop_execute,
    .last_error = NULL,
};

static const rs_op_desc* const table[] = {&desc, NULL};
#else
static const rs_op_desc* const* const table = NULL;
#endif

static const rs_plugin malformed_plugin = {
    .abi_version = RUSTRAIN_ABI_VERSION,
    .struct_size = sizeof(rs_plugin),
    .plugin_name = "cdata_malformed",
    .plugin_version = "0.1.0",
    .n_ops = 2,
    ._pad = 0,
    .ops = table,
    .init = NULL,
};

const rs_plugin* rustrain_plugin_v1(void) {
    return &malformed_plugin;
}
