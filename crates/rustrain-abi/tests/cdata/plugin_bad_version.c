/*
 * A structurally valid plugin whose header reports abi_version = 999.
 * Contract C-2: the loader must reject it instead of guessing a layout.
 */
#include "rustrain_op.h"

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

static const rs_requires req = {
    .dtype_mask = 1u << RS_F32,
    .min_sm = 0,
    .min_world_size = 0,
    .needs_groups = 0,
    ._pad = 0,
};

static const rs_op_desc desc = {
    .abi_version = RUSTRAIN_ABI_VERSION,
    .struct_size = sizeof(rs_op_desc),
    .id = {.name = "add", .variant = "c", .version = 1},
    .doc = "never reached: the plugin header is refused first",
    .requires = &req,
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

static const rs_op_desc* const ops[] = {&desc};

static const rs_plugin bad_version_plugin = {
    .abi_version = 999u,
    .struct_size = sizeof(rs_plugin),
    .plugin_name = "cdata_bad_version",
    .plugin_version = "0.1.0",
    .n_ops = 1,
    ._pad = 0,
    .ops = ops,
    .init = NULL,
};

const rs_plugin* rustrain_plugin_v1(void) {
    return &bad_version_plugin;
}
