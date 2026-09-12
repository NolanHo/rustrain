/*
 * A plugin whose only operator has no execute function.
 * The descriptor is malformed, not merely incomplete: the loader must reject
 * it at load time (the host has no way to run an operator without a body).
 */
#include "rustrain_op.h"

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
    .doc = "declared without an execute body",
    .requires = &req,
    .numerics = {0},
    .infer = NULL,
    .memory = NULL,
    .expansion = NULL,
    .backward = RS_NONDIFF,
    .backward_op = {.name = NULL, .variant = NULL, .version = 0},
    .collectives = NULL,
    .n_collectives = 0,
    .execute = NULL,
    .last_error = NULL,
};

static const rs_op_desc* const ops[] = {&desc};

static const rs_plugin no_execute_plugin = {
    .abi_version = RUSTRAIN_ABI_VERSION,
    .struct_size = sizeof(rs_plugin),
    .plugin_name = "cdata_no_execute",
    .plugin_version = "0.1.0",
    .n_ops = 1,
    ._pad = 0,
    .ops = ops,
    .init = NULL,
};

const rs_plugin* rustrain_plugin_v1(void) {
    return &no_execute_plugin;
}
