/*
 * A plugin that uses the optional init() hook (contract C-3).
 *
 * `probe` returns 0 only when init() handed it a service table it understands,
 * which is how a test observes service injection across the boundary without
 * the plugin exporting a second symbol.
 *
 * Built twice: with no -D (init succeeds) and with -DINIT_STATUS=42 (init
 * fails, which the loader must report as InitFailed).
 */
#include "rustrain_op.h"

#ifndef INIT_STATUS
#define INIT_STATUS 0
#endif

static rs_services* g_services = NULL;

static int32_t probe_execute(rs_ctx* ctx, const rs_tensor* const* in, uint32_t n_in,
                             rs_tensor* const* out, uint32_t n_out, const rs_attrs* attrs) {
    (void)ctx;
    (void)in;
    (void)n_in;
    (void)out;
    (void)n_out;
    (void)attrs;

    if (g_services == NULL) {
        return 10; /* the host supplied no service table */
    }
    if (g_services->abi_version != RUSTRAIN_ABI_VERSION) {
        return 11;
    }
    if (g_services->struct_size != sizeof(rs_services)) {
        return 12;
    }
    return 0;
}

static const rs_op_desc desc = {
    .abi_version = RUSTRAIN_ABI_VERSION,
    .struct_size = sizeof(rs_op_desc),
    .id = {.name = "probe", .variant = "c", .version = 1},
    .doc = "reports whether init() received a usable service table",
    .requires = NULL,
    .numerics = {0},
    .infer = NULL,
    .memory = NULL,
    .expansion = NULL,
    .backward = RS_NONDIFF,
    .backward_op = {.name = NULL, .variant = NULL, .version = 0},
    .collectives = NULL,
    .n_collectives = 0,
    .execute = probe_execute,
    .last_error = NULL,
};

static const rs_op_desc* const ops[] = {&desc};

static int32_t plugin_init(rs_services* svc) {
    g_services = svc;
    return INIT_STATUS;
}

static const rs_plugin init_plugin = {
    .abi_version = RUSTRAIN_ABI_VERSION,
    .struct_size = sizeof(rs_plugin),
    .plugin_name = "cdata_init",
    .plugin_version = "0.1.0",
    .n_ops = 1,
    ._pad = 0,
    .ops = ops,
    .init = plugin_init,
};

const rs_plugin* rustrain_plugin_v1(void) {
    return &init_plugin;
}
