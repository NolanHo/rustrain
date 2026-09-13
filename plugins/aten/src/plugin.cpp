// The plugin entry point: builds the ABI table once, from the operator
// families' authoring form.
//
// The ABI wants a process-lifetime, immutable table behind
// `rustrain_plugin_v1`; C++ gives that with a function-local static built on
// first call (thread-safe since C++11). The descriptions themselves live in a
// `std::deque`, whose addresses stay put as the table grows — a `std::vector`
// would move them.
#include <deque>
#include <mutex>
#include <vector>

#include "common.h"

namespace rsaten {
namespace {

struct Tables {
    std::deque<rs_requires> requires;
    std::deque<rs_numerics> numerics;
    std::deque<rs_op_desc> descs;
    std::vector<const rs_op_desc*> desc_ptrs;
    rs_plugin plugin{};
};

Tables build() {
    std::vector<OpDef> ops;
    add_meta_ops(ops);
    add_compute_ops(ops);
    add_norm_ops(ops);
    add_recurrent_ops(ops);
    add_moe_ops(ops);

    Tables t;
    t.requires.resize(ops.size());
    t.numerics.resize(ops.size());
    t.descs.resize(ops.size());
    t.desc_ptrs.reserve(ops.size());

    for (size_t i = 0; i < ops.size(); ++i) {
        const OpDef& def = ops[i];

        rs_requires& req = t.requires[i];
        req = rs_requires{};
        // A dtype mask of 0 means "declares no constraint" to the framework
        // (rustrain-ops::capability::declares_nothing), so an operator that
        // forgot its mask would silently accept every dtype. This plugin is
        // f32 today; say so.
        req.dtype_mask = def.dtype_mask == 0 ? f32_mask() : def.dtype_mask;
        req.min_sm = 0;
        req.min_world_size = 0;
        req.needs_groups = 0;

        rs_numerics& num = t.numerics[i];
        num = rs_numerics{};
        num.in_dtype = RS_F32;
        num.out_dtype = RS_F32;
        num.accum_dtype = RS_F32;
        num.grad_dtype = RS_F32;
        num.quant = RS_Q_NONE;
        num.scale_dtype = RS_F32;
        num.scale_mode = RS_SCALE_STATIC;

        rs_op_desc& desc = t.descs[i];
        desc = rs_op_desc{};
        desc.abi_version = RUSTRAIN_ABI_VERSION;
        desc.struct_size = sizeof(rs_op_desc);
        desc.id.name = def.name;
        desc.id.variant = VARIANT;
        desc.id.version = 1;
        desc.doc = def.doc;
        desc.requires = &req;
        desc.numerics = num;
        desc.infer = def.infer;
        desc.memory = memory_zero;
        desc.expansion = def.expansion;
        desc.backward = def.backward;
        desc.backward_op = rs_op_id{nullptr, nullptr, 0};
        desc.collectives = def.collectives;
        desc.n_collectives = def.n_collectives;
        desc.execute = def.execute;
        desc.last_error = last_error;

        t.desc_ptrs.push_back(&desc);
    }

    t.plugin.abi_version = RUSTRAIN_ABI_VERSION;
    t.plugin.struct_size = sizeof(rs_plugin);
    t.plugin.plugin_name = "aten";
    t.plugin.plugin_version = "0.1.0";
    t.plugin.n_ops = static_cast<uint32_t>(t.desc_ptrs.size());
    t.plugin.ops = t.desc_ptrs.data();
    t.plugin.init = nullptr;
    return t;
}

const Tables& tables() {
    static const Tables t = build();
    return t;
}

}  // namespace
}  // namespace rsaten

// Contract C-1: the single exported symbol.
extern "C" const rs_plugin* rustrain_plugin_v1(void) { return &rsaten::tables().plugin; }
