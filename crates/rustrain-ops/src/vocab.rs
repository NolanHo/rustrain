//! The fixed operator vocabulary of spec §2.4.
//!
//! Adding a *primitive* to this list is a framework evolution that needs
//! review; adding an implementation of an existing operator needs nothing but
//! a plugin. The registry uses the composite half of the list to answer "is
//! this operator required to declare an expansion?" (contract R-4), which is
//! what makes `describe()` able to flag a fused operator that shipped without
//! one.

/// Composite and model-block operators: every one of these *must* publish an
/// `expansion` (contract R-4, spec §2.4 rows "复合" and "模型块").
pub const COMPOSITE_OPS: &[&str] = &[
    // composites
    "sdpa",
    "flash_attn",
    "topk_router",
    "expert_dispatch",
    "expert_combine",
    "cross_entropy",
    "adamw",
    // model blocks
    "mlp_swiglu",
    "moe_layer",
    "transformer_layer",
    "dsa_attention",
    "gated_delta_rule",
];

/// True when `name` is a composite/model block, i.e. an operator that must
/// declare its primitive expansion.
pub fn is_composite_op(name: &str) -> bool {
    COMPOSITE_OPS.contains(&name)
}
