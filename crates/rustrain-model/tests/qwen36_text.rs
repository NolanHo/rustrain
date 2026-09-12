//! D1 的真实 fixture 门禁：`tests/fixtures/qwen36-text/` 的 `config.json` + `model.json`
//! 必须能展开成一个全局 Plan。
//!
//! 模型事实（1045 个张量、40 层 3 linear + 1 full、MTP 19 个张量、真实形状）来自
//! `docs/design/qwen36-5d-example.md` 与 `_internal_docs/archive/qwen36-index-2026-04-24.json`；
//! fixture 里的形状与真实 safetensors 头部逐张量核对过。

use std::collections::BTreeSet;
use std::path::PathBuf;

use rustrain_model::Expanded;
use rustrain_parallel::ParallelLayout;
use rustrain_plan::SlotKind;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen36-text")
}

fn expanded() -> Expanded {
    rustrain_model::expand_dir(&fixture_dir()).expect("真实描述必须能展开")
}

fn weight_slots(expanded: &Expanded) -> usize {
    expanded
        .plan
        .slots
        .iter()
        .filter(|slot| slot.kind == SlotKind::Weight)
        .count()
}

/// D1 的可观察结果：节点数 > 900、slot 数 > 900。
///
/// **weight slot 的真实数是 884**，不是 D1 注释里估的 "约 903"：那条估算按
/// `24 slot × 30 + 18 slot × 10 + 根 3` 算，而每一层真实是 23 / 17 个权重 slot —— 估算每层多算了 1
/// （把被 `split` 拆掉的融合张量本身也当成一个 slot 了）。这里把真实值与它的算术钉住，
/// 见交付报告"契约里仍不清楚或我认为有错的"。
#[test]
fn the_real_model_expands_to_a_plan_over_nine_hundred_nodes_and_slots() {
    let expanded = expanded();
    let plan = &expanded.plan;

    assert!(
        plan.nodes.len() > 900,
        "D1 要求节点数 > 900，实际 {}",
        plan.nodes.len()
    );
    assert!(
        plan.slots.len() > 900,
        "D1 要求 slot 数 > 900，实际 {}",
        plan.slots.len()
    );

    // 712 个 checkpoint 张量（文本 693 = 根 3 + 层内 80+270+60+280，加 MTP 19）
    // + de-fuse 增量 = 884 个 weight slot：
    //   in_proj_qkv → Q|K|V：30 层 × (+2)  = +60
    //   conv1d      → q|k|v：30 层 × (+2)  = +60
    //   q_proj      → q|gate：11 层 × (+1) = +11（10 个 full 层 + MTP 的那一层）
    //   gate_up_proj→ gate|up：41 层 × (+1) = +41
    let tensors = 712;
    let expected_weights = tensors + 2 * 30 + 2 * 30 + 11 + 41;
    assert_eq!(expected_weights, 884);
    assert_eq!(
        weight_slots(&expanded),
        expected_weights,
        "weight slot 数必须等于 712 个张量加 de-fuse 增量"
    );
}

#[test]
fn the_layer_type_pattern_is_the_real_one() {
    let expanded = expanded();
    let names: Vec<&str> = expanded
        .plan
        .slots
        .iter()
        .map(|slot| slot.name.as_str())
        .collect();
    let has = |name: &str| names.contains(&name);

    let mut linear = BTreeSet::new();
    let mut full = BTreeSet::new();
    for layer in 0..40 {
        if has(&format!("layers.{layer}.linear_attn.A_log")) {
            linear.insert(layer);
        }
        if has(&format!("layers.{layer}.self_attn.q_norm")) {
            full.insert(layer);
        }
    }
    let full: Vec<usize> = full.into_iter().collect();
    assert_eq!(linear.len(), 30);
    assert_eq!(full, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
    assert_eq!(
        linear.len() + full.len(),
        40,
        "每层要么是 linear attention，要么是 full attention"
    );
}

#[test]
fn fused_storage_is_split_into_semantic_slots() {
    let expanded = expanded();
    let by_source = |needle: &str| {
        expanded
            .bindings
            .iter()
            .find(|binding| binding.source == needle)
            .unwrap_or_else(|| panic!("missing binding for {needle}"))
    };

    // q_proj [8192 = q(4096) | gate(4096), 2048]：10 个 full 层 × 2 段，MTP 那一层另有一条 binding。
    let q = by_source("model.language_model.layers.{*}.self_attn.q_proj.weight");
    assert_eq!(q.split.as_ref().unwrap().sizes, vec![4096, 4096]);
    assert_eq!(q.transform, vec!["transpose(0,1)"]);
    assert_eq!(q.slots.len(), 20);
    assert!(
        q.slots
            .iter()
            .any(|s| s.slot == "layers.3.self_attn.gate_proj")
    );
    let q_mtp = by_source("mtp.layers.{*}.self_attn.q_proj.weight");
    assert_eq!(q_mtp.slots.len(), 2);
    assert_eq!(q_mtp.slots[0].slot, "mtp.layers.0.self_attn.q_proj");

    // in_proj_qkv [8192 = Q | K | V, 2048]：30 层 × 3 段。
    let qkv = by_source("model.language_model.layers.{*}.linear_attn.in_proj_qkv.weight");
    assert_eq!(qkv.split.as_ref().unwrap().sizes, vec![2048, 2048, 4096]);
    assert_eq!(qkv.slots.len(), 90);

    // conv1d [8192, 1, 4]：depthwise，按 Q|K|V 切三段。
    let conv = by_source("model.language_model.layers.{*}.linear_attn.conv1d.weight");
    assert_eq!(conv.split.as_ref().unwrap().sizes, vec![2048, 2048, 4096]);
    assert_eq!(conv.slots.len(), 90);

    // experts.gate_up_proj [256, 1024, 2048]：40 层 × 2 段（+ MTP 那一层的 2 段）。
    let gate_up = by_source("model.language_model.layers.{*}.mlp.experts.gate_up_proj");
    assert_eq!(gate_up.split.as_ref().unwrap().sizes, vec![512, 512]);
    assert_eq!(gate_up.transform, vec!["transpose(1,2)"]);
    assert_eq!(gate_up.slots.len(), 80);
    let gate_up_mtp = by_source("mtp.layers.{*}.mlp.experts.gate_up_proj");
    assert_eq!(gate_up_mtp.slots.len(), 2);
}

#[test]
fn the_mtp_layer_is_described() {
    let expanded = expanded();
    let names: Vec<&str> = expanded
        .plan
        .slots
        .iter()
        .map(|slot| slot.name.as_str())
        .collect();
    for name in [
        "mtp.fc",
        "mtp.pre_fc_norm_embedding",
        "mtp.pre_fc_norm_hidden",
        "mtp.head.norm",
        "mtp.layers.0.input_layernorm",
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.self_attn.gate_proj",
        "mtp.layers.0.mlp.experts.gate_proj",
        "mtp.layers.0.mlp.shared_expert_gate",
    ] {
        assert!(names.contains(&name), "MTP 缺少 slot `{name}`");
    }
    // MTP 的 head 复用主 head 的权重：同一个 slot 被两个节点读。
    let head = expanded.plan.slot_id("lm_head.w").unwrap();
    let readers = expanded
        .plan
        .nodes
        .iter()
        .filter(|node| node.inputs.contains(&head))
        .count();
    assert_eq!(readers, 2, "lm_head.w 被主 head 与 MTP head 共用");
}

#[test]
fn every_weight_slot_is_bound_exactly_once() {
    let expanded = expanded();
    let bound: usize = expanded.bindings.iter().map(|b| b.slots.len()).sum();
    assert_eq!(bound, weight_slots(&expanded));
    assert_eq!(expanded.bindings.len(), 46, "46 条 binding 覆盖 712 个张量");
}

#[test]
fn the_global_plan_is_replicated_and_bf16() {
    let expanded = expanded();
    for slot in &expanded.plan.slots {
        assert_eq!(
            slot.layout,
            ParallelLayout::Replicate,
            "全局 Plan 的 layout 必须全 Replicate（§4.1）：{}",
            slot.name
        );
    }
    // 唯一的例外是 token 输入：`embedding` 读 i64 索引（§3.6 #6 就是为它修的）。
    let ids = expanded.plan.slot_id("input_ids").unwrap();
    assert_eq!(expanded.plan.slot(ids).dtype.name(), "i64");
    for slot in expanded
        .plan
        .slots
        .iter()
        .filter(|slot| slot.kind != SlotKind::Input)
    {
        assert_eq!(slot.dtype.name(), "bf16", "slot {}", slot.name);
    }
    // 形状是具体的（§4.1 第 3 步），没有符号维这种东西。
    for slot in &expanded.plan.slots {
        assert!(!slot.shape.is_empty(), "slot {} 没有形状", slot.name);
        assert!(
            slot.shape.iter().all(|d| *d > 0),
            "slot {} 形状非正",
            slot.name
        );
    }
}

#[test]
fn expansion_is_deterministic() {
    let first = serde_json::to_vec(&expanded().plan).unwrap();
    let second = serde_json::to_vec(&expanded().plan).unwrap();
    assert_eq!(
        first, second,
        "同一份描述 + 同一份 config 必须得到同一个 Plan"
    );
}
