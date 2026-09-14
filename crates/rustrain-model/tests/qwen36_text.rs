//! D1 的真实 fixture 门禁：`tests/fixtures/qwen36-text/` 的 `config.json` + `model.json`
//! 必须能展开成一个全局 Plan。
//!
//! 模型事实（1045 个张量、40 层 3 linear + 1 full、MTP 19 个张量、真实形状）来自
//! `docs/design/qwen36-5d-example.md` 与 `_internal_docs/archive/qwen36-index-2026-04-24.json`；
//! fixture 里的形状与真实 safetensors 头部逐张量核对过。

use std::collections::BTreeSet;
use std::path::PathBuf;

use rustrain_model::Expanded;
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
/// **weight slot 的真实数是 873**，而不是 D1 注释里估的 "约 903"：那条估算按
/// `24 slot × 30 + 18 slot × 10 + 根 3` 算，而每一层真实是 23 / 16 个权重 slot —— 估算每层多算了 1
/// （把被 `split` 拆掉的融合张量本身也当成一个 slot 了）。
///
/// `q_proj` 一项从 +11 变成 0：真实段序是 per-head 交错 `[q₀|gate₀|q₁|gate₁|…]`
/// （`docs/design/qwen36-5d-example.md` §3），连续切分正好按 head 对切开，所以它**不需要 de-fuse**，
/// 一个张量就是一个 slot。
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
    // + de-fuse 增量 = 873 个 weight slot：
    //   in_proj_qkv → Q|K|V：30 层 × (+2)  = +60
    //   conv1d      → q|k|v：30 层 × (+2)  = +60
    //   q_proj      → 不拆（per-head 交错）：      +0
    //   gate_up_proj→ gate|up：41 层 × (+1) = +41
    let tensors = 712;
    let expected_weights = tensors + 2 * 30 + 2 * 30 + 41;
    assert_eq!(expected_weights, 873);
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

    // q_proj [8192, 2048]：真实段序是 per-head 交错 `[q₀(256)|gate₀(256)|q₁|gate₁|…]`（§3），
    // 所以**不拆** —— 连续切分按 head 对切，每个 rank 都拿到完整的 head 对。
    // q 与 gate 的分离在激活上做（`the_q_gate_split_is_a_graph_fact`）。
    // 10 个 full 层各一个 slot，MTP 那一层另有一条 binding。
    let q = by_source("model.language_model.layers.{*}.self_attn.q_proj.weight");
    assert!(q.split.is_none(), "q_proj 不该 de-fuse（§3）");
    assert_eq!(q.transform, vec!["transpose(0,1)"]);
    assert_eq!(q.slots.len(), 10);
    for slot in &q.slots {
        assert!(
            slot.slot.ends_with(".self_attn.qg"),
            "q_proj 只喂一个融合 slot，实际 {}",
            slot.slot
        );
        assert_eq!(slot.axes.get("1"), Some(&vec!["tp".to_string()]));
    }
    let q_mtp = by_source("mtp.layers.{*}.self_attn.q_proj.weight");
    assert!(q_mtp.split.is_none());
    assert_eq!(q_mtp.slots.len(), 1);
    assert_eq!(q_mtp.slots[0].slot, "mtp.layers.0.self_attn.qg");

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

/// §3：`q_proj` 的融合张量不拆，q 与 gate 的分离**画在图里** —— `reshape` 到
/// `[seq, heads, 2, head_dim]`，两个 `narrow(dim=2)` 各取一半，再各自 `reshape` 成
/// `[seq, heads, head_dim]` 的 per-head 视图。于是"哪一半是 q"这件事是计划里的节点，不是加载期的隐式约定。
///
/// **D5 已裁定（op-vocabulary §3.1）**：`q_norm` / `k_norm` / `rope` / `sdpa` 以及
/// output gate 全部吃 **per-head** `[seq, heads, head_dim]`（`[head_dim]` 的 norm 权重只对最后一维成立），
/// 只有 `o_proj` 之前才 `reshape` 回 `[seq, q_size]`。早期版本把 q 提前拍平回 `[512, 4096]`，
/// 让 `q_norm` 在 4096 维的跨 head 轴上做归一 —— 那是描述缺陷，不是模型。
#[test]
fn the_q_gate_split_is_a_graph_fact_not_a_load_time_split() {
    let expanded = expanded();
    let plan = &expanded.plan;
    let id = |name: &str| {
        plan.slot_id(name)
            .unwrap_or_else(|| panic!("缺少 slot {name}"))
    };
    let name_of = |slot: rustrain_plan::SlotId| plan.slot(slot).name.clone();

    // 一个 linear 读融合权重，产出交错布局的 [seq, 2 * q_size]。
    let qg = id("layers.3.self_attn.qg");
    assert_eq!(plan.slot(qg).shape, vec![2048, 8192]);
    let linear = plan
        .nodes
        .iter()
        .find(|node| node.op.name == "linear" && node.inputs.contains(&qg))
        .expect("没有节点读 layers.3.self_attn.qg");
    assert_eq!(name_of(linear.outputs[0]), "layers.3.qgw");
    assert_eq!(plan.slot(linear.outputs[0]).shape, vec![512, 8192]);

    // reshape → [seq, heads, 2, head_dim]。
    let view = linear.outputs[0];
    let reshape = plan
        .nodes
        .iter()
        .find(|node| node.op.name == "reshape" && node.inputs.contains(&view))
        .expect("交错布局没有被 reshape 成 [seq, heads, 2, head_dim]");
    // The head dim is spelled `-1`, not `16`: the description is topology-free, and a literal
    // head count would be wrong the moment the head axis is sharded (a tp=2 rank holds 8 of
    // them). `-1` resolves against the local element count, so the same description is right at
    // every degree — and a wrong spelling is a hard error, never a silently mis-shaped tensor.
    assert_eq!(
        reshape.attrs.i64s("shape"),
        Some([512, -1, 2, 256].as_slice())
    );
    assert_eq!(name_of(reshape.outputs[0]), "layers.3.qgh");

    // 两个 narrow(dim=2)：start 0 是 q，start 1 是 gate。
    let halves: Vec<&rustrain_plan::PlanNode> = plan
        .nodes
        .iter()
        .filter(|node| node.op.name == "narrow" && node.inputs.contains(&reshape.outputs[0]))
        .collect();
    assert_eq!(halves.len(), 2, "q 与 gate 各一次 narrow");
    let half = |start: i64| {
        let node = halves
            .iter()
            .find(|node| node.attrs.i64("start") == Some(start))
            .unwrap_or_else(|| panic!("没有 start={start} 的 narrow"));
        assert_eq!(node.attrs.i64("dim"), Some(2));
        assert_eq!(node.attrs.i64("length"), Some(1));
        assert_eq!(plan.slot(node.outputs[0]).shape, vec![512, 16, 1, 256]);
        node.outputs[0]
    };

    // 各自 reshape 成 per-head `[seq, heads, head_dim]`（丢掉 narrow 留下的 1 维）：
    // `q` 进 q_norm，`qg` 进门控 sigmoid —— 两者都按 head 作用（op-vocabulary §3.1）。
    for (source, target) in [(half(0), "layers.3.q"), (half(1), "layers.3.qg")] {
        let reshape = plan
            .nodes
            .iter()
            .find(|node| node.op.name == "reshape" && node.inputs.contains(&source))
            .unwrap_or_else(|| panic!("{target} 的半边没有被 reshape 成 per-head"));
        assert_eq!(
            reshape.attrs.i64s("shape"),
            Some([512, -1, 256].as_slice()),
            "{target}: the head axis is `-1`, so a tp rank's local head count resolves from the tensor"
        );
        assert_eq!(name_of(reshape.outputs[0]), target);
        assert_eq!(plan.slot(id(target)).shape, vec![512, 16, 256]);
    }

    // k / v 在进 norm 与 sdpa 之前 reshape 成 per-head `[seq, kv_heads, head_dim]`（§3.1），
    // 而 o（per-head，与 gate 同形相乘）在 o_proj 之前才拍平回 `[seq, q_size]`。
    assert_eq!(plan.slot(id("layers.3.k")).shape, vec![512, 512]);
    assert_eq!(plan.slot(id("layers.3.kh")).shape, vec![512, 2, 256]);
    assert_eq!(plan.slot(id("layers.3.v")).shape, vec![512, 512]);
    assert_eq!(plan.slot(id("layers.3.vh")).shape, vec![512, 2, 256]);
    assert_eq!(plan.slot(id("layers.3.qn")).shape, vec![512, 16, 256]);
    assert_eq!(plan.slot(id("layers.3.kn")).shape, vec![512, 2, 256]);
    assert_eq!(plan.slot(id("layers.3.o")).shape, vec![512, 16, 256]);
    assert_eq!(plan.slot(id("layers.3.og")).shape, vec![512, 16, 256]);
    assert_eq!(plan.slot(id("layers.3.ogf")).shape, vec![512, 4096]);
}

/// D5 修正后的 GDN 与 MoE 契约，两条都钉在图里：
///
/// **GDN 的 l2norm 是 per-head 的**（HF `Qwen3_5MoeGatedDeltaNet.forward` 先把 q/k
/// `reshape` 成 `[b, s, num_k_heads, head_k_dim]`，再在 kernel 里 `l2norm(..., dim=-1)`）。
/// 描述因此要先把 `cq`/`ck` reshape 成 `[seq, lin_k_heads, lin_k_dim]`，l2norm 按
/// `dim = -1` 作用在 128 维的 head 上，再 reshape 回 `[seq, lin_qk]` —— `gated_delta_rule`
/// 的契约吃扁平输入（它内部的 `1/sqrt(head_dim)` 只作用在 q 上，与 HF 一致）。
/// 早期版本在扁平的 `[seq, 2048]` 上直接 l2norm —— 那是在 16 个 head 拼起来的 2048 维上归一，
/// 与 `decoder_full` 早先的跨 head 轴归一是一样的描述缺陷。
///
/// **router 出两个张量、moe 吃十个输入**：`topk_router` 的 op 契约是
/// `(routing_weights f32, routing_indices i32)` 两个输出；`moe_layer` 按这个顺序消费它们，
/// 加上 de-fused 的 `gate_proj` / `up_proj`（融合的 `[E, 2I, H]` 在 TP 切分下会把 2I 轴
/// 从 gate/up 边界切开，没法正确切分 —— 这正是 D1 把它 de-fuse 掉的原因）。
#[test]
fn the_gdn_l2norm_is_per_head_and_the_router_moe_contract_is_de_fused() {
    let expanded = expanded();
    let plan = &expanded.plan;
    let id = |name: &str| {
        plan.slot_id(name)
            .unwrap_or_else(|| panic!("缺少 slot {name}"))
    };
    let name_of = |slot: rustrain_plan::SlotId| plan.slot(slot).name.clone();

    // 层 0 是 linear_attention：cq/ck 扁平 [seq, 2048]，经 per-head reshape 后按
    // dim=-1 归一，再拍平回 [seq, 2048] 交给 gated_delta_rule。
    for (flat, headed, normed, out) in [
        ("layers.0.cq", "layers.0.qh", "layers.0.qhn", "layers.0.qn"),
        ("layers.0.ck", "layers.0.kh", "layers.0.khn", "layers.0.kn"),
    ] {
        assert_eq!(plan.slot(id(flat)).shape, vec![512, 2048]);
        let reshape = plan
            .nodes
            .iter()
            .find(|node| node.op.name == "reshape" && node.inputs.contains(&id(flat)))
            .unwrap_or_else(|| panic!("{flat} 没有被 reshape 成 per-head"));
        assert_eq!(
            reshape.attrs.i64s("shape"),
            Some([512, -1, 128].as_slice()),
            "{flat}: the head axis is `-1` so the shape stays true under sharding"
        );
        assert_eq!(name_of(reshape.outputs[0]), headed);
        assert_eq!(plan.slot(id(headed)).shape, vec![512, 16, 128]);

        let norm = plan
            .nodes
            .iter()
            .find(|node| node.op.name == "l2norm" && node.inputs.contains(&id(headed)))
            .unwrap_or_else(|| panic!("{headed} 没有接 l2norm"));
        assert_eq!(
            norm.attrs.i64("dim"),
            Some(-1),
            "l2norm 必须按 head 的最后一维归一"
        );
        assert_eq!(name_of(norm.outputs[0]), normed);

        let flat_back = plan
            .nodes
            .iter()
            .find(|node| node.op.name == "reshape" && node.inputs.contains(&id(normed)))
            .unwrap_or_else(|| panic!("{normed} 没有被 reshape 回扁平形式"));
        assert_eq!(
            flat_back.attrs.i64s("shape"),
            Some([512, -1].as_slice()),
            "{normed}: flattening back keeps whatever the local head count is"
        );
        assert_eq!(name_of(flat_back.outputs[0]), out);
        assert_eq!(plan.slot(id(out)).shape, vec![512, 2048]);
    }
    let gdn = plan
        .nodes
        .iter()
        .find(|node| node.op.name == "gated_delta_rule" && node.inputs.contains(&id("layers.0.qn")))
        .expect("gated_delta_rule 吃拍平后的 qn/kn");
    let gdn_inputs: Vec<String> = gdn.inputs.iter().map(|s| name_of(*s)).collect();
    assert_eq!(
        gdn_inputs,
        vec![
            "layers.0.qn",
            "layers.0.kn",
            "layers.0.cv",
            "layers.0.g",
            "layers.0.beta",
        ],
        "gated_delta_rule 的输入顺序是 (qn, kn, v, g, beta)，q/k 用拍平后的 2048 维形式"
    );

    // router：两个输出，顺序是 op 契约的 (weights, indices)；top_k 是 op 读的属性名。
    let router = plan
        .nodes
        .iter()
        .find(|node| node.op.name == "topk_router" && node.inputs.contains(&id("layers.0.rlogits")))
        .expect("层 0 有 topk_router");
    assert_eq!(router.attrs.i64("top_k"), Some(8));
    assert_eq!(router.outputs.len(), 2, "router 必须出两个张量");
    assert_eq!(name_of(router.outputs[0]), "layers.0.routing_weights");
    assert_eq!(name_of(router.outputs[1]), "layers.0.routing_indices");
    assert_eq!(plan.slot(router.outputs[0]).shape, vec![512, 8]);
    assert_eq!(plan.slot(router.outputs[1]).shape, vec![512, 8]);
    assert_eq!(plan.slot(router.outputs[1]).dtype.name(), "i32");

    // moe_layer：十个输入，按 op 契约的顺序（h, weights, indices, gate, up, down,
    // shared gate/up/down, shared gate）；没有属性（top_k / norm_topk_prob 属于 router）。
    let moe = plan
        .nodes
        .iter()
        .find(|node| node.op.name == "moe_layer" && node.inputs.contains(&id("layers.0.h3")))
        .expect("层 0 有 moe_layer");
    let inputs: Vec<String> = moe.inputs.iter().map(|s| name_of(*s)).collect();
    assert_eq!(
        inputs,
        vec![
            "layers.0.h3",
            "layers.0.routing_weights",
            "layers.0.routing_indices",
            "layers.0.mlp.experts.gate_proj",
            "layers.0.mlp.experts.up_proj",
            "layers.0.mlp.experts.down_proj",
            "layers.0.mlp.shared_expert.gate_proj",
            "layers.0.mlp.shared_expert.up_proj",
            "layers.0.mlp.shared_expert.down_proj",
            "layers.0.mlp.shared_expert_gate",
        ],
        "moe_layer 的输入顺序必须等于 op 契约"
    );
    assert!(
        moe.attrs.is_empty(),
        "moe_layer 节点不得携带 router 的 top_k / norm_topk_prob 属性"
    );

    // 权重 slot 保持 de-fused，方向按 `moe_layer` 的 op 契约：gate/up 与 down 都是
    // [experts, hidden, moe_inter]（per-expert [H, I]，checkpoint 的原生朝向，不进融合张量）。
    for name in [
        "layers.0.mlp.experts.gate_proj",
        "layers.0.mlp.experts.up_proj",
        "layers.0.mlp.experts.down_proj",
    ] {
        assert_eq!(plan.slot(id(name)).shape, vec![256, 2048, 512], "{name}");
    }
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
        "mtp.layers.0.self_attn.qg",
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
        assert!(
            slot.layout.is_replicated(),
            "全局 Plan 的 layout 必须全 replicated（§4.1）：{}",
            slot.name
        );
    }
    // 例外一：token 输入 `input_ids` 是 i64 索引（§3.6 #6 就是为它修的）。
    // 例外二：`routing_indices` 是 `topk_router` 的第二个输出（op 契约里就是 i32 的
    // 专家索引张量，不是可替代精度的浮点激活），全局 Plan 用描述里声明的 i32，不进
    // `--dtype` 覆盖 —— 与 `input_ids` 同一条"索引不是精度"的口径。
    let ids = expanded.plan.slot_id("input_ids").unwrap();
    assert_eq!(expanded.plan.slot(ids).dtype.name(), "i64");
    for slot in expanded
        .plan
        .slots
        .iter()
        .filter(|slot| slot.kind != SlotKind::Input)
    {
        if slot.name.ends_with(".routing_indices") {
            assert_eq!(slot.dtype.name(), "i32", "slot {}", slot.name);
        } else {
            assert_eq!(slot.dtype.name(), "bf16", "slot {}", slot.name);
        }
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

/// 主干 `rmsnorm` 必须显式声明 `eps = text_config.rms_norm_eps`（1e-6）。
///
/// 这不是风格问题：provider 的缺省值是 1e-5，而 Qwen3.6 的 embedding 方差只有 ~9e-5，
/// 两者差一个量级，归一化标度因此差 sqrt((9e-5 + 1e-5)/(9e-5 + 1e-6)) ≈ 0.954 —— **第一个
/// 算子就带 ~4.6% 的尺度误差**，后面每一层都继承它。逐元素对比正是靠这个把
/// `layers.0.h1` 从 5.5e-2 的 rel_L2 拉回 7e-8。
#[test]
fn the_trunk_rmsnorms_declare_the_config_eps() {
    let expanded = expanded();
    let plan = &expanded.plan;
    let rmsnorms: Vec<&rustrain_plan::PlanNode> = plan
        .nodes
        .iter()
        .filter(|node| node.op.name == "rmsnorm")
        .collect();
    assert!(!rmsnorms.is_empty(), "fixture 里必须有 rmsnorm 节点");
    for node in rmsnorms {
        assert_eq!(
            node.attrs.f64("eps"),
            Some(1e-6),
            "主干 rmsnorm `{}` 必须声明 eps = text_config.rms_norm_eps (1e-6)；\
             缺省时 provider 用 1e-5，embedding 方差 ~9e-5 下首算子就差 ~4.6%",
            plan.slot(node.outputs[0]).name
        );
    }
}
