//! `expand` 的行为：链式接线、`select` 按列表下标、`split` → `targets`、以及六条错误路径。
//!
//! 用小描述在内存里跑（不动文件系统），真实模型的 fixture 在 `tests/qwen36_text.rs`。

use rustrain_model::{ModelDesc, ModelError, expand};
use rustrain_plan::SlotKind;

fn config() -> serde_json::Value {
    serde_json::json!({
        "text_config": {
            "hidden_size": 16,
            "num_hidden_layers": 3,
            "max_position_embeddings": 8
        }
    })
}

/// 与 `crates/rustrain-cli/tests/fixtures/model-desc/ok` 同构的最小模型。
fn tiny() -> ModelDesc {
    serde_json::from_str(
        r#"{
        "format": "rustrain.model.v1",
        "name": "tiny-text",
        "dtype": "f32",
        "inputs": { "hidden_in": { "shape": ["seq", "hidden"], "kind": "input" } },
        "params": {
            "seq": { "from": "text_config.max_position_embeddings", "default": 8 },
            "hidden": { "from": "text_config.hidden_size" },
            "inter": { "expr": "2 * hidden" },
            "layers": { "from": "text_config.num_hidden_layers" }
        },
        "templates": {
            "norm": {
                "inputs": { "x": { "shape": ["seq", "hidden"], "kind": "activation" } },
                "outputs": { "y": { "shape": ["seq", "hidden"], "kind": "activation" } },
                "slots": [ { "name": "w", "kind": "weight", "shape": ["hidden"] } ],
                "nodes": [ { "op": "rmsnorm", "in": ["x", "w"], "out": ["y"] } ]
            },
            "decoder": {
                "inputs": { "x": { "shape": ["seq", "hidden"], "kind": "activation" } },
                "outputs": { "y": { "shape": ["seq", "hidden"], "kind": "activation" } },
                "slots": [
                    { "name": "attn_norm", "kind": "weight", "shape": ["hidden"] },
                    { "name": "qkv", "kind": "weight", "shape": ["hidden", "hidden"] },
                    { "name": "wo", "kind": "weight", "shape": ["hidden", "hidden"] },
                    { "name": "mlp_norm", "kind": "weight", "shape": ["hidden"] },
                    { "name": "wg", "kind": "weight", "shape": ["hidden", "inter"] },
                    { "name": "wu", "kind": "weight", "shape": ["hidden", "inter"] },
                    { "name": "wd", "kind": "weight", "shape": ["inter", "hidden"] },
                    { "name": "h1", "kind": "activation", "dtype": "f32", "shape": ["seq", "hidden"] },
                    { "name": "q", "kind": "activation", "dtype": "f32", "shape": ["seq", "hidden"] },
                    { "name": "qa", "kind": "activation", "dtype": "f32", "shape": ["seq", "hidden"] },
                    { "name": "attn", "kind": "activation", "dtype": "f32", "shape": ["seq", "hidden"] },
                    { "name": "h2", "kind": "activation", "dtype": "f32", "shape": ["seq", "hidden"] },
                    { "name": "h3", "kind": "activation", "dtype": "f32", "shape": ["seq", "hidden"] },
                    { "name": "g", "kind": "activation", "dtype": "f32", "shape": ["seq", "inter"] },
                    { "name": "gs", "kind": "activation", "dtype": "f32", "shape": ["seq", "inter"] },
                    { "name": "u", "kind": "activation", "dtype": "f32", "shape": ["seq", "inter"] },
                    { "name": "gu", "kind": "activation", "dtype": "f32", "shape": ["seq", "inter"] }
                ],
                "nodes": [
                    { "op": "rmsnorm", "in": ["x", "attn_norm"], "out": ["h1"] },
                    { "op": "linear", "in": ["h1", "qkv"], "out": ["q"] },
                    { "op": "elementwise_unary", "in": ["q"], "out": ["qa"], "attrs": { "kind": "silu" } },
                    { "op": "linear", "in": ["qa", "wo"], "out": ["attn"] },
                    { "op": "elementwise_binary", "in": ["h1", "attn"], "out": ["h2"], "attrs": { "kind": "mul" } },
                    { "op": "rmsnorm", "in": ["h2", "mlp_norm"], "out": ["h3"] },
                    { "op": "linear", "in": ["h3", "wg"], "out": ["g"] },
                    { "op": "elementwise_unary", "in": ["g"], "out": ["gs"], "attrs": { "kind": "silu" } },
                    { "op": "linear", "in": ["h3", "wu"], "out": ["u"] },
                    { "op": "elementwise_binary", "in": ["gs", "u"], "out": ["gu"], "attrs": { "kind": "mul" } },
                    { "op": "linear", "in": ["gu", "wd"], "out": ["y"] }
                ]
            }
        },
        "stack": [
            { "template": "norm", "prefix": "norm_in", "inputs": { "x": "hidden_in" } },
            { "template": "decoder", "prefix": "layers.{l}", "repeat": { "count": "layers", "index": "l" } },
            { "template": "norm", "prefix": "norm_out" }
        ],
        "binding": [
            { "slot": "norm_in.w", "source": "model.norm_in.weight" },
            { "slot": "norm_out.w", "source": "model.norm.weight" },
            { "slot": "layers.*.attn_norm", "source": "model.layers.{*}.attn_norm.weight" },
            { "slot": "layers.*.qkv", "source": "model.layers.{*}.qkv.weight" },
            { "slot": "layers.*.wo", "source": "model.layers.{*}.wo.weight" },
            { "slot": "layers.*.mlp_norm", "source": "model.layers.{*}.mlp_norm.weight" },
            { "slot": "layers.*.wg", "source": "model.layers.{*}.wg.weight" },
            { "slot": "layers.*.wu", "source": "model.layers.{*}.wu.weight" },
            { "slot": "layers.*.wd", "source": "model.layers.{*}.wd.weight" }
        ]
    }"#,
    )
    .unwrap()
}

fn weight_slots(plan: &rustrain_plan::Plan) -> usize {
    plan.slots
        .iter()
        .filter(|slot| slot.kind == SlotKind::Weight)
        .count()
}

#[test]
fn expands_the_stack_with_chained_wiring() {
    let expanded = expand(&tiny(), &config()).unwrap();
    let plan = &expanded.plan;

    assert_eq!(plan.nodes.len(), 35, "1 norm + 3 decoder + 1 norm");
    assert_eq!(weight_slots(plan), 23);
    assert_eq!(plan.slots.len(), 59);

    // 链式接线：第一层的输入是 norm_in 的输出，之后是上一层的输出。
    for layer in 0..3 {
        let node = plan
            .nodes
            .iter()
            .find(|n| n.source.path == format!("layers.{layer}.h1"))
            .expect("每一层都有 input_layernorm 节点");
        let expected = if layer == 0 {
            "norm_in.y".to_string()
        } else {
            format!("layers.{}.y", layer - 1)
        };
        assert_eq!(plan.slot(node.inputs[0]).name, expected);
    }
    // 实例端口接的是已存在的全局 slot，不会为每个实例新造一个 `layers.N.x`。
    assert!(plan.slot_id("layers.0.x").is_none());

    // 中间激活在模板里显式声明（§3.7 #1），形状来自声明本身。
    let h1 = plan.slot_id("layers.0.h1").unwrap();
    assert_eq!(plan.slot(h1).shape, vec![8, 16]);
    assert_eq!(plan.slot(h1).kind, SlotKind::Activation);
    // MLP 中间激活有自己的形状（`inter = 2 * hidden`），不是第一个输入的形状。
    let g = plan.slot_id("layers.0.g").unwrap();
    assert_eq!(plan.slot(g).shape, vec![8, 32]);
    // 声明的 output 用自己的形状。
    let y = plan.slot_id("layers.0.y").unwrap();
    assert_eq!(plan.slot(y).shape, vec![8, 16]);
}

#[test]
fn bindings_cover_every_weight_slot_and_the_plan_is_deterministic() {
    let first = expand(&tiny(), &config()).unwrap();
    let second = expand(&tiny(), &config()).unwrap();

    let bound: usize = first.bindings.iter().map(|b| b.slots.len()).sum();
    assert_eq!(bound, weight_slots(&first.plan));
    assert_eq!(first.bindings.len(), 9);

    let a = serde_json::to_string(&first.plan).unwrap();
    let b = serde_json::to_string(&second.plan).unwrap();
    assert_eq!(a, b, "同一份描述必须得到同一个 Plan");
    assert!(
        a.contains("\"replicate\""),
        "全局 Plan 的 layout 全 Replicate"
    );
}

#[test]
fn select_picks_a_template_by_list_index() {
    let mut desc = tiny();
    let mut extra: ModelDesc = serde_json::from_str(
        r#"{
        "format": "rustrain.model.v1",
        "name": "two-templates",
        "dtype": "f32",
        "inputs": { "hidden_in": { "shape": ["seq", "hidden"], "kind": "input" } },
        "params": {
            "seq": { "from": "text_config.max_position_embeddings", "default": 8 },
            "hidden": { "from": "text_config.hidden_size" },
            "layers": { "from": "text_config.num_hidden_layers" },
            "layer_types": ["full", "linear", "full"]
        },
        "templates": {
            "full": {
                "inputs": { "x": { "shape": ["seq", "hidden"], "kind": "activation" } },
                "outputs": { "y": { "shape": ["seq", "hidden"], "kind": "activation" } },
                "slots": [ { "name": "w", "kind": "weight", "shape": ["hidden"] } ],
                "nodes": [ { "op": "rmsnorm", "in": ["x", "w"], "out": ["y"] } ]
            },
            "linear": {
                "inputs": { "x": { "shape": ["seq", "hidden"], "kind": "activation" } },
                "outputs": { "y": { "shape": ["seq", "hidden"], "kind": "activation" } },
                "slots": [ { "name": "w", "kind": "weight", "shape": ["hidden"] } ],
                "nodes": [ { "op": "silu_gate", "in": ["x", "w"], "out": ["y"] } ]
            }
        },
        "stack": [
            { "prefix": "layers.{l}", "repeat": { "count": "layers", "index": "l" },
              "inputs": { "x": "hidden_in" },
              "select": { "by": "layer_types[l]", "cases": { "full": "full", "linear": "linear" } } }
        ],
        "binding": [ { "slot": "layers.*.w", "source": "model.layers.{*}.w" } ]
    }"#,
    )
    .unwrap();
    extra.inputs = desc.inputs.clone();
    desc = extra;

    let expanded = expand(&desc, &config()).unwrap();
    let ops: Vec<&str> = expanded
        .plan
        .nodes
        .iter()
        .map(|n| n.op.name.as_str())
        .collect();
    assert_eq!(ops, vec!["rmsnorm", "silu_gate", "rmsnorm"]);
}

#[test]
fn a_binding_that_matches_nothing_names_the_pattern() {
    let mut desc = tiny();
    desc.binding.push(
        serde_json::from_str(
            r#"{ "slot": "layers.*.no_such_weight", "source": "model.layers.{*}.no_such_weight" }"#,
        )
        .unwrap(),
    );
    let err = expand(&desc, &config()).unwrap_err();
    assert!(err.to_string().contains("layers.*.no_such_weight"), "{err}");
}

#[test]
fn a_weight_slot_without_a_binding_is_rejected() {
    let mut desc = tiny();
    desc.binding
        .retain(|b| b.slot.as_deref() != Some("layers.*.wd"));
    let err = expand(&desc, &config()).unwrap_err();
    assert!(err.to_string().contains("layers.0.wd"), "{err}");
}

#[test]
fn a_duplicate_instance_prefix_names_it() {
    let mut desc = tiny();
    let mut entry = desc.stack[0].clone();
    entry.prefix = "norm_in".to_string();
    desc.stack.insert(1, entry);
    let err = expand(&desc, &config()).unwrap_err();
    assert!(err.to_string().contains("norm_in"), "{err}");
}

#[test]
fn a_split_binding_feeds_several_slots() {
    let mut desc = tiny();

    // layers.*.qkv -> q | k：一条 source 拆成两个 slot。
    desc.binding
        .retain(|b| b.slot.as_deref() != Some("layers.*.qkv"));
    let decoder = desc.templates.get_mut("decoder").unwrap();
    decoder.slots.retain(|s| s.name != "qkv");
    decoder.slots.push(
        serde_json::from_str(
            r#"{ "name": "wq", "kind": "weight", "shape": ["hidden", "hidden"] }"#,
        )
        .unwrap(),
    );
    decoder.slots.push(
        serde_json::from_str(
            r#"{ "name": "wk", "kind": "weight", "shape": ["hidden", "hidden"] }"#,
        )
        .unwrap(),
    );
    let node = decoder
        .nodes
        .iter_mut()
        .find(|n| n.inputs == vec!["h1".to_string(), "qkv".to_string()])
        .expect("第一个 linear 消费 qkv");
    node.inputs[1] = "wq".to_string();
    // `wk` has to be read too: a declared slot no node touches is a dead hook (§3.7 #11), so the
    // second linear consumes it instead of `wo`.
    let node = decoder
        .nodes
        .iter_mut()
        .find(|n| n.inputs == vec!["qa".to_string(), "wo".to_string()])
        .expect("第二个 linear 消费 wo");
    node.inputs[1] = "wk".to_string();
    decoder.slots.retain(|s| s.name != "wo");
    desc.binding
        .retain(|b| b.slot.as_deref() != Some("layers.*.wo"));
    desc.binding.push(
        serde_json::from_str(
            r#"{
                "source": "model.layers.{*}.qkv.weight",
                "transform": ["transpose(0,1)"],
                "split": { "dim": 0, "sizes": ["hidden", "hidden"] },
                "targets": [
                    { "slot": "layers.*.wq", "axes": { "1": ["tp"] } },
                    { "slot": "layers.*.wk", "axes": { "1": ["tp"] } }
                ]
            }"#,
        )
        .unwrap(),
    );

    let expanded = expand(&desc, &config()).unwrap();
    let split = expanded
        .bindings
        .iter()
        .find(|b| b.split.is_some())
        .expect("split binding");
    assert_eq!(split.slots.len(), 6, "3 层 × 2 个 target");
    assert_eq!(split.split.as_ref().unwrap().sizes, vec![16, 16]);
    assert_eq!(
        split.slots[0].axes.get("1").unwrap(),
        &vec!["tp".to_string()]
    );
}

#[test]
fn a_duplicate_slot_name_names_both_origins() {
    let mut desc = tiny();
    let duplicate = desc.templates["norm"].slots[0].clone();
    desc.templates
        .get_mut("norm")
        .unwrap()
        .slots
        .push(duplicate);
    let err = expand(&desc, &config()).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("norm_in.w"), "{text}");
    assert!(text.contains("declared twice"), "{text}");
    assert!(text.contains("template slot `w`"), "{text}");
}

#[test]
fn a_node_writing_an_undeclared_slot_is_rejected() {
    let mut desc = tiny();
    let decoder = desc.templates.get_mut("decoder").unwrap();
    // `h9` 既不是模板 slot，也不是这个实例的 output：没有声明的地方，也没有可继承的形状。
    decoder.nodes[0].outputs = vec!["h9".to_string()];
    let err = expand(&desc, &config()).unwrap_err();
    assert!(matches!(err, ModelError::Invalid(_)), "{err}");
}

#[test]
fn the_undeclared_slot_error_names_the_template_and_the_name() {
    let mut desc = tiny();
    let decoder = desc.templates.get_mut("decoder").unwrap();
    decoder.nodes[0].outputs = vec!["h9".to_string()];
    let err = expand(&desc, &config()).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("decoder"), "报错必须指出模板名: {text}");
    assert!(text.contains("h9"), "报错必须指出未声明的名字: {text}");
}

#[test]
fn a_node_may_not_write_a_weight_slot() {
    let mut desc = tiny();
    let decoder = desc.templates.get_mut("decoder").unwrap();
    // 第一个 linear 的输出去写它自己读的权重 slot。
    let node = decoder
        .nodes
        .iter_mut()
        .find(|n| n.inputs == vec!["h1".to_string(), "qkv".to_string()])
        .unwrap();
    node.outputs = vec!["qkv".to_string()];
    let err = expand(&desc, &config()).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("qkv"), "{text}");
    assert!(text.contains("Weight"), "{text}");
}

#[test]
fn a_wrong_format_string_is_rejected() {
    let mut desc = tiny();
    desc.format = "rustrain.model.v2".to_string();
    match expand(&desc, &config()) {
        Err(ModelError::Format { found, expected }) => {
            assert_eq!(found, "rustrain.model.v2");
            assert_eq!(expected, "rustrain.model.v1");
        }
        other => panic!(
            "expected a format error, got {other:?}",
            other = other.err()
        ),
    }
}

#[test]
fn a_typo_in_the_description_is_not_silently_ignored() {
    let text = r#"{ "format": "rustrain.model.v1", "name": "x", "stak": [] }"#;
    let err = serde_json::from_str::<ModelDesc>(text).unwrap_err();
    assert!(err.to_string().contains("stak"), "{err}");
}

/// §3.7 #13: `select` and `template` would be two sources for the same fact, so an entry that
/// carries both is rejected instead of picking one of them.
#[test]
fn select_and_template_together_are_rejected_as_two_sources() {
    let mut desc = tiny();
    desc.stack[0].select = Some(
        serde_json::from_str(r#"{ "by": "layer_types[l]", "cases": { "full": "norm" } }"#).unwrap(),
    );
    desc.stack[0].template = Some("norm".to_string());
    let err = expand(&desc, &config()).unwrap_err();
    let text = err.to_string();
    assert!(
        text.contains("`select`") && text.contains("`template`"),
        "{text}"
    );
    assert!(text.contains("two"), "报错必须点明这是两个兜底来源: {text}");
}

/// The other half of the same ruling: with no `select`, `template` is still required.
#[test]
fn an_entry_with_neither_select_nor_template_is_rejected() {
    let mut desc = tiny();
    desc.stack[0].template = None;
    let err = expand(&desc, &config()).unwrap_err();
    let text = err.to_string();
    assert!(
        text.contains("template") && text.contains("select"),
        "{text}"
    );
    assert!(text.contains("norm_in"), "报错必须能定位到那一项: {text}");
}

/// §3.7 #11: a declared slot that no node reads and no node writes is a dead hook. The positive
/// half is the baseline `tiny()`, whose templates declare exactly the slots their nodes touch.
#[test]
fn a_slot_that_no_node_reads_or_writes_is_rejected() {
    assert!(expand(&tiny(), &config()).is_ok(), "声明齐全的模板必须通过");

    let mut desc = tiny();
    desc.templates.get_mut("norm").unwrap().slots.push(
        serde_json::from_str(r#"{ "name": "dead", "kind": "weight", "shape": ["hidden"] }"#)
            .unwrap(),
    );

    let err = expand(&desc, &config()).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("norm"), "报错必须指出模板名: {text}");
    assert!(text.contains("dead"), "报错必须指出那个 slot: {text}");
}

/// §3.7 #4: one checkpoint tensor feeds one slot. Two bindings on the same source would load it
/// into two places, and neither the expansion nor the load check would notice.
#[test]
fn two_bindings_on_one_source_are_rejected() {
    let mut desc = tiny();
    desc.binding[1].source = desc.binding[0].source.clone();
    let err = expand(&desc, &config()).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("binding 0"), "{text}");
    assert!(text.contains("binding 1"), "{text}");
    assert!(text.contains("model.norm_in.weight"), "{text}");
}

/// C6 gives `**` to `ignore` alone: in a binding it would pair one checkpoint tensor with a whole
/// subtree of slots (the report then contradicts its own `weights` counter).
#[test]
fn a_multi_segment_wildcard_in_a_binding_source_is_rejected() {
    let mut desc = tiny();
    desc.binding[0].source = "model.**.weight".to_string();
    let err = expand(&desc, &config()).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("model.**.weight"), "{text}");
    assert!(text.contains("ignore"), "{text}");
}

/// C6: an `ignore` entry that matches nothing is a warning at load-check time, but an entry with
/// no segment at all can never match anything, so it is an error (I-5).
#[test]
fn an_empty_ignore_pattern_is_rejected() {
    let mut desc = tiny();
    desc.ignore = vec![String::new()];
    let err = expand(&desc, &config()).unwrap_err();
    assert!(err.to_string().contains("ignore"), "{err}");

    let mut desc = tiny();
    desc.ignore = vec!["model..visual".to_string()];
    let err = expand(&desc, &config()).unwrap_err();
    assert!(err.to_string().contains("model..visual"), "{err}");

    // The syntax `ignore` is for: `**` spans any number of segments.
    let mut desc = tiny();
    desc.ignore = vec!["model.visual.**".to_string()];
    assert!(expand(&desc, &config()).is_ok());
}

/// F2 (C5): `ignore` is the *explicit* declaration of the tensors a description drops on purpose, so
/// every entry has to be anchored at a concrete segment. A pattern whose first segment is a wildcard
/// declares nothing in particular — `**` and `*` drop the whole checkpoint, `*.visual.**` drops
/// whatever the tower happens to be called — and three such spellings used to produce reports that
/// differed only in the number written next to "ignored".
#[test]
fn an_unanchored_ignore_pattern_is_rejected() {
    for pattern in ["**", "*", "{*}", "*.visual.**", "{*}.visual.**", "**.**"] {
        let mut desc = tiny();
        desc.ignore = vec![pattern.to_string()];
        let err = expand(&desc, &config()).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("ignore"), "{text}");
        assert!(
            text.contains(pattern),
            "the error must name the pattern: {text}"
        );
    }

    // The all-wildcard spelling has to say *why* it is not a declaration, not only that it is odd.
    let mut desc = tiny();
    desc.ignore = vec!["**".to_string()];
    let text = expand(&desc, &config()).unwrap_err().to_string();
    assert!(text.contains("explicit declaration"), "{text}");

    // An anchored pattern stays legal, `**` included: naming the subtree to drop is the point.
    for pattern in [
        "model",
        "model.**",
        "model.visual.**",
        "model.visual.*",
        "model.visual.weight",
    ] {
        let mut desc = tiny();
        desc.ignore = vec![pattern.to_string()];
        assert!(
            expand(&desc, &config()).is_ok(),
            "`{pattern}` must stay legal"
        );
    }
}

/// F6: two ways to claim one slot read differently, so the message has to tell them apart — one
/// binding naming the same target twice is a typo inside one entry, two bindings on one slot is a
/// conflict between two entries. The old wording printed the same source name on both sides
/// (`claimed by two bindings: s.a and s.a`) for the first case.
#[test]
fn a_slot_claimed_twice_says_which_of_the_two_conflicts_it_is() {
    // One binding, the same target twice. The `split` is only there because `targets` requires one
    // (§3.4); the duplicate is what the message has to name, before any shape is looked at.
    let mut desc = tiny();
    desc.binding[0] = serde_json::from_str(
        r#"{
            "source": "model.norm_in.weight",
            "split": { "dim": 0, "sizes": ["1", "1"] },
            "targets": [ { "slot": "norm_in.w" }, { "slot": "norm_in.w" } ]
        }"#,
    )
    .unwrap();
    let same = expand(&desc, &config()).unwrap_err().to_string();
    assert!(same.contains("binding 0"), "{same}");
    assert!(same.contains("names slot `norm_in.w` twice"), "{same}");
    assert!(same.contains("`norm_in.w` and `norm_in.w`"), "{same}");
    assert!(
        !same.contains("claimed by two bindings"),
        "one binding is not two bindings: {same}"
    );

    // Two bindings, one slot.
    let mut desc = tiny();
    desc.binding.push(
        serde_json::from_str(r#"{ "slot": "norm_in.w", "source": "model.norm_in.weight_again" }"#)
            .unwrap(),
    );
    let two = expand(&desc, &config()).unwrap_err().to_string();
    assert!(
        two.contains("slot `norm_in.w` is claimed by two bindings"),
        "{two}"
    );
    assert!(two.contains("model.norm_in.weight_again"), "{two}");
    assert!(
        two.contains("binding 0") && two.contains("binding 9"),
        "{two}"
    );
    assert!(
        !two.contains("twice"),
        "two bindings are not one binding: {two}"
    );
}
