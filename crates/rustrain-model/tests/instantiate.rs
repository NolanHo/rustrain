//! D4 acceptance: `instantiate` — PP pruning by declared stage, the declared-axes × real-shape
//! join, divisibility as a named implementation-free failure, unknown axes, `tp = 1`, and the
//! stage declaration errors — all against the real Qwen3.6 text description
//! (`tests/fixtures/qwen36-text`) and small in-memory descriptions for the error paths.

use std::collections::BTreeSet;
use std::path::PathBuf;

use rustrain_model::{ModelDesc, expand};
use rustrain_parallel::{GroupMask, Mesh, ParallelConfig, ShardError};
use rustrain_plan::shard::propagate;
use rustrain_plan::{PlanError, SlotKind, instantiate};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen36-text")
}

fn expanded() -> rustrain_model::Expanded {
    rustrain_model::expand_dir(&fixture_dir()).expect("the real description must expand")
}

fn mesh(tp: usize, cp: usize, ep: usize, dp: usize, pp: usize) -> Mesh {
    Mesh::from_config(&ParallelConfig {
        tensor: tp,
        context: cp,
        expert: ep,
        data: dp,
        pipeline: pp,
    })
}

/// Instantiate the real description on `rank`.
fn instantiated(mesh: &Mesh, rank: usize) -> rustrain_plan::Plan {
    let expanded = expanded();
    instantiate(&expanded.plan, &expanded.declarations(), mesh, rank)
        .unwrap_or_else(|e| panic!("instantiate must succeed: {e}"))
}

/// The instance a node belongs to: the longest declared prefix the trace path starts with
/// (`emit_node` traces nodes as `{prefix}.{first output}`).
fn node_prefix(node: &rustrain_plan::PlanNode, declared: &rustrain_plan::DeclaredAxes) -> String {
    declared
        .instances
        .iter()
        .filter(|instance| {
            node.source.path == instance.prefix
                || node
                    .source
                    .path
                    .starts_with(&format!("{}.", instance.prefix))
        })
        .max_by_key(|instance| instance.prefix.len())
        .unwrap_or_else(|| {
            panic!(
                "node `{}` belongs to no declared instance",
                node.source.path
            )
        })
        .prefix
        .clone()
}

/// D4 acceptance: `pp = 2` keeps exactly the instances of the rank's stage.
///
/// Rank 0 (stage 0) keeps `embed` and layers 0–19; rank 1 (stage 1) keeps layers 20–39, `norm`,
/// `lm_head` and the three MTP instances. Asserted on **instance prefixes** (from the declared
/// stage list) and on the **boundary slots**: `layers.19.y` — the last layer-19 activation, the
/// only slot crossing the 19/20 seam — is a plan output on rank 0 and a plan input on rank 1,
/// and `embed.y` is a plan input on rank 1 (the MTP head consumes it).
#[test]
fn pp_pruning_keeps_exactly_the_rank_stage() {
    let m = mesh(1, 1, 1, 1, 2);
    let rank0 = instantiated(&m, 0);
    let rank1 = instantiated(&m, 1);

    let expanded = expanded();
    let declared = expanded.declarations();

    // Rank 0: the embed instance and layers 0–19, nothing else.
    let rank0_prefixes: BTreeSet<String> = rank0
        .nodes
        .iter()
        .map(|node| node_prefix(node, &declared))
        .collect();
    let expected0: BTreeSet<String> = std::iter::once("embed".to_string())
        .chain((0..20).map(|l| format!("layers.{l}")))
        .collect();
    assert_eq!(
        rank0_prefixes, expected0,
        "rank 0 (stage 0) must keep exactly embed + layers 0-19"
    );

    // Rank 1: the mirror image — layers 20–39, the final norm, the head, and all three MTP
    // instances (`mtp`, `mtp.layers.0`, `mtp.head`).
    let rank1_prefixes: BTreeSet<String> = rank1
        .nodes
        .iter()
        .map(|node| node_prefix(node, &declared))
        .collect();
    let expected1: BTreeSet<String> = (20..40)
        .map(|l| format!("layers.{l}"))
        .chain(["norm", "lm_head", "mtp", "mtp.layers.0", "mtp.head"].map(str::to_string))
        .collect();
    assert_eq!(
        rank1_prefixes, expected1,
        "rank 1 (stage 1) must keep exactly layers 20-39 + norm + lm_head + the MTP instances"
    );

    // Boundary slots: the seam between layer 19 and layer 20.
    // On rank 0 `layers.19.y` is produced but only stage 1 reads it → a plan output.
    let seam0 = rank0
        .slot_id("layers.19.y")
        .expect("rank 0 writes layers.19.y");
    assert_eq!(
        rank0.slot(seam0).kind,
        SlotKind::Output,
        "layers.19.y is read only by stage 1, so rank 0 hands it out as a plan output"
    );
    // On rank 1 it is written by stage 0 → a plan input.
    let seam1 = rank1
        .slot_id("layers.19.y")
        .expect("rank 1 reads layers.19.y");
    assert_eq!(
        rank1.slot(seam1).kind,
        SlotKind::Input,
        "layers.19.y is written by stage 0, so rank 1 receives it as a plan input"
    );

    // `embed.y` crosses too: stage 0 produces it, stage 1's MTP consumes it.
    let embed_y = rank1.slot_id("embed.y").expect("rank 1 reads embed.y");
    assert_eq!(
        rank1.slot(embed_y).kind,
        SlotKind::Input,
        "embed.y is written on stage 0 and read by mtp on stage 1 → a plan input on rank 1"
    );
    // On rank 0 it is produced *and* read (layer 0 chains off it) → an ordinary activation.
    let embed_y0 = rank0.slot_id("embed.y").expect("rank 0 writes embed.y");
    assert_eq!(rank0.slot(embed_y0).kind, SlotKind::Activation);

    // Stage-1-only and stage-0-only slots are absent on the other rank.
    assert!(
        rank0.slot_id("norm.w").is_none() && rank0.slot_id("lm_head.w").is_none(),
        "rank 0 must not carry the stage-1 weights"
    );
    assert!(
        rank0.slot_id("layers.20.input_layernorm").is_none(),
        "rank 0 must not carry layer 20"
    );
    assert!(
        rank1.slot_id("embed.w").is_none(),
        "rank 1 must not carry the stage-0 embedding table"
    );
    assert!(
        rank1.slot_id("layers.19.self_attn.qg").is_none(),
        "rank 1 must not carry layer 19"
    );
    // The MTP entries land on stage 1 only.
    assert!(
        rank0.slot_id("mtp.fc").is_none(),
        "rank 0 must not carry mtp"
    );
    assert!(
        rank1.slot_id("mtp.fc").is_some(),
        "rank 1 keeps the mtp instance"
    );

    // The final output is produced on stage 1.
    let logits = rank1
        .slot_id("lm_head.y")
        .expect("rank 1 produces lm_head.y");
    assert_eq!(rank1.slot(logits).kind, SlotKind::Output);
}

/// The fixture's 40-element stage list travels through `declarations()`: layers 0–19 on stage 0,
/// layers 20–39 on stage 1, everything else as R1 freezes.
#[test]
fn the_fixture_declares_its_forty_layer_stages_explicitly() {
    let declared = expanded().declarations();
    let stage_of = |prefix: &str| {
        declared
            .instances
            .iter()
            .find(|instance| instance.prefix == prefix)
            .unwrap_or_else(|| panic!("missing instance `{prefix}`"))
            .stage
    };

    assert_eq!(stage_of("embed"), Some(0));
    assert_eq!(stage_of("norm"), Some(1));
    assert_eq!(stage_of("lm_head"), Some(1));
    assert_eq!(stage_of("mtp"), Some(1));
    assert_eq!(stage_of("mtp.layers.0"), Some(1));
    assert_eq!(stage_of("mtp.head"), Some(1));

    let layers: Vec<Option<i64>> = declared
        .instances
        .iter()
        .filter(|instance| {
            instance.prefix.starts_with("layers.")
                && instance.prefix["layers.".len()..].parse::<usize>().is_ok()
        })
        .map(|instance| instance.stage)
        .collect();
    assert_eq!(layers.len(), 40, "all 40 layer instances are staged");
    assert_eq!(
        layers,
        (0..40)
            .map(|l| Some(if l < 20 { 0 } else { 1 }))
            .collect::<Vec<_>>(),
        "the stage list must be [0; 20] + [1; 20]"
    );
}

/// D4 acceptance: the real-shape join. The description's declared axes
/// (`binding[].targets[].axes`) × the real checkpoint shapes × the mesh — no hand-built
/// layout anywhere.
///
/// `mlp.experts.gate_proj` is `[E = 256, H = 2048, I = 512]` (the slot orientation of the
/// `[K, N]` convention, §3.4: per-expert `[H, I]`), declared `axes {0: [ep], 2: [tp]}` →
/// `[256/4, 2048, 512/2] = [64, 2048, 256]`. That is the transposed spelling of the frozen
/// figure `[64, 256, 2048]` — the same fact in checkpoint orientation `[E, I, H]`, which is the
/// shape the D3 test in `rustrain-parallel` pins; a description cannot write the slot as
/// `[E, I, H]` without breaking `[K, N]`. The frozen figure appears **verbatim** on
/// `mlp.experts.down_proj`, whose slot *is* `[E, I, H]` with `axes {0: [ep], 1: [tp]}` →
/// `[256/4, 512/2, 2048] = [64, 256, 2048]`.
#[test]
fn the_real_shape_join_resolves_declared_axes_end_to_end() {
    let m = mesh(2, 2, 4, 2, 2);
    let plan = instantiated(&m, 0); // pp coordinate 0 → layers 0–19 are present
    let ep = GroupMask::single(m.index_of("ep").unwrap()).unwrap();
    let tp = GroupMask::single(m.index_of("tp").unwrap()).unwrap();

    let gate = plan
        .slot_id("layers.3.mlp.experts.gate_proj")
        .expect("layer 3 is on stage 0");
    let gate_slot = plan.slot(gate);
    assert_eq!(
        gate_slot.shape,
        vec![64, 2048, 256],
        "gate_proj [256, 2048, 512] over axes {{0: ep, 2: tp}} must localize to [64, 2048, 256]"
    );
    assert_eq!(
        gate_slot.layout.dims,
        vec![
            rustrain_parallel::ShardSpec { dim: 0, group: ep },
            rustrain_parallel::ShardSpec { dim: 2, group: tp },
        ],
        "ep × tp is two independent specs on one tensor"
    );

    // The frozen figure, literally, on the slot whose orientation is [E, I, H].
    let down = plan
        .slot_id("layers.3.mlp.experts.down_proj")
        .expect("layer 3 is on stage 0");
    let down_slot = plan.slot(down);
    assert_eq!(
        down_slot.shape,
        vec![64, 256, 2048],
        "down_proj [256, 512, 2048] over axes {{0: ep, 1: tp}} must localize to [64, 256, 2048]"
    );
    assert_eq!(
        down_slot.layout.dims,
        vec![
            rustrain_parallel::ShardSpec { dim: 0, group: ep },
            rustrain_parallel::ShardSpec { dim: 1, group: tp },
        ]
    );

    // The same declared axes divide the embedding table on its own axis.
    let embed = plan.slot_id("embed.w").expect("embed is on stage 0");
    assert_eq!(
        plan.slot(embed).shape,
        vec![124160, 2048],
        "embed.w [248320, 2048] over {{0: tp}} must localize to [124160, 2048]"
    );

    // Stage 1 (rank 63 = pp coordinate 1) instantiates too: its MoE layers localize identically
    // — shapes and layouts are a function of the global plan, only the node set depends on the
    // rank's stage.
    let rank1 = instantiated(&m, 63);
    assert!(
        rank1.slot_id("embed.w").is_none(),
        "stage 1 has no embedding table"
    );
    let gate1 = rank1
        .slot_id("layers.25.mlp.experts.gate_proj")
        .expect("layer 25 is on stage 1");
    assert_eq!(rank1.slot(gate1).shape, vec![64, 2048, 256]);
}

/// D4 acceptance: a declared shard that does not divide is a hard error naming dim / global
/// size / divisor — raised by `instantiate`, which never touches the registry, so it fires
/// even where no implementation exists.
///
/// With `tp = 3` the very first declared slot already violates it: `embed.w` shards dim 0
/// (`vocab = 248320`) and `248320 % 3 != 0`.
#[test]
fn non_divisible_shard_is_a_named_implementation_free_failure() {
    let expanded = expanded();
    let m = mesh(3, 1, 1, 1, 1);
    let err = instantiate(&expanded.plan, &expanded.declarations(), &m, 0).unwrap_err();

    match &err {
        PlanError::Instantiate {
            slot,
            source:
                ShardError::NotDivisible {
                    dim,
                    global,
                    divisor,
                },
        } => {
            assert_eq!(*dim, 0, "the failure is on the vocab axis");
            assert_eq!(*global, 248320, "the global size is the vocab size");
            assert_eq!(*divisor, 3, "the divisor is the tp degree");
            assert!(
                slot.contains("embed"),
                "the error names the slot, which names the constraint: {slot}"
            );
        }
        other => panic!("expected NotDivisible naming dim/global/divisor, got {other:?}"),
    }

    // The message carries all three numbers: the caller reads the constraint off it.
    let text = err.to_string();
    assert!(text.contains("dim 0"), "{text}");
    assert!(text.contains("248320"), "{text}");
    assert!(text.contains("3"), "{text}");
}

/// D4 acceptance: a declaration naming an axis the mesh does not have is reported naming the
/// axis and the slot — the name-level counterpart of `GroupUnavailable` (a mask only exists
/// once the name resolves, so a missing name is reported before any mask is formed).
#[test]
fn an_axis_the_mesh_does_not_have_names_the_axis_and_slot() {
    let expanded = expanded();
    let mut declared = expanded.declarations();
    declared
        .slots
        .get_mut("embed.w")
        .expect("embed.w declares axes")
        .get_mut("0")
        .expect("dim 0 is declared")[0] = "vpp".to_string();

    let m = mesh(2, 1, 1, 1, 1); // the canonical five axes; no `vpp`
    let err = instantiate(&expanded.plan, &declared, &m, 0).unwrap_err();
    match err {
        PlanError::UnknownAxis {
            slot, dim, axis, ..
        } => {
            assert_eq!(slot, "embed.w");
            assert_eq!(dim, "0");
            assert_eq!(axis, "vpp");
        }
        other => panic!("expected UnknownAxis naming axis and slot, got {other:?}"),
    }
}

/// D4 acceptance: `tp = 1` is a legal size-1 group — the same instantiation succeeds, and every
/// layout over `tp` divides by 1 (its degree is 1).
#[test]
fn tp_one_is_a_legal_size_one_group() {
    let expanded = expanded();
    let m = mesh(1, 1, 1, 1, 1);
    let plan = instantiate(&expanded.plan, &expanded.declarations(), &m, 0)
        .expect("tp = 1 must instantiate");

    let tp = m.index_of("tp").unwrap();
    let mut over_tp = 0usize;
    for slot in &plan.slots {
        for spec in &slot.layout.dims {
            if spec.group.contains(tp) {
                over_tp += 1;
                assert_eq!(
                    spec.group.degree(&m).unwrap(),
                    1,
                    "a shard over tp with tp = 1 is a size-1 group (slot {})",
                    slot.name
                );
            }
        }
        if let Some(partial) = &slot.layout.partial
            && partial.group.contains(tp)
        {
            over_tp += 1;
            assert_eq!(partial.group.degree(&m).unwrap(), 1, "slot {}", slot.name);
        }
    }
    assert!(
        over_tp > 0,
        "the description declares tp shards, so the plan must actually carry them"
    );

    // Division by 1 leaves the global shape: the declared layout exists, the tensor is whole.
    let gate = plan.slot_id("layers.3.mlp.experts.gate_proj").unwrap();
    assert_eq!(plan.slot(gate).shape, vec![256, 2048, 512]);
}

/// D4 acceptance (error path): with `pp = 2`, an instance that declares no stage is an error
/// naming the entry — `lm_head` on stage 0 by silence is exactly the guess the framework
/// refuses.
#[test]
fn pp_above_one_without_a_declared_stage_names_the_entry() {
    let expanded = expanded();
    let mut declared = expanded.declarations();
    for instance in &mut declared.instances {
        if instance.prefix == "lm_head" {
            instance.stage = None;
        }
    }
    let m = mesh(1, 1, 1, 1, 2);
    let err = instantiate(&expanded.plan, &declared, &m, 0).unwrap_err();
    match err {
        PlanError::MissingStage { prefix, pp } => {
            assert_eq!(prefix, "lm_head");
            assert_eq!(pp, 2);
        }
        other => panic!("expected MissingStage naming the entry, got {other:?}"),
    }
}

/// A stage list whose length does not match the entry's instance count is an error naming the
/// entry (its prefix) and both counts, raised by `expand` before any plan is built.
#[test]
fn a_stage_list_must_have_one_entry_per_instance() {
    let desc: ModelDesc = serde_json::from_str(
        r#"{
        "format": "rustrain.model.v1",
        "name": "staged",
        "dtype": "f32",
        "inputs": { "hidden_in": { "shape": ["h"], "kind": "input" } },
        "params": { "h": { "expr": "16" }, "layers": { "expr": "3" } },
        "templates": {
            "norm": {
                "inputs": { "x": { "shape": ["h"], "kind": "activation" } },
                "outputs": { "y": { "shape": ["h"], "kind": "activation" } },
                "slots": [ { "name": "w", "kind": "weight", "shape": ["h"] } ],
                "nodes": [ { "op": "rmsnorm", "in": ["x", "w"], "out": ["y"] } ]
            }
        },
        "stack": [
            { "template": "norm", "prefix": "layers.{l}",
              "repeat": { "count": "layers", "index": "l" },
              "stage": [0, 0],
              "inputs": { "x": "hidden_in" } }
        ],
        "binding": [ { "slot": "layers.*.w", "source": "model.layers.{*}.w" } ]
    }"#,
    )
    .unwrap();

    let err = expand(&desc, &serde_json::json!({})).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("layers.{l}"), "must name the entry: {text}");
    assert!(text.contains('2'), "must name the declared count: {text}");
    assert!(text.contains('3'), "must name the instance count: {text}");
}

/// The positive half of the stage parser: an integer stage covers every instance, a list is
/// indexed by the repeat counter, and both travel through `declarations()`.
#[test]
fn declared_stages_travel_through_declarations() {
    let desc: ModelDesc = serde_json::from_str(
        r#"{
        "format": "rustrain.model.v1",
        "name": "staged-ok",
        "dtype": "f32",
        "inputs": { "hidden_in": { "shape": ["h"], "kind": "input" } },
        "params": { "h": { "expr": "16" }, "layers": { "expr": "3" } },
        "templates": {
            "norm": {
                "inputs": { "x": { "shape": ["h"], "kind": "activation" } },
                "outputs": { "y": { "shape": ["h"], "kind": "activation" } },
                "slots": [ { "name": "w", "kind": "weight", "shape": ["h"] } ],
                "nodes": [ { "op": "rmsnorm", "in": ["x", "w"], "out": ["y"] } ]
            }
        },
        "stack": [
            { "template": "norm", "prefix": "layers.{l}",
              "repeat": { "count": "layers", "index": "l" },
              "stage": [0, 1, 0],
              "inputs": { "x": "hidden_in" } }
        ],
        "binding": [ { "slot": "layers.*.w", "source": "model.layers.{*}.w" } ]
    }"#,
    )
    .unwrap();

    let declared = expand(&desc, &serde_json::json!({}))
        .unwrap()
        .declarations();
    let stages: Vec<Option<i64>> = declared.instances.iter().map(|i| i.stage).collect();
    assert_eq!(
        stages,
        vec![Some(0), Some(1), Some(0)],
        "the stage list indexes instances in expansion order"
    );

    // And it instantiates: pp = 2, rank 1 keeps only layer 1.
    let m = mesh(1, 1, 1, 1, 2);
    let plan = instantiate(
        &expand(&desc, &serde_json::json!({})).unwrap().plan,
        &declared,
        &m,
        1,
    )
    .unwrap();
    let kept: Vec<String> = plan
        .slots
        .iter()
        .filter(|slot| slot.name.starts_with("layers."))
        .map(|slot| slot.name.clone())
        .collect();
    // Every instance wires `x` to `hidden_in` explicitly (per-entry wiring), so no layer
    // output crosses the stage seam here — the crossing paths are covered by the real
    // fixture's `pp_pruning_keeps_exactly_the_rank_stage`.
    assert_eq!(
        kept,
        vec!["layers.1.w".to_string(), "layers.1.y".to_string()],
        "rank 1 keeps only the layer whose declared stage is 1"
    );
    assert_eq!(
        plan.slot_id("layers.0.w"),
        None,
        "layer 0's weight is not read on stage 1"
    );
    assert_eq!(
        plan.slot_id("layers.0.y"),
        None,
        "layer 0's output is not read on stage 1"
    );
    assert!(
        plan.slot_id("hidden_in").is_some(),
        "hidden_in feeds every layer, so stage 1 keeps it"
    );
}

/// `declarations()` exposes the R2 data: slot **names** → (logical dim → axis names), read
/// straight off the description's `binding[].targets[].axes`.
#[test]
fn declarations_expose_the_binding_axes_by_slot_name() {
    let declared = expanded().declarations();

    let gate = declared
        .slots
        .get("layers.3.mlp.experts.gate_proj")
        .expect("gate_proj declares axes");
    assert_eq!(gate.get("0").unwrap(), &vec!["ep".to_string()]);
    assert_eq!(gate.get("2").unwrap(), &vec!["tp".to_string()]);

    let o_proj = declared
        .slots
        .get("layers.3.self_attn.o_proj")
        .expect("o_proj declares axes");
    assert_eq!(o_proj.get("0").unwrap(), &vec!["tp".to_string()]);

    // A slot without declared axes is absent from the map, not an empty entry.
    assert!(!declared.slots.contains_key("input_ids"));
    assert!(!declared.slots.contains_key("layers.3.mlp.gate"));
}

/// D4 acceptance prerequisite: an instantiated **real** plan must compile, and the vocabulary-sharded
/// embedding table is where that first showed a hole.
///
/// `embed.w` is declared `{0: tp}` — the vocabulary axis. A lookup is a linear over its table (the
/// ids select rows), so the same rule that makes a row-parallel `linear` owe a `partial(sum, tp)`
/// applies here: each rank can only contribute the ids whose rows it owns, and the full activation
/// is the sum over the group. The rule table therefore classified `embedding` as `Elementwise`,
/// which asks for the *activation's* layout (replicate for ids) and makes the declared shard on the
/// table an unconvertible conflict.
///
/// This test drives the whole path — description → instantiate → compile — and asserts both halves:
/// the instantiated `embed.y` is a `partial(sum, tp)`, and `compile` materializes it with an
/// `all_reduce` over `tp`.
#[test]
fn a_vocabulary_sharded_embedding_compiles_and_owes_an_all_reduce() {
    use rustrain_parallel::{GroupMask, ReduceOp};

    let mesh = mesh(2, 1, 1, 1, 1);
    let plan = instantiated(&mesh, 0);

    let tp = GroupMask::single(mesh.index_of("tp").expect("canonical mesh has tp")).unwrap();
    let embed_y = plan
        .slots
        .iter()
        .find(|slot| slot.name == "embed.y")
        .expect("the embedding activation slot");
    assert_eq!(
        embed_y.layout.partial.as_ref().map(|p| (p.op, p.group)),
        Some((ReduceOp::Sum, tp)),
        "a vocab-sharded embedding owes a partial sum over tp, not a conflict: {:?}",
        embed_y.layout
    );

    // The propagation pass must be able to express that layout's conversion: it is the compile step
    // that failed before the rule table learned this op.
    let propagation = propagate(&plan).expect("the instantiated plan must propagate");
    assert!(
        propagation
            .inserted
            .iter()
            .any(|c| c.op == "intrinsic.all_reduce" && c.group == tp),
        "propagate must insert the all_reduce that fulfils the embedding's partial: {:?}",
        propagation
            .inserted
            .iter()
            .map(|c| (c.op, c.group))
            .collect::<Vec<_>>()
    );
}

/// D4 acceptance: the whole real path on the five-axis acceptance mesh `tp=2, cp=2, ep=4, dp=2,
/// pp=2`, rank 0 and the last rank.
///
/// Rank 0 (stage 0) instantiates **and** propagates: 26 collectives, the first of them the
/// all-reduce that fulfils `embed.y`'s partial over `tp`. The last rank (stage 1) instantiates too
/// — shapes and layouts are a function of the global plan, only the node set is the rank's stage.
///
/// What stage 1 cannot do yet is propagate, and the refusal is pinned here on purpose: `embed.y`
/// crosses the pipeline seam as a plan **input** carrying `partial(sum, tp)` (stage 0's embedding
/// owes the partial; stage 0's own walk also inserts the completing all-reduce on its side), and
/// the mtp head's first rmsnorm needs `replicate`. The conversion's owner is off-rank — where the
/// all-reduce sits relative to the stage handoff (before the send on stage 0, or after the receive
/// on stage 1) is the cross-stage communication decision D4 deliberately does not make. The honest
/// answer today is the D3 refusal "on a model input, which no node can convert", not a guessed
/// placement; when D5 lands the seam, this assertion moves with it.
#[test]
fn the_real_description_instantiates_and_propagates_on_the_acceptance_mesh() {
    let m = mesh(2, 2, 4, 2, 2);
    let tp = GroupMask::single(m.index_of("tp").expect("canonical mesh has tp")).unwrap();

    // Rank 0: instantiate and propagate, with the embedding's all-reduce first.
    let rank0 = instantiated(&m, 0);
    let propagation = propagate(&rank0).expect("rank 0 must propagate at the acceptance mesh");
    assert_eq!(
        propagation.inserted.len(),
        26,
        "rank 0 inserts the embedding's all-reduce, one per layer's row-parallel projection and \
         one gather per full-attention layer's gate: {:?}",
        propagation
            .inserted
            .iter()
            .map(|c| (c.op, c.group))
            .collect::<Vec<_>>()
    );
    assert!(
        propagation
            .inserted
            .iter()
            .any(|c| c.op == "intrinsic.all_reduce" && c.group == tp),
        "the embedding's partial must be fulfilled by an all_reduce over tp"
    );

    // The last rank instantiates: the stage-1 node set, local shapes intact.
    let rank_last = instantiated(&m, m.world_size() - 1);
    assert!(
        rank_last
            .slot_id("layers.25.mlp.experts.gate_proj")
            .is_some(),
        "the last rank keeps its stage's layers"
    );
    let embed = rank_last.slot_id("embed.y").expect("stage 1 reads embed.y");
    assert_eq!(
        rank_last.slot(embed).kind,
        SlotKind::Input,
        "embed.y crosses the seam as a plan input"
    );

    // The seam refusal: the stage-1 input holds the partial, and the conversion's owner is on
    // stage 0 — see the doc comment above for why this is a D5 decision, not a D4 bug.
    let err = propagate(&rank_last).expect_err("stage 1 cannot propagate until the seam lands");
    match err {
        PlanError::LayoutConflict { held, .. } => {
            assert!(
                held.contains("on a model input, which no node can convert"),
                "the refusal must name the seam problem: {held}"
            );
        }
        other => panic!("expected the seam layout conflict, got {other:?}"),
    }
}
