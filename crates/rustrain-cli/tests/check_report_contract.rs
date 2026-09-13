//! C6's report contract, pinned: the shape of `rustrain check --json`, the exact set of check ids,
//! and which of them are `skip` on this machine (F3).
//!
//! `check_l2.rs` gates D2's *outcomes* (exit codes, counter values, the four rejections). It cannot
//! see a check that stopped being run: three of C2's L1 sub-checks are `skip` today because
//! `rustrain check` does not run `Plan::compile` yet (the compiler is D5's planner half, still
//! outstanding) — so a description that fails shape inference still reports `exit 0` and a
//! green-looking report, which a reviewer reproduced with a `matmul` declaring a rank-3 output.
//! This file closes the other half of the gate: the report must contain **exactly** C6's sixteen
//! ids, and exactly the ones expected to be `skip` must be `skip` — three compile-dependent skips
//! always, plus implementation availability at every dtype the reference provider rejects (bf16 /
//! f16 / the description's own bf16). At `--dtype f32` every node now resolves (moe_layer
//! included), so availability is a `pass` there. A check that starts running, stops running,
//! disappears or is renamed turns this file red, and the *status* of every id is pinned, so "still
//! skipped" cannot be mistaken for "still checked".
//!
//! It also pins **what the checks found**, not only that they ran: which model and checkpoint the
//! report is about, the dtype it ran at, the value of each of the eight counters, and the per-object
//! `details` of the two items that carry any. A pinned key set alone would not catch a report that
//! keeps every id and status while writing `nodes: null`, emptying the availability list, naming a
//! model it never read, or recording a `dtype` it never checked.
//!
//! What it deliberately does **not** claim: prose. The numbers this file reads are tied to each other
//! (counters ↔ details ↔ the reason's totals), but a `reason` sentence that contradicts its own
//! counters is not in scope — nor is a report whose numbers are fabricated rather than computed,
//! since they would match the constants here. `check_l2.rs`'s deliberately broken fixtures are what
//! stand against the latter.
//!
//! One limit worth naming: `model`/`checkpoint` are pinned to the paths the test *passed*, which is
//! what the report promises to echo — an implementation that echoed them while reading a different
//! description inside that directory would still be green. Catching that needs a probe with a
//! distinguishable witness, not a report-shape assertion.
//!
//! Everything goes through the real binary (`env!("CARGO_BIN_EXE_rustrain")`); no Rust internals.
//! Contract: `docs/design/qwen36-text/spec.md` C2 (the report and the Pass/Fail/Warning/Skip
//! discipline) + C6 (the report shape and the id list).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// C6's "check id 的完整清单": sixteen ids — the six C6 fixes in its report paragraph, plus the
/// ten this unit added (the six compile-dependent L1 sub-checks, description-side binding
/// coverage, the `ignore` coverage item, the argument check, and D4's `l1.instantiate`). This is
/// the whole vocabulary a report may use: an id outside it means the contract changed without
/// this test, and an id missing from the rejected-arguments run below means a check no longer
/// runs.
const C6_CHECK_IDS: [&str; 16] = [
    "cli.arguments",
    "l1.binding_coverage",
    "l1.collective_axes",
    "l1.compile",
    "l1.implementation_availability",
    "l1.instantiate",
    "l1.layout_propagation",
    "l1.operator_shapes",
    "l1.partial_fulfillment",
    "l1.slot_allocation",
    "l1.structure",
    "l2.binding_coverage",
    "l2.dtype_compatibility",
    "l2.ignore_coverage",
    "l2.shape_reconciliation",
    "l2.tensor_consumption",
];

/// The three L1 sub-checks that still need `Plan::compile`, which `rustrain check` does not run
/// yet (D5's planner half is outstanding) — a `skip` at every dtype, no matter how well
/// resolution goes. D4 turned the other three compile-dependent sub-checks real: with the mesh,
/// `instantiate` and the propagation pass run without any implementation, so
/// `l1.layout_propagation`, `l1.partial_fulfillment` and `l1.collective_axes` are `pass` on this
/// fixture now.
const COMPILE_DEPENDENT_SKIPS: [&str; 3] =
    ["l1.compile", "l1.operator_shapes", "l1.slot_allocation"];

/// The `skip` set of an accepted run **at `--dtype f32`**: every node of the real fixture resolves
/// (moe_layer included — D5's provider half is closed), so implementation availability is a
/// `pass` and only the three compile-dependent sub-checks are left.
const EXPECTED_SKIPS: [&str; 3] = COMPILE_DEPENDENT_SKIPS;

/// The `skip` set at bf16 / f16 (and the description's own bf16): the reference provider accepts
/// `f32` only, so every node is unresolved and implementation availability joins the three
/// compile-dependent skips as the fourth — with its reasons spelled out (C2 makes an unresolved
/// primitive a skip, never a fail and never a silent pass).
const BF16_EXPECTED_SKIPS: [&str; 4] = [
    "l1.compile",
    "l1.implementation_availability",
    "l1.operator_shapes",
    "l1.slot_allocation",
];

/// C6's eight counters **with their values on the real fixture**. The status table above says which
/// checks ran; this says what they found, and it is pinned for the same reason the statuses are:
/// `bindings`/`nodes`/`slots`/`weights` come from the description alone and are exact, and D2's
/// acceptance is the four zeros. A counter that stops being counted, or is emitted as `null`, keeps
/// the eight keys intact and would otherwise stay green.
///
/// Every value is an integer count; `as_i64` is deliberately strict, so `46.0` is a failure too.
const EXPECTED_COUNTS: [(&str, i64); 8] = [
    ("bindings", 46),
    ("dtype_mismatch", 0),
    ("nodes", 1285),
    ("shape_mismatch", 0),
    ("slots", 2222),
    ("slots_unbound", 0),
    ("tensors_unconsumed", 0),
    ("weights", 873),
];

/// Why one operator has no implementation on this host, as `l1.implementation_availability` says it:
/// either nothing publishes the primitive, or a provider exists whose variant rejects the dtype the
/// checks were asked to run at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Unavailable {
    /// No loaded plugin publishes the primitive at all — it is a missing primitive.
    NoProvider,
    /// A provider is loaded but its variant does not accept this dtype.
    DtypeRejected,
}

/// One `--dtype`'s availability list: every `(operator, node count, why)` a `skip` must carry, and
/// the total its `reason` must state. An **empty** table is the pass form: every node resolved,
/// no per-operator list, and the reason counts the whole plan as resolved.
///
/// The two lists differ by **cause**, not only by length. At `--dtype f32` nothing is left — with
/// `moe_layer` published (D5's provider half is closed), all 1285 nodes resolve. At the
/// description's own `bf16` — which is also what an explicit `--dtype bf16` or `f16` asks for —
/// every node whose `reference.f32` variant rejects that dtype is unresolved, which is the entire
/// plan (1285 of 1285 nodes over 16 operators, `moe_layer` among them). A list that is merely
/// *truncated* while its reason keeps saying `41 of 1285` is exactly the false green this pins
/// down.
struct Availability {
    entries: &'static [(&'static str, i64, Unavailable)],
    total: i64,
}

/// `--dtype f32`: every node resolves, so the availability item is the **pass** form — no
/// per-operator list, and the reason counts the whole plan.
const F32_AVAILABILITY: Availability = Availability {
    entries: &[],
    total: 0,
};

/// `bf16` (the description's dtype) or `f16`: the reference provider accepts `f32` only, so every
/// node of the plan is unresolved — `moe_layer` now published but rejected by dtype, like the
/// other fifteen.
const BF16_AVAILABILITY: Availability = Availability {
    entries: &[
        ("cat", 1, Unavailable::DtypeRejected),
        ("causal_conv1d", 90, Unavailable::DtypeRejected),
        ("elementwise_binary", 153, Unavailable::DtypeRejected),
        ("elementwise_unary", 101, Unavailable::DtypeRejected),
        ("embedding", 1, Unavailable::DtypeRejected),
        ("gated_delta_rule", 30, Unavailable::DtypeRejected),
        ("l2norm", 60, Unavailable::DtypeRejected),
        ("linear", 298, Unavailable::DtypeRejected),
        ("moe_layer", 41, Unavailable::DtypeRejected),
        ("narrow", 22, Unavailable::DtypeRejected),
        ("reshape", 276, Unavailable::DtypeRejected),
        ("rmsnorm", 108, Unavailable::DtypeRejected),
        ("rmsnorm_gated", 30, Unavailable::DtypeRejected),
        ("rope", 22, Unavailable::DtypeRejected),
        ("sdpa", 11, Unavailable::DtypeRejected),
        ("topk_router", 41, Unavailable::DtypeRejected),
    ],
    total: 1285,
};

/// Both tables must agree on which primitives nothing publishes (now: none — `moe_layer` is
/// published, so both lists' unpublished subsets are empty): the `f32` list is the subset of the
/// `bf16` one that nothing can do anything about.
fn unpublished_component<'a>(table: &'a [(&'a str, i64, Unavailable)]) -> Vec<(&'a str, i64)> {
    let mut out: Vec<(&'a str, i64)> = table
        .iter()
        .filter(|(_, _, why)| *why == Unavailable::NoProvider)
        .map(|(op, nodes, _)| (*op, *nodes))
        .collect();
    out.sort_unstable();
    out
}

/// The one `ignore` pattern of the description and how many tensors it covers.
///
/// A pattern that stops matching anything is a **`warning`** on `l2.ignore_coverage`, not a `fail`
/// (C6: the same description may be checked against another checkpoint). What *is* a `fail` is the
/// consequence: the tensors it stopped ignoring become unconsumed, and `check_l2.rs` gates exactly
/// that pair with `models/tiny-ignore-typo`. Here the count is pinned because it is what makes the
/// `pass` mean something on the real 1045-tensor snapshot.
const IGNORE_PATTERN: &str = "model.visual.**";
const IGNORED_TENSORS: i64 = 333;

/// **C5's witness at `pp = 1`**: the single stage is the whole plan, so the detail line pins
/// the global counts. A no-op `instantiate` (returning the global plan unchanged) would still
/// produce this line — which is why the five-axis run below pins the *pruned* per-stage
/// counts, where a no-op cannot hide.
const STAGE0_ONLY: [&str; 1] = ["stage 0 (rank 0): 1285 node(s), 2222 slot(s)"];

/// **C5's witness at the five-axis acceptance mesh**: the per-stage node/slot counts a real
/// `instantiate` produces after PP pruning. `instantiate` replaced by `return Ok(plan.clone())`
/// reports 1285 node(s)/2222 slot(s) for *both* stages and turns this gate red — the tripwire
/// the reviewer's no-op attack walks into.
const FIVE_AXIS_STAGES: [&str; 2] = [
    "stage 0 (rank 0): 626 node(s), 1083 slot(s)",
    "stage 1 (rank 32): 659 node(s), 1142 slot(s)",
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Pass,
    Fail,
    Warning,
    Skip,
}

impl Status {
    fn parse(text: &str) -> Option<Self> {
        match text {
            "pass" => Some(Status::Pass),
            "fail" => Some(Status::Fail),
            "warning" => Some(Status::Warning),
            "skip" => Some(Status::Skip),
            _ => None,
        }
    }
}

/// The status of every id the real fixture produces **when the arguments are accepted**, which is
/// the fifteen ids a normal run has: `cli.arguments` exists only to reject bad arguments.
///
/// This table is the **bf16** truth (the description's own dtype): the reference provider accepts
/// `f32` only, so `l1.implementation_availability` is a `skip` there. At `--dtype f32` every node
/// resolves and that one id flips to `pass` — [`expected_status_for`] derives the f32 table from
/// this one, one id at a time, never by editing the shared expectation.
///
/// Pinning the status is the point. `l1.structure` passing is not evidence that L1 ran: the three
/// compile-dependent sub-checks (which need `Plan::compile`, not run by `check` yet) and — at
/// bf16 — the availability check are the ones that would have caught a description that cannot
/// compile, and they are `skip` by contract — while `l1.instantiate` and the three propagation
/// checks D4 made real must be `pass` on this fixture, or the D4 headline acceptance silently
/// stopped running.
const EXPECTED_STATUS: [(&str, Status); 15] = [
    ("l1.structure", Status::Pass),
    ("l1.instantiate", Status::Pass),
    ("l1.compile", Status::Skip),
    ("l1.operator_shapes", Status::Skip),
    ("l1.layout_propagation", Status::Pass),
    ("l1.partial_fulfillment", Status::Pass),
    ("l1.collective_axes", Status::Pass),
    ("l1.slot_allocation", Status::Skip),
    ("l1.implementation_availability", Status::Skip),
    ("l1.binding_coverage", Status::Pass),
    ("l2.binding_coverage", Status::Pass),
    ("l2.tensor_consumption", Status::Pass),
    ("l2.ignore_coverage", Status::Pass),
    ("l2.shape_reconciliation", Status::Pass),
    ("l2.dtype_compatibility", Status::Pass),
];

/// The status table a run expects, derived from [`EXPECTED_STATUS`] with the one dtype-dependent
/// id overridden: implementation availability is a `pass` when the availability table is the
/// empty (all-resolved) form and a `skip` otherwise. Deriving one id from the shared table is
/// deliberate — the other fourteen ids are dtype-independent and stay pinned in one place.
fn expected_status_for(availability: &Availability) -> BTreeMap<String, Status> {
    let mut map: BTreeMap<String, Status> = EXPECTED_STATUS
        .iter()
        .map(|(id, status)| ((*id).to_string(), *status))
        .collect();
    let availability_status = if availability.entries.is_empty() {
        Status::Pass
    } else {
        Status::Skip
    };
    map.insert(
        "l1.implementation_availability".to_string(),
        availability_status,
    );
    map
}

/// The `skip` set a run expects: the three compile-dependent skips, plus implementation
/// availability exactly when that run's availability table is the non-empty (unresolved) form.
/// Both sets are named constants — `EXPECTED_SKIPS` is the f32 pin (4 → 3 the day moe_layer
/// landed), `BF16_EXPECTED_SKIPS` the dtype-rejected pin.
fn skip_set_for(availability: &Availability) -> &'static [&'static str] {
    if availability.entries.is_empty() {
        &EXPECTED_SKIPS
    } else {
        &BF16_EXPECTED_SKIPS
    }
}

fn cli_binary() -> &'static str {
    option_env!("CARGO_BIN_EXE_rustrain")
        .or(option_env!("CARGO_BIN_EXE_rustrain-cli"))
        .expect("cargo did not inject CARGO_BIN_EXE_<bin>: check rustrain-cli's binary target name")
}

/// D1's real description: the one model of this unit, and the one whose availability check has
/// something to skip.
fn qwen36_model_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../rustrain-model/tests/fixtures/qwen36-text")
}

/// The 1045-tensor snapshot D2 reconciles (shard headers only, no weights, no network).
fn qwen36_checkpoint() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/check-l2/checkpoints/qwen36-35b-a3b.safetensors.meta.json")
}

struct Run {
    code: Option<i32>,
    stdout: String,
}

impl Run {
    fn json(&self) -> Value {
        serde_json::from_str(&self.stdout).unwrap_or_else(|e| {
            panic!(
                "--json did not print one JSON report to stdout: {e}\n{}",
                self.stdout
            )
        })
    }
}

/// `rustrain check --model <qwen36> --checkpoint <snapshot> <extra args> --json`.
fn check(extra: &[&str]) -> Run {
    let output = Command::new(cli_binary())
        .arg("check")
        .arg("--model")
        .arg(qwen36_model_dir())
        .arg("--checkpoint")
        .arg(qwen36_checkpoint())
        .args(extra)
        .arg("--json")
        .output()
        .unwrap_or_else(|e| panic!("failed to start {}: {e}", cli_binary()));
    Run {
        code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
    }
}

/// One check item. C6 fixes the shape: `id`, `status`, `reason` are mandatory, `reason` is
/// non-empty, and `details` is always an array (empty when there is nothing per-object to list).
struct Item {
    id: String,
    status: Status,
    reason: String,
    details: Vec<String>,
}

/// Every check item, with the report's own text if an item is malformed.
fn items(doc: &Value) -> Vec<Item> {
    let checks = doc["checks"]
        .as_array()
        .unwrap_or_else(|| panic!("the report has no `checks` array: {doc}"));
    assert!(!checks.is_empty(), "the report has an empty `checks` array");

    checks
        .iter()
        .map(|item| {
            let id = item["id"]
                .as_str()
                .unwrap_or_else(|| panic!("a check has no `id`: {item}"))
                .to_string();
            let label = item["status"]
                .as_str()
                .unwrap_or_else(|| panic!("check `{id}` has no `status`: {item}"));
            let status = Status::parse(label).unwrap_or_else(|| {
                panic!("check `{id}` has status `{label}`; C6 allows pass|fail|warning|skip only")
            });
            let reason = item["reason"]
                .as_str()
                .unwrap_or_else(|| panic!("check `{id}` has no `reason`: {item}"));
            assert!(
                !reason.trim().is_empty(),
                "check `{id}` has an empty `reason`; C2 requires every Skip to say what is missing"
            );
            let details = item["details"]
                .as_array()
                .unwrap_or_else(|| {
                    panic!(
                        "check `{id}` has no `details` array; C6's report shape fixes it, empty when \
                         there is nothing per-object to list: {item}"
                    )
                })
                .iter()
                .map(|detail| {
                    detail
                        .as_str()
                        .unwrap_or_else(|| {
                            panic!("check `{id}` has a non-string `details` entry: {detail}")
                        })
                        .to_string()
                })
                .collect();
            Item {
                id,
                status,
                reason: reason.to_string(),
                details,
            }
        })
        .collect()
}

/// id → status for a report. C6 gives every id **at most one** item: when a check has several
/// per-object facts they go in that item's `details` (`l2.ignore_coverage` reports one line per
/// unmatched pattern inside one item, never one item per pattern). So two items with the same id are
/// a report that says two things about one check — and `assert_id_set` compares this map's *keys*,
/// which would make the duplicate disappear from the comparison.
fn statuses(items: &[Item]) -> BTreeMap<String, Status> {
    let mut map: BTreeMap<String, Status> = BTreeMap::new();
    for item in items {
        if let Some(previous) = map.insert(item.id.clone(), item.status) {
            panic!(
                "`{}` appears twice (as {previous:?} and {:?}); C6 gives every id one item and one \
                 status, with per-object facts in `details`",
                item.id, item.status
            );
        }
    }
    map
}

/// C6: the report's fixed shape — `format`, which model and checkpoint it is about, the dtype the
/// checks ran at, the eight counters **with their values**, and the per-item fields.
///
/// `expected_dtype` is pinned per run rather than merely asserted to be a string (C6 calls the field
/// "the dtype the checks ran at"): without `--dtype` the description's own `bf16` is used, and a
/// *rejected* `--dtype` falls back to it as well — the argument error is the `fail`, and the checks
/// still run at the declared precision.
fn assert_c6_shape(doc: &Value, expected_dtype: &str) {
    assert_eq!(
        doc["format"].as_str(),
        Some("rustrain.check.v1"),
        "C6 fixes the report format: {doc}"
    );
    assert_eq!(
        doc["dtype"].as_str(),
        Some(expected_dtype),
        "the report must record the dtype the checks actually ran at: {doc}"
    );

    // Both paths were passed to the binary, so the report has to name exactly them: a report about a
    // *different* model or checkpoint would otherwise satisfy every shape assertion here.
    assert_eq!(
        doc["model"].as_str(),
        Some(qwen36_model_dir().to_string_lossy().as_ref()),
        "the report must name the model directory it was given: {doc}"
    );
    assert_eq!(
        doc["checkpoint"].as_str(),
        Some(qwen36_checkpoint().to_string_lossy().as_ref()),
        "the report must name the checkpoint it was given: {doc}"
    );

    let counts = doc["counts"]
        .as_object()
        .unwrap_or_else(|| panic!("the report has no `counts` object: {doc}"));
    let mut keys: Vec<&str> = counts.keys().map(String::as_str).collect();
    keys.sort_unstable();
    let mut expected: Vec<&str> = EXPECTED_COUNTS.iter().map(|(key, _)| *key).collect();
    expected.sort_unstable();
    assert_eq!(
        keys, expected,
        "C6 fixes `counts` to exactly eight counters: {doc}"
    );

    // The values, not just the keys: a counter that is no longer counted, or is emitted as `null`
    // or `46.0`, keeps the key set and would otherwise pass every assertion in this file.
    for (key, value) in EXPECTED_COUNTS {
        let observed = counts.get(key).unwrap_or_else(|| {
            panic!("the key-set assertion above means `{key}` is present: {doc}")
        });
        assert_eq!(
            observed.as_i64(),
            Some(value),
            "counter `{key}` = {observed}, expected the integer {value}; C6's counters are what the \
             report found, and this test pins them: {doc}"
        );
    }
}

/// N1: the items that carry per-object `details`, and what the rest must not carry.
///
/// The status table says which checks ran; `assert_c6_shape` says what the counters are. This is the
/// third leg: the per-object lists that back the wording of a `pass` or a `skip`. `availability` is
/// the table the run's dtype must produce — an emptied list, a truncated list, a renamed operator, a
/// node count that drifts, a reason that enumerates a different set, or a `why` that does not match
/// its cause all turn it red — while every id and status stays exactly as pinned. `stage_lines` is
/// C5's witness: the `l1.instantiate` details must carry exactly the instantiated node and slot
/// counts per checked stage, one line per stage.
fn assert_details(doc: &Value, items: &[Item], availability: &Availability, stage_lines: &[&str]) {
    // The tables are constants; this makes a typo in either of them a self-inconsistency rather than
    // a re-definition of what the `skip`'s reason is asserted against.
    let sum: i64 = availability
        .entries
        .iter()
        .map(|(_, nodes, _)| *nodes)
        .sum();
    assert_eq!(
        sum, availability.total,
        "the availability table's entries do not sum to its own total"
    );
    assert!(
        availability.entries.iter().all(|(_, nodes, _)| *nodes > 0),
        "every listed operator covers at least one node: {:?}",
        availability.entries
    );

    let plan_nodes = doc["counts"]["nodes"]
        .as_i64()
        .unwrap_or_else(|| panic!("the plan size is a counter of the report: {doc}"));

    let mut seen_availability = false;
    let mut seen_ignore = false;
    let mut seen_instantiate = false;
    for item in items {
        match item.id.as_str() {
            "l1.instantiate" => {
                seen_instantiate = true;
                let expected: Vec<String> =
                    stage_lines.iter().map(|line| line.to_string()).collect();
                assert_eq!(
                    item.details, expected,
                    "C5: the `l1.instantiate` details must carry the instantiated node and slot \
                     counts per checked stage, in stage order — a no-op instantiate would report \
                     the global counts for every stage and turn this red: {:?}",
                    item.details
                );
            }
            "l1.implementation_availability" => {
                seen_availability = true;
                if availability.entries.is_empty() {
                    // The pass form (--dtype f32): every node resolved, so the item is a `pass`
                    // with NO per-operator list, and the reason counts the whole plan.
                    assert_eq!(
                        item.status,
                        Status::Pass,
                        "at a dtype every node resolves at, the availability item is a `pass`: \
                         {}",
                        item.reason
                    );
                    assert!(
                        item.details.is_empty(),
                        "the pass form carries no per-operator list; got: {:?}",
                        item.details
                    );
                    let lead = format!(
                        "all {} node(s) resolve to an implementation on this host",
                        plan_nodes
                    );
                    assert!(
                        item.reason.starts_with(&lead),
                        "the pass reason must open with `{lead}` — the number of resolved nodes \
                         out of the plan: {}",
                        item.reason
                    );
                } else {
                    assert_eq!(
                        item.status,
                        Status::Skip,
                        "the availability item carries the per-operator list and is a `skip` on \
                         this host at this dtype"
                    );
                    assert_eq!(
                        item.details.len(),
                        availability.entries.len(),
                        "the `skip` must list one entry per unavailable operator at this dtype; got: {:?}",
                        item.details
                    );
                    for (op, nodes, why_kind) in availability.entries {
                        let prefix = format!("{op}: {nodes} node(s)");
                        let entry = item
                            .details
                            .iter()
                            .find(|detail| detail.starts_with(&prefix))
                            .unwrap_or_else(|| {
                                panic!(
                                    "no `details` entry starts with `{prefix}`; the report says what is \
                                     missing per operator and C6 keeps that list machine-readable. \
                                     entries: {:?}",
                                    item.details
                                )
                            });
                        // `{op}: {n} node(s): {why}` — the count is only half of it. C2 makes the `Skip`
                        // answer *what is missing*, so the tail must state the cause the table gives,
                        // not merely be non-empty.
                        let why = entry[prefix.len()..].trim_start_matches(':').trim();
                        match why_kind {
                            Unavailable::NoProvider => assert_eq!(
                                why,
                                format!("no loaded plugin publishes `{op}`"),
                                "`{op}` is listed as a missing primitive, but its entry does not say so: \
                                 {entry}"
                            ),
                            Unavailable::DtypeRejected => assert!(
                                why.contains("is not accepted"),
                                "`{op}` has a provider that rejects this dtype, but its entry does not \
                                 say that: {entry}"
                            ),
                        }
                    }

                    // The reason inlines the same list it hands to `details` (`op ×count (why)`), so the
                    // two must be the *same* list. The `×` spelling is this report's own convention:
                    // if it is ever reworded deliberately, this assertion is the one to update, and the
                    // message below names the operator it could not find.
                    assert_eq!(
                        item.reason.matches('×').count(),
                        availability.entries.len(),
                        "the reason enumerates a different number of operators than `details` does: {}",
                        item.reason
                    );
                    for (op, nodes, _) in availability.entries {
                        assert!(
                            item.reason.contains(&format!("{op} ×{nodes}")),
                            "the reason does not carry the `{op} ×{nodes}` its `details` report: {}",
                            item.reason
                        );
                    }

                    // The reason's *leading* claim, not just "the plan size appears somewhere": a
                    // reason reading `41 of 425273 node(s) … (the plan has 1285)` must not pass.
                    let lead = format!("{} of {} node(s)", availability.total, plan_nodes);
                    assert!(
                        item.reason.starts_with(&lead),
                        "the reason must open with `{lead}` — the number of unresolved nodes out of the \
                         plan: {}",
                        item.reason
                    );
                }
            }
            "l2.ignore_coverage" => {
                seen_ignore = true;
                assert_eq!(
                    item.details.len(),
                    1,
                    "one `ignore` pattern, one detail line: {:?}",
                    item.details
                );
                let detail = &item.details[0];
                assert!(
                    detail.contains(IGNORE_PATTERN) && mentions_number(detail, IGNORED_TENSORS),
                    "the detail must name the pattern and how many tensors it covers \
                     (`{IGNORE_PATTERN}`, {IGNORED_TENSORS}): {detail}"
                );
                assert!(
                    mentions_number(&item.reason, IGNORED_TENSORS),
                    "the reason must carry the same total as its detail: {}",
                    item.reason
                );
            }
            id => assert!(
                item.details.is_empty(),
                "check `{id}` grew a `details` array and this test does not look at it; when a \
                 check starts reporting per-object facts (D3/D4 turn six `l1.*` items from `skip` \
                 into real checks), extend this function rather than leaving the facts unread: {:?}",
                item.details
            ),
        }
    }

    assert!(
        seen_availability && seen_ignore && seen_instantiate,
        "the items with per-object details must all be present (availability: \
         {seen_availability}, ignore coverage: {seen_ignore}, instantiate: {seen_instantiate})"
    );
}

/// `text` contains `n` as a whole number (so `90` is not found inside `1900`), which is how the
/// reasons and details are tied back to the pinned counters.
fn mentions_number(text: &str, n: i64) -> bool {
    let needle = n.to_string();
    text.split(|c: char| !c.is_ascii_digit())
        .any(|token| token == needle)
}

/// C6's id list: every id of the report comes from the list, and the list is covered exactly.
fn assert_id_set(observed: &BTreeMap<String, Status>, expected: &[&str], what: &str) {
    let mut expected = expected.to_vec();
    expected.sort_unstable();
    let observed_ids: Vec<&str> = observed.keys().map(String::as_str).collect();
    assert_eq!(
        observed_ids, expected,
        "{what}: the check id set must be exactly C6's list (nothing missing, nothing invented)"
    );
}

/// F3: the `skip` set of an accepted run is exactly what this machine is expected to have **at
/// the run's dtype** — the three compile-dependent sub-checks always, plus implementation
/// availability exactly when the reference provider rejects the dtype. If D5's planner half makes
/// one of the three real, or a description stops expanding, this fails and says which.
fn assert_skip_set(observed: &BTreeMap<String, Status>, expected: &[&str]) {
    let mut skips: Vec<&str> = observed
        .iter()
        .filter(|(_, status)| **status == Status::Skip)
        .map(|(id, _)| id.as_str())
        .collect();
    skips.sort_unstable();
    let mut expected = expected.to_vec();
    expected.sort_unstable();
    assert_eq!(
        skips, expected,
        "the `skip` set is not the one this machine is expected to have at this dtype"
    );
}

/// One **accepted** run: `check(&extra)` must exit 0, and the report must be C6's shape with exactly
/// the fifteen ids a normal run has and the statuses pinned per dtype (bf16: [`EXPECTED_STATUS`];
/// f32: the same table with availability a `pass`, derived in [`expected_status_for`]).
///
/// The per-object `details` are deliberately not part of this: availability is dtype-dependent (at
/// `--dtype f32` it is the empty pass form; at the description's own `bf16` every node whose
/// plugin variant rejects bf16 is listed), so each run pins its own.
fn accepted_run(
    extra: &[&str],
    expected_dtype: &str,
    availability: &Availability,
) -> (Value, Vec<Item>) {
    let run = check(extra);
    assert_eq!(
        run.code,
        Some(0),
        "an accepted run must exit 0 (no check is a `fail`)\n{}",
        run.stdout
    );

    let doc = run.json();
    assert_c6_shape(&doc, expected_dtype);
    let items = items(&doc);
    let observed = statuses(&items);

    // C6's sixteen ids, minus `cli.arguments`: that item exists to reject the arguments, and these
    // arguments are accepted. The rejected-arguments test below covers the sixteenth.
    let accepted: Vec<&str> = C6_CHECK_IDS
        .iter()
        .copied()
        .filter(|id| *id != "cli.arguments")
        .collect();
    assert_id_set(&observed, &accepted, "accepted arguments");
    let expected_skips = skip_set_for(availability);
    assert_skip_set(&observed, expected_skips);

    // Every id's status, not just the skips: a check that silently stops running is the failure
    // mode this file exists for.
    let expected = expected_status_for(availability);
    assert_eq!(
        observed, expected,
        "the status of at least one check changed; the expectation is pinned in EXPECTED_STATUS"
    );

    // The skips are pinned twice on purpose: as a set (F3's assertion) and inside the status
    // table. A table that drifted from the set would make one of the two red.
    let mut pinned: Vec<&str> = expected
        .iter()
        .filter(|(_, status)| **status == Status::Skip)
        .map(|(id, _)| id.as_str())
        .collect();
    pinned.sort_unstable();
    let mut skips_sorted = expected_skips.to_vec();
    skips_sorted.sort_unstable();
    assert_eq!(
        pinned, skips_sorted,
        "EXPECTED_STATUS and the expected skip set disagree"
    );

    (doc, items)
}

#[test]
fn the_report_has_c6s_shape_and_exactly_c6s_check_ids() {
    let (doc, items) = accepted_run(&["--dtype", "f32"], "f32", &F32_AVAILABILITY);
    assert_eq!(
        unpublished_component(F32_AVAILABILITY.entries),
        unpublished_component(BF16_AVAILABILITY.entries),
        "the f32 list must be exactly the unpublished primitives of the bf16 list: the same \
         primitives nothing publishes, with the same node counts"
    );
    assert_details(&doc, &items, &F32_AVAILABILITY, &STAGE0_ONLY);
}

/// C2's other half: `--dtype` is optional, and without it the **description's own** dtype is what
/// the checks ran at (the real fixture declares `bf16`).
///
/// This run exists because a dtype-conditional branch is invisible to the run that passes
/// `--dtype f32`: a report that emptied its per-operator list only when no dtype was overridden
/// would have satisfied every other assertion in this file.
#[test]
fn the_default_dtype_run_is_the_same_report_at_the_descriptions_own_dtype() {
    let (doc, items) = accepted_run(&[], "bf16", &BF16_AVAILABILITY);
    // Counts, ids, statuses and the skip set are already pinned by `accepted_run`: the plan and the
    // checkpoint do not depend on the dtype. What differs is the availability list, and at bf16 it
    // is the whole plan — every node of the reference provider rejects the dtype.
    assert_details(&doc, &items, &BF16_AVAILABILITY, &STAGE0_ONLY);
}

/// An **explicitly given** dtype is not the same code path as an omitted one, and `bf16` is the one
/// value where the two are easy to confuse: on this fixture both end up reporting `bf16`, but the
/// first goes through an override and the second through the description's declared dtype. On a
/// description that declares `f32` (the `tiny` fixtures) they differ outright, so a bug that only
/// fires on an explicit override — or only on an explicit `bf16`/`f16` — has no other gate here.
///
/// `f16` is pinned by the same table: the reference provider accepts `f32` alone, so every node is
/// unresolved for it too, with the same operators and the same counts.
#[test]
fn an_explicit_dtype_run_is_gated_at_every_dtype_it_accepts() {
    for dtype in ["bf16", "f16"] {
        let (doc, items) = accepted_run(&["--dtype", dtype], dtype, &BF16_AVAILABILITY);
        assert_details(&doc, &items, &BF16_AVAILABILITY, &STAGE0_ONLY);
    }
}

/// `cli.arguments` is the one id a normal run does not have, and C2 wants a full report even when
/// the arguments are rejected. A rejected `--dtype` therefore produces **all sixteen** ids, with
/// the other fifteen unchanged: an argument error must not silently change what was checked.
///
/// The counters are pinned here too (`assert_c6_shape` runs on both reports), because a rejected
/// argument must not change what was *found* either. The availability item is the one deliberate
/// exception: a rejected `--dtype` falls back to the description's own `bf16`, so availability is
/// a `skip` with the whole-plan list, while the accepted f32 run compared against it is a `pass` —
/// the dtype legitimately decides that one item (C2's `skip` still says what is missing, one
/// operator per line). The other fourteen statuses must not move, and the ignore coverage is
/// dtype-independent.
#[test]
fn a_rejected_argument_emits_the_sixteenth_id_and_changes_nothing_else() {
    let accepted = statuses(&items(&check(&["--dtype", "f32"]).json()));

    let run = check(&["--dtype", "not-a-dtype"]);
    assert_eq!(
        run.code,
        Some(1),
        "a rejected `--dtype` is a `fail`, so the exit code is 1\n{}",
        run.stdout
    );

    let doc = run.json();
    // The rejected dtype is not a dtype at all, so the checks fall back to the description's own
    // `bf16`; the exit code is already decided by the `cli.arguments` failure either way.
    assert_c6_shape(&doc, "bf16");
    let rejected_items = items(&doc);
    let rejected = statuses(&rejected_items);
    assert_id_set(&rejected, &C6_CHECK_IDS, "rejected arguments");
    assert_skip_set(&rejected, &BF16_EXPECTED_SKIPS);
    assert_eq!(
        rejected.get("cli.arguments"),
        Some(&Status::Fail),
        "the rejected argument must be the `cli.arguments` failure\n{}",
        run.stdout
    );

    for (id, status) in &accepted {
        if id == "l1.implementation_availability" {
            // The one status the dtype legitimately decides: the fallback run is bf16 (skip with
            // the whole-plan list), the accepted run is f32 (pass). See the doc comment.
            continue;
        }
        assert_eq!(
            rejected.get(id),
            Some(status),
            "`{id}` changed status because an unrelated argument was rejected\n{}",
            run.stdout
        );
    }

    // C2: the reason names the argument, so the report is actionable on its own.
    let argument = rejected_items
        .iter()
        .find(|item| item.id == "cli.arguments")
        .expect("cli.arguments must be present");
    assert!(
        argument.reason.contains("--dtype") && argument.reason.contains("not-a-dtype"),
        "the `cli.arguments` reason must name the rejected flag and value: {}",
        argument.reason
    );
}

/// **D4's headline acceptance.** The five-axis mesh `tp=2, cp=2, ep=4, dp=2, pp=2` must produce
/// exactly the same report an accepted run does: every id, every status, every counter, the same
/// availability result — and exit 0. `instantiate` runs for real on one representative rank per
/// stage (C5's witness pins the pruned per-stage counts), and the three propagation checks run
/// on stage 0 (rank 0), so the whole status table (with the four `pass` items D4 made real) is
/// re-pinned at the acceptance mesh, not only at the trivial all-ones mesh.
#[test]
fn the_five_axis_acceptance_mesh_exits_zero_with_the_same_report() {
    let (doc, items) = accepted_run(
        &[
            "--dtype", "f32", "--tp", "2", "--cp", "2", "--ep", "4", "--dp", "2", "--pp", "2",
        ],
        "f32",
        &F32_AVAILABILITY,
    );
    assert_details(&doc, &items, &F32_AVAILABILITY, &FIVE_AXIS_STAGES);
}

/// **D4's rejection acceptance.** `--tp 3` must exit non-zero and name the constraint the
/// description violates: `embed.w` shards dim 0 (the vocabulary, 248320 tokens) over `tp`, and
/// `248320 % 3 != 0` — the `NotDivisible` from `local_shape`, reported per slot, not a panic and
/// not a silent fallback. The three propagation checks then say why they could not run, and the
/// id set stays complete — a `fail` must not silently drop checks from the report.
#[test]
fn a_non_divisible_degree_exits_non_zero_naming_the_constraint() {
    let run = check(&["--dtype", "f32", "--tp", "3"]);
    assert_eq!(
        run.code,
        Some(1),
        "a non-divisible degree is a `fail`, so the exit code is 1\n{}",
        run.stdout
    );

    let doc = run.json();
    let items = items(&doc);
    let observed = statuses(&items);
    // The degrees are *accepted* — the failure is the description's non-divisible shard, so the
    // report carries the fifteen ids of an accepted run, not `cli.arguments`.
    let accepted: Vec<&str> = C6_CHECK_IDS
        .iter()
        .copied()
        .filter(|id| *id != "cli.arguments")
        .collect();
    assert_id_set(&observed, &accepted, "rejected degrees");

    let instantiate = items
        .iter()
        .find(|item| item.id == "l1.instantiate")
        .expect("l1.instantiate must be present");
    assert_eq!(
        instantiate.status,
        Status::Fail,
        "the non-divisible shard must be an `l1.instantiate` failure"
    );
    let text = format!("{} {}", instantiate.reason, run.stdout);
    assert!(
        text.contains("dim 0") && text.contains("248320") && text.contains('3'),
        "the failure must name the constraint — the dim, the global size and the divisor: {}",
        instantiate.reason
    );
    assert!(
        text.contains("embed.w"),
        "the failure names the slot, which names the constraint: {}",
        instantiate.reason
    );
    assert!(
        !run.stdout.contains("panicked at"),
        "the rejection is a report, not a panic\n{}",
        run.stdout
    );

    for id in [
        "l1.layout_propagation",
        "l1.partial_fulfillment",
        "l1.collective_axes",
    ] {
        assert_eq!(
            observed.get(id),
            Some(&Status::Skip),
            "`{id}` cannot run when `l1.instantiate` failed, and must say so"
        );
    }
}
