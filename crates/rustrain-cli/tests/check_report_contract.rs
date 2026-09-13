//! C6's report contract, pinned: the shape of `rustrain check --json`, the exact set of check ids,
//! and which of them are `skip` on this machine (F3).
//!
//! `check_l2.rs` gates D2's *outcomes* (exit codes, counter values, the four rejections). It cannot
//! see a check that stopped being run: six of C2's L1 sub-checks are `skip` today because
//! `Plan::compile` needs a mesh (D3) and an implementation to ask for shapes, so a description that
//! fails shape inference still reports `exit 0` and a green-looking report — a reviewer reproduced
//! exactly that with a `matmul` declaring a rank-3 output. This file closes the other half of the
//! gate: the report must contain **exactly** C6's fifteen ids, and exactly the seven of them that
//! are expected to be `skip` must be `skip`. A check that starts running, stops running, disappears
//! or is renamed turns this file red, and the *status* of every id is pinned, so "still skipped"
//! cannot be mistaken for "still checked".
//!
//! Everything goes through the real binary (`env!("CARGO_BIN_EXE_rustrain")`); no Rust internals.
//! Contract: `docs/design/qwen36-text/spec.md` C2 (the report and the Pass/Fail/Warning/Skip
//! discipline) + C6 (the report shape and the id list).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// C6's "check id 的完整清单": fifteen ids — the six C6 fixes in its report paragraph, plus the nine
/// this unit added (the six compile-dependent L1 sub-checks, description-side binding coverage, the
/// `ignore` coverage item, and the argument check). This is the whole vocabulary a report may use:
/// an id outside it means the contract changed without this test, and an id missing from the
/// rejected-arguments run below means a check no longer runs.
const C6_CHECK_IDS: [&str; 15] = [
    "cli.arguments",
    "l1.binding_coverage",
    "l1.collective_axes",
    "l1.compile",
    "l1.implementation_availability",
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

/// The six C2 sub-checks that need `Plan::compile` (a mesh: D3/D4), plus implementation
/// availability — five primitives of this description have no implementation on this host, which
/// C2 makes a `skip`, never a `fail` and never a silent `pass`.
const EXPECTED_SKIPS: [&str; 7] = [
    "l1.collective_axes",
    "l1.compile",
    "l1.implementation_availability",
    "l1.layout_propagation",
    "l1.operator_shapes",
    "l1.partial_fulfillment",
    "l1.slot_allocation",
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
/// the fourteen ids a normal run has: `cli.arguments` exists only to reject bad arguments.
///
/// Pinning the status is the point. `l1.structure` passing is not evidence that L1 ran: the six
/// compile-dependent sub-checks and the availability check are the ones that would have caught a
/// description that cannot compile, and they are `skip` today by contract.
const EXPECTED_STATUS: [(&str, Status); 14] = [
    ("l1.structure", Status::Pass),
    ("l1.compile", Status::Skip),
    ("l1.operator_shapes", Status::Skip),
    ("l1.layout_propagation", Status::Skip),
    ("l1.partial_fulfillment", Status::Skip),
    ("l1.collective_axes", Status::Skip),
    ("l1.slot_allocation", Status::Skip),
    ("l1.implementation_availability", Status::Skip),
    ("l1.binding_coverage", Status::Pass),
    ("l2.binding_coverage", Status::Pass),
    ("l2.tensor_consumption", Status::Pass),
    ("l2.ignore_coverage", Status::Pass),
    ("l2.shape_reconciliation", Status::Pass),
    ("l2.dtype_compatibility", Status::Pass),
];

/// C6's eight counters, exactly.
const COUNT_KEYS: [&str; 8] = [
    "slots",
    "nodes",
    "weights",
    "bindings",
    "slots_unbound",
    "tensors_unconsumed",
    "shape_mismatch",
    "dtype_mismatch",
];

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

/// The 1045-tensor snapshot D2 reconciles (26 shard headers, no weights, no network).
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

/// Every check item, as `(id, status, reason)`, with the report's own error text if an item does
/// not carry all three fields (C6: `id`, `status`, `reason` are mandatory and `reason` is non-empty).
fn items(doc: &Value) -> Vec<(String, Status, String)> {
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
            assert!(
                item["details"].is_array(),
                "check `{id}` has no `details` array; C6's report shape fixes it, empty when there \
                 is nothing per-object to list: {item}"
            );
            (id, status, reason.to_string())
        })
        .collect()
}

/// id → status for a report; two items with the same id (C6 lets `l2.ignore_coverage` report one
/// warning per unmatched pattern) must agree, or the report contradicts itself.
fn statuses(items: &[(String, Status, String)]) -> BTreeMap<String, Status> {
    let mut map: BTreeMap<String, Status> = BTreeMap::new();
    for (id, status, _) in items {
        if let Some(previous) = map.insert(id.clone(), *status) {
            assert_eq!(
                previous, *status,
                "`{id}` appears twice with different statuses ({previous:?} vs {status:?})"
            );
        }
    }
    map
}

/// C6: the report's fixed shape — `format`, the eight counters, and the per-item fields.
fn assert_c6_shape(doc: &Value) {
    assert_eq!(
        doc["format"].as_str(),
        Some("rustrain.check.v1"),
        "C6 fixes the report format: {doc}"
    );
    assert!(
        doc["dtype"].is_string(),
        "C6: the report records the dtype the checks ran at: {doc}"
    );

    let counts = doc["counts"]
        .as_object()
        .unwrap_or_else(|| panic!("the report has no `counts` object: {doc}"));
    let mut keys: Vec<&str> = counts.keys().map(String::as_str).collect();
    keys.sort_unstable();
    let mut expected = COUNT_KEYS.to_vec();
    expected.sort_unstable();
    assert_eq!(
        keys, expected,
        "C6 fixes `counts` to exactly eight counters: {doc}"
    );
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

/// F3: on this machine the six compile-dependent L1 sub-checks and implementation availability are
/// `skip` — no more, no fewer. If D3/D4 make one of them real, or a description stops expanding,
/// this fails and says which.
fn assert_skip_set(observed: &BTreeMap<String, Status>) {
    let mut skips: Vec<&str> = observed
        .iter()
        .filter(|(_, status)| **status == Status::Skip)
        .map(|(id, _)| id.as_str())
        .collect();
    skips.sort_unstable();
    let mut expected = EXPECTED_SKIPS.to_vec();
    expected.sort_unstable();
    assert_eq!(
        skips, expected,
        "the `skip` set is not the one this machine is expected to have"
    );
}

#[test]
fn the_report_has_c6s_shape_and_exactly_c6s_check_ids() {
    let run = check(&["--dtype", "f32"]);
    assert_eq!(
        run.code,
        Some(0),
        "the accepted-arguments run must exit 0 (no check is a `fail`)\n{}",
        run.stdout
    );

    let doc = run.json();
    assert_c6_shape(&doc);
    let items = items(&doc);
    let observed = statuses(&items);

    // C6's fifteen ids, minus `cli.arguments`: that item exists to reject the arguments, and these
    // arguments are accepted. The rejected-arguments test below covers the fifteenth.
    let accepted: Vec<&str> = C6_CHECK_IDS
        .iter()
        .copied()
        .filter(|id| *id != "cli.arguments")
        .collect();
    assert_id_set(&observed, &accepted, "accepted arguments");
    assert_skip_set(&observed);

    // Every id's status, not just the skips: a check that silently stops running is the failure
    // mode this file exists for.
    let expected: BTreeMap<String, Status> = EXPECTED_STATUS
        .iter()
        .map(|(id, status)| ((*id).to_string(), *status))
        .collect();
    assert_eq!(
        observed, expected,
        "the status of at least one check changed; the expectation is pinned in EXPECTED_STATUS"
    );

    // The seven skips are pinned twice on purpose: as a set (F3's assertion) and inside the status
    // table. A table that drifted from the set would make one of the two red.
    let pinned: Vec<&str> = EXPECTED_STATUS
        .iter()
        .filter(|(_, status)| *status == Status::Skip)
        .map(|(id, _)| *id)
        .collect();
    let mut pinned_sorted = pinned.clone();
    pinned_sorted.sort_unstable();
    let mut skips_sorted = EXPECTED_SKIPS.to_vec();
    skips_sorted.sort_unstable();
    assert_eq!(
        pinned_sorted, skips_sorted,
        "EXPECTED_STATUS and EXPECTED_SKIPS disagree"
    );
}

/// `cli.arguments` is the one id a normal run does not have, and C2 wants a full report even when
/// the arguments are rejected. A rejected `--dtype` therefore produces **all fifteen** ids, with
/// the other fourteen unchanged: an argument error must not silently change what was checked.
#[test]
fn a_rejected_argument_emits_the_fifteenth_id_and_changes_nothing_else() {
    let accepted = statuses(&items(&check(&["--dtype", "f32"]).json()));

    let run = check(&["--dtype", "not-a-dtype"]);
    assert_eq!(
        run.code,
        Some(1),
        "a rejected `--dtype` is a `fail`, so the exit code is 1\n{}",
        run.stdout
    );

    let doc = run.json();
    assert_c6_shape(&doc);
    let rejected_items = items(&doc);
    let rejected = statuses(&rejected_items);
    assert_id_set(&rejected, &C6_CHECK_IDS, "rejected arguments");
    assert_skip_set(&rejected);
    assert_eq!(
        rejected.get("cli.arguments"),
        Some(&Status::Fail),
        "the rejected argument must be the `cli.arguments` failure\n{}",
        run.stdout
    );

    for (id, status) in &accepted {
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
        .find(|(id, _, _)| id == "cli.arguments")
        .expect("cli.arguments must be present");
    assert!(
        argument.2.contains("--dtype") && argument.2.contains("not-a-dtype"),
        "the `cli.arguments` reason must name the rejected flag and value: {}",
        argument.2
    );
}
