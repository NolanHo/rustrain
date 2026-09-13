//! D2 acceptance gate: `rustrain check --model <dir> --checkpoint <snapshot>`.
//!
//! Everything goes through the real binary (`env!("CARGO_BIN_EXE_rustrain")`): exit code, stdout,
//! stderr, no internal Rust API. No network, no weights: the checkpoint side is a
//! `rustrain.ckpt_meta.v1` snapshot, which C5 makes the offline, reproducible form of
//! "checkpoint metadata".
//!
//! Contract: `docs/design/qwen36-text/spec.md` C2 (what `check` reports and the
//! Pass/Fail/Warning/Skip discipline) + C5 (checkpoint metadata and its two shapes) + D2
//! (acceptance); declaration semantics: `docs/design/model-description.md` §3.4 (binding, and the
//! `[out, in] -> [in, out]` direction convention) and §3.5 (the three L2 mandates).
//!
//! The synthetic family all derives from `models/tiny`: **one** description with two weight slots
//! whose checkpoint layout is transposed. Every negative case is that same description with
//! exactly one thing broken, so a report that passes the positive case by checking nothing cannot
//! also produce the four distinct rejections. `models/tiny-ignored` differs from `models/tiny` by
//! the `ignore` entry alone, and the pair is run against the *same* snapshot.
//!
//! Deliberate minimal interpretations (the contract fixes the outcomes, not every spelling; see
//! the delivery report — nothing here invents a rule):
//!
//! 1. The report's JSON layout is specified only as "machine-readable, per-item, with Skip/Warning
//!    reasons" plus the four counter names D2 writes down. So: counters are found **by name at any
//!    depth**, and one check item is any object carrying a `status`/`result`/`state`/`outcome`
//!    field whose value is one of the four verdicts, with its wording in `reason`/`detail`/... —
//!    the shape `ops check` already emits (`rustrain-runtime/src/conformance.rs`).
//! 2. C2's "implementation availability" item is located by name over a small alias set; its
//!    verdict, its reason and the names it lists are what the contract actually fixes.
//! 3. `ignore` is read as a top-level array of checkpoint tensor names/patterns (C5). The fixture
//!    uses a literal tensor name, so no wildcard flavour is assumed.
//! 4. The snapshot file is spelled `*.safetensors.meta.json` (C5).
//! 5. `--dtype f32` is passed on every run (C2: it declares the precision being checked).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// The binary under test (§3.6 #9 spells the target `rustrain`; both spellings are accepted so
/// that a rename cannot turn this gate into a compile error).
fn cli_binary() -> &'static str {
    option_env!("CARGO_BIN_EXE_rustrain")
        .or(option_env!("CARGO_BIN_EXE_rustrain-cli"))
        .expect("cargo did not inject CARGO_BIN_EXE_<bin>: check rustrain-cli's binary target name")
}

/// `crates/rustrain-cli/tests/fixtures/check-l2/models/<name>` — a complete model directory
/// (`config.json` + `model.json`).
fn fixture_model(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/check-l2/models")
        .join(name)
}

/// `crates/rustrain-cli/tests/fixtures/check-l2/checkpoints/<name>.safetensors.meta.json`.
fn fixture_checkpoint(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/check-l2/checkpoints")
        .join(format!("{name}.safetensors.meta.json"))
}

/// D1's real description, one crate over: D2 is about the real 1045 tensors, not only the
/// synthetic family. Its model directory is the sibling fixture of `rustrain-model`.
fn qwen36_model_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../rustrain-model/tests/fixtures/qwen36-text")
}

/// One CLI invocation's observables. stdout/stderr stay as raw bytes.
struct Run {
    code: Option<i32>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl Run {
    fn stdout_text(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    fn stderr_text(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }

    /// stdout and stderr together. A rejection may be a text diagnostic or a report with a
    /// non-zero exit code; the contract only fixes that the offending name shows up.
    fn diagnostic(&self) -> String {
        format!("{}\n{}", self.stdout_text(), self.stderr_text())
    }

    fn dump(&self) -> String {
        format!(
            "exit = {:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            self.code,
            self.stdout_text(),
            self.stderr_text()
        )
    }

    fn expect_success(&self) {
        assert_eq!(self.code, Some(0), "expected exit code 0\n{}", self.dump());
    }

    /// A rejection must be a *reported* rejection: a real non-zero exit (not a signal) with
    /// something written, and no panic backtrace (I-5: a wrong input is diagnosed, not crashed on).
    fn expect_readable_failure(&self) {
        match self.code {
            Some(0) | None => panic!(
                "expected a non-zero exit code (a signal is not a diagnosis)\n{}",
                self.dump()
            ),
            Some(_) => {}
        }
        assert!(
            !self.stdout.is_empty() || !self.stderr.is_empty(),
            "the rejection says nothing\n{}",
            self.dump()
        );
        assert!(
            !self.diagnostic().contains("panicked at"),
            "a failed check must be reported, not panicked on\n{}",
            self.dump()
        );
    }

    fn json(&self) -> Value {
        serde_json::from_slice(&self.stdout)
            .unwrap_or_else(|e| panic!("stdout is not a JSON report: {e}\n{}", self.dump()))
    }
}

/// `rustrain check --model <dir> --checkpoint <snapshot> --dtype f32 --json`.
fn check(model: &Path, checkpoint: &Path) -> Run {
    let output = Command::new(cli_binary())
        .args(["check", "--model"])
        .arg(model)
        .arg("--checkpoint")
        .arg(checkpoint)
        .args(["--dtype", "f32", "--json"])
        .output()
        .unwrap_or_else(|e| panic!("failed to start {}: {e}", cli_binary()));
    Run {
        code: output.status.code(),
        stdout: output.stdout,
        stderr: output.stderr,
    }
}

/// D2's four counters, as integers. Absent `dtype_mismatch` is not silently zero.
fn assert_zero_counters(doc: &Value, run: &Run) {
    for key in [
        "slots_unbound",
        "tensors_unconsumed",
        "shape_mismatch",
        "dtype_mismatch",
    ] {
        let found = find_key(doc, key);
        let value = match found {
            Some(Value::Number(n)) => {
                n.as_i64()
                    .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64))
                    .unwrap_or_else(|| panic!("report `{key}` = {n}, not an integer count\n{}", run.dump()))
            }
            Some(other) => panic!("report `{key}` = {other}, not a count\n{}", run.dump()),
            None => panic!(
                "the report has no `{key}` counter (top-level keys: {:?})\n{}",
                doc.as_object().map(|o| o.keys().cloned().collect::<Vec<_>>()),
                run.dump()
            ),
        };
        assert_eq!(value, 0, "`{key}` = {value}, expected 0\n{}", run.dump());
    }
}

/// First value stored under `key`, at any depth: the contract fixes the counter names, not where
/// they sit in the report.
fn find_key<'a>(node: &'a Value, key: &str) -> Option<&'a Value> {
    match node {
        Value::Object(map) => {
            if let Some(found) = map.get(key) {
                return Some(found);
            }
            map.values().find_map(|v| find_key(v, key))
        }
        Value::Array(items) => items.iter().find_map(|v| find_key(v, key)),
        _ => None,
    }
}

/// Every string value under a key containing `needle`. Used for C2's "the report records the dtype
/// actually used".
fn strings_under_keys(node: &Value, needle: &str) -> Vec<String> {
    let mut out = Vec::new();
    collect_strings(node, needle, &mut out);
    out
}

fn collect_strings(node: &Value, needle: &str, out: &mut Vec<String>) {
    match node {
        Value::Object(map) => {
            for (key, value) in map {
                if key.to_ascii_lowercase().contains(needle) {
                    if let Some(text) = value.as_str() {
                        out.push(text.to_string());
                    }
                }
                collect_strings(value, needle, out);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_strings(item, needle, out);
            }
        }
        _ => {}
    }
}

/// C2's verdict vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Pass,
    Fail,
    Warning,
    Skip,
}

impl Status {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "pass" | "passed" | "ok" => Some(Self::Pass),
            "fail" | "failed" | "failure" | "error" => Some(Self::Fail),
            "warning" | "warn" => Some(Self::Warning),
            "skip" | "skipped" => Some(Self::Skip),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Pass => "Pass",
            Self::Fail => "Fail",
            Self::Warning => "Warning",
            Self::Skip => "Skip",
        }
    }
}

/// One check item of the report.
#[derive(Debug)]
struct Item {
    /// Where it was found (`$`, `.key`, `[i]`), for diagnostics.
    path: String,
    name: String,
    status: Status,
    reason: String,
}

const STATUS_KEYS: [&str; 5] = ["status", "result", "state", "outcome", "verdict"];
const REASON_KEYS: [&str; 7] = [
    "reason", "detail", "details", "why", "message", "note", "notes",
];
const NAME_KEYS: [&str; 6] = ["name", "check", "item", "id", "key", "kind"];

/// Every per-item verdict in the report: any object with a status field from the vocabulary.
fn check_items(doc: &Value) -> Vec<Item> {
    let mut out = Vec::new();
    walk(doc, "$".to_string(), &mut out);
    out
}

fn walk(node: &Value, path: String, out: &mut Vec<Item>) {
    match node {
        Value::Object(map) => {
            let status = STATUS_KEYS
                .iter()
                .find_map(|key| map.get(*key).and_then(Value::as_str))
                .and_then(Status::parse);
            if let Some(status) = status {
                let name = NAME_KEYS
                    .iter()
                    .find_map(|key| map.get(*key).and_then(Value::as_str))
                    .unwrap_or_default()
                    .to_string();
                let reason = REASON_KEYS
                    .iter()
                    .filter_map(|key| map.get(*key).and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join(" ");
                out.push(Item {
                    path: path.clone(),
                    name,
                    status,
                    reason,
                });
            }
            for (key, value) in map {
                walk(value, format!("{path}.{key}"), out);
            }
        }
        Value::Array(items) => {
            for (index, value) in items.iter().enumerate() {
                walk(value, format!("{path}[{index}]"), out);
            }
        }
        _ => {}
    }
}

fn describe(items: &[Item]) -> String {
    let mut text = String::new();
    for item in items {
        text.push_str(&format!(
            "  {} {}: {} — {}\n",
            item.path,
            if item.name.is_empty() {
                "(unnamed)"
            } else {
                item.name.as_str()
            },
            item.status.label(),
            item.reason.trim()
        ));
    }
    text
}

/// C2's "implementation availability" item: verdict and reasons are fixed, the spelling is not.
fn availability_item(items: &[Item]) -> &Item {
    const ALIASES: [&str; 6] = [
        "implementation",
        "availability",
        "resolution",
        "resolve",
        "unresolved",
        "provider",
    ];
    let matches = |item: &Item| {
        let haystack = format!("{} {}", item.name, item.path).to_ascii_lowercase();
        ALIASES.iter().any(|alias| haystack.contains(alias))
    };
    items.iter().find(|item| matches(item)).unwrap_or_else(|| {
        panic!(
            "the report has no item that looks like C2's \"implementation availability\" \
             (expected a `name` containing one of {ALIASES:?}); items found:\n{}",
            describe(items)
        )
    })
}

/// Everything the report's items say, name and reason together.
fn items_text(items: &[Item]) -> String {
    items
        .iter()
        .map(|item| format!("{} {} {}", item.name, item.reason, item.path))
        .collect::<Vec<_>>()
        .join("\n")
}

/// `text` contains `n` as a whole number (so `96` is not found inside `1960`).
fn mentions_number(text: &str, n: i64) -> bool {
    let needle = n.to_string();
    text.split(|c: char| !c.is_ascii_digit())
        .any(|token| token == needle)
}

/// C2 + D2: the positive case. Two weight slots, a snapshot that matches them exactly.
#[test]
fn tiny_model_with_a_matching_snapshot_passes_l2() {
    let run = check(&fixture_model("tiny"), &fixture_checkpoint("tiny"));
    run.expect_success();

    let doc = run.json();
    assert_zero_counters(&doc, &run);

    let recorded = strings_under_keys(&doc, "dtype");
    assert!(
        recorded.iter().any(|d| d == "f32"),
        "C2: the report must record the dtype it actually checked; dtype-ish strings found: \
         {recorded:?}\n{}",
        run.dump()
    );
}

/// D2: a case that deliberately drops one binding must `Fail` and name the slot pattern it left
/// unbound.
///
/// `models/missing-binding` is `models/tiny` without the `mlp.down` binding.
#[test]
fn a_weight_slot_without_a_binding_is_reported_by_its_slot_name() {
    let run = check(&fixture_model("missing-binding"), &fixture_checkpoint("tiny"));
    run.expect_readable_failure();

    assert!(
        run.diagnostic().contains("mlp.down"),
        "the unbound slot `mlp.down` must be named\n{}",
        run.dump()
    );
}

/// D2: a case that deliberately declares a checkpoint tensor the description cannot consume must
/// `Fail` and name that pattern.
///
/// `models/extra-declaration` binds `mlp.up` to `model.up.weight_v2`, which the snapshot does not
/// have (the real `model.up.weight` then also ends up unconsumed — a second, expected report line
/// that this test does not depend on).
#[test]
fn a_binding_for_a_tensor_the_checkpoint_lacks_is_reported_by_its_source() {
    let run = check(&fixture_model("extra-declaration"), &fixture_checkpoint("tiny"));
    run.expect_readable_failure();

    assert!(
        run.diagnostic().contains("model.up.weight_v2"),
        "the declared source `model.up.weight_v2` must be named\n{}",
        run.dump()
    );
}

/// D2: a case with a deliberately wrong `transform` (a dropped `transpose`) must `Fail`.
///
/// `models/wrong-transform` drops `transpose(0,1)` from the `mlp.up` binding, so the checkpoint's
/// `[160, 96]` no longer maps onto the declared slot `[96, 160]`. Failing is not enough: the
/// report has to say *which* slot and *what* the gap is (§3.5's shape mandate).
#[test]
fn a_binding_missing_its_transpose_fails_and_shows_the_shape_gap() {
    let run = check(&fixture_model("wrong-transform"), &fixture_checkpoint("tiny"));
    run.expect_readable_failure();

    let text = run.diagnostic();
    assert!(
        text.contains("mlp.up"),
        "the offending slot `mlp.up` must be named\n{}",
        run.dump()
    );
    assert!(
        mentions_number(&text, 160) && mentions_number(&text, 96),
        "the report must show the shape difference (checkpoint [160, 96] vs slot [96, 160])\n{}",
        run.dump()
    );
}

/// D2 + §3.5: every checkpoint tensor is either consumed by a binding or explicitly ignored.
/// `checkpoints/tiny-extra` adds `model.extra.weight`, which nothing consumes.
#[test]
fn a_checkpoint_tensor_no_binding_consumes_is_reported_by_name() {
    let run = check(&fixture_model("tiny"), &fixture_checkpoint("tiny-extra"));
    run.expect_readable_failure();

    assert!(
        run.diagnostic().contains("model.extra.weight"),
        "the unconsumed tensor `model.extra.weight` must be named\n{}",
        run.dump()
    );
}

/// C5: the vision tower is dropped by an *explicit* `ignore` list, never silently. The same run as
/// the test above, with `models/tiny-ignored` (identical description plus one `ignore` entry).
#[test]
fn an_ignored_checkpoint_tensor_stops_failing() {
    let run = check(&fixture_model("tiny-ignored"), &fixture_checkpoint("tiny-extra"));
    run.expect_success();

    assert_zero_counters(&run.json(), &run);
}

/// C2's ruling: an unresolved implementation is a `Skip`, and the exit code is decided by `Fail`
/// alone. The fixture uses `nonexistent_op`, which no provider publishes, so the whole run must
/// still exit 0 and say what is missing.
#[test]
fn an_unregistered_operator_is_a_reasoned_skip_not_a_fail() {
    let run = check(
        &fixture_model("unregistered-op"),
        &fixture_checkpoint("unregistered-op"),
    );
    run.expect_success();

    let doc = run.json();
    assert_zero_counters(&doc, &run);

    let items = check_items(&doc);
    let availability = availability_item(&items);
    assert_eq!(
        availability.status,
        Status::Skip,
        "implementation availability must be Skip, not {}\n{}",
        availability.status.label(),
        describe(&items)
    );
    for item in items.iter().filter(|i| i.status == Status::Skip) {
        assert!(
            !item.reason.trim().is_empty(),
            "C2: every Skip states its reason; {} has none\n{}",
            item.path,
            describe(&items)
        );
    }
    assert!(
        items_text(&items).contains("nonexistent_op"),
        "the skip must answer \"what is missing\": the unresolved operator `nonexistent_op` must \
         be named\n{}",
        describe(&items)
    );
}

/// D2 + C5: the real description against the real checkpoint metadata (26 shard headers, 1045
/// tensors, no weights downloaded). Everything reconciles, and the one thing this machine cannot
/// do — run it — is a reasoned `Skip`, not a `Fail`.
#[test]
fn the_real_qwen36_description_reconciles_the_real_checkpoint_metadata() {
    let run = check(&qwen36_model_dir(), &fixture_checkpoint("qwen36-35b-a3b"));
    run.expect_success();

    let doc = run.json();
    assert_zero_counters(&doc, &run);

    let items = check_items(&doc);
    let availability = availability_item(&items);
    assert_eq!(
        availability.status,
        Status::Skip,
        "implementation availability on this machine must be Skip, not {}\n{}",
        availability.status.label(),
        describe(&items)
    );
    assert!(
        !availability.reason.trim().is_empty(),
        "C2: the Skip must list its reasons one by one\n{}",
        describe(&items)
    );

    const UNIMPLEMENTED_PRIMITIVES: [&str; 5] = [
        "causal_conv1d",
        "gated_delta_rule",
        "l2norm",
        "moe_layer",
        "rmsnorm_gated",
    ];
    let text = items_text(&items);
    assert!(
        UNIMPLEMENTED_PRIMITIVES.iter().any(|op| text.contains(op)),
        "the availability Skip must say what is missing: the description uses five primitives no \
         provider publishes on this machine ({})\n{}",
        UNIMPLEMENTED_PRIMITIVES.join(", "),
        run.dump()
    );
}
