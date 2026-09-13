//! C3/C4's acceptance on the CLI side: `l1.instantiate` checks **every** PP stage at one
//! representative rank, so a stage-1-only divisibility failure is a `fail` — not masked by a
//! clean stage 0 — and a stage that instantiates to no nodes is a stage-declaration error,
//! not a clean bill of health. C3's other half is pinned here too: the propagation items'
//! reasons must say they evaluated stage 0 (rank 0) only, and that the other stages are not
//! propagated (the cross-stage seam decision is D5's).
//!
//! Everything goes through the real binary (`env!("CARGO_BIN_EXE_rustrain")`); the fixtures
//! are the tiny two-stage descriptions under `fixtures/check-instantiate/`.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn cli_binary() -> &'static str {
    option_env!("CARGO_BIN_EXE_rustrain")
        .or(option_env!("CARGO_BIN_EXE_rustrain-cli"))
        .expect("cargo did not inject CARGO_BIN_EXE_<bin>: check rustrain-cli's binary target name")
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/check-instantiate")
        .join(name)
}

struct Run {
    code: Option<i32>,
    doc: Value,
}

fn check(model: &str, extra: &[&str]) -> Run {
    let output = Command::new(cli_binary())
        .arg("check")
        .arg("--model")
        .arg(fixture(model))
        .args(extra)
        .arg("--json")
        .output()
        .unwrap_or_else(|e| panic!("failed to start {}: {e}", cli_binary()));
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let doc: Value = serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!("--json did not print one JSON report to stdout: {e}\n{stdout}")
    });
    Run {
        code: output.status.code(),
        doc,
    }
}

fn item<'a>(run: &'a Run, id: &str) -> &'a Value {
    run.doc["checks"]
        .as_array()
        .unwrap_or_else(|| panic!("the report has no `checks` array"))
        .iter()
        .find(|item| item["id"].as_str() == Some(id))
        .unwrap_or_else(|| panic!("the report has no `{id}` item"))
}

/// **Reviewer C3 (MEDIUM).** `stage1-bad` declares a clean stage 0 and a stage-1 weight
/// `post.w [9, 8]` sharded `{0: tp}`: `9 % 2 != 0`. Checking only rank 0 (stage 0) used to
/// exit 0; the check must instantiate every stage and fail, naming stage 1 and the
/// constraint.
#[test]
fn a_stage_one_only_non_divisible_shard_fails_the_run() {
    let run = check("stage1-bad", &["--tp", "2", "--pp", "2"]);
    assert_eq!(
        run.code,
        Some(1),
        "a stage-1-only divisibility failure is a `fail`\n{}",
        run.doc
    );
    let instantiate = item(&run, "l1.instantiate");
    assert_eq!(instantiate["status"].as_str(), Some("fail"));
    let reason = instantiate["reason"].as_str().expect("a reason");
    assert!(
        reason.contains("stage 1") && reason.contains("rank 2"),
        "the failure names the stage and its representative rank: {reason}"
    );
    assert!(
        reason.contains("post.w") && reason.contains("dim 0") && reason.contains("9"),
        "the failure names the slot and the constraint: {reason}"
    );
    for id in [
        "l1.layout_propagation",
        "l1.partial_fulfillment",
        "l1.collective_axes",
    ] {
        let skipped = item(&run, id);
        assert_eq!(
            skipped["status"].as_str(),
            Some("skip"),
            "`{id}` cannot run when `l1.instantiate` failed"
        );
        assert!(
            skipped["reason"]
                .as_str()
                .expect("a skip reason")
                .contains("stage 0 (rank 0) only"),
            "the `{id}` skip names the rank-0-only propagation scope"
        );
    }
}

/// The same description at `--pp 1` keeps failing — there is one stage, and it carries the
/// non-divisible shard. This pins that the per-stage check did not weaken the pp=1 path.
#[test]
fn the_same_description_at_pp_one_still_fails() {
    let run = check("stage1-bad", &["--tp", "2"]);
    assert_eq!(run.code, Some(1), "pp=1 sees the same shard\n{}", run.doc);
    let instantiate = item(&run, "l1.instantiate");
    assert_eq!(instantiate["status"].as_str(), Some("fail"));
    let reason = instantiate["reason"].as_str().expect("a reason");
    assert!(
        reason.contains("stage 0") && reason.contains("post.w"),
        "at pp=1 the failure is stage 0's: {reason}"
    );
}

/// **Reviewer C4 (MEDIUM).** A stage that owns no work: every instance of `empty-stage` is
/// on stage 0, so at `--pp 2` stage 1 instantiates to zero nodes. That is a
/// stage-declaration error — a `fail` naming the stage — never a pass over "0 collective(s),
/// 0 slot(s)".
#[test]
fn an_empty_stage_is_a_stage_declaration_error() {
    let run = check("empty-stage", &["--pp", "2"]);
    assert_eq!(
        run.code,
        Some(1),
        "an empty stage is a stage-declaration error\n{}",
        run.doc
    );
    let instantiate = item(&run, "l1.instantiate");
    assert_eq!(instantiate["status"].as_str(), Some("fail"));
    let reason = instantiate["reason"].as_str().expect("a reason");
    assert!(
        reason.contains("stage 1") && reason.contains("no nodes"),
        "the failure names the stage and the emptiness: {reason}"
    );
}

/// The same fixture at `--pp 1` is a normal one-stage plan: the empty-stage refusal must not
/// fire when there is no second stage.
#[test]
fn the_empty_stage_fixture_at_pp_one_passes() {
    let run = check("empty-stage", &[]);
    assert_eq!(
        run.code,
        Some(0),
        "pp=1 has one non-empty stage\n{}",
        run.doc
    );
    let instantiate = item(&run, "l1.instantiate");
    assert_eq!(instantiate["status"].as_str(), Some("pass"));
    // C5's witness: the details carry the per-stage node/slot counts.
    let details = instantiate["details"]
        .as_array()
        .expect("per-stage details");
    assert_eq!(details.len(), 1, "one stage, one detail line");
    let detail = details[0].as_str().expect("a detail line");
    assert!(
        detail.starts_with("stage 0 (rank 0): ")
            && detail.contains("node(s)")
            && detail.contains("slot(s)"),
        "the detail carries the instantiated counts: {detail}"
    );
    // C3's wording: the propagation reasons name the stage-0 (rank 0) scope and D5's seam.
    let propagation = item(&run, "l1.layout_propagation");
    assert_eq!(propagation["status"].as_str(), Some("pass"));
    let reason = propagation["reason"].as_str().expect("a reason");
    assert!(
        reason.contains("stage-0 (rank 0)") && reason.contains("D5"),
        "the pass names the rank-0-only scope and the seam decision: {reason}"
    );
}
