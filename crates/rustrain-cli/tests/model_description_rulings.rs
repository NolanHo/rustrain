//! End-to-end cases for the §3.7 rulings the frozen gate cannot reach (ruling #12).
//!
//! `model_description.rs` covers the six original contract paths and is frozen; the newer rulings
//! — `out` must be declared (§3.7 #1), `select`/`template` are mutually exclusive (#13), a slot no
//! node reads or writes is a dead hook (#11) — get their own fixtures and their own file.
//!
//! Everything goes through the real binary: exit code, stdout, stderr, no internal Rust API.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The binary under test.
///
/// `CARGO_BIN_EXE_<target>` is injected by cargo and `<target>` is the binary target name. Both
/// spellings are accepted so that renaming the target to `rustrain` (contract §3.6 #9) does not
/// turn this gate into a compile error.
fn cli_binary() -> &'static str {
    option_env!("CARGO_BIN_EXE_rustrain")
        .or(option_env!("CARGO_BIN_EXE_rustrain-cli"))
        .expect("cargo did not inject CARGO_BIN_EXE_<bin>: check rustrain-cli's binary target name")
}

/// `crates/rustrain-cli/tests/fixtures/model-desc-rulings/<name>`, a complete model directory.
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/model-desc-rulings")
        .join(name)
}

struct Run {
    code: Option<i32>,
    stderr: Vec<u8>,
}

impl Run {
    fn stderr_text(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }

    fn dump(&self) -> String {
        format!(
            "exit = {:?}\n--- stderr ---\n{}",
            self.code,
            self.stderr_text()
        )
    }

    /// Every rejection in this file must be a readable diagnostic: non-zero exit, a message on
    /// stderr, no panic backtrace (contract §3.6 #8).
    fn expect_readable_failure(&self) {
        assert_ne!(
            self.code,
            Some(0),
            "expected a non-zero exit\n{}",
            self.dump()
        );
        assert!(!self.stderr.is_empty(), "stderr is empty\n{}", self.dump());
        assert!(
            !self.stderr_text().contains("panicked at"),
            "a bad description must be reported, not panicked on\n{}",
            self.dump()
        );
    }
}

/// Run `rustrain plan explain --model <model-dir> --json`.
fn plan_explain_json(model_dir: &Path) -> Run {
    let output = Command::new(cli_binary())
        .args(["plan", "explain", "--model"])
        .arg(model_dir)
        .arg("--json")
        .output()
        .unwrap_or_else(|e| panic!("failed to start {}: {e}", cli_binary()));
    Run {
        code: output.status.code(),
        stderr: output.stderr,
    }
}

/// §3.7 #1: a node may only write a declared slot or one of the template's declared `outputs`.
/// `h9` is neither, so the description has nowhere to take its shape from.
#[test]
fn a_node_writing_an_undeclared_slot_is_rejected() {
    let run = plan_explain_json(&fixture("undeclared-slot"));
    run.expect_readable_failure();

    let stderr = run.stderr_text();
    assert!(
        stderr.contains("h9"),
        "stderr must name the undeclared slot\n{}",
        run.dump()
    );
    assert!(
        stderr.contains("norm"),
        "stderr must name the template\n{}",
        run.dump()
    );
}

/// §3.7 #13: with `select`, a `template` beside it would be a second fallback source for the same
/// fact, so the pair is rejected instead of silently resolved.
#[test]
fn select_and_template_together_are_rejected_as_two_sources() {
    let run = plan_explain_json(&fixture("select-template-conflict"));
    run.expect_readable_failure();

    let stderr = run.stderr_text();
    assert!(
        stderr.contains("select") && stderr.contains("template"),
        "stderr must name both keys\n{}",
        run.dump()
    );
    assert!(
        stderr.contains("two"),
        "stderr must say these are two sources for one fact\n{}",
        run.dump()
    );
}

/// §3.7 #11: a declared slot no node reads or writes is a dead hook. The fixture's slot is even
/// bound to a checkpoint tensor, so nothing else in the pipeline would notice it.
#[test]
fn a_declared_slot_no_node_reads_or_writes_is_rejected() {
    let run = plan_explain_json(&fixture("dead-slot"));
    run.expect_readable_failure();

    let stderr = run.stderr_text();
    assert!(
        stderr.contains("dead_weight"),
        "stderr must name the untouched slot\n{}",
        run.dump()
    );
    assert!(
        stderr.contains("norm"),
        "stderr must name the template\n{}",
        run.dump()
    );
}
