//! `launch` and the rank child it starts: the process-level half of D6 that this
//! box can actually exercise.
//!
//! The numeric half needs GPUs and lives on the verification host. What is
//! testable here is the machinery around it: a rank child runs exactly one rank
//! and writes metrics a launcher can read; a rank child refuses the shapes that
//! cannot work (a CPU world of more than one rank); `launch` refuses a CPU
//! device outright; and a rank that cannot start takes the world down with its
//! own stderr instead of leaving the survivors blocked in a collective.

use std::path::{Path, PathBuf};
use std::process::Command;

fn cli_binary() -> &'static str {
    option_env!("CARGO_BIN_EXE_rustrain")
        .or(option_env!("CARGO_BIN_EXE_rustrain-cli"))
        .expect("cargo did not inject CARGO_BIN_EXE_<bin>: check rustrain-cli's binary target name")
}

fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let lower = bits & 0xffff;
    let mut upper = (bits >> 16) as u16;
    if lower > 0x8000 || (lower == 0x8000 && upper & 1 == 1) {
        upper = upper.wrapping_add(1);
    }
    upper
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&to_bf16(*v).to_le_bytes());
    }
    out
}

/// The same tiny checkpoint `run_tiny` uses: embed [8, 4], up [6, 4], down
/// [4, 6], written as real safetensors bytes beside a model directory.
fn write_checkpoint(dir: &Path) {
    let tensors: Vec<(&str, Vec<i64>, Vec<f32>)> = vec![
        (
            "model.embed.weight",
            vec![8, 4],
            (0..32).map(|o| (o + 1) as f32).collect(),
        ),
        (
            "model.up.weight",
            vec![6, 4],
            (0..24).map(|o| (o + 1) as f32).collect(),
        ),
        (
            "model.down.weight",
            vec![4, 6],
            (0..24).map(|o| (o + 1) as f32).collect(),
        ),
    ];

    let mut payload = Vec::new();
    let mut header = serde_json::Map::new();
    for (name, shape, values) in &tensors {
        let start = payload.len();
        payload.extend_from_slice(&bf16_bytes(values));
        let end = payload.len();
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut shard = Vec::new();
    shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
    shard.extend_from_slice(&header);
    shard.extend_from_slice(&payload);
    std::fs::write(dir.join("model.safetensors"), &shard).unwrap();

    let index = serde_json::json!({
        "metadata": {"total_size": payload.len()},
        "weight_map": {
            "model.embed.weight": "model.safetensors",
            "model.up.weight": "model.safetensors",
            "model.down.weight": "model.safetensors",
        },
    });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index).unwrap() + "\n",
    )
    .unwrap();
}

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/run-tiny")
}

/// A private scratch directory per test.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rustrain-launch-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

struct Outcome {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run_cli(args: &[String]) -> Outcome {
    let output = Command::new(cli_binary())
        .args(args)
        .output()
        .expect("the CLI binary must run");
    Outcome {
        ok: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn s(value: impl ToString) -> String {
    value.to_string()
}

/// A rank child is a complete run of exactly one rank: it loads the model,
/// instantiates **its** plan, executes, and leaves its metrics where the
/// launcher reads them — and it writes no dump of its own, because the world's
/// dump is rank 0's and the launcher assembles it.
#[test]
fn a_rank_child_runs_one_rank_and_reports_metrics_for_the_launcher() {
    let dir = scratch("child");
    write_checkpoint(&dir);
    let metrics = dir.join("rank-0.json");
    let outcome = run_cli(&[
        s("run"),
        s("--model"),
        fixture_model().display().to_string(),
        s("--checkpoint"),
        dir.display().to_string(),
        s("--seq"),
        s(4),
        s("--rank"),
        s(0),
        s("--world"),
        s(1),
        s("--device"),
        s("cpu"),
        s("--metrics"),
        metrics.display().to_string(),
    ]);
    assert!(
        outcome.ok,
        "a one-rank child must run\nstdout: {}\nstderr: {}",
        outcome.stdout, outcome.stderr
    );
    let text = std::fs::read_to_string(&metrics).expect("the child must write its metrics");
    let report: serde_json::Value = serde_json::from_str(&text).expect("the metrics must be JSON");
    assert_eq!(report["rank"].as_u64(), Some(0));
    assert!(
        report["plan_steps"].as_u64().unwrap_or(0) > 0,
        "the child's metrics must describe the work it did: {text}"
    );
    assert!(
        report["wall_seconds"].as_f64().unwrap_or(0.0) > 0.0,
        "the child must report its own wall clock: {text}"
    );
    assert!(
        report["logits"]["values"].as_array().is_some(),
        "rank 0's metrics carry the logits the launcher dumps: {text}"
    );
}

/// The CPU multi-rank path is N threads in **one** process — that is `run --tp
/// n` without `--rank`. A rank child asking for it is told which flag it meant.
#[test]
fn a_rank_child_refuses_a_cpu_world_of_more_than_one() {
    let dir = scratch("cpu-world");
    write_checkpoint(&dir);
    let outcome = run_cli(&[
        s("run"),
        s("--model"),
        fixture_model().display().to_string(),
        s("--checkpoint"),
        dir.display().to_string(),
        s("--seq"),
        s(4),
        s("--tp"),
        s(2),
        s("--rank"),
        s(0),
        s("--world"),
        s(2),
        s("--device"),
        s("cpu"),
        s("--metrics"),
        dir.join("rank-0.json").display().to_string(),
    ]);
    assert!(!outcome.ok, "a CPU rank child must be refused");
    assert!(
        outcome.stderr.contains("--device cuda"),
        "the refusal must name the flag that would work: {}",
        outcome.stderr
    );
}

/// `launch` is defined as one process per GPU, and says so instead of starting
/// a world that cannot carry a collective.
#[test]
fn launch_refuses_a_cpu_device_by_naming_the_gpu_flag() {
    let dir = scratch("cpu-launch");
    write_checkpoint(&dir);
    let outcome = run_cli(&[
        s("launch"),
        s("--model"),
        fixture_model().display().to_string(),
        s("--checkpoint"),
        dir.display().to_string(),
        s("--seq"),
        s(4),
        s("--out"),
        dir.join("out.npz").display().to_string(),
        s("--device"),
        s("cpu"),
    ]);
    assert!(!outcome.ok, "a CPU launch must be refused");
    assert!(
        outcome.stderr.contains("--device cuda"),
        "the refusal must name the flag that would work: {}",
        outcome.stderr
    );
}

/// Zero tolerance for silent worlds: a rank that cannot even load NCCL fails its
/// own process, the launcher reports which rank died and what it said, and the
/// surviving rank's bounded rendezvous wait keeps it from hanging forever. This
/// is also the only launcher path that is deterministic on a machine with no
/// GPUs, which is why it is the one pinned here.
#[test]
fn a_rank_that_cannot_start_takes_the_world_down_with_its_stderr() {
    let dir = scratch("failing-world");
    write_checkpoint(&dir);
    let missing = dir.join("no-such-libnccl.so");
    let outcome = run_cli(&[
        s("launch"),
        s("--model"),
        fixture_model().display().to_string(),
        s("--checkpoint"),
        dir.display().to_string(),
        s("--seq"),
        s(4),
        s("--tp"),
        s(2),
        s("--out"),
        dir.join("out.npz").display().to_string(),
        s("--device"),
        s("cuda"),
        s("--nccl-lib"),
        missing.display().to_string(),
    ]);
    assert!(
        !outcome.ok,
        "a world whose ranks cannot start must fail\nstdout: {}",
        outcome.stdout
    );
    assert!(
        outcome.stderr.contains("rank 0") || outcome.stderr.contains("rank 1"),
        "the failure must name the rank that died: {}",
        outcome.stderr
    );
    assert!(
        outcome.stderr.contains("no-such-libnccl.so"),
        "the failure must carry the rank's own diagnosis: {}",
        outcome.stderr
    );
}
