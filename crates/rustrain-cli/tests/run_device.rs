//! The `--device` argument, on a box with no GPU.
//!
//! Everything goes through the real binary. No test here needs a GPU: the
//! world-size guard fires *before* the runner touches the model, the weights
//! or the device, and the `ops check` device test accepts exactly the two
//! honest outcomes (loud failure naming the driver on a GPU-less box, success
//! on a box that has one).

use std::path::{Path, PathBuf};
use std::process::Command;

fn cli_binary() -> &'static str {
    option_env!("CARGO_BIN_EXE_rustrain")
        .or(option_env!("CARGO_BIN_EXE_rustrain-cli"))
        .expect("cargo did not inject CARGO_BIN_EXE_<bin>: check rustrain-cli's binary target name")
}

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/run-tiny")
}

/// The CUDA guard refuses a multi-rank mesh **before** anything touches the
/// device: the message says why (one context, one thread; one process per
/// rank is D6's next step) and never mentions the driver, which would mean the
/// run had already gone looking for one.
#[test]
fn run_with_a_cuda_device_and_world_over_one_is_refused_up_front() {
    let out = std::env::temp_dir().join(format!("rustrain-device-{}.npz", std::process::id()));
    let output = Command::new(cli_binary())
        .args([
            "run",
            "--model",
            fixture_model().to_str().unwrap(),
            "--checkpoint",
            fixture_model().to_str().unwrap(),
            "--tokens",
            "0",
            "--out",
            out.to_str().unwrap(),
            "--device",
            "cuda",
            "--tp",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "a CUDA device on a world-2 mesh must be refused"
    );
    let diagnostic = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        diagnostic.contains("one process per rank"),
        "the refusal must say why multi-rank CUDA cannot run in one process, got:\n{diagnostic}"
    );
    assert!(
        diagnostic.contains("world size 2"),
        "the refusal must name the world size, got:\n{diagnostic}"
    );
    assert!(
        !diagnostic.contains("libcuda") && !diagnostic.contains("cuInit"),
        "the guard must fire before the driver is ever looked up, got:\n{diagnostic}"
    );
}

/// The same guard covers every sweep configuration, up front.
#[test]
fn run_sweep_with_a_cuda_device_refuses_any_multi_rank_config() {
    let out =
        std::env::temp_dir().join(format!("rustrain-device-sweep-{}.json", std::process::id()));
    let output = Command::new(cli_binary())
        .args([
            "run",
            "--model",
            fixture_model().to_str().unwrap(),
            "--checkpoint",
            fixture_model().to_str().unwrap(),
            "--tokens",
            "0",
            "--out",
            out.to_str().unwrap(),
            "--sweep",
            "tp=1;tp=2",
            "--device",
            "cuda",
        ])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "the tp=2 sweep config must be refused"
    );
    let diagnostic = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        diagnostic.contains("one process per rank"),
        "the refusal must say why, got:\n{diagnostic}"
    );
}

/// A device spelling outside `cpu` / `cuda` / `cuda:<index>` is an argument
/// error naming the offending text.
#[test]
fn run_rejects_an_unknown_device_spelling() {
    let out = std::env::temp_dir().join(format!("rustrain-device-bad-{}.npz", std::process::id()));
    let output = Command::new(cli_binary())
        .args([
            "run",
            "--model",
            fixture_model().to_str().unwrap(),
            "--checkpoint",
            fixture_model().to_str().unwrap(),
            "--tokens",
            "0",
            "--out",
            out.to_str().unwrap(),
            "--device",
            "cuda:",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let diagnostic = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        diagnostic.contains("cuda:"),
        "the error must name the offending spelling, got:\n{diagnostic}"
    );
}

/// `--device cpu` (the explicit form of the default) must leave `ops check`
/// byte-identical: the CPU report contract is pinned, not approximated.
#[test]
fn ops_check_cpu_report_is_byte_identical_with_and_without_the_flag() {
    let base = Command::new(cli_binary())
        .args(["ops", "check"])
        .output()
        .unwrap();
    assert!(
        base.status.success(),
        "ops check must pass:\n{}",
        String::from_utf8_lossy(&base.stderr)
    );
    let explicit = Command::new(cli_binary())
        .args(["ops", "check", "--device", "cpu"])
        .output()
        .unwrap();
    assert!(
        explicit.status.success(),
        "ops check --device cpu must pass:\n{}",
        String::from_utf8_lossy(&explicit.stderr)
    );
    assert_eq!(
        base.stdout, explicit.stdout,
        "the CPU report must be byte-identical with and without --device cpu"
    );
}

/// A device provider requested without a device: the gate must fail loudly —
/// up front, even though the registry holds only the host reference — and
/// never pretend. On a box with a real device the same command runs green.
/// The expected outcome is decided by probing the allocator exactly as the
/// CLI does, so the test stays honest on both kinds of machine.
#[test]
fn ops_check_with_a_cuda_device_fails_loudly_without_one_or_runs_with_one() {
    let output = Command::new(cli_binary())
        .args(["ops", "check", "--device", "cuda"])
        .output()
        .unwrap();
    let diagnostic = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    match rustrain_runtime::CudaAllocator::new(0) {
        Err(_) => {
            // No device on this box: the gate must refuse up front, naming the
            // library or the failing driver call.
            assert!(
                !output.status.success(),
                "without a device, `ops check --device cuda` must fail loudly:\n{diagnostic}"
            );
            assert!(
                diagnostic.contains("tried libcuda.so.1")
                    || diagnostic.contains("cuInit")
                    || diagnostic.contains("cuDeviceGet")
                    || diagnostic.contains("cuMemAlloc"),
                "the failure must name the library or the driver call, got:\n{diagnostic}"
            );
            assert!(
                !diagnostic.contains("panicked at"),
                "a missing device is reported, never panicked on:\n{diagnostic}"
            );
        }
        Ok(_) => {
            // A real device: the gate runs the reference on the host and must
            // come back green.
            assert!(
                output.status.success(),
                "with a device, `ops check --device cuda` must pass:\n{diagnostic}"
            );
        }
    }
}
