//! Ray GPU worker integration via rayrust.
//!
//! Replaces the Python-based `gpu_run.sh` with a native Rust implementation.
//! Uses `runtime_env` to inject environment variables into Ray Python workers,
//! then calls a Python function via `task_call_python` to run the command.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::json;

/// Result from the Python runner function.
#[derive(Debug, Deserialize)]
struct RunResult {
    stdout: String,
    stderr: String,
    returncode: i64,
}

/// Paths and env vars for the GPU worker environment.
fn build_runtime_env(runner_path: &str) -> String {
    let ts = "/usr/local/lib/python3.13/dist-packages";
    let ld_library = format!(
        "/usr/local/lib/python3.13/dist-packages/ray/cpp/lib:\
         {ts}/torch/lib:{ts}/nvidia/cuda_runtime/lib:{ts}/nvidia/cuda_cupti/lib:\
         {ts}/nvidia/cuda_nvrtc/lib:{ts}/nvidia/cublas/lib:{ts}/nvidia/cudnn/lib:\
         {ts}/nvidia/cufft/lib:{ts}/nvidia/curand/lib:{ts}/nvidia/cusolver/lib:\
         {ts}/nvidia/cusparse/lib:{ts}/nvidia/cusparselt/lib:{ts}/nvidia/nccl/lib:\
         {ts}/nvidia/nvjitlink/lib:{ts}/nvidia/nvshmem/lib:{ts}/nvidia/nvtx/lib:\
         /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
    );

    let path = format!(
        "/vePFS-Mindverse/share/huggingface/rustrain-deps/cargo/bin:\
         /vePFS-Mindverse/share/huggingface/rustrain-deps/venv/bin:\
         /usr/local/nvidia/bin:/usr/local/cuda/bin:\
         /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    );

    json!({
        "env_vars": {
            "PATH": path,
            "RUSTUP_HOME": "/vePFS-Mindverse/share/huggingface/rustrain-deps/rustup",
            "CARGO_HOME": "/vePFS-Mindverse/share/huggingface/rustrain-deps/cargo",
            "CARGO_TARGET_DIR": "/tmp/rustrain-target-a800",
            "LIBTORCH_USE_PYTORCH": "1",
            "LIBTORCH_BYPASS_VERSION_CHECK": "1",
            "PYTHONPATH": runner_path,
            "HOME": "/root",
        }
    })
    .to_string()
}

pub fn run(
    num_gpus: usize,
    ray_address: Option<String>,
    runner_path: &str,
    command: &[String],
) -> Result<()> {
    let cmd_str = command.join(" ");
    eprintln!("rustrain ray-gpu: num_gpus={num_gpus}");
    eprintln!("rustrain ray-gpu: command={cmd_str}");

    // Resolve Ray head address
    let address = ray_address.unwrap_or_else(|| {
        std::fs::read_to_string("/vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or_else(|| "192.168.42.174".to_string())
    });
    let full_address = if address.contains(':') {
        address
    } else {
        format!("{address}:6379")
    };
    eprintln!("rustrain ray-gpu: ray_address={full_address}");

    // Build runtime_env with all needed env vars
    let runtime_env = build_runtime_env(runner_path);

    // Initialize Ray
    let config = rayrust::runtime::RayConfig::new(&full_address).runtime_env(runtime_env);

    rayrust::init_with_config(&config).context("Failed to initialize Ray via rayrust")?;

    // Serialize args for the Python function
    let cmd_bytes = rayrust::serialize(&cmd_str).context("Failed to serialize command")?;
    let num_gpus_bytes = rayrust::serialize(&num_gpus).context("Failed to serialize num_gpus")?;

    // Call the Python runner function
    eprintln!("rustrain ray-gpu: submitting to Ray worker...");
    let obj = rayrust::task_call_python(
        "rustrain_ray_runner",
        "run_command",
        &[&cmd_bytes, &num_gpus_bytes],
        &[],
    )
    .context("Failed to call Python task on Ray worker")?;

    // Get the result — cast to RunResult and use get() (handles xlang header)
    let result: RunResult = obj
        .cast::<RunResult>()
        .get_timeout(300_000) // 5 min timeout
        .context("Failed to get result from Ray worker")?;

    // Print output
    if !result.stdout.is_empty() {
        print!("{}", result.stdout);
    }
    if !result.stderr.is_empty() {
        eprint!("{}", result.stderr);
    }

    rayrust::shutdown();

    if result.returncode != 0 {
        bail!(
            "Ray worker command failed with exit code {}",
            result.returncode
        );
    }

    Ok(())
}
