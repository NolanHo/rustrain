//! Two-GPU GLM5 expert-parallel forward/backward smoke test.
//!
//! Launcher mode (builds once, then starts both ranks concurrently):
//!   cargo run -p rustrain-deepseek-v4 --example glm5_ep_nccl_smoke
//!
//! Rank-worker mode (for torchrun, Slurm, or another process launcher):
//!   RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 GLM5_EP_SMOKE_DIR=/shared/unique/run ...
//!   RANK=1 WORLD_SIZE=2 LOCAL_RANK=1 GLM5_EP_SMOKE_DIR=/shared/unique/run ...

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use rustrain_deepseek_v4::fp8_kernel::glm5_moe_layer_ep_cpp;
use rustrain_nccl::nccl::NcclPersistentComm;
use tch::{Device, Kind, Tensor};

const WORLD_SIZE: usize = 2;
const HIDDEN_SIZE: i64 = 2;
const INTERMEDIATE_SIZE: i64 = 3;
const ROUTED_SCALING_FACTOR: f64 = 1.25;
const TOLERANCE: f64 = 3.0e-4;

struct MlpWeights {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

fn matrix(data: &[f32], rows: i64, cols: i64, device: Device, trainable: bool) -> Tensor {
    let tensor = Tensor::from_slice(data)
        .reshape([rows, cols])
        .to_device(device);
    if trainable {
        tensor.set_requires_grad(true)
    } else {
        tensor
    }
}

fn router_weight(device: Device, trainable: bool) -> Tensor {
    // Positive x[0] selects expert 0; negative x[0] selects expert 1.
    matrix(&[2.0, 0.0, -2.0, 0.0], 2, 2, device, trainable)
}

fn shared_weights(device: Device, trainable: bool) -> MlpWeights {
    MlpWeights {
        gate: matrix(
            &[0.5, -0.25, 0.1, 0.3, -0.2, 0.4],
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            device,
            trainable,
        ),
        up: matrix(
            &[0.2, 0.6, -0.4, 0.5, 0.7, -0.1],
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            device,
            trainable,
        ),
        down: matrix(
            &[0.4, -0.2, 0.1, -0.3, 0.5, 0.2],
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            device,
            trainable,
        ),
    }
}

fn expert_weights(expert: usize, device: Device, trainable: bool) -> MlpWeights {
    let (gate, up, down): (&[f32], &[f32], &[f32]) = match expert {
        0 => (
            &[0.8, -0.1, -0.3, 0.6, 0.2, 0.5],
            &[0.4, 0.7, -0.6, 0.2, 0.3, -0.5],
            &[0.5, -0.4, 0.2, -0.1, 0.6, 0.3],
        ),
        1 => (
            &[-0.5, 0.9, 0.7, 0.1, -0.2, -0.4],
            &[0.6, -0.3, 0.2, 0.8, -0.7, 0.5],
            &[0.3, 0.2, -0.6, 0.4, -0.5, 0.7],
        ),
        _ => unreachable!("the smoke test has exactly two experts"),
    };
    MlpWeights {
        gate: matrix(gate, INTERMEDIATE_SIZE, HIDDEN_SIZE, device, trainable),
        up: matrix(up, INTERMEDIATE_SIZE, HIDDEN_SIZE, device, trainable),
        down: matrix(down, HIDDEN_SIZE, INTERMEDIATE_SIZE, device, trainable),
    }
}

fn rank_input(rank: usize, device: Device, trainable: bool) -> Tensor {
    let (values, seq): (&[f32], i64) = match rank {
        // Routing counts are [2, 1] for rank 0.
        0 => (&[1.0, 0.5, -2.0, 1.0, 0.75, -0.25], 3),
        // Routing counts are [1, 1] for rank 1.
        1 => (&[-3.0, -0.5, 4.0, -1.0], 2),
        _ => unreachable!("the smoke test has exactly two ranks"),
    };
    let tensor = Tensor::from_slice(values)
        .reshape([1, seq, HIDDEN_SIZE])
        .to_device(device);
    if trainable {
        tensor.set_requires_grad(true)
    } else {
        tensor
    }
}

fn rank_loss_coefficients(rank: usize, device: Device) -> Tensor {
    let (values, seq): (&[f32], i64) = match rank {
        0 => (&[0.7, -0.2, 1.1, 0.4, -0.5, 0.8], 3),
        1 => (&[-0.3, 0.9, 0.6, -1.2], 2),
        _ => unreachable!("the smoke test has exactly two ranks"),
    };
    Tensor::from_slice(values)
        .reshape([1, seq, HIDDEN_SIZE])
        .to_device(device)
}

fn mlp(input: &Tensor, weights: &MlpWeights) -> Tensor {
    let gate = input.linear::<&Tensor>(&weights.gate, None).silu();
    let up = input.linear::<&Tensor>(&weights.up, None);
    (gate * up).linear::<&Tensor>(&weights.down, None)
}

fn reference_moe(
    input: &Tensor,
    router: &Tensor,
    shared: &MlpWeights,
    experts: &[MlpWeights],
) -> Tensor {
    let scores = input.linear::<&Tensor>(router, None).sigmoid();
    let (topk_weights, topk_indices) = scores.topk(1, -1, true, true);
    let mut routed = Tensor::zeros_like(input);
    for (expert_id, expert) in experts.iter().enumerate() {
        let selected = topk_indices.eq(expert_id as i64).to_kind(input.kind());
        let weight =
            (&topk_weights * selected).sum_dim_intlist([-1].as_slice(), true, input.kind())
                * ROUTED_SCALING_FACTOR;
        routed += mlp(input, expert) * weight;
    }
    routed + mlp(input, shared)
}

fn max_abs_diff(actual: &Tensor, expected: &Tensor) -> f64 {
    (actual.detach().to_device(Device::Cpu) - expected.detach().to_device(Device::Cpu))
        .abs()
        .max()
        .double_value(&[])
}

fn check_close(name: &str, actual: &Tensor, expected: &Tensor) -> Result<f64> {
    if !actual.defined() || !expected.defined() {
        bail!("{name} is undefined (actual={}, expected={})", actual.defined(), expected.defined());
    }
    let diff = max_abs_diff(actual, expected);
    if !diff.is_finite() || diff > TOLERANCE {
        bail!("{name} mismatch: max_abs={diff:.8e}, tolerance={TOLERANCE:.8e}");
    }
    Ok(diff)
}

fn parse_env_usize(name: &str) -> Result<usize> {
    env::var(name)
        .with_context(|| format!("missing {name}"))?
        .parse()
        .with_context(|| format!("invalid {name}"))
}

fn run_rank() -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != WORLD_SIZE || rank >= WORLD_SIZE {
        bail!("GLM5 EP smoke requires WORLD_SIZE=2 and RANK in 0..2");
    }
    if !tch::Cuda::is_available() || tch::Cuda::device_count() < WORLD_SIZE as i64 {
        bail!("GLM5 EP smoke requires at least two visible CUDA devices");
    }
    let exchange_dir =
        PathBuf::from(env::var("GLM5_EP_SMOKE_DIR").context("missing GLM5_EP_SMOKE_DIR")?);
    let device = Device::Cuda(local_rank);
    let comm = NcclPersistentComm::new_group(&exchange_dir, rank, world_size, local_rank)?;

    let input = rank_input(rank, device, true);
    let router = router_weight(device, true);
    let shared = shared_weights(device, true);
    let local_expert = expert_weights(rank, device, true);
    let output = glm5_moe_layer_ep_cpp(
        &input,
        &shared.gate,
        &shared.up,
        &shared.down,
        None,
        None,
        None,
        &router,
        None,
        &[&local_expert.gate],
        &[&local_expert.up],
        &[&local_expert.down],
        &[None],
        &[None],
        &[None],
        &[rank],
        2,
        1,
        1,
        1,
        0,
        0,
        false,
        ROUTED_SCALING_FACTOR,
        comm.raw_comm_ptr(),
        rank as i32,
        world_size as i32,
        local_rank as i32,
    )?;
    let loss = (&output * rank_loss_coefficients(rank, device)).sum(Kind::Float);
    loss.backward();

    // One CPU graph represents both origin ranks. Router/shared parameters are
    // distinct per origin, while expert parameters are shared so owner-side
    // gradients include assignments returned from both ranks.
    let reference_inputs = [
        rank_input(0, Device::Cpu, true),
        rank_input(1, Device::Cpu, true),
    ];
    let reference_routers = [
        router_weight(Device::Cpu, true),
        router_weight(Device::Cpu, true),
    ];
    let reference_shared = [
        shared_weights(Device::Cpu, true),
        shared_weights(Device::Cpu, true),
    ];
    let reference_experts = [
        expert_weights(0, Device::Cpu, true),
        expert_weights(1, Device::Cpu, true),
    ];
    let mut reference_outputs = Vec::with_capacity(WORLD_SIZE);
    let mut reference_loss: Option<Tensor> = None;
    for origin in 0..WORLD_SIZE {
        let origin_output = reference_moe(
            &reference_inputs[origin],
            &reference_routers[origin],
            &reference_shared[origin],
            &reference_experts,
        );
        let origin_loss =
            (&origin_output * rank_loss_coefficients(origin, Device::Cpu)).sum(Kind::Float);
        reference_loss = Some(match reference_loss {
            Some(total) => total + origin_loss,
            None => origin_loss,
        });
        reference_outputs.push(origin_output);
    }
    reference_loss.expect("two reference ranks").backward();

    let mut worst = 0.0_f64;
    let mut validation_error = None;
    for (name, actual, expected) in [
        ("output", &output, &reference_outputs[rank]),
        ("input.grad", &input.grad(), &reference_inputs[rank].grad()),
        (
            "router.grad",
            &router.grad(),
            &reference_routers[rank].grad(),
        ),
        (
            "shared_gate.grad",
            &shared.gate.grad(),
            &reference_shared[rank].gate.grad(),
        ),
        (
            "shared_up.grad",
            &shared.up.grad(),
            &reference_shared[rank].up.grad(),
        ),
        (
            "shared_down.grad",
            &shared.down.grad(),
            &reference_shared[rank].down.grad(),
        ),
        (
            "expert_gate.grad",
            &local_expert.gate.grad(),
            &reference_experts[rank].gate.grad(),
        ),
        (
            "expert_up.grad",
            &local_expert.up.grad(),
            &reference_experts[rank].up.grad(),
        ),
        (
            "expert_down.grad",
            &local_expert.down.grad(),
            &reference_experts[rank].down.grad(),
        ),
    ] {
        match check_close(name, actual, expected) {
            Ok(diff) => worst = worst.max(diff),
            Err(error) if validation_error.is_none() => validation_error = Some(error),
            Err(_) => {}
        }
    }

    // Final collective is also a barrier and propagates local validation
    // failures, so neither process destroys the communicator while its peer
    // is still checking gradients.
    let status = Tensor::from_slice(&[(rank + 1) as f32, f32::from(validation_error.is_some())])
        .to_device(device);
    let status_sum = comm.all_reduce(&status)?.to_device(Device::Cpu);
    let ready_sum = status_sum.double_value(&[0]);
    let failed_ranks = status_sum.double_value(&[1]);
    if (ready_sum - 3.0).abs() > f32::EPSILON as f64 {
        bail!("final NCCL barrier returned ready sum {ready_sum}, expected 3");
    }
    if let Some(error) = validation_error {
        return Err(error).context(format!("rank {rank} validation failed"));
    }
    if failed_ranks != 0.0 {
        bail!("a peer rank reported GLM5 EP validation failure");
    }
    println!("GLM5_EP_SMOKE_PASS rank={rank} local_rank={local_rank} worst_max_abs={worst:.8e}");
    Ok(())
}

fn launcher_dir() -> Result<PathBuf> {
    let base = env::var_os("GLM5_EP_SMOKE_BASE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(env::temp_dir);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX_EPOCH")?
        .as_nanos();
    Ok(base.join(format!(
        "rustrain-glm5-ep-smoke-{}-{timestamp}",
        std::process::id()
    )))
}

fn launch_two_ranks(executable: &Path) -> Result<()> {
    let exchange_dir = launcher_dir()?;
    std::fs::create_dir_all(&exchange_dir)
        .with_context(|| format!("failed to create {}", exchange_dir.display()))?;
    println!(
        "GLM5 EP smoke exchange directory: {}",
        exchange_dir.display()
    );

    let mut children = Vec::with_capacity(WORLD_SIZE);
    for rank in 0..WORLD_SIZE {
        let child = Command::new(executable)
            .env("RANK", rank.to_string())
            .env("WORLD_SIZE", WORLD_SIZE.to_string())
            .env("LOCAL_RANK", rank.to_string())
            .env("GLM5_EP_SMOKE_DIR", &exchange_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to spawn rank {rank}"))?;
        children.push((rank, child));
    }

    let deadline = std::time::Instant::now() + Duration::from_secs(180);
    let mut statuses = vec![None; WORLD_SIZE];
    loop {
        for (rank, child) in &mut children {
            if statuses[*rank].is_none() {
                statuses[*rank] = child
                    .try_wait()
                    .with_context(|| format!("failed to poll rank {rank}"))?;
            }
        }
        let observed_failure = statuses.iter().flatten().any(|status| !status.success());
        if statuses.iter().all(Option::is_some) {
            break;
        }
        if observed_failure || std::time::Instant::now() >= deadline {
            for (rank, child) in &mut children {
                if statuses[*rank].is_none() {
                    let _ = child.kill();
                    statuses[*rank] = Some(
                        child
                            .wait()
                            .with_context(|| format!("failed to reap rank {rank}"))?,
                    );
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    let failed: Vec<_> = statuses
        .into_iter()
        .enumerate()
        .filter_map(|(rank, status)| status.filter(|status| !status.success()).map(|s| (rank, s)))
        .collect();
    if !failed.is_empty() {
        bail!("GLM5 EP smoke rank failures: {failed:?}");
    }
    println!("GLM5_EP_SMOKE_PASS world_size={WORLD_SIZE}");
    Ok(())
}

fn main() -> Result<()> {
    if env::var_os("RANK").is_some() {
        run_rank()
    } else {
        launch_two_ranks(&env::current_exe().context("failed to locate smoke executable")?)
    }
}
