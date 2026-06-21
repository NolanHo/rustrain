use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use tch::{
    Cuda, Device, Kind, Reduction, Tensor, nn,
    nn::{Init, Module, OptimizerConfig},
    no_grad,
};
use tracing::info;

use crate::{
    metrics::memory_rss_mb,
    nccl_smoke,
    runtime::{Config, DType, Device as RuntimeDevice, LrScheduler},
};

#[derive(Debug, Clone)]
pub struct TchTrainSmokeSummary {
    pub initial_loss: f64,
    pub final_loss: f64,
    pub embedding_grad_defined: bool,
    pub lm_head_grad_defined: bool,
    pub first_step_grad_norm: f64,
    pub final_learning_rate: f64,
    pub compute_kind: String,
    pub memory_rss_mb: Option<f64>,
    pub gpu_memory_allocated_mb: Option<f64>,
    pub data_parallel_size: usize,
    pub dp_grad_max_delta: Option<f32>,
    pub dp_loss_delta: Option<f64>,
}

#[derive(Debug, serde::Serialize)]
struct TchCudaProbeSummary {
    cuda_available: bool,
    device_count: i64,
}

#[derive(Debug, serde::Serialize)]
struct TchDpGradientRankSummary {
    rank: usize,
    world_size: usize,
    local_rank: usize,
    local_loss: f64,
    global_loss: f64,
    expected_loss: f64,
    loss_delta: f64,
    reduced_grad: Vec<f32>,
    expected_grad: Vec<f32>,
    grad_max_delta: f32,
}

#[derive(Debug, serde::Serialize)]
struct TchMoeSmokeSummary {
    device: String,
    train_steps: usize,
    learning_rate: f64,
    aux_loss_weight: f64,
    tokens: i64,
    hidden_size: i64,
    expert_hidden_size: i64,
    num_experts: i64,
    top_k: i64,
    initial_loss: f64,
    final_loss: f64,
    initial_task_loss: f64,
    final_task_loss: f64,
    initial_load_balance_loss: f64,
    final_load_balance_loss: f64,
    checkpoint_output: String,
    reloaded_loss: f64,
    reload_delta: f64,
    continuous_second_loss: f64,
    resumed_second_loss: f64,
    second_step_delta: f64,
    second_step_router_max_abs: f64,
    second_step_expert_up_max_abs: f64,
    second_step_expert_down_max_abs: f64,
    expert_load: Vec<usize>,
    total_params: usize,
    activated_params: usize,
    router_grad_defined: bool,
    expert_up_grad_defined: bool,
    expert_down_grad_defined: bool,
    router_grad_norm: f64,
    expert_up_grad_norm: f64,
    expert_down_grad_norm: f64,
    router_delta_norm: f64,
    expert_up_delta_norm: f64,
    expert_down_delta_norm: f64,
    memory_rss_mb: Option<f64>,
    gpu_memory_allocated_mb: Option<f64>,
}

struct TchMoeForward {
    loss: Tensor,
    task_loss: Tensor,
    load_balance_loss: Tensor,
    expert_load: Vec<usize>,
}

struct TchMoeSgdStep {
    loss_before: f64,
    router_grad_defined: bool,
    expert_up_grad_defined: bool,
    expert_down_grad_defined: bool,
    router_grad_norm: f64,
    expert_up_grad_norm: f64,
    expert_down_grad_norm: f64,
}

pub fn probe_tch_cuda() -> Result<()> {
    let summary = TchCudaProbeSummary {
        cuda_available: Cuda::is_available(),
        device_count: Cuda::device_count(),
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if !summary.cuda_available {
        return Err(anyhow!("tch CUDA is not available"));
    }
    if summary.device_count == 0 {
        return Err(anyhow!("tch CUDA is available but reports zero devices"));
    }

    Ok(())
}

pub fn run_tch_moe_smoke() -> Result<()> {
    let summary = tch_moe_smoke_summary()?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn tch_moe_smoke_summary() -> Result<TchMoeSmokeSummary> {
    if !Cuda::is_available() || Cuda::device_count() == 0 {
        bail!("tch MoE smoke requires a visible CUDA GPU");
    }

    let device = Device::Cuda(0);
    let tokens = 6;
    let hidden_size = 4;
    let expert_hidden_size = 6;
    let num_experts = 3;
    let top_k = 1;
    let train_steps = 8;
    let learning_rate = 0.2;
    let aux_loss_weight = 0.01;

    let input = Tensor::from_slice(&[
        0.2_f32, -0.1, 0.4, 0.7, -0.3, 0.8, 0.1, -0.5, 0.6, 0.2, -0.4, 0.3, 0.9, -0.7, 0.2, 0.1,
        -0.5, -0.2, 0.8, 0.4, 0.3, 0.5, -0.6, 0.2,
    ])
    .reshape([tokens, hidden_size])
    .to_device(device);
    let target = Tensor::zeros([tokens, hidden_size], (Kind::Float, device));

    let mut router = Tensor::from_slice(&[
        0.3_f32, -0.2, 0.1, -0.1, 0.4, 0.2, 0.2, 0.1, -0.3, -0.4, 0.2, 0.5,
    ])
    .reshape([hidden_size, num_experts])
    .to_device(device)
    .set_requires_grad(true);
    let mut expert_up = (Tensor::arange(
        num_experts * hidden_size * expert_hidden_size,
        (Kind::Float, device),
    )
    .reshape([num_experts, hidden_size, expert_hidden_size])
        / 80.0
        - 0.4)
        .set_requires_grad(true);
    let mut expert_down = (Tensor::arange(
        num_experts * expert_hidden_size * hidden_size,
        (Kind::Float, device),
    )
    .reshape([num_experts, expert_hidden_size, hidden_size])
        / 90.0
        - 0.3)
        .set_requires_grad(true);

    let initial_router = router.detach().to_device(Device::Cpu);
    let initial_expert_up = expert_up.detach().to_device(Device::Cpu);
    let initial_expert_down = expert_down.detach().to_device(Device::Cpu);
    let initial = tch_moe_forward(
        &input,
        &target,
        &router,
        &expert_up,
        &expert_down,
        num_experts,
        aux_loss_weight,
    )?;
    let initial_loss = initial.loss.double_value(&[]);
    let initial_task_loss = initial.task_loss.double_value(&[]);
    let initial_load_balance_loss = initial.load_balance_loss.double_value(&[]);

    let mut first_step: Option<TchMoeSgdStep> = None;

    for step in 1..=train_steps {
        let step_summary = tch_moe_sgd_step(
            &input,
            &target,
            &mut router,
            &mut expert_up,
            &mut expert_down,
            num_experts,
            aux_loss_weight,
            learning_rate,
        )?;
        if step == 1 {
            first_step = Some(step_summary);
        }
    }
    let first_step = first_step.ok_or_else(|| anyhow!("missing first MoE SGD step summary"))?;

    let final_forward = tch_moe_forward(
        &input,
        &target,
        &router,
        &expert_up,
        &expert_down,
        num_experts,
        aux_loss_weight,
    )?;
    let final_loss = final_forward.loss.double_value(&[]);
    let final_task_loss = final_forward.task_loss.double_value(&[]);
    let final_load_balance_loss = final_forward.load_balance_loss.double_value(&[]);
    let checkpoint_output = std::env::var("RUSTRAIN_TCH_MOE_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::temp_dir().join(format!(
                "rustrain-tch-moe-{}.safetensors",
                std::process::id()
            ))
        });
    write_tch_moe_checkpoint(&checkpoint_output, &router, &expert_up, &expert_down)?;
    let (mut reloaded_router, mut reloaded_expert_up, mut reloaded_expert_down) =
        read_tch_moe_checkpoint(&checkpoint_output, device)?;
    let reloaded_forward = tch_moe_forward(
        &input,
        &target,
        &reloaded_router,
        &reloaded_expert_up,
        &reloaded_expert_down,
        num_experts,
        aux_loss_weight,
    )?;
    let reloaded_loss = reloaded_forward.loss.double_value(&[]);
    let reload_delta = (reloaded_loss - final_loss).abs();

    let mut continuous_router = router.detach().set_requires_grad(true);
    let mut continuous_expert_up = expert_up.detach().set_requires_grad(true);
    let mut continuous_expert_down = expert_down.detach().set_requires_grad(true);
    let continuous_second = tch_moe_sgd_step(
        &input,
        &target,
        &mut continuous_router,
        &mut continuous_expert_up,
        &mut continuous_expert_down,
        num_experts,
        aux_loss_weight,
        learning_rate,
    )?;
    let resumed_second = tch_moe_sgd_step(
        &input,
        &target,
        &mut reloaded_router,
        &mut reloaded_expert_up,
        &mut reloaded_expert_down,
        num_experts,
        aux_loss_weight,
        learning_rate,
    )?;
    let continuous_second_loss = continuous_second.loss_before;
    let resumed_second_loss = resumed_second.loss_before;
    let second_step_delta = (continuous_second_loss - resumed_second_loss).abs();
    let second_step_router_max_abs = tensor_max_abs_diff(&continuous_router, &reloaded_router)?;
    let second_step_expert_up_max_abs =
        tensor_max_abs_diff(&continuous_expert_up, &reloaded_expert_up)?;
    let second_step_expert_down_max_abs =
        tensor_max_abs_diff(&continuous_expert_down, &reloaded_expert_down)?;

    if final_loss >= initial_loss {
        bail!(
            "tch MoE smoke did not reduce loss: initial_loss={initial_loss}, final_loss={final_loss}"
        );
    }
    if final_task_loss >= initial_task_loss {
        bail!(
            "tch MoE smoke did not reduce task loss: initial_task_loss={initial_task_loss}, final_task_loss={final_task_loss}"
        );
    }
    if !first_step.router_grad_defined
        || !first_step.expert_up_grad_defined
        || !first_step.expert_down_grad_defined
    {
        bail!(
            "tch MoE smoke missing gradients: router={}, expert_up={}, expert_down={}",
            first_step.router_grad_defined,
            first_step.expert_up_grad_defined,
            first_step.expert_down_grad_defined
        );
    }
    if first_step.router_grad_norm <= 0.0
        || first_step.expert_up_grad_norm <= 0.0
        || first_step.expert_down_grad_norm <= 0.0
    {
        bail!(
            "tch MoE smoke gradients must be positive: router={}, expert_up={}, expert_down={}",
            first_step.router_grad_norm,
            first_step.expert_up_grad_norm,
            first_step.expert_down_grad_norm
        );
    }
    if reload_delta > 1e-7
        || second_step_delta > 1e-7
        || second_step_router_max_abs > 1e-7
        || second_step_expert_up_max_abs > 1e-7
        || second_step_expert_down_max_abs > 1e-7
    {
        bail!(
            "tch MoE checkpoint resume parity failed: reload_delta={reload_delta}, second_step_delta={second_step_delta}, router={second_step_router_max_abs}, expert_up={second_step_expert_up_max_abs}, expert_down={second_step_expert_down_max_abs}"
        );
    }

    let total_params = (hidden_size * num_experts
        + num_experts * hidden_size * expert_hidden_size
        + num_experts * expert_hidden_size * hidden_size) as usize;
    let activated_params = (hidden_size * num_experts
        + top_k * (hidden_size * expert_hidden_size + expert_hidden_size * hidden_size))
        as usize;

    Ok(TchMoeSmokeSummary {
        device: format!("{device:?}"),
        train_steps,
        learning_rate,
        aux_loss_weight,
        tokens,
        hidden_size,
        expert_hidden_size,
        num_experts,
        top_k,
        initial_loss,
        final_loss,
        initial_task_loss,
        final_task_loss,
        initial_load_balance_loss,
        final_load_balance_loss,
        checkpoint_output: checkpoint_output.display().to_string(),
        reloaded_loss,
        reload_delta,
        continuous_second_loss,
        resumed_second_loss,
        second_step_delta,
        second_step_router_max_abs,
        second_step_expert_up_max_abs,
        second_step_expert_down_max_abs,
        expert_load: final_forward.expert_load,
        total_params,
        activated_params,
        router_grad_defined: first_step.router_grad_defined,
        expert_up_grad_defined: first_step.expert_up_grad_defined,
        expert_down_grad_defined: first_step.expert_down_grad_defined,
        router_grad_norm: first_step.router_grad_norm,
        expert_up_grad_norm: first_step.expert_up_grad_norm,
        expert_down_grad_norm: first_step.expert_down_grad_norm,
        router_delta_norm: tensor_l2_norm(
            &(router.detach().to_device(Device::Cpu) - &initial_router),
        ),
        expert_up_delta_norm: tensor_l2_norm(
            &(expert_up.detach().to_device(Device::Cpu) - &initial_expert_up),
        ),
        expert_down_delta_norm: tensor_l2_norm(
            &(expert_down.detach().to_device(Device::Cpu) - &initial_expert_down),
        ),
        memory_rss_mb: memory_rss_mb(),
        gpu_memory_allocated_mb: crate::metrics::gpu_memory_allocated_mb(),
    })
}

pub fn train_tch_tiny_lm(config: &Config) -> Result<TchTrainSmokeSummary> {
    if config.train.max_steps == 0 {
        return Err(anyhow!("tch_tiny_lm requires train.max_steps > 0"));
    }
    if config.parallel.data_parallel_size > 1 {
        return train_tch_tiny_lm_data_parallel(config);
    }

    let device = tch_device(config)?;
    let compute_kind = tch_compute_kind(config);
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let embedding = nn::embedding(
        &root / "embed_tokens",
        config.model.vocab_size as i64,
        config.model.hidden_size as i64,
        Default::default(),
    );
    let lm_head = nn::linear(
        &root / "lm_head",
        config.model.hidden_size as i64,
        config.model.vocab_size as i64,
        nn::LinearConfig {
            bias: false,
            ..Default::default()
        },
    );
    let mut optimizer = nn::AdamW {
        beta1: config.train.adam_beta1 as f64,
        beta2: config.train.adam_beta2 as f64,
        wd: config.train.weight_decay as f64,
        eps: config.train.adam_eps as f64,
        amsgrad: false,
    }
    .build(&vs, config.train.learning_rate as f64)?;

    let input_ids = fixed_tch_batch(config.model.vocab_size as i64, config.model.seq_len as i64)
        .to_device(device);
    let targets = input_ids.narrow(1, 1, config.model.seq_len as i64 - 1);
    let initial_loss =
        tch_lm_loss(&embedding, &lm_head, &input_ids, &targets, compute_kind).double_value(&[]);
    let mut embedding_grad_defined = false;
    let mut lm_head_grad_defined = false;
    let mut first_step_grad_norm = 0.0;
    let mut final_learning_rate = config.train.learning_rate as f64;

    for step in 1..=config.train.max_steps {
        let learning_rate = learning_rate_for_step(config, step);
        optimizer.set_lr(learning_rate);
        optimizer.zero_grad();
        let loss = tch_lm_loss(&embedding, &lm_head, &input_ids, &targets, compute_kind);
        loss.backward();
        let grad_norm = grad_norm(&vs.trainable_variables());
        if let Some(max_grad_norm) = config.train.max_grad_norm {
            optimizer.clip_grad_norm(max_grad_norm as f64);
        }
        if step == 1 {
            embedding_grad_defined = embedding.ws.grad().defined();
            lm_head_grad_defined = lm_head.ws.grad().defined();
            first_step_grad_norm = grad_norm;
        }
        final_learning_rate = learning_rate;
        optimizer.step();

        if step == 1 || step == config.train.max_steps || step % 10 == 0 {
            info!(
                step,
                loss = loss.double_value(&[]),
                lr = learning_rate,
                grad_norm,
                "tch tiny lm train step"
            );
        }
    }

    let final_loss =
        tch_lm_loss(&embedding, &lm_head, &input_ids, &targets, compute_kind).double_value(&[]);
    if final_loss >= initial_loss {
        return Err(anyhow!(
            "tch tiny lm failed to reduce loss: initial_loss={initial_loss}, final_loss={final_loss}"
        ));
    }
    if !embedding_grad_defined || !lm_head_grad_defined {
        return Err(anyhow!(
            "tch tiny lm missing gradients: embedding_grad_defined={embedding_grad_defined}, lm_head_grad_defined={lm_head_grad_defined}"
        ));
    }

    Ok(TchTrainSmokeSummary {
        initial_loss,
        final_loss,
        embedding_grad_defined,
        lm_head_grad_defined,
        first_step_grad_norm,
        final_learning_rate,
        compute_kind: format!("{compute_kind:?}"),
        memory_rss_mb: memory_rss_mb(),
        gpu_memory_allocated_mb: crate::metrics::gpu_memory_allocated_mb(),
        data_parallel_size: config.parallel.data_parallel_size,
        dp_grad_max_delta: None,
        dp_loss_delta: None,
    })
}

fn train_tch_tiny_lm_data_parallel(config: &Config) -> Result<TchTrainSmokeSummary> {
    if config.parallel.data_parallel_size != 2 {
        bail!("tch_tiny_lm data-parallel smoke currently expects data_parallel_size=2");
    }
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != config.parallel.data_parallel_size {
        bail!(
            "WORLD_SIZE={world_size} does not match data_parallel_size={}",
            config.parallel.data_parallel_size
        );
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }

    let device = match config.train.device {
        RuntimeDevice::Cuda => Device::Cuda(local_rank),
        RuntimeDevice::Cpu => bail!("tch_tiny_lm data-parallel smoke requires device=cuda"),
    };
    let local = tch_dp_gradient_for_rank(rank, world_size, device)?;
    let reduce_dir = std::env::var("RUSTRAIN_LAUNCH_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            config
                .run
                .base_dir
                .join("tch-trainer-dp-gradient")
                .join(&config.run.name)
        })
        .join("trainer-dp-reduce");
    let reduced = nccl_smoke::all_reduce_f32_for_launch(&reduce_dir.join("grad"), &local.grad)?;
    let averaged_grad = reduced
        .iter()
        .map(|value| *value / world_size as f32)
        .collect::<Vec<_>>();
    let reduced_loss =
        nccl_smoke::all_reduce_f32_for_launch(&reduce_dir.join("loss"), &[local.loss as f32])?[0];
    let global_loss = reduced_loss as f64 / world_size as f64;

    let expected = tch_dp_gradient_for_rank(0, 1, device)?;
    let grad_max_delta = averaged_grad
        .iter()
        .copied()
        .zip(expected.grad.iter().copied())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let loss_delta = (global_loss - expected.loss).abs();
    if grad_max_delta > 1e-5 || loss_delta > 1e-5 {
        bail!(
            "trainer tch DP gradient mismatch: rank={rank}, grad_max_delta={grad_max_delta}, loss_delta={loss_delta}"
        );
    }

    Ok(TchTrainSmokeSummary {
        initial_loss: expected.loss,
        final_loss: global_loss,
        embedding_grad_defined: true,
        lm_head_grad_defined: true,
        first_step_grad_norm: averaged_grad
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>()
            .sqrt(),
        final_learning_rate: learning_rate_for_step(config, 1),
        compute_kind: format!("{:?}", tch_compute_kind(config)),
        memory_rss_mb: memory_rss_mb(),
        gpu_memory_allocated_mb: crate::metrics::gpu_memory_allocated_mb(),
        data_parallel_size: world_size,
        dp_grad_max_delta: Some(grad_max_delta),
        dp_loss_delta: Some(loss_delta),
    })
}

pub fn run_tch_dp_gradient_rank_smoke(output_dir: PathBuf) -> Result<()> {
    let rank = parse_env_usize("RANK")?;
    let local_rank = parse_env_usize("LOCAL_RANK")?;
    let world_size = parse_env_usize("WORLD_SIZE")?;
    if world_size != 2 {
        bail!("tch DP gradient smoke expects WORLD_SIZE=2");
    }
    if rank >= world_size {
        bail!("rank {rank} must be smaller than world_size {world_size}");
    }
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let device = Device::Cuda(local_rank);
    let local = tch_dp_gradient_for_rank(rank, world_size, device)?;
    let reduced = nccl_smoke::all_reduce_f32_for_launch(&output_dir, &local.grad)?;
    let averaged_grad = reduced
        .iter()
        .map(|value| *value / world_size as f32)
        .collect::<Vec<_>>();
    let reduced_loss =
        nccl_smoke::all_reduce_f32_for_launch(&output_dir.join("loss"), &[local.loss as f32])?[0];
    let global_loss = reduced_loss as f64 / world_size as f64;

    let expected = tch_dp_gradient_for_rank(0, 1, device)?;
    let grad_max_delta = averaged_grad
        .iter()
        .copied()
        .zip(expected.grad.iter().copied())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let loss_delta = (global_loss - expected.loss).abs();
    if grad_max_delta > 1e-5 || loss_delta > 1e-5 {
        bail!(
            "tch DP gradient mismatch: rank={rank}, grad_max_delta={grad_max_delta}, loss_delta={loss_delta}"
        );
    }

    let summary = TchDpGradientRankSummary {
        rank,
        world_size,
        local_rank,
        local_loss: local.loss,
        global_loss,
        expected_loss: expected.loss,
        loss_delta,
        reduced_grad: averaged_grad,
        expected_grad: expected.grad,
        grad_max_delta,
    };
    let summary_path = output_dir.join(format!("tch-dp-gradient-rank-{rank}.json"));
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);

    Ok(())
}

fn learning_rate_for_step(config: &Config, step: u64) -> f64 {
    match config.train.lr_scheduler {
        LrScheduler::Constant => config.train.learning_rate as f64,
        LrScheduler::LinearDecay => {
            let max_steps = config.train.max_steps.max(1) as f64;
            let progress = (step.saturating_sub(1) as f64 / max_steps).clamp(0.0, 1.0);
            config.train.learning_rate as f64 * (1.0 - progress)
        }
    }
}

fn tch_moe_forward(
    input: &Tensor,
    target: &Tensor,
    router: &Tensor,
    expert_up: &Tensor,
    expert_down: &Tensor,
    num_experts: i64,
    aux_loss_weight: f64,
) -> Result<TchMoeForward> {
    let logits = input.matmul(router);
    let routing_probs = logits.softmax(-1, Kind::Float);
    let expert_indices = routing_probs.argmax(-1, false);
    let selected_weights = routing_probs
        .gather(1, &expert_indices.unsqueeze(1), false)
        .squeeze_dim(1);

    let mut token_outputs = Vec::with_capacity(input.size()[0] as usize);
    let mut expert_load = vec![0usize; num_experts as usize];
    for token_index in 0..input.size()[0] {
        let expert_index = expert_indices.int64_value(&[token_index]);
        expert_load[expert_index as usize] += 1;
        let token = input.get(token_index).unsqueeze(0);
        let up = expert_up.get(expert_index);
        let down = expert_down.get(expert_index);
        let output = token.matmul(&up).relu().matmul(&down).squeeze_dim(0)
            * selected_weights.get(token_index);
        token_outputs.push(output);
    }
    let output = Tensor::stack(&token_outputs.iter().collect::<Vec<_>>(), 0);
    let task_loss = output.mse_loss(target, Reduction::Mean);
    let load = Tensor::from_slice(
        &expert_load
            .iter()
            .map(|load| *load as f32)
            .collect::<Vec<_>>(),
    )
    .to_device(input.device());
    let load_fraction = &load / load.sum(Kind::Float).clamp_min(1.0);
    let prob_fraction = routing_probs.mean_dim([0].as_slice(), false, Kind::Float);
    let expected = Tensor::full(
        [num_experts],
        1.0 / num_experts as f64,
        (Kind::Float, input.device()),
    );
    let load_balance_loss = (&load_fraction - &expected).square().mean(Kind::Float)
        + (&prob_fraction - &expected).square().mean(Kind::Float);
    let loss = &task_loss + &load_balance_loss * aux_loss_weight;

    Ok(TchMoeForward {
        loss,
        task_loss,
        load_balance_loss,
        expert_load,
    })
}

fn tch_moe_sgd_step(
    input: &Tensor,
    target: &Tensor,
    router: &mut Tensor,
    expert_up: &mut Tensor,
    expert_down: &mut Tensor,
    num_experts: i64,
    aux_loss_weight: f64,
    learning_rate: f64,
) -> Result<TchMoeSgdStep> {
    router.zero_grad();
    expert_up.zero_grad();
    expert_down.zero_grad();
    let forward = tch_moe_forward(
        input,
        target,
        router,
        expert_up,
        expert_down,
        num_experts,
        aux_loss_weight,
    )?;
    let loss_before = forward.loss.double_value(&[]);
    forward.loss.backward();

    let router_grad_defined = router.grad().defined();
    let expert_up_grad_defined = expert_up.grad().defined();
    let expert_down_grad_defined = expert_down.grad().defined();
    let router_grad_norm = tensor_l2_norm(&router.grad());
    let expert_up_grad_norm = tensor_l2_norm(&expert_up.grad());
    let expert_down_grad_norm = tensor_l2_norm(&expert_down.grad());

    no_grad(|| -> Result<()> {
        let _ = router.f_sub_(&(&router.grad() * learning_rate))?;
        let _ = expert_up.f_sub_(&(&expert_up.grad() * learning_rate))?;
        let _ = expert_down.f_sub_(&(&expert_down.grad() * learning_rate))?;
        Ok(())
    })?;

    Ok(TchMoeSgdStep {
        loss_before,
        router_grad_defined,
        expert_up_grad_defined,
        expert_down_grad_defined,
        router_grad_norm,
        expert_up_grad_norm,
        expert_down_grad_norm,
    })
}

fn write_tch_moe_checkpoint(
    path: &Path,
    router: &Tensor,
    expert_up: &Tensor,
    expert_down: &Tensor,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    Tensor::write_safetensors(
        &[
            ("router.weight", router),
            ("experts.up.weight", expert_up),
            ("experts.down.weight", expert_down),
        ],
        path,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn read_tch_moe_checkpoint(path: &Path, device: Device) -> Result<(Tensor, Tensor, Tensor)> {
    let tensors = Tensor::read_safetensors(path)
        .with_context(|| format!("failed to read {}", path.display()))?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let router = tensor_from_map(&tensors, "router.weight")?
        .to_device(device)
        .set_requires_grad(true);
    let expert_up = tensor_from_map(&tensors, "experts.up.weight")?
        .to_device(device)
        .set_requires_grad(true);
    let expert_down = tensor_from_map(&tensors, "experts.down.weight")?
        .to_device(device)
        .set_requires_grad(true);
    Ok((router, expert_up, expert_down))
}

fn tensor_from_map<'a>(tensors: &'a BTreeMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| anyhow!("missing tensor {name}"))
}

struct TchDpGradient {
    loss: f64,
    grad: Vec<f32>,
}

fn tch_dp_gradient_for_rank(
    rank: usize,
    world_size: usize,
    device: Device,
) -> Result<TchDpGradient> {
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let mut weight = root.var("weight", &[2, 2], Init::Const(0.0));
    let _ = no_grad(|| {
        weight.f_copy_(
            &Tensor::from_slice(&[0.2_f32, -0.1, 0.4, 0.3])
                .reshape([2, 2])
                .to_device(device),
        )
    })
    .context("failed to initialize DP smoke weight")?;

    let (input, target) = tch_dp_batch_for_rank(rank, world_size, device)?;
    let logits = input.matmul(&weight);
    let loss = logits.mse_loss(&target, Reduction::Mean);
    let loss_value = loss.double_value(&[]);
    loss.backward();
    let grad = weight.grad();
    if !grad.defined() {
        bail!("tch DP smoke weight gradient is not defined");
    }
    let grad = tensor_to_f32_vec(&grad)?;

    Ok(TchDpGradient {
        loss: loss_value,
        grad,
    })
}

fn tch_dp_batch_for_rank(
    rank: usize,
    world_size: usize,
    device: Device,
) -> Result<(Tensor, Tensor)> {
    const INPUTS: [[f32; 2]; 4] = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]];
    const TARGETS: [[f32; 2]; 4] = [[0.5, -0.2], [0.1, 0.3], [0.7, 0.0], [1.0, -0.5]];

    let mut input_values = Vec::new();
    let mut target_values = Vec::new();
    for (sample_index, (input, target)) in INPUTS.iter().zip(TARGETS).enumerate() {
        if sample_index % world_size != rank {
            continue;
        }
        input_values.extend_from_slice(input);
        target_values.extend_from_slice(&target);
    }
    let sample_count = input_values.len() / 2;
    if sample_count == 0 {
        bail!("rank {rank} received no samples for world_size {world_size}");
    }

    let input = Tensor::from_slice(&input_values)
        .reshape([sample_count as i64, 2])
        .to_device(device);
    let target = Tensor::from_slice(&target_values)
        .reshape([sample_count as i64, 2])
        .to_device(device);
    Ok((input, target))
}

fn tensor_to_f32_vec(tensor: &Tensor) -> Result<Vec<f32>> {
    let tensor = tensor
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .reshape([-1]);
    Vec::<f32>::try_from(&tensor)
        .map_err(|error| anyhow!("failed to copy tensor to Vec<f32>: {error}"))
}

fn parse_env_usize(name: &str) -> Result<usize> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set; run through rustrain launch"))?
        .parse::<usize>()
        .with_context(|| format!("{name} must be a usize"))
}

fn grad_norm(trainable_variables: &[Tensor]) -> f64 {
    trainable_variables
        .iter()
        .filter_map(|tensor| {
            let grad = tensor.grad();
            grad.defined()
                .then(|| grad.square().sum(Kind::Float).double_value(&[]))
        })
        .sum::<f64>()
        .sqrt()
}

fn tensor_l2_norm(tensor: &Tensor) -> f64 {
    tensor.square().sum(Kind::Float).sqrt().double_value(&[])
}

fn tensor_max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<f64> {
    if actual.size() != expected.size() {
        bail!(
            "shape mismatch: actual {:?}, expected {:?}",
            actual.size(),
            expected.size()
        );
    }
    Ok((actual - expected)
        .abs()
        .max()
        .to_device(Device::Cpu)
        .double_value(&[]))
}

fn fixed_tch_batch(vocab_size: i64, seq_len: i64) -> Tensor {
    let tokens: Vec<i64> = (0..seq_len).map(|position| position % vocab_size).collect();
    Tensor::from_slice(&tokens).reshape([1, seq_len])
}

fn tch_device(config: &Config) -> Result<Device> {
    match config.train.device {
        RuntimeDevice::Cpu => Ok(Device::Cpu),
        RuntimeDevice::Cuda => {
            if Cuda::is_available() {
                Ok(Device::Cuda(0))
            } else {
                Err(anyhow!(
                    "config requested device=cuda, but tch CUDA is not available"
                ))
            }
        }
    }
}

fn tch_compute_kind(config: &Config) -> Kind {
    match config.train.dtype {
        DType::Fp32 => Kind::Float,
        DType::Fp16 => Kind::Half,
        DType::Bf16 => Kind::BFloat16,
    }
}

fn tch_lm_loss(
    embedding: &nn::Embedding,
    lm_head: &nn::Linear,
    input_ids: &Tensor,
    targets: &Tensor,
    compute_kind: Kind,
) -> Tensor {
    let hidden = embedding.forward(input_ids).to_kind(compute_kind);
    let hidden = hidden.narrow(1, 0, input_ids.size()[1] - 1);
    let weight = lm_head.ws.to_kind(compute_kind);
    let logits = hidden
        .linear::<&Tensor>(&weight, lm_head.bs.as_ref())
        .to_kind(Kind::Float);
    let vocab_size = logits.size()[2];
    logits
        .reshape([-1, vocab_size])
        .log_softmax(-1, Kind::Float)
        .g_nll_loss::<&Tensor>(&targets.reshape([-1]), None, Reduction::Mean, -100)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendKind;
    use crate::runtime::{
        Config, DType, Device as RuntimeDevice, ModelConfig, ParallelConfig, RunConfig, TrainConfig,
    };

    #[test]
    fn tch_tiny_lm_trains_all_parameter_groups() {
        let config = tiny_tch_config(RuntimeDevice::Cpu);

        let summary = train_tch_tiny_lm(&config).expect("tch tiny lm should train");

        assert!(summary.final_loss < summary.initial_loss);
        assert!(summary.embedding_grad_defined);
        assert!(summary.lm_head_grad_defined);
        assert!(summary.first_step_grad_norm > 0.0);
        assert!((summary.final_learning_rate - 1e-2).abs() < 1e-8);
        assert_eq!(summary.compute_kind, "Float");
        assert!(summary.memory_rss_mb.is_none_or(|value| value > 0.0));
        assert!(
            summary
                .gpu_memory_allocated_mb
                .is_none_or(|value| value >= 0.0)
        );
    }

    #[test]
    fn tch_cuda_request_fails_clearly_when_unavailable() {
        if tch::Cuda::is_available() {
            return;
        }
        let config = tiny_tch_config(RuntimeDevice::Cuda);

        let error = train_tch_tiny_lm(&config).expect_err("missing CUDA should fail");

        assert!(
            error.to_string().contains("tch CUDA is not available"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn tch_tiny_lm_applies_linear_decay_and_reports_grad_norm() {
        let mut config = tiny_tch_config(RuntimeDevice::Cpu);
        config.train.lr_scheduler = crate::runtime::LrScheduler::LinearDecay;
        config.train.max_grad_norm = Some(0.01);

        let summary = train_tch_tiny_lm(&config).expect("tch tiny lm should train");

        assert!(summary.first_step_grad_norm > 0.0);
        assert!(summary.final_learning_rate < config.train.learning_rate as f64);
    }

    #[test]
    fn tch_dtype_policy_maps_runtime_dtype_to_compute_kind() {
        let mut config = tiny_tch_config(RuntimeDevice::Cpu);
        config.train.dtype = DType::Fp32;
        assert_eq!(tch_compute_kind(&config), Kind::Float);
        config.train.dtype = DType::Fp16;
        assert_eq!(tch_compute_kind(&config), Kind::Half);
        config.train.dtype = DType::Bf16;
        assert_eq!(tch_compute_kind(&config), Kind::BFloat16);
    }

    #[test]
    fn tch_dp_gradient_partitions_match_global_batch() {
        let rank0 =
            tch_dp_gradient_for_rank(0, 2, Device::Cpu).expect("rank0 gradient should compute");
        let rank1 =
            tch_dp_gradient_for_rank(1, 2, Device::Cpu).expect("rank1 gradient should compute");
        let single =
            tch_dp_gradient_for_rank(0, 1, Device::Cpu).expect("global gradient should compute");

        let averaged_grad = rank0
            .grad
            .iter()
            .zip(rank1.grad.iter())
            .map(|(left, right)| (left + right) / 2.0)
            .collect::<Vec<_>>();
        for (actual, expected) in averaged_grad.into_iter().zip(single.grad) {
            assert!((actual - expected).abs() < 1e-6);
        }
        let global_loss = (rank0.loss + rank1.loss) / 2.0;
        assert!((global_loss - single.loss).abs() < 1e-6);
    }

    fn tiny_tch_config(device: RuntimeDevice) -> Config {
        Config {
            run: RunConfig {
                name: "test".to_string(),
                base_dir: "runs".into(),
                seed: 0,
            },
            model: ModelConfig {
                name: "tch_tiny_lm".to_string(),
                architecture: "tch_tiny_lm".to_string(),
                model_path: None,
                vocab_size: 16,
                hidden_size: 8,
                num_layers: 1,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                intermediate_size: 16,
                seq_len: 8,
                norm: "rmsnorm".to_string(),
                activation: "swiglu".to_string(),
                rope: true,
                rms_norm_eps: 1e-6,
                trainable_layers: None,
            },
            train: TrainConfig {
                max_steps: 3,
                resume_from: None,
                backend: BackendKind::Tch,
                micro_batch_size: 1,
                global_batch_size: 1,
                gradient_accumulation_steps: 1,
                learning_rate: 1e-2,
                weight_decay: 0.0,
                adam_beta1: 0.9,
                adam_beta2: 0.999,
                adam_eps: 1e-8,
                lr_scheduler: crate::runtime::LrScheduler::Constant,
                max_grad_norm: None,
                dtype: DType::Fp32,
                device,
                checkpoint_every: 0,
                eval_every: 0,
            },
            data: None,
            lora: None,
            parallel: ParallelConfig {
                tensor_model_parallel_size: 1,
                pipeline_model_parallel_size: 1,
                data_parallel_size: 1,
                expert_model_parallel_size: 1,
                context_parallel_size: 1,
            },
        }
    }
}
