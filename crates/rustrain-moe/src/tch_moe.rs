use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use tch::{Cuda, Device, Kind, Reduction, Tensor, no_grad};

use rustrain_train::metrics::{gpu_memory_allocated_mb, memory_rss_mb};

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
    optimizer_output: String,
    manifest_output: String,
    reloaded_loss: f64,
    reload_delta: f64,
    reload_optimizer_max_abs: f64,
    continuous_second_loss: f64,
    resumed_second_loss: f64,
    second_step_delta: f64,
    second_step_router_max_abs: f64,
    second_step_expert_up_max_abs: f64,
    second_step_expert_down_max_abs: f64,
    second_step_optimizer_max_abs: f64,
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

struct TchMoeAdamState {
    router_m: Tensor,
    router_v: Tensor,
    expert_up_m: Tensor,
    expert_up_v: Tensor,
    expert_down_m: Tensor,
    expert_down_v: Tensor,
    step: i64,
}

struct TchMoeTrainStep {
    loss_before: f64,
    router_grad_defined: bool,
    expert_up_grad_defined: bool,
    expert_down_grad_defined: bool,
    router_grad_norm: f64,
    expert_up_grad_norm: f64,
    expert_down_grad_norm: f64,
    updated_state: TchMoeAdamState,
}

struct TchMoeCheckpoint {
    router: Tensor,
    expert_up: Tensor,
    expert_down: Tensor,
    state: TchMoeAdamState,
}

#[derive(Debug, serde::Serialize)]
struct TchMoeCheckpointManifest {
    format: String,
    global_step: i64,
    model_safetensors: String,
    optimizer_safetensors: String,
    model_tensors: Vec<TchMoeTensorManifest>,
    optimizer_slots: Vec<TchMoeTensorManifest>,
    optimizer_step_tensor: TchMoeTensorManifest,
}

#[derive(Debug, serde::Serialize)]
struct TchMoeTensorManifest {
    name: String,
    shape: Vec<i64>,
    dtype: String,
}

impl TchMoeAdamState {
    fn zeros_like(router: &Tensor, expert_up: &Tensor, expert_down: &Tensor) -> Self {
        Self {
            router_m: Tensor::zeros_like(router),
            router_v: Tensor::zeros_like(router),
            expert_up_m: Tensor::zeros_like(expert_up),
            expert_up_v: Tensor::zeros_like(expert_up),
            expert_down_m: Tensor::zeros_like(expert_down),
            expert_down_v: Tensor::zeros_like(expert_down),
            step: 0,
        }
    }

    fn clone_state(&self) -> Self {
        Self {
            router_m: self.router_m.shallow_clone(),
            router_v: self.router_v.shallow_clone(),
            expert_up_m: self.expert_up_m.shallow_clone(),
            expert_up_v: self.expert_up_v.shallow_clone(),
            expert_down_m: self.expert_down_m.shallow_clone(),
            expert_down_v: self.expert_down_v.shallow_clone(),
            step: self.step,
        }
    }

    fn max_abs_diff(&self, other: &Self) -> Result<f64> {
        Ok([
            tensor_max_abs_diff(&self.router_m, &other.router_m)?,
            tensor_max_abs_diff(&self.router_v, &other.router_v)?,
            tensor_max_abs_diff(&self.expert_up_m, &other.expert_up_m)?,
            tensor_max_abs_diff(&self.expert_up_v, &other.expert_up_v)?,
            tensor_max_abs_diff(&self.expert_down_m, &other.expert_down_m)?,
            tensor_max_abs_diff(&self.expert_down_v, &other.expert_down_v)?,
            (self.step - other.step).abs() as f64,
        ]
        .into_iter()
        .fold(0.0_f64, f64::max))
    }
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
    let learning_rate = 0.01;
    let aux_loss_weight = 0.01;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let weight_decay = 0.0;

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

    let mut state = TchMoeAdamState::zeros_like(&router, &expert_up, &expert_down);
    let mut first_step: Option<TchMoeTrainStep> = None;

    for step in 1..=train_steps {
        let step_summary = tch_moe_adamw_step(
            &input,
            &target,
            &mut router,
            &mut expert_up,
            &mut expert_down,
            num_experts,
            aux_loss_weight,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            &state,
        )?;
        state = step_summary.updated_state.clone_state();
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
    let optimizer_output = checkpoint_output.with_file_name(format!(
        "{}.optimizer.safetensors",
        checkpoint_output
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("rustrain-tch-moe")
    ));
    let manifest_output = tch_moe_manifest_path(&checkpoint_output);
    write_tch_moe_checkpoint(
        &checkpoint_output,
        &optimizer_output,
        &manifest_output,
        &router,
        &expert_up,
        &expert_down,
        &state,
    )?;
    let TchMoeCheckpoint {
        router: mut reloaded_router,
        expert_up: mut reloaded_expert_up,
        expert_down: mut reloaded_expert_down,
        state: reloaded_state,
    } = read_tch_moe_checkpoint(&checkpoint_output, device)?;
    let reload_optimizer_max_abs = state.max_abs_diff(&reloaded_state)?;
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
    let continuous_second = tch_moe_adamw_step(
        &input,
        &target,
        &mut continuous_router,
        &mut continuous_expert_up,
        &mut continuous_expert_down,
        num_experts,
        aux_loss_weight,
        learning_rate,
        beta1,
        beta2,
        eps,
        weight_decay,
        &state,
    )?;
    let resumed_second = tch_moe_adamw_step(
        &input,
        &target,
        &mut reloaded_router,
        &mut reloaded_expert_up,
        &mut reloaded_expert_down,
        num_experts,
        aux_loss_weight,
        learning_rate,
        beta1,
        beta2,
        eps,
        weight_decay,
        &reloaded_state,
    )?;
    let continuous_second_loss = continuous_second.loss_before;
    let resumed_second_loss = resumed_second.loss_before;
    let second_step_delta = (continuous_second_loss - resumed_second_loss).abs();
    let second_step_router_max_abs = tensor_max_abs_diff(&continuous_router, &reloaded_router)?;
    let second_step_expert_up_max_abs =
        tensor_max_abs_diff(&continuous_expert_up, &reloaded_expert_up)?;
    let second_step_expert_down_max_abs =
        tensor_max_abs_diff(&continuous_expert_down, &reloaded_expert_down)?;
    let second_step_optimizer_max_abs = continuous_second
        .updated_state
        .max_abs_diff(&resumed_second.updated_state)?;

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
        || reload_optimizer_max_abs > 1e-7
        || second_step_delta > 1e-7
        || second_step_router_max_abs > 1e-7
        || second_step_expert_up_max_abs > 1e-7
        || second_step_expert_down_max_abs > 1e-7
        || second_step_optimizer_max_abs > 1e-7
    {
        bail!(
            "tch MoE checkpoint resume parity failed: reload_delta={reload_delta}, reload_optimizer_max_abs={reload_optimizer_max_abs}, second_step_delta={second_step_delta}, router={second_step_router_max_abs}, expert_up={second_step_expert_up_max_abs}, expert_down={second_step_expert_down_max_abs}, optimizer={second_step_optimizer_max_abs}"
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
        optimizer_output: optimizer_output.display().to_string(),
        manifest_output: manifest_output.display().to_string(),
        reloaded_loss,
        reload_delta,
        reload_optimizer_max_abs,
        continuous_second_loss,
        resumed_second_loss,
        second_step_delta,
        second_step_router_max_abs,
        second_step_expert_up_max_abs,
        second_step_expert_down_max_abs,
        second_step_optimizer_max_abs,
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
        gpu_memory_allocated_mb: gpu_memory_allocated_mb(),
    })
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

fn tch_moe_adamw_step(
    input: &Tensor,
    target: &Tensor,
    router: &mut Tensor,
    expert_up: &mut Tensor,
    expert_down: &mut Tensor,
    num_experts: i64,
    aux_loss_weight: f64,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    state: &TchMoeAdamState,
) -> Result<TchMoeTrainStep> {
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

    let next_step = state.step + 1;
    let (router_update, router_m, router_v) = adamw_parameter_update(
        router,
        &router.grad(),
        &state.router_m,
        &state.router_v,
        learning_rate,
        beta1,
        beta2,
        next_step,
        eps,
        weight_decay,
    );
    let (expert_up_update, expert_up_m, expert_up_v) = adamw_parameter_update(
        expert_up,
        &expert_up.grad(),
        &state.expert_up_m,
        &state.expert_up_v,
        learning_rate,
        beta1,
        beta2,
        next_step,
        eps,
        weight_decay,
    );
    let (expert_down_update, expert_down_m, expert_down_v) = adamw_parameter_update(
        expert_down,
        &expert_down.grad(),
        &state.expert_down_m,
        &state.expert_down_v,
        learning_rate,
        beta1,
        beta2,
        next_step,
        eps,
        weight_decay,
    );

    no_grad(|| -> Result<()> {
        let _ = router.f_sub_(&router_update)?;
        let _ = expert_up.f_sub_(&expert_up_update)?;
        let _ = expert_down.f_sub_(&expert_down_update)?;
        Ok(())
    })?;

    Ok(TchMoeTrainStep {
        loss_before,
        router_grad_defined,
        expert_up_grad_defined,
        expert_down_grad_defined,
        router_grad_norm,
        expert_up_grad_norm,
        expert_down_grad_norm,
        updated_state: TchMoeAdamState {
            router_m,
            router_v,
            expert_up_m,
            expert_up_v,
            expert_down_m,
            expert_down_v,
            step: next_step,
        },
    })
}

fn adamw_parameter_update(
    parameter: &Tensor,
    grad: &Tensor,
    previous_m: &Tensor,
    previous_v: &Tensor,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    step: i64,
    eps: f64,
    weight_decay: f64,
) -> (Tensor, Tensor, Tensor) {
    let m = previous_m * beta1 + grad * (1.0 - beta1);
    let v = previous_v * beta2 + grad.square() * (1.0 - beta2);
    let bias_correction1 = 1.0 - beta1.powi(step as i32);
    let bias_correction2 = 1.0 - beta2.powi(step as i32);
    let m_hat = &m / bias_correction1;
    let v_hat = &v / bias_correction2;
    let adaptive_update = m_hat / (v_hat.sqrt() + eps);
    let update = if weight_decay == 0.0 {
        adaptive_update * learning_rate
    } else {
        (adaptive_update + parameter * weight_decay) * learning_rate
    };
    (update, m, v)
}

fn write_tch_moe_checkpoint(
    model_path: &Path,
    optimizer_path: &Path,
    manifest_path: &Path,
    router: &Tensor,
    expert_up: &Tensor,
    expert_down: &Tensor,
    state: &TchMoeAdamState,
) -> Result<()> {
    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    Tensor::write_safetensors(
        &[
            ("router.weight", router),
            ("experts.up.weight", expert_up),
            ("experts.down.weight", expert_down),
        ],
        model_path,
    )
    .with_context(|| format!("failed to write {}", model_path.display()))?;
    let optimizer_step = Tensor::from_slice(&[state.step]);
    Tensor::write_safetensors(
        &[
            ("router.weight.adam_m", &state.router_m),
            ("router.weight.adam_v", &state.router_v),
            ("experts.up.weight.adam_m", &state.expert_up_m),
            ("experts.up.weight.adam_v", &state.expert_up_v),
            ("experts.down.weight.adam_m", &state.expert_down_m),
            ("experts.down.weight.adam_v", &state.expert_down_v),
            ("optimizer.step", &optimizer_step),
        ],
        optimizer_path,
    )
    .with_context(|| format!("failed to write {}", optimizer_path.display()))?;
    let manifest = TchMoeCheckpointManifest {
        format: "rustrain.tch_moe.v1".to_string(),
        global_step: state.step,
        model_safetensors: model_path.display().to_string(),
        optimizer_safetensors: optimizer_path.display().to_string(),
        model_tensors: vec![
            tch_moe_tensor_manifest("router.weight", router),
            tch_moe_tensor_manifest("experts.up.weight", expert_up),
            tch_moe_tensor_manifest("experts.down.weight", expert_down),
        ],
        optimizer_slots: vec![
            tch_moe_tensor_manifest("router.weight.adam_m", &state.router_m),
            tch_moe_tensor_manifest("router.weight.adam_v", &state.router_v),
            tch_moe_tensor_manifest("experts.up.weight.adam_m", &state.expert_up_m),
            tch_moe_tensor_manifest("experts.up.weight.adam_v", &state.expert_up_v),
            tch_moe_tensor_manifest("experts.down.weight.adam_m", &state.expert_down_m),
            tch_moe_tensor_manifest("experts.down.weight.adam_v", &state.expert_down_v),
        ],
        optimizer_step_tensor: tch_moe_tensor_manifest("optimizer.step", &optimizer_step),
    };
    fs::write(
        manifest_path,
        serde_json::to_string_pretty(&manifest)? + "\n",
    )
    .with_context(|| format!("failed to write {}", manifest_path.display()))
}

fn read_tch_moe_checkpoint(model_path: &Path, device: Device) -> Result<TchMoeCheckpoint> {
    let optimizer_path = model_path.with_file_name(format!(
        "{}.optimizer.safetensors",
        model_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("rustrain-tch-moe")
    ));
    let tensors = Tensor::read_safetensors(model_path)
        .with_context(|| format!("failed to read {}", model_path.display()))?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let optimizer_tensors = Tensor::read_safetensors(&optimizer_path)
        .with_context(|| format!("failed to read {}", optimizer_path.display()))?
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
    let step = tensor_from_map(&optimizer_tensors, "optimizer.step")?
        .to_device(Device::Cpu)
        .int64_value(&[0]);
    Ok(TchMoeCheckpoint {
        router,
        expert_up,
        expert_down,
        state: TchMoeAdamState {
            router_m: tensor_from_map(&optimizer_tensors, "router.weight.adam_m")?
                .to_device(device),
            router_v: tensor_from_map(&optimizer_tensors, "router.weight.adam_v")?
                .to_device(device),
            expert_up_m: tensor_from_map(&optimizer_tensors, "experts.up.weight.adam_m")?
                .to_device(device),
            expert_up_v: tensor_from_map(&optimizer_tensors, "experts.up.weight.adam_v")?
                .to_device(device),
            expert_down_m: tensor_from_map(&optimizer_tensors, "experts.down.weight.adam_m")?
                .to_device(device),
            expert_down_v: tensor_from_map(&optimizer_tensors, "experts.down.weight.adam_v")?
                .to_device(device),
            step,
        },
    })
}

fn tch_moe_manifest_path(model_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.json", model_path.display()))
}

fn tch_moe_tensor_manifest(name: &str, tensor: &Tensor) -> TchMoeTensorManifest {
    TchMoeTensorManifest {
        name: name.to_string(),
        shape: tensor.size(),
        dtype: format!("{:?}", tensor.kind()),
    }
}

fn tensor_from_map<'a>(tensors: &'a BTreeMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| anyhow!("missing tensor {name}"))
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
