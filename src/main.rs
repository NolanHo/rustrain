// rustrain CLI: thin dispatch layer
// 4 user commands: train / inspect / launch / probe

mod inspect;
#[cfg(feature = "ray")]
mod ray_gpu;

use rustrain_core::runtime::{
    init_logging, load_config, prepare_run_directory, validate_config, write_resolved_config,
};
use rustrain_tch_tiny::tch_train;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use tracing::info;

#[derive(Debug, Parser)]
#[command(name = "rustrain")]
#[command(about = "A Rust LLM training engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Train a model from TOML config
    Train {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(long)]
        resume_from: Option<PathBuf>,
    },

    /// Inspect a HuggingFace model directory
    Inspect {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long, default_value = "rustrain")]
        prompt: String,
        #[arg(long, default_value_t = 12)]
        tensor_limit: usize,
    },

    /// Launch distributed rank processes
    Launch {
        #[arg(long)]
        nproc_per_node: usize,
        #[arg(long, default_value = "1")]
        nnodes: usize,
        #[arg(long, default_value = "0")]
        node_rank: usize,
        #[arg(long, default_value = "/tmp/rustrain-runs/launch")]
        output_dir: PathBuf,
        #[arg(long, default_value = "127.0.0.1")]
        master_addr: String,
        #[arg(long, default_value_t = 29500)]
        master_port: u16,
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        command: Vec<String>,
    },

    /// Probe tch-rs CUDA availability
    Probe,

    /// Start training server (HTTP + gRPC)
    Server {
        #[arg(long, default_value_t = 8080)]
        http_port: u16,
        #[arg(long, default_value_t = 50051)]
        grpc_port: u16,
        #[arg(long, default_value = "/tmp/rustrain-server")]
        metrics_dir: PathBuf,
    },

    /// Start EP server (HTTP server forks N GPU workers)
    EpServer {
        #[arg(long, default_value_t = 8080)]
        http_port: u16,
        #[arg(long, default_value_t = 50051)]
        grpc_port: u16,
        #[arg(long, default_value = "/tmp/rustrain-ep")]
        metrics_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        world_size: usize,
    },

    /// EP benchmark — no HTTP, direct IPC dispatch in-process
    EpBench {
        #[arg(long, default_value = "/tmp/rustrain-ep")]
        metrics_dir: PathBuf,
        #[arg(long, default_value_t = 8)]
        world_size: usize,
        #[arg(long, default_value_t = 100)]
        n_adapters: i32,
        #[arg(long, default_value_t = 1)]
        lora_rank: i32,
        #[arg(long, default_value_t = 60)]
        duration: u64, // seconds, 0 = single step
        #[arg(long, default_value = "512")]
        seq_len: usize,
    },

    /// EP worker process (launched by ep-server, not for direct use)
    EpWorker {
        #[arg(long)]
        shm_name: String,
        #[arg(long)]
        rank: usize,
        #[arg(long)]
        world_size: usize,
        #[arg(long)]
        metrics_path: PathBuf,
    },

    /// Run a command on a Ray GPU worker (via rayrust native SDK)
    #[cfg(feature = "ray")]
    RayGpu {
        #[arg(long, default_value = "1")]
        num_gpus: usize,
        #[arg(long)]
        ray_address: Option<String>,
        #[arg(long, default_value = "/vePFS-Mindverse/user/nolanho/code")]
        runner_path: String,
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        command: Vec<String>,
    },
}

fn main() -> Result<()> {
    // ── CUDA_VISIBLE_DEVICES fix for Ray workers ──────────────────────────
    // Ray strips CUDA_VISIBLE_DEVICES from child processes at the Python
    // _posixsubprocess level. When rustrain is launched via Ray, the env var
    // is empty even though GPUs are available.
    // Fix: if RUSTRAIN_FORCE_CUDA_VISIBLE_DEVICES is set, force-set
    // CUDA_VISIBLE_DEVICES before any CUDA initialization.
    // See docs/ray-gpu-env-injection.md for details.
    if let Ok(force_gpus) = std::env::var("RUSTRAIN_FORCE_CUDA_VISIBLE_DEVICES") {
        if !force_gpus.is_empty() {
            // SAFETY: Setting CUDA_VISIBLE_DEVICES before any CUDA initialization.
            // This is safe because no other threads are accessing the environment.
            unsafe {
                std::env::set_var("CUDA_VISIBLE_DEVICES", &force_gpus);
            }
            eprintln!(
                "rustrain: forced CUDA_VISIBLE_DEVICES={force_gpus} \
                 (from RUSTRAIN_FORCE_CUDA_VISIBLE_DEVICES)"
            );
        }
    }

    let cli = Cli::parse();

    match cli.command {
        Command::Train {
            config,
            resume_from,
        } => dispatch_train(&config, resume_from),
        Command::Inspect {
            model_path,
            prompt,
            tensor_limit,
        } => inspect::inspect_model(&model_path, &prompt, tensor_limit),
        Command::Launch {
            nproc_per_node,
            nnodes,
            node_rank,
            output_dir,
            master_addr,
            master_port,
            command,
        } => rustrain_parallel::launcher::launch_multi(
            nproc_per_node,
            nnodes,
            node_rank,
            &output_dir,
            &master_addr,
            master_port,
            &command,
        ),
        Command::Probe => tch_train::probe_tch_cuda(),
        Command::Server {
            http_port,
            grpc_port,
            metrics_dir,
        } => run_server(http_port, grpc_port, metrics_dir),
        Command::EpServer {
            http_port,
            grpc_port,
            metrics_dir,
            world_size,
        } => run_ep_server(http_port, grpc_port, metrics_dir, world_size),
        Command::EpBench {
            metrics_dir,
            world_size,
            n_adapters,
            lora_rank,
            duration,
            seq_len,
        } => run_ep_bench(
            metrics_dir,
            world_size,
            n_adapters,
            lora_rank,
            duration,
            seq_len,
        ),
        Command::EpWorker {
            shm_name,
            rank,
            world_size,
            metrics_path,
        } => {
            use rustrain_server::ep::worker_main;
            worker_main(&shm_name, rank, world_size, metrics_path)
                .map_err(|e| anyhow!("EP worker error: {}", e))?;
            Ok(())
        }
        #[cfg(feature = "ray")]
        Command::RayGpu {
            num_gpus,
            ray_address,
            runner_path,
            command,
        } => crate::ray_gpu::run(num_gpus, ray_address, &runner_path, &command),
    }
}

// ── Train dispatch ──────────────────────────────────────────────

fn dispatch_train(config_path: &Path, resume_from: Option<PathBuf>) -> Result<()> {
    let mut config = load_config(config_path)?;
    if let Some(resume_from) = resume_from {
        config.train.resume_from = Some(resume_from);
    }
    validate_config(&config)?;

    let run_paths = prepare_run_directory(&config.run)?;
    let world_size = std::env::var("WORLD_SIZE")
        .ok()
        .map(|value| value.parse::<usize>().context("WORLD_SIZE must be a usize"))
        .transpose()?
        .unwrap_or(1);
    let rank = std::env::var("RANK")
        .ok()
        .map(|value| value.parse::<usize>().context("RANK must be a usize"))
        .transpose()?
        .unwrap_or(0);
    let rank_log_dir =
        rustrain_core::runtime::prepare_rank_log_directory(&run_paths, rank, world_size)?;
    let _log_guard = init_logging(&rank_log_dir)?;
    if rank == 0 {
        write_resolved_config(&config, &run_paths.resolved_config)?;
    }

    info!(config_path = %config_path.display(), "loaded config");
    info!(run_dir = %run_paths.root.display(), "created run directory");
    info!(seed = config.run.seed, "seed configured");
    info!(
        device = ?config.train.device,
        dtype = ?config.train.dtype,
        "training policy configured"
    );

    let arch = config.model.architecture.as_str();
    let is_tch = matches!(
        config.train.backend,
        rustrain_core::backend::BackendKind::Tch
    );

    if is_tch && arch == "tch_tiny_lm" {
        let summary = tch_train::train_tch_tiny_lm(&config)?;
        info!(
            initial_loss = summary.initial_loss,
            final_loss = summary.final_loss,
            "tch tiny lm complete"
        );
        println!("rustrain tch tiny lm complete");
        println!("run_dir: {}", run_paths.root.display());
        println!("initial_loss: {:.6}", summary.initial_loss);
        println!("final_loss: {:.6}", summary.final_loss);
        return Ok(());
    }

    if is_tch && arch == "qwen_trainable_session" {
        if config.parallel.tensor_model_parallel_size == 2
            && config.parallel.data_parallel_size == 1
        {
            rustrain_qwen::qwen_module::train_qwen_session_tp_from_config(&config, &run_paths)?;
            println!("rustrain qwen trainable session TP complete");
            println!("run_dir: {}", run_paths.root.display());
        } else if config.parallel.data_parallel_size == 1 {
            let summary = rustrain_qwen::qwen_module::train_qwen_session_single_from_config(
                &config, &run_paths,
            )?;
            println!("rustrain qwen trainable session complete");
            println!("run_dir: {}", run_paths.root.display());
            println!("initial_loss: {:.9}", summary.initial_loss);
            println!("final_loss: {:.9}", summary.final_loss);
            println!("trainable_tensors: {}", summary.trainable_tensors.len());
        } else {
            rustrain_qwen::qwen_module::train_qwen_session_dp_from_config(&config, &run_paths)?;
            println!("rustrain qwen trainable session DP complete");
            println!("run_dir: {}", run_paths.root.display());
        }
        return Ok(());
    }

    if is_tch && arch == "qwen_lora_sft" {
        let summary =
            rustrain_qwen::qwen_module::train_qwen_lora_sft_from_config(&config, &run_paths)?;
        println!("rustrain qwen LoRA SFT complete");
        println!("run_dir: {}", run_paths.root.display());
        println!("adapter_checkpoint: {}", summary.adapter_output);
        println!("initial_loss: {:.9}", summary.initial_loss);
        println!("final_loss: {:.9}", summary.final_loss);
        return Ok(());
    }

    if is_tch && arch == "tch_moe_ep_session" {
        let stats = rustrain_moe::moe::deepseek_moe_stats();
        info!(
            deepseek_moe_layers = stats.layers.len(),
            "parallel process group configured"
        );
        for layer in &stats.layers {
            info!(
                layer = layer.layer_index,
                routed_expert_load = ?layer.routed_expert_load,
                "deepseek moe layer stats"
            );
        }
        println!("rustrain MoE EP session complete");
        println!("run_dir: {}", run_paths.root.display());
        return Ok(());
    }

    if is_tch && arch == "qwen3_trainable_session" {
        if config.parallel.tensor_model_parallel_size == 2
            && config.parallel.data_parallel_size == 1
        {
            rustrain_qwen3::qwen3_module::train_qwen3_session_tp_from_config(&config, &run_paths)?;
            println!("rustrain qwen3 trainable session TP complete");
            println!("run_dir: {}", run_paths.root.display());
        } else if config.parallel.data_parallel_size == 1 {
            let summary = rustrain_qwen3::qwen3_module::train_qwen3_session_single_from_config(
                &config, &run_paths,
            )?;
            println!("rustrain qwen3 trainable session complete");
            println!("run_dir: {}", run_paths.root.display());
            println!("initial_loss: {:.9}", summary.initial_loss);
            println!("final_loss: {:.9}", summary.final_loss);
            println!("trainable_tensors: {}", summary.trainable_tensors.len());
        } else {
            rustrain_qwen3::qwen3_module::train_qwen3_session_dp_from_config(&config, &run_paths)?;
            println!("rustrain qwen3 trainable session DP complete");
            println!("run_dir: {}", run_paths.root.display());
        }
        return Ok(());
    }

    if is_tch && arch == "qwen3_lora_sft" {
        let summary =
            rustrain_qwen3::qwen3_module::train_qwen3_lora_sft_from_config(&config, &run_paths)?;
        println!("rustrain qwen3 LoRA SFT complete");
        println!("run_dir: {}", run_paths.root.display());
        println!("adapter_checkpoint: {}", summary.adapter_output);
        println!("initial_loss: {:.9}", summary.initial_loss);
        println!("final_loss: {:.9}", summary.final_loss);
        return Ok(());
    }

    if is_tch && arch == "deepseek_trainable_session" {
        rustrain_deepseek::deepseek_module::train_deepseek_session_single_from_config(
            &config, &run_paths,
        )?;
        println!("rustrain DeepSeek-V3 session complete");
        println!("run_dir: {}", run_paths.root.display());
        return Ok(());
    }

    if is_tch && arch == "deepseek_tp_rank" {
        let model_path = config
            .model
            .model_path
            .as_ref()
            .context("DeepSeek TP requires model.model_path")?;
        let model_path =
            rustrain_deepseek::deepseek_module::resolve_deepseek_model_path(model_path)?;
        let runtime_config = rustrain_deepseek::deepseek_module::read_deepseek_config(
            &model_path.join("config.json"),
        )?;
        let kind = match config.train.dtype {
            rustrain_core::runtime::DType::Fp32 => tch::Kind::Float,
            rustrain_core::runtime::DType::Bf16 => tch::Kind::BFloat16,
            _ => tch::Kind::Float,
        };
        let output_dir = std::env::var("RUSTRAIN_LAUNCH_OUTPUT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| config.run.base_dir.join("deepseek-tp"));
        rustrain_deepseek::tp::deepseek_tp_rank(&model_path, &output_dir, &runtime_config, kind)?;
        return Ok(());
    }

    if is_tch && arch == "deepseek_ep_rank" {
        let model_path = config
            .model
            .model_path
            .as_ref()
            .context("DeepSeek EP requires model.model_path")?;
        let model_path =
            rustrain_deepseek::deepseek_module::resolve_deepseek_model_path(model_path)?;
        let runtime_config = rustrain_deepseek::deepseek_module::read_deepseek_config(
            &model_path.join("config.json"),
        )?;
        let kind = match config.train.dtype {
            rustrain_core::runtime::DType::Fp32 => tch::Kind::Float,
            rustrain_core::runtime::DType::Bf16 => tch::Kind::BFloat16,
            _ => tch::Kind::Float,
        };
        let output_dir = std::env::var("RUSTRAIN_LAUNCH_OUTPUT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| config.run.base_dir.join("deepseek-ep"));
        rustrain_deepseek::ep::deepseek_ep_rank(&model_path, &output_dir, &runtime_config, kind)?;
        return Ok(());
    }

    if is_tch && arch == "deepseek_v4_session" {
        rustrain_deepseek_v4::deepseek_v4_module::train_v4_session_single_from_config(
            &config, &run_paths,
        )?;
        println!("rustrain DeepSeek V4 session complete");
        println!("run_dir: {}", run_paths.root.display());
        return Ok(());
    }

    if is_tch && arch == "deepseek_v4_tp_rank" {
        let model_path = config
            .model
            .model_path
            .as_ref()
            .context("V4 TP requires model.model_path")?;
        let model_path =
            rustrain_deepseek_v4::deepseek_v4_module::resolve_v4_model_path(model_path)?;
        let runtime_config = rustrain_deepseek_v4::deepseek_v4_module::read_v4_config(
            &model_path.join("config.json"),
        )?;
        let kind = match config.train.dtype {
            rustrain_core::runtime::DType::Fp32 => tch::Kind::Float,
            rustrain_core::runtime::DType::Bf16 => tch::Kind::BFloat16,
            _ => tch::Kind::Float,
        };
        let output_dir = std::env::var("RUSTRAIN_LAUNCH_OUTPUT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| config.run.base_dir.join("deepseek-v4-tp"));
        rustrain_deepseek_v4::tp::deepseek_v4_tp_rank(
            &model_path,
            &output_dir,
            &runtime_config,
            kind,
        )?;
        return Ok(());
    }

    if is_tch && arch == "deepseek_v4_tp_train" {
        let model_path = config
            .model
            .model_path
            .as_ref()
            .context("V4 TP train requires model.model_path")?;
        let model_path =
            rustrain_deepseek_v4::deepseek_v4_module::resolve_v4_model_path(model_path)?;
        let runtime_config = rustrain_deepseek_v4::deepseek_v4_module::read_v4_config(
            &model_path.join("config.json"),
        )?;
        let kind = match config.train.dtype {
            rustrain_core::runtime::DType::Fp32 => tch::Kind::Float,
            rustrain_core::runtime::DType::Bf16 => tch::Kind::BFloat16,
            _ => tch::Kind::Float,
        };
        let output_dir = std::env::var("RUSTRAIN_LAUNCH_OUTPUT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| config.run.base_dir.join("deepseek-v4-tp-train"));
        rustrain_deepseek_v4::tp::deepseek_v4_tp_train(
            &model_path,
            &output_dir,
            &runtime_config,
            kind,
        )?;
        return Ok(());
    }

    if is_tch && arch == "deepseek_v4_ep_rank" {
        let model_path = config
            .model
            .model_path
            .as_ref()
            .context("V4 EP requires model.model_path")?;
        let model_path =
            rustrain_deepseek_v4::deepseek_v4_module::resolve_v4_model_path(model_path)?;
        let runtime_config = rustrain_deepseek_v4::deepseek_v4_module::read_v4_config(
            &model_path.join("config.json"),
        )?;
        let kind = match config.train.dtype {
            rustrain_core::runtime::DType::Fp32 => tch::Kind::Float,
            rustrain_core::runtime::DType::Bf16 => tch::Kind::BFloat16,
            _ => tch::Kind::Float,
        };
        let output_dir = std::env::var("RUSTRAIN_LAUNCH_OUTPUT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| config.run.base_dir.join("deepseek-v4-ep"));
        rustrain_deepseek_v4::ep::deepseek_v4_ep_rank(
            &model_path,
            &output_dir,
            &runtime_config,
            kind,
        )?;
        return Ok(());
    }

    if is_tch && arch == "deepseek_v4_ep_train" {
        let model_path = config
            .model
            .model_path
            .as_ref()
            .context("V4 EP train requires model.model_path")?;
        let model_path =
            rustrain_deepseek_v4::deepseek_v4_module::resolve_v4_model_path(model_path)?;
        let runtime_config = rustrain_deepseek_v4::deepseek_v4_module::read_v4_config(
            &model_path.join("config.json"),
        )?;
        let kind = match config.train.dtype {
            rustrain_core::runtime::DType::Fp32 => tch::Kind::Float,
            rustrain_core::runtime::DType::Bf16 => tch::Kind::BFloat16,
            _ => tch::Kind::Float,
        };
        let output_dir = std::env::var("RUSTRAIN_LAUNCH_OUTPUT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| config.run.base_dir.join("deepseek-v4-ep-train"));
        rustrain_deepseek_v4::ep::deepseek_v4_ep_train(
            &model_path,
            &output_dir,
            &runtime_config,
            kind,
        )?;
        return Ok(());
    }

    if is_tch && arch == "deepseek_v4_tp_ep_train" {
        let model_path = config
            .model
            .model_path
            .as_ref()
            .context("V4 TP+EP train requires model.model_path")?;
        let model_path =
            rustrain_deepseek_v4::deepseek_v4_module::resolve_v4_model_path(model_path)?;
        let runtime_config = rustrain_deepseek_v4::deepseek_v4_module::read_v4_config(
            &model_path.join("config.json"),
        )?;
        let kind = match config.train.dtype {
            rustrain_core::runtime::DType::Fp32 => tch::Kind::Float,
            rustrain_core::runtime::DType::Bf16 => tch::Kind::BFloat16,
            _ => tch::Kind::Float,
        };
        let output_dir = std::env::var("RUSTRAIN_LAUNCH_OUTPUT_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| config.run.base_dir.join("deepseek-v4-tp-ep"));
        rustrain_deepseek_v4::tp::deepseek_v4_tp_ep_train(
            &model_path,
            &output_dir,
            &runtime_config,
            kind,
        )?;
        return Ok(());
    }

    if is_tch && arch == "deepseek_v4_lora_sft" {
        let summary = rustrain_deepseek_v4::deepseek_v4_module::train_v4_lora_sft_from_config(
            &config, &run_paths,
        )?;
        println!("rustrain DeepSeek V4 LoRA SFT complete");
        println!("run_dir: {}", run_paths.root.display());
        println!("adapter_checkpoint: {}", summary.adapter_output);
        println!("initial_loss: {:.9}", summary.initial_loss);
        println!("final_loss: {:.9}", summary.final_loss);
        return Ok(());
    }

    if is_tch && arch == "deepseek_v4_lora_sft_ep" {
        let summary = rustrain_deepseek_v4::session_ep::train_v4_lora_sft_ep(&config, &run_paths)?;
        println!("rustrain DeepSeek V4 LoRA SFT EP complete");
        println!("run_dir: {}", run_paths.root.display());
        println!("adapter_checkpoint: {}", summary.adapter_output);
        println!("initial_loss: {:.9}", summary.initial_loss);
        println!("final_loss: {:.9}", summary.final_loss);
        println!("trainable_params: {}", summary.trainable_params);
        return Ok(());
    }

    if is_tch && arch == "deepseek_lora_sft" {
        let summary = rustrain_deepseek::deepseek_module::train_deepseek_lora_sft_from_config(
            &config, &run_paths,
        )?;
        println!("rustrain DeepSeek LoRA SFT complete");
        println!("run_dir: {}", run_paths.root.display());
        println!("adapter_checkpoint: {}", summary.adapter_output);
        println!("initial_loss: {:.9}", summary.initial_loss);
        println!("final_loss: {:.9}", summary.final_loss);
        return Ok(());
    }

    if is_tch && arch == "glm5_lora_sft_ep" {
        // Check if TP or CP is requested
        let tp_size = config.parallel.tensor_model_parallel_size;
        let cp_size = config.parallel.context_parallel_size;
        if tp_size > 1 || cp_size > 1 {
            let summary =
                rustrain_glm5::session_tp_cp::train_glm5_lora_sft_tp_cp_ep(&config, &run_paths)?;
            println!("rustrain GLM-5.2 LoRA SFT TP+CP+EP complete");
            println!("run_dir: {}", run_paths.root.display());
            println!("adapter_checkpoint: {}", summary.adapter_output);
            println!("initial_loss: {:.9}", summary.initial_loss);
            println!("final_loss: {:.9}", summary.final_loss);
            println!("trainable_params: {}", summary.trainable_params);
            return Ok(());
        }
        let summary = rustrain_glm5::session_ep::train_glm5_lora_sft_ep(&config, &run_paths)?;
        println!("rustrain GLM-5.2 LoRA SFT EP complete");
        println!("run_dir: {}", run_paths.root.display());
        println!("adapter_checkpoint: {}", summary.adapter_output);
        println!("initial_loss: {:.9}", summary.initial_loss);
        println!("final_loss: {:.9}", summary.final_loss);
        println!("trainable_params: {}", summary.trainable_params);
        return Ok(());
    }

    if is_tch && matches!(arch, "qwen3_5_lora_sft" | "qwen3_6_lora_sft") {
        let summary = rustrain_qwen3_6::session::train_qwen3_6_lora_sft(&config, &run_paths)?;
        println!("rustrain Qwen3.5/3.6 LoRA SFT complete");
        println!("run_dir: {}", run_paths.root.display());
        println!("adapter_checkpoint: {}", summary.adapter_output);
        println!("initial_loss: {:.9}", summary.initial_loss);
        println!("final_loss: {:.9}", summary.final_loss);
        println!("trainable_params: {}", summary.trainable_params);
        return Ok(());
    }

    if is_tch && matches!(arch, "qwen3_5_lora_sft_ep" | "qwen3_6_lora_sft_ep") {
        let summary = rustrain_qwen3_6::session::train_qwen3_6_lora_sft_ep(&config, &run_paths)?;
        println!("rustrain Qwen3.5/3.6 LoRA SFT EP complete");
        println!("run_dir: {}", run_paths.root.display());
        println!("adapter_checkpoint: {}", summary.adapter_output);
        println!("initial_loss: {:.9}", summary.initial_loss);
        println!("final_loss: {:.9}", summary.final_loss);
        println!("trainable_params: {}", summary.trainable_params);
        return Ok(());
    }

    // Default: ndarray toy model
    rustrain_toy::trainer::train(&config, &run_paths)
}

fn run_server(http_port: u16, grpc_port: u16, metrics_dir: PathBuf) -> Result<()> {
    use rustrain_server::grpc::train::train_service_server::TrainServiceServer;
    use rustrain_server::{api, grpc, state::SessionManager};

    std::fs::create_dir_all(&metrics_dir)?;
    let manager = std::sync::Arc::new(SessionManager::new(metrics_dir.clone()));

    let http_addr = format!("0.0.0.0:{http_port}");
    let grpc_addr = format!("0.0.0.0:{grpc_port}").parse()?;
    info!("HTTP server will listen on {http_addr}");
    info!("gRPC server will listen on 0.0.0.0:{grpc_port}");

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        // HTTP (axum)
        let app_state = std::sync::Arc::new(api::AppState {
            manager: manager.clone(),
        });
        let http_router = api::router(app_state);
        let http_listener = tokio::net::TcpListener::bind(&http_addr)
            .await
            .unwrap_or_else(|e| panic!("bind {http_addr}: {e}"));

        // gRPC (tonic)
        let grpc_svc = TrainServiceServer::new(grpc::TrainServiceImpl {
            manager: manager.clone(),
        });

        tokio::select! {
            r = axum::serve(http_listener, http_router) => { if let Err(e) = r { tracing::error!("HTTP server error: {e}"); } }
            r = tonic::transport::Server::builder().add_service(grpc_svc).serve(grpc_addr) => { if let Err(e) = r { tracing::error!("gRPC server error: {e}"); } }
        }
    });

    Ok(())
}

/// Start EP server: fork N GPU workers, then run HTTP server in parent.
/// HTTP server dispatches commands to workers via shared memory IPC.
fn run_ep_server(
    http_port: u16,
    grpc_port: u16,
    metrics_dir: PathBuf,
    world_size: usize,
) -> Result<()> {
    use rustrain_server::{api, ep::EpCoordinator};

    // Note: parent process must NOT call any CUDA functions.
    // tch library is linked but CUDA context is only initialized on first
    // CUDA call (lazy init). Parent only does HTTP + IPC, no GPU ops.
    // Workers (spawned via exec) have their own fresh CUDA context.

    std::fs::create_dir_all(&metrics_dir)?;

    info!("Starting EP server: world_size={world_size}, HTTP port={http_port}");

    // Fork worker processes — children never return
    let coordinator = EpCoordinator::launch(world_size, metrics_dir.clone())
        .map_err(|e| anyhow!("Failed to launch EP workers: {}", e))?;

    let coordinator = std::sync::Arc::new(coordinator);

    let http_addr = format!("0.0.0.0:{http_port}");
    info!("EP HTTP server will listen on {http_addr}");

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let app_state = std::sync::Arc::new(api::EpAppState {
            coordinator: coordinator.clone(),
            world_size,
        });
        let http_router = api::ep_router(app_state);
        let http_listener = tokio::net::TcpListener::bind(&http_addr)
            .await
            .unwrap_or_else(|e| panic!("bind {http_addr}: {e}"));

        info!("EP HTTP server listening on {http_addr}");

        if let Err(e) = axum::serve(http_listener, http_router).await {
            tracing::error!("HTTP server error: {e}");
        }
    });

    // Workers are killed when coordinator is dropped
    info!("EP server shutting down");
    Ok(())
}

fn run_ep_bench(
    metrics_dir: PathBuf,
    world_size: usize,
    n_adapters: i32,
    lora_rank: i32,
    duration: u64,
    seq_len: usize,
) -> Result<()> {
    use rustrain_ipc::{EpCommand, EpResult};
    use rustrain_server::ep::EpCoordinator;

    std::fs::create_dir_all(&metrics_dir)?;

    info!(
        "EP bench: world_size={}, n_adapters={}, rank={}, duration={}s, seq={}",
        world_size, n_adapters, lora_rank, duration, seq_len
    );

    let coordinator = EpCoordinator::launch(world_size, metrics_dir.clone())
        .map_err(|e| anyhow!("Failed to launch EP workers: {}", e))?;

    // Session setup
    let sid = "bench";
    let model_path = std::env::var("RUSTRAIN_MODEL_PATH")
        .unwrap_or_else(|_| "/mnt/workspace/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0".to_string());
    let lt = "[\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"]";
    let lt_full = std::iter::repeat(lt).take(8).collect::<Vec<_>>().join(",");
    let config_toml = format!(
        r#"
[run]
name="bench"
seed=42
[model]
name="bench"
architecture="qwen3_6_lora_sft"
model_path="{model_path}"
vocab_size=248320
hidden_size=4096
num_layers=32
num_attention_heads=16
num_key_value_heads=4
head_dim=256
seq_len={seq_len}
norm="rmsnorm"
activation="swiglu"
rope=true
rms_norm_eps=0.000001
partial_rotary_factor=0.25
layer_types={lt_full}
linear_num_key_heads=16
linear_key_head_dim=128
linear_num_value_heads=32
linear_value_head_dim=128
linear_conv_kernel_dim=4
num_experts=0
num_experts_per_tok=0
moe_intermediate_size=0
[train]
max_steps=100000
backend="tch"
micro_batch_size=1
global_batch_size=1
gradient_accumulation_steps=1
learning_rate=0.0001
dtype="bf16"
device="cuda"
checkpoint_every=0
eval_every=0
[parallel]
tensor_model_parallel_size=1
pipeline_model_parallel_size=1
data_parallel_size=1
expert_model_parallel_size=1
context_parallel_size=1
[lora]
rank=8
alpha=16
target_layers=[]
target_modules=["q_proj","k_proj","v_proj","o_proj","in_proj_qkv","in_proj_z","out_proj"]
[data]
kind="instruction_jsonl"
paths=["/tmp/qwen3_6_test.jsonl"]
"#
    );

    eprintln!("[bench] create session...");
    let _ = coordinator.dispatch(&EpCommand::CreateSession {
        session_id: sid.to_string(),
    });

    eprintln!("[bench] load_model...");
    match coordinator.dispatch(&EpCommand::LoadModel {
        session_id: sid.to_string(),
        model_path: model_path.to_string(),
        config_toml: config_toml.clone(),
    }) {
        EpResult::Ok => {}
        EpResult::Error(e) => bail!("load_model failed: {}", e),
        _ => bail!("load_model unexpected result"),
    }

    eprintln!("[bench] load_dataset...");
    let _ = coordinator.dispatch(&EpCommand::LoadDataset {
        session_id: sid.to_string(),
        jsonl_path: "/tmp/qwen3_6_test.jsonl".to_string(),
        seq_len,
    });

    eprintln!("[bench] init_lora...");
    match coordinator.dispatch(&EpCommand::InitLora {
        session_id: sid.to_string(),
        rank: 8,
        alpha: 16.0,
        target_layers: vec![],
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
            "in_proj_qkv".to_string(),
            "in_proj_z".to_string(),
            "out_proj".to_string(),
        ],
        lr: 0.0001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 0.00000001,
    }) {
        EpResult::Count(_) => {}
        EpResult::Error(e) => bail!("init_lora failed: {}", e),
        _ => bail!("init_lora unexpected result"),
    }

    eprintln!(
        "[bench] batch_add_lora ({} adapters, rank={})...",
        n_adapters, lora_rank
    );
    match coordinator.dispatch(&EpCommand::BatchAddLora {
        session_id: sid.to_string(),
        count: n_adapters,
        rank: lora_rank as i64,
        alpha: (lora_rank * 2) as f64,
        target_layers: vec![],
        target_modules: "".to_string(),
    }) {
        EpResult::Count(n) => eprintln!("[bench] added {} adapters", n),
        EpResult::Error(e) => bail!("batch_add_lora failed: {}", e),
        _ => bail!("batch_add_lora unexpected result"),
    }

    // Build one n_total-row tenant batch for every independent source coordinate.
    let topology = rustrain_parallel::topology::ParallelTopology::from_env_with_world_size(
        world_size,
    )?;
    let ep_source_sharded = std::env::var("QWEN36_EP_A2A_SHARDED")
        .map(|value| !value.is_empty() && value != "0")
        .unwrap_or(false);
    let source_parallel_size = if ep_source_sharded {
        topology
            .data_parallel_size()
            .checked_mul(topology.expert_model_parallel_size())
            .ok_or_else(|| anyhow!("source-parallel size overflowed usize"))?
    } else {
        topology.data_parallel_size()
    };
    let global_batch_size = usize::try_from(n_adapters)
        .map_err(|_| anyhow!("n_adapters must be positive"))?
        .checked_mul(source_parallel_size)
        .ok_or_else(|| anyhow!("benchmark global batch size overflowed usize"))?;
    if global_batch_size == 0 {
        bail!("n_adapters and source-parallel size must be positive");
    }
    let tensor_elements = seq_len
        .checked_mul(global_batch_size)
        .ok_or_else(|| anyhow!("benchmark tensor element count overflowed usize"))?;
    let ids = vec![1i64; tensor_elements];
    let mut mask_row = vec![1i64; seq_len];
    mask_row[..20.min(seq_len)].fill(0);
    let mask = mask_row.repeat(global_batch_size);
    let attn = vec![1i64; tensor_elements];

    // Warmup
    eprintln!("[bench] warmup...");
    let t0 = std::time::Instant::now();
    let warmup_loss = match coordinator.dispatch(&EpCommand::TrainMultiLora {
        session_id: sid.to_string(),
        input_ids: ids.clone(),
        target_mask: mask.clone(),
        attention_mask: attn.clone(),
        batch_size: global_batch_size,
        seq_len,
        n_total: n_adapters,
        lora_rank,
        adapter_ids: vec![],
        expected_steps: vec![],
    }) {
        EpResult::Train { loss, .. } => loss,
        EpResult::Error(e) => {
            bail!("warmup failed: {}", e);
        }
        _ => bail!("warmup unexpected result"),
    };
    let warmup_ms = t0.elapsed().as_millis();
    eprintln!(
        "[bench] warmup: loss={:.6} time={}ms",
        warmup_loss, warmup_ms
    );

    if duration == 0 {
        // Single step only
        eprintln!("[bench] single step done");
        let _ = coordinator.dispatch(&EpCommand::DeleteSession {
            session_id: sid.to_string(),
        });
        return Ok(());
    }

    // Throughput measurement
    eprintln!("[bench] measuring throughput for {}s...", duration);
    let mut total_adapters = 0i64;
    let mut total_steps = 0i64;
    let mut losses = Vec::new();
    let start = std::time::Instant::now();

    while start.elapsed().as_secs() < duration {
        let t0 = std::time::Instant::now();
        match coordinator.dispatch(&EpCommand::TrainMultiLora {
            session_id: sid.to_string(),
            input_ids: ids.clone(),
            target_mask: mask.clone(),
            attention_mask: attn.clone(),
            batch_size: global_batch_size,
            seq_len,
            n_total: n_adapters,
            lora_rank,
            adapter_ids: vec![],
            expected_steps: vec![],
        }) {
            EpResult::Train { loss, .. } => {
                losses.push(loss);
                total_adapters += n_adapters as i64;
                total_steps += 1;
                let elapsed = start.elapsed().as_secs_f64();
                if total_steps % 5 == 0 || total_steps == 1 {
                    let rate = total_adapters as f64 / elapsed;
                    eprintln!(
                        "  step {}: loss={:.6} time={}ms total={} adp in {:.0}s = {:.1} adp/s",
                        total_steps,
                        loss,
                        t0.elapsed().as_millis(),
                        total_adapters,
                        elapsed,
                        rate
                    );
                }
            }
            EpResult::Error(e) => {
                eprintln!("[bench] step {} FAILED: {}", total_steps, e);
                break;
            }
            _ => break,
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let rate = total_adapters as f64 / elapsed;

    eprintln!();
    eprintln!("============================================================");
    eprintln!(
        "  RESULTS: {} adapters, rank={}, EP={}",
        n_adapters, lora_rank, world_size
    );
    eprintln!("============================================================");
    eprintln!("  Duration: {:.0}s ({:.1} min)", elapsed, elapsed / 60.0);
    eprintln!("  Steps: {}", total_steps);
    eprintln!("  Total adapters processed: {}", total_adapters);
    eprintln!(
        "  Throughput: {:.2} adapters/s ({:.0} adapters/min)",
        rate,
        rate * 60.0
    );
    if !losses.is_empty() {
        eprintln!(
            "  Loss: {:.6} -> {:.6}",
            losses[0],
            losses[losses.len() - 1]
        );
    }
    eprintln!("  Failures: {}", total_steps - losses.len() as i64);

    let _ = coordinator.dispatch(&EpCommand::DeleteSession {
        session_id: sid.to_string(),
    });
    Ok(())
}
