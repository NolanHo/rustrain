// rustrain CLI: thin dispatch layer
// 4 user commands: train / inspect / launch / probe

mod inspect;

use rustrain_core::runtime::{
    init_logging, load_config, prepare_run_directory, validate_config, write_resolved_config,
};
use rustrain_tch_tiny::tch_train;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
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
}

fn main() -> Result<()> {
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
            output_dir,
            master_addr,
            master_port,
            command,
        } => rustrain_parallel::launcher::launch(
            nproc_per_node,
            &output_dir,
            &master_addr,
            master_port,
            &command,
        ),
        Command::Probe => tch_train::probe_tch_cuda(),
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
    let _log_guard = init_logging(&run_paths.logs)?;
    write_resolved_config(&config, &run_paths.resolved_config)?;

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

    // Default: ndarray toy model
    rustrain_toy::trainer::train(&config, &run_paths)
}
