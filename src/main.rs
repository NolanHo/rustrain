mod backend;
mod inspect;
mod lora;
mod moe;
mod parallel;
mod parallel_modules;
mod qwen_module;
mod qwen_parity;
mod runtime;
mod tch_train;
mod text_data;
mod toy_model;
mod trainer;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "rustrain")]
#[command(about = "A Rust LLM training engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Train {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(long)]
        resume_from: Option<PathBuf>,
    },
    Inspect {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long, default_value = "rustrain")]
        prompt: String,
        #[arg(long, default_value_t = 12)]
        tensor_limit: usize,
    },
    QwenParitySmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long, default_value = "data/parity/qwen_prompt.txt")]
        prompt_file: PathBuf,
        #[arg(long, default_value = "data/parity/qwen2_5_0_5b_logits_summary.json")]
        reference_summary: PathBuf,
    },
    QwenModuleParity {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct/model.safetensors"
        )]
        model_safetensors: PathBuf,
        #[arg(long, default_value = "runs/parity/qwen_layer0_modules.safetensors")]
        fixture: PathBuf,
    },
    QwenLogitsParity {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long, default_value = "runs/parity/qwen2_5_0_5b_logits.safetensors")]
        reference_fixture: PathBuf,
    },
    QwenGenerateParity {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long, default_value = "runs/parity/qwen2_5_0_5b_generate.safetensors")]
        reference_fixture: PathBuf,
    },
    QwenTiedHeadTrainSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long, default_value = "runs/parity/qwen2_5_0_5b_logits.safetensors")]
        reference_fixture: PathBuf,
        #[arg(
            long,
            default_value = "runs/parity/qwen2_5_0_5b_tied_head_delta.safetensors"
        )]
        delta_output: PathBuf,
        #[arg(long, default_value_t = 1e-4)]
        learning_rate: f64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Train {
            config,
            resume_from,
        } => trainer::train(&config, resume_from),
        Command::Inspect {
            model_path,
            prompt,
            tensor_limit,
        } => inspect::inspect_model(&model_path, &prompt, tensor_limit),
        Command::QwenParitySmoke {
            model_path,
            prompt_file,
            reference_summary,
        } => qwen_parity::qwen_parity_smoke(&model_path, &prompt_file, &reference_summary),
        Command::QwenModuleParity {
            model_safetensors,
            fixture,
        } => qwen_module::qwen_module_parity(&model_safetensors, &fixture),
        Command::QwenLogitsParity {
            model_path,
            reference_fixture,
        } => qwen_module::qwen_logits_parity(&model_path, &reference_fixture),
        Command::QwenGenerateParity {
            model_path,
            reference_fixture,
        } => qwen_module::qwen_generate_parity(&model_path, &reference_fixture),
        Command::QwenTiedHeadTrainSmoke {
            model_path,
            reference_fixture,
            delta_output,
            learning_rate,
        } => qwen_module::qwen_tied_head_train_smoke(
            &model_path,
            &reference_fixture,
            &delta_output,
            learning_rate,
        ),
    }
}
