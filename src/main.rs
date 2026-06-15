mod backend;
mod inspect;
mod lora;
mod moe;
mod parallel;
mod parallel_modules;
mod qwen_parity;
mod runtime;
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
    }
}
