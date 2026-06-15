mod backend;
mod inspect;
mod lora;
mod parallel;
mod parallel_modules;
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
    }
}
