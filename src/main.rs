mod backend;
mod distributed_smoke;
mod inspect;
mod launcher;
mod lora;
mod metrics;
mod moe;
mod nccl_smoke;
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
    QwenSamplingSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long, default_value = "runs/parity/qwen2_5_0_5b_generate.safetensors")]
        reference_fixture: PathBuf,
        #[arg(long, default_value_t = 4)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        #[arg(long, default_value_t = 20)]
        top_k: usize,
        #[arg(long, default_value_t = 0.9)]
        top_p: f64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    QwenKvCacheParity {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long, default_value = "runs/parity/qwen2_5_0_5b_generate.safetensors")]
        reference_fixture: PathBuf,
        #[arg(long, default_value_t = 4)]
        max_new_tokens: usize,
    },
    QwenLoraSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long, default_value = "runs/parity/qwen_layer0_modules.safetensors")]
        fixture: PathBuf,
        #[arg(long, default_value = "/tmp/rustrain-qwen-lora-adapter.safetensors")]
        adapter_output: PathBuf,
        #[arg(long, default_value_t = 4)]
        rank: i64,
        #[arg(long, default_value_t = 8.0)]
        alpha: f64,
    },
    QwenLoraTrainSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long, default_value = "runs/parity/qwen_layer0_modules.safetensors")]
        fixture: PathBuf,
        #[arg(
            long,
            default_value = "/tmp/rustrain-qwen-lora-trained-adapter.safetensors"
        )]
        adapter_output: PathBuf,
        #[arg(long, default_value_t = 4)]
        rank: i64,
        #[arg(long, default_value_t = 8.0)]
        alpha: f64,
        #[arg(long, default_value_t = 1e3)]
        learning_rate: f64,
    },
    QwenLoraSftSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(
            long,
            default_value = "/tmp/rustrain-qwen-lora-sft-adapter.safetensors"
        )]
        adapter_output: PathBuf,
        #[arg(long)]
        sft_jsonl: Option<PathBuf>,
        #[arg(long, default_value_t = 2)]
        sft_batch_size: usize,
        #[arg(long, default_value = "Reply with the project name.")]
        instruction: String,
        #[arg(long, default_value = "rustrain")]
        response: String,
        #[arg(long, default_value_t = 4)]
        rank: i64,
        #[arg(long, default_value_t = 8.0)]
        alpha: f64,
        #[arg(long, default_value_t = 100.0)]
        learning_rate: f64,
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
    QwenFullTrainSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long, default_value = "runs/parity/qwen2_5_0_5b_logits.safetensors")]
        reference_fixture: PathBuf,
        #[arg(
            long,
            default_value = "/tmp/rustrain-qwen-full-train-delta.safetensors"
        )]
        delta_output: PathBuf,
        #[arg(long, default_value = "fp32")]
        dtype: String,
        #[arg(long, default_value_t = 1e-6)]
        learning_rate: f64,
    },
    MoeSmoke,
    TchMoeSmoke,
    #[command(hide = true)]
    QwenDpGradientRankSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long, default_value = "runs/parity/qwen_layer0_modules.safetensors")]
        reference_fixture: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value = "fp32")]
        dtype: String,
        #[arg(long, default_value_t = 1)]
        steps: usize,
        #[arg(long, default_value_t = 1.0)]
        learning_rate: f64,
    },
    #[command(hide = true)]
    QwenSessionDpRankSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value = "fp32")]
        dtype: String,
        #[arg(long, default_value_t = 1)]
        steps: usize,
        #[arg(long, default_value_t = 1e-6)]
        learning_rate: f64,
    },
    #[command(hide = true)]
    QwenSessionDpDataPlan {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value_t = 2)]
        world_size: usize,
        #[arg(long, default_value_t = 0)]
        data_cursor_start: usize,
    },
    #[command(hide = true)]
    QwenTpLinearRankSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    #[command(hide = true)]
    QwenTpAttentionRankSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    #[command(hide = true)]
    QwenTpAttentionNcclRankSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    #[command(hide = true)]
    QwenTpMlpRankSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    #[command(hide = true)]
    QwenTpMlpNcclRankSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
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
    TchCudaProbe,
    ParallelDpSmoke {
        #[arg(long, default_value = "runs/parallel-dp-smoke")]
        output_dir: PathBuf,
        #[arg(long, default_value_t = 2)]
        world_size: usize,
    },
    ParallelTpSmoke {
        #[arg(long, default_value_t = 2)]
        world_size: usize,
    },
    ParallelEpSmoke {
        #[arg(long, default_value_t = 2)]
        world_size: usize,
    },
    #[command(hide = true)]
    ParallelDpRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        rank: Option<usize>,
        #[arg(long)]
        world_size: Option<usize>,
    },
    #[command(hide = true)]
    ParallelEpRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
    },
    #[command(hide = true)]
    ParallelEpNcclRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
    },
    #[command(hide = true)]
    ParallelEpSparseRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
    },
    #[command(hide = true)]
    PrintLaunchEnv,
    #[command(hide = true)]
    NcclAllReduceRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
    },
    #[command(hide = true)]
    NcclDpGradientRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
    },
    #[command(hide = true)]
    TchDpGradientRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
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
        Command::QwenSamplingSmoke {
            model_path,
            reference_fixture,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            seed,
        } => qwen_module::qwen_sampling_smoke(
            &model_path,
            &reference_fixture,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            seed,
        ),
        Command::QwenKvCacheParity {
            model_path,
            reference_fixture,
            max_new_tokens,
        } => qwen_module::qwen_kv_cache_parity(&model_path, &reference_fixture, max_new_tokens),
        Command::QwenLoraSmoke {
            model_path,
            fixture,
            adapter_output,
            rank,
            alpha,
        } => qwen_module::qwen_lora_smoke(&model_path, &fixture, &adapter_output, rank, alpha),
        Command::QwenLoraTrainSmoke {
            model_path,
            fixture,
            adapter_output,
            rank,
            alpha,
            learning_rate,
        } => qwen_module::qwen_lora_train_smoke(
            &model_path,
            &fixture,
            &adapter_output,
            rank,
            alpha,
            learning_rate,
        ),
        Command::QwenLoraSftSmoke {
            model_path,
            adapter_output,
            sft_jsonl,
            sft_batch_size,
            instruction,
            response,
            rank,
            alpha,
            learning_rate,
        } => qwen_module::qwen_lora_sft_smoke(
            &model_path,
            &adapter_output,
            sft_jsonl.as_deref(),
            sft_batch_size,
            &instruction,
            &response,
            rank,
            alpha,
            learning_rate,
        ),
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
        Command::QwenFullTrainSmoke {
            model_path,
            reference_fixture,
            delta_output,
            dtype,
            learning_rate,
        } => qwen_module::qwen_full_train_smoke(
            &model_path,
            &reference_fixture,
            &delta_output,
            qwen_module::QwenComputeDType::parse(&dtype)?,
            learning_rate,
        ),
        Command::MoeSmoke => {
            println!(
                "{}",
                serde_json::to_string_pretty(&moe::moe_smoke_summary())?
            );
            Ok(())
        }
        Command::TchMoeSmoke => tch_train::run_tch_moe_smoke(),
        Command::QwenDpGradientRankSmoke {
            model_path,
            reference_fixture,
            output_dir,
            dtype,
            steps,
            learning_rate,
        } => qwen_module::qwen_dp_gradient_smoke(
            &model_path,
            &reference_fixture,
            output_dir,
            qwen_module::QwenComputeDType::parse(&dtype)?,
            steps,
            learning_rate,
        ),
        Command::QwenSessionDpRankSmoke {
            model_path,
            output_dir,
            dtype,
            steps,
            learning_rate,
        } => qwen_module::qwen_session_dp_rank_smoke(
            &model_path,
            output_dir,
            qwen_module::QwenComputeDType::parse(&dtype)?,
            steps,
            learning_rate,
            &[0],
            None,
            None,
        ),
        Command::QwenSessionDpDataPlan {
            config,
            world_size,
            data_cursor_start,
        } => qwen_module::qwen_session_dp_data_plan(&config, world_size, data_cursor_start),
        Command::QwenTpLinearRankSmoke {
            model_path,
            output_dir,
        } => qwen_module::qwen_tp_linear_rank_smoke(&model_path, output_dir),
        Command::QwenTpAttentionRankSmoke {
            model_path,
            output_dir,
        } => qwen_module::qwen_tp_attention_rank_smoke(&model_path, output_dir),
        Command::QwenTpAttentionNcclRankSmoke {
            model_path,
            output_dir,
        } => qwen_module::qwen_tp_attention_nccl_rank_smoke(&model_path, output_dir),
        Command::QwenTpMlpRankSmoke {
            model_path,
            output_dir,
        } => qwen_module::qwen_tp_mlp_rank_smoke(&model_path, output_dir),
        Command::QwenTpMlpNcclRankSmoke {
            model_path,
            output_dir,
        } => qwen_module::qwen_tp_mlp_nccl_rank_smoke(&model_path, output_dir),
        Command::Launch {
            nproc_per_node,
            output_dir,
            master_addr,
            master_port,
            command,
        } => launcher::launch(
            nproc_per_node,
            &output_dir,
            &master_addr,
            master_port,
            &command,
        ),
        Command::TchCudaProbe => tch_train::probe_tch_cuda(),
        Command::ParallelDpSmoke {
            output_dir,
            world_size,
        } => distributed_smoke::run_data_parallel_smoke(&output_dir, world_size),
        Command::ParallelTpSmoke { world_size } => {
            distributed_smoke::run_tensor_parallel_smoke(world_size)
        }
        Command::ParallelEpSmoke { world_size } => {
            distributed_smoke::run_expert_parallel_smoke(world_size)
        }
        Command::ParallelDpRankSmoke {
            output_dir,
            rank,
            world_size,
        } => distributed_smoke::run_data_parallel_rank_from_args(output_dir, rank, world_size),
        Command::ParallelEpRankSmoke { output_dir } => {
            distributed_smoke::run_expert_parallel_rank_smoke(output_dir)
        }
        Command::ParallelEpNcclRankSmoke { output_dir } => {
            distributed_smoke::run_expert_parallel_nccl_rank_smoke(output_dir)
        }
        Command::ParallelEpSparseRankSmoke { output_dir } => {
            distributed_smoke::run_expert_parallel_sparse_rank_smoke(output_dir)
        }
        Command::PrintLaunchEnv => launcher::print_launch_env(),
        Command::NcclAllReduceRankSmoke { output_dir } => {
            nccl_smoke::run_nccl_all_reduce_rank(output_dir)
        }
        Command::NcclDpGradientRankSmoke { output_dir } => {
            nccl_smoke::run_nccl_dp_gradient_rank(output_dir)
        }
        Command::TchDpGradientRankSmoke { output_dir } => {
            tch_train::run_tch_dp_gradient_rank_smoke(output_dir)
        }
    }
}
