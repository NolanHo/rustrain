// rustrain CLI: thin dispatch layer
// 4 user commands (train/inspect/launch/probe) + 4 model namespaces (qwen/moe/nccl/tch-tiny)

mod inspect;

use rustrain_core::runtime::{
    init_logging, load_config, prepare_run_directory, validate_config, write_resolved_config,
};
use rustrain_moe::{distributed_smoke, moe};
use rustrain_nccl::nccl_smoke;
use rustrain_parallel::launcher;
use rustrain_qwen::{qwen_module, qwen_parity};
use rustrain_tch_tiny::tch_train;

use std::path::{Path, PathBuf};

use anyhow::Result;
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

    /// Print launcher environment variables (debug)
    #[command(hide = true)]
    PrintLaunchEnv,

    /// Qwen model commands
    #[command(subcommand)]
    Qwen(QwenCommand),

    /// MoE commands
    #[command(subcommand)]
    Moe(MoeCommand),

    /// NCCL rank commands (invoked by launcher)
    #[command(hide = true, subcommand)]
    Nccl(NcclCommand),

    /// tch-tiny rank commands (invoked by launcher)
    #[command(hide = true, subcommand)]
    TchTiny(TchTinyCommand),
}

// ── Qwen namespace ──────────────────────────────────────────────

#[derive(Debug, Subcommand)]
enum QwenCommand {
    /// Qwen weight mapping parity smoke
    ParitySmoke {
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
    /// Module-level parity (RMSNorm / Attention / MLP vs Python reference)
    ModuleParity {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct/model.safetensors"
        )]
        model_safetensors: PathBuf,
        #[arg(long, default_value = "runs/parity/qwen_layer0_modules.safetensors")]
        fixture: PathBuf,
    },
    /// Full-model logits parity vs Python reference
    LogitsParity {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long, default_value = "runs/parity/qwen2_5_0_5b_logits.safetensors")]
        reference_fixture: PathBuf,
    },
    /// Greedy generate parity vs Python reference
    GenerateParity {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long, default_value = "runs/parity/qwen2_5_0_5b_generate.safetensors")]
        reference_fixture: PathBuf,
    },
    /// Sampling smoke (temperature / top-k / top-p)
    SamplingSmoke {
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
    /// KV cache incremental decode parity
    KvCacheParity {
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
    /// LoRA injection smoke (zero-init = base)
    LoraSmoke {
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
    /// LoRA training smoke
    LoraTrainSmoke {
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
        #[arg(long, default_value_t = 1e-4)]
        learning_rate: f64,
    },
    /// LoRA SFT instruction smoke
    LoraSftSmoke {
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
        #[arg(long, default_value_t = 1e-4)]
        learning_rate: f64,
    },
    /// SFT streaming data plan (invoked by trainer)
    #[command(hide = true)]
    SftStreamingDataPlan {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value_t = 1)]
        world_size: usize,
        #[arg(long, default_value_t = 0)]
        data_cursor_start: usize,
    },
    /// SFT streaming batch plan (invoked by trainer)
    #[command(hide = true)]
    SftStreamingBatchPlan {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value_t = 1)]
        world_size: usize,
        #[arg(long, default_value_t = 0)]
        data_cursor_start: usize,
        #[arg(long)]
        index_cache: Option<PathBuf>,
    },
    /// Arrow source summary (invoked by trainer)
    #[command(hide = true)]
    SftArrowSourceSummary {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 128)]
        limit: usize,
        #[arg(long, default_value = "instruction")]
        instruction_column: String,
        #[arg(long, default_value = "input")]
        input_column: String,
        #[arg(long, default_value = "output")]
        response_column: String,
    },
    /// Arrow batch plan (invoked by trainer)
    #[command(hide = true)]
    SftArrowBatchPlan {
        #[arg(long)]
        input: PathBuf,
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long, default_value_t = 2)]
        world_size: usize,
        #[arg(long, default_value_t = 2)]
        local_batch_size: usize,
        #[arg(long, default_value_t = 2)]
        train_steps: usize,
        #[arg(long, default_value_t = 94)]
        data_cursor_start: usize,
        #[arg(long, default_value_t = 128)]
        limit: usize,
        #[arg(long, default_value_t = 0.75)]
        train_split: f32,
        #[arg(long, default_value = "instruction")]
        instruction_column: String,
        #[arg(long, default_value = "input")]
        input_column: String,
        #[arg(long, default_value = "output")]
        response_column: String,
        #[arg(long, default_value = "Instruction: {instruction}\\nResponse: ")]
        prompt_template: String,
        #[arg(
            long,
            default_value = "Instruction: {instruction}\\nInput: {input}\\nResponse: "
        )]
        prompt_with_input_template: String,
    },
    /// Tied embedding training smoke
    TiedHeadTrainSmoke {
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
    /// Full-parameter Qwen training smoke
    FullTrainSmoke {
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
    /// DP gradient rank process (invoked by launcher)
    #[command(hide = true)]
    DpGradientRankSmoke {
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
    /// Session DP rank process (invoked by launcher)
    #[command(hide = true)]
    SessionDpRankSmoke {
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
    /// Session DP data plan (invoked by launcher)
    #[command(hide = true)]
    SessionDpDataPlan {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value_t = 2)]
        world_size: usize,
        #[arg(long, default_value_t = 0)]
        data_cursor_start: usize,
    },
    /// TP linear rank process (invoked by launcher)
    #[command(hide = true)]
    TpLinearRankSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// TP attention rank process (invoked by launcher)
    #[command(hide = true)]
    TpAttentionRankSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// TP attention NCCL rank process (invoked by launcher)
    #[command(hide = true)]
    TpAttentionNcclRankSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// TP MLP rank process (invoked by launcher)
    #[command(hide = true)]
    TpMlpRankSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// TP MLP NCCL rank process (invoked by launcher)
    #[command(hide = true)]
    TpMlpNcclRankSmoke {
        #[arg(
            long,
            default_value = "/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct"
        )]
        model_path: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
}

// ── MoE namespace ───────────────────────────────────────────────

#[derive(Debug, Subcommand)]
enum MoeCommand {
    /// ndarray MoE forward smoke (TinyMoE + DeepSeekMoE stats)
    Smoke,
    /// tch-rs CUDA MoE training smoke
    TchSmoke,
    /// DP smoke (multi-process data parallel loss/gradient verification)
    ParallelDpSmoke {
        #[arg(long, default_value = "runs/parallel-dp-smoke")]
        output_dir: PathBuf,
        #[arg(long, default_value_t = 2)]
        world_size: usize,
    },
    /// TP smoke (tensor parallel column/row shard verification)
    ParallelTpSmoke {
        #[arg(long, default_value_t = 2)]
        world_size: usize,
    },
    /// EP smoke (expert parallel all-to-all verification)
    ParallelEpSmoke {
        #[arg(long, default_value_t = 2)]
        world_size: usize,
    },
    /// DP rank process (invoked by launcher)
    #[command(hide = true)]
    ParallelDpRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        rank: Option<usize>,
        #[arg(long)]
        world_size: Option<usize>,
    },
    /// EP rank process (invoked by launcher)
    #[command(hide = true)]
    ParallelEpRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// EP NCCL rank process (invoked by launcher)
    #[command(hide = true)]
    ParallelEpNcclRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// EP sparse rank process (invoked by launcher)
    #[command(hide = true)]
    ParallelEpSparseRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// EP tch MoE rank process (invoked by launcher)
    #[command(hide = true)]
    ParallelEpTchMoeRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
    },
}

// ── NCCL namespace (hidden, invoked by launcher) ───────────────

#[derive(Debug, Subcommand)]
enum NcclCommand {
    /// NCCL all-reduce rank smoke
    AllReduceRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// NCCL DP gradient rank smoke
    DpGradientRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
    },
}

// ── tch-tiny namespace (hidden, invoked by launcher) ────────────

#[derive(Debug, Subcommand)]
enum TchTinyCommand {
    /// tch-rs DP gradient rank smoke
    DpGradientRankSmoke {
        #[arg(long)]
        output_dir: PathBuf,
    },
}

// ── Dispatch ────────────────────────────────────────────────────

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
        } => launcher::launch(
            nproc_per_node,
            &output_dir,
            &master_addr,
            master_port,
            &command,
        ),
        Command::Probe => tch_train::probe_tch_cuda(),
        Command::PrintLaunchEnv => launcher::print_launch_env(),

        Command::Qwen(cmd) => dispatch_qwen(cmd),
        Command::Moe(cmd) => dispatch_moe(cmd),
        Command::Nccl(cmd) => dispatch_nccl(cmd),
        Command::TchTiny(cmd) => dispatch_tch_tiny(cmd),
    }
}

fn dispatch_qwen(cmd: QwenCommand) -> Result<()> {
    use qwen_module::QwenComputeDType;
    match cmd {
        QwenCommand::ParitySmoke {
            model_path,
            prompt_file,
            reference_summary,
        } => qwen_parity::qwen_parity_smoke(&model_path, &prompt_file, &reference_summary),
        QwenCommand::ModuleParity {
            model_safetensors,
            fixture,
        } => qwen_module::qwen_module_parity(&model_safetensors, &fixture),
        QwenCommand::LogitsParity {
            model_path,
            reference_fixture,
        } => qwen_module::qwen_logits_parity(&model_path, &reference_fixture),
        QwenCommand::GenerateParity {
            model_path,
            reference_fixture,
        } => qwen_module::qwen_generate_parity(&model_path, &reference_fixture),
        QwenCommand::SamplingSmoke {
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
        QwenCommand::KvCacheParity {
            model_path,
            reference_fixture,
            max_new_tokens,
        } => qwen_module::qwen_kv_cache_parity(&model_path, &reference_fixture, max_new_tokens),
        QwenCommand::LoraSmoke {
            model_path,
            fixture,
            adapter_output,
            rank,
            alpha,
        } => qwen_module::qwen_lora_smoke(&model_path, &fixture, &adapter_output, rank, alpha),
        QwenCommand::LoraTrainSmoke {
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
        QwenCommand::LoraSftSmoke {
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
        QwenCommand::SftStreamingDataPlan {
            config,
            world_size,
            data_cursor_start,
        } => qwen_module::qwen_sft_streaming_data_plan(&config, world_size, data_cursor_start),
        QwenCommand::SftStreamingBatchPlan {
            config,
            world_size,
            data_cursor_start,
            index_cache,
        } => qwen_module::qwen_sft_streaming_batch_plan(
            &config,
            world_size,
            data_cursor_start,
            index_cache.as_deref(),
        ),
        QwenCommand::SftArrowSourceSummary {
            input,
            limit,
            instruction_column,
            input_column,
            response_column,
        } => qwen_module::qwen_sft_arrow_source_summary(
            &input,
            limit,
            &instruction_column,
            &input_column,
            &response_column,
        ),
        QwenCommand::SftArrowBatchPlan {
            input,
            model_path,
            world_size,
            local_batch_size,
            train_steps,
            data_cursor_start,
            limit,
            train_split,
            instruction_column,
            input_column,
            response_column,
            prompt_template,
            prompt_with_input_template,
        } => qwen_module::qwen_sft_arrow_batch_plan(
            &input,
            &model_path,
            world_size,
            local_batch_size,
            train_steps,
            data_cursor_start,
            limit,
            train_split,
            &instruction_column,
            &input_column,
            &response_column,
            &prompt_template,
            &prompt_with_input_template,
        ),
        QwenCommand::TiedHeadTrainSmoke {
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
        QwenCommand::FullTrainSmoke {
            model_path,
            reference_fixture,
            delta_output,
            dtype,
            learning_rate,
        } => qwen_module::qwen_full_train_smoke(
            &model_path,
            &reference_fixture,
            &delta_output,
            QwenComputeDType::parse(&dtype)?,
            learning_rate,
        ),
        QwenCommand::DpGradientRankSmoke {
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
            QwenComputeDType::parse(&dtype)?,
            steps,
            learning_rate,
        ),
        QwenCommand::SessionDpRankSmoke {
            model_path,
            output_dir,
            dtype,
            steps,
            learning_rate,
        } => qwen_module::qwen_session_dp_rank_smoke(
            &model_path,
            output_dir,
            QwenComputeDType::parse(&dtype)?,
            steps,
            learning_rate,
            &[0],
            None,
            None,
        ),
        QwenCommand::SessionDpDataPlan {
            config,
            world_size,
            data_cursor_start,
        } => qwen_module::qwen_session_dp_data_plan(&config, world_size, data_cursor_start),
        QwenCommand::TpLinearRankSmoke {
            model_path,
            output_dir,
        } => qwen_module::qwen_tp_linear_rank_smoke(&model_path, output_dir),
        QwenCommand::TpAttentionRankSmoke {
            model_path,
            output_dir,
        } => qwen_module::qwen_tp_attention_rank_smoke(&model_path, output_dir),
        QwenCommand::TpAttentionNcclRankSmoke {
            model_path,
            output_dir,
        } => qwen_module::qwen_tp_attention_nccl_rank_smoke(&model_path, output_dir),
        QwenCommand::TpMlpRankSmoke {
            model_path,
            output_dir,
        } => qwen_module::qwen_tp_mlp_rank_smoke(&model_path, output_dir),
        QwenCommand::TpMlpNcclRankSmoke {
            model_path,
            output_dir,
        } => qwen_module::qwen_tp_mlp_nccl_rank_smoke(&model_path, output_dir),
    }
}

fn dispatch_moe(cmd: MoeCommand) -> Result<()> {
    match cmd {
        MoeCommand::Smoke => {
            println!(
                "{}",
                serde_json::to_string_pretty(&moe::moe_smoke_summary())?
            );
            Ok(())
        }
        MoeCommand::TchSmoke => rustrain_moe::tch_moe::run_tch_moe_smoke(),
        MoeCommand::ParallelDpSmoke {
            output_dir,
            world_size,
        } => rustrain_parallel::dp_tp_smoke::run_data_parallel_smoke(&output_dir, world_size),
        MoeCommand::ParallelTpSmoke { world_size } => {
            rustrain_parallel::dp_tp_smoke::run_tensor_parallel_smoke(world_size)
        }
        MoeCommand::ParallelEpSmoke { world_size } => {
            distributed_smoke::run_expert_parallel_smoke(world_size)
        }
        MoeCommand::ParallelDpRankSmoke {
            output_dir,
            rank,
            world_size,
        } => rustrain_parallel::dp_tp_smoke::run_data_parallel_rank_from_args(
            output_dir, rank, world_size,
        ),
        MoeCommand::ParallelEpRankSmoke { output_dir } => {
            distributed_smoke::run_expert_parallel_rank_smoke(output_dir)
        }
        MoeCommand::ParallelEpNcclRankSmoke { output_dir } => {
            distributed_smoke::run_expert_parallel_nccl_rank_smoke(output_dir)
        }
        MoeCommand::ParallelEpSparseRankSmoke { output_dir } => {
            distributed_smoke::run_expert_parallel_sparse_rank_smoke(output_dir)
        }
        MoeCommand::ParallelEpTchMoeRankSmoke { output_dir } => {
            distributed_smoke::run_expert_parallel_tch_moe_rank_smoke(output_dir, None)
        }
    }
}

fn dispatch_nccl(cmd: NcclCommand) -> Result<()> {
    match cmd {
        NcclCommand::AllReduceRankSmoke { output_dir } => {
            nccl_smoke::run_nccl_all_reduce_rank(output_dir)
        }
        NcclCommand::DpGradientRankSmoke { output_dir } => {
            nccl_smoke::run_nccl_dp_gradient_rank(output_dir)
        }
    }
}

fn dispatch_tch_tiny(cmd: TchTinyCommand) -> Result<()> {
    match cmd {
        TchTinyCommand::DpGradientRankSmoke { output_dir } => {
            tch_train::run_tch_dp_gradient_rank_smoke(output_dir)
        }
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
    info!(device = ?config.train.device, dtype = ?config.train.dtype, "training policy configured");

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
        println!("rustrain tch tiny lm smoke complete");
        println!("run_dir: {}", run_paths.root.display());
        println!("initial_loss: {:.6}", summary.initial_loss);
        println!("final_loss: {:.6}", summary.final_loss);
        return Ok(());
    }

    if is_tch && arch == "qwen_trainable_session" {
        if config.parallel.tensor_model_parallel_size == 2
            && config.parallel.data_parallel_size == 1
        {
            qwen_module::train_qwen_session_tp_from_config(&config, &run_paths)?;
            println!("rustrain qwen trainable session TP complete");
            println!("run_dir: {}", run_paths.root.display());
        } else if config.parallel.data_parallel_size == 1 {
            let summary = qwen_module::train_qwen_session_single_from_config(&config, &run_paths)?;
            println!("rustrain qwen trainable session complete");
            println!("run_dir: {}", run_paths.root.display());
            println!("initial_loss: {:.9}", summary.initial_loss);
            println!("final_loss: {:.9}", summary.final_loss);
            println!("trainable_tensors: {}", summary.trainable_tensors.len());
        } else {
            qwen_module::train_qwen_session_dp_from_config(&config, &run_paths)?;
            println!("rustrain qwen trainable session DP complete");
            println!("run_dir: {}", run_paths.root.display());
        }
        return Ok(());
    }

    if is_tch && arch == "qwen_lora_sft" {
        let summary = qwen_module::train_qwen_lora_sft_from_config(&config, &run_paths)?;
        println!("rustrain qwen LoRA SFT complete");
        println!("run_dir: {}", run_paths.root.display());
        println!("adapter_checkpoint: {}", summary.adapter_output);
        println!("initial_loss: {:.9}", summary.initial_loss);
        println!("final_loss: {:.9}", summary.final_loss);
        return Ok(());
    }

    if is_tch && arch == "tch_moe_ep_session" {
        let stats = moe::deepseek_moe_smoke();
        info!(
            deepseek_moe_layers = stats.layers.len(),
            "parallel process group configured"
        );
        for layer in &stats.layers {
            info!(layer = layer.layer_index, routed_expert_load = ?layer.routed_expert_load, "deepseek moe layer stats");
        }
        // MoE EP session runs the expert-parallel smoke via the launcher
        println!("rustrain MoE EP session complete");
        println!("run_dir: {}", run_paths.root.display());
        return Ok(());
    }

    // Default: ndarray toy model
    rustrain_toy::trainer::train(&config, &run_paths)
}
