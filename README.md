# rustrain

An educational Rust LLM training engine. Built from scratch to understand how
training systems work internally — from toy model forward passes to real Qwen
checkpoint loading, LoRA SFT, Megatron-style parallel modules, MoE, and
distributed DP/TP/EP training.

> **Status:** Educational project, not a production trainer. All training paths
> are config-driven and verified on GPU.

## Quick Start

```sh
# Probe CUDA availability
cargo run -- probe

# Train a toy model (ndarray, CPU)
cargo run -- train --config configs/qwen3_mini.toml

# Train with tch-rs on CUDA
cargo run -- train --config configs/tch_smoke_cuda.toml

# LoRA SFT on real Qwen2.5-0.5B
cargo run -- train --config configs/qwen_lora_sft.toml

# Inspect a HuggingFace model directory
cargo run -- inspect --model-path /path/to/Qwen2.5-0.5B-Instruct

# Launch distributed training (2 GPUs)
cargo run -- launch --nproc-per-node 2 --output-dir /tmp/runs/dp2 \
  train --config configs/qwen_session_dp2.toml
```

## CLI

```
rustrain train   --config <config.toml> [--resume-from <path>]
rustrain inspect --model-path <hf_model_dir>
rustrain launch  --nproc-per-node N -- <child-command>
rustrain probe
```

## Supported Architectures

| Architecture | Backend | Config Example | Description |
|---|---|---|---|
| `qwen_like` | ndarray (CPU) | `configs/qwen3_mini.toml` | Qwen-shaped toy model, random init |
| `tch_tiny_lm` | tch-rs (CUDA) | `configs/tch_smoke_cuda.toml` | Tiny LM with full autograd training |
| `qwen_trainable_session` | tch-rs (CUDA) | `configs/qwen_session_single.toml` | Real Qwen2.5-0.5B full-parameter training |
| `qwen_lora_sft` | tch-rs (CUDA) | `configs/qwen_lora_sft.toml` | LoRA instruction fine-tuning with JSONL data |
| `tch_moe_ep_session` | tch-rs (CUDA) | `configs/tch_moe_ep2.toml` | MoE expert-parallel training |

### Parallel Config

```toml
[parallel]
tensor_model_parallel_size = 1   # TP
data_parallel_size = 1           # DP
expert_model_parallel_size = 1   # EP
pipeline_model_parallel_size = 1 # PP
context_parallel_size = 1       # CP
```

Set `data_parallel_size = 2` and launch with `--nproc-per-node 2` for DP training.
Set `tensor_model_parallel_size = 2` for TP training.

### Compute Precision

```toml
[train]
dtype = "fp32"   # or "bf16"
device = "cuda"
```

## Project Structure

```
rustrain/
├── crates/
│   ├── rustrain-core/        # Config, DType, Device, Backend trait, RunPaths
│   ├── rustrain-data/        # Tokenizer, dataset, SFT field transforms, Arrow IPC
│   ├── rustrain-nccl/        # NCCL FFI bindings
│   ├── rustrain-parallel/    # ProcessGroup, launcher, TP=1 Megatron modules
│   ├── rustrain-checkpoint/  # Manifest schema, safetensors I/O
│   ├── rustrain-train/       # AdamW, LR scheduler, gradient clipping, metrics
│   ├── rustrain-toy/         # ndarray Qwen-shaped toy model + LoRA
│   ├── rustrain-tch-tiny/    # tch-rs tiny LM training
│   ├── rustrain-qwen/        # Real Qwen: model, session, LoRA, SFT, rank processes
│   └── rustrain-moe/         # TinyMoE, DeepSeekMoE, EP rank processes
├── configs/                  # TOML training configs
├── scripts/                  # GPU verification + Ray submission
├── docs/
│   └── checkpoints.md        # Checkpoint format contract
└── src/
    ├── main.rs               # Thin CLI dispatch
    └── inspect.rs            # HuggingFace model inspector
```

### Crate Dependencies

```
core ← data, nccl, parallel, checkpoint, train
              ↑
    ┌─────────┼──────────┐
    │         │          │
  toy     tch-tiny    qwen  moe
    │         │          │
    └─────────┴──────────┘
              ↑
           cli (root)
```

Model crates are **independent** — no cross-dependencies. `tch` and `nccl` are
optional features, so crates that don't need them compile without libtorch.

## GPU Verification

All training, testing, and `cargo check` run on Ray GPU workers. The local
machine is for editing and git only.

```sh
# Run a command on a GPU worker
scripts/gpu_run.sh cargo check
scripts/gpu_run.sh cargo run -- probe
scripts/gpu_run.sh cargo run -- train --config configs/tch_smoke_cuda.toml

# Single-GPU verification suite
scripts/verify_gpu.sh

# 2-GPU distributed verification suite (DP/TP/EP)
scripts/verify_gpu_distributed.sh
```

`scripts/gpu_run.sh` stages the current working tree into a temporary Ray
worker directory, so you always get feedback from your latest local edits.
Set `RUSTRAIN_RAY_NUM_GPUS=2` for two-rank collective checks.

## Features

- **Real Qwen2.5-0.5B training** — load safetensors checkpoints, forward/backward
  with tch-rs autograd, full-parameter trainable sessions
- **LoRA SFT** — instruction fine-tuning with JSONL/Arrow data, response-only loss,
  field mapping, streaming offset-index cache, checkpoint resume
- **Data parallel (DP=2)** — NCCL gradient all-reduce, rank0 checkpoint, sharded
  manifest with data provenance/cursor
- **Tensor parallel (TP=2)** — column/row weight sharding, attention/MLP parity,
  fused layer0 TP with causal-LM loss
- **MoE** — TinyMoE + DeepSeekMoE (shared/routed experts), expert parallel (EP=2)
  with sparse token dispatch
- **Checkpoint system** — delta safetensors + optimizer sidecar + JSON manifest,
  with data cursor, provenance, and resume support
- **SFT data pipeline** — tokenizer-backed JSONL/Arrow loading, field transform DSL
  (replace/regex/case/affix/strip/split/truncate), dedup, source weighting, streaming

## Tech Stack

| Component | Choice |
|---|---|
| Training backend | `tch-rs` (PyTorch C++ bindings, autograd + CUDA) |
| Toy backend | `ndarray` (CPU, no autograd, for teaching) |
| Tokenizer | HuggingFace `tokenizers` |
| Checkpoint | `safetensors` |
| Config | `serde` + `toml` |
| CLI | `clap` |
| Logging | `tracing` |
| Distributed | NCCL FFI (direct `extern "C"`) |
| Data | `arrow` IPC, `serde_json` |

## License

MIT
