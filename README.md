# rustrain

A high-performance LLM training engine in Rust. Built on tch-rs (PyTorch C++ bindings)
with native FP8 GEMM, expert parallelism, and multi-GPU distributed training.

> **Status:** Active development. DeepSeek V4 Flash EP=8 LoRA SFT verified on 8× H20-3e.

## Highlights

- **FP8 native GEMM** — C++ FFI to CUTLASS via `at::_scaled_mm`, block-wise scale
  (128×128), no Python dependency in the training loop
- **Expert Parallel (EP=8)** — sharded MoE experts across GPUs, NCCL all-reduce,
  persistent communicator (single init, reused across all layers)
- **DeepSeek V4 Flash** — full architecture: MLA attention, MoE with noaux_tc
  Sinkhorn routing, compress/decompress, HC sparse attention, YaRN RoPE, MTP loss
- **LoRA SFT** — instruction fine-tuning with JSONL data, response-only loss,
  Adam optimizer, gradient sync, adapter save/load
- **Pure Rust + C++** — no Python runtime dependency for training; safetensors
  parsed via mmap, FP8 tensors created via `at::from_blob`

## Quick Start

```sh
# Probe CUDA availability
cargo run -- probe

# Train a toy model (ndarray, CPU)
cargo run -- train --config configs/qwen3_mini.toml

# Train with tch-rs on CUDA
cargo run -- train --config configs/tch_smoke_cuda.toml

# LoRA SFT on Qwen2.5-0.5B
cargo run -- train --config configs/qwen_lora_sft.toml

# Distributed EP=8 training (8 GPUs, direct SSH)
cargo run --release -- launch --nproc-per-node 8 \
  --output-dir /tmp/runs/v4-ep8 \
  train --config configs/deepseek_v4_flash_lora_sft_ep8.toml

# Inspect a HuggingFace model directory
cargo run -- inspect --model-path /path/to/model
```

## CLI

```
rustrain train   --config <config.toml> [--resume-from <path>]
rustrain inspect --model-path <hf_model_dir>
rustrain launch  --nproc-per-node N --output-dir <dir> -- <child-command>
rustrain probe
```

## Supported Models

| Model | Backend | Parallelism | Status |
|---|---|---|---|
| Qwen2.5-0.5B | tch-rs (CUDA) | DP, TP, single | ✅ Verified |
| Qwen2.5-0.5B LoRA SFT | tch-rs (CUDA) | DP, single | ✅ Verified |
| TinyMoE / DeepSeekMoE | tch-rs (CUDA) | EP=2 | ✅ Verified |
| DeepSeek V4 Flash | tch-rs + C++ FP8 | EP=8 | ✅ Verified (8× H20-3e) |
| DeepSeek V4 Flash LoRA SFT | tch-rs + C++ FP8 | EP=8 | ✅ Verified (20 steps) |
| DeepSeek V4 Pro | tch-rs + C++ FP8 | — | ❌ Needs 13+ GPUs |

### V4 Flash Architecture

```
safetensors (FP8) → Rust mmap → parse header (serde_json)
    → C++ at::from_blob (FP8 tensor creation)
    → C++ at::_scaled_mm (CUTLASS FP8 GEMM) → bf16 output
    → LoRA backward → NCCL all-reduce → Adam → adapter save
```

Key V4 features implemented:

- **MLA Attention** — wq_a→q_norm→wq_b, MQA shared KV, o_groups output projection
- **MoE + noaux_tc routing** — Sinkhorn normalization, over-selection, top-k
- **Compress/Decompress** — per-layer sequence compression (model architecture, always on)
- **HC sparse attention** — learned hash bias on compressed sequences
- **YaRN RoPE scaling** — beta_fast/beta_slow interpolation, compress_rope_theta
- **MTP multi-layer loss** — multi-token prediction auxiliary loss
- **ue8m0 scale** — uint8 exponent format for FP8 block scales

### Parallelism

```toml
[parallel]
tensor_model_parallel_size = 1   # TP
data_parallel_size = 1           # DP
expert_model_parallel_size = 8   # EP
pipeline_model_parallel_size = 1 # PP
context_parallel_size = 1        # CP
```

### Compute Precision

```toml
[train]
dtype = "bf16"   # or "fp32"
device = "cuda"
```

## Project Structure

```
rustrain/
├── crates/
│   ├── rustrain-core/           # Config, DType, Device, Backend trait, RunPaths
│   ├── rustrain-data/           # Tokenizer, dataset, SFT field transforms, Arrow IPC
│   ├── rustrain-nccl/           # NCCL FFI bindings + persistent communicator
│   ├── rustrain-parallel/        # ProcessGroup, launcher, TP=1 Megatron modules
│   ├── rustrain-checkpoint/      # Manifest schema, safetensors I/O
│   ├── rustrain-train/           # AdamW, LR scheduler, gradient clipping, metrics
│   ├── rustrain-toy/             # ndarray Qwen-shaped toy model + LoRA
│   ├── rustrain-tch-tiny/        # tch-rs tiny LM training
│   ├── rustrain-qwen/            # Real Qwen: model, session, LoRA, SFT
│   ├── rustrain-moe/             # TinyMoE, DeepSeekMoE, EP rank processes
│   └── rustrain-deepseek-v4/     # V4 Flash: FP8 GEMM, HC attention, EP LoRA SFT
│       ├── kernels/fp8_gemm.cpp  # C++ shim: at::_scaled_mm + at::from_blob
│       ├── src/
│       │   ├── model.rs          # Config, MLA, MoE, compress, MTP, forward
│       │   ├── fp8_kernel.rs     # FFI binding + mmap safetensors parser
│       │   ├── session_ep.rs     # EP=8 LoRA SFT training loop
│       │   ├── hc.rs             # Hash/Content sparse attention
│       │   ├── tp.rs             # TP sharding + training + TP×EP hybrid
│       │   ├── ep.rs             # EP sharding + training
│       │   ├── lora.rs           # LoRA adapter registry
│       │   ├── sft.rs            # SFT dataset (synthetic + JSONL)
│       │   └── generate.rs       # Greedy / sampling generation
│       └── build.rs              # g++ compilation of C++ kernel
├── configs/                      # TOML training configs
├── scripts/                      # SSH-based GPU execution + verification
└── src/
    ├── main.rs                   # CLI dispatch
    └── inspect.rs                # HuggingFace model inspector
```

### Crate Dependencies

```
core ← data, nccl, parallel, checkpoint, train
              ↑
    ┌─────────┼──────────┐
    │         │          │
  toy     tch-tiny    qwen  moe  deepseek-v4
    │         │          │
    └─────────┴──────────┘
              ↑
           cli (root)
```

Model crates are **independent** — no cross-dependencies. `tch` and `nccl` are
optional features, so crates that don't need them compile without libtorch.

## Tech Stack

| Component | Choice |
|---|---|
| Training backend | `tch-rs` (PyTorch C++ bindings, autograd + CUDA) |
| FP8 GEMM | C++ FFI → `at::_scaled_mm` (CUTLASS), no Python |
| Toy backend | `ndarray` (CPU, no autograd) |
| Tokenizer | HuggingFace `tokenizers` |
| Checkpoint | `safetensors` (mmap, native Rust parser) |
| Config | `serde` + `toml` |
| CLI | `clap` |
| Logging | `tracing` |
| Distributed | NCCL FFI (direct `unsafe extern "C"`, persistent communicator) |
| Data | `arrow` IPC, `serde_json` |

## GPU Execution

Training runs directly on GPU servers via SSH — no Ray, no Kubernetes.

```sh
# Run on a remote GPU server
scripts/gpu_run_ssh.sh cargo run -- train --config configs/deepseek_v4_flash_lora_sft_ep8.toml

# 8-GPU distributed verification
scripts/verify_gpu_distributed_ssh.sh

# Ray-based execution (deprecated, in scripts/ray/)
```

## License

MIT
