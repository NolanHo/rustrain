# rustrain

A high-performance LLM training engine in Rust. Built on tch-rs (PyTorch C++ bindings)
with native FP8 GEMM, expert parallelism, C++ FFI kernels, and multi-GPU distributed training.

> **Status:** Active development. Supports DeepSeek V4 Flash, GLM-5.2 FP8, Qwen3.6-35B-A3B, Qwen3, and Qwen2.5. Verified on 8× H20-3e (143GB).

## Highlights

- **C++ FFI Kernels** — coarse-grained `v4_*` kernels: one FFI call per layer
  (attention + MLP + MoE + residual), eliminating tch-rs from the hot path
- **FP8 Native GEMM** — C++ FFI to CUTLASS via `at::_scaled_mm`, block-wise scale
  (128×128), no Python dependency in the training loop
- **FP8 Dequant** — byte-level C++ `dequant_fp8` with block-wise `weight_scale_inv`
  expansion, bypassing tch-rs `to_kind()` view bug
- **Expert Parallel (EP=8)** — sharded MoE experts across GPUs, NCCL all-reduce,
  persistent communicator (single init, reused across all layers)
- **Async NCCL Pipeline** — `all_reduce_async` + `stream_wait_event` for layer
  overlap, hiding communication latency behind computation
- **DeepSeek V4 Flash** — full architecture: MLA attention, MoE with noaux_tc
  Sinkhorn routing, compress/decompress, HC sparse attention, YaRN RoPE, MTP loss
- **GLM-5.2** — DSA sparse attention, IndexShare, FP8 full 78-layer training,
  C++ `v4_glm5_layer_forward` (1 FFI/layer), TP+CP+EP support
- **Qwen3.6-35B-A3B** — hybrid attention (Full GQA + Gated Delta Rule), 256-expert
  MoE, vision encoder, MTP, C++ kernel (1 FFI/layer), EP4
- **Qwen3 / Qwen2.5** — full-param training + LoRA SFT, DP/TP support
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

# Distributed EP=8 training (8 GPUs)
cargo run --release -- launch --nproc-per-node 8 \
  --output-dir /tmp/runs/v4-ep8 \
  train --config configs/deepseek_v4_flash_lora_sft_ep8.toml

# GLM-5.2 FP8 full 78-layer LoRA SFT (8 GPUs)
cargo run --release -- launch --nproc-per-node 8 \
  --output-dir /tmp/runs/glm5-fp8 \
  train --config configs/glm5_lora_sft_ep8.toml

# Qwen3.6-35B-A3B LoRA SFT (single GPU)
cargo run --release -- train --config configs/qwen3_6_lora_sft.toml

# Qwen3.6-35B-A3B LoRA SFT (EP4, 4 GPUs)
cargo run --release -- launch --nproc-per-node 4 \
  --output-dir /tmp/runs/qwen36-ep4 \
  train --config configs/qwen3_6_lora_sft_ep4.toml

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

| Model | Architecture | Backend | Parallelism | Status |
|---|---|---|---|---|
| Qwen2.5-0.5B | `qwen_trainable_session` | tch-rs | DP, TP, single | ✅ Verified |
| Qwen2.5-0.5B LoRA SFT | `qwen_lora_sft` | tch-rs | DP, single | ✅ Verified |
| Qwen3-0.6B / 8B / 30B-A3B | `qwen3_trainable_session` | tch-rs | DP, TP, single | ✅ Verified |
| Qwen3-0.6B LoRA SFT | `qwen3_lora_sft` | tch-rs | single | ✅ Verified |
| Qwen3.6-35B-A3B LoRA SFT | `qwen3_6_lora_sft` | tch-rs + C++ | single | ✅ Verified (loss 21→15, 20 steps) |
| Qwen3.6-35B-A3B LoRA SFT | `qwen3_6_lora_sft_ep` | tch-rs + C++ | EP4 | ✅ Verified (5 steps) |
| TinyMoE / DeepSeekMoE | `tch_moe_ep_session` | tch-rs | EP=2 | ✅ Verified |
| DeepSeek V4 Flash | `deepseek_v4_*` | tch-rs + C++ FP8 | EP=8, TP, TP+EP | ✅ Verified (8× H20-3e) |
| DeepSeek V4 Flash LoRA SFT | `deepseek_v4_lora_sft_ep` | tch-rs + C++ FP8 | EP=8 | ✅ Verified (20 steps) |
| GLM-5.2 / GLM-5.2-FP8 | `glm5_lora_sft_ep` | tch-rs + C++ FP8 | EP=8, TP+CP+EP | ✅ Verified (78 layers) |

### Qwen3.6-35B-A3B Architecture

- **Hybrid attention**: 40 layers — 3 Gated Delta Rule (GDN) + 1 Full attention alternating
  - Full: GQA + MRoPE (interleaved, partial_rotary=0.25) + output gate
  - Linear: Gated Delta Rule (matrix formulation, log-space decay)
- **MoE**: 256 experts, fused gate_up_proj, shared expert + shared_expert_gate
- **Vision encoder**: ViT 27 layers + patch merger
- **MTP**: 1 layer with full attention + MoE, cross-entropy loss (0.5× weight)
- **C++ kernel** (`qwen3_6_kernels.cpp`): full layer forward (RMSNorm + attention +
  MoE + LoRA delta) in one FFI call, gradient checkpointing via `autograd::Function`

### GLM-5.2 Architecture

```
safetensors (FP8) → Rust mmap → C++ v4_glm5_layer_forward (1 FFI/layer)
    → DSA attention (Q/K/V, RoPE, indexer, SDPA, o_proj)
    → MoE routing + expert dispatch + shared + combine
    → residual + RMSNorm
    → LoRA backward → async NCCL all-reduce → Adam → adapter save
```

Key GLM-5.2 features:

- **DSA Sparse Attention** — `v4_glm5_dsa_attention` (Q/K/V, RoPE, indexer, SDPA, o_proj)
- **IndexShare** — reuses indexer across every 4 sparse attention layers
- **FP8 dequant** — `dequant_fp8` with block-wise `weight_scale_inv` expansion
- **Async NCCL** — `all_reduce_async` + `stream_wait_event` for layer overlap
- **TP + CP + EP** — tensor, context, and expert parallelism

### DeepSeek V4 Flash Architecture

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
│   ├── rustrain-nccl/           # NCCL FFI + persistent comm + async all-reduce
│   ├── rustrain-parallel/       # ProcessGroup, launcher, TP=1 Megatron modules
│   ├── rustrain-checkpoint/     # Manifest schema, safetensors I/O
│   ├── rustrain-train/          # AdamW, LR scheduler, gradient clipping, metrics
│   ├── rustrain-toy/            # ndarray Qwen-shaped toy model + LoRA
│   ├── rustrain-tch-tiny/       # tch-rs tiny LM training
│   ├── rustrain-qwen/           # Qwen2.5: model, session, LoRA, SFT
│   ├── rustrain-qwen3/          # Qwen3: 0.6B/8B/30B-A3B, MoE, session, LoRA
│   ├── rustrain-qwen3-6/        # Qwen3.6-35B-A3B: hybrid attn, MoE, vision, MTP
│   │   ├── kernels/qwen3_6_kernels.cpp  # C++ full-layer forward + checkpointing
│   │   └── src/
│   │       ├── model.rs         # Hybrid attention, MoE, forward
│   │       ├── kernel.rs        # FFI binding + dlopen
│   │       ├── config.rs        # text_config + vision_config parsing
│   │       ├── session.rs       # LoRA SFT training (single + EP4)
│   │       ├── lora.rs          # 10 target modules
│   │       ├── mtp.rs           # Multi-token prediction
│   │       ├── vision.rs        # ViT encoder + patch merger
│   │       └── sft.rs           # SFT dataset
│   ├── rustrain-moe/            # TinyMoE, DeepSeekMoE, EP rank processes
│   ├── rustrain-deepseek-v4/    # V4 Flash + GLM-5.2 C++ kernels
│   │   ├── kernels/
│   │   │   ├── fp8_gemm.cpp     # C++ at::_scaled_mm + at::from_blob + dequant
│   │   │   └── glm5_attention.cpp # C++ DSA attn, MoE, layer forward, CE loss, Adam
│   │   └── src/
│   │       ├── fp8_kernel.rs    # FFI binding + mmap safetensors + dequant_fp8_weight
│   │       ├── model.rs         # V4 Config, MLA, MoE, compress, MTP, forward
│   │       ├── session_ep.rs    # V4 EP=8 LoRA SFT training loop
│   │       ├── hc.rs            # Hash/Content sparse attention
│   │       ├── tp.rs / ep.rs    # TP / EP sharding + training
│   │       ├── lora.rs          # LoRA adapter registry
│   │       ├── sft.rs           # SFT dataset (synthetic + JSONL)
│   │       └── generate.rs      # Greedy / sampling generation
│   ├── rustrain-glm5/           # GLM-5.2: DSA, IndexShare, FP8 EP/TP/CP LoRA SFT
│   │   └── src/
│   │       ├── model.rs         # Config, DSA attention, IndexShare, MoE
│   │       ├── session_ep.rs    # EP=8 LoRA SFT (C++ + Rust paths, async NCCL)
│   │       ├── session_tp_cp.rs # TP+CP+EP training loop
│   │       ├── tp_cp.rs         # TP+CP attention implementation
│   │       ├── lora.rs          # LoRA with FP8 dequant
│   │       └── sft.rs           # SFT dataset (GLM chat format)
│   └── rustrain-deepseek/       # DeepSeek V3.2 DSA indexer forward
├── configs/                     # TOML training configs
└── src/
    ├── main.rs                  # CLI dispatch
    └── inspect.rs               # HuggingFace model inspector
```

### Crate Dependencies

```
core ← data, nccl, parallel, checkpoint, train
              ↑
    ┌─────────┼──────────┬────────────┐
    │         │          │            │
  toy    tch-tiny    qwen/qwen3   qwen3-6   moe   deepseek-v4   glm5
    │         │          │            │
    └─────────┴──────────┴────────────┘
              ↑
           cli (root)
```

Model crates are **independent** — no cross-dependencies. `tch` and `nccl` are
optional features, so crates that don't need them compile without libtorch.

## Tech Stack

| Component | Choice |
|---|---|
| Training backend | `tch-rs` (PyTorch C++ bindings, autograd + CUDA) |
| C++ kernels | `v4_*` FFI functions, 1 call/layer (attention + MLP + MoE) |
| FP8 GEMM | C++ FFI → `at::_scaled_mm` (CUTLASS), no Python |
| FP8 dequant | C++ byte-level `dequant_fp8` with block-wise scale expansion |
| Toy backend | `ndarray` (CPU, no autograd) |
| Tokenizer | HuggingFace `tokenizers` |
| Checkpoint | `safetensors` (mmap, native Rust parser) |
| Config | `serde` + `toml` |
| CLI | `clap` |
| Logging | `tracing` |
| Distributed | NCCL FFI (direct `unsafe extern "C"`, persistent + async) |
| Data | `arrow` IPC, `serde_json` |
| Python env | `uv` (pip/venv management, preferred) |

## License

MIT
