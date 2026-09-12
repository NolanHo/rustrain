# rustrain

A high-performance LLM training engine in Rust. Built on tch-rs (PyTorch C++ bindings)
with native FP8 GEMM, expert parallelism, C++ FFI kernels, CUDA delta rule kernel,
and multi-GPU distributed training.

> **Status:** Active development. Supports DeepSeek V4 Flash, GLM-5.2 FP8, Qwen3.5/3.6
> full series (0.8B–35B), Qwen3, and Qwen2.5. Verified on 8× H20-3e (143GB).
>
> **Verification:** [Qwen3.5/3.6 training results](docs/qwen3-verification.md) — all 6 models
> verified, HF precision alignment within 2%.

## Highlights

- **C++ FFI Kernels** — coarse-grained `v4_*` kernels: one FFI call per layer
  (attention + MLP + MoE + residual), eliminating tch-rs from the hot path
- **CUDA Delta Rule Kernel** — custom CUDA kernel for Gated Delta Rule linear
  attention, state in shared memory, single kernel launch per layer (40× faster
  than per-token bmm loop)
- **Long Context Training** — up to 1M+ tokens on single EP8 GPU:
  - Manual sequential checkpoint (no autograd::Function graph retention)
  - Chunked cross-entropy with detached hidden (per-chunk backward)
  - `emptyCache()` after each group to release CUDA allocator cache
  - Activation offload to CPU pinned memory
  - Sub-layer checkpointing (attn + mlp segments)
- **SDPA Flash Attention** — `at::scaled_dot_product_attention` for O(seq) memory
  full attention (replaces manual matmul → masked_fill → softmax)
- **FP8 Native GEMM** — C++ FFI to CUTLASS via `at::_scaled_mm`, block-wise scale
  (128×128), no Python dependency in the training loop
- **FP8 Dequant** — byte-level C++ `dequant_fp8` with block-wise `weight_scale_inv`
  expansion, bypassing tch-rs `to_kind()` view bug
- **Expert Parallel (EP=8)** — sharded MoE experts across GPUs, NCCL all-reduce,
  persistent communicator (single init, reused across all layers)
- **Async NCCL Pipeline** — `all_reduce_async` + `stream_wait_event` for layer
  overlap, hiding communication latency behind computation
- **Qwen3.5/3.6 Full Series** — 6 models (0.8B/2B/4B/9B/27B/35B-A3B):
  - Dense + MoE support via `config.is_moe` automatic switching
  - Hybrid attention: GDN linear (30 layers) + Full GQA (10 layers)
  - 256-expert MoE with shared expert + shared_expert_gate
  - Vision encoder (ViT 27 layers + patch merger)
  - MTP (multi-token prediction, Megatron convention)
  - `tie_word_embeddings` support
- **DeepSeek V4 Flash** — full architecture: MLA attention, MoE with noaux_tc
  Sinkhorn routing, compress/decompress, HC sparse attention, YaRN RoPE, MTP loss
- **GLM-5.2** — DSA sparse attention, IndexShare, FP8 full 78-layer training,
  C++ `v4_glm5_layer_forward` (1 FFI/layer), TP+CP+EP support
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

# Qwen3.5-0.8B LoRA SFT (single GPU, dense model)
cargo run --release -- train --config configs/qwen3_5_0_8b_test.toml

# Qwen3.6-35B-A3B LoRA SFT (EP8, 8 GPUs)
cargo run --release -- launch --nproc-per-node 8 \
  --output-dir /tmp/runs/qwen36-ep8 \
  train --config configs/qwen3_6_lora_sft_ep8.toml

# Qwen3.6-35B-A3B LoRA SFT with 1M context (EP8)
QWEN36_CHECKPOINT_GROUP=3 \
QWEN36_OFFLOAD_ACTIVATIONS=1 \
QWEN36_SEQ_CHUNK=65536 \
QWEN36_DISABLE_MTP=1 \
cargo run --release -- launch --nproc-per-node 8 \
  --output-dir /tmp/runs/qwen36-1m \
  train --config configs/qwen3_6_lora_sft_ep8.toml

# Inspect a HuggingFace model directory
cargo run -- inspect --model-path /path/to/model
```

## Supported Models

| Model | Architecture | Backend | Parallelism | Status |
|---|---|---|---|---|
| Qwen2.5-0.5B | `qwen_trainable_session` | tch-rs | DP, TP, single | ✅ Verified |
| Qwen2.5-0.5B LoRA SFT | `qwen_lora_sft` | tch-rs | DP, single | ✅ Verified |
| Qwen3-0.6B / 8B / 30B-A3B | `qwen3_trainable_session` | tch-rs | DP, TP, single | ✅ Verified |
| Qwen3-0.6B LoRA SFT | `qwen3_lora_sft` | tch-rs | single | ✅ Verified |
| Qwen3.5-0.8B LoRA SFT | `qwen3_6_lora_sft` | tch-rs + C++ | single | ✅ Verified (loss 2.37→1.35) |
| Qwen3.5-2B LoRA SFT | `qwen3_6_lora_sft` | tch-rs + C++ | single | ✅ Verified (loss 1.17→0.0001) |
| Qwen3.5-4B LoRA SFT | `qwen3_6_lora_sft` | tch-rs + C++ | single | ✅ Verified (loss 1.28→0.0009) |
| Qwen3.5-9B LoRA SFT | `qwen3_6_lora_sft` | tch-rs + C++ | single | ✅ Verified (loss 1.44→0.0001) |
| Qwen3.6-27B LoRA SFT | `qwen3_6_lora_sft` | tch-rs + C++ | single | ✅ Verified (loss 1.61→0.0004) |
| Qwen3.6-35B-A3B LoRA SFT | `qwen3_6_lora_sft` | tch-rs + C++ | single | ✅ Verified (loss 1.65→0.001, HF=1.68) |
| Qwen3.6-35B-A3B LoRA SFT | `qwen3_6_lora_sft_ep` | tch-rs + C++ | EP4 | ✅ Verified |
| TinyMoE / DeepSeekMoE | `tch_moe_ep_session` | tch-rs | EP=2 | ✅ Verified |
| DeepSeek V4 Flash | `deepseek_v4_*` | tch-rs + C++ FP8 | EP=8, TP, TP+EP | ✅ Verified (8× H20-3e) |
| DeepSeek V4 Flash LoRA SFT | `deepseek_v4_lora_sft_ep` | tch-rs + C++ FP8 | EP=8 | ✅ Verified (20 steps) |
| GLM-5.2 / GLM-5.2-FP8 | `glm5_lora_sft_ep` | tch-rs + C++ FP8 | EP=8, TP+CP+EP | ✅ Verified (78 layers) |

### Qwen3.5/3.6 Architecture

- **Hybrid attention**: 40 layers — 3 Gated Delta Rule (GDN) + 1 Full attention alternating
  - Full: GQA + MRoPE (interleaved, partial_rotary=0.25) + output gate + SDPA (Flash Attention)
  - Linear: Gated Delta Rule (recurrent, log-space decay, L2-normalized Q/K, CUDA kernel)
- **MoE**: 256 experts, fused gate_up_proj, shared expert + shared_expert_gate
- **Dense MLP**: gate_proj + up_proj + down_proj (SwiGLU), for non-MoE models
- **Vision encoder**: ViT 27 layers + patch merger
- **MTP**: 1 layer with full attention + MoE, predicts token t+2 (Megatron convention)
- **C++ kernel** (`qwen3_6_kernels.cpp`): full layer forward (RMSNorm + attention +
  MoE + LoRA delta) in one FFI call, gradient checkpointing via manual sequential backward
- **CUDA kernel** (`delta_rule.cu`): Gated Delta Rule with shared memory state,
  single kernel launch per layer (nvcc compiled)

### Long Context Training

| seq_len | Group | Time/step | GPU Peak | Status |
|---------|-------|-----------|----------|--------|
| 32K | 4 | ~2 min | ~14 GB | ✅ Verified |
| 64K | 4 | ~3 min | ~20 GB | ✅ Verified (with MTP) |
| 128K | 4 | ~9 min | ~29 GB | ✅ Verified (with MTP) |
| 256K | 4 | ~6 min | ~42 GB | ✅ Verified (no MTP) |
| 512K | 2 | ~1 hr | ~70 GB | ✅ No OOM |
| 1M | 3 | ~15 min | ~25 GB | ✅ Verified (EP8, no MTP) |

Environment variables for long context:

```sh
QWEN36_CHECKPOINT_GROUP=3       # Group checkpoint size (1-4)
QWEN36_OFFLOAD_ACTIVATIONS=1    # Offload group inputs to CPU pinned memory
QWEN36_SEQ_CHUNK=65536          # Linear attention chunk size
QWEN36_SUBCKPT=1                # Sub-layer checkpointing (attn + mlp segments)
QWEN36_DISABLE_MTP=1            # Skip MTP loss (saves memory)
QWEN36_DISABLE_MTP=             # Enable MTP (default, for seq ≤ 128K)
```

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
│   ├── rustrain-qwen3-6/        # Qwen3.5/3.6: hybrid attn, MoE, vision, MTP
│   │   ├── kernels/
│   │   │   ├── qwen3_6_kernels.cpp  # C++ full-layer forward + checkpointing + CE
│   │   │   ├── delta_rule.cu         # CUDA kernel for Gated Delta Rule
│   │   │   └── delta_rule.cuh        # Kernel template + host launcher
│   │   └── src/
│   │       ├── model.rs         # Hybrid attention, MoE, forward (Rust eager, removed)
│   │       ├── kernel.rs        # FFI binding + dlopen + CppTrainingContext
│   │       ├── config.rs        # text_config + vision_config parsing
│   │       ├── session.rs       # LoRA SFT training (single + EP8, C++ path only)
│   │       ├── lora.rs          # 10 target modules
│   │       ├── mtp.rs           # Multi-token prediction (Megatron convention)
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
| CUDA kernels | `delta_rule.cu` — Gated Delta Rule, shared memory state |
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
