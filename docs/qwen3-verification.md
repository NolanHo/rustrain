# Qwen3.5/3.6 Training Verification Results

Verified on 8× H20-3e (143GB), single GPU, BF16 precision.

## SFT Data

Single example: `"What is 2+2?" → "4"`, formatted with Qwen3.6 chat template (including `<|im_start|>`, `<|im_end|>`, `think`, `answer` special tokens).

## Precision: HF vs rustrain

Same token sequence, same loss computation (response-only masked cross-entropy):

| Implementation | initial_loss | Notes |
|---|---|---|
| HuggingFace transformers | 1.6795 | Reference |
| rustrain C++ kernel | 1.6467 | 2% BF16 precision diff |

Layer-by-layer hidden state comparison (Qwen3.6-35B-A3B, 512 tokens padded):
- Layers 0-38: mean/std/max match within BF16 precision (< 1%)
- Layer 39: diverges without attention mask; fixed with padding mask

## Training Results (20 steps, LoRA rank=8, alpha=16, lr=1e-4)

| Model | Type | Layers | Hidden | Heads | initial_loss | final_loss | Converges |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | Dense | 24 | 1024 | 8 | 2.370 | 1.354 (5步) | ✅ |
| Qwen3.5-2B | Dense | 24 | 2048 | 8 | 1.170 | 0.000134 | ✅ |
| Qwen3.5-4B | Dense | 32 | 2560 | 16 | 1.279 | 0.000905 | ✅ |
| Qwen3.5-9B | Dense | 32 | 4096 | 16 | 1.442 | 0.000128 | ✅ |
| Qwen3.6-27B | Dense | 64 | 5120 | 24 | 1.615 | 0.000383 | ✅ |
| Qwen3.6-35B-A3B | MoE (256 experts) | 40 | 2048 | 16 | 1.647 | 0.001256 | ✅ |

## Architecture Details

### Dense Models (Qwen3.5-0.8B/2B/4B/9B, Qwen3.6-27B)
- **Hybrid attention**: 3 Linear Attention (Gated Delta Rule) + 1 Full Attention (GQA + MRoPE) alternating
- **MLP**: SwiGLU (gate_proj + up_proj + down_proj)
- **MTP**: 1 layer (full attention + dense MLP)

### MoE Model (Qwen3.6-35B-A3B)
- Same hybrid attention as dense
- **MLP**: 256 experts, top-8 routing, shared expert + shared_expert_gate
- **MTP**: 1 layer (full attention + MoE)

## C++ Kernel Architecture

```
Rust:  config + weight loading + SFT tokenize + training loop + attention mask
C++:   forward (all layers) + CE loss + MTP loss + backward (autograd) + Adam
       ↓ ATen dispatch
       cuDNN (SDPA Flash Attention) + cuBLAS (GEMM) + CUTLASS (FP8)
```

Single `ctx.train_step(input_ids, target_mask, attention_mask)` FFI call per training step.

## Key Features
- Attention mask for padding tokens (causal + key padding)
- Qwen3.6 chat template with think/answer special tokens
- Dense MLP (SwiGLU) and MoE (256 experts) both supported
- Hand-written CUDA fused kernels (RMSNorm, SwiGLU) — compiled but not enabled (ATen ops used)
- Gradient checkpointing via `autograd::Function`
- EP4 distributed training verified (Qwen3.6-35B-A3B)
