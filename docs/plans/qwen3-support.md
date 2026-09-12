# Plan: Qwen3 Support

## Summary

新增 `rustrain-qwen3` crate，支持 Qwen3-0.6B 的 forward、LoRA SFT、full-parameter
training 和 DP/TP 分布式训练。Qwen3 相比 Qwen2.5 的核心差异是 QK-norm（每个
attention layer 多了 `q_norm.weight` 和 `k_norm.weight`），以及 `head_dim` 从
config.json 显式读取而非推导。

## Qwen3 vs Qwen2.5 差异

| 维度 | Qwen2.5-0.5B | Qwen3-0.6B |
|------|-------------|-----------|
| QK-norm | 无 | `q_norm.weight` + `k_norm.weight` per layer |
| head_dim | `hidden_size / num_heads` | config.json 显式 `head_dim: 128` |
| attention bias | q/k/v/o 有 bias | 无 bias（0 个 bias tensor） |
| 每 layer 张量数 | 13（含 4 个 bias） | 11（多 q_norm/k_norm，无 bias） |
| 权重名 | `model.layers.N.self_attn.q_proj.weight` | 相同 + `q_norm/k_norm` |
| tie_word_embeddings | true | true |
| MLP (SwiGLU) | gate/up/down | 相同 |
| RMSNorm | input_layernorm + post_attention_layernorm | 相同 |
| 总张量数 | ~197 (24 layers × 13 + 5) | 311 (28 layers × 11 + 3) |
| config.json model_type | "qwen2" | "qwen3" |
| sliding_window | 无 | config 有字段但 Qwen3-0.6B 为 null |

QK-norm 的 forward 变化（在 RoPE 之前）：

```
Qwen2.5:  Q = proj(input) → reshape → RoPE → attention
Qwen3:    Q = proj(input) → reshape → RMSNorm(q_norm) → RoPE → attention
          K = proj(input) → reshape → RMSNorm(k_norm) → RoPE → attention
```

## Changes

### 1. `crates/rustrain-qwen3/` — 新 crate

从 `rustrain-qwen` 复制结构，做以下修改：

#### `crates/rustrain-qwen3/Cargo.toml`
- 与 rustrain-qwen 相同的依赖

#### `crates/rustrain-qwen3/src/lib.rs`
- 模块声明：model / generate / lora / session / sft / rank / qwen3_module

#### `crates/rustrain-qwen3/src/model.rs`
- `Qwen3ModelConfig`: 添加 `head_dim: i64`, `sliding_window`, `max_window_layers` 字段
- `Qwen3LayerWeights`: 添加 `q_norm: Tensor`, `k_norm: Tensor` 字段；移除 bias 字段
- `Qwen3LayerWeights::load()`: 加载 `q_norm.weight` 和 `k_norm.weight`
- `qwen3_attention()`: 在 q_proj/k_proj 后、RoPE 前应用 RMSNorm
- `head_dim`: 从 config 读取而非 `hidden_size / num_heads`
- `read_runtime_config()`: 解析 `head_dim` 字段

#### `crates/rustrain-qwen3/src/session.rs`
- `qwen3_trainable_tensors_for_layers()`: 每层包含 q_norm/k_norm
- 训练入口函数名：`train_qwen3_session_*_from_config`

#### `crates/rustrain-qwen3/src/lora.rs`
- LoRA target modules 可选 q_norm/k_norm（但默认只做 q/k/v/o + gate/up/down）
- 训练入口：`train_qwen3_lora_sft_from_config`

#### `crates/rustrain-qwen3/src/sft.rs`
- 从 rustrain-qwen 复制 SFT 数据流（直接复用 rustrain-data）

#### `crates/rustrain-qwen3/src/rank.rs`
- DP/TP rank 进程实现（从 rustrain-qwen 复制，适配 q_norm/k_norm）

#### `crates/rustrain-qwen3/src/generate.rs`
- 生成/采样/KV cache（attention 需要处理 q_norm/k_norm）

### 2. `crates/rustrain-qwen3/src/qwen3_module.rs` — re-export hub

```rust
pub use crate::generate::*;
pub use crate::lora::*;
pub use crate::model::*;
pub use crate::rank::*;
pub use crate::session::*;
pub use crate::sft::*;
```

### 3. `src/main.rs` — CLI dispatch

添加 dispatch 分支：

```rust
if is_tch && arch == "qwen3_trainable_session" { ... }
if is_tch && arch == "qwen3_lora_sft" { ... }
```

### 4. `Cargo.toml` (root)

添加 `rustrain-qwen3 = { path = "crates/rustrain-qwen3" }` 依赖。

### 5. `configs/` — 新配置文件

- `configs/qwen3_session_single.toml` — 单 GPU full-param training
- `configs/qwen3_lora_sft.toml` — LoRA SFT
- `configs/qwen3_session_dp2.toml` — DP=2
- `configs/qwen3_session_tp2.toml` — TP=2

### 6. `scripts/verify_gpu.sh` — 添加验证

- `cargo run -- train --config configs/qwen3_lora_sft.toml`
- `cargo run -- train --config configs/qwen3_session_single.toml`

## Checkpoint

Qwen3-0.6B 位于 GPU host：
```
/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<hash>/
```

包含：`config.json`, `tokenizer.json`, `model.safetensors`

config.json 关键字段：
```json
{
  "model_type": "qwen3",
  "head_dim": 128,
  "hidden_size": 1024,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "num_hidden_layers": 28,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "tie_word_embeddings": true,
  "vocab_size": 151936
}
```

每层 11 个张量（无 bias，多 q_norm/k_norm）：
```
model.layers.N.input_layernorm.weight
model.layers.N.self_attn.q_norm.weight      ← NEW
model.layers.N.self_attn.q_proj.weight
model.layers.N.self_attn.k_norm.weight      ← NEW
model.layers.N.self_attn.k_proj.weight
model.layers.N.self_attn.v_proj.weight
model.layers.N.self_attn.o_proj.weight
model.layers.N.post_attention_layernorm.weight
model.layers.N.mlp.gate_proj.weight
model.layers.N.mlp.up_proj.weight
model.layers.N.mlp.down_proj.weight
```

## Risks

- **head_dim ≠ hidden_size / num_heads**: Qwen3-0.6B 的 head_dim=128，
  但 hidden_size=1024 / num_heads=16 = 64。所有 reshape 必须用 config 的 head_dim。
  Qwen2.5 也有这个问题（某些型号 head_dim 和 hidden_size/heads 不一致），
  但 Qwen3 更明确。mitigation: 所有 head_dim 从 config 读取。

- **QK-norm 影响 TP 分片**: q_norm/k_norm 是 per-head 的 RMSNorm，
  TP 分片时需要按 head 分。mitigation: 先做单 GPU 验证，TP 后做。

- **sliding_window**: Qwen3-0.6B 的 sliding_window 为 null，
  但更大模型可能启用。mitigation: 第一阶段忽略 sliding_window，
  后续按需添加。

## Definition of Done

- [ ] `cargo check -p rustrain-qwen3` 编译通过
- [ ] `cargo run -- train --config configs/qwen3_session_single.toml` 在 GPU 上完成训练
- [ ] `cargo run -- train --config configs/qwen3_lora_sft.toml` 在 GPU 上完成 LoRA SFT
- [ ] LoRA SFT loss 下降，checkpoint reload parity 通过
- [ ] `cargo run -- launch --nproc-per-node 2 -- train --config configs/qwen3_session_dp2.toml` DP=2 训练通过
- [ ] `scripts/verify_gpu.sh` 包含 Qwen3 验证项且全部通过

## Open Questions

- sliding_window 是否需要第一阶段支持？（建议：否，Qwen3-0.6B 不启用）
- LoRA 是否需要支持 q_norm/k_norm 作为 target？（建议：否，只做 q/k/v/o + gate/up/down）
