# Megakernel Plan: Fused Forward+Backward for Qwen3.6

## 当前问题分析

### 代码混乱问题
`qwen3_6_kernels.cpp` (1730行) 混杂了 5 种 forward 路径：

| 路径 | 触发条件 | 说明 |
|------|---------|------|
| `forward_full` | 无 checkpoint | 直接 forward，保留全部 graph |
| `forward_full_checkpoint` | `CHECKPOINT_GROUP` | manual checkpoint + manual_group_backward |
| `forward_full_fused` | `FUSED_LAYER` | FusedLayerFunction per layer |
| `forward_single_layer_subckpt` | `SUBCKPT` | SubLayerCkpt (attn+mlp segments) |
| `GroupCheckpointFunction` | checkpoint 内部 | autograd::Function for group |

### 性能瓶颈 (1M, 15min)
```
Forward (no-grad):  ~3min   — 40 层 × PyTorch 算子
CE backward:        ~5min   — 64 chunks × retain_graph backward
Manual backward:    ~7min   — 14 groups × (recompute 3 层 + grad)
```

Backward 占 47%，因为 checkpoint 需要 recompute forward。

### FusedLayer 测试结果
1M fused = 14min44s ≈ checkpoint 版本 14min47s。**没有加速**。
原因：PyTorch autograd 遍历 graph 的开销 ≈ recompute 开销。

## Megakernel 方案

### 核心思路
**不依赖 PyTorch autograd**。手写 forward + backward，中间结果只在 registers/shared memory 中传递。

```
当前:
  forward → PyTorch 算子 (matmul, SDPA, MoE) → 中间张量写 HBM
  backward → PyTorch autograd 遍历 graph → 读 HBM 中间张量 → 计算梯度

Megakernel:
  forward+backward → 单个 C++ 函数 → 中间结果在 registers → 直接算梯度
  不构建 autograd graph，不写 HBM 中间张量
```

### 架构设计

```cpp
// 新文件: kernels/megakernel.cu
// 一层 = 一个 CUDA kernel launch (forward + backward fused)

struct LayerBackwardResult {
    at::Tensor grad_input;  // [B, S, hidden]
    // LoRA grads 直接累积到 ctx->lora_a/b
};

// 每层一个函数: forward + backward fused
LayerBackwardResult layer_forward_backward(
    TrainingContext* ctx,
    const at::Tensor& input,        // [B, S, hidden]
    const at::Tensor& grad_output,  // [B, S, hidden] — 从后一层传来的梯度
    int64_t layer_idx
);
```

### 实现分解

每层 = 3 个 segment，每个 segment 有 forward + backward：

#### Segment 1: Attention
```
Forward:
  attn_input = rms_norm(hidden, input_norm)     // [B, S, hidden]
  attn_out = attention(attn_input, weights)       // [B, S, hidden]
  residual = hidden + attn_out                    // [B, S, hidden]

Backward (given grad_residual):
  grad_attn_out = grad_residual
  grad_attn_input, grad_weights = attention_backward(attn_input, attn_out, grad_attn_out)
  grad_hidden_from_attn = rms_norm_backward(hidden, grad_attn_input)
  grad_hidden = grad_residual + grad_hidden_from_attn
```

#### Segment 2: MLP/MoE
```
Forward:
  moe_input = rms_norm(residual, post_norm)
  mlp_out = moe_forward(moe_input, weights)
  output = residual + mlp_out

Backward (given grad_output):
  grad_mlp_out = grad_output
  grad_moe_input, grad_weights = moe_backward(moe_input, mlp_out, grad_mlp_out)
  grad_residual_from_mlp = rms_norm_backward(residual, grad_moe_input)
  grad_residual = grad_output + grad_residual_from_mlp
```

### 关键 backward 实现

| 操作 | Forward | Backward | 难度 |
|------|---------|----------|------|
| rms_norm | `x * rsqrt(var+eps) * (1+w)` | 手写: `grad_x = (1+w) * rsqrt * (grad - mean(grad * normed) * normed)` | 中 |
| matmul | `y = x @ W^T` | `grad_x = grad_y @ W`, `grad_W = grad_y^T @ x` | 低 (cuBLAS) |
| SiLU | `silu(x)` | `silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))` | 低 |
| conv1d | depthwise conv + silu | 手写或用 PyTorch autograd | 中 |
| SDPA | Flash Attention | PyTorch 有 `sdpa_backward` | 低 |
| delta rule | CUDA kernel | 手写 backward kernel | 高 |
| MoE routing | softmax + topk + expert dispatch | 手写 per-expert backward | 中 |

### 实现优先级

1. **Phase 1: Linear attention layer** (30/40 层，最简单)
   - rms_norm backward: 手写 (5 行公式)
   - matmul backward: cuBLAS (`grad_x = grad_y @ W`)
   - conv1d backward: 用 PyTorch autograd (保存 input)
   - delta rule backward: 手写 CUDA kernel
   - gated norm backward: 手写

2. **Phase 2: Full attention layer** (10/40 层)
   - SDPA backward: `at::scaled_dot_product_attention` 有 backward
   - RoPE backward: 手写 (rotate_half 的 backward = rotate_half)

3. **Phase 3: MoE backward** (两种层共用)
   - routing backward: softmax + topk 的 backward
   - expert backward: per-expert matmul backward

4. **Phase 4: 整合 + 验证**
   - 替换 `train_step` 中的 forward + backward
   - 对比梯度是否匹配 PyTorch autograd
   - 测试 1M 性能

### 预期性能

| 方案 | Forward | Backward | Total (1M) | 说明 |
|------|---------|----------|------------|------|
| 当前 (checkpoint) | 3min | 7min (recompute) | 15min | 14 groups × recompute |
| FusedLayer | — | — | 15min | autograd graph 遍历 |
| **Megakernel** | **3min** | **3min** (no recompute) | **~8min** | 手写 backward |

### 文件结构

```
kernels/
  qwen3_6_kernels.cpp     — 主文件 (保留，简化)
  delta_rule.cu           — CUDA kernel (保留，加 backward)
  delta_rule.cuh          — kernel header (保留)
  megakernel.cu           — 新: fused forward+backward CUDA kernels
  megakernel.cuh          — 新: header
  backward.h              — 新: backward 实现公式
```

### 风险
- **精度**: 手写 backward 必须匹配 PyTorch autograd (BF16 精度)
- **MoE backward**: 最复杂，需要 per-expert 梯度计算
- **delta rule backward**: 需要逆向遍历 state 更新序列

## Definition of Done
- [ ] Phase 1: Linear attention layer backward (手写)
- [ ] 梯度验证: 对比 PyTorch autograd 的 grad_input
- [ ] Phase 2: Full attention layer backward
- [ ] Phase 3: MoE backward
- [ ] 1M completes in <10min (was 15min)
- [ ] 2M completes in <30min (was 56min)
