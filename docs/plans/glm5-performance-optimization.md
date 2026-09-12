# Plan: GLM-5.2 Rust 性能优化

## Summary
GLM-5.2 的 EP8 LoRA SFT 训练存在多个性能瓶颈。按影响排序：1) 每层每步都从 CPU 传输 expert 权重到 GPU（~87GB/层/步），2) LoRA 梯度 all-reduce 串行同步，3) LoRA 权重融合导致 FP8 GEMM 降级为 BF16，4) MoE expert dispatch 对所有 token 计算。本计划优先解决前两个瓶颈——它们不需要改 C++ kernel，只改 Rust 代码。

## 模型概况
- GLM-5.2: 78 层, hidden=6144, 64 heads, head_dim=192
- MoE: 256 experts, top-8, moe_intermediate=2048
- 权重: BF16 1.4TB / FP8 704GB
- 并行: EP8, 每卡 32 experts
- C++ kernel: `glm5_attention.cpp` (1007 行), 已编译为 `libglm5_attention.so`

## 瓶颈分析

| # | 瓶颈 | 位置 | 影响 | 修复难度 |
|---|------|------|------|---------|
| 1 | Expert 权重每层每步 CPU→GPU 传输 | glm5_attention.cpp:670-675 | ~87GB/层 PCIe 传输 | 中（改 C++）|
| 2 | LoRA 梯度 all-reduce 串行同步 | session_ep.rs:984-1000 | N 次 cudaStreamSynchronize | 低（改 Rust）|
| 3 | LoRA 融合导致 FP8 GEMM 降级 | lora.rs:177-209 | 每层 full-weight matmul + dequant | 高（改 C++）|
| 4 | MoE expert 对所有 token 计算 | glm5_attention.cpp:688 | (1-8/256) 的计算浪费 | 中（改 C++）|
| 5 | RoPE 表每次重新计算 | model.rs:597-599 | 78× 冗余 | 低（改 Rust）|
| 6 | empty_cache 只在 step 0 | session_ep.rs:979 | 内存碎片 | 低（改 Rust）|

## 优化步骤（按 ROI 排序）

### Step 1: 缓存 expert 权重到 GPU（最大瓶颈）
**文件**: `crates/rustrain-glm5/src/session_ep.rs`

当前：expert 权重存在 CPU (`expert_weights_cpu`)，每次 forward 调用 C++ kernel 时传 CPU tensor，C++ 内部做 `.to(device)`。

优化：首次 forward 前，将 local experts 预加载到 GPU 并缓存。后续步骤直接传 GPU tensor 给 C++。

```rust
// session_ep.rs, 在 layer loop 之前
let mut expert_weights_gpu: Vec<Tensor> = Vec::new();
for e in 0..experts_per_rank {
    expert_weights_gpu.push(expert_gate_weights_cpu[e].to(device));
    expert_weights_gpu.push(expert_up_weights_cpu[e].to(device));
    expert_weights_gpu.push(expert_down_weights_cpu[e].to(device));
}
// 传 GPU tensor 给 C++ — .to(device) 变成 no-op
```

**注意**: 32 experts × 3 weights × ~2.8GB = ~87GB GPU 内存。加上 attention 权重 ~20GB + activations，总需 ~110GB，140GB 够用。

### Step 2: 批量化 LoRA 梯度 all-reduce
**文件**: `crates/rustrain-glm5/src/session_ep.rs`

当前：逐个 variable 调用同步 `all_reduce()`，每次 `cudaStreamSynchronize`。

优化：用 `all_reduce_async` + CUDA event 批量同步。

```rust
// 替换 lines 984-1000
let mut events = Vec::new();
for var in &trainable_vars {
    if let Some(grad) = var.grad() {
        let event = all_reduce_async(&grad, comm, comm_stream);
        events.push(event);
    }
}
// 一次同步等待所有 all-reduce 完成
stream_synchronize(comm_stream);
```

### Step 3: empty_cache 每步执行
**文件**: `crates/rustrain-glm5/src/session_ep.rs`

当前：只在 step 0 做 `empty_cache()`（line 979）。

优化：每步 backward 后都做 `empty_cache()`。

```rust
// line 979: 移到 if 外面
CudaCachingAllocator::empty_cache();  // 每步都执行
```

### Step 4: 缓存 RoPE cos/sin 表
**文件**: `crates/rustrain-glm5/src/model.rs`

当前：每层每次 forward 都重新计算 `rope_cos_sin()`。

优化：用 `RefCell<BTreeMap<usize, (Tensor, Tensor)>>` 缓存。

### Step 5（可选）: MoE expert 只计算 routed tokens
**文件**: `crates/rustrain-deepseek-v4/kernels/glm5_attention.cpp`

当前：每个 expert 对所有 token 做完整 MLP，然后乘 mask。

优化：先 `index_select` 只取 routed tokens，做 MLP，再 `index_add` 回去。

## 风险
- **GPU 内存不足**: 缓存 expert 权重需要 ~87GB GPU 内存。如果 attention 激活 + LoRA 权重超过剩余 ~53GB，会 OOM。缓解：用 `QWEN36_CHECKPOINT_GROUP` 式的分组加载。
- **all_reduce_async 兼容性**: 需要确认 `rustrain-nccl` 的 async API 是否稳定。缓解：如果 async 不可用，可以用 `nccl_group_start/end` 批量提交。
- **C++ kernel 修改风险**: Step 5 修改 C++ expert dispatch，可能引入数值偏差。缓解：先不改 C++，只做 Rust 优化。

## Definition of Done
- [ ] Step 1: Expert 权重缓存到 GPU，首次 forward 后不再有 CPU→GPU 传输
- [ ] Step 2: LoRA 梯度 all-reduce 批量化，只有一次同步
- [ ] Step 3: 每步 backward 后 empty_cache
- [ ] Step 4: RoPE cos/sin 缓存
- [ ] 编译通过 + 0.8B 或 GLM-5.2 短序列测试 loss 正确
- [ ] GLM-5.2 benchmark 对比优化前后耗时

## Open Questions
- GLM-5.2 FP8 版本是否可用（704GB vs 1.4TB）？FP8 需要 `_scaled_mm`，BF16 不需要。
- 是否需要支持长上下文（>512）？当前 config seq_len=512。
