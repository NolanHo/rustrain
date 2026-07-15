# AGENTS.md — rustrain 架构规则

## 核心原则：计算在 C++，控制流在 Rust

训练热路径的计算密集操作必须用 C++ 实现（`v4_*` 函数，通过 FFI 调用）。
Rust 只负责控制流：配置解析、训练循环编排、权重加载、checkpoint 管理。

### 禁止在训练热路径中使用 tch-rs

以下操作不得出现在 `for step in ...` 循环内部或 `for layer in ...` 循环内部：

```
❌ Tensor::matmul()          → 用 v4_* C++ kernel
❌ Tensor::linear()           → 用 v4_* C++ kernel
❌ Tensor::topk()             → 用 v4_* C++ kernel
❌ Tensor::sigmoid()          → 用 v4_* C++ kernel
❌ Tensor::scatter_()         → 用 v4_* C++ kernel
❌ Tensor::reshape()          → 在 C++ 内部用 at::reshape
❌ Tensor::narrow()           → 在 C++ 内部用 at::narrow
❌ Tensor::cat()              → 在 C++ 内部用 at::cat
❌ Tensor::zeros/ones()       → 在 C++ 内部用 at::zeros/ones
❌ Tensor::embedding()        → 写 C++ v4_embedding wrapper
❌ rms_norm() (Rust)          → 用 v4_glm5_rms_norm_cpp()
❌ glm5_dsa_attention() (Rust) → 用 v4_glm5_dsa_attention C++ kernel
❌ glm5_mlp_fp8() (Rust)      → 用 v4_glm5_mlp_fp8 C++ kernel
```

### 允许在 Rust 中保留的操作

```
✅ loss.backward()           — PyTorch autograd engine 入口，C++/Rust 统一
✅ Tensor::shallow_clone()   — 零拷贝引用计数，无 FFI 开销
✅ Tensor::to_device()       — CPU→GPU 数据传输（权重加载，循环外）
✅ Tensor::to_kind()         — dtype 转换（权重加载，循环外）
✅ nccl_comm.all_reduce()   — NCCL 通信（Rust FFI 管理 comm_stream）
✅ var.grad() / var.zero_grad() — autograd 状态查询
✅ loss.double_value(&[])    — 标量读取（日志，非计算）
```

### 新增计算操作的流程

当需要添加一个新的计算操作时：

1. **在 C++ 实现** (`kernels/glm5_attention.cpp` 或新 `.cpp` 文件)
   - 使用 `at::` 命名空间的 C++ API（`at::matmul`, `at::topk`, `at::sigmoid` 等）
   - 函数签名：`extern "C" void* v4_new_op(void* input_ptr, ...)`
   - 输入输出：接收 `at::Tensor*`（通过 `void*` 传递），返回 `new at::Tensor*`
   - 错误处理：try/catch，失败返回 `nullptr`

2. **在 Rust FFI 绑定** (`fp8_kernel.rs`)
   - `unsafe extern "C" { fn v4_new_op(...) }` 声明
   - `pub fn new_op_cpp(...) -> Result<Tensor>` 包装函数
   - 用 `Tensor::clone_from_ptr` 从 `at::Tensor*` 构造 `tch::Tensor`
   - 用 `v4_glm5_free_at_tensor` 释放 C++ 返回的 `at::Tensor*`

3. **在训练循环中调用**
   - `if use_cpp_attention { v4_new_op_cpp(...) } else { rust_fallback(...) }`
   - 保留 Rust fallback 直到 C++ 路径验证通过

### 粗粒度 kernel 设计原则

- **一层一次 FFI**：`v4_glm5_layer_forward()` 一次调用完成整层
  （attention + RMSNorm + MoE routing + expert dispatch + residual + all-reduce）
- **中间张量留在 C++ 栈上**：不要把中间结果返回 Rust 再传回 C++
- **FP8 dispatch 在 C++ 内部**：C++ kernel 内部判断权重是否 FP8，选择 `_scaled_mm` 或 `at::linear`
- **NCCL 通信从 C++ 调用**：`v4_glm5_layer_forward` 内部直接调 `ncclAllReduce`，不返回 Rust

### 已实现的 C++ kernels

| C++ 函数 | 功能 | 替代的 tch-rs 调用 |
|---|---|---|
| `v4_glm5_layer_forward` | **整层** (RMSNorm+attn+residual+RMSNorm+MoE+dense+residual) | ~25 次 tch-rs → 1 次 FFI |
| `v4_glm5_dsa_attention` | DSA attention (Q/K/V, RoPE, indexer, SDPA, o_proj) | ~30 次 tch-rs → 1 次 FFI |
| `v4_glm5_moe_layer` | MoE routing + expert dispatch + shared + combine | ~20 次 tch-rs → 1 次 FFI |
| `v4_glm5_mlp_fp8` | SwiGLU MLP | ~5 次 tch-rs → 1 次 FFI |
| `v4_glm5_rms_norm` | RMSNorm | ~5 次 tch-rs → 1 次 FFI |
| `v4_glm5_cross_entropy_loss` | 分块交叉熵 loss | ~15 次 tch-rs → 1 次 FFI |
| `v4_glm5_embedding` | embedding lookup | 1 次 tch-rs → 1 次 FFI |
| `v4_adam_step` | Adam optimizer | ~10 次 tch-rs → 1 次 FFI |
| `v4_checkpoint` | Gradient checkpointing | autograd Function |

### 文件职责划分

| 文件 | 职责 | 允许 tch-rs? |
|------|------|-------------|
| `kernels/glm5_attention.cpp` | C++ 计算内核 | N/A (C++) |
| `kernels/fp8_gemm.cpp` | FP8 GEMM + checkpoint | N/A (C++) |
| `fp8_kernel.rs` | Rust FFI 绑定 + 包装函数 | 仅 `Tensor::clone_from_ptr`, `as_ptr` |
| `model.rs` | 模型配置 + 权重容器 | 仅数据结构，无计算 |
| `session_ep.rs` | 训练循环控制流 | 仅 §"允许保留" 列表 |
| `lora.rs` | LoRA adapter 管理 | 权重初始化可 tch-rs, 前向用 C++ |
| `sft.rs` | 数据加载 + tokenization | 允许 (非热路径) |
| `nccl.rs` | NCCL 通信 | 允许 (Tensor 传参) |

### C++ 编译

- `build.rs` 编译每个 `.cpp` 为独立 `.so`（`libfp8_gemm.so`, `libglm5_attention.so`）
- 链接 `libtorch`, `libtorch_cuda`, `libc10`, `libc10_cuda`
- 运行时通过 `LD_LIBRARY_PATH` 加载

### 最终目标

```
Rust:  配置 + 权重加载 + 训练循环控制 + NCCL 通信管理
C++:   所有计算（attention, MLP, MoE, loss, optimizer, embedding）
```

`Cargo.toml` 中 `rustrain-glm5` 最终去掉 `tch` 依赖。
Rust 侧 `Tensor` 类型最终替换为自定义薄包装（`*mut c_void` + shape/kind）。

## GOTCHAS

- **QKV split layout**: Qwen3.5/3.6 `in_proj_qkv` outputs **flat** layout `[Q_all | K_all | V_all]`, NOT per-head interleaved. Use `torch.split(qkv, [q_size, k_size, v_size])` / `qkv.narrow(-1, offset, size)`. See `docs/agent/linear-attention.md`.
- **emptyCache() = hidden sync**: `CUDACachingAllocator::emptyCache()` calls `cudaDeviceSynchronize()`. Remove from training loop. Only call for seq>4096.
- **ABI mismatch**: System PyTorch 2.5.1 (ABI=0) vs rustrain binary (ABI=1). Use `/mnt/workspace/rustrain-env/` PyTorch 2.12.1 (ABI=1).
- **GLIBC on remote**: Binary compiled with GLIBC 2.39 won't run on Ubuntu 20.04 (GLIBC 2.35). Need matching build environment.

## Knowledge Files

- `docs/agent/linear-attention.md` — Linear attention architecture, QKV split layout, delta rule, diagnostic dumps
- `docs/plans/qwen3.6-support.md` — Qwen3.6 model architecture, weight map, config details
