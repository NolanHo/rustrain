# AGENTS.md — rustrain 架构规则

## 核心原则：编排是第一公民，算子是插件

rustrain 是一个训练框架，不是 kernel 库。差异化价值在**并行策略、通信调度、显存与精度编排**，
不在手写 GEMM——那些交给 cuBLAS / CUTLASS / FlashAttention / Tilelang。

框架不认识任何模型，也不认识任何具体 kernel。它只做三件事：

1. **解析** —— 把模型表达成算子图（plan）。
2. **解析实现** —— 按 recipe 为图中每个算子选出具体实现。
3. **驱动** —— 用编译好的计划把实现跑起来，并调度其中的通信。

模型是数据。算子是插件。精度是配置。

设计契约见 `docs/design/kernel-first/spec.md`。

## 分层与依赖方向

```
rustrain-cli        train | ops | plan
rustrain-runtime    执行器 / 显存池 / 流与 collective 调度
rustrain-plan       Plan IR / 切分传播 / 校验 / 编译 / digest
rustrain-parallel   进程组 / rank 布局 / 切分规格 / 通信规划
rustrain-ops        算子描述 / 注册表 / requires / numerics / expansion / recipe
rustrain-abi        插件 ABI v1 / 装载
──────────────────────────────────────────────────────
plugins (.so)       aten / reference / 第三方高性能 kernel
```

### 不变式（违反即架构破坏）

- **I-1**：`rustrain-{abi,ops,parallel,plan,runtime}` 的依赖闭包中不得出现 `tch`、`libtorch`、`cuda`。
  这是"换 kernel 不重编框架"的全部依据，也是核心能在无 GPU 机器上跑完整测试的前提。
- **I-2**：算子实现只能通过 ABI 注册，不能被框架静态引用。`rustrain-kernels` 是**一个插件**，不是框架的一部分。
- **I-3**：并行切分是 plan 里的数据（`ParallelLayout`），不是手写在训练循环里的通信调用。
  布局不匹配处的 `all_reduce`/`all_gather`/`reduce_scatter` 由传播 pass 插入。

## 禁止的模式

```
❌ 用环境变量选择算子实现        → 写进 recipe（配置来自文件）
❌ 静默 fallback 到另一个实现    → 显式声明 fallback 列表，解析不到就报错
❌ 在 C++/kernel 里硬编码量化方案 → 量化格式/粒度/block/scale 模式由 rs_numerics 声明
❌ 从张量形状反推量化方案        → 同上；形状碰巧对不代表方案对
❌ 直接在框架里 unsafe 调某个 kernel → 通过 ABI 注册 + 注册表解析
❌ 融合算子不声明原语展开        → expansion 是等价性验收的唯一依据
❌ 算子内自己 malloc/建 stream/建 NCCL comm → 通过 rs_services 向框架申请
```

## 如何新增一个算子实现

1. 新建插件 crate，`extern "C"` 导出**唯一**符号 `rustrain_plugin_v1`（见 `crates/rustrain-abi/include/rustrain_op.h`）。
2. 填写 `rs_op_desc`：`requires`（dtype/SM/world_size/group）、`numerics`、`memory`、
   `backward` 模式、`collectives`；若是融合算子，必须填 `expansion`。
3. 跑 `rustrain ops check`：数值 / 展开等价 / 梯度 / 确定性四项全过，才允许被 plan 选中。

**不需要改框架任何一行代码，也不需要重编译框架。**

## 如何切换算子 / 精度 / 量化

改 recipe：

```toml
[kernel.ops.mlp_swiglu]
forward  = "cuda.fp8_block128"
backward = "autodiff"

[kernel.precision]
compute      = "bf16"
accumulate   = "fp32"
weights      = "fp8_e4m3"
quant_scheme = "per_block"
block        = [128, 128]
```

前向与反向精度独立可选。改动会反映在 plan digest 与 run manifest 里。

## GOTCHAS

- **QKV split layout**：Qwen3.5/3.6 `in_proj_qkv` 输出 **flat** 布局 `[Q_all | K_all | V_all]`，
  不是 per-head 交错。用 `split(qkv, [q_size, k_size, v_size])` / `narrow(-1, offset, size)`。
- **`emptyCache()` 是隐式同步**：`CUDACachingAllocator::emptyCache()` 内部 `cudaDeviceSynchronize()`。
  不得出现在训练循环里，只在 seq>4096 时调用。
- **CXX11 ABI**：插件与宿主必须用同一个 `_GLIBCXX_USE_CXX11_ABI` 和同一个 libtorch 构建，
  否则跨界传 `at::Tensor*` 会静默出错。
- **GLIBC**：编译机与运行机的 GLIBC 版本必须匹配（2.39 编译的二进制在 2.35 上跑不起来）。

## 知识文件

- `docs/design/kernel-first/spec.md` — 架构契约、交付物、验收标准
- `docs/agent/linear-attention.md` — 线性注意力结构、QKV 布局、delta rule（重启前遗留，仍然准确）
