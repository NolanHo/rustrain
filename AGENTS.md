# AGENTS.md — rustrain 架构规则

## 核心原则：编排是第一公民，算子是插件

rustrain 是训练框架，不是 kernel 库。差异化价值在**并行策略、通信调度、显存与精度编排**，
不在手写 GEMM —— 那些交给 cuBLAS / CUTLASS / FlashAttention / Tilelang。

框架不认识任何模型，也不认识任何具体 kernel。它只做三件事：

1. **解析** —— 把模型表达成算子图（plan）。
2. **解析实现** —— 按 recipe 为图中每个算子选出具体实现。
3. **驱动** —— 用编译好的计划把实现跑起来，并调度其中的通信。

**模型是数据。算子是插件。精度是配置。**

> **状态：从零重建中（2026-09-12 起）。** 重构前的实现整体作废，设计归档在
> `_internal_docs/archive/pre-rewrite/`，不要再从那里的文档推导设计。当前仓库是骨架：
> 算子管线 + plan 编译器 + 一致性门禁，无训练循环、无模型描述层。

## 开工前先读

| 文件 | 内容 |
|---|---|
| **`skills/architecture/SKILL.md`** | **架构操作规则：边界契约、不变式、禁止模式、四类变更的 checklist、验证门禁。改动 crate 边界 / ABI / plan IR / 算子注册 / 切分 / 显存策略 / 模型描述之前必须读。** |
| `docs/architecture.md` | 架构是什么：边界契约、核心模型（描述 → 编译 → plan）、算子与 Kernel 契约、计算路径、加载路径与无 GPU check 阶梯、先例与证据、crate 职责、缺口、待定决定 |
| `docs/design/kernel-first/spec.md` | 本次重构的变更规格与交付物验收 |

**规则只有一份，在 skill 里。** 这份文件不重复它 —— 两份不变式一定会漂移，而这个代码库的架构
存在的理由就是消灭"同一个事实有两个来源"。

## 分层与依赖方向

```
rustrain-cli        ops | plan                  （组合根）
rustrain-runtime    执行器 / 显存池 / collective 后端 / 一致性门禁
rustrain-plan       Plan IR / 切分传播与通信插入 / 显存规划 / 编译 / digest
rustrain-parallel   进程组 / rank 布局 / 切分规格与转换规则
rustrain-ops        算子字典 / 注册表 / recipe 解析
rustrain-abi        插件 ABI v1 / 装载 / 作者辅助
──────────────────────────────────────────────────────────
plugins (.so)       reference（语义真值） / aten / 第三方高性能 kernel
```

实际依赖图（`cargo tree` 实测，非声明）：

```
parallel  abi                       ← 无内部依赖
   │       │
   │       ├── ops ──┬── plan ── runtime
   │       │         │     ▲        ▲
   │       └─────────┴─────┴────────┘
   │       └── kernels（插件，不是框架的一部分）
   └────────────── cli（组合根）
```

## 一条最重要的判据

**T1 / T2 自由，T3 重编框架。** 换实现体、换声明契约 = 丢一个 `.so`；出现框架没见过的
**数学形态** = 重编。所以：

- 随实现变化 → **插件**
- 描述契约 → **描述符里的数据**
- 模型结构（哪些算子、怎么连）→ **模型描述里的数据**
- 框架必须自己判断 → **框架代码**（新增一种"判断"就是 T3，应罕见且应被注意到）

推论：**规则不得按算子名或张量名查框架侧的表**。`match op { "linear" => ... }` 会把 T2 泄漏成 T3。
细节与 checklist 见 skill。

## GOTCHAS

- **QKV split layout**：Qwen3.5/3.6 `in_proj_qkv` 输出 **flat** 布局 `[Q_all | K_all | V_all]`，
  不是 per-head 交错。用 `split(qkv, [q_size, k_size, v_size])` / `narrow(-1, offset, size)`。
- **执行器必须采纳算子返回的指针与 strides**：view 算子（`transpose`/`narrow`/`reshape`/`broadcast`）
  会把 `out.data` 指回输入，`broadcast` 更会给出 stride=0。按"形状 × 宽度"线性读会读到缓冲区外，
  而且**前几个元素恰好正确** —— 一致性门禁第一次运行抓到的就是这个。
- **CXX11 ABI**：插件与宿主必须用同一个 `_GLIBCXX_USE_CXX11_ABI` 和同一个 libtorch 构建。
- **GLIBC**：编译机与运行机版本必须匹配（2.39 编译的二进制在 2.35 上跑不起来）。

## 验证宿主

`root@47.94.214.197:26002`（8× L20X，CUDA 13，torch 2.11，Rust 1.98.1）。
本机无 GPU / 无 torch，是编辑与编译盒。门禁命令见 skill §6。
