# rustrain

一个 Rust 写的 LLM 训练框架。**编排是第一公民，算子是插件，模型是数据。**

> **状态：从零重建中。** 2026-09-12 起，重构前的实现（tch-rs + C++ FFI + 按模型分 crate 的硬编码结构）
> 整体作废并归档。当前仓库是新的骨架：**算子管线 + plan 编译器 + 一致性门禁，能在无 GPU 的机器上
> 跑完整测试**。还没有训练循环，也没有模型描述层。

## 这个框架是什么

差异化价值在**并行策略、通信调度、显存与精度编排**，不在手写 GEMM —— 那些交给 cuBLAS / CUTLASS /
FlashAttention / Tilelang。框架不认识任何模型，也不认识任何具体 kernel。它只做三件事：

1. **解析** —— 把模型表达成算子图（plan）。
2. **解析实现** —— 按 recipe 为图中每个算子选出具体实现。
3. **驱动** —— 用编译好的计划把实现跑起来，并调度其中的通信。

边界契约是 **T1 / T2 自由，T3 重编框架**：换实现体、换声明契约 = 丢一个 `.so` + 改 recipe；
出现框架没见过的**数学形态** = 重编框架（接受这个代价，换来实现的完全自由）。

完整定义见 [`docs/architecture.md`](docs/architecture.md)；操作规则（不变式、禁止模式、三类变更 checklist、
验证门禁）见 [`skills/architecture/SKILL.md`](skills/architecture/SKILL.md)。

## 当前有什么

| crate | 职责 |
|---|---|
| `rustrain-abi` | 插件 ABI v1：POD 描述符、插件入口、装载、作者辅助 |
| `rustrain-ops` | 算子字典、注册表、recipe 解析 |
| `rustrain-parallel` | 进程组、rank 布局、切分规格与转换规则 |
| `rustrain-plan` | Plan IR、切分传播与通信插入、显存规划、编译、digest |
| `rustrain-runtime` | 执行器、显存池、collective 后端、一致性门禁 |
| `rustrain-kernels` | reference provider（语义真值）—— **是插件，不是框架** |
| `rustrain-cli` | 组合根 |

```sh
export PATH=/root/.cargo/bin:$PATH

cargo test --workspace                     # 全部通过（本机无 GPU / 无 torch）
cargo clippy --workspace --all-targets     # 零 warning
cargo run -q -p rustrain-cli -- ops list   # 本机有哪些实现
cargo run -q -p rustrain-cli -- ops check  # 一致性门禁：两个实现算的是同一件事吗
cargo run -q -p rustrain-cli -- plan explain --tp 2   # 一个 plan 编译成了什么
```

**设计上的硬约束**：`abi / ops / parallel / plan / runtime` 的依赖闭包里不得出现 tch / libtorch / cuda。
这是核心能在无 GPU 机器上跑完整测试、以及"无 GPU 形状检查"能成立的前提。

## 下一步

**模型描述格式**（`docs/architecture.md` §1.2 与 §8 D8）：结构是数据（子图模板 + 重复），插件只提供原语。
**第一个验证样本：`Qwen/Qwen3.6-35B-A3B`** —— 先用它把架构走通、看会不会出问题；通过了再加其他模型。
模型事实（config、1045 个张量的命名与真实形状、TP 可整除性约束）在 `docs/design/qwen36-5d-example.md`。

## 验证宿主

`root@47.94.214.197:26002`（8× L20X，CUDA 13，torch 2.11，Rust 1.98.1）。
本机是编辑与编译盒，无 GPU / 无 torch。

## License

MIT
