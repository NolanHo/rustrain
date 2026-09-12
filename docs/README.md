# rustrain Docs

架构定义只有一个来源：

| 文件 | 内容 |
|---|---|
| [`architecture.md`](architecture.md) | **架构是什么**：边界契约（T1/T2/T3）、核心模型（描述 → 编译 → plan）、算子与 Kernel 契约、计算路径、加载路径与无 GPU check 阶梯、先例与证据、crate 职责、缺口、待定决定 |
| [`design/model-description.md`](design/model-description.md) | **模型描述的设计（D8/D9）**：轴与 mesh 的表示、layout 的推广（多维分片）、描述文件的四个部分、`expand` / `instantiate` 语义、与 L1/L2 的对应、以及两处被推翻的早期结论 |
| [`design/qwen36-5d-example.md`](design/qwen36-5d-example.md) | **实例走查（真实数据）**：Qwen3.6-35B-A3B 的 config、1045 个张量的命名与真实形状、TP 可整除性约束、五轴逐轴走查 |
| [`design/op-vocabulary.md`](design/op-vocabulary.md) | **算子词表与分解图**（唯一权威）：粒度规则、对现有 27 个原语的对账、需新增的 5 个原语、一层的分解图与每处通信的归属、MoE 形状的三条路 |
| [`design/plan-ir-baseline.md`](design/plan-ir-baseline.md) | **Plan IR 与编译器的现状基线**：逐字段清点（含死钩子）、pass 顺序与失败模式、编译产物与 digest 内容、intrinsic 词表、layout 的表达能力、运行期消费方式、模型描述必须满足的接口 |
| [`design/kernel-first/spec.md`](design/kernel-first/spec.md) | 本次重构的**变更规格与交付物验收** |

操作规则（不变式、禁止模式、三类变更 checklist、验证门禁）在 `skills/architecture/SKILL.md` ——
**规则只有一份，文档不重复它。**

重构**之前**的设计、计划与验证记录已作废，归档在 `_internal_docs/archive/pre-rewrite/`（不会被提交）。
其中的事实性结论已吸收进 `architecture.md` §5.2。
