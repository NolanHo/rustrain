# rustrain Docs

架构定义只有一个来源：

| 文件 | 内容 |
|---|---|
| [`architecture.md`](architecture.md) | **架构是什么**：边界契约（T1/T2/T3）、核心模型（描述 → 编译 → plan）、算子与 Kernel 契约、计算路径、加载路径与无 GPU check 阶梯、先例与证据、crate 职责、缺口、待定决定 |
| [`design/kernel-first/spec.md`](design/kernel-first/spec.md) | 本次重构的**变更规格与交付物验收** |

操作规则（不变式、禁止模式、三类变更 checklist、验证门禁）在 `skills/architecture/SKILL.md` ——
**规则只有一份，文档不重复它。**

重构**之前**的设计、计划与验证记录已作废，归档在 `_internal_docs/archive/pre-rewrite/`（不会被提交）。
其中的事实性结论已吸收进 `architecture.md` §5.2。
