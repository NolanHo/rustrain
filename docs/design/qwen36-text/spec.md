---
type: ChangeSpec
title: Qwen3.6-35B-A3B 文本路径：从模型描述到前向对齐 HuggingFace
description: 把 Qwen/Qwen3.6-35B-A3B 的文本解码路径表达成数据、编译成 plan、在无 GPU 机器上通过 L1/L2、并在验证宿主上让前向数值对齐 HF。
tags: [rustrain, qwen36, model-description, plan, check-ladder]
timestamp: 2026-09-12T12:00:00Z
---

# Spec — Qwen3.6-35B-A3B 文本路径

> **本文件是执行者的唯一 checkpoint 工件。** 零对话历史的 agent 从本文件恢复：先读契约层，再读上下文层，
> 校验 `待解决` 为空且各交付物状态与代码一致，然后从状态标记处继续。**禁止靠对话历史脑补。**

---

## 目标

把 **`Qwen/Qwen3.6-35B-A3B` 的文本解码路径**从"硬编码的 C++ 实现"变成"数据 + 插件"，
并让它在新的算子管线里跑通到**前向数值对齐 HuggingFace**。这是整套架构（描述 → 编译 → plan → 门禁）
的第一个端到端验证样本；通过之后才谈第二个模型。

**已完成的前置设计**（契约层的稳定来源，执行者必读）：

| 文档 | 它定什么 |
|---|---|
| `docs/architecture.md` | 边界契约 T1/T2/T3、不变式、kernel 契约、check 阶梯、五轴归类 |
| `docs/design/model-description.md` | 描述语言的四个部分、`expand`/`instantiate` 语义、轴与 mesh、layout 推广 |
| `docs/design/op-vocabulary.md` | 算子词表（唯一权威）、一层的分解图、融合粒度 |
| `docs/design/qwen36-5d-example.md` | 真实 config、1045 个张量的命名与形状、TP 可整除性约束、五轴走查 |
| `docs/design/plan-ir-baseline.md` | 现状基线（编译器 7 个 pass、死钩子、运行期消费方式） |

---

## 契约层（durable）

### 行为与稳定接口

**C1 · 描述文件的行为。** 一个 JSON 文件（`format: "rustrain.model.v1"`）加一个模型目录（含 `config.json`），
必须能确定性地产出**全局 Plan**：全部 slot 形状具体、`layout` 全 `Replicate`、每个 weight slot 带符号 binding。
同名冲突、表达式成环、binding 未命中任何 slot → 报错并指出冲突双方。

**C2 · 检查命令的行为。**

```
rustrain check --model <model-dir> [--checkpoint <dir>] [--tp N --cp N --ep N --dp N --pp N] [--json]
```

- **总是**跑 L1（结构）：plan 可编译、每个节点的算子可解析到实现、每个算子的 `infer()` 与声明形状一致、
  layout 传播完成、每个 `Partial` 都被兑现、每个 collective 都绑了轴、每个 slot 都有分配且无别名冲突。
- 给了 `--checkpoint` 时**追加** L2（加载）：每个 weight slot 恰好被一条 binding 命中；checkpoint 的每个
  张量要么被消费、要么被显式 `ignore`；transform + axes 推出的本地形状与 slot 形状一致；dtype 相容。
- **内存预算只产生 warning，不影响退出码**（见 `docs/architecture.md` §8 D12）。
- **退出码只由 `Fail` 决定。** 每个检查项的结果是 `Pass` / `Fail` / `Warning` / `Skip`，**每条 `Skip` 必须写明理由**
  （沿用 `ops check` 的纪律：skip 不等于 pass，必须能回答"缺什么、什么时候能补上"）。
- **本机对 bf16 的"实现可用性"是 `Skip` 而不是 `Fail`**：reference provider 只声明 `f32`，而真实描述是 `bf16`，
  且 `moe_layer` 尚未实现（5 个新原语已有 4 个落地）—— 这是**这台机器的限制 + 尚未开工的 D5**，不是 plan 的缺陷。理由必须逐条列出。
- `--dtype <name>` 覆盖描述里的 dtype，用于显式声明"我在什么精度下检查"；报告里记录实际用的 dtype。
- 退出码 0 = 无 `Fail`；`--json` 输出机器可读报告（逐项结果、`Skip`/`Warning` 及理由）。
- **不执行任何计算，不需要设备**：不得创建 CUDA 上下文；插件在 `init()` 之前不得碰设备。

**C5 · checkpoint 元数据的来源与形状检查**

- `--checkpoint <path>` 接受两种形式：**真实模型目录**（`model.safetensors.index.json` + 分片，只读头部）
  或**元数据快照文件**（`*.safetensors.meta.json`）。快照是小的纯 JSON，让测试可以离线、可重复。
- 快照格式：`{"format":"rustrain.ckpt_meta.v1","source":<url或路径>,"tensors":{"<name>":{"dtype":"bf16","shape":[..]}}}`。
- **再生脚本**：`scripts/fetch_qwen36_meta.py` 通过 HTTP Range 只读分片头部（**不下载权重**），
  生成快照。脚本进仓库；快照进 fixture（约 100KB）；两者都能独立重建。
- L2 的检查**必须显式按 slot 名配对**，**不得把 source 实例顺序与 `ResolvedBinding.slots` 直接 zip**
  （后者的顺序是按 target 分组的，见 D1 审查记录）。
- 视觉塔的 333 个张量由描述里的 `ignore` 列表**显式声明**，不得静默丢弃。

**C6 · 报告形状与判据（精确定义，实现者不得自行发明）**

`--json` 的报告形状固定为：

```json
{ "format": "rustrain.check.v1",
  "model": "<path>", "checkpoint": "<path|null>", "dtype": "<生效的检查精度>",
  "counts": { "slots": 0, "nodes": 0, "weights": 0, "bindings": 0,
              "slots_unbound": 0, "tensors_unconsumed": 0, "shape_mismatch": 0, "dtype_mismatch": 0 },
  "checks": [ { "id": "<稳定 id>", "status": "pass|fail|warning|skip",
                "reason": "<必填，非空>", "details": ["<一行一条>"] } ] }
```

- **退出码 = 0 当且仅当没有任何 check 的 `status` 是 `fail`。** `warning`/`skip` 不影响退出码。
- `--json` **无论成败都往 stdout 打完整报告**；诊断也可以同时进 stderr。
- `check.id` 的稳定命名：`l1.structure`、`l1.implementation_availability`、`l2.binding_coverage`、
  `l2.tensor_consumption`、`l2.shape_reconciliation`、`l2.dtype_compatibility`。
- **`l1.implementation_availability` 是 `skip`**（不是 `fail`），当某算子**已知但本机没有可用实现**时；
  `reason` 必须列出未解析的算子清单与原因（这既覆盖"reference provider 只声明 f32"，也覆盖
  "尚未实现的新原语（如 `moe_layer`）"）。**每条 skip 都要能回答"缺什么、什么时候能补上"**（沿用 `ops check` 纪律）。
- **`--dtype <name>` 的语义**：只影响 **L1 解析实现时使用的精度**（默认取描述的顶层 `dtype`），
  **不影响 L2 的 dtype 比对** —— L2 永远拿**描述声明的** dtype 与 checkpoint 的 dtype 比。
  这样"在 f32 下检查结构"与"校验 bf16 权重与描述是否相符"两件事互不干扰。
- `ignore` 的模式语法与 binding 的 source 一致，额外支持 **`**` = 任意多段**；视觉塔用 `"model.visual.**"`。
- **`ignore` 模式命中 0 个张量 → `warning`（不是 `fail`）**：同一份描述可能被另一份 checkpoint 复用，
  但"声明了却匹配不到任何东西"很可能就是拼错，必须有人看得见 —— reason 必须**点名那个模式**。
- **`transform` 的完整词表只有两个动词**：`transpose(i, j)` 与 `slice(dim, start, len)`；
  多段拆分由 binding 的 **`split` 字段**表达，不在 `transform` 里。
  早期列的 `take` / `concat` / `split(dim,sizes)` **移除** —— 没有消费者、也没有定义 = 死钩子。
  **未知动词在 `expand` 期报错**，不得降级成 `skip`（"契约没写"不是 skip 的理由）。
- **`slice(dim, start, len)` 的语义**：把该轴长度换成 `len`；要求 `start >= 0`、`len > 0`、
  `start + len <= size`（`checked_add`）；负 `dim` 从末尾数。越界 = `shape_mismatch`（`fail`），
  **不存在"动词求不了值所以 skip"的分支**。
- **`**` 只属于 `ignore`**：`binding.source` 里出现 `**` 在 `expand` 期报错 ——
  一条 binding 的职责是"一个具体 checkpoint 张量 ↔ 一个具体 slot"，否则会退化成笛卡尔配对。
- **配对必须是一一对应**：`pairs` 数必须等于 binding 覆盖的 weight slot 数；不等则 `fail` 并点名，
  且 shape/dtype 两项改为 `skip`（不得在不可信的配对上给结论）。
- **报告的 check id 完整清单**（C6 的六个 + 本轮新增）：
  `l1.structure`、`l1.compile`、`l1.operator_shapes`、`l1.layout_propagation`、`l1.partial_fulfillment`、
  `l1.collective_axes`、`l1.slot_allocation`、`l1.implementation_availability`、`l1.instantiate`、
  `l1.binding_coverage`、`l2.binding_coverage`、`l2.tensor_consumption`、`l2.shape_reconciliation`、
  `l2.dtype_compatibility`、`l2.ignore_coverage`、`cli.arguments`。
- **`l1.instantiate` 检查什么**：用 mesh 与描述声明把全局 plan 实例化成**每个 PP stage 的一个代表
  rank**（stage 的 pp 坐标；`pp=1` 即 rank 0 一个）——声明轴名必须在 mesh 里、每个声明分片必须整除成
  本地形状、每个 stage 必须实例化出节点（空 stage 是 stage 声明错误，`fail`）。`details` 逐 stage 给出
  该代表 rank 的节点数/槽数（机器可读绊线）。传播三项只评估 stage 0（rank 0），其余 stage 不传播
  （跨 stage 接缝决策是 D5 的），且三项的 reason 必须写明这一点。
- **`l1.structure` 只声称它真做的两件事**（load/expand + `check_structure`）：C2 里依赖 compile 的
  **三条**（可编译 / `infer` 比对 / slot 分配）由各自的 `l1.*` 项**显式 `skip` 并写明"缺什么、什么时候
  能补"**，不得用 `pass` 冒充"没跑"（layout 传播 / `Partial` 兑现 / collective 绑轴自 D4 起是真检查）。
- **门禁必须断言期望的 skip 集合**：`check_report_contract` 把 16 个 check id、4 个 `skip`、每项状态、
  `counts` 的八个键**及其在真实 fixture 上的取值**、报告自称的 model / checkpoint / dtype、以及
  `l1.instantiate` 的 `details`（逐 stage 的节点数/槽数，五轴 mesh 下钉死 PP 裁剪后的两 stage 计数）与
  另外两处 `details` 的内容写死。将来某项从 skip 变真检查（D5）会让它按设计变红 —— 那是绊线，不是噪音。
- **`ignore` 模式必须锚定**：首段必须是**字面名**。`**`、`*`、`{*}.visual.**`、`*.visual.**` 全在
  expand 期报错 —— "全部忽略"等于放弃显式声明（C5）。锚定只管首段（`model.**` 合法）；报告里
  **逐模式**给出命中数。
- **没有任何配对可查**（`pairs` 空 ∧ 配对本身一一对应 ∧ 无未绑 slot）→ 四个 L2 项**全部 `warning`**，
  不得 6/6 全绿。
- **"没测量"写 `null`，不写 `0`**：`shape_mismatch` / `dtype_mismatch` 在配对不可信（`pairs != covered`）
  或为空时是 `null`。把"没查"写成 0 与本文件自己的原则冲突。
- **已知边界（记录、不修）**：配对"一一对应"是**计数**不变式，保护漏配 / 重配 / 笛卡尔 / 一 tensor 两 slot，
  **不保护"配错了人"**（source 与 target 互换且形状 dtype 全同 → 全绿）。真正的验证要靠数值对齐（D5）。

**C3 · 并行语义。** axis 是 mesh 里的**有序命名轴**，组是轴掩码；一个 slot 的 layout 是
**多个 `(dim, group)` 分片 + 至多一个 partial**。本地形状 = 全局形状沿分片维除以该组度数之积，
**不整除 = 编译期错误**。转换只做单步单轴；多步转换必须由描述显式写出中间 layout。
五轴里 TP/CP/DP/EP 只改形状与通信，**只有 PP 改节点集合**。

**C4 · 前向语义。** `run --model <dir> --seq <n>` 在**单进程**下跑一次前向，输出 logits 与每层 hidden 的摘要。

### 不变量（违反即失败）

- **I-1**：`rustrain-{abi,ops,parallel,plan,runtime}` 的依赖闭包里不得出现 tch / libtorch / cuda。
- **I-2**：模型结构只出现在描述文件里；框架代码与 kernel 实现里不得出现本模型的任何张量名或层数。
- **I-3**：路由/通信要么由布局算术推出，要么由算子的 `collectives` 声明 —— 不得有第三条路。
- **I-4**：切分轴住在 binding（参数映射）里；不得在框架侧按算子名或张量名查表。
- **I-5**：写错的描述必须**报错**，不得静默降级。

### 已确认约束（来源标注）

| 约束 | 来源 |
|---|---|
| 只做**文本路径**；视觉塔（333 个张量）本 spec 排除 | 用户陈述 |
| 第一个验证样本是 Qwen3.6-35B-A3B；设计不得为它窄化（加第二个模型不应改语言） | 用户陈述 |
| 只做**前向**；反向、优化器、训练循环、五轴实际运行不在本 spec | 用户陈述 + 本 spec 范围 |
| 内存管理留空：峰值只 warn，不拦编译 | 用户陈述（`architecture.md` D12） |
| 不引入新的融合算子；先跑展开形态 | 用户陈述（粒度选"中"，迭代 1 只需原语 + 模板） |
| 可复现性：fixture 用**再生脚本**，不 vendor 72GB 权重 | dev-sop 宪法 |
| 无 GPU 机器必须能跑全部 L1/L2 测试 | `architecture.md` I-1 |

---

## 交付物

（状态标记：`- [ ]` 未开始 / `[-]` 进行中 / `- [x]` 完成。完成须带证据：commit / 测试输出。）

### D1 — 描述文件能表达这个模型

**可观察结果**：存在一份 `model.json` + 一个模型目录（`config.json` 来自 HF 公开仓库），
展开后得到一个全局 Plan：**节点数 > 900**、**总 slot 数 > 900**。实测（D5 runner 收尾后的描述）：**1285 节点 / 2222 slot / 873 weight slot** —— 数字随 D4/D5 的逐 head 修正与 MoE 契约对齐增长，权重槽与 binding 始终未变。

> **注意：weight slot 数与 checkpoint 张量数不相等，也不该相等。**
> checkpoint 有 **712** 个文本+MTP 权重张量（文本 693 = 根 3 + 层内合计 80+270+60+280，加 MTP 19）。
> 描述只对**三处**拆开融合存储 —— `in_proj_qkv` → Q/K/V、`conv1d` → q/k/v、融合 `gate_up` → gate/up ——
> 因为它们的段序是连续 `[Q\|K\|V]` / `[gate\|up]`，按 TP 连续切一刀会切开语义边界。
> **`q_proj` 不拆**：HF 的真实段序是 **per-head 交错** `[q₀\|gate₀\|q₁\|…]`，连续切分正好给出完整 head 对；
> q/gate 的分离在**激活**上用 `reshape`+`narrow` 做（`docs/design/qwen36-5d-example.md` §3）。
> **weight slot 实测 = 873**（= 712 + 2×30 `in_proj_qkv` + 2×30 `conv1d` + 41 `gate_up`）。
> 权威验收是上面两个 `> 900`；**不要为了凑 weight slot 的数字把被拆掉的融合张量本身也留成 slot** ——
> 那会造出无人读取的 slot。
> "每个 slot 都有来源、每个张量都被消费"的对账是 **D2** 的验收（离线预演已通过：46 条 binding 精确覆盖 712/712）。

**交付位置**：描述文件与 fixture 同处（见 D2），格式定义在 `docs/design/model-description.md` §3 与 §3.6。
**验收与证据**：
- `cargo run -q -p rustrain-cli -- plan explain --model <dir> --json | jq '.nodes|length'` > 900 且 `'.slots|length'` > 900
- 同一输入两次运行得到**逐字节相同**的 plan JSON（确定性）
- 同名冲突 / 表达式成环 / binding 未命中 / 写未声明 slot / `select`+`template` 冲突 / 声明了没人读的 slot，各有测试证明会报错
- **形状对账**：每个 weight slot 的 `transform`（+`split`）必须能把真实 checkpoint 形状映射成声明的 slot 形状 ——
  已用真实分片头部独立跑过（`scripts/fetch_qwen36_meta.py`：index 的 `weight_map` + 每个分片的头部，无权重）：**873/873 一致**（这是唯一能证伪 `transform` 的机械手段）

### D2 — L2 加载检查对账真实的 1045 个张量

**可观察结果**：`rustrain check --model <dir> --checkpoint <ckpt-meta>` 退出 0，
报告 `slots_unbound = 0`、`tensors_unconsumed = 0`、`shape_mismatch = 0`。
**交付位置**：`rustrain check` 子命令 + 一个再生脚本（拉 `model.safetensors.index.json` 与分片头部，
通过 HTTP Range，不下载权重）。
**验收与证据**：
- `rustrain check --model crates/rustrain-model/tests/fixtures/qwen36-text --checkpoint <快照> --dtype f32` 退出 0，
  报告 `slots_unbound = 0`、`tensors_unconsumed = 0`、`shape_mismatch = 0`、`dtype_mismatch = 0`；
  "实现可用性"为 `Skip` 且逐条写明理由（见 C2）
- 一个**故意漏掉一条 binding** 的用例必须 `Fail` 并指出漏了哪个 slot 模式
- 一个**故意在描述里多声明一个不存在的 checkpoint 张量**的用例必须 `Fail` 并指出那个模式
- 一个**故意写错 `transform`**（例如漏掉 `transpose`）的用例必须 `Fail` —— 这正是 D1 里 `lm_head.w` 的漏网之鱼
- `tensors_unconsumed` 走的是描述里显式的 `ignore` 列表（视觉 333 个）；删掉 `ignore` 必须 `Fail`
- 形状对账是**机械**的（快照来自真实分片头部），不是人工核对

### D3 — 轴与 mesh：任意组合组 + 形状算术

**可观察结果**：`GroupMask` 能表达 `tp|ep`、`tp|dp` 等任意组合；`ParallelLayout` 能表达同一张量上
多条互不相干的分片；本地形状算术对 `docs/design/qwen36-5d-example.md` §5 的表格逐行成立。
**验收与证据**：
- `cargo test -p rustrain-parallel -p rustrain-plan` 全绿
- 一个测试用**真实形状**断言：`ep=4, tp=2` 下 `experts.gate_up_proj` 的 gate 段本地形状 = `[64, 256, 2048]`
- 一个测试断言 `tp=3`（`16 % 3 != 0`）在编译期报错，而不是运行期

### D4 — instantiate 与 L1 全绿

**可观察结果**：给定 mesh，`instantiate` 产出具体 Plan（本地形状、组掩码；**位置常量推迟到 D5**），
L1 全绿（按契约 skip 的四个项除外）；`l1.instantiate` 的 `details` 逐 PP stage 给出代表 rank 的
节点数/槽数（机器可读绊线，钉在 `check_report_contract` 里）。
**验收与证据**：
- `rustrain check --model <dir> --tp 2 --cp 2 --ep 4 --dp 2 --pp 2` 退出 0（L1 部分）
- `rustrain check ... --tp 3` 非 0，且 `l1.instantiate` 点名**哪个约束**：`slot embed.w` + dim 0 +
  全局 248320 + 除数 3（vocabulary 轴上的 tp 分片不整除，`248320 % 3 != 0`）
- 一个测试证明 `Partial` 被兑现：row-parallel linear 之后插入了 `all_reduce({tp})`
- 一个测试证明 PP 裁剪：`pp=2` 时 stage 0 的节点集合不含 layers 20–39

### D6 — 多 rank 执行与并行效果（用户 2026-09 追加）

**用户的三条决定**：① 权重用宿主上的共享路径 `/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B`（67 GB，**不下载**）；② **8 卡并行，要测出 TP / EP 等并行效果**；③ 卡上占显存的进程"没啥用"（但那 8 个进程在另一个 PID namespace 里，宿主 `ps` 看不到；且**本次不需要**——每卡 ~82 GB 空闲，8 卡分片后每 rank 只需 ~9 GB）。

**必须说清的前提**：reference provider 是**纯 Rust CPU** 实现（I-1）。所以 D6 能测的是**并行机制的正确性与代价**，不是 GPU 壁钟加速：

1. **正确性**：8 个 rank 进程（同机 localhost）按 mesh 各持自己的分片，真实执行集合通信，logits 必须与 world=1 的单进程前向一致（容差另定，与 HF 的 1% 是两回事）。
2. **代价指标**（每 rank、每配置）：权重字节数（分片是否真的减少）、步骤数、集合通信次数与字节量、峰值显存/内存投影 —— 这些是"并行效果"在计划层的真实读数，也是 TP/EP 选择该看的数。
3. **壁钟**：8 个 CPU rank 的吞吐不会加速（reference kernel 是标量 Rust，且 8 份进程争同一台机的核），把它测出来并如实报告，不假装是加速。

**GPU 加速不在 D6**：需要一个新的 kernel 插件（CUDA/Tilelang/CUTLASS），按架构那是 T1 实现体，换插件不改框架。它是 D6 之后的独立交付，也是唯一能把"并行效果"变成倍数的路径。

**配置扫描**（每项都要跑通并出数）：`tp=8`；`tp=4,ep=2`；`tp=2,ep=4`；`tp=2,cp=2,ep=2`；`dp=8`；以及 world=1 作为基准。

### D6 状态（2026-09，逐步推进）

1. **ATen 插件**（提交 `14136b0`）：28 个算子映射到 ATen（cuBLAS / FlashAttention-2 / cuDNN /
   ATen 组合），C++17 + ABI v1；宿主上 `ops list --plugin` 列出 60 个实现，GPU 逐算子自检 31/31。
2. **框架设备路径**（`14a9255`）：`--device cuda` 的 driver-API 分配器（核心 crate 仍零 CUDA 依赖）、
   按设备对齐的显存打包、`ops check` / `run` 的插件与 recipe 参数、按变体声明选设备执行。
3. **框架门禁**：`ops check --plugin librustrain_aten.so --recipe plugins/aten/aten.toml
   --device cuda` → **59 case / 0 failing**，reference 留在 host、`cuda.*` 跑在显存。
4. **第一次真前向找到的 4 个 bug**（`612a33a`，全部是"形状对、数值错"）：集合通信自己解引用设备指针
   （SIGSEGV）、`reshape` 拒绝跨步输入（4 层模型跑不动）、runner dump 读编译前的旧 slot（编译器把
   消费者改写到补全后的 twin）、**原地集合通信的输入没有被保活到输出**（pool 把它的字节发给了之后的
   激活）。四个都补了回归测试并各自反证。
5. **全模型在 GPU 上跑通**：40 层、window 512、f32，前向 7.2 s（1382 步 / 1328 算子 / 54 次集合
   通信），峰值 134.1 GiB，42 个 hidden 摘要——单卡 140 GiB 装得下。
6. **判定结果：未达标（诚实记录）**。`compare` 的读数：row 0（embedding）2e-6 ✓；row 1 起开始
   发散（最早以 `mean` 超差，而 `mean` 是近零统计量，被相对化放大）；`std` 约 1%（2 层）→ 7%
   （第 11 层）→ 末层 `max` 54%；`logits` 相对差 **1.29**（max_abs_diff 16.75 / max_abs 13.0），
   即 logits 是错的。**离"对齐"还差得远，不是 bf16 舍入能解释的量级。**
7. **两个真 bug 已修（`6ec7e93`、`0369cd0`，都有反证过的回归测试）**：
   - `rope` 的位置轴：契约说"位置沿第一轴"，实现却把位置放在**中间轴**（`[rows, seq, dim]` 展平）。
     rank-2 的 conformance case 里 `rows == 1`，两种索引顺序一致，所以门禁看不见；plan 的 q/k 是
     `[512, 16, 256]`，于是每个全注意力层的 q/k 用错了角度。已加 rank-3 用例 + 语义测试。
   - **整个 view 家族都是别名**：`compute_lifetimes` 只对名字叫 `view` 的算子延长输入的生存期，
     而 `reshape`/`narrow`/`transpose`/`broadcast` 同样交回输入的指针（执行器采纳它）。全注意力层的
     `reshape → narrow → reshape` 链都指向 q_proj 的缓冲，pool 却把该缓冲发给了之后的 MoE 激活 ——
     注意力读的是被覆写的字节。另外这条链还要**逆序**传播（`a → reshape → narrow` 的正向单遍会把头部
     用过期的尾部值延长），这两点都是新测试先变红逼出来的。
8. **修完后的三方对比（f32 与 bf16 参考都跑了；数字是 42 行摘要的最坏相对差）**：

   | 比较 | std 最坏（中位） | max 最坏 | logits rel_max | logits rel_L2 | argmax |
   |---|---|---|---|---|---|
   | HF f32 vs HF bf16（参考自身的精度敏感度） | 2.15e-2（2.3e-3） | 8.0e-2 | **1.52e-1** | **8.4e-2** | 220 = 220 |
   | rustrain f32 vs HF f32 | 2.60e-2（3.4e-3） | 6.1e-2 | **9.9e-2** | **6.3e-2** | 220 = 220 |
   | rustrain f32 vs HF bf16 | 2.51e-2（1.0e-2） | 1.06e-1 | 1.57e-1 | 9.3e-2 | 220 = 220 |

   也就是说：**我们与 HF f32 的差，比 HF 自己 bf16 与 f32 的差还小**（logits 上小得多），而两者的
   argmax 一致、top-5 有 4 个重合。末层 `norm.y`（lm_head 的输入）std 差 0.1%、max 差 1.5%。
9b. **逐元素对比推翻了"只是累积误差"的说法（`4c0099a` 后测的）**：双方都 dump 每层 probe 行的原值后，
   f32-vs-f32 的逐层 **rel_L2** 是：row 0（embedding）**0.0**（逐位相同）、row 1（**只过了一层 GDN**）
   **4.9e-2**、row 2–3 约 6.5e-2、row 4–10 5–11e-2、row 11+ 稳定在 2.4–3.7e-2。
   也就是说：**第一层之后就差了 ~5%，这不是四十层累积出来的**，而"std 只差 1%"之所以看起来还行，是因为
   误差与信号近似正交（5% 的 L2 误差落在 std 上只有 ~0.1%）。摘要掩盖了它，逐元素暴露了它。
   **下一步**：在 layer 0 内部逐算子对比——把 `outputs.hidden` 写成
   `[embed.y, layers.0.h1, layers.0.attn, layers.0.h2, layers.0.h3, layers.0.moe, layers.0.y, layers.*.y, norm.y]`
   就能 dump 出该层的每个中间张量；HF 侧用 forward hook 抓 `input_layernorm` / mixer / `post_attention_layernorm` /
   MoE 的返回值，然后在同一个 8-token 探针上逐元素比。谁先偏 1% 就是谁。

9. **关于"<1%"这条判据的结论（需要用户裁决）**：1% 是对 **bf16 参考** 定的，而这份模型自己的
   f32↔bf16 差就有 8–15%（logits）——即 1% **低于参考自身的噪声地板**。逐层的 mean 也几乎为 0
   （|Δmean|/std ≤ 3e-3），用"相对 mean"衡量只会放大噪声。诚实的判据应该是二者之一：
   （a）**逐层逐算子把差压到噪声地板以下**（1 层/3 层截断的 f32 对比，正在做）；
   （b）把判据改成"我们的 f32 与 HF f32 的差 ≤ HF 自身跨精度差"，并说明为什么。
10. **下一步（正在做）**：把 1 层描述 + 只含 layer 0 的 checkpoint 跑出来，与 HF 的 1 层截断前向
   **逐元素**对比，再逐算子往层内走（HF 侧用 forward hook 抓中间张量；rustrain 侧把中间 slot 名写进
   `outputs.hidden` 就能 dump）。
11. **已排除的（逐条对着 HF 源码 + 实测核对，不是猜）**：
   - `causal_conv1d` 的核朝向：宿主上实测 `causal_conv1d_fn(x, w) == F.conv1d(x, w, padding=K-1)[:, :, :L]`
     **逐位相等**（差 0.0）→ 不需要翻转权重，我们 `at::conv1d` 的用法正确；`conv1d` 没有 bias
     （`nn.Conv1d(bias=False)`，checkpoint 里也确实没有 `conv1d.bias`）。
   - MoE 权重朝向：`gate_up_proj [E,2I,H]` → `transpose(1,2)` + `split(dim=2)` ✓；`down_proj [E,H,I]`
     不转置、算子内 `h @ down[e]^T` ✓；`mlp.gate` 转置后按 `linear` 用 ✓；shared expert 三个权重是
     `[out,in]` 按 `h @ W^T` 用 ✓；router 的**无条件重归一化**（`norm_topk_prob=true`）✓；
     `hidden_act=silu` ✓。
   - 全注意力层：`q_proj` 逐 head 的 `[q|gate]` 交错（`reshape [512,16,2,256]` + `narrow`）✓、
     `q_norm`/`k_norm` 用 `1+w` 且只在 head 维 ✓、`rope` 只转前 64 维（`partial_rotary_factor=0.25`）✓、
     `sdpa` 的 `1/sqrt(256)` ✓、`o * sigmoid(gate)` ✓、`o_proj` ✓；残差与两层 norm 的顺序与 HF 的
     `Qwen3_5MoeDecoderLayer.forward` 逐行一致。
   - `rmsnorm` 的 `weight_offset=1.0`（HF `Qwen3_5MoeRMSNorm` 确实是 `x_norm * (1 + weight)`）✓；
     GDN 的 `rmsnorm_gated` 用**裸权重**（HF `RMSNormGated` 是 `w * x_norm * silu(gate)`）✓；
     `l2norm` 的 eps=1e-6、逐 128 head ✓；`gated_delta_rule` 的 `1/sqrt(128)` 查询缩放 + l2norm 在
     核外 ✓；`beta = sigmoid(b)`、`g = -exp(A_log) * softplus(a + dt_bias)` ✓。
12. **关于"每层 1% 复利"的旧假设已被本次结论取代**：修掉 rope 与 view 别名之后，深层尺度不再漂移
   （第 11 层 std 差 0.04%，第 40 层 0.8%，末层 0.1%），剩下的差已降到参考自身的精度敏感度之下。
   旧假设的文字保留在这里只为记录判断过程：不是布线错误，而是**每层约 1% 的乘性偏差在复利**——证据是偏差随深度单调放大
   （std 相对差 1% → 7% → 末端 1.5–2 倍），而 `norm.y`（RMSNorm 会归一掉尺度）只差 6%，末 token 的
   `argmax` 两边都是 220（方向对了）。1% 的每层偏差：(1.01)^40 ≈ 1.5 与观测吻合。要定位它，只能像
   §7 那样逐层逐算子对值，而不是看摘要。

### D5 — 前向数值对齐 HuggingFace

**可观察结果**：在验证宿主（8× L20X，sm_89）上，同一段 token、同样的 `input_ids`，rustrain 的 logits 与
HF transformers 的 logits 在容差内一致；每层 hidden 的 mean/std/max 差异 < 1%（沿用旧框架验证过的方法）。

**判定用哪条执行路径（2026-09 用户裁定后的修正）**：用户禁止**除对照以外**的 CPU 执行，且对照只能小规模
谨慎启动。因此候选侧**不再是** CPU reference provider 的整模型 f32 前向，而是 **GPU 插件**：
`plugins/aten/`（`cuda.aten.f32`，ATen 实现体），缓冲区在显存，recipe 用 `plugins/aten/aten.toml`。
reference provider 退回它唯一该在的位置——`ops check` 的逐算子 oracle（小张量）。

**上机步骤**（在 `root@47.94.214.197:26002` 上；本机到这一步为止的部分已全部完成）：

```bash
# 0) 构建 GPU 插件（宿主：python3.13 venv，torch 2.11.0+cu130）
python3 plugins/aten/build.py --out /root/rustrain-gpu/aten-build

# 1) 参考侧：HF 前向（bf16，固定探针，8 个 token），权重用共享路径、不下载
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=<一张卡> python3 scripts/hf_qwen36_reference.py dump \
        --model /vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<rev> \
        --dtype bf16 --device-map auto --out /var/tmp/hf-ref.npz

# 2) 候选侧：rustrain 前向（同一固定探针；权重 bf16 原样读入、精确加宽到 f32 执行）
cargo run --release -p rustrain-cli -- run \
        --plugin  /root/rustrain-gpu/aten-build/librustrain_aten.so \
        --recipe  plugins/aten/aten.toml \
        --device  cuda \
        --model   crates/rustrain-model/tests/fixtures/qwen36-text \
        --checkpoint <真实 safetensors 目录或 index.json> \
        --tokens "9707,11,1879,0,323,358,314,279" \
        --out /var/tmp/rustrain.npz

# 3) 判定：logits 与逐层 hidden 摘要的相对差 < 1%，并打印第一处超差的层
python3 scripts/hf_qwen36_reference.py compare \
        --reference /var/tmp/hf-ref.npz --candidate /var/tmp/rustrain.npz --json
```

两侧的前向各自独立：参考侧是 HF 的 bf16（跑在 GPU 上），候选侧是"同样的 bf16 权重加宽到 f32 后执行"。
1% 的容差要吸收的是 **HF 的 bf16 舍入**，不是我们的加宽（bf16 ⊂ f32，加宽是精确的）。`compare` 先比对
`input_ids`，不一致直接判定"不可比"。

**已知上机风险（诚实清单）**：① 整模型 f32 常驻约 134 GB —— 单卡放不下，单卡跑必须分片（`--tp 2` 起）；
② `tp/cp/ep/dp > 1` 的分片执行需要真实的 collective 后端，多卡 CUDA 还需要"一进程一卡"的启动形态
（当前 `run` 的多 rank 是同进程多线程，只适用于 CPU 对照），NCCL 后端是 D6 第 4 步；
③ `pp > 1` 拒绝执行（跨 stage 接缝是未决项）；④ 探针超过 512 需要新描述（§3.10 #3）。

**验收与证据**：
- bf16 下 `max_abs_diff(logits) / max_abs(logits)` < 1%
- 逐层 hidden 摘要差异 < 1%（前 n 层逐层打印，定位第一处发散的层）
- `rustrain ops check --plugin librustrain_aten.so --recipe aten.toml --device cuda` 退出 0，
  每个 aten 变体都与 reference oracle 逐算子对齐；每条 skip 写明理由

---

## 依赖与门

- D2 依赖 D1；D4 依赖 D3；D5 依赖 D4。
- **人类门**：D5 需要在验证宿主上跑，且需要 HF 作为对照实现 —— 提交前需用户确认可在该机跑（占用 GPU）。
- **不在本 spec**：反向图、优化器、训练循环、PP 微批调度、五轴真实运行、视觉塔、MRoPE、量化、内存管理。
- 本 spec 默认**单 PR**；拆多 PR 是偏离，需用户显式指定。

---

## 待解决

（空 —— 非空则阻塞执行。）

---

## 上下文层（volatile 快照）

> **快照于 `ae61912` + 2026-09-12**。会过时；执行者须重新探索核对，不盲信本层。

### 基线锁定

- 目标分支：`main`；基线提交：`ae61912`。
- 工作树：干净（本 spec 提交时）。验证宿主：`root@47.94.214.197:26002`（8× L20X，CUDA 13，torch 2.11，Rust 1.98.1）。
- 本机（编辑/编译盒）：无 GPU、无 torch。`CARGO_TARGET_DIR=/tmp/tgt-lead`，cargo 在 `/root/.cargo/bin`。
- 门禁现状：`cargo test --workspace` 219 通过、clippy 零 warning、`ops check` 退出 0。

### 探索证据（快照，file:line 会漂移）

- 编译器 7 个 pass 与其真实行为：见 `docs/design/plan-ir-baseline.md`（含死钩子清单）。
- `shard::propagate` 收到 `ProcessGroups` 后 `let _ = groups;` 丢弃 —— 拓扑今天没人校验。
- `ParallelLayout` 是单值 `Shard{dim, group}`；`GroupKind` 封闭六值 + `ProcessGroups.groups: [_; 6]` 定长。
- `memory.rs` 只取 `req.workspace_bytes`，从不累加 `save_for_backward_bytes`（D12 已决定延后）。
- 真实模型事实（config、1045 张量、形状、可整除性）见 `docs/design/qwen36-5d-example.md`。

### 探索笔记的位置（与 spec 异地的说明）

本 spec 未新建 `qwen36-text/notes.md`：探索笔记就是上面五份设计文档，它们同时被
`docs/README.md` 与 `architecture.md` 索引。**移动它们会破坏索引**，所以采用"引用而非复制"。
这是对 spec 规范"co-locate"的**有意偏离**，已在此披露。

### 驱动力

旧实现把模型结构写在 Rust + C++ 两处（`docs/architecture.md` §5.2），七个模型的权重加载各写一遍，
新增一个模型 = 改框架。本 spec 验证的假设是：**结构是数据，插件只提供原语，那么支持一个模型 = 写一份
描述 +（必要时）几个原语**，并且这份描述能在无 GPU 机器上被机械检查。

### 被放弃的方案（指针）

- 细粒度（每个数学步骤一个节点）/ 粗粒度（每层一个算子）→ 选"中"：`op-vocabulary.md` §0。
- 结构化分片（一个 slot 记住段边界）→ 选"拆成多个 slot"：`op-vocabulary.md` §8.1。
- EP 当作 instantiation → 修正为 layout + 显式 routing：`model-description.md` §6.1。
- 度数晚绑定（一个 plan 跑所有度数）→ 选"度数作为编译输入"：`architecture.md` §1.6。
- 内存预算硬失败 → 改为 warning：`architecture.md` §8 D12。

### 交付物履行状态

- [x] **D1 — 描述文件能表达这个模型** —— 证据：提交 `746acc1`（主体）+ `3057544`（返工）；
  实测 `nodes 1285 / slots 2222 / weight slots 873`，两次展开逐字节相同（有专门测试钉确定性，不在此钉某个哈希 —— 描述改一次哈希就该变一次）；
  6 + 3 条契约测试 + 工作区 265 passed / clippy 0 warning / `ops check` exit 0；
  46 条 binding 独立对账 **712/712** 文本+MTP 张量（0 未命中、0 未覆盖、0 重复、0 视觉）；
  **形状对账 873/873**（快照取自真实分片头部，含 `transform`/`split` 的机械验证）；
  独立审查 `APPROVED_WITH_NOTES`，其指出的缺陷已全部修掉（见下）。
- [x] **D2 — L2 加载检查对账 1045 个张量** —— 证据：提交 `460178f`（主体）+ `8f9728a`、`d37f936`（两轮返工）+ `b793362`、`27c4bfd`、`5da56ba`、`a2c47ca`（第四、五轮：审查复现的七条"假绿"全部关闭）；
  真实快照（1045 张量 / 仅分片头部 / 无权重 / 无网络）下 `rustrain check --dtype f32` 退出 0，
  counts = `{bindings 46, nodes 1285, slots 2222, weights 873, slots_unbound 0, tensors_unconsumed 0, shape_mismatch 0, dtype_mismatch 0}`；
  "实现可用性"在 `--dtype f32` 下已由 `skip` 变 **pass**（D5 把最后一个缺失算子 `moe_layer` 补齐后，当前 1285 个节点全部解析）；在描述自己的 `bf16` 下仍是 `skip` 且逐行写明成因（reference provider 只收 f32），该形态由 declared-dtype 那次运行钉住；
  四个故意破坏的用例各自 `Fail` 并点名：漏一条 binding → 槽名、多声明一个不存在的张量 → source 模式、漏掉 `transpose` → 槽名 + 两个形状数字、未被消费的张量 → 张量名；`ignore` 删掉即 `Fail`；
  门禁 `check_l2.rs` **9 条** + `check_report_contract.rs` **4 条**：C6 的 15 个 id、14 个状态、8 个计数的**值**、报告自称的 model / checkpoint / dtype、两处 `details` 的**内容**（按 dtype 分表：`f32` 该项为 pass、`details` 为空；`bf16`·`f16` 与默认路径是 16 行 / 1285，每行还要求 `why` 与成因相符、reason 的**开头总数**与表一致）、以及**不带 `--dtype`** 与**显式 `--dtype bf16` / `f16`** 三条路径（这三条原本都没有门禁）；
  每条新断言都用**变异反证**过，十四种变异全部变红：计数漂移 / 计数写成 `null` 或 `46.0` / 清空 `details` / 只在默认 dtype 下清空 / 把 bf16 表截断回 5 行 / 把显式 bf16 改成 `pass` / 单条算子计数改动 / 单条理由掏空 / 理由与成因矛盾 / 重复条目 / ignore 总数或 reason 总数改动 / reason 丢掉或篡改总数 / 表与和互不自洽 / `dtype` 谎报 / model 指向别的目录 / 同一 id 两次；
  工作区 **304 passed** / clippy 0 warning / `ops check` exit 0；
  独立审查**五轮**：第三轮 `APPROVED_WITH_NOTES`（N1 已关；N2 锚定只约束首段、N3 前后空白当字面段 → `model-description.md` §3.8）；第四、五轮（三个独立视角）各自复现了"报告无用但门禁全绿"的路径（`dtype`/`model`/`checkpoint` 谎报、同一 id 两条、reason 与 details 数目不符、默认 dtype 下清空 details、bf16 表可被截断、显式 dtype 无门禁、reason 可篡改总数、`why` 可自相矛盾），**全部已修并各自反证**。
  已知局限（写在这里而不是假装不存在）：`model`/`checkpoint` 只能断言"回显了传进去的路径"，一个回显 argv 却读别处描述的实现在这份门禁下仍是绿的；被拒绝的 `--dtype` 那一次运行的 `details` 内容按设计不钉（该列表随 dtype 变化）。
- [x] **D3 — 轴与 mesh + 形状算术** —— 证据：提交 `3fe05a5`（`rustrain-parallel`：`Mesh` / `GroupMask` / 多分片 `ParallelLayout` / 形状算术）+ `b00b22a`（`rustrain-plan`：plan 带 mesh 指纹、`ATTR_GROUP` 变成掩码整数、`GroupUnavailable` 首次真正触发）+ `cfe60b3`（runtime / model / cli 收尾，工作区转绿）+ `a11d6c0`（D3 让哪些文档失效就更正哪些）+ `ce5bbca`（对抗审查复现的 5 条错误答案一律改为拒绝）；
  `GroupMask(u32)` 能表达 `tp|ep`、`tp|dp`、`ep|dp` 与全掩码；`ParallelLayout { dims, partial }` 让同一张量上两条互不相干的分片成为可能；
  **真实形状**：`ep=4, tp=2` 下 `experts.gate_up_proj` 的 gate 段（全局 `[256, 512, 2048]`）本地形状 = **`[64, 256, 2048]`**（`rustrain-parallel/tests/layout.rs`，数字取自 `qwen36-5d-example.md` §2/§3/§5）；
  **不整除是编译期错误**：16 heads 切 `tp=3` → `NotDivisible { dim: 0, global: 16, divisor: 3 }`，消息点名 dim / 全局尺寸 / 除数与 "compile-time"；
  **旧算术等价**：D3 之前的 `stride_extent` 闭式被原样抄进测试，对 11 组 mesh（含 `4,3,2,5,2` 与 degree 1）逐 rank 比对 `group_index` / `group_id` / 成组顺序 —— 0 处不同；审查者另写独立暴力 oracle 扫 654 组 mesh / 436,762 次检查，同样 0 处不符；
  `GroupUnavailable` 从"声明了从不构造"变成活的：`propagate` 拿 plan 自己的 mesh 指纹校验每个 layout，越界即报错并点名节点 / 算子 / 掩码；
  门禁：`cargo test -p rustrain-parallel -p rustrain-plan` 全绿；工作区 **345 passed** / clippy 0 warning / `ops check` exit 0；
  独立审查两条（不同视角）：**行为保持** = `APPROVED_WITH_NOTES` —— `check` 报告与 `ops check` 输出**逐字节不变**，三处差异全在设计内（digest、新的 `mesh` 字段、collective 组名 `Tp`→`tp`），42 处测试改动逐一核对**无一处被削弱**（多处被加强），findings 全是文档账，随 `a11d6c0` 修掉；**算术与转换表** = `CHANGES_REQUIRED` —— 审查者用**数据级模拟器**（把 layout 实例化成每 rank 的具体张量、逐元素执行发出的集合通信）复现 5 条：相交组的 `shard+partial` 被回答而非拒绝（只发一次集合通信，编译器会放行）、`reduce_scatter` 与 gather 的次序、同组双维分片不可能重建、未兑现的目标分片被接受、`ATTR_GROUP` 的 `i64→u32` 静默截断。随 `ce5bbca` 全部改为**拒绝**，并用审查者自己的 oracle 复跑验证：`attack3_table` 从 FAIL（8053 findings）变为 **PASS（9612 次模拟一致 / 0 findings / 0 panic）**，组算术与形状算术两次攻击保持 PASS。三条新规则写进 `model-description.md` §2.3。
- [x] **D4 — instantiate 与 L1 全绿** —— 证据：提交 `2d00afa`（两条裁定：`stage` 与符号轴名）+ `98c84db`（`instantiate` + 规则表的 rank 感知）+ `c602c1c`（CLI 用真实 mesh 驱动 `check`、三项 L1 变真检查、内存预算改 advisory）；
  `rustrain check --model <qwen36 描述> --checkpoint <真实快照> --dtype f32 --tp 2 --cp 2 --ep 4 --dp 2 --pp 2` **退出 0**：15 项中 11 pass / 4 skip / **0 fail**（新增 `l1.instantiate`；`l1.layout_propagation`、`l1.partial_fulfillment`、`l1.collective_axes` 由 skip 变真检查）；`--tp 3` **退出 1**，`l1.instantiate` 点名 `slot embed.w` + dim 0 + 全局 248320 + 除数 3；
  `instantiate` **不碰注册表**，所以在"`moe_layer` 本机没有实现"的真实 bf16 描述上照样跑 —— 不整除因此能在解析之前就 `fail`。剩下 4 条 skip 各自写明"解析不完整 / 本机没有实现"与何时能跑（C2：skip 必须回答缺什么、什么时候能补），**没有把没跑的检查写成 pass**；
  **PP 裁剪**：`pp=2` 时 rank 0 的节点集合 = `embed` + layers 0–19，不含 20–39、`norm`、`lm_head`、MTP；跨界 slot 变成 plan 的 Input/Output（`pp_pruning_keeps_exactly_the_rank_stage` 按实例前缀与 slot kind 断言）；
  **Partial 兑现**：row-parallel 的声明分片让 linear 输出成为 `partial(sum, tp)`，`Compiler::compile` 插入的 `all_reduce` 掩码就是 `tp`；vocabulary 分片的 embedding 同样欠一次 `all_reduce({tp})`（两条测试各一）；
  **真实形状贯通**：声明的切分 → 本地形状在真实描述上端到端成立（`ep=4, tp=2` 的 gate / down 段）；
  **预算改 advisory**：`enforce_budget` 仍是唯一检测点，但超预算只进 `CompiledPlan.warnings`（`plan explain` 打印、JSON 里也在），退出码不受影响 —— §8 D12 落地；
  工作区 **384 passed** / clippy 0 warning / `ops check` exit 0；绊线按设计变红并更新为 D4 的新真相（16 个 id、4 个 skip、三项新状态，`l1.instantiate` 的 `details` 逐 stage 钉住节点/slot 计数），其余断言原样保留：8 个计数的**值**、provenance、两处 `details`、重复 id 规则、三条 dtype 路径；
  独立审查两条：**契约诚实性** = `APPROVED_WITH_NOTES`（绊线无一处被削弱、三条新 pass 各自构造反例证明能失败、六个 D3 pair 测试仍走到转换表、L2 半部带 mesh 逐字节不变、digest 与声明切分解耦）；**算术对抗** = `CHANGES_REQUIRED` —— 复现三条**静默错布局**（`MatMul` 丢掉一个操作数的分片、rank 变化的 unary view 给轴改名、`check` 只看 rank 0 而放过只在 stage 1 的不整除）与两条较小项，随 `0b7aa0f` 全部关闭（每条都先写成红色复现再修）；
  **已知缺口（诚实记录，不是"以后再说"）**：① **位置常量**（flat QKV 通道偏移 / CP 序列偏移 / 本地专家范围）推迟到 D5，`instantiate` 里留注释指到 §4.2 第 4a 行；② **PP 接缝上的 partial 没有所属 rank**：stage 1 的 `propagate` 在 MTP 的 rmsnorm 处按"转换的所有者不在本 rank"拒绝 —— 把 partial 原样交给下一 stage，还是在 seam 前补完，是 D5 的跨 stage 通信决策；模型测试里钉住这个拒绝并注明 D5，不特判、不假装通过
- [ ] D5 — 前向数值对齐 HuggingFace

**D2 的已知输入（来自 D1 审查）已兑现**：`ResolvedBinding.slots` 的顺序是**按 target 分组**（先所有 `q` 槽、
再所有 `gate` 槽），不是按 source 实例交错。D2 没有 zip 两个顺序，而是**按捕获替换逐张量配对**
（`main.rs` 的 `Pair`：source 命中哪些 checkpoint 张量、捕获如何替换成 slot 实例名），并要求结果**一一对应**，
否则报"不是配对"而不是把笛卡尔积当对账。

**D2 已知未设门禁的 C6 分支**（都是人手验证过的正确行为，只是没有 fixture 钉住；不是缺陷，但下一个动
`check` 的人要知道它们没有绊线）：

| C6 分支 | 现状 | 谁验过 |
|---|---|---|
| 没有任何配对可查 → 四个 L2 项**全部 `warning`**，不得 6/6 全绿 | 正确（空 weight slot 描述 + 空快照 → 4 warning，`l2.ignore_coverage` 是 `skip`，退出 0） | 独立验证（手工 fixture，未提交） |
| "没测量"写 `null` 不写 `0` | 正确（不带 `--checkpoint` 时 `shape_mismatch` / `dtype_mismatch` / `tensors_unconsumed` = `null`，`slots_unbound` 是真测过的 0） | 独立验证（手工运行） |
| `pairs != covered` → 点名 `fail`，shape/dtype 改 `skip` | 无 fixture 触发 | 未验 |
| 多条 `ignore` 模式 → **一项**、N 行 `details` | 正确（2 条模式 → 1 项 2 行） | 独立验证（手工 fixture） |
| 被拒绝的 `--dtype` 那次运行里可用性 `details` 的内容 | 故意不钉：该列表随 dtype 变化（拒绝后回落到描述自己的 `bf16`），只钉了计数与 id 集合 | 独立验证（16 行 vs 5 行） |
