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

### D6 — 多 rank 执行与并行效果（2026-09 修订：GPU-only、一进程一卡）

**用户的三条决定（不变）**：① 权重用宿主上的共享路径
`/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B`（67 GB，**不下载**）；
② **8 卡并行，要测出 TP / EP 等并行效果**；③ 卡上占显存的进程本次不需要。

**2026-09 修订（用户约束覆盖本节原始形态）**：用户裁定 **除小规模对照外不再允许 CPU 执行**。
原始 D6 的形态（8 个 CPU reference-provider 进程 + 共享内存 rendezvous）**作废**：它违反该约束，
而且参考 kernel 是标量 Rust，测出来的不是并行效果。修订后的 D6 = **一进程一卡 + GPU 插件
（`cuda.aten.f32`）+ 真实集合通信（NCCL）**；reference provider 退回 `ops check` 的逐算子 oracle。

**交付物**

- **D6.1 传输与启动**：`rustrain-runtime` 的 NCCL 后端（运行期 `dlopen`，核心 crate 的依赖闭包仍
  零 CUDA —— I-1 不变）；`--rank/--world/--rdzv` 的单 rank 执行形态；`rustrain launch` 以 N 个
  OS 进程一进程一卡启动（`std::process::Command`，不残留进程）；NCCL unique id 经**文件
  rendezvous** 交换（每次运行独立目录，超时报错而不是挂死）。
- **D6.2 正确性**：同一段 token 下，每个可执行的 world>1 配置的 logits 与 **world=1 GPU 前向**
  一致；判据是逐元素 rel_L2 与 `max|diff| / max|logits|`（阈值随数据记录），**与"对齐 HF 的 1%"是两回事**。
- **D6.3 代价指标**（每 rank、每配置）：权重字节（分片是否真的减少）、步数、集合通信次数与收发
  字节、峰值显存投影 / 实测、墙钟。8 个 rank 的墙钟**不承诺加速**，如实报告。

**可行配置矩阵（2026-09 实测判定；写在这里，而不是假装都能跑）**

| 配置 | 状态 | 证据 / 阻断点 |
|---|---|---|
| `world=1`（GPU 基线） | ✅ | D5：40 层 f32，1382 步，6.6 s，峰值 134.2 GiB |
| `tp=2` | ✅ 可执行（编译通过，数值待多进程验证） | 见下面的 D6.0：规则改为描述符声明后才编译得通；2 个 kv 头 ÷ 2 = 1，GQA 分组语义正确 |
| `tp=4` / `tp=8` | ✅ **可执行（2026-09 落地，见 §D6.6）** | `check --tp 4`/`--tp 8` 全绿；宿主机 4/8 进程真多卡：logits 与 world=1 `max\|diff\| 3.34e-5 / 2.77e-5`（界 1.298e-4）**PASS**，42 层 hidden 最坏 rel_L2 **1.65e-6 / 1.84e-6**，argmax 全同；每 rank 权重 34.6 / 18.3 GiB（world=1 是 132 GiB），峰值 35.8 / 19.3 GiB |
| `ep>1` | ⛔ 阻断 | `moe_layer` 声明了两个 `ALL_TO_ALL {tp, ep}`，但**计划器与编译器从不读 `RsCollective`**（读者只有 ABI 访问器和一个断言 `n_collectives == 2` 的测试）→ 声明式集合通信这条路径（I-3 的第二条）没有实现，EP 跑起来不会通信 |
| `cp>1` | ⛔ 空转 | 描述里没有任何槽声明 `cp` 轴（只有 `tp`/`ep`），`--cp N` 只是给 mesh 加一个没人用的轴；rope 位置、卷积边界、跨 rank K/V 交换都还没有位置常量与通信声明 |
| `dp>1` | ⚠️ 无通信 | 前向没有梯度可归约；DP 只复制权重。如实报告"权重字节不减少、集合通信 0 次" |

**执行顺序（2026-09 修订）**：**D6.0 规则体系**（tp=2 先编译通过；实测它一直编译不过）→ D6.1（任何配置都需要的
传输）→ 用 `tp=2` 端到端验收 D6.2 / D6.3 → 再逐个打开阻断项
（tp≥4 的 KV 复制 + head 偏移位置常量；EP 的声明式集合通信路径）。**每个阻断项都改契约面，开工前
单独向用户确认。**

**不在 D6**：反向、训练循环、CP 的真实语义、多节点（跨机）rendezvous、上游速度 kernel。

### D6 状态（2026-09，逐步推进）

> 1–12 是**修订前的历史记录**（ATen 插件 → GPU 前向 → D5 判定），保留为判断过程的来源；
> D6.1 / D6.2 / D6.3 的当前状态追加在 12 之后。

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

### D6.0 — 切分规则成为描述符的声明（ABI v2，2026-09）

**为什么先做这个**：D6 原本以为 TP 只差传输。实测不是 —— `rustrain run --tp 2` 在真实描述上**编译不过**：

```
node NodeId(36) (causal_conv1d@reference.f32) inferred output 0 shape [512, 1024] but slot SlotId(30) declares [512, 2048]
```

根因在 `shard::rule_for`：它**按算子名**分类，而模型自己的 7 个算子（`causal_conv1d`、`gated_delta_rule`、
`l2norm`、`rmsnorm_gated`、`sdpa`、`topk_router`、`moe_layer`）全部落到 `ShardRule::Declared`，
于是分片传不过去：输入被切成 `[512,1024]`，输出还是 replicate 的 `[512,2048]`。而 `check --tp 2` 报全绿
是因为 `l1.compile` / `l1.operator_shapes` / `l1.slot_allocation` 三项是 **skip**（见 C6 的 D5 账）。
**`tp > 1` 在这个模型上从未可执行过**。这正是 SKILL 的 I-5（"规则不得按算子名查框架侧的表"）所禁止的形态。

**做了什么（用户裁定：规则进描述符，ABI v2）**：

1. **ABI v1 → v2**：`rs_op_desc` 追加 `shard`（`DECLARED` / `ELEMENTWISE` / `LINEAR` / `EMBEDDING` /
   `MATMUL` / `PASS_THROUGH`），184 → 192 字节；C 头、C fixture 的 `_Static_assert`、Rust 尺寸/偏移断言同步。
2. **框架**：`shard::rule_for` 删除，改为 `ShardRules` 声明查询；`Registry` 实现它（**同一算子的多个实现
   必须声明同一条规则**，不一致是错误；未知编号也是错误，不降级）；`instantiate` 与 `propagate` 都按声明求值。
   新增规则种类 `PASS_THROUGH`（输出跟随输入 0、其余输入保持自己的声明）—— 这是 T3，值得，因为它
   一次性覆盖 7 个算子。
3. **两个 provider 各自声明**：reference（32 个算子）与 ATen 插件（28 个），同名算子声明一致。
4. **描述侧的三处真错**（都是 tp>1 才暴露的"形状对、语义错"）：
   - reshape 的**被分片维度写成字面量**（`[512,16,128]`）→ 改为 `-1`（按本地元素数解析）：
     描述是拓扑无关的，字面量的 head 数是全局事实，tp=2 上本地只有 8 个 head。写错是硬错误，不静默。
   - **MoE 的 tp 轴按错了维**：`experts.down_proj` 是行并行，要切**收缩维 I**（dim 2），原来切的是输出维 H
     （dim 1）；shared expert 的 gate/up 是列并行（切输出维 dim 0），原来切的是输入维 H；shared down 同理
     反过来。切错维**形状照样整除、网络照样跑**，只有数值对比能发现。
   - `sdpa` 的 `num_heads` / `num_kv_heads` 是**全局**计数，分片后与张量矛盾 → 改为 `per_head` 布尔声明
     形式，头数**从张量自己的轴读**（GQA 分组 = 本地 q 头 / 本地 kv 头）。
5. **`check` 的边界随之变清楚**：一个**没有任何 provider 发布**的算子没有可读的切分规则，`l1.instantiate`
   直接 `fail` 并点名（`nonexistent_op`）；"有实现但本机跑不了"（dtype/device/sm）仍然是
   `l1.implementation_availability` 的 `skip`，退出码不受影响。`check_l2` 的对应用例按新真相重写。

**证据**：新增回归 `run::tests::the_real_plan_compiles_after_the_runner_surgery_at_tp_two`
（真实描述 × tp=2 → 实例化 + runner surgery + 编译通过；断言 layer 0 的 q 本地形状 `[512,1024]` 且插入了
all_reduce）。工作区测试全绿、clippy 0 warning、`ops check` exit 0。
**D6.0b — 门禁补洞（同一轮，提交 `ba2afb3`）**：`l1.compile` / `l1.operator_shapes` /
`l1.slot_allocation` 从 `skip` 变成**真检查** —— stage-0 的实例化 plan 真的过 `Plan::compile`，
三项分别报"编译通过（步数/插入的集合通信/digest）"、"每个节点的 `infer` 与声明形状一致"、
"每个 slot 都被放置且没有运行时执行不了的 policy"。`--dtype` 在编译前应用到 plan 上（否则
`--dtype f32` 会去编译 bf16 的 plan，被 f32-only 的 reference 拒绝，把宿主的限制报成 plan 的缺陷）。
报告契约的绊线按设计变红并更新：`--dtype f32` 的 skip 集合**从 3 项变成空**，bf16 / 被拒绝的 dtype
仍是 4 项。实测：`check --dtype f32` 在真实 checkpoint 元数据上 exit 0、15/15 全绿
（1327 步、42 次插入的集合通信、峰值 143.3 GB）；`check --tp 2` 现在能看到 tp=2 的 plan 编译通过；
`check --tp 4` 仍 fail 并点名 `layers.3.kh`。

**仍未做（诚实记录）**：tp≥4 卡在 KV 头整除（需 KV 复制 + head 偏移位置常量）；EP 卡在声明式集合通信
未接线（`moe_layer` 声明的两个 `all_to_all` 仍然没有消费者 —— 门禁也看不见它，因为没有检查项断言
"声明的集合通信都被接进 plan 了"）；`cp>1` 仍是空转（没有槽声明 `cp` 轴）。

### D6.1 — NCCL 传输与 `launch`：一个只在真多卡上现形的偏移错

**交付**：`rustrain-runtime` 的 NCCL 后端（运行期 `dlopen` `libnccl.so.2`，核心 crate 的依赖闭包仍零
CUDA 链接）；`run --rank/--world/--rdzv` 的单 rank 形态；`rustrain launch` 以 `world = tp×cp×ep×dp×pp`
个 OS 进程一进程一卡启动（rank `i` → `cuda:base+i`，CUDA context 只得在一个线程里 current，所以一张卡
一个进程）；NCCL unique id 走**文件 rendezvous**（root 写 `comm-<mask>.id`，write-then-rename，其余成员
轮询、有界超时，失败点名文件与 root）。运行期依赖仍是 host 侧只有 `libloading` + `libc`。

**首次 tp=2 端到端跑出来的缺陷（本节的重量级内容）**：

- **症状**：`tp=2` 的 logits 与 world=1 的 `max|diff| = 12.70`（相对界 1.3e-4 → FAIL）。但逐段看：
  **前半（rank 0 自己的 vocab 分片）逐元素正确**（`max|d| 2.67e-5`，正是 GEMM 分块噪声量级），
  **后半（rank 1 贡献的 124160 列）全错**；而两半的**统计量几乎相同**（`mean -2.1609` vs `-2.1696`、
  `std 1.4812` vs `1.5039`、`max 7.46846` vs `7.46847`，8 行 argmax 全对）—— 看上去像"噪声偏大"。
- **定位（三步，都可复现）**：① 按行做匹配矩阵 → `t1[i] == s1[i+1]` **逐位相等**（8 行里 7 行），即不是
  置换、不是尺度错，而是**整体平移一个 slab**；② 一个 slab = 本地 vocab 宽度 = `along*inner` →
  指向"本 rank 的 send 偏移多算了一个 slab"；③ 读代码：`all_gather` 的 send 写成
  `member_offset(outer, comm.index, along, inner, 1)` —— `degree` 传 1 却把成员序号折进了**本地**缓冲区
  一侧，等价于 `(outer + index) * slab`。rank 0 恰好正确（`index=0`），rank 1 整块平移，且最后一个
  outer block 会**读过本地缓冲区尾部**（第 8 行匹配不上的原因）。`reduce_scatter` 的 recv 是**同一个
  错误**（镜像方向），当时没触发而已，一并修掉。
- **为什么门禁没抓住**：host 组装后端（`collective.rs`）是**另一个实现**且写法正确、测试全绿；CPU 盒上
  的 D6 验收（`run_multi.rs` 的 `the_sharded_logits_agree_with_the_world1_forward`）跑的就是它，而
  `run-tiny-par` 的 logits 在 dump 时已是 replicate —— **没有任何测试曾经 gather 过一个
  按 vocab 切分的张量**。一个后端的绿灯对另一个后端的偏移算术零信息。
- **修法与三条防复发（都已落地）**：
  1. 偏移算术**只有一处**：`all_gather_offsets` / `reduce_scatter_offsets`，执行路径与测试共用；
     "本地缓冲区没有成员轴"写在函数名与注释里，`local_offset` 不再接受成员序号（错误形态**写不出来**）。
  2. `nccl.rs` 新增两条**内存模拟测试**（不需要 GPU）：把 rank 标进数据、复现 NCCL 的选块语义
     （`send` 是本地块，成员轴只在目标形状里），断言组装结果等于数学拼接；**变异反证过**
     （把成员序号折回本地一侧 → 立即越界 panic）。
  3. 新 fixture `run-tiny-logits`：head 绑在 vocab 轴上，于是 **NCCL 端到端**也有秒级回归
     （宿主上 `launch --sweep tp=2;tp=4`）。变异反证：把 send 改回错误形态 → tiny 也 FAIL
     （`max|diff| 2.7e12`），修好后 **逐位相同**（`0.000e0`）。CPU 盒的 `run_multi` 覆盖计划侧
     （dump 宽度 + 数值），宿主覆盖执行侧。
- 规则已写进 `skills/architecture/SKILL.md` §3 禁止模式与 §5，以及 `docs/architecture.md` §3.1。

### D6.2 — 正确性：tp=2 与 world=1 前向（**PASS**）

同一段 token（`9707,11,1879,0,323,358,314,279`）、同一份 67 GB checkpoint、同一 `cuda.aten.f32` 插件、
f32 执行，`rustrain launch --sweep tp=2` 的真实数字：

| 判据 | 读数 | 界 |
|---|---|---|
| `logits` `max\|diff\|`（8×248320 全张量） | **2.670e-5** | `1e-5 × max\|logits\| = 1.298e-4` → **PASS** |
| `logits` rel_L2 | **1.331e-6** | — |
| 前半 / 后半 vocab 的 `max\|d\|` | 2.670e-5 / 2.575e-5 | 分片两侧**对称**（修好后才对称） |
| 8 行 argmax | 全同 | — |
| 42 层 hidden 逐层 rel_L2 | 最坏 **1.638e-6**（42 行全部 ≤1.6e-6） | `first divergence > 1e-3: None` |

**这些数字说明什么**：两侧都是 f32、同一批权重，差别只可能来自 GEMM 分块与归约顺序 —— 2.7e-5 正是
f32 重结合的噪声量级，比任何"切错维/漏一次通信/拿错 slab"的量级低 6 个数量级（错误版本是 12.70）。
**与"对齐 HF 的 1%"是两回事**：那条判据在 D5 已判定为低于参考自身噪声地板；这里是分片实现的自洽性检验，
它**不**证明数学对，证明数学对的是 D5 的 HF 对比。rel_L2 与 `max|diff|/max|logits|` 都要看，不看 `mean`
（近零量，相对化只会放大噪声）。

### D6.3 — 代价指标（每 rank，真实模型，从 sweep 报告读出）

| 指标 | world=1 基线 | `tp=2` rank 0 | `tp=2` rank 1 |
|---|---|---|---|
| 权重字节 | 142,021,005,824（132.27 GiB） | **72,087,929,600（67.14 GiB）** | 同左 |
| 峰值字节（投影） | 144,068,886,528（134.17 GiB） | **73,788,211,200（68.72 GiB）** | 同左 |
| plan 步数 | 1370 | 1411 | 1411 |
| 集合通信次数 | 42（度数 1 的恒等） | **83**（1 all_gather + 82 all_reduce，组 = tp） | 同左 |
| 集合通信字节（发/收） | 0 / 0 | **598,212,608 / 852,492,288** | 同左 |
| 前向墙钟 | 6.525 s | **6.740 s** | **10.730 s** |
| checkpoint 读取字节 | 117,051,115,776（109 GiB） | 同左（**未分片**） | 同左 |

**如实报告**：① TP 把权重与峰值显存**各减半**（67.14 GiB/rank、68.72 GiB/rank），这是真实的并行收益；
② 墙钟**没有加速**（rank 0 6.74 s ≈ 基线 6.53 s，rank 1 10.73 s 更慢），本就不承诺 —— 但 rank 1 比
rank 0 慢 4 s 是**未解释的偏差**，记录为开放测量项：已排除的两个解释是"rank 序号相关的代码路径"
（tiny 上 tp=2/tp=4 的 per-rank 墙钟对称）与"这张卡慢"（宿主实测 `cuda:0` / `cuda:1` 的 8192³ f32
matmul 都是 21.46 ms / 51.2 TFLOPS、拷贝都是 ~4.25 TB/s）—— 下一步是在 rank 1 上打步级时间戳，
看这 4 s 是花在计算、集合通信等待，还是某个只在非 0 rank 上跑的收尾步骤；
③ **每 rank 仍读整份 109 GiB checkpoint**（`checkpoint_bytes_read` 不随分片下降），加载耗时是整个
sweep ~23 分钟的主因，是明确的性能缺口；④ 「8 个 rank 的墙钟」这件事本文没有承诺，也没有测。

**复现命令（宿主，`/root/rustrain-gpu/`）**：

```sh
# 秒级 NCCL 端到端回归（vocab 分片的 head → 真 gather）：tp=2 / tp=4 均须逐位相同
python3 scripts/make_tiny_checkpoint.py --out /root/rustrain-gpu/d6tiny-logits --with-head
./target/release/rustrain launch --model crates/rustrain-cli/tests/fixtures/run-tiny-logits \
  --checkpoint /root/rustrain-gpu/d6tiny-logits --tokens 0,1,2,3 --sweep "tp=2;tp=4" \
  --out /var/tmp/d6-tiny-logits-sweep.json --plugin /root/rustrain-gpu/aten-build/librustrain_aten.so \
  --recipe plugins/aten/aten.toml --device cuda --nccl-lib "$NCCL" --keep-rdzv

# 真实模型验收（baseline world=1 + tp=2，各读一遍 67 GB checkpoint）
bash /root/rustrain-gpu/host-d6-run.sh        # ./rustrain launch --sweep tp=2 ... --keep-rdzv
python3 /root/rustrain-gpu/host-compare.py    # 逐层 rel_L2 + logits（dump 在 <out>.rdzv/）
python3 /root/rustrain-gpu/host-logits.py     # 两半 vocab 的 max|d| / rel_L2 / 差异列
bash /root/rustrain-gpu/load-bench.sh         # 加载分相计时（read / fill / write）
python3 /root/rustrain-gpu/check_load.py      # 与上一次的 dump 逐位比对 + 新加载分相
```

### D6.4 — 加载路径：从 10 分钟到 1 分钟（2026-09）

**为什么单独做**：D6.2/D6.3 每次验收要跑两遍加载，一次 sweep 23 分钟里只有 14 秒是前向。
Debug 速度卡在这里，所以先修它。

**先量再改**（宿主机，world=1，真实 67 GB checkpoint）。这张表的读数来自一次**临时插桩的构建**：
先把分相计时加进加载器、测出下表的分解，再动优化；插桩本身随 `9d0d6eb` 一起提交，所以历史里没有单独的
"只有计时"提交（重跑办法见本节末尾的 `load-bench.sh`）：

| 相位 | 单线程读数 | 说明 |
|---|---|---|
| transform | **420.1 s** | `transpose` 的逐元素下标除法 |
| slice | 74.2 s | transform 内 slice + `split` 段 + 分片 slab，各自 `vec![0.0; n]` |
| widen | 70.6 s | bf16 → f32，全张量 |
| read | 34.2 s | 873 次整张量读（**热缓存**；冷缓存单流实测 ~160 MB/s） |
| write | 15.0 s | 逐个 slot 的 `cudaMemcpyHtoD`（串行，全部读完之后） |
| 合计 | **614.0 s**（进程 624.9 s） | CPU 占 92%，`Threads: 1` |

两个被数字纠正的判断：① `checkpoint_bytes_read` = 117,051,115,776 **不是**"整个 checkpoint" ——
它 = 71,010,502,912 的**去重字节**（= `weight_bytes`/4 × 2，正好 66.1 GiB）+ 46,040,612,864 的**重复读**：
873 个 pair 落在 712 个张量上，`split` 的每个段都把同一个张量整读一遍 —— 多出来的 161 次读 =
`in_proj_qkv` 3 段 ×30 层（+2 次/层）+ `conv1d` 3 段 ×30 层（+2 次/层）+ `gate_up_proj` 2 段 ×41 处
（+1 次/处），与 `qwen36-text` 描述的 split 一一对应；② 磁盘不是瓶颈（热缓存 34 s / 66 GiB ≈ 3 GB/s），
**CPU 是**。

**三个改动（都在 `crates/rustrain-cli/src/load.rs`）**：

1. **按 `(tensor, transform)` 合组**：一个张量只读一次、只 widen 一次，`split` 的各段从同一份数据上切。
   `bytes_read == bytes_distinct` 从此是**不变量**（`run_split.rs` 用 metrics 钉住：5 个 pair、3 个张量、
   读到的字节 = 三个张量之和）。
2. **16 个 worker 并行**（`LOAD_WORKERS`）：文件读与 CPU 各自并行。16 比 48 快，而且在**新旧两条路径上
   都成立**（旧链：61.2 s vs 48.6 s 进程墙钟、CPU-sum 407 s vs 565 s；新链：30.5 s vs 38.9 s）——
   更多 worker 只增加缓存压力与页错误。
3. **`Cuts`：把 transform / split / 分片 slab 合成为一张下标映射，一次遍历填充**。
   每个 binding 操作只有两种形态 —— "交换两个轴"（transpose）或"取某一轴的子区间"（slice、split 段、
   分片 slab），所以整条链可以在**不搬一个字节**的情况下合成：结果轴的 `(checkpoint 轴, 起点, 长度)`。
   之后按结果的 row-major 序走一趟 odometer，每个输出元素直接从未经中间缓冲的 checkpoint 字节读出来。
   原来那条链（widen 一份拷贝 → transpose 一份 → slice 一份 → 再 slice 一份）每个字节搬四遍。

   被替换掉的四个函数（`widen` / `transpose_axes` / `slice_axis` / `row_major_strides`）**没有删除**，
   改为 `#[cfg(test)]`：它们是 `Cuts` 的**参考实现**，`the_composed_walk_matches_the_explicit_chain`
   用一条覆盖四种操作的链逐元素比对新旧两条路径（bf16 取值精确可表示，所以比对是精确相等）。

**结果（宿主实测）**：

| | 改前 | 改后（三个提交依次落地） |
|---|---|---|
| 一次 sweep（baseline + tp=2） | **22 m 43 s** | **58–65 s**（22–23×；最近一次 1 m 01.8 s） |
| world=1 单次运行（进程） | 624.9 s | **30.5–32.9 s**（20×） |
| world=1 的**加载**墙钟 | ~614 s | **22.6 s** |
| 加载各相位读数之和（read/fill 是 16 worker 相加，write 是墙钟；同一次运行） | 406.9 s | **198.8 s**（fill 156.5 + read 24.0 + write 18.3） |
| 读到的字节（world=1） | 109.0 GiB | **66.1 GiB**（= 去重字节，整份 checkpoint） |
| 读到的字节（tp=2 每 rank） | 109.0 GiB | **43.8 GiB**（189,637 次收窄读，−34%） |
| `widen` / `transform` / `slice` 三个相位 | 71 / 420 / 74 s | **不存在了**（合并成一次 `fill`） |
| 设备回写 | 串行，全部读完之后 15 s | **与读/填充重叠**，写者 = 调用线程；**但回写本身仍是 18.3 s** |

`checkpoint_load` 的口径（读它的人必须先看这一句）：`wall_seconds` 是从 `load_weights` 入口起算的墙钟（含索引解析与 pairing；`LoadOutcome.wall` 在
加载函数的第一行起表）；
`read_cpu_seconds` / `fill_cpu_seconds` 是**对 `workers` 个线程求和**（不是墙钟，不能与前者相加）；
`write_seconds` 是调用线程上设备回写的墙钟，与两者重叠。四个数字相加是无意义的。

三个提交：`9d0d6eb`（去重 + 16 worker + `Cuts` 一次遍历）、`addc250`（把回写流式化：每个 slot 就绪即写，
`Executor` 不是 `Send`，所以**调用线程**当写者）、以及文档提交。回写这一步同时还去掉了宿主里那份
"每位权重再存一份 f32" 的 142 GB 副本。

**数值不变（这是本次改动的验收判据）**：新加载路径下重跑 tp=2，rank 0 的 `logits` 与全部 42 层
hidden **与改动前逐位相同**（`np.array_equal` → True，`max|d| 0.0`），digest 不变
（`4290629e4ac02128…`），sweep 判据仍是 `max|diff| 2.670e-5 / bound 1.298e-4 PASS`。加载器只决定
"同样的 f32 值怎么进 slot"，不决定值本身；这条判据把"只改了搬字节的方式"钉死。

**回写是最后的大头（改完才看见）**：world=1 的加载墙钟 22.6 s 里，调用线程在 `write_f32` 里真正花掉
**18.3 s**（142 GB pageable 源 → ~7.8 GB/s）；tp=2 因为每个 rank 只写自己那半（67 GiB f32），回写
10.0 s、加载墙钟 12.5 s —— 两个数同量级，也就是说**瓶颈是那次内存拷贝**，不是布局运算，也不是磁盘。下一步很明确：`cudaHostAlloc` 的 pinned 暂存 + 分批
`cudaMemcpyAsync`（pageable 拷贝要过一次内部暂存，pinned 能到 20+ GB/s），或按 slot 对齐后合并成大块。

### D6.5 — 只读这个 rank 需要的切片（2026-09）

`Cuts` 本身就是一张"哪些 checkpoint 元素是这次要用的"的地图，所以收窄读是它现成的推论，而不是又一次改动：

- **`Cuts::runs`**：把一个 box 分解成 checkpoint 里的连续段。规则很短 —— 取**最内侧那个"没有整轴覆盖"的轴**
  `k`，它之后就全是满轴，于是一段 = `len_k × stride_k` 个元素，段数 = `k` 之前各轴保留长度的乘积。
  专家张量 `[E, 2I, H]` 的中间轴是 tp slab 时，就是**每个专家一段**（256 段），而不是每行一段。
- 组内各成员的段取并集、排序合并，`read_at` 每段一次；**超过 `READ_RUN_CAP`（8192）段**就退回读覆盖窗口
  —— 这是 fallback，只可能读得更多，不会更少。
- 实测：**tp=2 每 rank 43.8 GiB / 189,637 次读**（收窄前 66.1 GiB），world=1 仍是 66.1 GiB（本来就该全读，
  这是正确性的一面）；加载墙钟 tp=2 14.4 → 12.5 s，读的 CPU 时间 31.9 → 23.7 s。
- **"读的正好是需要的"有两条可复现证据**：① tiny fixture（6 个张量、含 vocab 分片的 head）在 tp=2 下
  实测读 **384 B，等于手算的理想值 384 B**（= embed 64 + t0.up/2 + t0.down/2 + t1.up + t1.down + head/2）；
  ② 把 `READ_RUN_CAP` 放大 32 倍（262144）重跑 tp=2，读数与读数次数**完全相同**（43.8 GiB / 189,637）——
  说明没有张量落在 fallback 上，剩下的差距不在加载器。
- 数值不变：tp=2 的 dump 与加载器重写前**逐位相同**（`array_equal`，digest 不变），sweep 判据不变。

**本次没做的（诚实记录）**：① `pinned` 暂存（pageable 的 `cuMemcpyHtoD` 约 7.8 GB/s，回写 18.3 s 是加载里
最大的一段，且在 tp=2 上已经和 fill 同量级）；② 冷缓存时单流读只有 ~160 MB/s（16 worker 已经并行，但
checkpoint 不在 page cache 时仍是主要成本之一）；③ 更高度数（tp=8 / ep）受益更大，但那些配置本身还在
D6 的阻断项里（见下）。

### D6.6 — tp≥4 的 KV 复制：声明式 unit + 消费端边界（2026-09，已落地并验收）

**要解决什么**：`check --tp 4` 死在 `slot layers.3.self_attn.k_proj`：`dim 1` 全局 512
（2 个 KV 头 × 256）按 4 等分要切出 128 = **半个头**。
`qwen36-5d-example.md` §4 早就写了"tp≥4 需 KV 复制"，但它没说清**复制的粒度**。

**这一轮想清楚的三件事**：

1. **"head 偏移位置常量"其实不需要**。原先的设想是：每个 rank 持有一部分 q 头，sdpa 需要知道自己的
   q 头在全局的偏移才能把 q 头映射到 kv 头（GQA 的 repeats=8）。真正需要的是**让每个 rank 恰好持有它
   那部分 q 头需要的 kv 头**：q 头按 tp 连续切，rank `c` 的 q 头 `[c·q/tp,(c+1)·q/tp)` 映射到的 kv 头
   区间是 **`[⌊c·kv/tp⌋, ⌈(c+1)·kv/tp⌉)`**（`q = kv × repeats`，把 q 头号除以 repeats 即得，两边同一条
   除法），所以只要 **kv 轴的 slab 用同一条规则算**，本地 `sdpa` 的"本地 q 头 / 本地 kv 头"分组就自动
   正确 —— 偏移被 slab 本身吃掉了，算子不需要新的位置常量。
   **修正（独立审查抓到的）**：这条边界**不等于**"每 rank `ceil(kv/tp)` 个头"。两者只在 `kv` 是
   `tp` 的倍数、或每 rank 恰好 1 个头时相等。反例 `kv = 3, tp = 4`：rank 1 的 q 头需要 kv 头
   `[⌊3/4⌋, ⌈6/4⌉) = {0,1}` 两个头，而 `ceil(3/4) = 1` 只给一个 —— 按老写法算出来的 slab 会让
   attention **静默少一个头**。现在的实现对复制模式取 `[⌊c·U/d⌋, ⌈(c+1)·U/d⌉)` 的**整单元**（`U` =
   单元数），并在**各 rank slab 长度不一致**时拒绝：一个 slot 只有一个形状（`ShardError::NonUniformSlabs`），
   所以 `kv = 3, tp = 4` 是**编译期报错**，而不是让边界 rank 少一个头、或让别的 rank 拿一个装不下的
   缓冲区。`kv = 2, tp = 4/8` 下每 rank 恰好 1 个头，是合法用例，也是本模型实际用的那一档。
2. **复制必须按"单位"而不是按元素**。KV 投影的输出特征轴是 512（2 头 × 256），tp=4 时按元素
   `ceil(512/4)=128` 会切出**半个头**（本轮实测：`l1.instantiate` 通过后 `l1.compile` 在 reshape 上报
   `[512,128] cannot fill -1`，正是这个错误被抓住）。所以复制模式带 **unit**：
   轴按 `unit` 分成若干单元，rank `c` 拿 `floor(c·U/d)` 起、`ceil((c+1)·U/d)` 止的**全部整单元**
   （slab 可以重叠）。KV 权重声明 `unit = head_dim` → `kv=2, tp=4/8` 时每 rank 恰好 1 个头。
3. **声明式，不是 fallback**。`Divide`（默认，严格整除，不整除仍是硬错误）与
   `Replicate{unit}`（声明后才允许重叠）是两种**声明的**语义；描述里写成
   `"axes": {"1": [{"axis": "tp", "mode": "replicate", "unit": "head_dim"}]}`，字符串形式
   `["tp"]` 不变（既有描述零改动）。unit 与 `split.sizes` 一样解析参数名或整数字面量；
   **写了 unit 却没写 `mode: replicate` 是描述错误**（此前 unit 会被静默丢弃，等于描述说"按头切"、
   计划按元素切）。

**同一轮补上的拒绝清单（每条都有测试）**：

| 情况 | 之前 | 现在 |
|---|---|---|
| `unit` 无 `mode: replicate` | 静默丢弃 unit | `expand` 报描述错误，点名 slot 与 unit |
| refold 的最后一个轴不是 operand 的最后一轴（如 `[512,128] → [64,4,4,64]`） | 轴号照搬 | `UnmappableViewShard`（对所有 mode，`Divide` 同样拒绝） |
| 合轴（rank 减少）时复制单元落在被折叠的轴上 | 轴号照搬 | 拒绝（单元的"长度"不再对应输出轴的元素） |
| 复制权重落在 `linear`/`embedding`/`matmul` 的收缩轴上 | 有的分支静默接受 | `UnsupportedWeightLayout`（all_reduce 会把重叠部分算两遍） |
| 复制轴各 rank slab 长度不同（`kv=3, tp=4`） | 按 coord 0 的形状凑合 | `NonUniformSlabs` |
| 同一个 dim 两个 spec、mode 不一致 | 取**最后**一个 spec 的 mode | 取更弱的承诺（复制优先，同为复制取更粗的 unit），有测试 |

**修正一条写错的话**：§落地里曾说"digest 只哈希决策，不哈希 layout"。实际 `compute_digest` 的
preimage 是 `DigestInput { plan, decisions }`，**plan 里的 slot layout 是被哈希的**。真正成立的是另一半：
`ShardSpec::mode` 在 `Divide`（默认）时**不序列化**，所以既有计划的 wire form 与 digest 逐字节不变；
新增字段只在**真的声明了复制**的计划里改变 digest —— 那种计划本来就是另一套分片，digest 变了才是对的。

**落地了什么（本次提交，全部有测试）**：
`rustrain-parallel`：`ShardMode{Divide,Replicate{unit}}` 成为 `ShardSpec` 的字段（默认 `Divide`，
**序列化时省略**，所以既有 plan 的 wire form 不变），`local_shape` 与 `slab(global,dim,coord,degree,rank)`
走同一个函数（`local_shape` 走 `uniform_slab`：逐 coord 校验长度一致）；
`rustrain-plan`：`DeclaredAxis`（plan 的词汇）+ `instantiate` 从声明建 spec + 传播链路
（linear 列并行、view 的维度平移、二元算子的 dim 合并）都携带 mode；
`rustrain-model`：描述侧的 `AxisDecl`（untagged：字符串或对象）与 unit 解析；
`rustrain-cli`：加载器不再自己算 `coord*local`，改成问**声明它的那个 spec 的 mode** 要 `slab`
（同一个 dim 上可能有多个 spec，layout 级的 dim 查询会取错）—— 一个事实一个来源。
测试：`a_replicating_shard_hands_each_rank_the_units_it_needs`（512/4 → 每 rank 一个头、
4 单位 8 rank 的重叠、单位不整除被拒、严格路径不变）、`an_axis_may_declare_how_its_slabs_relate`
（两种写法 + unit 解析 + 有 unit 无 mode 被拒）、`a_replicating_slab_covers_the_span_the_consumer_needs`
（消费端边界：`kv=3,tp=4` 的 1/2/2/1 头跨度 + 各 (units, degree) 的性质测试）、
`a_replicating_axis_with_different_slab_lengths_has_no_local_shape`（`NonUniformSlabs`）、
`the_real_description_instantiates_and_propagates_on_the_acceptance_mesh` 所在的
`instantiate.rs` 里另加了 `tp=4` 下 k_proj 的 unit=256 与本地形状 = 一个整头。

**同一轮补上的那一步：reshape/view 的 unit 换算**。k_proj 的**权重**按 head 单位切一次就对，但**激活**
路径 `k = x @ w [seq,512]` → `reshape [seq,2,256]` 的映射原先只平移轴号，不知道"512 特征轴上的
unit=256"到了 `[2,256]` 上就是"头轴 unit=1"。现在 `derive` 多收两个形状（每个 operand 的 rank 与
shape 一起传），`carry_to_output_rank` 在 refold 时把最后一个轴的 unit 除以新内层轴的乘积
（`unit % Πinner == 0` 才可表达，否则**拒绝**——半个头不会被默默算出来）。测试
`a_refold_rescales_a_replicating_unit` 钉住这两面（换算 + 拒绝）。

**验收（宿主实测，`launch --sweep tp=N`，各含一次 world=1 基线）**。数值列是**当前**构建
（`main`，逐位与首次验收相同）；耗时列分两段给出，前半是 2026-09 首次验收、后半是本轮
（MoE 去同步 + 通信器暖机 + 分块流式加载之后）：

| | tp=2（2 进程） | tp=4（4 进程） | tp=8（8 进程） |
|---|---|---|---|
| logits vs world=1 `max\|diff\|` | **2.670e-5** | **3.338e-5**（界 1.298e-4） | **2.766e-5** |
| logits rel_L2 | — | 1.497e-6 | 1.416e-6 |
| 42 层 hidden 最坏 rel_L2 | — | 1.652e-6 | 1.838e-6 |
| argmax（8 行） | 全同 | 全同 | 全同 |
| 每 rank 权重 / 峰值 | 67.1 / 68.4 GiB | 34.6 / 35.8 GiB | 18.3 / 19.3 GiB |
| 每 rank 读到的字节（`distinct` 66.1 GiB） | 43.8 GiB | 32.7 GiB（189,637 runs，873 pairs / 712 tensors） | 27.1 GiB |
| 每 rank 加载墙钟 | 12.5 → **11.7–11.9 s** | 9.1 → **7.6–8.6 s** | 9.0 → **8.2–8.9 s** |
| 每 rank 前向墙钟 | 7.8 → **3.46–3.72 s** | 9.45–9.93 → **3.6–5.4 s** | 9.42–9.91 → **4.08–4.72 s** |
| 每 rank 写入的块数（8 MiB 级） | 9128 | 5004 | 2993 |
| 集合通信（每 rank） | 83 次 | 83 次（1 all_gather + 82 all_reduce） | 同左 |
| plan 步数 | 1411 | 1411 | 1411 |

**一次调试周期**（加载 + 前向，tp=4）从工作流起点的「每 rank 109 GiB、单次 sweep 约 23 分钟」
降到 **约 12 s**。

两次运行（`0a24e75` 与 `f719954`，后者改了 slab 规则与一批拒绝路径）数值**逐位相同**：
logits `max|diff|` 3.338e-5 / 2.766e-5、hidden 最坏 rel_L2 1.652e-6 / 1.838e-6、argmax 全同、
每 rank 读字节数不变 —— 对 `kv=2, tp=4/8` 新规则与旧规则给出同一批 slab，这正是它该有的样子。

`check --tp 2/4/8` 现在都是 exit 0；`check --tp 3` 仍精确点名不可整除的槽（严格路径没有被削弱）；
`kv=3, tp=4` 这类"各 rank slab 长度不同"的声明是 `NonUniformSlabs` 编译期错误（不再是静默少一个头）。

**顺带的一条证据（给 tp=2 那个 4 s 偏差）**：tp=4/tp=8 的 per-rank 墙钟**很紧**（9.4–9.9 s），
而 tp=2 上 rank 1 曾比 rank 0 慢 4 s——说明那不是"多进程都这样"，而是 **2-rank 这一档特有的**
（下一步就在 tp=2 上打步级时间戳）。

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

### D6.7 — 每 rank 的墙钟去了哪里（2026-09，本轮）

**为什么加**：tp=2 上出现过 rank 1 比 rank 0 慢 4 s（也出现过 rank 1 *更快* 0.8 s、以及慢 15 s），
而当时的遥测只有"forward 墙钟"一个数——没有东西能把差值归到通信、算子体还是加载上。按"排查时间要
压到极低"的同一逻辑，这个洞必须补上，否则每次偏差都要重新起一轮 profiling。

**加了什么**：`RunStats` 新增四项，`run` 的 per-rank JSON 与人类可读摘要都输出（纳秒存储，
避免浮点进入 `Eq` 派生的统计结构）：

| 字段 | 含义 |
|---|---|
| `collective_seconds` | 本 rank 花在集合通信后端里的时间总和 |
| `collective_seconds_by_kind` | 同上，按内在算子分（`all_gather` / `all_reduce` / …） |
| `first_collective_seconds` | **第一次**集合通信的耗时——communicator 是首次使用时才建的（`ncclCommInitRank` + id 文件轮询），晚到全组的 rank 全在这里付账 |
| `op_seconds` | 本 rank 花在插件 `execute` 里的时间总和 |

`wall_seconds - collective_seconds - op_seconds` 就是执行器自己的余量。四个数一列，
"两个 rank 的差别在哪一类工作里"就是直接读出来的，不需要 profiler。测试：
`crates/rustrain-runtime/tests/end_to_end.rs` 里钉住"没有集合通信时三项为 0、有集合通信时
`by_kind` 之和 == 总数、`first_collective <= 总数`"。

**历史时间线（同一台宿主，`--tokens` 同一串，tp=2，rank 墙钟）**：

| 构建 | rank 0 | rank 1 | 差值 | 备注 |
|---|---|---|---|---|
| `d6-run.log`（更早，1371 步 / 43 次通信） | 6.42 s | 5.64 s | **−0.78 s** | rank 1 更快 |
| `d6-run2.log`（1411 步 / 83 次通信） | 7.51 s | 22.95 s | **+15.4 s** | 最坏的一次 |
| `d6-run3.log`（1411 步 / 83 次通信） | 6.74 s | 10.73 s | **+3.99 s** | 就是"慢 4 s"那次观测 |
| 当前（连跑三次，1411 步 / 83 次通信） | 6.64 / 6.67 / 8.03 | 6.60 / 6.55 / 8.02 | **−0.04 / −0.12 / −0.01** | 对称 |

**机制（本轮用相位时间戳测出来，不是推断）**。每个 rank 的 JSON 现在带三个同一台机器可比的
`SystemTime`：`load_finished_unix` / `forward_started_unix` / `forward_finished_unix`。
两条 tp=2 实测（同一构建，连跑）：

| 运行 | rank | 加载完成（相对最早者） | 首次集合通信耗时 | 前向墙钟 | 前向**结束**（相对） |
|---|---|---|---|---|---|
| 1 | rank 0 | +0.42 s | 1.44 s | 7.95 s | **+8.38 s** |
| 1 | rank 1 | +0.00 s | 1.89 s | 8.39 s | **+8.39 s** |
| 2 | rank 0 | +1.12 s | 1.08 s | 6.86 s | **+7.97 s** |
| 2 | rank 1 | +0.00 s | 2.21 s | 7.97 s | **+7.98 s** |

两个结论是硬的：
1. **两个 rank 在同一瞬间结束前向**（差 10 ms 以内）——前向本身是逐 rank 对称的；
2. **每个 rank 在"首次集合通信"里等的时间，正好等于另一个 rank 的到达偏差**
   （0.45 ≈ 0.42、1.13 ≈ 1.12）——因为 `ncclCommInitRank` 是组内同步点：**先到的 rank 替全组
   付等待**。所以"per-rank 墙钟差"测的**不是算力差，而是前向前阶段的到达偏差**。
   `op_seconds` 三次运行 5.41–5.46 s、逐 rank 相同到 0.1%，也印证这一点。

**"慢 4 s"因此完全解释**：旧加载路径每个 rank 都读 ~110 GiB 全量张量（§D6.4/§D6.5 之前），
两个进程争同一个文件系统，谁先读完谁就先到首次通信的会合点，于是**先到者的墙钟多出对手的
全部滞后**。`d6-run.log` 里 rank 1 反而更快（−0.78 s）正是同一机制的反向排列。§D6.5 的
narrowed read（32.7/27.1 GiB）把到达偏差压到 0.4–1.1 s，差值随之消失。**没有找到也不需要
任何"rank 1 多做一份工作"的代码路径**：逐 rank 的 plan 步数、通信次数、通信字节、算子时间
全部相同。

**残留成本与修法（已实现）**：组建立（`ncclCommInitRank` 握手 + id 文件轮询）在 tp=2 上实测
1.1–2.6 s，全部落在"前向墙钟"里。修法：`CollectiveBackend::warm`（默认 no-op）+ 运行时的
`SharedBackend`（同一后端实例跨线程共享，加载路径不碰后端所以锁不竞争）+ CLI 在 **compile 之后、
加载之前**起一个暖机线程预热本 rank 计划里所有 degree>1 的组，加载完成后再 join——join 的等待本身
也记进 `warm_seconds` / `warm_finished_unix`，所以**没有任何等待被藏起来**。

效果（宿主实测，同一构建）：

| 配置 | 每 rank 加载 | 首次集合通信 | `op_seconds` | 前向墙钟 | 前向结束（同一次运行内） |
|---|---|---|---|---|---|
| tp=4（4 rank） | 9.3–9.5 s | 0.343–0.550 s | 6.84–6.88 s | 7.22–7.40 s | 一致（±10 ms） |
| tp=8（8 rank） | 9.0–10.2 s | 0.125–1.082 s | 6.00–6.16 s | 6.83–7.94 s | 一致 |
| tp=2（2 rank） | 12.8–13.6 s | 0.17–1.01 s | 5.43 / 5.89 s | 5.61–6.90 s | 一致（±10 ms） |

（tp=2 的加载比 tp=4/8 长是**读得多**：每 rank 43.8 GiB vs 32.7/27.1 GiB，不是暖机的代价。）
提交后在最终树上复验一次（`ee1ee8b`，tp=2 + tp=4）：`warm_seconds` 全为 **0.000**（暖机在加载窗口内
完成，正是设计意图），tp=2 墙钟 6.11/6.06（`first` 0.228/0.174、`op` 5.871/5.880、`other` +0.003）、
tp=4 墙钟 7.00–7.41（`first` 0.305–0.812 正好解释掉墙钟的全部散布、`op` 6.59–6.68、`rest`/`other`
逐 rank 毫秒级相同），四个 rank 的**前向结束时刻相同（+7.41 s）**。

**暖机改变了 `first_collective_seconds` 的含义，这条必须跟着改**：暖机后它不再包含 communicator
建立，剩下的是**第一个分发式集合通信的到达等待**——先到的 rank 仍要等其他人；`degree == 1` 的组会
被显式排除（本地拷贝不能代表世界的时序）。所以"前向墙钟差"读的仍然是这个字段的差，而"组建立"的
成本已经挪到加载窗口里。

**审查要的那条决定性证据（"是不是 rank 非对称的代码路径"）**：把每个 rank 的墙钟拆成
`first` / `rest`（`collective_seconds − first_collective_seconds`）/ `op` / `other`，tp=4 四个 rank：

```
rank0 wall 7.40  first 0.550  rest 0.005  op 6.845  other +0.0021
rank1 wall 7.23  first 0.343  rest 0.005  op 6.882  other +0.0021
rank2 wall 7.23  first 0.344  rest 0.004  op 6.881  other +0.0049
rank3 wall 7.22  first 0.346  rest 0.005  op 6.864  other +0.0021
```

tp=8 八个 rank 的 `rest` 是 0.698–0.710 s、`other` 是 +0.0017…+0.0032 s。**除 `first` 外每一项
逐 rank 相同到毫秒级；墙钟差恰好等于 `first` 的差**。这正是"不是算力/代码路径不对称，而是到达偏差"
的判据——审查指出这是原论证缺的那条腿，现在补上了。

### D6.8 — 前向的 6 秒里，5.1 秒是两个算子体（2026-09，本轮）

**能力**：`RUSTRAIN_STEP_TRACE=<n>`（opt-in）让执行器按**算子标签**聚合并打印最重的 n 个步骤到
stderr；`launch` 在请求了 trace 时把各 rank 的这几行转发出来（rank 的 stderr 平时被捕获，只在失败
时打印，所以成功的 rank 必须显式转发，否则 trace 正好被吞掉）。它不改变任何指标。

**实测（tp=4，`cuda.aten.f32`，同一段 token）**，`step trace` 的 top 行：

| 算子 | 调用 | S=8 总耗时 / 每次 | S=512 总耗时 / 每次 |
|---|---|---|---|
| **`moe_layer`** | 41 | **4.03 s / 98.2 ms** | **5.54 s / 135.0 ms** |
| **`gated_delta_rule`** | 30 | **1.08 s / 36.0 ms** | 1.21 s / 40.4 ms |
| `intrinsic.all_reduce` | 82 | 0.42 s / 5.1 ms | 0.28–0.84 s |
| `rope` | 22 | 0.063 s | 0.13 s |
| `linear` | 298 | 0.063 s / 0.21 ms | 0.058 s |
| `rmsnorm` | 108 | 0.055 s | 0.052 s |
| 其余 8 类 | — | 各 <0.05 s | 各 <0.05 s |
| **合计** | 1411 步 | **5.86–6.11 s** | **7.41–8.01 s** |

**读数**：`moe_layer` + `gated_delta_rule` = 全部前向时间的 **87%**；其余 1400 步加起来 <0.6 s。

**为什么**（不是算力，也不是带宽）：token 数从 8 涨到 512（**64×**），`moe_layer` 只从 98 ms 涨到
135 ms（1.37×）——说明成本几乎是**每次调用的固定开销**：ATen 参考体对**每个专家**做一次
`index_select` + 三次小 matmul + `index_add_`，256 个专家的循环就是 ~100 ms，与有多少 token 路由过去
无关。按每 rank 每层 768 MB 的专家权重算，135 ms 对应约 5.7 GB/s，远低于 HBM 带宽，同样证明它不在
搬数据。`gated_delta_rule` 的 36–40 ms 同源（分块递推的 Python/ATen 循环）。

**这是调试速度下一个最大的杠杆，但它是"换实现体"，属于你已保留裁定的范畴**：`moe_layer` 的
descriptor 文档自己写着"A grouped GEMM (`torch._grouped_mm`, or an upstream kernel) is the fast path
this body deliberately leaves for later"。方案是**加一个实现变体**（例如 `cuda.aten.f32.grouped`），
参考体原样保留当 oracle（`ops check` 一致性门禁继续用它对照），recipe 里选变体——
预计每次前向省 4–5 s（8 s → 3 s 量级）。代价是数值重验：S=512、tp=4 的验收已经跑过
（`max|diff| 3.815e-5` / 界 `2.041e-4` PASS），换实现体后要按同样的判据重跑。

### D6.9 — `moe_layer` 的 98 ms 里 92 ms 是同步（2026-09，本轮已修）

**根因**：ATen 的 moe 体在 `for e in 0..E { for k in 0..K }` 里调用 `at::nonzero(indices[:,k] == e)`。
`nonzero` 的**输出形状依赖数据**，所以每次调用都要同步一次设备：`E × K = 2048` 次同步/层，
2048 × ~45 µs ≈ **92 ms**，正好是这层 98 ms 的全部。

**修法**（`plugins/aten/src/ops_moe.cpp`，数值不变）：把"哪些行选了哪个专家"这件**数据事实**在循环外
一次性算好——`indices` 是 `[rows, K]`，把它拷到 host（一次）后逐元素扫出 `rows_by[e][k]`；
循环里只做原来的 `index_select`/`matmul`/`index_copy_`。行序仍是升序，选中张量与矩阵形状都不变，
所以结果**逐位不变**：验收 `max|diff|` 与修前完全一致（S=8 `3.338e-5`、S=512 `3.815e-5`）。
顺带补上了文档里写的"越界专家下标是硬错误"——原实现里越界下标只会静默匹配不到任何 `e` 而被丢掉。

**效果（tp=4，同一构建、同一段 token，`cuda.aten.f32`）**：

| | 修前 | 修后 |
|---|---|---|
| `moe_layer` | 41 × 98.2 ms = 4.03 s（S=8）/ 41 × 135 ms = 5.54 s（S=512） | 41 × **39–42 ms** = 1.6–1.7 s |
| 前向墙钟 | 6.03–6.75 s | **3.99–4.49 s** |
| 验收 | `3.338e-5` / `3.815e-5` PASS | **同值** PASS |

**两条新遥测**（都是这轮加的）：`collective_paths`（staged/direct）与 `step trace` 里的
**最慢单步 + 步号**。它们立刻给出另外两个大头：

| 步 | 项目 | 实测 | 说明 |
|---|---|---|---|
| step 1369（最后一步） | `intrinsic.all_gather` | **766–784 ms**，485 MiB/rank，`direct`，≈0.6–0.8 GB/s | 按外层块分块：512 行 × 4 rank = 512 次顺序的小 NCCL 调用，每次 ~1.5 ms。要修需要设备侧 permute（新原语/内核，需裁定）或改布局（让这一步不再需要 gather） |
| step 59（第一个集合通信） | `intrinsic.all_reduce` | 578–736 ms | 就是 §D6.7 的**到达偏差**，不是传输量 |
| 30 × `gated_delta_rule` | 算子体 | 37–57 ms/次 | 与 moe 同源的分块循环结构，下一项 |

**另一个必须记住的事实**：`rows` 是**声明的窗口**（`params.seq = 512`），不是实际 token 数
（8）。所以"8 个 token"的调试跑仍然让 MoE 处理 512 行 × K = 4096 个 (行, 专家) 对——
这也解释了为什么 S=8 与 S=512 的 `moe_layer` 耗时几乎相同。

### D6.10 — 加载到底卡在哪：不是文件系统，是拷贝管线（2026-09，本轮）

**先量天花板**（同一台宿主、同一份 checkpoint、16 线程 pread 4 MiB 块）：

| 读法 | 吞吐 |
|---|---|
| 单进程 16 线程，8 GiB | **55.5 GB/s** |
| 四进程各 16 线程并行，各 8 GiB | 四路 **27.3–29.6 GB/s**（每进程） |
| 单进程 16 线程，31.7 GiB（就是一个 rank 要读的量） | **29.7 GB/s**（1.15 s 读完） |

而 tp=4 的每 rank 加载墙钟是 **9.0–10.6 s**、读 32.7 GiB ⇒ 有效 **3.6 GB/s**，比挂载点给同
一进程的能力低 **8×**。所以 §D6.5 之后剩下的时间**不是 I/O**。

**是拷贝管线**（新增 `write_wait_seconds` 之后可以直接读出来）：写者自己的跨度
`write + write_wait` 就是整个加载墙钟的 **96–97%**（10.81 / 11.20 s），其中
`write = 7.3–8.6 s`（34.6 GiB 的 f32 权重 → 设备，**4.3 GiB/s**），`wait` 只有 1.7–2.3 s
——生产者（16 个 worker：read 30–36 CPU 秒、fill 54–58 CPU 秒）一直有富余。

**试过并否掉的修法（诚实记录）**：pinned 暂存 + `cuMemcpyHtoDAsync` 的环形缓冲（4 个 buffer、
事件回收、写合并内存）。原语本身很快——单独测得宿主 memcpy 进 pinned 13.6 GB/s、异步 H2D
52 GB/s，而 pageable 的 `cuMemcpyHtoD` 只有 ~6 GB/s——但**整机加载墙钟没有变化**（9.0–10.6 s
对 9.3–11.2 s），写者的时间仍然是 6.8–8.4 s。原因是这条管线受**宿主内存带宽**限制：每 rank 一次
加载要搬 32.7 GiB（读）+ ~34.6 GiB（写进 f32 缓冲）+ 34.6 GiB（读出来拷贝）+ 34.6 GiB（进设备）
≈ 136 GiB 的宿主流量，四个 rank 加 64 个线程一起抢。多出来的 pinned/stream/event 复杂度换不来
时间，已回退（保留 `write_wait_seconds` 这个判据）。

**杠杆一（已实现）：分块流式**。每个成员不再物化成一整个 `Vec<f32>`，而是按 8 MiB 块
（对外层轴切片，`Cuts::fill_chunk_into`）读→展宽→写设备（新增 `Executor::write_f32_at` 做
带偏移的部分写），块缓冲在 worker 与写者之间用一个有界池复用。实测（tp=4，同一段 token）：

| | 修前 | 修后 |
|---|---|---|
| 每 rank 加载墙钟 | 9.0–11.2 s | **7.7–8.2 s** |
| `fill_cpu_seconds`（每 rank） | 54–58 s | **39–42 s** |
| `write_wait_seconds`（写者等待） | 1.7–2.3 s | **0.74–1.1 s** |
| `write_seconds`（设备拷贝） | 7.3–8.6 s | 6.8–7.3 s |
| 验收 | — | `max|diff| 3.338e-5` **同值** PASS |

**剩下那 5 GB/s 的拷贝：已定案——是宿主内存读带宽，不是代码**。用 `nsys` 逐进程读出真实速率
（同一份 checkpoint、同一构建）：

| 进程 | H2D 字节 | 设备侧拷贝时间 | 速率 |
|---|---|---|---|
| baseline world=1（单进程、单卡） | 132.9 GiB | 14.83 s | **9.62 GB/s** |
| tp=4 的四个 rank（每 rank） | 34.6 GiB | 6.2–6.6 s | **5.7–6.0 GB/s** |

而"独立探针"能到 15–17 GB/s 的原因也找到了：**它的源只有 64 MiB，常驻 L3**。同一探针把源放大到
真实规模后：256 MiB → 11.5 GB/s、1 GiB → 9.8 GB/s —— 与 loader 单进程的 9.6 GB/s 吻合。
四 rank 并行时每 rank 降到 5.8 GB/s，是因为 4 个 rank 的 34.6 GiB 都要从 DRAM 拉，而同一时间它们
各自的 16 个 worker 还在写同样多的 f32 块。

被这批实验**排除**的解释（每条都有对照数据）：平台 PCIe 上限（四卡同时 15–17 GB/s/卡）、文件系统
（同进程同量 29.7 GB/s）、页缓存读取争用（不影响拷贝）、进程间争用（加载中另起探针仍 9.8 GB/s）、
NUMA 亲和（探针两节点都快；把 loader 绑到 node 0 反而更慢，因为 4 卡的 64 线程全挤一个节点）、
目的地址分散（873 个分配逐个写仍 16.6 GB/s）、块大小（8 与 32 MiB 无差别）、块缓冲池大小
（36 与 16 个缓冲无差别）。

**结论：f32 管线下加载已经到宿主内存带宽地板。** 唯一能再砍一半的动作是**少搬字节**——
即下面的 bf16 变体。（另一个理论上的方向是 mmap 直接读 checkpoint 少一次 32.7 GiB 的宿主拷贝，
但在并行文件系统上 mmap 的风险高于收益，暂不做。）

**本轮的另一项读数**：`RUSTRAIN_LOAD_WORKERS` 实测 16 最优（8 → 墙钟 8.5–9.1 s，12 → 8.2–8.4，
16 → 7.5–8.2，24 → 8.4–8.8），所以 worker 数不是拷贝慢的原因，16 保持。

**杠杆二（已落地）**：bf16 变体——直接写 bf16 权重，字节数减半；判据随之重定，见 D6.11。

### D6.11 — bf16 的判据：不是 20×，是判据本身错了（2026-09，本轮）

**报出的现象**：bf16 默认落地后，对照读出"我们前层损失 6.4e-2、HF 自身 bf16-vs-f32 只有 2.8e-3"，
像是差 20×；同时 tp=4 验收在 bf16 下 FAIL（`max|diff| 1.219` / 界 `1.306e-4`）。

**根因两条，都是判据，不是实现**（**第一次写这段时的解释是错的，独立审查逐位复算后推翻，下面是更正版**）：

1. **两个数根本不是同一条统计量。** 旧判据是 `per_layer = max{ |Δmean|/|mean|, |Δstd|/std,
   |Δmax|/max }`，取三者里最大的那个。用它重算 row 1（参考 = `hf-ref42-f32`）：
   **我们 6.3773e-2，主导项是 `mean`**；**HF 2.8113e-3，主导项是 `max`**（HF 同行的 `|Δstd|/std`
   是 9.111e-5，比它小 31×）。两者相除 22.7 ≈ 那个"20×"。也就是说这个 `max` 聚合器在两侧挑了
   **不同的统计量**，而 `mean` 的除数（残差流的均值）本身近零 —— 它是"噪声 ÷ 噪声"。
   **这不是"一阶 vs 二阶"，是同一个聚合器里两个量纲不同的分母。**（第一次我写成"我们是一阶 rel_L2、
   HF 是二阶 std 差"：两个都不对，我们的 row 1 rel_L2 是 1.339e-2，HF 的 2.8e-3 是 `max` 项。）
   修法不变，而且正是针对这条根因：`|Δmean|`、`|Δmax|` 改除以该层的 `max|x|`（不会消失的量）。
2. **"std 看不见 bf16"也是我第一次写错的。** std **不是**二阶盲：把一行 f32 逐元素 RNE 舍入到 bf16，
   实测 std 的相对变化是 row 1 的 7.27e-5 到 42 行里的最大 2.76e-3（中位 2.11e-4），而同一舍入的
   逐元素 rel_L2 是 1.61e-3 —— 只看得到 8–22×，不是 3–4 个数量级。第一次引的 `(2^-8)²/24` 本身也算错了
   （= 6.36e-7，不是 2.5e-6），而且拿**绝对**的 `1.67e-6` 去对**相对**的预测值。真正成立的说法是：
   **在头几层这条统计量被量化本身饱和了** —— HF row 1 的 `|Δstd|/std = 9.11e-5` 只有纯舍入地板
   `7.27e-5` 的 1.25×，所以它在早期分辨不出额外的漂移；能分辨的时候（后段 2.1e-2）已经无关紧要。
3. **拿 f32 的界判 bf16。** sweep 的 `bound_relative` 是一个与 dtype 无关的常量 `1e-5`，它是按 f32 的
   重结合噪声定的；bf16 单次求和就舍入到 `2^-8 ≈ 3.9e-3`。

**决定性证据**：补做 HF 的 bf16 逐元素参考（`hf_qwen36_reference.py dump --dtype bf16`，宿主 12 s、
42 个 hidden state），四份 dump 逐层比（同一组 8 个 probe token、world=1）：

| 比法 | logits `max\|diff\|/max\|ref\|` | 逐元素 rel_L2 最大 |
|---|---|---|
| 我们 f32 vs HF f32 | **2.000e-3** | **1.694e-3** |
| 我们 bf16 vs HF bf16 | **8.812e-2** | **1.317e-1** |
| HF bf16 vs HF f32 | 1.523e-1 | 1.634e-1 |
| 我们 bf16 vs HF f32 | 1.447e-1 | 1.688e-1 |

**读数**：

- 我们 bf16 与 HF bf16 的差（8.8e-2）**比 HF 自己 bf16 与 f32 的差（1.52e-1）还小**；两份 bf16 的
  embedding 行逐位相同，两边 dump 的 bf16 元素也都是精确的 bf16（f32 位型的低 16 bit 全零，已验证）。
- 逐层漂移比（我们 bf16 相对自己 f32 ÷ HF bf16 相对自己 f32，当前构建）：层 0–5 是 **3.24 / 2.03 /
  2.27 / 2.89 / 2.26 / 1.57×**，**层 6–9（row 7–10）是 2.18 / 1.39 / 1.25 / 1.13×**，
  层 11–39 是 **0.96–1.16×**。

**结论：撤回"20×"** —— 它是同一个聚合器里 `mean` 与 `max` 两条不同统计量相除的结果。

**替代刻画（"前几层 2–3×"）也不够稳，两条独立理由**：

- **口径依赖。** 换一条一阶口径就变：`max|Δ|/max|x|` 下层 6–9（row 7–10）是 **12.3 / 6.7 / 6.7 /
  4.9×**，而同一量在 41 行里有 15 行 <1（我们更好）；1−cos 是**平方**量纲，早期比值约等于 rel_L2 比值的
  平方（3.24² = 10.5）。所以只有 rel_L2 支持"2–3×"，`max` 口径给出的早期差距大得多。
- **构建依赖。** 修 f32 统计量之前的 bf16 构建（`rustrain-bf16.npz`/`bf16b`，两者逐位相同）在
  row 1–6 的比值是 **1.61 / 1.06 / 1.11 / 1.10 / 0.99 / 0.97**，后段 0.66–0.96（**比 HF 还好**）——
  却离 HF bf16 的 logits **更远**（`max|diff|/max|ref|` **1.683e-1** vs 当前 **8.812e-2**，自己的
  bf16-vs-f32 漂移也是 1.531e-1 vs 1.441e-1）。**早期这个比值不是质量指标**：漂移小的那份反而更偏。

**所以正确的记录是**：当前构建的 bf16 在 logits 上比修统计量之前**近一倍**，且总漂移更小；"前几层 2–3×"
是**当前构建 + rel_L2 口径**下的观察，不是框架的性质，换口径到 12×、换构建到 1.0×。

**剩下那个观察（未定案，不阻塞）**：当前构建在层 0–5 的 bf16 漂移比 HF 大（rel_L2 口径 1.57–3.24×，
`max` 口径更大）。已排除：**GDN 精度**——`plugins/aten/src/ops_recurrent.cpp:233-239` 与 HF
`modeling_qwen3_5_moe.py:261,333` 做法相同（q/k/v/g/beta 全部上抬 f32、state f32、出口转回 io dtype）。
“两侧漂移互不相关”只在 row 1 成立（`cos(e1,e2)=0.03`），到第 30 层已升到 0.75 —— 也是这条"独立性"
不能外推的证据。候选解释仍是"一个算子 = 一个 plan 节点 = 一个 bf16 slot"带来的每算子边界一次舍入
（同类量在 HF 的融合核里只舍一次）。**下一步的定案实验**：`bisect0-ours.npz`（48 个 slot，含
`layers.0.{h1,attn,h2,h3,moe,y}`）与 `bisect0-hf.npz`（同名张量）已经存在，按 stage 逐段比 bf16/bf16
与 f32/f32，就能定位是哪个算子边界多了一次舍入；再叠一次 512 token、每构建 ≥3 次的四方 dump 把
单次实现的比值变成集合量。

**判据修正（三处，全部落地）**：

1. `run` 的 sidecar 增加机器可读 `dtype` 字段（`crates/rustrain-cli/src/run.rs`），散文 `precision`
   保留给人看——判据脚本读的是 token，不是那句话。
2. `scripts/hf_qwen36_reference.py compare`：容差按 dtype **对**取（f32/f32 = 1%、bf16/bf16 = 25%），
   **混 dtype 对直接拒绝**（`--allow-dtype-mismatch` 才报数，且报出来的数就是 dtype 自身的展宽）；
   隐藏态统计量不再除以近零的 `mean`（`|Δmean|`、`|Δmax|` 除以该层 `max|x|`，`|Δstd|` 除以参考 `std`
   ——旧写法在**我们的 f32 vs HF f32** 上都报出 2.6e-2 的假超差）；两侧都带 `hidden_values` 时再加一条
   逐元素 rel_L2。
3. sweep 的 `bound_relative` 变成 dtype 相关：f32 = `1e-5`、bf16 = `3e-1`；报告与首行都写明 dtype 与所用界，
   并且加一个机器可读的 `gating` 字段（f32 为真、bf16 为假）+ `gating_note` —— bf16 的 `pass` 是"落在
   dtype 自身噪声内"，不能被读成分片等价的证明。
4. 两条**静默**路径补上：sidecar 缺失/无声时不再退回 f32 的 1% 界（那是"合法的 bf16/bf16 对报 FAIL"的
   来源），改为与混 dtype 一样**拒绝**（同样有 `--allow-dtype-mismatch` 出口）；两侧 `hidden_values`
   形状不一致时不再静默跳过逐元素检查，而是记成失败并说明"这条检查没有跑"（`conformance.rs` 自己的规则：
   skip 必须写明缺什么）。
5. 新增测试钉住新事实：`agreement_bound_relative` 的映射、sidecar 的 `dtype` token、sweep 报告里的
   `dtype`/`bound_relative`/`gating`。同一事实散落在 `main.rs`/`load.rs` 里的两份**旧判据散文**（"1% 容差
   吸收 HF 的 bf16 舍入"）已删除，改指 D6.11。

**宿主实测（本轮，`launch --sweep tp=2;tp=4;tp=8`，S=8，bf16 基线 `max|logits|` 1.306e1 / f32 1.298e1）**：

| dtype | tp=2 | tp=4 | tp=8 | 界 | 结论 |
|---|---|---|---|---|---|
| bf16 | 9.63e-2 | 9.33e-2 | 1.03e-1 | 3.92e0（3e-1 × 13.06） | PASS，但 `gating: false`（dtype 噪声水位） |
| **f32** | **2.670e-5** | **3.338e-5** | **2.766e-5** | 1.298e-4（1e-5 × 12.98） | **PASS，`gating: true`（这才是分片等价性的门禁）** |

**结论：分片等价性的门禁跑 `--dtype f32`** —— bf16 的重结合噪声（1e-1）与"掉了一个 partial"这类 bug
同量级，界没有分辨力；bf16 的门禁是与**同 dtype** 的 HF 参考对照（8.8e-2，落在 dtype 自身噪声内）。

**顺带记录（本轮 step trace，tp=4、bf16、S=8，1411 步 / 4.74 s）**：`moe_layer` 41 次 × **59.3 ms** =
2.43 s、`gated_delta_rule` 30 次 × **51.6 ms** = 1.55 s —— 两个算子占前向的 **84%**，其余 1340 步合计
<0.8 s，且两者都是**与 token 数几乎无关的固定开销**（S 从 8 到 512，`moe_layer` 只从 98 ms 涨到 135 ms）。
即：接下来最大的调试速度杠杆是这两处的实现体（grouped GEMM / 分块并行递推）。

## 待解决

**① EP 的 dispatch/combine 落地方式待用户裁定**（改契约面，不擅自决定）：
三条路线的性能/灵活性/正确性后果、宿主实测数字与源码位置都在
`docs/design/ep-dispatch-combine.md`。裁定前 `check --ep >1` 继续以明确原因拒绝运行
（不是静默少通信）。

**② bf16 判据已定案（D6.11），剩一个可选的精度观察**：当前构建在层 0–5 的 bf16 漂移比 HF 大
（rel_L2 口径 1.57–3.24×，`max` 口径到 12.3×；且该比值随构建从 1.0× 变到 3.2×，不是质量指标）。
定案实验已备好：用现成的 `bisect0-{ours,hf}.npz` 逐 stage 比，定位哪个算子边界多了一次 bf16 舍入。
要收窄的杠杆是**按 slot 选 dtype**（残差流与归一化统计量留 f32、大 GEMM 走 bf16）——"精度是配置"的范畴，
不擅自改。

**③ 两个算子体的快路径待排期**（换实现体 = T2，参考体保留当 oracle）：`moe_layer` 的 grouped GEMM
（宿主实测 `torch._grouped_mm` 0.21 ms vs 现行循环）与 `gated_delta_rule` 的分块并行递推；
这是本项目目前最大的调试速度杠杆（占前向 84%）。

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
- [x] **D5 — 前向数值对齐 HuggingFace** —— 证据：宿主上 40 层真前向（单卡 f32，前向 6.6 s，峰值 134.2 GiB）
  与 HF 参考对比。**逐元素**（8 个 probe 行 × 42 层 hidden，双方都是 f32）：
  **最坏 rel_L2 1.69e-3、最坏 rel_max 5.77e-3**，没有一行超过 1e-2；`logits` 相对差 **2.0e-3**（`compare` 判 `[ok]`，
  max_abs_diff 0.026 / 13.0）；8 行 argmax 全部与参考一致。对 spec runbook 的 **bf16** 参考：
  `logits` 相对差 1.52e-1 —— 与 **HF 自己 bf16 vs f32 的差（1.52e-1）逐位相同**，即候选落在参考自身的精度散布内。
  到达这一步修掉的四个真 bug（都有反证过的回归测试）：`rope` 位置轴（`6ec7e93`）、view 族别名生命周期
  （`0369cd0`）、原地集合通信输入的生存期、以及 **trunk `rmsnorm` 漏声明 `eps`**（`3fb3edc`，
  HF 是 1e-6、缺省 1e-5；embedding 方差只有 ~9e-5，第一个算子就差 4.6%）。
  **判据的诚实说明**：spec 原文的"每层 mean/std/max 相对差 < 1%"里，`mean` 是近零量（|Δmean|/std ≤ 3e-3），
  用它做相对比较只会放大噪声；真正的对齐判据应是逐元素 rel_L2 + logits 相对差，两条都过了。


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
