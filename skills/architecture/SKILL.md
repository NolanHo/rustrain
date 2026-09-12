---
name: rustrain-architecture
description: rustrain 的架构操作规则。改动 crate 边界、ABI、plan IR、算子注册、切分、显存策略或模型描述之前必须读。包含边界契约、不变式、三类变更的 checklist、验证门禁与禁止模式。
---

# rustrain 架构规则

**读这份文件的时机**：任何改动 crate 边界、ABI、plan IR、算子注册表、切分规则、显存策略、recipe 或
模型描述的工作之前。描述"架构是什么"的是 `docs/architecture.md`；这份文件只讲"怎么在这个架构上工作"。

---

## 1. 边界契约（最高优先级）

**T1 / T2 自由，T3 重编框架。**

| 层 | 变什么 | 代价 |
|---|---|---|
| T1 实现体 | 同一契约的不同代码：SIMD / CUDA / Tilelang / CUTLASS / fp8 变体 / 融合或拆开 | 换 `.so` + 改 recipe |
| T2 声明契约 | arity、形状规则、dtype、numerics 与量化方案、expansion、collectives、反向接线 | 换 `.so` + 换 plan |
| T3 数学形态 | 框架没见过的**规则种类** | 重编框架 |

**把代码放对位置的判据**：

- 随实现变化的东西 → **插件**（`.so`）。
- 描述契约的东西 → **描述符里的数据**。
- **模型结构**（哪些算子、什么顺序、怎么连）→ **模型描述里的数据**（子图模板 + 重复）。
  插件只提供**原语**，不提供结构；`expansion` 只承载"融合 ↔ 分解"的实现语义。
- 框架必须**自己判断**的东西 → 框架代码。**新增一种"判断"就是 T3**，应该罕见，而且应该被注意到。

**为什么值得**：Kernel 正确性是 **Kernel 的责任**。框架只欠三件事 —— 拿到实现、知道契约、
验证它和别人算的是同一件事。换来的是随时装卸 kernel 做对照，这比编译期检查对研究更有价值。

**因此明确放弃**：类型化与编译期的接口检查。错的插件是**运行时**错误，代价由一致性门禁承担。
**门禁不是"顺手做的检查"，它是这条边界的承重结构。**

---

## 2. 不变式（违反即架构破坏）

- **I-1 · 核心不含计算后端。** `rustrain-{abi,ops,parallel,plan,runtime}` 的依赖闭包里不得出现
  tch / libtorch / cuda。这是"核心能在无 GPU 机器上跑完整测试"的唯一依据，也是无 GPU check 的前提。
  验证：`cargo tree -p <crate> -e normal | grep -iE 'tch|libtorch|cuda'` 必须为空。
- **I-2 · 算子只能通过 ABI 注册。** 框架不得静态引用任何具体实现；`rustrain-kernels` 是**插件**，不是框架的一部分。
- **I-3 · 切分是 plan 里的数据。** 通信由编译期插入，不得手写在模型或训练循环里。
- **I-4 · 算子实现不得自带策略。** 显存策略、精度方案、切分规则都不许写死在 kernel 里 ——
  它们来自 recipe 与描述符。
- **I-5 · 规则不得按算子名或张量名查框架侧的表。** `match op { "linear" => ... }` 形式的规则表把 T2
  泄漏成 T3：加一个规则形态框架已经完全认识的新算子，却要重编框架。规则要作为描述符里的**声明**
  （`{kind, 参数}`）。**现状违反此条**（`shard::rule_for`），是重构要修的第一件事。
  切分轴与"从 checkpoint 取哪一块"是同一条事实，住在**参数映射**里。
- **I-6 · plan 里没有 topology 这个概念，只有它的结果。** plan 携带具体 layout（含度数）、axis id、
  形状与节点集合，以及 digest 里的 topology 指纹；**不得携带可遍历的 mesh / rank 列表**。
  拓扑是编译输入与运行输入。推论：**描述是可移植产物，plan 是拓扑相关的产品**，
  所以 checkpoint 映射挂在描述上（全局），不挂在 plan 上（局部）。
- **I-7 · kernel 与拓扑无关。** kernel 不持有 mesh、不按度数分支；它只有三样：本地张量、
  描述符里声明的轴 id、执行期由 runtime 绑定的组句柄。**禁止给 kernel 传独立的度数标量** ——
  那是同一事实的第二个来源，一定会和句柄漂移。

---

## 3. 禁止模式（都是旧代码的真实病根）

```
❌ 用环境变量选择实现             → 进 recipe
❌ 静默 fallback / 静默跳过       → 显式声明，或报错并写明理由
❌ 声明了却没人读的字段           → 见 §5 的"死钩子"
❌ 从张量形状反推量化方案         → 量化方案是声明的数据
❌ 融合算子不声明 expansion       → expansion 是校验、切分与反向推导的唯一依据
❌ 算子内自己 malloc / 建 stream / 建 comm → 通过 rs_services 申请
❌ 手写 all_reduce / detach 技巧   → 由切分传播插入
❌ 给 kernel 传 tp_size 之类的度数 → 度数从 communicator 读（I-7）
❌ planner 规划融合体本体          → 永远规划 primitive expansion，融合是解析期替换（§4.1）
❌ plan 里塞 mesh / rank 列表      → 只放结果与指纹（I-6）
❌ 插件在 init() 之前碰设备        → 否则无 GPU check 变成真执行
❌ 描述里重复 config 的数值        → 引用参数名，不复制数值
❌ 模型描述只支持"纯列表"或"纯派生"一种形式 → 两种都必须支持（Qwen 与 GLM5 各用一种）
```

---

## 4. 三类变更的 checklist

### 4.1 新增一个算子**实现**（T1，最常见）

1. 新建插件 crate，`extern "C"` 导出**唯一**符号 `rustrain_plugin_v1`（`PluginBuilder` 提供）。
2. 填写 `rs_op_desc`：`requires` / `numerics` / `memory` / `backward` / `collectives`；
   融合的必须填 `expansion`；**`infer` 必须与 plan 声明的形状一致**（编译器会逐节点比对）。
3. **融合体声明的 `collectives` 必须等于 planner 在它的 expansion 上会插入的集合。**
   不等 → 这次融合被拒绝，回落到分解形式。planner 永远规划 expansion，不规划融合体。
4. 装进运行时可发现的路径，在 recipe 里选中它。
5. 跑 `rustrain ops check`，四项全过。
6. **不需要改框架任何一行，也不需要重编框架。** 如果发现需要 —— 那说明规则按名字查表了，回去看 I-5。

### 4.2 新增一个**原语**（T2/T3）

1. 先问：能不能用现有的 kind / 属性表达？**能就别加原语。** 词表越小越好。
2. 加进 `docs/design/op-vocabulary.md` 的词表（**词表的唯一权威**；spec §2.4 只是指针）。
3. 在 reference provider 实现，含 `infer` / `memory` / `doc`（把不受数学约束的选择写进 doc）。
4. 补切分规则 —— **作为描述符里的声明**，不是框架里的 `match`（I-5）。
5. 补一致性门禁的 case。
6. 补 VJP 规则（若可导）。

**判定**：如果新原语需要框架新增一种**规则种类**，那是 T3，重编框架是预期的；如果只是又一个同形态的算子，
那不该动框架 —— 动了就是 I-5 违规。

### 4.3 改 ABI（T3，最贵）

1. `include/rustrain_op.h` 与 `src/ffi.rs` **必须同时改**，且字段**只能追加、不得重排**。
2. 更新两端钉死的尺寸断言（Rust 侧 `ffi::tests::layout` + C fixture 的 `_Static_assert`）。
3. 提高 `RUSTRAIN_ABI_VERSION`，并确认装载期的版本协商会拒绝旧插件而不是猜测兼容。
4. 更新 `docs/architecture.md` 的边界契约一节。

### 4.4 改**模型描述**格式（T2/T3 边界）

1. 判据有两个：**（a）它能不能把 `Qwen/Qwen3.6-35B-A3B`（第一个验证样本）完整表达成数据**；
   **（b）加第二个模型时要不要改语言** —— 要改就是设计不够。模型事实见 `docs/design/qwen36-5d-example.md`。
2. 每次改动都要在**无 GPU** 的机器上跑 L1 + L2（`rustrain check`），两条都要。
3. 结构进描述、实现进 recipe —— 越界就是把两件事捆在一起。

---

## 5. 过程纪律

- **接口没有消费者就不进设计。** 这个代码库撞过*五次*死钩子：`CheckpointPolicy`（每个节点都声明，
  从没被读过）、`RsMemReq.save_for_backward_bytes`（在 ABI 里，从没被查询）、`SlotKind::State`、
  `StreamPolicy::Side`、以及 `MemoryPlan` 的偏移（算了，执行器却逐槽分配）。
  旧代码里还有第六种形态：`config.json` 里解析了却从不读的字段（`full_attention_interval`、`mrope_*` …）——
  **它比没有更糟，因为它看起来像支持。**
  **写下一个字段时，同一步里就要写下谁读它。**
- **凡是对错要靠事实判断的地方，必须有两个实现 + 一个数值基准。** 单一实现可以自洽且错误：
  一份切分规则把权重约定写反了，**全部测试通过**，直到第二个实现出现才暴露。
- **"图即数据"没有先例可抄**（vLLM / SGLang / MaxText / torchtitan 的图都是 `forward()`）。
  写错的图**编译不会报错**。所以每个模型的描述必须过 L1/L2，并且至少一次与参考实现（HF logits/loss）对齐 ——
  **check 是开发内环，不是正确性证明。**
- **`git stash` / `git add -A` 是全局写操作，不得与其他写者并发。** 实测撞过：stash 收走了子代理
  正在写的半成品，`add -A` 把它未完成的工作提交进了文档 commit。**一个 crate 同时只有一个写者。**
- **commit 只装一件事。** 文档 commit 里不该有内核代码。
- **降级必须可追溯。** 任何 relaxation / fallback 都要记录原因，并进入 digest。

---

## 6. 验证门禁

改动完成前必须全绿：

```sh
export PATH=/root/.cargo/bin:$PATH
cargo test --workspace                    # 全部通过
cargo clippy --workspace --all-targets    # 零 warning
cargo run -q -p rustrain-cli -- ops check # exit 0；skip 必须写明理由
```

涉及 GPU 或宿主环境的改动，还要在验证宿主上跑一遍：
`root@47.94.214.197:26002`（8× L20X，CUDA 13，Rust 1.98.1）。

**`ops check` 的 skip 不等于 pass。** 每条 skip 必须能回答"缺什么、什么时候能补上"。门禁是 §1 那条边界
的承重结构，让 skip 静默通过等于把承重墙拆了。

---

## 7. 未决事项（不要擅自实现）

`docs/architecture.md` §8 列了待定决定。**D2 已定**：结构是数据（子图模板 + 重复），插件只提供原语，
`expansion` 只承载实现语义。

**下一个设计产物是 D8（描述文件的语法与展开语义）。** 在它定案之前不要写"模型描述 → plan"的代码。
但 §1 / §2 的规则**已经生效**：切分不是算子、度数作为编译输入、推导只兑现声明不猜声明、
planner 只规划 expansion、kernel 与拓扑无关、plan 不携带 topology 对象。

用户已明确的原则：**减少自动推导 —— 布局由模型显式声明，框架只做校验与插通信。**
