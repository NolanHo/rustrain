---
name: rustrain-architecture
description: rustrain 的架构操作规则。改动 crate 边界、ABI、plan IR、算子注册、切分或显存策略之前必须读。包含边界契约、不变式、三类变更的 checklist、验证门禁与禁止模式。
---

# rustrain 架构规则

**读这份文件的时机**：任何改动 crate 边界、ABI、plan IR、算子注册表、切分规则、显存策略或 recipe
的工作之前。描述"架构是什么"的是 `docs/architecture.md`；这份文件只讲"怎么在这个架构上工作"。

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
- 框架必须**自己判断**的东西 → 框架代码。**新增一种"判断"就是 T3**，应该罕见，而且应该被注意到。

**为什么值得**：Kernel 正确性是 **Kernel 的责任**。框架只欠三件事 —— 拿到实现、知道契约、验证它和别人算的是同一件事。
换来的是随时装卸 kernel 做对照，这比编译期检查对研究更有价值。

**因此明确放弃**：类型化与编译期的接口检查。错的插件是**运行时**错误，代价由一致性门禁承担。
**门禁不是"顺手做的检查"，它是这条边界的承重结构。**

---

## 2. 不变式（违反即架构破坏）

- **I-1 · 核心不含计算后端。** `rustrain-{abi,ops,parallel,plan,runtime}` 的依赖闭包里不得出现
  tch / libtorch / cuda。这是"核心能在无 GPU 机器上跑完整测试"的唯一依据。
  验证：`cargo tree -p <crate> -e normal | grep -iE 'tch|libtorch|cuda'` 必须为空。
- **I-2 · 算子只能通过 ABI 注册。** 框架不得静态引用任何具体实现；`rustrain-kernels` 是**插件**，不是框架的一部分。
- **I-3 · 切分是 plan 里的数据。** 通信由编译期插入，不得手写在模型或训练循环里。
- **I-4 · 算子实现不得自带策略。** 显存策略、精度方案、切分规则都不许写死在 kernel 里 ——
  它们来自 recipe 与描述符。
- **I-5 · 规则不得按算子名查框架侧的表。** `match op { "linear" => ... }` 形式的规则表把 T2 泄漏成 T3：
  加一个规则形态框架已经完全认识的新算子，却要重编框架。规则要作为描述符里的**声明**（`{kind, 参数}`）。
  **现状违反此条**（`shard::rule_for`），是重构要修的第一件事。

---

## 3. 禁止模式（都是旧代码的真实病根）

```
❌ 用环境变量选择实现            → 进 recipe
❌ 静默 fallback / 静默跳过      → 显式声明，或报错并写明理由
❌ 声明了却没人读的字段          → 见 §5 的"死钩子"
❌ 从张量形状反推量化方案        → 量化方案是声明的数据
❌ 融合算子不声明 expansion      → expansion 是一致性校验与反向推导的唯一依据
❌ 算子内自己 malloc / 建 stream / 建 comm → 通过 rs_services 申请
❌ 手写 all_reduce / detach 技巧  → 由切分传播插入
```

---

## 4. 三类变更的 checklist

### 4.1 新增一个算子**实现**（T1，最常见）

1. 新建插件 crate，`extern "C"` 导出**唯一**符号 `rustrain_plugin_v1`（`PluginBuilder` 提供）。
2. 填写 `rs_op_desc`：`requires` / `numerics` / `memory` / `backward` / `collectives`；
   融合的必须填 `expansion`；**`infer` 必须与 plan 声明的形状一致**（编译器会逐节点比对）。
3. 装进运行时可发现的路径，在 recipe 里选中它。
4. 跑 `rustrain ops check`，四项全过。
5. **不需要改框架任何一行，也不需要重编框架。** 如果发现需要 —— 那说明规则按名字查表了，回去看 I-5。

### 4.2 新增一个**原语**（T2/T3）

1. 先问：能不能用现有的 kind / 属性表达？**能就别加原语。** 词表越小越好。
2. 加进 `docs/architecture.md` 的词表与 spec §2.4。
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

---

## 5. 过程纪律

- **接口没有消费者就不进设计。** 这个代码库撞过*五次*死钩子：`CheckpointPolicy`（每个节点都声明，
  从没被读过）、`RsMemReq.save_for_backward_bytes`（在 ABI 里，从没被查询）、`SlotKind::State`、
  `StreamPolicy::Side`、以及 `MemoryPlan` 的偏移（算了，执行器却逐槽分配）。
  **写下一个字段时，同一步里就要写下谁读它。**
- **凡是对错要靠事实判断的地方，必须有两个实现 + 一个数值基准。** 单一实现可以自洽且错误：
  一份切分规则把权重约定写反了，**全部测试通过**，直到第二个实现出现才暴露。
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

`docs/architecture.md` §5 列了六个待定决定（D2–D6），其中 D2（模型如何表达 + 切分谁决定）是crux，
未定之前不要写模型层代码。用户已明确的原则：**减少自动推导 —— 布局由模块显式声明，框架只做校验与插通信。**
