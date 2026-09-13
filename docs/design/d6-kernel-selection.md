# D6 kernel 插件选型（待用户确认）

用户决定（2026-09）：**要 GPU 插件，CPU 插件不用做**；允许引入外部 CUDA / Tilelang / CUTLASS，
**尽可能不手写**；**引入哪些 kernel 需要用户确认**。本文是那份清单，确认后再动手。

## 0. 原则

框架的职责是编排（分片、通信、显存、精度）；kernel 是实现体（T1），换实现体不该动框架。
所以选型标准只有三条：**覆盖我们的算子词表**、**上游在用同一份实现**（数值可比、责任可追）、
**不自己写**.

## 1. 逐算子映射（按成本从低到高）

| 我们的算子（plan 里的节点数） | 用哪个上游实现 | 为什么是它 |
|---|---|---|
| `linear` / `matmul` / `bmm`（298） | **cuBLAS**（经 ATen） | 没有理由自己写 GEMM |
| `elementwise_*`、`softmax`、`rmsnorm`、`layernorm`、`embedding`、`narrow`/`reshape`/`view`/`cat`/`quantize` 等（~600） | **ATen CUDA kernel** | 逐元素与索引类算子，ATen 的实现足够好且零维护 |
| `sdpa`（22） | **`torch.nn.functional.scaled_dot_product_attention`**（CUDA 上自动选 FlashAttention-2 / mem-efficient / math） | GQA + causal + padding mask 全都支持；HF 也是走它 |
| `causal_conv1d`（90） | **`causal-conv1d`**（Dao，`causal_conv1d_cuda`） | **HF 调的就是这个 kernel**，数值可对齐 |
| `gated_delta_rule`（30） | **FLA `chunk_gated_delta_rule`**（flash-linear-attention） | 同上：HF 调的就是它；自己写这个 recurrence 是自找麻烦 |
| `rmsnorm_gated`（30）、`l2norm`（60） | **FLA 的 `RMSNormGated` / `l2_norm`** | 同上；也是 HF 的实际调用点 |
| `moe_layer`（41） | **vLLM 的 `fused_moe`**（Triton）或 **`grouped_gemm`**（CUDA） | 需要专家维的 grouped GEMM；HF 那条路是 Python 循环，性能上不能接受 |
| `rope`（22） | ATen 组合或 FLA 的 rotary | 已经在图上拆成逐 head，属轻量算子 |
| `topk_router`（41） | ATen（`softmax` + `topk` + 可选重归一化） | 数据依赖的索引类算子，不需要自定义 kernel |

**一句话**：**一个 ATen 插件吃掉九成节点**（GEMM/逐元素/SDPA/索引），
**三个上游 CUDA 库补上 MoE 与线性注意力的专门 kernel**（`causal-conv1d`、`flash-linear-attention`、
MoE grouped GEMM）。**不手写任何 kernel**；只有在某个算子三个来源都不覆盖时才考虑 Tilelang
（届时应再次确认）。

## 2. 与现有架构的关系

- 插件形态沿用 `rustrain-abi` v1：`.so` + `execute` + `infer` + `requires`。ATen 插件是
  "aten provider"（`crates/rustrain-kernels/Cargo.toml` 的描述里原本就写着它，只是从未实现）。
- **约束**：插件与宿主必须同一个 `_GLIBCXX_USE_CXX11_ABI` 与同一份 libtorch（仓库 `AGENTS.md` 的 GOTCHAS）。
  宿主上 torch 2.11.0+cu130 已就位；CUDA 13 与三个上游库需要按该组合编译。
- **多 rank**：8 卡并行需要 NCCL 后端（`CollectiveBackend` 的另一个实现），与 kernel 插件正交。
  上游库（FLA/vLLM MoE）自身都假设 NCCL，所以这一步同时把通信后端换成 NCCL 是自然的。

## 3. CPU reference provider 的处置（与用户原话有出入，需确认）

用户说"CPU 插件直接禁用或者删掉"。我的建议是**分开两件事**：

- **执行路径**：`run` 默认不再用 reference provider，改用 GPU 插件（这条完全照办）。
- **一致性门禁**：`ops check` 的 reference provider 是**语义真值** —— 门禁要有对照物才有意义
  （"两个实现 + 一个数值参考才能判对错"是这个仓库的核心纪律，D5 的数值对比也依赖它）。
  把它删掉等于把唯一的对照物也删掉。

所以建议：**reference provider 保留为"仅门禁用的 oracle"，从执行路径上禁用**。
如果你的意思是连门禁的对照物也不要，我照办 —— 但那会让"数值对齐"失去意义，需要你明确。

## 3.5 用户确认（2026-09）与两条附加约束

**Kernel 清单已确认**：ATen 插件（GEMM/逐元素/SDPA）+ `causal-conv1d` + FLA（`chunk_gated_delta_rule` /
`RMSNormGated` / `l2_norm`）+ MoE grouped GEMM。**不手写 kernel**。

用户附加的两条约束（比 §3 更严）：

1. **CPU 执行不再允许** —— reference provider **只能作为"对照"（oracle）**，执行路径上一律走 GPU 插件。
2. **对照要谨慎启动** —— CPU 太慢，不能拿来跑真模型：只用于**逐算子的小规模数值对照**与门禁用例，
   不用于整模型前向。也就是说 `ops check` 的 conformance 用例继续用 CPU 是合适的（小张量），
   但"整模型 CPU 前向"这条路彻底关闭 —— 包括此前考虑过的 CPU 多 rank 执行。

（§3 里"保留 reference provider 作为门禁 oracle"的建议，按这两条约束执行：**保留、但仅限小规模对照**。）

## 3.6 预编译 wheel 坐标（用户提示 wheels.astral.sh，2026-09 核实）

**不自己编译**。宿主是 CUDA 13.0 + torch 2.11 + py3.12，索引里正好有对应档：

```
causal_conv1d-1.6.2.post1+cu.13.0.torch.2.11-cp312-cp312-manylinux_2_28_x86_64.whl
```

索引根：`https://wheels.astral.sh/simple/cu130/`（17 个包），含我们需要的大部分：

| 我们要的 | 索引里的包 |
|---|---|
| `causal_conv1d` | `causal-conv1d`（1.6.2.post1，有 torch 2.11 档） |
| `sdpa` | `flash-attn`（2.8.3.post1） |
| MoE grouped GEMM | `grouped-gemm`、`megablocks`，以及 `vllm`（内含 fused MoE Triton kernel） |
| 别的可选 | `transformer-engine`、`deepgemm`、`deepep`、`sageattention` |
| **FLA（`gated_delta_rule`/`RMSNormGated`/`l2_norm`）** | 索引里**没有** —— 它是纯 Python/Triton 包，PyPI 直接装，同样**不需要编译** |

**调用路径**（这一点决定了插件的形态）：这些 wheel 是**带 CUDA 扩展的 Python 包**，注册成 torch 的自定义算子。所以我们的插件不写 kernel，而是**链 libtorch 的 C++ 分发器**：`at::matmul` / `at::scaled_dot_product_attention` 直接走 ATen，`causal_conv1d` 与 FLA 的算子通过 `torch.ops.*` 注册进分发器后同样能在 C++ 侧调到。换句话说：**一个链 libtorch 的 Rust 插件**，把我们的算子词表映射到 ATen + 那几个扩展已注册的算子 —— 零 kernel 编译、零手写。

前置条件（宿主上逐条确认）：`_GLIBCXX_USE_CXX11_ABI` 与 wheel 的 `cxx11abiTRUE` 一致、torch 2.11.0+cu130 的 C++ 头文件与 libtorch 可链、Python 3.12。

## 4. 确认后我按什么顺序做

1. **ATen 插件骨架**：`.so` + ABI v1 + 最小算子集（`linear` + `elementwise` + `rmsnorm`），
   在宿主上跑通一个真前向的一层；
2. **SDPA + causal-conv1d + FLA** 三件接入，跑通 full-attention 层与 GDN 层各一层；
3. **MoE grouped GEMM**；
4. **NCCL 后端** → 8 卡 TP/EP 真跑，出你想要的并行效果数据；
5. 每步都要有"与 reference provider 逐算子数值对齐"的门禁（这就是 oracle 的用处）。
