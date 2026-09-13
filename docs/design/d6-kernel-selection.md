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
causal_conv1d-1.6.2.post1+cu.13.0.torch.2.11-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

索引根：`https://wheels.astral.sh/simple/cu130/`（17 个包），含我们需要的大部分：

| 我们要的 | 索引里的包 |
|---|---|
| `causal_conv1d` | `causal-conv1d`（1.6.2.post1，有 torch 2.11 档） |
| `sdpa` | `flash-attn`（2.8.3.post1） |
| MoE grouped GEMM | `grouped-gemm`、`megablocks`，以及 `vllm`（内含 fused MoE Triton kernel） |
| 别的可选 | `transformer-engine`、`deepgemm`、`deepep`、`sageattention` |
| **FLA（`gated_delta_rule`/`RMSNormGated`/`l2_norm`）** | 索引里**没有** —— 它是纯 Python/Triton 包，PyPI 直接装，同样**不需要编译** |

## 3.7 落地时的实测结论（2026-09，逐包核对；**推翻上表三行**）

上表是按"PyPI 上有包"写的，真正去把 wheel 拆开看之后，三处不成立：

1. **`causal-conv1d` 的 wheel 里没有头文件，也没有可链接库。**
   整个 wheel 只有 `causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so`（CPython 扩展）与 4 个
   `.py`；`csrc/`、`include/` 不进 wheel。C++ 入口确实在（`nm -D` + `c++filt`）：
   全局命名空间的
   `causal_conv1d_fwd(const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&, …, bool)`，
   要用只能 `dlopen` + 自己声明 mangled 符号。分发器的注册在 Python 侧
   （`torch.library.custom_op`，库名 `DaoAILab`，`import causal_conv1d` 之后才存在），
   而那个 `.so` 的 `DT_NEEDED` 里有 **`libtorch_python.so`**。
2. **FLA 在 C++ 侧按名字调不到。** `flash-linear-attention` / `fla-core` 全部是 `py3-none-any`
   纯 Python 包；稀疏克隆整仓后 `.cpp/.cu/.cuh/.h/.hpp/.so` 数量为 **0**，`TORCH_LIBRARY` 与
   `torch.ops.` 出现次数为 0。`chunk_gated_delta_rule` / `l2_norm` / `RMSNormGated`
   （实际类名 `FusedRMSNormGated`）都是普通 Python 函数 → `@triton.jit`。`@dispatch` 是它自己的
   Python 后端选择器，从不碰 torch 分发器。
3. **MoE grouped GEMM 没有任何预编译二进制。** `grouped-gemm`（tgale96，0.3.0）与
   `nv-grouped-gemm`（fanshiqing，1.1.4.post8）在 PyPI 上**只有 sdist**；`megablocks` 0.10.0
   同样只有 sdist，且 `install_requires` 钉死 `torch>=2.7.0,<2.7.1`（排除 2.11）。
   `wheels.astral.sh/simple/cu130` 里这三个包**不存在**。

`torch.utils.cpp_extension` 也有一个要绕开的默认：`CppExtension` / `CUDAExtension` 会无条件
追加 `-ltorch_python`（只有 `py_limited_api=True` 才跳过），而我们的插件是被**没有 Python
解释器的 Rust 宿主** `dlopen` 的。构建因此只借它拿 include / library 路径，链接项自己写。

**因此 D6 第一步的形态是**：一个链 libtorch 的 C++ 插件（`plugins/aten/`），把词表映射到 ATen
——GEMM 走 cuBLAS、注意力走 ATen 的 FlashAttention-2 / mem-efficient、因果卷积走 cuDNN 的
`conv1d`、GDN 走声明的 recurrence、MoE 走按专家的 ATen 组合。**上游专用 kernel（causal-conv1d
的融合 silu、FLA 的分块、grouped GEMM）是"速度选项"，不是正确性前提**；引入它们要么把 CPython
嵌进宿主、要么自己编译扩展，两条都超出"用户已确认"的范围，故未做。

## 3.8 位置常量与下一步

- `causal_conv1d` 的快速路径：`dlopen` 上游 `.so` + dlsym mangled 符号，或嵌 CPython 走
  `DaoAILab::_causal_conv1d_fwd_cpp`。两者都需要用户点头。
- MoE 快速路径：先看 torch 2.11 是否带 `torch._grouped_mm`（ATen 内置的 grouped GEMM，
  零外部依赖）；否则按 sdist 自己编译 `nv-grouped-gemm`。
- GDN 快速路径：FLA 的分块 kernel 只能通过嵌入式 Python bridge 到达。


## 4. 实际执行顺序（按 §3.7 的实测修正）

1. **ATen 插件骨架**（已做）：`plugins/aten/`，C++17 + ABI v1，28 个算子里先做完全套前向要用的
   那批；`rustrain ops list --plugin librustrain_aten.so` 已经能列出 60 个实现（32 reference +
   28 aten），证明 Rust 宿主 `dlopen` C++ 插件这条边界是通的；
2. **框架的设备路径**（进行中）：`--device cuda` 的分配器（driver API，`dlopen`，核心 crate
   仍然零 CUDA 依赖）、按设备对齐的显存打包、`run` / `ops check` 的插件与 recipe 参数；
3. **逐算子门禁**：`rustrain ops check --plugin ... --recipe plugins/aten/aten.toml --device cuda`
   ——每个 aten 变体都与 reference（CPU oracle，小张量）对拍；
4. **一层真前向**：真实权重、单层、与 HF 的同层 hidden 对比；
5. **NCCL 后端** → 8 卡 TP/EP 真跑，出并行效果数据（权重字节、步数、集合通信次数与字节量）；
6. 速度选项（`causal-conv1d` / FLA / grouped GEMM / `_grouped_mm`）按 §3.8 逐个向用户确认后再接。

