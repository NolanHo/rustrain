# ATen 插件（`cuda.aten.f32`）

框架只负责编排：拿到实现、知道契约、验证算的是同一件事。这个 `.so` 就是"实现体"（T1）——
它把算子词表映射到 **ATen**（cuBLAS / cuDNN / torch 的 CUDA kernel），不手写任何 kernel。

- **契约**：与内建 reference provider 同名、同操作数顺序、同属性、同数值约定；
  变体名 `cuda.aten.f32`（`cuda` 前缀是框架识别"这个实现要设备"的约定）。
- **形态**：`librustrain_aten.so`，导出唯一符号 `rustrain_plugin_v1`，由框架 `dlopen`。
- **语言**：C++17 + `rustrain_op.h`（ABI v1 是 C 接口，实现语言不属于契约的一部分）。
  选 C++ 而不是 Rust 的唯一理由：所有上游 kernel（ATen 本身、causal-conv1d、FLA）都是
  C++/CUDA 或 Python，Rust 侧要调 ATen 仍然得写一层 C++ shim —— 那就直接写在 C++ 里。

## 构建

```bash
# 用"将来会跟插件一起跑"的那个 torch（同一 _GLIBCXX_USE_CXX11_ABI、同一 CUDA 构建）
python3 plugins/aten/build.py --out /somewhere/build
```

`build.py` 只用 `torch.utils.cpp_extension` 取 include / library 路径，**不用它的
`CppExtension` 链接列表**——那个列表默认带 `-ltorch_python`，而这个插件是被没有 Python
解释器的 Rust 宿主 `dlopen` 的，不能依赖 libtorch_python。链接项是
`-ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10`，并写死 rpath 指向 torch 的 `lib/`。
当前用 `-std=c++17`：ABI 头里的字段名 `requires` 在 C++20 是关键字，改名是一次纯源码修改，
排在下一个动 ABI 的改动里。

## 跑

```bash
# 逐算子数值门禁：每个变体都与 reference provider（语义真值）对拍
rustrain ops check --plugin /abs/path/librustrain_aten.so --recipe plugins/aten/aten.toml --device cuda

# 前向：整条计划都解析到 aten 变体，缓冲区在显存里
rustrain run --plugin ... --recipe plugins/aten/aten.toml --device cuda \
        --model crates/rustrain-model/tests/fixtures/qwen36-text --checkpoint <真实权重> \
        --tokens 9707,11,1879,0,323,358,314,279 --out /var/tmp/rustrain.npz
```

## 验证（2026-09，宿主 8× L20X，sm_89）

| 证据 | 命令 | 结果 |
|---|---|---|
| Rust 宿主 `dlopen` C++ 插件 | `rustrain ops list --plugin librustrain_aten.so` | 60 个实现（32 reference + 28 aten） |
| 逐算子对拍（框架门禁） | `rustrain ops check --plugin … --recipe plugins/aten/aten.toml --device cuda` | **59 case / 0 failing**，exit 0；reference 留在 host、`cuda.*` 变体跑在显存里 |
| 插件自检（对照 Rust oracle 源码） | 宿主 `gpu-work/smoke_aten.py` | 31/31 case，26 个算子，两次运行字节一致 |

开发过程中被门禁抓到的两类真问题（都是"看起来对"的）：

1. **`infer` 里建了 ATen 张量**（`bmm` / `elementwise_binary` / `reduce`）：infer 跑在任何分配之前，
   descriptor 的 `data` 是 null，`at::from_blob(nullptr, …, CUDA)` 会去问"这个指针在哪块设备上"并报
   `The specified pointer resides on host memory`。形状只能来自 shape/stride 算术。已修，并写进
   `skills/architecture/SKILL.md` §3 的禁止模式。
2. **MoE 的 down 投影漏了 `.t()`**：`experts_down_proj` 是 `[E, H, I]`（H 是输出），参考实现是
   `act @ down[e]ᵀ`。H ≠ I 时报形状错，H == I 时**静默算错**（GPU 自检 maxdiff 1.06）。已修。

## 覆盖

| 算子 | 怎么实现 |
|---|---|
| `matmul` / `linear` / `bmm` | `at::matmul`（cuBLAS）。`linear` 的 `w` 是 `[K, N]`，不是 torch `nn.Linear` 的 `[N, K]` |
| `elementwise_unary` / `elementwise_binary` / `compare` / `reduce` / `softmax` | ATen 逐元素与归约 kernel，公式按 reference 的写法逐条抄（`silu = x/(1+e^-x)`、`softplus` 的 20 阈值、NaN 一律 0 的 mask 等） |
| `rmsnorm` / `layernorm` / `l2norm` / `rmsnorm_gated` | ATen 组合。`weight_offset`（trunk 的 `1+w` vs GDN 的裸 `w`）是数据，读属性，不猜 |
| `sdpa` | `at::scaled_dot_product_attention`（CUDA 上即 FlashAttention-2 / mem-efficient），GQA 走 `repeat_interleave`，causal 与 additive mask 合成一个再加 |
| `rope` | ATen 组合：half-split、`inv_freq[j] = theta^(-2j/rotary_dim)`、位置沿第 0 轴 |
| `causal_conv1d` | `at::conv1d`（cuDNN）depthwise + 裁回长度 L |
| `gated_delta_rule` | ATen 组合直接跑声明的 recurrence（`S_t = S_{t-1}e^{g_t} + k_t((v_t - S_{t-1}k_t)β_t)`，更新后读）。`chunk_size` 是分块快速路径的尺寸，不改变取值，所以这条 body 不读它 |
| `topk_router` | `softmax` + 稳定降序排序后取前 k（并列取小专家号），`norm_topk_prob` 照声明 |
| `embedding` / `gather` / `scatter` | ATen 索引 kernel；负索引一律拒绝（不绕回） |
| `moe_layer` | ATen 组合：按专家选 token → 三个 matmul → `index_add_`，dropless、升序累加，共享专家 sigmoid 门控 |
| `view` / `reshape` / `transpose` / `narrow` / `broadcast` | 零拷贝视图：把结果张量的指针 / 形状 / 步长交回执行器（`out.data` 指回输入） |
| `cat` | 唯一真拷贝的移动算子 |
| `quantize` / `dequantize` / `amax_update` / `adamw` | **未发布**：前向用不到，属于训练 / 量化路径 |

## 上游 kernel 的实测结论（2026-09 核实，决定了上面的覆盖表）

最初计划引入 `causal-conv1d`、FLA、MoE grouped GEMM。逐个查过 wheel 之后：

- **`causal-conv1d`**：`wheels.astral.sh/simple/cu130` 上确有 torch 2.11 + cu13 档，但 wheel 里
  只有 `causal_conv1d_cuda*.so`（CPython 扩展）和 4 个 `.py`，**没有头文件、没有可链接库**；
  它的 C++ 入口是全局命名空间的 `causal_conv1d_fwd(at::Tensor, ...)`（mangled），分发器注册
  写在 Python 里（`torch.library.custom_op`，库名 `DaoAILab`），而且那个 `.so` 依赖
  `libtorch_python`。要用它就得 `dlopen` + 自己声明 mangled 符号，或者把 CPython 嵌进宿主。
  **当前用 ATen 的 `conv1d` 顶上**：数学一样（同一 `out[t,c] = Σ_k w[c,0,k]·x[t+k-pad,c]`），
  门禁能对拍；缺的是融合 silu 与"一次 kernel 走完"的速度。
- **FLA（`chunk_gated_delta_rule` / `RMSNormGated` / `l2_norm`）**：PyPI 上是纯 Python 包
  （`flash-linear-attention` + `fla-core`，全部 `py3-none-any`），仓库里 **0 个** `.cpp/.cu/.h/.so`，
  也**没有**把这三个算子注册进 torch 分发器（`@dispatch` 是它自己的 Python 后端选择器）。
  C++ 插件按名字调不到。**当前用 ATen 组合跑等价的 recurrence**。
- **MoE grouped GEMM**：`grouped-gemm`、`nv-grouped-gemm`、`megablocks` 在 PyPI 上**只有 sdist**
  （要自己编译），`megablocks` 还钉死 `torch<2.7.1`；`wheels.astral.sh` 索引里**没有**这三个包。
  **当前用 ATen 组合**；`torch._grouped_mm` 若在 torch 2.11 可用，是下一步的快速路径。

结论：**ATen 一条路就能把整个前向跑完**（GEMM 走 cuBLAS、注意力走 FlashAttention-2、
卷积走 cuDNN），上游专用 kernel 是速度选项而不是正确性前提。引入它们需要宿主里嵌 CPython
或自己编译扩展，这两条都属于"要用户确认"的范围，因此没有擅自做。

## 已知限制

- 只声明 f32。bf16/fp8 变体是后续的 `cuda.aten.bf16`（dtype 会让解析自动选中它）。
- `gated_delta_rule` 是逐 token 的 recurrence：数学与 reference 一致，但 S=512 时是 512 次
  迭代 × 每步 ~6 个小 kernel。分块形式（或 FLA）是快速路径。
- `moe_layer` 每个专家一次 `nonzero`：E=256 时每次调用都有一次 device→host 同步。
  `_grouped_mm` / 排序分桶是快速路径。
- 三处 `expand` 语义的算子（`sdpa` 的 mask、`broadcast`）会在视图上做一次 `reshape`；
  非连续输入的 `reshape` 会隐式拷贝，代价在但正确。
