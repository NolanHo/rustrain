# ATen 插件（`cuda.aten.f32` / `cuda.aten.bf16`）

框架只负责编排：拿到实现、知道契约、验证算的是同一件事。这个 `.so` 就是"实现体"（T1）——
它把算子词表映射到 **ATen**（cuBLAS / cuDNN / torch 的 CUDA kernel），不手写任何 kernel。

- **契约**：与内建 reference provider 同名、同操作数顺序、同属性、同数值约定；每个算子发布
  两个变体：`cuda.aten.f32` 与 `cuda.aten.bf16`（`cuda` 前缀是框架识别"这个实现要设备"的
  约定），由 plan 槽位的 dtype 决定解析到哪一个。bf16 变体的 numerics：in/out bf16、
  accum f32（ATen 的 bf16 matmul 以 fp32 累加），grad bf16。
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

## 调度（schedule）：同一个算子的第二种算法

一个算子的 **body** 可以在保持契约与数值语义的前提下换一种算法，这不需要新变体、新 provider
或新配方——变体是 `(provider, dtype)` 的解析单位，同一算子的两个都能接受该 dtype 的变体是
**歧义**而不是选择（`Registry::resolve_by_default_provider` 按 provider 名匹配，两个都能跑就
报 ambiguous）。所以调度住在 body 里，由 body 自己挑，并把挑选规则写在一处。

今天只有 `moe_layer` 有两个调度，规则是一句话加一条前提：

| 条件 | 调度 |
|---|---|
| `torch._grouped_mm` 接受这些操作数（contraction 维是 16 字节的整数倍，`GroupedMMUtils.h: check_valid_strides_and_return_transposed`） | **grouped**：把 (row, slot) 对按专家序排好（host 侧一次，和 reference 循环用的是同一趟），三次 `_grouped_mm` 代替 `E×K` 次小算子 |
| 形状不满足对齐 | **reference 循环**：逐 (专家, slot) 的 `index_select` + 三次 matmul + `index_copy_` |

两点取舍写在 `ops_moe.cpp` 里，不藏在代码里：

- **为什么走 grouped**：循环的开销是**每次调用的固定成本**付 `E × K` 遍（本模型几何下 2048 遍），
  与有多少 token 路由过去无关；本模型的两个调度差 30×（宿主实测 45.4 → 1.50 ms/层，见 spec D6.12）。
- **形状不满足对齐时**回退到循环而不是报错：形状合法、循环就在同一个 body 里，这是**调度**选择
  （ATen 自己也按 shape 挑 kernel），不是解析降级，也不是契约变化。代价是**下游看不出跑了哪个**：
  ABI 没有通道、plan digest 记的是变体而不是 body 内部分支 —— 所以门禁用**两条 case 各跑一条**来
  覆盖（`H=I=2` 走循环、`H=I=8` 走 grouped），而不是靠"记录回退"。

## 验证（2026-09，宿主 8× L20X，sm_89）

| 证据 | 命令 | 结果 |
|---|---|---|
| Rust 宿主 `dlopen` C++ 插件 | `rustrain ops list --plugin librustrain_aten.so` | 88 个实现（32 reference + 28×2 aten：f32 与 bf16 各一） |
| 逐算子对拍（框架门禁） | `rustrain ops check --plugin … --recipe plugins/aten/aten.toml --device cuda` | `94 case(s): 0 failing`；`cuda.aten.f32` 的 numeric+determinism 全过；`cuda.aten.bf16` 行显示 **skip**（理由写明"dtype f32 is not accepted… 门禁今天每个 case 只生成一种 dtype"），不是 fail——**bf16 变体今天不在门禁的数值判定里**，框架侧 gap |
| `moe_layer` 的两条调度各自被门禁覆盖 | 同上，`--op moe_layer`（新增 `H=I=8` 对齐 case） | 两条 case：`H=I=2` 走循环、`H=I=8` 走 grouped，都 pass；把 `offsets` 改回"组起点"的变异构建让第二条 **FAIL**（gate exit 1），第一条仍 pass —— 即这条 case 真的覆盖 grouped |
| 插件自检（对照 Rust oracle 源码） | 宿主 `gpu-work/smoke_aten.py` | 31/31 case，26 个算子，两次运行字节一致 |
| bf16 变体自检（bf16 跑 vs f32 跑后取整到 bf16） | 宿主 `gpu-work/smoke_bf16.py`（2026-09-14） | 61/61 case 通过：全部 28 个算子（含 bf16 的 `gated_delta_rule` fp32 state、`sdpa`、`causal_conv1d`、`moe_layer`），两次运行字节一致；56 个描述符的 per-variant mask / numerics 逐条断言通过 |
| `moe_layer` 的 grouped 调度（变异对照：把调度条件翻成 false 的同一份源码另编一个 `.so`） | 宿主 `rustrain run`（world=1、同 probe token、bf16 / f32） | bf16 **逐位相同**（logits 与 42 个 hidden state 全等，不是"在容差内"）；f32 差 1.6e-6（重结合顺序不同）；tp=4 的 `max\|diff\|` vs world-1 两条调度都是 **1.21875** |
| grouped 调度的空专家组 | 宿主 python 探针（同一串 ATen 调用） | 三种稀疏度（0 / 0 / 192 个空专家，共 256）下与循环逐位相同 |

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

- 发布 f32 与 bf16 两个变体。bf16 变体的 numerics：in/out bf16、accum f32（ATen 的 bf16
  matmul 以 fp32 累加）；`gated_delta_rule` 的 recurrence state 按契约保持 fp32，输入 cast 到
  f32、输出写回输入 dtype。fp8 变体是后续工作。
- `gated_delta_rule` 是逐 token 的 recurrence：数学与 reference 一致，但 S=512 时是 512 次
  迭代 × 每步 ~6 个小 kernel。分块形式（或 FLA）是快速路径。
- `moe_layer` 每个专家一次 `nonzero`：E=256 时每次调用都有一次 device→host 同步。
  `_grouped_mm` / 排序分桶是快速路径。
- 三处 `expand` 语义的算子（`sdpa` 的 mask、`broadcast`）会在视图上做一次 `reshape`；
  非连续输入的 `reshape` 会隐式拷贝，代价在但正确。

## ABI v2：算子的切分规则也是声明

每个算子在 `OpDef` 里声明 `shard`（`RS_SHARD_*`），`plugin.cpp` 把它填进 `rs_op_desc::shard`。
规则属于**算子**而不是变体：同一个算子的 reference 实现与 ATen 实现必须声明同一条规则，否则框架报
"implementations of `X` disagree about its sharding rule" 并拒绝推导该节点的布局。

| 规则 | 本插件的算子 |
|---|---|
| `RS_SHARD_DECLARED` | `reduce` |
| `RS_SHARD_ELEMENTWISE` | `view`/`reshape`/`transpose`/`narrow`/`cat`/`broadcast`/`elementwise_*`/`compare`/`softmax`/`rmsnorm`/`layernorm`/`rope`/`gather`/`scatter`/`cross_entropy` |
| `RS_SHARD_LINEAR` | `linear` |
| `RS_SHARD_EMBEDDING` | `embedding` |
| `RS_SHARD_MATMUL` | `matmul`/`bmm` |
| `RS_SHARD_PASS_THROUGH` | `causal_conv1d`/`gated_delta_rule`/`l2norm`/`rmsnorm_gated`/`sdpa`/`topk_router`/`moe_layer` |

`sdpa` 的 per-head 形式也随 v2 改成**声明形式、计数来自张量**：`per_head: true`，头数读
`q.shape[-2]` / `k.shape[-2]`，GQA 分组 = 本地 q 头 ÷ 本地 kv 头。原来的 `num_heads` / `num_kv_heads`
是全局计数，分片后会和它描述的张量矛盾（`qwen36-5d-example.md` §4）。
