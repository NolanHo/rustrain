---
type: ChangeSpec
title: GLM-5.2 Complete Implementation
description: Make GLM-5.2 EP, TP, and CP paths numerically correct, feature-complete, and independently testable.
tags: [glm5, training, ep, tp, cp, fp8]
status: accepted
---

# Problem

The repository contains a working-shaped GLM-5.2 implementation, but several
paths are incomplete or silently diverge from the model contract: RoPE math,
IndexShare wiring, MoE dispatch, expert caching, FP8 dequantization, TP/CP
LoRA application, CP global attention/loss alignment, and final loss reporting.

# Target Outcome

GLM-5.2 training has one explicit model contract shared by the Rust fallback
and C++ EP path. Native MTP follows Megatron's training semantics for every
supported topology. This change supports the checkpoint's one native MTP layer
with TP-only and EP-only. The trunk and native MTP extra layer use the same
owner dispatch/return contract for EP-only; changing only the extra layer would
feed it a trunk hidden state reconstructed by the old same-position EP
all-reduce. Combined TP+EP remains a loud pre-load error until Megatron's
sequence-parallel token scatter/gather and independent expert-EP/expert-DP
groups are wired. A larger native MTP depth or MTP with CP above one also fails
loudly: the existing CP ring is not autograd-aware for the MTP data path.
Accumulation preserves global loss numerator/count normalization across the
optimizer step.

# Contract

- Standard RoPE uses `theta^(-(2*i/head_dim))`; interleaved and contiguous
  layouts must match the configured checkpoint layout.
- DSA indexer top-k is causal, stateful, and uses the configured
  `index_topk_freq`, `indexer_types`, and source-layer weights. IndexShare
  state owns its buffers and is freed exactly once.
- MoE computes every selected local expert, preserves each top-k weight, and
  returns zero for ranks with no local assignment. Router behavior follows
  `topk_method`/`scoring_func` and group settings; no hidden fallback.
- Expert caches are keyed by layer, device, dtype, shape, and weight identity
  (or are scoped per layer) and can never reuse another layer's weights.
- FP8 block scales are applied exactly once. LoRA fusion must preserve the
  base-weight scale semantics or explicitly dequantize through the safe path.
- TP applies LoRA to sharded attention weights. CP computes global top-k and
  aligns cross-rank shifted targets/loss for the non-MTP path; MTP with CP>1 is
  rejected until its autograd ring is proven.
- `initial_loss` and `final_loss` are tracked separately. The current EP and
  TP/CP loops expose the first and last training-forward measurements; a
  post-update evaluation remains an explicit follow-up before convergence
  claims are made.
- C++/Rust FFI types use an explicit stable ABI (`int32_t`/`bool` agreement),
  and all null/error paths return a Rust error rather than silently continuing.
- MTP uses Megatron's extra-token data contract: a configured sequence length
  `S` has `S` input tokens and `S` next-token labels sourced from `S+1` raw
  tokens. MTP rolls labels/masks once, keeps the sequence tensor length, and
  excludes the invalid tail with the rolled mask.
- MTP loss is token-normalized after rolling, then multiplied by the configured
  `mtp_loss_scaling_factor` (default `0.1`, matching Megatron). The factor is
  the total auxiliary weight, not a per-rank or per-token multiplier.

# Requirements

1. Correct shared model math: config validation, RoPE/YaRN, DSA/IndexShare,
   causal sparse masking, MoE routing, native MTP, and safe LoRA overlays.
2. Correct C++ kernels and FFI: layer forward, FP8 dequant/GEMM, expert
   dispatch/cache, state lifecycle, batch shapes, and error propagation.
3. Correct training orchestration: EP, TP+EP, CP+EP, LoRA gradients, NCCL
   reductions, checkpointing, and loss evaluation.
4. Regression tests that run without GPUs for pure math/routing/config logic,
   plus CUDA-gated synthetic parity tests for Rust versus C++ where available.
5. Documentation of the supported configurations and explicit failure modes.

# Out Of Scope

- Replacing tch-rs or redesigning unrelated Qwen/DeepSeek engines.
- Production deployment, GPU cluster operations, or publishing a PR.
- Claiming full-model convergence without a real GLM-5.2 checkpoint run.

# Deliverables

## Shared GLM5 contract

Observable outcome: Rust and C++ implement the same validated GLM5 config,
RoPE, DSA/IndexShare, router, and MTP semantics.

Acceptance: CPU tests cover known RoPE frequencies, causal top-k, shared-state
reuse, router weights, and invalid divisibility/config cases; Rust fallback
tests pass.

## Correct EP C++ path

Observable outcome: EP layer forward uses the current layer's expert weights,
all selected local experts, correctly scaled FP8 weights, and leak-free state.

Acceptance: a synthetic multi-layer/multi-expert fixture demonstrates distinct
layer outputs and exact routed-weight accumulation; CUDA parity tests pass when
libtorch/CUDA are available.

## Correct TP/CP training

Observable outcome: TP LoRA is applied to sharded projections; CP handles global
indexer/loss alignment and preserves gradients.

Acceptance: synthetic TP=2 and CP=2 tests verify adapter deltas affect logits,
cross-boundary targets are included, and all ranks agree on loss.

## Verifiable training results

Observable outcome: training summaries report measured initial/final losses and
adapter tensors change when a trainable fixture is run.

Acceptance: CPU synthetic checks pass; a documented CUDA smoke command is
provided and its unavailable environment is reported without false success.

# Dependency Constraints

Implement shared contracts before C++/TP/CP consumers. Keep workers on
non-overlapping write sets. Integrate C++/FFI before session orchestration, then
run cross-path tests and a fresh independent review.

# Native MTP Contract

- GLM-5.2 stores each native MTP parameter layer as the extra decoder layer
  `model.layers.{num_hidden_layers + i}`, not under an `mtp.*` prefix.
- Teacher-forced MTP consumes the trunk's post-final-norm hidden state. For
  prediction position `t`, it computes
  `eh_proj(cat(enorm(embed(token[t+1])), hnorm(hidden[t])))`, then executes the
  complete extra DSA+MoE decoder layer.
- Logits use `shared_head.norm` followed by the model's shared `lm_head` and
  target `token[t+2]`. The supported native MTP layer is response-mask
  normalized after one Megatron-compatible label/mask roll. Its auxiliary loss
  is multiplied by the configured `mtp_loss_scaling_factor`, whose default is
  `0.1`.
- The native MTP layer uses token offset `1` in the C++ prepare fusion and
  target offset `2` in the C++ postprocess/CE operation. Checkpoints declaring
  more than one native layer are rejected explicitly.
- The MTP decoder layer owns its attention, DSA indexer, router, shared expert,
  and routed experts. TP follows Megatron's column/row-parallel tensor layouts.
  The extra-layer EP kernel performs stable token dispatch to expert owners and
  an inverse return; routed output is not reconstructed with an all-reduce over
  zero-filled full-token buffers. This kernel is not promoted as a supported EP
  session until the frozen trunk uses the same owner-dispatch contract.
- `index_share_for_mtp_iteration` controls top-k reuse across autoregressive
  draft iterations. It does not permit reusing an unrelated trunk-layer top-k
  state during teacher-forced training.

Implementation boundary: teacher-forcing fusion and shared-head norm+CE use
coarse-grained C++/FFI operations, returning the loss/count tensors required
for Megatron-compatible normalization. The one native decoder layer is
dispatched through one stable descriptor-based FFI call; C++ owns its
attention, indexer, MLP, and differentiable collectives, while Rust retains
descriptor assembly and the outer training loop. MTP with CP above one is
rejected before communicator construction. EP-only uses the same
autograd-aware owner dispatch/return in both the frozen trunk and extra layer.
Combined TP+EP remains rejected until its independent expert groups and
sequence-parallel token movement are wired. Default RoPE and the checkpoint's
complete YaRN parameterization are both carried through the descriptor.

For TP, `eh_proj` follows Megatron's column-parallel projection followed by a
gather whose backward splits the gradient. The shared output head and CE are
vocabulary-parallel: ranks compute local logits and participate in the global
log-sum-exp/target-logit reduction without materializing a replicated full-vocab
logit tensor. Dense and shared-expert MLP gate/up projections are column
parallel and down projections are row parallel. The implemented extra-layer EP
kernel distinguishes routed expert shards from replicated shared/attention/LoRA
parameters; no single world-group scaling rule may be used for both. Router
decisions are made in global expert coordinates before dispatch, routing
weights are applied once, and the shared expert is evaluated once on the
originating token rank. These rules are also the promotion contract for the
currently incompatible trunk EP path.

The gradient-accumulation promotion contract is explicit. Across microbatch
`j`, the future loop must preserve base and MTP numerators/counts and scale the
MTP contribution as `lambda * (N_j / N'_j) * MTP_sum_j`; the optimizer-visible
loss is divided once by `sum_j N_j`. A mean of per-microbatch means is not
accepted as equivalent. The GLM5 sessions carry these numerator/count tensors
through the optimizer step.

## Kernel performance contract

- MTP offset alignment, embedding gather, projection, shared-head RMSNorm, and
  CE are dispatched from Rust through coarse-grained C++ calls. Rust no longer
  allocates per-layer hidden halos or performs hot-loop pad/cat/narrow work.
- The native MTP decoder is one C++ call per rank. TP projection gathers,
  row-parallel reductions, vocabulary CE collectives, and EP token dispatch
  are autograd-aware C++ operations; Rust does not shuttle hot-path tensors
  between collective calls. EP-only uses the same dispatch contract in the
  frozen trunk and the native extra layer.
- The supported boundary is explicit: exactly one native MTP layer and CP=1.
  A checkpoint declaring more than one layer fails before weights or
  communicators are loaded. TP-only and EP-only are accepted after the CPU
  decomposition tests and static C++/Rust checks in this change pass. Combined
  TP+EP remains an explicit pre-load error until expert groups and sequence
  parallelism are aligned.
- The CE path uses a single-chunk fast path for normal MTP windows and keeps a
  chunked fallback for longer windows, avoiding repeated FFI crossings. TP uses
  the vocabulary-parallel CE ABI directly and never gathers full-vocabulary
  logits.
- Variable-size EP dispatch/return currently copies the small per-peer count
  vectors to host memory once per exchange so NCCL send/receive offsets are
  known before launch. This is semantically correct but remains a measurable
  synchronization cost; device-side offsets or pinned count metadata is a
  follow-up optimization.
- A fused CUDA linear-CE/online-vocabulary reduction remains a follow-up
  optimization: this environment has no visible GPU or CUDA compiler, so no
  throughput or peak-memory claim is made here.

# Validation Limits

- No real GLM-5.2 checkpoint, CUDA device, or multi-rank NCCL runtime is
  available in this checkout, so numerical parity and convergence are not
  claimed.
- No CUDA parity run has exercised multi-rank TP or EP MTP against a real
  checkpoint. Combined TP+EP is additionally blocked by independent process
  groups and sequence-parallel token movement. CPU
  reference tests cover the extra-token contract,
  target/mask roll, valid-token count, loss scaling, TP decomposition, stable
  EP permutation/inverse permutation, and unsupported CP topology. Multi-GPU
  numerical parity remains a promotion gate; the absence of hardware is stated
  explicitly rather than replaced by a throughput or convergence claim.

# Open Questions

- Confirm GPU numerical parity for the native MTP layer against the official
  checkpoint once the model shards and an appropriate multi-GPU host are
  available.
