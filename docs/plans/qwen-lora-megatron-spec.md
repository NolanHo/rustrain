---
type: ChangeSpec
title: Qwen3.5/3.6 LoRA Megatron-level runtime
description: Contract for LoRA-only distributed parallelism, MoE performance, GDN, and independent multi-tenant training.
tags: [qwen3.5, qwen3.6, lora, megatron, distributed]
timestamp: 2026-07-17T00:00:00Z
---

# Problem

The native Qwen3.5/3.6 path has correct and tested single-rank, EP, and dense-DP slices, but it is not yet a Megatron-LM-level LoRA runtime. In particular, Qwen native TP/PP/CP are not implemented, dynamic tenants share optimizer state semantics, and the MoE path lacks fused asynchronous token dispatch.

# Target Outcome

Provide a Qwen3.5/3.6 LoRA-only training backend whose parallel groups, rank-local adapter parameters, optimizer/checkpoint state, and runtime scheduling are correct for the supported TP/PP/DP/EP/CP topology. Base-model parameters remain frozen, but every trainable LoRA projection and every tenant update must be independent and reproducible.

# Contract

- Rust owns configuration, scheduling, data partitioning, checkpoint orchestration, and transport. C++/CUDA owns all training-path math and collectives.
- No Python JIT or ad-hoc dependency builds. A dependency is used only after a compatible prebuilt package is found and verified on the target runtime.
- TP and EP communicators are distinct. EP routed-output collectives must never be reused for LoRA TP output reduction.
- A global LoRA rank is partitioned over TP ranks (`rank % tp_size == 0`); rank-local A/B shapes and checkpoint metadata identify the topology.
- PP owns disjoint layer ranges and a real microbatch schedule. CP owns sequence/KV exchange; configuration fields alone do not satisfy either contract.
- Each dynamic adapter has independent optimizer step, m/v, gradient accumulation state, and checkpoint metadata. A request may train a selected subset of tenants without mutating others.

# Requirements

1. Correct Qwen3.5 full-attention and Qwen3.6 hybrid GDN/full-attention forward/backward with LoRA target projections.
2. Correct LoRA-only TP, DP, EP, PP, and CP group semantics, with explicit unsupported combinations rejected before model allocation.
3. MoE grouped/fused dispatch and communication overlap where a verified prebuilt dependency exists; otherwise retain a correct native fallback and report the gap.
4. FP32 gradient accumulation and logical optimizer boundaries, including abort/zero behavior.
5. Dynamic multi-LoRA selected-tenant scheduling, independent optimizer clocks, and no cross-tenant parameter or state updates.
6. Rank-aware checkpoint save/load and resume equivalence.
7. Target-machine smoke, numerical parity, throughput, peak-memory, and scaling evidence against a Megatron-LM reference under matched settings.

# Out Of Scope

Full-parameter training, unfrozen base weights, unverified third-party packages, Python JIT extensions, and claiming Megatron parity from topology metadata or unit tests alone.

# Acceptance Evidence

| Criterion | Direct evidence |
| --- | --- |
| TP LoRA rank sharding | two-rank native smoke checks local A/B shapes, TP delta all-reduce, and finite loss/update |
| DP/EP separation | multi-rank smoke checks replicated LoRA reduction and local expert ownership separately |
| PP/CP | multi-rank stage/ring smoke with layer ownership and sequence/KV exchange assertions |
| Tenant independence | selected-adapter test proves untouched tenant tensors, m/v, and step are unchanged |
| Accumulation | FP32 accumulation test matches concatenated-batch reference and abort clears pending state |
| Checkpoint resume | save/load continuation matches uninterrupted optimizer state and bias correction |
| Performance | target-machine matched benchmark records tokens/s, step time, peak memory, and communication share |

# Dependency Constraints

The target host currently exposes PyTorch 2.5.1+cu121 with ABI0 and no importable Transformer Engine, flash-attn, DeepEP, or Triton package. Public prebuilt availability must be rechecked before any dependency change; no package is added based on an unavailable search backend.

# Known Baseline

The repository currently has a working native EP/dense-DP subset, grouped MoE microbenchmark evidence, and a separate 5D topology helper. These are prerequisites, not proof of the target outcome.
