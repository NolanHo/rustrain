---
type: ProgressRecord
title: Qwen LoRA Megatron runtime progress
description: Verified milestones and open acceptance gaps for the Qwen native LoRA runtime.
tags: [qwen3.5, qwen3.6, lora, progress]
timestamp: 2026-07-17T00:00:00Z
---

# Source Spec

[qwen-lora-megatron-spec.md](/docs/plans/qwen-lora-megatron-spec.md)

# Current State

Verified: Qwen3.5/3.6 native forward/backward slices, GDN CUDA path, grouped MoE fallback, EP smoke, dense replicated-DP smoke, dynamic batch logical-step update, ABI9 fixed and per-tenant optimizer-step restore, selected-tenant isolation, and 5D topology mapping.

Not yet verified or implemented: Qwen native TP/PP/CP, FP32 accumulation/abort, rank-sharded checkpoint topology, DeepEP/TE prebuilt integration, and matched Megatron throughput.

# Durable Milestones

- `4e242ee`: added Megatron-style 5D topology contract and launcher normalization; `cargo test -p rustrain-parallel --lib` passed 14/14.
- `afcf091`: restored native C++ Adam step on checkpoint load and added ABI8 smoke assertions; server checkpoint tests passed 2/2.
- ABI9 working tree: selected adapter IDs flow through HTTP, IPC, Rust, and C++; each dynamic tenant owns its Adam clock, checkpoint metadata preserves it, and failed selection restores the complete registry.
- H20 `123.57.26.97:28004`: ABI8 native smoke passed grouped/fallback parity, GDN, dense/MoE LoRA, dynamic adapters, and step setter validation.
- H20 `123.57.26.97:28004`: ABI9 native smoke passed selected-tenant training with a positive selected update, exactly zero unselected update, independent clocks (`2` vs `1`), and registry preservation after an unknown ID.
- Target runtime probe: PyTorch 2.5.1+cu121, ABI0; Transformer Engine, flash-attn, DeepEP, Triton, and DeepSpeed are not importable.

# Decisions During Execution

- Keep TP and EP communicators separate; do not reuse the existing EP `LayerConfig.nccl_comm` for LoRA TP deltas.
- Do not enable Qwen TP/PP/CP by merely relaxing runtime validation.
- Treat Exa/Jina dependency search failures as missing evidence, not as proof that a package is compatible.

# Verification

Passed: `cargo test -p rustrain-core --lib` (8), `cargo test -p rustrain-parallel --lib` (14), `cargo test -p rustrain-server --lib` (3), Qwen integration (6), remote ABI8 smoke, and remote ABI9 selected-tenant native smoke.

Not run: full Qwen TP/PP/CP smoke, FP32 accumulation equivalence, rank-sharded checkpoint resume, and matched Megatron performance benchmark.
