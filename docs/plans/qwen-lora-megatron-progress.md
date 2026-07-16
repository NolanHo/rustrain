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

Verified: Qwen3.5/3.6 native forward/backward slices, GDN CUDA path, grouped MoE fallback, EP smoke, dense replicated-DP smoke, TP-only latent-rank-sharded LoRA smoke, dynamic batch logical-step update, ABI10 fixed and per-tenant optimizer-step restore, selected-tenant isolation, and 5D topology mapping.

Not yet verified or implemented: Megatron-style frozen base-weight TP, multi-axis TP+DP/EP, PP/CP, FP32 accumulation/abort, rank-sharded checkpoint topology, DeepEP/TE prebuilt integration, and matched Megatron throughput.

# Durable Milestones

- `4e242ee`: added Megatron-style 5D topology contract and launcher normalization; `cargo test -p rustrain-parallel --lib` passed 14/14.
- `afcf091`: restored native C++ Adam step on checkpoint load and added ABI8 smoke assertions; server checkpoint tests passed 2/2.
- ABI9 working tree: selected adapter IDs flow through HTTP, IPC, Rust, and C++; each dynamic tenant owns its Adam clock, checkpoint metadata preserves it, and failed selection restores the complete registry.
- ABI10 working tree: TP-only mode shards each fixed and dynamic adapter's latent rank, all-reduces only activation-level LoRA deltas on a split TP communicator, and keeps replicated MoE layers off the EP communicator.
- H20 `123.57.26.97:28004`: ABI8 native smoke passed grouped/fallback parity, GDN, dense/MoE LoRA, dynamic adapters, and step setter validation.
- H20 `123.57.26.97:28004`: ABI9 native smoke passed selected-tenant training with a positive selected update, exactly zero unselected update, independent clocks (`2` vs `1`), and registry preservation after an unknown ID.
- H20 `123.57.26.97:28004`: ABI10 two-rank TP native smoke passed on both ranks for MoE, dense MLP, GDN/linear attention, dynamic multi-LoRA, and selected-tenant isolation. Rank-local LoRA tensors used distinct rank `4` slices for global rank `8`; losses matched and both shards had positive updates.
- Target runtime probe: PyTorch 2.5.1+cu121, ABI0; Transformer Engine, flash-attn, DeepEP, Triton, and DeepSpeed are not importable.

# Decisions During Execution

- Keep TP and EP communicators separate; do not reuse the existing EP `LayerConfig.nccl_comm` for LoRA TP deltas.
- Scope multi-LoRA `n_max` rendezvous files per native context; reusing one filename across sessions can give ranks different chunk schedules and deadlock TP collectives.
- Do not enable Qwen TP/PP/CP by merely relaxing runtime validation.
- Treat Exa/Jina dependency search failures as missing evidence, not as proof that a package is compatible.

# Verification

Passed: `cargo test -p rustrain-core --lib` (8), `cargo test -p rustrain-parallel --lib` (14), `cargo test -p rustrain-server --lib` (3), Qwen unit tests (3), Qwen integration (6), remote ABI8 smoke, remote ABI9 selected-tenant native smoke, remote ABI10 single-rank native smoke, and remote ABI10 two-rank TP native smoke.

Not run: Megatron-style base-model TP, multi-axis TP+DP/EP, PP/CP, FP32 accumulation equivalence, rank-sharded checkpoint resume, and matched Megatron performance benchmark.
