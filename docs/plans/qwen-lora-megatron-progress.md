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

Verified: Qwen3.5/3.6 native forward/backward slices, GDN CUDA path, grouped MoE fallback, EP smoke, dense replicated-DP smoke, TP-only latent-rank-sharded LoRA smoke, dynamic batch logical-step update, ABI11 FP32 gradient storage/aggregation, per-tenant optimizer-step restore, selected-tenant isolation, same-topology rank-aware checkpointing, and 5D topology mapping.

Not yet verified or implemented: Megatron-style frozen base-weight TP, multi-axis TP+DP/EP, PP/CP, variable-split token A2A, DeepEP/TE prebuilt integration, cross-topology checkpoint resharding, and matched Megatron throughput.

# Durable Milestones

- `4e242ee`: added Megatron-style 5D topology contract and launcher normalization; `cargo test -p rustrain-parallel --lib` passed 14/14.
- `afcf091`: restored native C++ Adam step on checkpoint load and added ABI8 smoke assertions; server checkpoint tests passed 2/2.
- ABI9 working tree: selected adapter IDs flow through HTTP, IPC, Rust, and C++; each dynamic tenant owns its Adam clock, checkpoint metadata preserves it, and failed selection restores the complete registry.
- ABI10 working tree: TP-only mode shards each fixed and dynamic adapter's latent rank, all-reduces only activation-level LoRA deltas on a split TP communicator, and keeps replicated MoE layers off the EP communicator.
- `5aa4ea4`: ABI11 stores fixed and dynamic LoRA leaf gradients in FP32 accumulators between micro-batches; fused Adam consumes those accumulators directly, fixed windows use token-weighted numerator/denominator reduction, and explicit abort/failure cleanup clears the pending window. This is FP32 storage/aggregation; the autograd leaf backward remains BF16-typed.
- `ed25daa`: TP rank-aware checkpoint format v3 writes `rank-xxxxx/` manifests and adapter/optimizer shards with topology, global/local shape, latent partition axis, offset, and replica identity metadata. v1/v2 single-rank loading remains compatible; v3 requires the same topology and rank.
- `ff382a0`: native ABI rejects unsupported TP+DP/EP mixtures before training. Dynamic per-tenant batch input under DP is explicitly rejected because the current reduction contract has one aggregate token count; shared batch-1/equal-mask dynamic DP remains the supported contract.
- H20 `123.57.26.97:28004`: ABI8 native smoke passed grouped/fallback parity, GDN, dense/MoE LoRA, dynamic adapters, and step setter validation.
- H20 `123.57.26.97:28004`: ABI9 native smoke passed selected-tenant training with a positive selected update, exactly zero unselected update, independent clocks (`2` vs `1`), and registry preservation after an unknown ID.
- H20 `123.57.26.97:28004`: ABI10 two-rank TP native smoke passed on both ranks for MoE, dense MLP, GDN/linear attention, dynamic multi-LoRA, and selected-tenant isolation. Rank-local LoRA tensors used distinct rank `4` slices for global rank `8`; losses matched and both shards had positive updates.
- H20 `123.57.26.97:28004`: ABI11 single-rank smoke passed FP32 accumulator dtype, two-micro accumulation, NaN/explicit abort cleanup, successful step commit, topology rejection, and dynamic-DP per-tenant batch rejection. ABI11 two-rank TP smoke passed on both ranks for MoE, dense MLP, GDN/linear attention, dynamic multi-LoRA, selected-tenant isolation, and the new guards (`rank_statuses=0,0`).
- Target runtime probe: PyTorch 2.5.1+cu121, ABI0; Transformer Engine, flash-attn, DeepEP, Triton, and DeepSpeed are not importable.

# Decisions During Execution

- Keep TP and EP communicators separate; do not reuse the existing EP `LayerConfig.nccl_comm` for LoRA TP deltas.
- Scope multi-LoRA `n_max` rendezvous files per native context; reusing one filename across sessions can give ranks different chunk schedules and deadlock TP collectives.
- Do not enable Qwen TP/PP/CP by merely relaxing runtime validation.
- Treat Exa/Jina dependency search failures as missing evidence, not as proof that a package is compatible.

# Verification

Passed: `cargo check -p rustrain-qwen3-6 -p rustrain-server -p rustrain-ipc`, `cargo test -p rustrain-server --lib` (6), Qwen unit tests (3), Qwen integration (6), remote ABI8 smoke, remote ABI9 selected-tenant native smoke, remote ABI10 single-rank native smoke, remote ABI10 two-rank TP native smoke, and remote ABI11 single/two-rank native smoke.

Not run: Megatron-style base-model TP, variable-split EP token dispatch, multi-axis TP+DP/EP, PP/CP, cross-topology resharding, a numerical FP32-accumulation oracle against a concatenated batch, and matched Megatron performance benchmark.
