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

Verified: Qwen3.5/3.6 native forward/backward slices, GDN CUDA path, grouped MoE fallback, EP parity against a full-expert reference, variable-split EP A2A with fixed-LoRA data sharding, dense replicated-DP smoke with per-tenant token weighting, TP-only latent-rank-sharded LoRA smoke, dynamic batch logical-step update, ABI11 FP32 gradient storage/aggregation, standard Adam bias correction, per-tenant optimizer-step restore, selected-tenant isolation, same-topology rank-aware checkpointing, and 5D topology mapping.

Not yet verified or implemented: Megatron-style frozen base-weight TP, multi-axis TP+DP/EP, PP/CP, DeepEP/TE prebuilt integration, dynamic multi-LoRA source metadata under sharded A2A, cross-topology checkpoint resharding, and matched Megatron throughput.

# Durable Milestones

- `4e242ee`: added Megatron-style 5D topology contract and launcher normalization; `cargo test -p rustrain-parallel --lib` passed 14/14.
- `afcf091`: restored native C++ Adam step on checkpoint load and added ABI8 smoke assertions; server checkpoint tests passed 2/2.
- ABI9 working tree: selected adapter IDs flow through HTTP, IPC, Rust, and C++; each dynamic tenant owns its Adam clock, checkpoint metadata preserves it, and failed selection restores the complete registry.
- ABI10 working tree: TP-only mode shards each fixed and dynamic adapter's latent rank, all-reduces only activation-level LoRA deltas on a split TP communicator, and keeps replicated MoE layers off the EP communicator.
- `5aa4ea4`: ABI11 stores fixed and dynamic LoRA leaf gradients in FP32 accumulators between micro-batches; fused Adam consumes those accumulators directly, fixed windows use token-weighted numerator/denominator reduction, and explicit abort/failure cleanup clears the pending window. This is FP32 storage/aggregation; the autograd leaf backward remains BF16-typed.
- `ed25daa`: TP rank-aware checkpoint format v3 writes `rank-xxxxx/` manifests and adapter/optimizer shards with topology, global/local shape, latent partition axis, offset, and replica identity metadata. v1/v2 single-rank loading remains compatible; v3 requires the same topology and rank.
- `ff382a0`: native ABI rejects unsupported TP+DP/EP mixtures before training. Dynamic per-tenant DP batches now use an all-reduced per-adapter token-count vector and weighted FP32 LoRA gradient reduction, including grouped expert LoRA.
- `76eace6`: direct native context creation also rejects PP/CP sizes greater than one, so unsupported pipeline/context parallelism cannot silently fall back to replicated single-stage execution.
- `b7050ea`: fixed standard Adam bias correction, separated pure-DP gradient NCCL from MoE activation collectives, made dynamic-MoE DP token weighting numerical, scoped n_max rendezvous by invocation with atomic publication, and added full-expert EP parity plus Adam oracles.
- `bfa3d8e`: added a gated variable-split NCCL dispatch/inverse-combine prototype with differentiable forward/backward collectives; default legacy EP remains unchanged.
- Working tree after `bfa3d8e`: `QWEN36_EP_A2A_SHARDED=1` disjoins EP input rows, all-reduces global fixed-LoRA token numerators, keeps grouped expert LoRA local after A2A, and explicitly rejects dynamic multi-LoRA until source tenant metadata exists.
- H20 `123.57.26.97:28004`: ABI8 native smoke passed grouped/fallback parity, GDN, dense/MoE LoRA, dynamic adapters, and step setter validation.
- H20 `123.57.26.97:28004`: ABI9 native smoke passed selected-tenant training with a positive selected update, exactly zero unselected update, independent clocks (`2` vs `1`), and registry preservation after an unknown ID.
- H20 `123.57.26.97:28004`: ABI10 two-rank TP native smoke passed on both ranks for MoE, dense MLP, GDN/linear attention, dynamic multi-LoRA, and selected-tenant isolation. Rank-local LoRA tensors used distinct rank `4` slices for global rank `8`; losses matched and both shards had positive updates.
- H20 `123.57.26.97:28004`: ABI11 single-rank smoke passed FP32 accumulator dtype, two-micro accumulation, NaN/explicit abort cleanup, successful step commit, standard Adam parameter oracle, TP/DP/EP and PP/CP topology rejection, and dynamic multi-LoRA. ABI11 two-rank TP smoke passed on both ranks for MoE, dense MLP, GDN/linear attention, dynamic multi-LoRA, selected-tenant isolation, invocation-scoped n_max rendezvous, and the new guards (`rank_statuses=0,0`).
- H20 `123.57.26.97:28004`: ABI11 DP2 smoke passed with per-tenant masks `[1,3]` and `[3,1]`: weighted m relative error `2.43e-8`, grouped-expert error `2.27e-8`, v error `7.33e-8`, BF16 Adam delta error `0`, and nonzero gap `7.96e-3` versus the old equal-count formula (`rank_statuses=0,0`).
- H20 `123.57.26.97:28004`: ABI11 EP2 full-expert parity smoke passed with distinct rank-local base and LoRA expert slices; loss, A/B updates, m/v, and standard Adam first-step oracle all matched the rank-local full-expert reference (`rank_statuses=0,0`).
- H20 `123.57.26.97:28004`: gated A2A EP2 smoke passed in legacy, replicated-source, and sharded-source modes (`rank_statuses=0,0`). Sharded mode used token counts `[1,3]`; weighted global loss differed from the full-batch reference by `9.6e-7`, expert m/v maxima were `1.22e-5` / `3.92e-9`, and every standard Adam oracle was zero. One distributed BF16 parameter landed in the adjacent rounding bin (`9.61e-4`), bounded separately by the smoke.
- Target runtime probe: PyTorch 2.5.1+cu121, ABI0; Transformer Engine, flash-attn, DeepEP, Triton, and DeepSpeed are not importable.

# Decisions During Execution

- Keep TP and EP communicators separate; do not reuse the existing EP `LayerConfig.nccl_comm` for LoRA TP deltas.
- Scope multi-LoRA `n_max` rendezvous files per native context; reusing one filename across sessions can give ranks different chunk schedules and deadlock TP collectives.
- Do not enable Qwen TP/PP/CP by merely relaxing runtime validation.
- Treat Exa/Jina dependency search failures as missing evidence, not as proof that a package is compatible.

# Verification

Passed: `cargo check -p rustrain-qwen3-6 -p rustrain-server -p rustrain-ipc` (with the repository PyTorch 2.12.1 host venv), `cargo test -p rustrain-server --lib` (6), Qwen unit tests (3), Qwen integration (6), remote ABI8 smoke, remote ABI9 selected-tenant native smoke, remote ABI10 single-rank native smoke, remote ABI10 two-rank TP native smoke, and remote ABI11 single/TP2/DP2/EP2 native smoke with numerical Adam and parity oracles.

Not run: Megatron-style base-model TP, multi-axis TP+DP/EP, PP/CP, dynamic multi-LoRA source metadata under sharded A2A, cross-topology resharding, and matched Megatron performance benchmark. The target host lacks importable Megatron/Transformer Engine/DeepEP/flash-attn prebuilt packages, so no dependency installation or JIT workaround was used.
