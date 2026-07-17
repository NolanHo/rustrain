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

Verified: Qwen3.5/3.6 native forward/backward slices, GDN CUDA path, grouped MoE fallback, EP parity against a full-expert reference, variable-split EP A2A with fixed-LoRA data sharding, dense replicated-DP smoke with per-tenant token weighting, latent-rank and projection-aware LoRA TP, frozen full-attention/GDN/dense-MLP base-weight TP, orthogonal dense TP x DP process groups, vocabulary-parallel embedding/LM-head/distributed cross entropy, composable LoRA-only TP x EP process groups with expert-local and dense-replicated gradient semantics, dynamic batch logical-step update, FP32 gradient storage/aggregation, standard Adam bias correction, per-tenant optimizer-step restore, selected-tenant isolation, checkpoint v4 projection layouts and fixed-slot identities, and 5D topology mapping.

Not yet verified or implemented: the three-axis TP x EP x expert-DP combination, expert-tensor-parallel base-weight sharding, MLP-targeted LoRA under base TP, fused QKV/gate-up and sequence parallel, PP/CP, DeepEP/TE/FLA prebuilt integration, server-side source-sharded dynamic dispatch, TP x EP checkpoint metadata/resharding, heterogeneous dynamic adapter signatures, dynamic+MTP objective normalization, cross-topology checkpoint resharding, old-v3 attention checkpoint migration, and matched Megatron throughput. The server rejects MoE TP x EP until request source coordinates and checkpoint ownership are represented. Native direct dynamic source metadata under sharded A2A is implemented and full-reference smoke-tested.

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
- Working tree after `bfa3d8e`: `QWEN36_EP_A2A_SHARDED=1` disjoins EP input rows, all-reduces global fixed-LoRA token numerators, and keeps grouped expert LoRA local after A2A. Native dynamic multi-LoRA now reuses the transported source row for tenant mapping, all-reduces per-tenant counts, skips zero-global-token tenants, and rejects ordinary single-adapter entry points while a dynamic registry is active.
- H20 `123.57.26.97:28004`: ABI8 native smoke passed grouped/fallback parity, GDN, dense/MoE LoRA, dynamic adapters, and step setter validation.
- H20 `123.57.26.97:28004`: ABI9 native smoke passed selected-tenant training with a positive selected update, exactly zero unselected update, independent clocks (`2` vs `1`), and registry preservation after an unknown ID.
- H20 `123.57.26.97:28004`: ABI10 two-rank TP native smoke passed on both ranks for MoE, dense MLP, GDN/linear attention, dynamic multi-LoRA, and selected-tenant isolation. Rank-local LoRA tensors used distinct rank `4` slices for global rank `8`; losses matched and both shards had positive updates.
- H20 `123.57.26.97:28004`: ABI11 single-rank smoke passed FP32 accumulator dtype, two-micro accumulation, NaN/explicit abort cleanup, successful step commit, standard Adam parameter oracle, TP/DP/EP and PP/CP topology rejection, and dynamic multi-LoRA. ABI11 two-rank TP smoke passed on both ranks for MoE, dense MLP, GDN/linear attention, dynamic multi-LoRA, selected-tenant isolation, invocation-scoped n_max rendezvous, and the new guards (`rank_statuses=0,0`).
- H20 `123.57.26.97:28004`: ABI11 DP2 smoke passed with per-tenant masks `[1,3]` and `[3,1]`: weighted m relative error `2.43e-8`, grouped-expert error `2.27e-8`, v error `7.33e-8`, BF16 Adam delta error `0`, and nonzero gap `7.96e-3` versus the old equal-count formula (`rank_statuses=0,0`).
- H20 `123.57.26.97:28004`: ABI11 EP2 full-expert parity smoke passed with distinct rank-local base and LoRA expert slices; loss, A/B updates, m/v, and standard Adam first-step oracle all matched the rank-local full-expert reference (`rank_statuses=0,0`).
- H20 `123.57.26.97:28004`: gated A2A EP2 smoke passed in legacy, replicated-source, and sharded-source modes (`rank_statuses=0,0`). Sharded mode used token counts `[1,3]`; weighted global loss differed from the full-batch reference by `9.6e-7`, expert m/v maxima were `1.22e-5` / `3.92e-9`, and every standard Adam oracle was zero. One distributed BF16 parameter landed in the adjacent rounding bin (`9.61e-4`), bounded separately by the smoke.
- Target runtime probe: PyTorch 2.5.1+cu121, ABI0; Transformer Engine, flash-attn, DeepEP, Triton, and DeepSpeed are not importable.
- H20 `123.57.26.97:28004`: `native_ep_bench.cpp` fresh ABI0 benchmark (`seq=128, hidden=256, experts=8, intermediate=256, warmup=2, iters=10`) passed legacy and sharded A2A with `rank_statuses=0,0`. Legacy median was about `5.47 ms` / `46.4k processed tokens/s` (`23.2k unique tokens/s`), while sharded A2A was about `6.72 ms` / `37.8k processed and unique tokens/s`. This is a synthetic native baseline, not Megatron-LM parity.
- H20 `123.57.26.97:28004`: fresh ABI1 dynamic sharded native smoke passed on both ranks (`rank_statuses=0,0`) against a full-expert reference with complementary source masks. Dynamic grouped-expert parameter/m/v maxima were `1.53e-5` / `4.88e-5` / `5.75e-8`; it also exercised a third tenant with zero global target tokens, clocks `[2,2,0]`, and explicit rejection of ordinary `train_step`. The server path still broadcasts replicated source batches, and dynamic+MTP is explicitly rejected until its two objective denominators are separated.
- H20 `123.57.26.97:28004`: fresh ABI13 dense base-MLP TP2 smoke passed on both ranks. CLI and server use one CPU sharding helper for gate/up rows and matching down columns; the per-context native flag validates local shapes without process-global env state. C++ reduces the row-parallel output and all-reduces column-parallel input dgrad. Eval/train loss differed from the replicated full-weight reference by `1.84e-5`; FP32 LoRA gradient-accumulator maxima were `3.05e-5` / `4.58e-5`, and the largest post-Adam LoRA slice difference was `9.31e-9`. The smoke also rejects MLP LoRA targets, prevents incompatible TP state transitions, and covers eval-to-train cache invalidation.
- ABI14 working tree: frozen full attention shards Q/K/V output heads and O input columns, while fixed and selected dynamic Q/K/V/O LoRA use projection-aware layouts. Replicated-side gradients are summed once at the optimizer boundary. Latent-rank LoRA now applies copy-to-TP-region before its local A/B path so a later sharded branch sums input dgrad before it reaches preceding replicated layers. Checkpoint v4 records the layout, replicated tensor geometry, and exact fixed native slot identity; v3 remains loadable only for latent-rank tensors and attention migration is explicitly rejected.
- H20 target: ABI14 4Q/2KV-head full-attention TP2 smoke passed Q/K/V/O fixed and selected-dynamic full-reference oracles on both ranks. Fixed eval/loss differed by `4.40e-4`; Q/K/V-B and O-A gradient maxima were `1.83e-4` / `1.14e-5` / `3.05e-4` / `1.83e-4`, FP32 m/v maxima were `1.53e-5` / `6.63e-9`, and the local standard-Adam formula error was below `3.73e-9`. Selected-dynamic loss differed by `6.82e-5` and the largest Q/K/V/O parameter difference was `3.05e-5`.
- H20 target: ABI14 two-layer GDN latent-rank TP2 full-reference smoke passed on both ranks. Eval/loss differed by `5.46e-5`; A/B gradient maxima were `9.54e-7` / `4.77e-7`, FP32 m/v maxima were `4.77e-8` / `3.71e-14`, and the local standard-Adam formula error was zero. ABI14 general single-GPU native smoke and dense-MLP TP2 regression also passed.
- `99aecef`: ABI16 introduced explicit orthogonal TP/DP groups, dense TP x DP server batch sharding, distributed adapter-registry consensus, and a world4 native TPDP oracle. The H20 `TP2 x DP2` smoke passed fixed and dynamic LoRA with unequal DP token counts and GDN TP2 remained green.
- ABI17 working tree: CLI and server shard Qwen3.5 tied embeddings and Qwen3.6 untied embedding/LM-head rows on CPU before device transfer. C++ performs masked local embedding plus TP SUM and a two-pass distributed cross entropy with TP MAX/SUM statistics and TP-summed frozen-head dgrad. MTP remains explicitly rejected because its weights are not sharded.
- H20 target: ABI17 TP2 GDN smoke passed tied fixed-LoRA and untied dynamic multi-LoRA against full-vocabulary references. Fixed loss differed by `5.90e-4`, dynamic loss by `1.16e-3`, FP32 m/v maxima were `1.07e-5` / `1.22e-9`, and the standard Adam formula error was zero. The world4 `dp-tp` rank-order oracle used TP groups `[0,2]` and `[1,3]`, proving vocabulary ranges follow explicit `tp_rank`; local loss differed by at most `2.31e-3`, with FP32 m/v maxima `3.44e-5` / `1.96e-8`.
- H20 target: ABI17 real-vocabulary GDN benchmark (`batch=1`, `seq=128`, `hidden=2048`, `layers=3`, `vocab=248320`, warmup `2`, iterations `10`) recorded single-rank p50 `24.92 ms`, `5.10k model tokens/s`, and `3.33 GiB` peak resident memory. Vocab-parallel TP2 recorded p50 `23.17 ms`, `5.48k model tokens/s`, and `2.80 GiB` per-rank peak resident memory. The previous replicated-vocabulary ABI16 TP2 baseline was about `24.25 ms` and `4.22 GiB` per rank. This is a native scaling benchmark, not matched Megatron-LM parity.
- ABI18 working tree: native topology owns independent TP, EP, and expert-DP communicators and validates `WORLD_SIZE=TP_SIZE*EP_SIZE*DP_SIZE`. Dense LoRA parameters synchronize and reduce over EP and expert-DP replicas; routed expert parameters remain local to EP and reduce only over expert-DP. TP-latent expert LoRA copies its input into the TP autograd region so backward sums the input gradient exactly once. CLI weight loading composes attention/GDN/vocabulary TP shards with expert EP shards; the server explicitly rejects the source-sharding contract it cannot yet represent.
- H20 target: ABI18 TP2 x EP2 sharded-A2A smoke passed fixed Q and grouped expert gate-up/down LoRA against a full-model reference with unequal EP token counts. Standard Adam parameter oracles were zero on all four ranks; fixed FP32 m/v maxima were `1.01e-4` / `3.08e-9`. Selected dynamic multi-LoRA produced positive selected updates, exactly zero unselected updates, and clocks `[1,0]`; invalid topology and MTP guards fired.
- H20 target: ABI18 regressions passed TP2 x DP2, GDN TP2 fixed/dynamic, and pure EP2 sharded A2A fixed/dynamic tests. GDN fixed/dynamic FP32 m/v maxima remained within `1.07e-5` / `1.22e-9`, and all standard Adam formula errors were zero. Pure EP fixed expert parameter differences were bounded by BF16 (`9.62e-4` maximum) with m/v maxima `1.22e-5` / `3.92e-9`.

# Megatron-LM Performance Gap

ABI18 establishes correct process groups and update ownership, but it does not
close the MoE throughput gap. Base expert weights are replicated across TP
ranks instead of using expert tensor parallelism. The native dispatcher still
iterates routing slots and local experts, launches variable-split A2A work on
the current stream, and performs TP LoRA reductions around individual expert
paths. It lacks Megatron-style fused permutation, capacity-aligned dispatch,
communication overlap, grouped expert GEMM scheduling across ETP, and inverse
dispatch fusion. These are the next performance slice; the current evidence is
correctness and regression evidence, not a matched throughput claim.

# Decisions During Execution

- Keep TP and EP communicators separate; do not reuse the existing EP `LayerConfig.nccl_comm` for LoRA TP deltas.
- Publish multi-LoRA `n_max` directly from rank 0 with `ncclBroadcast`; filesystem rendezvous can reuse stale files across process restarts and give ranks different chunk schedules.
- Do not enable Qwen TP/PP/CP by merely relaxing runtime validation.
- Treat dense base-MLP TP as one accepted slice, not full model TP. Until projection-aware LoRA collectives exist, reject gate/up/down LoRA targets instead of applying the replicated-projection reduction rule to disjoint output shards.
- For frozen full-attention TP, use Q/K/V output-head sharding and O input-column sharding. Q/K/V replicate LoRA A and shard B; O shards A and replicates B. Sum only the replicated-side gradient at the optimizer boundary so the activation collective is not duplicated.
- Do not reinterpret v3 attention LoRA tensors as projection-aware shards. Their latent-rank geometry is incompatible, so require v4 for Q/K/V/O resume and keep v3 compatibility for latent-rank-only modules.
- Treat Exa/Jina dependency search failures as missing evidence, not as proof that a package is compatible.

# Verification

Passed: `cargo check -p rustrain-core -p rustrain-qwen3-6 -p rustrain-server` (with the repository host venv), focused runtime/topology unit tests, remote ABI8-ABI14 native smokes listed above, ABI16 TP2 x DP2 and GDN TP2 smokes, ABI17 TP2 tied/untied vocabulary parity plus world4 custom-rank-order TPDP smokes, and ABI18 TP2 x EP2, TP2 x DP2, GDN TP2, and pure EP2 sharded-A2A native smokes.

Not run: TP x EP x expert-DP, PP/CP, server-side source-sharded dynamic dispatch, TP x EP checkpoint resume, heterogeneous dynamic adapter signatures, dynamic+MTP, cross-topology resharding, and a matched Megatron performance benchmark. No dependency installation or JIT workaround was used for ABI18; the implementation uses the existing prebuilt PyTorch ABI1 and NCCL runtime.
