# rustrain

rustrain is an educational Rust LLM training engine. It starts with a small
Qwen-like toy trainer and incrementally adds real Qwen checkpoint loading,
generation, LoRA/SFT building blocks, Megatron-style parallel module concepts,
MoE toy modules, and CUDA smoke tests through `tch-rs`.

This repository is not yet a production trainer. The current code is organized
as milestone-sized, verifiable smokes.

## GPU-Only Verification

All smoke, test, parity, training, and `cargo check` verification runs on Ray
GPU workers submitted through the GPU host. The local machine is only for
editing, git operations, and launching remote jobs. Local CPU smoke is not a
fallback at any stage: do not use it for development feedback, debugging
evidence, or final acceptance.

Run one command on a Ray GPU worker:

```sh
scripts/gpu_run.sh cargo run -- tch-cuda-probe
```

By default `scripts/gpu_run.sh` stages the current working tree into a
temporary Ray worker directory before running the command, so development
feedback comes from the current local edits on GPU, not from an older remote
checkout:

```sh
scripts/gpu_run.sh cargo check
```

Run the focused verification suite:

```sh
scripts/verify_gpu.sh
```

Run the representative 2-GPU distributed verification suite:

```sh
scripts/verify_gpu_distributed.sh
```

Both scripts SSH to `root@192.168.42.106:2222`, submit work to Ray with GPU
resources, enter the remote checkout, and source `scripts/tch_a800_env.sh`
before running project commands. `scripts/gpu_run.sh`, `scripts/verify_gpu.sh`,
and `scripts/verify_gpu_distributed.sh` all stage the current working tree by
default; set `RUSTRAIN_SYNC_TO_WORKER=0` only when you explicitly want to
validate the remote checkout as-is. Do not run `cargo check`,
`cargo test`, train smokes, parity commands, or local CPU checks on the local
machine. Do not run them directly in the plain SSH shell either; that shell does
not expose the GPU devices. If a check is worth running, run it on a Ray GPU
worker.

The preferred remote checkout is:

```sh
/vePFS-Mindverse/user/nolanho/code/rustrain
```

If that shared checkout is not available on the Ray worker,
`scripts/gpu_run.sh` automatically falls back to `/root/rustrain` only when
`RUSTRAIN_SYNC_TO_WORKER=0`. Normal development commands keep the default
`RUSTRAIN_SYNC_TO_WORKER=1`, stage the current working tree into a temporary
worker directory, and do not depend on either remote checkout.

To refresh the bootstrap fallback manually:

```sh
tar --exclude .git --exclude target --exclude runs -cf - . \
  | ssh -p 2222 root@192.168.42.106 'rm -rf /root/rustrain && mkdir -p /root/rustrain && cd /root/rustrain && tar -xf -'
scripts/verify_gpu.sh
```

The A800 config writes runs to `/tmp/rustrain-runs` because the shared project
checkout may be read-only from the worker. If `tch-cuda-probe` reports
`device_count: 0`, the SSH target has not exposed GPU devices and milestone
acceptance is blocked until the runtime is fixed.

## Real Qwen Smokes

The current real-model path targets the Qwen2.5 checkpoint under
`/vePFS-Mindverse/share/huggingface` on the GPU host:

```sh
scripts/gpu_run.sh cargo run -- qwen-logits-parity
```

The full-train smoke command is present, but GPU execution must be used for the
real checkpoint:

```sh
scripts/gpu_run.sh cargo run -- qwen-full-train-smoke
```

## Distributed Launch

A minimal per-node rank launcher is available inside the Ray GPU worker for rank
process management:

```sh
scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 print-launch-env
scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 parallel-dp-rank-smoke --output-dir /tmp/rustrain-runs/launch-dp-rank-smoke/dp
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/nccl-all-reduce-smoke nccl-all-reduce-rank-smoke --output-dir /tmp/rustrain-runs/nccl-all-reduce-smoke/ranks
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/nccl-dp-gradient-smoke nccl-dp-gradient-rank-smoke --output-dir /tmp/rustrain-runs/nccl-dp-gradient-smoke/ranks
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/tch-dp-gradient-smoke tch-dp-gradient-rank-smoke --output-dir /tmp/rustrain-runs/tch-dp-gradient-smoke/ranks
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/tch-trainer-dp2-launch train --config configs/tch_smoke_cuda_dp2.toml
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=300 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-dp-gradient-smoke-fp32 qwen-dp-gradient-rank-smoke --dtype fp32 --output-dir /tmp/rustrain-runs/qwen-dp-gradient-smoke-fp32/ranks
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=300 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-dp-gradient-smoke-steps3 qwen-dp-gradient-rank-smoke --dtype fp32 --steps 3 --learning-rate 1.0 --output-dir /tmp/rustrain-runs/qwen-dp-gradient-smoke-steps3/ranks
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=600 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-dp-adamw-smoke qwen-session-dp-rank-smoke --dtype fp32 --steps 2 --learning-rate 0.000001 --output-dir /tmp/rustrain-runs/qwen-session-dp-adamw-smoke/ranks
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=600 cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-trainer-dp2 train --config configs/qwen_session_dp2.toml
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/qwen-session-trainer-tp2 train --config configs/qwen_session_tp2.toml
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/ep-rank-local-smoke parallel-ep-rank-smoke --output-dir /tmp/rustrain-runs/ep-rank-local-smoke/ranks
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh cargo run -- launch --nproc-per-node 2 --output-dir /tmp/rustrain-runs/ep-nccl-smoke parallel-ep-nccl-rank-smoke --output-dir /tmp/rustrain-runs/ep-nccl-smoke/ranks
scripts/verify_gpu_distributed.sh
```

The launcher sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`,
`MASTER_ADDR`, and `MASTER_PORT`, captures rank-local logs, and writes a
`launch-summary.json`. When `CUDA_VISIBLE_DEVICES` is set, it rejects launches
that request more local ranks than visible GPUs and records the assigned visible
GPU token plus ordinal for each rank in both `launch-summary.json` and
`print-launch-env`. `scripts/gpu_run.sh` defaults to one Ray GPU; set
`RUSTRAIN_RAY_NUM_GPUS=2` for two-rank GPU collective smokes. Minimal NCCL f32
all-reduce, toy DP gradient all-reduce, and `tch` autograd DP gradient smokes
exist, including a `trainer::train` config path for tiny `tch` DP=2 and a
focused real-Qwen layer0 attention DP gradient-signature smoke. A focused
real-Qwen TP=2 linear smoke shards the real layer0 attention `q_proj`, `k_proj`,
`v_proj`, and `o_proj` output dimensions across two CUDA ranks and verifies the
gathered shard outputs match the full linear projections. A companion TP=2
attention smoke shards Q/K/V heads, applies rank-local attention, sums O-proj
input-column contributions on rank0, and verifies parity against full layer0
attention; a second attention smoke uses NCCL all-reduce for the same summed
output contribution parity on every rank. A matching TP=2 MLP smoke shards
gate/up intermediate rows and
down-proj input columns, sums rank-local contributions on rank0, and verifies
parity against the full layer0 MLP. A second MLP smoke uses NCCL all-reduce to
sum those rank-local output contributions on every rank and checks the same
full-MLP parity. The `qwen_trainable_session` trainer entry also accepts
`tensor_model_parallel_size = 2` for a focused layer0 TP config path: it checks
attention and MLP NCCL output parity, then runs focused attention and MLP shard
backward/update smokes with positive q/k/v/o plus gate/up/down shard gradients
and lower post-update global MSE losses. It also runs a fused layer0 TP smoke
that all-reduces attention output before the post-attention norm, all-reduces
MLP contributions, verifies full layer0 output parity, and checks a
loss-reducing joint attention+MLP shard update. It now also runs a focused
causal-LM train-step smoke over a real token batch: rank-local layer0 TP
contributions are all-reduced, later layers/final norm/tied LM head compute a
real causal loss, and an explicit output-gradient bridge verifies q/k/v/o plus
gate/up/down shard gradients and a loss-reducing shard update. The smoke writes
rank-owned focused TP shard manifests for layer0 attention/MLP tensors under
the shared `rustrain.qwen_sharded.v1` schema, with zero Adam slots as
checkpoint-contract evidence, and restores those rank-owned shards to reproduce
the focused fused layer0 output plus the next focused shard update within
tolerance. Real production tensor-parallel Qwen training is not implemented
yet; the remaining TP gap is full-parameter TP backward/update, autograd-aware
production collectives, and trainer-owned sharded checkpoint resume.
The Qwen DP smoke writes a rank0-only JSON checkpoint manifest after gradient
sync succeeds; non-rank0 summaries record the same checkpoint path but do not
write it.
It also all-reduces full CUDA gradient tensors for the focused layer0 attention
parameter set, applies configurable multi-step averaged SGD on each rank, and
checks that the global post-update loss is lower. This is still a focused
layer0 attention DP smoke, not trainer-owned full-Qwen distributed training.
The `QwenTrainableSession` path also has a representative DP rank smoke: it
runs rank-local Qwen forward/backward on CUDA, all-reduces 13 representative
layer0/norm/MLP gradients with NCCL, applies multi-step averaged AdamW updates,
checks global loss improvement, writes rank0 delta and AdamW optimizer
safetensors, and verifies next-step resume parity from that rank0 checkpoint.
The same representative path is wired through
`train --config configs/qwen_session_dp2.toml`; bf16 coverage for the same
representative DP path is available through
`configs/qwen_session_dp2_bf16.toml`. The session registry now expands
trainable transformer layers from `model.trainable_layers` and defaults to
`[0]`; `configs/qwen_session_single_layers01.toml` verifies a layer0+layer1
single-GPU path with 26 trainable tensors, and
`configs/qwen_session_single_layers03.toml` verifies a layer0-layer3
single-GPU path with 50 trainable tensors. The same layer0-layer3 path also
has bf16 coverage through `configs/qwen_session_single_layers03_bf16.toml`.
Toy MoE has an explicit single-rank smoke command and verifier:
`cargo run -- moe-smoke` prints a JSON summary for TinyMoe and DeepSeek-style
toy MoE stats, and `scripts/verify_moe_smoke_worker.sh` asserts expert load,
load-balance loss, activated/total parameter counts, and output summaries on a
Ray GPU worker. A CUDA `tch-rs` autograd MoE smoke is also available through
`cargo run -- tch-moe-smoke`; `scripts/verify_tch_moe_smoke_worker.sh` checks
that router and expert tensors receive gradients, the task loss decreases,
activated parameters remain below total parameters, and model plus AdamW
optimizer safetensors reload reproduces the next train step on a Ray GPU worker.
It also writes a `rustrain.tch_moe.v1` manifest covering model tensor shapes,
AdamW slot names, checkpoint paths, and completed step metadata.
Expert-parallel coverage is still toy-sized, but it now has three real
launcher-backed two-rank paths: `parallel-ep-rank-smoke` verifies rank-local
expert ownership and token coverage, `parallel-ep-nccl-rank-smoke` builds
rank-local expert output tensors on CUDA, combines dense contributions with
NCCL all-reduce, verifies an explicit gradient-bridge AdamW update lowers a
tiny MSE target, writes rank-owned expert scale and optimizer safetensors under
`rustrain.ep_sharded.v1`, and verifies reload plus next-step parity from those
rank-owned expert checkpoints. `parallel-ep-sparse-rank-smoke` verifies a
toy sparse NCCL send/recv dispatch and combine path: source ranks dispatch
routed token payloads to expert-owner ranks, owner ranks compute local expert
outputs, and outputs are sent back to source ranks for parity with the
single-process reference; it also reports the global expert load and toy
load-balancing auxiliary loss on every rank. The same sparse smoke now sends
output gradients back to expert-owner ranks, accumulates rank-local owned
expert scale gradients, applies a smoke-only SGD update, and verifies a second
sparse forward lowers the MSE loss. It also writes rank-owned toy expert
model/optimizer safetensors under `rustrain.ep_sharded.v1`, reloads the owned
expert scales and optimizer tensors, verifies reloaded sparse-forward loss
parity, and checks the next sparse update from reloaded state matches the
continuous next update. The sparse verifier checks the rank-owned manifest
contract, safetensors paths, optimizer slot names, shard shapes, dtype, and
expert-model-parallel partition metadata. A focused two-rank CUDA
`parallel-ep-tch-moe-rank-smoke` now uses the same sparse send/recv plan with
rank-owned `tch-rs` expert up/down MLP weights, bridges returned output
gradients through real autograd, verifies positive expert gradients and
loss-reducing AdamW updates, then writes two rank-owned expert shards under
`rustrain.ep_sharded.v1` and verifies reload plus next-step parity. Production
EP now also has a focused trainer-entry config, `configs/tch_moe_ep2.toml`,
which runs the same two-rank path through `train --config` and writes a shared
`rustrain.ep_sharded.v1` global manifest that embeds the two rank manifests,
the EP topology, global step, consumed sample/token counts, dtype, optimizer,
and scheduler metadata. Full production EP is still open: autograd-aware sparse
collectives, trainer-owned production MoE integration, and full expert
optimizer/checkpoint ownership are not implemented.
Full production Qwen trainer ownership, full real data streaming, and
production-grade sharded checkpoint ownership remain open.

## Current Major Gaps

- G5 representative checkpoint resume is implemented for the Qwen full-train
  smoke: manifest-driven reload restores delta tensors plus Adam optimizer
  slots and proves next-step parity. The representative Qwen DP trainer path
  also has rank0 delta/optimizer artifacts and next-step resume parity. A
  production sharded checkpoint schema is defined and validated, and the
  representative DP smoke writes rank-owned shard manifests and verifies
  rank-local sharded reload plus next-step resume parity. The focused TP=2
  layer0 smoke writes rank-owned shards and verifies focused restore output
  plus next-update parity from the global sharded manifest. The single-GPU
  session trainer also has a tokenizer-backed JSONL batch/resume verifier at
  `configs/qwen_session_single_sft.toml`; the DP=2 session trainer has the
  same JSONL batch path at `configs/qwen_session_dp2_sft.toml`. Production
  checkpoint/resume over full real data streams remains open.
- C4/G1 trainer-owned Qwen training is still incomplete, but the full-train
  smoke now uses a reusable `QwenTrainableSession` surface, and representative
  single-GPU plus DP=2 config paths are wired through `train --config`. The
  single-GPU path now respects configured `max_steps` and reports step losses.
  It can also resume from its saved delta manifest through `--resume-from` and
  reports throughput, gradient norm, RSS, and GPU memory metrics. The session
  registry can now expand configured trainable layer sets instead of being
  fixed to layer0, and `configs/qwen_session_single_sft.toml` feeds
  tokenizer-backed instruction JSONL batches through the same trainer path.
  `configs/qwen_session_dp2_sft.toml` extends that representative JSONL batch
  path through the 2-rank DP trainer.
  Full model/data/checkpoint trainer ownership remains open.
- G6 trainer-level real SFT data now has minimal Qwen LoRA SFT config paths:
  `train --config configs/qwen_lora_sft.toml` loads tokenizer-backed
  instruction JSONL batches, trains configured attention and MLP LoRA targets,
  reloads the adapter, and supports `--resume-from` for saved adapters. The
  trainer path now verifies the saved adapter through full-Qwen forward logits,
  greedy generation reload parity, and focused merge/unmerge parity. It also
  uses the trainer scheduler and grad clipping knobs, logs `eval_every` step
  eval history, applies seeded deterministic dataset ordering, reports dataset
  sample/token/mask summaries, and has a bf16 variant at
  `configs/qwen_lora_sft_bf16.toml`.
  Production data loading and arbitrary-module LoRA injection are still open.
- Real Qwen module-level LoRA now uses a target-layer/module registry for
  configured attention and MLP projection modules; trainer-owned full-model LoRA
  injection is not done yet. The current Qwen LoRA SFT config exposes rank,
  alpha, target layers, and target modules for that focused path.
- KV-cache greedy parity, cached sampling parity, and Python Transformers
  cached-generation parity are implemented.
- G4 launcher process management plus NCCL scalar, toy DP gradient, `tch`
  autograd DP=2 trainer smokes, focused multi-step Qwen layer0 attention DP,
  representative `QwenTrainableSession` DP smokes, and a representative Qwen
  DP `train --config` path with rank0 checkpoint/resume parity exist. Focused
  TP=2 attention/MLP NCCL output parity plus attention/MLP shard
  backward/update, fused layer0 TP, focused causal-LM train-step, and focused
  TP sharded-manifest smokes also run through
  `train --config configs/qwen_session_tp2.toml`; that focused TP path restores
  rank-owned shards through the global sharded manifest and checks fused layer0
  output plus next-update parity. EP has toy rank-local and CUDA/NCCL combine
  smokes, a toy sparse NCCL send/recv dispatch/combine/backward-update smoke,
  sparse rank-owned toy expert checkpoint reload plus next-step parity, and
  dense rank-owned toy expert scale/optimizer checkpoint reload plus next-step
  parity, but not production MoE or trainer-owned expert state. Real production
  distributed training is still missing: full Qwen model/data integration,
  full-parameter production TP backward/update, production collectives,
  production EP, and production sharded checkpoint ownership are not yet
  implemented.
- Production distributed checkpoint rules are documented in
  [docs/checkpoints.md](docs/checkpoints.md), with a validated
  `rustrain.qwen_sharded.v1` manifest schema and representative rank-owned
  writer/restore/next-step resume smoke. Full production sharded resume over
  external streaming real data remains open.
- Trainer production basics such as scheduler, grad clipping, RSS memory
  metrics, and Ray-worker GPU memory reporting are implemented for toy/tch
  paths; real tokenizer-backed padded LoRA SFT batching is wired through
  minimal fp32/bf16 Qwen trainer configs with scheduler and grad clipping. The
  tiny `tch` CUDA path, representative Qwen session single/DP paths, the
  layer0-layer3 Qwen session path, and Qwen LoRA SFT resume verifier now have
  explicit bf16 compute-policy coverage.
  Mixed precision over future full production Qwen model/data paths remains
  future work.

Internal planning details live in `_internal_docs/TODO.md`; that directory is
ignored by git.
