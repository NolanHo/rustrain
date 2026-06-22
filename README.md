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

Run the focused single-GPU verification suite:

```sh
scripts/verify_gpu.sh
```

Run the representative 2-GPU distributed verification suite:

```sh
scripts/verify_gpu_distributed.sh
```

Both scripts SSH to `root@192.168.42.106:2222`, submit work to Ray with GPU
resources, enter the remote checkout, and source `scripts/tch_a800_env.sh`
before running project commands. The focused suite uses the default one-GPU Ray
worker path; DP/TP/EP and other two-rank checks live in
`scripts/verify_gpu_distributed.sh`, which reserves two Ray GPUs for each
command. `scripts/gpu_run.sh`, `scripts/verify_gpu.sh`, and
`scripts/verify_gpu_distributed.sh` all stage the current working tree by
default; set `RUSTRAIN_SYNC_TO_WORKER=0` only when you explicitly want to
validate the remote checkout as-is. Do not run `cargo check`, `cargo test`,
train smokes, parity commands, or local CPU checks on the local machine. Do not
run them directly in the plain SSH shell either; that shell does not expose the
GPU devices. If a check is worth running, run it on a Ray GPU worker.

Current cluster defaults:

- Ray head: `192.168.42.141:6379` (`mint-head`).
- SSH submission/GPU driver host: `root@192.168.42.106:2222`
  (`mint-driver`).

Set `RUSTRAIN_RAY_ADDRESS` only when intentionally targeting a different Ray
cluster. If the shared Ray head file is unavailable, `scripts/gpu_run.sh` falls
back to `192.168.42.141:6379`; it does not fall back to local or plain-SSH
execution.

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
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 bash scripts/verify_qwen_session_dp2_layers01_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers01_bf16.toml RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 bash scripts/verify_qwen_session_dp2_layers01_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers03.toml RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=49 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.3.self_attn.q_proj.weight,model.layers.3.mlp.down_proj.weight,model.norm.weight bash scripts/verify_qwen_session_dp2_layers01_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers03_bf16.toml RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=49 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.3.self_attn.q_proj.weight,model.layers.3.mlp.down_proj.weight,model.norm.weight bash scripts/verify_qwen_session_dp2_layers01_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers07.toml RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=97 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.7.self_attn.q_proj.weight,model.layers.7.mlp.down_proj.weight,model.norm.weight bash scripts/verify_qwen_session_dp2_layers01_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers07_bf16.toml RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=97 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.7.self_attn.q_proj.weight,model.layers.7.mlp.down_proj.weight,model.norm.weight bash scripts/verify_qwen_session_dp2_layers01_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers01_sft.toml RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 bash scripts/verify_qwen_session_dp2_layers01_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers03_sft.toml RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=49 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.3.self_attn.q_proj.weight,model.layers.3.mlp.down_proj.weight,model.norm.weight RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 bash scripts/verify_qwen_session_dp2_layers01_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers07_sft.toml RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=97 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.7.self_attn.q_proj.weight,model.layers.7.mlp.down_proj.weight,model.norm.weight RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 bash scripts/verify_qwen_session_dp2_layers01_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers01_sft_bf16.toml RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 bash scripts/verify_qwen_session_dp2_layers01_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers03_sft_bf16.toml RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=49 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.3.self_attn.q_proj.weight,model.layers.3.mlp.down_proj.weight,model.norm.weight RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 bash scripts/verify_qwen_session_dp2_layers01_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_QWEN_SESSION_DP_LAYERS_CONFIG=configs/qwen_session_dp2_layers07_sft_bf16.toml RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=97 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.7.self_attn.q_proj.weight,model.layers.7.mlp.down_proj.weight,model.norm.weight RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 bash scripts/verify_qwen_session_dp2_layers01_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR=/tmp/rustrain-runs/qwen-session-dp2-layers01-sft-resume-base RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR=/tmp/rustrain-runs/qwen-session-dp2-layers01-sft-resume-continue RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_layers01_sft.toml RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=25 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.1.self_attn.q_proj.weight,model.layers.1.mlp.down_proj.weight,model.norm.weight bash scripts/verify_qwen_session_dp2_resume_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR=/tmp/rustrain-runs/qwen-session-dp2-layers01-sft-bf16-resume-base RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR=/tmp/rustrain-runs/qwen-session-dp2-layers01-sft-bf16-resume-continue RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_layers01_sft_bf16.toml RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=25 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.1.self_attn.q_proj.weight,model.layers.1.mlp.down_proj.weight,model.norm.weight bash scripts/verify_qwen_session_dp2_resume_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR=/tmp/rustrain-runs/qwen-session-dp2-layers03-sft-resume-base RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR=/tmp/rustrain-runs/qwen-session-dp2-layers03-sft-resume-continue RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_layers03_sft.toml RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=49 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.3.self_attn.q_proj.weight,model.layers.3.mlp.down_proj.weight,model.norm.weight bash scripts/verify_qwen_session_dp2_resume_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR=/tmp/rustrain-runs/qwen-session-dp2-layers07-sft-resume-base RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR=/tmp/rustrain-runs/qwen-session-dp2-layers07-sft-resume-continue RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_layers07_sft.toml RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=97 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.7.self_attn.q_proj.weight,model.layers.7.mlp.down_proj.weight,model.norm.weight bash scripts/verify_qwen_session_dp2_resume_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 bash scripts/verify_qwen_session_dp2_sft_max_samples_resume_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR=/tmp/rustrain-runs/qwen-session-dp2-layers03-sft-bf16-resume-base RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR=/tmp/rustrain-runs/qwen-session-dp2-layers03-sft-bf16-resume-continue RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_layers03_sft_bf16.toml RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=49 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.3.self_attn.q_proj.weight,model.layers.3.mlp.down_proj.weight,model.norm.weight bash scripts/verify_qwen_session_dp2_resume_worker.sh
RUSTRAIN_RAY_NUM_GPUS=2 scripts/gpu_run.sh env RUSTRAIN_LAUNCH_TIMEOUT_SECS=900 RUSTRAIN_DISTRIBUTED_BASE_OUTPUT_DIR=/tmp/rustrain-runs/qwen-session-dp2-layers07-sft-bf16-resume-base RUSTRAIN_DISTRIBUTED_RESUME_OUTPUT_DIR=/tmp/rustrain-runs/qwen-session-dp2-layers07-sft-bf16-resume-continue RUSTRAIN_QWEN_SESSION_DP_CONFIG=configs/qwen_session_dp2_layers07_sft_bf16.toml RUSTRAIN_EXPECTED_QWEN_COMPUTE_KIND=bf16 RUSTRAIN_EXPECTED_DATASET_ORDER_SEED=777 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_TENSORS=97 RUSTRAIN_EXPECTED_QWEN_DP_TRAINABLE_NAMES=model.layers.0.self_attn.q_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.7.self_attn.q_proj.weight,model.layers.7.mlp.down_proj.weight,model.norm.weight bash scripts/verify_qwen_session_dp2_resume_worker.sh
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
gathered shard outputs match the full linear projections. The focused TP
verifiers also require every rank summary to report the same resolved complete
Qwen checkpoint path, including `config.json`, `tokenizer.json`, and
`model.safetensors`; this keeps the public legacy default
`/vePFS-Mindverse/share/huggingface/Qwen2.5-0.5B-Instruct` usable on workers
that only expose the HuggingFace `hub/` snapshot layout. A companion TP=2
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
the shared `rustrain.qwen_sharded.v1` schema. Its optimizer safetensors contain
first-step AdamW m/v smoke slots for the TP row/column shards, while the
replicated norm smoke slots remain zero. The smoke restores those rank-owned
shards to reproduce the focused fused layer0 output plus the next focused shard
update within tolerance. The TP verifier also checks the global manifest
identity, progress/provenance defaults, embedded rank manifests, every declared
rank-owned model shard plus AdamW slot shape, and non-zero optimizer slots for
TP-owned trainable shards in the written safetensors files. The TP verifiers
also check the AdamW first-step slot formulas for focused causal gradients:
`adam_m.sum == (1 - beta1) * grad.sum` and
`adam_v.sum == (1 - beta2) * grad_norm^2`. The external TP resume verifier
repeats those checks for both the base manifest and the resumed launch manifest
while verifying restore and next-update parity.
Real production tensor-parallel Qwen training is not implemented yet; the
remaining TP gap is full-parameter TP backward/update, autograd-aware
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
`configs/qwen_session_single_layers07.toml` extends the single-GPU
representative trainer path to layer0-layer7 with 98 trainable tensors,
including the tied embedding for the single-rank path.
`configs/qwen_session_single_layers07_bf16.toml` verifies the same
single-GPU layer0-layer7 path under bf16 compute.
`configs/qwen_session_single_layers07_sft.toml` extends the single-GPU JSONL
SFT resume path to layer0-layer7 with 98 trainable tensors and manifest-backed
data cursor continuity.
`configs/qwen_session_single_layers07_sft_bf16.toml` verifies the same
single-GPU layer0-layer7 JSONL SFT resume path under bf16 compute.
Single-GPU Qwen delta manifests and Qwen LoRA SFT adapter manifests also record
the resolved complete Qwen checkpoint in `base_model_path`; their verifiers
reject manifests whose recorded checkpoint is missing `config.json`,
`tokenizer.json`, or `model.safetensors`.
`configs/qwen_session_dp2_layers01.toml` extends the representative DP=2
trainer path to layer0+layer1. Its verifier expects 25 trainable tensors
because the representative DP path intentionally excludes the tied embedding
and trains the configured transformer layers plus final norm.
`configs/qwen_session_dp2_layers01_bf16.toml` verifies the same 25-tensor
layer0+layer1 DP path under bf16 compute.
`configs/qwen_session_dp2_layers03.toml` extends the representative DP=2
trainer path to layer0-layer3 with 49 trainable tensors, again excluding the
tied embedding by design.
`configs/qwen_session_dp2_layers03_bf16.toml` verifies the same layer0-layer3
DP path under bf16 compute.
`configs/qwen_session_dp2_layers07.toml` extends the fixed-token
representative DP=2 trainer path to layer0-layer7 with 97 trainable tensors,
still excluding the tied embedding by design and still checking rank0 plus
sharded checkpoint next-step parity.
`configs/qwen_session_dp2_layers07_bf16.toml` verifies the same layer0-layer7
fixed-token DP path under bf16 compute.
The DP sharded global manifest records the resolved complete Qwen checkpoint in
`base_model_path` plus its `tokenizer_path`; the DP verifiers reject manifests
whose recorded checkpoint lacks `config.json`, `tokenizer.json`, or
`model.safetensors`.
`configs/qwen_session_dp2_layers01_sft.toml` runs the same layer0+layer1
DP trainer over tokenizer-backed JSONL instruction batches and verifies dataset
provenance, sample cursor progress, rank0 checkpoint parity, and sharded
checkpoint next-step parity. `configs/qwen_session_dp2_layers03_sft.toml`
extends that JSONL path to layer0-layer3 with 49 trainable tensors.
`configs/qwen_session_dp2_layers07_sft.toml` extends the same JSONL path to
layer0-layer7 with 97 trainable tensors and keeps the dataset
provenance/cursor and rank0/sharded checkpoint parity checks.
`configs/qwen_session_dp2_sft_max_samples.toml` verifies that the DP=2 JSONL
SFT data plan applies `data.max_samples` before train/eval splitting, keeps
global cursor math consistent with the truncated train set, and records only
consumed source files in provenance. Its resume verifier also checks that the
rank0 and sharded checkpoint manifests preserve the truncated provenance and
continue from the saved global cursor. The DP2 max-samples and eval-paths
data-plan verifiers run from `scripts/verify_gpu_distributed.sh`, so they use
the same two-GPU Ray reservation as the rest of the distributed suite. DP=2
JSONL train batches use the same raw-index streaming window as the single-GPU
path, with the materialized dataset retained as a parity guard.
`configs/qwen_session_dp2_layers01_sft_bf16.toml`
verifies the layer0+layer1 tokenizer-backed DP path under bf16 compute,
`configs/qwen_session_dp2_layers03_sft_bf16.toml` verifies the layer0-layer3
JSONL DP path under bf16 compute, and
`configs/qwen_session_dp2_layers07_sft_bf16.toml` verifies the layer0-layer7
JSONL DP path under bf16 compute.
The external resume verifier also covers the layer0+layer1, layer0-layer3, and
layer0-layer7 fp32/bf16 SFT configs. It starts a fresh DP=2 JSONL run, resumes
from the emitted rank0 manifest, requires the resumed cursor to continue from
the base `data_cursor_next`, and checks the expected trainable set in the
resumed rank summaries and manifest. The bf16 external resume commands also
assert `compute_kind = bf16`. The layer0+layer1 paths expect 25 trainable
tensors, layer0-layer3 paths expect 49 trainable tensors, and the
layer0-layer7 paths expect 97 trainable tensors; all still exclude the tied
embedding by design.
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
and scheduler metadata, plus focused smoke identity and dataset provenance/data
progress slots. A second trainer launch can pass that manifest through
`--resume-from` to restore the focused rank-owned expert model/AdamW shards and
verify reload plus next-step parity. Full production EP is still open:
autograd-aware sparse collectives, trainer-owned production MoE integration,
and full expert optimizer/checkpoint ownership are not implemented.
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
  session trainer also has tokenizer-backed JSONL batch/resume verifiers for
  the default layer0 path and the layer0-layer7 fp32/bf16 paths. The DP=2
  session trainer has tokenizer-backed JSONL batch and external resume
  verifiers through layer0-layer7 in fp32/bf16. Production checkpoint/resume
  over full real data streams remains open.
- C4/G1 trainer-owned Qwen training is still incomplete, but the full-train
  smoke now uses a reusable `QwenTrainableSession` surface, and representative
  single-GPU plus DP=2 config paths are wired through `train --config`. The
  single-GPU path now respects configured `max_steps` and reports step losses.
  It can also resume from its saved delta manifest through `--resume-from` and
  reports throughput, gradient norm, RSS, and GPU memory metrics. The saved
  delta manifest records the resolved complete Qwen `base_model_path`, and the
  single-GPU verifiers reject incomplete legacy paths. The session registry can
  now expand configured trainable layer sets instead of being fixed to layer0,
  and `configs/qwen_session_single_sft.toml` feeds
  tokenizer-backed instruction JSONL batches through the same trainer path.
  `configs/qwen_session_dp2_sft.toml` extends that representative JSONL batch
  path through the 2-rank DP trainer. `configs/qwen_session_dp2_layers01.toml`
  verifies the same representative DP trainer over layer0+layer1 with sharded
  checkpoint reload and next-step resume parity,
  `configs/qwen_session_dp2_layers01_bf16.toml` verifies the same multi-layer
  DP trainable set under bf16 compute, and
  `configs/qwen_session_dp2_layers03.toml` extends the representative DP
  trainable set to layer0-layer3 with 49 trainable tensors and the same
  rank0/sharded checkpoint parity checks, and
  `configs/qwen_session_dp2_layers03_bf16.toml` verifies that four-layer DP
  path under bf16 compute.
  `configs/qwen_session_dp2_layers01_sft.toml` extends that multi-layer DP
  evidence to tokenizer-backed JSONL batches with manifest-backed data cursor
  and provenance checks, and `configs/qwen_session_dp2_layers03_sft.toml`
  extends the tokenizer-backed DP path to layer0-layer3.
  `configs/qwen_session_single_layers07.toml` extends the single-GPU
  representative trainer path to layer0-layer7 with 98 trainable tensors,
  including the tied embedding for the single-rank path.
  `configs/qwen_session_single_layers07_bf16.toml` verifies the same
  single-GPU layer0-layer7 path under bf16 compute.
  `configs/qwen_session_single_layers07_sft.toml` extends the single-GPU
  JSONL SFT resume path to layer0-layer7 with 98 trainable tensors and
  manifest-backed data cursor continuity.
  `configs/qwen_session_single_sft_max_samples.toml` verifies that the
  representative single-GPU JSONL SFT session path applies `data.max_samples`
  during JSONL reading and records only the consumed source files in checkpoint
  provenance.
  `configs/qwen_session_single_layers07_sft_bf16.toml` verifies the same
  single-GPU layer0-layer7 JSONL SFT resume path under bf16 compute.
  `configs/qwen_session_dp2_layers01_sft_bf16.toml`
  verifies the tokenizer-backed layer0+layer1 DP path under bf16 compute, and
  `configs/qwen_session_dp2_layers03_sft_bf16.toml` verifies the
  tokenizer-backed layer0-layer3 DP path under bf16 compute, and
  `configs/qwen_session_dp2_layers07_sft_bf16.toml` verifies the
  tokenizer-backed layer0-layer7 DP path under bf16 compute. The
  layers01, layers03, and layers07 JSONL paths now also have fp32 and bf16
  external resume verifiers. These prove the next run starts from the prior
  `data_cursor_next` while preserving dataset provenance and the expected
  trainable registry size.
  Full model/data/checkpoint trainer ownership remains open.
- G6 trainer-level real SFT data now has minimal Qwen LoRA SFT config paths:
  `train --config configs/qwen_lora_sft.toml` loads tokenizer-backed
  instruction JSONL batches, trains configured attention and MLP LoRA targets,
  reloads the adapter, and supports `--resume-from` for saved adapters. The
  trainer path now verifies the saved adapter through full-Qwen forward logits,
  greedy generation reload parity, and focused merge/unmerge parity. It also
  uses the trainer scheduler and grad clipping knobs, logs `eval_every` step
  eval history, applies seeded deterministic dataset ordering, reports dataset
  sample/token/mask summaries, applies `data.max_samples` during JSONL reading
  instead of after loading every source file, and has a bf16 variant at
  `configs/qwen_lora_sft_bf16.toml`. Manifest-backed adapter resume rejects
  `compute_kind` drift from the current train dtype; direct `.safetensors`
  adapter resume remains available for compatibility when manifest metadata is
  absent, restores adapter weights, and writes a fresh manifest without claiming
  saved data-cursor continuity. The adapter manifest records the same resolved
  complete Qwen `base_model_path` contract as delta checkpoints.
  `cargo run -- qwen-sft-streaming-data-plan --config ...` provides a
  tokenizer-free streaming JSONL scan for SFT provenance, source sample counts,
  split sizes, explicit `data.eval_paths`, fingerprints, and cursor/epoch
  windows without materializing tokenized samples.
  Instruction JSONL configs can map external dataset schemas with
  `data.instruction_field`, `data.input_field`, and `data.response_field`; the
  defaults remain `instruction`, `input`, and `response`, with missing input
  fields treated as empty strings. They can also override
  `data.prompt_template` and `data.prompt_with_input_template` to render
  external instruction formats before the response is appended for response-only
  loss; the default templates preserve the existing `Instruction`/`Input`/
  `Response` format and support `{instruction}` plus `{input}` placeholders.
  `data.trim_fields` defaults to `true` and trims JSONL instruction/input/
  response strings before template rendering; set it to `false` to preserve
  exact field whitespace. `data.min_response_chars` defaults to `1` and skips
  JSONL records whose normalized response is empty or shorter than the
  configured character count; `data.max_response_chars` can optionally skip
  records whose normalized response is longer than the configured character
  count. `data.min_instruction_chars` and `data.max_instruction_chars` can
  optionally skip records whose normalized instruction is outside configured
  character bounds. `data.min_input_chars` and `data.max_input_chars` apply the
  same optional bounds to the normalized input field. `data.min_prompt_chars`
  and `data.max_prompt_chars` apply optional bounds to the rendered prompt
  after template substitution and before appending the response.
  `data.min_sample_chars` and `data.max_sample_chars` apply optional bounds to
  the rendered prompt plus normalized response. These filters run before
  train/eval splitting, `max_samples`, and streaming offset-index construction.
  `data.source_weights` can be empty,
  length 1, or match `data.paths`; it repeats valid training samples from each
  configured source before `max_samples` and splitting, while explicit
  `data.eval_paths` remain unweighted held-out data.
  `cargo run -- qwen-sft-streaming-batch-plan --config ...` resolves the next
  cursor window to raw JSONL source indices, reads and tokenizes only those
  window records, then verifies the padded `input_ids` plus response masks match
  the current materialized dataset path. The LoRA SFT trainer plus single-GPU
  and DP=2 `qwen_trainable_session` JSONL SFT trainer paths now use the same
  raw-index streaming window to build train batches, while keeping materialized
  dataset metadata as a parity guard. These trainer paths also honor
  `data.index_cache`: single-GPU and LoRA SFT runs reuse the configured
  offset-index cache path directly, while DP=2 derives rank-local cache files to
  avoid shared-writer races. Trainer summaries/logs expose
  `streaming_index_cache_path`, `streaming_index_cache_hit`, and
  `streaming_index_cache_written`, and the focused GPU suites run each cache
  verifier twice to prove first-run writes and second-run hits. Offset-index
  cache files also record the JSONL field mapping, prompt templates, field
  trimming policy, instruction/input/response length filters, and source-weighting
  policy, so a cache created for one external schema is rejected if reused with
  a different field map, prompt format, normalization policy, filtering policy,
  or weighting policy.
  Those trainer summaries now expose
  `streaming_train_batches = true` for tokenizer-backed
  JSONL SFT runs, and the focused LoRA, single-GPU session, and DP session
  verifiers require that field in summaries and checkpoint manifests so stdout,
  rank JSON, or saved artifacts cannot pass while hiding a materialized
  train-batch path. When explicit `data.eval_paths` are configured, the
  streaming window
  keeps the full train source instead of applying `train_split` to it; the
  focused GPU suite checks both the
  tokenizer-free data plan and tokenizer-backed batch-plan parity for that
  case. A production zero-materialization loader for large external streams is
  still open.
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
  writer/restore/next-step resume smoke. Qwen sharded manifests record the
  resolved base checkpoint path rather than assuming the configured legacy path
  exists, and verifiers require that path to be a complete local Qwen artifact.
  Full production sharded resume over external streaming real data remains
  open.
- Trainer production basics such as scheduler, grad clipping, RSS memory
  metrics, and Ray-worker GPU memory reporting are implemented for toy/tch
  paths; real tokenizer-backed padded LoRA SFT batching is wired through
  minimal fp32/bf16 Qwen trainer configs with scheduler and grad clipping. The
  tiny `tch` CUDA path, representative Qwen session single/DP paths, the
  single-GPU layer0-layer7 Qwen session path, the DP layer0+layer1,
  layer0-layer3, and layer0-layer7 Qwen session paths, the single-GPU and DP
  layer0-layer7 JSONL SFT resume paths, and Qwen LoRA SFT resume verifier now
  have explicit bf16 compute-policy coverage.
  Mixed precision over future full production Qwen model/data paths remains
  future work.

Internal planning details live in `_internal_docs/TODO.md`; that directory is
ignored by git.
