# rustrain

rustrain is an educational Rust LLM training engine. It starts with a small
Qwen-like toy trainer and incrementally adds real Qwen checkpoint loading,
generation, LoRA/SFT building blocks, Megatron-style parallel module concepts,
MoE toy modules, and CUDA smoke tests through `tch-rs`.

This repository is not yet a production trainer. The current code is organized
as milestone-sized, verifiable smokes.

## GPU Verification

All verification runs on the GPU host. The local machine is only for editing and
git operations.

```sh
scripts/verify_gpu.sh
```

The script SSHes to `root@192.168.42.106:2222`, enters the remote checkout, and
sources `scripts/tch_a800_env.sh` before running project commands.

The required remote checkout is:

```sh
/vePFS-Mindverse/user/nolanho/code/rustrain
```

Current bootstrap fallback while the shared checkout is being prepared:

tar --exclude .git --exclude target --exclude runs -cf - . \
  | ssh -p 2222 root@192.168.42.106 'rm -rf /root/rustrain && mkdir -p /root/rustrain && cd /root/rustrain && tar -xf -'
RUSTRAIN_REMOTE_DIR=/root/rustrain scripts/verify_gpu.sh
```

The A800 config writes runs to `/tmp/rustrain-runs` because the shared project
checkout may be read-only from the worker. If `tch-cuda-probe` reports
`device_count: 0`, the SSH target has not exposed GPU devices and milestone
acceptance is blocked until the runtime is fixed.

## Real Qwen Smokes

The current real-model path targets the Qwen2.5 checkpoint under
`/vePFS-Mindverse/share/huggingface` on the GPU host:

```sh
ssh -p 2222 root@192.168.42.106 \
  'cd /vePFS-Mindverse/user/nolanho/code/rustrain && source scripts/tch_a800_env.sh && cargo run -- qwen-logits-parity'
```

The full-train smoke command is present, but GPU execution must be used for the
real checkpoint:

```sh
ssh -p 2222 root@192.168.42.106 \
  'cd /vePFS-Mindverse/user/nolanho/code/rustrain && source scripts/tch_a800_env.sh && cargo run -- qwen-full-train-smoke'
```

## Current Major Gaps

- Real full-Qwen training is only partially done: the command and representative
  trainable tensor path exist, but the A800 full-checkpoint smoke still needs to
  be run when the worker is reachable.
- Real Qwen module-level LoRA train/reload smoke for layer0 attention
  `q_proj`/`v_proj` is implemented; trainer-integrated LoRA SFT is not done yet.
- KV-cache greedy parity is implemented; cached sampling and Python cached
  generation parity are future work.
- Multi-GPU DP/TP/EP are toy smokes, not real NCCL-backed distributed training.
- Real Qwen checkpoint manifest support has started with delta metadata, but
  optimizer reload parity and distributed checkpoint layout are not done.
- Trainer production basics such as scheduler, grad clipping, RSS memory
  metrics, and a reserved GPU-memory metric field are implemented for toy/tch
  paths; actual GPU allocator/NVML memory values and real tokenizer-backed
  batching remain partial.

Internal planning details live in `_internal_docs/TODO.md`; that directory is
ignored by git.
