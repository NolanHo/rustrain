# Plan: TP + CP Support for GLM-5.2

## Summary

Add Tensor Parallelism (TP) and Context Parallelism (CP) to GLM-5.2 training, enabling 8→24 GPU scaling from ~52K to ~344K context length. TP splits attention heads across GPUs (all-reduce after output). CP splits the sequence dimension (ring attention K/V exchange). Both compose with existing EP (expert offloading) and all-layer checkpointing.

## Phase 1: TP (Tensor Parallelism)

### `crates/rustrain-glm5/src/model.rs`
- **Add `Glm5TpShard` struct** — mirrors V4's `V4TpShard`: `tp_rank`, `tp_size`, `heads_per_rank`, `head_start`, `idx_heads_per_rank`, `idx_head_start`
- **Add `Glm5TpAttentionWeights`** — TP-sharded attention weights:
  - `q_b_proj`: narrow rows by `[head_start * (qk_nope+qk_rope), heads_per_rank * (qk_nope+qk_rope)]` (column parallel)
  - `kv_b_proj`: narrow rows by `[head_start * (qk_nope+v_head), heads_per_rank * (qk_nope+v_head)]` (column parallel)
  - `o_proj`: narrow cols by same head range (row parallel → all-reduce after)
  - `indexer_wq_b`: narrow by `idx_head_start * idx_head_dim` (column parallel)
  - `q_a_proj`, `kv_a_proj`, `indexer_wk`, norms: **replicated** (low-rank, shared)
- **Add `glm5_dsa_attention_tp()`** — same as `glm5_dsa_attention` but uses `heads_per_rank` instead of `num_heads` for reshapes. The `num_heads` in IndexShareState becomes `heads_per_rank`. Indexer topk uses local `idx_n_heads` but K is replicated (single-head key), so topk_indices are identical across TP ranks — no cross-rank communication needed for indexer.

### `crates/rustrain-glm5/src/session_ep.rs`
- **Add `train_glm5_lora_sft_tp_ep()`** — new training function:
  - Parse `tp_size` from `config.parallel.tensor_model_parallel_size`
  - Compute `tp_rank = rank % tp_size`, `ep_rank = rank / tp_size`
  - Validate: `num_attention_heads % tp_size == 0`, `world_size == tp_size * ep_size`
  - Load TP-sharded attention weights (only `heads_per_rank` heads' q_b/kv_b/o_proj)
  - Load EP-sharded expert weights (CPU offload, same as now)
  - **Attention all-reduce**: after `o_proj`, all-reduce across ALL ranks, divide by `ep_size` (TP ranks produce identical partials within EP group)
  - **MoE all-reduce**: same as current, divide by `tp_size` (EP ranks produce identical partials within TP group)
  - **LoRA gradient all-reduce**: across ALL ranks, divide by `world_size`
  - IndexShare checkpointing: `IndexShareState` already per-rank sized (uses `heads_per_rank`), no change needed

### `src/main.rs`
- **Add dispatch**: if `config.parallel.tensor_model_parallel_size > 1`, route to `train_glm5_lora_sft_tp_ep`

### `crates/rustrain-parallel/src/launcher.rs`
- **Pass `TP_SIZE` env var** to child processes from `config.parallel.tensor_model_parallel_size` (matches V4 pattern)

## Phase 2: CP (Context Parallelism)

### `crates/rustrain-nccl/src/nccl.rs`
- **Add `send_recv()` method to `NcclPersistentComm`** — persistent-comm send/recv for ring attention:
  ```rust
  pub fn send_recv(&self, send_tensor: &Tensor, recv_peer: usize) -> Result<Tensor>
  ```
  - Uses `ncclGroupStart()` → `ncclSend()` + `ncclRecv()` → `ncclGroupEnd()`
  - Supports BF16 directly (no F32 upcast — unlike all_reduce which forces F32)
  - Must handle arbitrary tensor shapes (KV blocks)

### `crates/rustrain-glm5/src/model.rs`
- **Add `glm5_dsa_attention_cp()`** — ring attention with DSA sparse mask:
  - Split Q along seq dim: `q_local = q_full.narrow(2, cp_rank * s_local, s_local)`
  - K/V rotate through CP ranks via send/recv (CP_SIZE steps per layer)
  - For each K/V block from peer `j`:
    - Compute causal mask: query positions `[cp_rank*S/CP, (cp_rank+1)*S/CP)` vs key positions `[j*S/CP, (j+1)*S/CP)`
    - Apply DSA sparse mask: topk_indices for local query positions are pre-computed; need to filter to keys in this block
    - Accumulate attention output with online softmax (running max trick) — needed because softmax is split across K/V blocks
  - DSA indexer: topk computation uses local K only (`idx_k` for local S/CP tokens), then merge topk across CP ranks via all_gather (or iterative send/recv merge, similar to chunked DSA)

### `crates/rustrain-glm5/src/session_ep.rs`
- **Extend `train_glm5_lora_sft_tp_ep()` to accept CP** (or new `train_glm5_lora_sft_tp_cp_ep()`):
  - Parse `cp_size` from `config.parallel.context_parallel_size`
  - Compute rank decomposition: `tp_rank = rank % tp_size`, `cp_rank = (rank / tp_size) % cp_size`, `ep_rank = rank / (tp_size * cp_size)`
  - Local seq = `S / cp_size`
  - Checkpoint saves `s_local` instead of `S` (halved+ memory)
  - Ring attention K/V exchange per layer (CP_SIZE send/recv rounds)

### `crates/rustrain-glm5/src/sft.rs`
- **Adjust data pipeline**: split input sequence across CP ranks (each rank gets `S/CP` consecutive tokens)

## Risks

- **Online softmax for ring attention**: Standard SDPA can't be used directly with rotating K/V. Need manual attention with running max + exp normalization. This is the hardest part — the DSA sparse mask complicates it further (mask is per-block).
  - *Mitigation*: Start with CP=2 (only 2 K/V blocks), simpler ring. Validate correctness against CP=1.
- **BF16 all-reduce precision**: `NcclPersistentComm.all_reduce` forces F32. For TP attention output, this is acceptable (matches V4).
- **DSA indexer under CP**: topk needs full-sequence K to find global top-k. With CP, each rank has S/CP keys. Need cross-rank topk merge.
  - *Mitigation*: Reuse the chunked topk merge logic — each CP rank computes local topk, then merge via all-reduce (sum of indicator vectors) or send/recv exchange.
- **Checkpoint closures with TP**: `Arc<Mutex<Option<IndexShareState>>>` already uses local `heads_per_rank`. TP doesn't change the closure pattern — just the state size.

## Implementation Order

1. **TP only** (no CP) — smallest change, immediate benefit (52K→115K on 8 GPU)
2. **Test TP** — seq_len=8192 with TP=2, compare loss to TP=1 baseline
3. **CP=2** — add send/recv to persistent comm, ring attention, online softmax
4. **Test CP=2** — seq_len=16384 with TP=2+CP=2, compare loss
5. **CP=3** — generalize ring to arbitrary CP_SIZE

## Definition of Done

- [ ] TP=2 on 8 GPU: loss matches TP=1 baseline at same seq_len
- [ ] TP=2+EP=4 on 16 GPU: loss matches, experts offloaded
- [ ] TP=8 on 8 GPU: loss matches, seq_len=32K runs without OOM
- [ ] CP=2 on 16 GPU: loss matches CP=1 at seq_len=1024
- [ ] TP=2+CP=2 on 16 GPU: seq_len=32K runs without OOM
- [ ] `cargo check --release` passes

## Open Questions

- CP ring attention: should we implement online softmax manually in Rust, or wrap a C++ flash attention kernel? (Plan: Rust manual first, C++ optimization later if too slow)
- Should CP split happen before or after embedding? (Plan: after — embed full sequence, then narrow to local chunk. Simpler, embed is replicated)
