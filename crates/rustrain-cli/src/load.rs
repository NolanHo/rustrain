//! D5's weight loader: real safetensors bytes → the f32 tensors the widened plan executes.
//!
//! The mapping is the one fact, one source rule applied to bytes: this module consumes the *same*
//! checkpoint↔slot pairing `rustrain check` reconciles (`crate::pairing`), applies each binding's
//! `transform` and `split` to the actual data (the same vocabulary and the same shape math as
//! `crate::transformed_shape` / `crate::split_shape`), extracts this rank's slice of every
//! declared shard, and asserts the produced tensor equals the slot's **local** shape before a
//! single weight reaches the executor.
//!
//! Precision (the frozen decision): the checkpoint and HF are bf16 while the reference provider
//! is f32-only, so the loader widens bf16/f16/f32 weights to f32 — exact for bf16 and f16
//! (both are subsets of f32) — and the plan runs f32. HF's reference dump is bf16; the spec's 1%
//! tolerance absorbs HF's own bf16 rounding.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use rustrain_parallel::Mesh;
use rustrain_plan::{Plan, SlotId};

use crate::{
    CheckpointMeta, axis, consumption, load_checkpoint, pairing, shape_text, split_shape,
    transformed_shape,
};

/// What the load cost, by phase. The loader used to be the slowest part of a run by an order of
/// magnitude, so "how long did it take" is not enough: the file read and the host-side work that
/// turns bytes into the slot's local tensor have different fixes, and each used to be invisible.
/// Reported in the metrics JSON.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct LoadStats {
    /// Bytes actually handed to `read` (this is what `checkpoint_bytes_read` means).
    pub bytes_read: u64,
    /// Bytes a *single full read* per tensor would have needed, counted over the pairing's tensor
    /// names independently of how the loader groups and narrows them. A rank that needs every
    /// tensor — world 1 — reads exactly this; on a sharded mesh `bytes_read` is smaller by the
    /// slices the other ranks own.
    pub bytes_distinct: u64,
    pub tensors_read: usize,
    /// How many `read` calls the narrowed reads took, for the days the window is not enough.
    pub read_runs: usize,
    pub pairs_total: usize,
    pub read: Duration,
    /// The composed walk: widening, layout and the rank's slab, done in one pass per member.
    pub fill: Duration,
}

/// What a load produced: the phase timings, the slots that reached the device, and the wall time
/// the device writes took (they overlap the reads now, so it is not a phase sum).
pub(crate) struct LoadOutcome {
    pub stats: LoadStats,
    /// Weight slots written into the executor.
    pub slots: usize,
    /// Widened f32 bytes that reached the device — the number that must fall as a mesh widens.
    pub weight_bytes: u64,
    /// Wall time of the whole load — the index parse, the pairing, the workers and the device
    /// writes.
    pub wall: Duration,
    /// Time the calling thread spent *inside* the device copies, not waiting for the workers to
    /// hand over the next tensor — the number that says what the copies cost, and the one that
    /// would change if the copies were done with pinned memory or a wider transfer.
    pub write: Duration,
    /// Time the calling thread spent *waiting* for the pool rather than copying. `write + write_wait`
    /// is the writer's whole span, so the pair says which stage the load is bound by.
    pub write_wait: Duration,
    /// Device copies the writer issued, and the element count they moved. `bytes/4/chunks` is the
    /// average piece size — the number that says whether a slow copy is per-call overhead (small
    /// pieces) or the transport (large ones).
    pub write_chunks: usize,
    /// Per-chunk copy durations as `(bucket upper bound in microseconds, chunks, bytes)`, for the
    /// buckets `<500 us`, `<1 ms`, `<2 ms`, `<5 ms`, and `>=5 ms`. An average hides the shape: 5004
    /// copies at 5 GB/s is either a uniform slow transport or a fast path with pathological
    /// outliers, and the two have different fixes.
    pub write_histogram: Vec<(u64, usize, u64)>,
    /// How many workers the pool ran (`LOAD_WORKERS` capped by the group count); the phase times
    /// below are sums over them.
    pub workers: usize,
}

/// What the writer thread keeps while the workers hand it tensors: which slots arrived, and what
/// reached the device.
#[derive(Default)]
struct Sink {
    seen: Vec<bool>,
    slots: usize,
    bytes: u64,
    /// Elements written into each slot. A streamed weight arrives in pieces, so "the slot was
    /// seen" is no longer enough to prove it was *filled*: a lost, duplicated or misplaced chunk
    /// would leave a gap or an overlap that no other check would notice (a zeroed region of a
    /// weight is a plausible-looking tensor). This counter is the structural check — one number
    /// per slot, compared against the slot's length at the end of the load.
    written: Vec<u64>,
}

/// One weight slot's data — or a slice of it — ready to write into the executor.
///
/// A weight is streamed in chunks of `CHUNK_BYTES` rather than materialised whole: the host copies
/// it once instead of three times, and the device copy starts before the last chunk is prepared.
/// `element_offset` is where this chunk starts in the slot's row-major element order.
pub(crate) struct LoadedWeight {
    /// The slot in the **instantiated** (local) plan.
    pub slot: SlotId,
    pub name: String,
    /// Where this chunk starts, in elements of the slot's local shape.
    pub element_offset: usize,
    /// Widened to f32, laid out contiguously in row-major order of the chunk's own shape.
    pub values: Vec<f32>,
}

/// How many chunk buffers the pool keeps. See the note where it is built: this is a cache-locality
/// knob as much as a memory bound.
const POOL_BUFFERS: usize = 16;

/// How large a chunk a member is streamed in.
///
/// One member used to be materialised as a single `Vec<f32>`: a 40 MB weight became 40 MB of host
/// allocation touched three times (fill writes it, the copy reads it, the device reads it), and the
/// fresh allocation faulted its pages in every time. Eight megabytes keeps a chunk inside L2/L3
/// across the fill→copy handoff while still being large enough that the per-chunk overhead (one
/// channel message, one driver call) is nothing.
const CHUNK_BYTES: usize = 8 << 20;

/// How many tensors are read and prepared at once.
///
/// The loader used to be one serial pass, and it was the slowest part of a run by two orders of
/// magnitude: the measured split of a 614 s load at 1 thread was transposes 420 s, transform
/// slices 74 s, widening 71 s, reading 34 s — 92% of it host CPU, on one of 160 cores. The
/// checkpoint mount is the mirror image: one stream reads at ~160 MB/s, eight streams at
/// multiple GB/s. One worker per tensor fixes both sides at once, and the memory it costs is one
/// tensor's checkpoint bytes plus its slots per worker.
const LOAD_WORKERS: usize = 16;

/// `LOAD_WORKERS`, overridable for measurement. The pool's shape has changed twice while the
/// constant stayed put (the narrowed read, then the 8 MiB chunk stream), and the right number is a
/// property of the pipeline, not of the machine — so it can be measured without a rebuild.
fn load_workers() -> usize {
    std::env::var("RUSTRAIN_LOAD_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 1)
        .unwrap_or(LOAD_WORKERS)
}

/// The most read syscalls a group's narrowed read may cost. Past this the covering window is
/// cheaper than the syscalls, so the loader reads the whole window instead — a fallback that only
/// ever reads *more* than the runs would.
const READ_RUN_CAP: usize = 8192;

/// One slot's share of a group: what to cut out of the prepared tensor, and what the two
/// independent shape facts say the result must be.
struct Member {
    slot: SlotId,
    slot_name: String,
    /// The binding's `source` pattern, for the errors that name it.
    source: String,
    /// The slot's local shape in the instantiated plan.
    local_shape: Vec<i64>,
    /// The `split` segment to cut, as `(axis, start, len)` in the transformed shape.
    segment: Option<(usize, usize, usize)>,
    /// The declared shards to cut, as `(axis, start, len)` in the shape after the segment.
    shards: Vec<(usize, usize, usize)>,
    /// What `transformed_shape` / `split_shape` — the *other* evaluation of the same shape fact —
    /// say the transformed and split tensor must be.
    expected: Vec<i64>,
}

/// One group's result: its position in the group list (so errors are reported in a fixed order)
/// and what loading it produced.
type GroupOutcome = (usize, Result<LoadStats>);

/// One checkpoint tensor plus every pair that reads it with the same transform.
///
/// A `split` binding produces one pair per segment, and each pair used to read the whole tensor
/// again: 873 pairs over 712 tensors, +64.8% bytes read. Grouping by `(tensor, transform)` makes
/// the shared work shared, and is why `bytes_read == bytes_distinct` holds today.
struct Group {
    tensor: String,
    shape: Vec<i64>,
    dtype: String,
    shard: PathBuf,
    data: (u64, u64),
    steps: Vec<rustrain_model::Transform>,
    members: Vec<Member>,
}

/// Loads every weight slot of `plan` (the instantiated, rank-local plan) from the real
/// safetensors checkpoint, applying transform → split → shard extraction and asserting the
/// result against the slot's local shape.
///
/// `expanded` supplies the bindings and the *declared* dtypes; `plan` supplies the local shapes
/// and layouts (the plan was widened to f32 after `instantiate`, so dtype expectations come from
/// the expanded plan, not from `plan`).
///
/// Every failure is the same error `check` reports — missing tensor, extra tensor, unbound slot,
/// dtype disagreement, shape mismatch — never a weaker one.
///
/// This is the one place where the loader is allowed to be clever about *how* it gets the bytes,
/// so it is also the place where the order of the two shape facts is fixed: the groups are built
/// (and every check that needs no bytes is run) before a single byte is read, and each group's
/// data walk is then asserted against the shape math it was grouped by.
pub(crate) fn load_weights(
    expanded: &rustrain_model::Expanded,
    desc: &rustrain_model::ModelDesc,
    plan: &Plan,
    mesh: &Mesh,
    rank: usize,
    checkpoint: &Path,
    executor: &mut rustrain_runtime::Executor,
) -> Result<LoadOutcome> {
    let load_started = Instant::now();
    let meta = load_checkpoint(checkpoint)?;

    // `run` needs the bytes, not just the metadata: a snapshot has no `data_offsets`, so nothing
    // in it can be loaded. Refuse it by name rather than failing on the first tensor.
    if meta
        .tensors
        .values()
        .any(|t| t.data.is_none() || t.shard.is_none())
    {
        bail!(
            "{} is checkpoint *metadata* without weight bytes; `run` needs the real safetensors \
             directory or its model.safetensors.index.json",
            meta.source
        );
    }

    let p = pairing(expanded, &meta);
    reject_broken_pairing(expanded, desc, &meta, &p)?;

    // ---- fold the pairs into one job per (tensor, transform) ------------------
    let mut groups: Vec<Group> = Vec::new();
    let mut group_of: std::collections::HashMap<(String, Vec<String>), usize> =
        std::collections::HashMap::new();
    // One entry per checkpoint tensor this rank reads, whatever grouping does with it.
    let mut distinct: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for pair in &p.pairs {
        let binding = &expanded.bindings[pair.binding];
        let tensor = meta.tensors.get(&pair.tensor).ok_or_else(|| {
            anyhow::anyhow!(
                "checkpoint tensor `{}` disappeared from the pairing",
                pair.tensor
            )
        })?;

        // The declared dtype of the slot — from the *global* plan, before the runner widened it.
        let Some(declared_id) = expanded.plan.slot_id(&pair.slot) else {
            bail!(
                "slot `{}` is not a slot of the expanded plan (binding `{}`)",
                pair.slot,
                binding.source
            );
        };
        let declared = expanded.plan.slot(declared_id);
        if tensor.dtype != declared.dtype.name() {
            bail!(
                "slot `{}` <- `{}` ({}): the checkpoint declares `{}`, the description declares \
                 `{}`",
                pair.slot,
                pair.tensor,
                binding.source,
                tensor.dtype,
                declared.dtype.name()
            );
        }

        // The local plan may not carry this slot (another PP stage's weight); that tensor belongs
        // to a rank this process is not executing. Skip it — the final coverage walk below only
        // demands the slots *this* plan has.
        let Some(slot_id) = plan.slot_id(&pair.slot) else {
            continue;
        };
        let slot = plan.slot(slot_id);

        // ---- the shape math, evaluated (the same functions `check` uses) ----
        let transformed =
            transformed_shape(&tensor.shape, &binding.transform).map_err(anyhow::Error::msg)?;
        let expected = match &binding.split {
            Some(split) => {
                split_shape(&transformed, split, pair.segment).map_err(anyhow::Error::msg)?
            }
            None => transformed.clone(),
        };

        // ---- what this member cuts out, resolved before any byte is read ----
        // Transforms never change the number of axes, so one rank describes every step.
        let rank_dims = transformed.len();
        let segment = match &binding.split {
            Some(split) => {
                let d = axis(split.dim, rank_dims).ok_or_else(|| {
                    anyhow::anyhow!(
                        "split dim {} is out of range for the {} the transform produced",
                        split.dim,
                        shape_text(&tensor.shape)
                    )
                })?;
                let sizes: Vec<usize> = split.sizes.iter().map(|s| *s as usize).collect();
                let start: usize = sizes[..pair.segment].iter().sum();
                Some((d, start, sizes[pair.segment]))
            }
            None => None,
        };
        let mut after_segment = transformed.clone();
        if let Some((d, _, len)) = segment {
            after_segment[d] = len as i64;
        }
        let mut after_segment_here = after_segment.clone();
        let shards = shard_slabs(
            &slot.name,
            &slot.layout.dims,
            &mut after_segment_here,
            rank_dims,
            mesh,
            rank,
        )?;
        let key = (pair.tensor.clone(), binding.transform.clone());
        let index = match group_of.get(&key) {
            Some(index) => *index,
            None => {
                let (shard, data) = match (&tensor.shard, tensor.data) {
                    (Some(shard), Some(range)) => (shard.clone(), range),
                    _ => bail!(
                        "checkpoint tensor `{}` has no weight bytes to load",
                        pair.tensor
                    ),
                };
                distinct.insert(pair.tensor.clone(), data.1 - data.0);
                let steps: Vec<rustrain_model::Transform> = binding
                    .transform
                    .iter()
                    .map(|step| {
                        rustrain_model::parse_transform(step)
                            .map_err(anyhow::Error::msg)
                            .with_context(|| format!("binding `{}`", binding.source))
                    })
                    .collect::<Result<_>>()?;
                groups.push(Group {
                    tensor: pair.tensor.clone(),
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype.clone(),
                    shard,
                    data,
                    steps,
                    members: Vec::new(),
                });
                group_of.insert(key, groups.len() - 1);
                groups.len() - 1
            }
        };
        groups[index].members.push(Member {
            slot: slot_id,
            slot_name: pair.slot.clone(),
            source: binding.source.clone(),
            local_shape: slot.shape.clone(),
            segment,
            shards,
            expected,
        });
    }

    // ---- read and prepare the groups in parallel, writing each slot as it is ready ----------
    // One worker per tensor, not one per core: the host-side fill is the bulk of the work and a
    // worker holds one tensor's checkpoint bytes plus the slots cut out of it, so the pool is
    // sized by what the mount and the memory want, not by the CPU count. Sixteen beats forty-eight
    // on both sides (measured: 30.5 s vs 38.9 s for one run) — more workers only add contention.
    //
    // The tensors go straight into the executor from a writer thread rather than accumulating in
    // host memory: that hides the host-to-device copies behind the reads (they used to be a
    // serial 19-23 s after every byte had been read), and the rank stops holding a second, f32
    // copy of every weight it owns.
    let workers = load_workers().min(groups.len()).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let outcomes: std::sync::Mutex<Vec<GroupOutcome>> =
        std::sync::Mutex::new(Vec::with_capacity(groups.len()));
    // A *bounded* channel: the point of streaming the weights into the executor is that the host
    // does not hold a second copy of every tensor it owns, and an unbounded queue would quietly
    // hand back all of it whenever the fill runs faster than the copies do (it does: ~9 s of fill
    // against ~19 s of copies). One slot in flight per worker is the bound.
    let (tx, rx) = std::sync::mpsc::sync_channel::<LoadedWeight>(workers);
    // Warm chunk buffers: a streamed member is filled into one of these and the writer hands it
    // back after the device copy. Without the pool every 8 MiB chunk is a fresh allocation whose
    // pages must be faulted in before the copy can read them — and the copy is the load's
    // bottleneck (spec.md §D6.10), so the faults were being paid on the critical path. Bounded:
    // the workers and the channel can hold a few chunks each, and nothing else wants one.
    // A *small* pool on purpose: each buffer is filled by a worker and then read by the writer's
    // device copy, and the copy reads host memory at the DRAM rate for a cold source (9.8-11.5
    // GB/s measured) but at 15-16 GB/s when the source is still in cache. Sixteen buffers of 8 MiB
    // is 128 MiB of in-flight chunks, which is the point where the two are told apart without
    // starving the writer (the producer is faster than the copier, so it waits anyway).
    let pool: std::sync::Arc<std::sync::Mutex<Vec<Vec<f32>>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::with_capacity(POOL_BUFFERS)));
    let mut sink = Sink {
        seen: vec![false; plan.slots.len()],
        written: vec![0u64; plan.slots.len()],
        ..Sink::default()
    };
    let mut write_busy = Duration::ZERO;
    // Time the writer spent *waiting* for the pool rather than copying. Together with `write_busy`
    // this says which side of the pipeline the load is bound by: a small wait means the writer is
    // the bottleneck (the fix is cheaper copies), a large one means the producers are (the fix is
    // more, or better-shaped, workers) — and the two have opposite fixes, which is why the number
    // has to exist rather than be inferred from `read_cpu_seconds`.
    let mut write_wait = Duration::ZERO;
    let mut write_chunks = 0usize;
    let mut buckets: [(u64, usize, u64); 5] = [
        (500, 0, 0),
        (1_000, 0, 0),
        (2_000, 0, 0),
        (5_000, 0, 0),
        (u64::MAX, 0, 0),
    ];
    // Set when a device copy fails: the workers check it before taking another tensor, so a
    // failure does not turn into "read the rest of the checkpoint first, then report".
    let abort = std::sync::atomic::AtomicBool::new(false);
    let written: Result<()> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let tx = tx.clone();
            let pool = std::sync::Arc::clone(&pool);
            // Only the sender moves; the pool's shared state stays borrowed, so the calling
            // thread still owns `outcomes` and the sink after the scope.
            let (abort, next, groups, outcomes) = (&abort, &next, &groups, &outcomes);
            handles.push(scope.spawn(move || {
                loop {
                    if abort.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(group) = groups.get(index) else {
                        break;
                    };
                    let outcome = load_group(group, rank, &pool, &tx, abort);
                    outcomes
                        .lock()
                        .expect("the loader's result lock")
                        .push((index, outcome));
                }
            }));
        }
        drop(tx);
        // The calling thread is the writer. `Executor` owns device pointers and is not `Send`, so
        // it cannot move to a thread of its own — and it does not need to: the workers only ever
        // touch the groups and the channel, and the copies happen here, overlapped with them.
        // The device copy path is deliberately the plain synchronous one. A pinned-staging ring
        // with asynchronous copies was built and measured (commit history: 2026-09 round 8): the
        // primitives are fast in isolation (host memcpy into pinned 13.6 GB/s, async H2D 52 GB/s
        // versus pageable `cuMemcpyHtoD` at ~6 GB/s), but the load's wall clock did not move —
        // 9.0-10.6 s staged against 9.3-11.2 s plain, with the writer's own time still 6.8-8.4 s.
        // The pipeline is bound by the host work the sixteen workers and the writer do against the
        // same memory bandwidth (fill 54-58 CPU-seconds per rank, read 30-36), not by the
        // transport, so the ring bought complexity instead of time and was removed.
        loop {
            let waited = Instant::now();
            let Ok(weight) = rx.recv() else {
                write_wait += waited.elapsed();
                break;
            };
            write_wait += waited.elapsed();
            write_chunks += 1;
            let started = Instant::now();
            let chunk_bytes = (weight.values.len() * 4) as u64;
            let copied = executor
                .write_f32_at(weight.slot, weight.element_offset, &weight.values)
                .with_context(|| format!("writing the weight slot `{}`", weight.name));
            let taken = started.elapsed();
            write_busy += taken;
            let micros = taken.as_micros() as u64;
            for bucket in &mut buckets {
                if micros < bucket.0 {
                    bucket.1 += 1;
                    bucket.2 += chunk_bytes;
                    break;
                }
            }
            let () = match copied {
                Ok(()) => {}
                Err(error) => {
                    abort.store(true, std::sync::atomic::Ordering::Relaxed);
                    return Err(error);
                }
            };
            sink.bytes += (weight.values.len() * 4) as u64;
            sink.written[weight.slot.0] += weight.values.len() as u64;
            // A streamed weight arrives as several chunks; the slot is counted once, when its
            // first chunk lands, and `seen` is simply idempotent.
            if !sink.seen[weight.slot.0] {
                sink.slots += 1;
                sink.seen[weight.slot.0] = true;
            }
            // Hand the buffer back for the next chunk: it is warm, which is the whole point.
            let mut pool = pool.lock().expect("the chunk pool");
            if pool.len() < POOL_BUFFERS {
                pool.push(weight.values);
            }
        }

        // Reaching here means `rx` yielded everything it ever will: either every worker finished
        // and dropped its sender, or a failed copy returned early and dropped the receiver (which
        // unblocks any worker waiting in `send`). The joins are what is left of the pool.
        for handle in handles {
            handle.join().expect("a loader worker panicked");
        }
        Ok(())
    });
    // Deterministic error reporting: the lowest group index that failed is the one reported, no
    // matter which worker got there first. A failed device copy is reported only after that — it
    // is the later, less specific failure (it says a slot could not be written, not what was
    // wrong with the tensor), and it would otherwise mask a description the loader can name.
    let mut outcomes = outcomes.into_inner().expect("the loader's result lock");
    outcomes.sort_by_key(|(index, _)| *index);
    let mut stats = LoadStats {
        pairs_total: p.pairs.len(),
        bytes_distinct: distinct.values().sum(),
        ..LoadStats::default()
    };
    let mut group_error: Option<anyhow::Error> = None;
    for (_, outcome) in outcomes {
        match outcome {
            Ok(group_stats) => {
                stats.bytes_read += group_stats.bytes_read;
                stats.tensors_read += group_stats.tensors_read;
                stats.read_runs += group_stats.read_runs;
                stats.read += group_stats.read;
                stats.fill += group_stats.fill;
            }
            Err(error) => {
                group_error = Some(error);
                break;
            }
        }
    }
    if let Some(error) = group_error {
        // The copy failure is the less specific of the two, but it is not dropped: on a device
        // out-of-memory it is the sentence that explains what actually happened.
        return Err(match written {
            Ok(()) => error,
            Err(copy_error) => {
                anyhow::anyhow!("{error:#}; a device copy also failed: {copy_error:#}")
            }
        });
    }
    written?;

    // Every weight slot of *this* plan must have been loaded exactly once; a weight slot with no
    // pairing is an unbound slot (already rejected above), and one loaded twice would be a
    // non-bijective pairing (rejected too) — this walk is the loader's own backstop.
    for (index, slot) in plan.slots.iter().enumerate() {
        if slot.kind != rustrain_plan::SlotKind::Weight {
            continue;
        }
        if !sink.seen[index] {
            bail!(
                "weight slot `{}` of the rank-{rank} plan was loaded from no checkpoint tensor",
                slot.name
            );
        }
        // Every element exactly once. A streamed weight is written in pieces, and this is what
        // makes "in pieces" as verifiable as "in one go" used to be.
        let expected: u64 = slot.shape.iter().map(|axis| *axis as u64).product();
        if sink.written[index] != expected {
            bail!(
                "weight slot `{}` of the rank-{rank} plan received {} element(s), but its local \
                 shape holds {expected}: a chunk was lost, duplicated or written twice",
                slot.name,
                sink.written[index]
            );
        }
    }

    Ok(LoadOutcome {
        stats,
        slots: sink.slots,
        weight_bytes: sink.bytes,
        wall: load_started.elapsed(),
        write: write_busy,
        write_wait,
        write_chunks,
        write_histogram: buckets.to_vec(),
        workers,
    })
}

/// How one member's tensor is cut out of its checkpoint tensor: for every axis of the **result**,
/// which checkpoint axis feeds it and which slice of that axis is kept.
///
/// Every operation a binding can ask for is either "swap two axes" (`transpose`) or "keep a
/// sub-range of one axis" (`slice`, a `split` segment, a shard slab). All of them therefore
/// compose into this map *without touching a byte*, and the loader can then walk the result once,
/// reading the checkpoint element each output element comes from. The chain it replaced —
/// widen a copy, transpose a copy, slice a copy, slice a copy — moved every byte four times, and
/// its per-element index arithmetic was 92% of a run's startup (measured: 420 s of transposes,
/// 74 s of slices, 71 s of widening, 34 s of reading, for a 614 s load at one thread).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Cuts {
    /// Per result axis: `(checkpoint axis, start, len)`, in row-major result order.
    axes: Vec<(usize, i64, i64)>,
}

impl Cuts {
    /// The result starts out identical to the checkpoint tensor.
    fn identity(shape: &[i64]) -> Self {
        Self {
            axes: shape
                .iter()
                .enumerate()
                .map(|(axis, len)| (axis, 0, *len))
                .collect(),
        }
    }

    /// `transpose(i, j)`: the two result axes swap places, so the checkpoint axes feeding them do.
    fn transpose(&mut self, i: usize, j: usize) {
        self.axes.swap(i, j);
    }

    /// Keep `[start, start + len)` of result axis `dim` — the shape of `slice`, of a `split`
    /// segment and of a shard slab is the same cut on one axis.
    fn narrow(&mut self, dim: usize, start: i64, len: i64, what: &str) -> Result<()> {
        let (_, axis_start, axis_len) = self.axes[dim];
        // A zero-length window has no valid source span (`source_span` computes `start + len - 1`),
        // and its runs would address a zero-byte slice at a position that can be past the end of
        // the tensor. Nothing in the description language can ask for one (`transform` requires
        // `len > 0`, a `split` size is positive, a shard slab is at least one element), so this is
        // a guard against a panic with no message rather than a reachable case.
        if len <= 0 {
            bail!(
                "{what}: a slice of length {len} is not a cut — axis {dim} holds {axis_len} \
                 element(s) and a cut keeps at least one of them"
            );
        }
        if start < 0 || start + len > axis_len {
            bail!(
                "{what}: axis {dim} of this tensor holds {axis_len} element(s), and the cut asks \
                 for [{start}, {})",
                start + len
            );
        }
        self.axes[dim] = (self.axes[dim].0, axis_start + start, len);
        Ok(())
    }

    /// The result's shape.
    fn shape(&self) -> Vec<i64> {
        self.axes.iter().map(|(_, _, len)| *len).collect()
    }

    /// The checkpoint element range the result reads, as `(lowest, highest)`.
    ///
    /// This is what makes a rank read only what it needs: every cut is a range on one axis, so the
    /// union of the members' spans is the exact byte window the group has to touch — an eighth of
    /// every sharded tensor at `tp = 8`, instead of the whole tensor cut in host memory.
    fn source_span(&self, strides: &[i64]) -> (i64, i64) {
        if self.axes.is_empty() {
            return (0, 0);
        }
        let low: i64 = self
            .axes
            .iter()
            .map(|(axis, start, _)| start * strides[*axis])
            .sum();
        let high: i64 = self
            .axes
            .iter()
            .map(|(axis, start, len)| (start + len - 1) * strides[*axis])
            .sum();
        (low, high)
    }

    /// The contiguous runs of checkpoint elements the result reads, in increasing order.
    ///
    /// A box is contiguous across every axis *inside* its innermost partial one, so the read can be
    /// narrowed to these runs rather than to the whole window that covers them: the expert tensor
    /// whose middle axis is a `tp` slab is one run per expert, not one element per row. `cap` is
    /// the point at which the run list stops being worth its syscalls and the caller reads the
    /// covering window instead — the result is then a single run.
    fn runs(
        &self,
        source_dims: &[i64],
        strides: &[i64],
        cap: usize,
        window: (i64, i64),
    ) -> Vec<(i64, i64)> {
        // Back to source-axis order: one kept range per checkpoint axis.
        let mut ranges: Vec<(usize, i64, i64)> = self
            .axes
            .iter()
            .map(|(axis, start, len)| (*axis, *start, *len))
            .collect();
        ranges.sort_by_key(|(axis, _, _)| *axis);

        // The innermost checkpoint axis that is *not* fully covered decides the run length: every
        // axis after it is full, so one run covers it and all of them.
        let partial = ranges
            .iter()
            .rposition(|(axis, start, len)| *start != 0 || *len != source_dims[*axis]);
        let Some(k) = partial else {
            // The whole tensor: one run, however the axes are ordered.
            return vec![(window.0, window.1 - window.0 + 1)];
        };

        let (axis_k, start_k, len_k) = ranges[k];
        let run_elems = len_k * strides[axis_k];
        let inner: Vec<(usize, i64, i64)> = ranges[..k].to_vec();
        let count: i64 = inner.iter().map(|(_, _, len)| *len).product();
        if count > cap as i64 {
            return vec![(window.0, window.1 - window.0 + 1)];
        }

        // Odometer over the kept indices of the axes before `k`; their summed offset is the base.
        let mut runs = Vec::with_capacity(count as usize);
        let mut index = vec![0i64; inner.len()];
        loop {
            let base: i64 = inner
                .iter()
                .zip(&index)
                .map(|((axis, start, _), i)| (start + i) * strides[*axis])
                .sum();
            runs.push((base + start_k * strides[axis_k], run_elems));
            let mut a = inner.len();
            loop {
                if a == 0 {
                    return runs;
                }
                a -= 1;
                index[a] += 1;
                if index[a] < inner[a].2 {
                    break;
                }
                index[a] = 0;
            }
        }
    }

    /// Every element of the result, read out of the checkpoint bytes at its composed offset.
    ///
    /// The walk is an odometer over the result's axes with the checkpoint stride of each axis, so
    /// the inner loop is one pointer step and one conversion per element — no division, no
    /// intermediate buffer, and each output element is written exactly once.
    /// `window` is the checkpoint element the buffer's first byte holds: the read is narrowed to
    /// the group's span, so every offset is relative to it.
    fn fill(&self, bytes: &[u8], dtype: &str, strides: &[i64], window: i64) -> Result<Vec<f32>> {
        let rows = self.axes.first().map(|(_, _, len)| *len).unwrap_or(1);
        self.fill_chunk(bytes, dtype, strides, window, 0, rows as usize)
    }

    /// The member's outer axis slice `[start, start + rows)`, as [`Cuts::fill`] would produce it.
    ///
    /// The walk is linear in the outermost axis — its checkpoint step is one constant — so a slice
    /// is the same walk with one fewer axis and the base moved by `start * step[0]`. Nothing is
    /// recomputed and nothing is duplicated: the member is exactly the concatenation of its slices,
    /// which is what lets the loader stream a weight to the device in pieces instead of
    /// materialising all of it first.
    fn fill_chunk(
        &self,
        bytes: &[u8],
        dtype: &str,
        strides: &[i64],
        window: i64,
        start: i64,
        rows: usize,
    ) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        self.fill_chunk_into(&mut out, bytes, dtype, strides, window, start, rows)?;
        Ok(out)
    }

    /// [`Cuts::fill_chunk`] into a caller-owned buffer, so a streamed member can reuse one warm
    /// allocation per worker instead of faulting a fresh one in for every chunk.
    ///
    /// Eight arguments is deliberate: the buffer, the bytes, the dtype, the checkpoint strides, the
    /// read window, the slice and its length are all facts the caller already holds, and bundling
    /// them into a struct would exist only to satisfy this lint.
    #[allow(clippy::too_many_arguments)]
    fn fill_chunk_into(
        &self,
        out: &mut Vec<f32>,
        bytes: &[u8],
        dtype: &str,
        strides: &[i64],
        window: i64,
        start: i64,
        rows: usize,
    ) -> Result<()> {
        let mut shape: Vec<usize> = self.axes.iter().map(|(_, _, len)| *len as usize).collect();
        let step: Vec<i64> = self
            .axes
            .iter()
            .map(|(axis, _, _)| strides[*axis])
            .collect();
        let base: i64 = self
            .axes
            .iter()
            .map(|(axis, start, _)| start * strides[*axis])
            .sum::<i64>()
            - window;
        // The slice: the outer axis contributes `start` of its own steps, and the walk then covers
        // `rows` of them instead of the whole axis.
        let base = if let Some(step) = step.first().copied() {
            base + start * step
        } else {
            base
        };
        if let Some(first) = shape.first_mut() {
            *first = rows;
        }
        match dtype {
            "bf16" => {
                walk_into(out, &shape, &step, base, |offset| {
                    let byte = offset as usize * 2;
                    f32::from_bits(
                        (u16::from_le_bytes([bytes[byte], bytes[byte + 1]]) as u32) << 16,
                    )
                });
                Ok(())
            }
            "f16" => {
                walk_into(out, &shape, &step, base, |offset| {
                    f16_to_f32(u16::from_le_bytes([
                        bytes[(offset as usize) * 2],
                        bytes[(offset as usize) * 2 + 1],
                    ]))
                });
                Ok(())
            }
            "f32" => {
                walk_into(out, &shape, &step, base, |offset| {
                    let byte = offset as usize * 4;
                    f32::from_le_bytes([
                        bytes[byte],
                        bytes[byte + 1],
                        bytes[byte + 2],
                        bytes[byte + 3],
                    ])
                });
                Ok(())
            }
            // `dtype_width` is the loader's one gate on loadable dtypes and runs before any
            // bytes are read, so an unknown one is a bug here rather than a user error.
            other => unreachable!("dtype_width admitted `{other}` but `fill` cannot read it"),
        }
    }
}

/// Walks `shape` in row-major order, reading `read(offset)` and stepping the source offset by
/// `step[axis]` as each axis advances.
fn walk_into<F: Fn(i64) -> f32>(
    out: &mut Vec<f32>,
    shape: &[usize],
    step: &[i64],
    base: i64,
    read: F,
) {
    let total: usize = shape.iter().product();
    out.clear();
    out.reserve(total);
    if total == 0 {
        return;
    }
    // A scalar tensor has no axes to walk: it is one element, and it is legal in a safetensors
    // header and in a description.
    if shape.is_empty() {
        out.push(read(base));
        return;
    }
    if shape.len() == 1 {
        for i in 0..shape[0] {
            out.push(read(base + i as i64 * step[0]));
        }
        return;
    }
    let mut index = vec![0usize; shape.len()];
    let mut offset = base;
    loop {
        out.push(read(offset));
        let mut axis = shape.len() - 1;
        loop {
            if index[axis] + 1 < shape[axis] {
                index[axis] += 1;
                offset += step[axis];
                break;
            }
            index[axis] = 0;
            offset -= (shape[axis] as i64 - 1) * step[axis];
            if axis == 0 {
                return;
            }
            axis -= 1;
        }
    }
}

/// Row-major strides of a checkpoint tensor's shape, in elements.
fn strides_of(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

/// Reads one checkpoint tensor's bytes and cuts every member's slot out of them.
fn load_group(
    group: &Group,
    rank: usize,
    pool: &std::sync::Mutex<Vec<Vec<f32>>>,
    sink: &std::sync::mpsc::SyncSender<LoadedWeight>,
    abort: &std::sync::atomic::AtomicBool,
) -> Result<LoadStats> {
    // A failed device copy elsewhere means nothing this group produces can be written; stop
    // before the read, not after it.
    if abort.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(LoadStats::default());
    }
    let mut stats = LoadStats {
        tensors_read: 1,
        ..LoadStats::default()
    };

    // ---- what this group has to touch, decided before a byte is read ----
    // The transform, the `split` segment and this rank's shard slabs are all "swap two axes" or
    // "keep a sub-range of one axis", so they compose into one map *before* any data moves
    // (`Cuts`). That map gives two things at once: the exact elements this rank needs — so the
    // read is narrowed to them, instead of reading the whole tensor and cutting it in host memory
    // — and the one pass that turns those bytes into the slot.
    let strides = strides_of(&group.shape);
    let width = dtype_width(&group.dtype)?;
    let numel: i64 = group.shape.iter().try_fold(1i64, |acc, axis| {
        acc.checked_mul(*axis).ok_or_else(|| {
            anyhow::anyhow!(
                "tensor `{}`: the shape's element count overflows",
                group.tensor
            )
        })
    })?;
    let declared_bytes = (group.data.1 - group.data.0) as i64;
    let expected_bytes = numel
        .checked_mul(width as i64)
        .ok_or_else(|| anyhow::anyhow!("tensor `{}`: numel × width overflows", group.tensor))?;
    if declared_bytes != expected_bytes {
        bail!(
            "tensor `{}` {}: the shard declares {declared_bytes} bytes, but the shape times \
             {width}-byte elements is {expected_bytes}",
            group.tensor,
            shape_text(&group.shape)
        );
    }
    // A tensor with a zero-length axis holds nothing to read — but its members still exist and
    // still have to be delivered, or the coverage walk below reports a weight slot that is simply
    // empty as "loaded from no checkpoint tensor".
    let empty = numel == 0;
    let mut planned: Vec<Cuts> = Vec::with_capacity(group.members.len());
    for member in &group.members {
        planned.push(member_cuts(group, member, rank)?);
    }
    if planned.is_empty() {
        return Ok(stats);
    }
    if empty {
        // Nothing to read, but every member has to arrive: its values are empty and its slot is
        // zero-sized, which `Executor::write_f32` accepts and the coverage walk requires.
        for (index, member) in group.members.iter().enumerate() {
            let cuts = &planned[index];
            let values = cuts.fill(&[], &group.dtype, &strides, 0)?;
            if sink
                .send(LoadedWeight {
                    slot: member.slot,
                    name: member.slot_name.clone(),
                    element_offset: 0,
                    values,
                })
                .is_err()
            {
                break;
            }
        }
        return Ok(stats);
    }
    let mut window = (i64::MAX, i64::MIN);
    for cuts in &planned {
        let (low, high) = cuts.source_span(&strides);
        window = (window.0.min(low), window.1.max(high));
    }
    let first = window.0.max(0);
    let last = window.1.min(numel - 1);
    // The bytes to read are the union of the members' runs, not the window that covers them: a
    // tensor whose middle axis is this rank's slab is read per expert, and only in part.
    let mut runs: Vec<(i64, i64)> = Vec::new();
    for cuts in &planned {
        runs.extend(cuts.runs(&group.shape, &strides, READ_RUN_CAP, window));
    }
    runs.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::with_capacity(runs.len());
    for (start, len) in runs {
        match merged.last_mut() {
            // Overlapping or adjacent runs merge; the list stays sorted and disjoint.
            Some((last_start, last_len)) if start <= *last_start + *last_len => {
                *last_len = (*last_start + *last_len).max(start + len) - *last_start;
            }
            _ => merged.push((start, len)),
        }
    }
    if merged.len() > READ_RUN_CAP || merged.is_empty() {
        merged = vec![(first, last - first + 1)];
    }

    // ---- the bytes, one read per run ----
    let read_started = Instant::now();
    let len = ((last - first + 1) as u64)
        .checked_mul(width as u64)
        .ok_or_else(|| anyhow::anyhow!("tensor `{}`: the read length overflows", group.tensor))?
        as usize;
    let mut bytes = vec![0u8; len];
    let file = std::fs::File::open(&group.shard)
        .with_context(|| format!("opening the safetensors shard {}", group.shard.display()))?;
    // `data_offsets` are relative to the *data section*: the shard is 8 bytes of header
    // length, the header, then the data. Reading the length again (8 bytes per shard) is the
    // only way to know where the data begins without re-parsing the header.
    use std::os::unix::fs::FileExt;
    let mut header_len = [0u8; 8];
    file.read_exact_at(&mut header_len, 0)
        .with_context(|| format!("reading the header length of {}", group.shard.display()))?;
    let base = 8 + u64::from_le_bytes(header_len);
    let mut bytes_read = 0u64;
    for (start, len_elems) in &merged {
        let offset = base + group.data.0 + *start as u64 * width as u64;
        let at = (*start - first) as usize * width;
        let take = *len_elems as usize * width;
        file.read_exact_at(&mut bytes[at..at + take], offset)
            .with_context(|| {
                format!(
                    "reading `{}` ({take} bytes at element {start}) from {}",
                    group.tensor,
                    group.shard.display()
                )
            })?;
        bytes_read += take as u64;
    }
    stats.read += read_started.elapsed();
    stats.bytes_read += bytes_read;
    stats.read_runs += merged.len();

    // ---- every member's slot, out of those bytes in one pass ----
    for (index, cuts) in planned.iter().enumerate() {
        if abort.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let member = &group.members[index];
        // Stream the member's outer axis in chunks: the whole-member buffer the loader used to
        // build cost three passes over it (fill, copy, device read) and a fresh page-faulting
        // allocation each time (spec.md §D6.10 measured the load as copy-pipeline bound, 96% of the
        // wall in the writer's span).
        let rows = cuts.axes.first().map(|(_, _, len)| *len).unwrap_or(1);
        let row_elements: usize = cuts
            .axes
            .iter()
            .skip(1)
            .map(|(_, _, len)| *len as usize)
            .product();
        let rows_per_chunk = (CHUNK_BYTES / 4 / row_elements.max(1)).max(1);
        let mut row = 0i64;
        while row < rows {
            let take = (rows - row).min(rows_per_chunk as i64) as usize;
            let fill_started = Instant::now();
            let mut values = pool
                .lock()
                .expect("the chunk pool")
                .pop()
                .unwrap_or_default();
            cuts.fill_chunk_into(
                &mut values,
                &bytes,
                &group.dtype,
                &strides,
                first,
                row,
                take,
            )?;
            stats.fill += fill_started.elapsed();
            // A closed channel means the writer already failed; it reports its own error, so the
            // workers just stop handing it tensors.
            if sink
                .send(LoadedWeight {
                    slot: member.slot,
                    name: member.slot_name.clone(),
                    element_offset: row as usize * row_elements,
                    values,
                })
                .is_err()
            {
                return Ok(stats);
            }
            row += take as i64;
        }
    }
    Ok(stats)
}

/// The composed read map for one member of a group, with both shape facts checked against it.
///
/// All of this is index arithmetic — no data is touched — which is why the group can decide its
/// read window from it before opening the shard, and why a wrong `split` or a wrong slot shape is
/// reported before the checkpoint is read at all.
fn member_cuts(group: &Group, member: &Member, rank: usize) -> Result<Cuts> {
    let mut cuts = Cuts::identity(&group.shape);
    let resolved = |index: i64, what: &str| -> Result<usize> {
        axis(index, group.shape.len()).ok_or_else(|| anyhow::anyhow!("{what}"))
    };
    for step in &group.steps {
        match *step {
            rustrain_model::Transform::Transpose { i, j } => {
                let a = resolved(
                    i,
                    &format!(
                        "transform `transpose({i},{j})`: axis {i} is out of range for {}",
                        shape_text(&group.shape)
                    ),
                )?;
                let b = resolved(
                    j,
                    &format!(
                        "transform `transpose({i},{j})`: axis {j} is out of range for {}",
                        shape_text(&group.shape)
                    ),
                )?;
                cuts.transpose(a, b);
            }
            rustrain_model::Transform::Slice { dim, start, len } => {
                let d = resolved(
                    dim,
                    &format!(
                        "transform `slice({dim},{start},{len})`: axis {dim} is out of range \
                         for {}",
                        shape_text(&group.shape)
                    ),
                )?;
                cuts.narrow(
                    d,
                    start,
                    len,
                    &format!("transform `slice({dim},{start},{len})`"),
                )?;
            }
        }
    }
    if let Some((d, start, len)) = member.segment {
        cuts.narrow(
            d,
            start as i64,
            len as i64,
            &format!("slot `{}`: split segment", member.slot_name),
        )?;
    }

    // The data walk and the shape math are two implementations of one fact: the transform
    // + split result they arrive at must agree, or one of them drifted. The shard cuts come
    // after this check — `expected` describes the split tensor, not this rank's slab.
    let shape = cuts.shape();
    if shape != member.expected {
        bail!(
            "slot `{}` <- `{}` (binding `{}`): the data walk produced shape {:?} but the shape \
             math says {:?}",
            member.slot_name,
            group.tensor,
            member.source,
            shape,
            member.expected
        );
    }

    for &(d, start, len) in &member.shards {
        cuts.narrow(
            d,
            start as i64,
            len as i64,
            &format!("slot `{}`: shard slab on axis {d}", member.slot_name),
        )?;
    }

    // ---- the load-time assertion: the produced tensor IS the slot's local shape ----
    let shape = cuts.shape();
    if shape != member.local_shape {
        bail!(
            "slot `{}` <- `{}` (binding `{}`): transform + split + sharding produce {}, but \
             the slot's local shape on rank {rank} is {}",
            member.slot_name,
            group.tensor,
            member.source,
            shape_text(&shape),
            shape_text(&member.local_shape)
        );
    }
    Ok(cuts)
}

/// The slabs one rank takes out of `after_segment` — the axis lengths a `split` segment left —
/// composed left to right, one entry per declared spec in declaration order.
///
/// `slot_name` is only there for the error messages: a slab a group cannot give this rank is
/// reported against the slot it belongs to.
fn shard_slabs(
    slot_name: &str,
    specs: &[rustrain_parallel::ShardSpec],
    after_segment: &mut [i64],
    rank_dims: usize,
    mesh: &Mesh,
    rank: usize,
) -> Result<Vec<(usize, usize, usize)>> {
    let mut shards: Vec<(usize, usize, usize)> = Vec::new();
    for spec in specs {
        let d = axis(spec.dim, rank_dims).ok_or_else(|| {
            anyhow::anyhow!(
                "slot `{slot_name}`: shard dim {} is out of range for a {rank_dims}-axis tensor",
                spec.dim
            )
        })?;
        let group = spec.group;
        let (degree, coord) = group_degree_and_coord(mesh, rank, group).ok_or_else(|| {
            anyhow::anyhow!(
                "slot `{slot_name}`: shard group {group} cannot be sliced for rank {rank}"
            )
        })?;
        if degree <= 1 {
            continue;
        }
        let global = after_segment[d];
        // The slab comes from the spec being sliced, not from `coord * global / degree`: a
        // declared replicating axis (`tp` with fewer key/value heads than ranks) hands two
        // coordinates the same slice, and only the spec's own mode knows that.
        let (offset, local) = spec
            .mode
            .slab(global, coord as i64, degree as i64)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "slot `{slot_name}`: rank {rank} cannot take a slab of axis {d} (global \
                     {global}, degree {degree}, mode {})",
                    spec.mode
                )
            })?;
        // The next spec on this dim composes on what this one left behind — `local_shape` divides
        // by the product of the two degrees, so the slabs have to compose in the same order the
        // shape algebra assumed. Reading `after_segment[d]` again without this update would slice
        // the second spec against the *original* length and hand every rank a window the slot is
        // not shaped for (the shape check catches it, loudly, but the declaration is legal and
        // would be unloadable).
        after_segment[d] = local;
        shards.push((d, offset as usize, local as usize));
    }
    Ok(shards)
}

/// The pairing errors, with the same wording `check` reports — never weaker ones.
fn reject_broken_pairing(
    expanded: &rustrain_model::Expanded,
    desc: &rustrain_model::ModelDesc,
    meta: &CheckpointMeta,
    p: &crate::Pairing,
) -> Result<()> {
    if !expanded.unbound_slots.is_empty() {
        bail!(
            "{} weight slot(s) have no binding: {}",
            expanded.unbound_slots.len(),
            rustrain_model::summarize(&expanded.unbound_slots)
        );
    }
    if !p.missing_sources.is_empty() {
        bail!(
            "{} binding source(s) match no checkpoint tensor: {}",
            p.missing_sources.len(),
            rustrain_model::summarize(&p.missing_sources)
        );
    }
    if !p.unpaired.is_empty() {
        bail!(
            "{} pairing(s) between a checkpoint tensor and a slot could not be resolved",
            p.unpaired.len()
        );
    }
    if !p.shared_tensors.is_empty() {
        bail!(
            "{} checkpoint tensor(s) are claimed by more than one binding, so one tensor would \
             be loaded into two slots (§3.7 #4)",
            p.shared_tensors.len()
        );
    }
    if !p.is_bijection {
        bail!(
            "the checkpoint↔slot pairing is not one-to-one: {} pairing(s) for the {} weight \
             slot(s) the bindings cover",
            p.pairs.len(),
            p.covered
        );
    }
    let c = consumption(expanded, desc, meta);
    if !c.unconsumed.is_empty() {
        bail!(
            "{} of {} checkpoint tensor(s) are neither consumed by a binding nor matched by an \
             `ignore` entry: {}",
            c.unconsumed.len(),
            meta.tensors.len(),
            rustrain_model::summarize(&c.unconsumed)
        );
    }
    Ok(())
}

/// Element width of a checkpoint dtype the loader can widen; quantized and integer weights are
/// refused by name — the runner executes f32, and widening those would be guessing.
fn dtype_width(dtype: &str) -> Result<usize> {
    match dtype {
        "f32" => Ok(4),
        "f16" | "bf16" => Ok(2),
        other => bail!(
            "checkpoint dtype `{other}` is not loadable: the runner widens bf16, f16 and f32 \
             weights to f32 (the frozen D5 precision decision); a quantized checkpoint needs a \
             quantized provider first"
        ),
    }
}

/// The four-pass chain the loader used to run — widen, transpose, slice, slice — kept as the
/// **reference implementation** the composed `Cuts` walk is tested against
/// (`the_composed_walk_matches_the_explicit_chain`). Nothing in the load path calls it any more:
/// materialising four copies of every tensor was the cost this replaced.
#[cfg(test)]
fn widen(bytes: &[u8], dtype: &str) -> Result<Vec<f32>> {
    match dtype {
        "f32" => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "bf16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| {
                let half = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((half as u32) << 16)
            })
            .collect()),
        "f16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        other => bail!("no widener for checkpoint dtype `{other}`"),
    }
}

/// IEEE 754 binary16 → binary32, exact (f16 ⊂ f32), subnormals included.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h as u32) >> 10) & 0x1f;
    let mant = (h as u32) & 0x3ff;
    let bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                // Subnormal: value = mant × 2^-24. Normalise the mantissa to the
                // f32 hidden-bit form (1 + frac/2^10) × 2^(-14 - shift).
                let mut m = mant;
                let mut shift = 0u32;
                while m & 0x400 == 0 {
                    m <<= 1;
                    shift += 1;
                }
                sign | ((113u32 - shift) << 23) | ((m & 0x3ff) << 13)
            }
        }
        0x1f => sign | (0xff << 23) | (mant << 13), // Inf / NaN, mantissa carried
        _ => sign | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

/// Permutes two axes of a row-major buffer.
#[cfg(test)]
fn transpose_axes(values: &[f32], shape: &[usize], a: usize, b: usize) -> Vec<f32> {
    let rank = shape.len();
    let strides = row_major_strides(shape);
    let mut out_shape = shape.to_vec();
    out_shape.swap(a, b);
    let mut out = vec![0.0f32; values.len()];
    let mut index = vec![0usize; rank];
    for dst in out.iter_mut() {
        let mut offset = 0usize;
        for (d, count) in index.iter().enumerate() {
            let src_axis = if d == a {
                b
            } else if d == b {
                a
            } else {
                d
            };
            offset += *count * strides[src_axis];
        }
        *dst = values[offset];
        for d in (0..rank).rev() {
            index[d] += 1;
            if index[d] < out_shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    out
}

/// Copies the `[start, start + len)` sub-tensor along `dim` out of a row-major buffer.
///
/// The slab is contiguous in memory only when `dim` is the last axis; the general case is an
/// element-wise gather over the output's row-major order (a one-time load cost, never a hot path).
#[cfg(test)]
fn slice_axis(values: &[f32], shape: &[usize], dim: usize, start: usize, len: usize) -> Vec<f32> {
    let strides = row_major_strides(shape);
    let mut out_shape = shape.to_vec();
    out_shape[dim] = len;
    let total: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; total];
    let mut index = vec![0usize; out_shape.len()];
    for dst in out.iter_mut() {
        let mut offset = start * strides[dim];
        for (i, count) in index.iter().enumerate() {
            offset += *count * strides[i];
        }
        *dst = values[offset];
        for d in (0..out_shape.len()).rev() {
            index[d] += 1;
            if index[d] < out_shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    out
}

/// Row-major strides (in elements).
#[cfg(test)]
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// The degree of a single-axis shard group and this rank's coordinate along it.
///
/// `instantiate` produces one `ShardSpec` per declared axis, so a group is a single axis; a
/// multi-axis group cannot be sliced to a local slab by this loader and is refused.
fn group_degree_and_coord(
    mesh: &Mesh,
    rank: usize,
    group: rustrain_parallel::GroupMask,
) -> Option<(usize, usize)> {
    let axes: Vec<usize> = mesh
        .axes()
        .iter()
        .enumerate()
        .filter(|(axis, _)| group.contains(*axis))
        .map(|(axis, _)| axis)
        .collect();
    if axes.len() != 1 {
        return None;
    }
    let axis = axes[0];
    let degree = mesh.degree(axis)?;
    let stride = mesh.stride(axis)?;
    let coord = (rank / stride) % degree;
    Some((degree, coord))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bf16 bytes, exactly representable so the comparison cannot be about rounding.
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for value in values {
            let bits = value.to_bits();
            let lower = bits & 0xffff;
            let mut upper = (bits >> 16) as u16;
            if lower > 0x8000 || (lower == 0x8000 && upper & 1 == 1) {
                upper = upper.wrapping_add(1);
            }
            out.extend_from_slice(&upper.to_le_bytes());
        }
        out
    }

    /// The composed walk is one pass over the checkpoint bytes; the chain it replaced was four
    /// passes with a materialised copy in between. They must agree exactly, on a chain that
    /// exercises every operation a binding can ask for: a transpose, a transform slice, a `split`
    /// segment and a shard slab, in that order, over a 3-D tensor.
    #[test]
    fn the_composed_walk_matches_the_explicit_chain() {
        let source = [3i64, 4, 5];
        let values: Vec<f32> = (0..60).map(|i| i as f32 * 0.5 - 7.0).collect();
        let bytes = bf16_bytes(&values);

        // The chain the loader used to run.
        let mut explicit = widen(&bytes, "bf16").unwrap();
        let mut shape = vec![3usize, 4, 5];
        explicit = transpose_axes(&explicit, &shape, 1, 2);
        shape.swap(1, 2);
        explicit = slice_axis(&explicit, &shape, 0, 1, 2);
        shape[0] = 2;
        explicit = slice_axis(&explicit, &shape, 2, 1, 3);
        shape[2] = 3;
        explicit = slice_axis(&explicit, &shape, 1, 0, 2);
        shape[1] = 2;

        // The composed map: the same four operations, no data moved.
        let mut cuts = Cuts::identity(&source);
        cuts.transpose(1, 2);
        cuts.narrow(0, 1, 2, "test").unwrap();
        cuts.narrow(2, 1, 3, "test").unwrap();
        cuts.narrow(1, 0, 2, "test").unwrap();
        assert_eq!(
            cuts.shape(),
            shape.iter().map(|d| *d as i64).collect::<Vec<_>>()
        );
        let composed = cuts.fill(&bytes, "bf16", &strides_of(&source), 0).unwrap();

        assert_eq!(composed.len(), explicit.len());
        assert_eq!(composed, explicit, "the composed walk must be exact");
    }

    /// The dtypes the loader is allowed to widen, read out of the checkpoint's own bytes. bf16
    /// and f16 are exact subsets of f32, so the comparison is equality, not a tolerance.
    #[test]
    fn fill_reads_every_loadable_dtype_exactly() {
        let values = [1.0f32, -2.5, 3.25];
        let bf16: Vec<u8> = bf16_bytes(&values);
        assert_eq!(
            Cuts::identity(&[3])
                .fill(&bf16, "bf16", &strides_of(&[3]), 0)
                .unwrap(),
            values
        );
        // f16: exact for these too, and a different bit pattern from bf16.
        let mut f16 = Vec::new();
        for value in values {
            let bits = value.to_bits();
            let sign = ((bits >> 16) & 0x8000) as u16;
            let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            let mant = ((bits >> 13) & 0x3ff) as u16;
            f16.extend_from_slice(&(sign | ((exp as u16) << 10) | mant).to_le_bytes());
        }
        assert_eq!(
            Cuts::identity(&[3])
                .fill(&f16, "f16", &strides_of(&[3]), 0)
                .unwrap(),
            values
        );
        let mut f32_bytes = Vec::new();
        for value in values {
            f32_bytes.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(
            Cuts::identity(&[3])
                .fill(&f32_bytes, "f32", &strides_of(&[3]), 0)
                .unwrap(),
            values
        );
    }

    /// A scalar tensor is one element with no axes to walk, and an empty axis is zero elements:
    /// neither may panic, and a scalar used to be loaded fine before the composed walk.
    #[test]
    fn fill_handles_scalars_and_empty_axes() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&7.5f32.to_le_bytes());
        let scalar = Cuts::identity(&[]);
        assert_eq!(
            scalar.fill(&bytes, "f32", &strides_of(&[]), 0).unwrap(),
            vec![7.5]
        );
        assert_eq!(scalar.shape(), Vec::<i64>::new());

        let empty = Cuts::identity(&[0, 3]);
        assert!(
            empty
                .fill(&bytes, "f32", &strides_of(&[0, 3]), 0)
                .unwrap()
                .is_empty()
        );
    }

    /// Two transposes in a row, and a cut on an axis that a later transpose moves: the map has to
    /// compose all of them, not just the single-swap case the fixture happens to use.
    #[test]
    fn the_map_composes_repeated_and_interleaved_operations() {
        let source = [2i64, 3, 4];
        let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let bytes = bf16_bytes(&values);

        let mut explicit = widen(&bytes, "bf16").unwrap();
        let mut shape = vec![2usize, 3, 4];
        explicit = transpose_axes(&explicit, &shape, 0, 1);
        shape.swap(0, 1);
        explicit = slice_axis(&explicit, &shape, 2, 1, 2);
        shape[2] = 2;
        explicit = transpose_axes(&explicit, &shape, 1, 2);
        shape.swap(1, 2);

        let mut cuts = Cuts::identity(&source);
        cuts.transpose(0, 1);
        cuts.narrow(2, 1, 2, "test").unwrap();
        cuts.transpose(1, 2);
        assert_eq!(cuts.shape(), vec![3, 2, 2]);
        assert_eq!(
            cuts.fill(&bytes, "bf16", &strides_of(&source), 0).unwrap(),
            explicit
        );
    }

    /// An aborted load must stop *before* the read. The group points at a shard that cannot be
    /// opened, so a `load_group` that ignores the flag fails loudly and one that honours it
    /// returns with nothing read — which is the difference between "the pool stops when a device
    /// copy fails" and "the pool finishes reading the checkpoint first and then reports".
    #[test]
    fn an_aborted_group_returns_before_touching_the_disk() {
        let group = Group {
            tensor: "model.absent.weight".to_string(),
            shape: vec![2, 2],
            dtype: "bf16".to_string(),
            shard: std::path::PathBuf::from("/nonexistent/rustrain/shard.safetensors"),
            data: (0, 8),
            steps: Vec::new(),
            // One member that wants the whole tensor: the map is the identity, so nothing but the
            // read can fail — and with the flag set, the read must not be attempted at all.
            members: vec![Member {
                slot: SlotId(0),
                slot_name: "model.absent.weight".to_string(),
                source: "model.absent.weight".to_string(),
                local_shape: vec![2, 2],
                segment: None,
                shards: Vec::new(),
                expected: vec![2, 2],
            }],
        };
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        let pool = std::sync::Mutex::new(Vec::new());

        let aborted = std::sync::atomic::AtomicBool::new(true);
        let stats =
            load_group(&group, 0, &pool, &tx, &aborted).expect("an aborted group reads nothing");
        assert_eq!(stats.bytes_read, 0);
        assert_eq!(stats.tensors_read, 0);

        let running = std::sync::atomic::AtomicBool::new(false);
        let error = load_group(&group, 0, &pool, &tx, &running)
            .expect_err("without the flag the missing shard is a real error");
        assert!(
            format!("{error:#}").contains("shard"),
            "the error must be the read, not the flag: {error:#}"
        );
    }

    /// Two shard specs on one dim: the second composes on what the first left behind, exactly as
    /// `local_shape` divides by the product of the two degrees. Before this composed, the second
    /// spec was computed against the *original* axis length, so a declaration `instantiate`
    /// accepted became unloadable (every rank, loudly, at the shape check) — never a silent wrong
    /// write, but a legal plan the loader refused to execute.
    #[test]
    fn a_second_spec_on_one_dim_composes_with_the_first() {
        use rustrain_parallel::{GroupMask, ShardSpec};

        let mesh = Mesh::new(vec![("tp".to_string(), 2), ("ep".to_string(), 2)]).unwrap();
        let tp = GroupMask::single(mesh.index_of("tp").unwrap()).unwrap();
        let ep = GroupMask::single(mesh.index_of("ep").unwrap()).unwrap();
        // Rank 1 of both axes: tp takes [0,4) of an 8-element axis, ep then takes the second half
        // of *that*, not of 8. The mesh's rank order is the mesh's business, so ask it.
        let rank = (0..mesh.world_size())
            .find(|candidate| {
                mesh.group_index(tp, *candidate).ok() == Some(1)
                    && mesh.group_index(ep, *candidate).ok() == Some(1)
            })
            .expect("the mesh has a rank with both coordinates 1");

        let specs = vec![ShardSpec::shard(0, tp), ShardSpec::shard(0, ep)];
        let mut after = vec![8i64, 3];
        let slabs = shard_slabs("w", &specs, &mut after, 2, &mesh, rank).expect("the composed cut");
        assert_eq!(
            slabs,
            vec![(0, 4, 4), (0, 2, 2)],
            "tp halves the axis, ep halves what tp left"
        );
        assert_eq!(after[0], 2, "and the axis is left at the composed length");
    }

    /// A weight tensor with a zero-length axis holds nothing, but its member still has to reach
    /// the executor: the coverage walk reports an undelivered weight slot as "loaded from no
    /// checkpoint tensor", so returning early turned a valid empty tensor into a false failure.
    #[test]
    fn a_zero_element_tensor_still_delivers_its_members() {
        let group = Group {
            tensor: "empty.weight".to_string(),
            shape: vec![0, 3],
            dtype: "bf16".to_string(),
            shard: std::path::PathBuf::from("/nonexistent/rustrain/shard.safetensors"),
            data: (0, 0),
            steps: Vec::new(),
            members: vec![Member {
                slot: SlotId(0),
                slot_name: "empty.weight".to_string(),
                source: "empty.weight".to_string(),
                local_shape: vec![0, 3],
                segment: None,
                shards: Vec::new(),
                expected: vec![0, 3],
            }],
        };
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let pool = std::sync::Mutex::new(Vec::new());
        let running = std::sync::atomic::AtomicBool::new(false);
        let stats =
            load_group(&group, 0, &pool, &tx, &running).expect("an empty tensor is not an error");
        assert_eq!(stats.bytes_read, 0, "and it reads nothing");
        let delivered = rx.try_recv().expect("the member still arrives");
        assert_eq!(delivered.slot, SlotId(0));
        assert!(delivered.values.is_empty());
    }

    /// A zero-length cut has no source span (`start + len - 1`) and its runs would address an empty
    /// slice at a position that may be past the tensor. Nothing can ask for one; the guard is what
    /// keeps that from being a panic instead of a message.
    #[test]
    fn a_zero_length_slice_is_refused() {
        let error = Cuts::identity(&[3, 4])
            .narrow(1, 1, 0, "test")
            .expect_err("a zero-length window is not a cut");
        assert!(
            format!("{error:#}").contains("length 0"),
            "the refusal names the length: {error:#}"
        );
    }

    /// The span is what narrows the read, so it has to be tight where a cut is contiguous and
    /// honest where it is not.
    #[test]
    fn the_source_span_covers_exactly_what_the_member_reads() {
        let strides = strides_of(&[4, 8]);

        // A row range — the shape of every `tp` shard of an output axis — is one contiguous slab.
        let mut rows = Cuts::identity(&[4, 8]);
        rows.narrow(0, 1, 2, "test").unwrap();
        assert_eq!(rows.source_span(&strides), (8, 23), "two whole rows");
        // …and the fill reads nothing outside it.
        let bytes: Vec<u8> = bf16_bytes(&(0..32).map(|i| i as f32).collect::<Vec<f32>>());
        let whole = rows.fill(&bytes, "bf16", &strides, 0).unwrap();
        let narrowed = rows
            .fill(&bytes[8 * 2..24 * 2], "bf16", &strides, 8)
            .unwrap();
        assert_eq!(whole, narrowed);
        assert_eq!(whole.len(), 16);

        // A column range is not contiguous: the span covers the whole rows it crosses, which is
        // the honest trade — one contiguous read of 28 elements instead of a gather of 16.
        let mut cols = Cuts::identity(&[4, 8]);
        cols.narrow(1, 2, 4, "test").unwrap();
        assert_eq!(cols.source_span(&strides), (2, 29));

        // A cut on the innermost axis of a 1-D tensor is contiguous again, element for element.
        let mut flat = Cuts::identity(&[10]);
        flat.narrow(0, 3, 4, "test").unwrap();
        assert_eq!(flat.source_span(&strides_of(&[10])), (3, 6));
    }

    /// The runs are what the narrowed read actually asks the file system for, so they must cover
    /// the box exactly: no element outside it, and none of its elements missing.
    #[test]
    fn the_runs_cover_exactly_the_elements_the_box_reads() {
        let dims = [3i64, 6, 5];
        let strides = strides_of(&dims);

        // A partial *middle* axis — the shape of an expert tensor whose `I` axis is this rank's
        // `tp` slab: three runs, one per expert, each the whole trailing axis.
        let mut middle = Cuts::identity(&dims);
        middle.narrow(1, 2, 3, "test").unwrap();
        let window = middle.source_span(&strides);
        let runs = middle.runs(&dims, &strides, 1024, window);
        assert_eq!(runs.len(), 3, "one run per outer index: {runs:?}");
        assert!(runs.iter().all(|(_, len)| *len == 3 * 5), "{runs:?}");

        // A partial *inner* axis: one run per row, and no row's remainder.
        let mut inner = Cuts::identity(&dims);
        inner.narrow(2, 1, 2, "test").unwrap();
        let window = inner.source_span(&strides);
        let runs = inner.runs(&dims, &strides, 1024, window);
        assert_eq!(runs.len(), 3 * 6, "one run per (outer, middle) pair");

        // Both, checked against a naive walk of the box: the runs must be exactly its elements.
        let mut both = Cuts::identity(&dims);
        both.narrow(1, 1, 4, "test").unwrap();
        both.narrow(2, 2, 2, "test").unwrap();
        let window = both.source_span(&strides);
        let runs = both.runs(&dims, &strides, 4096, window);
        let mut from_runs: Vec<i64> = runs
            .iter()
            .flat_map(|(start, len)| *start..*start + *len)
            .collect();
        from_runs.sort_unstable();
        let mut naive: Vec<i64> = Vec::new();
        for a in 0..3 {
            for b in 1..5 {
                for c in 2..4 {
                    naive.push(a * 30 + b * 5 + c);
                }
            }
        }
        naive.sort_unstable();
        assert_eq!(from_runs, naive);

        // A transpose *and* a partial axis: the innermost partial axis is a different source axis
        // after the swap, so the run length must follow the source strides, not the result order.
        let mut swapped_partial = Cuts::identity(&dims);
        swapped_partial.transpose(0, 2);
        swapped_partial.narrow(2, 1, 2, "test").unwrap();
        let window = swapped_partial.source_span(&strides);
        let runs = swapped_partial.runs(&dims, &strides, 1024, window);
        let mut from_runs: Vec<i64> = runs
            .iter()
            .flat_map(|(start, len)| *start..*start + *len)
            .collect();
        from_runs.sort_unstable();
        let mut naive: Vec<i64> = Vec::new();
        for a in 1..3 {
            for b in 0..6 {
                for c in 0..5 {
                    naive.push(a * 30 + b * 5 + c);
                }
            }
        }
        naive.sort_unstable();
        assert_eq!(from_runs, naive, "the runs must follow the source strides");

        // A cut that starts at zero but keeps less than the axis is still a partial axis.
        let mut prefix = Cuts::identity(&dims);
        prefix.narrow(1, 0, 2, "test").unwrap();
        assert_eq!(
            prefix
                .runs(&dims, &strides, 1024, prefix.source_span(&strides))
                .len(),
            3
        );

        // Past the cap the caller gets the covering window: more bytes, never fewer.
        let capped = both.runs(&dims, &strides, 2, window);
        assert_eq!(capped, vec![(window.0, window.1 - window.0 + 1)]);

        // A box that is the whole tensor is one run however the axes are ordered.
        let mut swapped = Cuts::identity(&dims);
        swapped.transpose(0, 2);
        let window = swapped.source_span(&strides);
        assert_eq!(
            swapped.runs(&dims, &strides, 8, window),
            vec![(window.0, window.1 - window.0 + 1)]
        );
    }

    /// The map is only worth anything if a cut that is out of range is refused rather than read
    /// past the end of the tensor.
    #[test]
    fn a_cut_outside_the_axis_is_refused() {
        let mut cuts = Cuts::identity(&[2, 3]);
        let error = cuts.narrow(1, 2, 2, "test").unwrap_err().to_string();
        assert!(
            error.contains('3') && error.contains("test"),
            "the error must name the extent and the cut: {error}"
        );
        assert_eq!(cuts.shape(), vec![2, 3], "a refused cut changes nothing");
    }

    /// The expected bytes of a small transposed tensor, written out by hand.
    #[test]
    fn transpose_axes_permutes_a_row_major_buffer() {
        // [ [0 1 2], [3 4 5] ] -> transpose(0,1) -> [ [0 3], [1 4], [2 5] ].
        let values: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let out = transpose_axes(&values, &[2, 3], 0, 1);
        assert_eq!(out, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        // 3-D: [2, 2, 3] transpose(1,2): index [e, a, b] -> [e, b, a].
        let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let out = transpose_axes(&values, &[2, 2, 3], 1, 2);
        // [e, b, a] = e*6 + b*3 + a -> reads values[e*6 + a*3 + b].
        let expected: Vec<f32> = (0..12)
            .map(|o| {
                // out_shape [2, 3, 2]: o = e*6 + b*2 + a.
                let e = o / 6;
                let b = (o / 2) % 3;
                let a = o % 2;
                (e * 6 + a * 3 + b) as f32
            })
            .collect();
        assert_eq!(out, expected);
    }

    /// A transposed *and* sliced case with the expected bytes written out — the loader's
    /// combined transform on hand-built data.
    #[test]
    fn transpose_then_slice_produces_the_written_bytes() {
        // The real checkpoint shape class: [E, 2I, H] -> transpose(1,2) -> [E, H, 2I]
        // -> slice(2, 0, I). Small instance: E=2, I=2, H=3.
        let shape = [2usize, 4, 3];
        let values: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let (t, ts) = {
            let t = transpose_axes(&values, &shape, 1, 2);
            let mut ts = shape.to_vec();
            ts.swap(1, 2);
            (t, ts)
        };
        assert_eq!(ts, vec![2, 3, 4]);
        let s = slice_axis(&t, &ts, 2, 0, 2);
        // element [e, h, i] = original [e, i, h] = e*12 + i*3 + h.
        let expected: Vec<f32> = (0..12)
            .map(|o| {
                let e = o / 6;
                let h = (o % 6) / 2;
                let i = o % 2;
                (e * 12 + i * 3 + h) as f32
            })
            .collect();
        assert_eq!(s, expected);
        // The second slice segment: slice(2, 2, 2) — [e, h, i+2].
        let s2 = slice_axis(&t, &ts, 2, 2, 2);
        let expected2: Vec<f32> = (0..12)
            .map(|o| {
                let e = o / 6;
                let h = (o % 6) / 2;
                let i = o % 2;
                (e * 12 + (i + 2) * 3 + h) as f32
            })
            .collect();
        assert_eq!(s2, expected2);
    }

    /// A rank's slice of a declared shard: rank 1 of a dim-0 split takes rows [2, 4).
    #[test]
    fn slice_axis_takes_the_rank_local_slab() {
        let values: Vec<f32> = (0..32).map(|v| v as f32).collect();
        let out = slice_axis(&values, &[4, 8], 0, 2, 2);
        assert_eq!(out, (16..32).map(|v| v as f32).collect::<Vec<_>>());
        let mid = slice_axis(&values, &[4, 8], 1, 4, 2);
        // rows [r][4..6]
        let expected: Vec<f32> = (0..4)
            .flat_map(|r| (r * 8 + 4..r * 8 + 6).map(|v| v as f32))
            .collect();
        assert_eq!(mid, expected);
    }

    /// Widening is exact: bf16 and f16 are subsets of f32.
    #[test]
    fn widening_is_exact() {
        assert_eq!(widen(&1.0f32.to_le_bytes(), "f32").unwrap(), vec![1.0]);
        // 0x3F80 = 1.0 in bf16; 0xC000 = -2.0; 0x0001 is the smallest subnormal.
        assert_eq!(widen(&0x3F80u16.to_le_bytes(), "bf16").unwrap(), vec![1.0]);
        assert_eq!(widen(&0xC000u16.to_le_bytes(), "bf16").unwrap(), vec![-2.0]);
        assert_eq!(
            widen(&0x0001u16.to_le_bytes(), "bf16").unwrap(),
            vec![f32::from_bits(0x0001 << 16)]
        );
        // f16: 0x3C00 = 1.0, 0xC000 = -2.0, 0x0001 = 2^-24.
        assert_eq!(widen(&0x3C00u16.to_le_bytes(), "f16").unwrap(), vec![1.0]);
        assert_eq!(widen(&0xC000u16.to_le_bytes(), "f16").unwrap(), vec![-2.0]);
        assert_eq!(
            widen(&0x0001u16.to_le_bytes(), "f16").unwrap(),
            vec![f32::from_bits((127u32 - 24) << 23)]
        );
        assert!(widen(&[0u8], "f8e4m3").is_err());
        assert!(widen(&[0u8], "i64").is_err());
    }

    /// The loader's dtype gate: only the three float widths it can widen.
    #[test]
    fn loadable_dtypes_are_the_three_widenable_widths() {
        assert_eq!(dtype_width("f32").unwrap(), 4);
        assert_eq!(dtype_width("bf16").unwrap(), 2);
        assert_eq!(dtype_width("f16").unwrap(), 2);
        assert!(dtype_width("f8e4m3").is_err());
        assert!(dtype_width("u8").is_err());
    }

    #[test]
    fn strides_are_row_major() {
        assert_eq!(strides_of(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(strides_of(&[5]), vec![1]);
        assert_eq!(strides_of(&[]), Vec::<i64>::new());
        // The reference chain's own helper must agree with the one the loader uses.
        assert_eq!(
            row_major_strides(&[2, 3, 4])
                .iter()
                .map(|s| *s as i64)
                .collect::<Vec<_>>(),
            strides_of(&[2, 3, 4])
        );
    }
}
