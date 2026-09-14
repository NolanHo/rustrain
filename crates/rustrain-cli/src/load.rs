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

use std::io::{Read, Seek, SeekFrom};
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
    /// Bytes of checkpoint that a *single* read per tensor would have needed: the difference to
    /// `bytes_read` is pure duplication (`split` bindings share one tensor between segments).
    pub bytes_distinct: u64,
    pub tensors_read: usize,
    pub pairs: usize,
    pub read: Duration,
    /// The composed walk: widening, layout and the rank's slab, done in one pass per member.
    pub fill: Duration,
}

/// One weight slot's data, ready to write into the executor.
pub(crate) struct LoadedWeight {
    /// The slot in the **instantiated** (local) plan.
    pub slot: SlotId,
    pub name: String,
    /// Widened to f32, laid out contiguously in row-major order of the slot's local shape.
    pub values: Vec<f32>,
}

/// How many tensors are read and prepared at once.
///
/// The loader used to be one serial pass, and it was the slowest part of a run by two orders of
/// magnitude: the measured split of a 614 s load at 1 thread was transposes 420 s, transform
/// slices 74 s, widening 71 s, reading 34 s — 92% of it host CPU, on one of 160 cores. The
/// checkpoint mount is the mirror image: one stream reads at ~160 MB/s, eight streams at
/// multiple GB/s. One worker per tensor fixes both sides at once, and the memory it costs is one
/// tensor's transients (raw + widened + transformed) per worker.
const LOAD_WORKERS: usize = 16;

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
type GroupOutcome = (usize, Result<(Vec<LoadedWeight>, LoadStats)>);

/// One checkpoint tensor plus every pair that reads it with the same transform.
///
/// A `split` binding produces one pair per segment, and each pair used to read the whole tensor
/// again: 873 pairs over 712 tensors, +64.8% bytes read, widened and transposed. Grouping by
/// `(tensor, transform)` makes the shared work shared, and is why `bytes_read == bytes_distinct`
/// is now an invariant rather than a hope.
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
) -> Result<(Vec<LoadedWeight>, LoadStats)> {
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
        let mut shards: Vec<(usize, usize, usize)> = Vec::new();
        for spec in &slot.layout.dims {
            let d = axis(spec.dim, rank_dims).ok_or_else(|| {
                anyhow::anyhow!(
                    "slot `{}`: shard dim {} is out of range for {}",
                    slot.name,
                    spec.dim,
                    shape_text(&slot.shape)
                )
            })?;
            let group = spec.group;
            let (degree, coord) = group_degree_and_coord(mesh, rank, group).ok_or_else(|| {
                anyhow::anyhow!(
                    "slot `{}`: shard group {group} cannot be sliced for rank {rank}",
                    slot.name
                )
            })?;
            if degree <= 1 {
                continue;
            }
            let global = after_segment[d];
            let local = global / degree as i64;
            shards.push((d, coord * local as usize, local as usize));
        }

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

    // ---- read, widen and transform the groups, in parallel --------------------
    // One worker per tensor, not one per core: the transposes are the bulk of the work and every
    // worker also holds one tensor's raw + widened + transformed bytes, so the pool is sized by
    // what the mount and the memory want, not by the CPU count.
    let workers = LOAD_WORKERS.min(groups.len()).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let outcomes: std::sync::Mutex<Vec<GroupOutcome>> =
        std::sync::Mutex::new(Vec::with_capacity(groups.len()));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(group) = groups.get(index) else {
                        break;
                    };
                    let outcome = load_group(group, rank);
                    outcomes.lock().expect("the loader's result lock").push((index, outcome));
                }
            });
        }
    });

    // Deterministic error reporting: the lowest group index that failed is the one reported, no
    // matter which worker got there first.
    let mut outcomes = outcomes.into_inner().expect("the loader's result lock");
    outcomes.sort_by_key(|(index, _)| *index);

    let mut loaded: Vec<LoadedWeight> = Vec::with_capacity(p.pairs.len());
    let mut seen: Vec<bool> = vec![false; plan.slots.len()];
    let mut stats = LoadStats {
        pairs: p.pairs.len(),
        ..LoadStats::default()
    };
    for (_, outcome) in outcomes {
        let (weights, group_stats) = outcome?;
        stats.bytes_read += group_stats.bytes_read;
        stats.bytes_distinct += group_stats.bytes_distinct;
        stats.tensors_read += group_stats.tensors_read;
        stats.read += group_stats.read;
        stats.fill += group_stats.fill;
        for weight in weights {
            seen[weight.slot.0] = true;
            loaded.push(weight);
        }
    }

    // Every weight slot of *this* plan must have been loaded exactly once; a weight slot with no
    // pairing is an unbound slot (already rejected above), and one loaded twice would be a
    // non-bijective pairing (rejected too) — this walk is the loader's own backstop.
    for (index, slot) in plan.slots.iter().enumerate() {
        if slot.kind == rustrain_plan::SlotKind::Weight && !seen[index] {
            bail!(
                "weight slot `{}` of the rank-{rank} plan was loaded from no checkpoint tensor",
                slot.name
            );
        }
    }

    loaded.sort_by_key(|w| w.slot);
    Ok((loaded, stats))
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
        if start < 0 || len < 0 || start + len > axis_len {
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

    /// Every element of the result, read out of the checkpoint bytes at its composed offset.
    ///
    /// The walk is an odometer over the result's axes with the checkpoint stride of each axis, so
    /// the inner loop is one pointer step and one conversion per element — no division, no
    /// intermediate buffer, and each output element is written exactly once.
    fn fill(&self, bytes: &[u8], dtype: &str, strides: &[i64]) -> Result<Vec<f32>> {
        let shape: Vec<usize> = self.axes.iter().map(|(_, _, len)| *len as usize).collect();
        let step: Vec<i64> = self
            .axes
            .iter()
            .map(|(axis, _, _)| strides[*axis])
            .collect();
        let base: i64 = self
            .axes
            .iter()
            .map(|(axis, start, _)| start * strides[*axis])
            .sum();
        match dtype {
            "bf16" => Ok(walk(&shape, &step, base, |offset| {
                let byte = offset as usize * 2;
                f32::from_bits((u16::from_le_bytes([bytes[byte], bytes[byte + 1]]) as u32) << 16)
            })),
            "f16" => Ok(walk(&shape, &step, base, |offset| {
                f16_to_f32(u16::from_le_bytes([
                    bytes[(offset as usize) * 2],
                    bytes[(offset as usize) * 2 + 1],
                ]))
            })),
            "f32" => Ok(walk(&shape, &step, base, |offset| {
                let byte = offset as usize * 4;
                f32::from_le_bytes([
                    bytes[byte],
                    bytes[byte + 1],
                    bytes[byte + 2],
                    bytes[byte + 3],
                ])
            })),
            other => bail!(
                "the checkpoint declares dtype `{other}` for a weight; the loader widens bf16, \
                 f16 and f32 only"
            ),
        }
    }
}

/// Walks `shape` in row-major order, reading `read(offset)` and stepping the source offset by
/// `step[axis]` as each axis advances.
fn walk<F: Fn(i64) -> f32>(shape: &[usize], step: &[i64], base: i64, read: F) -> Vec<f32> {
    let total: usize = shape.iter().product();
    let mut out: Vec<f32> = Vec::with_capacity(total);
    if total == 0 {
        return out;
    }
    if shape.len() == 1 {
        for i in 0..shape[0] {
            out.push(read(base + i as i64 * step[0]));
        }
        return out;
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
                return out;
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
fn load_group(group: &Group, rank: usize) -> Result<(Vec<LoadedWeight>, LoadStats)> {
    let mut stats = LoadStats {
        tensors_read: 1,
        ..LoadStats::default()
    };

    // ---- the bytes ----
    let read_started = Instant::now();
    let len = (group.data.1 - group.data.0) as usize;
    let mut bytes = vec![0u8; len];
    let mut file = std::fs::File::open(&group.shard).with_context(|| {
        format!("opening the safetensors shard {}", group.shard.display())
    })?;
    // `data_offsets` are relative to the *data section*: the shard is 8 bytes of header
    // length, the header, then the data. Reading the length again (8 bytes per shard) is the
    // only way to know where the data begins without re-parsing the header.
    let mut header_len = [0u8; 8];
    file.read_exact(&mut header_len)
        .with_context(|| format!("reading the header length of {}", group.shard.display()))?;
    let base = 8 + u64::from_le_bytes(header_len);
    file.seek(SeekFrom::Start(base + group.data.0)).with_context(|| {
        format!(
            "seeking `{}` to data byte {} in {}",
            group.tensor,
            group.data.0,
            group.shard.display()
        )
    })?;
    file.read_exact(&mut bytes).with_context(|| {
        format!(
            "reading `{}` ({len} bytes) from {}",
            group.tensor,
            group.shard.display()
        )
    })?;
    stats.read += read_started.elapsed();
    stats.bytes_read += len as u64;
    stats.bytes_distinct += len as u64;

    let numel: i64 = group.shape.iter().product();
    let width = dtype_width(&group.dtype)?;
    let expected_bytes = numel
        .checked_mul(width as i64)
        .ok_or_else(|| anyhow::anyhow!("tensor `{}`: numel × width overflows", group.tensor))?;
    if bytes.len() as i64 != expected_bytes {
        bail!(
            "tensor `{}` {}: the shard holds {} bytes, but the shape times {width}-byte \
             elements is {expected_bytes}",
            group.tensor,
            shape_text(&group.shape),
            bytes.len()
        );
    }

    // ---- every member's slot, read out of the checkpoint bytes in one pass ----
    // The transform, the `split` segment and this rank's shard slabs are all "swap two axes" or
    // "keep a sub-range of one axis", so they compose into one map *before* any data moves
    // (`Cuts`), and each member is then filled directly from the checkpoint bytes. The passes it
    // replaced — widen a copy, transpose a copy, slice a copy, slice a copy — were 92% of a run's
    // startup, and the widened intermediate was one more full copy of the tensor in host memory.
    let strides = strides_of(&group.shape);
    let mut out = Vec::with_capacity(group.members.len());
    for member in &group.members {
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
                    cuts.narrow(d, start, len, &format!("transform `slice({dim},{start},{len})`"))?;
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
        let member_shape = cuts.shape();
        if member_shape != member.expected {
            bail!(
                "slot `{}` <- `{}` (binding `{}`): the data walk produced shape {:?} but the                  shape math says {:?}",
                member.slot_name,
                group.tensor,
                member.source,
                member_shape,
                member.expected
            );
        }

        for &(d, start, len) in &member.shards {
            cuts.narrow(
                d,
                start as i64,
                len as i64,
                &format!(
                    "slot `{}`: shard slab on axis {d}",
                    member.slot_name
                ),
            )?;
        }

        // ---- the load-time assertion: the produced tensor IS the slot's local shape ----
        let shape_i64 = cuts.shape();
        if shape_i64 != member.local_shape {
            bail!(
                "slot `{}` <- `{}` (binding `{}`): transform + split + sharding produce {}, but \
                 the slot's local shape on rank {rank} is {}",
                member.slot_name,
                group.tensor,
                member.source,
                shape_text(&shape_i64),
                shape_text(&member.local_shape)
            );
        }

        let fill_started = Instant::now();
        let values = cuts.fill(&bytes, &group.dtype, &strides)?;
        stats.fill += fill_started.elapsed();
        out.push(LoadedWeight {
            slot: member.slot,
            name: member.slot_name.clone(),
            values,
        });
    }

    Ok((out, stats))
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
        let composed = cuts.fill(&bytes, "bf16", &strides_of(&source)).unwrap();

        assert_eq!(composed.len(), explicit.len());
        assert_eq!(composed, explicit, "the composed walk must be exact");
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
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(row_major_strides(&[5]), vec![1]);
        assert_eq!(row_major_strides(&[]), Vec::<usize>::new());
    }
}
