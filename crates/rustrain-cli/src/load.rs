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
use std::path::Path;

use anyhow::{Context, Result, bail};
use rustrain_parallel::Mesh;
use rustrain_plan::{Plan, SlotId};

use crate::{
    CheckpointMeta, axis, consumption, load_checkpoint, pairing, shape_text, split_shape,
    transformed_shape,
};

/// One weight slot's data, ready to write into the executor.
pub(crate) struct LoadedWeight {
    /// The slot in the **instantiated** (local) plan.
    pub slot: SlotId,
    pub name: String,
    /// Widened to f32, laid out contiguously in row-major order of the slot's local shape.
    pub values: Vec<f32>,
    /// How many checkpoint bytes this tensor took.
    pub checkpoint_bytes: u64,
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
pub(crate) fn load_weights(
    expanded: &rustrain_model::Expanded,
    desc: &rustrain_model::ModelDesc,
    plan: &Plan,
    mesh: &Mesh,
    rank: usize,
    checkpoint: &Path,
) -> Result<Vec<LoadedWeight>> {
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

    let mut loaded: Vec<LoadedWeight> = Vec::with_capacity(p.pairs.len());
    let mut seen: Vec<bool> = vec![false; plan.slots.len()];
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
            None => transformed,
        };

        // ---- the bytes ----
        let (shard, (start, end)) = match (&tensor.shard, tensor.data) {
            (Some(shard), Some(range)) => (shard, range),
            _ => bail!(
                "checkpoint tensor `{}` has no weight bytes to load",
                pair.tensor
            ),
        };
        let len = (end - start) as usize;
        let mut bytes = vec![0u8; len];
        let mut file = std::fs::File::open(shard)
            .with_context(|| format!("opening the safetensors shard {}", shard.display()))?;
        // `data_offsets` are relative to the *data section*: the shard is 8 bytes of header
        // length, the header, then the data. Reading the length again (8 bytes per shard) is the
        // only way to know where the data begins without re-parsing the header.
        let mut header_len = [0u8; 8];
        file.read_exact(&mut header_len)
            .with_context(|| format!("reading the header length of {}", shard.display()))?;
        let base = 8 + u64::from_le_bytes(header_len);
        file.seek(SeekFrom::Start(base + start)).with_context(|| {
            format!(
                "seeking `{}` to data byte {start} in {}",
                pair.tensor,
                shard.display()
            )
        })?;
        file.read_exact(&mut bytes).with_context(|| {
            format!(
                "reading `{}` ({len} bytes) from {}",
                pair.tensor,
                shard.display()
            )
        })?;

        let numel: i64 = tensor.shape.iter().product();
        let width = dtype_width(&tensor.dtype)?;
        let expected_bytes = numel
            .checked_mul(width as i64)
            .ok_or_else(|| anyhow::anyhow!("tensor `{}`: numel × width overflows", pair.tensor))?;
        if bytes.len() as i64 != expected_bytes {
            bail!(
                "tensor `{}` {}: the shard holds {} bytes, but the shape times {width}-byte \
                 elements is {expected_bytes}",
                pair.tensor,
                shape_text(&tensor.shape),
                bytes.len()
            );
        }

        // ---- widen → f32, then transform → split → shard on the actual data ----
        let mut values = widen(&bytes, &tensor.dtype)?;
        let mut shape: Vec<usize> = tensor.shape.iter().map(|d| *d as usize).collect();

        let steps: Vec<rustrain_model::Transform> = binding
            .transform
            .iter()
            .map(|step| {
                rustrain_model::parse_transform(step)
                    .map_err(anyhow::Error::msg)
                    .with_context(|| format!("binding `{}`", binding.source))
            })
            .collect::<Result<_>>()?;
        for step in &steps {
            match *step {
                rustrain_model::Transform::Transpose { i, j } => {
                    let a = axis(i, shape.len()).ok_or_else(|| {
                        anyhow::anyhow!(
                            "transform `transpose({i},{j})`: axis {i} is out of range for {}",
                            shape_text(&tensor.shape)
                        )
                    })?;
                    let b = axis(j, shape.len()).ok_or_else(|| {
                        anyhow::anyhow!(
                            "transform `transpose({i},{j})`: axis {j} is out of range for {}",
                            shape_text(&tensor.shape)
                        )
                    })?;
                    values = transpose_axes(&values, &shape, a, b);
                    shape.swap(a, b);
                }
                rustrain_model::Transform::Slice { dim, start, len } => {
                    let d = axis(dim, shape.len()).ok_or_else(|| {
                        anyhow::anyhow!(
                            "transform `slice({dim},{start},{len})`: axis {dim} is out of range \
                             for {}",
                            shape_text(&tensor.shape)
                        )
                    })?;
                    let start = start as usize;
                    let len = len as usize;
                    values = slice_axis(&values, &shape, d, start, len);
                    shape[d] = len;
                }
            }
        }

        if let Some(split) = &binding.split {
            let d = axis(split.dim, shape.len()).ok_or_else(|| {
                anyhow::anyhow!(
                    "split dim {} is out of range for the {} the transform produced",
                    split.dim,
                    shape_text(&tensor.shape)
                )
            })?;
            let sizes: Vec<usize> = split.sizes.iter().map(|s| *s as usize).collect();
            let start: usize = sizes[..pair.segment].iter().sum();
            let len = sizes[pair.segment];
            values = slice_axis(&values, &shape, d, start, len);
            shape[d] = len;
        }

        // The data walk and the shape math are two implementations of one fact: the transform
        // + split result they arrive at must agree, or one of them drifted.
        if shape != expected.iter().map(|d| *d as usize).collect::<Vec<_>>() {
            bail!(
                "slot `{}` <- `{}` (binding `{}`): the data walk produced shape {:?} but the                  shape math says {:?}",
                pair.slot,
                pair.tensor,
                binding.source,
                shape,
                expected
            );
        }

        // ---- this rank's slice of every declared shard ----
        for spec in &slot.layout.dims {
            let d = axis(spec.dim, shape.len()).ok_or_else(|| {
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
            let global = shape[d];
            let local = global / degree;
            values = slice_axis(&values, &shape, d, coord * local, local);
            shape[d] = local;
        }

        // ---- the load-time assertion: the produced tensor IS the slot's local shape ----
        let shape_i64: Vec<i64> = shape.iter().map(|d| *d as i64).collect();
        if shape_i64 != slot.shape {
            bail!(
                "slot `{}` <- `{}` (binding `{}`): transform + split + sharding produce {}, but \
                 the slot's local shape on rank {rank} is {}",
                pair.slot,
                pair.tensor,
                binding.source,
                shape_text(&shape_i64),
                shape_text(&slot.shape)
            );
        }

        seen[slot_id.0] = true;
        loaded.push(LoadedWeight {
            slot: slot_id,
            name: pair.slot.clone(),
            values,
            checkpoint_bytes: len as u64,
        });
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
    Ok(loaded)
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

/// Widens little-endian checkpoint bytes to f32. bf16 and f16 are subsets of f32, so the
/// widening is exact — the 1% comparison tolerance absorbs HF's bf16 rounding, not this one.
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
