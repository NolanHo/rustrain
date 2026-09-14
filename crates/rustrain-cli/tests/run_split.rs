//! The `split` + `transform` binding path, end to end.
//!
//! One checkpoint tensor (`model.fused.weight`, `[9, 4]`) is transposed and then cut along the
//! transformed axis into three **unequal** segments (`[2, 3, 4]`) that feed three different weight
//! slots (`splitter.w_q`, `splitter.w_k`, `splitter.w_v`). All three segments reach the dumped
//! logits — through the three linears, the `cat` that regroups them and the head — so a segment
//! sliced from the wrong offset, with a row stride taken from the wrong size, or aliasing its
//! neighbour's buffer changes the logits *and* the hidden state.
//!
//! Nothing else covers this data path. The loader's own unit tests pin its composed index map
//! (and keep the explicit `widen`/`transpose_axes`/`slice_axis` chain as the reference it is
//! compared against), and every other `run` fixture binds one tensor to one slot, so the
//! composition `transform → split → slot` had no end-to-end test at all: a bug in it would be
//! silent.
//!
//! The expected numbers are computed here, from the checkpoint values, with plain f32 arithmetic —
//! never by calling framework code. Every checkpoint value is an integer below 2^8 (exact in
//! bf16) and every partial sum stays inside f32's exact integer range, so the comparison is
//! expected to be bit-exact; the 1e-6 bound is slack, not a fudge factor.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The probe: the fixture's whole window (`seq = 4`).
const TOKENS: [usize; 4] = [0, 1, 2, 3];
/// `hidden`: the embedding width and the contraction axis of every weight.
const HIDDEN: usize = 4;
/// `vocab`: the embedding table's row count and the logits width.
const VOCAB: usize = 4;
/// `fused`: the length of the axis the binding splits, `q_out + k_out + v_out = 2 + 3 + 4`.
const FUSED: usize = 9;
/// The split sizes, in `split.sizes` order. Unequal on purpose: a segment read from the wrong
/// offset, or with a stride taken from the wrong size, must not land on the right values by
/// coincidence.
const SIZES: [usize; 3] = [2, 3, 4];

fn cli_binary() -> &'static str {
    option_env!("CARGO_BIN_EXE_rustrain")
        .or(option_env!("CARGO_BIN_EXE_rustrain-cli"))
        .expect("cargo did not inject CARGO_BIN_EXE_<bin>: check rustrain-cli's binary target name")
}

/// f32 → bf16 (round-to-nearest-even on the dropped 16 bits). Every value the test writes is an
/// integer below 2^8, where the conversion is exact.
fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let lower = bits & 0xffff;
    let mut upper = (bits >> 16) as u16;
    if lower > 0x8000 || (lower == 0x8000 && upper & 1 == 1) {
        upper = upper.wrapping_add(1);
    }
    upper
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&to_bf16(*v).to_le_bytes());
    }
    out
}

/// `model.embed.weight` `[vocab, hidden]`, value = index + 1.
fn embed_values() -> Vec<f32> {
    (0..VOCAB * HIDDEN).map(|i| (i + 1) as f32).collect()
}

/// `model.fused.weight` `[fused, hidden]` in the checkpoint's own layout, row-major,
/// value = index*3 + 1: asymmetric in both axes, so a segment read from the wrong row, or from
/// the wrong column of the wrong row, cannot coincide with the right one.
fn fused_values() -> Vec<f32> {
    (0..FUSED * HIDDEN).map(|i| (i * 3 + 1) as f32).collect()
}

/// `model.head.weight` `[fused, vocab]`, value = index*5 + 2.
fn head_values() -> Vec<f32> {
    (0..FUSED * VOCAB).map(|i| (i * 5 + 2) as f32).collect()
}

/// The checkpoint's own size: the three tensors as bf16.
fn checkpoint_bytes() -> usize {
    (VOCAB * HIDDEN + FUSED * HIDDEN + FUSED * VOCAB) * 2
}

/// Writes the one-shard safetensors checkpoint the fixture binds (header + bf16 payload + index),
/// the same way `run_multi.rs` and `run_tiny.rs` do — no binary file is checked in.
fn write_checkpoint(dir: &Path) {
    let tensors: Vec<(&str, Vec<i64>, Vec<f32>)> = vec![
        (
            "model.embed.weight",
            vec![VOCAB as i64, HIDDEN as i64],
            embed_values(),
        ),
        (
            "model.fused.weight",
            vec![FUSED as i64, HIDDEN as i64],
            fused_values(),
        ),
        (
            "model.head.weight",
            vec![FUSED as i64, VOCAB as i64],
            head_values(),
        ),
    ];

    let mut payload = Vec::new();
    let mut header = serde_json::Map::new();
    for (name, shape, values) in &tensors {
        let start = payload.len();
        payload.extend_from_slice(&bf16_bytes(values));
        let end = payload.len();
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut shard = Vec::new();
    shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
    shard.extend_from_slice(&header);
    shard.extend_from_slice(&payload);
    std::fs::write(dir.join("model.safetensors"), &shard).unwrap();

    let index = serde_json::json!({
        "metadata": {"total_size": payload.len()},
        "weight_map": {
            "model.embed.weight": "model.safetensors",
            "model.fused.weight": "model.safetensors",
            "model.head.weight": "model.safetensors",
        },
    });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index).unwrap() + "\n",
    )
    .unwrap();
}

// ---- an independent .npz reader (zip central directory + .npy header) ----

fn npz_entry(path: &Path, name: &str) -> Vec<u8> {
    let file = std::fs::read(path).unwrap();
    // EOCD is the last 22 bytes (this writer emits no comment).
    let eocd = file.len() - 22;
    assert_eq!(&file[eocd..eocd + 4], &0x0605_4b50u32.to_le_bytes());
    let count = u16::from_le_bytes([file[eocd + 10], file[eocd + 11]]) as usize;
    let cd_offset = u32::from_le_bytes(file[eocd + 16..eocd + 20].try_into().unwrap()) as usize;
    let mut cursor = cd_offset;
    for _ in 0..count {
        assert_eq!(&file[cursor..cursor + 4], &0x0201_4b50u32.to_le_bytes());
        let method = u16::from_le_bytes([file[cursor + 10], file[cursor + 11]]);
        assert_eq!(method, 0, "the dump must be STORED");
        let size = u32::from_le_bytes(file[cursor + 20..cursor + 24].try_into().unwrap()) as usize;
        let nlen = u16::from_le_bytes([file[cursor + 28], file[cursor + 29]]) as usize;
        let elen = u16::from_le_bytes([file[cursor + 30], file[cursor + 31]]) as usize;
        let clen = u16::from_le_bytes([file[cursor + 32], file[cursor + 33]]) as usize;
        let local = u32::from_le_bytes(file[cursor + 42..cursor + 46].try_into().unwrap()) as usize;
        let entry_name = std::str::from_utf8(&file[cursor + 46..cursor + 46 + nlen]).unwrap();
        if entry_name == name {
            let data_at = local + 30 + nlen + elen;
            let npy = &file[data_at..data_at + size];
            // .npy: magic, version, header length, padded header, then the raw bytes.
            assert_eq!(&npy[..6], &[0x93, b'N', b'U', b'M', b'P', b'Y']);
            let hlen = u16::from_le_bytes([npy[8], npy[9]]) as usize;
            return npy[10 + hlen..].to_vec();
        }
        cursor += 46 + nlen + elen + clen;
    }
    panic!("no `{name}` entry in {}", path.display());
}

fn npy_f32(path: &Path, name: &str) -> Vec<f32> {
    npz_entry(path, name)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn npy_i64(path: &Path, name: &str) -> Vec<i64> {
    npz_entry(path, name)
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/run-tiny-split")
}

/// How the loader walks one segment of the split. Both are bugs this fixture has to separate from
/// the documented read; neither is what the code under test does.
#[derive(Clone, Copy, Debug)]
enum Misread {
    /// The segment read as one contiguous run of the transposed buffer (the "shape × width" read
    /// that ignores the source row stride). The first row comes out right — which is exactly why
    /// the bug is silent without a fixture like this one.
    FlatRun,
    /// The segment read from the wrong source row: the third segment starts one row early.
    OffByOne,
}

/// The expected `(logits, hidden)` of the fixture's forward.
///
/// The documented semantics: `transform` first (the `transpose(0,1)` that turns the checkpoint's
/// `[fused, hidden]` into the `[hidden, fused]` the split cuts), then the split's `sizes` in
/// target order, each segment starting where the previous one ended. Segment `s` is
/// `T[:, dst .. dst + size]`, i.e. the weight `[in, out]` = `[HIDDEN, size]` of the linear that
/// owns it, and `cat` lays the three back out along the fused axis in the same order:
/// `z[t][dst + n] = sum_j x[t][j] * T[j][dst + n]`.
///
/// `misread` re-reads the same intervals through one of the loader bugs above (with the
/// destinations untouched, which is what a loader bug does: it slices from the wrong place but
/// still fills the slot the binding names). Passing `None` computes the expectation itself.
///
/// Plain nested loops in f32, in the accumulation order the kernels document (ascending
/// contraction index), reading nothing but the checkpoint values.
fn expected_forward(misread: Option<Misread>, tokens: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let embed = embed_values();
    let fused = fused_values();
    let head = head_values();

    // The buffer the loader's split runs over: the transposed tensor `T = transpose(fused)`,
    // row-major `[HIDDEN, FUSED]`, so the flat index `j*FUSED + k` reads `fused[k*HIDDEN + j]`.
    let transposed_at = |flat: usize| -> f32 {
        let (j, k) = (flat / FUSED, flat % FUSED);
        fused[k * HIDDEN + j]
    };

    // x[t][j] = embed[ids[t]][j]
    let mut x = vec![vec![0.0f32; HIDDEN]; tokens.len()];
    for (t, id) in tokens.iter().enumerate() {
        for j in 0..HIDDEN {
            x[t][j] = embed[id * HIDDEN + j];
        }
    }

    let mut z = vec![vec![0.0f32; FUSED]; tokens.len()];
    let mut dst = 0usize;
    for (segment, &size) in SIZES.iter().enumerate() {
        // The destination is the sizes' cumulative sum in target order; only the source row the
        // segment is read from is up for misreading.
        let src = match misread {
            Some(Misread::OffByOne) if segment == 2 => dst - 1,
            _ => dst,
        };
        let read = |j: usize, n: usize| -> f32 {
            match misread {
                // One run counted from the segment's first element, regrouped as [HIDDEN, size].
                Some(Misread::FlatRun) => transposed_at(dst + j * size + n),
                // The documented read: T[j][src + n], row-major over a [HIDDEN, FUSED] buffer.
                _ => transposed_at(j * FUSED + src + n),
            }
        };
        for (t, row) in x.iter().enumerate() {
            for n in 0..size {
                z[t][dst + n] = row
                    .iter()
                    .enumerate()
                    .map(|(j, xj)| xj * read(j, n))
                    .sum::<f32>();
            }
        }
        dst += size;
    }

    // logits[t][v] = sum_k z[t][k] * head[k][v]
    let mut logits = vec![0.0f32; tokens.len() * VOCAB];
    for t in 0..tokens.len() {
        for v in 0..VOCAB {
            logits[t * VOCAB + v] = (0..FUSED).map(|k| z[t][k] * head[k * VOCAB + v]).sum();
        }
    }

    // The hidden state the fixture dumps is the cat output itself, [tokens, fused].
    (logits, z.into_iter().flatten().collect())
}

/// Runs the CLI to completion, returning (status, stdout, stderr).
fn run_cli(args: &[&str]) -> (bool, String, String) {
    let output = Command::new(cli_binary()).args(args).output().unwrap();
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

/// The binding path this file exists for: `transpose(0,1)` then `split(dim 1, [2, 3, 4])` over
/// three targets, through the real binary, against values computed here.
#[test]
fn a_transposed_tensor_splits_into_three_asymmetric_slots() {
    let dir = std::env::temp_dir().join(format!("rustrain-run-split-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_checkpoint(&dir);

    let out = dir.join("split.npz");
    let metrics = dir.join("split.json");
    let model_dir = fixture();
    let (ok, stdout, stderr) = run_cli(&[
        "run",
        "--model",
        model_dir.to_str().unwrap(),
        "--checkpoint",
        dir.to_str().unwrap(),
        "--tokens",
        "0,1,2,3",
        "--out",
        out.to_str().unwrap(),
        "--metrics",
        metrics.to_str().unwrap(),
    ]);
    assert!(ok, "run failed:\nstdout: {stdout}\nstderr: {stderr}");

    // ---- one read per tensor, however many segments a split carves out of it ----
    // Four bindings touch three tensors (embed, the fused one through three targets, head), so a
    // loader that reads per *pair* reports more bytes read than the checkpoint holds. This is the
    // arithmetic that says `split` segments share one read.
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&metrics).unwrap()).unwrap();
    let load = &report["ranks"][0]["checkpoint_load"];
    assert_eq!(load["pairs_total"], 5, "three targets plus embed and head");
    assert_eq!(load["tensors_read"], 3, "one read per tensor, not per pair");
    assert_eq!(
        load["bytes_read"], load["bytes_distinct"],
        "the bytes read must be exactly the bytes the checkpoint holds: {load}"
    );
    assert_eq!(
        load["bytes_read"].as_u64().unwrap(),
        checkpoint_bytes() as u64,
        "and they must be the three tensors' own sizes"
    );

    assert_eq!(npy_i64(&out, "input_ids.npy"), vec![0, 1, 2, 3]);

    // ---- the expectation, computed from the checkpoint values alone ----
    // The split's segments are the intervals the sizes carve out: [0, 2), [2, 5), [5, 9).
    let (want_logits, want_hidden) = expected_forward(None, &TOKENS);

    // ---- the logits ----
    let logits = npy_f32(&out, "logits.npy");
    assert_eq!(
        logits.len(),
        TOKENS.len() * VOCAB,
        "logits must be [probe rows, vocab]"
    );
    for (index, (got, want)) in logits.iter().zip(&want_logits).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "logits[{index}] = (row {}, vocab {}): got {got}, want {want} — the split's segments \
             are wrong (sizes {SIZES:?})",
            index / VOCAB,
            index % VOCAB
        );
    }

    // ---- the hidden state: the cat of the three segments, dumped in full ----
    let hidden = npy_f32(&out, "hidden_values.npy");
    assert_eq!(
        hidden.len(),
        TOKENS.len() * FUSED,
        "hidden_values must carry the whole [rows, fused] tensor"
    );
    for (index, (got, want)) in hidden.iter().zip(&want_hidden).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "hidden_values[{index}] = (row {}, fused column {}): got {got}, want {want}",
            index / FUSED,
            index % FUSED
        );
    }

    // ---- the fixture's own teeth ----
    // Each misread re-reads the same intervals the way a broken loader would, and must produce
    // different logits: if a wrong offset or a flat (stride-ignoring) run landed on the same
    // numbers, this test would not be able to tell a correct split from a broken one.
    for misread in [Misread::OffByOne, Misread::FlatRun] {
        let (other_logits, _) = expected_forward(Some(misread), &TOKENS);
        let delta = logits
            .iter()
            .zip(&other_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            delta > 1e-3,
            "the fixture cannot separate the documented split from {misread:?}: the largest logit \
             difference is only {delta} — the checkpoint values or the sizes stopped being \
             asymmetric"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}
