//! D5 acceptance, runnable on this box: the **whole** `run` pipeline — description → expand →
//! instantiate → widen → load real safetensors bytes (with a transpose transform on the data) →
//! compile → execute → dump — on a hand-built tiny checkpoint whose expected output is written
//! out by hand.
//!
//! This is not the GPU evidence (the real comparison against HuggingFace needs the real weights,
//! which this box does not have); it is the pipeline evidence that everything between the CLI
//! argument and the `.npz` bytes works. The checkpoint is real safetensors — a header plus raw
//! bf16 little-endian data, read through the same index/header/data path the real checkpoint
//! takes — and the `.npz` is read back with an independent test-side parser, not with the
//! writer's own code.

use std::path::{Path, PathBuf};
use std::process::Command;

fn cli_binary() -> &'static str {
    option_env!("CARGO_BIN_EXE_rustrain")
        .or(option_env!("CARGO_BIN_EXE_rustrain-cli"))
        .expect("cargo did not inject CARGO_BIN_EXE_<bin>: check rustrain-cli's binary target name")
}

/// f32 → bf16 (round-to-nearest-even on the dropped 16 bits). The test's values are small
/// integers, so the conversion is exact and the expected math is written in f32.
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

/// Writes a one-shard safetensors checkpoint into `dir`: header (8-byte length + JSON with
/// `data_offsets`) followed by the raw little-endian bytes, plus the index.
fn write_checkpoint(dir: &Path) {
    let tensors: Vec<(&str, Vec<i64>, Vec<f32>)> = vec![
        // embed [8, 4]: E[i][j] = i*4 + j + 1
        (
            "model.embed.weight",
            vec![8, 4],
            (0..32).map(|o| (o + 1) as f32).collect(),
        ),
        // up [6, 4] (checkpoint [out, in]): U[k][j] = k*4 + j + 1
        (
            "model.up.weight",
            vec![6, 4],
            (0..24).map(|o| (o + 1) as f32).collect(),
        ),
        // down [4, 6]: D[r][c] = r*6 + c + 1
        (
            "model.down.weight",
            vec![4, 6],
            (0..24).map(|o| (o + 1) as f32).collect(),
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
            "model.up.weight": "model.safetensors",
            "model.down.weight": "model.safetensors",
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

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/run-tiny")
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

/// The whole pipeline, with every number of the expected output written out by hand.
#[test]
fn run_executes_the_tiny_forward_and_writes_the_dump_the_script_reads() {
    let dir = std::env::temp_dir().join(format!("rustrain-run-tiny-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_checkpoint(&dir);

    let out = dir.join("candidate.npz");
    let status = Command::new(cli_binary())
        .args([
            "run",
            // The tiny checkpoint is bf16 but the reference provider computes f32: the test
            // exercises the widened f32 path, which is what `check --dtype f32` and the CPU
            // conformance oracles use.
            "--dtype",
            "f32",
            "--model",
            fixture_model().to_str().unwrap(),
            "--checkpoint",
            dir.to_str().unwrap(),
            "--tokens",
            "0,1,2,3",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        status.status.success(),
        "run failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&status.stdout),
        String::from_utf8_lossy(&status.stderr)
    );

    // ---- the dump contents ----
    assert_eq!(npy_i64(&out, "input_ids.npy"), vec![0, 1, 2, 3]);

    let logits = npy_f32(&out, "logits.npy"); // [4, 4]
    assert_eq!(logits.len(), 16);

    // Expected math, by hand:
    //   embed E[i][j] = i*4+j+1; the probe is ids 0..3, so e_i[j] = i*4+j+1.
    //   up: checkpoint U[k][j] = k*4+j+1 ([6,4]), transpose -> W_up[a][c] = c*4+a+1 ([4,6]):
    //       h_i[c] = sum_a e_i[a] * W_up[a][c].
    //   down: checkpoint D[r][c] = r*6+c+1 ([4,6]), transpose -> W_down[c][r] = r*6+c+1 ([6,4]):
    //       y_i[r] = sum_c h_i[c] * W_down[c][r].
    let mut expected_logits = vec![0.0f32; 16];
    for i in 0..4 {
        // h = [sum_a e_i[a] * W_up[a][c] for c in 0..6], indexed elementwise.
        let h: Vec<f32> = (0..6)
            .map(|c| {
                (0..4)
                    .map(|a| (i * 4 + a + 1) as f32 * (c * 4 + a + 1) as f32)
                    .sum()
            })
            .collect();
        for r in 0..4 {
            expected_logits[i * 4 + r] = h
                .iter()
                .enumerate()
                .map(|(c, v)| v * (r * 6 + c + 1) as f32)
                .sum();
        }
    }
    for (got, want) in logits.iter().zip(&expected_logits) {
        assert!(approx(*got, *want), "logits: got {got}, want {want}");
    }

    // ---- hidden summaries [2, 3]: embed.y and tower.y over the 4 probe rows ----
    let summaries = npy_f32(&out, "hidden_summaries.npy");
    assert_eq!(summaries.len(), 6);
    // embed.y rows = E[0..4] = all values 1..16: mean 8.5, population std of 1..16, max 16.
    let values: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let mean = 8.5f32;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 16.0;
    assert!(approx(summaries[0], mean));
    assert!(approx(summaries[1], var.sqrt()));
    assert!(approx(summaries[2], 16.0));
    // tower.y = the logits: mean/std/max over the same 16 values.
    let y_mean = expected_logits.iter().sum::<f32>() / 16.0;
    let y_var = expected_logits
        .iter()
        .map(|v| (v - y_mean) * (v - y_mean))
        .sum::<f32>()
        / 16.0;
    let y_max = expected_logits
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()));
    assert!(approx(summaries[3], y_mean));
    assert!(approx(summaries[4], y_var.sqrt()));
    assert!(approx(summaries[5], y_max));

    // ---- the report names the pipeline stages and the sidecar exists ----
    let stdout = String::from_utf8_lossy(&status.stdout);
    assert!(
        stdout.contains("layer   0"),
        "the report prints the per-layer summary"
    );
    assert!(
        stdout.contains("layer   1"),
        "the report prints the per-layer summary"
    );
    assert!(stdout.contains("wrote"), "the report names the outputs");
    assert!(
        dir.join("candidate.npz.json").is_file(),
        "the sidecar lands next to the dump"
    );
    // The comparison script reads this token to decide which tolerance the pair gets; without
    // it a bf16 dump was judged against an f32 reference at the f32 bound (spec D6.11).
    let sidecar: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("candidate.npz.json")).unwrap())
            .unwrap();
    assert_eq!(
        sidecar["dtype"], "f32",
        "the sidecar names the dtype as a token, not only as prose"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The probe the comparison script pins, end to end: the dump's input_ids must be exactly those
/// eight ids.
#[test]
fn the_fixed_probe_tokens_land_byte_identical_in_the_dump() {
    let dir = std::env::temp_dir().join(format!("rustrain-run-probe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_checkpoint(&dir);

    let out = dir.join("probe.npz");
    let status = Command::new(cli_binary())
        .args([
            "run",
            // The tiny checkpoint is bf16 but the reference provider computes f32: the test
            // exercises the widened f32 path, which is what `check --dtype f32` and the CPU
            // conformance oracles use.
            "--dtype",
            "f32",
            "--model",
            fixture_model().to_str().unwrap(),
            "--checkpoint",
            dir.to_str().unwrap(),
            "--tokens",
            "0,1,2,3",
            "--seq",
            "4",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        status.status.success(),
        "run failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&status.stdout),
        String::from_utf8_lossy(&status.stderr)
    );
    assert_eq!(npy_i64(&out, "input_ids.npy"), vec![0, 1, 2, 3]);
    std::fs::remove_dir_all(&dir).ok();
}

/// A probe shorter than the window is right-padded with 0 and the dump keeps only the probe
/// rows — the rows a causal model computes identically either way.
#[test]
fn a_short_probe_is_padded_and_the_dump_keeps_the_probe_rows() {
    let dir = std::env::temp_dir().join(format!("rustrain-run-pad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_checkpoint(&dir);

    let out = dir.join("pad.npz");
    let status = Command::new(cli_binary())
        .args([
            "run",
            // The tiny checkpoint is bf16 but the reference provider computes f32: the test
            // exercises the widened f32 path, which is what `check --dtype f32` and the CPU
            // conformance oracles use.
            "--dtype",
            "f32",
            "--model",
            fixture_model().to_str().unwrap(),
            "--checkpoint",
            dir.to_str().unwrap(),
            "--tokens",
            "0,1",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        status.status.success(),
        "run failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&status.stdout),
        String::from_utf8_lossy(&status.stderr)
    );
    assert!(String::from_utf8_lossy(&status.stdout).contains("padded with 0"));

    // The same math as the full probe, restricted to the first two rows.
    let logits = npy_f32(&out, "logits.npy");
    assert_eq!(logits.len(), 8, "2 rows x 4 vocab");
    let mut expected = vec![0.0f32; 8];
    for i in 0..2 {
        let h: Vec<f32> = (0..6)
            .map(|c| {
                (0..4)
                    .map(|a| (i * 4 + a + 1) as f32 * (c * 4 + a + 1) as f32)
                    .sum()
            })
            .collect();
        for r in 0..4 {
            expected[i * 4 + r] = h
                .iter()
                .enumerate()
                .map(|(c, v)| v * (r * 6 + c + 1) as f32)
                .sum();
        }
    }
    for (got, want) in logits.iter().zip(&expected) {
        assert!(approx(*got, *want), "logits: got {got}, want {want}");
    }
    let summaries = npy_f32(&out, "hidden_summaries.npy");
    assert_eq!(summaries.len(), 6, "two hidden states, still [2, 3]");

    std::fs::remove_dir_all(&dir).ok();
}

/// A metadata snapshot has no weight bytes and is refused by name before any shard is opened.
#[test]
fn a_metadata_snapshot_is_refused_by_name() {
    let dir = std::env::temp_dir().join(format!("rustrain-run-meta-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("meta.npz");
    let snapshot = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/check-l2/checkpoints/tiny.safetensors.meta.json");
    let status = Command::new(cli_binary())
        .args([
            "run",
            // The tiny checkpoint is bf16 but the reference provider computes f32: the test
            // exercises the widened f32 path, which is what `check --dtype f32` and the CPU
            // conformance oracles use.
            "--dtype",
            "f32",
            "--model",
            fixture_model().to_str().unwrap(),
            "--checkpoint",
            snapshot.to_str().unwrap(),
            "--tokens",
            "0,1,2,3",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!status.status.success());
    let stderr = String::from_utf8_lossy(&status.stderr);
    assert!(
        stderr.contains("metadata") && stderr.contains("weight bytes"),
        "the refusal names the snapshot form: {stderr}"
    );
    assert!(!out.exists());
    std::fs::remove_dir_all(&dir).ok();
}

/// A probe longer than the description's window, and a `--tokens`/`--seq` disagreement, both fail
/// before any weight is read.
#[test]
fn length_disagreements_fail_early() {
    let dir = std::env::temp_dir().join(format!("rustrain-run-early-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_checkpoint(&dir);

    let out = dir.join("early.npz");
    // tokens != seq: a hard error, not a dump.
    let disagree = Command::new(cli_binary())
        .args([
            "run",
            // The tiny checkpoint is bf16 but the reference provider computes f32: the test
            // exercises the widened f32 path, which is what `check --dtype f32` and the CPU
            // conformance oracles use.
            "--dtype",
            "f32",
            "--model",
            fixture_model().to_str().unwrap(),
            "--checkpoint",
            dir.to_str().unwrap(),
            "--tokens",
            "0,1,2,3",
            "--seq",
            "5",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!disagree.status.success());
    assert!(
        String::from_utf8_lossy(&disagree.stderr).contains("--seq"),
        "the disagreement names --seq: {}",
        String::from_utf8_lossy(&disagree.stderr)
    );
    assert!(!out.exists(), "no dump may be written for a rejected probe");

    // Longer than the declared window of 4: also refused, naming the slot.
    let too_long = Command::new(cli_binary())
        .args([
            "run",
            // The tiny checkpoint is bf16 but the reference provider computes f32: the test
            // exercises the widened f32 path, which is what `check --dtype f32` and the CPU
            // conformance oracles use.
            "--dtype",
            "f32",
            "--model",
            fixture_model().to_str().unwrap(),
            "--checkpoint",
            dir.to_str().unwrap(),
            "--tokens",
            "0,1,2,3,4",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!too_long.status.success());
    assert!(
        String::from_utf8_lossy(&too_long.stderr).contains("input_ids"),
        "the window error names the input slot: {}",
        String::from_utf8_lossy(&too_long.stderr)
    );

    std::fs::remove_dir_all(&dir).ok();
}
