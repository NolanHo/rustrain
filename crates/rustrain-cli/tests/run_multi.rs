//! D6 acceptance, runnable on this box: the **whole** multi-rank pipeline —
//! description → expand → per-rank instantiate → per-rank weight slices from a
//! real safetensors checkpoint → compile → execute on N threads with real
//! collective exchanges → rank-0 dump — on a hand-built tiny checkpoint.
//!
//! The claim being pinned is D6's own: the sharded forward's logits must agree
//! with the **world-size-1** forward of the same model, both f32, within a
//! relative bound of `1e-5` (the collectives reorder f32 summation, so
//! bit-identity is not the claim; `1e-5` sits ~100× above the measured
//! reassociation noise and orders of magnitude below any wrong slice or
//! dropped partial). The meshes: tp=2, tp=4, tp=2,ep=2, tp=2,cp=2, cp=2,
//! cp=4, dp=2, tp=2,dp=2 — the cp/dp cases against the fixtures that declare
//! the input sharded along those axes, where the runner's completion view
//! forces the all-gather and the dump reads the gathered tensor.
//!
//! `run-tiny-logits` closes the last hole in that table: every other fixture's
//! logits are replicated by the time the dump reads them, so no case gathered
//! a **vocabulary-sharded** tensor — the shape the real model's `lm_head` has.
//! Its head is bound on the vocabulary axis, so rank 0 can only produce the
//! baseline's width and values if the completion view, the inserted gather and
//! the per-member slab placement are all right. (The NCCL half of that placement
//! needs GPUs; it is pinned separately in `rustrain-runtime`'s `nccl` tests and
//! exercised end-to-end on the verification host with this same fixture.)
//!
//! **This box runs the host-assembly collective backend.** A defect that lives
//! only in the NCCL backend's pointer arithmetic is invisible here — one
//! backend's green tests are not evidence about the other.

use std::path::{Path, PathBuf};
use std::process::Command;

fn cli_binary() -> &'static str {
    option_env!("CARGO_BIN_EXE_rustrain")
        .or(option_env!("CARGO_BIN_EXE_rustrain-cli"))
        .expect("cargo did not inject CARGO_BIN_EXE_<bin>: check rustrain-cli's binary target name")
}

/// f32 → bf16 (round-to-nearest-even on the dropped 16 bits).
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

/// The D6 tiny checkpoint: one embed table, two towers' up/down pairs and the
/// vocabulary head, all real safetensors bytes (header + bf16 data + index) —
/// the same shapes every fixture in this file binds. `t0` carries the tp
/// declarations, `t1` the ep ones; the cp/dp fixtures bind the same tensors
/// with no axes (replicated weights, sharded sequence); `run-tiny-logits`
/// binds the head on the vocabulary axis, so its logits are sharded until the
/// runner's completion view gathers them.
fn write_checkpoint(dir: &Path) {
    write_tensors(dir, &tiny_tensors());
}

/// The same checkpoint plus the vocabulary head, for `run-tiny-logits`: a
/// checkpoint tensor no binding consumes is a hard error, so the head can only
/// live in a checkpoint the head fixture is the one to load.
fn write_checkpoint_with_head(dir: &Path) {
    let mut tensors = tiny_tensors();
    // head [8, 4] (checkpoint [vocab, hidden], transposed into the [4, 8]
    // slot): the vocabulary axis is the one that gets sharded, so only the
    // completed (gathered) tensor has the baseline's width.
    tensors.push((
        "model.head.weight",
        vec![8, 4],
        (0..32).map(|o| (o + 5) as f32).collect(),
    ));
    write_tensors(dir, &tensors);
}

/// Every tensor the fixtures other than `run-tiny-logits` bind.
fn tiny_tensors() -> Vec<(&'static str, Vec<i64>, Vec<f32>)> {
    vec![
        // embed [8, 4]: E[i][j] = i*4 + j + 1
        (
            "model.embed.weight",
            vec![8, 4],
            (0..32).map(|o| (o + 1) as f32).collect(),
        ),
        // t0.up [12, 4] (checkpoint [out, in], transposed into the [4, 12] slot)
        (
            "model.t0.up.weight",
            vec![12, 4],
            (0..48).map(|o| (o + 1) as f32).collect(),
        ),
        // t0.down [4, 12]
        (
            "model.t0.down.weight",
            vec![4, 12],
            (0..48).map(|o| (o + 1) as f32).collect(),
        ),
        // t1's pair: distinct values so the second tower is not a copy of the
        // first.
        (
            "model.t1.up.weight",
            vec![12, 4],
            (0..48).map(|o| (o + 8) as f32).collect(),
        ),
        (
            "model.t1.down.weight",
            vec![4, 12],
            (0..48).map(|o| (o + 3) as f32).collect(),
        ),
    ]
}

fn write_tensors(dir: &Path, tensors: &[(&str, Vec<i64>, Vec<f32>)]) {
    let mut payload = Vec::new();
    let mut header = serde_json::Map::new();
    for (name, shape, values) in tensors {
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

    let weight_map: serde_json::Map<String, serde_json::Value> = tensors
        .iter()
        .map(|(name, _, _)| {
            (
                (*name).to_string(),
                serde_json::json!("model.safetensors"),
            )
        })
        .collect();
    let index = serde_json::json!({
        "metadata": {"total_size": payload.len()},
        "weight_map": weight_map,
    });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index).unwrap() + "\n",
    )
    .unwrap();
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

// ---- an independent .npz reader (zip central directory + .npy header) ----

fn npz_entry(path: &Path, name: &str) -> Vec<u8> {
    let file = std::fs::read(path).unwrap();
    let eocd = file.len() - 22;
    assert_eq!(&file[eocd..eocd + 4], &0x0605_4b50u32.to_le_bytes());
    let count = u16::from_le_bytes([file[eocd + 10], file[eocd + 11]]) as usize;
    let cd_offset = u32::from_le_bytes(file[eocd + 16..eocd + 20].try_into().unwrap()) as usize;
    let mut cursor = cd_offset;
    for _ in 0..count {
        assert_eq!(&file[cursor..cursor + 4], &0x0201_4b50u32.to_le_bytes());
        let size = u32::from_le_bytes(file[cursor + 20..cursor + 24].try_into().unwrap()) as usize;
        let nlen = u16::from_le_bytes([file[cursor + 28], file[cursor + 29]]) as usize;
        let elen = u16::from_le_bytes([file[cursor + 30], file[cursor + 31]]) as usize;
        let clen = u16::from_le_bytes([file[cursor + 32], file[cursor + 33]]) as usize;
        let local = u32::from_le_bytes(file[cursor + 42..cursor + 46].try_into().unwrap()) as usize;
        let entry_name = std::str::from_utf8(&file[cursor + 46..cursor + 46 + nlen]).unwrap();
        if entry_name == name {
            let data_at = local + 30 + nlen + elen;
            let npy = &file[data_at..data_at + size];
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

/// Runs the CLI to completion, returning (status, stdout, stderr).
fn run_cli(args: &[&str]) -> (bool, String, String) {
    let output = Command::new(cli_binary()).args(args).output().unwrap();
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

/// The D6 acceptance: for every mesh, the rank-0 logits of the sharded run
/// agree with the world-1 logits of the same fixture within the relative
/// bound. The table is (fixture, mesh flags) — the cp/dp axes need the
/// fixtures that declare them.
#[test]
fn the_sharded_logits_agree_with_the_world1_forward() {
    let dir = std::env::temp_dir().join(format!("rustrain-run-multi-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_checkpoint(&dir);
    // `run-tiny-logits` binds a checkpoint tensor the other fixtures do not, and
    // an unconsumed tensor is a hard error, so it loads its own checkpoint.
    let head_dir = dir.join("with-head");
    std::fs::create_dir_all(&head_dir).unwrap();
    write_checkpoint_with_head(&head_dir);
    let checkpoint_of = |fixture_name: &str| -> &Path {
        if fixture_name == "run-tiny-logits" {
            &head_dir
        } else {
            &dir
        }
    };

    // (fixture, mesh flags, label)
    let cases: &[(&str, &[&str], &str)] = &[
        ("run-tiny-par", &["--tp", "2"], "tp=2"),
        ("run-tiny-par", &["--tp", "4"], "tp=4"),
        ("run-tiny-par", &["--tp", "2", "--ep", "2"], "tp=2,ep=2"),
        ("run-tiny-par", &["--ep", "2"], "ep=2"),
        ("run-tiny-cp", &["--cp", "2"], "cp=2"),
        ("run-tiny-cp", &["--cp", "4"], "cp=4"),
        ("run-tiny-cp", &["--tp", "2", "--cp", "2"], "tp=2,cp=2"),
        ("run-tiny-dp", &["--dp", "2"], "dp=2"),
        ("run-tiny-dp", &["--tp", "2", "--dp", "2"], "tp=2,dp=2"),
        // The vocabulary head is sharded: the dump can only have the baseline's
        // width if the completion view plus its gather ran and landed the
        // members' slabs in member order.
        ("run-tiny-logits", &["--tp", "2"], "tp=2, sharded head"),
        ("run-tiny-logits", &["--tp", "4"], "tp=4, sharded head"),
    ];

    // The world-1 baselines, one per fixture.
    let mut baselines: Vec<(&str, Vec<f32>)> = Vec::new();
    for fixture_name in ["run-tiny-par", "run-tiny-cp", "run-tiny-dp", "run-tiny-logits"] {
        let out = dir.join(format!("base-{fixture_name}.npz"));
        let model_dir = fixture(fixture_name);
        let (ok, _, stderr) = run_cli(&[
            "run",
            "--model",
            model_dir.to_str().unwrap(),
            "--checkpoint",
            checkpoint_of(fixture_name).to_str().unwrap(),
            "--tokens",
            "0,1,2,3",
            "--out",
            out.to_str().unwrap(),
        ]);
        assert!(ok, "world-1 baseline for {fixture_name} failed: {stderr}");
        baselines.push((fixture_name, npy_f32(&out, "logits.npy")));
    }

    for (fixture_name, flags, label) in cases {
        let out = dir.join(format!("case-{}.npz", label.replace(',', "_")));
        let model_dir = fixture(fixture_name);
        let mut args = vec![
            "run",
            "--model",
            model_dir.to_str().unwrap(),
            "--checkpoint",
            checkpoint_of(fixture_name).to_str().unwrap(),
            "--tokens",
            "0,1,2,3",
            "--out",
            out.to_str().unwrap(),
        ];
        args.extend_from_slice(flags);
        let (ok, stdout, stderr) = run_cli(&args);
        assert!(ok, "{label} failed:\nstdout: {stdout}\nstderr: {stderr}");

        let got = npy_f32(&out, "logits.npy");
        let baseline = baselines
            .iter()
            .find(|(f, _)| *f == *fixture_name)
            .map(|(_, b)| b)
            .unwrap();
        assert_eq!(
            got.len(),
            baseline.len(),
            "{label}: the sharded dump must keep the full logits"
        );
        let max_abs: f64 = baseline.iter().fold(0.0, |a, v| a.max((*v as f64).abs()));
        let diff: f64 = got
            .iter()
            .zip(baseline)
            .map(|(a, b)| ((*a as f64) - (*b as f64)).abs())
            .fold(0.0, f64::max);
        let bound = 1e-5 * max_abs;
        assert!(
            diff <= bound,
            "{label}: max|diff| {diff:.3e} exceeds the relative bound {bound:.3e} (baseline \
             max|logits| {max_abs:.3e})"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// The metrics the sweep and the sidecar carry: weight bytes per rank fall as
/// the mesh widens, plan steps are PP-only, and the collective volume is
/// readable per (kind, group).
#[test]
fn the_metrics_report_counts_what_parallel_effects_mean() {
    let dir = std::env::temp_dir().join(format!("rustrain-run-metrics-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_checkpoint(&dir);

    let report = dir.join("sweep.json");
    let (ok, _, stderr) = run_cli(&[
        "run",
        "--model",
        fixture("run-tiny-par").to_str().unwrap(),
        "--checkpoint",
        dir.to_str().unwrap(),
        "--tokens",
        "0,1,2,3",
        "--out",
        report.to_str().unwrap(),
        "--sweep",
        "tp=2;tp=4;tp=2,ep=2",
    ]);
    assert!(ok, "the sweep failed: {stderr}");

    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    assert_eq!(doc["format"], "rustrain.sweep.v1");
    assert_eq!(doc["probe_tokens"], serde_json::json!([0, 1, 2, 3]));

    let baseline_bytes = doc["baseline"]["ranks"][0]["weight_bytes"]
        .as_u64()
        .unwrap();
    assert!(baseline_bytes > 0, "the baseline counts weight bytes");

    let mut last_bytes = baseline_bytes;
    for config in doc["configs"].as_array().unwrap() {
        let per_rank: Vec<u64> = config["ranks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["weight_bytes"].as_u64().unwrap())
            .collect();
        let world = config["world_size"].as_u64().unwrap() as usize;
        assert_eq!(per_rank.len(), world, "one metric entry per rank");
        // Weights per rank must actually fall as tp/ep widen — and stay equal
        // across ranks (uniform sharding).
        assert!(per_rank.iter().all(|b| *b == per_rank[0]));
        assert!(
            per_rank[0] <= last_bytes,
            "{}: {} B after {} B — weights must not grow",
            config["degrees"],
            per_rank[0],
            last_bytes
        );
        if config["degrees"]["tp"].as_u64().unwrap() > 1
            || config["degrees"]["ep"].as_u64().unwrap() > 1
        {
            assert!(
                per_rank[0] < last_bytes,
                "the mesh widened but weights did not fall"
            );
        }
        last_bytes = per_rank[0];
        // Plan steps are PP-only: constant across the accepted meshes.
        for rank in config["ranks"].as_array().unwrap() {
            assert_eq!(
                rank["plan_steps"].as_u64().unwrap(),
                12,
                "steps change with PP only"
            );
            assert_eq!(rank["ops"].as_u64().unwrap(), 10);
            assert_eq!(rank["collectives"].as_u64().unwrap(), 2);
        }
        // The collective volume is readable per (kind, group).
        let kinds = config["ranks"][0]["collectives_by_kind"]
            .as_array()
            .unwrap();
        assert_eq!(kinds.len(), 2, "one all_reduce on tp, one on ep: {kinds:?}");
        for entry in kinds {
            assert_eq!(entry["kind"], "all_reduce");
            assert!(entry["calls"].as_u64().unwrap() >= 1);
            assert!(entry["sent_bytes"].as_u64().unwrap() > 0);
            assert_eq!(entry["sent_bytes"], entry["recv_bytes"]);
        }
        // Every config agrees with the world-1 baseline within the bound.
        assert_eq!(config["pass"], serde_json::Value::Bool(true));
        assert!(config["max_abs_diff"].as_f64().unwrap() <= config["bound"].as_f64().unwrap());
    }
    // tp=2,ep=2 shards both towers: embed stays (128 B), t0 and t1 each
    // halve their two 192 B tensors.
    assert_eq!(
        last_bytes, 512,
        "tp=2,ep=2 shards both towers: embed stays, t0 and t1 halve"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The standalone `--metrics` report lands next to the sidecar and carries one
/// entry per rank.
#[test]
fn run_writes_the_standalone_metrics_report() {
    let dir = std::env::temp_dir().join(format!("rustrain-run-mreport-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_checkpoint(&dir);

    let out = dir.join("case.npz");
    let metrics = dir.join("case.metrics.json");
    let (ok, stdout, stderr) = run_cli(&[
        "run",
        "--model",
        fixture("run-tiny-par").to_str().unwrap(),
        "--checkpoint",
        dir.to_str().unwrap(),
        "--tokens",
        "0,1,2,3",
        "--out",
        out.to_str().unwrap(),
        "--tp",
        "4",
        "--metrics",
        metrics.to_str().unwrap(),
    ]);
    assert!(ok, "the run failed:\nstdout: {stdout}\nstderr: {stderr}");
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&metrics).unwrap()).unwrap();
    assert_eq!(doc["format"], "rustrain.metrics.v1");
    assert_eq!(doc["world_size"], 4);
    assert_eq!(doc["ranks"].as_array().unwrap().len(), 4);
    // The per-rank table is printed for humans too.
    assert!(stdout.contains("per-rank metrics"), "{stdout}");
    assert!(stdout.contains("all_reduce"), "{stdout}");

    std::fs::remove_dir_all(&dir).ok();
}

/// A mesh the tiny fixture cannot divide (inter=12 % 5) is refused loudly with
/// the slot and the numbers named — a compile-time error, exit 1, no dump.
#[test]
fn a_non_dividing_mesh_is_refused_with_the_constraint_named() {
    let dir = std::env::temp_dir().join(format!("rustrain-run-nd-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_checkpoint(&dir);

    let out = dir.join("never.npz");
    let (ok, _, stderr) = run_cli(&[
        "run",
        "--model",
        fixture("run-tiny-par").to_str().unwrap(),
        "--checkpoint",
        dir.to_str().unwrap(),
        "--tokens",
        "0,1,2,3",
        "--out",
        out.to_str().unwrap(),
        "--tp",
        "5",
    ]);
    assert!(!ok, "tp=5 against inter=12 must fail");
    assert!(
        stderr.contains("t0.w_up") && stderr.contains("12"),
        "the refusal must name the slot and the global size: {stderr}"
    );
    assert!(!out.exists(), "no dump may be written for a refused mesh");

    std::fs::remove_dir_all(&dir).ok();
}

/// `pp > 1` stays refused — the cross-stage seam is still an open decision,
/// and refusing loudly is the correct behavior.
#[test]
fn pipeline_parallelism_is_still_refused() {
    let dir = std::env::temp_dir().join(format!("rustrain-run-pp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_checkpoint(&dir);
    let out = dir.join("never.npz");
    let (ok, _, stderr) = run_cli(&[
        "run",
        "--model",
        fixture("run-tiny-par").to_str().unwrap(),
        "--checkpoint",
        dir.to_str().unwrap(),
        "--tokens",
        "0,1,2,3",
        "--out",
        out.to_str().unwrap(),
        "--pp",
        "2",
    ]);
    assert!(!ok);
    assert!(
        stderr.contains("cross-stage seam"),
        "the refusal must name the open decision: {stderr}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The world-1 logits of the tp/ep fixture, hand-computed: pinning the fixture
/// itself (embed → t0 linear pair → silu → t1 linear pair → silu), so the
/// sharded-vs-world-1 agreement cannot be "two sides of the same wrong
/// pipeline". silu is evaluated in f64 and the comparison is loose (the kernel
/// accumulates in f32; the values reach ~2e10, far past f32's exact range).
#[test]
fn the_world1_logits_match_the_hand_computed_math() {
    let dir = std::env::temp_dir().join(format!("rustrain-run-hand-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_checkpoint(&dir);
    let out = dir.join("hand.npz");
    let (ok, _, stderr) = run_cli(&[
        "run",
        "--model",
        fixture("run-tiny-par").to_str().unwrap(),
        "--checkpoint",
        dir.to_str().unwrap(),
        "--tokens",
        "0,1,2,3",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(ok, "the world-1 run failed: {stderr}");
    let got = npy_f32(&out, "logits.npy");

    let silu = |x: f64| x / (1.0 + (-x).exp());
    let mut expected = vec![0.0f64; 16];
    for i in 0..4 {
        // embed.y[a] = i*4 + a + 1 (a < 4).
        // t0: h[c] = Σ_a e[a] * (c*4 + a + 1); y[r] = Σ_c h[c] * (r*12 + c + 1); yc = silu(y).
        let h: Vec<f64> = (0..12)
            .map(|c| {
                (0..4)
                    .map(|a| (i * 4 + a + 1) as f64 * (c * 4 + a + 1) as f64)
                    .sum()
            })
            .collect();
        let y: Vec<f64> = (0..4)
            .map(|r| {
                h.iter()
                    .enumerate()
                    .map(|(c, v)| v * (r * 12 + c + 1) as f64)
                    .sum()
            })
            .map(silu)
            .collect();
        // t1: the checkpoint values are `o + 8` (up) and `o + 3` (down) — the
        // transposed slot reads W1[a][c] = c*4 + a + 8 and W1[c][r] = r*12 + c + 3.
        let h2: Vec<f64> = (0..12)
            .map(|c| (0..4).map(|a| y[a] * (c * 4 + a + 8) as f64).sum())
            .collect();
        for r in 0..4 {
            let logit: f64 = h2
                .iter()
                .enumerate()
                .map(|(c, v)| v * (r * 12 + c + 3) as f64)
                .sum();
            expected[i * 4 + r] = silu(logit);
        }
    }
    let max_abs: f64 = expected.iter().fold(0.0, |a, v| a.max(v.abs()));
    for (got, want) in got.iter().zip(&expected) {
        let diff = ((*got as f64) - want).abs();
        assert!(
            diff <= 1e-3 * max_abs,
            "hand math {want:.3e} vs pipeline {got:.3e} (diff {diff:.3e})"
        );
    }

    // The hidden summaries must read the *hidden* states, not the pool bytes a
    // later activation reused: embed.y is exactly 1..16 (mean 8.5, population
    // std of 1..16, max 16). This pins the planner's view-alias lifetime rule
    // (a `view` output shares its input's storage, so the input must survive
    // as long as the output).
    let summaries = npy_f32(&out, "hidden_summaries.npy");
    let values: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let mean = 8.5f32;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 16.0;
    assert!(
        (summaries[0] - mean).abs() < 1e-4,
        "embed.y mean: {}",
        summaries[0]
    );
    assert!(
        (summaries[1] - var.sqrt()).abs() < 1e-4,
        "embed.y std: {}",
        summaries[1]
    );
    assert!(
        (summaries[2] - 16.0).abs() < 1e-4,
        "embed.y max: {}",
        summaries[2]
    );

    std::fs::remove_dir_all(&dir).ok();
}
