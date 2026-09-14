//! `rustrain run` — spec C4 and delivery D5: one forward, executed by the real runtime against
//! the reference registry, then the candidate dump the HF comparison reads. D6 extends it to a
//! **mesh**: `--tp/--cp/--ep/--dp` drive real multi-rank execution (world = their product), every
//! rank loads its own weight slices and instantiates its own plan, and the collectives run through
//! a real backend (N threads, one per rank, shared buffers).
//!
//! Precision (the frozen decision, stated here because a runner user has to know it): the
//! reference provider is f32-only while the checkpoint and HF are bf16, so `run` widens the
//! bf16 weights to f32 — exact, bf16 ⊂ f32 — and executes f32. The HF reference is dumped with
//! `--dtype bf16`; the spec's 1% tolerance (`max_abs_diff / max_abs` on the logits and the
//! per-layer summaries) absorbs HF's own bf16 rounding, not this widening.
//!
//! D6's numeric claim is a different one, against **our own** world-1 run: the collectives
//! reorder f32 summation (an all-reduce adds partials instead of one sequential accumulation),
//! so bit-identity is not claimed; the acceptance bound is a *relative* one —
//! `max_abs_diff / max_abs(baseline) ≤ 1e-5` — comfortably above the reassociation noise
//! (≲1e-7 for these magnitudes) and orders of magnitude below what a wrong shard slice or a
//! dropped partial would produce.

use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Args;
use rustrain_abi::ffi::RsDtype;
use rustrain_parallel::{GroupMask, Mesh, ParallelConfig, ParallelLayout};
use rustrain_plan::{Plan, SlotId};
use rustrain_runtime::{
    CollectiveBackend, Executor, HostAllocator, NcclBackend, SharedBackend, SingleRank,
    ThreadBackend, ThreadShared, required_inputs,
};

use crate::device::DeviceSpec;
use crate::load::load_weights;
use crate::npz::{self, Npy};

/// The completion view's output slot name — where the runner reads the full
/// (replicated) logits when the plan leaves them distributed.
const LOGITS_COMPLETE: &str = "__run__.logits.complete";
/// The D6 numeric acceptance bound, relative to the baseline's max magnitude.
pub(crate) const AGREEMENT_BOUND_RELATIVE: f64 = 1e-5;
pub(crate) const METRICS_FORMAT: &str = "rustrain.metrics.v1";
pub(crate) const SWEEP_FORMAT: &str = "rustrain.sweep.v1";

#[derive(Args)]
pub(crate) struct RunArgs {
    /// Model directory (`config.json` + `model.json`).
    #[arg(long, value_name = "DIR")]
    pub model: PathBuf,

    /// The **real** safetensors checkpoint: a model directory or its
    /// `model.safetensors.index.json`. A `*.safetensors.meta.json` snapshot has no weight bytes
    /// and is refused.
    #[arg(long, value_name = "PATH")]
    pub checkpoint: PathBuf,

    /// The probe tokens, comma-separated non-negative integers — the comparison script's fixed
    /// probe is `9707,11,1879,0,323,358,314,279`.
    #[arg(long, value_name = "LIST")]
    pub tokens: Option<String>,

    /// Shorthand for `--tokens 0,1,..,<N-1>` (C4). When given beside `--tokens`, the token count
    /// must equal it — a disagreement fails early, before any weight is read.
    #[arg(long, value_name = "N")]
    pub seq: Option<usize>,

    /// Where to write the candidate dump (a `.npz`); a `.json` sidecar lands next to it. In
    /// `--sweep` mode this is the JSON report path instead. Required for a normal run; a rank
    /// child of `launch` reports through `--metrics` and dumps nothing of its own.
    #[arg(long, value_name = "PATH")]
    pub out: Option<PathBuf>,

    /// The mesh degrees. `tp=cp=ep=dp=pp=1` runs the whole model on rank 0 — the first
    /// comparison. Any other combination runs a real multi-rank forward (N threads in this
    /// process, one per rank); `pp > 1` is refused outright because the cross-stage seam is
    /// D5's open decision.
    #[arg(long, default_value_t = 1)]
    pub tp: usize,
    #[arg(long, default_value_t = 1)]
    pub cp: usize,
    #[arg(long, default_value_t = 1)]
    pub ep: usize,
    #[arg(long, default_value_t = 1)]
    pub dp: usize,
    #[arg(long, default_value_t = 1)]
    pub pp: usize,

    /// Write the standalone per-rank metrics report (JSON) here. Absent, the metrics live in the
    /// run's `.json` sidecar.
    #[arg(long, value_name = "PATH")]
    pub metrics: Option<PathBuf>,

    /// Run the same forward over several meshes and write ONE JSON report to `--out` (no `.npz`
    /// dump). Configs are `;`-separated, each a `,`-separated list of `axis=degree`
    /// (`tp`/`cp`/`ep`/`dp`/`pp`). A world-1 baseline always runs first, and every config's
    /// rank-0 logits are compared against it with the D6 bound.
    #[arg(long, value_name = "LIST")]
    pub sweep: Option<String>,

    /// A plugin `.so` to load in addition to the built-in provider.
    #[arg(long = "plugin", value_name = "PATH")]
    pub plugins: Vec<PathBuf>,

    /// Recipe file deciding which implementation runs. Omitted: the built-in reference provider
    /// is selected.
    #[arg(long, value_name = "PATH")]
    pub recipe: Option<PathBuf>,

    /// Where the slot buffers live: `cpu` (default), `cuda`, or `cuda:<index>` (`cuda` alone is
    /// device 0). A CUDA device with a mesh whose world size is > 1 is refused up front **in this
    /// process**: a rank's buffers, its collectives and its callbacks belong to one execution
    /// thread, and a second rank in the same process would have to interleave with it.
    /// `rustrain launch` runs the world as one process per rank instead.
    #[arg(long, value_name = "SPEC", default_value = "cpu")]
    pub device: String,

    /// Run **one rank** of a multi-process world: this process owns rank `N`, its own CUDA
    /// device, and its own collectives. `rustrain launch` passes it; running one rank by hand is
    /// a debugging tool for a stuck world.
    #[arg(long, value_name = "N")]
    pub rank: Option<usize>,

    /// The world size `--rank` belongs to. Defaults to the mesh's own size, and a disagreement
    /// with the mesh is an error rather than a reinterpretation.
    #[arg(long, value_name = "N")]
    pub world: Option<usize>,

    /// The directory the ranks of a multi-process world exchange NCCL's unique ids through. One
    /// per run: a stale file from an earlier run would hand a rank the wrong communicator.
    #[arg(long, value_name = "DIR")]
    pub rdzv: Option<PathBuf>,

    /// The NCCL library to load. Omitted, the loader's own search runs (`libnccl.so.2`, then
    /// `libnccl.so`).
    #[arg(long, value_name = "PATH")]
    pub nccl_lib: Option<PathBuf>,
}

/// One hidden state's summary row, in the comparison's `[mean, std, max]` order.
fn summarize(values: &[f32]) -> [f32; 3] {
    let n = values.len().max(1);
    let mean = values.iter().sum::<f32>() / n as f32;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let max = values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    [mean, var.sqrt(), max]
}

pub(crate) fn run(args: RunArgs) -> Result<()> {
    let device = DeviceSpec::parse(&args.device)?;
    let degrees = [args.tp, args.cp, args.ep, args.dp, args.pp];
    if let Some(degree) = degrees.iter().find(|d| **d == 0) {
        bail!("a mesh degree of {degree} is not a mesh: every axis degree must be at least 1");
    }
    if args.pp > 1 {
        bail!(
            "`run` executes one rank's forward, and with pp={} rank 0 owns only stage 0: the \
             cross-stage seam (complete the partial before the seam, or hand it over) is D5's \
             open decision, so a pipeline-parallel forward is refused rather than dumped \
             half-executed",
            args.pp
        );
    }
    let world = degrees.iter().fold(1usize, |a, b| a.saturating_mul(*b));

    // ---- the probe tokens -----------------------------------------------
    let tokens = probe_tokens(args.tokens.as_deref(), args.seq)?;

    // One rank of a multi-process world: `launch` starts `world` of these, each
    // with its own rank, device and rendezvous directory. The rank runs its own
    // forward and writes its metrics where the launcher reads them — it dumps
    // nothing itself, because the world's dump is rank 0's, assembled by the
    // launcher.
    if let Some(rank) = args.rank {
        return run_one_rank(&args, &tokens, rank, world, device);
    }

    // The CUDA guard applies to *this process*: a context is current on one
    // thread, so one allocator serves one execution thread. A multi-rank CUDA
    // world is one process per rank instead — `rustrain launch` starts it.
    if device.is_cuda() && world > 1 {
        bail!(
            "`--device cuda` with a mesh of world size {world} in one process: a CUDA context can \
             only be current on one host thread at a time, so one allocator/context serves one \
             execution thread. Run the world as one process per rank with `rustrain launch`, or \
             run a single rank here with `--rank 0 --world {world}` (plus `--rdzv <DIR>`), or drop \
             `--device cuda`"
        );
    }

    if let Some(list) = &args.sweep {
        return run_sweep(&args, &tokens, list, device);
    }

    let config = ParallelConfig {
        tensor: args.tp,
        context: args.cp,
        expert: args.ep,
        data: args.dp,
        pipeline: args.pp,
    };
    let result = execute_mesh(
        &args.model,
        &args.checkpoint,
        &tokens,
        &config,
        &args.plugins,
        args.recipe.as_deref(),
        device,
    )
    .with_context(|| {
        format!(
            "executing the forward on the mesh degrees {}",
            mesh_text(&config)
        )
    })?;

    emit_result(&args, &tokens, &config, result)
}

/// One rank of a multi-process world: this process's rank, its own CUDA device,
/// its own NCCL communicator — and its metrics on disk, where the launcher that
/// started it reads them.
///
/// Deliberately not a "small run": the rank loads the model, instantiates its
/// own plan, loads its own weight slabs, compiles, and executes, exactly as a
/// single-process run would. The only difference is where its collectives go and
/// who collects the answer.
fn run_one_rank(
    args: &RunArgs,
    tokens: &[i64],
    rank: usize,
    world: usize,
    device: DeviceSpec,
) -> Result<()> {
    let config = ParallelConfig {
        tensor: args.tp,
        context: args.cp,
        expert: args.ep,
        data: args.dp,
        pipeline: args.pp,
    };
    let mesh = Mesh::from_config(&config);
    if world != mesh.world_size() {
        bail!(
            "`--world {world}` disagrees with the mesh `{}`: the rank's plan and the world it \
             synchronises with must be the same world",
            mesh_text(&config)
        );
    }
    if rank >= world {
        bail!("`--rank {rank}` is outside a world of {world} rank(s)");
    }
    if world > 1 && !device.is_cuda() {
        bail!(
            "a rank child of a multi-process world needs `--device cuda:<index>`: the CPU \
             multi-rank path is N threads in *one* process (run without `--rank`), and the \
             reference provider is the conformance oracle, not an execution path"
        );
    }
    let metrics_path = args.metrics.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "a rank child reports its metrics for the launcher: pass `--metrics <PATH>`"
        )
    })?;

    let transport = if world == 1 {
        // A one-rank world has no peers: no rendezvous directory, no NCCL.
        Transport::Single
    } else {
        let rendezvous = args.rdzv.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "`--rank` in a world of {world} needs `--rdzv <DIR>`: that is the directory the \
                 ranks exchange NCCL's unique ids through, and it must hold no stale ids from an \
                 earlier run"
            )
        })?;
        Transport::Nccl {
            rank,
            world_size: world,
            rendezvous,
            library: args.nccl_lib.as_deref(),
        }
    };
    let started = Instant::now();
    let report = run_rank(
        &args.model,
        &args.checkpoint,
        tokens,
        &mesh,
        rank,
        &transport,
        &args.plugins,
        args.recipe.as_deref(),
        device,
    )
    .with_context(|| format!("rank {rank} of {}", mesh_text(&config)))?;

    if let Some(parent) = metrics_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(metrics_path, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("writing the rank metrics {}", metrics_path.display()))?;
    eprintln!(
        "rank {rank} of {} finished in {:.1} s -> {}",
        mesh_text(&config),
        started.elapsed().as_secs_f64(),
        metrics_path.display()
    );
    Ok(())
}

/// Writes everything a finished world produces: the `.npz` dump, the `.json`
/// sidecar, the optional metrics report, and the human summary on stdout.
///
/// Separate from [`run`] because `launch` assembles the same [`MeshResult`] from
/// `world` child processes and must produce exactly the same artifacts — one
/// dump format, one sidecar format, one report, whatever started the world.
pub(crate) fn emit_result(
    args: &RunArgs,
    tokens: &[i64],
    config: &ParallelConfig,
    result: MeshResult,
) -> Result<()> {
    let out = args.out.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "--out is required: this run dumps the candidate logits there (a rank child of \
             `launch` reports through --metrics instead)"
        )
    })?;
    let window = result.window;
    let vocab = result.vocab;

    // ---- the dump ----------------------------------------------------------
    let mut logits_le = Vec::with_capacity(result.logits.len() * 4);
    for v in &result.logits {
        logits_le.extend_from_slice(&v.to_le_bytes());
    }
    let mut summary_le = Vec::with_capacity(result.summaries.len() * 3 * 4);
    for row in &result.summaries {
        for v in row {
            summary_le.extend_from_slice(&v.to_le_bytes());
        }
    }
    let tokens_i64: Vec<i64> = tokens.to_vec();
    let mut ids_le = Vec::with_capacity(tokens_i64.len() * 8);
    for id in &tokens_i64 {
        ids_le.extend_from_slice(&id.to_le_bytes());
    }

    let mut hidden_le = Vec::with_capacity(result.hidden_values.len() * 4);
    for v in &result.hidden_values {
        hidden_le.extend_from_slice(&v.to_le_bytes());
    }
    npz::write_npz(
        out,
        &[
            Npy {
                name: "input_ids",
                descr: npz::I8,
                shape: &[tokens.len()],
                data: &ids_le,
            },
            Npy {
                name: "logits",
                descr: npz::F4,
                shape: &[tokens.len(), vocab],
                data: &logits_le,
            },
            Npy {
                name: "hidden_summaries",
                descr: npz::F4,
                shape: &[result.summaries.len(), 3],
                data: &summary_le,
            },
            Npy {
                name: "hidden_values",
                descr: npz::F4,
                shape: &[
                    result.summaries.len(),
                    result.hidden_rows,
                    result.hidden_cols,
                ],
                data: &hidden_le,
            },
        ],
    )
    .context("writing the candidate dump")?;

    // ---- the sidecar and the human report --------------------------------
    let sidecar_path = out.with_extension(format!(
        "{}json",
        out.extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default()
    ));
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    let degrees = serde_json::json!({
        "tp": config.tensor, "cp": config.context, "ep": config.expert,
        "dp": config.data, "pp": config.pipeline,
    });
    let mut sidecar = serde_json::json!({
        "format": "rustrain.run.v1",
        "model": args.model.display().to_string(),
        "checkpoint": args.checkpoint.display().to_string(),
        "digest": result.digest,
        "world_size": result.world,
        "degrees": degrees.clone(),
        "window": window,
        "probe_tokens": tokens,
        "logits_shape": [tokens.len(), vocab],
        "hidden_states": result.summaries.len(),
        "hidden_slots": result.hidden_names,
        "weights": result.loaded_count,
        "checkpoint_bytes": result.checkpoint_bytes,
        // Reading, widening and the transposes are separate costs with separate fixes; the
        // load is the slowest part of a run, so its breakdown travels with the run report.
        "checkpoint_load": result.checkpoint_load,
        "steps": result.rank0_steps,
        "ops": result.rank0_ops,
        "collectives": result.rank0_collectives,
        "peak_bytes": result.peak_bytes,
        "wall_seconds": result.wall.as_secs_f64(),
        "precision": "weights widened bf16 -> f32 (exact; bf16 is a subset of f32), forward in f32",
    });
    if result.world > 1 {
        sidecar["metrics"] = serde_json::json!({
            "format": METRICS_FORMAT,
            "world_size": result.world,
            "degrees": degrees,
            "ranks": result.ranks,
            "note": "the logits are the replicated tensor read from rank 0; the hidden \
                     summaries are rank 0's local view of the hidden slots",
        });
        if let Some(path) = &args.metrics {
            std::fs::write(
                path,
                serde_json::to_string_pretty(&sidecar["metrics"])? + "\n",
            )
            .with_context(|| format!("writing the metrics report {}", path.display()))?;
        }
    } else if let Some(path) = &args.metrics {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&serde_json::json!({
                "format": METRICS_FORMAT,
                "world_size": result.world,
                "degrees": degrees,
                "ranks": result.ranks,
            }))? + "\n",
        )
        .with_context(|| format!("writing the metrics report {}", path.display()))?;
    }
    std::fs::write(
        &sidecar_path,
        serde_json::to_string_pretty(&sidecar)? + "\n",
    )
    .with_context(|| format!("writing the sidecar {}", sidecar_path.display()))?;

    println!(
        "run {}  rank 0 of {} (world {})",
        result.name,
        mesh_text(config),
        result.world
    );
    println!("  digest {}", &result.digest[..result.digest.len().min(12)]);
    println!(
        "  weights {} slot(s)  {:.1} GiB checkpoint bytes -> f32 (bf16 ⊂ f32, widening exact)",
        result.loaded_count,
        gib(result.checkpoint_bytes)
    );
    println!(
        "  forward {} step(s) ({} ops, {} collectives)  wall {:.3} s  peak {:.1} GiB",
        result.rank0_steps,
        result.rank0_ops,
        result.rank0_collectives,
        result.wall.as_secs_f64(),
        gib(result.peak_bytes)
    );
    // Where rank 0's own wall clock went, so a skew between ranks can be attributed without a
    // profiler: the collective backends, the first of them (on a NCCL world that is the wait for
    // the other ranks to arrive — the communicators themselves are warmed during the load), the
    // plugin bodies, and the rest (the executor's own walks).
    println!(
        "  rank 0 time: collectives {:.3} s (first {:.3} s), ops {:.3} s, other {:.3} s",
        result.rank0_collective_seconds,
        result.rank0_first_collective_seconds,
        result.rank0_op_seconds,
        (result.wall.as_secs_f64() - result.rank0_collective_seconds - result.rank0_op_seconds)
            .max(0.0)
    );
    println!(
        "  probe {} token(s) in the declared window of {window} (padded with 0; causal execution \
         keeps rows 0..{} exact)",
        tokens.len(),
        tokens.len()
    );
    println!(
        "  kept {} hidden state(s) alive: the plan was extended with one view node each, so the \
         memory pool cannot reuse their bytes before the dump reads them",
        result.hidden_names.len()
    );
    println!(
        "  logits [{}, {}]  hidden summaries [{}, 3]",
        tokens.len(),
        vocab,
        result.summaries.len()
    );
    if result.world > 1 {
        println!("  per-rank metrics (weight bytes, plan steps, collective volume):");
        print_rank_table(&result, &gib);
        println!(
            "    note: hidden summaries below are rank 0's local view; the logits are the \
             replicated tensor"
        );
    }
    println!("  wrote {}", out.display());
    println!("  wrote {}", sidecar_path.display());
    if result.world > 1 {
        if let Some(path) = &args.metrics {
            println!("  wrote {}", path.display());
        }
    }
    println!(
        "  per-layer summary (mean, std, max over the {} probe position(s)):",
        tokens.len()
    );
    for (index, (name, row)) in result
        .hidden_names
        .iter()
        .zip(&result.summaries)
        .enumerate()
    {
        println!(
            "    layer {index:3}  mean {:+.6e}  std {:.6e}  max {:.6e}  ({name})",
            row[0], row[1], row[2]
        );
    }
    Ok(())
}

/// Everything one mesh execution produced, from rank 0's point of view.
pub(crate) struct MeshResult {
    name: String,
    window: i64,
    vocab: usize,
    world: usize,
    logits: Vec<f32>,
    summaries: Vec<[f32; 3]>,
    hidden_names: Vec<String>,
    /// The probe rows of every kept hidden state, row-major
    /// `[hidden states, probe rows, hidden_cols]`.
    hidden_values: Vec<f32>,
    hidden_rows: usize,
    hidden_cols: usize,
    digest: String,
    peak_bytes: u64,
    wall: Duration,
    loaded_count: usize,
    checkpoint_bytes: u64,
    /// Rank 0's loader breakdown (bytes read vs distinct, and the phase timings).
    checkpoint_load: serde_json::Value,
    rank0_steps: usize,
    rank0_ops: usize,
    rank0_collectives: usize,
    /// Rank 0's own forward split into collective time, op time, and the executor's remainder.
    rank0_collective_seconds: f64,
    rank0_first_collective_seconds: f64,
    rank0_op_seconds: f64,
    /// Per-rank metrics (JSON), rank order.
    ranks: Vec<serde_json::Value>,
}

/// Wall-clock seconds since the epoch, for the phase boundaries of one rank's run.
///
/// `Instant` carries no shared epoch — the standard library exposes neither the raw clock nor an
/// origin another process could use — so two ranks' `Instant`s cannot be subtracted. That is what
/// the arrival comparison needs, and `SystemTime` is the only reference two processes on one host
/// share. (Its one failure mode is a clock step between two samples; the forward's own stamps are
/// taken microseconds apart, and a step would have to land between the load stamps to matter.)
fn unix_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// The distinct groups this rank's plan exchanges over, minus the trivial ones.
///
/// A group of one has nothing to form and nothing to warm, and the executor would only waste a
/// communicator on it.
fn collective_groups(compiled: &rustrain_plan::CompiledPlan, mesh: &Mesh) -> Vec<GroupMask> {
    let mut groups: Vec<GroupMask> = Vec::new();
    for step in &compiled.steps {
        if let rustrain_plan::CompiledStep::Intrinsic { group, .. } = step {
            let distributes = group.degree(mesh).map(|degree| degree > 1).unwrap_or(false);
            if distributes && !groups.contains(group) {
                groups.push(*group);
            }
        }
    }
    groups
}

fn mesh_text(cfg: &ParallelConfig) -> String {
    format!(
        "tp={}, cp={}, ep={}, dp={}, pp={}",
        cfg.tensor, cfg.context, cfg.expert, cfg.data, cfg.pipeline
    )
}

/// How one rank's collectives reach the other ranks.
///
/// The three answers are three different machines, not three settings: a single
/// rank needs no transport at all, the CPU world exchanges through shared memory
/// between threads, and a GPU world exchanges through NCCL between processes —
/// because a CUDA context is current on one thread, so one process can only own
/// one rank.
pub(crate) enum Transport<'a> {
    /// World size 1: every collective is a local copy.
    Single,
    /// N ranks as N threads in this process, through shared scratch buffers.
    Threads(&'a Arc<ThreadShared>),
    /// N ranks as N processes, each on its own CUDA device, through NCCL.
    Nccl {
        rank: usize,
        world_size: usize,
        rendezvous: &'a Path,
        library: Option<&'a Path>,
    },
}

/// One rank's forward, or why it failed. Everything the rank touches — model
/// load, instantiate, weight load, compile, execute — happens inside, so a
/// rank is a complete unit the multi-rank driver runs on its own thread or in
/// its own process.
#[allow(clippy::too_many_arguments)]
fn run_rank(
    model_dir: &Path,
    checkpoint: &Path,
    tokens: &[i64],
    mesh: &Mesh,
    rank: usize,
    transport: &Transport<'_>,
    plugins: &[PathBuf],
    recipe_path: Option<&Path>,
    device: DeviceSpec,
) -> Result<serde_json::Value> {
    // ---- description → global plan → this rank's plan --------------------
    let model = rustrain_model::Model::load(model_dir)
        .with_context(|| format!("loading the model description in {}", model_dir.display()))?;
    let expanded = model
        .expand()
        .context("the description did not expand into a runnable plan")?;

    // The providers come first: an operator's shard rule is its own declaration
    // (ABI v2), so `instantiate` reads them from the registry rather than from a
    // framework-side name table (invariant I-5).
    let registry = crate::load_registry(plugins).context("loading the operator providers")?;
    let recipe = crate::load_recipe(recipe_path).context("loading the recipe")?;

    // The transport comes up before any weight is read: a missing NCCL library,
    // a device that will not open, or a rendezvous directory that cannot be
    // created is a configuration error, and it should cost seconds rather than
    // a full checkpoint load on every rank.
    // A NCCL world's group formation (`ncclCommInitRank`, and the id file's root publishing it)
    // blocks every member until the last one arrives, measured at 1.1-2.6 s on the verification
    // host and spent inside the forward's own wall clock. The handle below lets a second thread
    // pay it during the checkpoint load instead; the loader never touches the backend (it writes
    // through the executor's allocator), so the lock stays uncontended until the forward.
    let mut warm_handle: Option<
        std::sync::Arc<std::sync::Mutex<Box<dyn CollectiveBackend + Send>>>,
    > = None;
    let backend: Box<dyn CollectiveBackend + Send> = match transport {
        Transport::Single => Box::new(SingleRank::new(mesh.world_size())),
        Transport::Threads(shared) => {
            Box::new(ThreadBackend::new(rank, mesh.clone(), Arc::clone(shared)))
        }
        Transport::Nccl {
            rank: transport_rank,
            world_size,
            rendezvous,
            library,
        } => {
            if *transport_rank != rank || *world_size != mesh.world_size() {
                bail!(
                    "this process was given rank {rank} but the transport was set up for rank \
                     {transport_rank} of world {world_size} (the mesh has {})",
                    mesh.world_size()
                );
            }
            let index = match device {
                DeviceSpec::Cuda(index) => index,
                DeviceSpec::Cpu => bail!(
                    "the NCCL transport needs a CUDA device: rank {rank} of a multi-process world \
                     runs one rank per GPU"
                ),
            };
            let backend = NcclBackend::new(rank, mesh.clone(), index, rendezvous, *library)
                .map_err(|error| {
                    anyhow::anyhow!("starting NCCL for rank {rank} on device {index}: {error}")
                })?;
            eprintln!(
                "rank {rank} on cuda:{index}: NCCL {} (rank world size {})",
                backend.library(),
                mesh.world_size()
            );
            let shared = SharedBackend::new(Box::new(backend));
            warm_handle = Some(shared.handle());
            Box::new(shared)
        }
    };
    let mut plan = rustrain_plan::instantiate(
        &expanded.plan,
        &expanded.declarations(),
        mesh,
        rank,
        &registry,
    )
    .with_context(|| format!("instantiating rank {rank} of the mesh"))?;

    // The input the runner feeds: the description declares exactly one external input — the token
    // stream. A description with more inputs cannot be fed by `run`, which understands tokens and
    // nothing else, and says so rather than inventing position ids or masks.
    let input_names: Vec<&String> = model.desc.inputs.keys().collect();
    if input_names.len() != 1 {
        bail!(
            "the description declares {} external input(s) ({}), but `run` feeds exactly one — the \
             token stream; it cannot produce the others",
            input_names.len(),
            input_names
                .iter()
                .map(|n| n.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let input_name = input_names[0].as_str();
    let input_id = plan
        .slot_id(input_name)
        .ok_or_else(|| anyhow::anyhow!("input slot `{input_name}` is not a slot of the plan"))?;
    let input_slot = plan.slot(input_id);
    if input_slot.dtype != RsDtype::I64 || input_slot.shape.len() != 1 {
        bail!(
            "input slot `{input_name}` is {} of rank {}; `run` feeds a 1-D i64 token stream",
            input_slot.dtype,
            input_slot.shape.len()
        );
    }
    // The window check runs against the *global* plan: a cp/dp-sharded rank
    // holds a slice of the window, not the whole probe.
    let global_id = expanded
        .plan
        .slot_id(input_name)
        .ok_or_else(|| anyhow::anyhow!("input slot `{input_name}` is not in the global plan"))?;
    let window = expanded.plan.slot(global_id).shape[0];
    if window < 1 {
        bail!("input slot `{input_name}` declares a window of {window} positions");
    }
    if tokens.len() as i64 > window {
        bail!(
            "{} token(s) do not fit the input slot `{input_name}`, which declares a window of \
             {window} positions; the description bakes its sequence length, and a longer probe \
             needs a description with a longer window",
            tokens.len()
        );
    }

    // ---- widen: the frozen precision decision ---------------------------
    widen_to_f32(&mut plan);

    // The hidden states are intermediate activations whose pool storage the memory planner
    // reuses once their last consumer ran — reading them after the forward would read the
    // *later* activation that reused the bytes. So the runner extends the plan: every slot a
    // hidden pattern names gets a `view` node at the end, which makes it a graph output whose
    // storage survives the run (the same thing HF's `output_hidden_states=True` does).
    let outputs = model.desc.outputs.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "the description declares no `outputs` section; `run` needs `outputs.logits` and \
             `outputs.hidden` to know which slots are the logits and the per-layer hidden states \
             (D5's dump contract)"
        )
    })?;
    let hidden_ids = keep_hidden_states(&mut plan, &outputs.hidden)?;

    // The logits the dump reads must be the *complete* tensor. When the plan
    // leaves them distributed (a cp/dp-sharded output, or a partial awaiting
    // its all-reduce), the runner extends the plan with one more view node
    // whose output declares `replicate` — the compiler then inserts exactly
    // the completing collective, the same mechanism `keep_hidden_states`
    // already uses for partials.
    let completed_logits = complete_logits(&mut plan, &outputs.logits)?;

    // ---- compile against the configured providers ------------------------
    // The compile target's device follows the allocator: a CUDA device means
    // CUDA variants resolve and the memory plan aligns slot buffers for the
    // device; the default stays exactly `TargetEnv::default()` (CPU).
    let mut env = rustrain_ops::TargetEnv::default();
    if device.is_cuda() {
        env.device = device.kind();
    }
    let compiled = rustrain_plan::Compiler::new(&registry, &recipe, env)
        .compile(&plan)
        .context(
            "compiling the plan (every node must resolve and its inferred shapes must agree)",
        )?;

    let plan_steps = compiled.steps.len();
    let digest = compiled.digest.clone();
    let peak_bytes = compiled.memory.peak_bytes;

    // The plan is known now, so the groups a NCCL world will form are known too: start paying for
    // their formation in a thread of its own, while this one loads the checkpoint.
    let warm_thread = warm_handle.map(|handle| {
        let groups = collective_groups(&compiled, mesh);
        std::thread::spawn(move || match handle.lock() {
            Ok(mut backend) => backend.warm(&groups),
            Err(_) => Err("the collective backend's lock was poisoned".to_string()),
        })
    });

    // The slot that holds the complete logits after compilation: the
    // completion view's output when one was added, else the pattern's single
    // match — which must then already be replicated.
    let logits_slot =
        resolve_completed_logits(&compiled.plan, completed_logits.is_some(), &outputs.logits)?;

    // ---- feed, execute, read --------------------------------------------
    // The framework owns the memory: host for the CPU default, the
    // runtime-loaded CUDA driver for a device run. One CudaAllocator per rank
    // thread (the CUDA guard above refuses world > 1 with a device).
    let allocator: Box<dyn rustrain_runtime::Allocator + Send> = match device {
        DeviceSpec::Cpu => Box::new(HostAllocator::new()),
        DeviceSpec::Cuda(index) => Box::new(rustrain_runtime::CudaAllocator::new(index).map_err(
            |error| {
                anyhow::anyhow!(
                    "initialising CUDA device {index} for `--device cuda:{index}`: the driver \
                     could not be loaded or initialised: {error}"
                )
            },
        )?),
    };
    let mut executor =
        Executor::new(compiled, allocator, backend).context("preparing the executor")?;

    // Every slot no node produces must be fed: weights from the loader, the token stream from the
    // probe. Anything else is an input this runner cannot produce — named, not guessed.
    for (slot, kind) in required_inputs(executor.plan()) {
        match kind {
            rustrain_plan::SlotKind::Weight => {}
            rustrain_plan::SlotKind::Input if slot == input_id => {}
            other => bail!(
                "slot `{}` ({other:?}) is produced by no node, and `run` cannot feed it",
                executor.plan().plan.slot(slot).name
            ),
        }
    }

    // The probe padded to the declared window: causal execution means positions 0..n-1 never read
    // the pad, so rows 0..n-1 are exactly HF's n-token forward (the same right-padding HF itself
    // uses); the dump then keeps only those rows. A rank whose input slot is sharded along the
    // sequence feeds only its own slice of the padded probe.
    let mut ids = vec![0i64; window as usize];
    ids[..tokens.len()].copy_from_slice(tokens);
    let rows = rank_input_rows(executor.plan(), input_id, mesh, rank, &ids)?;
    let mut id_bytes = Vec::with_capacity(rows.len() * 8);
    for id in rows {
        id_bytes.extend_from_slice(&id.to_le_bytes());
    }
    executor
        .write_raw(input_id, &id_bytes)
        .context("feeding the token stream")?;

    // ---- the weights, through the same pairing `check` verifies ---------
    // The loader writes each slot into the executor as it is prepared, so the host does not hold
    // a second, f32 copy of every weight and the device copies overlap the reads. The rank-local
    // metric that must fall as the mesh widens is the widened f32 bytes the rank holds — which is
    // what `weight_bytes` counts, not the raw read.
    let load = load_weights(
        &expanded,
        &model.desc,
        &plan,
        mesh,
        rank,
        checkpoint,
        &mut executor,
    )
    .context("loading the checkpoint weights")?;
    let loaded_count = load.slots;
    let rank_weight_bytes = load.weight_bytes;
    let checkpoint_bytes = load.stats.bytes_read;
    let write_seconds = load.write.as_secs_f64();
    let load_finished_unix = unix_seconds();
    // The warm is over before this point in any real run (loading 67 GB outlasts a communicator
    // handshake and its arrival skew), but it is not *assumed* to be: the join is what makes the
    // forward's clock start on a warmed backend, and how long that join waited is reported as
    // `warm_seconds` so the wait is attributed rather than lost. A failure is reported too — the
    // first collective of that group would otherwise fail with the same reason, later and less
    // clearly.
    let mut warm_seconds = 0.0;
    if let Some(thread) = warm_thread {
        let joined_at = Instant::now();
        // A failed warm stops the rank. Continuing would be worse than a crash: `comm()` publishes
        // the rendezvous id *before* it initialises, so a retry at the first collective
        // regenerates the id and overwrites the file while the peers are parked inside
        // `ncclCommInitRank` with the previous one — a world-wide deadlock that no rank exits and
        // no metric reports. Failing here leaves the peers to be terminated by the launcher, which
        // reports *this* rank's reason.
        match thread.join() {
            Ok(Ok(())) => {}
            Ok(Err(reason)) => bail!(
                "rank {rank}: warming the collective groups failed: {reason}; the rank stops \
                 rather than enter a rendezvous its peers have already left"
            ),
            Err(_) => bail!(
                "rank {rank}: the collective warm-up thread panicked; the rank stops rather than \
                 enter a rendezvous its peers have already left"
            ),
        }
        warm_seconds = joined_at.elapsed().as_secs_f64();
    }
    let warm_finished_unix = unix_seconds();

    // The pre-forward phases are timed against the wall clock, not against an `Instant`: ranks
    // are separate processes on one host, so `SystemTime` is the only clock that can say which
    // rank reached the first collective first. That is the whole question behind a per-rank wall
    // difference — a rank that arrives early pays the group's formation inside its own forward.
    let forward_started_unix = unix_seconds();
    let started = Instant::now();
    let stats = executor.run().context("executing the forward")?;
    let wall = started.elapsed();
    let forward_finished_unix = unix_seconds();

    // ---- the outputs, from rank 0 only -----------------------------------
    let (logits, vocab, summaries, hidden_names, hidden_values, hidden_cols) = if rank == 0 {
        let logits_bytes = executor.read_raw(logits_slot).with_context(|| {
            format!(
                "reading the logits slot `{}`",
                executor.plan().plan.slot(logits_slot).name
            )
        })?;
        let logits: Vec<f32> = logits_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let logits_shape = executor.plan().plan.slot(logits_slot).shape.clone();
        if logits_shape.len() != 2 || logits_shape[0] != window {
            bail!(
                "the logits slot `{}` is {:?}; the dump needs [window, vocab] = [{window}, …]",
                executor.plan().plan.slot(logits_slot).name,
                logits_shape
            );
        }
        let vocab = logits_shape[1] as usize;
        let probe_logits = &logits[..tokens.len() * vocab];

        let mut summaries: Vec<[f32; 3]> = Vec::new();
        let mut hidden_names: Vec<String> = Vec::new();
        let mut probe_rows: Vec<f32> = Vec::new();
        let mut probe_cols: usize = 0;
        for (id, name) in &hidden_ids {
            let name = name.clone();
            let bytes = executor
                .read_raw(*id)
                .with_context(|| format!("reading the hidden state slot `{name}`"))?;
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let shape = executor.plan().plan.slot(*id).shape.clone();
            let per_row: usize = shape[1..]
                .iter()
                .map(|d| *d as usize)
                .product::<usize>()
                .max(1);
            // Only the probe rows: the pad rows are not part of the HF tensor.
            let probe: &[f32] = &values[..tokens.len().min(shape[0] as usize) * per_row];
            summaries.push(summarize(probe));
            // The rows themselves, so a comparison can be made per element instead of
            // three statistics per layer — the summaries say a layer differs, these
            // say where.
            probe_rows.extend_from_slice(probe);
            probe_cols = per_row;
            hidden_names.push(name);
        }
        if summaries.is_empty() {
            bail!(
                "`outputs.hidden` matched no slot of the rank-0 plan; the dump needs at least one \
                 hidden state"
            );
        }
        (
            probe_logits.to_vec(),
            vocab,
            summaries,
            hidden_names,
            probe_rows,
            probe_cols,
        )
    } else {
        (Vec::new(), 0, Vec::new(), Vec::new(), Vec::new(), 0)
    };

    // ---- the per-rank metrics --------------------------------------------
    let collectives_by_kind = collective_breakdown(mesh, &stats);
    Ok(serde_json::json!({
        "rank": rank,
        "weight_slots": loaded_count,
        "weight_bytes": rank_weight_bytes,
        "checkpoint_bytes_read": checkpoint_bytes,
        // The phases the load splits into. `bytes_distinct` is what a full read of every tensor
        // this rank binds would cost; `bytes_read` is what it actually read. They are equal at
        // world 1 and `bytes_read` is smaller on a sharded mesh, by the slices other ranks own.
        "checkpoint_load": {
            "bytes_read": load.stats.bytes_read,
            "bytes_distinct": load.stats.bytes_distinct,
            "tensors_read": load.stats.tensors_read,
            "pairs_total": load.stats.pairs_total,
            "workers": load.workers,
            "read_runs": load.stats.read_runs,
            // The load's own wall clock, and the two sums that run inside it: `read` and `fill`
            // are added up over `workers` threads, so they must NOT be summed with each other or
            // with the wall time; the device writes happen on the calling thread and overlap both.
            "wall_seconds": load.wall.as_secs_f64(),
            "read_cpu_seconds": load.stats.read.as_secs_f64(),
            "fill_cpu_seconds": load.stats.fill.as_secs_f64(),
            "write_seconds": write_seconds,
            // The writer's other half: how long it waited for the pool. `write_seconds +
            // write_wait_seconds` spans the streaming phase, so a load that is slow on the copy
            // side and one that is slow on the read side are told apart by these two numbers.
            "write_wait_seconds": load.write_wait.as_secs_f64(),
            "write_chunks": load.write_chunks,
            // Copy time per chunk, bucketed: a uniform transport and a fast path with outliers have
            // the same average and completely different fixes.
            "write_histogram_us": load
                .write_histogram
                .iter()
                .map(|(bound, chunks, bytes)| {
                    serde_json::json!({
                        "under": if *bound == u64::MAX { "inf".to_string() } else { bound.to_string() },
                        "chunks": chunks,
                        "mib": *bytes as f64 / (1u64 << 20) as f64,
                    })
                })
                .collect::<Vec<_>>(),
        },
        "plan_steps": plan_steps,
        "ops": stats.ops,
        "collectives": stats.collectives,
        "collective_sent_bytes": stats.collective_sent_bytes,
        "collective_recv_bytes": stats.collective_recv_bytes,
        "collectives_by_kind": collectives_by_kind,
        // Device bytes or a host round trip: a staged shape-changing collective runs about an
        // order of magnitude slower than a direct one, so the split is the first thing to read
        // when a collective shows up in the step trace.
        "collective_paths": {
            "staged": stats.staged_collectives,
            "direct": stats.direct_collectives,
        },
        // Where the forward's wall clock went: the sum of the collective backends' own time, the
        // same per intrinsic kind, the first distributing collective alone, and the plugin bodies.
        // On a warmed NCCL world the first collective is where the group's *arrival skew* lands:
        // the rank that reaches it first waits for the others, so a per-rank `wall_seconds`
        // difference between ranks that end together is this number's difference, not compute.
        // Each is the rank's own time; a difference between ranks in one of these *is* the
        // asymmetry, which is what they exist to localize.
        "collective_seconds": stats.collective_nanos as f64 / 1e9,
        "collective_seconds_by_kind": stats
            .collective_nanos_by_kind
            .iter()
            .map(|(kind, nanos)| (kind.clone(), *nanos as f64 / 1e9))
            .collect::<std::collections::BTreeMap<_, _>>(),
        "first_collective_seconds": stats.first_collective_nanos as f64 / 1e9,
        "op_seconds": stats.op_nanos as f64 / 1e9,
        "peak_bytes": peak_bytes,
        "wall_seconds": wall.as_secs_f64(),
        // Unix seconds at three phase boundaries. Same host, separate processes: comparing them
        // across ranks shows the arrival skew, which is what a per-rank wall difference really
        // measures when the first collective is a group rendezvous.
        "load_finished_unix": load_finished_unix,
        // The time the main thread spent waiting for the group warm-up to finish before starting
        // the forward's clock. Zero when there is no warm-up to wait for (world size 1, the CPU
        // thread transport, or a warm that finished during the load, which is the normal case).
        "warm_seconds": warm_seconds,
        "warm_finished_unix": warm_finished_unix,
        "forward_started_unix": forward_started_unix,
        "forward_finished_unix": forward_finished_unix,
        "window": window,
        // Rank 0 only; the other ranks carry the same structure with nothing.
        "logits": if rank == 0 {
            serde_json::json!({ "rows": tokens.len(), "vocab": vocab, "values": logits })
        } else {
            serde_json::Value::Null
        },
        "hidden_summaries": summaries,
        "hidden_names": hidden_names,
        // Row-major, `[hidden states, probe rows, per-row width]`; the name of each
        // hidden state is `hidden_names[i]` in the same order.
        "hidden_values": hidden_values,
        "hidden_rows": tokens.len(),
        "hidden_cols": hidden_cols,
        "digest": digest,
    }))
}

/// Executes one forward over the mesh `cfg` describes, N threads in this
/// process, and returns rank 0's view of the run plus every rank's metrics.
fn execute_mesh(
    model_dir: &Path,
    checkpoint: &Path,
    tokens: &[i64],
    cfg: &ParallelConfig,
    plugins: &[PathBuf],
    recipe_path: Option<&Path>,
    device: DeviceSpec,
) -> Result<MeshResult> {
    let mesh = Mesh::from_config(cfg);
    let world = mesh.world_size();

    let name = rustrain_model::Model::load(model_dir)
        .with_context(|| format!("loading the model description in {}", model_dir.display()))?
        .desc
        .name
        .clone();

    let rank_results: Vec<Result<serde_json::Value>> = if world == 1 {
        vec![run_rank(
            model_dir,
            checkpoint,
            tokens,
            &mesh,
            0,
            &Transport::Single,
            plugins,
            recipe_path,
            device,
        )]
    } else {
        let shared = ThreadShared::new(world);
        let (tx, rx) = mpsc::channel::<(usize, Result<serde_json::Value>)>();
        let mut handles = Vec::with_capacity(world - 1);
        for rank in 1..world {
            let tx = tx.clone();
            let shared = shared.clone();
            let mesh = mesh.clone();
            let model_dir = model_dir.to_path_buf();
            let checkpoint = checkpoint.to_path_buf();
            let tokens = tokens.to_vec();
            let plugins = plugins.to_vec();
            let recipe_path = recipe_path.map(Path::to_path_buf);
            handles.push(std::thread::spawn(move || {
                let result = run_rank(
                    &model_dir,
                    &checkpoint,
                    &tokens,
                    &mesh,
                    rank,
                    &Transport::Threads(&shared),
                    &plugins,
                    recipe_path.as_deref(),
                    device,
                );
                // Any failure must wake every rank blocked at a rendezvous —
                // a silently hung world is worse than a reported one.
                if let Err(error) = &result {
                    shared.poison(&format!("rank {rank} failed: {error:#}"));
                }
                tx.send((rank, result)).expect("the result channel is open");
            }));
        }
        drop(tx);

        let rank0 = {
            let result = run_rank(
                model_dir,
                checkpoint,
                tokens,
                &mesh,
                0,
                &Transport::Threads(&shared),
                plugins,
                recipe_path,
                device,
            );
            if let Err(error) = &result {
                shared.poison(&format!("rank 0 failed: {error:#}"));
            }
            result
        };

        for handle in handles {
            // The rendezvous cannot hang: a failing rank poisons the world and
            // every waiter finishes with that error.
            let _ = handle.join();
        }
        let mut workers: Vec<Result<serde_json::Value>> =
            rx.iter().map(|(_, result)| result).collect();
        workers.sort_by_key(|r| match r {
            Ok(value) => value["rank"].as_u64().unwrap_or(0) as usize,
            Err(_) => usize::MAX,
        });

        let mut results = Vec::with_capacity(world);
        results.push(rank0);
        results.extend(workers);
        results
    };

    assemble_ranks(name, world, rank_results)
}

/// Assembles the world's result from every rank's metrics report.
///
/// The ranks are `serde_json::Value`s either way — an in-process world passes
/// what its threads returned, a multi-process world what its children wrote —
/// so the validation, the rank table and the dump are one code path for both.
pub(crate) fn assemble_ranks(
    name: String,
    world: usize,
    results: Vec<Result<serde_json::Value>>,
) -> Result<MeshResult> {
    // Collect: every rank must have produced metrics and rank 0 the logits;
    // any rank error fails the whole run, with the worker errors named.
    let mut errors: Vec<String> = Vec::new();
    let mut ranks = Vec::with_capacity(world);
    for result in results {
        match result {
            Ok(value) => ranks.push(value),
            Err(error) => errors.push(format!("{error:#}")),
        }
    }
    if !errors.is_empty() {
        bail!("the multi-rank forward failed: {}", errors.join("; "));
    }
    ranks.sort_by_key(|value| value["rank"].as_u64().unwrap_or(0) as usize);
    if ranks.len() != world {
        bail!("expected {world} rank result(s), got {}", ranks.len());
    }

    let rank0 = &ranks[0];
    let logits: Vec<f32> = rank0["logits"]["values"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("rank 0's metrics carry no logits"))?
        .iter()
        .map(|v| v.as_f64().map(|f| f as f32).unwrap_or(f32::NAN))
        .collect();
    let summaries: Vec<[f32; 3]> = rank0["hidden_summaries"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .map(|row| {
                    let r: Vec<f32> = row
                        .as_array()
                        .map(|vals| {
                            vals.iter()
                                .map(|v| v.as_f64().map(|f| f as f32).unwrap_or(f32::NAN))
                                .collect()
                        })
                        .unwrap_or_default();
                    [r[0], r[1], r[2]]
                })
                .collect()
        })
        .unwrap_or_default();
    let hidden_names: Vec<String> = rank0["hidden_names"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // The window comes from rank 0's metrics: every rank validated the probe
    // against the global window before running.
    let window = rank0["window"].as_i64().unwrap_or(0);
    if window < 1 {
        bail!("rank 0's metrics carry no window; the forward cannot be reported");
    }

    Ok(MeshResult {
        name,
        window,
        vocab: rank0["logits"]["vocab"].as_u64().unwrap_or(0) as usize,
        world,
        logits,
        summaries,
        hidden_names,
        hidden_values: rank0["hidden_values"]
            .as_array()
            .map(|values| {
                values
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .unwrap_or_default(),
        hidden_rows: rank0["hidden_rows"].as_u64().unwrap_or(0) as usize,
        hidden_cols: rank0["hidden_cols"].as_u64().unwrap_or(0) as usize,
        digest: rank0["digest"].as_str().unwrap_or_default().to_string(),
        peak_bytes: rank0["peak_bytes"].as_u64().unwrap_or(0),
        wall: Duration::from_secs_f64(rank0["wall_seconds"].as_f64().unwrap_or(0.0)),
        loaded_count: rank0["weight_slots"].as_u64().unwrap_or(0) as usize,
        checkpoint_bytes: rank0["checkpoint_bytes_read"].as_u64().unwrap_or(0),
        checkpoint_load: rank0["checkpoint_load"].clone(),
        rank0_steps: rank0["plan_steps"].as_u64().unwrap_or(0) as usize,
        rank0_ops: rank0["ops"].as_u64().unwrap_or(0) as usize,
        rank0_collectives: rank0["collectives"].as_u64().unwrap_or(0) as usize,
        rank0_collective_seconds: rank0["collective_seconds"].as_f64().unwrap_or(0.0),
        rank0_first_collective_seconds: rank0["first_collective_seconds"].as_f64().unwrap_or(0.0),
        rank0_op_seconds: rank0["op_seconds"].as_f64().unwrap_or(0.0),
        ranks,
    })
}

fn collective_breakdown(mesh: &Mesh, stats: &rustrain_runtime::RunStats) -> Vec<serde_json::Value> {
    let mut by_key: std::collections::BTreeMap<(String, String), (usize, u64, u64)> =
        std::collections::BTreeMap::new();
    for record in &stats.collective_records {
        let kind = record
            .kind
            .strip_prefix(rustrain_plan::intrinsic::PREFIX)
            .unwrap_or(&record.kind)
            .to_string();
        let group = GroupMask::from_bits(record.group);
        let group_name = mesh.group_name(group).unwrap_or_else(|_| group.to_string());
        let entry = by_key.entry((kind, group_name)).or_default();
        entry.0 += 1;
        entry.1 += record.sent_bytes;
        entry.2 += record.recv_bytes;
    }
    by_key
        .into_iter()
        .map(|((kind, group), (calls, sent, recv))| {
            serde_json::json!({
                "kind": kind,
                "group": group,
                "calls": calls,
                "sent_bytes": sent,
                "recv_bytes": recv,
            })
        })
        .collect()
}

/// The rows of the padded probe this rank feeds: the whole probe for a
/// replicated input slot, the rank's slice for one sharded along dim 0.
/// Anything else is a distribution `run` cannot feed — reported, not guessed.
fn rank_input_rows(
    compiled: &rustrain_plan::CompiledPlan,
    input_id: SlotId,
    mesh: &Mesh,
    rank: usize,
    padded: &[i64],
) -> Result<Vec<i64>> {
    let slot = compiled.plan.slot(input_id);
    let layout = &slot.layout;
    if layout.is_replicated() {
        return Ok(padded.to_vec());
    }
    match layout.dims.as_slice() {
        [spec] if spec.dim == 0 => {
            let local = slot.shape[0];
            if local <= 0 || padded.len() as i64 % local != 0 {
                bail!(
                    "input slot `{}` is sharded into {} row(s) per rank along dim 0, which does \
                     not tile the {} padded position(s)",
                    slot.name,
                    local,
                    padded.len()
                );
            }
            let index = mesh
                .group_index(spec.group, rank)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let start = index * local as usize;
            let end = start + local as usize;
            if end > padded.len() {
                bail!(
                    "input slot `{}`: rank {rank} would feed rows {start}..{end} of a {} row \
                     probe",
                    slot.name,
                    padded.len()
                );
            }
            Ok(padded[start..end].to_vec())
        }
        _ => bail!(
            "input slot `{}` holds layout {}, which `run` cannot feed: it understands a \
             replicated token stream or one sharded along dim 0",
            slot.name,
            layout
        ),
    }
}

/// The probe tokens: `--tokens` verbatim, `--seq` as `0..n-1`, and the two must agree when both
/// are given.
pub(crate) fn probe_tokens(tokens: Option<&str>, seq: Option<usize>) -> Result<Vec<i64>> {
    let parsed = tokens.map(parse_tokens).transpose()?;
    let generated = seq.map(|n| {
        if n == 0 {
            bail!("`--seq 0` is an empty forward; a probe needs at least one token")
        }
        Ok((0..n as i64).collect::<Vec<i64>>())
    });
    let generated: Option<Vec<i64>> = generated.transpose()?;

    match (parsed, generated) {
        (Some(list), Some(generated)) => {
            if list.len() != generated.len() {
                bail!(
                    "`--tokens` has {} id(s) but `--seq` declares {}; the dump's input_ids must \
                     be one sequence — this fails early instead of writing a dump the comparison \
                     would reject",
                    list.len(),
                    generated.len()
                );
            }
            Ok(list)
        }
        (Some(list), None) => Ok(list),
        (None, Some(generated)) => Ok(generated),
        (None, None) => bail!("one of `--tokens <list>` or `--seq <n>` is required"),
    }
}

/// `9707,11,1879,0,323,358,314,279` → the eight probe ids; anything else is an error naming the
/// entry.
fn parse_tokens(text: &str) -> Result<Vec<i64>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        bail!("`--tokens` is empty; expected comma-separated non-negative integers");
    }
    trimmed
        .split(',')
        .map(|entry| {
            let entry = entry.trim();
            if entry.is_empty() {
                bail!("`--tokens` `{trimmed}` has an empty entry");
            }
            let value: i64 = entry
                .parse()
                .with_context(|| format!("`--tokens` entry `{entry}` is not an integer"))?;
            if value < 0 {
                bail!("`--tokens` entry `{entry}` is negative; token ids are non-negative");
            }
            Ok(value)
        })
        .collect()
}

/// Every float slot becomes f32 — the frozen precision decision, applied to the plan so the
/// f32-only reference provider can resolve every node.
fn widen_to_f32(plan: &mut Plan) {
    for slot in &mut plan.slots {
        if slot.dtype.is_float() {
            slot.dtype = RsDtype::F32;
        }
    }
}

/// The one slot a `outputs` pattern names — a family match is a declaration error.
fn single_match(plan: &Plan, pattern: &str, what: &str) -> Result<SlotId> {
    let ids = matching_slots(plan, pattern);
    match ids.as_slice() {
        [id] => Ok(*id),
        [] => bail!("`{what}` pattern `{pattern}` matches no slot of the rank-0 plan"),
        many => bail!(
            "`{what}` pattern `{pattern}` matches {} slot(s) ({}); the logits declaration must \
             name exactly one",
            many.len(),
            many.iter()
                .map(|id| plan.slot(*id).name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The plan slots a pattern names, in plan order.
fn matching_slots(plan: &Plan, pattern: &str) -> Vec<SlotId> {
    plan.slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| rustrain_model::matches(pattern, &slot.name))
        .map(|(index, _)| SlotId(index))
        .collect()
}

/// Extends the plan so every slot a hidden pattern names stays readable after the run: one
/// `view` node at the end per slot (a view aliases its input, so the extension costs a
/// descriptor, not a copy), with a uniquely named output slot the pool keeps alive.
///
/// Returns, in pattern order, **the keeper's output slot** paired with the display name of the
/// hidden state it keeps.
///
/// The output — not the matched input — is what a caller must read back. The compiler rewires a
/// consumer to the completed twin whenever a slot carries a partial (an embedding sharded over tp,
/// say), the keeper included; the original slot is then dead after the collective, and the pool is
/// free to hand its bytes to a later activation. Reading the original would return whatever was
/// written over it — bytes with the right shape, the right name and the wrong values, which is
/// exactly what the first D6 device run produced. The keeper's output dies with the plan, so its
/// bytes are still its own at the end.
fn keep_hidden_states(plan: &mut Plan, patterns: &[String]) -> Result<Vec<(SlotId, String)>> {
    let mut ids: Vec<SlotId> = Vec::new();
    for pattern in patterns {
        let matched = matching_slots(plan, pattern);
        if matched.is_empty() {
            bail!("`outputs.hidden` pattern `{pattern}` matches no slot of the rank-0 plan");
        }
        ids.extend(matched);
    }
    let phase = plan.meta.phase;
    let mut kept: Vec<(SlotId, String)> = Vec::with_capacity(ids.len());
    for (index, id) in ids.iter().enumerate() {
        let slot = plan.slot(*id);
        let mut out = slot.clone();
        out.name = format!("__run__.hidden.{index}.out");
        // The hidden slot may carry a partial at this point (the all-reduce that completes it is
        // inserted by the compiler, which rewires consumers to the converted slot). The view's
        // output is what the dump reads — the complete tensor — so it declares the completed
        // layout: replicate where the input is partial, the input's own layout otherwise.
        if slot.layout.partial.is_some() {
            out.layout = ParallelLayout::replicate();
        }
        if plan.slot_id(&out.name).is_some() {
            bail!(
                "the hidden-state keeper collides with existing slot `{}`",
                out.name
            );
        }
        out.kind = rustrain_plan::SlotKind::Output;
        let out_id = SlotId(plan.slots.len());
        kept.push((out_id, slot.name.clone()));
        plan.slots.push(out);
        plan.nodes.push(rustrain_plan::PlanNode {
            op: rustrain_plan::OpRef::new("view"),
            inputs: vec![*id],
            outputs: vec![out_id],
            attrs: rustrain_plan::Attrs::new(),
            phase,
            precision: rustrain_plan::PrecisionOverride::default(),
            checkpoint: rustrain_plan::CheckpointPolicy::None,
            stream: rustrain_plan::StreamPolicy::Default,
            source: rustrain_plan::Trace::new(format!("run.hidden.{index}")),
        });
    }
    Ok(kept)
}

/// When the logits slot is not replicated, appends one `view` node whose
/// output **declares** `replicate` (the same shape as the input — the view
/// itself cannot grow a tensor). The compiler then inserts the completing
/// collective after the view: shard-propagation sees the declared-replicate
/// output disagree with the shard the view actually produces, splices the
/// collective, and gives the *converted* twin the gathered global shape (D6's
/// shape math in `rustrain_plan::shard::propagate`). The dump reads the twin,
/// resolved by name below. Returns the view's output slot id when one was
/// added.
fn complete_logits(plan: &mut Plan, pattern: &str) -> Result<Option<SlotId>> {
    let id = single_match(plan, pattern, "outputs.logits")?;
    if plan.slot(id).layout.is_replicated() {
        return Ok(None);
    }
    let slot = plan.slot(id);
    let mut out = slot.clone();
    out.name = LOGITS_COMPLETE.to_string();
    out.layout = ParallelLayout::replicate();
    out.kind = rustrain_plan::SlotKind::Output;
    if plan.slot_id(&out.name).is_some() {
        bail!("the logits completion view collides with existing slot `{LOGITS_COMPLETE}`");
    }
    let out_id = SlotId(plan.slots.len());
    plan.slots.push(out);
    plan.nodes.push(rustrain_plan::PlanNode {
        op: rustrain_plan::OpRef::new("view"),
        inputs: vec![id],
        outputs: vec![out_id],
        attrs: rustrain_plan::Attrs::new(),
        phase: plan.meta.phase,
        precision: rustrain_plan::PrecisionOverride::default(),
        checkpoint: rustrain_plan::CheckpointPolicy::None,
        stream: rustrain_plan::StreamPolicy::Default,
        source: rustrain_plan::Trace::new("run.logits.complete"),
    });
    Ok(Some(out_id))
}

/// The slot holding the complete logits after compilation. For a plain
/// (already-replicated) plan: the pattern's single match. When the runner
/// added a completion view: the converted twin of its output — the slot named
/// `<LOGITS_COMPLETE>__<intrinsic>...` whose layout is replicate and whose
/// shape is the global one. Exactly one such slot must exist; two would mean
/// two different completions, and none means the compiler did not insert the
/// completing collective.
fn resolve_completed_logits(plan: &Plan, completed: bool, pattern: &str) -> Result<SlotId> {
    if !completed {
        let id = single_match(plan, pattern, "outputs.logits")?;
        if !plan.slot(id).layout.is_replicated() {
            bail!(
                "the logits slot `{}` compiled to {}, not replicate; the runner cannot dump \
                 a distributed tensor",
                plan.slot(id).name,
                plan.slot(id).layout
            );
        }
        return Ok(id);
    }
    let prefix = format!("{LOGITS_COMPLETE}__");
    let twins: Vec<SlotId> = plan
        .slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.name.starts_with(&prefix) && slot.layout.is_replicated())
        .map(|(index, _)| SlotId(index))
        .collect();
    match twins.as_slice() {
        [id] => Ok(*id),
        [] => bail!(
            "the logits completion view compiled to no replicate twin; the completing \
             collective was not inserted"
        ),
        many => bail!(
            "the logits completion view compiled to {} replicate twin(s): {}",
            many.len(),
            many.iter()
                .map(|id| plan.slot(*id).name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

// ---- the configuration sweep -----------------------------------------------

/// `--sweep "tp=2;tp=4;tp=2,ep=2"` → one `ParallelConfig` per `;`-separated entry.
pub(crate) fn parse_sweep(list: &str) -> Result<Vec<ParallelConfig>> {
    let mut configs = Vec::new();
    for (index, entry) in list.split(';').enumerate() {
        let entry = entry.trim();
        if entry.is_empty() {
            bail!("`--sweep` entry {} is empty", index + 1);
        }
        let mut cfg = ParallelConfig::default();
        let mut seen = std::collections::BTreeSet::new();
        for pair in entry.split(',') {
            let pair = pair.trim();
            let Some((axis, degree)) = pair.split_once('=') else {
                bail!("`--sweep` entry `{entry}`: `{pair}` is not `axis=degree`");
            };
            let degree: usize = degree.trim().parse().with_context(|| {
                format!("`--sweep` entry `{entry}`: `{degree}` is not an integer")
            })?;
            if degree == 0 {
                bail!("`--sweep` entry `{entry}`: `{axis}` degree must be at least 1");
            }
            if !seen.insert(axis.trim().to_string()) {
                bail!("`--sweep` entry `{entry}` declares axis `{axis}` twice");
            }
            match axis.trim() {
                "tp" => cfg.tensor = degree,
                "cp" => cfg.context = degree,
                "ep" => cfg.expert = degree,
                "dp" => cfg.data = degree,
                "pp" => cfg.pipeline = degree,
                other => bail!("`--sweep` entry `{entry}`: `{other}` is not one of tp/cp/ep/dp/pp"),
            }
        }
        configs.push(cfg);
    }
    if configs.is_empty() {
        bail!("`--sweep` lists no configurations");
    }
    Ok(configs)
}

fn max_abs(values: &[f32]) -> f64 {
    values
        .iter()
        .fold(0.0f64, |acc, v| acc.max((*v as f64).abs()))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0f64, f64::max)
}

/// Runs the same forward over every listed mesh and writes ONE JSON report to
/// `--out`. A world-1 baseline always runs first; every configuration's rank-0
/// logits are compared against it with the D6 relative bound.
fn run_sweep(args: &RunArgs, tokens: &[i64], list: &str, device: DeviceSpec) -> Result<()> {
    let out = args
        .out
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--sweep writes one JSON report: pass --out <PATH>"))?;
    let configs = parse_sweep(list)?;

    // The same CUDA guard as the single run, checked for every listed mesh
    // before anything executes.
    for cfg in &configs {
        let world = cfg
            .tensor
            .saturating_mul(cfg.expert)
            .saturating_mul(cfg.context)
            .saturating_mul(cfg.data)
            .saturating_mul(cfg.pipeline);
        if device.is_cuda() && world > 1 {
            bail!(
                "`--device cuda` with the {} config (world size {world}): a CUDA context can only \
                 be current on one host thread at a time, so one allocator/context serves one \
                 execution thread; a multi-rank CUDA launch needs one process per rank, which is \
                 the next step of D6",
                mesh_text(cfg)
            );
        }
    }

    let baseline_cfg = ParallelConfig::default();
    let baseline = execute_mesh(
        &args.model,
        &args.checkpoint,
        tokens,
        &baseline_cfg,
        &args.plugins,
        args.recipe.as_deref(),
        device,
    )
    .context("running the world-1 baseline")?;
    if baseline.logits.is_empty() {
        bail!("the world-1 baseline produced no logits");
    }

    let mut runs = Vec::with_capacity(configs.len());
    for cfg in &configs {
        let result = execute_mesh(
            &args.model,
            &args.checkpoint,
            tokens,
            cfg,
            &args.plugins,
            args.recipe.as_deref(),
            device,
        )
        .with_context(|| format!("executing the sweep config {}", mesh_text(cfg)))?;
        runs.push((*cfg, result));
    }
    sweep_report(
        out,
        &args.model,
        &args.checkpoint,
        tokens,
        &baseline_cfg,
        &baseline,
        &runs,
    )
}

/// Turns a baseline plus one result per configuration into the sweep report —
/// one JSON on disk and one line per configuration on stdout.
///
/// Shared by the in-process sweep and by `launch`, so a multi-process sweep and
/// a threaded one cannot drift into two different report formats: the numbers
/// differ, the shape does not.
pub(crate) fn sweep_report(
    out: &Path,
    model: &Path,
    checkpoint: &Path,
    tokens: &[i64],
    baseline_cfg: &ParallelConfig,
    baseline: &MeshResult,
    runs: &[(ParallelConfig, MeshResult)],
) -> Result<()> {
    let baseline_max = max_abs(&baseline.logits);
    if baseline.logits.is_empty() {
        bail!("the world-1 baseline produced no logits");
    }
    let mut entries = Vec::with_capacity(runs.len());
    for (cfg, result) in runs {
        if result.logits.len() != baseline.logits.len() {
            bail!(
                "the {} config produced {} logits, the baseline {}: the meshes are not the same \
                 forward",
                mesh_text(cfg),
                result.logits.len(),
                baseline.logits.len()
            );
        }
        let diff = max_abs_diff(&baseline.logits, &result.logits);
        let bound = AGREEMENT_BOUND_RELATIVE * baseline_max;
        entries.push(serde_json::json!({
            "degrees": {
                "tp": cfg.tensor, "cp": cfg.context, "ep": cfg.expert,
                "dp": cfg.data, "pp": cfg.pipeline,
            },
            "world_size": result.world,
            "digest": result.digest,
            "wall_seconds": result.wall.as_secs_f64(),
            "max_abs_diff": diff,
            "bound": bound,
            "pass": diff <= bound,
            "ranks": result.ranks,
        }));
    }

    let report = serde_json::json!({
        "format": SWEEP_FORMAT,
        "model": model.display().to_string(),
        "checkpoint": checkpoint.display().to_string(),
        "probe_tokens": tokens,
        "bound_relative": AGREEMENT_BOUND_RELATIVE,
        "baseline": {
            "degrees": {
                "tp": baseline_cfg.tensor, "cp": baseline_cfg.context, "ep": baseline_cfg.expert,
                "dp": baseline_cfg.data, "pp": baseline_cfg.pipeline,
            },
            "world_size": baseline.world,
            "digest": baseline.digest,
            "wall_seconds": baseline.wall.as_secs_f64(),
            "max_abs": baseline_max,
            "ranks": baseline.ranks,
        },
        "configs": entries,
    });
    std::fs::write(out, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("writing the sweep report {}", out.display()))?;

    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    println!(
        "sweep {} over {} config(s)  baseline world 1 (max |logits| {:.3e})",
        model.display(),
        runs.len(),
        baseline_max
    );
    for ((cfg, _), entry) in runs.iter().zip(&entries) {
        let diff = entry["max_abs_diff"].as_f64().unwrap_or(f64::NAN);
        let bound = entry["bound"].as_f64().unwrap_or(f64::NAN);
        let pass = entry["pass"].as_bool().unwrap_or(false);
        let weight_bytes: u64 = entry["ranks"]
            .as_array()
            .map(|ranks| {
                ranks
                    .iter()
                    .map(|r| r["weight_bytes"].as_u64().unwrap_or(0))
                    .sum()
            })
            .unwrap_or(0);
        println!(
            "  {}  world {:>2}  wall {:>8.3} s  weights {:>7.2} GiB  logits vs world-1: \
             max|diff| {:.3e} / bound {:.3e} {}",
            mesh_text(cfg),
            entry["world_size"],
            entry["wall_seconds"].as_f64().unwrap_or(0.0),
            gib(weight_bytes),
            diff,
            bound,
            if pass { "PASS" } else { "FAIL" }
        );
    }
    println!("  wrote {}", out.display());
    Ok(())
}

/// The compact human per-rank table for a multi-rank run.
fn print_rank_table(result: &MeshResult, gib: &impl Fn(u64) -> f64) {
    println!(
        "    {:<5} {:>12} {:>8} {:>6} {:>6} {:>10} {:>12}",
        "rank", "weights", "steps", "ops", "colls", "coll bytes", "wall s"
    );
    for rank in &result.ranks {
        println!(
            "    {:<5} {:>9.2} GiB {:>8} {:>6} {:>6} {:>10} {:>12.3}",
            rank["rank"].as_u64().unwrap_or(0),
            gib(rank["weight_bytes"].as_u64().unwrap_or(0)),
            rank["plan_steps"].as_u64().unwrap_or(0),
            rank["ops"].as_u64().unwrap_or(0),
            rank["collectives"].as_u64().unwrap_or(0),
            rank["collective_sent_bytes"].as_u64().unwrap_or(0)
                + rank["collective_recv_bytes"].as_u64().unwrap_or(0),
            rank["wall_seconds"].as_f64().unwrap_or(0.0),
        );
    }
    let mut printed = false;
    for rank in &result.ranks {
        for entry in rank["collectives_by_kind"].as_array().into_iter().flatten() {
            printed = true;
            println!(
                "      rank {}  {:<12} group {:<6} {} call(s)  sent {}  recv {}",
                rank["rank"].as_u64().unwrap_or(0),
                entry["kind"].as_str().unwrap_or("?"),
                entry["group"].as_str().unwrap_or("?"),
                entry["calls"].as_u64().unwrap_or(0),
                entry["sent_bytes"].as_u64().unwrap_or(0),
                entry["recv_bytes"].as_u64().unwrap_or(0),
            );
        }
    }
    if !printed {
        println!("      (no collectives)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// The real description goes through the whole runner-side plan surgery and **compiles**
    /// against the reference provider: expand → instantiate → widen → keep the hidden states →
    /// compile. The forward itself cannot run on this box (no weights) — that is the GPU run —
    /// but every step the runner performs on the plan before any weight byte is read is pinned
    /// here, so nothing between the CLI and the executor is untested.
    #[test]
    fn the_real_plan_compiles_after_the_runner_surgery() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../rustrain-model/tests/fixtures/qwen36-text");
        let model = rustrain_model::Model::load(&dir).expect("load");
        let expanded = model.expand().expect("expand");
        let mesh = Mesh::from_config(&ParallelConfig::default());
        let registry = crate::load_registry(&[]).expect("the built-in registry");
        let mut plan = rustrain_plan::instantiate(
            &expanded.plan,
            &expanded.declarations(),
            &mesh,
            0,
            &registry,
        )
        .expect("instantiate rank 0");
        widen_to_f32(&mut plan);

        let outputs = model.desc.outputs.as_ref().expect("outputs declared");
        let logits_before = matching_slots(&plan, &outputs.logits);
        assert_eq!(logits_before.len(), 1, "the logits pattern names one slot");
        let hidden = keep_hidden_states(&mut plan, &outputs.hidden).expect("keeper");
        // HF's output_hidden_states = embedding output + 40 layer outputs + final norm = 42 rows.
        assert_eq!(hidden.len(), 42, "embed + layers.0..39 + norm");
        assert_eq!(
            hidden[0].1, "embed.y",
            "the first hidden state is the embedding output"
        );
        assert_eq!(
            hidden[41].1, "norm.y",
            "the last hidden state is the final norm"
        );
        // The ids the runner reads back are the keeper *outputs*, not the slots they were built
        // from: the compiler rewires consumers to a completed twin (the embedding is sharded over
        // tp and carries a partial), and the original slot is dead after that collective — the
        // pool may hand its bytes to a later activation. Reading the original returned the right
        // shape and the wrong numbers on the first D6 device run.
        for (id, name) in &hidden {
            let keeper = plan.slot(*id);
            assert!(
                keeper.name.starts_with("__run__.hidden."),
                "hidden state `{name}` is read from `{}`, which is not the keeper's output",
                keeper.name
            );
            assert_eq!(keeper.kind, rustrain_plan::SlotKind::Output);
        }
        let produced_by = plan
            .nodes
            .iter()
            .find(|n| n.outputs.contains(&hidden[0].0))
            .expect("the keeper output has a producer");
        assert_eq!(
            produced_by.op.name, "view",
            "the keeper is a view node, so it costs a descriptor and not a copy"
        );
        assert_eq!(
            produced_by.inputs.len(),
            1,
            "the keeper reads the one hidden slot it keeps"
        );

        let nodes_after = plan.nodes.len();
        let registry = crate::load_registry(&[]).expect("the built-in registry");
        let recipe = crate::load_recipe(None).expect("recipe");
        let compiled =
            rustrain_plan::Compiler::new(&registry, &recipe, rustrain_ops::TargetEnv::default())
                .compile(&plan)
                .unwrap_or_else(|e| panic!("the real plan must compile after the surgery: {e}"));
        assert_eq!(
            compiled.steps.len(),
            nodes_after + compiled.inserted.len(),
            "every node and every inserted collective is a step"
        );
        // The declared tp/ep sharding on degree-1 axes still reconciles through identity
        // collectives at world size 1. The count moved 53 -> 42 when the operators started
        // declaring their shard rules (ABI v2: the model-specific operators are `pass_through`
        // instead of "unknown, therefore no derivation", so eleven layouts now agree with their
        // producers by construction), and 42 -> 41 when the embedding table stopped being
        // vocab-sharded: a sharded lookup needs its rows offset by rank, that position constant is
        // not implemented, and replicating the table is exact at every degree. The exact count is
        // pinned by the check gate too.
        assert_eq!(compiled.inserted.len(), 41);
    }

    /// The same surgery at `tp = 2`: this is the regression for the ABI v2 rule
    /// declarations. Before them the framework classified operators by name, the
    /// seven model-specific operators were "unknown, therefore no derivation",
    /// their outputs stayed replicated while their inputs were split, and
    /// compilation refused the plan — so `tp > 1` was never executable on this
    /// model. Now every operator declares its rule and the plan compiles.
    #[test]
    fn the_real_plan_compiles_after_the_runner_surgery_at_tp_two() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../rustrain-model/tests/fixtures/qwen36-text");
        let model = rustrain_model::Model::load(&dir).expect("load");
        let expanded = model.expand().expect("expand");
        let mesh = Mesh::from_config(&ParallelConfig {
            tensor: 2,
            ..Default::default()
        });
        let registry = crate::load_registry(&[]).expect("the built-in registry");
        let mut plan = rustrain_plan::instantiate(
            &expanded.plan,
            &expanded.declarations(),
            &mesh,
            0,
            &registry,
        )
        .expect("rank 0 instantiates at tp=2");
        widen_to_f32(&mut plan);
        let outputs = model.desc.outputs.as_ref().expect("outputs declared");
        keep_hidden_states(&mut plan, &outputs.hidden).expect("keeper");
        complete_logits(&mut plan, &outputs.logits).expect("logits completion");

        let recipe = crate::load_recipe(None).expect("recipe");
        let compiled =
            rustrain_plan::Compiler::new(&registry, &recipe, rustrain_ops::TargetEnv::default())
                .compile(&plan)
                .unwrap_or_else(|e| panic!("the real plan must compile at tp=2: {e}"));

        // The split really reaches the activations: the local head count halves
        // and the vocabulary is cut in two.
        let qr = compiled
            .plan
            .slot_id("layers.0.q")
            .expect("the layer-0 q activation is in the plan");
        assert_eq!(
            compiled.plan.slot(qr).shape,
            vec![512, 1024],
            "a tp=2 q projection holds half the model's 2048 channels"
        );
        assert!(
            compiled
                .inserted
                .iter()
                .any(|c| c.op == "intrinsic.all_reduce"),
            "the row-parallel halves owe an all-reduce: {:?}",
            compiled
                .inserted
                .iter()
                .map(|c| (c.op, c.group))
                .collect::<Vec<_>>()
        );
    }

    /// The comparison script's fixed probe must parse to exactly its eight ids.
    #[test]
    fn the_fixed_probe_parses_to_its_eight_ids() {
        assert_eq!(
            parse_tokens("9707,11,1879,0,323,358,314,279").unwrap(),
            vec![9707, 11, 1879, 0, 323, 358, 314, 279]
        );
        // Whitespace is tolerated around entries.
        assert_eq!(
            parse_tokens(" 9707, 11 ,1879").unwrap(),
            vec![9707, 11, 1879]
        );
    }

    #[test]
    fn token_parsing_rejects_garbage_by_name() {
        for bad in [
            "",
            "  ",
            "1,,2",
            "1,2,x",
            "-3,4",
            "1.5",
            "99999999999999999999999",
        ] {
            let error = parse_tokens(bad).unwrap_err();
            assert!(!error.to_string().is_empty(), "`{bad}` must be an error");
        }
        assert!(parse_tokens("1,2,x").unwrap_err().to_string().contains('x'));
        assert!(
            parse_tokens("-3,4")
                .unwrap_err()
                .to_string()
                .contains("negative")
        );
    }

    /// `--seq n` is `--tokens 0..n-1`; the two forms must agree when both are given.
    #[test]
    fn seq_generates_the_arange_probe_and_checks_consistency() {
        assert_eq!(
            probe_tokens(None, Some(8)).unwrap(),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(probe_tokens(Some("5,6,7"), Some(3)).unwrap(), vec![5, 6, 7]);
        let error = probe_tokens(Some("5,6,7"), Some(4)).unwrap_err();
        assert!(error.to_string().contains("--seq"), "{error}");
        assert!(probe_tokens(None, Some(0)).is_err());
        assert!(probe_tokens(None, None).is_err());
    }

    /// The per-row statistics the comparison reads: population std (torch's `.std()` default) and
    /// the max of absolute values.
    #[test]
    fn summaries_are_population_mean_std_and_abs_max() {
        let row = summarize(&[1.0, 2.0, 3.0]);
        assert_eq!(row[0], 2.0);
        let var = ((1.0f32 + 0.0 + 1.0) / 3.0).sqrt();
        assert_eq!(row[1], var);
        assert_eq!(row[2], 3.0);
        let signed = summarize(&[-4.0, 2.0]);
        assert_eq!(signed[2], 4.0, "max is over absolute values");
    }

    /// Widening touches float slots only: indices stay i64.
    #[test]
    fn widening_reaches_floats_and_leaves_indices_alone() {
        let mut b = rustrain_plan::PlanBuilder::new(
            "w",
            rustrain_ops::Phase::Forward,
            Mesh::from_config(&ParallelConfig::default()).fingerprint(),
        );
        let x = b.slot("x", RsDtype::I64, vec![4], rustrain_plan::SlotKind::Input);
        let w = b.slot(
            "w",
            RsDtype::BF16,
            vec![4, 8],
            rustrain_plan::SlotKind::Weight,
        );
        let y = b.slot(
            "y",
            RsDtype::BF16,
            vec![4, 8],
            rustrain_plan::SlotKind::Output,
        );
        b.node(
            rustrain_plan::OpRef::new("linear"),
            vec![x, w],
            vec![y],
            rustrain_plan::Attrs::new(),
            "lin",
        );
        let mut plan = b.build().unwrap();
        widen_to_f32(&mut plan);
        assert_eq!(plan.slot(plan.slot_id("x").unwrap()).dtype, RsDtype::I64);
        assert_eq!(plan.slot(plan.slot_id("w").unwrap()).dtype, RsDtype::F32);
        assert_eq!(plan.slot(plan.slot_id("y").unwrap()).dtype, RsDtype::F32);
    }

    /// The sweep list parser: strict `axis=degree` pairs, unknown axes and
    /// duplicates refused, empty entries rejected.
    #[test]
    fn sweep_list_parses_strictly() {
        let configs = parse_sweep("tp=2;tp=4,ep=2;dp=2").unwrap();
        assert_eq!(configs.len(), 3);
        assert_eq!(configs[0].tensor, 2);
        assert_eq!(configs[1].tensor, 4);
        assert_eq!(configs[1].expert, 2);
        assert_eq!(configs[2].data, 2);
        assert_eq!(configs[2].tensor, 1);
        for bad in ["", "tp=0", "tp=x", "tp", "foo=2", "tp=2,tp=4", ";tp=2"] {
            let error = parse_sweep(bad).unwrap_err();
            assert!(!error.to_string().is_empty(), "`{bad}` must be an error");
        }
    }
}
