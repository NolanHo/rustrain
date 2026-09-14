//! `rustrain launch`: one rank per process, one GPU per rank.
//!
//! `run` executes one rank and says so: a CUDA context is current on one thread,
//! so one process owns one GPU, and a world of `N` ranks is `N` processes. This
//! subcommand is the thing that starts them, hands each its rank, its device and
//! the rendezvous directory, waits, and then produces exactly the artifacts a
//! single-process run produces — one dump, one sidecar, one sweep report — from
//! the ranks' own metrics.
//!
//! # Why the world is only as big as the mesh
//!
//! `world = tp × cp × ep × dp × pp`. Nothing here pads a world to fill the
//! machine: a `tp=2` config on an 8-GPU host starts two processes and leaves six
//! cards idle, because the plan's collectives are defined over the mesh, not
//! over the host.
//!
//! # What the launcher owns
//!
//! * the **rendezvous directory** — created empty for each config (a stale
//!   unique id from an earlier run would hand a rank the wrong communicator)
//!   and removed when the launch ends unless `--keep-rdzv` asks otherwise;
//! * the **rank → device** mapping — rank `i` runs on `cuda:(base + i)`;
//! * **failure reporting** — a rank that dies takes the world down with its
//!   stderr attached, instead of leaving the others blocked in a collective.
//!
//! A rank that cannot start (no NCCL, no device, a rank that never reaches the
//! rendezvous) fails its own process and the launcher reports it; the surviving
//! ranks then fail their own rendezvous waits with a bounded timeout rather than
//! hanging, which is what the runtime's file rendezvous is for.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use clap::Args;
use rustrain_parallel::ParallelConfig;

use crate::device::DeviceSpec;
use crate::run::{self, MeshResult, RunArgs};

#[derive(Args)]
pub(crate) struct LaunchArgs {
    /// Model directory (`config.json` + `model.json`).
    #[arg(long, value_name = "DIR")]
    pub model: PathBuf,

    /// The real safetensors checkpoint (a model directory or its index).
    #[arg(long, value_name = "PATH")]
    pub checkpoint: PathBuf,

    /// The probe tokens, comma-separated.
    #[arg(long, value_name = "LIST")]
    pub tokens: Option<String>,

    /// Shorthand for `--tokens 0,1,..,<N-1>`.
    #[arg(long, value_name = "N")]
    pub seq: Option<usize>,

    /// The dtype the plan's float slots execute in, forwarded to every rank: `bf16` (the default)
    /// keeps the checkpoint's own weights, `f32` widens them.
    #[arg(long, value_enum, default_value = "bf16")]
    pub dtype: run::RunDtype,

    /// Where the world's dump lands (a `.npz` plus a `.json` sidecar); in
    /// `--sweep` mode this is the JSON report path instead.
    #[arg(long, value_name = "PATH")]
    pub out: PathBuf,

    /// Write the world's per-rank metrics report here as well.
    #[arg(long, value_name = "PATH")]
    pub metrics: Option<PathBuf>,

    /// The mesh degrees. `world = tp × cp × ep × dp × pp` processes are started,
    /// one per rank.
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

    /// Run the world once per listed config and write ONE JSON report to `--out`.
    /// Configs are `;`-separated, each a `,`-separated list of `axis=degree`. A
    /// world-1 baseline runs first and every config's rank-0 logits are compared
    /// against it.
    #[arg(long, value_name = "LIST")]
    pub sweep: Option<String>,

    /// A plugin `.so` to load in each rank.
    #[arg(long = "plugin", value_name = "PATH")]
    pub plugins: Vec<PathBuf>,

    /// Recipe file deciding which implementation runs.
    #[arg(long, value_name = "PATH")]
    pub recipe: Option<PathBuf>,

    /// `cuda` or `cuda:<base>`: rank `i` runs on device `base + i`.
    #[arg(long, value_name = "SPEC", default_value = "cuda")]
    pub device: String,

    /// The NCCL library to load in each rank (default: the loader's search).
    #[arg(long, value_name = "PATH")]
    pub nccl_lib: Option<PathBuf>,

    /// Where the ranks exchange NCCL's unique ids (default: `<out>.rdzv/`).
    #[arg(long, value_name = "DIR")]
    pub rdzv: Option<PathBuf>,

    /// Keep the rendezvous directory and each rank's metrics files around after
    /// the launch (they are the evidence when a world fails).
    #[arg(long)]
    pub keep_rdzv: bool,
}

pub(crate) fn launch(args: LaunchArgs) -> Result<()> {
    // The probe is validated once, here, so a malformed `--tokens`/`--seq` pair
    // fails before `world` processes are started with it — and so the dump the
    // launcher writes carries the same token list the ranks ran.
    let tokens = run::probe_tokens(args.tokens.as_deref(), args.seq)?;
    let base_device = match DeviceSpec::parse(&args.device)? {
        DeviceSpec::Cuda(index) => index,
        DeviceSpec::Cpu => bail!(
            "`launch` starts one process per GPU: pass `--device cuda[:base]` (rank i takes \
             device base + i). The CPU multi-rank path is N threads in one process — that is \
             `run --tp N` without `--rank`"
        ),
    };

    let configs: Vec<ParallelConfig> = match &args.sweep {
        Some(list) => run::parse_sweep(list)?,
        None => vec![ParallelConfig {
            tensor: args.tp,
            context: args.cp,
            expert: args.ep,
            data: args.dp,
            pipeline: args.pp,
        }],
    };
    if configs.iter().any(|cfg| cfg.pipeline > 1) {
        bail!(
            "`pp > 1`: the cross-stage seam is still an open decision, so a pipelined world is \
             refused rather than half-executed"
        );
    }

    // The model name belongs to the description, not to a rank's metrics report:
    // the launcher reads it once and hands it to the report assembly.
    let name = rustrain_model::Model::load(&args.model)
        .with_context(|| format!("loading the model description in {}", args.model.display()))?
        .desc
        .name
        .clone();

    let rdzv_base = args
        .rdzv
        .clone()
        .unwrap_or_else(|| args.out.with_extension("rdzv"));
    std::fs::create_dir_all(&rdzv_base)
        .with_context(|| format!("creating the rendezvous directory {}", rdzv_base.display()))?;

    let outcome = if args.sweep.is_some() {
        let baseline_cfg = ParallelConfig::default();
        let baseline = run_world(
            &name,
            &baseline_cfg,
            &args,
            &rdzv_base.join(slug(&baseline_cfg)),
            base_device,
        )
        .context("running the world-1 baseline")?;
        let mut runs = Vec::with_capacity(configs.len());
        for cfg in &configs {
            let result = run_world(&name, cfg, &args, &rdzv_base.join(slug(cfg)), base_device)
                .with_context(|| format!("running the world {}", mesh_text(cfg)))?;
            runs.push((*cfg, result));
        }
        run::sweep_report(
            &args.out,
            &args.model,
            &args.checkpoint,
            &tokens,
            &baseline_cfg,
            &baseline,
            &runs,
        )
    } else {
        let cfg = configs[0];
        let result = run_world(&name, &cfg, &args, &rdzv_base.join(slug(&cfg)), base_device)
            .with_context(|| format!("running the world {}", mesh_text(&cfg)))?;
        run::emit_result(&args.as_run_args()?, &tokens, &cfg, result)
    };

    if !args.keep_rdzv {
        // The directory holds one 128-byte file per communicator, plus each
        // rank's metrics. Removing it keeps a failed launch from poisoning the
        // next one; `--keep-rdzv` is how a stuck world gets investigated.
        let _ = std::fs::remove_dir_all(&rdzv_base);
    }
    outcome
}

impl LaunchArgs {
    /// The `run`-shaped view of these arguments, for the shared dump and sidecar
    /// writer: the launcher produces exactly what a single-process run produces.
    fn as_run_args(&self) -> Result<RunArgs> {
        Ok(RunArgs {
            model: self.model.clone(),
            checkpoint: self.checkpoint.clone(),
            tokens: self.tokens.clone(),
            seq: self.seq,
            dtype: self.dtype,
            out: Some(self.out.clone()),
            tp: self.tp,
            cp: self.cp,
            ep: self.ep,
            dp: self.dp,
            pp: self.pp,
            metrics: self.metrics.clone(),
            sweep: None,
            plugins: self.plugins.clone(),
            recipe: self.recipe.clone(),
            device: self.device.clone(),
            rank: None,
            world: None,
            rdzv: None,
            nccl_lib: None,
        })
    }
}

/// Whether the ranks were asked for a per-step trace (`RUSTRAIN_STEP_TRACE`).
fn trace_requested() -> bool {
    std::env::var_os("RUSTRAIN_STEP_TRACE").is_some()
}

fn mesh_text(cfg: &ParallelConfig) -> String {
    format!(
        "tp={}, cp={}, ep={}, dp={}, pp={}",
        cfg.tensor, cfg.context, cfg.expert, cfg.data, cfg.pipeline
    )
}

/// A filesystem-safe name for one config's rendezvous directory.
fn slug(cfg: &ParallelConfig) -> String {
    format!(
        "tp{}-cp{}-ep{}-dp{}-pp{}",
        cfg.tensor, cfg.context, cfg.expert, cfg.data, cfg.pipeline
    )
}

fn world_size(cfg: &ParallelConfig) -> usize {
    cfg.tensor
        .saturating_mul(cfg.context)
        .saturating_mul(cfg.expert)
        .saturating_mul(cfg.data)
        .saturating_mul(cfg.pipeline)
}

/// Starts one process per rank, waits for all of them, and assembles their
/// metrics into the world's result.
fn run_world(
    name: &str,
    cfg: &ParallelConfig,
    args: &LaunchArgs,
    run_dir: &Path,
    base_device: usize,
) -> Result<MeshResult> {
    let world = world_size(cfg);
    if world == 0 {
        bail!("a mesh degree of 0 is not a mesh");
    }
    // Fresh directory per config: a unique id from an earlier run would be a
    // different communicator, and reading it would hang the world instead of
    // failing it.
    let _ = std::fs::remove_dir_all(run_dir);
    std::fs::create_dir_all(run_dir).with_context(|| format!("creating {}", run_dir.display()))?;

    let executable = std::env::current_exe().context("finding this executable")?;
    println!(
        "launch {}  world {world}  device cuda:{base_device}..{}  rendezvous {}",
        mesh_text(cfg),
        base_device + world - 1,
        run_dir.display()
    );

    let mut children = Vec::with_capacity(world);
    for rank in 0..world {
        let metrics = run_dir.join(format!("rank-{rank}.json"));
        let mut command = Command::new(&executable);
        command
            .arg("run")
            .arg("--model")
            .arg(&args.model)
            .arg("--checkpoint")
            .arg(&args.checkpoint)
            .arg("--tp")
            .arg(cfg.tensor.to_string())
            .arg("--cp")
            .arg(cfg.context.to_string())
            .arg("--ep")
            .arg(cfg.expert.to_string())
            .arg("--dp")
            .arg(cfg.data.to_string())
            .arg("--pp")
            .arg(cfg.pipeline.to_string())
            .arg("--device")
            .arg(format!("cuda:{}", base_device + rank))
            .arg("--rank")
            .arg(rank.to_string())
            .arg("--world")
            .arg(world.to_string())
            .arg("--rdzv")
            .arg(run_dir)
            .arg("--metrics")
            .arg(&metrics);
        if let Some(list) = &args.tokens {
            command.arg("--tokens").arg(list);
        }
        if let Some(seq) = args.seq {
            command.arg("--seq").arg(seq.to_string());
        }
        // The dtype is one fact per world, not per rank: forward it verbatim so a rank child runs
        // the same plan its launcher reports.
        command.arg("--dtype").arg(args.dtype.to_string());
        for plugin in &args.plugins {
            command.arg("--plugin").arg(plugin);
        }
        if let Some(recipe) = &args.recipe {
            command.arg("--recipe").arg(recipe);
        }
        if let Some(library) = &args.nccl_lib {
            command.arg("--nccl-lib").arg(library);
        }
        let child = command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("starting rank {rank} of {}", mesh_text(cfg)))?;
        children.push((rank, metrics, child));
    }

    // All children are already running; waiting in rank order does not serialise
    // them.
    let mut failures: Vec<String> = Vec::new();
    let mut results: Vec<Result<serde_json::Value>> = Vec::with_capacity(world);
    // A failed rank leaves the rest of the world parked in a collective it will never answer, and
    // the collective backends block by design (that blocking handshake is how they prove the world
    // is in step). So the first failure terminates the survivors: the run ends with the reason
    // instead of hanging, which is the difference between a diagnosable bug and a stuck terminal.
    let mut children = children.into_iter();
    while let Some((rank, metrics, child)) = children.next() {
        let output = child
            .wait_with_output()
            .with_context(|| format!("waiting for rank {rank}"))?;
        if !output.status.success() {
            failures.push(format!(
                "rank {rank} exited with {}:\n{}",
                output.status,
                tail(&String::from_utf8_lossy(&output.stderr), 12)
            ));
            results.push(Err(anyhow::anyhow!("rank {rank} exited unsuccessfully")));
            for (other_rank, _, mut other) in children {
                let killed = other.kill().is_ok();
                let _ = other.wait();
                failures.push(format!(
                    "rank {other_rank} was {} because rank {rank} failed",
                    if killed { "terminated" } else { "already gone" }
                ));
                results.push(Err(anyhow::anyhow!("rank {rank} failed first")));
            }
            break;
        }
        // A rank's stderr is captured, so a *successful* rank's diagnostics would be swallowed —
        // which is exactly wrong for the opt-in step trace: it exists to be read. Only the trace's
        // own lines are forwarded (the rest of a successful rank's stderr is progress chatter, and
        // with four ranks it would bury the table it belongs to).
        if trace_requested() {
            for line in String::from_utf8_lossy(&output.stderr).lines() {
                if line.contains("step trace")
                    || line.contains("call(s),")
                    || line.contains("slowest single step")
                    || line.contains("staging:")
                    || line.trim_end().ends_with(" ms")
                {
                    eprintln!("rank {rank}: {line}");
                }
            }
        }
        let text = match std::fs::read_to_string(&metrics) {
            Ok(text) => text,
            Err(error) => {
                failures.push(format!(
                    "rank {rank} reported success but wrote no metrics to {}: {error}",
                    metrics.display()
                ));
                results.push(Err(anyhow::anyhow!("rank {rank} wrote no metrics")));
                continue;
            }
        };
        match serde_json::from_str(&text) {
            Ok(value) => results.push(Ok(value)),
            Err(error) => {
                failures.push(format!(
                    "rank {rank}'s metrics in {} do not parse: {error}",
                    metrics.display()
                ));
                results.push(Err(anyhow::anyhow!("rank {rank} wrote unreadable metrics")));
            }
        }
    }
    if !failures.is_empty() {
        bail!(
            "the world {} failed ({} of {world} rank(s)):\n{}",
            mesh_text(cfg),
            failures.len(),
            failures.join("\n")
        );
    }
    run::assemble_ranks(name.to_string(), world, results)
}

/// The last `lines` lines of a child's stderr, for the failure report. A world
/// failure is diagnosed from what the failing rank said, not from an exit code.
fn tail(text: &str, lines: usize) -> String {
    let all: Vec<&str> = text.lines().collect();
    let start = all.len().saturating_sub(lines);
    all[start..].join("\n")
}
