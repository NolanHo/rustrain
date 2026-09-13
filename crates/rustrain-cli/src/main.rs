//! `rustrain` — the operator-first command line.
//!
//! Three commands, each of which answers a question the old framework could not:
//!
//! * `ops list` — what implementations exist on this machine.
//! * `plan explain` — what will actually run, with which implementation, at
//!   which precision, with which communication spliced in, and how much memory
//!   it projects. With `--model <dir>` it instead expands a model description
//!   into the *global* plan (topology-free, every layout replicated) and reports
//!   which operators this machine has an implementation for.
//! * `check` — the L1/L2 ladder of spec C2: does the description expand into a
//!   sound global plan, does every node resolve to an implementation here, and
//!   does a checkpoint's metadata reconcile with the description's bindings.
//!
//! None of them reads an environment variable, and none needs a GPU: `check`
//! reads safetensors *headers*, never weights, and creates no device context.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};

use rustrain_abi::Plugin;
use rustrain_abi::ffi::{RsDeviceKind, RsDtype};
use rustrain_ops::{Phase, Recipe, Registry, ResolveError, TargetEnv};
use rustrain_parallel::{GroupMask, Mesh, ParallelConfig, ParallelLayout};
use rustrain_plan::{Attrs, OpRef, Plan, PlanBuilder, PlanNode, Slot, SlotKind};

mod device;
mod load;
mod npz;
mod run;

#[derive(Parser)]
#[command(
    name = "rustrain",
    about = "Operator-first training framework",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect the operator registry.
    Ops(OpsArgs),
    /// Inspect what a plan resolves to.
    Plan(PlanArgs),
    /// Run the L1 (structure) and L2 (loading) checks on a model description.
    Check(CheckArgs),
    /// Run one forward pass (spec C4) and write the D5 candidate dump (.npz + .json sidecar).
    ///
    /// Precision: the checkpoint and HF are bf16 while the reference provider is f32-only, so the
    /// weights are widened bf16 -> f32 (exact — bf16 is a subset of f32) and the forward executes
    /// f32; the HF reference is dumped with `--dtype bf16`, and the spec's 1% tolerance on the
    /// logits and the per-layer summaries absorbs HF's bf16 rounding, not this widening.
    Run(run::RunArgs),
}

#[derive(Args)]
struct OpsArgs {
    #[command(subcommand)]
    command: OpsCommand,
}

#[derive(Subcommand)]
enum OpsCommand {
    /// List every registered operator implementation.
    List {
        /// A plugin `.so` to load in addition to the built-in provider.
        #[arg(long = "plugin", value_name = "PATH")]
        plugins: Vec<PathBuf>,
        /// Emit JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// Run the conformance gate: is every implementation the same operator?
    Check {
        /// A plugin `.so` to load in addition to the built-in provider.
        #[arg(long = "plugin", value_name = "PATH")]
        plugins: Vec<PathBuf>,
        /// Recipe file deciding which implementation runs. Omitted: the
        /// built-in reference provider is selected.
        #[arg(long, value_name = "PATH")]
        recipe: Option<PathBuf>,
        /// Run the gate against a device provider: `cpu` (default), `cuda`,
        /// or `cuda:<index>` (`cuda` alone is device 0). With `cuda` the
        /// candidates compile and execute on that device; an unavailable
        /// device is a loud failure, never a skip.
        #[arg(long, value_name = "SPEC", default_value = "cpu")]
        device: String,
        /// Restrict to one operator.
        #[arg(long = "op", value_name = "NAME")]
        op: Option<String>,
        /// Emit JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Args)]
struct PlanArgs {
    #[command(subcommand)]
    command: PlanCommand,
}

#[derive(Subcommand)]
enum PlanCommand {
    /// Compile a plan and print every decision it made.
    Explain {
        /// Recipe file. Omitted: the built-in reference provider is selected.
        #[arg(long, value_name = "PATH")]
        recipe: Option<PathBuf>,
        /// Model directory (`config.json` + `model.json`). Given it, the global
        /// plan is expanded instead of the demonstration plan being compiled.
        #[arg(long, value_name = "DIR")]
        model: Option<PathBuf>,
        /// Tensor-parallel degree for the demonstration plan.
        #[arg(long, default_value_t = 1)]
        tp: usize,
        /// A plugin `.so` to load in addition to the built-in provider.
        #[arg(long = "plugin", value_name = "PATH")]
        plugins: Vec<PathBuf>,
        /// Emit JSON instead of text.
        #[arg(long)]
        json: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Ops(args) => match args.command {
            OpsCommand::List { plugins, json } => ops_list(&plugins, json),
            OpsCommand::Check {
                plugins,
                recipe,
                device,
                op,
                json,
            } => ops_check(&plugins, recipe.as_deref(), &device, op.as_deref(), json),
        },
        Command::Plan(args) => match args.command {
            PlanCommand::Explain {
                recipe,
                model,
                tp,
                plugins,
                json,
            } => match model {
                Some(dir) => plan_explain_model(&dir, recipe.as_deref(), tp, &plugins, json),
                None => plan_explain(recipe.as_deref(), tp, &plugins, json),
            },
        },
        Command::Check(args) => check(args),
        Command::Run(args) => run::run(args),
    }
}

/// Loads the built-in provider plus any requested plugins.
///
/// Order matters: a duplicate `op@variant` is an error naming both origins, so
/// registering the built-in first means a user's plugin cannot silently shadow
/// it — it has to collide out loud.
pub(crate) fn load_registry(plugins: &[PathBuf]) -> Result<Registry> {
    let mut registry = Registry::new();

    // SAFETY: the built-in descriptors are leaked by `PluginBuilder`, so they
    // live as long as the process.
    let builtin = unsafe { Plugin::from_static(rustrain_kernels::plugin(), "<built-in>") }
        .context("the built-in reference provider failed ABI validation")?;
    registry
        .add_plugin(builtin)
        .context("registering the built-in reference provider")?;

    for path in plugins {
        // SAFETY: the registry holds the handle that keeps the `.so` mapped for
        // as long as any descriptor from it can be used.
        let plugin = unsafe { Plugin::load(path, None) }
            .with_context(|| format!("loading plugin {}", path.display()))?;
        let n = registry
            .add_plugin(plugin)
            .with_context(|| format!("registering plugin {}", path.display()))?;
        eprintln!("loaded {n} operator(s) from {}", path.display());
    }

    Ok(registry)
}

pub(crate) fn load_recipe(path: Option<&Path>) -> Result<Recipe> {
    match path {
        Some(p) => {
            let text = std::fs::read_to_string(p)
                .with_context(|| format!("reading recipe {}", p.display()))?;
            Recipe::from_toml(&text).with_context(|| format!("parsing recipe {}", p.display()))
        }
        None => Recipe::from_toml("[kernel]\ndefault = \"reference\"\n")
            .context("the built-in recipe must parse"),
    }
}

fn ops_list(plugins: &[PathBuf], json: bool) -> Result<()> {
    let registry = load_registry(plugins)?;
    let summaries = registry.describe();

    if json {
        println!("{}", serde_json::to_string_pretty(&summaries)?);
        return Ok(());
    }

    println!(
        "{:<34} {:<22} {:<10} {:>7} {:>10}",
        "OP@VARIANT", "PLUGIN", "BACKWARD", "DTYPES", "EXPANSION"
    );
    for s in &summaries {
        println!(
            "{:<34} {:<22} {:<10} {:>7} {:>10}",
            format!("{}@{}", s.op, s.variant),
            s.plugin,
            format!("{:?}", s.backward).to_lowercase(),
            s.dtypes.len(),
            if s.has_expansion { "declared" } else { "-" },
        );
    }
    println!("\n{} implementation(s)", summaries.len());
    Ok(())
}

fn ops_check(
    plugins: &[PathBuf],
    recipe_path: Option<&Path>,
    device_spec: &str,
    only: Option<&str>,
    json: bool,
) -> Result<()> {
    use rustrain_runtime::conformance::{Harness, default_cases, uncovered_operators};

    let registry = load_registry(plugins)?;
    let recipe = load_recipe(recipe_path)?;
    let device = device::DeviceSpec::parse(device_spec)?;
    let mut harness = Harness::new(&registry, &recipe);
    if let device::DeviceSpec::Cuda(index) = device {
        // Fail up front, not per case: a device provider was requested, and
        // the gate must say loudly when no device is available — never skip.
        rustrain_runtime::CudaAllocator::new(index).map_err(|error| {
            anyhow::anyhow!(
                "`--device cuda:{index}`: no CUDA device is available for the gate to run on; \
                 the driver could not be initialised: {error}"
            )
        })?;
        harness = harness.device(RsDeviceKind::CUDA, index);
    }

    let mut cases = default_cases();
    if let Some(name) = only {
        cases.retain(|c| c.op == name);
        if cases.is_empty() {
            bail!(
                "no conformance case for `{name}`; run without --op to see which operators are \
                 covered, and check the uncovered list"
            );
        }
    }

    let mut report = rustrain_runtime::conformance::Report::default();
    for case in &cases {
        report.results.extend(harness.run(case));
    }

    if json {
        let mut doc = report.to_json();
        if let Some(obj) = doc.as_object_mut() {
            obj.insert(
                "uncovered".to_string(),
                serde_json::json!(
                    uncovered_operators()
                        .iter()
                        .map(|(op, why)| serde_json::json!({ "op": op, "reason": why }))
                        .collect::<Vec<_>>()
                ),
            );
        }
        println!("{}", serde_json::to_string_pretty(&doc)?);
    } else {
        print!("{}", report.explain());
        let uncovered = uncovered_operators();
        if !uncovered.is_empty() {
            println!("\nno case written yet:");
            for (op, why) in uncovered {
                println!("  {op}: {why}");
            }
        }
    }

    if !report.passed() {
        std::process::exit(1);
    }
    Ok(())
}

/// The demonstration plan: a two-layer MLP with a tensor-parallel pair.
///
/// Column-parallel first (`w` sharded on its output dim), then row-parallel
/// (`w` sharded on its contraction dim). The second leaves a partial sum in
/// every rank, so the compiler has to splice an all-reduce — and nothing in this
/// function asks for one.
fn demo_plan(tp: usize) -> Result<rustrain_plan::Plan> {
    if tp == 0 {
        bail!("--tp must be at least 1");
    }
    use rustrain_abi::ffi::RsDtype;

    let parallel = ParallelConfig {
        tensor: tp,
        ..Default::default()
    };
    // The mesh is the compile input; the plan only carries its fingerprint. The mask for `tp` is
    // looked up by name rather than written as a bit, so the demo cannot drift from the axis order.
    let mesh = Mesh::from_config(&parallel);
    let tp_axis = mesh
        .index_of("tp")
        .context("the canonical mesh always has a `tp` axis")?;
    let tp_mask = GroupMask::single(tp_axis).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut b = PlanBuilder::new("demo-mlp", Phase::Forward, mesh.fingerprint());
    let x = b.slot("hidden", RsDtype::F32, vec![8, 64], SlotKind::Input);

    // Column parallel: output features (dim 1 of a [K, N] weight) split.
    let w1 = b.slot_with_layout(
        "mlp.up.weight",
        RsDtype::F32,
        vec![64, 128],
        SlotKind::Weight,
        ParallelLayout::shard(1, tp_mask),
    );
    // The up-projection's output stays sharded: it feeds the row-parallel
    // down-projection directly, which is precisely why the pair needs a single
    // collective rather than one per layer.
    let h1 = b.slot_with_layout(
        "mlp.up.out",
        RsDtype::F32,
        vec![8, 128],
        SlotKind::Activation,
        ParallelLayout::shard(-1, tp_mask),
    );
    b.node(
        OpRef::new("linear"),
        vec![x, w1],
        vec![h1],
        Attrs::new(),
        "mlp.up",
    );

    let h2 = b.slot_with_layout(
        "mlp.act.out",
        RsDtype::F32,
        vec![8, 128],
        SlotKind::Activation,
        ParallelLayout::shard(-1, tp_mask),
    );
    b.node(
        OpRef::new("elementwise_unary"),
        vec![h1],
        vec![h2],
        Attrs::new().set("kind", "silu"),
        "mlp.act",
    );

    // Row parallel: the contraction dim split, so every rank holds a partial sum.
    let w2 = b.slot_with_layout(
        "mlp.down.weight",
        RsDtype::F32,
        vec![128, 64],
        SlotKind::Weight,
        ParallelLayout::shard(0, tp_mask),
    );
    let out = b.slot("mlp.down.out", RsDtype::F32, vec![8, 64], SlotKind::Output);
    b.node(
        OpRef::new("linear"),
        vec![h2, w2],
        vec![out],
        Attrs::new(),
        "mlp.down",
    );

    b.build()
        .context("the demonstration plan must be well formed")
}

fn plan_explain(
    recipe_path: Option<&Path>,
    tp: usize,
    plugins: &[PathBuf],
    json: bool,
) -> Result<()> {
    let registry = load_registry(plugins)?;
    let recipe = load_recipe(recipe_path)?;
    let plan = demo_plan(tp)?;

    let compiled = rustrain_plan::Compiler::new(&registry, &recipe, TargetEnv::default())
        .compile(&plan)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    // A group's name needs the mesh (a mask is only bit positions); the plan carries its
    // fingerprint, so the JSON can stay human-readable without a second name table.
    let mesh = compiled.mesh.clone();

    if json {
        let doc = serde_json::json!({
            "name": compiled.plan.meta.name,
            "digest": compiled.digest,
            // The mask vocabulary is bit positions, so the fingerprint is what makes a layout or a
            // collective readable: `"group": 1` means `tp` only against this axis list (§1.3 keeps
            // the fingerprint in the plan for exactly this).
            "mesh": compiled.mesh,
            "world_size": compiled.mesh.world_size(),
            "counts": {
                "slots": compiled.plan.slots.len(),
                "nodes": compiled.plan.nodes.len(),
                "steps": compiled.steps.len(),
            },
            "slots": compiled.plan.slots.iter().map(slot_json).collect::<Vec<_>>(),
            "nodes": plan_nodes_json(&compiled.plan),
            "implementations": compiled.resolved.iter().map(|r| serde_json::json!({
                "node": r.node.0,
                "op": r.spec_name,
                "plugin": r.plugin,
            })).collect::<Vec<_>>(),
            "collectives": compiled.inserted.iter().map(|c| serde_json::json!({
                "op": c.op,
                // Named through the mesh, falling back to the raw bits: a mask whose bit is outside
                // the mesh has no name, and `GroupMask`'s `Display` is the honest answer.
                "group": mesh.group_name(c.group).unwrap_or_else(|_| c.group.to_string()),
                "reason": c.reason,
                "source": c.source,
            })).collect::<Vec<_>>(),
            "memory": {
                "peak_bytes": compiled.memory.peak_bytes,
                "persistent_bytes": compiled.memory.persistent_bytes,
                "activation_pool_bytes": compiled.memory.transient_pool_bytes,
                "max_workspace_bytes": compiled.memory.max_workspace_bytes,
                "budget_bytes": compiled.memory.budget_bytes,
            },
            // D12: advisories the compiler did not refuse (a projected peak over budget is one).
            "warnings": compiled.warnings,
        });
        println!("{}", serde_json::to_string_pretty(&doc)?);
        return Ok(());
    }

    print!("{}", compiled.explain());
    Ok(())
}

/// `plan explain --model <dir>`: description → **global plan**, then report whether each operator
/// has an implementation on this machine.
///
/// Nothing is compiled here: the global plan's shapes and `layout`s stay "fully concrete + all
/// Replicate" until `instantiate` has a mesh (`docs/design/model-description.md` §0). That is what
/// contract §3.6 #10 asks for: "`explain` does not fail when a dtype has no implementation" —
/// unresolved operators go into the `implementations` report and the exit code stays 0.
fn plan_explain_model(
    dir: &Path,
    recipe_path: Option<&Path>,
    tp: usize,
    plugins: &[PathBuf],
    json: bool,
) -> Result<()> {
    let expanded = rustrain_model::expand_dir(dir)
        .with_context(|| format!("expanding the model description in {}", dir.display()))?;
    if tp != 1 {
        eprintln!(
            "note: --tp {tp} is ignored with --model: the global plan is topology-free and every \
             layout is replicate"
        );
    }

    let registry = load_registry(plugins)?;
    let recipe = load_recipe(recipe_path)?;
    let env = TargetEnv::default();
    let plan = expanded.plan;

    // Node by node: which implementation would run, or why none can (contract R-1's rejection
    // reason, passed through verbatim).
    let mut implementations = Vec::with_capacity(plan.nodes.len());
    let mut unresolved: BTreeMap<(String, String), usize> = BTreeMap::new();
    for (index, node) in plan.nodes.iter().enumerate() {
        let dtypes: Vec<_> = node
            .inputs
            .iter()
            .map(|slot| plan.slot(*slot).dtype)
            .collect();
        match recipe.resolve(&registry, &node.op.name, node.phase, &dtypes, &env) {
            Ok(op) => implementations.push(serde_json::json!({
                "node": index,
                "op": op.op.spec_name(),
                "plugin": op.op.plugin_identity(),
            })),
            Err(e) => {
                let reason = e.to_string();
                *unresolved
                    .entry((node.op.name.clone(), reason.clone()))
                    .or_default() += 1;
                implementations.push(serde_json::json!({
                    "node": index,
                    "op": node.op.name,
                    "resolved": false,
                    "reason": reason,
                }));
            }
        }
    }

    let digest = blake3::hash(&serde_json::to_vec(&plan)?)
        .to_hex()
        .to_string();
    let weights = plan
        .slots
        .iter()
        .filter(|slot| slot.kind == SlotKind::Weight)
        .count();

    if json {
        let doc = serde_json::json!({
            "name": plan.meta.name,
            "digest": digest,
            // See the compiled path above: a group mask is bit positions, and this is the axis list
            // they are positions *in*.
            "mesh": plan.meta.mesh,
            "world_size": plan.meta.mesh.world_size(),
            "counts": {
                "slots": plan.slots.len(),
                "nodes": plan.nodes.len(),
                // The global plan was never compiled, so there are no steps (a node becomes a step
                // only after instantiate).
                "steps": 0,
                "weights": weights,
                "bindings": expanded.bindings.len(),
            },
            "slots": plan.slots.iter().map(slot_json).collect::<Vec<_>>(),
            "nodes": plan_nodes_json(&plan),
            "implementations": implementations,
            "collectives": Vec::<serde_json::Value>::new(),
            "memory": {
                "peak_bytes": 0,
                "persistent_bytes": 0,
                "activation_pool_bytes": 0,
                "max_workspace_bytes": 0,
                "budget_bytes": 0,
            },
        });
        println!("{}", serde_json::to_string_pretty(&doc)?);
        return Ok(());
    }

    println!(
        "plan {}  digest {}  global (no mesh; every layout is replicate)",
        plan.meta.name,
        &digest[..12.min(digest.len())]
    );
    println!(
        "  counts  slots {}  nodes {}  weights {}  bindings {}",
        plan.slots.len(),
        plan.nodes.len(),
        weights,
        expanded.bindings.len()
    );
    let unresolved_nodes: usize = unresolved.values().sum();
    if unresolved_nodes == 0 {
        println!(
            "  implementations: all {} node(s) resolve on this machine",
            plan.nodes.len()
        );
    } else {
        println!(
            "  implementations: {unresolved_nodes} of {} node(s) have no implementation here \
             (exit code stays 0, §3.6 #10):",
            plan.nodes.len()
        );
        for ((op, reason), count) in &unresolved {
            // The first line of a failure is enough to locate it; the candidate table and the
            // contract references are in `--json`.
            let headline = reason.lines().next().unwrap_or(reason);
            println!("    {op} × {count}: {headline}");
        }
    }
    Ok(())
}

/// The JSON form of one slot.
///
/// `dtype` uses the spelling of [`rustrain_abi::ffi::RsDtype::name`] (§3.6 #5's vocabulary): the
/// derived `Serialize` would write the ABI's integer code, which means nothing to a reader of the
/// plan.
fn slot_json(slot: &Slot) -> serde_json::Value {
    serde_json::json!({
        "name": slot.name,
        "dtype": slot.dtype.name(),
        "shape": slot.shape,
        "kind": format!("{:?}", slot.kind).to_lowercase(),
        "layout": slot.layout,
    })
}

/// The JSON form of one node. Inputs and outputs are listed by slot name, because indices are
/// unreadable in a plan of 1000 nodes.
fn plan_nodes_json(plan: &Plan) -> Vec<serde_json::Value> {
    plan.nodes
        .iter()
        .enumerate()
        .map(|(index, node): (usize, &PlanNode)| {
            serde_json::json!({
                "id": index,
                "op": node.op.display(),
                "inputs": node.inputs.iter().map(|s| plan.slot(*s).name.clone()).collect::<Vec<_>>(),
                "outputs": node.outputs.iter().map(|s| plan.slot(*s).name.clone()).collect::<Vec<_>>(),
                "attrs": node.attrs,
                "phase": node.phase,
                "source": node.source.path,
            })
        })
        .collect()
}

// ===========================================================================
// `check` — the L1/L2 ladder (spec C2, C5, C6; delivery D2)
// ===========================================================================

/// The report's format identifier (C6).
const CHECK_FORMAT: &str = "rustrain.check.v1";

#[derive(Args)]
struct CheckArgs {
    /// Model directory (`config.json` + `model.json`).
    #[arg(long, value_name = "DIR")]
    model: PathBuf,
    /// Checkpoint to reconcile against: a `*.safetensors.meta.json` snapshot (C5), or a real model
    /// directory / `model.safetensors.index.json`, whose shard headers are read without any weight.
    #[arg(long, value_name = "PATH")]
    checkpoint: Option<PathBuf>,
    /// The precision L1 resolves implementations at (C6). L2 always compares the dtype the
    /// description declares: "check the structure at f32" and "verify bf16 weights against the
    /// description" are two independent questions, and this flag only answers the first.
    #[arg(long, value_name = "NAME")]
    dtype: Option<String>,
    /// The five axes are part of `check`'s interface (C2). Structure and loading both need no
    /// mesh, so this unit accepts them and does not use them yet (a mesh arrives with D3/D4).
    #[arg(long, default_value_t = 1)]
    tp: usize,
    #[arg(long, default_value_t = 1)]
    cp: usize,
    #[arg(long, default_value_t = 1)]
    ep: usize,
    #[arg(long, default_value_t = 1)]
    dp: usize,
    #[arg(long, default_value_t = 1)]
    pp: usize,
    /// Emit the machine-readable report (C6) instead of text.
    #[arg(long)]
    json: bool,
}

/// C2's verdict vocabulary, in the spelling C6 fixes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    Pass,
    Fail,
    /// Never affects the exit code. Two real producers: an `ignore` pattern that matches no
    /// checkpoint tensor (C6) and a pairing set with nothing in it.
    Warning,
    Skip,
}

impl Verdict {
    fn as_str(self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Fail => "fail",
            Verdict::Warning => "warning",
            Verdict::Skip => "skip",
        }
    }
}

/// One `checks[]` entry: a stable id, a verdict, a reason that is never empty, and the per-object
/// lines that make the reason actionable.
struct CheckItem {
    id: &'static str,
    status: Verdict,
    reason: String,
    details: Vec<String>,
}

impl CheckItem {
    fn new(id: &'static str, status: Verdict, reason: String, details: Vec<String>) -> Self {
        Self {
            id,
            status,
            reason,
            details,
        }
    }

    fn pass(id: &'static str, reason: String) -> Self {
        Self::new(id, Verdict::Pass, reason, Vec::new())
    }

    /// A `pass` whose `details` carry the machine-readable witness C5 pins: the instantiated
    /// node and slot counts per checked stage, so a no-op `instantiate` (returning the global
    /// plan) turns the report-contract gate red instead of looking identical.
    fn pass_with_details(id: &'static str, reason: String, details: Vec<String>) -> Self {
        Self::new(id, Verdict::Pass, reason, details)
    }

    fn fail(id: &'static str, reason: String, details: Vec<String>) -> Self {
        Self::new(id, Verdict::Fail, reason, details)
    }

    fn warning(id: &'static str, reason: String) -> Self {
        Self::new(id, Verdict::Warning, reason, Vec::new())
    }

    fn skip(id: &'static str, reason: impl Into<String>) -> Self {
        Self::new(id, Verdict::Skip, reason.into(), Vec::new())
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "status": self.status.as_str(),
            "reason": self.reason,
            "details": self.details,
        })
    }
}

/// C6's eight counters.
///
/// `None` means "not measured": a description that never expanded has no slot counts, and writing
/// 0 there would be a claim instead of a measurement.
#[derive(Default)]
struct Counts {
    slots: Option<usize>,
    nodes: Option<usize>,
    weights: Option<usize>,
    bindings: Option<usize>,
    slots_unbound: Option<usize>,
    tensors_unconsumed: Option<usize>,
    shape_mismatch: Option<usize>,
    dtype_mismatch: Option<usize>,
}

impl Counts {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "slots": self.slots,
            "nodes": self.nodes,
            "weights": self.weights,
            "bindings": self.bindings,
            "slots_unbound": self.slots_unbound,
            "tensors_unconsumed": self.tensors_unconsumed,
            "shape_mismatch": self.shape_mismatch,
            "dtype_mismatch": self.dtype_mismatch,
        })
    }

    fn line(&self) -> String {
        let show = |value: Option<usize>| match value {
            Some(n) => n.to_string(),
            None => "-".to_string(),
        };
        format!(
            "slots {}  nodes {}  weights {}  bindings {}  slots_unbound {}  tensors_unconsumed {}  \
             shape_mismatch {}  dtype_mismatch {}",
            show(self.slots),
            show(self.nodes),
            show(self.weights),
            show(self.bindings),
            show(self.slots_unbound),
            show(self.tensors_unconsumed),
            show(self.shape_mismatch),
            show(self.dtype_mismatch),
        )
    }
}

/// C6's report.
struct CheckReport {
    model: String,
    checkpoint: Option<String>,
    dtype: String,
    counts: Counts,
    checks: Vec<CheckItem>,
}

impl CheckReport {
    /// C6: the exit code is 0 **iff** no check is a `fail`; `warning` and `skip` never count.
    fn failed(&self) -> bool {
        self.checks.iter().any(|item| item.status == Verdict::Fail)
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "format": CHECK_FORMAT,
            "model": self.model,
            "checkpoint": self.checkpoint,
            "dtype": self.dtype,
            "counts": self.counts.to_json(),
            "checks": self.checks.iter().map(CheckItem::to_json).collect::<Vec<_>>(),
        })
    }

    fn explain(&self) -> String {
        let mut text = format!("check {}  dtype {}", self.model, self.dtype);
        if let Some(checkpoint) = &self.checkpoint {
            text.push_str(&format!("  checkpoint {checkpoint}"));
        }
        text.push('\n');
        text.push_str(&format!("  counts  {}\n", self.counts.line()));
        for item in &self.checks {
            text.push_str(&format!(
                "  {:<7} {}: {}\n",
                item.status.as_str(),
                item.id,
                item.reason
            ));
            for detail in &item.details {
                text.push_str(&format!("          {detail}\n"));
            }
        }
        let failed = self
            .checks
            .iter()
            .filter(|item| item.status == Verdict::Fail)
            .count();
        if failed == 0 {
            text.push_str("  no fail: exit code 0\n");
        } else {
            text.push_str(&format!("  {failed} fail(s): exit code 1\n"));
        }
        text
    }
}

/// One tensor as a checkpoint declares it: dtype plus shape — and, when the metadata came from a
/// real shard header, where the bytes live (`shard` = the file, `data` = the byte range in it).
/// A snapshot has neither: no weights, no device (C5).
#[derive(Clone, Debug)]
pub(crate) struct CkptTensor {
    dtype: String,
    shape: Vec<i64>,
    /// The shard file this tensor's data lives in (`load_shard_index` only).
    pub(crate) shard: Option<PathBuf>,
    /// `[start, end)` of the tensor's bytes inside `shard` (`load_shard_index` only).
    pub(crate) data: Option<(u64, u64)>,
}

impl CkptTensor {
    fn new(dtype: String, shape: Vec<i64>) -> Self {
        Self {
            dtype,
            shape,
            shard: None,
            data: None,
        }
    }
}

/// C5's checkpoint metadata, in both accepted shapes.
struct CheckpointMeta {
    /// Where the metadata came from: the snapshot's `source`, or the index that was read.
    source: String,
    tensors: BTreeMap<String, CkptTensor>,
}

fn check(args: CheckArgs) -> Result<()> {
    let CheckArgs {
        model: model_dir,
        checkpoint,
        dtype,
        tp,
        cp,
        ep,
        dp,
        pp,
        json,
    } = args;

    // C6: `--dtype` is an L1 input, never an L2 one, and the report records the precision the
    // checks actually ran at — the flag when given, else the description's own default.
    //
    // C6 also requires a full report on stdout whether the run passes or fails, so an unusable
    // `--dtype` is a `fail` item inside the report rather than an `anyhow` error that would leave
    // stdout empty. The other checks then run at the dtype the description declares (which is what
    // the report records), and the exit code already says the run is not to be trusted.
    let override_dtype = dtype.as_deref().and_then(RsDtype::parse);
    let dtype_error = dtype
        .as_deref()
        .filter(|name| RsDtype::parse(name).is_none())
        .map(|name| {
            format!(
                "`--dtype {name}` is not a dtype; expected one of {}",
                RsDtype::ALL
                    .iter()
                    .map(|d| d.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        });

    let mut report = CheckReport {
        model: model_dir.display().to_string(),
        checkpoint: checkpoint.as_ref().map(|path| path.display().to_string()),
        dtype: RsDtype::F32.name().to_string(),
        counts: Counts::default(),
        checks: Vec::new(),
    };
    if let Some(reason) = dtype_error {
        report
            .checks
            .push(CheckItem::fail("cli.arguments", reason, Vec::new()));
    }

    // D4: the five degrees build the mesh the L1 propagation checks run on. A degree of 0 is not
    // a mesh — it is rejected as an argument error the same way `--dtype` is, and the
    // mesh-dependent items are then reported as `skip` naming that reason.
    let zero_degree = [
        ("--tp", tp),
        ("--cp", cp),
        ("--ep", ep),
        ("--dp", dp),
        ("--pp", pp),
    ]
    .iter()
    .find(|(_, degree)| *degree == 0)
    .map(|(flag, _)| *flag);
    if let Some(flag) = zero_degree {
        let reason =
            format!("`{flag} 0` is not a mesh degree: every axis degree must be at least 1");
        match report
            .checks
            .iter_mut()
            .find(|item| item.id == "cli.arguments")
        {
            Some(item) => item.reason = format!("{}; {reason}", item.reason),
            None => report
                .checks
                .push(CheckItem::fail("cli.arguments", reason, Vec::new())),
        }
    }
    let mesh = if zero_degree.is_none() {
        Some(Mesh::from_config(&ParallelConfig {
            tensor: tp,
            context: cp,
            expert: ep,
            data: dp,
            pipeline: pp,
        }))
    } else {
        None
    };
    let mesh_names = format!("tp={tp}, cp={cp}, ep={ep}, dp={dp}, pp={pp}");

    // ---- L1: the description expands into one sound global plan -------------
    //
    // §3.5's unbound-slot mandate is deliberately *not* part of `l1.structure`: with `--checkpoint`
    // it belongs to `l2.binding_coverage`, which can name the slot and count it. `l1.binding_coverage`
    // reports the same number from the description alone, so a run without a checkpoint still sees
    // it.
    //
    // What `l1.structure` covers is exactly what the `--model` path can run: load + expand +
    // `check_structure`. C2's other L1 sub-checks all need `Plan::compile`, so they are reported
    // separately as `skip` with their reasons — never folded into a `pass`.
    let mut expanded = None;
    match rustrain_model::Model::load(&model_dir) {
        Err(e) => report.checks.push(CheckItem::fail(
            "l1.structure",
            format!("the description did not load: {e}"),
            Vec::new(),
        )),
        Ok(model) => {
            report.dtype = override_dtype
                .map(|dtype| dtype.name().to_string())
                .or_else(|| model.desc.dtype.clone())
                .unwrap_or_else(|| RsDtype::F32.name().to_string());
            match model.expand_lenient() {
                Err(e) => report.checks.push(CheckItem::fail(
                    "l1.structure",
                    format!("the description did not expand: {e}"),
                    Vec::new(),
                )),
                Ok(plan) => match plan.plan.check_structure() {
                    Err(e) => report.checks.push(CheckItem::fail(
                        "l1.structure",
                        format!("the expanded plan is not structurally sound: {e}"),
                        Vec::new(),
                    )),
                    Ok(()) => {
                        report.checks.push(CheckItem::pass(
                            "l1.structure",
                            format!(
                                "the description loads and expands into one global plan that passes \
                                 `check_structure` (topological order, every node produces \
                                 something, no slot written twice): {} node(s), {} slot(s), every \
                                 layout replicated. C2's compile-dependent L1 sub-checks are not \
                                 covered by this item; they are reported separately",
                                plan.plan.nodes.len(),
                                plan.plan.slots.len()
                            ),
                        ));
                        report.counts.slots = Some(plan.plan.slots.len());
                        report.counts.nodes = Some(plan.plan.nodes.len());
                        report.counts.weights = Some(
                            plan.plan
                                .slots
                                .iter()
                                .filter(|slot| slot.kind == SlotKind::Weight)
                                .count(),
                        );
                        report.counts.bindings = Some(plan.bindings.len());
                        // The expander computes this whether or not a checkpoint follows; a run
                        // without one has to show the number instead of leaving it unmeasured.
                        report.counts.slots_unbound = Some(plan.unbound_slots.len());
                        expanded = Some((model, plan));
                    }
                },
            }
        }
    }

    // ---- L1: instantiate one representative rank per PP stage, then propagate stage 0 ----
    //
    // `instantiate` needs no implementation: it joins the description's declarations with the
    // mesh and the real shapes, so its errors — a shard that does not divide, an axis the mesh
    // lacks, a stage the pipeline degree cannot address — are `fail` items even when operator
    // resolution below is incomplete. Propagation is implementation-free too, which is what
    // turns `l1.layout_propagation`, `l1.partial_fulfillment` and `l1.collective_axes` from the
    // D2 `skip`s into real checks.
    //
    // C3: every PP stage is instantiated at one representative rank (the rank whose pp
    // coordinate is the stage's), so a stage-1-only divisibility failure is not masked by a
    // clean stage 0. C4: a stage that instantiates to no nodes is a stage-declaration error,
    // not a clean bill of health. C5: the pass item's `details` carry the instantiated node
    // and slot counts per stage — a machine-readable witness that `instantiate` really pruned
    // and really ran (the report contract pins the numbers).
    //
    // Propagation deliberately stays stage-0-only: the cross-stage seam decision (complete the
    // partial before the seam, or hand it over) is D5's, and every propagation reason says so.
    match (&expanded, mesh.as_ref()) {
        (Some((_, plan)), Some(mesh)) => {
            let stages = rustrain_plan::instantiate_stages(&plan.plan, &plan.declarations(), mesh);
            let mut failures: Vec<String> = Vec::new();
            let mut empty: Vec<usize> = Vec::new();
            let mut stage_counts: Vec<String> = Vec::new();
            let mut stage_zero: Option<rustrain_plan::Plan> = None;
            for stage in stages {
                match stage.result {
                    Err(e) => {
                        failures.push(format!("stage {} (rank {}): {e}", stage.stage, stage.rank))
                    }
                    Ok(stage_plan) => {
                        if stage_plan.nodes.is_empty() {
                            empty.push(stage.stage);
                        } else {
                            if stage.stage == 0 {
                                stage_zero = Some(stage_plan.clone());
                            }
                            stage_counts.push(format!(
                                "stage {} (rank {}): {} node(s), {} slot(s)",
                                stage.stage,
                                stage.rank,
                                stage_plan.nodes.len(),
                                stage_plan.slots.len()
                            ));
                        }
                    }
                }
            }

            if !failures.is_empty() {
                report.checks.push(CheckItem::fail(
                    "l1.instantiate",
                    format!(
                        "the plan does not instantiate on every PP stage of the {mesh_names} \
                         mesh: {}",
                        failures.join("; ")
                    ),
                    Vec::new(),
                ));
                for id in [
                    "l1.layout_propagation",
                    "l1.partial_fulfillment",
                    "l1.collective_axes",
                ] {
                    report.checks.push(CheckItem::skip(
                        id,
                        format!(
                            "not evaluated: `l1.instantiate` did not succeed on every PP stage; \
                             {PROPAGATION_SCOPE}"
                        ),
                    ));
                }
            } else if !empty.is_empty() {
                let stages = empty
                    .iter()
                    .map(|stage| format!("stage {stage}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                report.checks.push(CheckItem::fail(
                    "l1.instantiate",
                    format!(
                        "{stages} of the {mesh_names} mesh instantiates to no nodes and no \
                         slots; a stage that owns no work is a stage-declaration error, not a \
                         clean bill of health"
                    ),
                    Vec::new(),
                ));
                for id in [
                    "l1.layout_propagation",
                    "l1.partial_fulfillment",
                    "l1.collective_axes",
                ] {
                    report.checks.push(CheckItem::skip(
                        id,
                        format!(
                            "not evaluated: `l1.instantiate` did not succeed on every PP stage; \
                             {PROPAGATION_SCOPE}"
                        ),
                    ));
                }
            } else {
                let stage_zero =
                    stage_zero.expect("stage 0 is not empty when every stage instantiated cleanly");
                report.checks.push(CheckItem::pass_with_details(
                    "l1.instantiate",
                    format!(
                        "the global plan instantiates on every PP stage of the {mesh_names} \
                         mesh (one representative rank per stage): every declared shard divides \
                         into a local shape"
                    ),
                    stage_counts,
                ));
                match rustrain_plan::shard::propagate(&stage_zero) {
                    Err(e) => {
                        report.checks.push(CheckItem::fail(
                            "l1.layout_propagation",
                            format!(
                                "sharding does not propagate on the stage-0 (rank 0) plan: {e}; \
                                 {PROPAGATION_SCOPE}"
                            ),
                            Vec::new(),
                        ));
                        for id in ["l1.partial_fulfillment", "l1.collective_axes"] {
                            report.checks.push(CheckItem::skip(
                                id,
                                format!(
                                    "not evaluated: layout propagation did not succeed on the \
                                     stage-0 (rank 0) plan; {PROPAGATION_SCOPE}"
                                ),
                            ));
                        }
                    }
                    Ok(propagation) => {
                        report.checks.push(CheckItem::pass(
                            "l1.layout_propagation",
                            format!(
                                "sharding propagated on the stage-0 (rank 0) plan: {} \
                                 collective(s) inserted to reconcile declared and derived \
                                 layouts; {PROPAGATION_SCOPE}",
                                propagation.inserted.len()
                            ),
                        ));
                        report.checks.push(partial_fulfillment(&propagation));
                        report.checks.push(collective_axes(&propagation, mesh));
                    }
                }
            }
        }
        (Some(_), None) => {
            // `cli.arguments` already failed on the zero degree; the mesh-dependent items
            // cannot run and say why.
            for id in [
                "l1.instantiate",
                "l1.layout_propagation",
                "l1.partial_fulfillment",
                "l1.collective_axes",
            ] {
                report.checks.push(CheckItem::skip(
                    id,
                    "not evaluated: an axis degree of 0 is not a mesh, so there is no rank 0 to \
                     instantiate on",
                ));
            }
        }
        (None, _) => {
            for id in [
                "l1.instantiate",
                "l1.layout_propagation",
                "l1.partial_fulfillment",
                "l1.collective_axes",
            ] {
                report.checks.push(CheckItem::skip(
                    id,
                    "not evaluated: the description did not expand into a plan to instantiate",
                ));
            }
        }
    }

    report
        .checks
        .extend(compile_dependent_l1_checks(expanded.is_some()));

    // ---- L1: implementation availability (never a `fail`, always a reason) --
    match &expanded {
        Some((_, plan)) => {
            let registry = load_registry(&[])?;
            let recipe = load_recipe(None)?;
            report.checks.push(implementation_availability(
                &plan.plan,
                &registry,
                &recipe,
                override_dtype,
            ));
        }
        None => report.checks.push(CheckItem::skip(
            "l1.implementation_availability",
            "not evaluated: the description did not expand into a plan to resolve operators for",
        )),
    }

    // ---- L1: §3.5's first mandate, seen from the description alone ---------
    match &expanded {
        Some((_, plan)) => report
            .checks
            .push(binding_coverage_from_the_description(plan)),
        None => report.checks.push(CheckItem::skip(
            "l1.binding_coverage",
            "not evaluated: the description did not expand into a plan whose weight slots could be \
             counted",
        )),
    }

    // ---- L2: the checkpoint reconciles with the description ----------------
    if let Some(path) = &checkpoint {
        match load_checkpoint(path) {
            Err(e) => {
                report.checks.push(CheckItem::fail(
                    "l2.binding_coverage",
                    format!("the checkpoint metadata could not be read: {e}"),
                    Vec::new(),
                ));
                for id in [
                    "l2.tensor_consumption",
                    "l2.shape_reconciliation",
                    "l2.dtype_compatibility",
                ] {
                    report.checks.push(CheckItem::skip(
                        id,
                        "not evaluated: the checkpoint metadata could not be read",
                    ));
                }
            }
            Ok(meta) => match &expanded {
                Some((model, plan)) => {
                    let l2 = l2_checks(plan, &model.desc, &meta);
                    report.counts.slots_unbound = Some(l2.slots_unbound);
                    report.counts.tensors_unconsumed = Some(l2.tensors_unconsumed);
                    // C6's shape/dtype counters stay `null` when those two items are a skip or a
                    // warning: nothing was compared, so there is no count to report.
                    report.counts.shape_mismatch = l2.shape_mismatch;
                    report.counts.dtype_mismatch = l2.dtype_mismatch;
                    report.checks.extend(l2.items);
                }
                None => {
                    for id in [
                        "l2.binding_coverage",
                        "l2.tensor_consumption",
                        "l2.shape_reconciliation",
                        "l2.dtype_compatibility",
                    ] {
                        report.checks.push(CheckItem::skip(
                            id,
                            "not evaluated: the description did not expand into a plan to check \
                             the checkpoint against",
                        ));
                    }
                }
            },
        }
    }

    // C6: `--json` writes the whole report to stdout whether it passes or fails; a text run gets
    // the same content in a readable form. The exit code is the verdict, nothing else.
    let failed = report.failed();
    if json {
        println!("{}", serde_json::to_string_pretty(&report.to_json())?);
        for item in report
            .checks
            .iter()
            .filter(|item| item.status == Verdict::Fail)
        {
            eprintln!("{}: {}", item.id, item.reason);
        }
    } else {
        print!("{}", report.explain());
    }
    if failed {
        std::process::exit(1);
    }
    Ok(())
}

/// C2's "implementation availability": can every node resolve to an implementation **at the
/// precision being checked**.
///
/// A primitive this machine has no implementation for is a `skip` with its reasons spelled out —
/// never a `fail` (the exit code is decided by `Fail` alone) and never a silent `pass`.
fn implementation_availability(
    plan: &Plan,
    registry: &Registry,
    recipe: &Recipe,
    override_dtype: Option<RsDtype>,
) -> CheckItem {
    const ID: &str = "l1.implementation_availability";
    let env = TargetEnv::default();
    let mut unresolved: BTreeMap<String, (usize, String)> = BTreeMap::new();

    for node in &plan.nodes {
        let dtypes: Vec<RsDtype> = node
            .inputs
            .iter()
            .map(|slot| checked_dtype(plan.slot(*slot).dtype, override_dtype))
            .collect();
        if let Err(e) = recipe.resolve(registry, &node.op.name, node.phase, &dtypes, &env) {
            let entry = unresolved
                .entry(node.op.name.clone())
                .or_insert_with(|| (0, unresolved_reason(&e)));
            entry.0 += 1;
        }
    }

    if unresolved.is_empty() {
        return CheckItem::pass(
            ID,
            format!(
                "all {} node(s) resolve to an implementation on this host ({env})",
                plan.nodes.len()
            ),
        );
    }

    let nodes: usize = unresolved.values().map(|(count, _)| count).sum();
    let listed = unresolved
        .iter()
        .map(|(op, (count, why))| format!("{op} ×{count} ({why})"))
        .collect::<Vec<_>>()
        .join("; ");
    let details = unresolved
        .iter()
        .map(|(op, (count, why))| format!("{op}: {count} node(s): {why}"))
        .collect();
    CheckItem::new(
        ID,
        Verdict::Skip,
        format!(
            "{nodes} of {} node(s) have no implementation on this host: {listed}. An operator no \
             loaded plugin publishes is a missing primitive, not a plan defect, so C2 makes it a \
             skip and the exit code stays 0",
            plan.nodes.len()
        ),
        details,
    )
}

/// C6's `--dtype`: it replaces the **precision** being checked, and precision is a floating-point
/// notion — an index (`i64`) or a mask (`u8`) is not a precision, so it keeps its declared dtype.
fn checked_dtype(declared: RsDtype, override_dtype: Option<RsDtype>) -> RsDtype {
    match override_dtype {
        Some(dtype) if declared.is_float() => dtype,
        _ => declared,
    }
}

/// §3.5's first mandate seen from the description alone: every weight slot is hit by a binding.
///
/// `l2.binding_coverage` answers the same question against a checkpoint and fails the run. This
/// item always carries the number — a run without `--checkpoint` must still show how many slots
/// the description leaves unbound — and it is a `warning`, not a `fail`, because C2 couples the
/// mandate to the loading check (C2 appends L2 only when a `--checkpoint` is given).
fn binding_coverage_from_the_description(plan: &rustrain_model::Expanded) -> CheckItem {
    const ID: &str = "l1.binding_coverage";
    let weights = plan
        .plan
        .slots
        .iter()
        .filter(|slot| slot.kind == SlotKind::Weight)
        .count();
    if plan.unbound_slots.is_empty() {
        return CheckItem::pass(
            ID,
            format!("every one of the {weights} weight slot(s) is hit by a binding"),
        );
    }
    CheckItem::warning(
        ID,
        format!(
            "{} of {weights} weight slot(s) have no binding: {}. §3.5's first mandate is not met by \
             the description itself; a run with `--checkpoint` reports this as a failure \
             (`l2.binding_coverage`), and a run without one sees it here",
            plan.unbound_slots.len(),
            rustrain_model::summarize(&plan.unbound_slots)
        ),
    )
}

/// Why the propagation checks evaluate stage 0 (rank 0) only, appended to every propagation
/// item's reason so a `pass` is never mistaken for "every stage was checked" (C3): the
/// cross-stage seam decision — complete the partial before the seam, or hand it over to the
/// next stage — is D5's. `propagate`'s honest seam refusal stays exactly as it is; this
/// string only says the check does not go there.
const PROPAGATION_SCOPE: &str = "propagation is evaluated on stage 0 (rank 0) only — the other PP stages are not \
     propagated (the cross-stage seam decision is D5's)";

/// C2's "every Partial is fulfilled": after propagation, a slot that still carries a partial must
/// be consumed only by the intrinsic that completes it — a compute node reading a partial means
/// no collective was inserted for it, and the plan is not runnable as declared. This is the
/// implementation-free half of the contract: the collectives are `propagate`'s output, so the
/// check can run before any operator resolves.
fn partial_fulfillment(propagation: &rustrain_plan::shard::ShardPropagation) -> CheckItem {
    const ID: &str = "l1.partial_fulfillment";
    let plan = &propagation.plan;
    let mut unfulfilled: Vec<String> = Vec::new();
    let mut fulfilled = 0usize;
    for (index, slot) in plan.slots.iter().enumerate() {
        if slot.layout.partial.is_none() {
            continue;
        }
        let read_by_compute = plan
            .nodes
            .iter()
            .filter(|node| node.inputs.iter().any(|input| input.0 == index))
            .any(|node| !rustrain_plan::ir::intrinsic::is_intrinsic(&node.op.name));
        if read_by_compute {
            unfulfilled.push(slot.name.clone());
        } else {
            fulfilled += 1;
        }
    }
    if unfulfilled.is_empty() {
        return CheckItem::pass(
            ID,
            format!(
                "every one of the {fulfilled} partial slot(s) on the stage-0 (rank 0) plan is \
                 fulfilled by an inserted collective ({} insertion(s) in total); \
                 {PROPAGATION_SCOPE}",
                propagation.inserted.len()
            ),
        );
    }
    CheckItem::fail(
        ID,
        format!(
            "{} partial slot(s) on the stage-0 (rank 0) plan are read by a compute node with \
             no inserted collective completing them: {}; {PROPAGATION_SCOPE}",
            unfulfilled.len(),
            unfulfilled.join(", ")
        ),
        Vec::new(),
    )
}

/// C2's "every collective is bound to its axes": each inserted collective's group mask must
/// address axes of the mesh, and each dim must be an axis of the tensor it converts. Propagation
/// already enforces both — the check re-reads its output so the contract is verified, not assumed.
fn collective_axes(propagation: &rustrain_plan::shard::ShardPropagation, mesh: &Mesh) -> CheckItem {
    const ID: &str = "l1.collective_axes";
    let mut unbound: Vec<String> = Vec::new();
    for collective in &propagation.inserted {
        if collective.group.validate(mesh).is_err() {
            unbound.push(format!(
                "{} group {} is not a group of the mesh",
                collective.op, collective.group
            ));
            continue;
        }
        if let Some(dim) = collective.dim {
            let rank = propagation.plan.slot(collective.produced_slot).shape.len() as i64;
            if dim < 0 || dim >= rank {
                unbound.push(format!(
                    "{} dim {dim} is not an axis of the tensor it converts",
                    collective.op
                ));
            }
        }
    }
    if unbound.is_empty() {
        return CheckItem::pass(
            ID,
            format!(
                "all {} inserted collective(s) on the stage-0 (rank 0) plan bind to axes of the \
                 mesh; {PROPAGATION_SCOPE}",
                propagation.inserted.len()
            ),
        );
    }
    CheckItem::fail(
        ID,
        format!(
            "{} inserted collective(s) on the stage-0 (rank 0) plan are not bound to the mesh \
             axes: {}; {PROPAGATION_SCOPE}",
            unbound.len(),
            unbound.join("; ")
        ),
        Vec::new(),
    )
}

/// C2's remaining L1 sub-checks that still cannot run, one `skip` each.
///
/// C2 lists seven things L1 covers. `l1.structure` covers what `load` + `expand` +
/// `check_structure` can answer, `l1.implementation_availability` covers operator resolution, and
/// with D4 the mesh exists — `l1.instantiate`, `l1.layout_propagation`, `l1.partial_fulfillment`
/// and `l1.collective_axes` now run for real. The three left over all need a compiled plan, and
/// `rustrain check` does not run `Plan::compile` yet — the compiler is D5's planner half, still
/// outstanding. (Resolution itself is no longer the reason: with `moe_layer` published, every node
/// of the real description resolves at `--dtype f32`. A check that never runs its compile is still
/// a skip, not a pass.)
fn compile_dependent_l1_checks(expanded: bool) -> Vec<CheckItem> {
    // `(id, why this sub-check needs a compiled plan)`, in C2's order.
    const SUBCHECKS: [(&str, &str); 3] = [
        (
            "l1.compile",
            "`Plan::compile` is not run by `rustrain check` yet — the compiler is D5's planner \
             half, still outstanding; until it lands this sub-check cannot run",
        ),
        (
            "l1.operator_shapes",
            "operators are only asked for their shapes by the compiler's shape-inference pass, \
             and `rustrain check` does not run the compiler yet (D5's planner half is \
             outstanding)",
        ),
        (
            "l1.slot_allocation",
            "allocation and alias analysis live in the plan's memory pass, and `rustrain check` \
             does not run the compiler yet (D5's planner half is outstanding)",
        ),
    ];
    SUBCHECKS
        .iter()
        .map(|(id, why)| {
            let reason = if expanded {
                format!(
                    "not evaluated: {why}. `l1.structure` covers expansion and `check_structure` \
                     only and does not stand in for this sub-check"
                )
            } else {
                format!(
                    "not evaluated: the description did not expand into a plan, and `{id}` needs a \
                     compiled one (D5)"
                )
            };
            CheckItem::skip(id, reason)
        })
        .collect()
}

/// Why one node's operator did not resolve, as the single fact a 1000-node report can carry. The
/// candidate table and the contract references an error prints are for a human reading one
/// failure.
fn unresolved_reason(error: &ResolveError) -> String {
    match error {
        ResolveError::UnknownOp { name, .. } => format!("no loaded plugin publishes `{name}`"),
        other => match other.failure() {
            Some(failure) => failure
                .candidates
                .iter()
                .map(|row| format!("{}: {}", row.variant, row.reason))
                .collect::<Vec<_>>()
                .join("; "),
            None => other
                .to_string()
                .lines()
                .next()
                .unwrap_or_default()
                .to_string(),
        },
    }
}

// ---- L2: the four checks §3.5 mandates -------------------------------------

/// One `(checkpoint tensor → slot)` pairing, worked out by capture substitution rather than by
/// position (C5: `ResolvedBinding::slots` is grouped by target, so a zip pairs the wrong layer).
///
/// Both `check` (shape/dtype reconciliation) and `run` (loading the actual bytes) consume this
/// list — the loader must move the *same* mapping the check verifies, not a second one.
pub(crate) struct Pair {
    /// Index into `Expanded::bindings`.
    pub(crate) binding: usize,
    pub(crate) tensor: String,
    pub(crate) slot: String,
    /// Which segment of the binding's `split` this slot is.
    pub(crate) segment: usize,
}

/// What pairing a checkpoint against a description found, before any verdict is drawn.
pub(crate) struct Pairing {
    pub(crate) pairs: Vec<Pair>,
    /// A source matched a tensor but its captures filled no target pattern.
    pub(crate) unpaired: Vec<String>,
    /// Binding sources no checkpoint tensor matched.
    pub(crate) missing_sources: Vec<String>,
    /// Tensors claimed by more than one binding.
    pub(crate) shared_tensors: Vec<String>,
    /// The number of weight slots the bindings declare.
    pub(crate) covered: usize,
    /// The pairing is only a reconciliation when it is one-to-one.
    pub(crate) is_bijection: bool,
    /// An empty pairing set that is nonetheless one-to-one: nothing to reconcile.
    pub(crate) nothing_to_reconcile: bool,
}

/// The `(checkpoint tensor → slot)` pairing, by capture substitution (C5 forbids zipping
/// `ResolvedBinding::slots` against source order).
fn pairing(expanded: &rustrain_model::Expanded, meta: &CheckpointMeta) -> Pairing {
    // One checkpoint tensor instance → one slot, by capture substitution. `covered` is the
    // number of weight slots the bindings declare; a pairing that does not come out one-to-one
    // is not a pairing, and the counts below would be a Cartesian product dressed up as a
    // reconciliation.
    let mut pairs: Vec<Pair> = Vec::new();
    let mut unpaired: Vec<String> = Vec::new();
    let mut missing_sources: Vec<String> = Vec::new();
    let mut shared_tensors: Vec<String> = Vec::new();
    let covered: usize = expanded
        .bindings
        .iter()
        .map(|binding| binding.slots.len())
        .sum();

    // Distinct patterns cannot overlap today (`**` belongs to `ignore`), but nothing stops two
    // *different* sources from matching the same tensor, which would feed it to two slots.
    let mut shared: BTreeSet<&String> = BTreeSet::new();
    for name in meta.tensors.keys() {
        let sources: Vec<&str> = expanded
            .bindings
            .iter()
            .filter(|binding| rustrain_model::matches(&binding.source, name))
            .map(|binding| binding.source.as_str())
            .collect();
        if sources.len() > 1 {
            shared.insert(name);
            shared_tensors.push(format!(
                "checkpoint tensor `{name}` is claimed by {} bindings: {}",
                sources.len(),
                sources.join(", ")
            ));
        }
    }

    for (index, binding) in expanded.bindings.iter().enumerate() {
        let instances: Vec<(&String, Vec<String>)> = meta
            .tensors
            .keys()
            .filter_map(|name| {
                rustrain_model::match_name(&binding.source, name).map(|captures| (name, captures))
            })
            .collect();
        if instances.is_empty() {
            missing_sources.push(binding.source.clone());
            continue;
        }
        let patterns = target_patterns(binding);
        for (name, captures) in instances {
            // A tensor two bindings claim is reported once, as itself; pairing it twice is what
            // produced the "4 pairing(s)" that contradicted `counts.weights = 2`.
            if shared.contains(name) {
                continue;
            }
            for (segment, pattern) in patterns.iter().enumerate() {
                let Some(slot) = rustrain_model::apply_captures(pattern, &captures) else {
                    unpaired.push(format!(
                        "binding `{}`: tensor `{name}` matches, but its captures do not fill the \
                         target pattern `{pattern}`",
                        binding.source
                    ));
                    continue;
                };
                if expanded.plan.slot_id(&slot).is_none()
                    || !binding.slots.iter().any(|resolved| resolved.slot == slot)
                {
                    unpaired.push(format!(
                        "binding `{}`: tensor `{name}` maps onto slot `{slot}`, which the plan does \
                         not declare as a target of this binding",
                        binding.source
                    ));
                    continue;
                }
                pairs.push(Pair {
                    binding: index,
                    tensor: name.clone(),
                    slot,
                    segment,
                });
            }
        }
    }

    let is_bijection = shared_tensors.is_empty() && pairs.len() == covered;
    // C6's "nothing to reconcile" state: an empty pairing set that is nonetheless one-to-one.
    let nothing_to_reconcile =
        pairs.is_empty() && is_bijection && expanded.unbound_slots.is_empty();

    Pairing {
        pairs,
        unpaired,
        missing_sources,
        shared_tensors,
        covered,
        is_bijection,
        nothing_to_reconcile,
    }
}

/// What the four L2 checks found: C6's four counters plus the items that carry the reasons.
struct L2Result {
    items: Vec<CheckItem>,
    slots_unbound: usize,
    tensors_unconsumed: usize,
    /// `None` when no shape (resp. dtype) was actually compared, which is the same "not measured"
    /// rule [`Counts`] spells out: the two items are a `skip` or a `warning` there, and writing `0`
    /// next to a check that compared nothing is a claim, not a measurement.
    shape_mismatch: Option<usize>,
    dtype_mismatch: Option<usize>,
}

fn l2_checks(
    expanded: &rustrain_model::Expanded,
    desc: &rustrain_model::ModelDesc,
    meta: &CheckpointMeta,
) -> L2Result {
    let plan = &expanded.plan;

    // ---- pairing ----
    //
    // One checkpoint tensor instance → one slot, by capture substitution (C5 forbids zipping
    // `ResolvedBinding::slots` against source order). `covered` is the number of weight slots the
    // bindings declare; a pairing that does not come out one-to-one is not a pairing, and the
    // counts below would be a Cartesian product dressed up as a reconciliation.
    let Pairing {
        pairs,
        unpaired,
        missing_sources,
        shared_tensors,
        covered,
        is_bijection: pairing_is_a_bijection,
        nothing_to_reconcile,
    } = pairing(expanded, meta);

    // The shape/dtype counters are a measurement only when the pairing is one-to-one *and* there is
    // a pairing to measure; every other state is a skip or a warning, and `0` would claim a
    // comparison that never happened.
    let compared = pairing_is_a_bijection && !pairs.is_empty();

    // ---- l2.binding_coverage: every weight slot has a binding, every source a tensor ----
    let mut coverage_details: Vec<String> = Vec::new();
    let mut coverage_reason: Vec<String> = Vec::new();
    if !expanded.unbound_slots.is_empty() {
        coverage_details.push(format!(
            "weight slot(s) no binding hits: {}",
            rustrain_model::summarize(&expanded.unbound_slots)
        ));
        coverage_reason.push(format!(
            "{} weight slot(s) have no binding: {}",
            expanded.unbound_slots.len(),
            rustrain_model::summarize(&expanded.unbound_slots)
        ));
    }
    if !missing_sources.is_empty() {
        coverage_details.push(format!(
            "binding source(s) no checkpoint tensor matches: {}",
            rustrain_model::summarize(&missing_sources)
        ));
        coverage_reason.push(format!(
            "{} binding source(s) match no checkpoint tensor: {}",
            missing_sources.len(),
            rustrain_model::summarize(&missing_sources)
        ));
    }
    if !unpaired.is_empty() {
        coverage_details.extend(unpaired.iter().cloned());
        coverage_reason.push(format!(
            "{} pairing(s) between a checkpoint tensor and a slot could not be resolved",
            unpaired.len()
        ));
    }
    if !shared_tensors.is_empty() {
        coverage_details.extend(shared_tensors.iter().cloned());
        coverage_reason.push(format!(
            "{} checkpoint tensor(s) are claimed by more than one binding, so one tensor would be \
             loaded into two slots (§3.7 #4)",
            shared_tensors.len()
        ));
    }
    if shared_tensors.is_empty() && pairs.len() != covered {
        coverage_reason.push(format!(
            "the checkpoint↔slot pairing is not one-to-one: {} pairing(s) for the {covered} weight \
             slot(s) the bindings cover",
            pairs.len()
        ));
    }
    let coverage = if nothing_to_reconcile {
        CheckItem::warning(
            "l2.binding_coverage",
            format!(
                "no pairing to check: the description declares no weight slot a binding covers \
                 ({} binding(s) in total) and leaves no weight slot unbound, so no checkpoint \
                 tensor can be paired with a slot; nothing was measured here",
                expanded.bindings.len()
            ),
        )
    } else if coverage_reason.is_empty() {
        CheckItem::pass(
            "l2.binding_coverage",
            format!(
                "every one of the {} weight slot(s) is hit by exactly one binding, every one of the \
                 {} binding source(s) matches a tensor of {}, and the resulting pairing is \
                 one-to-one ({covered} pairing(s))",
                expanded
                    .plan
                    .slots
                    .iter()
                    .filter(|slot| slot.kind == SlotKind::Weight)
                    .count(),
                expanded.bindings.len(),
                meta.source
            ),
        )
    } else {
        CheckItem::fail(
            "l2.binding_coverage",
            format!(
                "{}; §3.5: every weight slot must be hit by exactly one binding, every binding's \
                 source must name checkpoint tensors, and one tensor must feed one slot",
                coverage_reason.join("; ")
            ),
            coverage_details,
        )
    };

    // ---- l2.tensor_consumption: consumed by a binding, or explicitly ignored ----
    let Consumption {
        unconsumed,
        ignored,
        ignore_hits,
    } = consumption(expanded, desc, meta);
    let consumption_item = if unconsumed.is_empty() && nothing_to_reconcile {
        CheckItem::warning(
            "l2.tensor_consumption",
            format!(
                "no pairing to check: the description declares no weight slot, so no tensor of {} \
                 could be consumed by a binding; {} of its {} tensor(s) are matched by the {} \
                 explicit `ignore` pattern(s) and nothing was measured here",
                meta.source,
                ignored,
                meta.tensors.len(),
                desc.ignore.len()
            ),
        )
    } else if unconsumed.is_empty() {
        CheckItem::pass(
            "l2.tensor_consumption",
            format!(
                "all {} tensor(s) of {} are either consumed by a binding or matched by one of the \
                 {} explicit `ignore` pattern(s) ({ignored} ignored)",
                meta.tensors.len(),
                meta.source,
                desc.ignore.len()
            ),
        )
    } else {
        CheckItem::fail(
            "l2.tensor_consumption",
            format!(
                "{} of {} checkpoint tensor(s) are neither consumed by a binding nor matched by an \
                 `ignore` entry: {}",
                unconsumed.len(),
                meta.tensors.len(),
                rustrain_model::summarize(&unconsumed)
            ),
            unconsumed.clone(),
        )
    };

    // C6: an `ignore` pattern that matches 0 tensors is a `warning`, not a `fail` — the same
    // description may be checked against another checkpoint, but a pattern that matches nothing is
    // usually a typo that only a report can show.
    //
    // Every pattern's own hit count goes into `details` (C5's "explicit declaration"): a single
    // total cannot show *which* pattern drops what, so `model.visual.**` and a pattern that happens
    // to cover the same 333 tensors would look alike in the report.
    let ignore_details: Vec<String> = desc
        .ignore
        .iter()
        .zip(&ignore_hits)
        .map(|(pattern, hits)| format!("`{pattern}` matches {hits} tensor(s)"))
        .collect();
    let mut ignore_checks: Vec<CheckItem> = Vec::new();
    if desc.ignore.is_empty() {
        ignore_checks.push(CheckItem::skip(
            "l2.ignore_coverage",
            "not evaluated: the description declares no `ignore` pattern, so there is none to match \
             against this checkpoint",
        ));
    } else {
        for (pattern, hits) in desc.ignore.iter().zip(&ignore_hits) {
            if *hits == 0 {
                ignore_checks.push(CheckItem::new(
                    "l2.ignore_coverage",
                    Verdict::Warning,
                    format!(
                        "`ignore` pattern `{pattern}` matches none of the {} tensor(s) of {}; a \
                         declared pattern that matches nothing is usually a typo",
                        meta.tensors.len(),
                        meta.source
                    ),
                    ignore_details.clone(),
                ));
            }
        }
        if ignore_checks.is_empty() {
            ignore_checks.push(CheckItem::new(
                "l2.ignore_coverage",
                Verdict::Pass,
                format!(
                    "every one of the {} `ignore` pattern(s) matches at least one tensor ({ignored} \
                     tensor(s) ignored in total)",
                    desc.ignore.len()
                ),
                ignore_details,
            ));
        }
    }

    // ---- l2.shape_reconciliation + l2.dtype_compatibility ----
    let mut shape_details: Vec<String> = Vec::new();
    let mut shape_mismatch = 0usize;
    let mut dtype_details: Vec<String> = Vec::new();
    let mut dtype_mismatch = 0usize;

    for pair in &pairs {
        let binding = &expanded.bindings[pair.binding];
        let tensor = &meta.tensors[&pair.tensor];
        // The pairing above already proved the slot exists; this repeats the lookup without a
        // panic path, so a report can never be a crash.
        let Some(slot_id) = plan.slot_id(&pair.slot) else {
            continue;
        };
        let slot = plan.slot(slot_id);
        // `transform` is evaluated, not guessed: C6's two verbs are implemented here, and a
        // transform that does not fit *this* checkpoint shape is a mismatch, never a skip.
        match transformed_shape(&tensor.shape, &binding.transform) {
            Err(why) => {
                shape_mismatch += 1;
                shape_details.push(format!(
                    "slot `{}` <- `{}` {}: {why}",
                    pair.slot,
                    pair.tensor,
                    shape_text(&tensor.shape)
                ));
            }
            Ok(shape) => {
                let expected = match &binding.split {
                    Some(split) => split_shape(&shape, split, pair.segment),
                    None => Ok(shape),
                };
                match expected {
                    Err(why) => {
                        shape_mismatch += 1;
                        shape_details.push(format!(
                            "slot `{}` <- `{}` {}: {why}",
                            pair.slot,
                            pair.tensor,
                            shape_text(&tensor.shape)
                        ));
                    }
                    Ok(expected) if expected != slot.shape => {
                        shape_mismatch += 1;
                        shape_details.push(format!(
                            "slot `{}` <- `{}` {}: the checkpoint shape maps onto {} (transform: \
                             {}), but the slot declares {}",
                            pair.slot,
                            pair.tensor,
                            shape_text(&tensor.shape),
                            shape_text(&expected),
                            transform_text(&binding.transform),
                            shape_text(&slot.shape)
                        ));
                    }
                    Ok(_) => {}
                }
            }
        }
        if tensor.dtype != slot.dtype.name() {
            dtype_mismatch += 1;
            dtype_details.push(format!(
                "slot `{}` <- `{}` ({}): the checkpoint declares `{}`, the description declares \
                 `{}`",
                pair.slot,
                pair.tensor,
                binding.source,
                tensor.dtype,
                slot.dtype.name()
            ));
        }
    }

    let not_a_bijection = |id: &'static str, what: &str| {
        CheckItem::skip(
            id,
            format!(
                "not evaluated: the checkpoint↔slot pairing is not one-to-one ({} pairing(s) for \
                 {covered} weight slot(s)), so there is no {what} to compare; see \
                 `l2.binding_coverage`",
                pairs.len()
            ),
        )
    };

    let shape = if !pairing_is_a_bijection {
        not_a_bijection("l2.shape_reconciliation", "trustworthy checkpoint shape")
    } else if shape_mismatch > 0 {
        CheckItem::fail(
            "l2.shape_reconciliation",
            format!(
                "{shape_mismatch} of {} pairing(s) do not reconcile: `transform` + `split` cannot \
                 map the checkpoint shape onto the shape the slot declares",
                pairs.len()
            ),
            shape_details,
        )
    } else if pairs.is_empty() {
        // Nothing was compared: an empty checkpoint, or a description with no weight slot. A
        // `pass` here would read as "verified" when nothing was.
        CheckItem::warning(
            "l2.shape_reconciliation",
            format!(
                "no pairing to reconcile: the description declares {covered} weight slot(s) and \
                 {} declares {} tensor(s), so no checkpoint shape was compared with any slot",
                meta.source,
                meta.tensors.len()
            ),
        )
    } else {
        CheckItem::pass(
            "l2.shape_reconciliation",
            format!(
                "all {} pairing(s) reconcile: `transform` + `split` map every checkpoint shape \
                 onto the shape its slot declares",
                pairs.len()
            ),
        )
    };

    let dtype = if !pairing_is_a_bijection {
        not_a_bijection("l2.dtype_compatibility", "trustworthy checkpoint dtype")
    } else if dtype_mismatch > 0 {
        CheckItem::fail(
            "l2.dtype_compatibility",
            format!(
                "{dtype_mismatch} of {} pairing(s) disagree about dtype: the checkpoint's dtype is \
                 compared with the dtype the description declares (C6: `--dtype` does not change \
                 this comparison)",
                pairs.len()
            ),
            dtype_details,
        )
    } else if pairs.is_empty() {
        CheckItem::warning(
            "l2.dtype_compatibility",
            format!(
                "no pairing to compare: the description declares {covered} weight slot(s) and {} \
                 declares {} tensor(s), so no checkpoint dtype was compared with any slot",
                meta.source,
                meta.tensors.len()
            ),
        )
    } else {
        CheckItem::pass(
            "l2.dtype_compatibility",
            format!(
                "all {} pairing(s) agree: every checkpoint tensor has the dtype the description \
                 declares for its slot",
                pairs.len()
            ),
        )
    };

    let mut items = vec![coverage, consumption_item];
    items.extend(ignore_checks);
    items.push(shape);
    items.push(dtype);
    L2Result {
        items,
        slots_unbound: expanded.unbound_slots.len(),
        tensors_unconsumed: unconsumed.len(),
        shape_mismatch: compared.then_some(shape_mismatch),
        dtype_mismatch: compared.then_some(dtype_mismatch),
    }
}

/// What the consumption pass found: which checkpoint tensors no binding consumes and no `ignore`
/// pattern covers, plus the per-pattern hit counts.
struct Consumption {
    unconsumed: Vec<String>,
    ignored: usize,
    /// One hit count per `desc.ignore` entry, in declaration order.
    ignore_hits: Vec<usize>,
}

/// §3.5's second mandate: every checkpoint tensor is consumed by a binding or explicitly ignored.
/// Shared by `check`'s `l2.tensor_consumption` and the loader's extra-tensor error — one count,
/// one source.
fn consumption(
    expanded: &rustrain_model::Expanded,
    desc: &rustrain_model::ModelDesc,
    meta: &CheckpointMeta,
) -> Consumption {
    let mut unconsumed: Vec<String> = Vec::new();
    let mut ignored = 0usize;
    // C6: an `ignore` pattern that matches nothing is a warning. Per pattern, because the pattern
    // that matched nothing is the fact worth reporting.
    let mut ignore_hits = vec![0usize; desc.ignore.len()];
    for name in meta.tensors.keys() {
        let consumed = expanded
            .bindings
            .iter()
            .any(|binding| rustrain_model::matches(&binding.source, name));
        // Every pattern is asked about every tensor of the checkpoint: C6's warning is about a
        // pattern that matches *nothing*, and two `ignore` entries may well overlap on one tensor.
        let mut matched_by_ignore = false;
        for (index, pattern) in desc.ignore.iter().enumerate() {
            if rustrain_model::matches(pattern, name) {
                ignore_hits[index] += 1;
                matched_by_ignore = true;
            }
        }
        if consumed {
            continue;
        }
        if matched_by_ignore {
            ignored += 1;
        } else {
            unconsumed.push(name.clone());
        }
    }
    Consumption {
        unconsumed,
        ignored,
        ignore_hits,
    }
}

/// One binding's distinct target patterns, in declaration order: the index of a pattern is the
/// segment index of `ResolvedBinding::split`.
pub(crate) fn target_patterns(binding: &rustrain_model::ResolvedBinding) -> Vec<String> {
    let mut patterns: Vec<String> = Vec::new();
    for slot in &binding.slots {
        if !patterns.contains(&slot.pattern) {
            patterns.push(slot.pattern.clone());
        }
    }
    patterns
}

/// Apply C6's `transform` vocabulary to a checkpoint tensor's shape.
///
/// The two verbs are the whole vocabulary and both are evaluated: `transpose(i, j)` permutes, and
/// `slice(dim, start, len)` keeps `len` positions from `start` along `dim`. The grammar is
/// [`rustrain_model::parse_transform`]'s — the same parser `expand` validates descriptions with —
/// so an error here is about *this* shape (an axis out of range, or a slice that leaves the axis),
/// never about a verb nobody implemented.
pub(crate) fn transformed_shape(shape: &[i64], transform: &[String]) -> Result<Vec<i64>, String> {
    use rustrain_model::Transform;

    let mut shape = shape.to_vec();
    for step in transform {
        let parsed = rustrain_model::parse_transform(step)?;
        match parsed {
            Transform::Transpose { i, j } => {
                let a = axis(i, shape.len()).ok_or_else(|| {
                    format!(
                        "transform `{step}`: axis {i} is out of range for {}",
                        shape_text(&shape)
                    )
                })?;
                let b = axis(j, shape.len()).ok_or_else(|| {
                    format!(
                        "transform `{step}`: axis {j} is out of range for {}",
                        shape_text(&shape)
                    )
                })?;
                shape.swap(a, b);
            }
            Transform::Slice { dim, start, len } => {
                let d = axis(dim, shape.len()).ok_or_else(|| {
                    format!(
                        "transform `{step}`: axis {dim} is out of range for {}",
                        shape_text(&shape)
                    )
                })?;
                let size = shape[d];
                let end = start.checked_add(len).ok_or_else(|| {
                    format!("transform `{step}`: start {start} + len {len} overflows i64")
                })?;
                if end > size {
                    return Err(format!(
                        "transform `{step}`: the slice takes [{start}, {end}) along axis {dim}, \
                         which has {size} position(s)"
                    ));
                }
                shape[d] = len;
            }
        }
    }
    Ok(shape)
}

/// One segment of a fused storage (§3.4's `split`): the segment's size replaces the split axis, and
/// the sizes must add up to the axis they split (§3.5: the intervals must be in range).
pub(crate) fn split_shape(
    shape: &[i64],
    split: &rustrain_model::ResolvedSplit,
    segment: usize,
) -> Result<Vec<i64>, String> {
    let Some(dim) = axis(split.dim, shape.len()) else {
        return Err(format!(
            "split dim {} is out of range for the {} the transform produced",
            split.dim,
            shape_text(shape)
        ));
    };
    // A size is a description literal, so its sum can leave `i64`; adding the sizes unchecked is
    // how a wrong description used to abort the whole report (§3.6 #8: report, never panic).
    let mut total: i64 = 0;
    for size in &split.sizes {
        total = total.checked_add(*size).ok_or_else(|| {
            format!(
                "split sizes {:?} overflow i64 when summed: {total} + {size} along dim {dim}",
                split.sizes
            )
        })?;
    }
    if total != shape[dim] {
        return Err(format!(
            "split sizes {:?} sum to {total}, but the split axis of {} is {}",
            split.sizes,
            shape_text(shape),
            shape[dim]
        ));
    }
    let Some(size) = split.sizes.get(segment) else {
        return Err(format!(
            "the split declares {} segment(s), so segment {segment} does not exist",
            split.sizes.len()
        ));
    };
    let mut out = shape.to_vec();
    out[dim] = *size;
    Ok(out)
}

/// A dimension index; negative counts from the end, as `transpose(i, j)` is a general permutation.
pub(crate) fn axis(dim: i64, rank: usize) -> Option<usize> {
    if dim < 0 {
        let from_end = dim + rank as i64;
        (from_end >= 0).then_some(from_end as usize)
    } else {
        ((dim as usize) < rank).then_some(dim as usize)
    }
}

pub(crate) fn shape_text(shape: &[i64]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(crate) fn transform_text(transform: &[String]) -> String {
    if transform.is_empty() {
        "none".to_string()
    } else {
        transform.join(", ")
    }
}

// ---- C5: checkpoint metadata, both accepted shapes -------------------------

/// Read C5's checkpoint metadata from one of its two forms: a `*.safetensors.meta.json` snapshot,
/// or a real model directory (or its index file), whose shard **headers** are read and no weight
/// is ever touched.
pub(crate) fn load_checkpoint(path: &Path) -> Result<CheckpointMeta> {
    if path.is_dir() {
        let index = path.join("model.safetensors.index.json");
        if !index.is_file() {
            bail!(
                "{} is a directory without model.safetensors.index.json; C5 accepts a real model \
                 directory or a `*.safetensors.meta.json` snapshot",
                path.display()
            );
        }
        return load_shard_index(&index);
    }
    if !path.is_file() {
        bail!("no such file or directory: {}", path.display());
    }
    if path
        .file_name()
        .is_some_and(|name| name == "model.safetensors.index.json")
    {
        return load_shard_index(path);
    }
    load_snapshot(path)
}

/// C5's offline, reproducible form: `{"format": "rustrain.ckpt_meta.v1", "source": …, "tensors":
/// {name: {dtype, shape}}}`.
fn load_snapshot(path: &Path) -> Result<CheckpointMeta> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading the checkpoint snapshot {}", path.display()))?;
    let doc: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing the checkpoint snapshot {}", path.display()))?;
    let format = doc
        .get("format")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if format != "rustrain.ckpt_meta.v1" {
        bail!(
            "{} is not a checkpoint metadata snapshot: `format` is `{format}`, expected \
             `rustrain.ckpt_meta.v1`",
            path.display()
        );
    }
    let source = doc
        .get("source")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    let tensors = doc
        .get("tensors")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow!("{} has no `tensors` object", path.display()))?;
    let mut out = BTreeMap::new();
    for (name, entry) in tensors {
        out.insert(name.clone(), tensor_of(entry, name)?);
    }
    Ok(CheckpointMeta {
        source: if source.is_empty() {
            path.display().to_string()
        } else {
            source
        },
        tensors: out,
    })
}

/// The real form: an index maps every tensor to a shard, and each shard's header is read with two
/// `read_exact`s — 8 bytes of length, then the JSON table. No weight byte is ever read.
pub(crate) fn load_shard_index(index: &Path) -> Result<CheckpointMeta> {
    let text = std::fs::read_to_string(index)
        .with_context(|| format!("reading the safetensors index {}", index.display()))?;
    let doc: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing the safetensors index {}", index.display()))?;
    let weight_map = doc
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow!("{} has no `weight_map` object", index.display()))?;
    let dir = index.parent().unwrap_or_else(|| Path::new("."));

    // Group by shard: one header read per shard, not one per tensor.
    let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (tensor, shard) in weight_map {
        let shard = shard
            .as_str()
            .ok_or_else(|| anyhow!("{}: `{tensor}` maps to a non-string shard", index.display()))?;
        by_shard
            .entry(shard.to_string())
            .or_default()
            .push(tensor.clone());
    }

    let mut tensors = BTreeMap::new();
    for (shard, names) in by_shard {
        let path = dir.join(&shard);
        let header = read_safetensors_header(&path)?;
        for name in names {
            let entry = header.get(&name).ok_or_else(|| {
                anyhow!(
                    "{}: the index lists `{name}`, but the shard header does not",
                    path.display()
                )
            })?;
            tensors.insert(name, entry.clone());
        }
    }
    Ok(CheckpointMeta {
        source: index.display().to_string(),
        tensors,
    })
}

/// One `.safetensors` shard's tensor table, **header only**.
pub(crate) fn read_safetensors_header(path: &Path) -> Result<BTreeMap<String, CkptTensor>> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("opening the safetensors shard {}", path.display()))?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length)
        .with_context(|| format!("reading the header length of {}", path.display()))?;
    let length = u64::from_le_bytes(length);
    // A header that claims a gigabyte is a corrupt or non-safetensors file, not a big model.
    if length > 1 << 30 {
        bail!(
            "{}: the safetensors header claims {length} bytes; refusing to read it",
            path.display()
        );
    }
    let mut header = vec![0u8; length as usize];
    file.read_exact(&mut header)
        .with_context(|| format!("reading the header of {}", path.display()))?;
    let doc: serde_json::Value = serde_json::from_slice(&header)
        .with_context(|| format!("parsing the header of {}", path.display()))?;
    let entries = doc.as_object().ok_or_else(|| {
        anyhow!(
            "{}: the safetensors header is not an object",
            path.display()
        )
    })?;
    let mut out = BTreeMap::new();
    for (name, entry) in entries {
        if name == "__metadata__" {
            continue;
        }
        let mut tensor = tensor_of(entry, name)?;
        // A shard header also says where the bytes live — the byte range the loader seeks to.
        tensor.shard = Some(path.to_path_buf());
        out.insert(name.clone(), tensor);
    }
    Ok(out)
}

/// One `{"dtype": …, "shape": […], "data_offsets": [start, end]}` entry, from either metadata
/// form. A snapshot has no `data_offsets`, so `CkptTensor::data` stays `None` there — which is
/// exactly what marks it as metadata without bytes.
pub(crate) fn tensor_of(entry: &serde_json::Value, name: &str) -> Result<CkptTensor> {
    let dtype = entry
        .get("dtype")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("tensor `{name}` has no string `dtype`"))?;
    let shape = entry
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow!("tensor `{name}` has no `shape` array"))?;
    let shape = shape
        .iter()
        .map(|dim| {
            dim.as_i64()
                .ok_or_else(|| anyhow!("tensor `{name}`: a shape entry is not an integer"))
        })
        .collect::<Result<Vec<i64>>>()?;
    let data = match entry.get("data_offsets") {
        None => None,
        Some(offsets) => {
            let offsets = offsets
                .as_array()
                .ok_or_else(|| anyhow!("tensor `{name}`: `data_offsets` is not an array"))?;
            let start = offsets
                .first()
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| anyhow!("tensor `{name}`: `data_offsets[0]` is not an integer"))?;
            let end = offsets
                .get(1)
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| anyhow!("tensor `{name}`: `data_offsets[1]` is not an integer"))?;
            if end < start {
                bail!("tensor `{name}`: `data_offsets` [{start}, {end}) is empty or reversed");
            }
            Some((start, end))
        }
    };
    let mut out = CkptTensor::new(
        // safetensors spells dtypes in upper case (`BF16`, `F8_E4M3`); §3.6 #5's vocabulary is
        // `RsDtype::name()`'s lower-case spelling.
        dtype.to_ascii_lowercase().replace('_', ""),
        shape,
    );
    out.data = data;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steps(transform: &[&str]) -> Vec<String> {
        transform.iter().map(|step| step.to_string()).collect()
    }

    /// §3.4's direction convention: the checkpoint is HF's `[out, in]`, the slot is `[in, out]`.
    #[test]
    fn transform_maps_the_checkpoint_shape_onto_the_slot_shape() {
        assert_eq!(
            transformed_shape(&[160, 96], &steps(&["transpose(0,1)"])).unwrap(),
            vec![96, 160]
        );
        // The three-channel expert weight: [E, out, in] -> [E, in, out].
        assert_eq!(
            transformed_shape(&[32, 96, 2048], &steps(&["transpose(1,2)"])).unwrap(),
            vec![32, 2048, 96]
        );
        // `slice(dim, start, len)` keeps `len` positions from `start`: the 128 positions of axis 1
        // become the 96 the slot declares, and the other axes are untouched.
        assert_eq!(
            transformed_shape(&[8, 128, 32], &steps(&["slice(1, 0, 96)"])).unwrap(),
            vec![8, 96, 32]
        );
        assert_eq!(
            transformed_shape(&[8, 128, 32], &steps(&["slice(1, 32, 96)"])).unwrap(),
            vec![8, 96, 32]
        );
        // A negative axis counts from the end.
        assert_eq!(
            transformed_shape(&[4, 8], &steps(&["transpose(0,-1)"])).unwrap(),
            vec![8, 4]
        );
        assert_eq!(
            transformed_shape(&[4, 8], &steps(&["slice(-1, 0, 3)"])).unwrap(),
            vec![4, 3]
        );
    }

    /// The vocabulary is C6's two verbs; anything else is an error naming the step. `take` used to
    /// pass through as a rename, which is exactly the silent acceptance C6 removed.
    #[test]
    fn a_transform_outside_the_vocabulary_is_an_error_not_a_guess() {
        for step in [
            "take(other.name)",
            "concat(0)",
            "split(1, [96, 64])",
            "flip(0)",
        ] {
            let error = transformed_shape(&[96, 160], &steps(&[step])).unwrap_err();
            assert!(error.contains(step), "{error}");
        }
        // A verb whose arguments do not fit *this* shape is reported too: a slice may not leave
        // the axis, and an axis has to exist.
        let error = transformed_shape(&[96, 160], &steps(&["slice(1, 96, 96)"])).unwrap_err();
        assert!(error.contains("slice(1, 96, 96)"), "{error}");
        assert!(transformed_shape(&[96, 160], &steps(&["transpose(0,9)"])).is_err());
        assert!(transformed_shape(&[96, 160], &steps(&["transpose(0)"])).is_err());
        assert!(transformed_shape(&[96, 160], &steps(&["transpose(0,1"])).is_err());
    }

    /// §3.5: the intervals a `slice`/`split` cuts have to stay inside the axis — the segments must
    /// add up to it.
    #[test]
    fn a_split_must_add_up_to_the_axis_it_splits() {
        let split = rustrain_model::ResolvedSplit {
            dim: 1,
            sizes: vec![64, 64],
        };
        assert_eq!(
            split_shape(&[8, 128, 32], &split, 0).unwrap(),
            vec![8, 64, 32]
        );
        assert_eq!(
            split_shape(&[8, 128, 32], &split, 1).unwrap(),
            vec![8, 64, 32]
        );
        let short = rustrain_model::ResolvedSplit {
            dim: 1,
            sizes: vec![64, 32],
        };
        let error = split_shape(&[8, 128, 32], &short, 0).unwrap_err();
        assert!(error.contains("sum to 96"), "{error}");
        let out_of_range = rustrain_model::ResolvedSplit {
            dim: 3,
            sizes: vec![64, 64],
        };
        assert!(split_shape(&[8, 128, 32], &out_of_range, 0).is_err());
    }

    /// §3.6 #8: a wrong description is reported, never panicked on. Two sizes that each fit in an
    /// `i64` but do not fit together used to abort the process with an arithmetic overflow.
    #[test]
    fn split_sizes_that_overflow_i64_are_an_error() {
        let split = rustrain_model::ResolvedSplit {
            dim: 0,
            sizes: vec![i64::MAX, i64::MAX],
        };
        let error = split_shape(&[4, 4], &split, 0).unwrap_err();
        assert!(error.contains("overflow"), "{error}");
        assert!(error.contains(&i64::MAX.to_string()), "{error}");
        assert!(error.contains("dim 0"), "{error}");
    }

    #[test]
    fn axis_counts_from_the_end_for_negative_dimensions() {
        assert_eq!(axis(-1, 3), Some(2));
        assert_eq!(axis(-4, 3), None);
        assert_eq!(axis(3, 3), None);
        assert_eq!(axis(0, 0), None);
    }

    /// C6's verdict vocabulary, and C2's exit rule: only `fail` stops the run.
    #[test]
    fn only_a_fail_makes_the_report_fail() {
        let item = |status: Verdict| {
            CheckItem::new("l1.structure", status, "reason".to_string(), Vec::new())
        };
        let report = |status: Verdict| CheckReport {
            model: "m".to_string(),
            checkpoint: None,
            dtype: "f32".to_string(),
            counts: Counts::default(),
            checks: vec![item(status)],
        };
        assert!(report(Verdict::Fail).failed());
        assert!(!report(Verdict::Pass).failed());
        assert!(!report(Verdict::Skip).failed());
    }
}
