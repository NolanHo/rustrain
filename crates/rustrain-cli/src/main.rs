//! `rustrain` — the operator-first command line.
//!
//! Two commands, both of which answer questions the old framework could not:
//!
//! * `ops list` — what implementations exist on this machine.
//! * `plan explain` — what will actually run, with which implementation, at
//!   which precision, with which communication spliced in, and how much memory
//!   it projects.
//!
//! Neither reads an environment variable, and neither needs a GPU.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};

use rustrain_abi::Plugin;
use rustrain_ops::{Phase, Recipe, Registry, TargetEnv};
use rustrain_parallel::{GroupKind, ParallelConfig, ParallelLayout};
use rustrain_plan::{Attrs, OpRef, PlanBuilder, SlotKind};

#[derive(Parser)]
#[command(name = "rustrain", about = "Operator-first training framework", version)]
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
            OpsCommand::Check { plugins, op, json } => ops_check(&plugins, op.as_deref(), json),
        },
        Command::Plan(args) => match args.command {
            PlanCommand::Explain {
                recipe,
                tp,
                plugins,
                json,
            } => plan_explain(recipe.as_deref(), tp, &plugins, json),
        },
    }
}

/// Loads the built-in provider plus any requested plugins.
///
/// Order matters: a duplicate `op@variant` is an error naming both origins, so
/// registering the built-in first means a user's plugin cannot silently shadow
/// it — it has to collide out loud.
fn load_registry(plugins: &[PathBuf]) -> Result<Registry> {
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

fn load_recipe(path: Option<&Path>) -> Result<Recipe> {
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

fn ops_check(plugins: &[PathBuf], only: Option<&str>, json: bool) -> Result<()> {
    use rustrain_runtime::conformance::{Harness, default_cases, uncovered_operators};

    let registry = load_registry(plugins)?;
    let recipe = load_recipe(None)?;
    let harness = Harness::new(&registry, &recipe);

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
                serde_json::json!(uncovered_operators()
                    .iter()
                    .map(|(op, why)| serde_json::json!({ "op": op, "reason": why }))
                    .collect::<Vec<_>>()),
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

    let mut b = PlanBuilder::new("demo-mlp", Phase::Forward, parallel);
    let x = b.slot("hidden", RsDtype::F32, vec![8, 64], SlotKind::Input);

    // Column parallel: output features (dim 1 of a [K, N] weight) split.
    let w1 = b.slot_with_layout(
        "mlp.up.weight",
        RsDtype::F32,
        vec![64, 128],
        SlotKind::Weight,
        ParallelLayout::Shard {
            dim: 1,
            group: GroupKind::Tp,
        },
    );
    // The up-projection's output stays sharded: it feeds the row-parallel
    // down-projection directly, which is precisely why the pair needs a single
    // collective rather than one per layer.
    let h1 = b.slot_with_layout(
        "mlp.up.out",
        RsDtype::F32,
        vec![8, 128],
        SlotKind::Activation,
        ParallelLayout::Shard {
            dim: -1,
            group: GroupKind::Tp,
        },
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
        ParallelLayout::Shard {
            dim: -1,
            group: GroupKind::Tp,
        },
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
        ParallelLayout::Shard {
            dim: 0,
            group: GroupKind::Tp,
        },
    );
    let out = b.slot(
        "mlp.down.out",
        RsDtype::F32,
        vec![8, 64],
        SlotKind::Output,
    );
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
    let parallel = plan.meta.parallel;

    let compiled = rustrain_plan::Compiler::new(&registry, &recipe, TargetEnv::default(), parallel)
        .compile(&plan)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if json {
        let doc = serde_json::json!({
            "name": compiled.plan.meta.name,
            "digest": compiled.digest,
            "world_size": compiled.parallel.world_size(),
            "slots": compiled.plan.slots.len(),
            "steps": compiled.steps.len(),
            "implementations": compiled.resolved.iter().map(|r| serde_json::json!({
                "node": r.node.0,
                "op": r.spec_name,
                "plugin": r.plugin,
            })).collect::<Vec<_>>(),
            "collectives": compiled.inserted.iter().map(|c| serde_json::json!({
                "op": c.op,
                "group": format!("{:?}", c.group),
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
        });
        println!("{}", serde_json::to_string_pretty(&doc)?);
        return Ok(());
    }

    print!("{}", compiled.explain());
    Ok(())
}
