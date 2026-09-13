//! `rustrain run` — spec C4 and delivery D5: one forward in one process, executed by the real
//! runtime against the reference registry, then the candidate dump the HF comparison reads.
//!
//! Precision (the frozen decision, stated here because a runner user has to know it): the
//! reference provider is f32-only while the checkpoint and HF are bf16, so `run` widens the
//! bf16 weights to f32 — exact, bf16 ⊂ f32 — and executes f32. The HF reference is dumped with
//! `--dtype bf16`; the spec's 1% tolerance (`max_abs_diff / max_abs` on the logits and the
//! per-layer summaries) absorbs HF's own bf16 rounding, not this widening.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Args;
use rustrain_abi::ffi::RsDtype;
use rustrain_parallel::{Mesh, ParallelConfig};
use rustrain_plan::{Plan, SlotId};
use rustrain_runtime::{Executor, HostAllocator, SingleRank, required_inputs};

use crate::load::load_weights;
use crate::npz::{self, Npy};

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

    /// Where to write the candidate dump (a `.npz`); a `.json` sidecar lands next to it.
    #[arg(long, value_name = "PATH")]
    pub out: PathBuf,

    /// The mesh degrees. `tp=cp=ep=dp=pp=1` runs the whole model on rank 0 — the first
    /// comparison. A sharded run loads each rank's local slice but cannot execute the collectives
    /// without a real backend (the executor refuses them); `pp > 1` is refused outright because
    /// the cross-stage seam is D5's open decision.
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

    // ---- the probe tokens -----------------------------------------------
    let tokens = probe_tokens(args.tokens.as_deref(), args.seq)?;

    // ---- description → global plan → rank 0's plan ----------------------
    let model = rustrain_model::Model::load(&args.model)
        .with_context(|| format!("loading the model description in {}", args.model.display()))?;
    let expanded = model
        .expand()
        .context("the description did not expand into a runnable plan")?;

    let mesh = Mesh::from_config(&ParallelConfig {
        tensor: args.tp,
        context: args.cp,
        expert: args.ep,
        data: args.dp,
        pipeline: args.pp,
    });
    let mut plan = rustrain_plan::instantiate(&expanded.plan, &expanded.declarations(), &mesh, 0)
        .context("instantiating rank 0 of the mesh")?;

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
    let window = input_slot.shape[0];
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
    // storage survives the run (the same thing HF's `output_hidden_states=True` does). The
    // logits slot needs none: it is already an unread output and dies with the plan.
    let outputs = model.desc.outputs.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "the description declares no `outputs` section; `run` needs `outputs.logits` and \
             `outputs.hidden` to know which slots are the logits and the per-layer hidden states \
             (D5's dump contract)"
        )
    })?;
    let hidden_ids = keep_hidden_states(&mut plan, &outputs.hidden)?;

    // ---- the weights, through the same pairing `check` verifies ---------
    let loaded = load_weights(&expanded, &model.desc, &plan, &mesh, 0, &args.checkpoint)
        .context("loading the checkpoint weights")?;
    let loaded_count = loaded.len();

    // ---- compile against the reference provider -------------------------
    let registry = crate::load_registry(&[]).context("loading the reference provider")?;
    let recipe = crate::load_recipe(None).context("loading the default recipe")?;
    let compiled =
        rustrain_plan::Compiler::new(&registry, &recipe, rustrain_ops::TargetEnv::default())
            .compile(&plan)
            .context(
                "compiling the plan (every node must resolve and its inferred shapes must agree)",
            )?;

    // ---- feed, execute, read --------------------------------------------
    let digest = compiled.digest.clone();
    let peak_bytes = compiled.memory.peak_bytes;
    let mut executor = Executor::new(
        compiled,
        Box::new(HostAllocator::new()),
        Box::new(SingleRank::new(mesh.world_size())),
    )
    .context("preparing the executor")?;

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
    // uses); the dump then keeps only those rows.
    let mut ids = vec![0i64; window as usize];
    ids[..tokens.len()].copy_from_slice(&tokens);
    let mut id_bytes = Vec::with_capacity(ids.len() * 8);
    for id in &ids {
        id_bytes.extend_from_slice(&id.to_le_bytes());
    }
    executor
        .write_raw(input_id, &id_bytes)
        .context("feeding the token stream")?;

    // Each weight's widened f32 buffer is dropped as it is copied into the executor's persistent
    // region, so the peak is the executor's f32 weights, not the executor's plus the loader's.
    let mut checkpoint_bytes: u64 = 0;
    for weight in loaded {
        checkpoint_bytes += weight.checkpoint_bytes;
        executor
            .write_f32(weight.slot, &weight.values)
            .with_context(|| format!("writing the weight slot `{}`", weight.name))?;
    }

    let started = Instant::now();
    let stats = executor.run().context("executing the forward")?;
    let wall = started.elapsed();

    // ---- the outputs the description declares ---------------------------
    let logits_id = single_match(&plan, &outputs.logits, "outputs.logits")?;
    let logits_bytes = executor
        .read_raw(logits_id)
        .with_context(|| format!("reading the logits slot `{}`", plan.slot(logits_id).name))?;
    let logits: Vec<f32> = logits_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let logits_shape = plan.slot(logits_id).shape.clone();
    if logits_shape.len() != 2 || logits_shape[0] != window {
        bail!(
            "the logits slot `{}` is {:?}; the dump needs [window, vocab] = [{window}, …]",
            plan.slot(logits_id).name,
            logits_shape
        );
    }
    let vocab = logits_shape[1] as usize;
    let probe_logits = &logits[..tokens.len() * vocab];

    let mut summaries: Vec<[f32; 3]> = Vec::new();
    let mut hidden_names: Vec<String> = Vec::new();
    for id in &hidden_ids {
        {
            let name = plan.slot(*id).name.clone();
            let bytes = executor
                .read_raw(*id)
                .with_context(|| format!("reading the hidden state slot `{name}`"))?;
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let shape = plan.slot(*id).shape.clone();
            let per_row: usize = shape[1..]
                .iter()
                .map(|d| *d as usize)
                .product::<usize>()
                .max(1);
            // Only the probe rows: the pad rows are not part of the HF tensor.
            let probe: &[f32] = &values[..tokens.len() * per_row];
            summaries.push(summarize(probe));
            hidden_names.push(name);
        }
    }
    if summaries.is_empty() {
        bail!(
            "`outputs.hidden` matched no slot of the rank-0 plan; the dump needs at least one \
             hidden state"
        );
    }

    // ---- the dump ----------------------------------------------------------
    let mut logits_le = Vec::with_capacity(probe_logits.len() * 4);
    for v in probe_logits {
        logits_le.extend_from_slice(&v.to_le_bytes());
    }
    let mut summary_le = Vec::with_capacity(summaries.len() * 3 * 4);
    for row in &summaries {
        for v in row {
            summary_le.extend_from_slice(&v.to_le_bytes());
        }
    }
    let tokens_i64: Vec<i64> = tokens.clone();
    let mut ids_le = Vec::with_capacity(tokens_i64.len() * 8);
    for id in &tokens_i64 {
        ids_le.extend_from_slice(&id.to_le_bytes());
    }

    npz::write_npz(
        &args.out,
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
                shape: &[summaries.len(), 3],
                data: &summary_le,
            },
        ],
    )
    .context("writing the candidate dump")?;

    // ---- the sidecar and the human report --------------------------------
    let sidecar_path = args.out.with_extension(format!(
        "{}json",
        args.out
            .extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default()
    ));
    let sidecar = serde_json::json!({
        "format": "rustrain.run.v1",
        "model": args.model.display().to_string(),
        "checkpoint": args.checkpoint.display().to_string(),
        "digest": digest,
        "world_size": mesh.world_size(),
        "degrees": {"tp": args.tp, "cp": args.cp, "ep": args.ep, "dp": args.dp, "pp": args.pp},
        "window": window,
        "probe_tokens": tokens,
        "logits_shape": [tokens.len(), vocab],
        "hidden_states": summaries.len(),
        "hidden_slots": hidden_names,
        "weights": loaded_count,
        "checkpoint_bytes": checkpoint_bytes,
        "steps": stats.steps,
        "ops": stats.ops,
        "collectives": stats.collectives,
        "peak_bytes": peak_bytes,
        "wall_seconds": wall.as_secs_f64(),
        "precision": "weights widened bf16 -> f32 (exact; bf16 is a subset of f32), forward in f32",
    });
    std::fs::write(
        &sidecar_path,
        serde_json::to_string_pretty(&sidecar)? + "\n",
    )
    .with_context(|| format!("writing the sidecar {}", sidecar_path.display()))?;

    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    println!(
        "run {}  rank 0 of tp={} cp={} ep={} dp={} pp={} (world {})",
        expanded.plan.meta.name,
        args.tp,
        args.cp,
        args.ep,
        args.dp,
        args.pp,
        mesh.world_size()
    );
    println!("  digest {}", &digest[..digest.len().min(12)]);
    println!(
        "  weights {} slot(s)  {:.1} GiB checkpoint bytes -> f32 (bf16 ⊂ f32, widening exact)",
        loaded_count,
        gib(checkpoint_bytes)
    );
    println!(
        "  forward {} step(s) ({} ops, {} collectives)  wall {:.3} s  peak {:.1} GiB",
        stats.steps,
        stats.ops,
        stats.collectives,
        wall.as_secs_f64(),
        gib(peak_bytes)
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
        hidden_ids.len()
    );
    println!(
        "  logits [{}, {}]  hidden summaries [{}, 3]",
        tokens.len(),
        vocab,
        summaries.len()
    );
    println!("  wrote {}", args.out.display());
    println!("  wrote {}", sidecar_path.display());
    println!(
        "  per-layer summary (mean, std, max over the {} probe position(s)):",
        tokens.len()
    );
    for (index, (name, row)) in hidden_names.iter().zip(&summaries).enumerate() {
        println!(
            "    layer {index:3}  mean {:+.6e}  std {:.6e}  max {:.6e}  ({name})",
            row[0], row[1], row[2]
        );
    }
    Ok(())
}

/// The probe tokens: `--tokens` verbatim, `--seq` as `0..n-1`, and the two must agree when both
/// are given.
fn probe_tokens(tokens: Option<&str>, seq: Option<usize>) -> Result<Vec<i64>> {
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
/// Returns the hidden slot ids in pattern order.
fn keep_hidden_states(plan: &mut Plan, patterns: &[String]) -> Result<Vec<SlotId>> {
    let mut ids: Vec<SlotId> = Vec::new();
    for pattern in patterns {
        let matched = matching_slots(plan, pattern);
        if matched.is_empty() {
            bail!("`outputs.hidden` pattern `{pattern}` matches no slot of the rank-0 plan");
        }
        ids.extend(matched);
    }
    let phase = plan.meta.phase;
    for (index, id) in ids.iter().enumerate() {
        let slot = plan.slot(*id);
        let mut out = slot.clone();
        out.name = format!("__run__.hidden.{index}.out");
        // The hidden slot may carry a partial at this point (the all-reduce that completes it is
        // inserted by the compiler, which rewires consumers to the converted slot). The view's
        // output is what the dump reads — the complete tensor — so it declares the completed
        // layout: replicate where the input is partial, the input's own layout otherwise.
        if slot.layout.partial.is_some() {
            out.layout = rustrain_parallel::ParallelLayout::replicate();
        }
        if plan.slot_id(&out.name).is_some() {
            bail!(
                "the hidden-state keeper collides with existing slot `{}`",
                out.name
            );
        }
        out.kind = rustrain_plan::SlotKind::Output;
        let out_id = SlotId(plan.slots.len());
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
    Ok(ids)
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
        let mut plan =
            rustrain_plan::instantiate(&expanded.plan, &expanded.declarations(), &mesh, 0)
                .expect("instantiate rank 0");
        widen_to_f32(&mut plan);

        let outputs = model.desc.outputs.as_ref().expect("outputs declared");
        let logits_before = matching_slots(&plan, &outputs.logits);
        assert_eq!(logits_before.len(), 1, "the logits pattern names one slot");
        let hidden = keep_hidden_states(&mut plan, &outputs.hidden).expect("keeper");
        // HF's output_hidden_states = embedding output + 40 layer outputs + final norm = 42 rows.
        assert_eq!(hidden.len(), 42, "embed + layers.0..39 + norm");
        assert_eq!(
            plan.slot(hidden[0]).name,
            "embed.y",
            "the first hidden state is the embedding output"
        );
        assert_eq!(
            plan.slot(hidden[41]).name,
            "norm.y",
            "the last hidden state is the final norm"
        );

        let nodes_after = plan.nodes.len();
        let registry = crate::load_registry(&[]).expect("registry");
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
        // collectives at world size 1; the exact count is the check gate's 53.
        assert_eq!(compiled.inserted.len(), 53);
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
}
