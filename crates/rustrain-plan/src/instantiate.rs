//! `instantiate`: the global plan × declarations × mesh → this rank's concrete plan.
//!
//! The global plan a description expands to is topology-free: every layout replicated, every
//! shape concrete, sharding recorded only as symbolic axis names on `binding`s. `instantiate`
//! turns it into the plan one rank executes (`docs/design/model-description.md` §4.2):
//!
//! 1. **PP**: only the instances whose declared stage equals the rank's `pp` coordinate are
//!    kept; a slot written on another stage becomes a plan input, a slot read only by another
//!    stage becomes a plan output. PP is the only axis that changes the node set (§6.1).
//! 2. **Layouts**: a declared slot gets one [`ShardSpec`] per `(dim, axis)` it names — a dim
//!    declared with several axes gets them as separate specs, which is what makes `ep × tp`
//!    expressible. Every other slot's layout is derived from those by the same rule table
//!    [`crate::shard::propagate`] walks with ([`crate::shard::ShardRules`] +
//!    [`crate::shard::derive`]); there is no second rule table.
//! 3. **Shapes**: `ParallelLayout::local_shape` on every kept slot. Non-divisibility is the
//!    hard [`rustrain_parallel::ShardError::NotDivisible`], reported per slot so the caller
//!    can name the constraint — `tp = 3` against 16 heads fails here, before any
//!    implementation is resolved.
//! 4. Position constants: rank-dependent compile-time constants (flat-QKV channel offsets,
//!    CP sequence offsets, local expert ranges) baked into node attributes.
//!    // Deferred to D5.
//! 5. **Group availability**: every mask is validated against the mesh. A declared axis name
//!    the mesh does not have is reported before any mask is formed
//!    ([`PlanError::UnknownAxis`], the name-level counterpart of [`PlanError::GroupUnavailable`]).
//!
//! `instantiate` never touches the operator registry: it runs on the real bf16 description
//! where five primitives have no provider, and a divisibility error is a hard failure even
//! when resolution is incomplete.

use std::collections::BTreeMap;

use rustrain_parallel::{GroupMask, Mesh, ParallelLayout, ShardSpec, ShardMode};

use crate::PlanError;
use crate::ir::{NodeId, Plan, PlanNode, Slot, SlotId, SlotKind};
use crate::shard::{ShardRules, canonicalize, derive};

/// One declared axis: the mesh axis a slot dim is sharded by, and how its slabs relate to the
/// axis (disjoint or overlapping). The description spells this with its own serde type
/// (`rustrain_model::AxisDecl`); `Expanded::declarations()` converts, so the two crates keep their
/// own vocabularies and adding a mode is one compile error, not a silent default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredAxis {
    pub axis: String,
    pub mode: ShardMode,
}

impl DeclaredAxis {
    /// A strict shard: `global` must divide by the group's degree.
    pub fn divide(axis: impl Into<String>) -> Self {
        Self {
            axis: axis.into(),
            mode: ShardMode::Divide,
        }
    }

    /// A shard in units of `unit` elements whose slabs may overlap when a rank cannot be given a
    /// whole unit per rank.
    pub fn replicate(axis: impl Into<String>, unit: i64) -> Self {
        Self {
            axis: axis.into(),
            mode: ShardMode::Replicate { unit },
        }
    }
}

/// The description's declarations, carried from `expand` to [`instantiate`] as **input**.
///
/// The plan stays portable: it records neither axis names nor stages, so its digest does not
/// change when a description declares a sharding (`docs/design/model-description.md` §0 keeps
/// the "description × topology → plan" operands visible). Both fields are written by
/// `rustrain-model`'s `Expanded::declarations()` and read by [`instantiate`] — nothing else.
pub struct DeclaredAxes {
    /// Slot **name** → (logical dim as written in the description → the axes declared on it, in
    /// declaration order). Names, not ids: a description's `targets[].slot` refers to a name.
    pub slots: BTreeMap<String, BTreeMap<String, Vec<DeclaredAxis>>>,
    /// Every expanded stack instance in stack order, with the stage its stack entry declared
    /// (`None` = the entry has no `stage`). PP pruning indexes by this list.
    pub instances: Vec<InstanceStage>,
}

/// One expanded stack instance: its prefix and the stage its entry declared.
pub struct InstanceStage {
    /// The instance prefix, after `{l}` / `{last}` substitution — the dotted prefix of every
    /// slot the instance owns.
    pub prefix: String,
    /// `None` = the entry declared no `stage`. Legal only while the mesh's `pp` degree is 1:
    /// with `pp > 1` an absent stage is a reported error, never stage 0 by silence (R1).
    pub stage: Option<i64>,
}

/// The global plan × declarations × mesh → the plan `rank` executes.
///
/// # Errors
///
/// - [`PlanError::RankOutOfRange`] if `rank` is outside the mesh;
/// - [`PlanError::MissingStage`] / [`PlanError::StageOutOfRange`] when `pp > 1` and an
///   instance declares no stage or a stage the pipeline degree cannot address;
/// - [`PlanError::UnknownDeclaredSlot`] / [`PlanError::BadDeclaredDim`] /
///   [`PlanError::UnknownAxis`] for a broken declaration;
/// - [`PlanError::ShardRuleUndeclared`] when a node's operator declares no usable rule;
/// - [`PlanError::ShardDerivation`] when the rule cannot derive a node's distribution;
/// - [`PlanError::Instantiate`] when a layout does not divide into a local shape.
///
/// `rules` is where the operators' declared shard rules come from: the registry
/// in production, a [`crate::shard::RuleTable`] for synthetic plans. The
/// declaration is data — this function never matches on an operator name.
pub fn instantiate(
    plan: &Plan,
    declared: &DeclaredAxes,
    mesh: &Mesh,
    rank: usize,
    rules: &dyn ShardRules,
) -> Result<Plan, PlanError> {
    let world_size = mesh.world_size();
    if rank >= world_size {
        return Err(PlanError::RankOutOfRange { rank, world_size });
    }

    // The rank's pipeline coordinate. `pp` is a mesh axis *name*, not a fixed axis: a mesh
    // without a `pp` axis is a degree-1 pipeline, i.e. one stage and nothing to prune.
    let pp_axis = mesh.index_of("pp");
    let pp_degree = pp_axis.and_then(|axis| mesh.degree(axis)).unwrap_or(1);
    let pp_coord = match pp_axis {
        Some(axis) => {
            let stride = mesh.stride(axis).expect("index_of found the axis");
            (rank / stride) % pp_degree
        }
        None => 0,
    };

    // Declared stages. With `pp > 1` every instance must declare one (R1: an absent stage
    // means stage 0 only when `pp == 1`). With `pp == 1` there is exactly one stage, nothing
    // is pruned, and a declared stage is ignored rather than validated against a degree it
    // cannot exceed.
    let mut stages: BTreeMap<String, i64> = BTreeMap::new();
    if pp_degree > 1 {
        for instance in &declared.instances {
            let stage = match instance.stage {
                Some(stage) => stage,
                None => {
                    return Err(PlanError::MissingStage {
                        prefix: instance.prefix.clone(),
                        pp: pp_degree,
                    });
                }
            };
            if stage < 0 || stage as usize >= pp_degree {
                return Err(PlanError::StageOutOfRange {
                    prefix: instance.prefix.clone(),
                    stage,
                    pp: pp_degree,
                });
            }
            stages.insert(instance.prefix.clone(), stage);
        }
    }

    let mut out = plan.clone();

    // ---- layouts: declared slots, then the rule-table walk -----------------

    for (slot_name, dims) in &declared.slots {
        let id = out
            .slot_id(slot_name)
            .ok_or_else(|| PlanError::UnknownDeclaredSlot {
                slot: slot_name.clone(),
            })?;
        let mut specs: Vec<ShardSpec> = Vec::new();
        for (dim_text, axes) in dims {
            let dim: i64 = dim_text.parse().map_err(|_| PlanError::BadDeclaredDim {
                slot: slot_name.clone(),
                dim: dim_text.clone(),
            })?;
            for axis in axes {
                let index = mesh.index_of(&axis.axis).ok_or_else(|| PlanError::UnknownAxis {
                    slot: slot_name.clone(),
                    dim: dim_text.clone(),
                    axis: axis.axis.clone(),
                    axes: axis_names(mesh),
                })?;
                // A mesh has at most `Mesh::MAX_AXES` axes, so `single` cannot overflow; an
                // impossible overflow is still reported, not unwrapped.
                let group =
                    GroupMask::single(index).map_err(|source| PlanError::Mesh { source })?;
                // The description's mode travels into the layout, which is the only thing the
                // propagation walk, the loader and the display all read.
                specs.push(ShardSpec {
                    dim,
                    group,
                    mode: axis.mode,
                });
            }
        }
        // One spelling for every distribution the plan carries: each declared dim is resolved
        // against the slot's own rank, so the walk below, the D3 materialization check and the
        // transition table all compare the same axes (`shard(0, g)` on a rank-1 tensor and
        // `shard(-1, g)` on a rank-2 one are different axes, and mixing the spellings is how a
        // declared weight and the rule reading it used to disagree about nothing).
        let rank = out.slot(id).shape.len() as i64;
        out.slot_mut(id).layout = canonicalize(
            &ParallelLayout {
                dims: specs,
                partial: None,
            },
            rank,
        );
    }

    // Every other slot's layout is derived from the declared ones, walking the nodes in
    // emission order (the global plan is topological). `effective` mirrors the walk inside
    // `shard::propagate` minus the insertion: turning one layout into another is `compile`'s
    // job, and the layout a slot carries here is what its producer writes. Consumers that
    // need a different layout are reconciled — with the collectives — by the propagation
    // pass when the instantiated plan compiles. Everything stored is canonical (dims resolved
    // against the slot's rank), the same spelling `propagate` compares with.
    let mut effective: Vec<ParallelLayout> = out
        .slots
        .iter()
        .map(|s| canonicalize(&s.layout, s.shape.len() as i64))
        .collect();
    for (i, node) in plan.nodes.iter().enumerate() {
        let rule = rules
            .rule(&node.op.name)
            .map_err(|reason| PlanError::ShardRuleUndeclared {
                node: NodeId(i),
                op: node.op.name.clone(),
                reason,
            })?;
        let eff_in: Vec<ParallelLayout> =
            node.inputs.iter().map(|s| effective[s.0].clone()).collect();
        let input_ranks: Vec<i64> = node
            .inputs
            .iter()
            .map(|s| plan.slot(*s).shape.len() as i64)
            .collect();
        let declared_out: Vec<ParallelLayout> = node
            .outputs
            .iter()
            .map(|s| out.slot(*s).layout.clone())
            .collect();
        let output_ranks: Vec<i64> = node
            .outputs
            .iter()
            .map(|s| plan.slot(*s).shape.len() as i64)
            .collect();
        let derived = derive(
            rule,
            &node.op.name,
            &eff_in,
            &declared_out,
            &input_ranks,
            &output_ranks,
        )
        .map_err(|source| PlanError::ShardDerivation {
            node: NodeId(i),
            source,
        })?;
        for (j, o) in node.outputs.iter().enumerate() {
            let produced = derived
                .outputs
                .get(j)
                .cloned()
                .unwrap_or_else(ParallelLayout::replicate);
            effective[o.0] = produced.clone();
            out.slot_mut(*o).layout = produced;
        }
    }

    // ---- PP: keep this rank's stage -----------------------------------------

    if pp_degree > 1 {
        prune_pipeline_stage(&mut out, plan, &stages, pp_degree, pp_coord)?;
    }

    // ---- local shapes --------------------------------------------------------

    for slot in &mut out.slots {
        slot.shape = slot
            .layout
            .local_shape(&slot.shape, mesh)
            .map_err(|source| PlanError::Instantiate {
                slot: slot.name.clone(),
                source,
            })?;
    }

    // The instantiated plan is topology-*dependent*: it carries the mesh it was built for as
    // its fingerprint, the same way a compiled plan does (design §1.3, invariant I-6).
    out.meta.mesh = mesh.fingerprint();
    out.check_structure()?;
    Ok(out)
}

/// Keeps the nodes whose instance stage is `pp_coord` and rewrites boundary slots: a slot
/// written on another stage becomes a plan input, a slot read only by another stage becomes
/// a plan output.
fn prune_pipeline_stage(
    out: &mut Plan,
    plan: &Plan,
    stages: &BTreeMap<String, i64>,
    pp_degree: usize,
    pp_coord: usize,
) -> Result<(), PlanError> {
    let node_stage = |node: &PlanNode| -> Result<i64, PlanError> {
        // All of a node's outputs live in its instance, so the first output names the
        // instance. A name no instance prefix covers cannot be staged: with `pp > 1` that is
        // the same "declares no stage" refusal as an entry without `stage`.
        let name = &out.slots[node.outputs[0].0].name;
        instance_stage(stages, name).ok_or_else(|| PlanError::MissingStage {
            prefix: name.clone(),
            pp: pp_degree,
        })
    };

    let mut stage_of_node = Vec::with_capacity(plan.nodes.len());
    for node in &plan.nodes {
        stage_of_node.push(node_stage(node)?);
    }

    // Which stages read each slot, over the *global* graph.
    let mut readers: Vec<Vec<i64>> = vec![Vec::new(); plan.slots.len()];
    for (node, stage) in plan.nodes.iter().zip(&stage_of_node) {
        for input in &node.inputs {
            readers[input.0].push(*stage);
        }
    }
    let producers = plan.producers();

    let mut keep: Vec<bool> = vec![false; plan.slots.len()];
    let mut kinds: Vec<SlotKind> = plan.slots.iter().map(|s| s.kind).collect();
    for i in 0..plan.slots.len() {
        let read_here = readers[i].contains(&(pp_coord as i64));
        let read_elsewhere = readers[i].iter().any(|&s| s != pp_coord as i64);
        match producers[i] {
            // No producer: a plan input (weight or description input) is kept exactly when
            // this stage reads it.
            None => keep[i] = read_here,
            Some(node) => {
                if stage_of_node[node.0] == pp_coord as i64 {
                    // Produced on this stage: kept for its producer; an output when only
                    // other stages read it.
                    keep[i] = true;
                    if !read_here && read_elsewhere {
                        kinds[i] = SlotKind::Output;
                    }
                } else {
                    // Produced on another stage: this stage's copy is a plan input.
                    keep[i] = read_here;
                    if read_here {
                        kinds[i] = SlotKind::Input;
                    }
                }
            }
        }
    }

    let mut slot_map: Vec<Option<SlotId>> = vec![None; plan.slots.len()];
    let mut slots: Vec<Slot> = Vec::new();
    for (i, slot) in out.slots.iter().enumerate() {
        if !keep[i] {
            continue;
        }
        slot_map[i] = Some(SlotId(slots.len()));
        let mut slot = slot.clone();
        slot.kind = kinds[i];
        slots.push(slot);
    }

    let mut nodes = Vec::new();
    for (i, node) in plan.nodes.iter().enumerate() {
        if stage_of_node[i] != pp_coord as i64 {
            continue;
        }
        let mut node = node.clone();
        for input in &mut node.inputs {
            *input = slot_map[input.0].expect("a kept node's inputs are kept");
        }
        for output in &mut node.outputs {
            *output = slot_map[output.0].expect("a kept node's outputs are kept");
        }
        nodes.push(node);
    }

    out.slots = slots;
    out.nodes = nodes;
    Ok(())
}

/// The stage of the instance `name` belongs to: the longest declared prefix that is a dotted
/// prefix of it. Instance prefixes are unique (§3.6 #7), and a slot lives in the instance its
/// name starts with (`embed.y` in `embed`, `mtp.layers.0.x` in `mtp.layers.0` — never the
/// shorter `mtp`), so the longest match is the owner.
fn instance_stage(stages: &BTreeMap<String, i64>, name: &str) -> Option<i64> {
    stages
        .iter()
        .filter(|(prefix, _)| {
            let dotted = [prefix.as_str(), "."].concat();
            name == prefix.as_str() || name.starts_with(&dotted)
        })
        .max_by_key(|(prefix, _)| prefix.len())
        .map(|(_, stage)| *stage)
}

fn axis_names(mesh: &Mesh) -> String {
    mesh.axes()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// One PP stage's instantiation outcome: the representative rank (the rank whose pp
/// coordinate is `stage`) and the plan it executes, or why it failed.
pub struct StageInstantiation {
    pub stage: usize,
    pub rank: usize,
    pub result: Result<Plan, PlanError>,
}

/// Instantiates one representative rank per PP stage — **every** stage, so a divisibility
/// failure or an empty stage on a stage other than 0 is not masked by a clean stage 0
/// (reviewer findings C3/C4). With `pp = 1` (or a mesh without a `pp` axis) that is one
/// stage, exactly what `instantiate(&plan, .., 0)` was.
///
/// The representative rank is the minimal rank with that pp coordinate — `stage * stride(pp)`
/// with every other coordinate 0 — which is the same arithmetic `instantiate` uses internally
/// to read a rank's pp coordinate.
pub fn instantiate_stages(
    plan: &Plan,
    declared: &DeclaredAxes,
    mesh: &Mesh,
    rules: &dyn ShardRules,
) -> Vec<StageInstantiation> {
    let pp_axis = mesh.index_of("pp");
    let pp_degree = pp_axis.and_then(|axis| mesh.degree(axis)).unwrap_or(1);
    let stride = pp_axis.and_then(|axis| mesh.stride(axis)).unwrap_or(1);
    (0..pp_degree)
        .map(|stage| {
            let rank = stage.saturating_mul(stride);
            StageInstantiation {
                stage,
                rank,
                result: instantiate(plan, declared, mesh, rank, rules),
            }
        })
        .collect()
}
