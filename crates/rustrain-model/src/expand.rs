//! `expand`: description + `config.json` → the global `Plan` (§4.1).
//!
//! The five steps follow the contract: evaluate `params` → walk `stack` in order instantiating
//! templates (allocate slots, emit nodes) → evaluate shapes (fully concrete) → `check_structure()`
//! → attach the symbolic declarations of `binding`.
//!
//! Every `layout` in the global plan is replicated: axes and sharding wait for `instantiate` to
//! see a mesh, so `expand` only records them ([`ResolvedBinding`]) and never touches a shape.

use std::collections::{BTreeMap, BTreeSet};

use rustrain_abi::ffi::RsDtype;
use rustrain_parallel::{Mesh, MeshFingerprint, ParallelConfig};
use rustrain_plan::{
    AttrValue, Attrs, DeclaredAxis, OpRef, Phase, Plan, PlanBuilder, SlotId, SlotKind,
};

use crate::ModelError;
use crate::desc::{
    AttrLiteral, AxisDecl, AxisMode, FORMAT, ModelDesc, NodeDecl, StackEntry, StageDecl, Target,
    Template,
};
use crate::params::{Params, Value};
use crate::pattern::is_wildcard_segment;
use crate::transform::parse_transform;

/// The index variable `until` expansion uses (the counterpart of `repeat.index`).
const UNTIL_INDEX: &str = "l";
/// The `{last}` placeholder: the last index of the most recent repeat/until expansion.
const LAST_INDEX: &str = "last";

/// The mesh fingerprint every global plan is built with.
///
/// `expand` never sees a real mesh — axes and sharding are declared on `binding`s and applied by
/// `instantiate` — so the default (all degree-1) config's fingerprint stands in. The two builder
/// sites share this one helper so the fact stays visibly the same.
fn global_mesh() -> MeshFingerprint {
    Mesh::from_config(&ParallelConfig::default()).fingerprint()
}

/// What `expand` produces.
#[derive(Debug)]
pub struct Expanded {
    /// The global plan: concrete shapes, every `layout` replicated, nodes in emission order.
    pub plan: Plan,
    /// The slots every `binding` hit (input of the L2 load check,
    /// `docs/design/model-description.md` §3.5).
    pub bindings: Vec<ResolvedBinding>,
    /// Weight slots no `binding` hit, in plan order (§3.5's first mandate). [`expand`] rejects a
    /// non-empty list; [`expand_lenient`] hands it back instead, because naming the unbound slot is
    /// precisely what a loading check has to report.
    pub unbound_slots: Vec<String>,
    /// Every expanded stack instance in stack order: `(prefix, declared stage)`, `None` for an
    /// entry without `stage`. Read by [`Expanded::declarations`]; not read anywhere else.
    stages: Vec<(String, Option<i64>)>,
    /// The `axes` declared on the top-level `inputs` (D6): an input slot can
    /// itself be distributed — cp / dp shard the token stream. Merged into
    /// [`Expanded::declarations`] next to the binding axes.
    input_axes: BTreeMap<String, BTreeMap<String, Vec<AxisDecl>>>,
}

/// The description's axis declarations, in the layout vocabulary `instantiate` reads.
///
/// The two spellings of an axis (`"tp"` and `{"axis": "tp", "mode": "replicate"}`) collapse here;
/// a third mode would fail to compile in this match rather than default to a silent divide.
/// Input axes are activations: they have no unit, and the object form is refused with a reason
/// rather than quietly sharded by single elements.
fn declared_axes_plain(
    axes: &BTreeMap<String, Vec<AxisDecl>>,
) -> BTreeMap<String, Vec<DeclaredAxis>> {
    // Input axes are activations: they have no unit (a declared one is refused where the port is
    // read), so they resolve to unit one and go through the very same mapping as a binding's.
    let resolved: BTreeMap<String, Vec<ResolvedAxis>> = axes
        .iter()
        .map(|(dim, list)| {
            (
                dim.clone(),
                list.iter()
                    .map(|axis| ResolvedAxis {
                        axis: axis.axis().to_string(),
                        mode: axis.mode(),
                        unit: 1,
                    })
                    .collect(),
            )
        })
        .collect();
    declared_axes(&resolved)
}

fn declared_axes(
    axes: &BTreeMap<String, Vec<ResolvedAxis>>,
) -> BTreeMap<String, Vec<DeclaredAxis>> {
    axes.iter()
        .map(|(dim, axes)| {
            (
                dim.clone(),
                axes.iter()
                    .map(|axis| DeclaredAxis {
                        axis: axis.axis.clone(),
                        mode: match axis.mode {
                            AxisMode::Divide => rustrain_plan::ShardMode::Divide,
                            AxisMode::Replicate => {
                                rustrain_plan::ShardMode::Replicate { unit: axis.unit }
                            }
                        },
                    })
                    .collect(),
            )
        })
        .collect()
}

impl Expanded {
    /// The declarations `instantiate` consumes: the `binding` axes by slot name, and the stage
    /// of every expanded stack instance.
    ///
    /// They travel as **input** to `rustrain_plan::instantiate`, never inside the plan: the plan
    /// stays portable and its digest does not change when a description declares a sharding
    /// (`docs/design/model-description.md` §0, D4 rulings R2).
    pub fn declarations(&self) -> rustrain_plan::DeclaredAxes {
        let mut slots: BTreeMap<String, BTreeMap<String, Vec<DeclaredAxis>>> = BTreeMap::new();
        for binding in &self.bindings {
            for hit in &binding.slots {
                if hit.axes.is_empty() {
                    // Absent axes = replicated: absent from the map, never an empty entry.
                    continue;
                }
                slots.insert(hit.slot.clone(), declared_axes(&hit.axes));
            }
        }
        // Input axes (D6): a top-level input declares its own distribution. A
        // name cannot collide with a binding target (bindings hit weight slots
        // only, `expand` enforces it), so a second insert would be a
        // description error that cannot occur — the bindings' entry is kept.
        for (name, axes) in &self.input_axes {
            if !axes.is_empty() {
                let declared = declared_axes_plain(axes);
                slots.entry(name.clone()).or_insert(declared);
            }
        }
        rustrain_plan::DeclaredAxes {
            slots,
            instances: self
                .stages
                .iter()
                .map(|(prefix, stage)| rustrain_plan::InstanceStage {
                    prefix: prefix.clone(),
                    stage: *stage,
                })
                .collect(),
        }
    }
}

/// One axis declaration after resolution: its unit resolved against the parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAxis {
    pub axis: String,
    pub mode: AxisMode,
    /// The granularity the axis shards at, in elements (1 unless declared otherwise).
    pub unit: i64,
}

/// One `binding` after resolution.
#[derive(Debug)]
pub struct ResolvedBinding {
    /// The checkpoint-side tensor name pattern.
    pub source: String,
    /// Transforms to apply before use, kept verbatim (`expand` never runs them).
    pub transform: Vec<String>,
    /// The split of a fused storage; `None` means this binding feeds one slot directly.
    pub split: Option<ResolvedSplit>,
    /// The slots this binding feeds, in declaration order.
    pub slots: Vec<ResolvedBindingSlot>,
}

/// One resolved split declaration.
#[derive(Debug)]
pub struct ResolvedSplit {
    pub dim: i64,
    pub sizes: Vec<i64>,
}

/// A slot hit by a `binding`.
#[derive(Debug)]
pub struct ResolvedBindingSlot {
    pub slot: String,
    /// The `slot` / `targets[].slot` pattern this slot was resolved from. It is what pairs a
    /// concrete checkpoint tensor with a concrete slot: substitute the tensor's captures and the
    /// result is the slot name (§3.4's shared capture). Without it a loader could only zip source
    /// instances against `slots` by position, which C5 forbids.
    pub pattern: String,
    /// slot dimension (decimal string) → symbolic axis names.
    pub axes: BTreeMap<String, Vec<ResolvedAxis>>,
}

/// Description → global plan, applying every §3.5 mandate `expand` can check.
pub fn expand(desc: &ModelDesc, config: &serde_json::Value) -> Result<Expanded, ModelError> {
    let expanded = expand_lenient(desc, config)?;
    if !expanded.unbound_slots.is_empty() {
        return Err(ModelError::Invalid(format!(
            "{} weight slot(s) are not bound to any checkpoint tensor: {}",
            expanded.unbound_slots.len(),
            summarize(&expanded.unbound_slots)
        )));
    }
    Ok(expanded)
}

/// [`expand`] without §3.5's unbound-slot mandate: a description whose weight slots are not all
/// bound still expands, and [`Expanded::unbound_slots`] says which are missing.
///
/// The mandate itself is unchanged — it lives in [`expand`], and in the L2 gate that reads
/// `unbound_slots` (`rustrain check`'s `l2.binding_coverage`). §3.7 #4 keeps both out of the
/// pattern layer.
pub fn expand_lenient(
    desc: &ModelDesc,
    config: &serde_json::Value,
) -> Result<Expanded, ModelError> {
    if desc.format != FORMAT {
        return Err(ModelError::Format {
            found: desc.format.clone(),
            expected: FORMAT.to_string(),
        });
    }

    let params = Params::resolve(&desc.params, config)?;
    check_template_slots(desc)?;
    check_ignore_patterns(desc)?;
    let default_dtype = match &desc.dtype {
        Some(name) => parse_dtype(name)?,
        None => RsDtype::F32,
    };

    let mut expander = Expander::new(desc, params, default_dtype);
    expander.expand_inputs()?;
    expander.expand_stack()?;

    let builder = std::mem::replace(
        &mut expander.builder,
        PlanBuilder::new("", Phase::Forward, global_mesh()),
    );
    let plan = builder.build()?;
    let (bindings, unbound_slots) = expander.bind()?;
    let stages = expander.stages;
    let input_axes = expander.input_axes;
    check_outputs(desc, &plan)?;

    Ok(Expanded {
        plan,
        bindings,
        unbound_slots,
        stages,
        input_axes,
    })
}

/// Validates `outputs` (D5's runner contract) against the expanded plan: every pattern uses the
/// binding-target syntax (single-segment `*` only), and every pattern must name at least one slot
/// of the plan — a declaration that matches nothing is a typo, and I-5 says a wrong description is
/// a hard error at expand time, not a silent empty report at run time.
fn check_outputs(desc: &ModelDesc, plan: &Plan) -> Result<(), ModelError> {
    let Some(outputs) = &desc.outputs else {
        return Ok(());
    };
    let mut patterns: Vec<(String, &str)> = Vec::new();
    patterns.push((outputs.logits.clone(), "outputs.logits"));
    for pattern in &outputs.hidden {
        patterns.push((pattern.clone(), "outputs.hidden"));
    }
    for (pattern, what) in &patterns {
        check_output_pattern(pattern, what)?;
        let matched = plan
            .slots
            .iter()
            .any(|slot| crate::pattern::matches(pattern, &slot.name));
        if !matched {
            return Err(ModelError::Invalid(format!(
                "`{what}` pattern `{pattern}` matches none of the plan's {} slot(s); the \
                 declaration must name the slots the runner reports",
                plan.slots.len()
            )));
        }
    }
    Ok(())
}

/// One `outputs` pattern: a non-empty dotted name with single-segment wildcards only.
fn check_output_pattern(pattern: &str, what: &str) -> Result<(), ModelError> {
    if pattern.is_empty() {
        return Err(ModelError::Invalid(format!(
            "`{what}` is empty; a pattern must name at least one segment"
        )));
    }
    for segment in pattern.split('.') {
        if segment.is_empty() {
            return Err(ModelError::Invalid(format!(
                "`{what}` pattern `{pattern}` has an empty segment: a `.` with nothing on one \
                 side of it matches no slot name"
            )));
        }
        if segment == "**" {
            return Err(ModelError::Invalid(format!(
                "`{what}` pattern `{pattern}` uses `**`: a multi-segment wildcard belongs to \
                 `ignore` only; an outputs pattern names the slots a report reads, one pattern \
                 one slot family"
            )));
        }
    }
    Ok(())
}

/// `a, b, c` for the first few names, then `… (+N more)`: an error message has to stay readable
/// when a whole description is unchecked. The one summarizer, shared with the loading check's
/// report.
pub fn summarize(names: &[String]) -> String {
    const SHOWN: usize = 8;
    let mut list = names
        .iter()
        .take(SHOWN)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ");
    if names.len() > SHOWN {
        list.push_str(&format!(", … (+{} more)", names.len() - SHOWN));
    }
    list
}

/// `ignore`'s pattern syntax is `binding.source`'s plus `**` (C6). An entry that matches nothing is
/// a warning at load-check time — the same description may be checked against another checkpoint —
/// but an entry that cannot name a subtree is an error here (I-5).
///
/// The rule is **anchoring**: an `ignore` entry has to start with a concrete segment. `ignore` is
/// C5's "explicit declaration" of the tensors a description drops on purpose, and a pattern whose
/// first segment is a wildcard declares nothing in particular — `**` drops the entire checkpoint,
/// `*` drops every tensor whose second segment happens to match, and `*.visual.**` drops whatever
/// the tower is called today. Those are blanket exemptions written in the shape of a declaration,
/// so they are rejected rather than reported as a very successful `ignore`.
fn check_ignore_patterns(desc: &ModelDesc) -> Result<(), ModelError> {
    for pattern in &desc.ignore {
        if pattern.is_empty() {
            return Err(ModelError::Invalid(
                "`ignore` has an empty pattern; an entry must name at least one segment (the \
                 vision tower is dropped by writing `model.visual.**`, C6)"
                    .to_string(),
            ));
        }
        if pattern.split('.').any(str::is_empty) {
            return Err(ModelError::Invalid(format!(
                "`ignore` pattern `{pattern}` has an empty segment: a `.` with nothing on one side \
                 of it matches no tensor name"
            )));
        }
        let mut segments = pattern.split('.');
        let first = segments.next().unwrap_or_default();
        if is_wildcard_segment(first) {
            let all_wildcards = pattern.split('.').all(is_wildcard_segment);
            let what = if all_wildcards {
                "is made of nothing but wildcards: ignoring every tensor is not an explicit \
                 declaration, it is giving the declaration up (C5)"
            } else {
                "starts with a wildcard segment: an `ignore` entry declares *which* subtree the \
                 description drops on purpose, so the first segment has to be a concrete name, not \
                 a wildcard that blanket-drops whatever sits at that level"
            };
            return Err(ModelError::Invalid(format!(
                "`ignore` pattern `{pattern}` {what}. Name the subtree to drop, e.g. \
                 `model.visual.**`"
            )));
        }
    }
    Ok(())
}

/// A `binding` pattern takes single-segment wildcards only: `{*}` in a source, `*` in a target
/// (§3.4). `**` belongs to `ignore` alone (C6) — a multi-segment wildcard in a binding is how one
/// checkpoint tensor silently feeds a whole subtree of slots.
fn check_binding_pattern(pattern: &str, what: &str) -> Result<(), String> {
    if pattern.is_empty() {
        return Err(format!(
            "{what} is empty; a pattern must name at least one segment"
        ));
    }
    for segment in pattern.split('.') {
        if segment.is_empty() {
            return Err(format!(
                "{what} `{pattern}` has an empty segment: a `.` with nothing on one side of it \
                 matches no slot name"
            ));
        }
        if segment == "**" {
            return Err(format!(
                "{what} `{pattern}` uses `**`: a multi-segment wildcard belongs to `ignore` only \
                 (C6); a binding pairs one concrete checkpoint tensor with one concrete slot"
            ));
        }
    }
    Ok(())
}

/// Every declared template slot must be read or written by some node of that template (§3.7 #11).
///
/// A slot nothing touches is a dead hook: it looks like part of the model, it costs a `Plan` slot
/// and it will need a `binding`, but no computation ever reaches it. Checked per template rather
/// than per instance so that a template instantiated zero times is caught too.
fn check_template_slots(desc: &ModelDesc) -> Result<(), ModelError> {
    for (template_name, template) in &desc.templates {
        let mut touched: BTreeSet<&str> = BTreeSet::new();
        for node in &template.nodes {
            touched.extend(node.inputs.iter().map(String::as_str));
            touched.extend(node.outputs.iter().map(String::as_str));
        }
        for decl in &template.slots {
            if !touched.contains(decl.name.as_str()) {
                return Err(ModelError::Invalid(format!(
                    "template `{template_name}` declares slot `{}`, which no node in the template \
                     reads or writes; a declared slot must be an input or an output of some node \
                     (§3.7 #11)",
                    decl.name
                )));
            }
        }
    }
    Ok(())
}

/// `select` and `template` name the same thing, so at most one of them may be present (§3.7 #13).
///
/// With `select` the only fallback is `select.default`; a `template` beside it would be a second
/// source for the fallback, which is the exact failure this project exists to avoid. Without
/// `select` the entry has no other way to name its template.
fn check_template_source(index: usize, entry: &StackEntry) -> Result<(), ModelError> {
    match (&entry.select, &entry.template) {
        (Some(_), Some(_)) | (None, None) => Err(template_source_conflict(index, entry)),
        _ => Ok(()),
    }
}

/// Both "two sources" and "no source" are the same broken stack entry, reported the same way.
fn template_source_conflict(index: usize, entry: &StackEntry) -> ModelError {
    let prefix = &entry.prefix;
    match (&entry.select, &entry.template) {
        (Some(_), Some(_)) => ModelError::Invalid(format!(
            "stack entry {index} (prefix `{prefix}`) declares both `select` and `template`: they \
             are two fallback sources for one fact, and `select` already picks the template by \
             list index — keep `select` (its only fallback is `select.default`) and drop `template` \
             (§3.7 #13)"
        )),
        _ => ModelError::Invalid(format!(
            "stack entry {index} (prefix `{prefix}`) declares neither `template` nor `select`; \
             without `select` the entry must name its template with `template`"
        )),
    }
}

struct Expander<'a> {
    desc: &'a ModelDesc,
    params: Params,
    default_dtype: RsDtype,
    builder: PlanBuilder,
    /// slot global name → id.
    slots: BTreeMap<String, SlotId>,
    /// id → slot global name (`SlotId` is allocation order).
    names: Vec<String>,
    /// id → (kind, shape).
    slot_info: Vec<(SlotKind, Vec<i64>)>,
    /// slot global name → where it was declared (a duplicate reports both origins, §3.6 #7).
    origins: BTreeMap<String, String>,
    /// instance prefix → the stack index it first appeared at, used to report duplicates.
    prefixes: BTreeMap<String, usize>,
    /// The outputs of the previous stack entry's last instance (the source of chained wiring).
    prev_outputs: Vec<SlotId>,
    /// The last index of the most recent repeat/until expansion, for `{last}`.
    last_index: Option<i64>,
    /// Every expanded instance in stack order: `(prefix, declared stage)`.
    stages: Vec<(String, Option<i64>)>,
    /// The `axes` declared on the top-level `inputs` (D6), keyed by slot name.
    input_axes: BTreeMap<String, BTreeMap<String, Vec<AxisDecl>>>,
}

impl<'a> Expander<'a> {
    fn new(desc: &'a ModelDesc, params: Params, default_dtype: RsDtype) -> Self {
        Self {
            desc,
            params,
            default_dtype,
            builder: PlanBuilder::new(desc.name.clone(), Phase::Forward, global_mesh()),
            slots: BTreeMap::new(),
            names: Vec::new(),
            slot_info: Vec::new(),
            origins: BTreeMap::new(),
            prefixes: BTreeMap::new(),
            prev_outputs: Vec::new(),
            last_index: None,
            stages: Vec::new(),
            input_axes: BTreeMap::new(),
        }
    }

    // ---- top level: the inputs section --------------------------------------

    fn expand_inputs(&mut self) -> Result<(), ModelError> {
        let desc = self.desc;
        // An input is an activation: it has no unit, and a description that declares one is
        // told so here rather than sharded by single elements.
        for (name, port) in &desc.inputs {
            for (dim, axes) in &port.axes {
                if axes
                    .iter()
                    .any(|axis| matches!(axis, AxisDecl::Sharded { unit: Some(_), .. }))
                {
                    return Err(ModelError::Invalid(format!(
                        "input `{name}` declares a `unit` on axis {dim}; units belong to a slot's \
                         binding, not to an input activation"
                    )));
                }
            }
        }
        let desc = self.desc;
        for (name, port) in &desc.inputs {
            let dtype = self.dtype_of(port.dtype.as_deref())?;
            let kind = parse_kind(&port.kind)?;
            let shape = self.shape_of(&port.shape)?;
            let origin = format!("the description's `inputs` section (`{name}`)");
            self.add_slot(name, dtype, shape, kind, &origin)?;
            // The input's own distribution (D6): recorded here and handed to
            // `instantiate` through `declarations()`, exactly like the binding
            // axes. Absent axes = replicated = absent from the map.
            if !port.axes.is_empty() {
                self.input_axes.insert(name.clone(), port.axes.clone());
            }
        }
        Ok(())
    }

    // ---- top level: the stack section ---------------------------------------

    fn expand_stack(&mut self) -> Result<(), ModelError> {
        let desc = self.desc;
        for (index, entry) in desc.stack.iter().enumerate() {
            self.expand_entry(index, entry)?;
        }
        Ok(())
    }

    fn expand_entry(&mut self, index: usize, entry: &'a StackEntry) -> Result<(), ModelError> {
        // Checked before the zero-instance early return: a description that names its template
        // twice is wrong even when this entry expands to nothing.
        check_template_source(index, entry)?;

        let (count, index_var) = match (&entry.repeat, &entry.until) {
            (Some(repeat), None) => (
                self.int_of(&repeat.count, "repeat.count")?,
                Some(repeat.index.as_str()),
            ),
            (None, Some(until)) => (self.int_of(until, "until")?, Some(UNTIL_INDEX)),
            (None, None) => (1, None),
            (Some(_), Some(_)) => {
                return Err(ModelError::Invalid(format!(
                    "stack entry {index} (prefix `{}`) declares both `repeat` and `until`",
                    entry.prefix
                )));
            }
        };

        if count <= 0 {
            return Ok(());
        }
        if index_var.is_some() {
            self.last_index = Some(count - 1);
        }

        // R1: the stage of each instance, declared on the entry — an integer for every
        // instance, or a list indexed by the repeat counter (one entry per instance). No
        // derivation: the generator writes the list, exactly like `layer_types` (§3.1).
        let stages: Vec<Option<i64>> = match &entry.stage {
            None => vec![None; count as usize],
            Some(StageDecl::Int(stage)) => vec![Some(*stage); count as usize],
            Some(StageDecl::List(list)) => {
                if list.len() != count as usize {
                    return Err(ModelError::Invalid(format!(
                        "stack entry {index} (prefix `{}`) declares {} stage(s) but expands to \
                         {count} instance(s); a stage list must have one entry per instance",
                        entry.prefix,
                        list.len()
                    )));
                }
                list.iter().copied().map(Some).collect()
            }
        };

        // Chained wiring: instance 0 takes the previous stack entry's last instance, every later
        // instance takes the one before it.
        let mut chained: Vec<SlotId> = std::mem::take(&mut self.prev_outputs);
        for position in 0..count {
            let position = position as usize;
            let template_name = self.select_template(index, entry, position, index_var)?;
            let prefix = self.substitute(index, &entry.prefix, position, index_var)?;
            let template = self.desc.templates.get(&template_name).ok_or_else(|| {
                ModelError::Invalid(format!(
                    "stack entry {index} (prefix `{prefix}`) names template `{template_name}`, \
                     which the description does not define"
                ))
            })?;

            if let Some(first) = self.prefixes.insert(prefix.clone(), index) {
                return Err(ModelError::Invalid(format!(
                    "stack instance prefix `{prefix}` is declared twice (stack entries {first} and \
                     {index}); instance prefixes must be unique (§3.6 #7)"
                )));
            }

            // Recorded in stack order so `Expanded::declarations()` hands `instantiate` the
            // instance list exactly as the description laid it out.
            self.stages.push((prefix.clone(), stages[position]));

            let wiring = self.wiring_for(
                index,
                entry,
                &prefix,
                &template_name,
                template,
                position,
                index_var,
                &chained,
            )?;
            chained = self.expand_instance(&prefix, &template_name, template, &wiring)?;
        }
        self.prev_outputs = chained;
        Ok(())
    }

    /// Resolve one instance's input wiring and check that port shapes match the source slots.
    #[allow(clippy::too_many_arguments)]
    fn wiring_for(
        &self,
        entry_index: usize,
        entry: &StackEntry,
        prefix: &str,
        template_name: &str,
        template: &Template,
        position: usize,
        index_var: Option<&str>,
        chained: &[SlotId],
    ) -> Result<BTreeMap<String, SlotId>, ModelError> {
        let wiring: BTreeMap<String, SlotId> = match &entry.inputs {
            Some(map) => {
                let mut wiring = BTreeMap::new();
                for (local_name, global_name) in map {
                    if !template.inputs.contains_key(local_name) {
                        return Err(ModelError::Invalid(format!(
                            "stack entry {entry_index} (prefix `{prefix}`) wires `{local_name}`, \
                             which template `{template_name}` does not declare as an input"
                        )));
                    }
                    let global_name =
                        self.substitute(entry_index, global_name, position, index_var)?;
                    let id = *self.slots.get(&global_name).ok_or_else(|| {
                        ModelError::Invalid(format!(
                            "stack entry {entry_index} (prefix `{prefix}`) wires input \
                             `{local_name}` to `{global_name}`, which is not a slot of this model"
                        ))
                    })?;
                    wiring.insert(local_name.clone(), id);
                }
                wiring
            }
            None => {
                if template.inputs.len() != chained.len() {
                    return Err(ModelError::Invalid(format!(
                        "stack entry {entry_index} (prefix `{prefix}`, template `{template_name}`) \
                         declares {} input(s) but the previous instance produces {} output(s); \
                         wire it explicitly with `inputs`",
                        template.inputs.len(),
                        chained.len()
                    )));
                }
                template
                    .inputs
                    .keys()
                    .zip(chained.iter())
                    .map(|(name, id)| (name.clone(), *id))
                    .collect()
            }
        };

        for (local_name, port) in &template.inputs {
            let id = *wiring.get(local_name).ok_or_else(|| {
                ModelError::Invalid(format!(
                    "stack entry {entry_index} (prefix `{prefix}`) leaves input `{local_name}` of \
                     template `{template_name}` unwired"
                ))
            })?;
            // `axes` is a top-level-input declaration (D6); on a template
            // port it would be silently unread, which I-5 forbids — reject it
            // by name instead.
            if !port.axes.is_empty() {
                return Err(ModelError::Invalid(format!(
                    "stack entry {entry_index} (prefix `{prefix}`) declares `axes` on input \
                     `{local_name}` of template `{template_name}`; sharding declarations belong \
                     on the top-level `inputs` section or on `binding` entries"
                )));
            }
            let declared = self.shape_of(&port.shape)?;
            let (_, held) = self.info(id)?;
            if declared != held {
                return Err(ModelError::Invalid(format!(
                    "stack entry {entry_index} (prefix `{prefix}`) wires input `{local_name}` of \
                     template `{template_name}`, declared {declared:?}, to slot `{}` of shape {held:?}",
                    self.name_of(id)?
                )));
            }
        }

        Ok(wiring)
    }

    fn expand_instance(
        &mut self,
        prefix: &str,
        template_name: &str,
        template: &Template,
        wiring: &BTreeMap<String, SlotId>,
    ) -> Result<Vec<SlotId>, ModelError> {
        let mut local: BTreeMap<String, SlotId> = wiring.clone();

        // 1. Template slots: the global name is the instance prefix plus the local name.
        for decl in &template.slots {
            let name = format!("{prefix}.{}", decl.name);
            let dtype = self.dtype_of(decl.dtype.as_deref())?;
            let kind = parse_kind(&decl.kind)?;
            let shape = self.shape_of(&decl.shape)?;
            let origin = format!("instance `{prefix}`, template slot `{}`", decl.name);
            let id = self.add_slot(&name, dtype, shape, kind, &origin)?;
            local.insert(decl.name.clone(), id);
        }

        // 2. Nodes: template order is emission order.
        let mut produced: BTreeSet<String> = BTreeSet::new();
        for (node_index, node) in template.nodes.iter().enumerate() {
            self.emit_node(
                prefix,
                template_name,
                template,
                node_index,
                node,
                &mut local,
                &mut produced,
            )?;
        }

        // 3. A declared output must actually be produced, otherwise downstream wiring connects to
        // thin air.
        let mut outputs = Vec::with_capacity(template.outputs.len());
        for (name, port) in &template.outputs {
            if !port.axes.is_empty() {
                return Err(ModelError::Invalid(format!(
                    "instance `{prefix}`: template `{template_name}` declares `axes` on output \
                     `{name}`; sharding declarations belong on the top-level `inputs` section \
                     or on `binding` entries"
                )));
            }
            let id = local.get(name).ok_or_else(|| {
                ModelError::Invalid(format!(
                    "instance `{prefix}`: template declares output `{name}`, but no node in this \
                     instance produces it"
                ))
            })?;
            outputs.push(*id);
        }
        Ok(outputs)
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_node(
        &mut self,
        prefix: &str,
        template_name: &str,
        template: &Template,
        node_index: usize,
        node: &NodeDecl,
        local: &mut BTreeMap<String, SlotId>,
        produced: &mut BTreeSet<String>,
    ) -> Result<(), ModelError> {
        if node.outputs.is_empty() {
            return Err(ModelError::Invalid(format!(
                "instance `{prefix}`: node {node_index} (`{}`) declares no output; every node must \
                 produce something",
                node.op
            )));
        }

        let mut inputs = Vec::with_capacity(node.inputs.len());
        for name in &node.inputs {
            let id = local.get(name).ok_or_else(|| {
                ModelError::Invalid(format!(
                    "instance `{prefix}`: node {node_index} (`{}`) reads `{name}`, which is not a \
                     template slot, an instance input, or an output of an earlier node",
                    node.op
                ))
            })?;
            inputs.push(*id);
        }

        let mut outputs = Vec::with_capacity(node.outputs.len());
        for name in &node.outputs {
            if produced.contains(name) {
                return Err(ModelError::Invalid(format!(
                    "instance `{prefix}`: `{name}` is written by more than one node"
                )));
            }

            // §3.7 #1: a node's `out` may only reference a slot the template declares or an
            // `outputs` entry of this instance. It may **not** be inferred by the compiler (the old
            // implementation inherited the shape of the node's first input): that would make the
            // GPU-free L1 shape check depend on "a resolvable implementation exists", and real
            // descriptions are bf16 while the reference provider only has f32 — the whole of L1
            // would be dead, and L1 is the reason this architecture exists.
            let declared_slot = template.slots.iter().any(|decl| decl.name == *name);
            let id = match local.get(name) {
                Some(id) if declared_slot => {
                    // Weights are not produced by operators: writing to a weight slot means the
                    // description conflated two different things.
                    let (kind, _) = self.info(*id)?;
                    if !matches!(
                        kind,
                        SlotKind::Activation | SlotKind::Temp | SlotKind::Output
                    ) {
                        return Err(ModelError::Invalid(format!(
                            "instance `{prefix}`: node {node_index} (`{}`) writes `{name}`, which is \
                             declared as a {kind:?} slot; a node produces activations, not parameters",
                            node.op
                        )));
                    }
                    *id
                }
                _ => {
                    // Only one legal source is left: this instance's declared `outputs` (the slot is
                    // allocated the first time the name is written).
                    let Some(port) = template.outputs.get(name) else {
                        return Err(ModelError::Invalid(format!(
                            "instance `{prefix}`: node {node_index} (`{}`) writes `{name}`, which \
                             template `{template_name}` declares neither as a slot nor as an \
                             output (§3.7 #1)",
                            node.op
                        )));
                    };
                    let dtype = self.dtype_of(port.dtype.as_deref())?;
                    let kind = parse_kind(&port.kind)?;
                    let shape = self.shape_of(&port.shape)?;
                    let global = format!("{prefix}.{name}");
                    let origin = format!(
                        "instance `{prefix}`, output `{name}` of node {node_index} (`{}`)",
                        node.op
                    );
                    let id = self.add_slot(&global, dtype, shape, kind, &origin)?;
                    local.insert(name.clone(), id);
                    id
                }
            };
            outputs.push(id);
            produced.insert(name.clone());
        }

        let attrs = attrs_of(&node.attrs);
        let trace = format!("{prefix}.{}", node.outputs[0]);
        self.builder
            .node(OpRef::new(node.op.clone()), inputs, outputs, attrs, trace);
        Ok(())
    }

    // ---- helpers ------------------------------------------------------------

    /// A slot by its global name: allocate in declaration order first, then record name and shape.
    fn add_slot(
        &mut self,
        name: &str,
        dtype: RsDtype,
        shape: Vec<i64>,
        kind: SlotKind,
        origin: &str,
    ) -> Result<SlotId, ModelError> {
        if let Some(first) = self.origins.get(name) {
            return Err(ModelError::Invalid(format!(
                "slot `{name}` is declared twice: by {first}, and by {origin}; slot names must be \
                 unique (§3.6 #7)"
            )));
        }
        let id = self
            .builder
            .slot(name.to_string(), dtype, shape.clone(), kind);
        self.slots.insert(name.to_string(), id);
        self.names.push(name.to_string());
        self.slot_info.push((kind, shape));
        self.origins.insert(name.to_string(), origin.to_string());
        Ok(id)
    }

    fn name_of(&self, id: SlotId) -> Result<String, ModelError> {
        self.names
            .get(id.0)
            .cloned()
            .ok_or_else(|| self.unknown_slot(id))
    }

    fn info(&self, id: SlotId) -> Result<(SlotKind, Vec<i64>), ModelError> {
        self.slot_info
            .get(id.0)
            .cloned()
            .ok_or_else(|| self.unknown_slot(id))
    }

    /// A `SlotId` outside this expander's own table.
    ///
    /// Ids come only from [`Self::add_slot`], so this is an internal invariant, not user input.
    /// It must stay an error: defaulting to `Activation` (what this used to do) would let an
    /// unknown id pass the "a node may not write a weight slot" check by accident.
    fn unknown_slot(&self, id: SlotId) -> ModelError {
        debug_assert!(false, "SlotId {id:?} was never allocated by this expander");
        ModelError::Invalid(format!(
            "internal error: slot id {} was never allocated by this expander",
            id.0
        ))
    }

    fn dtype_of(&self, name: Option<&str>) -> Result<RsDtype, ModelError> {
        match name {
            Some(name) => parse_dtype(name),
            None => Ok(self.default_dtype),
        }
    }

    fn shape_of(&self, shape: &[String]) -> Result<Vec<i64>, ModelError> {
        shape.iter().map(|entry| self.dim_of(entry)).collect()
    }

    fn dim_of(&self, entry: &str) -> Result<i64, ModelError> {
        match self.params.get(entry) {
            Some(Value::Int(v)) => Ok(*v),
            Some(Value::List(_)) => Err(ModelError::Invalid(format!(
                "shape entry `{entry}` names a list param; shapes take integers"
            ))),
            None => entry.parse::<i64>().map_err(|_| {
                ModelError::Invalid(format!(
                    "shape entry `{entry}` is neither a declared integer param nor an integer literal"
                ))
            }),
        }
    }

    fn int_of(&self, name: &str, what: &str) -> Result<i64, ModelError> {
        self.params
            .int(name)
            .map_err(|e| ModelError::Invalid(format!("{what}: {e}")))
    }

    /// Pick the template for one instance (§3.3: selection goes by list index only, never
    /// arithmetic).
    fn select_template(
        &self,
        entry_index: usize,
        entry: &StackEntry,
        position: usize,
        index_var: Option<&str>,
    ) -> Result<String, ModelError> {
        let select = match &entry.select {
            Some(select) => select,
            // `check_template_source` already rejected an entry with neither source.
            None => {
                return entry
                    .template
                    .clone()
                    .ok_or_else(|| template_source_conflict(entry_index, entry));
            }
        };
        let (list_name, index_name) = parse_indexed(&select.by).ok_or_else(|| {
            ModelError::Invalid(format!(
                "stack entry {entry_index}: `select.by` = `{}` is not `<param>[<index>]`",
                select.by
            ))
        })?;
        match index_var {
            Some(var) if var == index_name => {}
            other => {
                return Err(ModelError::Invalid(format!(
                    "stack entry {entry_index}: `select.by` = `{}` indexes with `{index_name}`, but \
                     this entry's index variable is {}",
                    select.by,
                    other.unwrap_or("(none)")
                )));
            }
        }
        let items = self
            .params
            .list(&list_name)
            .map_err(|e| ModelError::Invalid(format!("stack entry {entry_index}: {e}")))?;
        let value = items.get(position).ok_or_else(|| {
            ModelError::Invalid(format!(
                "stack entry {entry_index}: `{}` has {} entries, so instance {position} has no value",
                select.by,
                items.len()
            ))
        })?;
        if let Some(template) = select.cases.get(value) {
            return Ok(template.clone());
        }
        if let Some(fallback) = &select.default {
            return Ok(fallback.clone());
        }
        Err(ModelError::Invalid(format!(
            "stack entry {entry_index}: `{}` = `{value}` matches no case (cases: {}) and no \
             `default` is declared",
            select.by,
            select.cases.keys().cloned().collect::<Vec<_>>().join(", ")
        )))
    }

    /// Substitute `{<index variable>}` and `{last}`; a leftover placeholder is an error.
    fn substitute(
        &self,
        entry_index: usize,
        text: &str,
        position: usize,
        index_var: Option<&str>,
    ) -> Result<String, ModelError> {
        let mut out = text.to_string();
        if let Some(var) = index_var {
            out = out.replace(&format!("{{{var}}}"), &position.to_string());
        }
        if out.contains(&format!("{{{LAST_INDEX}}}")) {
            let last = self.last_index.ok_or_else(|| {
                ModelError::Invalid(format!(
                    "stack entry {entry_index}: `{text}` uses `{{last}}`, but no `repeat`/`until` \
                     expansion precedes it"
                ))
            })?;
            out = out.replace(&format!("{{{LAST_INDEX}}}"), &last.to_string());
        }
        if out.contains('{') {
            return Err(ModelError::Invalid(format!(
                "stack entry {entry_index}: `{text}` still has an unsubstituted placeholder \
                 (index variable: {})",
                index_var.unwrap_or("(none)")
            )));
        }
        Ok(out)
    }

    // ---- binding ----------------------------------------------------------

    /// Check that every `binding` hits weight slots and only weight slots (§3.5, §4.1 step 5).
    ///
    /// Returns the resolved bindings plus the weight slots nothing claimed; [`expand`] turns a
    /// non-empty second list into the §3.5 error, [`expand_lenient`] passes it on.
    fn bind(&self) -> Result<(Vec<ResolvedBinding>, Vec<String>), ModelError> {
        let weights: Vec<String> = self
            .slot_info
            .iter()
            .enumerate()
            .filter(|(_, (kind, _))| *kind == SlotKind::Weight)
            .map(|(id, _)| self.names[id].clone())
            .collect();

        // Slot → the claim on it: the binding index, its source and the pattern that hit the slot,
        // so a collision can say *which* of the two things went wrong — one binding naming the same
        // target twice reads very differently from two bindings fighting over one slot.
        let mut claimed: BTreeMap<String, (usize, String, String)> = BTreeMap::new();
        let mut sources: BTreeMap<&str, usize> = BTreeMap::new();
        let mut resolved = Vec::with_capacity(self.desc.binding.len());

        for (index, binding) in self.desc.binding.iter().enumerate() {
            // §3.7 #4: one checkpoint tensor feeds one slot. Two bindings on the same source would
            // load that tensor into two places and neither report would notice.
            if let Some(first) = sources.insert(binding.source.as_str(), index) {
                return Err(ModelError::Invalid(format!(
                    "binding {first} and binding {index} both declare source `{}`: a checkpoint \
                     tensor is consumed once, by one binding (§3.7 #4)",
                    binding.source
                )));
            }
            check_binding_pattern(&binding.source, "source").map_err(|e| {
                ModelError::Invalid(format!("binding {index} (`{}`): {e}", binding.source))
            })?;
            if let Some(slot) = &binding.slot {
                check_binding_pattern(slot, "slot").map_err(|e| {
                    ModelError::Invalid(format!("binding {index} (`{}`): {e}", binding.source))
                })?;
            }

            let targets: Vec<(String, BTreeMap<String, Vec<AxisDecl>>)> =
                match (&binding.slot, binding.targets.is_empty()) {
                    (Some(slot), true) => vec![(slot.clone(), binding.axes.clone())],
                    (None, false) => binding
                        .targets
                        .iter()
                        .map(|t: &Target| (t.slot.clone(), t.axes.clone()))
                        .collect(),
                    (Some(_), false) => {
                        return Err(ModelError::Invalid(format!(
                            "binding {index} (`{}`) declares both `slot` and `targets`; `split` \
                             targets replace `slot`",
                            binding.source
                        )));
                    }
                    (None, true) => {
                        return Err(ModelError::Invalid(format!(
                            "binding {index} (`{}`) declares neither `slot` nor `targets`",
                            binding.source
                        )));
                    }
                };

            for (target_index, target) in binding.targets.iter().enumerate() {
                check_binding_pattern(&target.slot, &format!("targets[{target_index}].slot"))
                    .map_err(|e| {
                        ModelError::Invalid(format!("binding {index} (`{}`): {e}", binding.source))
                    })?;
            }

            let split = match &binding.split {
                Some(split) => {
                    if binding.targets.is_empty() {
                        return Err(ModelError::Invalid(format!(
                            "binding {index} (`{}`) declares `split` without `targets`",
                            binding.source
                        )));
                    }
                    if split.sizes.len() != targets.len() {
                        return Err(ModelError::Invalid(format!(
                            "binding {index} (`{}`): `split.sizes` has {} entr(ies) but {} target(s) \
                             are declared",
                            binding.source,
                            split.sizes.len(),
                            targets.len()
                        )));
                    }
                    let mut sizes = Vec::with_capacity(split.sizes.len());
                    for size in &split.sizes {
                        let value = self.dim_of(size)?;
                        if value <= 0 {
                            return Err(ModelError::Invalid(format!(
                                "binding {index} (`{}`): split size `{size}` = {value} is not positive",
                                binding.source
                            )));
                        }
                        sizes.push(value);
                    }
                    Some(ResolvedSplit {
                        dim: split.dim,
                        sizes,
                    })
                }
                None => {
                    if !binding.targets.is_empty() {
                        return Err(ModelError::Invalid(format!(
                            "binding {index} (`{}`) declares `targets` without `split`",
                            binding.source
                        )));
                    }
                    None
                }
            };

            for transform in &binding.transform {
                parse_transform(transform).map_err(|e| {
                    ModelError::Invalid(format!("binding {index} (`{}`): {e}", binding.source))
                })?;
            }

            // Resolve every declared axis' unit here, next to `split.sizes`: both name a
            // parameter or a literal, and both are needed before a slot exists. The resolved
            // targets are a `Vec`, in the order the bindings were read: a slot claimed twice must
            // stay claimed twice so the conflict is reported, which a map keyed by slot would
            // quietly merge away.
            let mut resolved_targets: Vec<(String, BTreeMap<String, Vec<ResolvedAxis>>)> =
                Vec::with_capacity(targets.len());
            for (slot, axes) in &targets {
                let mut per_dim: BTreeMap<String, Vec<ResolvedAxis>> = BTreeMap::new();
                for (dim, declarations) in axes {
                    let mut resolved = Vec::with_capacity(declarations.len());
                    for axis in declarations {
                        // A unit says how coarse a *replicating* shard is. On a strict shard it has
                        // no meaning at all, and dropping it (which is what reading the mode alone
                        // would do) would hand the kernel slabs nobody declared.
                        if let AxisDecl::Sharded {
                            mode: AxisMode::Divide,
                            unit: Some(text),
                            ..
                        } = axis
                        {
                            return Err(ModelError::Invalid(format!(
                                "binding {index} (`{}`): axis unit `{text}` is only meaningful \
                                 with `\"mode\": \"replicate\"`; a strict shard has no unit",
                                binding.source
                            )));
                        }
                        let unit = match axis {
                            AxisDecl::Name(_) | AxisDecl::Sharded { unit: None, .. } => 1,
                            AxisDecl::Sharded {
                                unit: Some(text), ..
                            } => {
                                let value = self.dim_of(text)?;
                                if value <= 0 {
                                    return Err(ModelError::Invalid(format!(
                                        "binding {index} (`{}`): axis unit `{text}` = {value} is \
                                         not positive",
                                        binding.source
                                    )));
                                }
                                value
                            }
                        };
                        resolved.push(ResolvedAxis {
                            axis: axis.axis().to_string(),
                            mode: axis.mode(),
                            unit,
                        });
                    }
                    per_dim.insert(dim.clone(), resolved);
                }
                resolved_targets.push((slot.clone(), per_dim));
            }
            let targets = resolved_targets;

            let captures = source_captures(&binding.source);
            let mut slots = Vec::new();
            for (pattern, axes) in &targets {
                let wildcards = wildcard_count(pattern);
                if wildcards != captures {
                    return Err(ModelError::Invalid(format!(
                        "binding {index} (`{}`): pattern `{pattern}` has {wildcards} `*` but the \
                         source has {captures} `{{*}}`; a shared capture needs one of each",
                        binding.source
                    )));
                }
                let hits: Vec<&String> = weights
                    .iter()
                    .filter(|name| pattern_matches(pattern.as_str(), name.as_str()))
                    .collect();
                if hits.is_empty() {
                    return Err(ModelError::Invalid(format!(
                        "binding for source `{}` matches no slot: pattern `{pattern}`",
                        binding.source
                    )));
                }
                for hit in hits {
                    let claim = (index, binding.source.clone(), pattern.clone());
                    if let Some((previous_index, previous_source, previous_pattern)) =
                        claimed.insert(hit.clone(), claim)
                    {
                        let reason = if previous_index == index {
                            format!(
                                "binding {index} (`{}`) names slot `{hit}` twice: patterns \
                                 `{previous_pattern}` and `{pattern}` both hit it, so the same slot \
                                 would be loaded from two places",
                                binding.source
                            )
                        } else {
                            format!(
                                "slot `{hit}` is claimed by two bindings: binding \
                                 {previous_index} (`{previous_source}`, pattern \
                                 `{previous_pattern}`) and binding {index} (`{}`, pattern \
                                 `{pattern}`)",
                                binding.source
                            )
                        };
                        return Err(ModelError::Invalid(reason));
                    }
                    slots.push(ResolvedBindingSlot {
                        slot: hit.clone(),
                        pattern: pattern.clone(),
                        axes: axes.clone(),
                    });
                }
            }

            resolved.push(ResolvedBinding {
                source: binding.source.clone(),
                transform: binding.transform.clone(),
                split,
                slots,
            });
        }

        let unbound: Vec<String> = weights
            .iter()
            .filter(|name| !claimed.contains_key(*name))
            .cloned()
            .collect();

        Ok((resolved, unbound))
    }
}

/// The canonical spelling of `kind`.
fn parse_kind(name: &str) -> Result<SlotKind, ModelError> {
    match name {
        "weight" => Ok(SlotKind::Weight),
        "gradient" => Ok(SlotKind::Gradient),
        "state" => Ok(SlotKind::State),
        "activation" => Ok(SlotKind::Activation),
        "temp" => Ok(SlotKind::Temp),
        "input" => Ok(SlotKind::Input),
        "output" => Ok(SlotKind::Output),
        other => Err(ModelError::Invalid(format!(
            "unknown kind `{other}`; expected one of weight, gradient, state, activation, temp, \
             input, output"
        ))),
    }
}

/// The canonical spelling of `dtype`, identical to [`RsDtype::name`] (§3.6 #5).
fn parse_dtype(name: &str) -> Result<RsDtype, ModelError> {
    RsDtype::parse(name).ok_or_else(|| {
        ModelError::Invalid(format!(
            "unknown dtype `{name}`; expected one of {}",
            RsDtype::ALL
                .iter()
                .map(|d| d.name())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })
}

fn attrs_of(map: &BTreeMap<String, AttrLiteral>) -> Attrs {
    let mut attrs = Attrs::new();
    for (key, value) in map {
        let value = match value {
            AttrLiteral::Bool(v) => AttrValue::Bool(*v),
            AttrLiteral::Int(v) => AttrValue::I64(*v),
            AttrLiteral::Float(v) => AttrValue::F64(*v),
            AttrLiteral::Str(v) => AttrValue::Str(v.clone()),
            AttrLiteral::I64s(v) => AttrValue::I64s(v.clone()),
        };
        attrs.insert(key.clone(), value);
    }
    attrs
}

/// The syntax of `select.by`: `<param name>[<index variable>]`.
fn parse_indexed(text: &str) -> Option<(String, String)> {
    let (list, rest) = text.split_once('[')?;
    let index = rest.strip_suffix(']')?;
    let list = list.trim();
    let index = index.trim();
    if list.is_empty() || index.is_empty() {
        return None;
    }
    Some((list.to_string(), index.to_string()))
}

/// `*` matches **exactly one** dotted segment (§3.4; the one matcher lives in [`crate::pattern`]).
fn pattern_matches(pattern: &str, name: &str) -> bool {
    crate::pattern::matches(pattern, name)
}

fn wildcard_count(pattern: &str) -> usize {
    pattern.split('.').filter(|segment| *segment == "*").count()
}

fn source_captures(source: &str) -> usize {
    source.matches("{*}").count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn star_matches_exactly_one_segment() {
        assert!(pattern_matches("layers.*.q", "layers.0.q"));
        assert!(!pattern_matches("layers.*.q", "layers.0.1.q"));
        assert!(!pattern_matches("layers.*.q", "layers.0.q_norm"));
        assert!(!pattern_matches("layers.*.q", "layers.q"));
        assert!(pattern_matches("norm_in.w", "norm_in.w"));
    }

    #[test]
    fn the_transform_vocabulary_is_two_verbs_and_a_wrong_one_is_an_error() {
        use crate::transform::parse_transform as parse;
        assert!(parse("transpose(0,1)").is_ok());
        assert!(parse("slice(1, 0, 96)").is_ok());
        // C6 removed these three; they must not be accepted, least of all silently.
        for removed in ["take(model.up.weight)", "concat(0)", "split(1, [96, 64])"] {
            assert!(parse(removed).is_err(), "{removed} must not parse");
        }
        assert!(parse("transpose").is_err());
        assert!(parse("flip(0)").is_err());
        assert!(parse("transpose(0,1").is_err());
    }

    #[test]
    fn a_binding_pattern_takes_single_segment_wildcards_only() {
        assert!(check_binding_pattern("layers.{*}.q", "source").is_ok());
        assert!(check_binding_pattern("layers.*.q", "slot").is_ok());
        assert!(check_binding_pattern("model.visual.**", "source").is_err());
        assert!(check_binding_pattern("", "source").is_err());
        assert!(check_binding_pattern("layers..q", "slot").is_err());
        let error = check_binding_pattern("model.**.weight", "source").unwrap_err();
        assert!(error.contains("ignore"), "{error}");
    }

    #[test]
    fn kinds_and_dtypes_use_the_contract_spelling() {
        assert_eq!(parse_kind("weight").unwrap(), SlotKind::Weight);
        assert!(parse_kind("Weight").is_err());
        assert_eq!(parse_dtype("bf16").unwrap(), RsDtype::BF16);
        assert!(parse_dtype("bfloat16").is_err());
    }

    #[test]
    fn indexed_selector_is_parsed() {
        assert_eq!(
            parse_indexed("layer_types[l]"),
            Some(("layer_types".to_string(), "l".to_string()))
        );
        assert_eq!(parse_indexed("layer_types"), None);
    }
}

#[cfg(test)]
mod axes_declaration_tests {
    use super::*;

    fn desc_with(json: &str) -> serde_json::Value {
        serde_json::from_str(json).expect("fixture json parses")
    }

    fn minimal(template_inputs: &str, inputs: &str) -> String {
        format!(
            r#"{{
                "format": "rustrain.model.v1",
                "name": "axes-ports",
                "dtype": "f32",
                "inputs": {inputs},
                "params": {{ "n": {{ "from": "text_config.n" }} }},
                "templates": {{
                    "t": {{
                        "inputs": {template_inputs},
                        "outputs": {{ "y": {{ "shape": ["n"], "kind": "activation" }} }},
                        "nodes": [{{ "op": "scale", "in": ["x"], "out": ["y"] }}]
                    }}
                }},
                "stack": [{{ "template": "t", "prefix": "t", "inputs": {{ "x": "tok" }} }}],
                "outputs": {{ "logits": "t.y", "hidden": ["t.y"] }}
            }}"#
        )
    }

    /// `axes` on a template port is rejected by name — it would be silently
    /// unread otherwise (I-5: a wrong description is a hard error).
    #[test]
    fn axes_on_a_template_input_port_are_rejected() {
        let desc: ModelDesc = serde_json::from_str(&minimal(
            r#"{ "x": { "shape": ["n"], "kind": "activation", "axes": { "0": ["tp"] } } }"#,
            r#"{ "tok": { "shape": ["n"], "kind": "input", "dtype": "i64" } }"#,
        ))
        .unwrap();
        let config = desc_with(r#"{"text_config": {"n": 4}}"#);
        let error = expand(&desc, &config).unwrap_err();
        assert!(
            error.to_string().contains("sharding declarations belong"),
            "the rejection must say where axes belong: {error}"
        );
    }

    /// A top-level input **may** declare its own distribution (D6): it flows
    /// into `declarations()` and the plan instantiates with it.
    #[test]
    fn axes_on_a_top_level_input_flow_into_the_declarations() {
        let desc: ModelDesc = serde_json::from_str(&minimal(
            r#"{ "x": { "shape": ["n"], "kind": "activation" } }"#,
            r#"{ "tok": { "shape": ["n"], "kind": "input", "dtype": "i64", "axes": { "0": ["dp"] } } }"#,
        ))
        .unwrap();
        let config = desc_with(r#"{"text_config": {"n": 4}}"#);
        let expanded = expand(&desc, &config).unwrap();
        let declarations = expanded.declarations();
        assert_eq!(
            declarations.slots.get("tok").and_then(|d| d.get("0")),
            Some(&vec![DeclaredAxis::divide("dp")]),
            "the input's dp shard must reach instantiate"
        );
    }
}
