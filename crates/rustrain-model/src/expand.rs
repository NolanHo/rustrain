//! `expand`: description + `config.json` → the global `Plan` (§4.1).
//!
//! The five steps follow the contract: evaluate `params` → walk `stack` in order instantiating
//! templates (allocate slots, emit nodes) → evaluate shapes (fully concrete) → `check_structure()`
//! → attach the symbolic declarations of `binding`.
//!
//! Every `layout` in the global plan is `Replicate`: axes and sharding wait for `instantiate` to
//! see a mesh, so `expand` only records them ([`ResolvedBinding`]) and never touches a shape.

use std::collections::{BTreeMap, BTreeSet};

use rustrain_abi::ffi::RsDtype;
use rustrain_parallel::ParallelConfig;
use rustrain_plan::{AttrValue, Attrs, OpRef, Phase, Plan, PlanBuilder, SlotId, SlotKind};

use crate::ModelError;
use crate::desc::{AttrLiteral, FORMAT, ModelDesc, NodeDecl, StackEntry, Target, Template};
use crate::params::{Params, Value};

/// The index variable `until` expansion uses (the counterpart of `repeat.index`).
const UNTIL_INDEX: &str = "l";
/// The `{last}` placeholder: the last index of the most recent repeat/until expansion.
const LAST_INDEX: &str = "last";

/// What `expand` produces.
#[derive(Debug)]
pub struct Expanded {
    /// The global plan: concrete shapes, every `layout` `Replicate`, nodes in emission order.
    pub plan: Plan,
    /// The slots every `binding` hit (input of the L2 load check,
    /// `docs/design/model-description.md` §3.5).
    pub bindings: Vec<ResolvedBinding>,
    /// Weight slots no `binding` hit, in plan order (§3.5's first mandate). [`expand`] rejects a
    /// non-empty list; [`expand_lenient`] hands it back instead, because naming the unbound slot is
    /// precisely what a loading check has to report.
    pub unbound_slots: Vec<String>,
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
    pub axes: BTreeMap<String, Vec<String>>,
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
pub fn expand_lenient(desc: &ModelDesc, config: &serde_json::Value) -> Result<Expanded, ModelError> {
    if desc.format != FORMAT {
        return Err(ModelError::Format {
            found: desc.format.clone(),
            expected: FORMAT.to_string(),
        });
    }

    let params = Params::resolve(&desc.params, config)?;
    check_template_slots(desc)?;
    let default_dtype = match &desc.dtype {
        Some(name) => parse_dtype(name)?,
        None => RsDtype::F32,
    };

    let mut expander = Expander::new(desc, params, default_dtype);
    expander.expand_inputs()?;
    expander.expand_stack()?;

    let builder = std::mem::replace(
        &mut expander.builder,
        PlanBuilder::new("", Phase::Forward, ParallelConfig::default()),
    );
    let plan = builder.build()?;
    let (bindings, unbound_slots) = expander.bind()?;

    Ok(Expanded {
        plan,
        bindings,
        unbound_slots,
    })
}

/// `a, b, c` for the first few names, then `… (+N more)`: an error message has to stay readable
/// when a whole description is unbound.
fn summarize(names: &[String]) -> String {
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
}

impl<'a> Expander<'a> {
    fn new(desc: &'a ModelDesc, params: Params, default_dtype: RsDtype) -> Self {
        Self {
            desc,
            params,
            default_dtype,
            builder: PlanBuilder::new(desc.name.clone(), Phase::Forward, ParallelConfig::default()),
            slots: BTreeMap::new(),
            names: Vec::new(),
            slot_info: Vec::new(),
            origins: BTreeMap::new(),
            prefixes: BTreeMap::new(),
            prev_outputs: Vec::new(),
            last_index: None,
        }
    }

    // ---- top level: the inputs section --------------------------------------

    fn expand_inputs(&mut self) -> Result<(), ModelError> {
        let desc = self.desc;
        for (name, port) in &desc.inputs {
            let dtype = self.dtype_of(port.dtype.as_deref())?;
            let kind = parse_kind(&port.kind)?;
            let shape = self.shape_of(&port.shape)?;
            let origin = format!("the description's `inputs` section (`{name}`)");
            self.add_slot(name, dtype, shape, kind, &origin)?;
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
        for name in template.outputs.keys() {
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

        let mut claimed: BTreeMap<String, String> = BTreeMap::new();
        let mut resolved = Vec::with_capacity(self.desc.binding.len());

        for (index, binding) in self.desc.binding.iter().enumerate() {
            let targets: Vec<(String, BTreeMap<String, Vec<String>>)> =
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
                validate_transform(transform).map_err(|e| {
                    ModelError::Invalid(format!("binding {index} (`{}`): {e}", binding.source))
                })?;
            }

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
                    if let Some(previous) = claimed.insert(hit.clone(), binding.source.clone()) {
                        return Err(ModelError::Invalid(format!(
                            "slot `{hit}` is claimed by two bindings: `{previous}` and `{}`",
                            binding.source
                        )));
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

/// The `transform` vocabulary (§3.4): five verbs, and only five; argument semantics belong to the
/// loader.
fn validate_transform(text: &str) -> Result<(), String> {
    const VERBS: [&str; 5] = ["take", "slice", "transpose", "split", "concat"];
    let (verb, rest) = text
        .split_once('(')
        .ok_or_else(|| format!("transform `{text}` is not `<verb>(<args>)`"))?;
    if !VERBS.contains(&verb) {
        return Err(format!(
            "transform `{text}`: unknown verb `{verb}`; expected one of {}",
            VERBS.join(", ")
        ));
    }
    if !rest.ends_with(')') {
        return Err(format!("transform `{text}` is missing its closing `)`"));
    }
    Ok(())
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
    fn transform_verbs_are_checked() {
        assert!(validate_transform("transpose(0,1)").is_ok());
        assert!(validate_transform("split(2, [a, b])").is_ok());
        assert!(validate_transform("transpose").is_err());
        assert!(validate_transform("flip(0)").is_err());
        assert!(validate_transform("transpose(0,1").is_err());
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
