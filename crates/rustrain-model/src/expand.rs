//! `expand`：描述 + `config.json` → 全局 `Plan`（§4.1）。
//!
//! 五个步骤与契约一致：求值 `params` → 按 `stack` 顺序展开模板（分配 slot、发射节点）→
//! 形状求值（全量形状）→ `check_structure()` → 挂 `binding` 的符号声明。
//!
//! 全局 Plan 的 `layout` 全 `Replicate`：轴与切分要等 `instantiate` 拿到 mesh 才解析，
//! `expand` 只记录它们（[`ResolvedBinding`]），不改形状。

use std::collections::{BTreeMap, BTreeSet};

use rustrain_abi::ffi::RsDtype;
use rustrain_parallel::ParallelConfig;
use rustrain_plan::{AttrValue, Attrs, OpRef, Phase, Plan, PlanBuilder, SlotId, SlotKind};

use crate::ModelError;
use crate::desc::{AttrLiteral, FORMAT, ModelDesc, NodeDecl, StackEntry, Target, Template};
use crate::params::{Params, Value};

/// `until` 展开使用的索引变量名（`repeat.index` 的对称物）。
const UNTIL_INDEX: &str = "l";
/// `{last}` 占位符：最近一次 repeat/until 展开的最后一个下标。
const LAST_INDEX: &str = "last";

/// `expand` 的产物。
#[derive(Debug)]
pub struct Expanded {
    /// 全局 Plan：形状全量、`layout` 全 `Replicate`、节点按发射顺序拓扑。
    pub plan: Plan,
    /// 每条 `binding` 命中的 slot（L2 加载检查的输入，`docs/design/model-description.md` §3.5）。
    pub bindings: Vec<ResolvedBinding>,
}

/// 一条 `binding` 解析后的形态。
#[derive(Debug)]
pub struct ResolvedBinding {
    /// checkpoint 侧的张量名模式。
    pub source: String,
    /// 取值前的变换，原样保留（`expand` 不执行它们）。
    pub transform: Vec<String>,
    /// 融合存储的切分；`None` 表示这条 binding 直接喂一个 slot。
    pub split: Option<ResolvedSplit>,
    /// 这条 binding 喂的 slot，按声明顺序。
    pub slots: Vec<ResolvedBindingSlot>,
}

/// 一条解析后的切分声明。
#[derive(Debug)]
pub struct ResolvedSplit {
    pub dim: i64,
    pub sizes: Vec<i64>,
}

/// `binding` 命中的一个 slot。
#[derive(Debug)]
pub struct ResolvedBindingSlot {
    pub slot: String,
    /// slot 维度（十进制字符串）→ 符号轴名。
    pub axes: BTreeMap<String, Vec<String>>,
}

/// 描述 → 全局 Plan。
pub fn expand(desc: &ModelDesc, config: &serde_json::Value) -> Result<Expanded, ModelError> {
    if desc.format != FORMAT {
        return Err(ModelError::Format {
            found: desc.format.clone(),
            expected: FORMAT.to_string(),
        });
    }

    let params = Params::resolve(&desc.params, config)?;
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
    let bindings = expander.bind()?;

    Ok(Expanded { plan, bindings })
}

struct Expander<'a> {
    desc: &'a ModelDesc,
    params: Params,
    default_dtype: RsDtype,
    builder: PlanBuilder,
    /// slot 全局名 → id。
    slots: BTreeMap<String, SlotId>,
    /// id → slot 全局名（`SlotId` 就是分配序）。
    names: Vec<String>,
    /// id → (kind, shape)。
    slot_info: Vec<(SlotKind, Vec<i64>)>,
    /// slot 全局名 → 它是在哪里被声明的（重名时报出两个来源，§3.6 #7）。
    origins: BTreeMap<String, String>,
    /// 实例前缀 → 首次出现的 stack 下标，用于报出重名。
    prefixes: BTreeMap<String, usize>,
    /// 上一个 stack 项最后一个实例的 outputs（链式接线的来源）。
    prev_outputs: Vec<SlotId>,
    /// 最近一次 repeat/until 的最后一个下标，供 `{last}` 使用。
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

    // ---- 顶层：inputs 段 ---------------------------------------------------

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

    // ---- 顶层：stack 段 ---------------------------------------------------

    fn expand_stack(&mut self) -> Result<(), ModelError> {
        let desc = self.desc;
        for (index, entry) in desc.stack.iter().enumerate() {
            self.expand_entry(index, entry)?;
        }
        Ok(())
    }

    fn expand_entry(&mut self, index: usize, entry: &'a StackEntry) -> Result<(), ModelError> {
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

        // 链式接线：第 0 个实例接上一个 stack 项的最后一个实例，之后接上一个实例。
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

    /// 解析一个实例的输入接线，并检查端口形状与来源 slot 一致。
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
            let (_, held) = self.info(id);
            if declared != held {
                return Err(ModelError::Invalid(format!(
                    "stack entry {entry_index} (prefix `{prefix}`) wires input `{local_name}` of \
                     template `{template_name}`, declared {declared:?}, to slot `{}` of shape {held:?}",
                    self.name_of(id)
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

        // 1. 模板 slot：全局名 = 实例前缀 + 局部名。
        for decl in &template.slots {
            let name = format!("{prefix}.{}", decl.name);
            let dtype = self.dtype_of(decl.dtype.as_deref())?;
            let kind = parse_kind(&decl.kind)?;
            let shape = self.shape_of(&decl.shape)?;
            let origin = format!("instance `{prefix}`, template slot `{}`", decl.name);
            let id = self.add_slot(&name, dtype, shape, kind, &origin)?;
            local.insert(decl.name.clone(), id);
        }

        // 2. 节点：模板内顺序即发射顺序。
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

        // 3. 声明的 outputs 必须真的被产出来，否则下游接线会接空气。
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

            // §3.7 #1：节点的 `out` 只能引用模板已声明的 slot 或该实例声明的 `outputs`。
            // **不能**由编译器 infer 回填（旧实现继承本节点第一个输入的形状）：那会让无 GPU 的
            // L1 形状检查依赖"存在可解析的实现"，而真实描述是 bf16、reference provider 只有 f32 ——
            // 整条 L1 就废了，而 L1 正是这套架构存在的理由。
            let declared_slot = template.slots.iter().any(|decl| decl.name == *name);
            let id = match local.get(name) {
                Some(id) if declared_slot => {
                    // 权重不是算子的产物：写到 weight slot 上说明描述把两件事混了。
                    let (kind, _) = self.info(*id);
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
                    // 只剩一种合法来源：该实例声明的 `outputs`（首次被写时才分配 slot）。
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

    // ---- 辅助 -------------------------------------------------------------

    /// 全局名里的 slot：先按声明顺序分配，再登记名字与形状。
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

    fn name_of(&self, id: SlotId) -> String {
        self.names
            .get(id.0)
            .cloned()
            .unwrap_or_else(|| format!("slot#{}", id.0))
    }

    fn info(&self, id: SlotId) -> (SlotKind, Vec<i64>) {
        self.slot_info
            .get(id.0)
            .cloned()
            .unwrap_or((SlotKind::Activation, Vec::new()))
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

    /// 按列表下标选模板（§3.3：选择只按列表下标，不做算术）。
    fn select_template(
        &self,
        entry_index: usize,
        entry: &StackEntry,
        position: usize,
        index_var: Option<&str>,
    ) -> Result<String, ModelError> {
        let Some(select) = &entry.select else {
            return Ok(entry.template.clone());
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

    /// 替换 `{<索引变量>}` 与 `{last}`；还剩占位符就报错。
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

    /// 校验每条 `binding` 命中且只命中 weight slot（§3.5、§4.1 第 5 步）。
    fn bind(&self) -> Result<Vec<ResolvedBinding>, ModelError> {
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

        let unbound: Vec<&String> = weights
            .iter()
            .filter(|name| !claimed.contains_key(*name))
            .collect();
        if !unbound.is_empty() {
            let mut list = unbound
                .iter()
                .take(8)
                .map(|name| name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            if unbound.len() > 8 {
                list.push_str(&format!(", … (+{} more)", unbound.len() - 8));
            }
            return Err(ModelError::Invalid(format!(
                "{} weight slot(s) are not bound to any checkpoint tensor: {list}",
                unbound.len()
            )));
        }

        Ok(resolved)
    }
}

/// `kind` 的规范拼写。
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

/// `dtype` 的规范拼写与 [`RsDtype::name`] 一致（§3.6 #5）。
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
        };
        attrs.insert(key.clone(), value);
    }
    attrs
}

/// `transform` 词表（§3.4）：只认五个动词；参数语义属于加载器。
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

/// `select.by` 的语法：`<参数名>[<索引变量>]`。
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

/// `*` 匹配**恰好一个**点分段。
fn pattern_matches(pattern: &str, name: &str) -> bool {
    let pattern: Vec<&str> = pattern.split('.').collect();
    let name: Vec<&str> = name.split('.').collect();
    pattern.len() == name.len()
        && pattern
            .iter()
            .zip(name.iter())
            .all(|(p, n)| *p == "*" || p == n)
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
