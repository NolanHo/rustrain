//! The serde types of a description file: the four sections of `docs/design/model-description.md`
//! §3 plus the §3.6 rulings.
//!
//! This layer checks *shape* only (key names, types, required fields); all semantics live in
//! [`crate::expand`]. Every struct is `deny_unknown_fields`: a misspelled key must fail loudly
//! rather than be silently dropped (invariant I-5).

use std::collections::BTreeMap;

use serde::Deserialize;

/// The description format identifier (§3.6 #1: the file always sits at `model.json` in the model
/// directory).
pub const FORMAT: &str = "rustrain.model.v1";
/// The description file name, a sibling of `config.json`.
pub const DESC_FILE: &str = "model.json";
/// The model's own config file; `params.*.from` reads values out of it.
pub const CONFIG_FILE: &str = "config.json";

/// One model description.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDesc {
    pub format: String,
    pub name: String,
    /// Default dtype, used by template slots and ports that declare none (§3.6 #6).
    #[serde(default)]
    pub dtype: Option<String>,
    /// The model's external inputs. Same shape as `templates.*.inputs` (§3.6 #2).
    #[serde(default)]
    pub inputs: BTreeMap<String, PortSpec>,
    pub params: BTreeMap<String, ParamSpec>,
    pub templates: BTreeMap<String, Template>,
    pub stack: Vec<StackEntry>,
    #[serde(default)]
    pub binding: Vec<Binding>,
    /// Checkpoint tensors this description deliberately does not consume (C5, §3.5's second
    /// mandate). The pattern syntax is `binding.source`'s, plus `**` for any number of segments:
    /// the vision tower is dropped by writing `"model.visual.**"`, never silently.
    #[serde(default)]
    pub ignore: Vec<String>,
}

/// A declared port (input or output).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortSpec {
    pub shape: Vec<String>,
    pub kind: String,
    #[serde(default)]
    pub dtype: Option<String>,
}

/// One `params` value: read from `config.json`, a parameter expression, or a literal list (§3.1).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ParamSpec {
    /// A literal list; per-layer type lists are its typical use.
    List(Vec<String>),
    From(FromSpec),
    Expr(ExprSpec),
}

/// `{"from": "text_config.hidden_size", "default": 4}`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FromSpec {
    pub from: String,
    #[serde(default)]
    pub default: Option<i64>,
}

/// `{"expr": "2 * heads * head_dim"}`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExprSpec {
    pub expr: String,
}

/// A named subgraph: math and wiring only, sharding lives in `binding` (§3.2).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Template {
    #[serde(default)]
    pub inputs: BTreeMap<String, PortSpec>,
    #[serde(default)]
    pub outputs: BTreeMap<String, PortSpec>,
    #[serde(default)]
    pub slots: Vec<SlotDecl>,
    #[serde(default)]
    pub nodes: Vec<NodeDecl>,
}

/// A slot inside a template; the name is local to that template.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlotDecl {
    pub name: String,
    pub kind: String,
    pub shape: Vec<String>,
    #[serde(default)]
    pub dtype: Option<String>,
}

/// A node inside a template.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeDecl {
    pub op: String,
    #[serde(default, rename = "in")]
    pub inputs: Vec<String>,
    #[serde(rename = "out")]
    pub outputs: Vec<String>,
    #[serde(default)]
    pub attrs: BTreeMap<String, AttrLiteral>,
}

/// A node attribute literal. Attributes take literals only, never parameter references (the
/// contract defines no parameterised attributes).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AttrLiteral {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    /// A list of integers: `reshape` takes its target shape as a list (`rustrain-kernels` reads
    /// `shape` as an `i64` list), and a list of literals is still a literal — no parameter
    /// reference is involved (§3.7 #5).
    I64s(Vec<i64>),
}

/// One `stack` item: a template invocation with arguments (§3.3).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StackEntry {
    /// The template to instantiate when the entry has no `select`; required exactly then, because
    /// `select` and `template` would otherwise be two sources for the same fact (§3.7 #13).
    #[serde(default)]
    pub template: Option<String>,
    /// Instance prefix; `{l}` / `{last}` are substituted by the instantiator.
    pub prefix: String,
    #[serde(default)]
    pub repeat: Option<Repeat>,
    /// Expand to the count of a parameter (0 = no instances); the index variable is fixed at `l`.
    #[serde(default)]
    pub until: Option<String>,
    #[serde(default)]
    pub select: Option<Select>,
    /// `{local name: global name}`; the default is chaining (the previous instance's outputs).
    #[serde(default, rename = "inputs")]
    pub inputs: Option<BTreeMap<String, String>>,
}

/// `{"count": "layers", "index": "l"}`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Repeat {
    pub count: String,
    pub index: String,
}

/// Pick a template by list index (§3.3: selection goes by list index only, never arithmetic).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Select {
    /// `<param name>[<index variable>]`.
    pub by: String,
    #[serde(default)]
    pub cases: BTreeMap<String, String>,
    #[serde(default)]
    pub default: Option<String>,
}

/// One parameter mapping (§3.4). `slot` and `split`+`targets` are mutually exclusive.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    #[serde(default)]
    pub slot: Option<String>,
    pub source: String,
    #[serde(default)]
    pub transform: Vec<String>,
    /// slot dimension → symbolic axis names. The global plan is all `Replicate`; the axes are only
    /// resolved once `instantiate` has a mesh.
    #[serde(default)]
    pub axes: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub split: Option<Split>,
    #[serde(default)]
    pub targets: Vec<Target>,
}

/// One segment of a fused-storage split.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Split {
    pub dim: i64,
    pub sizes: Vec<String>,
}

/// One target slot of a `split`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub slot: String,
    #[serde(default)]
    pub axes: BTreeMap<String, Vec<String>>,
}
