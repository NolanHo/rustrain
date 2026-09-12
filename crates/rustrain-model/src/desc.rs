//! 描述文件的 serde 类型：`docs/design/model-description.md` §3 的四个部分 + §3.6 的裁定。
//!
//! 这一层只做「形状」的检查（键名、类型、必填），语义全在 [`crate::expand`] 里。
//! 所有结构都 `deny_unknown_fields`：拼错的键必须报错，不能被静默忽略（不变式 I-5）。

use std::collections::BTreeMap;

use serde::Deserialize;

/// 描述格式标识（§3.6 #1：文件固定在模型目录下的 `model.json`）。
pub const FORMAT: &str = "rustrain.model.v1";
/// 描述文件名，与 `config.json` 同级。
pub const DESC_FILE: &str = "model.json";
/// 模型自身的配置文件；`params.*.from` 从这里取。
pub const CONFIG_FILE: &str = "config.json";

/// 一份模型描述。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDesc {
    pub format: String,
    pub name: String,
    /// 缺省 dtype；模板 slot 与端口没写 dtype 时用它（§3.6 #6）。
    #[serde(default)]
    pub dtype: Option<String>,
    /// 模型的外部输入。与 `templates.*.inputs` 同构（§3.6 #2）。
    #[serde(default)]
    pub inputs: BTreeMap<String, PortSpec>,
    pub params: BTreeMap<String, ParamSpec>,
    pub templates: BTreeMap<String, Template>,
    pub stack: Vec<StackEntry>,
    #[serde(default)]
    pub binding: Vec<Binding>,
}

/// 一个端口（输入或输出）的声明。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortSpec {
    pub shape: Vec<String>,
    pub kind: String,
    #[serde(default)]
    pub dtype: Option<String>,
}

/// `params` 的一个值：来自 `config.json`、参数表达式、或一个字面列表（§3.1）。
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ParamSpec {
    /// 列表值，逐层类型就是它的典型用法。
    List(Vec<String>),
    From(FromSpec),
    Expr(ExprSpec),
}

/// `{"from": "text_config.hidden_size", "default": 4}`。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FromSpec {
    pub from: String,
    #[serde(default)]
    pub default: Option<i64>,
}

/// `{"expr": "2 * heads * head_dim"}`。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExprSpec {
    pub expr: String,
}

/// 一个具名子图：只有数学与连接，切分住在 `binding`（§3.2）。
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

/// 模板里的一个 slot：名字是模板内局部名。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlotDecl {
    pub name: String,
    pub kind: String,
    pub shape: Vec<String>,
    #[serde(default)]
    pub dtype: Option<String>,
}

/// 模板里的一个节点。
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

/// 节点属性的字面量。属性只接受字面量，不接参数引用（契约没定义参数化属性）。
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AttrLiteral {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

/// `stack` 的一项：一次带实参的模板调用（§3.3）。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StackEntry {
    /// 无 `select` 时实例化的模板；有 `select` 时是兜底名字。
    pub template: String,
    /// 实例前缀，`{l}` / `{last}` 由实例化器替换。
    pub prefix: String,
    #[serde(default)]
    pub repeat: Option<Repeat>,
    /// 按参数计数展开（0 = 不展开）；索引变量固定为 `l`。
    #[serde(default)]
    pub until: Option<String>,
    #[serde(default)]
    pub select: Option<Select>,
    /// `{局部名: 全局名}`；缺省是链式（上一实例的 outputs）。
    #[serde(default, rename = "inputs")]
    pub inputs: Option<BTreeMap<String, String>>,
}

/// `{"count": "layers", "index": "l"}`。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Repeat {
    pub count: String,
    pub index: String,
}

/// 按列表下标选模板（§3.3：选择只按列表下标，不做算术）。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Select {
    /// `<参数名>[<索引变量>]`。
    pub by: String,
    #[serde(default)]
    pub cases: BTreeMap<String, String>,
    #[serde(default)]
    pub default: Option<String>,
}

/// 一条参数映射（§3.4）。`slot` 与 `split`+`targets` 二选一。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    #[serde(default)]
    pub slot: Option<String>,
    pub source: String,
    #[serde(default)]
    pub transform: Vec<String>,
    /// slot 维度 → 符号轴名。全局 Plan 全 `Replicate`，轴要到 instantiate 才解析。
    #[serde(default)]
    pub axes: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub split: Option<Split>,
    #[serde(default)]
    pub targets: Vec<Target>,
}

/// 融合存储的一段切分。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Split {
    pub dim: i64,
    pub sizes: Vec<String>,
}

/// `split` 的一个目标 slot。
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub slot: String,
    #[serde(default)]
    pub axes: BTreeMap<String, Vec<String>>,
}
