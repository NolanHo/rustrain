//! 模型描述 → 全局 Plan（`docs/design/model-description.md` §0、§3、§4.1）。
//!
//! 一个模型目录里有两个文件：`config.json`（模型自己的超参，来自公开仓库）与 `model.json`
//! （描述：`params` / `templates` / `stack` / `binding`）。[`expand_dir`] 把两者变成**一个**
//! [`rustrain_plan::Plan`] —— 全局模型，`layout` 全 `Replicate`，形状全量，切分只以符号轴的形式
//! 记录在 [`ResolvedBinding`] 里。
//!
//! 这里不读环境变量、不碰设备、不认识任何算子：模型是数据。
//!
//! ```no_run
//! let expanded = rustrain_model::expand_dir(std::path::Path::new("model-dir"))?;
//! assert!(!expanded.plan.nodes.is_empty());
//! # Ok::<(), rustrain_model::ModelError>(())
//! ```

// 与 `rustrain-plan` 同样的取舍：`ModelError` 里带着路径、serde 错误与 `PlanError` 的候选表，
// 那些正是让报错可操作的东西。为了缩小 `Err` 而装箱，等于给每次 `expand` 加一次解引用，
// 而 expand 一次只跑一遍。
#![allow(clippy::result_large_err)]

mod desc;
mod expand;
mod params;

use std::path::{Path, PathBuf};

pub use desc::{
    AttrLiteral, Binding, CONFIG_FILE, DESC_FILE, ExprSpec, FORMAT, FromSpec, ModelDesc, NodeDecl,
    ParamSpec, PortSpec, Select, SlotDecl, Split, StackEntry, Target, Template,
};
pub use expand::{Expanded, ResolvedBinding, ResolvedBindingSlot, ResolvedSplit, expand};

/// 描述层的一切失败。
///
/// 契约 §3.6 #8：六条错误路径都必须"非 0 退出 + stderr 说明名字 / 模式 / 路径"，不得 panic。
/// 所以这里只有 `Result`，没有 `unwrap`。
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("cannot read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("cannot parse {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("unsupported description format `{found}`; this build reads `{expected}`")]
    Format { found: String, expected: String },

    #[error("{0}")]
    Invalid(String),

    #[error(transparent)]
    Plan(#[from] rustrain_plan::PlanError),
}

/// 一个已加载的模型目录。
pub struct Model {
    pub dir: PathBuf,
    pub desc: ModelDesc,
    /// `config.json` 原样保留；`params.*.from` 按点分路径从这里取值。
    pub config: serde_json::Value,
}

impl Model {
    /// 读模型目录下的 `config.json` 与 `model.json`。
    pub fn load(dir: &Path) -> Result<Self, ModelError> {
        let config: serde_json::Value = read_json(&dir.join(CONFIG_FILE))?;
        let desc: ModelDesc = read_json(&dir.join(DESC_FILE))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            desc,
            config,
        })
    }

    /// 展开成全局 Plan。
    pub fn expand(&self) -> Result<Expanded, ModelError> {
        expand::expand(&self.desc, &self.config)
    }
}

/// 便捷入口：`expand_dir(dir)` = `Model::load(dir)?.expand()`。
pub fn expand_dir(dir: &Path) -> Result<Expanded, ModelError> {
    Model::load(dir)?.expand()
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, ModelError> {
    let text = std::fs::read_to_string(path).map_err(|source| ModelError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| ModelError::Json {
        path: path.to_path_buf(),
        source,
    })
}
