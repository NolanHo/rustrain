//! A model description becomes one global `Plan` (`docs/design/model-description.md` §0, §3, §4.1).
//!
//! A model directory holds two files: `config.json` (the model's own hyper-parameters, as published
//! upstream) and `model.json` (the description: `params` / `templates` / `stack` / `binding`).
//! [`expand_dir`] turns the pair into **one** [`rustrain_plan::Plan`] — the global model, every
//! `layout` `Replicate`, every shape concrete, sharding recorded only as symbolic axes inside
//! [`ResolvedBinding`].
//!
//! Nothing here reads the environment, touches a device, or knows any operator: models are data.
//!
//! ```no_run
//! let expanded = rustrain_model::expand_dir(std::path::Path::new("model-dir"))?;
//! assert!(!expanded.plan.nodes.is_empty());
//! # Ok::<(), rustrain_model::ModelError>(())
//! ```

// The same trade-off `rustrain-plan` makes: `ModelError` carries paths, serde errors and
// `PlanError`'s candidate table, and those are exactly what makes a message actionable. Boxing it
// to shrink the `Err` would push a deref onto every `expand`, and an expand runs once.
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

/// Everything that can fail in the description layer.
///
/// Contract §3.6 #8: all six error paths must "exit non-zero with a stderr message naming the
/// name / pattern / path", never panic. Hence `Result` everywhere and no `unwrap`.
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

/// A loaded model directory.
pub struct Model {
    pub dir: PathBuf,
    pub desc: ModelDesc,
    /// `config.json` kept verbatim; `params.*.from` reads values out of it by dotted path.
    pub config: serde_json::Value,
}

impl Model {
    /// Read `config.json` and `model.json` from a model directory.
    pub fn load(dir: &Path) -> Result<Self, ModelError> {
        let config: serde_json::Value = read_json(&dir.join(CONFIG_FILE))?;
        let desc: ModelDesc = read_json(&dir.join(DESC_FILE))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            desc,
            config,
        })
    }

    /// Expand into the global plan.
    pub fn expand(&self) -> Result<Expanded, ModelError> {
        expand::expand(&self.desc, &self.config)
    }
}

/// Convenience entry point: `expand_dir(dir)` = `Model::load(dir)?.expand()`.
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
