//! Error type for ABI loading and dispatch.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum AbiError {
    #[error("failed to open plugin {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },

    #[error("plugin {path} does not export {symbol}: {source}")]
    MissingSymbol {
        path: PathBuf,
        symbol: String,
        #[source]
        source: libloading::Error,
    },

    #[error("plugin {path} returned a null plugin descriptor")]
    NullDescriptor { path: PathBuf },

    #[error(
        "plugin {path} reports ABI version {found}, this build implements {expected}; \
         rebuild the plugin against the current header"
    )]
    VersionMismatch {
        path: PathBuf,
        found: u32,
        expected: u32,
    },

    #[error(
        "plugin {path} descriptor is too small: struct_size {found} < {expected}; \
         the plugin was built against an older header"
    )]
    DescriptorTooSmall {
        path: PathBuf,
        found: u32,
        expected: u32,
    },

    #[error("plugin {path} declares {count} op(s) but the op table is null")]
    NullOpTable { path: PathBuf, count: u32 },

    #[error("plugin {path} op #{index} has a null descriptor")]
    NullOp { path: PathBuf, index: usize },

    #[error("plugin {path} op #{index} is missing a name")]
    OpWithoutName { path: PathBuf, index: usize },

    #[error("plugin {path} op #{index} is missing a variant")]
    OpWithoutVariant { path: PathBuf, index: usize },

    #[error("plugin {path} op `{op}` ({variant}) has no execute function")]
    OpWithoutExecute {
        path: PathBuf,
        op: String,
        variant: String,
    },

    #[error("plugin {path} op `{op}` ({variant}) is ABI version {found}, expected {expected}")]
    OpVersionMismatch {
        path: PathBuf,
        op: String,
        variant: String,
        found: u32,
        expected: u32,
    },

    #[error("plugin {path} declares a composite op `{op}` ({variant}) with no expansion")]
    CompositeWithoutExpansion {
        path: PathBuf,
        op: String,
        variant: String,
    },

    #[error("plugin {path} op `{op}` ({variant}) declares an explicit backward op with an empty name")]
    ExplicitBackwardWithoutOp {
        path: PathBuf,
        op: String,
        variant: String,
    },

    #[error("plugin `{name}` reported a non-zero status from init(): {status}")]
    InitFailed { name: String, status: i32 },

    #[error("op `{op}` ({variant}) execute() failed with status {status}: {message}")]
    ExecuteFailed {
        op: String,
        variant: String,
        status: i32,
        message: String,
    },

    #[error("op `{op}` ({variant}) shape inference failed with status {status}")]
    InferFailed {
        op: String,
        variant: String,
        status: i32,
    },

    #[error("op `{op}` ({variant}) memory query failed with status {status}")]
    MemoryFailed {
        op: String,
        variant: String,
        status: i32,
    },

    #[error("plugin `{name}` is already loaded from {existing}; refusing to load a second copy from {requested}")]
    DuplicatePlugin {
        name: String,
        existing: PathBuf,
        requested: PathBuf,
    },
}

pub type Result<T> = std::result::Result<T, AbiError>;
