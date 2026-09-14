//! rustrain operator plugin ABI, version 1.
//!
//! This crate is the only contract between the framework and a kernel plugin.
//! It deliberately has no dependency on `tch`, libtorch or CUDA (invariant I-1
//! in `docs/design/kernel-first/spec.md`), which is what allows a kernel to be
//! swapped without recompiling the framework.
//!
//! Layout safety: every `#[repr(C)]` type here mirrors a struct in
//! `include/rustrain_op.h`. The layouts are not checked by the compiler across
//! the two languages, so [`tests::layout`] pins the sizes and offsets, and the
//! C test plugin exercises the real thing end to end.

pub mod author;
pub mod error;
pub mod ffi;
pub mod loader;

pub use author::{OpSpec, PluginBuilder};
pub use error::AbiError;
pub use ffi::*;
pub use loader::{LoadedOp, Plugin};

/// ABI version understood by this build. A plugin reporting anything else is
/// rejected at load time (contract C-2).
pub const ABI_VERSION: u32 = 2;

/// Name of the single symbol a plugin must export (contract C-1).
pub const PLUGIN_SYMBOL: &[u8] = b"rustrain_plugin_v1\0";
