//! [`RegisteredOp`] — the framework's handle on one operator implementation.

use std::fmt;
use std::path::{Path, PathBuf};

use rustrain_abi::loader::{LoadedOp, Plugin};
use rustrain_abi::{
    RsBackwardKind, RsCollective, RsDeviceKind, RsDtype, RsExpansion, RsOpDesc, RsRequires,
    RsShardRule,
};
use serde::{Deserialize, Serialize};

use crate::capability::{backward_name, declared_device, declares_nothing};
use crate::vocab::is_composite_op;

/// Where the descriptor behind a [`RegisteredOp`] came from.
///
/// Production registrations always carry a loaded [`Plugin`]: the only public
/// way into a [`crate::Registry`] is [`crate::Registry::add_plugin`], and the
/// handle keeps the plugin (and therefore the mapped `.so`) alive. The detached
/// variant exists so that this crate's own tests can register descriptors that
/// were never compiled into a `.so`; it is `#[cfg(test)]`, so it cannot be
/// constructed — or observed — from production code.
#[derive(Clone, Debug)]
pub(crate) enum PluginSlot {
    Loaded(Plugin),
    #[cfg(test)]
    Detached(std::sync::Arc<PluginStamp>),
}

/// Identity of a plugin that has no `.so` behind it (tests only).
#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct PluginStamp {
    pub name: String,
    pub version: String,
}

impl PluginSlot {
    pub(crate) fn name(&self) -> &str {
        match self {
            PluginSlot::Loaded(p) => p.name(),
            #[cfg(test)]
            PluginSlot::Detached(s) => &s.name,
        }
    }

    pub(crate) fn version(&self) -> &str {
        match self {
            PluginSlot::Loaded(p) => p.version(),
            #[cfg(test)]
            PluginSlot::Detached(s) => &s.version,
        }
    }

    fn identity(&self) -> String {
        format!("{}@{}", self.name(), self.version())
    }
}

/// Where the descriptor comes from. Both arms keep the owning plugin mapped,
/// so the descriptor outlives the handle without any pointer arithmetic of our
/// own: a `LoadedOp` holds the `Arc<PluginInner>` that owns the library, and a
/// test descriptor is leaked.
#[derive(Clone)]
enum DescSource {
    Loaded(LoadedOp),
    #[cfg(test)]
    Detached(&'static RsOpDesc),
}

/// A cloneable handle on one operator implementation published by a plugin.
///
/// The handle owns (an `Arc` of) the plugin, which is what keeps the descriptor
/// and every string in it alive; cloning is cheap and never borrows from the
/// registry, so a resolved handle can outlive the `Registry` it came from.
/// Accessors hand out references tied to the handle, not to the registry — and
/// not `'static`, because a handle can be dropped, and with it the last
/// reference to the plugin.
#[derive(Clone)]
pub struct RegisteredOp {
    desc: DescSource,
    plugin: PluginSlot,
    origin: PathBuf,
}

impl RegisteredOp {
    /// Handle for a descriptor that has no plugin behind it (tests only).
    #[cfg(test)]
    pub(crate) fn new(desc: &'static RsOpDesc, plugin: PluginSlot, origin: PathBuf) -> Self {
        Self {
            desc: DescSource::Detached(desc),
            plugin,
            origin,
        }
    }

    /// Builds a handle from a descriptor published by a loaded plugin.
    pub(crate) fn from_loaded(op: LoadedOp, plugin: Plugin, origin: PathBuf) -> Self {
        Self {
            desc: DescSource::Loaded(op),
            plugin: PluginSlot::Loaded(plugin),
            origin,
        }
    }

    fn descriptor(&self) -> &RsOpDesc {
        match &self.desc {
            DescSource::Loaded(op) => op.desc(),
            #[cfg(test)]
            DescSource::Detached(desc) => desc,
        }
    }

    /// Operator name, e.g. `rmsnorm`. This is the fixed vocabulary word from
    /// spec §2.4, not an implementation name.
    pub fn name(&self) -> &str {
        // SAFETY: the loader rejects descriptors with a null name, and the
        // string lives in the plugin.
        unsafe { cstr(self.descriptor().id.name) }.unwrap_or("<unnamed>")
    }

    /// Variant name, e.g. `cuda.fp8_block128`. The recipe spells it as part of
    /// `op@variant`.
    pub fn variant(&self) -> &str {
        // SAFETY: as `name`; the loader rejects a null variant.
        unsafe { cstr(self.descriptor().id.variant) }.unwrap_or("<unknown>")
    }

    /// `op@variant` — the exact string a recipe uses to name this
    /// implementation.
    pub fn spec_name(&self) -> String {
        format!("{}@{}", self.name(), self.variant())
    }

    /// The raw ABI descriptor. Shape/type inference, memory requirements,
    /// expansion and the call itself all live here.
    pub fn desc(&self) -> &RsOpDesc {
        self.descriptor()
    }

    /// Variant version declared by the plugin (ABI `rs_op_id.version`).
    pub fn version(&self) -> u32 {
        self.descriptor().id.version
    }

    /// The loaded plugin that published this operator.
    ///
    /// There is always one in a production build: [`crate::Registry::add_plugin`]
    /// is the only public way in.
    ///
    /// # Panics
    /// Only on a handle created by this crate's internal test-registration
    /// path, which has no plugin behind it by construction. Use
    /// [`RegisteredOp::plugin_identity`] when only the identity is needed: it
    /// works for both.
    pub fn plugin(&self) -> &Plugin {
        match &self.plugin {
            PluginSlot::Loaded(plugin) => plugin,
            #[cfg(test)]
            PluginSlot::Detached(_) => panic!(
                "{} has no loaded plugin: it was registered through the crate-internal test \
                 path; use plugin_identity() or plugin_name() instead",
                self.spec_name()
            ),
        }
    }

    /// `plugin@version` — stable identity for manifests and plan digests.
    pub fn plugin_identity(&self) -> String {
        self.plugin.identity()
    }

    pub fn plugin_name(&self) -> &str {
        self.plugin.name()
    }

    pub fn plugin_version(&self) -> &str {
        self.plugin.version()
    }

    /// Path of the `.so` this operator was loaded from.
    pub fn origin(&self) -> &Path {
        &self.origin
    }

    /// Environment requirements, or `None` when the plugin declared none
    /// (`rs_op_desc.requires` is null).
    pub fn requires(&self) -> Option<&RsRequires> {
        let requires = self.descriptor().requires;
        if requires.is_null() {
            None
        } else {
            // SAFETY: non-null and owned by the plugin, which the handle keeps
            // alive.
            unsafe { requires.as_ref() }
        }
    }

    /// Collectives this variant performs internally (contract S-2).
    /// How this implementation says its sharded distribution propagates. The
    /// plan's derivation algebra reads it through `shard::ShardRules`; two
    /// implementations of one operator must agree.
    pub fn shard(&self) -> RsShardRule {
        self.descriptor().shard
    }

    pub fn collectives(&self) -> &[RsCollective] {
        let desc = self.descriptor();
        if desc.collectives.is_null() || desc.n_collectives == 0 {
            &[]
        } else {
            // SAFETY: non-null with a length, owned by the plugin.
            unsafe { std::slice::from_raw_parts(desc.collectives, desc.n_collectives as usize) }
        }
    }

    /// Declared primitive expansion, if any (contract R-4).
    pub fn expansion(&self) -> Option<&RsExpansion> {
        let expansion = self.descriptor().expansion;
        if expansion.is_null() {
            None
        } else {
            // SAFETY: non-null and owned by the plugin.
            unsafe { expansion.as_ref() }
        }
    }

    /// One-line description written by the plugin author.
    pub fn doc(&self) -> &str {
        // SAFETY: `doc` is either null (handled) or a NUL-terminated string
        // owned by the plugin this handle keeps mapped.
        unsafe { cstr(self.descriptor().doc) }.unwrap_or("")
    }

    /// How the framework obtains gradients for this variant.
    pub fn backward_kind(&self) -> RsBackwardKind {
        self.descriptor().backward
    }

    /// The registered operator that implements this variant's backward pass,
    /// as `(name, variant)` — present only when `backward == EXPLICIT`.
    ///
    /// Resolution returns the *forward* handle for `Phase::Backward` (the
    /// descriptor is what declares how the gradient is produced); a consumer
    /// that wants the executing forward/backward kernel follows this pair and
    /// resolves it like any other operator.
    pub fn backward_op_id(&self) -> Option<(String, String)> {
        let backward_op = self.descriptor().backward_op;
        if self.backward_kind() != RsBackwardKind::EXPLICIT || backward_op.name.is_null() {
            return None;
        }
        // SAFETY: non-null and owned by the plugin; variant may be null, which
        // `cstr` maps to `None`.
        let name = unsafe { cstr(backward_op.name) }?;
        let variant = unsafe { cstr(backward_op.variant) }.unwrap_or("");
        Some((name.to_string(), variant.to_string()))
    }

    /// True when the variant publishes a non-empty primitive expansion.
    pub fn has_expansion(&self) -> bool {
        self.expansion().is_some_and(|e| e.n_nodes > 0)
    }

    /// True when the operator name is a composite/model block in the fixed
    /// vocabulary of spec §2.4 — i.e. an operator that *must* declare an
    /// expansion (contract R-4).
    pub fn is_composite(&self) -> bool {
        is_composite_op(self.name())
    }

    /// The device this variant declares through its namespace, if any. See
    /// [`declared_device`].
    pub fn declared_device(&self) -> Option<RsDeviceKind> {
        declared_device(self.variant())
    }

    /// Summary row for `rustrain ops list` and run manifests.
    pub fn summary(&self) -> OpSummary {
        OpSummary::of(self)
    }
}

impl PartialEq for RegisteredOp {
    /// Two handles are the same implementation when they name the same variant
    /// published by the same plugin version.
    fn eq(&self, other: &Self) -> bool {
        self.spec_name() == other.spec_name()
            && self.plugin_identity() == other.plugin_identity()
            && self.origin == other.origin
    }
}

impl Eq for RegisteredOp {}

impl fmt::Debug for RegisteredOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredOp")
            .field("spec", &self.spec_name())
            .field("plugin", &self.plugin_identity())
            .field("origin", &self.origin)
            .field("backward", &backward_name(self.backward_kind()))
            .field("has_expansion", &self.has_expansion())
            .finish()
    }
}

/// Plain, serialisable summary of one registered operator.
///
/// Deliberately free of ABI pointers and Rust enums: this is what ends up in
/// `manifest.json` and in report output, so it must stay stable and diffable.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct OpSummary {
    /// Operator name from the fixed vocabulary, e.g. `rmsnorm`.
    pub op: String,
    /// Variant name, e.g. `cuda.fp8_block128`.
    pub variant: String,
    /// `plugin@version` that published it.
    pub plugin: String,
    /// Path of the `.so` it was loaded from.
    pub plugin_origin: String,
    /// Accepted dtypes, by name, in ABI order.
    pub dtypes: Vec<String>,
    /// `AUTODIFF` | `EXPLICIT` | `NONDIFF`.
    pub backward: String,
    /// The variant publishes a primitive expansion (contract R-4).
    pub has_expansion: bool,
    /// The operator name is a composite/model block in the spec §2.4
    /// vocabulary, and therefore *must* declare an expansion.
    pub is_composite: bool,
}

impl OpSummary {
    pub fn of(op: &RegisteredOp) -> Self {
        // Report the *effective* accepted set: a variant that declared no
        // constraints accepts every dtype the ABI knows, and saying "no
        // dtypes" here would read as "accepts nothing" — the opposite.
        let dtypes: Vec<String> = match op.requires() {
            Some(requires) if !declares_nothing(requires) => requires
                .dtypes()
                .into_iter()
                .map(|d| d.name().to_string())
                .collect(),
            _ => RsDtype::ALL
                .into_iter()
                .map(|d| d.name().to_string())
                .collect(),
        };
        Self {
            op: op.name().to_string(),
            variant: op.variant().to_string(),
            plugin: op.plugin_identity(),
            plugin_origin: op.origin().display().to_string(),
            dtypes,
            backward: backward_name(op.backward_kind()).to_string(),
            has_expansion: op.has_expansion(),
            is_composite: op.is_composite(),
        }
    }
}

/// Reads a NUL-terminated C string.
///
/// # Safety
/// `p` must be null or point to a NUL-terminated string that stays alive for
/// the returned borrow.
unsafe fn cstr<'a>(p: *const std::ffi::c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    // SAFETY: forwarded from the caller.
    unsafe { std::ffi::CStr::from_ptr(p) }.to_str().ok()
}
