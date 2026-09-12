//! Loading a plugin `.so` and validating what it publishes.
//!
//! Loading is strict by design (contracts C-1, C-2 and R-1): a plugin that
//! reports the wrong ABI version, omits the entry symbol, or publishes a
//! malformed descriptor is rejected here rather than misbehaving later.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{AbiError, Result};
use crate::ffi::*;
use crate::{ABI_VERSION, PLUGIN_SYMBOL};

/// Shared state of a loaded plugin. `Arc`-ed so that handles taken from it keep
/// the underlying library mapped.
pub struct PluginInner {
    /// Dropping this unmaps the `.so`; every descriptor borrowed from it
    /// becomes dangling, which is why handles hold an `Arc<PluginInner>`.
    /// `None` for an in-process plugin — one whose descriptors were built by
    /// `PluginBuilder` in this binary rather than loaded from disk. Those live
    /// as long as the process and need no keeper.
    _library: Option<libloading::Library>,
    /// Validated descriptors, in declaration order. Keeping them here (rather
    /// than re-walking the plugin's table) means the null-slot and null-table
    /// checks happen exactly once, at load time.
    ops: Vec<&'static RsOpDesc>,
    origin: PathBuf,
    name: String,
    version: String,
}

/// A loaded plugin.
#[derive(Clone)]
pub struct Plugin {
    inner: Arc<PluginInner>,
}

// SAFETY: a plugin is process-global immutable data. `libloading::Library` is
// already `Send + Sync`, the library stays mapped as long as any handle holds
// the `Arc`, and descriptors are published once and never mutated afterwards.
// Whether a plugin's `execute` may be *called* concurrently is a separate
// question, covered by that method's safety contract rather than by this one.
unsafe impl Send for PluginInner {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for PluginInner {}

impl Plugin {
    /// Opens `path`, resolves the entry symbol and validates the plugin header.
    ///
    /// `services` is handed to the plugin's optional `init` hook (contract C-3).
    ///
    /// # Safety
    /// The caller must ensure `services`, if supplied, stays alive for as long
    /// as the plugin may be used — the plugin is allowed to cache the pointer
    /// during `init` and read the table later. Every function pointer in it
    /// must be valid for the process lifetime.
    pub unsafe fn load(path: impl AsRef<Path>, services: Option<&RsServices>) -> Result<Self> {
        let path = path.as_ref();
        let origin = path.to_path_buf();

        // SAFETY: `path` is caller-supplied; the library is kept alive inside
        // `PluginInner` for as long as any handle exists.
        let library = unsafe { libloading::Library::new(path) }.map_err(|source| AbiError::Open {
            path: origin.clone(),
            source,
        })?;

        let entry: libloading::Symbol<RustrainPluginV1Fn> =
            // SAFETY: symbol lookup; the signature is the ABI's own.
            unsafe { library.get(PLUGIN_SYMBOL) }.map_err(|source| AbiError::MissingSymbol {
                path: origin.clone(),
                symbol: "rustrain_plugin_v1".to_string(),
                source,
            })?;

        // SAFETY: the plugin contract says this returns a pointer to a
        // process-lifetime `RsPlugin` or null.
        let raw = unsafe { entry() };
        if raw.is_null() {
            return Err(AbiError::NullDescriptor { path: origin });
        }

        // SAFETY: non-null and, per the ABI contract, valid for the process
        // lifetime. The `'static` is contained rather than exposed: every
        // handle owns the `Arc<PluginInner>` that keeps `library` mapped, and
        // no accessor returns a borrow that outlives `&self`.
        let plugin: &'static RsPlugin = unsafe { &*raw };

        if plugin.abi_version != ABI_VERSION {
            return Err(AbiError::VersionMismatch {
                path: origin,
                found: plugin.abi_version,
                expected: ABI_VERSION,
            });
        }
        // Checked before any field past the two-word prefix is read: a plugin
        // built against an older header may be shorter than this build's
        // `RsPlugin`, and reading past its end is the one thing we cannot
        // recover from.
        let min_size = std::mem::size_of::<RsPlugin>() as u32;
        if plugin.struct_size < min_size {
            return Err(AbiError::DescriptorTooSmall {
                path: origin,
                found: plugin.struct_size,
                expected: min_size,
            });
        }

        let name = unsafe { cstr(plugin.plugin_name) }
            .unwrap_or("<unnamed>")
            .to_string();
        let version = unsafe { cstr(plugin.plugin_version) }
            .unwrap_or("<unknown>")
            .to_string();

        // Validation runs before `init` so that a plugin we are going to reject
        // never gets to execute code: `init` may spawn threads or register
        // callbacks that would dangle once the rejected library is unmapped.
        let ops = parse_op_table(&origin, plugin)?;
        for (index, desc) in ops.iter().enumerate() {
            validate_op(&origin, index, desc)?;
        }

        if let Some(init) = plugin.init {
            // SAFETY: the plugin owns this function pointer. A null table means
            // "no services", which is what the caller asked for by passing
            // `None`; a plugin whose `init` cannot cope with that must declare
            // its requirement rather than rely on this call.
            let status =
                unsafe { init(services.map_or(std::ptr::null_mut(), |s| s as *const _ as *mut _)) };
            if status != 0 {
                return Err(AbiError::InitFailed { name, status });
            }
        }

        Ok(Self {
            inner: Arc::new(PluginInner {
                _library: Some(library),
                ops,
                origin,
                name,
                version,
            }),
        })
    }

    /// Registers a plugin whose descriptors are already in this binary.
    ///
    /// This is the same validation path as [`Plugin::load`] minus the dynamic
    /// linking, which makes a Rust-authored provider (see
    /// [`crate::author::PluginBuilder`]) usable without shipping a `.so`: the
    /// framework's own built-in operators, and any test that wants a real
    /// registered implementation, take this route.
    ///
    /// # Safety
    /// `plugin` must point at a valid `RsPlugin` whose descriptors, strings and
    /// expansion arrays stay alive for the rest of the process — which is what
    /// `PluginBuilder::build` guarantees by leaking them.
    pub unsafe fn from_static(
        plugin: &'static RsPlugin,
        origin: impl Into<PathBuf>,
    ) -> Result<Self> {
        let origin = origin.into();

        if plugin.abi_version != ABI_VERSION {
            return Err(AbiError::VersionMismatch {
                path: origin,
                found: plugin.abi_version,
                expected: ABI_VERSION,
            });
        }
        let min_size = std::mem::size_of::<RsPlugin>() as u32;
        if plugin.struct_size < min_size {
            return Err(AbiError::DescriptorTooSmall {
                path: origin,
                found: plugin.struct_size,
                expected: min_size,
            });
        }

        let name = unsafe { cstr(plugin.plugin_name) }
            .unwrap_or("<unnamed>")
            .to_string();
        let version = unsafe { cstr(plugin.plugin_version) }
            .unwrap_or("<unknown>")
            .to_string();

        let ops = parse_op_table(&origin, plugin)?;
        for (index, desc) in ops.iter().enumerate() {
            validate_op(&origin, index, desc)?;
        }

        Ok(Self {
            inner: Arc::new(PluginInner {
                _library: None,
                ops,
                origin,
                name,
                version,
            }),
        })
    }

    /// Every operator this plugin publishes.
    pub fn ops(&self) -> Vec<LoadedOp> {
        self.inner
            .ops
            .iter()
            .map(|&desc| LoadedOp {
                desc,
                owner: Arc::clone(&self.inner),
            })
            .collect()
    }

    pub fn name(&self) -> &str {
        &self.inner.name
    }

    pub fn version(&self) -> &str {
        &self.inner.version
    }

    pub fn origin(&self) -> &Path {
        &self.inner.origin
    }

    /// Stable identity used in report output and plan digests.
    pub fn identity(&self) -> String {
        format!("{}@{}", self.inner.name, self.inner.version)
    }
}

impl std::fmt::Debug for Plugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Plugin")
            .field("name", &self.inner.name)
            .field("version", &self.inner.version)
            .field("origin", &self.inner.origin)
            .field("ops", &self.inner.ops.len())
            .finish()
    }
}

/// Reads the plugin's op table, rejecting a missing table or a missing slot.
///
/// Indices are kept in declaration order: dropping a null slot would silently
/// renumber the ops after it, so an op a recipe can name would change meaning.
fn parse_op_table(path: &Path, plugin: &RsPlugin) -> Result<Vec<&'static RsOpDesc>> {
    if plugin.n_ops == 0 {
        return Ok(Vec::new());
    }
    if plugin.ops.is_null() {
        return Err(AbiError::NullOpTable {
            path: path.to_path_buf(),
            count: plugin.n_ops,
        });
    }

    // SAFETY: `n_ops > 0` and the table is non-null; the plugin owns the array
    // for the process lifetime, and `library` is still mapped here.
    let table = unsafe { std::slice::from_raw_parts(plugin.ops, plugin.n_ops as usize) };
    let mut ops = Vec::with_capacity(table.len());
    for (index, &desc) in table.iter().enumerate() {
        // SAFETY: same lifetime argument as the table itself.
        let desc = unsafe { desc.as_ref() }.ok_or_else(|| AbiError::NullOp {
            path: path.to_path_buf(),
            index,
        })?;
        ops.push(desc);
    }
    Ok(ops)
}

/// Validates one descriptor against the ABI's structural rules.
fn validate_op(path: &Path, index: usize, desc: &RsOpDesc) -> Result<()> {
    // `struct_size` is read before anything past the header: a descriptor built
    // against an older header may not contain the fields we want to name it by.
    let min_size = std::mem::size_of::<RsOpDesc>() as u32;
    if desc.struct_size < min_size {
        return Err(AbiError::DescriptorTooSmall {
            path: path.to_path_buf(),
            found: desc.struct_size,
            expected: min_size,
        });
    }
    if desc.abi_version != ABI_VERSION {
        return Err(AbiError::OpVersionMismatch {
            path: path.to_path_buf(),
            op: unsafe { cstr(desc.id.name) }
                .unwrap_or("<unnamed>")
                .to_string(),
            variant: unsafe { cstr(desc.id.variant) }
                .unwrap_or("<unknown>")
                .to_string(),
            found: desc.abi_version,
            expected: ABI_VERSION,
        });
    }

    let op = unsafe { cstr(desc.id.name) }
        .ok_or_else(|| AbiError::OpWithoutName {
            path: path.to_path_buf(),
            index,
        })?
        .to_string();
    let variant = unsafe { cstr(desc.id.variant) }
        .ok_or_else(|| AbiError::OpWithoutVariant {
            path: path.to_path_buf(),
            index,
        })?
        .to_string();

    if desc.execute.is_none() {
        return Err(AbiError::OpWithoutExecute {
            path: path.to_path_buf(),
            op,
            variant,
        });
    }
    // A non-null expansion that cannot be walked is as good as absent, and
    // `RsExpansion::as_slice` would report zero nodes for it later.
    if let Some(expansion) = unsafe { desc.expansion.as_ref() }
        && (expansion.n_nodes == 0 || expansion.nodes.is_null())
    {
        return Err(AbiError::CompositeWithoutExpansion {
            path: path.to_path_buf(),
            op,
            variant,
        });
    }
    if desc.backward == RsBackwardKind::EXPLICIT && desc.backward_op.name.is_null() {
        return Err(AbiError::ExplicitBackwardWithoutOp {
            path: path.to_path_buf(),
            op,
            variant,
        });
    }
    Ok(())
}

/// A handle to one operator published by a plugin.
///
/// Holding one keeps the owning library mapped, so `desc` stays valid.
#[derive(Clone)]
pub struct LoadedOp {
    desc: &'static RsOpDesc,
    owner: Arc<PluginInner>,
}

impl LoadedOp {
    /// The raw descriptor. The borrow is tied to `self` on purpose: the
    /// descriptor lives in the plugin's library, which is unmapped as soon as
    /// the last handle to it is dropped.
    pub fn desc(&self) -> &RsOpDesc {
        self.desc
    }

    pub fn name(&self) -> &str {
        unsafe { cstr(self.desc.id.name) }.unwrap_or("<unnamed>")
    }

    pub fn variant(&self) -> &str {
        unsafe { cstr(self.desc.id.variant) }.unwrap_or("<unknown>")
    }

    pub fn doc(&self) -> &str {
        unsafe { cstr(self.desc.doc) }.unwrap_or("")
    }

    pub fn plugin_identity(&self) -> String {
        format!("{}@{}", self.owner.name, self.owner.version)
    }

    pub fn origin(&self) -> &Path {
        &self.owner.origin
    }

    /// `op@variant` — the name a recipe uses.
    pub fn spec_name(&self) -> String {
        format!("{}@{}", self.name(), self.variant())
    }

    pub fn requires(&self) -> Option<&RsRequires> {
        unsafe { self.desc.requires.as_ref() }
    }

    pub fn collectives(&self) -> &[RsCollective] {
        if self.desc.collectives.is_null() || self.desc.n_collectives == 0 {
            &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(self.desc.collectives, self.desc.n_collectives as usize)
            }
        }
    }

    pub fn expansion(&self) -> Option<&RsExpansion> {
        unsafe { self.desc.expansion.as_ref() }
    }

    /// Invokes the operator.
    ///
    /// # Safety
    /// `ctx` must be a valid context; every descriptor in `inputs` / `outputs`
    /// must describe memory the operator is allowed to read or write, with
    /// shapes and dtypes accepted by this implementation. Calling this from
    /// several threads at once is only sound when `ctx` and the buffers are
    /// disjoint and the plugin itself is thread-safe.
    pub unsafe fn execute(
        &self,
        ctx: *mut RsCtx,
        inputs: &[*const RsTensor],
        outputs: &[*mut RsTensor],
        attrs: *const RsAttrs,
    ) -> Result<()> {
        let Some(f) = self.desc.execute else {
            return Err(AbiError::OpWithoutExecute {
                path: self.owner.origin.clone(),
                op: self.name().to_string(),
                variant: self.variant().to_string(),
            });
        };
        // SAFETY: forwarded from the caller under the same contract.
        let status = unsafe {
            f(
                ctx,
                inputs.as_ptr(),
                inputs.len() as u32,
                outputs.as_ptr(),
                outputs.len() as u32,
                attrs,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(AbiError::ExecuteFailed {
                op: self.name().to_string(),
                variant: self.variant().to_string(),
                status,
                message: self.last_error(ctx),
            })
        }
    }

    fn last_error(&self, ctx: *mut RsCtx) -> String {
        let Some(f) = self.desc.last_error else {
            return String::new();
        };
        // SAFETY: the plugin owns this function pointer; `ctx` is the caller's.
        let p = unsafe { f(ctx) };
        unsafe { cstr(p) }.unwrap_or("").to_string()
    }
}

impl std::fmt::Debug for LoadedOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedOp")
            .field("op", &self.name())
            .field("variant", &self.variant())
            .field("plugin", &self.plugin_identity())
            .finish()
    }
}
