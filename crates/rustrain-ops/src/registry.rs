//! [`Registry`] — the single source of truth for "which operator
//! implementations exist" — and the resolution that picks one of them.
//!
//! Resolution is the place where contract R-1 stops being a slogan: a variant
//! named by `prefer` either runs or the whole resolution fails, `fallback`
//! entries are tried in the order the recipe wrote them, and every rejection is
//! kept so it can be printed (and digested) even when resolution succeeds.

use std::collections::BTreeSet;
use std::fmt;

use rustrain_abi::RsDtype;
use rustrain_abi::loader::Plugin;
use thiserror::Error;

use crate::capability::{Phase, RejectReason, TargetEnv, reject_for, reject_reason};
use crate::registered::{OpSummary, RegisteredOp};

/// The set of operator implementations the framework can see.
///
/// Owns cloneable [`RegisteredOp`] handles: a handle taken out of the registry
/// (or produced by [`Registry::resolve`]) keeps the plugin's `.so` mapped and
/// stays valid for as long as it is held, so the plan compiler can collect
/// handles and drop the registry.
///
/// Ordering is deterministic everywhere it is observable: candidates are sorted
/// by variant and `describe()` by `(op, variant)`. Nothing in this crate
/// iterates a hash map, so two runs with the same plugins produce the same
/// bytes.
#[derive(Clone, Debug, Default)]
pub struct Registry {
    ops: Vec<RegisteredOp>,
    /// Identity of every plugin that was loaded, including one that published
    /// no operator at all. Kept so that a failure can tell "the provider you
    /// named is not loaded" apart from "it is loaded but offers nothing for
    /// this operator".
    plugins: BTreeSet<String>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers every operator published by `plugin`.
    ///
    /// Returns how many operators were added. Registration is all-or-nothing:
    /// if any descriptor collides with one already registered (or with another
    /// descriptor in the same plugin), nothing is added and the error names
    /// both plugins. A half-registered plugin would make "which plugin won"
    /// depend on iteration order, which is exactly the kind of ambiguity this
    /// registry exists to remove.
    pub fn add_plugin(&mut self, plugin: Plugin) -> Result<usize, RegistryError> {
        let origin = plugin.origin().to_path_buf();
        let identity = plugin.identity();
        let incoming = plugin
            .ops()
            .into_iter()
            .map(|op| RegisteredOp::from_loaded(op, plugin.clone(), origin.clone()))
            .collect();
        let added = self.register(incoming)?;
        // Recorded separately from the operators: a plugin that publishes an
        // empty op table is still loaded, and telling a user "no plugin has
        // been loaded" would send them hunting for the wrong problem.
        self.plugins.insert(identity);
        Ok(added)
    }

    /// Registers descriptors that have no `.so` behind them.
    ///
    /// Tests only. Production code goes through [`Registry::add_plugin`], which
    /// is the only path that can produce a [`RegisteredOp`] carrying a loaded
    /// [`Plugin`].
    #[cfg(test)]
    pub(crate) fn add_detached(
        &mut self,
        plugin_name: &str,
        plugin_version: &str,
        origin: impl Into<std::path::PathBuf>,
        descs: &[&'static rustrain_abi::RsOpDesc],
    ) -> Result<usize, RegistryError> {
        use std::sync::Arc;

        use crate::registered::{PluginSlot, PluginStamp};

        let origin = origin.into();
        let stamp = Arc::new(PluginStamp {
            name: plugin_name.to_string(),
            version: plugin_version.to_string(),
        });
        let incoming = descs
            .iter()
            .map(|desc| {
                RegisteredOp::new(
                    desc,
                    PluginSlot::Detached(Arc::clone(&stamp)),
                    origin.clone(),
                )
            })
            .collect();
        let added = self.register(incoming)?;
        self.plugins
            .insert(format!("{plugin_name}@{plugin_version}"));
        Ok(added)
    }

    fn register(&mut self, incoming: Vec<RegisteredOp>) -> Result<usize, RegistryError> {
        let mut batch: BTreeSet<String> = BTreeSet::new();
        for op in &incoming {
            let spec_name = op.spec_name();
            if let Some(existing) = self.ops.iter().find(|e| e.spec_name() == spec_name) {
                return Err(RegistryError::DuplicateOp {
                    spec_name,
                    first_plugin: existing.plugin_identity(),
                    first_origin: existing.origin().display().to_string(),
                    second_plugin: op.plugin_identity(),
                    second_origin: op.origin().display().to_string(),
                });
            }
            if !batch.insert(spec_name.clone()) {
                // One plugin publishing the same op@variant twice. Reported
                // separately: "already published by X and now also by X" reads
                // like a two-plugin collision and sends the reader looking for
                // the wrong file.
                return Err(RegistryError::DuplicateWithinPlugin {
                    spec_name,
                    plugin: op.plugin_identity(),
                    origin: op.origin().display().to_string(),
                });
            }
        }
        let added = incoming.len();
        self.ops.extend(incoming);
        Ok(added)
    }

    /// Every variant of `name`, sorted by variant so that digests and error
    /// messages are reproducible.
    pub fn candidates(&self, name: &str) -> Vec<&RegisteredOp> {
        let mut found: Vec<&RegisteredOp> =
            self.ops.iter().filter(|op| op.name() == name).collect();
        found.sort_by(|a, b| a.variant().cmp(b.variant()));
        found
    }

    /// Distinct operator names, sorted.
    pub fn op_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.ops.iter().map(|op| op.name()).collect();
        names.sort_unstable();
        names.dedup();
        names
    }

    /// `plugin@version` of every loaded plugin, sorted — including a plugin
    /// that published no operator.
    pub fn plugin_names(&self) -> Vec<&str> {
        self.plugins.iter().map(String::as_str).collect()
    }

    /// Number of registered implementations (variants), not distinct operators.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Serialisable summary of the whole registry, sorted by `(op, variant)`.
    pub fn describe(&self) -> Vec<OpSummary> {
        let mut summaries: Vec<OpSummary> = self.ops.iter().map(OpSummary::of).collect();
        summaries.sort_by(|a, b| (&a.op, &a.variant).cmp(&(&b.op, &b.variant)));
        summaries
    }

    /// Picks the one implementation that will run for `req`, or explains why
    /// none can (contract R-1).
    ///
    /// `prefer` is absolute: a missing or capability-rejected preferred variant
    /// is an error and never falls through. `fallback` is tried in order, and
    /// the first entry that passes capability wins; entries that were skipped
    /// come back on [`ResolvedOp::rejected`] so the degradation can be logged
    /// and digested (contract R-2).
    ///
    /// With neither `prefer` nor `fallback` the request does not say who should
    /// win, so this returns [`ResolveError::Ambiguous`]. The recipe's
    /// `[kernel].default` provider — the last resort in the precedence chain of
    /// spec §2.7 — is applied by [`Registry::resolve_with_default`], which is
    /// what [`crate::Recipe::resolve`] calls.
    pub fn resolve(&self, req: &ResolveRequest) -> Result<ResolvedOp, ResolveError> {
        self.resolve_with_default(req, None)
    }

    /// [`Registry::resolve`] with the recipe's `[kernel].default` provider.
    ///
    /// The default provider decides only when neither `prefer` nor `fallback`
    /// named anything, and only when exactly one variant *that can run here*
    /// comes from it. Two runnable variants, or none, is
    /// [`ResolveError::Ambiguous`] — the recipe has to say which one it wants.
    pub fn resolve_with_default(
        &self,
        req: &ResolveRequest,
        default_provider: Option<&str>,
    ) -> Result<ResolvedOp, ResolveError> {
        let provider = default_provider.filter(|name| !name.trim().is_empty());
        let candidates = self.candidates(&req.name);
        if candidates.is_empty() {
            return Err(ResolveError::UnknownOp {
                name: req.name.clone(),
                known: self.op_names().into_iter().map(str::to_string).collect(),
                plugins: self
                    .plugin_names()
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            });
        }
        match req.prefer.as_deref() {
            Some(variant) => self.resolve_preferred(req, &candidates, variant, provider),
            None => self.resolve_chain(req, &candidates, provider),
        }
    }

    fn resolve_preferred(
        &self,
        req: &ResolveRequest,
        candidates: &[&RegisteredOp],
        variant: &str,
        provider: Option<&str>,
    ) -> Result<ResolvedOp, ResolveError> {
        let named = [variant.to_string()];
        let Some(op) = candidates.iter().find(|c| c.variant() == variant) else {
            return Err(ResolveError::PreferredNotPublished {
                op: req.name.clone(),
                variant: variant.to_string(),
                // R-1 wants the candidate table on *every* failure, and this is
                // the failure where it helps most: the user picked a name that
                // does not exist, and the table says which names do — and
                // whether any of them would have run here.
                failure: Box::new(self.failure(
                    req,
                    candidates,
                    format!("prefer = `{variant}`"),
                    &named,
                    provider,
                )),
            });
        };
        match reject_for(op, req.phase, &req.dtypes, &req.env) {
            None => Ok(ResolvedOp {
                op: RegisteredOp::clone(op),
                rejected: Vec::new(),
            }),
            Some(reason) => Err(ResolveError::PreferredRejected {
                op: req.name.clone(),
                variant: variant.to_string(),
                reason,
                failure: Box::new(self.failure(
                    req,
                    candidates,
                    format!("prefer = `{variant}` (no fallback is attempted for `prefer`)"),
                    &named,
                    provider,
                )),
            }),
        }
    }

    fn resolve_chain(
        &self,
        req: &ResolveRequest,
        candidates: &[&RegisteredOp],
        provider: Option<&str>,
    ) -> Result<ResolvedOp, ResolveError> {
        if req.fallback.is_empty() {
            return self.resolve_by_default_provider(req, candidates, provider);
        }

        let mut rejected: Vec<(String, RejectReason)> = Vec::new();
        for variant in &req.fallback {
            match candidates.iter().find(|c| c.variant() == variant) {
                None => rejected.push((
                    variant.clone(),
                    RejectReason::NotPublished {
                        variant: variant.clone(),
                    },
                )),
                Some(op) => match reject_for(op, req.phase, &req.dtypes, &req.env) {
                    None => {
                        let resolved = ResolvedOp {
                            op: RegisteredOp::clone(op),
                            rejected,
                        };
                        if resolved.degraded() {
                            // Contract R-2: degradation is recorded in the data
                            // (for the digest) *and* in the log (for the human).
                            tracing::warn!(
                                op = %req.name,
                                phase = %req.phase,
                                variant = %resolved.op.variant(),
                                skipped = %resolved.skipped_summary(),
                                "operator resolved through an explicit fallback"
                            );
                        }
                        return Ok(resolved);
                    }
                    Some(reason) => rejected.push((variant.clone(), reason)),
                },
            }
        }

        let strategy = format!(
            "fallback chain [{}]",
            req.fallback
                .iter()
                .map(|variant| format!("`{variant}`"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        Err(ResolveError::Unresolved(Box::new(self.failure(
            req,
            candidates,
            strategy,
            &req.fallback,
            provider,
        ))))
    }

    fn resolve_by_default_provider(
        &self,
        req: &ResolveRequest,
        candidates: &[&RegisteredOp],
        provider: Option<&str>,
    ) -> Result<ResolvedOp, ResolveError> {
        let strategy = match provider {
            Some(provider) => format!(
                "no `prefer`, no `fallback`: the default provider `{provider}` ([kernel].default) \
                 decides"
            ),
            None => {
                "no `prefer`, no `fallback`, and no default provider ([kernel].default is unset)"
                    .to_string()
            }
        };

        // The failure table already computes which variants of the default
        // provider are runnable; the decision reads it instead of recomputing.
        let failure = self.failure(req, candidates, strategy, &[], provider);
        if let [only] = failure.provider_eligible.as_slice() {
            let op = candidates
                .iter()
                .find(|candidate| candidate.variant() == only)
                .expect("provider_eligible names a published variant");
            return Ok(ResolvedOp {
                op: RegisteredOp::clone(op),
                rejected: Vec::new(),
            });
        }

        Err(ResolveError::Ambiguous(Box::new(failure)))
    }

    /// Turns the registration that `Phase::Backward` resolved to into the one
    /// that actually executes the backward pass.
    ///
    /// [`Registry::resolve`] with `Phase::Backward` returns the *forward*
    /// descriptor, because that is the object which declares how its gradient
    /// is produced (`backward = EXPLICIT` plus a `backward_op` id). That
    /// declaration points at another registered operator, and this is how a
    /// caller follows it: the declared variant must exist and must satisfy the
    /// environment, otherwise the error says exactly which part is missing.
    /// Without this step a backward node could execute the forward kernel, or
    /// an implementation that cannot run on this host — both of which contract
    /// R-1 exists to prevent.
    pub fn backward_of(
        &self,
        forward: &RegisteredOp,
        dtypes: &[RsDtype],
        env: &TargetEnv,
    ) -> Result<RegisteredOp, ResolveError> {
        let Some((name, variant)) = forward.backward_op_id() else {
            return Err(ResolveError::BackwardNotDeclared {
                forward: forward.spec_name(),
                backward_kind: forward.backward_kind(),
            });
        };
        let declared = format!("{name}@{variant}");
        let candidates = self.candidates(&name);
        let Some(op) = candidates.iter().find(|c| c.variant() == variant) else {
            return Err(ResolveError::BackwardOpMissing {
                forward: forward.spec_name(),
                declared,
                published: candidates.iter().map(|c| c.variant().to_string()).collect(),
                ops: self.op_names().into_iter().map(str::to_string).collect(),
            });
        };
        // Environment only: the backward kernel is not itself selected for a
        // phase, so its own `backward` declaration is irrelevant here.
        match reject_reason(op, dtypes, env) {
            None => Ok(RegisteredOp::clone(op)),
            Some(reason) => Err(ResolveError::BackwardOpRejected {
                forward: forward.spec_name(),
                declared,
                reason,
            }),
        }
    }

    /// Builds the candidate table contract R-1 requires in every failure
    /// message: every published variant, with the reason it was not selected.
    fn failure(
        &self,
        req: &ResolveRequest,
        candidates: &[&RegisteredOp],
        strategy: String,
        named: &[String],
        provider: Option<&str>,
    ) -> ResolutionFailure {
        let mut rows = Vec::with_capacity(candidates.len());
        let mut eligible = Vec::new();
        for op in candidates {
            let reason = match reject_for(op, req.phase, &req.dtypes, &req.env) {
                Some(reason) => reason,
                None => {
                    eligible.push(op.variant().to_string());
                    RejectReason::NotListed
                }
            };
            rows.push(CandidateRejection {
                variant: op.variant().to_string(),
                plugin: op.plugin_identity(),
                plugin_origin: op.origin().display().to_string(),
                reason,
            });
        }

        let published: BTreeSet<&str> = candidates.iter().map(|c| c.variant()).collect();
        let unpublished = named
            .iter()
            .filter(|variant| !published.contains(variant.as_str()))
            .cloned()
            .collect();

        // What the default provider offered, so an "ambiguous" message can say
        // whether the provider published nothing, published only unusable
        // variants, or published several runnable ones.
        let eligible_variants: BTreeSet<&str> = eligible.iter().map(String::as_str).collect();
        let provider_published: Vec<String> = candidates
            .iter()
            .filter(|op| Some(op.plugin_name()) == provider)
            .map(|op| op.variant().to_string())
            .collect();
        let provider_eligible: Vec<String> = candidates
            .iter()
            .filter(|op| Some(op.plugin_name()) == provider)
            .filter(|op| eligible_variants.contains(op.variant()))
            .map(|op| op.variant().to_string())
            .collect();

        ResolutionFailure {
            op: req.name.clone(),
            phase: req.phase,
            dtypes: req.dtypes.clone(),
            env: req.env.clone(),
            default_provider: provider.map(str::to_string),
            provider_loaded: provider
                .map(|provider| {
                    let prefix = format!("{provider}@");
                    self.plugins.iter().any(|p| p.starts_with(&prefix))
                })
                .unwrap_or(false),
            strategy,
            candidates: rows,
            unpublished,
            eligible,
            provider_published,
            provider_eligible,
        }
    }
}

/// One candidate the request could have used, with the reason it was not
/// selected.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CandidateRejection {
    pub variant: String,
    /// `plugin@version` that publishes the variant.
    pub plugin: String,
    /// Path of the `.so` the variant was loaded from. Two plugins can declare
    /// the same `name@version`, so this is the field that always tells them
    /// apart.
    pub plugin_origin: String,
    pub reason: RejectReason,
}

/// Everything a reader needs to fix a resolution failure: what was asked for,
/// which strategy was used, and the fate of every candidate.
#[derive(Clone, PartialEq, Debug)]
pub struct ResolutionFailure {
    pub op: String,
    pub phase: Phase,
    pub dtypes: Vec<RsDtype>,
    pub env: TargetEnv,
    /// The provider `[kernel].default` names, if any.
    pub default_provider: Option<String>,
    /// Whether that provider is loaded at all: distinguishes "you named a
    /// plugin that is not here" from "it is here and offers nothing usable".
    pub provider_loaded: bool,
    /// What the request named, in words.
    pub strategy: String,
    /// Every published variant, with the reason it was not selected.
    pub candidates: Vec<CandidateRejection>,
    /// Variants the request named that no plugin publishes (typos, or a plugin
    /// that was not loaded).
    pub unpublished: Vec<String>,
    /// Variants that *could* run here, sorted. They are listed so a user can
    /// see that the recipe, not the environment, is what failed.
    pub eligible: Vec<String>,
    /// Variants of this operator published by the default provider, sorted.
    pub provider_published: Vec<String>,
    /// Of [`ResolutionFailure::provider_published`], the ones that could run
    /// here. The "default provider decides" rule needs exactly one.
    pub provider_eligible: Vec<String>,
}

impl ResolutionFailure {
    fn dtypes_line(&self) -> String {
        if self.dtypes.is_empty() {
            "dtypes [unspecified: no dtype constraint was checked]".to_string()
        } else {
            format!(
                "dtypes [{}]",
                self.dtypes
                    .iter()
                    .map(|d| d.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

impl fmt::Display for ResolutionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "  request: phase {}; {}; {}\n  strategy: {}\n  candidates ({}):",
            self.phase,
            self.dtypes_line(),
            self.env,
            self.strategy,
            self.candidates.len()
        )?;
        for row in &self.candidates {
            write!(
                f,
                "\n    - {}@{} [{} from {}]: {}",
                self.op, row.variant, row.plugin, row.plugin_origin, row.reason
            )?;
        }
        if !self.unpublished.is_empty() {
            write!(
                f,
                "\n  named but not published: {}",
                self.unpublished
                    .iter()
                    .map(|v| format!("{}@{}", self.op, v))
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
        }
        if !self.eligible.is_empty() {
            write!(
                f,
                "\n  runnable here but never named by the request: {}",
                self.eligible.join(", ")
            )?;
        }
        f.write_str(
            "\n  contract R-1: nothing above may be selected implicitly — name a working \
             implementation in the recipe, or change the launch environment so that a named one \
             runs.",
        )
    }
}

/// Everything the resolver is allowed to know about a node.
///
/// Build it from a [`crate::Recipe`] with [`crate::Recipe::resolve_request`] so
/// that the phase variant and the fallback chain come from the same file as the
/// default provider passed to [`Registry::resolve_with_default`].
#[derive(Clone, PartialEq, Debug, Default)]
pub struct ResolveRequest {
    pub name: String,
    pub phase: Phase,
    /// Exact variant, e.g. `cuda.fp8_block128`. Absolute: never falls through.
    pub prefer: Option<String>,
    /// Exact variants, tried in order. The first that passes capability wins.
    pub fallback: Vec<String>,
    pub dtypes: Vec<RsDtype>,
    pub env: TargetEnv,
}

/// The implementation that won, plus the ones that were passed over.
#[derive(Clone, PartialEq, Debug)]
pub struct ResolvedOp {
    pub op: RegisteredOp,
    /// `(variant, reason)` for every candidate the request named that did not
    /// win, in the order they were tried. Empty for an undegraded resolution.
    pub rejected: Vec<(String, RejectReason)>,
}

impl ResolvedOp {
    /// True when resolution had to skip something (contract R-2: record it).
    pub fn degraded(&self) -> bool {
        !self.rejected.is_empty()
    }

    pub fn variant(&self) -> &str {
        self.op.variant()
    }

    pub fn spec_name(&self) -> String {
        self.op.spec_name()
    }

    /// One-line summary for logs and plan manifests.
    pub fn report(&self) -> String {
        if !self.degraded() {
            return format!("{} [{}]", self.op.spec_name(), self.op.plugin_identity());
        }
        format!(
            "{} [{}]; degraded, skipped: {}",
            self.op.spec_name(),
            self.op.plugin_identity(),
            self.skipped_summary()
        )
    }

    /// `variant (reason); variant (reason)` for every candidate that was passed
    /// over, in the order the request tried them.
    pub fn skipped_summary(&self) -> String {
        self.rejected
            .iter()
            .map(|(variant, reason)| format!("{variant} ({reason})"))
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// Registration failures.
#[derive(Clone, PartialEq, Eq, Debug, Error)]
pub enum RegistryError {
    /// Two plugins publish the same `op@variant`.
    #[error(
        "duplicate operator `{spec_name}`: already published by plugin `{first_plugin}` (loaded \
         from {first_origin}) and now also published by plugin `{second_plugin}` (loaded from \
         {second_origin}); op@variant identifies an implementation for the whole process, so \
         rename one of the variants"
    )]
    DuplicateOp {
        spec_name: String,
        first_plugin: String,
        first_origin: String,
        second_plugin: String,
        second_origin: String,
    },
    /// One plugin publishes the same `op@variant` twice.
    #[error(
        "plugin `{plugin}` ({origin}) publishes `{spec_name}` more than once; the name would \
         resolve to whichever descriptor happened to come first, so give those variants distinct \
         names"
    )]
    DuplicateWithinPlugin {
        spec_name: String,
        plugin: String,
        origin: String,
    },
}

/// Why no implementation could be selected.
#[derive(Clone, PartialEq, Debug)]
pub enum ResolveError {
    /// The operator itself is unknown: no loaded plugin publishes it.
    UnknownOp {
        name: String,
        known: Vec<String>,
        /// Every loaded plugin, sorted — including plugins that published no
        /// operator at all.
        plugins: Vec<String>,
    },
    /// `prefer` named a variant that is not published. No fallback is tried.
    PreferredNotPublished {
        op: String,
        variant: String,
        failure: Box<ResolutionFailure>,
    },
    /// `prefer` named a published variant that cannot run here. No fallback is
    /// tried.
    PreferredRejected {
        op: String,
        variant: String,
        reason: RejectReason,
        failure: Box<ResolutionFailure>,
    },
    /// The `fallback` chain was exhausted. Carries the full candidate table.
    Unresolved(Box<ResolutionFailure>),
    /// No `prefer`, no `fallback`, and the default-provider rule did not leave
    /// exactly one runnable implementation.
    Ambiguous(Box<ResolutionFailure>),
    /// A `strict = true` recipe refused a resolution that had to skip a
    /// candidate (see [`crate::Recipe::resolve`]).
    StrictDegraded {
        op: String,
        skipped: Vec<(String, RejectReason)>,
    },
    /// [`Registry::backward_of`] was asked for the backward pass of a variant
    /// that does not declare one.
    BackwardNotDeclared {
        forward: String,
        backward_kind: rustrain_abi::RsBackwardKind,
    },
    /// The backward operator a variant declares is not registered.
    BackwardOpMissing {
        forward: String,
        declared: String,
        /// Variants of that operator name that *are* published.
        published: Vec<String>,
        /// Operator names that are registered at all.
        ops: Vec<String>,
    },
    /// The declared backward operator exists but cannot run in this
    /// environment.
    BackwardOpRejected {
        forward: String,
        declared: String,
        reason: RejectReason,
    },
}

/// `""` for one, `"s"` otherwise. Error text is user-facing, so it should not
/// say "1 runnable variants".
fn plural(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

impl ResolveError {
    /// The candidate table, when the error has one.
    pub fn failure(&self) -> Option<&ResolutionFailure> {
        match self {
            ResolveError::PreferredNotPublished { failure, .. }
            | ResolveError::PreferredRejected { failure, .. }
            | ResolveError::Unresolved(failure)
            | ResolveError::Ambiguous(failure) => Some(failure),
            _ => None,
        }
    }
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResolveError::UnknownOp {
                name,
                known,
                plugins,
            } => {
                write!(f, "no operator `{name}` is registered")?;
                if known.is_empty() && plugins.is_empty() {
                    return f.write_str(
                        "\n  nothing is registered at all: no plugin has been loaded (check the \
                         plugin path and that every plugin reports ABI v1)",
                    );
                }
                if known.is_empty() {
                    write!(
                        f,
                        "\n  {} plugin{} loaded ({}) but none published an operator",
                        plugins.len(),
                        plural(plugins.len()),
                        plugins.join(", ")
                    )?;
                } else {
                    write!(f, "\n  registered operators: {}", known.join(", "))?;
                }
                Ok(())
            }
            ResolveError::PreferredNotPublished {
                op,
                variant,
                failure,
            } => {
                write!(
                    f,
                    "`prefer` names `{op}@{variant}`, which no loaded plugin publishes"
                )?;
                let published = failure
                    .candidates
                    .iter()
                    .map(|row| row.variant.as_str())
                    .collect::<Vec<_>>();
                if published.is_empty() {
                    write!(f, "\n  `{op}` publishes nothing at all")?;
                } else {
                    write!(f, "\n  `{op}` publishes: {}", published.join(", "))?;
                }
                write!(
                    f,
                    "\n  contract R-1: `prefer` never falls through — fix the variant name, or \
                     move it to `fallback` and prefer something else.\n{failure}"
                )
            }
            ResolveError::PreferredRejected {
                op,
                variant,
                reason,
                failure,
            } => write!(
                f,
                "`prefer` names `{op}@{variant}`, which is published but cannot run here: \
                 {reason}\n  contract R-1: no fallback was attempted for `prefer`.\n{failure}"
            ),
            ResolveError::Unresolved(failure) => {
                write!(
                    f,
                    "cannot resolve `{}` for phase {}: no implementation named by the request can \
                     run here\n{failure}",
                    failure.op, failure.phase
                )
            }
            ResolveError::Ambiguous(failure) => {
                write!(
                    f,
                    "cannot resolve `{}` for phase {}: the request does not pick exactly one \
                     implementation\n  ",
                    failure.op, failure.phase
                )?;
                let published = failure.provider_published.len();
                let runnable = failure.provider_eligible.len();
                match failure.default_provider.as_deref() {
                    None => write!(
                        f,
                        "no default provider is set, so nothing decides between the candidates: \
                         set `[kernel].default`, or name a variant in `[kernel.ops.{}]`",
                        failure.op
                    )?,
                    Some(provider) if !failure.provider_loaded => write!(
                        f,
                        "no loaded plugin is named `{provider}` (`[kernel].default`); loaded \
                         plugins: {}",
                        failure
                            .candidates
                            .iter()
                            .map(|row| row.plugin.as_str())
                            .collect::<BTreeSet<_>>()
                            .into_iter()
                            .collect::<Vec<_>>()
                            .join(", ")
                    )?,
                    Some(provider) if published == 0 => write!(
                        f,
                        "provider `{provider}` is loaded but publishes no `{}` at all",
                        failure.op
                    )?,
                    Some(provider) if runnable == 0 => write!(
                        f,
                        "provider `{provider}` publishes {published} variant{} of `{}`, none of \
                         which can run here",
                        plural(published),
                        failure.op
                    )?,
                    Some(provider) => write!(
                        f,
                        "provider `{provider}` publishes {runnable} runnable variant{} ({}): name \
                         one in `[kernel.ops.{}]`",
                        plural(runnable),
                        failure.provider_eligible.join(", "),
                        failure.op
                    )?,
                }
                write!(f, "\n{failure}")
            }
            ResolveError::StrictDegraded { op, skipped } => {
                write!(
                    f,
                    "strict recipe refused a degraded resolution of `{op}`: {} candidate(s) had to \
                     be skipped",
                    skipped.len()
                )?;
                for (variant, reason) in skipped {
                    write!(f, "\n    - {op}@{variant}: {reason}")?;
                }
                f.write_str(
                    "\n  `[kernel].strict = true` forbids running anything but the named variant: \
                     make the named variant runnable, or set strict = false.",
                )
            }
            ResolveError::BackwardNotDeclared {
                forward,
                backward_kind,
            } => write!(
                f,
                "`{forward}` does not declare a backward implementation (backward = {}): there is \
                 nothing to run for the backward pass. Select a variant whose backward is \
                 EXPLICIT, or differentiate its declared expansion instead.",
                crate::capability::backward_name(*backward_kind)
            ),
            ResolveError::BackwardOpMissing {
                forward,
                declared,
                published,
                ops,
            } => {
                write!(
                    f,
                    "`{forward}` declares its backward as `{declared}`, which is not registered"
                )?;
                let name = declared.split('@').next().unwrap_or(declared.as_str());
                if published.is_empty() {
                    write!(f, "\n  `{name}` publishes no variant at all")?;
                    if !ops.is_empty() {
                        write!(f, "; registered operators: {}", ops.join(", "))?;
                    }
                } else {
                    write!(f, "\n  `{name}` publishes: {}", published.join(", "))?;
                }
                f.write_str(
                    "\n  contract R-1: a backward node must not silently execute the forward \
                     kernel — load the plugin that registers it, or pick another variant.",
                )
            }
            ResolveError::BackwardOpRejected {
                forward,
                declared,
                reason,
            } => write!(
                f,
                "`{forward}` declares its backward as `{declared}`, which is registered but cannot \
                 run here: {reason}\n  the forward variant itself is runnable, so this would have \
                 failed at execution time instead of at plan time (contract PL-1)."
            ),
        }
    }
}

impl std::error::Error for ResolveError {}
