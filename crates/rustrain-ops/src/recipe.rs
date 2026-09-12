//! The recipe: the file that decides which implementation runs, at which
//! precision, on which topology (spec §2.7).
//!
//! Two properties matter more than the field list:
//!
//! * **Nothing is ignored.** Every struct is `#[serde(deny_unknown_fields)]`,
//!   so a typo'd key is a hard error instead of a setting that silently does
//!   nothing — the legacy behaviour this rewrite exists to remove.
//! * **Configuration is data.** Dtype, quantization scheme, block size, scale
//!   mode and amax history are carried into [`RsNumerics`]; no kernel may
//!   hardcode them and none may infer them from a tensor's shape (contract
//!   R-3).

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use rustrain_abi::{RsDtype, RsNumerics, RsQuantKind, RsScaleMode};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use thiserror::Error;

use crate::capability::{Phase, TargetEnv};
use crate::registry::{Registry, ResolveError, ResolveRequest, ResolvedOp};

macro_rules! name_newtype {
    ($(#[$meta:meta])* $name:ident, $inner:ty, $kind:literal,
     [$($variant:ident => $text:literal),+ $(,)?]
     $(, aliases: [$($alias:literal => $alias_variant:ident),* $(,)?])?
    ) => {
        $(#[$meta])*
        #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        pub struct $name($inner);

        impl $name {
            /// The canonical spellings, in declaration order. These are what a
            /// recipe *writes back* and what an error message prints, so the
            /// list is exactly the ABI vocabulary.
            pub const ACCEPTED: &'static [&'static str] = &[$($text),+];

            /// Spellings accepted on input that are not canonical and never
            /// appear in output. Kept deliberately short: every entry is a
            /// spelling that appears verbatim in this repository's own
            /// specification documents, so a recipe copied out of
            /// `docs/design/kernel-first/spec.md` parses.
            pub const ALIASES: &'static [&'static str] = &[$($($alias),*)?];

            /// Every value, in declaration order.
            pub const ALL: &'static [$name] = &[$($name(<$inner>::$variant)),+];

            pub const fn get(self) -> $inner {
                self.0
            }

            /// The canonical `snake_case` spelling used in recipe files.
            pub fn name(self) -> &'static str {
                $(if self.0 == <$inner>::$variant { return $text; })+
                "unknown"
            }

            pub fn parse(text: &str) -> Result<Self, RecipeError> {
                $(if text == $text { return Ok($name(<$inner>::$variant)); })+
                $( $(if text == $alias { return Ok($name(<$inner>::$alias_variant)); })* )?
                Err(RecipeError::UnknownName {
                    kind: $kind,
                    value: text.to_string(),
                    accepted: Self::ACCEPTED.join(", "),
                })
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self(<$inner>::default())
            }
        }

        impl From<$name> for $inner {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.name())
            }
        }

        impl std::str::FromStr for $name {
            type Err = RecipeError;

            fn from_str(text: &str) -> Result<Self, Self::Err> {
                Self::parse(text)
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(self.name())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let text = String::deserialize(deserializer)?;
                Self::parse(&text).map_err(de::Error::custom)
            }
        }
    };
}

name_newtype! {
    /// A dtype as spelled in a recipe file.
    ///
    /// Canonical spellings are exactly [`RsDtype::name`]'s; a unit test pins
    /// the two lists together so an ABI addition cannot drift. The aliases
    /// (`fp32`, `fp8e4m3`, …) exist because spec §2.7 and `AGENTS.md` spell
    /// these dtypes that way, and a recipe copied from the documentation has to
    /// keep working; output always uses the canonical ABI spelling, so an alias
    /// can never reach a plan digest.
    DtypeName, RsDtype, "dtype name",
    [
        F32 => "f32", F16 => "f16", BF16 => "bf16",
        F8E4M3 => "f8e4m3", F8E5M2 => "f8e5m2", FP4E2M1 => "fp4e2m1",
        I32 => "i32", I64 => "i64", U8 => "u8",
    ],
    aliases: [
        "fp32" => F32,
        "fp16" => F16,
        "fp8e4m3" => F8E4M3,
        "fp8_e4m3" => F8E4M3,
        "fp8e5m2" => F8E5M2,
        "fp8_e5m2" => F8E5M2,
    ]
}

name_newtype! {
    /// Quantization granularity (contract R-3: this is data, not a shape).
    QuantName, RsQuantKind, "quant scheme",
    [
        NONE => "none",
        PER_TENSOR => "per_tensor",
        PER_TOKEN => "per_token",
        PER_BLOCK => "per_block",
    ]
}

name_newtype! {
    /// How quantization scales are produced.
    ScaleName, RsScaleMode, "scale mode",
    [
        STATIC => "static",
        DYNAMIC_AMAX => "dynamic_amax",
        DELAYED => "delayed",
    ]
}

/// The one spelling of `[kernel.ops.<op>].backward` that is a *strategy* rather
/// than a variant name.
///
/// Spec §2.7 writes `backward = "autodiff"` for `mlp_swiglu`. No plugin
/// publishes a variant by that name (the ABI models it as
/// `rs_backward_kind.AUTODIFF`), so it must never be looked up as a variant —
/// see [`Recipe::backward_plan`].
pub const AUTODIFF: &str = "autodiff";

/// What `[kernel.ops.<op>].backward` asked for.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum BackwardPlan {
    /// A separate registered implementation runs the backward pass. The
    /// variant must declare `backward = EXPLICIT`, and
    /// [`crate::Registry::backward_of`] follows its `backward_op` id.
    Variant(String),
    /// No separate implementation: the framework differentiates the forward
    /// variant's declared expansion (contract R-4, `check_expansion`).
    Autodiff,
}

/// Per-operator recipe entry (`[kernel.ops.<op>]`).
#[derive(Clone, Default, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpRecipe {
    /// Exact variant for the forward pass, e.g. `cuda.fp8_block128`.
    ///
    /// The resolver treats this as `prefer`: if it is published but cannot run
    /// here, resolution fails rather than degrading (contract R-1).
    #[serde(default)]
    pub forward: Option<String>,
    /// Exact variant for the backward pass. Selectable only when it declares
    /// `backward = EXPLICIT`.
    #[serde(default)]
    pub backward: Option<String>,
    /// Variants that may be substituted, in order. A degraded resolution is
    /// recorded on the [`ResolvedOp`] (contract R-2).
    ///
    /// Consulted only for a phase that has no `forward`/`backward` variant:
    /// `prefer` never falls through, so a phase with a configured variant
    /// ignores this list.
    #[serde(default)]
    pub fallback: Vec<String>,
    /// Require this operator's fused variant to publish an expansion, and
    /// `rustrain ops check` to verify it against the primitives (contract R-4).
    #[serde(default)]
    pub check_expansion: bool,
}

/// `[kernel.precision]` — the numerics contract handed to every operator.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecisionRecipe {
    #[serde(default)]
    pub compute: DtypeName,
    #[serde(default)]
    pub accumulate: DtypeName,
    #[serde(default)]
    pub master_weights: DtypeName,
    #[serde(default)]
    pub grad: DtypeName,
    #[serde(default)]
    pub weights: DtypeName,
    #[serde(default)]
    pub quant_scheme: QuantName,
    /// Quantization block, as `[rows, cols]`. Data, never inferred (R-3).
    #[serde(default)]
    pub block: [u32; 2],
    #[serde(default)]
    pub scale_mode: ScaleName,
    #[serde(default)]
    pub amax_history: u32,
}

impl Default for PrecisionRecipe {
    /// The ABI's zero state: everything `f32`, no quantization, static scales.
    /// Deliberately equal to `RsNumerics::default()`, so "no `[kernel.precision]`
    /// section" means "no numerics override" rather than an accidental `bf16`.
    fn default() -> Self {
        Self {
            compute: DtypeName::default(),
            accumulate: DtypeName::default(),
            master_weights: DtypeName::default(),
            grad: DtypeName::default(),
            weights: DtypeName::default(),
            quant_scheme: QuantName::default(),
            block: [0, 0],
            scale_mode: ScaleName::default(),
            amax_history: 0,
        }
    }
}

impl PrecisionRecipe {
    /// Builds the ABI numerics contract for one phase.
    ///
    /// This mapping *is* the contract between the recipe file and a kernel, so
    /// it is spelled out here and nowhere else:
    ///
    /// | phase    | `in_dtype`       | `out_dtype`   | notes |
    /// |----------|------------------|---------------|-------|
    /// | forward  | `compute`        | `compute`     | activations in, activations out |
    /// | backward | `grad`           | `grad`        | output gradients in, input gradients out |
    /// | update   | `master_weights` | `weights`     | optimizer step: fp32 master in, parameter dtype out |
    ///
    /// `accum_dtype` is `accumulate` and `grad_dtype` is `grad` in every phase:
    /// accumulation precision and gradient precision are properties of the run,
    /// not of the direction. Quantization fields come from `quant_scheme`,
    /// `block`, `scale_mode` and `amax_history`; `scale_dtype` is the
    /// accumulation dtype, because scales are computed in it.
    ///
    /// `weights` is the one recipe field with no slot in [`RsNumerics`]: the
    /// dtype a *weight tensor* is stored in is a per-operator concern (a fused
    /// FP8 linear keeps an FP8 weight and an FP32 master copy), so the
    /// descriptor's own numerics — not this table — describe it. It is used
    /// here only as the output dtype of the update phase, which is literally
    /// the parameter write.
    pub fn numerics(&self, phase: Phase) -> RsNumerics {
        let (in_dtype, out_dtype) = match phase {
            Phase::Forward => (self.compute.get(), self.compute.get()),
            Phase::Backward => (self.grad.get(), self.grad.get()),
            Phase::Update => (self.master_weights.get(), self.weights.get()),
        };
        RsNumerics {
            in_dtype,
            out_dtype,
            accum_dtype: self.accumulate.get(),
            grad_dtype: self.grad.get(),
            quant: self.quant_scheme.get(),
            block_m: self.block[0],
            block_n: self.block[1],
            scale_dtype: self.accumulate.get(),
            scale_mode: self.scale_mode.get(),
            amax_history: self.amax_history,
            _pad: 0,
        }
    }
}

fn one() -> usize {
    1
}

/// `[kernel.parallel]` — the topology the run is launched with.
///
/// Degrees default to `1` (the identity) rather than `0`: an omitted degree
/// means "not parallel along this axis", and `0` would make the world size a
/// meaningless product. `world_size()` is therefore always well defined.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelRecipe {
    #[serde(default = "one")]
    pub tensor: usize,
    #[serde(default = "one")]
    pub expert: usize,
    #[serde(default = "one")]
    pub context: usize,
    #[serde(default = "one")]
    pub data: usize,
    #[serde(default = "one")]
    pub pipeline: usize,
    #[serde(default)]
    pub overlap_collectives: bool,
}

impl Default for ParallelRecipe {
    fn default() -> Self {
        Self {
            tensor: 1,
            expert: 1,
            context: 1,
            data: 1,
            pipeline: 1,
            overlap_collectives: false,
        }
    }
}

impl ParallelRecipe {
    /// Ranks implied by the topology: `tensor × expert × context × data ×
    /// pipeline`.
    ///
    /// Saturating rather than wrapping: a recipe is user input, and a product
    /// that overflows `usize` must not panic a run (or silently wrap into a
    /// plausible-looking topology).
    pub fn world_size(&self) -> usize {
        self.tensor
            .saturating_mul(self.expert)
            .saturating_mul(self.context)
            .saturating_mul(self.data)
            .saturating_mul(self.pipeline)
    }
}

/// The `[kernel]` table of a recipe file.
#[derive(Clone, Default, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    /// Plugin whose implementation wins when nothing else names one. Empty
    /// means "no default provider", which makes an unqualified resolution
    /// ambiguous rather than arbitrary.
    #[serde(default)]
    pub default: String,
    /// Refuse a resolution that had to skip a candidate (see
    /// [`Recipe::resolve`]).
    #[serde(default)]
    pub strict: bool,
    #[serde(default)]
    pub ops: BTreeMap<String, OpRecipe>,
    #[serde(default)]
    pub precision: PrecisionRecipe,
    #[serde(default)]
    pub parallel: ParallelRecipe,
}

/// The document shape of a recipe file: a single `[kernel]` table.
///
/// `deny_unknown_fields` applies here too, so a misspelled section
/// (`[kernels]`) fails instead of being ignored.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecipeFile {
    #[serde(default)]
    kernel: Recipe,
}

#[derive(Serialize)]
struct RecipeFileRef<'a> {
    kernel: &'a Recipe,
}

impl Recipe {
    /// Parses the `[kernel]` table of a recipe file in TOML form.
    pub fn from_toml(text: &str) -> Result<Self, RecipeError> {
        let file: RecipeFile = toml::from_str(text)?;
        file.kernel.validate()?;
        Ok(file.kernel)
    }

    /// Reads and parses a recipe file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RecipeError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| RecipeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_toml(&text)
    }

    /// Renders the recipe back to TOML (used for manifests and round-trip
    /// tests; the run manifest stores the recipe verbatim, per spec §2.9).
    pub fn to_toml(&self) -> Result<String, RecipeError> {
        Ok(toml::to_string_pretty(&RecipeFileRef { kernel: self })?)
    }

    /// Rejects combinations that parse but cannot mean anything.
    ///
    /// Only contradictions that would otherwise be *silently* ignored are
    /// checked here: a per-block scheme without a block size, and delayed
    /// scaling with no amax history. Both would otherwise reach a kernel as
    /// zeros and be resolved by guesswork — the exact failure mode contract R-3
    /// forbids.
    pub fn validate(&self) -> Result<(), RecipeError> {
        let precision = &self.precision;
        if precision.quant_scheme.get() == RsQuantKind::PER_BLOCK
            && (precision.block[0] == 0 || precision.block[1] == 0)
        {
            return Err(RecipeError::Invalid(format!(
                "quant_scheme = \"per_block\" needs an explicit block size, but block = [{}, {}]; \
                 set e.g. block = [128, 128] (contract R-3: the block size is data, never inferred \
                 from tensor shapes)",
                precision.block[0], precision.block[1]
            )));
        }
        if precision.scale_mode.get() == RsScaleMode::DELAYED && precision.amax_history == 0 {
            return Err(RecipeError::Invalid(
                "scale_mode = \"delayed\" needs amax_history > 0; a delayed scaling scheme with no \
                 history has nothing to delay by"
                    .to_string(),
            ));
        }

        // A variant is named exactly. `forward = ""` would otherwise select a
        // plugin's empty-name variant, which is the "unset-looking config
        // silently chooses an implementation" failure this rewrite exists to
        // remove; padding is the same typo with invisible characters.
        for (op, entry) in &self.ops {
            check_variant_spelling(
                &format!("kernel.ops.{op}.forward"),
                entry.forward.as_deref(),
            )?;
            if entry.backward.as_deref() != Some(AUTODIFF) {
                check_variant_spelling(
                    &format!("kernel.ops.{op}.backward"),
                    entry.backward.as_deref(),
                )?;
            }
            for (index, variant) in entry.fallback.iter().enumerate() {
                check_variant_spelling(
                    &format!("kernel.ops.{op}.fallback[{index}]"),
                    Some(variant.as_str()),
                )?;
            }
        }

        let parallel = &self.parallel;
        for (axis, degree) in [
            ("tensor", parallel.tensor),
            ("expert", parallel.expert),
            ("context", parallel.context),
            ("data", parallel.data),
            ("pipeline", parallel.pipeline),
        ] {
            if degree == 0 {
                return Err(RecipeError::Invalid(format!(
                    "kernel.parallel.{axis} = 0 is not a parallelism degree; omit the key to get \
                     the default 1, or set the number of ranks along that axis"
                )));
            }
        }
        Ok(())
    }

    /// What `[kernel.ops.<op>].backward` asked for, if it asked for anything.
    pub fn backward_plan(&self, op: &str) -> Option<BackwardPlan> {
        let raw = self.ops.get(op)?.backward.as_deref()?;
        if raw == AUTODIFF {
            Some(BackwardPlan::Autodiff)
        } else {
            Some(BackwardPlan::Variant(raw.to_string()))
        }
    }

    /// The plugin named by `[kernel].default`, if it is set.
    pub fn default_provider(&self) -> Option<&str> {
        let trimmed = self.default.trim();
        (!trimmed.is_empty()).then_some(trimmed)
    }

    pub fn op_recipe(&self, op: &str) -> Option<&OpRecipe> {
        self.ops.get(op)
    }

    /// Exact variant configured for `op` in `phase`, if any.
    ///
    /// Precedence, per spec §2.7: the op-level `forward`/`backward` wins; when
    /// neither is set for the phase, the op-level `fallback` list is the
    /// resolver's business (see [`Recipe::resolve_request`]), and this returns
    /// `None`. `Phase::Update` has no op-level key by design: an optimizer step
    /// falls through to the default provider or the fallback chain.
    pub fn variant_for(&self, op: &str, phase: Phase) -> Option<&str> {
        let entry = self.ops.get(op)?;
        match phase {
            Phase::Forward => entry.forward.as_deref(),
            // `autodiff` is a strategy, not a variant name: returning it here
            // would send the resolver looking for a variant no plugin can
            // publish. `backward_plan` is where that spelling is answered.
            Phase::Backward => entry.backward.as_deref().filter(|raw| *raw != AUTODIFF),
            Phase::Update => None,
        }
    }

    /// True when `op` asked for expansion equivalence checking (contract R-4).
    pub fn check_expansion_for(&self, op: &str) -> bool {
        self.ops.get(op).is_some_and(|entry| entry.check_expansion)
    }

    /// Translates the recipe into the resolver's input.
    ///
    /// This is the only place where recipe syntax meets resolution policy:
    /// `forward`/`backward` become `prefer` (absolute), `fallback` stays a
    /// chain, and `[kernel].default` becomes the provider consulted when
    /// neither is set.
    pub fn resolve_request(
        &self,
        name: &str,
        phase: Phase,
        dtypes: &[RsDtype],
        env: &TargetEnv,
    ) -> ResolveRequest {
        ResolveRequest {
            name: name.to_string(),
            phase,
            prefer: self.variant_for(name, phase).map(str::to_string),
            fallback: self
                .op_recipe(name)
                .map(|entry| entry.fallback.clone())
                .unwrap_or_default(),
            dtypes: dtypes.to_vec(),
            env: env.clone(),
        }
    }

    /// Resolves `name` for `phase` through this recipe, enforcing `strict`.
    ///
    /// `strict = true` means "no degradation at all": a resolution that had to
    /// skip a candidate is refused, with the skipped candidates and their
    /// reasons in the error. R-2 already requires degradation to be recorded;
    /// this is the recipe's way of saying it must not happen.
    pub fn resolve(
        &self,
        registry: &Registry,
        name: &str,
        phase: Phase,
        dtypes: &[RsDtype],
        env: &TargetEnv,
    ) -> Result<ResolvedOp, ResolveError> {
        let request = self.resolve_request(name, phase, dtypes, env);
        let resolved = registry.resolve_with_default(&request, self.default_provider())?;
        if self.strict && resolved.degraded() {
            return Err(ResolveError::StrictDegraded {
                op: name.to_string(),
                skipped: resolved.rejected.clone(),
            });
        }
        Ok(resolved)
    }
}

/// Rejects a variant spelling that cannot be a variant name: empty, or padded
/// with whitespace that the author cannot see.
fn check_variant_spelling(field: &str, value: Option<&str>) -> Result<(), RecipeError> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_empty() || value.trim() != value {
        return Err(RecipeError::Invalid(format!(
            "{field} = {value:?} is not a usable variant name; a variant is an exact spelling \
             with no surrounding whitespace (omit the key to let the default provider or the \
             fallback chain decide)"
        )));
    }
    Ok(())
}

/// Recipe parsing and validation failures.
#[derive(Debug, Error)]
pub enum RecipeError {
    #[error("cannot read recipe {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid recipe: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("cannot encode recipe: {0}")]
    Encode(#[from] toml::ser::Error),
    /// The document parses but contradicts itself.
    #[error("invalid recipe: {0}")]
    Invalid(String),
    #[error("unknown {kind} `{value}`: expected one of {accepted}")]
    UnknownName {
        kind: &'static str,
        value: String,
        accepted: String,
    },
}
