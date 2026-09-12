//! Capability model: what a variant declares it needs versus what the target
//! environment actually provides.
//!
//! Every rejection in this module exists to be *printed*. Contract R-1 says a
//! resolution failure must name each candidate and the specific reason it was
//! rejected, so a [`RejectReason`] carries the numbers on both sides of the
//! comparison (`required` and `available`, not just "too low"): a user reading
//! the error has to be able to fix the recipe or the launch without reading
//! this source.
//!
//! Exactly one reason is reported per candidate. The checks run in declaration
//! order — dtype, SM, world size, group — so a candidate with two problems
//! reports the first one a user would fix, and the environment/device check
//! (which is a rustrain-ops convention, see [`declared_device`]) comes first
//! because it is structural: a CUDA variant on a CPU host is never going to
//! run regardless of anything else in the request.

use std::fmt;

use rustrain_abi::{RsBackwardKind, RsDeviceKind, RsDtype, RsGroupKind, RsRequires};
use serde::{Deserialize, Serialize};

use crate::registered::RegisteredOp;

/// The phase a variant is being selected for (spec §2.6 `PlanNode.phase`).
#[derive(
    Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    #[default]
    Forward,
    Backward,
    Update,
}

impl Phase {
    pub const ALL: [Phase; 3] = [Phase::Forward, Phase::Backward, Phase::Update];

    /// Stable lower-case name. Used in plan digests, so it must not change.
    pub const fn name(self) -> &'static str {
        match self {
            Phase::Forward => "forward",
            Phase::Backward => "backward",
            Phase::Update => "update",
        }
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Group kinds in their canonical order. Rejections are reported in this order
/// so that a message is reproducible across runs.
pub const GROUP_ORDER: [RsGroupKind; 4] = [
    RsGroupKind::TP,
    RsGroupKind::EP,
    RsGroupKind::CP,
    RsGroupKind::DP,
];

/// `RsGroupKind` has no `Display` in the ABI crate; report output needs one.
pub fn group_name(group: RsGroupKind) -> &'static str {
    match group.0 {
        1 => "tp",
        2 => "ep",
        4 => "cp",
        8 => "dp",
        _ => "unknown-group",
    }
}

/// `RsDeviceKind` has no `Display` in the ABI crate; report output needs one.
pub fn device_name(device: RsDeviceKind) -> &'static str {
    match device.0 {
        0 => "cpu",
        1 => "cuda",
        _ => "unknown-device",
    }
}

/// `RsBackwardKind` has no `Display` in the ABI crate; errors need one.
pub fn backward_name(kind: RsBackwardKind) -> &'static str {
    match kind.0 {
        0 => "AUTODIFF",
        1 => "EXPLICIT",
        2 => "NONDIFF",
        _ => "unknown",
    }
}

/// The environment a variant would run in.
///
/// `sm` is optional on purpose: "no GPU here" and "a GPU whose compute
/// capability we failed to query" are different situations, and only the second
/// one should ever be able to satisfy an `min_sm` requirement silently. A
/// variant that declares `min_sm` against an unknown device is rejected, not
/// assumed to fit.
///
/// Not `Serialize`: `RsGroupKind` / `RsDeviceKind` are foreign ABI newtypes
/// without serde impls, and this type is an input to resolution rather than
/// something that lands in a digest. Callers that need to record the
/// environment serialise the launch configuration they built it from.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TargetEnv {
    pub world_size: usize,
    /// Compute capability of the target device, e.g. `89` for sm_89.
    pub sm: Option<u32>,
    /// Group kinds this launch actually created.
    pub groups: Vec<RsGroupKind>,
    pub device: RsDeviceKind,
}

impl TargetEnv {
    /// Single-process CPU host — the environment `cargo test` runs in, and the
    /// honest default for a resolver that was handed no launch topology.
    pub fn cpu_local() -> Self {
        Self {
            world_size: 1,
            sm: None,
            groups: Vec::new(),
            device: RsDeviceKind::CPU,
        }
    }

    pub fn has_group(&self, group: RsGroupKind) -> bool {
        self.groups.contains(&group)
    }

    /// Group names, in [`GROUP_ORDER`]; used by report output.
    pub fn group_names(&self) -> Vec<&'static str> {
        GROUP_ORDER
            .iter()
            .filter(|g| self.has_group(**g))
            .map(|g| group_name(*g))
            .collect()
    }
}

impl Default for TargetEnv {
    fn default() -> Self {
        Self::cpu_local()
    }
}

impl fmt::Display for TargetEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "world_size={}, sm={}, device={}",
            self.world_size,
            match self.sm {
                Some(sm) => sm.to_string(),
                None => "unknown".to_string(),
            },
            device_name(self.device)
        )?;
        let groups = self.group_names();
        if groups.is_empty() {
            f.write_str(", groups=none")
        } else {
            write!(f, ", groups=[{}]", groups.join(", "))
        }
    }
}

/// Why a specific variant cannot serve a specific request.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RejectReason {
    /// The variant does not accept one of the requested dtypes.
    DtypeUnsupported {
        /// The dtype that failed. The first one in the request's order.
        requested: RsDtype,
        /// Everything the variant does accept, sorted by dtype id.
        accepted: Vec<RsDtype>,
    },
    /// The variant needs a newer compute capability than the target has.
    SmTooLow { required: u32, available: u32 },
    /// The variant needs a compute capability but the target reports none.
    SmUnknown { required: u32 },
    /// The variant needs more ranks than the launch has.
    WorldSizeTooSmall { required: usize, available: usize },
    /// The variant needs a process group the launch did not create.
    MissingGroup { group: RsGroupKind },
    /// The variant is device-specific and this is not that device.
    DeviceUnsupported {
        requested: RsDeviceKind,
        supported: RsDeviceKind,
    },
    /// The request asked for `Backward` but the variant does not declare an
    /// explicit backward implementation.
    PhaseUnsupported {
        phase: Phase,
        backward_kind: RsBackwardKind,
    },
    /// The request named a variant that no loaded plugin publishes.
    NotPublished { variant: String },
    /// The variant would run here, but the request never named it. Selecting it
    /// anyway is exactly the silent fallback contract R-1 forbids.
    NotListed,
}

impl fmt::Display for RejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RejectReason::DtypeUnsupported {
                requested,
                accepted,
            } => {
                if accepted.is_empty() {
                    write!(
                        f,
                        "dtype {requested} is not accepted: the variant declares no supported dtype \
                         (requires.dtype_mask is 0)"
                    )
                } else {
                    write!(
                        f,
                        "dtype {requested} is not accepted (variant accepts: {})",
                        join_dtypes(accepted)
                    )
                }
            }
            RejectReason::SmTooLow {
                required,
                available,
            } => write!(f, "sm {available} < required {required}"),
            RejectReason::SmUnknown { required } => write!(
                f,
                "requires sm >= {required} but the target reports no compute capability"
            ),
            RejectReason::WorldSizeTooSmall {
                required,
                available,
            } => write!(
                f,
                "world_size {available} < required {required} (this variant is only correct across \
                 {required} or more ranks)"
            ),
            RejectReason::MissingGroup { group } => write!(
                f,
                "requires the {} process group, which this launch did not create",
                group_name(*group)
            ),
            RejectReason::DeviceUnsupported {
                requested,
                supported,
            } => write!(
                f,
                "device {} requested but this variant is a {} implementation",
                device_name(*requested),
                device_name(*supported)
            ),
            RejectReason::PhaseUnsupported {
                phase,
                backward_kind,
            } => write!(
                f,
                "phase {phase} is not selectable: the variant declares backward = {}, and only \
                 EXPLICIT variants can be selected for the backward pass",
                backward_name(*backward_kind)
            ),
            RejectReason::NotPublished { variant } => write!(
                f,
                "no loaded plugin publishes variant `{variant}` (typo, or the plugin was not loaded)"
            ),
            RejectReason::NotListed => write!(
                f,
                "never named by this request (prefer/fallback/default); contract R-1 forbids \
                 selecting an implementation the recipe did not ask for"
            ),
        }
    }
}

impl std::error::Error for RejectReason {}

fn join_dtypes(dtypes: &[RsDtype]) -> String {
    dtypes
        .iter()
        .map(|d| d.name().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The device a variant declares through its namespace, or `None` when it is
/// device-agnostic.
///
/// The ABI's `rs_requires` has no device field (the header is frozen at v1, and
/// `RsDeviceKind` only appears in the service table), so the only in-band
/// declaration available is the variant's namespace: `cuda.fp8_block128` is a
/// CUDA implementation, `cpu.avx512` is a CPU one, and `reference`, `autodiff`
/// or `aten.linear` promise nothing and are therefore accepted everywhere the
/// resolver is asked about. Unknown namespaces intentionally impose no
/// constraint: a convention that silently rejected unfamiliar prefixes would be
/// worse than no convention.
///
/// Follow-up for ABI v2: carry a device mask in `rs_requires` and delete this
/// function.
pub fn declared_device(variant: &str) -> Option<RsDeviceKind> {
    // Split on `.` only, never on `_`: `cpu_offload.f32` is not a CPU-only
    // kernel, and treating it as one would reject a perfectly usable variant on
    // a CUDA host. Only an exact namespace word counts.
    let namespace = variant.split('.').next().unwrap_or("");
    match namespace {
        "cuda" | "gpu" => Some(RsDeviceKind::CUDA),
        "cpu" | "host" => Some(RsDeviceKind::CPU),
        _ => None,
    }
}

/// `RsRequires` with every field zero means "declares nothing" — the same
/// convention the header documents for `min_sm` and `min_world_size`, applied
/// consistently so that a zero-initialised C struct is not read as "accepts no
/// dtype at all".
pub(crate) fn declares_nothing(requires: &RsRequires) -> bool {
    // `min_world_size <= 0` counts as unset, matching the header's "0 = no
    // constraint": otherwise a nonsensical negative value would silently flip
    // the reading of `dtype_mask == 0` from "declares nothing" to "accepts no
    // dtype", and the user would be told to fix a field they never set.
    requires.dtype_mask == 0
        && requires.min_sm == 0
        && requires.min_world_size <= 0
        && requires.needs_groups == 0
}

/// Environment-only capability check (dtype set, SM, world size, groups,
/// device).
///
/// Returns the first reason the variant cannot serve `dtypes` on `env`, or
/// `None` when it can. This deliberately ignores the phase: use [`reject_for`]
/// for a full selection check.
pub fn reject_reason(
    op: &RegisteredOp,
    dtypes: &[RsDtype],
    env: &TargetEnv,
) -> Option<RejectReason> {
    if let Some(supported) = declared_device(op.variant()) {
        // No `let`-chain here: the workspace's declared MSRV is 1.85 and
        // let-chains only became available with edition 2024 / Rust 1.88.
        if supported != env.device {
            return Some(RejectReason::DeviceUnsupported {
                requested: env.device,
                supported,
            });
        }
    }

    // `requires == NULL` means the plugin declared no constraints at all.
    let requires = op.requires()?;
    if declares_nothing(requires) {
        return None;
    }

    for dtype in dtypes {
        if !requires.accepts(*dtype) {
            return Some(RejectReason::DtypeUnsupported {
                requested: *dtype,
                accepted: requires.dtypes(),
            });
        }
    }

    if requires.min_sm != 0 {
        match env.sm {
            None => {
                return Some(RejectReason::SmUnknown {
                    required: requires.min_sm,
                });
            }
            Some(available) if available < requires.min_sm => {
                return Some(RejectReason::SmTooLow {
                    required: requires.min_sm,
                    available,
                });
            }
            Some(_) => {}
        }
    }

    if requires.min_world_size > 0 && (env.world_size as i64) < requires.min_world_size {
        return Some(RejectReason::WorldSizeTooSmall {
            required: requires.min_world_size as usize,
            available: env.world_size,
        });
    }

    for group in GROUP_ORDER {
        if requires.needs_group(group) && !env.has_group(group) {
            return Some(RejectReason::MissingGroup { group });
        }
    }

    None
}

/// Phase-only capability check.
///
/// `Phase::Backward` asks "which variant supplies the backward pass of this
/// operator". Only a variant that declares `backward = EXPLICIT` can answer:
/// `AUTODIFF` means the gradient comes from composing the variant's own
/// expansion (so there is no separate backward kernel to select, and the plan
/// synthesises one — see `Recipe::backward_plan`), and `NONDIFF` means no
/// gradient exists at all.
///
/// The selected variant is the *forward* descriptor that declares the
/// backward; [`crate::Registry::backward_of`] follows its `backward_op` id to
/// the registration that actually executes.
pub fn phase_reject_reason(op: &RegisteredOp, phase: Phase) -> Option<RejectReason> {
    let backward_kind = op.backward_kind();
    match phase {
        Phase::Backward if backward_kind != RsBackwardKind::EXPLICIT => {
            Some(RejectReason::PhaseUnsupported {
                phase,
                backward_kind,
            })
        }
        _ => None,
    }
}

/// Full selection check: phase first (structural), then the environment.
pub fn reject_for(
    op: &RegisteredOp,
    phase: Phase,
    dtypes: &[RsDtype],
    env: &TargetEnv,
) -> Option<RejectReason> {
    phase_reject_reason(op, phase).or_else(|| reject_reason(op, dtypes, env))
}
