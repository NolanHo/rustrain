//! The `--device` argument shared by `run` and `ops check`.
//!
//! `cpu` is the default and keeps the legacy behaviour; `cuda` means device 0,
//! and `cuda:<index>` names a specific device. Anything else is an argument
//! error, never a guess.

use anyhow::{Result, bail};

use rustrain_abi::ffi::RsDeviceKind;

/// A parsed `--device` specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeviceSpec {
    Cpu,
    Cuda(usize),
}

impl DeviceSpec {
    pub(crate) fn parse(text: &str) -> Result<Self> {
        match text {
            "cpu" => Ok(DeviceSpec::Cpu),
            "cuda" => Ok(DeviceSpec::Cuda(0)),
            _ => {
                let Some(index) = text.strip_prefix("cuda:") else {
                    bail!("`--device {text}` is not one of `cpu`, `cuda`, or `cuda:<index>`");
                };
                let index: usize = index.parse().map_err(|_| {
                    anyhow::anyhow!(
                        "`--device {text}`: the device index after `cuda:` is not a non-negative \
                         integer"
                    )
                })?;
                Ok(DeviceSpec::Cuda(index))
            }
        }
    }

    pub(crate) const fn kind(self) -> RsDeviceKind {
        match self {
            DeviceSpec::Cpu => RsDeviceKind::CPU,
            DeviceSpec::Cuda(_) => RsDeviceKind::CUDA,
        }
    }

    pub(crate) const fn is_cuda(self) -> bool {
        matches!(self, DeviceSpec::Cuda(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_spec_parses_the_three_spellings() {
        assert_eq!(DeviceSpec::parse("cpu").unwrap(), DeviceSpec::Cpu);
        assert_eq!(DeviceSpec::parse("cuda").unwrap(), DeviceSpec::Cuda(0));
        assert_eq!(DeviceSpec::parse("cuda:3").unwrap(), DeviceSpec::Cuda(3));
        assert_eq!(DeviceSpec::parse("cuda:0").unwrap(), DeviceSpec::Cuda(0));
    }

    #[test]
    fn device_spec_rejects_everything_else() {
        for bad in ["", "gpu", "cuda:", "cuda:-1", "cuda:x", "CUDA"] {
            let error = DeviceSpec::parse(bad).unwrap_err();
            assert!(
                error.to_string().contains(bad.trim_end_matches(':')),
                "the error must name the offending spelling, got: {error}"
            );
        }
    }

    #[test]
    fn device_spec_reports_the_kind() {
        assert_eq!(DeviceSpec::Cpu.kind(), RsDeviceKind::CPU);
        assert_eq!(DeviceSpec::Cuda(2).kind(), RsDeviceKind::CUDA);
        assert!(!DeviceSpec::Cpu.is_cuda());
        assert!(DeviceSpec::Cuda(0).is_cuda());
    }
}
