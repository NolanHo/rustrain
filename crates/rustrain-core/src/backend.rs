use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    NdArray,
    Tch,
}

pub trait Backend {
    fn kind(&self) -> BackendKind;
    fn supports_autograd(&self) -> bool;
    fn supports_cuda(&self) -> bool;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NdArrayBackend;

#[derive(Debug, Clone, Copy, Default)]
pub struct TchBackend;

impl Backend for NdArrayBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::NdArray
    }

    fn supports_autograd(&self) -> bool {
        false
    }

    fn supports_cuda(&self) -> bool {
        false
    }
}

#[cfg(feature = "tch")]
impl Backend for TchBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Tch
    }

    fn supports_autograd(&self) -> bool {
        true
    }

    fn supports_cuda(&self) -> bool {
        tch::Cuda::is_available()
    }
}

#[cfg(feature = "tch")]
pub fn tch_cpu_autograd_smoke() -> bool {
    let weight = tch::Tensor::from_slice(&[1.0_f32, -2.0, 0.5, 3.0])
        .reshape([2, 2])
        .set_requires_grad(true);
    let input = tch::Tensor::from_slice(&[2.0_f32, 1.0]).reshape([1, 2]);
    let output = input.matmul(&weight);
    let loss = output.square().mean(tch::Kind::Float);

    loss.backward();
    let grad = weight.grad();

    grad.defined() && grad.size() == vec![2, 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndarray_backend_is_cpu_forward_backend() {
        let backend = NdArrayBackend;

        assert_eq!(backend.kind(), BackendKind::NdArray);
        assert!(!backend.supports_autograd());
        assert!(!backend.supports_cuda());
    }

    #[cfg(feature = "tch")]
    #[test]
    fn tch_backend_reports_autograd_and_runs_cpu_backward() {
        let backend = TchBackend;

        assert_eq!(backend.kind(), BackendKind::Tch);
        assert!(backend.supports_autograd());
        assert!(tch_cpu_autograd_smoke());
    }
}
