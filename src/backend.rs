use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    NdArray,
}

pub trait Backend {
    fn kind(&self) -> BackendKind;
    fn supports_autograd(&self) -> bool;
    fn supports_cuda(&self) -> bool;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NdArrayBackend;

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
}
