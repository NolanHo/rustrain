pub mod command;
pub mod shm;

pub use command::{EpCommand, EpResult, TENSOR_SPAN_ALIGNMENT, TensorSlabRef, TensorSpan};
pub use shm::{DEFAULT_TENSOR_SLAB_BYTES, EpChannel, EpWorker};
