//! Executing a compiled plan.
//!
//! The executor is deliberately small. Compilation already decided *what* runs
//! (operator, variant, numerics) and *where communication goes* (the spliced
//! collectives). What is left is mechanical: give every slot a buffer, walk the
//! steps in order, hand each implementation the descriptors it asked for.
//!
//! Two things are pluggable, and both exist so the core can be exercised without
//! a GPU:
//!
//! * [`Allocator`] — where slot buffers come from. [`HostAllocator`] is the
//!   in-process one; a CUDA allocator implements the same trait later.
//! * [`CollectiveBackend`] — what a spliced collective actually does.
//!   [`SingleRank`] is the identity, which is exactly right when the parallel
//!   configuration is 1×1×1×1×1 and is what makes a TP plan testable on a laptop.

// Same reasoning as `rustrain-plan`: the error carries structured diagnostics.
#![allow(clippy::result_large_err)]

pub mod conformance;

use std::ffi::c_void;

use rustrain_abi::ffi::{RsCollectiveKind, RsCtx, RsDeviceKind, RsDtype, RsServices, RsTensor};
use rustrain_parallel::{GroupKind, ReduceOp};
use rustrain_plan::{CompiledPlan, CompiledStep, Slot, SlotId, SlotKind};

/// Anything that can go wrong while running a plan.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(
        "slot {slot:?} ({name}) has no buffer; it is produced by no node and was never written"
    )]
    SlotUnwritten { slot: SlotId, name: String },

    #[error("slot {slot:?} ({name}) holds {expected} elements but {actual} were supplied")]
    LengthMismatch {
        slot: SlotId,
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("slot {slot:?} ({name}) is {dtype}, not f32")]
    NotF32 {
        slot: SlotId,
        name: String,
        dtype: String,
    },

    #[error("slot {slot:?} ({name}) holds {expected} bytes but {actual} were supplied")]
    ByteLengthMismatch {
        slot: SlotId,
        name: String,
        expected: u64,
        actual: u64,
    },

    #[error("unknown dtype `{dtype}`")]
    UnknownDtype { dtype: String },

    #[error("cannot allocate {bytes} bytes for slot {slot:?}: {reason}")]
    Alloc {
        slot: SlotId,
        bytes: u64,
        reason: String,
    },

    #[error("step {index} ({op}) failed: {message}")]
    Op {
        index: usize,
        op: String,
        message: String,
    },

    #[error("step {index} {op} is a collective but no backend can execute it: {reason}")]
    Collective {
        index: usize,
        op: String,
        reason: String,
    },

    #[error("slot {slot:?} has no data pointer")]
    NullData { slot: SlotId },
}

/// Where slot buffers come from.
///
/// # Safety
/// `alloc` must return a pointer valid for `bytes` of read and write until
/// `dealloc` is called with the same `(ptr, bytes)` pair.
pub unsafe trait Allocator {
    fn alloc(&mut self, bytes: u64, device: RsDeviceKind) -> Result<*mut c_void, String>;
    fn dealloc(&mut self, ptr: *mut c_void, bytes: u64);
    fn device(&self) -> RsDeviceKind;
}

/// Host memory. Sufficient for the reference provider and for every test that
/// does not need a device.
#[derive(Debug, Default)]
pub struct HostAllocator {
    live: u64,
    peak: u64,
}

impl HostAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes currently allocated.
    pub fn live_bytes(&self) -> u64 {
        self.live
    }

    /// High-water mark since construction.
    pub fn peak_bytes(&self) -> u64 {
        self.peak
    }
}

// SAFETY: `alloc` returns 64-byte-aligned memory from a `Layout` that `dealloc`
// reconstructs exactly; the pointer is valid for `bytes` until then.
unsafe impl Allocator for HostAllocator {
    fn alloc(&mut self, bytes: u64, _device: RsDeviceKind) -> Result<*mut c_void, String> {
        let bytes = bytes.max(1);
        let layout = std::alloc::Layout::from_size_align(bytes as usize, 64)
            .map_err(|e| format!("invalid layout for {bytes} bytes: {e}"))?;
        // SAFETY: the layout is non-zero-sized thanks to `max(1)`.
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(format!("host allocation of {bytes} bytes failed"));
        }
        self.live += bytes;
        self.peak = self.peak.max(self.live);
        Ok(ptr as *mut c_void)
    }

    fn dealloc(&mut self, ptr: *mut c_void, bytes: u64) {
        if ptr.is_null() {
            return;
        }
        let bytes = bytes.max(1);
        if let Ok(layout) = std::alloc::Layout::from_size_align(bytes as usize, 64) {
            // SAFETY: the pointer came from `alloc` with this exact layout.
            unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
        }
        self.live = self.live.saturating_sub(bytes);
    }

    fn device(&self) -> RsDeviceKind {
        RsDeviceKind::CPU
    }
}

/// Performs the collective a spliced node represents.
pub trait CollectiveBackend {
    fn execute(
        &mut self,
        kind: RsCollectiveKind,
        group: GroupKind,
        reduce: Option<ReduceOp>,
        dim: Option<i64>,
        tensor: &mut RsTensor,
    ) -> Result<(), String>;
}

/// The identity backend.
///
/// Correct precisely when nothing is actually distributed, and the only thing
/// that can back a TP plan inside one process. It refuses when the world size is
/// larger than one rather than silently pretending: a plan whose sharding
/// requires a real all-reduce cannot be executed correctly by doing nothing.
#[derive(Debug, Default)]
pub struct SingleRank {
    world_size: usize,
}

impl SingleRank {
    pub fn new(world_size: usize) -> Self {
        Self { world_size }
    }
}

impl CollectiveBackend for SingleRank {
    fn execute(
        &mut self,
        kind: RsCollectiveKind,
        group: GroupKind,
        reduce: Option<ReduceOp>,
        dim: Option<i64>,
        _tensor: &mut RsTensor,
    ) -> Result<(), String> {
        if self.world_size > 1 {
            return Err(format!(
                "collective {kind:?} on group {group:?} was requested with world_size={}, but no \
                 distributed backend is installed",
                self.world_size
            ));
        }
        let _ = (reduce, dim);
        Ok(())
    }
}

/// One slot's backing storage.
struct SlotBuffer {
    /// Where the slot's data currently lives.
    ///
    /// Not always the allocation we made: a view operator (`transpose`,
    /// `narrow`, `reshape`, `view`, `broadcast`) returns a descriptor whose
    /// `data` points into its input, and the runtime has to adopt it. Ignoring
    /// the returned pointer reads uninitialised memory belonging to the
    /// executor's own buffer — which is what the conformance gate caught the
    /// first time it ran.
    ptr: *mut c_void,
    /// The allocation this executor owns and must free, if any. A view slot
    /// adopts a pointer but still owns nothing.
    owned: Option<(*mut c_void, u64)>,
    /// The slot's logical shape and strides, adopted from the descriptor the
    /// operator returned. A view can change both: `broadcast` gives size-1 dims
    /// stride 0, so the elements are *not* laid out contiguously and reading
    /// `numel * width` bytes from `ptr` walks off the end of the buffer. The
    /// conformance gate caught exactly that, reporting two runs that produced
    /// "different" bytes whose first few values matched.
    shape: [i64; rustrain_abi::ffi::MAX_RANK],
    strides: [i64; rustrain_abi::ffi::MAX_RANK],
    rank: u32,
    elem_width: u32,
    bytes: u64,
    /// Set when this slot shares another slot's storage — true for the output of
    /// a spliced collective, which reduces in place.
    alias_of: Option<SlotId>,
}

/// What a run did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunStats {
    pub steps: usize,
    pub ops: usize,
    pub collectives: usize,
    pub resident_bytes: u64,
}

/// Walks a [`CompiledPlan`].
pub struct Executor {
    plan: CompiledPlan,
    allocator: Box<dyn Allocator + Send>,
    collectives: Box<dyn CollectiveBackend + Send>,
    buffers: Vec<Option<SlotBuffer>>,
    aliases: Vec<Option<SlotId>>,
    services: Box<RsServices>,
    stats: RunStats,
}

impl Executor {
    /// Prepares storage for every slot the plan declares.
    pub fn new(
        plan: CompiledPlan,
        mut allocator: Box<dyn Allocator + Send>,
        collectives: Box<dyn CollectiveBackend + Send>,
    ) -> Result<Self, RuntimeError> {
        let n = plan.plan.slots.len();

        // A spliced collective reduces a tensor in place, so its output slot is
        // the input slot's storage under another name. Resolving the chain up
        // front keeps the walk below free of special cases.
        let mut aliases: Vec<Option<SlotId>> = vec![None; n];
        for step in &plan.steps {
            if let CompiledStep::Intrinsic { input, output, .. } = step {
                let root = aliases[input.0].unwrap_or(*input);
                aliases[output.0] = Some(root);
            }
        }

        let device = allocator.device();
        let mut buffers: Vec<Option<SlotBuffer>> = Vec::with_capacity(n);
        for (i, alias) in aliases.iter().enumerate().take(n) {
            if let Some(root) = *alias {
                let root_buf = buffers[root.0].as_ref().map(|b| (b.ptr, b.bytes));
                let (ptr, bytes) = root_buf.ok_or(RuntimeError::NullData { slot: root })?;
                let src = buffers[root.0].as_ref().expect("checked above");
                buffers.push(Some(SlotBuffer {
                    ptr,
                    owned: None,
                    shape: src.shape,
                    strides: src.strides,
                    rank: src.rank,
                    elem_width: src.elem_width,
                    bytes,
                    alias_of: Some(root),
                }));
                continue;
            }

            let bytes = slot_element_bytes(&plan.plan.slots[i]).map_err(|reason| {
                RuntimeError::Alloc {
                    slot: SlotId(i),
                    bytes: 0,
                    reason,
                }
            })?;
            let ptr = allocator
                .alloc(bytes, device)
                .map_err(|reason| RuntimeError::Alloc {
                    slot: SlotId(i),
                    bytes,
                    reason,
                })?;
            let slot = &plan.plan.slots[i];
            let mut shape = [0i64; rustrain_abi::ffi::MAX_RANK];
            let mut strides = [0i64; rustrain_abi::ffi::MAX_RANK];
            let rank = (slot.shape.len() as u32).min(rustrain_abi::ffi::MAX_RANK as u32);
            let mut acc = 1i64;
            for d in (0..rank as usize).rev() {
                shape[d] = slot.shape[d];
                strides[d] = acc;
                acc *= slot.shape[d].max(1);
            }
            buffers.push(Some(SlotBuffer {
                ptr,
                owned: Some((ptr, bytes)),
                shape,
                strides,
                rank,
                elem_width: slot.dtype.byte_width().unwrap_or(4),
                bytes,
                alias_of: None,
            }));
        }

        let resident_bytes = buffers
            .iter()
            .filter_map(Option::as_ref)
            .filter(|b| b.alias_of.is_none())
            .map(|b| b.bytes)
            .sum();

        Ok(Self {
            plan,
            allocator,
            collectives,
            buffers,
            aliases,
            services: Box::new(no_services()),
            stats: RunStats {
                resident_bytes,
                ..Default::default()
            },
        })
    }

    pub fn plan(&self) -> &CompiledPlan {
        &self.plan
    }

    pub fn stats(&self) -> &RunStats {
        &self.stats
    }

    /// A descriptor for one slot, pointing at its storage.
    pub fn descriptor(&self, id: SlotId) -> Result<RsTensor, RuntimeError> {
        let slot = self.plan.plan.slot(id);
        let root = self.aliases[id.0].unwrap_or(id);
        let buf = self
            .buffers
            .get(root.0)
            .and_then(Option::as_ref)
            .ok_or_else(|| RuntimeError::SlotUnwritten {
                slot: id,
                name: slot.name.clone(),
            })?;
        if buf.ptr.is_null() {
            return Err(RuntimeError::NullData { slot: id });
        }
        // Built field by field rather than via `RsTensor::new`, because the slot
        // may hold a strided view whose shape is not the plan's and whose
        // strides are not contiguous.
        Ok(RsTensor {
            dtype: slot.dtype,
            rank: buf.rank,
            shape: buf.shape,
            stride: buf.strides,
            data: buf.ptr,
            ..RsTensor::default()
        })
    }

    /// Element count a slot expects.
    pub fn slot_len(&self, id: SlotId) -> usize {
        self.plan.plan.slot(id).shape.iter().product::<i64>().max(0) as usize
    }

    /// The slots a caller must fill before running: those no node produces.
    pub fn input_slots(&self) -> Vec<SlotId> {
        self.plan.plan.input_slots()
    }

    /// Writes host f32 data into a slot.
    pub fn write_f32(&mut self, id: SlotId, data: &[f32]) -> Result<(), RuntimeError> {
        self.check_f32(id)?;
        let expected = self.slot_len(id);
        if expected != data.len() {
            return Err(RuntimeError::LengthMismatch {
                slot: id,
                name: self.plan.plan.slot(id).name.clone(),
                expected,
                actual: data.len(),
            });
        }
        let ptr = self.data_ptr(id)?;
        // SAFETY: the buffer holds at least `expected` f32 (sized from the same
        // shape table) and `data` has exactly that many elements.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut f32, data.len()) };
        Ok(())
    }

    /// Reads a slot back as host f32 data.
    pub fn read_f32(&self, id: SlotId) -> Result<Vec<f32>, RuntimeError> {
        self.check_f32(id)?;
        let len = self.slot_len(id);
        let ptr = self.data_ptr(id)?;
        let mut out = vec![0f32; len];
        // SAFETY: as in `write_f32`.
        unsafe { std::ptr::copy_nonoverlapping(ptr as *const f32, out.as_mut_ptr(), len) };
        Ok(out)
    }

    /// Writes raw bytes into a slot, for inputs that are not f32 (indices,
    /// masks). The caller is responsible for the layout matching the slot.
    pub fn write_raw(&mut self, id: SlotId, bytes: &[u8]) -> Result<(), RuntimeError> {
        let expected = self.slot_bytes(id)?;
        if bytes.len() as u64 != expected {
            return Err(RuntimeError::ByteLengthMismatch {
                slot: id,
                name: self.plan.plan.slot(id).name.clone(),
                expected,
                actual: bytes.len() as u64,
            });
        }
        let ptr = self.data_ptr(id)?;
        // SAFETY: the buffer is exactly `expected` bytes and `bytes` is too.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len()) };
        Ok(())
    }

    /// Reads a slot back as a **contiguous** byte buffer, materialising a
    /// strided view if the slot holds one.
    pub fn read_raw(&self, id: SlotId) -> Result<Vec<u8>, RuntimeError> {
        let root = self.aliases[id.0].unwrap_or(id);
        let buf = self
            .buffers
            .get(root.0)
            .and_then(Option::as_ref)
            .ok_or(RuntimeError::NullData { slot: id })?;
        if buf.ptr.is_null() {
            return Err(RuntimeError::NullData { slot: id });
        }
        Ok(materialise(buf.ptr, buf.shape.as_slice(), buf.strides.as_slice(), buf.rank, buf.elem_width))
    }

    /// Byte size of a slot's buffer.
    pub fn slot_bytes(&self, id: SlotId) -> Result<u64, RuntimeError> {
        let root = self.aliases[id.0].unwrap_or(id);
        self.buffers
            .get(root.0)
            .and_then(Option::as_ref)
            .map(|b| b.bytes)
            .ok_or(RuntimeError::NullData { slot: id })
    }

    fn check_f32(&self, id: SlotId) -> Result<(), RuntimeError> {
        let slot = self.plan.plan.slot(id);
        if slot.dtype != RsDtype::F32 {
            return Err(RuntimeError::NotF32 {
                slot: id,
                name: slot.name.clone(),
                dtype: slot.dtype.to_string(),
            });
        }
        Ok(())
    }

    fn data_ptr(&self, id: SlotId) -> Result<*mut c_void, RuntimeError> {
        let root = self.aliases[id.0].unwrap_or(id);
        self.buffers
            .get(root.0)
            .and_then(Option::as_ref)
            .map(|b| b.ptr)
            .ok_or(RuntimeError::NullData { slot: id })
    }

    /// Runs every step, in plan order.
    ///
    /// Not resumable: a second call re-runs from the top. A partially executed
    /// step would leave slot contents describing neither its input nor its
    /// output, which is worse than redoing the work.
    pub fn run(&mut self) -> Result<RunStats, RuntimeError> {
        let mut stats = RunStats {
            resident_bytes: self.stats.resident_bytes,
            ..Default::default()
        };

        for index in 0..self.plan.steps.len() {
            let mut adopted: Vec<(SlotId, RsTensor)> = Vec::new();
            let (inputs, outputs, label) = {
                let step = &self.plan.steps[index];
                let inputs = match step {
                    CompiledStep::Op { inputs, .. } => inputs.clone(),
                    CompiledStep::Intrinsic { input, .. } => vec![*input],
                };
                let outputs = match step {
                    CompiledStep::Op { outputs, .. } => outputs.clone(),
                    CompiledStep::Intrinsic { output, .. } => vec![*output],
                };
                (inputs, outputs, step.label())
            };

            let in_tensors = self.descriptors(&inputs)?;
            let mut out_tensors = self.descriptors(&outputs)?;

            match &self.plan.steps[index] {
                CompiledStep::Intrinsic { op, group, reduce, dim, .. } => {
                    let kind = match op.as_str() {
                        rustrain_plan::intrinsic::ALL_REDUCE => RsCollectiveKind::ALL_REDUCE,
                        rustrain_plan::intrinsic::ALL_GATHER => RsCollectiveKind::ALL_GATHER,
                        rustrain_plan::intrinsic::REDUCE_SCATTER => {
                            RsCollectiveKind::REDUCE_SCATTER
                        }
                        other => {
                            return Err(RuntimeError::Collective {
                                index,
                                op: other.to_string(),
                                reason: "no backend implements this intrinsic".to_string(),
                            });
                        }
                    };
                    let mut t = out_tensors.remove(0);
                    self.collectives
                        .execute(kind, *group, *reduce, *dim, &mut t)
                        .map_err(|reason| RuntimeError::Collective {
                            index,
                            op: label.clone(),
                            reason,
                        })?;
                    stats.collectives += 1;
                    stats.steps += 1;
                }

                CompiledStep::Op { op, attrs, .. } => {
                    let in_ptrs: Vec<*const RsTensor> =
                        in_tensors.iter().map(std::ptr::from_ref).collect();
                    let mut out_ptrs: Vec<*mut RsTensor> =
                        out_tensors.iter_mut().map(std::ptr::from_mut).collect();
                    let attrs_ptr = attrs.as_ptr();

                    let desc = op.desc();
                    let execute = desc.execute.ok_or_else(|| RuntimeError::Op {
                        index,
                        op: label.clone(),
                        message: "implementation has no execute function (validated at load)"
                            .to_string(),
                    })?;

                    let mut ctx = RsCtx {
                        user: std::ptr::null_mut(),
                        svc: self.services.as_ref() as *const RsServices,
                    };
                    // SAFETY: every descriptor points at storage owned by this
                    // executor, sized from the same slot table the compiler used
                    // for shape inference; the attribute array is owned by the
                    // step and outlives the call. The implementation may write
                    // only the outputs it declared, which `infer` verified.
                    let status = unsafe {
                        execute(
                            &mut ctx,
                            in_ptrs.as_ptr(),
                            in_ptrs.len() as u32,
                            out_ptrs.as_mut_ptr(),
                            out_ptrs.len() as u32,
                            attrs_ptr,
                        )
                    };
                    if status != 0 {
                        let message = desc
                            .last_error
                            .map(|f| {
                                // SAFETY: the plugin owns this function pointer.
                                let p = unsafe { f(&mut ctx) };
                                if p.is_null() {
                                    String::new()
                                } else {
                                    unsafe { std::ffi::CStr::from_ptr(p) }
                                        .to_string_lossy()
                                        .into_owned()
                                }
                            })
                            .unwrap_or_default();
                        return Err(RuntimeError::Op {
                            index,
                            op: label.clone(),
                            message: if message.is_empty() {
                                format!("execute returned {status}")
                            } else {
                                message
                            },
                        });
                    }
                    // A view operator hands back a descriptor pointing into its
                    // input. Adopt it, or every later read of this slot reads
                    // the executor's own untouched buffer.
                    for (slot, t) in outputs.iter().zip(&out_tensors) {
                        if !t.data.is_null() {
                            adopted.push((*slot, *t));
                        }
                    }

                    stats.ops += 1;
                    stats.steps += 1;
                }
            }

            for (slot, t) in adopted {
                let root = self.aliases[slot.0].unwrap_or(slot);
                if let Some(buf) = self.buffers.get_mut(root.0).and_then(Option::as_mut) {
                    buf.ptr = t.data;
                    buf.shape = t.shape;
                    buf.strides = t.stride;
                    buf.rank = t.rank;
                }
            }
        }

        self.stats = stats.clone();
        Ok(stats)
    }

    fn descriptors(&self, ids: &[SlotId]) -> Result<Vec<RsTensor>, RuntimeError> {
        ids.iter().map(|id| self.descriptor(*id)).collect()
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        // Only owners free; aliases point into the same allocation.
        for buf in self.buffers.iter().flatten() {
            if let Some((ptr, bytes)) = buf.owned {
                self.allocator.dealloc(ptr, bytes);
            }
        }
    }
}

/// Copies a strided view into a contiguous buffer.
///
/// Strides are in elements. A zero stride (from `broadcast`) repeats one element,
/// which a flat `copy_nonoverlapping` cannot express — it would read past the
/// allocation instead.
fn materialise(
    ptr: *const c_void,
    shape: &[i64],
    strides: &[i64],
    rank: u32,
    elem_width: u32,
) -> Vec<u8> {
    let rank = (rank as usize).min(shape.len());
    let shape = &shape[..rank];
    let strides = &strides[..rank];
    let total: usize = shape.iter().product::<i64>().max(0) as usize;
    let width = elem_width.max(1) as usize;
    let mut out = vec![0u8; total * width];
    if total == 0 || width == 0 {
        return out;
    }

    let mut idx = vec![0i64; rank];
    for linear in 0..total {
        let mut offset = 0i64;
        for d in 0..rank {
            offset += idx[d] * strides[d];
        }
        // SAFETY: the operator that produced this descriptor declared these
        // shape/strides over its own buffer, so `offset` is in bounds.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (ptr as *const u8).add(offset as usize * width),
                out.as_mut_ptr().add(linear * width),
                width,
            );
        }
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

/// Byte size of one slot's element buffer.
///
/// Sub-byte dtypes (fp4) have no whole-byte width yet, and the runtime refuses
/// them rather than guessing a packing.
fn slot_element_bytes(slot: &Slot) -> Result<u64, String> {
    let width = slot
        .dtype
        .byte_width()
        .ok_or_else(|| format!("dtype {} has no whole-byte width", slot.dtype))?;
    let numel = slot.shape.iter().product::<i64>().max(0) as u64;
    Ok(numel * width as u64)
}

/// The service table handed to plugins.
///
/// Allocation, stream and collective callbacks are not wired yet: a provider that
/// wants to allocate through the framework rather than inside its own runtime
/// cannot be served by this executor. The table is still passed, so the ABI
/// shape is exercised and a provider can tell that it has no services.
fn no_services() -> RsServices {
    RsServices {
        abi_version: rustrain_abi::ABI_VERSION,
        struct_size: std::mem::size_of::<RsServices>() as u32,
        user: std::ptr::null_mut(),
        alloc: None,
        free: None,
        current_stream: None,
        collective: None,
        log: None,
    }
}

/// Slot ids a caller must fill before the first run, with their kinds.
pub fn required_inputs(plan: &CompiledPlan) -> Vec<(SlotId, SlotKind)> {
    plan.plan
        .input_slots()
        .into_iter()
        .map(|id| (id, plan.plan.slot(id).kind))
        .collect()
}
