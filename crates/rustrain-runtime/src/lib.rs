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
//!   in-process one; [`CudaAllocator`] serves device memory through the
//!   runtime-loaded CUDA driver (no CUDA in the dependency closure).
//! * [`CollectiveBackend`] — what a spliced collective actually does.
//!   [`collective::SingleRank`] is the identity at world 1;
//!   [`collective::ThreadBackend`] is the D6 reference transport (N threads,
//!   shared buffers, rendezvous). Both sit behind the same trait so a
//!   GPU/NCCL backend can replace them without the executor changing.

// Same reasoning as `rustrain-plan`: the error carries structured diagnostics.
#![allow(clippy::result_large_err)]

pub mod collective;
pub mod conformance;
pub mod device;
pub mod nccl;

pub use collective::{
    CollectiveBackend, CollectiveKind, CollectiveReport, CollectiveRequest, SharedBackend,
    SingleRank, ThreadBackend, ThreadShared,
};
pub use device::CudaAllocator;
pub use nccl::NcclBackend;

use std::ffi::c_void;

use rustrain_abi::ffi::{RsCtx, RsDeviceKind, RsDtype, RsServices, RsTensor};
use rustrain_plan::{CompiledPlan, CompiledStep, SlotId, SlotKind};

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

    /// A byte-range write was asked for a slot whose buffer is a strided view.
    #[error(
        "slot `{slot:?}` ({name}) is not densely laid out (strides {strides:?}), so a write at a \
         byte offset would land in the wrong elements"
    )]
    NotDense {
        slot: SlotId,
        name: String,
        strides: Vec<i64>,
    },

    #[error("slot {slot:?} ({name}) is {dtype}, not f32")]
    NotF32 {
        slot: SlotId,
        name: String,
        dtype: String,
    },

    /// A partial raw write addresses elements, so a dtype whose elements are not whole bytes
    /// (fp4) has no element offset to write at.
    #[error(
        "slot {slot:?} ({name}) is {dtype}, whose elements are not whole bytes, so a raw write \
         at an element offset cannot address them"
    )]
    SubByteElements {
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

    #[error("slot {slot:?} ({name}) has no placement in the memory plan")]
    UnplannedSlot { slot: SlotId, name: String },

    #[error(
        "node {node} ({op}) asks for memory policy `{policy}`, which this runtime cannot \
         execute; a plan must not rely on a strategy that is not implemented"
    )]
    UnsupportedMemoryPolicy {
        node: usize,
        op: String,
        policy: String,
    },

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

    #[error("cannot copy {bytes} bytes for slot {slot:?}: {reason}")]
    Copy {
        slot: SlotId,
        bytes: u64,
        reason: String,
    },
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

    /// Copies a host slice into an allocation this allocator made.
    ///
    /// The default is an honest host `memcpy`, correct for [`HostAllocator`];
    /// a device allocator overrides it with a driver copy. The executor routes
    /// every host write through this, never through the raw pointer, so the
    /// memory's actual location stays the allocator's business.
    fn copy_in(&mut self, dst: *mut c_void, bytes: u64, src: &[u8]) -> Result<(), String> {
        if dst.is_null() {
            return Err("copy_in: null destination".to_string());
        }
        if src.len() as u64 != bytes {
            return Err(format!(
                "copy_in: {bytes} byte(s) requested but the slice holds {}",
                src.len()
            ));
        }
        // SAFETY: the caller guarantees `dst` holds at least `bytes` (== the
        // slice length) writable bytes, and the slice is valid to read.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst as *mut u8, src.len()) };
        Ok(())
    }

    /// Copies an allocation back into a fresh host `Vec<u8>`.
    ///
    /// The default is an honest host `memcpy`; a device allocator overrides it
    /// with a driver copy (synchronising first: the data may have been written
    /// by kernels on a stream the driver copy does not order against).
    fn copy_out(&self, src: *const c_void, bytes: u64) -> Result<Vec<u8>, String> {
        if src.is_null() {
            return Err("copy_out: null source".to_string());
        }
        let n = usize::try_from(bytes)
            .map_err(|_| format!("copy_out: {bytes} bytes does not fit a host buffer"))?;
        let mut out = vec![0u8; n];
        // SAFETY: the caller guarantees `src` holds at least `bytes` readable
        // bytes.
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, out.as_mut_ptr(), n) };
        Ok(out)
    }
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
}

/// What a run did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunStats {
    pub steps: usize,
    pub ops: usize,
    pub collectives: usize,
    pub resident_bytes: u64,
    /// Bytes this rank handed to the collective exchanges, summed over the run.
    pub collective_sent_bytes: u64,
    /// Bytes this rank took out of the collective exchanges, summed over the run.
    pub collective_recv_bytes: u64,
    /// One entry per executed collective, in execution order — the per-step
    /// evidence the D6 metrics report aggregates by kind and group.
    pub collective_records: Vec<CollectiveRecord>,
    /// Nanoseconds spent inside collective backends, summed over the run. Kept in
    /// nanoseconds so `RunStats` stays `Eq` (a float here would make the whole
    /// struct uncomparable) and so the metrics report can decide its own unit.
    pub collective_nanos: u64,
    /// The same, per intrinsic op name: `intrinsic.all_gather` → nanoseconds. Read next to
    /// `first_collective_nanos`: a communicator is created per *group* (one mask, one
    /// `ncclCommInitRank`), not per kind, so the by-kind split deliberately does not try to say
    /// which entry paid for one.
    pub collective_nanos_by_kind: std::collections::BTreeMap<String, u64>,
    /// Nanoseconds the first *distributing* collective of the run took — the first one whose
    /// group has more than one rank, because a degree-1 group is a local copy and says nothing
    /// about the world's timing.
    ///
    /// Where a backend creates its communicators on first use, this carries that handshake; where
    /// the runner warms them beforehand (the NCCL path does, during the checkpoint load), what is
    /// left here is the wait for the other ranks to reach this collective — which is exactly what
    /// a per-rank wall difference measures when the ranks end together.
    pub first_collective_nanos: u64,
    /// Nanoseconds spent inside plugin `execute` calls, summed over the run.
    pub op_nanos: u64,
    /// Collectives that travelled through host staging, and those that moved device bytes
    /// directly. Reported because staging is where an expensive exchange silently becomes
    /// expensive: a non-contiguous operand pays a device→host→device round trip, and this count is
    /// the only place that says so.
    pub staged_collectives: u64,
    pub direct_collectives: u64,
}

/// One executed collective, as the metrics report reads it. The group travels
/// as raw mask bits: a name needs the mesh, which a bare record does not carry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CollectiveRecord {
    /// Step index in the compiled plan.
    pub step: usize,
    /// The intrinsic op name (`intrinsic.all_reduce`, …).
    pub kind: String,
    /// The group's mask bits.
    pub group: u32,
    pub sent_bytes: u64,
    pub recv_bytes: u64,
}

/// Walks a [`CompiledPlan`].
pub struct Executor {
    plan: CompiledPlan,
    allocator: Box<dyn Allocator + Send>,
    collectives: Box<dyn CollectiveBackend + Send>,
    buffers: Vec<Option<SlotBuffer>>,
    /// Buffers handed to collective outputs that could not reuse their input's
    /// storage (grown gathers, strided-view inputs); allocated through the
    /// allocator outside the two planner regions and freed on drop.
    loose: Vec<(*mut c_void, u64)>,
    /// The two allocations every slot lives inside. `persistent` holds weights,
    /// gradients and optimizer state for the whole run; `pool` holds activations
    /// whose offsets the planner already assigned so that non-overlapping
    /// lifetimes share storage.
    persistent_region: Option<(*mut c_void, u64)>,
    pool_region: Option<(*mut c_void, u64)>,
    services: Box<RsServices>,
    stats: RunStats,
}

impl Executor {
    /// Prepares storage for every slot the plan declares, at the offsets the
    /// memory planner assigned.
    ///
    /// Allocation is two regions rather than one per slot, which is what makes
    /// the planner's reuse decisions real instead of advisory: two activations
    /// the planner found non-overlapping are literally the same bytes here.
    pub fn new(
        plan: CompiledPlan,
        mut allocator: Box<dyn Allocator + Send>,
        collectives: Box<dyn CollectiveBackend + Send>,
    ) -> Result<Self, RuntimeError> {
        let n = plan.plan.slots.len();

        if let Some((node, op, policy)) = plan.memory.unsupported.first() {
            return Err(RuntimeError::UnsupportedMemoryPolicy {
                node: node.0,
                op: op.clone(),
                policy: format!("{policy:?}"),
            });
        }

        let device = allocator.device();

        // Two regions, sized by the planner. A zero-sized region is skipped
        // rather than allocated, so a plan with no activations costs one
        // allocation instead of one per slot.
        let mut alloc_region =
            |bytes: u64, what: &str| -> Result<Option<(*mut c_void, u64)>, RuntimeError> {
                if bytes == 0 {
                    return Ok(None);
                }
                let ptr = allocator
                    .alloc(bytes, device)
                    .map_err(|reason| RuntimeError::Alloc {
                        slot: SlotId(0),
                        bytes,
                        reason: format!("{what} region: {reason}"),
                    })?;
                Ok(Some((ptr, bytes)))
            };
        let persistent_region = alloc_region(plan.memory.persistent_bytes, "persistent")?;
        let pool_region = alloc_region(plan.memory.transient_pool_bytes, "activation pool")?;

        let base =
            |region: Option<(*mut c_void, u64)>, what: &str| -> Result<*mut c_void, RuntimeError> {
                region.map(|(p, _)| p).ok_or(RuntimeError::Alloc {
                    slot: SlotId(0),
                    bytes: 0,
                    reason: format!("the plan needs a {what} region but none was allocated"),
                })
            };

        // Buffers the executor hands to a collective output that cannot reuse
        // its input's storage (a grown gather, or an input that is a strided
        // view). The planner counted these under the alias; the extra region is
        // freed on drop and reported as part of the peak in the metrics.
        let mut loose: Vec<(*mut c_void, u64)> = Vec::new();

        let mut buffers: Vec<Option<SlotBuffer>> = Vec::with_capacity(n);
        for (i, slot) in plan.plan.slots.iter().enumerate() {
            let alloc =
                plan.memory
                    .allocation(SlotId(i))
                    .ok_or_else(|| RuntimeError::UnplannedSlot {
                        slot: SlotId(i),
                        name: slot.name.clone(),
                    })?;

            // A spliced collective whose output fits its input reuses the
            // input's storage (in place); the planner marks that as
            // `Placement::Aliased(root)`. Two conditions make reuse unsafe:
            // the output is larger than the input (all_gather, an uneven
            // all_to_all — the planner already refuses to alias those), or the
            // input holds a strided view whose data is not a dense prefix of
            // its region. Either way the output gets its own buffer at its
            // planned size instead.
            if let rustrain_plan::Placement::Aliased(root) = alloc.placement {
                let src = buffers[root.0]
                    .as_ref()
                    .ok_or(RuntimeError::NullData { slot: root })?;
                let (shape, strides, rank, elem_width) = slot_descriptor_shape(slot);
                if is_dense(src) && alloc.bytes <= src.bytes {
                    buffers.push(Some(SlotBuffer {
                        ptr: src.ptr,
                        shape,
                        strides,
                        rank,
                        elem_width,
                        bytes: alloc.bytes,
                    }));
                } else {
                    let ptr = allocator.alloc(alloc.bytes, device).map_err(|reason| {
                        RuntimeError::Alloc {
                            slot: SlotId(i),
                            bytes: alloc.bytes,
                            reason: format!("collective output region: {reason}"),
                        }
                    })?;
                    loose.push((ptr, alloc.bytes));
                    buffers.push(Some(SlotBuffer {
                        ptr,
                        shape,
                        strides,
                        rank,
                        elem_width,
                        bytes: alloc.bytes,
                    }));
                }
                continue;
            }

            let ptr = match alloc.placement {
                // SAFETY: the planner sized each region to cover every offset it
                // assigned into it, so `offset..offset + bytes` is in bounds.
                rustrain_plan::Placement::Persistent { offset } => unsafe {
                    (base(persistent_region, "persistent")? as *mut u8).add(offset as usize)
                        as *mut c_void
                },
                rustrain_plan::Placement::Pool { offset } => unsafe {
                    (base(pool_region, "activation pool")? as *mut u8).add(offset as usize)
                        as *mut c_void
                },
                rustrain_plan::Placement::Aliased(_) => unreachable!("handled above"),
                // The compiler records a policy the runtime cannot execute, and
                // `new` refuses such a plan above; reaching here means a slot was
                // planned as non-resident without being reported.
                rustrain_plan::Placement::NonResident => {
                    return Err(RuntimeError::UnsupportedMemoryPolicy {
                        node: 0,
                        op: slot.name.clone(),
                        policy: format!("{:?}", alloc.policy),
                    });
                }
            };

            let (shape, strides, rank, elem_width) = slot_descriptor_shape(slot);
            buffers.push(Some(SlotBuffer {
                ptr,
                shape,
                strides,
                rank,
                elem_width,
                bytes: alloc.bytes,
            }));
        }

        // The planner's projection, so a caller comparing `RunStats` against
        // `plan explain` sees the same number.
        let resident_bytes = plan.memory.peak_bytes;

        Ok(Self {
            plan,
            allocator,
            collectives,
            buffers,
            loose,
            persistent_region,
            pool_region,
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
        let buf = self
            .buffers
            .get(id.0)
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
        let ptr = self.dense_ptr(id)?;
        let bytes = std::mem::size_of_val(data) as u64;
        // SAFETY: `data` holds exactly `expected` f32 (checked above), so the
        // byte view covers exactly `bytes`.
        let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes as usize) };
        self.allocator
            .copy_in(ptr, bytes, raw)
            .map_err(|reason| RuntimeError::Copy {
                slot: id,
                bytes,
                reason,
            })
    }

    /// Writes part of a slot: `data` lands at `element_offset`, and the rest of the slot is left
    /// alone.
    ///
    /// This is what lets a loader stream a weight in pieces instead of materialising all of it in
    /// host memory first — the caller writes every range exactly once, and the slot is complete
    /// when the last chunk lands. The range is bounds-checked here so a chunked writer cannot
    /// scribble past its slot.
    pub fn write_f32_at(
        &mut self,
        id: SlotId,
        element_offset: usize,
        data: &[f32],
    ) -> Result<(), RuntimeError> {
        self.check_f32(id)?;
        let len = self.slot_len(id);
        let end = element_offset.saturating_add(data.len());
        if end > len {
            return Err(RuntimeError::LengthMismatch {
                slot: id,
                name: self.plan.plan.slot(id).name.clone(),
                expected: len,
                actual: end,
            });
        }
        let ptr = self.dense_ptr(id)?;
        let bytes = std::mem::size_of_val(data) as u64;
        // SAFETY: `data` holds `data.len()` f32, so the byte view covers exactly `bytes`; the
        // destination is `element_offset` f32 past the slot's start, and the check above proves the
        // whole range is inside the allocation.
        let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes as usize) };
        // SAFETY: pointer arithmetic on the slot's own allocation, inside the bounds just checked.
        let dst = unsafe { (ptr as *mut u8).add(element_offset * std::mem::size_of::<f32>()) };
        self.allocator
            .copy_in(dst as *mut c_void, bytes, raw)
            .map_err(|reason| RuntimeError::Copy {
                slot: id,
                bytes,
                reason,
            })
    }

    /// Writes part of a slot in raw bytes: `bytes` land at `element_offset` elements past the
    /// slot's start, and the rest of the slot is left alone.
    ///
    /// The raw twin of [`Self::write_f32_at`]: the streaming loader feeds a bf16 weight slot the
    /// checkpoint's own little-endian bytes, at the slot's declared element width, where
    /// `write_f32_at` would demand an f32 slot and quadruple the traffic. The caller is
    /// responsible for the byte width matching the slot's declared dtype; the range is
    /// bounds-checked here, so a chunked writer cannot scribble past its slot.
    pub fn write_raw_at(
        &mut self,
        id: SlotId,
        element_offset: usize,
        bytes: &[u8],
    ) -> Result<(), RuntimeError> {
        let slot = self.plan.plan.slot(id);
        let width = slot
            .dtype
            .byte_width()
            .ok_or_else(|| RuntimeError::SubByteElements {
                slot: id,
                name: slot.name.clone(),
                dtype: slot.dtype.to_string(),
            })?;
        let len = self.slot_len(id);
        let elements = bytes.len() / width as usize;
        let end = element_offset.saturating_add(elements);
        if bytes.len() % width as usize != 0 || end > len {
            return Err(RuntimeError::LengthMismatch {
                slot: id,
                name: slot.name.clone(),
                expected: len,
                actual: end,
            });
        }
        let ptr = self.dense_ptr(id)?;
        // SAFETY: pointer arithmetic on the slot's own allocation: the destination is
        // `element_offset` elements past its start, and the check above proves the whole range
        // is inside it.
        let dst = unsafe { (ptr as *mut u8).add(element_offset * width as usize) };
        self.allocator
            .copy_in(dst as *mut c_void, bytes.len() as u64, bytes)
            .map_err(|reason| RuntimeError::Copy {
                slot: id,
                bytes: bytes.len() as u64,
                reason,
            })
    }

    /// Reads a slot back as host f32 data.
    pub fn read_f32(&self, id: SlotId) -> Result<Vec<f32>, RuntimeError> {
        self.check_f32(id)?;
        let len = self.slot_len(id);
        let ptr = self.data_ptr(id)?;
        let bytes = (len * std::mem::size_of::<f32>()) as u64;
        let host = self
            .allocator
            .copy_out(ptr, bytes)
            .map_err(|reason| RuntimeError::Copy {
                slot: id,
                bytes,
                reason,
            })?;
        let mut out = vec![0f32; len];
        // SAFETY: `copy_out` returned exactly `bytes` = `len` f32.
        unsafe {
            std::ptr::copy_nonoverlapping(host.as_ptr() as *const f32, out.as_mut_ptr(), len)
        };
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
        self.allocator
            .copy_in(ptr, expected, bytes)
            .map_err(|reason| RuntimeError::Copy {
                slot: id,
                bytes: expected,
                reason,
            })
    }

    /// Reads a slot back as a **contiguous** byte buffer, materialising a
    /// strided view if the slot holds one.
    ///
    /// The whole buffer the view covers comes to host in one piece first
    /// (`copy_out(buf.ptr, buf.bytes)`) and `materialise` walks that host
    /// slice: a strided view cannot be walked element by element through
    /// device memory, and `buf.bytes` is exactly the byte span the view reads
    /// (see the adoption site in `run`).
    pub fn read_raw(&self, id: SlotId) -> Result<Vec<u8>, RuntimeError> {
        let buf = self
            .buffers
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or(RuntimeError::NullData { slot: id })?;
        if buf.ptr.is_null() {
            return Err(RuntimeError::NullData { slot: id });
        }
        let host = self
            .allocator
            .copy_out(buf.ptr, buf.bytes)
            .map_err(|reason| RuntimeError::Copy {
                slot: id,
                bytes: buf.bytes,
                reason,
            })?;
        Ok(materialise(
            host.as_ptr() as *const c_void,
            buf.shape.as_slice(),
            buf.strides.as_slice(),
            buf.rank,
            buf.elem_width,
        ))
    }

    /// Byte size of a slot's buffer.
    pub fn slot_bytes(&self, id: SlotId) -> Result<u64, RuntimeError> {
        self.buffers
            .get(id.0)
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

    /// The slot's data pointer, **only when the slot is densely laid out**.
    ///
    /// A host write addresses the slot by byte offset from its base pointer, which is only the
    /// element at that offset when the buffer is canonical row-major — and a view operator
    /// (`transpose`, an inner-axis `narrow`, `broadcast`) leaves the slot describing a different
    /// stride arrangement. Writing anyway lands in the wrong elements *silently*, because the
    /// shape still checks out; the whole-slot writer has the same assumption, so both go through
    /// here.
    fn dense_ptr(&self, id: SlotId) -> Result<*mut c_void, RuntimeError> {
        let buf = self
            .buffers
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or(RuntimeError::NullData { slot: id })?;
        if !is_dense(buf) {
            let rank = (buf.rank as usize).min(buf.shape.len());
            return Err(RuntimeError::NotDense {
                slot: id,
                name: self.plan.plan.slot(id).name.clone(),
                strides: buf.strides[..rank].to_vec(),
            });
        }
        Ok(buf.ptr)
    }

    fn data_ptr(&self, id: SlotId) -> Result<*mut c_void, RuntimeError> {
        self.buffers
            .get(id.0)
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
        // The mesh, for the one question the timing asks of it: does this collective's group have
        // more than one rank? A degree-1 group is a local copy, and letting one capture
        // `first_collective_nanos` would report a wait that no other rank can be part of.
        let mesh = self.plan.plan.meta.mesh.to_mesh().ok();
        // An opt-in per-step trace, because "the forward took 6 s" is not a debugging surface: with
        // small tensors a step's cost is neither FLOPs nor bytes, and only a per-label breakdown
        // says which one to look at. `RUSTRAIN_STEP_TRACE=<n>` prints the n heaviest labels (by
        // total time) to stderr and leaves every metric alone.
        let trace_top: Option<usize> = std::env::var("RUSTRAIN_STEP_TRACE")
            .ok()
            .and_then(|value| value.parse().ok());
        let mut traced: Vec<(usize, String, u64)> = Vec::new();
        // An explicit flag rather than "the field is still zero": a first collective that measured
        // zero nanoseconds would be indistinguishable from "not yet".
        let mut first_distributing_seen = false;

        for index in 0..self.plan.steps.len() {
            let step_started = std::time::Instant::now();
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
                CompiledStep::Intrinsic {
                    op,
                    group,
                    reduce,
                    dim,
                    split,
                    src,
                    ..
                } => {
                    let kind = match op.as_str() {
                        rustrain_plan::intrinsic::ALL_REDUCE => CollectiveKind::AllReduce,
                        rustrain_plan::intrinsic::ALL_GATHER => CollectiveKind::AllGather,
                        rustrain_plan::intrinsic::REDUCE_SCATTER => CollectiveKind::ReduceScatter,
                        rustrain_plan::intrinsic::BROADCAST => CollectiveKind::Broadcast,
                        rustrain_plan::intrinsic::ALL_TO_ALL => CollectiveKind::AllToAll,
                        rustrain_plan::intrinsic::SYNC => CollectiveKind::Sync,
                        other => {
                            return Err(RuntimeError::Collective {
                                index,
                                op: other.to_string(),
                                reason: "no backend implements this intrinsic".to_string(),
                            });
                        }
                    };
                    let request = CollectiveRequest {
                        kind,
                        group: *group,
                        reduce: *reduce,
                        dim: *dim,
                        split: split.clone(),
                        src: *src,
                    };
                    let input = in_tensors[0];
                    let mut output = out_tensors.remove(0);
                    let collective_started = std::time::Instant::now();
                    let report = self
                        .collectives
                        .execute(&request, &input, &mut output, self.allocator.as_mut())
                        .map_err(|reason| RuntimeError::Collective {
                            index,
                            op: label.clone(),
                            reason,
                        })?;
                    let nanos = collective_started.elapsed().as_nanos() as u64;
                    let distributes = mesh
                        .as_ref()
                        .and_then(|mesh| group.degree(mesh).ok())
                        .map(|degree| degree > 1)
                        .unwrap_or(false);
                    if distributes && !first_distributing_seen {
                        first_distributing_seen = true;
                        stats.first_collective_nanos = nanos;
                    }
                    stats.collective_nanos += nanos;
                    *stats
                        .collective_nanos_by_kind
                        .entry(op.clone())
                        .or_default() += nanos;
                    stats.collectives += 1;
                    stats.steps += 1;
                    stats.collective_sent_bytes += report.sent_bytes;
                    stats.collective_recv_bytes += report.recv_bytes;
                    stats.collective_records.push(CollectiveRecord {
                        step: index,
                        kind: op.clone(),
                        group: group.bits(),
                        sent_bytes: report.sent_bytes,
                        recv_bytes: report.recv_bytes,
                    });
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
                    let op_started = std::time::Instant::now();
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

                    stats.op_nanos += op_started.elapsed().as_nanos() as u64;
                    stats.ops += 1;
                    stats.steps += 1;
                }
            }

            if trace_top.is_some() {
                traced.push((
                    index,
                    self.plan.steps[index].label(),
                    step_started.elapsed().as_nanos() as u64,
                ));
            }

            for (slot, t) in adopted {
                if let Some(buf) = self.buffers.get_mut(slot.0).and_then(Option::as_mut) {
                    buf.ptr = t.data;
                    buf.shape = t.shape;
                    buf.strides = t.stride;
                    buf.rank = t.rank;
                    // The slot now holds a strided view into another buffer, so
                    // its bytes are the contiguous span the view reads — the
                    // amount `read_raw` must copy to host before materialising.
                    buf.bytes = span_bytes(buf.shape, buf.strides, buf.rank, buf.elem_width);
                }
            }
        }

        self.stats = stats.clone();
        let (staged, direct) = self.collectives.path_counts();
        stats.staged_collectives = staged;
        stats.direct_collectives = direct;

        if let Some(top) = trace_top {
            let mut by_label: std::collections::BTreeMap<String, (usize, u64)> =
                std::collections::BTreeMap::new();
            for (_, label, nanos) in &traced {
                let entry = by_label.entry(label.clone()).or_default();
                entry.0 += 1;
                entry.1 += nanos;
            }
            let mut rows: Vec<(String, usize, u64)> = by_label
                .into_iter()
                .map(|(label, (count, total))| (label, count, total))
                .collect();
            rows.sort_by_key(|(_, _, total)| std::cmp::Reverse(*total));
            let traced_total: u64 = rows.iter().map(|(_, _, total)| *total).sum();
            eprintln!(
                "step trace: {} step(s), {:.3} s inside them; heaviest {} label(s):",
                traced.len(),
                traced_total as f64 / 1e9,
                top
            );
            for (label, count, total) in rows.into_iter().take(top) {
                eprintln!(
                    "  {label}: {count} call(s), {:.3} s total, {:.3} ms each",
                    total as f64 / 1e9,
                    total as f64 / 1e6 / count.max(1) as f64
                );
            }
            // An aggregate hides the shape of the distribution: one 2-second call and 41 uniform
            // 60 ms calls have the same total and completely different causes. The slowest single
            // steps are printed next to it so the two are distinguishable at a glance.
            let mut singles = traced.clone();
            singles.sort_by_key(|(_, _, nanos)| std::cmp::Reverse(*nanos));
            eprintln!("  slowest single step(s):");
            for (index, label, nanos) in singles.iter().take(top.min(5)) {
                eprintln!("    step {index} {label}: {:.3} ms", *nanos as f64 / 1e6);
            }
        }
        Ok(stats)
    }

    fn descriptors(&self, ids: &[SlotId]) -> Result<Vec<RsTensor>, RuntimeError> {
        ids.iter().map(|id| self.descriptor(*id)).collect()
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        // Only owners free; in-place collective outputs point into these
        // regions, and `loose` holds the buffers that could not alias.
        for region in [self.persistent_region, self.pool_region]
            .into_iter()
            .flatten()
        {
            self.allocator.dealloc(region.0, region.1);
        }
        for (ptr, bytes) in self.loose.drain(..) {
            self.allocator.dealloc(ptr, bytes);
        }
    }
}

/// The canonical contiguous descriptor a plan slot's buffer starts with:
/// row-major strides over the slot's shape, `width` bytes per element.
fn slot_descriptor_shape(slot: &rustrain_plan::Slot) -> ([i64; 8], [i64; 8], u32, u32) {
    let mut shape = [0i64; rustrain_abi::ffi::MAX_RANK];
    let mut strides = [0i64; rustrain_abi::ffi::MAX_RANK];
    let rank = (slot.shape.len() as u32).min(rustrain_abi::ffi::MAX_RANK as u32);
    let mut acc = 1i64;
    for d in (0..rank as usize).rev() {
        shape[d] = slot.shape[d];
        strides[d] = acc;
        acc *= slot.shape[d].max(1);
    }
    let elem_width = slot.dtype.byte_width().unwrap_or(4);
    (shape, strides, rank, elem_width)
}

/// Whether a buffer holds a dense, contiguous region: canonical row-major
/// strides and exactly its own bytes. A collective output may reuse such a
/// buffer in place; a strided view (transpose, an inner-dim narrow) is handed
/// its own buffer instead.
fn is_dense(buf: &SlotBuffer) -> bool {
    let rank = (buf.rank as usize).min(buf.shape.len());
    let numel: i64 = buf.shape[..rank].iter().product::<i64>().max(0);
    let mut acc = 1i64;
    for d in (0..rank).rev() {
        if buf.strides[d] != acc {
            return false;
        }
        acc *= buf.shape[d].max(1);
    }
    buf.bytes == (numel.max(0) as u64) * (buf.elem_width.max(1) as u64)
}

/// The contiguous byte span a strided view covers: from its `data` pointer up
/// to and including the highest element `materialise` walks. `read_raw` copies
/// exactly this many bytes to host, so the span must be the largest offset
/// plus one element width.
fn span_bytes(shape: [i64; 8], strides: [i64; 8], rank: u32, elem_width: u32) -> u64 {
    let rank = (rank as usize).min(shape.len());
    let width = elem_width.max(1) as u64;
    let mut max_offset = 0u64;
    for d in 0..rank {
        let dim = shape[d].max(0) as u64;
        if dim == 0 {
            continue;
        }
        // Negative strides do not occur in the operators this runtime has run;
        // treating them as 0 matches a descriptor whose data points at the
        // lowest address it reads.
        max_offset += (dim - 1) * strides[d].max(0) as u64;
    }
    (max_offset + 1) * width
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

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(shape: [i64; 8], strides: [i64; 8], rank: u32, width: u32, bytes: u64) -> SlotBuffer {
        SlotBuffer {
            ptr: std::ptr::null_mut(),
            shape,
            strides,
            rank,
            elem_width: width,
            bytes,
        }
    }

    /// A dense buffer (canonical strides, exactly its own bytes) may be reused
    /// in place by a collective; a strided view (a transpose) may not — the
    /// executor hands it a fresh buffer instead of writing a contiguous result
    /// over scattered elements.
    #[test]
    fn density_gates_the_in_place_collective_reuse() {
        // [2, 3] f32, contiguous.
        let dense = buf([2, 3, 0, 0, 0, 0, 0, 0], [3, 1, 0, 0, 0, 0, 0, 0], 2, 4, 24);
        assert!(is_dense(&dense));
        // The same logical shape with transposed strides.
        let transposed = buf([2, 3, 0, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0], 2, 4, 24);
        assert!(!is_dense(&transposed));
        // A narrow along dim 0 of a larger tensor keeps canonical strides and
        // owns a contiguous region — dense, safe to write in place.
        let narrow_rows = buf([2, 3, 0, 0, 0, 0, 0, 0], [3, 1, 0, 0, 0, 0, 0, 0], 2, 4, 24);
        assert!(is_dense(&narrow_rows));
        // A size-1 dim with stride 0 (broadcast) is not dense.
        let broadcast = buf([1, 3, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], 2, 4, 12);
        assert!(!is_dense(&broadcast));
    }

    /// The byte span a view covers: from its data pointer to its highest
    /// element. `read_raw` copies exactly this much, so the span must never
    /// under-count a strided view.
    #[test]
    fn span_bytes_covers_the_strided_view() {
        // A narrow along the last dim of a [4, 8] f32 tensor: elements sit at
        // offsets i*8 + j, so the highest one is 27 and the span is 28 f32.
        let narrow = span_bytes([4, 4, 0, 0, 0, 0, 0, 0], [8, 1, 0, 0, 0, 0, 0, 0], 2, 4);
        assert_eq!(narrow, 28 * 4);
        // A transpose of [4, 8]: highest offset 7*1 + 3*8 = 31, span 32 f32.
        let transposed = span_bytes([8, 4, 0, 0, 0, 0, 0, 0], [1, 8, 0, 0, 0, 0, 0, 0], 2, 4);
        assert_eq!(transposed, 32 * 4);
        // A broadcast [4, 1] -> [4, 8]: stride 0 contributes nothing, the span
        // is the input's own 4 f32.
        let broadcast = span_bytes([4, 8, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], 2, 4);
        assert_eq!(broadcast, 4 * 4);
    }

    /// The default `copy_in`/`copy_out` do the honest host thing, so an
    /// allocator that implements only the three core methods keeps working.
    #[test]
    fn allocator_defaults_copy_on_the_host() {
        let mut allocator = HostAllocator::new();
        let ptr = allocator.alloc(64, RsDeviceKind::CPU).unwrap();
        let data: Vec<u8> = (0..64u8).collect();
        allocator.copy_in(ptr, 64, &data).unwrap();
        assert_eq!(allocator.copy_out(ptr, 64).unwrap(), data);
        allocator.dealloc(ptr, 64);
    }

    /// Proves the executor moves host data through the allocator's copy hooks
    /// instead of dereferencing the slot pointer directly: the counters only
    /// move when `write_f32`/`write_raw`/`read_f32`/`read_raw` call the trait.
    /// Without the routing, every counter stays 0 and this test fails.
    #[test]
    fn executor_routes_host_data_through_the_allocator_copy_hooks() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use rustrain_abi::Plugin;
        use rustrain_parallel::{Mesh, ParallelConfig};
        use rustrain_plan::{Attrs, OpRef, PlanBuilder, SlotKind};

        struct RecordingAllocator {
            inner: HostAllocator,
            copies_in: Arc<AtomicUsize>,
            copies_out: Arc<AtomicUsize>,
        }

        unsafe impl Allocator for RecordingAllocator {
            fn alloc(&mut self, bytes: u64, device: RsDeviceKind) -> Result<*mut c_void, String> {
                self.inner.alloc(bytes, device)
            }

            fn dealloc(&mut self, ptr: *mut c_void, bytes: u64) {
                self.inner.dealloc(ptr, bytes);
            }

            fn device(&self) -> RsDeviceKind {
                self.inner.device()
            }

            fn copy_in(&mut self, dst: *mut c_void, bytes: u64, src: &[u8]) -> Result<(), String> {
                self.copies_in.fetch_add(1, Ordering::SeqCst);
                self.inner.copy_in(dst, bytes, src)
            }

            fn copy_out(&self, src: *const c_void, bytes: u64) -> Result<Vec<u8>, String> {
                self.copies_out.fetch_add(1, Ordering::SeqCst);
                self.inner.copy_out(src, bytes)
            }
        }

        let mut registry = rustrain_ops::Registry::new();
        // SAFETY: the built-in provider descriptor is leaked by the builder and
        // lives for the process.
        let reference = unsafe { Plugin::from_static(rustrain_kernels::plugin(), "<built-in>") }
            .expect("the built-in provider passes ABI validation");
        registry.add_plugin(reference).expect("registering it");
        let recipe = rustrain_ops::Recipe::from_toml("[kernel]\ndefault = \"reference\"\n")
            .expect("recipe parses");

        let mut b = PlanBuilder::new(
            "copy-routing",
            rustrain_ops::Phase::Forward,
            Mesh::from_config(&ParallelConfig::default()).fingerprint(),
        );
        let x = b.slot("x", RsDtype::F32, vec![4], SlotKind::Input);
        let y = b.slot("y", RsDtype::F32, vec![4], SlotKind::Output);
        b.node(
            OpRef::new("elementwise_unary"),
            vec![x],
            vec![y],
            Attrs::new().set("kind", "relu"),
            "relu",
        );
        let plan = b.build().unwrap();
        let compiled =
            rustrain_plan::Compiler::new(&registry, &recipe, rustrain_ops::TargetEnv::default())
                .compile(&plan)
                .unwrap();

        let copies_in = Arc::new(AtomicUsize::new(0));
        let copies_out = Arc::new(AtomicUsize::new(0));
        let mut executor = Executor::new(
            compiled,
            Box::new(RecordingAllocator {
                inner: HostAllocator::new(),
                copies_in: copies_in.clone(),
                copies_out: copies_out.clone(),
            }),
            Box::new(SingleRank::new(1)),
        )
        .unwrap();

        executor.write_f32(x, &[1.0, -2.0, 3.0, -4.0]).unwrap();
        // The same values through the raw path, to exercise `write_raw` too.
        let bytes: Vec<u8> = [1.0f32, -2.0, 3.0, -4.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        executor.write_raw(x, &bytes).unwrap();

        executor.run().unwrap();
        assert_eq!(executor.read_f32(y).unwrap(), vec![1.0, 0.0, 3.0, 0.0]);
        // `read_raw` returns the kernel's relu output, in its byte form.
        let relu_bytes: Vec<u8> = [1.0f32, 0.0, 3.0, 0.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        assert_eq!(executor.read_raw(y).unwrap(), relu_bytes);

        assert_eq!(
            copies_in.load(Ordering::SeqCst),
            2,
            "both write paths copy in"
        );
        assert_eq!(
            copies_out.load(Ordering::SeqCst),
            2,
            "both read paths copy out"
        );
    }

    /// The raw partial write the streamed bf16 loader needs: bytes land at `element_offset`
    /// elements past the slot's start, the rest of the slot is untouched, and a range that would
    /// run past the slot (or a byte count that is not whole elements) is refused.
    #[test]
    fn write_raw_at_places_a_chunk_and_refuses_out_of_range() {
        use rustrain_abi::Plugin;
        use rustrain_parallel::{Mesh, ParallelConfig};
        use rustrain_plan::{PlanBuilder, SlotKind};

        let mut registry = rustrain_ops::Registry::new();
        // SAFETY: the built-in provider descriptor is leaked by the builder and lives for the
        // process.
        let reference = unsafe { Plugin::from_static(rustrain_kernels::plugin(), "<built-in>") }
            .expect("the built-in provider passes ABI validation");
        registry.add_plugin(reference).expect("registering it");
        let recipe = rustrain_ops::Recipe::from_toml("[kernel]\ndefault = \"reference\"\n")
            .expect("recipe parses");

        let mut b = PlanBuilder::new(
            "raw-partial",
            rustrain_ops::Phase::Forward,
            Mesh::from_config(&ParallelConfig::default()).fingerprint(),
        );
        // A bf16 weight slot, fed by the loader and by no node — the shape a bf16 weight takes.
        let w = b.slot("w", RsDtype::BF16, vec![4], SlotKind::Weight);
        // The plan needs one node to build; it computes nothing the test reads.
        let x = b.slot("x", RsDtype::F32, vec![4], SlotKind::Input);
        let y = b.slot("y", RsDtype::F32, vec![4], SlotKind::Output);
        b.node(
            rustrain_plan::OpRef::new("elementwise_unary"),
            vec![x],
            vec![y],
            rustrain_plan::Attrs::new().set("kind", "relu"),
            "relu",
        );
        let plan = b.build().unwrap();
        let compiled =
            rustrain_plan::Compiler::new(&registry, &recipe, rustrain_ops::TargetEnv::default())
                .compile(&plan)
                .unwrap();
        let mut executor = Executor::new(
            compiled,
            Box::new(HostAllocator::new()),
            Box::new(SingleRank::new(1)),
        )
        .unwrap();

        // bf16 as 2 little-endian bytes; the values are small integers, exact in bf16.
        let bf16 = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        };
        let seed = bf16(&[1.0, 2.0, 3.0, 4.0]);
        executor.write_raw(w, &seed).unwrap();
        // A chunk that overwrites the middle two elements, and only them.
        let chunk = bf16(&[-5.0, -6.0]);
        executor.write_raw_at(w, 1, &chunk).unwrap();
        let mut expected = seed.clone();
        expected[2..6].copy_from_slice(&chunk);
        assert_eq!(
            executor.read_raw(w).unwrap(),
            expected,
            "the chunk lands at its element offset and the rest of the slot is untouched"
        );

        // Out of range: two elements at offset 3 exceed the 4-element slot.
        assert!(matches!(
            executor.write_raw_at(w, 3, &chunk),
            Err(RuntimeError::LengthMismatch { .. })
        ));
        // A byte count that is not whole elements is refused rather than silently truncated.
        assert!(matches!(
            executor.write_raw_at(w, 0, &[0u8; 3]),
            Err(RuntimeError::LengthMismatch { .. })
        ));
        // An offset that cannot even be added to is refused rather than wrapping.
        assert!(matches!(
            executor.write_raw_at(w, usize::MAX, &[0u8; 2]),
            Err(RuntimeError::LengthMismatch { .. })
        ));
    }
}
