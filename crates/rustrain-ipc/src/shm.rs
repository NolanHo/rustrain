use std::cell::Cell;
use std::ffi::CString;
use std::io;
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use crate::command::{EpCommand, EpResult, TENSOR_SPAN_ALIGNMENT, TensorSpan};

/// Each slot has a fixed transport header followed by compact JSON control data.
const SLOT_HEADER: usize = 16;
const SLOT_DATA: usize = 256 * 1024;
const SLOT_SIZE: usize = SLOT_HEADER + SLOT_DATA;
const SLOT_LEN_OFFSET: usize = 0;
const SLOT_EPOCH_OFFSET: usize = 8;

/// Shared memory layout:
///   [0..SLOT_SIZE)           command slot (parent writes, all workers read)
///   [SLOT_SIZE..SLOT_SIZE*(1+world_size))  per-worker result slots
///   [aligned control end..semaphore end) process-shared semaphores
///   [64-byte aligned slab header][64-byte aligned raw tensor slab]
const SLAB_ALIGNMENT: usize = 64;
const SLAB_HEADER_SIZE: usize = 64;
const SLAB_MAGIC: [u8; 8] = *b"RTSLAB01";
const SLAB_VERSION: u32 = 2;
const HEADER_MAGIC_OFFSET: usize = 0;
const HEADER_VERSION_OFFSET: usize = 8;
const HEADER_WORLD_SIZE_OFFSET: usize = 12;
const HEADER_SHM_SIZE_OFFSET: usize = 16;
const HEADER_PAYLOAD_OFFSET_OFFSET: usize = 24;
const HEADER_CAPACITY_OFFSET: usize = 32;
const HEADER_EPOCH_OFFSET: usize = 40;
const HEADER_PAYLOAD_LEN_OFFSET: usize = 48;

pub const DEFAULT_TENSOR_SLAB_BYTES: usize = 32 * 1024 * 1024;
const TENSOR_SLAB_BYTES_ENV: &str = "RUSTRAIN_EP_TENSOR_SLAB_BYTES";

/// Default upper bound for one command across all workers.
pub const DEFAULT_BROADCAST_TIMEOUT: Duration = Duration::from_secs(30 * 60);

fn multi_lora_results_are_consistent(reference: &EpResult, candidate: &EpResult) -> bool {
    match (reference, candidate) {
        (
            EpResult::MultiLoraTrain {
                loss: reference_loss,
                step: reference_step,
                adapter_losses: reference_adapters,
            },
            EpResult::MultiLoraTrain {
                loss: candidate_loss,
                step: candidate_step,
                adapter_losses: candidate_adapters,
            },
        ) => {
            reference_loss.to_bits() == candidate_loss.to_bits()
                && reference_step == candidate_step
                && reference_adapters.len() == candidate_adapters.len()
                && reference_adapters.iter().zip(candidate_adapters).all(
                    |(reference_adapter, candidate_adapter)| {
                        reference_adapter.adapter_id == candidate_adapter.adapter_id
                            && reference_adapter.loss.to_bits() == candidate_adapter.loss.to_bits()
                    },
                )
        }
        (EpResult::MultiLoraTrain { .. }, _) | (_, EpResult::MultiLoraTrain { .. }) => false,
        _ => true,
    }
}

fn cmd_offset() -> usize {
    0
}
fn checked_align_up(value: usize, alignment: usize) -> io::Result<usize> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("alignment {alignment} is not a nonzero power of two"),
        ));
    }
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned & !(alignment - 1))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "shared layout overflow"))
}

#[derive(Debug, Clone, Copy)]
struct ShmLayout {
    world_size: usize,
    sem_start: usize,
    slab_header_offset: usize,
    slab_payload_offset: usize,
    slab_capacity: usize,
    shm_size: usize,
}

impl ShmLayout {
    fn new(world_size: usize, slab_capacity: usize) -> io::Result<Self> {
        if world_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "EP world_size must be positive",
            ));
        }
        if u32::try_from(world_size).is_err() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "EP world_size does not fit the shared layout header",
            ));
        }
        let control_slots = world_size.checked_add(1).ok_or_else(layout_overflow)?;
        let control_end = SLOT_SIZE
            .checked_mul(control_slots)
            .ok_or_else(layout_overflow)?;
        let sem_alignment = std::mem::align_of::<libc::sem_t>();
        let sem_start = checked_align_up(control_end, sem_alignment)?;
        let sem_bytes = world_size
            .checked_mul(2)
            .and_then(|count| count.checked_mul(std::mem::size_of::<libc::sem_t>()))
            .ok_or_else(layout_overflow)?;
        let sem_end = sem_start
            .checked_add(sem_bytes)
            .ok_or_else(layout_overflow)?;
        let slab_header_offset = checked_align_up(sem_end, SLAB_ALIGNMENT)?;
        let slab_payload_offset = slab_header_offset
            .checked_add(SLAB_HEADER_SIZE)
            .ok_or_else(layout_overflow)?;
        let shm_size = slab_payload_offset
            .checked_add(slab_capacity)
            .ok_or_else(layout_overflow)?;
        Ok(Self {
            world_size,
            sem_start,
            slab_header_offset,
            slab_payload_offset,
            slab_capacity,
            shm_size,
        })
    }

    fn result_offset(&self, rank: usize) -> io::Result<usize> {
        if rank >= self.world_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "worker rank {rank} is outside world_size {}",
                    self.world_size
                ),
            ));
        }
        SLOT_SIZE
            .checked_mul(rank.checked_add(1).ok_or_else(layout_overflow)?)
            .ok_or_else(layout_overflow)
    }

    fn semaphore_offsets(&self, rank: usize) -> io::Result<(usize, usize)> {
        if rank >= self.world_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "worker rank {rank} is outside world_size {}",
                    self.world_size
                ),
            ));
        }
        let sem_size = std::mem::size_of::<libc::sem_t>();
        let request = rank
            .checked_mul(2)
            .and_then(|index| index.checked_mul(sem_size))
            .and_then(|offset| self.sem_start.checked_add(offset))
            .ok_or_else(layout_overflow)?;
        let done = request.checked_add(sem_size).ok_or_else(layout_overflow)?;
        Ok((request, done))
    }
}

fn layout_overflow() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "shared memory layout overflow")
}

fn configured_slab_capacity() -> io::Result<usize> {
    match std::env::var(TENSOR_SLAB_BYTES_ENV) {
        Ok(value) => value.parse::<usize>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{TENSOR_SLAB_BYTES_ENV} must be a non-negative integer"),
            )
        }),
        Err(std::env::VarError::NotPresent) => Ok(DEFAULT_TENSOR_SLAB_BYTES),
        Err(error) => Err(io::Error::new(io::ErrorKind::InvalidInput, error)),
    }
}

/// Parent-side coordinator: signals all workers, waits for completion.
pub struct EpChannel {
    shm_ptr: *mut u8,
    shm_size: usize,
    shm_fd: i32,
    sem_request: Vec<*mut libc::sem_t>,
    sem_done: Vec<*mut libc::sem_t>,
    world_size: usize,
    layout: ShmLayout,
    shm_name: String,
    default_timeout: Duration,
    poisoned: AtomicBool,
    next_epoch: AtomicU64,
    broadcast_lock: Mutex<()>,
}

/// Worker-side endpoint: waits for commands, signals completion.
pub struct EpWorker {
    shm_ptr: *mut u8,
    shm_size: usize,
    sem_request: *mut libc::sem_t,
    sem_done: *mut libc::sem_t,
    rank: usize,
    layout: ShmLayout,
    current_epoch: Cell<u64>,
    current_payload_len: Cell<usize>,
}

unsafe impl Send for EpChannel {}
unsafe impl Sync for EpChannel {}
unsafe impl Send for EpWorker {}

impl EpChannel {
    pub fn new(world_size: usize) -> io::Result<Self> {
        Self::new_with_timeout(world_size, DEFAULT_BROADCAST_TIMEOUT)
    }

    pub fn new_with_timeout(world_size: usize, default_timeout: Duration) -> io::Result<Self> {
        Self::new_with_timeout_and_slab(world_size, default_timeout, configured_slab_capacity()?)
    }

    pub fn new_with_timeout_and_slab(
        world_size: usize,
        default_timeout: Duration,
        slab_capacity: usize,
    ) -> io::Result<Self> {
        let layout = ShmLayout::new(world_size, slab_capacity)?;
        let shm_size_i64 = i64::try_from(layout.shm_size).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "shared memory size does not fit off_t",
            )
        })?;
        let shm_name = format!("/rustrain-ep-{}", std::process::id());
        let c_name = CString::new(shm_name.as_str()).unwrap();

        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        if unsafe { libc::ftruncate(fd, shm_size_i64) } < 0 {
            let e = io::Error::last_os_error();
            unsafe {
                libc::close(fd);
                libc::shm_unlink(c_name.as_ptr())
            };
            return Err(e);
        }
        let allocation_error = unsafe { libc::posix_fallocate(fd, 0, shm_size_i64) };
        if allocation_error != 0
            && allocation_error != libc::ENOSYS
            && allocation_error != libc::EOPNOTSUPP
        {
            let error = io::Error::from_raw_os_error(allocation_error);
            unsafe {
                libc::close(fd);
                libc::shm_unlink(c_name.as_ptr());
            }
            return Err(io::Error::new(
                error.kind(),
                format!(
                    "reserve {} bytes for EP shared memory: {error}",
                    layout.shm_size
                ),
            ));
        }
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                layout.shm_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let e = io::Error::last_os_error();
            unsafe {
                libc::close(fd);
                libc::shm_unlink(c_name.as_ptr())
            };
            return Err(e);
        }

        // The tensor slab is intentionally left untouched. Clearing it would fault in and
        // commit the full configured capacity before the first request arrives.
        unsafe { ptr::write_bytes(ptr as *mut u8, 0, layout.slab_payload_offset) };
        unsafe { write_static_header(ptr as *mut u8, &layout) };

        let mut sems_request = Vec::with_capacity(world_size);
        let mut sems_done = Vec::with_capacity(world_size);

        for i in 0..world_size {
            let (off_req, off_done) = layout.semaphore_offsets(i)?;
            if unsafe { libc::sem_init((ptr as *mut u8).add(off_req) as *mut libc::sem_t, 1, 0) }
                != 0
            {
                return Err(io::Error::last_os_error());
            }
            if unsafe { libc::sem_init((ptr as *mut u8).add(off_done) as *mut libc::sem_t, 1, 0) }
                != 0
            {
                return Err(io::Error::last_os_error());
            }
            sems_request.push(unsafe { (ptr as *mut u8).add(off_req) as *mut libc::sem_t });
            sems_done.push(unsafe { (ptr as *mut u8).add(off_done) as *mut libc::sem_t });
        }

        tracing::info!(
            "EP IPC: created shm '{}' ({}KB), {} workers",
            shm_name,
            layout.shm_size / 1024,
            world_size
        );

        Ok(Self {
            shm_ptr: ptr as *mut u8,
            shm_size: layout.shm_size,
            shm_fd: fd,
            sem_request: sems_request,
            sem_done: sems_done,
            world_size,
            layout,
            shm_name,
            default_timeout,
            poisoned: AtomicBool::new(false),
            next_epoch: AtomicU64::new(0),
            broadcast_lock: Mutex::new(()),
        })
    }

    pub fn shm_name(&self) -> &str {
        &self.shm_name
    }
    pub fn world_size(&self) -> usize {
        self.world_size
    }
    pub fn slab_capacity(&self) -> usize {
        self.layout.slab_capacity
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Send a command to all workers and propagate any rank's error.
    pub fn broadcast(&self, cmd: &EpCommand) -> io::Result<EpResult> {
        self.broadcast_timeout(cmd, self.default_timeout)
    }

    /// Send a command with an explicit deadline suitable for bounded operations and tests.
    pub fn broadcast_timeout(&self, cmd: &EpCommand, timeout: Duration) -> io::Result<EpResult> {
        self.broadcast_with_slab_timeout(cmd, &[], timeout)
    }

    pub fn broadcast_with_slab(&self, cmd: &EpCommand, payload: &[u8]) -> io::Result<EpResult> {
        self.broadcast_with_slab_timeout(cmd, payload, self.default_timeout)
    }

    pub fn broadcast_with_slab_timeout(
        &self,
        cmd: &EpCommand,
        payload: &[u8],
        timeout: Duration,
    ) -> io::Result<EpResult> {
        self.ensure_healthy()?;
        let _guard = self
            .broadcast_lock
            .lock()
            .map_err(|error| self.poison(format!("EP broadcast lock poisoned: {error}")))?;
        self.ensure_healthy()?;

        let json = serde_json::to_vec(cmd).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("command serialization failed: {error}"),
            )
        })?;
        if json.len() > SLOT_DATA {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "command too large: {} bytes exceeds {SLOT_DATA}",
                    json.len()
                ),
            ));
        }
        self.validate_slab_command(cmd, payload)?;
        let epoch = self
            .next_epoch
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map(|previous| previous + 1)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "EP IPC epoch exhausted"))?;

        // The semaphore post publishes the immutable slab and command to every worker.
        unsafe {
            ptr::copy_nonoverlapping(
                payload.as_ptr(),
                self.shm_ptr.add(self.layout.slab_payload_offset),
                payload.len(),
            );
            write_u64(
                self.shm_ptr,
                self.layout.slab_header_offset + HEADER_EPOCH_OFFSET,
                epoch,
            );
            write_u64(
                self.shm_ptr,
                self.layout.slab_header_offset + HEADER_PAYLOAD_LEN_OFFSET,
                payload.len() as u64,
            );
            write_u32(
                self.shm_ptr,
                cmd_offset() + SLOT_LEN_OFFSET,
                json.len() as u32,
            );
            write_u64(self.shm_ptr, cmd_offset() + SLOT_EPOCH_OFFSET, epoch);
            ptr::copy_nonoverlapping(
                json.as_ptr(),
                self.shm_ptr.add(cmd_offset() + SLOT_HEADER),
                json.len(),
            );
        }

        // Signal all workers simultaneously
        for i in 0..self.world_size {
            if unsafe { libc::sem_post(self.sem_request[i]) } != 0 {
                let error = io::Error::last_os_error();
                return Err(self.poison(format!("failed to signal worker rank {i}: {error}")));
            }
        }

        // One absolute deadline bounds the complete broadcast, not each worker in turn.
        let deadline = realtime_deadline(timeout).map_err(|error| {
            self.poison(format!("failed to create EP broadcast deadline: {error}"))
        })?;
        for i in 0..self.world_size {
            if let Err(error) = sem_timedwait(self.sem_done[i], &deadline) {
                return Err(self.poison(format!(
                    "worker rank {i} did not complete before the EP broadcast deadline: {error}"
                )));
            }
        }

        let mut rank_zero = None;
        let mut worker_error = None;
        for rank in 0..self.world_size {
            let result = self
                .read_result(rank, epoch)
                .map_err(|error| self.poison(error.to_string()))?;
            if let EpResult::Error(error) = &result {
                if worker_error.is_none() {
                    worker_error = Some(EpResult::Error(format!("worker rank {rank}: {error}")));
                }
            }
            if rank == 0 {
                rank_zero = Some(result);
            } else if worker_error.is_none()
                && !multi_lora_results_are_consistent(
                    rank_zero
                        .as_ref()
                        .expect("rank zero result must be read before later ranks"),
                    &result,
                )
            {
                worker_error = Some(EpResult::Error(format!(
                    "worker rank {rank} returned a multi-LoRA result inconsistent with rank 0"
                )));
            }
        }
        if let Some(error) = worker_error {
            return Ok(error);
        }
        rank_zero.ok_or_else(|| io::Error::other("EP broadcast has no rank 0 worker"))
    }

    fn validate_slab_command(&self, cmd: &EpCommand, payload: &[u8]) -> io::Result<()> {
        if payload.len() > self.layout.slab_capacity {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "tensor slab payload {} bytes exceeds capacity {}",
                    payload.len(),
                    self.layout.slab_capacity
                ),
            ));
        }
        match cmd.tensor_slab() {
            Some(tensors) => tensors
                .validate(payload.len())
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error)),
            None if payload.is_empty() => Ok(()),
            None => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "raw tensor slab payload requires a slab command variant",
            )),
        }
    }

    fn ensure_healthy(&self) -> io::Result<()> {
        if self.is_poisoned() {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "EP channel is poisoned by an earlier incomplete broadcast",
            ))
        } else {
            Ok(())
        }
    }

    fn poison(&self, message: impl Into<String>) -> io::Error {
        self.poisoned.store(true, Ordering::Release);
        io::Error::new(io::ErrorKind::BrokenPipe, message.into())
    }

    fn read_result(&self, rank: usize, expected_epoch: u64) -> io::Result<EpResult> {
        let offset = self.layout.result_offset(rank)?;
        unsafe {
            let result_len = read_u32(self.shm_ptr, offset + SLOT_LEN_OFFSET) as usize;
            if result_len == 0 || result_len > SLOT_DATA {
                return Err(io::Error::other(format!(
                    "worker rank {rank} returned an invalid result length {result_len}"
                )));
            }
            let result_epoch = read_u64(self.shm_ptr, offset + SLOT_EPOCH_OFFSET);
            if result_epoch != expected_epoch {
                return Err(io::Error::other(format!(
                    "worker rank {rank} returned epoch {result_epoch}, expected {expected_epoch}"
                )));
            }
            let result_bytes =
                std::slice::from_raw_parts(self.shm_ptr.add(offset + SLOT_HEADER), result_len);
            serde_json::from_slice::<EpResult>(result_bytes).map_err(|error| {
                io::Error::other(format!(
                    "worker rank {rank} result deserialization failed: {error}"
                ))
            })
        }
    }
}

unsafe fn write_static_header(base: *mut u8, layout: &ShmLayout) {
    unsafe {
        ptr::copy_nonoverlapping(
            SLAB_MAGIC.as_ptr(),
            base.add(layout.slab_header_offset + HEADER_MAGIC_OFFSET),
            SLAB_MAGIC.len(),
        );
        write_u32(
            base,
            layout.slab_header_offset + HEADER_VERSION_OFFSET,
            SLAB_VERSION,
        );
        write_u32(
            base,
            layout.slab_header_offset + HEADER_WORLD_SIZE_OFFSET,
            layout.world_size as u32,
        );
        write_u64(
            base,
            layout.slab_header_offset + HEADER_SHM_SIZE_OFFSET,
            layout.shm_size as u64,
        );
        write_u64(
            base,
            layout.slab_header_offset + HEADER_PAYLOAD_OFFSET_OFFSET,
            layout.slab_payload_offset as u64,
        );
        write_u64(
            base,
            layout.slab_header_offset + HEADER_CAPACITY_OFFSET,
            layout.slab_capacity as u64,
        );
    }
}

unsafe fn validate_static_header(base: *const u8, layout: &ShmLayout) -> io::Result<()> {
    let magic = unsafe {
        std::slice::from_raw_parts(
            base.add(layout.slab_header_offset + HEADER_MAGIC_OFFSET),
            SLAB_MAGIC.len(),
        )
    };
    if magic != SLAB_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "EP tensor slab magic does not match",
        ));
    }
    let version = unsafe { read_u32(base, layout.slab_header_offset + HEADER_VERSION_OFFSET) };
    if version != SLAB_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("EP tensor slab version {version} does not match {SLAB_VERSION}"),
        ));
    }
    let header_world =
        unsafe { read_u32(base, layout.slab_header_offset + HEADER_WORLD_SIZE_OFFSET) } as usize;
    let header_shm = unsafe { read_u64(base, layout.slab_header_offset + HEADER_SHM_SIZE_OFFSET) };
    let header_payload = unsafe {
        read_u64(
            base,
            layout.slab_header_offset + HEADER_PAYLOAD_OFFSET_OFFSET,
        )
    };
    let header_capacity =
        unsafe { read_u64(base, layout.slab_header_offset + HEADER_CAPACITY_OFFSET) };
    if header_world != layout.world_size
        || header_shm != layout.shm_size as u64
        || header_payload != layout.slab_payload_offset as u64
        || header_capacity != layout.slab_capacity as u64
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "EP tensor slab layout mismatch: world={header_world} shm={header_shm} payload={header_payload} capacity={header_capacity}"
            ),
        ));
    }
    Ok(())
}

unsafe fn write_u32(base: *mut u8, offset: usize, value: u32) {
    unsafe { ptr::write_unaligned(base.add(offset).cast::<u32>(), value.to_le()) };
}

unsafe fn write_u64(base: *mut u8, offset: usize, value: u64) {
    unsafe { ptr::write_unaligned(base.add(offset).cast::<u64>(), value.to_le()) };
}

unsafe fn read_u32(base: *const u8, offset: usize) -> u32 {
    u32::from_le(unsafe { ptr::read_unaligned(base.add(offset).cast::<u32>()) })
}

unsafe fn read_u64(base: *const u8, offset: usize) -> u64 {
    u64::from_le(unsafe { ptr::read_unaligned(base.add(offset).cast::<u64>()) })
}

fn realtime_deadline(timeout: Duration) -> io::Result<libc::timespec> {
    let mut deadline = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut deadline) } != 0 {
        return Err(io::Error::last_os_error());
    }

    let timeout_secs = timeout.as_secs().min(libc::time_t::MAX as u64) as libc::time_t;
    deadline.tv_sec = deadline.tv_sec.saturating_add(timeout_secs);
    deadline.tv_nsec += timeout.subsec_nanos() as libc::c_long;
    if deadline.tv_nsec >= 1_000_000_000 {
        deadline.tv_sec = deadline.tv_sec.saturating_add(1);
        deadline.tv_nsec -= 1_000_000_000;
    }
    Ok(deadline)
}

fn sem_timedwait(sem: *mut libc::sem_t, deadline: &libc::timespec) -> io::Result<()> {
    loop {
        if unsafe { libc::sem_timedwait(sem, deadline) } == 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EINTR) {
            return Err(error);
        }
    }
}

impl Drop for EpChannel {
    fn drop(&mut self) {
        for i in 0..self.world_size {
            unsafe {
                libc::sem_destroy(self.sem_request[i]);
                libc::sem_destroy(self.sem_done[i])
            };
        }
        unsafe {
            libc::munmap(self.shm_ptr as *mut libc::c_void, self.shm_size);
            libc::close(self.shm_fd);
            let c_name = CString::new(self.shm_name.as_str()).unwrap();
            libc::shm_unlink(c_name.as_ptr());
        }
    }
}

impl EpWorker {
    pub fn attach(shm_name: &str, rank: usize, world_size: usize) -> io::Result<Self> {
        if rank >= world_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("worker rank {rank} is outside world_size {world_size}"),
            ));
        }
        let base_layout = ShmLayout::new(world_size, 0)?;
        let c_name = CString::new(shm_name).unwrap();
        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        let mut stat = std::mem::MaybeUninit::<libc::stat>::zeroed();
        if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } != 0 {
            let error = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(error);
        }
        let stat = unsafe { stat.assume_init() };
        let shm_size = usize::try_from(stat.st_size).map_err(|_| {
            unsafe { libc::close(fd) };
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("shared memory size {} does not fit usize", stat.st_size),
            )
        })?;
        if shm_size < base_layout.slab_payload_offset {
            unsafe { libc::close(fd) };
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "shared memory size {shm_size} is smaller than layout prefix {}",
                    base_layout.slab_payload_offset
                ),
            ));
        }
        let layout = ShmLayout::new(world_size, shm_size - base_layout.slab_payload_offset)?;

        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                shm_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let e = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(e);
        }
        unsafe { libc::close(fd) };

        if let Err(error) = unsafe { validate_static_header(ptr.cast(), &layout) } {
            unsafe { libc::munmap(ptr, shm_size) };
            return Err(error);
        }
        let (off_req, off_done) = layout.semaphore_offsets(rank)?;
        let sem_request = unsafe { (ptr as *mut u8).add(off_req) as *mut libc::sem_t };
        let sem_done = unsafe { (ptr as *mut u8).add(off_done) as *mut libc::sem_t };

        tracing::info!("EP worker {}: attached to shm '{}'", rank, shm_name);
        Ok(Self {
            shm_ptr: ptr as *mut u8,
            shm_size,
            sem_request,
            sem_done,
            rank,
            layout,
            current_epoch: Cell::new(0),
            current_payload_len: Cell::new(0),
        })
    }

    pub fn wait_command(&self) -> io::Result<EpCommand> {
        loop {
            if unsafe { libc::sem_wait(self.sem_request) } == 0 {
                break;
            }
            let error = io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::EINTR) {
                return Err(error);
            }
        }
        unsafe {
            let cmd_len = read_u32(self.shm_ptr, cmd_offset() + SLOT_LEN_OFFSET) as usize;
            if cmd_len == 0 || cmd_len > SLOT_DATA {
                return Err(io::Error::other(format!(
                    "invalid command length: {}",
                    cmd_len
                )));
            }
            let command_epoch = read_u64(self.shm_ptr, cmd_offset() + SLOT_EPOCH_OFFSET);
            let slab_epoch = read_u64(
                self.shm_ptr,
                self.layout.slab_header_offset + HEADER_EPOCH_OFFSET,
            );
            if command_epoch == 0 || command_epoch != slab_epoch {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("command epoch {command_epoch} does not match slab epoch {slab_epoch}"),
                ));
            }
            if command_epoch <= self.current_epoch.get() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "command epoch {command_epoch} is not newer than worker epoch {}",
                        self.current_epoch.get()
                    ),
                ));
            }
            let payload_len_u64 = read_u64(
                self.shm_ptr,
                self.layout.slab_header_offset + HEADER_PAYLOAD_LEN_OFFSET,
            );
            let payload_len = usize::try_from(payload_len_u64).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "tensor slab payload length does not fit usize",
                )
            })?;
            if payload_len > self.layout.slab_capacity {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "tensor slab payload {payload_len} exceeds mapped capacity {}",
                        self.layout.slab_capacity
                    ),
                ));
            }
            let cmd_bytes =
                std::slice::from_raw_parts(self.shm_ptr.add(cmd_offset() + SLOT_HEADER), cmd_len);
            let command = serde_json::from_slice::<EpCommand>(cmd_bytes)
                .map_err(|e| io::Error::other(format!("command deserialization: {e}")))?;
            match command.tensor_slab() {
                Some(tensors) => tensors
                    .validate(payload_len)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
                None if payload_len == 0 => {}
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "legacy command unexpectedly carries a tensor slab payload",
                    ));
                }
            }
            self.current_epoch.set(command_epoch);
            self.current_payload_len.set(payload_len);
            Ok(command)
        }
    }

    /// Returns a zero-copy view into the current broadcast's raw little-endian i64 slab.
    ///
    /// # Safety
    /// The view must not be retained or accessed after `signal_done`, because the parent may
    /// reuse the shared slab as soon as every worker reports completion.
    pub unsafe fn slab_i64(&self, span: TensorSpan) -> io::Result<&[i64]> {
        if cfg!(target_endian = "big") {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "tensor slab i64 views require a little-endian host",
            ));
        }
        let offset = usize::try_from(span.offset_bytes).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "tensor slab offset does not fit usize",
            )
        })?;
        let len = usize::try_from(span.len_bytes).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "tensor slab length does not fit usize",
            )
        })?;
        if offset % TENSOR_SPAN_ALIGNMENT != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor slab offset {offset} is not {TENSOR_SPAN_ALIGNMENT}-byte aligned"),
            ));
        }
        if len % std::mem::size_of::<i64>() != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tensor slab length {len} is not a multiple of 8 bytes"),
            ));
        }
        let end = offset.checked_add(len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "tensor slab span overflow")
        })?;
        if end > self.current_payload_len.get() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor slab span {offset}..{end} exceeds current payload {}",
                    self.current_payload_len.get()
                ),
            ));
        }
        let pointer = unsafe { self.shm_ptr.add(self.layout.slab_payload_offset + offset) };
        if pointer.align_offset(std::mem::align_of::<i64>()) != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "mapped tensor slab pointer is not i64 aligned",
            ));
        }
        Ok(unsafe { std::slice::from_raw_parts(pointer.cast::<i64>(), len / 8) })
    }

    pub fn signal_done(&self, result: &EpResult) -> io::Result<()> {
        let json = encode_result_for_slot(result)?;

        // Write result to THIS worker's dedicated slot (no collision with other workers)
        let off = self.layout.result_offset(self.rank)?;
        unsafe {
            write_u32(self.shm_ptr, off + SLOT_LEN_OFFSET, json.len() as u32);
            write_u64(
                self.shm_ptr,
                off + SLOT_EPOCH_OFFSET,
                self.current_epoch.get(),
            );
            ptr::copy_nonoverlapping(
                json.as_ptr(),
                self.shm_ptr.add(off + SLOT_HEADER),
                json.len(),
            );
        }

        if unsafe { libc::sem_post(self.sem_done) } != 0 {
            return Err(io::Error::last_os_error());
        }
        self.current_payload_len.set(0);
        Ok(())
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
}

fn encode_result_for_slot(result: &EpResult) -> io::Result<Vec<u8>> {
    let json = serde_json::to_vec(result).map_err(|e| io::Error::other(e.to_string()))?;
    if json.len() <= SLOT_DATA {
        return Ok(json);
    }
    serde_json::to_vec(&EpResult::Error(format!(
        "worker result exceeded the {SLOT_DATA}-byte IPC slot"
    )))
    .map_err(|e| io::Error::other(e.to_string()))
}

impl Drop for EpWorker {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.shm_ptr as *mut libc::c_void, self.shm_size) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_lora_result_consistency_checks_rank_values_and_shape() {
        let reference = EpResult::MultiLoraTrain {
            loss: 2.0,
            step: 7,
            adapter_losses: vec![crate::command::AdapterLoss {
                adapter_id: 41,
                loss: 2.0,
            }],
        };
        assert!(multi_lora_results_are_consistent(&reference, &reference));

        let different_loss = EpResult::MultiLoraTrain {
            loss: 2.0,
            step: 7,
            adapter_losses: vec![crate::command::AdapterLoss {
                adapter_id: 41,
                loss: 2.5,
            }],
        };
        assert!(!multi_lora_results_are_consistent(
            &reference,
            &different_loss
        ));
        assert!(!multi_lora_results_are_consistent(
            &reference,
            &EpResult::Train { loss: 2.0, step: 7 }
        ));
    }

    #[test]
    fn oversized_result_is_replaced_by_a_compact_error() {
        let encoded = encode_result_for_slot(&EpResult::Error("x".repeat(SLOT_DATA))).unwrap();
        assert!(encoded.len() <= SLOT_DATA);
        let decoded: EpResult = serde_json::from_slice(&encoded).unwrap();
        match decoded {
            EpResult::Error(message) => assert!(message.contains("exceeded")),
            other => panic!("unexpected oversized-result replacement: {other:?}"),
        }
    }
    use crate::command::TensorSlabRef;

    static TEST_CHANNEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn test_channel(world_size: usize, slab_capacity: usize) -> EpChannel {
        EpChannel::new_with_timeout_and_slab(world_size, DEFAULT_BROADCAST_TIMEOUT, slab_capacity)
            .expect("create test channel")
    }

    fn append_i64(payload: &mut Vec<u8>, values: &[i64]) -> TensorSpan {
        let aligned = payload
            .len()
            .checked_add(crate::command::TENSOR_SPAN_ALIGNMENT - 1)
            .map(|value| value & !(crate::command::TENSOR_SPAN_ALIGNMENT - 1))
            .unwrap();
        payload.resize(aligned, 0);
        let offset_bytes = payload.len() as u64;
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        TensorSpan {
            offset_bytes,
            len_bytes: (values.len() * std::mem::size_of::<i64>()) as u64,
        }
    }

    #[test]
    fn test_channel_create_destroy() {
        let _guard = TEST_CHANNEL_LOCK.lock().unwrap();
        let ch = test_channel(2, 4096);
        assert_eq!(ch.world_size(), 2);
        assert_eq!(ch.slab_capacity(), 4096);
        drop(ch);
    }

    #[test]
    fn checked_layout_aligns_semaphores_and_tensor_slab() {
        for world_size in [1, 2, 8] {
            let layout = ShmLayout::new(world_size, 4096).unwrap();
            assert_eq!(layout.sem_start % std::mem::align_of::<libc::sem_t>(), 0);
            assert_eq!(layout.slab_header_offset % SLAB_ALIGNMENT, 0);
            assert_eq!(layout.slab_payload_offset % SLAB_ALIGNMENT, 0);
            assert_eq!(layout.shm_size - layout.slab_payload_offset, 4096);
        }
        assert_eq!(
            ShmLayout::new(0, 4096).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn broadcast_propagates_nonzero_rank_error() {
        let _guard = TEST_CHANNEL_LOCK.lock().unwrap();
        let channel = test_channel(2, 4096);
        let worker_zero = EpWorker::attach(channel.shm_name(), 0, 2).expect("attach rank 0");
        let worker_one = EpWorker::attach(channel.shm_name(), 1, 2).expect("attach rank 1");
        let rank_zero = std::thread::spawn(move || {
            assert!(matches!(
                worker_zero.wait_command().unwrap(),
                EpCommand::Shutdown
            ));
            worker_zero.signal_done(&EpResult::Ok).unwrap();
        });
        let rank_one = std::thread::spawn(move || {
            assert!(matches!(
                worker_one.wait_command().unwrap(),
                EpCommand::Shutdown
            ));
            worker_one
                .signal_done(&EpResult::Error("rank one failed".into()))
                .unwrap();
        });

        let result = channel.broadcast(&EpCommand::Shutdown).unwrap();
        rank_zero.join().unwrap();
        rank_one.join().unwrap();
        assert!(!channel.is_poisoned());

        match result {
            EpResult::Error(error) => {
                assert_eq!(error, "worker rank 1: rank one failed");
            }
            _ => panic!("rank 1 error was not propagated"),
        }
    }

    #[test]
    fn broadcast_timeout_permanently_poisons_channel() {
        let _guard = TEST_CHANNEL_LOCK.lock().unwrap();
        let channel = test_channel(1, 4096);
        let worker = EpWorker::attach(channel.shm_name(), 0, 1).expect("attach rank 0");
        let worker_thread = std::thread::spawn(move || {
            assert!(matches!(
                worker.wait_command().unwrap(),
                EpCommand::Shutdown
            ));
            // Simulate a worker crash after accepting the command.
        });

        let started = std::time::Instant::now();
        let error = channel
            .broadcast_timeout(&EpCommand::Shutdown, Duration::from_millis(100))
            .expect_err("missing completion must time out");
        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(channel.is_poisoned());
        worker_thread.join().unwrap();

        let retry_started = std::time::Instant::now();
        let retry_error = channel
            .broadcast_timeout(&EpCommand::Shutdown, Duration::from_secs(1))
            .expect_err("poisoned channel must reject future commands");
        assert_eq!(retry_error.kind(), io::ErrorKind::BrokenPipe);
        assert!(retry_started.elapsed() < Duration::from_millis(100));
    }

    #[test]
    fn tensor_slab_round_trip_exposes_zero_copy_i64_views() {
        let _guard = TEST_CHANNEL_LOCK.lock().unwrap();
        let channel = test_channel(1, 256);
        let worker = EpWorker::attach(channel.shm_name(), 0, 1).expect("attach rank 0");
        let mut payload = Vec::new();
        let input_ids = append_i64(&mut payload, &[11, 12, 13, 14]);
        let target_mask = append_i64(&mut payload, &[1, 0, 1, 0]);
        let attention_mask = append_i64(&mut payload, &[1, 1, 1, 1]);
        let tensors = TensorSlabRef {
            input_ids,
            target_mask,
            attention_mask,
            batch_size: 2,
            seq_len: 2,
        };
        let worker_thread = std::thread::spawn(move || {
            let command = worker.wait_command().unwrap();
            let tensors = match command {
                EpCommand::TrainStepSlab { tensors, .. } => tensors,
                other => panic!("unexpected command: {other:?}"),
            };
            unsafe {
                assert_eq!(
                    worker.slab_i64(tensors.input_ids).unwrap(),
                    [11, 12, 13, 14]
                );
                assert_eq!(worker.slab_i64(tensors.target_mask).unwrap(), [1, 0, 1, 0]);
                assert_eq!(
                    worker.slab_i64(tensors.attention_mask).unwrap(),
                    [1, 1, 1, 1]
                );
            }
            worker.signal_done(&EpResult::Ok).unwrap();
        });
        let result = channel
            .broadcast_with_slab(
                &EpCommand::TrainStepSlab {
                    session_id: "session".into(),
                    tensors,
                },
                &payload,
            )
            .unwrap();
        assert!(matches!(result, EpResult::Ok));
        assert!(!channel.is_poisoned());
        worker_thread.join().unwrap();
    }

    #[test]
    fn invalid_or_oversized_slab_is_recoverable_before_publish() {
        let _guard = TEST_CHANNEL_LOCK.lock().unwrap();
        let channel = test_channel(1, 32);
        let oversized = TensorSlabRef {
            input_ids: TensorSpan {
                offset_bytes: 0,
                len_bytes: 16,
            },
            target_mask: TensorSpan {
                offset_bytes: 64,
                len_bytes: 16,
            },
            attention_mask: TensorSpan {
                offset_bytes: 128,
                len_bytes: 16,
            },
            batch_size: 1,
            seq_len: 2,
        };
        let error = channel
            .broadcast_with_slab(
                &EpCommand::TrainStepSlab {
                    session_id: "session".into(),
                    tensors: oversized,
                },
                &[0; 144],
            )
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(!channel.is_poisoned());

        let overlapping = TensorSlabRef {
            input_ids: TensorSpan {
                offset_bytes: 0,
                len_bytes: 8,
            },
            target_mask: TensorSpan {
                offset_bytes: 0,
                len_bytes: 8,
            },
            attention_mask: TensorSpan {
                offset_bytes: 64,
                len_bytes: 8,
            },
            batch_size: 1,
            seq_len: 1,
        };
        let error = channel
            .broadcast_with_slab(
                &EpCommand::TrainStepSlab {
                    session_id: "session".into(),
                    tensors: overlapping,
                },
                &[0; 72],
            )
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(!channel.is_poisoned());

        let worker = EpWorker::attach(channel.shm_name(), 0, 1).expect("attach rank 0");
        let worker_thread = std::thread::spawn(move || {
            assert!(matches!(
                worker.wait_command().unwrap(),
                EpCommand::Shutdown
            ));
            worker.signal_done(&EpResult::Ok).unwrap();
        });
        assert!(matches!(
            channel.broadcast(&EpCommand::Shutdown).unwrap(),
            EpResult::Ok
        ));
        worker_thread.join().unwrap();
    }
}
