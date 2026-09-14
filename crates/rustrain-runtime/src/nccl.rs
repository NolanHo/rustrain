//! NCCL collectives, loaded at runtime.
//!
//! The third [`CollectiveBackend`](crate::CollectiveBackend), and the only one
//! that talks to other **processes**: one process per rank, one GPU per
//! process, NCCL doing the exchange. It is loaded with `libloading` exactly
//! like the CUDA driver in [`crate::device`], so the core crates keep no CUDA
//! or NCCL dependency at link time (invariant I-1) and a machine with neither
//! still builds and tests.
//!
//! # Why processes and not threads
//!
//! [`ThreadBackend`](crate::ThreadBackend) shows the machinery with a
//! rendezvous that is trivially correct, and it is still the right backend for
//! CPU runs. It cannot carry a GPU run: a CUDA context is current on one
//! thread, the executor's allocator and the plugins' kernels both use that one
//! context, and two ranks in one process would fight over it. One rank per
//! process is what the hardware requires — `rustrain run --rank i --world n`
//! is that process and `rustrain launch` starts the world.
//!
//! # Rendezvous without a network service
//!
//! `ncclCommInitRank` needs every member to hold the same 128-byte
//! `ncclUniqueId`. The group's lowest-ranked member writes it to a file the
//! launcher prepared, the others read it, and **the wait is bounded**: a rank
//! that never sees the file fails with the path and the timeout instead of
//! hanging a world. A file is enough for one host (the scope here); a
//! multi-node launch would need a real store, which is a deliberate
//! non-goal until the single-host path is measured.
//!
//! # Lockstep is still the correctness argument
//!
//! Every rank executes the same compiled plan (tp/cp/ep/dp change shapes and
//! groups, never the node set; `pp > 1` is refused by the runner), so every
//! rank visits the same communicators and the same collective calls in the
//! same order. Communicators are created lazily, on the first collective that
//! needs one — but all members create it at the same point in that order, and
//! `ncclCommInitRank` is itself the handshake that proves it.
//!
//! # What the operands may be
//!
//! Two paths, and the backend says which one it took:
//!
//! * **direct** — both descriptors are row-major for their logical shape, so
//!   NCCL reads and writes the slot buffers themselves;
//! * **staged** — a strided operand is materialised on the host, uploaded to a
//!   device scratch buffer, exchanged, downloaded and scattered back. Correct
//!   for any layout, and counted (`staged_calls`) rather than hidden: a run
//!   where every collective stages is a run whose layouts are worth fixing.
//!
//! Shape-changing collectives (`all_gather`, `reduce_scatter`, `all_to_all`)
//! are chunked one level above NCCL: for a split along `dim` the exchange runs
//! per *outer block* (`product(shape[..dim])` of them), because the runtime's
//! contract places member `j`'s slab at `[j*along, (j+1)*along)` along `dim`
//! *inside every outer block* — a single contiguous `ncclAllGather` would only
//! be right for the outermost dim.

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rustrain_abi::ffi::{RsDeviceKind, RsDtype, RsTensor};
use rustrain_parallel::{GroupMask, Mesh, ReduceOp};

use crate::Allocator;
use crate::collective::{
    CollectiveBackend, CollectiveKind, CollectiveReport, CollectiveRequest, copy_tensor,
    element_width, is_row_major, logical_shape, materialise, resolve_dim, scatter, shape_error,
    span_of,
};
use crate::device::CudaContext;

// ── the NCCL ABI ────────────────────────────────────────────────────────────

type NcclResult = i32;
type NcclComm = *mut c_void;
/// `cudaStream_t`; null is the legacy default stream, which is where torch's
/// kernels run too, so NCCL and the operators stay ordered without the
/// framework owning stream handles.
type CudaStream = *mut c_void;

const NCCL_SUCCESS: NcclResult = 0;
const NCCL_UNIQUE_ID_BYTES: usize = 128;

/// `ncclDataType_t`, as `nccl.h` numbers it.
fn nccl_dtype(dtype: RsDtype) -> Result<i32, String> {
    Ok(match dtype.raw() {
        0 => 7, // ncclFloat32
        1 => 6, // ncclFloat16
        2 => 9, // ncclBfloat16
        6 => 2, // ncclInt32
        7 => 4, // ncclInt64
        8 => 1, // ncclUint8
        _ => {
            return Err(format!(
                "NCCL has no mapping for `{dtype}` in this backend; run the plan at f32/f16/bf16"
            ));
        }
    })
}

/// `ncclRedOp_t`, as `nccl.h` numbers it. The plan's vocabulary stops at the
/// three ops NCCL defines first; anything else is refused rather than guessed.
fn nccl_reduce(op: ReduceOp) -> Result<i32, String> {
    Ok(match op {
        ReduceOp::Sum => 0,
        ReduceOp::Max => 2,
        ReduceOp::Min => 3,
    })
}

/// `ncclUniqueId`: 128 opaque bytes.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct NcclUniqueId {
    pub data: [u8; NCCL_UNIQUE_ID_BYTES],
}

impl NcclUniqueId {
    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != NCCL_UNIQUE_ID_BYTES {
            return Err(format!(
                "a rendezvous file holds {} byte(s); an ncclUniqueId is {NCCL_UNIQUE_ID_BYTES}",
                bytes.len()
            ));
        }
        let mut data = [0u8; NCCL_UNIQUE_ID_BYTES];
        data.copy_from_slice(bytes);
        Ok(Self { data })
    }
}

impl std::fmt::Debug for NcclUniqueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The id is opaque; printing it invites comparisons that mean nothing.
        f.write_str("NcclUniqueId(128 bytes)")
    }
}

/// The NCCL entry points this backend uses, resolved once from a runtime-loaded
/// library.
struct NcclLib {
    /// Kept mapped for the lifetime of the backend so the pointers stay valid.
    _library: libloading::Library,
    path: String,
    get_unique_id: unsafe extern "C" fn(*mut NcclUniqueId) -> NcclResult,
    comm_init_rank: unsafe extern "C" fn(*mut NcclComm, i32, NcclUniqueId, i32) -> NcclResult,
    comm_destroy: unsafe extern "C" fn(NcclComm) -> NcclResult,
    all_reduce: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        i32,
        i32,
        NcclComm,
        CudaStream,
    ) -> NcclResult,
    all_gather: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        i32,
        NcclComm,
        CudaStream,
    ) -> NcclResult,
    reduce_scatter: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        i32,
        i32,
        NcclComm,
        CudaStream,
    ) -> NcclResult,
    broadcast: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        i32,
        i32,
        NcclComm,
        CudaStream,
    ) -> NcclResult,
    send: unsafe extern "C" fn(*const c_void, usize, i32, i32, NcclComm, CudaStream) -> NcclResult,
    recv: unsafe extern "C" fn(*mut c_void, usize, i32, i32, NcclComm, CudaStream) -> NcclResult,
    group_start: unsafe extern "C" fn() -> NcclResult,
    group_end: unsafe extern "C" fn() -> NcclResult,
    get_error_string: Option<unsafe extern "C" fn(NcclResult) -> *const std::ffi::c_char>,
}

impl NcclLib {
    /// Loads NCCL. `path` names the library explicitly (the host keeps one
    /// inside torch's wheel); without it the loader's own search runs.
    fn load(path: Option<&Path>) -> Result<Self, String> {
        let library = match path {
            Some(path) => unsafe { libloading::Library::new(path) }
                .map_err(|error| format!("cannot load NCCL from {}: {error}", path.display()))?,
            None => unsafe { libloading::Library::new("libnccl.so.2") }.or_else(|first| {
                unsafe { libloading::Library::new("libnccl.so") }.map_err(|second| {
                    format!(
                        "cannot load NCCL: tried libnccl.so.2 ({first}) and libnccl.so ({second}); \
                         pass --nccl-lib <path> to name it"
                    )
                })
            })?,
        };
        let resolved = path
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "libnccl.so.2".to_string());

        // SAFETY: each name is a static, NUL-terminated symbol; the signatures
        // are NCCL's, and the library is held for the backend's lifetime.
        macro_rules! sym {
            ($name:literal) => {
                unsafe {
                    *library
                        .get::<_>(concat!($name, "\0").as_bytes())
                        .map_err(|error| format!("NCCL is missing {}: {error}", $name))?
                }
            };
        }
        let get_error_string = unsafe {
            library
                .get::<unsafe extern "C" fn(NcclResult) -> *const std::ffi::c_char>(
                    b"ncclGetErrorString\0",
                )
                .ok()
                .map(|symbol| *symbol)
        };

        Ok(Self {
            get_unique_id: sym!("ncclGetUniqueId"),
            comm_init_rank: sym!("ncclCommInitRank"),
            comm_destroy: sym!("ncclCommDestroy"),
            all_reduce: sym!("ncclAllReduce"),
            all_gather: sym!("ncclAllGather"),
            reduce_scatter: sym!("ncclReduceScatter"),
            broadcast: sym!("ncclBroadcast"),
            send: sym!("ncclSend"),
            recv: sym!("ncclRecv"),
            group_start: sym!("ncclGroupStart"),
            group_end: sym!("ncclGroupEnd"),
            get_error_string,
            _library: library,
            path: resolved,
        })
    }

    /// Names a failing call: NCCL's own string when the symbol is there, the
    /// raw code otherwise. Never guessed.
    fn describe(&self, code: NcclResult, call: &str) -> String {
        if code == NCCL_SUCCESS {
            return format!("{call} succeeded");
        }
        match self.get_error_string {
            // SAFETY: NCCL returns a static NUL-terminated string for any code.
            Some(f) => {
                let text = unsafe { f(code) };
                if text.is_null() {
                    format!("{call} returned {code}")
                } else {
                    let text = unsafe { std::ffi::CStr::from_ptr(text) };
                    format!("{call} returned {code} ({})", text.to_string_lossy())
                }
            }
            None => format!("{call} returned {code}"),
        }
    }

    fn check(&self, code: NcclResult, call: &str) -> Result<(), String> {
        if code == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(self.describe(code, call))
        }
    }
}

// ── the rendezvous ──────────────────────────────────────────────────────────

/// The handshake that gives every member of a group the same `ncclUniqueId`.
///
/// Deliberately a file and not a socket: one host, one launcher-owned
/// directory, no ports to collide and nothing to leave listening. The wait is
/// bounded — a missing file is an error naming the path, never a hang.
struct Rendezvous {
    dir: PathBuf,
    rank: usize,
    timeout: Duration,
}

impl Rendezvous {
    fn new(dir: &Path, rank: usize, timeout: Duration) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|error| {
            format!(
                "cannot create the rendezvous directory {}: {error}",
                dir.display()
            )
        })?;
        Ok(Self {
            dir: dir.to_path_buf(),
            rank,
            timeout,
        })
    }

    fn path_for(&self, mask: GroupMask) -> PathBuf {
        self.dir.join(format!("comm-{:08x}.id", mask.bits()))
    }

    /// The id for one communicator: the group's lowest-ranked member generates
    /// it, every member (that rank included) reads it back from disk.
    ///
    /// Generation is a closure so the protocol is testable without NCCL.
    fn unique_id(
        &self,
        mask: GroupMask,
        members: &[usize],
        generate: impl FnOnce() -> Result<NcclUniqueId, String>,
    ) -> Result<NcclUniqueId, String> {
        let root = members
            .iter()
            .min()
            .copied()
            .ok_or_else(|| format!("group {} has no members", mask))?;
        let path = self.path_for(mask);

        if self.rank == root {
            let id = generate()?;
            // Write-then-rename: a reader either sees no file or the whole id.
            let staging = path.with_extension(format!("tmp{}", std::process::id()));
            std::fs::write(&staging, id.data).map_err(|error| {
                format!(
                    "cannot write the rendezvous file {}: {error}",
                    staging.display()
                )
            })?;
            std::fs::rename(&staging, &path).map_err(|error| {
                format!(
                    "cannot publish the rendezvous file {}: {error}",
                    path.display()
                )
            })?;
            return Ok(id);
        }

        let deadline = Instant::now() + self.timeout;
        loop {
            match std::fs::read(&path) {
                Ok(bytes) => return NcclUniqueId::from_bytes(&bytes),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(format!(
                        "cannot read the rendezvous file {}: {error}",
                        path.display()
                    ));
                }
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "rank {} waited {:?} for `{}` and did not find it; the group's lowest-ranked \
                     member (rank {root}) publishes it, so either that rank never reached this \
                     collective or it failed first",
                    self.rank,
                    self.timeout,
                    path.display()
                ));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

// ── the backend ─────────────────────────────────────────────────────────────

/// One communicator plus the facts needed to use it.
struct Comm {
    handle: NcclComm,
    /// The group's member ranks in ascending order (the runtime's contract).
    members: Vec<usize>,
    /// This rank's position in `members` — the index NCCL and every offset
    /// computation address members by.
    index: usize,
    /// A 4-byte device buffer for the barrier (`ncclAllReduce` needs something
    /// to reduce; one element per rank is the cheapest collective there is).
    barrier: Option<*mut c_void>,
}

/// NCCL collectives between processes, one rank per process.
pub struct NcclBackend {
    rank: usize,
    mesh: Mesh,
    context: CudaContext,
    lib: NcclLib,
    rendezvous: Rendezvous,
    comms: HashMap<u32, Comm>,
    /// Collectives whose operands were not contiguous and had to travel
    /// through host staging. Reported, not hidden.
    staged_calls: u64,
    direct_calls: u64,
}

/// How long a non-root rank waits for the id file. Generous on purpose: the
/// file appears when the *other process* reaches the same collective, and both
/// processes load a 67 GB checkpoint before that.
const RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(1800);

impl NcclBackend {
    /// Builds the backend for `rank` of `mesh` on CUDA device `device_index`.
    ///
    /// `rendezvous_dir` is a directory the launcher owns and every rank of the
    /// world can write; `library` names the NCCL `.so` explicitly when the
    /// loader cannot find one.
    pub fn new(
        rank: usize,
        mesh: Mesh,
        device_index: usize,
        rendezvous_dir: &Path,
        library: Option<&Path>,
    ) -> Result<Self, String> {
        let world = mesh.world_size();
        if rank >= world {
            return Err(format!(
                "rank {rank} is outside the mesh: world size {world}"
            ));
        }
        let lib = NcclLib::load(library)?;
        // Retaining the primary context is what makes NCCL and torch agree on
        // one device context; `open` also makes it current on this thread.
        let context = CudaContext::open(device_index)?;
        let rendezvous = Rendezvous::new(rendezvous_dir, rank, RENDEZVOUS_TIMEOUT)?;
        Ok(Self {
            rank,
            mesh,
            context,
            lib,
            rendezvous,
            comms: HashMap::new(),
            staged_calls: 0,
            direct_calls: 0,
        })
    }

    /// Which NCCL library this backend loaded (for the run report).
    pub fn library(&self) -> &str {
        &self.lib.path
    }

    /// The communicator for `mask`, created on first use — and already there when `warm` ran.
    ///
    /// Every member creates it, and `ncclCommInitRank` blocks until the whole group arrives — that
    /// blocking handshake *is* the proof the world is in step. Where the handshake happens is the
    /// caller's choice: `warm` runs it in the runner's load window, and a run that never calls
    /// `warm` pays it here, inside the first collective that needs the group. Both are the same
    /// rendezvous; only the clock it lands on differs.
    fn comm(&mut self, mask: GroupMask) -> Result<&mut Comm, String> {
        let key = mask.bits();
        if !self.comms.contains_key(&key) {
            let members = self
                .mesh
                .group_ranks(mask, self.rank)
                .map_err(|error| format!("group {} is not a group of this mesh: {error}", mask))?;
            let index = self
                .mesh
                .group_index(mask, self.rank)
                .map_err(|error| format!("rank {} is not in group {}: {error}", self.rank, mask))?;
            let lib = &self.lib;
            let id = self.rendezvous.unique_id(mask, &members, || {
                let mut id = NcclUniqueId {
                    data: [0u8; NCCL_UNIQUE_ID_BYTES],
                };
                // SAFETY: the pointer is to a live local of exactly the size
                // NCCL writes.
                let code = unsafe { (lib.get_unique_id)(&mut id) };
                lib.check(code, "ncclGetUniqueId")?;
                Ok(id)
            })?;
            self.context.set_current()?;
            let nranks = i32::try_from(members.len())
                .map_err(|_| format!("group {} has too many members", mask))?;
            let index_i32 =
                i32::try_from(index).map_err(|_| format!("rank index {index} is too large"))?;
            let mut handle: NcclComm = std::ptr::null_mut();
            // SAFETY: `handle` is a live local; the id and both integers are
            // passed by value as NCCL's signature requires.
            let code = unsafe { (self.lib.comm_init_rank)(&mut handle, nranks, id, index_i32) };
            self.lib.check(code, "ncclCommInitRank").map_err(|error| {
                format!(
                    "initialising the NCCL communicator for group {} (rank {} of {:?}): {error}",
                    mask, self.rank, members
                )
            })?;
            self.comms.insert(
                key,
                Comm {
                    handle,
                    members,
                    index,
                    barrier: None,
                },
            );
        }
        Ok(self
            .comms
            .get_mut(&key)
            .expect("the communicator was just inserted"))
    }

    /// A copy of the communicator's facts. `exchange` takes `&self` (it only
    /// reads the library and the context), so it cannot hold the `&mut Comm`
    /// that creating a communicator returns; the copy is small (a handle, a
    /// member list and an index).
    fn readonly_comm(&mut self, mask: GroupMask) -> Result<Comm, String> {
        let comm = self.comm(mask)?;
        Ok(Comm {
            handle: comm.handle,
            members: comm.members.clone(),
            index: comm.index,
            barrier: None,
        })
    }

    fn barrier(&mut self, mask: GroupMask, host: &mut dyn Allocator) -> Result<(), String> {
        // An entry point owns making the retained context current on *its* thread: a warmed
        // communicator returns from `comm()` without running the creation path that used to be the
        // only place this happened.
        self.context.set_current()?;
        let comm = self.comm(mask)?;
        let handle = comm.handle;
        if comm.barrier.is_none() {
            let buffer = host.alloc(4, RsDeviceKind::CUDA)?;
            host.copy_in(buffer, 4, &[0u8; 4])?;
            comm.barrier = Some(buffer);
        }
        let buffer = comm.barrier.expect("just allocated");
        let lib = &self.lib;
        // SAFETY: the buffer is a live 4-byte device allocation; count 1 of
        // int32 is one element. In-place all-reduce is what NCCL defines.
        let code = unsafe {
            (lib.all_reduce)(
                buffer,
                buffer,
                1,
                nccl_dtype(RsDtype::I32)?,
                nccl_reduce(ReduceOp::Sum)?,
                handle,
                std::ptr::null_mut(),
            )
        };
        lib.check(code, "ncclAllReduce (barrier)")
    }
}

// SAFETY: the backend is `Send` because nothing in it is shared between
// threads: one process owns one rank, the executor runs that rank's whole plan
// on one thread, and every entry point makes the retained context current on
// the calling thread before it touches NCCL. The `*mut c_void` handles are
// NCCL's and the CUDA driver's; they are never dereferenced here and they are
// only ever passed back to the library that produced them. The executor stores
// backends as `Box<dyn CollectiveBackend + Send>` because the in-process thread
// world moves one per rank, so the bound has to hold for this backend too.
unsafe impl Send for NcclBackend {}

impl Drop for NcclBackend {
    fn drop(&mut self) {
        // Destroy the communicators while the context is still retained by
        // this backend (its own retain, independent of the allocator's).
        for (_, comm) in self.comms.drain() {
            // SAFETY: the handle came from `ncclCommInitRank` and is destroyed
            // exactly once.
            unsafe { (self.lib.comm_destroy)(comm.handle) };
        }
    }
}

/// The exchange geometry of a shape-changing collective: the tensor is
/// `[outer, along, inner]` around the split dim, and member `j`'s slab occupies
/// `[j*along, (j+1)*along)` *inside every outer block*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Segments {
    outer: usize,
    along: usize,
    inner: usize,
}

impl Segments {
    fn of(shape: &[i64], dim: usize) -> Result<Self, String> {
        let total: i64 = shape.iter().product();
        let along = shape[dim];
        let inner: i64 = shape[dim + 1..].iter().product();
        let block = along
            .checked_mul(inner)
            .ok_or_else(|| "the collective's block size overflows".to_string())?;
        if block <= 0 || total % block != 0 {
            return Err(format!(
                "cannot describe {shape:?} around dim {dim}: {total} element(s) are not a whole \
                 number of {block}-element blocks"
            ));
        }
        Ok(Self {
            outer: (total / block) as usize,
            along: along as usize,
            inner: inner as usize,
        })
    }
}

/// The byte offsets of one member's slab inside the gathered/scattered tensor,
/// for a given outer block. `along` is the per-member extent along the split
/// dim *in the source*, `inner` the trailing extent.
fn member_offset(
    outer: usize,
    member_index: usize,
    along: usize,
    inner: usize,
    degree: usize,
) -> usize {
    (outer * degree * along + member_index * along) * inner
}

/// The element offset of one outer block in a *local* buffer — one that is
/// already sharded, and therefore has no member axis. Folding a member index
/// into this offset is the one way the collective arithmetic goes silently
/// wrong on every rank but the root: the root's own slab lands where the
/// destination expects it either way.
fn local_offset(outer: usize, along: usize, inner: usize) -> usize {
    member_offset(outer, 0, along, inner, 1)
}

/// The `(send, recv)` element offsets this rank passes to `ncclAllGather` for
/// one outer block. The source is this rank's local buffer; the destination is
/// the gathered tensor, which *does* interleave members — NCCL adds this
/// rank's own slot (`rank * count`) to `recv` inside the call.
fn all_gather_offsets(outer: usize, along: usize, inner: usize, degree: usize) -> (usize, usize) {
    (
        local_offset(outer, along, inner),
        member_offset(outer, 0, along, inner, degree),
    )
}

/// The `(send, recv)` element offsets for `ncclReduceScatter`, the mirror
/// image: the source holds `degree` chunks per outer block and NCCL selects
/// this rank's chunk inside the call, so the destination — the local buffer —
/// is the side without a member axis.
fn reduce_scatter_offsets(
    outer: usize,
    along: usize,
    inner: usize,
    degree: usize,
) -> (usize, usize) {
    (
        member_offset(outer, 0, along, inner, degree),
        local_offset(outer, along, inner),
    )
}

impl NcclBackend {
    /// The contiguous-buffer implementation every collective goes through: the
    /// direct path passes the slots' own pointers, the staged path passes
    /// scratch buffers holding the same bytes.
    #[allow(clippy::too_many_arguments)]
    fn exchange(
        &self,
        kind: CollectiveKind,
        req: &CollectiveRequest,
        in_shape: &[i64],
        out_shape: &[i64],
        in_ptr: *const c_void,
        out_ptr: *mut c_void,
        width: usize,
        comm: &Comm,
        dtype: i32,
    ) -> Result<(), String> {
        let degree = comm.members.len();
        let stream: CudaStream = std::ptr::null_mut();
        let call = |code: NcclResult, what: &str| self.lib.check(code, what);

        match kind {
            CollectiveKind::AllReduce => {
                let op = req
                    .reduce
                    .ok_or_else(|| "all_reduce without a reduction op".to_string())?;
                let count = in_shape.iter().product::<i64>().max(0) as usize;
                // SAFETY: both pointers are live buffers of `count` elements,
                // the same buffer when the executor aliased them.
                let code = unsafe {
                    (self.lib.all_reduce)(
                        in_ptr,
                        out_ptr,
                        count,
                        dtype,
                        nccl_reduce(op)?,
                        comm.handle,
                        stream,
                    )
                };
                call(code, "ncclAllReduce")
            }

            CollectiveKind::Broadcast => {
                let count = in_shape.iter().product::<i64>().max(0) as usize;
                let root = req.src.unwrap_or(0);
                if root >= degree {
                    return Err(format!(
                        "broadcast source index {root} is outside the group of {degree} member(s)"
                    ));
                }
                let root = i32::try_from(root)
                    .map_err(|_| "broadcast source index does not fit an i32".to_string())?;
                // SAFETY: as above; `root` is a member index, which is what
                // NCCL takes.
                let code = unsafe {
                    (self.lib.broadcast)(in_ptr, out_ptr, count, dtype, root, comm.handle, stream)
                };
                call(code, "ncclBroadcast")
            }

            CollectiveKind::AllGather => {
                let dim = resolve_dim(req, in_shape)?;
                let segments = Segments::of(in_shape, dim)?;
                let mut expected = in_shape.to_vec();
                expected[dim] = in_shape[dim]
                    .checked_mul(degree as i64)
                    .ok_or_else(|| "the gathered extent overflows".to_string())?;
                if expected != out_shape {
                    return Err(shape_error(req, &expected, out_shape));
                }
                call(unsafe { (self.lib.group_start)() }, "ncclGroupStart")?;
                for outer in 0..segments.outer {
                    let (send, recv) =
                        all_gather_offsets(outer, segments.along, segments.inner, degree);
                    let count = segments.along * segments.inner;
                    // SAFETY: the block offsets stay inside both buffers, whose
                    // extents the shape check above established.
                    let code = unsafe {
                        (self.lib.all_gather)(
                            (in_ptr as *const u8).add(send * width) as *const c_void,
                            (out_ptr as *mut u8).add(recv * width) as *mut c_void,
                            count,
                            dtype,
                            comm.handle,
                            stream,
                        )
                    };
                    self.lib.check(code, "ncclAllGather")?;
                }
                call(unsafe { (self.lib.group_end)() }, "ncclGroupEnd")
            }

            CollectiveKind::ReduceScatter => {
                let dim = resolve_dim(req, in_shape)?;
                let segments = Segments::of(out_shape, dim)?;
                let mut expected = out_shape.to_vec();
                expected[dim] = out_shape[dim]
                    .checked_mul(degree as i64)
                    .ok_or_else(|| "the reduced extent overflows".to_string())?;
                if expected != in_shape {
                    return Err(shape_error(req, &expected, in_shape));
                }
                call(unsafe { (self.lib.group_start)() }, "ncclGroupStart")?;
                for outer in 0..segments.outer {
                    // The source holds `degree` chunks per outer block and NCCL
                    // picks this rank's own chunk inside the call; the
                    // destination is this rank's local buffer.
                    let (send, recv) =
                        reduce_scatter_offsets(outer, segments.along, segments.inner, degree);
                    let count = segments.along * segments.inner;
                    // SAFETY: as in all_gather — offsets inside both buffers.
                    let code = unsafe {
                        (self.lib.reduce_scatter)(
                            (in_ptr as *const u8).add(send * width) as *const c_void,
                            (out_ptr as *mut u8).add(recv * width) as *mut c_void,
                            count,
                            dtype,
                            0, // ncclSum — the only reduction reduce_scatter uses here
                            comm.handle,
                            stream,
                        )
                    };
                    self.lib.check(code, "ncclReduceScatter")?;
                }
                call(unsafe { (self.lib.group_end)() }, "ncclGroupEnd")
            }

            CollectiveKind::AllToAll => {
                let dim = resolve_dim(req, in_shape)?;
                let segments = Segments::of(in_shape, dim)?;
                let sizes: Vec<i64> = match &req.split {
                    Some(sizes) => {
                        if sizes.len() != degree {
                            return Err(format!(
                                "all_to_all split has {} entr(y|ies) for a group of {degree}",
                                sizes.len()
                            ));
                        }
                        sizes.clone()
                    }
                    None => {
                        if in_shape[dim] % degree as i64 != 0 {
                            return Err(format!(
                                "all_to_all: {} element(s) along dim {dim} do not split equally \
                                 into {degree} rank(s)",
                                in_shape[dim]
                            ));
                        }
                        vec![in_shape[dim] / degree as i64; degree]
                    }
                };
                if sizes.iter().sum::<i64>() != in_shape[dim] {
                    return Err(format!(
                        "all_to_all split {sizes:?} does not sum to the input's extent {} along dim \
                         {dim}",
                        in_shape[dim]
                    ));
                }
                let receive = sizes[comm.index] as usize;
                let mut expected = in_shape.to_vec();
                expected[dim] = receive as i64 * degree as i64;
                if expected != out_shape {
                    return Err(shape_error(req, &expected, out_shape));
                }
                let mut starts = vec![0usize; degree];
                for i in 1..degree {
                    starts[i] = starts[i - 1] + sizes[i - 1] as usize;
                }

                call(unsafe { (self.lib.group_start)() }, "ncclGroupStart")?;
                for outer in 0..segments.outer {
                    for (peer, size) in sizes.iter().enumerate() {
                        if peer == comm.index {
                            continue; // my own chunk stays where it is
                        }
                        let peer_rank = comm.members[peer];
                        let peer_i32 = i32::try_from(peer_rank)
                            .map_err(|_| format!("rank {peer_rank} does not fit an i32"))?;
                        let send = (outer * segments.along + starts[peer]) * segments.inner;
                        let recv = (outer * degree * receive + peer * receive) * segments.inner;
                        // SAFETY: the offsets are inside the buffers the shape
                        // check established; every peer pair is issued inside
                        // one group so the sends and receives match.
                        let code = unsafe {
                            (self.lib.send)(
                                (in_ptr as *const u8).add(send * width) as *const c_void,
                                *size as usize * segments.inner,
                                dtype,
                                peer_i32,
                                comm.handle,
                                stream,
                            )
                        };
                        self.lib.check(code, "ncclSend")?;
                        // SAFETY: as above.
                        let code = unsafe {
                            (self.lib.recv)(
                                (out_ptr as *mut u8).add(recv * width) as *mut c_void,
                                receive * segments.inner,
                                dtype,
                                peer_i32,
                                comm.handle,
                                stream,
                            )
                        };
                        self.lib.check(code, "ncclRecv")?;
                    }
                    // My own chunk: a local copy, so the group stays symmetric.
                    let my = comm.index;
                    let send = (outer * segments.along + starts[my]) * segments.inner;
                    let recv = (outer * degree * receive + my * receive) * segments.inner;
                    if !std::ptr::eq(in_ptr, out_ptr) {
                        // SAFETY: both ranges are inside their buffers and do
                        // not overlap for the same reason NCCL's peers do not.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                (in_ptr as *const u8).add(send * width),
                                (out_ptr as *mut u8).add(recv * width),
                                receive * segments.inner * width,
                            );
                        }
                    }
                }
                call(unsafe { (self.lib.group_end)() }, "ncclGroupEnd")
            }

            CollectiveKind::Sync => unreachable!("handled by the barrier path"),
        }
    }
}

impl CollectiveBackend for NcclBackend {
    /// How many collectives took the staging path, and how many ran directly on
    /// the slots' buffers.
    fn path_counts(&self) -> (u64, u64) {
        (self.staged_calls, self.direct_calls)
    }

    /// Creates each group's communicator now, while the caller has something else to do.
    ///
    /// `ncclCommInitRank` blocks until every member of the group has arrived, and the id file's
    /// root has to publish it first — measured at 1.1-2.6 s per multi-rank run on the verification
    /// host, all of it inside the forward's own wall clock. Warming moves that wait into the
    /// checkpoint load, which is I/O-bound and has the whole machine idle apart from 16 reader
    /// threads.
    fn warm(&mut self, groups: &[GroupMask]) -> Result<(), String> {
        for mask in groups {
            self.comm(*mask)?;
        }
        Ok(())
    }

    fn execute(
        &mut self,
        req: &CollectiveRequest,
        input: &RsTensor,
        output: &mut RsTensor,
        host: &mut dyn Allocator,
    ) -> Result<CollectiveReport, String> {
        let in_shape = logical_shape(input);
        let out_shape = logical_shape(output);
        let sent = in_shape.iter().product::<i64>().max(0) as u64
            * input.dtype.byte_width().unwrap_or(1) as u64;
        let recv = out_shape.iter().product::<i64>().max(0) as u64
            * output.dtype.byte_width().unwrap_or(1) as u64;
        let report = || CollectiveReport {
            sent_bytes: sent,
            recv_bytes: recv,
        };

        let degree = req
            .group
            .degree(&self.mesh)
            .map_err(|error| format!("group {} is not a group of this mesh: {error}", req.group))?;
        if degree == 1 {
            // A group of one is the identity: no communicator, no NCCL, no
            // rendezvous — the same answer `SingleRank` gives, reached through
            // the same copy path.
            copy_tensor(input, output, host)?;
            return Ok(report());
        }

        let width = element_width(input, output)?;
        let dtype = nccl_dtype(input.dtype)?;
        if nccl_dtype(output.dtype)? != dtype {
            return Err(format!(
                "the collective's operands differ in dtype ({} vs {})",
                input.dtype, output.dtype
            ));
        }

        if req.kind == CollectiveKind::Sync {
            self.barrier(req.group, host)?;
            self.direct_calls += 1;
            return Ok(CollectiveReport {
                sent_bytes: 0,
                recv_bytes: 0,
            });
        }

        let in_rank = (input.rank as usize).min(in_shape.len());
        let out_rank = (output.rank as usize).min(out_shape.len());
        let direct = !input.data.is_null()
            && !output.data.is_null()
            && is_row_major(input, &in_shape[..in_rank])
            && is_row_major(output, &out_shape[..out_rank]);
        // In-place is only what NCCL defines as in-place: all_reduce (and the
        // broadcast, where the root writes what it reads). A shape-changing
        // collective that aliases its output would rewrite its own source
        // before the peers read it, so those take a private copy.
        let aliased = std::ptr::eq(input.data, output.data);
        let needs_private_input = aliased
            && matches!(
                req.kind,
                CollectiveKind::AllGather
                    | CollectiveKind::ReduceScatter
                    | CollectiveKind::AllToAll
            );

        if direct {
            // An aliased shape-changing collective reads its source while it
            // writes its own output: give it a private copy of the input, on
            // the device, so the direct path still moves device bytes only.
            let mut scratch: Option<(*mut c_void, u64)> = None;
            if needs_private_input {
                let bytes = span_of(input, width);
                let ptr = host.alloc(bytes, RsDeviceKind::CUDA)?;
                self.context.copy_device(ptr, input.data, bytes)?;
                scratch = Some((ptr, bytes));
            }
            let in_ptr = scratch.map(|(ptr, _)| ptr).unwrap_or(input.data);
            self.context.set_current()?;
            let comm = self.readonly_comm(req.group)?;
            self.direct_calls += 1;
            let exchanged = self.exchange(
                req.kind,
                req,
                &in_shape,
                &out_shape,
                in_ptr,
                output.data,
                width,
                &comm,
                dtype,
            );
            if let Some((ptr, bytes)) = scratch {
                host.dealloc(ptr, bytes);
            }
            exchanged?;
            return Ok(report());
        }

        // Staged: bring the operand to the host, upload it, exchange on
        // contiguous device scratch, download and scatter back. Correct for any
        // layout — and counted, so a run that stages everything says so.
        self.staged_calls += 1;
        self.context.set_current()?;
        let in_bytes = materialise(input, width, host)?;
        let in_ptr = host.alloc(in_bytes.len() as u64, RsDeviceKind::CUDA)?;
        host.copy_in(in_ptr, in_bytes.len() as u64, &in_bytes)?;
        let out_len = out_shape.iter().product::<i64>().max(0) as usize * width;
        let out_ptr = host.alloc(out_len as u64, RsDeviceKind::CUDA)?;
        let comm = self.readonly_comm(req.group)?;
        self.exchange(
            req.kind, req, &in_shape, &out_shape, in_ptr, out_ptr, width, &comm, dtype,
        )?;
        let produced = host.copy_out(out_ptr, out_len as u64)?;
        scatter(output, &produced, width, host)?;
        host.dealloc(in_ptr, in_bytes.len() as u64);
        host.dealloc(out_ptr, out_len as u64);
        Ok(report())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(bits: u32) -> GroupMask {
        GroupMask::from_bits(bits)
    }

    /// Two processes, one directory: the root publishes, the other reads, and
    /// the bytes are the same ones. The protocol is the only part of the
    /// backend that can be tested without a GPU.
    #[test]
    fn the_rendezvous_delivers_the_roots_id_to_the_other_members() {
        let dir = std::env::temp_dir().join(format!("rustrain-rdzv-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let timeout = Duration::from_secs(5);
        let root = Rendezvous::new(&dir, 0, timeout).unwrap();
        let member = Rendezvous::new(&dir, 3, timeout).unwrap();
        let mut data = [0u8; NCCL_UNIQUE_ID_BYTES];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }
        let expected = NcclUniqueId { data };

        let published = root
            .unique_id(mask(0b1), &[0, 2, 3], || Ok(expected))
            .expect("the root publishes");
        let read_back = member
            .unique_id(mask(0b1), &[0, 2, 3], || {
                panic!("a non-root must never generate an id")
            })
            .expect("a member reads it");
        assert_eq!(published, read_back);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The wait is bounded: a rank whose root never publishes says which file
    /// it waited for instead of hanging the world.
    #[test]
    fn a_missing_publisher_is_an_error_naming_the_file_and_the_root() {
        let dir =
            std::env::temp_dir().join(format!("rustrain-rdzv-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let member = Rendezvous::new(&dir, 2, Duration::from_millis(60)).unwrap();
        let error = member
            .unique_id(mask(0b10), &[1, 2, 3], || {
                panic!("a non-root must never generate an id")
            })
            .expect_err("no publisher");
        assert!(
            error.contains("comm-00000002.id") && error.contains("rank 1"),
            "the error must name the file and the root, got: {error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The split geometry: the trailing dims are the inner extent, the leading
    /// ones the outer blocks — and a single contiguous exchange would only be
    /// right when the split dim comes first.
    #[test]
    fn segments_describe_the_exchange_around_the_split_dim() {
        assert_eq!(
            Segments::of(&[2, 4, 3], 1).unwrap(),
            Segments {
                outer: 2,
                along: 4,
                inner: 3
            }
        );
        assert_eq!(
            Segments::of(&[6, 4], 0).unwrap(),
            Segments {
                outer: 1,
                along: 6,
                inner: 4
            }
        );
        assert_eq!(
            Segments::of(&[5], 0).unwrap(),
            Segments {
                outer: 1,
                along: 5,
                inner: 1
            }
        );
        // Member `j`'s slab sits after `j * along` elements in every outer
        // block.
        assert_eq!(member_offset(0, 1, 4, 3, 2), 12);
        assert_eq!(member_offset(1, 1, 4, 3, 2), 12 + 24);
    }

    /// `ncclAllGather`'s two offsets, emulated over in-memory buffers: every
    /// rank sends its *local* block and NCCL walks the members of the gathered
    /// tensor. Folding a member index into the send offset shifts every
    /// non-root rank's slab by one block and reads past the local buffer on the
    /// last outer block — while the root's own half stays correct, which is the
    /// one thing a "the first half agrees" smoke test does check.
    #[test]
    fn all_gather_offsets_place_every_member_s_slab_where_the_gathered_shape_says() {
        for (local_shape, dim, degree) in [
            (&[4i64, 3][..], 0usize, 2usize),
            (&[8, 6][..], 1, 3),
            (&[2, 3, 4][..], 1, 2),
        ] {
            let segments = Segments::of(local_shape, dim).unwrap();
            let count = segments.along * segments.inner;
            let local_total: usize = local_shape.iter().product::<i64>() as usize;
            // Every rank labels its own elements, so a misplaced slab cannot
            // pass for a correct one.
            let locals: Vec<Vec<i64>> = (0..degree)
                .map(|rank| {
                    (0..local_total)
                        .map(|k| rank as i64 * 1000 + k as i64)
                        .collect()
                })
                .collect();
            let mut gathered = vec![-1i64; local_total * degree];
            for outer in 0..segments.outer {
                let (send, recv) =
                    all_gather_offsets(outer, segments.along, segments.inner, degree);
                for (rank, local) in locals.iter().enumerate() {
                    // NCCL adds the member's own slot to `recv` inside the call.
                    let at = recv + rank * count;
                    gathered[at..at + count].copy_from_slice(&local[send..send + count]);
                }
            }
            let block = degree * count;
            let expected: Vec<i64> = (0..gathered.len())
                .map(|flat| {
                    let outer = flat / block;
                    let rank = (flat % block) / count;
                    rank as i64 * 1000 + (outer * count + flat % count) as i64
                })
                .collect();
            assert_eq!(
                gathered, expected,
                "{local_shape:?} along dim {dim} over {degree} rank(s)"
            );
        }
    }

    /// The mirror image: `ncclReduceScatter` reads the full tensor, picks this
    /// rank's chunk inside the call, and writes it into the rank's *local*
    /// buffer — so the member index belongs in the chunk NCCL selects, never in
    /// the `recv` offset.
    #[test]
    fn reduce_scatter_offsets_keep_each_rank_s_own_chunk_in_its_local_buffer() {
        for (global_shape, dim, degree) in [
            (&[8i64, 3][..], 0usize, 2usize),
            (&[8, 6][..], 1, 3),
            (&[2, 4, 3][..], 1, 2),
        ] {
            let mut local_shape = global_shape.to_vec();
            local_shape[dim] /= degree as i64;
            let segments = Segments::of(&local_shape, dim).unwrap();
            let count = segments.along * segments.inner;
            let source: Vec<i64> = (0..global_shape.iter().product::<i64>()).collect();
            for rank in 0..degree {
                let mut local = vec![-1i64; source.len() / degree];
                for outer in 0..segments.outer {
                    let (send, recv) =
                        reduce_scatter_offsets(outer, segments.along, segments.inner, degree);
                    // NCCL picks this rank's chunk out of the send buffer.
                    let at = send + rank * count;
                    local[recv..recv + count].copy_from_slice(&source[at..at + count]);
                }
                let expected: Vec<i64> = (0..local.len())
                    .map(|flat| {
                        ((flat / count) * degree + rank) as i64 * count as i64
                            + (flat % count) as i64
                    })
                    .collect();
                assert_eq!(
                    local, expected,
                    "{global_shape:?} along dim {dim} over {degree} rank(s), rank {rank}"
                );
            }
        }
    }

    /// The dtype and reduce-op tables are the ABI's, and an unnameable value is
    /// reported rather than mapped to something plausible.
    #[test]
    fn nccl_tables_name_the_abi_values_and_refuse_the_rest() {
        assert_eq!(nccl_dtype(RsDtype::F32).unwrap(), 7);
        assert_eq!(nccl_dtype(RsDtype::BF16).unwrap(), 9);
        assert_eq!(nccl_dtype(RsDtype::I64).unwrap(), 4);
        assert!(nccl_dtype(RsDtype::F8E4M3).is_err());
        assert_eq!(nccl_reduce(ReduceOp::Sum).unwrap(), 0);
        assert_eq!(nccl_reduce(ReduceOp::Max).unwrap(), 2);
        assert_eq!(nccl_reduce(ReduceOp::Min).unwrap(), 3);
    }
}
