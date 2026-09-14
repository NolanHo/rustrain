//! Collective backends: what a spliced collective actually does.
//!
//! The plan compiler decides *where* communication happens; this module decides
//! *how*. Two backends exist, both behind [`CollectiveBackend`] so an NCCL
//! backend can replace them without the executor knowing:
//!
//! * [`SingleRank`] — the identity with a copy, correct precisely when the
//!   world size is 1. It refuses anything else rather than pretending.
//! * [`ThreadBackend`] — the D6 reference transport: N threads in one process,
//!   one per rank, exchanging bytes through shared scratch buffers behind a
//!   two-phase rendezvous.
//!
//! # Why threads and shared buffers
//!
//! D6 measures the *machinery*, not the transport: the reference provider is
//! pure scalar Rust (invariant I-1), so no transport can make eight CPU ranks
//! faster than one. The transport therefore has exactly one job — be obviously
//! correct — and shared-memory threads are the least machinery that does it:
//! no sockets, no wire protocol, no per-collective framing to desynchronise.
//! Every collective is *meet → write my contribution → meet → read the group's
//! contributions*, which is visibly right by construction, while an optimised
//! ring or tree hides its correctness behind index arithmetic that is very easy
//! to get subtly wrong.
//!
//! # Correctness argument
//!
//! **Lockstep.** All ranks execute the same compiled plan, which has the same
//! node set on every rank (tp/cp/ep/dp change shapes and groups, never nodes;
//! `pp > 1` is refused by the runner), so every rank walks the same sequence of
//! collective calls. The rendezvous is keyed by that shared ordinal: it admits
//! a rank only when all of them have reached the same call, and no rank can run
//! ahead of the slowest. The two meets inside one call separate the write phase
//! from the read phase, so no rank reads a contribution that has not been
//! written.
//!
//! **Groups.** Every rank belongs to exactly one instance of every [`GroupMask`]
//! (`Mesh::group_ranks`), so a collective over a subset of the world runs its
//! exchange among that instance's members while the other instances exchange
//! among themselves in the same shared slot table (one slot per *world rank*,
//! so instances never alias).
//!
//! **Failure.** A poisoned rendezvous wakes every waiter with the recorded
//! error instead of deadlocking: if any rank fails — at an operator, which
//! `poison` is called for from the driver, or at a collective — the others
//! finish their current call with that error and the driver joins cleanly.
//!
//! **`all_to_all` split ordering.** The plan's `split` (one positive entry per
//! rank in the group, summing to the input's extent along `dim` — validated at
//! compile time) partitions the *input* of every rank into per-destination
//! chunks: chunk `i` goes to the member with group index `i`. Rank `r` receives
//! `split[r]` elements from every member, concatenated in ascending group-index
//! order, so its output extent along `dim` is `degree × split[r]`. Both facts
//! are re-validated here against the real tensor shapes before any byte moves —
//! a plan whose `split` does not match the local extents is a reported error,
//! never a silent mis-redistribution.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};

use rustrain_abi::ffi::RsTensor;
use rustrain_parallel::{GroupMask, Mesh, ReduceOp};

use crate::Allocator;

/// The collectives a runtime backend must perform. The plugin ABI's
/// `RsCollectiveKind` does not carry `broadcast` or `sync`, so the runtime
/// vocabulary is its own enum; the executor maps the plan's intrinsic names
/// onto it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CollectiveKind {
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
    AllToAll,
    Sync,
}

impl CollectiveKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CollectiveKind::AllReduce => "all_reduce",
            CollectiveKind::AllGather => "all_gather",
            CollectiveKind::ReduceScatter => "reduce_scatter",
            CollectiveKind::Broadcast => "broadcast",
            CollectiveKind::AllToAll => "all_to_all",
            CollectiveKind::Sync => "sync",
        }
    }
}

/// Everything a spliced collective step carries, in one request.
pub struct CollectiveRequest {
    pub kind: CollectiveKind,
    pub group: GroupMask,
    /// `all_reduce` only.
    pub reduce: Option<ReduceOp>,
    /// `all_gather` / `reduce_scatter` / `all_to_all` only.
    pub dim: Option<i64>,
    /// `all_to_all` only: the per-rank send sizes along `dim` (one entry per
    /// rank in the group). `None` = equal split.
    pub split: Option<Vec<i64>>,
    /// `broadcast` only: the source rank's index inside the group.
    pub src: Option<usize>,
}

/// What one collective call moved, seen from this rank.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollectiveReport {
    /// Bytes this rank handed to the exchange.
    pub sent_bytes: u64,
    /// Bytes this rank took out of it.
    pub recv_bytes: u64,
}

/// Performs the collective a spliced node represents.
///
/// `input` is the collective's operand (possibly a strided view); `output` is
/// where the result lands — either the input's storage (in-place, when the
/// executor aliased it) or a dedicated buffer. The backend must read all of
/// `input` before writing `output` (the two may be one buffer) and must honour
/// both descriptors' shapes and strides.
pub trait CollectiveBackend {
    /// `host` is where the operands' bytes actually live: slot buffers may be
    /// host memory or device memory, and only the allocator created for them
    /// knows how to move them. A backend that walked the descriptors' pointers
    /// itself would read device memory from the host and fault (the D6 GPU run
    /// found exactly that), so every read and write goes through it.
    fn execute(
        &mut self,
        req: &CollectiveRequest,
        input: &RsTensor,
        output: &mut RsTensor,
        host: &mut dyn Allocator,
    ) -> Result<CollectiveReport, String>;

    /// Pre-create whatever each group needs, so the group's *formation* — a rendezvous with every
    /// other member, which is what `ncclCommInitRank` is — happens before the run's measured work
    /// instead of inside its first collective.
    ///
    /// The default does nothing, and that is the right default: a backend whose groups cost
    /// nothing to form (the identity, the in-process threads) has nothing to warm. A backend that
    /// implements it must make the work *idempotent* — `execute` finds the communicator already
    /// there — because a run may execute the same group many times.
    fn warm(&mut self, _groups: &[GroupMask]) -> Result<(), String> {
        Ok(())
    }
}

/// A backend behind a mutex, so a second thread can warm it while the main thread does something
/// else — loading a 67 GB checkpoint, typically.
///
/// The lock is uncontended during the run: once the warming thread has been joined, the executor
/// is the only caller. What makes that true is that the *loading* path does not touch the
/// backend: the checkpoint loader writes through the executor's allocator (`Executor::write_raw`),
/// never through a collective.
pub struct SharedBackend {
    inner: Arc<Mutex<Box<dyn CollectiveBackend + Send>>>,
}

impl SharedBackend {
    /// Takes ownership of `backend` and returns the shared handle.
    pub fn new(backend: Box<dyn CollectiveBackend + Send>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(backend)),
        }
    }

    /// A second handle, for the thread that warms the groups while this one loads.
    pub fn handle(&self) -> Arc<Mutex<Box<dyn CollectiveBackend + Send>>> {
        Arc::clone(&self.inner)
    }

    /// The message a poisoned lock gets. A panic inside a collective is not something to retry:
    /// the backend's state (a half-written communicator, a rendezvous ordinal) is unknown.
    fn poisoned() -> String {
        "the collective backend's lock was poisoned by a panic in another thread; the backend's \
        state is unknown, so the run is refused rather than continued"
            .to_string()
    }
}

impl CollectiveBackend for SharedBackend {
    fn execute(
        &mut self,
        req: &CollectiveRequest,
        input: &RsTensor,
        output: &mut RsTensor,
        host: &mut dyn Allocator,
    ) -> Result<CollectiveReport, String> {
        self.inner
            .lock()
            .map_err(|_| Self::poisoned())?
            .execute(req, input, output, host)
    }

    fn warm(&mut self, groups: &[GroupMask]) -> Result<(), String> {
        self.inner
            .lock()
            .map_err(|_| Self::poisoned())?
            .warm(groups)
    }
}

/// The identity backend.
///
/// Correct precisely when nothing is actually distributed, and the only thing
/// that can back a sharded plan inside one process. It refuses when the world
/// size is larger than one rather than silently pretending: a plan whose
/// sharding requires a real all-reduce cannot be executed correctly by doing
/// nothing.
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
        req: &CollectiveRequest,
        input: &RsTensor,
        output: &mut RsTensor,
        host: &mut dyn Allocator,
    ) -> Result<CollectiveReport, String> {
        if self.world_size > 1 {
            return Err(format!(
                "collective {} on group {} was requested with world_size={}, but no \
                 distributed backend is installed",
                req.kind.as_str(),
                req.group,
                self.world_size
            ));
        }
        // World size 1: every collective is the identity, and a degree-1 group
        // makes the declared shapes agree by construction (the gather is over
        // one slice, the split has one entry — the whole tensor, validated at
        // compile time).
        let bytes = copy_tensor(input, output, host)?;
        Ok(CollectiveReport {
            sent_bytes: 0,
            recv_bytes: bytes,
        })
    }
}

// ── the shared-memory multi-rank backend ─────────────────────────────────────

/// State shared by every rank of one [`ThreadBackend`] world.
pub struct ThreadShared {
    world: usize,
    state: Mutex<RendezvousState>,
    cv: Condvar,
    /// Collective ordinal → one contiguous byte slot per world rank. Keyed by
    /// ordinal because the ranks visit collectives in lockstep: the same key is
    /// current on every rank at the same time, and consecutive calls never
    /// alias.
    scratch: Mutex<HashMap<u64, Vec<Vec<u8>>>>,
}

#[derive(Default)]
struct RendezvousState {
    /// The ordinal the rendezvous currently admits.
    generation: u64,
    /// Arrivals at the current `gen`.
    arrived: usize,
    /// Set when any rank fails; every waiter is then woken with this error.
    failed: Option<String>,
}

impl ThreadShared {
    pub fn new(world: usize) -> Arc<Self> {
        Arc::new(Self {
            world,
            state: Mutex::new(RendezvousState::default()),
            cv: Condvar::new(),
            scratch: Mutex::new(HashMap::new()),
        })
    }

    /// Records a failure and wakes every blocked rank. Idempotent.
    pub fn poison(&self, message: &str) {
        let mut state = self.state.lock().expect("rendezvous state poisoned");
        state.failed.get_or_insert_with(|| message.to_string());
        self.cv.notify_all();
    }

    /// Blocks until every rank has arrived at `generation`, or the world has failed.
    fn meet(&self, generation: u64) -> Result<(), String> {
        let mut state = self.state.lock().expect("rendezvous state poisoned");
        loop {
            if let Some(error) = &state.failed {
                return Err(error.clone());
            }
            if state.generation == generation {
                break;
            }
            // Cannot happen while the walk is in lockstep (a rank only admits
            // the generation it has not reached yet), but do not block on a
            // generation that already passed either.
            if state.generation > generation {
                return Ok(());
            }
            state = self.cv.wait(state).expect("rendezvous state poisoned");
        }
        state.arrived += 1;
        if state.arrived == self.world {
            state.generation += 1;
            state.arrived = 0;
            self.cv.notify_all();
            return Ok(());
        }
        loop {
            if let Some(error) = &state.failed {
                return Err(error.clone());
            }
            if state.generation > generation {
                return Ok(());
            }
            state = self.cv.wait(state).expect("rendezvous state poisoned");
        }
    }
}

/// The D6 reference transport: N threads in one process, one per rank.
pub struct ThreadBackend {
    rank: usize,
    mesh: Mesh,
    shared: Arc<ThreadShared>,
    /// The number of collective calls this rank has executed; the rendezvous
    /// ordinal (×2, for the two meets of one call).
    ordinal: u64,
}

impl ThreadBackend {
    pub fn new(rank: usize, mesh: Mesh, shared: Arc<ThreadShared>) -> Self {
        Self {
            rank,
            mesh,
            shared,
            ordinal: 0,
        }
    }

    /// Tells the rest of the world this rank cannot continue.
    pub fn poison(&self, message: &str) {
        self.shared.poison(message);
    }

    /// This rank's members of `group`, and its own index among them.
    fn members(&self, group: GroupMask) -> Result<(Vec<usize>, usize), String> {
        let members = self
            .mesh
            .group_ranks(group, self.rank)
            .map_err(|e| e.to_string())?;
        let index = self
            .mesh
            .group_index(group, self.rank)
            .map_err(|e| e.to_string())?;
        Ok((members, index))
    }

    /// Writes this rank's contribution into its scratch slot.
    fn contribute(&self, ordinal: u64, bytes: &[u8]) {
        let mut scratch = self.shared.scratch.lock().expect("scratch state poisoned");
        let slots = scratch.entry(ordinal).or_default();
        if slots.len() < self.shared.world {
            slots.resize_with(self.shared.world, Vec::new);
        }
        slots[self.rank].clear();
        slots[self.rank].extend_from_slice(bytes);
    }

    /// Reads one rank's contribution for this ordinal.
    fn take(&self, ordinal: u64, from: usize) -> Vec<u8> {
        let scratch = self.shared.scratch.lock().expect("scratch state poisoned");
        scratch
            .get(&ordinal)
            .and_then(|slots| slots.get(from))
            .cloned()
            .unwrap_or_default()
    }
}

impl CollectiveBackend for ThreadBackend {
    fn execute(
        &mut self,
        req: &CollectiveRequest,
        input: &RsTensor,
        output: &mut RsTensor,
        host: &mut dyn Allocator,
    ) -> Result<CollectiveReport, String> {
        let ordinal = self.ordinal;
        self.ordinal += 1;
        let write_gen = ordinal * 2;
        let read_gen = ordinal * 2 + 1;

        // A failed rank must never block the world: report the recorded error
        // before touching the rendezvous.
        self.shared.meet(write_gen)?;

        let width = element_width(input, output)?;

        // `sync` exchanges no data; both meets still run so the ranks stay in
        // lockstep.
        if req.kind == CollectiveKind::Sync {
            self.shared.meet(read_gen)?;
            let bytes = copy_tensor(input, output, host)?;
            return Ok(CollectiveReport {
                sent_bytes: 0,
                recv_bytes: bytes,
            });
        }

        let (members, my_index) = self.members(req.group)?;
        let degree = members.len();

        // Validate the semantics against the real local shapes *before* any
        // byte moves, then produce this rank's contribution.
        let in_shape = logical_shape(input);
        let out_shape = logical_shape(output);
        let contribution: Vec<u8> = match req.kind {
            CollectiveKind::AllReduce => {
                if in_shape != out_shape {
                    return Err(shape_error(req, &in_shape, &out_shape));
                }
                if req.reduce.is_none() {
                    return Err(format!(
                        "collective {} on group {} carries no reduction",
                        req.kind.as_str(),
                        req.group
                    ));
                }
                materialise(input, width, host)?
            }
            CollectiveKind::AllGather => {
                let dim = resolve_dim(req, &in_shape)?;
                let along = in_shape[dim];
                let mut expected = in_shape.clone();
                let gathered = along
                    .checked_mul(degree as i64)
                    .ok_or_else(|| "the gathered extent overflows".to_string())?;
                expected[dim] = gathered;
                if expected != out_shape {
                    return Err(shape_error(req, &expected, &out_shape));
                }
                materialise(input, width, host)?
            }
            CollectiveKind::ReduceScatter => {
                let dim = resolve_dim(req, &in_shape)?;
                let along = in_shape[dim];
                if along % degree as i64 != 0 {
                    return Err(format!(
                        "reduce_scatter on group {}: the input holds {along} element(s) along \
                         dim {dim}, which does not divide into {degree} rank(s)",
                        req.group
                    ));
                }
                let mut expected = in_shape.clone();
                expected[dim] = along / degree as i64;
                if expected != out_shape {
                    return Err(shape_error(req, &expected, &out_shape));
                }
                materialise(input, width, host)?
            }
            CollectiveKind::Broadcast => {
                let src = req.src.unwrap_or(0);
                if src >= degree {
                    return Err(format!(
                        "broadcast on group {}: source index {src} is out of range for a group \
                         of {degree} rank(s)",
                        req.group
                    ));
                }
                if in_shape != out_shape {
                    return Err(shape_error(req, &in_shape, &out_shape));
                }
                if my_index == src {
                    materialise(input, width, host)?
                } else {
                    Vec::new()
                }
            }
            CollectiveKind::AllToAll => {
                let dim = resolve_dim(req, &in_shape)?;
                let along = in_shape[dim];
                let sizes: Vec<i64> = match &req.split {
                    Some(sizes) => {
                        if sizes.len() != degree {
                            return Err(format!(
                                "all_to_all on group {}: the split holds {} entr(ies) for a \
                                 group of {degree} rank(s)",
                                req.group,
                                sizes.len()
                            ));
                        }
                        let sum: i64 = sizes.iter().sum();
                        if sum != along {
                            return Err(format!(
                                "all_to_all on group {}: the split sums to {sum}, but the input \
                                 holds {along} element(s) along dim {dim}",
                                req.group
                            ));
                        }
                        sizes.clone()
                    }
                    None => {
                        if along % degree as i64 != 0 {
                            return Err(format!(
                                "all_to_all on group {}: the input holds {along} element(s) \
                                 along dim {dim}, which does not split equally into {degree} \
                                 rank(s)",
                                req.group
                            ));
                        }
                        vec![along / degree as i64; degree]
                    }
                };
                // Split entry `i` = the input chunk destined for group index
                // `i` (compiled that way, re-derived here). What I receive is
                // `split[my_index]` from each of the `degree` senders.
                let receive = sizes[my_index]
                    .checked_mul(degree as i64)
                    .ok_or_else(|| "the all_to_all output extent overflows".to_string())?;
                let mut expected = in_shape.clone();
                expected[dim] = receive;
                if expected != out_shape {
                    return Err(shape_error(req, &expected, &out_shape));
                }
                materialise(input, width, host)?
            }
            CollectiveKind::Sync => unreachable!("handled above"),
        };

        let sent = contribution.len() as u64;
        self.contribute(ordinal, &contribution);
        self.shared.meet(read_gen)?;

        // The result, built from the group's contributions. Every member's slot
        // must hold exactly the bytes its rank contributed; anything else is a
        // desynchronised world, reported before arithmetic on it.
        let input_bytes = in_shape.iter().product::<i64>().max(0) as usize * width;
        let result: Vec<u8> = match req.kind {
            CollectiveKind::AllReduce => {
                let op = req.reduce.expect("validated above");
                let elements = contribution.len() / width;
                let mut parts = Vec::with_capacity(degree);
                for member in &members {
                    let part = self.take(ordinal, *member);
                    if part.len() != contribution.len() {
                        return Err(mismatched_contribution(req, *member, &members));
                    }
                    parts.push(part);
                }
                reduce_parts(&parts, elements, op, width)?
            }
            CollectiveKind::AllGather => {
                let dim = resolve_dim(req, &in_shape)?;
                // Each member contributes its whole local slice. The gathered
                // tensor places member `j`'s slab at `[j*along, (j+1)*along)`
                // along `dim`; a per-element assembly over the output's
                // row-major order is correct for any dim (plain concatenation
                // would only be right when `dim` is the outermost axis).
                let mut parts = Vec::with_capacity(degree);
                for member in &members {
                    let part = self.take(ordinal, *member);
                    if part.len() != input_bytes {
                        return Err(mismatched_contribution(req, *member, &members));
                    }
                    parts.push(part);
                }
                let along = in_shape[dim];
                assemble(&parts, &in_shape, &out_shape, dim, width, |c| {
                    ((c / along) as usize, c % along)
                })
            }
            CollectiveKind::ReduceScatter => {
                let dim = resolve_dim(req, &in_shape)?;
                let elements = contribution.len() / width;
                let mut parts = Vec::with_capacity(degree);
                for member in &members {
                    let part = self.take(ordinal, *member);
                    if part.len() != contribution.len() {
                        return Err(mismatched_contribution(req, *member, &members));
                    }
                    parts.push(part);
                }
                let full = reduce_parts(&parts, elements, ReduceOp::Sum, width)?;
                let chunk = in_shape[dim] as usize / degree;
                slice_axis(&full, &in_shape, dim, my_index * chunk, chunk, width)
            }
            CollectiveKind::Broadcast => {
                let src_rank = members[req.src.unwrap_or(0)];
                let part = self.take(ordinal, src_rank);
                if part.len() != input_bytes {
                    return Err(format!(
                        "broadcast on group {}: source rank {src_rank} contributed {} byte(s) \
                         but the tensor holds {input_bytes}",
                        req.group,
                        part.len()
                    ));
                }
                part
            }
            CollectiveKind::AllToAll => {
                let dim = resolve_dim(req, &in_shape)?;
                let sizes: Vec<i64> = match &req.split {
                    Some(sizes) => sizes.clone(),
                    None => vec![in_shape[dim] / degree as i64; degree],
                };
                let mut starts = vec![0i64; degree];
                for i in 1..degree {
                    starts[i] = starts[i - 1] + sizes[i - 1];
                }
                // I receive `sizes[my_index]` elements along `dim` from every
                // member; sender `j`'s slab is its input chunk
                // `[starts[my_index], +sizes[my_index])`. Same per-element
                // assembly as all_gather, keyed by the sender.
                let mut parts = Vec::with_capacity(degree);
                for member in &members {
                    let part = self.take(ordinal, *member);
                    if part.len() != input_bytes {
                        return Err(mismatched_contribution(req, *member, &members));
                    }
                    parts.push(part);
                }
                let receive = sizes[my_index];
                assemble(&parts, &in_shape, &out_shape, dim, width, |c| {
                    ((c / receive) as usize, starts[my_index] + c % receive)
                })
            }
            CollectiveKind::Sync => unreachable!("handled above"),
        };

        let recv = result.len() as u64;
        scatter(output, &result, width, host)?;
        Ok(CollectiveReport {
            sent_bytes: sent,
            recv_bytes: recv,
        })
    }
}

pub(crate) fn element_width(input: &RsTensor, output: &mut RsTensor) -> Result<usize, String> {
    let in_width = input.dtype.byte_width().map(|w| w as usize);
    let out_width = output.dtype.byte_width().map(|w| w as usize);
    match (in_width, out_width) {
        (Some(a), Some(b)) if a == b => Ok(a.max(1)),
        _ => Err(format!(
            "the collective's operands have unmappable dtypes ({} vs {})",
            input.dtype, output.dtype
        )),
    }
}

/// The logical shape a descriptor presents (dims only; strides are the view's
/// business and `materialise`/`scatter` handle them).
pub(crate) fn logical_shape(t: &RsTensor) -> Vec<i64> {
    t.dims().to_vec()
}

pub(crate) fn resolve_dim(req: &CollectiveRequest, shape: &[i64]) -> Result<usize, String> {
    let rank = shape.len() as i64;
    let dim = req.dim.unwrap_or(0);
    let dim = if dim < 0 { dim + rank } else { dim };
    if dim < 0 || dim >= rank {
        return Err(format!(
            "collective {} on group {}: dim {dim} is out of range for a rank-{rank} tensor",
            req.kind.as_str(),
            req.group
        ));
    }
    Ok(dim as usize)
}

pub(crate) fn shape_error(req: &CollectiveRequest, expected: &[i64], actual: &[i64]) -> String {
    format!(
        "collective {} on group {}: the exchange produces shape {expected:?}, but the output \
         slot holds {actual:?}",
        req.kind.as_str(),
        req.group
    )
}

fn mismatched_contribution(req: &CollectiveRequest, rank: usize, members: &[usize]) -> String {
    format!(
        "collective {} on group {}: rank {rank} (member of {members:?}) contributed a \
         different number of bytes than the rest; the ranks are no longer in lockstep",
        req.kind.as_str(),
        req.group
    )
}

/// The byte span a descriptor reads from its base pointer: the largest offset
/// its shape and strides reach, plus one element. A device operand must come
/// back in one transfer of the whole span — walking it element by element is
/// only possible once it is host memory.
pub(crate) fn span_of(t: &RsTensor, width: usize) -> u64 {
    let shape = logical_shape(t);
    let rank = (t.rank as usize).min(shape.len());
    let mut max_offset = 0i64;
    for (dim, stride) in shape[..rank].iter().zip(&t.stride[..rank]) {
        if *dim > 0 {
            max_offset += (*dim - 1) * (*stride).max(0);
        }
    }
    (max_offset.max(0) as u64 + 1) * width.max(1) as u64
}

/// Copies a strided view into a contiguous host buffer, in row-major logical
/// order. The operand's bytes are brought to the host through `host` first, so
/// a device slot is staged in one transfer instead of being dereferenced here.
pub(crate) fn materialise(
    t: &RsTensor,
    width: usize,
    host: &dyn Allocator,
) -> Result<Vec<u8>, String> {
    let shape = logical_shape(t);
    let rank = (t.rank as usize).min(shape.len());
    let strides = &t.stride[..rank];
    let total: usize = shape[..rank].iter().product::<i64>().max(0) as usize;
    let mut out = vec![0u8; total * width];
    if total == 0 || t.data.is_null() {
        return Ok(out);
    }
    let staged = host.copy_out(t.data, span_of(t, width))?;
    let mut index = vec![0i64; rank];
    for linear in 0..total {
        let mut offset = 0i64;
        for d in 0..rank {
            offset += index[d] * strides[d];
        }
        let src = offset.max(0) as usize * width;
        let dst = linear * width;
        if src + width > staged.len() || dst + width > out.len() {
            return Err(format!(
                "the descriptor reads element {linear} at byte {src}, past the {} byte(s) its \
                 buffer covers",
                staged.len()
            ));
        }
        out[dst..dst + width].copy_from_slice(&staged[src..src + width]);
        for d in (0..rank).rev() {
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    Ok(out)
}

/// Whether a descriptor's layout is plain row-major for its logical shape, so a
/// transfer can move the bytes as they are instead of element by element.
///
/// Strides are in *elements* (`rs_tensor`'s contract), so the expected stride
/// starts at one and grows by the shape, not by the element width.
pub(crate) fn is_row_major(t: &RsTensor, shape: &[i64]) -> bool {
    let mut expected = 1i64;
    for d in (0..shape.len()).rev() {
        if shape[d] > 1 && t.stride[d] != expected {
            return false;
        }
        expected *= shape[d].max(1);
    }
    true
}

/// Writes a contiguous row-major result into a (possibly strided) descriptor.
/// A row-major output is one transfer; a strided one stages the span, edits it
/// on the host and writes it back.
pub(crate) fn scatter(
    t: &mut RsTensor,
    data: &[u8],
    width: usize,
    host: &mut dyn Allocator,
) -> Result<(), String> {
    let shape = logical_shape(t);
    let rank = (t.rank as usize).min(shape.len());
    let strides = &t.stride[..rank];
    let total: usize = shape[..rank].iter().product::<i64>().max(0) as usize;
    if data.len() != total * width {
        return Err(format!(
            "the collective produced {} byte(s) but the output holds {total} element(s) of \
             {width} byte(s)",
            data.len()
        ));
    }
    if t.data.is_null() {
        return Err("the collective's output slot has no data pointer".to_string());
    }
    if is_row_major(t, &shape[..rank]) {
        return host.copy_in(t.data, data.len() as u64, data);
    }
    let span = span_of(t, width);
    let mut staged = host.copy_out(t.data, span)?;
    let mut index = vec![0i64; rank];
    for linear in 0..total {
        let mut offset = 0i64;
        for d in 0..rank {
            offset += index[d] * strides[d];
        }
        let dst = offset.max(0) as usize * width;
        if dst + width > staged.len() {
            return Err(format!(
                "the output descriptor writes element {linear} at byte {dst}, past the {} \
                 byte(s) its buffer covers",
                staged.len()
            ));
        }
        staged[dst..dst + width].copy_from_slice(&data[linear * width..(linear + 1) * width]);
        for d in (0..rank).rev() {
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    host.copy_in(t.data, span, &staged)
}

/// Copies one tensor into another, honouring both descriptors' shapes and
/// strides. The shapes must agree; the data layouts need not.
pub(crate) fn copy_tensor(
    input: &RsTensor,
    output: &mut RsTensor,
    host: &mut dyn Allocator,
) -> Result<u64, String> {
    let width = {
        let in_width = input.dtype.byte_width().map(|w| w as usize);
        let out_width = output.dtype.byte_width().map(|w| w as usize);
        match (in_width, out_width) {
            (Some(a), Some(b)) if a == b => a.max(1),
            _ => {
                return Err(format!(
                    "the collective's operands have unmappable dtypes ({} vs {})",
                    input.dtype, output.dtype
                ));
            }
        }
    };
    if logical_shape(input) != logical_shape(output) {
        return Err(format!(
            "the collective's operands have different logical shapes ({:?} vs {:?})",
            logical_shape(input),
            logical_shape(output)
        ));
    }
    let data = materialise(input, width, host)?;
    scatter(output, &data, width, host)?;
    Ok(data.len() as u64)
}

/// Elementwise reduction over equal-length byte buffers, `width` bytes per
/// element. Only the widths the reference provider executes (f32/f64) are
/// reduced; the plan is widened to f32 before any run, and anything else is a
/// reported error rather than a guessed reduction.
fn reduce_parts(
    parts: &[Vec<u8>],
    elements: usize,
    op: ReduceOp,
    width: usize,
) -> Result<Vec<u8>, String> {
    let mut out = vec![0u8; elements * width];
    match width {
        4 => {
            let mut acc: Vec<f32> = vec![0.0; elements];
            let mut first = true;
            for part in parts {
                if part.len() != elements * width {
                    return Err(format!(
                        "a rank contributed {} byte(s) for a {}-element reduction",
                        part.len(),
                        elements
                    ));
                }
                let values: Vec<f32> = part
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                if first {
                    acc.copy_from_slice(&values);
                    first = false;
                } else {
                    for (a, v) in acc.iter_mut().zip(&values) {
                        *a = match op {
                            ReduceOp::Sum => *a + *v,
                            ReduceOp::Max => a.max(*v),
                            ReduceOp::Min => a.min(*v),
                        };
                    }
                }
            }
            for (a, bytes) in acc.iter().zip(out.chunks_exact_mut(4)) {
                bytes.copy_from_slice(&a.to_ne_bytes());
            }
        }
        8 => {
            let mut acc: Vec<f64> = vec![0.0; elements];
            let mut first = true;
            for part in parts {
                if part.len() != elements * width {
                    return Err(format!(
                        "a rank contributed {} byte(s) for a {}-element reduction",
                        part.len(),
                        elements
                    ));
                }
                let values: Vec<f64> = part
                    .chunks_exact(8)
                    .map(|c| f64::from_ne_bytes(c.try_into().expect("8-byte chunks")))
                    .collect();
                if first {
                    acc.copy_from_slice(&values);
                    first = false;
                } else {
                    for (a, v) in acc.iter_mut().zip(&values) {
                        *a = match op {
                            ReduceOp::Sum => *a + *v,
                            ReduceOp::Max => a.max(*v),
                            ReduceOp::Min => a.min(*v),
                        };
                    }
                }
            }
            for (a, bytes) in acc.iter().zip(out.chunks_exact_mut(8)) {
                bytes.copy_from_slice(&a.to_ne_bytes());
            }
        }
        other => {
            return Err(format!(
                "the reference collective backend reduces 4- and 8-byte elements, not \
                 {other}-byte ones; run the plan at f32"
            ));
        }
    }
    Ok(out)
}

/// Copies the `[start, start + len)` sub-tensor along `dim` out of a row-major
/// buffer (element-wise gather over the output's order).
fn slice_axis(
    values: &[u8],
    shape: &[i64],
    dim: usize,
    start: usize,
    len: usize,
    width: usize,
) -> Vec<u8> {
    let strides = row_major_strides(shape);
    let mut out_shape: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
    out_shape[dim] = len;
    let total: usize = out_shape.iter().product();
    let mut out = vec![0u8; total * width];
    let mut index = vec![0usize; out_shape.len()];
    for dst in out.chunks_exact_mut(width) {
        let mut offset = start * strides[dim];
        for (i, count) in index.iter().enumerate() {
            offset += *count * strides[i];
        }
        dst.copy_from_slice(&values[offset * width..offset * width + width]);
        for d in (0..out_shape.len()).rev() {
            index[d] += 1;
            if index[d] < out_shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    out
}

/// Assembles an exchange result element by element: for every output coordinate
/// along `dim`, `map` says which member's local buffer holds the element and at
/// which local coordinate along `dim`. Correct for any dim, because the walk
/// follows the output's row-major order explicitly instead of assuming a
/// contiguous concatenation.
fn assemble(
    parts: &[Vec<u8>],
    in_shape: &[i64],
    out_shape: &[i64],
    dim: usize,
    width: usize,
    map: impl Fn(i64) -> (usize, i64),
) -> Vec<u8> {
    let total: usize = out_shape.iter().product::<i64>().max(0) as usize;
    let mut out = vec![0u8; total * width];
    let local_strides = row_major_strides(in_shape);
    let mut index = vec![0i64; out_shape.len()];
    for linear in 0..total {
        let (member, local) = map(index[dim]);
        let mut offset: i64 = local * local_strides[dim] as i64;
        for (d, count) in index.iter().enumerate() {
            if d == dim {
                continue;
            }
            offset += *count * local_strides[d] as i64;
        }
        let src = &parts[member][offset as usize * width..offset as usize * width + width];
        out[linear * width..linear * width + width].copy_from_slice(src);
        for d in (0..out_shape.len()).rev() {
            index[d] += 1;
            if index[d] < out_shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    out
}

/// Row-major strides (in elements).
fn row_major_strides(shape: &[i64]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1].max(0) as usize;
    }
    strides
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use rustrain_abi::ffi::{RsDeviceKind, RsDtype, RsTensor};
    use rustrain_parallel::GroupMask;

    use super::*;
    use crate::HostAllocator;

    /// A shared backend must hand the *same* backend both to the executor and to the warming
    /// thread, or the warm would create a communicator nobody uses and the first collective
    /// would still pay for a second one. This pins the delegation *and* the records it leaves
    /// behind: a `SharedBackend` whose `warm` was a no-op, or whose `handle()` returned a fresh
    /// backend, fails here.
    #[test]
    fn a_shared_backend_warms_the_backend_it_executes_on() {
        #[derive(Default)]
        struct Log {
            warmed: Vec<Vec<u32>>,
            executed: usize,
        }

        /// The log lives outside the backend, because the backend is moved into the adapter and a
        /// `Box<dyn CollectiveBackend>` cannot be taken back to be inspected.
        struct Recording(Arc<Mutex<Log>>);

        impl CollectiveBackend for Recording {
            fn execute(
                &mut self,
                _req: &CollectiveRequest,
                _input: &RsTensor,
                _output: &mut RsTensor,
                _host: &mut dyn Allocator,
            ) -> Result<CollectiveReport, String> {
                self.0.lock().expect("not poisoned").executed += 1;
                Ok(CollectiveReport::default())
            }

            fn warm(&mut self, groups: &[GroupMask]) -> Result<(), String> {
                self.0
                    .lock()
                    .expect("not poisoned")
                    .warmed
                    .push(groups.iter().map(|group| group.bits()).collect());
                Ok(())
            }
        }

        let log = Arc::new(Mutex::new(Log::default()));
        let shared = SharedBackend::new(Box::new(Recording(Arc::clone(&log))));
        let handle = shared.handle();
        let groups = [GroupMask::from_bits(0b1), GroupMask::from_bits(0b1 | 0b100)];
        // From another thread, exactly as the runner does it.
        let warming = std::thread::spawn(move || {
            let mut backend = handle.lock().expect("not poisoned");
            backend.warm(&groups[..])?;
            // The trait's contract is idempotency: warming twice must not need a second
            // communicator, and the runner may warm a group the plan repeats.
            backend.warm(&groups[..])
        });
        warming.join().unwrap().unwrap();

        // And the executor's own path still works afterwards, on the same object.
        let mut shared = shared;
        let req = CollectiveRequest {
            kind: CollectiveKind::Sync,
            group: GroupMask::from_bits(0b1),
            reduce: None,
            dim: None,
            split: None,
            src: None,
        };
        let input = RsTensor::default();
        let mut output = RsTensor::default();
        let mut host = HostAllocator::new();
        shared
            .execute(&req, &input, &mut output, &mut host)
            .expect("the shared backend executes");
        let log = log.lock().expect("not poisoned");
        assert_eq!(
            log.warmed,
            vec![vec![0b1, 0b1 | 0b100], vec![0b1, 0b1 | 0b100]],
            "both warms reached the backend the executor uses"
        );
        assert_eq!(log.executed, 1, "and the execute went to the same one");
        drop(log);
    }

    /// A panic inside a collective leaves the backend's state unknown, and the adapter says so
    /// instead of panicking in the caller or handing out a half-built backend.
    #[test]
    fn a_poisoned_backend_refuses_instead_of_panicking() {
        struct Panics;

        impl CollectiveBackend for Panics {
            fn execute(
                &mut self,
                _req: &CollectiveRequest,
                _input: &RsTensor,
                _output: &mut RsTensor,
                _host: &mut dyn Allocator,
            ) -> Result<CollectiveReport, String> {
                panic!("a collective panicked");
            }

            fn warm(&mut self, _groups: &[GroupMask]) -> Result<(), String> {
                panic!("a warm panicked");
            }
        }

        let shared = SharedBackend::new(Box::new(Panics));
        let handle = shared.handle();
        let _ = std::thread::spawn(move || {
            let mut guard = handle.lock().expect("not poisoned yet");
            let _ = guard.warm(&[]);
        })
        .join();
        let handle = shared.handle();
        assert!(
            handle.lock().is_err(),
            "the panic poisoned the lock, which is the flag the adapter reads"
        );

        let mut shared = shared;
        let req = CollectiveRequest {
            kind: CollectiveKind::Sync,
            group: GroupMask::from_bits(0b1),
            reduce: None,
            dim: None,
            split: None,
            src: None,
        };
        let input = RsTensor::default();
        let mut output = RsTensor::default();
        let mut host = HostAllocator::new();
        let error = shared
            .execute(&req, &input, &mut output, &mut host)
            .expect_err("a poisoned backend refuses");
        assert!(
            error.contains("poisoned"),
            "the refusal names the reason: {error}"
        );
        assert!(
            shared.warm(&[]).is_err(),
            "and the warm path refuses the same way"
        );
    }

    /// The allocator is the only thing that knows where a buffer lives, so a
    /// collective must move every byte through it. A backend that dereferenced
    /// the descriptors itself would pass on the host and read device memory
    /// from the host on a GPU — which is exactly how the first D6 device run
    /// died, in `materialise`, with a SIGSEGV.
    #[test]
    fn single_rank_moves_bytes_through_the_allocator() {
        struct RecordingAllocator {
            inner: HostAllocator,
            in_calls: Arc<AtomicUsize>,
            out_calls: Arc<AtomicUsize>,
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
                self.in_calls.fetch_add(1, Ordering::SeqCst);
                self.inner.copy_in(dst, bytes, src)
            }

            fn copy_out(&self, src: *const c_void, bytes: u64) -> Result<Vec<u8>, String> {
                self.out_calls.fetch_add(1, Ordering::SeqCst);
                self.inner.copy_out(src, bytes)
            }
        }

        let mut host = HostAllocator::new();
        let input_bytes = 4u64 * 4;
        let input_ptr = host.alloc(input_bytes, RsDeviceKind::CPU).unwrap();
        host.copy_in(
            input_ptr,
            input_bytes,
            &[
                1.0f32.to_le_bytes(),
                2.0f32.to_le_bytes(),
                3.0f32.to_le_bytes(),
                4.0f32.to_le_bytes(),
            ]
            .concat(),
        )
        .unwrap();
        let output_ptr = host.alloc(input_bytes, RsDeviceKind::CPU).unwrap();

        let mut input = RsTensor::new(RsDtype::F32, &[4]);
        input.data = input_ptr;
        let mut output = RsTensor::new(RsDtype::F32, &[4]);
        output.data = output_ptr;

        let in_calls = Arc::new(AtomicUsize::new(0));
        let out_calls = Arc::new(AtomicUsize::new(0));
        let mut allocator = RecordingAllocator {
            inner: host,
            in_calls: in_calls.clone(),
            out_calls: out_calls.clone(),
        };
        let request = CollectiveRequest {
            kind: CollectiveKind::AllReduce,
            group: GroupMask::from_bits(1),
            reduce: Some(ReduceOp::Sum),
            dim: None,
            split: None,
            src: None,
        };
        let report = SingleRank::new(1)
            .execute(&request, &input, &mut output, &mut allocator)
            .expect("the identity collective on one rank");
        assert_eq!(report.recv_bytes, input_bytes);
        assert_eq!(
            out_calls.load(Ordering::SeqCst),
            1,
            "the input was not read through the allocator"
        );
        assert_eq!(
            in_calls.load(Ordering::SeqCst),
            1,
            "the output was not written through the allocator"
        );

        let written = allocator.inner.copy_out(output_ptr, input_bytes).unwrap();
        let values: Vec<f32> = written
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
        allocator.inner.dealloc(input_ptr, input_bytes);
        allocator.inner.dealloc(output_ptr, input_bytes);
    }
}
