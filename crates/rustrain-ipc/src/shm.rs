use std::ffi::CString;
use std::io;
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::command::{EpCommand, EpResult};

/// Each slot: 4 bytes len + up to SLOT_DATA bytes JSON
const SLOT_HEADER: usize = 4;
const SLOT_DATA: usize = 256 * 1024; // 256KB per slot (supports seq_len up to ~32K)
const SLOT_SIZE: usize = SLOT_HEADER + SLOT_DATA;

/// Shared memory layout:
///   [0..SLOT_SIZE)           command slot (parent writes, all workers read)
///   [SLOT_SIZE..SLOT_SIZE*(1+world_size))  per-worker result slots
///
/// Semaphores are allocated after the data region.
const SHM_SIZE: usize = 256 * 1024;

/// Default upper bound for one command across all workers.
pub const DEFAULT_BROADCAST_TIMEOUT: Duration = Duration::from_secs(30 * 60);

fn cmd_offset() -> usize {
    0
}
fn result_offset(rank: usize) -> usize {
    SLOT_SIZE * (1 + rank)
}
fn sem_region_start(world_size: usize) -> usize {
    SLOT_SIZE * (1 + world_size)
}

/// Parent-side coordinator: signals all workers, waits for completion.
pub struct EpChannel {
    shm_ptr: *mut u8,
    shm_size: usize,
    shm_fd: i32,
    sem_request: Vec<*mut libc::sem_t>,
    sem_done: Vec<*mut libc::sem_t>,
    world_size: usize,
    shm_name: String,
    default_timeout: Duration,
    poisoned: AtomicBool,
    broadcast_lock: Mutex<()>,
}

/// Worker-side endpoint: waits for commands, signals completion.
pub struct EpWorker {
    shm_ptr: *mut u8,
    shm_size: usize,
    sem_request: *mut libc::sem_t,
    sem_done: *mut libc::sem_t,
    rank: usize,
}

unsafe impl Send for EpChannel {}
unsafe impl Sync for EpChannel {}
unsafe impl Send for EpWorker {}

impl EpChannel {
    pub fn new(world_size: usize) -> io::Result<Self> {
        Self::new_with_timeout(world_size, DEFAULT_BROADCAST_TIMEOUT)
    }

    pub fn new_with_timeout(world_size: usize, default_timeout: Duration) -> io::Result<Self> {
        let shm_name = format!("/rustrain-ep-{}", std::process::id());
        let c_name = CString::new(shm_name.as_str()).unwrap();

        let needed =
            sem_region_start(world_size) + world_size * 2 * std::mem::size_of::<libc::sem_t>();
        let shm_size = needed.max(SHM_SIZE);

        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        if unsafe { libc::ftruncate(fd, shm_size as i64) } < 0 {
            let e = io::Error::last_os_error();
            unsafe {
                libc::close(fd);
                libc::shm_unlink(c_name.as_ptr())
            };
            return Err(e);
        }
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
            unsafe {
                libc::close(fd);
                libc::shm_unlink(c_name.as_ptr())
            };
            return Err(e);
        }

        unsafe { ptr::write_bytes(ptr as *mut u8, 0, shm_size) };

        let sem_start = sem_region_start(world_size);
        let sem_sz = std::mem::size_of::<libc::sem_t>();
        let mut sems_request = Vec::with_capacity(world_size);
        let mut sems_done = Vec::with_capacity(world_size);

        for i in 0..world_size {
            let off_req = sem_start + i * 2 * sem_sz;
            let off_done = sem_start + (i * 2 + 1) * sem_sz;
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
            shm_size / 1024,
            world_size
        );

        Ok(Self {
            shm_ptr: ptr as *mut u8,
            shm_size,
            shm_fd: fd,
            sem_request: sems_request,
            sem_done: sems_done,
            world_size,
            shm_name,
            default_timeout,
            poisoned: AtomicBool::new(false),
            broadcast_lock: Mutex::new(()),
        })
    }

    pub fn shm_name(&self) -> &str {
        &self.shm_name
    }
    pub fn world_size(&self) -> usize {
        self.world_size
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
        self.ensure_healthy()?;
        let _guard = self
            .broadcast_lock
            .lock()
            .map_err(|error| self.poison(format!("EP broadcast lock poisoned: {error}")))?;
        self.ensure_healthy()?;

        let json = serde_json::to_vec(cmd).map_err(|e| io::Error::other(e.to_string()))?;
        if json.len() > SLOT_DATA {
            return Err(io::Error::other(format!(
                "command too large: {} bytes",
                json.len()
            )));
        }

        // Write command to shared slot (offset 0)
        unsafe {
            let len = json.len() as u32;
            ptr::copy_nonoverlapping(
                &len as *const u32 as *const u8,
                self.shm_ptr.add(cmd_offset()),
                SLOT_HEADER,
            );
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
                .read_result(rank)
                .map_err(|error| self.poison(error.to_string()))?;
            if let EpResult::Error(error) = &result {
                if worker_error.is_none() {
                    worker_error = Some(EpResult::Error(format!("worker rank {rank}: {error}")));
                }
            }
            if rank == 0 {
                rank_zero = Some(result);
            }
        }
        if let Some(error) = worker_error {
            return Ok(error);
        }
        rank_zero.ok_or_else(|| io::Error::other("EP broadcast has no rank 0 worker"))
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

    fn read_result(&self, rank: usize) -> io::Result<EpResult> {
        let offset = result_offset(rank);
        unsafe {
            let result_len = *(self.shm_ptr.add(offset) as *const u32) as usize;
            if result_len == 0 || result_len > SLOT_DATA {
                return Err(io::Error::other(format!(
                    "worker rank {rank} returned an invalid result length {result_len}"
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
        let c_name = CString::new(shm_name).unwrap();
        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        let needed =
            sem_region_start(world_size) + world_size * 2 * std::mem::size_of::<libc::sem_t>();
        let shm_size = needed.max(SHM_SIZE);

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

        let sem_start = sem_region_start(world_size);
        let sem_sz = std::mem::size_of::<libc::sem_t>();
        let off_req = sem_start + rank * 2 * sem_sz;
        let off_done = sem_start + (rank * 2 + 1) * sem_sz;
        let sem_request = unsafe { (ptr as *mut u8).add(off_req) as *mut libc::sem_t };
        let sem_done = unsafe { (ptr as *mut u8).add(off_done) as *mut libc::sem_t };

        tracing::info!("EP worker {}: attached to shm '{}'", rank, shm_name);
        Ok(Self {
            shm_ptr: ptr as *mut u8,
            shm_size,
            sem_request,
            sem_done,
            rank,
        })
    }

    pub fn wait_command(&self) -> io::Result<EpCommand> {
        if unsafe { libc::sem_wait(self.sem_request) } != 0 {
            return Err(io::Error::last_os_error());
        }
        unsafe {
            let cmd_len = *(self.shm_ptr.add(cmd_offset()) as *const u32) as usize;
            if cmd_len == 0 || cmd_len > SLOT_DATA {
                return Err(io::Error::other(format!(
                    "invalid command length: {}",
                    cmd_len
                )));
            }
            let cmd_bytes =
                std::slice::from_raw_parts(self.shm_ptr.add(cmd_offset() + SLOT_HEADER), cmd_len);
            serde_json::from_slice::<EpCommand>(cmd_bytes)
                .map_err(|e| io::Error::other(format!("command deserialization: {e}")))
        }
    }

    pub fn signal_done(&self, result: &EpResult) -> io::Result<()> {
        let json = serde_json::to_vec(result).map_err(|e| io::Error::other(e.to_string()))?;
        if json.len() > SLOT_DATA {
            return Err(io::Error::other(format!(
                "result too large: {} bytes",
                json.len()
            )));
        }

        // Write result to THIS worker's dedicated slot (no collision with other workers)
        let off = result_offset(self.rank);
        unsafe {
            let len = json.len() as u32;
            ptr::copy_nonoverlapping(
                &len as *const u32 as *const u8,
                self.shm_ptr.add(off),
                SLOT_HEADER,
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
        Ok(())
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl Drop for EpWorker {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.shm_ptr as *mut libc::c_void, self.shm_size) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_CHANNEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_channel_create_destroy() {
        let _guard = TEST_CHANNEL_LOCK.lock().unwrap();
        let ch = EpChannel::new(2).expect("create channel");
        assert_eq!(ch.world_size(), 2);
        drop(ch);
    }

    #[test]
    fn broadcast_propagates_nonzero_rank_error() {
        let _guard = TEST_CHANNEL_LOCK.lock().unwrap();
        let channel = EpChannel::new(2).expect("create channel");
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
        let channel = EpChannel::new(1).expect("create channel");
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
}
