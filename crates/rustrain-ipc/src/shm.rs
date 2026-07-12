use std::ffi::CString;
use std::io;
use std::ptr;

use crate::command::{EpCommand, EpResult};

/// Each slot: 4 bytes len + up to SLOT_DATA bytes JSON
const SLOT_HEADER: usize = 4;
const SLOT_DATA: usize = 64 * 1024; // 64KB per slot
const SLOT_SIZE: usize = SLOT_HEADER + SLOT_DATA;

/// Shared memory layout:
///   [0..SLOT_SIZE)           command slot (parent writes, all workers read)
///   [SLOT_SIZE..SLOT_SIZE*(1+world_size))  per-worker result slots
///
/// Semaphores are allocated after the data region.
const SHM_SIZE: usize = 256 * 1024;

fn cmd_offset() -> usize { 0 }
fn result_offset(rank: usize) -> usize { SLOT_SIZE * (1 + rank) }
fn sem_region_start(world_size: usize) -> usize { SLOT_SIZE * (1 + world_size) }

/// Parent-side coordinator: signals all workers, waits for completion.
pub struct EpChannel {
    shm_ptr: *mut u8,
    shm_size: usize,
    shm_fd: i32,
    sem_request: Vec<*mut libc::sem_t>,
    sem_done: Vec<*mut libc::sem_t>,
    world_size: usize,
    shm_name: String,
}

/// Worker-side endpoint: waits for commands, signals completion.
pub struct EpWorker {
    shm_ptr: *mut u8,
    shm_size: usize,
    sem_request: *mut libc::sem_t,
    sem_done: *mut libc::sem_t,
    rank: usize,
    world_size: usize,
}

unsafe impl Send for EpChannel {}
unsafe impl Sync for EpChannel {}
unsafe impl Send for EpWorker {}

impl EpChannel {
    pub fn new(world_size: usize) -> io::Result<Self> {
        let shm_name = format!("/rustrain-ep-{}", std::process::id());
        let c_name = CString::new(shm_name.as_str()).unwrap();

        let needed = sem_region_start(world_size) + world_size * 2 * std::mem::size_of::<libc::sem_t>();
        let shm_size = needed.max(SHM_SIZE);

        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        if unsafe { libc::ftruncate(fd, shm_size as i64) } < 0 {
            let e = io::Error::last_os_error();
            unsafe { libc::close(fd); libc::shm_unlink(c_name.as_ptr()) };
            return Err(e);
        }
        let ptr = unsafe {
            libc::mmap(ptr::null_mut(), shm_size, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0)
        };
        if ptr == libc::MAP_FAILED {
            let e = io::Error::last_os_error();
            unsafe { libc::close(fd); libc::shm_unlink(c_name.as_ptr()) };
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
            if unsafe { libc::sem_init((ptr as *mut u8).add(off_req) as *mut libc::sem_t, 1, 0) } != 0 {
                return Err(io::Error::last_os_error());
            }
            if unsafe { libc::sem_init((ptr as *mut u8).add(off_done) as *mut libc::sem_t, 1, 0) } != 0 {
                return Err(io::Error::last_os_error());
            }
            sems_request.push(unsafe { (ptr as *mut u8).add(off_req) as *mut libc::sem_t });
            sems_done.push(unsafe { (ptr as *mut u8).add(off_done) as *mut libc::sem_t });
        }

        tracing::info!("EP IPC: created shm '{}' ({}KB), {} workers", shm_name, shm_size / 1024, world_size);

        Ok(Self { shm_ptr: ptr as *mut u8, shm_size, shm_fd: fd, sem_request: sems_request, sem_done: sems_done, world_size, shm_name })
    }

    pub fn shm_name(&self) -> &str { &self.shm_name }
    pub fn world_size(&self) -> usize { self.world_size }

    /// Send command to ALL workers, wait for ALL to complete, return rank 0's result.
    pub fn broadcast(&self, cmd: &EpCommand) -> io::Result<EpResult> {
        let json = serde_json::to_vec(cmd).map_err(|e| io::Error::other(e.to_string()))?;
        if json.len() > SLOT_DATA {
            return Err(io::Error::other(format!("command too large: {} bytes", json.len())));
        }

        // Write command to shared slot (offset 0)
        unsafe {
            let len = json.len() as u32;
            ptr::copy_nonoverlapping(&len as *const u32 as *const u8, self.shm_ptr.add(cmd_offset()), SLOT_HEADER);
            ptr::copy_nonoverlapping(json.as_ptr(), self.shm_ptr.add(cmd_offset() + SLOT_HEADER), json.len());
        }

        // Signal all workers simultaneously
        for i in 0..self.world_size {
            if unsafe { libc::sem_post(self.sem_request[i]) } != 0 {
                return Err(io::Error::last_os_error());
            }
        }

        // Wait for all workers to complete
        for i in 0..self.world_size {
            if unsafe { libc::sem_wait(self.sem_done[i]) } != 0 {
                return Err(io::Error::last_os_error());
            }
        }

        // Read rank 0's result from its dedicated slot
        let r0_off = result_offset(0);
        unsafe {
            let result_len = *(self.shm_ptr.add(r0_off) as *const u32) as usize;
            if result_len == 0 || result_len > SLOT_DATA {
                return Err(io::Error::other("worker returned empty result"));
            }
            let result_bytes = std::slice::from_raw_parts(self.shm_ptr.add(r0_off + SLOT_HEADER), result_len);
            serde_json::from_slice::<EpResult>(result_bytes)
                .map_err(|e| io::Error::other(format!("result deserialization: {e}")))
        }
    }
}

impl Drop for EpChannel {
    fn drop(&mut self) {
        for i in 0..self.world_size {
            unsafe { libc::sem_destroy(self.sem_request[i]); libc::sem_destroy(self.sem_done[i]) };
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
        if fd < 0 { return Err(io::Error::last_os_error()); }

        let needed = sem_region_start(world_size) + world_size * 2 * std::mem::size_of::<libc::sem_t>();
        let shm_size = needed.max(SHM_SIZE);

        let ptr = unsafe { libc::mmap(ptr::null_mut(), shm_size, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0) };
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
        Ok(Self { shm_ptr: ptr as *mut u8, shm_size, sem_request, sem_done, rank, world_size })
    }

    pub fn wait_command(&self) -> io::Result<EpCommand> {
        if unsafe { libc::sem_wait(self.sem_request) } != 0 {
            return Err(io::Error::last_os_error());
        }
        unsafe {
            let cmd_len = *(self.shm_ptr.add(cmd_offset()) as *const u32) as usize;
            if cmd_len == 0 || cmd_len > SLOT_DATA {
                return Err(io::Error::other(format!("invalid command length: {}", cmd_len)));
            }
            let cmd_bytes = std::slice::from_raw_parts(self.shm_ptr.add(cmd_offset() + SLOT_HEADER), cmd_len);
            serde_json::from_slice::<EpCommand>(cmd_bytes)
                .map_err(|e| io::Error::other(format!("command deserialization: {e}")))
        }
    }

    pub fn signal_done(&self, result: &EpResult) -> io::Result<()> {
        let json = serde_json::to_vec(result).map_err(|e| io::Error::other(e.to_string()))?;
        if json.len() > SLOT_DATA {
            return Err(io::Error::other(format!("result too large: {} bytes", json.len())));
        }

        // Write result to THIS worker's dedicated slot (no collision with other workers)
        let off = result_offset(self.rank);
        unsafe {
            let len = json.len() as u32;
            ptr::copy_nonoverlapping(&len as *const u32 as *const u8, self.shm_ptr.add(off), SLOT_HEADER);
            ptr::copy_nonoverlapping(json.as_ptr(), self.shm_ptr.add(off + SLOT_HEADER), json.len());
        }

        if unsafe { libc::sem_post(self.sem_done) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub fn rank(&self) -> usize { self.rank }
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
    fn test_channel_create_destroy() {
        let ch = EpChannel::new(2).expect("create channel");
        assert_eq!(ch.world_size(), 2);
        drop(ch);
    }
}
