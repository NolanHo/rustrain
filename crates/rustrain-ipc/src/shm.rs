use std::ffi::CString;
use std::io;
use std::ptr;

use crate::command::{EpCommand, EpResult};

/// Size of the shared memory segment for command/result payload.
/// 256 KB is enough for large JSON payloads (train_step with seq=512 ≈ 12 KB).
const SHM_SIZE: usize = 256 * 1024;

/// Header layout in shared memory:
///   [0..4]      command_len (u32, little-endian)
///   [4..8]      result_len   (u32, little-endian)
///   [8..8+cmd]  command JSON
///   [8+cmd..8+cmd+res] result JSON
const HEADER_SIZE: usize = 8;

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
}

unsafe impl Send for EpChannel {}
unsafe impl Sync for EpChannel {}
unsafe impl Send for EpWorker {}

impl EpChannel {
    /// Create shared memory + semaphores for `world_size` workers.
    /// Must be called BEFORE forking workers.
    pub fn new(world_size: usize) -> io::Result<Self> {
        let shm_name = format!("/rustrain-ep-{}", std::process::id());
        let c_name = CString::new(shm_name.as_str()).unwrap();

        // Create shared memory
        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        if unsafe { libc::ftruncate(fd, SHM_SIZE as i64) } < 0 {
            let e = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            unsafe { libc::shm_unlink(c_name.as_ptr()) };
            return Err(e);
        }
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                SHM_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let e = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            unsafe { libc::shm_unlink(c_name.as_ptr()) };
            return Err(e);
        }

        // Zero the shared memory
        unsafe { ptr::write_bytes(ptr as *mut u8, 0, SHM_SIZE) };

        // Create semaphores (unnamed, in shared memory)
        // We allocate them at fixed offsets after the header
        let sem_size = std::mem::size_of::<libc::sem_t>();
        let sem_region_start = HEADER_SIZE + 64 * 1024; // leave 64KB for data
        let mut sems_request = Vec::with_capacity(world_size);
        let mut sems_done = Vec::with_capacity(world_size);

        for i in 0..world_size {
            let offset_req = sem_region_start + i * 2 * sem_size;
            let offset_done = sem_region_start + (i * 2 + 1) * sem_size;

            let sem_req = unsafe {
                libc::sem_init(
                    (ptr as *mut u8).add(offset_req) as *mut libc::sem_t,
                    1, // shared between processes
                    0, // initial value 0 (workers wait)
                )
            };
            if sem_req != 0 {
                return Err(io::Error::last_os_error());
            }

            let sem_done = unsafe {
                libc::sem_init(
                    (ptr as *mut u8).add(offset_done) as *mut libc::sem_t,
                    1,
                    0,
                )
            };
            if sem_done != 0 {
                return Err(io::Error::last_os_error());
            }

            sems_request.push(unsafe {
                (ptr as *mut u8).add(offset_req) as *mut libc::sem_t
            });
            sems_done.push(unsafe {
                (ptr as *mut u8).add(offset_done) as *mut libc::sem_t
            });
        }

        tracing::info!(
            "EP IPC: created shared memory '{}' ({}KB), {} workers",
            shm_name,
            SHM_SIZE / 1024,
            world_size
        );

        Ok(Self {
            shm_ptr: ptr as *mut u8,
            shm_size: SHM_SIZE,
            shm_fd: fd,
            sem_request: sems_request,
            sem_done: sems_done,
            world_size,
            shm_name,
        })
    }

    /// Get the shm name (workers need this to attach).
    pub fn shm_name(&self) -> &str {
        &self.shm_name
    }

    /// World size.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Send a command to ALL workers simultaneously, then wait for ALL to complete.
    /// Returns the result from rank 0 (all ranks should have identical results for EP).
    pub fn broadcast(&self, cmd: &EpCommand) -> io::Result<EpResult> {
        // Serialize command
        let json = serde_json::to_vec(cmd).map_err(|e| io::Error::other(e.to_string()))?;
        if json.len() + HEADER_SIZE > 64 * 1024 {
            return Err(io::Error::other(format!(
                "command too large: {} bytes",
                json.len()
            )));
        }

        // Write command to shared memory
        unsafe {
            let len = json.len() as u32;
            ptr::copy_nonoverlapping(
                &len as *const u32 as *const u8,
                self.shm_ptr,
                4,
            );
            ptr::copy_nonoverlapping(
                json.as_ptr(),
                self.shm_ptr.add(4),
                json.len(),
            );
        }

        // Signal all workers simultaneously
        for i in 0..self.world_size {
            let r = unsafe { libc::sem_post(self.sem_request[i]) };
            if r != 0 {
                return Err(io::Error::last_os_error());
            }
        }

        // Wait for all workers to complete
        for i in 0..self.world_size {
            let r = unsafe { libc::sem_wait(self.sem_done[i]) };
            if r != 0 {
                return Err(io::Error::last_os_error());
            }
        }

        // Read result from shared memory (rank 0's result)
        let result = unsafe {
            let result_len = *(self.shm_ptr.add(4) as *const u32) as usize;
            if result_len == 0 || result_len > 64 * 1024 {
                return Err(io::Error::other("worker returned empty result"));
            }
            let result_bytes =
                std::slice::from_raw_parts(self.shm_ptr.add(HEADER_SIZE), result_len);
            serde_json::from_slice::<EpResult>(result_bytes)
                .map_err(|e| io::Error::other(format!("result deserialization: {e}")))?
        };

        Ok(result)
    }
}

impl Drop for EpChannel {
    fn drop(&mut self) {
        // Destroy semaphores
        for i in 0..self.world_size {
            unsafe {
                libc::sem_destroy(self.sem_request[i]);
                libc::sem_destroy(self.sem_done[i]);
            }
        }
        // Unmap shared memory
        unsafe {
            libc::munmap(self.shm_ptr as *mut libc::c_void, self.shm_size);
            libc::close(self.shm_fd);
            let c_name = CString::new(self.shm_name.as_str()).unwrap();
            libc::shm_unlink(c_name.as_ptr());
        }
        tracing::debug!("EP IPC: cleaned up shared memory '{}'", self.shm_name);
    }
}

impl EpWorker {
    /// Attach to an existing shared memory segment (created by parent).
    pub fn attach(shm_name: &str, rank: usize, world_size: usize) -> io::Result<Self> {
        let c_name = CString::new(shm_name).unwrap();

        // Open existing shared memory
        let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0o600) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                SHM_SIZE,
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

        // Locate semaphores
        let sem_size = std::mem::size_of::<libc::sem_t>();
        let sem_region_start = HEADER_SIZE + 64 * 1024;
        let offset_req = sem_region_start + rank * 2 * sem_size;
        let offset_done = sem_region_start + (rank * 2 + 1) * sem_size;

        let sem_request =
            unsafe { (ptr as *mut u8).add(offset_req) as *mut libc::sem_t };
        let sem_done =
            unsafe { (ptr as *mut u8).add(offset_done) as *mut libc::sem_t };

        tracing::info!("EP worker {}: attached to shared memory '{}'", rank, shm_name);

        Ok(Self {
            shm_ptr: ptr as *mut u8,
            shm_size: SHM_SIZE,
            sem_request,
            sem_done,
            rank,
        })
    }

    /// Wait for the next command from the parent.
    /// Blocks until parent signals.
    pub fn wait_command(&self) -> io::Result<EpCommand> {
        // Wait for parent to signal
        let r = unsafe { libc::sem_wait(self.sem_request) };
        if r != 0 {
            return Err(io::Error::last_os_error());
        }

        // Read command from shared memory
        let cmd = unsafe {
            let cmd_len = *(self.shm_ptr as *const u32) as usize;
            if cmd_len == 0 || cmd_len > 64 * 1024 {
                return Err(io::Error::other("invalid command length"));
            }
            let cmd_bytes =
                std::slice::from_raw_parts(self.shm_ptr.add(4), cmd_len);
            serde_json::from_slice::<EpCommand>(cmd_bytes)
                .map_err(|e| io::Error::other(format!("command deserialization: {e}")))?
        };

        Ok(cmd)
    }

    /// Signal the parent that this worker is done, with the given result.
    pub fn signal_done(&self, result: &EpResult) -> io::Result<()> {
        // Write result to shared memory
        let json = serde_json::to_vec(result).map_err(|e| io::Error::other(e.to_string()))?;
        if json.len() + HEADER_SIZE > 64 * 1024 {
            return Err(io::Error::other(format!(
                "result too large: {} bytes",
                json.len()
            )));
        }

        unsafe {
            let len = json.len() as u32;
            ptr::copy_nonoverlapping(
                &len as *const u32 as *const u8,
                self.shm_ptr.add(4),
                4,
            );
            ptr::copy_nonoverlapping(
                json.as_ptr(),
                self.shm_ptr.add(HEADER_SIZE),
                json.len(),
            );
        }

        // Signal parent
        let r = unsafe { libc::sem_post(self.sem_done) };
        if r != 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(())
    }

    /// Worker rank.
    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl Drop for EpWorker {
    fn drop(&mut self) {
        // Don't destroy semaphores here — parent owns them
        unsafe {
            libc::munmap(self.shm_ptr as *mut libc::c_void, self.shm_size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_create_destroy() {
        // Just verify we can create and drop without panic
        let ch = EpChannel::new(2).expect("create channel");
        assert_eq!(ch.world_size(), 2);
        drop(ch);
    }
}
