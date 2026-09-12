use std::{fs, path::PathBuf, process::Command};

pub fn memory_rss_mb() -> Option<f64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kb = line
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<f64>().ok())?;
    Some(kb / 1024.0)
}

pub fn gpu_memory_allocated_mb() -> Option<f64> {
    gpu_memory_for_current_pid_mb().or_else(gpu_memory_for_visible_device_mb)
}

fn gpu_memory_for_current_pid_mb() -> Option<f64> {
    let pid = std::process::id().to_string();
    let output = Command::new(nvidia_smi_command()?)
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    stdout
        .lines()
        .filter_map(|line| {
            let mut fields = line.split(',').map(str::trim);
            let line_pid = fields.next()?;
            let memory_mb = fields.next()?.parse::<f64>().ok()?;
            (line_pid == pid).then_some(memory_mb)
        })
        .reduce(f64::max)
}

fn gpu_memory_for_visible_device_mb() -> Option<f64> {
    let visible_device = std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .and_then(|value| value.split(',').next().map(str::trim).map(str::to_string))
        .filter(|value| !value.is_empty())?;
    let output = Command::new(nvidia_smi_command()?)
        .args([
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    stdout.lines().find_map(|line| {
        let mut fields = line.split(',').map(str::trim);
        let index = fields.next()?;
        let memory_mb = fields.next()?.parse::<f64>().ok()?;
        (index == visible_device).then_some(memory_mb)
    })
}

fn nvidia_smi_command() -> Option<PathBuf> {
    if let Some(command) = std::env::var_os("RUSTRAIN_NVIDIA_SMI").map(PathBuf::from) {
        return command.exists().then_some(command);
    }
    [
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/usr/local/cuda/bin/nvidia-smi",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.exists())
    .or_else(|| find_command_in_path("nvidia-smi"))
}

fn find_command_in_path(command: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var)
        .map(|path| path.join(command))
        .find(|path| path.exists())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn memory_rss_metric_is_optional_but_positive_when_available() {
        if let Some(memory_rss_mb) = memory_rss_mb() {
            assert!(memory_rss_mb > 0.0);
        }
    }

    #[test]
    fn gpu_memory_metric_is_optional() {
        assert!(gpu_memory_allocated_mb().is_none_or(|value| value >= 0.0));
    }

    #[test]
    fn nvidia_smi_command_honors_existing_env_override() {
        let _guard = env_lock().lock().expect("env lock should not be poisoned");
        let temp_dir =
            std::env::temp_dir().join(format!("rustrain-nvidia-smi-test-{}", std::process::id()));
        fs::create_dir_all(&temp_dir).expect("temp dir should be created");
        let fake_smi = temp_dir.join("nvidia-smi");
        fs::write(&fake_smi, "").expect("fake nvidia-smi should be written");
        let previous = std::env::var_os("RUSTRAIN_NVIDIA_SMI");
        unsafe {
            std::env::set_var("RUSTRAIN_NVIDIA_SMI", &fake_smi);
        }
        assert_eq!(nvidia_smi_command(), Some(fake_smi));
        unsafe {
            if let Some(previous) = previous {
                std::env::set_var("RUSTRAIN_NVIDIA_SMI", previous);
            } else {
                std::env::remove_var("RUSTRAIN_NVIDIA_SMI");
            }
        }
        fs::remove_dir_all(temp_dir).expect("temp dir should be removed");
    }
}
