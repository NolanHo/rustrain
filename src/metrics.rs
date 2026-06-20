use std::{fs, process::Command};

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
    let output = Command::new("nvidia-smi")
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
    let output = Command::new("nvidia-smi")
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
