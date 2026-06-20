use std::fs;

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
    // tch 0.20 exposes CUDA availability but not allocator memory counters.
    // Keep the metric surface stable so a future NVML or allocator backend can fill it in.
    None
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
