//! Metrics sink trait + file-based implementation.

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepMetric {
    pub step: u64,
    pub loss: f64,
    pub lr: f64,
    pub mem_gb: f64,
    pub timestamp_unix: i64,
}

pub trait MetricsSink: Send + Sync {
    fn record_step(&self, metric: StepMetric);
    fn read_metrics(&self) -> Vec<StepMetric>;
}

/// File-based metrics sink. Appends one JSON per line (JSONL format).
pub struct FileMetricsSink {
    path: PathBuf,
    metrics: Arc<RwLock<Vec<StepMetric>>>,
}

impl FileMetricsSink {
    pub fn new(path: PathBuf) -> Self {
        let metrics = Arc::new(RwLock::new(Vec::new()));
        // Load existing metrics if file exists
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                let loaded: Vec<StepMetric> = content
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .filter_map(|l| serde_json::from_str(l).ok())
                    .collect();
                if let Ok(mut guard) = metrics.try_write() {
                    *guard = loaded;
                }
            }
        }
        Self { path, metrics }
    }
}

impl MetricsSink for FileMetricsSink {
    fn record_step(&self, metric: StepMetric) {
        // Write to in-memory buffer
        if let Ok(mut guard) = self.metrics.try_write() {
            guard.push(metric.clone());
        }
        // Append to file
        if let Ok(line) = serde_json::to_string(&metric) {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
            {
                let _ = writeln!(f, "{line}");
            }
        }
    }

    fn read_metrics(&self) -> Vec<StepMetric> {
        self.metrics
            .try_read()
            .map(|g| g.clone())
            .unwrap_or_default()
    }
}
