//! Session manager — holds training sessions by ID.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tch::{Device, Kind};
use tokio::sync::Mutex;

use crate::session::{Qwen36Session, SessionState, TrainingSession};

pub struct SessionManager {
    sessions: Mutex<HashMap<String, Arc<Mutex<Box<dyn TrainingSession>>>>>,
    metrics_dir: PathBuf,
}

impl SessionManager {
    pub fn new(metrics_dir: PathBuf) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            metrics_dir,
        }
    }

    pub async fn create_session(&self, session_id: String) -> Result<(), String> {
        let mut sessions = self.sessions.lock().await;
        if sessions.contains_key(&session_id) {
            return Err(format!("session {session_id} already exists"));
        }
        let metrics_path = self.metrics_dir.join(format!("{session_id}_metrics.jsonl"));
        // For EP: use LOCAL_RANK to select GPU; default to GPU 0
        let local_rank = std::env::var("LOCAL_RANK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let session = Qwen36Session::new(Device::Cuda(local_rank), Kind::BFloat16, metrics_path);
        sessions.insert(session_id, Arc::new(Mutex::new(Box::new(session))));
        Ok(())
    }

    pub async fn get_session(
        &self,
        session_id: &str,
    ) -> Option<Arc<Mutex<Box<dyn TrainingSession>>>> {
        self.sessions.lock().await.get(session_id).cloned()
    }

    pub async fn list_sessions(&self) -> Vec<String> {
        self.sessions.lock().await.keys().cloned().collect()
    }
}
