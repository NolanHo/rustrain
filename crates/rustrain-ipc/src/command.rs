use serde::{Deserialize, Serialize};

/// Commands that the HTTP server can send to workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EpCommand {
    CreateSession { session_id: String },
    DeleteSession { session_id: String },
    LoadModel {
        session_id: String,
        model_path: String,
        config_toml: String,
    },
    LoadDataset {
        session_id: String,
        jsonl_path: String,
        seq_len: usize,
    },
    InitLora {
        session_id: String,
        rank: i64,
        alpha: i64,
        target_layers: Vec<usize>,
        target_modules: Vec<String>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
    },
    AddLora {
        session_id: String,
        rank: i64,
        alpha: f64,
        target_layers: Vec<i64>,
        target_modules: String,
    },
    BatchAddLora {
        session_id: String,
        count: i32,
        rank: i64,
        alpha: f64,
        target_layers: Vec<i64>,
        target_modules: String,
    },
    RemoveLora {
        session_id: String,
        adapter_id: i64,
    },
    ListLora { session_id: String },
    TrainStep {
        session_id: String,
        input_ids: Vec<i64>,
        target_mask: Vec<i64>,
        attention_mask: Vec<i64>,
        seq_len: usize,
    },
    TrainMultiLora {
        session_id: String,
        input_ids: Vec<i64>,
        target_mask: Vec<i64>,
        attention_mask: Vec<i64>,
        seq_len: usize,
        n_total: i32,
        lora_rank: i32,
    },
    EvalStep {
        session_id: String,
        input_ids: Vec<i64>,
        target_mask: Vec<i64>,
        attention_mask: Vec<i64>,
        seq_len: usize,
    },
    ExportAdapter {
        session_id: String,
        path: String,
    },
    Status { session_id: String },
    Shutdown,
}

/// Results that workers return to the HTTP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EpResult {
    Ok,
    Loss(f64),
    AdapterId(i64),
    AdapterIds(Vec<i64>),
    Count(usize),
    Status {
        state: String,
        step: u64,
        last_loss: f64,
        model_path: String,
    },
    Error(String),
}

impl EpResult {
    pub fn ok() -> Self {
        EpResult::Ok
    }

    pub fn error(msg: impl Into<String>) -> Self {
        EpResult::Error(msg.into())
    }
}
