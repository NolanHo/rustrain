use serde::{Deserialize, Serialize};

/// Commands that the HTTP server can send to workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EpCommand {
    CreateSession {
        session_id: String,
    },
    DeleteSession {
        session_id: String,
    },
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
        alpha: f64,
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
    ListLora {
        session_id: String,
    },
    TrainStep {
        session_id: String,
        input_ids: Vec<i64>,
        target_mask: Vec<i64>,
        attention_mask: Vec<i64>,
        #[serde(default = "default_batch_size")]
        batch_size: usize,
        seq_len: usize,
    },
    TrainMultiLora {
        session_id: String,
        input_ids: Vec<i64>,
        target_mask: Vec<i64>,
        attention_mask: Vec<i64>,
        #[serde(default = "default_batch_size")]
        batch_size: usize,
        seq_len: usize,
        n_total: i32,
        lora_rank: i32,
        #[serde(default)]
        adapter_ids: Vec<i64>,
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
        adapter_id: Option<i64>,
        generation: String,
    },
    Status {
        session_id: String,
    },
    Shutdown,
}

const fn default_batch_size() -> usize {
    1
}

/// Results that workers return to the HTTP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EpResult {
    Ok,
    Loss(f64),
    Train {
        loss: f64,
        step: u64,
    },
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

#[cfg(test)]
mod tests {
    use super::{EpCommand, EpResult};

    #[test]
    fn train_step_serde_preserves_batch_and_sequence_shape() {
        let command = EpCommand::TrainStep {
            session_id: "session".into(),
            input_ids: vec![1, 2, 3, 4, 5, 6],
            target_mask: vec![1; 6],
            attention_mask: vec![1; 6],
            batch_size: 2,
            seq_len: 3,
        };

        let json = serde_json::to_string(&command).unwrap();
        let decoded: EpCommand = serde_json::from_str(&json).unwrap();
        match decoded {
            EpCommand::TrainStep {
                batch_size,
                seq_len,
                ..
            } => {
                assert_eq!(batch_size, 2);
                assert_eq!(seq_len, 3);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn legacy_train_step_command_defaults_to_one_batch_row() {
        let json = r#"{
            "TrainStep": {
                "session_id": "session",
                "input_ids": [1, 2, 3],
                "target_mask": [1, 1, 1],
                "attention_mask": [1, 1, 1],
                "seq_len": 3
            }
        }"#;

        let decoded: EpCommand = serde_json::from_str(json).unwrap();
        match decoded {
            EpCommand::TrainStep { batch_size, .. } => assert_eq!(batch_size, 1),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn train_result_serde_preserves_logical_step() {
        let result = EpResult::Train {
            loss: 1.25,
            step: 7,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: EpResult = serde_json::from_str(&json).unwrap();
        match decoded {
            EpResult::Train { loss, step } => {
                assert_eq!(loss, 1.25);
                assert_eq!(step, 7);
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }
}
