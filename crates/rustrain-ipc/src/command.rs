use serde::{Deserialize, Serialize};

pub const TENSOR_SPAN_ALIGNMENT: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSpan {
    pub offset_bytes: u64,
    pub len_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSlabRef {
    pub input_ids: TensorSpan,
    pub target_mask: TensorSpan,
    pub attention_mask: TensorSpan,
    pub batch_size: usize,
    pub seq_len: usize,
}

impl TensorSlabRef {
    pub fn spans(&self) -> [TensorSpan; 3] {
        [self.input_ids, self.target_mask, self.attention_mask]
    }

    pub fn validate(&self, payload_len: usize) -> Result<(), String> {
        if self.batch_size == 0 || self.seq_len == 0 {
            return Err(format!(
                "tensor slab batch_size and seq_len must be positive, got batch_size={} seq_len={}",
                self.batch_size, self.seq_len
            ));
        }
        let expected_bytes = self
            .batch_size
            .checked_mul(self.seq_len)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<i64>()))
            .ok_or_else(|| "tensor slab shape byte count overflowed usize".to_string())?;
        let mut ranges = Vec::with_capacity(3);
        for (name, span) in [
            ("input_ids", self.input_ids),
            ("target_mask", self.target_mask),
            ("attention_mask", self.attention_mask),
        ] {
            let offset = usize::try_from(span.offset_bytes)
                .map_err(|_| format!("{name} slab offset does not fit usize"))?;
            let len = usize::try_from(span.len_bytes)
                .map_err(|_| format!("{name} slab length does not fit usize"))?;
            if offset % TENSOR_SPAN_ALIGNMENT != 0 {
                return Err(format!(
                    "{name} slab offset {offset} is not {TENSOR_SPAN_ALIGNMENT}-byte aligned"
                ));
            }
            if len != expected_bytes {
                return Err(format!(
                    "{name} slab length {len} does not match batch_size={} seq_len={} ({expected_bytes} bytes)",
                    self.batch_size, self.seq_len
                ));
            }
            let end = offset
                .checked_add(len)
                .ok_or_else(|| format!("{name} slab range overflowed usize"))?;
            if end > payload_len {
                return Err(format!(
                    "{name} slab range {offset}..{end} exceeds payload length {payload_len}"
                ));
            }
            ranges.push((offset, end, name));
        }
        ranges.sort_unstable_by_key(|range| range.0);
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(format!(
                    "tensor slab spans {} and {} overlap",
                    pair[0].2, pair[1].2
                ));
            }
        }
        Ok(())
    }
}

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
    TrainStepSlab {
        session_id: String,
        tensors: TensorSlabRef,
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
    TrainMultiLoraSlab {
        session_id: String,
        tensors: TensorSlabRef,
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
    EvalStepSlab {
        session_id: String,
        tensors: TensorSlabRef,
    },
    ExportAdapter {
        session_id: String,
        path: String,
        adapter_id: Option<i64>,
        generation: String,
    },
    PrepareSaveCheckpoint {
        session_id: String,
        path: String,
        generation: String,
    },
    PrepareLoadCheckpoint {
        session_id: String,
        path: String,
        transaction_id: String,
    },
    CommitLoadCheckpoint {
        session_id: String,
        transaction_id: String,
    },
    AbortLoadCheckpoint {
        session_id: String,
        transaction_id: String,
    },
    Status {
        session_id: String,
    },
    Shutdown,
}

impl EpCommand {
    pub fn tensor_slab(&self) -> Option<&TensorSlabRef> {
        match self {
            Self::TrainStepSlab { tensors, .. }
            | Self::TrainMultiLoraSlab { tensors, .. }
            | Self::EvalStepSlab { tensors, .. } => Some(tensors),
            _ => None,
        }
    }
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
    Checkpoint {
        step: u64,
        loss: f64,
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
    use super::{EpCommand, EpResult, TensorSlabRef, TensorSpan};

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

    #[test]
    fn checkpoint_commands_preserve_transaction_identity() {
        let commands = [
            EpCommand::PrepareSaveCheckpoint {
                session_id: "session".into(),
                path: "/tmp/checkpoint.partial".into(),
                generation: "ep-1-2-3-save".into(),
            },
            EpCommand::PrepareLoadCheckpoint {
                session_id: "session".into(),
                path: "/tmp/checkpoint".into(),
                transaction_id: "ep-1-2-4-load".into(),
            },
            EpCommand::CommitLoadCheckpoint {
                session_id: "session".into(),
                transaction_id: "ep-1-2-4-load".into(),
            },
            EpCommand::AbortLoadCheckpoint {
                session_id: "session".into(),
                transaction_id: "ep-1-2-4-load".into(),
            },
        ];

        for command in commands {
            let encoded = serde_json::to_string(&command).unwrap();
            let decoded: EpCommand = serde_json::from_str(&encoded).unwrap();
            assert_eq!(
                serde_json::to_value(decoded).unwrap(),
                serde_json::to_value(command).unwrap()
            );
        }
    }

    #[test]
    fn checkpoint_result_serde_preserves_step_and_loss() {
        let result = EpResult::Checkpoint {
            step: 19,
            loss: 0.625,
        };
        let encoded = serde_json::to_string(&result).unwrap();
        let decoded: EpResult = serde_json::from_str(&encoded).unwrap();
        match decoded {
            EpResult::Checkpoint { step, loss } => {
                assert_eq!(step, 19);
                assert_eq!(loss, 0.625);
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn tensor_slab_command_serde_preserves_spans() {
        let tensors = TensorSlabRef {
            input_ids: TensorSpan {
                offset_bytes: 0,
                len_bytes: 32,
            },
            target_mask: TensorSpan {
                offset_bytes: 64,
                len_bytes: 32,
            },
            attention_mask: TensorSpan {
                offset_bytes: 128,
                len_bytes: 32,
            },
            batch_size: 2,
            seq_len: 2,
        };
        let command = EpCommand::TrainStepSlab {
            session_id: "session".into(),
            tensors: tensors.clone(),
        };
        let encoded = serde_json::to_vec(&command).unwrap();
        let decoded: EpCommand = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.tensor_slab(), Some(&tensors));
        tensors.validate(160).unwrap();
    }

    #[test]
    fn tensor_slab_validation_rejects_overlapping_spans() {
        let tensors = TensorSlabRef {
            input_ids: TensorSpan {
                offset_bytes: 0,
                len_bytes: 16,
            },
            target_mask: TensorSpan {
                offset_bytes: 0,
                len_bytes: 16,
            },
            attention_mask: TensorSpan {
                offset_bytes: 128,
                len_bytes: 16,
            },
            batch_size: 1,
            seq_len: 2,
        };
        assert!(tensors.validate(144).unwrap_err().contains("overlap"));
    }
}
