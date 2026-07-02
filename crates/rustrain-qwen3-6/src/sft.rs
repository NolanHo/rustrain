//! Qwen3.6 SFT dataset — JSONL loading, tokenization, response-only masking.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use tokenizers::Tokenizer;
use tracing::info;

#[derive(Debug, Deserialize)]
pub struct SftExample {
    pub instruction: String,
    #[serde(default)]
    pub input: String,
    pub response: String,
    #[serde(default)]
    pub system: Option<String>,
}

pub struct SftBatch {
    pub input_ids: Vec<Vec<i64>>,
    pub target_mask: Vec<Vec<f32>>,
    pub seq_len: usize,
}

impl SftBatch {
    pub fn to_tensors(&self, device: tch::Device, kind: tch::Kind) -> (tch::Tensor, tch::Tensor) {
        let batch_size = self.input_ids.len();
        let seq_len = self.seq_len;

        let flat_ids: Vec<i64> = self.input_ids.iter().flatten().copied().collect();
        let flat_mask: Vec<f32> = self.target_mask.iter().flatten().copied().collect();

        let input_ids = tch::Tensor::from_slice(&flat_ids)
            .view([batch_size as i64, seq_len as i64])
            .to_device(device)
            .to_kind(tch::Kind::Int64);  // token IDs must be Int64, not compute_kind
        let target_mask = tch::Tensor::from_slice(&flat_mask)
            .view([batch_size as i64, seq_len as i64])
            .to_device(device)
            .to_kind(tch::Kind::Float);

        (input_ids, target_mask)
    }
}

pub struct SftDataset {
    pub examples: Vec<SftExample>,
    pub tokenizer: Tokenizer,
    pub seq_len: usize,
}

impl SftDataset {
    pub fn from_jsonl(
        path: &Path,
        tokenizer_path: &Path,
        seq_len: usize,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        let content = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;

        let examples: Vec<SftExample> = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|line| {
                serde_json::from_str(line)
                    .with_context(|| format!("failed to parse JSONL line: {line}"))
            })
            .collect::<Result<Vec<_>>>()?;

        info!("loaded {} SFT examples from {}", examples.len(), path.display());

        Ok(Self { examples, tokenizer, seq_len })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Encode a single example: system + instruction + response → token IDs + target mask.
    pub fn encode(&self, example: &SftExample) -> (Vec<i64>, Vec<f32>) {
        let prompt = match &example.system {
            Some(sys) => format!("{sys}\n\n{instruction}\n{input}", instruction = example.instruction, input = example.input),
            None => format!("{}\n{}", example.instruction, example.input),
        };

        let prompt_encoding = self.tokenizer.encode(prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))
            .unwrap();
        let response_encoding = self.tokenizer.encode(example.response.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))
            .unwrap();

        let prompt_ids = prompt_encoding.get_ids();
        let response_ids = response_encoding.get_ids();

        // Concatenate: prompt (mask=0) + response (mask=1) + EOS (mask=1)
        let mut token_ids = Vec::with_capacity(prompt_ids.len() + response_ids.len() + 1);
        let mut mask = Vec::with_capacity(token_ids.capacity());

        token_ids.extend(prompt_ids.iter().map(|&id| id as i64));
        mask.extend(std::iter::repeat(0.0_f32).take(prompt_ids.len()));

        token_ids.extend(response_ids.iter().map(|&id| id as i64));
        mask.extend(std::iter::repeat(1.0_f32).take(response_ids.len()));

        // Add EOS token
        let eos_id = self.tokenizer.token_to_id("<|im_end|>")
            .unwrap_or(prompt_ids.last().copied().unwrap_or(0)) as i64;
        token_ids.push(eos_id);
        mask.push(1.0);

        // Truncate or pad to seq_len
        let seq_len = self.seq_len;
        if token_ids.len() > seq_len {
            token_ids.truncate(seq_len);
            mask.truncate(seq_len);
        } else {
            let pad_id = self.tokenizer.token_to_id("<|endoftext|>")
                .unwrap_or(0) as i64;
            let pad_count = seq_len - token_ids.len();
            token_ids.extend(std::iter::repeat(pad_id).take(pad_count));
            mask.extend(std::iter::repeat(0.0_f32).take(pad_count));
        }

        (token_ids, mask)
    }

    /// Create a batch of `batch_size` examples starting at `start` index.
    pub fn batch(&self, start: usize, batch_size: usize) -> SftBatch {
        let batch_size = batch_size.min(self.examples.len() - start);
        let mut input_ids = Vec::with_capacity(batch_size);
        let mut target_mask = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let (ids, mask) = self.encode(&self.examples[start + i]);
            input_ids.push(ids);
            target_mask.push(mask);
        }

        SftBatch { input_ids, target_mask, seq_len: self.seq_len }
    }
}
