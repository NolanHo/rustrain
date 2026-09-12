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

    /// Get the pad token ID used for padding.
    pub fn pad_token_id(&self) -> i64 {
        self.tokenizer.token_to_id("")
            .unwrap_or(0) as i64
    }

    /// Encode a single example using Qwen3.6 chat template format.
    /// Format: <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n\n\n{response}<|im_end|>\n
    pub fn encode(&self, example: &SftExample) -> (Vec<i64>, Vec<f32>) {
        // Build prompt: <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n
        let prompt_text = match &example.system {
            Some(sys) => format!(
                "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{instruction} {input}<|im_end|>\n<|im_start|>assistant\n",
                instruction = example.instruction, input = example.input
            ),
            None => format!(
                "<|im_start|>user\n{instruction} {input}<|im_end|>\n<|im_start|>assistant\n",
                instruction = example.instruction, input = example.input
            ),
        };

        let prompt_encoding = self.tokenizer.encode(prompt_text.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))
            .unwrap();
        let prompt_ids = prompt_encoding.get_ids();

        // Qwen3.6 chat template inserts think + \n\n + answer + \n\n before response
        let think_id = self.tokenizer.token_to_id("").map(|t| t as i64).unwrap_or(248068);
        let answer_id = self.tokenizer.token_to_id("").map(|t| t as i64).unwrap_or(248069);
        let newline_id = self.tokenizer.token_to_id("\n\n").map(|t| t as i64).unwrap_or(271);

        // Response tokens
        let response_encoding = self.tokenizer.encode(example.response.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenizer error: {e}"))
            .unwrap();
        let response_ids = response_encoding.get_ids();

        let eos_id = self.tokenizer.token_to_id("<|im_end|>")
            .unwrap_or(0) as i64;

        // Build full sequence: prompt (mask=0) + think + \n\n + answer + \n\n + response (mask=1) + eos (mask=1)
        let mut token_ids = Vec::with_capacity(prompt_ids.len() + 4 + response_ids.len() + 1);
        let mut mask = Vec::with_capacity(token_ids.capacity());

        // Prompt (no loss)
        token_ids.extend(prompt_ids.iter().map(|&id| id as i64));
        mask.extend(std::iter::repeat(0.0_f32).take(prompt_ids.len()));

        // think + \n\n (no loss — part of template)
        token_ids.push(think_id);
        mask.push(0.0);
        token_ids.push(newline_id);
        mask.push(0.0);

        // answer + \n\n (loss starts here — model should predict answer token)
        token_ids.push(answer_id);
        mask.push(1.0);
        token_ids.push(newline_id);
        mask.push(1.0);

        // Response content (loss)
        token_ids.extend(response_ids.iter().map(|&id| id as i64));
        mask.extend(std::iter::repeat(1.0_f32).take(response_ids.len()));

        // EOS (loss)
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
