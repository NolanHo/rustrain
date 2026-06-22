use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::runtime::DataConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteTokenizer {
    vocab_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizedDataset {
    pub tokenizer: ByteTokenizer,
    pub train_tokens: Vec<usize>,
    pub eval_tokens: Vec<usize>,
    pub train_sequences: Vec<Vec<usize>>,
    pub eval_sequences: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftDataset {
    pub tokenizer: ByteTokenizer,
    pub train_samples: Vec<SftSample>,
    pub eval_samples: Vec<SftSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftSample {
    pub tokens: Vec<usize>,
    pub target_mask: Vec<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct InstructionRecord {
    instruction: String,
    #[serde(default)]
    input: String,
    response: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenCache {
    source_paths: Vec<PathBuf>,
    vocab_size: usize,
    train_tokens: Vec<usize>,
    eval_tokens: Vec<usize>,
    train_sequences: Vec<Vec<usize>>,
    eval_sequences: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SftCache {
    source_paths: Vec<PathBuf>,
    vocab_size: usize,
    train_samples: Vec<SftSample>,
    eval_samples: Vec<SftSample>,
}

impl ByteTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.bytes()
            .map(|byte| (byte as usize) % self.vocab_size)
            .collect()
    }

    pub fn decode_lossy(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .copied()
            .filter_map(|token| {
                if (32..=126).contains(&token) || token == b'\n' as usize {
                    Some(token as u8 as char)
                } else {
                    None
                }
            })
            .collect()
    }
}

pub fn load_text_dataset(
    data: &DataConfig,
    vocab_size: usize,
    seq_len: usize,
    cache_dir: &Path,
) -> Result<TokenizedDataset> {
    let tokenizer = ByteTokenizer::new(vocab_size);
    let (text, mut source_paths) = read_text_paths(&data.paths)?;

    let tokens = tokenizer.encode(&text);
    if tokens.len() < seq_len * 2 {
        return Err(anyhow!(
            "text dataset needs at least {} tokens, got {}",
            seq_len * 2,
            tokens.len()
        ));
    }

    let (train_tokens, eval_tokens) = if data.eval_paths.is_empty() {
        let split_at = ((tokens.len() as f32) * data.train_split).floor() as usize;
        let split_at = split_at.clamp(seq_len, tokens.len() - seq_len);
        (tokens[..split_at].to_vec(), tokens[split_at..].to_vec())
    } else {
        let (eval_text, eval_source_paths) = read_text_paths(&data.eval_paths)?;
        source_paths.extend(eval_source_paths);
        let eval_tokens = tokenizer.encode(&eval_text);
        if eval_tokens.len() < seq_len {
            return Err(anyhow!(
                "text eval dataset needs at least {} tokens, got {}",
                seq_len,
                eval_tokens.len()
            ));
        }
        if tokens.len() < seq_len {
            return Err(anyhow!(
                "text train dataset needs at least {} tokens, got {}",
                seq_len,
                tokens.len()
            ));
        }
        (tokens, eval_tokens)
    };
    let train_sequences = pack_sequences(&train_tokens, seq_len);
    let eval_sequences = pack_sequences(&eval_tokens, seq_len);

    let dataset = TokenizedDataset {
        tokenizer,
        train_tokens,
        eval_tokens,
        train_sequences,
        eval_sequences,
    };

    write_cache(cache_dir, &source_paths, vocab_size, &dataset)?;

    Ok(dataset)
}

pub fn load_sft_dataset(
    data: &DataConfig,
    vocab_size: usize,
    seq_len: usize,
    cache_dir: &Path,
) -> Result<SftDataset> {
    let tokenizer = ByteTokenizer::new(vocab_size);
    let (mut samples, mut source_paths) = read_sft_paths(&data.paths, &tokenizer, seq_len)?;

    if samples.len() < 2 {
        return Err(anyhow!("SFT dataset needs at least two samples"));
    }
    if let Some(max_samples) = data.max_samples {
        samples.truncate(max_samples);
        if samples.len() < 2 {
            return Err(anyhow!(
                "SFT dataset needs at least two samples after max_samples"
            ));
        }
    }

    let (train_samples, eval_samples) = if data.eval_paths.is_empty() {
        let split_at = ((samples.len() as f32) * data.train_split).floor() as usize;
        let split_at = split_at.clamp(1, samples.len() - 1);
        (samples[..split_at].to_vec(), samples[split_at..].to_vec())
    } else {
        let (eval_samples, eval_source_paths) =
            read_sft_paths(&data.eval_paths, &tokenizer, seq_len)?;
        if eval_samples.is_empty() {
            return Err(anyhow!("SFT eval dataset needs at least one sample"));
        }
        source_paths.extend(eval_source_paths);
        (samples, eval_samples)
    };
    let dataset = SftDataset {
        tokenizer,
        train_samples,
        eval_samples,
    };

    write_sft_cache(cache_dir, &source_paths, vocab_size, &dataset)?;

    Ok(dataset)
}

fn format_sft_sample(
    tokenizer: &ByteTokenizer,
    seq_len: usize,
    record: &InstructionRecord,
) -> Result<SftSample> {
    let prompt = if record.input.trim().is_empty() {
        format!("Instruction:\n{}\n\nResponse:\n", record.instruction)
    } else {
        format!(
            "Instruction:\n{}\n\nInput:\n{}\n\nResponse:\n",
            record.instruction, record.input
        )
    };
    let response = format!("{}\n", record.response);
    let prompt_tokens = tokenizer.encode(&prompt);
    let response_tokens = tokenizer.encode(&response);

    if response_tokens.is_empty() {
        return Err(anyhow!("SFT response must not be empty"));
    }
    if prompt_tokens.len() + response_tokens.len() < 2 {
        return Err(anyhow!("SFT sample must contain at least two tokens"));
    }

    let prompt_len = prompt_tokens.len();
    let mut tokens = prompt_tokens;
    tokens.extend(response_tokens);
    let truncate_start = tokens.len().saturating_sub(seq_len);
    if tokens.len() > seq_len {
        tokens = tokens[truncate_start..].to_vec();
    }

    let response_start = prompt_len.saturating_sub(truncate_start);
    let response_start = response_start.min(tokens.len() - 1);
    let target_mask = (0..tokens.len() - 1)
        .map(|target_index| target_index + 1 >= response_start)
        .collect::<Vec<_>>();

    if !target_mask.iter().any(|enabled| *enabled) {
        return Err(anyhow!("SFT sample response mask is empty"));
    }

    Ok(SftSample {
        tokens,
        target_mask,
    })
}

fn pack_sequences(tokens: &[usize], seq_len: usize) -> Vec<Vec<usize>> {
    tokens
        .windows(seq_len)
        .step_by(seq_len)
        .map(|window| window.to_vec())
        .collect()
}

fn read_text_paths(paths: &[PathBuf]) -> Result<(String, Vec<PathBuf>)> {
    let mut text = String::new();
    let mut source_paths = Vec::new();
    for path in paths {
        if path.is_dir() {
            for file in sorted_files(path)? {
                text.push_str(
                    &fs::read_to_string(&file)
                        .with_context(|| format!("failed to read {}", file.display()))?,
                );
                text.push('\n');
                source_paths.push(file);
            }
        } else {
            text.push_str(
                &fs::read_to_string(path)
                    .with_context(|| format!("failed to read {}", path.display()))?,
            );
            text.push('\n');
            source_paths.push(path.clone());
        }
    }
    Ok((text, source_paths))
}

fn read_sft_paths(
    paths: &[PathBuf],
    tokenizer: &ByteTokenizer,
    seq_len: usize,
) -> Result<(Vec<SftSample>, Vec<PathBuf>)> {
    let mut samples = Vec::new();
    let mut source_paths = Vec::new();
    for path in paths {
        let files = if path.is_dir() {
            sorted_files(path)?
        } else {
            vec![path.clone()]
        };

        for file in files {
            let contents = fs::read_to_string(&file)
                .with_context(|| format!("failed to read {}", file.display()))?;
            for (line_index, line) in contents.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let record: InstructionRecord = serde_json::from_str(line).with_context(|| {
                    format!(
                        "failed to parse JSONL record {}:{}",
                        file.display(),
                        line_index + 1
                    )
                })?;
                samples.push(format_sft_sample(tokenizer, seq_len, &record)?);
            }
            source_paths.push(file);
        }
    }
    Ok((samples, source_paths))
}

fn sorted_files(path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = BTreeSet::new();
    for entry in fs::read_dir(path).with_context(|| format!("failed to list {}", path.display()))? {
        let entry = entry.with_context(|| format!("failed to read entry in {}", path.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", entry.path().display()))?;
        if file_type.is_file() {
            files.insert(entry.path());
        }
    }
    Ok(files.into_iter().collect())
}

fn write_cache(
    cache_dir: &Path,
    source_paths: &[PathBuf],
    vocab_size: usize,
    dataset: &TokenizedDataset,
) -> Result<()> {
    let cache = TokenCache {
        source_paths: source_paths.to_vec(),
        vocab_size,
        train_tokens: dataset.train_tokens.clone(),
        eval_tokens: dataset.eval_tokens.clone(),
        train_sequences: dataset.train_sequences.clone(),
        eval_sequences: dataset.eval_sequences.clone(),
    };
    let contents = toml::to_string(&cache).context("failed to serialize tokenized cache")?;
    fs::write(cache_dir.join("tokenized.toml"), contents).with_context(|| {
        format!(
            "failed to write {}",
            cache_dir.join("tokenized.toml").display()
        )
    })
}

fn write_sft_cache(
    cache_dir: &Path,
    source_paths: &[PathBuf],
    vocab_size: usize,
    dataset: &SftDataset,
) -> Result<()> {
    let cache = SftCache {
        source_paths: source_paths.to_vec(),
        vocab_size,
        train_samples: dataset.train_samples.clone(),
        eval_samples: dataset.eval_samples.clone(),
    };
    let contents = toml::to_string(&cache).context("failed to serialize SFT tokenized cache")?;
    fs::write(cache_dir.join("sft_tokenized.toml"), contents).with_context(|| {
        format!(
            "failed to write {}",
            cache_dir.join("sft_tokenized.toml").display()
        )
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;
    use crate::runtime::{DataConfig, DataKind};

    #[test]
    fn text_dataset_is_packed_and_cached() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("corpus");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("corpus dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        let mut file = fs::File::create(data_dir.join("sample.txt")).expect("file should open");
        writeln!(
            file,
            "rustrain packs causal language model text into fixed windows"
        )
        .expect("file should write");

        let data = DataConfig {
            kind: DataKind::Text,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.75,
            max_samples: None,
            shuffle: true,
            index_cache: None,
        };
        let dataset = load_text_dataset(&data, 64, 8, &cache_dir).expect("dataset should load");

        assert!(!dataset.train_sequences.is_empty());
        assert!(!dataset.eval_sequences.is_empty());
        assert_eq!(dataset.train_sequences[0].len(), 8);
        assert!(cache_dir.join("tokenized.toml").exists());
    }

    #[test]
    fn text_dataset_uses_explicit_eval_paths() {
        let dir = tempdir().expect("temp dir should be created");
        let train_dir = dir.path().join("train");
        let eval_dir = dir.path().join("eval");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&train_dir).expect("train dir should be created");
        fs::create_dir_all(&eval_dir).expect("eval dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            train_dir.join("train.txt"),
            "aaaaaaaa bbbbbbbb cccccccc dddddddd eeeeeeee\n",
        )
        .expect("train file should write");
        fs::write(eval_dir.join("eval.txt"), "zzzzzzzz yyyyyyyy xxxxxxxx\n")
            .expect("eval file should write");

        let data = DataConfig {
            kind: DataKind::Text,
            paths: vec![train_dir],
            eval_paths: vec![eval_dir],
            train_split: 0.5,
            max_samples: None,
            shuffle: true,
            index_cache: None,
        };
        let dataset = load_text_dataset(&data, 128, 8, &cache_dir).expect("dataset should load");

        assert!(
            dataset
                .tokenizer
                .decode_lossy(&dataset.train_sequences[0])
                .contains("a")
        );
        assert!(
            dataset
                .tokenizer
                .decode_lossy(&dataset.eval_sequences[0])
                .contains("z")
        );
        assert!(cache_dir.join("tokenized.toml").exists());
    }

    #[test]
    fn sft_dataset_builds_response_only_masks_and_cache() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        let mut file = fs::File::create(data_dir.join("sample.jsonl")).expect("file should open");
        writeln!(
            file,
            "{{\"instruction\":\"say hi\",\"response\":\"hi\"}}\n{{\"instruction\":\"say bye\",\"input\":\"politely\",\"response\":\"bye\"}}"
        )
        .expect("file should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            shuffle: true,
            index_cache: None,
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(
            dataset.train_samples[0]
                .target_mask
                .iter()
                .any(|enabled| *enabled)
        );
        assert!(
            dataset.train_samples[0]
                .target_mask
                .iter()
                .any(|enabled| !*enabled)
        );
        let response_targets = dataset.train_samples[0]
            .target_mask
            .iter()
            .filter(|enabled| **enabled)
            .count();
        assert!(response_targets > 2);
        assert!(cache_dir.join("sft_tokenized.toml").exists());
    }

    #[test]
    fn sft_dataset_uses_explicit_eval_paths() {
        let dir = tempdir().expect("temp dir should be created");
        let train_dir = dir.path().join("train_sft");
        let eval_dir = dir.path().join("eval_sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&train_dir).expect("train SFT dir should be created");
        fs::create_dir_all(&eval_dir).expect("eval SFT dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            train_dir.join("train.jsonl"),
            "{\"instruction\":\"train one\",\"response\":\"alpha\"}\n{\"instruction\":\"train two\",\"response\":\"beta\"}\n{\"instruction\":\"train three\",\"response\":\"gamma\"}\n",
        )
        .expect("train jsonl should write");
        fs::write(
            eval_dir.join("eval.jsonl"),
            "{\"instruction\":\"eval one\",\"response\":\"delta\"}\n",
        )
        .expect("eval jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![train_dir],
            eval_paths: vec![eval_dir],
            train_split: 0.34,
            max_samples: None,
            shuffle: true,
            index_cache: None,
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 3);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(cache_dir.join("sft_tokenized.toml").exists());
    }

    #[test]
    fn sft_dataset_respects_max_samples_before_split() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        let mut file = fs::File::create(data_dir.join("sample.jsonl")).expect("file should open");
        writeln!(
            file,
            "{{\"instruction\":\"one\",\"response\":\"1\"}}\n{{\"instruction\":\"two\",\"response\":\"2\"}}\n{{\"instruction\":\"three\",\"response\":\"3\"}}\n{{\"instruction\":\"four\",\"response\":\"4\"}}"
        )
        .expect("file should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: Some(2),
            shuffle: true,
            index_cache: None,
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
    }
}
