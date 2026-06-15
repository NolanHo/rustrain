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
struct TokenCache {
    source_paths: Vec<PathBuf>,
    vocab_size: usize,
    train_tokens: Vec<usize>,
    eval_tokens: Vec<usize>,
    train_sequences: Vec<Vec<usize>>,
    eval_sequences: Vec<Vec<usize>>,
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
}

pub fn load_text_dataset(
    data: &DataConfig,
    vocab_size: usize,
    seq_len: usize,
    cache_dir: &Path,
) -> Result<TokenizedDataset> {
    let tokenizer = ByteTokenizer::new(vocab_size);
    let mut text = String::new();
    let mut source_paths = Vec::new();

    for path in &data.paths {
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

    let tokens = tokenizer.encode(&text);
    if tokens.len() < seq_len * 2 {
        return Err(anyhow!(
            "text dataset needs at least {} tokens, got {}",
            seq_len * 2,
            tokens.len()
        ));
    }

    let split_at = ((tokens.len() as f32) * data.train_split).floor() as usize;
    let split_at = split_at.clamp(seq_len, tokens.len() - seq_len);
    let train_tokens = tokens[..split_at].to_vec();
    let eval_tokens = tokens[split_at..].to_vec();
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

fn pack_sequences(tokens: &[usize], seq_len: usize) -> Vec<Vec<usize>> {
    tokens
        .windows(seq_len)
        .step_by(seq_len)
        .map(|window| window.to_vec())
        .collect()
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
            train_split: 0.75,
        };
        let dataset = load_text_dataset(&data, 64, 8, &cache_dir).expect("dataset should load");

        assert!(!dataset.train_sequences.is_empty());
        assert!(!dataset.eval_sequences.is_empty());
        assert_eq!(dataset.train_sequences[0].len(), 8);
        assert!(cache_dir.join("tokenized.toml").exists());
    }
}
