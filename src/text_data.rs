use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::runtime::{
    DataConfig, FieldAffix, FieldCaseTransform, FieldCaseTransformKind, FieldDefault,
    FieldDefaultTarget, FieldRegexFilter, FieldRegexReplacement, FieldReplacement,
    FieldReplacementTarget, FieldSplit, FieldSplitSide, FieldStrip, FieldTransform,
    FieldTransformOp, FieldTruncation,
};

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

#[derive(Debug, Clone)]
struct InstructionRecord {
    system: String,
    instruction: String,
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
    instruction_field: String,
    input_field: String,
    response_field: String,
    source_instruction_fields: Vec<String>,
    source_input_fields: Vec<String>,
    source_response_fields: Vec<String>,
    system_field: Option<String>,
    chat_messages_field: Option<String>,
    prompt_template: String,
    prompt_with_input_template: String,
    trim_fields: bool,
    min_response_chars: usize,
    max_response_chars: Option<usize>,
    instruction_contains_any: Vec<String>,
    instruction_excludes_any: Vec<String>,
    response_contains_any: Vec<String>,
    response_excludes_any: Vec<String>,
    input_contains_any: Vec<String>,
    input_excludes_any: Vec<String>,
    field_regex_contains_any: Vec<FieldRegexFilter>,
    field_regex_excludes_any: Vec<FieldRegexFilter>,
    field_replacements: Vec<FieldReplacement>,
    field_regex_replacements: Vec<FieldRegexReplacement>,
    normalize_whitespace: bool,
    field_defaults: Vec<FieldDefault>,
    field_case_transforms: Vec<FieldCaseTransform>,
    field_affixes: Vec<FieldAffix>,
    field_strips: Vec<FieldStrip>,
    field_splits: Vec<FieldSplit>,
    field_truncations: Vec<FieldTruncation>,
    field_transforms: Vec<FieldTransform>,
    max_eval_samples: Option<usize>,
    min_system_chars: Option<usize>,
    max_system_chars: Option<usize>,
    system_contains_any: Vec<String>,
    system_excludes_any: Vec<String>,
    min_instruction_chars: Option<usize>,
    max_instruction_chars: Option<usize>,
    min_input_chars: Option<usize>,
    max_input_chars: Option<usize>,
    min_prompt_chars: Option<usize>,
    max_prompt_chars: Option<usize>,
    min_sample_chars: Option<usize>,
    max_sample_chars: Option<usize>,
    dedupe_samples: bool,
    source_weights: Vec<usize>,
    source_max_samples: Vec<usize>,
    skip_invalid_records: bool,
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
    let (mut samples, mut source_paths) = read_sft_paths(
        data,
        &data.paths,
        &tokenizer,
        seq_len,
        true,
        data.max_samples,
    )?;

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
        let (eval_samples, eval_source_paths) = read_sft_paths(
            data,
            &data.eval_paths,
            &tokenizer,
            seq_len,
            false,
            data.max_eval_samples,
        )?;
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

    write_sft_cache(
        cache_dir,
        &source_paths,
        vocab_size,
        &data.instruction_field,
        &data.input_field,
        &data.response_field,
        &data.source_instruction_fields,
        &data.source_input_fields,
        &data.source_response_fields,
        data.system_field.clone(),
        data.chat_messages_field.clone(),
        &data.prompt_template,
        &data.prompt_with_input_template,
        data.trim_fields,
        data.min_response_chars,
        data.max_response_chars,
        &data.instruction_contains_any,
        &data.instruction_excludes_any,
        &data.response_contains_any,
        &data.response_excludes_any,
        &data.input_contains_any,
        &data.input_excludes_any,
        &data.field_regex_contains_any,
        &data.field_regex_excludes_any,
        &data.field_replacements,
        &data.field_regex_replacements,
        data.normalize_whitespace,
        &data.field_defaults,
        &data.field_case_transforms,
        &data.field_affixes,
        &data.field_strips,
        &data.field_splits,
        &data.field_truncations,
        &data.field_transforms,
        data.max_eval_samples,
        data.min_system_chars,
        data.max_system_chars,
        &data.system_contains_any,
        &data.system_excludes_any,
        data.min_instruction_chars,
        data.max_instruction_chars,
        data.min_input_chars,
        data.max_input_chars,
        data.min_prompt_chars,
        data.max_prompt_chars,
        data.min_sample_chars,
        data.max_sample_chars,
        data.dedupe_samples,
        &data.source_weights,
        &data.source_max_samples,
        data.skip_invalid_records,
        &dataset,
    )?;

    Ok(dataset)
}

fn format_sft_sample(
    data: &DataConfig,
    tokenizer: &ByteTokenizer,
    seq_len: usize,
    record: &InstructionRecord,
) -> Result<SftSample> {
    let prompt = render_sft_prompt(
        record,
        &data.prompt_template,
        &data.prompt_with_input_template,
    )?;
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
    data: &DataConfig,
    paths: &[PathBuf],
    tokenizer: &ByteTokenizer,
    seq_len: usize,
    apply_source_weights: bool,
    max_samples: Option<usize>,
) -> Result<(Vec<SftSample>, Vec<PathBuf>)> {
    let mut samples = Vec::new();
    let mut source_paths = Vec::new();
    let mut seen_records = data.dedupe_samples.then(HashSet::new);
    let source_weights = if apply_source_weights {
        sft_source_weights(data, paths.len())?
    } else {
        vec![1; paths.len()]
    };
    let source_max_samples = if apply_source_weights {
        sft_source_max_samples(data, paths.len())?
    } else {
        vec![None; paths.len()]
    };
    for ((path, source_weight), source_limit) in paths
        .iter()
        .zip(source_weights.iter().copied())
        .zip(source_max_samples.iter().copied())
    {
        if max_samples.is_some_and(|limit| samples.len() >= limit) {
            break;
        }
        let mut source_samples = 0usize;
        let files = if path.is_dir() {
            sorted_files(path)?
        } else {
            vec![path.clone()]
        };

        for file in files {
            if max_samples.is_some_and(|limit| samples.len() >= limit)
                || source_limit.is_some_and(|limit| source_samples >= limit)
            {
                break;
            }
            let contents = fs::read_to_string(&file)
                .with_context(|| format!("failed to read {}", file.display()))?;
            for (line_index, line) in contents.lines().enumerate() {
                if max_samples.is_some_and(|limit| samples.len() >= limit)
                    || source_limit.is_some_and(|limit| source_samples >= limit)
                {
                    break;
                }
                if line.trim().is_empty() {
                    continue;
                }
                let Some(record) =
                    maybe_instruction_record_from_jsonl_line(data, line, &file, line_index + 1)?
                else {
                    continue;
                };
                if !sft_record_passes_filters(data, &record)? {
                    continue;
                }
                if let Some(seen_records) = &mut seen_records {
                    if !seen_records.insert(sft_record_dedupe_key(&record)) {
                        continue;
                    }
                }
                let sample = format_sft_sample(data, tokenizer, seq_len, &record)?;
                for _ in 0..source_weight {
                    if max_samples.is_some_and(|limit| samples.len() >= limit) {
                        break;
                    }
                    samples.push(sample.clone());
                }
                source_samples += 1;
            }
            source_paths.push(file);
        }
    }
    Ok((samples, source_paths))
}

fn sft_record_dedupe_key(record: &InstructionRecord) -> String {
    format!(
        "{}\0{}\0{}\0{}",
        record.system, record.instruction, record.input, record.response
    )
}

fn sft_source_weights(data: &DataConfig, path_count: usize) -> Result<Vec<usize>> {
    if data.source_weights.is_empty() {
        return Ok(vec![1; path_count]);
    }
    let weights = if data.source_weights.len() == 1 {
        vec![data.source_weights[0]; path_count]
    } else if data.source_weights.len() == path_count {
        data.source_weights.clone()
    } else {
        return Err(anyhow!(
            "data.source_weights must be empty, length 1, or match data.paths length"
        ));
    };
    if weights.iter().any(|weight| *weight == 0) {
        return Err(anyhow!(
            "data.source_weights entries must be greater than zero"
        ));
    }
    Ok(weights)
}

fn sft_source_max_samples(data: &DataConfig, path_count: usize) -> Result<Vec<Option<usize>>> {
    if data.source_max_samples.is_empty() {
        return Ok(vec![None; path_count]);
    }
    let limits = if data.source_max_samples.len() == 1 {
        vec![Some(data.source_max_samples[0]); path_count]
    } else if data.source_max_samples.len() == path_count {
        data.source_max_samples
            .iter()
            .map(|limit| Some(*limit))
            .collect()
    } else {
        return Err(anyhow!(
            "data.source_max_samples must be empty, length 1, or match data.paths length"
        ));
    };
    if limits.iter().any(|limit| matches!(limit, Some(0))) {
        return Err(anyhow!(
            "data.source_max_samples entries must be greater than zero"
        ));
    }
    Ok(limits)
}

fn instruction_record_from_jsonl_line(data: &DataConfig, line: &str) -> Result<InstructionRecord> {
    let values: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(line).context("invalid JSON object")?;
    if let Some(messages_field) = &data.chat_messages_field {
        let mut record = instruction_record_from_chat_messages(&values, messages_field, data)?;
        apply_field_defaults(&mut record, &data.field_defaults);
        apply_field_replacements(&mut record, &data.field_replacements);
        apply_field_regex_replacements(&mut record, &data.field_regex_replacements)?;
        apply_field_case_transforms(&mut record, &data.field_case_transforms);
        apply_field_affixes(&mut record, &data.field_affixes);
        apply_field_strips(&mut record, &data.field_strips);
        apply_field_splits(&mut record, &data.field_splits);
        apply_field_truncations(&mut record, &data.field_truncations);
        apply_field_transforms(&mut record, &data.field_transforms)?;
        if data.normalize_whitespace {
            normalize_record_whitespace(&mut record);
        }
        return Ok(record);
    }
    let instruction = normalize_jsonl_field(
        defaultable_jsonl_string_field(
            &values,
            &data.instruction_field,
            field_default_value(&data.field_defaults, FieldDefaultTarget::Instruction),
        )?,
        data,
    );
    let system = match &data.system_field {
        Some(field) => normalize_jsonl_field(optional_jsonl_string_field(&values, field)?, data),
        None => String::new(),
    };
    let input = normalize_jsonl_field(
        optional_jsonl_string_field(&values, &data.input_field)?,
        data,
    );
    let response = normalize_jsonl_field(
        defaultable_jsonl_string_field(
            &values,
            &data.response_field,
            field_default_value(&data.field_defaults, FieldDefaultTarget::Response),
        )?,
        data,
    );
    let mut record = InstructionRecord {
        system,
        instruction,
        input,
        response,
    };
    apply_field_defaults(&mut record, &data.field_defaults);
    apply_field_replacements(&mut record, &data.field_replacements);
    apply_field_regex_replacements(&mut record, &data.field_regex_replacements)?;
    apply_field_case_transforms(&mut record, &data.field_case_transforms);
    apply_field_affixes(&mut record, &data.field_affixes);
    apply_field_strips(&mut record, &data.field_strips);
    apply_field_splits(&mut record, &data.field_splits);
    apply_field_truncations(&mut record, &data.field_truncations);
    apply_field_transforms(&mut record, &data.field_transforms)?;
    if data.normalize_whitespace {
        normalize_record_whitespace(&mut record);
    }
    Ok(record)
}

fn instruction_record_from_chat_messages(
    values: &BTreeMap<String, serde_json::Value>,
    messages_field: &str,
    data: &DataConfig,
) -> Result<InstructionRecord> {
    let messages = match jsonl_path_value(values, messages_field) {
        Some(serde_json::Value::Array(messages)) => messages,
        Some(_) => return Err(anyhow!("JSONL field {messages_field} must be an array")),
        None => {
            return Err(anyhow!(
                "JSONL record missing required field {messages_field}"
            ));
        }
    };
    let mut system = String::new();
    let mut instruction = None;
    let mut response = None;
    for (index, message) in messages.iter().enumerate() {
        let object = match message {
            serde_json::Value::Object(object) => object,
            _ => {
                return Err(anyhow!(
                    "chat message {messages_field}[{index}] must be an object"
                ));
            }
        };
        let role = required_chat_message_string_field(object, "role", messages_field, index)?;
        let content = required_chat_message_string_field(object, "content", messages_field, index)?;
        let content = normalize_jsonl_field(content, data);
        match role.as_str() {
            "system" => {
                if system.is_empty() {
                    system = content;
                } else if !content.is_empty() {
                    if !system.is_empty() {
                        system.push('\n');
                    }
                    system.push_str(&content);
                }
            }
            "user" | "human" => instruction = Some(content),
            "assistant" | "gpt" => response = Some(content),
            _ => {}
        }
    }
    let instruction = instruction
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("chat record missing user message in field {messages_field}"))?;
    let response = response.filter(|value| !value.is_empty()).ok_or_else(|| {
        anyhow!("chat record missing assistant message in field {messages_field}")
    })?;
    Ok(InstructionRecord {
        system,
        instruction,
        input: String::new(),
        response,
    })
}

fn maybe_instruction_record_from_jsonl_line(
    data: &DataConfig,
    line: &str,
    file: &Path,
    line_number: usize,
) -> Result<Option<InstructionRecord>> {
    match instruction_record_from_jsonl_line(data, line) {
        Ok(record) => Ok(Some(record)),
        Err(error) if data.skip_invalid_records => {
            tracing::warn!(
                path = %file.display(),
                line = line_number,
                error = %error,
                "skipping invalid SFT JSONL record"
            );
            Ok(None)
        }
        Err(error) => Err(error).with_context(|| {
            format!(
                "failed to parse JSONL record {}:{}",
                file.display(),
                line_number
            )
        }),
    }
}

fn sft_record_passes_filters(data: &DataConfig, record: &InstructionRecord) -> Result<bool> {
    let needs_prompt_chars = data.min_prompt_chars.is_some()
        || data.max_prompt_chars.is_some()
        || data.min_sample_chars.is_some()
        || data.max_sample_chars.is_some();
    let prompt_chars = if needs_prompt_chars {
        Some(
            render_sft_prompt(
                record,
                &data.prompt_template,
                &data.prompt_with_input_template,
            )?
            .chars()
            .count(),
        )
    } else {
        None
    };
    Ok(length_filter_passes(
        record.response.chars().count(),
        Some(data.min_response_chars),
        data.max_response_chars,
    ) && string_contains_any_filter_passes(&record.response, &data.response_contains_any)
        && string_excludes_any_filter_passes(&record.response, &data.response_excludes_any)
        && string_contains_any_filter_passes(&record.instruction, &data.instruction_contains_any)
        && string_excludes_any_filter_passes(&record.instruction, &data.instruction_excludes_any)
        && string_contains_any_filter_passes(&record.input, &data.input_contains_any)
        && string_excludes_any_filter_passes(&record.input, &data.input_excludes_any)
        && string_contains_any_filter_passes(&record.system, &data.system_contains_any)
        && string_excludes_any_filter_passes(&record.system, &data.system_excludes_any)
        && field_regex_contains_any_filter_passes(record, &data.field_regex_contains_any)?
        && field_regex_excludes_any_filter_passes(record, &data.field_regex_excludes_any)?
        && length_filter_passes(
            record.instruction.chars().count(),
            data.min_instruction_chars,
            data.max_instruction_chars,
        )
        && length_filter_passes(
            record.input.chars().count(),
            data.min_input_chars,
            data.max_input_chars,
        )
        && length_filter_passes(
            record.system.chars().count(),
            data.min_system_chars,
            data.max_system_chars,
        )
        && prompt_chars.is_none_or(|chars| {
            length_filter_passes(chars, data.min_prompt_chars, data.max_prompt_chars)
                && length_filter_passes(
                    chars + record.response.chars().count(),
                    data.min_sample_chars,
                    data.max_sample_chars,
                )
        }))
}

fn length_filter_passes(chars: usize, min_chars: Option<usize>, max_chars: Option<usize>) -> bool {
    min_chars.is_none_or(|limit| chars >= limit) && max_chars.is_none_or(|limit| chars <= limit)
}

fn string_contains_any_filter_passes(value: &str, needles: &[String]) -> bool {
    needles.is_empty() || needles.iter().any(|needle| value.contains(needle))
}

fn string_excludes_any_filter_passes(value: &str, needles: &[String]) -> bool {
    needles.iter().all(|needle| !value.contains(needle))
}

fn field_regex_contains_any_filter_passes(
    record: &InstructionRecord,
    filters: &[FieldRegexFilter],
) -> Result<bool> {
    if filters.is_empty() {
        return Ok(true);
    }
    for filter in filters {
        let regex = Regex::new(&filter.pattern).with_context(|| {
            format!(
                "invalid data field regex filter pattern {:?}",
                filter.pattern
            )
        })?;
        if field_regex_filter_matches(record, filter, &regex) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn field_regex_excludes_any_filter_passes(
    record: &InstructionRecord,
    filters: &[FieldRegexFilter],
) -> Result<bool> {
    for filter in filters {
        let regex = Regex::new(&filter.pattern).with_context(|| {
            format!(
                "invalid data field regex filter pattern {:?}",
                filter.pattern
            )
        })?;
        if field_regex_filter_matches(record, filter, &regex) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn field_regex_filter_matches(
    record: &InstructionRecord,
    filter: &FieldRegexFilter,
    regex: &Regex,
) -> bool {
    match filter.field {
        FieldReplacementTarget::System => regex.is_match(&record.system),
        FieldReplacementTarget::Instruction => regex.is_match(&record.instruction),
        FieldReplacementTarget::Input => regex.is_match(&record.input),
        FieldReplacementTarget::Response => regex.is_match(&record.response),
        FieldReplacementTarget::All => {
            regex.is_match(&record.system)
                || regex.is_match(&record.instruction)
                || regex.is_match(&record.input)
                || regex.is_match(&record.response)
        }
    }
}

fn normalize_jsonl_field(value: String, data: &DataConfig) -> String {
    if data.trim_fields {
        value.trim().to_string()
    } else {
        value
    }
}

fn apply_field_replacements(record: &mut InstructionRecord, replacements: &[FieldReplacement]) {
    for replacement in replacements {
        if matches!(
            replacement.field,
            FieldReplacementTarget::System | FieldReplacementTarget::All
        ) {
            record.system = record
                .system
                .replace(&replacement.pattern, &replacement.replacement);
        }
        if matches!(
            replacement.field,
            FieldReplacementTarget::Instruction | FieldReplacementTarget::All
        ) {
            record.instruction = record
                .instruction
                .replace(&replacement.pattern, &replacement.replacement);
        }
        if matches!(
            replacement.field,
            FieldReplacementTarget::Input | FieldReplacementTarget::All
        ) {
            record.input = record
                .input
                .replace(&replacement.pattern, &replacement.replacement);
        }
        if matches!(
            replacement.field,
            FieldReplacementTarget::Response | FieldReplacementTarget::All
        ) {
            record.response = record
                .response
                .replace(&replacement.pattern, &replacement.replacement);
        }
    }
}

fn apply_field_regex_replacements(
    record: &mut InstructionRecord,
    replacements: &[FieldRegexReplacement],
) -> Result<()> {
    for replacement in replacements {
        let regex = Regex::new(&replacement.pattern).with_context(|| {
            format!(
                "invalid data.field_regex_replacements pattern {:?}",
                replacement.pattern
            )
        })?;
        if matches!(
            replacement.field,
            FieldReplacementTarget::System | FieldReplacementTarget::All
        ) {
            record.system = regex
                .replace_all(&record.system, replacement.replacement.as_str())
                .into_owned();
        }
        if matches!(
            replacement.field,
            FieldReplacementTarget::Instruction | FieldReplacementTarget::All
        ) {
            record.instruction = regex
                .replace_all(&record.instruction, replacement.replacement.as_str())
                .into_owned();
        }
        if matches!(
            replacement.field,
            FieldReplacementTarget::Input | FieldReplacementTarget::All
        ) {
            record.input = regex
                .replace_all(&record.input, replacement.replacement.as_str())
                .into_owned();
        }
        if matches!(
            replacement.field,
            FieldReplacementTarget::Response | FieldReplacementTarget::All
        ) {
            record.response = regex
                .replace_all(&record.response, replacement.replacement.as_str())
                .into_owned();
        }
    }
    Ok(())
}

fn apply_field_defaults(record: &mut InstructionRecord, defaults: &[FieldDefault]) {
    for default in defaults {
        match default.field {
            FieldDefaultTarget::System if record.system.trim().is_empty() => {
                record.system = default.value.clone();
            }
            FieldDefaultTarget::Instruction if record.instruction.trim().is_empty() => {
                record.instruction = default.value.clone();
            }
            FieldDefaultTarget::Input if record.input.trim().is_empty() => {
                record.input = default.value.clone();
            }
            FieldDefaultTarget::Response if record.response.trim().is_empty() => {
                record.response = default.value.clone();
            }
            _ => {}
        }
    }
}

fn apply_field_case_transforms(record: &mut InstructionRecord, transforms: &[FieldCaseTransform]) {
    for transform in transforms {
        if matches!(
            transform.field,
            FieldReplacementTarget::System | FieldReplacementTarget::All
        ) {
            record.system = apply_field_case_transform(&record.system, transform.case);
        }
        if matches!(
            transform.field,
            FieldReplacementTarget::Instruction | FieldReplacementTarget::All
        ) {
            record.instruction = apply_field_case_transform(&record.instruction, transform.case);
        }
        if matches!(
            transform.field,
            FieldReplacementTarget::Input | FieldReplacementTarget::All
        ) {
            record.input = apply_field_case_transform(&record.input, transform.case);
        }
        if matches!(
            transform.field,
            FieldReplacementTarget::Response | FieldReplacementTarget::All
        ) {
            record.response = apply_field_case_transform(&record.response, transform.case);
        }
    }
}

fn apply_field_case_transform(value: &str, case: FieldCaseTransformKind) -> String {
    match case {
        FieldCaseTransformKind::Lowercase => value.to_lowercase(),
        FieldCaseTransformKind::Uppercase => value.to_uppercase(),
    }
}

fn apply_field_affixes(record: &mut InstructionRecord, affixes: &[FieldAffix]) {
    for affix in affixes {
        if matches!(
            affix.field,
            FieldReplacementTarget::System | FieldReplacementTarget::All
        ) {
            record.system = apply_field_affix(&record.system, affix);
        }
        if matches!(
            affix.field,
            FieldReplacementTarget::Instruction | FieldReplacementTarget::All
        ) {
            record.instruction = apply_field_affix(&record.instruction, affix);
        }
        if matches!(
            affix.field,
            FieldReplacementTarget::Input | FieldReplacementTarget::All
        ) {
            record.input = apply_field_affix(&record.input, affix);
        }
        if matches!(
            affix.field,
            FieldReplacementTarget::Response | FieldReplacementTarget::All
        ) {
            record.response = apply_field_affix(&record.response, affix);
        }
    }
}

fn apply_field_affix(value: &str, affix: &FieldAffix) -> String {
    let mut transformed =
        String::with_capacity(affix.prefix.len() + value.len() + affix.suffix.len());
    transformed.push_str(&affix.prefix);
    transformed.push_str(value);
    transformed.push_str(&affix.suffix);
    transformed
}

fn apply_field_strips(record: &mut InstructionRecord, strips: &[FieldStrip]) {
    for strip in strips {
        if matches!(
            strip.field,
            FieldReplacementTarget::System | FieldReplacementTarget::All
        ) {
            record.system = apply_field_strip(&record.system, strip);
        }
        if matches!(
            strip.field,
            FieldReplacementTarget::Instruction | FieldReplacementTarget::All
        ) {
            record.instruction = apply_field_strip(&record.instruction, strip);
        }
        if matches!(
            strip.field,
            FieldReplacementTarget::Input | FieldReplacementTarget::All
        ) {
            record.input = apply_field_strip(&record.input, strip);
        }
        if matches!(
            strip.field,
            FieldReplacementTarget::Response | FieldReplacementTarget::All
        ) {
            record.response = apply_field_strip(&record.response, strip);
        }
    }
}

fn apply_field_strip(value: &str, strip: &FieldStrip) -> String {
    let without_prefix = value
        .strip_prefix(&strip.prefix)
        .filter(|_| !strip.prefix.is_empty())
        .unwrap_or(value);
    without_prefix
        .strip_suffix(&strip.suffix)
        .filter(|_| !strip.suffix.is_empty())
        .unwrap_or(without_prefix)
        .to_string()
}

fn apply_field_splits(record: &mut InstructionRecord, splits: &[FieldSplit]) {
    for split in splits {
        if matches!(
            split.field,
            FieldReplacementTarget::System | FieldReplacementTarget::All
        ) {
            record.system = apply_field_split(&record.system, split);
        }
        if matches!(
            split.field,
            FieldReplacementTarget::Instruction | FieldReplacementTarget::All
        ) {
            record.instruction = apply_field_split(&record.instruction, split);
        }
        if matches!(
            split.field,
            FieldReplacementTarget::Input | FieldReplacementTarget::All
        ) {
            record.input = apply_field_split(&record.input, split);
        }
        if matches!(
            split.field,
            FieldReplacementTarget::Response | FieldReplacementTarget::All
        ) {
            record.response = apply_field_split(&record.response, split);
        }
    }
}

fn apply_field_split(value: &str, split: &FieldSplit) -> String {
    match value.split_once(&split.delimiter) {
        Some((before, after)) => match split.side {
            FieldSplitSide::Before => before.to_string(),
            FieldSplitSide::After => after.to_string(),
        },
        None => value.to_string(),
    }
}

fn apply_field_truncations(record: &mut InstructionRecord, truncations: &[FieldTruncation]) {
    for truncation in truncations {
        if matches!(
            truncation.field,
            FieldReplacementTarget::System | FieldReplacementTarget::All
        ) {
            record.system = truncate_chars(&record.system, truncation.max_chars);
        }
        if matches!(
            truncation.field,
            FieldReplacementTarget::Instruction | FieldReplacementTarget::All
        ) {
            record.instruction = truncate_chars(&record.instruction, truncation.max_chars);
        }
        if matches!(
            truncation.field,
            FieldReplacementTarget::Input | FieldReplacementTarget::All
        ) {
            record.input = truncate_chars(&record.input, truncation.max_chars);
        }
        if matches!(
            truncation.field,
            FieldReplacementTarget::Response | FieldReplacementTarget::All
        ) {
            record.response = truncate_chars(&record.response, truncation.max_chars);
        }
    }
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn apply_field_transforms(
    record: &mut InstructionRecord,
    transforms: &[FieldTransform],
) -> Result<()> {
    for transform in transforms {
        match transform.op {
            FieldTransformOp::Default => {
                apply_field_transform_targets(record, transform.field, |value| {
                    if value.trim().is_empty() {
                        transform.value.clone()
                    } else {
                        value.to_string()
                    }
                });
            }
            FieldTransformOp::Replace => {
                apply_field_transform_targets(record, transform.field, |value| {
                    value.replace(&transform.pattern, &transform.replacement)
                });
            }
            FieldTransformOp::RegexReplace => {
                let regex = Regex::new(&transform.pattern).with_context(|| {
                    format!(
                        "invalid data.field_transforms regex_replace pattern {:?}",
                        transform.pattern
                    )
                })?;
                apply_field_transform_targets(record, transform.field, |value| {
                    regex
                        .replace_all(value, transform.replacement.as_str())
                        .into_owned()
                });
            }
            FieldTransformOp::Case => {
                let Some(case) = transform.case else {
                    continue;
                };
                apply_field_transform_targets(record, transform.field, |value| {
                    apply_field_case_transform(value, case)
                });
            }
            FieldTransformOp::Affix => {
                apply_field_transform_targets(record, transform.field, |value| {
                    let mut transformed = String::with_capacity(
                        transform.prefix.len() + value.len() + transform.suffix.len(),
                    );
                    transformed.push_str(&transform.prefix);
                    transformed.push_str(value);
                    transformed.push_str(&transform.suffix);
                    transformed
                });
            }
            FieldTransformOp::Strip => {
                apply_field_transform_targets(record, transform.field, |value| {
                    let without_prefix = value
                        .strip_prefix(&transform.prefix)
                        .filter(|_| !transform.prefix.is_empty())
                        .unwrap_or(value);
                    without_prefix
                        .strip_suffix(&transform.suffix)
                        .filter(|_| !transform.suffix.is_empty())
                        .unwrap_or(without_prefix)
                        .to_string()
                });
            }
            FieldTransformOp::Split => {
                let Some(side) = transform.side else {
                    continue;
                };
                apply_field_transform_targets(record, transform.field, |value| {
                    match value.split_once(&transform.delimiter) {
                        Some((before, after)) => match side {
                            FieldSplitSide::Before => before.to_string(),
                            FieldSplitSide::After => after.to_string(),
                        },
                        None => value.to_string(),
                    }
                });
            }
            FieldTransformOp::Truncate => {
                let Some(max_chars) = transform.max_chars else {
                    continue;
                };
                apply_field_transform_targets(record, transform.field, |value| {
                    truncate_chars(value, max_chars)
                });
            }
        }
    }
    Ok(())
}

fn apply_field_transform_targets<F>(
    record: &mut InstructionRecord,
    field: FieldReplacementTarget,
    mut transform: F,
) where
    F: FnMut(&str) -> String,
{
    if matches!(
        field,
        FieldReplacementTarget::System | FieldReplacementTarget::All
    ) {
        record.system = transform(&record.system);
    }
    if matches!(
        field,
        FieldReplacementTarget::Instruction | FieldReplacementTarget::All
    ) {
        record.instruction = transform(&record.instruction);
    }
    if matches!(
        field,
        FieldReplacementTarget::Input | FieldReplacementTarget::All
    ) {
        record.input = transform(&record.input);
    }
    if matches!(
        field,
        FieldReplacementTarget::Response | FieldReplacementTarget::All
    ) {
        record.response = transform(&record.response);
    }
}

fn field_default_value(defaults: &[FieldDefault], field: FieldDefaultTarget) -> Option<&str> {
    defaults
        .iter()
        .find(|default| default.field == field)
        .map(|default| default.value.as_str())
}

fn normalize_record_whitespace(record: &mut InstructionRecord) {
    record.system = normalize_whitespace(&record.system);
    record.instruction = normalize_whitespace(&record.instruction);
    record.input = normalize_whitespace(&record.input);
    record.response = normalize_whitespace(&record.response);
}

fn normalize_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn optional_jsonl_string_field(
    values: &BTreeMap<String, serde_json::Value>,
    field: &str,
) -> Result<String> {
    match jsonl_path_value(values, field) {
        Some(serde_json::Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(anyhow!("JSONL field {field} must be a string")),
        None => Ok(String::new()),
    }
}

fn defaultable_jsonl_string_field(
    values: &BTreeMap<String, serde_json::Value>,
    field: &str,
    default_value: Option<&str>,
) -> Result<String> {
    match jsonl_path_value(values, field) {
        Some(serde_json::Value::String(value)) if value.trim().is_empty() => {
            Ok(default_value.unwrap_or(value).to_string())
        }
        Some(serde_json::Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(anyhow!("JSONL field {field} must be a string")),
        None => default_value
            .map(|value| value.to_string())
            .ok_or_else(|| anyhow!("JSONL record missing required field {field}")),
    }
}

fn jsonl_path_value<'a>(
    values: &'a BTreeMap<String, serde_json::Value>,
    field: &str,
) -> Option<&'a serde_json::Value> {
    if let Some(value) = values.get(field) {
        return Some(value);
    }
    let mut segments = field.split('.');
    let first = segments.next()?;
    let mut value = values.get(first)?;
    for segment in segments {
        match value {
            serde_json::Value::Object(object) => {
                value = object.get(segment)?;
            }
            _ => return None,
        }
    }
    Some(value)
}

fn required_chat_message_string_field(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    messages_field: &str,
    index: usize,
) -> Result<String> {
    match object.get(field) {
        Some(serde_json::Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(anyhow!(
            "chat message {messages_field}[{index}].{field} must be a string"
        )),
        None => Err(anyhow!(
            "chat message {messages_field}[{index}] missing required field {field}"
        )),
    }
}

fn render_sft_prompt(
    record: &InstructionRecord,
    prompt_template: &str,
    prompt_with_input_template: &str,
) -> Result<String> {
    if prompt_template.is_empty() {
        return Err(anyhow!("data.prompt_template must not be empty"));
    }
    if prompt_with_input_template.is_empty() {
        return Err(anyhow!("data.prompt_with_input_template must not be empty"));
    }
    let template = if record.input.trim().is_empty() {
        prompt_template
    } else {
        prompt_with_input_template
    };
    Ok(template
        .replace("{system}", &record.system)
        .replace("{instruction}", &record.instruction)
        .replace("{input}", &record.input))
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
    instruction_field: &str,
    input_field: &str,
    response_field: &str,
    source_instruction_fields: &[String],
    source_input_fields: &[String],
    source_response_fields: &[String],
    system_field: Option<String>,
    chat_messages_field: Option<String>,
    prompt_template: &str,
    prompt_with_input_template: &str,
    trim_fields: bool,
    min_response_chars: usize,
    max_response_chars: Option<usize>,
    instruction_contains_any: &[String],
    instruction_excludes_any: &[String],
    response_contains_any: &[String],
    response_excludes_any: &[String],
    input_contains_any: &[String],
    input_excludes_any: &[String],
    field_regex_contains_any: &[FieldRegexFilter],
    field_regex_excludes_any: &[FieldRegexFilter],
    field_replacements: &[FieldReplacement],
    field_regex_replacements: &[FieldRegexReplacement],
    normalize_whitespace: bool,
    field_defaults: &[FieldDefault],
    field_case_transforms: &[FieldCaseTransform],
    field_affixes: &[FieldAffix],
    field_strips: &[FieldStrip],
    field_splits: &[FieldSplit],
    field_truncations: &[FieldTruncation],
    field_transforms: &[FieldTransform],
    max_eval_samples: Option<usize>,
    min_system_chars: Option<usize>,
    max_system_chars: Option<usize>,
    system_contains_any: &[String],
    system_excludes_any: &[String],
    min_instruction_chars: Option<usize>,
    max_instruction_chars: Option<usize>,
    min_input_chars: Option<usize>,
    max_input_chars: Option<usize>,
    min_prompt_chars: Option<usize>,
    max_prompt_chars: Option<usize>,
    min_sample_chars: Option<usize>,
    max_sample_chars: Option<usize>,
    dedupe_samples: bool,
    source_weights: &[usize],
    source_max_samples: &[usize],
    skip_invalid_records: bool,
    dataset: &SftDataset,
) -> Result<()> {
    let cache = SftCache {
        source_paths: source_paths.to_vec(),
        vocab_size,
        instruction_field: instruction_field.to_string(),
        input_field: input_field.to_string(),
        response_field: response_field.to_string(),
        source_instruction_fields: source_instruction_fields.to_vec(),
        source_input_fields: source_input_fields.to_vec(),
        source_response_fields: source_response_fields.to_vec(),
        system_field,
        chat_messages_field,
        prompt_template: prompt_template.to_string(),
        prompt_with_input_template: prompt_with_input_template.to_string(),
        trim_fields,
        min_response_chars,
        max_response_chars,
        instruction_contains_any: instruction_contains_any.to_vec(),
        instruction_excludes_any: instruction_excludes_any.to_vec(),
        response_contains_any: response_contains_any.to_vec(),
        response_excludes_any: response_excludes_any.to_vec(),
        input_contains_any: input_contains_any.to_vec(),
        input_excludes_any: input_excludes_any.to_vec(),
        field_regex_contains_any: field_regex_contains_any.to_vec(),
        field_regex_excludes_any: field_regex_excludes_any.to_vec(),
        field_replacements: field_replacements.to_vec(),
        field_regex_replacements: field_regex_replacements.to_vec(),
        normalize_whitespace,
        field_defaults: field_defaults.to_vec(),
        field_case_transforms: field_case_transforms.to_vec(),
        field_affixes: field_affixes.to_vec(),
        field_strips: field_strips.to_vec(),
        field_splits: field_splits.to_vec(),
        field_truncations: field_truncations.to_vec(),
        field_transforms: field_transforms.to_vec(),
        max_eval_samples,
        min_system_chars,
        max_system_chars,
        system_contains_any: system_contains_any.to_vec(),
        system_excludes_any: system_excludes_any.to_vec(),
        min_instruction_chars,
        max_instruction_chars,
        min_input_chars,
        max_input_chars,
        min_prompt_chars,
        max_prompt_chars,
        min_sample_chars,
        max_sample_chars,
        dedupe_samples,
        source_weights: source_weights.to_vec(),
        source_max_samples: source_max_samples.to_vec(),
        skip_invalid_records,
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
    use crate::runtime::{
        DataConfig, DataKind, FieldRegexReplacement, FieldReplacement, FieldReplacementTarget,
    };

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
            max_eval_samples: None,
            shuffle: true,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
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
            max_eval_samples: None,
            shuffle: true,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
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
            max_eval_samples: None,
            shuffle: true,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
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
    fn sft_dataset_uses_configured_prompt_templates() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"name project\",\"input\":\"rust trainer\",\"response\":\"rustrain\"}\n{\"instruction\":\"name language\",\"response\":\"Rust\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Q: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "Q: {instruction}\nContext: {input}\nA: ".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .tokenizer
            .decode_lossy(&dataset.train_samples[0].tokens);

        assert!(decoded.contains("Q: name project"));
        assert!(decoded.contains("Context: rust trainer"));
        assert!(decoded.contains("A: rustrain"));
        assert!(!decoded.contains("Instruction:"));
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
    }

    #[test]
    fn sft_dataset_uses_configured_system_field_in_prompt_templates() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"system\":\"Be concise.\",\"instruction\":\"name project\",\"response\":\"rustrain\"}\n{\"system\":\"Use one word.\",\"instruction\":\"name language\",\"response\":\"Rust\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: Some("system".to_string()),
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");

        assert!(decoded.contains("System: Be concise."));
        assert!(decoded.contains("System: Use one word."));
    }

    #[test]
    fn sft_dataset_filters_system_lengths_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"system\":\"ai\",\"instruction\":\"too short\",\"response\":\"skip\"}\n{\"system\":\"brief\",\"instruction\":\"first kept\",\"response\":\"one\"}\n{\"system\":\"concise\",\"instruction\":\"second kept\",\"response\":\"two\"}\n{\"system\":\"this system prompt is too long\",\"instruction\":\"too long\",\"response\":\"skip\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: Some(2),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: Some("system".to_string()),
            chat_messages_field: None,
            min_system_chars: Some(4),
            max_system_chars: Some(8),
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");

        assert!(decoded.contains("first kept"));
        assert!(decoded.contains("second kept"));
        assert!(!decoded.contains("too short"));
        assert!(!decoded.contains("too long"));
    }

    #[test]
    fn sft_dataset_filters_systems_by_substring_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"system\":\"keep system\",\"instruction\":\"first\",\"response\":\"answer one\"}\n{\"system\":\"ordinary system\",\"instruction\":\"skip\",\"response\":\"answer skip\"}\n{\"system\":\"selected system\",\"instruction\":\"second\",\"response\":\"answer two\"}\n{\"system\":\"keep extra\",\"instruction\":\"third\",\"response\":\"answer three\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.67,
            max_samples: Some(3),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: Some("system".to_string()),
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: vec!["keep".to_string(), "selected".to_string()],
            system_excludes_any: Vec::new(),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(dataset.train_samples.len(), 2);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("System: keep system"));
        assert!(decoded.contains("System: selected system"));
        assert!(decoded.contains("System: keep extra"));
        assert!(!decoded.contains("ordinary system"));
        assert!(!decoded.contains("answer skip"));
    }

    #[test]
    fn sft_dataset_filters_systems_by_exclude_substring_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"system\":\"keep system\",\"instruction\":\"first\",\"response\":\"answer one\"}\n{\"system\":\"banned system\",\"instruction\":\"skip\",\"response\":\"answer skip\"}\n{\"system\":\"selected system\",\"instruction\":\"second\",\"response\":\"answer two\"}\n{\"system\":\"keep extra\",\"instruction\":\"third\",\"response\":\"answer three\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.67,
            max_samples: Some(3),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: Some("system".to_string()),
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: vec!["banned".to_string()],
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(dataset.train_samples.len(), 2);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("System: keep system"));
        assert!(decoded.contains("System: selected system"));
        assert!(decoded.contains("System: keep extra"));
        assert!(!decoded.contains("banned system"));
        assert!(!decoded.contains("answer skip"));
    }

    #[test]
    fn sft_dataset_trim_fields_controls_prompt_normalization() {
        let record = InstructionRecord {
            system: "  stay brief  ".to_string(),
            instruction: "  name project  ".to_string(),
            input: "  rust trainer  ".to_string(),
            response: "  rustrain  ".to_string(),
        };
        let trim_data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: Vec::new(),
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Q:{instruction}\nA:".to_string(),
            prompt_with_input_template: "Q:{instruction}\nI:{input}\nA:".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let raw_data = DataConfig {
            trim_fields: false,
            ..trim_data.clone()
        };
        let trimmed = InstructionRecord {
            system: normalize_jsonl_field(record.system.clone(), &trim_data),
            instruction: normalize_jsonl_field(record.instruction.clone(), &trim_data),
            input: normalize_jsonl_field(record.input.clone(), &trim_data),
            response: normalize_jsonl_field(record.response.clone(), &trim_data),
        };
        let raw = InstructionRecord {
            system: normalize_jsonl_field(record.system.clone(), &raw_data),
            instruction: normalize_jsonl_field(record.instruction.clone(), &raw_data),
            input: normalize_jsonl_field(record.input.clone(), &raw_data),
            response: normalize_jsonl_field(record.response.clone(), &raw_data),
        };

        assert_eq!(
            render_sft_prompt(
                &trimmed,
                &trim_data.prompt_template,
                &trim_data.prompt_with_input_template,
            )
            .expect("trimmed prompt should render"),
            "Q:name project\nI:rust trainer\nA:"
        );
        assert_eq!(
            render_sft_prompt(
                &raw,
                &raw_data.prompt_template,
                &raw_data.prompt_with_input_template,
            )
            .expect("raw prompt should render"),
            "Q:  name project  \nI:  rust trainer  \nA:"
        );
        assert_eq!(trimmed.response, "rustrain");
        assert_eq!(raw.response, "  rustrain  ");
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
            max_eval_samples: None,
            shuffle: true,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 3);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(cache_dir.join("sft_tokenized.toml").exists());
    }

    #[test]
    fn sft_dataset_limits_explicit_eval_paths() {
        let dir = tempdir().expect("temp dir should be created");
        let train_dir = dir.path().join("train_sft");
        let eval_dir = dir.path().join("eval_sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&train_dir).expect("train SFT dir should be created");
        fs::create_dir_all(&eval_dir).expect("eval SFT dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            train_dir.join("train.jsonl"),
            "{\"instruction\":\"train one\",\"response\":\"alpha\"}\n{\"instruction\":\"train two\",\"response\":\"beta\"}\n",
        )
        .expect("train jsonl should write");
        fs::write(
            eval_dir.join("eval.jsonl"),
            "{\"instruction\":\"eval one\",\"response\":\"gamma\"}\n{\"instruction\":\"eval two\",\"response\":\"delta\"}\n{\"instruction\":\"eval three\",\"response\":\"epsilon\"}\n",
        )
        .expect("eval jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![train_dir],
            eval_paths: vec![eval_dir],
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: Some(2),
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 2);
        assert_eq!(dataset.eval_samples.len(), 2);
        let eval_decoded = dataset
            .eval_samples
            .iter()
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(eval_decoded.contains("eval one"));
        assert!(eval_decoded.contains("eval two"));
        assert!(!eval_decoded.contains("eval three"));
    }

    #[test]
    fn sft_dataset_filters_short_responses_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"empty\",\"response\":\"\"}\n{\"instruction\":\"short\",\"response\":\"ok\"}\n{\"instruction\":\"first\",\"response\":\"valid\"}\n{\"instruction\":\"second\",\"response\":\"works\"}\n{\"instruction\":\"third\",\"response\":\"later\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: Some(2),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 5,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        let first = dataset
            .tokenizer
            .decode_lossy(&dataset.train_samples[0].tokens);
        let second = dataset
            .tokenizer
            .decode_lossy(&dataset.eval_samples[0].tokens);
        assert!(first.contains("first"));
        assert!(second.contains("second"));
        assert!(!first.contains("empty"));
        assert!(!first.contains("short"));
    }

    #[test]
    fn sft_dataset_filters_responses_by_substring_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"first\",\"response\":\"approved answer\"}\n{\"instruction\":\"skip\",\"response\":\"ordinary reply\"}\n{\"instruction\":\"second\",\"response\":\"contains verified marker\"}\n{\"instruction\":\"third\",\"response\":\"approved final\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.67,
            max_samples: Some(3),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: vec!["approved".to_string(), "verified".to_string()],
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(dataset.train_samples.len(), 2);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("first"));
        assert!(decoded.contains("second"));
        assert!(decoded.contains("third"));
        assert!(!decoded.contains("ordinary reply"));
    }

    #[test]
    fn sft_dataset_filters_instructions_by_substring_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"keep first task\",\"response\":\"answer one\"}\n{\"instruction\":\"skip ordinary\",\"response\":\"answer skip\"}\n{\"instruction\":\"second task\",\"response\":\"answer two\"}\n{\"instruction\":\"keep third\",\"response\":\"answer three\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.67,
            max_samples: Some(3),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: vec!["task".to_string(), "keep".to_string()],
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(dataset.train_samples.len(), 2);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("keep first task"));
        assert!(decoded.contains("second task"));
        assert!(decoded.contains("keep third"));
        assert!(!decoded.contains("skip ordinary"));
        assert!(!decoded.contains("answer skip"));
    }

    #[test]
    fn sft_dataset_filters_instructions_by_exclude_substring_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"first clean task\",\"response\":\"answer one\"}\n{\"instruction\":\"skip banned task\",\"response\":\"answer skip\"}\n{\"instruction\":\"second safe task\",\"response\":\"answer two\"}\n{\"instruction\":\"third clean task\",\"response\":\"answer three\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.67,
            max_samples: Some(3),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: vec!["banned".to_string()],
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(dataset.train_samples.len(), 2);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("first clean task"));
        assert!(decoded.contains("second safe task"));
        assert!(decoded.contains("third clean task"));
        assert!(!decoded.contains("skip banned task"));
        assert!(!decoded.contains("answer skip"));
    }

    #[test]
    fn sft_dataset_filters_responses_by_exclude_substring_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"first\",\"response\":\"clean answer\"}\n{\"instruction\":\"skip\",\"response\":\"contains banned marker\"}\n{\"instruction\":\"second\",\"response\":\"safe reply\"}\n{\"instruction\":\"third\",\"response\":\"clean final\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.67,
            max_samples: Some(3),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: vec!["banned".to_string()],
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(dataset.train_samples.len(), 2);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("first"));
        assert!(decoded.contains("second"));
        assert!(decoded.contains("third"));
        assert!(!decoded.contains("skip"));
        assert!(!decoded.contains("banned marker"));
    }

    #[test]
    fn sft_dataset_filters_long_responses_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"first\",\"response\":\"valid\"}\n{\"instruction\":\"too long\",\"response\":\"toolong\"}\n{\"instruction\":\"second\",\"response\":\"works\"}\n{\"instruction\":\"third\",\"response\":\"later\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: Some(2),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: Some(5),
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        let first = dataset
            .tokenizer
            .decode_lossy(&dataset.train_samples[0].tokens);
        let second = dataset
            .tokenizer
            .decode_lossy(&dataset.eval_samples[0].tokens);
        assert!(first.contains("first"));
        assert!(second.contains("second"));
        assert!(!first.contains("too long"));
        assert!(!second.contains("too long"));
    }

    #[test]
    fn sft_dataset_filters_instruction_lengths_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"a\",\"response\":\"skip\"}\n{\"instruction\":\"first\",\"response\":\"valid\"}\n{\"instruction\":\"too long\",\"response\":\"skip\"}\n{\"instruction\":\"second\",\"response\":\"works\"}\n{\"instruction\":\"third\",\"response\":\"later\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: Some(2),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: Some(3),
            max_instruction_chars: Some(6),
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        let first = dataset
            .tokenizer
            .decode_lossy(&dataset.train_samples[0].tokens);
        let second = dataset
            .tokenizer
            .decode_lossy(&dataset.eval_samples[0].tokens);
        assert!(first.contains("first"));
        assert!(second.contains("second"));
        assert!(!first.contains("too long"));
        assert!(!second.contains("too long"));
        assert!(!first.contains("Instruction:\na"));
        assert!(!second.contains("Instruction:\na"));
    }

    #[test]
    fn sft_dataset_filters_input_lengths_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"skip short\",\"input\":\"x\",\"response\":\"short\"}\n{\"instruction\":\"first\",\"input\":\"ok\",\"response\":\"valid\"}\n{\"instruction\":\"skip long\",\"input\":\"toolong\",\"response\":\"long\"}\n{\"instruction\":\"second\",\"input\":\"mid\",\"response\":\"works\"}\n{\"instruction\":\"third\",\"input\":\"fit\",\"response\":\"later\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: Some(2),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: Some(2),
            max_input_chars: Some(3),
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        let first = dataset
            .tokenizer
            .decode_lossy(&dataset.train_samples[0].tokens);
        let second = dataset
            .tokenizer
            .decode_lossy(&dataset.eval_samples[0].tokens);
        assert!(first.contains("first"));
        assert!(first.contains("Input:\nok"));
        assert!(second.contains("second"));
        assert!(second.contains("Input:\nmid"));
        assert!(!first.contains("skip short"));
        assert!(!second.contains("skip short"));
        assert!(!first.contains("skip long"));
        assert!(!second.contains("skip long"));
    }

    #[test]
    fn sft_dataset_filters_inputs_by_substring_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"first\",\"input\":\"keep context\",\"response\":\"answer one\"}\n{\"instruction\":\"skip\",\"input\":\"ordinary context\",\"response\":\"answer skip\"}\n{\"instruction\":\"second\",\"input\":\"selected context\",\"response\":\"answer two\"}\n{\"instruction\":\"third\",\"input\":\"keep extra\",\"response\":\"answer three\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.67,
            max_samples: Some(3),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: vec!["keep".to_string(), "selected".to_string()],
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(dataset.train_samples.len(), 2);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("first"));
        assert!(decoded.contains("Input:\nkeep context"));
        assert!(decoded.contains("second"));
        assert!(decoded.contains("Input:\nselected context"));
        assert!(decoded.contains("third"));
        assert!(decoded.contains("Input:\nkeep extra"));
        assert!(!decoded.contains("skip"));
        assert!(!decoded.contains("ordinary context"));
        assert!(!decoded.contains("answer skip"));
    }

    #[test]
    fn sft_dataset_filters_inputs_by_exclude_substring_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"first\",\"input\":\"clean context\",\"response\":\"answer one\"}\n{\"instruction\":\"skip\",\"input\":\"contains banned context\",\"response\":\"answer skip\"}\n{\"instruction\":\"second\",\"input\":\"safe context\",\"response\":\"answer two\"}\n{\"instruction\":\"third\",\"input\":\"clean extra\",\"response\":\"answer three\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.67,
            max_samples: Some(3),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: vec!["banned".to_string()],
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(dataset.train_samples.len(), 2);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("first"));
        assert!(decoded.contains("Input:\nclean context"));
        assert!(decoded.contains("second"));
        assert!(decoded.contains("Input:\nsafe context"));
        assert!(decoded.contains("third"));
        assert!(decoded.contains("Input:\nclean extra"));
        assert!(!decoded.contains("skip"));
        assert!(!decoded.contains("banned context"));
        assert!(!decoded.contains("answer skip"));
    }

    #[test]
    fn sft_dataset_filters_prompt_lengths_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"a\",\"response\":\"skip\"}\n{\"instruction\":\"first\",\"input\":\"ok\",\"response\":\"valid\"}\n{\"instruction\":\"this prompt is too long\",\"response\":\"skip\"}\n{\"instruction\":\"second\",\"response\":\"works\"}\n{\"instruction\":\"third\",\"response\":\"later\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: Some(2),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Q:{instruction}\nA:".to_string(),
            prompt_with_input_template: "Q:{instruction}\nI:{input}\nA:".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: Some(11),
            max_prompt_chars: Some(15),
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        let first = dataset
            .tokenizer
            .decode_lossy(&dataset.train_samples[0].tokens);
        let second = dataset
            .tokenizer
            .decode_lossy(&dataset.eval_samples[0].tokens);
        assert!(first.contains("first"));
        assert!(first.contains("I:ok"));
        assert!(second.contains("second"));
        assert!(!first.contains("Q:a"));
        assert!(!second.contains("Q:a"));
        assert!(!first.contains("this prompt is too long"));
        assert!(!second.contains("this prompt is too long"));
    }

    #[test]
    fn sft_dataset_filters_sample_lengths_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"a\",\"response\":\"x\"}\n{\"instruction\":\"first\",\"input\":\"ok\",\"response\":\"valid\"}\n{\"instruction\":\"too long\",\"response\":\"this response is too long\"}\n{\"instruction\":\"second\",\"response\":\"works\"}\n{\"instruction\":\"tiny\",\"response\":\"z\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: Some(2),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Q:{instruction}\nA:".to_string(),
            prompt_with_input_template: "Q:{instruction}\nI:{input}\nA:".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: Some(16),
            max_sample_chars: Some(22),
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        let first = dataset
            .tokenizer
            .decode_lossy(&dataset.train_samples[0].tokens);
        let second = dataset
            .tokenizer
            .decode_lossy(&dataset.eval_samples[0].tokens);
        assert!(first.contains("first"));
        assert!(first.contains("valid"));
        assert!(second.contains("second"));
        assert!(second.contains("works"));
        assert!(!first.contains("Q:a"));
        assert!(!second.contains("Q:a"));
        assert!(!first.contains("this response is too long"));
        assert!(!second.contains("this response is too long"));
    }

    #[test]
    fn sft_dataset_applies_source_weights_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let first_dir = dir.path().join("first");
        let second_dir = dir.path().join("second");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&first_dir).expect("first dir should be created");
        fs::create_dir_all(&second_dir).expect("second dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            first_dir.join("first.jsonl"),
            "{\"instruction\":\"first\",\"response\":\"alpha\"}\n",
        )
        .expect("first jsonl should write");
        fs::write(
            second_dir.join("second.jsonl"),
            "{\"instruction\":\"second\",\"response\":\"beta\"}\n",
        )
        .expect("second jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![first_dir, second_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: Some(3),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: vec![2, 2],
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 2);
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>();
        assert!(decoded[0].contains("first"));
        assert!(decoded[1].contains("first"));
        assert!(decoded[2].contains("second"));
    }

    #[test]
    fn sft_dataset_applies_source_max_samples_before_weighting_and_split() {
        let dir = tempdir().expect("temp dir should be created");
        let first_dir = dir.path().join("first");
        let second_dir = dir.path().join("second");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&first_dir).expect("first dir should be created");
        fs::create_dir_all(&second_dir).expect("second dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            first_dir.join("first.jsonl"),
            "{\"instruction\":\"first-a\",\"response\":\"alpha\"}\n{\"instruction\":\"first-b\",\"response\":\"skip\"}\n",
        )
        .expect("first jsonl should write");
        fs::write(
            second_dir.join("second.jsonl"),
            "{\"instruction\":\"second-a\",\"response\":\"beta\"}\n{\"instruction\":\"second-b\",\"response\":\"gamma\"}\n{\"instruction\":\"second-c\",\"response\":\"skip\"}\n",
        )
        .expect("second jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![first_dir, second_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: Some(6),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: vec![2, 2],
            source_max_samples: vec![1, 2],
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(dataset.train_samples.len(), 3);
        assert_eq!(dataset.eval_samples.len(), 3);
        assert_eq!(decoded.matches("first-a").count(), 2);
        assert_eq!(decoded.matches("second-a").count(), 2);
        assert_eq!(decoded.matches("second-b").count(), 2);
        assert!(!decoded.contains("first-b"));
        assert!(!decoded.contains("second-c"));
    }

    #[test]
    fn sft_dataset_dedupes_samples_before_split_and_limit() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"first\",\"response\":\"valid\"}\n{\"instruction\":\"first\",\"response\":\"valid\"}\n{\"instruction\":\"second\",\"response\":\"works\"}\n{\"instruction\":\"second\",\"response\":\"works\"}\n{\"instruction\":\"third\",\"response\":\"later\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: Some(2),
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: true,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        let first = dataset
            .tokenizer
            .decode_lossy(&dataset.train_samples[0].tokens);
        let second = dataset
            .tokenizer
            .decode_lossy(&dataset.eval_samples[0].tokens);
        assert!(first.contains("first"));
        assert!(second.contains("second"));
        assert!(!first.contains("third"));
        assert!(!second.contains("third"));
    }

    #[test]
    fn sft_dataset_skip_invalid_records_keeps_valid_rows() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"first\",\"response\":\"alpha\"}\nnot-json\n{\"instruction\":\"missing-response\"}\n{\"instruction\":\"second\",\"response\":\"beta\"}\n{\"instruction\":\"third\",\"response\":\"gamma\"}\n",
        )
        .expect("jsonl should write");

        let mut data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.67,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };

        let strict_error = load_sft_dataset(&data, 128, 64, &cache_dir)
            .expect_err("strict SFT loading should reject invalid rows")
            .to_string();
        assert!(strict_error.contains("failed to parse JSONL record"));

        data.skip_invalid_records = true;
        let dataset = load_sft_dataset(&data, 128, 64, &cache_dir).expect("valid rows should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(dataset.train_samples.len(), 2);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("first"));
        assert!(decoded.contains("second"));
        assert!(decoded.contains("third"));
        assert!(!decoded.contains("missing-response"));
    }

    #[test]
    fn sft_dataset_supports_chat_messages_records() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("chat.jsonl"),
            "{\"messages\":[{\"role\":\"system\",\"content\":\"Be concise.\"},{\"role\":\"user\",\"content\":\"Name the project.\"},{\"role\":\"assistant\",\"content\":\"rustrain\"}]}\n{\"messages\":[{\"role\":\"system\",\"content\":\"Be concise.\"},{\"role\":\"human\",\"content\":\"Name the language.\"},{\"role\":\"gpt\",\"content\":\"Rust\"}]}\n{\"messages\":[{\"role\":\"system\",\"content\":\"Ignore verbosity.\"},{\"role\":\"user\",\"content\":\"Filtered row.\"},{\"role\":\"assistant\",\"content\":\"skip me\"}]}\n{\"messages\":[{\"role\":\"assistant\",\"content\":\"missing user\"}]}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: Some("messages".to_string()),
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: vec!["concise".to_string()],
            system_excludes_any: Vec::new(),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: true,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("chat SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        let cache_text =
            fs::read_to_string(cache_dir.join("sft_tokenized.toml")).expect("cache should read");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("System: Be concise."));
        assert!(decoded.contains("Q: Name the project."));
        assert!(decoded.contains("rustrain"));
        assert!(decoded.contains("Rust"));
        assert!(!decoded.contains("Filtered row"));
        assert!(!decoded.contains("skip me"));
        assert!(!decoded.contains("missing user"));
        assert!(cache_text.contains("chat_messages_field = \"messages\""));
    }

    #[test]
    fn sft_dataset_supports_dotted_jsonl_field_paths() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("nested.jsonl"),
            "{\"meta\":{\"system\":\"Be exact.\"},\"payload\":{\"prompt\":\"Name the project.\",\"context\":\"Rust trainer\",\"answer\":\"rustrain\"}}\n{\"meta\":{\"system\":\"Use one word.\"},\"payload\":{\"prompt\":\"Name the language.\",\"answer\":\"Rust\"}}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "payload.prompt".to_string(),
            input_field: "payload.context".to_string(),
            response_field: "payload.answer".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: Some("meta.system".to_string()),
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("nested SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        let cache_text =
            fs::read_to_string(cache_dir.join("sft_tokenized.toml")).expect("cache should read");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("System: Be exact."));
        assert!(decoded.contains("Q: Name the project."));
        assert!(decoded.contains("I: Rust trainer"));
        assert!(decoded.contains("rustrain"));
        assert!(decoded.contains("Use one word."));
        assert!(decoded.contains("Rust"));
        assert!(cache_text.contains("payload.prompt"));
        assert!(cache_text.contains("meta.system"));
    }

    #[test]
    fn sft_dataset_applies_field_replacements_before_filters_and_cache() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"Keep PROJECT_TOKEN\",\"response\":\"APPROVED_TOKEN\"}\n{\"instruction\":\"Also PROJECT_TOKEN\",\"response\":\"APPROVED_TOKEN\"}\n{\"instruction\":\"Drop PROJECT_TOKEN\",\"response\":\"DENIED_TOKEN\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 8,
            max_response_chars: None,
            instruction_contains_any: vec!["rustrain".to_string()],
            instruction_excludes_any: Vec::new(),
            response_contains_any: vec!["approved".to_string()],
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: vec![
                FieldReplacement {
                    field: FieldReplacementTarget::Instruction,
                    pattern: "PROJECT_TOKEN".to_string(),
                    replacement: "rustrain".to_string(),
                },
                FieldReplacement {
                    field: FieldReplacementTarget::Response,
                    pattern: "APPROVED_TOKEN".to_string(),
                    replacement: "approved".to_string(),
                },
            ],
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        let cache_text =
            fs::read_to_string(cache_dir.join("sft_tokenized.toml")).expect("cache should read");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("Keep rustrain"));
        assert!(decoded.contains("approved"));
        assert!(!decoded.contains("PROJECT_TOKEN"));
        assert!(cache_text.contains("field_replacements"));
        assert!(cache_text.contains("PROJECT_TOKEN"));
    }

    #[test]
    fn sft_dataset_applies_regex_replacements_before_filters_and_cache() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"Keep project-123 token\",\"response\":\"APPROVED:42\"}\n{\"instruction\":\"Also project-777 token\",\"response\":\"APPROVED:84\"}\n{\"instruction\":\"Drop project-456 token\",\"response\":\"DENIED:99\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 8,
            max_response_chars: None,
            instruction_contains_any: vec!["project-id token".to_string()],
            instruction_excludes_any: Vec::new(),
            response_contains_any: vec!["approved id".to_string()],
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: vec![
                FieldRegexReplacement {
                    field: FieldReplacementTarget::Instruction,
                    pattern: r"project-\d+".to_string(),
                    replacement: "project-id".to_string(),
                },
                FieldRegexReplacement {
                    field: FieldReplacementTarget::Response,
                    pattern: r"APPROVED:\d+".to_string(),
                    replacement: "approved id".to_string(),
                },
            ],
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        let cache_text =
            fs::read_to_string(cache_dir.join("sft_tokenized.toml")).expect("cache should read");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("project-id token"));
        assert!(decoded.contains("approved id"));
        assert!(!decoded.contains("project-123"));
        assert!(cache_text.contains("field_regex_replacements"));
        assert!(cache_text.contains("project-id"));
    }

    #[test]
    fn sft_dataset_filters_fields_by_regex_before_split_and_cache() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"Keep project-123 token\",\"input\":\"GPU context\",\"response\":\"approved answer\"}\n{\"instruction\":\"Also project-789 token\",\"input\":\"GPU context\",\"response\":\"approved reply\"}\n{\"instruction\":\"Drop project-456 token\",\"input\":\"GPU context\",\"response\":\"DENIED answer\"}\n{\"instruction\":\"Skip project-abc token\",\"input\":\"GPU context\",\"response\":\"approved answer\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Q: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "Q: {instruction}\nI: {input}\nA: ".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: vec![FieldRegexFilter {
                field: FieldReplacementTarget::Instruction,
                pattern: r"project-\d+".to_string(),
            }],
            field_regex_excludes_any: vec![FieldRegexFilter {
                field: FieldReplacementTarget::Response,
                pattern: r"DENIED|blocked".to_string(),
            }],
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        let cache_text =
            fs::read_to_string(cache_dir.join("sft_tokenized.toml")).expect("cache should read");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("Keep project-123 token"));
        assert!(decoded.contains("Also project-789 token"));
        assert!(decoded.contains("approved answer"));
        assert!(decoded.contains("approved reply"));
        assert!(!decoded.contains("Drop project-456 token"));
        assert!(!decoded.contains("DENIED answer"));
        assert!(!decoded.contains("Skip project-abc token"));
        assert!(cache_text.contains("field_regex_contains_any"));
        assert!(cache_text.contains("field_regex_excludes_any"));
        assert!(cache_text.contains("DENIED|blocked"));
    }

    #[test]
    fn sft_dataset_applies_field_defaults_before_filters_and_cache() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"\",\"response\":\"\",\"input\":\"\"}\n{\"response\":\"\",\"input\":\"\"}\n{\"response\":\"kept\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: vec!["assistant".to_string()],
            system_excludes_any: Vec::new(),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            trim_fields: true,
            min_response_chars: 10,
            max_response_chars: None,
            instruction_contains_any: vec!["default instruction".to_string()],
            instruction_excludes_any: Vec::new(),
            response_contains_any: vec!["default response".to_string()],
            response_excludes_any: Vec::new(),
            input_contains_any: vec!["default input".to_string()],
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: vec![
                FieldDefault {
                    field: FieldDefaultTarget::System,
                    value: "system assistant".to_string(),
                },
                FieldDefault {
                    field: FieldDefaultTarget::Instruction,
                    value: "default instruction".to_string(),
                },
                FieldDefault {
                    field: FieldDefaultTarget::Input,
                    value: "default input".to_string(),
                },
                FieldDefault {
                    field: FieldDefaultTarget::Response,
                    value: "default response".to_string(),
                },
            ],
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        let cache_text =
            fs::read_to_string(cache_dir.join("sft_tokenized.toml")).expect("cache should read");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("System: system assistant"));
        assert!(decoded.contains("Q: default instruction"));
        assert!(decoded.contains("I: default input"));
        assert!(decoded.contains("default response"));
        assert!(!decoded.contains("kept"));
        assert!(cache_text.contains("field_defaults"));
        assert!(cache_text.contains("default instruction"));
    }

    #[test]
    fn sft_dataset_normalizes_whitespace_after_replacements_before_filters_and_cache() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"Keep PROJECT_TOKEN\",\"response\":\"approved\\t\\tanswer\"}\n{\"instruction\":\"Also PROJECT_TOKEN\",\"response\":\"approved   answer\"}\n{\"instruction\":\"Drop PROJECT_TOKEN\",\"response\":\"denied   answer\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: vec!["rust train".to_string()],
            instruction_excludes_any: Vec::new(),
            response_contains_any: vec!["approved answer".to_string()],
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: vec![FieldReplacement {
                field: FieldReplacementTarget::Instruction,
                pattern: "PROJECT_TOKEN".to_string(),
                replacement: "rust   train".to_string(),
            }],
            field_regex_replacements: Vec::new(),
            normalize_whitespace: true,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        let cache_text =
            fs::read_to_string(cache_dir.join("sft_tokenized.toml")).expect("cache should read");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("rust train"));
        assert!(decoded.contains("approved answer"));
        assert!(!decoded.contains("rust   train"));
        assert!(!decoded.contains("approved   answer"));
        assert!(cache_text.contains("normalize_whitespace = true"));
    }

    #[test]
    fn sft_dataset_applies_field_transform_dsl_before_filters_and_cache() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"Keep PROJECT-123::tail\",\"input\":\"gpu context\",\"response\":\"APPROVED:42 extra\"}\n{\"instruction\":\"Keep PROJECT-456::tail\",\"input\":\"gpu eval\",\"response\":\"APPROVED:99 extra\"}\n{\"instruction\":\"Drop PROJECT-789::tail\",\"input\":\"cpu context\",\"response\":\"DENIED:99 extra\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 11,
            max_response_chars: Some(11),
            instruction_contains_any: vec!["task Keep project-id".to_string()],
            instruction_excludes_any: Vec::new(),
            response_contains_any: vec!["approved id".to_string()],
            response_excludes_any: Vec::new(),
            input_contains_any: vec!["GPU".to_string()],
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: vec![
                FieldTransform {
                    field: FieldReplacementTarget::Instruction,
                    op: FieldTransformOp::RegexReplace,
                    pattern: r"PROJECT-\d+".to_string(),
                    replacement: "project-id".to_string(),
                    ..FieldTransform::default()
                },
                FieldTransform {
                    field: FieldReplacementTarget::Instruction,
                    op: FieldTransformOp::Split,
                    delimiter: "::".to_string(),
                    side: Some(FieldSplitSide::Before),
                    ..FieldTransform::default()
                },
                FieldTransform {
                    field: FieldReplacementTarget::Instruction,
                    op: FieldTransformOp::Affix,
                    prefix: "task ".to_string(),
                    ..FieldTransform::default()
                },
                FieldTransform {
                    field: FieldReplacementTarget::Input,
                    op: FieldTransformOp::Case,
                    case: Some(FieldCaseTransformKind::Uppercase),
                    ..FieldTransform::default()
                },
                FieldTransform {
                    field: FieldReplacementTarget::Response,
                    op: FieldTransformOp::RegexReplace,
                    pattern: r"APPROVED:\d+".to_string(),
                    replacement: "approved id".to_string(),
                    ..FieldTransform::default()
                },
                FieldTransform {
                    field: FieldReplacementTarget::Response,
                    op: FieldTransformOp::Truncate,
                    max_chars: Some(11),
                    ..FieldTransform::default()
                },
            ],
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        let cache_text =
            fs::read_to_string(cache_dir.join("sft_tokenized.toml")).expect("cache should read");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("task Keep project-id"));
        assert!(decoded.contains("GPU CONTEXT"));
        assert!(decoded.contains("approved id"));
        assert!(!decoded.contains("tail"));
        assert!(!decoded.contains("extra"));
        assert!(cache_text.contains("field_transforms"));
        assert!(cache_text.contains("regex_replace"));
    }

    #[test]
    fn sft_dataset_applies_case_transforms_before_filters_and_cache() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"Keep PROJECT_TOKEN\",\"input\":\"GPU CONTEXT\",\"response\":\"APPROVED ANSWER\"}\n{\"instruction\":\"Also PROJECT_TOKEN\",\"input\":\"GPU CONTEXT\",\"response\":\"APPROVED ANSWER\"}\n{\"instruction\":\"Drop PROJECT_TOKEN\",\"input\":\"GPU CONTEXT\",\"response\":\"DENIED ANSWER\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 8,
            max_response_chars: None,
            instruction_contains_any: vec!["rustrain".to_string()],
            instruction_excludes_any: Vec::new(),
            response_contains_any: vec!["approved answer".to_string()],
            response_excludes_any: Vec::new(),
            input_contains_any: vec!["gpu context".to_string()],
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: vec![FieldReplacement {
                field: FieldReplacementTarget::Instruction,
                pattern: "PROJECT_TOKEN".to_string(),
                replacement: "RUSTRain".to_string(),
            }],
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: vec![
                FieldCaseTransform {
                    field: FieldReplacementTarget::Instruction,
                    case: FieldCaseTransformKind::Lowercase,
                },
                FieldCaseTransform {
                    field: FieldReplacementTarget::Input,
                    case: FieldCaseTransformKind::Lowercase,
                },
                FieldCaseTransform {
                    field: FieldReplacementTarget::Response,
                    case: FieldCaseTransformKind::Lowercase,
                },
            ],
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        let cache_text =
            fs::read_to_string(cache_dir.join("sft_tokenized.toml")).expect("cache should read");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("keep rustrain"));
        assert!(decoded.contains("gpu context"));
        assert!(decoded.contains("approved answer"));
        assert!(!decoded.contains("PROJECT_TOKEN"));
        assert!(!decoded.contains("APPROVED ANSWER"));
        assert!(cache_text.contains("field_case_transforms"));
        assert!(cache_text.contains("lowercase"));
    }

    #[test]
    fn sft_dataset_applies_field_affixes_before_filters_and_cache() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"Name the project\",\"input\":\"GPU\",\"response\":\"rustrain\"}\n{\"instruction\":\"Name the language\",\"input\":\"GPU\",\"response\":\"Rust\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: vec!["system:".to_string()],
            system_excludes_any: Vec::new(),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            trim_fields: true,
            min_response_chars: 10,
            max_response_chars: None,
            instruction_contains_any: vec!["Q: Name".to_string()],
            instruction_excludes_any: Vec::new(),
            response_contains_any: vec!["</answer>".to_string()],
            response_excludes_any: Vec::new(),
            input_contains_any: vec!["ctx=GPU".to_string()],
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: vec![FieldDefault {
                field: FieldDefaultTarget::System,
                value: "concise".to_string(),
            }],
            field_case_transforms: Vec::new(),
            field_affixes: vec![
                FieldAffix {
                    field: FieldReplacementTarget::System,
                    prefix: "system: ".to_string(),
                    suffix: String::new(),
                },
                FieldAffix {
                    field: FieldReplacementTarget::Instruction,
                    prefix: "Q: ".to_string(),
                    suffix: "?".to_string(),
                },
                FieldAffix {
                    field: FieldReplacementTarget::Input,
                    prefix: "ctx=".to_string(),
                    suffix: String::new(),
                },
                FieldAffix {
                    field: FieldReplacementTarget::Response,
                    prefix: String::new(),
                    suffix: "</answer>".to_string(),
                },
            ],
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        let cache_text =
            fs::read_to_string(cache_dir.join("sft_tokenized.toml")).expect("cache should read");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("System: system: concise"));
        assert!(decoded.contains("Q: Q: Name the project?"));
        assert!(decoded.contains("I: ctx=GPU"));
        assert!(decoded.contains("rustrain</answer>"));
        assert!(cache_text.contains("field_affixes"));
        assert!(cache_text.contains("</answer>"));
    }

    #[test]
    fn sft_dataset_applies_field_strips_before_filters_and_cache() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"PROMPT: Keep GPU prompt\",\"input\":\"ctx=GPU context\",\"response\":\"<answer>approved answer</answer>\"}\n{\"instruction\":\"PROMPT: Keep GPU eval prompt\",\"input\":\"ctx=GPU context\",\"response\":\"<answer>approved answer eval</answer>\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Q: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "Q: {instruction}\nI: {input}\nA: ".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: vec!["Keep GPU".to_string()],
            instruction_excludes_any: Vec::new(),
            response_contains_any: vec!["approved answer".to_string()],
            response_excludes_any: vec!["</answer>".to_string()],
            input_contains_any: vec!["GPU context".to_string()],
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: vec![
                FieldStrip {
                    field: FieldReplacementTarget::Instruction,
                    prefix: "PROMPT: ".to_string(),
                    suffix: String::new(),
                },
                FieldStrip {
                    field: FieldReplacementTarget::Input,
                    prefix: "ctx=".to_string(),
                    suffix: String::new(),
                },
                FieldStrip {
                    field: FieldReplacementTarget::Response,
                    prefix: "<answer>".to_string(),
                    suffix: "</answer>".to_string(),
                },
            ],
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        let cache_text =
            fs::read_to_string(cache_dir.join("sft_tokenized.toml")).expect("cache should read");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("Q: Keep GPU prompt"));
        assert!(decoded.contains("I: GPU context"));
        assert!(decoded.contains("approved answer"));
        assert!(!decoded.contains("PROMPT:"));
        assert!(!decoded.contains("ctx="));
        assert!(!decoded.contains("</answer>"));
        assert!(cache_text.contains("field_strips"));
        assert!(cache_text.contains("PROMPT: "));
    }

    #[test]
    fn sft_dataset_applies_field_truncations_before_filters_and_cache() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"Name the rustrain project precisely\",\"input\":\"GPU worker context\",\"response\":\"approved response with trailing text\"}\n{\"instruction\":\"Name the rustrain language precisely\",\"input\":\"GPU worker context\",\"response\":\"accepted response with trailing text\"}\n{\"instruction\":\"Name the rustrain project precisely\",\"input\":\"CPU worker context\",\"response\":\"denied response with trailing text\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Q: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "Q: {instruction}\nI: {input}\nA: ".to_string(),
            trim_fields: true,
            min_response_chars: 17,
            max_response_chars: Some(17),
            instruction_contains_any: vec!["Name the rustrain".to_string()],
            instruction_excludes_any: Vec::new(),
            response_contains_any: vec!["response".to_string()],
            response_excludes_any: Vec::new(),
            input_contains_any: vec!["GPU".to_string()],
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: vec![
                FieldTruncation {
                    field: FieldReplacementTarget::Instruction,
                    max_chars: 17,
                },
                FieldTruncation {
                    field: FieldReplacementTarget::Input,
                    max_chars: 3,
                },
                FieldTruncation {
                    field: FieldReplacementTarget::Response,
                    max_chars: 17,
                },
            ],
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        let cache_text =
            fs::read_to_string(cache_dir.join("sft_tokenized.toml")).expect("cache should read");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("Q: Name the rustrain"));
        assert!(decoded.contains("I: GPU"));
        assert!(decoded.contains("approved response"));
        assert!(decoded.contains("accepted response"));
        assert!(!decoded.contains("trailing text"));
        assert!(cache_text.contains("field_truncations"));
        assert!(cache_text.contains("max_chars = 17"));
    }

    #[test]
    fn sft_dataset_applies_field_splits_before_filters_and_cache() {
        let dir = tempdir().expect("temp dir should be created");
        let data_dir = dir.path().join("sft");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&data_dir).expect("sft dir should be created");
        fs::create_dir_all(&cache_dir).expect("cache dir should be created");
        fs::write(
            data_dir.join("sample.jsonl"),
            "{\"instruction\":\"metadata :: Keep GPU prompt\",\"input\":\"GPU context || discard\",\"response\":\"draft -> approved answer\"}\n{\"instruction\":\"metadata :: Also GPU prompt\",\"input\":\"GPU context || discard\",\"response\":\"draft -> accepted answer\"}\n{\"instruction\":\"metadata :: Drop CPU prompt\",\"input\":\"CPU context || discard\",\"response\":\"draft -> denied answer\"}\n",
        )
        .expect("jsonl should write");

        let data = DataConfig {
            kind: DataKind::InstructionJsonl,
            paths: vec![data_dir],
            eval_paths: Vec::new(),
            train_split: 0.5,
            max_samples: None,
            max_eval_samples: None,
            shuffle: false,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Q: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "Q: {instruction}\nI: {input}\nA: ".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: vec!["GPU prompt".to_string()],
            instruction_excludes_any: Vec::new(),
            response_contains_any: vec!["answer".to_string()],
            response_excludes_any: Vec::new(),
            input_contains_any: vec!["GPU context".to_string()],
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: vec![
                FieldSplit {
                    field: FieldReplacementTarget::Instruction,
                    delimiter: " :: ".to_string(),
                    side: FieldSplitSide::After,
                },
                FieldSplit {
                    field: FieldReplacementTarget::Input,
                    delimiter: " || ".to_string(),
                    side: FieldSplitSide::Before,
                },
                FieldSplit {
                    field: FieldReplacementTarget::Response,
                    delimiter: " -> ".to_string(),
                    side: FieldSplitSide::After,
                },
            ],
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 96, &cache_dir).expect("SFT dataset should load");
        let decoded = dataset
            .train_samples
            .iter()
            .chain(dataset.eval_samples.iter())
            .map(|sample| dataset.tokenizer.decode_lossy(&sample.tokens))
            .collect::<Vec<_>>()
            .join("\n");
        let cache_text =
            fs::read_to_string(cache_dir.join("sft_tokenized.toml")).expect("cache should read");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
        assert!(decoded.contains("Q: Keep GPU prompt"));
        assert!(decoded.contains("Q: Also GPU prompt"));
        assert!(decoded.contains("I: GPU context"));
        assert!(decoded.contains("approved answer"));
        assert!(decoded.contains("accepted answer"));
        assert!(!decoded.contains("metadata ::"));
        assert!(!decoded.contains("draft ->"));
        assert!(!decoded.contains("discard"));
        assert!(cache_text.contains("field_splits"));
        assert!(cache_text.contains("delimiter = \" :: \""));
        assert!(cache_text.contains("side = \"after\""));
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
            max_eval_samples: None,
            shuffle: true,
            index_cache: None,
            instruction_field: "instruction".to_string(),
            input_field: "input".to_string(),
            response_field: "response".to_string(),
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            system_field: None,
            chat_messages_field: None,
            min_system_chars: None,
            max_system_chars: None,
            system_contains_any: Vec::new(),
            system_excludes_any: Vec::new(),
            prompt_template: "Instruction:\n{instruction}\n\nResponse:\n".to_string(),
            prompt_with_input_template:
                "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:\n".to_string(),
            trim_fields: true,
            min_response_chars: 1,
            max_response_chars: None,
            instruction_contains_any: Vec::new(),
            instruction_excludes_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            input_contains_any: Vec::new(),
            input_excludes_any: Vec::new(),
            field_regex_contains_any: Vec::new(),
            field_regex_excludes_any: Vec::new(),
            min_instruction_chars: None,
            max_instruction_chars: None,
            min_input_chars: None,
            max_input_chars: None,
            min_prompt_chars: None,
            max_prompt_chars: None,
            min_sample_chars: None,
            max_sample_chars: None,
            dedupe_samples: false,
            field_replacements: Vec::new(),
            field_regex_replacements: Vec::new(),
            normalize_whitespace: false,
            field_defaults: Vec::new(),
            field_case_transforms: Vec::new(),
            field_affixes: Vec::new(),
            field_strips: Vec::new(),
            field_splits: Vec::new(),
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            source_weights: Vec::new(),
            source_max_samples: Vec::new(),
            skip_invalid_records: false,
            external_metadata_paths: Vec::new(),
        };
        let dataset =
            load_sft_dataset(&data, 128, 64, &cache_dir).expect("SFT dataset should load");

        assert_eq!(dataset.train_samples.len(), 1);
        assert_eq!(dataset.eval_samples.len(), 1);
    }
}
