//! sft module - split from qwen_module.rs

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    io::{BufRead, BufReader, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use arrow::{
    array::{Array, LargeStringArray, RecordBatch, StringArray},
    datatypes::{DataType, SchemaRef},
    ipc::reader::{FileReader as ArrowFileReader, StreamReader as ArrowStreamReader},
};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use tch::{Device, IndexOp, Kind, Reduction, Tensor, no_grad};
use tokenizers::Tokenizer;
use tracing::info;

use rustrain_checkpoint::io::{
    delta_manifest_path, optimizer_state_path, qwen_lora_sft_adapter_manifest_path,
    read_qwen_lora_sft_resume_manifest, write_qwen_delta_manifest,
    write_qwen_lora_sft_adapter_manifest,
};
use rustrain_checkpoint::manifest::*;
use rustrain_checkpoint::safetensors::{read_safetensors_map, tensor};
use rustrain_core::runtime::{
    Config, DataConfig as RuntimeDataConfig, DataKind as RuntimeDataKind, Device as RuntimeDevice,
    FieldAffix, FieldCaseTransform, FieldCaseTransformKind, FieldDefault, FieldDefaultTarget,
    FieldRegexFilter, FieldRegexReplacement, FieldReplacement, FieldReplacementTarget, FieldSplit,
    FieldSplitSide, FieldStrip, FieldTransform, FieldTransformOp, FieldTruncation,
    LoraConfig as RuntimeLoraConfig, LrScheduler, RunPaths, load_config,
};
use rustrain_nccl::nccl_smoke;

use crate::generate::*;
use crate::lora::*;
use crate::model::*;
use crate::parity::*;
use crate::rank_smoke::*;
use crate::session::*;

#[derive(Clone)]
pub(crate) struct QwenSftTokenSample {
    pub(crate) prompt_tokens: usize,
    pub(crate) response_tokens: usize,
    pub(crate) masked_positions: usize,
    pub(crate) token_ids: Vec<i64>,
    pub(crate) mask_values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QwenSftExample {
    pub(crate) system: String,
    pub(crate) instruction: String,
    pub(crate) input: String,
    pub(crate) response: String,
}

#[derive(Debug)]
pub(crate) struct QwenSftExampleSet {
    pub(crate) examples: Vec<QwenSftExample>,
    pub(crate) source_files: Vec<String>,
    pub(crate) source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct QwenSftRawSampleIndex {
    pub(crate) path: String,
    pub(crate) index_in_file: usize,
    pub(crate) global_index: usize,
    pub(crate) byte_offset: u64,
    #[serde(default)]
    pub(crate) field_map: QwenSftSourceFieldMap,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct QwenSftArrowRawSampleIndex {
    pub(crate) path: String,
    pub(crate) row_index: usize,
    pub(crate) global_index: usize,
    #[serde(default)]
    pub(crate) field_map: QwenSftSourceFieldMap,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QwenSftStreamingSourceIndex {
    pub(crate) samples: Vec<QwenSftRawSampleIndex>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QwenSftArrowSourceIndex {
    pub(crate) samples: Vec<QwenSftArrowRawSampleIndex>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QwenSftStreamingSourceScan {
    pub(crate) index: QwenSftStreamingSourceIndex,
    pub(crate) summary: QwenSftStreamingSourceSummary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QwenSftArrowSourceScan {
    pub(crate) index: QwenSftArrowSourceIndex,
    pub(crate) summary: QwenSftStreamingSourceSummary,
}

pub(crate) struct QwenSftArrowSourceRowScan {
    pub(crate) row_indices: Vec<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct QwenSftStreamingSourceIndexCache {
    pub(crate) format: String,
    pub(crate) paths: Vec<String>,
    pub(crate) source_files: Vec<QwenSftStreamingSourceFileMetadata>,
    pub(crate) max_samples: Option<usize>,
    pub(crate) field_map: QwenSftFieldMap,
    pub(crate) min_response_chars: usize,
    pub(crate) summary: QwenSftStreamingSourceSummary,
    pub(crate) samples: Vec<QwenSftRawSampleIndex>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct QwenSftArrowSourceIndexCache {
    pub(crate) format: String,
    pub(crate) paths: Vec<String>,
    pub(crate) source_files: Vec<QwenSftStreamingSourceFileMetadata>,
    pub(crate) max_samples: Option<usize>,
    pub(crate) field_map: QwenSftFieldMap,
    pub(crate) summary: QwenSftStreamingSourceSummary,
    pub(crate) samples: Vec<QwenSftArrowRawSampleIndex>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct QwenSftStreamingSourceFileMetadata {
    pub(crate) path: String,
    pub(crate) len: u64,
    pub(crate) modified_unix_nanos: u128,
}

#[derive(Debug)]
pub(crate) struct QwenSftStreamingSourceIndexLoad {
    pub(crate) index: QwenSftStreamingSourceIndex,
    pub(crate) summary: Option<QwenSftStreamingSourceSummary>,
    pub(crate) cache_hit: bool,
    pub(crate) cache_written: bool,
}

#[derive(Debug)]
pub(crate) struct QwenSftArrowSourceIndexLoad {
    pub(crate) index: QwenSftArrowSourceIndex,
    pub(crate) summary: Option<QwenSftStreamingSourceSummary>,
    pub(crate) cache_hit: bool,
    pub(crate) cache_written: bool,
}

pub(crate) struct QwenSftStreamingTokenWindow {
    pub(crate) samples: Vec<QwenSftTokenSample>,
    pub(crate) raw_sample_indices: Vec<QwenSftRawSampleIndex>,
    pub(crate) raw_samples_read: usize,
    pub(crate) source_index_cache_hit: bool,
    pub(crate) source_index_cache_written: bool,
}

pub(crate) struct QwenSftArrowStreamingTokenWindow {
    pub(crate) samples: Vec<QwenSftTokenSample>,
    pub(crate) raw_samples_read: usize,
    pub(crate) source_index_cache_hit: bool,
    pub(crate) source_index_cache_written: bool,
}

pub(crate) struct QwenSftRawExampleWindow {
    pub(crate) examples: Vec<QwenSftExample>,
    pub(crate) raw_samples_read: usize,
}

pub(crate) struct QwenSftRecord {
    pub(crate) system: String,
    pub(crate) instruction: String,
    pub(crate) input: String,
    pub(crate) response: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct QwenSftSourceFieldMap {
    pub(crate) instruction: Option<String>,
    pub(crate) input: Option<String>,
    pub(crate) response: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QwenSftArrowColumn {
    Present(usize),
    MissingOptional,
}

pub(crate) struct QwenCompiledRegexReplacement {
    pub(crate) field: FieldReplacementTarget,
    pub(crate) regex: regex::Regex,
    pub(crate) replacement: String,
}

pub(crate) struct QwenCompiledRegexFieldTransform {
    pub(crate) index: usize,
    pub(crate) field: FieldReplacementTarget,
    pub(crate) regex: regex::Regex,
    pub(crate) replacement: String,
}

pub(crate) struct QwenCompiledRegexFilter {
    pub(crate) field: FieldReplacementTarget,
    pub(crate) regex: regex::Regex,
}

pub(crate) struct QwenSftRegexPlan {
    pub(crate) replacements: Vec<QwenCompiledRegexReplacement>,
    pub(crate) transform_regex_replacements: Vec<QwenCompiledRegexFieldTransform>,
    pub(crate) contains_any: Vec<QwenCompiledRegexFilter>,
    pub(crate) excludes_any: Vec<QwenCompiledRegexFilter>,
}

impl QwenSftRegexPlan {
    pub(crate) fn compile(field_map: &QwenSftFieldMap) -> Result<Self> {
        let replacements = field_map
            .field_regex_replacements
            .iter()
            .map(|replacement| {
                let regex = regex::Regex::new(&replacement.pattern).with_context(|| {
                    format!(
                        "invalid data.field_regex_replacements pattern {:?}",
                        replacement.pattern
                    )
                })?;
                Ok(QwenCompiledRegexReplacement {
                    field: replacement.field.clone(),
                    regex,
                    replacement: replacement.replacement.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let transform_regex_replacements = field_map
            .field_transforms
            .iter()
            .enumerate()
            .filter(|(_, transform)| matches!(transform.op, FieldTransformOp::RegexReplace))
            .map(|(index, transform)| {
                let regex = regex::Regex::new(&transform.pattern).with_context(|| {
                    format!(
                        "invalid data.field_transforms regex_replace pattern {:?}",
                        transform.pattern
                    )
                })?;
                Ok(QwenCompiledRegexFieldTransform {
                    index,
                    field: transform.field.clone(),
                    regex,
                    replacement: transform.replacement.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let contains_any = qwen_compile_regex_filters(&field_map.field_regex_contains_any)?;
        let excludes_any = qwen_compile_regex_filters(&field_map.field_regex_excludes_any)?;
        Ok(Self {
            replacements,
            transform_regex_replacements,
            contains_any,
            excludes_any,
        })
    }
}

pub(crate) fn qwen_compile_regex_filters(
    filters: &[FieldRegexFilter],
) -> Result<Vec<QwenCompiledRegexFilter>> {
    filters
        .iter()
        .map(|filter| {
            let regex = regex::Regex::new(&filter.pattern).with_context(|| {
                format!(
                    "invalid data field regex filter pattern {:?}",
                    filter.pattern
                )
            })?;
            Ok(QwenCompiledRegexFilter {
                field: filter.field.clone(),
                regex,
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct QwenSftFieldMap {
    pub(crate) instruction: String,
    pub(crate) input: String,
    pub(crate) response: String,
    #[serde(default)]
    pub(crate) system: Option<String>,
    #[serde(default)]
    pub(crate) chat_messages: Option<String>,
    #[serde(default)]
    pub(crate) min_system_chars: Option<usize>,
    #[serde(default)]
    pub(crate) max_system_chars: Option<usize>,
    #[serde(default)]
    pub(crate) system_contains_any: Vec<String>,
    #[serde(default)]
    pub(crate) system_excludes_any: Vec<String>,
    pub(crate) prompt_template: String,
    pub(crate) prompt_with_input_template: String,
    pub(crate) trim_fields: bool,
    pub(crate) min_response_chars: usize,
    #[serde(default)]
    pub(crate) max_response_chars: Option<usize>,
    #[serde(default)]
    pub(crate) instruction_contains_any: Vec<String>,
    #[serde(default)]
    pub(crate) instruction_excludes_any: Vec<String>,
    #[serde(default)]
    pub(crate) response_contains_any: Vec<String>,
    #[serde(default)]
    pub(crate) response_excludes_any: Vec<String>,
    #[serde(default)]
    pub(crate) input_contains_any: Vec<String>,
    #[serde(default)]
    pub(crate) input_excludes_any: Vec<String>,
    #[serde(default)]
    pub(crate) field_regex_contains_any: Vec<FieldRegexFilter>,
    #[serde(default)]
    pub(crate) field_regex_excludes_any: Vec<FieldRegexFilter>,
    #[serde(default)]
    pub(crate) min_instruction_chars: Option<usize>,
    #[serde(default)]
    pub(crate) max_instruction_chars: Option<usize>,
    #[serde(default)]
    pub(crate) min_input_chars: Option<usize>,
    #[serde(default)]
    pub(crate) max_input_chars: Option<usize>,
    #[serde(default)]
    pub(crate) min_prompt_chars: Option<usize>,
    #[serde(default)]
    pub(crate) max_prompt_chars: Option<usize>,
    #[serde(default)]
    pub(crate) min_sample_chars: Option<usize>,
    #[serde(default)]
    pub(crate) max_sample_chars: Option<usize>,
    #[serde(default)]
    pub(crate) dedupe_samples: bool,
    #[serde(default)]
    pub(crate) field_replacements: Vec<FieldReplacement>,
    #[serde(default)]
    pub(crate) field_regex_replacements: Vec<FieldRegexReplacement>,
    #[serde(default)]
    pub(crate) normalize_whitespace: bool,
    #[serde(default)]
    pub(crate) field_defaults: Vec<FieldDefault>,
    #[serde(default)]
    pub(crate) field_case_transforms: Vec<FieldCaseTransform>,
    #[serde(default)]
    pub(crate) field_affixes: Vec<FieldAffix>,
    #[serde(default)]
    pub(crate) field_strips: Vec<FieldStrip>,
    #[serde(default)]
    pub(crate) field_splits: Vec<FieldSplit>,
    #[serde(default)]
    pub(crate) field_truncations: Vec<FieldTruncation>,
    #[serde(default)]
    pub(crate) field_transforms: Vec<FieldTransform>,
    pub(crate) source_weights: Vec<usize>,
    #[serde(default)]
    pub(crate) source_max_samples: Vec<usize>,
    #[serde(default)]
    pub(crate) source_instruction_fields: Vec<String>,
    #[serde(default)]
    pub(crate) source_input_fields: Vec<String>,
    #[serde(default)]
    pub(crate) source_response_fields: Vec<String>,
    #[serde(default)]
    pub(crate) skip_invalid_records: bool,
    #[serde(default)]
    pub(crate) external_metadata: Vec<QwenSftExternalMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct QwenSftExternalMetadata {
    pub(crate) path: String,
    pub(crate) contents: String,
}

pub(crate) struct QwenSftArrowExampleSet {
    pub(crate) examples: Vec<QwenSftExample>,
    pub(crate) row_indices: Vec<usize>,
    pub(crate) source_rows: usize,
    pub(crate) arrow_ipc_format: String,
    pub(crate) columns: Vec<String>,
    pub(crate) source_files: Vec<String>,
    pub(crate) source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) fingerprint: String,
}

pub(crate) struct QwenSftArrowPlanData {
    pub(crate) dataset_summary: QwenSftDatasetSummary,
    pub(crate) train_dataset: QwenSftDataset,
    pub(crate) eval_dataset: QwenSftDataset,
}

pub(crate) struct QwenSftStreamingDatasetPlan {
    pub(crate) dataset_total_samples: usize,
    pub(crate) dataset_train_samples: usize,
    pub(crate) dataset_eval_samples: usize,
    pub(crate) dataset_source_files: Vec<String>,
    pub(crate) dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) dataset_fingerprint: String,
    pub(crate) dataset_shuffle: bool,
    pub(crate) local_batch_size: usize,
    pub(crate) data_epoch_start: usize,
    pub(crate) data_epoch_end: usize,
    pub(crate) data_epoch_next: usize,
    pub(crate) data_sample_offset_start: usize,
    pub(crate) data_sample_offset_end: usize,
    pub(crate) data_sample_offset_next: usize,
    pub(crate) train_batches: Vec<Tensor>,
    pub(crate) initial_input_ids: Tensor,
    pub(crate) sequence_tokens: usize,
    pub(crate) streaming_index_cache_hit: bool,
    pub(crate) streaming_index_cache_written: bool,
}

impl QwenSftFieldMap {
    pub(crate) fn from_runtime_data(data: &RuntimeDataConfig) -> Result<Self> {
        let map = Self {
            instruction: data.instruction_field.clone(),
            input: data.input_field.clone(),
            response: data.response_field.clone(),
            system: data.system_field.clone(),
            chat_messages: data.chat_messages_field.clone(),
            min_system_chars: data.min_system_chars,
            max_system_chars: data.max_system_chars,
            system_contains_any: data.system_contains_any.clone(),
            system_excludes_any: data.system_excludes_any.clone(),
            prompt_template: data.prompt_template.clone(),
            prompt_with_input_template: data.prompt_with_input_template.clone(),
            trim_fields: data.trim_fields,
            min_response_chars: data.min_response_chars,
            max_response_chars: data.max_response_chars,
            instruction_contains_any: data.instruction_contains_any.clone(),
            instruction_excludes_any: data.instruction_excludes_any.clone(),
            response_contains_any: data.response_contains_any.clone(),
            response_excludes_any: data.response_excludes_any.clone(),
            input_contains_any: data.input_contains_any.clone(),
            input_excludes_any: data.input_excludes_any.clone(),
            field_regex_contains_any: data.field_regex_contains_any.clone(),
            field_regex_excludes_any: data.field_regex_excludes_any.clone(),
            min_instruction_chars: data.min_instruction_chars,
            max_instruction_chars: data.max_instruction_chars,
            min_input_chars: data.min_input_chars,
            max_input_chars: data.max_input_chars,
            min_prompt_chars: data.min_prompt_chars,
            max_prompt_chars: data.max_prompt_chars,
            min_sample_chars: data.min_sample_chars,
            max_sample_chars: data.max_sample_chars,
            dedupe_samples: data.dedupe_samples,
            field_replacements: data.field_replacements.clone(),
            field_regex_replacements: data.field_regex_replacements.clone(),
            normalize_whitespace: data.normalize_whitespace,
            field_defaults: data.field_defaults.clone(),
            field_case_transforms: data.field_case_transforms.clone(),
            field_affixes: data.field_affixes.clone(),
            field_strips: data.field_strips.clone(),
            field_splits: data.field_splits.clone(),
            field_truncations: data.field_truncations.clone(),
            field_transforms: data.field_transforms.clone(),
            source_weights: data.source_weights.clone(),
            source_max_samples: data.source_max_samples.clone(),
            source_instruction_fields: data.source_instruction_fields.clone(),
            source_input_fields: data.source_input_fields.clone(),
            source_response_fields: data.source_response_fields.clone(),
            skip_invalid_records: data.skip_invalid_records,
            external_metadata: qwen_sft_external_metadata_from_paths(
                &data.external_metadata_paths,
            )?,
        };
        map.validate()?;
        Ok(map)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let has_system_source = self.has_system_source();
        if self.instruction.trim().is_empty() {
            bail!("data.instruction_field must not be empty");
        }
        if self.response.trim().is_empty() {
            bail!("data.response_field must not be empty");
        }
        if self
            .system
            .as_ref()
            .is_some_and(|field| field.trim().is_empty())
        {
            bail!("data.system_field must not be empty when set");
        }
        if self
            .chat_messages
            .as_ref()
            .is_some_and(|field| field.trim().is_empty())
        {
            bail!("data.chat_messages_field must not be empty when set");
        }
        if self.min_system_chars.is_some() && !has_system_source {
            bail!(
                "data.min_system_chars requires data.system_field or data.chat_messages_field to be set"
            );
        }
        if self.max_system_chars.is_some() && !has_system_source {
            bail!(
                "data.max_system_chars requires data.system_field or data.chat_messages_field to be set"
            );
        }
        if !self.system_contains_any.is_empty() && !has_system_source {
            bail!(
                "data.system_contains_any requires data.system_field or data.chat_messages_field to be set"
            );
        }
        if !self.system_excludes_any.is_empty() && !has_system_source {
            bail!(
                "data.system_excludes_any requires data.system_field or data.chat_messages_field to be set"
            );
        }
        if self.prompt_template.is_empty() {
            bail!("data.prompt_template must not be empty");
        }
        if self.prompt_with_input_template.is_empty() {
            bail!("data.prompt_with_input_template must not be empty");
        }
        if self.source_weights.iter().any(|weight| *weight == 0) {
            bail!("data.source_weights entries must be greater than zero");
        }
        if self.source_max_samples.iter().any(|limit| *limit == 0) {
            bail!("data.source_max_samples entries must be greater than zero");
        }
        for (name, values, required) in [
            (
                "data.source_instruction_fields",
                &self.source_instruction_fields,
                true,
            ),
            ("data.source_input_fields", &self.source_input_fields, false),
            (
                "data.source_response_fields",
                &self.source_response_fields,
                true,
            ),
        ] {
            if required && values.iter().any(|field| field.trim().is_empty()) {
                bail!("{name} entries must not be empty");
            }
        }
        if let Some(max_response_chars) = self.max_response_chars {
            if max_response_chars == 0 {
                bail!("data.max_response_chars must be greater than zero");
            }
            if max_response_chars < self.min_response_chars {
                bail!(
                    "data.max_response_chars must be greater than or equal to data.min_response_chars"
                );
            }
        }
        if self
            .instruction_contains_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            bail!("data.instruction_contains_any entries must not be empty");
        }
        if self
            .instruction_excludes_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            bail!("data.instruction_excludes_any entries must not be empty");
        }
        if self
            .response_contains_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            bail!("data.response_contains_any entries must not be empty");
        }
        if self
            .response_excludes_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            bail!("data.response_excludes_any entries must not be empty");
        }
        if self
            .input_contains_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            bail!("data.input_contains_any entries must not be empty");
        }
        if self
            .input_excludes_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            bail!("data.input_excludes_any entries must not be empty");
        }
        if self
            .system_contains_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            bail!("data.system_contains_any entries must not be empty");
        }
        if self
            .system_excludes_any
            .iter()
            .any(|needle| needle.is_empty())
        {
            bail!("data.system_excludes_any entries must not be empty");
        }
        if let Some(min_instruction_chars) = self.min_instruction_chars {
            if min_instruction_chars == 0 {
                bail!("data.min_instruction_chars must be greater than zero");
            }
        }
        if let Some(max_instruction_chars) = self.max_instruction_chars {
            if max_instruction_chars == 0 {
                bail!("data.max_instruction_chars must be greater than zero");
            }
            if self
                .min_instruction_chars
                .is_some_and(|min_instruction_chars| max_instruction_chars < min_instruction_chars)
            {
                bail!(
                    "data.max_instruction_chars must be greater than or equal to data.min_instruction_chars"
                );
            }
        }
        if let Some(min_input_chars) = self.min_input_chars {
            if min_input_chars == 0 {
                bail!("data.min_input_chars must be greater than zero");
            }
        }
        if let Some(max_input_chars) = self.max_input_chars {
            if max_input_chars == 0 {
                bail!("data.max_input_chars must be greater than zero");
            }
            if self
                .min_input_chars
                .is_some_and(|min_input_chars| max_input_chars < min_input_chars)
            {
                bail!("data.max_input_chars must be greater than or equal to data.min_input_chars");
            }
        }
        if let Some(min_system_chars) = self.min_system_chars {
            if min_system_chars == 0 {
                bail!("data.min_system_chars must be greater than zero");
            }
        }
        if let Some(max_system_chars) = self.max_system_chars {
            if max_system_chars == 0 {
                bail!("data.max_system_chars must be greater than zero");
            }
            if self
                .min_system_chars
                .is_some_and(|min_system_chars| max_system_chars < min_system_chars)
            {
                bail!(
                    "data.max_system_chars must be greater than or equal to data.min_system_chars"
                );
            }
        }
        if let Some(min_prompt_chars) = self.min_prompt_chars {
            if min_prompt_chars == 0 {
                bail!("data.min_prompt_chars must be greater than zero");
            }
        }
        if let Some(max_prompt_chars) = self.max_prompt_chars {
            if max_prompt_chars == 0 {
                bail!("data.max_prompt_chars must be greater than zero");
            }
            if self
                .min_prompt_chars
                .is_some_and(|min_prompt_chars| max_prompt_chars < min_prompt_chars)
            {
                bail!(
                    "data.max_prompt_chars must be greater than or equal to data.min_prompt_chars"
                );
            }
        }
        if let Some(min_sample_chars) = self.min_sample_chars {
            if min_sample_chars == 0 {
                bail!("data.min_sample_chars must be greater than zero");
            }
        }
        if let Some(max_sample_chars) = self.max_sample_chars {
            if max_sample_chars == 0 {
                bail!("data.max_sample_chars must be greater than zero");
            }
            if self
                .min_sample_chars
                .is_some_and(|min_sample_chars| max_sample_chars < min_sample_chars)
            {
                bail!(
                    "data.max_sample_chars must be greater than or equal to data.min_sample_chars"
                );
            }
        }
        for replacement in &self.field_replacements {
            if replacement.pattern.is_empty() {
                bail!("data.field_replacements pattern entries must not be empty");
            }
            if matches!(replacement.field, FieldReplacementTarget::System) && !has_system_source {
                bail!(
                    "data.field_replacements targeting system requires data.system_field or data.chat_messages_field to be set"
                );
            }
        }
        for replacement in &self.field_regex_replacements {
            if replacement.pattern.is_empty() {
                bail!("data.field_regex_replacements pattern entries must not be empty");
            }
            regex::Regex::new(&replacement.pattern).map_err(|error| {
                anyhow!(
                    "data.field_regex_replacements invalid regex pattern {:?}: {error}",
                    replacement.pattern
                )
            })?;
            if matches!(replacement.field, FieldReplacementTarget::System) && !has_system_source {
                bail!(
                    "data.field_regex_replacements targeting system requires data.system_field, data.chat_messages_field, or a system field_default to be set"
                );
            }
        }
        for filter in self
            .field_regex_contains_any
            .iter()
            .chain(self.field_regex_excludes_any.iter())
        {
            if filter.pattern.is_empty() {
                bail!("data field regex filter pattern entries must not be empty");
            }
            regex::Regex::new(&filter.pattern).map_err(|error| {
                anyhow!(
                    "data field regex filter invalid regex pattern {:?}: {error}",
                    filter.pattern
                )
            })?;
            if matches!(filter.field, FieldReplacementTarget::System) && !has_system_source {
                bail!(
                    "data field regex filters targeting system require data.system_field, data.chat_messages_field, or a system field_default to be set"
                );
            }
        }
        for default in &self.field_defaults {
            if default.value.is_empty() {
                bail!("data.field_defaults value entries must not be empty");
            }
        }
        for transform in &self.field_case_transforms {
            if matches!(transform.field, FieldReplacementTarget::System) && !has_system_source {
                bail!(
                    "data.field_case_transforms targeting system requires data.system_field, data.chat_messages_field, or a system field_default to be set"
                );
            }
        }
        for affix in &self.field_affixes {
            if affix.prefix.is_empty() && affix.suffix.is_empty() {
                bail!("data.field_affixes entries must set prefix, suffix, or both");
            }
            if matches!(affix.field, FieldReplacementTarget::System) && !has_system_source {
                bail!(
                    "data.field_affixes targeting system requires data.system_field, data.chat_messages_field, or a system field_default to be set"
                );
            }
        }
        for strip in &self.field_strips {
            if strip.prefix.is_empty() && strip.suffix.is_empty() {
                bail!("data.field_strips entries must set prefix, suffix, or both");
            }
            if matches!(strip.field, FieldReplacementTarget::System) && !has_system_source {
                bail!(
                    "data.field_strips targeting system requires data.system_field, data.chat_messages_field, or a system field_default to be set"
                );
            }
        }
        for split in &self.field_splits {
            if split.delimiter.is_empty() {
                bail!("data.field_splits delimiter entries must not be empty");
            }
            if matches!(split.field, FieldReplacementTarget::System) && !has_system_source {
                bail!(
                    "data.field_splits targeting system requires data.system_field, data.chat_messages_field, or a system field_default to be set"
                );
            }
        }
        for truncation in &self.field_truncations {
            if truncation.max_chars == 0 {
                bail!("data.field_truncations max_chars entries must be greater than zero");
            }
            if matches!(truncation.field, FieldReplacementTarget::System) && !has_system_source {
                bail!(
                    "data.field_truncations targeting system requires data.system_field, data.chat_messages_field, or a system field_default to be set"
                );
            }
        }
        for transform in &self.field_transforms {
            match transform.op {
                FieldTransformOp::Default => {
                    if matches!(transform.field, FieldReplacementTarget::All) {
                        bail!("data.field_transforms default op cannot target all");
                    }
                    if transform.value.is_empty() {
                        bail!("data.field_transforms default op requires non-empty value");
                    }
                }
                FieldTransformOp::Replace => {
                    if transform.pattern.is_empty() {
                        bail!("data.field_transforms replace op requires non-empty pattern");
                    }
                }
                FieldTransformOp::RegexReplace => {
                    if transform.pattern.is_empty() {
                        bail!("data.field_transforms regex_replace op requires non-empty pattern");
                    }
                    regex::Regex::new(&transform.pattern).map_err(|error| {
                        anyhow!(
                            "data.field_transforms invalid regex_replace pattern {:?}: {error}",
                            transform.pattern
                        )
                    })?;
                }
                FieldTransformOp::Case => {
                    if transform.case.is_none() {
                        bail!("data.field_transforms case op requires case");
                    }
                }
                FieldTransformOp::Affix => {
                    if transform.prefix.is_empty() && transform.suffix.is_empty() {
                        bail!("data.field_transforms affix op requires prefix, suffix, or both");
                    }
                }
                FieldTransformOp::Strip => {
                    if transform.prefix.is_empty() && transform.suffix.is_empty() {
                        bail!("data.field_transforms strip op requires prefix, suffix, or both");
                    }
                }
                FieldTransformOp::Split => {
                    if transform.delimiter.is_empty() {
                        bail!("data.field_transforms split op requires non-empty delimiter");
                    }
                    if transform.side.is_none() {
                        bail!("data.field_transforms split op requires side");
                    }
                }
                FieldTransformOp::Truncate => {
                    if transform.max_chars.is_none_or(|max_chars| max_chars == 0) {
                        bail!(
                            "data.field_transforms truncate op requires max_chars greater than zero"
                        );
                    }
                }
            }
            if matches!(transform.field, FieldReplacementTarget::System) && !has_system_source {
                bail!(
                    "data.field_transforms targeting system requires data.system_field, data.chat_messages_field, or a system field_default/field_transform default to be set"
                );
            }
        }
        Ok(())
    }

    pub(crate) fn has_system_source(&self) -> bool {
        self.system.is_some()
            || self.chat_messages.is_some()
            || self
                .field_defaults
                .iter()
                .any(|default| matches!(default.field, FieldDefaultTarget::System))
            || self.field_transforms.iter().any(|transform| {
                matches!(transform.field, FieldReplacementTarget::System)
                    && matches!(transform.op, FieldTransformOp::Default)
            })
    }
}

impl Default for QwenSftFieldMap {
    fn default() -> Self {
        Self {
            instruction: "instruction".to_string(),
            input: "input".to_string(),
            response: "response".to_string(),
            system: None,
            chat_messages: None,
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
            source_instruction_fields: Vec::new(),
            source_input_fields: Vec::new(),
            source_response_fields: Vec::new(),
            skip_invalid_records: false,
            external_metadata: Vec::new(),
        }
    }
}

pub(crate) fn qwen_sft_external_metadata_from_paths(
    paths: &[PathBuf],
) -> Result<Vec<QwenSftExternalMetadata>> {
    let mut metadata = Vec::new();
    for path in paths {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        metadata.push(QwenSftExternalMetadata {
            path: path.display().to_string(),
            contents,
        });
    }
    Ok(metadata)
}

#[derive(Clone)]
pub(crate) struct QwenSftDataset {
    pub(crate) samples: Vec<QwenSftTokenSample>,
    pub(crate) pad_token_id: i64,
    pub(crate) epoch_shuffle_seed: Option<u64>,
    pub(crate) source_files: Vec<String>,
    pub(crate) source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) fingerprint: String,
}

pub(crate) struct QwenSftBatch {
    pub(crate) input_ids: Tensor,
    pub(crate) target_mask: Tensor,
    pub(crate) prompt_tokens: Vec<usize>,
    pub(crate) response_tokens: Vec<usize>,
    pub(crate) masked_positions: usize,
    pub(crate) padding_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QwenSftDatasetSummary {
    pub(crate) samples: usize,
    pub(crate) total_tokens: usize,
    pub(crate) response_tokens: usize,
    pub(crate) masked_positions: usize,
    pub(crate) max_sequence_tokens: usize,
    pub(crate) source_files: Vec<String>,
    pub(crate) source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) fingerprint: String,
    pub(crate) shuffle: bool,
}

pub(crate) struct QwenSftTrainEvalDatasets {
    pub(crate) combined_summary: QwenSftDatasetSummary,
    pub(crate) train_dataset: QwenSftDataset,
    pub(crate) eval_dataset: QwenSftDataset,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct QwenSftStreamingSourceSummary {
    pub(crate) samples: usize,
    pub(crate) source_files: Vec<String>,
    pub(crate) source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct QwenSftStreamingCursorEntry {
    pub(crate) cursor: usize,
    pub(crate) epoch: usize,
    pub(crate) sample_offset: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenSftStreamingDataPlanSummary {
    pub(crate) config_path: String,
    pub(crate) data_paths: Vec<String>,
    pub(crate) eval_paths: Vec<String>,
    pub(crate) max_samples: Option<usize>,
    pub(crate) max_eval_samples: Option<usize>,
    pub(crate) train_split: f32,
    pub(crate) world_size: usize,
    pub(crate) local_batch_size: usize,
    pub(crate) global_batch_size: usize,
    pub(crate) train_steps: usize,
    pub(crate) required_batches: usize,
    pub(crate) data_cursor_start: usize,
    pub(crate) data_cursor_end: usize,
    pub(crate) data_cursor_next: usize,
    pub(crate) data_epoch_start: usize,
    pub(crate) data_epoch_end: usize,
    pub(crate) data_epoch_next: usize,
    pub(crate) data_sample_offset_start: usize,
    pub(crate) data_sample_offset_end: usize,
    pub(crate) data_sample_offset_next: usize,
    pub(crate) train_window_start_cursor: usize,
    pub(crate) train_window_end_cursor_exclusive: usize,
    pub(crate) train_window_sample_cursors: Vec<QwenSftStreamingCursorEntry>,
    pub(crate) dataset_total_samples: usize,
    pub(crate) dataset_train_samples: usize,
    pub(crate) dataset_eval_samples: usize,
    pub(crate) dataset_source_files: Vec<String>,
    pub(crate) dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) dataset_fingerprint: String,
    pub(crate) dataset_order_seed: u64,
    pub(crate) dataset_shuffle: bool,
    pub(crate) train_source_files: Vec<String>,
    pub(crate) train_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) train_fingerprint: String,
    pub(crate) eval_source_files: Vec<String>,
    pub(crate) eval_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) eval_fingerprint: String,
    pub(crate) streaming_index_cache_path: Option<String>,
    pub(crate) streaming_index_cache_hit: bool,
    pub(crate) streaming_index_cache_written: bool,
    pub(crate) tokenizer_loaded: bool,
    pub(crate) tokenized_samples_materialized: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenSftStreamingBatchPlanSummary {
    pub(crate) config_path: String,
    pub(crate) model_path: String,
    pub(crate) world_size: usize,
    pub(crate) local_batch_size: usize,
    pub(crate) global_batch_size: usize,
    pub(crate) train_steps: usize,
    pub(crate) required_batches: usize,
    pub(crate) train_batch_count: usize,
    pub(crate) data_cursor_start: usize,
    pub(crate) data_cursor_end: usize,
    pub(crate) data_cursor_next: usize,
    pub(crate) train_window_start_cursor: usize,
    pub(crate) train_window_end_cursor_exclusive: usize,
    pub(crate) train_window_sample_cursors: Vec<QwenSftStreamingCursorEntry>,
    pub(crate) dataset_total_samples: usize,
    pub(crate) dataset_train_samples: usize,
    pub(crate) dataset_eval_samples: usize,
    pub(crate) dataset_source_files: Vec<String>,
    pub(crate) dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) dataset_fingerprint: String,
    pub(crate) dataset_order_seed: u64,
    pub(crate) dataset_shuffle: bool,
    pub(crate) tokenizer_loaded: bool,
    pub(crate) tokenized_samples_materialized: bool,
    pub(crate) reference_tokenized_samples_materialized: bool,
    pub(crate) streaming_index_cache_path: Option<String>,
    pub(crate) streaming_index_cache_hit: bool,
    pub(crate) streaming_index_cache_written: bool,
    pub(crate) streaming_window_samples: usize,
    pub(crate) streaming_raw_samples_read: usize,
    pub(crate) streaming_raw_sample_indices: Vec<QwenSftRawSampleIndex>,
    pub(crate) batch_sequence_tokens: Vec<usize>,
    pub(crate) batch_masked_positions: Vec<usize>,
    pub(crate) batch_padding_tokens: Vec<usize>,
    pub(crate) batch_token_fingerprints: Vec<String>,
    pub(crate) materialized_input_max_delta: i64,
    pub(crate) materialized_mask_max_delta: f64,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenSftArrowSourceSummary {
    pub(crate) input: String,
    pub(crate) arrow_ipc_format: String,
    pub(crate) limit: usize,
    pub(crate) source_rows: usize,
    pub(crate) source_rows_exact: bool,
    pub(crate) columns: Vec<String>,
    pub(crate) column_map: QwenSftArrowColumnMap,
    pub(crate) samples: usize,
    pub(crate) source_files: Vec<String>,
    pub(crate) source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) fingerprint: String,
    pub(crate) tokenized_samples_materialized: bool,
    pub(crate) jsonl_materialized: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenSftArrowBatchPlanSummary {
    pub(crate) input: String,
    pub(crate) model_path: String,
    pub(crate) arrow_ipc_format: String,
    pub(crate) world_size: usize,
    pub(crate) local_batch_size: usize,
    pub(crate) global_batch_size: usize,
    pub(crate) train_steps: usize,
    pub(crate) required_batches: usize,
    pub(crate) train_batch_count: usize,
    pub(crate) data_cursor_start: usize,
    pub(crate) data_cursor_end: usize,
    pub(crate) data_cursor_next: usize,
    pub(crate) train_window_start_cursor: usize,
    pub(crate) train_window_end_cursor_exclusive: usize,
    pub(crate) train_window_sample_cursors: Vec<QwenSftStreamingCursorEntry>,
    pub(crate) dataset_total_samples: usize,
    pub(crate) dataset_train_samples: usize,
    pub(crate) dataset_eval_samples: usize,
    pub(crate) dataset_source_files: Vec<String>,
    pub(crate) dataset_source_sample_counts: Vec<QwenSftSourceSampleCount>,
    pub(crate) dataset_fingerprint: String,
    pub(crate) column_map: QwenSftArrowColumnMap,
    pub(crate) tokenizer_loaded: bool,
    pub(crate) tokenized_samples_materialized: bool,
    pub(crate) jsonl_materialized: bool,
    pub(crate) streaming_window_samples: usize,
    pub(crate) streaming_raw_samples_read: usize,
    pub(crate) streaming_raw_sample_indices: Vec<QwenSftRawSampleIndex>,
    pub(crate) batch_sequence_tokens: Vec<usize>,
    pub(crate) batch_masked_positions: Vec<usize>,
    pub(crate) batch_padding_tokens: Vec<usize>,
    pub(crate) batch_token_fingerprints: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct QwenSftArrowColumnMap {
    pub(crate) instruction: String,
    pub(crate) input: String,
    pub(crate) response: String,
}

pub fn qwen_session_dp_data_plan(
    config_path: &Path,
    world_size: usize,
    data_cursor_start: usize,
) -> Result<()> {
    if world_size == 0 {
        bail!("qwen session DP data plan requires world_size > 0");
    }
    let config = load_config(config_path)?;
    if config.model.architecture != "qwen_trainable_session" {
        bail!(
            "qwen session DP data plan expects architecture = qwen_trainable_session, got {}",
            config.model.architecture
        );
    }
    if config.parallel.data_parallel_size != world_size {
        bail!(
            "qwen session DP data plan world_size {world_size} does not match config data_parallel_size {}",
            config.parallel.data_parallel_size
        );
    }
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("qwen session DP data plan requires model.model_path")?;
    let model_path = resolve_qwen_model_path(model_path)?;
    let data_config = config
        .data
        .as_ref()
        .context("qwen session DP data plan requires [data]")?;
    if data_config.kind != RuntimeDataKind::InstructionJsonl {
        bail!("qwen session DP data plan supports kind = instruction_jsonl");
    }
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let field_map = QwenSftFieldMap::from_runtime_data(data_config)?;
    let datasets = qwen_sft_train_eval_datasets_from_paths(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.max_eval_samples,
        data_config.train_split,
        data_config.shuffle,
        config.run.seed,
        &field_map,
    )?;
    let dataset_summary = datasets.combined_summary;
    let train_dataset = datasets.train_dataset;
    let eval_dataset = datasets.eval_dataset;
    let local_batch_size = config
        .train
        .micro_batch_size
        .min(train_dataset.len())
        .max(1);
    let global_batch_size = local_batch_size * world_size;
    let train_steps = config.train.max_steps as usize;
    let required_batches = train_steps * global_batch_size + 1;
    let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
    let data_cursor_next = data_cursor_end;
    let (data_epoch_start, data_sample_offset_start) =
        qwen_data_epoch_and_offset(data_cursor_start, train_dataset.len())?;
    let (data_epoch_end, data_sample_offset_end) =
        qwen_data_epoch_and_offset(data_cursor_end, train_dataset.len())?;
    let (data_epoch_next, data_sample_offset_next) =
        qwen_data_epoch_and_offset(data_cursor_next, train_dataset.len())?;
    let summary = QwenSessionDpDataPlanSummary {
        config_path: config_path.display().to_string(),
        model_path: model_path.display().to_string(),
        world_size,
        local_batch_size,
        global_batch_size,
        train_steps,
        required_batches,
        data_cursor_start,
        data_cursor_end,
        data_cursor_next,
        data_epoch_start,
        data_epoch_end,
        data_epoch_next,
        data_sample_offset_start,
        data_sample_offset_end,
        data_sample_offset_next,
        dataset_total_samples: dataset_summary.samples,
        dataset_total_tokens: dataset_summary.total_tokens,
        dataset_train_samples: train_dataset.len(),
        dataset_eval_samples: eval_dataset.len(),
        dataset_source_files: dataset_summary.source_files,
        dataset_source_sample_counts: dataset_summary.source_sample_counts,
        dataset_fingerprint: dataset_summary.fingerprint,
        dataset_order_seed: config.run.seed,
        dataset_shuffle: dataset_summary.shuffle,
        streaming_train_batches: true,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

pub fn qwen_sft_streaming_data_plan(
    config_path: &Path,
    world_size: usize,
    data_cursor_start: usize,
) -> Result<()> {
    if world_size == 0 {
        bail!("qwen SFT streaming data plan requires world_size > 0");
    }
    let config = load_config(config_path)?;
    let data_config = config
        .data
        .as_ref()
        .context("qwen SFT streaming data plan requires [data]")?;
    if data_config.kind == RuntimeDataKind::InstructionArrow {
        return qwen_sft_arrow_streaming_data_plan(
            config_path,
            &config,
            data_config,
            world_size,
            data_cursor_start,
        );
    }
    if data_config.kind != RuntimeDataKind::InstructionJsonl {
        bail!("qwen SFT streaming data plan supports kind = instruction_jsonl");
    }

    let field_map = QwenSftFieldMap::from_runtime_data(data_config)?;
    let source_index_load = data_config
        .index_cache
        .as_deref()
        .map(|index_cache| {
            qwen_sft_streaming_source_index_with_cache(
                &data_config.paths,
                data_config.max_samples,
                Some(index_cache),
                &field_map,
            )
        })
        .transpose()?;
    let train_summary = if let Some(source_index_load) = &source_index_load {
        source_index_load
            .summary
            .clone()
            .context("qwen SFT streaming data-plan index cache did not return source summary")?
    } else {
        qwen_sft_streaming_source_summary(&data_config.paths, data_config.max_samples, &field_map)?
    };
    if let Some(source_index_load) = &source_index_load {
        let indexed_samples = source_index_load.index.samples.len();
        if indexed_samples != train_summary.samples {
            bail!(
                "qwen SFT streaming data-plan index cache sample count {} does not match summary {}",
                indexed_samples,
                train_summary.samples
            );
        }
    }
    let (
        dataset_total_samples,
        dataset_train_samples,
        dataset_eval_samples,
        dataset_source_files,
        dataset_source_sample_counts,
        dataset_fingerprint,
        eval_summary,
    ) = if data_config.eval_paths.is_empty() {
        let (train_samples, eval_samples) =
            qwen_sft_train_eval_sample_counts(train_summary.samples, data_config.train_split)?;
        (
            train_summary.samples,
            train_samples,
            eval_samples,
            train_summary.source_files.clone(),
            train_summary.source_sample_counts.clone(),
            train_summary.fingerprint.clone(),
            QwenSftStreamingSourceSummary {
                samples: eval_samples,
                source_files: Vec::new(),
                source_sample_counts: Vec::new(),
                fingerprint: String::new(),
            },
        )
    } else {
        let eval_field_map = qwen_sft_eval_field_map(&field_map);
        let eval_summary = qwen_sft_streaming_source_summary(
            &data_config.eval_paths,
            data_config.max_eval_samples,
            &eval_field_map,
        )?;
        let combined_source_files =
            qwen_merge_sft_source_files(&train_summary.source_files, &eval_summary.source_files);
        let combined_source_sample_counts = qwen_merge_sft_source_sample_counts(
            &train_summary.source_sample_counts,
            &eval_summary.source_sample_counts,
        );
        let combined_fingerprint = qwen_combine_sft_fingerprints(
            &combined_source_files,
            &train_summary.fingerprint,
            &eval_summary.fingerprint,
        );
        (
            train_summary.samples + eval_summary.samples,
            train_summary.samples,
            eval_summary.samples,
            combined_source_files,
            combined_source_sample_counts,
            combined_fingerprint,
            eval_summary,
        )
    };

    let local_batch_size = config
        .train
        .micro_batch_size
        .min(dataset_train_samples)
        .max(1);
    let global_batch_size = local_batch_size * world_size;
    let train_steps = config.train.max_steps as usize;
    let required_batches = train_steps * global_batch_size + 1;
    let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
    let data_cursor_next = data_cursor_end;
    let (data_epoch_start, data_sample_offset_start) =
        qwen_data_epoch_and_offset(data_cursor_start, dataset_train_samples)?;
    let (data_epoch_end, data_sample_offset_end) =
        qwen_data_epoch_and_offset(data_cursor_end, dataset_train_samples)?;
    let (data_epoch_next, data_sample_offset_next) =
        qwen_data_epoch_and_offset(data_cursor_next, dataset_train_samples)?;
    let train_window_sample_cursors = qwen_sft_streaming_cursor_window(
        data_cursor_start,
        required_batches,
        global_batch_size,
        dataset_train_samples,
    )?;
    let train_window_start_cursor = data_cursor_start;
    let train_window_end_cursor_exclusive = train_window_sample_cursors
        .last()
        .map(|entry| entry.cursor + 1)
        .unwrap_or(data_cursor_start);

    let summary = QwenSftStreamingDataPlanSummary {
        config_path: config_path.display().to_string(),
        data_paths: data_config
            .paths
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        eval_paths: data_config
            .eval_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        max_samples: data_config.max_samples,
        max_eval_samples: data_config.max_eval_samples,
        train_split: data_config.train_split,
        world_size,
        local_batch_size,
        global_batch_size,
        train_steps,
        required_batches,
        data_cursor_start,
        data_cursor_end,
        data_cursor_next,
        data_epoch_start,
        data_epoch_end,
        data_epoch_next,
        data_sample_offset_start,
        data_sample_offset_end,
        data_sample_offset_next,
        train_window_start_cursor,
        train_window_end_cursor_exclusive,
        train_window_sample_cursors,
        dataset_total_samples,
        dataset_train_samples,
        dataset_eval_samples,
        dataset_source_files,
        dataset_source_sample_counts,
        dataset_fingerprint,
        dataset_order_seed: config.run.seed,
        dataset_shuffle: data_config.shuffle,
        train_source_files: train_summary.source_files,
        train_source_sample_counts: train_summary.source_sample_counts,
        train_fingerprint: train_summary.fingerprint,
        eval_source_files: eval_summary.source_files,
        eval_source_sample_counts: eval_summary.source_sample_counts,
        eval_fingerprint: eval_summary.fingerprint,
        streaming_index_cache_path: data_config
            .index_cache
            .as_ref()
            .map(|path| path.display().to_string()),
        streaming_index_cache_hit: source_index_load
            .as_ref()
            .is_some_and(|load| load.cache_hit),
        streaming_index_cache_written: source_index_load
            .as_ref()
            .is_some_and(|load| load.cache_written),
        tokenizer_loaded: false,
        tokenized_samples_materialized: false,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

pub fn qwen_sft_streaming_batch_plan(
    config_path: &Path,
    world_size: usize,
    data_cursor_start: usize,
    index_cache: Option<&Path>,
) -> Result<()> {
    if world_size == 0 {
        bail!("qwen SFT streaming batch plan requires world_size > 0");
    }
    let config = load_config(config_path)?;
    let model_path = config
        .model
        .model_path
        .as_ref()
        .context("qwen SFT streaming batch plan requires model.model_path")?;
    let model_path = resolve_qwen_model_path(model_path)?;
    let data_config = config
        .data
        .as_ref()
        .context("qwen SFT streaming batch plan requires [data]")?;
    if data_config.kind == RuntimeDataKind::InstructionArrow {
        return qwen_sft_arrow_streaming_batch_plan(
            config_path,
            &config,
            data_config,
            &model_path,
            world_size,
            data_cursor_start,
            index_cache,
        );
    }
    if data_config.kind != RuntimeDataKind::InstructionJsonl {
        bail!("qwen SFT streaming batch plan supports kind = instruction_jsonl");
    }

    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let field_map = QwenSftFieldMap::from_runtime_data(data_config)?;
    let datasets = qwen_sft_train_eval_datasets_from_paths(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.max_eval_samples,
        data_config.train_split,
        data_config.shuffle,
        config.run.seed,
        &field_map,
    )?;
    let dataset_summary = datasets.combined_summary;
    let train_dataset = datasets.train_dataset;
    let eval_dataset = datasets.eval_dataset;

    let local_batch_size = config
        .train
        .micro_batch_size
        .min(train_dataset.len())
        .max(1);
    let global_batch_size = local_batch_size * world_size;
    let train_steps = config.train.max_steps as usize;
    let required_batches = train_steps * global_batch_size + 1;
    let train_batch_count = train_steps + 1;
    let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
    let data_cursor_next = data_cursor_end;
    let train_window_sample_cursors = qwen_sft_streaming_cursor_window(
        data_cursor_start,
        required_batches,
        global_batch_size,
        train_dataset.len(),
    )?;
    let train_window_end_cursor_exclusive = train_window_sample_cursors
        .last()
        .map(|entry| entry.cursor + 1)
        .unwrap_or(data_cursor_start);

    let streaming_window = qwen_sft_streaming_token_window_from_jsonl(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.train_split,
        data_config.shuffle,
        config.run.seed,
        data_cursor_start,
        train_window_sample_cursors.len(),
        index_cache,
        &field_map,
    )?;
    let mut batch_sequence_tokens = Vec::with_capacity(train_batch_count);
    let mut batch_masked_positions = Vec::with_capacity(train_batch_count);
    let mut batch_padding_tokens = Vec::with_capacity(train_batch_count);
    let mut batch_token_fingerprints = Vec::with_capacity(train_batch_count);
    let mut materialized_input_max_delta = 0_i64;
    let mut materialized_mask_max_delta = 0.0_f64;

    for batch_index in 0..train_batch_count {
        let offset = batch_index * global_batch_size;
        let end = offset + global_batch_size;
        let streaming_batch = qwen_sft_padded_batch(
            &streaming_window.samples[offset..end],
            train_dataset.pad_token_id,
        )?;
        let materialized_batch =
            train_dataset.padded_batch(data_cursor_start + offset, global_batch_size)?;
        batch_sequence_tokens.push(streaming_batch.input_ids.size()[1] as usize);
        batch_masked_positions.push(streaming_batch.masked_positions);
        batch_padding_tokens.push(streaming_batch.padding_tokens);
        batch_token_fingerprints.push(qwen_tensor_i64_fingerprint(&streaming_batch.input_ids)?);
        materialized_input_max_delta = materialized_input_max_delta.max(tensor_i64_max_abs_diff(
            &streaming_batch.input_ids,
            &materialized_batch.input_ids,
        )?);
        materialized_mask_max_delta = materialized_mask_max_delta.max(tensor_max_abs_diff(
            &streaming_batch.target_mask,
            &materialized_batch.target_mask,
        )?);
    }

    let summary = QwenSftStreamingBatchPlanSummary {
        config_path: config_path.display().to_string(),
        model_path: model_path.display().to_string(),
        world_size,
        local_batch_size,
        global_batch_size,
        train_steps,
        required_batches,
        train_batch_count,
        data_cursor_start,
        data_cursor_end,
        data_cursor_next,
        train_window_start_cursor: data_cursor_start,
        train_window_end_cursor_exclusive,
        train_window_sample_cursors,
        dataset_total_samples: dataset_summary.samples,
        dataset_train_samples: train_dataset.len(),
        dataset_eval_samples: eval_dataset.len(),
        dataset_source_files: dataset_summary.source_files,
        dataset_source_sample_counts: dataset_summary.source_sample_counts,
        dataset_fingerprint: dataset_summary.fingerprint,
        dataset_order_seed: config.run.seed,
        dataset_shuffle: dataset_summary.shuffle,
        tokenizer_loaded: true,
        tokenized_samples_materialized: true,
        reference_tokenized_samples_materialized: true,
        streaming_index_cache_path: index_cache.map(|path| path.display().to_string()),
        streaming_index_cache_hit: streaming_window.source_index_cache_hit,
        streaming_index_cache_written: streaming_window.source_index_cache_written,
        streaming_window_samples: streaming_window.samples.len(),
        streaming_raw_samples_read: streaming_window.raw_samples_read,
        streaming_raw_sample_indices: streaming_window.raw_sample_indices,
        batch_sequence_tokens,
        batch_masked_positions,
        batch_padding_tokens,
        batch_token_fingerprints,
        materialized_input_max_delta,
        materialized_mask_max_delta,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

pub fn qwen_sft_arrow_source_summary(
    input: &Path,
    limit: usize,
    instruction_column: &str,
    input_column: &str,
    response_column: &str,
) -> Result<()> {
    if limit == 0 {
        bail!("qwen SFT Arrow source summary requires limit > 0");
    }
    if !input.is_file() {
        bail!(
            "qwen SFT Arrow source input must be a file: {}",
            input.display()
        );
    }
    let field_map = QwenSftFieldMap {
        instruction: instruction_column.to_string(),
        input: input_column.to_string(),
        response: response_column.to_string(),
        ..QwenSftFieldMap::default()
    };
    field_map.validate()?;
    let summary = qwen_sft_arrow_source_summary_from_ipc(input, limit, &field_map)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

pub(crate) fn qwen_sft_arrow_validate_config_scope(
    data_config: &RuntimeDataConfig,
    context: &str,
) -> Result<()> {
    for path in &data_config.paths {
        if !path.is_file() {
            bail!(
                "{context} data.paths entry must be an Arrow IPC file: {}",
                path.display()
            );
        }
    }
    if data_config.eval_paths.len() > 1 {
        bail!("{context} supports at most one data.eval_paths Arrow IPC file for now");
    }
    if !data_config.eval_paths.is_empty() {
        for path in &data_config.eval_paths {
            if !path.is_file() {
                bail!(
                    "{context} data.eval_paths entry must be an Arrow IPC file: {}",
                    path.display()
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn qwen_sft_arrow_has_sample_bounds(data_config: &RuntimeDataConfig) -> bool {
    data_config.max_samples.is_some() || !data_config.source_max_samples.is_empty()
}

pub(crate) fn qwen_sft_arrow_require_cache_or_bounds(
    data_config: &RuntimeDataConfig,
    index_cache: Option<&Path>,
    context: &str,
) -> Result<()> {
    if qwen_sft_arrow_has_sample_bounds(data_config) || index_cache.is_some() {
        return Ok(());
    }
    bail!(
        "{context} instruction_arrow requires data.max_samples, data.source_max_samples, or an index cache to avoid unbounded Arrow source scans"
    )
}

pub(crate) fn qwen_sft_arrow_dataset_from_paths(
    tokenizer: &Tokenizer,
    paths: &[PathBuf],
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftDataset> {
    let arrow = qwen_sft_arrow_examples_from_paths_with_limit(paths, max_samples, field_map)?;
    let samples = arrow
        .examples
        .iter()
        .map(|example| qwen_sft_token_sample(tokenizer, example, field_map))
        .collect::<Result<Vec<_>>>()?;
    Ok(QwenSftDataset {
        samples,
        pad_token_id: qwen_pad_token_id(tokenizer),
        epoch_shuffle_seed: None,
        source_files: arrow.source_files,
        source_sample_counts: arrow.source_sample_counts,
        fingerprint: arrow.fingerprint,
    })
}

pub(crate) fn qwen_sft_arrow_dataset_from_ipc(
    tokenizer: &Tokenizer,
    path: &Path,
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftDataset> {
    qwen_sft_arrow_dataset_from_paths(tokenizer, &[path.to_path_buf()], max_samples, field_map)
}

pub(crate) fn qwen_sft_arrow_plan_data(
    tokenizer: &Tokenizer,
    data_config: &RuntimeDataConfig,
    seed: u64,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftArrowPlanData> {
    let train_dataset = qwen_sft_arrow_dataset_from_paths(
        tokenizer,
        &data_config.paths,
        data_config.max_samples,
        field_map,
    )?;
    if data_config.eval_paths.is_empty() {
        let dataset = qwen_apply_sft_shuffle(train_dataset, data_config.shuffle, seed);
        let dataset_summary = dataset.summary();
        let (train_dataset, eval_dataset) = dataset.train_eval_split(data_config.train_split)?;
        return Ok(QwenSftArrowPlanData {
            dataset_summary,
            train_dataset,
            eval_dataset,
        });
    }

    let eval_input = data_config
        .eval_paths
        .first()
        .context("instruction_arrow eval_paths expected one entry")?;
    let eval_field_map = qwen_sft_eval_field_map(field_map);
    let eval_dataset = qwen_sft_arrow_dataset_from_ipc(
        tokenizer,
        eval_input,
        data_config.max_eval_samples,
        &eval_field_map,
    )?;
    let train_summary = train_dataset.summary();
    let eval_summary = eval_dataset.summary();
    let combined_source_files =
        qwen_merge_sft_source_files(&train_summary.source_files, &eval_summary.source_files);
    let combined_source_sample_counts = qwen_merge_sft_source_sample_counts(
        &train_summary.source_sample_counts,
        &eval_summary.source_sample_counts,
    );
    let combined_fingerprint = qwen_combine_sft_fingerprints(
        &combined_source_files,
        &train_summary.fingerprint,
        &eval_summary.fingerprint,
    );
    let train_dataset = qwen_apply_sft_shuffle(
        train_dataset.with_source_metadata(
            combined_source_files.clone(),
            combined_source_sample_counts.clone(),
            combined_fingerprint.clone(),
        ),
        data_config.shuffle,
        seed,
    );
    let eval_dataset = eval_dataset.with_source_metadata(
        combined_source_files.clone(),
        combined_source_sample_counts.clone(),
        combined_fingerprint.clone(),
    );
    Ok(QwenSftArrowPlanData {
        dataset_summary: QwenSftDatasetSummary {
            samples: train_summary.samples + eval_summary.samples,
            total_tokens: train_summary.total_tokens + eval_summary.total_tokens,
            response_tokens: train_summary.response_tokens + eval_summary.response_tokens,
            masked_positions: train_summary.masked_positions + eval_summary.masked_positions,
            max_sequence_tokens: train_summary
                .max_sequence_tokens
                .max(eval_summary.max_sequence_tokens),
            source_files: combined_source_files,
            source_sample_counts: combined_source_sample_counts,
            fingerprint: combined_fingerprint,
            shuffle: data_config.shuffle,
        },
        train_dataset,
        eval_dataset,
    })
}

pub(crate) fn qwen_sft_arrow_streaming_data_plan(
    config_path: &Path,
    config: &Config,
    data_config: &RuntimeDataConfig,
    world_size: usize,
    data_cursor_start: usize,
) -> Result<()> {
    let context = "qwen SFT Arrow streaming data plan";
    qwen_sft_arrow_validate_config_scope(data_config, context)?;
    qwen_sft_arrow_require_cache_or_bounds(
        data_config,
        data_config.index_cache.as_deref(),
        context,
    )?;
    let field_map = QwenSftFieldMap::from_runtime_data(data_config)?;
    let source_index_load = data_config
        .index_cache
        .as_deref()
        .map(|index_cache| {
            qwen_sft_arrow_source_index_with_cache(
                &data_config.paths,
                data_config.max_samples,
                Some(index_cache),
                &field_map,
            )
        })
        .transpose()?;
    let train_summary = if let Some(source_index_load) = &source_index_load {
        source_index_load
            .summary
            .clone()
            .context("qwen SFT Arrow data-plan index cache did not return source summary")?
    } else {
        qwen_sft_arrow_source_summary_from_paths(
            &data_config.paths,
            data_config.max_samples,
            &field_map,
        )?
    };
    if let Some(source_index_load) = &source_index_load {
        let indexed_samples = source_index_load.index.samples.len();
        if indexed_samples != train_summary.samples {
            bail!(
                "qwen SFT Arrow data-plan index cache sample count {} does not match summary {}",
                indexed_samples,
                train_summary.samples
            );
        }
    }
    let train_source_files = train_summary.source_files.clone();
    let train_source_sample_counts = train_summary.source_sample_counts.clone();
    let (
        dataset_total_samples,
        dataset_train_samples,
        dataset_eval_samples,
        dataset_source_files,
        dataset_source_sample_counts,
        dataset_fingerprint,
        eval_source_files,
        eval_source_sample_counts,
        eval_fingerprint,
    ) = if let Some(eval_input) = data_config.eval_paths.first() {
        let eval_field_map = qwen_sft_eval_field_map(&field_map);
        let eval_summary = qwen_sft_arrow_source_summary_from_paths(
            std::slice::from_ref(eval_input),
            data_config.max_eval_samples,
            &eval_field_map,
        )?;
        let eval_source_files = eval_summary.source_files.clone();
        let eval_source_sample_counts = eval_summary.source_sample_counts.clone();
        let combined_source_files =
            qwen_merge_sft_source_files(&train_source_files, &eval_source_files);
        let combined_source_sample_counts = qwen_merge_sft_source_sample_counts(
            &train_source_sample_counts,
            &eval_source_sample_counts,
        );
        let combined_fingerprint = qwen_combine_sft_fingerprints(
            &combined_source_files,
            &train_summary.fingerprint,
            &eval_summary.fingerprint,
        );
        (
            train_summary.samples + eval_summary.samples,
            train_summary.samples,
            eval_summary.samples,
            combined_source_files,
            combined_source_sample_counts,
            combined_fingerprint,
            eval_source_files,
            eval_source_sample_counts,
            eval_summary.fingerprint,
        )
    } else {
        let (train_samples, eval_samples) =
            qwen_sft_train_eval_sample_counts(train_summary.samples, data_config.train_split)?;
        (
            train_summary.samples,
            train_samples,
            eval_samples,
            train_source_files.clone(),
            train_source_sample_counts.clone(),
            train_summary.fingerprint.clone(),
            Vec::new(),
            Vec::new(),
            String::new(),
        )
    };

    let local_batch_size = config
        .train
        .micro_batch_size
        .min(dataset_train_samples)
        .max(1);
    let global_batch_size = local_batch_size * world_size;
    let train_steps = config.train.max_steps as usize;
    let required_batches = train_steps * global_batch_size + 1;
    let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
    let data_cursor_next = data_cursor_end;
    let (data_epoch_start, data_sample_offset_start) =
        qwen_data_epoch_and_offset(data_cursor_start, dataset_train_samples)?;
    let (data_epoch_end, data_sample_offset_end) =
        qwen_data_epoch_and_offset(data_cursor_end, dataset_train_samples)?;
    let (data_epoch_next, data_sample_offset_next) =
        qwen_data_epoch_and_offset(data_cursor_next, dataset_train_samples)?;
    let train_window_sample_cursors = qwen_sft_streaming_cursor_window(
        data_cursor_start,
        required_batches,
        global_batch_size,
        dataset_train_samples,
    )?;
    let train_window_start_cursor = data_cursor_start;
    let train_window_end_cursor_exclusive = train_window_sample_cursors
        .last()
        .map(|entry| entry.cursor + 1)
        .unwrap_or(data_cursor_start);

    let summary = QwenSftStreamingDataPlanSummary {
        config_path: config_path.display().to_string(),
        data_paths: data_config
            .paths
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        eval_paths: data_config
            .eval_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        max_samples: data_config.max_samples,
        max_eval_samples: data_config.max_eval_samples,
        train_split: data_config.train_split,
        world_size,
        local_batch_size,
        global_batch_size,
        train_steps,
        required_batches,
        data_cursor_start,
        data_cursor_end,
        data_cursor_next,
        data_epoch_start,
        data_epoch_end,
        data_epoch_next,
        data_sample_offset_start,
        data_sample_offset_end,
        data_sample_offset_next,
        train_window_start_cursor,
        train_window_end_cursor_exclusive,
        train_window_sample_cursors,
        dataset_total_samples,
        dataset_train_samples,
        dataset_eval_samples,
        dataset_source_files: dataset_source_files.clone(),
        dataset_source_sample_counts: dataset_source_sample_counts.clone(),
        dataset_fingerprint,
        dataset_order_seed: config.run.seed,
        dataset_shuffle: data_config.shuffle,
        train_source_files,
        train_source_sample_counts,
        train_fingerprint: train_summary.fingerprint,
        eval_source_files,
        eval_source_sample_counts,
        eval_fingerprint,
        streaming_index_cache_path: data_config
            .index_cache
            .as_ref()
            .map(|path| path.display().to_string()),
        streaming_index_cache_hit: source_index_load
            .as_ref()
            .is_some_and(|load| load.cache_hit),
        streaming_index_cache_written: source_index_load
            .as_ref()
            .is_some_and(|load| load.cache_written),
        tokenizer_loaded: false,
        tokenized_samples_materialized: false,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

pub(crate) fn qwen_sft_arrow_streaming_batch_plan(
    config_path: &Path,
    config: &Config,
    data_config: &RuntimeDataConfig,
    model_path: &Path,
    world_size: usize,
    data_cursor_start: usize,
    index_cache: Option<&Path>,
) -> Result<()> {
    let context = "qwen SFT Arrow streaming batch plan";
    qwen_sft_arrow_validate_config_scope(data_config, context)?;
    let effective_index_cache = index_cache.or(data_config.index_cache.as_deref());
    qwen_sft_arrow_require_cache_or_bounds(data_config, effective_index_cache, context)?;
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let field_map = QwenSftFieldMap::from_runtime_data(data_config)?;
    let plan_data = qwen_sft_arrow_plan_data(&tokenizer, data_config, config.run.seed, &field_map)?;
    let dataset_summary = plan_data.dataset_summary;
    let train_dataset = plan_data.train_dataset;
    let eval_dataset = plan_data.eval_dataset;

    let local_batch_size = config
        .train
        .micro_batch_size
        .min(train_dataset.len())
        .max(1);
    let global_batch_size = local_batch_size * world_size;
    let train_steps = config.train.max_steps as usize;
    let required_batches = train_steps * global_batch_size + 1;
    let train_batch_count = train_steps + 1;
    let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
    let data_cursor_next = data_cursor_end;
    let train_window_sample_cursors = qwen_sft_streaming_cursor_window(
        data_cursor_start,
        required_batches,
        global_batch_size,
        train_dataset.len(),
    )?;
    let train_window_end_cursor_exclusive = train_window_sample_cursors
        .last()
        .map(|entry| entry.cursor + 1)
        .unwrap_or(data_cursor_start);

    let streaming_window = qwen_sft_arrow_streaming_token_window(
        &tokenizer,
        &data_config.paths,
        &data_config.eval_paths,
        data_config.max_samples,
        data_config.train_split,
        data_config.shuffle,
        config.run.seed,
        data_cursor_start,
        train_window_sample_cursors.len(),
        effective_index_cache,
        &field_map,
    )?;
    let mut batch_sequence_tokens = Vec::with_capacity(train_batch_count);
    let mut batch_masked_positions = Vec::with_capacity(train_batch_count);
    let mut batch_padding_tokens = Vec::with_capacity(train_batch_count);
    let mut batch_token_fingerprints = Vec::with_capacity(train_batch_count);
    let mut materialized_input_max_delta = 0_i64;
    let mut materialized_mask_max_delta = 0.0_f64;

    for batch_index in 0..train_batch_count {
        let offset = batch_index * global_batch_size;
        let end = offset + global_batch_size;
        let streaming_batch = qwen_sft_padded_batch(
            &streaming_window.samples[offset..end],
            train_dataset.pad_token_id,
        )?;
        let materialized_batch =
            train_dataset.padded_batch(data_cursor_start + offset, global_batch_size)?;
        batch_sequence_tokens.push(streaming_batch.input_ids.size()[1] as usize);
        batch_masked_positions.push(streaming_batch.masked_positions);
        batch_padding_tokens.push(streaming_batch.padding_tokens);
        batch_token_fingerprints.push(qwen_tensor_i64_fingerprint(&streaming_batch.input_ids)?);
        materialized_input_max_delta = materialized_input_max_delta.max(tensor_i64_max_abs_diff(
            &streaming_batch.input_ids,
            &materialized_batch.input_ids,
        )?);
        materialized_mask_max_delta = materialized_mask_max_delta.max(tensor_max_abs_diff(
            &streaming_batch.target_mask,
            &materialized_batch.target_mask,
        )?);
    }

    let summary = QwenSftStreamingBatchPlanSummary {
        config_path: config_path.display().to_string(),
        model_path: model_path.display().to_string(),
        world_size,
        local_batch_size,
        global_batch_size,
        train_steps,
        required_batches,
        train_batch_count,
        data_cursor_start,
        data_cursor_end,
        data_cursor_next,
        train_window_start_cursor: data_cursor_start,
        train_window_end_cursor_exclusive,
        train_window_sample_cursors,
        dataset_total_samples: dataset_summary.samples,
        dataset_train_samples: train_dataset.len(),
        dataset_eval_samples: eval_dataset.len(),
        dataset_source_files: dataset_summary.source_files,
        dataset_source_sample_counts: dataset_summary.source_sample_counts,
        dataset_fingerprint: dataset_summary.fingerprint,
        dataset_order_seed: config.run.seed,
        dataset_shuffle: dataset_summary.shuffle,
        tokenizer_loaded: true,
        tokenized_samples_materialized: false,
        reference_tokenized_samples_materialized: true,
        streaming_index_cache_path: effective_index_cache.map(|path| path.display().to_string()),
        streaming_index_cache_hit: streaming_window.source_index_cache_hit,
        streaming_index_cache_written: streaming_window.source_index_cache_written,
        streaming_window_samples: streaming_window.samples.len(),
        streaming_raw_samples_read: streaming_window.raw_samples_read,
        streaming_raw_sample_indices: Vec::new(),
        batch_sequence_tokens,
        batch_masked_positions,
        batch_padding_tokens,
        batch_token_fingerprints,
        materialized_input_max_delta,
        materialized_mask_max_delta,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

pub fn qwen_sft_arrow_batch_plan(
    input: &Path,
    model_path: &Path,
    world_size: usize,
    local_batch_size: usize,
    train_steps: usize,
    data_cursor_start: usize,
    limit: usize,
    train_split: f32,
    instruction_column: &str,
    input_column: &str,
    response_column: &str,
    prompt_template: &str,
    prompt_with_input_template: &str,
) -> Result<()> {
    if world_size == 0 {
        bail!("qwen SFT Arrow batch plan requires world_size > 0");
    }
    if local_batch_size == 0 {
        bail!("qwen SFT Arrow batch plan requires local_batch_size > 0");
    }
    if limit == 0 {
        bail!("qwen SFT Arrow batch plan requires limit > 0");
    }
    if !input.is_file() {
        bail!(
            "qwen SFT Arrow source input must be a file: {}",
            input.display()
        );
    }
    let model_path = resolve_qwen_model_path(model_path)?;
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|error| anyhow!("failed to load tokenizer: {error}"))?;
    let field_map = QwenSftFieldMap {
        instruction: instruction_column.to_string(),
        input: input_column.to_string(),
        response: response_column.to_string(),
        prompt_template: qwen_decode_cli_template_escapes(prompt_template),
        prompt_with_input_template: qwen_decode_cli_template_escapes(prompt_with_input_template),
        ..QwenSftFieldMap::default()
    };
    field_map.validate()?;
    let arrow = qwen_sft_arrow_examples_from_ipc(input, Some(limit), &field_map)?;
    let (dataset_train_samples, dataset_eval_samples) =
        qwen_sft_train_eval_sample_counts(arrow.examples.len(), train_split)?;
    let global_batch_size = local_batch_size * world_size;
    let required_batches = train_steps * global_batch_size + 1;
    let train_batch_count = train_steps + 1;
    let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
    let data_cursor_next = data_cursor_end;
    let train_window_sample_cursors = qwen_sft_streaming_cursor_window(
        data_cursor_start,
        required_batches,
        global_batch_size,
        dataset_train_samples,
    )?;
    let train_window_end_cursor_exclusive = train_window_sample_cursors
        .last()
        .map(|entry| entry.cursor + 1)
        .unwrap_or(data_cursor_start);
    let train_examples = &arrow.examples[..dataset_train_samples];
    let window_examples = train_window_sample_cursors
        .iter()
        .map(|cursor| train_examples[cursor.sample_offset].clone())
        .collect::<Vec<_>>();
    let samples = window_examples
        .iter()
        .map(|example| qwen_sft_token_sample(&tokenizer, example, &field_map))
        .collect::<Result<Vec<_>>>()?;
    let pad_token_id = qwen_pad_token_id(&tokenizer);
    let mut batch_sequence_tokens = Vec::with_capacity(train_batch_count);
    let mut batch_masked_positions = Vec::with_capacity(train_batch_count);
    let mut batch_padding_tokens = Vec::with_capacity(train_batch_count);
    let mut batch_token_fingerprints = Vec::with_capacity(train_batch_count);
    for batch_index in 0..train_batch_count {
        let offset = batch_index * global_batch_size;
        let end = offset + global_batch_size;
        let batch = qwen_sft_padded_batch(&samples[offset..end], pad_token_id)?;
        batch_sequence_tokens.push(batch.input_ids.size()[1] as usize);
        batch_masked_positions.push(batch.masked_positions);
        batch_padding_tokens.push(batch.padding_tokens);
        batch_token_fingerprints.push(qwen_tensor_i64_fingerprint(&batch.input_ids)?);
    }
    let input = input.display().to_string();
    let summary = QwenSftArrowBatchPlanSummary {
        input: input.clone(),
        model_path: model_path.display().to_string(),
        arrow_ipc_format: arrow.arrow_ipc_format,
        world_size,
        local_batch_size,
        global_batch_size,
        train_steps,
        required_batches,
        train_batch_count,
        data_cursor_start,
        data_cursor_end,
        data_cursor_next,
        train_window_start_cursor: data_cursor_start,
        train_window_end_cursor_exclusive,
        train_window_sample_cursors,
        dataset_total_samples: arrow.examples.len(),
        dataset_train_samples,
        dataset_eval_samples,
        dataset_source_files: vec![input.clone()],
        dataset_source_sample_counts: vec![QwenSftSourceSampleCount {
            path: input.clone(),
            samples: arrow.examples.len(),
        }],
        dataset_fingerprint: arrow.fingerprint,
        column_map: QwenSftArrowColumnMap {
            instruction: field_map.instruction,
            input: field_map.input,
            response: field_map.response,
        },
        tokenizer_loaded: true,
        tokenized_samples_materialized: false,
        jsonl_materialized: false,
        streaming_window_samples: samples.len(),
        streaming_raw_samples_read: window_examples.len(),
        streaming_raw_sample_indices: Vec::new(),
        batch_sequence_tokens,
        batch_masked_positions,
        batch_padding_tokens,
        batch_token_fingerprints,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

impl QwenSftDataset {
    pub(crate) fn from_instruction_pairs(
        tokenizer: &Tokenizer,
        examples: &[QwenSftExample],
    ) -> Result<Self> {
        if examples.is_empty() {
            bail!("SFT dataset must contain at least one example");
        }
        let samples = examples
            .iter()
            .map(|example| qwen_sft_token_sample(tokenizer, example, &QwenSftFieldMap::default()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            samples,
            pad_token_id: qwen_pad_token_id(tokenizer),
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: qwen_sft_dataset_fingerprint(&[], examples, &QwenSftFieldMap::default()),
        })
    }

    pub(crate) fn from_jsonl_paths_with_limit(
        tokenizer: &Tokenizer,
        paths: &[PathBuf],
        max_samples: Option<usize>,
        field_map: &QwenSftFieldMap,
    ) -> Result<Self> {
        let example_set =
            qwen_sft_examples_from_jsonl_paths_with_limit(paths, max_samples, field_map)?;
        if example_set.examples.is_empty() {
            bail!("SFT dataset must contain at least one example");
        }
        let samples = example_set
            .examples
            .iter()
            .map(|example| qwen_sft_token_sample(tokenizer, example, field_map))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            samples,
            pad_token_id: qwen_pad_token_id(tokenizer),
            epoch_shuffle_seed: None,
            source_files: example_set.source_files,
            source_sample_counts: example_set.source_sample_counts,
            fingerprint: example_set.fingerprint,
        })
    }

    pub(crate) fn train_eval_split(&self, train_split: f32) -> Result<(Self, Self)> {
        if !(0.0..1.0).contains(&train_split) {
            bail!("SFT train_split must be in (0, 1)");
        }
        if self.samples.len() < 2 {
            bail!("SFT train/eval split requires at least two samples");
        }
        let (split_at, _) = qwen_sft_train_eval_sample_counts(self.samples.len(), train_split)?;
        Ok((
            Self {
                samples: self.samples[..split_at].to_vec(),
                pad_token_id: self.pad_token_id,
                epoch_shuffle_seed: self.epoch_shuffle_seed,
                source_files: self.source_files.clone(),
                source_sample_counts: self.source_sample_counts.clone(),
                fingerprint: self.fingerprint.clone(),
            },
            Self {
                samples: self.samples[split_at..].to_vec(),
                pad_token_id: self.pad_token_id,
                epoch_shuffle_seed: self.epoch_shuffle_seed,
                source_files: self.source_files.clone(),
                source_sample_counts: self.source_sample_counts.clone(),
                fingerprint: self.fingerprint.clone(),
            },
        ))
    }

    pub(crate) fn shuffle_by_seed(mut self, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        self.samples.shuffle(&mut rng);
        self.epoch_shuffle_seed = Some(seed);
        self
    }

    pub(crate) fn summary(&self) -> QwenSftDatasetSummary {
        QwenSftDatasetSummary {
            samples: self.samples.len(),
            total_tokens: self
                .samples
                .iter()
                .map(|sample| sample.token_ids.len())
                .sum(),
            response_tokens: self
                .samples
                .iter()
                .map(|sample| sample.response_tokens)
                .sum(),
            masked_positions: self
                .samples
                .iter()
                .map(|sample| sample.masked_positions)
                .sum(),
            max_sequence_tokens: self
                .samples
                .iter()
                .map(|sample| sample.token_ids.len())
                .max()
                .unwrap_or(0),
            source_files: self.source_files.clone(),
            source_sample_counts: self.source_sample_counts.clone(),
            fingerprint: self.fingerprint.clone(),
            shuffle: self.epoch_shuffle_seed.is_some(),
        }
    }

    pub(crate) fn sample_at_cursor(&self, cursor: usize) -> Result<QwenSftTokenSample> {
        if self.samples.is_empty() {
            bail!("SFT dataset must contain at least one sample");
        }
        let dataset_len = self.samples.len();
        let epoch = cursor / dataset_len;
        let offset = cursor % dataset_len;
        let index = if let Some(seed) = self.epoch_shuffle_seed {
            qwen_epoch_permutation_index(dataset_len, seed, epoch, offset)
        } else {
            offset
        };
        self.samples
            .get(index)
            .cloned()
            .ok_or_else(|| anyhow!("SFT cursor resolved out-of-range sample index {index}"))
    }

    pub(crate) fn padded_batch(&self, start: usize, batch_size: usize) -> Result<QwenSftBatch> {
        if batch_size == 0 {
            bail!("SFT batch size must be positive");
        }
        let samples = (0..batch_size)
            .map(|offset| self.sample_at_cursor(start + offset))
            .collect::<Result<Vec<_>>>()?;
        qwen_sft_padded_batch(&samples, self.pad_token_id)
    }

    pub(crate) fn len(&self) -> usize {
        self.samples.len()
    }

    pub(crate) fn with_source_metadata(
        mut self,
        source_files: Vec<String>,
        source_sample_counts: Vec<QwenSftSourceSampleCount>,
        fingerprint: String,
    ) -> Self {
        self.source_files = source_files;
        self.source_sample_counts = source_sample_counts;
        self.fingerprint = fingerprint;
        self
    }
}

pub(crate) fn qwen_sft_train_eval_sample_counts(
    total_samples: usize,
    train_split: f32,
) -> Result<(usize, usize)> {
    if !(0.0..1.0).contains(&train_split) {
        bail!("SFT train_split must be in (0, 1)");
    }
    if total_samples < 2 {
        bail!("SFT train/eval split requires at least two samples");
    }
    let train_samples = ((total_samples as f32) * train_split).floor() as usize;
    let train_samples = train_samples.clamp(1, total_samples - 1);
    Ok((train_samples, total_samples - train_samples))
}

pub(crate) fn qwen_sft_streaming_cursor_window(
    data_cursor_start: usize,
    required_batches: usize,
    global_batch_size: usize,
    train_sample_count: usize,
) -> Result<Vec<QwenSftStreamingCursorEntry>> {
    if required_batches == 0 {
        bail!("SFT streaming cursor window requires at least one batch");
    }
    if global_batch_size == 0 {
        bail!("SFT streaming cursor window requires global_batch_size > 0");
    }
    if train_sample_count == 0 {
        bail!("SFT streaming cursor window requires at least one training sample");
    }
    let needed_samples = required_batches + global_batch_size - 1;
    (0..needed_samples)
        .map(|relative| {
            let cursor = data_cursor_start + relative;
            let (epoch, sample_offset) = qwen_data_epoch_and_offset(cursor, train_sample_count)?;
            Ok(QwenSftStreamingCursorEntry {
                cursor,
                epoch,
                sample_offset,
            })
        })
        .collect()
}

pub(crate) fn qwen_sft_streaming_index_cache_path(base_dir: &Path, label: &str) -> PathBuf {
    base_dir.join(format!("{label}-offset-index.json"))
}

pub(crate) fn qwen_sft_rank_index_cache_path(path: &Path, rank: usize) -> PathBuf {
    let extension = path.extension().and_then(|extension| extension.to_str());
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("offset-index");
    let file_name = match extension {
        Some(extension) if !extension.is_empty() => format!("{stem}.rank-{rank}.{extension}"),
        _ => format!("{stem}.rank-{rank}"),
    };
    path.with_file_name(file_name)
}

pub(crate) fn qwen_sft_streaming_token_window_from_jsonl(
    tokenizer: &Tokenizer,
    paths: &[PathBuf],
    eval_paths: &[PathBuf],
    max_samples: Option<usize>,
    train_split: f32,
    shuffle: bool,
    seed: u64,
    data_cursor_start: usize,
    window_samples: usize,
    index_cache: Option<&Path>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftStreamingTokenWindow> {
    if window_samples == 0 {
        bail!("SFT streaming token window requires at least one sample");
    }
    let regex_plan = QwenSftRegexPlan::compile(field_map)?;
    let source_index_load =
        qwen_sft_streaming_source_index_with_cache(paths, max_samples, index_cache, field_map)?;
    let source_index = source_index_load.index;
    let train_samples = if eval_paths.is_empty() {
        let (train_samples, _) =
            qwen_sft_train_eval_sample_counts(source_index.samples.len(), train_split)?;
        train_samples
    } else {
        source_index.samples.len()
    };
    let mut train_indices = source_index.samples;
    if shuffle {
        let mut rng = StdRng::seed_from_u64(seed);
        train_indices.shuffle(&mut rng);
    }
    train_indices.truncate(train_samples);
    if train_indices.is_empty() {
        bail!("SFT streaming token window requires at least one training sample");
    }

    let raw_sample_indices = (0..window_samples)
        .map(|relative| {
            let cursor = data_cursor_start + relative;
            let epoch = cursor / train_indices.len();
            let offset = cursor % train_indices.len();
            let index = if shuffle {
                qwen_epoch_permutation_index(train_indices.len(), seed, epoch, offset)
            } else {
                offset
            };
            train_indices[index].clone()
        })
        .collect::<Vec<_>>();
    let raw_window = qwen_sft_examples_by_raw_indices_with_regex_plan(
        &raw_sample_indices,
        field_map,
        &regex_plan,
    )?;
    let samples = raw_window
        .examples
        .iter()
        .map(|example| qwen_sft_token_sample(tokenizer, example, field_map))
        .collect::<Result<Vec<_>>>()?;
    Ok(QwenSftStreamingTokenWindow {
        samples,
        raw_sample_indices,
        raw_samples_read: raw_window.raw_samples_read,
        source_index_cache_hit: source_index_load.cache_hit,
        source_index_cache_written: source_index_load.cache_written,
    })
}

pub(crate) fn qwen_apply_sft_shuffle(
    dataset: QwenSftDataset,
    shuffle: bool,
    seed: u64,
) -> QwenSftDataset {
    if shuffle {
        dataset.shuffle_by_seed(seed)
    } else {
        dataset
    }
}

pub(crate) fn qwen_sft_train_eval_datasets_from_paths(
    tokenizer: &Tokenizer,
    train_paths: &[PathBuf],
    eval_paths: &[PathBuf],
    max_samples: Option<usize>,
    max_eval_samples: Option<usize>,
    train_split: f32,
    shuffle: bool,
    seed: u64,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftTrainEvalDatasets> {
    let train_dataset = QwenSftDataset::from_jsonl_paths_with_limit(
        tokenizer,
        train_paths,
        max_samples,
        field_map,
    )?;
    if eval_paths.is_empty() {
        let dataset = qwen_apply_sft_shuffle(train_dataset, shuffle, seed);
        let combined_summary = dataset.summary();
        let (train_dataset, eval_dataset) = dataset.train_eval_split(train_split)?;
        return Ok(QwenSftTrainEvalDatasets {
            combined_summary,
            train_dataset,
            eval_dataset,
        });
    }

    let eval_field_map = qwen_sft_eval_field_map(field_map);
    let eval_dataset = QwenSftDataset::from_jsonl_paths_with_limit(
        tokenizer,
        eval_paths,
        max_eval_samples,
        &eval_field_map,
    )?;
    let train_summary = train_dataset.summary();
    let eval_summary = eval_dataset.summary();
    let combined_source_files =
        qwen_merge_sft_source_files(&train_summary.source_files, &eval_summary.source_files);
    let combined_source_sample_counts = qwen_merge_sft_source_sample_counts(
        &train_summary.source_sample_counts,
        &eval_summary.source_sample_counts,
    );
    let combined_fingerprint = qwen_combine_sft_fingerprints(
        &combined_source_files,
        &train_summary.fingerprint,
        &eval_summary.fingerprint,
    );
    let train_dataset = qwen_apply_sft_shuffle(
        train_dataset.with_source_metadata(
            combined_source_files.clone(),
            combined_source_sample_counts.clone(),
            combined_fingerprint.clone(),
        ),
        shuffle,
        seed,
    );
    let eval_dataset = eval_dataset.with_source_metadata(
        combined_source_files.clone(),
        combined_source_sample_counts.clone(),
        combined_fingerprint.clone(),
    );
    Ok(QwenSftTrainEvalDatasets {
        combined_summary: QwenSftDatasetSummary {
            samples: train_summary.samples + eval_summary.samples,
            total_tokens: train_summary.total_tokens + eval_summary.total_tokens,
            response_tokens: train_summary.response_tokens + eval_summary.response_tokens,
            masked_positions: train_summary.masked_positions + eval_summary.masked_positions,
            max_sequence_tokens: train_summary
                .max_sequence_tokens
                .max(eval_summary.max_sequence_tokens),
            source_files: combined_source_files,
            source_sample_counts: combined_source_sample_counts,
            fingerprint: combined_fingerprint,
            shuffle,
        },
        train_dataset,
        eval_dataset,
    })
}

pub(crate) fn qwen_sft_eval_field_map(field_map: &QwenSftFieldMap) -> QwenSftFieldMap {
    let mut eval_field_map = field_map.clone();
    eval_field_map.source_weights.clear();
    eval_field_map.source_max_samples.clear();
    eval_field_map.source_instruction_fields.clear();
    eval_field_map.source_input_fields.clear();
    eval_field_map.source_response_fields.clear();
    eval_field_map
}

pub(crate) fn qwen_merge_sft_source_files(train: &[String], eval: &[String]) -> Vec<String> {
    train
        .iter()
        .chain(eval.iter())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn qwen_merge_sft_source_sample_counts(
    train: &[QwenSftSourceSampleCount],
    eval: &[QwenSftSourceSampleCount],
) -> Vec<QwenSftSourceSampleCount> {
    let mut counts = BTreeMap::new();
    for source_count in train.iter().chain(eval.iter()) {
        *counts.entry(source_count.path.clone()).or_insert(0) += source_count.samples;
    }
    counts
        .into_iter()
        .map(|(path, samples)| QwenSftSourceSampleCount { path, samples })
        .collect()
}

pub(crate) fn qwen_combine_sft_fingerprints(
    source_files: &[String],
    train_fingerprint: &str,
    eval_fingerprint: &str,
) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    qwen_sft_hash_bytes(&mut hash, b"train");
    qwen_sft_hash_bytes(&mut hash, train_fingerprint.as_bytes());
    qwen_sft_hash_bytes(&mut hash, b"\0eval");
    qwen_sft_hash_bytes(&mut hash, eval_fingerprint.as_bytes());
    for file in source_files {
        qwen_sft_hash_bytes(&mut hash, b"\0path");
        qwen_sft_hash_bytes(&mut hash, file.as_bytes());
    }
    format!("{hash:016x}")
}

pub(crate) fn qwen_epoch_permutation_index(
    dataset_len: usize,
    dataset_order_seed: u64,
    epoch: usize,
    offset: usize,
) -> usize {
    let mut order = (0..dataset_len).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(
        dataset_order_seed ^ ((epoch as u64).wrapping_add(1)).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    );
    order.shuffle(&mut rng);
    order[offset]
}

pub(crate) fn qwen_sft_examples_from_jsonl_paths_with_limit(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftExampleSet> {
    if paths.is_empty() {
        bail!("SFT dataset must contain at least one JSONL path");
    }
    if max_samples == Some(0) {
        bail!("SFT data.max_samples must be greater than zero");
    }
    let source_weights = qwen_sft_source_weights(paths.len(), field_map)?;
    let source_max_samples = qwen_sft_source_max_samples(paths.len(), field_map)?;
    let source_field_maps = qwen_sft_source_field_maps(paths.len(), field_map)?;
    let mut examples = Vec::new();
    let mut source_files = BTreeSet::new();
    let mut source_sample_counts = BTreeMap::new();
    let mut seen_records = field_map.dedupe_samples.then(HashSet::new);
    for (((path, source_weight), source_limit), source_field_map) in paths
        .iter()
        .zip(source_weights.iter().copied())
        .zip(source_max_samples.iter().copied())
        .zip(source_field_maps.iter())
    {
        if max_samples.is_some_and(|limit| examples.len() >= limit) {
            break;
        }
        let global_remaining = max_samples.map(|limit| limit.saturating_sub(examples.len()));
        let local_field_map = qwen_sft_field_map_for_source(field_map, source_field_map);
        let local_regex_plan = QwenSftRegexPlan::compile(&local_field_map)?;
        let example_set = qwen_sft_examples_from_jsonl_path_with_limit(
            path,
            global_remaining,
            source_limit,
            source_weight,
            &local_field_map,
            &local_regex_plan,
            &mut seen_records,
        )?;
        examples.extend(example_set.examples);
        source_files.extend(example_set.source_files);
        for source_count in example_set.source_sample_counts {
            *source_sample_counts.entry(source_count.path).or_insert(0) += source_count.samples;
        }
    }
    if examples.is_empty() {
        bail!("SFT dataset must contain at least one example");
    }
    let source_files = source_files.into_iter().collect::<Vec<_>>();
    let source_sample_counts = source_sample_counts
        .into_iter()
        .map(|(path, samples)| QwenSftSourceSampleCount { path, samples })
        .collect::<Vec<_>>();
    let fingerprint = qwen_sft_dataset_fingerprint(&source_files, &examples, field_map);
    Ok(QwenSftExampleSet {
        examples,
        source_files,
        source_sample_counts,
        fingerprint,
    })
}

#[cfg(test)]
pub(crate) fn qwen_sft_streaming_source_index(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftStreamingSourceIndex> {
    Ok(qwen_sft_streaming_source_scan(paths, max_samples, field_map)?.index)
}

pub(crate) fn qwen_sft_streaming_source_scan(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftStreamingSourceScan> {
    if paths.is_empty() {
        bail!("SFT dataset must contain at least one JSONL path");
    }
    if max_samples == Some(0) {
        bail!("SFT data.max_samples must be greater than zero");
    }
    let source_weights = qwen_sft_source_weights(paths.len(), field_map)?;
    let source_max_samples = qwen_sft_source_max_samples(paths.len(), field_map)?;
    let source_field_maps = qwen_sft_source_field_maps(paths.len(), field_map)?;
    let mut samples = Vec::new();
    let mut source_files = BTreeSet::new();
    let mut source_sample_counts = BTreeMap::new();
    let mut seen_records = field_map.dedupe_samples.then(HashSet::new);

    for (((path, source_weight), source_limit), source_field_map) in paths
        .iter()
        .zip(source_weights.iter().copied())
        .zip(source_max_samples.iter().copied())
        .zip(source_field_maps.iter())
    {
        if max_samples.is_some_and(|limit| samples.len() >= limit) {
            break;
        }
        let local_field_map = qwen_sft_field_map_for_source(field_map, source_field_map);
        let local_regex_plan = QwenSftRegexPlan::compile(&local_field_map)?;
        let mut source_samples = 0usize;
        for file in qwen_sft_jsonl_files(path)? {
            if max_samples.is_some_and(|limit| samples.len() >= limit)
                || source_limit.is_some_and(|limit| source_samples >= limit)
            {
                break;
            }
            let file_path = file.display().to_string();
            let mut reader = BufReader::new(
                fs::File::open(&file)
                    .with_context(|| format!("failed to read {}", file.display()))?,
            );
            let mut line = String::new();
            let mut line_index = 0usize;
            loop {
                if max_samples.is_some_and(|limit| samples.len() >= limit)
                    || source_limit.is_some_and(|limit| source_samples >= limit)
                {
                    break;
                }
                let byte_offset = reader.stream_position().with_context(|| {
                    format!("failed to seek SFT JSONL record {}", file.display())
                })?;
                line.clear();
                let bytes_read = reader.read_line(&mut line).with_context(|| {
                    format!(
                        "failed to read SFT JSONL record {}:{}",
                        file.display(),
                        line_index + 1
                    )
                })?;
                if bytes_read == 0 {
                    break;
                }
                if line.trim().is_empty() {
                    line_index += 1;
                    continue;
                }
                let Some(record) = maybe_qwen_sft_record_from_jsonl_line(
                    &line,
                    &local_field_map,
                    &local_regex_plan,
                    &file,
                    line_index + 1,
                )?
                else {
                    line_index += 1;
                    continue;
                };
                if !qwen_sft_record_passes_filters(&record, &local_field_map, &local_regex_plan) {
                    line_index += 1;
                    continue;
                }
                if let Some(seen_records) = &mut seen_records {
                    if !seen_records.insert(qwen_sft_record_dedupe_key(&record)) {
                        line_index += 1;
                        continue;
                    }
                }
                for _ in 0..source_weight {
                    if max_samples.is_some_and(|limit| samples.len() >= limit) {
                        break;
                    }
                    source_files.insert(file_path.clone());
                    samples.push(QwenSftRawSampleIndex {
                        path: file_path.clone(),
                        index_in_file: line_index,
                        global_index: samples.len(),
                        byte_offset,
                        field_map: source_field_map.clone(),
                    });
                    *source_sample_counts.entry(file_path.clone()).or_insert(0) += 1;
                }
                source_samples += 1;
                line_index += 1;
            }
        }
    }
    if samples.is_empty() {
        bail!("SFT dataset must contain at least one example");
    }

    let source_files = source_files.into_iter().collect::<Vec<_>>();
    let index = QwenSftStreamingSourceIndex { samples };
    let fingerprint = qwen_sft_streaming_fingerprint_from_index(&index, &source_files, field_map)?;
    Ok(QwenSftStreamingSourceScan {
        summary: QwenSftStreamingSourceSummary {
            samples: index.samples.len(),
            source_files,
            source_sample_counts: source_sample_counts
                .into_iter()
                .map(|(path, samples)| QwenSftSourceSampleCount { path, samples })
                .collect(),
            fingerprint,
        },
        index,
    })
}

pub(crate) fn qwen_sft_streaming_source_index_with_cache(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    cache_path: Option<&Path>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftStreamingSourceIndexLoad> {
    let expected_paths = paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    let expected_source_files = qwen_sft_streaming_source_file_metadata(paths)?;
    if let Some(cache_path) = cache_path {
        if cache_path.exists() {
            let contents = fs::read_to_string(cache_path)
                .with_context(|| format!("failed to read {}", cache_path.display()))?;
            let cache: QwenSftStreamingSourceIndexCache = serde_json::from_str(&contents)
                .with_context(|| format!("failed to parse {}", cache_path.display()))?;
            if cache.format != "rustrain.qwen_sft_offset_index.v7" {
                bail!(
                    "unsupported SFT streaming index cache format {} in {}",
                    cache.format,
                    cache_path.display()
                );
            }
            if cache.paths != expected_paths {
                bail!(
                    "SFT streaming index cache paths {:?} do not match {:?}",
                    cache.paths,
                    expected_paths
                );
            }
            if cache.source_files != expected_source_files {
                bail!(
                    "SFT streaming index cache source_files {:?} do not match {:?}",
                    cache.source_files,
                    expected_source_files
                );
            }
            if cache.max_samples != max_samples {
                bail!(
                    "SFT streaming index cache max_samples {:?} does not match {:?}",
                    cache.max_samples,
                    max_samples
                );
            }
            if cache.field_map != *field_map {
                bail!(
                    "SFT streaming index cache field_map {:?} does not match {:?}",
                    cache.field_map,
                    field_map
                );
            }
            if cache.min_response_chars != field_map.min_response_chars {
                bail!(
                    "SFT streaming index cache min_response_chars {} does not match {}",
                    cache.min_response_chars,
                    field_map.min_response_chars
                );
            }
            if cache.samples.is_empty() {
                bail!(
                    "SFT streaming index cache {} contains no samples",
                    cache_path.display()
                );
            }
            if cache.summary.samples != cache.samples.len() {
                bail!(
                    "SFT streaming index cache summary sample count {} does not match {} raw offsets",
                    cache.summary.samples,
                    cache.samples.len()
                );
            }
            return Ok(QwenSftStreamingSourceIndexLoad {
                index: QwenSftStreamingSourceIndex {
                    samples: cache.samples,
                },
                summary: Some(cache.summary),
                cache_hit: true,
                cache_written: false,
            });
        }
    }

    let scan = qwen_sft_streaming_source_scan(paths, max_samples, field_map)?;
    let summary = if cache_path.is_some() {
        Some(scan.summary.clone())
    } else {
        None
    };
    let mut cache_written = false;
    if let Some(cache_path) = cache_path {
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let cache = QwenSftStreamingSourceIndexCache {
            format: "rustrain.qwen_sft_offset_index.v7".to_string(),
            paths: expected_paths,
            source_files: expected_source_files,
            max_samples,
            field_map: field_map.clone(),
            min_response_chars: field_map.min_response_chars,
            summary: summary
                .clone()
                .context("SFT streaming index cache write requires source summary")?,
            samples: scan.index.samples.clone(),
        };
        let contents = serde_json::to_string_pretty(&cache)
            .context("failed to serialize SFT streaming index cache")?;
        fs::write(cache_path, contents)
            .with_context(|| format!("failed to write {}", cache_path.display()))?;
        cache_written = true;
    }
    Ok(QwenSftStreamingSourceIndexLoad {
        index: scan.index,
        summary,
        cache_hit: false,
        cache_written,
    })
}

pub(crate) fn qwen_sft_streaming_source_file_metadata(
    paths: &[PathBuf],
) -> Result<Vec<QwenSftStreamingSourceFileMetadata>> {
    let mut source_files = Vec::new();
    for path in paths {
        for file in qwen_sft_jsonl_files(path)? {
            let metadata = fs::metadata(&file)
                .with_context(|| format!("failed to inspect {}", file.display()))?;
            let modified = metadata
                .modified()
                .with_context(|| format!("failed to inspect mtime for {}", file.display()))?;
            let modified_unix_nanos = modified
                .duration_since(UNIX_EPOCH)
                .with_context(|| {
                    format!(
                        "mtime for {} is earlier than the Unix epoch",
                        file.display()
                    )
                })?
                .as_nanos();
            source_files.push(QwenSftStreamingSourceFileMetadata {
                path: file.display().to_string(),
                len: metadata.len(),
                modified_unix_nanos,
            });
        }
    }
    Ok(source_files)
}

pub(crate) fn qwen_sft_examples_by_raw_indices(
    raw_indices: &[QwenSftRawSampleIndex],
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftRawExampleWindow> {
    let regex_plan = QwenSftRegexPlan::compile(field_map)?;
    qwen_sft_examples_by_raw_indices_with_regex_plan(raw_indices, field_map, &regex_plan)
}

pub(crate) fn qwen_sft_examples_by_raw_indices_with_regex_plan(
    raw_indices: &[QwenSftRawSampleIndex],
    field_map: &QwenSftFieldMap,
    regex_plan: &QwenSftRegexPlan,
) -> Result<QwenSftRawExampleWindow> {
    if raw_indices.is_empty() {
        bail!("SFT streaming raw index read requires at least one sample");
    }
    let mut by_path: BTreeMap<String, Vec<&QwenSftRawSampleIndex>> = BTreeMap::new();
    for raw_index in raw_indices {
        by_path
            .entry(raw_index.path.clone())
            .or_default()
            .push(raw_index);
    }

    let mut loaded = BTreeMap::new();
    let mut offsets_by_sample = BTreeMap::new();
    for raw_index in raw_indices {
        offsets_by_sample
            .entry((raw_index.path.clone(), raw_index.index_in_file))
            .or_insert(raw_index.byte_offset);
    }
    for (path, wanted_raw_indices) in &by_path {
        let wanted_indices = wanted_raw_indices
            .iter()
            .map(|raw_index| raw_index.index_in_file)
            .collect::<BTreeSet<_>>();
        let mut file = fs::File::open(path).with_context(|| format!("failed to read {path}"))?;
        for raw_index in wanted_raw_indices {
            let index_in_file = raw_index.index_in_file;
            let byte_offset = *offsets_by_sample
                .get(&(path.clone(), index_in_file))
                .ok_or_else(|| {
                    anyhow!(
                        "SFT streaming raw sample offset not found: {}:{}",
                        path,
                        index_in_file + 1
                    )
                })?;
            if !wanted_indices.contains(&index_in_file) {
                continue;
            }
            file.seek(SeekFrom::Start(byte_offset)).with_context(|| {
                format!(
                    "failed to seek SFT JSONL record {path}:{} at byte offset {}",
                    index_in_file + 1,
                    byte_offset
                )
            })?;
            let mut reader = BufReader::new(&file);
            let mut line = String::new();
            reader.read_line(&mut line).with_context(|| {
                format!(
                    "failed to read SFT JSONL record {path}:{} at byte offset {}",
                    index_in_file + 1,
                    byte_offset
                )
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let local_field_map = qwen_sft_field_map_for_source(field_map, &raw_index.field_map);
            let local_regex_plan = if raw_index.field_map == QwenSftSourceFieldMap::default() {
                None
            } else {
                Some(QwenSftRegexPlan::compile(&local_field_map)?)
            };
            let record = qwen_sft_record_from_jsonl_line(
                &line,
                &local_field_map,
                local_regex_plan.as_ref().unwrap_or(regex_plan),
            )
            .with_context(|| {
                format!(
                    "failed to parse SFT JSONL record {path}:{} at byte offset {}",
                    index_in_file + 1,
                    byte_offset
                )
            })?;
            loaded.insert(
                (path.clone(), index_in_file),
                QwenSftExample {
                    system: record.system,
                    instruction: record.instruction,
                    input: record.input,
                    response: record.response,
                },
            );
        }
    }

    let examples = raw_indices
        .iter()
        .map(|raw_index| {
            loaded
                .get(&(raw_index.path.clone(), raw_index.index_in_file))
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "SFT streaming raw sample not found: {}:{}",
                        raw_index.path,
                        raw_index.index_in_file + 1
                    )
                })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(QwenSftRawExampleWindow {
        examples,
        raw_samples_read: loaded.len(),
    })
}

pub(crate) fn qwen_sft_examples_from_jsonl_path_with_limit(
    path: &Path,
    max_samples: Option<usize>,
    source_max_samples: Option<usize>,
    source_weight: usize,
    field_map: &QwenSftFieldMap,
    regex_plan: &QwenSftRegexPlan,
    seen_records: &mut Option<HashSet<String>>,
) -> Result<QwenSftExampleSet> {
    let files = qwen_sft_jsonl_files(path)?;

    if files.is_empty() {
        bail!("SFT JSONL path {} did not contain files", path.display());
    }

    let mut examples = Vec::new();
    let mut source_sample_counts = Vec::new();
    let mut source_samples = 0usize;
    for file in &files {
        if max_samples.is_some_and(|limit| examples.len() >= limit)
            || source_max_samples.is_some_and(|limit| source_samples >= limit)
        {
            break;
        }
        let reader = BufReader::new(
            fs::File::open(file).with_context(|| format!("failed to read {}", file.display()))?,
        );
        let before = examples.len();
        for (line_index, line) in reader.lines().enumerate() {
            if max_samples.is_some_and(|limit| examples.len() >= limit)
                || source_max_samples.is_some_and(|limit| source_samples >= limit)
            {
                break;
            }
            let line = line.with_context(|| {
                format!(
                    "failed to read SFT JSONL record {}:{}",
                    file.display(),
                    line_index + 1
                )
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let Some(record) = maybe_qwen_sft_record_from_jsonl_line(
                &line,
                field_map,
                regex_plan,
                file,
                line_index + 1,
            )?
            else {
                continue;
            };
            if !qwen_sft_record_passes_filters(&record, field_map, regex_plan) {
                continue;
            }
            if let Some(seen_records) = seen_records {
                if !seen_records.insert(qwen_sft_record_dedupe_key(&record)) {
                    continue;
                }
            }
            let example = QwenSftExample {
                system: record.system,
                instruction: record.instruction,
                input: record.input,
                response: record.response,
            };
            for _ in 0..source_weight {
                if max_samples.is_some_and(|limit| examples.len() >= limit) {
                    break;
                }
                examples.push(example.clone());
            }
            source_samples += 1;
        }
        let consumed = examples.len() - before;
        if consumed > 0 {
            source_sample_counts.push(QwenSftSourceSampleCount {
                path: file.display().to_string(),
                samples: consumed,
            });
        }
    }

    if examples.is_empty() {
        bail!("SFT JSONL path {} did not contain examples", path.display());
    }
    let source_files = source_sample_counts
        .iter()
        .map(|count| count.path.clone())
        .collect::<Vec<_>>();
    let fingerprint = qwen_sft_dataset_fingerprint(&source_files, &examples, field_map);
    Ok(QwenSftExampleSet {
        examples,
        source_files,
        source_sample_counts,
        fingerprint,
    })
}

pub(crate) fn qwen_sft_streaming_source_summary(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftStreamingSourceSummary> {
    Ok(qwen_sft_streaming_source_scan(paths, max_samples, field_map)?.summary)
}

pub(crate) fn qwen_sft_arrow_source_summary_from_ipc(
    path: &Path,
    limit: usize,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftArrowSourceSummary> {
    let arrow = qwen_sft_arrow_examples_from_ipc(path, Some(limit), field_map)?;
    let source_file = path.display().to_string();
    Ok(QwenSftArrowSourceSummary {
        input: source_file.clone(),
        arrow_ipc_format: arrow.arrow_ipc_format,
        limit,
        source_rows: arrow.source_rows,
        source_rows_exact: true,
        columns: arrow.columns,
        column_map: QwenSftArrowColumnMap {
            instruction: field_map.instruction.clone(),
            input: field_map.input.clone(),
            response: field_map.response.clone(),
        },
        samples: arrow.examples.len(),
        source_files: arrow.source_files,
        source_sample_counts: arrow.source_sample_counts,
        fingerprint: arrow.fingerprint,
        tokenized_samples_materialized: false,
        jsonl_materialized: false,
    })
}

pub(crate) fn qwen_sft_arrow_examples_from_paths_with_limit(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftArrowExampleSet> {
    let scan = qwen_sft_arrow_source_scan(paths, max_samples, field_map)?;
    let raw_window = qwen_sft_arrow_examples_by_raw_indices(&scan.index.samples, field_map)?;
    Ok(QwenSftArrowExampleSet {
        examples: raw_window.examples,
        row_indices: scan
            .index
            .samples
            .iter()
            .map(|sample| sample.row_index)
            .collect(),
        source_rows: scan.summary.samples,
        arrow_ipc_format: "indexed".to_string(),
        columns: Vec::new(),
        source_files: scan.summary.source_files,
        source_sample_counts: scan.summary.source_sample_counts,
        fingerprint: scan.summary.fingerprint,
    })
}

pub(crate) fn qwen_sft_arrow_source_summary_from_paths(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftStreamingSourceSummary> {
    Ok(qwen_sft_arrow_source_scan(paths, max_samples, field_map)?.summary)
}

pub(crate) fn qwen_sft_arrow_source_scan(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftArrowSourceScan> {
    if paths.is_empty() {
        bail!("SFT Arrow dataset must contain at least one path");
    }
    if max_samples == Some(0) {
        bail!("qwen SFT Arrow source max_samples must be greater than zero");
    }
    let source_weights = qwen_sft_source_weights(paths.len(), field_map)?;
    let source_max_samples = qwen_sft_source_max_samples(paths.len(), field_map)?;
    let source_field_maps = qwen_sft_source_field_maps(paths.len(), field_map)?;
    let mut samples = Vec::new();
    let mut source_files = BTreeSet::new();
    let mut source_sample_counts = BTreeMap::new();
    let mut seen_examples = field_map.dedupe_samples.then(HashSet::new);

    for (((path, source_weight), source_limit), source_field_map) in paths
        .iter()
        .zip(source_weights.iter().copied())
        .zip(source_max_samples.iter().copied())
        .zip(source_field_maps.iter())
    {
        if max_samples.is_some_and(|limit| samples.len() >= limit) {
            break;
        }
        if !path.is_file() {
            bail!(
                "instruction_arrow data path must be an Arrow IPC file: {}",
                path.display()
            );
        }
        let source_remaining = match (max_samples, source_limit) {
            (Some(global_limit), Some(source_limit)) => {
                Some(source_limit.min(global_limit.saturating_sub(samples.len())))
            }
            (Some(global_limit), None) => Some(global_limit.saturating_sub(samples.len())),
            (None, Some(source_limit)) => Some(source_limit),
            (None, None) => None,
        };
        if source_remaining == Some(0) {
            continue;
        }
        let local_field_map = qwen_sft_field_map_for_source(field_map, source_field_map);
        let source_file = path.display().to_string();
        let source_scan = qwen_sft_arrow_scan_indices_from_ipc(
            path,
            source_remaining,
            &local_field_map,
            &mut seen_examples,
        )?;
        for row_index in source_scan.row_indices {
            if max_samples.is_some_and(|limit| samples.len() >= limit) {
                break;
            }
            for _ in 0..source_weight {
                if max_samples.is_some_and(|limit| samples.len() >= limit) {
                    break;
                }
                source_files.insert(source_file.clone());
                samples.push(QwenSftArrowRawSampleIndex {
                    path: source_file.clone(),
                    row_index,
                    global_index: samples.len(),
                    field_map: source_field_map.clone(),
                });
                *source_sample_counts.entry(source_file.clone()).or_insert(0) += 1;
            }
        }
    }

    if samples.is_empty() {
        bail!("SFT Arrow dataset must contain at least one example");
    }
    if let Some(limit) = max_samples
        && samples.len() < limit
    {
        bail!(
            "SFT Arrow dataset produced {} examples, below limit {}",
            samples.len(),
            limit
        );
    }
    let source_files = source_files.into_iter().collect::<Vec<_>>();
    let index = QwenSftArrowSourceIndex { samples };
    let fingerprint =
        qwen_sft_arrow_streaming_fingerprint_from_index(&index, &source_files, field_map)?;
    let source_sample_counts = source_sample_counts
        .into_iter()
        .map(|(path, samples)| QwenSftSourceSampleCount { path, samples })
        .collect::<Vec<_>>();
    Ok(QwenSftArrowSourceScan {
        summary: QwenSftStreamingSourceSummary {
            samples: index.samples.len(),
            source_files,
            source_sample_counts,
            fingerprint,
        },
        index,
    })
}

pub(crate) fn qwen_sft_arrow_source_index_with_cache(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    cache_path: Option<&Path>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftArrowSourceIndexLoad> {
    let expected_paths = paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    let expected_source_files = qwen_sft_single_file_source_metadata(paths)?;
    if let Some(cache_path) = cache_path
        && cache_path.exists()
    {
        let contents = fs::read_to_string(cache_path)
            .with_context(|| format!("failed to read {}", cache_path.display()))?;
        let cache: QwenSftArrowSourceIndexCache = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse {}", cache_path.display()))?;
        if cache.format != "rustrain.qwen_sft_arrow_row_index.v1" {
            bail!(
                "unsupported SFT Arrow index cache format {} in {}",
                cache.format,
                cache_path.display()
            );
        }
        if cache.paths != expected_paths {
            bail!(
                "SFT Arrow index cache paths {:?} do not match {:?}",
                cache.paths,
                expected_paths
            );
        }
        if cache.source_files != expected_source_files {
            bail!(
                "SFT Arrow index cache source_files {:?} do not match {:?}",
                cache.source_files,
                expected_source_files
            );
        }
        if cache.max_samples != max_samples {
            bail!(
                "SFT Arrow index cache max_samples {:?} does not match {:?}",
                cache.max_samples,
                max_samples
            );
        }
        if cache.field_map != *field_map {
            bail!(
                "SFT Arrow index cache field_map {:?} does not match {:?}",
                cache.field_map,
                field_map
            );
        }
        if cache.samples.is_empty() {
            bail!(
                "SFT Arrow index cache {} contains no samples",
                cache_path.display()
            );
        }
        if cache.summary.samples != cache.samples.len() {
            bail!(
                "SFT Arrow index cache summary sample count {} does not match {} row indices",
                cache.summary.samples,
                cache.samples.len()
            );
        }
        return Ok(QwenSftArrowSourceIndexLoad {
            index: QwenSftArrowSourceIndex {
                samples: cache.samples,
            },
            summary: Some(cache.summary),
            cache_hit: true,
            cache_written: false,
        });
    }

    let scan = qwen_sft_arrow_source_scan(paths, max_samples, field_map)?;
    let summary = Some(scan.summary.clone());
    let mut cache_written = false;
    if let Some(cache_path) = cache_path {
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let cache = QwenSftArrowSourceIndexCache {
            format: "rustrain.qwen_sft_arrow_row_index.v1".to_string(),
            paths: expected_paths,
            source_files: expected_source_files,
            max_samples,
            field_map: field_map.clone(),
            summary: summary
                .clone()
                .context("SFT Arrow index cache write requires source summary")?,
            samples: scan.index.samples.clone(),
        };
        let contents =
            serde_json::to_string_pretty(&cache).context("failed to serialize SFT Arrow cache")?;
        fs::write(cache_path, contents)
            .with_context(|| format!("failed to write {}", cache_path.display()))?;
        cache_written = true;
    }
    Ok(QwenSftArrowSourceIndexLoad {
        index: scan.index,
        summary,
        cache_hit: false,
        cache_written,
    })
}

pub(crate) fn qwen_sft_single_file_source_metadata(
    paths: &[PathBuf],
) -> Result<Vec<QwenSftStreamingSourceFileMetadata>> {
    let mut source_files = Vec::new();
    for path in paths {
        if !path.is_file() {
            bail!("SFT source path must be a file: {}", path.display());
        }
        let metadata =
            fs::metadata(path).with_context(|| format!("failed to inspect {}", path.display()))?;
        let modified = metadata
            .modified()
            .with_context(|| format!("failed to inspect mtime for {}", path.display()))?;
        let modified_unix_nanos = modified
            .duration_since(UNIX_EPOCH)
            .with_context(|| {
                format!(
                    "mtime for {} is earlier than the Unix epoch",
                    path.display()
                )
            })?
            .as_nanos();
        source_files.push(QwenSftStreamingSourceFileMetadata {
            path: path.display().to_string(),
            len: metadata.len(),
            modified_unix_nanos,
        });
    }
    Ok(source_files)
}

pub(crate) fn qwen_sft_arrow_streaming_token_window(
    tokenizer: &Tokenizer,
    paths: &[PathBuf],
    eval_paths: &[PathBuf],
    max_samples: Option<usize>,
    train_split: f32,
    shuffle: bool,
    seed: u64,
    data_cursor_start: usize,
    window_samples: usize,
    index_cache: Option<&Path>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftArrowStreamingTokenWindow> {
    if window_samples == 0 {
        bail!("SFT Arrow streaming token window requires at least one sample");
    }
    let source_index_load =
        qwen_sft_arrow_source_index_with_cache(paths, max_samples, index_cache, field_map)?;
    qwen_sft_arrow_streaming_token_window_from_index_load(
        tokenizer,
        eval_paths,
        train_split,
        shuffle,
        seed,
        data_cursor_start,
        window_samples,
        field_map,
        source_index_load,
    )
}

pub(crate) fn qwen_sft_arrow_streaming_token_window_from_index_load(
    tokenizer: &Tokenizer,
    eval_paths: &[PathBuf],
    train_split: f32,
    shuffle: bool,
    seed: u64,
    data_cursor_start: usize,
    window_samples: usize,
    field_map: &QwenSftFieldMap,
    source_index_load: QwenSftArrowSourceIndexLoad,
) -> Result<QwenSftArrowStreamingTokenWindow> {
    let source_index = source_index_load.index;
    let train_samples = if eval_paths.is_empty() {
        let (train_samples, _) =
            qwen_sft_train_eval_sample_counts(source_index.samples.len(), train_split)?;
        train_samples
    } else {
        source_index.samples.len()
    };
    let mut train_indices = source_index.samples;
    if shuffle {
        let mut rng = StdRng::seed_from_u64(seed);
        train_indices.shuffle(&mut rng);
    }
    train_indices.truncate(train_samples);
    if train_indices.is_empty() {
        bail!("SFT Arrow streaming token window requires at least one training sample");
    }
    let raw_sample_indices = (0..window_samples)
        .map(|relative| {
            let cursor = data_cursor_start + relative;
            let epoch = cursor / train_indices.len();
            let offset = cursor % train_indices.len();
            let index = if shuffle {
                qwen_epoch_permutation_index(train_indices.len(), seed, epoch, offset)
            } else {
                offset
            };
            train_indices[index].clone()
        })
        .collect::<Vec<_>>();
    let raw_window = qwen_sft_arrow_examples_by_raw_indices(&raw_sample_indices, field_map)?;
    let samples = raw_window
        .examples
        .iter()
        .map(|example| qwen_sft_token_sample(tokenizer, example, field_map))
        .collect::<Result<Vec<_>>>()?;
    Ok(QwenSftArrowStreamingTokenWindow {
        samples,
        raw_samples_read: raw_window.raw_samples_read,
        source_index_cache_hit: source_index_load.cache_hit,
        source_index_cache_written: source_index_load.cache_written,
    })
}

pub(crate) fn qwen_sft_arrow_streaming_dataset_plan(
    tokenizer: &Tokenizer,
    data_config: &RuntimeDataConfig,
    seed: u64,
    data_cursor_start: usize,
    train_steps: usize,
    local_batch_size_config: usize,
    world_size: usize,
    index_cache: Option<&Path>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftStreamingDatasetPlan> {
    let context = "qwen trainable session instruction_arrow streaming runtime";
    qwen_sft_arrow_validate_config_scope(data_config, context)?;
    qwen_sft_arrow_require_cache_or_bounds(data_config, index_cache, context)?;
    if local_batch_size_config == 0 {
        bail!("qwen trainable session micro_batch_size must be greater than zero");
    }
    if world_size == 0 {
        bail!("qwen trainable session world_size must be greater than zero");
    }
    if train_steps == 0 {
        bail!("qwen trainable session max_steps must be greater than zero");
    }

    let source_index_load = qwen_sft_arrow_source_index_with_cache(
        &data_config.paths,
        data_config.max_samples,
        index_cache,
        field_map,
    )?;
    let train_summary =
        source_index_load
            .summary
            .clone()
            .unwrap_or_else(|| QwenSftStreamingSourceSummary {
                samples: source_index_load.index.samples.len(),
                source_files: Vec::new(),
                source_sample_counts: Vec::new(),
                fingerprint: String::new(),
            });

    let train_source_files = train_summary.source_files.clone();
    let train_source_sample_counts = train_summary.source_sample_counts.clone();
    let (
        dataset_total_samples,
        dataset_train_samples,
        dataset_eval_samples,
        dataset_source_files,
        dataset_source_sample_counts,
        dataset_fingerprint,
    ) = if let Some(eval_input) = data_config.eval_paths.first() {
        let eval_field_map = qwen_sft_eval_field_map(field_map);
        let eval_summary = qwen_sft_arrow_source_summary_from_paths(
            std::slice::from_ref(eval_input),
            data_config.max_eval_samples,
            &eval_field_map,
        )?;
        let eval_source_files = eval_summary.source_files.clone();
        let eval_source_sample_counts = eval_summary.source_sample_counts.clone();
        let combined_source_files =
            qwen_merge_sft_source_files(&train_source_files, &eval_source_files);
        let combined_source_sample_counts = qwen_merge_sft_source_sample_counts(
            &train_source_sample_counts,
            &eval_source_sample_counts,
        );
        let combined_fingerprint = qwen_combine_sft_fingerprints(
            &combined_source_files,
            &train_summary.fingerprint,
            &eval_summary.fingerprint,
        );
        (
            train_summary.samples + eval_summary.samples,
            train_summary.samples,
            eval_summary.samples,
            combined_source_files,
            combined_source_sample_counts,
            combined_fingerprint,
        )
    } else {
        let (train_samples, eval_samples) =
            qwen_sft_train_eval_sample_counts(train_summary.samples, data_config.train_split)?;
        (
            train_summary.samples,
            train_samples,
            eval_samples,
            train_source_files,
            train_source_sample_counts,
            train_summary.fingerprint,
        )
    };
    if dataset_train_samples == 0 {
        bail!("qwen trainable session Arrow streaming runtime requires training samples");
    }

    let local_batch_size = local_batch_size_config.min(dataset_train_samples).max(1);
    let global_batch_size = local_batch_size * world_size;
    let required_batches = train_steps * global_batch_size + 1;
    let required_window_samples = required_batches + global_batch_size - 1;
    let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
    let data_cursor_next = data_cursor_end;
    let (data_epoch_start, data_sample_offset_start) =
        qwen_data_epoch_and_offset(data_cursor_start, dataset_train_samples)?;
    let (data_epoch_end, data_sample_offset_end) =
        qwen_data_epoch_and_offset(data_cursor_end, dataset_train_samples)?;
    let (data_epoch_next, data_sample_offset_next) =
        qwen_data_epoch_and_offset(data_cursor_next, dataset_train_samples)?;
    let _train_window_sample_cursors = qwen_sft_streaming_cursor_window(
        data_cursor_start,
        required_batches,
        global_batch_size,
        dataset_train_samples,
    )?;

    let streaming_window = qwen_sft_arrow_streaming_token_window_from_index_load(
        tokenizer,
        &data_config.eval_paths,
        data_config.train_split,
        data_config.shuffle,
        seed,
        data_cursor_start,
        required_window_samples,
        field_map,
        source_index_load,
    )?;
    if streaming_window.samples.len() != required_window_samples {
        bail!(
            "qwen trainable session Arrow streaming window produced {} samples, expected {}",
            streaming_window.samples.len(),
            required_window_samples
        );
    }
    let pad_token_id = qwen_pad_token_id(tokenizer);
    let train_batches = (0..required_batches)
        .map(|relative_cursor| {
            let end = relative_cursor + global_batch_size;
            qwen_sft_padded_batch(
                &streaming_window.samples[relative_cursor..end],
                pad_token_id,
            )
            .map(|batch| batch.input_ids)
        })
        .collect::<Result<Vec<_>>>()?;
    let initial_input_ids = train_batches
        .first()
        .ok_or_else(|| anyhow!("qwen trainable session Arrow streaming plan produced no batches"))?
        .shallow_clone();
    let sequence_tokens = initial_input_ids.size()[1] as usize;

    Ok(QwenSftStreamingDatasetPlan {
        dataset_total_samples,
        dataset_train_samples,
        dataset_eval_samples,
        dataset_source_files,
        dataset_source_sample_counts,
        dataset_fingerprint,
        dataset_shuffle: data_config.shuffle,
        local_batch_size,
        data_epoch_start,
        data_epoch_end,
        data_epoch_next,
        data_sample_offset_start,
        data_sample_offset_end,
        data_sample_offset_next,
        train_batches,
        initial_input_ids,
        sequence_tokens,
        streaming_index_cache_hit: streaming_window.source_index_cache_hit,
        streaming_index_cache_written: streaming_window.source_index_cache_written,
    })
}

pub(crate) fn qwen_sft_arrow_examples_by_raw_indices(
    raw_indices: &[QwenSftArrowRawSampleIndex],
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftRawExampleWindow> {
    if raw_indices.is_empty() {
        bail!("SFT Arrow raw index read requires at least one sample");
    }
    let mut max_row_by_path: BTreeMap<String, (&QwenSftSourceFieldMap, usize)> = BTreeMap::new();
    let wanted = raw_indices
        .iter()
        .map(|index| {
            max_row_by_path
                .entry(index.path.clone())
                .and_modify(|(field_map, max_row)| {
                    debug_assert_eq!(*field_map, &index.field_map);
                    *max_row = (*max_row).max(index.row_index);
                })
                .or_insert((&index.field_map, index.row_index));
            (index.path.clone(), index.row_index)
        })
        .collect::<BTreeSet<_>>();
    let mut loaded = BTreeMap::new();
    for (path, (source_field_map, max_row_index)) in max_row_by_path {
        let path_buf = PathBuf::from(&path);
        let local_field_map = qwen_sft_field_map_for_source(field_map, source_field_map);
        let arrow = qwen_sft_arrow_examples_from_ipc_with_limit_policy(
            &path_buf,
            Some(max_row_index + 1),
            false,
            &local_field_map,
        )?;
        for (row_index, example) in arrow.row_indices.into_iter().zip(arrow.examples) {
            if wanted.contains(&(path.clone(), row_index)) {
                loaded.insert((path.clone(), row_index), example);
            }
        }
    }
    let examples = raw_indices
        .iter()
        .map(|raw_index| {
            loaded
                .get(&(raw_index.path.clone(), raw_index.row_index))
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "SFT Arrow raw sample not found: {}:{}",
                        raw_index.path,
                        raw_index.row_index
                    )
                })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(QwenSftRawExampleWindow {
        examples,
        raw_samples_read: loaded.len(),
    })
}

pub(crate) fn qwen_sft_arrow_streaming_fingerprint_from_index(
    index: &QwenSftArrowSourceIndex,
    source_files: &[String],
    field_map: &QwenSftFieldMap,
) -> Result<String> {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    qwen_sft_hash_field_map(&mut hash, field_map);
    for file in source_files {
        qwen_sft_hash_source_file(&mut hash, file);
    }

    let mut offset = 0usize;
    while offset < index.samples.len() {
        let path = index.samples[offset].path.clone();
        let end = index.samples[offset..]
            .iter()
            .position(|sample| sample.path != path)
            .map(|relative| offset + relative)
            .unwrap_or(index.samples.len());
        if index.samples[offset..end]
            .iter()
            .any(|sample| sample.field_map != index.samples[offset].field_map)
        {
            bail!("SFT Arrow raw samples for {path} used multiple source field maps");
        }
        let local_field_map =
            qwen_sft_field_map_for_source(field_map, &index.samples[offset].field_map);
        qwen_sft_arrow_hash_index_segment_from_ipc(
            Path::new(&path),
            &index.samples[offset..end],
            &local_field_map,
            &mut hash,
        )?;
        offset = end;
    }

    Ok(format!("{hash:016x}"))
}

#[cfg(test)]
pub(crate) fn qwen_sft_arrow_fingerprint_from_index(
    index: &QwenSftArrowSourceIndex,
    source_files: &[String],
    field_map: &QwenSftFieldMap,
) -> Result<String> {
    let raw_window = qwen_sft_arrow_examples_by_raw_indices(&index.samples, field_map)?;
    Ok(qwen_sft_dataset_fingerprint(
        source_files,
        &raw_window.examples,
        field_map,
    ))
}

pub(crate) fn qwen_sft_arrow_scan_indices_from_ipc(
    path: &Path,
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
    seen_records: &mut Option<HashSet<String>>,
) -> Result<QwenSftArrowSourceRowScan> {
    if max_samples == Some(0) {
        bail!("qwen SFT Arrow source max_samples must be greater than zero");
    }
    let mut stream_seen_records = seen_records.clone();
    let stream_attempt = fs::File::open(path)
        .with_context(|| format!("failed to read {}", path.display()))
        .and_then(|file| {
            let reader = ArrowStreamReader::try_new(file, None).with_context(|| {
                format!("failed to open {} as Arrow IPC stream", path.display())
            })?;
            qwen_sft_arrow_scan_indices_from_batches(
                path,
                max_samples,
                field_map,
                &mut stream_seen_records,
                reader,
            )
        });
    match stream_attempt {
        Ok(scan) => {
            *seen_records = stream_seen_records;
            Ok(scan)
        }
        Err(stream_error) => {
            let mut file_seen_records = seen_records.clone();
            let file = fs::File::open(path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let reader = ArrowFileReader::try_new(file, None).with_context(|| {
                format!(
                    "failed to open {} as Arrow IPC stream or file; stream error: {stream_error}",
                    path.display()
                )
            })?;
            let scan = qwen_sft_arrow_scan_indices_from_batches(
                path,
                max_samples,
                field_map,
                &mut file_seen_records,
                reader,
            )?;
            *seen_records = file_seen_records;
            Ok(scan)
        }
    }
}

pub(crate) fn qwen_sft_arrow_examples_from_ipc(
    path: &Path,
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftArrowExampleSet> {
    qwen_sft_arrow_examples_from_ipc_with_limit_policy(path, max_samples, true, field_map)
}

pub(crate) fn qwen_sft_arrow_examples_from_ipc_with_limit_policy(
    path: &Path,
    max_samples: Option<usize>,
    require_exact_limit: bool,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftArrowExampleSet> {
    if max_samples == Some(0) {
        bail!("qwen SFT Arrow source max_samples must be greater than zero");
    }
    let stream_attempt = fs::File::open(path)
        .with_context(|| format!("failed to read {}", path.display()))
        .and_then(|file| {
            let reader = ArrowStreamReader::try_new(file, None).with_context(|| {
                format!("failed to open {} as Arrow IPC stream", path.display())
            })?;
            qwen_sft_arrow_examples_from_batches(
                path,
                max_samples,
                require_exact_limit,
                field_map,
                "stream",
                reader,
            )
        });
    match stream_attempt {
        Ok(summary) => Ok(summary),
        Err(stream_error) => {
            let file = fs::File::open(path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let reader = ArrowFileReader::try_new(file, None).with_context(|| {
                format!(
                    "failed to open {} as Arrow IPC stream or file; stream error: {stream_error}",
                    path.display()
                )
            })?;
            qwen_sft_arrow_examples_from_batches(
                path,
                max_samples,
                require_exact_limit,
                field_map,
                "file",
                reader,
            )
        }
    }
}

pub(crate) fn qwen_sft_arrow_scan_indices_from_batches<I>(
    path: &Path,
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
    seen_records: &mut Option<HashSet<String>>,
    mut batches: I,
) -> Result<QwenSftArrowSourceRowScan>
where
    I: Iterator<Item = std::result::Result<RecordBatch, arrow::error::ArrowError>>,
{
    let schema = batches
        .next()
        .transpose()
        .with_context(|| format!("failed to read first Arrow batch from {}", path.display()))?
        .map(|batch| {
            let schema = batch.schema();
            (schema, Some(batch))
        });
    let Some((schema, first_batch)) = schema else {
        bail!(
            "qwen SFT Arrow source {} contained no record batches",
            path.display()
        );
    };
    let instruction_index =
        qwen_arrow_required_column_index(&schema, &field_map.instruction, &["question"])?;
    let input_index = qwen_arrow_optional_column_index(&schema, &field_map.input)?;
    let response_index =
        qwen_arrow_required_column_index(&schema, &field_map.response, &["answer"])?;
    let regex_plan = QwenSftRegexPlan::compile(field_map)?;
    let mut row_indices = Vec::new();
    let mut source_rows = 0usize;

    if let Some(batch) = first_batch {
        let row_base = source_rows;
        source_rows += batch.num_rows();
        qwen_sft_arrow_collect_indices_from_batch(
            &batch,
            instruction_index,
            input_index,
            response_index,
            max_samples,
            field_map,
            &regex_plan,
            seen_records,
            row_base,
            &mut row_indices,
        )?;
    }
    for batch in batches {
        let batch =
            batch.with_context(|| format!("failed to read Arrow batch from {}", path.display()))?;
        let row_base = source_rows;
        source_rows += batch.num_rows();
        if !max_samples.is_some_and(|limit| row_indices.len() >= limit) {
            qwen_sft_arrow_collect_indices_from_batch(
                &batch,
                instruction_index,
                input_index,
                response_index,
                max_samples,
                field_map,
                &regex_plan,
                seen_records,
                row_base,
                &mut row_indices,
            )?;
        }
    }
    if row_indices.is_empty() {
        bail!(
            "SFT Arrow source {} did not contain examples",
            path.display()
        );
    }
    Ok(QwenSftArrowSourceRowScan { row_indices })
}

pub(crate) fn qwen_sft_arrow_examples_from_batches<I>(
    path: &Path,
    max_samples: Option<usize>,
    require_exact_limit: bool,
    field_map: &QwenSftFieldMap,
    arrow_ipc_format: &str,
    mut batches: I,
) -> Result<QwenSftArrowExampleSet>
where
    I: Iterator<Item = std::result::Result<RecordBatch, arrow::error::ArrowError>>,
{
    let schema = batches
        .next()
        .transpose()
        .with_context(|| format!("failed to read first Arrow batch from {}", path.display()))?
        .map(|batch| {
            let schema = batch.schema();
            (schema, Some(batch))
        });
    let Some((schema, first_batch)) = schema else {
        bail!(
            "qwen SFT Arrow source {} contained no record batches",
            path.display()
        );
    };
    let columns = schema
        .fields()
        .iter()
        .map(|field| field.name().to_string())
        .collect::<Vec<_>>();
    let instruction_index =
        qwen_arrow_required_column_index(&schema, &field_map.instruction, &["question"])?;
    let input_index = qwen_arrow_optional_column_index(&schema, &field_map.input)?;
    let response_index =
        qwen_arrow_required_column_index(&schema, &field_map.response, &["answer"])?;
    let regex_plan = QwenSftRegexPlan::compile(field_map)?;
    let mut examples = Vec::new();
    let mut row_indices = Vec::new();
    let mut seen_records = field_map.dedupe_samples.then(HashSet::new);
    let mut source_rows = 0usize;

    if let Some(batch) = first_batch {
        let row_base = source_rows;
        source_rows += batch.num_rows();
        qwen_sft_arrow_collect_examples_from_batch(
            &batch,
            instruction_index,
            input_index,
            response_index,
            max_samples,
            field_map,
            &regex_plan,
            &mut seen_records,
            row_base,
            &mut examples,
            &mut row_indices,
        )?;
    }
    for batch in batches {
        let batch =
            batch.with_context(|| format!("failed to read Arrow batch from {}", path.display()))?;
        let row_base = source_rows;
        source_rows += batch.num_rows();
        if !max_samples.is_some_and(|limit| examples.len() >= limit) {
            qwen_sft_arrow_collect_examples_from_batch(
                &batch,
                instruction_index,
                input_index,
                response_index,
                max_samples,
                field_map,
                &regex_plan,
                &mut seen_records,
                row_base,
                &mut examples,
                &mut row_indices,
            )?;
        }
    }
    if examples.is_empty() {
        bail!(
            "SFT Arrow source {} did not contain examples",
            path.display()
        );
    }
    if require_exact_limit
        && let Some(limit) = max_samples
        && examples.len() < limit
    {
        bail!(
            "SFT Arrow source {} produced {} examples, below limit {}",
            path.display(),
            examples.len(),
            limit
        );
    }
    let source_file = path.display().to_string();
    let source_files = vec![source_file.clone()];
    let sample_count = examples.len();
    let fingerprint = qwen_sft_dataset_fingerprint(&source_files, &examples, field_map);
    Ok(QwenSftArrowExampleSet {
        examples,
        row_indices,
        source_rows,
        arrow_ipc_format: arrow_ipc_format.to_string(),
        columns,
        source_files,
        source_sample_counts: vec![QwenSftSourceSampleCount {
            path: source_file,
            samples: sample_count,
        }],
        fingerprint,
    })
}

pub(crate) fn qwen_sft_arrow_hash_index_segment_from_ipc(
    path: &Path,
    samples: &[QwenSftArrowRawSampleIndex],
    field_map: &QwenSftFieldMap,
    hash: &mut u64,
) -> Result<()> {
    debug_assert!(!samples.is_empty());
    let mut stream_hash = *hash;
    let stream_attempt = fs::File::open(path)
        .with_context(|| format!("failed to read {}", path.display()))
        .and_then(|file| {
            let reader = ArrowStreamReader::try_new(file, None).with_context(|| {
                format!("failed to open {} as Arrow IPC stream", path.display())
            })?;
            qwen_sft_arrow_hash_index_segment_from_batches(
                path,
                samples,
                field_map,
                &mut stream_hash,
                reader,
            )
        });
    match stream_attempt {
        Ok(()) => {
            *hash = stream_hash;
            Ok(())
        }
        Err(stream_error) => {
            let mut file_hash = *hash;
            let file = fs::File::open(path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let reader = ArrowFileReader::try_new(file, None).with_context(|| {
                format!(
                    "failed to open {} as Arrow IPC stream or file; stream error: {stream_error}",
                    path.display()
                )
            })?;
            qwen_sft_arrow_hash_index_segment_from_batches(
                path,
                samples,
                field_map,
                &mut file_hash,
                reader,
            )?;
            *hash = file_hash;
            Ok(())
        }
    }
}

pub(crate) fn qwen_sft_arrow_hash_index_segment_from_batches<I>(
    path: &Path,
    samples: &[QwenSftArrowRawSampleIndex],
    field_map: &QwenSftFieldMap,
    hash: &mut u64,
    mut batches: I,
) -> Result<()>
where
    I: Iterator<Item = std::result::Result<RecordBatch, arrow::error::ArrowError>>,
{
    let schema = batches
        .next()
        .transpose()
        .with_context(|| format!("failed to read first Arrow batch from {}", path.display()))?
        .map(|batch| {
            let schema = batch.schema();
            (schema, Some(batch))
        });
    let Some((schema, first_batch)) = schema else {
        bail!(
            "qwen SFT Arrow source {} contained no record batches",
            path.display()
        );
    };
    let instruction_index =
        qwen_arrow_required_column_index(&schema, &field_map.instruction, &["question"])?;
    let input_index = qwen_arrow_optional_column_index(&schema, &field_map.input)?;
    let response_index =
        qwen_arrow_required_column_index(&schema, &field_map.response, &["answer"])?;
    let regex_plan = QwenSftRegexPlan::compile(field_map)?;
    let mut sample_offset = 0usize;
    let mut source_rows = 0usize;

    if let Some(batch) = first_batch {
        let row_base = source_rows;
        source_rows += batch.num_rows();
        qwen_sft_arrow_hash_samples_from_batch(
            &batch,
            instruction_index,
            input_index,
            response_index,
            field_map,
            &regex_plan,
            row_base,
            samples,
            &mut sample_offset,
            hash,
        )?;
    }
    for batch in batches {
        if sample_offset >= samples.len() {
            break;
        }
        let batch =
            batch.with_context(|| format!("failed to read Arrow batch from {}", path.display()))?;
        let row_base = source_rows;
        source_rows += batch.num_rows();
        qwen_sft_arrow_hash_samples_from_batch(
            &batch,
            instruction_index,
            input_index,
            response_index,
            field_map,
            &regex_plan,
            row_base,
            samples,
            &mut sample_offset,
            hash,
        )?;
    }
    if sample_offset != samples.len() {
        let missing = &samples[sample_offset];
        bail!(
            "SFT Arrow raw sample not found while hashing: {}:{}",
            missing.path,
            missing.row_index
        );
    }
    Ok(())
}

pub(crate) fn qwen_arrow_required_column_index(
    schema: &SchemaRef,
    column: &str,
    fallback_columns: &[&str],
) -> Result<usize> {
    if column.trim().is_empty() {
        bail!("Arrow required SFT column name must not be empty");
    }
    for candidate in std::iter::once(column).chain(fallback_columns.iter().copied()) {
        if let Some(index) = qwen_arrow_schema_column_index(schema, candidate)? {
            return Ok(index);
        }
    }
    bail!(
        "Arrow input is missing required SFT column {column}; fallbacks={fallback_columns:?}; columns={:?}",
        qwen_arrow_schema_column_names(schema)
    )
}

pub(crate) fn qwen_arrow_optional_column_index(
    schema: &SchemaRef,
    column: &str,
) -> Result<QwenSftArrowColumn> {
    if column.trim().is_empty() {
        return Ok(QwenSftArrowColumn::MissingOptional);
    }
    Ok(qwen_arrow_schema_column_index(schema, column)?
        .map(QwenSftArrowColumn::Present)
        .unwrap_or(QwenSftArrowColumn::MissingOptional))
}

pub(crate) fn qwen_arrow_schema_column_index(
    schema: &SchemaRef,
    column: &str,
) -> Result<Option<usize>> {
    let Ok(index) = schema.index_of(column) else {
        return Ok(None);
    };
    let field = schema.field(index);
    match field.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 => Ok(Some(index)),
        data_type => bail!("Arrow SFT column {column} must be utf8/large_utf8, got {data_type:?}"),
    }
}

pub(crate) fn qwen_arrow_schema_column_names(schema: &SchemaRef) -> Vec<&str> {
    schema
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect()
}

pub(crate) fn qwen_sft_arrow_collect_examples_from_batch(
    batch: &RecordBatch,
    instruction_index: usize,
    input_index: QwenSftArrowColumn,
    response_index: usize,
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
    regex_plan: &QwenSftRegexPlan,
    seen_records: &mut Option<HashSet<String>>,
    row_base: usize,
    examples: &mut Vec<QwenSftExample>,
    row_indices: &mut Vec<usize>,
) -> Result<()> {
    for row in 0..batch.num_rows() {
        if max_samples.is_some_and(|limit| examples.len() >= limit) {
            break;
        }
        let record = qwen_sft_arrow_record_from_batch(
            batch,
            instruction_index,
            input_index,
            response_index,
            field_map,
            regex_plan,
            row,
        )?;
        if !qwen_sft_record_passes_filters(&record, field_map, regex_plan) {
            continue;
        }
        if let Some(seen_records) = seen_records {
            if !seen_records.insert(qwen_sft_record_dedupe_key(&record)) {
                continue;
            }
        }
        examples.push(QwenSftExample {
            system: record.system,
            instruction: record.instruction,
            input: record.input,
            response: record.response,
        });
        row_indices.push(row_base + row);
    }
    Ok(())
}

pub(crate) fn qwen_sft_arrow_collect_indices_from_batch(
    batch: &RecordBatch,
    instruction_index: usize,
    input_index: QwenSftArrowColumn,
    response_index: usize,
    max_samples: Option<usize>,
    field_map: &QwenSftFieldMap,
    regex_plan: &QwenSftRegexPlan,
    seen_records: &mut Option<HashSet<String>>,
    row_base: usize,
    row_indices: &mut Vec<usize>,
) -> Result<()> {
    for row in 0..batch.num_rows() {
        if max_samples.is_some_and(|limit| row_indices.len() >= limit) {
            break;
        }
        let record = qwen_sft_arrow_record_from_batch(
            batch,
            instruction_index,
            input_index,
            response_index,
            field_map,
            regex_plan,
            row,
        )?;
        if !qwen_sft_record_passes_filters(&record, field_map, regex_plan) {
            continue;
        }
        if let Some(seen_records) = seen_records {
            if !seen_records.insert(qwen_sft_record_dedupe_key(&record)) {
                continue;
            }
        }
        row_indices.push(row_base + row);
    }
    Ok(())
}

pub(crate) fn qwen_sft_arrow_hash_samples_from_batch(
    batch: &RecordBatch,
    instruction_index: usize,
    input_index: QwenSftArrowColumn,
    response_index: usize,
    field_map: &QwenSftFieldMap,
    regex_plan: &QwenSftRegexPlan,
    row_base: usize,
    samples: &[QwenSftArrowRawSampleIndex],
    sample_offset: &mut usize,
    hash: &mut u64,
) -> Result<()> {
    while *sample_offset < samples.len() {
        let row_index = samples[*sample_offset].row_index;
        if row_index < row_base {
            bail!(
                "SFT Arrow raw sample indices must be sorted by row, got {} before batch base {}",
                row_index,
                row_base
            );
        }
        let relative_row = row_index - row_base;
        if relative_row >= batch.num_rows() {
            break;
        }
        let record = qwen_sft_arrow_record_from_batch(
            batch,
            instruction_index,
            input_index,
            response_index,
            field_map,
            regex_plan,
            relative_row,
        )?;
        let example = QwenSftExample {
            system: record.system,
            instruction: record.instruction,
            input: record.input,
            response: record.response,
        };
        qwen_sft_hash_example(hash, &example, field_map.has_system_source());
        *sample_offset += 1;
    }
    Ok(())
}

pub(crate) fn qwen_sft_arrow_record_from_batch(
    batch: &RecordBatch,
    instruction_index: usize,
    input_index: QwenSftArrowColumn,
    response_index: usize,
    field_map: &QwenSftFieldMap,
    regex_plan: &QwenSftRegexPlan,
    row: usize,
) -> Result<QwenSftRecord> {
    let mut record = QwenSftRecord {
        system: String::new(),
        instruction: qwen_arrow_string_value(batch.column(instruction_index).as_ref(), row)
            .with_context(|| format!("failed to read Arrow instruction row {row}"))?,
        input: qwen_arrow_optional_string_value(batch, input_index, row)
            .with_context(|| format!("failed to read Arrow input row {row}"))?,
        response: qwen_arrow_string_value(batch.column(response_index).as_ref(), row)
            .with_context(|| format!("failed to read Arrow response row {row}"))?,
    };
    record.instruction = qwen_normalize_jsonl_field(record.instruction, field_map);
    record.input = qwen_normalize_jsonl_field(record.input, field_map);
    record.response = qwen_normalize_jsonl_field(record.response, field_map);
    qwen_apply_field_defaults(&mut record, &field_map.field_defaults);
    qwen_apply_field_replacements(&mut record, &field_map.field_replacements);
    qwen_apply_field_regex_replacements(&mut record, &regex_plan.replacements);
    qwen_apply_field_case_transforms(&mut record, &field_map.field_case_transforms);
    qwen_apply_field_affixes(&mut record, &field_map.field_affixes);
    qwen_apply_field_strips(&mut record, &field_map.field_strips);
    qwen_apply_field_splits(&mut record, &field_map.field_splits);
    qwen_apply_field_truncations(&mut record, &field_map.field_truncations);
    qwen_apply_field_transforms(&mut record, &field_map.field_transforms, regex_plan);
    if field_map.normalize_whitespace {
        qwen_normalize_record_whitespace(&mut record);
    }
    Ok(record)
}

pub(crate) fn qwen_arrow_optional_string_value(
    batch: &RecordBatch,
    column: QwenSftArrowColumn,
    row: usize,
) -> Result<String> {
    match column {
        QwenSftArrowColumn::Present(index) => {
            qwen_arrow_string_value(batch.column(index).as_ref(), row)
        }
        QwenSftArrowColumn::MissingOptional => Ok(String::new()),
    }
}

pub(crate) fn qwen_arrow_string_value(array: &dyn Array, row: usize) -> Result<String> {
    if array.is_null(row) {
        return Ok(String::new());
    }
    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(array.value(row).to_string());
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(array.value(row).to_string());
    }
    bail!(
        "Arrow SFT column must be utf8/large_utf8, got {:?}",
        array.data_type()
    )
}

pub(crate) fn qwen_sft_source_weights(
    path_count: usize,
    field_map: &QwenSftFieldMap,
) -> Result<Vec<usize>> {
    if field_map.source_weights.is_empty() {
        return Ok(vec![1; path_count]);
    }
    let weights = if field_map.source_weights.len() == 1 {
        vec![field_map.source_weights[0]; path_count]
    } else if field_map.source_weights.len() == path_count {
        field_map.source_weights.clone()
    } else {
        bail!("data.source_weights must be empty, length 1, or match data.paths length");
    };
    if weights.iter().any(|weight| *weight == 0) {
        bail!("data.source_weights entries must be greater than zero");
    }
    Ok(weights)
}

pub(crate) fn qwen_sft_source_max_samples(
    path_count: usize,
    field_map: &QwenSftFieldMap,
) -> Result<Vec<Option<usize>>> {
    if field_map.source_max_samples.is_empty() {
        return Ok(vec![None; path_count]);
    }
    let limits = if field_map.source_max_samples.len() == 1 {
        vec![Some(field_map.source_max_samples[0]); path_count]
    } else if field_map.source_max_samples.len() == path_count {
        field_map
            .source_max_samples
            .iter()
            .map(|limit| Some(*limit))
            .collect()
    } else {
        bail!("data.source_max_samples must be empty, length 1, or match data.paths length");
    };
    if limits.iter().any(|limit| matches!(limit, Some(0))) {
        bail!("data.source_max_samples entries must be greater than zero");
    }
    Ok(limits)
}

pub(crate) fn qwen_sft_source_field_maps(
    path_count: usize,
    field_map: &QwenSftFieldMap,
) -> Result<Vec<QwenSftSourceFieldMap>> {
    let instructions = qwen_sft_source_string_overrides(
        path_count,
        &field_map.source_instruction_fields,
        "data.source_instruction_fields",
        true,
    )?;
    let inputs = qwen_sft_source_string_overrides(
        path_count,
        &field_map.source_input_fields,
        "data.source_input_fields",
        false,
    )?;
    let responses = qwen_sft_source_string_overrides(
        path_count,
        &field_map.source_response_fields,
        "data.source_response_fields",
        true,
    )?;
    Ok((0..path_count)
        .map(|index| QwenSftSourceFieldMap {
            instruction: instructions[index].clone(),
            input: inputs[index].clone(),
            response: responses[index].clone(),
        })
        .collect())
}

pub(crate) fn qwen_sft_source_string_overrides(
    path_count: usize,
    values: &[String],
    name: &str,
    required: bool,
) -> Result<Vec<Option<String>>> {
    if values.is_empty() {
        return Ok(vec![None; path_count]);
    }
    let expanded = if values.len() == 1 {
        vec![values[0].clone(); path_count]
    } else if values.len() == path_count {
        values.to_vec()
    } else {
        bail!("{name} must be empty, length 1, or match data.paths length");
    };
    if required && expanded.iter().any(|field| field.trim().is_empty()) {
        bail!("{name} entries must not be empty");
    }
    Ok(expanded.into_iter().map(Some).collect())
}

pub(crate) fn qwen_sft_field_map_for_source(
    field_map: &QwenSftFieldMap,
    source_field_map: &QwenSftSourceFieldMap,
) -> QwenSftFieldMap {
    let mut local = field_map.clone();
    if let Some(instruction) = &source_field_map.instruction {
        local.instruction = instruction.clone();
    }
    if let Some(input) = &source_field_map.input {
        local.input = input.clone();
    }
    if let Some(response) = &source_field_map.response {
        local.response = response.clone();
    }
    local.source_weights.clear();
    local.source_max_samples.clear();
    local.source_instruction_fields.clear();
    local.source_input_fields.clear();
    local.source_response_fields.clear();
    local
}

#[cfg(test)]
pub(crate) fn qwen_sft_streaming_fingerprint(
    paths: &[PathBuf],
    max_samples: Option<usize>,
    source_files: &[String],
    field_map: &QwenSftFieldMap,
) -> Result<String> {
    let index = match qwen_sft_streaming_source_scan(paths, max_samples, field_map) {
        Ok(scan) => scan.index,
        Err(error)
            if error
                .to_string()
                .contains("SFT dataset must contain at least one example") =>
        {
            QwenSftStreamingSourceIndex {
                samples: Vec::new(),
            }
        }
        Err(error) => return Err(error),
    };
    qwen_sft_streaming_fingerprint_from_index(&index, source_files, field_map)
}

pub(crate) fn qwen_sft_streaming_fingerprint_from_index(
    index: &QwenSftStreamingSourceIndex,
    source_files: &[String],
    field_map: &QwenSftFieldMap,
) -> Result<String> {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    qwen_sft_hash_field_map(&mut hash, field_map);
    for file in source_files {
        qwen_sft_hash_bytes(&mut hash, b"path");
        qwen_sft_hash_bytes(&mut hash, file.as_bytes());
        qwen_sft_hash_bytes(&mut hash, b"\0");
    }

    if !index.samples.is_empty() {
        let raw_window = qwen_sft_examples_by_raw_indices(&index.samples, field_map)?;
        for example in &raw_window.examples {
            qwen_sft_hash_example(&mut hash, example, field_map.has_system_source());
        }
    }
    Ok(format!("{hash:016x}"))
}

pub(crate) fn qwen_sft_jsonl_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_dir() {
        let mut sorted = BTreeSet::new();
        for entry in
            fs::read_dir(path).with_context(|| format!("failed to list {}", path.display()))?
        {
            let entry =
                entry.with_context(|| format!("failed to read entry in {}", path.display()))?;
            let file_type = entry
                .file_type()
                .with_context(|| format!("failed to inspect {}", entry.path().display()))?;
            if file_type.is_file()
                && entry.path().extension().and_then(|value| value.to_str()) == Some("jsonl")
            {
                sorted.insert(entry.path());
            }
        }
        Ok(sorted.into_iter().collect())
    } else {
        Ok(vec![path.to_path_buf()])
    }
}

pub(crate) fn qwen_sft_record_from_jsonl_line(
    line: &str,
    field_map: &QwenSftFieldMap,
    regex_plan: &QwenSftRegexPlan,
) -> Result<QwenSftRecord> {
    let values: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(line).context("invalid JSON object")?;
    if let Some(messages_field) = &field_map.chat_messages {
        let mut record = qwen_sft_record_from_chat_messages(&values, messages_field, field_map)?;
        qwen_apply_field_defaults(&mut record, &field_map.field_defaults);
        qwen_apply_field_replacements(&mut record, &field_map.field_replacements);
        qwen_apply_field_regex_replacements(&mut record, &regex_plan.replacements);
        qwen_apply_field_case_transforms(&mut record, &field_map.field_case_transforms);
        qwen_apply_field_affixes(&mut record, &field_map.field_affixes);
        qwen_apply_field_strips(&mut record, &field_map.field_strips);
        qwen_apply_field_splits(&mut record, &field_map.field_splits);
        qwen_apply_field_truncations(&mut record, &field_map.field_truncations);
        qwen_apply_field_transforms(&mut record, &field_map.field_transforms, regex_plan);
        if field_map.normalize_whitespace {
            qwen_normalize_record_whitespace(&mut record);
        }
        return Ok(record);
    }
    let instruction = qwen_normalize_jsonl_field(
        qwen_defaultable_jsonl_string_field(
            &values,
            &field_map.instruction,
            qwen_field_default_value(&field_map.field_defaults, FieldDefaultTarget::Instruction),
        )?,
        field_map,
    );
    let system = match &field_map.system {
        Some(field) => {
            qwen_normalize_jsonl_field(qwen_optional_jsonl_string_field(&values, field)?, field_map)
        }
        None => String::new(),
    };
    let input = qwen_normalize_jsonl_field(
        qwen_optional_jsonl_string_field(&values, &field_map.input)?,
        field_map,
    );
    let response = qwen_normalize_jsonl_field(
        qwen_defaultable_jsonl_string_field(
            &values,
            &field_map.response,
            qwen_field_default_value(&field_map.field_defaults, FieldDefaultTarget::Response),
        )?,
        field_map,
    );
    let mut record = QwenSftRecord {
        system,
        instruction,
        input,
        response,
    };
    qwen_apply_field_defaults(&mut record, &field_map.field_defaults);
    qwen_apply_field_replacements(&mut record, &field_map.field_replacements);
    qwen_apply_field_regex_replacements(&mut record, &regex_plan.replacements);
    qwen_apply_field_case_transforms(&mut record, &field_map.field_case_transforms);
    qwen_apply_field_affixes(&mut record, &field_map.field_affixes);
    qwen_apply_field_strips(&mut record, &field_map.field_strips);
    qwen_apply_field_splits(&mut record, &field_map.field_splits);
    qwen_apply_field_truncations(&mut record, &field_map.field_truncations);
    qwen_apply_field_transforms(&mut record, &field_map.field_transforms, regex_plan);
    if field_map.normalize_whitespace {
        qwen_normalize_record_whitespace(&mut record);
    }
    Ok(record)
}

pub(crate) fn qwen_sft_record_from_chat_messages(
    values: &BTreeMap<String, serde_json::Value>,
    messages_field: &str,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftRecord> {
    let messages = match qwen_jsonl_path_value(values, messages_field) {
        Some(serde_json::Value::Array(messages)) => messages,
        Some(_) => bail!("SFT JSONL field {messages_field} must be an array"),
        None => bail!("SFT JSONL record missing required field {messages_field}"),
    };
    let mut system = String::new();
    let mut instruction = None;
    let mut response = None;
    for (index, message) in messages.iter().enumerate() {
        let object = match message {
            serde_json::Value::Object(object) => object,
            _ => bail!("SFT chat message {messages_field}[{index}] must be an object"),
        };
        let role = qwen_required_chat_message_string_field(object, "role", messages_field, index)?;
        let content =
            qwen_required_chat_message_string_field(object, "content", messages_field, index)?;
        let content = qwen_normalize_jsonl_field(content, field_map);
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
        .ok_or_else(|| anyhow!("SFT chat record missing user message in field {messages_field}"))?;
    let response = response.filter(|value| !value.is_empty()).ok_or_else(|| {
        anyhow!("SFT chat record missing assistant message in field {messages_field}")
    })?;
    Ok(QwenSftRecord {
        system,
        instruction,
        input: String::new(),
        response,
    })
}

pub(crate) fn qwen_required_chat_message_string_field(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    messages_field: &str,
    index: usize,
) -> Result<String> {
    match object.get(field) {
        Some(serde_json::Value::String(value)) => Ok(value.clone()),
        Some(_) => bail!("SFT chat message {messages_field}[{index}].{field} must be a string"),
        None => bail!("SFT chat message {messages_field}[{index}] missing required field {field}"),
    }
}

pub(crate) fn maybe_qwen_sft_record_from_jsonl_line(
    line: &str,
    field_map: &QwenSftFieldMap,
    regex_plan: &QwenSftRegexPlan,
    file: &Path,
    line_number: usize,
) -> Result<Option<QwenSftRecord>> {
    match qwen_sft_record_from_jsonl_line(line, field_map, regex_plan) {
        Ok(record) => Ok(Some(record)),
        Err(error) if field_map.skip_invalid_records => {
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
                "failed to parse SFT JSONL record {}:{}",
                file.display(),
                line_number
            )
        }),
    }
}

pub(crate) fn qwen_normalize_jsonl_field(value: String, field_map: &QwenSftFieldMap) -> String {
    if field_map.trim_fields {
        value.trim().to_string()
    } else {
        value
    }
}

pub(crate) fn qwen_apply_field_replacements(
    record: &mut QwenSftRecord,
    replacements: &[FieldReplacement],
) {
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

pub(crate) fn qwen_apply_field_regex_replacements(
    record: &mut QwenSftRecord,
    replacements: &[QwenCompiledRegexReplacement],
) {
    for replacement in replacements {
        if matches!(
            replacement.field,
            FieldReplacementTarget::System | FieldReplacementTarget::All
        ) {
            record.system = replacement
                .regex
                .replace_all(&record.system, replacement.replacement.as_str())
                .into_owned();
        }
        if matches!(
            replacement.field,
            FieldReplacementTarget::Instruction | FieldReplacementTarget::All
        ) {
            record.instruction = replacement
                .regex
                .replace_all(&record.instruction, replacement.replacement.as_str())
                .into_owned();
        }
        if matches!(
            replacement.field,
            FieldReplacementTarget::Input | FieldReplacementTarget::All
        ) {
            record.input = replacement
                .regex
                .replace_all(&record.input, replacement.replacement.as_str())
                .into_owned();
        }
        if matches!(
            replacement.field,
            FieldReplacementTarget::Response | FieldReplacementTarget::All
        ) {
            record.response = replacement
                .regex
                .replace_all(&record.response, replacement.replacement.as_str())
                .into_owned();
        }
    }
}

pub(crate) fn qwen_apply_field_defaults(record: &mut QwenSftRecord, defaults: &[FieldDefault]) {
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

pub(crate) fn qwen_apply_field_case_transforms(
    record: &mut QwenSftRecord,
    transforms: &[FieldCaseTransform],
) {
    for transform in transforms {
        if matches!(
            transform.field,
            FieldReplacementTarget::System | FieldReplacementTarget::All
        ) {
            record.system = qwen_apply_field_case_transform(&record.system, transform.case);
        }
        if matches!(
            transform.field,
            FieldReplacementTarget::Instruction | FieldReplacementTarget::All
        ) {
            record.instruction =
                qwen_apply_field_case_transform(&record.instruction, transform.case);
        }
        if matches!(
            transform.field,
            FieldReplacementTarget::Input | FieldReplacementTarget::All
        ) {
            record.input = qwen_apply_field_case_transform(&record.input, transform.case);
        }
        if matches!(
            transform.field,
            FieldReplacementTarget::Response | FieldReplacementTarget::All
        ) {
            record.response = qwen_apply_field_case_transform(&record.response, transform.case);
        }
    }
}

pub(crate) fn qwen_apply_field_case_transform(value: &str, case: FieldCaseTransformKind) -> String {
    match case {
        FieldCaseTransformKind::Lowercase => value.to_lowercase(),
        FieldCaseTransformKind::Uppercase => value.to_uppercase(),
    }
}

pub(crate) fn qwen_apply_field_affixes(record: &mut QwenSftRecord, affixes: &[FieldAffix]) {
    for affix in affixes {
        if matches!(
            affix.field,
            FieldReplacementTarget::System | FieldReplacementTarget::All
        ) {
            record.system = qwen_apply_field_affix(&record.system, affix);
        }
        if matches!(
            affix.field,
            FieldReplacementTarget::Instruction | FieldReplacementTarget::All
        ) {
            record.instruction = qwen_apply_field_affix(&record.instruction, affix);
        }
        if matches!(
            affix.field,
            FieldReplacementTarget::Input | FieldReplacementTarget::All
        ) {
            record.input = qwen_apply_field_affix(&record.input, affix);
        }
        if matches!(
            affix.field,
            FieldReplacementTarget::Response | FieldReplacementTarget::All
        ) {
            record.response = qwen_apply_field_affix(&record.response, affix);
        }
    }
}

pub(crate) fn qwen_apply_field_affix(value: &str, affix: &FieldAffix) -> String {
    let mut transformed =
        String::with_capacity(affix.prefix.len() + value.len() + affix.suffix.len());
    transformed.push_str(&affix.prefix);
    transformed.push_str(value);
    transformed.push_str(&affix.suffix);
    transformed
}

pub(crate) fn qwen_apply_field_strips(record: &mut QwenSftRecord, strips: &[FieldStrip]) {
    for strip in strips {
        if matches!(
            strip.field,
            FieldReplacementTarget::System | FieldReplacementTarget::All
        ) {
            record.system = qwen_apply_field_strip(&record.system, strip);
        }
        if matches!(
            strip.field,
            FieldReplacementTarget::Instruction | FieldReplacementTarget::All
        ) {
            record.instruction = qwen_apply_field_strip(&record.instruction, strip);
        }
        if matches!(
            strip.field,
            FieldReplacementTarget::Input | FieldReplacementTarget::All
        ) {
            record.input = qwen_apply_field_strip(&record.input, strip);
        }
        if matches!(
            strip.field,
            FieldReplacementTarget::Response | FieldReplacementTarget::All
        ) {
            record.response = qwen_apply_field_strip(&record.response, strip);
        }
    }
}

pub(crate) fn qwen_apply_field_strip(value: &str, strip: &FieldStrip) -> String {
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

pub(crate) fn qwen_apply_field_splits(record: &mut QwenSftRecord, splits: &[FieldSplit]) {
    for split in splits {
        if matches!(
            split.field,
            FieldReplacementTarget::System | FieldReplacementTarget::All
        ) {
            record.system = qwen_apply_field_split(&record.system, split);
        }
        if matches!(
            split.field,
            FieldReplacementTarget::Instruction | FieldReplacementTarget::All
        ) {
            record.instruction = qwen_apply_field_split(&record.instruction, split);
        }
        if matches!(
            split.field,
            FieldReplacementTarget::Input | FieldReplacementTarget::All
        ) {
            record.input = qwen_apply_field_split(&record.input, split);
        }
        if matches!(
            split.field,
            FieldReplacementTarget::Response | FieldReplacementTarget::All
        ) {
            record.response = qwen_apply_field_split(&record.response, split);
        }
    }
}

pub(crate) fn qwen_apply_field_split(value: &str, split: &FieldSplit) -> String {
    match value.split_once(&split.delimiter) {
        Some((before, after)) => match split.side {
            FieldSplitSide::Before => before.to_string(),
            FieldSplitSide::After => after.to_string(),
        },
        None => value.to_string(),
    }
}

pub(crate) fn qwen_apply_field_truncations(
    record: &mut QwenSftRecord,
    truncations: &[FieldTruncation],
) {
    for truncation in truncations {
        if matches!(
            truncation.field,
            FieldReplacementTarget::System | FieldReplacementTarget::All
        ) {
            record.system = qwen_truncate_chars(&record.system, truncation.max_chars);
        }
        if matches!(
            truncation.field,
            FieldReplacementTarget::Instruction | FieldReplacementTarget::All
        ) {
            record.instruction = qwen_truncate_chars(&record.instruction, truncation.max_chars);
        }
        if matches!(
            truncation.field,
            FieldReplacementTarget::Input | FieldReplacementTarget::All
        ) {
            record.input = qwen_truncate_chars(&record.input, truncation.max_chars);
        }
        if matches!(
            truncation.field,
            FieldReplacementTarget::Response | FieldReplacementTarget::All
        ) {
            record.response = qwen_truncate_chars(&record.response, truncation.max_chars);
        }
    }
}

pub(crate) fn qwen_truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

pub(crate) fn qwen_apply_field_transforms(
    record: &mut QwenSftRecord,
    transforms: &[FieldTransform],
    regex_plan: &QwenSftRegexPlan,
) {
    let mut regex_replacements = regex_plan.transform_regex_replacements.iter();
    for (index, transform) in transforms.iter().enumerate() {
        match transform.op {
            FieldTransformOp::Default => {
                qwen_apply_field_transform_targets(record, transform.field, |value| {
                    if value.trim().is_empty() {
                        transform.value.clone()
                    } else {
                        value.to_string()
                    }
                });
            }
            FieldTransformOp::Replace => {
                qwen_apply_field_transform_targets(record, transform.field, |value| {
                    value.replace(&transform.pattern, &transform.replacement)
                });
            }
            FieldTransformOp::RegexReplace => {
                let Some(compiled) = regex_replacements.next() else {
                    continue;
                };
                debug_assert_eq!(compiled.index, index);
                qwen_apply_field_transform_targets(record, compiled.field, |value| {
                    compiled
                        .regex
                        .replace_all(value, compiled.replacement.as_str())
                        .into_owned()
                });
            }
            FieldTransformOp::Case => {
                let Some(case) = transform.case else {
                    continue;
                };
                qwen_apply_field_transform_targets(record, transform.field, |value| {
                    qwen_apply_field_case_transform(value, case)
                });
            }
            FieldTransformOp::Affix => {
                qwen_apply_field_transform_targets(record, transform.field, |value| {
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
                qwen_apply_field_transform_targets(record, transform.field, |value| {
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
                qwen_apply_field_transform_targets(record, transform.field, |value| {
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
                qwen_apply_field_transform_targets(record, transform.field, |value| {
                    qwen_truncate_chars(value, max_chars)
                });
            }
        }
    }
}

pub(crate) fn qwen_apply_field_transform_targets<F>(
    record: &mut QwenSftRecord,
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

pub(crate) fn qwen_field_default_value(
    defaults: &[FieldDefault],
    field: FieldDefaultTarget,
) -> Option<&str> {
    defaults
        .iter()
        .find(|default| default.field == field)
        .map(|default| default.value.as_str())
}

pub(crate) fn qwen_normalize_record_whitespace(record: &mut QwenSftRecord) {
    record.system = qwen_normalize_whitespace(&record.system);
    record.instruction = qwen_normalize_whitespace(&record.instruction);
    record.input = qwen_normalize_whitespace(&record.input);
    record.response = qwen_normalize_whitespace(&record.response);
}

pub(crate) fn qwen_normalize_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn qwen_sft_record_dedupe_key(record: &QwenSftRecord) -> String {
    format!(
        "{}\0{}\0{}\0{}",
        record.system, record.instruction, record.input, record.response
    )
}

pub(crate) fn qwen_sft_record_passes_filters(
    record: &QwenSftRecord,
    field_map: &QwenSftFieldMap,
    regex_plan: &QwenSftRegexPlan,
) -> bool {
    let needs_prompt_chars = field_map.min_prompt_chars.is_some()
        || field_map.max_prompt_chars.is_some()
        || field_map.min_sample_chars.is_some()
        || field_map.max_sample_chars.is_some();
    let prompt_chars = if needs_prompt_chars {
        Some(
            qwen_render_sft_record_prompt(record, field_map)
                .chars()
                .count(),
        )
    } else {
        None
    };
    qwen_sft_length_filter_passes(
        record.response.chars().count(),
        Some(field_map.min_response_chars),
        field_map.max_response_chars,
    ) && qwen_sft_string_contains_any_filter_passes(
        &record.response,
        &field_map.response_contains_any,
    ) && qwen_sft_string_excludes_any_filter_passes(
        &record.response,
        &field_map.response_excludes_any,
    ) && qwen_sft_string_contains_any_filter_passes(
        &record.instruction,
        &field_map.instruction_contains_any,
    ) && qwen_sft_string_excludes_any_filter_passes(
        &record.instruction,
        &field_map.instruction_excludes_any,
    ) && qwen_sft_length_filter_passes(
        record.instruction.chars().count(),
        field_map.min_instruction_chars,
        field_map.max_instruction_chars,
    ) && qwen_sft_string_contains_any_filter_passes(&record.input, &field_map.input_contains_any)
        && qwen_sft_string_excludes_any_filter_passes(&record.input, &field_map.input_excludes_any)
        && qwen_sft_length_filter_passes(
            record.input.chars().count(),
            field_map.min_input_chars,
            field_map.max_input_chars,
        )
        && qwen_sft_string_contains_any_filter_passes(
            &record.system,
            &field_map.system_contains_any,
        )
        && qwen_sft_string_excludes_any_filter_passes(
            &record.system,
            &field_map.system_excludes_any,
        )
        && qwen_sft_regex_contains_any_filter_passes(record, &regex_plan.contains_any)
        && qwen_sft_regex_excludes_any_filter_passes(record, &regex_plan.excludes_any)
        && qwen_sft_length_filter_passes(
            record.system.chars().count(),
            field_map.min_system_chars,
            field_map.max_system_chars,
        )
        && prompt_chars.is_none_or(|chars| {
            qwen_sft_length_filter_passes(
                chars,
                field_map.min_prompt_chars,
                field_map.max_prompt_chars,
            ) && qwen_sft_length_filter_passes(
                chars + record.response.chars().count(),
                field_map.min_sample_chars,
                field_map.max_sample_chars,
            )
        })
}

pub(crate) fn qwen_sft_length_filter_passes(
    chars: usize,
    min_chars: Option<usize>,
    max_chars: Option<usize>,
) -> bool {
    min_chars.is_none_or(|limit| chars >= limit) && max_chars.is_none_or(|limit| chars <= limit)
}

pub(crate) fn qwen_sft_string_contains_any_filter_passes(value: &str, needles: &[String]) -> bool {
    needles.is_empty() || needles.iter().any(|needle| value.contains(needle))
}

pub(crate) fn qwen_sft_string_excludes_any_filter_passes(value: &str, needles: &[String]) -> bool {
    needles.iter().all(|needle| !value.contains(needle))
}

pub(crate) fn qwen_sft_regex_contains_any_filter_passes(
    record: &QwenSftRecord,
    filters: &[QwenCompiledRegexFilter],
) -> bool {
    if filters.is_empty() {
        return true;
    }
    for filter in filters {
        if qwen_sft_regex_filter_matches(record, filter) {
            return true;
        }
    }
    false
}

pub(crate) fn qwen_sft_regex_excludes_any_filter_passes(
    record: &QwenSftRecord,
    filters: &[QwenCompiledRegexFilter],
) -> bool {
    for filter in filters {
        if qwen_sft_regex_filter_matches(record, filter) {
            return false;
        }
    }
    true
}

pub(crate) fn qwen_sft_regex_filter_matches(
    record: &QwenSftRecord,
    filter: &QwenCompiledRegexFilter,
) -> bool {
    match filter.field {
        FieldReplacementTarget::System => filter.regex.is_match(&record.system),
        FieldReplacementTarget::Instruction => filter.regex.is_match(&record.instruction),
        FieldReplacementTarget::Input => filter.regex.is_match(&record.input),
        FieldReplacementTarget::Response => filter.regex.is_match(&record.response),
        FieldReplacementTarget::All => {
            filter.regex.is_match(&record.system)
                || filter.regex.is_match(&record.instruction)
                || filter.regex.is_match(&record.input)
                || filter.regex.is_match(&record.response)
        }
    }
}

pub(crate) fn qwen_render_sft_record_prompt(
    record: &QwenSftRecord,
    field_map: &QwenSftFieldMap,
) -> String {
    let template = if record.input.trim().is_empty() {
        &field_map.prompt_template
    } else {
        &field_map.prompt_with_input_template
    };
    template
        .replace("{system}", &record.system)
        .replace("{instruction}", &record.instruction)
        .replace("{input}", &record.input)
}

pub(crate) fn qwen_optional_jsonl_string_field(
    values: &BTreeMap<String, serde_json::Value>,
    field: &str,
) -> Result<String> {
    match qwen_jsonl_path_value(values, field) {
        Some(serde_json::Value::String(value)) => Ok(value.clone()),
        Some(_) => bail!("SFT JSONL field {field} must be a string"),
        None => Ok(String::new()),
    }
}

pub(crate) fn qwen_defaultable_jsonl_string_field(
    values: &BTreeMap<String, serde_json::Value>,
    field: &str,
    default_value: Option<&str>,
) -> Result<String> {
    match qwen_jsonl_path_value(values, field) {
        Some(serde_json::Value::String(value)) if value.trim().is_empty() => {
            Ok(default_value.unwrap_or(value).to_string())
        }
        Some(serde_json::Value::String(value)) => Ok(value.clone()),
        Some(_) => bail!("SFT JSONL field {field} must be a string"),
        None => default_value
            .map(|value| value.to_string())
            .ok_or_else(|| anyhow!("SFT JSONL record missing required field {field}")),
    }
}

pub(crate) fn qwen_jsonl_path_value<'a>(
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

pub(crate) fn qwen_sft_dataset_fingerprint(
    source_files: &[String],
    examples: &[QwenSftExample],
    field_map: &QwenSftFieldMap,
) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    qwen_sft_hash_field_map(&mut hash, field_map);
    for file in source_files {
        qwen_sft_hash_source_file(&mut hash, file);
    }
    for example in examples {
        qwen_sft_hash_example(&mut hash, example, field_map.has_system_source());
    }
    format!("{hash:016x}")
}

pub(crate) fn qwen_sft_hash_source_file(hash: &mut u64, file: &str) {
    qwen_sft_hash_bytes(hash, b"path");
    qwen_sft_hash_bytes(hash, file.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
}

pub(crate) fn qwen_sft_hash_field_map(hash: &mut u64, field_map: &QwenSftFieldMap) {
    qwen_sft_hash_bytes(hash, b"field_map");
    qwen_sft_hash_bytes(hash, field_map.instruction.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
    qwen_sft_hash_bytes(hash, field_map.input.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
    qwen_sft_hash_bytes(hash, field_map.response.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
    if let Some(system) = &field_map.system {
        qwen_sft_hash_bytes(hash, b"system");
        qwen_sft_hash_bytes(hash, system.as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(chat_messages) = &field_map.chat_messages {
        qwen_sft_hash_bytes(hash, b"chat_messages");
        qwen_sft_hash_bytes(hash, chat_messages.as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    qwen_sft_hash_bytes(hash, field_map.prompt_template.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
    qwen_sft_hash_bytes(hash, field_map.prompt_with_input_template.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
    qwen_sft_hash_bytes(
        hash,
        if field_map.trim_fields {
            b"trim".as_slice()
        } else {
            b"raw".as_slice()
        },
    );
    qwen_sft_hash_bytes(hash, b"\0");
    qwen_sft_hash_bytes(hash, b"min_response_chars");
    qwen_sft_hash_bytes(hash, field_map.min_response_chars.to_string().as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
    if let Some(max_response_chars) = field_map.max_response_chars {
        qwen_sft_hash_bytes(hash, b"max_response_chars");
        qwen_sft_hash_bytes(hash, max_response_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if !field_map.field_replacements.is_empty() {
        qwen_sft_hash_bytes(hash, b"field_replacements");
        for replacement in &field_map.field_replacements {
            qwen_sft_hash_bytes(hash, format!("{:?}", replacement.field).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, replacement.pattern.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, replacement.replacement.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
        }
    }
    if !field_map.field_regex_replacements.is_empty() {
        qwen_sft_hash_bytes(hash, b"field_regex_replacements");
        for replacement in &field_map.field_regex_replacements {
            qwen_sft_hash_bytes(hash, format!("{:?}", replacement.field).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, replacement.pattern.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, replacement.replacement.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
        }
    }
    if field_map.normalize_whitespace {
        qwen_sft_hash_bytes(hash, b"normalize_whitespace");
        qwen_sft_hash_bytes(hash, b"true");
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if !field_map.field_defaults.is_empty() {
        qwen_sft_hash_bytes(hash, b"field_defaults");
        for default in &field_map.field_defaults {
            qwen_sft_hash_bytes(hash, format!("{:?}", default.field).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, default.value.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
        }
    }
    if !field_map.field_case_transforms.is_empty() {
        qwen_sft_hash_bytes(hash, b"field_case_transforms");
        for transform in &field_map.field_case_transforms {
            qwen_sft_hash_bytes(hash, format!("{:?}", transform.field).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, format!("{:?}", transform.case).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
        }
    }
    if !field_map.field_affixes.is_empty() {
        qwen_sft_hash_bytes(hash, b"field_affixes");
        for affix in &field_map.field_affixes {
            qwen_sft_hash_bytes(hash, format!("{:?}", affix.field).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, affix.prefix.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, affix.suffix.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
        }
    }
    if !field_map.field_strips.is_empty() {
        qwen_sft_hash_bytes(hash, b"field_strips");
        for strip in &field_map.field_strips {
            qwen_sft_hash_bytes(hash, format!("{:?}", strip.field).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, strip.prefix.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, strip.suffix.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
        }
    }
    if !field_map.field_splits.is_empty() {
        qwen_sft_hash_bytes(hash, b"field_splits");
        for split in &field_map.field_splits {
            qwen_sft_hash_bytes(hash, format!("{:?}", split.field).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, split.delimiter.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, format!("{:?}", split.side).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
        }
    }
    if !field_map.field_truncations.is_empty() {
        qwen_sft_hash_bytes(hash, b"field_truncations");
        for truncation in &field_map.field_truncations {
            qwen_sft_hash_bytes(hash, format!("{:?}", truncation.field).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, truncation.max_chars.to_string().as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
        }
    }
    if !field_map.field_transforms.is_empty() {
        qwen_sft_hash_bytes(hash, b"field_transforms");
        for transform in &field_map.field_transforms {
            qwen_sft_hash_bytes(hash, format!("{:?}", transform.field).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, format!("{:?}", transform.op).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, transform.value.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, transform.pattern.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, transform.replacement.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, transform.prefix.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, transform.suffix.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, transform.delimiter.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(
                hash,
                transform
                    .side
                    .map(|side| format!("{side:?}"))
                    .unwrap_or_default()
                    .as_bytes(),
            );
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(
                hash,
                transform
                    .max_chars
                    .map(|max_chars| max_chars.to_string())
                    .unwrap_or_default()
                    .as_bytes(),
            );
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(
                hash,
                transform
                    .case
                    .map(|case| format!("{case:?}"))
                    .unwrap_or_default()
                    .as_bytes(),
            );
            qwen_sft_hash_bytes(hash, b"\0");
        }
    }
    if !field_map.response_contains_any.is_empty() {
        qwen_sft_hash_bytes(hash, b"response_contains_any");
        for needle in &field_map.response_contains_any {
            qwen_sft_hash_bytes(hash, needle.as_bytes());
            qwen_sft_hash_bytes(hash, b",");
        }
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if !field_map.response_excludes_any.is_empty() {
        qwen_sft_hash_bytes(hash, b"response_excludes_any");
        for needle in &field_map.response_excludes_any {
            qwen_sft_hash_bytes(hash, needle.as_bytes());
            qwen_sft_hash_bytes(hash, b",");
        }
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if !field_map.instruction_contains_any.is_empty() {
        qwen_sft_hash_bytes(hash, b"instruction_contains_any");
        for needle in &field_map.instruction_contains_any {
            qwen_sft_hash_bytes(hash, needle.as_bytes());
            qwen_sft_hash_bytes(hash, b",");
        }
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if !field_map.instruction_excludes_any.is_empty() {
        qwen_sft_hash_bytes(hash, b"instruction_excludes_any");
        for needle in &field_map.instruction_excludes_any {
            qwen_sft_hash_bytes(hash, needle.as_bytes());
            qwen_sft_hash_bytes(hash, b",");
        }
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if !field_map.input_contains_any.is_empty() {
        qwen_sft_hash_bytes(hash, b"input_contains_any");
        for needle in &field_map.input_contains_any {
            qwen_sft_hash_bytes(hash, needle.as_bytes());
            qwen_sft_hash_bytes(hash, b",");
        }
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if !field_map.input_excludes_any.is_empty() {
        qwen_sft_hash_bytes(hash, b"input_excludes_any");
        for needle in &field_map.input_excludes_any {
            qwen_sft_hash_bytes(hash, needle.as_bytes());
            qwen_sft_hash_bytes(hash, b",");
        }
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if !field_map.field_regex_contains_any.is_empty() {
        qwen_sft_hash_bytes(hash, b"field_regex_contains_any");
        for filter in &field_map.field_regex_contains_any {
            qwen_sft_hash_bytes(hash, format!("{:?}", filter.field).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, filter.pattern.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
        }
    }
    if !field_map.field_regex_excludes_any.is_empty() {
        qwen_sft_hash_bytes(hash, b"field_regex_excludes_any");
        for filter in &field_map.field_regex_excludes_any {
            qwen_sft_hash_bytes(hash, format!("{:?}", filter.field).as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, filter.pattern.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
        }
    }
    if !field_map.system_contains_any.is_empty() {
        qwen_sft_hash_bytes(hash, b"system_contains_any");
        for needle in &field_map.system_contains_any {
            qwen_sft_hash_bytes(hash, needle.as_bytes());
            qwen_sft_hash_bytes(hash, b",");
        }
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if !field_map.system_excludes_any.is_empty() {
        qwen_sft_hash_bytes(hash, b"system_excludes_any");
        for needle in &field_map.system_excludes_any {
            qwen_sft_hash_bytes(hash, needle.as_bytes());
            qwen_sft_hash_bytes(hash, b",");
        }
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(min_instruction_chars) = field_map.min_instruction_chars {
        qwen_sft_hash_bytes(hash, b"min_instruction_chars");
        qwen_sft_hash_bytes(hash, min_instruction_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(max_instruction_chars) = field_map.max_instruction_chars {
        qwen_sft_hash_bytes(hash, b"max_instruction_chars");
        qwen_sft_hash_bytes(hash, max_instruction_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(min_input_chars) = field_map.min_input_chars {
        qwen_sft_hash_bytes(hash, b"min_input_chars");
        qwen_sft_hash_bytes(hash, min_input_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(max_input_chars) = field_map.max_input_chars {
        qwen_sft_hash_bytes(hash, b"max_input_chars");
        qwen_sft_hash_bytes(hash, max_input_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(min_system_chars) = field_map.min_system_chars {
        qwen_sft_hash_bytes(hash, b"min_system_chars");
        qwen_sft_hash_bytes(hash, min_system_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(max_system_chars) = field_map.max_system_chars {
        qwen_sft_hash_bytes(hash, b"max_system_chars");
        qwen_sft_hash_bytes(hash, max_system_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(min_prompt_chars) = field_map.min_prompt_chars {
        qwen_sft_hash_bytes(hash, b"min_prompt_chars");
        qwen_sft_hash_bytes(hash, min_prompt_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(max_prompt_chars) = field_map.max_prompt_chars {
        qwen_sft_hash_bytes(hash, b"max_prompt_chars");
        qwen_sft_hash_bytes(hash, max_prompt_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(min_sample_chars) = field_map.min_sample_chars {
        qwen_sft_hash_bytes(hash, b"min_sample_chars");
        qwen_sft_hash_bytes(hash, min_sample_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if let Some(max_sample_chars) = field_map.max_sample_chars {
        qwen_sft_hash_bytes(hash, b"max_sample_chars");
        qwen_sft_hash_bytes(hash, max_sample_chars.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if field_map.dedupe_samples {
        qwen_sft_hash_bytes(hash, b"dedupe_samples");
        qwen_sft_hash_bytes(hash, b"true");
        qwen_sft_hash_bytes(hash, b"\0");
    }
    if field_map.skip_invalid_records {
        qwen_sft_hash_bytes(hash, b"skip_invalid_records");
        qwen_sft_hash_bytes(hash, b"true");
        qwen_sft_hash_bytes(hash, b"\0");
    }
    qwen_sft_hash_bytes(hash, b"source_weights");
    for source_weight in &field_map.source_weights {
        qwen_sft_hash_bytes(hash, source_weight.to_string().as_bytes());
        qwen_sft_hash_bytes(hash, b",");
    }
    qwen_sft_hash_bytes(hash, b"\0");
    if !field_map.source_max_samples.is_empty() {
        qwen_sft_hash_bytes(hash, b"source_max_samples");
        for source_limit in &field_map.source_max_samples {
            qwen_sft_hash_bytes(hash, source_limit.to_string().as_bytes());
            qwen_sft_hash_bytes(hash, b",");
        }
        qwen_sft_hash_bytes(hash, b"\0");
    }
    qwen_sft_hash_source_field_overrides(
        hash,
        b"source_instruction_fields",
        &field_map.source_instruction_fields,
    );
    qwen_sft_hash_source_field_overrides(
        hash,
        b"source_input_fields",
        &field_map.source_input_fields,
    );
    qwen_sft_hash_source_field_overrides(
        hash,
        b"source_response_fields",
        &field_map.source_response_fields,
    );
    if !field_map.external_metadata.is_empty() {
        qwen_sft_hash_bytes(hash, b"external_metadata");
        for metadata in &field_map.external_metadata {
            qwen_sft_hash_bytes(hash, metadata.path.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
            qwen_sft_hash_bytes(hash, metadata.contents.as_bytes());
            qwen_sft_hash_bytes(hash, b"\0");
        }
    }
}

pub(crate) fn qwen_sft_hash_source_field_overrides(
    hash: &mut u64,
    label: &[u8],
    values: &[String],
) {
    if values.is_empty() {
        return;
    }
    qwen_sft_hash_bytes(hash, label);
    for value in values {
        qwen_sft_hash_bytes(hash, value.as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
}

pub(crate) fn qwen_sft_hash_example(
    hash: &mut u64,
    example: &QwenSftExample,
    include_system: bool,
) {
    if include_system {
        qwen_sft_hash_bytes(hash, b"system");
        qwen_sft_hash_bytes(hash, example.system.as_bytes());
        qwen_sft_hash_bytes(hash, b"\0");
    }
    qwen_sft_hash_bytes(hash, b"instruction");
    qwen_sft_hash_bytes(hash, example.instruction.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0input");
    qwen_sft_hash_bytes(hash, example.input.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0response");
    qwen_sft_hash_bytes(hash, example.response.as_bytes());
    qwen_sft_hash_bytes(hash, b"\0");
}

pub(crate) fn qwen_sft_hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

pub(crate) fn qwen_validate_sft_resume_dataset(
    manifest_source_files: &[String],
    manifest_source_sample_counts: &[QwenSftSourceSampleCount],
    manifest_fingerprint: &str,
    manifest_shuffle: bool,
    dataset_summary: &QwenSftDatasetSummary,
    context: &str,
) -> Result<()> {
    qwen_validate_optional_sft_resume_dataset(
        manifest_source_files,
        manifest_source_sample_counts,
        manifest_fingerprint,
        manifest_shuffle,
        Some(&dataset_summary.source_files),
        Some(&dataset_summary.source_sample_counts),
        Some(&dataset_summary.fingerprint),
        Some(dataset_summary.shuffle),
        context,
    )
}

pub(crate) fn qwen_validate_optional_sft_resume_dataset(
    manifest_source_files: &[String],
    manifest_source_sample_counts: &[QwenSftSourceSampleCount],
    manifest_fingerprint: &str,
    manifest_shuffle: bool,
    dataset_source_files: Option<&[String]>,
    dataset_source_sample_counts: Option<&[QwenSftSourceSampleCount]>,
    dataset_fingerprint: Option<&str>,
    dataset_shuffle: Option<bool>,
    context: &str,
) -> Result<()> {
    if manifest_fingerprint.is_empty()
        && manifest_source_files.is_empty()
        && manifest_source_sample_counts.is_empty()
    {
        return Ok(());
    }
    let Some(dataset_fingerprint) = dataset_fingerprint else {
        bail!("{context} manifest has dataset provenance but current run has no JSONL dataset");
    };
    if dataset_fingerprint.is_empty() {
        bail!("{context} current JSONL dataset fingerprint is empty");
    }
    if manifest_fingerprint != dataset_fingerprint {
        bail!(
            "{context} dataset fingerprint mismatch: manifest={manifest_fingerprint}, current={dataset_fingerprint}"
        );
    }
    let Some(dataset_shuffle) = dataset_shuffle else {
        bail!("{context} manifest has dataset shuffle provenance but current run has none");
    };
    if manifest_shuffle != dataset_shuffle {
        bail!(
            "{context} dataset shuffle mismatch: manifest={manifest_shuffle}, current={dataset_shuffle}"
        );
    }
    let Some(dataset_source_files) = dataset_source_files else {
        bail!("{context} manifest has dataset source files but current run has none");
    };
    if manifest_source_files != dataset_source_files {
        bail!(
            "{context} dataset source files mismatch: manifest={manifest_source_files:?}, current={dataset_source_files:?}"
        );
    }
    if !manifest_source_sample_counts.is_empty() {
        let Some(dataset_source_sample_counts) = dataset_source_sample_counts else {
            bail!("{context} manifest has dataset source sample counts but current run has none");
        };
        if manifest_source_sample_counts != dataset_source_sample_counts {
            bail!(
                "{context} dataset source sample counts mismatch: manifest={manifest_source_sample_counts:?}, current={dataset_source_sample_counts:?}"
            );
        }
    }
    Ok(())
}

pub(crate) fn qwen_sft_token_sample(
    tokenizer: &Tokenizer,
    example: &QwenSftExample,
    field_map: &QwenSftFieldMap,
) -> Result<QwenSftTokenSample> {
    let prompt = qwen_render_sft_prompt(example, field_map)?;
    qwen_sft_token_sample_from_prompt(tokenizer, &prompt, &example.response)
}

pub(crate) fn qwen_render_sft_prompt(
    example: &QwenSftExample,
    field_map: &QwenSftFieldMap,
) -> Result<String> {
    field_map.validate()?;
    let template = if example.input.trim().is_empty() {
        &field_map.prompt_template
    } else {
        &field_map.prompt_with_input_template
    };
    Ok(template
        .replace("{system}", &example.system)
        .replace("{instruction}", &example.instruction)
        .replace("{input}", &example.input))
}

pub(crate) fn qwen_decode_cli_template_escapes(value: &str) -> String {
    let mut decoded = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            decoded.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => decoded.push('\n'),
            Some('r') => decoded.push('\r'),
            Some('t') => decoded.push('\t'),
            Some('\\') => decoded.push('\\'),
            Some('"') => decoded.push('"'),
            Some(other) => {
                decoded.push('\\');
                decoded.push(other);
            }
            None => decoded.push('\\'),
        }
    }
    decoded
}

pub(crate) fn qwen_sft_token_sample_from_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    response: &str,
) -> Result<QwenSftTokenSample> {
    let response = format!("{response}\n");
    let prompt_encoding = tokenizer
        .encode(prompt, false)
        .map_err(|error| anyhow!("failed to encode prompt: {error}"))?;
    let response_encoding = tokenizer
        .encode(response.as_str(), false)
        .map_err(|error| anyhow!("failed to encode response: {error}"))?;
    let prompt_tokens: Vec<i64> = prompt_encoding
        .get_ids()
        .iter()
        .map(|token| i64::from(*token))
        .collect();
    let response_tokens: Vec<i64> = response_encoding
        .get_ids()
        .iter()
        .map(|token| i64::from(*token))
        .collect();
    if prompt_tokens.is_empty() || response_tokens.is_empty() {
        bail!("SFT prompt and response must both tokenize to at least one token");
    }

    let mut token_ids = prompt_tokens.clone();
    token_ids.extend(response_tokens.iter().copied());
    if token_ids.len() < 2 {
        bail!("SFT sample must contain at least two tokens");
    }
    let target_len = token_ids.len() - 1;
    let prompt_len = prompt_tokens.len();
    let mask_values: Vec<f32> = (0..target_len)
        .map(|target_index| {
            if target_index + 1 >= prompt_len {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let masked_positions = mask_values.iter().filter(|value| **value > 0.0).count();
    if masked_positions == 0 {
        bail!("SFT response-only mask is empty");
    }

    Ok(QwenSftTokenSample {
        prompt_tokens: prompt_tokens.len(),
        response_tokens: response_tokens.len(),
        masked_positions,
        token_ids,
        mask_values,
    })
}

pub(crate) fn qwen_pad_token_id(tokenizer: &Tokenizer) -> i64 {
    tokenizer
        .get_padding()
        .map(|padding| i64::from(padding.pad_id))
        .or_else(|| tokenizer.token_to_id("<|endoftext|>").map(i64::from))
        .unwrap_or(0)
}

pub(crate) fn qwen_sft_padded_batch(
    samples: &[QwenSftTokenSample],
    pad_token_id: i64,
) -> Result<QwenSftBatch> {
    if samples.is_empty() {
        bail!("SFT batch must contain at least one sample");
    }
    let max_len = samples
        .iter()
        .map(|sample| sample.token_ids.len())
        .max()
        .ok_or_else(|| anyhow!("SFT batch must contain at least one sample"))?;
    if max_len < 2 {
        bail!("SFT batch sequence length must be at least two tokens");
    }

    let batch_size = samples.len();
    let mut input_values = Vec::with_capacity(batch_size * max_len);
    let mut mask_values = Vec::with_capacity(batch_size * (max_len - 1));
    let mut prompt_tokens = Vec::with_capacity(batch_size);
    let mut response_tokens = Vec::with_capacity(batch_size);
    let mut masked_positions = 0usize;
    let mut padding_tokens = 0usize;

    for sample in samples {
        prompt_tokens.push(sample.prompt_tokens);
        response_tokens.push(sample.response_tokens);
        input_values.extend(sample.token_ids.iter().copied());
        let pad_len = max_len - sample.token_ids.len();
        input_values.extend(std::iter::repeat(pad_token_id).take(pad_len));
        padding_tokens += pad_len;

        mask_values.extend(sample.mask_values.iter().copied());
        masked_positions += sample.masked_positions;
        mask_values.extend(std::iter::repeat(0.0).take(max_len - 1 - sample.mask_values.len()));
    }

    if masked_positions == 0 {
        bail!("SFT batch response-only mask is empty");
    }

    Ok(QwenSftBatch {
        input_ids: Tensor::from_slice(&input_values)
            .to_kind(Kind::Int64)
            .reshape([batch_size as i64, max_len as i64]),
        target_mask: Tensor::from_slice(&mask_values).reshape([
            batch_size as i64,
            (max_len - 1) as i64,
            1,
        ]),
        prompt_tokens,
        response_tokens,
        masked_positions,
        padding_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_module::test_utils::*;

    #[test]
    fn qwen_sft_padded_batch_masks_padding_targets() {
        let samples = vec![
            QwenSftTokenSample {
                prompt_tokens: 2,
                response_tokens: 2,
                masked_positions: 2,
                token_ids: vec![10, 11, 12, 13],
                mask_values: vec![0.0, 1.0, 1.0],
            },
            QwenSftTokenSample {
                prompt_tokens: 1,
                response_tokens: 1,
                masked_positions: 1,
                token_ids: vec![20, 21],
                mask_values: vec![1.0],
            },
        ];

        let batch = qwen_sft_padded_batch(&samples, 0).expect("batch should build");
        let input_values: Vec<i64> = Vec::<i64>::try_from(batch.input_ids.reshape([-1])).unwrap();
        let mask_values: Vec<f32> = Vec::<f32>::try_from(batch.target_mask.reshape([-1])).unwrap();

        assert_eq!(batch.input_ids.size(), vec![2, 4]);
        assert_eq!(batch.target_mask.size(), vec![2, 3, 1]);
        assert_eq!(input_values, vec![10, 11, 12, 13, 20, 21, 0, 0]);
        assert_eq!(mask_values, vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(batch.masked_positions, 3);
        assert_eq!(batch.padding_tokens, 2);
    }

    #[test]
    fn qwen_sft_dataset_builds_wrapping_padded_batches() {
        let dataset = QwenSftDataset {
            samples: vec![
                QwenSftTokenSample {
                    prompt_tokens: 2,
                    response_tokens: 1,
                    masked_positions: 1,
                    token_ids: vec![1, 2, 3],
                    mask_values: vec![0.0, 1.0],
                },
                QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: 2,
                    masked_positions: 2,
                    token_ids: vec![4, 5, 6],
                    mask_values: vec![1.0, 1.0],
                },
            ],
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        };

        let batch = dataset
            .padded_batch(1, 3)
            .expect("wrapping batch should build");
        let input_values: Vec<i64> = Vec::<i64>::try_from(batch.input_ids.reshape([-1])).unwrap();
        let mask_values: Vec<f32> = Vec::<f32>::try_from(batch.target_mask.reshape([-1])).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(batch.input_ids.size(), vec![3, 3]);
        assert_eq!(input_values, vec![4, 5, 6, 1, 2, 3, 4, 5, 6]);
        assert_eq!(mask_values, vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(batch.masked_positions, 5);
        assert_eq!(batch.padding_tokens, 0);
    }

    #[test]
    fn qwen_sft_dataset_split_keeps_train_and_eval_batches() {
        let dataset = QwenSftDataset {
            samples: (0..5)
                .map(|index| QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: 1,
                    masked_positions: 1,
                    token_ids: vec![index, index + 10],
                    mask_values: vec![1.0],
                })
                .collect(),
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        };

        let (train, eval) = dataset
            .train_eval_split(0.6)
            .expect("split should keep both sides");
        let train_batch = train
            .padded_batch(2, 3)
            .expect("train wrapping batch should build");
        let eval_batch = eval.padded_batch(0, 2).expect("eval batch should build");
        let train_values: Vec<i64> =
            Vec::<i64>::try_from(train_batch.input_ids.reshape([-1])).unwrap();
        let eval_values: Vec<i64> =
            Vec::<i64>::try_from(eval_batch.input_ids.reshape([-1])).unwrap();

        assert_eq!(train.len(), 3);
        assert_eq!(eval.len(), 2);
        assert_eq!(train_values, vec![2, 12, 0, 10, 1, 11]);
        assert_eq!(eval_values, vec![3, 13, 4, 14]);
    }

    #[test]
    fn qwen_sft_dataset_shuffle_is_seeded_and_summarized() {
        let dataset = QwenSftDataset {
            samples: (0..5)
                .map(|index| QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: (index + 1) as usize,
                    masked_positions: (index + 1) as usize,
                    token_ids: vec![index, index + 10, index + 20],
                    mask_values: vec![0.0, 1.0],
                })
                .collect(),
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        };

        let summary = dataset.summary();
        let shuffled_a = dataset.clone().shuffle_by_seed(17);
        let shuffled_b = dataset.clone().shuffle_by_seed(17);
        let shuffled_c = dataset.shuffle_by_seed(18);
        let order_a = shuffled_a
            .samples
            .iter()
            .map(|sample| sample.token_ids[0])
            .collect::<Vec<_>>();
        let order_b = shuffled_b
            .samples
            .iter()
            .map(|sample| sample.token_ids[0])
            .collect::<Vec<_>>();
        let order_c = shuffled_c
            .samples
            .iter()
            .map(|sample| sample.token_ids[0])
            .collect::<Vec<_>>();

        assert_eq!(summary.samples, 5);
        assert_eq!(summary.total_tokens, 15);
        assert_eq!(summary.response_tokens, 15);
        assert_eq!(summary.masked_positions, 15);
        assert_eq!(summary.max_sequence_tokens, 3);
        assert!(!summary.shuffle);
        assert!(shuffled_a.summary().shuffle);
        assert_eq!(order_a, order_b);
        assert_ne!(order_a, order_c);
    }

    #[test]
    fn qwen_sft_dataset_shuffle_can_be_disabled() {
        let dataset = QwenSftDataset {
            samples: (0..5)
                .map(|index| QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: 1,
                    masked_positions: 1,
                    token_ids: vec![index, index + 10],
                    mask_values: vec![1.0],
                })
                .collect(),
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        };

        let unshuffled = qwen_apply_sft_shuffle(dataset.clone(), false, 777);
        let shuffled = qwen_apply_sft_shuffle(dataset, true, 777);
        let unshuffled_order = (0..unshuffled.len())
            .map(|cursor| unshuffled.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();
        let shuffled_order = (0..shuffled.len())
            .map(|cursor| shuffled.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();

        assert_eq!(unshuffled_order, vec![0, 1, 2, 3, 4]);
        assert!(!unshuffled.summary().shuffle);
        assert!(shuffled.summary().shuffle);
        assert_ne!(unshuffled_order, shuffled_order);
    }

    #[test]
    fn qwen_sft_dataset_epoch_shuffle_is_cursor_stable() {
        let dataset = QwenSftDataset {
            samples: (0..6)
                .map(|index| QwenSftTokenSample {
                    prompt_tokens: 1,
                    response_tokens: 1,
                    masked_positions: 1,
                    token_ids: vec![index, index + 10],
                    mask_values: vec![1.0],
                })
                .collect(),
            pad_token_id: 0,
            epoch_shuffle_seed: None,
            source_files: Vec::new(),
            source_sample_counts: Vec::new(),
            fingerprint: String::new(),
        }
        .shuffle_by_seed(777);

        let epoch0_a = (0..dataset.len())
            .map(|cursor| dataset.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();
        let epoch0_b = (0..dataset.len())
            .map(|cursor| dataset.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();
        let epoch1 = (dataset.len()..dataset.len() * 2)
            .map(|cursor| dataset.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();
        let epoch2 = (dataset.len() * 2..dataset.len() * 3)
            .map(|cursor| dataset.sample_at_cursor(cursor).unwrap().token_ids[0])
            .collect::<Vec<_>>();

        assert_eq!(epoch0_a, epoch0_b);
        assert!(epoch0_a != epoch1 || epoch0_a != epoch2);

        let wrapped_batch = dataset
            .padded_batch(dataset.len() - 1, 3)
            .expect("epoch-crossing batch should build");
        let wrapped_values: Vec<i64> =
            Vec::<i64>::try_from(wrapped_batch.input_ids.reshape([-1])).unwrap();
        assert_eq!(
            wrapped_values,
            vec![
                epoch0_a[dataset.len() - 1],
                epoch0_a[dataset.len() - 1] + 10,
                epoch1[0],
                epoch1[0] + 10,
                epoch1[1],
                epoch1[1] + 10,
            ]
        );
    }

    #[test]
    fn qwen_sft_jsonl_reader_loads_instruction_input_response_records() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        fs::write(
            &jsonl,
            r#"{"instruction":"Reply with the project name.","response":"rustrain"}
{"instruction":"Name the language.","input":"rustrain implementation","response":"Rust"}
"#,
        )
        .expect("jsonl should write");

        let field_map = QwenSftFieldMap::default();
        let regex_plan = QwenSftRegexPlan::compile(&field_map).expect("regex plan should compile");
        let example_set = qwen_sft_examples_from_jsonl_path_with_limit(
            &jsonl,
            None,
            None,
            1,
            &field_map,
            &regex_plan,
            &mut None,
        )
        .expect("examples should load from jsonl");
        let examples = &example_set.examples;

        assert_eq!(examples.len(), 2);
        assert_eq!(example_set.source_files, vec![jsonl.display().to_string()]);
        assert_eq!(
            example_set.source_sample_counts,
            vec![QwenSftSourceSampleCount {
                path: jsonl.display().to_string(),
                samples: 2,
            }]
        );
        assert!(!example_set.fingerprint.is_empty());
        assert_eq!(examples[0].instruction, "Reply with the project name.");
        assert_eq!(examples[0].input, "");
        assert_eq!(examples[0].response, "rustrain");
        assert_eq!(examples[1].instruction, "Name the language.");
        assert_eq!(examples[1].input, "rustrain implementation");
        assert_eq!(examples[1].response, "Rust");
    }

    #[test]
    fn qwen_sft_jsonl_reader_supports_configurable_field_names() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"sys":"Be concise.","prompt":"Summarize the project.","context":"Rust training code","answer":"rustrain"}
{"sys":"Use one word.","prompt":"Name the language.","answer":"Rust"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction: "prompt".to_string(),
            input: "context".to_string(),
            response: "answer".to_string(),
            system: Some("sys".to_string()),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("custom field examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("custom field streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("custom field raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("custom field cache should write");
        let mismatched_field_map = QwenSftFieldMap {
            response: "completion".to_string(),
            ..field_map.clone()
        };
        let mismatched_system_map = QwenSftFieldMap {
            system: Some("system_prompt".to_string()),
            ..field_map.clone()
        };
        let mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &mismatched_field_map,
        )
        .expect_err("cache should reject different field maps")
        .to_string();
        let system_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &mismatched_system_map,
        )
        .expect_err("cache should reject different system fields")
        .to_string();

        assert_eq!(loaded.examples.len(), 2);
        assert_eq!(loaded.examples[0].system, "Be concise.");
        assert_eq!(loaded.examples[0].instruction, "Summarize the project.");
        assert_eq!(loaded.examples[0].input, "Rust training code");
        assert_eq!(loaded.examples[0].response, "rustrain");
        assert_eq!(loaded.examples[1].system, "Use one word.");
        assert_eq!(loaded.examples[1].input, "");
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(raw_window.examples[0].response, "rustrain");
        assert!(first_cache.cache_written);
        assert!(mismatch.contains("field_map"));
        assert!(system_mismatch.contains("field_map"));
        assert!(
            qwen_render_sft_prompt(&loaded.examples[0], &field_map)
                .expect("system prompt should render")
                .contains("System: Be concise.")
        );
    }

    #[test]
    fn qwen_sft_jsonl_reader_supports_dotted_field_paths() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("nested.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"meta":{"system":"Be exact."},"payload":{"prompt":"Name the project.","context":"Rust trainer","answer":"rustrain"}}
{"meta":{"system":"Use one word."},"payload":{"prompt":"Name the language.","answer":"Rust"}}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction: "payload.prompt".to_string(),
            input: "payload.context".to_string(),
            response: "payload.answer".to_string(),
            system: Some("meta.system".to_string()),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("nested field examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("nested field streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("nested field source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("nested field raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("nested field cache should write");
        let changed_field_map = QwenSftFieldMap {
            instruction: "payload.question".to_string(),
            ..field_map.clone()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &changed_field_map,
        )
        .expect_err("cache should reject changed dotted field paths")
        .to_string();

        assert_eq!(loaded.examples.len(), 2);
        assert_eq!(loaded.examples[0].system, "Be exact.");
        assert_eq!(loaded.examples[0].instruction, "Name the project.");
        assert_eq!(loaded.examples[0].input, "Rust trainer");
        assert_eq!(loaded.examples[0].response, "rustrain");
        assert_eq!(loaded.examples[1].system, "Use one word.");
        assert_eq!(loaded.examples[1].input, "");
        assert_eq!(loaded.examples[1].response, "Rust");
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(raw_window.examples, loaded.examples);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert!(
            qwen_render_sft_prompt(&loaded.examples[0], &field_map)
                .expect("nested field prompt should render")
                .contains("I: Rust trainer")
        );
    }

    #[test]
    fn qwen_sft_jsonl_reader_supports_chat_messages_records() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("chat.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"Name the project."},{"role":"assistant","content":"rustrain"}]}
{"messages":[{"role":"human","content":"Name the language."},{"role":"gpt","content":"Rust"}]}
{"messages":[{"role":"assistant","content":"missing user"}]}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            chat_messages: Some("messages".to_string()),
            system_contains_any: vec!["concise".to_string()],
            skip_invalid_records: true,
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("chat examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("chat streaming summary should scan");
        let source_index =
            qwen_sft_streaming_source_index(&paths, None, &field_map).expect("index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("chat raw window should replay");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("chat cache should write");
        let cache_hit =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("chat cache should hit");
        let default_parse_error = qwen_sft_examples_from_jsonl_paths_with_limit(
            &paths,
            None,
            &QwenSftFieldMap::default(),
        )
        .expect_err("default field map cannot parse chat rows")
        .to_string();
        let changed_chat_field = QwenSftFieldMap {
            chat_messages: Some("conversations".to_string()),
            ..field_map.clone()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &changed_chat_field,
        )
        .expect_err("cache should reject changed chat messages field")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].system, "Be concise.");
        assert_eq!(loaded.examples[0].instruction, "Name the project.");
        assert_eq!(loaded.examples[0].input, "");
        assert_eq!(loaded.examples[0].response, "rustrain");
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(source_index.samples.len(), 1);
        assert_eq!(source_index.samples[0].index_in_file, 0);
        assert!(first_cache.cache_written);
        assert!(cache_hit.cache_hit);
        assert_eq!(cache_hit.index.samples, source_index.samples);
        assert!(default_parse_error.contains("failed to parse SFT JSONL record"));
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_field_defaults_fill_empty_fields_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"","response":"","input":""}
{"response":"kept"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            system_contains_any: vec!["assistant".to_string()],
            instruction_contains_any: vec!["default instruction".to_string()],
            input_contains_any: vec!["default input".to_string()],
            response_contains_any: vec!["default response".to_string()],
            min_response_chars: 10,
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
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("defaulted examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("defaulted streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("defaulted source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("defaulted raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("defaulted cache should write");
        let defaults_without_filters = QwenSftFieldMap {
            system_contains_any: Vec::new(),
            instruction_contains_any: Vec::new(),
            input_contains_any: Vec::new(),
            response_contains_any: Vec::new(),
            min_response_chars: 1,
            ..field_map.clone()
        };
        let unfiltered_default_fingerprint = qwen_sft_streaming_fingerprint(
            &paths,
            None,
            &loaded.source_files,
            &defaults_without_filters,
        )
        .expect("defaulted field map fingerprint should compute");
        let changed_defaults = QwenSftFieldMap {
            field_defaults: vec![FieldDefault {
                field: FieldDefaultTarget::Instruction,
                value: "other instruction".to_string(),
            }],
            ..field_map.clone()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &changed_defaults,
        )
        .expect_err("cache should reject changed defaults")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].system, "system assistant");
        assert_eq!(loaded.examples[0].instruction, "default instruction");
        assert_eq!(loaded.examples[0].input, "default input");
        assert_eq!(loaded.examples[0].response, "default response");
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(raw_window.examples, loaded.examples);
        assert!(first_cache.cache_written);
        assert_ne!(loaded.fingerprint, unfiltered_default_fingerprint);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_field_replacements_apply_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Keep PROJECT_TOKEN","response":"APPROVED_TOKEN"}
{"instruction":"Drop PROJECT_TOKEN","response":"DENIED_TOKEN"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["rustrain".to_string()],
            response_contains_any: vec!["approved".to_string()],
            min_response_chars: 8,
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
            ..QwenSftFieldMap::default()
        };
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("replacement examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("replacement streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("replacement source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("replacement raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("replacement cache should write");
        let default_fingerprint = qwen_sft_streaming_fingerprint(
            &paths,
            None,
            &loaded.source_files,
            &QwenSftFieldMap::default(),
        )
        .expect("default fingerprint should compute");
        let replacement_mismatch = QwenSftFieldMap {
            field_replacements: vec![FieldReplacement {
                field: FieldReplacementTarget::Instruction,
                pattern: "PROJECT_TOKEN".to_string(),
                replacement: "other".to_string(),
            }],
            ..field_map.clone()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &replacement_mismatch,
        )
        .expect_err("cache should reject changed replacements")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "Keep rustrain");
        assert_eq!(loaded.examples[0].response, "approved");
        assert_eq!(raw_window.examples.len(), loaded.examples.len());
        assert_eq!(
            raw_window.examples[0].instruction,
            loaded.examples[0].instruction
        );
        assert_eq!(raw_window.examples[0].response, loaded.examples[0].response);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, default_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_regex_replacements_apply_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Keep project-123 token","response":"APPROVED:42"}
{"instruction":"Drop project-456 token","response":"DENIED:99"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["project-id token".to_string()],
            response_contains_any: vec!["approved id".to_string()],
            min_response_chars: 8,
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
            ..QwenSftFieldMap::default()
        };
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("regex replacement examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("regex replacement streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("regex replacement source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("regex replacement raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("regex replacement cache should write");
        let default_fingerprint = qwen_sft_streaming_fingerprint(
            &paths,
            None,
            &loaded.source_files,
            &QwenSftFieldMap::default(),
        )
        .expect("default fingerprint should compute");
        let regex_mismatch = QwenSftFieldMap {
            field_regex_replacements: vec![FieldRegexReplacement {
                field: FieldReplacementTarget::Instruction,
                pattern: r"project-\d+".to_string(),
                replacement: "other".to_string(),
            }],
            ..field_map.clone()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &regex_mismatch,
        )
        .expect_err("cache should reject changed regex replacements")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "Keep project-id token");
        assert_eq!(loaded.examples[0].response, "approved id");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, default_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_field_transform_dsl_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Keep PROJECT-123::tail","input":"gpu context","response":"APPROVED:42 extra"}
{"instruction":"Drop PROJECT-456::tail","input":"cpu context","response":"DENIED:99 extra"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["task Keep project-id".to_string()],
            input_contains_any: vec!["GPU".to_string()],
            response_contains_any: vec!["approved id".to_string()],
            min_response_chars: 11,
            max_response_chars: Some(11),
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
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("dsl-transformed examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("dsl-transformed streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("dsl-transformed source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("dsl-transformed raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("dsl-transformed cache should write");
        let raw_map = QwenSftFieldMap {
            field_transforms: Vec::new(),
            input_contains_any: Vec::new(),
            response_contains_any: Vec::new(),
            max_response_chars: None,
            ..field_map.clone()
        };
        let raw_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_map)
                .expect("raw fingerprint should compute");
        let cache_mismatch =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &raw_map)
                .expect_err("cache should reject changed DSL transforms")
                .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "task Keep project-id");
        assert_eq!(loaded.examples[0].input, "GPU CONTEXT");
        assert_eq!(loaded.examples[0].response, "approved id");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_filters_fields_by_regex_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Keep project-123 token","input":"GPU context","response":"approved answer"}
{"instruction":"Drop project-456 token","input":"GPU context","response":"DENIED answer"}
{"instruction":"Skip project-abc token","input":"GPU context","response":"approved answer"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            field_regex_contains_any: vec![FieldRegexFilter {
                field: FieldReplacementTarget::Instruction,
                pattern: r"project-\d+".to_string(),
            }],
            field_regex_excludes_any: vec![FieldRegexFilter {
                field: FieldReplacementTarget::Response,
                pattern: r"DENIED|blocked".to_string(),
            }],
            ..QwenSftFieldMap::default()
        };
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("regex-filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("regex-filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("regex-filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("regex-filtered raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("regex-filtered cache should write");
        let default_fingerprint = qwen_sft_streaming_fingerprint(
            &paths,
            None,
            &loaded.source_files,
            &QwenSftFieldMap::default(),
        )
        .expect("default fingerprint should compute");
        let filter_mismatch = QwenSftFieldMap {
            field_regex_contains_any: vec![FieldRegexFilter {
                field: FieldReplacementTarget::Instruction,
                pattern: r"project-[a-z]+".to_string(),
            }],
            ..field_map.clone()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &filter_mismatch,
        )
        .expect_err("cache should reject changed regex filters")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "Keep project-123 token");
        assert_eq!(loaded.examples[0].response, "approved answer");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(source_index.samples.len(), 1);
        assert_eq!(source_index.samples[0].index_in_file, 0);
        assert_ne!(loaded.fingerprint, default_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_reuses_compiled_regex_plan_across_records() {
        let field_map = QwenSftFieldMap {
            response_contains_any: vec!["approved id".to_string()],
            field_regex_replacements: vec![FieldRegexReplacement {
                field: FieldReplacementTarget::Response,
                pattern: r"APPROVED:\d+".to_string(),
                replacement: "approved id".to_string(),
            }],
            field_regex_contains_any: vec![FieldRegexFilter {
                field: FieldReplacementTarget::Instruction,
                pattern: r"project-\d+".to_string(),
            }],
            field_regex_excludes_any: vec![FieldRegexFilter {
                field: FieldReplacementTarget::Input,
                pattern: r"blocked".to_string(),
            }],
            ..QwenSftFieldMap::default()
        };
        let regex_plan = QwenSftRegexPlan::compile(&field_map).expect("regex plan should compile");
        let first = qwen_sft_record_from_jsonl_line(
            r#"{"instruction":"Keep project-123","input":"GPU context","response":"APPROVED:42"}"#,
            &field_map,
            &regex_plan,
        )
        .expect("first record should parse");
        let second = qwen_sft_record_from_jsonl_line(
            r#"{"instruction":"Drop project-456","input":"blocked context","response":"APPROVED:99"}"#,
            &field_map,
            &regex_plan,
        )
        .expect("second record should parse");

        assert_eq!(first.response, "approved id");
        assert!(qwen_sft_record_passes_filters(
            &first,
            &field_map,
            &regex_plan
        ));
        assert_eq!(second.response, "approved id");
        assert!(!qwen_sft_record_passes_filters(
            &second,
            &field_map,
            &regex_plan
        ));
        assert_eq!(regex_plan.replacements.len(), 1);
        assert_eq!(regex_plan.contains_any.len(), 1);
        assert_eq!(regex_plan.excludes_any.len(), 1);
    }

    #[test]
    fn qwen_sft_normalizes_whitespace_after_replacements_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Keep PROJECT_TOKEN","response":"approved\t\tanswer"}
{"instruction":"Also PROJECT_TOKEN","response":"approved   answer"}
{"instruction":"Drop PROJECT_TOKEN","response":"denied   answer"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["rust train".to_string()],
            response_contains_any: vec!["approved answer".to_string()],
            field_replacements: vec![FieldReplacement {
                field: FieldReplacementTarget::Instruction,
                pattern: "PROJECT_TOKEN".to_string(),
                replacement: "rust   train".to_string(),
            }],
            field_regex_replacements: Vec::new(),
            normalize_whitespace: true,
            ..QwenSftFieldMap::default()
        };
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("normalized examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("normalized streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("normalized source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("normalized raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("normalized cache should write");
        let raw_map = QwenSftFieldMap {
            normalize_whitespace: false,
            ..field_map.clone()
        };
        let cache_mismatch =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &raw_map)
                .expect_err("cache should reject changed whitespace normalization")
                .to_string();
        let raw_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_map)
                .expect("raw fingerprint should compute");

        assert_eq!(loaded.examples.len(), 2);
        assert_eq!(loaded.examples[0].instruction, "Keep rust train");
        assert_eq!(loaded.examples[0].response, "approved answer");
        assert_eq!(raw_window.examples.len(), loaded.examples.len());
        assert_eq!(raw_window.examples[1].instruction, "Also rust train");
        assert_eq!(raw_window.examples[1].response, "approved answer");
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_case_transforms_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Keep PROJECT_TOKEN","input":"GPU CONTEXT","response":"APPROVED ANSWER"}
{"instruction":"Drop PROJECT_TOKEN","input":"GPU CONTEXT","response":"DENIED ANSWER"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["rustrain".to_string()],
            input_contains_any: vec!["gpu context".to_string()],
            response_contains_any: vec!["approved answer".to_string()],
            field_replacements: vec![FieldReplacement {
                field: FieldReplacementTarget::Instruction,
                pattern: "PROJECT_TOKEN".to_string(),
                replacement: "RUSTRain".to_string(),
            }],
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
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("case-transformed examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("case-transformed streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("case-transformed source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("case-transformed raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("case-transformed cache should write");
        let raw_case_map = QwenSftFieldMap {
            field_case_transforms: Vec::new(),
            ..field_map.clone()
        };
        let raw_case_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_case_map)
                .expect("raw-case fingerprint should compute");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &raw_case_map,
        )
        .expect_err("cache should reject changed case transforms")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "keep rustrain");
        assert_eq!(loaded.examples[0].input, "gpu context");
        assert_eq!(loaded.examples[0].response, "approved answer");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_case_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_field_affixes_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Name the project","input":"GPU","response":"rustrain"}
{"instruction":"Name the language","input":"GPU","response":"Rust"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            system_contains_any: vec!["system:".to_string()],
            instruction_contains_any: vec!["Q: Name".to_string()],
            input_contains_any: vec!["ctx=GPU".to_string()],
            response_contains_any: vec!["</answer>".to_string()],
            field_defaults: vec![FieldDefault {
                field: FieldDefaultTarget::System,
                value: "concise".to_string(),
            }],
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
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("affixed examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("affixed streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("affixed source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("affixed raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("affixed cache should write");
        let raw_affix_map = QwenSftFieldMap {
            field_affixes: Vec::new(),
            system_contains_any: Vec::new(),
            instruction_contains_any: Vec::new(),
            input_contains_any: Vec::new(),
            response_contains_any: Vec::new(),
            ..field_map.clone()
        };
        let raw_affix_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_affix_map)
                .expect("raw-affix fingerprint should compute");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &raw_affix_map,
        )
        .expect_err("cache should reject changed affixes")
        .to_string();

        assert_eq!(loaded.examples.len(), 2);
        assert_eq!(loaded.examples[0].system, "system: concise");
        assert_eq!(loaded.examples[0].instruction, "Q: Name the project?");
        assert_eq!(loaded.examples[0].input, "ctx=GPU");
        assert_eq!(loaded.examples[0].response, "rustrain</answer>");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_affix_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_field_strips_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"PROMPT: Keep GPU prompt","input":"ctx=GPU context","response":"<answer>approved answer</answer>"}
{"instruction":"PROMPT: Drop CPU prompt","input":"ctx=CPU context","response":"<answer>denied answer</answer>"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["Keep GPU".to_string()],
            input_contains_any: vec!["GPU context".to_string()],
            response_contains_any: vec!["approved answer".to_string()],
            response_excludes_any: vec!["</answer>".to_string()],
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
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("stripped examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("stripped streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("stripped source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("stripped raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("stripped cache should write");
        let raw_strip_map = QwenSftFieldMap {
            field_strips: Vec::new(),
            instruction_contains_any: Vec::new(),
            input_contains_any: Vec::new(),
            response_contains_any: Vec::new(),
            response_excludes_any: Vec::new(),
            ..field_map.clone()
        };
        let raw_strip_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_strip_map)
                .expect("raw-strip fingerprint should compute");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &raw_strip_map,
        )
        .expect_err("cache should reject changed strips")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "Keep GPU prompt");
        assert_eq!(loaded.examples[0].input, "GPU context");
        assert_eq!(loaded.examples[0].response, "approved answer");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_strip_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_field_truncations_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"Name the rustrain project precisely","input":"GPU worker context","response":"approved response with trailing text"}
{"instruction":"Name the rustrain project precisely","input":"CPU worker context","response":"denied response with trailing text"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["Name the rustrain".to_string()],
            input_contains_any: vec!["GPU".to_string()],
            response_contains_any: vec!["approved response".to_string()],
            min_response_chars: 17,
            max_response_chars: Some(17),
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
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("truncated examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("truncated streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("truncated source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("truncated raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("truncated cache should write");
        let raw_truncation_map = QwenSftFieldMap {
            field_truncations: Vec::new(),
            field_transforms: Vec::new(),
            input_contains_any: Vec::new(),
            response_contains_any: Vec::new(),
            max_response_chars: None,
            ..field_map.clone()
        };
        let raw_truncation_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_truncation_map)
                .expect("raw-truncation fingerprint should compute");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &raw_truncation_map,
        )
        .expect_err("cache should reject changed truncations")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "Name the rustrain");
        assert_eq!(loaded.examples[0].input, "GPU");
        assert_eq!(loaded.examples[0].response, "approved response");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_truncation_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_field_splits_before_filters_and_fingerprint() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"metadata :: Keep GPU prompt","input":"GPU context || discard","response":"draft -> approved answer"}
{"instruction":"metadata :: Drop CPU prompt","input":"CPU context || discard","response":"draft -> denied answer"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["Keep GPU".to_string()],
            input_contains_any: vec!["GPU context".to_string()],
            response_contains_any: vec!["approved answer".to_string()],
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
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("split examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, None, &field_map)
            .expect("split streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, None, &field_map)
            .expect("split source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("split raw window should read");
        let first_cache =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("split cache should write");
        let raw_split_map = QwenSftFieldMap {
            field_splits: Vec::new(),
            instruction_contains_any: Vec::new(),
            input_contains_any: Vec::new(),
            response_contains_any: Vec::new(),
            ..field_map.clone()
        };
        let raw_split_fingerprint =
            qwen_sft_streaming_fingerprint(&paths, None, &loaded.source_files, &raw_split_map)
                .expect("raw-split fingerprint should compute");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &raw_split_map,
        )
        .expect_err("cache should reject changed splits")
        .to_string();

        assert_eq!(loaded.examples.len(), 1);
        assert_eq!(loaded.examples[0].instruction, "Keep GPU prompt");
        assert_eq!(loaded.examples[0].input, "GPU context");
        assert_eq!(loaded.examples[0].response, "approved answer");
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_ne!(loaded.fingerprint, raw_split_fingerprint);
        assert!(first_cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_prompt_template_changes_tokenized_prompt_and_fingerprint() {
        let example = QwenSftExample {
            system: String::new(),
            instruction: "Name the project.".to_string(),
            input: "Rust trainer".to_string(),
            response: "rustrain".to_string(),
        };
        let default_map = QwenSftFieldMap::default();
        let custom_map = QwenSftFieldMap {
            prompt_template: "### User\n{instruction}\n### Assistant\n".to_string(),
            prompt_with_input_template:
                "### User\n{instruction}\nContext: {input}\n### Assistant\n".to_string(),
            ..QwenSftFieldMap::default()
        };
        let default_prompt =
            qwen_render_sft_prompt(&example, &default_map).expect("default prompt should render");
        let custom_prompt =
            qwen_render_sft_prompt(&example, &custom_map).expect("custom prompt should render");
        let default_fingerprint = qwen_sft_dataset_fingerprint(
            &["data/train.jsonl".to_string()],
            std::slice::from_ref(&example),
            &default_map,
        );
        let custom_fingerprint = qwen_sft_dataset_fingerprint(
            &["data/train.jsonl".to_string()],
            std::slice::from_ref(&example),
            &custom_map,
        );

        assert_eq!(
            default_prompt,
            "Instruction:\nName the project.\n\nInput:\nRust trainer\n\nResponse:\n"
        );
        assert_eq!(
            custom_prompt,
            "### User\nName the project.\nContext: Rust trainer\n### Assistant\n"
        );
        assert_ne!(default_fingerprint, custom_fingerprint);
    }

    #[test]
    fn qwen_sft_cli_template_escapes_match_config_templates() {
        let example = QwenSftExample {
            system: String::new(),
            instruction: "Name the project.".to_string(),
            input: "Rust trainer".to_string(),
            response: "rustrain".to_string(),
        };
        let field_map = QwenSftFieldMap {
            prompt_template: qwen_decode_cli_template_escapes(
                "Instruction: {instruction}\\nResponse: ",
            ),
            prompt_with_input_template: qwen_decode_cli_template_escapes(
                "Instruction: {instruction}\\nInput: {input}\\nResponse: ",
            ),
            ..QwenSftFieldMap::default()
        };

        let prompt =
            qwen_render_sft_prompt(&example, &field_map).expect("CLI prompt should render");

        assert_eq!(
            prompt,
            "Instruction: Name the project.\nInput: Rust trainer\nResponse: "
        );
        assert_eq!(qwen_decode_cli_template_escapes(r"one\ttwo"), "one\ttwo");
        assert_eq!(qwen_decode_cli_template_escapes(r"one\\two"), r"one\two");
        assert_eq!(qwen_decode_cli_template_escapes(r#"one\"two"#), "one\"two");
    }

    #[test]
    fn qwen_sft_trim_fields_controls_record_normalization_and_fingerprint() {
        let line = r#"{"instruction":"  Name the project.  ","input":"  Rust trainer  ","response":"  rustrain  "}"#;
        let trim_map = QwenSftFieldMap::default();
        let raw_map = QwenSftFieldMap {
            trim_fields: false,
            ..QwenSftFieldMap::default()
        };
        let trim_regex_plan =
            QwenSftRegexPlan::compile(&trim_map).expect("trim regex plan should compile");
        let raw_regex_plan =
            QwenSftRegexPlan::compile(&raw_map).expect("raw regex plan should compile");
        let trimmed = qwen_sft_record_from_jsonl_line(line, &trim_map, &trim_regex_plan)
            .expect("trimmed record should parse");
        let raw = qwen_sft_record_from_jsonl_line(line, &raw_map, &raw_regex_plan)
            .expect("raw record should parse");
        let trimmed_example = QwenSftExample {
            system: trimmed.system.clone(),
            instruction: trimmed.instruction.clone(),
            input: trimmed.input.clone(),
            response: trimmed.response.clone(),
        };
        let raw_example = QwenSftExample {
            system: raw.system.clone(),
            instruction: raw.instruction.clone(),
            input: raw.input.clone(),
            response: raw.response.clone(),
        };

        assert_eq!(trimmed.instruction, "Name the project.");
        assert_eq!(trimmed.input, "Rust trainer");
        assert_eq!(trimmed.response, "rustrain");
        assert_eq!(raw.instruction, "  Name the project.  ");
        assert_eq!(raw.input, "  Rust trainer  ");
        assert_eq!(raw.response, "  rustrain  ");
        assert_ne!(
            qwen_sft_dataset_fingerprint(&[], &[trimmed_example], &trim_map),
            qwen_sft_dataset_fingerprint(&[], &[raw_example], &raw_map)
        );
    }

    #[test]
    fn qwen_sft_explicit_eval_metadata_combines_train_and_eval_sources() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let train_jsonl = temp.path().join("train.jsonl");
        let eval_jsonl = temp.path().join("eval.jsonl");
        let train_file = train_jsonl.display().to_string();
        let eval_file = eval_jsonl.display().to_string();
        let source_files = qwen_merge_sft_source_files(
            std::slice::from_ref(&train_file),
            std::slice::from_ref(&eval_file),
        );
        let source_counts = qwen_merge_sft_source_sample_counts(
            &[QwenSftSourceSampleCount {
                path: train_file.clone(),
                samples: 3,
            }],
            &[QwenSftSourceSampleCount {
                path: eval_file.clone(),
                samples: 2,
            }],
        );
        let fingerprint =
            qwen_combine_sft_fingerprints(&source_files, "train-fingerprint", "eval-fingerprint");

        assert_eq!(source_files, vec![eval_file.clone(), train_file.clone()]);
        assert_eq!(
            source_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: eval_file,
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: train_file,
                    samples: 3,
                },
            ]
        );
        assert!(!fingerprint.is_empty());
        assert_ne!(
            fingerprint,
            qwen_combine_sft_fingerprints(&source_files, "train-fingerprint", "other-eval")
        );
    }

    #[test]
    fn qwen_sft_limits_explicit_eval_paths() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let train_jsonl = temp.path().join("train.jsonl");
        let eval_jsonl = temp.path().join("eval.jsonl");
        fs::write(
            &train_jsonl,
            "{\"instruction\":\"train one\",\"response\":\"alpha\"}\n{\"instruction\":\"train two\",\"response\":\"beta\"}\n",
        )
        .expect("train jsonl should write");
        fs::write(
            &eval_jsonl,
            "{\"instruction\":\"eval one\",\"response\":\"gamma\"}\n{\"instruction\":\"eval two\",\"response\":\"delta\"}\n{\"instruction\":\"eval three\",\"response\":\"epsilon\"}\n",
        )
        .expect("eval jsonl should write");
        let train_paths = vec![train_jsonl.clone()];
        let eval_paths = vec![eval_jsonl.clone()];
        let field_map = QwenSftFieldMap::default();
        let eval_field_map = qwen_sft_eval_field_map(&field_map);

        let train_set =
            qwen_sft_examples_from_jsonl_paths_with_limit(&train_paths, None, &field_map)
                .expect("train examples should load");
        let limited_eval_set =
            qwen_sft_examples_from_jsonl_paths_with_limit(&eval_paths, Some(2), &eval_field_map)
                .expect("limited eval examples should load");
        let streaming_eval_summary =
            qwen_sft_streaming_source_summary(&eval_paths, Some(2), &eval_field_map)
                .expect("limited eval streaming summary should scan");

        assert_eq!(train_set.examples.len(), 2);
        assert_eq!(limited_eval_set.examples.len(), 2);
        assert_eq!(
            limited_eval_set
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["eval one", "eval two"]
        );
        assert_eq!(
            streaming_eval_summary.samples,
            limited_eval_set.examples.len()
        );
        assert_eq!(
            streaming_eval_summary.source_sample_counts,
            limited_eval_set.source_sample_counts
        );
        assert_eq!(
            streaming_eval_summary.fingerprint,
            limited_eval_set.fingerprint
        );

        let combined_source_files =
            qwen_merge_sft_source_files(&train_set.source_files, &limited_eval_set.source_files);
        let combined_source_sample_counts = qwen_merge_sft_source_sample_counts(
            &train_set.source_sample_counts,
            &limited_eval_set.source_sample_counts,
        );
        assert_eq!(
            combined_source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: eval_jsonl.display().to_string(),
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: train_jsonl.display().to_string(),
                    samples: 2,
                },
            ]
        );
        assert_ne!(
            qwen_combine_sft_fingerprints(
                &combined_source_files,
                &train_set.fingerprint,
                &limited_eval_set.fingerprint,
            ),
            qwen_combine_sft_fingerprints(
                &combined_source_files,
                &train_set.fingerprint,
                "unlimited-eval-fingerprint",
            )
        );
    }

    #[test]
    fn qwen_sft_resume_dataset_validation_rejects_changed_data() {
        let summary = QwenSftDatasetSummary {
            samples: 2,
            total_tokens: 8,
            response_tokens: 2,
            masked_positions: 2,
            max_sequence_tokens: 4,
            source_files: vec!["data/train.jsonl".to_string()],
            source_sample_counts: vec![QwenSftSourceSampleCount {
                path: "data/train.jsonl".to_string(),
                samples: 2,
            }],
            fingerprint: "fingerprint-a".to_string(),
            shuffle: true,
        };

        qwen_validate_sft_resume_dataset(
            &summary.source_files,
            &summary.source_sample_counts,
            &summary.fingerprint,
            true,
            &summary,
            "test resume",
        )
        .expect("matching provenance should pass");
        qwen_validate_sft_resume_dataset(&[], &[], "", true, &summary, "legacy resume")
            .expect("legacy manifests without provenance should pass");

        let fingerprint_error = qwen_validate_sft_resume_dataset(
            &summary.source_files,
            &summary.source_sample_counts,
            "fingerprint-b",
            true,
            &summary,
            "test resume",
        )
        .expect_err("changed content fingerprint should fail")
        .to_string();
        assert!(fingerprint_error.contains("dataset fingerprint mismatch"));

        let source_error = qwen_validate_sft_resume_dataset(
            &["data/other.jsonl".to_string()],
            &summary.source_sample_counts,
            &summary.fingerprint,
            true,
            &summary,
            "test resume",
        )
        .expect_err("changed source files should fail")
        .to_string();
        assert!(source_error.contains("dataset source files mismatch"));

        let count_error = qwen_validate_sft_resume_dataset(
            &summary.source_files,
            &[QwenSftSourceSampleCount {
                path: "data/train.jsonl".to_string(),
                samples: 3,
            }],
            &summary.fingerprint,
            true,
            &summary,
            "test resume",
        )
        .expect_err("changed source sample counts should fail")
        .to_string();
        assert!(count_error.contains("dataset source sample counts mismatch"));

        let shuffle_error = qwen_validate_sft_resume_dataset(
            &summary.source_files,
            &summary.source_sample_counts,
            &summary.fingerprint,
            false,
            &summary,
            "test resume",
        )
        .expect_err("changed shuffle policy should fail")
        .to_string();
        assert!(shuffle_error.contains("dataset shuffle mismatch"));
    }

    #[test]
    fn qwen_sft_jsonl_reader_aggregates_multiple_paths_and_directories() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let dir = temp.path().join("shard_dir");
        fs::create_dir(&dir).expect("shard dir should be created");
        let second = dir.join("b.jsonl");
        let third = dir.join("a.jsonl");
        let ignored = dir.join("ignored.txt");
        fs::write(
            &first,
            r#"{"instruction":"first","response":"one"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"instruction":"third","response":"three"}
"#,
        )
        .expect("second jsonl should write");
        fs::write(
            &third,
            r#"{"instruction":"second","response":"two"}
"#,
        )
        .expect("third jsonl should write");
        fs::write(
            &ignored,
            r#"{"instruction":"ignored","response":"ignored"}
"#,
        )
        .expect("ignored file should write");

        let example_set = qwen_sft_examples_from_jsonl_paths_with_limit(
            &[first.clone(), dir.clone()],
            None,
            &QwenSftFieldMap::default(),
        )
        .expect("examples should aggregate from multiple paths");
        let examples = &example_set.examples;

        assert_eq!(examples.len(), 3);
        assert_eq!(example_set.source_files.len(), 3);
        assert_eq!(
            example_set.source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: first.display().to_string(),
                    samples: 1,
                },
                QwenSftSourceSampleCount {
                    path: third.display().to_string(),
                    samples: 1,
                },
                QwenSftSourceSampleCount {
                    path: second.display().to_string(),
                    samples: 1,
                },
            ]
        );
        assert!(
            example_set
                .source_files
                .iter()
                .all(|path| path.ends_with(".jsonl"))
        );
        assert!(!example_set.fingerprint.is_empty());
        assert_eq!(examples[0].instruction, "first");
        assert_eq!(examples[1].instruction, "second");
        assert_eq!(examples[2].instruction, "third");
    }

    #[test]
    fn qwen_sft_jsonl_limit_stops_before_unneeded_files() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        let third = temp.path().join("third.jsonl");
        fs::write(
            &first,
            r#"{"instruction":"one","response":"a"}
{"instruction":"two","response":"b"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"instruction":"three","response":"c"}
{"instruction":"four","response":"d"}
"#,
        )
        .expect("second jsonl should write");
        fs::write(
            &third,
            r#"{"instruction":"five","response":"e"}
"#,
        )
        .expect("third jsonl should write");

        let limited = qwen_sft_examples_from_jsonl_paths_with_limit(
            &[first.clone(), second.clone(), third],
            Some(3),
            &QwenSftFieldMap::default(),
        )
        .expect("limited examples should load");

        assert_eq!(limited.examples.len(), 3);
        assert_eq!(
            limited.source_files,
            vec![first.display().to_string(), second.display().to_string()]
        );
        assert_eq!(
            limited.source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: first.display().to_string(),
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: second.display().to_string(),
                    samples: 1,
                },
            ]
        );
        assert_eq!(limited.examples[0].instruction, "one");
        assert_eq!(limited.examples[2].instruction, "three");
        assert_eq!(
            limited.fingerprint,
            qwen_sft_dataset_fingerprint(
                &limited.source_files,
                &limited.examples,
                &QwenSftFieldMap::default()
            )
        );
    }

    #[test]
    fn qwen_sft_streaming_summary_matches_jsonl_reader_without_materializing_tokens() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        let third = temp.path().join("third.jsonl");
        fs::write(
            &first,
            r#"{"instruction":"one","response":"a"}
{"instruction":"two","input":"input","response":"b"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"instruction":"three","response":"c"}
{"instruction":"four","response":"d"}
"#,
        )
        .expect("second jsonl should write");
        fs::write(
            &third,
            r#"{"instruction":"five","response":"e"}
"#,
        )
        .expect("third jsonl should write");

        let paths = vec![first.clone(), second.clone(), third];
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(
            &paths,
            Some(3),
            &QwenSftFieldMap::default(),
        )
        .expect("limited examples should load");
        let streamed =
            qwen_sft_streaming_source_summary(&paths, Some(3), &QwenSftFieldMap::default())
                .expect("streaming summary should scan");

        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
    }

    #[test]
    fn qwen_sft_filters_short_responses_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        fs::write(
            &jsonl,
            r#"{"instruction":"empty","response":""}
{"instruction":"short","response":"ok"}
{"instruction":"first","response":"valid"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            min_response_chars: 5,
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_responses_by_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","response":"approved answer"}
{"instruction":"skip","response":"ordinary reply"}
{"instruction":"second","response":"contains verified marker"}
{"instruction":"third","response":"approved final"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            response_contains_any: vec!["approved".to_string(), "verified".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("response substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap::default()
            )
            .expect("default fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_instructions_by_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"keep first task","response":"answer one"}
{"instruction":"skip ordinary","response":"answer skip"}
{"instruction":"second task","response":"answer two"}
{"instruction":"keep third","response":"answer three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_contains_any: vec!["task".to_string(), "keep".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("instruction substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["keep first task", "second task", "keep third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap::default()
            )
            .expect("default fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["keep first task", "second task", "keep third"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_instructions_by_exclude_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first clean task","response":"answer one"}
{"instruction":"skip banned task","response":"answer skip"}
{"instruction":"second safe task","response":"answer two"}
{"instruction":"third clean task","response":"answer three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            instruction_excludes_any: vec!["banned".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("instruction exclude substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first clean task", "second safe task", "third clean task"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap::default()
            )
            .expect("default fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first clean task", "second safe task", "third clean task"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_responses_by_exclude_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","response":"clean answer"}
{"instruction":"skip","response":"contains banned marker"}
{"instruction":"second","response":"safe reply"}
{"instruction":"third","response":"clean final"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            response_excludes_any: vec!["banned".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("response exclude substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap::default()
            )
            .expect("default fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_long_responses_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","response":"valid"}
{"instruction":"too long","response":"toolong"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            max_response_chars: Some(5),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("max response drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
    }

    #[test]
    fn qwen_sft_filters_instruction_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"a","response":"skip"}
{"instruction":"first","response":"valid"}
{"instruction":"too long","response":"skip"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            min_instruction_chars: Some(3),
            max_instruction_chars: Some(6),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("instruction filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_input_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"skip short","input":"x","response":"short"}
{"instruction":"first","input":"ok","response":"valid"}
{"instruction":"skip long","input":"toolong","response":"long"}
{"instruction":"second","input":"mid","response":"works"}
{"instruction":"third","input":"fit","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            min_input_chars: Some(2),
            max_input_chars: Some(3),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("input filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.input.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "ok"), ("second", "mid")]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.input.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "ok"), ("second", "mid")]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_inputs_by_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","input":"keep context","response":"answer one"}
{"instruction":"skip","input":"ordinary context","response":"answer skip"}
{"instruction":"second","input":"selected context","response":"answer two"}
{"instruction":"third","input":"keep extra","response":"answer three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            input_contains_any: vec!["keep".to_string(), "selected".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("input substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap::default()
            )
            .expect("default fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.input.as_str())
                .collect::<Vec<_>>(),
            vec!["keep context", "selected context", "keep extra"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_inputs_by_exclude_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","input":"clean context","response":"answer one"}
{"instruction":"skip","input":"contains banned context","response":"answer skip"}
{"instruction":"second","input":"safe context","response":"answer two"}
{"instruction":"third","input":"clean extra","response":"answer three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            input_excludes_any: vec!["banned".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("input exclude substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap::default()
            )
            .expect("default fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.input.as_str())
                .collect::<Vec<_>>(),
            vec!["clean context", "safe context", "clean extra"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_system_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"system":"ai","instruction":"skip short","response":"short"}
{"system":"brief","instruction":"first","response":"valid"}
{"system":"this system prompt is too long","instruction":"skip long","response":"long"}
{"system":"concise","instruction":"second","response":"works"}
{"system":"direct","instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            system: Some("system".to_string()),
            min_system_chars: Some(4),
            max_system_chars: Some(8),
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap {
                system: Some("system".to_string()),
                ..QwenSftFieldMap::default()
            },
        )
        .expect_err("system filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (example.system.as_str(), example.instruction.as_str()))
                .collect::<Vec<_>>(),
            vec![("brief", "first"), ("concise", "second")]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| (example.system.as_str(), example.instruction.as_str()))
                .collect::<Vec<_>>(),
            vec![("brief", "first"), ("concise", "second")]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_systems_by_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"system":"keep system","instruction":"first","response":"answer one"}
{"system":"ordinary system","instruction":"skip","response":"answer skip"}
{"system":"selected system","instruction":"second","response":"answer two"}
{"system":"keep extra","instruction":"third","response":"answer three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            system: Some("system".to_string()),
            system_contains_any: vec!["keep".to_string(), "selected".to_string()],
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap {
                system: Some("system".to_string()),
                ..QwenSftFieldMap::default()
            },
        )
        .expect_err("system substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap {
                    system: Some("system".to_string()),
                    ..QwenSftFieldMap::default()
                },
            )
            .expect("default system fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.system.as_str())
                .collect::<Vec<_>>(),
            vec!["keep system", "selected system", "keep extra"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_systems_by_exclude_substring_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"system":"keep system","instruction":"first","response":"answer one"}
{"system":"banned system","instruction":"skip","response":"answer skip"}
{"system":"selected system","instruction":"second","response":"answer two"}
{"system":"keep extra","instruction":"third","response":"answer three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            system: Some("system".to_string()),
            system_excludes_any: vec!["banned".to_string()],
            prompt_template: "System: {system}\nQ: {instruction}\nA: ".to_string(),
            prompt_with_input_template: "System: {system}\nQ: {instruction}\nI: {input}\nA: "
                .to_string(),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap {
                system: Some("system".to_string()),
                ..QwenSftFieldMap::default()
            },
        )
        .expect_err("system exclude substring drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(
                &paths,
                Some(3),
                &loaded.source_files,
                &QwenSftFieldMap {
                    system: Some("system".to_string()),
                    ..QwenSftFieldMap::default()
                },
            )
            .expect("default system fingerprint should compute")
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.system.as_str())
                .collect::<Vec<_>>(),
            vec!["keep system", "selected system", "keep extra"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_prompt_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"a","response":"skip"}
{"instruction":"first","input":"ok","response":"valid"}
{"instruction":"this prompt is too long","response":"skip"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            prompt_template: "Q:{instruction}\nA:".to_string(),
            prompt_with_input_template: "Q:{instruction}\nI:{input}\nA:".to_string(),
            min_prompt_chars: Some(11),
            max_prompt_chars: Some(15),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("prompt filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.input.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "ok"), ("second", "")]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.input.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "ok"), ("second", "")]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_filters_sample_lengths_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"a","response":"x"}
{"instruction":"first","input":"ok","response":"valid"}
{"instruction":"too long","response":"this response is too long"}
{"instruction":"second","response":"works"}
{"instruction":"tiny","response":"z"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            prompt_template: "Q:{instruction}\nA:".to_string(),
            prompt_with_input_template: "Q:{instruction}\nI:{input}\nA:".to_string(),
            min_sample_chars: Some(16),
            max_sample_chars: Some(22),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("filtered examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("filtered streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("filtered source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("filtered raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("filtered cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("sample filter drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.response.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "valid"), ("second", "works")]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| (example.instruction.as_str(), example.response.as_str()))
                .collect::<Vec<_>>(),
            vec![("first", "valid"), ("second", "works")]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn qwen_sft_dedupes_samples_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("samples.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"first","response":"valid"}
{"instruction":"first","response":"valid"}
{"instruction":"second","response":"works"}
{"instruction":"second","response":"works"}
{"instruction":"third","response":"later"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let field_map = QwenSftFieldMap {
            dedupe_samples: true,
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("deduped examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("deduped streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("deduped source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("deduped raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("deduped cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("dedupe drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
    }

    #[test]
    fn qwen_sft_applies_source_weights_before_limit_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &first,
            r#"{"instruction":"first","response":"alpha"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"instruction":"second","response":"beta"}
"#,
        )
        .expect("second jsonl should write");
        let paths = vec![first.clone(), second.clone()];
        let field_map = QwenSftFieldMap {
            source_weights: vec![2, 1],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(3), &field_map)
            .expect("weighted examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(3), &field_map)
            .expect("weighted streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(3), &field_map)
            .expect("weighted source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("weighted raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &field_map,
        )
        .expect("weighted cache should write");
        let unweighted_map = QwenSftFieldMap::default();
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &unweighted_map,
        )
        .expect_err("source weight drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "first", "second"]
        );
        assert_eq!(
            loaded.source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: first.display().to_string(),
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: second.display().to_string(),
                    samples: 1,
                },
            ]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "first", "second"]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 0, 0]
        );
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_ne!(
            loaded.fingerprint,
            qwen_sft_streaming_fingerprint(&paths, Some(3), &loaded.source_files, &unweighted_map)
                .expect("unweighted fingerprint should compute")
        );
    }

    #[test]
    fn qwen_sft_applies_per_source_jsonl_field_maps_to_streaming_cache() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &first,
            r#"{"instruction":"alpha prompt","input":"alpha context","response":"alpha answer"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"question":"beta prompt","answer":"beta answer"}
"#,
        )
        .expect("second jsonl should write");
        let paths = vec![first.clone(), second.clone()];
        let field_map = QwenSftFieldMap {
            source_instruction_fields: vec!["instruction".to_string(), "question".to_string()],
            source_input_fields: vec!["input".to_string(), String::new()],
            source_response_fields: vec!["response".to_string(), "answer".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &field_map)
            .expect("per-source JSONL examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &field_map)
            .expect("per-source streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &field_map)
            .expect("per-source streaming index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("per-source raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("per-source cache should write");
        let cached = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &field_map,
        )
        .expect("per-source cache should hit");
        let changed_map = QwenSftFieldMap {
            source_instruction_fields: vec!["instruction".to_string(), "prompt".to_string()],
            source_input_fields: vec!["input".to_string(), String::new()],
            source_response_fields: vec!["response".to_string(), "answer".to_string()],
            ..QwenSftFieldMap::default()
        };
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &changed_map,
        )
        .expect_err("changed per-source field map should reject cache")
        .to_string();

        assert_eq!(
            loaded.examples,
            vec![
                QwenSftExample {
                    system: String::new(),
                    instruction: "alpha prompt".to_string(),
                    input: "alpha context".to_string(),
                    response: "alpha answer".to_string(),
                },
                QwenSftExample {
                    system: String::new(),
                    instruction: "beta prompt".to_string(),
                    input: String::new(),
                    response: "beta answer".to_string(),
                },
            ]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(raw_window.examples, loaded.examples);
        assert!(cache.cache_written);
        assert!(cached.cache_hit);
        assert_eq!(cached.index.samples, source_index.samples);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_applies_source_max_samples_before_weighting_and_streaming_index() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.jsonl");
        let second = temp.path().join("second.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &first,
            r#"{"instruction":"first-a","response":"alpha"}
{"instruction":"first-b","response":"skip"}
"#,
        )
        .expect("first jsonl should write");
        fs::write(
            &second,
            r#"{"instruction":"second-a","response":"beta"}
{"instruction":"second-b","response":"gamma"}
{"instruction":"second-c","response":"skip"}
"#,
        )
        .expect("second jsonl should write");
        let paths = vec![first.clone(), second.clone()];
        let field_map = QwenSftFieldMap {
            source_weights: vec![2, 2],
            source_max_samples: vec![1, 2],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(6), &field_map)
            .expect("source-limited examples should load");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(6), &field_map)
            .expect("source-limited streaming summary should scan");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(6), &field_map)
            .expect("source-limited source index should build");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &field_map)
            .expect("source-limited raw window should read");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(6),
            Some(&cache_path),
            &field_map,
        )
        .expect("source-limited cache should write");
        let cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(6),
            Some(&cache_path),
            &QwenSftFieldMap {
                source_weights: vec![2, 2],
                ..QwenSftFieldMap::default()
            },
        )
        .expect_err("source max-samples drift should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec![
                "first-a", "first-a", "second-a", "second-a", "second-b", "second-b"
            ]
        );
        assert_eq!(
            loaded.source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: first.display().to_string(),
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: second.display().to_string(),
                    samples: 4,
                },
            ]
        );
        assert_eq!(streamed.samples, loaded.examples.len());
        assert_eq!(streamed.source_files, loaded.source_files);
        assert_eq!(streamed.source_sample_counts, loaded.source_sample_counts);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(cache.index.samples, source_index.samples);
        assert!(cache.cache_written);
        assert!(cache_mismatch.contains("field_map"));
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec![
                "first-a", "first-a", "second-a", "second-a", "second-b", "second-b"
            ]
        );
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| (sample.path.clone(), sample.index_in_file))
                .collect::<Vec<_>>(),
            vec![
                (first.display().to_string(), 0),
                (first.display().to_string(), 0),
                (second.display().to_string(), 0),
                (second.display().to_string(), 0),
                (second.display().to_string(), 1),
                (second.display().to_string(), 1),
            ]
        );
    }

    #[test]
    fn qwen_sft_arrow_applies_source_limits_and_weights_before_global_limit() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.arrow");
        let second = temp.path().join("second.arrow");
        write_test_sft_arrow(
            &first,
            &[
                ("first-a", "", "alpha"),
                ("first-b", "", "beta"),
                ("first-c", "", "gamma"),
            ],
        );
        write_test_sft_arrow(
            &second,
            &[
                ("second-a", "", "delta"),
                ("second-b", "", "epsilon"),
                ("second-c", "", "zeta"),
            ],
        );
        let paths = vec![first.clone(), second.clone()];
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            source_weights: vec![2, 1],
            source_max_samples: vec![1, 2],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_arrow_examples_from_paths_with_limit(&paths, Some(4), &field_map)
            .expect("weighted Arrow examples should load");
        let unweighted = qwen_sft_arrow_examples_from_paths_with_limit(
            &paths,
            Some(4),
            &QwenSftFieldMap {
                response: "output".to_string(),
                ..QwenSftFieldMap::default()
            },
        )
        .expect("unweighted Arrow examples should load");

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["first-a", "first-a", "second-a", "second-b"]
        );
        assert_eq!(
            loaded.source_files,
            vec![first.display().to_string(), second.display().to_string()]
        );
        assert_eq!(
            loaded.source_sample_counts,
            vec![
                QwenSftSourceSampleCount {
                    path: first.display().to_string(),
                    samples: 2,
                },
                QwenSftSourceSampleCount {
                    path: second.display().to_string(),
                    samples: 2,
                },
            ]
        );
        assert_ne!(loaded.fingerprint, unweighted.fingerprint);
    }

    #[test]
    fn qwen_sft_arrow_index_cache_writes_reuses_and_rejects_drift() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let arrow = temp.path().join("train.arrow");
        let cache_path = temp.path().join("cache").join("arrow-row-index.json");
        write_test_sft_arrow(
            &arrow,
            &[
                ("first", "", "alpha"),
                ("second", "", "beta"),
                ("third", "", "gamma"),
            ],
        );
        let paths = vec![arrow.clone()];
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            ..QwenSftFieldMap::default()
        };

        let first =
            qwen_sft_arrow_source_index_with_cache(&paths, Some(2), Some(&cache_path), &field_map)
                .expect("first Arrow cache load should build row index");
        assert!(!first.cache_hit);
        assert!(first.cache_written);
        assert_eq!(first.index.samples.len(), 2);
        assert_eq!(
            first
                .index
                .samples
                .iter()
                .map(|sample| sample.row_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );

        let second =
            qwen_sft_arrow_source_index_with_cache(&paths, Some(2), Some(&cache_path), &field_map)
                .expect("second Arrow cache load should hit");
        assert!(second.cache_hit);
        assert!(!second.cache_written);
        assert_eq!(second.index.samples, first.index.samples);

        let max_samples_mismatch =
            qwen_sft_arrow_source_index_with_cache(&paths, Some(3), Some(&cache_path), &field_map)
                .expect_err("changed max_samples should reject Arrow cache")
                .to_string();
        assert!(max_samples_mismatch.contains("max_samples"));

        let field_mismatch = qwen_sft_arrow_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("changed field map should reject Arrow cache")
        .to_string();
        assert!(field_mismatch.contains("field_map"));

        write_test_sft_arrow(
            &arrow,
            &[
                ("first", "", "alpha"),
                ("second", "", "beta"),
                ("third", "", "gamma"),
                ("fourth", "", "delta"),
            ],
        );
        let source_mismatch =
            qwen_sft_arrow_source_index_with_cache(&paths, Some(2), Some(&cache_path), &field_map)
                .expect_err("changed Arrow source metadata should reject cache")
                .to_string();
        assert!(source_mismatch.contains("source_files"));
    }

    #[test]
    fn qwen_sft_arrow_streaming_policy_requires_cache_or_bounds() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let arrow = temp.path().join("train.arrow");
        let cache_path = temp.path().join("cache").join("arrow-row-index.json");
        write_test_sft_arrow(
            &arrow,
            &[
                ("first", "", "alpha"),
                ("second", "", "beta"),
                ("third", "", "gamma"),
            ],
        );
        let mut data_config: RuntimeDataConfig = toml::from_str(
            r#"
kind = "instruction_arrow"
paths = ["placeholder.arrow"]
response_field = "output"
"#,
        )
        .expect("test data config should parse");
        data_config.paths = vec![arrow];

        qwen_sft_arrow_validate_config_scope(&data_config, "test")
            .expect("test Arrow config scope should validate");
        let unbounded = qwen_sft_arrow_require_cache_or_bounds(&data_config, None, "test")
            .expect_err("unbounded Arrow streaming config should require cache or bounds")
            .to_string();
        assert!(unbounded.contains("requires data.max_samples, data.source_max_samples"));

        data_config.max_samples = Some(2);
        qwen_sft_arrow_require_cache_or_bounds(&data_config, None, "test")
            .expect("global max_samples should bound Arrow streaming");
        data_config.max_samples = None;

        data_config.source_max_samples = vec![2];
        qwen_sft_arrow_require_cache_or_bounds(&data_config, None, "test")
            .expect("source_max_samples should bound Arrow streaming");
        data_config.source_max_samples.clear();

        qwen_sft_arrow_require_cache_or_bounds(&data_config, Some(&cache_path), "test")
            .expect("index cache should allow unbounded Arrow streaming");
    }

    #[test]
    fn qwen_sft_arrow_raw_index_read_uses_original_row_indices_after_filters() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let arrow = temp.path().join("train.arrow");
        write_test_sft_arrow(
            &arrow,
            &[
                ("drop", "", "skip"),
                ("keep-one", "", "alpha"),
                ("keep-two", "", "beta"),
            ],
        );
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            response_contains_any: vec!["alpha".to_string(), "beta".to_string()],
            ..QwenSftFieldMap::default()
        };
        let scan = qwen_sft_arrow_source_scan(&[arrow.clone()], Some(2), &field_map)
            .expect("filtered Arrow source should scan");

        assert_eq!(
            scan.index
                .samples
                .iter()
                .map(|sample| sample.row_index)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        let raw_window = qwen_sft_arrow_examples_by_raw_indices(&scan.index.samples, &field_map)
            .expect("filtered original rows should read");
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["keep-one", "keep-two"]
        );
        assert_eq!(raw_window.raw_samples_read, 2);
    }

    #[test]
    fn qwen_sft_arrow_source_scan_streaming_fingerprint_matches_index_readback() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let arrow = temp.path().join("train.arrow");
        write_test_sft_arrow(
            &arrow,
            &[
                ("drop", "", "skip"),
                ("keep-one", "", "alpha"),
                ("keep-two", "", "beta"),
                ("keep-three", "", "gamma"),
            ],
        );
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            response_contains_any: vec!["alpha".to_string(), "beta".to_string()],
            ..QwenSftFieldMap::default()
        };

        let scan = qwen_sft_arrow_source_scan(&[arrow], Some(2), &field_map)
            .expect("Arrow source scan should build row index");
        let readback_fingerprint = qwen_sft_arrow_fingerprint_from_index(
            &scan.index,
            &scan.summary.source_files,
            &field_map,
        )
        .expect("explicit index readback should hash");

        assert_eq!(
            scan.index
                .samples
                .iter()
                .map(|sample| sample.row_index)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(scan.summary.fingerprint, readback_fingerprint);
    }

    #[test]
    fn qwen_sft_arrow_source_scan_streaming_fingerprint_accounts_for_weights() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let arrow = temp.path().join("train.arrow");
        write_test_sft_arrow(&arrow, &[("first", "", "alpha"), ("second", "", "beta")]);
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            source_weights: vec![2],
            ..QwenSftFieldMap::default()
        };

        let scan = qwen_sft_arrow_source_scan(&[arrow], Some(4), &field_map)
            .expect("weighted Arrow source scan should build row index");
        let raw_window = qwen_sft_arrow_examples_by_raw_indices(&scan.index.samples, &field_map)
            .expect("weighted row index should read back");
        let expected_fingerprint = qwen_sft_dataset_fingerprint(
            &scan.summary.source_files,
            &raw_window.examples,
            &field_map,
        );

        assert_eq!(
            scan.index
                .samples
                .iter()
                .map(|sample| sample.row_index)
                .collect::<Vec<_>>(),
            vec![0, 0, 1, 1]
        );
        assert_eq!(scan.summary.source_sample_counts[0].samples, 4);
        assert_eq!(scan.summary.fingerprint, expected_fingerprint);
    }

    #[test]
    fn qwen_sft_arrow_source_scan_streaming_fingerprint_omits_unused_paths() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.arrow");
        let second = temp.path().join("second.arrow");
        write_test_sft_arrow(&first, &[("first", "", "alpha")]);
        write_test_sft_arrow(&second, &[("second", "", "beta")]);
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            ..QwenSftFieldMap::default()
        };

        let scan =
            qwen_sft_arrow_source_scan(&[first.clone(), second.clone()], Some(1), &field_map)
                .expect("limited Arrow source scan should stop after first source");
        let materialized = qwen_sft_arrow_examples_from_ipc(&first, Some(1), &field_map)
            .expect("first Arrow source should materialize");

        assert_eq!(scan.summary.source_files, vec![first.display().to_string()]);
        assert_eq!(scan.summary.fingerprint, materialized.fingerprint);
    }

    #[test]
    fn qwen_sft_arrow_reads_question_answer_without_input_column() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let arrow = temp.path().join("qa.arrow");
        write_test_qa_arrow(
            &arrow,
            &[
                ("What is Rust?", "A systems programming language."),
                ("What is CUDA?", "A GPU programming platform."),
            ],
        );
        let field_map = QwenSftFieldMap {
            input: String::new(),
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_arrow_examples_from_ipc(&arrow, Some(2), &field_map)
            .expect("question/answer Arrow should materialize");
        let scan = qwen_sft_arrow_source_scan(&[arrow.clone()], Some(2), &field_map)
            .expect("question/answer Arrow source should scan");
        let raw_window = qwen_sft_arrow_examples_by_raw_indices(&scan.index.samples, &field_map)
            .expect("question/answer raw rows should read back");

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (
                    example.instruction.as_str(),
                    example.input.as_str(),
                    example.response.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("What is Rust?", "", "A systems programming language."),
                ("What is CUDA?", "", "A GPU programming platform."),
            ]
        );
        assert_eq!(loaded.examples, raw_window.examples);
        assert_eq!(
            scan.index
                .samples
                .iter()
                .map(|sample| sample.row_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(scan.summary.fingerprint, loaded.fingerprint);
    }

    #[test]
    fn qwen_sft_arrow_applies_per_source_field_maps_to_cache_and_readback() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let first = temp.path().join("first.arrow");
        let second = temp.path().join("second.arrow");
        let cache_path = temp.path().join("cache").join("arrow-row-index.json");
        write_test_sft_arrow(&first, &[("alpha prompt", "alpha context", "alpha answer")]);
        write_test_custom_sft_arrow(
            &second,
            ("prompt", "context", "completion"),
            &[("beta prompt", "", "beta answer")],
        );
        let paths = vec![first.clone(), second.clone()];
        let field_map = QwenSftFieldMap {
            response: "output".to_string(),
            source_instruction_fields: vec!["instruction".to_string(), "prompt".to_string()],
            source_input_fields: vec!["input".to_string(), "context".to_string()],
            source_response_fields: vec!["output".to_string(), "completion".to_string()],
            ..QwenSftFieldMap::default()
        };

        let loaded = qwen_sft_arrow_examples_from_paths_with_limit(&paths, Some(2), &field_map)
            .expect("per-source Arrow examples should load");
        let scan = qwen_sft_arrow_source_scan(&paths, Some(2), &field_map)
            .expect("per-source Arrow scan should build");
        let raw_window = qwen_sft_arrow_examples_by_raw_indices(&scan.index.samples, &field_map)
            .expect("per-source Arrow raw rows should read back");
        let cached =
            qwen_sft_arrow_source_index_with_cache(&paths, Some(2), Some(&cache_path), &field_map)
                .expect("per-source Arrow cache should write");
        let cache_hit =
            qwen_sft_arrow_source_index_with_cache(&paths, Some(2), Some(&cache_path), &field_map)
                .expect("per-source Arrow cache should hit");
        let changed_map = QwenSftFieldMap {
            response: "output".to_string(),
            source_instruction_fields: vec!["instruction".to_string(), "question".to_string()],
            source_input_fields: vec!["input".to_string(), "context".to_string()],
            source_response_fields: vec!["output".to_string(), "completion".to_string()],
            ..QwenSftFieldMap::default()
        };
        let cache_mismatch = qwen_sft_arrow_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &changed_map,
        )
        .expect_err("changed per-source Arrow field map should reject cache")
        .to_string();

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| (
                    example.instruction.as_str(),
                    example.input.as_str(),
                    example.response.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("alpha prompt", "alpha context", "alpha answer"),
                ("beta prompt", "", "beta answer"),
            ]
        );
        assert_eq!(raw_window.examples, loaded.examples);
        assert_eq!(scan.summary.fingerprint, loaded.fingerprint);
        assert!(cached.cache_written);
        assert!(cache_hit.cache_hit);
        assert_eq!(cache_hit.index.samples, scan.index.samples);
        assert!(cache_mismatch.contains("field_map"));
    }

    #[test]
    fn qwen_sft_streaming_window_uses_train_split_sample_count() {
        let (train_samples, eval_samples) =
            qwen_sft_train_eval_sample_counts(4, 0.75).expect("split should compute");
        let world_size = 2usize;
        let local_batch_size = 1usize;
        let global_batch_size = local_batch_size * world_size;
        let train_steps = 1usize;
        let data_cursor_start = 2usize;
        let data_cursor_end = data_cursor_start + train_steps * global_batch_size;
        let (epoch_start, offset_start) =
            qwen_data_epoch_and_offset(data_cursor_start, train_samples)
                .expect("start cursor should map");
        let (epoch_next, offset_next) = qwen_data_epoch_and_offset(data_cursor_end, train_samples)
            .expect("next cursor should map");

        assert_eq!(train_samples, 3);
        assert_eq!(eval_samples, 1);
        assert_eq!(data_cursor_end, 4);
        assert_eq!((epoch_start, offset_start), (0, 2));
        assert_eq!((epoch_next, offset_next), (1, 1));
    }

    #[test]
    fn qwen_sft_streaming_cursor_window_covers_next_batch_overlap() {
        let cursors =
            qwen_sft_streaming_cursor_window(2, 3, 2, 3).expect("cursor window should build");
        let compact = cursors
            .iter()
            .map(|entry| (entry.cursor, entry.epoch, entry.sample_offset))
            .collect::<Vec<_>>();

        assert_eq!(compact, vec![(2, 0, 2), (3, 1, 0), (4, 1, 1), (5, 1, 2)]);
    }

    #[test]
    fn qwen_sft_streaming_raw_index_reads_only_cursor_window() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("train.jsonl");
        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
{"instruction":"one","response":"one"}
{"instruction":"two","response":"two"}
{"instruction":"three","response":"three"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let source_index =
            qwen_sft_streaming_source_index(&paths, None, &QwenSftFieldMap::default())
                .expect("source index should build");
        let (train_samples, _) =
            qwen_sft_train_eval_sample_counts(source_index.samples.len(), 0.75)
                .expect("split should compute");
        let mut train_indices = source_index.samples;
        let mut rng = StdRng::seed_from_u64(777);
        train_indices.shuffle(&mut rng);
        train_indices.truncate(train_samples);
        let raw_indices = (0..4)
            .map(|relative| {
                let cursor = 2 + relative;
                let epoch = cursor / train_indices.len();
                let offset = cursor % train_indices.len();
                let index = qwen_epoch_permutation_index(train_indices.len(), 777, epoch, offset);
                train_indices[index].clone()
            })
            .collect::<Vec<_>>();
        let raw_window =
            qwen_sft_examples_by_raw_indices(&raw_indices, &QwenSftFieldMap::default())
                .expect("raw examples should read");

        assert_eq!(raw_window.examples.len(), 4);
        assert_eq!(raw_window.raw_samples_read, 3);
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["three", "three", "two", "one"]
        );
        assert_eq!(
            raw_indices
                .iter()
                .map(|index| index.index_in_file)
                .collect::<Vec<_>>(),
            vec![3, 3, 2, 1]
        );
        assert_eq!(
            raw_indices
                .iter()
                .map(|index| index.byte_offset)
                .collect::<Vec<_>>(),
            vec![119, 119, 80, 41]
        );
        assert_eq!(
            raw_indices
                .iter()
                .map(|index| index.path.clone())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([jsonl.display().to_string()])
        );
    }

    #[test]
    fn qwen_sft_streaming_source_index_parses_records_before_indexing_offsets() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("train.jsonl");
        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
not-json
{"instruction":"two","response":"two"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let error = match qwen_sft_streaming_source_index(&paths, None, &QwenSftFieldMap::default())
        {
            Ok(_) => panic!("malformed JSONL row should fail while building the offset index"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("failed to parse SFT JSONL record"));
    }

    #[test]
    fn qwen_sft_skip_invalid_records_keeps_valid_rows_across_paths() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("train.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
not-json
{"instruction":"missing-response"}
{"instruction":"three","response":"three"}
{"instruction":"four","response":"four"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];
        let strict_error = match qwen_sft_examples_from_jsonl_paths_with_limit(
            &paths,
            None,
            &QwenSftFieldMap::default(),
        ) {
            Ok(_) => panic!("strict materialized read should reject invalid rows"),
            Err(error) => error.to_string(),
        };
        assert!(strict_error.contains("failed to parse SFT JSONL record"));

        let skip_map = QwenSftFieldMap {
            skip_invalid_records: true,
            ..QwenSftFieldMap::default()
        };
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, Some(2), &skip_map)
            .expect("materialized read should skip invalid rows");
        let streamed = qwen_sft_streaming_source_summary(&paths, Some(2), &skip_map)
            .expect("streaming summary should skip invalid rows");
        let source_index = qwen_sft_streaming_source_index(&paths, Some(2), &skip_map)
            .expect("source index should skip invalid rows");
        let raw_window = qwen_sft_examples_by_raw_indices(&source_index.samples, &skip_map)
            .expect("raw indexed window should read valid rows");
        let cache = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &skip_map,
        )
        .expect("skip-invalid cache should write");
        let default_cache_mismatch = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect_err("cache should reject skip-invalid policy drift")
        .to_string();
        let default_fingerprint = qwen_sft_dataset_fingerprint(
            &loaded.source_files,
            &loaded.examples,
            &QwenSftFieldMap::default(),
        );

        assert_eq!(
            loaded
                .examples
                .iter()
                .map(|example| example.instruction.as_str())
                .collect::<Vec<_>>(),
            vec!["zero", "three"]
        );
        assert_eq!(streamed.samples, 2);
        assert_eq!(streamed.fingerprint, loaded.fingerprint);
        assert_eq!(source_index.samples.len(), 2);
        assert_eq!(
            source_index
                .samples
                .iter()
                .map(|sample| sample.index_in_file)
                .collect::<Vec<_>>(),
            vec![0, 3]
        );
        assert_eq!(
            raw_window
                .examples
                .iter()
                .map(|example| example.response.as_str())
                .collect::<Vec<_>>(),
            vec!["zero", "three"]
        );
        assert!(cache.cache_written);
        assert!(default_cache_mismatch.contains("field_map"));
        assert_ne!(loaded.fingerprint, default_fingerprint);
    }

    #[test]
    fn qwen_sft_streaming_source_index_cache_writes_and_reuses_offsets() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("train.jsonl");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
{"instruction":"one","response":"one"}
{"instruction":"two","response":"two"}
"#,
        )
        .expect("jsonl should write");
        let paths = vec![jsonl.clone()];

        let first = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect("first cache load should build index");
        assert!(!first.cache_hit);
        assert!(first.cache_written);
        assert_eq!(first.index.samples.len(), 2);
        assert!(cache_path.exists());

        let second = qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        )
        .expect("second cache load should hit cache");
        assert!(second.cache_hit);
        assert!(!second.cache_written);
        assert_eq!(second.index.samples, first.index.samples);

        let mismatch = match qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(3),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        ) {
            Ok(_) => panic!("mismatched max_samples should reject cache"),
            Err(error) => error.to_string(),
        };
        assert!(mismatch.contains("max_samples"));

        let min_response_mismatch_map = QwenSftFieldMap {
            min_response_chars: 2,
            ..QwenSftFieldMap::default()
        };
        let min_response_mismatch = match qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &min_response_mismatch_map,
        ) {
            Ok(_) => panic!("mismatched min_response_chars should reject cache"),
            Err(error) => error.to_string(),
        };
        assert!(min_response_mismatch.contains("field_map"));

        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
{"instruction":"one","response":"one"}
{"instruction":"two","response":"two"}
{"instruction":"three","response":"three"}
"#,
        )
        .expect("jsonl rewrite should update source metadata");
        let source_metadata_mismatch = match qwen_sft_streaming_source_index_with_cache(
            &paths,
            Some(2),
            Some(&cache_path),
            &QwenSftFieldMap::default(),
        ) {
            Ok(_) => panic!("changed source file metadata should reject cache"),
            Err(error) => error.to_string(),
        };
        assert!(source_metadata_mismatch.contains("source_files"));
    }

    #[test]
    fn qwen_sft_external_metadata_participates_in_fingerprint_and_cache() {
        let temp = tempfile::tempdir().expect("temp dir should be created");
        let jsonl = temp.path().join("train.jsonl");
        let metadata_path = temp.path().join("arrow-export.json");
        let cache_path = temp.path().join("cache").join("offset-index.json");
        fs::write(
            &jsonl,
            r#"{"instruction":"zero","response":"zero"}
{"instruction":"one","response":"one"}
"#,
        )
        .expect("jsonl should write");
        fs::write(
            &metadata_path,
            r#"{"source_arrow":"/datasets/source.arrow","exported_rows":2}"#,
        )
        .expect("metadata should write");
        let paths = vec![jsonl.clone()];
        let metadata = qwen_sft_external_metadata_from_paths(std::slice::from_ref(&metadata_path))
            .expect("metadata should load");
        let field_map = QwenSftFieldMap {
            external_metadata: metadata.clone(),
            ..QwenSftFieldMap::default()
        };
        let no_metadata_map = QwenSftFieldMap::default();
        let loaded = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &field_map)
            .expect("examples should load");
        let without_metadata =
            qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &no_metadata_map)
                .expect("examples should load without metadata");

        assert_ne!(loaded.fingerprint, without_metadata.fingerprint);

        let first =
            qwen_sft_streaming_source_index_with_cache(&paths, None, Some(&cache_path), &field_map)
                .expect("first cache load should build index");
        assert!(first.cache_written);

        let cache_mismatch = match qwen_sft_streaming_source_index_with_cache(
            &paths,
            None,
            Some(&cache_path),
            &no_metadata_map,
        ) {
            Ok(_) => panic!("missing external metadata should reject cache"),
            Err(error) => error.to_string(),
        };
        assert!(cache_mismatch.contains("field_map"));

        fs::write(
            &metadata_path,
            r#"{"source_arrow":"/datasets/changed.arrow","exported_rows":2}"#,
        )
        .expect("metadata rewrite should update provenance");
        let changed_metadata =
            qwen_sft_external_metadata_from_paths(std::slice::from_ref(&metadata_path))
                .expect("changed metadata should load");
        let changed_map = QwenSftFieldMap {
            external_metadata: changed_metadata,
            ..QwenSftFieldMap::default()
        };
        let changed = qwen_sft_examples_from_jsonl_paths_with_limit(&paths, None, &changed_map)
            .expect("examples should load with changed metadata");

        assert_ne!(loaded.fingerprint, changed.fingerprint);
    }

    #[test]
    fn qwen_sft_rank_index_cache_path_keeps_extension() {
        let path = PathBuf::from("/tmp/rustrain/cache/offset-index.json");
        assert_eq!(
            qwen_sft_rank_index_cache_path(&path, 1),
            PathBuf::from("/tmp/rustrain/cache/offset-index.rank-1.json")
        );

        let no_extension = PathBuf::from("/tmp/rustrain/cache/offset-index");
        assert_eq!(
            qwen_sft_rank_index_cache_path(&no_extension, 2),
            PathBuf::from("/tmp/rustrain/cache/offset-index.rank-2")
        );
    }
}
