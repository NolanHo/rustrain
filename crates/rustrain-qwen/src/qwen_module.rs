//! qwen_module - re-export hub for split modules

pub use crate::generate::*;
pub use crate::lora::*;
pub use crate::model::*;
pub use crate::rank::*;
pub use crate::session::*;
pub use crate::sft::*;

#[cfg(test)]
pub(crate) mod test_utils {
    use super::*;

    pub(crate) fn write_test_sft_arrow(path: &Path, rows: &[(&str, &str, &str)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("instruction", DataType::Utf8, false),
            Field::new("input", DataType::Utf8, false),
            Field::new("output", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(
                    rows.iter()
                        .map(|(instruction, _, _)| *instruction)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|(_, input, _)| *input).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter()
                        .map(|(_, _, response)| *response)
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .expect("test Arrow batch should build");
        let file = fs::File::create(path).expect("test Arrow file should create");
        let mut writer = StreamWriter::try_new(file, &schema).expect("test Arrow writer");
        writer.write(&batch).expect("test Arrow batch should write");
        writer.finish().expect("test Arrow stream should finish");
    }

    pub(crate) fn write_test_custom_sft_arrow(
        path: &Path,
        fields: (&str, &str, &str),
        rows: &[(&str, &str, &str)],
    ) {
        let schema = Arc::new(Schema::new(vec![
            Field::new(fields.0, DataType::Utf8, false),
            Field::new(fields.1, DataType::Utf8, false),
            Field::new(fields.2, DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(
                    rows.iter()
                        .map(|(instruction, _, _)| *instruction)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|(_, input, _)| *input).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter()
                        .map(|(_, _, response)| *response)
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .expect("test custom Arrow batch should build");
        let file = fs::File::create(path).expect("test custom Arrow file should create");
        let mut writer = StreamWriter::try_new(file, &schema).expect("test custom Arrow writer");
        writer
            .write(&batch)
            .expect("test custom Arrow batch should write");
        writer
            .finish()
            .expect("test custom Arrow stream should finish");
    }

    pub(crate) fn write_test_qa_arrow(path: &Path, rows: &[(&str, &str)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("question", DataType::Utf8, false),
            Field::new("answer", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(
                    rows.iter()
                        .map(|(question, _)| *question)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|(_, answer)| *answer).collect::<Vec<_>>(),
                )),
            ],
        )
        .expect("test QA Arrow batch should build");
        let file = fs::File::create(path).expect("test QA Arrow file should create");
        let mut writer = StreamWriter::try_new(file, &schema).expect("test QA Arrow writer");
        writer
            .write(&batch)
            .expect("test QA Arrow batch should write");
        writer.finish().expect("test QA Arrow stream should finish");
    }

    pub(crate) fn tiny_qwen_sharded_manifest() -> QwenShardedCheckpointManifest {
        QwenShardedCheckpointManifest {
            format: "rustrain.qwen_sharded.v1".to_string(),
            base_model_path: "/models/qwen".to_string(),
            tokenizer_path: "/models/qwen/tokenizer.json".to_string(),
            global_step: 3,
            consumed_samples: 8,
            consumed_tokens: 40,
            data_cursor_next: Some(8),
            data_epoch_next: Some(1),
            data_sample_offset_next: Some(3),
            data_train_samples: Some(5),
            dataset_source_files: vec!["data/train.jsonl".to_string()],
            dataset_source_sample_counts: vec![QwenSftSourceSampleCount {
                path: "data/train.jsonl".to_string(),
                samples: 5,
            }],
            dataset_fingerprint: "abc123".to_string(),
            dataset_shuffle: true,
            streaming_train_batches: Some(true),
            seed: 42,
            dtype: "float32".to_string(),
            optimizer: "adamw".to_string(),
            scheduler: "linear_decay".to_string(),
            parallel: QwenShardedParallelManifest {
                data_parallel_size: 2,
                tensor_model_parallel_size: 1,
                pipeline_model_parallel_size: 1,
                expert_model_parallel_size: 1,
                context_parallel_size: 1,
            },
            ranks: vec![
                QwenRankShardManifest {
                    rank: 0,
                    data_parallel_rank: 0,
                    tensor_model_parallel_rank: 0,
                    pipeline_model_parallel_rank: 0,
                    expert_model_parallel_rank: 0,
                    context_parallel_rank: 0,
                    model_safetensors: "rank0/model.safetensors".to_string(),
                    optimizer_safetensors: "rank0/optimizer.safetensors".to_string(),
                    shards: vec![QwenTensorShardManifestEntry {
                        name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                        shard_name: "rank0.q_proj".to_string(),
                        optimizer_m_name: "rank0.q_proj.m".to_string(),
                        optimizer_v_name: "rank0.q_proj.v".to_string(),
                        global_shape: vec![4, 4],
                        shard_shape: vec![4, 4],
                        dtype: "float32".to_string(),
                        partition: "replicated_dp".to_string(),
                        tied_group: None,
                    }],
                },
                QwenRankShardManifest {
                    rank: 1,
                    data_parallel_rank: 1,
                    tensor_model_parallel_rank: 0,
                    pipeline_model_parallel_rank: 0,
                    expert_model_parallel_rank: 0,
                    context_parallel_rank: 0,
                    model_safetensors: "rank1/model.safetensors".to_string(),
                    optimizer_safetensors: "rank1/optimizer.safetensors".to_string(),
                    shards: vec![QwenTensorShardManifestEntry {
                        name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                        shard_name: "rank1.q_proj".to_string(),
                        optimizer_m_name: "rank1.q_proj.m".to_string(),
                        optimizer_v_name: "rank1.q_proj.v".to_string(),
                        global_shape: vec![4, 4],
                        shard_shape: vec![4, 4],
                        dtype: "float32".to_string(),
                        partition: "replicated_dp".to_string(),
                        tied_group: None,
                    }],
                },
            ],
        }
    }

    pub(crate) fn tiny_qwen_sharded_manifest_with_artifacts(
        root: &Path,
    ) -> Result<QwenShardedCheckpointManifest> {
        let mut manifest = tiny_qwen_sharded_manifest();
        for rank in &mut manifest.ranks {
            let rank_dir = root.join(format!("rank{}", rank.rank));
            fs::create_dir_all(&rank_dir)
                .with_context(|| format!("failed to create {}", rank_dir.display()))?;
            let model_safetensors = rank_dir.join("model.safetensors");
            let optimizer_safetensors = rank_dir.join("optimizer.safetensors");
            let model_entries = rank
                .shards
                .iter()
                .map(|shard| {
                    (
                        shard.shard_name.clone(),
                        Tensor::ones(shard.shard_shape.as_slice(), (Kind::Float, Device::Cpu)),
                    )
                })
                .collect::<Vec<_>>();
            let optimizer_entries = rank
                .shards
                .iter()
                .flat_map(|shard| {
                    [
                        (
                            shard.optimizer_m_name.clone(),
                            Tensor::zeros(shard.shard_shape.as_slice(), (Kind::Float, Device::Cpu)),
                        ),
                        (
                            shard.optimizer_v_name.clone(),
                            Tensor::zeros(shard.shard_shape.as_slice(), (Kind::Float, Device::Cpu)),
                        ),
                    ]
                })
                .collect::<Vec<_>>();
            let model_refs = model_entries
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor))
                .collect::<Vec<_>>();
            let optimizer_refs = optimizer_entries
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor))
                .collect::<Vec<_>>();
            Tensor::write_safetensors(&model_refs, &model_safetensors)
                .with_context(|| format!("failed to write {}", model_safetensors.display()))?;
            Tensor::write_safetensors(&optimizer_refs, &optimizer_safetensors)
                .with_context(|| format!("failed to write {}", optimizer_safetensors.display()))?;
            rank.model_safetensors = model_safetensors.display().to_string();
            rank.optimizer_safetensors = optimizer_safetensors.display().to_string();
        }
        Ok(manifest)
    }

    pub(crate) fn tiny_qwen_weights() -> BTreeMap<String, Tensor> {
        let mut weights = BTreeMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::arange(24, (Kind::Float, Device::Cpu)).reshape([6, 4]) / 24.0,
        );
        weights.insert(
            "model.norm.weight".to_string(),
            Tensor::ones([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            Tensor::ones([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            Tensor::ones([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            Tensor::eye(4, (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.q_proj.bias".to_string(),
            Tensor::zeros([4], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            Tensor::ones([2, 4], (Kind::Float, Device::Cpu)) * 0.05,
        );
        weights.insert(
            "model.layers.0.self_attn.k_proj.bias".to_string(),
            Tensor::zeros([2], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            Tensor::ones([2, 4], (Kind::Float, Device::Cpu)) * 0.03,
        );
        weights.insert(
            "model.layers.0.self_attn.v_proj.bias".to_string(),
            Tensor::zeros([2], (Kind::Float, Device::Cpu)),
        );
        weights.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            Tensor::ones([4, 4], (Kind::Float, Device::Cpu)) * 0.02,
        );
        weights.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            Tensor::ones([8, 4], (Kind::Float, Device::Cpu)) * 0.01,
        );
        weights.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            Tensor::ones([8, 4], (Kind::Float, Device::Cpu)) * 0.02,
        );
        weights.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            Tensor::ones([4, 8], (Kind::Float, Device::Cpu)) * 0.03,
        );
        weights
    }

    pub(crate) fn two_layer_tiny_qwen_weights() -> BTreeMap<String, Tensor> {
        let mut weights = tiny_qwen_weights();
        let layer0_names = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];
        for suffix in layer0_names {
            let layer0_name = format!("model.layers.0.{suffix}");
            let layer1_name = format!("model.layers.1.{suffix}");
            let value = tensor(&weights, &layer0_name)
                .expect("layer0 tensor should exist")
                .shallow_clone();
            weights.insert(layer1_name, value * 0.9);
        }
        weights
    }
}
