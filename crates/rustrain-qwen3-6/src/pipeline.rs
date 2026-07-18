use std::collections::HashSet;
use std::ops::Range;

use anyhow::{Result, bail};

use crate::config::{LayerType, Qwen36RuntimeConfig};
use crate::lora::{Qwen36LoraTargetModule, Qwen36NativeLoraSlot};

#[derive(Debug, Clone)]
pub struct PipelineLoraSlot {
    pub global_index: usize,
    pub local_index: usize,
    pub layer: usize,
    pub module: Qwen36LoraTargetModule,
    pub active: bool,
}

/// Contiguous pipeline ownership for one physical PP rank.
///
/// Layer IDs remain global at every external boundary. `layer_range` is the
/// only place where they become a stage-local contiguous slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStageLayout {
    pub pipeline_rank: usize,
    pub pipeline_size: usize,
    pub global_num_layers: usize,
    pub layer_range: Range<usize>,
}

impl PipelineStageLayout {
    pub fn new(
        global_num_layers: usize,
        pipeline_rank: usize,
        pipeline_size: usize,
    ) -> Result<Self> {
        if pipeline_size == 0 {
            bail!("pipeline_size must be positive");
        }
        if pipeline_rank >= pipeline_size {
            bail!("pipeline rank {pipeline_rank} is outside pipeline_size={pipeline_size}");
        }
        if global_num_layers < pipeline_size {
            bail!(
                "global_num_layers={global_num_layers} must be at least pipeline_size={pipeline_size}"
            );
        }
        let start = global_num_layers * pipeline_rank / pipeline_size;
        let end = global_num_layers * (pipeline_rank + 1) / pipeline_size;
        Ok(Self {
            pipeline_rank,
            pipeline_size,
            global_num_layers,
            layer_range: start..end,
        })
    }

    pub fn full(global_num_layers: usize) -> Result<Self> {
        Self::new(global_num_layers, 0, 1)
    }

    pub fn is_first(&self) -> bool {
        self.pipeline_rank == 0
    }

    pub fn is_last(&self) -> bool {
        self.pipeline_rank + 1 == self.pipeline_size
    }

    pub fn local_num_layers(&self) -> usize {
        self.layer_range.len()
    }

    pub fn owns_layer(&self, global_layer: usize) -> bool {
        self.layer_range.contains(&global_layer)
    }

    pub fn local_target_layers(&self, global_targets: &[usize]) -> Vec<usize> {
        global_targets
            .iter()
            .copied()
            .filter(|layer| self.owns_layer(*layer))
            .collect()
    }

    pub(crate) fn native_flags(&self) -> i32 {
        i32::from(self.is_first()) | (i32::from(self.is_last()) << 1)
    }
}

pub fn stage_lora_slots(
    global_slots: &[Qwen36NativeLoraSlot],
    stage: &PipelineStageLayout,
) -> Vec<PipelineLoraSlot> {
    global_slots
        .iter()
        .filter(|slot| stage.owns_layer(slot.layer))
        .enumerate()
        .map(|(local_index, slot)| PipelineLoraSlot {
            global_index: slot.index,
            local_index,
            layer: slot.layer,
            module: slot.module,
            active: slot.active,
        })
        .collect()
}

/// Frozen base weights owned by one pipeline stage.
///
/// Tied vocabulary weights are intentionally present on both boundary stages:
/// the first consumes them as embeddings and the last consumes them as the LM
/// head. All other layer weights have exactly one owner.
pub fn stage_text_needed_weights(
    config: &Qwen36RuntimeConfig,
    stage: &PipelineStageLayout,
) -> HashSet<String> {
    let prefix = &config.weight_prefix;
    let mut needed = HashSet::new();

    if stage.is_first() || (stage.is_last() && config.tie_word_embeddings) {
        needed.insert(format!("{prefix}embed_tokens.weight"));
    }
    if stage.is_last() {
        needed.insert(format!("{prefix}norm.weight"));
        if !config.tie_word_embeddings {
            needed.insert("lm_head.weight".to_string());
        }
    }

    for layer in stage.layer_range.clone() {
        let layer_prefix = format!("{prefix}layers.{layer}");
        needed.insert(format!("{layer_prefix}.input_layernorm.weight"));
        needed.insert(format!("{layer_prefix}.post_attention_layernorm.weight"));

        match config.layer_types[layer] {
            LayerType::FullAttention => {
                for weight in ["q_proj", "q_norm", "k_proj", "k_norm", "v_proj", "o_proj"] {
                    needed.insert(format!("{layer_prefix}.self_attn.{weight}.weight"));
                }
            }
            LayerType::LinearAttention => {
                needed.insert(format!("{layer_prefix}.linear_attn.A_log"));
                needed.insert(format!("{layer_prefix}.linear_attn.conv1d.weight"));
                needed.insert(format!("{layer_prefix}.linear_attn.dt_bias"));
                needed.insert(format!("{layer_prefix}.linear_attn.norm.weight"));
                for weight in [
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_a",
                    "in_proj_b",
                    "out_proj",
                ] {
                    needed.insert(format!("{layer_prefix}.linear_attn.{weight}.weight"));
                }
            }
        }

        if config.is_moe {
            needed.insert(format!("{layer_prefix}.mlp.gate.weight"));
            needed.insert(format!("{layer_prefix}.mlp.shared_expert_gate.weight"));
            needed.insert(format!("{layer_prefix}.mlp.shared_expert.gate_proj.weight"));
            needed.insert(format!("{layer_prefix}.mlp.shared_expert.up_proj.weight"));
            needed.insert(format!("{layer_prefix}.mlp.shared_expert.down_proj.weight"));
            needed.insert(format!("{layer_prefix}.mlp.experts.gate_up_proj"));
            needed.insert(format!("{layer_prefix}.mlp.experts.down_proj"));
        } else {
            needed.insert(format!("{layer_prefix}.mlp.gate_proj.weight"));
            needed.insert(format!("{layer_prefix}.mlp.up_proj.weight"));
            needed.insert(format!("{layer_prefix}.mlp.down_proj.weight"));
        }
    }

    needed
}

pub fn stage_needed_weights(
    config: &Qwen36RuntimeConfig,
    stage: &PipelineStageLayout,
) -> HashSet<String> {
    let mut needed = stage_text_needed_weights(config, stage);
    if config.has_vision && stage.is_first() {
        needed.extend(crate::vision::VisionWeights::weight_names(config));
    }

    needed
}

#[cfg(test)]
mod tests {
    use super::{PipelineStageLayout, stage_lora_slots, stage_needed_weights};
    use crate::config::{LayerType, Qwen36RuntimeConfig};
    use crate::lora::{Qwen36LoraTargetModule, Qwen36NativeLoraSlot};

    fn runtime() -> Qwen36RuntimeConfig {
        Qwen36RuntimeConfig {
            num_hidden_layers: 2,
            hidden_size: 16,
            vocab_size: 32,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            hidden_act: "silu".into(),
            layer_types: vec![LayerType::FullAttention, LayerType::LinearAttention],
            full_attention_interval: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 8,
            attention_bias: false,
            attn_output_gate: false,
            rope_theta: 1e6,
            partial_rotary_factor: 1.0,
            mrope_interleaved: false,
            mrope_section: vec![],
            linear_num_key_heads: 2,
            linear_key_head_dim: 8,
            linear_num_value_heads: 2,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            mamba_ssm_dtype: "float32".into(),
            is_moe: false,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            router_aux_loss_coef: 0.0,
            intermediate_size: 32,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            has_vision: false,
            vision_depth: 0,
            vision_hidden_size: 0,
            vision_num_heads: 0,
            vision_patch_size: 0,
            vision_spatial_merge_size: 0,
            vision_temporal_patch_size: 0,
            vision_out_hidden_size: 0,
            weight_prefix: "model.".into(),
        }
    }

    #[test]
    fn uneven_layers_are_contiguous_and_cover_the_model() {
        let stages = (0..2)
            .map(|rank| PipelineStageLayout::new(5, rank, 2).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(stages[0].layer_range, 0..2);
        assert_eq!(stages[1].layer_range, 2..5);
        assert!(stages[0].is_first());
        assert!(stages[1].is_last());
        assert_eq!(stages[0].layer_range.end, stages[1].layer_range.start);
        assert_eq!(stages[1].layer_range.end, 5);
    }

    #[test]
    fn target_layers_keep_global_identity() {
        let first = PipelineStageLayout::new(4, 0, 2).unwrap();
        let last = PipelineStageLayout::new(4, 1, 2).unwrap();
        assert_eq!(first.local_target_layers(&[0, 3]), vec![0]);
        assert_eq!(last.local_target_layers(&[0, 3]), vec![3]);
    }

    #[test]
    fn rejects_empty_stages_and_invalid_rank() {
        assert!(PipelineStageLayout::new(1, 0, 2).is_err());
        assert!(PipelineStageLayout::new(4, 2, 2).is_err());
        assert!(PipelineStageLayout::new(4, 0, 0).is_err());
    }

    #[test]
    fn stage_slots_preserve_global_identity_and_compact_native_indices() {
        let global_slots = vec![
            Qwen36NativeLoraSlot {
                index: 0,
                layer: 0,
                module: Qwen36LoraTargetModule::QProj,
                active: true,
            },
            Qwen36NativeLoraSlot {
                index: 1,
                layer: 0,
                module: Qwen36LoraTargetModule::OProj,
                active: false,
            },
            Qwen36NativeLoraSlot {
                index: 2,
                layer: 1,
                module: Qwen36LoraTargetModule::InProjQkv,
                active: true,
            },
        ];
        let stage = PipelineStageLayout::new(2, 1, 2).unwrap();
        let local = stage_lora_slots(&global_slots, &stage);
        assert_eq!(local.len(), 1);
        assert_eq!(local[0].global_index, 2);
        assert_eq!(local[0].local_index, 0);
        assert_eq!(local[0].layer, 1);
    }

    #[test]
    fn boundary_weights_are_owned_without_loading_remote_layers() {
        let config = runtime();
        let first = stage_needed_weights(
            &config,
            &PipelineStageLayout::new(config.num_hidden_layers, 0, 2).unwrap(),
        );
        let last = stage_needed_weights(
            &config,
            &PipelineStageLayout::new(config.num_hidden_layers, 1, 2).unwrap(),
        );

        assert!(first.contains("model.embed_tokens.weight"));
        assert!(!first.contains("model.norm.weight"));
        assert!(first.iter().any(|name| name.starts_with("model.layers.0.")));
        assert!(!first.iter().any(|name| name.starts_with("model.layers.1.")));

        assert!(last.contains("model.embed_tokens.weight"));
        assert!(last.contains("model.norm.weight"));
        assert!(last.iter().any(|name| name.starts_with("model.layers.1.")));
        assert!(!last.iter().any(|name| name.starts_with("model.layers.0.")));
    }
}
