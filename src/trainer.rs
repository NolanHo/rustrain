use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
    runtime::{
        init_logging, load_config, prepare_run_directory, validate_config, write_resolved_config,
    },
    toy_model::{AdamW, QwenLikeModel},
};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TrainingCheckpoint {
    step: u64,
    model: QwenLikeModel,
    optimizer: AdamW,
}

pub fn train(config_path: &Path) -> Result<()> {
    let config = load_config(config_path)?;
    validate_config(&config)?;

    let run_paths = prepare_run_directory(&config.run)?;
    let _log_guard = init_logging(&run_paths.logs)?;

    write_resolved_config(&config, &run_paths.resolved_config)?;

    info!(config_path = %config_path.display(), "loaded config");
    info!(run_dir = %run_paths.root.display(), "created run directory");
    info!(checkpoints_dir = %run_paths.checkpoints.display(), "created checkpoint directory");
    info!(seed = config.run.seed, "seed configured");
    info!(device = ?config.train.device, dtype = ?config.train.dtype, "training policy configured");
    info!(model = ?config.model, "model config");
    info!(train = ?config.train, "train config");
    info!(parallel = ?config.parallel, "parallel config");

    if config.model.architecture == "none" || config.train.max_steps == 0 {
        info!("M0 skeleton complete; no model training is run for this config");

        println!("rustrain M0 complete");
        println!("run_dir: {}", run_paths.root.display());
        println!("resolved_config: {}", run_paths.resolved_config.display());
        println!("log_file: {}", run_paths.logs.join("train.log").display());

        return Ok(());
    }

    let tokens = fixed_overfit_batch(config.model.vocab_size, config.model.seq_len);
    let mut model = QwenLikeModel::new(config.model.clone(), config.run.seed);
    let mut optimizer = AdamW::new(
        model.lm_head_dim(),
        config.train.learning_rate,
        config.train.adam_beta1,
        config.train.adam_beta2,
        config.train.adam_eps,
        config.train.weight_decay,
    );
    let initial = model.loss(&tokens).loss;

    info!(initial_loss = initial, "starting one-batch overfit");

    for step in 1..=config.train.max_steps {
        let mut grad_accum = Array2::zeros(model.lm_head_dim());
        let mut step_loss = 0.0;

        for _ in 0..config.train.gradient_accumulation_steps {
            let output = model.loss(&tokens);
            step_loss += output.loss;
            grad_accum += &model.lm_head_gradient(&output);
        }

        step_loss /= config.train.gradient_accumulation_steps as f32;
        grad_accum /= config.train.gradient_accumulation_steps as f32;
        optimizer.step_lm_head(&mut model, &grad_accum);

        if step == 1 || step == config.train.max_steps || step % 10 == 0 {
            info!(
                step,
                loss = step_loss,
                grad_accumulation_steps = config.train.gradient_accumulation_steps,
                "train step"
            );
        }

        if config.train.checkpoint_every > 0 && step % config.train.checkpoint_every == 0 {
            let checkpoint_path = run_paths
                .checkpoints
                .join(format!("model-step-{step}.toml"));
            save_checkpoint(&checkpoint_path, step, &model, &optimizer)?;
            info!(step, checkpoint = %checkpoint_path.display(), "saved checkpoint");
        }
    }

    let final_loss = model.loss(&tokens).loss;
    if final_loss >= initial {
        return Err(anyhow!(
            "overfit one batch failed: initial_loss={initial}, final_loss={final_loss}"
        ));
    }

    let checkpoint_path = run_paths.checkpoints.join("model-final.toml");
    save_checkpoint(&checkpoint_path, config.train.max_steps, &model, &optimizer)?;

    let reloaded = load_checkpoint(&checkpoint_path)?;
    let reload_loss = reloaded.model.loss(&tokens).loss;
    let reload_delta = (final_loss - reload_loss).abs();
    if reload_delta > 1e-5 {
        return Err(anyhow!(
            "checkpoint reload parity failed: final_loss={final_loss}, reload_loss={reload_loss}"
        ));
    }

    let prompt_len = tokens.len().min(4);
    let generated = reloaded.model.generate_greedy(&tokens[..prompt_len], 4);

    info!(
        final_loss,
        reload_loss, reload_delta, "checkpoint reload parity"
    );
    info!(?generated, "generate smoke test");

    println!("rustrain M1 complete");
    println!("run_dir: {}", run_paths.root.display());
    println!("initial_loss: {initial:.6}");
    println!("final_loss: {final_loss:.6}");
    println!("reload_loss: {reload_loss:.6}");
    println!("checkpoint: {}", checkpoint_path.display());
    println!("generated_tokens: {generated:?}");

    Ok(())
}

fn fixed_overfit_batch(vocab_size: usize, seq_len: usize) -> Vec<usize> {
    (0..seq_len)
        .map(|index| ((index * 7) + 3) % vocab_size)
        .collect()
}

pub(crate) fn save_checkpoint(
    path: &Path,
    step: u64,
    model: &QwenLikeModel,
    optimizer: &AdamW,
) -> Result<()> {
    let checkpoint = TrainingCheckpoint {
        step,
        model: model.clone(),
        optimizer: optimizer.clone(),
    };
    let contents =
        toml::to_string(&checkpoint).context("failed to serialize training checkpoint")?;
    std::fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))
}

pub(crate) fn load_checkpoint(path: &Path) -> Result<TrainingCheckpoint> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read checkpoint {}", path.display()))?;
    toml::from_str(&contents)
        .with_context(|| format!("failed to parse checkpoint {}", path.display()))
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use super::*;
    use crate::runtime::ModelConfig;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            name: "test_qwen_like".to_string(),
            architecture: "qwen_like".to_string(),
            vocab_size: 16,
            hidden_size: 16,
            num_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 32,
            seq_len: 8,
            norm: "rmsnorm".to_string(),
            activation: "swiglu".to_string(),
            rope: true,
            rms_norm_eps: 1e-6,
        }
    }

    #[test]
    fn training_checkpoint_preserves_model_and_optimizer_state() {
        let mut model = QwenLikeModel::new(tiny_config(), 23);
        let mut optimizer = AdamW::new(model.lm_head_dim(), 0.05, 0.9, 0.999, 1e-8, 0.01);
        let tokens = vec![3, 10, 1, 8, 15, 6, 13, 4];
        let output = model.loss(&tokens);
        let grad = model.lm_head_gradient(&output);
        optimizer.step_lm_head(&mut model, &grad);

        let before = model.loss(&tokens).loss;
        let file = NamedTempFile::new().expect("temp checkpoint should be created");
        save_checkpoint(file.path(), 1, &model, &optimizer).expect("checkpoint should save");

        let reloaded = load_checkpoint(file.path()).expect("checkpoint should load");
        let after = reloaded.model.loss(&tokens).loss;

        assert_eq!(reloaded.step, 1);
        assert!((before - after).abs() < 1e-6);
    }
}
