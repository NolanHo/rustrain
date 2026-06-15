use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
    runtime::{
        Config, init_logging, load_config, prepare_run_directory, validate_config,
        write_resolved_config,
    },
    text_data::{TokenizedDataset, load_text_dataset},
    toy_model::{AdamW, QwenLikeModel},
};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TrainingCheckpoint {
    step: u64,
    model: QwenLikeModel,
    optimizer: AdamW,
}

pub fn train(config_path: &Path, resume_from: Option<PathBuf>) -> Result<()> {
    let mut config = load_config(config_path)?;
    if let Some(resume_from) = resume_from {
        config.train.resume_from = Some(resume_from);
    }
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

    if config.data.is_some() {
        return train_text_data(&config, &run_paths);
    }

    train_fixed_batch(&config, &run_paths)
}

fn train_fixed_batch(config: &Config, run_paths: &crate::runtime::RunPaths) -> Result<()> {
    let tokens = fixed_overfit_batch(config.model.vocab_size, config.model.seq_len);
    let (mut model, mut optimizer, start_step) = load_or_initialize(config)?;
    let initial = model.loss(&tokens).loss;

    info!(initial_loss = initial, "starting one-batch overfit");

    for step in (start_step + 1)..=config.train.max_steps {
        train_step(
            config,
            &mut model,
            &mut optimizer,
            std::slice::from_ref(&tokens),
        )?;

        let step_loss = model.loss(&tokens).loss;
        if step == 1 || step == config.train.max_steps || step % 10 == 0 {
            info!(
                step,
                loss = step_loss,
                grad_accumulation_steps = config.train.gradient_accumulation_steps,
                "train step"
            );
        }

        maybe_save_checkpoint(config, run_paths, step, &model, &optimizer)?;
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

fn train_text_data(config: &Config, run_paths: &crate::runtime::RunPaths) -> Result<()> {
    let data_config = config.data.as_ref().expect("data config should exist");
    let dataset = load_text_dataset(
        data_config,
        config.model.vocab_size,
        config.model.seq_len,
        &run_paths.cache,
    )?;
    let TokenizedDataset {
        train_sequences,
        eval_sequences,
        ..
    } = dataset;
    let (mut model, mut optimizer, start_step) = load_or_initialize(config)?;
    let initial_eval = eval_loss(&model, &eval_sequences);
    let mut last_eval = initial_eval;
    let total_tokens = config.train.max_steps as usize
        * config.train.gradient_accumulation_steps
        * config.model.seq_len;
    let started = std::time::Instant::now();

    info!(
        train_sequences = train_sequences.len(),
        eval_sequences = eval_sequences.len(),
        initial_eval_loss = initial_eval,
        start_step,
        "starting text-data training"
    );

    for step in (start_step + 1)..=config.train.max_steps {
        let sequence = &train_sequences[(step as usize - 1) % train_sequences.len()];
        train_step(
            config,
            &mut model,
            &mut optimizer,
            std::slice::from_ref(sequence),
        )?;

        let train_loss = model.loss(sequence).loss;
        if step == 1 || step == config.train.max_steps || step % 10 == 0 {
            info!(step, train_loss, "text train step");
        }

        if config.train.eval_every > 0 && step % config.train.eval_every == 0 {
            last_eval = eval_loss(&model, &eval_sequences);
            info!(step, eval_loss = last_eval, "eval step");
        }

        maybe_save_checkpoint(config, run_paths, step, &model, &optimizer)?;
    }

    let final_eval = eval_loss(&model, &eval_sequences);
    let elapsed = started.elapsed().as_secs_f32().max(1e-6);
    let tokens_per_second = total_tokens as f32 / elapsed;
    let checkpoint_path = run_paths.checkpoints.join("model-final.toml");
    save_checkpoint(&checkpoint_path, config.train.max_steps, &model, &optimizer)?;

    let reloaded = load_checkpoint(&checkpoint_path)?;
    let reload_eval = eval_loss(&reloaded.model, &eval_sequences);
    if (final_eval - reload_eval).abs() > 1e-5 {
        return Err(anyhow!(
            "text checkpoint reload parity failed: final_eval={final_eval}, reload_eval={reload_eval}"
        ));
    }

    info!(
        final_eval,
        reload_eval, tokens_per_second, "text-data training complete"
    );

    println!("rustrain M2-lite complete");
    println!("run_dir: {}", run_paths.root.display());
    println!("initial_eval_loss: {initial_eval:.6}");
    println!("last_logged_eval_loss: {last_eval:.6}");
    println!("final_eval_loss: {final_eval:.6}");
    println!("reload_eval_loss: {reload_eval:.6}");
    println!("tokens_per_second: {tokens_per_second:.2}");
    println!("checkpoint: {}", checkpoint_path.display());
    println!(
        "tokenized_cache: {}",
        run_paths.cache.join("tokenized.toml").display()
    );

    Ok(())
}

fn load_or_initialize(config: &Config) -> Result<(QwenLikeModel, AdamW, u64)> {
    if let Some(path) = &config.train.resume_from {
        let checkpoint = load_checkpoint(path)?;
        return Ok((checkpoint.model, checkpoint.optimizer, checkpoint.step));
    }

    let model = QwenLikeModel::new(config.model.clone(), config.run.seed);
    let optimizer = AdamW::new(
        model.lm_head_dim(),
        config.train.learning_rate,
        config.train.adam_beta1,
        config.train.adam_beta2,
        config.train.adam_eps,
        config.train.weight_decay,
    );
    Ok((model, optimizer, 0))
}

fn train_step(
    config: &Config,
    model: &mut QwenLikeModel,
    optimizer: &mut AdamW,
    sequences: &[Vec<usize>],
) -> Result<()> {
    let mut grad_accum = Array2::zeros(model.lm_head_dim());
    for accumulation_index in 0..config.train.gradient_accumulation_steps {
        let sequence = &sequences[accumulation_index % sequences.len()];
        let output = model.loss(sequence);
        grad_accum += &model.lm_head_gradient(&output);
    }

    grad_accum /= config.train.gradient_accumulation_steps as f32;
    optimizer.step_lm_head(model, &grad_accum);

    Ok(())
}

fn maybe_save_checkpoint(
    config: &Config,
    run_paths: &crate::runtime::RunPaths,
    step: u64,
    model: &QwenLikeModel,
    optimizer: &AdamW,
) -> Result<()> {
    if config.train.checkpoint_every > 0 && step % config.train.checkpoint_every == 0 {
        let checkpoint_path = run_paths
            .checkpoints
            .join(format!("model-step-{step}.toml"));
        save_checkpoint(&checkpoint_path, step, model, optimizer)?;
        info!(step, checkpoint = %checkpoint_path.display(), "saved checkpoint");
    }
    Ok(())
}

fn eval_loss(model: &QwenLikeModel, sequences: &[Vec<usize>]) -> f32 {
    let total = sequences
        .iter()
        .map(|sequence| model.loss(sequence).loss)
        .sum::<f32>();
    total / sequences.len() as f32
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
