use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use tracing::info;

use rustrain_core::runtime::{Config, LrScheduler, RunPaths};
use rustrain_data::text_data::{
    SftDataset, SftSample, TokenizedDataset, load_sft_dataset, load_text_dataset,
};
use rustrain_train::metrics::{gpu_memory_allocated_mb, memory_rss_mb};

use crate::lora::LoraLinear;
use crate::toy_model::{AdamW, QwenLikeModel, masked_cross_entropy_loss, masked_logits_gradient};

#[derive(Debug, Serialize, Deserialize)]
struct TrainingCheckpoint {
    step: u64,
    model: QwenLikeModel,
    optimizer: AdamW,
}

pub fn train(config: &Config, run_paths: &RunPaths) -> Result<()> {
    if config.model.architecture == "none" || config.train.max_steps == 0 {
        info!("M0 skeleton complete; no model training is run for this config");
        println!("rustrain M0 complete");
        println!("run_dir: {}", run_paths.root.display());
        println!("resolved_config: {}", run_paths.resolved_config.display());
        println!("log_file: {}", run_paths.logs.join("train.log").display());
        return Ok(());
    }

    if let Some(data) = &config.data {
        return match data.kind {
            rustrain_core::runtime::DataKind::Text => train_text_data(config, run_paths),
            rustrain_core::runtime::DataKind::InstructionJsonl => train_sft_data(config, run_paths),
            rustrain_core::runtime::DataKind::InstructionArrow => Err(anyhow!(
                "trainer data.kind = instruction_arrow is not implemented yet; use qwen-sft-streaming-data-plan or qwen-sft-arrow-batch-plan for direct Arrow validation"
            )),
        };
    }

    train_fixed_batch(config, run_paths)
}

fn train_fixed_batch(config: &Config, run_paths: &rustrain_core::runtime::RunPaths) -> Result<()> {
    let tokens = fixed_overfit_batch(config.model.vocab_size, config.model.seq_len);
    let (mut model, mut optimizer, start_step) = load_or_initialize(config)?;
    let initial = model.loss(&tokens).loss;

    info!(initial_loss = initial, "starting one-batch overfit");

    for step in (start_step + 1)..=config.train.max_steps {
        let metrics = train_step(
            config,
            &mut model,
            &mut optimizer,
            step,
            std::slice::from_ref(&tokens),
        )?;

        let step_loss = model.loss(&tokens).loss;
        if step == 1 || step == config.train.max_steps || step % 10 == 0 {
            info!(
                step,
                loss = step_loss,
                lr = metrics.learning_rate,
                grad_norm = metrics.grad_norm,
                clipped_grad_norm = metrics.clipped_grad_norm,
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
        reload_loss,
        reload_delta,
        memory_rss_mb = memory_rss_mb(),
        gpu_memory_allocated_mb = gpu_memory_allocated_mb(),
        "checkpoint reload parity"
    );
    info!(?generated, "generate smoke test");

    println!("rustrain M1 complete");
    println!("run_dir: {}", run_paths.root.display());
    println!("initial_loss: {initial:.6}");
    println!("final_loss: {final_loss:.6}");
    println!("reload_loss: {reload_loss:.6}");
    if let Some(memory_rss_mb) = memory_rss_mb() {
        println!("memory_rss_mb: {memory_rss_mb:.2}");
    }
    if let Some(gpu_memory_allocated_mb) = gpu_memory_allocated_mb() {
        println!("gpu_memory_allocated_mb: {gpu_memory_allocated_mb:.2}");
    }
    println!("checkpoint: {}", checkpoint_path.display());
    println!("generated_tokens: {generated:?}");

    Ok(())
}

fn train_text_data(config: &Config, run_paths: &rustrain_core::runtime::RunPaths) -> Result<()> {
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
    let total_samples = config.train.max_steps as usize * config.train.gradient_accumulation_steps;
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
        let metrics = train_step(
            config,
            &mut model,
            &mut optimizer,
            step,
            std::slice::from_ref(sequence),
        )?;

        let train_loss = model.loss(sequence).loss;
        if step == 1 || step == config.train.max_steps || step % 10 == 0 {
            info!(
                step,
                train_loss,
                lr = metrics.learning_rate,
                grad_norm = metrics.grad_norm,
                clipped_grad_norm = metrics.clipped_grad_norm,
                "text train step"
            );
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
    let samples_per_second = total_samples as f32 / elapsed;
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
        reload_eval,
        tokens_per_second,
        samples_per_second,
        memory_rss_mb = memory_rss_mb(),
        gpu_memory_allocated_mb = gpu_memory_allocated_mb(),
        "text-data training complete"
    );

    println!("rustrain M2-lite complete");
    println!("run_dir: {}", run_paths.root.display());
    println!("initial_eval_loss: {initial_eval:.6}");
    println!("last_logged_eval_loss: {last_eval:.6}");
    println!("final_eval_loss: {final_eval:.6}");
    println!("reload_eval_loss: {reload_eval:.6}");
    println!("tokens_per_second: {tokens_per_second:.2}");
    println!("samples_per_second: {samples_per_second:.2}");
    if let Some(memory_rss_mb) = memory_rss_mb() {
        println!("memory_rss_mb: {memory_rss_mb:.2}");
    }
    if let Some(gpu_memory_allocated_mb) = gpu_memory_allocated_mb() {
        println!("gpu_memory_allocated_mb: {gpu_memory_allocated_mb:.2}");
    }
    println!("checkpoint: {}", checkpoint_path.display());
    println!(
        "tokenized_cache: {}",
        run_paths.cache.join("tokenized.toml").display()
    );

    Ok(())
}

fn train_sft_data(config: &Config, run_paths: &rustrain_core::runtime::RunPaths) -> Result<()> {
    let data_config = config.data.as_ref().expect("data config should exist");
    let dataset = load_sft_dataset(
        data_config,
        config.model.vocab_size,
        config.model.seq_len,
        &run_paths.cache,
    )?;
    let SftDataset {
        tokenizer,
        train_samples,
        eval_samples,
    } = dataset;
    let base_model = QwenLikeModel::new(config.model.clone(), config.run.seed);
    let mut adapter = initialize_lm_head_lora(&base_model, config.run.seed + 1, 4, 8.0);
    let prompt = train_samples[0].tokens[..(train_samples[0].tokens.len() / 2).max(1)].to_vec();
    let before_generated = base_model.generate_greedy(&prompt, 4);
    let initial_eval = eval_sft_loss(&base_model, &adapter, &eval_samples);
    let mut last_eval = initial_eval;
    let total_samples = config.train.max_steps as usize * config.train.gradient_accumulation_steps;
    let started = std::time::Instant::now();

    info!(
        train_samples = train_samples.len(),
        eval_samples = eval_samples.len(),
        trainable_adapter_params = adapter.adapter_param_count(),
        initial_eval_loss = initial_eval,
        "starting local SFT training"
    );

    for step in 1..=config.train.max_steps {
        for accumulation_index in 0..config.train.gradient_accumulation_steps {
            let sample =
                &train_samples[(step as usize + accumulation_index - 1) % train_samples.len()];
            let learning_rate = learning_rate_for_step(config, step);
            sft_step(&base_model, &mut adapter, sample, learning_rate);
        }

        let train_loss = sft_loss(
            &base_model,
            &adapter,
            &train_samples[(step as usize - 1) % train_samples.len()],
        );
        if step == 1 || step == config.train.max_steps || step % 10 == 0 {
            info!(
                step,
                train_loss,
                lr = learning_rate_for_step(config, step),
                "SFT train step"
            );
        }

        if config.train.eval_every > 0 && step % config.train.eval_every == 0 {
            last_eval = eval_sft_loss(&base_model, &adapter, &eval_samples);
            info!(step, eval_loss = last_eval, "SFT eval step");
        }
    }

    let final_eval = eval_sft_loss(&base_model, &adapter, &eval_samples);
    let elapsed = started.elapsed().as_secs_f32().max(1e-6);
    let samples_per_second = total_samples as f32 / elapsed;
    let adapter_path = run_paths.checkpoints.join("adapter-final.toml");
    adapter.save_adapter(&adapter_path)?;

    let mut reloaded_adapter = initialize_lm_head_lora(&base_model, config.run.seed + 1, 4, 8.0);
    reloaded_adapter.load_adapter(&adapter_path)?;
    let reload_eval = eval_sft_loss(&base_model, &reloaded_adapter, &eval_samples);
    if (final_eval - reload_eval).abs() > 1e-5 {
        return Err(anyhow!(
            "SFT adapter reload parity failed: final_eval={final_eval}, reload_eval={reload_eval}"
        ));
    }

    let after_generated = generate_with_lora_lm_head(&base_model, &reloaded_adapter, &prompt, 4);
    let before_path = run_paths.root.join("generate_before.txt");
    let after_path = run_paths.root.join("generate_after.txt");
    std::fs::write(&before_path, tokenizer.decode_lossy(&before_generated))
        .with_context(|| format!("failed to write {}", before_path.display()))?;
    std::fs::write(&after_path, tokenizer.decode_lossy(&after_generated))
        .with_context(|| format!("failed to write {}", after_path.display()))?;

    info!(
        final_eval,
        reload_eval,
        adapter_checkpoint = %adapter_path.display(),
        generate_before = %before_path.display(),
        generate_after = %after_path.display(),
        samples_per_second,
        memory_rss_mb = memory_rss_mb(),
        gpu_memory_allocated_mb = gpu_memory_allocated_mb(),
        "SFT training complete"
    );

    println!("rustrain M7-lite complete");
    println!("run_dir: {}", run_paths.root.display());
    println!("initial_eval_loss: {initial_eval:.6}");
    println!("last_logged_eval_loss: {last_eval:.6}");
    println!("final_eval_loss: {final_eval:.6}");
    println!("reload_eval_loss: {reload_eval:.6}");
    println!("samples_per_second: {samples_per_second:.2}");
    if let Some(memory_rss_mb) = memory_rss_mb() {
        println!("memory_rss_mb: {memory_rss_mb:.2}");
    }
    if let Some(gpu_memory_allocated_mb) = gpu_memory_allocated_mb() {
        println!("gpu_memory_allocated_mb: {gpu_memory_allocated_mb:.2}");
    }
    println!("adapter_checkpoint: {}", adapter_path.display());
    println!("generate_before: {}", before_path.display());
    println!("generate_after: {}", after_path.display());

    Ok(())
}

fn initialize_lm_head_lora(
    model: &QwenLikeModel,
    seed: u64,
    rank: usize,
    alpha: f32,
) -> LoraLinear {
    let (in_features, out_features) = model.lm_head_dim();
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 0.02).expect("normal init should be valid");
    let lora_a = Array2::from_shape_fn((in_features, rank), |_| normal.sample(&mut rng));
    let lora_b = Array2::zeros((rank, out_features));
    LoraLinear::with_adapter(model.lm_head_weight(), lora_a, lora_b, alpha)
}

fn sft_step(model: &QwenLikeModel, adapter: &mut LoraLinear, sample: &SftSample, lr: f32) {
    let inputs = &sample.tokens[..sample.tokens.len() - 1];
    let targets = sample.tokens[1..].to_vec();
    let activations = model.forward(inputs);
    let logits = adapter.forward(&activations.hidden);
    let grad_logits = masked_logits_gradient(&logits, &targets, &sample.target_mask);
    adapter.step_adapter(&activations.hidden, &grad_logits, lr);
}

fn sft_loss(model: &QwenLikeModel, adapter: &LoraLinear, sample: &SftSample) -> f32 {
    let inputs = &sample.tokens[..sample.tokens.len() - 1];
    let targets = sample.tokens[1..].to_vec();
    let activations = model.forward(inputs);
    let logits = adapter.forward(&activations.hidden);
    masked_cross_entropy_loss(&logits, &targets, &sample.target_mask)
}

fn eval_sft_loss(model: &QwenLikeModel, adapter: &LoraLinear, samples: &[SftSample]) -> f32 {
    samples
        .iter()
        .map(|sample| sft_loss(model, adapter, sample))
        .sum::<f32>()
        / samples.len() as f32
}

fn generate_with_lora_lm_head(
    model: &QwenLikeModel,
    adapter: &LoraLinear,
    prompt: &[usize],
    max_new_tokens: usize,
) -> Vec<usize> {
    let mut tokens = prompt.to_vec();

    for _ in 0..max_new_tokens {
        let activations = model.forward(&tokens);
        let logits = adapter.forward(&activations.hidden);
        let last = logits.row(logits.nrows() - 1);
        let next = last
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .expect("logits should be non-empty");
        tokens.push(next);
    }

    tokens
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
    step: u64,
    sequences: &[Vec<usize>],
) -> Result<TrainStepMetrics> {
    let mut grad_accum = Array2::zeros(model.lm_head_dim());
    for accumulation_index in 0..config.train.gradient_accumulation_steps {
        let sequence = &sequences[accumulation_index % sequences.len()];
        let output = model.loss(sequence);
        grad_accum += &model.lm_head_gradient(&output);
    }

    grad_accum /= config.train.gradient_accumulation_steps as f32;
    let grad_norm = l2_norm(&grad_accum);
    let clipped_grad_norm = clip_gradient(&mut grad_accum, config.train.max_grad_norm);
    let learning_rate = learning_rate_for_step(config, step);
    optimizer.step_lm_head_with_lr(model, &grad_accum, learning_rate);

    Ok(TrainStepMetrics {
        learning_rate,
        grad_norm,
        clipped_grad_norm,
    })
}

#[derive(Debug, Clone, Copy)]
struct TrainStepMetrics {
    learning_rate: f32,
    grad_norm: f32,
    clipped_grad_norm: f32,
}

fn learning_rate_for_step(config: &Config, step: u64) -> f32 {
    match config.train.lr_scheduler {
        LrScheduler::Constant => config.train.learning_rate,
        LrScheduler::LinearDecay => {
            let max_steps = config.train.max_steps.max(1) as f32;
            let progress = (step.saturating_sub(1) as f32 / max_steps).clamp(0.0, 1.0);
            config.train.learning_rate * (1.0 - progress)
        }
    }
}

fn clip_gradient(grad: &mut Array2<f32>, max_grad_norm: Option<f32>) -> f32 {
    let grad_norm = l2_norm(grad);
    if let Some(max_grad_norm) = max_grad_norm {
        if grad_norm > max_grad_norm {
            *grad *= max_grad_norm / (grad_norm + 1e-12);
            return max_grad_norm;
        }
    }
    grad_norm
}

fn l2_norm(values: &Array2<f32>) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn maybe_save_checkpoint(
    config: &Config,
    run_paths: &rustrain_core::runtime::RunPaths,
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

fn save_checkpoint(path: &Path, step: u64, model: &QwenLikeModel, optimizer: &AdamW) -> Result<()> {
    let checkpoint = TrainingCheckpoint {
        step,
        model: model.clone(),
        optimizer: optimizer.clone(),
    };
    let contents =
        toml::to_string(&checkpoint).context("failed to serialize training checkpoint")?;
    std::fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))
}

fn load_checkpoint(path: &Path) -> Result<TrainingCheckpoint> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read checkpoint {}", path.display()))?;
    toml::from_str(&contents)
        .with_context(|| format!("failed to parse checkpoint {}", path.display()))
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use super::*;
    use rustrain_core::{
        backend::BackendKind,
        runtime::{
            DType, Device, LrScheduler, ModelConfig, ParallelConfig, RunConfig, TrainConfig,
        },
    };

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            name: "test_qwen_like".to_string(),
            architecture: "qwen_like".to_string(),
            model_path: None,
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
            trainable_layers: None,
        }
    }

    #[test]
    fn training_checkpoint_preserves_model_and_optimizer_state() {
        let mut model = QwenLikeModel::new(tiny_config(), 23);
        let mut optimizer = AdamW::new(model.lm_head_dim(), 0.05, 0.9, 0.999, 1e-8, 0.01);
        let tokens = vec![3, 10, 1, 8, 15, 6, 13, 4];
        let output = model.loss(&tokens);
        let grad = model.lm_head_gradient(&output);
        optimizer.step_lm_head_with_lr(&mut model, &grad, 0.05);

        let before = model.loss(&tokens).loss;
        let file = NamedTempFile::new().expect("temp checkpoint should be created");
        save_checkpoint(file.path(), 1, &model, &optimizer).expect("checkpoint should save");

        let reloaded = load_checkpoint(file.path()).expect("checkpoint should load");
        let after = reloaded.model.loss(&tokens).loss;

        assert_eq!(reloaded.step, 1);
        assert!((before - after).abs() < 1e-6);
    }

    #[test]
    fn train_step_applies_scheduler_and_grad_clipping() {
        let config = tiny_train_config(LrScheduler::LinearDecay, Some(0.001), 4);
        let mut model = QwenLikeModel::new(config.model.clone(), 23);
        let mut optimizer = AdamW::new(model.lm_head_dim(), 0.05, 0.9, 0.999, 1e-8, 0.01);
        let tokens = vec![3, 10, 1, 8, 15, 6, 13, 4];

        let metrics = train_step(&config, &mut model, &mut optimizer, 3, &[tokens])
            .expect("train step should run");

        assert_eq!(metrics.learning_rate, 0.025);
        assert!(metrics.grad_norm > metrics.clipped_grad_norm);
        assert!((metrics.clipped_grad_norm - 0.001).abs() < 1e-8);
    }

    fn tiny_train_config(
        lr_scheduler: LrScheduler,
        max_grad_norm: Option<f32>,
        max_steps: u64,
    ) -> Config {
        Config {
            run: RunConfig {
                name: "test".to_string(),
                base_dir: "runs".into(),
                seed: 0,
            },
            model: tiny_config(),
            train: TrainConfig {
                max_steps,
                resume_from: None,
                backend: BackendKind::NdArray,
                micro_batch_size: 1,
                global_batch_size: 1,
                gradient_accumulation_steps: 1,
                learning_rate: 0.05,
                weight_decay: 0.01,
                adam_beta1: 0.9,
                adam_beta2: 0.999,
                adam_eps: 1e-8,
                lr_scheduler,
                max_grad_norm,
                dtype: DType::Fp32,
                device: Device::Cpu,
                checkpoint_every: 0,
                eval_every: 0,
            },
            data: None,
            lora: None,
            parallel: ParallelConfig {
                tensor_model_parallel_size: 1,
                pipeline_model_parallel_size: 1,
                data_parallel_size: 1,
                expert_model_parallel_size: 1,
                context_parallel_size: 1,
            },
        }
    }
}
