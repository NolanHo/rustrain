use anyhow::{Result, anyhow};
use tch::{
    Cuda, Device, Kind, Reduction, Tensor, nn,
    nn::{Module, OptimizerConfig},
};
use tracing::info;

use crate::runtime::{Config, Device as RuntimeDevice};

#[derive(Debug, Clone)]
pub struct TchTrainSmokeSummary {
    pub initial_loss: f64,
    pub final_loss: f64,
    pub embedding_grad_defined: bool,
    pub lm_head_grad_defined: bool,
}

pub fn train_tch_tiny_lm(config: &Config) -> Result<TchTrainSmokeSummary> {
    if config.train.max_steps == 0 {
        return Err(anyhow!("tch_tiny_lm requires train.max_steps > 0"));
    }

    let device = tch_device(config)?;
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let embedding = nn::embedding(
        &root / "embed_tokens",
        config.model.vocab_size as i64,
        config.model.hidden_size as i64,
        Default::default(),
    );
    let lm_head = nn::linear(
        &root / "lm_head",
        config.model.hidden_size as i64,
        config.model.vocab_size as i64,
        nn::LinearConfig {
            bias: false,
            ..Default::default()
        },
    );
    let mut optimizer = nn::AdamW {
        beta1: config.train.adam_beta1 as f64,
        beta2: config.train.adam_beta2 as f64,
        wd: config.train.weight_decay as f64,
        eps: config.train.adam_eps as f64,
        amsgrad: false,
    }
    .build(&vs, config.train.learning_rate as f64)?;

    let input_ids = fixed_tch_batch(config.model.vocab_size as i64, config.model.seq_len as i64)
        .to_device(device);
    let targets = input_ids.narrow(1, 1, config.model.seq_len as i64 - 1);
    let initial_loss = tch_lm_loss(&embedding, &lm_head, &input_ids, &targets).double_value(&[]);
    let mut embedding_grad_defined = false;
    let mut lm_head_grad_defined = false;

    for step in 1..=config.train.max_steps {
        optimizer.zero_grad();
        let loss = tch_lm_loss(&embedding, &lm_head, &input_ids, &targets);
        loss.backward();
        if step == 1 {
            embedding_grad_defined = embedding.ws.grad().defined();
            lm_head_grad_defined = lm_head.ws.grad().defined();
        }
        optimizer.step();

        if step == 1 || step == config.train.max_steps || step % 10 == 0 {
            info!(
                step,
                loss = loss.double_value(&[]),
                "tch tiny lm train step"
            );
        }
    }

    let final_loss = tch_lm_loss(&embedding, &lm_head, &input_ids, &targets).double_value(&[]);
    if final_loss >= initial_loss {
        return Err(anyhow!(
            "tch tiny lm failed to reduce loss: initial_loss={initial_loss}, final_loss={final_loss}"
        ));
    }
    if !embedding_grad_defined || !lm_head_grad_defined {
        return Err(anyhow!(
            "tch tiny lm missing gradients: embedding_grad_defined={embedding_grad_defined}, lm_head_grad_defined={lm_head_grad_defined}"
        ));
    }

    Ok(TchTrainSmokeSummary {
        initial_loss,
        final_loss,
        embedding_grad_defined,
        lm_head_grad_defined,
    })
}

fn fixed_tch_batch(vocab_size: i64, seq_len: i64) -> Tensor {
    let tokens: Vec<i64> = (0..seq_len).map(|position| position % vocab_size).collect();
    Tensor::from_slice(&tokens).reshape([1, seq_len])
}

fn tch_device(config: &Config) -> Result<Device> {
    match config.train.device {
        RuntimeDevice::Cpu => Ok(Device::Cpu),
        RuntimeDevice::Cuda => {
            if Cuda::is_available() {
                Ok(Device::Cuda(0))
            } else {
                Err(anyhow!(
                    "config requested device=cuda, but tch CUDA is not available"
                ))
            }
        }
    }
}

fn tch_lm_loss(
    embedding: &nn::Embedding,
    lm_head: &nn::Linear,
    input_ids: &Tensor,
    targets: &Tensor,
) -> Tensor {
    let hidden = embedding.forward(input_ids);
    let hidden = hidden.narrow(1, 0, input_ids.size()[1] - 1);
    let logits = lm_head.forward(&hidden);
    let vocab_size = logits.size()[2];
    logits
        .reshape([-1, vocab_size])
        .log_softmax(-1, Kind::Float)
        .g_nll_loss::<&Tensor>(&targets.reshape([-1]), None, Reduction::Mean, -100)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendKind;
    use crate::runtime::{
        Config, DType, Device as RuntimeDevice, ModelConfig, ParallelConfig, RunConfig, TrainConfig,
    };

    #[test]
    fn tch_tiny_lm_trains_all_parameter_groups() {
        let config = tiny_tch_config(RuntimeDevice::Cpu);

        let summary = train_tch_tiny_lm(&config).expect("tch tiny lm should train");

        assert!(summary.final_loss < summary.initial_loss);
        assert!(summary.embedding_grad_defined);
        assert!(summary.lm_head_grad_defined);
    }

    #[test]
    fn tch_cuda_request_fails_clearly_when_unavailable() {
        if tch::Cuda::is_available() {
            return;
        }
        let config = tiny_tch_config(RuntimeDevice::Cuda);

        let error = train_tch_tiny_lm(&config).expect_err("missing CUDA should fail");

        assert!(
            error.to_string().contains("tch CUDA is not available"),
            "unexpected error: {error:#}"
        );
    }

    fn tiny_tch_config(device: RuntimeDevice) -> Config {
        Config {
            run: RunConfig {
                name: "test".to_string(),
                base_dir: "runs".into(),
                seed: 0,
            },
            model: ModelConfig {
                name: "tch_tiny_lm".to_string(),
                architecture: "tch_tiny_lm".to_string(),
                vocab_size: 16,
                hidden_size: 8,
                num_layers: 1,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                intermediate_size: 16,
                seq_len: 8,
                norm: "rmsnorm".to_string(),
                activation: "swiglu".to_string(),
                rope: true,
                rms_norm_eps: 1e-6,
            },
            train: TrainConfig {
                max_steps: 3,
                resume_from: None,
                backend: BackendKind::Tch,
                micro_batch_size: 1,
                global_batch_size: 1,
                gradient_accumulation_steps: 1,
                learning_rate: 1e-2,
                weight_decay: 0.0,
                adam_beta1: 0.9,
                adam_beta2: 0.999,
                adam_eps: 1e-8,
                dtype: DType::Fp32,
                device,
                checkpoint_every: 0,
                eval_every: 0,
            },
            data: None,
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
