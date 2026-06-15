use ndarray::{Array1, Array2, Array3, Axis, s};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::runtime::ModelConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenLikeModel {
    config: ModelConfig,
    token_embedding: Array2<f32>,
    layers: Vec<TransformerLayer>,
    final_norm_weight: Array1<f32>,
    lm_head: Array2<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransformerLayer {
    input_norm_weight: Array1<f32>,
    q_proj: Array2<f32>,
    k_proj: Array2<f32>,
    v_proj: Array2<f32>,
    o_proj: Array2<f32>,
    post_attention_norm_weight: Array1<f32>,
    gate_proj: Array2<f32>,
    up_proj: Array2<f32>,
    down_proj: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct ForwardActivations {
    pub hidden: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct LossOutput {
    pub loss: f32,
    pub logits: Array2<f32>,
    pub targets: Vec<usize>,
    pub target_mask: Vec<bool>,
    pub activations: ForwardActivations,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamW {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: usize,
    lm_head_m: Array2<f32>,
    lm_head_v: Array2<f32>,
}

impl QwenLikeModel {
    pub fn new(config: ModelConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = 0.02;
        let layers = (0..config.num_layers)
            .map(|_| TransformerLayer::new(&config, &mut rng, scale))
            .collect();

        Self {
            token_embedding: rand_matrix(config.vocab_size, config.hidden_size, &mut rng, scale),
            final_norm_weight: Array1::ones(config.hidden_size),
            lm_head: rand_matrix(config.hidden_size, config.vocab_size, &mut rng, scale),
            config,
            layers,
        }
    }

    pub fn forward(&self, input_ids: &[usize]) -> ForwardActivations {
        let seq_len = input_ids.len();
        let hidden_size = self.config.hidden_size;
        let mut hidden = Array2::zeros((seq_len, hidden_size));

        for (position, token_id) in input_ids.iter().copied().enumerate() {
            hidden
                .row_mut(position)
                .assign(&self.token_embedding.row(token_id));
        }

        for layer in &self.layers {
            hidden = layer.forward(&self.config, &hidden);
        }

        hidden = rms_norm(&hidden, &self.final_norm_weight, self.config.rms_norm_eps);

        ForwardActivations { hidden }
    }

    pub fn logits_from_activations(&self, activations: &ForwardActivations) -> Array2<f32> {
        activations.hidden.dot(&self.lm_head)
    }

    pub fn loss(&self, tokens: &[usize]) -> LossOutput {
        let target_mask = vec![true; tokens.len() - 1];
        self.loss_with_target_mask(tokens, &target_mask)
    }

    pub fn loss_with_target_mask(&self, tokens: &[usize], target_mask: &[bool]) -> LossOutput {
        let inputs = &tokens[..tokens.len() - 1];
        let targets = tokens[1..].to_vec();
        assert_eq!(
            target_mask.len(),
            targets.len(),
            "target mask should align with causal targets"
        );
        let activations = self.forward(inputs);
        let logits = self.logits_from_activations(&activations);
        let loss = masked_cross_entropy_loss(&logits, &targets, target_mask);

        LossOutput {
            loss,
            logits,
            targets,
            target_mask: target_mask.to_vec(),
            activations,
        }
    }

    pub fn lm_head_gradient(&self, output: &LossOutput) -> Array2<f32> {
        let grad_logits =
            masked_logits_gradient(&output.logits, &output.targets, &output.target_mask);

        output.activations.hidden.t().dot(&grad_logits)
    }

    pub fn apply_lm_head_update(&mut self, update: &Array2<f32>) {
        self.lm_head -= update;
    }

    pub fn lm_head_dim(&self) -> (usize, usize) {
        self.lm_head.dim()
    }

    pub fn lm_head_weight(&self) -> Array2<f32> {
        self.lm_head.clone()
    }

    pub fn generate_greedy(&self, prompt: &[usize], max_new_tokens: usize) -> Vec<usize> {
        let mut tokens = prompt.to_vec();

        for _ in 0..max_new_tokens {
            let window_start = tokens.len().saturating_sub(self.config.seq_len);
            let window = &tokens[window_start..];
            let activations = self.forward(window);
            let logits = self.logits_from_activations(&activations);
            let last = logits.row(logits.nrows() - 1);
            let next = argmax(last.as_slice().expect("logits row should be contiguous"));
            tokens.push(next);
        }

        tokens
    }
}

impl TransformerLayer {
    fn new(config: &ModelConfig, rng: &mut StdRng, scale: f32) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let kv_hidden = config.num_key_value_heads * head_dim;

        Self {
            input_norm_weight: Array1::ones(config.hidden_size),
            q_proj: rand_matrix(config.hidden_size, config.hidden_size, rng, scale),
            k_proj: rand_matrix(config.hidden_size, kv_hidden, rng, scale),
            v_proj: rand_matrix(config.hidden_size, kv_hidden, rng, scale),
            o_proj: rand_matrix(config.hidden_size, config.hidden_size, rng, scale),
            post_attention_norm_weight: Array1::ones(config.hidden_size),
            gate_proj: rand_matrix(config.hidden_size, config.intermediate_size, rng, scale),
            up_proj: rand_matrix(config.hidden_size, config.intermediate_size, rng, scale),
            down_proj: rand_matrix(config.intermediate_size, config.hidden_size, rng, scale),
        }
    }

    fn forward(&self, config: &ModelConfig, input: &Array2<f32>) -> Array2<f32> {
        let normed = rms_norm(input, &self.input_norm_weight, config.rms_norm_eps);
        let attention_output = attention(config, &normed, self);
        let after_attention = input + &attention_output;

        let mlp_input = rms_norm(
            &after_attention,
            &self.post_attention_norm_weight,
            config.rms_norm_eps,
        );
        let gated = silu(&mlp_input.dot(&self.gate_proj)) * mlp_input.dot(&self.up_proj);
        let mlp_output = gated.dot(&self.down_proj);

        after_attention + mlp_output
    }
}

fn attention(config: &ModelConfig, input: &Array2<f32>, layer: &TransformerLayer) -> Array2<f32> {
    let seq_len = input.nrows();
    let hidden_size = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = hidden_size / num_heads;
    let kv_repeat = num_heads / num_kv_heads;
    let kv_hidden = num_kv_heads * head_dim;

    let q = input
        .dot(&layer.q_proj)
        .into_shape_with_order((seq_len, num_heads, head_dim))
        .expect("q projection shape should match config");
    let k = input
        .dot(&layer.k_proj)
        .into_shape_with_order((seq_len, num_kv_heads, head_dim))
        .expect("k projection shape should match config");
    let v = input
        .dot(&layer.v_proj)
        .into_shape_with_order((seq_len, num_kv_heads, head_dim))
        .expect("v projection shape should match config");
    debug_assert_eq!(kv_hidden, layer.k_proj.ncols());

    let q = apply_rope(q);
    let k = apply_rope(k);
    let mut context = Array3::zeros((seq_len, num_heads, head_dim));

    for query_pos in 0..seq_len {
        for head in 0..num_heads {
            let kv_head = (head / kv_repeat) % num_kv_heads;
            let mut scores = Vec::with_capacity(query_pos + 1);

            for key_pos in 0..=query_pos {
                let q_vec = q.slice(s![query_pos, head, ..]);
                let k_vec = k.slice(s![key_pos, kv_head, ..]);
                scores.push(q_vec.dot(&k_vec) / (head_dim as f32).sqrt());
            }

            let probs = softmax_vec(&scores);
            for (key_pos, prob) in probs.into_iter().enumerate() {
                let v_vec = v.slice(s![key_pos, kv_head, ..]);
                for dim in 0..head_dim {
                    context[[query_pos, head, dim]] += prob * v_vec[dim];
                }
            }
        }
    }

    context
        .into_shape_with_order((seq_len, hidden_size))
        .expect("attention context shape should match config")
        .dot(&layer.o_proj)
}

fn rms_norm(input: &Array2<f32>, weight: &Array1<f32>, eps: f32) -> Array2<f32> {
    let mut output = input.clone();
    for mut row in output.axis_iter_mut(Axis(0)) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        row *= scale;
        row *= weight;
    }
    output
}

fn apply_rope(mut values: Array3<f32>) -> Array3<f32> {
    let (_, _, head_dim) = values.dim();
    for position in 0..values.dim().0 {
        for pair in (0..head_dim).step_by(2) {
            let theta = position as f32 / 10000_f32.powf(pair as f32 / head_dim as f32);
            let cos = theta.cos();
            let sin = theta.sin();

            for head in 0..values.dim().1 {
                let x0 = values[[position, head, pair]];
                let x1 = values[[position, head, pair + 1]];
                values[[position, head, pair]] = x0 * cos - x1 * sin;
                values[[position, head, pair + 1]] = x0 * sin + x1 * cos;
            }
        }
    }
    values
}

fn silu(values: &Array2<f32>) -> Array2<f32> {
    values.mapv(|value| value / (1.0 + (-value).exp()))
}

fn softmax(logits: &Array2<f32>) -> Array2<f32> {
    let mut output = logits.clone();
    for mut row in output.axis_iter_mut(Axis(0)) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.mapv_inplace(|value| (value - max).exp());
        let sum = row.sum();
        row /= sum;
    }
    output
}

fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = logits
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let sum = probs.iter().sum::<f32>();
    for prob in &mut probs {
        *prob /= sum;
    }
    probs
}

pub fn masked_cross_entropy_loss(
    logits: &Array2<f32>,
    targets: &[usize],
    target_mask: &[bool],
) -> f32 {
    let probs = softmax(logits);
    let mut loss = 0.0;
    let mut denom = 0usize;

    for (position, target) in targets.iter().copied().enumerate() {
        if !target_mask[position] {
            continue;
        }
        loss -= probs[[position, target]].max(1e-9).ln();
        denom += 1;
    }

    assert!(denom > 0, "masked loss requires at least one target");
    loss / denom as f32
}

pub fn masked_logits_gradient(
    logits: &Array2<f32>,
    targets: &[usize],
    target_mask: &[bool],
) -> Array2<f32> {
    let mut grad_logits = softmax(logits);
    let denom = target_mask.iter().filter(|enabled| **enabled).count();
    assert!(denom > 0, "masked gradient requires at least one target");

    for (position, target) in targets.iter().copied().enumerate() {
        if target_mask[position] {
            grad_logits[[position, target]] -= 1.0;
            grad_logits
                .row_mut(position)
                .mapv_inplace(|value| value / denom as f32);
        } else {
            grad_logits.row_mut(position).fill(0.0);
        }
    }

    grad_logits
}

fn rand_matrix(rows: usize, cols: usize, rng: &mut StdRng, scale: f32) -> Array2<f32> {
    let normal = Normal::new(0.0, scale).expect("normal init should be valid");
    Array2::from_shape_fn((rows, cols), |_| normal.sample(rng))
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("argmax requires non-empty values")
}

impl AdamW {
    pub fn new(
        lm_head_dim: (usize, usize),
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            step: 0,
            lm_head_m: Array2::zeros(lm_head_dim),
            lm_head_v: Array2::zeros(lm_head_dim),
        }
    }

    pub fn step_lm_head(&mut self, model: &mut QwenLikeModel, grad: &Array2<f32>) {
        self.step += 1;
        self.lm_head_m = &self.lm_head_m * self.beta1 + grad * (1.0 - self.beta1);
        self.lm_head_v =
            &self.lm_head_v * self.beta2 + grad.mapv(|value| value * value) * (1.0 - self.beta2);

        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);
        let m_hat = &self.lm_head_m / bias_correction1;
        let v_hat = &self.lm_head_v / bias_correction2;
        let adam_update = m_hat / v_hat.mapv(|value| value.sqrt() + self.eps);
        let decoupled_decay = model.lm_head.mapv(|value| self.weight_decay * value);
        let update = (adam_update + decoupled_decay) * self.learning_rate;

        model.apply_lm_head_update(&update);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn one_batch_loss_decreases_with_adamw_step() {
        let mut model = QwenLikeModel::new(tiny_config(), 7);
        let mut optimizer = AdamW::new(model.lm_head_dim(), 0.05, 0.9, 0.999, 1e-8, 0.01);
        let tokens = vec![3, 10, 1, 8, 15, 6, 13, 4];
        let initial = model.loss(&tokens).loss;

        for _ in 0..30 {
            let output = model.loss(&tokens);
            let grad = model.lm_head_gradient(&output);
            optimizer.step_lm_head(&mut model, &grad);
        }

        let final_loss = model.loss(&tokens).loss;
        assert!(
            final_loss < initial,
            "expected final loss {final_loss} to be lower than initial loss {initial}"
        );
    }

    #[test]
    fn checkpoint_reload_preserves_loss() {
        let model = QwenLikeModel::new(tiny_config(), 11);
        let tokens = vec![3, 10, 1, 8, 15, 6, 13, 4];
        let before = model.loss(&tokens).loss;
        let contents = toml::to_string(&model).expect("model should serialize");
        let reloaded: QwenLikeModel = toml::from_str(&contents).expect("model should deserialize");
        let after = reloaded.loss(&tokens).loss;

        assert!((before - after).abs() < 1e-6);
    }

    #[test]
    fn greedy_generate_appends_tokens() {
        let model = QwenLikeModel::new(tiny_config(), 13);
        let prompt = vec![3, 10, 1, 8];
        let generated = model.generate_greedy(&prompt, 3);

        assert_eq!(generated.len(), prompt.len() + 3);
        assert!(generated.iter().all(|token| *token < 16));
    }

    #[test]
    fn repeated_microbatch_gradient_accumulation_matches_single_gradient() {
        let model = QwenLikeModel::new(tiny_config(), 17);
        let tokens = vec![3, 10, 1, 8, 15, 6, 13, 4];
        let output = model.loss(&tokens);
        let single = model.lm_head_gradient(&output);
        let mut accumulated = Array2::zeros(model.lm_head_dim());

        for _ in 0..4 {
            let output = model.loss(&tokens);
            accumulated += &model.lm_head_gradient(&output);
        }
        accumulated /= 4.0;

        let max_delta = (&single - &accumulated)
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f32::max);
        assert!(max_delta < 1e-7, "max gradient delta was {max_delta}");
    }

    #[test]
    fn masked_loss_ignores_disabled_targets() {
        let logits = ndarray::array![[8.0, 0.0], [0.0, 8.0]];
        let targets = vec![1, 1];
        let masked = masked_cross_entropy_loss(&logits, &targets, &[false, true]);
        let unmasked = masked_cross_entropy_loss(&logits, &targets, &[true, true]);

        assert!(masked < 0.001);
        assert!(unmasked > 3.0);
    }
}
