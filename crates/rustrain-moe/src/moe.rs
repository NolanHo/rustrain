use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinyMoe {
    router: Array2<f32>,
    experts: Vec<Expert>,
    top_k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Expert {
    up: Array2<f32>,
    down: Array2<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekMoe {
    layers: Vec<DeepSeekMoeLayer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepSeekMoeLayer {
    shared_experts: Vec<Expert>,
    routed_experts: TinyMoe,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MoeStats {
    pub expert_load: Vec<usize>,
    pub load_balance_loss: f32,
    pub total_params: usize,
    pub activated_params: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DeepSeekMoeStats {
    pub layers: Vec<DeepSeekMoeLayerStats>,
    pub shared_params: usize,
    pub routed_params: usize,
    pub total_params: usize,
    pub activated_params: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DeepSeekMoeLayerStats {
    pub layer_index: usize,
    pub routed_expert_load: Vec<usize>,
    pub load_balance_loss: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MoeSmokeSummary {
    pub tiny_hidden_shape: [usize; 2],
    pub tiny_hidden_sum: f32,
    pub tiny: MoeStats,
    pub deepseek_hidden_shape: [usize; 2],
    pub deepseek_hidden_sum: f32,
    pub deepseek: DeepSeekMoeStats,
}

#[derive(Debug, Clone)]
pub struct MoeOutput {
    pub hidden: Array2<f32>,
    pub stats: MoeStats,
}

#[derive(Debug, Clone)]
pub struct DeepSeekMoeOutput {
    pub hidden: Array2<f32>,
    pub stats: DeepSeekMoeStats,
}

impl TinyMoe {
    pub fn new(
        router: Array2<f32>,
        experts: Vec<(Array2<f32>, Array2<f32>)>,
        top_k: usize,
    ) -> Self {
        Self {
            router,
            experts: experts
                .into_iter()
                .map(|(up, down)| Expert { up, down })
                .collect(),
            top_k,
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> MoeOutput {
        let logits = input.dot(&self.router);
        let mut output = Array2::zeros(input.dim());
        let mut expert_load = vec![0usize; self.experts.len()];

        for token_index in 0..input.nrows() {
            let scores = logits.row(token_index).to_vec();
            let selected = top_k_indices(&scores, self.top_k);
            let weights = softmax_selected(&scores, &selected);

            for (expert_index, weight) in selected.into_iter().zip(weights) {
                expert_load[expert_index] += 1;
                let expert_output =
                    self.experts[expert_index].forward(&input.row(token_index).to_owned());
                for hidden_index in 0..output.ncols() {
                    output[[token_index, hidden_index]] += weight * expert_output[hidden_index];
                }
            }
        }

        MoeOutput {
            hidden: output,
            stats: MoeStats {
                load_balance_loss: load_balance_loss(&expert_load),
                total_params: self.total_params(),
                activated_params: self.activated_params(),
                expert_load,
            },
        }
    }

    pub fn total_params(&self) -> usize {
        self.router.len()
            + self
                .experts
                .iter()
                .map(|expert| expert.up.len() + expert.down.len())
                .sum::<usize>()
    }

    pub fn activated_params(&self) -> usize {
        let per_expert = self
            .experts
            .first()
            .map(|expert| expert.up.len() + expert.down.len())
            .unwrap_or(0);
        self.router.len() + self.top_k * per_expert
    }
}

impl DeepSeekMoe {
    pub fn new(
        layers: Vec<(
            Vec<(Array2<f32>, Array2<f32>)>,
            Array2<f32>,
            Vec<(Array2<f32>, Array2<f32>)>,
            usize,
        )>,
    ) -> Self {
        Self {
            layers: layers
                .into_iter()
                .map(
                    |(shared_experts, router, routed_experts, top_k)| DeepSeekMoeLayer {
                        shared_experts: shared_experts
                            .into_iter()
                            .map(|(up, down)| Expert { up, down })
                            .collect(),
                        routed_experts: TinyMoe::new(router, routed_experts, top_k),
                    },
                )
                .collect(),
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> DeepSeekMoeOutput {
        let mut hidden = input.clone();
        let mut layer_stats = Vec::with_capacity(self.layers.len());

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let routed = layer.routed_experts.forward(&hidden);
            let shared = layer.shared_forward(&hidden);
            hidden = routed.hidden + shared;
            layer_stats.push(DeepSeekMoeLayerStats {
                layer_index,
                routed_expert_load: routed.stats.expert_load,
                load_balance_loss: routed.stats.load_balance_loss,
            });
        }

        DeepSeekMoeOutput {
            hidden,
            stats: DeepSeekMoeStats {
                layers: layer_stats,
                shared_params: self.shared_params(),
                routed_params: self.routed_params(),
                total_params: self.total_params(),
                activated_params: self.activated_params(),
            },
        }
    }

    pub fn shared_params(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.shared_params())
            .sum::<usize>()
    }

    pub fn routed_params(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.routed_experts.total_params())
            .sum::<usize>()
    }

    pub fn total_params(&self) -> usize {
        self.shared_params() + self.routed_params()
    }

    pub fn activated_params(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.shared_params() + layer.routed_experts.activated_params())
            .sum::<usize>()
    }
}

impl DeepSeekMoeLayer {
    fn shared_forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = Array2::zeros(input.dim());
        if self.shared_experts.is_empty() {
            return output;
        }

        for token_index in 0..input.nrows() {
            let token = input.row(token_index).to_owned();
            for expert in &self.shared_experts {
                let expert_output = expert.forward(&token);
                for hidden_index in 0..output.ncols() {
                    output[[token_index, hidden_index]] +=
                        expert_output[hidden_index] / self.shared_experts.len() as f32;
                }
            }
        }

        output
    }

    fn shared_params(&self) -> usize {
        self.shared_experts
            .iter()
            .map(|expert| expert.up.len() + expert.down.len())
            .sum::<usize>()
    }
}

impl Expert {
    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let hidden = input.dot(&self.up).mapv(|value| value.max(0.0));
        hidden.dot(&self.down)
    }
}
pub fn deepseek_moe_smoke() -> DeepSeekMoeStats {
    let router = Array2::ones((2, 2));
    let shared = vec![(Array2::eye(2), Array2::eye(2))];
    let routed = vec![
        (Array2::eye(2), Array2::eye(2)),
        (
            Array2::from_diag(&Array1::from_vec(vec![0.5, 0.5])),
            Array2::eye(2),
        ),
    ];
    let moe = DeepSeekMoe::new(vec![
        (shared.clone(), router.clone(), routed.clone(), 1),
        (shared, router, routed, 1),
    ]);
    let input = Array2::ones((2, 2));
    let output = moe.forward(&input);
    debug_assert_eq!(output.hidden.nrows(), 2);
    output.stats
}
fn top_k_indices(scores: &[f32], top_k: usize) -> Vec<usize> {
    let mut indices = (0..scores.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        scores[*right]
            .total_cmp(&scores[*left])
            .then(left.cmp(right))
    });
    indices.truncate(top_k.min(indices.len()));
    indices
}

fn softmax_selected(scores: &[f32], selected: &[usize]) -> Vec<f32> {
    let max = selected
        .iter()
        .map(|index| scores[*index])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut values = selected
        .iter()
        .map(|index| (scores[*index] - max).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    for value in &mut values {
        *value /= sum;
    }
    values
}

fn load_balance_loss(expert_load: &[usize]) -> f32 {
    let total = expert_load.iter().sum::<usize>().max(1) as f32;
    let expected = 1.0 / expert_load.len().max(1) as f32;
    expert_load
        .iter()
        .map(|load| {
            let fraction = *load as f32 / total;
            (fraction - expected).powi(2)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    fn tiny_moe() -> TinyMoe {
        let router = array![[2.0, 0.0], [0.0, 2.0]];
        let expert0 = (array![[1.0, 0.0], [0.0, 1.0]], Array2::eye(2));
        let expert1 = (array![[0.5, 0.0], [0.0, 0.5]], Array2::eye(2));
        TinyMoe::new(router, vec![expert0, expert1], 1)
    }

    #[test]
    fn router_records_expert_load() {
        let moe = tiny_moe();
        let input = array![[1.0, 0.0], [0.0, 1.0]];
        let output = moe.forward(&input);

        assert_eq!(output.stats.expert_load, vec![1, 1]);
        assert_eq!(output.hidden.nrows(), 2);
    }

    #[test]
    fn activated_params_are_less_than_total_params() {
        let router = Array2::ones((2, 4));
        let expert = (Array2::eye(2), Array2::eye(2));
        let moe = TinyMoe::new(
            router,
            vec![expert.clone(), expert.clone(), expert.clone(), expert],
            1,
        );

        assert!(moe.activated_params() < moe.total_params());
    }

    #[test]
    fn load_balance_loss_is_zero_for_uniform_load() {
        assert_eq!(load_balance_loss(&[2, 2]), 0.0);
    }

    #[test]
    fn checkpoint_roundtrip_preserves_moe_output() {
        let moe = tiny_moe();
        let input = array![[1.0, 0.0], [0.0, 1.0]];
        let before = moe.forward(&input).hidden;
        let contents = toml::to_string(&moe).expect("moe should serialize");
        let loaded: TinyMoe = toml::from_str(&contents).expect("moe should deserialize");

        assert_eq!(before, loaded.forward(&input).hidden);
    }

    #[test]
    fn deepseek_moe_combines_shared_and_routed_paths() {
        let shared = vec![(Array2::eye(2), Array2::eye(2))];
        let router = array![[2.0, 0.0], [0.0, 2.0]];
        let routed0 = (Array2::eye(2), Array2::eye(2));
        let routed1 = (
            Array2::from_diag(&Array1::from_vec(vec![2.0, 2.0])),
            Array2::eye(2),
        );
        let moe = DeepSeekMoe::new(vec![(shared, router, vec![routed0, routed1], 1)]);
        let input = array![[1.0, 0.0], [0.0, 1.0]];
        let output = moe.forward(&input);

        assert_eq!(output.hidden, array![[2.0, 0.0], [0.0, 3.0]]);
        assert_eq!(output.stats.layers[0].routed_expert_load, vec![1, 1]);
    }

    #[test]
    fn deepseek_moe_reports_shared_and_routed_parameter_counts() {
        let shared = vec![
            (Array2::eye(2), Array2::eye(2)),
            (Array2::eye(2), Array2::eye(2)),
        ];
        let router = Array2::ones((2, 4));
        let routed_expert = (Array2::eye(2), Array2::eye(2));
        let moe = DeepSeekMoe::new(vec![(
            shared,
            router,
            vec![
                routed_expert.clone(),
                routed_expert.clone(),
                routed_expert.clone(),
                routed_expert,
            ],
            1,
        )]);

        let stats = moe.forward(&array![[1.0, 1.0]]).stats;

        assert_eq!(stats.shared_params, 16);
        assert_eq!(stats.routed_params, 40);
        assert_eq!(stats.total_params, 56);
        assert_eq!(stats.activated_params, 32);
        assert!(stats.activated_params < stats.total_params);
    }
}
