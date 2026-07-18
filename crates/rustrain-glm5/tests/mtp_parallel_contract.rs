//! CPU-only mathematical parity tests for Megatron-style MTP parallelism.
//!
//! The tests model collective operations with ordinary `Vec<f64>` values. They
//! intentionally avoid `tch`, CUDA, NCCL, and the dynamically loaded kernels so
//! that the distributed contracts remain testable on a CPU-only build host.

const EPS: f64 = 1.0e-11;

#[test]
fn megatron_tp_ep_rank_groups_are_not_cartesian_tp_times_ep() {
    use rustrain_glm5::tp_cp::{
        glm5_megatron_dense_dp_group, glm5_megatron_expert_dp_group, glm5_megatron_expert_ep_group,
        glm5_megatron_rank_coordinates,
    };

    // CP=1, ETP=1, TP=4, EP=8, DP=8. Rank 13 is dense TP=1/DP=3 but
    // expert EP=5/expert-DP=1, exactly as Megatron's two rank generators.
    let coords = glm5_megatron_rank_coordinates(13, 32, 4, 8).unwrap();
    assert_eq!(coords.tp_rank, 1);
    assert_eq!(coords.dense_dp_rank, 3);
    assert_eq!(coords.ep_rank, 5);
    assert_eq!(coords.expert_dp_rank, 1);
    assert_eq!(
        glm5_megatron_dense_dp_group(coords),
        vec![1, 5, 9, 13, 17, 21, 25, 29]
    );
    assert_eq!(
        glm5_megatron_expert_ep_group(coords),
        (8..16).collect::<Vec<_>>()
    );
    assert_eq!(glm5_megatron_expert_dp_group(coords), vec![5, 13, 21, 29]);
    assert!(glm5_megatron_rank_coordinates(0, 31, 4, 8).is_err());
}

#[test]
fn megatron_tp4_ep8_groups_cover_each_rank_in_the_correct_domain() {
    use rustrain_glm5::tp_cp::{
        glm5_megatron_dense_dp_group, glm5_megatron_expert_dp_group, glm5_megatron_expert_ep_group,
        glm5_megatron_rank_coordinates,
    };

    for rank in 0..32 {
        let coords = glm5_megatron_rank_coordinates(rank, 32, 4, 8).unwrap();
        let tp_start = coords.dense_dp_rank * coords.tp_size;
        let tp_group = (tp_start..tp_start + coords.tp_size).collect::<Vec<_>>();
        assert_eq!(
            tp_group,
            (rank / 4 * 4..rank / 4 * 4 + 4).collect::<Vec<_>>()
        );
        assert!(tp_group.contains(&rank));

        let dense_dp = glm5_megatron_dense_dp_group(coords);
        assert_eq!(dense_dp.len(), 8);
        assert!(dense_dp.contains(&rank));
        assert!(dense_dp.iter().all(|member| member % 4 == rank % 4));

        let expert_ep = glm5_megatron_expert_ep_group(coords);
        assert_eq!(expert_ep.len(), 8);
        assert!(expert_ep.contains(&rank));
        assert!(expert_ep.iter().all(|member| member / 8 == rank / 8));

        let expert_dp = glm5_megatron_expert_dp_group(coords);
        assert_eq!(expert_dp.len(), 4);
        assert!(expert_dp.contains(&rank));
        assert!(expert_dp.iter().all(|member| member % 8 == rank % 8));
    }
}

#[test]
fn loss_numerator_and_count_reduce_over_dense_dp_not_expert_ep() {
    use rustrain_glm5::tp_cp::{
        glm5_megatron_dense_dp_group, glm5_megatron_expert_ep_group, glm5_megatron_rank_coordinates,
    };

    // Each normal TP group sees one dense-DP sample, so ranks 4*d..4*d+3
    // carry the same numerator/count. Counts intentionally differ by sample.
    let numerators = [1.0_f64, 9.0, 4.0, 20.0, 7.0, 18.0, 11.0, 32.0];
    let counts = [2.0_f64, 3.0, 5.0, 4.0, 7.0, 6.0, 8.0, 5.0];
    let expected = numerators.iter().sum::<f64>() / counts.iter().sum::<f64>();

    let coords = glm5_megatron_rank_coordinates(0, 32, 4, 8).unwrap();
    let dense_dp = glm5_megatron_dense_dp_group(coords);
    let dense_sum = dense_dp
        .iter()
        .map(|rank| numerators[rank / 4])
        .sum::<f64>();
    let dense_count = dense_dp.iter().map(|rank| counts[rank / 4]).sum::<f64>();
    assert!((dense_sum / dense_count - expected).abs() < EPS);

    // An expert-EP group contains four SP/TP ranks for each of two samples.
    // Using it for loss normalization only sees that pair and is not global DP.
    let expert_ep = glm5_megatron_expert_ep_group(coords);
    let ep_sum = expert_ep
        .iter()
        .map(|rank| numerators[rank / 4])
        .sum::<f64>();
    let ep_count = expert_ep.iter().map(|rank| counts[rank / 4]).sum::<f64>();
    assert!((ep_sum / ep_count - expected).abs() > 0.25);
}

#[test]
fn dense_adapter_gradients_reduce_over_tp_then_dense_dp() {
    use rustrain_glm5::tp_cp::{glm5_megatron_dense_dp_group, glm5_megatron_rank_coordinates};

    // One full-sized dense adapter is sliced by normal TP in the forward.
    // Its gradient must first sum the four shard contributions for one sample,
    // then sum the eight samples through the matching dense-DP group.
    let local_gradient = (0..32)
        .map(|rank| (rank / 4 + 1) as f64 * (rank % 4 + 1) as f64)
        .collect::<Vec<_>>();
    let expected = local_gradient.iter().sum::<f64>();

    let mut tp_reduced = [0.0_f64; 8];
    for dense_dp_rank in 0..8 {
        tp_reduced[dense_dp_rank] = (0..4)
            .map(|tp_rank| local_gradient[dense_dp_rank * 4 + tp_rank])
            .sum();
    }

    let coords = glm5_megatron_rank_coordinates(0, 32, 4, 8).unwrap();
    let dense_dp = glm5_megatron_dense_dp_group(coords);
    let reduced = dense_dp
        .iter()
        .map(|rank| tp_reduced[rank / 4])
        .sum::<f64>();
    assert!((reduced - expected).abs() < EPS);
}

#[test]
fn expert_gradients_reduce_over_expert_dp_replicas() {
    use rustrain_glm5::tp_cp::{glm5_megatron_expert_dp_group, glm5_megatron_rank_coordinates};

    // EP group k dispatches two dense-DP samples (2k and 2k+1) to each
    // expert owner. The same expert's replicas are e + 8*k.
    let per_sample_gradient = [0.5_f64, -0.2, 1.1, 0.7, -0.4, 0.9, 1.6, -0.3];
    let expert = 5;
    let coords = glm5_megatron_rank_coordinates(expert, 32, 4, 8).unwrap();
    let expert_dp = glm5_megatron_expert_dp_group(coords);
    assert_eq!(expert_dp, vec![5, 13, 21, 29]);

    let reduced = expert_dp
        .iter()
        .map(|rank| {
            let replica = rank / 8;
            per_sample_gradient[2 * replica] + per_sample_gradient[2 * replica + 1]
        })
        .sum::<f64>();
    assert!((reduced - per_sample_gradient.iter().sum::<f64>()).abs() < EPS);
}

#[test]
fn session_rejects_combined_tp_ep_before_runtime_collectives() {
    use rustrain_glm5::session_tp_cp::validate_glm5_tp_ep_session_topology;

    assert!(validate_glm5_tp_ep_session_topology(4, 1).is_ok());
    assert!(validate_glm5_tp_ep_session_topology(1, 8).is_ok());
    let error = validate_glm5_tp_ep_session_topology(4, 8).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("sequence-parallel"));
    assert!(message.contains("expert-EP"));
    assert!(message.contains("expert-DP"));
    assert!(validate_glm5_tp_ep_session_topology(0, 1).is_err());
    assert!(validate_glm5_tp_ep_session_topology(1, 0).is_err());
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = EPS * (1.0 + expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
        );
    }
}

fn matmul_rhs_transpose(
    lhs: &[f64],
    rows: usize,
    input_size: usize,
    rhs: &[f64],
    output_size: usize,
) -> Vec<f64> {
    assert_eq!(lhs.len(), rows * input_size);
    assert_eq!(rhs.len(), output_size * input_size);
    let mut output = vec![0.0; rows * output_size];
    for row in 0..rows {
        for out in 0..output_size {
            output[row * output_size + out] = (0..input_size)
                .map(|input| lhs[row * input_size + input] * rhs[out * input_size + input])
                .sum();
        }
    }
    output
}

fn input_gradient(
    output_gradient: &[f64],
    rows: usize,
    output_size: usize,
    weight: &[f64],
    input_size: usize,
) -> Vec<f64> {
    assert_eq!(output_gradient.len(), rows * output_size);
    assert_eq!(weight.len(), output_size * input_size);
    let mut gradient = vec![0.0; rows * input_size];
    for row in 0..rows {
        for input in 0..input_size {
            gradient[row * input_size + input] = (0..output_size)
                .map(|out| {
                    output_gradient[row * output_size + out] * weight[out * input_size + input]
                })
                .sum();
        }
    }
    gradient
}

fn output_rows(matrix: &[f64], rows: usize, columns: usize, start: usize, len: usize) -> Vec<f64> {
    assert!(start + len <= columns);
    let mut shard = Vec::with_capacity(rows * len);
    for row in 0..rows {
        shard.extend_from_slice(&matrix[row * columns + start..row * columns + start + len]);
    }
    shard
}

fn add_in_place(destination: &mut [f64], source: &[f64]) {
    assert_eq!(destination.len(), source.len());
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += source;
    }
}

#[test]
fn eh_proj_column_parallel_forward_concat_and_backward_input_sum_match_full_linear() {
    // MTP eh_proj maps [embedding_norm, hidden_norm] from 2H to H. Megatron
    // shards its H output features across TP ranks, gathers them before the
    // decoder, and sums the per-rank dX contributions in backward.
    let tokens = 3;
    let input_size = 6;
    let output_size = 4;
    let tp_size = 2;
    let input = vec![
        0.2, -0.1, 0.4, 0.7, -0.3, 0.5, // token 0
        -0.6, 0.8, 0.1, -0.2, 0.9, 0.3, // token 1
        0.5, 0.4, -0.7, 0.6, 0.2, -0.8, // token 2
    ];
    let weight = vec![
        0.3, -0.2, 0.5, 0.1, -0.4, 0.7, // output 0
        -0.6, 0.9, 0.2, -0.3, 0.8, 0.4, // output 1
        0.1, 0.5, -0.7, 0.6, 0.2, -0.8, // output 2
        0.4, -0.9, 0.3, 0.7, -0.1, 0.2, // output 3
    ];
    let output_gradient = vec![
        0.7, -0.2, 0.4, 0.9, // token 0
        -0.3, 0.8, -0.5, 0.1, // token 1
        0.6, 0.2, -0.7, 0.5, // token 2
    ];

    let full_output = matmul_rhs_transpose(&input, tokens, input_size, &weight, output_size);
    let full_input_gradient =
        input_gradient(&output_gradient, tokens, output_size, &weight, input_size);

    let shard_size = output_size / tp_size;
    let mut gathered_output = vec![0.0; tokens * output_size];
    let mut reduced_input_gradient = vec![0.0; tokens * input_size];
    for rank in 0..tp_size {
        let output_start = rank * shard_size;
        let weight_shard =
            &weight[output_start * input_size..(output_start + shard_size) * input_size];
        let local_output =
            matmul_rhs_transpose(&input, tokens, input_size, weight_shard, shard_size);
        let local_output_gradient = output_rows(
            &output_gradient,
            tokens,
            output_size,
            output_start,
            shard_size,
        );
        let local_input_gradient = input_gradient(
            &local_output_gradient,
            tokens,
            shard_size,
            weight_shard,
            input_size,
        );

        for token in 0..tokens {
            gathered_output[token * output_size + output_start
                ..token * output_size + output_start + shard_size]
                .copy_from_slice(&local_output[token * shard_size..(token + 1) * shard_size]);
        }
        add_in_place(&mut reduced_input_gradient, &local_input_gradient);
    }

    assert_close(&gathered_output, &full_output);
    assert_close(&reduced_input_gradient, &full_input_gradient);
}

#[derive(Debug)]
struct CrossEntropyResult {
    losses: Vec<f64>,
    logit_gradient: Vec<f64>,
}

fn full_cross_entropy(logits: &[f64], targets: &[usize], vocab_size: usize) -> CrossEntropyResult {
    let tokens = targets.len();
    assert_eq!(logits.len(), tokens * vocab_size);
    let mut losses = Vec::with_capacity(tokens);
    let mut logit_gradient = vec![0.0; logits.len()];
    for (token, &target) in targets.iter().enumerate() {
        let row = &logits[token * vocab_size..(token + 1) * vocab_size];
        let global_max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator: f64 = row.iter().map(|logit| (logit - global_max).exp()).sum();
        losses.push(denominator.ln() + global_max - row[target]);
        for vocab in 0..vocab_size {
            logit_gradient[token * vocab_size + vocab] =
                (row[vocab] - global_max).exp() / denominator - f64::from(vocab == target);
        }
    }
    CrossEntropyResult {
        losses,
        logit_gradient,
    }
}

fn vocab_parallel_cross_entropy(
    logits: &[f64],
    targets: &[usize],
    vocab_size: usize,
    tp_size: usize,
) -> CrossEntropyResult {
    let tokens = targets.len();
    assert_eq!(logits.len(), tokens * vocab_size);
    assert_eq!(vocab_size % tp_size, 0);
    let shard_size = vocab_size / tp_size;
    let mut losses = vec![0.0; tokens];
    let mut logit_gradient = vec![0.0; logits.len()];

    for token in 0..tokens {
        // MAX all-reduce across the maxima computed from each vocab shard.
        let local_maxima: Vec<f64> = (0..tp_size)
            .map(|rank| {
                let start = token * vocab_size + rank * shard_size;
                logits[start..start + shard_size]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        let global_max = local_maxima
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // SUM all-reduce across local exponent sums. Only the rank owning the
        // target contributes a non-zero target logit to the target SUM.
        let mut global_exp_sum = 0.0;
        let mut reduced_target_logit = 0.0;
        for rank in 0..tp_size {
            let vocab_start = rank * shard_size;
            let vocab_end = vocab_start + shard_size;
            let row_start = token * vocab_size + vocab_start;
            global_exp_sum += logits[row_start..row_start + shard_size]
                .iter()
                .map(|logit| (logit - global_max).exp())
                .sum::<f64>();
            if (vocab_start..vocab_end).contains(&targets[token]) {
                reduced_target_logit += logits[token * vocab_size + targets[token]];
            }
        }

        losses[token] = global_exp_sum.ln() + global_max - reduced_target_logit;
        for rank in 0..tp_size {
            let vocab_start = rank * shard_size;
            for local_vocab in 0..shard_size {
                let vocab = vocab_start + local_vocab;
                logit_gradient[token * vocab_size + vocab] =
                    (logits[token * vocab_size + vocab] - global_max).exp() / global_exp_sum
                        - f64::from(vocab == targets[token]);
            }
        }
    }

    CrossEntropyResult {
        losses,
        logit_gradient,
    }
}

#[test]
fn vocab_parallel_ce_matches_full_loss_logit_gradient_and_hidden_gradient() {
    let tokens = 3;
    let hidden_size = 3;
    let vocab_size = 8;
    let tp_size = 4;
    let hidden = vec![0.2, -0.4, 0.7, -0.1, 0.8, 0.3, 0.9, 0.2, -0.5];
    let vocab_weight = vec![
        0.1, 0.3, -0.2, -0.4, 0.7, 0.5, 0.6, -0.1, 0.8, -0.7, 0.2, 0.4, 0.9, 0.5, -0.3, 0.2, -0.8,
        0.6, -0.5, 0.4, 0.1, 0.8, -0.6, -0.2,
    ];
    let targets = vec![0, 5, 7]; // Exercise the first, middle, and final TP shards.
    let logits = matmul_rhs_transpose(&hidden, tokens, hidden_size, &vocab_weight, vocab_size);

    let full = full_cross_entropy(&logits, &targets, vocab_size);
    let parallel = vocab_parallel_cross_entropy(&logits, &targets, vocab_size, tp_size);
    assert_close(&parallel.losses, &full.losses);
    assert_close(&parallel.logit_gradient, &full.logit_gradient);

    let full_hidden_gradient = input_gradient(
        &full.logit_gradient,
        tokens,
        vocab_size,
        &vocab_weight,
        hidden_size,
    );
    let shard_size = vocab_size / tp_size;
    let mut reduced_hidden_gradient = vec![0.0; tokens * hidden_size];
    for rank in 0..tp_size {
        let vocab_start = rank * shard_size;
        let local_gradient = output_rows(
            &parallel.logit_gradient,
            tokens,
            vocab_size,
            vocab_start,
            shard_size,
        );
        let local_weight =
            &vocab_weight[vocab_start * hidden_size..(vocab_start + shard_size) * hidden_size];
        add_in_place(
            &mut reduced_hidden_gradient,
            &input_gradient(
                &local_gradient,
                tokens,
                shard_size,
                local_weight,
                hidden_size,
            ),
        );
    }
    assert_close(&reduced_hidden_gradient, &full_hidden_gradient);
}

#[test]
fn column_then_row_parallel_mlp_forward_and_input_gradient_match_full_mlp() {
    let tokens = 2;
    let hidden_size = 3;
    let intermediate_size = 6;
    let tp_size = 3;
    let input = vec![0.2, -0.6, 0.9, -0.5, 0.7, 0.3];
    let up_weight = vec![
        0.1, 0.4, -0.2, -0.3, 0.8, 0.5, 0.7, -0.6, 0.2, 0.5, 0.1, -0.9, -0.4, 0.3, 0.6, 0.9, -0.2,
        0.1,
    ];
    // Row-parallel down projection: contiguous columns correspond to the
    // intermediate features owned by the same TP rank as the up projection.
    let down_weight = vec![
        0.3, -0.2, 0.7, 0.1, -0.5, 0.8, -0.6, 0.4, 0.2, 0.9, 0.3, -0.1, 0.5, 0.7, -0.4, -0.2, 0.6,
        0.1,
    ];
    let output_gradient = vec![0.4, -0.7, 0.2, -0.3, 0.5, 0.8];

    let full_pre_activation =
        matmul_rhs_transpose(&input, tokens, hidden_size, &up_weight, intermediate_size);
    let full_activation: Vec<f64> = full_pre_activation
        .iter()
        .map(|value| value.tanh())
        .collect();
    let full_output = matmul_rhs_transpose(
        &full_activation,
        tokens,
        intermediate_size,
        &down_weight,
        hidden_size,
    );
    let mut full_pre_activation_gradient = input_gradient(
        &output_gradient,
        tokens,
        hidden_size,
        &down_weight,
        intermediate_size,
    );
    for (gradient, activation) in full_pre_activation_gradient
        .iter_mut()
        .zip(&full_activation)
    {
        *gradient *= 1.0 - activation * activation;
    }
    let full_input_gradient = input_gradient(
        &full_pre_activation_gradient,
        tokens,
        intermediate_size,
        &up_weight,
        hidden_size,
    );

    let shard_size = intermediate_size / tp_size;
    let mut reduced_output = vec![0.0; tokens * hidden_size];
    let mut reduced_input_gradient = vec![0.0; tokens * hidden_size];
    for rank in 0..tp_size {
        let intermediate_start = rank * shard_size;
        let up_shard = &up_weight
            [intermediate_start * hidden_size..(intermediate_start + shard_size) * hidden_size];
        let local_pre_activation =
            matmul_rhs_transpose(&input, tokens, hidden_size, up_shard, shard_size);
        let local_activation: Vec<f64> = local_pre_activation
            .iter()
            .map(|value| value.tanh())
            .collect();

        let mut down_shard = Vec::with_capacity(hidden_size * shard_size);
        for output in 0..hidden_size {
            down_shard.extend_from_slice(
                &down_weight[output * intermediate_size + intermediate_start
                    ..output * intermediate_size + intermediate_start + shard_size],
            );
        }
        add_in_place(
            &mut reduced_output,
            &matmul_rhs_transpose(
                &local_activation,
                tokens,
                shard_size,
                &down_shard,
                hidden_size,
            ),
        );

        let mut local_pre_activation_gradient = input_gradient(
            &output_gradient,
            tokens,
            hidden_size,
            &down_shard,
            shard_size,
        );
        for (gradient, activation) in local_pre_activation_gradient
            .iter_mut()
            .zip(&local_activation)
        {
            *gradient *= 1.0 - activation * activation;
        }
        add_in_place(
            &mut reduced_input_gradient,
            &input_gradient(
                &local_pre_activation_gradient,
                tokens,
                shard_size,
                up_shard,
                hidden_size,
            ),
        );
    }

    assert_close(&reduced_output, &full_output);
    assert_close(&reduced_input_gradient, &full_input_gradient);
}

#[test]
fn tp2_column_projection_forward_and_low_rank_input_gradient_match_full_attention() {
    let tokens = 2;
    let hidden = 3;
    let low_rank = 2;
    let output = 4;
    let tp_size = 2;
    let input = vec![0.2, -0.6, 0.9, -0.5, 0.7, 0.3];
    let projection_a = vec![0.4, -0.2, 0.8, -0.7, 0.5, 0.1];
    let projection_b = vec![0.3, -0.9, 0.6, 0.2, -0.4, 0.7, 0.8, -0.1];
    let output_gradient = vec![0.2, -0.5, 0.7, 0.1, -0.3, 0.9, -0.4, 0.6];

    let low_rank_input = matmul_rhs_transpose(&input, tokens, hidden, &projection_a, low_rank);
    let full_output =
        matmul_rhs_transpose(&low_rank_input, tokens, low_rank, &projection_b, output);
    let full_low_rank_gradient =
        input_gradient(&output_gradient, tokens, output, &projection_b, low_rank);
    let full_input_gradient = input_gradient(
        &full_low_rank_gradient,
        tokens,
        low_rank,
        &projection_a,
        hidden,
    );

    let output_per_rank = output / tp_size;
    let mut gathered_output = vec![0.0; tokens * output];
    let mut summed_low_rank_gradient = vec![0.0; tokens * low_rank];
    for rank in 0..tp_size {
        let start = rank * output_per_rank;
        let b_shard = &projection_b[start * low_rank..(start + output_per_rank) * low_rank];
        let local_output =
            matmul_rhs_transpose(&low_rank_input, tokens, low_rank, b_shard, output_per_rank);
        let mut local_gradient = vec![0.0; tokens * output_per_rank];
        for token in 0..tokens {
            gathered_output[token * output + start..token * output + start + output_per_rank]
                .copy_from_slice(
                    &local_output[token * output_per_rank..(token + 1) * output_per_rank],
                );
            local_gradient[token * output_per_rank..(token + 1) * output_per_rank].copy_from_slice(
                &output_gradient[token * output + start..token * output + start + output_per_rank],
            );
        }
        add_in_place(
            &mut summed_low_rank_gradient,
            &input_gradient(&local_gradient, tokens, output_per_rank, b_shard, low_rank),
        );
    }
    let tp_input_gradient = input_gradient(
        &summed_low_rank_gradient,
        tokens,
        low_rank,
        &projection_a,
        hidden,
    );

    assert_close(&gathered_output, &full_output);
    assert_close(&summed_low_rank_gradient, &full_low_rank_gradient);
    assert_close(&tp_input_gradient, &full_input_gradient);
}

#[test]
fn tp2_replicated_kv_projection_sums_low_rank_and_rope_bypass_dgrad() {
    let tokens = 1;
    let hidden = 3;
    let kv_lora = 2;
    let rope = 2;
    let kv_a_weight = vec![
        0.4, -0.2, 0.8, -0.7, 0.5, 0.1, 0.3, 0.6, -0.4, -0.1, 0.9, 0.2,
    ];
    // Each TP rank contributes through its kv_b shard and through its local
    // attention heads consuming the replicated k_rope branch.
    let rank0_output_gradient = vec![0.2, -0.5, 0.7, 0.1];
    let rank1_output_gradient = vec![-0.3, 0.9, -0.4, 0.6];
    let summed_output_gradient = rank0_output_gradient
        .iter()
        .zip(&rank1_output_gradient)
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    let expected_input_gradient = input_gradient(
        &summed_output_gradient,
        tokens,
        kv_lora + rope,
        &kv_a_weight,
        hidden,
    );

    let rank0_input_gradient = input_gradient(
        &rank0_output_gradient,
        tokens,
        kv_lora + rope,
        &kv_a_weight,
        hidden,
    );
    let rank1_input_gradient = input_gradient(
        &rank1_output_gradient,
        tokens,
        kv_lora + rope,
        &kv_a_weight,
        hidden,
    );
    let mut reduced_input_gradient = rank0_input_gradient.clone();
    add_in_place(&mut reduced_input_gradient, &rank1_input_gradient);

    assert_close(&reduced_input_gradient, &expected_input_gradient);
    assert!(
        rank0_input_gradient
            .iter()
            .zip(&expected_input_gradient)
            .any(|(local, expected)| (local - expected).abs() > EPS),
        "the fixture must detect a k_rope branch that bypasses TP dgrad SUM"
    );
}

fn expert_transform(expert: usize, token: &[f64]) -> Vec<f64> {
    let scale = 0.5 + expert as f64 * 0.4;
    vec![
        scale * token[0] + 0.1 * token[1],
        -0.2 * token[0] + scale * token[1],
    ]
}

fn shared_transform(token: &[f64]) -> Vec<f64> {
    vec![
        0.25 * token[0] - 0.15 * token[1],
        0.3 * token[0] + 0.2 * token[1],
    ]
}

#[test]
fn ep_owner_permutation_and_inverse_preserve_router_weights_and_add_shared_once() {
    let hidden_size = 2;
    let tokens = vec![0.4, -0.2, -0.7, 0.5, 0.1, 0.9, 0.8, -0.6];
    let token_count = tokens.len() / hidden_size;
    let topk = 2;
    let experts = 4;
    let ep_size = 2;
    let expert_indices = vec![3, 0, 1, 2, 2, 0, 3, 1];
    let router_weights = vec![0.7, 0.3, 0.4, 0.6, 0.55, 0.45, 0.2, 0.8];

    let mut reference = vec![0.0; tokens.len()];
    for token in 0..token_count {
        let input = &tokens[token * hidden_size..(token + 1) * hidden_size];
        let shared = shared_transform(input);
        reference[token * hidden_size..(token + 1) * hidden_size].copy_from_slice(&shared);
        for slot in 0..topk {
            let assignment = token * topk + slot;
            let expert_output = expert_transform(expert_indices[assignment], input);
            for hidden in 0..hidden_size {
                reference[token * hidden_size + hidden] +=
                    router_weights[assignment] * expert_output[hidden];
            }
        }
    }

    // Stable sort models the all-to-all dispatch permutation. The owner is the
    // original flattened (token, top-k slot) index and is the inverse-return key.
    let mut dispatch_order: Vec<usize> = (0..token_count * topk).collect();
    dispatch_order.sort_by_key(|&owner| {
        let expert = expert_indices[owner];
        let ep_rank = expert / (experts / ep_size);
        (ep_rank, expert, owner)
    });
    assert_ne!(dispatch_order, (0..token_count * topk).collect::<Vec<_>>());

    let mut inverse_order = vec![usize::MAX; dispatch_order.len()];
    let mut returned_by_owner = vec![vec![0.0; hidden_size]; dispatch_order.len()];
    let mut returned_router_weights = vec![0.0; dispatch_order.len()];
    for (permuted_index, &owner) in dispatch_order.iter().enumerate() {
        inverse_order[owner] = permuted_index;
        let token = owner / topk;
        let expert = expert_indices[owner];
        let input = &tokens[token * hidden_size..(token + 1) * hidden_size];
        returned_by_owner[owner] = expert_transform(expert, input);
        returned_router_weights[owner] = router_weights[owner];
    }
    for owner in 0..dispatch_order.len() {
        assert_eq!(dispatch_order[inverse_order[owner]], owner);
    }
    assert_close(&returned_router_weights, &router_weights);

    let mut distributed = vec![0.0; tokens.len()];
    for token in 0..token_count {
        let input = &tokens[token * hidden_size..(token + 1) * hidden_size];
        let shared = shared_transform(input);
        distributed[token * hidden_size..(token + 1) * hidden_size].copy_from_slice(&shared);
    }
    for owner in 0..token_count * topk {
        let token = owner / topk;
        for hidden in 0..hidden_size {
            distributed[token * hidden_size + hidden] +=
                returned_router_weights[owner] * returned_by_owner[owner][hidden];
        }
    }

    assert_close(&distributed, &reference);

    // Adding shared output once per routed copy is a real top-k bug and this
    // fixture must distinguish it from the Megatron contract.
    let mut incorrectly_repeated_shared = distributed.clone();
    for token in 0..token_count {
        let input = &tokens[token * hidden_size..(token + 1) * hidden_size];
        let shared = shared_transform(input);
        for hidden in 0..hidden_size {
            incorrectly_repeated_shared[token * hidden_size + hidden] +=
                (topk - 1) as f64 * shared[hidden];
        }
    }
    assert!(
        incorrectly_repeated_shared
            .iter()
            .zip(&reference)
            .any(|(actual, expected)| (actual - expected).abs() > 1.0e-6)
    );
}

#[test]
fn unequal_microbatches_accumulate_loss_numerator_and_token_count_before_normalizing() {
    // Variable-length microbatches must not contribute equally. Megatron's
    // token-loss path aggregates loss numerators and valid-token counts, then
    // normalizes once over the global batch.
    let microbatch_losses = [vec![0.2, 0.6], vec![1.0, 1.4, 1.8, 2.2, 2.6]];
    let numerator: f64 = microbatch_losses.iter().flatten().sum();
    let token_count: usize = microbatch_losses.iter().map(Vec::len).sum();
    let globally_normalized = numerator / token_count as f64;

    let flattened: Vec<f64> = microbatch_losses.iter().flatten().copied().collect();
    assert_close(
        &[globally_normalized],
        &[flattened.iter().sum::<f64>() / 7.0],
    );

    let incorrectly_averaged_microbatch_means = microbatch_losses
        .iter()
        .map(|losses| losses.iter().sum::<f64>() / losses.len() as f64)
        .sum::<f64>()
        / microbatch_losses.len() as f64;
    assert!((globally_normalized - incorrectly_averaged_microbatch_means).abs() > 0.25);

    // d(sum(loss_i) / global_count)/d(loss_i) is identical for every valid
    // token, independent of which microbatch contains it.
    let expected_gradient = 1.0 / token_count as f64;
    let gradients: Vec<Vec<f64>> = microbatch_losses
        .iter()
        .map(|losses| vec![expected_gradient; losses.len()])
        .collect();
    assert_close(&gradients[0], &[1.0 / 7.0; 2]);
    assert_close(&gradients[1], &[1.0 / 7.0; 5]);
}
