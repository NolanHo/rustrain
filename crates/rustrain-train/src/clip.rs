use ndarray::Array2;

pub fn clip_gradient(grad: &mut Array2<f32>, max_grad_norm: Option<f32>) -> f32 {
    let grad_norm = l2_norm(grad);
    if let Some(max_grad_norm) = max_grad_norm {
        if grad_norm > max_grad_norm {
            *grad *= max_grad_norm / (grad_norm + 1e-12);
            return max_grad_norm;
        }
    }
    grad_norm
}

pub fn l2_norm(values: &Array2<f32>) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}
