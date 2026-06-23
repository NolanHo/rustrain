use rustrain_core::runtime::{Config, LrScheduler};

pub fn learning_rate_for_step(config: &Config, step: u64) -> f64 {
    match config.train.lr_scheduler {
        LrScheduler::Constant => config.train.learning_rate as f64,
        LrScheduler::LinearDecay => {
            let max_steps = config.train.max_steps.max(1) as f64;
            let progress = (step.saturating_sub(1) as f64 / max_steps).clamp(0.0, 1.0);
            config.train.learning_rate as f64 * (1.0 - progress)
        }
    }
}
