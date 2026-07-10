//! HTTP API (axum) — RESTful endpoints for training session management.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::StreamExt;

use crate::metrics::StepMetric;
use crate::session::{InitLoRARequest, SessLoadDatasetRequest, SessLoadModelRequest, TrainInput};
use crate::state::SessionManager;

pub struct AppState {
    pub manager: Arc<SessionManager>,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/sessions", post(create_session))
        .route("/v1/sessions", get(list_sessions))
        .route("/v1/sessions/:id/load_model", post(load_model))
        .route("/v1/sessions/:id/load_dataset", post(load_dataset))
        .route("/v1/sessions/:id/init_lora", post(init_lora))
        .route("/v1/sessions/:id/train_step", post(train_step))
        .route("/v1/sessions/:id/eval_step", post(eval_step))
        .route("/v1/sessions/:id/save_checkpoint", post(save_checkpoint))
        .route("/v1/sessions/:id/load_checkpoint", post(load_checkpoint))
        .route("/v1/sessions/:id/export_adapter", post(export_adapter))
        .route("/v1/sessions/:id/add_lora", post(add_lora))
        .route("/v1/sessions/:id/remove_lora", post(remove_lora))
        .route("/v1/sessions/:id/list_lora", get(list_lora))
        .route("/v1/sessions/:id/metrics", get(stream_metrics))
        .route("/v1/sessions/:id/status", get(get_status))
        .with_state(state)
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn err_resp(msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse { error: msg.to_string() }),
    )
}

async fn create_session(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let session_id = req.session_id;
    state
        .manager
        .create_session(session_id.clone())
        .await
        .map_err(|e| err_resp(&e))?;
    Ok(Json(CreateSessionResponse { session_id }))
}

#[derive(Deserialize)]
struct CreateSessionRequest {
    session_id: String,
}
#[derive(Serialize)]
struct CreateSessionResponse {
    session_id: String,
}

async fn list_sessions(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<String>> {
    Json(state.manager.list_sessions().await)
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<LoadModelHttp>,
) -> Result<Json<()>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let mut s = session.lock().await;
    s.load_model(SessLoadModelRequest {
        model_path: req.model_path,
        config_toml: req.config_toml,
    })
    .map_err(|e| err_resp(&e.to_string()))?;
    Ok(Json(()))
}

#[derive(Deserialize)]
struct LoadModelHttp {
    model_path: String,
    config_toml: String,
}

async fn load_dataset(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<LoadDatasetHttp>,
) -> Result<Json<LoadDatasetResponse>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let mut s = session.lock().await;
    let n = s
        .load_dataset(SessLoadDatasetRequest {
            jsonl_path: req.jsonl_path,
            seq_len: req.seq_len,
        })
        .map_err(|e| err_resp(&e.to_string()))?;
    Ok(Json(LoadDatasetResponse { num_examples: n }))
}

#[derive(Deserialize)]
struct LoadDatasetHttp {
    jsonl_path: String,
    seq_len: usize,
}
#[derive(Serialize)]
struct LoadDatasetResponse {
    num_examples: usize,
}

async fn init_lora(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<InitLoRAHttp>,
) -> Result<Json<InitLoRAResponse>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let mut s = session.lock().await;
    let count = s
        .init_lora(InitLoRARequest {
            rank: req.rank,
            alpha: req.alpha,
            target_layers: req.target_layers,
            target_modules: req.target_modules,
            lr: req.lr,
            beta1: req.beta1,
            beta2: req.beta2,
            eps: req.eps,
        })
        .map_err(|e| err_resp(&e.to_string()))?;
    Ok(Json(InitLoRAResponse {
        lora_param_count: count,
    }))
}

#[derive(Deserialize)]
struct InitLoRAHttp {
    rank: i64,
    alpha: i64,
    target_layers: Vec<usize>,
    target_modules: Vec<String>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
}
#[derive(Serialize)]
struct InitLoRAResponse {
    lora_param_count: usize,
}

async fn train_step(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<TrainStepHttp>,
) -> Result<Json<TrainStepResponse>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let input_ids = decode_tensor(&req.input_ids)
        .map_err(|e| err_resp(&e))?;
    let target_mask = decode_tensor(&req.target_mask)
        .map_err(|e| err_resp(&e))?;
    let attention_mask = decode_tensor(&req.attention_mask)
        .map_err(|e| err_resp(&e))?;

    let mut s = session.lock().await;
    let result = s
        .train_step(TrainInput {
            input_ids,
            target_mask,
            attention_mask,
        })
        .map_err(|e| err_resp(&e.to_string()))?;
    Ok(Json(TrainStepResponse {
        loss: result.loss,
        step: result.step,
    }))
}

#[derive(Deserialize)]
struct TensorHttp {
    data: String, // base64 encoded
    shape: Vec<i64>,
    dtype: String,
}
#[derive(Deserialize)]
struct TrainStepHttp {
    input_ids: TensorHttp,
    target_mask: TensorHttp,
    attention_mask: TensorHttp,
}
#[derive(Serialize)]
struct TrainStepResponse {
    loss: f64,
    step: u64,
}

fn decode_tensor(t: &TensorHttp) -> Result<tch::Tensor, String> {
    use base64::{engine::general_purpose, Engine};
    let bytes = general_purpose::STANDARD
        .decode(&t.data)
        .map_err(|e| format!("base64 decode: {e}"))?;
    let kind = match t.dtype.as_str() {
        "int64" => tch::Kind::Int64,
        "float32" => tch::Kind::Float,
        "bfloat16" => tch::Kind::BFloat16,
        "float64" => tch::Kind::Double,
        _ => return Err(format!("unsupported dtype: {}", t.dtype)),
    };
    let tensor = match kind {
        tch::Kind::Int64 => {
            let vals: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            tch::Tensor::from_slice(&vals)
        }
        tch::Kind::Float => {
            let vals: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            tch::Tensor::from_slice(&vals)
        }
        _ => return Err("only int64 and float32 supported via HTTP".into()),
    };
    Ok(tensor.reshape(&t.shape).to_device(tch::Device::Cuda(0)))
}

async fn eval_step(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<TrainStepHttp>,
) -> Result<Json<EvalStepResponse>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let input_ids = decode_tensor(&req.input_ids).map_err(|e| err_resp(&e))?;
    let target_mask = decode_tensor(&req.target_mask).map_err(|e| err_resp(&e))?;
    let attention_mask = decode_tensor(&req.attention_mask).map_err(|e| err_resp(&e))?;
    let s = session.lock().await;
    let result = s
        .eval_step(TrainInput {
            input_ids,
            target_mask,
            attention_mask,
        })
        .map_err(|e| err_resp(&e.to_string()))?;
    Ok(Json(EvalStepResponse { loss: result.loss }))
}
#[derive(Serialize)]
struct EvalStepResponse {
    loss: f64,
}

#[derive(Deserialize)]
struct CheckpointHttp {
    path: String,
}
#[derive(Serialize)]
struct CheckpointResponse {
    step: u64,
    loss: f64,
    path: String,
}

async fn save_checkpoint(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<CheckpointHttp>,
) -> Result<Json<CheckpointResponse>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let s = session.lock().await;
    let (step, loss) = s
        .save_checkpoint(&req.path)
        .map_err(|e| err_resp(&e.to_string()))?;
    Ok(Json(CheckpointResponse {
        step,
        loss,
        path: req.path,
    }))
}

async fn load_checkpoint(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<CheckpointHttp>,
) -> Result<Json<CheckpointResponse>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let mut s = session.lock().await;
    let (step, loss) = s
        .load_checkpoint(&req.path)
        .map_err(|e| err_resp(&e.to_string()))?;
    Ok(Json(CheckpointResponse {
        step,
        loss,
        path: req.path,
    }))
}

#[derive(Deserialize)]
struct ExportHttp {
    path: String,
}
#[derive(Serialize)]
struct ExportResponse {
    path: String,
    param_count: usize,
}

async fn export_adapter(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<ExportHttp>,
) -> Result<Json<ExportResponse>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let s = session.lock().await;
    let count = s
        .export_adapter(&req.path)
        .map_err(|e| err_resp(&e.to_string()))?;
    Ok(Json(ExportResponse {
        path: req.path,
        param_count: count,
    }))
}

async fn get_status(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<StatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let s = session.lock().await;
    let status = s.status();
    Ok(Json(StatusResponse {
        state: status.state,
        step: status.step,
        last_loss: status.last_loss,
        model_path: status.model_path,
    }))
}

#[derive(Deserialize)]
struct AddLoRAHttp {
    rank: i64,
    alpha: f64,
    target_layers: Vec<i64>,
    target_modules: String,
}
#[derive(Serialize)]
struct AddLoRAResponse {
    adapter_id: i64,
}

async fn add_lora(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<AddLoRAHttp>,
) -> Result<Json<AddLoRAResponse>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let mut s = session.lock().await;
    let adapter_id = s
        .add_lora(crate::session::AddLoRARequest {
            rank: req.rank,
            alpha: req.alpha,
            target_layers: req.target_layers,
            target_modules: req.target_modules,
        })
        .map_err(|e| err_resp(&e.to_string()))?;
    Ok(Json(AddLoRAResponse { adapter_id }))
}

#[derive(Deserialize)]
struct RemoveLoRAHttp {
    adapter_id: i64,
}
#[derive(Serialize)]
struct RemoveLoRAResponse {
    removed: bool,
}

async fn remove_lora(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<RemoveLoRAHttp>,
) -> Result<Json<RemoveLoRAResponse>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let mut s = session.lock().await;
    let removed = s
        .remove_lora(req.adapter_id)
        .map_err(|e| err_resp(&e.to_string()))?;
    Ok(Json(RemoveLoRAResponse { removed }))
}

async fn list_lora(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Vec<i64>>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let s = session.lock().await;
    Ok(Json(s.list_lora()))
}

#[derive(Serialize)]
struct StatusResponse {
    state: String,
    step: u64,
    last_loss: f64,
    model_path: String,
}

async fn stream_metrics(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let s = session.lock().await;
    let metrics = s.get_metrics();
    drop(s);

    let stream = stream::iter(metrics.into_iter().map(|m| {
        Ok(Event::default().json_data(StepMetricJson {
            step: m.step,
            loss: m.loss,
            lr: m.lr,
            mem_gb: m.mem_gb,
            timestamp_unix: m.timestamp_unix,
        }).unwrap_or_default())
    }));

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

#[derive(Serialize)]
struct StepMetricJson {
    step: u64,
    loss: f64,
    lr: f64,
    mem_gb: f64,
    timestamp_unix: i64,
}
