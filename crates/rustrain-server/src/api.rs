//! HTTP API (axum) — RESTful endpoints for training session management.

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
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

/// EP mode: HTTP server dispatches to workers via IPC coordinator.
pub struct EpAppState {
    pub coordinator: Arc<crate::ep::EpCoordinator>,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/sessions", post(create_session))
        .route("/v1/sessions", get(list_sessions))
        .route("/v1/sessions/{id}", axum::routing::delete(delete_session))
        .route("/v1/sessions/{id}/load_model", post(load_model))
        .route("/v1/sessions/{id}/load_dataset", post(load_dataset))
        .route("/v1/sessions/{id}/init_lora", post(init_lora))
        .route("/v1/sessions/{id}/train_step", post(train_step))
        .route("/v1/sessions/{id}/eval_step", post(eval_step))
        .route("/v1/sessions/{id}/save_checkpoint", post(save_checkpoint))
        .route("/v1/sessions/{id}/load_checkpoint", post(load_checkpoint))
        .route("/v1/sessions/{id}/export_adapter", post(export_adapter))
        .route("/v1/sessions/{id}/import_adapter", post(import_adapter))
        .route("/v1/sessions/{id}/add_lora", post(add_lora))
        .route("/v1/sessions/{id}/remove_lora", post(remove_lora))
        .route("/v1/sessions/{id}/list_lora", get(list_lora))
        .route("/v1/sessions/{id}/metrics", get(stream_metrics))
        .route("/v1/sessions/{id}/status", get(get_status))
        .with_state(state)
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn err_resp(msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
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

async fn delete_session(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    state
        .manager
        .delete_session(&id)
        .await
        .map_err(|e| err_resp(&e))?;
    Ok(Json(serde_json::json!({"deleted": id})))
}

#[derive(Deserialize)]
struct CreateSessionRequest {
    session_id: String,
}
#[derive(Serialize)]
struct CreateSessionResponse {
    session_id: String,
}

async fn list_sessions(State(state): State<Arc<AppState>>) -> Json<Vec<String>> {
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
    alpha: f64,
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
    let input_ids = decode_tensor(&req.input_ids).map_err(|e| err_resp(&e))?;
    let target_mask = decode_tensor(&req.target_mask).map_err(|e| err_resp(&e))?;
    let attention_mask = decode_tensor(&req.attention_mask).map_err(|e| err_resp(&e))?;

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

#[derive(Deserialize)]
struct TrainMultiLoraHttp {
    input_ids: TensorHttp,
    target_mask: TensorHttp,
    attention_mask: TensorHttp,
    n_total: i32,
    lora_rank: i32,
    #[serde(default)]
    adapter_ids: Vec<i64>,
}
#[derive(Serialize)]
struct TrainStepResponse {
    loss: f64,
    step: u64,
}

/// Decode a base64-encoded int64 tensor to Vec<i64> (for EP IPC).
fn decode_int64_vec(t: &TensorHttp) -> Result<Vec<i64>, String> {
    use base64::{Engine, engine::general_purpose};
    let bytes = general_purpose::STANDARD
        .decode(&t.data)
        .map_err(|e| format!("base64 decode: {e}"))?;
    let values = bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect::<Vec<_>>();
    let expected = t
        .shape
        .iter()
        .try_fold(1usize, |acc, dim| {
            usize::try_from(*dim)
                .ok()
                .and_then(|dim| acc.checked_mul(dim))
        })
        .ok_or_else(|| format!("invalid tensor shape {:?}", t.shape))?;
    if expected != values.len() {
        return Err(format!(
            "tensor shape {:?} expects {} int64 values, got {}",
            t.shape,
            expected,
            values.len()
        ));
    }
    Ok(values)
}

fn validate_train_http_shapes(
    input_ids: &TensorHttp,
    target_mask: &TensorHttp,
    attention_mask: &TensorHttp,
) -> Result<(usize, usize), String> {
    if input_ids.shape.len() != 1 && input_ids.shape.len() != 2 {
        return Err(format!(
            "input_ids must have shape [seq] or [batch, seq], got {:?}",
            input_ids.shape
        ));
    }
    if target_mask.shape != input_ids.shape || attention_mask.shape != input_ids.shape {
        return Err(format!(
            "masks must have the same shape as input_ids: input={:?} target={:?} attention={:?}",
            input_ids.shape, target_mask.shape, attention_mask.shape
        ));
    }
    let seq_len = *input_ids
        .shape
        .last()
        .ok_or_else(|| "input_ids shape is empty".to_string())?;
    if seq_len <= 0 {
        return Err(format!("sequence length must be positive, got {seq_len}"));
    }
    let batch_size = if input_ids.shape.len() == 2 {
        input_ids.shape[0]
    } else {
        1
    };
    if batch_size <= 0 {
        return Err(format!("batch size must be positive, got {batch_size}"));
    }
    Ok((
        usize::try_from(batch_size).map_err(|_| format!("invalid batch size {batch_size}"))?,
        usize::try_from(seq_len).map_err(|_| format!("invalid sequence length {seq_len}"))?,
    ))
}

fn validate_multi_lora_http_shapes(
    input_ids: &TensorHttp,
    target_mask: &TensorHttp,
    attention_mask: &TensorHttp,
    n_total: i32,
) -> Result<(usize, usize), String> {
    if n_total <= 0 {
        return Err(format!("n_total must be positive, got {n_total}"));
    }
    let (batch_size, seq_len) = validate_train_http_shapes(input_ids, target_mask, attention_mask)?;
    let n_total = n_total as usize;
    if batch_size != 1 && batch_size % n_total != 0 {
        return Err(format!(
            "multi-LoRA batch must be 1 or a positive multiple of n_total={n_total}, got {batch_size}"
        ));
    }
    Ok((batch_size, seq_len))
}

fn decode_tensor(t: &TensorHttp) -> Result<tch::Tensor, String> {
    use base64::{Engine, engine::general_purpose};
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
    let local_rank = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    Ok(tensor
        .reshape(&t.shape)
        .to_device(tch::Device::Cuda(local_rank)))
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
    #[serde(default)]
    adapter_id: Option<i64>,
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
        .export_adapter(&req.path, req.adapter_id)
        .map_err(|e| err_resp(&e.to_string()))?;
    Ok(Json(ExportResponse {
        path: req.path,
        param_count: count,
    }))
}

#[derive(Deserialize)]
struct ImportHttp {
    path: String,
}

#[derive(Serialize)]
struct ImportResponse {
    adapter_id: i64,
}

async fn import_adapter(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<ImportHttp>,
) -> Result<Json<ImportResponse>, (StatusCode, Json<ErrorResponse>)> {
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let mut s = session.lock().await;
    let adapter_id = s
        .import_adapter(&req.path)
        .map_err(|e| err_resp(&e.to_string()))?;
    Ok(Json(ImportResponse { adapter_id }))
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
        Ok(Event::default()
            .json_data(StepMetricJson {
                step: m.step,
                loss: m.loss,
                lr: m.lr,
                mem_gb: m.mem_gb,
                timestamp_unix: m.timestamp_unix,
            })
            .unwrap_or_default())
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

// ──────────────────────────────────────────────────────────────────────
// EP Mode: HTTP handlers that dispatch to workers via IPC
// ──────────────────────────────────────────────────────────────────────

pub fn ep_router(state: Arc<EpAppState>) -> Router {
    Router::new()
        .route("/v1/sessions", post(ep_create_session))
        .route(
            "/v1/sessions/{id}",
            axum::routing::delete(ep_delete_session),
        )
        .route("/v1/sessions/{id}/load_model", post(ep_load_model))
        .route("/v1/sessions/{id}/load_dataset", post(ep_load_dataset))
        .route("/v1/sessions/{id}/init_lora", post(ep_init_lora))
        .route("/v1/sessions/{id}/train_step", post(ep_train_step))
        .route("/v1/sessions/{id}/train_multi", post(ep_train_multi_lora))
        .route("/v1/sessions/{id}/eval_step", post(ep_eval_step))
        .route(
            "/v1/sessions/{id}/save_checkpoint",
            post(ep_save_checkpoint),
        )
        .route(
            "/v1/sessions/{id}/load_checkpoint",
            post(ep_load_checkpoint),
        )
        .route("/v1/sessions/{id}/add_lora", post(ep_add_lora))
        .route("/v1/sessions/{id}/batch_add_lora", post(ep_batch_add_lora))
        .route("/v1/sessions/{id}/remove_lora", post(ep_remove_lora))
        .route("/v1/sessions/{id}/list_lora", get(ep_list_lora))
        .route("/v1/sessions/{id}/export_adapter", post(ep_export_adapter))
        .route("/v1/sessions/{id}/status", get(ep_get_status))
        .route("/v1/health", get(ep_health))
        .with_state(state)
}

async fn ep_health(
    State(state): State<Arc<EpAppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    if state.coordinator.is_healthy() {
        (
            StatusCode::OK,
            Json(serde_json::json!({"status": "ok", "mode": "ep"})),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "error", "mode": "ep"})),
        )
    }
}

async fn ep_create_session(
    State(state): State<Arc<EpAppState>>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::CreateSession {
        session_id: req.session_id.clone(),
    };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::Ok => Ok(Json(CreateSessionResponse {
            session_id: req.session_id,
        })),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_delete_session(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::DeleteSession {
        session_id: id.clone(),
    };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::Ok => Ok(Json(serde_json::json!({"deleted": id}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_load_model(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
    Json(req): Json<LoadModelHttp>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::LoadModel {
        session_id: id,
        model_path: req.model_path,
        config_toml: req.config_toml,
    };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::Ok => Ok(Json(serde_json::json!({"loaded": true}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_load_dataset(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
    Json(req): Json<LoadDatasetHttp>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::LoadDataset {
        session_id: id,
        jsonl_path: req.jsonl_path,
        seq_len: req.seq_len,
    };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::Count(n) => Ok(Json(serde_json::json!({"samples": n}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_init_lora(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
    Json(req): Json<InitLoRAHttp>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::InitLora {
        session_id: id,
        rank: req.rank,
        alpha: req.alpha,
        target_layers: req.target_layers,
        target_modules: req.target_modules,
        lr: req.lr,
        beta1: req.beta1,
        beta2: req.beta2,
        eps: req.eps,
    };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::Count(n) => Ok(Json(serde_json::json!({"lora_count": n}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_train_step(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
    Json(req): Json<TrainStepHttp>,
) -> Result<Json<TrainStepResponse>, (StatusCode, Json<ErrorResponse>)> {
    let (batch_size, seq_len) =
        validate_train_http_shapes(&req.input_ids, &req.target_mask, &req.attention_mask)
            .map_err(|e| err_resp(&e))?;
    let input_ids = decode_int64_vec(&req.input_ids).map_err(|e| err_resp(&e))?;
    let target_mask = decode_int64_vec(&req.target_mask).map_err(|e| err_resp(&e))?;
    let attention_mask = decode_int64_vec(&req.attention_mask).map_err(|e| err_resp(&e))?;

    let cmd = rustrain_ipc::EpCommand::TrainStep {
        session_id: id,
        input_ids,
        target_mask,
        attention_mask,
        batch_size,
        seq_len,
    };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::Train { loss, step } => {
            Ok(Json(TrainStepResponse { loss, step }))
        }
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_eval_step(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
    Json(req): Json<TrainStepHttp>,
) -> Result<Json<EvalStepResponse>, (StatusCode, Json<ErrorResponse>)> {
    let input_ids = decode_int64_vec(&req.input_ids).map_err(|e| err_resp(&e))?;
    let target_mask = decode_int64_vec(&req.target_mask).map_err(|e| err_resp(&e))?;
    let attention_mask = decode_int64_vec(&req.attention_mask).map_err(|e| err_resp(&e))?;
    let seq_len = input_ids.len();

    let cmd = rustrain_ipc::EpCommand::EvalStep {
        session_id: id,
        input_ids,
        target_mask,
        attention_mask,
        seq_len,
    };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::Loss(loss) => Ok(Json(EvalStepResponse { loss })),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_train_multi_lora(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
    Json(req): Json<TrainMultiLoraHttp>,
) -> Result<Json<TrainStepResponse>, (StatusCode, Json<ErrorResponse>)> {
    let input_ids = decode_int64_vec(&req.input_ids).map_err(|e| err_resp(&e))?;
    let target_mask = decode_int64_vec(&req.target_mask).map_err(|e| err_resp(&e))?;
    let attention_mask = decode_int64_vec(&req.attention_mask).map_err(|e| err_resp(&e))?;
    let (batch_size, seq_len) = validate_multi_lora_http_shapes(
        &req.input_ids,
        &req.target_mask,
        &req.attention_mask,
        req.n_total,
    )
    .map_err(|e| err_resp(&e))?;
    if !req.adapter_ids.is_empty() {
        if req.adapter_ids.len() != req.n_total as usize {
            return Err(err_resp(&format!(
                "adapter_ids length {} must match n_total={}",
                req.adapter_ids.len(),
                req.n_total
            )));
        }
        if req.adapter_ids.iter().any(|id| *id <= 0) {
            return Err(err_resp("adapter_ids must contain only positive IDs"));
        }
    }

    let cmd = rustrain_ipc::EpCommand::TrainMultiLora {
        session_id: id,
        input_ids,
        target_mask,
        attention_mask,
        batch_size,
        seq_len,
        n_total: req.n_total,
        lora_rank: req.lora_rank,
        adapter_ids: req.adapter_ids,
    };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::Train { loss, step } => {
            Ok(Json(TrainStepResponse { loss, step }))
        }
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

#[cfg(test)]
mod tensor_http_shape_tests {
    use super::{TensorHttp, validate_multi_lora_http_shapes, validate_train_http_shapes};

    fn tensor(shape: &[i64]) -> TensorHttp {
        TensorHttp {
            data: String::new(),
            shape: shape.to_vec(),
            dtype: "int64".into(),
        }
    }

    #[test]
    fn train_shape_keeps_batch_and_sequence_dimensions() {
        let input = tensor(&[8, 128]);
        let target = tensor(&[8, 128]);
        let attention = tensor(&[8, 128]);

        assert_eq!(
            validate_train_http_shapes(&input, &target, &attention).unwrap(),
            (8, 128)
        );
    }

    #[test]
    fn train_shape_rejects_mismatched_masks_and_non_positive_dimensions() {
        let input = tensor(&[4, 16]);
        assert!(validate_train_http_shapes(&input, &tensor(&[2, 16]), &tensor(&[4, 16])).is_err());
        assert!(
            validate_train_http_shapes(&tensor(&[0, 16]), &tensor(&[0, 16]), &tensor(&[0, 16]))
                .is_err()
        );
        assert!(
            validate_train_http_shapes(&tensor(&[4, 0]), &tensor(&[4, 0]), &tensor(&[4, 0]))
                .is_err()
        );
    }

    #[test]
    fn multi_lora_shape_accepts_replica_and_global_row_layouts() {
        for rows in [1, 3, 6] {
            let input = tensor(&[rows, 32]);
            assert_eq!(
                validate_multi_lora_http_shapes(
                    &input,
                    &tensor(&[rows, 32]),
                    &tensor(&[rows, 32]),
                    3
                )
                .unwrap(),
                (rows as usize, 32)
            );
        }
        let invalid = tensor(&[5, 32]);
        assert!(
            validate_multi_lora_http_shapes(&invalid, &tensor(&[5, 32]), &tensor(&[5, 32]), 3)
                .is_err()
        );
    }
}

async fn ep_add_lora(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
    Json(req): Json<AddLoRAHttp>,
) -> Result<Json<AddLoRAResponse>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::AddLora {
        session_id: id,
        rank: req.rank,
        alpha: req.alpha,
        target_layers: req.target_layers,
        target_modules: req.target_modules,
    };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::AdapterId(id) => Ok(Json(AddLoRAResponse { adapter_id: id })),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

#[derive(Deserialize)]
struct BatchAddLoRAHttp {
    count: i32,
    rank: i64,
    alpha: f64,
    target_layers: Vec<i64>,
    target_modules: String,
}

async fn ep_batch_add_lora(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
    Json(req): Json<BatchAddLoRAHttp>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::BatchAddLora {
        session_id: id,
        count: req.count,
        rank: req.rank,
        alpha: req.alpha,
        target_layers: req.target_layers,
        target_modules: req.target_modules,
    };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::Count(n) => Ok(Json(serde_json::json!({"count": n}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_remove_lora(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
    Json(req): Json<RemoveLoRAHttp>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::RemoveLora {
        session_id: id,
        adapter_id: req.adapter_id,
    };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::Ok => Ok(Json(serde_json::json!({"removed": true}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_list_lora(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::ListLora { session_id: id };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::AdapterIds(ids) => Ok(Json(serde_json::json!({"adapters": ids}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_export_adapter(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
    Json(req): Json<ExportHttp>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let generation = state
        .coordinator
        .next_generation("export")
        .map_err(|error| err_resp(&error))?;
    let cmd = rustrain_ipc::EpCommand::ExportAdapter {
        session_id: id,
        path: req.path,
        adapter_id: req.adapter_id,
        generation,
    };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::Count(n) => Ok(Json(serde_json::json!({"exported": n}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_save_checkpoint(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
    Json(req): Json<CheckpointHttp>,
) -> Result<Json<CheckpointResponse>, (StatusCode, Json<ErrorResponse>)> {
    let path = req.path;
    let response_path = path.clone();
    let coordinator = Arc::clone(&state.coordinator);
    let (step, loss) =
        tokio::task::spawn_blocking(move || coordinator.coordinated_save_checkpoint(&id, &path))
            .await
            .map_err(|error| err_resp(&format!("checkpoint save task failed: {error}")))?
            .map_err(|error| err_resp(&error))?;
    Ok(Json(CheckpointResponse {
        step,
        loss,
        path: response_path,
    }))
}

async fn ep_load_checkpoint(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
    Json(req): Json<CheckpointHttp>,
) -> Result<Json<CheckpointResponse>, (StatusCode, Json<ErrorResponse>)> {
    let path = req.path;
    let response_path = path.clone();
    let coordinator = Arc::clone(&state.coordinator);
    let (step, loss) =
        tokio::task::spawn_blocking(move || coordinator.coordinated_load_checkpoint(&id, &path))
            .await
            .map_err(|error| err_resp(&format!("checkpoint load task failed: {error}")))?
            .map_err(|error| err_resp(&error))?;
    Ok(Json(CheckpointResponse {
        step,
        loss,
        path: response_path,
    }))
}

async fn ep_get_status(
    State(state): State<Arc<EpAppState>>,
    Path(id): Path<String>,
) -> Result<Json<StatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::Status { session_id: id };
    match state.coordinator.dispatch(&cmd) {
        rustrain_ipc::EpResult::Status {
            state,
            step,
            last_loss,
            model_path,
        } => Ok(Json(StatusResponse {
            state,
            step,
            last_loss,
            model_path,
        })),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}
