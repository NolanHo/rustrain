//! HTTP API (axum) — RESTful endpoints for training session management.

use axum::{
    body::Bytes,
    extract::DefaultBodyLimit,
    extract::Request,
    extract::{Path, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::sse::{Event, KeepAlive, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{oneshot, Semaphore};
use tokio_stream::StreamExt;

use crate::ep_dispatch::{configured_queue_capacity, EpDispatchScheduleError, EpDispatchScheduler};
use crate::metrics::StepMetric;
use crate::session::{InitLoRARequest, SessLoadDatasetRequest, SessLoadModelRequest, TrainInput};
use crate::state::SessionManager;

pub struct AppState {
    pub manager: Arc<SessionManager>,
}

/// EP mode: HTTP server dispatches to workers via IPC coordinator.
pub struct EpAppState {
    pub coordinator: Arc<crate::ep::EpCoordinator>,
    pub world_size: usize,
}

struct EpRouterState {
    coordinator: Arc<crate::ep::EpCoordinator>,
    world_size: usize,
    dispatcher: EpDispatchScheduler,
    dispatch_submission: Mutex<()>,
    multi_lora_batcher: MultiLoraBatcher,
}

const DEFAULT_MULTI_LORA_BATCH_WINDOW_US: u64 = 2_000;
const HARD_MAX_MULTI_LORA_BATCH_WINDOW_US: u64 = 100_000;
const DEFAULT_MULTI_LORA_BATCH_REQUESTS: usize = 16;
const HARD_MAX_MULTI_LORA_BATCH_REQUESTS: usize = 4_096;
const DEFAULT_MULTI_LORA_BATCH_ADAPTERS: usize = 64;
const DEFAULT_MULTI_LORA_BATCH_RANK_WORK: usize = 32_768;
const HARD_MAX_MULTI_LORA_BATCH_RANK_WORK: usize = 4_194_304;
// AdapterLoss is JSON-encoded into a fixed 256 KiB IPC result slot. Keep a
// conservative margin for worst-case i64/f64 strings and the result envelope.
const HARD_MAX_MULTI_LORA_BATCH_ADAPTERS: usize = 2_048;
const MULTI_LORA_CAPABILITY_HEADER: &str = "x-rustrain-multi-lora-capability";
const MULTI_LORA_CAPABILITY_V1: &str = "v1";
const MULTI_LORA_RESPONSE_CAPABILITIES: &[&str] = &[
    "per_adapter_loss_v1",
    "optimizer_steps_v1",
    "coalesced_loss_scope_v1",
];

#[derive(Clone, Copy)]
struct MultiLoraBatchConfig {
    window: Duration,
    max_requests: usize,
    max_adapters: usize,
    max_rank_work: usize,
    max_payload_bytes: usize,
}

impl MultiLoraBatchConfig {
    fn from_env() -> Self {
        let window_us = configured_positive_us(
            "RUSTRAIN_MULTI_LORA_BATCH_WINDOW_US",
            DEFAULT_MULTI_LORA_BATCH_WINDOW_US,
            HARD_MAX_MULTI_LORA_BATCH_WINDOW_US,
        );
        let max_requests = configured_positive_usize(
            "RUSTRAIN_MULTI_LORA_BATCH_MAX_REQUESTS",
            DEFAULT_MULTI_LORA_BATCH_REQUESTS,
            HARD_MAX_MULTI_LORA_BATCH_REQUESTS,
        );
        let max_adapters = configured_positive_usize(
            "RUSTRAIN_MULTI_LORA_BATCH_MAX_ADAPTERS",
            DEFAULT_MULTI_LORA_BATCH_ADAPTERS,
            HARD_MAX_MULTI_LORA_BATCH_ADAPTERS,
        );
        let max_rank_work = configured_positive_usize(
            "RUSTRAIN_MULTI_LORA_BATCH_MAX_RANK_WORK",
            DEFAULT_MULTI_LORA_BATCH_RANK_WORK,
            HARD_MAX_MULTI_LORA_BATCH_RANK_WORK,
        );
        let slab_bytes = configured_positive_usize(
            "RUSTRAIN_EP_TENSOR_SLAB_BYTES",
            rustrain_ipc::DEFAULT_TENSOR_SLAB_BYTES,
            usize::MAX,
        );
        let max_payload_bytes = configured_positive_usize(
            "RUSTRAIN_MULTI_LORA_BATCH_MAX_BYTES",
            slab_bytes,
            slab_bytes,
        );
        Self {
            window: Duration::from_micros(window_us),
            max_requests,
            max_adapters,
            max_rank_work,
            max_payload_bytes,
        }
    }
}

fn configured_positive_us(name: &str, default: u64, hard_max: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
        .min(hard_max)
}

fn configured_positive_usize(name: &str, default: usize, hard_max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
        .min(hard_max)
}

struct MultiLoraDispatchRequest {
    session_id: String,
    tensors: rustrain_ipc::TensorSlabRef,
    payload: Vec<u8>,
    n_total: usize,
    lora_rank: i32,
    adapter_ids: Vec<i64>,
    expected_steps: Vec<u64>,
    source_count: usize,
    normalized_payload_bytes: usize,
    allow_aggregate_loss: bool,
    response: oneshot::Sender<MultiLoraDispatchOutcome>,
}

impl MultiLoraDispatchRequest {
    fn coalescible(&self) -> bool {
        self.allow_aggregate_loss && !self.adapter_ids.is_empty()
    }
}

struct MultiLoraDispatchOutcome {
    result: rustrain_ipc::EpResult,
    request_count: usize,
}

struct MultiLoraResponseTarget {
    response: oneshot::Sender<MultiLoraDispatchOutcome>,
    adapter_range: std::ops::Range<usize>,
}

fn project_multi_lora_result(
    result: &rustrain_ipc::EpResult,
    adapter_range: std::ops::Range<usize>,
) -> rustrain_ipc::EpResult {
    match result {
        rustrain_ipc::EpResult::MultiLoraTrain {
            loss,
            step,
            adapter_losses,
            ..
        } if adapter_losses.is_empty() => rustrain_ipc::EpResult::Train {
            loss: *loss,
            step: *step,
        },
        rustrain_ipc::EpResult::MultiLoraTrain {
            loss,
            step,
            adapter_losses,
            adapter_steps,
        } if adapter_range.end <= adapter_losses.len()
            && (adapter_steps.is_empty() || adapter_range.end <= adapter_steps.len()) =>
        {
            rustrain_ipc::EpResult::MultiLoraTrain {
                loss: *loss,
                step: *step,
                adapter_losses: adapter_losses[adapter_range.clone()].to_vec(),
                adapter_steps: if adapter_steps.is_empty() {
                    Vec::new()
                } else {
                    adapter_steps[adapter_range].to_vec()
                },
            }
        }
        rustrain_ipc::EpResult::MultiLoraTrain {
            adapter_losses,
            adapter_steps,
            ..
        } => rustrain_ipc::EpResult::Error(format!(
            "native adapter result counts (losses={}, steps={}) do not cover coalesced range {:?}",
            adapter_losses.len(),
            adapter_steps.len(),
            adapter_range
        )),
        _ => result.clone(),
    }
}

struct MultiLoraWindow {
    session_id: String,
    seq_len: usize,
    source_count: usize,
    adapter_ids: HashSet<i64>,
    adapter_count: usize,
    max_lora_rank: usize,
    rank_work: usize,
    payload_bytes: usize,
    coalescible: bool,
    validates_steps: bool,
    requests: Vec<MultiLoraDispatchRequest>,
}

impl MultiLoraWindow {
    fn new(request: MultiLoraDispatchRequest) -> Self {
        let adapter_ids = request.adapter_ids.iter().copied().collect();
        Self {
            session_id: request.session_id.clone(),
            seq_len: request.tensors.seq_len,
            source_count: request.source_count,
            adapter_ids,
            adapter_count: request.n_total,
            max_lora_rank: request.lora_rank.max(0) as usize,
            rank_work: request
                .n_total
                .saturating_mul(request.lora_rank.max(0) as usize),
            payload_bytes: request.normalized_payload_bytes,
            coalescible: request.coalescible(),
            validates_steps: !request.expected_steps.is_empty(),
            requests: vec![request],
        }
    }

    fn can_accept(&self, request: &MultiLoraDispatchRequest, config: MultiLoraBatchConfig) -> bool {
        if !self.coalescible
            || !request.coalescible()
            || self.session_id != request.session_id
            || self.seq_len != request.tensors.seq_len
            || self.source_count != request.source_count
            || self.validates_steps != !request.expected_steps.is_empty()
            || self.requests.len() >= config.max_requests
            || self.adapter_count.saturating_add(request.n_total) > config.max_adapters
            || request.lora_rank <= 0
            || self
                .payload_bytes
                .saturating_add(request.normalized_payload_bytes)
                > config.max_payload_bytes
        {
            return false;
        }
        let projected_count = self.adapter_count.saturating_add(request.n_total);
        let projected_max_rank = self.max_lora_rank.max(request.lora_rank as usize);
        let Some(projected_rank_work) = projected_count.checked_mul(projected_max_rank) else {
            return false;
        };
        if projected_rank_work > config.max_rank_work {
            return false;
        }
        request
            .adapter_ids
            .iter()
            .all(|adapter_id| !self.adapter_ids.contains(adapter_id))
    }

    fn push(&mut self, request: MultiLoraDispatchRequest) {
        self.adapter_count += request.n_total;
        self.max_lora_rank = self.max_lora_rank.max(request.lora_rank as usize);
        self.rank_work = self.adapter_count.saturating_mul(self.max_lora_rank);
        self.payload_bytes += request.normalized_payload_bytes;
        self.adapter_ids.extend(request.adapter_ids.iter().copied());
        self.requests.push(request);
    }

    fn at_capacity(&self, config: MultiLoraBatchConfig) -> bool {
        self.requests.len() >= config.max_requests
            || self.adapter_count >= config.max_adapters
            || self.rank_work >= config.max_rank_work
            || self.payload_bytes >= config.max_payload_bytes
    }
}

#[derive(Default)]
struct MultiLoraBatchState {
    next_window_id: u64,
    current_window_id: Option<u64>,
    windows: HashMap<u64, MultiLoraWindow>,
}

#[derive(Clone)]
struct MultiLoraBatcher {
    state: Arc<Mutex<MultiLoraBatchState>>,
    window_wakeup: Arc<Condvar>,
    config: MultiLoraBatchConfig,
}

impl MultiLoraBatcher {
    fn new(config: MultiLoraBatchConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(MultiLoraBatchState::default())),
            window_wakeup: Arc::new(Condvar::new()),
            config,
        }
    }

    fn seal_current(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.current_window_id = None;
        self.window_wakeup.notify_all();
    }

    /// Wait for a coalescing window to be sealed or for its normal deadline.
    /// The window ID predicate makes spurious notifications harmless and
    /// treats replacement by a newer window as sealing the old one. The
    /// scheduler remains single-consumer, so this only changes when the
    /// existing FIFO job becomes dispatchable.
    fn wait_for_window(&self, window_id: u64, wait: Duration) {
        if wait.is_zero() {
            return;
        }
        let deadline = Instant::now() + wait;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.current_window_id == Some(window_id) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let (next_state, timeout) = self
                .window_wakeup
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next_state;
            if timeout.timed_out() {
                break;
            }
        }
        if state.current_window_id == Some(window_id) {
            state.current_window_id = None;
        }
    }

    fn seal_window_if_at_capacity(&self, state: &mut MultiLoraBatchState, window_id: u64) {
        if state.current_window_id == Some(window_id)
            && state
                .windows
                .get(&window_id)
                .is_some_and(|window| window.at_capacity(self.config))
        {
            state.current_window_id = None;
            self.window_wakeup.notify_all();
        }
    }

    fn submit(
        &self,
        scheduler: &EpDispatchScheduler,
        coordinator: Arc<crate::ep::EpCoordinator>,
        request: MultiLoraDispatchRequest,
    ) -> Result<(), EpDispatchScheduleError> {
        if request.n_total > self.config.max_adapters
            || request.normalized_payload_bytes > self.config.max_payload_bytes
            || request.lora_rank <= 0
            || request
                .n_total
                .checked_mul(request.lora_rank as usize)
                .map_or(true, |rank_work| rank_work > self.config.max_rank_work)
        {
            return Err(EpDispatchScheduleError::QueueFull);
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(window_id) = state.current_window_id {
            if state
                .windows
                .get(&window_id)
                .is_some_and(|window| window.can_accept(&request, self.config))
            {
                state.windows.get_mut(&window_id).unwrap().push(request);
                self.seal_window_if_at_capacity(&mut state, window_id);
                return Ok(());
            }
            state.current_window_id = None;
            self.window_wakeup.notify_all();
        }

        let window_id = state.next_window_id;
        state.next_window_id = state.next_window_id.wrapping_add(1);
        let coalescible = request.coalescible();
        state
            .windows
            .insert(window_id, MultiLoraWindow::new(request));
        if coalescible {
            state.current_window_id = Some(window_id);
            self.seal_window_if_at_capacity(&mut state, window_id);
        }

        let batcher = self.clone();
        let wait = if coalescible {
            self.config.window
        } else {
            Duration::ZERO
        };
        let scheduled = scheduler.submit(move || {
            batcher.wait_for_window(window_id, wait);
            batcher.execute_window(window_id, &coordinator);
        });
        if let Err(error) = scheduled {
            state.windows.remove(&window_id);
            if state.current_window_id == Some(window_id) {
                state.current_window_id = None;
                self.window_wakeup.notify_all();
            }
            return Err(error);
        }
        drop(scheduled);
        Ok(())
    }

    fn execute_window(&self, window_id: u64, coordinator: &crate::ep::EpCoordinator) {
        let window = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.current_window_id == Some(window_id) {
                state.current_window_id = None;
            }
            state.windows.remove(&window_id)
        };
        let Some(window) = window else {
            return;
        };
        let (command, payload, responses) = match build_multi_lora_window(window) {
            Ok(result) => result,
            Err((error, responses)) => {
                let request_count = responses.len();
                for target in responses {
                    let _ = target.response.send(MultiLoraDispatchOutcome {
                        result: rustrain_ipc::EpResult::Error(error.clone()),
                        request_count,
                    });
                }
                return;
            }
        };
        if payload.len() > self.config.max_payload_bytes {
            let request_count = responses.len();
            for target in responses {
                let _ = target.response.send(MultiLoraDispatchOutcome {
                    result: rustrain_ipc::EpResult::Error(format!(
                        "coalesced tensor payload {} exceeds configured limit {}",
                        payload.len(),
                        self.config.max_payload_bytes
                    )),
                    request_count,
                });
            }
            return;
        }
        tracing::debug!(
            requests = responses.len(),
            payload_bytes = payload.len(),
            "dispatching coalesced multi-LoRA training batch"
        );
        let result = if coordinator.is_healthy() {
            coordinator.dispatch_with_slab(&command, &payload)
        } else {
            rustrain_ipc::EpResult::Error("EP coordinator is unavailable".to_string())
        };
        let request_count = responses.len();
        for target in responses {
            let projected = project_multi_lora_result(&result, target.adapter_range);
            let _ = target.response.send(MultiLoraDispatchOutcome {
                result: projected,
                request_count,
            });
        }
    }
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
        .route("/v1/sessions/{id}/eval_multi_lora", post(eval_multi_lora))
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
struct EvalMultiLoraHttp {
    input_ids: TensorHttp,
    target_mask: TensorHttp,
    attention_mask: TensorHttp,
    adapter_ids: Vec<i64>,
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
    #[serde(default)]
    expected_steps: Vec<u64>,
    #[serde(default)]
    allow_aggregate_loss: bool,
}

const MULTI_LORA_BINARY_MAGIC: [u8; 4] = *b"RLM1";
const MULTI_LORA_BINARY_VERSION: u16 = 1;
const MULTI_LORA_BINARY_HEADER_BYTES: usize = 56;

struct BinaryMultiLoraRequest {
    tensors: rustrain_ipc::TensorSlabRef,
    payload: Vec<u8>,
    batch_size: usize,
    seq_len: usize,
    n_total: i32,
    lora_rank: i32,
    adapter_ids: Vec<i64>,
    expected_steps: Vec<u64>,
}

fn read_binary_u16(bytes: &[u8], offset: &mut usize) -> Result<u16, String> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| "binary header offset overflowed".to_string())?;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| "binary multi-LoRA header is truncated".to_string())?;
    *offset = end;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_binary_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| "binary header offset overflowed".to_string())?;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| "binary multi-LoRA header is truncated".to_string())?;
    *offset = end;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn read_binary_i32(bytes: &[u8], offset: &mut usize) -> Result<i32, String> {
    Ok(read_binary_u32(bytes, offset)? as i32)
}

fn read_binary_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, String> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| "binary header offset overflowed".to_string())?;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| "binary multi-LoRA header is truncated".to_string())?;
    *offset = end;
    Ok(u64::from_le_bytes(value.try_into().unwrap()))
}

fn read_binary_i64_vec(
    bytes: &[u8],
    offset: &mut usize,
    count: usize,
    label: &str,
) -> Result<Vec<i64>, String> {
    let byte_count = count
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| format!("binary {label} count overflowed"))?;
    let end = offset
        .checked_add(byte_count)
        .ok_or_else(|| format!("binary {label} offset overflowed"))?;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| format!("binary {label} section is truncated"))?;
    *offset = end;
    value
        .chunks_exact(8)
        .map(|chunk| Ok(i64::from_le_bytes(chunk.try_into().unwrap())))
        .collect()
}

fn parse_binary_multi_lora_request(bytes: &[u8]) -> Result<BinaryMultiLoraRequest, String> {
    if bytes.len() < MULTI_LORA_BINARY_HEADER_BYTES {
        return Err("binary multi-LoRA request is shorter than its header".to_string());
    }
    if bytes[..4] != MULTI_LORA_BINARY_MAGIC {
        return Err("binary multi-LoRA request has an invalid magic".to_string());
    }
    let mut offset = 4;
    let version = read_binary_u16(bytes, &mut offset)?;
    let flags = read_binary_u16(bytes, &mut offset)?;
    if version != MULTI_LORA_BINARY_VERSION {
        return Err(format!(
            "unsupported binary multi-LoRA version {version}, expected {MULTI_LORA_BINARY_VERSION}"
        ));
    }
    if flags != 0 {
        return Err("binary multi-LoRA request contains unsupported flags".to_string());
    }
    let batch_size = usize::try_from(read_binary_u32(bytes, &mut offset)?)
        .map_err(|_| "binary batch size does not fit usize".to_string())?;
    let seq_len = usize::try_from(read_binary_u32(bytes, &mut offset)?)
        .map_err(|_| "binary sequence length does not fit usize".to_string())?;
    let n_total = read_binary_i32(bytes, &mut offset)?;
    let lora_rank = read_binary_i32(bytes, &mut offset)?;
    let adapter_count = usize::try_from(read_binary_u32(bytes, &mut offset)?)
        .map_err(|_| "binary adapter count does not fit usize".to_string())?;
    let expected_count = usize::try_from(read_binary_u32(bytes, &mut offset)?)
        .map_err(|_| "binary expected-step count does not fit usize".to_string())?;
    let input_bytes = usize::try_from(read_binary_u64(bytes, &mut offset)?)
        .map_err(|_| "binary input length does not fit usize".to_string())?;
    let target_bytes = usize::try_from(read_binary_u64(bytes, &mut offset)?)
        .map_err(|_| "binary target length does not fit usize".to_string())?;
    let attention_bytes = usize::try_from(read_binary_u64(bytes, &mut offset)?)
        .map_err(|_| "binary attention length does not fit usize".to_string())?;
    if offset != MULTI_LORA_BINARY_HEADER_BYTES {
        return Err("binary multi-LoRA header size mismatch".to_string());
    }
    if batch_size == 0 || seq_len <= 1 || n_total <= 0 || lora_rank <= 0 {
        return Err("binary multi-LoRA dimensions and counts must be positive".to_string());
    }
    let n_total_usize = usize::try_from(n_total)
        .map_err(|_| "binary adapter count does not fit usize".to_string())?;
    if adapter_count != n_total_usize {
        return Err(format!(
            "binary adapter count {adapter_count} does not match n_total {n_total}"
        ));
    }
    if expected_count != 0 && expected_count != n_total_usize {
        return Err("binary expected-step count must be zero or n_total".to_string());
    }
    let elements = batch_size
        .checked_mul(seq_len)
        .ok_or_else(|| "binary tensor element count overflowed".to_string())?;
    let expected_tensor_bytes = elements
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| "binary tensor byte count overflowed".to_string())?;
    if input_bytes != expected_tensor_bytes
        || target_bytes != expected_tensor_bytes
        || attention_bytes != expected_tensor_bytes
    {
        return Err(format!(
            "binary tensor lengths must all equal {expected_tensor_bytes}"
        ));
    }
    let adapter_ids = read_binary_i64_vec(bytes, &mut offset, n_total_usize, "adapter IDs")?;
    let expected_steps = if expected_count == 0 {
        Vec::new()
    } else {
        let values = read_binary_i64_vec(bytes, &mut offset, expected_count, "expected steps")?;
        values
            .into_iter()
            .map(|value| {
                u64::try_from(value)
                    .map_err(|_| "binary expected steps must be non-negative".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let tensor_sections = [input_bytes, target_bytes, attention_bytes];
    let mut payload = Vec::with_capacity(
        tensor_sections
            .iter()
            .try_fold(0usize, |total, size| total.checked_add(*size))
            .and_then(|total| total.checked_add(2 * (rustrain_ipc::TENSOR_SPAN_ALIGNMENT - 1)))
            .ok_or_else(|| "binary tensor slab capacity overflowed".to_string())?,
    );
    let mut spans = Vec::with_capacity(3);
    for section_size in tensor_sections {
        let aligned = payload
            .len()
            .checked_add(rustrain_ipc::TENSOR_SPAN_ALIGNMENT - 1)
            .map(|value| value & !(rustrain_ipc::TENSOR_SPAN_ALIGNMENT - 1))
            .ok_or_else(|| "binary tensor slab alignment overflowed".to_string())?;
        payload.resize(aligned, 0);
        let start = offset;
        let end = start
            .checked_add(section_size)
            .ok_or_else(|| "binary tensor section offset overflowed".to_string())?;
        payload.extend_from_slice(
            bytes
                .get(start..end)
                .ok_or_else(|| "binary tensor section is truncated".to_string())?,
        );
        offset = end;
        spans.push(rustrain_ipc::TensorSpan {
            offset_bytes: u64::try_from(aligned)
                .map_err(|_| "binary tensor span offset exceeds u64".to_string())?,
            len_bytes: u64::try_from(section_size)
                .map_err(|_| "binary tensor span length exceeds u64".to_string())?,
        });
    }
    if offset != bytes.len() {
        return Err("binary multi-LoRA request has trailing bytes".to_string());
    }
    let tensors = rustrain_ipc::TensorSlabRef {
        input_ids: spans[0],
        target_mask: spans[1],
        attention_mask: spans[2],
        batch_size,
        seq_len,
    };
    tensors.validate(payload.len())?;
    Ok(BinaryMultiLoraRequest {
        tensors,
        payload,
        batch_size,
        seq_len,
        n_total,
        lora_rank,
        adapter_ids,
        expected_steps,
    })
}
#[derive(Serialize)]
struct TrainStepResponse {
    loss: f64,
    step: u64,
}

#[derive(Serialize)]
struct TrainMultiLoraResponse {
    capability_version: u32,
    capabilities: &'static [&'static str],
    loss: f64,
    step: u64,
    loss_scope: &'static str,
    coalesced_requests: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    adapter_losses: Vec<rustrain_ipc::command::AdapterLoss>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    adapter_steps: Vec<rustrain_ipc::command::AdapterStep>,
}

fn multi_lora_capability_v1(headers: &HeaderMap) -> bool {
    headers
        .get(MULTI_LORA_CAPABILITY_HEADER)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .split(',')
                .map(str::trim)
                .any(|token| token.eq_ignore_ascii_case(MULTI_LORA_CAPABILITY_V1))
        })
}

/// Decode the little-endian wire representation without materializing host values.
fn decode_int64_bytes(t: &TensorHttp) -> Result<Vec<u8>, String> {
    use base64::{engine::general_purpose, Engine};
    if t.dtype != "int64" {
        return Err(format!("EP tensor dtype must be int64, got {}", t.dtype));
    }
    let bytes = general_purpose::STANDARD
        .decode(&t.data)
        .map_err(|e| format!("base64 decode: {e}"))?;
    let expected = t
        .shape
        .iter()
        .try_fold(1usize, |acc, dim| {
            usize::try_from(*dim)
                .ok()
                .and_then(|dim| acc.checked_mul(dim))
        })
        .ok_or_else(|| format!("invalid tensor shape {:?}", t.shape))?;
    let expected_bytes = expected
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| format!("tensor shape {:?} byte size overflows usize", t.shape))?;
    if expected_bytes != bytes.len() {
        return Err(format!(
            "tensor shape {:?} expects {} int64 bytes, got {}",
            t.shape,
            expected_bytes,
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn decode_int64_values(t: &TensorHttp) -> Result<Vec<i64>, String> {
    let bytes = decode_int64_bytes(t)?;
    bytes
        .chunks_exact(std::mem::size_of::<i64>())
        .map(|chunk| {
            let bytes: [u8; 8] = chunk
                .try_into()
                .map_err(|_| "invalid int64 byte width".to_string())?;
            Ok(i64::from_le_bytes(bytes))
        })
        .collect()
}

fn pack_tensor_slab(
    input_ids: &TensorHttp,
    target_mask: &TensorHttp,
    attention_mask: &TensorHttp,
    batch_size: usize,
    seq_len: usize,
) -> Result<(rustrain_ipc::TensorSlabRef, Vec<u8>), String> {
    let decoded = [
        decode_int64_bytes(input_ids)?,
        decode_int64_bytes(target_mask)?,
        decode_int64_bytes(attention_mask)?,
    ];
    let total = decoded
        .iter()
        .try_fold(0usize, |sum, bytes| sum.checked_add(bytes.len()))
        .and_then(|sum| sum.checked_add(2 * (rustrain_ipc::TENSOR_SPAN_ALIGNMENT - 1)))
        .ok_or_else(|| "tensor slab payload size overflows usize".to_string())?;
    let mut payload = Vec::with_capacity(total);
    let mut spans = Vec::with_capacity(3);
    for bytes in decoded {
        let aligned = payload
            .len()
            .checked_add(rustrain_ipc::TENSOR_SPAN_ALIGNMENT - 1)
            .map(|value| value & !(rustrain_ipc::TENSOR_SPAN_ALIGNMENT - 1))
            .ok_or_else(|| "tensor slab alignment overflows usize".to_string())?;
        payload.resize(aligned, 0);
        let offset_bytes =
            u64::try_from(aligned).map_err(|_| "tensor slab offset exceeds u64".to_string())?;
        let len_bytes = u64::try_from(bytes.len())
            .map_err(|_| "tensor slab tensor length exceeds u64".to_string())?;
        payload.extend_from_slice(&bytes);
        spans.push(rustrain_ipc::TensorSpan {
            offset_bytes,
            len_bytes,
        });
    }
    Ok((
        rustrain_ipc::TensorSlabRef {
            input_ids: spans[0],
            target_mask: spans[1],
            attention_mask: spans[2],
            batch_size,
            seq_len,
        },
        payload,
    ))
}

fn multi_lora_source_count(batch_size: usize, n_total: usize) -> Result<usize, String> {
    if n_total == 0 {
        return Err("multi-LoRA adapter count must be positive".to_string());
    }
    if batch_size == 1 {
        return Ok(1);
    }
    if batch_size % n_total != 0 {
        return Err(format!(
            "multi-LoRA batch_size={batch_size} must be 1 or a multiple of n_total={n_total}"
        ));
    }
    Ok(batch_size / n_total)
}

fn normalized_multi_lora_payload_bytes(
    seq_len: usize,
    n_total: usize,
    source_count: usize,
) -> Result<usize, String> {
    n_total
        .checked_mul(source_count)
        .and_then(|rows| rows.checked_mul(seq_len))
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<i64>()))
        .and_then(|tensor_bytes| tensor_bytes.checked_mul(3))
        .and_then(|bytes| bytes.checked_add(2 * (rustrain_ipc::TENSOR_SPAN_ALIGNMENT - 1)))
        .ok_or_else(|| "normalized multi-LoRA payload size overflowed usize".to_string())
}

type MultiLoraWindowBuild = (
    rustrain_ipc::EpCommand,
    Vec<u8>,
    Vec<MultiLoraResponseTarget>,
);

fn build_multi_lora_window(
    window: MultiLoraWindow,
) -> Result<MultiLoraWindowBuild, (String, Vec<MultiLoraResponseTarget>)> {
    let build_result = build_multi_lora_window_payload(&window);
    let mut adapter_start = 0usize;
    let responses = window
        .requests
        .into_iter()
        .map(|request| {
            let adapter_end = adapter_start + request.n_total;
            let target = MultiLoraResponseTarget {
                response: request.response,
                adapter_range: adapter_start..adapter_end,
            };
            adapter_start = adapter_end;
            target
        })
        .collect::<Vec<_>>();
    match build_result {
        Ok((tensors, payload, adapter_ids, lora_rank, expected_steps)) => {
            let n_total = match i32::try_from(adapter_ids.len()) {
                Ok(n_total) => n_total,
                Err(_) => {
                    return Err(("coalesced adapter count exceeds i32".to_string(), responses));
                }
            };
            Ok((
                rustrain_ipc::EpCommand::TrainMultiLoraSlab {
                    session_id: window.session_id,
                    tensors,
                    n_total,
                    lora_rank,
                    adapter_ids,
                    expected_steps,
                },
                payload,
                responses,
            ))
        }
        Err(error) => Err((error, responses)),
    }
}

fn build_multi_lora_window_payload(
    window: &MultiLoraWindow,
) -> Result<
    (
        rustrain_ipc::TensorSlabRef,
        Vec<u8>,
        Vec<i64>,
        i32,
        Vec<u64>,
    ),
    String,
> {
    if window.requests.is_empty() {
        return Err("coalesced multi-LoRA window is empty".to_string());
    }
    let total_adapters = window
        .requests
        .iter()
        .try_fold(0usize, |total, request| total.checked_add(request.n_total))
        .ok_or_else(|| "coalesced adapter count overflowed usize".to_string())?;
    let batch_size = total_adapters
        .checked_mul(window.source_count)
        .ok_or_else(|| "coalesced batch size overflowed usize".to_string())?;
    let row_bytes = window
        .seq_len
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| "coalesced row byte count overflowed usize".to_string())?;
    let tensor_bytes = batch_size
        .checked_mul(row_bytes)
        .ok_or_else(|| "coalesced tensor byte count overflowed usize".to_string())?;
    let payload_capacity = tensor_bytes
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(2 * (rustrain_ipc::TENSOR_SPAN_ALIGNMENT - 1)))
        .ok_or_else(|| "coalesced payload capacity overflowed usize".to_string())?;
    let mut payload = Vec::with_capacity(payload_capacity);
    let mut spans = Vec::with_capacity(3);

    for tensor_index in 0..3 {
        let aligned = payload
            .len()
            .checked_add(rustrain_ipc::TENSOR_SPAN_ALIGNMENT - 1)
            .map(|value| value & !(rustrain_ipc::TENSOR_SPAN_ALIGNMENT - 1))
            .ok_or_else(|| "coalesced tensor alignment overflowed usize".to_string())?;
        payload.resize(aligned, 0);
        let offset_bytes = u64::try_from(aligned)
            .map_err(|_| "coalesced tensor offset exceeds u64".to_string())?;
        for source_index in 0..window.source_count {
            for request in &window.requests {
                request.tensors.validate(request.payload.len())?;
                if request.source_count != window.source_count
                    || request.tensors.seq_len != window.seq_len
                {
                    return Err("coalesced request layout changed after admission".to_string());
                }
                let span = request.tensors.spans()[tensor_index];
                let tensor_start = usize::try_from(span.offset_bytes)
                    .map_err(|_| "request tensor offset exceeds usize".to_string())?;
                for adapter_index in 0..request.n_total {
                    let row_index = if request.tensors.batch_size == 1 {
                        0
                    } else {
                        source_index
                            .checked_mul(request.n_total)
                            .and_then(|row| row.checked_add(adapter_index))
                            .ok_or_else(|| "request row index overflowed usize".to_string())?
                    };
                    let start = tensor_start
                        .checked_add(
                            row_index
                                .checked_mul(row_bytes)
                                .ok_or_else(|| "request row offset overflowed usize".to_string())?,
                        )
                        .ok_or_else(|| "request tensor range overflowed usize".to_string())?;
                    let end = start
                        .checked_add(row_bytes)
                        .ok_or_else(|| "request tensor range overflowed usize".to_string())?;
                    let row = request
                        .payload
                        .get(start..end)
                        .ok_or_else(|| "request tensor row exceeds payload".to_string())?;
                    payload.extend_from_slice(row);
                }
            }
        }
        spans.push(rustrain_ipc::TensorSpan {
            offset_bytes,
            len_bytes: u64::try_from(tensor_bytes)
                .map_err(|_| "coalesced tensor length exceeds u64".to_string())?,
        });
    }

    let adapter_ids = window
        .requests
        .iter()
        .flat_map(|request| request.adapter_ids.iter().copied())
        .collect::<Vec<_>>();
    if adapter_ids.len() != total_adapters && !adapter_ids.is_empty() {
        return Err("coalesced adapter ID count does not match batch geometry".to_string());
    }
    let expected_steps = window
        .requests
        .iter()
        .flat_map(|request| request.expected_steps.iter().copied())
        .collect::<Vec<_>>();
    if !expected_steps.is_empty() && expected_steps.len() != total_adapters {
        return Err("expected step count does not match coalesced adapter count".to_string());
    }
    let lora_rank = window
        .requests
        .iter()
        .map(|request| request.lora_rank)
        .max()
        .unwrap_or(0);
    let tensors = rustrain_ipc::TensorSlabRef {
        input_ids: spans[0],
        target_mask: spans[1],
        attention_mask: spans[2],
        batch_size,
        seq_len: window.seq_len,
    };
    tensors.validate(payload.len())?;
    Ok((tensors, payload, adapter_ids, lora_rank, expected_steps))
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

fn validate_selected_eval_http_shapes(
    input_ids: &TensorHttp,
    target_mask: &TensorHttp,
    attention_mask: &TensorHttp,
    adapter_ids: &[i64],
) -> Result<(usize, usize), String> {
    if adapter_ids.is_empty() || adapter_ids.iter().any(|id| *id <= 0) {
        return Err("adapter_ids must contain positive IDs".to_string());
    }
    let (batch_size, seq_len) = validate_train_http_shapes(input_ids, target_mask, attention_mask)?;
    if batch_size != 1 && batch_size != adapter_ids.len() {
        return Err(format!(
            "selected eval batch_size must be 1 or adapter count {}, got {}",
            adapter_ids.len(),
            batch_size
        ));
    }
    Ok((batch_size, seq_len))
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

async fn eval_multi_lora(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<EvalMultiLoraHttp>,
) -> Result<Json<TrainMultiLoraEvalResponse>, (StatusCode, Json<ErrorResponse>)> {
    let (batch_size, seq_len) = validate_selected_eval_http_shapes(
        &req.input_ids,
        &req.target_mask,
        &req.attention_mask,
        &req.adapter_ids,
    )
    .map_err(|e| err_resp(&e))?;
    let input_ids = decode_int64_values(&req.input_ids).map_err(|e| err_resp(&e))?;
    let target_mask = decode_int64_values(&req.target_mask).map_err(|e| err_resp(&e))?;
    let attention_mask = decode_int64_values(&req.attention_mask).map_err(|e| err_resp(&e))?;
    let session = state
        .manager
        .get_session(&id)
        .await
        .ok_or_else(|| err_resp("session not found"))?;
    let s = session.lock().await;
    let output = s
        .eval_multi_lora_host_i64(
            &input_ids,
            &target_mask,
            &attention_mask,
            batch_size,
            seq_len,
            &req.adapter_ids,
        )
        .map_err(|e| err_resp(&e.to_string()))?;
    Ok(Json(TrainMultiLoraEvalResponse {
        adapter_losses: output
            .adapter_losses
            .into_iter()
            .map(|(adapter_id, loss)| rustrain_ipc::command::AdapterLoss { adapter_id, loss })
            .collect(),
    }))
}

#[derive(Serialize)]
struct TrainMultiLoraEvalResponse {
    adapter_losses: Vec<rustrain_ipc::command::AdapterLoss>,
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
    #[serde(default)]
    optimizer_lr: Option<f64>,
    #[serde(default)]
    optimizer_beta1: Option<f64>,
    #[serde(default)]
    optimizer_beta2: Option<f64>,
    #[serde(default)]
    optimizer_eps: Option<f64>,
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
            optimizer_lr: req.optimizer_lr,
            optimizer_beta1: req.optimizer_beta1,
            optimizer_beta2: req.optimizer_beta2,
            optimizer_eps: req.optimizer_eps,
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
    let batch_config = MultiLoraBatchConfig::from_env();
    let multi_lora_batcher = MultiLoraBatcher::new(batch_config);
    let state = Arc::new(EpRouterState {
        coordinator: Arc::clone(&state.coordinator),
        world_size: state.world_size,
        dispatcher: EpDispatchScheduler::new(configured_queue_capacity()),
        dispatch_submission: Mutex::new(()),
        multi_lora_batcher,
    });
    let tensor_routes = Router::new()
        .route("/v1/sessions/{id}/train_step", post(ep_train_step))
        .route("/v1/sessions/{id}/eval_step", post(ep_eval_step))
        .route(
            "/v1/sessions/{id}/eval_multi_lora",
            post(ep_eval_multi_lora),
        )
        .route_layer(middleware::from_fn_with_state(
            Arc::new(Semaphore::new(1)),
            ep_tensor_admission,
        ));
    let multi_lora_route = Router::new()
        .route("/v1/sessions/{id}/train_multi", post(ep_train_multi_lora))
        .route(
            "/v1/sessions/{id}/train_multi_binary",
            post(ep_train_multi_lora_binary),
        )
        .route_layer(middleware::from_fn_with_state(
            Arc::new(Semaphore::new(batch_config.max_requests)),
            ep_tensor_admission,
        ));

    Router::new()
        .merge(tensor_routes)
        .merge(multi_lora_route)
        .route("/v1/sessions", post(ep_create_session))
        .route(
            "/v1/sessions/{id}",
            axum::routing::delete(ep_delete_session),
        )
        .route("/v1/sessions/{id}/load_model", post(ep_load_model))
        .route("/v1/sessions/{id}/load_dataset", post(ep_load_dataset))
        .route("/v1/sessions/{id}/init_lora", post(ep_init_lora))
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
        .layer(DefaultBodyLimit::max(48 * 1024 * 1024))
        .with_state(state)
}

async fn ep_tensor_admission(
    State(gate): State<Arc<Semaphore>>,
    request: Request,
    next: Next,
) -> Response {
    // The IPC coordinator serializes dispatches, so decoding more than one tensor request only
    // increases peak memory without creating training concurrency.
    match gate.try_acquire_owned() {
        Ok(_permit) => next.run(request).await,
        Err(_) => (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse {
                error: "EP tensor admission capacity is exhausted".to_string(),
            }),
        )
            .into_response(),
    }
}

async fn ep_health(
    State(state): State<Arc<EpRouterState>>,
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
    State(state): State<Arc<EpRouterState>>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::CreateSession {
        session_id: req.session_id.clone(),
    };
    match dispatch_ep(&state, cmd).await? {
        rustrain_ipc::EpResult::Ok => Ok(Json(CreateSessionResponse {
            session_id: req.session_id,
        })),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_delete_session(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::DeleteSession {
        session_id: id.clone(),
    };
    match dispatch_ep(&state, cmd).await? {
        rustrain_ipc::EpResult::Ok => Ok(Json(serde_json::json!({"deleted": id}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_load_model(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
    Json(req): Json<LoadModelHttp>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::LoadModel {
        session_id: id,
        model_path: req.model_path,
        config_toml: req.config_toml,
    };
    match dispatch_ep(&state, cmd).await? {
        rustrain_ipc::EpResult::Ok => Ok(Json(serde_json::json!({"loaded": true}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_load_dataset(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
    Json(req): Json<LoadDatasetHttp>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::LoadDataset {
        session_id: id,
        jsonl_path: req.jsonl_path,
        seq_len: req.seq_len,
    };
    match dispatch_ep(&state, cmd).await? {
        rustrain_ipc::EpResult::Count(n) => Ok(Json(serde_json::json!({"samples": n}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_init_lora(
    State(state): State<Arc<EpRouterState>>,
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
    match dispatch_ep(&state, cmd).await? {
        rustrain_ipc::EpResult::Count(n) => Ok(Json(serde_json::json!({"lora_count": n}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_train_step(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
    Json(req): Json<TrainStepHttp>,
) -> Result<Json<TrainStepResponse>, (StatusCode, Json<ErrorResponse>)> {
    let (batch_size, seq_len) =
        validate_train_http_shapes(&req.input_ids, &req.target_mask, &req.attention_mask)
            .map_err(|e| err_resp(&e))?;
    let (tensors, payload) = pack_tensor_slab(
        &req.input_ids,
        &req.target_mask,
        &req.attention_mask,
        batch_size,
        seq_len,
    )
    .map_err(|e| err_resp(&e))?;

    let cmd = rustrain_ipc::EpCommand::TrainStepSlab {
        session_id: id,
        tensors,
    };
    match dispatch_slab(&state, cmd, payload).await? {
        rustrain_ipc::EpResult::Train { loss, step } => Ok(Json(TrainStepResponse { loss, step })),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_eval_step(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
    Json(req): Json<TrainStepHttp>,
) -> Result<Json<EvalStepResponse>, (StatusCode, Json<ErrorResponse>)> {
    let (batch_size, seq_len) =
        validate_train_http_shapes(&req.input_ids, &req.target_mask, &req.attention_mask)
            .map_err(|e| err_resp(&e))?;
    let (tensors, payload) = pack_tensor_slab(
        &req.input_ids,
        &req.target_mask,
        &req.attention_mask,
        batch_size,
        seq_len,
    )
    .map_err(|e| err_resp(&e))?;

    let cmd = rustrain_ipc::EpCommand::EvalStepSlab {
        session_id: id,
        tensors,
    };
    match dispatch_slab(&state, cmd, payload).await? {
        rustrain_ipc::EpResult::Loss(loss) => Ok(Json(EvalStepResponse { loss })),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_eval_multi_lora(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
    Json(req): Json<EvalMultiLoraHttp>,
) -> Result<Json<TrainMultiLoraEvalResponse>, (StatusCode, Json<ErrorResponse>)> {
    let (batch_size, seq_len) = validate_selected_eval_http_shapes(
        &req.input_ids,
        &req.target_mask,
        &req.attention_mask,
        &req.adapter_ids,
    )
    .map_err(|e| err_resp(&e))?;
    let (tensors, payload) = pack_tensor_slab(
        &req.input_ids,
        &req.target_mask,
        &req.attention_mask,
        batch_size,
        seq_len,
    )
    .map_err(|e| err_resp(&e))?;
    let cmd = rustrain_ipc::EpCommand::EvalMultiLoraSlab {
        session_id: id,
        tensors,
        adapter_ids: req.adapter_ids,
    };
    match dispatch_slab(&state, cmd, payload).await? {
        rustrain_ipc::EpResult::MultiLoraEval { adapter_losses } => {
            Ok(Json(TrainMultiLoraEvalResponse { adapter_losses }))
        }
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_train_multi_lora(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<TrainMultiLoraHttp>,
) -> Result<Json<TrainMultiLoraResponse>, (StatusCode, Json<ErrorResponse>)> {
    let (batch_size, seq_len) = validate_multi_lora_http_shapes(
        &req.input_ids,
        &req.target_mask,
        &req.attention_mask,
        req.n_total,
    )
    .map_err(|e| err_resp(&e))?;
    crate::ep::validate_multi_lora_global_batch_size(batch_size, req.n_total, state.world_size)
        .map_err(|error| err_resp(&error))?;
    let (tensors, payload) = pack_tensor_slab(
        &req.input_ids,
        &req.target_mask,
        &req.attention_mask,
        batch_size,
        seq_len,
    )
    .map_err(|e| err_resp(&e))?;
    submit_ep_multi_lora(
        state,
        id,
        headers,
        tensors,
        payload,
        batch_size,
        seq_len,
        req.n_total,
        req.lora_rank,
        req.adapter_ids,
        req.expected_steps,
        req.allow_aggregate_loss,
    )
    .await
}

async fn ep_train_multi_lora_binary(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<TrainMultiLoraResponse>, (StatusCode, Json<ErrorResponse>)> {
    let request = parse_binary_multi_lora_request(&body).map_err(|e| err_resp(&e))?;
    crate::ep::validate_multi_lora_global_batch_size(
        request.batch_size,
        request.n_total,
        state.world_size,
    )
    .map_err(|error| err_resp(&error))?;
    submit_ep_multi_lora(
        state,
        id,
        headers,
        request.tensors,
        request.payload,
        request.batch_size,
        request.seq_len,
        request.n_total,
        request.lora_rank,
        request.adapter_ids,
        request.expected_steps,
        false,
    )
    .await
}

async fn submit_ep_multi_lora(
    state: Arc<EpRouterState>,
    id: String,
    headers: HeaderMap,
    tensors: rustrain_ipc::TensorSlabRef,
    payload: Vec<u8>,
    batch_size: usize,
    seq_len: usize,
    n_total: i32,
    lora_rank: i32,
    adapter_ids: Vec<i64>,
    expected_steps: Vec<u64>,
    allow_aggregate_loss: bool,
) -> Result<Json<TrainMultiLoraResponse>, (StatusCode, Json<ErrorResponse>)> {
    if n_total <= 0 || lora_rank <= 0 {
        return Err(err_resp(
            "multi-LoRA n_total and lora_rank must be positive",
        ));
    }
    if !adapter_ids.is_empty() {
        if adapter_ids.len() != n_total as usize {
            return Err(err_resp(&format!(
                "adapter_ids length {} must match n_total={}",
                adapter_ids.len(),
                n_total
            )));
        }
        if adapter_ids.iter().any(|id| *id <= 0) {
            return Err(err_resp("adapter_ids must contain only positive IDs"));
        }
        if adapter_ids.iter().copied().collect::<HashSet<_>>().len() != adapter_ids.len() {
            return Err(err_resp("adapter_ids must not contain duplicates"));
        }
        if !expected_steps.is_empty() && expected_steps.len() != adapter_ids.len() {
            return Err(err_resp(&format!(
                "expected_steps length {} must match adapter_ids length {}",
                expected_steps.len(),
                adapter_ids.len()
            )));
        }
    } else if !expected_steps.is_empty() {
        return Err(err_resp("expected_steps requires adapter_ids"));
    }
    let source_count =
        multi_lora_source_count(batch_size, n_total as usize).map_err(|error| err_resp(&error))?;
    let normalized_payload_bytes =
        normalized_multi_lora_payload_bytes(seq_len, n_total as usize, source_count)
            .map_err(|error| err_resp(&error))?;
    if !state.coordinator.is_healthy() {
        return Err(ep_dispatch_unavailable("EP coordinator is unavailable"));
    }
    let (response, receiver) = oneshot::channel();
    let request = MultiLoraDispatchRequest {
        session_id: id,
        tensors,
        payload,
        n_total: n_total as usize,
        lora_rank,
        adapter_ids,
        expected_steps,
        source_count,
        normalized_payload_bytes,
        // Capability v1 means the client understands per-adapter losses,
        // optimizer steps, and the explicit coalesced loss scope returned
        // below. Keep the body flag as a backwards-compatible override.
        allow_aggregate_loss: allow_aggregate_loss || multi_lora_capability_v1(&headers),
        response,
    };
    {
        let _submission = state
            .dispatch_submission
            .lock()
            .map_err(|_| ep_dispatch_schedule_error(EpDispatchScheduleError::WorkerFailed))?;
        state
            .multi_lora_batcher
            .submit(&state.dispatcher, Arc::clone(&state.coordinator), request)
            .map_err(ep_dispatch_schedule_error)?;
    }
    let outcome = receiver
        .await
        .map_err(|_| ep_dispatch_schedule_error(EpDispatchScheduleError::WorkerFailed))?;
    match outcome.result {
        rustrain_ipc::EpResult::MultiLoraTrain {
            loss,
            step,
            adapter_losses,
            adapter_steps,
        } => Ok(Json(TrainMultiLoraResponse {
            capability_version: 1,
            capabilities: MULTI_LORA_RESPONSE_CAPABILITIES,
            loss,
            step,
            loss_scope: if outcome.request_count > 1 {
                "coalesced_batch"
            } else {
                "request"
            },
            coalesced_requests: outcome.request_count,
            adapter_losses,
            adapter_steps,
        })),
        rustrain_ipc::EpResult::Train { loss, step } => Ok(Json(TrainMultiLoraResponse {
            capability_version: 1,
            capabilities: MULTI_LORA_RESPONSE_CAPABILITIES,
            loss,
            step,
            loss_scope: if outcome.request_count > 1 {
                "coalesced_batch"
            } else {
                "request"
            },
            coalesced_requests: outcome.request_count,
            adapter_losses: Vec::new(),
            adapter_steps: Vec::new(),
        })),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn dispatch_slab(
    state: &EpRouterState,
    command: rustrain_ipc::EpCommand,
    payload: Vec<u8>,
) -> Result<rustrain_ipc::EpResult, (StatusCode, Json<ErrorResponse>)> {
    if !state.coordinator.is_healthy() {
        return Err(ep_dispatch_unavailable("EP coordinator is unavailable"));
    }
    let coordinator = Arc::clone(&state.coordinator);
    let receiver = {
        let _submission = state
            .dispatch_submission
            .lock()
            .map_err(|_| ep_dispatch_schedule_error(EpDispatchScheduleError::WorkerFailed))?;
        state.multi_lora_batcher.seal_current();
        state
            .dispatcher
            .submit(move || coordinator.dispatch_with_slab(&command, &payload))
            .map_err(ep_dispatch_schedule_error)?
    };
    receiver
        .await
        .map_err(|_| ep_dispatch_schedule_error(EpDispatchScheduleError::WorkerFailed))
}

async fn dispatch_ep(
    state: &EpRouterState,
    command: rustrain_ipc::EpCommand,
) -> Result<rustrain_ipc::EpResult, (StatusCode, Json<ErrorResponse>)> {
    if !state.coordinator.is_healthy() {
        return Err(ep_dispatch_unavailable("EP coordinator is unavailable"));
    }
    let coordinator = Arc::clone(&state.coordinator);
    let receiver = {
        let _submission = state
            .dispatch_submission
            .lock()
            .map_err(|_| ep_dispatch_schedule_error(EpDispatchScheduleError::WorkerFailed))?;
        state.multi_lora_batcher.seal_current();
        state
            .dispatcher
            .submit(move || coordinator.dispatch(&command))
            .map_err(ep_dispatch_schedule_error)?
    };
    receiver
        .await
        .map_err(|_| ep_dispatch_schedule_error(EpDispatchScheduleError::WorkerFailed))
}

fn ep_dispatch_schedule_error(error: EpDispatchScheduleError) -> (StatusCode, Json<ErrorResponse>) {
    let status = match error {
        EpDispatchScheduleError::QueueFull => StatusCode::TOO_MANY_REQUESTS,
        EpDispatchScheduleError::QueueClosed | EpDispatchScheduleError::WorkerFailed => {
            StatusCode::SERVICE_UNAVAILABLE
        }
    };
    (
        status,
        Json(ErrorResponse {
            error: error.to_string(),
        }),
    )
}

fn ep_dispatch_unavailable(message: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: message.to_string(),
        }),
    )
}

#[cfg(test)]
mod tensor_http_shape_tests {
    use base64::{engine::general_purpose, Engine};

    use super::{
        build_multi_lora_window_payload, decode_int64_bytes, ep_dispatch_schedule_error,
        multi_lora_capability_v1, multi_lora_source_count, normalized_multi_lora_payload_bytes,
        pack_tensor_slab, parse_binary_multi_lora_request, project_multi_lora_result,
        validate_multi_lora_http_shapes, validate_selected_eval_http_shapes,
        validate_train_http_shapes, EpDispatchScheduleError, HeaderMap, HeaderValue,
        MultiLoraBatchConfig, MultiLoraBatcher, MultiLoraDispatchRequest, MultiLoraWindow,
        StatusCode, TensorHttp, MULTI_LORA_CAPABILITY_HEADER,
    };

    fn tensor(shape: &[i64]) -> TensorHttp {
        TensorHttp {
            data: String::new(),
            shape: shape.to_vec(),
            dtype: "int64".into(),
        }
    }

    fn encoded_tensor(rows: &[i64]) -> TensorHttp {
        TensorHttp {
            data: general_purpose::STANDARD.encode(
                rows.iter()
                    .copied()
                    .flat_map(i64::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
            shape: vec![rows.len() as i64, 1],
            dtype: "int64".into(),
        }
    }

    fn batch_request(
        session_id: &str,
        rows: &[i64],
        n_total: usize,
        adapter_ids: &[i64],
    ) -> MultiLoraDispatchRequest {
        let tensor = encoded_tensor(rows);
        let (tensors, payload) =
            pack_tensor_slab(&tensor, &tensor, &tensor, rows.len(), 1).unwrap();
        let source_count = multi_lora_source_count(rows.len(), n_total).unwrap();
        let normalized_payload_bytes =
            normalized_multi_lora_payload_bytes(1, n_total, source_count).unwrap();
        let (response, _receiver) = tokio::sync::oneshot::channel();
        MultiLoraDispatchRequest {
            session_id: session_id.to_string(),
            tensors,
            payload,
            n_total,
            lora_rank: 8,
            adapter_ids: adapter_ids.to_vec(),
            expected_steps: Vec::new(),
            source_count,
            normalized_payload_bytes,
            allow_aggregate_loss: true,
            response,
        }
    }

    fn batch_request_with_rank(
        session_id: &str,
        rows: &[i64],
        n_total: usize,
        adapter_ids: &[i64],
        lora_rank: i32,
    ) -> MultiLoraDispatchRequest {
        let mut request = batch_request(session_id, rows, n_total, adapter_ids);
        request.lora_rank = lora_rank;
        request
    }

    fn slab_i64(payload: &[u8], span: rustrain_ipc::TensorSpan) -> Vec<i64> {
        let start = span.offset_bytes as usize;
        let end = start + span.len_bytes as usize;
        payload[start..end]
            .chunks_exact(8)
            .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
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
    fn multi_lora_capability_v1_accepts_versioned_header_tokens() {
        let mut headers = HeaderMap::new();
        assert!(!multi_lora_capability_v1(&headers));
        headers.insert(MULTI_LORA_CAPABILITY_HEADER, HeaderValue::from_static("v1"));
        assert!(multi_lora_capability_v1(&headers));
        headers.insert(
            MULTI_LORA_CAPABILITY_HEADER,
            HeaderValue::from_static("v0, V1"),
        );
        assert!(multi_lora_capability_v1(&headers));
        headers.insert(
            MULTI_LORA_CAPABILITY_HEADER,
            HeaderValue::from_static("v10"),
        );
        assert!(!multi_lora_capability_v1(&headers));
    }

    #[test]
    fn binary_multi_lora_request_parses_wire_sections_without_base64() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RLM1");
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes()); // batch
        bytes.extend_from_slice(&2u32.to_le_bytes()); // sequence
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_total
        bytes.extend_from_slice(&4i32.to_le_bytes()); // LoRA rank
        bytes.extend_from_slice(&1u32.to_le_bytes()); // adapter count
        bytes.extend_from_slice(&1u32.to_le_bytes()); // expected-step count
        for _ in 0..3 {
            bytes.extend_from_slice(&32u64.to_le_bytes());
        }
        bytes.extend_from_slice(&7i64.to_le_bytes());
        bytes.extend_from_slice(&3i64.to_le_bytes());
        for value in 0i64..12 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let request = parse_binary_multi_lora_request(&bytes).unwrap();
        assert_eq!(request.batch_size, 2);
        assert_eq!(request.seq_len, 2);
        assert_eq!(request.adapter_ids, vec![7]);
        assert_eq!(request.expected_steps, vec![3]);
        assert_eq!(request.payload.len(), 160);
        assert_eq!(request.tensors.input_ids.len_bytes, 32);
        assert_eq!(request.tensors.target_mask.len_bytes, 32);
        assert_eq!(request.tensors.attention_mask.len_bytes, 32);
        request.tensors.validate(request.payload.len()).unwrap();
    }

    #[test]
    fn binary_multi_lora_request_rejects_trailing_or_invalid_sections() {
        let mut bytes = b"RLM1".to_vec();
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&[0; 48]);
        assert!(parse_binary_multi_lora_request(&bytes).is_err());
        bytes[0] = b'X';
        assert!(parse_binary_multi_lora_request(&bytes).is_err());
    }

    #[test]
    fn train_shape_rejects_mismatched_masks_and_non_positive_dimensions() {
        let input = tensor(&[4, 16]);
        assert!(validate_train_http_shapes(&input, &tensor(&[2, 16]), &tensor(&[4, 16])).is_err());
        assert!(validate_train_http_shapes(
            &tensor(&[0, 16]),
            &tensor(&[0, 16]),
            &tensor(&[0, 16])
        )
        .is_err());
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

    #[test]
    fn selected_eval_shape_requires_one_row_or_one_row_per_adapter() {
        let one = tensor(&[1, 32]);
        assert_eq!(
            validate_selected_eval_http_shapes(&one, &one, &one, &[11, 12]).unwrap(),
            (1, 32)
        );
        let two = tensor(&[2, 32]);
        assert_eq!(
            validate_selected_eval_http_shapes(&two, &two, &two, &[11, 12]).unwrap(),
            (2, 32)
        );
        assert!(validate_selected_eval_http_shapes(&two, &two, &two, &[11]).is_err());
        assert!(validate_selected_eval_http_shapes(&one, &one, &one, &[0, 12]).is_err());
    }

    #[test]
    fn coalesced_multi_lora_preserves_source_major_adapter_rows() {
        let mut window = MultiLoraWindow::new(batch_request(
            "tenant-session",
            &[10, 11, 20, 21],
            2,
            &[101, 102],
        ));
        window.push(batch_request("tenant-session", &[12, 22], 1, &[103]));

        let (tensors, payload, adapter_ids, _, _) =
            build_multi_lora_window_payload(&window).unwrap();
        assert_eq!(adapter_ids, vec![101, 102, 103]);
        assert_eq!(tensors.batch_size, 6);
        assert_eq!(
            slab_i64(&payload, tensors.input_ids),
            vec![10, 11, 12, 20, 21, 22]
        );
        assert_eq!(
            slab_i64(&payload, tensors.target_mask),
            vec![10, 11, 12, 20, 21, 22]
        );
        tensors.validate(payload.len()).unwrap();
    }

    #[test]
    fn coalesced_multi_lora_expands_request_shared_rows_per_adapter() {
        let mut window =
            MultiLoraWindow::new(batch_request("tenant-session", &[7], 2, &[101, 102]));
        window.push(batch_request("tenant-session", &[8], 1, &[103]));

        let (tensors, payload, adapter_ids, _, _) =
            build_multi_lora_window_payload(&window).unwrap();
        assert_eq!(adapter_ids, vec![101, 102, 103]);
        assert_eq!(slab_i64(&payload, tensors.input_ids), vec![7, 7, 8]);
    }

    #[test]
    fn coalesced_multi_lora_preserves_expected_optimizer_steps() {
        let mut first = batch_request("tenant-session", &[7], 2, &[101, 102]);
        first.expected_steps = vec![3, 5];
        let mut window = MultiLoraWindow::new(first);
        let mut second = batch_request("tenant-session", &[8], 1, &[103]);
        second.expected_steps = vec![9];
        window.push(second);

        let (_, _, adapter_ids, _, expected_steps) =
            build_multi_lora_window_payload(&window).unwrap();
        assert_eq!(adapter_ids, vec![101, 102, 103]);
        assert_eq!(expected_steps, vec![3, 5, 9]);
    }

    #[test]
    fn coalesced_result_projects_only_the_request_adapter_range() {
        let result = rustrain_ipc::EpResult::MultiLoraTrain {
            loss: 2.0,
            step: 9,
            adapter_losses: vec![
                rustrain_ipc::command::AdapterLoss {
                    adapter_id: 101,
                    loss: 1.0,
                },
                rustrain_ipc::command::AdapterLoss {
                    adapter_id: 102,
                    loss: 2.0,
                },
                rustrain_ipc::command::AdapterLoss {
                    adapter_id: 103,
                    loss: 3.0,
                },
            ],
            adapter_steps: vec![
                rustrain_ipc::command::AdapterStep {
                    adapter_id: 101,
                    step: 4,
                },
                rustrain_ipc::command::AdapterStep {
                    adapter_id: 102,
                    step: 5,
                },
                rustrain_ipc::command::AdapterStep {
                    adapter_id: 103,
                    step: 6,
                },
            ],
        };
        let projected = project_multi_lora_result(&result, 1..3);
        match projected {
            rustrain_ipc::EpResult::MultiLoraTrain {
                loss,
                step,
                adapter_losses,
                adapter_steps,
            } => {
                assert_eq!(loss, 2.0);
                assert_eq!(step, 9);
                assert_eq!(
                    adapter_losses
                        .iter()
                        .map(|item| item.adapter_id)
                        .collect::<Vec<_>>(),
                    vec![102, 103]
                );
                assert_eq!(
                    adapter_steps
                        .iter()
                        .map(|item| (item.adapter_id, item.step))
                        .collect::<Vec<_>>(),
                    vec![(102, 5), (103, 6)]
                );
            }
            other => panic!("unexpected projected result: {other:?}"),
        }

        let legacy = rustrain_ipc::EpResult::MultiLoraTrain {
            loss: 4.0,
            step: 10,
            adapter_losses: Vec::new(),
            adapter_steps: Vec::new(),
        };
        assert!(matches!(
            project_multi_lora_result(&legacy, 0..1),
            rustrain_ipc::EpResult::Train {
                loss: 4.0,
                step: 10
            }
        ));
    }

    #[test]
    fn coalescing_rejects_overlapping_tenants_and_capacity_overflow() {
        let config = MultiLoraBatchConfig {
            window: std::time::Duration::from_millis(1),
            max_requests: 2,
            max_adapters: 3,
            max_rank_work: 64,
            max_payload_bytes: usize::MAX,
        };
        let mut window = MultiLoraWindow::new(batch_request("session", &[1], 1, &[11]));
        assert!(!window.can_accept(&batch_request("session", &[2], 1, &[11]), config));
        assert!(!window.can_accept(&batch_request("other", &[2], 1, &[12]), config));
        assert!(window.can_accept(
            &batch_request_with_rank("session", &[2], 1, &[12], 16),
            config
        ));
        let tight_config = MultiLoraBatchConfig {
            max_rank_work: 16,
            ..config
        };
        assert!(!window.can_accept(
            &batch_request_with_rank("session", &[2], 1, &[12], 16),
            tight_config
        ));
        let mut guarded = batch_request("session", &[2], 1, &[12]);
        guarded.expected_steps = vec![0];
        assert!(!window.can_accept(&guarded, config));
        assert!(window.can_accept(&batch_request("session", &[2, 3], 2, &[12, 13]), config));
        window.push(batch_request("session", &[2], 1, &[12]));
        assert!(!window.can_accept(&batch_request("session", &[3], 1, &[13]), config));
    }

    #[test]
    fn coalescing_wait_wakes_when_current_window_is_sealed() {
        let config = MultiLoraBatchConfig {
            window: std::time::Duration::from_secs(2),
            max_requests: 4,
            max_adapters: 4,
            max_rank_work: 64,
            max_payload_bytes: usize::MAX,
        };
        let batcher = MultiLoraBatcher::new(config);
        let window_id = 7;
        batcher.state.lock().unwrap().current_window_id = Some(window_id);

        let waiting_batcher = batcher.clone();
        let (started, started_rx) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let start = std::time::Instant::now();
            started.send(()).unwrap();
            waiting_batcher.wait_for_window(window_id, config.window);
            start.elapsed()
        });
        started_rx.recv().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        batcher.seal_current();

        assert!(
            waiter.join().unwrap() < std::time::Duration::from_millis(500),
            "sealed coalescing window must not wait for its full deadline"
        );
    }

    #[test]
    fn coalescing_wait_wakes_when_window_reaches_capacity() {
        let config = MultiLoraBatchConfig {
            window: std::time::Duration::from_secs(2),
            max_requests: 1,
            max_adapters: 4,
            max_rank_work: 64,
            max_payload_bytes: usize::MAX,
        };
        let batcher = MultiLoraBatcher::new(config);
        let window_id = 11;
        {
            let mut state = batcher.state.lock().unwrap();
            state.current_window_id = Some(window_id);
            state.windows.insert(
                window_id,
                MultiLoraWindow::new(batch_request("session", &[1], 1, &[11])),
            );
        }

        let waiting_batcher = batcher.clone();
        let (started, started_rx) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let start = std::time::Instant::now();
            started.send(()).unwrap();
            waiting_batcher.wait_for_window(window_id, config.window);
            start.elapsed()
        });
        started_rx.recv().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        {
            let mut state = batcher.state.lock().unwrap();
            batcher.seal_window_if_at_capacity(&mut state, window_id);
        }

        assert!(
            waiter.join().unwrap() < std::time::Duration::from_millis(500),
            "full coalescing window must not wait for its full deadline"
        );
        assert_eq!(batcher.state.lock().unwrap().current_window_id, None);
    }

    #[test]
    fn coalescing_wait_keeps_the_deadline_for_an_open_window() {
        let config = MultiLoraBatchConfig {
            window: std::time::Duration::from_millis(30),
            max_requests: 4,
            max_adapters: 4,
            max_rank_work: 64,
            max_payload_bytes: usize::MAX,
        };
        let batcher = MultiLoraBatcher::new(config);
        let window_id = 13;
        batcher.state.lock().unwrap().current_window_id = Some(window_id);

        let start = std::time::Instant::now();
        batcher.wait_for_window(window_id, config.window);
        assert!(
            start.elapsed() >= std::time::Duration::from_millis(20),
            "open coalescing window must retain its normal batching deadline"
        );
        assert_eq!(batcher.state.lock().unwrap().current_window_id, None);
    }

    #[test]
    fn implicit_registry_request_is_an_exclusive_window() {
        let config = MultiLoraBatchConfig {
            window: std::time::Duration::from_millis(1),
            max_requests: 4,
            max_adapters: 4,
            max_rank_work: 64,
            max_payload_bytes: usize::MAX,
        };
        let window = MultiLoraWindow::new(batch_request("session", &[1], 1, &[]));
        assert!(!window.can_accept(&batch_request("session", &[2], 1, &[12]), config));
    }

    #[test]
    fn request_local_loss_opt_out_is_an_exclusive_window() {
        let config = MultiLoraBatchConfig {
            window: std::time::Duration::from_millis(1),
            max_requests: 4,
            max_adapters: 4,
            max_rank_work: 64,
            max_payload_bytes: usize::MAX,
        };
        let mut request = batch_request("session", &[1], 1, &[11]);
        request.allow_aggregate_loss = false;
        let window = MultiLoraWindow::new(request);
        assert!(!window.can_accept(&batch_request("session", &[2], 1, &[12]), config));
    }

    #[test]
    fn int64_decode_rejects_wrong_dtype_and_trailing_bytes() {
        let mut value = tensor(&[1]);
        value.dtype = "float32".into();
        value.data = general_purpose::STANDARD.encode(1_i64.to_le_bytes());
        assert!(decode_int64_bytes(&value).is_err());

        value.dtype = "int64".into();
        value.data = general_purpose::STANDARD.encode([0_u8; 9]);
        assert!(decode_int64_bytes(&value).is_err());
    }

    #[test]
    fn tensor_slab_pack_aligns_spans_and_preserves_wire_bytes() {
        let wire = [1_i64, -2_i64]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>();
        let encoded = general_purpose::STANDARD.encode(&wire);
        let make = || TensorHttp {
            data: encoded.clone(),
            shape: vec![1, 2],
            dtype: "int64".into(),
        };
        let (reference, payload) = pack_tensor_slab(&make(), &make(), &make(), 1, 2).unwrap();
        for span in reference.spans() {
            assert_eq!(span.offset_bytes % 64, 0);
            let start = span.offset_bytes as usize;
            let end = start + span.len_bytes as usize;
            assert_eq!(&payload[start..end], wire);
        }
        reference.validate(payload.len()).unwrap();
    }

    #[test]
    fn dispatch_pressure_and_worker_failure_have_distinct_statuses() {
        assert_eq!(
            ep_dispatch_schedule_error(EpDispatchScheduleError::QueueFull).0,
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            ep_dispatch_schedule_error(EpDispatchScheduleError::QueueClosed).0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            ep_dispatch_schedule_error(EpDispatchScheduleError::WorkerFailed).0,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}

async fn ep_add_lora(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
    Json(req): Json<AddLoRAHttp>,
) -> Result<Json<AddLoRAResponse>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::AddLora {
        session_id: id,
        rank: req.rank,
        alpha: req.alpha,
        target_layers: req.target_layers,
        target_modules: req.target_modules,
        optimizer_lr: req.optimizer_lr,
        optimizer_beta1: req.optimizer_beta1,
        optimizer_beta2: req.optimizer_beta2,
        optimizer_eps: req.optimizer_eps,
    };
    match dispatch_ep(&state, cmd).await? {
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
    #[serde(default)]
    optimizer_lr: Option<f64>,
    #[serde(default)]
    optimizer_beta1: Option<f64>,
    #[serde(default)]
    optimizer_beta2: Option<f64>,
    #[serde(default)]
    optimizer_eps: Option<f64>,
}

async fn ep_batch_add_lora(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
    Json(req): Json<BatchAddLoRAHttp>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    crate::ep::validate_batch_add_lora_count(req.count).map_err(|error| err_resp(&error))?;
    let cmd = rustrain_ipc::EpCommand::BatchAddLora {
        session_id: id,
        count: req.count,
        rank: req.rank,
        alpha: req.alpha,
        target_layers: req.target_layers,
        target_modules: req.target_modules,
        optimizer_lr: req.optimizer_lr,
        optimizer_beta1: req.optimizer_beta1,
        optimizer_beta2: req.optimizer_beta2,
        optimizer_eps: req.optimizer_eps,
    };
    match dispatch_ep(&state, cmd).await? {
        rustrain_ipc::EpResult::Count(n) => Ok(Json(serde_json::json!({"count": n}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_remove_lora(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
    Json(req): Json<RemoveLoRAHttp>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::RemoveLora {
        session_id: id,
        adapter_id: req.adapter_id,
    };
    match dispatch_ep(&state, cmd).await? {
        rustrain_ipc::EpResult::Ok => Ok(Json(serde_json::json!({"removed": true}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_list_lora(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::ListLora { session_id: id };
    match dispatch_ep(&state, cmd).await? {
        rustrain_ipc::EpResult::AdapterIds(ids) => Ok(Json(serde_json::json!({"adapters": ids}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_export_adapter(
    State(state): State<Arc<EpRouterState>>,
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
    match dispatch_ep(&state, cmd).await? {
        rustrain_ipc::EpResult::Count(n) => Ok(Json(serde_json::json!({"exported": n}))),
        rustrain_ipc::EpResult::Error(e) => Err(err_resp(&e)),
        _ => Err(err_resp("unexpected result")),
    }
}

async fn ep_save_checkpoint(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
    Json(req): Json<CheckpointHttp>,
) -> Result<Json<CheckpointResponse>, (StatusCode, Json<ErrorResponse>)> {
    let path = req.path;
    let response_path = path.clone();
    if !state.coordinator.is_healthy() {
        return Err(ep_dispatch_unavailable("EP coordinator is unavailable"));
    }
    let coordinator = Arc::clone(&state.coordinator);
    let receiver = {
        let _submission = state
            .dispatch_submission
            .lock()
            .map_err(|_| ep_dispatch_schedule_error(EpDispatchScheduleError::WorkerFailed))?;
        state.multi_lora_batcher.seal_current();
        state
            .dispatcher
            .submit(move || coordinator.coordinated_save_checkpoint(&id, &path))
            .map_err(ep_dispatch_schedule_error)?
    };
    let (step, loss) = receiver
        .await
        .map_err(|_| ep_dispatch_schedule_error(EpDispatchScheduleError::WorkerFailed))?
        .map_err(|error| err_resp(&error))?;
    Ok(Json(CheckpointResponse {
        step,
        loss,
        path: response_path,
    }))
}

async fn ep_load_checkpoint(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
    Json(req): Json<CheckpointHttp>,
) -> Result<Json<CheckpointResponse>, (StatusCode, Json<ErrorResponse>)> {
    let path = req.path;
    let response_path = path.clone();
    if !state.coordinator.is_healthy() {
        return Err(ep_dispatch_unavailable("EP coordinator is unavailable"));
    }
    let coordinator = Arc::clone(&state.coordinator);
    let receiver = {
        let _submission = state
            .dispatch_submission
            .lock()
            .map_err(|_| ep_dispatch_schedule_error(EpDispatchScheduleError::WorkerFailed))?;
        state.multi_lora_batcher.seal_current();
        state
            .dispatcher
            .submit(move || coordinator.coordinated_load_checkpoint(&id, &path))
            .map_err(ep_dispatch_schedule_error)?
    };
    let (step, loss) = receiver
        .await
        .map_err(|_| ep_dispatch_schedule_error(EpDispatchScheduleError::WorkerFailed))?
        .map_err(|error| err_resp(&error))?;
    Ok(Json(CheckpointResponse {
        step,
        loss,
        path: response_path,
    }))
}

async fn ep_get_status(
    State(state): State<Arc<EpRouterState>>,
    Path(id): Path<String>,
) -> Result<Json<StatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let cmd = rustrain_ipc::EpCommand::Status { session_id: id };
    match dispatch_ep(&state, cmd).await? {
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
