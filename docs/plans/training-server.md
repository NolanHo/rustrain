# Plan: Training Platform Server (Step Mode)

## Summary

新建 `rustrain-server` crate，提供 HTTP + gRPC 双栈 API。Step 模式：`load_model → init_lora → train_step(loop) → export_adapter`。同进程，GPU 操作用 `spawn_blocking`。Metrics trait 抽象，默认写 JSONL 文件。预留 `session_id` 多任务接口。

## 架构

```
┌──────────────────────────────────────────────────┐
│                rustrain-server (单进程)            │
│                                                   │
│  axum (HTTP:8080)    tonic (gRPC:50051)           │
│       ↓                    ↓                      │
│       └────  TrainApi (共享) ──┘                   │
│                    ↓                              │
│         SessionManager (HashMap<session_id, Session>)│
│                    ↓                              │
│            Qwen36Session                          │
│            ├─ CppTrainingContext (GPU)             │
│            ├─ SftDataset                           │
│            └─ FileMetricsSink                     │
│                    ↓                              │
│            tokio::task::spawn_blocking(train_step) │
└──────────────────────────────────────────────────┘
```

## Tensor 传输格式

同进程零拷贝。HTTP/gRPC 用 binary protobuf：
```protobuf
message TensorData {
  bytes data = 1;              // raw bytes
  repeated int64 shape = 2;
  string dtype = 3;            // "int64", "bfloat16", "float32"
}
```
Rust 侧 `tch::Tensor::from_blob()` 直接构造 GPU tensor，传给 C++ kernel。

## gRPC Proto

```protobuf
service TrainService {
  rpc LoadModel(LoadModelRequest) returns (LoadModelResponse);
  rpc LoadDataset(LoadDatasetRequest) returns (LoadDatasetResponse);
  rpc InitLoRA(InitLoRARequest) returns (InitLoRAResponse);
  rpc TrainStep(TrainStepRequest) returns (TrainStepResponse);
  rpc EvalStep(EvalStepRequest) returns (EvalStepResponse);
  rpc SaveCheckpoint(SaveCheckpointRequest) returns (CheckpointInfo);
  rpc LoadCheckpoint(LoadCheckpointRequest) returns (CheckpointInfo);
  rpc ExportAdapter(ExportAdapterRequest) returns (AdapterInfo);
  rpc StreamMetrics(StreamMetricsRequest) returns (stream StepMetric);
  rpc GetStatus(GetStatusRequest) returns (SessionStatus);
}
```

## HTTP (axum)

```
POST   /v1/sessions                    创建 session → { session_id }
POST   /v1/sessions/:id/load_model      body: { model_path, config }
POST   /v1/sessions/:id/load_dataset     body: { jsonl_path, seq_len }
POST   /v1/sessions/:id/init_lora        body: { rank, alpha, target_layers, target_modules, lr, ... }
POST   /v1/sessions/:id/train_step       body: binary protobuf (TensorData × 3)
POST   /v1/sessions/:id/eval_step        body: binary protobuf
POST   /v1/sessions/:id/save_checkpoint  body: { path }
POST   /v1/sessions/:id/load_checkpoint  body: { path }
POST   /v1/sessions/:id/export_adapter   body: { path }
GET    /v1/sessions/:id/metrics          SSE stream
GET    /v1/sessions/:id/status
GET    /v1/sessions                      list all
```

## Rust Traits

```rust
pub enum SessionState {
    Unloaded, Loaded, Ready, Training, Paused, Error(String),
}

pub trait TrainingSession: Send {
    fn load_model(&mut self, req: LoadModelRequest) -> Result<()>;
    fn load_dataset(&mut self, req: LoadDatasetRequest) -> Result<()>;
    fn init_lora(&mut self, req: InitLoRARequest) -> Result<()>;
    fn train_step(&mut self, req: TrainStepRequest) -> Result<TrainStepResponse>;
    fn eval_step(&self, req: EvalStepRequest) -> Result<EvalStepResponse>;
    fn save_checkpoint(&self, path: &str) -> Result<CheckpointInfo>;
    fn load_checkpoint(&mut self, path: &str) -> Result<CheckpointInfo>;
    fn export_adapter(&self, path: &str) -> Result<()>;
    fn status(&self) -> SessionStatus;
}

pub trait MetricsSink: Send + Sync {
    fn record_step(&self, metric: StepMetric);
    fn read_metrics(&self) -> Vec<StepMetric>;
}
```

## Checkpoint 格式

```
checkpoint_dir/
  manifest.json          # { format, step, loss, model_path, lora_config }
  adapter.safetensors    # LoRA A/B matrices
  optimizer.safetensors   # Adam m/v states (FP32)
```

## Changes

### 新建 `crates/rustrain-server/`
- `Cargo.toml` — axum, tonic, prost, tokio, tower
- `src/lib.rs` — re-exports
- `src/proto/train.proto` — gRPC 定义
- `src/api.rs` — HTTP routes (axum)
- `src/grpc.rs` — gRPC service (tonic)
- `src/session.rs` — TrainingSession trait + Qwen36Session 实现
- `src/metrics.rs` — MetricsSink trait + FileMetricsSink
- `src/checkpoint.rs` — checkpoint save/load
- `src/state.rs` — SessionManager + SessionState

### 修改 `crates/rustrain-qwen3-6/src/kernel.rs`
- `export_optimizer_state()` — 导出 Adam m/v
- `import_optimizer_state()` — 导入 Adam m/v
- `get_step_count()` — 返回当前 step

### 修改 `crates/rustrain-qwen3-6/kernels/qwen3_6_kernels.cpp`
- `qwen36_eval_step()` — 前向 + loss，不 backward/Adam
- `qwen36_export_optimizer_state()` — 导出 Adam m/v
- `qwen36_import_optimizer_state()` — 导入 Adam m/v
- `qwen36_get_step_count()` — 返回 step

### 修改 `src/main.rs`
- 新增 `server` 子命令

### 修改根 `Cargo.toml`
- 添加 rustrain-server 依赖 + axum/tonic/tokio 到 workspace

## Definition of Done

- [ ] proto 定义 + 代码生成
- [ ] TrainingSession trait + Qwen36Session 实现
- [ ] FileMetricsSink 实现
- [ ] Checkpoint save/load (adapter + optimizer + step)
- [ ] HTTP API: load_model → init_lora → train_step → export_adapter 全流程
- [ ] gRPC API: 同上
- [ ] C++ FFI: eval_step + export/import optimizer + get_step_count
- [ ] `rustrain server` 启动双栈 server
- [ ] cargo build --release 通过

## Open Questions

- 无（方案已确认：同进程、二进制 protobuf tensor、预留 session_id）
