//! gRPC service (tonic) — TrainService implementation.

use tonic::{Request, Response, Status};

use crate::session::{InitLoRARequest, TrainInput};
use crate::state::SessionManager;

pub mod train {
    tonic::include_proto!("rustrain.train.v1");
}

use train::train_service_server::TrainService;
use train::*;

pub struct TrainServiceImpl {
    pub manager: std::sync::Arc<SessionManager>,
}

#[tonic::async_trait]
impl TrainService for TrainServiceImpl {
    async fn load_model(
        &self,
        request: Request<train::LoadModelRequest>,
    ) -> Result<Response<LoadModelResponse>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;
        let mut s = session.lock().await;
        s.load_model(crate::session::SessLoadModelRequest {
            model_path: req.model_path,
            config_toml: req.config_toml,
        })
        .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(LoadModelResponse {
            model_id: "loaded".into(),
        }))
    }

    async fn load_dataset(
        &self,
        request: Request<train::LoadDatasetRequest>,
    ) -> Result<Response<LoadDatasetResponse>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;
        let mut s = session.lock().await;
        let n = s
            .load_dataset(crate::session::SessLoadDatasetRequest {
                jsonl_path: req.jsonl_path,
                seq_len: req.seq_len as usize,
            })
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(LoadDatasetResponse {
            num_examples: n as i64,
        }))
    }

    async fn init_lo_ra(
        &self,
        request: Request<InitLoRaRequest>,
    ) -> Result<Response<InitLoRaResponse>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;
        let mut s = session.lock().await;
        let count = s
            .init_lora(InitLoRARequest {
                rank: req.rank,
                alpha: req.alpha,
                target_layers: req.target_layers.iter().map(|&l| l as usize).collect(),
                target_modules: req.target_modules,
                lr: req.lr,
                beta1: req.beta1,
                beta2: req.beta2,
                eps: req.eps,
            })
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(InitLoRaResponse {
            lora_param_count: count as i64,
        }))
    }

    async fn train_step(
        &self,
        request: Request<TrainStepRequest>,
    ) -> Result<Response<TrainStepResponse>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;

        let input_ids = decode_tensor_data(&req.input_ids)?;
        let target_mask = decode_tensor_data(&req.target_mask)?;
        let attention_mask = decode_tensor_data(&req.attention_mask)?;

        let mut s = session.lock().await;
        let result = s
            .train_step(TrainInput {
                input_ids,
                target_mask,
                attention_mask,
            })
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(TrainStepResponse {
            loss: result.loss,
            step: result.step as i64,
        }))
    }

    async fn eval_step(
        &self,
        request: Request<EvalStepRequest>,
    ) -> Result<Response<EvalStepResponse>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;

        let input_ids = decode_tensor_data(&req.input_ids)?;
        let target_mask = decode_tensor_data(&req.target_mask)?;
        let attention_mask = decode_tensor_data(&req.attention_mask)?;

        let s = session.lock().await;
        let result = s
            .eval_step(TrainInput {
                input_ids,
                target_mask,
                attention_mask,
            })
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(EvalStepResponse { loss: result.loss }))
    }

    async fn save_checkpoint(
        &self,
        request: Request<SaveCheckpointRequest>,
    ) -> Result<Response<CheckpointInfo>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;
        let s = session.lock().await;
        let (step, loss) = s
            .save_checkpoint(&req.path)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(CheckpointInfo {
            step: step as i64,
            loss,
            path: req.path,
        }))
    }

    async fn load_checkpoint(
        &self,
        request: Request<LoadCheckpointRequest>,
    ) -> Result<Response<CheckpointInfo>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;
        let mut s = session.lock().await;
        let (step, loss) = s
            .load_checkpoint(&req.path)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(CheckpointInfo {
            step: step as i64,
            loss,
            path: req.path,
        }))
    }

    async fn export_adapter(
        &self,
        request: Request<ExportAdapterRequest>,
    ) -> Result<Response<AdapterInfo>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;
        let s = session.lock().await;
        let count = s
            .export_adapter(&req.path, Some(req.adapter_id))
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(AdapterInfo {
            path: req.path,
            param_count: count as i64,
        }))
    }

    async fn import_adapter(
        &self,
        request: Request<ImportAdapterRequest>,
    ) -> Result<Response<ImportAdapterResponse>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;
        let mut s = session.lock().await;
        let adapter_id = s
            .import_adapter(&req.path)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(ImportAdapterResponse { adapter_id }))
    }

    type StreamMetricsStream =
        std::pin::Pin<Box<dyn futures::Stream<Item = Result<StepMetric, Status>> + Send>>;

    async fn stream_metrics(
        &self,
        request: Request<StreamMetricsRequest>,
    ) -> Result<Response<Self::StreamMetricsStream>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;
        let s = session.lock().await;
        let metrics = s.get_metrics();
        drop(s);

        let stream = futures::stream::iter(metrics.into_iter().map(|m| {
            Ok(StepMetric {
                step: m.step as i64,
                loss: m.loss,
                lr: m.lr,
                mem_gb: m.mem_gb,
                timestamp_unix: m.timestamp_unix,
            })
        }));

        Ok(Response::new(Box::pin(stream)))
    }

    async fn get_status(
        &self,
        request: Request<GetStatusRequest>,
    ) -> Result<Response<SessionStatus>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;
        let s = session.lock().await;
        let status = s.status();
        Ok(Response::new(SessionStatus {
            state: status.state,
            step: status.step as i64,
            last_loss: status.last_loss,
            model_path: status.model_path,
        }))
    }

    async fn add_lo_ra(
        &self,
        request: Request<train::AddLoRaRequest>,
    ) -> Result<Response<train::AddLoRaResponse>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;
        let mut s = session.lock().await;
        let id = s
            .add_lora(crate::session::AddLoRARequest {
                rank: req.rank,
                alpha: req.alpha,
                target_layers: req.target_layers,
                target_modules: req.target_modules,
            })
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(train::AddLoRaResponse { adapter_id: id }))
    }

    async fn remove_lo_ra(
        &self,
        request: Request<train::RemoveLoRaRequest>,
    ) -> Result<Response<train::RemoveLoRaResponse>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;
        let mut s = session.lock().await;
        let removed = s
            .remove_lora(req.adapter_id)
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(train::RemoveLoRaResponse { removed }))
    }

    async fn list_lo_ra(
        &self,
        request: Request<train::ListLoRaRequest>,
    ) -> Result<Response<train::ListLoRaResponse>, Status> {
        let req = request.into_inner();
        let session = self
            .manager
            .get_session(&req.session_id)
            .await
            .ok_or_else(|| Status::not_found("session not found"))?;
        let s = session.lock().await;
        Ok(Response::new(train::ListLoRaResponse {
            adapter_ids: s.list_lora(),
        }))
    }
}

fn decode_tensor_data(td: &Option<TensorData>) -> Result<tch::Tensor, Status> {
    let td = td
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("missing tensor data"))?;
    let kind = match td.dtype.as_str() {
        "int64" => tch::Kind::Int64,
        "float32" => tch::Kind::Float,
        "bfloat16" => tch::Kind::BFloat16,
        _ => {
            return Err(Status::invalid_argument(format!(
                "unsupported dtype: {}",
                td.dtype
            )));
        }
    };
    let tensor = match kind {
        tch::Kind::Int64 => {
            let vals: Vec<i64> = td
                .data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            tch::Tensor::from_slice(&vals)
        }
        tch::Kind::Float => {
            let vals: Vec<f32> = td
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            tch::Tensor::from_slice(&vals)
        }
        _ => return Err(Status::invalid_argument("only int64 and float32 supported")),
    };
    let local_rank = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    Ok(tensor
        .reshape(&td.shape)
        .to_device(tch::Device::Cuda(local_rank)))
}
