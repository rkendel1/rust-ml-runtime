use async_trait::async_trait;
use axum::body::Body;
use axum::extract::DefaultBodyLimit;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use bytes::Bytes;
use futures_util::StreamExt;
use ml_runtime::Runtime;
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::{ExecutionPolicy, InferenceChunk, InferenceRequest, InferenceResult};
use ml_runtime_model::{FilesystemModelCatalog, ModelCatalog};
use ml_runtime_protocol::{
    CapabilitiesResponse, ErrorEnvelope, HealthResponse, InferRequest, InferResponse, ModelInfo,
    StreamInferResponse,
};
use ml_runtime_provider::{Provider, ProviderCapabilities, ProviderCapability};
use std::{collections::BTreeMap, convert::Infallible, path::Path, sync::Arc};
use tokio_stream::iter;

#[derive(Clone, Debug)]
pub struct ServerProvider {
    endpoint: String,
}

impl ServerProvider {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
        }
    }
    pub fn descriptor(&self) -> ProviderCapability {
        ProviderCapability::from_parts(
            "server",
            ProviderCapabilities {
                available: true,
                local: false,
                remote: true,
                streaming: false,
                cancellation: true,
                structured_output: true,
                batching: true,
                endpoint: Some(self.endpoint.clone()),
                notes: vec![],
            },
        )
    }
}

#[async_trait]
impl Provider for ServerProvider {
    fn name(&self) -> &str {
        "server"
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            available: true,
            local: false,
            remote: true,
            streaming: false,
            cancellation: true,
            structured_output: true,
            batching: true,
            endpoint: Some(self.endpoint.clone()),
            notes: vec![],
        }
    }
    async fn infer(
        &self,
        _request: InferenceRequest,
        _cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<InferenceResult> {
        Err(RuntimeError::transport(
            self.name(),
            "use HttpProvider for HTTP inference",
        ))
    }
    async fn infer_stream(
        &self,
        _request: InferenceRequest,
        _cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>> {
        Ok(Box::pin(iter(vec![Err(RuntimeError::transport(
            self.name(),
            "server provider is a protocol server",
        ))])))
    }
}

#[derive(Clone)]
struct ServerState {
    runtime: Arc<Runtime>,
    models: Arc<Vec<ModelInfo>>,
}

pub async fn serve(
    mut runtime: Runtime,
    models_dir: impl AsRef<Path>,
    bind: impl AsRef<str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let catalog = FilesystemModelCatalog::new(models_dir.as_ref().to_path_buf());
    let descriptors = catalog.list().await?;
    let models = descriptors
        .iter()
        .map(|descriptor| ModelInfo {
            id: descriptor.id.name.clone(),
            version: Some(descriptor.id.version.clone()),
            format: descriptor.format.clone(),
            inputs: descriptor.manifest.inputs.clone(),
            outputs: descriptor.manifest.outputs.clone(),
            metadata: descriptor.manifest.metadata.clone(),
        })
        .collect();
    runtime.set_catalog(catalog);
    let app = Router::new()
        .route("/v1/health", get(health))
        .route("/v1/capabilities", get(capabilities))
        .route("/v1/models", get(model_list))
        .route("/v1/infer", post(infer))
        .route("/v1/infer/stream", post(infer_stream))
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
        .with_state(ServerState {
            runtime: Arc::new(runtime),
            models: Arc::new(models),
        });
    let listener = tokio::net::TcpListener::bind(bind.as_ref()).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_owned(),
    })
}

async fn capabilities(State(state): State<ServerState>) -> Json<CapabilitiesResponse> {
    let capabilities = state.runtime.capabilities();
    let streaming = capabilities
        .backends
        .iter()
        .any(|backend| backend.streaming);
    let mut formats = capabilities
        .backends
        .iter()
        .flat_map(|b| b.supported_formats.clone())
        .collect::<Vec<_>>();
    formats.sort_by_key(|f| format!("{f:?}"));
    formats.dedup();
    Json(CapabilitiesResponse {
        runtime_version: env!("CARGO_PKG_VERSION").to_owned(),
        supported_model_formats: formats,
        available_providers: capabilities
            .providers
            .into_iter()
            .filter(|p| p.available)
            .map(|p| p.name)
            .collect(),
        available_backends: capabilities
            .backends
            .into_iter()
            .filter(|b| b.available)
            .map(|b| b.name)
            .collect(),
        supported_tensor_types: vec!["f32".to_owned()],
        streaming,
        metadata: BTreeMap::new(),
    })
}

async fn infer_stream(
    State(state): State<ServerState>,
    Json(mut payload): Json<InferRequest>,
) -> Response {
    let valid_reference = matches!(
        &payload.request.model,
        ml_runtime_model::ModelReference::Id { id, .. }
            if !id.is_empty() && !id.contains('/') && !id.contains('\\') && !id.contains("..")
    );
    if !valid_reference {
        let error = ErrorEnvelope {
            code: "invalid_request".to_owned(),
            message: "remote streaming requires a valid model id and version".to_owned(),
            retryable: false,
        };
        return (
            StatusCode::BAD_REQUEST,
            Json(StreamInferResponse::error(error)),
        )
            .into_response();
    }
    normalize_client_policy(&mut payload.request);
    let stream = match state.runtime.infer_stream(payload.request).await {
        Ok(stream) => stream,
        Err(error) => {
            let envelope = error_envelope(&error);
            return (
                status_for(&envelope.code),
                Json(StreamInferResponse::error(envelope)),
            )
                .into_response();
        }
    };
    let body = stream.map(|item| {
        let response = match item {
            Ok(event) => StreamInferResponse::event(event),
            Err(error) => StreamInferResponse::error(error_envelope(&error)),
        };
        let mut bytes = serde_json::to_vec(&response).expect("stream response serializes");
        bytes.push(b'\n');
        Ok::<Bytes, Infallible>(Bytes::from(bytes))
    });
    Body::from_stream(body).into_response()
}

async fn model_list(State(state): State<ServerState>) -> Json<Vec<ModelInfo>> {
    Json((*state.models).clone())
}

async fn infer(
    State(state): State<ServerState>,
    Json(mut payload): Json<InferRequest>,
) -> (StatusCode, Json<InferResponse>) {
    let valid_reference = match &payload.request.model {
        ml_runtime_model::ModelReference::Id { id, .. } => {
            !id.is_empty() && !id.contains('/') && !id.contains('\\') && !id.contains("..")
        }
        ml_runtime_model::ModelReference::Spec(_) => false,
    };
    if !valid_reference {
        let error = ErrorEnvelope {
            code: "invalid_request".to_owned(),
            message: "remote inference requires a valid model id and version".to_owned(),
            retryable: false,
        };
        return (StatusCode::BAD_REQUEST, Json(InferResponse::error(error)));
    }
    normalize_client_policy(&mut payload.request);
    match state.runtime.infer(payload.request).await {
        Ok(result) => (StatusCode::OK, Json(InferResponse::success(result))),
        Err(error) => {
            let envelope = error_envelope(&error);
            (
                status_for(&envelope.code),
                Json(InferResponse::error(envelope)),
            )
        }
    }
}

fn normalize_client_policy(request: &mut InferenceRequest) {
    if request.options.require_remote
        || matches!(request.options.execution, ExecutionPolicy::RemoteOnly)
    {
        request.options.require_remote = false;
        request.options.execution = ExecutionPolicy::LocalOnly;
    }
}

fn error_envelope(error: &RuntimeError) -> ErrorEnvelope {
    let (code, retryable) = match error {
        RuntimeError::InvalidRequest { .. } => ("invalid_request", false),
        RuntimeError::ModelNotFound { .. } => ("model_not_found", false),
        RuntimeError::ModelUnavailable { .. } => ("model_unavailable", true),
        RuntimeError::ModelIntegrity { .. } => ("model_integrity", false),
        RuntimeError::InvalidInput { .. } | RuntimeError::UnsupportedInput { .. } => {
            ("invalid_input", false)
        }
        RuntimeError::Timeout { .. } => ("timeout", true),
        RuntimeError::Cancelled => ("request_cancelled", false),
        RuntimeError::BackendUnavailable { .. } => ("backend_unavailable", true),
        RuntimeError::UnsupportedModel { .. } => ("unsupported_model_format", false),
        RuntimeError::ProviderUnavailable { .. } => ("model_unavailable", true),
        RuntimeError::Protocol { .. } => ("protocol_error", false),
        _ => ("inference_failed", false),
    };
    ErrorEnvelope {
        code: code.to_owned(),
        message: error.to_string(),
        retryable,
    }
}

fn status_for(code: &str) -> StatusCode {
    match code {
        "model_not_found" => StatusCode::NOT_FOUND,
        "invalid_request" | "invalid_input" => StatusCode::BAD_REQUEST,
        "model_integrity" => StatusCode::UNPROCESSABLE_ENTITY,
        "model_unavailable" | "backend_unavailable" => StatusCode::SERVICE_UNAVAILABLE,
        "timeout" => StatusCode::GATEWAY_TIMEOUT,
        "protocol_error" => StatusCode::BAD_GATEWAY,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}
