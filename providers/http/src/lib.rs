use async_trait::async_trait;
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::{InferenceChunk, InferenceRequest, InferenceResult};
use ml_runtime_protocol::{CapabilitiesResponse, InferRequest, InferResponse, ModelInfo};
use ml_runtime_provider::{Provider, ProviderCapabilities, ProviderCapability};
use tokio_stream::iter;

#[derive(Clone, Debug)]
pub struct HttpProvider {
    endpoint: String,
    client: reqwest::Client,
}

impl HttpProvider {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into().trim_end_matches('/').to_owned(),
            client: reqwest::Client::new(),
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub async fn discover_capabilities(&self) -> RuntimeResult<CapabilitiesResponse> {
        self.get_json("/v1/capabilities").await
    }

    pub async fn models(&self) -> RuntimeResult<Vec<ModelInfo>> {
        self.get_json("/v1/models").await
    }

    async fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> RuntimeResult<T> {
        self.client
            .get(format!("{}{}", self.endpoint, path))
            .send()
            .await
            .map_err(|error| RuntimeError::transport(self.name(), error.to_string()))?
            .error_for_status()
            .map_err(|error| RuntimeError::transport(self.name(), error.to_string()))?
            .json()
            .await
            .map_err(|error| RuntimeError::transport(self.name(), error.to_string()))
    }

    pub fn descriptor(&self) -> ProviderCapability {
        ProviderCapability::from_parts(self.name(), self.capabilities())
    }
}

#[async_trait]
impl Provider for HttpProvider {
    fn name(&self) -> &str {
        "remote"
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
            notes: vec!["Versioned runtime HTTP protocol".to_owned()],
        }
    }

    async fn infer(
        &self,
        request: InferenceRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<InferenceResult> {
        let send = self
            .client
            .post(format!("{}/v1/infer", self.endpoint))
            .json(&InferRequest { request })
            .send();
        let response = if let Some(token) = cancellation {
            tokio::select! {
                _ = token.cancelled() => return Err(RuntimeError::Cancelled),
                response = send => response,
            }
        } else {
            send.await
        };
        let response =
            response.map_err(|error| RuntimeError::transport(self.name(), error.to_string()))?;
        let status = response.status();
        let body: InferResponse = response
            .json()
            .await
            .map_err(|error| RuntimeError::transport(self.name(), error.to_string()))?;
        if let Some(result) = body.result {
            return Ok(result);
        }
        let error = body
            .error
            .unwrap_or_else(|| ml_runtime_protocol::ErrorEnvelope {
                code: "internal_error".to_owned(),
                message: "invalid server response".to_owned(),
                retryable: false,
            });
        Err(map_error(status, error))
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>> {
        let result = self.infer(request, cancellation).await?;
        Ok(Box::pin(iter(vec![Ok(InferenceChunk {
            output: result.output,
            done: true,
            metadata: Some(result.metadata),
        })])))
    }
}

fn map_error(
    status: reqwest::StatusCode,
    error: ml_runtime_protocol::ErrorEnvelope,
) -> RuntimeError {
    match error.code.as_str() {
        "model_not_found" => RuntimeError::ModelNotFound {
            model: error.message,
        },
        "invalid_request" => RuntimeError::InvalidInput {
            reason: error.message,
        },
        "invalid_input" => RuntimeError::InvalidInput {
            reason: error.message,
        },
        "timeout" => RuntimeError::Timeout {
            operation: "remote inference".to_owned(),
        },
        "request_cancelled" => RuntimeError::Cancelled,
        "backend_unavailable" => RuntimeError::BackendUnavailable {
            backend: "remote".to_owned(),
            reason: error.message,
        },
        "model_unavailable" | "unsupported_model_format" => RuntimeError::CapabilityMismatch {
            reason: error.message,
        },
        _ => RuntimeError::transport("remote", format!("HTTP {status}: {}", error.message)),
    }
}
