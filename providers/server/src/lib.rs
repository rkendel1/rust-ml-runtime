use async_trait::async_trait;
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::{InferenceChunk, InferenceRequest, InferenceResult};
use ml_runtime_provider::{Provider, ProviderCapabilities, ProviderCapability};
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
        ProviderCapability::from_parts(self.name(), self.capabilities())
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
            streaming: true,
            cancellation: true,
            structured_output: true,
            batching: true,
            endpoint: Some(self.endpoint.clone()),
            notes: vec![
                "Self-hosted server provider boundary established for daemons and local sockets"
                    .to_owned(),
            ],
        }
    }

    async fn infer(
        &self,
        _request: InferenceRequest,
        _cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<InferenceResult> {
        Err(RuntimeError::transport(
            self.name(),
            format!(
                "Server transport is not implemented for endpoint {}",
                self.endpoint
            ),
        ))
    }

    async fn infer_stream(
        &self,
        _request: InferenceRequest,
        _cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>> {
        Ok(Box::pin(iter(vec![Err(RuntimeError::transport(
            self.name(),
            "server streaming transport is not implemented",
        ))])))
    }
}
