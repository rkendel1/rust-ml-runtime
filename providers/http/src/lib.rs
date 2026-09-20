use async_trait::async_trait;
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::{InferenceChunk, InferenceRequest, InferenceResult};
use ml_runtime_provider::{Provider, ProviderCapabilities, ProviderCapability};
use tokio_stream::iter;

#[derive(Clone, Debug)]
pub struct HttpProvider {
    endpoint: String,
}

impl HttpProvider {
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
            notes: vec![
                "HTTP provider boundary established; transport integration remains pluggable"
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
                "HTTP transport is not implemented for endpoint {}",
                self.endpoint
            ),
        ))
    }

    async fn infer_stream(
        &self,
        _request: InferenceRequest,
        _cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>> {
        Ok(Box::pin(iter(vec![Err(
            RuntimeError::CapabilityMismatch {
                reason: "HTTP streaming is not implemented in this scaffold provider".to_owned(),
            },
        )])))
    }
}
