use async_trait::async_trait;
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::{InferenceChunk, InferenceRequest, InferenceResult};
use ml_runtime_provider::{Provider, ProviderCapabilities, ProviderCapability};
use tokio_stream::iter;

#[derive(Clone, Debug, Default)]
pub struct LocalProvider;

impl LocalProvider {
    pub fn descriptor() -> ProviderCapability {
        ProviderCapability::from_parts("local", Self::default().capabilities())
    }
}

#[async_trait]
impl Provider for LocalProvider {
    fn name(&self) -> &str {
        "local"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            available: true,
            local: true,
            remote: false,
            streaming: true,
            cancellation: true,
            structured_output: true,
            batching: true,
            endpoint: None,
            notes: vec!["Local execution is coordinated directly by the Rust runtime".to_owned()],
        }
    }

    async fn infer(
        &self,
        _request: InferenceRequest,
        _cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<InferenceResult> {
        Err(RuntimeError::provider_unavailable(
            self.name(),
            "local execution is handled directly by Runtime using registered backends",
        ))
    }

    async fn infer_stream(
        &self,
        _request: InferenceRequest,
        _cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>> {
        Ok(Box::pin(iter(vec![Err(
            RuntimeError::provider_unavailable(
                self.name(),
                "local streaming is handled directly by Runtime using registered backends",
            ),
        )])))
    }
}
