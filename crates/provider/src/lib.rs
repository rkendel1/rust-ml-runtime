use async_trait::async_trait;
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeResult};
use ml_runtime_inference::{InferenceChunk, InferenceRequest, InferenceResult};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderCapabilities {
    pub available: bool,
    pub local: bool,
    pub remote: bool,
    pub streaming: bool,
    pub cancellation: bool,
    pub structured_output: bool,
    pub batching: bool,
    pub endpoint: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderCapability {
    pub name: String,
    pub available: bool,
    pub local: bool,
    pub remote: bool,
    pub streaming: bool,
    pub cancellation: bool,
    pub structured_output: bool,
    pub batching: bool,
    pub endpoint: Option<String>,
    pub notes: Vec<String>,
}

impl ProviderCapability {
    pub fn from_parts(name: impl Into<String>, capabilities: ProviderCapabilities) -> Self {
        Self {
            name: name.into(),
            available: capabilities.available,
            local: capabilities.local,
            remote: capabilities.remote,
            streaming: capabilities.streaming,
            cancellation: capabilities.cancellation,
            structured_output: capabilities.structured_output,
            batching: capabilities.batching,
            endpoint: capabilities.endpoint,
            notes: capabilities.notes,
        }
    }
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> ProviderCapabilities;
    async fn infer(
        &self,
        request: InferenceRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<InferenceResult>;
    async fn infer_stream(
        &self,
        request: InferenceRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>>;
}
