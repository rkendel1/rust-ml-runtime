use async_trait::async_trait;
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeResult};
use ml_runtime_inference::{InferenceChunk, InferenceRequest, InferenceResult};
use ml_runtime_model::{ModelFormat, ModelHandle, ModelSpec};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub available: bool,
    pub local: bool,
    pub streaming: bool,
    pub cancellation: bool,
    pub structured_output: bool,
    pub batching: bool,
    pub supported_formats: Vec<ModelFormat>,
    pub accelerators: Vec<String>,
    pub hardware: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendCapability {
    pub name: String,
    pub available: bool,
    pub local: bool,
    pub streaming: bool,
    pub cancellation: bool,
    pub structured_output: bool,
    pub batching: bool,
    pub supported_formats: Vec<ModelFormat>,
    pub accelerators: Vec<String>,
    pub hardware: Option<String>,
    pub notes: Vec<String>,
}

impl BackendCapability {
    pub fn from_parts(name: impl Into<String>, capabilities: BackendCapabilities) -> Self {
        Self {
            name: name.into(),
            available: capabilities.available,
            local: capabilities.local,
            streaming: capabilities.streaming,
            cancellation: capabilities.cancellation,
            structured_output: capabilities.structured_output,
            batching: capabilities.batching,
            supported_formats: capabilities.supported_formats,
            accelerators: capabilities.accelerators,
            hardware: capabilities.hardware,
            notes: capabilities.notes,
        }
    }
}

#[async_trait]
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> BackendCapabilities;
    fn supports(&self, model: &ModelSpec) -> bool;
    async fn load(&self, model: &ModelSpec) -> RuntimeResult<ModelHandle>;
    async fn infer(
        &self,
        model: &ModelHandle,
        request: &InferenceRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<InferenceResult>;
    async fn infer_stream(
        &self,
        model: &ModelHandle,
        request: &InferenceRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>>;
}
