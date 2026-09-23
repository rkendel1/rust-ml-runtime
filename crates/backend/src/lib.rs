use async_trait::async_trait;
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeResult};
use ml_runtime_inference::{
    DecisionModelCapabilities, DecisionRequest, DecisionResult, ModelDescription,
};
use ml_runtime_inference::{InferenceChunk, InferenceRequest, InferenceResult};
use ml_runtime_model::{ModelFormat, ModelHandle, ModelSpec};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub available: bool,
    pub local: bool,
    pub streaming: bool,
    pub cancellation: bool,
    pub structured_output: bool,
    pub batching: bool,
    pub max_batch_size: Option<usize>,
    pub supported_batch_shapes: Vec<Vec<usize>>,
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
    pub max_batch_size: Option<usize>,
    pub supported_batch_shapes: Vec<Vec<usize>>,
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
            max_batch_size: capabilities.max_batch_size,
            supported_batch_shapes: capabilities.supported_batch_shapes,
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
    async fn infer_batch(
        &self,
        model: &ModelHandle,
        requests: &[InferenceRequest],
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<Vec<InferenceResult>> {
        let mut results = Vec::with_capacity(requests.len());
        for request in requests {
            results.push(self.infer(model, request, cancellation.clone()).await?);
        }
        Ok(results)
    }
    async fn infer_stream(
        &self,
        model: &ModelHandle,
        request: &InferenceRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>>;
}

/// Loaded model-neutral typed-decision model.
pub trait DecisionModel: Send + Sync {
    fn describe(&self) -> ModelDescription;
    fn capabilities(&self) -> DecisionModelCapabilities {
        DecisionModelCapabilities::conservative(self.describe().backend)
    }
    fn decide(&self, request: &DecisionRequest) -> RuntimeResult<DecisionResult>;
}

/// Backend extension capable of loading a local typed-decision artifact.
pub trait DecisionModelProvider: Send + Sync {
    fn name(&self) -> &str;
    fn supports_artifact(&self, artifact: &Path) -> bool;
    fn load_decision_model(&self, artifact: &Path) -> RuntimeResult<Box<dyn DecisionModel>>;
}
