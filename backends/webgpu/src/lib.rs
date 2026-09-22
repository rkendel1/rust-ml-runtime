use async_trait::async_trait;
use futures_util::stream;
use ml_runtime_backend::{Backend, BackendCapabilities, BackendCapability};
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::{InferenceChunk, InferenceRequest, InferenceResult};
use ml_runtime_model::{ModelFormat, ModelHandle, ModelSpec};

#[derive(Clone, Debug, Default)]
pub struct WebgpuBackend;

impl WebgpuBackend {
    pub fn descriptor() -> BackendCapability {
        BackendCapability::from_parts("webgpu", Self.capabilities())
    }
}

#[async_trait]
impl Backend for WebgpuBackend {
    fn name(&self) -> &str {
        "webgpu"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            available: false,
            local: true,
            streaming: false,
            cancellation: false,
            structured_output: false,
            batching: false,
            max_batch_size: None,
            supported_batch_shapes: Vec::new(),
            supported_formats: vec![ModelFormat::Unknown],
            accelerators: vec!["webgpu".to_owned()],
            hardware: Some("webgpu".to_owned()),
            notes: vec!["Architecture boundary established for webgpu integration".to_owned()],
        }
    }

    fn supports(&self, _model: &ModelSpec) -> bool {
        false
    }

    async fn load(&self, _model: &ModelSpec) -> RuntimeResult<ModelHandle> {
        Err(RuntimeError::backend_unavailable(
            self.name(),
            "backend is not yet implemented",
        ))
    }

    async fn infer(
        &self,
        _model: &ModelHandle,
        _request: &InferenceRequest,
        _cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<InferenceResult> {
        Err(RuntimeError::backend_unavailable(
            self.name(),
            "backend is not yet implemented",
        ))
    }

    async fn infer_stream(
        &self,
        _model: &ModelHandle,
        _request: &InferenceRequest,
        _cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>> {
        Ok(Box::pin(stream::iter(vec![Err(
            RuntimeError::CapabilityMismatch {
                reason: "streaming is not available for the webgpu scaffold backend".to_owned(),
            },
        )])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn descriptor_matches_boundary() {
        assert_eq!(WebgpuBackend::descriptor().name, "webgpu");
    }
}
