use async_trait::async_trait;
use futures_util::stream;
use ml_runtime_backend::{Backend, BackendCapabilities, BackendCapability};
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::{InferenceChunk, InferenceRequest, InferenceResult};
use ml_runtime_model::{ModelFormat, ModelHandle, ModelSpec};

#[derive(Clone, Debug, Default)]
pub struct CoreMlBackend;

impl CoreMlBackend {
    pub fn descriptor() -> BackendCapability {
        BackendCapability::from_parts("coreml", Self::default().capabilities())
    }
}

#[async_trait]
impl Backend for CoreMlBackend {
    fn name(&self) -> &str {
        "coreml"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            available: cfg!(target_os = "macos"),
            local: true,
            streaming: false,
            cancellation: false,
            structured_output: false,
            batching: false,
            supported_formats: vec![ModelFormat::CoreMl],
            accelerators: vec!["ane".to_owned(), "gpu".to_owned(), "cpu".to_owned()],
            hardware: Some("coreml".to_owned()),
            notes: vec![
                "Architecture boundary established for Apple-native acceleration".to_owned(),
            ],
        }
    }

    fn supports(&self, model: &ModelSpec) -> bool {
        matches!(model.format, ModelFormat::CoreMl)
    }

    async fn load(&self, model: &ModelSpec) -> RuntimeResult<ModelHandle> {
        Err(RuntimeError::backend_unavailable(
            self.name(),
            format!("{} support is not implemented in this scaffold", model.id),
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
            "inference path is not implemented",
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
                reason: "streaming is not available for the coreml scaffold backend".to_owned(),
            },
        )])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn descriptor_matches_coreml_boundary() {
        assert_eq!(CoreMlBackend::descriptor().name, "coreml");
    }
}
