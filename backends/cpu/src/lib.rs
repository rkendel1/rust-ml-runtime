use async_trait::async_trait;
use futures_util::stream;
use ml_runtime_backend::{Backend, BackendCapabilities, BackendCapability};
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::{
    ExecutionMetadata, InferenceChunk, InferenceRequest, InferenceResult, Input, Output,
};
use ml_runtime_model::{ModelFormat, ModelHandle, ModelSpec};
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct CpuBackend;

#[derive(Clone, Debug)]
struct CpuLoadedModel {
    spec: ModelSpec,
}

impl CpuBackend {
    pub fn descriptor() -> BackendCapability {
        BackendCapability::from_parts("cpu", Self::default().capabilities())
    }
}

#[async_trait]
impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            available: true,
            local: true,
            streaming: true,
            cancellation: true,
            structured_output: true,
            batching: true,
            supported_formats: vec![
                ModelFormat::Onnx,
                ModelFormat::CoreMl,
                ModelFormat::Gguf,
                ModelFormat::Safetensors,
                ModelFormat::TensorRt,
                ModelFormat::Unknown,
            ],
            accelerators: Vec::new(),
            hardware: Some("cpu".to_owned()),
            notes: vec!["Portable CI backend for runtime validation".to_owned()],
        }
    }

    fn supports(&self, _model: &ModelSpec) -> bool {
        true
    }

    async fn load(&self, model: &ModelSpec) -> RuntimeResult<ModelHandle> {
        Ok(ModelHandle::new(
            model.id.clone(),
            self.name(),
            model.format.clone(),
            Arc::new(CpuLoadedModel {
                spec: model.clone(),
            }),
        ))
    }

    async fn infer(
        &self,
        model: &ModelHandle,
        request: &InferenceRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<InferenceResult> {
        if cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(RuntimeError::Cancelled);
        }

        let loaded = model
            .state::<CpuLoadedModel>()
            .ok_or_else(|| RuntimeError::InvalidInput {
                reason: "model handle does not belong to the CPU backend".to_owned(),
            })?;

        let output = match &request.input {
            Input::Text(text) => Output::Text(text.clone()),
            Input::Tokens(tokens) => Output::Tokens(tokens.clone()),
            Input::Tensor(tensor) => Output::Tensor(tensor.clone()),
            Input::Binary(bytes) => Output::Binary(bytes.clone()),
            Input::Structured(value) => Output::Structured(value.clone()),
            Input::Image(_) | Input::Audio(_) => {
                return Err(RuntimeError::InvalidInput {
                    reason: "the CPU backend scaffold currently supports text, tokens, tensors, binary, and structured inputs".to_owned(),
                })
            }
        };

        Ok(InferenceResult {
            output,
            metadata: ExecutionMetadata {
                model: loaded.spec.id.clone(),
                provider: "local".to_owned(),
                backend: self.name().to_owned(),
                hardware: Some("cpu".to_owned()),
                ..ExecutionMetadata::default()
            },
        })
    }

    async fn infer_stream(
        &self,
        model: &ModelHandle,
        request: &InferenceRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>> {
        if cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(RuntimeError::Cancelled);
        }
        let loaded = model
            .state::<CpuLoadedModel>()
            .ok_or_else(|| RuntimeError::InvalidInput {
                reason: "model handle does not belong to the CPU backend".to_owned(),
            })?;

        let chunks = match &request.input {
            Input::Text(text) => text
                .split_whitespace()
                .enumerate()
                .map(|(index, part)| {
                    Ok(InferenceChunk {
                        output: Output::Text(part.to_owned()),
                        done: index + 1 == text.split_whitespace().count(),
                        metadata: Some(ExecutionMetadata {
                            model: loaded.spec.id.clone(),
                            provider: "local".to_owned(),
                            backend: self.name().to_owned(),
                            hardware: Some("cpu".to_owned()),
                            ..ExecutionMetadata::default()
                        }),
                    })
                })
                .collect::<Vec<_>>(),
            _ => vec![Ok(InferenceChunk {
                output: self.infer(model, request, cancellation).await?.output,
                done: true,
                metadata: Some(ExecutionMetadata {
                    model: loaded.spec.id.clone(),
                    provider: "local".to_owned(),
                    backend: self.name().to_owned(),
                    hardware: Some("cpu".to_owned()),
                    ..ExecutionMetadata::default()
                }),
            })],
        };

        Ok(Box::pin(stream::iter(chunks)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn advertises_cpu_capabilities() {
        let capabilities = CpuBackend::default().capabilities();
        assert!(capabilities.available);
        assert!(capabilities.local);
        assert!(capabilities.streaming);
    }

    #[tokio::test]
    async fn supports_round_trip_text_inference() {
        let backend = CpuBackend::default();
        let spec = ModelSpec::new(
            "echo",
            ModelFormat::Unknown,
            ml_runtime_model::ModelLocation::Memory,
        );
        let handle = backend.load(&spec).await.unwrap();
        let result = backend
            .infer(
                &handle,
                &InferenceRequest {
                    model: spec.into(),
                    input: Input::Text("hello".to_owned()),
                    options: Default::default(),
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(result.output, Output::Text("hello".to_owned()));
    }
}
