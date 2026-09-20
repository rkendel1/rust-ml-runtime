use async_trait::async_trait;
use futures_util::stream;
use ml_runtime_backend::{Backend, BackendCapabilities, BackendCapability};
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::{
    ExecutionMetadata, InferenceChunk, InferenceRequest, InferenceResult, Input, Output, Tensor,
};
use ml_runtime_model::{ModelFormat, ModelHandle, ModelLocation, ModelSpec};
use ort::{session::Session, value::Tensor as OrtTensor};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Default)]
pub struct OnnxBackend;

struct OnnxLoadedModel {
    spec: ModelSpec,
    session: Mutex<Session>,
}

impl OnnxBackend {
    pub fn descriptor() -> BackendCapability {
        BackendCapability::from_parts("onnx", Self::default().capabilities())
    }

    fn load_session(model: &ModelSpec) -> RuntimeResult<Session> {
        let ModelLocation::Path(path) = &model.location else {
            return Err(RuntimeError::InvalidInput {
                reason: "ONNX models require a filesystem artifact".to_owned(),
            });
        };
        Session::builder()
            .and_then(|mut builder| builder.commit_from_file(path))
            .map_err(|error| RuntimeError::execution("load", format!("load ONNX model: {error}")))
    }
}

#[async_trait]
impl Backend for OnnxBackend {
    fn name(&self) -> &str {
        "onnx"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            available: true,
            local: true,
            streaming: true,
            cancellation: true,
            structured_output: false,
            batching: true,
            supported_formats: vec![ModelFormat::Onnx],
            accelerators: Vec::new(),
            hardware: Some("cpu".to_owned()),
            notes: vec!["ONNX Runtime CPU execution".to_owned()],
        }
    }

    fn supports(&self, model: &ModelSpec) -> bool {
        matches!(model.format, ModelFormat::Onnx)
    }

    async fn load(&self, model: &ModelSpec) -> RuntimeResult<ModelHandle> {
        let session = Self::load_session(model)?;
        Ok(ModelHandle::new(
            model.id.clone(),
            self.name(),
            model.format.clone(),
            Arc::new(OnnxLoadedModel {
                spec: model.clone(),
                session: Mutex::new(session),
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
        let loaded =
            model
                .state::<OnnxLoadedModel>()
                .ok_or_else(|| RuntimeError::InvalidInput {
                    reason: "model handle does not belong to the ONNX backend".to_owned(),
                })?;
        let Input::Tensor(input) = &request.input else {
            return Err(RuntimeError::InvalidInput {
                reason: "ONNX backend currently accepts tensor input".to_owned(),
            });
        };
        if input.shape.iter().product::<usize>() != input.values.len() {
            return Err(RuntimeError::InvalidInput {
                reason: "tensor shape does not match tensor values".to_owned(),
            });
        }
        let mut session = loaded
            .session
            .lock()
            .map_err(|_| RuntimeError::execution("infer", "ONNX session lock poisoned"))?;
        let input_name = session
            .inputs()
            .first()
            .map(|input| input.name().to_owned())
            .ok_or_else(|| RuntimeError::execution("infer", "ONNX model has no inputs"))?;
        let tensor = OrtTensor::from_array((input.shape.clone(), input.values.clone())).map_err(
            |error| RuntimeError::execution("infer", format!("create ONNX input: {error}")),
        )?;
        let outputs = session
            .run(ort::inputs![input_name => tensor])
            .map_err(|error| {
                RuntimeError::execution("infer", format!("execute ONNX model: {error}"))
            })?;
        if outputs.len() == 0 {
            return Err(RuntimeError::execution(
                "infer",
                "ONNX model produced no outputs",
            ));
        }
        let (_, values) = outputs[0].try_extract_tensor::<f32>().map_err(|error| {
            RuntimeError::execution("infer", format!("read ONNX output: {error}"))
        })?;
        let shape = outputs[0]
            .shape()
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap_or(0))
            .collect();
        Ok(InferenceResult {
            output: Output::Tensor(Tensor {
                shape,
                values: values.to_vec(),
            }),
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
        let result = self.infer(model, request, cancellation).await?;
        Ok(Box::pin(stream::iter(vec![Ok(InferenceChunk {
            output: result.output,
            done: true,
            metadata: Some(result.metadata),
        })])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn descriptor_matches_boundary() {
        assert_eq!(OnnxBackend::descriptor().name, "onnx");
    }

    #[test]
    fn advertises_onnx_format() {
        let capabilities = OnnxBackend::default().capabilities();
        assert!(capabilities.available);
        assert_eq!(capabilities.supported_formats, vec![ModelFormat::Onnx]);
        assert!(OnnxBackend::default().supports(&ModelSpec::new(
            "model",
            ModelFormat::Onnx,
            ModelLocation::Memory,
        )));
    }
}
