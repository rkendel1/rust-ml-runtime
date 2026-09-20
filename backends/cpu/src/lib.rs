use async_trait::async_trait;
use futures_util::stream;
use ml_runtime_backend::{Backend, BackendCapabilities, BackendCapability};
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::{
    ExecutionMetadata, InferenceChunk, InferenceRequest, InferenceResult, Input, Output,
};
use ml_runtime_model::{ModelFormat, ModelHandle, ModelSpec};
use serde::Deserialize;
use std::sync::Arc;
use std::{fs, path::Path};

#[derive(Clone, Debug, Default)]
pub struct CpuBackend;

#[derive(Clone, Debug)]
struct CpuLoadedModel {
    spec: ModelSpec,
    linear: Option<LinearModel>,
}

#[derive(Clone, Debug, Deserialize)]
struct LinearModel {
    weights: Vec<Vec<f32>>,
    #[serde(default)]
    bias: Vec<f32>,
}

fn validate_linear_model(model: &LinearModel) -> RuntimeResult<()> {
    if model.weights.is_empty()
        || model
            .weights
            .iter()
            .any(|row| row.len() != model.weights[0].len())
        || (!model.bias.is_empty() && model.bias.len() != model.weights.len())
    {
        return Err(RuntimeError::InvalidInput {
            reason: "linear model weights and bias have incompatible shapes".to_owned(),
        });
    }
    Ok(())
}

fn run_linear(
    model: &LinearModel,
    input: &ml_runtime_inference::Tensor,
) -> RuntimeResult<ml_runtime_inference::Tensor> {
    let inputs = input.shape.last().copied().unwrap_or(0);
    if inputs != model.weights[0].len() || input.values.len() != inputs {
        return Err(RuntimeError::InvalidInput {
            reason: format!("expected a tensor with {} values", model.weights[0].len()),
        });
    }
    let values = model
        .weights
        .iter()
        .enumerate()
        .map(|(row, weights)| {
            weights
                .iter()
                .zip(&input.values)
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
                + model.bias.get(row).copied().unwrap_or(0.0)
        })
        .collect();
    Ok(ml_runtime_inference::Tensor {
        shape: vec![model.weights.len()],
        values,
    })
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
        let linear = match &model.location {
            ml_runtime_model::ModelLocation::Path(path) => {
                let bytes = fs::read(path).map_err(|error| {
                    RuntimeError::execution(
                        "load",
                        format!("read model artifact {}: {error}", Path::new(path).display()),
                    )
                })?;
                let artifact: LinearModel = serde_json::from_slice(&bytes).map_err(|error| {
                    RuntimeError::execution("load", format!("parse model artifact: {error}"))
                })?;
                validate_linear_model(&artifact)?;
                Some(artifact)
            }
            _ => None,
        };
        Ok(ModelHandle::new(
            model.id.clone(),
            self.name(),
            model.format.clone(),
            Arc::new(CpuLoadedModel {
                spec: model.clone(),
                linear,
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

        let output = match (&loaded.linear, &request.input) {
            (Some(linear), Input::Tensor(tensor)) => Output::Tensor(run_linear(linear, tensor)?),
            (Some(_), _) => {
                return Err(RuntimeError::InvalidInput {
                    reason: "this model accepts tensor input".to_owned(),
                })
            }
            (None, Input::Text(text)) => Output::Text(text.clone()),
            (None, Input::Tokens(tokens)) => Output::Tokens(tokens.clone()),
            (None, Input::Tensor(tensor)) => Output::Tensor(tensor.clone()),
            (None, Input::Binary(bytes)) => Output::Binary(bytes.clone()),
            (None, Input::Structured(value)) => Output::Structured(value.clone()),
            (None, Input::Image(_) | Input::Audio(_)) => return Err(RuntimeError::InvalidInput {
                reason:
                    "the CPU backend supports text, tokens, tensors, binary, and structured inputs"
                        .to_owned(),
            }),
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

    #[tokio::test]
    async fn executes_linear_model_artifact() {
        let artifact =
            std::env::temp_dir().join(format!("ml-runtime-linear-{}.json", std::process::id()));
        std::fs::write(
            &artifact,
            r#"{"weights":[[2.0, 0.0],[0.0, 3.0]],"bias":[1.0,-1.0]}"#,
        )
        .unwrap();
        let backend = CpuBackend::default();
        let spec = ModelSpec::new(
            "linear",
            ModelFormat::Unknown,
            ml_runtime_model::ModelLocation::Path(artifact.display().to_string()),
        );
        let handle = backend.load(&spec).await.unwrap();
        let result = backend
            .infer(
                &handle,
                &InferenceRequest {
                    model: spec.into(),
                    input: Input::Tensor(ml_runtime_inference::Tensor {
                        shape: vec![2],
                        values: vec![2.0, 4.0],
                    }),
                    options: Default::default(),
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            result.output,
            Output::Tensor(ml_runtime_inference::Tensor {
                shape: vec![2],
                values: vec![5.0, 11.0],
            })
        );
        std::fs::remove_file(artifact).unwrap();
    }
}
