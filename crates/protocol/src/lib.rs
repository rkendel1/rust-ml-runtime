use ml_runtime_inference::{InferenceRequest, InferenceResult, InferenceStreamEvent};
use ml_runtime_model::{ModelFormat, ModelTensorSpec};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const API_VERSION: &str = "v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct HealthResponse {
    pub status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilitiesResponse {
    pub runtime_version: String,
    pub supported_model_formats: Vec<ModelFormat>,
    pub available_providers: Vec<String>,
    pub available_backends: Vec<String>,
    pub supported_tensor_types: Vec<String>,
    pub streaming: bool,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    pub version: Option<String>,
    pub format: ModelFormat,
    pub inputs: Vec<ModelTensorSpec>,
    pub outputs: Vec<ModelTensorSpec>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InferRequest {
    pub request: InferenceRequest,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InferResponse {
    pub result: Option<InferenceResult>,
    pub error: Option<ErrorEnvelope>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StreamInferResponse {
    pub event: Option<InferenceStreamEvent>,
    pub error: Option<ErrorEnvelope>,
}

impl StreamInferResponse {
    pub fn event(event: InferenceStreamEvent) -> Self {
        Self {
            event: Some(event),
            error: None,
        }
    }

    pub fn error(error: ErrorEnvelope) -> Self {
        Self {
            event: None,
            error: Some(error),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorEnvelope {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

impl InferResponse {
    pub fn success(result: InferenceResult) -> Self {
        Self {
            result: Some(result),
            error: None,
        }
    }

    pub fn error(error: ErrorEnvelope) -> Self {
        Self {
            result: None,
            error: Some(error),
        }
    }
}
