use bytes::Bytes;
use ml_runtime_model::ModelReference;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionPolicy {
    #[default]
    LocalOnly,
    RemoteOnly,
    PreferLocal,
    PreferRemote,
    LocalThenRemote,
    RemoteThenLocal,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Image {
    pub mime_type: String,
    pub data: Bytes,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Audio {
    pub mime_type: String,
    pub data: Bytes,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceOptions {
    pub timeout_ms: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub batch_size: Option<usize>,
    pub require_local: bool,
    pub require_remote: bool,
    pub stream: bool,
    #[serde(default)]
    pub execution: ExecutionPolicy,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self {
            timeout_ms: None,
            max_output_tokens: None,
            batch_size: None,
            require_local: false,
            require_remote: false,
            stream: false,
            execution: ExecutionPolicy::LocalOnly,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Input {
    Text(String),
    Tokens(Vec<u32>),
    Tensor(Tensor),
    Image(Image),
    Audio(Audio),
    Binary(Bytes),
    Structured(Value),
}

impl Input {
    pub fn batch_size(&self) -> usize {
        match self {
            Self::Tokens(_) => 1,
            Self::Tensor(tensor) => tensor.shape.first().copied().unwrap_or(1),
            _ => 1,
        }
    }
}

impl From<String> for Input {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for Input {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Output {
    Text(String),
    Tokens(Vec<u32>),
    Tensor(Tensor),
    Embedding(Vec<f32>),
    Structured(Value),
    Binary(Bytes),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ExecutionMetadata {
    pub model: String,
    pub model_version: Option<String>,
    pub provider: String,
    pub backend: String,
    pub execution_target: String,
    pub routing_policy: ExecutionPolicy,
    pub fallback: bool,
    pub fallback_from: Option<String>,
    pub fallback_reason: Option<String>,
    pub latency_ms: Option<f64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub hardware: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InferenceRequest {
    pub model: ModelReference,
    pub input: Input,
    pub options: InferenceOptions,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InferenceResult {
    pub output: Output,
    pub metadata: ExecutionMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InferenceChunk {
    pub output: Output,
    pub done: bool,
    pub metadata: Option<ExecutionMetadata>,
}
