use bytes::Bytes;
use ml_runtime_model::ModelReference;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Determines where the runtime attempts to execute a request and whether it may fall back.
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

impl Tensor {
    pub fn new(shape: impl Into<Vec<usize>>, values: impl Into<Vec<f32>>) -> Self {
        Self {
            shape: shape.into(),
            values: values.into(),
        }
    }
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
    #[serde(default)]
    pub request_id: Option<String>,
    pub model: String,
    pub model_version: Option<String>,
    /// Execution location or transport selected by this runtime (`local`, `remote`, etc.).
    pub provider: String,
    /// Computation implementation reported by the execution target (`onnx`, `cpu`, etc.).
    pub backend: String,
    /// Observable target selected by this runtime. For a client this may be remote even when
    /// `backend` reports the server's local computation backend.
    pub execution_target: String,
    pub routing_policy: ExecutionPolicy,
    pub fallback: bool,
    pub fallback_from: Option<String>,
    pub fallback_reason: Option<String>,
    pub latency_ms: Option<f64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub hardware: Option<String>,
    #[serde(default)]
    pub cache_hit: bool,
    #[serde(default)]
    pub batch_size: usize,
    #[serde(default)]
    pub queue_wait_ms: Option<f64>,
    #[serde(default)]
    pub model_load_ms: Option<f64>,
    #[serde(default)]
    pub execution_ms: Option<f64>,
    #[serde(default)]
    pub streaming_requested: bool,
    #[serde(default)]
    pub streaming_supported: bool,
    #[serde(default)]
    pub output_event_count: usize,
    #[serde(default)]
    pub completion_state: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InferenceRequest {
    pub model: ModelReference,
    pub input: Input,
    pub options: InferenceOptions,
}

impl InferenceRequest {
    /// Creates a request with default local-only execution options.
    pub fn new(model: impl Into<ModelReference>, input: impl Into<Input>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            options: InferenceOptions::default(),
        }
    }

    /// Replaces the request's execution options.
    pub fn with_options(mut self, options: InferenceOptions) -> Self {
        self.options = options;
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InferenceResult {
    pub output: Output,
    pub metadata: ExecutionMetadata,
}

impl InferenceResult {
    pub fn output(&self) -> &Output {
        &self.output
    }

    pub fn metadata(&self) -> &ExecutionMetadata {
        &self.metadata
    }

    pub fn into_output(self) -> Output {
        self.output
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InferenceChunk {
    pub output: Output,
    pub done: bool,
    pub metadata: Option<ExecutionMetadata>,
}

pub type InferenceOutput = Output;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InferenceStreamEvent {
    Started(ExecutionMetadata),
    Output(InferenceOutput),
    Completed(ExecutionMetadata),
}
