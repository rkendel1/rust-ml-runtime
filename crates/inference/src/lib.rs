use bytes::Bytes;
use ml_runtime_model::ModelReference;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf, time::Duration};

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

/// A model-neutral request for one or more typed decisions about immutable input state.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DecisionRequest {
    pub state: Value,
    pub decisions: Vec<DecisionQuestion>,
}

/// Backend/model execution facts used by the structured-decision planner.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecisionModelCapabilities {
    pub backend: String,
    pub device: DeviceKind,
    pub model_architecture: ModelArchitecture,
    pub supports_batching: bool,
    pub max_batch_size: Option<usize>,
    pub batch_size: Option<usize>,
    pub supports_async: bool,
    /// True only when an in-flight native operation can actually be interrupted.
    pub supports_cancellation: bool,
    pub supports_parallel_execution: bool,
    pub recommended_parallelism: Option<usize>,
    pub supports_structured_decisions: bool,
}

impl DecisionModelCapabilities {
    pub fn conservative(backend: impl Into<String>) -> Self {
        Self {
            backend: backend.into(),
            device: DeviceKind::Unknown,
            model_architecture: ModelArchitecture::StructuredDecision,
            supports_batching: false,
            max_batch_size: Some(1),
            batch_size: Some(1),
            supports_async: false,
            supports_cancellation: false,
            supports_parallel_execution: false,
            recommended_parallelism: Some(1),
            supports_structured_decisions: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    Cpu,
    Gpu,
    NeuralEngine,
    WebAssembly,
    Unknown,
    Other(String),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    StructuredDecision,
    Tensor,
    Generative,
    Other(String),
}

/// Application constraints. These bound execution without prescribing a backend strategy.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecisionExecutionPolicy {
    #[serde(default)]
    pub latency_target: Option<Duration>,
    #[serde(default)]
    pub max_parallelism: Option<usize>,
    #[serde(default)]
    pub max_batch_size: Option<usize>,
    #[serde(default = "default_true")]
    pub allow_parallel: bool,
    #[serde(default = "default_true")]
    pub allow_batching: bool,
    #[serde(default = "default_true")]
    pub allow_selective_execution: bool,
}

fn default_true() -> bool {
    true
}

impl Default for DecisionExecutionPolicy {
    fn default() -> Self {
        Self {
            latency_target: None,
            max_parallelism: None,
            max_batch_size: None,
            allow_parallel: true,
            allow_batching: true,
            allow_selective_execution: true,
        }
    }
}

/// One decision node and the predicates that control whether it is required.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DecisionNode {
    pub question: DecisionQuestion,
    #[serde(default)]
    pub dependencies: Vec<DecisionDependency>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DecisionDependency {
    pub decision: String,
    #[serde(default)]
    pub condition: DecisionDependencyCondition,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum DecisionDependencyCondition {
    #[default]
    Completed,
    ChoiceEquals(String),
    NoulEquals(bool),
    ScoreAtLeast(f64),
    ScoreAtMost(f64),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DecisionGraphRequest {
    pub state: Value,
    pub nodes: Vec<DecisionNode>,
    #[serde(default)]
    pub policy: DecisionExecutionPolicy,
}

impl DecisionGraphRequest {
    pub fn independent(request: DecisionRequest, policy: DecisionExecutionPolicy) -> Self {
        Self {
            state: request.state,
            nodes: request
                .decisions
                .into_iter()
                .map(|question| DecisionNode {
                    question,
                    dependencies: Vec::new(),
                })
                .collect(),
            policy,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionExecutionStrategy {
    Single,
    Sequential,
    Parallel { concurrency: usize },
    Batched { batch_size: usize },
    Selective,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecisionPlanStage {
    pub nodes: Vec<String>,
    pub batches: Vec<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecisionExecutionPlan {
    pub strategy: DecisionExecutionStrategy,
    pub reason: String,
    pub stages: Vec<DecisionPlanStage>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecisionExecutionDiagnostics {
    pub capabilities: DecisionModelCapabilities,
    pub strategy: DecisionExecutionStrategy,
    pub reason: String,
    pub requested_nodes: usize,
    pub executed_nodes: usize,
    pub skipped_nodes: usize,
    pub batches: Vec<Vec<String>>,
    pub max_concurrency: usize,
    pub cancelled: bool,
}

impl DecisionRequest {
    pub fn new(state: impl Into<Value>, decisions: Vec<DecisionQuestion>) -> Self {
        Self {
            state: state.into(),
            decisions,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DecisionQuestion {
    pub name: String,
    pub instructions: String,
    pub kind: DecisionType,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionType {
    Choice {
        options: Vec<DecisionOption>,
    },
    Score {
        levels: Vec<Value>,
    },
    /// Laya's binary yes/no decision head ("noul" in the model contract).
    Noul {
        #[serde(default)]
        false_description: Option<Value>,
        #[serde(default)]
        true_description: Option<Value>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DecisionOption {
    pub label: String,
    #[serde(default)]
    pub description: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DecisionResult {
    pub model: ModelIdentity,
    pub backend: String,
    pub decisions: Vec<TypedDecision>,
    pub execution: DecisionExecution,
    pub provenance: DecisionProvenance,
}

/// Result returned by the capability-aware executor. Flattening preserves the
/// existing JSON shape while keeping `DecisionResult` source-compatible.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PlannedDecisionResult {
    #[serde(flatten)]
    pub result: DecisionResult,
    pub planning: DecisionExecutionDiagnostics,
}

impl std::ops::Deref for PlannedDecisionResult {
    type Target = DecisionResult;

    fn deref(&self) -> &Self::Target {
        &self.result
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelIdentity {
    pub identifier: String,
    pub revision: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TypedDecision {
    pub name: String,
    pub kind: String,
    pub value: DecisionValue,
    pub probabilities: BTreeMap<String, f64>,
    pub confidence: f64,
    pub action_probability: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum DecisionValue {
    Choice(String),
    Score(f64),
    Noul(bool),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecisionExecution {
    pub latency: Duration,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecisionProvenance {
    pub artifact_path: PathBuf,
    pub artifact_sha256: String,
    pub runtime_version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelDescription {
    pub identifier: String,
    pub revision: Option<String>,
    pub backend: String,
    pub artifact_path: PathBuf,
    pub artifact_sha256: String,
    pub decision_types: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InferenceStreamEvent {
    Started(ExecutionMetadata),
    Output(InferenceOutput),
    Completed(ExecutionMetadata),
}
