//! Model-neutral generative inference contract.
//!
//! This is a separate contract from typed inference (`InferenceRequest` /
//! `InferenceResult`). Provider wire formats (for example OpenAI-compatible
//! JSON) never appear here; adapters translate at their own boundary.
//!
//! Metadata is reported honestly: when a provider does not report token
//! usage, a finish reason, or a model identity, the corresponding field is
//! `None` rather than an estimate. Generated text is model output and is not
//! execution evidence; the evidence is [`ChatMetadata`].

use crate::ProviderCapabilities;
use async_trait::async_trait;
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fmt, time::Duration};

/// Default upper bound on accumulated generated output (4 MiB).
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System, content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User, content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant, content)
    }
}

/// Message content may be sensitive, so `Debug` reports only its length.
impl fmt::Debug for ChatMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatMessage")
            .field("role", &self.role)
            .field("content_bytes", &self.content.len())
            .finish()
    }
}

/// Requested output shape. `JsonSchema` is a request to the provider; it is
/// never a guarantee. The runtime validates the result separately.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        #[serde(default)]
        strict: bool,
    },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GenerationOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Vec<String>,
    pub seed: Option<i64>,
    #[serde(default)]
    pub response_format: ResponseFormat,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatRequest {
    /// Explicit model identity; the provider never substitutes another model.
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub options: GenerationOptions,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            options: GenerationOptions::default(),
        }
    }

    pub fn with_options(mut self, options: GenerationOptions) -> Self {
        self.options = options;
        self
    }

    /// Provider-independent request validation. Providers still reject
    /// options they do not support.
    pub fn validate(&self) -> RuntimeResult<()> {
        use ml_runtime_common::RuntimeError::InvalidRequest;
        let invalid = |reason: &str| {
            Err(InvalidRequest {
                reason: reason.to_owned(),
            })
        };
        if self.model.trim().is_empty() {
            return invalid("chat request requires an explicit model");
        }
        if self.messages.is_empty() {
            return invalid("chat request requires at least one message");
        }
        if !self.messages.iter().any(|m| m.role == ChatRole::User) {
            return invalid("chat request requires at least one user message");
        }
        let o = &self.options;
        if o.temperature.is_some_and(|t| !t.is_finite() || t < 0.0) {
            return invalid("temperature must be a finite non-negative number");
        }
        if o.top_p.is_some_and(|p| !(0.0..=1.0).contains(&p)) {
            return invalid("top_p must be within 0.0..=1.0");
        }
        if o.max_tokens == Some(0) {
            return invalid("max_tokens must be greater than zero");
        }
        if o.stop.iter().any(|s| s.is_empty()) {
            return invalid("stop sequences must not be empty");
        }
        if let ResponseFormat::JsonSchema { name, .. } = &o.response_format {
            if name.is_empty()
                || !name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                return invalid("json schema name must be non-empty [A-Za-z0-9_-]");
            }
        }
        Ok(())
    }
}

/// Why generation stopped, when the provider says. `Other` preserves an
/// unrecognized provider value verbatim.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    Other(String),
}

/// Token accounting as reported by the provider; never estimated.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionLocation {
    Local,
    Remote,
}

/// Execution evidence recorded by the runtime/provider (not by the model).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatMetadata {
    pub provider: String,
    pub transport: String,
    pub execution: ExecutionLocation,
    /// True when `execution` was inferred from the endpoint host rather than
    /// configured explicitly.
    pub execution_inferred: bool,
    /// Endpoint with userinfo, query, and fragment removed.
    pub endpoint: String,
    pub requested_model: String,
    pub reported_model: Option<String>,
    pub response_id: Option<String>,
    pub streamed: bool,
    /// Number of content deltas received when streamed.
    pub stream_events: Option<u64>,
    pub time_to_first_token: Option<Duration>,
    pub duration: Duration,
    pub runtime_version: String,
}

/// Result of checking generated text against the requested format.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StructuredOutput {
    /// Parsed as JSON and, for `JsonSchema`, validated against the schema.
    Validated {
        value: Value,
    },
    InvalidJson {
        error: String,
    },
    SchemaViolation {
        errors: Vec<String>,
    },
}

impl StructuredOutput {
    pub fn is_validated(&self) -> bool {
        matches!(self, Self::Validated { .. })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatResponse {
    pub message: ChatMessage,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    pub metadata: ChatMetadata,
    /// Present only when a JSON response format was requested. `None` means
    /// the output was not checked, never that it passed.
    pub structured: Option<StructuredOutput>,
}

/// Terminal stream summary. It deliberately omits the full text, so a stream
/// never forces the runtime to buffer unbounded output.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatStreamSummary {
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    pub metadata: ChatMetadata,
    pub structured: Option<StructuredOutput>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatStreamEvent {
    /// Exactly once, first.
    Started { provider: String, model: String },
    /// Zero or more incremental content fragments, in order.
    Delta { content: String },
    /// Exactly once on success. Never emitted after an error or cancellation.
    Completed(Box<ChatStreamSummary>),
}

/// Optional features of a generation provider. The OpenAI-compatible adapter
/// cannot discover these, so they are configured and enforced up front.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GenerationCapabilities {
    pub streaming: bool,
    pub stop_sequences: bool,
    pub seed: bool,
    pub json_object: bool,
    pub json_schema: bool,
}

impl GenerationCapabilities {
    /// Rejects options the provider is not declared to support.
    pub fn check(&self, request: &ChatRequest, stream: bool) -> RuntimeResult<()> {
        use ml_runtime_common::RuntimeError::CapabilityMismatch;
        let unsupported = |what: &str| {
            Err(CapabilityMismatch {
                reason: format!("provider does not support {what}"),
            })
        };
        if stream && !self.streaming {
            return unsupported("streaming");
        }
        let o = &request.options;
        if !o.stop.is_empty() && !self.stop_sequences {
            return unsupported("stop sequences");
        }
        if o.seed.is_some() && !self.seed {
            return unsupported("seed");
        }
        match o.response_format {
            ResponseFormat::JsonObject if !self.json_object => unsupported("JSON object output"),
            ResponseFormat::JsonSchema { .. } if !self.json_schema => {
                unsupported("JSON schema output")
            }
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GenerationModelInfo {
    pub id: String,
    pub owned_by: Option<String>,
}

/// A provider of generative inference. Separate from [`crate::Provider`],
/// which serves typed inference.
#[async_trait]
pub trait GenerationProvider: Send + Sync {
    fn name(&self) -> &str;
    /// Shared provider descriptor (local/remote, endpoint, streaming, ...).
    fn capabilities(&self) -> ProviderCapabilities;
    fn generation_capabilities(&self) -> GenerationCapabilities;
    /// Lists models when the upstream supports discovery; otherwise returns
    /// a `CapabilityMismatch` error.
    async fn list_models(
        &self,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<Vec<GenerationModelInfo>>;
    async fn chat(
        &self,
        request: ChatRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<ChatResponse>;
    /// Dropping the returned stream must release the upstream connection.
    async fn chat_stream(
        &self,
        request: ChatRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<ChatStreamEvent>>>;
}
