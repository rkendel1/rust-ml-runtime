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
    /// A caller-supplied result for an earlier [`ToolCall`] proposal.
    Tool,
}

impl ChatRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// Longest tool name accepted (matches common provider limits).
pub const MAX_TOOL_NAME_BYTES: usize = 64;
const MAX_TOOLS: usize = 128;

/// Tool names are restricted to `[A-Za-z0-9_-]{1,64}` so they are safe on every
/// provider wire and in logs.
pub fn is_valid_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_TOOL_NAME_BYTES
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// A tool the model may *propose* to call. The runtime never executes it: the
/// schema constrains the proposed arguments and grants no authority.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema (an object) for the tool's arguments.
    pub input_schema: Value,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, input_schema: Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            input_schema,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Provider-neutral tool selection policy. Absent means the provider default.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    /// The model decides whether to propose tool calls.
    Auto,
    /// The model must not propose tool calls.
    None,
    /// The model must propose a call to this named tool.
    Tool { name: String },
}

/// A tool-call **proposal** produced by a model. It is not evidence that the
/// operation ran, was authorized, or would succeed; the calling application
/// decides whether and how to execute it.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    /// Provider call ID; a [`ChatRole::Tool`] message echoes it in `tool_call_id`.
    pub id: String,
    pub name: String,
    /// Parsed arguments (always a JSON object when produced by a provider).
    pub arguments: Value,
}

impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}

/// Arguments may contain sensitive data, so `Debug` omits them.
impl fmt::Debug for ToolCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolCall")
            .field("id", &self.id)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// Tool-call proposals; only on `Assistant` messages. Text and calls may
    /// both be present where the provider produced both.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Only on `Tool` messages: the [`ToolCall::id`] this content answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
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
    /// An assistant turn that proposed tool calls (replayed in the next request).
    pub fn assistant_with_tool_calls(content: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        Self {
            tool_calls: calls,
            ..Self::new(ChatRole::Assistant, content)
        }
    }
    /// The caller-supplied result for the call with `call_id`.
    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: Some(call_id.into()),
            ..Self::new(ChatRole::Tool, content)
        }
    }
}

/// Message content may be sensitive, so `Debug` reports only its length.
impl fmt::Debug for ChatMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatMessage")
            .field("role", &self.role)
            .field("content_bytes", &self.content.len())
            .field("tool_calls", &self.tool_calls.len())
            .field("tool_call_id", &self.tool_call_id)
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
    /// Tools the model may propose to call. Empty means none are offered.
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
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
        self.validate_tools()?;
        self.validate_messages()
    }

    fn validate_tools(&self) -> RuntimeResult<()> {
        use ml_runtime_common::RuntimeError::InvalidRequest;
        let invalid = |reason: String| Err(InvalidRequest { reason });
        let o = &self.options;
        if o.tools.len() > MAX_TOOLS {
            return invalid(format!("at most {MAX_TOOLS} tools may be offered"));
        }
        let mut seen = std::collections::HashSet::new();
        for tool in &o.tools {
            if !is_valid_tool_name(&tool.name) {
                return invalid(format!(
                    "tool name must match [A-Za-z0-9_-]{{1,{MAX_TOOL_NAME_BYTES}}}"
                ));
            }
            if !seen.insert(tool.name.as_str()) {
                return invalid(format!("duplicate tool name {:?}", tool.name));
            }
            if !tool.input_schema.is_object() {
                return invalid(format!(
                    "input schema of tool {:?} must be a JSON object",
                    tool.name
                ));
            }
        }
        match &o.tool_choice {
            None => {}
            Some(_) if o.tools.is_empty() => {
                return invalid("tool_choice requires at least one tool".to_owned())
            }
            Some(ToolChoice::Tool { name }) if !seen.contains(name.as_str()) => {
                return invalid(format!("tool_choice names undefined tool {name:?}"))
            }
            Some(_) => {}
        }
        Ok(())
    }

    /// Checks message shape and that every tool result answers an earlier
    /// proposal exactly once. Nothing here implies a tool ever ran.
    fn validate_messages(&self) -> RuntimeResult<()> {
        use ml_runtime_common::RuntimeError::InvalidRequest;
        let invalid = |reason: &str| {
            Err(InvalidRequest {
                reason: reason.to_owned(),
            })
        };
        let mut open: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut proposed: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for message in &self.messages {
            match message.role {
                ChatRole::Assistant => {
                    if message.tool_call_id.is_some() {
                        return invalid("only tool messages may carry tool_call_id");
                    }
                    for call in &message.tool_calls {
                        if call.id.is_empty() || !is_valid_tool_name(&call.name) {
                            return invalid(
                                "tool call requires a non-empty id and a valid tool name",
                            );
                        }
                        if !call.arguments.is_object() {
                            return invalid("tool call arguments must be a JSON object");
                        }
                        if !proposed.insert(call.id.as_str()) {
                            return invalid("duplicate tool call id");
                        }
                        open.insert(call.id.as_str());
                    }
                }
                ChatRole::Tool => {
                    if !message.tool_calls.is_empty() {
                        return invalid("tool messages cannot propose tool calls");
                    }
                    let Some(id) = message.tool_call_id.as_deref().filter(|id| !id.is_empty())
                    else {
                        return invalid("tool message requires tool_call_id");
                    };
                    if !open.remove(id) {
                        return invalid(
                            "tool message does not answer an earlier unanswered tool call",
                        );
                    }
                }
                ChatRole::System | ChatRole::User => {
                    if !message.tool_calls.is_empty() || message.tool_call_id.is_some() {
                        return invalid("only assistant and tool messages carry tool fields");
                    }
                }
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
    /// The model stopped to propose tool calls. Proposals only: nothing ran.
    ToolCalls,
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
    /// Raw, unvalidated fragment of a tool call being assembled. For display
    /// only: it is never a usable or authorized proposal.
    ToolCallDelta {
        index: u32,
        id: Option<String>,
        name: Option<String>,
        arguments_fragment: Option<String>,
    },
    /// A fully assembled and validated proposal. Emitted only after the whole
    /// response was received, immediately before `Completed`, in call order.
    ToolCallCompleted { call: ToolCall },
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
    /// Tool definitions, assistant tool calls, and tool-result messages.
    pub tools: bool,
    /// The `tool_choice` control (`auto` / `none` / named tool).
    pub tool_choice: bool,
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
        if (!o.tools.is_empty()
            || request
                .messages
                .iter()
                .any(|m| m.role == ChatRole::Tool || !m.tool_calls.is_empty()))
            && !self.tools
        {
            return unsupported("tool calling");
        }
        if o.tool_choice.is_some() && !self.tool_choice {
            return unsupported("tool_choice");
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
