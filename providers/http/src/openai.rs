//! OpenAI-compatible chat-completions adapter.
//!
//! All OpenAI wire types live in this module and never escape it; callers see
//! only the model-neutral contract from `ml-runtime-provider`. The adapter
//! never logs, and its `Debug` output and errors never contain the API key or
//! message contents.

use async_trait::async_trait;
use futures_util::StreamExt;
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_provider::is_valid_tool_name;
use ml_runtime_provider::{
    ChatMessage, ChatMetadata, ChatRequest, ChatResponse, ChatStreamEvent, ChatStreamSummary,
    ExecutionLocation, FinishReason, GenerationCapabilities, GenerationModelInfo,
    GenerationProvider, ProviderCapabilities, ResponseFormat, ToolCall, ToolChoice, Usage,
    DEFAULT_MAX_OUTPUT_BYTES,
};
use reqwest::{Response, StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fmt,
    future::Future,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

pub const PROVIDER_NAME: &str = "openai-compatible";
const TRANSPORT: &str = "http";
const ERROR_BODY_LIMIT: usize = 64 * 1024;
const ERROR_MESSAGE_CHARS: usize = 300;

/// A bearer credential. `Debug` is redacted and the type is not serializable.
#[derive(Clone)]
pub struct ApiKey(String);

impl ApiKey {
    pub fn new(key: impl Into<String>) -> RuntimeResult<Self> {
        let key = key.into();
        if key.is_empty() || key.chars().any(|c| c.is_control() || c.is_whitespace()) {
            return Err(RuntimeError::InvalidRequest {
                reason: "API key must be non-empty and contain no whitespace".to_owned(),
            });
        }
        Ok(Self(key))
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

#[derive(Clone, Debug)]
pub struct OpenAiCompatibleConfig {
    /// Base URL including the API prefix, e.g. `http://127.0.0.1:11434/v1`.
    pub base_url: String,
    pub api_key: Option<ApiKey>,
    /// Explicit local/remote classification. When `None`, loopback hosts are
    /// reported as local and everything else as remote (flagged as inferred).
    pub location: Option<ExecutionLocation>,
    pub capabilities: GenerationCapabilities,
    /// Ask streaming servers to include token usage in the final chunk.
    pub stream_usage: bool,
    pub connect_timeout: Duration,
    /// Bound on time-to-response-headers for streams and on the whole
    /// operation for non-streaming requests and model listing.
    pub request_timeout: Duration,
    /// Maximum silence between stream chunks.
    pub stream_idle_timeout: Duration,
    pub max_output_bytes: usize,
    pub max_response_bytes: usize,
    pub max_event_bytes: usize,
    /// Maximum tool calls accepted in one response.
    pub max_tool_calls: usize,
    /// Bounded channel depth between the network task and the consumer.
    pub stream_buffer: usize,
}

impl OpenAiCompatibleConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            location: None,
            capabilities: GenerationCapabilities {
                streaming: true,
                stop_sequences: true,
                seed: true,
                // Structured output support varies by server and cannot be
                // discovered; it must be declared.
                json_object: false,
                json_schema: false,
                // Tool support is likewise per-server and must be declared.
                tools: false,
                tool_choice: false,
            },
            stream_usage: true,
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(120),
            stream_idle_timeout: Duration::from_secs(60),
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            max_response_bytes: 16 * 1024 * 1024,
            max_event_bytes: 1024 * 1024,
            max_tool_calls: 32,
            stream_buffer: 32,
        }
    }

    pub fn with_api_key(mut self, key: ApiKey) -> Self {
        self.api_key = Some(key);
        self
    }
}

#[derive(Clone)]
pub struct OpenAiCompatibleProvider {
    base: String,
    config: OpenAiCompatibleConfig,
    location: ExecutionLocation,
    location_inferred: bool,
    client: reqwest::Client,
}

impl fmt::Debug for OpenAiCompatibleProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiCompatibleProvider")
            .field("endpoint", &self.base)
            .field("location", &self.location)
            .field("authenticated", &self.config.api_key.is_some())
            .finish()
    }
}

fn config_error(reason: impl Into<String>) -> RuntimeError {
    RuntimeError::InvalidRequest {
        reason: reason.into(),
    }
}

fn is_loopback(url: &Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

impl OpenAiCompatibleProvider {
    pub fn new(config: OpenAiCompatibleConfig) -> RuntimeResult<Self> {
        let url = Url::parse(&config.base_url)
            .map_err(|_| config_error("endpoint is not a valid URL"))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(config_error("endpoint scheme must be http or https"));
        }
        if url.host().is_none() {
            return Err(config_error("endpoint must include a host"));
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(config_error(
                "endpoint must not embed credentials; supply the API key separately",
            ));
        }
        if url.query().is_some() || url.fragment().is_some() {
            return Err(config_error(
                "endpoint must not include a query or fragment",
            ));
        }
        let loopback = is_loopback(&url);
        if config.api_key.is_some() && url.scheme() == "http" && !loopback {
            return Err(config_error(
                "refusing to send an API key over plain http to a non-loopback host",
            ));
        }
        for (name, value) in [
            ("max_output_bytes", config.max_output_bytes),
            ("max_response_bytes", config.max_response_bytes),
            ("max_event_bytes", config.max_event_bytes),
            ("max_tool_calls", config.max_tool_calls),
            ("stream_buffer", config.stream_buffer),
        ] {
            if value == 0 {
                return Err(config_error(format!("{name} must be greater than zero")));
            }
        }
        let (location, location_inferred) = match config.location {
            Some(location) => (location, false),
            None => (
                if loopback {
                    ExecutionLocation::Local
                } else {
                    ExecutionLocation::Remote
                },
                true,
            ),
        };
        let client = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            // A POST carrying a bearer token must not be silently re-sent.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| {
                RuntimeError::transport(PROVIDER_NAME, error.without_url().to_string())
            })?;
        Ok(Self {
            base: url.as_str().trim_end_matches('/').to_owned(),
            config,
            location,
            location_inferred,
            client,
        })
    }

    pub fn endpoint(&self) -> &str {
        &self.base
    }

    pub fn location(&self) -> ExecutionLocation {
        self.location
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let builder = self
            .client
            .request(method, format!("{}{}", self.base, path));
        match &self.config.api_key {
            Some(key) => builder.bearer_auth(&key.0),
            None => builder,
        }
    }

    fn metadata(&self, request_model: &str, tracker: &Tracker, streamed: bool) -> ChatMetadata {
        ChatMetadata {
            provider: PROVIDER_NAME.to_owned(),
            transport: TRANSPORT.to_owned(),
            execution: self.location,
            execution_inferred: self.location_inferred,
            endpoint: self.base.clone(),
            requested_model: request_model.to_owned(),
            reported_model: tracker.model.clone(),
            response_id: tracker.id.clone(),
            streamed,
            stream_events: streamed.then_some(tracker.deltas),
            time_to_first_token: tracker.first_token,
            duration: tracker.started.elapsed(),
            runtime_version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }

    fn wire_request<'a>(&self, request: &'a ChatRequest, stream: bool) -> WireRequest<'a> {
        let o = &request.options;
        WireRequest {
            model: &request.model,
            messages: request
                .messages
                .iter()
                .map(|m| WireMessage {
                    role: m.role.as_str(),
                    content: (m.tool_calls.is_empty() || !m.content.is_empty())
                        .then_some(m.content.as_str()),
                    tool_calls: (!m.tool_calls.is_empty()).then(|| {
                        m.tool_calls
                            .iter()
                            .map(|c| WireToolCallIn {
                                id: &c.id,
                                r#type: "function",
                                function: WireFunctionIn {
                                    name: &c.name,
                                    arguments: c.arguments.to_string(),
                                },
                            })
                            .collect()
                    }),
                    tool_call_id: m.tool_call_id.as_deref(),
                })
                .collect(),
            stream,
            stream_options: (stream && self.config.stream_usage).then_some(WireStreamOptions {
                include_usage: true,
            }),
            temperature: o.temperature,
            top_p: o.top_p,
            max_tokens: o.max_tokens,
            stop: (!o.stop.is_empty()).then_some(&o.stop),
            seed: o.seed,
            response_format: match &o.response_format {
                ResponseFormat::Text => None,
                ResponseFormat::JsonObject => Some(WireResponseFormat::JsonObject),
                ResponseFormat::JsonSchema {
                    name,
                    schema,
                    strict,
                } => Some(WireResponseFormat::JsonSchema {
                    json_schema: WireJsonSchema {
                        name,
                        schema,
                        strict: *strict,
                    },
                }),
            },
            tools: (!o.tools.is_empty()).then(|| {
                o.tools
                    .iter()
                    .map(|t| WireTool {
                        r#type: "function",
                        function: WireToolFunction {
                            name: &t.name,
                            description: t.description.as_deref(),
                            parameters: &t.input_schema,
                        },
                    })
                    .collect()
            }),
            tool_choice: o.tool_choice.as_ref().map(|choice| match choice {
                ToolChoice::Auto => WireToolChoice::Mode("auto"),
                ToolChoice::None => WireToolChoice::Mode("none"),
                ToolChoice::Tool { name } => WireToolChoice::Named {
                    r#type: "function",
                    function: WireFunctionName { name },
                },
            }),
        }
    }

    async fn error_for_status(&self, response: Response) -> RuntimeError {
        let status = response.status();
        let body = read_limited(response, ERROR_BODY_LIMIT)
            .await
            .unwrap_or_default();
        map_http_error(status, &body)
    }
}

// ---- wire types (private to this adapter) --------------------------------

#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'static str,
    /// `null` for an assistant turn that only proposed tool calls.
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<WireToolCallIn<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
}

#[derive(Serialize)]
struct WireToolCallIn<'a> {
    id: &'a str,
    r#type: &'static str,
    function: WireFunctionIn<'a>,
}

#[derive(Serialize)]
struct WireFunctionIn<'a> {
    name: &'a str,
    /// The wire carries arguments as a JSON-encoded string.
    arguments: String,
}

#[derive(Serialize)]
struct WireTool<'a> {
    r#type: &'static str,
    function: WireToolFunction<'a>,
}

#[derive(Serialize)]
struct WireToolFunction<'a> {
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<&'a str>,
    parameters: &'a Value,
}

#[derive(Serialize)]
#[serde(untagged)]
enum WireToolChoice<'a> {
    Mode(&'static str),
    Named {
        r#type: &'static str,
        function: WireFunctionName<'a>,
    },
}

#[derive(Serialize)]
struct WireFunctionName<'a> {
    name: &'a str,
}

#[derive(Serialize)]
struct WireStreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireResponseFormat<'a> {
    JsonObject,
    JsonSchema { json_schema: WireJsonSchema<'a> },
}

#[derive(Serialize)]
struct WireJsonSchema<'a> {
    name: &'a str,
    schema: &'a Value,
    strict: bool,
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<WireStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<&'a Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<WireResponseFormat<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<WireTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<WireToolChoice<'a>>,
}

#[derive(Deserialize)]
struct WireUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

impl From<WireUsage> for Usage {
    fn from(w: WireUsage) -> Self {
        Usage {
            prompt_tokens: w.prompt_tokens,
            completion_tokens: w.completion_tokens,
            total_tokens: w.total_tokens,
        }
    }
}

#[derive(Deserialize)]
struct WireCompletion {
    id: Option<String>,
    model: Option<String>,
    #[serde(default)]
    choices: Vec<WireChoice>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireChoice {
    message: Option<WireResponseMessage>,
    delta: Option<WireResponseMessage>,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct WireResponseMessage {
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<WireToolCallOut>>,
}

#[derive(Deserialize)]
struct WireToolCallOut {
    index: Option<u32>,
    id: Option<String>,
    function: Option<WireFunctionOut>,
}

#[derive(Deserialize)]
struct WireFunctionOut {
    name: Option<String>,
    arguments: Option<Value>,
}

#[derive(Deserialize)]
struct WireModels {
    data: Vec<WireModel>,
}

#[derive(Deserialize)]
struct WireModel {
    id: String,
    owned_by: Option<String>,
}

fn finish_reason(raw: String) -> FinishReason {
    match raw.as_str() {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        "tool_calls" => FinishReason::ToolCalls,
        _ => FinishReason::Other(raw),
    }
}

fn protocol(operation: &str, reason: impl Into<String>) -> RuntimeError {
    RuntimeError::Protocol {
        operation: operation.to_owned(),
        reason: reason.into(),
    }
}

// ---- tool-call assembly ----------------------------------------------------------

struct PendingCall {
    index: u32,
    id: String,
    name: String,
    arguments: String,
}

/// Raw fragment, reported to the consumer for display only.
struct CallFragment {
    index: u32,
    id: Option<String>,
    name: Option<String>,
    arguments: Option<String>,
}

/// Accumulates tool-call fragments (streamed or whole) and turns them into
/// validated proposals only in [`ToolAssembler::finish`]. Nothing is invented:
/// a call without an id or name, with non-JSON or non-object arguments, or a
/// duplicate id fails the whole response.
struct ToolAssembler {
    calls: Vec<PendingCall>,
    max_calls: usize,
    bytes: usize,
}

impl ToolAssembler {
    fn new(max_calls: usize) -> Self {
        Self {
            calls: Vec::new(),
            max_calls,
            bytes: 0,
        }
    }

    fn feed(
        &mut self,
        wire: Vec<WireToolCallOut>,
        operation: &str,
    ) -> RuntimeResult<Vec<CallFragment>> {
        let mut fragments = Vec::new();
        for call in wire {
            let function = call.function;
            let name = function.as_ref().and_then(|f| f.name.clone());
            let arguments = match function.and_then(|f| f.arguments) {
                None | Some(Value::Null) => None,
                Some(Value::String(text)) => Some(text),
                Some(_) => {
                    return Err(protocol(
                        operation,
                        "tool call arguments must be a JSON-encoded string",
                    ))
                }
            };
            let slot = match (
                call.index,
                self.calls.iter().position(|c| Some(c.index) == call.index),
            ) {
                (_, Some(slot)) => slot,
                // Servers that omit `index`: an id starts a new call, anything
                // else continues the latest one.
                (None, None) if call.id.is_none() && !self.calls.is_empty() => self.calls.len() - 1,
                (index, None) => {
                    if self.calls.len() >= self.max_calls {
                        return Err(RuntimeError::ResourceLimit {
                            reason: format!(
                                "response proposes more than {} tool calls",
                                self.max_calls
                            ),
                        });
                    }
                    let index = index.unwrap_or(self.calls.len() as u32);
                    self.calls.push(PendingCall {
                        index,
                        id: String::new(),
                        name: String::new(),
                        arguments: String::new(),
                    });
                    self.calls.len() - 1
                }
            };
            let pending = &mut self.calls[slot];
            // Fragments are concatenated as the wire specifies. A server that
            // repeats a whole name/id per chunk corrupts the value, which is
            // then rejected (unknown tool / failed correlation), not guessed.
            let id = call.id.filter(|f| !f.is_empty());
            let name = name.filter(|f| !f.is_empty());
            let arguments = arguments.filter(|f| !f.is_empty());
            self.bytes += id.as_deref().map_or(0, str::len)
                + name.as_deref().map_or(0, str::len)
                + arguments.as_deref().map_or(0, str::len);
            if let Some(id) = &id {
                pending.id.push_str(id);
            }
            if let Some(name) = &name {
                pending.name.push_str(name);
            }
            if let Some(arguments) = &arguments {
                pending.arguments.push_str(arguments);
            }
            if id.is_some() || name.is_some() || arguments.is_some() {
                fragments.push(CallFragment {
                    index: pending.index,
                    id,
                    name,
                    arguments,
                });
            }
        }
        Ok(fragments)
    }

    fn finish(self, operation: &str) -> RuntimeResult<Vec<ToolCall>> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::with_capacity(self.calls.len());
        for call in self.calls {
            if call.id.is_empty() || call.name.is_empty() {
                return Err(protocol(operation, "tool call is missing its id or name"));
            }
            if !seen.insert(call.id.clone()) {
                return Err(protocol(operation, "duplicate tool call id"));
            }
            if !is_valid_tool_name(&call.name) {
                return Err(RuntimeError::InvalidOutput {
                    reason: "model proposed a tool call with a malformed tool name".to_owned(),
                });
            }
            let invalid = |why: &str| RuntimeError::InvalidOutput {
                reason: format!("arguments for tool call to {:?} {why}", call.name),
            };
            let arguments: Value = serde_json::from_str(&call.arguments)
                .map_err(|_| invalid("are not complete, valid JSON"))?;
            if !arguments.is_object() {
                return Err(invalid("are not a JSON object"));
            }
            out.push(ToolCall::new(call.id, call.name, arguments));
        }
        Ok(out)
    }
}

// ---- errors ----------------------------------------------------------------

fn truncate(text: &str) -> String {
    let mut out: String = text.chars().take(ERROR_MESSAGE_CHARS).collect();
    if text.chars().count() > ERROR_MESSAGE_CHARS {
        out.push('…');
    }
    out
}

/// Extracts a provider's error text from the common body shapes.
fn provider_error_text(body: &[u8]) -> (Option<String>, String) {
    let Ok(value) = serde_json::from_slice::<Value>(body) else {
        return (None, truncate(String::from_utf8_lossy(body).trim()));
    };
    let error = value.get("error").unwrap_or(&value);
    if let Some(text) = error.as_str() {
        return (None, truncate(text));
    }
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("no error message");
    let code = error
        .get("code")
        .and_then(|c| c.as_str().map(str::to_owned))
        .or_else(|| error.get("type").and_then(Value::as_str).map(str::to_owned));
    (code, truncate(message))
}

/// Maps an upstream HTTP error to a structured runtime error. The upstream
/// message is untrusted text and is truncated.
fn map_http_error(status: StatusCode, body: &[u8]) -> RuntimeError {
    let (code, message) = provider_error_text(body);
    let detail = format!("HTTP {}: {message}", status.as_u16());
    match status.as_u16() {
        401 | 403 => RuntimeError::provider_unavailable(
            PROVIDER_NAME,
            format!("authentication or authorization failed ({detail})"),
        ),
        404 if code.as_deref() == Some("model_not_found")
            || message.to_ascii_lowercase().contains("model") =>
        {
            RuntimeError::ModelNotFound { model: message }
        }
        408 | 504 => RuntimeError::Timeout {
            operation: format!("provider response ({detail})"),
        },
        400 | 404 | 405 | 409 | 413 | 415 | 422 => RuntimeError::InvalidRequest {
            reason: format!("provider rejected request ({detail})"),
        },
        429 | 500..=599 => RuntimeError::provider_unavailable(PROVIDER_NAME, detail),
        _ => protocol("provider request", detail),
    }
}

fn map_send_error(error: reqwest::Error) -> RuntimeError {
    let error = error.without_url();
    if error.is_timeout() {
        RuntimeError::Timeout {
            operation: "connecting to provider".to_owned(),
        }
    } else {
        RuntimeError::transport(PROVIDER_NAME, error.to_string())
    }
}

/// Reads at most `limit` bytes of a response body.
async fn read_limited(response: Response, limit: usize) -> RuntimeResult<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|n| n as usize > limit)
    {
        return Err(RuntimeError::ResourceLimit {
            reason: format!("provider response exceeds {limit} bytes"),
        });
    }
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(map_send_error)?;
        if body.len() + chunk.len() > limit {
            return Err(RuntimeError::ResourceLimit {
                reason: format!("provider response exceeds {limit} bytes"),
            });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

async fn cancelled(token: &Option<CancellationToken>) {
    match token {
        Some(token) => token.cancelled().await,
        None => std::future::pending().await,
    }
}

/// Runs `future` under a deadline and cooperative cancellation.
async fn bounded<T>(
    token: &Option<CancellationToken>,
    deadline: Duration,
    operation: &str,
    future: impl Future<Output = RuntimeResult<T>>,
) -> RuntimeResult<T> {
    tokio::select! {
        biased;
        _ = cancelled(token) => Err(RuntimeError::Cancelled),
        result = tokio::time::timeout(deadline, future) => match result {
            Ok(result) => result,
            Err(_) => Err(RuntimeError::Timeout { operation: operation.to_owned() }),
        },
    }
}

struct Tracker {
    started: Instant,
    id: Option<String>,
    model: Option<String>,
    first_token: Option<Duration>,
    deltas: u64,
}

impl Tracker {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            id: None,
            model: None,
            first_token: None,
            deltas: 0,
        }
    }

    fn observe(&mut self, id: &Option<String>, model: &Option<String>) {
        if self.id.is_none() {
            self.id.clone_from(id);
        }
        if self.model.is_none() {
            self.model.clone_from(model);
        }
    }
}

// ---- provider ----------------------------------------------------------------

#[async_trait]
impl GenerationProvider for OpenAiCompatibleProvider {
    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn capabilities(&self) -> ProviderCapabilities {
        let g = &self.config.capabilities;
        ProviderCapabilities {
            available: true,
            local: self.location == ExecutionLocation::Local,
            remote: self.location == ExecutionLocation::Remote,
            streaming: g.streaming,
            cancellation: true,
            structured_output: g.json_object || g.json_schema,
            batching: false,
            endpoint: Some(self.base.clone()),
            notes: vec!["OpenAI-compatible chat completions over HTTP".to_owned()],
        }
    }

    fn generation_capabilities(&self) -> GenerationCapabilities {
        self.config.capabilities.clone()
    }

    async fn list_models(
        &self,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<Vec<GenerationModelInfo>> {
        let operation = "listing models";
        bounded(
            &cancellation,
            self.config.request_timeout,
            operation,
            async {
                let response = self
                    .request(reqwest::Method::GET, "/models")
                    .send()
                    .await
                    .map_err(map_send_error)?;
                let status = response.status();
                if matches!(status.as_u16(), 404 | 405 | 501) {
                    return Err(RuntimeError::CapabilityMismatch {
                        reason: "upstream does not implement model listing".to_owned(),
                    });
                }
                if !status.is_success() {
                    return Err(self.error_for_status(response).await);
                }
                let body = read_limited(response, self.config.max_response_bytes).await?;
                let models: WireModels = serde_json::from_slice(&body)
                    .map_err(|_| protocol(operation, "malformed model list"))?;
                Ok(models
                    .data
                    .into_iter()
                    .map(|m| GenerationModelInfo {
                        id: m.id,
                        owned_by: m.owned_by,
                    })
                    .collect())
            },
        )
        .await
    }

    async fn chat(
        &self,
        request: ChatRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<ChatResponse> {
        request.validate()?;
        self.config.capabilities.check(&request, false)?;
        let operation = "chat completion";
        let mut tracker = Tracker::new();
        let completion = bounded(
            &cancellation,
            self.config.request_timeout,
            operation,
            async {
                let response = self
                    .request(reqwest::Method::POST, "/chat/completions")
                    .json(&self.wire_request(&request, false))
                    .send()
                    .await
                    .map_err(map_send_error)?;
                if !response.status().is_success() {
                    return Err(self.error_for_status(response).await);
                }
                let body = read_limited(response, self.config.max_response_bytes).await?;
                serde_json::from_slice::<WireCompletion>(&body)
                    .map_err(|_| protocol(operation, "malformed completion payload"))
            },
        )
        .await?;

        tracker.observe(&completion.id, &completion.model);
        let choice = completion
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| protocol(operation, "completion contained no choices"))?;
        let message = choice
            .message
            .ok_or_else(|| protocol(operation, "choice contained no message"))?;
        if message.role.as_deref().is_some_and(|r| r != "assistant") {
            return Err(protocol(
                operation,
                "response message role is not assistant",
            ));
        }
        let content = message.content.unwrap_or_default();
        let mut assembler = ToolAssembler::new(self.config.max_tool_calls);
        assembler.feed(message.tool_calls.unwrap_or_default(), operation)?;
        if content.len() + assembler.bytes > self.config.max_output_bytes {
            return Err(RuntimeError::ResourceLimit {
                reason: format!(
                    "generated output exceeds {} bytes",
                    self.config.max_output_bytes
                ),
            });
        }
        let tool_calls = assembler.finish(operation)?;
        Ok(ChatResponse {
            message: ChatMessage::assistant_with_tool_calls(content, tool_calls),
            finish_reason: choice.finish_reason.map(finish_reason),
            usage: completion.usage.map(Usage::from),
            metadata: self.metadata(&request.model, &tracker, false),
            structured: None,
        })
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<ChatStreamEvent>>> {
        request.validate()?;
        self.config.capabilities.check(&request, true)?;
        let tracker = Tracker::new();
        let response = bounded(
            &cancellation,
            self.config.request_timeout,
            "waiting for stream response",
            async {
                let response = self
                    .request(reqwest::Method::POST, "/chat/completions")
                    .json(&self.wire_request(&request, true))
                    .send()
                    .await
                    .map_err(map_send_error)?;
                if !response.status().is_success() {
                    return Err(self.error_for_status(response).await);
                }
                Ok(response)
            },
        )
        .await?;
        if response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.trim_start().starts_with("application/json"))
        {
            return Err(protocol(
                "chat stream",
                "provider returned a non-streaming response to a streaming request",
            ));
        }

        let (sender, receiver) = mpsc::channel(self.config.stream_buffer);
        let provider = self.clone();
        tokio::spawn(async move {
            provider
                .pump_stream(request, response, tracker, cancellation, sender)
                .await
        });
        Ok(Box::pin(ReceiverStream::new(receiver)))
    }
}

type EventSender = mpsc::Sender<RuntimeResult<ChatStreamEvent>>;

impl OpenAiCompatibleProvider {
    /// Owns the connection. Returns (dropping the response and so closing the
    /// connection) when the consumer drops the stream, on cancellation, on
    /// error, or on completion.
    async fn pump_stream(
        &self,
        request: ChatRequest,
        response: Response,
        mut tracker: Tracker,
        cancellation: Option<CancellationToken>,
        sender: EventSender,
    ) {
        let outcome = self
            .read_stream(&request, response, &mut tracker, &cancellation, &sender)
            .await;
        if let Err(error) = outcome {
            // The consumer may already be gone; that is not an error here.
            let _ = sender.try_send(Err(error.clone()));
            if sender.capacity() == 0 && !sender.is_closed() {
                // Buffer full: wait for room, but never past cancellation.
                tokio::select! {
                    _ = cancelled(&cancellation) => {}
                    _ = sender.send(Err(error)) => {}
                }
            }
        }
    }

    async fn emit(
        sender: &EventSender,
        cancellation: &Option<CancellationToken>,
        event: ChatStreamEvent,
    ) -> RuntimeResult<()> {
        tokio::select! {
            biased;
            _ = cancelled(cancellation) => Err(RuntimeError::Cancelled),
            sent = sender.send(Ok(event)) => sent.map_err(|_| RuntimeError::Cancelled),
        }
    }

    async fn read_stream(
        &self,
        request: &ChatRequest,
        response: Response,
        tracker: &mut Tracker,
        cancellation: &Option<CancellationToken>,
        sender: &EventSender,
    ) -> RuntimeResult<()> {
        let operation = "chat stream";
        Self::emit(
            sender,
            cancellation,
            ChatStreamEvent::Started {
                provider: PROVIDER_NAME.to_owned(),
                model: request.model.clone(),
            },
        )
        .await?;

        let mut parser = SseParser::new(self.config.max_event_bytes);
        let mut bytes = response.bytes_stream();
        let mut finish = None;
        let mut usage = None;
        let mut output_bytes = 0usize;
        let mut done = false;
        let mut assembler = ToolAssembler::new(self.config.max_tool_calls);

        'read: loop {
            let chunk = tokio::select! {
                biased;
                _ = sender.closed() => return Ok(()),
                _ = cancelled(cancellation) => return Err(RuntimeError::Cancelled),
                chunk = tokio::time::timeout(self.config.stream_idle_timeout, bytes.next()) => chunk,
            };
            let events = match chunk {
                Err(_) if finish.is_some() => break 'read, // finished, server just kept the socket open
                Err(_) => {
                    return Err(RuntimeError::Timeout {
                        operation: "waiting for stream data".to_owned(),
                    })
                }
                Ok(None) => parser.finish(),
                Ok(Some(Err(error))) => return Err(map_send_error(error)),
                Ok(Some(Ok(chunk))) => parser.push(&chunk),
            }
            .map_err(|e| protocol(operation, e))?;
            let at_eof = events.eof;

            for event in events.events {
                if event.data.trim() == "[DONE]" {
                    done = true;
                    break 'read;
                }
                let value: Value = serde_json::from_str(&event.data)
                    .map_err(|_| protocol(operation, "malformed stream event"))?;
                if let Some(error) = value.get("error") {
                    let body = serde_json::to_vec(&serde_json::json!({ "error": error }))
                        .unwrap_or_default();
                    return Err(map_stream_error(&body));
                }
                let chunk: WireCompletion = serde_json::from_value(value)
                    .map_err(|_| protocol(operation, "malformed stream chunk"))?;
                tracker.observe(&chunk.id, &chunk.model);
                if let Some(u) = chunk.usage {
                    usage = Some(Usage::from(u));
                }
                if let Some(choice) = chunk.choices.into_iter().next() {
                    if let Some(reason) = choice.finish_reason {
                        finish = Some(finish_reason(reason));
                    }
                    let mut delta = choice.delta;
                    if let Some(wire) = delta.as_mut().and_then(|d| d.tool_calls.take()) {
                        let fragments = assembler.feed(wire, operation)?;
                        if output_bytes + assembler.bytes > self.config.max_output_bytes {
                            return Err(RuntimeError::ResourceLimit {
                                reason: format!(
                                    "generated output exceeds {} bytes",
                                    self.config.max_output_bytes
                                ),
                            });
                        }
                        for fragment in fragments {
                            Self::emit(
                                sender,
                                cancellation,
                                ChatStreamEvent::ToolCallDelta {
                                    index: fragment.index,
                                    id: fragment.id,
                                    name: fragment.name,
                                    arguments_fragment: fragment.arguments,
                                },
                            )
                            .await?;
                        }
                    }
                    if let Some(content) = delta.and_then(|d| d.content) {
                        if !content.is_empty() {
                            output_bytes += content.len();
                            if output_bytes + assembler.bytes > self.config.max_output_bytes {
                                return Err(RuntimeError::ResourceLimit {
                                    reason: format!(
                                        "generated output exceeds {} bytes",
                                        self.config.max_output_bytes
                                    ),
                                });
                            }
                            tracker.deltas += 1;
                            tracker
                                .first_token
                                .get_or_insert_with(|| tracker.started.elapsed());
                            Self::emit(sender, cancellation, ChatStreamEvent::Delta { content })
                                .await?;
                        }
                    }
                }
            }
            if at_eof {
                break;
            }
        }

        if !done && finish.is_none() {
            return Err(protocol(
                operation,
                "stream ended before the provider signalled completion",
            ));
        }
        // Assemble and validate every call before any proposal is exposed.
        for call in assembler.finish(operation)? {
            Self::emit(
                sender,
                cancellation,
                ChatStreamEvent::ToolCallCompleted { call },
            )
            .await?;
        }
        Self::emit(
            sender,
            cancellation,
            ChatStreamEvent::Completed(Box::new(ChatStreamSummary {
                finish_reason: finish,
                usage,
                metadata: self.metadata(&request.model, tracker, true),
                structured: None,
            })),
        )
        .await
    }
}

fn map_stream_error(body: &[u8]) -> RuntimeError {
    // Mid-stream errors carry no HTTP status; they are provider failures.
    let (_, message) = provider_error_text(body);
    RuntimeError::provider_unavailable(PROVIDER_NAME, format!("stream error: {message}"))
}

// ---- incremental SSE parsing ---------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SseEvent {
    #[allow(dead_code)]
    pub event: Option<String>,
    pub data: String,
}

#[derive(Debug, Default)]
pub(crate) struct SseBatch {
    pub events: Vec<SseEvent>,
    pub eof: bool,
}

/// Incremental Server-Sent Events parser. Input may be split at arbitrary
/// byte boundaries (including inside a UTF-8 sequence or between `\r` and
/// `\n`); only complete lines are decoded. Line and event size are bounded.
pub(crate) struct SseParser {
    line: Vec<u8>,
    data: String,
    has_data: bool,
    event: Option<String>,
    pending_cr: bool,
    first_line: bool,
    max: usize,
}

impl SseParser {
    pub fn new(max: usize) -> Self {
        Self {
            line: Vec::new(),
            data: String::new(),
            has_data: false,
            event: None,
            pending_cr: false,
            first_line: true,
            max,
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<SseBatch, String> {
        let mut batch = SseBatch::default();
        for &byte in bytes {
            if std::mem::take(&mut self.pending_cr) && byte == b'\n' {
                continue;
            }
            match byte {
                b'\r' => {
                    self.pending_cr = true;
                    self.end_line(&mut batch)?;
                }
                b'\n' => self.end_line(&mut batch)?,
                _ => {
                    self.line.push(byte);
                    if self.line.len() > self.max {
                        return Err("stream line exceeds the configured size limit".to_owned());
                    }
                }
            }
        }
        Ok(batch)
    }

    /// Called at end of input. Unlike the strict SSE rules, a final
    /// unterminated line/event is still dispatched: truncated JSON then fails
    /// to parse and is reported, rather than a `[DONE]` being lost.
    pub fn finish(&mut self) -> Result<SseBatch, String> {
        let mut batch = SseBatch::default();
        if !self.line.is_empty() {
            self.end_line(&mut batch)?;
        }
        self.dispatch(&mut batch);
        batch.eof = true;
        Ok(batch)
    }

    fn end_line(&mut self, batch: &mut SseBatch) -> Result<(), String> {
        let bytes = std::mem::take(&mut self.line);
        let mut line = String::from_utf8(bytes).map_err(|_| "stream is not valid UTF-8")?;
        if std::mem::take(&mut self.first_line) {
            if let Some(stripped) = line.strip_prefix('\u{feff}') {
                line = stripped.to_owned();
            }
        }
        if line.is_empty() {
            self.dispatch(batch);
            return Ok(());
        }
        if line.starts_with(':') {
            return Ok(());
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line.as_str(), ""),
        };
        match field {
            "data" => {
                if self.data.len() + value.len() + 1 > self.max {
                    return Err("stream event exceeds the configured size limit".to_owned());
                }
                if self.has_data {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.has_data = true;
            }
            "event" => self.event = Some(value.to_owned()),
            _ => {} // id, retry, and unknown fields are irrelevant here
        }
        Ok(())
    }

    fn dispatch(&mut self, batch: &mut SseBatch) {
        if self.has_data {
            batch.events.push(SseEvent {
                event: self.event.take(),
                data: std::mem::take(&mut self.data),
            });
        }
        self.has_data = false;
        self.event = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_split(input: &[u8], step: usize) -> Vec<String> {
        let mut parser = SseParser::new(1024);
        let mut out = Vec::new();
        for piece in input.chunks(step) {
            out.extend(
                parser
                    .push(piece)
                    .unwrap()
                    .events
                    .into_iter()
                    .map(|e| e.data),
            );
        }
        out.extend(parser.finish().unwrap().events.into_iter().map(|e| e.data));
        out
    }

    #[test]
    fn events_are_identical_at_every_chunk_size() {
        let input = "data: {\"a\":\"é😀\"}\n\n: comment\r\ndata: two\r\ndata: lines\r\n\r\ndata: [DONE]\n\n";
        let expected = vec!["{\"a\":\"é😀\"}", "two\nlines", "[DONE]"];
        for step in 1..=input.len() {
            assert_eq!(parse_split(input.as_bytes(), step), expected, "step {step}");
        }
    }

    #[test]
    fn lone_cr_terminators_and_bom() {
        assert_eq!(
            parse_split("\u{feff}data: x\r\rdata: y\r\r".as_bytes(), 2),
            vec!["x", "y"]
        );
    }

    #[test]
    fn unterminated_final_event_is_dispatched_at_eof() {
        assert_eq!(parse_split(b"data: [DONE]", 3), vec!["[DONE]"]);
    }

    #[test]
    fn oversized_lines_and_events_are_rejected() {
        let mut parser = SseParser::new(8);
        assert!(parser.push(b"data: 0123456789").is_err());
        let mut parser = SseParser::new(8);
        assert!(parser.push(b"data: 1234\ndata: 5678\n").is_err());
    }

    #[test]
    fn invalid_utf8_is_rejected() {
        let mut parser = SseParser::new(64);
        assert!(parser.push(b"data: \xff\xfe\n").is_err());
    }
}
