//! Generative inference entry points on [`Runtime`].
//!
//! The caller names the provider explicitly; there is no routing and no
//! fallback to another provider, model, or execution mode. Providers own
//! transport; the runtime owns request validation, concurrency admission,
//! structured-output validation, and the stream termination contract.

use super::Runtime;
use futures_util::StreamExt;
use jsonschema::Validator;
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_provider::{
    ChatRequest, ChatResponse, ChatStreamEvent, GenerationModelInfo, GenerationProvider,
    ProviderCapability, ResponseFormat, StructuredOutput, ToolCall, ToolChoice, ToolDefinition,
    DEFAULT_MAX_OUTPUT_BYTES,
};
use serde_json::Value;
use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};
use tokio::sync::OwnedSemaphorePermit;

const MAX_REPORTED_VIOLATIONS: usize = 16;
const MAX_VIOLATION_CHARS: usize = 200;

/// How generated text is checked for a requested response format.
enum OutputCheck {
    /// Valid JSON of any shape.
    Json,
    Schema(Arc<Validator>),
}

impl OutputCheck {
    /// Compiles the schema up front so an unusable schema is an
    /// `InvalidRequest` rather than a late generation failure.
    fn for_format(format: &ResponseFormat) -> RuntimeResult<Option<Self>> {
        match format {
            ResponseFormat::Text => Ok(None),
            ResponseFormat::JsonObject => Ok(Some(Self::Json)),
            ResponseFormat::JsonSchema { schema, .. } => jsonschema::validator_for(schema)
                .map(|validator| Some(Self::Schema(Arc::new(validator))))
                .map_err(|error| RuntimeError::InvalidRequest {
                    reason: format!("unusable JSON schema: {}", clip(&error.to_string())),
                }),
        }
    }

    fn check(&self, text: &str, format: &ResponseFormat) -> StructuredOutput {
        let value: Value = match serde_json::from_str(text) {
            Ok(value) => value,
            Err(error) => {
                return StructuredOutput::InvalidJson {
                    error: error.to_string(),
                }
            }
        };
        // `json_object` promises an object; a bare scalar or array is a
        // violation of the requested format.
        if matches!(format, ResponseFormat::JsonObject) && !value.is_object() {
            return StructuredOutput::SchemaViolation {
                errors: vec![
                    "JSON object output was requested but the value is not an object".to_owned(),
                ],
            };
        }
        match self {
            Self::Json => StructuredOutput::Validated { value },
            Self::Schema(validator) => {
                let errors: Vec<String> = validator
                    .iter_errors(&value)
                    .take(MAX_REPORTED_VIOLATIONS)
                    .map(|error| clip(&error.to_string()))
                    .collect();
                if errors.is_empty() {
                    StructuredOutput::Validated { value }
                } else {
                    StructuredOutput::SchemaViolation { errors }
                }
            }
        }
    }
}

/// Validates tool-call *proposals* against the definitions the caller offered.
/// A schema constrains arguments; passing it grants no authority and does not
/// mean the operation ran.
struct ToolChecks {
    validators: BTreeMap<String, Validator>,
    choice: Option<ToolChoice>,
}

impl ToolChecks {
    /// Compiles every tool schema up front so an unusable one is an
    /// `InvalidRequest` before anything is sent.
    fn compile(tools: &[ToolDefinition], choice: &Option<ToolChoice>) -> RuntimeResult<Self> {
        let mut validators = BTreeMap::new();
        for tool in tools {
            let validator = jsonschema::validator_for(&tool.input_schema).map_err(|error| {
                RuntimeError::InvalidRequest {
                    reason: format!(
                        "unusable input schema for tool {:?}: {}",
                        tool.name,
                        clip(&error.to_string())
                    ),
                }
            })?;
            validators.insert(tool.name.clone(), validator);
        }
        Ok(Self {
            validators,
            choice: choice.clone(),
        })
    }

    fn check(&self, calls: &[ToolCall]) -> RuntimeResult<()> {
        let invalid = |reason: String| Err(RuntimeError::InvalidOutput { reason });
        if calls.is_empty() {
            return Ok(());
        }
        if matches!(self.choice, Some(ToolChoice::None)) {
            return invalid("model proposed tool calls although tool_choice is none".to_owned());
        }
        for call in calls {
            let Some(validator) = self.validators.get(&call.name) else {
                return invalid(format!(
                    "model proposed a call to undefined tool {:?}",
                    call.name
                ));
            };
            if let Some(ToolChoice::Tool { name }) = &self.choice {
                if *name != call.name {
                    return invalid(format!(
                        "model proposed tool {:?} although {name:?} was required",
                        call.name
                    ));
                }
            }
            let paths: Vec<String> = validator
                .iter_errors(&call.arguments)
                .take(MAX_REPORTED_VIOLATIONS)
                .map(|error| error.instance_path().to_string())
                .collect();
            if !paths.is_empty() {
                // Locations only: argument values are not echoed into errors.
                return invalid(format!(
                    "arguments proposed for tool {:?} violate its input schema at: {}",
                    call.name,
                    paths.join(", ")
                ));
            }
        }
        Ok(())
    }
}

fn clip(text: &str) -> String {
    text.chars().take(MAX_VIOLATION_CHARS).collect()
}

impl Runtime {
    fn generation_provider(&self, name: &str) -> RuntimeResult<&Arc<dyn GenerationProvider>> {
        self.generation_providers.get(name).ok_or_else(|| {
            RuntimeError::provider_unavailable(name, ": no such generation provider is registered")
        })
    }

    /// Descriptors of registered generation providers. These are separate
    /// from typed-inference providers and never take part in their routing.
    pub fn generation_providers(&self) -> Vec<ProviderCapability> {
        self.generation_providers
            .iter()
            .map(|(name, provider)| {
                ProviderCapability::from_parts(name.clone(), provider.capabilities())
            })
            .collect()
    }

    /// Lists models from the named provider, where the upstream supports it.
    pub async fn list_generation_models(
        &self,
        provider: &str,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<Vec<GenerationModelInfo>> {
        self.generation_provider(provider)?
            .list_models(cancellation)
            .await
    }

    async fn admit(
        &self,
        cancellation: &Option<CancellationToken>,
    ) -> RuntimeResult<OwnedSemaphorePermit> {
        let acquire = self.inference_semaphore.clone().acquire_owned();
        let permit = match cancellation {
            Some(token) => tokio::select! {
                biased;
                _ = token.cancelled() => return Err(RuntimeError::Cancelled),
                permit = acquire => permit,
            },
            None => acquire.await,
        };
        permit.map_err(|_| RuntimeError::ResourceLimit {
            reason: "runtime is shut down".to_owned(),
        })
    }

    /// Single-response generation through the explicitly named provider.
    ///
    /// When a JSON response format is requested the returned
    /// [`ChatResponse::structured`] states whether the text parsed and
    /// validated; invalid output is reported, never repaired.
    pub async fn chat(
        &self,
        provider: &str,
        request: ChatRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<ChatResponse> {
        request.validate()?;
        let format = request.options.response_format.clone();
        let check = OutputCheck::for_format(&format)?;
        let tools = ToolChecks::compile(&request.options.tools, &request.options.tool_choice)?;
        let provider = self.generation_provider(provider)?;
        let _permit = self.admit(&cancellation).await?;
        let mut response = provider.chat(request, cancellation).await?;
        tools.check(&response.message.tool_calls)?;
        // A response that proposes tool calls is not an answer in the requested
        // format, so it is not checked as one (`structured` stays `None`).
        if let (Some(check), true) = (check, response.message.tool_calls.is_empty()) {
            response.structured = Some(check.check(&response.message.content, &format));
        }
        Ok(response)
    }

    /// Streaming generation. Yields `Started`, zero or more `Delta`, then
    /// exactly one `Completed` on success. Any failure or cancellation ends
    /// the stream with an error and never with `Completed`; a stream that ends
    /// without either is reported as a protocol error. Dropping the stream
    /// releases the upstream connection and the concurrency permit.
    pub async fn chat_stream(
        &self,
        provider: &str,
        request: ChatRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<ChatStreamEvent>>> {
        request.validate()?;
        let format = request.options.response_format.clone();
        let check = OutputCheck::for_format(&format)?;
        let tools = ToolChecks::compile(&request.options.tools, &request.options.tool_choice)?;
        let provider = self.generation_provider(provider)?;
        let permit = self.admit(&cancellation).await?;
        let inner = provider.chat_stream(request, cancellation.clone()).await?;

        struct State {
            inner: BoxStream<RuntimeResult<ChatStreamEvent>>,
            // Text is retained only when it must be validated, and is bounded.
            accumulated: Option<String>,
            check: Option<OutputCheck>,
            tools: ToolChecks,
            // Proposals are held until the whole response validates, so a
            // consumer never sees a partial set.
            held: Vec<ToolCall>,
            outbox: VecDeque<RuntimeResult<ChatStreamEvent>>,
            format: ResponseFormat,
            cancellation: Option<CancellationToken>,
            completed: bool,
            finished: bool,
            _permit: OwnedSemaphorePermit,
        }
        let state = State {
            inner,
            accumulated: check.is_some().then(String::new),
            check,
            tools,
            held: Vec::new(),
            outbox: VecDeque::new(),
            format,
            cancellation,
            completed: false,
            finished: false,
            _permit: permit,
        };
        Ok(Box::pin(futures_util::stream::unfold(
            state,
            |mut s| async move {
                loop {
                    if let Some(item) = s.outbox.pop_front() {
                        return Some((item, s));
                    }
                    if s.finished {
                        return None;
                    }
                    let item = match s.inner.next().await {
                        Some(Ok(ChatStreamEvent::Delta { content })) => {
                            if let Some(text) = &mut s.accumulated {
                                if text.len() + content.len() > DEFAULT_MAX_OUTPUT_BYTES {
                                    s.finished = true;
                                    return Some((
                                        Err(RuntimeError::ResourceLimit {
                                            reason:
                                                "structured output exceeds the validation buffer"
                                                    .to_owned(),
                                        }),
                                        s,
                                    ));
                                }
                                text.push_str(&content);
                            }
                            Ok(ChatStreamEvent::Delta { content })
                        }
                        Some(Ok(ChatStreamEvent::ToolCallCompleted { call })) => {
                            s.held.push(call);
                            continue;
                        }
                        Some(Ok(ChatStreamEvent::Completed(mut summary))) => {
                            s.completed = true;
                            s.finished = true;
                            match s.tools.check(&s.held) {
                                Err(error) => Err(error),
                                Ok(()) => {
                                    if s.held.is_empty() {
                                        if let (Some(check), Some(text)) =
                                            (&s.check, &s.accumulated)
                                        {
                                            summary.structured = Some(check.check(text, &s.format));
                                        }
                                    }
                                    let held = std::mem::take(&mut s.held);
                                    s.outbox.extend(held.into_iter().map(|call| {
                                        Ok(ChatStreamEvent::ToolCallCompleted { call })
                                    }));
                                    s.outbox.push_back(Ok(ChatStreamEvent::Completed(summary)));
                                    continue;
                                }
                            }
                        }
                        Some(Ok(event)) => Ok(event),
                        Some(Err(error)) => {
                            s.finished = true;
                            Err(error)
                        }
                        None if s.completed => return None,
                        None => {
                            s.finished = true;
                            Err(
                                if s.cancellation.as_ref().is_some_and(|t| t.is_cancelled()) {
                                    RuntimeError::Cancelled
                                } else {
                                    RuntimeError::Protocol {
                                        operation: "chat stream".to_owned(),
                                        reason: "stream ended without completion".to_owned(),
                                    }
                                },
                            )
                        }
                    };
                    return Some((item, s));
                }
            },
        )))
    }
}
