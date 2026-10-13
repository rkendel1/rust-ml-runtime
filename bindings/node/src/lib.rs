//! In-process Node-API boundary for model-neutral typed decisions.
//!
//! Model installation, backend selection, preprocessing, compiled caches, and
//! output decoding remain owned by `ml-runtime`. Consumers exchange only the
//! public `DecisionRequest` and `DecisionResult` JSON contracts.
//!
//! The generative contract (`ChatRequest`, `ChatResponse`, `ChatStreamEvent`)
//! is exposed the same way. Transport, SSE parsing, validation, cancellation,
//! and credentials stay in Rust; JavaScript only exchanges JSON.

use futures_util::StreamExt;
use ml_runtime::{
    ApiKey, ChatRequest, ExecutionLocation, OpenAiCompatibleConfig, OpenAiCompatibleProvider,
    RuntimeError,
};
use ml_runtime::{
    CancellationToken, DecisionExecutionPolicy, DecisionGraphRequest, DecisionModel,
    DecisionRequest, InferenceRequest, ModelFormat, ModelLocation, ModelSpec, Runtime,
};
use napi::{
    bindgen_prelude::{AsyncTask, PromiseRaw},
    Env, Error, Result, Status, Task,
};
use napi_derive::napi;
use std::{sync::Arc, time::Duration};

fn native_error(context: &str, error: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, format!("{context}: {error}"))
}

/// Runs a minimal in-process CPU inference through the same native runtime
/// linked into the addon. Release smoke tests use this to prove more than a
/// successful dynamic-library load.
#[napi(js_name = "runtimeSmokeTest")]
pub fn runtime_smoke_test() -> Result<String> {
    let model = ModelSpec::new(
        "node-release-smoke",
        ModelFormat::Unknown,
        ModelLocation::Memory,
    );
    let executor = tokio::runtime::Runtime::new()
        .map_err(|error| native_error("could not initialize runtime smoke executor", error))?;
    let result = executor
        .block_on(Runtime::new().infer(InferenceRequest::new(model, "bookworm-smoke")))
        .map_err(|error| native_error("runtime smoke inference failed", error))?;
    serde_json::to_string(&result)
        .map_err(|error| native_error("could not encode runtime smoke inference", error))
}

#[napi]
pub struct LocalDecisionModel {
    model: std::sync::Arc<dyn DecisionModel>,
}

/// Cooperative cancellation handle for queued or multi-stage decision work.
/// Backends report separately whether an already-running native operation can
/// be interrupted.
#[napi]
pub struct DecisionCancellation {
    token: CancellationToken,
}

impl Default for DecisionCancellation {
    fn default() -> Self {
        Self {
            token: CancellationToken::new(),
        }
    }
}

enum DecisionWork {
    Independent(DecisionRequest, DecisionExecutionPolicy),
    Graph(DecisionGraphRequest),
}

pub struct DecisionTask {
    model: std::sync::Arc<dyn DecisionModel>,
    work: Option<DecisionWork>,
    cancellation: Option<CancellationToken>,
}

impl Task for DecisionTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<Self::Output> {
        let work = self.work.take().ok_or_else(|| {
            native_error(
                "decision task could not start",
                "request was already consumed",
            )
        })?;
        let executor = tokio::runtime::Runtime::new()
            .map_err(|error| native_error("could not initialize decision executor", error))?;
        let runtime = Runtime::new();
        let result = executor
            .block_on(async {
                match work {
                    DecisionWork::Independent(request, policy) => {
                        runtime
                            .decide_async(
                                self.model.clone(),
                                request,
                                policy,
                                self.cancellation.clone(),
                            )
                            .await
                    }
                    DecisionWork::Graph(request) => {
                        runtime
                            .execute_decision_graph(
                                self.model.clone(),
                                request,
                                self.cancellation.clone(),
                            )
                            .await
                    }
                }
            })
            .map_err(|error| native_error("asynchronous decision execution failed", error))?;
        serde_json::to_string(&result)
            .map_err(|error| native_error("could not encode DecisionResult", error))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

#[napi]
impl DecisionCancellation {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[napi]
    pub fn cancel(&self) {
        self.token.cancel();
    }

    #[napi(getter, js_name = "isCancelled")]
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }
}

#[napi]
impl LocalDecisionModel {
    /// Loads and validates one installed model. The returned object owns the
    /// loaded native model and reuses it for every subsequent decision.
    #[napi(constructor)]
    pub fn new(model: String, model_root: Option<String>) -> Result<Self> {
        let mut builder = Runtime::builder();
        if let Some(root) = model_root {
            builder = builder.model_root(root);
        }
        let loaded = builder
            .build()
            .load_model(&model)
            .map_err(|error| native_error(&format!("could not load model {model:?}"), error))?;
        Ok(Self {
            model: std::sync::Arc::from(loaded),
        })
    }

    /// Returns the runtime-owned model description without exposing any
    /// backend-specific representation.
    #[napi(js_name = "descriptionJson")]
    pub fn description_json(&self) -> Result<String> {
        serde_json::to_string(&self.model.describe())
            .map_err(|error| native_error("could not encode model description", error))
    }

    /// Returns model/backend/device execution facts consumed by the planner.
    #[napi(js_name = "capabilitiesJson")]
    pub fn capabilities_json(&self) -> Result<String> {
        serde_json::to_string(&self.model.capabilities())
            .map_err(|error| native_error("could not encode model capabilities", error))
    }

    /// Produces the deterministic execution plan without running inference.
    #[napi(js_name = "explainDecisionJson")]
    pub fn explain_decision_json(&self, request_json: String) -> Result<String> {
        let request: DecisionGraphRequest = serde_json::from_str(&request_json)
            .map_err(|error| native_error("invalid DecisionGraphRequest JSON", error))?;
        let plan = Runtime::new()
            .explain_decision(self.model.as_ref(), &request)
            .map_err(|error| native_error("could not plan decision graph", error))?;
        serde_json::to_string(&plan)
            .map_err(|error| native_error("could not encode decision plan", error))
    }

    /// Executes the runtime's model-neutral typed-decision contract.
    #[napi(js_name = "decideJson")]
    pub fn decide_json(&self, request_json: String) -> Result<String> {
        let request: DecisionRequest = serde_json::from_str(&request_json)
            .map_err(|error| native_error("invalid DecisionRequest JSON", error))?;
        let result = self
            .model
            .decide(&request)
            .map_err(|error| native_error("local model inference failed", error))?;
        serde_json::to_string(&result)
            .map_err(|error| native_error("could not encode DecisionResult", error))
    }

    /// Runs independent typed decisions away from the JavaScript event loop.
    #[napi(js_name = "decideAsyncJson")]
    pub fn decide_async_json(
        &self,
        request_json: String,
        policy_json: Option<String>,
        cancellation: Option<&DecisionCancellation>,
    ) -> Result<AsyncTask<DecisionTask>> {
        let request: DecisionRequest = serde_json::from_str(&request_json)
            .map_err(|error| native_error("invalid DecisionRequest JSON", error))?;
        let policy = policy_json
            .map(|json| {
                serde_json::from_str::<DecisionExecutionPolicy>(&json)
                    .map_err(|error| native_error("invalid DecisionExecutionPolicy JSON", error))
            })
            .transpose()?
            .unwrap_or_default();
        let token = cancellation.map(|handle| handle.token.clone());
        Ok(AsyncTask::new(DecisionTask {
            model: self.model.clone(),
            work: Some(DecisionWork::Independent(request, policy)),
            cancellation: token,
        }))
    }

    /// Executes a dependency-aware decision graph away from the JavaScript
    /// event loop, using only capabilities declared by the loaded model.
    #[napi(js_name = "executeGraphJson")]
    pub fn execute_graph_json(
        &self,
        request_json: String,
        cancellation: Option<&DecisionCancellation>,
    ) -> Result<AsyncTask<DecisionTask>> {
        let request: DecisionGraphRequest = serde_json::from_str(&request_json)
            .map_err(|error| native_error("invalid DecisionGraphRequest JSON", error))?;
        let token = cancellation.map(|handle| handle.token.clone());
        Ok(AsyncTask::new(DecisionTask {
            model: self.model.clone(),
            work: Some(DecisionWork::Graph(request)),
            cancellation: token,
        }))
    }
}

// ---- generative inference ------------------------------------------------------

const GENERATION_PROVIDER: &str = "openai-compatible";

fn error_code(error: &RuntimeError) -> &'static str {
    match error {
        RuntimeError::InvalidRequest { .. } => "invalid_request",
        RuntimeError::ModelNotFound { .. } => "model_not_found",
        RuntimeError::ModelUnavailable { .. } => "model_unavailable",
        RuntimeError::ModelIntegrity { .. } => "model_integrity",
        RuntimeError::UnsupportedModel { .. } => "unsupported_model",
        RuntimeError::BackendUnavailable { .. } => "backend_unavailable",
        RuntimeError::ProviderUnavailable { .. } => "provider_unavailable",
        RuntimeError::InvalidInput { .. } => "invalid_input",
        RuntimeError::UnsupportedInput { .. } => "unsupported_input",
        RuntimeError::InvalidOutput { .. } => "invalid_output",
        RuntimeError::CapabilityMismatch { .. } => "capability_mismatch",
        RuntimeError::ResourceLimit { .. } => "resource_limit",
        RuntimeError::Timeout { .. } => "timeout",
        RuntimeError::Cancelled => "cancelled",
        RuntimeError::Transport { .. } => "transport",
        RuntimeError::Protocol { .. } => "protocol",
        RuntimeError::Execution { .. } => "execution",
    }
}

/// Errors cross the boundary as `[code] message`; the JavaScript wrapper
/// turns the prefix into `GenerationError.code`.
fn generation_error(error: RuntimeError) -> Error {
    Error::new(
        Status::GenericFailure,
        format!("[{}] {error}", error_code(&error)),
    )
}

fn bad_input(reason: impl std::fmt::Display) -> Error {
    Error::new(Status::InvalidArg, format!("[invalid_request] {reason}"))
}

fn config_u64(config: &serde_json::Value, key: &str) -> Result<Option<u64>> {
    match config.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .filter(|v| *v > 0)
            .map(Some)
            .ok_or_else(|| bad_input(format!("{key} must be a positive integer"))),
    }
}

fn config_string<'a>(config: &'a serde_json::Value, key: &str) -> Result<Option<&'a str>> {
    match config.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(Some)
            .ok_or_else(|| bad_input(format!("{key} must be a string"))),
    }
}

fn provider_config(config_json: &str) -> Result<OpenAiCompatibleConfig> {
    let config: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| bad_input(format!("invalid config: {e}")))?;
    let endpoint =
        config_string(&config, "endpoint")?.ok_or_else(|| bad_input("endpoint is required"))?;
    let mut out = OpenAiCompatibleConfig::new(endpoint);
    match (
        config_string(&config, "apiKey")?,
        config_string(&config, "apiKeyEnv")?,
    ) {
        (Some(_), Some(_)) => return Err(bad_input("set only one of apiKey and apiKeyEnv")),
        (Some(key), None) => out.api_key = Some(ApiKey::new(key).map_err(generation_error)?),
        (None, Some(name)) => {
            let key = std::env::var(name)
                .map_err(|_| bad_input(format!("environment variable {name} is not set")))?;
            out.api_key = Some(ApiKey::new(key).map_err(generation_error)?);
        }
        (None, None) => {}
    }
    out.location = match config_string(&config, "location")? {
        None => None,
        Some("local") => Some(ExecutionLocation::Local),
        Some("remote") => Some(ExecutionLocation::Remote),
        Some(_) => return Err(bad_input("location must be \"local\" or \"remote\"")),
    };
    if let Some(capabilities) = config.get("capabilities").filter(|v| !v.is_null()) {
        for (key, slot) in [
            ("streaming", &mut out.capabilities.streaming),
            ("stopSequences", &mut out.capabilities.stop_sequences),
            ("seed", &mut out.capabilities.seed),
            ("jsonObject", &mut out.capabilities.json_object),
            ("jsonSchema", &mut out.capabilities.json_schema),
        ] {
            if let Some(value) = capabilities.get(key) {
                *slot = value
                    .as_bool()
                    .ok_or_else(|| bad_input(format!("capabilities.{key} must be a boolean")))?;
            }
        }
    }
    for (key, slot) in [
        ("connectTimeoutMs", &mut out.connect_timeout),
        ("requestTimeoutMs", &mut out.request_timeout),
        ("streamIdleTimeoutMs", &mut out.stream_idle_timeout),
    ] {
        if let Some(ms) = config_u64(&config, key)? {
            *slot = Duration::from_millis(ms);
        }
    }
    Ok(out)
}

fn parse_chat_request(request_json: &str) -> Result<ChatRequest> {
    serde_json::from_str(request_json).map_err(|e| bad_input(format!("invalid ChatRequest: {e}")))
}

fn encode<T: serde::Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(value).map_err(|e| native_error("could not encode result", e))
}

/// Native owner of one OpenAI-compatible endpoint. It holds the provider (and
/// so any credential) and never exposes it back to JavaScript.
#[napi]
pub struct GenerationClient {
    runtime: Arc<Runtime>,
}

#[napi]
impl GenerationClient {
    #[napi(constructor)]
    pub fn new(config_json: String) -> Result<Self> {
        let provider = OpenAiCompatibleProvider::new(provider_config(&config_json)?)
            .map_err(generation_error)?;
        Ok(Self {
            runtime: Arc::new(
                Runtime::builder()
                    .register_generation_provider(provider)
                    .build(),
            ),
        })
    }

    #[napi(js_name = "chatJson")]
    pub fn chat_json<'env>(
        &self,
        env: &'env Env,
        request_json: String,
        cancellation: Option<&DecisionCancellation>,
    ) -> Result<PromiseRaw<'env, String>> {
        let request = parse_chat_request(&request_json)?;
        let token = cancellation.map(|c| c.token.clone());
        let runtime = self.runtime.clone();
        env.spawn_future(async move {
            let response = runtime
                .chat(GENERATION_PROVIDER, request, token)
                .await
                .map_err(generation_error)?;
            encode(&response)
        })
    }

    #[napi(js_name = "listModelsJson")]
    pub fn list_models_json<'env>(
        &self,
        env: &'env Env,
        cancellation: Option<&DecisionCancellation>,
    ) -> Result<PromiseRaw<'env, String>> {
        let token = cancellation.map(|c| c.token.clone());
        let runtime = self.runtime.clone();
        env.spawn_future(async move {
            let models = runtime
                .list_generation_models(GENERATION_PROVIDER, token)
                .await
                .map_err(generation_error)?;
            encode(&models)
        })
    }

    /// Creates a stream handle. The request is sent on the first `next()`.
    #[napi(js_name = "openChatStream")]
    pub fn open_chat_stream(
        &self,
        request_json: String,
        cancellation: Option<&DecisionCancellation>,
    ) -> Result<ChatStreamHandle> {
        let request = parse_chat_request(&request_json)?;
        // A child token lets `close()` stop this stream without cancelling the
        // caller's shared handle, while caller cancellation still reaches it.
        let token = cancellation
            .map(|c| c.token.child_token())
            .unwrap_or_default();
        Ok(ChatStreamHandle {
            inner: Arc::new(StreamInner {
                runtime: self.runtime.clone(),
                pending: std::sync::Mutex::new(Some(request)),
                stream: tokio::sync::Mutex::new(None),
                token,
            }),
        })
    }
}

type NativeStream = ml_runtime::ml_runtime_common::BoxStream<
    ml_runtime::RuntimeResult<ml_runtime::ChatStreamEvent>,
>;

struct StreamInner {
    runtime: Arc<Runtime>,
    pending: std::sync::Mutex<Option<ChatRequest>>,
    stream: tokio::sync::Mutex<Option<NativeStream>>,
    token: CancellationToken,
}

impl Drop for StreamInner {
    fn drop(&mut self) {
        // A handle collected without being closed must not leave an upstream
        // connection running.
        self.token.cancel();
    }
}

/// One in-flight chat stream. `next()` resolves one JSON event per call, then
/// `null` after the stream has ended.
#[napi]
pub struct ChatStreamHandle {
    inner: Arc<StreamInner>,
}

#[napi]
impl ChatStreamHandle {
    #[napi]
    pub fn next<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, Option<String>>> {
        let inner = self.inner.clone();
        env.spawn_future(async move {
            let mut guard = inner.stream.lock().await;
            if guard.is_none() {
                let request = inner
                    .pending
                    .lock()
                    .expect("stream request poisoned")
                    .take();
                let Some(request) = request else {
                    return Ok(None);
                };
                let opened = inner
                    .runtime
                    .chat_stream(GENERATION_PROVIDER, request, Some(inner.token.clone()))
                    .await
                    .map_err(generation_error)?;
                *guard = Some(opened);
            }
            let item = guard.as_mut().expect("stream was just opened").next().await;
            match item {
                Some(Ok(event)) => encode(&event).map(Some),
                Some(Err(error)) => {
                    *guard = None;
                    Err(generation_error(error))
                }
                None => {
                    *guard = None;
                    Ok(None)
                }
            }
        })
    }

    /// Stops the stream and releases the upstream connection. Idempotent.
    #[napi]
    pub fn close(&self) {
        self.inner.token.cancel();
        self.inner
            .pending
            .lock()
            .expect("stream request poisoned")
            .take();
        // If a `next()` is in flight the cancelled token ends it and the
        // stream is dropped there; otherwise drop it now.
        if let Ok(mut guard) = self.inner.stream.try_lock() {
            *guard = None;
        }
    }
}
