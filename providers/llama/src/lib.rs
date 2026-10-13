//! Embedded llama.cpp (GGUF) implementation of [`GenerationProvider`].
//!
//! A spike that answers one question: can rust-ml-runtime run a local
//! generative model in-process behind its existing generation contract? The
//! provider loads exactly one explicitly configured GGUF file, never downloads
//! anything, and rejects every feature it cannot honor:
//!
//! * supported: streaming, sampling (`temperature`, `top_p`, `seed`),
//!   `max_tokens`, stop sequences, grammar-constrained `json_object` and
//!   `json_schema` output (llama.cpp GBNF sampler);
//! * rejected with `CapabilityMismatch`: tools, tool choice.
//!
//! Model identity is explicit: a request for any model id other than the
//! configured one fails with `ModelNotFound`; nothing is substituted.

pub mod synthetic;

use async_trait::async_trait;
use futures_util::StreamExt as _;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, LlamaChatMessage, LlamaModel},
    sampling::LlamaSampler,
};
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_provider::{
    ChatMessage, ChatMetadata, ChatRequest, ChatResponse, ChatRole, ChatStreamEvent,
    ChatStreamSummary, ExecutionLocation, FinishReason, GenerationCapabilities,
    GenerationModelInfo, GenerationProvider, ProviderCapabilities, ResponseFormat, Usage,
    DEFAULT_MAX_OUTPUT_BYTES,
};
use sha2::{Digest, Sha256};
use std::{
    fmt,
    io::Read,
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tokio_stream::wrappers::ReceiverStream;

pub const PROVIDER_NAME: &str = "llama-cpp";
const TRANSPORT: &str = "in-process";
const BACKEND_NAME: &str = "llama.cpp";

#[derive(Clone, Debug)]
pub struct LlamaConfig {
    /// Path to a GGUF file that is already on disk.
    pub model_path: PathBuf,
    /// The only model id this provider answers to (what `ChatRequest.model` must be).
    pub model_id: String,
    /// Hex SHA-256 of the file. When set it is verified at load and a mismatch
    /// is a `ModelIntegrity` error. Hashing reads the whole file once.
    pub expected_sha256: Option<String>,
    /// Context window per request (prompt + generated tokens).
    pub context_tokens: u32,
    /// Upper bound on generated tokens per request; also the default.
    pub max_output_tokens: u32,
    /// Maximum requests executing at once (each owns a KV cache of
    /// `context_tokens`). Further requests wait, cancellably.
    pub max_concurrent_requests: usize,
    /// Prompt-processing chunk. Cancellation is observed between chunks.
    pub batch_tokens: u32,
    /// CPU threads per request (llama.cpp default when `None`).
    pub threads: Option<i32>,
    /// Layers offloaded to an accelerator. `0` means CPU only. Only meaningful
    /// when the crate is built with a llama.cpp accelerator feature.
    pub gpu_layers: u32,
    /// Overrides the template embedded in the GGUF.
    pub chat_template: Option<String>,
    /// Bounded channel depth between the generation thread and the consumer.
    pub stream_buffer: usize,
}

impl LlamaConfig {
    pub fn new(model_path: impl Into<PathBuf>, model_id: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            model_id: model_id.into(),
            expected_sha256: None,
            context_tokens: 2048,
            max_output_tokens: 512,
            max_concurrent_requests: 1,
            batch_tokens: 512,
            threads: None,
            gpu_layers: 0,
            chat_template: None,
            stream_buffer: 32,
        }
    }
}

/// Facts about the model that was actually loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedModelInfo {
    pub id: String,
    /// `general.name` from the GGUF, falling back to the file name.
    pub reported_name: String,
    pub architecture: String,
    pub parameters: u64,
    pub file_bytes: u64,
    pub train_context: u32,
    pub load_time: Duration,
}

pub struct LlamaProvider {
    inner: Arc<Inner>,
}

struct Inner {
    config: LlamaConfig,
    info: LoadedModelInfo,
    file_name: String,
    backend: &'static LlamaBackend,
    model: Arc<LlamaModel>,
    template: Option<String>,
    admission: Arc<Semaphore>,
    // llama.cpp mutates shared model state while creating a context, so
    // context creation is serialized.
    context_creation: Mutex<()>,
}

impl fmt::Debug for LlamaProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlamaProvider")
            .field("model", &self.inner.info.id)
            .field("reported_name", &self.inner.info.reported_name)
            .finish()
    }
}

fn backend() -> RuntimeResult<&'static LlamaBackend> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            LlamaBackend::init()
                .map(|mut backend| {
                    // llama.cpp logs to stderr by default; keep the host quiet.
                    backend.void_logs();
                    backend
                })
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|reason| RuntimeError::backend_unavailable(BACKEND_NAME, reason.clone()))
}

fn sha256_hex(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

impl LlamaProvider {
    /// Loads the model. This is the explicit "load" step: it blocks for the
    /// duration of the load (call it from `spawn_blocking` in async code) and
    /// never touches the network.
    pub fn new(config: LlamaConfig) -> RuntimeResult<Self> {
        let invalid = |reason: &str| RuntimeError::InvalidRequest {
            reason: reason.to_owned(),
        };
        if config.model_id.trim().is_empty() {
            return Err(invalid("model_id must not be empty"));
        }
        for (name, value) in [
            ("context_tokens", config.context_tokens as usize),
            ("max_output_tokens", config.max_output_tokens as usize),
            ("max_concurrent_requests", config.max_concurrent_requests),
            ("batch_tokens", config.batch_tokens as usize),
            ("stream_buffer", config.stream_buffer),
        ] {
            if value == 0 {
                return Err(RuntimeError::InvalidRequest {
                    reason: format!("{name} must be greater than zero"),
                });
            }
        }
        if config.max_output_tokens >= config.context_tokens {
            return Err(invalid(
                "max_output_tokens must be smaller than context_tokens",
            ));
        }
        let id = config.model_id.clone();
        let unavailable = |reason: String| RuntimeError::ModelUnavailable {
            model: id.clone(),
            provider: PROVIDER_NAME.to_owned(),
            reason,
        };
        let metadata = std::fs::metadata(&config.model_path).map_err(|error| {
            unavailable(format!(
                "model file {} is not available ({error}); obtain it first, weights are never downloaded automatically",
                config.model_path.display()
            ))
        })?;
        if !metadata.is_file() {
            return Err(unavailable(format!(
                "{} is not a file",
                config.model_path.display()
            )));
        }
        let mut magic = [0u8; 4];
        std::fs::File::open(&config.model_path)
            .and_then(|mut f| f.read_exact(&mut magic))
            .map_err(|e| unavailable(format!("cannot read model file: {e}")))?;
        if &magic != b"GGUF" {
            return Err(RuntimeError::UnsupportedModel {
                model: id.clone(),
                backend: BACKEND_NAME.to_owned(),
                reason: "not a GGUF file (llama.cpp loads GGUF only)".to_owned(),
            });
        }
        if let Some(expected) = &config.expected_sha256 {
            let actual = sha256_hex(&config.model_path)
                .map_err(|e| unavailable(format!("cannot hash model file: {e}")))?;
            if !actual.eq_ignore_ascii_case(expected) {
                return Err(RuntimeError::ModelIntegrity {
                    model: id.clone(),
                    reason: "SHA-256 of the model file does not match expected_sha256".to_owned(),
                });
            }
        }

        let backend = backend()?;
        let started = Instant::now();
        let params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
        let model =
            LlamaModel::load_from_file(backend, &config.model_path, &params).map_err(|error| {
                RuntimeError::UnsupportedModel {
                    model: id.clone(),
                    backend: BACKEND_NAME.to_owned(),
                    reason: format!("llama.cpp could not load the model: {error}"),
                }
            })?;
        let file_name = config.model_path.file_name().map_or_else(
            || "model.gguf".to_owned(),
            |n| n.to_string_lossy().into_owned(),
        );
        let info = LoadedModelInfo {
            id: id.clone(),
            reported_name: model
                .meta_val_str("general.name")
                .unwrap_or_else(|_| file_name.clone()),
            architecture: model
                .meta_val_str("general.architecture")
                .unwrap_or_else(|_| "unknown".to_owned()),
            parameters: model.n_params(),
            file_bytes: metadata.len(),
            train_context: model.n_ctx_train(),
            load_time: started.elapsed(),
        };
        let template = config.chat_template.clone().or_else(|| {
            model
                .chat_template(None)
                .ok()
                .and_then(|t| t.to_string().ok())
        });
        Ok(Self {
            inner: Arc::new(Inner {
                admission: Arc::new(Semaphore::new(config.max_concurrent_requests)),
                config,
                info,
                file_name,
                backend,
                model: Arc::new(model),
                template,
                context_creation: Mutex::new(()),
            }),
        })
    }

    pub fn model_info(&self) -> &LoadedModelInfo {
        &self.inner.info
    }

    /// Requests that could start right now (free admission slots).
    pub fn available_slots(&self) -> usize {
        self.inner.admission.available_permits()
    }
}

// ---- generation ------------------------------------------------------------------

/// Withholds text that might be the start of a stop sequence until it is
/// resolved, so stop sequences split across tokens are honored and never leak.
struct StopFilter {
    stops: Vec<String>,
    held: String,
}

enum StopOutcome {
    Continue(String),
    Stopped(String),
}

impl StopFilter {
    fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            held: String::new(),
        }
    }

    fn push(&mut self, text: &str) -> StopOutcome {
        self.held.push_str(text);
        if let Some(index) = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()))
            .min()
        {
            let out = self.held[..index].to_owned();
            self.held.clear();
            return StopOutcome::Stopped(out);
        }
        // Longest suffix of `held` that is a proper prefix of some stop sequence.
        let mut keep = 0;
        for stop in &self.stops {
            for len in (1..stop.len().min(self.held.len() + 1)).rev() {
                if stop.is_char_boundary(len)
                    && self.held.is_char_boundary(self.held.len() - len)
                    && self.held.ends_with(&stop[..len])
                {
                    keep = keep.max(len);
                    break;
                }
            }
        }
        let cut = self.held.len() - keep;
        StopOutcome::Continue(self.held.drain(..cut).collect())
    }

    fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// Reassembles UTF-8 split across token pieces.
#[derive(Default)]
struct Utf8Buffer(Vec<u8>);

impl Utf8Buffer {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.0.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.0) {
                Ok(text) => {
                    out.push_str(text);
                    self.0.clear();
                    return out;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.0[..valid]).unwrap_or_default());
                    match error.error_len() {
                        Some(bad) => {
                            out.push('\u{fffd}');
                            self.0.drain(..valid + bad);
                        }
                        None => {
                            self.0.drain(..valid);
                            return out;
                        }
                    }
                }
            }
        }
    }
}

enum Flow {
    Continue,
    Stop,
}

struct Outcome {
    finish: FinishReason,
    prompt_tokens: u64,
    completion_tokens: u64,
    first_token: Option<Duration>,
    reported: String,
}

fn execution(operation: &str, reason: impl fmt::Display) -> RuntimeError {
    RuntimeError::execution(operation, reason.to_string())
}

impl Inner {
    fn check_model(&self, requested: &str) -> RuntimeResult<()> {
        if requested == self.config.model_id {
            Ok(())
        } else {
            Err(RuntimeError::ModelNotFound {
                model: requested.to_owned(),
            })
        }
    }

    fn build_prompt(&self, messages: &[ChatMessage]) -> RuntimeResult<String> {
        let unsupported = |reason: &str| RuntimeError::UnsupportedModel {
            model: self.config.model_id.clone(),
            backend: BACKEND_NAME.to_owned(),
            reason: reason.to_owned(),
        };
        let template = self.template.as_deref().ok_or_else(|| {
            unsupported("the model has no chat template; set LlamaConfig::chat_template")
        })?;
        let template = llama_cpp_2::model::LlamaChatTemplate::new(template)
            .map_err(|_| unsupported("chat template contains a NUL byte"))?;
        let chat = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    // Capability checks reject tools earlier; never drop a role silently.
                    ChatRole::Tool => {
                        return Err(RuntimeError::CapabilityMismatch {
                            reason: "provider does not support tool messages".to_owned(),
                        })
                    }
                };
                LlamaChatMessage::new(role.to_owned(), m.content.clone()).map_err(|_| {
                    RuntimeError::InvalidRequest {
                        reason: "message content must not contain NUL bytes".to_owned(),
                    }
                })
            })
            .collect::<RuntimeResult<Vec<_>>>()?;
        self.model
            .apply_chat_template(&template, &chat, true)
            .map_err(|_| {
                unsupported("llama.cpp's built-in templater does not recognize this chat template")
            })
    }

    fn sampler(&self, request: &ChatRequest) -> RuntimeResult<LlamaSampler> {
        let o = &request.options;
        let mut chain = Vec::new();
        let grammar = match &o.response_format {
            ResponseFormat::Text => None,
            ResponseFormat::JsonObject => Some(r#"{"type":"object"}"#.to_owned()),
            ResponseFormat::JsonSchema { schema, .. } => Some(schema.to_string()),
        };
        if let Some(schema) = grammar {
            let gbnf = llama_cpp_2::json_schema_to_grammar(&schema).map_err(|e| {
                RuntimeError::InvalidRequest {
                    reason: format!("schema cannot be converted to a grammar: {e}"),
                }
            })?;
            chain.push(
                LlamaSampler::grammar(&self.model, &gbnf, "root").map_err(|e| {
                    RuntimeError::InvalidRequest {
                        reason: format!("grammar rejected by llama.cpp: {e}"),
                    }
                })?,
            );
        }
        match o.temperature {
            Some(t) if t > 0.0 => {
                if let Some(p) = o.top_p {
                    chain.push(LlamaSampler::top_p(p, 1));
                }
                chain.push(LlamaSampler::temp(t));
                let seed = o.seed.map_or_else(
                    || {
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_or(0, |d| d.subsec_nanos())
                    },
                    |s| s as u32,
                );
                chain.push(LlamaSampler::dist(seed));
            }
            // Unspecified or zero temperature is greedy and deterministic;
            // `seed` and `top_p` cannot change a greedy choice.
            _ => chain.push(LlamaSampler::greedy()),
        }
        Ok(LlamaSampler::chain_simple(chain))
    }

    /// The blocking generation loop. `emit` receives text and says whether to go on.
    fn generate(
        &self,
        request: &ChatRequest,
        cancellation: &Option<CancellationToken>,
        mut emit: impl FnMut(String) -> Flow,
    ) -> RuntimeResult<Outcome> {
        let started = Instant::now();
        let cancelled = || cancellation.as_ref().is_some_and(|t| t.is_cancelled());
        let o = &request.options;
        let max_new = o.max_tokens.unwrap_or(self.config.max_output_tokens);
        if max_new > self.config.max_output_tokens {
            return Err(RuntimeError::ResourceLimit {
                reason: format!(
                    "max_tokens {max_new} exceeds this provider's limit of {}",
                    self.config.max_output_tokens
                ),
            });
        }
        let prompt = self.build_prompt(&request.messages)?;
        let vocab = self.model.vocab();
        let tokens = vocab.tokenize(prompt.as_bytes(), true, true);
        if tokens.len() as u64 + u64::from(max_new) > u64::from(self.config.context_tokens) {
            return Err(RuntimeError::ResourceLimit {
                reason: format!(
                    "prompt ({} tokens) plus max_tokens ({max_new}) exceeds the {}-token context window",
                    tokens.len(),
                    self.config.context_tokens
                ),
            });
        }
        let mut sampler = self.sampler(request)?;
        if cancelled() {
            return Err(RuntimeError::Cancelled);
        }

        let mut params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.config.context_tokens))
            .with_n_batch(self.config.batch_tokens)
            .with_n_ubatch(self.config.batch_tokens);
        if let Some(threads) = self.config.threads {
            params = params.with_n_threads(threads);
        }
        let mut ctx = {
            let _guard = self
                .context_creation
                .lock()
                .map_err(|_| execution("create context", "context lock poisoned"))?;
            self.model.new_context(self.backend, params).map_err(|e| {
                RuntimeError::ResourceLimit {
                    reason: format!("could not allocate an inference context: {e}"),
                }
            })?
        };

        let chunk = self.config.batch_tokens as usize;
        let mut batch = LlamaBatch::new(chunk.max(1), 1);
        let last = tokens.len() - 1;
        for (start, piece) in tokens.chunks(chunk).enumerate() {
            batch.clear();
            for (i, token) in piece.iter().enumerate() {
                let position = start * chunk + i;
                batch
                    .add(*token, position as i32, &[0], position == last)
                    .map_err(|e| execution("prompt batch", e))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| execution("decode prompt", e))?;
            // Cancellation is observed between prompt chunks and between tokens.
            if cancelled() {
                return Err(RuntimeError::Cancelled);
            }
        }

        let mut position = tokens.len() as i32;
        let mut stop = StopFilter::new(o.stop.clone());
        let mut utf8 = Utf8Buffer::default();
        let mut produced = 0u64;
        let mut first_token = None;
        let mut output_bytes = 0usize;
        let mut finish = FinishReason::Length;
        let mut send = |text: String, output_bytes: &mut usize| -> RuntimeResult<Flow> {
            if text.is_empty() {
                return Ok(Flow::Continue);
            }
            *output_bytes += text.len();
            if *output_bytes > DEFAULT_MAX_OUTPUT_BYTES {
                return Err(RuntimeError::ResourceLimit {
                    reason: format!("generated output exceeds {DEFAULT_MAX_OUTPUT_BYTES} bytes"),
                });
            }
            Ok(emit(text))
        };
        'generate: while produced < u64::from(max_new) {
            if cancelled() {
                return Err(RuntimeError::Cancelled);
            }
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            if vocab.is_eog(token) {
                finish = FinishReason::Stop;
                break;
            }
            produced += 1;
            first_token.get_or_insert_with(|| started.elapsed());
            let text = utf8.push(&vocab.token_to_piece(token, false, None));
            match stop.push(&text) {
                StopOutcome::Continue(out) => {
                    if matches!(send(out, &mut output_bytes)?, Flow::Stop) {
                        return Err(RuntimeError::Cancelled);
                    }
                }
                StopOutcome::Stopped(out) => {
                    if matches!(send(out, &mut output_bytes)?, Flow::Stop) {
                        return Err(RuntimeError::Cancelled);
                    }
                    finish = FinishReason::Stop;
                    break 'generate;
                }
            }
            if produced >= u64::from(max_new) {
                break;
            }
            batch.clear();
            batch
                .add(token, position, &[0], true)
                .map_err(|e| execution("token batch", e))?;
            position += 1;
            ctx.decode(&mut batch)
                .map_err(|e| execution("decode token", e))?;
        }
        // Natural end or length: text still held back is real output. After a
        // stop sequence the filter is already empty.
        let rest = stop.flush();
        if matches!(send(rest, &mut output_bytes)?, Flow::Stop) {
            return Err(RuntimeError::Cancelled);
        }
        Ok(Outcome {
            finish,
            prompt_tokens: tokens.len() as u64,
            completion_tokens: produced,
            first_token,
            reported: self.info.reported_name.clone(),
        })
    }

    fn metadata(
        &self,
        request: &ChatRequest,
        outcome: &Outcome,
        streamed: bool,
        events: u64,
        started: Instant,
    ) -> ChatMetadata {
        ChatMetadata {
            provider: PROVIDER_NAME.to_owned(),
            transport: TRANSPORT.to_owned(),
            execution: ExecutionLocation::Local,
            execution_inferred: false,
            endpoint: format!("local-file:{}", self.file_name),
            requested_model: request.model.clone(),
            reported_model: Some(outcome.reported.clone()),
            response_id: None,
            streamed,
            stream_events: streamed.then_some(events),
            time_to_first_token: outcome.first_token,
            duration: started.elapsed(),
            runtime_version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }
}

async fn admit(
    semaphore: &Arc<Semaphore>,
    cancellation: &Option<CancellationToken>,
) -> RuntimeResult<OwnedSemaphorePermit> {
    let acquire = semaphore.clone().acquire_owned();
    let permit = match cancellation {
        Some(token) => tokio::select! {
            biased;
            _ = token.cancelled() => return Err(RuntimeError::Cancelled),
            permit = acquire => permit,
        },
        None => acquire.await,
    };
    permit.map_err(|_| RuntimeError::ResourceLimit {
        reason: "provider is shut down".to_owned(),
    })
}

type EventSender = mpsc::Sender<RuntimeResult<ChatStreamEvent>>;

/// Delivers an event, applying backpressure but never blocking past
/// cancellation or the consumer going away.
fn deliver(
    sender: &EventSender,
    cancellation: &Option<CancellationToken>,
    mut item: RuntimeResult<ChatStreamEvent>,
) -> Flow {
    loop {
        match sender.try_send(item) {
            Ok(()) => return Flow::Continue,
            Err(mpsc::error::TrySendError::Closed(_)) => return Flow::Stop,
            Err(mpsc::error::TrySendError::Full(back)) => {
                if cancellation.as_ref().is_some_and(|t| t.is_cancelled()) {
                    return Flow::Stop;
                }
                item = back;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

#[async_trait]
impl GenerationProvider for LlamaProvider {
    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn capabilities(&self) -> ProviderCapabilities {
        let backend = if self.inner.config.gpu_layers > 0 {
            format!(
                "{BACKEND_NAME} with {} offloaded layers requested",
                self.inner.config.gpu_layers
            )
        } else {
            format!("{BACKEND_NAME} CPU (no accelerator layers requested)")
        };
        ProviderCapabilities {
            available: true,
            local: true,
            remote: false,
            streaming: true,
            cancellation: true,
            structured_output: true,
            batching: false,
            endpoint: Some(format!("local-file:{}", self.inner.file_name)),
            notes: vec![
                backend,
                format!("llama-cpp-2 {}", "0.1.159"),
                "cancellation is observed between tokens and prompt chunks".to_owned(),
            ],
        }
    }

    fn generation_capabilities(&self) -> GenerationCapabilities {
        GenerationCapabilities {
            streaming: true,
            stop_sequences: true,
            seed: true,
            json_object: true,
            json_schema: true,
            tools: false,
            tool_choice: false,
        }
    }

    async fn list_models(
        &self,
        _cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<Vec<GenerationModelInfo>> {
        let info = &self.inner.info;
        Ok(vec![GenerationModelInfo {
            id: info.id.clone(),
            owned_by: Some(format!(
                "local gguf {} \"{}\" ({} params)",
                info.architecture, info.reported_name, info.parameters
            )),
        }])
    }

    async fn chat(
        &self,
        request: ChatRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<ChatResponse> {
        request.validate()?;
        self.generation_capabilities().check(&request, false)?;
        self.inner.check_model(&request.model)?;
        let permit = admit(&self.inner.admission, &cancellation).await?;
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let started = Instant::now();
            let mut text = String::new();
            let outcome = inner.generate(&request, &cancellation, |piece| {
                text.push_str(&piece);
                Flow::Continue
            })?;
            Ok(ChatResponse {
                message: ChatMessage::assistant(text),
                finish_reason: Some(outcome.finish.clone()),
                usage: Some(Usage {
                    prompt_tokens: Some(outcome.prompt_tokens),
                    completion_tokens: Some(outcome.completion_tokens),
                    total_tokens: Some(outcome.prompt_tokens + outcome.completion_tokens),
                }),
                metadata: inner.metadata(&request, &outcome, false, 0, started),
                structured: None,
            })
        })
        .await
        .map_err(|e| execution("generation task", e))?
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<ChatStreamEvent>>> {
        request.validate()?;
        self.generation_capabilities().check(&request, true)?;
        self.inner.check_model(&request.model)?;
        let permit = admit(&self.inner.admission, &cancellation).await?;
        let (sender, receiver) = mpsc::channel(self.inner.config.stream_buffer);
        let inner = self.inner.clone();
        // One OS thread's worth of work per request, held under the admission
        // permit. It ends when generation ends, the consumer drops the stream,
        // or the token is cancelled; it cannot outlive an abandoned stream by
        // more than one token (or one prompt chunk).
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let started = Instant::now();
            let first = ChatStreamEvent::Started {
                provider: PROVIDER_NAME.to_owned(),
                model: request.model.clone(),
            };
            if matches!(deliver(&sender, &cancellation, Ok(first)), Flow::Stop) {
                return;
            }
            let mut events = 0u64;
            let result = inner.generate(&request, &cancellation, |content| {
                events += 1;
                deliver(
                    &sender,
                    &cancellation,
                    Ok(ChatStreamEvent::Delta { content }),
                )
            });
            let last = match result {
                Ok(outcome) => Ok(ChatStreamEvent::Completed(Box::new(ChatStreamSummary {
                    finish_reason: Some(outcome.finish.clone()),
                    usage: Some(Usage {
                        prompt_tokens: Some(outcome.prompt_tokens),
                        completion_tokens: Some(outcome.completion_tokens),
                        total_tokens: Some(outcome.prompt_tokens + outcome.completion_tokens),
                    }),
                    metadata: inner.metadata(&request, &outcome, true, events, started),
                    structured: None,
                }))),
                // The consumer dropped the stream: nobody is listening.
                Err(RuntimeError::Cancelled) if sender.is_closed() => return,
                Err(error) => Err(error),
            };
            let _ = deliver(&sender, &cancellation, last);
        });
        let stream: BoxStream<RuntimeResult<ChatStreamEvent>> =
            Box::pin(ReceiverStream::new(receiver).map(|item| item));
        Ok(stream)
    }
}
