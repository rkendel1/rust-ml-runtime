//! Packaging probe: llama.cpp local generation behind Node-API.
//!
//! JSON in, JSON out, exactly like the shipped `GenerationClient`: the request
//! and response are the existing `ChatRequest`/`ChatResponse` contracts, and
//! the runtime still validates structured output. The model is a file the
//! caller already has; nothing is downloaded.

use ml_runtime::{CancellationToken, ChatRequest, Runtime, RuntimeError};
use ml_runtime_llama_provider::{LlamaConfig, LlamaProvider, PROVIDER_NAME};
use napi::{bindgen_prelude::PromiseRaw, Env, Error, Result, Status};
use napi_derive::napi;
use std::sync::Arc;

fn code(error: &RuntimeError) -> &'static str {
    match error {
        RuntimeError::InvalidRequest { .. } => "invalid_request",
        RuntimeError::ModelNotFound { .. } => "model_not_found",
        RuntimeError::ModelUnavailable { .. } => "model_unavailable",
        RuntimeError::ModelIntegrity { .. } => "model_integrity",
        RuntimeError::UnsupportedModel { .. } => "unsupported_model",
        RuntimeError::ResourceLimit { .. } => "resource_limit",
        RuntimeError::Cancelled => "cancelled",
        _ => "execution_failed",
    }
}

fn failure(error: RuntimeError) -> Error {
    Error::new(
        Status::GenericFailure,
        format!("[{}] {error}", code(&error)),
    )
}

fn bad_input(reason: impl std::fmt::Display) -> Error {
    Error::new(Status::InvalidArg, format!("[invalid_request] {reason}"))
}

/// Cancels in-flight generation from JavaScript.
#[napi]
pub struct LocalCancellation {
    token: CancellationToken,
}

#[napi]
impl LocalCancellation {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
        }
    }

    #[napi]
    pub fn cancel(&self) {
        self.token.cancel();
    }
}

impl Default for LocalCancellation {
    fn default() -> Self {
        Self::new()
    }
}

/// One explicitly configured local GGUF model.
#[napi]
pub struct LocalGenerationClient {
    runtime: Arc<Runtime>,
}

#[napi]
impl LocalGenerationClient {
    /// `{ "modelPath", "modelId", "sha256"?, "contextTokens"?, "maxOutputTokens"?,
    /// "maxConcurrentRequests"?, "gpuLayers"?, "threads"? }`. Loads the model
    /// synchronously (hashing it first when `sha256` is set).
    #[napi(constructor)]
    pub fn new(config_json: String) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_str(&config_json)
            .map_err(|e| bad_input(format!("invalid configuration: {e}")))?;
        let text = |key: &str| config.get(key).and_then(|v| v.as_str());
        let number = |key: &str| config.get(key).and_then(|v| v.as_u64());
        let (Some(path), Some(id)) = (text("modelPath"), text("modelId")) else {
            return Err(bad_input("modelPath and modelId are required"));
        };
        let mut llama = LlamaConfig::new(path, id);
        llama.expected_sha256 = text("sha256").map(str::to_owned);
        if let Some(v) = number("contextTokens") {
            llama.context_tokens = v as u32;
        }
        if let Some(v) = number("maxOutputTokens") {
            llama.max_output_tokens = v as u32;
        }
        if let Some(v) = number("maxConcurrentRequests") {
            llama.max_concurrent_requests = v as usize;
        }
        if let Some(v) = number("gpuLayers") {
            llama.gpu_layers = v as u32;
        }
        if let Some(v) = number("threads") {
            llama.threads = Some(v as i32);
        }
        let provider = LlamaProvider::new(llama).map_err(failure)?;
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
        cancellation: Option<&LocalCancellation>,
    ) -> Result<PromiseRaw<'env, String>> {
        let request: ChatRequest = serde_json::from_str(&request_json)
            .map_err(|e| bad_input(format!("invalid ChatRequest: {e}")))?;
        let token = cancellation.map(|c| c.token.clone());
        let runtime = self.runtime.clone();
        env.spawn_future(async move {
            let response = runtime
                .chat(PROVIDER_NAME, request, token)
                .await
                .map_err(failure)?;
            serde_json::to_string(&response)
                .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))
        })
    }

    /// Provider capability notes, including the ggml devices compiled in.
    #[napi(js_name = "describeJson")]
    pub fn describe_json(&self) -> Result<String> {
        serde_json::to_string(&self.runtime.generation_providers())
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))
    }
}
