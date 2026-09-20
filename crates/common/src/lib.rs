use futures_core::Stream;
use std::pin::Pin;
use thiserror::Error;

pub use tokio_util::sync::CancellationToken;

pub type RuntimeResult<T> = Result<T, RuntimeError>;
pub type BoxStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("model not found: {model}")]
    ModelNotFound { model: String },
    #[error("unsupported model {model} for backend {backend}: {reason}")]
    UnsupportedModel {
        model: String,
        backend: String,
        reason: String,
    },
    #[error("backend unavailable: {backend}{reason}")]
    BackendUnavailable { backend: String, reason: String },
    #[error("provider unavailable: {provider}{reason}")]
    ProviderUnavailable { provider: String, reason: String },
    #[error("invalid input: {reason}")]
    InvalidInput { reason: String },
    #[error("invalid output: {reason}")]
    InvalidOutput { reason: String },
    #[error("capability mismatch: {reason}")]
    CapabilityMismatch { reason: String },
    #[error("resource limit exceeded: {reason}")]
    ResourceLimit { reason: String },
    #[error("timeout while {operation}")]
    Timeout { operation: String },
    #[error("operation cancelled")]
    Cancelled,
    #[error("transport error from {provider}: {reason}")]
    Transport { provider: String, reason: String },
    #[error("execution error in {operation}: {reason}")]
    Execution { operation: String, reason: String },
}

impl RuntimeError {
    pub fn backend_unavailable(backend: impl Into<String>, reason: impl Into<String>) -> Self {
        let reason = reason.into();
        Self::BackendUnavailable {
            backend: backend.into(),
            reason: if reason.is_empty() {
                String::new()
            } else {
                format!(": {reason}")
            },
        }
    }

    pub fn provider_unavailable(provider: impl Into<String>, reason: impl Into<String>) -> Self {
        let reason = reason.into();
        Self::ProviderUnavailable {
            provider: provider.into(),
            reason: if reason.is_empty() {
                String::new()
            } else {
                format!(": {reason}")
            },
        }
    }

    pub fn transport(provider: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Transport {
            provider: provider.into(),
            reason: reason.into(),
        }
    }

    pub fn execution(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Execution {
            operation: operation.into(),
            reason: reason.into(),
        }
    }
}
