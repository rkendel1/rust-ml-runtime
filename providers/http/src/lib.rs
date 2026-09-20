use async_trait::async_trait;
use futures_util::StreamExt;
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::InferenceStreamEvent;
use ml_runtime_inference::{InferenceChunk, InferenceRequest, InferenceResult};
use ml_runtime_protocol::{
    CapabilitiesResponse, InferRequest, InferResponse, ModelInfo, StreamInferResponse,
};
use ml_runtime_provider::{Provider, ProviderCapabilities, ProviderCapability};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Clone, Debug)]
pub struct HttpProvider {
    endpoint: String,
    client: reqwest::Client,
}

impl HttpProvider {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into().trim_end_matches('/').to_owned(),
            client: reqwest::Client::new(),
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub async fn discover_capabilities(&self) -> RuntimeResult<CapabilitiesResponse> {
        self.get_json("/v1/capabilities").await
    }

    pub async fn models(&self) -> RuntimeResult<Vec<ModelInfo>> {
        self.get_json("/v1/models").await
    }

    async fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> RuntimeResult<T> {
        self.client
            .get(format!("{}{}", self.endpoint, path))
            .send()
            .await
            .map_err(|error| RuntimeError::transport(self.name(), error.to_string()))?
            .error_for_status()
            .map_err(|error| RuntimeError::transport(self.name(), error.to_string()))?
            .json()
            .await
            .map_err(|error| RuntimeError::transport(self.name(), error.to_string()))
    }

    pub fn descriptor(&self) -> ProviderCapability {
        ProviderCapability::from_parts(self.name(), self.capabilities())
    }
}

#[async_trait]
impl Provider for HttpProvider {
    fn name(&self) -> &str {
        "remote"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            available: true,
            local: false,
            remote: true,
            streaming: true,
            cancellation: true,
            structured_output: true,
            batching: true,
            endpoint: Some(self.endpoint.clone()),
            notes: vec!["Versioned runtime HTTP protocol".to_owned()],
        }
    }

    async fn infer(
        &self,
        request: InferenceRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<InferenceResult> {
        let send = self
            .client
            .post(format!("{}/v1/infer", self.endpoint))
            .json(&InferRequest { request })
            .send();
        let response = if let Some(token) = cancellation {
            tokio::select! {
                _ = token.cancelled() => return Err(RuntimeError::Cancelled),
                response = send => response,
            }
        } else {
            send.await
        };
        let response =
            response.map_err(|error| RuntimeError::transport(self.name(), error.to_string()))?;
        let status = response.status();
        let body: InferResponse = response
            .json()
            .await
            .map_err(|error| RuntimeError::transport(self.name(), error.to_string()))?;
        if let Some(result) = body.result {
            return Ok(result);
        }
        let error = body
            .error
            .unwrap_or_else(|| ml_runtime_protocol::ErrorEnvelope {
                code: "internal_error".to_owned(),
                message: "invalid server response".to_owned(),
                retryable: false,
            });
        Err(map_error(status, error))
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>> {
        let response = self
            .client
            .post(format!("{}/v1/infer/stream", self.endpoint))
            .json(&InferRequest { request })
            .send()
            .await
            .map_err(|error| RuntimeError::transport(self.name(), error.to_string()))?;
        if !response.status().is_success() {
            let status = response.status();
            let body: StreamInferResponse = response
                .json()
                .await
                .map_err(|error| RuntimeError::transport(self.name(), error.to_string()))?;
            return Err(map_error(
                status,
                body.error.unwrap_or(ml_runtime_protocol::ErrorEnvelope {
                    code: "internal_error".to_owned(),
                    message: "invalid server response".to_owned(),
                    retryable: false,
                }),
            ));
        }
        let (sender, receiver) = mpsc::channel(32);
        let provider = self.name().to_owned();
        let token = cancellation;
        tokio::spawn(async move {
            let mut pending = String::new();
            let mut last_output = None;
            let mut bytes = response.bytes_stream();
            while let Some(chunk) = tokio::select! {
                chunk = bytes.next() => chunk,
                _ = async {
                    if let Some(token) = &token { token.cancelled().await; }
                }, if token.is_some() => {
                    let _ = sender.send(Err(RuntimeError::Cancelled)).await;
                    return;
                }
            } {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        let _ = sender
                            .send(Err(RuntimeError::transport(&provider, error.to_string())))
                            .await;
                        return;
                    }
                };
                pending.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(index) = pending.find('\n') {
                    let line = pending.drain(..=index).collect::<String>();
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let message: StreamInferResponse = match serde_json::from_str(line) {
                        Ok(message) => message,
                        Err(error) => {
                            let _ = sender
                                .send(Err(RuntimeError::transport(&provider, error.to_string())))
                                .await;
                            return;
                        }
                    };
                    let item = match (message.event, message.error) {
                        (Some(InferenceStreamEvent::Output(output)), _) => {
                            let previous = last_output.replace(output);
                            previous.map(|output| {
                                Ok(InferenceChunk {
                                    output,
                                    done: false,
                                    metadata: None,
                                })
                            })
                        }
                        (Some(InferenceStreamEvent::Started(_)), _) => None,
                        (Some(InferenceStreamEvent::Completed(metadata)), _) => {
                            last_output.take().map(|output| {
                                Ok(InferenceChunk {
                                    output,
                                    done: true,
                                    metadata: Some(metadata),
                                })
                            })
                        }
                        (None, Some(error)) => Some(Err(map_error(reqwest::StatusCode::OK, error))),
                        (None, None) => Some(Err(RuntimeError::transport(
                            &provider,
                            "empty stream event",
                        ))),
                    };
                    match item {
                        Some(Ok(item)) => {
                            if sender.send(Ok(item)).await.is_err() {
                                return;
                            }
                        }
                        Some(Err(error)) => {
                            let _ = sender.send(Err(error)).await;
                            return;
                        }
                        None => {}
                    }
                }
            }
        });
        Ok(Box::pin(ReceiverStream::new(receiver)))
    }
}

fn map_error(
    status: reqwest::StatusCode,
    error: ml_runtime_protocol::ErrorEnvelope,
) -> RuntimeError {
    match error.code.as_str() {
        "model_not_found" => RuntimeError::ModelNotFound {
            model: error.message,
        },
        "invalid_request" => RuntimeError::InvalidInput {
            reason: error.message,
        },
        "invalid_input" => RuntimeError::InvalidInput {
            reason: error.message,
        },
        "timeout" => RuntimeError::Timeout {
            operation: "remote inference".to_owned(),
        },
        "request_cancelled" => RuntimeError::Cancelled,
        "backend_unavailable" => RuntimeError::BackendUnavailable {
            backend: "remote".to_owned(),
            reason: error.message,
        },
        "model_unavailable" | "unsupported_model_format" => RuntimeError::CapabilityMismatch {
            reason: error.message,
        },
        _ => RuntimeError::transport("remote", format!("HTTP {status}: {}", error.message)),
    }
}
