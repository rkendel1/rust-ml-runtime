//! Generative inference through the public `Runtime` API, against an
//! in-process mock server (no network, no credentials).

#[path = "../../../providers/http/tests/support/mod.rs"]
mod support;

use async_trait::async_trait;
use futures_util::StreamExt;
use rust_ml_runtime::ml_runtime_common::BoxStream;
use rust_ml_runtime::ml_runtime_provider::{GenerationProvider, ProviderCapabilities};
use rust_ml_runtime::{
    CancellationToken, ChatMessage, ChatRequest, ChatResponse, ChatStreamEvent, ExecutionLocation,
    GenerationCapabilities, GenerationModelInfo, GenerationOptions, OpenAiCompatibleConfig,
    OpenAiCompatibleProvider, ResponseFormat, Runtime, RuntimeError, RuntimeResult,
    StructuredOutput,
};
use serde_json::json;
use std::time::Duration;
use support::{Mock, Reply};

fn runtime_for(mock: &Mock, structured: bool) -> Runtime {
    let mut config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url));
    config.capabilities.json_object = structured;
    config.capabilities.json_schema = structured;
    Runtime::builder()
        .register_generation_provider(OpenAiCompatibleProvider::new(config).unwrap())
        .build()
}

fn completion(content: &str) -> String {
    json!({"model":"m","choices":[{"message":{"role":"assistant","content":content},"finish_reason":"stop"}]}).to_string()
}

fn schema_request(schema: serde_json::Value) -> ChatRequest {
    ChatRequest::new("m", vec![ChatMessage::user("go")]).with_options(GenerationOptions {
        response_format: ResponseFormat::JsonSchema {
            name: "out".into(),
            schema,
            strict: true,
        },
        ..Default::default()
    })
}

fn int_schema() -> serde_json::Value {
    json!({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false})
}

fn sse_chunks(deltas: &[&str]) -> Vec<Vec<u8>> {
    let mut events: Vec<String> = deltas
        .iter()
        .map(|d| json!({"choices":[{"delta":{"content":d},"finish_reason":null}]}).to_string())
        .collect();
    events.push(json!({"choices":[{"delta":{},"finish_reason":"stop"}]}).to_string());
    events.push("[DONE]".into());
    events
        .iter()
        .map(|e| format!("data: {e}\n\n").into_bytes())
        .collect()
}

#[tokio::test]
async fn chat_reports_provider_model_transport_and_location() {
    let mock = Mock::start(|_| Reply::Json(200, completion("hi"))).await;
    let runtime = runtime_for(&mock, false);
    let listed = runtime.generation_providers();
    assert_eq!(listed.len(), 1);
    assert_eq!(
        (listed[0].name.as_str(), listed[0].local, listed[0].remote),
        ("openai-compatible", true, false)
    );

    let response = runtime
        .chat(
            "openai-compatible",
            ChatRequest::new("m", vec![ChatMessage::user("x")]),
            None,
        )
        .await
        .unwrap();
    assert_eq!(response.message.content, "hi");
    assert_eq!(response.metadata.provider, "openai-compatible");
    assert_eq!(response.metadata.transport, "http");
    assert_eq!(response.metadata.execution, ExecutionLocation::Local);
    assert!(
        response.structured.is_none(),
        "unrequested structure must not be reported"
    );
}

#[tokio::test]
async fn provider_is_explicit_and_there_is_no_fallback() {
    let mock = Mock::start(|_| Reply::Json(200, completion("hi"))).await;
    let runtime = runtime_for(&mock, false);
    let request = ChatRequest::new("m", vec![ChatMessage::user("x")]);
    for name in ["local", "remote", "OpenAI-Compatible", ""] {
        let error = runtime.chat(name, request.clone(), None).await.unwrap_err();
        assert!(
            matches!(error, RuntimeError::ProviderUnavailable { .. }),
            "{name}: {error:?}"
        );
        assert!(matches!(
            runtime.chat_stream(name, request.clone(), None).await,
            Err(RuntimeError::ProviderUnavailable { .. })
        ));
    }
    assert!(mock.captured().is_empty());
    // A runtime without generation providers cannot chat at all.
    let error = Runtime::new()
        .chat("openai-compatible", request, None)
        .await
        .unwrap_err();
    assert!(matches!(error, RuntimeError::ProviderUnavailable { .. }));
}

#[tokio::test]
async fn structured_output_outcomes_are_distinguished() {
    let cases = [
        (r#"{"n": 3}"#, "validated"),
        ("not json {", "invalid_json"),
        (r#"{"n": "three"}"#, "schema_violation"),
        (r#"{"n": 3, "extra": true}"#, "schema_violation"),
        ("", "invalid_json"),
    ];
    for (text, expected) in cases {
        let mock = Mock::start(move |_| Reply::Json(200, completion(text))).await;
        let response = runtime_for(&mock, true)
            .chat("openai-compatible", schema_request(int_schema()), None)
            .await
            .unwrap();
        // The model text is returned verbatim, never repaired.
        assert_eq!(response.message.content, text);
        let structured = response
            .structured
            .expect("requested structure must be reported");
        let status = serde_json::to_value(&structured).unwrap()["status"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(status, expected, "{text}");
        if let StructuredOutput::Validated { value } = structured {
            assert_eq!(value, json!({"n": 3}));
        }
    }
}

#[tokio::test]
async fn json_object_requires_an_object() {
    for (text, ok) in [(r#"{"a":1}"#, true), ("[1]", false), ("3", false)] {
        let mock = Mock::start(move |_| Reply::Json(200, completion(text))).await;
        let request =
            ChatRequest::new("m", vec![ChatMessage::user("x")]).with_options(GenerationOptions {
                response_format: ResponseFormat::JsonObject,
                ..Default::default()
            });
        let response = runtime_for(&mock, true)
            .chat("openai-compatible", request, None)
            .await
            .unwrap();
        assert_eq!(response.structured.unwrap().is_validated(), ok, "{text}");
    }
}

#[tokio::test]
async fn structured_output_failure_modes_are_errors_not_outputs() {
    // Unsupported by the provider's declared capabilities.
    let mock = Mock::start(|_| Reply::Json(200, completion("{}"))).await;
    let error = runtime_for(&mock, false)
        .chat("openai-compatible", schema_request(int_schema()), None)
        .await
        .unwrap_err();
    assert!(
        matches!(error, RuntimeError::CapabilityMismatch { .. }),
        "{error:?}"
    );
    assert!(mock.captured().is_empty());

    // An uncompilable schema is rejected before any request.
    let error = runtime_for(&mock, true)
        .chat(
            "openai-compatible",
            schema_request(json!({"type": 12})),
            None,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, RuntimeError::InvalidRequest { .. }),
        "{error:?}"
    );
    assert!(mock.captured().is_empty());

    // Rejection by the provider itself.
    let mock = Mock::start(|_| {
        Reply::Json(
            400,
            json!({"error":{"message":"response_format unsupported"}}).to_string(),
        )
    })
    .await;
    let error = runtime_for(&mock, true)
        .chat("openai-compatible", schema_request(int_schema()), None)
        .await
        .unwrap_err();
    assert!(
        matches!(error, RuntimeError::InvalidRequest { ref reason } if reason.contains("provider rejected")),
        "{error:?}"
    );
}

#[tokio::test]
async fn streaming_validates_structured_output_in_the_summary() {
    for (deltas, validated) in [
        (vec![r#"{"n""#, ": 5}"], true),
        (vec![r#"{"n": "#, "x}"], false),
    ] {
        let chunks = sse_chunks(&deltas);
        let mock = Mock::start(move |_| Reply::Sse {
            chunks: chunks.clone(),
            pause: Duration::ZERO,
            hold_open: false,
        })
        .await;
        let mut stream = runtime_for(&mock, true)
            .chat_stream("openai-compatible", schema_request(int_schema()), None)
            .await
            .unwrap();
        let mut summary = None;
        while let Some(event) = stream.next().await {
            if let ChatStreamEvent::Completed(s) = event.unwrap() {
                summary = Some(s);
            }
        }
        assert_eq!(
            summary.unwrap().structured.unwrap().is_validated(),
            validated
        );
    }
}

#[tokio::test]
async fn text_streams_carry_no_structured_verdict() {
    let chunks = sse_chunks(&["a", "b"]);
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: chunks.clone(),
        pause: Duration::ZERO,
        hold_open: false,
    })
    .await;
    let mut stream = runtime_for(&mock, false)
        .chat_stream(
            "openai-compatible",
            ChatRequest::new("m", vec![ChatMessage::user("x")]),
            None,
        )
        .await
        .unwrap();
    let mut last = None;
    while let Some(event) = stream.next().await {
        last = Some(event.unwrap());
    }
    let Some(ChatStreamEvent::Completed(summary)) = last else {
        panic!()
    };
    assert!(summary.structured.is_none());
}

#[tokio::test]
async fn concurrency_admission_is_shared_and_released_on_drop() {
    let chunks = sse_chunks(&["a"]);
    let first = chunks[0].clone();
    let mock = Mock::start(move |req| {
        if req.json()["stream"] == true {
            Reply::Sse {
                chunks: vec![first.clone()],
                pause: Duration::ZERO,
                hold_open: true,
            }
        } else {
            Reply::Json(200, completion("second"))
        }
    })
    .await;
    let runtime = Runtime::builder()
        .max_concurrent_inferences(1)
        .register_generation_provider(
            OpenAiCompatibleProvider::new(OpenAiCompatibleConfig::new(format!("{}/v1", mock.url)))
                .unwrap(),
        )
        .build();
    let request = ChatRequest::new("m", vec![ChatMessage::user("x")]);
    let mut stream = runtime
        .chat_stream("openai-compatible", request.clone(), None)
        .await
        .unwrap();
    assert!(matches!(
        stream.next().await,
        Some(Ok(ChatStreamEvent::Started { .. }))
    ));

    let token = CancellationToken::new();
    let canceller = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        canceller.cancel();
    });
    let blocked = runtime
        .chat("openai-compatible", request.clone(), Some(token))
        .await
        .unwrap_err();
    assert_eq!(blocked, RuntimeError::Cancelled);

    drop(stream);
    let response = tokio::time::timeout(
        Duration::from_secs(5),
        runtime.chat("openai-compatible", request, None),
    )
    .await
    .expect("permit must be released when the stream is dropped")
    .unwrap();
    assert_eq!(response.message.content, "second");
}

#[tokio::test]
async fn stream_cancellation_surfaces_cancelled_and_never_completes() {
    let first = sse_chunks(&["a"])[0].clone();
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![first.clone()],
        pause: Duration::ZERO,
        hold_open: true,
    })
    .await;
    let token = CancellationToken::new();
    let mut stream = runtime_for(&mock, false)
        .chat_stream(
            "openai-compatible",
            ChatRequest::new("m", vec![ChatMessage::user("x")]),
            Some(token.clone()),
        )
        .await
        .unwrap();
    let mut events = vec![stream.next().await.unwrap()];
    token.cancel();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    assert!(
        matches!(events.last(), Some(Err(RuntimeError::Cancelled))),
        "{events:?}"
    );
    assert!(!events
        .iter()
        .any(|e| matches!(e, Ok(ChatStreamEvent::Completed(_)))));
    assert!(mock.wait_disconnected().await);
}

/// A provider whose stream ends silently, to pin the runtime's termination contract.
struct SilentProvider;

#[async_trait]
impl GenerationProvider for SilentProvider {
    fn name(&self) -> &str {
        "silent"
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }
    fn generation_capabilities(&self) -> GenerationCapabilities {
        GenerationCapabilities {
            streaming: true,
            stop_sequences: true,
            seed: true,
            json_object: true,
            json_schema: true,
            tools: true,
            tool_choice: true,
        }
    }
    async fn list_models(
        &self,
        _: Option<CancellationToken>,
    ) -> RuntimeResult<Vec<GenerationModelInfo>> {
        Ok(vec![])
    }
    async fn chat(
        &self,
        _: ChatRequest,
        _: Option<CancellationToken>,
    ) -> RuntimeResult<ChatResponse> {
        Err(RuntimeError::Cancelled)
    }
    async fn chat_stream(
        &self,
        request: ChatRequest,
        _: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<ChatStreamEvent>>> {
        let started = ChatStreamEvent::Started {
            provider: "silent".into(),
            model: request.model,
        };
        Ok(Box::pin(futures_util::stream::iter(vec![Ok(started)])))
    }
}

#[tokio::test]
async fn a_stream_that_ends_without_completion_is_an_error() {
    let runtime = Runtime::builder()
        .register_generation_provider(SilentProvider)
        .build();
    let request = ChatRequest::new("m", vec![ChatMessage::user("x")]);
    let events: Vec<_> = runtime
        .chat_stream("silent", request.clone(), None)
        .await
        .unwrap()
        .collect()
        .await;
    assert!(
        matches!(events.last(), Some(Err(RuntimeError::Protocol { .. }))),
        "{events:?}"
    );

    let token = CancellationToken::new();
    token.cancel();
    // Admission observes the already-cancelled token first.
    assert!(matches!(
        runtime.chat_stream("silent", request, Some(token)).await,
        Err(RuntimeError::Cancelled)
    ));
}

#[test]
fn contract_types_round_trip_through_json() {
    let request = ChatRequest::new("m", vec![ChatMessage::system("s"), ChatMessage::user("u")])
        .with_options(GenerationOptions {
            temperature: Some(0.25),
            stop: vec!["x".into()],
            response_format: ResponseFormat::JsonSchema {
                name: "n".into(),
                schema: json!({"type":"object"}),
                strict: true,
            },
            ..Default::default()
        });
    let text = serde_json::to_string(&request).unwrap();
    assert_eq!(serde_json::from_str::<ChatRequest>(&text).unwrap(), request);
    // Minimal wire form from other languages deserializes with defaults.
    let minimal: ChatRequest =
        serde_json::from_value(json!({"model":"m","messages":[{"role":"user","content":"u"}]}))
            .unwrap();
    assert_eq!(minimal.options, GenerationOptions::default());
    // Message content never appears in Debug output.
    assert!(!format!("{:?}", ChatMessage::user("topsecret")).contains("topsecret"));
}

#[tokio::test]
async fn typed_inference_is_unaffected_by_generation_providers() {
    use rust_ml_runtime::{InferenceRequest, ModelFormat, ModelLocation, ModelSpec};
    let mock = Mock::start(|_| Reply::Json(200, completion("hi"))).await;
    let runtime = runtime_for(&mock, false);
    // Generation providers are a separate registry: they are not typed-inference
    // providers and never serve (or fall back for) typed requests.
    assert!(runtime
        .providers()
        .list()
        .iter()
        .all(|p| p.name != "openai-compatible"));
    let spec = ModelSpec::new("compat", ModelFormat::Unknown, ModelLocation::Memory);
    let result = runtime
        .infer(InferenceRequest::new(spec, "text"))
        .await
        .unwrap();
    assert_eq!(result.metadata().provider, "local");
    assert!(
        mock.captured().is_empty(),
        "typed inference must not reach the generation endpoint"
    );
}
