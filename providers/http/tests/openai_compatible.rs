mod support;

use futures_util::StreamExt;
use ml_runtime_common::{CancellationToken, RuntimeError};
use ml_runtime_http_provider::openai::{
    ApiKey, OpenAiCompatibleConfig, OpenAiCompatibleProvider, PROVIDER_NAME,
};
use ml_runtime_provider::{
    ChatMessage, ChatRequest, ChatStreamEvent, ExecutionLocation, FinishReason, GenerationOptions,
    GenerationProvider, ResponseFormat,
};
use serde_json::json;
use std::time::Duration;
use support::{Mock, Reply};

const SECRET: &str = "sk-test-SECRET-123";

fn completion(content: &str) -> String {
    json!({
        "id": "chatcmpl-1", "model": "reported-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    })
    .to_string()
}

fn request() -> ChatRequest {
    ChatRequest::new(
        "m1",
        vec![
            ChatMessage::system("be brief"),
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
            ChatMessage::user("again"),
        ],
    )
}

fn provider(mock: &Mock) -> OpenAiCompatibleProvider {
    OpenAiCompatibleProvider::new(OpenAiCompatibleConfig::new(format!("{}/v1", mock.url))).unwrap()
}

fn sse(events: &[&str]) -> Vec<u8> {
    events
        .iter()
        .map(|e| format!("data: {e}\n\n"))
        .collect::<String>()
        .into_bytes()
}

fn delta(content: &str) -> String {
    json!({"id":"c1","model":"reported-model","choices":[{"index":0,"delta":{"content":content},"finish_reason":null}]}).to_string()
}

fn finish() -> String {
    json!({"id":"c1","model":"reported-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}).to_string()
}

fn usage() -> String {
    json!({"id":"c1","model":"reported-model","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}).to_string()
}

fn full_stream() -> Vec<u8> {
    sse(&[
        &delta("Hel"),
        &delta("lo é😀"),
        &finish(),
        &usage(),
        "[DONE]",
    ])
}

async fn collect(
    mut stream: ml_runtime_common::BoxStream<ml_runtime_common::RuntimeResult<ChatStreamEvent>>,
) -> Vec<Result<ChatStreamEvent, RuntimeError>> {
    let mut out = Vec::new();
    while let Some(item) = stream.next().await {
        out.push(item);
    }
    out
}

#[tokio::test]
async fn chat_serializes_ordered_messages_and_options_and_reports_metadata() {
    let mock = Mock::start(|_| Reply::Json(200, completion("Hello"))).await;
    let mut req = request();
    req.options = GenerationOptions {
        temperature: Some(0.5),
        top_p: Some(0.9),
        max_tokens: Some(16),
        stop: vec!["END".into()],
        seed: Some(7),
        response_format: ResponseFormat::Text,
        ..Default::default()
    };
    let response = provider(&mock).chat(req, None).await.unwrap();

    let captured = mock.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].method, "POST");
    assert_eq!(captured[0].path, "/v1/chat/completions");
    let body = captured[0].json();
    let roles: Vec<_> = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| (m["role"].as_str().unwrap(), m["content"].as_str().unwrap()))
        .collect();
    assert_eq!(
        roles,
        [
            ("system", "be brief"),
            ("user", "hi"),
            ("assistant", "hello"),
            ("user", "again")
        ]
    );
    assert_eq!(body["model"], "m1");
    assert_eq!(body["stream"], false);
    assert_eq!(body["max_tokens"], 16);
    assert_eq!(body["stop"], json!(["END"]));
    assert_eq!(body["seed"], 7);
    assert!((body["temperature"].as_f64().unwrap() - 0.5).abs() < 1e-6);
    assert!(body.get("response_format").is_none());
    assert!(body.get("stream_options").is_none());

    assert_eq!(response.message.content, "Hello");
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    let usage = response.usage.unwrap();
    assert_eq!(
        (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens
        ),
        (Some(3), Some(2), Some(5))
    );
    let m = response.metadata;
    assert_eq!(m.provider, PROVIDER_NAME);
    assert_eq!(m.transport, "http");
    assert_eq!(m.execution, ExecutionLocation::Local);
    assert!(m.execution_inferred);
    assert_eq!(m.requested_model, "m1");
    assert_eq!(m.reported_model.as_deref(), Some("reported-model"));
    assert_eq!(m.response_id.as_deref(), Some("chatcmpl-1"));
    assert!(!m.streamed);
    assert_eq!(m.endpoint, format!("{}/v1", mock.url));
    assert!(response.structured.is_none());
}

#[tokio::test]
async fn absent_usage_and_finish_reason_are_not_fabricated() {
    let body = json!({"choices":[{"message":{"role":"assistant","content":"x"}}]}).to_string();
    let mock = Mock::start(move |_| Reply::Json(200, body.clone())).await;
    let response = provider(&mock).chat(request(), None).await.unwrap();
    assert!(response.usage.is_none());
    assert!(response.finish_reason.is_none());
    assert!(response.metadata.reported_model.is_none());
    assert_eq!(response.metadata.requested_model, "m1");
}

#[tokio::test]
async fn unknown_finish_reason_is_preserved() {
    let body = json!({"choices":[{"message":{"content":"x"},"finish_reason":"vendor_specific"}]})
        .to_string();
    let mock = Mock::start(move |_| Reply::Json(200, body.clone())).await;
    let response = provider(&mock).chat(request(), None).await.unwrap();
    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Other("vendor_specific".into()))
    );
}

#[tokio::test]
async fn bearer_header_is_sent_and_never_leaks() {
    let mock =
        Mock::start(|_| Reply::Json(401, json!({"error":{"message":"bad key"}}).to_string())).await;
    let config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url))
        .with_api_key(ApiKey::new(SECRET).unwrap());
    let provider = OpenAiCompatibleProvider::new(config).unwrap();
    let error = provider.chat(request(), None).await.unwrap_err();
    assert_eq!(
        mock.captured()[0].header("authorization"),
        Some(format!("Bearer {SECRET}").as_str())
    );
    for text in [
        format!("{error}"),
        format!("{error:?}"),
        format!("{provider:?}"),
    ] {
        assert!(!text.contains(SECRET), "leaked: {text}");
    }
    assert!(matches!(error, RuntimeError::ProviderUnavailable { .. }));
    assert!(format!("{:?}", ApiKey::new(SECRET).unwrap()).contains("redacted"));
}

#[tokio::test]
async fn no_authorization_header_without_a_key() {
    let mock = Mock::start(|_| Reply::Json(200, completion("x"))).await;
    provider(&mock).chat(request(), None).await.unwrap();
    assert!(mock.captured()[0].header("authorization").is_none());
}

#[test]
fn endpoint_validation() {
    let make = |url: &str, with_key: bool| {
        let mut config = OpenAiCompatibleConfig::new(url);
        if with_key {
            config = config.with_api_key(key());
        }
        OpenAiCompatibleProvider::new(config)
    };
    assert!(make("not a url", false).is_err());
    assert!(make("ftp://host/v1", false).is_err());
    assert!(make("http://user:pw@host/v1", false).is_err());
    assert!(make("https://host/v1?x=1", false).is_err());
    // A key over cleartext to a non-loopback host is refused.
    assert!(make("http://example.com/v1", true).is_err());
    assert!(make("https://example.com/v1", true).is_ok());
    assert!(make("http://127.0.0.1:11434/v1", true).is_ok());
    assert!(make("http://[::1]:11434/v1", true).is_ok());
    assert!(ApiKey::new("").is_err());
    assert!(ApiKey::new("a b").is_err());
}

fn key() -> ApiKey {
    ApiKey::new(SECRET).unwrap()
}

#[test]
fn execution_location_is_reported_and_overridable() {
    let make = |url: &str, location| {
        let mut config = OpenAiCompatibleConfig::new(url);
        config.location = location;
        OpenAiCompatibleProvider::new(config).unwrap()
    };
    assert_eq!(
        make("http://localhost:11434/v1", None).location(),
        ExecutionLocation::Local
    );
    assert_eq!(
        make("https://api.example.com/v1", None).location(),
        ExecutionLocation::Remote
    );
    // e.g. a loopback SSH tunnel to a remote host
    let tunnelled = make("http://127.0.0.1:9/v1", Some(ExecutionLocation::Remote));
    assert_eq!(tunnelled.location(), ExecutionLocation::Remote);
    let caps = tunnelled.capabilities();
    assert!(caps.remote && !caps.local);
    assert_eq!(caps.endpoint.as_deref(), Some("http://127.0.0.1:9/v1"));
}

#[tokio::test]
async fn provider_errors_are_distinguished() {
    type Case = (u16, serde_json::Value, fn(&RuntimeError) -> bool);
    let cases: Vec<Case> = vec![
        (400, json!({"error":{"message":"bad param"}}), |e| {
            matches!(e, RuntimeError::InvalidRequest { .. })
        }),
        (
            404,
            json!({"error":{"message":"model 'm1' not found","code":"model_not_found"}}),
            |e| matches!(e, RuntimeError::ModelNotFound { .. }),
        ),
        (429, json!({"error":{"message":"slow down"}}), |e| {
            matches!(e, RuntimeError::ProviderUnavailable { .. })
        }),
        (503, json!({"error":"overloaded"}), |e| {
            matches!(e, RuntimeError::ProviderUnavailable { .. })
        }),
        (418, json!({"error":{"message":"teapot"}}), |e| {
            matches!(e, RuntimeError::Protocol { .. })
        }),
    ];
    for (status, body, check) in cases {
        let body = body.to_string();
        let mock = Mock::start(move |_| Reply::Json(status, body.clone())).await;
        let error = provider(&mock).chat(request(), None).await.unwrap_err();
        assert!(check(&error), "status {status}: {error:?}");
        assert!(error.to_string().contains(&status.to_string()) || status == 404);
    }
}

#[tokio::test]
async fn malformed_payloads_are_protocol_errors() {
    for body in [
        "not json",
        "{}",
        r#"{"choices":[]}"#,
        r#"{"choices":[{"finish_reason":"stop"}]}"#,
        r#"{"choices":[{"message":{"role":"user","content":"x"}}]}"#,
    ] {
        let mock = Mock::start(move |_| Reply::Json(200, body.to_owned())).await;
        let error = provider(&mock).chat(request(), None).await.unwrap_err();
        assert!(
            matches!(error, RuntimeError::Protocol { .. }),
            "{body}: {error:?}"
        );
    }
}

#[tokio::test]
async fn transport_failure_is_distinct_from_provider_error() {
    let provider =
        OpenAiCompatibleProvider::new(OpenAiCompatibleConfig::new("http://127.0.0.1:1/v1"))
            .unwrap();
    let error = provider.chat(request(), None).await.unwrap_err();
    assert!(matches!(error, RuntimeError::Transport { .. }), "{error:?}");
}

#[tokio::test]
async fn request_timeout_and_cancellation() {
    let mock = Mock::start(|_| Reply::Stall).await;
    let mut config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url));
    config.request_timeout = Duration::from_millis(100);
    let slow = OpenAiCompatibleProvider::new(config).unwrap();
    let error = slow.chat(request(), None).await.unwrap_err();
    assert!(matches!(error, RuntimeError::Timeout { .. }), "{error:?}");
    assert!(
        mock.wait_disconnected().await,
        "timed-out request must close the connection"
    );

    let mock = Mock::start(|_| Reply::Stall).await;
    let token = CancellationToken::new();
    let canceller = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        canceller.cancel();
    });
    let error = provider(&mock)
        .chat(request(), Some(token))
        .await
        .unwrap_err();
    assert_eq!(error, RuntimeError::Cancelled);
    assert!(mock.wait_disconnected().await);
}

#[tokio::test]
async fn oversized_responses_are_bounded() {
    let mock = Mock::start(|_| Reply::Json(200, completion(&"x".repeat(2048)))).await;
    let mut config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url));
    config.max_output_bytes = 100;
    let error = OpenAiCompatibleProvider::new(config)
        .unwrap()
        .chat(request(), None)
        .await
        .unwrap_err();
    assert!(
        matches!(error, RuntimeError::ResourceLimit { .. }),
        "{error:?}"
    );

    let mut config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url));
    config.max_response_bytes = 100;
    let error = OpenAiCompatibleProvider::new(config)
        .unwrap()
        .chat(request(), None)
        .await
        .unwrap_err();
    assert!(
        matches!(error, RuntimeError::ResourceLimit { .. }),
        "{error:?}"
    );
}

#[tokio::test]
async fn unsupported_capabilities_are_rejected_before_any_request() {
    let mock = Mock::start(|_| Reply::Json(200, completion("x"))).await;
    let p = provider(&mock);
    let with = |format: ResponseFormat| {
        request().with_options(GenerationOptions {
            response_format: format,
            ..Default::default()
        })
    };
    for req in [
        with(ResponseFormat::JsonObject),
        with(ResponseFormat::JsonSchema {
            name: "s".into(),
            schema: json!({}),
            strict: false,
        }),
    ] {
        let error = p.chat(req, None).await.unwrap_err();
        assert!(
            matches!(error, RuntimeError::CapabilityMismatch { .. }),
            "{error:?}"
        );
    }
    let mut config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url));
    config.capabilities.streaming = false;
    config.capabilities.stop_sequences = false;
    config.capabilities.seed = false;
    let p = OpenAiCompatibleProvider::new(config).unwrap();
    assert!(matches!(
        p.chat_stream(request(), None).await,
        Err(RuntimeError::CapabilityMismatch { .. })
    ));
    for options in [
        GenerationOptions {
            stop: vec!["x".into()],
            ..Default::default()
        },
        GenerationOptions {
            seed: Some(1),
            ..Default::default()
        },
    ] {
        let error = p
            .chat(request().with_options(options), None)
            .await
            .unwrap_err();
        assert!(matches!(error, RuntimeError::CapabilityMismatch { .. }));
    }
    assert!(
        mock.captured().is_empty(),
        "rejections must not reach the network"
    );
}

#[tokio::test]
async fn declared_structured_output_is_sent_in_native_form() {
    let mock = Mock::start(|_| Reply::Json(200, completion("{}"))).await;
    let mut config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url));
    config.capabilities.json_schema = true;
    config.capabilities.json_object = true;
    let p = OpenAiCompatibleProvider::new(config).unwrap();
    let schema = json!({"type":"object"});
    let options = GenerationOptions {
        response_format: ResponseFormat::JsonSchema {
            name: "answer".into(),
            schema: schema.clone(),
            strict: true,
        },
        ..Default::default()
    };
    p.chat(request().with_options(options), None).await.unwrap();
    p.chat(
        request().with_options(GenerationOptions {
            response_format: ResponseFormat::JsonObject,
            ..Default::default()
        }),
        None,
    )
    .await
    .unwrap();
    let captured = mock.captured();
    assert_eq!(
        captured[0].json()["response_format"],
        json!({"type":"json_schema","json_schema":{"name":"answer","schema":schema,"strict":true}})
    );
    assert_eq!(
        captured[1].json()["response_format"],
        json!({"type":"json_object"})
    );
}

#[tokio::test]
async fn invalid_requests_are_rejected_locally() {
    let mock = Mock::start(|_| Reply::Json(200, completion("x"))).await;
    let p = provider(&mock);
    for req in [
        ChatRequest::new("", vec![ChatMessage::user("x")]),
        ChatRequest::new("m", vec![]),
        ChatRequest::new("m", vec![ChatMessage::system("x")]),
        request().with_options(GenerationOptions {
            max_tokens: Some(0),
            ..Default::default()
        }),
        request().with_options(GenerationOptions {
            top_p: Some(2.0),
            ..Default::default()
        }),
        request().with_options(GenerationOptions {
            stop: vec![String::new()],
            ..Default::default()
        }),
    ] {
        assert!(matches!(
            p.chat(req, None).await,
            Err(RuntimeError::InvalidRequest { .. })
        ));
    }
    assert!(mock.captured().is_empty());
}

#[tokio::test]
async fn list_models_and_unsupported_listing() {
    let body = json!({"object":"list","data":[{"id":"a","owned_by":"x"},{"id":"b"}]}).to_string();
    let mock = Mock::start(move |_| Reply::Json(200, body.clone())).await;
    let models = provider(&mock).list_models(None).await.unwrap();
    assert_eq!(
        models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(mock.captured()[0].path, "/v1/models");
    assert_eq!(mock.captured()[0].method, "GET");

    let mock = Mock::start(|_| Reply::Json(404, "{}".into())).await;
    let error = provider(&mock).list_models(None).await.unwrap_err();
    assert!(matches!(error, RuntimeError::CapabilityMismatch { .. }));

    let mock = Mock::start(|_| Reply::Json(200, "{\"nope\":1}".into())).await;
    assert!(matches!(
        provider(&mock).list_models(None).await,
        Err(RuntimeError::Protocol { .. })
    ));
}

#[tokio::test]
async fn redirects_are_not_followed() {
    let mock = Mock::start(|_| Reply::Json(307, "{}".into())).await;
    let error = provider(&mock).chat(request(), None).await.unwrap_err();
    assert!(matches!(error, RuntimeError::Protocol { .. }), "{error:?}");
}

// ---- streaming ---------------------------------------------------------------

#[tokio::test]
async fn stream_completes_for_every_chunk_boundary() {
    let bytes = full_stream();
    for step in [1usize, 2, 3, 7, 11, 64, bytes.len()] {
        let chunks: Vec<Vec<u8>> = bytes.chunks(step).map(<[u8]>::to_vec).collect();
        let mock = Mock::start(move |_| Reply::Sse {
            chunks: chunks.clone(),
            pause: Duration::ZERO,
            hold_open: false,
        })
        .await;
        let stream = provider(&mock).chat_stream(request(), None).await.unwrap();
        let events = collect(stream).await;
        let events: Vec<_> = events.into_iter().map(Result::unwrap).collect();
        assert!(
            matches!(&events[0], ChatStreamEvent::Started { provider, model } if provider == PROVIDER_NAME && model == "m1"),
            "step {step}"
        );
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                ChatStreamEvent::Delta { content } => Some(content.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "Hello é😀", "step {step}");
        let ChatStreamEvent::Completed(summary) = events.last().unwrap() else {
            panic!("step {step}")
        };
        assert_eq!(summary.finish_reason, Some(FinishReason::Stop));
        assert_eq!(summary.usage.as_ref().unwrap().total_tokens, Some(6));
        assert!(summary.metadata.streamed);
        assert_eq!(summary.metadata.stream_events, Some(2));
        assert!(summary.metadata.time_to_first_token.is_some());
        assert_eq!(
            summary.metadata.reported_model.as_deref(),
            Some("reported-model")
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, ChatStreamEvent::Completed(_)))
                .count(),
            1
        );
    }
}

#[tokio::test]
async fn stream_request_asks_for_usage_and_sets_stream_flag() {
    let bytes = full_stream();
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![bytes.clone()],
        pause: Duration::ZERO,
        hold_open: false,
    })
    .await;
    collect(provider(&mock).chat_stream(request(), None).await.unwrap()).await;
    let body = mock.captured()[0].json();
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
}

#[tokio::test]
async fn stream_without_done_but_with_finish_reason_completes() {
    let bytes = sse(&[&delta("a"), &finish()]);
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![bytes.clone()],
        pause: Duration::ZERO,
        hold_open: false,
    })
    .await;
    let events = collect(provider(&mock).chat_stream(request(), None).await.unwrap()).await;
    let Some(Ok(ChatStreamEvent::Completed(summary))) = events.last() else {
        panic!("{events:?}")
    };
    assert!(summary.usage.is_none());
}

#[tokio::test]
async fn truncated_stream_is_an_error_not_a_completion() {
    let bytes = sse(&[&delta("a"), &delta("b")]);
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![bytes.clone()],
        pause: Duration::ZERO,
        hold_open: false,
    })
    .await;
    let events = collect(provider(&mock).chat_stream(request(), None).await.unwrap()).await;
    assert!(
        matches!(events.last(), Some(Err(RuntimeError::Protocol { .. }))),
        "{events:?}"
    );
    assert!(!events
        .iter()
        .any(|e| matches!(e, Ok(ChatStreamEvent::Completed(_)))));
}

#[tokio::test]
async fn mid_stream_provider_error_and_malformed_event() {
    let error_event = json!({"error":{"message":"out of memory"}}).to_string();
    let bytes = sse(&[&delta("a"), &error_event]);
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![bytes.clone()],
        pause: Duration::ZERO,
        hold_open: false,
    })
    .await;
    let events = collect(provider(&mock).chat_stream(request(), None).await.unwrap()).await;
    assert!(
        matches!(
            events.last(),
            Some(Err(RuntimeError::ProviderUnavailable { .. }))
        ),
        "{events:?}"
    );

    let bytes = sse(&[&delta("a"), "{not json"]);
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![bytes.clone()],
        pause: Duration::ZERO,
        hold_open: false,
    })
    .await;
    let events = collect(provider(&mock).chat_stream(request(), None).await.unwrap()).await;
    assert!(
        matches!(events.last(), Some(Err(RuntimeError::Protocol { .. }))),
        "{events:?}"
    );
}

#[tokio::test]
async fn stream_http_error_is_reported_before_the_stream_starts() {
    let mock =
        Mock::start(|_| Reply::Json(400, json!({"error":{"message":"nope"}}).to_string())).await;
    let result = provider(&mock).chat_stream(request(), None).await;
    assert!(matches!(result, Err(RuntimeError::InvalidRequest { .. })));
    let mock = Mock::start(|_| Reply::Json(200, completion("x"))).await;
    let result = provider(&mock).chat_stream(request(), None).await;
    assert!(matches!(result, Err(RuntimeError::Protocol { .. })));
}

#[tokio::test]
async fn dropping_the_stream_closes_the_connection() {
    let bytes = sse(&[&delta("a")]);
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![bytes.clone()],
        pause: Duration::ZERO,
        hold_open: true,
    })
    .await;
    let mut stream = provider(&mock).chat_stream(request(), None).await.unwrap();
    assert!(matches!(
        stream.next().await,
        Some(Ok(ChatStreamEvent::Started { .. }))
    ));
    assert!(matches!(
        stream.next().await,
        Some(Ok(ChatStreamEvent::Delta { .. }))
    ));
    drop(stream);
    assert!(
        mock.wait_disconnected().await,
        "upstream connection must be released"
    );
}

#[tokio::test]
async fn cancellation_terminates_the_stream_with_cancelled() {
    let bytes = sse(&[&delta("a")]);
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![bytes.clone()],
        pause: Duration::ZERO,
        hold_open: true,
    })
    .await;
    let token = CancellationToken::new();
    let mut stream = provider(&mock)
        .chat_stream(request(), Some(token.clone()))
        .await
        .unwrap();
    assert!(matches!(
        stream.next().await,
        Some(Ok(ChatStreamEvent::Started { .. }))
    ));
    token.cancel();
    let mut last = None;
    while let Some(item) = stream.next().await {
        last = Some(item);
    }
    assert!(
        matches!(last, Some(Err(RuntimeError::Cancelled))),
        "{last:?}"
    );
    assert!(mock.wait_disconnected().await);
}

#[tokio::test]
async fn stream_idle_timeout() {
    let bytes = sse(&[&delta("a")]);
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![bytes.clone()],
        pause: Duration::ZERO,
        hold_open: true,
    })
    .await;
    let mut config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url));
    config.stream_idle_timeout = Duration::from_millis(100);
    let p = OpenAiCompatibleProvider::new(config).unwrap();
    let events = collect(p.chat_stream(request(), None).await.unwrap()).await;
    assert!(
        matches!(events.last(), Some(Err(RuntimeError::Timeout { .. }))),
        "{events:?}"
    );
    assert!(mock.wait_disconnected().await);
}

#[tokio::test]
async fn streamed_output_is_bounded() {
    let bytes = sse(&[
        &delta(&"x".repeat(64)),
        &delta(&"y".repeat(64)),
        &finish(),
        "[DONE]",
    ]);
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![bytes.clone()],
        pause: Duration::ZERO,
        hold_open: false,
    })
    .await;
    let mut config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url));
    config.max_output_bytes = 100;
    let p = OpenAiCompatibleProvider::new(config).unwrap();
    let events = collect(p.chat_stream(request(), None).await.unwrap()).await;
    assert!(
        matches!(events.last(), Some(Err(RuntimeError::ResourceLimit { .. }))),
        "{events:?}"
    );
}

/// Opt-in check against a real OpenAI-compatible server (for example Ollama).
/// Excluded from the default offline run:
///
/// ```text
/// ML_RUNTIME_TEST_OPENAI_ENDPOINT=http://127.0.0.1:11434/v1 \
/// ML_RUNTIME_TEST_OPENAI_MODEL=llama3.2 \
///   cargo test -p ml-runtime-http-provider --test openai_compatible -- --ignored
/// ```
///
/// `ML_RUNTIME_API_KEY` is used as the bearer token when set.
#[tokio::test]
#[ignore = "requires a running OpenAI-compatible endpoint"]
async fn live_endpoint_round_trip() {
    let endpoint =
        std::env::var("ML_RUNTIME_TEST_OPENAI_ENDPOINT").expect("ML_RUNTIME_TEST_OPENAI_ENDPOINT");
    let model =
        std::env::var("ML_RUNTIME_TEST_OPENAI_MODEL").expect("ML_RUNTIME_TEST_OPENAI_MODEL");
    let mut config = OpenAiCompatibleConfig::new(endpoint);
    if let Ok(key) = std::env::var("ML_RUNTIME_API_KEY") {
        config = config.with_api_key(ApiKey::new(key).unwrap());
    }
    let provider = OpenAiCompatibleProvider::new(config).unwrap();
    let request = ChatRequest::new(
        model,
        vec![ChatMessage::user("Reply with the single word: ok")],
    )
    .with_options(GenerationOptions {
        max_tokens: Some(16),
        temperature: Some(0.0),
        ..Default::default()
    });

    let response = provider.chat(request.clone(), None).await.unwrap();
    assert!(!response.message.content.is_empty());
    println!("chat: {:?} {:?}", response.finish_reason, response.usage);

    let events = collect(provider.chat_stream(request, None).await.unwrap()).await;
    assert!(
        matches!(events.last(), Some(Ok(ChatStreamEvent::Completed(_)))),
        "{events:?}"
    );
}
