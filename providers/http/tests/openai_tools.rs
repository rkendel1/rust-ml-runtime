//! Tool-call proposals over the OpenAI-compatible adapter (mock server only).
//! Nothing here executes a tool: the adapter only carries proposals and results.

mod support;

use futures_util::StreamExt;
use ml_runtime_common::{CancellationToken, RuntimeError};
use ml_runtime_http_provider::openai::{OpenAiCompatibleConfig, OpenAiCompatibleProvider};
use ml_runtime_provider::{
    ChatMessage, ChatRequest, ChatStreamEvent, FinishReason, GenerationOptions, GenerationProvider,
    ToolCall, ToolChoice, ToolDefinition,
};
use serde_json::{json, Value};
use std::time::Duration;
use support::{Mock, Reply};

fn weather_tool() -> ToolDefinition {
    ToolDefinition::new(
        "get_weather",
        json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}),
    )
    .with_description("Look up the weather")
}

fn options(choice: Option<ToolChoice>) -> GenerationOptions {
    GenerationOptions {
        tools: vec![weather_tool()],
        tool_choice: choice,
        ..Default::default()
    }
}

fn request() -> ChatRequest {
    ChatRequest::new("m", vec![ChatMessage::user("weather in Paris?")]).with_options(options(None))
}

fn provider(mock: &Mock) -> OpenAiCompatibleProvider {
    let mut config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url));
    config.capabilities.tools = true;
    config.capabilities.tool_choice = true;
    OpenAiCompatibleProvider::new(config).unwrap()
}

fn tool_response(calls: Value, content: Value, finish: &str) -> String {
    json!({"id":"c1","model":"r","choices":[{"message":{"role":"assistant","content":content,"tool_calls":calls},"finish_reason":finish}]}).to_string()
}

fn wire_call(id: &str, name: &str, args: &str) -> Value {
    json!({"id":id,"type":"function","function":{"name":name,"arguments":args}})
}

async fn run(body: String) -> Result<ml_runtime_provider::ChatResponse, RuntimeError> {
    let mock = Mock::start(move |_| Reply::Json(200, body.clone())).await;
    provider(&mock).chat(request(), None).await
}

// ---- request encoding ----------------------------------------------------------

#[tokio::test]
async fn definitions_and_choice_are_sent_in_the_wire_format() {
    let cases: [(Option<ToolChoice>, Option<Value>); 4] = [
        (None, None),
        (Some(ToolChoice::Auto), Some(json!("auto"))),
        (Some(ToolChoice::None), Some(json!("none"))),
        (
            Some(ToolChoice::Tool {
                name: "get_weather".into(),
            }),
            Some(json!({"type":"function","function":{"name":"get_weather"}})),
        ),
    ];
    for (choice, expected) in cases {
        let mock =
            Mock::start(|_| Reply::Json(200, tool_response(Value::Null, json!("hi"), "stop")))
                .await;
        let req = ChatRequest::new("m", vec![ChatMessage::user("x")]).with_options(options(choice));
        provider(&mock).chat(req, None).await.unwrap();
        let body = mock.captured()[0].json();
        assert_eq!(body.get("tool_choice").cloned(), expected);
        assert_eq!(
            body["tools"],
            json!([{"type":"function","function":{"name":"get_weather","description":"Look up the weather",
                "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}])
        );
    }
}

#[tokio::test]
async fn text_only_requests_carry_no_tool_fields() {
    let mock =
        Mock::start(|_| Reply::Json(200, tool_response(Value::Null, json!("hi"), "stop"))).await;
    provider(&mock)
        .chat(ChatRequest::new("m", vec![ChatMessage::user("x")]), None)
        .await
        .unwrap();
    let body = mock.captured()[0].json();
    assert!(body.get("tools").is_none() && body.get("tool_choice").is_none());
    assert_eq!(body["messages"][0], json!({"role":"user","content":"x"}));
}

#[tokio::test]
async fn assistant_calls_and_tool_results_are_replayed_in_order() {
    let mock = Mock::start(|_| {
        Reply::Json(
            200,
            tool_response(Value::Null, json!("It is sunny."), "stop"),
        )
    })
    .await;
    let call = ToolCall::new("call_1", "get_weather", json!({"city":"Paris"}));
    let req = ChatRequest::new(
        "m",
        vec![
            ChatMessage::user("weather in Paris?"),
            ChatMessage::assistant_with_tool_calls("", vec![call]),
            ChatMessage::tool_result("call_1", r#"{"temp_c":21}"#),
        ],
    )
    .with_options(options(None));
    let response = provider(&mock).chat(req, None).await.unwrap();
    assert_eq!(response.message.content, "It is sunny.");
    assert!(response.message.tool_calls.is_empty());
    let messages = mock.captured()[0].json()["messages"].clone();
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(
        messages[1]["content"],
        Value::Null,
        "call-only turns send null content"
    );
    assert_eq!(
        messages[1]["tool_calls"],
        json!([wire_call("call_1", "get_weather", r#"{"city":"Paris"}"#)])
    );
    assert_eq!(
        messages[2],
        json!({"role":"tool","content":r#"{"temp_c":21}"#,"tool_call_id":"call_1"})
    );
}

#[tokio::test]
async fn text_alongside_calls_is_replayed_with_both() {
    let mock =
        Mock::start(|_| Reply::Json(200, tool_response(Value::Null, json!("ok"), "stop"))).await;
    let call = ToolCall::new("c", "get_weather", json!({"city":"Rome"}));
    let req = ChatRequest::new(
        "m",
        vec![
            ChatMessage::user("x"),
            ChatMessage::assistant_with_tool_calls("Checking.", vec![call]),
            ChatMessage::tool_result("c", "sunny"),
        ],
    )
    .with_options(options(None));
    provider(&mock).chat(req, None).await.unwrap();
    assert_eq!(
        mock.captured()[0].json()["messages"][1]["content"],
        "Checking."
    );
}

// ---- unsupported features --------------------------------------------------------

#[tokio::test]
async fn undeclared_tool_support_is_rejected_before_any_request() {
    let mock = Mock::start(|_| Reply::Json(200, "{}".into())).await;
    let plain =
        OpenAiCompatibleProvider::new(OpenAiCompatibleConfig::new(format!("{}/v1", mock.url)))
            .unwrap();
    let error = plain.chat(request(), None).await.unwrap_err();
    assert!(
        matches!(error, RuntimeError::CapabilityMismatch { .. }),
        "{error:?}"
    );
    assert!(matches!(
        plain.chat_stream(request(), None).await,
        Err(RuntimeError::CapabilityMismatch { .. })
    ));

    // Replaying tool history also needs the capability.
    let history = ChatRequest::new(
        "m",
        vec![
            ChatMessage::user("x"),
            ChatMessage::assistant_with_tool_calls("", vec![ToolCall::new("a", "t", json!({}))]),
            ChatMessage::tool_result("a", "r"),
        ],
    );
    assert!(matches!(
        plain.chat(history, None).await,
        Err(RuntimeError::CapabilityMismatch { .. })
    ));

    // tools declared, tool_choice not: choice is not silently dropped.
    let mut config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url));
    config.capabilities.tools = true;
    let tools_only = OpenAiCompatibleProvider::new(config).unwrap();
    let req = ChatRequest::new("m", vec![ChatMessage::user("x")])
        .with_options(options(Some(ToolChoice::Auto)));
    let error = tools_only.chat(req, None).await.unwrap_err();
    assert!(
        matches!(&error, RuntimeError::CapabilityMismatch { reason } if reason.contains("tool_choice")),
        "{error:?}"
    );
    assert!(mock.captured().is_empty());
}

#[tokio::test]
async fn provider_rejection_is_distinct_from_unsupported_and_transport() {
    let mock = Mock::start(|_| {
        Reply::Json(
            400,
            json!({"error":{"message":"does not support tools"}}).to_string(),
        )
    })
    .await;
    let error = provider(&mock).chat(request(), None).await.unwrap_err();
    assert!(
        matches!(&error, RuntimeError::InvalidRequest { reason } if reason.contains("provider rejected")),
        "{error:?}"
    );
    let down = OpenAiCompatibleProvider::new({
        let mut c = OpenAiCompatibleConfig::new("http://127.0.0.1:1/v1");
        c.capabilities.tools = true;
        c
    })
    .unwrap();
    assert!(matches!(
        down.chat(request(), None).await,
        Err(RuntimeError::Transport { .. })
    ));
}

// ---- responses ---------------------------------------------------------------------

#[tokio::test]
async fn one_proposal_preserves_id_name_and_arguments() {
    let body = tool_response(
        json!([wire_call("call_9", "get_weather", r#"{"city":"Paris"}"#)]),
        Value::Null,
        "tool_calls",
    );
    let response = run(body).await.unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(response.message.content, "");
    assert_eq!(
        response.message.tool_calls,
        vec![ToolCall::new(
            "call_9",
            "get_weather",
            json!({"city":"Paris"})
        )]
    );
}

#[tokio::test]
async fn multiple_proposals_keep_order_and_text_may_accompany_them() {
    let body = tool_response(
        json!([
            wire_call("a", "get_weather", r#"{"city":"A"}"#),
            wire_call("b", "get_weather", r#"{"city":"B"}"#)
        ]),
        json!("Let me check both."),
        // Some servers report "stop" even when proposing calls; reported as-is.
        "stop",
    );
    let response = run(body).await.unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    assert_eq!(response.message.content, "Let me check both.");
    let ids: Vec<_> = response
        .message
        .tool_calls
        .iter()
        .map(|c| c.id.as_str())
        .collect();
    assert_eq!(ids, ["a", "b"]);
}

#[tokio::test]
async fn ordinary_answer_with_tools_available_has_no_proposals() {
    let response = run(tool_response(Value::Null, json!("No tool needed."), "stop"))
        .await
        .unwrap();
    assert!(response.message.tool_calls.is_empty());
    let response = run(tool_response(json!([]), json!("No tool needed."), "stop"))
        .await
        .unwrap();
    assert!(response.message.tool_calls.is_empty());
}

#[tokio::test]
async fn malformed_proposals_never_become_valid_ones() {
    let wire = |c: Value| tool_response(json!([c]), Value::Null, "tool_calls");
    // Wire-level malformation: Protocol.
    for body in [
        wire(json!({"function":{"name":"get_weather","arguments":"{}"}})), // no id
        wire(json!({"id":"a","function":{"arguments":"{}"}})),             // no name
        wire(json!({"id":"a"})),                                           // no function
        wire(json!({"id":"a","function":{"name":"get_weather","arguments":{"city":"x"}}})), // object, not string
        tool_response(
            json!([
                wire_call("d", "get_weather", "{}"),
                wire_call("d", "get_weather", "{}")
            ]),
            Value::Null,
            "tool_calls",
        ),
    ] {
        let error = run(body.clone()).await.unwrap_err();
        assert!(
            matches!(error, RuntimeError::Protocol { .. }),
            "{body}: {error:?}"
        );
    }
    // The model produced something unusable: InvalidOutput.
    for args in [
        "",
        "{",
        "not json",
        "[1]",
        "\"str\"",
        "null",
        r#"{"city":"x"} trailing"#,
    ] {
        let error = run(wire(wire_call("a", "get_weather", args)))
            .await
            .unwrap_err();
        assert!(
            matches!(error, RuntimeError::InvalidOutput { .. }),
            "{args:?}: {error:?}"
        );
        assert!(
            !error.to_string().contains("trailing"),
            "arguments must not be echoed"
        );
    }
    let error = run(wire(wire_call("a", "bad name!", "{}")))
        .await
        .unwrap_err();
    assert!(
        matches!(error, RuntimeError::InvalidOutput { .. }),
        "{error:?}"
    );
}

#[tokio::test]
async fn too_many_calls_is_a_resource_limit() {
    let calls: Vec<Value> = (0..40)
        .map(|i| wire_call(&format!("c{i}"), "get_weather", "{}"))
        .collect();
    let error = run(tool_response(json!(calls), Value::Null, "tool_calls"))
        .await
        .unwrap_err();
    assert!(
        matches!(error, RuntimeError::ResourceLimit { .. }),
        "{error:?}"
    );
}

#[tokio::test]
async fn existing_guarantees_still_hold_with_tools() {
    // Redirects are refused and credentials never leak into tool errors.
    let mock = Mock::start(|_| Reply::Json(307, "{}".into())).await;
    assert!(matches!(
        provider(&mock).chat(request(), None).await,
        Err(RuntimeError::Protocol { .. })
    ));
    let mock = Mock::start(|_| Reply::Stall).await;
    let mut config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url));
    config.capabilities.tools = true;
    config.request_timeout = Duration::from_millis(100);
    let slow = OpenAiCompatibleProvider::new(config).unwrap();
    assert!(matches!(
        slow.chat(request(), None).await,
        Err(RuntimeError::Timeout { .. })
    ));
}

// ---- streaming -----------------------------------------------------------------------

fn sse(events: &[Value]) -> Vec<u8> {
    let mut out = String::new();
    for event in events {
        out.push_str(&format!("data: {event}\n\n"));
    }
    out.push_str("data: [DONE]\n\n");
    out.into_bytes()
}

fn tc(index: u32, id: Option<&str>, name: Option<&str>, args: Option<&str>) -> Value {
    let mut function = serde_json::Map::new();
    if let Some(n) = name {
        function.insert("name".into(), json!(n));
    }
    if let Some(a) = args {
        function.insert("arguments".into(), json!(a));
    }
    let mut call = json!({"index": index, "function": function});
    if let Some(id) = id {
        call["id"] = json!(id);
    }
    call
}

fn chunk(delta: Value, finish: Option<&str>) -> Value {
    json!({"id":"s1","model":"r","choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
}

/// Two interleaved calls whose ids, names, and arguments are fragmented.
fn interleaved_stream() -> Vec<u8> {
    sse(&[
        chunk(json!({"role":"assistant","content":"Checking. "}), None),
        chunk(
            json!({"tool_calls":[tc(0, Some("call_"), Some("get_"), None)]}),
            None,
        ),
        chunk(
            json!({"tool_calls":[tc(1, Some("call_B"), Some("get_weather"), Some("{\"ci"))]}),
            None,
        ),
        chunk(
            json!({"tool_calls":[tc(0, Some("A"), Some("weather"), Some("{\"city\":"))]}),
            None,
        ),
        chunk(
            json!({"tool_calls":[tc(1, None, None, Some("ty\":\"Berlin\"}"))]}),
            None,
        ),
        chunk(
            json!({"tool_calls":[tc(0, None, None, Some("\"Oslo é😀\"}"))]}),
            None,
        ),
        chunk(json!({}), Some("tool_calls")),
    ])
}

async fn collect_stream(
    mock: &Mock,
    req: ChatRequest,
) -> Vec<Result<ChatStreamEvent, RuntimeError>> {
    let mut stream = provider(mock).chat_stream(req, None).await.unwrap();
    let mut out = Vec::new();
    while let Some(item) = stream.next().await {
        out.push(item);
    }
    out
}

#[tokio::test]
async fn streamed_calls_assemble_identically_at_every_byte_boundary() {
    let bytes = interleaved_stream();
    for step in [1usize, 2, 3, 5, 17, 101, bytes.len()] {
        let chunks: Vec<Vec<u8>> = bytes.chunks(step).map(<[u8]>::to_vec).collect();
        let mock = Mock::start(move |_| Reply::Sse {
            chunks: chunks.clone(),
            pause: Duration::ZERO,
            hold_open: false,
        })
        .await;
        let events: Vec<_> = collect_stream(&mock, request())
            .await
            .into_iter()
            .map(|e| e.unwrap())
            .collect();

        let completed: Vec<&ToolCall> = events
            .iter()
            .filter_map(|e| match e {
                ChatStreamEvent::ToolCallCompleted { call } => Some(call),
                _ => None,
            })
            .collect();
        assert_eq!(completed.len(), 2, "step {step}");
        assert_eq!(
            *completed[0],
            ToolCall::new("call_A", "get_weather", json!({"city":"Oslo é😀"})),
            "step {step}"
        );
        assert_eq!(
            *completed[1],
            ToolCall::new("call_B", "get_weather", json!({"city":"Berlin"})),
            "step {step}"
        );

        // Proposals only after every delta, immediately before Completed.
        let first_completed = events
            .iter()
            .position(|e| matches!(e, ChatStreamEvent::ToolCallCompleted { .. }))
            .unwrap();
        assert!(events[..first_completed]
            .iter()
            .all(|e| !matches!(e, ChatStreamEvent::ToolCallCompleted { .. })));
        assert!(
            events
                .iter()
                .rposition(|e| matches!(e, ChatStreamEvent::ToolCallDelta { .. }))
                .unwrap()
                < first_completed
        );
        assert!(
            matches!(events.last(), Some(ChatStreamEvent::Completed(s)) if s.finish_reason == Some(FinishReason::ToolCalls))
        );

        // Text, raw tool-call fragments, and proposals are distinct events.
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                ChatStreamEvent::Delta { content } => Some(content.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "Checking. ");
        let fragments: String = events
            .iter()
            .filter_map(|e| match e {
                ChatStreamEvent::ToolCallDelta {
                    index: 1,
                    arguments_fragment: Some(a),
                    ..
                } => Some(a.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(fragments, "{\"city\":\"Berlin\"}");
    }
}

#[tokio::test]
async fn streams_without_tool_calls_are_unchanged() {
    let bytes = sse(&[
        chunk(json!({"content":"Hi"}), None),
        chunk(json!({}), Some("stop")),
    ]);
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![bytes.clone()],
        pause: Duration::ZERO,
        hold_open: false,
    })
    .await;
    let events: Vec<_> = collect_stream(&mock, request())
        .await
        .into_iter()
        .map(|e| e.unwrap())
        .collect();
    assert!(matches!(events[0], ChatStreamEvent::Started { .. }));
    assert!(matches!(&events[1], ChatStreamEvent::Delta { content } if content == "Hi"));
    assert!(matches!(events[2], ChatStreamEvent::Completed(_)));
    assert_eq!(events.len(), 3);
}

#[tokio::test]
async fn bad_streamed_calls_fail_without_emitting_any_proposal() {
    let bad: Vec<(&str, Vec<Value>)> = vec![
        (
            "broken args",
            vec![
                chunk(
                    json!({"tool_calls":[tc(0, Some("a"), Some("get_weather"), Some("{\"city\":"))]}),
                    None,
                ),
                chunk(json!({}), Some("tool_calls")),
            ],
        ),
        (
            "truncated by length",
            vec![
                chunk(
                    json!({"tool_calls":[tc(0, Some("a"), Some("get_weather"), Some("{\"city\":\"Pa"))]}),
                    None,
                ),
                chunk(json!({}), Some("length")),
            ],
        ),
        (
            "no id",
            vec![
                chunk(
                    json!({"tool_calls":[tc(0, None, Some("get_weather"), Some("{}"))]}),
                    None,
                ),
                chunk(json!({}), Some("tool_calls")),
            ],
        ),
        (
            "no name",
            vec![
                chunk(
                    json!({"tool_calls":[tc(0, Some("a"), None, Some("{}"))]}),
                    None,
                ),
                chunk(json!({}), Some("tool_calls")),
            ],
        ),
        (
            "second call invalid",
            vec![
                chunk(
                    json!({"tool_calls":[tc(0, Some("a"), Some("get_weather"), Some("{}")), tc(1, Some("b"), Some("get_weather"), Some("{"))]}),
                    None,
                ),
                chunk(json!({}), Some("tool_calls")),
            ],
        ),
    ];
    for (label, events) in bad {
        let bytes = sse(&events);
        let mock = Mock::start(move |_| Reply::Sse {
            chunks: vec![bytes.clone()],
            pause: Duration::ZERO,
            hold_open: false,
        })
        .await;
        let out = collect_stream(&mock, request()).await;
        assert!(matches!(out.last(), Some(Err(_))), "{label}: {out:?}");
        assert!(
            !out.iter().any(|e| matches!(
                e,
                Ok(ChatStreamEvent::ToolCallCompleted { .. }) | Ok(ChatStreamEvent::Completed(_))
            )),
            "{label}: {out:?}"
        );
    }
}

#[tokio::test]
async fn an_incomplete_stream_is_an_error_not_a_proposal() {
    // Ends mid-call with neither [DONE] nor a finish reason.
    let mut bytes = Vec::new();
    bytes.extend(
        format!(
            "data: {}\n\n",
            chunk(
                json!({"tool_calls":[tc(0, Some("a"), Some("get_weather"), Some("{}"))]}),
                None
            )
        )
        .into_bytes(),
    );
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![bytes.clone()],
        pause: Duration::ZERO,
        hold_open: false,
    })
    .await;
    let out = collect_stream(&mock, request()).await;
    assert!(
        matches!(out.last(), Some(Err(RuntimeError::Protocol { .. }))),
        "{out:?}"
    );
    assert!(!out
        .iter()
        .any(|e| matches!(e, Ok(ChatStreamEvent::ToolCallCompleted { .. }))));
}

#[tokio::test]
async fn streaming_cancellation_and_drop_release_the_connection() {
    let first = format!(
        "data: {}\n\n",
        chunk(
            json!({"tool_calls":[tc(0, Some("a"), Some("get_weather"), Some("{\"c"))]}),
            None
        )
    )
    .into_bytes();
    let mock = Mock::start({
        let first = first.clone();
        move |_| Reply::Sse {
            chunks: vec![first.clone()],
            pause: Duration::ZERO,
            hold_open: true,
        }
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
    assert!(matches!(
        stream.next().await,
        Some(Ok(ChatStreamEvent::ToolCallDelta { .. }))
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

    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![first.clone()],
        pause: Duration::ZERO,
        hold_open: true,
    })
    .await;
    let mut stream = provider(&mock).chat_stream(request(), None).await.unwrap();
    stream.next().await;
    stream.next().await;
    drop(stream);
    assert!(mock.wait_disconnected().await);
}

#[tokio::test]
async fn streamed_tool_arguments_are_bounded() {
    let big = "x".repeat(200);
    let bytes = sse(&[
        chunk(
            json!({"tool_calls":[tc(0, Some("a"), Some("get_weather"), Some(&format!("{{\"city\":\"{big}")))]}),
            None,
        ),
        chunk(json!({}), Some("tool_calls")),
    ]);
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![bytes.clone()],
        pause: Duration::ZERO,
        hold_open: false,
    })
    .await;
    let mut config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url));
    config.capabilities.tools = true;
    config.max_output_bytes = 100;
    let p = OpenAiCompatibleProvider::new(config).unwrap();
    let mut stream = p.chat_stream(request(), None).await.unwrap();
    let mut last = None;
    while let Some(item) = stream.next().await {
        last = Some(item);
    }
    assert!(
        matches!(last, Some(Err(RuntimeError::ResourceLimit { .. }))),
        "{last:?}"
    );
}
