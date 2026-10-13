//! A complete caller-owned tool interaction through `Runtime`.
//!
//! The mock "model" proposes a tool call; the *test* plays the application: it
//! validates, authorizes, and executes a deterministic mock tool itself, then
//! submits the result. rust-ml-runtime never runs a tool.

#[path = "../../../providers/http/tests/support/mod.rs"]
mod support;

use futures_util::StreamExt;
use rust_ml_runtime::{
    ChatMessage, ChatRequest, ChatStreamEvent, GenerationOptions, OpenAiCompatibleConfig,
    OpenAiCompatibleProvider, Runtime, RuntimeError, ToolChoice, ToolDefinition,
};
use serde_json::{json, Value};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    sync::Arc,
    time::Duration,
};
use support::{Mock, Reply};

/// Application-owned tool. A real application would authorize before running.
fn get_weather(arguments: &Value) -> Value {
    json!({"city": arguments["city"], "temp_c": 21, "sky": "clear"})
}

fn weather_definition() -> ToolDefinition {
    ToolDefinition::new(
        "get_weather",
        json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}),
    )
    .with_description("Current weather for a city")
}

fn runtime(mock: &Mock) -> Runtime {
    let mut config = OpenAiCompatibleConfig::new(format!("{}/v1", mock.url));
    config.capabilities.tools = true;
    config.capabilities.tool_choice = true;
    Runtime::builder()
        .register_generation_provider(OpenAiCompatibleProvider::new(config).unwrap())
        .build()
}

fn options(choice: Option<ToolChoice>) -> GenerationOptions {
    GenerationOptions {
        tools: vec![weather_definition()],
        tool_choice: choice,
        ..Default::default()
    }
}

fn proposal(args: &str, name: &str) -> String {
    json!({"model":"m","choices":[{"message":{"role":"assistant","content":null,
        "tool_calls":[{"id":"call_1","type":"function","function":{"name":name,"arguments":args}}]},
        "finish_reason":"tool_calls"}]})
    .to_string()
}

fn answer(text: &str) -> String {
    json!({"model":"m","choices":[{"message":{"role":"assistant","content":text},"finish_reason":"stop"}]}).to_string()
}

#[tokio::test]
async fn caller_executes_the_tool_and_submits_the_result() {
    // The model asks for the tool until it sees a tool message.
    let mock = Mock::start(|req| {
        let has_result = req.json()["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["role"] == "tool");
        Reply::Json(
            200,
            if has_result {
                answer("It is 21C and clear in Paris.")
            } else {
                proposal(r#"{"city":"Paris"}"#, "get_weather")
            },
        )
    })
    .await;
    let runtime = runtime(&mock);
    let mut messages = vec![ChatMessage::user("What's the weather in Paris?")];

    let first = runtime
        .chat(
            "openai-compatible",
            ChatRequest::new("m", messages.clone()).with_options(options(Some(ToolChoice::Auto))),
            None,
        )
        .await
        .unwrap();
    // A proposal, nothing more: no result exists yet.
    assert_eq!(first.message.tool_calls.len(), 1);
    assert!(first.message.content.is_empty());

    // === Application-owned section: validate, authorize, execute. ===
    let executed = Arc::new(AtomicUsize::new(0));
    let mut results = Vec::new();
    for call in &first.message.tool_calls {
        let authorized = call.name == "get_weather"; // application policy
        assert!(authorized);
        executed.fetch_add(1, Ordering::SeqCst);
        results.push(ChatMessage::tool_result(
            &call.id,
            get_weather(&call.arguments).to_string(),
        ));
    }
    assert_eq!(executed.load(Ordering::SeqCst), 1);
    // =================================================================

    messages.push(first.message.clone());
    messages.extend(results);
    let second = runtime
        .chat(
            "openai-compatible",
            ChatRequest::new("m", messages).with_options(options(None)),
            None,
        )
        .await
        .unwrap();
    assert_eq!(second.message.content, "It is 21C and clear in Paris.");
    assert!(second.message.tool_calls.is_empty());

    let sent = mock.captured();
    assert_eq!(sent.len(), 2);
    let followup = sent[1].json();
    assert_eq!(followup["messages"][1]["tool_calls"][0]["id"], "call_1");
    assert_eq!(followup["messages"][2]["tool_call_id"], "call_1");
    assert_eq!(
        serde_json::from_str::<Value>(followup["messages"][2]["content"].as_str().unwrap())
            .unwrap()["temp_c"],
        21
    );
}

#[tokio::test]
async fn proposals_are_checked_against_what_the_caller_offered() {
    let cases: Vec<(&str, String, Option<ToolChoice>)> = vec![
        (
            "schema violation",
            proposal(r#"{"city":42}"#, "get_weather"),
            None,
        ),
        ("missing required", proposal("{}", "get_weather"), None),
        (
            "extra property",
            proposal(r#"{"city":"x","evil":true}"#, "get_weather"),
            None,
        ),
        ("undefined tool", proposal("{}", "delete_everything"), None),
        (
            "choice none violated",
            proposal(r#"{"city":"x"}"#, "get_weather"),
            Some(ToolChoice::None),
        ),
    ];
    for (label, body, choice) in cases {
        let mock = Mock::start(move |_| Reply::Json(200, body.clone())).await;
        let error = runtime(&mock)
            .chat(
                "openai-compatible",
                ChatRequest::new("m", vec![ChatMessage::user("x")]).with_options(options(choice)),
                None,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(error, RuntimeError::InvalidOutput { .. }),
            "{label}: {error:?}"
        );
        assert!(
            !error.to_string().contains("evil\""),
            "{label}: argument values must not be echoed"
        );
    }
}

#[tokio::test]
async fn a_named_choice_is_enforced_on_the_response_too() {
    let other = ToolDefinition::new("other", json!({"type":"object"}));
    let mock = Mock::start(|_| Reply::Json(200, proposal("{}", "other"))).await;
    let opts = GenerationOptions {
        tools: vec![weather_definition(), other],
        tool_choice: Some(ToolChoice::Tool {
            name: "get_weather".into(),
        }),
        ..Default::default()
    };
    let error = runtime(&mock)
        .chat(
            "openai-compatible",
            ChatRequest::new("m", vec![ChatMessage::user("x")]).with_options(opts),
            None,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, RuntimeError::InvalidOutput { .. }),
        "{error:?}"
    );
}

#[tokio::test]
async fn invalid_requests_never_reach_the_provider() {
    let mock = Mock::start(|_| Reply::Json(200, answer("x"))).await;
    let runtime = runtime(&mock);
    let unusable = GenerationOptions {
        tools: vec![ToolDefinition::new("t", json!({"type": 12}))],
        ..Default::default()
    };
    let error = runtime
        .chat(
            "openai-compatible",
            ChatRequest::new("m", vec![ChatMessage::user("x")]).with_options(unusable),
            None,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, RuntimeError::InvalidRequest { .. }),
        "{error:?}"
    );
    let orphan = ChatRequest::new(
        "m",
        vec![
            ChatMessage::user("x"),
            ChatMessage::tool_result("nope", "r"),
        ],
    );
    assert!(matches!(
        runtime.chat("openai-compatible", orphan, None).await,
        Err(RuntimeError::InvalidRequest { .. })
    ));
    assert!(mock.captured().is_empty());
}

fn stream_bytes(args: &[&str], finish: &str) -> Vec<u8> {
    let mut out = String::new();
    for (i, fragment) in args.iter().enumerate() {
        let mut call = json!({"index":0,"function":{"arguments":fragment}});
        if i == 0 {
            call["id"] = json!("call_s");
            call["function"]["name"] = json!("get_weather");
        }
        out.push_str(&format!(
            "data: {}\n\n",
            json!({"choices":[{"delta":{"tool_calls":[call]},"finish_reason":null}]})
        ));
    }
    out.push_str(&format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"choices":[{"delta":{},"finish_reason":finish}]})
    ));
    out.into_bytes()
}

#[tokio::test]
async fn streamed_proposals_are_validated_before_any_is_released() {
    for (args, expect_ok) in [
        (vec!["{\"cit", "y\":\"Pa", "ris\"}"], true),
        (vec!["{\"city\":", "7}"], false), // wrong type: nothing may be released
    ] {
        let bytes = stream_bytes(&args, "tool_calls");
        let mock = Mock::start(move |_| Reply::Sse {
            chunks: bytes.chunks(7).map(<[u8]>::to_vec).collect(),
            pause: Duration::ZERO,
            hold_open: false,
        })
        .await;
        let mut stream = runtime(&mock)
            .chat_stream(
                "openai-compatible",
                ChatRequest::new("m", vec![ChatMessage::user("x")]).with_options(options(None)),
                None,
            )
            .await
            .unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item);
        }
        let proposals = events
            .iter()
            .filter(|e| matches!(e, Ok(ChatStreamEvent::ToolCallCompleted { .. })))
            .count();
        if expect_ok {
            assert_eq!(proposals, 1);
            assert!(matches!(
                events.last(),
                Some(Ok(ChatStreamEvent::Completed(_)))
            ));
            let Some(Ok(ChatStreamEvent::ToolCallCompleted { call })) = events
                .iter()
                .find(|e| matches!(e, Ok(ChatStreamEvent::ToolCallCompleted { .. })))
            else {
                unreachable!()
            };
            assert_eq!(call.arguments, json!({"city":"Paris"}));
        } else {
            assert_eq!(proposals, 0, "{events:?}");
            assert!(
                matches!(events.last(), Some(Err(RuntimeError::InvalidOutput { .. }))),
                "{events:?}"
            );
            assert!(!events
                .iter()
                .any(|e| matches!(e, Ok(ChatStreamEvent::Completed(_)))));
        }
    }
}
