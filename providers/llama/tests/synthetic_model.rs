//! Real llama.cpp inference over deterministic, generated, random-weight GGUF
//! fixtures. Nothing is downloaded and no model licence is involved. Output text
//! is gibberish by construction; these tests prove the engine, the contract
//! mapping, streaming, cancellation, and resource handling, not model quality.

use futures_util::StreamExt;
use ml_runtime_llama_provider::{
    synthetic::{write_synthetic_gguf, SyntheticSpec},
    LlamaConfig, LlamaProvider, PROVIDER_NAME,
};
use rust_ml_runtime::{
    CancellationToken, ChatMessage, ChatRequest, ChatStreamEvent, ExecutionLocation, FinishReason,
    GenerationOptions, GenerationProvider, ResponseFormat, Runtime, RuntimeError, ToolDefinition,
};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn fixture(spec: &SyntheticSpec) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ml-runtime-llama-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("synthetic.gguf");
    write_synthetic_gguf(&path, spec).unwrap();
    path
}

fn config(spec: &SyntheticSpec) -> LlamaConfig {
    let mut config = LlamaConfig::new(fixture(spec), "synthetic");
    config.context_tokens = 512;
    config.max_output_tokens = 256;
    config.threads = Some(2);
    config
}

fn no_eos() -> SyntheticSpec {
    SyntheticSpec {
        with_eos: false,
        ..SyntheticSpec::tiny()
    }
}

fn provider(spec: &SyntheticSpec) -> LlamaProvider {
    LlamaProvider::new(config(spec)).unwrap()
}

fn runtime(provider: LlamaProvider) -> Runtime {
    Runtime::builder()
        .register_generation_provider(provider)
        .build()
}

fn request(max_tokens: u32) -> ChatRequest {
    ChatRequest::new(
        "synthetic",
        vec![
            ChatMessage::system("Be brief."),
            ChatMessage::user("Hello there"),
        ],
    )
    .with_options(GenerationOptions {
        max_tokens: Some(max_tokens),
        ..Default::default()
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn registers_lists_and_reports_the_loaded_model() {
    let p = provider(&no_eos());
    assert_eq!(p.model_info().reported_name, "synthetic-tiny");
    assert_eq!(p.model_info().architecture, "llama");
    let caps = p.capabilities();
    assert!(caps.local && !caps.remote && caps.streaming);
    let rt = runtime(p);
    let listed = rt.generation_providers();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].name, PROVIDER_NAME);
    let models = rt
        .list_generation_models(PROVIDER_NAME, None)
        .await
        .unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "synthetic");
    assert!(models[0]
        .owned_by
        .as_deref()
        .unwrap()
        .contains("synthetic-tiny"));
}

#[tokio::test(flavor = "multi_thread")]
async fn real_generation_is_deterministic_and_reports_exact_metadata() {
    let rt = runtime(provider(&no_eos()));
    let a = rt.chat(PROVIDER_NAME, request(24), None).await.unwrap();
    let b = rt.chat(PROVIDER_NAME, request(24), None).await.unwrap();
    assert_eq!(
        a.message.content, b.message.content,
        "greedy decoding is deterministic"
    );
    assert!(!a.message.content.is_empty());
    assert_eq!(a.finish_reason, Some(FinishReason::Length));
    let usage = a.usage.unwrap();
    assert_eq!(usage.completion_tokens, Some(24));
    assert_eq!(usage.total_tokens, Some(usage.prompt_tokens.unwrap() + 24));
    assert!(
        usage.prompt_tokens.unwrap() > 10,
        "the chat template adds tokens"
    );
    let m = a.metadata;
    assert_eq!(
        (m.provider.as_str(), m.transport.as_str()),
        (PROVIDER_NAME, "in-process")
    );
    assert_eq!(m.execution, ExecutionLocation::Local);
    assert!(!m.execution_inferred && !m.streamed);
    assert_eq!(m.requested_model, "synthetic");
    assert_eq!(m.reported_model.as_deref(), Some("synthetic-tiny"));
    assert!(m.time_to_first_token.is_some() && m.endpoint.starts_with("local-file:"));
    assert!(
        !m.endpoint.contains('/'),
        "no filesystem path is exposed: {}",
        m.endpoint
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_different_model_id_is_never_substituted() {
    let rt = runtime(provider(&no_eos()));
    let mut req = request(4);
    req.model = "something-else".into();
    let error = rt.chat(PROVIDER_NAME, req.clone(), None).await.unwrap_err();
    assert_eq!(
        error,
        RuntimeError::ModelNotFound {
            model: "something-else".into()
        }
    );
    assert!(matches!(
        rt.chat_stream(PROVIDER_NAME, req, None).await,
        Err(RuntimeError::ModelNotFound { .. })
    ));
}

#[test]
fn model_loading_failures_are_distinct() {
    // Missing file: a prerequisite error that says weights are not downloaded.
    let mut missing = LlamaConfig::new("/nonexistent/model.gguf", "m");
    missing.threads = Some(1);
    let error = LlamaProvider::new(missing).unwrap_err();
    assert!(
        matches!(&error, RuntimeError::ModelUnavailable { reason, .. } if reason.contains("never downloaded automatically")),
        "{error:?}"
    );

    // Not a GGUF.
    let dir = std::env::temp_dir().join(format!("ml-runtime-llama-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bad = dir.join("not.gguf");
    std::fs::write(&bad, b"this is not a model").unwrap();
    let error = LlamaProvider::new(LlamaConfig::new(&bad, "m")).unwrap_err();
    assert!(
        matches!(error, RuntimeError::UnsupportedModel { .. }),
        "{error:?}"
    );

    // Truncated GGUF: recognized header, fails to load.
    let good = fixture(&SyntheticSpec::tiny());
    let bytes = std::fs::read(&good).unwrap();
    let truncated = dir.join("truncated.gguf");
    std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
    let error = LlamaProvider::new(LlamaConfig::new(&truncated, "m")).unwrap_err();
    assert!(
        matches!(error, RuntimeError::UnsupportedModel { .. }),
        "{error:?}"
    );

    // Integrity.
    let mut wrong = LlamaConfig::new(&good, "m");
    wrong.expected_sha256 = Some("0".repeat(64));
    assert!(matches!(
        LlamaProvider::new(wrong).unwrap_err(),
        RuntimeError::ModelIntegrity { .. }
    ));
    // (A correct digest loads; computed with the same function the provider uses.)
    let mut right = LlamaConfig::new(&good, "m");
    right.expected_sha256 = Some({
        use sha2::{Digest, Sha256};
        Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    });
    right.threads = Some(1);
    right.context_tokens = 512;
    right.max_output_tokens = 256;
    LlamaProvider::new(right).unwrap();

    // A context window beyond what the model was trained for is refused, not silently degraded.
    let mut too_long = LlamaConfig::new(&good, "m");
    too_long.context_tokens = 4096;
    too_long.max_output_tokens = 256;
    assert!(matches!(
        LlamaProvider::new(too_long).unwrap_err(),
        RuntimeError::InvalidRequest { reason } if reason.contains("trained context")
    ));

    // Configuration validation.
    for bad in [
        LlamaConfig {
            model_id: " ".into(),
            ..LlamaConfig::new(&good, "m")
        },
        LlamaConfig {
            context_tokens: 0,
            ..LlamaConfig::new(&good, "m")
        },
        LlamaConfig {
            max_concurrent_requests: 0,
            ..LlamaConfig::new(&good, "m")
        },
        LlamaConfig {
            max_output_tokens: 4096,
            ..LlamaConfig::new(&good, "m")
        },
    ] {
        assert!(matches!(
            LlamaProvider::new(bad).unwrap_err(),
            RuntimeError::InvalidRequest { .. }
        ));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn unsupported_features_are_rejected_not_ignored() {
    let rt = runtime(provider(&no_eos()));
    let with_tool = request(4).with_options(GenerationOptions {
        tools: vec![ToolDefinition::new("t", json!({"type":"object"}))],
        ..Default::default()
    });
    assert!(matches!(
        rt.chat(PROVIDER_NAME, with_tool, None).await,
        Err(RuntimeError::CapabilityMismatch { .. })
    ));
    let history = ChatRequest::new(
        "synthetic",
        vec![
            ChatMessage::user("x"),
            ChatMessage::assistant_with_tool_calls(
                "",
                vec![rust_ml_runtime::ToolCall::new("c", "t", json!({}))],
            ),
            ChatMessage::tool_result("c", "r"),
        ],
    );
    assert!(matches!(
        rt.chat(PROVIDER_NAME, history, None).await,
        Err(RuntimeError::CapabilityMismatch { .. })
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn resource_limits_fail_cleanly() {
    let rt = runtime(provider(&no_eos()));
    // max_tokens above the configured cap.
    let error = rt
        .chat(PROVIDER_NAME, request(10_000), None)
        .await
        .unwrap_err();
    assert!(
        matches!(error, RuntimeError::ResourceLimit { .. }),
        "{error:?}"
    );
    // Prompt that cannot fit the context window.
    let huge = ChatRequest::new("synthetic", vec![ChatMessage::user("word ".repeat(4000))]);
    let error = rt.chat(PROVIDER_NAME, huge, None).await.unwrap_err();
    assert!(
        matches!(&error, RuntimeError::ResourceLimit { reason } if reason.contains("context window")),
        "{error:?}"
    );
    // The provider is still healthy afterwards.
    rt.chat(PROVIDER_NAME, request(4), None).await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_is_incremental_and_matches_the_blocking_result() {
    let rt = runtime(provider(&no_eos()));
    let whole = rt.chat(PROVIDER_NAME, request(40), None).await.unwrap();
    let mut stream = rt
        .chat_stream(PROVIDER_NAME, request(40), None)
        .await
        .unwrap();
    assert!(matches!(
        stream.next().await,
        Some(Ok(ChatStreamEvent::Started { .. }))
    ));
    let mut text = String::new();
    let mut deltas = 0;
    let mut completed = None;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            ChatStreamEvent::Delta { content } => {
                assert!(completed.is_none(), "no text after completion");
                deltas += 1;
                text.push_str(&content);
            }
            ChatStreamEvent::Completed(summary) => completed = Some(summary),
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(text, whole.message.content);
    assert!(
        deltas > 10,
        "text arrives token by token, got {deltas} deltas"
    );
    let summary = completed.unwrap();
    assert_eq!(summary.finish_reason, Some(FinishReason::Length));
    assert_eq!(summary.usage.unwrap().completion_tokens, Some(40));
    assert!(summary.metadata.streamed && summary.metadata.stream_events == Some(deltas));
    assert!(summary.metadata.time_to_first_token.unwrap() < summary.metadata.duration);
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_sequences_are_honored_even_when_split_across_tokens() {
    let p = provider(&no_eos());
    let rt = runtime(p);
    let base = rt
        .chat(PROVIDER_NAME, request(40), None)
        .await
        .unwrap()
        .message
        .content;
    let chars: Vec<char> = base.chars().collect();
    assert!(chars.len() > 12);
    // A multi-character stop sequence spanning several one-character tokens.
    let stop: String = chars[6..9].iter().collect();
    let expected_end = base.find(&stop).unwrap();
    let mut req = request(40);
    req.options.stop = vec![stop.clone()];
    let stopped = rt.chat(PROVIDER_NAME, req.clone(), None).await.unwrap();
    assert_eq!(stopped.message.content, &base[..expected_end]);
    assert_eq!(stopped.finish_reason, Some(FinishReason::Stop));
    assert!(!stopped.message.content.contains(&stop));
    // Streaming never leaks any part of the stop sequence.
    let mut stream = rt.chat_stream(PROVIDER_NAME, req, None).await.unwrap();
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        if let ChatStreamEvent::Delta { content } = event.unwrap() {
            text.push_str(&content);
        }
    }
    assert_eq!(text, &base[..expected_end]);
}

#[tokio::test(flavor = "multi_thread")]
async fn sampling_options_are_honored() {
    let rt = runtime(provider(&no_eos()));
    let sampled = |seed: i64| {
        let mut r = request(24);
        r.options.temperature = Some(1.5);
        r.options.top_p = Some(0.95);
        r.options.seed = Some(seed);
        r
    };
    let a = rt
        .chat(PROVIDER_NAME, sampled(1), None)
        .await
        .unwrap()
        .message
        .content;
    let b = rt
        .chat(PROVIDER_NAME, sampled(1), None)
        .await
        .unwrap()
        .message
        .content;
    let c = rt
        .chat(PROVIDER_NAME, sampled(2), None)
        .await
        .unwrap()
        .message
        .content;
    assert_eq!(a, b, "same seed, same sampling");
    assert_ne!(a, c, "different seed, different sample");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_eos_token_ends_generation_naturally() {
    let rt = runtime(provider(&SyntheticSpec::tiny()));
    let mut ended_early = false;
    for seed in 0..40 {
        let mut r = request(200);
        r.options.temperature = Some(2.0);
        r.options.seed = Some(seed);
        let response = rt.chat(PROVIDER_NAME, r, None).await.unwrap();
        if response.finish_reason == Some(FinishReason::Stop) {
            assert!(response.usage.unwrap().completion_tokens.unwrap() < 200);
            ended_early = true;
            break;
        }
    }
    assert!(
        ended_early,
        "the EOS token should end some generations before the limit"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn grammar_constrained_json_output_is_enforced_then_still_validated() {
    let rt = runtime(provider(&SyntheticSpec::tiny()));
    let schema = json!({"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false});
    let mut r = request(64);
    r.options.response_format = ResponseFormat::JsonSchema {
        name: "out".into(),
        schema,
        strict: true,
    };
    let response = rt.chat(PROVIDER_NAME, r, None).await.unwrap();
    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Stop),
        "{:?}",
        response.message.content
    );
    let value: serde_json::Value = serde_json::from_str(&response.message.content).unwrap();
    assert_eq!(value, json!({"ok": true}));
    assert!(response.structured.unwrap().is_validated());

    // json_object: any generated text parses as an object, or generation was cut off
    // and the runtime reports it as invalid rather than repairing it.
    let mut r = request(200);
    r.options.response_format = ResponseFormat::JsonObject;
    let response = rt.chat(PROVIDER_NAME, r, None).await.unwrap();
    let structured = response.structured.unwrap();
    if response.finish_reason == Some(FinishReason::Stop) {
        assert!(structured.is_validated(), "{}", response.message.content);
    } else {
        assert!(!structured.is_validated());
    }

    // A schema llama.cpp cannot convert is rejected up front.
    let mut r = request(8);
    r.options.response_format = ResponseFormat::JsonSchema {
        name: "bad".into(),
        schema: json!({"$ref": "#/nope"}),
        strict: false,
    };
    assert!(matches!(
        rt.chat(PROVIDER_NAME, r, None).await,
        Err(RuntimeError::InvalidRequest { .. })
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn cancellation_stops_generation_promptly_and_frees_the_slot() {
    let p = provider(&no_eos());
    let rt = runtime(p);
    let token = CancellationToken::new();
    let mut stream = rt
        .chat_stream(PROVIDER_NAME, request(250), Some(token.clone()))
        .await
        .unwrap();
    let mut seen = 0;
    let mut cancelled_at = None;
    let mut last = None;
    while let Some(event) = stream.next().await {
        if matches!(event, Ok(ChatStreamEvent::Delta { .. })) {
            seen += 1;
            if seen == 5 {
                token.cancel();
                cancelled_at = Some(Instant::now());
            }
        }
        last = Some(event);
    }
    assert!(
        matches!(last, Some(Err(RuntimeError::Cancelled))),
        "{last:?}"
    );
    // Bounded buffer (32) + in-flight token: generation cannot have run to the limit.
    assert!(seen < 5 + 40, "kept generating after cancel: {seen}");
    assert!(cancelled_at.unwrap().elapsed() < Duration::from_secs(5));
    // The slot is released: a new request completes.
    tokio::time::timeout(
        Duration::from_secs(20),
        rt.chat(PROVIDER_NAME, request(4), None),
    )
    .await
    .expect("slot must be free after cancellation")
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn dropping_a_stream_releases_generation_and_the_slot() {
    let p = provider(&no_eos());
    assert_eq!(p.available_slots(), 1);
    let rt = runtime(p);
    let mut stream = rt
        .chat_stream(PROVIDER_NAME, request(250), None)
        .await
        .unwrap();
    stream.next().await; // started
    stream.next().await; // first delta
    drop(stream);
    // No background generation may keep the single slot: the next request must run.
    let started = Instant::now();
    tokio::time::timeout(
        Duration::from_secs(20),
        rt.chat(PROVIDER_NAME, request(4), None),
    )
    .await
    .expect("abandoned stream must not hold the slot")
    .unwrap();
    assert!(started.elapsed() < Duration::from_secs(20));
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_requests_are_bounded_and_waiting_is_cancellable() {
    let rt = runtime(provider(&no_eos())); // max_concurrent_requests = 1
                                           // First request holds the only slot: it fills the stream buffer and blocks on backpressure.
    let mut first = rt
        .chat_stream(PROVIDER_NAME, request(250), None)
        .await
        .unwrap();
    first.next().await;
    // A second request must wait, and waiting must be cancellable.
    let token = CancellationToken::new();
    let t = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        t.cancel();
    });
    let waited = Instant::now();
    let error = rt
        .chat(PROVIDER_NAME, request(4), Some(token))
        .await
        .unwrap_err();
    assert_eq!(error, RuntimeError::Cancelled);
    assert!(
        waited.elapsed() >= Duration::from_millis(90),
        "second request must not have run concurrently"
    );
    // Finishing the first request frees the slot for the next.
    drop(first);
    tokio::time::timeout(
        Duration::from_secs(20),
        rt.chat(PROVIDER_NAME, request(4), None),
    )
    .await
    .unwrap()
    .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn two_slots_serve_two_requests_against_one_loaded_model() {
    let mut cfg = config(&no_eos());
    cfg.max_concurrent_requests = 2;
    let rt = runtime(LlamaProvider::new(cfg).unwrap());
    let (a, b) = tokio::join!(
        rt.chat(PROVIDER_NAME, request(16), None),
        rt.chat(PROVIDER_NAME, request(16), None)
    );
    let (a, b) = (a.unwrap(), b.unwrap());
    assert_eq!(
        a.message.content, b.message.content,
        "shared model, independent contexts"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_loaded_model_outlives_the_provider_while_a_stream_uses_it() {
    let rt = runtime(provider(&no_eos()));
    let mut stream = rt
        .chat_stream(PROVIDER_NAME, request(30), None)
        .await
        .unwrap();
    drop(rt); // provider handle dropped while the stream is mid-flight
    let mut completed = false;
    while let Some(event) = stream.next().await {
        completed |= matches!(event.unwrap(), ChatStreamEvent::Completed(_));
    }
    assert!(
        completed,
        "the in-flight request keeps the model alive and finishes"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn dropping_a_blocking_chat_future_stops_generation_and_frees_the_slot() {
    let rt = runtime(provider(&no_eos()));
    // The future is dropped by the timeout long before 256 tokens finish.
    let long = request(256);
    let timed_out = tokio::time::timeout(
        std::time::Duration::from_millis(30),
        rt.chat(PROVIDER_NAME, long, None),
    )
    .await;
    assert!(
        timed_out.is_err(),
        "the request was expected to outlive the timeout"
    );
    let started = std::time::Instant::now();
    rt.chat(PROVIDER_NAME, request(1), None).await.unwrap();
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "slot was not released promptly: {:?}",
        started.elapsed()
    );
}
