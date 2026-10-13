//! Opt-in tests against a real pretrained GGUF that is already on disk.
//!
//! Never downloads anything and is `#[ignore]`d, so ordinary CI never needs
//! weights. Run (see docs/embedded-generation.md for the exact model and how to
//! obtain it):
//!
//! ```text
//! ML_RUNTIME_TEST_GGUF=/path/to/model.gguf \
//! ML_RUNTIME_TEST_GGUF_SHA256=<hex> \
//!   cargo test --release --test real_model -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Assertions are behavioral (nonempty, structurally valid, bounded, correctly
//! terminated), never exact prose from a probabilistic model.

use futures_util::StreamExt;
use ml_runtime_llama_provider::{LlamaConfig, LlamaProvider, PROVIDER_NAME};
use rust_ml_runtime::{
    CancellationToken, ChatMessage, ChatRequest, ChatStreamEvent, FinishReason, GenerationOptions,
    ResponseFormat, Runtime, RuntimeError,
};
use serde_json::json;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

/// Turns the environment into a model path or a message naming what is missing.
fn prerequisite(var: Option<String>) -> Result<PathBuf, String> {
    let path = var.filter(|v| !v.trim().is_empty()).ok_or_else(|| {
        "PREREQUISITE MISSING: set ML_RUNTIME_TEST_GGUF to a GGUF file that is already on disk \
         (weights are never downloaded by these tests; see docs/embedded-generation.md)"
            .to_owned()
    })?;
    let path = PathBuf::from(path);
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "PREREQUISITE MISSING: ML_RUNTIME_TEST_GGUF={} is not a file",
            path.display()
        ))
    }
}

#[test]
fn prerequisite_detection_names_what_is_missing() {
    assert!(prerequisite(None)
        .unwrap_err()
        .contains("ML_RUNTIME_TEST_GGUF"));
    assert!(prerequisite(Some("  ".into()))
        .unwrap_err()
        .contains("PREREQUISITE"));
    let missing = prerequisite(Some("/definitely/not/here.gguf".into())).unwrap_err();
    assert!(missing.contains("is not a file"), "{missing}");
}

fn model_path() -> PathBuf {
    prerequisite(std::env::var("ML_RUNTIME_TEST_GGUF").ok()).unwrap_or_else(|m| panic!("{m}"))
}

fn config() -> LlamaConfig {
    let mut config = LlamaConfig::new(model_path(), "real-model");
    config.expected_sha256 = std::env::var("ML_RUNTIME_TEST_GGUF_SHA256").ok();
    config.context_tokens = 2048;
    config.max_output_tokens = 256;
    // ML_RUNTIME_TEST_GPU_LAYERS=99 runs the whole suite through Metal (macOS).
    if let Ok(layers) = std::env::var("ML_RUNTIME_TEST_GPU_LAYERS") {
        config.gpu_layers = layers
            .parse()
            .expect("ML_RUNTIME_TEST_GPU_LAYERS must be a number");
    }
    config
}

fn load() -> LlamaProvider {
    LlamaProvider::new(config()).unwrap_or_else(|e| panic!("could not load the model: {e}"))
}

fn runtime() -> Runtime {
    Runtime::builder()
        .register_generation_provider(load())
        .build()
}

fn ask(prompt: &str, max: u32) -> ChatRequest {
    ChatRequest::new("real-model", vec![ChatMessage::user(prompt)]).with_options(
        GenerationOptions {
            max_tokens: Some(max),
            temperature: Some(0.0),
            ..Default::default()
        },
    )
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires ML_RUNTIME_TEST_GGUF"]
async fn loads_reports_metadata_and_answers_usefully() {
    let provider = load();
    let info = provider.model_info().clone();
    println!(
        "model {:?} arch {} params {} bytes {} train_ctx {} verify {:?} load {:?}\ndevices {:?}",
        info.reported_name,
        info.architecture,
        info.parameters,
        info.file_bytes,
        info.train_context,
        info.verify_time,
        info.load_time,
        info.devices
    );
    assert!(info.parameters > 1_000_000 && info.file_bytes > 1_000_000);
    let rt = Runtime::builder()
        .register_generation_provider(provider)
        .build();
    let models = rt
        .list_generation_models(PROVIDER_NAME, None)
        .await
        .unwrap();
    assert_eq!(models[0].id, "real-model");

    let response = rt
        .chat(
            PROVIDER_NAME,
            ask("What is the capital of France? Answer in one word.", 32),
            None,
        )
        .await
        .unwrap();
    println!(
        "reply {:?} {:?}",
        response.message.content, response.finish_reason
    );
    // Behavioral, not exact prose: a 0.5B model answers this reliably, and the
    // answer must be about Paris. A garbage model or broken template does not.
    assert!(
        response.message.content.to_lowercase().contains("paris"),
        "{:?}",
        response.message.content
    );
    assert!(response.usage.unwrap().completion_tokens.unwrap() > 0);
    assert_eq!(
        response.metadata.reported_model.as_deref(),
        Some(info.reported_name.as_str())
    );
    assert_eq!(response.metadata.requested_model, "real-model");
}

#[test]
#[ignore = "requires ML_RUNTIME_TEST_GGUF"]
fn chat_template_renders_the_models_conversation_format() {
    let provider = load();
    let prompt = provider
        .render_prompt(&[
            ChatMessage::system("SYS-MARK"),
            ChatMessage::user("USER-MARK"),
            ChatMessage::assistant("ASSISTANT-MARK"),
            ChatMessage::user("SECOND-MARK"),
        ])
        .unwrap();
    println!("{prompt:?}");
    let at = |needle: &str| {
        prompt
            .find(needle)
            .unwrap_or_else(|| panic!("{needle:?} missing in {prompt:?}"))
    };
    // Roles appear in order, each marker is delimited by role tokens, and the
    // prompt ends by opening the assistant turn (so the model answers).
    assert!(at("SYS-MARK") < at("USER-MARK") && at("USER-MARK") < at("ASSISTANT-MARK"));
    assert!(at("ASSISTANT-MARK") < at("SECOND-MARK"));
    assert!(
        prompt.contains("<|im_start|>system\nSYS-MARK<|im_end|>"),
        "{prompt:?}"
    );
    assert!(
        prompt.contains("<|im_start|>user\nUSER-MARK<|im_end|>"),
        "{prompt:?}"
    );
    assert!(
        prompt.contains("<|im_start|>assistant\nASSISTANT-MARK<|im_end|>"),
        "{prompt:?}"
    );
    assert!(prompt.ends_with("<|im_start|>assistant\n"), "{prompt:?}");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires ML_RUNTIME_TEST_GGUF"]
async fn streaming_is_incremental_and_matches_the_blocking_reply() {
    let rt = runtime();
    let request = || ask("Write two sentences about the ocean.", 64);
    let blocking = rt.chat(PROVIDER_NAME, request(), None).await.unwrap();

    let started = Instant::now();
    let mut stream = rt
        .chat_stream(PROVIDER_NAME, request(), None)
        .await
        .unwrap();
    let (mut text, mut deltas, mut first, mut completed) = (String::new(), 0, None, false);
    let mut first_delta_gap = None;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            ChatStreamEvent::Delta { content } => {
                deltas += 1;
                text.push_str(&content);
                if first.is_none() {
                    first = Some(started.elapsed());
                } else if first_delta_gap.is_none() {
                    first_delta_gap = Some(started.elapsed());
                }
            }
            ChatStreamEvent::Completed(summary) => {
                completed = true;
                assert!(summary.usage.unwrap().completion_tokens.unwrap() > 1);
            }
            _ => {}
        }
    }
    let total = started.elapsed();
    println!("{deltas} deltas, first {first:?}, total {total:?}");
    assert!(completed);
    assert!(
        deltas > 5,
        "output arrived in {deltas} chunk(s), not incrementally"
    );
    assert!(
        first.unwrap() < total,
        "first delta must precede completion"
    );
    assert_eq!(
        text, blocking.message.content,
        "greedy stream equals the blocking reply"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires ML_RUNTIME_TEST_GGUF"]
async fn stop_sequences_and_length_limits_behave_as_documented() {
    let rt = runtime();
    let mut count = ask(
        "Count from 1 to 10 separated by commas. Output only the numbers.",
        64,
    );
    let free = rt.chat(PROVIDER_NAME, count.clone(), None).await.unwrap();
    println!("free {:?}", free.message.content);
    assert!(
        free.message.content.contains('5'),
        "{:?}",
        free.message.content
    );

    count.options.stop = vec!["5".to_owned()];
    let stopped = rt.chat(PROVIDER_NAME, count, None).await.unwrap();
    println!("stopped {:?}", stopped.message.content);
    assert_eq!(stopped.finish_reason, Some(FinishReason::Stop));
    assert!(
        !stopped.message.content.contains('5'),
        "the stop sequence leaked"
    );
    assert!(free.message.content.starts_with(&stopped.message.content));

    let limited = rt
        .chat(
            PROVIDER_NAME,
            ask("Write a long story about a dragon.", 7),
            None,
        )
        .await
        .unwrap();
    assert_eq!(limited.finish_reason, Some(FinishReason::Length));
    assert_eq!(limited.usage.unwrap().completion_tokens, Some(7));

    // Above the provider cap: refused, not clamped.
    assert!(matches!(
        rt.chat(PROVIDER_NAME, ask("hi", 257), None).await,
        Err(RuntimeError::ResourceLimit { .. })
    ));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires ML_RUNTIME_TEST_GGUF"]
async fn grammar_constrained_json_is_valid_and_still_validated_by_the_runtime() {
    let rt = runtime();
    let schema = json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "population_millions": {"type": "number"},
            "landmarks": {"type": "array", "items": {"type": "string"}, "maxItems": 3}
        },
        "required": ["city", "population_millions", "landmarks"],
        "additionalProperties": false
    });
    let mut request = ask(
        "Describe Tokyo as JSON with its name, population in millions, and up to three landmarks.",
        200,
    );
    request.options.response_format = ResponseFormat::JsonSchema {
        name: "city".into(),
        schema: schema.clone(),
        strict: true,
    };
    let response = rt.chat(PROVIDER_NAME, request.clone(), None).await.unwrap();
    println!(
        "{:?} {:?}",
        response.message.content, response.finish_reason
    );
    let value: serde_json::Value = serde_json::from_str(&response.message.content)
        .unwrap_or_else(|e| panic!("not JSON: {e}: {:?}", response.message.content));
    assert!(value["city"].is_string() && value["population_millions"].is_number());
    assert!(value["landmarks"].as_array().unwrap().len() <= 3);
    assert!(
        response
            .structured
            .expect("runtime validation must run")
            .is_validated(),
        "{:?}",
        response.message.content
    );

    // Streaming path: the runtime validates the assembled stream too.
    let mut stream = rt.chat_stream(PROVIDER_NAME, request, None).await.unwrap();
    let mut validated = None;
    while let Some(event) = stream.next().await {
        if let ChatStreamEvent::Completed(summary) = event.unwrap() {
            validated = summary.structured.map(|s| s.is_validated());
        }
    }
    assert_eq!(validated, Some(true));

    // Truncation is reported as invalid by the runtime, never repaired.
    let mut cut = ask("Describe Tokyo as JSON.", 3);
    cut.options.response_format = ResponseFormat::JsonObject;
    let cut = rt.chat(PROVIDER_NAME, cut, None).await.unwrap();
    assert_eq!(cut.finish_reason, Some(FinishReason::Length));
    assert!(!cut.structured.unwrap().is_validated());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires ML_RUNTIME_TEST_GGUF"]
async fn identity_and_integrity_failures_never_substitute_or_infer() {
    let rt = runtime();
    let mut other = ask("hi", 8);
    other.model = "some-other-model".to_owned();
    assert!(matches!(
        rt.chat(PROVIDER_NAME, other.clone(), None).await,
        Err(RuntimeError::ModelNotFound { model }) if model == "some-other-model"
    ));
    assert!(matches!(
        rt.chat_stream(PROVIDER_NAME, other, None).await,
        Err(RuntimeError::ModelNotFound { .. })
    ));

    // A wrong digest fails before llama.cpp is asked to load or run anything.
    let mut wrong = config();
    wrong.expected_sha256 = Some("0".repeat(64));
    assert!(matches!(
        LlamaProvider::new(wrong),
        Err(RuntimeError::ModelIntegrity { .. })
    ));

    // The right digest (when supplied) is what the load actually verified.
    if let Ok(digest) = std::env::var("ML_RUNTIME_TEST_GGUF_SHA256") {
        let provider = load();
        assert!(
            provider.model_info().verify_time > Duration::ZERO,
            "digest {digest} not checked"
        );
    }
}

#[test]
#[ignore = "requires ML_RUNTIME_TEST_GGUF"]
fn load_failures_are_useful() {
    let good = model_path();
    let dir = std::env::temp_dir().join(format!("ml-runtime-real-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // Truncated real file: GGUF magic, but not loadable.
    let truncated = dir.join("truncated.gguf");
    let bytes = std::fs::read(&good).unwrap();
    std::fs::write(&truncated, &bytes[..64 * 1024]).unwrap();
    let error = LlamaProvider::new(LlamaConfig::new(&truncated, "real-model")).unwrap_err();
    println!("truncated: {error}");
    assert!(
        matches!(error, RuntimeError::UnsupportedModel { .. }),
        "{error:?}"
    );
    assert!(error.to_string().to_lowercase().contains("llama.cpp"));

    // Missing file names the file and says nothing is downloaded.
    let error =
        LlamaProvider::new(LlamaConfig::new(dir.join("absent.gguf"), "real-model")).unwrap_err();
    assert!(matches!(error, RuntimeError::ModelUnavailable { .. }));
    assert!(error.to_string().contains("never downloaded"), "{error}");

    // Not GGUF at all.
    let text = dir.join("notes.gguf");
    std::fs::write(&text, b"hello").unwrap();
    assert!(matches!(
        LlamaProvider::new(LlamaConfig::new(&text, "real-model")).unwrap_err(),
        RuntimeError::UnsupportedModel { .. }
    ));

    // Context beyond the trained window.
    let mut config = config();
    config.context_tokens = 10_000_000;
    config.max_output_tokens = 256;
    assert!(matches!(
        LlamaProvider::new(config).unwrap_err(),
        RuntimeError::InvalidRequest { .. }
    ));
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires ML_RUNTIME_TEST_GGUF"]
async fn cancellation_terminates_generation_promptly() {
    let rt = runtime();
    let long = || ask("Write a very long story about a dragon and a knight.", 256);

    let token = CancellationToken::new();
    let mut stream = rt
        .chat_stream(PROVIDER_NAME, long(), Some(token.clone()))
        .await
        .unwrap();
    let (mut n, mut at, mut last) = (0, None, None);
    while let Some(event) = stream.next().await {
        if matches!(event, Ok(ChatStreamEvent::Delta { .. })) {
            n += 1;
            if n == 3 {
                at = Some(Instant::now());
                token.cancel();
            }
        }
        last = Some(event);
    }
    let latency = at
        .expect("generation ended before it could be cancelled")
        .elapsed();
    println!("cancel latency {latency:?}");
    assert!(
        matches!(last, Some(Err(RuntimeError::Cancelled))),
        "{last:?}"
    );
    assert!(latency < Duration::from_secs(1), "{latency:?}");
    assert!(n < 40, "{n} deltas arrived after cancel");

    // Dropping a stream frees the slot for the next request.
    let stream = rt.chat_stream(PROVIDER_NAME, long(), None).await.unwrap();
    drop(stream);
    let started = Instant::now();
    rt.chat(PROVIDER_NAME, ask("hi", 1), None).await.unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "{:?}",
        started.elapsed()
    );

    // Cancelled before admission or before the first token.
    let token = CancellationToken::new();
    token.cancel();
    assert!(matches!(
        rt.chat(PROVIDER_NAME, long(), Some(token)).await,
        Err(RuntimeError::Cancelled)
    ));
}
