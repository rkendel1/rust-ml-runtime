//! Opt-in test against a real GGUF model that is already on disk.
//!
//! Never downloads anything. Run (see docs/local-generation.md for how to obtain
//! a model):
//!
//! ```text
//! ML_RUNTIME_TEST_GGUF=/path/to/model.gguf \
//! ML_RUNTIME_TEST_GGUF_SHA256=<hex, optional> \
//!   cargo test --test real_model -- --ignored --nocapture
//! ```

use futures_util::StreamExt;
use ml_runtime_llama_provider::{LlamaConfig, LlamaProvider, PROVIDER_NAME};
use rust_ml_runtime::{
    CancellationToken, ChatMessage, ChatRequest, ChatStreamEvent, GenerationOptions, Runtime,
    RuntimeError,
};
use std::time::Instant;

fn load() -> LlamaProvider {
    let Ok(path) = std::env::var("ML_RUNTIME_TEST_GGUF") else {
        panic!(
            "PREREQUISITE MISSING: set ML_RUNTIME_TEST_GGUF to a GGUF file that is already on disk \
             (weights are never downloaded by this test; see docs/local-generation.md)"
        );
    };
    let mut config = LlamaConfig::new(&path, "real-model");
    config.expected_sha256 = std::env::var("ML_RUNTIME_TEST_GGUF_SHA256").ok();
    config.context_tokens = 2048;
    config.max_output_tokens = 256;
    LlamaProvider::new(config).unwrap_or_else(|e| panic!("could not load {path}: {e}"))
}

fn request(max: u32) -> ChatRequest {
    ChatRequest::new(
        "real-model",
        vec![ChatMessage::user(
            "Name three primary colors, comma separated.",
        )],
    )
    .with_options(GenerationOptions {
        max_tokens: Some(max),
        temperature: Some(0.0),
        ..Default::default()
    })
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires ML_RUNTIME_TEST_GGUF pointing at an installed GGUF model"]
async fn real_model_generates_streams_and_cancels() {
    let provider = load();
    let info = provider.model_info().clone();
    println!(
        "model: {} ({}), arch {}, {} params, {} bytes, train ctx {}, load {:?}",
        info.reported_name,
        info.id,
        info.architecture,
        info.parameters,
        info.file_bytes,
        info.train_context,
        info.load_time
    );
    let rt = Runtime::builder()
        .register_generation_provider(provider)
        .build();

    let response = rt.chat(PROVIDER_NAME, request(48), None).await.unwrap();
    println!("backend notes: {:?}", rt.generation_providers()[0].notes);
    println!(
        "reply: {:?} ({:?})",
        response.message.content, response.finish_reason
    );
    assert!(!response.message.content.trim().is_empty());
    assert_eq!(
        response.metadata.reported_model.as_deref(),
        Some(info.reported_name.as_str())
    );

    let mut stream = rt
        .chat_stream(PROVIDER_NAME, request(48), None)
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        if let ChatStreamEvent::Delta { content } = event.unwrap() {
            text.push_str(&content);
        }
    }
    assert_eq!(
        text, response.message.content,
        "greedy streaming matches the blocking reply"
    );

    let token = CancellationToken::new();
    let mut stream = rt
        .chat_stream(PROVIDER_NAME, request(256), Some(token.clone()))
        .await
        .unwrap();
    let mut last = None;
    let mut cancelled_at = None;
    let mut n = 0;
    while let Some(event) = stream.next().await {
        if matches!(event, Ok(ChatStreamEvent::Delta { .. })) {
            n += 1;
            if n == 3 {
                token.cancel();
                cancelled_at = Some(Instant::now());
            }
        }
        last = Some(event);
    }
    if let Some(at) = cancelled_at {
        println!("cancellation observed after {:?}", at.elapsed());
        assert!(
            matches!(last, Some(Err(RuntimeError::Cancelled))),
            "{last:?}"
        );
    }
}
