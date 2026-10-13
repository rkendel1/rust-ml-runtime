//! Reproducible measurement of the embedded provider.
//!   cargo run --release --example bench -- <model.gguf> [max_tokens=128] [context=2048]
//! Measures load time, time to first token (includes prompt processing), decode
//! speed, cancellation latency, and peak RSS (Linux). Nothing is downloaded.
use futures_util::StreamExt;
use ml_runtime_llama_provider::{LlamaConfig, LlamaProvider, PROVIDER_NAME};
use rust_ml_runtime::{
    CancellationToken, ChatMessage, ChatRequest, ChatStreamEvent, GenerationOptions, Runtime,
};
use std::time::Instant;

fn peak_rss_mib() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: f64 = status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kb / 1024.0)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: bench <model.gguf> [max_tokens] [context]");
    let max_tokens: u32 = args.next().map_or(128, |v| v.parse().unwrap());
    let context: u32 = args.next().map_or(2048, |v| v.parse().unwrap());

    let mut config = LlamaConfig::new(&path, "bench");
    config.context_tokens = context;
    config.max_output_tokens = max_tokens.min(context / 2);
    let load_started = Instant::now();
    let provider = LlamaProvider::new(config).expect("load");
    let load = load_started.elapsed();
    let info = provider.model_info().clone();
    let rt = Runtime::builder()
        .register_generation_provider(provider)
        .build();
    println!(
        "model        : {} ({} params, {:.1} MiB on disk, arch {})",
        info.reported_name,
        info.parameters,
        info.file_bytes as f64 / 1048576.0,
        info.architecture
    );
    println!("backend      : {:?}", rt.generation_providers()[0].notes);
    println!("load time    : {load:?}");
    println!("context      : {context} tokens, max_tokens {max_tokens}, greedy");

    let request = |max: u32| {
        ChatRequest::new(
            "bench",
            vec![
                ChatMessage::system("You are a helpful assistant."),
                ChatMessage::user("Write a short paragraph about the history of computing."),
            ],
        )
        .with_options(GenerationOptions {
            max_tokens: Some(max),
            ..Default::default()
        })
    };

    // Warm-up (page in weights), then measure.
    let _ = rt.chat(PROVIDER_NAME, request(8), None).await.unwrap();
    for run in 1..=3 {
        let started = Instant::now();
        let mut stream = rt
            .chat_stream(PROVIDER_NAME, request(max_tokens), None)
            .await
            .unwrap();
        let (mut first, mut last, mut deltas) = (None, started, 0u64);
        let mut usage = None;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                ChatStreamEvent::Delta { .. } => {
                    deltas += 1;
                    first.get_or_insert_with(|| started.elapsed());
                    last = Instant::now();
                }
                ChatStreamEvent::Completed(s) => usage = s.usage,
                _ => {}
            }
        }
        let usage = usage.unwrap();
        let (prompt, completion) = (
            usage.prompt_tokens.unwrap(),
            usage.completion_tokens.unwrap(),
        );
        let first = first.unwrap();
        let decode = last.duration_since(started).as_secs_f64() - first.as_secs_f64();
        println!(
            "run {run}: prompt {prompt} tok, generated {completion} tok ({deltas} deltas) | TTFT {first:.0?} (includes prompt processing) | decode {:.1} tok/s (excludes first token)",
            (completion.saturating_sub(1)) as f64 / decode.max(1e-9)
        );
    }

    // Cancellation latency: cancel after 8 deltas of a long generation; measure
    // until the stream reports Cancelled, then until the slot accepts new work.
    for run in 1..=3 {
        let token = CancellationToken::new();
        let mut stream = rt
            .chat_stream(PROVIDER_NAME, request(max_tokens), Some(token.clone()))
            .await
            .unwrap();
        let (mut n, mut at) = (0, None);
        let mut end = None;
        while let Some(event) = stream.next().await {
            if matches!(event, Ok(ChatStreamEvent::Delta { .. })) {
                n += 1;
                if n == 8 {
                    at = Some(Instant::now());
                    token.cancel();
                }
            }
            end = Some(event);
        }
        let to_error = at.map(|t| t.elapsed());
        let t = Instant::now();
        rt.chat(PROVIDER_NAME, request(1), None).await.unwrap();
        println!("cancel {run}: stream ended {:?} after cancel ({:?}); next request admitted+done in {:?}", to_error.unwrap(), end.map(|e| e.is_err()), t.elapsed());
    }
    match peak_rss_mib() {
        Some(mib) => {
            println!("peak RSS     : {mib:.0} MiB (whole process, incl. mmapped weights touched)")
        }
        None => println!("peak RSS     : not measured on this OS"),
    }
}
