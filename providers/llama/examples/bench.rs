//! Reproducible measurement of the embedded provider.
//!   cargo run --release --example bench -- <model.gguf> [max_tokens=128] [context=2048] [gpu_layers=0]
//! Set ML_RUNTIME_LLAMA_LOGS=1 to see llama.cpp's own device/offload log.
//! Wrap in `/usr/bin/time -l` (macOS) for peak RSS / peak memory footprint.
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
    let gpu_layers: u32 = args.next().map_or(0, |v| v.parse().unwrap());

    let mut config = LlamaConfig::new(&path, "bench");
    config.context_tokens = context;
    config.max_output_tokens = max_tokens.min(context / 2);
    config.gpu_layers = gpu_layers;
    let load_started = Instant::now();
    let provider = LlamaProvider::new(config.clone()).expect("load");
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
    println!("devices      : {:?}", info.devices);
    println!(
        "load time    : {load:?} total (llama.cpp loader/mmap {:?}, sha256 {:?}; first load in process incl. backend + device init)",
        info.load_time, info.verify_time
    );
    println!(
        "context      : {context} tokens, max_tokens {max_tokens}, greedy, gpu_layers {gpu_layers}"
    );

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

    // First request: weights are paged in / uploaded lazily, so this is the
    // real cost of the cold start that `load time` (mmap) does not show.
    let first = Instant::now();
    let _ = rt.chat(PROVIDER_NAME, request(8), None).await.unwrap();
    println!(
        "first request: {:?} (8 tokens; weight page-in / first-use cost)",
        first.elapsed()
    );

    // Prompt-processing speed: a long prompt, one generated token.
    let long_text =
        "The quick brown fox jumps over the lazy dog. ".repeat((context as usize / 14).min(100));
    let pp_request = ChatRequest::new("bench", vec![ChatMessage::user(long_text)]).with_options(
        GenerationOptions {
            max_tokens: Some(1),
            ..Default::default()
        },
    );
    for run in 1..=3 {
        let started = Instant::now();
        let response = rt
            .chat(PROVIDER_NAME, pp_request.clone(), None)
            .await
            .unwrap();
        let prompt = response.usage.unwrap().prompt_tokens.unwrap();
        let took = started.elapsed();
        println!(
            "prompt run {run}: {prompt} prompt tokens in {took:.0?} = {:.0} tok/s (includes context allocation)",
            prompt as f64 / took.as_secs_f64()
        );
    }
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
    // Warm reload: same process, page cache and device already initialized.
    drop(rt);
    let reload = Instant::now();
    let again = LlamaProvider::new(config).expect("reload");
    println!(
        "warm reload  : {:?} total (loader {:?}, sha256 {:?})",
        reload.elapsed(),
        again.model_info().load_time,
        again.model_info().verify_time
    );
    match peak_rss_mib() {
        Some(mib) => {
            println!("peak RSS     : {mib:.0} MiB (whole process, incl. mmapped weights touched)")
        }
        None => println!("peak RSS     : not measured on this OS"),
    }
}
