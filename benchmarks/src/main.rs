use ml_runtime::ml_runtime_inference::{InferenceOptions, InferenceRequest, Input};
use ml_runtime::ml_runtime_model::{ModelFormat, ModelLocation, ModelSpec};
use ml_runtime::Runtime;
use ml_runtime_cpu_backend::CpuBackend;
use serde_json::json;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::builder()
        .register_backend(CpuBackend::default())
        .build();
    let model = ModelSpec::new(
        "benchmark-model",
        ModelFormat::Unknown,
        ModelLocation::Memory,
    );

    let load_start = Instant::now();
    runtime.load(model.clone()).await?;
    let load_latency_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let first_start = Instant::now();
    let first = runtime
        .infer(InferenceRequest {
            model: model.clone().into(),
            input: Input::Text("benchmark input".to_owned()),
            options: InferenceOptions::default(),
        })
        .await?;
    let first_latency_ms = first_start.elapsed().as_secs_f64() * 1000.0;

    let steady_start = Instant::now();
    for _ in 0..5 {
        runtime
            .infer(InferenceRequest {
                model: model.clone().into(),
                input: Input::Text("benchmark input".to_owned()),
                options: InferenceOptions::default(),
            })
            .await?;
    }
    let steady_state_latency_ms = steady_start.elapsed().as_secs_f64() * 1000.0 / 5.0;

    let output = json!({
        "model": model.id,
        "provider": first.metadata.provider,
        "backend": first.metadata.backend,
        "hardware": first.metadata.hardware,
        "runtime_version": env!("CARGO_PKG_VERSION"),
        "measurements": {
            "model_load_latency_ms": load_latency_ms,
            "first_inference_latency_ms": first_latency_ms,
            "steady_state_latency_ms": steady_state_latency_ms,
            "throughput_requests_per_second": if steady_state_latency_ms > 0.0 { 1000.0 / steady_state_latency_ms } else { 0.0 },
            "memory_usage": "not yet instrumented",
            "streaming_latency": "available via infer_stream path",
            "cpu_utilization": "external profiler required",
            "accelerator_utilization": "backend-specific"
        }
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
