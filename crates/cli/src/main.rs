use clap::{Args, Parser, Subcommand};
use ml_runtime::ml_runtime_inference::{
    ExecutionPolicy, InferenceOptions, InferenceRequest, InferenceStreamEvent, Input, Output,
};
use ml_runtime::ml_runtime_model::{
    ModelFormat, ModelLocation, ModelPackage, ModelReference, ModelSpec,
};
use ml_runtime::{Runtime, RuntimeCapabilities};
use ml_runtime_coreml_backend::CoreMlBackend;
use ml_runtime_cpu_backend::CpuBackend;
use ml_runtime_cuda_backend::CudaBackend;
use ml_runtime_http_provider::HttpProvider;
use ml_runtime_local_provider::LocalProvider;
use ml_runtime_onnx_backend::OnnxBackend;
use ml_runtime_server_provider::ServerProvider;
use ml_runtime_webgpu_backend::WebgpuBackend;
use serde_json::json;

#[derive(Parser)]
#[command(name = "ml-runtime")]
#[command(about = "Diagnostics for the Rust-native ML runtime scaffold")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Models,
    Status {
        #[arg(long)]
        json: bool,
    },
    Metrics {
        #[arg(long)]
        json: bool,
    },
    Providers,
    Backends,
    Capabilities {
        #[arg(long)]
        json: bool,
    },
    Inspect {
        model: String,
        #[arg(long)]
        json: bool,
    },
    Run(RunArgs),
    Bench {
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[arg(long)]
        json: bool,
    },
    Serve(ServeArgs),
}

#[derive(Args)]
struct RunArgs {
    model: String,
    #[arg(long, default_value = "Hello world")]
    input: String,
    #[arg(long, value_delimiter = ',')]
    tensor: Option<Vec<f32>>,
    #[arg(long, value_delimiter = ',')]
    shape: Option<Vec<usize>>,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    backend: Option<String>,
    #[arg(long)]
    provider: Option<String>,
    #[arg(long)]
    prefer_acceleration: bool,
    #[arg(long)]
    allow_remote_fallback: bool,
    #[arg(long)]
    require_remote: bool,
    #[arg(long)]
    endpoint: Option<String>,
    #[arg(long)]
    execution: Option<String>,
    #[arg(long)]
    stream: bool,
    #[arg(long)]
    verbose: bool,
}

#[derive(Args)]
struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
    #[arg(long, default_value = "examples/models")]
    models: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Models => {
            let runtime = build_runtime(None, None, false, false, None);
            let models = runtime.models();
            if models.list().is_empty() {
                println!("No models loaded");
            } else {
                for model in models.list() {
                    println!("{} ({:?}) via {:?}", model.id, model.format, model.backend);
                }
            }
        }
        Commands::Status { json } => {
            let runtime = build_runtime(None, None, false, false, None);
            let status = runtime.status();
            let snapshot = runtime.snapshot();
            if json {
                println!("{}", serde_json::to_string_pretty(&snapshot)?);
            } else {
                println!("Loaded models: {}", status.loaded_models);
                println!("Running: {}", snapshot.active_inferences);
                println!("Queued: {}", snapshot.queued_inferences);
                println!(
                    "Model cache: {} / {}",
                    status.loaded_models, status.max_models
                );
                for resource in runtime.resources() {
                    println!(
                        "{}\t{}\t{:?}\t{} bytes\tlast_used={}",
                        resource.id,
                        resource.target,
                        resource.state,
                        resource.memory_bytes,
                        resource.last_used
                    );
                }
            }
        }
        Commands::Metrics { json } => {
            let runtime = build_runtime(None, None, false, false, None);
            let metrics = runtime.metrics();
            if json {
                println!("{}", serde_json::to_string_pretty(&metrics)?);
            } else {
                println!("Requests: {}", metrics.total_requests);
                println!("Succeeded: {}", metrics.successful_requests);
                println!("Failed: {}", metrics.failed_requests);
                println!("Cancelled: {}", metrics.cancelled_requests);
                println!("Local: {}", metrics.local_executions);
                println!("Remote: {}", metrics.remote_executions);
                println!("Fallbacks: {}", metrics.fallback_executions);
                println!("Streaming: {}", metrics.streaming_requests);
            }
        }
        Commands::Providers => {
            let runtime = build_runtime(None, None, false, false, None);
            for provider in runtime.providers().list() {
                println!(
                    "{}\tavailable={}\tremote={}\tlocal={}",
                    provider.name, provider.available, provider.remote, provider.local
                );
            }
        }
        Commands::Backends => {
            let runtime = build_runtime(None, None, false, false, None);
            for backend in runtime.backends().list() {
                println!(
                    "{}\tavailable={}\taccelerators={:?}",
                    backend.name, backend.available, backend.accelerators
                );
            }
        }
        Commands::Capabilities { json } => {
            let runtime = build_runtime(None, None, false, false, None);
            render_capabilities(&runtime.capabilities(), json)?;
        }
        Commands::Inspect { model, json } => {
            let package = ModelPackage::open(&model)
                .map_err(|error| format!("open model package: {error}"))?;
            let runtime = build_runtime(None, None, false, false, None);
            let selection = runtime.selection_for(&package.spec())?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "model": package.manifest.id,
                        "format": package.manifest.format,
                        "backend": selection.backend,
                        "execution": selection.provider,
                        "available": selection.backend.as_ref().and_then(|name| runtime
                            .backends()
                            .list()
                            .iter()
                            .find(|backend| &backend.name == name)
                            .map(|backend| backend.available))
                    }))?
                );
            } else {
                println!("Model: {}", package.manifest.id);
                println!("Format: {:?}", package.manifest.format);
                println!(
                    "Backend: {}",
                    selection.backend.as_deref().unwrap_or("none")
                );
                println!("Execution: {}", selection.provider);
            }
        }
        Commands::Run(args) => {
            let runtime = build_runtime(
                args.backend.clone(),
                args.provider.clone(),
                args.allow_remote_fallback,
                args.prefer_acceleration,
                args.endpoint.as_deref(),
            );
            let remote = args.endpoint.is_some();
            let model = if remote {
                let (id, version) = args
                    .model
                    .split_once('@')
                    .map_or((args.model.as_str(), None), |(id, version)| {
                        (id, Some(version.to_owned()))
                    });
                {
                    let mut spec = ModelSpec::new(id, ModelFormat::Unknown, ModelLocation::Memory);
                    spec.version = version;
                    spec
                }
            } else if std::path::Path::new(&args.model).is_dir() {
                ModelPackage::open(&args.model)
                    .map_err(|error| format!("open model package: {error}"))?
                    .spec()
            } else {
                ModelSpec::new(
                    args.model.clone(),
                    ModelFormat::Unknown,
                    ModelLocation::Memory,
                )
            };
            let input = match args.tensor {
                Some(values) => Input::Tensor(ml_runtime::ml_runtime_inference::Tensor {
                    shape: args.shape.unwrap_or_else(|| vec![values.len()]),
                    values,
                }),
                None => Input::Text(args.input),
            };
            let request = InferenceRequest {
                model: if remote {
                    ModelReference::id(model.id, model.version)
                } else {
                    ModelReference::Spec(model)
                },
                input,
                options: InferenceOptions {
                    require_remote: args.require_remote,
                    stream: args.stream,
                    execution: parse_execution_policy(
                        args.execution.as_deref().unwrap_or(if remote {
                            "remote-only"
                        } else {
                            "local-only"
                        }),
                    )?,
                    ..InferenceOptions::default()
                },
            };
            if args.stream {
                let mut stream = runtime.infer_stream(request).await?;
                while let Some(event) = futures_util::StreamExt::next(&mut stream).await {
                    let event = event?;
                    if args.json {
                        println!("{}", serde_json::to_string(&event)?);
                    } else {
                        match event {
                            InferenceStreamEvent::Started(_) => {}
                            InferenceStreamEvent::Output(Output::Text(text)) => println!("{text}"),
                            InferenceStreamEvent::Output(output) => {
                                println!("{}", serde_json::to_string(&output)?)
                            }
                            InferenceStreamEvent::Completed(_) => {}
                        }
                    }
                }
            } else {
                let result = runtime.infer(request).await?;
                if args.json {
                    println!("{}", serde_json::to_string_pretty(&result)?);
                } else {
                    match result.output {
                        Output::Text(text) => println!("{text}"),
                        other => println!("{}", serde_json::to_string_pretty(&other)?),
                    }

                    fn parse_execution_policy(value: &str) -> Result<ExecutionPolicy, String> {
                        match value {
                            "local-only" => Ok(ExecutionPolicy::LocalOnly),
                            "remote-only" => Ok(ExecutionPolicy::RemoteOnly),
                            "prefer-local" => Ok(ExecutionPolicy::PreferLocal),
                            "prefer-remote" => Ok(ExecutionPolicy::PreferRemote),
                            "local-then-remote" => Ok(ExecutionPolicy::LocalThenRemote),
                            "remote-then-local" => Ok(ExecutionPolicy::RemoteThenLocal),
                            _ => Err(format!("unsupported execution policy: {value}")),
                        }
                    }
                    println!(
                        "provider={} backend={} hardware={:?}",
                        result.metadata.provider, result.metadata.backend, result.metadata.hardware
                    );
                    if args.verbose {
                        println!(
                            "model={} target={} routing={:?} fallback={} cache_hit={} batch_size={} queue_wait_ms={:?} model_load_ms={:?} execution_ms={:?} total_ms={:?}",
                            result.metadata.model,
                            result.metadata.execution_target,
                            result.metadata.routing_policy,
                            result.metadata.fallback,
                            result.metadata.cache_hit,
                            result.metadata.batch_size,
                            result.metadata.queue_wait_ms,
                            result.metadata.model_load_ms,
                            result.metadata.execution_ms,
                            result.metadata.latency_ms,
                        );
                    }
                }
            }
        }
        Commands::Bench { iterations, json } => {
            let runtime = build_runtime(None, None, false, false, None);
            let model = ModelSpec::new(
                "benchmark-model",
                ModelFormat::Unknown,
                ModelLocation::Memory,
            );
            let load_start = std::time::Instant::now();
            runtime.load(model.clone()).await?;
            let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
            let start = std::time::Instant::now();
            for _ in 0..iterations.max(1) {
                runtime
                    .infer(InferenceRequest {
                        model: model.clone().into(),
                        input: Input::Text("benchmark input".to_owned()),
                        options: InferenceOptions::default(),
                    })
                    .await?;
            }
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let output = json!({
                "iterations": iterations.max(1),
                "model_load_ms": load_ms,
                "total_inference_ms": elapsed_ms,
                "average_inference_ms": elapsed_ms / iterations.max(1) as f64,
                "metrics": runtime.metrics(),
            });
            if json {
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("iterations={}", iterations.max(1));
                println!("model_load_ms={load_ms:.3}");
                println!(
                    "average_inference_ms={:.3}",
                    elapsed_ms / iterations.max(1) as f64
                );
            }
        }
        Commands::Serve(args) => {
            let runtime = build_runtime(None, None, false, false, None);
            ml_runtime_server_provider::serve(runtime, args.models, args.bind).await?;
        }
    }
    Ok(())
}

fn build_runtime(
    backend: Option<String>,
    provider: Option<String>,
    allow_remote_fallback: bool,
    prefer_acceleration: bool,
    endpoint: Option<&str>,
) -> Runtime {
    let mut builder = Runtime::builder()
        .register_backend(CpuBackend::default())
        .register_backend(CoreMlBackend::default())
        .register_backend(OnnxBackend::default())
        .register_backend(CudaBackend::default())
        .register_backend(WebgpuBackend::default())
        .register_provider(LocalProvider::default())
        .register_provider(HttpProvider::new(
            endpoint.unwrap_or("https://example.invalid"),
        ))
        .register_provider(ServerProvider::new("unix:///tmp/ml-runtime.sock"))
        .allow_remote_fallback(allow_remote_fallback)
        .prefer_acceleration(prefer_acceleration);

    if endpoint.is_some() {
        builder = builder.provider("remote");
    }

    if let Some(backend) = backend {
        builder = builder.backend(backend);
    }

    if let Some(provider) = provider {
        builder = builder.provider(provider);
    }

    builder.build()
}

fn render_capabilities(
    capabilities: &RuntimeCapabilities,
    as_json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(capabilities)?);
        return Ok(());
    }

    println!(
        "Platform: {:?}",
        capabilities
            .platforms
            .first()
            .unwrap_or(&ml_runtime::Platform::Unknown)
    );
    println!("Backends:");
    for backend in &capabilities.backends {
        println!(
            "  {} (available={}, accelerators={:?}, formats={:?})",
            backend.name, backend.available, backend.accelerators, backend.supported_formats
        );
    }
    println!("Providers:");
    for provider in &capabilities.providers {
        println!(
            "  {} (available={}, remote={}, endpoint={:?})",
            provider.name, provider.available, provider.remote, provider.endpoint
        );
    }
    println!("Models:");
    if capabilities.models.is_empty() {
        println!("  none loaded");
    } else {
        for model in &capabilities.models {
            println!("  {} ({:?})", model.id, model.format);
        }
    }

    let summary = json!({
        "invariant": "The application chooses the inference contract. The runtime chooses execution.",
        "distinction": {
            "model": "what is executed",
            "provider": "where inference is obtained",
            "backend": "how computation executes",
            "runtime": "coordinates them"
        }
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
