use clap::{Args, Parser, Subcommand};
use ml_runtime::ml_runtime_inference::{
    ExecutionPolicy, InferenceOptions, InferenceRequest, InferenceStreamEvent, Input, Output,
};
use ml_runtime::ml_runtime_model::{
    FileModelSource, FilesystemModelCatalog, InstallMode, InstallResult, ModelCatalog,
    ModelFetchRequest, ModelFormat, ModelId, ModelInstaller, ModelLocation, ModelPackage,
    ModelReference, ModelSourceReference, ModelSpec,
};
use ml_runtime::{Runtime, RuntimeCapabilities, VERSION};
use ml_runtime_coreml_backend::CoreMlBackend;
use ml_runtime_cpu_backend::CpuBackend;
use ml_runtime_cuda_backend::CudaBackend;
use ml_runtime_http_provider::HttpProvider;
use ml_runtime_local_provider::LocalProvider;
use ml_runtime_onnx_backend::OnnxBackend;
use ml_runtime_server_provider::ServerProvider;
use ml_runtime_webgpu_backend::WebgpuBackend;
use serde_json::json;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "ml-runtime", version, about = "Rust-native ML runtime")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Models {
        #[command(subcommand)]
        command: Option<ModelCommands>,
        #[arg(long, default_value = "examples/models")]
        models: String,
        #[arg(long)]
        json: bool,
    },
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
        #[command(subcommand)]
        command: Option<CapabilityCommands>,
        #[arg(long)]
        json: bool,
    },
    Doctor {
        #[arg(long, default_value = "examples/models")]
        models: String,
        #[arg(long)]
        json: bool,
    },
    Inspect {
        model: String,
        #[arg(long, default_value = "examples/models")]
        models: String,
        #[arg(long)]
        json: bool,
    },
    Run(RunArgs),
    Bench(BenchArgs),
    Serve(ServeArgs),
}

#[derive(Subcommand)]
enum CapabilityCommands {
    Inspect {
        capability: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum ModelCommands {
    Install {
        source: String,
        #[arg(long)]
        replace: bool,
        #[arg(long)]
        json: bool,
    },
    Verify {
        model: String,
        #[arg(long, default_value = "examples/models")]
        models: String,
        #[arg(long)]
        json: bool,
    },
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
    #[arg(long, default_value = "examples/models")]
    models: String,
}

#[derive(Args)]
struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
    #[arg(long, default_value = "examples/models")]
    models: String,
}

#[derive(Args)]
struct BenchArgs {
    model: Option<String>,
    #[arg(long, default_value_t = 5)]
    iterations: usize,
    #[arg(long, value_delimiter = ',')]
    tensor: Option<Vec<f32>>,
    #[arg(long, value_delimiter = ',')]
    shape: Option<Vec<usize>>,
    #[arg(long)]
    backend: Option<String>,
    #[arg(long, default_value = "examples/models")]
    models: String,
    #[arg(long)]
    json: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Models {
            command: None,
            models,
            json,
        } => {
            let catalog = FilesystemModelCatalog::new(models);
            let models = catalog.list().await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&models)?);
            } else if models.is_empty() {
                println!("No models loaded");
            } else {
                println!("MODEL\tVERSION\tFORMAT");
                for model in models {
                    println!(
                        "{}\t{}\t{:?}",
                        model.id.name, model.id.version, model.format
                    );
                }
            }
        }
        Commands::Models {
            command:
                Some(ModelCommands::Install {
                    source,
                    replace,
                    json,
                }),
            models,
            ..
        } => {
            let source_path = std::path::PathBuf::from(&source);
            let package = ModelPackage::open(&source_path)
                .map_err(|error| format!("open model package: {error}"))?;
            let version = package
                .manifest
                .model_version
                .clone()
                .or_else(|| package.manifest.version.clone())
                .ok_or("model package has no version")?;
            let request = ModelFetchRequest {
                model: ModelId::new(package.manifest.id.clone(), version)?,
                source: ModelSourceReference::Path(source_path),
            };
            let catalog = Arc::new(FilesystemModelCatalog::new(&models));
            let result = ModelInstaller::new(&models)
                .with_catalog(Arc::clone(&catalog))
                .install(
                    &FileModelSource::new(),
                    &request,
                    if replace {
                        InstallMode::Replace
                    } else {
                        InstallMode::KeepExisting
                    },
                    None,
                )
                .await?;
            let (descriptor, status) = match result {
                InstallResult::Installed(descriptor) => (descriptor, "installed"),
                InstallResult::AlreadyInstalled(descriptor) => (descriptor, "already_installed"),
            };
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "model": descriptor.id.name,
                        "version": descriptor.id.version,
                        "format": descriptor.format,
                        "installed_path": descriptor.package_path,
                        "status": status,
                    }))?
                );
            } else {
                println!(
                    "{}@{}: {}",
                    descriptor.id.name, descriptor.id.version, status
                );
                println!("installed path: {}", descriptor.package_path.display());
            }
        }
        Commands::Models {
            command:
                Some(ModelCommands::Verify {
                    model,
                    models,
                    json,
                }),
            ..
        } => {
            let catalog = FilesystemModelCatalog::new(models);
            let id = ModelId::parse(&model)?;
            let valid = catalog.validate(&id).await.is_ok();
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "model": id.name,
                        "version": id.version,
                        "installed": valid,
                        "valid": valid,
                        "loaded": false,
                    }))?
                );
            } else if valid {
                println!("{}: valid", id);
            } else {
                return Err(format!("model is not valid: {id}").into());
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
        Commands::Capabilities {
            command: Some(CapabilityCommands::Inspect { capability, json }),
            ..
        } => {
            let runtime = build_runtime(None, None, false, false, None);
            let capability = runtime
                .capability(&capability)
                .ok_or_else(|| format!("unknown capability: {capability}"))?;
            let resolution = runtime.resolve_capability(&ml_runtime::ExecutionRequest::new(
                &capability.id,
                serde_json::Value::Null,
            ))?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "capability": capability,
                        "resolution": resolution,
                    }))?
                );
            } else {
                println!(
                    "{}\tversion={}\tavailable={}",
                    capability.id, capability.version, capability.availability
                );
                println!("{}", capability.description);
                println!("Selected provider: {}", resolution.provider);
                println!("Resolution: {}", resolution.reason);
            }
        }
        Commands::Capabilities {
            json,
            command: None,
        } => {
            let runtime = build_runtime(None, None, false, false, None);
            render_capabilities(&runtime.capabilities(), json)?;
        }
        Commands::Doctor { models, json } => {
            let runtime = build_runtime(None, None, false, false, None);
            let capabilities = runtime.capabilities();
            let catalog_exists = std::path::Path::new(&models).is_dir();
            let cpu_available = capabilities
                .backends
                .iter()
                .any(|backend| backend.name == "cpu" && backend.available);
            let checks = vec![
                ("executable", true, "ml-runtime is running"),
                ("configuration", true, "default configuration is valid"),
                (
                    "CPU backend",
                    cpu_available,
                    if cpu_available {
                        "available"
                    } else {
                        "not available on this platform"
                    },
                ),
                (
                    "model catalog",
                    catalog_exists,
                    if catalog_exists {
                        "catalog directory found"
                    } else {
                        "catalog directory not found (create it to install models)"
                    },
                ),
                (
                    "environment",
                    !capabilities.environment.platform.is_empty(),
                    "discovered",
                ),
            ];
            let healthy = checks.iter().all(|(_, ok, _)| *ok);
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "runtime_version": VERSION,
                        "checks": checks.iter().map(|(name, ok, message)| json!({
                            "name": name,
                            "ok": ok,
                            "message": message,
                        })).collect::<Vec<_>>(),
                        "optional_backends": capabilities.backends,
                        "healthy": healthy,
                    }))?
                );
            } else {
                println!("ml-runtime {}", VERSION);
                for (name, ok, message) in &checks {
                    println!("  {} {}: {}", if *ok { "✓" } else { "-" }, name, message);
                }
                println!("Optional backends");
                for backend in capabilities.backends {
                    if backend.name != "cpu" {
                        println!(
                            "  {} {}: {}",
                            if backend.available { "✓" } else { "-" },
                            backend.name,
                            if backend.available {
                                "available".to_owned()
                            } else {
                                backend.notes.join("; ")
                            }
                        );
                    }
                }
            }
            if !healthy {
                return Err("doctor found installation problems".into());
            }
        }
        Commands::Inspect {
            model,
            models,
            json,
        } => {
            let catalog = FilesystemModelCatalog::new(models);
            let descriptor = if std::path::Path::new(&model).is_dir() {
                let package = ModelPackage::open(&model)
                    .map_err(|error| format!("open model package: {error}"))?;
                let version = package
                    .manifest
                    .model_version
                    .clone()
                    .or_else(|| package.manifest.version.clone())
                    .ok_or("model package has no version")?;
                catalog
                    .resolve(&ModelReference::id(package.manifest.id, Some(version)))
                    .await?
            } else {
                let (id, version) = model
                    .split_once('@')
                    .map_or((model.as_str(), None), |(id, version)| {
                        (id, Some(version.to_owned()))
                    });
                catalog.resolve(&ModelReference::id(id, version)).await?
            };
            let package = ModelPackage::open(&descriptor.package_path)
                .map_err(|error| format!("open model package: {error}"))?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "identity": descriptor.id,
                        "format": package.manifest.format,
                        "package_path": descriptor.package_path,
                        "artifact": package.artifact,
                        "inputs": package.manifest.inputs,
                        "outputs": package.manifest.outputs,
                        "validation": "valid"
                    }))?
                );
            } else {
                println!("Identity: {}", descriptor.id);
                println!("Format: {:?}", package.manifest.format);
                println!("Package: {}", descriptor.package_path.display());
                println!("Artifact: {}", package.artifact.display());
                println!("Validation: valid");
            }
        }
        Commands::Run(args) => {
            let mut runtime = build_runtime(
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
                ModelReference::id(id, version)
            } else if std::path::Path::new(&args.model).is_dir() {
                ModelReference::Spec(
                    ModelPackage::open(&args.model)
                        .map_err(|error| format!("open model package: {error}"))?
                        .spec(),
                )
            } else {
                runtime.set_catalog(FilesystemModelCatalog::new(&args.models));
                let (id, version) = args
                    .model
                    .split_once('@')
                    .map_or((args.model.as_str(), None), |(id, version)| {
                        (id, Some(version.to_owned()))
                    });
                ModelReference::id(id, version)
            };
            let input = match args.tensor {
                Some(values) => Input::Tensor(ml_runtime::ml_runtime_inference::Tensor {
                    shape: args.shape.unwrap_or_else(|| vec![values.len()]),
                    values,
                }),
                None => Input::Text(args.input),
            };
            let request = InferenceRequest {
                model,
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
        Commands::Bench(args) => {
            let mut runtime = build_runtime(args.backend.clone(), None, false, false, None);
            let (model, input) = if let Some(model) = args.model.as_deref() {
                let values = args
                    .tensor
                    .clone()
                    .ok_or("benchmarking a catalog model requires --tensor")?;
                let shape = args.shape.clone().unwrap_or_else(|| vec![values.len()]);
                let (id, version) = model
                    .split_once('@')
                    .map_or((model, None), |(id, version)| {
                        (id, Some(version.to_owned()))
                    });
                runtime.set_catalog(FilesystemModelCatalog::new(&args.models));
                (
                    ModelReference::id(id, version),
                    Input::Tensor(ml_runtime::ml_runtime_inference::Tensor { shape, values }),
                )
            } else {
                (
                    ModelSpec::new(
                        "benchmark-model",
                        ModelFormat::Unknown,
                        ModelLocation::Memory,
                    )
                    .into(),
                    Input::Text("benchmark input".to_owned()),
                )
            };
            let request = InferenceRequest {
                model,
                input,
                options: InferenceOptions::default(),
            };
            let iterations = args.iterations.max(1);
            let first_start = std::time::Instant::now();
            let first = runtime.infer(request.clone()).await?;
            let first_latency_ms = first_start.elapsed().as_secs_f64() * 1000.0;
            let steady_start = std::time::Instant::now();
            for _ in 1..iterations {
                runtime.infer(request.clone()).await?;
            }
            let steady_elapsed_ms = steady_start.elapsed().as_secs_f64() * 1000.0;
            let total_ms = first_latency_ms + steady_elapsed_ms;
            let output = json!({
                "model": first.metadata.model,
                "version": first.metadata.model_version,
                "provider": first.metadata.provider,
                "backend": first.metadata.backend,
                "execution_target": first.metadata.execution_target,
                "iterations": iterations,
                "model_load_ms": first.metadata.model_load_ms,
                "first_inference_ms": first_latency_ms,
                "total_inference_ms": total_ms,
                "average_inference_ms": total_ms / iterations as f64,
                "throughput_requests_per_second": iterations as f64 * 1000.0 / total_ms,
                "metrics": runtime.metrics(),
            });
            if args.json {
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!(
                    "model={}@{}",
                    first.metadata.model,
                    first
                        .metadata
                        .model_version
                        .as_deref()
                        .unwrap_or("unversioned")
                );
                println!(
                    "provider={} backend={} target={}",
                    first.metadata.provider,
                    first.metadata.backend,
                    first.metadata.execution_target
                );
                println!("iterations={iterations}");
                println!("first_inference_ms={first_latency_ms:.3}");
                println!("average_inference_ms={:.3}", total_ms / iterations as f64);
                println!(
                    "throughput_requests_per_second={:.3}",
                    iterations as f64 * 1000.0 / total_ms
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
    println!(
        "Environment: {} {} ({})",
        capabilities.environment.operating_system,
        capabilities.environment.architecture,
        capabilities.environment.cpu
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
    println!("Capabilities:");
    for capability in &capabilities.capabilities {
        println!(
            "  {}@{} (available={}, target={})",
            capability.id,
            capability.version,
            capability.availability,
            capability.execution_targets.join(",")
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
