use clap::{Args, Parser, Subcommand};
use ml_runtime::ml_runtime_inference::{InferenceOptions, InferenceRequest, Input, Output};
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
    Providers,
    Backends,
    Capabilities {
        #[arg(long)]
        json: bool,
    },
    Run(RunArgs),
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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Models => {
            let runtime = build_runtime(None, None, false, false);
            let models = runtime.models();
            if models.list().is_empty() {
                println!("No models loaded");
            } else {
                for model in models.list() {
                    println!("{} ({:?}) via {:?}", model.id, model.format, model.backend);
                }
            }
        }
        Commands::Providers => {
            let runtime = build_runtime(None, None, false, false);
            for provider in runtime.providers().list() {
                println!(
                    "{}\tavailable={}\tremote={}\tlocal={}",
                    provider.name, provider.available, provider.remote, provider.local
                );
            }
        }
        Commands::Backends => {
            let runtime = build_runtime(None, None, false, false);
            for backend in runtime.backends().list() {
                println!(
                    "{}\tavailable={}\taccelerators={:?}",
                    backend.name, backend.available, backend.accelerators
                );
            }
        }
        Commands::Capabilities { json } => {
            let runtime = build_runtime(None, None, false, false);
            render_capabilities(&runtime.capabilities(), json)?;
        }
        Commands::Run(args) => {
            let runtime = build_runtime(
                args.backend.clone(),
                args.provider.clone(),
                args.allow_remote_fallback,
                args.prefer_acceleration,
            );
            let model = if std::path::Path::new(&args.model).is_dir() {
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
                model: ModelReference::Spec(model),
                input,
                options: InferenceOptions {
                    require_remote: args.require_remote,
                    ..InferenceOptions::default()
                },
            };
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
            }
        }
    }
    Ok(())
}

fn build_runtime(
    backend: Option<String>,
    provider: Option<String>,
    allow_remote_fallback: bool,
    prefer_acceleration: bool,
) -> Runtime {
    let mut builder = Runtime::builder()
        .register_backend(CpuBackend::default())
        .register_backend(CoreMlBackend::default())
        .register_backend(OnnxBackend::default())
        .register_backend(CudaBackend::default())
        .register_backend(WebgpuBackend::default())
        .register_provider(LocalProvider::default())
        .register_provider(HttpProvider::new("https://example.invalid/infer"))
        .register_provider(ServerProvider::new("unix:///tmp/ml-runtime.sock"))
        .allow_remote_fallback(allow_remote_fallback)
        .prefer_acceleration(prefer_acceleration);

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
