# rust-ml-runtime

`rust-ml-runtime` is a native, model-neutral runtime for running local machine-learning models from Rust without Python.

```toml
[dependencies]
rust-ml-runtime = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use rust_ml_runtime::{
    InferenceRequest, ModelFormat, ModelLocation, ModelSpec, Output, Runtime,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::new();
    let model = ModelSpec::new("echo", ModelFormat::Unknown, ModelLocation::Memory);
    let result = runtime
        .infer(InferenceRequest::new(model, "local inference"))
        .await?;

    let Output::Text(text) = result.output() else {
        return Err("expected text output".into());
    };
    assert_eq!(text, "local inference");
    assert_eq!(result.metadata().backend, "cpu");
    assert_eq!(result.metadata().execution_target, "local");
    Ok(())
}
```

Models are distributed separately from the crate. Core ML, Laya-specific lifecycle code, the `ml-runtime` CLI, Node bindings, Jev, and application integrations are not part of the public Rust API.

See the [project repository](https://github.com/rkendel1/rust-ml-runtime) for model installation, backend support, CLI artifacts, and Node packages.
