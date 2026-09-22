use rust_ml_runtime::{
    InferenceOptions, Input, ModelFormat, ModelLocation, ModelSpec, Output, Runtime, Tensor,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let artifact = std::env::temp_dir().join(format!(
        "rust-ml-runtime-consumer-{}.json",
        std::process::id()
    ));
    std::fs::write(
        &artifact,
        r#"{"weights":[[2.0,-1.0],[0.5,3.0]],"bias":[1.0,-2.0]}"#,
    )?;

    let runtime = Runtime::new();
    let model = ModelSpec::new(
        "external-linear",
        ModelFormat::Unknown,
        ModelLocation::Path(artifact.to_string_lossy().into_owned()),
    );
    let handle = runtime.load(model).await?;
    let result = runtime
        .infer_loaded(
            &handle,
            Input::Tensor(Tensor::new([2], [4.0, 2.0])),
            InferenceOptions::default(),
        )
        .await?;

    let Output::Tensor(output) = result.output() else {
        return Err("expected tensor output".into());
    };
    assert_eq!(output.values, vec![7.0, 6.0]);
    assert_eq!(result.metadata().model, "external-linear");
    assert_eq!(result.metadata().provider, "local");
    assert_eq!(result.metadata().backend, "cpu");
    assert_eq!(result.metadata().execution_target, "local");
    println!(
        "model={} provider={} backend={} target={}",
        result.metadata().model,
        result.metadata().provider,
        result.metadata().backend,
        result.metadata().execution_target
    );

    std::fs::remove_file(artifact)?;
    Ok(())
}
