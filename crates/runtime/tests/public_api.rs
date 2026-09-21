use futures_util::StreamExt;
use ml_runtime::{
    CancellationToken, ExecutionPolicy, FilesystemModelCatalog, InferenceOptions, InferenceRequest,
    InferenceStreamEvent, Input, ModelReference, Output, Runtime, RuntimeError, Tensor,
};
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn values() -> Vec<f32> {
    include_str!("../../../examples/inputs/mnist-8-seven.csv")
        .trim()
        .split(',')
        .map(|value| value.parse().unwrap())
        .collect()
}

fn request() -> InferenceRequest {
    InferenceRequest::new(
        ModelReference::versioned("mnist-8", "8"),
        Input::Tensor(Tensor::new([1, 1, 28, 28], values())),
    )
}

fn runtime() -> Runtime {
    Runtime::builder()
        .catalog(FilesystemModelCatalog::new(
            workspace_root().join("examples/models"),
        ))
        .build()
}

#[tokio::test]
async fn canonical_public_api_executes_real_model() {
    let result = runtime().infer(request()).await.unwrap();
    let Output::Tensor(output) = result.output() else {
        panic!("MNIST must return tensor output");
    };
    let digit = output
        .values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    assert_eq!(digit, 7);
    assert_eq!(result.metadata().model, "mnist-8");
    assert_eq!(result.metadata().model_version.as_deref(), Some("8"));
    assert_eq!(result.metadata().provider, "local");
    assert_eq!(result.metadata().backend, "onnx");
    assert_eq!(result.metadata().execution_target, "local");
}

#[tokio::test]
async fn public_errors_and_cancellation_are_structured() {
    let invalid = request().with_options(InferenceOptions {
        require_local: true,
        require_remote: true,
        ..InferenceOptions::default()
    });
    assert!(matches!(
        runtime().infer(invalid).await,
        Err(RuntimeError::InvalidRequest { .. })
    ));
    assert!(matches!(
        runtime()
            .infer(InferenceRequest::new(
                ModelReference::versioned("missing", "1"),
                "input",
            ))
            .await,
        Err(RuntimeError::ModelNotFound { .. })
    ));
    assert!(matches!(
        runtime()
            .infer(InferenceRequest::new(
                ModelReference::versioned("mnist-8", "8"),
                "not a tensor",
            ))
            .await,
        Err(RuntimeError::UnsupportedInput { .. })
    ));

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert!(matches!(
        runtime()
            .infer_with_cancellation(request(), cancellation)
            .await,
        Err(RuntimeError::Cancelled)
    ));

    let source = workspace_root().join("examples/models/mnist-8");
    let root = std::env::temp_dir().join(format!("ml-runtime-api-{}", std::process::id()));
    let package = root.join("mnist-8");
    std::fs::create_dir_all(package.join("artifacts")).unwrap();
    std::fs::copy(source.join("manifest.json"), package.join("manifest.json")).unwrap();
    std::fs::copy(
        source.join("artifacts/model.onnx"),
        package.join("artifacts/model.onnx"),
    )
    .unwrap();
    let mut bytes = std::fs::read(package.join("artifacts/model.onnx")).unwrap();
    bytes[0] ^= 0xff;
    std::fs::write(package.join("artifacts/model.onnx"), bytes).unwrap();
    let corrupt = Runtime::builder()
        .catalog(FilesystemModelCatalog::new(&root))
        .build();
    assert!(matches!(
        corrupt.infer(request()).await,
        Err(RuntimeError::ModelIntegrity { .. })
    ));
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn public_stream_batch_and_loaded_model_paths_are_compatible() {
    let runtime = runtime();
    let mut stream = runtime
        .infer_stream(request().with_options(InferenceOptions {
            stream: true,
            ..InferenceOptions::default()
        }))
        .await
        .unwrap();
    assert!(matches!(
        stream.next().await.unwrap().unwrap(),
        InferenceStreamEvent::Started(_)
    ));
    assert!(matches!(
        stream.next().await.unwrap().unwrap(),
        InferenceStreamEvent::Output(Output::Tensor(_))
    ));
    assert!(matches!(
        stream.next().await.unwrap().unwrap(),
        InferenceStreamEvent::Completed(_)
    ));
    assert!(stream.next().await.is_none());

    let results = runtime
        .infer_batch(vec![request(), request()])
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|result| result.metadata.batch_size == 2));

    let descriptor = runtime
        .resolve_model(&ModelReference::versioned("mnist-8", "8"))
        .await
        .unwrap();
    let handle = runtime.load(descriptor.spec()).await.unwrap();
    let loaded = runtime
        .infer_loaded(
            &handle,
            Input::Tensor(Tensor::new([1, 1, 28, 28], values())),
            InferenceOptions {
                execution: ExecutionPolicy::LocalOnly,
                ..InferenceOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(loaded.metadata.cache_hit);
    assert_eq!(handle.backend(), "onnx");
}
