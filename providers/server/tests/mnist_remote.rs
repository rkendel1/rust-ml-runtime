use ml_runtime::{
    ExecutionPolicy, InferenceOptions, InferenceRequest, Input, ModelReference, Output, Runtime,
    Tensor,
};
use ml_runtime_http_provider::HttpProvider;
use ml_runtime_provider::Provider;
use std::path::{Path, PathBuf};
use std::time::Duration;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn request(policy: ExecutionPolicy) -> InferenceRequest {
    let values =
        std::fs::read_to_string(workspace_root().join("examples/inputs/mnist-8-seven.csv"))
            .unwrap()
            .trim()
            .split(',')
            .map(|value| value.parse().unwrap())
            .collect();
    InferenceRequest {
        model: ModelReference::id("mnist-8", Some("8".to_owned())),
        input: Input::Tensor(Tensor {
            shape: vec![1, 1, 28, 28],
            values,
        }),
        options: InferenceOptions {
            execution: policy,
            ..InferenceOptions::default()
        },
    }
}

#[tokio::test]
async fn serves_pretrained_mnist_over_http_with_truthful_metadata() {
    let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = probe.local_addr().unwrap();
    drop(probe);

    let server_runtime = Runtime::builder().build();
    let models = workspace_root().join("examples/models");
    let server = tokio::spawn(async move {
        let _ =
            ml_runtime_server_provider::serve(server_runtime, models, address.to_string()).await;
    });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let endpoint = format!("http://{address}");
    let provider = HttpProvider::new(&endpoint);
    let server_result = provider
        .infer(request(ExecutionPolicy::PreferLocal), None)
        .await
        .unwrap();
    assert_eq!(server_result.metadata.provider, "local");
    assert_eq!(server_result.metadata.backend, "onnx");

    let client = Runtime::builder().remote(endpoint).build();
    let client_result = client
        .infer(request(ExecutionPolicy::RemoteOnly))
        .await
        .unwrap();
    assert_eq!(client_result.metadata.provider, "remote");
    assert_eq!(client_result.metadata.backend, "onnx");
    let Output::Tensor(output) = client_result.output else {
        panic!("remote MNIST inference must return a tensor");
    };
    let predicted = output
        .values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    assert_eq!(predicted, 7);

    let hybrid = Runtime::builder()
        .catalog(ml_runtime::FilesystemModelCatalog::new(
            workspace_root().join("examples/models"),
        ))
        .remote(format!("http://{address}"))
        .build();
    let local = hybrid
        .infer(request(ExecutionPolicy::PreferLocal))
        .await
        .unwrap();
    assert_eq!(local.metadata.provider, "local");
    assert!(!local.metadata.fallback);
    let remote = hybrid
        .infer(request(ExecutionPolicy::RemoteThenLocal))
        .await
        .unwrap();
    assert_eq!(remote.metadata.provider, "remote");
    assert_eq!(remote.metadata.backend, "onnx");

    let mut invalid = request(ExecutionPolicy::LocalThenRemote);
    let Input::Tensor(tensor) = &mut invalid.input else {
        unreachable!();
    };
    tensor.values.pop();
    assert!(matches!(
        hybrid.infer(invalid).await,
        Err(ml_runtime::RuntimeError::UnsupportedInput { .. })
    ));

    server.abort();
}
