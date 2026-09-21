use ml_runtime::ml_runtime_model::{ModelCatalog, ModelPackage};
use ml_runtime::{
    FilesystemModelCatalog, InferenceOptions, InferenceRequest, Input, ModelFormat, ModelId,
    ModelReference, Output, Runtime, RuntimeError, Tensor,
};
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn input_values() -> Vec<f32> {
    std::fs::read_to_string(workspace_root().join("examples/inputs/mnist-8-seven.csv"))
        .expect("read checked-in MNIST input")
        .trim()
        .split(',')
        .map(|value| value.parse().expect("fixture contains f32 values"))
        .collect()
}

fn request(shape: Vec<usize>, values: Vec<f32>) -> InferenceRequest {
    InferenceRequest {
        model: ModelReference::id("mnist-8", Some("8".to_owned())),
        input: Input::Tensor(Tensor { shape, values }),
        options: InferenceOptions::default(),
    }
}

fn runtime() -> Runtime {
    Runtime::builder()
        .catalog(FilesystemModelCatalog::new(
            workspace_root().join("examples/models"),
        ))
        .build()
}

#[tokio::test]
async fn discovers_verifies_and_executes_pretrained_mnist() {
    let catalog = FilesystemModelCatalog::new(workspace_root().join("examples/models"));
    let id = ModelId::new("mnist-8", "8").unwrap();
    catalog.validate(&id).await.unwrap();
    let descriptor = catalog
        .resolve(&ModelReference::id("mnist-8", Some("8".to_owned())))
        .await
        .unwrap();
    assert_eq!(descriptor.format, ModelFormat::Onnx);
    assert_eq!(descriptor.manifest.artifact.size_bytes, Some(26_454));
    assert_eq!(
        descriptor.manifest.artifact.sha256.as_deref(),
        Some("2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf")
    );
    assert_eq!(descriptor.manifest.inputs[0].name, "Input3");
    assert_eq!(
        descriptor.manifest.inputs[0].shape,
        Some(vec![1, 1, 28, 28])
    );
    assert_eq!(descriptor.manifest.outputs[0].name, "Plus214_Output_0");
    assert_eq!(descriptor.manifest.outputs[0].shape, Some(vec![1, 10]));

    let result = runtime()
        .infer(request(vec![1, 1, 28, 28], input_values()))
        .await
        .unwrap();
    let Output::Tensor(output) = result.output else {
        panic!("MNIST must produce tensor output");
    };
    let expected = [
        -11.431_837,
        10.915_633,
        14.908_884,
        8.848_751,
        -10.990_991,
        -17.247_47,
        -19.191_664,
        25.551_865,
        -6.415_498_3,
        -3.321_460_5,
    ];
    assert_eq!(output.shape, vec![1, 10]);
    for (actual, expected) in output.values.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-3);
    }
    let predicted = output
        .values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    assert_eq!(predicted, 7);
    assert_eq!(result.metadata.provider, "local");
    assert_eq!(result.metadata.backend, "onnx");
}

#[tokio::test]
async fn reports_real_model_failure_paths() {
    let catalog = FilesystemModelCatalog::new(workspace_root().join("examples/models"));
    let wrong_id = catalog
        .resolve(&ModelReference::id("not-mnist", Some("8".to_owned())))
        .await
        .unwrap_err();
    assert!(wrong_id.to_string().contains("model not found"));

    let values = input_values();
    let invalid_length = runtime()
        .infer(request(vec![1, 1, 28, 28], values[..783].to_vec()))
        .await
        .unwrap_err();
    assert!(invalid_length
        .to_string()
        .contains("tensor shape does not match tensor values"));

    let invalid_shape = runtime()
        .infer(request(vec![1, 784], values.clone()))
        .await
        .unwrap_err();
    assert!(invalid_shape.to_string().contains("execute ONNX model"));

    let unavailable = Runtime::builder()
        .backend("missing-backend")
        .catalog(catalog)
        .build()
        .infer(request(vec![1, 1, 28, 28], values))
        .await
        .unwrap_err();
    assert!(matches!(
        unavailable,
        RuntimeError::BackendUnavailable { .. }
    ));
}

#[test]
fn rejects_missing_and_corrupted_mnist_artifacts() {
    let source = workspace_root().join("examples/models/mnist-8");
    let root = std::env::temp_dir().join(format!("ml-runtime-mnist-{}", std::process::id()));
    let missing = root.join("missing");
    std::fs::create_dir_all(&missing).unwrap();
    std::fs::copy(source.join("manifest.json"), missing.join("manifest.json")).unwrap();
    let error = ModelPackage::open(&missing).unwrap_err();
    assert!(error.contains("resolve model artifact"));

    let corrupt = root.join("corrupt");
    std::fs::create_dir_all(corrupt.join("artifacts")).unwrap();
    std::fs::copy(source.join("manifest.json"), corrupt.join("manifest.json")).unwrap();
    std::fs::copy(
        source.join("artifacts/model.onnx"),
        corrupt.join("artifacts/model.onnx"),
    )
    .unwrap();
    let mut bytes = std::fs::read(corrupt.join("artifacts/model.onnx")).unwrap();
    bytes[0] ^= 0xff;
    std::fs::write(corrupt.join("artifacts/model.onnx"), bytes).unwrap();
    let error = ModelPackage::open(&corrupt)
        .unwrap()
        .validate_artifact()
        .unwrap_err();
    assert!(error.to_string().contains("sha256 mismatch"));
    let _ = std::fs::remove_dir_all(root);
}
