#![cfg(all(target_os = "macos", feature = "coreml"))]

use ml_runtime::{DecisionQuestion, DecisionRequest, DecisionType, Runtime};
use serde_json::Value;
use std::path::PathBuf;

#[test]
fn executes_local_laya_artifact_through_coreml() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let requested = std::env::var_os("LAYA_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/laya"));
    let artifact = if requested.is_absolute() || requested.exists() {
        requested
    } else {
        workspace.join(requested)
    };
    if !artifact.exists() {
        eprintln!(
            "skipping local Core ML integration test; set LAYA_MODEL_PATH or place Laya at {}",
            artifact.display()
        );
        return;
    }
    assert!(artifact.is_dir(), "Laya artifact must be a directory");
    assert!(
        artifact.join("coreml_config.json").is_file(),
        "existing Laya artifact is missing coreml_config.json"
    );
    assert!(
        artifact.join("model.mlpackage/Manifest.json").is_file(),
        "existing Laya artifact is missing model.mlpackage/Manifest.json"
    );
    assert!(
        artifact.join("tokenizer/tokenizer.json").is_file(),
        "existing Laya artifact is missing tokenizer/tokenizer.json"
    );

    let model = Runtime::builder()
        .build()
        .load_decision_model(&artifact)
        .expect("load real local Laya artifact");
    let result = model
        .decide(&DecisionRequest::new(
            Value::String("The customer asks for a refund of a duplicate payment.".to_owned()),
            vec![DecisionQuestion {
                name: "refund".to_owned(),
                instructions: "Does the customer request a refund?".to_owned(),
                kind: DecisionType::Noul {
                    false_description: None,
                    true_description: None,
                },
            }],
        ))
        .expect("execute real Core ML inference");

    assert_eq!(result.backend, "coreml");
    assert_eq!(result.decisions.len(), 1);
    assert_eq!(result.decisions[0].kind, "noul");
    assert_eq!(result.decisions[0].probabilities.len(), 2);
    assert!(result.decisions[0]
        .probabilities
        .values()
        .all(|probability| probability.is_finite() && (0.0..=1.0).contains(probability)));
    assert!(!result.provenance.artifact_sha256.is_empty());
    assert!(result.provenance.artifact_path.is_absolute());
    assert!(!result.provenance.runtime_version.is_empty());
    assert!(!result.execution.latency.is_zero());
    eprintln!("model: {}", result.model.identifier);
    eprintln!("revision: {:?}", result.model.revision);
    eprintln!("backend: {}", result.backend);
    eprintln!("decision: {:#?}", result.decisions[0]);
    eprintln!(
        "latency_ms: {:.3}",
        result.execution.latency.as_secs_f64() * 1_000.0
    );
    eprintln!("artifact: {}", result.provenance.artifact_path.display());
    eprintln!("artifact_sha256: {}", result.provenance.artifact_sha256);
}
