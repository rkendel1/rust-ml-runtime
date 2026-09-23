#![cfg(target_os = "linux")]

use rust_ml_runtime::{
    DecisionExecutionPolicy, DecisionExecutionStrategy, DecisionModel, DecisionOption,
    DecisionQuestion, DecisionRequest, DecisionType, DecisionValue, Runtime,
};
use serde_json::json;
use std::{path::PathBuf, sync::Arc};

#[tokio::test]
async fn executes_real_laya_onnx_typed_decisions_through_planner() {
    let Some(artifact) = std::env::var_os("LAYA_ONNX_MODEL_PATH").map(PathBuf::from) else {
        eprintln!("skipping real ONNX integration test; set LAYA_ONNX_MODEL_PATH");
        return;
    };
    let runtime = Runtime::new();
    let model: Arc<dyn DecisionModel> = Arc::from(
        runtime
            .load_decision_model(&artifact)
            .expect("load real Laya ONNX bundle"),
    );
    let capabilities = runtime.decision_capabilities(model.as_ref());
    assert!(capabilities.supports_batching);
    assert_eq!(capabilities.max_batch_size, Some(16));
    let result = runtime
        .decide_async(
            model,
            DecisionRequest::new(
                json!({"message": "Please refund the duplicate invoice today."}),
                vec![
                    DecisionQuestion {
                        name: "department".to_owned(),
                        instructions: "Which department should handle this?".to_owned(),
                        kind: DecisionType::Choice {
                            options: vec![
                                DecisionOption {
                                    label: "billing".to_owned(),
                                    description: Some(json!("refunds and invoices")),
                                },
                                DecisionOption {
                                    label: "technical".to_owned(),
                                    description: Some(json!("product failures")),
                                },
                            ],
                        },
                    },
                    DecisionQuestion {
                        name: "urgency".to_owned(),
                        instructions: "How urgent is the request?".to_owned(),
                        kind: DecisionType::Score {
                            levels: vec![json!("routine"), json!("urgent")],
                        },
                    },
                    DecisionQuestion {
                        name: "refund".to_owned(),
                        instructions: "Is a refund requested?".to_owned(),
                        kind: DecisionType::Noul {
                            false_description: None,
                            true_description: None,
                        },
                    },
                ],
            ),
            DecisionExecutionPolicy::default(),
            None,
        )
        .await
        .expect("run real Laya ONNX inference");

    assert_eq!(result.backend, "onnx");
    assert!(matches!(
        result.decisions[0].value,
        DecisionValue::Choice(_)
    ));
    assert!(matches!(result.decisions[1].value, DecisionValue::Score(_)));
    assert!(matches!(result.decisions[2].value, DecisionValue::Noul(_)));
    let planning = &result.planning;
    assert_eq!(
        planning.strategy,
        DecisionExecutionStrategy::Batched { batch_size: 3 }
    );
    assert_eq!(planning.executed_nodes, 3);
    for decision in &result.decisions {
        assert!(decision.confidence.is_finite());
        assert!(decision.action_probability.is_finite());
        assert!((decision.probabilities.values().sum::<f64>() - 1.0).abs() < 1e-5);
    }
}
