use ml_runtime::{DecisionOption, DecisionQuestion, DecisionRequest, DecisionType, Runtime};
use serde_json::Value;
use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = PathBuf::from("models/laya");
    let mut input = "The customer asks for a refund of a duplicate payment.".to_owned();
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => model = PathBuf::from(args.next().ok_or("--model needs a path")?),
            "--input" => input = args.next().ok_or("--input needs text")?,
            unknown => return Err(format!("unknown argument {unknown:?}").into()),
        }
    }

    let runtime = Runtime::builder().build();
    let loaded = runtime.load_decision_model(&model)?;
    let result = loaded.decide(&DecisionRequest::new(
        Value::String(input),
        vec![DecisionQuestion {
            name: "department".to_owned(),
            instructions: "Which department should handle this request?".to_owned(),
            kind: DecisionType::Choice {
                options: vec![
                    DecisionOption {
                        label: "billing".to_owned(),
                        description: Some(Value::String(
                            "Payments, invoices, refunds, and duplicate charges.".to_owned(),
                        )),
                    },
                    DecisionOption {
                        label: "technical".to_owned(),
                        description: Some(Value::String(
                            "Broken features, errors, and troubleshooting.".to_owned(),
                        )),
                    },
                    DecisionOption {
                        label: "sales".to_owned(),
                        description: Some(Value::String(
                            "Pricing, upgrades, and new purchases.".to_owned(),
                        )),
                    },
                ],
            },
        }],
    ))?;

    println!("model: {}", result.model.identifier);
    println!("backend: {}", result.backend);
    for decision in result.decisions {
        println!("decision: {} ({})", decision.name, decision.kind);
        println!("result: {:?}", decision.value);
        println!("probabilities: {:?}", decision.probabilities);
    }
    println!(
        "latency_ms: {:.3}",
        result.execution.latency.as_secs_f64() * 1_000.0
    );
    println!("artifact: {}", result.provenance.artifact_path.display());
    println!("sha256: {}", result.provenance.artifact_sha256);
    Ok(())
}
