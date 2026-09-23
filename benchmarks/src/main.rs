use ml_runtime::ml_runtime_inference::{InferenceOptions, InferenceRequest, Input};
use ml_runtime::ml_runtime_model::{ModelFormat, ModelLocation, ModelSpec};
use ml_runtime::{
    CancellationToken, DecisionDependency, DecisionDependencyCondition, DecisionExecution,
    DecisionExecutionPolicy, DecisionGraphRequest, DecisionModel, DecisionModelCapabilities,
    DecisionNode, DecisionOption, DecisionProvenance, DecisionQuestion, DecisionRequest,
    DecisionResult, DecisionType, DecisionValue, DeviceKind, ModelArchitecture, ModelDescription,
    ModelIdentity, Runtime, RuntimeResult, TypedDecision,
};
use ml_runtime_cpu_backend::CpuBackend;
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

struct BenchmarkDecisionModel {
    capabilities: DecisionModelCapabilities,
    calls: AtomicUsize,
    cancel_after_call: Option<CancellationToken>,
}

impl BenchmarkDecisionModel {
    fn new(capabilities: DecisionModelCapabilities) -> Self {
        Self {
            capabilities,
            calls: AtomicUsize::new(0),
            cancel_after_call: None,
        }
    }
}

impl DecisionModel for BenchmarkDecisionModel {
    fn describe(&self) -> ModelDescription {
        ModelDescription {
            identifier: "decision-benchmark".to_owned(),
            revision: None,
            backend: "benchmark-fixture".to_owned(),
            artifact_path: PathBuf::from("benchmark-fixture"),
            artifact_sha256: "benchmark-fixture".to_owned(),
            decision_types: vec!["choice".to_owned(), "noul".to_owned()],
        }
    }

    fn capabilities(&self) -> DecisionModelCapabilities {
        self.capabilities.clone()
    }

    fn decide(&self, request: &DecisionRequest) -> RuntimeResult<DecisionResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let decisions = request
            .decisions
            .iter()
            .map(|question| {
                let (kind, value) = match question.kind {
                    DecisionType::Choice { .. } => {
                        ("choice", DecisionValue::Choice("event".to_owned()))
                    }
                    DecisionType::Score { .. } => ("score", DecisionValue::Score(1.0)),
                    DecisionType::Noul { .. } => ("noul", DecisionValue::Noul(true)),
                };
                TypedDecision {
                    name: question.name.clone(),
                    kind: kind.to_owned(),
                    value,
                    probabilities: BTreeMap::new(),
                    confidence: 1.0,
                    action_probability: 1.0,
                }
            })
            .collect();
        if let Some(token) = &self.cancel_after_call {
            token.cancel();
        }
        Ok(DecisionResult {
            model: ModelIdentity {
                identifier: "decision-benchmark".to_owned(),
                revision: None,
            },
            backend: "benchmark-fixture".to_owned(),
            decisions,
            execution: DecisionExecution {
                latency: Duration::ZERO,
                input_tokens: request.decisions.len() as u64,
                output_tokens: request.decisions.len() as u64,
            },
            provenance: DecisionProvenance {
                artifact_path: PathBuf::from("benchmark-fixture"),
                artifact_sha256: "benchmark-fixture".to_owned(),
                runtime_version: env!("CARGO_PKG_VERSION").to_owned(),
            },
        })
    }
}

fn decision_capabilities(batch: bool, parallel: bool) -> DecisionModelCapabilities {
    DecisionModelCapabilities {
        backend: "benchmark-fixture".to_owned(),
        device: DeviceKind::Cpu,
        model_architecture: ModelArchitecture::StructuredDecision,
        supports_batching: batch,
        max_batch_size: batch.then_some(16),
        batch_size: (!batch).then_some(1),
        supports_async: false,
        supports_cancellation: false,
        supports_parallel_execution: parallel,
        recommended_parallelism: parallel.then_some(4),
        supports_structured_decisions: true,
    }
}

fn question(name: &str) -> DecisionQuestion {
    DecisionQuestion {
        name: name.to_owned(),
        instructions: format!("evaluate {name}"),
        kind: DecisionType::Noul {
            false_description: None,
            true_description: None,
        },
    }
}

fn independent(count: usize) -> DecisionGraphRequest {
    DecisionGraphRequest::independent(
        DecisionRequest::new(
            json!({"benchmark": true}),
            (0..count)
                .map(|index| question(&format!("node-{index}")))
                .collect(),
        ),
        DecisionExecutionPolicy::default(),
    )
}

async fn benchmark_graph(
    runtime: &Runtime,
    label: &str,
    model: Arc<BenchmarkDecisionModel>,
    request: DecisionGraphRequest,
) -> RuntimeResult<serde_json::Value> {
    let started = Instant::now();
    let result = runtime
        .execute_decision_graph(model.clone(), request, None)
        .await?;
    Ok(json!({
        "case": label,
        "elapsed_ms": started.elapsed().as_secs_f64() * 1_000.0,
        "native_calls": model.calls.load(Ordering::SeqCst),
        "execution": result.planning,
    }))
}

async fn structured_benchmarks(runtime: &Runtime) -> RuntimeResult<Vec<serde_json::Value>> {
    let mut measurements = Vec::new();
    measurements.push(
        benchmark_graph(
            runtime,
            "single",
            Arc::new(BenchmarkDecisionModel::new(decision_capabilities(
                false, false,
            ))),
            independent(1),
        )
        .await?,
    );
    measurements.push(
        benchmark_graph(
            runtime,
            "serial_14",
            Arc::new(BenchmarkDecisionModel::new(decision_capabilities(
                false, false,
            ))),
            independent(14),
        )
        .await?,
    );

    let root = DecisionQuestion {
        name: "intent".to_owned(),
        instructions: "select intent".to_owned(),
        kind: DecisionType::Choice {
            options: vec![
                DecisionOption {
                    label: "event".to_owned(),
                    description: None,
                },
                DecisionOption {
                    label: "expense".to_owned(),
                    description: None,
                },
                DecisionOption {
                    label: "timer".to_owned(),
                    description: None,
                },
            ],
        },
    };
    let selective = DecisionGraphRequest {
        state: json!({"benchmark": true}),
        nodes: std::iter::once(DecisionNode {
            question: root,
            dependencies: Vec::new(),
        })
        .chain(
            ["event", "expense", "timer"]
                .into_iter()
                .map(|intent| DecisionNode {
                    question: question(&format!("{intent}-signal")),
                    dependencies: vec![DecisionDependency {
                        decision: "intent".to_owned(),
                        condition: DecisionDependencyCondition::ChoiceEquals(intent.to_owned()),
                    }],
                }),
        )
        .collect(),
        policy: DecisionExecutionPolicy::default(),
    };
    measurements.push(
        benchmark_graph(
            runtime,
            "selective_root_plus_3",
            Arc::new(BenchmarkDecisionModel::new(decision_capabilities(
                false, false,
            ))),
            selective,
        )
        .await?,
    );
    measurements.push(
        benchmark_graph(
            runtime,
            "parallel_4",
            Arc::new(BenchmarkDecisionModel::new(decision_capabilities(
                false, true,
            ))),
            independent(4),
        )
        .await?,
    );
    for count in [4, 8, 16] {
        measurements.push(
            benchmark_graph(
                runtime,
                &format!("batched_{count}"),
                Arc::new(BenchmarkDecisionModel::new(decision_capabilities(
                    true, false,
                ))),
                independent(count),
            )
            .await?,
        );
    }

    let token = CancellationToken::new();
    let model = Arc::new(BenchmarkDecisionModel {
        capabilities: decision_capabilities(false, false),
        calls: AtomicUsize::new(0),
        cancel_after_call: Some(token.clone()),
    });
    let started = Instant::now();
    let cancelled = runtime
        .execute_decision_graph(model.clone(), independent(4), Some(token))
        .await;
    measurements.push(json!({
        "case": "cancellation",
        "elapsed_ms": started.elapsed().as_secs_f64() * 1_000.0,
        "request_started": model.calls.load(Ordering::SeqCst) > 0,
        "cancelled": cancelled.is_err(),
        "native_calls": model.calls.load(Ordering::SeqCst),
        "work_remaining": 4_usize.saturating_sub(model.calls.load(Ordering::SeqCst)),
    }));
    Ok(measurements)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::builder().register_backend(CpuBackend).build();
    let model = ModelSpec::new(
        "benchmark-model",
        ModelFormat::Unknown,
        ModelLocation::Memory,
    );

    let load_start = Instant::now();
    runtime.load(model.clone()).await?;
    let load_latency_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let first_start = Instant::now();
    let first = runtime
        .infer(InferenceRequest {
            model: model.clone().into(),
            input: Input::Text("benchmark input".to_owned()),
            options: InferenceOptions::default(),
        })
        .await?;
    let first_latency_ms = first_start.elapsed().as_secs_f64() * 1000.0;

    let steady_start = Instant::now();
    for _ in 0..5 {
        runtime
            .infer(InferenceRequest {
                model: model.clone().into(),
                input: Input::Text("benchmark input".to_owned()),
                options: InferenceOptions::default(),
            })
            .await?;
    }
    let steady_state_latency_ms = steady_start.elapsed().as_secs_f64() * 1000.0 / 5.0;

    let output = json!({
        "model": model.id,
        "provider": first.metadata.provider,
        "backend": first.metadata.backend,
        "hardware": first.metadata.hardware,
        "runtime_version": env!("CARGO_PKG_VERSION"),
        "measurements": {
            "model_load_latency_ms": load_latency_ms,
            "first_inference_latency_ms": first_latency_ms,
            "steady_state_latency_ms": steady_state_latency_ms,
            "throughput_requests_per_second": if steady_state_latency_ms > 0.0 { 1000.0 / steady_state_latency_ms } else { 0.0 },
            "memory_usage": "not yet instrumented",
            "streaming_latency": "available via infer_stream path",
            "cpu_utilization": "external profiler required",
            "accelerator_utilization": "backend-specific"
        },
        "structured_decisions": structured_benchmarks(&runtime).await?,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
