use futures_util::{stream, StreamExt, TryStreamExt};
use ml_runtime_backend::DecisionModel;
use ml_runtime_common::{CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::{
    DecisionDependencyCondition, DecisionExecution, DecisionExecutionDiagnostics,
    DecisionExecutionPlan, DecisionExecutionPolicy, DecisionExecutionStrategy,
    DecisionGraphRequest, DecisionModelCapabilities, DecisionPlanStage, DecisionRequest,
    DecisionResult, DecisionValue, PlannedDecisionResult, TypedDecision,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

#[derive(Clone, Debug, Default)]
pub struct DecisionPlanner;

impl DecisionPlanner {
    pub fn plan(
        &self,
        request: &DecisionGraphRequest,
        capabilities: &DecisionModelCapabilities,
    ) -> RuntimeResult<DecisionExecutionPlan> {
        let levels = validate_graph(request)?;
        let count = request.nodes.len();
        let has_dependencies = request
            .nodes
            .iter()
            .any(|node| !node.dependencies.is_empty());
        let (strategy, reason) = if count == 1 {
            (DecisionExecutionStrategy::Single, "single_node")
        } else if has_dependencies && request.policy.allow_selective_execution {
            (DecisionExecutionStrategy::Selective, "dependency_graph")
        } else if let Some(batch_size) = effective_batch_size(capabilities, &request.policy, count)
        {
            (
                DecisionExecutionStrategy::Batched { batch_size },
                "backend_batching",
            )
        } else if let Some(concurrency) =
            effective_parallelism(capabilities, &request.policy, count)
        {
            (
                DecisionExecutionStrategy::Parallel { concurrency },
                "bounded_parallelism",
            )
        } else {
            (
                DecisionExecutionStrategy::Sequential,
                if capabilities.batch_size == Some(1) {
                    "batch_size_1"
                } else {
                    "serial_capability"
                },
            )
        };
        let stages = levels
            .into_iter()
            .map(|indices| {
                let nodes: Vec<_> = indices
                    .iter()
                    .map(|index| request.nodes[*index].question.name.clone())
                    .collect();
                let size = match strategy {
                    DecisionExecutionStrategy::Batched { batch_size } => batch_size,
                    DecisionExecutionStrategy::Selective => {
                        effective_batch_size(capabilities, &request.policy, nodes.len())
                            .unwrap_or(1)
                    }
                    _ => 1,
                };
                DecisionPlanStage {
                    batches: nodes.chunks(size).map(<[String]>::to_vec).collect(),
                    nodes,
                }
            })
            .collect();
        Ok(DecisionExecutionPlan {
            strategy,
            reason: reason.to_owned(),
            stages,
        })
    }
}

pub async fn execute_graph(
    model: Arc<dyn DecisionModel>,
    request: DecisionGraphRequest,
    cancellation: Option<CancellationToken>,
) -> RuntimeResult<PlannedDecisionResult> {
    if cancellation
        .as_ref()
        .is_some_and(CancellationToken::is_cancelled)
    {
        return Err(RuntimeError::Cancelled);
    }
    let capabilities = model.capabilities();
    if !capabilities.supports_structured_decisions {
        return Err(RuntimeError::CapabilityMismatch {
            reason: "loaded model does not support structured decisions".to_owned(),
        });
    }
    let plan = DecisionPlanner.plan(&request, &capabilities)?;
    let started = Instant::now();
    let mut pending: BTreeSet<usize> = (0..request.nodes.len()).collect();
    let mut skipped = BTreeSet::new();
    let mut resolved: BTreeMap<String, TypedDecision> = BTreeMap::new();
    let mut decisions: Vec<Option<TypedDecision>> = vec![None; request.nodes.len()];
    let mut first_result: Option<DecisionResult> = None;
    let mut input_tokens = 0_u64;
    let mut output_tokens = 0_u64;
    let mut batches = Vec::new();
    let mut max_concurrency = 1_usize;

    while !pending.is_empty() {
        if cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(RuntimeError::Cancelled);
        }
        if request.policy.allow_selective_execution {
            loop {
                let newly_skipped: Vec<_> = pending
                    .iter()
                    .copied()
                    .filter(|index| {
                        request.nodes[*index].dependencies.iter().any(|dependency| {
                            let dependency_index = request
                                .nodes
                                .iter()
                                .position(|node| node.question.name == dependency.decision)
                                .expect("validated dependency");
                            skipped.contains(&dependency_index)
                                || resolved.get(&dependency.decision).is_some_and(|decision| {
                                    !condition_matches(&dependency.condition, &decision.value)
                                })
                        })
                    })
                    .collect();
                if newly_skipped.is_empty() {
                    break;
                }
                for index in newly_skipped {
                    pending.remove(&index);
                    skipped.insert(index);
                }
            }
        }
        let eligible: Vec<_> = pending
            .iter()
            .copied()
            .filter(|index| {
                request.nodes[*index]
                    .dependencies
                    .iter()
                    .all(|dependency| resolved.contains_key(&dependency.decision))
            })
            .collect();
        if eligible.is_empty() {
            return Err(RuntimeError::InvalidRequest {
                reason: "decision graph could not make progress".to_owned(),
            });
        }
        let wave = execute_wave(
            model.clone(),
            &request,
            &capabilities,
            &eligible,
            cancellation.clone(),
        )
        .await?;
        max_concurrency = max_concurrency.max(wave.max_concurrency);
        batches.extend(wave.batches);
        for (index, result) in wave.results {
            input_tokens = input_tokens.saturating_add(result.execution.input_tokens);
            output_tokens = output_tokens.saturating_add(result.execution.output_tokens);
            let decision =
                result
                    .decisions
                    .first()
                    .cloned()
                    .ok_or_else(|| RuntimeError::InvalidOutput {
                        reason: "decision execution returned no typed result".to_owned(),
                    })?;
            resolved.insert(decision.name.clone(), decision.clone());
            decisions[index] = Some(decision);
            pending.remove(&index);
            first_result.get_or_insert(result);
        }
    }

    let mut result = first_result.ok_or_else(|| RuntimeError::InvalidRequest {
        reason: "decision graph executed no nodes".to_owned(),
    })?;
    result.decisions = decisions.into_iter().flatten().collect();
    result.execution = DecisionExecution {
        latency: started.elapsed(),
        input_tokens,
        output_tokens,
    };
    let planning = DecisionExecutionDiagnostics {
        capabilities,
        strategy: plan.strategy,
        reason: plan.reason,
        requested_nodes: request.nodes.len(),
        executed_nodes: result.decisions.len(),
        skipped_nodes: skipped.len(),
        batches,
        max_concurrency,
        cancelled: false,
    };
    Ok(PlannedDecisionResult { result, planning })
}

struct WaveResult {
    results: Vec<(usize, DecisionResult)>,
    batches: Vec<Vec<String>>,
    max_concurrency: usize,
}

async fn execute_wave(
    model: Arc<dyn DecisionModel>,
    request: &DecisionGraphRequest,
    capabilities: &DecisionModelCapabilities,
    eligible: &[usize],
    cancellation: Option<CancellationToken>,
) -> RuntimeResult<WaveResult> {
    if let Some(batch_size) = effective_batch_size(capabilities, &request.policy, eligible.len()) {
        let mut results = Vec::new();
        let mut batches = Vec::new();
        for chunk in eligible.chunks(batch_size) {
            check_cancelled(cancellation.as_ref())?;
            let questions = chunk
                .iter()
                .map(|index| request.nodes[*index].question.clone())
                .collect();
            let result = call_model(
                model.clone(),
                DecisionRequest::new(request.state.clone(), questions),
            )
            .await?;
            if result.decisions.len() != chunk.len() {
                return Err(RuntimeError::InvalidOutput {
                    reason: "batched decision result count does not match its request".to_owned(),
                });
            }
            batches.push(
                chunk
                    .iter()
                    .map(|index| request.nodes[*index].question.name.clone())
                    .collect(),
            );
            results.extend(chunk.iter().copied().zip(split_result(result)));
        }
        return Ok(WaveResult {
            results,
            batches,
            max_concurrency: 1,
        });
    }
    if let Some(concurrency) = effective_parallelism(capabilities, &request.policy, eligible.len())
    {
        let state = request.state.clone();
        let futures = eligible.iter().copied().map(|index| {
            let model = model.clone();
            let question = request.nodes[index].question.clone();
            let state = state.clone();
            let cancellation = cancellation.clone();
            async move {
                check_cancelled(cancellation.as_ref())?;
                let result = call_model(model, DecisionRequest::new(state, vec![question])).await?;
                Ok::<_, RuntimeError>((index, result))
            }
        });
        let results = stream::iter(futures)
            .buffered(concurrency)
            .try_collect::<Vec<_>>()
            .await?;
        return Ok(WaveResult {
            batches: eligible
                .iter()
                .map(|index| vec![request.nodes[*index].question.name.clone()])
                .collect(),
            results,
            max_concurrency: concurrency.min(eligible.len()),
        });
    }
    let mut results = Vec::new();
    for index in eligible {
        check_cancelled(cancellation.as_ref())?;
        let result = call_model(
            model.clone(),
            DecisionRequest::new(
                request.state.clone(),
                vec![request.nodes[*index].question.clone()],
            ),
        )
        .await?;
        results.push((*index, result));
    }
    Ok(WaveResult {
        batches: eligible
            .iter()
            .map(|index| vec![request.nodes[*index].question.name.clone()])
            .collect(),
        results,
        max_concurrency: 1,
    })
}

async fn call_model(
    model: Arc<dyn DecisionModel>,
    request: DecisionRequest,
) -> RuntimeResult<DecisionResult> {
    tokio::task::spawn_blocking(move || model.decide(&request))
        .await
        .map_err(|error| RuntimeError::execution("structured decision task", error.to_string()))?
}

fn split_result(result: DecisionResult) -> Vec<DecisionResult> {
    let DecisionResult {
        model,
        backend,
        decisions,
        execution,
        provenance,
    } = result;
    let count = decisions.len().max(1) as u64;
    decisions
        .into_iter()
        .map(|decision| DecisionResult {
            model: model.clone(),
            backend: backend.clone(),
            decisions: vec![decision],
            execution: DecisionExecution {
                latency: execution.latency,
                input_tokens: execution.input_tokens / count,
                output_tokens: execution.output_tokens / count,
            },
            provenance: provenance.clone(),
        })
        .collect()
}

fn effective_batch_size(
    capabilities: &DecisionModelCapabilities,
    policy: &DecisionExecutionPolicy,
    count: usize,
) -> Option<usize> {
    if count < 2 || !policy.allow_batching || !capabilities.supports_batching {
        return None;
    }
    let size = capabilities
        .max_batch_size
        .unwrap_or(count)
        .min(policy.max_batch_size.unwrap_or(count))
        .min(count);
    (size > 1 && capabilities.batch_size != Some(1)).then_some(size)
}

fn effective_parallelism(
    capabilities: &DecisionModelCapabilities,
    policy: &DecisionExecutionPolicy,
    count: usize,
) -> Option<usize> {
    if count < 2 || !policy.allow_parallel || !capabilities.supports_parallel_execution {
        return None;
    }
    let concurrency = capabilities
        .recommended_parallelism
        .unwrap_or(1)
        .min(policy.max_parallelism.unwrap_or(usize::MAX))
        .min(count);
    (concurrency > 1).then_some(concurrency)
}

fn validate_graph(request: &DecisionGraphRequest) -> RuntimeResult<Vec<Vec<usize>>> {
    if request.nodes.is_empty() {
        return Err(RuntimeError::InvalidRequest {
            reason: "decision graph requires at least one node".to_owned(),
        });
    }
    if request.policy.max_parallelism == Some(0) || request.policy.max_batch_size == Some(0) {
        return Err(RuntimeError::InvalidRequest {
            reason: "decision execution limits must be greater than zero".to_owned(),
        });
    }
    let mut names = BTreeMap::new();
    for (index, node) in request.nodes.iter().enumerate() {
        if node.question.name.is_empty() || names.insert(&node.question.name, index).is_some() {
            return Err(RuntimeError::InvalidRequest {
                reason: "decision graph node names must be nonempty and unique".to_owned(),
            });
        }
    }
    let mut remaining: BTreeSet<_> = (0..request.nodes.len()).collect();
    let mut completed = BTreeSet::new();
    let mut levels = Vec::new();
    while !remaining.is_empty() {
        let level: Vec<_> = remaining
            .iter()
            .copied()
            .filter(|index| {
                request.nodes[*index].dependencies.iter().all(|dependency| {
                    names
                        .get(&dependency.decision)
                        .is_some_and(|dependency_index| completed.contains(dependency_index))
                })
            })
            .collect();
        if level.is_empty() {
            let missing = request
                .nodes
                .iter()
                .flat_map(|node| &node.dependencies)
                .find(|dependency| !names.contains_key(&dependency.decision));
            return Err(RuntimeError::InvalidRequest {
                reason: missing.map_or_else(
                    || "decision graph contains a dependency cycle".to_owned(),
                    |dependency| {
                        format!(
                            "decision graph references missing node {:?}",
                            dependency.decision
                        )
                    },
                ),
            });
        }
        for index in &level {
            remaining.remove(index);
            completed.insert(*index);
        }
        levels.push(level);
    }
    Ok(levels)
}

fn condition_matches(condition: &DecisionDependencyCondition, value: &DecisionValue) -> bool {
    match (condition, value) {
        (DecisionDependencyCondition::Completed, _) => true,
        (DecisionDependencyCondition::ChoiceEquals(expected), DecisionValue::Choice(actual)) => {
            expected == actual
        }
        (DecisionDependencyCondition::NoulEquals(expected), DecisionValue::Noul(actual)) => {
            expected == actual
        }
        (DecisionDependencyCondition::ScoreAtLeast(expected), DecisionValue::Score(actual)) => {
            actual >= expected
        }
        (DecisionDependencyCondition::ScoreAtMost(expected), DecisionValue::Score(actual)) => {
            actual <= expected
        }
        _ => false,
    }
}

fn check_cancelled(cancellation: Option<&CancellationToken>) -> RuntimeResult<()> {
    if cancellation.is_some_and(CancellationToken::is_cancelled) {
        Err(RuntimeError::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_runtime_inference::{
        DecisionDependency, DecisionNode, DecisionOption, DecisionProvenance, DecisionQuestion,
        DecisionType, DeviceKind, ModelArchitecture, ModelDescription, ModelIdentity,
    };
    use serde_json::json;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    fn capabilities(batch: bool, parallel: bool) -> DecisionModelCapabilities {
        DecisionModelCapabilities {
            backend: "fixture".to_owned(),
            device: DeviceKind::Cpu,
            model_architecture: ModelArchitecture::StructuredDecision,
            supports_batching: batch,
            max_batch_size: batch.then_some(4),
            batch_size: (!batch).then_some(1),
            supports_async: false,
            supports_cancellation: false,
            supports_parallel_execution: parallel,
            recommended_parallelism: parallel.then_some(3),
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
        DecisionGraphRequest {
            state: json!({"fixture": true}),
            nodes: (0..count)
                .map(|index| DecisionNode {
                    question: question(&format!("node-{index}")),
                    dependencies: Vec::new(),
                })
                .collect(),
            policy: DecisionExecutionPolicy::default(),
        }
    }

    #[test]
    fn planner_selects_deterministic_strategies_and_batch_chunks() {
        let planner = DecisionPlanner;
        let batch = planner
            .plan(&independent(10), &capabilities(true, true))
            .unwrap();
        assert_eq!(
            batch.strategy,
            DecisionExecutionStrategy::Batched { batch_size: 4 }
        );
        assert_eq!(
            batch.stages[0]
                .batches
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            vec![4, 4, 2]
        );

        let parallel = planner
            .plan(&independent(4), &capabilities(false, true))
            .unwrap();
        assert_eq!(
            parallel.strategy,
            DecisionExecutionStrategy::Parallel { concurrency: 3 }
        );

        let mut serial_request = independent(4);
        serial_request.policy.allow_parallel = false;
        assert_eq!(
            planner
                .plan(&serial_request, &capabilities(false, true))
                .unwrap()
                .strategy,
            DecisionExecutionStrategy::Sequential
        );

        let mut selective = independent(2);
        selective.nodes[1].dependencies.push(DecisionDependency {
            decision: "node-0".to_owned(),
            condition: DecisionDependencyCondition::NoulEquals(true),
        });
        let plan = planner
            .plan(&selective, &capabilities(false, true))
            .unwrap();
        assert_eq!(plan.strategy, DecisionExecutionStrategy::Selective);
        assert_eq!(plan.reason, "dependency_graph");
    }

    #[test]
    fn planner_consumes_real_backend_capability_contracts() {
        let planner = DecisionPlanner;
        let onnx = ml_runtime_onnx_backend::OnnxBackend::laya_decision_capabilities();
        assert_eq!(
            planner.plan(&independent(8), &onnx).unwrap().strategy,
            DecisionExecutionStrategy::Batched { batch_size: 8 }
        );

        #[cfg(feature = "coreml")]
        {
            let coreml = ml_runtime_coreml_backend::CoreMlBackend::laya_decision_capabilities();
            assert_eq!(
                planner.plan(&independent(8), &coreml).unwrap().strategy,
                DecisionExecutionStrategy::Sequential
            );
        }
    }

    struct FixtureModel {
        capabilities: DecisionModelCapabilities,
        calls: AtomicUsize,
        cancel_after_call: Option<CancellationToken>,
    }

    impl FixtureModel {
        fn new(capabilities: DecisionModelCapabilities) -> Self {
            Self {
                capabilities,
                calls: AtomicUsize::new(0),
                cancel_after_call: None,
            }
        }
    }

    impl DecisionModel for FixtureModel {
        fn describe(&self) -> ModelDescription {
            ModelDescription {
                identifier: "fixture".to_owned(),
                revision: None,
                backend: "fixture".to_owned(),
                artifact_path: PathBuf::from("fixture"),
                artifact_sha256: "fixture".to_owned(),
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
                    let (kind, value) = match &question.kind {
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
                    identifier: "fixture".to_owned(),
                    revision: None,
                },
                backend: "fixture".to_owned(),
                decisions,
                execution: DecisionExecution {
                    latency: Duration::ZERO,
                    input_tokens: request.decisions.len() as u64,
                    output_tokens: 0,
                },
                provenance: DecisionProvenance {
                    artifact_path: PathBuf::from("fixture"),
                    artifact_sha256: "fixture".to_owned(),
                    runtime_version: "test".to_owned(),
                },
            })
        }
    }

    #[tokio::test]
    async fn selective_execution_runs_only_matching_children() {
        let root = DecisionQuestion {
            name: "intent".to_owned(),
            instructions: "classify intent".to_owned(),
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
                ],
            },
        };
        let graph = DecisionGraphRequest {
            state: json!({}),
            nodes: vec![
                DecisionNode {
                    question: root,
                    dependencies: vec![],
                },
                DecisionNode {
                    question: question("event-signal"),
                    dependencies: vec![DecisionDependency {
                        decision: "intent".to_owned(),
                        condition: DecisionDependencyCondition::ChoiceEquals("event".to_owned()),
                    }],
                },
                DecisionNode {
                    question: question("expense-signal"),
                    dependencies: vec![DecisionDependency {
                        decision: "intent".to_owned(),
                        condition: DecisionDependencyCondition::ChoiceEquals("expense".to_owned()),
                    }],
                },
            ],
            policy: DecisionExecutionPolicy::default(),
        };
        let model = Arc::new(FixtureModel::new(capabilities(false, false)));
        let result = execute_graph(model.clone(), graph, None).await.unwrap();
        assert_eq!(
            result
                .decisions
                .iter()
                .map(|decision| decision.name.as_str())
                .collect::<Vec<_>>(),
            vec!["intent", "event-signal"]
        );
        let diagnostics = result.planning;
        assert_eq!(diagnostics.strategy, DecisionExecutionStrategy::Selective);
        assert_eq!(diagnostics.executed_nodes, 2);
        assert_eq!(diagnostics.skipped_nodes, 1);
        assert_eq!(model.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn cancellation_stops_before_and_between_nodes() {
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let model = Arc::new(FixtureModel::new(capabilities(false, false)));
        assert_eq!(
            execute_graph(model.clone(), independent(2), Some(cancelled)).await,
            Err(RuntimeError::Cancelled)
        );
        assert_eq!(model.calls.load(Ordering::SeqCst), 0);

        let token = CancellationToken::new();
        let model = Arc::new(FixtureModel {
            capabilities: capabilities(false, false),
            calls: AtomicUsize::new(0),
            cancel_after_call: Some(token.clone()),
        });
        assert_eq!(
            execute_graph(model.clone(), independent(3), Some(token)).await,
            Err(RuntimeError::Cancelled)
        );
        assert_eq!(model.calls.load(Ordering::SeqCst), 1);
    }
}
