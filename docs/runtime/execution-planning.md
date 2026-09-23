# Structured-decision execution planning

Rust owns structured-decision semantics and execution. Applications describe
typed questions, dependencies, and resource constraints; the runtime reads
facts declared by the loaded model/backend and deterministically selects an
execution strategy.

```text
DecisionGraphRequest + DecisionExecutionPolicy
                    + DecisionModelCapabilities
                              |
                              v
                       DecisionPlanner
          single / sequential / parallel / batched / selective
                              |
                              v
                 typed DecisionResult + planning diagnostics
```

Model architecture, backend, device, and execution strategy are separate
facts. Laya is a structured-decision architecture. It can execute through
Core ML on Apple's CPU+GPU compute units or through ONNX Runtime on CPU, and
the application uses the same request and result contract for either.

## Public Rust API

Load a model with `Runtime::load_decision_model`, then inspect
`Runtime::decision_capabilities`. Use `Runtime::explain_decision` for a
side-effect-free plan, `Runtime::decide_async` for independent decisions, or
`Runtime::execute_decision_graph` for dependencies and selective execution.
The existing synchronous `DecisionModel::decide` API remains supported.

Each `DecisionNode` contains a normal `DecisionQuestion` and zero or more
dependencies. Conditions can require completion, a choice value, a noul
value, or a score threshold. A false condition skips that node and all nodes
which depend on it. Names are unique, references must exist, and cycles are
rejected before inference starts.

`DecisionExecutionPolicy` bounds the plan with `max_parallelism` and
`max_batch_size`, and can disable parallel, batched, or selective execution.
It never asserts that a backend supports an operation. `latency_target` is a
forward-compatible application constraint; the deterministic v1 planner does
not invent timing heuristics from it.

## Capability matrix

| Loaded decision model | Architecture | Backend | Device | Batch | Native async | Native interruption | Parallel calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Laya Core ML | structured decision | Core ML | Apple CPU+GPU | no; size 1 | no | no | no |
| Laya ONNX | structured decision | ONNX Runtime | CPU | yes; max 16 | no | no | no |
| External `DecisionModel` | declared by implementer | declared by implementer | declared by implementer | conservative default: no | no | no | no |

The async runtime moves synchronous native calls off the caller's executor
thread. `supports_async` describes the native backend, not whether the public
runtime offers an async wrapper. Generic CPU tensor inference remains a
separate architecture and uses the ordinary inference capability contract.

## Strategy rules

The v1 precedence is deterministic:

1. One node uses `Single`.
2. A dependency graph uses `Selective` when policy permits it.
3. Independent nodes use `Batched` when the model declares a usable batch.
4. Otherwise they use bounded `Parallel` when declared safe.
5. Otherwise they use `Sequential`.

Batch chunks never exceed the lower of the backend and policy limits.
Parallelism never exceeds the lower of the recommended and policy limits.
No generic planner branch checks a backend name.

For an intent plus conditional signals, a batch-size-1 model evaluates intent
first and only the matching signals second. For 16 independent questions on
Laya ONNX, the runtime can select one batch of 16. These are capability-driven
plans, not universal performance claims.

## Cancellation and diagnostics

Pass a `CancellationToken` to either async method. Cancellation before a call
prevents it; cancellation between graph stages prevents remaining nodes. A
backend reports `supports_cancellation: true` only if it can interrupt an
already-running native call. Core ML and Laya ONNX currently report false.

Successful graph results include `planning` diagnostics: the capability
snapshot, selected strategy and reason, requested/executed/skipped counts,
actual batches, and maximum concurrency. Diagnostics report observed execution
and do not estimate speedups.

## Node API

`LocalML.decide` and native `decideJson()` remain synchronous compatibility
APIs. Prefer `await local.decideAsync(...)` or
`await local.executeGraph(...)`; synchronous native work runs away from the
Node event loop. `local.capabilities(model)` and
`local.explainDecision(request)` expose typed, backend-neutral facts and plans.
Use `DecisionCancellation` for cooperative cancellation.

The Node layer serializes the public Rust contracts. It does not plan work,
decode model output, or reproduce Choice/Score/Noul semantics.

