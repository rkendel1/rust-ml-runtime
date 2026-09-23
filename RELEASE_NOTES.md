# rust-ml-runtime 0.2.3

A native, architecture-aware local ML runtime for applications without Python.

This release moves structured-decision optimization behind the Rust runtime
boundary. Applications describe typed questions, dependencies, and limits;
the runtime chooses a deterministic selective, sequential, bounded-parallel,
or batched plan from factual model/backend capabilities. The same API operates
over Laya Core ML on macOS and Laya ONNX on Linux.

## Included

- Explicit model architecture, backend, device, batch, async, cancellation,
  and parallel-execution capabilities
- Deterministic execution planning with policy bounds and plan explanation
- Generic dependency graphs with conditional selective execution
- Capability-gated batching and bounded parallel execution
- Cooperative cancellation before and between decision nodes, without falsely
  claiming native interruption
- Async Rust and Node structured-decision APIs that preserve synchronous APIs
- Factual execution diagnostics with actual batches, concurrency, and skipped work
- Runtime benchmarks for single, serial, selective, parallel, batch, and cancellation paths
- Typed `@rust-ml-runtime/node` capability and planning APIs
- Rust library distribution through `rust-ml-runtime = "0.2.3"` on crates.io
- Explicit model installation, verification, diagnosis, and removal
- SHA-256 verification for runtime and model artifacts
- Local Core ML execution with READY/offline lifecycle semantics
- Local ONNX Laya execution on Linux x86_64 and ARM64
- Typed decisions, probabilities, latency, and provenance
- In-process Node/TypeScript integration through `@rust-ml-runtime/node`
- Laya model support on Linux and compatible macOS/ARM64 hosts

Models remain separately distributed and are not embedded in the Rust crate.

## Example

```rust
let capabilities = runtime.decision_capabilities(model.as_ref());
let plan = runtime.explain_decision(model.as_ref(), &graph)?;
let result = runtime.execute_decision_graph(model, graph, None).await?;
```

## Current backends and model

Laya uses ONNX Runtime on supported Linux hosts and CoreML on macOS/ARM64.
These are implementations behind the model-neutral runtime boundary, not its
definition. The runtime does not make application policy or authorization
decisions. Jev remains a downstream consumer and owns its separate
compatibility validation against this public release.
