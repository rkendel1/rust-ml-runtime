# Public developer API

For dependency-aware Choice/Score/Noul requests, capability discovery,
execution policies, async execution, and planning diagnostics, see
[Structured-decision execution planning](runtime/execution-planning.md).

`rust-ml-runtime` is the supported application-facing Rust crate. Its core API is
the explicit, application-owned `Runtime`, plus model references, inference
requests/results, execution policies, and structured errors re-exported at the
crate root. The workspace is currently `0.2.x`; these APIs are supported but
still pre-1.0 and follow Cargo semantic-versioning conventions.

## Quickstart

```rust
use rust_ml_runtime::{
    FilesystemModelCatalog, InferenceRequest, Input, ModelReference, Runtime,
    Tensor,
};

# async fn example(values: Vec<f32>) -> rust_ml_runtime::RuntimeResult<()> {
let runtime = Runtime::builder()
    .catalog(FilesystemModelCatalog::new("examples/models"))
    .build();
let result = runtime
    .infer(InferenceRequest::new(
        ModelReference::versioned("mnist-8", "8"),
        Input::Tensor(Tensor::new([1, 1, 28, 28], values)),
    ))
    .await?;
println!("{:?} via {}/{}", result.output(), result.metadata().provider, result.metadata().backend);
# Ok(())
# }
```

The default runtime includes the built-in CPU and ONNX backends. Applications
choose a model, input, and policy; they do not register those built-ins or
construct backend sessions. `register_backend` and `register_provider` remain
experimental extension points for integrations outside the built-in surface.

## Core API and model lifecycle

- `ModelSpec`: explicit model definition, format, and artifact location.
- `ModelReference`: request-time catalog identity or explicit `ModelSpec`.
- `ModelId`: validated stable name and version used by catalogs/installers.
- `ModelDescriptor`: catalog discovery metadata; it does not load the model.
- `RuntimeModelHandle`: runtime-owned loaded resource returned by `load`.

Use `infer` for normal request-level execution. It resolves catalog identities,
loads on demand, and reuses the bounded runtime cache. Use `load` followed by
`infer_loaded` for an explicit repeated-execution lifecycle. Neither path
requires applications to manipulate caches or backend resources.

`InferenceRequest::new(model, input)` uses default options. `with_options`
applies an explicit `InferenceOptions`. `InferenceResult` exposes `output` and
`metadata` directly and through accessors; applications interpret outputs.

## Errors

`RuntimeError` is structured and matchable. Stable application-relevant
variants distinguish invalid requests, missing/unavailable/integrity-failed
models, unsupported models and inputs, unavailable backends/providers,
timeouts, cancellation, execution failures, transport failures, and protocol
failures. Variant fields retain model, provider, backend, or operation context
where known. Display strings are for people; applications should match variants.

## Execution policies and metadata

`LocalOnly`, `RemoteOnly`, `PreferLocal`, `PreferRemote`, `LocalThenRemote`,
and `RemoteThenLocal` are explicit. Invalid input and request-contract errors
never trigger fallback.

Metadata semantics:

- `model` / `model_version`: requested resolved identity.
- `provider`: location or transport selected by this runtime (`local`, `remote`).
- `backend`: computation implementation reported by the execution target.
- `execution_target`: target observed by this runtime; on a client this may be
  `remote` while `backend` is `onnx` on the server.
- `routing_policy`, `fallback`, `fallback_from`, `fallback_reason`: routing.
- `cache_hit`, `batch_size`: lifecycle/batch state without exposing cache data.
- streaming fields: requested/supported state, event count, completion state.
- timing fields: queue, model load, execution, and end-to-end latency where
  measured.

## Streaming

Set `options.stream = true`, call `infer_stream`, and consume:

```rust
use futures_util::StreamExt;
use ml_runtime::InferenceStreamEvent;

# async fn consume(mut stream: ml_runtime::InferenceStream) -> ml_runtime::RuntimeResult<()> {
while let Some(event) = stream.next().await {
    match event? {
        InferenceStreamEvent::Started(metadata) => { /* exactly once, first */ }
        InferenceStreamEvent::Output(output) => { /* zero or more */ }
        InferenceStreamEvent::Completed(metadata) => { /* exactly once on success */ }
    }
}
# Ok(())
# }
```

Errors terminate the stream and successful completion is not emitted after an
error or cancellation. `infer_stream_with_cancellation` propagates the existing
`CancellationToken` through runtime, provider, and backend. Buffering is bounded
by `max_stream_buffer`, so slow consumers apply backpressure. A non-incremental
backend emits one complete output through compatibility mode.

## Batching

`infer_batch(requests)` preserves input/result order and reports batch metadata.
It rejects configured/backend batch-limit violations. A batching backend may
execute the group directly; otherwise the runtime executes requests
individually. The current return type is `RuntimeResult<Vec<InferenceResult>>`,
so any request failure fails the call rather than returning per-item errors.
Caller-controlled batch cancellation is not currently a public capability.

## Remote execution

```rust
let runtime = ml_runtime::Runtime::builder()
    .remote("http://127.0.0.1:8080")
    .build();
```

The application still calls `infer`; the HTTP provider and `/v1` protocol stay
behind the runtime boundary. Hybrid applications can combine `.catalog(...)`
and `.remote(...)` and select an explicit execution policy.

## TypeScript

The public native package is [`@rust-ml-runtime/node`](node.md). The repository
also contains a private, unpublished `@ml-runtime/core` development binding. It
is a thin subprocess projection that requires an installed `ml-runtime`
executable and is not part of the v0.1.0 distributable surface. It accepts
typed model references and text/tensor inputs and returns typed runtime output
and metadata. `RuntimeClientError`
separates process and JSON protocol bridge failures without recreating Rust
inference semantics. Streaming is supported. The current subprocess bridge
does not expose Rust's batch or cancellation-token APIs; it does not emulate
them in TypeScript.

## Public, experimental, and internal

Supported application surface: runtime construction, catalogs/model
references, requests/options/results, execution metadata/policies, structured
errors, streaming, batching, loaded handles, capabilities, and observability
snapshots.

Experimental extension surface: direct backend/provider registration and the
individual backend/provider crates.

Internal implementation details: backend sessions, provider transports,
protocol envelopes/routes, cache keys, resource scheduling, and server
handlers. Their visibility inside the workspace does not make them stable
application APIs.
