# Architecture

The runtime is intentionally boring at the center:

`load -> select -> execute -> return (or stream) -> observe`

## Primary invariant

The application chooses the inference contract. The runtime chooses execution. Models, providers, and hardware remain replaceable.

## Core distinction

- **Model** = what is executed
- **Provider** = where inference is obtained
- **Backend** = how computation executes
- **Runtime** = coordinates them

## Current scaffold

- Rust runtime authority lives in `crates/runtime`
- Shared API types live in `crates/model`, `crates/inference`, `crates/backend`, `crates/provider`, and `crates/common`
- CPU execution works in CI through `backends/cpu`
- Core ML, ONNX, CUDA, and WebGPU crates establish execution boundaries without forcing hardware-specific CI
- Local, HTTP, and server providers establish local/remote/hybrid selection boundaries

## Execution modes

Local execution is `Application -> Runtime -> Backend`. Remote execution is
`Application -> Runtime -> Provider -> HTTP -> Runtime -> Backend`; the server
selects its own backend and the client only supplies a model identity
(`id` and optional `version`), never a filesystem path.

Start a local server and invoke it with the generic CLI:

```text
ml-runtime serve --bind 127.0.0.1:8080 --models examples/models
ml-runtime run example-linear@1 --endpoint http://127.0.0.1:8080 --execution prefer-local --tensor 2,4
```

The versioned wire contract is defined in `crates/protocol` under `/v1`.

## Hybrid routing

`InferenceOptions::execution` is an explicit, runtime-owned policy:
`LocalOnly`, `RemoteOnly`, `PreferLocal`, `PreferRemote`, `LocalThenRemote`,
or `RemoteThenLocal`. The runtime first checks registered capabilities and
uses stable registration/name ordering; backend selection happens only after
the local target has been selected.

Model artifacts are immutable application inputs; a loaded model is an
ephemeral runtime resource. Explicit `Runtime::load` returns a runtime-owned
handle, and lazy inference reuses the bounded, target-aware in-memory cache.
The execution order is `routing -> lifecycle -> scheduling -> batching ->
execution`; cache state is not authoritative application state.

`Runtime::infer_batch` preserves request order and result identity. Backends
that do not advertise batching continue to receive individual requests.
`Runtime::status` and the `ml-runtime status` command expose generic resource
state without exposing backend-private objects.

## Streaming

Streaming is a Rust-owned execution primitive. The execution order remains
`routing -> lifecycle -> scheduling -> batching -> backend/provider`, after
which the runtime normalizes output into `Started`, `Output`, and `Completed`
events. Consumers do not see provider or backend event types.

Backends advertise streaming capability truthfully. A backend without
incremental support uses compatibility mode and emits one complete `Output`
event, never synthetic token chunks. Runtime stream buffering is bounded by
`RuntimeConfig::max_stream_buffer`; a slow consumer therefore applies
backpressure rather than allowing unbounded output accumulation.

Cancellation is propagated while a stream is queued, executing, or producing
output. A canceled stream terminates with the structured cancellation error and
does not emit successful completion. Remote streaming uses the existing
versioned `/v1/infer/stream` protocol and preserves the same runtime-owned
events and execution metadata. `infer_loaded_stream` reuses the loaded model
resource and cache semantics of `infer_loaded`.

Fallback is policy-controlled, deterministic, and observable. Provider,
backend, timeout, transport, and model-availability failures may permit
fallback, while invalid inputs and contract violations never do. Every result
records the selected provider/backend, policy, target, model version, and any
fallback transition in `ExecutionMetadata`.
