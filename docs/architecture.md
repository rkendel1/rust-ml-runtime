# Architecture

The runtime is intentionally boring at the center:

`load -> select -> execute -> return -> observe`

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
ml-runtime run --endpoint http://127.0.0.1:8080 --model example-linear@1 --tensor 2,4
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

Fallback is policy-controlled, deterministic, and observable. Provider,
backend, timeout, transport, and model-availability failures may permit
fallback, while invalid inputs and contract violations never do. Every result
records the selected provider/backend, policy, target, model version, and any
fallback transition in `ExecutionMetadata`.
