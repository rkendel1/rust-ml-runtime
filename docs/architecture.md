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
