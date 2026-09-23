# rust-ml-runtime 0.2.0

A native local ML runtime for applications without Python.

## Included

- Deterministic Node native binding resolution with structured diagnostics
- Published npm artifact validation and Linux optional-dependency install checks
- Native runtime self-test support through `@rust-ml-runtime/node`
- Native Rust CLI releases for macOS, Linux, and Windows
- Rust library distribution through `rust-ml-runtime = "0.1"` on crates.io
- Explicit model installation, verification, diagnosis, and removal
- SHA-256 verification for runtime and model artifacts
- Local Core ML execution with READY/offline lifecycle semantics
- Typed decisions, probabilities, latency, and provenance
- In-process Node/TypeScript integration through `@rust-ml-runtime/node`
- A standalone healthcare decision demonstration through the public Node API
- Laya model support on compatible macOS hosts

Models remain separately distributed and are not embedded in the Rust crate.

## Example

```sh
ml-runtime model install laya
ml-runtime model doctor laya
ml-runtime laya "The customer asks for a refund."
```

## Current backend and model

Core ML is the first distributable backend. Laya is the first demonstrated
model. They are implementations behind the model-neutral runtime boundary, not
the definition of that boundary. The runtime does not make application policy
or authorization decisions. Jev remains a downstream consumer and owns its
separate compatibility validation against this public release.
