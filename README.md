# ML Runtime

`ml-runtime` is a Rust-native runtime for local, remote, and hybrid ML inference.
The supported developer surfaces are the `ml-runtime` executable, the Rust
runtime crate, and the thin `@ml-runtime/core` TypeScript package.

## Install

The most reproducible installation is a release archive from GitHub Releases.
Archives contain `ml-runtime`, the example model catalog, this README, and
`LICENSE`. Verify downloads with:

```sh
shasum -a 256 -c SHA256SUMS
```

For development, build the same executable with `cargo build --release -p ml-runtime-cli`.

## Quickstart

```sh
ml-runtime --version
ml-runtime doctor
ml-runtime models
ml-runtime capabilities
ml-runtime run examples/models/linear --tensor 1,2 --shape 2
ml-runtime bench
```

`doctor` checks the executable, configuration, CPU backend, and model catalog
without downloading or executing a large model. Model installation uses the
Rust acquisition API:

```sh
ml-runtime models install path/to/model-package --models models
ml-runtime inspect example-linear@1 --models models
```

## Release surface and compatibility

The CLI and Rust API use the workspace version (`0.1.0` currently), exposed by
`ml-runtime --version` and `CARGO_PKG_VERSION`. The wire protocol is `/v1`;
supported routes are `/v1/health`, `/v1/capabilities`, `/v1/models`,
`/v1/infer`, and `/v1/infer/stream`. Protocol versions are independent of
patch releases, but incompatible wire changes require a new API version.

Workspace crates other than the runtime and CLI, backend crates, providers, and
model/protocol crates are implementation details. ONNX and Core ML remain
backend details behind the runtime boundary.

## Models and server

The server owns model packages and clients send model identities, not server
filesystem paths:

```sh
ml-runtime serve --bind 127.0.0.1:8080 --models models
```

TypeScript is intentionally a thin wrapper over the Rust CLI and requires a
separately installed `ml-runtime`; it does not embed a native inference engine:

```ts
import { RuntimeClient } from "@ml-runtime/core";
```

## Supported platforms

Release builds target Linux x86_64, Linux ARM64, macOS Intel, and macOS Apple
Silicon when the corresponding CI job passes. CPU inference is available
without Core ML. Core ML is macOS-only; GPU, WebGPU/WASM, and CUDA support may
be unavailable or experimental depending on the build. Check
`ml-runtime capabilities` rather than assuming a backend is present.

## Configuration and exit codes

Commands use explicit `--models`, `--endpoint`, `--backend`, and `--provider`
options. No daemon or telemetry service is required. Exit code `0` means
success, `1` means a runtime/application failure, and `2` means invalid CLI
usage. `--json` changes output only, not exit-code semantics.

See `docs/architecture.md`, `docs/models.md`, and `docs/bindings.md` for deeper
implementation details.
