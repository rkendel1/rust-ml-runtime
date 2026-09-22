# ML Runtime

`ml-runtime` is a Rust-native capability runtime for intelligent applications.
ML was the initial capability family; the runtime now provides common discovery,
execution, policy, limits, and evidence for intelligence, data, filesystem,
process, network, and developer-tool capabilities.
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
ml-runtime capabilities inspect filesystem.read
ml-runtime providers list --providers providers
ml-runtime providers inspect providers/example
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

For embedding, the canonical Rust surface is model + input + options → result:

```rust
use ml_runtime::{FilesystemModelCatalog, InferenceRequest, ModelReference, Runtime};

let runtime = Runtime::builder()
    .catalog(FilesystemModelCatalog::new("examples/models"))
    .build();
let request = InferenceRequest::new(ModelReference::versioned("mnist-8", "8"), input);
let result = runtime.infer(request).await?;
```

Developer documentation is organized from usage toward implementation:

1. [Quickstart and core API](docs/api.md#quickstart)
2. [Models and lifecycle](docs/models.md)
3. [Execution policies and metadata](docs/api.md#execution-policies-and-metadata)
4. [Streaming](docs/api.md#streaming)
5. [Batching](docs/api.md#batching)
6. [Remote execution](docs/api.md#remote-execution)
7. [Observability](docs/observability.md)
8. [Architecture](docs/architecture.md)

## Release surface and compatibility

Applications use the same capability lifecycle—discover, resolve, authorize,
execute, observe, and return evidence—without depending on the library behind a
provider. Sensitive capabilities fail closed unless an application supplies its
authorization hook and explicit resource policy. Providers can be added behind
the runtime's capability boundary without changing application code.

The CLI and Rust API use the workspace version (`0.1.0` currently), exposed by
`ml-runtime --version` and `CARGO_PKG_VERSION`. The wire protocol is `/v1`;
supported routes are `/v1/health`, `/v1/capabilities`, `/v1/models`,
`/v1/infer`, and `/v1/infer/stream`. Protocol versions are independent of
patch releases, but incompatible wire changes require a new API version.

Workspace crates other than the runtime and CLI, backend crates, providers, and
model/protocol crates are implementation details. ONNX and Core ML remain
backend details behind the runtime boundary.

## Models and server

### Native Laya distribution (macOS)

Install the pinned Laya package into the runtime-owned application data
directory, then execute it without Python or a Hugging Face CLI:

```text
ml-runtime model install laya
ml-runtime laya "The customer asks for a refund."
```

`ML_RUNTIME_MODEL_DIR` overrides the installed-model root. Development and
offline tests can override the registry source with `--source ./models/laya`
or `ML_RUNTIME_LAYA_SOURCE`; normal installation uses the registry's pinned
HTTPS snapshot. The authoritative artifact remains `model.mlpackage` plus its
tokenizer and configuration. Core ML's compiled representation is disposable
runtime state under the runtime cache and is never used as package identity.

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

## Real model example

The checked-in `mnist-8@8` package is the 26 KB pretrained MNIST convolutional
network from the ONNX Model Zoo (MIT licensed, ONNX opset 8). Unlike
`onnx-example`, which is a tiny deterministic operator fixture for CI,
`mnist-8` was trained to classify handwritten digits and is the repository's
first real pretrained model. Its artifact is pinned by size and SHA-256 in the
authoritative manifest, so ordinary builds and tests never access the network.

Build the runtime and inspect its capabilities and catalog:

```sh
cargo build --release -p ml-runtime-cli
export PATH="$PWD/target/release:$PATH"
ml-runtime capabilities
ml-runtime models --models examples/models
ml-runtime inspect mnist-8@8 --models examples/models
ml-runtime models verify mnist-8@8 --models examples/models
```

The package can also pass through the normal verified local acquisition path;
installation is explicit and performs no download:

```sh
ml-runtime models install examples/models/mnist-8 --models .local-models
ml-runtime models verify mnist-8@8 --models .local-models
```

The model contract is `Input3: float32[1,1,28,28]` to
`Plus214_Output_0: float32[1,10]`. Input is one 28×28 grayscale image with a
black background, white foreground, and pixel values scaled to `[0,1]`. The
output contains logits for digits 0 through 9; selecting the largest logit is
the only postprocessing used here. Preprocessing remains application-owned.

The prepared tensor in `examples/inputs/mnist-8-seven.csv` draws a seven and
can be supplied directly to the existing general-purpose tensor flags:

```sh
MNIST_INPUT=$(tr -d '\n' < examples/inputs/mnist-8-seven.csv)
ml-runtime run mnist-8@8 \
  --models examples/models \
  --backend onnx \
  --tensor "$MNIST_INPUT" \
  --shape 1,1,28,28 \
  --verbose
```

The largest output logit is index 7 (approximately `25.5519`). Execution ends
with `provider=local backend=onnx hardware=Some("cpu")`; small floating-point
differences across ONNX Runtime builds are expected.

To exercise the remote transport, keep the server in one terminal:

```sh
ml-runtime serve --bind 127.0.0.1:8080 --models examples/models
```

Then run the identity-only request from another terminal. `prefer-local` tells
the server to resolve and execute its catalog model rather than forward it:

```sh
MNIST_INPUT=$(tr -d '\n' < examples/inputs/mnist-8-seven.csv)
ml-runtime run mnist-8@8 \
  --endpoint http://127.0.0.1:8080 \
  --execution prefer-local \
  --tensor "$MNIST_INPUT" \
  --shape 1,1,28,28 \
  --verbose
```

The client reports `provider=remote backend=onnx`: `remote` truthfully records
the HTTP transport while `onnx` records the server's actual execution backend.
The server-side integration test separately verifies `provider=local` and
`backend=onnx` before the client decorates transport metadata.

Establish a hardware-specific baseline without imposing a performance gate:

```sh
ml-runtime bench mnist-8@8 \
  --models examples/models \
  --backend onnx \
  --tensor "$MNIST_INPUT" \
  --shape 1,1,28,28 \
  --iterations 20
```

The report includes model/version, provider, backend, execution target, model
load time, iteration count, latency, and throughput. The equivalent public-API
examples are runnable with `cargo run -p ml-runtime-cli --example mnist` and
are also provided in `examples/typescript/mnist.ts` for `@ml-runtime/core`.
