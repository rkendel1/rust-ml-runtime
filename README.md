# rust-ml-runtime

A native local ML runtime for applications — no Python required.

`rust-ml-runtime` installs verified models, executes them locally, and returns
typed decisions with probabilities and provenance. The supported developer
surfaces are the `ml-runtime` executable, the Rust runtime crate, and the
in-process `@rust-ml-runtime/node` package.

## Quick start

```sh
# install ml-runtime (see the platform commands below)
ml-runtime --version
# install a model
ml-runtime model install laya
# verify
ml-runtime model doctor laya
# run local inference
ml-runtime laya "The customer asks for a refund."
```

The runtime and installed model execute locally. Laya uses ONNX Runtime on
Linux x86_64/ARM64 and CoreML on macOS/ARM64; unsupported platforms fail with
an explicit compatibility error. Model installation may use the network, while
inference from a READY installation does not.

## Install

Rust applications use the public crate:

```toml
[dependencies]
rust-ml-runtime = "0.2.3"
```

Node applications use the platform-selecting package:

```sh
npm install @rust-ml-runtime/node
```

The CLI is distributed as a native release artifact. Models are installed
separately and are not bundled in the Rust crate, npm package, or CLI archive.

The quickest installation on macOS and Linux is the verified installer:

```sh
curl --proto '=https' --tlsv1.2 -sSfL \
  https://raw.githubusercontent.com/rkendel1/rust-ml-runtime/main/install.sh | sh
```

The most reproducible manual installation is a release archive from GitHub Releases.
CLI archives contain `ml-runtime`, this README, and `LICENSE`. The public Node
package selects a separate platform-specific Node-API package; large model
artifacts are installed separately. Verify release downloads with:

```sh
shasum -a 256 -c SHA256SUMS
```

For development, build the same executable with `cargo build --release -p ml-runtime-cli`.
See the [native installation guide](docs/install.md) for supported archives,
the Python-free Laya lifecycle, storage locations, and diagnostics.

## General runtime commands

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
use rust_ml_runtime::{FilesystemModelCatalog, InferenceRequest, ModelReference, Runtime};

let runtime = Runtime::builder()
    .catalog(FilesystemModelCatalog::new("examples/models"))
    .build();
let request = InferenceRequest::new(ModelReference::versioned("mnist-8", "8"), input);
let result = runtime.infer(request).await?;
```

Developer documentation is organized from usage toward implementation:

1. [Native installation](docs/install.md)
2. [Node and TypeScript](docs/node.md)
3. [Release process](docs/release.md)
4. [Models and lifecycle](docs/models.md)
5. [Core API](docs/api.md#quickstart)
6. [Architecture](docs/architecture.md)
7. [Structured-decision planning](docs/runtime/execution-planning.md)
8. [Agent integration contract](docs/AGENTS.md)

## Public architecture

```text
Application
    │
    ▼
Node API / downstream consumers (for example, Jev)
    │
    ▼
rust-ml-runtime
    ├── model lifecycle
    ├── preprocessing
    ├── tokenizer
    ├── decoding
    ├── capability-aware execution planning
    └── provenance
    │
    ▼
Backend
    │
    ▼
Model
```

The boundaries are intentional: Model ≠ Runtime, Backend ≠ Runtime, Runtime ≠
Jev, and Jev ≠ Authority. ONNX Runtime and CoreML are distributable backends
for Laya, the first demonstrated model; neither defines the public runtime
abstraction. Application policy and authorization remain application-owned.
Jev is a downstream consumer and is not part of the runtime release gate.

## Release surface and compatibility

Applications use the same capability lifecycle—discover, resolve, authorize,
execute, observe, and return evidence—without depending on the library behind a
provider. Sensitive capabilities fail closed unless an application supplies its
authorization hook and explicit resource policy. Providers can be added behind
the runtime's capability boundary without changing application code.

The CLI and Rust API use the workspace version (`0.2.3` currently), exposed by
`ml-runtime --version` and `CARGO_PKG_VERSION`. The wire protocol is `/v1`;
supported routes are `/v1/health`, `/v1/capabilities`, `/v1/models`,
`/v1/infer`, and `/v1/infer/stream`. Protocol versions are independent of
patch releases, but incompatible wire changes require a new API version.

The support crates required by `rust-ml-runtime` are implementation details;
applications should depend only on `rust-ml-runtime`. The CLI is a separate
native product. ONNX and Core ML remain backend details behind the runtime
boundary.

## Models and server

### Native Laya distribution

Install the pinned Laya package into the runtime-owned application data
directory, then execute it without Python or a Hugging Face CLI:

```text
ml-runtime model install laya
ml-runtime model list
ml-runtime laya "The customer asks for a refund."
ml-runtime model doctor laya
```

`ML_RUNTIME_MODEL_DIR` overrides the installed-model root. Development and
offline tests can override the registry source with `--source ./models/laya`
or `ML_RUNTIME_LAYA_SOURCE`; normal installation uses the registry's pinned
HTTPS snapshot. Linux selects and validates the ONNX graph, external weights,
tokenizer, and calibration config. macOS/ARM64 compiles and validates CoreML.
Both paths publish a READY installation manifest only after backend
initialization succeeds, and neither silently downloads or repairs at
inference time.

The server owns model packages and clients send model identities, not server
filesystem paths:

```sh
ml-runtime serve --bind 127.0.0.1:8080 --models models
```

Node and TypeScript applications load the same Rust runtime in-process:

```ts
import { LocalML } from "@rust-ml-runtime/node";
```

See [Node and TypeScript](docs/node.md), [installation](docs/install.md), and
the [release contract](docs/release.md).

## Supported platforms

Release builds target Linux x86_64, Linux ARM64, macOS Intel, macOS Apple
Silicon, and Windows x86_64 when the corresponding CI job passes. Laya uses
ONNX on supported Linux hosts and CoreML on macOS/ARM64. GPU, WebGPU/WASM, and CUDA support may
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
load time, iteration count, latency, and throughput. The equivalent Rust
public-API example is runnable with `cargo run -p ml-runtime-cli --example
mnist`. `examples/typescript/mnist.ts` exercises the repository's private,
unpublished subprocess development binding; the public native Node package is
documented in [Node and TypeScript](docs/node.md).
