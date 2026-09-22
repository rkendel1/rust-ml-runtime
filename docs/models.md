# Models

`ModelSpec` is backend-independent and identifies the computational artifact:

- `id`
- `version`
- `format`
- `location`
- `metadata`

Recognized formats are currently:

- ONNX
- Core ML
- GGUF
- safetensors
- TensorRT
- Unknown

Recognition does not imply executability. Compatibility is decided by backends at runtime.

## Model packages

A portable model package is a directory containing `manifest.json` and an artifact
under `artifacts/`. The manifest resolves to a `ModelSpec` without making the
runtime depend on a backend-specific format:

```json
{
  "schema_version": 1,
  "id": "example-linear",
  "model_version": "1",
  "format": "Unknown",
  "artifact": "artifacts/model.json"
}
```

The checked-in `examples/models/linear` package is executable by the CPU
backend. Run it with `cargo run -p ml-runtime-cli -- run examples/models/linear
--tensor 2,4`.

## Catalog and execution

`FilesystemModelCatalog` discovers and validates packages without loading a
backend. A catalog descriptor has stable identity (`name@version`), package
location, format, and manifest metadata; resolving it produces a `ModelSpec`
for the runtime. The lifecycle is:

```
artifact -> package -> catalog descriptor -> model spec -> loaded model -> runtime resource
```

Catalogs answer where a model package is. Providers answer where inference
executes, and backends answer how it is computed. Discovery therefore does not
load models or select an execution target. Configure catalog roots explicitly
with `Runtime::builder().catalog(...)`.

## Acquisition and installation

Acquisition is separate from discovery and execution:

```
source -> acquire -> verify -> atomic install -> catalog -> resolve -> load -> execute
```

`ModelSource` obtains a package, while `ModelInstaller` verifies its manifest,
artifact integrity, and requested `ModelId` before renaming it into the
explicitly configured catalog root. Downloads are bounded, cancellation-aware,
and staged under `.staging`; failed operations are removed and never become
catalog entries. Installing does not load or execute a model. Existing valid
packages are reported as already installed unless replacement is explicitly
requested.

The small built-in registry adds named distributable packages without creating
a cloud registry. Each entry pins a source, revision, backend, authoritative
artifact layout, and file checksums. `RegisteredModelInstaller` streams HTTPS
files (or copies an explicit development source) into staging, verifies every
declared checksum, writes the model-neutral runtime manifest, and atomically
publishes the versioned package. On macOS the default root is
`~/Library/Application Support/ml-runtime/models`; set
`ML_RUNTIME_MODEL_DIR` to override it.

Laya is the first entry. Its installed layout keeps `model.mlpackage`,
`coreml_config.json`, `rl_agent_config.json`, and tokenizer files together.
Core ML compilation produces recreatable runtime state; the compiled output is
not authoritative and does not affect Laya's logical identity.

The CLI supports local package acquisition with:

```
ml-runtime models install examples/models/linear --models models --json
ml-runtime models verify example-linear@1 --models models
```

The real pretrained example is `examples/models/mnist-8`. Its manifest pins the
MIT-licensed ONNX Model Zoo artifact at revision
`a19f9a8c2333de1df9b03f10f5739f468b699a1a`, including its byte size and
SHA-256. See the README's “Real model example” for its explicit preprocessing
contract, local/remote commands, expected digit classification, and benchmark.
