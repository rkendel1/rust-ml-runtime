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
