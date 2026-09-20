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
