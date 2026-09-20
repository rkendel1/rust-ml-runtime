# Backends

Backends describe how computation executes.

Current workspace backends:

- `cpu`: portable local backend used in CI
- `coreml`: executable Apple Core ML backend on macOS; unavailable elsewhere
- `onnx`: ONNX Runtime integration boundary
- `cuda`: CUDA integration boundary
- `webgpu`: WebGPU/WASM integration boundary

The CPU and ONNX backends execute portably. Core ML executes tensor models through Apple's
`MLModel`/`MLMultiArray` APIs on macOS; its compiled `.mlmodelc` artifact remains inside the
normal model package. Core ML accepts dense `f32` tensors and reports `available: false` with an
unsupported-platform note on non-macOS systems. Availability does not claim that a particular
request used the Neural Engine or GPU.

Validate the native path on macOS with:

```text
cargo test -p ml-runtime-coreml-backend
```

The Core ML model package uses the same `manifest.json` and `artifacts/model.mlmodelc` layout as
other packages. Generate the compiled artifact with Xcode's `coremlcompiler` from a deterministic
Core ML model before running the macOS validation.
