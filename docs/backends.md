# Backends

Backends describe how computation executes.

Current workspace backends:

- `cpu`: portable local backend used in CI
- `coreml`: Apple-native accelerator boundary
- `onnx`: ONNX Runtime integration boundary
- `cuda`: CUDA integration boundary
- `webgpu`: WebGPU/WASM integration boundary

Only the CPU backend currently executes inference in this scaffold. The others intentionally advertise capabilities and availability without pretending to execute when support is not wired in.
