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
