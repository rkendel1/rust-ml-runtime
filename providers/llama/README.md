# ml-runtime-llama-provider

Embedded llama.cpp `GenerationProvider`. Standalone, unpublished, and excluded
from the repository workspace. Qualification results, measurements and the Node
packaging decision are in [docs/embedded-generation.md](../../docs/embedded-generation.md).

```text
cargo test --locked                                   # generated fixtures, no downloads
ML_RUNTIME_TEST_GGUF=… ML_RUNTIME_TEST_GGUF_SHA256=… \
  cargo test --release --test real_model -- --ignored --nocapture --test-threads=1
cargo run --release --example bench -- model.gguf 128 2048 [gpu_layers]
```

Needs CMake, a C++ compiler, and libclang. Never downloads model weights.
`gpu_layers = 0` is CPU only; Metal (macOS) is used only when `gpu_layers > 0`,
but it is always compiled in on Apple targets. `ML_RUNTIME_LLAMA_LOGS=1` shows
llama.cpp's own device and offload log.
