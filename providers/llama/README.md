# ml-runtime-llama-provider (spike)

Embedded llama.cpp `GenerationProvider`. Standalone, unpublished, and excluded
from the repository workspace. See [docs/embedded-generation.md](../../docs/embedded-generation.md).

```text
cargo test --test synthetic_model                      # real inference, generated fixtures
ML_RUNTIME_TEST_GGUF=… cargo test --test real_model -- --ignored --nocapture
cargo run --release --example bench -- model.gguf 128 2048
```

Needs CMake, a C++ compiler, and libclang. Never downloads model weights.
