# Backends

Backends describe how computation executes.

Current workspace backends:

- `cpu`: portable local backend used in CI
- `coreml`: executable Apple Core ML backend on macOS; unavailable elsewhere
- `onnx`: ONNX Runtime integration boundary
- `cuda`: CUDA integration boundary
- `webgpu`: WebGPU/WASM integration boundary

Loaded structured-decision models expose a second, model-specific capability
contract used by the execution planner. See the factual matrix and deterministic
strategy rules in [Structured-decision execution planning](runtime/execution-planning.md).

The CPU and ONNX backends execute portably. Core ML executes tensor models through Apple's
`MLModel`/`MLMultiArray` APIs on macOS; its compiled `.mlmodelc` artifact remains inside the
normal model package. Core ML accepts dense `f32` tensors and reports `available: false` with an
unsupported-platform note on non-macOS systems. Availability does not claim that a particular
request used the Neural Engine or GPU.

Validate the native path on macOS with:

```text
cargo test -p ml-runtime-coreml-backend
```

## Laya typed decisions

Enable the explicit `coreml` runtime feature to load a local
`aac6fef/laya-coreml` directory. The runtime reads and validates
`coreml_config.json`, `rl_agent_config.json`, tokenizer metadata, every supplied
file size/SHA-256, and the Core ML package tree hash. It never downloads a
model. The supplied `model.mlpackage` is compiled through Core ML's native
`compileModelAtURL:error:` API and the resulting `.mlmodelc` is loaded with the
validated CPU+GPU configuration. The compiled directory is published beneath
the runtime cache (`~/Library/Caches/ml-runtime` by default, configurable with
`ML_RUNTIME_CACHE_DIR`) under the verified package hash. It can be removed and
recreated without changing model identity. An already compiled `model.mlmodelc` is also
accepted when it is stored beside metadata whose file manifest describes the
local artifact contents.

The v1 mapping is intentionally Laya-specific and lives behind the generic
typed-decision traits:

| Runtime value | Core ML feature | Type and shape |
| --- | --- | --- |
| token IDs | `input_ids` | int32 `[1,L]` |
| valid-token mask | `attention_mask` | int32 `[1,L]` |
| option marker offsets | `marker_pos` | int32 `[1,32]` |
| valid-option mask | `marker_mask` | int32 `[1,32]` |
| choice/score/noul tag | `qtype` | int32 `[1]` |
| option scores | `logits` | float32 `[1,32]` |
| act/escalate scores | `action_logits` | float32 `[1,2]` |

The provider reads these descriptions from the compiled `MLModel` and checks
their names, data types, and default shapes against `coreml_config.json` before
accepting the model. The downloaded package's flexible token inputs report the
configured default length (`[1,128]`); requests are padded to one of the
exported lengths through 512. A successfully compiled model with a different
feature contract is rejected rather than executed.

Cargo runs an integration-test binary from a package-specific working
directory. Consequently, a relative `LAYA_MODEL_PATH=./models/laya` is resolved
against the workspace when it is not present in the process working directory.
The test skips only when the resolved directory truly does not exist;
incomplete metadata, tokenizer, or `model.mlpackage` contents are failures.

Prompt construction, tokenization, enumerated shape selection, calibration,
and typed decoding match the artifact's v1 contract. Core ML itself remains a
tensor backend and contains no decision semantics.

Run the real-model test and smoke path on macOS with a local artifact:

```text
LAYA_MODEL_PATH=./models/laya cargo test -p rust-ml-runtime --features coreml \
  --test laya_coreml -- --nocapture
cargo run -p rust-ml-runtime --features coreml --example laya -- \
  --model ./models/laya \
  --input "The customer asks for a refund of a duplicate payment."
```
