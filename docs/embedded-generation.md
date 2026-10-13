# Embedded local generation (llama.cpp spike)

**Outcome A, with stated limits.** `rust-ml-runtime` can run a GGUF model
in-process behind the existing `GenerationProvider` contract using
[`llama-cpp-2`](https://crates.io/crates/llama-cpp-2). Real llama.cpp inference was
proven end to end on Linux x86_64 CPU against **generated random-weight GGUF
models**. It was **not** proven with a pretrained model (none could be downloaded
from the audit sandbox) and **not** on Apple Silicon, Windows, or inside the Node
addon. The prototype lives in `providers/llama` as a standalone, unpublished
crate so nothing else in the repository builds llama.cpp.

(An earlier audit of Uzu reached a no-go for embedding; see PR #46.)

## Candidate audit

Both candidates were built and run here (stable Rust 1.97.0, 4 vCPU Intel Xeon @
2.1 GHz with AVX-512, Ubuntu 24.04, glibc 2.39). "Verified" means executed here.

| Requirement | `llama-cpp-2` 0.1.159 (`llama-cpp-sys-2` 0.1.159) | Candle 0.11.0 (`candle-core`, `candle-transformers`) |
|---|---|---|
| Publishability | On crates.io, MIT OR Apache-2.0; llama.cpp source is vendored in the sys crate (~40 MB unpacked). Verified: resolves and builds from the registry. | On crates.io, MIT OR Apache-2.0. Verified. |
| Toolchain | Stable. **Needs CMake, a C++ toolchain, and libclang** (bindgen). Cold build 3m43s (debug), 4m31s (release examples). Verified. | Stable, pure Rust, no external tools. Cold release build 1m58s. Verified. |
| Model format | GGUF, the common distribution format; no conversion. | Loads llama-family GGUF (`quantized_llama`; verified on the same file) but needs a **separate `tokenizer.json`** and per-architecture model code. |
| CPU inference | Real. Decode **107–115 tok/s**, TTFT ≈205 ms (164-token prompt) on the 85M-parameter F16 fixture. Verified. | Real. Same file and prompt length: decode **18–19 tok/s**, TTFT 267–307 ms (default features: no MKL/Accelerate). Verified. |
| Apple Silicon | `metal` feature exists. **Not tested** (no Mac). | `metal` feature exists. **Not tested.** |
| Streaming / cancellation | Token-by-token decode loop; cancellation checked per token and per prompt chunk. Implemented and measured (below). | Token-by-token possible; not implemented in this PR. |
| Structured output | Native GBNF grammar sampler plus `json_schema_to_grammar`. Implemented and tested (enforced, then still validated by the runtime). | No built-in grammar-constrained decoding. |
| Tool calling | No parser that maps output to the validated `ToolCall` contract. **Rejected** (`tools = false`). | Same. |
| Resource bounds | Fixed `n_ctx` per context, explicit output cap, admission semaphore. Note: creating contexts in parallel is documented as unsound upstream, so creation is serialized. | Would need to be built. |
| Node packaging | **Unverified; see below.** | Pure Rust would package more easily; unverified. |

Selection: **`llama-cpp-2`**. It was ~6x faster on identical weights in this
environment, supports GGUF without a separate tokenizer file, and provides
grammar-constrained decoding, which is the one structured-output mechanism that
is actually enforced. Caveat: Candle was measured only on an F16 GGUF with default
features; its quantized Q4/Q8 paths were not compared.

## What was built

`providers/llama` (`ml-runtime-llama-provider`, `publish = false`, its own
`[workspace]` and `Cargo.lock`):

```rust
let mut config = LlamaConfig::new("/models/model.gguf", "my-model"); // explicit path + id
config.context_tokens = 2048;        // per-request context window
config.max_output_tokens = 512;      // cap and default
config.max_concurrent_requests = 1;  // admission slots; each owns a KV cache
config.expected_sha256 = Some("…".into()); // optional integrity check
let provider = LlamaProvider::new(config)?; // blocking: loads the model, no network
let runtime = Runtime::builder().register_generation_provider(provider).build();
let reply = runtime.chat("llama-cpp", ChatRequest::new("my-model", messages), None).await?;
```

- **Identity:** a request for any id other than `model_id` is `ModelNotFound`;
  nothing is substituted. `list_generation_models` returns the loaded model;
  `metadata.reported_model` is the GGUF `general.name`. A GGUF has no verifiable
  id, so `model_id` is a caller-declared label; use `expected_sha256` to pin the
  file.
- **Loading errors:** missing file → `ModelUnavailable` ("weights are never
  downloaded automatically"); not GGUF or unloadable → `UnsupportedModel`; hash
  mismatch → `ModelIntegrity`; bad config → `InvalidRequest`.
- **Capabilities:** streaming, stop sequences, seed, `temperature`, `top_p`,
  `json_object` and `json_schema` (grammar-enforced). Tools and `tool_choice` are
  off, so the runtime rejects them with `CapabilityMismatch`. Unspecified or zero
  temperature is greedy (deterministic); `seed` and `top_p` only matter when
  `temperature > 0`.
- **Prompts:** the GGUF's embedded chat template via llama.cpp's built-in templater
  (or `LlamaConfig::chat_template`). Unrecognized Jinja templates are rejected
  with `UnsupportedModel` rather than guessed; roles are never dropped.
- **Limits:** `max_tokens` above the cap and prompts that do not fit the context
  are `ResourceLimit`; output is bounded by tokens and 4 MiB.
- **Concurrency/cleanup:** a semaphore bounds concurrent generations; waiting is
  cancellable. Each stream runs on a blocking thread holding its slot; it ends on
  completion, cancellation, error, or when the consumer drops the stream, then
  frees its context and slot. The model is `Arc`-shared, so it cannot be unloaded
  under a running request.
- **Metadata:** provider `llama-cpp`, transport `in-process`, execution `local`
  (not inferred), endpoint `local-file:<file name>` (no path). Token counts are
  exact.

### Default CI vs. this crate
`cargo test --workspace`, the release `--all-features` gates, the published crates
and the npm addon do **not** include `providers/llama` and do not compile
llama.cpp (verified with `cargo tree`). Run the spike's tests with:

```text
cd providers/llama
cargo test --test synthetic_model     # 16 tests, real inference, no downloads
```

`rust-ml-runtime` gained one small re-export, `GenerationProvider` at the crate
root, because implementing a provider otherwise required importing it from
`ml_runtime_provider`.

## Test model

**Offline default (used for everything verified here):** `synthetic.rs` writes a
deterministic llama-architecture GGUF with seeded random weights and a tiny
byte-level vocabulary (generated, so no licence applies; nothing is committed or
downloaded). `tiny` (~100k parameters) powers the tests; `100m` (85M parameters,
F16, 162.6 MiB) powers the benchmark. Output is gibberish by design.

**Real model (opt-in; not run):** the prerequisite is a GGUF already on disk. A
suitable small, permissively licensed choice is an instruction-tuned
Qwen2.5-0.5B GGUF (Apache-2.0 as published by its authors; roughly 0.4 GB at
Q4_K_M) or SmolLM2-135M-Instruct (Apache-2.0). I could not reach Hugging Face from
the sandbox, so **exact revision, size, and SHA-256 are unverified**: record the
SHA-256 of the file you obtain, check the licence on the model card, then:

```text
ML_RUNTIME_TEST_GGUF=/path/model.gguf ML_RUNTIME_TEST_GGUF_SHA256=<hex> \
  cargo test --test real_model -- --ignored --nocapture
cargo run --release --example bench -- /path/model.gguf 128 2048
```

Without `ML_RUNTIME_TEST_GGUF` the test fails with an explicit prerequisite
message. Nothing is downloaded by the provider, the tests, or the benchmark;
weights stay out of Git and npm.

## Measurements (Linux x86_64 CPU, synthetic fixture)

`cargo run --release --example bench -- synthetic-100m.gguf 128 2048` with
`llama-cpp-2` 0.1.159, CPU backend only (`gpu_layers = 0`, no accelerator
compiled in), 4 vCPU Xeon @ 2.1 GHz:

| Metric | Result |
|---|---|
| Model | generated `synthetic-100m`: 85,225,728 parameters, F16, 162.6 MiB, SHA-256 `f8f4edc0…c2936` (random weights) |
| Load time | 3.9 ms (mmap; weights are paged in lazily, so this is **not** a cold-read number) |
| Prompt / generation | 164 prompt tokens, 128 generated, greedy, context 2048 |
| Time to first token | 205–211 ms (**includes** prompt processing) |
| Decode speed | 106.6–114.6 tok/s (excludes the first token) |
| Cancellation latency | stream ended 13.1–15.2 ms after `cancel()` (includes draining up to 32 buffered deltas); next request then admitted and finished in 225–255 ms (mostly its own prompt processing) |
| Peak RSS | 272 MiB (whole process, includes touched mmap pages) |
| Backend used | llama.cpp CPU; no GPU offload; OpenMP disabled |
| Apple Silicon / Metal | **not run** |
| Real pretrained model | **not run** |

Throughput on random weights of this shape is a valid compute measurement for the
shape but says nothing about any real model, and must not be compared with other
models or quantizations.

### Cancellation semantics
llama.cpp is not interrupted mid-token. Cancellation (or dropping the stream) is
observed before each token and between prompt chunks (`batch_tokens`, default
512), so worst-case latency is one decode step or one prompt chunk. A cancelled
or abandoned request cannot keep running beyond that, and its slot is released.

## Node and packaging: not done, with the blockers
The Node addon does not include llama.cpp and this PR does not wire it in.
Packaging it would need proof on all five shipped targets, none of which was
done:

- the Linux x64/arm64 addons are built in the repository's Bookworm image
  (glibc 2.36 baseline): it would need CMake and libclang added, and `libstdc++`
  becomes a dynamic dependency. The binary built here reports `libstdc++.so.6`,
  `libgcc_s`, `libm`, `libc` and needs GLIBC 2.39 / GLIBCXX 3.4.29, which is
  meaningless against the 2.36 gate because it was built on Ubuntu 24.04;
- OpenMP is deliberately off (`default-features = false`), otherwise `libgomp`
  would be added;
- macOS (Metal/Accelerate choice) and Windows MSVC builds need their own proof;
- adding a feature to the addon would make the release `--all-features` gates
  compile llama.cpp on every runner.

## Next prerequisite
Run `real_model` and `bench` on a Mac and on a real GGUF, then prove the addon
build on one Bookworm image; those two results decide whether to promote the
crate into the workspace. Until then Node users can reach local models through a
local OpenAI-compatible server and the existing provider.
