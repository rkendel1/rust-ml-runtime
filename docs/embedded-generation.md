# Embedded local generation (llama.cpp): qualification

**Status: validated on real hardware, not yet shipped.** `providers/llama` is a
standalone, unpublished crate that runs a GGUF model in-process behind the
existing `GenerationProvider` contract using
[`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) 0.1.159. Two independent
gates were evaluated; neither implies the other.

| Gate | Result | Scope of the evidence |
|---|---|---|
| **Real model** | **Passed** | One pretrained model (Qwen2.5-0.5B-Instruct Q4_K_M) on one Apple Silicon machine (M4 Max, macOS 27.0), on **both** the CPU path and the Metal path |
| **Distribution** | **Passed for two Linux targets, as a probe** | `linux-arm64-gnu` (native) and `linux-x64-gnu` (emulated) built in the repository's Bookworm image and run in a clean `node:22-bookworm-slim` container. **Not** passed on macOS (dyld rejects the addon) or Windows (not attempted) |

Chosen distribution boundary: **Option A, a separate opt-in addon** (see
[Node distribution](#node-distribution-boundary)). Node consumers **cannot** use
embedded inference through the published `@rust-ml-runtime/node` package, and
nothing in this change adds that.

(An earlier audit of Uzu reached a no-go for embedding; see PR #46.)

## Candidate audit

Both candidates were built and run here (stable Rust, 4 vCPU Intel Xeon @
2.1 GHz with AVX-512, Ubuntu 24.04, glibc 2.39). "Verified" means executed here.

| Requirement | `llama-cpp-2` 0.1.159 (`llama-cpp-sys-2` 0.1.159) | Candle 0.11.0 (`candle-core`, `candle-transformers`) |
|---|---|---|
| Publishability | On crates.io, MIT OR Apache-2.0; llama.cpp source is vendored in the sys crate (~40 MB unpacked). Verified: resolves and builds from the registry. | On crates.io, MIT OR Apache-2.0. Verified. |
| Toolchain | Stable. **Needs CMake, a C++ toolchain, and libclang** (bindgen). Cold build 3m43s (debug), 4m31s (release examples). Verified. | Stable, pure Rust, no external tools. Cold release build 1m58s. Verified. |
| Model format | GGUF, the common distribution format; no conversion. | Loads llama-family GGUF (`quantized_llama`; verified on the same file) but needs a **separate `tokenizer.json`** and per-architecture model code. |
| CPU inference | Real. Decode **107–115 tok/s**, TTFT ≈205 ms (164-token prompt) on the 85M-parameter F16 fixture. Verified. | Real. Same file and prompt length: decode **18–19 tok/s**, TTFT 267–307 ms (default features: no MKL/Accelerate). Verified. |
| Apple Silicon | Verified on an M4 Max; see [Apple Silicon results](#apple-silicon-results). The crate's `metal` feature is **not** what enables Metal (see below). | `metal` feature exists. **Not tested.** |
| Streaming / cancellation | Token-by-token decode loop; cancellation checked per token and per prompt chunk. Implemented and measured (below). | Token-by-token possible; not implemented in this PR. |
| Structured output | Native GBNF grammar sampler plus `json_schema_to_grammar`. Implemented and tested (enforced, then still validated by the runtime). | No built-in grammar-constrained decoding. |
| Tool calling | No parser that maps output to the validated `ToolCall` contract. **Rejected** (`tools = false`). | Same. |
| Resource bounds | Fixed `n_ctx` per context, explicit output cap, admission semaphore. Note: creating contexts in parallel is documented as unsound upstream, so creation is serialized. | Would need to be built. |
| Node packaging | Built and run on Bookworm arm64/x64; see [Node distribution](#node-distribution-boundary). | Pure Rust would package more easily; unverified. |

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
cargo test --test synthetic_model     # 17 tests, real inference, no downloads
```

`rust-ml-runtime` gained one small re-export, `GenerationProvider` at the crate
root, because implementing a provider otherwise required importing it from
`ml_runtime_provider`.

## Audit of the prototype (PR #47) and fixes

Read and exercised before adding anything. **Verified as designed:**

- **Identity.** `ChatRequest.model` must equal `LlamaConfig::model_id`, checked
  before admission and before any work; anything else is `ModelNotFound`. There
  is no fallback and no second model.
- **Integrity.** `expected_sha256` is checked inside `LlamaProvider::new`, after
  the `GGUF` magic check and **before** llama.cpp is asked to load the file
  (`ModelIntegrity`). Hashing the 491 MB file took ~0.97 s. The file is hashed
  and then loaded by path, so a writer who can replace the file between the two
  steps defeats the check; this is acceptable only for a model directory you
  control. llama.cpp parses GGUF, so load only files you trust.
- **Loading.** Loading is eager and explicit (`new` blocks; call it from
  `spawn_blocking`). There is no lazy load. Weight *reads* are lazy because the
  file is memory-mapped, which is why `load_time` alone is not a cold-start
  number (see measurements).
- **Structured output.** The GBNF sampler constrains decoding; the runtime then
  still validates the text (`structured` is populated by the runtime, never by
  the provider), and a truncated JSON result is reported invalid, not repaired.
- **Resource bounds.** A semaphore bounds concurrent generations; waiting is
  cancellable; each generation owns one `context_tokens` KV cache; prompts that
  do not fit and `max_tokens` above the cap are `ResourceLimit`; output is
  capped at 4 MiB.
- **Isolation.** `providers/llama` and `bindings/node-llama` have their own
  `[workspace]` and `Cargo.lock`. `tools/check-local-engine-isolation.mjs`
  proves that default and `--all-features` workspace builds, the shipped addon,
  and the npm tarball contain no llama.cpp, no local addon, and no `.gguf`.

**Defects found and fixed in this change:**

1. Dropping a `chat()` future (a caller timeout) left the blocking generation
   running and holding its slot until `max_tokens`. Generation now runs under a
   token linked to the caller's token and cancelled when the future drops.
2. Dropping a stream was only noticed when the next *text* was emitted, so a
   long prompt or a grammar step with no visible text kept running. The stream
   now owns a drop guard on the same linked token.
3. `seed` was truncated with `as u32`, so seeds differing only in the high
   32 bits collided. High and low halves are now folded.
4. A prompt that tokenized to nothing would underflow `tokens.len() - 1`; it is
   now `InvalidRequest`.
5. `context_tokens` above the model's trained context was accepted and silently
   degraded; it is now `InvalidRequest`.
6. `threads` applied to decoding but not to prompt processing; both are set.
7. **`gpu_layers = 0` was not CPU-only on Apple Silicon.** llama.cpp's scheduler
   still moved large-batch (prompt) matrix multiplications to Metal (log:
   `MTL0 compute buffer`, 277 graph splits). The "CPU" prompt speed was
   inflated (983 tok/s measured) until op offload was disabled for
   `gpu_layers = 0`. CPU is now CPU (log: `graph splits = 1`, no `MTL0`
   buffers).
8. `LoadedModelInfo` now separates `verify_time` (hashing) from `load_time`
   (llama.cpp loader) and lists the ggml `devices` compiled in. Setting
   `ML_RUNTIME_LLAMA_LOGS=1` stops the provider from silencing llama.cpp's own
   log, which is how the device assignment below was proven.

**Known limitation kept:** cancellation is observed between tokens and between
prompt chunks of `batch_tokens` (512). Decode-time latency was measured at
4-6 ms; worst case during prompt processing is one chunk, which from the
measured prompt speeds is roughly 1.7 s on CPU and 60 ms on Metal (computed,
not measured).

## Real model

The model is **never** committed, downloaded by the provider, or shipped in npm.
Obtain it yourself:

```text
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --revision 9217f5db79a29953eb74d5343926648285ec7e67 --local-dir ~/models/qwen2.5-0.5b-instruct
shasum -a 256 ~/models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

| | |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, file `qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| Revision | `9217f5db79a29953eb74d5343926648285ec7e67` (Hugging Face) |
| Licence | Apache-2.0 (model card front matter; license link on the model card) |
| Size / SHA-256 | 491,400,032 bytes / `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` (equals the Hub's LFS digest) |
| Architecture / quantization | `qwen2`; `Q4_K - Medium`, 6.16 bits per weight; llama.cpp reports 630,167,424 parameters (including embeddings); trained context 32,768 |
| Tokenizer / template | Embedded in the GGUF (`gpt2`-style BPE; ChatML-family Jinja template). **No separate tokenizer or template file.** llama.cpp's built-in templater recognizes the template; unrecognized templates are rejected, never guessed |
| Test machine | Apple M4 Max, 36 GiB, macOS 27.0 (26A428), Darwin 27.0.0 arm64, Rust 1.96.0, Node 24.9.0, Apple ld 27037.1 |
| Engine | `llama-cpp-2` / `llama-cpp-sys-2` 0.1.159 (vendored llama.cpp, ggml 0.24.0) |

```text
ML_RUNTIME_TEST_GGUF=~/models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf \
ML_RUNTIME_TEST_GGUF_SHA256=74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db \
  cargo test --release --test real_model -- --ignored --nocapture --test-threads=1
# the same suite through Metal:
ML_RUNTIME_TEST_GPU_LAYERS=99 ...
```

`tests/real_model.rs` is `#[ignore]`d, so ordinary CI never needs weights; a
missing prerequisite fails with an explicit message (and a non-ignored test
pins that message). It asserts behavior, not prose. **All 8 real-model tests
passed on CPU (`gpu_layers = 0`) and on Metal (`gpu_layers = 99`):**

1. loads; reports metadata; "capital of France" answers contain "paris";
2. the chat template renders `<|im_start|>system/user/assistant ... <|im_end|>`
   in order and ends by opening an assistant turn;
3. streaming yields >5 incremental deltas, first delta before completion, and
   the greedy stream equals the blocking reply;
4. a stop sequence ends output without leaking it (`finish_reason = Stop`);
   `max_tokens = 7` yields exactly 7 tokens and `Length`; a cap above the
   provider limit is `ResourceLimit`;
5. schema-constrained JSON parses, matches the schema, and the runtime's own
   validation reports it valid on both the blocking and streaming paths; a
   truncated result is reported invalid;
6. another model id is `ModelNotFound` (blocking and streaming); a wrong
   SHA-256 is `ModelIntegrity` at construction, before inference;
7. truncated, missing, non-GGUF files and an oversized context each fail with a
   distinct, readable error;
8. cancellation ends the stream with `Cancelled` in single-digit milliseconds,
   a dropped stream frees its slot, and a pre-cancelled request never starts.

## Apple Silicon results

`cargo run --release --example bench -- <model> 128 2048 <gpu_layers>` under
`/usr/bin/time -l`, greedy, context 2048, `max_tokens` 128, llama.cpp default
threads, flash attention `auto`.

| | CPU (`gpu_layers=0`) | Metal (`gpu_layers=99`) |
|---|---|---|
| Execution proven by llama.cpp's log | `offloaded 0/25 layers to GPU`, CPU weights/KV, `graph splits = 1`, no `MTL0` buffer | `offloaded 25/25 layers to GPU`, `MTL0_Mapped model buffer 462.96 MiB`, `MTL0 KV buffer`, `graph splits = 2` |
| Load (in-process, after first-ever shader build) | 0.30 s total: loader/mmap 0.26 s, SHA-256 not requested | 0.20 s total: loader 0.15 s |
| Warm reload (same process) | 0.33 s | 0.25 s |
| First request after load (weight first use) | 122 ms (8 tokens) | 76 ms (8 tokens) |
| Time to first token, 29-token prompt | 96-100 ms | 28-30 ms |
| Prompt processing, 1009-token prompt | 301-318 tok/s | 8,755-9,061 tok/s |
| Decode | 161.8-167.4 tok/s | 273.7-322.8 tok/s |
| Cancellation (stream end after `cancel()`) | 6.3-6.4 ms | 3.7-4.2 ms |
| Next request admitted and finished after cancel | 104-110 ms | 34-37 ms |
| Peak RSS / peak memory footprint | 816 MiB / 499 MiB | 629 MiB / 266 MiB |

Notes: three runs each; the ranges are the observed min-max. Decode on Metal
fell from 323 to 274 tok/s across three consecutive runs (not investigated;
thermal or scheduling). **Cold start:** the first-ever process on this machine
spent **17.5 s** at backend initialization compiling Metal shader libraries
(`fa` alone 16.8 s) **even with `gpu_layers = 0`**, because llama.cpp
initializes every compiled-in device; later processes paid ~0.2 s (macOS caches
the compiled libraries). A page-cache-cold read of the file was not measured
(needs `sudo purge`). RSS counts mmap pages touched; footprint is the process's
own accounting.

### Metal qualification

Metal is **qualified for this model on this machine**, and for nothing else:

- It builds. Note that `llama-cpp-sys-2` 0.1.159's `build.rs` never reads its
  own `metal` cargo feature: on every Apple target (except watchOS) llama.cpp
  is compiled with Metal and the Metal/MetalKit/Accelerate frameworks are
  linked regardless. Metal therefore cannot be excluded by feature selection,
  and a Mac build always pays Metal device initialization (above).
- The intended path is selected (log lines above) and `gpu_layers` is what
  selects it. The provider's notes only state what was *requested*; the
  `devices` list says what is *compiled in*.
- Real inference succeeds and the full real-model suite passes through it.
- Not qualified: other Macs/chips/macOS versions (one machine, a macOS 27.0 SDK
  and linker), other models or quantizations, long contexts, concurrent
  generations on the GPU, and the macOS Node addon (below).

## Node distribution boundary

`bindings/node-llama` (`ml-runtime-node-llama`, unpublished, standalone
workspace) is a **packaging probe**: napi bindings exposing only
`LocalGenerationClient` (`chatJson`, `describeJson`) and `LocalCancellation`,
using the existing `ChatRequest`/`ChatResponse` JSON. It adds no public API to
the shipped package. `bindings/node-llama/verify-bookworm.sh` reproduces the
results below.

| Question | Evidence |
|---|---|
| Reproducible build in Bookworm? | Yes. `bindings/node-llama/Dockerfile.bookworm` (same pinned Rust 1.90.0 / node 22 / glibc 2.36 images as the shipped addon) plus `cmake` 3.25.1, `clang` 14.0.6 and `libclang-dev` (bindgen); `g++` 12.2 from `build-essential`. Release build 1 m 49 s (aarch64, Docker Desktop VM on an M4 Max). Lockfile-pinned (`--locked`) |
| Extra dynamic libraries vs the shipped addon? | **None.** Both need `libstdc++.so.6`, `libgcc_s.so.1`, `libm`, `libc` and the loader, and both need `GLIBCXX_3.4.30`. OpenMP is off, so no `libgomp` |
| glibc baseline | `tools/check-glibc-compat.mjs --dependencies` passes (artifact max `GLIBC_2.34`; the 2.36 baseline holds). That gate does **not** inspect `GLIBCXX_*`; both addons need Bookworm's gcc-12 libstdc++, so Debian 11 and similar would fail at load. A `GLIBCXX` gate is needed before publishing |
| Loads in the supported Node runtime? | Yes, Node 22.23.2 |
| Clean environment, no undeclared libraries? | `node:22-bookworm-slim` with `--network none`, no `cmake`/`gcc`/`g++`/`clang`: `test/local.test.cjs` 9/9, including real inference, wrong model id, wrong digest, schema-constrained output and cancellation through Node |
| Size | 42.6 MB (33.1 MB stripped) vs 38.8 MB (29.9 MB stripped) for the shipped addon |
| `linux-x64-gnu` | Built and run the same way under **emulation** (`--platform linux/amd64` on an Apple Silicon host): same libraries, `GLIBCXX_3.4.30`, glibc gate passes, 9/9. The first emulated build died in a build script and succeeded on retry (not diagnosed). Emulation does not validate real x86 CPU features |
| `static-stdcxx` | Tried; it still links `libstdc++.so.6` dynamically (`cc` adds it for the wrapper), so it was removed rather than shipped |
| macOS arm64 | The addon builds (1 m 06 s) and links Metal/MetalKit/Accelerate, but **dyld refuses to load it**: `mis-aligned LINKEDIT string pool` (the linker, ld 27037.1, emitted an odd indirect-symbol count of 1103 and a 4-byte-aligned string table; stripping, `-dead_strip_dylibs`, `-no_deduplicate` did not help). A fresh build of the *shipped* addon from the same tree loads, so this is specific to this link. **Unqualified; unresolved** |
| Windows | Not attempted |
| Excluded when local inference is off? | Yes: separate crate, separate `Cargo.lock`, no feature on any shipped crate. `cargo tree` for the workspace (default and `--all-features`) and for `ml-runtime-node` contains no llama; `npm pack --dry-run` lists 3 files |

**Choice: Option A.** A separate, opt-in addon is the only option the evidence
supports without compromising existing releases. Option B would put llama.cpp
(minutes of compilation, CMake and libclang, a Metal link on macOS that cannot
be disabled) behind the shipped addon's `--all-features` gates; Option C is not
necessary because Linux packaging works.

## Remaining limitations

- Model: one 0.5B Q4_K_M model; no other architecture, quantization, template
  or context length is qualified. Quality of output is only checked behaviorally.
- Hardware: one Mac. No Intel Mac, no M1-M3, no other macOS, no Windows.
- Node: Linux only, probe level. There is no npm package, platform-package
  layout, loader, streaming API, or release job; `chatJson` is blocking
  request/response. The macOS addon does not load.
- Tools and tool choice are rejected (`CapabilityMismatch`).
- Metal cannot be excluded from a Mac build; first-ever launch pays ~17 s of
  shader compilation.
- Cancellation granularity is a prompt chunk (see above).

## Recommended next step

Build the packaging PR for **`linux-x64-gnu` only** as a separate opt-in
package (`@rust-ml-runtime/node-llama` plus a platform package), with a
Bookworm release job, a `GLIBCXX` baseline check beside the glibc one, a loader
with the same diagnostics as `index.cjs`, and a streaming handle. In parallel,
decide whether the macOS dyld rejection is a toolchain defect to work around
(or wait out) before any Mac Node claim. Do not promote the provider into the
default runtime.

## Reproducing

```text
# provider, generated fixtures, no downloads
cd providers/llama && cargo test --locked

# real model (see above), then the benchmark
cargo run --release --example bench -- model.gguf 128 2048 99
ML_RUNTIME_LLAMA_LOGS=1 ... # show llama.cpp's device/offload log

# Linux addon: build + inspect + clean-container test (arm64 natively, or amd64 emulated)
ML_RUNTIME_TEST_GGUF=... ML_RUNTIME_TEST_GGUF_SHA256=... \
  bindings/node-llama/verify-bookworm.sh [linux/amd64]

# isolation
node tools/check-local-engine-isolation.mjs
```

## Earlier measurements (Linux x86_64 CPU, generated fixture)

Kept for context only; they are random weights and say nothing about a real
model. `synthetic.rs` writes a deterministic llama-architecture GGUF with
seeded random weights and a tiny byte-level vocabulary (no licence, nothing
committed). `tiny` (~100k parameters) powers the tests; `100m` (85M parameters,
F16) powered a benchmark on a 4 vCPU Xeon: 107-115 tok/s decode, TTFT ~205 ms,
272 MiB peak RSS, cancellation observed in 13-15 ms.
