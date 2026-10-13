# Local generation: Uzu audit and the supported path

**Status: no-go for embedding Uzu in-process in this repository today; go for
using a local Uzu server through the existing OpenAI-compatible provider.** No
runtime code changes accompany this document. The audit below explains why and
what would change the decision.

Audited: `rkendel1/uzu` at `432da1b` ("Remove f32 attention (#946)"), version
0.6.3, MIT. Evidence is source reading plus a build experiment on Linux x86_64.
**No model could be loaded and no generation was run**: Hugging Face and the Mirai
registry are unreachable from the audit sandbox, and there is no Apple Silicon
hardware. Anything below that depends on running a model is marked *unverified*.

## What Uzu is

- `crates/uzu-engine`: a low-level engine. `Engine::<B>::new()` →
  `load_language_model(model_dir)` → `LanguageModel::stream(&[u64] tokens,
  &mut LanguageModelState, options)`, a **token-in / token-out** pull iterator
  borrowing the model and state. It does no chat templating and no tool parsing.
- `crates/legacy/*` (`uzu`, `hanashi`, `shoji`, `nagare`, `download-manager`,
  registry, CLI, …): the chat layer — Jinja chat templates, tool-call parsing,
  registry/downloads, an OpenAI-compatible server. The workspace itself labels
  these "to be rewritten".
- Backends: `cpu` and `metal`, selected at runtime (`UZU_BACKEND`). Metal is
  compiled only for macOS/iOS-family targets; on Linux only `cpu` exists.

## Findings against the 12 audit questions

| # | Question | Finding |
|---|---|---|
| 1 | Embeddable? | Yes as a Rust library (`uzu-engine`, or `uzu`). It is **not on crates.io**; the documented dependency is a git dependency. It also ships a standalone OpenAI-compatible server (`cli server`). |
| 2 | Formats | Only Mirai's converted layout: `config.json` (strict `decoder_config`), `model.safetensors` (weights loaded as BF16), `tokenizer.json`. Not GGUF; not a stock Hugging Face checkpoint. Models come through Mirai's registry/conversion. |
| 3 | Offline after acquisition | Engine: yes (reads a local directory). The `uzu` layer downloads from `sdk.trymirai.com` / Hugging Face only when asked. |
| 4 | Platforms | Builds for macOS/iOS, Linux, Windows, wasm in its toolchain file. **Verified here**: `uzu-engine` (`cpu` feature) compiles on Linux x86_64 (nightly, 2m24s; also stable 1.97 with `--ignore-rust-version`). |
| 5 | Apple Silicon acceleration | Real Metal backend in code (generated kernels, unified memory). *Unverified*: no Mac available. This is direct Metal, **not** Core ML. |
| 6 | Load/unload | Load is synchronous (`load_language_model`); unload is drop. No cancellation of loading. |
| 7 | Prompt / multi-turn | Not in the engine. Templates and tool markup live in legacy `hanashi`; each model family needs its own template. |
| 8 | Streaming | Yes at token level (iterator) and as SSE in the server. *Unverified at runtime.* |
| 9 | Cancellation | Between tokens only; the server stops "at the next token" on client disconnect or cancel. Loading and prefill are not interruptible. |
| 10 | Memory / context / concurrency | `peak_memory_usage()` is `None` on CPU. `ContextOverflow` is an error. The server holds one loaded model; per-request concurrency is serialized by the server (read from code, *unverified*). |
| 11 | Structured output | Yes, grammar-constrained decoding via xgrammar for `json_object` and `json_schema`, behind the `grammar` feature (server: `capability-grammar`). |
| 12 | Tool calls | Prompt-level: tools are injected into the template and the model's markup is parsed. The server **coerces** parsed argument types toward the declared schema (`coerce_tool_call`). |

Further findings that bear on the decision:

- The CPU backend is a reference implementation: `ssd_prefill` contains
  `todo!()` (hybrid/Mamba models would panic on CPU), unit tests deliberately
  skip CPU for several kernels, and no CPU performance was measured.
- The server ignores the request's `model` field (`#[allow(dead_code)]`): the
  model is fixed when the server starts. It does not support `stop` or a
  per-request `seed` (seed is a server flag).
- `rust-version = 1.98` and edition 2024 are above this repository's stable
  toolchain (1.97) today.
- Dependency footprint: 128 crates for `uzu-engine` on CPU (138 with grammar),
  all MIT / Apache-2.0 / Unlicense / Zlib / BSD — compatible with the repository's
  MIT license. Model-weight licenses are separate and per-model.

## Runtime requirements vs. Uzu

| Requirement | Result |
|---|---|
| Implement `GenerationProvider` with prompts, roles, tool history | **Not satisfiable narrowly**: the engine is token-level; faithful prompt/tool-history rendering means depending on unstable legacy crates or reimplementing chat templates per model family. |
| Ship in a published crate / the npm addon | **Blocked**: a git-only dependency cannot be a dependency (even optional) of any crate published to crates.io, which this repository's release tooling does for `rust-ml-runtime`. Avoiding it needs release-boundary changes that are out of scope. |
| Local streaming with safe cancellation | Plausible (token iterator, cancel between tokens) but **unverifiable** here, and the borrowed `&mut` state forces a dedicated worker thread per generation. |
| Accurate capabilities, explicit model selection | Server path: possible with the declarations below; note it does not honour `model`. |
| Deterministic tests without model downloads | Engine tests need a downloaded model (`TEST_MODEL`, default Llama-3.2-1B); there is no synthetic fixture. An adapter could only be tested with mocks of its own seam, not of Uzu. |
| Metal / CPU target honesty | CPU is a reference backend; Metal is unverifiable here. Nothing could be claimed. |

**Decision:** an in-process adapter would be unverifiable (no model, no Metal),
unpublishable (git-only dependency), and coupled to crates the maintainers plan
to rewrite. Per the brief, the integration is stopped rather than disguised.

## Supported path today: Uzu's server through the existing provider

The OpenAI-compatible provider already speaks to Uzu's server. Local execution,
explicit model choice (at server start), streaming, and constrained JSON are
covered without new runtime code, and no hidden daemon is introduced: you start
the server yourself.

```text
# Start Uzu (model acquisition is Uzu's job; the runtime never downloads weights)
# (from a checkout of uzu; the binary is the `cli` package, built with the capability-grammar feature for JSON constraints)
cargo run --release -p cli -- server --model <model-id-or-local-dir> --host 127.0.0.1 --port 8000

# Use it from rust-ml-runtime
ml-runtime chat --provider openai-compatible --endpoint http://127.0.0.1:8000/v1 \
  --model <same-model-id> --location local \
  --capability json-object --capability json-schema \
  --capability tools --capability tool-choice \
  --capability no-stop --capability no-seed \
  --message "Why is the sky blue?" --stream
```

The same declarations apply in Rust (`config.capabilities.stop_sequences =
false; … seed = false; json_schema = true; …`) and Node
(`capabilities: { stopSequences: false, seed: false, jsonObject: true,
jsonSchema: true, tools: true, toolChoice: true }`).

Derived from Uzu's source and **not run** in this audit — verify on your machine:

- `location` is inferred `local` for a loopback host; set it explicitly if you
  tunnel to another machine.
- Because the server ignores `model`, check `metadata.reported_model` rather than
  assuming the requested model answered.
- Structured output still passes through the runtime's own validation; invalid
  output stays invalid. Tool-call proposals still pass the runtime's schema and
  tool-choice checks, but note Uzu has already coerced argument types
  server-side, so arguments are "as normalized by Uzu".
- Cancelling or dropping a stream closes the connection; Uzu stops at the next
  token. Loading and prefill are not interruptible.

## Benchmark procedure (not run)

No benchmark was run: no model weights are reachable from the sandbox and there is
no Apple Silicon hardware. To produce comparable numbers on a Mac:

1. Fix one model and quantization (record exact id, quantization, size on disk).
2. Start the server as above and record load time (process start → first
   successful `/v1/models`).
3. For a fixed prompt and `max_tokens` (e.g. 128, `temperature` 0), run
   `ml-runtime chat … --stream --json` and read `time_to_first_token`, `duration`,
   and `usage.completion_tokens` from the final event. Time-to-first-token
   includes prompt processing; sustained speed is
   `completion_tokens / (duration − time_to_first_token)`.
4. Record peak memory from the OS (`/usr/bin/time -l` on the server process),
   the context length used, and which backend the server reports at startup
   (`UZU_BACKEND`).
5. Test cancellation by dropping the stream mid-generation and checking that the
   server's CPU/GPU activity stops.

Do not compare results across different models or quantizations as backend
performance.

## What would reopen the in-process option

Any one of: `uzu-engine` (or a chat-level crate) published to crates.io with a
stable API; or a maintained, verifiable adapter seam where the repository can
run a small model in CI. Until then the smallest viable embedded alternatives
are crates that are published and ship a CPU path with GGUF or Hugging Face
checkpoints (for example `llama-cpp-2` or `candle`); **neither was audited here**
and each must pass the same checklist (publishability, cancellation, grammar /
structured output, model-format availability, and a model that CI can run).
