# Generative inference

`rust-ml-runtime` serves two complementary workloads through the same runtime:

| | Typed inference | Generative inference |
|---|---|---|
| Input | model + typed input + options | conversation messages + generation options |
| Output | typed result, probabilities, execution metadata | assistant message, finish reason, usage, execution metadata |
| API | `Runtime::infer` / `infer_stream`, `/v1/infer` | `Runtime::chat` / `chat_stream` |
| Provider trait | `Provider` | `GenerationProvider` |

They are separate public contracts. Existing `InferenceRequest`,
`InferenceResult`, typed decisions, the `/v1/infer*` protocol, and model
installation are unchanged. Generation providers live in their own registry
and never serve, or fall back for, typed requests.

OpenAI compatibility is a **protocol adapter**, not the runtime abstraction.
The OpenAI wire types are private to `providers/http/src/openai.rs`; the rest
of the runtime sees only `ChatRequest`, `ChatResponse`, and `ChatStreamEvent`.

## Rust

```rust
use rust_ml_runtime::{
    ChatMessage, ChatRequest, OpenAiCompatibleConfig, OpenAiCompatibleProvider, Runtime,
};

# async fn demo() -> rust_ml_runtime::RuntimeResult<()> {
let provider = OpenAiCompatibleProvider::new(
    OpenAiCompatibleConfig::new("http://127.0.0.1:11434/v1"), // Ollama
)?;
let runtime = Runtime::builder().register_generation_provider(provider).build();

let response = runtime
    .chat(
        "openai-compatible", // the caller names the provider; there is no routing
        ChatRequest::new("llama3.2", vec![
            ChatMessage::system("Answer in one sentence."),
            ChatMessage::user("Why is the sky blue?"),
        ]),
        None, // optional CancellationToken
    )
    .await?;
println!("{}", response.message.content);
println!("{:?} via {:?}", response.metadata.reported_model, response.metadata.execution);
# Ok(())
# }
```

`chat_stream` yields `Started`, zero or more `Delta`, then exactly one
`Completed(summary)`. Any failure or cancellation ends the stream with an error
and never with `Completed`; a stream that ends without either is reported as a
protocol error. Dropping the stream releases the upstream connection and the
runtime's concurrency permit. See `examples/rust/chat.rs`.

### Contract and honesty rules

- `ChatRequest.model` is required. Nothing substitutes another model.
- `finish_reason`, `usage`, `reported_model`, and `response_id` are `None`
  when the provider does not report them. Nothing is estimated. An
  unrecognized finish reason is preserved as `Other(raw)`.
- `ChatMetadata` is the execution evidence: provider, transport (`http`),
  `execution` (`local` or `remote`), the credential-free endpoint, requested
  and reported model, duration, time to first token, and runtime version.
  **Generated text is model output and is not execution evidence.**
- `execution` is `local` for loopback hosts and `remote` otherwise, and
  `execution_inferred` says so. Set `location` explicitly when that is wrong
  (for example a loopback SSH tunnel to a remote host).
- `ChatMessage`'s `Debug` output reports only the content length.

## OpenAI-compatible provider

Supports `GET {base}/models`, `POST {base}/chat/completions`, and SSE streaming.
`base` is the full API prefix such as `http://127.0.0.1:11434/v1`.

| Setting | Default | Notes |
|---|---|---|
| `api_key` | none | Bearer token. Debug-redacted, never serialized or returned. |
| `location` | inferred | `Local` / `Remote` override. |
| `capabilities` | streaming, stop, seed on; `json_object`, `json_schema` **off** | Optional features differ between servers and cannot be discovered; unsupported requests are rejected before any request is sent. |
| `connect_timeout` | 10 s | |
| `request_timeout` | 120 s | Whole operation for non-streaming calls and model listing; time to response headers for streams. |
| `stream_idle_timeout` | 60 s | Maximum silence between stream chunks. |
| `max_output_bytes` | 4 MiB | Bounds generated text. |
| `max_response_bytes` | 16 MiB | Bounds a non-streaming body. |
| `max_event_bytes` | 1 MiB | Bounds one SSE line/event. |
| `stream_buffer` | 32 | Bounded channel: a slow consumer applies backpressure to the socket. |

Safety properties:

- An endpoint may not embed credentials, a query, or a fragment, and an API key
  is refused over plain `http` to a non-loopback host.
- Redirects are not followed, so a bearer token is never re-sent elsewhere.
- The adapter does not log. Errors never include the key, request
  headers, or message contents; upstream error text is truncated.
- SSE input may be split at any byte boundary. A stream that ends before
  `[DONE]` or a `finish_reason` is a protocol error, not a completion.

### Errors

Failures reuse `RuntimeError`; no variants were added.

| Situation | Variant |
|---|---|
| Connection/DNS/TLS failure | `Transport` |
| Timeout (connect, request, stream idle) | `Timeout` |
| HTTP 400/404/405/409/413/415/422 (provider rejected the request) | `InvalidRequest` (404 with a model message: `ModelNotFound`) |
| HTTP 401/403, 429, 5xx, mid-stream `error` event | `ProviderUnavailable` |
| Unparseable or structurally invalid payload | `Protocol` |
| Option the provider is not declared to support | `CapabilityMismatch` |
| Output or response over a configured bound | `ResourceLimit` |
| Cancelled | `Cancelled` |

Model-generated content (including text that fails validation) is never an
error; it is returned in the response.

## Structured output

Request `ResponseFormat::JsonObject` or `ResponseFormat::JsonSchema { name,
schema, strict }`. The provider's native `response_format` is used; the runtime
then **validates the result itself**, because a request is not a guarantee.
`ChatResponse.structured` (and `ChatStreamSummary.structured` when streaming) is
`None` when no JSON format was requested, otherwise one of:

- `Validated { value }`
- `InvalidJson { error }`
- `SchemaViolation { errors }`

Text is returned verbatim; it is never repaired. Distinct from these, and
reported as errors: unsupported structured output (`CapabilityMismatch`), an
uncompilable schema (`InvalidRequest`, before sending), provider rejection
(`InvalidRequest`), and generation failure (`Timeout`, `ProviderUnavailable`,
and so on). A truncated generation (`finish_reason: length`) surfaces as
`InvalidJson` or `SchemaViolation`.

Validation uses the [`jsonschema`](https://crates.io/crates/jsonschema) crate
with default features disabled, so it neither fetches remote `$ref`s nor reads
files. Hand-rolling a validator would risk silently accepting schema keywords it
does not implement, which would present unvalidated output as verified. When
streaming with a JSON format the runtime buffers the text for validation, bounded
at 4 MiB.

## CLI

```text
ml-runtime chat --provider openai-compatible --endpoint http://127.0.0.1:11434/v1 \
  --model llama3.2 --system "Be terse." --message "Why is the sky blue?"
```

`--provider`, `--endpoint`, and `--model` are explicit and have no defaults or
fallback. Other flags: `--messages-json FILE|-` (full conversation),
`--temperature`, `--top-p`, `--max-tokens`, `--seed`, `--stop` (repeatable),
`--stream`, `--json`, `--list-models`, `--location local|remote`,
`--connect-timeout-secs`, `--timeout-secs`, `--idle-timeout-secs`.

Structured output: `--json-schema FILE [--schema-name N] [--strict]` or
`--json-object`, plus `--capability json-schema` / `json-object` to declare that
the endpoint supports it (also `no-streaming`, `no-stop`, `no-seed`).

Credentials: set `ML_RUNTIME_API_KEY`, or name another variable with
`--api-key-env NAME`. There is deliberately no `--api-key` flag, so keys do not
appear in process listings or shell history.

The answer goes to stdout; provider, transport, local/remote, model, finish
reason, usage (or `unreported`), and structured-output status go to stderr.
`--json` prints the full `ChatResponse` (NDJSON events with `--stream`).

Exit codes: `0` success; `1` failure (transport, provider, validation of
arguments); `2` usage error (clap); `3` the request succeeded but requested
structured output did not validate.

## Node and TypeScript

```js
const { GenerativeML } = require('@rust-ml-runtime/node');

const ml = new GenerativeML({ endpoint: 'http://127.0.0.1:11434/v1' });
const response = await ml.chat({
  model: 'llama3.2',
  messages: [{ role: 'user', content: 'Why is the sky blue?' }],
});

for await (const event of ml.chatStream({ model: 'llama3.2', messages })) {
  if (event.type === 'delta') process.stdout.write(event.content);
}
```

Transport, SSE parsing, validation, cancellation, and the credential all live in
the native addon; JavaScript exchanges JSON. Pass `apiKeyEnv` (preferred) or
`apiKey`; neither is retained in JavaScript, serialized, or included in errors.
Pass a `DecisionCancellation` as `{ cancellation }` to cancel a request or
stream. Breaking out of a `for await` loop, or calling `stream.close()`, aborts
the upstream request; an unreferenced stream is cancelled when collected.

Errors are `GenerationError` with `code` equal to the Rust variant category
(`timeout`, `transport`, `protocol`, `capability_mismatch`, `cancelled`, ...).
Types are in `index.d.ts`. No new npm package is introduced, and platform
packaging is unchanged.

## Scope

Not included: automatic provider routing, an interactive chat UI, tool/function
calling, multiple choices (`n > 1`), embeddings, and any agent loop.
Authorization and application policy remain the application's responsibility.
