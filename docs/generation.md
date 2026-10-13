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
| `capabilities` | streaming, stop, seed on; `json_object`, `json_schema`, `tools`, `tool_choice` **off** | Optional features differ between servers and cannot be discovered; unsupported requests are rejected before any request is sent. |
| `connect_timeout` | 10 s | |
| `request_timeout` | 120 s | Whole operation for non-streaming calls and model listing; time to response headers for streams. |
| `stream_idle_timeout` | 60 s | Maximum silence between stream chunks. |
| `max_output_bytes` | 4 MiB | Bounds generated text. |
| `max_response_bytes` | 16 MiB | Bounds a non-streaming body. |
| `max_event_bytes` | 1 MiB | Bounds one SSE line/event. |
| `max_tool_calls` | 32 | Maximum tool calls in one response. |
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
| Option the provider is not declared to support (including tools) | `CapabilityMismatch` |
| Model produced an unusable tool proposal (bad/undefined tool, arguments not a JSON object or not matching the schema) | `InvalidOutput` |
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

## Tool calling

Tool calling is a **protocol capability**: the caller offers tool definitions,
the model may answer with *proposals* to call them, and the caller sends back
results in a later request. **rust-ml-runtime never executes a tool**, grants
no authority, keeps no conversation state, and runs no loop. A proposal is not
evidence that an operation ran, was authorized, or succeeded; the model's
output is a proposal, not reality.

```text
caller ── tools + messages ──▶ runtime ──▶ provider ──▶ model
caller ◀── assistant message with tool_calls (proposals) ──
caller: validate → authorize → EXECUTE (application code, not the runtime)
caller ── messages + assistant tool_calls + tool result ──▶ runtime ──▶ …
caller ◀── final assistant answer ──
```

Working examples: `examples/rust/tool_calling.rs` and
`examples/typescript/tool-calling.ts`. Each runs a deterministic mock tool in
the application and marks authorization and execution as application-owned.

### Contract

- `GenerationOptions.tools: Vec<ToolDefinition>` — `name` (`[A-Za-z0-9_-]{1,64}`,
  unique), optional `description`, and `input_schema` (a JSON Schema object).
  The schema constrains proposed arguments; it is not authorization.
- `GenerationOptions.tool_choice: Option<ToolChoice>` — `Auto`, `None`, or
  `Tool { name }` (must name an offered tool). Absent means the provider default.
- `ChatMessage.tool_calls: Vec<ToolCall>` on assistant messages: `id`, `name`,
  and `arguments` (a parsed JSON object). A message may carry text, calls, or
  both. `ChatResponse.message` is where proposals arrive, with
  `FinishReason::ToolCalls` when the provider says so. Some servers report
  `stop` even when proposing calls; finish reasons are passed through, so check
  `tool_calls`, not the finish reason.
- `ChatRole::Tool` messages carry the caller-supplied result in `content`
  (serialize JSON yourself) and `tool_call_id`. Each must answer an earlier,
  still-unanswered assistant call; otherwise the request is an `InvalidRequest`
  before anything is sent. Replay the assistant turn unchanged.
- No hidden state: every request carries the full tool list and history.

### Validation and errors

Before sending, the runtime checks names, schemas (compiled with `jsonschema`),
the choice, and call/result correlation. After a response, a proposal is exposed
only if it is complete: it has an id and a valid tool name, its arguments are
complete JSON that is an object, the tool was actually offered, it satisfies
`tool_choice`, and it satisfies the tool's schema. Otherwise the whole response
is an error and nothing is returned as a proposal: `Protocol` for a malformed
wire payload (missing id/name, duplicate ids, non-string arguments),
`InvalidOutput` for a model-produced problem (undefined tool, malformed or
non-object arguments, schema violation, `tool_choice` violated),
`ResourceLimit` beyond `max_tool_calls`/`max_output_bytes`, and
`CapabilityMismatch` when the provider is not declared to support `tools` (or
`tool_choice`). Support cannot be discovered, so it is never assumed. Error
messages name locations in the arguments, never their values. `Debug` output of
`ToolCall` and messages omits arguments and content.

### Streaming

`chat_stream` adds two events to `Started` / `Delta` / `Completed`:

- `ToolCallDelta { index, id, name, arguments_fragment }` — a raw fragment while
  a call is being assembled, for display only. It is unvalidated and is never a
  proposal.
- `ToolCallCompleted { call }` — a fully assembled, validated proposal. These
  are released only after the whole response arrived and **every** call
  validated, in call order, immediately before `Completed`. If any call is bad,
  none is released and the stream ends with an error.

Id, name, and argument fragments are concatenated per `index` (calls may
interleave). A stream that ends before `[DONE]` or a finish reason, or whose
arguments are incomplete (including a `length` truncation), is an error.
Buffering, backpressure, idle timeouts, cancellation, and connection release are
unchanged; tool fragments count toward `max_output_bytes`.

### Provider limitations

- Support for `tools` and `tool_choice` must be declared (`--capability tools`,
  `--capability tool-choice`; `capabilities: { tools, toolChoice }` in Node).
  Servers differ: some ignore `tool_choice`, some stream a whole call in one
  chunk, and some report `stop` instead of `tool_calls`.
- Only function tools are supported. Arguments must arrive as a JSON-encoded
  string (as the OpenAI wire specifies); an empty string is treated as
  incomplete, not as `{}`.
- A server that repeats the whole id or name in every streamed chunk (instead of
  fragmenting it) yields a corrupted name/id, which is rejected as an undefined
  tool or failed correlation rather than guessed.
- Models can propose calls the application must refuse. Authorization, identity,
  rate limits, and execution policy remain the application's responsibility.

### Compatibility

The wire JSON is additive: text-only requests serialize exactly as before and
old JSON still deserializes. Rust source impacts (pre-1.0): `ChatRole`,
`FinishReason`, and `ChatStreamEvent` gained variants (exhaustive `match`es need
an arm); `ChatMessage`, `GenerationOptions`, and `GenerationCapabilities` gained
fields (use the constructors and `..Default::default()` instead of exhaustive
literals); `finish_reason: "tool_calls"` now maps to `FinishReason::ToolCalls`
instead of `Other("tool_calls")`.

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

Tool proposals can be requested with `--tool-file FILE` (a JSON array of
`{name, description?, input_schema}`), `--tool-choice auto|none|NAME`, and
`--capability tools` / `tool-choice`. The CLI prints each proposal to stderr as
`PROPOSED tool call (not executed, not authorized)`, never runs it, and does not
follow up; use `--messages-json` to submit a conversation containing tool results.

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

## Local models

A local model can be used today by pointing the provider at a local
OpenAI-compatible server. In-process engines were audited; see
[Local generation](local-generation.md).

## Scope

Not included: automatic provider routing, an interactive chat UI, tool
*execution* (see above), multiple choices (`n > 1`), embeddings, and any agent
loop.
Authorization and application policy remain the application's responsibility.
