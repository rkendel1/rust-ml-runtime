# Observability

Every successful `InferenceResult` includes `ExecutionMetadata` describing
identity, routing, provider/transport, backend, target, fallback, cache/batch/
stream state, and available timings. This is the request-level public surface.

`Runtime::metrics`, `Runtime::executions`, `Runtime::status`, and
`Runtime::snapshot` provide application-owned aggregate/runtime views. They do
not expose backend sessions, cache keys, request contents, or resource-manager
internals. No telemetry is sent automatically.

See [execution metadata semantics](api.md#execution-policies-and-metadata).
