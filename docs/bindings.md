# Bindings

The repository includes TypeScript bindings in `bindings/typescript`.

The binding layer is intentionally thin: it shells into the Rust CLI for
catalog operations, capability discovery, inference, and normalized streaming,
so the Rust runtime remains the semantic authority. The executable is a visible
deployment dependency and may be configured with `binaryPath`.

This keeps other languages from reimplementing runtime selection, capability handling, or execution semantics.

TypeScript model references may be `"name@version"` for compatibility or
`{ id: "name", version: "version" }`. Inputs are typed text/tensor values;
results expose the Rust output enum and execution metadata. `RuntimeClientError`
provides structured process/protocol bridge failures. It does not translate
Rust errors into a divergent TypeScript taxonomy.

The subprocess bridge currently supports inference and streaming. Rust batch
and cancellation-token APIs are not projected and are not emulated in
TypeScript. See `docs/api.md` and `examples/typescript/mnist.ts`.
