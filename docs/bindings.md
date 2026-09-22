# Bindings

The repository includes TypeScript bindings in `bindings/typescript`.

The separately packaged `@rust-ml-runtime/node` module contains the in-process
Node-API bridge and platform-specific native artifact. Its `LocalML` class
caches loaded models and accepts only model-neutral state plus typed decisions.
Core ML types, Laya tokenization, tensor names, and compiled cache paths are not
part of its public contract. Building/publishing this package is independent of
the standalone `ml-runtime` executable; the executable itself has no Node
dependency.

The Node-API binding loads the Rust runtime in-process for typed decisions. It
does not shell into the CLI. The CLI remains the authoritative, explicit model
installation and diagnostic surface; the Node package consumes READY models.

This keeps other languages from reimplementing runtime selection, capability handling, or execution semantics.

TypeScript model references may be `"name@version"` for compatibility or
`{ id: "name", version: "version" }`. Inputs are typed text/tensor values;
results expose the Rust output enum and execution metadata. `RuntimeClientError`
provides structured process/protocol bridge failures. It does not translate
Rust errors into a divergent TypeScript taxonomy. See [Node and
TypeScript](node.md) for the packaged `LocalML` API.

The subprocess bridge currently supports inference and streaming. Rust batch
and cancellation-token APIs are not projected and are not emulated in
TypeScript. See `docs/api.md` and `examples/typescript/mnist.ts`.
