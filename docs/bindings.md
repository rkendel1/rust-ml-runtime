# Bindings

The repository includes TypeScript bindings in `bindings/typescript`.

The binding layer is intentionally thin: it shells into the Rust CLI for capability discovery and inference so the Rust runtime remains the semantic authority.

This keeps other languages from reimplementing runtime selection, capability handling, or execution semantics.
