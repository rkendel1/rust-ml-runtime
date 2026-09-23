# Agent integration contract

- Rust is authoritative for model execution, Choice/Score/Noul semantics,
  validation, planning, cancellation, decoding, and diagnostics.
- Applications describe decisions, dependencies, and execution constraints.
  The runtime chooses a strategy from model/backend capabilities.
- Do not implement backend-specific inference policy in application code.
- Do not infer capabilities from artifact files or backend names.
- Do not assume batching is supported or that batch size is greater than one.
- Do not assume cancellation interrupts an in-flight native operation; inspect
  `supports_cancellation`.
- Do not duplicate Rust decision semantics or planning in TypeScript.
- Use `decision_capabilities`, `explain_decision`, `decide_async`, and
  `execute_decision_graph` in Rust, or their documented `LocalML` equivalents.
- Preserve the synchronous APIs for compatibility, but prefer async structured
  decisions in interactive Node applications.

The canonical API and capability matrix are in
[`runtime/execution-planning.md`](runtime/execution-planning.md).

