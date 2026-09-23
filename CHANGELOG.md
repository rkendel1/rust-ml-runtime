# Changelog

## 0.2.3

- Added explicit structured-decision capabilities that keep model architecture,
  backend, device, and execution strategy distinct.
- Added deterministic single, sequential, bounded-parallel, batched, and
  dependency-selective execution with application policy limits.
- Added cooperative cancellation, async Rust and Node execution, plan
  explanation, and factual execution diagnostics.
- Added planner, backend capability, generic hierarchical-decision, and real
  Laya planner integration coverage.
- Added observed execution benchmarks and a public capability matrix.

## 0.2.2

- Added the pinned, checksummed `receptron/laya-onnx` distribution for Linux
  x86_64 and ARM64 with deterministic platform/backend selection.
- Added a native Rust ONNX adapter for Laya tokenization, batched question
  tensors, cardinality calibration, and typed choice/score/noul results.
- Added real Rust and packaged Node inference coverage in pinned
  `node:22-trixie-slim` with glibc 2.41 and no Python dependency.
- Preserved the existing CoreML path as the macOS/ARM64 Laya variant.

## 0.2.1

- Rebuilt the Linux x64 Node addon in a controlled glibc 2.36 Bookworm image.
- Added direct and recursive ELF symbol-version compatibility gates for the
  packed npm artifact.
- Added clean `node:22-bookworm-slim` package install, native load, and local
  inference verification as a publication prerequisite.

## 0.2.0

- Hardened published Node native binding resolution and diagnostics.
- Added npm artifact validation and clean Linux optional-dependency coverage.
- Added native runtime self-test reporting for Node consumers.

## 0.1.0

- First Python-free native release of the `ml-runtime` CLI and Rust runtime.
- Added the public `rust-ml-runtime` crates.io package and packaged external
  Rust-consumer validation.
- Added explicit Laya installation, verification, diagnosis, prepared Core ML
  execution, offline inference, and removal on compatible macOS hosts.
- Added typed decisions, probabilities, latency, and model/runtime provenance.
- Added the public in-process `@rust-ml-runtime/node` package with native
  packages for the supported release matrix.
- Added verified native installers, release checksums, and machine-readable
  artifact metadata for macOS, Linux, and Windows runtime builds.
- Protocol `/v1`, model catalogs, local/remote execution, streaming,
  observability, and capability reporting remain available runtime surfaces.
