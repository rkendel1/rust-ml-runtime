# Changelog

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
