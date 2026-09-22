#!/usr/bin/env bash
set -euo pipefail

version="${1:?usage: package-crates.sh VERSION}"
dirty=()
if [[ "${CARGO_PACKAGE_ALLOW_DIRTY:-}" == "1" ]]; then dirty+=(--allow-dirty); fi
crates=(
  ml-runtime-common
  ml-runtime-model
  ml-runtime-inference
  ml-runtime-backend
  ml-runtime-provider
  ml-runtime-protocol
  ml-runtime-cpu-backend
  ml-runtime-onnx-backend
  ml-runtime-coreml-backend
  ml-runtime-http-provider
  rust-ml-runtime
)

selection=()
for crate in "${crates[@]}"; do selection+=(-p "$crate"); done
cargo package "${selection[@]}" "${dirty[@]}"
cargo publish --dry-run "${selection[@]}" "${dirty[@]}"
node tools/validate-crates.mjs "$version" target/package
