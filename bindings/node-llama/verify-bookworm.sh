#!/usr/bin/env bash
# Reproduces the packaging evidence for the opt-in local-inference addon:
#   1. builds it in the repository's Bookworm toolchain image (glibc 2.36),
#   2. inspects its dynamic dependencies and runs the repository's glibc gate,
#   3. loads it in a clean node:22-bookworm-slim container (no compilers, no
#      network) and runs test/local.test.cjs; with ML_RUNTIME_TEST_GGUF set
#      (a file already on disk; never downloaded) that includes real inference.
#
#   bindings/node-llama/verify-bookworm.sh [linux/arm64|linux/amd64]
#   ML_RUNTIME_TEST_GGUF=/path/model.gguf ML_RUNTIME_TEST_GGUF_SHA256=<hex> ...
set -euo pipefail
platform="${1:-}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$(mktemp -d)"
arch_flag=()
[[ -n "$platform" ]] && arch_flag=(--platform "$platform")
tag="rust-ml-node-llama-bookworm${platform:+-${platform#*/}}"
runtime_image="$(sed -n 's/^FROM \(node:[^ ]*\).*/\1/p' "$root/bindings/node-llama/Dockerfile.bookworm")"

docker build "${arch_flag[@]}" -f "$root/bindings/node-llama/Dockerfile.bookworm" -t "$tag" "$root"
docker run --rm "${arch_flag[@]}" \
  -v "$root":/workspace:ro -v "$out":/out \
  -v "ml-runtime-llama-target${platform:+-${platform#*/}}":/target -e CARGO_TARGET_DIR=/target \
  "$tag" bash -ceu '
    cd /workspace/bindings/node-llama
    cargo build --locked --release
    cp /target/release/libml_runtime_node_llama.so /out/ml_runtime_node_llama.node
    echo "--- NEEDED"; readelf -d /out/ml_runtime_node_llama.node | grep NEEDED
    echo "--- highest GLIBCXX"; readelf --version-info --wide /out/ml_runtime_node_llama.node | grep -o "GLIBCXX_[0-9.]*" | sort -uV | tail -1
    node /workspace/tools/check-glibc-compat.mjs --dependencies /out/ml_runtime_node_llama.node
  '
cp "$root/bindings/node-llama/test/local.test.cjs" "$out/"
model_args=()
if [[ -n "${ML_RUNTIME_TEST_GGUF:-}" ]]; then
  model_args=(-v "$(dirname "$ML_RUNTIME_TEST_GGUF")":/models:ro
    -e "ML_RUNTIME_TEST_GGUF=/models/$(basename "$ML_RUNTIME_TEST_GGUF")"
    -e "ML_RUNTIME_TEST_GGUF_SHA256=${ML_RUNTIME_TEST_GGUF_SHA256:-}")
fi
docker run --rm --network none "${arch_flag[@]}" -v "$out":/out:ro "${model_args[@]}" \
  -e ML_RUNTIME_LLAMA_ADDON=/out/ml_runtime_node_llama.node "$runtime_image" \
  bash -ceu '! command -v cmake gcc g++ clang >/dev/null && echo "clean image: no build tools"; node --test /out/local.test.cjs'
echo "artifact: $out/ml_runtime_node_llama.node"
