# Release process

A `v*` tag builds versioned CLI archives for macOS arm64/x64, Linux arm64/x64,
and Windows x64. The same matrix builds one npm native package per platform;
the platform-neutral `@rust-ml-runtime/node` package selects one through npm
optional dependencies.

Every archive has a `.sha256` sidecar. `release-manifest.json` records the
artifact name, independently versioned runtime/package version, product,
platform, architecture, and SHA-256. Publication is gated on formatting,
Clippy warnings, workspace tests, native artifact smoke tests, checksum
verification, a clean npm consumer, and the real macOS Laya lifecycle. The
Laya gate installs and diagnoses the model, runs inference offline twice, and
asserts that inference did not mutate the compiled Core ML artifact.

The release also includes `release-measurements.json`, produced from the
packaged macOS artifact after model installation. It records installation,
prepared-model loading, first inference, warm P50, and warm P95 measurements.
These are observations from that release run, not performance guarantees.

## Release boundary

The mandatory gate runs from a clean checkout of this repository. It does not
checkout FeltDB, inspect a sibling workspace, or import Jev source. The
healthcare fixture is a standalone Node application using only
`@rust-ml-runtime/node`; it demonstrates structured model inference while
keeping application policy outside the model.

Jev is a downstream consumer of the public Node package. Jev compatibility and
healthcare application integrations belong in their owning downstream
workflows after a runtime version is published. They may report compatibility
issues, but unpublished downstream source never blocks creation of runtime
artifacts.

Release archives do not contain Laya. Model revisions remain independently
pinned by the model registry and their READY installation manifest remains
valid only when its runtime requirement and artifact identities match.

## Publishing

The release job publishes native npm packages before the public wrapper, then
uploads archives, npm tarballs, checksum sidecars, `SHA256SUMS`, and the JSON
manifest to GitHub Releases. `NPM_TOKEN` must have publish access to the
`@rust-ml-runtime` scope. A missing token fails the release rather than
silently producing a partial developer product.
