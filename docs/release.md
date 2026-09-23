# Release process

A `v*` tag builds versioned CLI archives for macOS arm64/x64, Linux arm64/x64,
and Windows x64. The same matrix builds one npm native package per platform;
the platform-neutral `@rust-ml-runtime/node` package selects one through npm
optional dependencies.

The release also packages `rust-ml-runtime` and its required implementation
crates for crates.io. Rust applications install only the public crate:

```toml
[dependencies]
rust-ml-runtime = "0.1"
```

CLI, Node, benchmark, server-integration, and placeholder backend workspace
members remain private.

Every archive has a `.sha256` sidecar. `release-manifest.json` records the
artifact name, independently versioned runtime/package version, product,
platform, architecture, and SHA-256. Publication is gated on formatting,
Clippy warnings, workspace tests, Cargo package audits, crates.io dry-runs, a
clean packaged external Rust consumer, native artifact smoke tests, checksum
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

The release job runs `npm run publish:packages`, which publishes the required
Rust support crates one at a time in dependency order (allowing crates.io to
index each dependency) followed by `rust-ml-runtime`, then native npm packages
followed by the public wrapper. GitHub release artifacts are uploaded only
after registry publication succeeds. `CARGO_REGISTRY_TOKEN` and `NPM_TOKEN`
must have the corresponding publish access. Registry preflight rejects any
existing immutable version before the first publication attempt.

Pushing a `v*` tag publishes automatically. A manually triggered workflow run
builds and verifies artifacts without publishing by default; set its
`publish` input to `true` to opt into publication for the selected `version`.

For an authorized local release, export `CARGO_REGISTRY_TOKEN` and
`NODE_AUTH_TOKEN`, collect the release artifacts under `artifacts/`, and run:

```sh
npm run publish:packages
```

The version defaults to the root `package.json`; `--version X.Y.Z` and
`--artifacts PATH` override the defaults.
