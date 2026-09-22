# Providers

Providers describe where inference is obtained.

Current boundaries:

- `local`: capability descriptor for embedded execution coordinated by the Rust runtime
- `remote` HTTP: boundary for remote APIs
- `server`: boundary for self-hosted daemons or socket-based services

Providers expose capability metadata so applications can inspect remote support without assuming identical semantics across transports.
# Capability providers

Providers are implementation adapters behind the runtime capability boundary.
Applications should discover a capability by ID and submit an
`ExecutionRequest`; they should not depend on a provider's concrete library.
Implementations can expose local, remote, or hybrid execution targets while
returning the common `ExecutionResult` evidence envelope.

Sensitive providers must require the application's `CapabilityAuthorizer`.
The runtime does not grant authority itself.

## Capability packages

Capability providers can be distributed as local packages:

```text
providers/
  postgres/
    manifest.toml
    provider
```

`manifest.toml` declares package/provider identity, runtime compatibility,
capabilities, targets, platforms, backend and configuration requirements, and
dependencies. The runtime validates this metadata and registers its capabilities
only when loading is explicitly requested; discovering a package never executes
the provider binary.

Inspect packages without loading them:

```sh
ml-runtime providers list --providers providers
ml-runtime providers inspect providers/postgres
```
