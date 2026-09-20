# Providers

Providers describe where inference is obtained.

Current boundaries:

- `local`: capability descriptor for embedded execution coordinated by the Rust runtime
- `remote` HTTP: boundary for remote APIs
- `server`: boundary for self-hosted daemons or socket-based services

Providers expose capability metadata so applications can inspect remote support without assuming identical semantics across transports.
