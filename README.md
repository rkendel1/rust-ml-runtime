# rust-ml-runtime

A general-purpose, Rust-native ML runtime for application and agent workloads.

The runtime keeps the central boundary deliberately small:

```text
Application
     |
     v
ML Runtime API
     |
     v
Rust Runtime
     |
 ┌───┼───────────────┐
 v   v               v
Local             Remote          Hybrid
```

Primary invariant:

> The application chooses the inference contract. The runtime chooses execution. Models, providers, and hardware remain replaceable.
