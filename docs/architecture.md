# Architecture

The runtime is intentionally boring at the center:

`load -> select -> execute -> return -> observe`

## Primary invariant

The application chooses the inference contract. The runtime chooses execution. Models, providers, and hardware remain replaceable.

## Core distinction

- **Model** = what is executed
- **Provider** = where inference is obtained
- **Backend** = how computation executes
- **Runtime** = coordinates them

## Current scaffold

- Rust runtime authority lives in `crates/runtime`
- Shared API types live in `crates/model`, `crates/inference`, `crates/backend`, `crates/provider`, and `crates/common`
- CPU execution works in CI through `backends/cpu`
- Core ML, ONNX, CUDA, and WebGPU crates establish execution boundaries without forcing hardware-specific CI
- Local, HTTP, and server providers establish local/remote/hybrid selection boundaries
