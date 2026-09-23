# Install the native runtime

On macOS or Linux, the installer selects the current OS and CPU, verifies the
release SHA-256 before extraction, and installs one native executable:

```sh
curl --proto '=https' --tlsv1.2 -sSfL \
  https://raw.githubusercontent.com/rkendel1/rust-ml-runtime/main/install.sh | sh
```

On Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/rkendel1/rust-ml-runtime/main/install.ps1 | iex
```

Set `ML_RUNTIME_VERSION` to install a particular release and
`ML_RUNTIME_INSTALL_DIR` to choose the destination. The scripts fail closed if
the archive, checksum, checksum format, or archive contents are invalid. For a
manual installation, download the archive from GitHub Releases, verify it
against `SHA256SUMS` or `release-manifest.json`, extract it, and place
`ml-runtime` on `PATH`. Release archives
contain a native executable; Rust, Cargo, Python, pip, Node, and the Hugging
Face CLI are not runtime dependencies.

Supported release targets are macOS arm64/x64, Linux arm64/x64, and Windows
x64. Laya uses ONNX Runtime on Linux arm64/x64 and CoreML on macOS arm64.
Other platform/backend pairs report an explicit incompatibility without
downloading a model.

## Install and run Laya

```sh
ml-runtime --version
ml-runtime model install laya
ml-runtime model doctor laya
ml-runtime laya "The customer asks for a refund."
```

Laya runs locally through the native Rust ML runtime. Python is not required.
Installation downloads the pinned model with native HTTPS, verifies every
registered SHA-256, publishes the selected platform bundle atomically,
initializes and validates its backend, and writes `installation.json` only
after the installation is ready. The model is downloaded separately and is
not embedded in the executable.

## Inspect and diagnose

```sh
ml-runtime model list
ml-runtime model doctor laya
ml-runtime model remove laya
```

Normal inference never compiles or repairs a model. If the prepared artifact is
missing or invalid, inference fails and directs the user to `model doctor`.
Re-run installation with `--replace` to repair a diagnosed installation.

## Storage and versioning

The model root defaults to:

- macOS: `~/Library/Application Support/ml-runtime/models`
- Linux: `~/.local/share/ml-runtime/models`
- Windows: `%LOCALAPPDATA%/ml-runtime/models`

Set `ML_RUNTIME_MODEL_DIR` to override it. Re-creatable compiled Core ML state
defaults to `~/Library/Caches/ml-runtime/coreml` on macOS; set
`ML_RUNTIME_CACHE_DIR` to override it. The installation manifest binds the
model revision and checksum to the runtime version, backend, OS, architecture,
and prepared-artifact identity. The authoritative model bundle remains
separate from disposable caches.

Laya is distributed through ONNX Runtime on Linux and CoreML on macOS/ARM64.
The runtime backend boundary keeps the typed-decision contract identical.

## Local and offline behavior

The runtime is local. The model becomes local after explicit installation, and
inference is local with no cloud inference or Python process. Model installation
may access its configured artifact source. Once READY, loading and inference do
not require the network and never silently download, compile, or repair state.

## Platform support

| Platform | Native runtime | Laya backend |
| --- | --- | --- |
| macOS arm64 | Supported | Supported on macOS 15+ |
| macOS x64 | Supported | Unavailable (no registered Laya variant) |
| Linux x64 | Supported | ONNX Runtime CPU |
| Linux arm64 | Supported | ONNX Runtime CPU |
| Windows x64 | Supported | Unavailable |

Use `ml-runtime model doctor laya` as the first diagnostic command. Unsupported
platforms reject Laya installation before downloading.
