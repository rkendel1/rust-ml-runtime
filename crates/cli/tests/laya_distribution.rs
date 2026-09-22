#![cfg(target_os = "macos")]

use std::{path::PathBuf, process::Command};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn installs_and_executes_laya_without_python_on_path() {
    let requested = std::env::var_os("LAYA_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/laya"));
    let artifact = if requested.is_absolute() || requested.exists() {
        requested
    } else {
        workspace_root().join(requested)
    };
    if !artifact.exists() {
        eprintln!("skipping distribution smoke test: no local Laya artifact");
        return;
    }
    assert!(artifact.join("model.mlpackage/Manifest.json").is_file());

    let root = std::env::temp_dir().join(format!(
        "ml-runtime-laya-distribution-{}",
        std::process::id()
    ));
    let empty_path = root.join("empty-path");
    std::fs::create_dir_all(&empty_path).unwrap();
    let binary = env!("CARGO_BIN_EXE_ml-runtime");

    let install = Command::new(binary)
        .args(["model", "install", "laya"])
        .env("ML_RUNTIME_MODEL_DIR", &root)
        .env("ML_RUNTIME_CACHE_DIR", root.join("cache"))
        .env("ML_RUNTIME_LAYA_SOURCE", &artifact)
        .env("PATH", &empty_path)
        .output()
        .expect("run native model installer");
    assert!(
        install.status.success(),
        "install failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&install.stdout),
        String::from_utf8_lossy(&install.stderr)
    );
    eprintln!("{}", String::from_utf8_lossy(&install.stdout));

    let inference = Command::new(binary)
        .args(["laya", "The customer asks for a refund."])
        .env("ML_RUNTIME_MODEL_DIR", &root)
        .env("ML_RUNTIME_CACHE_DIR", root.join("cache"))
        .env("PATH", &empty_path)
        .output()
        .expect("run native Laya inference");
    let stdout = String::from_utf8_lossy(&inference.stdout);
    assert!(
        inference.status.success(),
        "inference failed:\nstdout={stdout}\nstderr={}",
        String::from_utf8_lossy(&inference.stderr)
    );
    for expected in [
        "model: laya",
        "backend: coreml",
        "decision: refund (noul)",
        "latency_ms:",
        "model_revision:",
        "artifact_hash:",
    ] {
        assert!(
            stdout.contains(expected),
            "missing {expected:?} in:\n{stdout}"
        );
    }
    eprintln!("{stdout}");
    let _ = std::fs::remove_dir_all(root);
}
