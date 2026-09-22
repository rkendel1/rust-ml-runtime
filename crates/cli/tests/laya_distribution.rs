#![cfg(target_os = "macos")]

use std::{path::PathBuf, process::Command, time::Instant};

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
    let temporary = root.join("tmp");
    std::fs::create_dir_all(&empty_path).unwrap();
    std::fs::create_dir_all(&temporary).unwrap();
    let binary = env!("CARGO_BIN_EXE_ml-runtime");

    let install_started = Instant::now();
    let install = Command::new(binary)
        .args(["model", "install", "laya"])
        .env("ML_RUNTIME_MODEL_DIR", &root)
        .env("ML_RUNTIME_CACHE_DIR", root.join("cache"))
        .env("ML_RUNTIME_LAYA_SOURCE", &artifact)
        .env("PATH", &empty_path)
        .env("TMPDIR", &temporary)
        .output()
        .expect("run native model installer");
    let install_ms = install_started.elapsed().as_secs_f64() * 1_000.0;
    assert!(
        install.status.success(),
        "install failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&install.stdout),
        String::from_utf8_lossy(&install.stderr)
    );
    eprintln!("{}", String::from_utf8_lossy(&install.stdout));
    let package = root
        .join("laya")
        .join("fff78b2d9750c6b748fe8c90fcbf8bed0a1522a9");
    let installation: serde_json::Value = serde_json::from_slice(
        &std::fs::read(package.join("installation.json")).expect("installation manifest"),
    )
    .expect("parse installation manifest");
    assert_eq!(installation["compiled_status"], "ready");
    let compiled = PathBuf::from(
        installation["compiled_artifact"]["path"]
            .as_str()
            .expect("compiled artifact path"),
    );
    assert!(compiled.is_dir());
    let compiled_modified = std::fs::metadata(&compiled).unwrap().modified().unwrap();

    let run_inference = || {
        let started = Instant::now();
        let output = Command::new(binary)
            .args(["laya", "The customer asks for a refund."])
            .env("ML_RUNTIME_MODEL_DIR", &root)
            .env("ML_RUNTIME_CACHE_DIR", root.join("cache"))
            .env("PATH", &empty_path)
            .env("TMPDIR", &temporary)
            .output()
            .expect("run native Laya inference");
        (output, started.elapsed().as_secs_f64() * 1_000.0)
    };
    let (inference, first_ms) = run_inference();
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
    let mut warm = Vec::new();
    for _ in 0..5 {
        let (output, elapsed) = run_inference();
        assert!(
            output.status.success(),
            "warm inference failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        warm.push(elapsed);
    }
    warm.sort_by(f64::total_cmp);
    assert_eq!(
        std::fs::metadata(&compiled).unwrap().modified().unwrap(),
        compiled_modified,
        "post-install inference must not recompile the model"
    );
    let p50 = warm[warm.len() / 2];
    let p95 = warm[((warm.len() as f64 * 0.95).ceil() as usize - 1).min(warm.len() - 1)];
    eprintln!(
        "release verification: install_ms={install_ms:.3} first_ms={first_ms:.3} warm_p50_ms={p50:.3} warm_p95_ms={p95:.3}"
    );

    let hidden_compiled = compiled.with_extension("mlmodelc.missing");
    std::fs::rename(&compiled, &hidden_compiled).unwrap();
    let (missing, _) = run_inference();
    assert!(!missing.status.success());
    assert!(
        String::from_utf8_lossy(&missing.stderr).contains("ml-runtime model doctor laya"),
        "missing compiled artifact must produce actionable diagnostics"
    );
    assert!(
        !compiled.exists(),
        "inference must not rebuild compiled state"
    );
    std::fs::rename(&hidden_compiled, &compiled).unwrap();

    for args in [vec!["model", "list"], vec!["model", "doctor", "laya"]] {
        let output = Command::new(binary)
            .args(args)
            .env("ML_RUNTIME_MODEL_DIR", &root)
            .env("ML_RUNTIME_CACHE_DIR", root.join("cache"))
            .env("PATH", &empty_path)
            .env("TMPDIR", &temporary)
            .output()
            .expect("run model lifecycle command");
        assert!(output.status.success(), "lifecycle command failed");
    }
    let removed = Command::new(binary)
        .args(["model", "remove", "laya"])
        .env("ML_RUNTIME_MODEL_DIR", &root)
        .env("ML_RUNTIME_CACHE_DIR", root.join("cache"))
        .env("PATH", &empty_path)
        .env("TMPDIR", &temporary)
        .output()
        .expect("remove installed model");
    assert!(removed.status.success(), "model remove failed");
    assert!(!package.exists());
    assert!(!compiled.exists());
    let _ = std::fs::remove_dir_all(root);
}
