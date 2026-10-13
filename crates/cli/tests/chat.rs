//! `ml-runtime chat` end to end against an in-process mock server.

#[path = "../../../providers/http/tests/support/mod.rs"]
mod support;

use serde_json::json;
use std::{process::Output, time::Duration};
use support::{Mock, Reply};

const SECRET: &str = "sk-cli-SECRET-456";

fn completion(content: &str) -> String {
    json!({"model":"m","choices":[{"message":{"role":"assistant","content":content},"finish_reason":"stop"}],
           "usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}})
    .to_string()
}

async fn chat(mock: &Mock, args: &[&str], env: &[(&str, &str)]) -> Output {
    let endpoint = format!("{}/v1", mock.url);
    let args: Vec<String> = [
        "chat",
        "--provider",
        "openai-compatible",
        "--endpoint",
        &endpoint,
    ]
    .into_iter()
    .map(str::to_owned)
    .chain(args.iter().map(|a| (*a).to_owned()))
    .collect();
    let env: Vec<(String, String)> = env
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    tokio::task::spawn_blocking(move || {
        std::process::Command::new(env!("CARGO_BIN_EXE_ml-runtime"))
            .args(args)
            .env_remove("ML_RUNTIME_API_KEY")
            .envs(env)
            .output()
            .unwrap()
    })
    .await
    .unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[tokio::test]
async fn single_request_prints_content_and_honest_diagnostics() {
    let mock = Mock::start(|_| Reply::Json(200, completion("hello there"))).await;
    let out = chat(
        &mock,
        &[
            "--model",
            "m",
            "--system",
            "sys",
            "--message",
            "hi",
            "--max-tokens",
            "9",
        ],
        &[],
    )
    .await;
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "hello there\n");
    let stderr = text(&out.stderr);
    assert!(
        stderr.contains("execution=Local (inferred from host)"),
        "{stderr}"
    );
    assert!(stderr.contains("transport=http") && stderr.contains("provider=openai-compatible"));
    assert!(stderr.contains("total=3"));
    let body = mock.captured()[0].json();
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][1]["content"], "hi");
    assert_eq!(body["max_tokens"], 9);
}

#[tokio::test]
async fn api_key_comes_from_the_environment_and_is_never_printed() {
    let mock =
        Mock::start(|_| Reply::Json(401, json!({"error":{"message":"denied"}}).to_string())).await;
    let out = chat(
        &mock,
        &[
            "--model",
            "m",
            "--message",
            "hi",
            "--api-key-env",
            "MY_KEY",
            "--json",
        ],
        &[("MY_KEY", SECRET)],
    )
    .await;
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        mock.captured()[0].header("authorization"),
        Some(format!("Bearer {SECRET}").as_str())
    );
    assert!(!text(&out.stdout).contains(SECRET) && !text(&out.stderr).contains(SECRET));

    // The default variable is honoured too.
    let out = chat(
        &mock,
        &["--model", "m", "--message", "hi"],
        &[("ML_RUNTIME_API_KEY", SECRET)],
    )
    .await;
    assert_eq!(out.status.code(), Some(1));
    assert!(!text(&out.stderr).contains(SECRET));

    // Naming an unset variable fails before any request is made.
    let before = mock.captured().len();
    let out = chat(
        &mock,
        &[
            "--model",
            "m",
            "--message",
            "hi",
            "--api-key-env",
            "NOPE_UNSET",
        ],
        &[],
    )
    .await;
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("NOPE_UNSET"));
    assert_eq!(mock.captured().len(), before);
}

#[tokio::test]
async fn json_output_has_provenance_and_no_credentials() {
    let mock = Mock::start(|_| Reply::Json(200, completion("x"))).await;
    let out = chat(
        &mock,
        &[
            "--model",
            "m",
            "--message",
            "hi",
            "--json",
            "--api-key-env",
            "K",
        ],
        &[("K", SECRET)],
    )
    .await;
    assert_eq!(out.status.code(), Some(0));
    let stdout = text(&out.stdout);
    assert!(!stdout.contains(SECRET));
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["metadata"]["execution"], "local");
    assert_eq!(value["metadata"]["requested_model"], "m");
    assert_eq!(value["message"]["content"], "x");
}

#[tokio::test]
async fn failures_exit_nonzero_with_a_diagnostic() {
    let mock =
        Mock::start(|_| Reply::Json(400, json!({"error":{"message":"bad"}}).to_string())).await;
    let out = chat(&mock, &["--model", "m", "--message", "hi"], &[]).await;
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stdout).is_empty());
    assert!(text(&out.stderr).contains("provider rejected request"));

    // Unreachable endpoint.
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ml-runtime"))
        .args([
            "chat",
            "--provider",
            "openai-compatible",
            "--endpoint",
            "http://127.0.0.1:1/v1",
            "--model",
            "m",
            "--message",
            "x",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("transport error"));
}

#[tokio::test]
async fn usage_errors_exit_2_and_send_nothing() {
    let mock = Mock::start(|_| Reply::Json(200, completion("x"))).await;
    for args in [
        vec!["--model", "m", "--message", "a", "--messages-json", "f"],
        vec!["--list-models", "--message", "a"],
    ] {
        let out = chat(&mock, &args, &[]).await;
        assert_eq!(out.status.code(), Some(2));
    }
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ml-runtime"))
        .args([
            "chat",
            "--endpoint",
            "http://127.0.0.1:1/v1",
            "--model",
            "m",
            "--message",
            "x",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2), "provider must be explicit");
    // Missing model / message are reported as failures before any request.
    assert_eq!(
        chat(&mock, &["--message", "a"], &[]).await.status.code(),
        Some(1)
    );
    assert_eq!(
        chat(&mock, &["--model", "m"], &[]).await.status.code(),
        Some(1)
    );
    assert!(mock.captured().is_empty());
}

#[tokio::test]
async fn streaming_prints_deltas_as_they_arrive() {
    let events: Vec<Vec<u8>> = [
        json!({"choices":[{"delta":{"content":"Hel"}}]}).to_string(),
        json!({"choices":[{"delta":{"content":"lo"}}]}).to_string(),
        json!({"choices":[{"delta":{},"finish_reason":"stop"}]}).to_string(),
        "[DONE]".to_owned(),
    ]
    .iter()
    .map(|e| format!("data: {e}\n\n").into_bytes())
    .collect();
    let chunks = events.clone();
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: chunks.clone(),
        pause: Duration::from_millis(5),
        hold_open: false,
    })
    .await;
    let out = chat(&mock, &["--model", "m", "--message", "hi", "--stream"], &[]).await;
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "Hello\n");

    let out = chat(
        &mock,
        &["--model", "m", "--message", "hi", "--stream", "--json"],
        &[],
    )
    .await;
    let lines: Vec<serde_json::Value> = text(&out.stdout)
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.first().unwrap()["type"], "started");
    assert_eq!(lines.last().unwrap()["type"], "completed");
    assert_eq!(lines.iter().filter(|l| l["type"] == "delta").count(), 2);
}

#[tokio::test]
async fn truncated_streams_fail() {
    let chunk = format!(
        "data: {}\n\n",
        json!({"choices":[{"delta":{"content":"par"}}]})
    )
    .into_bytes();
    let mock = Mock::start(move |_| Reply::Sse {
        chunks: vec![chunk.clone()],
        pause: Duration::ZERO,
        hold_open: false,
    })
    .await;
    let out = chat(&mock, &["--model", "m", "--message", "hi", "--stream"], &[]).await;
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("before the provider signalled completion"));
}

#[tokio::test]
async fn structured_output_exit_codes() {
    let dir = std::env::temp_dir().join(format!("ml-runtime-cli-schema-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let schema = dir.join("schema.json");
    std::fs::write(
        &schema,
        json!({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}).to_string(),
    )
    .unwrap();
    let schema = schema.to_str().unwrap();
    let args = [
        "--model",
        "m",
        "--message",
        "hi",
        "--json-schema",
        schema,
        "--capability",
        "json-schema",
    ];

    let mock = Mock::start(|_| Reply::Json(200, completion(r#"{"n":1}"#))).await;
    let out = chat(&mock, &args, &[]).await;
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert!(text(&out.stderr).contains("structured output: validated"));
    assert_eq!(
        mock.captured()[0].json()["response_format"]["type"],
        "json_schema"
    );

    for bad in ["nope", r#"{"n":"x"}"#] {
        let mock = Mock::start(move |_| Reply::Json(200, completion(bad))).await;
        let out = chat(&mock, &args, &[]).await;
        assert_eq!(out.status.code(), Some(3));
        assert_eq!(
            text(&out.stdout).trim(),
            bad,
            "model text is shown unrepaired"
        );
        assert!(text(&out.stderr).contains("NOT") || text(&out.stderr).contains("VIOLATION"));
    }

    // Without declaring the capability the request is refused before sending.
    let mock = Mock::start(|_| Reply::Json(200, completion("{}"))).await;
    let out = chat(
        &mock,
        &["--model", "m", "--message", "hi", "--json-schema", schema],
        &[],
    )
    .await;
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("--capability"));
    assert!(mock.captured().is_empty());
    std::fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn list_models_and_remote_classification() {
    let mock = Mock::start(|_| {
        Reply::Json(
            200,
            json!({"data":[{"id":"alpha"},{"id":"beta"}]}).to_string(),
        )
    })
    .await;
    let out = chat(&mock, &["--list-models"], &[]).await;
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(text(&out.stdout), "alpha\nbeta\n");

    let mock = Mock::start(|_| Reply::Json(200, completion("x"))).await;
    let out = chat(
        &mock,
        &["--model", "m", "--message", "hi", "--location", "remote"],
        &[],
    )
    .await;
    let stderr = text(&out.stderr);
    assert!(
        stderr.contains("execution=Remote") && !stderr.contains("inferred"),
        "{stderr}"
    );
}
