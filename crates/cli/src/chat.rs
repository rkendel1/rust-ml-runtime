//! `ml-runtime chat`: single-request generative inference through an
//! explicitly selected provider. Credentials are read from the environment
//! and are never printed.

use clap::Args;
use futures_util::StreamExt;
use ml_runtime::{
    ApiKey, ChatMessage, ChatMetadata, ChatRequest, ChatStreamEvent, ExecutionLocation,
    FinishReason, GenerationOptions, OpenAiCompatibleConfig, OpenAiCompatibleProvider,
    ResponseFormat, Runtime, RuntimeError, StructuredOutput, Usage,
};
use std::{io::Write, path::PathBuf, time::Duration};

const PROVIDER: &str = "openai-compatible";
const DEFAULT_KEY_ENV: &str = "ML_RUNTIME_API_KEY";

/// Exit code when a request succeeded but requested structured output did not validate.
pub const EXIT_STRUCTURED_INVALID: i32 = 3;

#[derive(Args, Debug)]
pub struct ChatArgs {
    /// Generation provider. There is no default and no fallback.
    #[arg(long, value_parser = [PROVIDER])]
    provider: String,
    /// Base URL including the API prefix, e.g. http://127.0.0.1:11434/v1
    #[arg(long)]
    endpoint: String,
    /// Explicit model identifier (required unless --list-models)
    #[arg(long)]
    model: Option<String>,
    /// List models reported by the endpoint and exit
    #[arg(long, conflicts_with_all = ["messages", "messages_json", "system", "stream"])]
    list_models: bool,
    #[arg(long, conflicts_with = "messages_json")]
    system: Option<String>,
    /// User message; repeat for several turns
    #[arg(long = "message", conflicts_with = "messages_json")]
    messages: Vec<String>,
    /// JSON file (or - for stdin) holding [{"role": "system|user|assistant", "content": "..."}]
    #[arg(long)]
    messages_json: Option<PathBuf>,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    max_tokens: Option<u32>,
    #[arg(long)]
    seed: Option<i64>,
    /// Stop sequence; repeatable
    #[arg(long)]
    stop: Vec<String>,
    #[arg(long)]
    stream: bool,
    /// Request JSON-schema constrained output (schema read from this file)
    #[arg(long, conflicts_with = "json_object")]
    json_schema: Option<PathBuf>,
    #[arg(long, default_value = "response")]
    schema_name: String,
    #[arg(long)]
    strict: bool,
    /// Request any valid JSON object
    #[arg(long)]
    json_object: bool,
    /// Environment variable holding the bearer token (default: ML_RUNTIME_API_KEY, if set)
    #[arg(long)]
    api_key_env: Option<String>,
    /// Classify the endpoint explicitly; by default loopback hosts are local
    #[arg(long, value_parser = ["local", "remote"])]
    location: Option<String>,
    /// Declare an endpoint capability (they cannot be discovered)
    #[arg(long = "capability", value_parser = ["json-object", "json-schema", "no-streaming", "no-stop", "no-seed"])]
    capabilities: Vec<String>,
    #[arg(long, default_value_t = 10)]
    connect_timeout_secs: u64,
    #[arg(long, default_value_t = 120)]
    timeout_secs: u64,
    #[arg(long, default_value_t = 60)]
    idle_timeout_secs: u64,
    /// Machine-readable output (one JSON document, or NDJSON when streaming)
    #[arg(long)]
    json: bool,
}

fn invalid(reason: impl Into<String>) -> String {
    reason.into()
}

fn read_messages_file(path: &PathBuf) -> Result<Vec<ChatMessage>, String> {
    let text = if path.as_os_str() == "-" {
        std::io::read_to_string(std::io::stdin())
            .map_err(|e| invalid(format!("reading stdin: {e}")))?
    } else {
        std::fs::read_to_string(path)
            .map_err(|e| invalid(format!("reading {}: {e}", path.display())))?
    };
    serde_json::from_str(&text).map_err(|e| invalid(format!("invalid messages JSON: {e}")))
}

impl ChatArgs {
    /// Builds the request from arguments: the optional system message first,
    /// then user messages in command-line order.
    fn request(&self) -> Result<ChatRequest, String> {
        let model = self
            .model
            .clone()
            .filter(|m| !m.trim().is_empty())
            .ok_or_else(|| invalid("--model is required"))?;
        let messages = match &self.messages_json {
            Some(path) => read_messages_file(path)?,
            None => {
                if self.messages.is_empty() {
                    return Err(invalid("provide at least one --message or --messages-json"));
                }
                self.system
                    .iter()
                    .map(ChatMessage::system)
                    .chain(self.messages.iter().map(ChatMessage::user))
                    .collect()
            }
        };
        let response_format = match (&self.json_schema, self.json_object) {
            (Some(path), _) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| invalid(format!("reading {}: {e}", path.display())))?;
                let schema = serde_json::from_str(&text)
                    .map_err(|e| invalid(format!("invalid JSON schema file: {e}")))?;
                ResponseFormat::JsonSchema {
                    name: self.schema_name.clone(),
                    schema,
                    strict: self.strict,
                }
            }
            (None, true) => ResponseFormat::JsonObject,
            (None, false) => ResponseFormat::Text,
        };
        Ok(
            ChatRequest::new(model, messages).with_options(GenerationOptions {
                temperature: self.temperature,
                top_p: self.top_p,
                max_tokens: self.max_tokens,
                stop: self.stop.clone(),
                seed: self.seed,
                response_format,
            }),
        )
    }

    fn config(&self) -> Result<OpenAiCompatibleConfig, String> {
        let mut config = OpenAiCompatibleConfig::new(&self.endpoint);
        for capability in &self.capabilities {
            match capability.as_str() {
                "json-object" => config.capabilities.json_object = true,
                "json-schema" => config.capabilities.json_schema = true,
                "no-streaming" => config.capabilities.streaming = false,
                "no-stop" => config.capabilities.stop_sequences = false,
                "no-seed" => config.capabilities.seed = false,
                _ => unreachable!("restricted by the argument parser"),
            }
        }
        config.location = self.location.as_deref().map(|l| match l {
            "local" => ExecutionLocation::Local,
            _ => ExecutionLocation::Remote,
        });
        config.connect_timeout = Duration::from_secs(self.connect_timeout_secs);
        config.request_timeout = Duration::from_secs(self.timeout_secs);
        config.stream_idle_timeout = Duration::from_secs(self.idle_timeout_secs);
        let key = match self.api_key_env.as_deref() {
            Some(name) => Some(std::env::var(name).map_err(|_| {
                invalid(format!(
                    "environment variable {name} is not set or not valid UTF-8"
                ))
            })?),
            None => std::env::var(DEFAULT_KEY_ENV)
                .ok()
                .filter(|k| !k.is_empty()),
        };
        if let Some(key) = key {
            // Report only the variable's name on failure, never its value.
            config.api_key = Some(
                ApiKey::new(key)
                    .map_err(|_| invalid("the API key in the environment is malformed"))?,
            );
        }
        Ok(config)
    }
}

fn describe_usage(usage: &Option<Usage>) -> String {
    let fmt = |v: Option<u64>| v.map_or_else(|| "unreported".to_owned(), |v| v.to_string());
    match usage {
        Some(u) => format!(
            "prompt={} completion={} total={}",
            fmt(u.prompt_tokens),
            fmt(u.completion_tokens),
            fmt(u.total_tokens)
        ),
        None => "unreported".to_owned(),
    }
}

fn describe_finish(finish: &Option<FinishReason>) -> String {
    match finish {
        Some(FinishReason::Other(raw)) => format!("other({raw})"),
        Some(other) => serde_json::to_value(other)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default(),
        None => "unreported".to_owned(),
    }
}

fn diagnostics(
    metadata: &ChatMetadata,
    finish: &Option<FinishReason>,
    usage: &Option<Usage>,
    structured: &Option<StructuredOutput>,
) {
    eprintln!(
        "provider={} transport={} execution={:?}{} endpoint={}",
        metadata.provider,
        metadata.transport,
        metadata.execution,
        if metadata.execution_inferred {
            " (inferred from host)"
        } else {
            ""
        },
        metadata.endpoint
    );
    eprintln!(
        "model requested={} reported={} finish={} usage: {} duration={:.0?}",
        metadata.requested_model,
        metadata.reported_model.as_deref().unwrap_or("unreported"),
        describe_finish(finish),
        describe_usage(usage),
        metadata.duration
    );
    match structured {
        Some(StructuredOutput::Validated { .. }) => eprintln!("structured output: validated"),
        Some(StructuredOutput::InvalidJson { error }) => {
            eprintln!("structured output: NOT VALID JSON ({error})")
        }
        Some(StructuredOutput::SchemaViolation { errors }) => {
            eprintln!("structured output: SCHEMA VIOLATION");
            for error in errors {
                eprintln!("  - {error}");
            }
        }
        None => {}
    }
}

fn structured_exit(structured: &Option<StructuredOutput>) -> i32 {
    match structured {
        Some(s) if !s.is_validated() => EXIT_STRUCTURED_INVALID,
        _ => 0,
    }
}

fn report_error(error: &RuntimeError, args: &ChatArgs) {
    eprintln!("error: {error}");
    if matches!(error, RuntimeError::CapabilityMismatch { .. })
        && (args.json_schema.is_some() || args.json_object)
    {
        eprintln!("hint: if the endpoint supports it, declare it with --capability json-schema or json-object");
    }
}

/// Runs the command and returns the process exit code (0 success, 1 failure,
/// 3 structured output did not validate; clap itself uses 2 for usage errors).
pub async fn run(args: ChatArgs) -> i32 {
    match execute(&args).await {
        Ok(code) => code,
        Err(Failure::Usage(message)) => {
            eprintln!("error: {message}");
            1
        }
        Err(Failure::Runtime(error)) => {
            report_error(&error, &args);
            1
        }
    }
}

enum Failure {
    Usage(String),
    Runtime(RuntimeError),
}

impl From<RuntimeError> for Failure {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(error)
    }
}

async fn execute(args: &ChatArgs) -> Result<i32, Failure> {
    let config = args.config().map_err(Failure::Usage)?;
    let provider = OpenAiCompatibleProvider::new(config)?;
    let runtime = Runtime::builder()
        .register_generation_provider(provider)
        .build();

    if args.list_models {
        let models = runtime.list_generation_models(&args.provider, None).await?;
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&models).unwrap_or_default()
            );
        } else {
            for model in models {
                println!("{}", model.id);
            }
        }
        return Ok(0);
    }

    let request = args.request().map_err(Failure::Usage)?;
    if args.stream {
        stream(&runtime, args, request).await
    } else {
        let response = runtime.chat(&args.provider, request, None).await?;
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&response).unwrap_or_default()
            );
        } else {
            println!("{}", response.message.content);
        }
        diagnostics(
            &response.metadata,
            &response.finish_reason,
            &response.usage,
            &response.structured,
        );
        Ok(structured_exit(&response.structured))
    }
}

async fn stream(runtime: &Runtime, args: &ChatArgs, request: ChatRequest) -> Result<i32, Failure> {
    let mut events = runtime.chat_stream(&args.provider, request, None).await?;
    let mut stdout = std::io::stdout();
    let mut code = 0;
    while let Some(event) = events.next().await {
        let event = event?;
        if args.json {
            let _ = writeln!(
                stdout,
                "{}",
                serde_json::to_string(&event).unwrap_or_default()
            );
        } else if let ChatStreamEvent::Delta { content } = &event {
            let _ = write!(stdout, "{content}");
        }
        let _ = stdout.flush();
        if let ChatStreamEvent::Completed(summary) = event {
            if !args.json {
                println!();
            }
            diagnostics(
                &summary.metadata,
                &summary.finish_reason,
                &summary.usage,
                &summary.structured,
            );
            code = structured_exit(&summary.structured);
        }
    }
    Ok(code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use ml_runtime::ChatRole;

    #[derive(Parser)]
    struct Wrapper {
        #[command(flatten)]
        args: ChatArgs,
    }

    fn parse(extra: &[&str]) -> Result<ChatArgs, clap::Error> {
        let mut argv = vec![
            "chat",
            "--provider",
            "openai-compatible",
            "--endpoint",
            "http://127.0.0.1:1/v1",
        ];
        argv.extend_from_slice(extra);
        Wrapper::try_parse_from(argv).map(|w| w.args)
    }

    #[test]
    fn messages_are_ordered_system_first_then_user_turns() {
        let args = parse(&[
            "--model",
            "m",
            "--message",
            "a",
            "--system",
            "s",
            "--message",
            "b",
        ])
        .unwrap();
        let request = args.request().unwrap();
        let order: Vec<_> = request
            .messages
            .iter()
            .map(|m| (m.role, m.content.as_str()))
            .collect();
        assert_eq!(
            order,
            [
                (ChatRole::System, "s"),
                (ChatRole::User, "a"),
                (ChatRole::User, "b")
            ]
        );
    }

    #[test]
    fn provider_is_required_and_restricted() {
        assert!(
            Wrapper::try_parse_from(["chat", "--endpoint", "http://x/v1", "--message", "m"])
                .is_err()
        );
        assert!(Wrapper::try_parse_from([
            "chat",
            "--provider",
            "other",
            "--endpoint",
            "http://x/v1"
        ])
        .is_err());
        assert!(Wrapper::try_parse_from(["chat", "--provider", "openai-compatible"]).is_err());
    }

    #[test]
    fn conflicting_and_missing_inputs_are_rejected() {
        assert!(parse(&[
            "--model",
            "m",
            "--message",
            "a",
            "--messages-json",
            "f.json"
        ])
        .is_err());
        assert!(parse(&["--list-models", "--message", "a"]).is_err());
        assert!(parse(&["--json-schema", "s.json", "--json-object"]).is_err());
        assert!(parse(&["--capability", "bogus"]).is_err());
        assert!(parse(&["--location", "moon"]).is_err());
        assert!(
            parse(&["--message", "a"]).unwrap().request().is_err(),
            "model is required"
        );
        assert!(
            parse(&["--model", "m"]).unwrap().request().is_err(),
            "a message is required"
        );
        assert!(parse(&["--model", " ", "--message", "a"])
            .unwrap()
            .request()
            .is_err());
    }

    #[test]
    fn capabilities_and_location_are_explicit() {
        let args = parse(&[
            "--capability",
            "json-schema",
            "--capability",
            "no-streaming",
            "--location",
            "remote",
        ])
        .unwrap();
        let config = args.config().unwrap();
        assert!(config.capabilities.json_schema && !config.capabilities.json_object);
        assert!(!config.capabilities.streaming && config.capabilities.stop_sequences);
        assert_eq!(config.location, Some(ExecutionLocation::Remote));
    }

    #[test]
    fn missing_key_variable_names_the_variable_but_no_value() {
        let args = parse(&["--api-key-env", "ML_RUNTIME_TEST_UNSET_KEY_VAR"]).unwrap();
        let error = args.config().unwrap_err();
        assert!(error.contains("ML_RUNTIME_TEST_UNSET_KEY_VAR"));
    }
}
