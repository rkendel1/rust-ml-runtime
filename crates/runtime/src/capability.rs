use async_trait::async_trait;
use ml_runtime_common::{CancellationToken, RuntimeError, RuntimeResult};
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    process::Stdio,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    process::Command,
    time::{timeout, Duration},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Capability {
    pub id: String,
    pub version: String,
    pub category: String,
    pub description: String,
    pub input_schema: Value,
    pub output_schema: Value,
    pub requirements: Vec<String>,
    pub execution_targets: Vec<String>,
    pub availability: bool,
    pub platform: String,
    pub backend: String,
    pub local: bool,
    pub remote: bool,
    pub resource_requirements: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AvailabilityState {
    Registered,
    Available,
    Ready,
    Healthy,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityAvailability {
    pub state: AvailabilityState,
    pub reason: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResolutionConstraints {
    pub local_only: bool,
    pub remote_allowed: bool,
    pub provider: Option<String>,
    pub backend: Option<String>,
    pub platform: Option<String>,
    pub maximum_latency_ms: Option<u64>,
    pub maximum_memory_bytes: Option<u64>,
    pub network_required: bool,
    pub network_forbidden: bool,
    pub filesystem_scope: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResolutionCandidate {
    pub provider: String,
    pub implementation: String,
    pub version: String,
    pub available: bool,
    pub local: bool,
    pub requirements_satisfied: Vec<String>,
    pub requirements_rejected: Vec<String>,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityResolution {
    pub capability: String,
    pub provider: String,
    pub implementation: String,
    pub target: String,
    pub version: String,
    pub candidates: Vec<ResolutionCandidate>,
    pub reason: String,
}

#[async_trait]
pub trait CapabilityProvider: Send + Sync {
    fn id(&self) -> &str;
    fn version(&self) -> &str;
    fn capabilities(&self) -> Vec<Capability>;
    fn availability(&self) -> CapabilityAvailability;
    async fn execute(&self, request: ExecutionRequest) -> RuntimeResult<ExecutionResult>;
}

pub struct BuiltinCapabilityProvider;

#[async_trait]
impl CapabilityProvider for BuiltinCapabilityProvider {
    fn id(&self) -> &str {
        "builtin"
    }

    fn version(&self) -> &str {
        "1"
    }

    fn capabilities(&self) -> Vec<Capability> {
        catalog()
    }

    fn availability(&self) -> CapabilityAvailability {
        CapabilityAvailability {
            state: AvailabilityState::Available,
            reason: "built-in capability implementation".to_owned(),
        }
    }

    async fn execute(&self, request: ExecutionRequest) -> RuntimeResult<ExecutionResult> {
        execute(request, None).await
    }
}

#[derive(Clone, Default)]
pub struct CapabilityRegistry {
    providers: BTreeMap<String, Arc<dyn CapabilityProvider>>,
}

impl CapabilityRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register(BuiltinCapabilityProvider);
        registry
    }

    pub fn register<P>(&mut self, provider: P)
    where
        P: CapabilityProvider + 'static,
    {
        self.providers
            .insert(provider.id().to_owned(), Arc::new(provider));
    }

    pub fn register_arc(&mut self, provider: Arc<dyn CapabilityProvider>) {
        self.providers.insert(provider.id().to_owned(), provider);
    }

    pub fn unregister(&mut self, provider: &str) -> Option<Arc<dyn CapabilityProvider>> {
        self.providers.remove(provider)
    }

    pub fn provider(&self, provider: &str) -> Option<&Arc<dyn CapabilityProvider>> {
        self.providers.get(provider)
    }

    pub fn providers(&self) -> Vec<Arc<dyn CapabilityProvider>> {
        self.providers.values().cloned().collect()
    }

    pub fn get(&self, capability: &str) -> Option<Capability> {
        self.list().into_iter().find(|item| item.id == capability)
    }

    pub fn list(&self) -> Vec<Capability> {
        let mut capabilities = BTreeMap::new();
        for provider in self.providers.values() {
            for capability in provider.capabilities() {
                capabilities
                    .entry(capability.id.clone())
                    .or_insert(capability);
            }
        }
        capabilities.into_values().collect()
    }

    pub fn find(&self, query: &str) -> Vec<Capability> {
        self.list()
            .into_iter()
            .filter(|capability| {
                capability.id.contains(query)
                    || capability.category.contains(query)
                    || capability.description.contains(query)
            })
            .collect()
    }

    fn implementations(&self, capability: &str) -> Vec<Arc<dyn CapabilityProvider>> {
        self.providers
            .values()
            .filter(|provider| {
                provider
                    .capabilities()
                    .iter()
                    .any(|item| item.id == capability)
            })
            .cloned()
            .collect()
    }

    pub fn resolve(
        &self,
        capability: &str,
        constraints: &ResolutionConstraints,
    ) -> RuntimeResult<CapabilityResolution> {
        let mut candidates = Vec::new();
        for provider in self.implementations(capability) {
            let metadata = provider
                .capabilities()
                .into_iter()
                .find(|item| item.id == capability)
                .expect("provider capability disappeared");
            let availability = provider.availability();
            let mut satisfied = Vec::new();
            let mut rejected = Vec::new();
            if availability.state != AvailabilityState::Registered && metadata.availability {
                satisfied.push("available".to_owned());
            } else {
                rejected.push("provider unavailable".to_owned());
            }
            if constraints.local_only && !metadata.local {
                rejected.push("local_only".to_owned());
            } else if metadata.local {
                satisfied.push("local".to_owned());
            }
            if !constraints.remote_allowed && !metadata.local {
                rejected.push("remote forbidden".to_owned());
            }
            if constraints.network_required && !metadata.remote {
                rejected.push("network required".to_owned());
            }
            if constraints.network_forbidden && metadata.remote {
                rejected.push("network forbidden".to_owned());
            }
            if constraints.filesystem_scope.is_some() && !metadata.category.eq("filesystem") {
                rejected.push("filesystem scope applies only to filesystem capabilities".to_owned());
            }
            if constraints.maximum_latency_ms.is_some() {
                rejected.push("latency is not declared by implementation".to_owned());
            }
            if constraints.maximum_memory_bytes.is_some() {
                rejected.push("memory usage is not declared by implementation".to_owned());
            }
            if let Some(expected) = &constraints.provider {
                if provider.id() != expected {
                    rejected.push(format!("provider != {expected}"));
                }
            }
            if let Some(expected) = &constraints.backend {
                if metadata.backend != *expected {
                    rejected.push(format!("backend != {expected}"));
                }
            }
            if let Some(expected) = &constraints.platform {
                if metadata.platform != *expected {
                    rejected.push(format!("platform != {expected}"));
                }
            }
            let available = rejected.is_empty();
            candidates.push(ResolutionCandidate {
                provider: provider.id().to_owned(),
                implementation: metadata.backend.clone(),
                version: provider.version().to_owned(),
                available,
                local: metadata.local,
                requirements_satisfied: satisfied,
                requirements_rejected: rejected,
                reason: if available {
                    "satisfies constraints".to_owned()
                } else {
                    "rejected by constraints".to_owned()
                },
            });
        }
        let selected = candidates
            .iter()
            .find(|candidate| candidate.available)
            .ok_or_else(|| RuntimeError::CapabilityMismatch {
                reason: format!("no available provider satisfies constraints for {capability}"),
            })?;
        Ok(CapabilityResolution {
            capability: capability.to_owned(),
            provider: selected.provider.clone(),
            implementation: selected.implementation.clone(),
            target: if selected.local { "local" } else { "remote" }.to_owned(),
            version: selected.version.clone(),
            candidates,
            reason: "selected first available provider in registry order".to_owned(),
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityLimits {
    pub timeout_ms: Option<u64>,
    pub max_input_bytes: Option<usize>,
    pub max_output_bytes: Option<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityPolicy {
    pub require_authorization: bool,
    pub filesystem_root: Option<PathBuf>,
    pub network: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExecutionRequest {
    pub capability: String,
    pub input: Value,
    pub context: Value,
    pub policy: CapabilityPolicy,
    pub correlation: String,
    pub limits: CapabilityLimits,
    #[serde(skip)]
    pub cancellation: Option<CancellationToken>,
}

impl ExecutionRequest {
    pub fn new(capability: impl Into<String>, input: Value) -> Self {
        Self {
            capability: capability.into(),
            input,
            context: Value::Null,
            policy: CapabilityPolicy::default(),
            correlation: format!("cap-{}", unique_id()),
            limits: CapabilityLimits::default(),
            cancellation: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CapabilityExecutionMetadata {
    pub capability: String,
    pub version: String,
    pub target: String,
    pub correlation: String,
    pub started_at_ms: u128,
    pub ended_at_ms: u128,
    pub duration_ms: u128,
    pub success: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExecutionResult {
    pub output: Value,
    pub evidence: Value,
    pub execution_metadata: CapabilityExecutionMetadata,
}

#[async_trait]
pub trait CapabilityAuthorizer: Send + Sync {
    async fn authorize(
        &self,
        request: &ExecutionRequest,
        capability: &Capability,
    ) -> RuntimeResult<()>;
}

pub fn catalog() -> Vec<Capability> {
    [
        ("intelligence.generate", "Intelligence generation"),
        ("intelligence.embed", "Intelligence embeddings"),
        ("filesystem.read", "Read a bounded local file"),
        ("filesystem.list", "List a local directory"),
        ("filesystem.write", "Write a bounded local file"),
        ("process.spawn", "Spawn a constrained local process"),
        ("network.http", "Perform an explicit HTTP request"),
        ("git.status", "Inspect repository status"),
        ("git.diff", "Inspect repository diff"),
        ("data.parse", "Parse structured data"),
    ]
    .into_iter()
    .map(|(id, description)| {
        let category = id.split('.').next().unwrap_or("runtime").to_owned();
        let sensitive = matches!(id, "filesystem.write" | "process.spawn" | "network.http");
        Capability {
            id: id.to_owned(),
            version: "1".to_owned(),
            category,
            description: description.to_owned(),
            input_schema: json!({"type": "object"}),
            output_schema: json!({"type": "object"}),
            requirements: if sensitive {
                vec!["authorization".to_owned()]
            } else {
                Vec::new()
            },
            execution_targets: vec!["local".to_owned()],
            availability: true,
            platform: std::env::consts::OS.to_owned(),
            backend: "builtin".to_owned(),
            local: true,
            remote: false,
            resource_requirements: vec!["explicit limits".to_owned()],
        }
    })
    .collect()
}

pub fn find(id: &str) -> Option<Capability> {
    catalog().into_iter().find(|capability| capability.id == id)
}

pub async fn execute(
    request: ExecutionRequest,
    authorizer: Option<&dyn CapabilityAuthorizer>,
) -> RuntimeResult<ExecutionResult> {
    let capability = find(&request.capability).ok_or_else(|| RuntimeError::CapabilityMismatch {
        reason: format!("unknown capability {}", request.capability),
    })?;
    if request.policy.require_authorization || !capability.requirements.is_empty() {
        let authorizer = authorizer.ok_or_else(|| RuntimeError::Execution {
            operation: request.capability.clone(),
            reason: "authorization required".to_owned(),
        })?;
        authorizer.authorize(&request, &capability).await?;
    }
    let started = now_ms();
    let operation = run(&request, &capability);
    let result = operation.await;
    let ended = now_ms();
    let success = result.is_ok();
    let output = result?;
    Ok(ExecutionResult {
        output,
        evidence: json!({"capability": capability.id, "target": "local"}),
        execution_metadata: CapabilityExecutionMetadata {
            capability: capability.id,
            version: capability.version,
            target: "local".to_owned(),
            correlation: request.correlation,
            started_at_ms: started,
            ended_at_ms: ended,
            duration_ms: ended.saturating_sub(started),
            success,
        },
    })
}

async fn run(request: &ExecutionRequest, capability: &Capability) -> RuntimeResult<Value> {
    let timeout_ms = request.limits.timeout_ms;
    let cancellation = request.cancellation.clone();
    let operation = async {
        match capability.id.as_str() {
            "filesystem.read" => filesystem_read(request),
            "filesystem.list" => filesystem_list(request),
            "filesystem.write" => filesystem_write(request),
            "process.spawn" => process_spawn(request).await,
            "network.http" => network_http(request).await,
            "git.status" | "git.diff" => git_command(request, &capability.id).await,
            "data.parse" => data_parse(request),
            _ => Err(RuntimeError::CapabilityMismatch {
                reason: format!("{} requires an ML provider", capability.id),
            }),
        }
    };
    let result = if let Some(token) = cancellation {
        tokio::select! {
            _ = token.cancelled() => Err(RuntimeError::Cancelled),
            result = operation => result,
        }
    } else if let Some(ms) = timeout_ms {
        timeout(Duration::from_millis(ms), operation)
            .await
            .map_err(|_| RuntimeError::Timeout {
                operation: request.capability.clone(),
            })?
    } else {
        operation.await
    }?;
    Ok(result)
}

fn path(request: &ExecutionRequest) -> RuntimeResult<PathBuf> {
    let value = request
        .input
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| RuntimeError::InvalidInput {
            reason: "path is required".to_owned(),
        })?;
    let path = PathBuf::from(value);
    if let Some(root) = &request.policy.filesystem_root {
        let candidate = std::fs::canonicalize(&path).map_err(|error| RuntimeError::Execution {
            operation: "filesystem".to_owned(),
            reason: error.to_string(),
        })?;
        if request
            .limits
            .max_output_bytes
            .is_some_and(|limit| output.stdout.len() + output.stderr.len() > limit)
        {
            return Err(RuntimeError::ResourceLimit {
                reason: "process output exceeds max_output_bytes".to_owned(),
            });
        }
        let root = std::fs::canonicalize(root).map_err(|error| RuntimeError::Execution {
            operation: "filesystem".to_owned(),
            reason: error.to_string(),
        })?;
        if !candidate.starts_with(root) {
            return Err(RuntimeError::CapabilityMismatch {
                reason: "filesystem path is outside the configured scope".to_owned(),
            });
        }
    }
    Ok(path)
}

fn filesystem_read(request: &ExecutionRequest) -> RuntimeResult<Value> {
    let path = path(request)?;
    let data = std::fs::read(&path).map_err(|error| RuntimeError::Execution {
        operation: "filesystem.read".to_owned(),
        reason: error.to_string(),
    })?;
    if request
        .limits
        .max_output_bytes
        .is_some_and(|limit| data.len() > limit)
    {
        return Err(RuntimeError::ResourceLimit {
            reason: "filesystem output exceeds max_output_bytes".to_owned(),
        });
    }
    Ok(json!({"path": path, "data": String::from_utf8_lossy(&data), "bytes": data.len()}))
}

fn filesystem_list(request: &ExecutionRequest) -> RuntimeResult<Value> {
    let path = path(request)?;
    let entries = std::fs::read_dir(&path)
        .map_err(|error| RuntimeError::Execution {
            operation: "filesystem.list".to_owned(),
            reason: error.to_string(),
        })?
        .filter_map(Result::ok)
        .map(|entry| json!({"path": entry.path(), "name": entry.file_name().to_string_lossy()}))
        .collect::<Vec<_>>();
    Ok(json!({"path": path, "entries": entries}))
}

fn filesystem_write(request: &ExecutionRequest) -> RuntimeResult<Value> {
    let path = path(request)?;
    let data = request
        .input
        .get("data")
        .and_then(Value::as_str)
        .ok_or_else(|| RuntimeError::InvalidInput {
            reason: "data is required".to_owned(),
        })?;
    if request
        .limits
        .max_input_bytes
        .is_some_and(|limit| data.len() > limit)
    {
        return Err(RuntimeError::ResourceLimit {
            reason: "filesystem input exceeds max_input_bytes".to_owned(),
        });
    }
    std::fs::write(&path, data).map_err(|error| RuntimeError::Execution {
        operation: "filesystem.write".to_owned(),
        reason: error.to_string(),
    })?;
    Ok(json!({"path": path, "bytes": data.len()}))
}

async fn process_spawn(request: &ExecutionRequest) -> RuntimeResult<Value> {
    let command = request
        .input
        .get("command")
        .and_then(Value::as_str)
        .ok_or_else(|| RuntimeError::InvalidInput {
            reason: "command is required".to_owned(),
        })?;
    let args = request
        .input
        .get("arguments")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut child = Command::new(command);
    child
        .args(args.iter().filter_map(Value::as_str))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(cwd) = request
        .input
        .get("working_directory")
        .and_then(Value::as_str)
    {
        child.current_dir(cwd);
    }
    let output = timeout(
        Duration::from_millis(request.limits.timeout_ms.unwrap_or(30_000)),
        child.output(),
    )
    .await
    .map_err(|_| RuntimeError::Timeout {
        operation: "process.spawn".to_owned(),
    })?
    .map_err(|error| RuntimeError::Execution {
        operation: "process.spawn".to_owned(),
        reason: error.to_string(),
    })?;
    Ok(
        json!({"status": output.status.code(), "stdout": String::from_utf8_lossy(&output.stdout), "stderr": String::from_utf8_lossy(&output.stderr)}),
    )
}

async fn network_http(request: &ExecutionRequest) -> RuntimeResult<Value> {
    if !request.policy.network {
        return Err(RuntimeError::CapabilityMismatch {
            reason: "network access is not enabled by policy".to_owned(),
        });
    }
    let method = request
        .input
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("GET")
        .parse::<Method>()
        .map_err(|_| RuntimeError::InvalidInput {
            reason: "invalid HTTP method".to_owned(),
        })?;
    let url = request
        .input
        .get("url")
        .and_then(Value::as_str)
        .ok_or_else(|| RuntimeError::InvalidInput {
            reason: "url is required".to_owned(),
        })?;
    let mut builder = reqwest::Client::new().request(method, url);
    if let Some(headers) = request.input.get("headers").and_then(Value::as_object) {
        for (name, value) in headers {
            if let Some(value) = value.as_str() {
                builder = builder.header(name, value);
            }
        }
    }
    if let Some(body) = request.input.get("body") {
        builder = builder.json(body);
    }
    let response = builder
        .send()
        .await
        .map_err(|error| RuntimeError::Transport {
            provider: "http".to_owned(),
            reason: error.to_string(),
        })?;
    let status = response.status().as_u16();
    let body = response
        .text()
        .await
        .map_err(|error| RuntimeError::Transport {
            provider: "http".to_owned(),
            reason: error.to_string(),
        })?;
    Ok(json!({"status": status, "body": body}))
}

async fn git_command(request: &ExecutionRequest, capability: &str) -> RuntimeResult<Value> {
    let cwd = request
        .input
        .get("repository")
        .and_then(Value::as_str)
        .unwrap_or(".");
    let argument = if capability == "git.status" {
        "status"
    } else {
        "diff"
    };
    let output = Command::new("git")
        .args(["-C", cwd, argument])
        .output()
        .await
        .map_err(|error| RuntimeError::Execution {
            operation: capability.to_owned(),
            reason: error.to_string(),
        })?;
    Ok(
        json!({"status": output.status.code(), "stdout": String::from_utf8_lossy(&output.stdout), "stderr": String::from_utf8_lossy(&output.stderr)}),
    )
}

fn data_parse(request: &ExecutionRequest) -> RuntimeResult<Value> {
    let format = request
        .input
        .get("format")
        .and_then(Value::as_str)
        .unwrap_or("json");
    let data = request
        .input
        .get("data")
        .and_then(Value::as_str)
        .ok_or_else(|| RuntimeError::InvalidInput {
            reason: "data is required".to_owned(),
        })?;
    match format {
        "json" => serde_json::from_str(data).map_err(|error| RuntimeError::InvalidInput {
            reason: error.to_string(),
        }),
        "csv" => Ok(Value::Array(
            data.lines()
                .map(|line| {
                    Value::Array(
                        line.split(',')
                            .map(|value| Value::String(value.trim().to_owned()))
                            .collect(),
                    )
                })
                .collect(),
        )),
        _ => Err(RuntimeError::CapabilityMismatch {
            reason: format!("unsupported data format {format}"),
        }),
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
fn unique_id() -> u128 {
    now_ms()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn discovers_and_executes_json_parse() {
        assert!(find("filesystem.read").is_some());
        let result = execute(
            ExecutionRequest::new(
                "data.parse",
                json!({"format": "json", "data": r#"{"ok":true}"#}),
            ),
            None,
        )
        .await
        .unwrap();
        assert_eq!(result.output["ok"], true);
        assert_eq!(result.execution_metadata.capability, "data.parse");
    }

    #[tokio::test]
    async fn sensitive_capabilities_fail_closed_without_authorizer() {
        let error = execute(
            ExecutionRequest::new("process.spawn", json!({"command": "true"})),
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, RuntimeError::Execution { .. }));
    }

    #[test]
    fn registry_resolves_and_explains_builtin_capabilities() {
        let registry = CapabilityRegistry::with_builtins();
        let result = registry
            .resolve("data.parse", &ResolutionConstraints::default())
            .unwrap();
        assert_eq!(result.provider, "builtin");
        assert_eq!(result.target, "local");
        assert_eq!(result.candidates.len(), 1);
        assert!(result.candidates[0]
            .requirements_satisfied
            .iter()
            .any(|requirement| requirement == "available"));
    }

    #[test]
    fn registry_rejects_local_only_for_remote_capabilities() {
        let registry = CapabilityRegistry::with_builtins();
        let result = registry.resolve(
            "data.parse",
            &ResolutionConstraints {
                provider: Some("remote".to_owned()),
                ..ResolutionConstraints::default()
            },
        );
        assert!(matches!(
            result,
            Err(RuntimeError::CapabilityMismatch { .. })
        ));
    }
}
