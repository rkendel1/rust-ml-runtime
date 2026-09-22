#![cfg_attr(test, allow(clippy::default_constructed_unit_structs))]

use futures_util::StreamExt;
use ml_runtime_backend::{Backend, BackendCapability};
use ml_runtime_common::BoxStream;
use ml_runtime_cpu_backend::CpuBackend;
use ml_runtime_http_provider::HttpProvider;
use ml_runtime_inference::InferenceChunk;
use ml_runtime_model::{CatalogError, ModelCatalog, ModelHandle};
use ml_runtime_onnx_backend::OnnxBackend;
use ml_runtime_provider::{Provider, ProviderCapability};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, RwLock,
    },
    time::{Duration, Instant},
};
use tokio::{
    sync::{mpsc, Semaphore},
    time::timeout,
};
use tokio_stream::wrappers::ReceiverStream;

mod capability;

pub use capability::{
    AvailabilityState, BuiltinCapabilityProvider, Capability, CapabilityAuthorizer,
    CapabilityAvailability, CapabilityExecutionMetadata, CapabilityLimits,
    CapabilityPackageManifest, CapabilityPolicy, CapabilityProvider, CapabilityRegistry,
    CapabilityResolution, ExecutionRequest, ExecutionResult, LocalProviderPackage,
    ManifestCapabilityProvider, PackageCapability, PackageManifest, ProviderDescriptor,
    ProviderManifest, ResolutionCandidate, ResolutionConstraints,
};

pub use ml_runtime_backend;
pub use ml_runtime_backend::{DecisionModel, DecisionModelProvider};
pub use ml_runtime_common;
pub use ml_runtime_common::{CancellationToken, RuntimeError, RuntimeResult};
pub use ml_runtime_inference;
pub use ml_runtime_inference::{
    DecisionExecution, DecisionOption, DecisionProvenance, DecisionQuestion, DecisionRequest,
    DecisionResult, DecisionType, DecisionValue, ExecutionMetadata, ExecutionPolicy,
    InferenceOptions, InferenceRequest, InferenceResult, InferenceStreamEvent, Input,
    ModelDescription, ModelIdentity, Output, Tensor, TypedDecision,
};
pub use ml_runtime_model;
pub use ml_runtime_model::{
    default_model_root, BuiltinModelRegistry, CompiledArtifactManifest, CompiledStatus,
    FilesystemModelCatalog, InstallationManifest, ModelDescriptor, ModelFormat, ModelId,
    ModelLocation, ModelReference, ModelSpec, RegisteredModelInstaller,
};
pub use ml_runtime_provider;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InstalledModelStatus {
    NotInstalled,
    InstalledNotCompiled,
    Ready,
    Corrupt,
    Incompatible,
    Unsupported,
}

impl std::fmt::Display for InstalledModelStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NotInstalled => "not_installed",
            Self::InstalledNotCompiled => "installed_not_compiled",
            Self::Ready => "ready",
            Self::Corrupt => "corrupt",
            Self::Incompatible => "incompatible",
            Self::Unsupported => "unsupported",
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstalledModelRecord {
    pub model: String,
    pub revision: String,
    pub backend: String,
    pub platform: String,
    pub status: InstalledModelStatus,
    pub compiled: bool,
    pub package_path: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelDiagnostic {
    pub check: String,
    pub ok: bool,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelDoctorReport {
    pub model: InstalledModelRecord,
    pub diagnostics: Vec<ModelDiagnostic>,
    pub manifest: Option<InstallationManifest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelLifecycleResult {
    pub model: InstalledModelRecord,
    pub already_installed: bool,
    pub duration_ms: u128,
}

/// Bounded asynchronous stream of normalized runtime-owned inference events.
pub type InferenceStream = BoxStream<RuntimeResult<InferenceStreamEvent>>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Platform {
    Linux,
    Macos,
    Windows,
    Unknown,
}

impl Platform {
    fn current() -> Self {
        if cfg!(target_os = "linux") {
            Self::Linux
        } else if cfg!(target_os = "macos") {
            Self::Macos
        } else if cfg!(target_os = "windows") {
            Self::Windows
        } else {
            Self::Unknown
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelCapability {
    pub id: String,
    pub format: ModelFormat,
    pub location: ModelLocation,
    pub loaded: bool,
    pub provider: Option<String>,
    pub backend: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RuntimeCapabilities {
    pub environment: RuntimeEnvironment,
    pub platforms: Vec<Platform>,
    pub providers: Vec<ProviderCapability>,
    pub backends: Vec<BackendCapability>,
    pub models: Vec<ModelCapability>,
    pub capabilities: Vec<Capability>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeEnvironment {
    pub platform: String,
    pub architecture: String,
    pub operating_system: String,
    pub cpu: String,
    pub memory_bytes: Option<u64>,
    pub accelerators: Vec<String>,
    pub filesystem: String,
    pub network: String,
    pub developer_tools: Vec<String>,
    pub installed_providers: Vec<String>,
}

impl RuntimeEnvironment {
    pub fn discover(provider_ids: Vec<String>) -> Self {
        Self {
            platform: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            operating_system: std::env::consts::OS.to_owned(),
            cpu: std::thread::available_parallelism()
                .map(|count| format!("{} logical cores", count.get()))
                .unwrap_or_else(|_| "unknown".to_owned()),
            memory_bytes: available_memory_bytes(),
            accelerators: Vec::new(),
            filesystem: std::env::current_dir()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|_| "unavailable".to_owned()),
            network: "available by explicit policy".to_owned(),
            developer_tools: ["git"]
                .iter()
                .filter(|tool| command_available(tool))
                .map(|tool| (*tool).to_owned())
                .collect(),
            installed_providers: provider_ids,
        }
    }
}

fn command_available(command: &str) -> bool {
    std::process::Command::new(command)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn available_memory_bytes() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    contents.lines().find_map(|line| {
        let value = line
            .strip_prefix("MemTotal:")?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()?;
        Some(value * 1024)
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeStatus {
    pub loaded_models: usize,
    pub max_models: usize,
    pub max_concurrent_inferences: usize,
    pub max_concurrent_model_loads: usize,
}

pub type ExecutionId = String;
pub type RequestId = String;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionCompletion {
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionRecord {
    pub execution_id: ExecutionId,
    pub request_id: RequestId,
    pub model_id: String,
    pub model_version: String,
    pub provider: String,
    pub backend: String,
    pub target: String,
    pub routing_policy: ExecutionPolicy,
    pub fallback: Option<String>,
    pub cache_hit: bool,
    pub batch_size: usize,
    pub queued_duration: Duration,
    pub model_load_duration: Duration,
    pub execution_duration: Duration,
    pub total_duration: Duration,
    pub streaming: bool,
    pub output_event_count: usize,
    pub completion: ExecutionCompletion,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RuntimeMetricsSnapshot {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub cancelled_requests: u64,
    pub local_executions: u64,
    pub remote_executions: u64,
    pub fallback_executions: u64,
    pub fallback_failures: u64,
    pub model_loads: u64,
    pub model_load_failures: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub queued_requests: u64,
    pub active_inferences: usize,
    pub active_model_loads: usize,
    pub batch_executions: u64,
    pub individual_executions: u64,
    pub total_requests_batched: u64,
    pub streaming_requests: u64,
    pub streamed_output_events: u64,
    pub cancelled_streams: u64,
    pub compatibility_streams: u64,
    pub queue_wait_micros: Vec<u64>,
    pub model_load_micros: Vec<u64>,
    pub execution_micros: Vec<u64>,
    pub total_duration_micros: Vec<u64>,
    pub batch_sizes: Vec<usize>,
}

#[derive(Default)]
pub struct RuntimeMetrics {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    cancelled_requests: AtomicU64,
    local_executions: AtomicU64,
    remote_executions: AtomicU64,
    fallback_executions: AtomicU64,
    fallback_failures: AtomicU64,
    model_loads: AtomicU64,
    model_load_failures: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    evictions: AtomicU64,
    queued_requests: AtomicU64,
    active_inferences: AtomicUsize,
    active_model_loads: AtomicUsize,
    batch_executions: AtomicU64,
    individual_executions: AtomicU64,
    total_requests_batched: AtomicU64,
    streaming_requests: AtomicU64,
    streamed_output_events: AtomicU64,
    cancelled_streams: AtomicU64,
    compatibility_streams: AtomicU64,
    queue_wait_micros: Mutex<Vec<u64>>,
    model_load_micros: Mutex<Vec<u64>>,
    execution_micros: Mutex<Vec<u64>>,
    total_duration_micros: Mutex<Vec<u64>>,
    batch_sizes: Mutex<Vec<usize>>,
}

impl RuntimeMetrics {
    pub fn snapshot(&self) -> RuntimeMetricsSnapshot {
        let values = |values: &Mutex<Vec<u64>>| values.lock().expect("metrics poisoned").clone();
        RuntimeMetricsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            cancelled_requests: self.cancelled_requests.load(Ordering::Relaxed),
            local_executions: self.local_executions.load(Ordering::Relaxed),
            remote_executions: self.remote_executions.load(Ordering::Relaxed),
            fallback_executions: self.fallback_executions.load(Ordering::Relaxed),
            fallback_failures: self.fallback_failures.load(Ordering::Relaxed),
            model_loads: self.model_loads.load(Ordering::Relaxed),
            model_load_failures: self.model_load_failures.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            queued_requests: self.queued_requests.load(Ordering::Relaxed),
            active_inferences: self.active_inferences.load(Ordering::Relaxed),
            active_model_loads: self.active_model_loads.load(Ordering::Relaxed),
            batch_executions: self.batch_executions.load(Ordering::Relaxed),
            individual_executions: self.individual_executions.load(Ordering::Relaxed),
            total_requests_batched: self.total_requests_batched.load(Ordering::Relaxed),
            streaming_requests: self.streaming_requests.load(Ordering::Relaxed),
            streamed_output_events: self.streamed_output_events.load(Ordering::Relaxed),
            cancelled_streams: self.cancelled_streams.load(Ordering::Relaxed),
            compatibility_streams: self.compatibility_streams.load(Ordering::Relaxed),
            queue_wait_micros: values(&self.queue_wait_micros),
            model_load_micros: values(&self.model_load_micros),
            execution_micros: values(&self.execution_micros),
            total_duration_micros: values(&self.total_duration_micros),
            batch_sizes: self.batch_sizes.lock().expect("metrics poisoned").clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelResource {
    pub id: String,
    pub version: Option<String>,
    pub target: String,
    pub state: ModelLifecycleState,
    pub memory_bytes: u64,
    pub last_used: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RuntimeSnapshot {
    pub active_inferences: usize,
    pub active_model_loads: usize,
    pub queued_inferences: usize,
    pub loaded_models: usize,
    pub cache_entries: usize,
    pub cache_memory_bytes: u64,
    pub metrics: RuntimeMetricsSnapshot,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelLifecycleState {
    Discovered,
    Validated,
    Loaded,
    Ready,
    InUse,
    Evicted,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub selected_backend: Option<String>,
    pub selected_provider: Option<String>,
    pub prefer_acceleration: bool,
    pub allow_remote_fallback: bool,
    pub timeout_ms: Option<u64>,
    pub max_concurrency: usize,
    pub max_batch_size: Option<usize>,
    pub max_concurrent_inferences: usize,
    pub max_concurrent_model_loads: usize,
    pub max_models: usize,
    pub max_memory_bytes: Option<u64>,
    pub max_stream_buffer: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            selected_backend: None,
            selected_provider: Some("local".to_owned()),
            prefer_acceleration: false,
            allow_remote_fallback: false,
            timeout_ms: None,
            max_concurrency: 4,
            max_batch_size: Some(16),
            max_concurrent_inferences: 4,
            max_concurrent_model_loads: 1,
            max_models: 8,
            max_memory_bytes: None,
            max_stream_buffer: 32,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeSelection {
    pub model: String,
    pub provider: String,
    pub backend: Option<String>,
    pub hardware: Option<String>,
    pub fallback: bool,
    pub fallback_from: Option<String>,
    pub reason: String,
    pub policy: ExecutionPolicy,
}

#[derive(Clone, Debug)]
pub struct ModelRegistry {
    models: Vec<ModelCapability>,
}

impl ModelRegistry {
    pub fn list(&self) -> &[ModelCapability] {
        &self.models
    }
}

#[derive(Clone, Debug)]
pub struct ProviderRegistry {
    providers: Vec<ProviderCapability>,
}

impl ProviderRegistry {
    pub fn list(&self) -> &[ProviderCapability] {
        &self.providers
    }
}

#[derive(Clone, Debug)]
pub struct BackendRegistry {
    backends: Vec<BackendCapability>,
}

impl BackendRegistry {
    pub fn list(&self) -> &[BackendCapability] {
        &self.backends
    }
}

#[derive(Clone)]
struct LoadedModel {
    spec: ModelSpec,
    handle: ModelHandle,
    backend_name: String,
    provider_name: String,
    last_used: u64,
    memory_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
/// Opaque runtime-owned reference to a loaded model resource.
pub struct RuntimeModelHandle {
    model_id: String,
    version: Option<String>,
    provider: String,
    backend: String,
    state: ModelLifecycleState,
}

impl RuntimeModelHandle {
    pub fn id(&self) -> &str {
        &self.model_id
    }
    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }
    pub fn provider(&self) -> &str {
        &self.provider
    }
    pub fn backend(&self) -> &str {
        &self.backend
    }
    pub fn state(&self) -> &ModelLifecycleState {
        &self.state
    }
}

#[derive(Clone, Debug, Eq)]
struct ModelCacheKey {
    id: String,
    version: Option<String>,
    provider: String,
    backend: String,
}

impl PartialEq for ModelCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.version == other.version
            && self.provider == other.provider
            && self.backend == other.backend
    }
}

impl Hash for ModelCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.version.hash(state);
        self.provider.hash(state);
        self.backend.hash(state);
    }
}

/// Explicit construction and extension surface for an application-owned [`Runtime`].
pub struct RuntimeBuilder {
    config: RuntimeConfig,
    backends: BTreeMap<String, Arc<dyn Backend>>,
    providers: BTreeMap<String, Arc<dyn Provider>>,
    decision_providers: BTreeMap<String, Arc<dyn DecisionModelProvider>>,
    catalog: Option<Arc<dyn ModelCatalog>>,
    capability_registry: CapabilityRegistry,
    model_root: Option<std::path::PathBuf>,
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        let mut builder = Self {
            config: RuntimeConfig::default(),
            backends: BTreeMap::new(),
            providers: BTreeMap::new(),
            decision_providers: BTreeMap::new(),
            catalog: None,
            capability_registry: CapabilityRegistry::with_builtins(),
            model_root: default_model_root().ok(),
        };
        builder
            .backends
            .insert("cpu".to_owned(), Arc::new(CpuBackend));
        builder
            .backends
            .insert("onnx".to_owned(), Arc::new(OnnxBackend));
        #[cfg(feature = "coreml")]
        builder.decision_providers.insert(
            "coreml".to_owned(),
            Arc::new(ml_runtime_coreml_backend::CoreMlBackend),
        );
        builder
    }
}

impl RuntimeBuilder {
    pub fn backend(mut self, name: impl Into<String>) -> Self {
        self.config.selected_backend = Some(name.into());
        self
    }

    pub fn provider(mut self, name: impl Into<String>) -> Self {
        self.config.selected_provider = Some(name.into());
        self
    }

    /// Configures the built-in HTTP provider as the selected remote target.
    pub fn remote(mut self, endpoint: impl Into<String>) -> Self {
        let provider = HttpProvider::new(endpoint);
        self.providers
            .insert(provider.name().to_owned(), Arc::new(provider));
        self
    }

    pub fn catalog<C>(mut self, catalog: C) -> Self
    where
        C: ModelCatalog + 'static,
    {
        self.catalog = Some(Arc::new(catalog));
        self
    }

    pub fn model_root(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.model_root = Some(root.into());
        self
    }

    pub fn prefer_acceleration(mut self, enabled: bool) -> Self {
        self.config.prefer_acceleration = enabled;
        self
    }

    pub fn allow_remote_fallback(mut self, enabled: bool) -> Self {
        self.config.allow_remote_fallback = enabled;
        self
    }

    pub fn timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.config.timeout_ms = Some(timeout_ms);
        self
    }

    pub fn concurrency(mut self, max_concurrency: usize) -> Self {
        self.config.max_concurrency = max_concurrency.max(1);
        self.config.max_concurrent_inferences = max_concurrency.max(1);
        self
    }

    pub fn max_concurrent_inferences(mut self, limit: usize) -> Self {
        self.config.max_concurrent_inferences = limit.max(1);
        self.config.max_concurrency = limit.max(1);
        self
    }

    pub fn max_concurrent_model_loads(mut self, limit: usize) -> Self {
        self.config.max_concurrent_model_loads = limit.max(1);
        self
    }

    pub fn max_models(mut self, limit: usize) -> Self {
        self.config.max_models = limit.max(1);
        self
    }

    pub fn max_memory_bytes(mut self, limit: u64) -> Self {
        self.config.max_memory_bytes = Some(limit);
        self
    }

    pub fn max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.config.max_batch_size = Some(max_batch_size.max(1));
        self
    }

    /// Registers an experimental/custom backend extension by its declared name.
    pub fn register_backend<B>(mut self, backend: B) -> Self
    where
        B: Backend + 'static,
    {
        let name = backend.name().to_owned();
        self.backends.insert(name, Arc::new(backend));
        self
    }

    pub fn register_backend_arc(mut self, backend: Arc<dyn Backend>) -> Self {
        self.backends.insert(backend.name().to_owned(), backend);
        self
    }

    /// Registers an experimental/custom provider extension by its declared name.
    pub fn register_provider<P>(mut self, provider: P) -> Self
    where
        P: Provider + 'static,
    {
        let name = provider.name().to_owned();
        self.providers.insert(name, Arc::new(provider));
        self
    }

    pub fn register_provider_arc(mut self, provider: Arc<dyn Provider>) -> Self {
        self.providers.insert(provider.name().to_owned(), provider);
        self
    }

    pub fn register_decision_provider<P>(mut self, provider: P) -> Self
    where
        P: DecisionModelProvider + 'static,
    {
        self.decision_providers
            .insert(provider.name().to_owned(), Arc::new(provider));
        self
    }

    pub fn register_capability_provider<P>(mut self, provider: P) -> Self
    where
        P: CapabilityProvider + 'static,
    {
        self.capability_registry.register(provider);
        self
    }

    pub fn build(self) -> Runtime {
        Runtime {
            config: self.config.clone(),
            backends: self.backends,
            providers: self.providers,
            decision_providers: self.decision_providers,
            catalog: self.catalog,
            capability_registry: self.capability_registry,
            model_root: self.model_root,
            loaded_models: RwLock::new(HashMap::new()),
            inference_semaphore: Arc::new(Semaphore::new(
                self.config.max_concurrent_inferences.max(1),
            )),
            load_semaphore: Arc::new(Semaphore::new(
                self.config.max_concurrent_model_loads.max(1),
            )),
            cache_clock: AtomicU64::new(0),
            metrics: Arc::new(RuntimeMetrics::default()),
            execution_clock: AtomicU64::new(0),
            execution_records: Mutex::new(VecDeque::new()),
        }
    }
}

/// Coordinates catalog resolution, routing, model lifecycle, and inference.
///
/// A runtime is explicit application state; it does not create global state or
/// background services. Use [`Runtime::builder`] to obtain built-in CPU and
/// ONNX execution.
pub struct Runtime {
    config: RuntimeConfig,
    backends: BTreeMap<String, Arc<dyn Backend>>,
    providers: BTreeMap<String, Arc<dyn Provider>>,
    decision_providers: BTreeMap<String, Arc<dyn DecisionModelProvider>>,
    catalog: Option<Arc<dyn ModelCatalog>>,
    capability_registry: CapabilityRegistry,
    model_root: Option<std::path::PathBuf>,
    loaded_models: RwLock<HashMap<ModelCacheKey, LoadedModel>>,
    inference_semaphore: Arc<Semaphore>,
    load_semaphore: Arc<Semaphore>,
    cache_clock: AtomicU64,
    metrics: Arc<RuntimeMetrics>,
    execution_clock: AtomicU64,
    execution_records: Mutex<VecDeque<ExecutionRecord>>,
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

impl Runtime {
    pub fn new() -> Self {
        Self::builder().build()
    }

    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Loads an immutable local typed-decision artifact through the single matching backend.
    /// There is no fallback: missing, ambiguous, or invalid artifacts fail closed.
    pub fn load_decision_model(
        &self,
        artifact: impl AsRef<Path>,
    ) -> RuntimeResult<Box<dyn DecisionModel>> {
        let artifact = artifact.as_ref();
        if !artifact.exists() {
            return Err(RuntimeError::ModelNotFound {
                model: artifact.display().to_string(),
            });
        }
        if !artifact.is_dir() {
            return Err(RuntimeError::ModelUnavailable {
                model: artifact.display().to_string(),
                provider: "local".to_owned(),
                reason: "typed-decision artifact must be a directory".to_owned(),
            });
        }
        let candidates: Vec<_> = self
            .decision_providers
            .values()
            .filter(|provider| provider.supports_artifact(artifact))
            .collect();
        match candidates.as_slice() {
            [] => Err(RuntimeError::UnsupportedModel {
                model: artifact.display().to_string(),
                backend: "none".to_owned(),
                reason: "no registered typed-decision provider accepts this artifact".to_owned(),
            }),
            [provider] => provider.load_decision_model(artifact),
            _ => Err(RuntimeError::CapabilityMismatch {
                reason: format!(
                    "multiple typed-decision providers accept {}",
                    artifact.display()
                ),
            }),
        }
    }

    fn registered_model_root(&self) -> RuntimeResult<&Path> {
        self.model_root
            .as_deref()
            .ok_or_else(|| RuntimeError::ModelUnavailable {
                model: "registered models".to_owned(),
                provider: "local".to_owned(),
                reason: "model directory unavailable; set ML_RUNTIME_MODEL_DIR".to_owned(),
            })
    }

    pub fn installed_models(&self) -> RuntimeResult<Vec<InstalledModelRecord>> {
        let root = self.registered_model_root()?;
        let installer = RegisteredModelInstaller::new(root)
            .map_err(|error| RuntimeError::execution("model list", error.to_string()))?;
        Ok(BuiltinModelRegistry
            .list()
            .into_iter()
            .map(|model| {
                let package = installer.installed_path(&model);
                let (status, compiled) = if !package.exists() {
                    (InstalledModelStatus::NotInstalled, false)
                } else if model.backend == "coreml" && !cfg!(target_os = "macos") {
                    (InstalledModelStatus::Unsupported, false)
                } else {
                    match InstallationManifest::read(&package) {
                        Ok(manifest)
                            if manifest.model == model.name
                                && manifest.model_revision == model.revision
                                && manifest.runtime_requirement == model.runtime_requirement
                                && ml_runtime_model::runtime_requirement_satisfied(
                                    &manifest.runtime_requirement,
                                    VERSION,
                                )
                                && manifest.os == std::env::consts::OS
                                && manifest.architecture == std::env::consts::ARCH
                                && manifest.compiled_status == CompiledStatus::Ready
                                && manifest
                                    .compiled_artifact
                                    .as_ref()
                                    .is_some_and(|artifact| artifact.path.is_dir()) =>
                        {
                            (InstalledModelStatus::Ready, true)
                        }
                        Ok(manifest) if manifest.compiled_status != CompiledStatus::Ready => {
                            (InstalledModelStatus::InstalledNotCompiled, false)
                        }
                        Ok(_) => (InstalledModelStatus::Incompatible, false),
                        Err(_) => (InstalledModelStatus::Corrupt, false),
                    }
                };
                InstalledModelRecord {
                    model: model.name,
                    revision: model.revision,
                    backend: model.backend,
                    platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
                    status,
                    compiled,
                    package_path: package,
                }
            })
            .collect())
    }

    pub async fn install_registered_model(
        &self,
        name: &str,
        source_override: Option<&str>,
        replace: bool,
    ) -> RuntimeResult<ModelLifecycleResult> {
        let started = Instant::now();
        let model =
            BuiltinModelRegistry
                .resolve(name)
                .map_err(|error| RuntimeError::ModelNotFound {
                    model: error.to_string(),
                })?;
        if model.backend == "coreml" && !cfg!(target_os = "macos") {
            return Err(RuntimeError::backend_unavailable(
                "coreml",
                "Laya installation requires macOS; Core ML is unavailable on this platform",
            ));
        }
        let root = self.registered_model_root()?;
        let installer = RegisteredModelInstaller::new(root)
            .map_err(|error| RuntimeError::execution("model install", error.to_string()))?;
        let installed = installer
            .install(
                &model,
                source_override,
                if replace {
                    ml_runtime_model::InstallMode::Replace
                } else {
                    ml_runtime_model::InstallMode::KeepExisting
                },
                None,
            )
            .await
            .map_err(|error| RuntimeError::execution("model install", error.to_string()))?;

        #[cfg(feature = "coreml")]
        let prepared = if model.backend == "coreml" {
            if installed.already_installed && !replace {
                ml_runtime_coreml_backend::CoreMlBackend::validate_prepared_laya(
                    &installed.package_path,
                )
                .or_else(|_| {
                    ml_runtime_coreml_backend::CoreMlBackend::prepare_laya(&installed.package_path)
                })?
            } else {
                ml_runtime_coreml_backend::CoreMlBackend::prepare_laya(&installed.package_path)?
            }
        } else {
            return Err(RuntimeError::UnsupportedModel {
                model: model.name.clone(),
                backend: model.backend.clone(),
                reason: "registered backend has no preparation lifecycle".to_owned(),
            });
        };
        #[cfg(not(feature = "coreml"))]
        let _ = (&started, &installed);
        #[cfg(not(feature = "coreml"))]
        return Err(RuntimeError::backend_unavailable(
            "coreml",
            "this runtime was built without Core ML support",
        ));

        #[cfg(feature = "coreml")]
        {
            let manifest = InstallationManifest {
                schema_version: 1,
                model: model.name.clone(),
                model_revision: model.revision.clone(),
                model_checksum: prepared.model_identity,
                runtime_version: VERSION.to_owned(),
                runtime_requirement: model.runtime_requirement.clone(),
                backend: model.backend.clone(),
                os: std::env::consts::OS.to_owned(),
                architecture: std::env::consts::ARCH.to_owned(),
                installed_at_unix_seconds: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                compiled_status: CompiledStatus::Ready,
                compiled_artifact: Some(CompiledArtifactManifest {
                    path: prepared.path,
                    identity: prepared.compiled_identity,
                }),
            };
            manifest
                .write_atomic(&installed.package_path)
                .map_err(|error| RuntimeError::execution("model install", error.to_string()))?;
            Ok(ModelLifecycleResult {
                model: InstalledModelRecord {
                    model: model.name,
                    revision: model.revision,
                    backend: model.backend,
                    platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
                    status: InstalledModelStatus::Ready,
                    compiled: true,
                    package_path: installed.package_path,
                },
                already_installed: installed.already_installed,
                duration_ms: started.elapsed().as_millis(),
            })
        }
    }

    pub fn doctor_registered_model(&self, name: &str) -> RuntimeResult<ModelDoctorReport> {
        let model =
            BuiltinModelRegistry
                .resolve(name)
                .map_err(|error| RuntimeError::ModelNotFound {
                    model: error.to_string(),
                })?;
        let root = self.registered_model_root()?;
        let installer = RegisteredModelInstaller::new(root)
            .map_err(|error| RuntimeError::execution("model doctor", error.to_string()))?;
        let package = installer.installed_path(&model);
        let mut diagnostics = Vec::new();
        if !package.is_dir() {
            diagnostics.push(ModelDiagnostic {
                check: "installation".to_owned(),
                ok: false,
                message: format!(
                    "model is not installed; run `ml-runtime model install {}`",
                    model.name
                ),
            });
            return Ok(ModelDoctorReport {
                model: InstalledModelRecord {
                    model: model.name,
                    revision: model.revision,
                    backend: model.backend,
                    platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
                    status: InstalledModelStatus::NotInstalled,
                    compiled: false,
                    package_path: package,
                },
                diagnostics,
                manifest: None,
            });
        }
        let artifact_valid = match installer.validate(&model, &package) {
            Ok(()) => {
                diagnostics.push(ModelDiagnostic {
                    check: "artifact".to_owned(),
                    ok: true,
                    message: "registered files, tokenizer, Core ML package, and SHA-256 checksums are valid".to_owned(),
                });
                true
            }
            Err(error) => {
                diagnostics.push(ModelDiagnostic {
                    check: "artifact".to_owned(),
                    ok: false,
                    message: format!(
                        "{error}; reinstall with `ml-runtime model install {} --replace`",
                        model.name
                    ),
                });
                false
            }
        };
        let manifest = InstallationManifest::read(&package).ok();
        let compatible = manifest.as_ref().is_some_and(|manifest| {
            manifest.schema_version == 1
                && manifest.model == model.name
                && manifest.model_revision == model.revision
                && manifest.backend == model.backend
                && ml_runtime_model::runtime_requirement_satisfied(
                    &manifest.runtime_requirement,
                    VERSION,
                )
                && manifest.os == std::env::consts::OS
                && manifest.architecture == std::env::consts::ARCH
        });
        diagnostics.push(ModelDiagnostic {
            check: "metadata".to_owned(),
            ok: compatible,
            message: if compatible {
                "installation metadata and runtime/backend compatibility are valid".to_owned()
            } else {
                format!(
                    "installation metadata is missing or incompatible; run `ml-runtime model install {} --replace`",
                    model.name
                )
            },
        });
        let supported = model.backend != "coreml" || cfg!(target_os = "macos");
        diagnostics.push(ModelDiagnostic {
            check: "platform".to_owned(),
            ok: supported,
            message: if supported {
                format!("{} is supported on this platform", model.backend)
            } else {
                "Core ML is only supported on macOS".to_owned()
            },
        });
        #[cfg(feature = "coreml")]
        let compiled_valid = supported
            && compatible
            && ml_runtime_coreml_backend::CoreMlBackend::validate_prepared_laya(&package).is_ok();
        #[cfg(not(feature = "coreml"))]
        let compiled_valid = false;
        diagnostics.push(ModelDiagnostic {
            check: "compiled_artifact".to_owned(),
            ok: compiled_valid,
            message: if compiled_valid {
                "compiled Core ML artifact identity and schema are valid".to_owned()
            } else {
                format!(
                    "compiled artifact is missing or invalid; run `ml-runtime model install {} --replace`",
                    model.name
                )
            },
        });
        let status = if !supported {
            InstalledModelStatus::Unsupported
        } else if !artifact_valid {
            InstalledModelStatus::Corrupt
        } else if !compatible {
            InstalledModelStatus::Incompatible
        } else if !compiled_valid {
            InstalledModelStatus::InstalledNotCompiled
        } else {
            InstalledModelStatus::Ready
        };
        Ok(ModelDoctorReport {
            model: InstalledModelRecord {
                model: model.name,
                revision: model.revision,
                backend: model.backend,
                platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
                compiled: compiled_valid,
                package_path: package,
                status,
            },
            diagnostics,
            manifest,
        })
    }

    pub fn remove_registered_model(&self, name: &str) -> RuntimeResult<InstalledModelRecord> {
        let model =
            BuiltinModelRegistry
                .resolve(name)
                .map_err(|error| RuntimeError::ModelNotFound {
                    model: error.to_string(),
                })?;
        let root = self.registered_model_root()?;
        let installer = RegisteredModelInstaller::new(root)
            .map_err(|error| RuntimeError::execution("model remove", error.to_string()))?;
        let package = installer.installed_path(&model);
        if package.exists() {
            #[cfg(feature = "coreml")]
            if model.backend == "coreml" {
                ml_runtime_coreml_backend::CoreMlBackend::remove_prepared_laya(&package)?;
            }
            std::fs::remove_dir_all(&package)
                .map_err(|error| RuntimeError::execution("model remove", error.to_string()))?;
        }
        Ok(InstalledModelRecord {
            model: model.name,
            revision: model.revision,
            backend: model.backend,
            platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            status: InstalledModelStatus::NotInstalled,
            compiled: false,
            package_path: package,
        })
    }

    /// Loads a validated installed model by its registry name. Source, artifact
    /// layout, tokenizer, and backend selection remain runtime concerns.
    pub fn load_model(&self, name: &str) -> RuntimeResult<Box<dyn DecisionModel>> {
        let registered =
            BuiltinModelRegistry
                .resolve(name)
                .map_err(|error| RuntimeError::ModelNotFound {
                    model: error.to_string(),
                })?;
        let root = self
            .model_root
            .as_ref()
            .ok_or_else(|| RuntimeError::ModelUnavailable {
                model: name.to_owned(),
                provider: "local".to_owned(),
                reason: "model directory unavailable; set ML_RUNTIME_MODEL_DIR".to_owned(),
            })?;
        let installer = RegisteredModelInstaller::new(root).map_err(|error| {
            RuntimeError::execution("installed model validation", error.to_string())
        })?;
        let package = installer.installed_path(&registered);
        if !package.exists() {
            return Err(RuntimeError::ModelNotFound {
                model: format!("{name}; install it with `ml-runtime model install {name}`"),
            });
        }
        let manifest =
            InstallationManifest::read(&package).map_err(|error| RuntimeError::ModelIntegrity {
                model: name.to_owned(),
                reason: format!("{error}; run `ml-runtime model doctor {name}`"),
            })?;
        let compatible = manifest.schema_version == 1
            && manifest.model == registered.name
            && manifest.model_revision == registered.revision
            && manifest.backend == registered.backend
            && ml_runtime_model::runtime_requirement_satisfied(
                &manifest.runtime_requirement,
                VERSION,
            )
            && manifest.os == std::env::consts::OS
            && manifest.architecture == std::env::consts::ARCH
            && manifest.compiled_status == CompiledStatus::Ready
            && manifest
                .compiled_artifact
                .as_ref()
                .is_some_and(|artifact| artifact.path.is_dir());
        if !compatible {
            return Err(RuntimeError::ModelUnavailable {
                model: name.to_owned(),
                provider: "local".to_owned(),
                reason: format!(
                    "installation is incomplete, incompatible, or not compiled; run `ml-runtime model doctor {name}`"
                ),
            });
        }
        self.load_decision_model(package)
    }

    pub fn set_catalog<C>(&mut self, catalog: C)
    where
        C: ModelCatalog + 'static,
    {
        self.catalog = Some(Arc::new(catalog));
    }

    pub async fn catalog_models(&self) -> RuntimeResult<Vec<ModelDescriptor>> {
        self.catalog
            .as_ref()
            .ok_or_else(|| RuntimeError::execution("catalog", "no model catalog configured"))?
            .list()
            .await
            .map_err(catalog_error)
    }

    pub async fn resolve_model(
        &self,
        reference: &ModelReference,
    ) -> RuntimeResult<ModelDescriptor> {
        self.catalog
            .as_ref()
            .ok_or_else(|| RuntimeError::execution("catalog", "no model catalog configured"))?
            .resolve(reference)
            .await
            .map_err(catalog_error)
    }

    pub async fn refresh_catalog(&self) -> RuntimeResult<()> {
        self.catalog
            .as_ref()
            .ok_or_else(|| RuntimeError::execution("catalog", "no model catalog configured"))?
            .refresh()
            .await
            .map_err(catalog_error)
    }

    pub fn models(&self) -> ModelRegistry {
        let models = self
            .loaded_models
            .read()
            .expect("loaded model registry poisoned")
            .values()
            .map(|entry| ModelCapability {
                id: entry.spec.id.clone(),
                format: entry.spec.format.clone(),
                location: entry.spec.location.clone(),
                loaded: true,
                provider: Some(entry.provider_name.clone()),
                backend: Some(entry.backend_name.clone()),
            })
            .collect();
        ModelRegistry { models }
    }

    pub fn providers(&self) -> ProviderRegistry {
        let providers = self
            .providers
            .iter()
            .map(|(name, provider)| {
                ProviderCapability::from_parts(name.clone(), provider.capabilities())
            })
            .collect();
        ProviderRegistry { providers }
    }

    pub fn backends(&self) -> BackendRegistry {
        let backends = self
            .backends
            .iter()
            .map(|(name, backend)| {
                BackendCapability::from_parts(name.clone(), backend.capabilities())
            })
            .collect();
        BackendRegistry { backends }
    }

    pub fn capabilities(&self) -> RuntimeCapabilities {
        RuntimeCapabilities {
            environment: self.environment(),
            platforms: vec![Platform::current()],
            providers: self.providers().providers,
            backends: self.backends().backends,
            models: self.models().models,
            capabilities: self.capability_registry.list(),
        }
    }

    pub fn capability(&self, id: &str) -> Option<Capability> {
        self.capability_registry.get(id)
    }

    pub fn capability_registry(&self) -> &CapabilityRegistry {
        &self.capability_registry
    }

    /// Explicitly discovers and registers local capability package metadata.
    ///
    /// Discovery never loads or executes a provider binary.
    pub fn load_capability_packages(
        &mut self,
        root: impl AsRef<std::path::Path>,
    ) -> RuntimeResult<Vec<String>> {
        self.capability_registry.load_local_packages(root)
    }

    pub fn environment(&self) -> RuntimeEnvironment {
        RuntimeEnvironment::discover(
            self.capability_registry
                .providers()
                .iter()
                .map(|provider| provider.id().to_owned())
                .collect(),
        )
    }

    pub fn resolve_capability(
        &self,
        request: &ExecutionRequest,
    ) -> RuntimeResult<CapabilityResolution> {
        self.resolve_capability_with_constraints(request, &ResolutionConstraints::default())
    }

    pub fn resolve_capability_with_constraints(
        &self,
        request: &ExecutionRequest,
        constraints: &ResolutionConstraints,
    ) -> RuntimeResult<CapabilityResolution> {
        self.capability_registry
            .resolve(&request.capability, constraints)
    }

    pub async fn execute_capability(
        &self,
        request: ExecutionRequest,
        authorizer: Option<&dyn CapabilityAuthorizer>,
    ) -> RuntimeResult<ExecutionResult> {
        let resolution = self
            .capability_registry
            .resolve(&request.capability, &ResolutionConstraints::default())?;
        let provider = self
            .capability_registry
            .provider(&resolution.provider)
            .ok_or_else(|| {
                RuntimeError::provider_unavailable(&resolution.provider, "not registered")
            })?;
        if resolution.provider == "builtin" {
            capability::execute(request, authorizer).await
        } else {
            if let Some(authorizer) = authorizer {
                if request.policy.require_authorization {
                    let capability = self
                        .capability_registry
                        .get(&request.capability)
                        .ok_or_else(|| RuntimeError::CapabilityMismatch {
                            reason: format!("unknown capability {}", request.capability),
                        })?;
                    authorizer.authorize(&request, &capability).await?;
                }
            } else if self
                .capability_registry
                .get(&request.capability)
                .is_some_and(|capability| !capability.requirements.is_empty())
            {
                return Err(RuntimeError::Execution {
                    operation: request.capability,
                    reason: "authorization required".to_owned(),
                });
            }
            provider.execute(request).await
        }
    }

    pub fn status(&self) -> RuntimeStatus {
        RuntimeStatus {
            loaded_models: self
                .loaded_models
                .read()
                .expect("loaded model registry poisoned")
                .len(),
            max_models: self.config.max_models,
            max_concurrent_inferences: self.config.max_concurrent_inferences,
            max_concurrent_model_loads: self.config.max_concurrent_model_loads,
        }
    }

    pub fn metrics(&self) -> RuntimeMetricsSnapshot {
        self.metrics.snapshot()
    }

    pub fn executions(&self) -> Vec<ExecutionRecord> {
        self.execution_records
            .lock()
            .expect("execution records poisoned")
            .iter()
            .cloned()
            .collect()
    }

    pub fn resources(&self) -> Vec<ModelResource> {
        self.loaded_models
            .read()
            .expect("loaded model registry poisoned")
            .values()
            .map(|model| ModelResource {
                id: model.spec.id.clone(),
                version: model.spec.version.clone(),
                target: format!("{}/{}", model.provider_name, model.backend_name),
                state: ModelLifecycleState::Ready,
                memory_bytes: model.memory_bytes,
                last_used: model.last_used,
            })
            .collect()
    }

    pub fn snapshot(&self) -> RuntimeSnapshot {
        let resources = self.resources();
        let metrics = self.metrics();
        RuntimeSnapshot {
            active_inferences: metrics.active_inferences,
            active_model_loads: metrics.active_model_loads,
            queued_inferences: (metrics.queued_requests as usize)
                .saturating_sub(metrics.active_inferences),
            loaded_models: resources.len(),
            cache_entries: resources.len(),
            cache_memory_bytes: resources.iter().map(|resource| resource.memory_bytes).sum(),
            metrics,
        }
    }

    pub fn selection_for(&self, model: &ModelSpec) -> RuntimeResult<RuntimeSelection> {
        let backend_name = self.select_backend(model)?;
        let backend = self
            .backends
            .get(&backend_name)
            .ok_or_else(|| RuntimeError::backend_unavailable(&backend_name, "not registered"))?;
        Ok(RuntimeSelection {
            model: model.id.clone(),
            provider: "local".to_owned(),
            backend: Some(backend_name),
            hardware: backend.capabilities().hardware,
            fallback: false,
            fallback_from: None,
            reason: "preferred local execution".to_owned(),
            policy: ExecutionPolicy::LocalOnly,
        })
    }

    /// Explicitly loads a model for repeated execution with [`Self::infer_loaded`].
    /// Normal one-off requests should use [`Self::infer`], which loads lazily.
    pub async fn load(&self, model: ModelSpec) -> RuntimeResult<RuntimeModelHandle> {
        if self
            .config
            .selected_provider
            .as_deref()
            .is_some_and(|provider| provider != "local")
        {
            return Err(RuntimeError::provider_unavailable(
                self.config
                    .selected_provider
                    .clone()
                    .unwrap_or_else(|| "remote".to_owned()),
                "load is only supported for local execution paths",
            ));
        }

        let backend_name = self.select_backend(&model)?;
        let key = ModelCacheKey {
            id: model.id.clone(),
            version: model.version.clone(),
            provider: "local".to_owned(),
            backend: backend_name.clone(),
        };
        if let Some(existing) = self
            .loaded_models
            .read()
            .expect("loaded model registry poisoned")
            .get(&key)
        {
            self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(runtime_handle(existing));
        }
        self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .active_model_loads
            .fetch_add(1, Ordering::Relaxed);
        let _permit = self
            .load_semaphore
            .acquire()
            .await
            .map_err(|_| RuntimeError::execution("load", "runtime load semaphore closed"))?;
        if let Some(existing) = self
            .loaded_models
            .read()
            .expect("loaded model registry poisoned")
            .get(&key)
        {
            self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(runtime_handle(existing));
        }
        let backend = self
            .backends
            .get(&backend_name)
            .ok_or_else(|| RuntimeError::backend_unavailable(&backend_name, "not registered"))?
            .clone();
        let timeout_ms = self.config.timeout_ms;
        let loaded = run_with_timeout(timeout_ms, "load", backend.load(&model)).await;
        self.metrics
            .active_model_loads
            .fetch_sub(1, Ordering::Relaxed);
        let handle = match loaded {
            Ok(handle) => handle,
            Err(error) => {
                self.metrics
                    .model_load_failures
                    .fetch_add(1, Ordering::Relaxed);
                return Err(error);
            }
        };

        let mut cache = self
            .loaded_models
            .write()
            .expect("loaded model registry poisoned");
        if cache.len() >= self.config.max_models {
            if let Some(oldest) = cache
                .iter()
                .min_by_key(|(_, model)| model.last_used)
                .map(|(key, _)| key.clone())
            {
                cache.remove(&oldest);
                self.metrics.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.metrics.model_loads.fetch_add(1, Ordering::Relaxed);
        let loaded = LoadedModel {
            memory_bytes: model_memory_bytes(&model),
            spec: model,
            handle,
            backend_name,
            provider_name: "local".to_owned(),
            last_used: self.cache_clock.fetch_add(1, Ordering::Relaxed),
        };
        let runtime_handle = runtime_handle(&loaded);
        cache.insert(key, loaded);
        Ok(runtime_handle)
    }

    pub async fn load_reference(
        &self,
        reference: ModelReference,
    ) -> RuntimeResult<RuntimeModelHandle> {
        match reference {
            ModelReference::Spec(spec) => self.load(spec).await,
            ModelReference::Id { id, version } => self
                .find_loaded(&id, version.as_ref())
                .map(|loaded| runtime_handle(&loaded)),
        }
    }

    /// Resolves, loads, routes, and executes one request.
    pub async fn infer(&self, request: InferenceRequest) -> RuntimeResult<InferenceResult> {
        self.infer_with_cancellation(request, CancellationToken::new())
            .await
    }

    /// Executes against a previously loaded handle without exposing cache internals.
    pub async fn infer_loaded(
        &self,
        model: &RuntimeModelHandle,
        input: Input,
        options: InferenceOptions,
    ) -> RuntimeResult<InferenceResult> {
        self.infer(InferenceRequest {
            model: ModelReference::id(model.id().to_owned(), model.version().map(str::to_owned)),
            input,
            options,
        })
        .await
    }

    pub async fn infer_loaded_stream(
        &self,
        model: &RuntimeModelHandle,
        input: Input,
        mut options: InferenceOptions,
    ) -> RuntimeResult<InferenceStream> {
        options.stream = true;
        self.infer_stream(InferenceRequest {
            model: ModelReference::id(model.id().to_owned(), model.version().map(str::to_owned)),
            input,
            options,
        })
        .await
    }

    /// Executes requests in stable input order.
    ///
    /// The call fails as a whole on a request error. Backends without native
    /// batching are invoked individually behind the same API.
    pub async fn infer_batch(
        &self,
        mut requests: Vec<InferenceRequest>,
    ) -> RuntimeResult<Vec<InferenceResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(limit) = self.config.max_batch_size {
            if requests.len() > limit {
                return Err(RuntimeError::ResourceLimit {
                    reason: format!(
                        "batch size {} exceeds configured limit {limit}",
                        requests.len()
                    ),
                });
            }
        }
        for request in &requests {
            self.validate_request(request)?;
        }
        for request in &mut requests {
            self.resolve_catalog_reference_for_local(request).await?;
        }
        let selection = self.select_for_request(&requests[0])?;
        let same_model = requests
            .iter()
            .all(|request| model_id(&request.model) == model_id(&requests[0].model));
        if selection.provider != "local" || !same_model {
            return futures_util::future::try_join_all(
                requests.into_iter().map(|request| self.infer(request)),
            )
            .await;
        }
        let loaded = self.resolve_loaded_model(&requests[0].model).await?;
        let backend = self
            .backends
            .get(&loaded.backend_name)
            .ok_or_else(|| {
                RuntimeError::backend_unavailable(&loaded.backend_name, "not registered")
            })?
            .clone();
        if !backend.capabilities().batching {
            return futures_util::future::try_join_all(
                requests.into_iter().map(|request| self.infer(request)),
            )
            .await;
        }
        let batch_size = requests.len();
        self.metrics
            .total_requests
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        self.metrics
            .batch_executions
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .total_requests_batched
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        self.metrics
            .batch_sizes
            .lock()
            .expect("metrics poisoned")
            .push(batch_size);
        let _permit = self.inference_semaphore.acquire().await.map_err(|_| {
            RuntimeError::execution("infer_batch", "runtime inference semaphore closed")
        })?;
        let mut results = run_with_timeout(
            self.config.timeout_ms,
            "infer_batch",
            backend.infer_batch(&loaded.handle, &requests, Some(CancellationToken::new())),
        )
        .await?;
        if results.len() != batch_size {
            return Err(RuntimeError::execution(
                "infer_batch",
                format!(
                    "backend returned {} results for {batch_size} requests",
                    results.len()
                ),
            ));
        }
        for (result, request) in results.iter_mut().zip(requests.iter()) {
            let output = result.output.clone();
            decorate_metadata(
                &mut result.metadata,
                &output,
                &loaded.spec.id,
                "local",
                &loaded.backend_name,
                backend.capabilities().hardware.clone(),
                Duration::ZERO,
                estimate_input_tokens(&request.input),
                loaded.spec.version.clone(),
                selection.policy.clone(),
                false,
                None,
                Some(selection.reason.clone()),
            );
            result.metadata.batch_size = batch_size;
            result.metadata.cache_hit = true;
        }
        self.metrics
            .successful_requests
            .fetch_add(results.len() as u64, Ordering::Relaxed);
        Ok(results)
    }

    /// Executes one request while propagating cancellation through scheduling,
    /// provider transport, and backend execution.
    pub async fn infer_with_cancellation(
        &self,
        request: InferenceRequest,
        cancellation: CancellationToken,
    ) -> RuntimeResult<InferenceResult> {
        let started = Instant::now();
        let request_id = format!(
            "request-{}",
            self.execution_clock.fetch_add(1, Ordering::Relaxed) + 1
        );
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .active_inferences
            .fetch_add(1, Ordering::Relaxed);
        let model = model_details(&request.model);
        let result = self.infer_inner(request.clone(), cancellation).await;
        self.metrics
            .active_inferences
            .fetch_sub(1, Ordering::Relaxed);
        match result {
            Ok(mut result) => {
                result.metadata.request_id = Some(request_id.clone());
                self.record_execution(
                    &request_id,
                    started.elapsed(),
                    &result.metadata,
                    ExecutionCompletion::Succeeded,
                );
                self.metrics
                    .successful_requests
                    .fetch_add(1, Ordering::Relaxed);
                Ok(result)
            }
            Err(error) => {
                let completion = if matches!(error, RuntimeError::Cancelled) {
                    self.metrics
                        .cancelled_requests
                        .fetch_add(1, Ordering::Relaxed);
                    ExecutionCompletion::Cancelled
                } else {
                    self.metrics.failed_requests.fetch_add(1, Ordering::Relaxed);
                    ExecutionCompletion::Failed
                };
                let metadata = ExecutionMetadata {
                    request_id: Some(request_id.clone()),
                    model: model.0,
                    model_version: model.1,
                    routing_policy: request.options.execution.clone(),
                    streaming_requested: request.options.stream,
                    ..ExecutionMetadata::default()
                };
                self.record_execution(&request_id, started.elapsed(), &metadata, completion);
                Err(error)
            }
        }
    }

    async fn infer_inner(
        &self,
        mut request: InferenceRequest,
        cancellation: CancellationToken,
    ) -> RuntimeResult<InferenceResult> {
        self.validate_request(&request)?;
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }

        let _permit = tokio::select! {
            permit = self.inference_semaphore.acquire() => permit
                .map_err(|_| RuntimeError::execution("infer", "runtime inference semaphore closed"))?,
            _ = cancellation.cancelled() => return Err(RuntimeError::Cancelled),
        };

        self.resolve_catalog_reference_for_local(&mut request)
            .await?;

        let mut selection = self.select_for_request(&request)?;
        let timeout_ms = request.options.timeout_ms.or(self.config.timeout_ms);
        let input_tokens = estimate_input_tokens(&request.input);

        match selection.provider.as_str() {
            "local" => {
                let cache_hit = self.is_loaded(&request.model);
                let model_load_started = Instant::now();
                if cache_hit {
                    self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);
                }

                let loaded = match self.resolve_loaded_model(&request.model).await {
                    Ok(loaded) => loaded,
                    Err(error) if self.can_fallback(&request, &selection, &error) => {
                        selection = self.remote_selection(
                            model_id(&request.model),
                            true,
                            Some("local model unavailable".to_owned()),
                            &request.options,
                        )?;
                        return self
                            .infer_remote(
                                request,
                                cancellation,
                                timeout_ms,
                                input_tokens,
                                selection,
                            )
                            .await;
                    }
                    Err(error) => return Err(error),
                };
                let backend = self
                    .backends
                    .get(&loaded.backend_name)
                    .ok_or_else(|| {
                        RuntimeError::backend_unavailable(&loaded.backend_name, "not registered")
                    })?
                    .clone();
                let started = Instant::now();
                let result = run_with_timeout(
                    timeout_ms,
                    "infer",
                    backend.infer(&loaded.handle, &request, Some(cancellation.clone())),
                )
                .await;
                let mut result = match result {
                    Ok(result) => result,
                    Err(error) if self.can_fallback(&request, &selection, &error) => {
                        selection = self.remote_selection(
                            model_id(&request.model),
                            true,
                            Some(error.to_string()),
                            &request.options,
                        )?;
                        return self
                            .infer_remote(
                                request,
                                cancellation,
                                timeout_ms,
                                input_tokens,
                                selection,
                            )
                            .await;
                    }
                    Err(error) => return Err(error),
                };
                result.metadata.model_load_ms = Some(if cache_hit {
                    0.0
                } else {
                    model_load_started.elapsed().as_secs_f64() * 1000.0
                });
                let capabilities = backend.capabilities();
                let output = result.output.clone();
                decorate_metadata(
                    &mut result.metadata,
                    &output,
                    &loaded.spec.id,
                    "local",
                    &loaded.backend_name,
                    capabilities.hardware,
                    started.elapsed(),
                    input_tokens,
                    loaded.spec.version.clone(),
                    selection.policy.clone(),
                    selection.fallback,
                    selection.fallback_from.clone(),
                    Some(selection.reason.clone()),
                );
                result.metadata.cache_hit = cache_hit;
                Ok(result)
            }
            _ => {
                self.infer_remote(request, cancellation, timeout_ms, input_tokens, selection)
                    .await
            }
        }
    }

    fn record_execution(
        &self,
        request_id: &str,
        total_duration: Duration,
        metadata: &ExecutionMetadata,
        completion: ExecutionCompletion,
    ) {
        let execution_id = format!(
            "execution-{}",
            self.execution_clock.fetch_add(1, Ordering::Relaxed) + 1
        );
        let execution_duration =
            Duration::from_secs_f64(metadata.execution_ms.unwrap_or(0.0) / 1000.0);
        let queued_duration =
            Duration::from_secs_f64(metadata.queue_wait_ms.unwrap_or(0.0) / 1000.0);
        let model_load_duration =
            Duration::from_secs_f64(metadata.model_load_ms.unwrap_or(0.0) / 1000.0);
        self.metrics
            .queue_wait_micros
            .lock()
            .expect("metrics poisoned")
            .push(queued_duration.as_micros() as u64);
        self.metrics
            .model_load_micros
            .lock()
            .expect("metrics poisoned")
            .push(model_load_duration.as_micros() as u64);
        self.metrics
            .execution_micros
            .lock()
            .expect("metrics poisoned")
            .push(execution_duration.as_micros() as u64);
        self.metrics
            .total_duration_micros
            .lock()
            .expect("metrics poisoned")
            .push(total_duration.as_micros() as u64);
        if metadata.provider == "local" {
            self.metrics
                .local_executions
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.metrics
                .remote_executions
                .fetch_add(1, Ordering::Relaxed);
        }
        if metadata.fallback {
            self.metrics
                .fallback_executions
                .fetch_add(1, Ordering::Relaxed);
        }
        if metadata.streaming_requested {
            self.metrics
                .streaming_requests
                .fetch_add(1, Ordering::Relaxed);
        }
        let record = ExecutionRecord {
            execution_id,
            request_id: request_id.to_owned(),
            model_id: metadata.model.clone(),
            model_version: metadata.model_version.clone().unwrap_or_default(),
            provider: metadata.provider.clone(),
            backend: metadata.backend.clone(),
            target: metadata.execution_target.clone(),
            routing_policy: metadata.routing_policy.clone(),
            fallback: metadata.fallback_from.clone(),
            cache_hit: metadata.cache_hit,
            batch_size: metadata.batch_size.max(1),
            queued_duration,
            model_load_duration,
            execution_duration,
            total_duration,
            streaming: metadata.streaming_requested,
            output_event_count: metadata.output_event_count,
            completion,
        };
        let mut records = self
            .execution_records
            .lock()
            .expect("execution records poisoned");
        records.push_back(record);
        while records.len() > 256 {
            records.pop_front();
        }
    }

    async fn infer_remote(
        &self,
        request: InferenceRequest,
        cancellation: CancellationToken,
        timeout_ms: Option<u64>,
        input_tokens: Option<u64>,
        selection: RuntimeSelection,
    ) -> RuntimeResult<InferenceResult> {
        let provider_name = selection.provider.as_str();
        let provider = self
            .providers
            .get(provider_name)
            .ok_or_else(|| RuntimeError::provider_unavailable(provider_name, "not registered"))?
            .clone();
        let started = Instant::now();
        let mut result = run_with_timeout(
            timeout_ms,
            "infer",
            provider.infer(request.clone(), Some(cancellation)),
        )
        .await?;
        let backend_name = result.metadata.backend.clone();
        let hardware = result.metadata.hardware.clone();
        let output = result.output.clone();
        let (model, version) = model_details(&request.model);
        decorate_metadata(
            &mut result.metadata,
            &output,
            &model,
            provider_name,
            backend_name.as_str(),
            hardware,
            started.elapsed(),
            input_tokens,
            version,
            selection.policy,
            selection.fallback,
            selection.fallback_from,
            Some(selection.reason),
        );
        Ok(result)
    }

    /// Starts normalized `Started` → `Output*` → `Completed` streaming.
    pub async fn infer_stream(&self, request: InferenceRequest) -> RuntimeResult<InferenceStream> {
        self.infer_stream_with_cancellation(request, CancellationToken::new())
            .await
    }

    /// Starts a bounded stream with caller-controlled cancellation.
    pub async fn infer_stream_with_cancellation(
        &self,
        mut request: InferenceRequest,
        cancellation: CancellationToken,
    ) -> RuntimeResult<InferenceStream> {
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .streaming_requests
            .fetch_add(1, Ordering::Relaxed);
        self.validate_request(&request)?;
        if !request.options.stream {
            return Err(RuntimeError::CapabilityMismatch {
                reason: "infer_stream requires InferenceOptions::stream to be true".to_owned(),
            });
        }
        self.resolve_catalog_reference_for_local(&mut request)
            .await?;
        let selection = self.select_for_request(&request)?;
        let buffer = self.config.max_stream_buffer;
        if buffer == 0 {
            return Err(RuntimeError::ResourceLimit {
                reason: "max_stream_buffer must be greater than zero".to_owned(),
            });
        }
        match selection.provider.as_str() {
            "local" => {
                let loaded = self.resolve_loaded_model(&request.model).await?;
                let backend = self
                    .backends
                    .get(&loaded.backend_name)
                    .ok_or_else(|| {
                        RuntimeError::backend_unavailable(&loaded.backend_name, "not registered")
                    })?
                    .clone();
                let capabilities = backend.capabilities();
                let stream = if capabilities.streaming {
                    backend
                        .infer_stream(&loaded.handle, &request, Some(cancellation.clone()))
                        .await?
                } else {
                    self.metrics
                        .compatibility_streams
                        .fetch_add(1, Ordering::Relaxed);
                    let result = backend
                        .infer(&loaded.handle, &request, Some(cancellation.clone()))
                        .await?;
                    Box::pin(futures_util::stream::iter(vec![Ok(InferenceChunk {
                        output: result.output,
                        done: true,
                        metadata: Some(result.metadata),
                    })])) as BoxStream<_>
                };
                Ok(normalize_stream(
                    stream,
                    request,
                    selection,
                    capabilities.streaming,
                    true,
                    buffer,
                    cancellation,
                    self.metrics.clone(),
                ))
            }
            provider_name => {
                let provider = self
                    .providers
                    .get(provider_name)
                    .ok_or_else(|| {
                        RuntimeError::provider_unavailable(provider_name, "not registered")
                    })?
                    .clone();
                let capabilities = provider.capabilities();
                let stream = if capabilities.streaming {
                    provider
                        .infer_stream(request.clone(), Some(cancellation.clone()))
                        .await?
                } else {
                    self.metrics
                        .compatibility_streams
                        .fetch_add(1, Ordering::Relaxed);
                    let result = provider
                        .infer(request.clone(), Some(cancellation.clone()))
                        .await?;
                    Box::pin(futures_util::stream::iter(vec![Ok(InferenceChunk {
                        output: result.output,
                        done: true,
                        metadata: Some(result.metadata),
                    })])) as BoxStream<_>
                };
                Ok(normalize_stream(
                    stream,
                    request,
                    selection,
                    capabilities.streaming,
                    false,
                    buffer,
                    cancellation,
                    self.metrics.clone(),
                ))
            }
        }
    }

    fn validate_request(&self, request: &InferenceRequest) -> RuntimeResult<()> {
        if request.options.require_local && request.options.require_remote {
            return Err(RuntimeError::InvalidRequest {
                reason: "request cannot require both local and remote execution".to_owned(),
            });
        }
        if let Some(limit) = self.config.max_batch_size {
            let batch_size = request
                .options
                .batch_size
                .unwrap_or_else(|| request.input.batch_size());
            if batch_size > limit {
                return Err(RuntimeError::ResourceLimit {
                    reason: format!("batch size {batch_size} exceeds configured limit {limit}"),
                });
            }
        }
        Ok(())
    }

    async fn resolve_catalog_reference_for_local(
        &self,
        request: &mut InferenceRequest,
    ) -> RuntimeResult<()> {
        let local_resolution = request.options.require_local
            || (!request.options.require_remote
                && matches!(
                    request.options.execution,
                    ExecutionPolicy::LocalOnly
                        | ExecutionPolicy::PreferLocal
                        | ExecutionPolicy::LocalThenRemote
                ));
        if local_resolution && !self.is_loaded(&request.model) {
            if let (Some(catalog), ModelReference::Id { .. }) = (&self.catalog, &request.model) {
                let descriptor = catalog
                    .resolve(&request.model)
                    .await
                    .map_err(catalog_error)?;
                catalog
                    .validate(&descriptor.id)
                    .await
                    .map_err(catalog_error)?;
                request.model = ModelReference::Spec(descriptor.spec());
            }
        }
        Ok(())
    }

    async fn resolve_loaded_model(&self, reference: &ModelReference) -> RuntimeResult<LoadedModel> {
        match reference {
            ModelReference::Id { id, version } => self.find_loaded(id, version.as_ref()),
            ModelReference::Spec(spec) => {
                if let Ok(existing) = self.find_loaded(&spec.id, spec.version.as_ref()) {
                    return Ok(existing);
                }
                self.load(spec.clone()).await?;
                self.find_loaded(&spec.id, spec.version.as_ref())
            }
        }
    }

    fn find_loaded(&self, id: &str, version: Option<&String>) -> RuntimeResult<LoadedModel> {
        let mut cache = self
            .loaded_models
            .write()
            .expect("loaded model registry poisoned");
        let found = cache.iter_mut().find(|(key, loaded)| {
            key.id == id
                && version.is_none_or(|version| loaded.spec.version.as_ref() == Some(version))
        });
        if let Some((_, loaded)) = found {
            loaded.last_used = self.cache_clock.fetch_add(1, Ordering::Relaxed);
            return Ok(loaded.clone());
        }

        Err(RuntimeError::ModelNotFound {
            model: id.to_owned(),
        })
    }

    fn is_loaded(&self, reference: &ModelReference) -> bool {
        let (id, version) = model_details(reference);
        self.loaded_models
            .read()
            .expect("loaded model registry poisoned")
            .keys()
            .any(|key| {
                key.id == id
                    && version
                        .as_ref()
                        .is_none_or(|v| key.version.as_ref() == Some(v))
            })
    }

    fn select_for_request(&self, request: &InferenceRequest) -> RuntimeResult<RuntimeSelection> {
        let model_id = model_id(&request.model);
        let policy = if request.options.require_remote {
            ExecutionPolicy::RemoteOnly
        } else if request.options.require_local {
            ExecutionPolicy::LocalOnly
        } else if self.config.allow_remote_fallback {
            ExecutionPolicy::LocalThenRemote
        } else {
            request.options.execution.clone()
        };
        match policy {
            ExecutionPolicy::RemoteOnly => self.remote_selection(
                model_id,
                false,
                Some("remote-only policy".to_owned()),
                &request.options,
            ),
            ExecutionPolicy::LocalOnly => self.local_selection(&request.model, policy),
            ExecutionPolicy::PreferLocal => match self
                .local_selection(&request.model, policy.clone())
            {
                Ok(selection) => Ok(selection),
                Err(error) => {
                    self.remote_selection(model_id, true, Some(error.to_string()), &request.options)
                }
            },
            ExecutionPolicy::LocalThenRemote => match self
                .local_selection(&request.model, policy.clone())
            {
                Ok(selection) => Ok(selection),
                Err(error) => {
                    self.remote_selection(model_id, true, Some(error.to_string()), &request.options)
                }
            },
            ExecutionPolicy::PreferRemote | ExecutionPolicy::RemoteThenLocal => match self
                .remote_selection(
                    model_id.clone(),
                    false,
                    Some("preferred remote execution".to_owned()),
                    &request.options,
                ) {
                Ok(selection) => Ok(selection),
                Err(_) => self.local_selection(&request.model, policy),
            },
        }
    }

    fn local_selection(
        &self,
        reference: &ModelReference,
        policy: ExecutionPolicy,
    ) -> RuntimeResult<RuntimeSelection> {
        match reference {
            ModelReference::Id { id, version } => {
                let loaded = self.find_loaded(id, version.as_ref())?;
                Ok(RuntimeSelection {
                    model: loaded.spec.id,
                    provider: "local".to_owned(),
                    backend: Some(loaded.backend_name.clone()),
                    hardware: self
                        .backends
                        .get(&loaded.backend_name)
                        .and_then(|backend| backend.capabilities().hardware),
                    fallback: false,
                    fallback_from: None,
                    reason: "preferred local execution".to_owned(),
                    policy,
                })
            }
            ModelReference::Spec(spec) => {
                let backend = self.select_backend(spec)?;
                let hardware = self
                    .backends
                    .get(&backend)
                    .and_then(|b| b.capabilities().hardware);
                Ok(RuntimeSelection {
                    model: spec.id.clone(),
                    provider: "local".to_owned(),
                    backend: Some(backend),
                    hardware,
                    fallback: false,
                    fallback_from: None,
                    reason: "preferred local execution".to_owned(),
                    policy,
                })
            }
        }
    }

    fn remote_selection(
        &self,
        model: String,
        fallback: bool,
        reason: Option<String>,
        options: &InferenceOptions,
    ) -> RuntimeResult<RuntimeSelection> {
        let selected = if let Some(provider_name) = self.config.selected_provider.as_deref() {
            if provider_name != "local" {
                Some(provider_name.to_owned())
            } else {
                None
            }
        } else {
            None
        }
        .or_else(|| {
            self.providers.iter().find_map(|(name, provider)| {
                let capabilities = provider.capabilities();
                if capabilities.available && capabilities.remote {
                    Some(name.clone())
                } else {
                    None
                }
            })
        })
        .ok_or_else(|| {
            RuntimeError::provider_unavailable("remote", "no remote provider registered")
        })?;

        Ok(RuntimeSelection {
            model,
            provider: selected,
            backend: None,
            hardware: None,
            fallback,
            fallback_from: fallback.then(|| "local".to_owned()),
            reason: reason.unwrap_or_else(|| "remote execution selected by policy".to_owned()),
            policy: if fallback && options.execution == ExecutionPolicy::LocalOnly {
                ExecutionPolicy::LocalThenRemote
            } else {
                options.execution.clone()
            },
        })
    }

    fn can_fallback(
        &self,
        request: &InferenceRequest,
        selection: &RuntimeSelection,
        error: &RuntimeError,
    ) -> bool {
        matches!(
            selection.policy,
            ExecutionPolicy::LocalThenRemote | ExecutionPolicy::PreferLocal
        ) && !request.options.require_local
            && matches!(
                error,
                RuntimeError::ProviderUnavailable { .. }
                    | RuntimeError::ModelUnavailable { .. }
                    | RuntimeError::BackendUnavailable { .. }
                    | RuntimeError::ModelNotFound { .. }
                    | RuntimeError::CapabilityMismatch { .. }
                    | RuntimeError::Timeout { .. }
                    | RuntimeError::Transport { .. }
                    | RuntimeError::Protocol { .. }
            )
    }

    fn select_backend(&self, model: &ModelSpec) -> RuntimeResult<String> {
        if let Some(selected) = self.config.selected_backend.as_deref() {
            let backend = self
                .backends
                .get(selected)
                .ok_or_else(|| RuntimeError::backend_unavailable(selected, "not registered"))?;
            let capabilities = backend.capabilities();
            if !capabilities.available {
                return Err(RuntimeError::backend_unavailable(
                    selected,
                    "backend is not available on this platform",
                ));
            }
            if !backend.supports(model) {
                return Err(RuntimeError::UnsupportedModel {
                    model: model.id.clone(),
                    backend: selected.to_owned(),
                    reason: "model format is not supported".to_owned(),
                });
            }
            return Ok(selected.to_owned());
        }

        let mut candidates: Vec<_> = self
            .backends
            .iter()
            .filter_map(|(name, backend)| {
                let capabilities = backend.capabilities();
                if capabilities.available && backend.supports(model) {
                    Some((
                        name.clone(),
                        !capabilities.accelerators.is_empty(),
                        capabilities.hardware.clone(),
                    ))
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|left, right| left.0.cmp(&right.0));
        if self.config.prefer_acceleration {
            candidates
                .sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
        }

        candidates
            .into_iter()
            .next()
            .map(|candidate| candidate.0)
            .ok_or_else(|| {
                RuntimeError::backend_unavailable("local", "no compatible backend registered")
            })
    }
}

async fn run_with_timeout<F, T>(
    timeout_ms: Option<u64>,
    operation: &str,
    future: F,
) -> RuntimeResult<T>
where
    F: std::future::Future<Output = RuntimeResult<T>>,
{
    if let Some(timeout_ms) = timeout_ms {
        timeout(Duration::from_millis(timeout_ms), future)
            .await
            .map_err(|_| RuntimeError::Timeout {
                operation: operation.to_owned(),
            })?
    } else {
        future.await
    }
}

fn model_id(reference: &ModelReference) -> String {
    match reference {
        ModelReference::Id { id, .. } => id.clone(),
        ModelReference::Spec(spec) => spec.id.clone(),
    }
}

fn runtime_handle(loaded: &LoadedModel) -> RuntimeModelHandle {
    RuntimeModelHandle {
        model_id: loaded.spec.id.clone(),
        version: loaded.spec.version.clone(),
        provider: loaded.provider_name.clone(),
        backend: loaded.backend_name.clone(),
        state: ModelLifecycleState::Ready,
    }
}

fn model_details(reference: &ModelReference) -> (String, Option<String>) {
    match reference {
        ModelReference::Id { id, version } => (id.clone(), version.clone()),
        ModelReference::Spec(spec) => (spec.id.clone(), spec.version.clone()),
    }
}

fn catalog_error(error: CatalogError) -> RuntimeError {
    match error {
        CatalogError::ModelNotFound(model) => RuntimeError::ModelNotFound { model },
        CatalogError::InvalidIdentity(model) => RuntimeError::InvalidRequest {
            reason: format!("invalid model identity: {model}"),
        },
        CatalogError::ArtifactIntegrity { model, reason } => {
            RuntimeError::ModelIntegrity { model, reason }
        }
        other => RuntimeError::Execution {
            operation: "catalog".to_owned(),
            reason: other.to_string(),
        },
    }
}

fn estimate_input_tokens(input: &Input) -> Option<u64> {
    match input {
        Input::Text(text) => Some(text.split_whitespace().count() as u64),
        Input::Tokens(tokens) => Some(tokens.len() as u64),
        _ => None,
    }
}

fn estimate_output_tokens(output: &Output) -> Option<u64> {
    match output {
        Output::Text(text) => Some(text.split_whitespace().count() as u64),
        Output::Tokens(tokens) => Some(tokens.len() as u64),
        _ => None,
    }
}

fn model_memory_bytes(model: &ModelSpec) -> u64 {
    match &model.location {
        ModelLocation::Path(path) => std::fs::metadata(path)
            .map(|metadata| metadata.len())
            .unwrap_or(0),
        _ => 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn decorate_metadata(
    metadata: &mut ExecutionMetadata,
    output: &Output,
    model: &str,
    provider: &str,
    backend: &str,
    hardware: Option<String>,
    elapsed: Duration,
    input_tokens: Option<u64>,
    model_version: Option<String>,
    routing_policy: ExecutionPolicy,
    fallback: bool,
    fallback_from: Option<String>,
    fallback_reason: Option<String>,
) {
    metadata.model = model.to_owned();
    metadata.model_version = model_version;
    metadata.provider = provider.to_owned();
    metadata.backend = backend.to_owned();
    metadata.execution_target = provider.to_owned();
    metadata.routing_policy = routing_policy;
    metadata.fallback = fallback;
    metadata.fallback_from = fallback_from;
    metadata.fallback_reason = fallback_reason;
    metadata.hardware = hardware;
    metadata.latency_ms = Some(elapsed.as_secs_f64() * 1000.0);
    metadata.execution_ms = metadata.latency_ms;
    metadata.input_tokens = input_tokens;
    metadata.output_tokens = estimate_output_tokens(output);
}

#[allow(clippy::too_many_arguments)]
fn normalize_stream(
    mut source: BoxStream<RuntimeResult<InferenceChunk>>,
    request: InferenceRequest,
    selection: RuntimeSelection,
    streaming_supported: bool,
    cache_hit: bool,
    buffer: usize,
    cancellation: CancellationToken,
    metrics: Arc<RuntimeMetrics>,
) -> InferenceStream {
    let (sender, receiver) = mpsc::channel(buffer);
    tokio::spawn(async move {
        let (model, version) = model_details(&request.model);
        let mut metadata = ExecutionMetadata {
            model,
            model_version: version,
            provider: selection.provider.clone(),
            execution_target: selection.provider.clone(),
            routing_policy: selection.policy.clone(),
            fallback: selection.fallback,
            fallback_from: selection.fallback_from.clone(),
            fallback_reason: Some(selection.reason.clone()),
            streaming_requested: true,
            streaming_supported,
            cache_hit,
            completion_state: Some("started".to_owned()),
            ..ExecutionMetadata::default()
        };
        if sender
            .send(Ok(InferenceStreamEvent::Started(metadata.clone())))
            .await
            .is_err()
        {
            return;
        }

        let mut output_count = 0;
        while let Some(item) = tokio::select! {
            item = source.next() => item,
            _ = cancellation.cancelled() => {
                metrics.cancelled_streams.fetch_add(1, Ordering::Relaxed);
                let _ = sender.send(Err(RuntimeError::Cancelled)).await;
                return;
            }
        } {
            match item {
                Ok(chunk) => {
                    if let Some(chunk_metadata) = chunk.metadata {
                        metadata = chunk_metadata;
                    }
                    metadata.streaming_requested = true;
                    metadata.streaming_supported = streaming_supported;
                    metadata.cache_hit = cache_hit;
                    metadata.output_event_count = output_count + 1;
                    output_count += 1;
                    if sender
                        .send(Ok(InferenceStreamEvent::Output(chunk.output)))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    metrics
                        .streamed_output_events
                        .fetch_add(1, Ordering::Relaxed);
                    if chunk.done {
                        metadata.output_event_count = output_count;
                        metadata.completion_state = Some("completed".to_owned());
                        let _ = sender
                            .send(Ok(InferenceStreamEvent::Completed(metadata)))
                            .await;
                        return;
                    }
                }
                Err(error) => {
                    let _ = sender.send(Err(error)).await;
                    return;
                }
            }
        }
        metadata.output_event_count = output_count;
        metadata.completion_state = Some("completed".to_owned());
        let _ = sender
            .send(Ok(InferenceStreamEvent::Completed(metadata)))
            .await;
    });
    Box::pin(ReceiverStream::new(receiver))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use ml_runtime_backend::BackendCapabilities;
    use ml_runtime_common::BoxStream;
    use ml_runtime_cpu_backend::CpuBackend;
    use ml_runtime_inference::InferenceOptions;
    use ml_runtime_provider::{Provider, ProviderCapabilities};
    use serde_json::json;
    use tokio_stream::iter;

    #[derive(Clone)]
    struct BlockingBackend {
        started: Arc<Semaphore>,
    }

    #[async_trait]
    impl Backend for BlockingBackend {
        fn name(&self) -> &str {
            "blocking"
        }

        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                available: true,
                local: true,
                streaming: true,
                cancellation: true,
                supported_formats: vec![ModelFormat::Unknown],
                ..BackendCapabilities::default()
            }
        }

        fn supports(&self, model: &ModelSpec) -> bool {
            model.format == ModelFormat::Unknown
        }

        async fn load(&self, model: &ModelSpec) -> RuntimeResult<ModelHandle> {
            Ok(ModelHandle::new(
                model.id.clone(),
                self.name(),
                model.format.clone(),
                Arc::new(()),
            ))
        }

        async fn infer(
            &self,
            _model: &ModelHandle,
            _request: &InferenceRequest,
            cancellation: Option<CancellationToken>,
        ) -> RuntimeResult<InferenceResult> {
            self.started.add_permits(1);
            cancellation
                .expect("runtime supplies cancellation")
                .cancelled()
                .await;
            Err(RuntimeError::Cancelled)
        }

        async fn infer_stream(
            &self,
            _model: &ModelHandle,
            _request: &InferenceRequest,
            _cancellation: Option<CancellationToken>,
        ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>> {
            Ok(Box::pin(futures_util::stream::pending()))
        }
    }

    #[test]
    fn runtime_version_is_workspace_version() {
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
    }

    #[derive(Clone)]
    struct MockRemoteProvider;

    #[async_trait]
    impl Provider for MockRemoteProvider {
        fn name(&self) -> &str {
            "mock-remote"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                available: true,
                local: false,
                remote: true,
                streaming: true,
                cancellation: true,
                structured_output: true,
                batching: true,
                endpoint: Some("mock://remote".to_owned()),
                notes: vec!["test remote provider".to_owned()],
            }
        }

        async fn infer(
            &self,
            request: InferenceRequest,
            cancellation: Option<CancellationToken>,
        ) -> RuntimeResult<InferenceResult> {
            if model_id(&request.model) == "blocking-remote" {
                cancellation
                    .expect("runtime supplies cancellation")
                    .cancelled()
                    .await;
                return Err(RuntimeError::Cancelled);
            }
            if cancellation
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
            {
                return Err(RuntimeError::Cancelled);
            }
            Ok(InferenceResult {
                output: Output::Structured(json!({"remote": model_id(&request.model)})),
                metadata: ExecutionMetadata {
                    backend: "remote-api".to_owned(),
                    hardware: Some("network".to_owned()),
                    ..ExecutionMetadata::default()
                },
            })
        }

        async fn infer_stream(
            &self,
            request: InferenceRequest,
            _cancellation: Option<CancellationToken>,
        ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>> {
            let stream = iter(vec![Ok(InferenceChunk {
                output: Output::Text(model_id(&request.model)),
                done: true,
                metadata: None,
            })]);
            Ok(Box::pin(stream))
        }
    }

    #[tokio::test]
    async fn loads_and_runs_local_cpu_models() {
        let runtime = Runtime::builder()
            .register_backend(CpuBackend::default())
            .build();
        let model = ModelSpec::new("echo", ModelFormat::Unknown, ModelLocation::Memory);
        runtime.load(model.clone()).await.unwrap();

        let result = runtime
            .infer(InferenceRequest {
                model: model.into(),
                input: Input::Text("hello runtime".to_owned()),
                options: InferenceOptions::default(),
            })
            .await
            .unwrap();

        assert_eq!(result.output, Output::Text("hello runtime".to_owned()));
        assert_eq!(result.metadata.provider, "local");
        assert_eq!(result.metadata.backend, "cpu");
    }

    #[tokio::test]
    async fn supports_remote_provider_selection() {
        let runtime = Runtime::builder()
            .register_backend(CpuBackend::default())
            .register_provider(MockRemoteProvider)
            .provider("mock-remote")
            .build();

        let result = runtime
            .infer(InferenceRequest {
                model: ModelReference::id("remote-model", None),
                input: Input::Text("hello".to_owned()),
                options: InferenceOptions {
                    require_remote: true,
                    ..InferenceOptions::default()
                },
            })
            .await
            .unwrap();

        assert_eq!(result.metadata.provider, "mock-remote");
        assert_eq!(result.metadata.backend, "remote-api");
    }

    #[tokio::test]
    async fn only_falls_back_to_remote_when_enabled() {
        let request = InferenceRequest {
            model: ModelReference::id("missing", None),
            input: Input::Text("hello".to_owned()),
            options: InferenceOptions::default(),
        };

        let local_only = Runtime::builder()
            .register_backend(CpuBackend::default())
            .register_provider(MockRemoteProvider)
            .build();
        assert!(matches!(
            local_only.infer(request.clone()).await,
            Err(RuntimeError::ModelNotFound { .. })
        ));

        let hybrid = Runtime::builder()
            .register_backend(CpuBackend::default())
            .register_provider(MockRemoteProvider)
            .allow_remote_fallback(true)
            .build();
        let result = hybrid.infer(request).await.unwrap();
        assert_eq!(result.metadata.provider, "mock-remote");
        assert!(result.metadata.fallback);
        assert_eq!(result.metadata.fallback_from.as_deref(), Some("local"));
    }

    #[tokio::test]
    async fn explicit_policies_choose_deterministic_targets() {
        let local = Runtime::builder()
            .register_backend(CpuBackend::default())
            .register_provider(MockRemoteProvider)
            .build();
        let model = ModelSpec::new("echo", ModelFormat::Unknown, ModelLocation::Memory);
        local.load(model.clone()).await.unwrap();

        let local_result = local
            .infer(InferenceRequest {
                model: model.clone().into(),
                input: Input::Text("hello".to_owned()),
                options: InferenceOptions {
                    execution: ExecutionPolicy::PreferLocal,
                    ..InferenceOptions::default()
                },
            })
            .await
            .unwrap();
        assert_eq!(local_result.metadata.provider, "local");

        let remote_result = local
            .infer(InferenceRequest {
                model: ModelReference::id("remote-model", None),
                input: Input::Text("hello".to_owned()),
                options: InferenceOptions {
                    execution: ExecutionPolicy::PreferRemote,
                    ..InferenceOptions::default()
                },
            })
            .await
            .unwrap();
        assert_eq!(remote_result.metadata.provider, "mock-remote");
        assert_eq!(
            remote_result.metadata.routing_policy,
            ExecutionPolicy::PreferRemote
        );
    }

    #[tokio::test]
    async fn rejects_requests_above_batch_limit() {
        let runtime = Runtime::builder()
            .register_backend(CpuBackend::default())
            .max_batch_size(2)
            .build();

        let err = runtime
            .infer(InferenceRequest {
                model: ModelSpec::new("echo", ModelFormat::Unknown, ModelLocation::Memory).into(),
                input: Input::Tokens(vec![1, 2, 3]),
                options: InferenceOptions {
                    batch_size: Some(3),
                    ..InferenceOptions::default()
                },
            })
            .await
            .unwrap_err();

        assert!(matches!(err, RuntimeError::ResourceLimit { .. }));
    }

    #[tokio::test]
    async fn propagates_cancellation() {
        let runtime = Runtime::builder()
            .register_backend(CpuBackend::default())
            .build();
        let token = CancellationToken::new();
        token.cancel();

        let err = runtime
            .infer_with_cancellation(
                InferenceRequest {
                    model: ModelSpec::new("echo", ModelFormat::Unknown, ModelLocation::Memory)
                        .into(),
                    input: Input::Text("hello".to_owned()),
                    options: InferenceOptions::default(),
                },
                token,
            )
            .await
            .unwrap_err();

        assert!(matches!(err, RuntimeError::Cancelled));
    }

    #[tokio::test]
    async fn cancels_active_and_queued_inference() {
        let started = Arc::new(Semaphore::new(0));
        let runtime = Arc::new(
            Runtime::builder()
                .register_backend(BlockingBackend {
                    started: Arc::clone(&started),
                })
                .backend("blocking")
                .concurrency(1)
                .build(),
        );
        let make_request = || {
            InferenceRequest::new(
                ModelSpec::new("blocking", ModelFormat::Unknown, ModelLocation::Memory),
                "input",
            )
        };

        let active_token = CancellationToken::new();
        let active = {
            let runtime = Arc::clone(&runtime);
            let token = active_token.clone();
            tokio::spawn(
                async move { runtime.infer_with_cancellation(make_request(), token).await },
            )
        };
        let permit = started.acquire().await.unwrap();
        permit.forget();

        let queued_token = CancellationToken::new();
        let queued = {
            let runtime = Arc::clone(&runtime);
            let token = queued_token.clone();
            tokio::spawn(
                async move { runtime.infer_with_cancellation(make_request(), token).await },
            )
        };
        tokio::task::yield_now().await;
        queued_token.cancel();
        assert!(matches!(
            queued.await.unwrap(),
            Err(RuntimeError::Cancelled)
        ));

        active_token.cancel();
        assert!(matches!(
            active.await.unwrap(),
            Err(RuntimeError::Cancelled)
        ));
    }

    #[tokio::test]
    async fn cancels_an_active_stream() {
        let runtime = Runtime::builder()
            .register_backend(BlockingBackend {
                started: Arc::new(Semaphore::new(0)),
            })
            .backend("blocking")
            .build();
        let token = CancellationToken::new();
        let mut stream = runtime
            .infer_stream_with_cancellation(
                InferenceRequest::new(
                    ModelSpec::new("blocking", ModelFormat::Unknown, ModelLocation::Memory),
                    "input",
                )
                .with_options(InferenceOptions {
                    stream: true,
                    ..InferenceOptions::default()
                }),
                token.clone(),
            )
            .await
            .unwrap();
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            InferenceStreamEvent::Started(_)
        ));
        token.cancel();
        assert!(matches!(
            stream.next().await.unwrap(),
            Err(RuntimeError::Cancelled)
        ));
    }

    #[tokio::test]
    async fn propagates_remote_cancellation() {
        let runtime = Arc::new(
            Runtime::builder()
                .register_provider(MockRemoteProvider)
                .provider("mock-remote")
                .build(),
        );
        let token = CancellationToken::new();
        let task = {
            let runtime = Arc::clone(&runtime);
            let token = token.clone();
            tokio::spawn(async move {
                runtime
                    .infer_with_cancellation(
                        InferenceRequest::new(ModelReference::latest("blocking-remote"), "input")
                            .with_options(InferenceOptions {
                                execution: ExecutionPolicy::RemoteOnly,
                                ..InferenceOptions::default()
                            }),
                        token,
                    )
                    .await
            })
        };
        tokio::task::yield_now().await;
        token.cancel();
        assert!(matches!(task.await.unwrap(), Err(RuntimeError::Cancelled)));
    }

    #[tokio::test]
    async fn streams_from_local_backend() {
        let runtime = Runtime::builder()
            .register_backend(CpuBackend::default())
            .build();

        let mut stream = runtime
            .infer_stream(InferenceRequest {
                model: ModelSpec::new("echo", ModelFormat::Unknown, ModelLocation::Memory).into(),
                input: Input::Text("hello runtime".to_owned()),
                options: InferenceOptions {
                    stream: true,
                    ..InferenceOptions::default()
                },
            })
            .await
            .unwrap();

        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            InferenceStreamEvent::Started(_)
        ));
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            InferenceStreamEvent::Output(Output::Text(text)) if text == "hello"
        ));
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            InferenceStreamEvent::Output(Output::Text(text)) if text == "runtime"
        ));
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            InferenceStreamEvent::Completed(metadata)
                if metadata.output_event_count == 2 && metadata.streaming_supported
        ));
    }

    #[tokio::test]
    async fn rejects_stream_calls_without_stream_flag() {
        let runtime = Runtime::builder()
            .register_backend(CpuBackend::default())
            .build();
        let result = runtime
            .infer_stream(InferenceRequest {
                model: ModelSpec::new("echo", ModelFormat::Unknown, ModelLocation::Memory).into(),
                input: Input::Text("hello".to_owned()),
                options: InferenceOptions::default(),
            })
            .await;

        assert!(matches!(
            result,
            Err(RuntimeError::CapabilityMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn streams_from_remote_provider() {
        let runtime = Runtime::builder()
            .register_backend(CpuBackend::default())
            .register_provider(MockRemoteProvider)
            .provider("mock-remote")
            .build();

        let mut stream = runtime
            .infer_stream(InferenceRequest {
                model: ModelReference::id("remote-model", None),
                input: Input::Text("hello".to_owned()),
                options: InferenceOptions {
                    stream: true,
                    require_remote: true,
                    ..InferenceOptions::default()
                },
            })
            .await
            .unwrap();

        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            InferenceStreamEvent::Started(_)
        ));
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            InferenceStreamEvent::Output(Output::Text(text)) if text == "remote-model"
        ));
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            InferenceStreamEvent::Completed(metadata)
                if metadata.output_event_count == 1
        ));
    }

    #[tokio::test]
    async fn explicit_handles_reuse_loaded_models_and_report_cache_hits() {
        let runtime = Runtime::builder()
            .register_backend(CpuBackend::default())
            .build();
        let model = ModelSpec::new("echo", ModelFormat::Unknown, ModelLocation::Memory);
        let handle = runtime.load(model).await.unwrap();
        assert_eq!(handle.id(), "echo");
        let result = runtime
            .infer_loaded(
                &handle,
                Input::Text("cached".to_owned()),
                Default::default(),
            )
            .await
            .unwrap();
        assert!(result.metadata.cache_hit);
        assert_eq!(runtime.models().list().len(), 1);
    }

    #[tokio::test]
    async fn loaded_stream_reuses_the_cached_model() {
        let runtime = Runtime::builder()
            .register_backend(CpuBackend::default())
            .build();
        let handle = runtime
            .load(ModelSpec::new(
                "echo",
                ModelFormat::Unknown,
                ModelLocation::Memory,
            ))
            .await
            .unwrap();
        let mut stream = runtime
            .infer_loaded_stream(
                &handle,
                Input::Text("cached stream".to_owned()),
                InferenceOptions::default(),
            )
            .await
            .unwrap();
        let events = futures_util::StreamExt::collect::<Vec<_>>(&mut stream).await;
        assert!(matches!(
            events.first().and_then(|event| event.as_ref().ok()),
            Some(InferenceStreamEvent::Started(_))
        ));
        assert!(matches!(
            events.last().and_then(|event| event.as_ref().ok()),
            Some(InferenceStreamEvent::Completed(metadata)) if metadata.cache_hit
        ));
        assert_eq!(runtime.models().list().len(), 1);
    }

    #[tokio::test]
    async fn cache_evicts_the_least_recently_used_target() {
        let runtime = Runtime::builder()
            .register_backend(CpuBackend::default())
            .max_models(1)
            .build();
        runtime
            .load(ModelSpec::new(
                "first",
                ModelFormat::Unknown,
                ModelLocation::Memory,
            ))
            .await
            .unwrap();
        runtime
            .load(ModelSpec::new(
                "second",
                ModelFormat::Unknown,
                ModelLocation::Memory,
            ))
            .await
            .unwrap();
        assert_eq!(runtime.models().list().len(), 1);
        assert_eq!(runtime.models().list()[0].id, "second");
    }

    #[tokio::test]
    async fn batch_inference_preserves_result_order_and_identity() {
        let runtime = Runtime::builder()
            .register_backend(CpuBackend::default())
            .build();
        let model = ModelSpec::new("echo", ModelFormat::Unknown, ModelLocation::Memory);
        let results = runtime
            .infer_batch(vec![
                InferenceRequest {
                    model: model.clone().into(),
                    input: Input::Text("a".to_owned()),
                    options: Default::default(),
                },
                InferenceRequest {
                    model: model.into(),
                    input: Input::Text("b".to_owned()),
                    options: Default::default(),
                },
            ])
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].output, Output::Text("a".to_owned()));
        assert_eq!(results[1].output, Output::Text("b".to_owned()));
        assert_eq!(results[0].metadata.batch_size, 2);
        assert_eq!(results[1].metadata.batch_size, 2);
    }

    #[tokio::test]
    async fn records_execution_metrics_and_snapshot() {
        let runtime = Runtime::builder()
            .register_backend(CpuBackend::default())
            .build();
        let model = ModelSpec::new("observed", ModelFormat::Unknown, ModelLocation::Memory);
        let result = runtime
            .infer(InferenceRequest {
                model: model.into(),
                input: Input::Text("hello".to_owned()),
                options: Default::default(),
            })
            .await
            .unwrap();

        let metrics = runtime.metrics();
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.successful_requests, 1);
        assert_eq!(runtime.executions().len(), 1);
        assert_eq!(
            runtime.executions()[0].completion,
            ExecutionCompletion::Succeeded
        );
        assert_eq!(
            runtime.executions()[0].request_id,
            result.metadata.request_id.clone().unwrap()
        );
        assert_eq!(runtime.snapshot().loaded_models, 1);
    }
}
