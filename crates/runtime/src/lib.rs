use ml_runtime_backend::{Backend, BackendCapability};
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::{
    ExecutionMetadata, ExecutionPolicy, InferenceChunk, InferenceOptions, InferenceRequest,
    InferenceResult, Input, Output,
};
use ml_runtime_model::{ModelFormat, ModelHandle, ModelLocation, ModelReference, ModelSpec};
use ml_runtime_provider::{Provider, ProviderCapability};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::{sync::Semaphore, time::timeout};

pub use ml_runtime_backend;
pub use ml_runtime_common;
pub use ml_runtime_inference;
pub use ml_runtime_model;
pub use ml_runtime_provider;

pub type InferenceStream = BoxStream<RuntimeResult<InferenceChunk>>;

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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeCapabilities {
    pub platforms: Vec<Platform>,
    pub providers: Vec<ProviderCapability>,
    pub backends: Vec<BackendCapability>,
    pub models: Vec<ModelCapability>,
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
}

pub struct RuntimeBuilder {
    config: RuntimeConfig,
    backends: BTreeMap<String, Arc<dyn Backend>>,
    providers: BTreeMap<String, Arc<dyn Provider>>,
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self {
            config: RuntimeConfig::default(),
            backends: BTreeMap::new(),
            providers: BTreeMap::new(),
        }
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
        self
    }

    pub fn max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.config.max_batch_size = Some(max_batch_size.max(1));
        self
    }

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

    pub fn build(self) -> Runtime {
        Runtime {
            config: self.config.clone(),
            backends: self.backends,
            providers: self.providers,
            loaded_models: RwLock::new(HashMap::new()),
            semaphore: Arc::new(Semaphore::new(self.config.max_concurrency.max(1))),
        }
    }
}

pub struct Runtime {
    config: RuntimeConfig,
    backends: BTreeMap<String, Arc<dyn Backend>>,
    providers: BTreeMap<String, Arc<dyn Provider>>,
    loaded_models: RwLock<HashMap<String, LoadedModel>>,
    semaphore: Arc<Semaphore>,
}

impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.config
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
            platforms: vec![Platform::current()],
            providers: self.providers().providers,
            backends: self.backends().backends,
            models: self.models().models,
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

    pub async fn load(&self, model: ModelSpec) -> RuntimeResult<ModelHandle> {
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

        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| RuntimeError::execution("load", "runtime semaphore closed"))?;
        let backend_name = self.select_backend(&model)?;
        let backend = self
            .backends
            .get(&backend_name)
            .ok_or_else(|| RuntimeError::backend_unavailable(&backend_name, "not registered"))?
            .clone();
        let timeout_ms = self.config.timeout_ms;
        let handle = run_with_timeout(timeout_ms, "load", backend.load(&model)).await?;

        self.loaded_models
            .write()
            .expect("loaded model registry poisoned")
            .insert(
                model.id.clone(),
                LoadedModel {
                    spec: model,
                    handle: handle.clone(),
                    backend_name,
                    provider_name: "local".to_owned(),
                },
            );
        Ok(handle)
    }

    pub async fn infer(&self, request: InferenceRequest) -> RuntimeResult<InferenceResult> {
        self.infer_with_cancellation(request, CancellationToken::new())
            .await
    }

    pub async fn infer_with_cancellation(
        &self,
        request: InferenceRequest,
        cancellation: CancellationToken,
    ) -> RuntimeResult<InferenceResult> {
        self.validate_request(&request)?;
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }

        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| RuntimeError::execution("infer", "runtime semaphore closed"))?;

        let mut selection = self.select_for_request(&request)?;
        let timeout_ms = request.options.timeout_ms.or(self.config.timeout_ms);
        let input_tokens = estimate_input_tokens(&request.input);

        match selection.provider.as_str() {
            "local" => {
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
                Ok(result)
            }
            _ => {
                self.infer_remote(request, cancellation, timeout_ms, input_tokens, selection)
                    .await
            }
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

    pub async fn infer_stream(&self, request: InferenceRequest) -> RuntimeResult<InferenceStream> {
        self.infer_stream_with_cancellation(request, CancellationToken::new())
            .await
    }

    pub async fn infer_stream_with_cancellation(
        &self,
        request: InferenceRequest,
        cancellation: CancellationToken,
    ) -> RuntimeResult<InferenceStream> {
        self.validate_request(&request)?;
        if !request.options.stream {
            return Err(RuntimeError::CapabilityMismatch {
                reason: "infer_stream requires InferenceOptions::stream to be true".to_owned(),
            });
        }
        let selection = self.select_for_request(&request)?;
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
                backend
                    .infer_stream(&loaded.handle, &request, Some(cancellation))
                    .await
            }
            provider_name => {
                let provider = self
                    .providers
                    .get(provider_name)
                    .ok_or_else(|| {
                        RuntimeError::provider_unavailable(provider_name, "not registered")
                    })?
                    .clone();
                provider.infer_stream(request, Some(cancellation)).await
            }
        }
    }

    fn validate_request(&self, request: &InferenceRequest) -> RuntimeResult<()> {
        if request.options.require_local && request.options.require_remote {
            return Err(RuntimeError::CapabilityMismatch {
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

    async fn resolve_loaded_model(&self, reference: &ModelReference) -> RuntimeResult<LoadedModel> {
        match reference {
            ModelReference::Id { id, version } => self
                .loaded_models
                .read()
                .expect("loaded model registry poisoned")
                .get(id)
                .cloned()
                .filter(|loaded| {
                    version
                        .as_ref()
                        .is_none_or(|version| loaded.spec.version.as_ref() == Some(version))
                })
                .ok_or_else(|| RuntimeError::ModelNotFound { model: id.clone() }),
            ModelReference::Spec(spec) => {
                if let Some(existing) = self
                    .loaded_models
                    .read()
                    .expect("loaded model registry poisoned")
                    .get(&spec.id)
                    .cloned()
                {
                    return Ok(existing);
                }
                self.load(spec.clone()).await?;
                self.loaded_models
                    .read()
                    .expect("loaded model registry poisoned")
                    .get(&spec.id)
                    .cloned()
                    .ok_or_else(|| RuntimeError::ModelNotFound {
                        model: spec.id.clone(),
                    })
            }
        }
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
                let loaded = self
                    .loaded_models
                    .read()
                    .expect("loaded model registry poisoned")
                    .get(id)
                    .cloned()
                    .filter(|loaded| {
                        version
                            .as_ref()
                            .is_none_or(|version| loaded.spec.version.as_ref() == Some(version))
                    })
                    .ok_or_else(|| RuntimeError::ModelNotFound { model: id.clone() })?;
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
            policy: options.execution.clone(),
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
                    | RuntimeError::BackendUnavailable { .. }
                    | RuntimeError::ModelNotFound { .. }
                    | RuntimeError::CapabilityMismatch { .. }
                    | RuntimeError::Timeout { .. }
                    | RuntimeError::Transport { .. }
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

fn model_details(reference: &ModelReference) -> (String, Option<String>) {
    match reference {
        ModelReference::Id { id, version } => (id.clone(), version.clone()),
        ModelReference::Spec(spec) => (spec.id.clone(), spec.version.clone()),
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
    metadata.input_tokens = input_tokens;
    metadata.output_tokens = estimate_output_tokens(output);
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use ml_runtime_common::BoxStream;
    use ml_runtime_cpu_backend::CpuBackend;
    use ml_runtime_inference::InferenceOptions;
    use ml_runtime_provider::{Provider, ProviderCapabilities};
    use serde_json::json;
    use tokio_stream::iter;

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

        let first = stream.next().await.unwrap().unwrap();
        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(first.output, Output::Text("hello".to_owned()));
        assert_eq!(second.output, Output::Text("runtime".to_owned()));
        assert!(second.done);
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

        let chunk = stream.next().await.unwrap().unwrap();
        assert_eq!(chunk.output, Output::Text("remote-model".to_owned()));
        assert!(chunk.done);
    }
}
