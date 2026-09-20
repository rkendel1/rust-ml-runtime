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
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
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
pub struct RuntimeStatus {
    pub loaded_models: usize,
    pub max_models: usize,
    pub max_concurrent_inferences: usize,
    pub max_concurrent_model_loads: usize,
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
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
            inference_semaphore: Arc::new(Semaphore::new(
                self.config.max_concurrent_inferences.max(1),
            )),
            load_semaphore: Arc::new(Semaphore::new(
                self.config.max_concurrent_model_loads.max(1),
            )),
            cache_clock: AtomicU64::new(0),
        }
    }
}

pub struct Runtime {
    config: RuntimeConfig,
    backends: BTreeMap<String, Arc<dyn Backend>>,
    providers: BTreeMap<String, Arc<dyn Provider>>,
    loaded_models: RwLock<HashMap<ModelCacheKey, LoadedModel>>,
    inference_semaphore: Arc<Semaphore>,
    load_semaphore: Arc<Semaphore>,
    cache_clock: AtomicU64,
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
            return Ok(runtime_handle(existing));
        }
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
            return Ok(runtime_handle(existing));
        }
        let backend = self
            .backends
            .get(&backend_name)
            .ok_or_else(|| RuntimeError::backend_unavailable(&backend_name, "not registered"))?
            .clone();
        let timeout_ms = self.config.timeout_ms;
        let handle = run_with_timeout(timeout_ms, "load", backend.load(&model)).await?;

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
            }
        }
        let loaded = LoadedModel {
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

    pub async fn infer(&self, request: InferenceRequest) -> RuntimeResult<InferenceResult> {
        self.infer_with_cancellation(request, CancellationToken::new())
            .await
    }

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

    pub async fn infer_batch(
        &self,
        requests: Vec<InferenceRequest>,
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
        let _permit = self.inference_semaphore.acquire().await.map_err(|_| {
            RuntimeError::execution("infer_batch", "runtime inference semaphore closed")
        })?;
        let batch_size = requests.len();
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
        Ok(results)
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

        let _permit = tokio::select! {
            permit = self.inference_semaphore.acquire() => permit
                .map_err(|_| RuntimeError::execution("infer", "runtime inference semaphore closed"))?,
            _ = cancellation.cancelled() => return Err(RuntimeError::Cancelled),
        };

        let mut selection = self.select_for_request(&request)?;
        let timeout_ms = request.options.timeout_ms.or(self.config.timeout_ms);
        let input_tokens = estimate_input_tokens(&request.input);

        match selection.provider.as_str() {
            "local" => {
                let cache_hit = self.is_loaded(&request.model);
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
                result.metadata.cache_hit = cache_hit;
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
            ModelReference::Id { id, version } => self.find_loaded(id, version.as_ref()),
            ModelReference::Spec(spec) => {
                if let Some(existing) = self.find_loaded(&spec.id, spec.version.as_ref()).ok() {
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
                && version.map_or(true, |version| {
                    loaded.spec.version.as_ref() == Some(version)
                })
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
                        .map_or(true, |v| key.version.as_ref() == Some(v))
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
    metadata.execution_ms = metadata.latency_ms;
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
}
