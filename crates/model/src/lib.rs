use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    any::Any,
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio_util::sync::CancellationToken;

pub type ModelMetadata = BTreeMap<String, String>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ModelId {
    pub name: String,
    pub version: String,
}

impl ModelId {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Result<Self, CatalogError> {
        let id = Self {
            name: name.into(),
            version: version.into(),
        };
        id.validate()?;
        Ok(id)
    }

    pub fn parse(reference: &str) -> Result<Self, CatalogError> {
        let (name, version) = reference
            .rsplit_once('@')
            .ok_or_else(|| CatalogError::InvalidIdentity(reference.to_owned()))?;
        Self::new(name, version)
    }

    pub fn validate(&self) -> Result<(), CatalogError> {
        if self.name.is_empty()
            || self.version.is_empty()
            || self.name.contains(['/', '\\', '@'])
            || self.version.contains(['/', '\\', '@'])
            || self.name.contains("..")
            || self.version.contains("..")
        {
            return Err(CatalogError::InvalidIdentity(format!(
                "{}@{}",
                self.name, self.version
            )));
        }
        Ok(())
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.name, self.version)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    #[serde(rename = "onnx", alias = "Onnx")]
    Onnx,
    #[serde(rename = "coreml", alias = "CoreMl")]
    CoreMl,
    #[serde(rename = "gguf", alias = "Gguf")]
    Gguf,
    #[serde(rename = "safetensors", alias = "Safetensors")]
    Safetensors,
    #[serde(rename = "tensorrt", alias = "TensorRt")]
    TensorRt,
    #[serde(rename = "unknown", alias = "Unknown")]
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelLocation {
    Path(String),
    Uri(String),
    Embedded(String),
    Memory,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelSpec {
    pub id: String,
    pub version: Option<String>,
    pub format: ModelFormat,
    pub location: ModelLocation,
    pub metadata: ModelMetadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelManifest {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(alias = "name")]
    pub id: String,
    #[serde(default)]
    pub model_version: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    pub format: ModelFormat,
    #[serde(alias = "artifact_path")]
    pub artifact: ArtifactSpec,
    #[serde(default)]
    pub metadata: ModelMetadata,
    #[serde(default)]
    pub inputs: Vec<ModelTensorSpec>,
    #[serde(default)]
    pub outputs: Vec<ModelTensorSpec>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ArtifactSpec {
    pub path: String,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub sha256: Option<String>,
}

impl<'de> Deserialize<'de> for ArtifactSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Input {
            Path(String),
            Metadata {
                path: String,
                #[serde(default)]
                size_bytes: Option<u64>,
                #[serde(default)]
                sha256: Option<String>,
            },
        }
        match Input::deserialize(deserializer)? {
            Input::Path(path) => Ok(Self {
                path,
                size_bytes: None,
                sha256: None,
            }),
            Input::Metadata {
                path,
                size_bytes,
                sha256,
            } => Ok(Self {
                path,
                size_bytes,
                sha256,
            }),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelTensorSpec {
    pub name: String,
    #[serde(default)]
    pub data_type: Option<String>,
    #[serde(default)]
    pub shape: Option<Vec<i64>>,
}

fn default_schema_version() -> u32 {
    1
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPackage {
    pub manifest: ModelManifest,
    pub root: PathBuf,
    pub artifact: PathBuf,
}

impl ModelPackage {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        let manifest_path = root.join("manifest.json");
        let manifest: ModelManifest = serde_json::from_slice(
            &fs::read(&manifest_path)
                .map_err(|error| format!("read {}: {error}", manifest_path.display()))?,
        )
        .map_err(|error| format!("parse {}: {error}", manifest_path.display()))?;
        if manifest.schema_version != 1 {
            return Err(format!(
                "unsupported model manifest schema version {}",
                manifest.schema_version
            ));
        }
        let artifact_name = manifest.artifact.path.as_str();
        if artifact_name.is_empty() {
            return Err("model artifact path is empty".to_owned());
        }
        let artifact_path = Path::new(artifact_name);
        if artifact_path.is_absolute()
            || artifact_path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err("model artifact path must stay inside the model package".to_owned());
        }
        let artifact = root.join(artifact_name);
        let canonical_root = root
            .canonicalize()
            .map_err(|error| format!("resolve model package: {error}"))?;
        let canonical_artifact = artifact
            .canonicalize()
            .map_err(|error| format!("resolve model artifact: {error}"))?;
        if canonical_artifact.strip_prefix(&canonical_root).is_err() {
            return Err("model artifact must be inside the model package".to_owned());
        }

        if !artifact.is_file() && !artifact.is_dir() {
            return Err(format!(
                "model artifact does not exist: {}",
                artifact.display()
            ));
        }
        Self::reject_symlinks(&artifact)
            .map_err(|reason| format!("model artifact is unsafe: {reason}"))?;
        Ok(Self {
            manifest,
            root,
            artifact,
        })
    }

    fn reject_symlinks(path: &Path) -> Result<(), String> {
        let metadata = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
        if metadata.file_type().is_symlink() {
            return Err("symbolic links are not allowed".to_owned());
        }
        if metadata.is_dir() {
            for entry in fs::read_dir(path).map_err(|error| error.to_string())? {
                Self::reject_symlinks(&entry.map_err(|error| error.to_string())?.path())?;
            }
        }
        Ok(())
    }

    pub fn spec(&self) -> ModelSpec {
        ModelSpec {
            id: self.manifest.id.clone(),
            version: self
                .manifest
                .model_version
                .clone()
                .or_else(|| self.manifest.version.clone()),
            format: self.manifest.format.clone(),
            location: ModelLocation::Path(self.artifact.display().to_string()),
            metadata: self.manifest.metadata.clone(),
        }
    }

    pub fn validate_artifact(&self) -> Result<(), CatalogError> {
        if let Some(expected) = self.manifest.artifact.size_bytes {
            let actual = artifact_size(&self.artifact)?;
            if actual != expected {
                return Err(CatalogError::ArtifactIntegrity {
                    model: self.manifest.id.clone(),
                    reason: format!("size mismatch: expected {expected}, got {actual}"),
                });
            }
        }
        if let Some(expected) = self.manifest.artifact.sha256.as_deref() {
            let actual = sha256(&self.artifact)?;
            if !expected.eq_ignore_ascii_case(&actual) {
                return Err(CatalogError::ArtifactIntegrity {
                    model: self.manifest.id.clone(),
                    reason: format!("sha256 mismatch: expected {expected}, got {actual}"),
                });
            }
        }
        Ok(())
    }
}

impl ModelSpec {
    pub fn new(id: impl Into<String>, format: ModelFormat, location: ModelLocation) -> Self {
        Self {
            id: id.into(),
            version: None,
            format,
            location,
            metadata: ModelMetadata::new(),
        }
    }
}

fn artifact_size(path: &Path) -> Result<u64, CatalogError> {
    if path.is_file() {
        return Ok(fs::metadata(path)
            .map_err(|error| CatalogError::Io(error.to_string()))?
            .len());
    }
    let mut total = 0;
    for entry in fs::read_dir(path).map_err(|error| CatalogError::Io(error.to_string()))? {
        total += artifact_size(
            &entry
                .map_err(|error| CatalogError::Io(error.to_string()))?
                .path(),
        )?;
    }
    Ok(total)
}

fn sha256(path: &Path) -> Result<String, CatalogError> {
    let mut digest = Sha256::new();
    if path.is_file() {
        digest.update(fs::read(path).map_err(|error| CatalogError::Io(error.to_string()))?);
    } else {
        let mut entries = fs::read_dir(path)
            .map_err(|error| CatalogError::Io(error.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| CatalogError::Io(error.to_string()))?;
        entries.sort_by_key(|entry| entry.path());
        for entry in entries {
            digest.update(sha256(&entry.path())?.as_bytes());
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelDescriptor {
    pub id: ModelId,
    pub format: ModelFormat,
    pub package_path: PathBuf,
    pub manifest: ModelManifest,
}

impl ModelDescriptor {
    pub fn spec(&self) -> ModelSpec {
        ModelPackage {
            manifest: self.manifest.clone(),
            root: self.package_path.clone(),
            artifact: self.package_path.join(&self.manifest.artifact.path),
        }
        .spec()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CatalogError {
    Io(String),
    InvalidIdentity(String),
    InvalidPackage { path: PathBuf, reason: String },
    ModelNotFound(String),
    AmbiguousModel(String),
    DuplicateIdentity(ModelId),
    ArtifactIntegrity { model: String, reason: String },
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(reason) => write!(f, "catalog unavailable: {reason}"),
            Self::InvalidIdentity(identity) => write!(f, "invalid model identity: {identity}"),
            Self::InvalidPackage { path, reason } => {
                write!(f, "invalid model package {}: {reason}", path.display())
            }
            Self::ModelNotFound(model) => write!(f, "model not found: {model}"),
            Self::AmbiguousModel(model) => write!(f, "ambiguous model: {model}"),
            Self::DuplicateIdentity(id) => write!(f, "duplicate model identity: {id}"),
            Self::ArtifactIntegrity { model, reason } => {
                write!(f, "artifact integrity failure for {model}: {reason}")
            }
        }
    }
}

impl std::error::Error for CatalogError {}

#[async_trait]
pub trait ModelCatalog: Send + Sync {
    async fn list(&self) -> Result<Vec<ModelDescriptor>, CatalogError>;
    async fn resolve(&self, reference: &ModelReference) -> Result<ModelDescriptor, CatalogError>;
    async fn validate(&self, id: &ModelId) -> Result<(), CatalogError>;
    async fn refresh(&self) -> Result<(), CatalogError>;
}

#[derive(Clone, Debug)]
pub struct FilesystemModelCatalog {
    roots: Vec<PathBuf>,
}

impl FilesystemModelCatalog {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            roots: vec![root.into()],
        }
    }

    pub fn from_roots(roots: impl IntoIterator<Item = PathBuf>) -> Self {
        Self {
            roots: roots.into_iter().collect(),
        }
    }

    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    fn discover(&self) -> Result<Vec<ModelDescriptor>, CatalogError> {
        let mut packages = Vec::new();
        for root in &self.roots {
            discover_dirs(root, &mut packages)?;
        }
        packages.sort_by_key(|descriptor| {
            (
                descriptor.id.name.clone(),
                descriptor.id.version.clone(),
                format!("{:?}", descriptor.format),
                descriptor.package_path.clone(),
            )
        });
        for pair in packages.windows(2) {
            if pair[0].id == pair[1].id {
                return Err(CatalogError::DuplicateIdentity(pair[0].id.clone()));
            }
        }
        Ok(packages)
    }
}

fn discover_dirs(path: &Path, packages: &mut Vec<ModelDescriptor>) -> Result<(), CatalogError> {
    if !path.is_dir() {
        return Err(CatalogError::Io(format!(
            "root is not a directory: {}",
            path.display()
        )));
    }
    if path.join("manifest.json").is_file() {
        let package = ModelPackage::open(path).map_err(|reason| CatalogError::InvalidPackage {
            path: path.to_path_buf(),
            reason,
        })?;
        let version = package
            .manifest
            .model_version
            .clone()
            .or_else(|| package.manifest.version.clone())
            .ok_or_else(|| CatalogError::InvalidPackage {
                path: path.to_path_buf(),
                reason: "model version is required".to_owned(),
            })?;
        let id = ModelId::new(package.manifest.id.clone(), version)?;
        packages.push(ModelDescriptor {
            id,
            format: package.manifest.format.clone(),
            package_path: path.to_path_buf(),
            manifest: package.manifest,
        });
        return Ok(());
    }
    let mut entries = fs::read_dir(path)
        .map_err(|error| CatalogError::Io(format!("read {}: {error}", path.display())))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| CatalogError::Io(error.to_string()))?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        if entry
            .file_type()
            .map_err(|error| CatalogError::Io(error.to_string()))?
            .is_dir()
        {
            discover_dirs(&entry.path(), packages)?;
        }
    }
    Ok(())
}

#[async_trait]
impl ModelCatalog for FilesystemModelCatalog {
    async fn list(&self) -> Result<Vec<ModelDescriptor>, CatalogError> {
        self.discover()
    }

    async fn resolve(&self, reference: &ModelReference) -> Result<ModelDescriptor, CatalogError> {
        let (name, version) = match reference {
            ModelReference::Id { id, version } => (id.as_str(), version.as_deref()),
            ModelReference::Spec(spec) => {
                return Err(CatalogError::InvalidIdentity(spec.id.clone()))
            }
        };
        if name.is_empty() || name.contains(['/', '\\', '@']) || name.contains("..") {
            return Err(CatalogError::InvalidIdentity(name.to_owned()));
        }
        let matches = self
            .discover()?
            .into_iter()
            .filter(|descriptor| {
                descriptor.id.name == name
                    && version.map_or(true, |version| descriptor.id.version == version)
            })
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => Err(CatalogError::ModelNotFound(match version {
                Some(version) => format!("{name}@{version}"),
                None => name.to_owned(),
            })),
            [descriptor] => Ok(descriptor.clone()),
            _ => Err(CatalogError::AmbiguousModel(name.to_owned())),
        }
    }

    async fn validate(&self, id: &ModelId) -> Result<(), CatalogError> {
        let descriptor = self
            .discover()?
            .into_iter()
            .find(|descriptor| descriptor.id == *id)
            .ok_or_else(|| CatalogError::ModelNotFound(id.to_string()))?;
        ModelPackage::open(&descriptor.package_path)
            .map_err(|reason| CatalogError::InvalidPackage {
                path: descriptor.package_path.clone(),
                reason,
            })?
            .validate_artifact()
    }

    async fn refresh(&self) -> Result<(), CatalogError> {
        self.discover().map(|_| ())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelReference {
    Id { id: String, version: Option<String> },
    Spec(ModelSpec),
}

impl ModelReference {
    pub fn id(id: impl Into<String>, version: Option<String>) -> Self {
        Self::Id {
            id: id.into(),
            version,
        }
    }
}

impl From<ModelSpec> for ModelReference {
    fn from(value: ModelSpec) -> Self {
        Self::Spec(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFetchRequest {
    pub model: ModelId,
    pub source: ModelSourceReference,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelSourceReference {
    Path(PathBuf),
    Uri(String),
}

#[derive(Clone, Debug)]
pub struct AcquisitionConfig {
    pub max_download_bytes: u64,
    pub timeout_secs: u64,
}

impl Default for AcquisitionConfig {
    fn default() -> Self {
        Self {
            max_download_bytes: 4 * 1024 * 1024 * 1024,
            timeout_secs: 300,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelArtifact {
    pub package_path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstallMode {
    KeepExisting,
    Replace,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstallResult {
    Installed(ModelDescriptor),
    AlreadyInstalled(ModelDescriptor),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AcquisitionError {
    Io(String),
    InvalidRequest(String),
    Cancelled,
    LimitExceeded { limit: u64, actual: u64 },
    InvalidPackage { path: PathBuf, reason: String },
    IdentityMismatch { requested: ModelId, found: ModelId },
    AlreadyInstalled(ModelId),
    Catalog(CatalogError),
}

impl std::fmt::Display for AcquisitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(reason) => write!(f, "acquisition I/O error: {reason}"),
            Self::InvalidRequest(reason) => write!(f, "invalid acquisition request: {reason}"),
            Self::Cancelled => write!(f, "model acquisition cancelled"),
            Self::LimitExceeded { limit, actual } => {
                write!(f, "model acquisition exceeds {limit} bytes (got {actual})")
            }
            Self::InvalidPackage { path, reason } => {
                write!(f, "invalid model package {}: {reason}", path.display())
            }
            Self::IdentityMismatch { requested, found } => {
                write!(f, "requested model {requested}, package contains {found}")
            }
            Self::AlreadyInstalled(id) => write!(f, "model already installed: {id}"),
            Self::Catalog(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for AcquisitionError {}

#[async_trait]
pub trait ModelSource: Send + Sync {
    async fn fetch(
        &self,
        request: &ModelFetchRequest,
        destination: &Path,
        config: &AcquisitionConfig,
        cancellation: Option<&CancellationToken>,
    ) -> Result<ModelArtifact, AcquisitionError>;
}

#[derive(Clone, Debug, Default)]
pub struct FileModelSource;

impl FileModelSource {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ModelSource for FileModelSource {
    async fn fetch(
        &self,
        request: &ModelFetchRequest,
        destination: &Path,
        config: &AcquisitionConfig,
        cancellation: Option<&CancellationToken>,
    ) -> Result<ModelArtifact, AcquisitionError> {
        let source = match &request.source {
            ModelSourceReference::Path(path) => path,
            ModelSourceReference::Uri(_) => {
                return Err(AcquisitionError::InvalidRequest(
                    "file source requires a filesystem path".to_owned(),
                ))
            }
        };
        if !source.is_dir() {
            return Err(AcquisitionError::InvalidRequest(format!(
                "model source is not a package directory: {}",
                source.display()
            )));
        }
        copy_package(source, destination, config.max_download_bytes, cancellation)?;
        let package =
            ModelPackage::open(destination).map_err(|reason| AcquisitionError::InvalidPackage {
                path: destination.to_path_buf(),
                reason,
            })?;
        package
            .validate_artifact()
            .map_err(AcquisitionError::Catalog)?;
        let id = package_id(&package)?;
        if id != request.model {
            return Err(AcquisitionError::IdentityMismatch {
                requested: request.model.clone(),
                found: id,
            });
        }
        Ok(ModelArtifact {
            package_path: destination.to_path_buf(),
            bytes: artifact_size(&package.artifact).map_err(|error| match error {
                CatalogError::Io(reason) => AcquisitionError::Io(reason),
                other => AcquisitionError::Catalog(other),
            })?,
            sha256: sha256(&package.artifact).map_err(|error| match error {
                CatalogError::Io(reason) => AcquisitionError::Io(reason),
                other => AcquisitionError::Catalog(other),
            })?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct HttpModelSource {
    client: reqwest::Client,
    allow_insecure_http: bool,
}

impl HttpModelSource {
    pub fn new() -> Result<Self, AcquisitionError> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .map_err(|error| AcquisitionError::Io(error.to_string()))?;
        Ok(Self {
            client,
            allow_insecure_http: false,
        })
    }

    pub fn allow_insecure_http(mut self, allow: bool) -> Self {
        self.allow_insecure_http = allow;
        self
    }
}

#[async_trait]
impl ModelSource for HttpModelSource {
    async fn fetch(
        &self,
        request: &ModelFetchRequest,
        destination: &Path,
        config: &AcquisitionConfig,
        cancellation: Option<&CancellationToken>,
    ) -> Result<ModelArtifact, AcquisitionError> {
        let base = match &request.source {
            ModelSourceReference::Uri(uri) => uri.trim_end_matches('/').to_owned(),
            ModelSourceReference::Path(_) => {
                return Err(AcquisitionError::InvalidRequest(
                    "HTTP source requires a URI".to_owned(),
                ))
            }
        };
        let manifest_url = if base.ends_with("manifest.json") {
            base
        } else {
            format!("{base}/manifest.json")
        };
        validate_http_uri(&manifest_url, self.allow_insecure_http)?;
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(config.timeout_secs),
            self.client.get(&manifest_url).send(),
        )
        .await
        .map_err(|_| AcquisitionError::Io("manifest request timed out".to_owned()))?
        .map_err(|error| AcquisitionError::Io(error.to_string()))?
        .error_for_status()
        .map_err(|error| AcquisitionError::Io(error.to_string()))?;
        let manifest: ModelManifest =
            response
                .json()
                .await
                .map_err(|error| AcquisitionError::InvalidPackage {
                    path: destination.to_path_buf(),
                    reason: error.to_string(),
                })?;
        let version = manifest
            .model_version
            .clone()
            .or_else(|| manifest.version.clone())
            .ok_or_else(|| AcquisitionError::InvalidPackage {
                path: destination.to_path_buf(),
                reason: "model version is required".to_owned(),
            })?;
        let found =
            ModelId::new(manifest.id.clone(), version).map_err(AcquisitionError::Catalog)?;
        if found != request.model {
            return Err(AcquisitionError::IdentityMismatch {
                requested: request.model.clone(),
                found,
            });
        }
        let artifact_path = Path::new(&manifest.artifact.path);
        if artifact_path.is_absolute()
            || artifact_path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(AcquisitionError::InvalidPackage {
                path: destination.to_path_buf(),
                reason: "artifact path must stay inside the package".to_owned(),
            });
        }
        let artifact_url = format!(
            "{}/{}",
            manifest_url
                .trim_end_matches("manifest.json")
                .trim_end_matches('/'),
            manifest.artifact.path
        );
        validate_http_uri(&artifact_url, self.allow_insecure_http)?;
        let response = self
            .client
            .get(artifact_url)
            .send()
            .await
            .map_err(|error| AcquisitionError::Io(error.to_string()))?
            .error_for_status()
            .map_err(|error| AcquisitionError::Io(error.to_string()))?;
        if response
            .content_length()
            .is_some_and(|size| size > config.max_download_bytes)
        {
            return Err(AcquisitionError::LimitExceeded {
                limit: config.max_download_bytes,
                actual: response.content_length().unwrap_or_default(),
            });
        }
        fs::create_dir_all(
            destination.join(artifact_path.parent().unwrap_or_else(|| Path::new(""))),
        )
        .map_err(|error| AcquisitionError::Io(error.to_string()))?;
        fs::write(
            destination.join(artifact_path),
            download_body(response, config, cancellation).await?,
        )
        .map_err(|error| AcquisitionError::Io(error.to_string()))?;
        fs::write(
            destination.join("manifest.json"),
            serde_json::to_vec(&manifest).map_err(|error| AcquisitionError::InvalidPackage {
                path: destination.to_path_buf(),
                reason: error.to_string(),
            })?,
        )
        .map_err(|error| AcquisitionError::Io(error.to_string()))?;
        let package =
            ModelPackage::open(destination).map_err(|reason| AcquisitionError::InvalidPackage {
                path: destination.to_path_buf(),
                reason,
            })?;
        package
            .validate_artifact()
            .map_err(AcquisitionError::Catalog)?;
        Ok(ModelArtifact {
            package_path: destination.to_path_buf(),
            bytes: artifact_size(&package.artifact)
                .map_err(|error| AcquisitionError::Catalog(error))?,
            sha256: sha256(&package.artifact).map_err(|error| AcquisitionError::Catalog(error))?,
        })
    }
}

fn validate_http_uri(uri: &str, allow_insecure_http: bool) -> Result<(), AcquisitionError> {
    let parsed = reqwest::Url::parse(uri)
        .map_err(|error| AcquisitionError::InvalidRequest(error.to_string()))?;
    if parsed.scheme() != "https" && !(allow_insecure_http && parsed.scheme() == "http") {
        return Err(AcquisitionError::InvalidRequest(
            "model acquisition requires HTTPS".to_owned(),
        ));
    }
    Ok(())
}

async fn download_body(
    mut response: reqwest::Response,
    config: &AcquisitionConfig,
    cancellation: Option<&CancellationToken>,
) -> Result<Vec<u8>, AcquisitionError> {
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| AcquisitionError::Io(error.to_string()))?
    {
        if cancellation.is_some_and(CancellationToken::is_cancelled) {
            return Err(AcquisitionError::Cancelled);
        }
        if body.len() as u64 + chunk.len() as u64 > config.max_download_bytes {
            return Err(AcquisitionError::LimitExceeded {
                limit: config.max_download_bytes,
                actual: body.len() as u64 + chunk.len() as u64,
            });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[derive(Clone)]
pub struct ModelInstaller {
    root: PathBuf,
    catalog: Option<Arc<dyn ModelCatalog>>,
    config: AcquisitionConfig,
}

impl ModelInstaller {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            catalog: None,
            config: AcquisitionConfig::default(),
        }
    }

    pub fn with_catalog<C: ModelCatalog + 'static>(mut self, catalog: Arc<C>) -> Self {
        self.catalog = Some(catalog);
        self
    }

    pub fn with_config(mut self, config: AcquisitionConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn install<S: ModelSource>(
        &self,
        source: &S,
        request: &ModelFetchRequest,
        mode: InstallMode,
        cancellation: Option<&CancellationToken>,
    ) -> Result<InstallResult, AcquisitionError> {
        request
            .model
            .validate()
            .map_err(AcquisitionError::Catalog)?;
        if let Some(catalog) = &self.catalog {
            if let Ok(existing) = catalog
                .resolve(&ModelReference::id(
                    request.model.name.clone(),
                    Some(request.model.version.clone()),
                ))
                .await
            {
                catalog
                    .validate(&existing.id)
                    .await
                    .map_err(AcquisitionError::Catalog)?;
                if mode == InstallMode::KeepExisting {
                    return Ok(InstallResult::AlreadyInstalled(existing));
                }
            }
        }
        if let Some(token) = cancellation {
            if token.is_cancelled() {
                return Err(AcquisitionError::Cancelled);
            }
        }
        fs::create_dir_all(&self.root).map_err(|error| AcquisitionError::Io(error.to_string()))?;
        let staging_root = self.root.join(".staging");
        fs::create_dir_all(&staging_root)
            .map_err(|error| AcquisitionError::Io(error.to_string()))?;
        let staging = staging_root.join(format!(
            "{}-{}-{}",
            request.model.name,
            request.model.version,
            std::process::id()
        ));
        if staging.exists() {
            fs::remove_dir_all(&staging)
                .map_err(|error| AcquisitionError::Io(error.to_string()))?;
        }
        fs::create_dir_all(&staging).map_err(|error| AcquisitionError::Io(error.to_string()))?;
        let result = async {
            source
                .fetch(request, &staging, &self.config, cancellation)
                .await?;
            if cancellation.is_some_and(CancellationToken::is_cancelled) {
                return Err(AcquisitionError::Cancelled);
            }
            let package = ModelPackage::open(&staging).map_err(|reason| {
                AcquisitionError::InvalidPackage {
                    path: staging.clone(),
                    reason,
                }
            })?;
            package
                .validate_artifact()
                .map_err(AcquisitionError::Catalog)?;
            let found = package_id(&package)?;
            if found != request.model {
                return Err(AcquisitionError::IdentityMismatch {
                    requested: request.model.clone(),
                    found,
                });
            }
            let final_path = self
                .root
                .join(&request.model.name)
                .join(&request.model.version);
            let parent = final_path.parent().expect("model version has a parent");
            fs::create_dir_all(parent).map_err(|error| AcquisitionError::Io(error.to_string()))?;
            if final_path.exists() {
                if mode == InstallMode::Replace {
                    let backup = parent.join(format!(".replace-{}", std::process::id()));
                    fs::rename(&final_path, &backup)
                        .map_err(|error| AcquisitionError::Io(error.to_string()))?;
                    if let Err(error) = fs::rename(&staging, &final_path) {
                        let _ = fs::rename(&backup, &final_path);
                        return Err(AcquisitionError::Io(error.to_string()));
                    }
                    let _ = fs::remove_dir_all(backup);
                } else {
                    return Err(AcquisitionError::AlreadyInstalled(request.model.clone()));
                }
            } else {
                fs::rename(&staging, &final_path)
                    .map_err(|error| AcquisitionError::Io(error.to_string()))?;
            }
            let catalog = self.catalog.as_ref().ok_or_else(|| {
                AcquisitionError::InvalidRequest(
                    "a catalog is required to finish installation".to_owned(),
                )
            })?;
            catalog.refresh().await.map_err(AcquisitionError::Catalog)?;
            let descriptor = catalog
                .resolve(&ModelReference::id(
                    request.model.name.clone(),
                    Some(request.model.version.clone()),
                ))
                .await
                .map_err(AcquisitionError::Catalog)?;
            Ok(InstallResult::Installed(descriptor))
        }
        .await;
        if staging.exists() {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    }
}

fn package_id(package: &ModelPackage) -> Result<ModelId, AcquisitionError> {
    let version = package
        .manifest
        .model_version
        .clone()
        .or_else(|| package.manifest.version.clone())
        .ok_or_else(|| AcquisitionError::InvalidPackage {
            path: package.root.clone(),
            reason: "model version is required".to_owned(),
        })?;
    ModelId::new(package.manifest.id.clone(), version).map_err(AcquisitionError::Catalog)
}

fn copy_package(
    source: &Path,
    destination: &Path,
    limit: u64,
    cancellation: Option<&CancellationToken>,
) -> Result<(), AcquisitionError> {
    let mut bytes = 0;
    copy_package_entry(source, destination, &mut bytes, limit, cancellation)?;
    Ok(())
}

fn copy_package_entry(
    source: &Path,
    destination: &Path,
    bytes: &mut u64,
    limit: u64,
    cancellation: Option<&CancellationToken>,
) -> Result<(), AcquisitionError> {
    if cancellation.is_some_and(CancellationToken::is_cancelled) {
        return Err(AcquisitionError::Cancelled);
    }
    let metadata =
        fs::symlink_metadata(source).map_err(|error| AcquisitionError::Io(error.to_string()))?;
    if metadata.file_type().is_symlink() {
        return Err(AcquisitionError::InvalidPackage {
            path: source.to_path_buf(),
            reason: "symbolic links are not allowed".to_owned(),
        });
    }
    if metadata.is_dir() {
        fs::create_dir_all(destination).map_err(|error| AcquisitionError::Io(error.to_string()))?;
        let mut entries = fs::read_dir(source)
            .map_err(|error| AcquisitionError::Io(error.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| AcquisitionError::Io(error.to_string()))?;
        entries.sort_by_key(|entry| entry.path());
        for entry in entries {
            copy_package_entry(
                &entry.path(),
                &destination.join(entry.file_name()),
                bytes,
                limit,
                cancellation,
            )?;
        }
    } else if metadata.is_file() {
        *bytes = bytes.saturating_add(metadata.len());
        if *bytes > limit {
            return Err(AcquisitionError::LimitExceeded {
                limit,
                actual: *bytes,
            });
        }
        fs::copy(source, destination).map_err(|error| AcquisitionError::Io(error.to_string()))?;
    } else {
        return Err(AcquisitionError::InvalidPackage {
            path: source.to_path_buf(),
            reason: "unsupported filesystem entry".to_owned(),
        });
    }
    Ok(())
}

#[derive(Clone)]
pub struct ModelHandle {
    model_id: String,
    backend: String,
    format: ModelFormat,
    inner: Arc<dyn Any + Send + Sync>,
}

impl ModelHandle {
    pub fn new(
        model_id: impl Into<String>,
        backend: impl Into<String>,
        format: ModelFormat,
        inner: Arc<dyn Any + Send + Sync>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            backend: backend.into(),
            format,
            inner,
        }
    }

    pub fn id(&self) -> &str {
        &self.model_id
    }

    pub fn backend(&self) -> &str {
        &self.backend
    }

    pub fn format(&self) -> &ModelFormat {
        &self.format
    }

    pub fn state<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.inner.as_ref().downcast_ref::<T>()
    }
}

impl std::fmt::Debug for ModelHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelHandle")
            .field("model_id", &self.model_id)
            .field("backend", &self.backend)
            .field("format", &self.format)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArtifactSpec, FilesystemModelCatalog, ModelCatalog, ModelFormat, ModelManifest,
        ModelReference,
    };
    use std::{fs, path::PathBuf};

    #[test]
    fn accepts_lowercase_manifest_formats() {
        let manifest: ModelManifest = serde_json::from_str(
            r#"{"id":"example","format":"onnx","artifact":"artifacts/model.onnx"}"#,
        )
        .unwrap();
        assert_eq!(manifest.format, ModelFormat::Onnx);
    }

    #[test]
    fn accepts_artifact_integrity_metadata() {
        let manifest: ModelManifest = serde_json::from_str(
            r#"{"id":"example","format":"onnx","artifact":{"path":"model.onnx","size_bytes":3,"sha256":"abc"}}"#,
        )
        .unwrap();
        assert_eq!(
            manifest.artifact,
            ArtifactSpec {
                path: "model.onnx".to_owned(),
                size_bytes: Some(3),
                sha256: Some("abc".to_owned()),
            }
        );
    }

    #[tokio::test]
    async fn discovers_and_resolves_existing_packages_deterministically() {
        let root = std::env::temp_dir().join(format!("ml-runtime-catalog-{}", std::process::id()));
        let first = root.join("nested/example-linear");
        fs::create_dir_all(first.join("artifacts")).unwrap();
        fs::write(
            first.join("manifest.json"),
            r#"{"id":"example-linear","model_version":"1.0.0","format":"unknown","artifact":"artifacts/model.json"}"#,
        )
        .unwrap();
        fs::write(first.join("artifacts/model.json"), "{}").unwrap();

        let catalog = FilesystemModelCatalog::new(&root);
        let models = catalog.list().await.unwrap();
        assert_eq!(models[0].id.to_string(), "example-linear@1.0.0");
        let resolved = catalog
            .resolve(&ModelReference::id(
                "example-linear",
                Some("1.0.0".to_owned()),
            ))
            .await
            .unwrap();
        assert_eq!(resolved.package_path, PathBuf::from(&first));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn installs_verified_package_atomically_and_refreshes_catalog() {
        use super::{
            AcquisitionConfig, FileModelSource, InstallMode, InstallResult, ModelFetchRequest,
            ModelId, ModelInstaller, ModelSourceReference,
        };
        use std::sync::Arc;
        let root = std::env::temp_dir().join(format!("ml-runtime-install-{}", std::process::id()));
        let source_root = root.join("source");
        let catalog_root = root.join("models");
        fs::create_dir_all(source_root.join("artifacts")).unwrap();
        fs::write(
            source_root.join("manifest.json"),
            r#"{"id":"acquired","model_version":"1.0.0","format":"unknown","artifact":"artifacts/model.bin"}"#,
        )
        .unwrap();
        fs::write(source_root.join("artifacts/model.bin"), b"model").unwrap();

        let catalog = Arc::new(FilesystemModelCatalog::new(&catalog_root));
        let installer = ModelInstaller::new(&catalog_root)
            .with_catalog(Arc::clone(&catalog))
            .with_config(AcquisitionConfig {
                max_download_bytes: 1024,
                timeout_secs: 30,
            });
        let request = ModelFetchRequest {
            model: ModelId::new("acquired", "1.0.0").unwrap(),
            source: ModelSourceReference::Path(source_root.clone()),
        };
        let result = installer
            .install(
                &FileModelSource::new(),
                &request,
                InstallMode::KeepExisting,
                None,
            )
            .await
            .unwrap();
        assert!(matches!(result, InstallResult::Installed(_)));
        assert_eq!(catalog.list().await.unwrap().len(), 1);
        let existing = installer
            .install(
                &FileModelSource::new(),
                &request,
                InstallMode::KeepExisting,
                None,
            )
            .await
            .unwrap();
        assert!(matches!(existing, InstallResult::AlreadyInstalled(_)));
        assert!(!catalog_root
            .join(".staging")
            .join("acquired-1.0.0-")
            .exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn rejects_identity_mismatch_without_catalog_entry() {
        use super::{
            FileModelSource, InstallMode, ModelFetchRequest, ModelId, ModelInstaller,
            ModelSourceReference,
        };
        use std::sync::Arc;
        let root = std::env::temp_dir().join(format!("ml-runtime-mismatch-{}", std::process::id()));
        let source = root.join("source");
        fs::create_dir_all(source.join("artifacts")).unwrap();
        fs::write(
            source.join("manifest.json"),
            r#"{"id":"other","model_version":"1.0.0","format":"unknown","artifact":"artifacts/model.bin"}"#,
        )
        .unwrap();
        fs::write(source.join("artifacts/model.bin"), b"model").unwrap();
        let catalog = Arc::new(FilesystemModelCatalog::new(root.join("models")));
        let installer = ModelInstaller::new(root.join("models")).with_catalog(Arc::clone(&catalog));
        let request = ModelFetchRequest {
            model: ModelId::new("requested", "1.0.0").unwrap(),
            source: ModelSourceReference::Path(source),
        };
        assert!(matches!(
            installer
                .install(
                    &FileModelSource::new(),
                    &request,
                    InstallMode::KeepExisting,
                    None
                )
                .await,
            Err(super::AcquisitionError::IdentityMismatch { .. })
        ));
        assert!(catalog.list().await.unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }
}
