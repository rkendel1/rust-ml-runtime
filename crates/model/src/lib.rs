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
        Ok(Self {
            manifest,
            root,
            artifact,
        })
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
}
