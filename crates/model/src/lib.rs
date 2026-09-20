use serde::{Deserialize, Serialize};
use std::{
    any::Any,
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

pub type ModelMetadata = BTreeMap<String, String>;

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
    pub artifact: String,
    #[serde(default)]
    pub metadata: ModelMetadata,
    #[serde(default)]
    pub inputs: Vec<ModelTensorSpec>,
    #[serde(default)]
    pub outputs: Vec<ModelTensorSpec>,
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
        let artifact = root.join(&manifest.artifact);
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelReference {
    Id(String),
    Spec(ModelSpec),
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
    use super::{ModelFormat, ModelManifest};

    #[test]
    fn accepts_lowercase_manifest_formats() {
        let manifest: ModelManifest = serde_json::from_str(
            r#"{"id":"example","format":"onnx","artifact":"artifacts/model.onnx"}"#,
        )
        .unwrap();
        assert_eq!(manifest.format, ModelFormat::Onnx);
    }
}
