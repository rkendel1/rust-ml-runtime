use serde::{Deserialize, Serialize};
use std::{any::Any, collections::BTreeMap, sync::Arc};

pub type ModelMetadata = BTreeMap<String, String>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    Onnx,
    CoreMl,
    Gguf,
    Safetensors,
    TensorRt,
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
