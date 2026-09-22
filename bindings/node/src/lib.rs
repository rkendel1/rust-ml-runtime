//! In-process Node-API boundary for model-neutral typed decisions.
//!
//! Model installation, backend selection, preprocessing, compiled caches, and
//! output decoding remain owned by `ml-runtime`. Consumers exchange only the
//! public `DecisionRequest` and `DecisionResult` JSON contracts.

use ml_runtime::{DecisionModel, DecisionRequest, Runtime};
use napi::{Error, Result, Status};
use napi_derive::napi;

fn native_error(context: &str, error: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, format!("{context}: {error}"))
}

#[napi]
pub struct LocalDecisionModel {
    model: Box<dyn DecisionModel>,
}

#[napi]
impl LocalDecisionModel {
    /// Loads and validates one installed model. The returned object owns the
    /// loaded native model and reuses it for every subsequent decision.
    #[napi(constructor)]
    pub fn new(model: String, model_root: Option<String>) -> Result<Self> {
        let mut builder = Runtime::builder();
        if let Some(root) = model_root {
            builder = builder.model_root(root);
        }
        let loaded = builder
            .build()
            .load_model(&model)
            .map_err(|error| native_error(&format!("could not load model {model:?}"), error))?;
        Ok(Self { model: loaded })
    }

    /// Returns the runtime-owned model description without exposing any
    /// backend-specific representation.
    #[napi(js_name = "descriptionJson")]
    pub fn description_json(&self) -> Result<String> {
        serde_json::to_string(&self.model.describe())
            .map_err(|error| native_error("could not encode model description", error))
    }

    /// Executes the runtime's model-neutral typed-decision contract.
    #[napi(js_name = "decideJson")]
    pub fn decide_json(&self, request_json: String) -> Result<String> {
        let request: DecisionRequest = serde_json::from_str(&request_json)
            .map_err(|error| native_error("invalid DecisionRequest JSON", error))?;
        let result = self
            .model
            .decide(&request)
            .map_err(|error| native_error("local model inference failed", error))?;
        serde_json::to_string(&result)
            .map_err(|error| native_error("could not encode DecisionResult", error))
    }
}
