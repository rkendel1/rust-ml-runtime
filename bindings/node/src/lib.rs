//! In-process Node-API boundary for model-neutral typed decisions.
//!
//! Model installation, backend selection, preprocessing, compiled caches, and
//! output decoding remain owned by `ml-runtime`. Consumers exchange only the
//! public `DecisionRequest` and `DecisionResult` JSON contracts.

use ml_runtime::{
    CancellationToken, DecisionExecutionPolicy, DecisionGraphRequest, DecisionModel,
    DecisionRequest, InferenceRequest, ModelFormat, ModelLocation, ModelSpec, Runtime,
};
use napi::{bindgen_prelude::AsyncTask, Env, Error, Result, Status, Task};
use napi_derive::napi;

fn native_error(context: &str, error: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, format!("{context}: {error}"))
}

/// Runs a minimal in-process CPU inference through the same native runtime
/// linked into the addon. Release smoke tests use this to prove more than a
/// successful dynamic-library load.
#[napi(js_name = "runtimeSmokeTest")]
pub fn runtime_smoke_test() -> Result<String> {
    let model = ModelSpec::new(
        "node-release-smoke",
        ModelFormat::Unknown,
        ModelLocation::Memory,
    );
    let executor = tokio::runtime::Runtime::new()
        .map_err(|error| native_error("could not initialize runtime smoke executor", error))?;
    let result = executor
        .block_on(Runtime::new().infer(InferenceRequest::new(model, "bookworm-smoke")))
        .map_err(|error| native_error("runtime smoke inference failed", error))?;
    serde_json::to_string(&result)
        .map_err(|error| native_error("could not encode runtime smoke inference", error))
}

#[napi]
pub struct LocalDecisionModel {
    model: std::sync::Arc<dyn DecisionModel>,
}

/// Cooperative cancellation handle for queued or multi-stage decision work.
/// Backends report separately whether an already-running native operation can
/// be interrupted.
#[napi]
pub struct DecisionCancellation {
    token: CancellationToken,
}

impl Default for DecisionCancellation {
    fn default() -> Self {
        Self {
            token: CancellationToken::new(),
        }
    }
}

enum DecisionWork {
    Independent(DecisionRequest, DecisionExecutionPolicy),
    Graph(DecisionGraphRequest),
}

pub struct DecisionTask {
    model: std::sync::Arc<dyn DecisionModel>,
    work: Option<DecisionWork>,
    cancellation: Option<CancellationToken>,
}

impl Task for DecisionTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<Self::Output> {
        let work = self.work.take().ok_or_else(|| {
            native_error(
                "decision task could not start",
                "request was already consumed",
            )
        })?;
        let executor = tokio::runtime::Runtime::new()
            .map_err(|error| native_error("could not initialize decision executor", error))?;
        let runtime = Runtime::new();
        let result = executor
            .block_on(async {
                match work {
                    DecisionWork::Independent(request, policy) => {
                        runtime
                            .decide_async(
                                self.model.clone(),
                                request,
                                policy,
                                self.cancellation.clone(),
                            )
                            .await
                    }
                    DecisionWork::Graph(request) => {
                        runtime
                            .execute_decision_graph(
                                self.model.clone(),
                                request,
                                self.cancellation.clone(),
                            )
                            .await
                    }
                }
            })
            .map_err(|error| native_error("asynchronous decision execution failed", error))?;
        serde_json::to_string(&result)
            .map_err(|error| native_error("could not encode DecisionResult", error))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

#[napi]
impl DecisionCancellation {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[napi]
    pub fn cancel(&self) {
        self.token.cancel();
    }

    #[napi(getter, js_name = "isCancelled")]
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }
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
        Ok(Self {
            model: std::sync::Arc::from(loaded),
        })
    }

    /// Returns the runtime-owned model description without exposing any
    /// backend-specific representation.
    #[napi(js_name = "descriptionJson")]
    pub fn description_json(&self) -> Result<String> {
        serde_json::to_string(&self.model.describe())
            .map_err(|error| native_error("could not encode model description", error))
    }

    /// Returns model/backend/device execution facts consumed by the planner.
    #[napi(js_name = "capabilitiesJson")]
    pub fn capabilities_json(&self) -> Result<String> {
        serde_json::to_string(&self.model.capabilities())
            .map_err(|error| native_error("could not encode model capabilities", error))
    }

    /// Produces the deterministic execution plan without running inference.
    #[napi(js_name = "explainDecisionJson")]
    pub fn explain_decision_json(&self, request_json: String) -> Result<String> {
        let request: DecisionGraphRequest = serde_json::from_str(&request_json)
            .map_err(|error| native_error("invalid DecisionGraphRequest JSON", error))?;
        let plan = Runtime::new()
            .explain_decision(self.model.as_ref(), &request)
            .map_err(|error| native_error("could not plan decision graph", error))?;
        serde_json::to_string(&plan)
            .map_err(|error| native_error("could not encode decision plan", error))
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

    /// Runs independent typed decisions away from the JavaScript event loop.
    #[napi(js_name = "decideAsyncJson")]
    pub fn decide_async_json(
        &self,
        request_json: String,
        policy_json: Option<String>,
        cancellation: Option<&DecisionCancellation>,
    ) -> Result<AsyncTask<DecisionTask>> {
        let request: DecisionRequest = serde_json::from_str(&request_json)
            .map_err(|error| native_error("invalid DecisionRequest JSON", error))?;
        let policy = policy_json
            .map(|json| {
                serde_json::from_str::<DecisionExecutionPolicy>(&json)
                    .map_err(|error| native_error("invalid DecisionExecutionPolicy JSON", error))
            })
            .transpose()?
            .unwrap_or_default();
        let token = cancellation.map(|handle| handle.token.clone());
        Ok(AsyncTask::new(DecisionTask {
            model: self.model.clone(),
            work: Some(DecisionWork::Independent(request, policy)),
            cancellation: token,
        }))
    }

    /// Executes a dependency-aware decision graph away from the JavaScript
    /// event loop, using only capabilities declared by the loaded model.
    #[napi(js_name = "executeGraphJson")]
    pub fn execute_graph_json(
        &self,
        request_json: String,
        cancellation: Option<&DecisionCancellation>,
    ) -> Result<AsyncTask<DecisionTask>> {
        let request: DecisionGraphRequest = serde_json::from_str(&request_json)
            .map_err(|error| native_error("invalid DecisionGraphRequest JSON", error))?;
        let token = cancellation.map(|handle| handle.token.clone());
        Ok(AsyncTask::new(DecisionTask {
            model: self.model.clone(),
            work: Some(DecisionWork::Graph(request)),
            cancellation: token,
        }))
    }
}
