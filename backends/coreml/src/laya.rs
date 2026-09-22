//! Adapter for the `aac6fef/laya-coreml` artifact contract.
//!
//! Laya is not the backend abstraction. This module translates runtime-owned typed
//! decisions to the artifact's five inputs and decodes its two outputs. The v1 contract is:
//! `input_ids`/`attention_mask` `[1,L]`, `marker_pos`/`marker_mask` `[1,32]`, and `qtype`
//! `[1]`, all int32; `logits` `[1,32]` and `action_logits` `[1,2]`, both float.

use ml_runtime_backend::DecisionModel;
use ml_runtime_common::{RuntimeError, RuntimeResult};
use ml_runtime_inference::{
    DecisionExecution, DecisionProvenance, DecisionRequest, DecisionResult, DecisionType,
    DecisionValue, ModelDescription, ModelIdentity, TypedDecision,
};
use ml_runtime_model::{CompiledStatus, InstallationManifest, INSTALLATION_MANIFEST_FILE};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};
use tokenizers::Tokenizer;

const FORMAT: &str = "laya-coreml";
const FORMAT_VERSION: u32 = 1;
const TEMP_MIN: f64 = 0.5;
const TEMP_MAX: f64 = 5.0;

#[derive(Debug, Deserialize)]
struct CoreMlConfig {
    format: String,
    format_version: u32,
    #[serde(default)]
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    repository: Option<String>,
    #[serde(default)]
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    revision: Option<String>,
    package_sha256: Option<String>,
    shape: Shape,
    files: BTreeMap<String, FileIdentity>,
}

#[derive(Debug, Deserialize)]
struct Shape {
    batch_size: usize,
    max_length: usize,
    min_length: usize,
    default_length: usize,
    max_options: usize,
    flexible: bool,
    #[serde(default)]
    lengths: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct FileIdentity {
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct AgentConfig {
    max_len: usize,
    head_max_len: usize,
    #[serde(default = "unit_temperatures")]
    temperature: [f64; 3],
    #[serde(default)]
    temperature_by_options: BTreeMap<String, f64>,
}

fn unit_temperatures() -> [f64; 3] {
    [1.0; 3]
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    cls_token: TokenValue,
    sep_token: TokenValue,
    pad_token: TokenValue,
    mask_token: TokenValue,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenValue {
    String(String),
    Object { content: String },
}

impl TokenValue {
    fn into_string(self) -> String {
        match self {
            Self::String(value) | Self::Object { content: value } => value,
        }
    }
}

struct SpecialTokens {
    cls: u32,
    sep: u32,
    pad: u32,
    mask: u32,
    mask_text: String,
}

pub(crate) struct LayaModel {
    root: PathBuf,
    artifact_hash: String,
    identifier: String,
    revision: Option<String>,
    config: CoreMlConfig,
    agent: AgentConfig,
    tokenizer: Tokenizer,
    tokens: SpecialTokens,
    #[cfg(target_os = "macos")]
    native: super::native::LoadedModel,
}

impl LayaModel {
    pub(crate) fn load(root: &Path) -> RuntimeResult<Self> {
        let root = root
            .canonicalize()
            .map_err(|error| RuntimeError::ModelNotFound {
                model: format!("{}: {error}", root.display()),
            })?;
        if !root.is_dir() {
            return Err(RuntimeError::ModelUnavailable {
                model: root.display().to_string(),
                provider: "coreml".to_owned(),
                reason: "Laya artifact must be a directory".to_owned(),
            });
        }
        reject_symlinks(&root)?;
        let config: CoreMlConfig = read_json(&root.join("coreml_config.json"))?;
        if config.format != FORMAT || config.format_version != FORMAT_VERSION {
            return Err(RuntimeError::UnsupportedModel {
                model: root.display().to_string(),
                backend: "coreml".to_owned(),
                reason: format!(
                    "expected {FORMAT} format version {FORMAT_VERSION}, got {} version {}",
                    config.format, config.format_version
                ),
            });
        }
        validate_shape(&config.shape)?;
        let prepared = prepared_artifact(&root)?;
        if prepared.is_none() {
            validate_files(&root, &config.files, config.package_sha256.is_some())?;
            if let Some(expected) = config.package_sha256.as_deref() {
                let package = root.join("model.mlpackage");
                if package.is_dir() {
                    let actual = tree_digest(&package)?;
                    if !actual.eq_ignore_ascii_case(expected) {
                        return Err(RuntimeError::ModelIntegrity {
                            model: root.display().to_string(),
                            reason: "Core ML package hash does not match coreml_config.json"
                                .to_owned(),
                        });
                    }
                }
            }
        }
        let agent: AgentConfig = read_json(&root.join("rl_agent_config.json"))?;
        validate_agent(&agent, &config.shape)?;
        let tokenizer_path = root.join("tokenizer/tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            RuntimeError::ModelIntegrity {
                model: root.display().to_string(),
                reason: format!("load {}: {error}", tokenizer_path.display()),
            }
        })?;
        let tokenizer_config: TokenizerConfig =
            read_json(&root.join("tokenizer/tokenizer_config.json"))?;
        let cls = tokenizer_config.cls_token.into_string();
        let sep = tokenizer_config.sep_token.into_string();
        let pad = tokenizer_config.pad_token.into_string();
        let mask_text = tokenizer_config.mask_token.into_string();
        let token_id = |name: &str, value: &str| {
            tokenizer
                .token_to_id(value)
                .ok_or_else(|| RuntimeError::ModelIntegrity {
                    model: root.display().to_string(),
                    reason: format!("tokenizer has no valid {name} token {value:?}"),
                })
        };
        let tokens = SpecialTokens {
            cls: token_id("cls", &cls)?,
            sep: token_id("sep", &sep)?,
            pad: token_id("pad", &pad)?,
            mask: token_id("mask", &mask_text)?,
            mask_text,
        };
        #[cfg(not(target_os = "macos"))]
        let _ = &tokens;
        let model_path = if let Some(path) = prepared.as_ref() {
            path.clone()
        } else if root.join("model.mlmodelc").is_dir() {
            root.join("model.mlmodelc")
        } else if root.join("model.mlpackage").is_dir() {
            root.join("model.mlpackage")
        } else {
            return Err(RuntimeError::ModelNotFound {
                model: format!("{}/model.mlmodelc or model.mlpackage", root.display()),
            });
        };
        #[cfg(not(target_os = "macos"))]
        let _ = &model_path;
        #[cfg(target_os = "macos")]
        let native = if prepared.is_some() {
            super::native::load_cpu_gpu(&model_path).map_err(|error| {
                RuntimeError::ModelIntegrity {
                    model: root.display().to_string(),
                    reason: format!(
                        "compiled Core ML artifact is invalid: {error}; run `ml-runtime model doctor laya`"
                    ),
                }
            })?
        } else {
            load_native_model(&model_path, &artifact_hash_for(&config, &root)?)?
        };
        #[cfg(target_os = "macos")]
        validate_native_schema(&native.schema, &config.shape)?;
        #[cfg(not(target_os = "macos"))]
        return Err(RuntimeError::backend_unavailable(
            "coreml",
            "Laya requires macOS",
        ));
        #[cfg(target_os = "macos")]
        {
            let artifact_hash = artifact_hash_for(&config, &root)?;
            Ok(Self {
                artifact_hash,
                identifier: config
                    .repository
                    .clone()
                    .unwrap_or_else(|| "laya".to_owned()),
                revision: config.revision.clone(),
                root,
                config,
                agent,
                tokenizer,
                tokens,
                native,
            })
        }
    }

    fn encode(&self, text: &str) -> RuntimeResult<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| RuntimeError::InvalidInput {
                reason: format!("Laya tokenization failed: {error}"),
            })
    }

    fn prepare(
        &self,
        state: &Value,
        question: &ml_runtime_inference::DecisionQuestion,
    ) -> RuntimeResult<Prepared> {
        let (kind_name, qtype, rendered) =
            match &question.kind {
                DecisionType::Choice { options } => {
                    if options.is_empty() || options.iter().any(|option| option.label.is_empty()) {
                        return Err(RuntimeError::InvalidInput {
                            reason: format!(
                                "choice {:?} needs nonempty labels and options",
                                question.name
                            ),
                        });
                    }
                    let mut seen = std::collections::BTreeSet::new();
                    if options.iter().any(|option| !seen.insert(&option.label)) {
                        return Err(RuntimeError::InvalidInput {
                            reason: format!("choice {:?} labels must be unique", question.name),
                        });
                    }
                    let values = options
                        .iter()
                        .map(|option| match &option.description {
                            None | Some(Value::Null) => option.label.clone(),
                            Some(Value::String(value)) if value.is_empty() => option.label.clone(),
                            Some(value) => format!("{}: {}", option.label, render_value(value)),
                        })
                        .collect();
                    ("choice", 0, values)
                }
                DecisionType::Score { levels } => {
                    if levels.is_empty() {
                        return Err(RuntimeError::InvalidInput {
                            reason: format!("score {:?} needs at least one level", question.name),
                        });
                    }
                    (
                        "score",
                        1,
                        levels
                            .iter()
                            .enumerate()
                            .map(|(index, value)| format!("level {index}: {}", render_value(value)))
                            .collect(),
                    )
                }
                DecisionType::Noul {
                    false_description,
                    true_description,
                } => (
                    "noul",
                    2,
                    vec![
                        format!(
                            "false: {}",
                            false_description
                                .as_ref()
                                .filter(|v| !v.is_null()
                                    && !matches!(v, Value::String(s) if s.is_empty()))
                                .map(render_value)
                                .unwrap_or_else(|| "no, the statement does not hold".to_owned())
                        ),
                        format!(
                            "true: {}",
                            true_description
                                .as_ref()
                                .filter(|v| !v.is_null()
                                    && !matches!(v, Value::String(s) if s.is_empty()))
                                .map(render_value)
                                .unwrap_or_else(|| "yes, the statement holds".to_owned())
                        ),
                    ],
                ),
            };
        if rendered.len() > self.config.shape.max_options {
            return Err(RuntimeError::InvalidInput {
                reason: format!(
                    "decision {:?} has {} options; this artifact supports {}",
                    question.name,
                    rendered.len(),
                    self.config.shape.max_options
                ),
            });
        }
        let instructions = question.instructions.replace(&self.tokens.mask_text, " ");
        let mut head = self.encode(&format!("{kind_name} question: {instructions}"))?;
        let mut options = Vec::with_capacity(rendered.len());
        for option in rendered {
            let mut ids = vec![self.tokens.mask];
            ids.extend(
                self.encode(&format!(" {}", option.replace(&self.tokens.mask_text, " ")))?
                    .into_iter()
                    .take(48),
            );
            options.push(ids);
        }
        let mut head_budget = self
            .agent
            .head_max_len
            .saturating_sub(options.iter().map(Vec::len).sum::<usize>());
        if head_budget < 16 {
            let per = ((self.agent.head_max_len.saturating_sub(16)) / options.len().max(1)).max(4);
            for option in &mut options {
                option.truncate(per);
            }
            head_budget = self
                .agent
                .head_max_len
                .saturating_sub(options.iter().map(Vec::len).sum::<usize>());
        }
        head.truncate(head_budget.max(8));
        let mut ids = vec![self.tokens.cls];
        ids.extend(head);
        ids.push(self.tokens.sep);
        let mut markers = Vec::with_capacity(options.len());
        for option in options {
            markers.push(ids.len());
            ids.extend(option);
        }
        ids.push(self.tokens.sep);
        let state_text = match state {
            Value::String(value) => value.clone(),
            value => render_json(value),
        };
        let state_ids = self.encode(&state_text.replace(&self.tokens.mask_text, " "))?;
        let room = self.agent.max_len.saturating_sub(ids.len() + 1);
        ids.extend(state_ids.into_iter().take(room));
        ids.push(self.tokens.sep);
        if markers.len() != rendered_len(&question.kind) || ids.len() > self.config.shape.max_length
        {
            return Err(RuntimeError::InvalidInput {
                reason: format!("decision {:?} exceeds the Laya token budget", question.name),
            });
        }
        Ok(Prepared {
            ids,
            markers,
            qtype,
            kind_name,
        })
    }

    fn infer(&self, prepared: &Prepared) -> RuntimeResult<(Vec<f32>, Vec<f32>)> {
        let length = padded_length(prepared.ids.len(), &self.config.shape)?;
        let mut ids = vec![self.tokens.pad as i32; length];
        let mut attention = vec![0_i32; length];
        for (index, value) in prepared.ids.iter().enumerate() {
            ids[index] = *value as i32;
            attention[index] = 1;
        }
        let mut marker_pos = vec![0_i32; self.config.shape.max_options];
        let mut marker_mask = vec![0_i32; self.config.shape.max_options];
        for (index, value) in prepared.markers.iter().enumerate() {
            marker_pos[index] = *value as i32;
            marker_mask[index] = 1;
        }
        #[cfg(target_os = "macos")]
        let (logits, actions) = super::native::predict_laya(
            &self.native,
            &[
                ("input_ids", vec![1, length], ids),
                ("attention_mask", vec![1, length], attention),
                (
                    "marker_pos",
                    vec![1, self.config.shape.max_options],
                    marker_pos,
                ),
                (
                    "marker_mask",
                    vec![1, self.config.shape.max_options],
                    marker_mask,
                ),
                ("qtype", vec![1], vec![prepared.qtype]),
            ],
        )
        .map_err(|error| RuntimeError::execution("Core ML Laya inference", error))?;
        #[cfg(target_os = "macos")]
        if logits.shape != [1, self.config.shape.max_options] || actions.shape != [1, 2] {
            return Err(RuntimeError::InvalidOutput {
                reason: format!(
                    "Laya returned logits {:?} and action_logits {:?}; expected [1, {}] and [1, 2]",
                    logits.shape, actions.shape, self.config.shape.max_options
                ),
            });
        }
        #[cfg(target_os = "macos")]
        return Ok((logits.values, actions.values));
        #[cfg(not(target_os = "macos"))]
        unreachable!()
    }
}

fn prepared_artifact(root: &Path) -> RuntimeResult<Option<PathBuf>> {
    if !root.join(INSTALLATION_MANIFEST_FILE).is_file() {
        return Ok(None);
    }
    let manifest =
        InstallationManifest::read(root).map_err(|error| RuntimeError::ModelIntegrity {
            model: root.display().to_string(),
            reason: format!("{error}; run `ml-runtime model doctor laya`"),
        })?;
    if manifest.compiled_status != CompiledStatus::Ready {
        return Err(RuntimeError::ModelIntegrity {
            model: root.display().to_string(),
            reason: "model is not compiled; run `ml-runtime model doctor laya`".to_owned(),
        });
    }
    let compiled = manifest
        .compiled_artifact
        .ok_or_else(|| RuntimeError::ModelIntegrity {
            model: root.display().to_string(),
            reason:
                "installation manifest has no compiled artifact; run `ml-runtime model doctor laya`"
                    .to_owned(),
        })?;
    #[cfg(target_os = "macos")]
    if compiled_cache_path(&manifest.model_checksum).as_deref() != Some(compiled.path.as_path()) {
        return Err(RuntimeError::ModelIntegrity {
            model: root.display().to_string(),
            reason: "compiled artifact path does not match the runtime cache layout; run `ml-runtime model doctor laya`"
                .to_owned(),
        });
    }
    if !compiled.path.is_dir() {
        return Err(RuntimeError::ModelIntegrity {
            model: root.display().to_string(),
            reason: format!(
                "compiled artifact is missing at {}; run `ml-runtime model doctor laya`",
                compiled.path.display()
            ),
        });
    }
    Ok(Some(compiled.path))
}

pub(crate) fn prepare(root: &Path) -> RuntimeResult<super::PreparedCoreMlArtifact> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = root;
        Err(RuntimeError::backend_unavailable(
            "coreml",
            "Laya requires macOS; this platform cannot compile Core ML models",
        ))
    }
    #[cfg(target_os = "macos")]
    {
        let manifest = root.join(INSTALLATION_MANIFEST_FILE);
        if manifest.exists() {
            fs::remove_file(&manifest).map_err(|error| RuntimeError::Execution {
                operation: "prepare Core ML model".to_owned(),
                reason: format!("remove stale {}: {error}", manifest.display()),
            })?;
        }
        let loaded = LayaModel::load(root)?;
        let path = compiled_cache_path(&loaded.artifact_hash).ok_or_else(|| {
            RuntimeError::execution(
                "Core ML model preparation",
                "cannot determine compiled cache directory; set ML_RUNTIME_CACHE_DIR",
            )
        })?;
        if !path.is_dir() {
            return Err(RuntimeError::execution(
                "Core ML model preparation",
                "Core ML compilation did not produce a persistent artifact",
            ));
        }
        let compiled_identity = tree_digest(&path)?;
        Ok(super::PreparedCoreMlArtifact {
            path,
            model_identity: loaded.artifact_hash.clone(),
            compiled_identity,
        })
    }
}

pub(crate) fn validate_prepared(root: &Path) -> RuntimeResult<super::PreparedCoreMlArtifact> {
    let loaded = LayaModel::load(root)?;
    let manifest =
        InstallationManifest::read(root).map_err(|error| RuntimeError::ModelIntegrity {
            model: root.display().to_string(),
            reason: error.to_string(),
        })?;
    let compiled = manifest
        .compiled_artifact
        .ok_or_else(|| RuntimeError::ModelIntegrity {
            model: root.display().to_string(),
            reason: "installation manifest has no compiled artifact".to_owned(),
        })?;
    let path = compiled.path;
    let actual = tree_digest(&path)?;
    if !actual.eq_ignore_ascii_case(&compiled.identity) {
        return Err(RuntimeError::ModelIntegrity {
            model: root.display().to_string(),
            reason: "compiled artifact identity does not match installation manifest".to_owned(),
        });
    }
    Ok(super::PreparedCoreMlArtifact {
        compiled_identity: actual,
        model_identity: loaded.artifact_hash,
        path,
    })
}

pub(crate) fn remove_prepared(root: &Path) -> RuntimeResult<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = root;
        Ok(())
    }
    #[cfg(target_os = "macos")]
    {
        let config: CoreMlConfig = read_json(&root.join("coreml_config.json"))?;
        let expected =
            compiled_cache_path(&artifact_hash_for(&config, root)?).ok_or_else(|| {
                RuntimeError::execution(
                    "remove compiled Core ML artifact",
                    "cannot determine the runtime cache directory",
                )
            })?;
        if let Ok(manifest) = InstallationManifest::read(root) {
            if manifest
                .compiled_artifact
                .is_some_and(|compiled| compiled.path != expected)
            {
                return Err(RuntimeError::ModelIntegrity {
                    model: root.display().to_string(),
                    reason: "installation manifest points outside the expected compiled cache path"
                        .to_owned(),
                });
            }
        }
        if expected.is_dir() {
            fs::remove_dir_all(&expected).map_err(|error| RuntimeError::Execution {
                operation: "remove compiled Core ML artifact".to_owned(),
                reason: error.to_string(),
            })?;
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn artifact_hash_for(config: &CoreMlConfig, root: &Path) -> RuntimeResult<String> {
    match config.package_sha256.clone() {
        Some(hash) => Ok(hash),
        None => hash_directory(root),
    }
}

#[cfg(target_os = "macos")]
fn load_native_model(
    authoritative: &Path,
    artifact_hash: &str,
) -> RuntimeResult<super::native::LoadedModel> {
    if authoritative
        .extension()
        .is_none_or(|extension| extension != "mlpackage")
    {
        return super::native::load_cpu_gpu(authoritative)
            .map_err(|error| RuntimeError::execution("Core ML model load", error));
    }
    let Some(cache) = compiled_cache_path(artifact_hash) else {
        return super::native::load_cpu_gpu(authoritative)
            .map_err(|error| RuntimeError::execution("Core ML model load", error));
    };
    if cache.is_dir() {
        match super::native::load_cpu_gpu(&cache) {
            Ok(model) => return Ok(model),
            Err(_) => fs::remove_dir_all(&cache).map_err(|error| RuntimeError::Execution {
                operation: "remove invalid Core ML cache".to_owned(),
                reason: error.to_string(),
            })?,
        }
    }
    super::native::compile_to_cache(authoritative, &cache)
        .map_err(|error| RuntimeError::execution("Core ML model compilation", error))?;
    super::native::load_cpu_gpu(&cache)
        .map_err(|error| RuntimeError::execution("Core ML compiled model load", error))
}

#[cfg(target_os = "macos")]
fn compiled_cache_path(artifact_hash: &str) -> Option<PathBuf> {
    let root = if let Some(root) = std::env::var_os("ML_RUNTIME_CACHE_DIR") {
        PathBuf::from(root)
    } else {
        PathBuf::from(std::env::var_os("HOME")?).join("Library/Caches/ml-runtime")
    };
    Some(
        root.join("coreml")
            .join(format!("laya-{artifact_hash}.mlmodelc")),
    )
}

impl DecisionModel for LayaModel {
    fn describe(&self) -> ModelDescription {
        ModelDescription {
            identifier: self.identifier.clone(),
            revision: self.revision.clone(),
            backend: "coreml".to_owned(),
            artifact_path: self.root.clone(),
            artifact_sha256: self.artifact_hash.clone(),
            decision_types: vec!["choice".to_owned(), "score".to_owned(), "noul".to_owned()],
        }
    }

    fn decide(&self, request: &DecisionRequest) -> RuntimeResult<DecisionResult> {
        if request.decisions.is_empty() {
            return Err(RuntimeError::InvalidRequest {
                reason: "at least one typed decision is required".to_owned(),
            });
        }
        let started = Instant::now();
        let mut decisions = Vec::with_capacity(request.decisions.len());
        let mut input_tokens = 0_u64;
        for question in &request.decisions {
            if question.name.is_empty() || question.instructions.is_empty() {
                return Err(RuntimeError::InvalidInput {
                    reason: "decision name and instructions must not be empty".to_owned(),
                });
            }
            let prepared = self.prepare(&request.state, question)?;
            input_tokens += prepared.ids.len() as u64;
            let (logits, action_logits) = self.infer(&prepared)?;
            let count = prepared.markers.len();
            if logits.len() < count || action_logits.len() < 2 {
                return Err(RuntimeError::InvalidOutput {
                    reason: "Laya output shape does not satisfy its declared contract".to_owned(),
                });
            }
            if logits
                .iter()
                .chain(&action_logits)
                .any(|value| !value.is_finite())
            {
                return Err(RuntimeError::InvalidOutput {
                    reason: "Laya returned non-finite values".to_owned(),
                });
            }
            let temperature = self.temperature(prepared.qtype as usize, count)?;
            let probabilities = softmax(&logits[..count], temperature)?;
            let actions = softmax(&action_logits[..2], 1.0)?;
            decisions.push(decode(
                question,
                prepared.kind_name,
                &probabilities,
                actions[0] as f64,
            )?);
        }
        Ok(DecisionResult {
            model: ModelIdentity {
                identifier: self.identifier.clone(),
                revision: self.revision.clone(),
            },
            backend: "coreml".to_owned(),
            decisions,
            execution: DecisionExecution {
                latency: started.elapsed(),
                input_tokens,
                output_tokens: 0,
            },
            provenance: DecisionProvenance {
                artifact_path: self.root.clone(),
                artifact_sha256: self.artifact_hash.clone(),
                runtime_version: env!("CARGO_PKG_VERSION").to_owned(),
            },
        })
    }
}

impl LayaModel {
    fn temperature(&self, qtype: usize, count: usize) -> RuntimeResult<f64> {
        let name =
            ["choice", "score", "noul"]
                .get(qtype)
                .ok_or_else(|| RuntimeError::InvalidOutput {
                    reason: "Laya returned an unsupported question type".to_owned(),
                })?;
        let bucket = if count <= 2 {
            "2"
        } else if count <= 5 {
            "3-5"
        } else if count <= 10 {
            "6-10"
        } else {
            "11+"
        };
        let value = self
            .agent
            .temperature_by_options
            .get(&format!("{name}:{bucket}"))
            .copied()
            .unwrap_or(self.agent.temperature[qtype]);
        Ok(value.clamp(TEMP_MIN, TEMP_MAX))
    }
}

struct Prepared {
    ids: Vec<u32>,
    markers: Vec<usize>,
    qtype: i32,
    kind_name: &'static str,
}

fn decode(
    question: &ml_runtime_inference::DecisionQuestion,
    kind: &str,
    probabilities: &[f32],
    action_probability: f64,
) -> RuntimeResult<TypedDecision> {
    let mut mapped = BTreeMap::new();
    let value = match &question.kind {
        DecisionType::Choice { options } => {
            for (option, probability) in options.iter().zip(probabilities) {
                mapped.insert(option.label.clone(), *probability as f64);
            }
            let selected = probabilities
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| options[index].label.clone())
                .ok_or_else(|| RuntimeError::InvalidOutput {
                    reason: "Laya returned no choice probabilities".to_owned(),
                })?;
            DecisionValue::Choice(selected)
        }
        DecisionType::Score { levels } => {
            let score = probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * *probability as f64)
                .sum();
            for (index, probability) in probabilities.iter().enumerate() {
                mapped.insert(index.to_string(), *probability as f64);
            }
            debug_assert_eq!(levels.len(), probabilities.len());
            DecisionValue::Score(score)
        }
        DecisionType::Noul { .. } => {
            mapped.insert("false".to_owned(), probabilities[0] as f64);
            mapped.insert("true".to_owned(), probabilities[1] as f64);
            DecisionValue::Noul(probabilities[1] >= 0.5)
        }
    };
    let confidence = if kind == "noul" {
        probabilities[0].max(probabilities[1]) as f64
    } else {
        entropy_confidence(probabilities)
    };
    Ok(TypedDecision {
        name: question.name.clone(),
        kind: kind.to_owned(),
        value,
        probabilities: mapped,
        confidence,
        action_probability,
    })
}

fn softmax(values: &[f32], temperature: f64) -> RuntimeResult<Vec<f32>> {
    if values.is_empty() || !temperature.is_finite() || temperature <= 0.0 {
        return Err(RuntimeError::InvalidOutput {
            reason: "invalid Laya logits or calibration temperature".to_owned(),
        });
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exponents: Vec<_> = values
        .iter()
        .map(|value| ((*value as f64 - max) / temperature).exp())
        .collect();
    let total: f64 = exponents.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(RuntimeError::InvalidOutput {
            reason: "could not normalize Laya logits".to_owned(),
        });
    }
    Ok(exponents
        .into_iter()
        .map(|value| (value / total) as f32)
        .collect())
}

fn entropy_confidence(probabilities: &[f32]) -> f64 {
    if probabilities.len() < 2 {
        return 1.0;
    }
    let entropy: f64 = probabilities
        .iter()
        .map(|probability| {
            let p = (*probability as f64).clamp(1e-12, 1.0);
            -p * p.ln()
        })
        .sum();
    (1.0 - entropy / (probabilities.len() as f64).ln()).clamp(0.0, 1.0)
}

fn rendered_len(kind: &DecisionType) -> usize {
    match kind {
        DecisionType::Choice { options } => options.len(),
        DecisionType::Score { levels } => levels.len(),
        DecisionType::Noul { .. } => 2,
    }
}

fn render_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        value => render_json(value),
    }
}

/// Match Python's `json.dumps(..., ensure_ascii=False)` spacing used by Laya.
fn render_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).expect("JSON string serialization"),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(render_json)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    serde_json::to_string(key).expect("JSON key serialization"),
                    render_json(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn padded_length(actual: usize, shape: &Shape) -> RuntimeResult<usize> {
    if actual > shape.max_length {
        return Err(RuntimeError::InvalidInput {
            reason: format!(
                "input has {actual} tokens; artifact supports {}",
                shape.max_length
            ),
        });
    }
    if !shape.lengths.is_empty() {
        return shape
            .lengths
            .iter()
            .copied()
            .find(|length| *length >= actual)
            .ok_or_else(|| RuntimeError::InvalidInput {
                reason: format!("no exported Core ML shape can hold {actual} tokens"),
            });
    }
    if shape.flexible {
        Ok(actual.max(shape.min_length).div_ceil(16) * 16)
    } else {
        Ok(shape.max_length)
    }
}

fn validate_shape(shape: &Shape) -> RuntimeResult<()> {
    if shape.batch_size != 1
        || shape.max_options != 32
        || shape.max_length == 0
        || shape.default_length < shape.min_length
        || shape.default_length > shape.max_length
    {
        return Err(RuntimeError::UnsupportedModel {
            model: "laya".to_owned(),
            backend: "coreml".to_owned(),
            reason: "this runtime supports the batch-1, 32-option Laya v1 export".to_owned(),
        });
    }
    if shape.lengths.windows(2).any(|pair| pair[0] >= pair[1])
        || shape
            .lengths
            .iter()
            .any(|length| *length > shape.max_length)
    {
        return Err(RuntimeError::ModelIntegrity {
            model: "laya".to_owned(),
            reason: "invalid exported sequence lengths".to_owned(),
        });
    }
    Ok(())
}

fn validate_agent(agent: &AgentConfig, shape: &Shape) -> RuntimeResult<()> {
    if agent.max_len != shape.max_length
        || agent.head_max_len == 0
        || agent.head_max_len > agent.max_len
        || agent
            .temperature
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || agent
            .temperature_by_options
            .values()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(RuntimeError::ModelIntegrity {
            model: "laya".to_owned(),
            reason: "invalid or inconsistent Laya agent configuration".to_owned(),
        });
    }
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> RuntimeResult<T> {
    let bytes = fs::read(path).map_err(|error| RuntimeError::ModelIntegrity {
        model: path.display().to_string(),
        reason: format!("read metadata: {error}"),
    })?;
    serde_json::from_slice(&bytes).map_err(|error| RuntimeError::ModelIntegrity {
        model: path.display().to_string(),
        reason: format!("parse metadata: {error}"),
    })
}

fn validate_files(
    root: &Path,
    files: &BTreeMap<String, FileIdentity>,
    package_hash_available: bool,
) -> RuntimeResult<()> {
    for (relative, identity) in files {
        let path = Path::new(relative);
        if path.is_absolute()
            || path
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err(RuntimeError::ModelIntegrity {
                model: root.display().to_string(),
                reason: format!("unsafe metadata path {relative:?}"),
            });
        }
        if package_hash_available && relative.starts_with("model.mlpackage/") {
            continue;
        }
        let path = root.join(path);
        let metadata = fs::metadata(&path).map_err(|error| RuntimeError::ModelIntegrity {
            model: root.display().to_string(),
            reason: format!("missing {}: {error}", path.display()),
        })?;
        if metadata.len() != identity.bytes {
            return Err(RuntimeError::ModelIntegrity {
                model: root.display().to_string(),
                reason: format!("size mismatch for {}", path.display()),
            });
        }
        let actual = hash_file(&path)?;
        if !actual.eq_ignore_ascii_case(&identity.sha256) {
            return Err(RuntimeError::ModelIntegrity {
                model: root.display().to_string(),
                reason: format!("SHA-256 mismatch for {}", path.display()),
            });
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_native_schema(schema: &super::native::ModelSchema, shape: &Shape) -> RuntimeResult<()> {
    const INT32: usize = 131104;
    const FLOAT32: usize = 65568;
    let expected_inputs = [
        ("attention_mask", INT32, vec![1, shape.default_length]),
        ("input_ids", INT32, vec![1, shape.default_length]),
        ("marker_mask", INT32, vec![1, shape.max_options]),
        ("marker_pos", INT32, vec![1, shape.max_options]),
        ("qtype", INT32, vec![1]),
    ];
    let expected_outputs = [
        ("action_logits", FLOAT32, vec![1, 2]),
        ("logits", FLOAT32, vec![1, shape.max_options]),
    ];
    validate_feature_group("input", &schema.inputs, &expected_inputs)?;
    validate_feature_group("output", &schema.outputs, &expected_outputs)
}

#[cfg(target_os = "macos")]
fn validate_feature_group(
    group: &str,
    actual: &[super::native::FeatureSchema],
    expected: &[(&str, usize, Vec<usize>)],
) -> RuntimeResult<()> {
    if actual.len() != expected.len()
        || actual.iter().zip(expected).any(|(actual, expected)| {
            actual.name != expected.0
                || actual.data_type != expected.1
                || actual.shape != expected.2
        })
    {
        let actual = actual
            .iter()
            .map(|feature| format!("{}:{}:{:?}", feature.name, feature.data_type, feature.shape))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(RuntimeError::UnsupportedModel {
            model: "laya".to_owned(),
            backend: "coreml".to_owned(),
            reason: format!("unexpected Core ML {group} contract: {actual}"),
        });
    }
    Ok(())
}

fn hash_file(path: &Path) -> RuntimeResult<String> {
    Ok(format!("{:x}", hash_file_bytes(path)?))
}

fn hash_file_bytes(path: &Path) -> RuntimeResult<sha2::digest::Output<Sha256>> {
    let mut digest = Sha256::new();
    let mut file = fs::File::open(path).map_err(|error| RuntimeError::ModelIntegrity {
        model: path.display().to_string(),
        reason: format!("hash artifact: {error}"),
    })?;
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| RuntimeError::ModelIntegrity {
                model: path.display().to_string(),
                reason: format!("hash artifact: {error}"),
            })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest.finalize())
}

fn tree_digest(root: &Path) -> RuntimeResult<String> {
    fn visit(root: &Path, path: &Path, digest: &mut Sha256) -> RuntimeResult<()> {
        let mut entries = fs::read_dir(path)
            .map_err(|error| RuntimeError::ModelIntegrity {
                model: root.display().to_string(),
                reason: error.to_string(),
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| RuntimeError::ModelIntegrity {
                model: root.display().to_string(),
                reason: error.to_string(),
            })?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                visit(root, &path, digest)?;
            } else {
                digest.update(
                    path.strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .as_bytes(),
                );
                digest.update(hash_file_bytes(&path)?);
            }
        }
        Ok(())
    }
    let mut digest = Sha256::new();
    visit(root, root, &mut digest)?;
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(target_os = "macos")]
fn hash_directory(root: &Path) -> RuntimeResult<String> {
    tree_digest(root)
}

fn reject_symlinks(path: &Path) -> RuntimeResult<()> {
    let metadata = fs::symlink_metadata(path).map_err(|error| RuntimeError::ModelIntegrity {
        model: path.display().to_string(),
        reason: error.to_string(),
    })?;
    if metadata.file_type().is_symlink() {
        return Err(RuntimeError::ModelIntegrity {
            model: path.display().to_string(),
            reason: "symbolic links are not accepted as immutable model input".to_owned(),
        });
    }
    if metadata.is_dir() {
        for entry in fs::read_dir(path).map_err(|error| RuntimeError::ModelIntegrity {
            model: path.display().to_string(),
            reason: error.to_string(),
        })? {
            reject_symlinks(
                &entry
                    .map_err(|error| RuntimeError::ModelIntegrity {
                        model: path.display().to_string(),
                        reason: error.to_string(),
                    })?
                    .path(),
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exported_lengths_are_selected_fail_closed() {
        let shape = Shape {
            batch_size: 1,
            max_length: 512,
            min_length: 16,
            default_length: 128,
            max_options: 32,
            flexible: true,
            lengths: vec![16, 32, 64, 128, 512],
        };
        assert_eq!(padded_length(17, &shape).unwrap(), 32);
        assert!(padded_length(513, &shape).is_err());
    }

    #[test]
    fn softmax_is_finite_and_normalized() {
        let values = softmax(&[1.0, 2.0, 3.0], 1.0).unwrap();
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(values.len(), 3);
    }

    #[test]
    fn structured_values_use_laya_json_spacing() {
        assert_eq!(
            render_json(&serde_json::json!({"a": [1, true], "b": "é"})),
            r#"{"a": [1, true], "b": "é"}"#
        );
    }
}
