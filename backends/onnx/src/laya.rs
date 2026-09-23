//! Typed-decision adapter for the pinned `receptron/laya-onnx` bundle.

use ml_runtime_backend::DecisionModel;
use ml_runtime_common::{RuntimeError, RuntimeResult};
use ml_runtime_inference::{
    DecisionExecution, DecisionModelCapabilities, DecisionProvenance, DecisionRequest,
    DecisionResult, DecisionType, DecisionValue, ModelDescription, ModelIdentity, TypedDecision,
};
use ml_runtime_model::InstallationManifest;
use ort::{memory::Allocator, session::Session, tensor::TensorElementType, value::Tensor};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Instant,
};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
struct AgentConfig {
    max_len: usize,
    head_max_len: usize,
    temperature: [f64; 3],
    #[serde(default)]
    temperature_by_options: BTreeMap<String, f64>,
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
    revision: Option<String>,
    config: AgentConfig,
    tokenizer: Tokenizer,
    tokens: SpecialTokens,
    session: Mutex<Session>,
}

struct Prepared {
    ids: Vec<u32>,
    markers: Vec<usize>,
    qtype: usize,
    kind: &'static str,
}

impl LayaModel {
    pub(crate) fn validate(root: &Path) -> RuntimeResult<String> {
        Self::load(root).map(|model| model.artifact_hash)
    }

    pub(crate) fn load(root: &Path) -> RuntimeResult<Self> {
        let root = root
            .canonicalize()
            .map_err(|error| RuntimeError::ModelNotFound {
                model: format!("{}: {error}", root.display()),
            })?;
        if !root.is_dir() {
            return Err(RuntimeError::ModelUnavailable {
                model: root.display().to_string(),
                provider: "onnx".to_owned(),
                reason: "Laya ONNX artifact must be a directory".to_owned(),
            });
        }
        let required = [
            "laya.onnx",
            "laya.onnx.data",
            "laya_config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ];
        for relative in required {
            let path = root.join(relative);
            let metadata =
                fs::symlink_metadata(&path).map_err(|error| RuntimeError::ModelIntegrity {
                    model: root.display().to_string(),
                    reason: format!("required bundle file {relative:?} is unavailable: {error}"),
                })?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(RuntimeError::ModelIntegrity {
                    model: root.display().to_string(),
                    reason: format!("required bundle entry {relative:?} must be a regular file"),
                });
            }
        }
        let config: AgentConfig = read_json(&root.join("laya_config.json"))?;
        if !(32..=4096).contains(&config.max_len)
            || config.head_max_len < 16
            || config.head_max_len >= config.max_len
            || config
                .temperature
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || config
                .temperature_by_options
                .values()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(RuntimeError::ModelIntegrity {
                model: root.display().to_string(),
                reason: "invalid Laya sequence or calibration configuration".to_owned(),
            });
        }
        let tokenizer_path = root.join("tokenizer/tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            RuntimeError::ModelIntegrity {
                model: root.display().to_string(),
                reason: format!("load {}: {error}", tokenizer_path.display()),
            }
        })?;
        let token_config: TokenizerConfig =
            read_json(&root.join("tokenizer/tokenizer_config.json"))?;
        let cls = token_config.cls_token.into_string();
        let sep = token_config.sep_token.into_string();
        let pad = token_config.pad_token.into_string();
        let mask_text = token_config.mask_token.into_string();
        let token_id = |name: &str, token: &str| {
            tokenizer
                .token_to_id(token)
                .ok_or_else(|| RuntimeError::ModelIntegrity {
                    model: root.display().to_string(),
                    reason: format!("tokenizer has no {name} token {token:?}"),
                })
        };
        let tokens = SpecialTokens {
            cls: token_id("cls", &cls)?,
            sep: token_id("sep", &sep)?,
            pad: token_id("pad", &pad)?,
            mask: token_id("mask", &mask_text)?,
            mask_text,
        };
        let model_path = root.join("laya.onnx");
        let session = Session::builder()
            .and_then(|builder| builder.commit_from_file(&model_path))
            .map_err(|error| RuntimeError::ModelIntegrity {
                model: root.display().to_string(),
                reason: format!("load ONNX graph and external weights: {error}"),
            })?;
        validate_schema(&session, &root)?;
        let artifact_hash = sha256_file(&model_path)?;
        let revision = InstallationManifest::read(&root)
            .ok()
            .map(|manifest| manifest.model_revision);
        Ok(Self {
            root,
            artifact_hash,
            revision,
            config,
            tokenizer,
            tokens,
            session: Mutex::new(session),
        })
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
        let (kind, qtype, rendered) = match &question.kind {
            DecisionType::Choice { options } => {
                if options.is_empty() || options.iter().any(|option| option.label.is_empty()) {
                    return Err(RuntimeError::InvalidInput {
                        reason: format!(
                            "choice {:?} needs nonempty labels and options",
                            question.name
                        ),
                    });
                }
                let mut seen = BTreeSet::new();
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
                        optional_description(false_description, "no, the statement does not hold")
                    ),
                    format!(
                        "true: {}",
                        optional_description(true_description, "yes, the statement holds")
                    ),
                ],
            ),
        };
        let scrub = |text: &str| text.replace(&self.tokens.mask_text, " ");
        let mut head = self.encode(&format!(
            "{kind} question: {}",
            scrub(&question.instructions)
        ))?;
        let mut options = Vec::with_capacity(rendered.len());
        for option in rendered {
            let mut ids = vec![self.tokens.mask];
            ids.extend(
                self.encode(&format!(" {}", scrub(&option)))?
                    .into_iter()
                    .take(48),
            );
            options.push(ids);
        }
        let mut head_budget = self
            .config
            .head_max_len
            .saturating_sub(options.iter().map(Vec::len).sum::<usize>());
        if head_budget < 16 {
            let per = ((self.config.head_max_len.saturating_sub(16)) / options.len().max(1)).max(4);
            options.iter_mut().for_each(|option| option.truncate(per));
            head_budget = self
                .config
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
        let room = self.config.max_len.saturating_sub(ids.len() + 1);
        ids.extend(self.encode(&scrub(&state_text))?.into_iter().take(room));
        ids.push(self.tokens.sep);
        if markers.len() != rendered_len(&question.kind) || ids.len() > self.config.max_len {
            return Err(RuntimeError::InvalidInput {
                reason: format!("decision {:?} exceeds the Laya token budget", question.name),
            });
        }
        Ok(Prepared {
            ids,
            markers,
            qtype,
            kind,
        })
    }

    fn temperature(&self, qtype: usize, count: usize) -> f64 {
        let kind = ["choice", "score", "noul"][qtype];
        let bucket = if count <= 2 {
            "2"
        } else if count <= 5 {
            "3-5"
        } else if count <= 10 {
            "6-10"
        } else {
            "11+"
        };
        self.config
            .temperature_by_options
            .get(&format!("{kind}:{bucket}"))
            .copied()
            .unwrap_or(self.config.temperature[qtype])
    }

    fn infer(&self, prepared: &[Prepared]) -> RuntimeResult<(Vec<f32>, Vec<f32>, usize)> {
        let n = prepared.len();
        let length = prepared
            .iter()
            .map(|item| item.ids.len())
            .max()
            .unwrap_or(0);
        let width = prepared
            .iter()
            .map(|item| item.markers.len())
            .max()
            .unwrap_or(0);
        let mut input_ids = vec![self.tokens.pad as i64; n * length];
        let mut attention = vec![0_i64; n * length];
        let mut marker_pos = vec![0_i64; n * width];
        let mut qtype = vec![0_i64; n];
        let mut marker_mask =
            Tensor::<bool>::new(&Allocator::default(), [n, width]).map_err(ort_input_error)?;
        let (_, mask_values) = marker_mask
            .try_extract_tensor_mut::<bool>()
            .map_err(ort_input_error)?;
        mask_values.fill(false);
        for (row, item) in prepared.iter().enumerate() {
            for (column, value) in item.ids.iter().enumerate() {
                input_ids[row * length + column] = *value as i64;
                attention[row * length + column] = 1;
            }
            for (column, value) in item.markers.iter().enumerate() {
                marker_pos[row * width + column] = *value as i64;
                mask_values[row * width + column] = true;
            }
            qtype[row] = item.qtype as i64;
        }
        let input_ids = Tensor::from_array(([n, length], input_ids)).map_err(ort_input_error)?;
        let attention = Tensor::from_array(([n, length], attention)).map_err(ort_input_error)?;
        let marker_pos = Tensor::from_array(([n, width], marker_pos)).map_err(ort_input_error)?;
        let qtype = Tensor::from_array(([n], qtype)).map_err(ort_input_error)?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| RuntimeError::execution("ONNX Laya inference", "session lock poisoned"))?;
        let outputs = session
            .run(ort::inputs! {
                "input_ids" => input_ids,
                "attention_mask" => attention,
                "marker_pos" => marker_pos,
                "marker_mask" => marker_mask,
                "qtype" => qtype,
            })
            .map_err(|error| RuntimeError::execution("ONNX Laya inference", error.to_string()))?;
        let (logits_shape, logits) =
            outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|error| RuntimeError::InvalidOutput {
                    reason: format!("read Laya logits: {error}"),
                })?;
        let (actions_shape, actions) =
            outputs["act_probs"]
                .try_extract_tensor::<f32>()
                .map_err(|error| RuntimeError::InvalidOutput {
                    reason: format!("read Laya act_probs: {error}"),
                })?;
        if logits_shape.as_ref() != [n as i64, width as i64]
            || actions_shape.as_ref() != [n as i64, 2]
        {
            return Err(RuntimeError::InvalidOutput { reason: format!("Laya returned logits {logits_shape:?} and act_probs {actions_shape:?}; expected [{n}, {width}] and [{n}, 2]") });
        }
        Ok((logits.to_vec(), actions.to_vec(), width))
    }
}

impl DecisionModel for LayaModel {
    fn describe(&self) -> ModelDescription {
        ModelDescription {
            identifier: "receptron/laya-onnx".to_owned(),
            revision: self.revision.clone(),
            backend: "onnx".to_owned(),
            artifact_path: self.root.clone(),
            artifact_sha256: self.artifact_hash.clone(),
            decision_types: vec!["choice".to_owned(), "score".to_owned(), "noul".to_owned()],
        }
    }

    fn capabilities(&self) -> DecisionModelCapabilities {
        crate::OnnxBackend::laya_decision_capabilities()
    }

    fn decide(&self, request: &DecisionRequest) -> RuntimeResult<DecisionResult> {
        if request.decisions.is_empty() {
            return Err(RuntimeError::InvalidRequest {
                reason: "at least one typed decision is required".to_owned(),
            });
        }
        let started = Instant::now();
        let prepared: Vec<_> = request
            .decisions
            .iter()
            .map(|question| {
                if question.name.is_empty() || question.instructions.is_empty() {
                    return Err(RuntimeError::InvalidInput {
                        reason: "decision name and instructions must not be empty".to_owned(),
                    });
                }
                self.prepare(&request.state, question)
            })
            .collect::<RuntimeResult<_>>()?;
        let input_tokens = prepared.iter().map(|item| item.ids.len() as u64).sum();
        let (logits, actions, width) = self.infer(&prepared)?;
        if logits
            .iter()
            .chain(&actions)
            .any(|value| !value.is_finite())
        {
            return Err(RuntimeError::InvalidOutput {
                reason: "Laya returned non-finite values".to_owned(),
            });
        }
        let decisions = request
            .decisions
            .iter()
            .zip(&prepared)
            .enumerate()
            .map(|(row, (question, item))| {
                let count = item.markers.len();
                let probabilities = softmax(
                    &logits[row * width..row * width + count],
                    self.temperature(item.qtype, count),
                )?;
                decode(question, item.kind, &probabilities, actions[row * 2] as f64)
            })
            .collect::<RuntimeResult<_>>()?;
        Ok(DecisionResult {
            model: ModelIdentity {
                identifier: "receptron/laya-onnx".to_owned(),
                revision: self.revision.clone(),
            },
            backend: "onnx".to_owned(),
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

fn validate_schema(session: &Session, root: &Path) -> RuntimeResult<()> {
    let inputs: BTreeSet<_> = session
        .inputs
        .iter()
        .map(|input| input.name.as_str())
        .collect();
    let outputs: BTreeSet<_> = session
        .outputs
        .iter()
        .map(|output| output.name.as_str())
        .collect();
    let expected_inputs = BTreeSet::from([
        "input_ids",
        "attention_mask",
        "marker_pos",
        "marker_mask",
        "qtype",
    ]);
    let expected_outputs = BTreeSet::from(["logits", "act_probs"]);
    if inputs != expected_inputs || outputs != expected_outputs {
        return Err(RuntimeError::ModelIntegrity {
            model: root.display().to_string(),
            reason: format!("unexpected Laya ONNX schema: inputs {inputs:?}, outputs {outputs:?}"),
        });
    }
    let input_contract = [
        ("input_ids", TensorElementType::Int64, 2),
        ("attention_mask", TensorElementType::Int64, 2),
        ("marker_pos", TensorElementType::Int64, 2),
        ("marker_mask", TensorElementType::Bool, 2),
        ("qtype", TensorElementType::Int64, 1),
    ];
    let output_contract = [
        ("logits", TensorElementType::Float32, 2),
        ("act_probs", TensorElementType::Float32, 2),
    ];
    let inputs_valid = input_contract.iter().all(|(name, element, rank)| {
        session
            .inputs
            .iter()
            .find(|input| input.name == *name)
            .is_some_and(|input| {
                input.input_type.tensor_type() == Some(*element)
                    && input
                        .input_type
                        .tensor_shape()
                        .is_some_and(|shape| shape.len() == *rank)
            })
    });
    let outputs_valid = output_contract.iter().all(|(name, element, rank)| {
        session
            .outputs
            .iter()
            .find(|output| output.name == *name)
            .is_some_and(|output| {
                output.output_type.tensor_type() == Some(*element)
                    && output
                        .output_type
                        .tensor_shape()
                        .is_some_and(|shape| shape.len() == *rank)
            })
    });
    if !inputs_valid || !outputs_valid {
        return Err(RuntimeError::ModelIntegrity {
            model: root.display().to_string(),
            reason: "Laya ONNX tensor element types or ranks do not match the v1 contract"
                .to_owned(),
        });
    }
    Ok(())
}

fn ort_input_error(error: ort::Error) -> RuntimeError {
    RuntimeError::execution("ONNX Laya input construction", error.to_string())
}

fn optional_description(value: &Option<Value>, fallback: &str) -> String {
    value
        .as_ref()
        .filter(|value| {
            !value.is_null() && !matches!(value, Value::String(text) if text.is_empty())
        })
        .map(render_value)
        .unwrap_or_else(|| fallback.to_owned())
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
            let index = probabilities
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index)
                .ok_or_else(|| RuntimeError::InvalidOutput {
                    reason: "Laya returned no choice probabilities".to_owned(),
                })?;
            DecisionValue::Choice(options[index].label.clone())
        }
        DecisionType::Score { .. } => {
            for (index, probability) in probabilities.iter().enumerate() {
                mapped.insert(index.to_string(), *probability as f64);
            }
            DecisionValue::Score(
                probabilities
                    .iter()
                    .enumerate()
                    .map(|(index, probability)| index as f64 * *probability as f64)
                    .sum(),
            )
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

fn render_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).expect("serialize JSON string"),
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
                    serde_json::to_string(key).expect("serialize JSON key"),
                    render_json(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> RuntimeResult<T> {
    let bytes = fs::read(path).map_err(|error| RuntimeError::ModelIntegrity {
        model: path.display().to_string(),
        reason: error.to_string(),
    })?;
    serde_json::from_slice(&bytes).map_err(|error| RuntimeError::ModelIntegrity {
        model: path.display().to_string(),
        reason: format!("invalid JSON: {error}"),
    })
}

fn sha256_file(path: &Path) -> RuntimeResult<String> {
    let mut file = fs::File::open(path).map_err(|error| RuntimeError::ModelIntegrity {
        model: path.display().to_string(),
        reason: error.to_string(),
    })?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| RuntimeError::ModelIntegrity {
                model: path.display().to_string(),
                reason: error.to_string(),
            })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_uses_temperature_without_clamping() {
        let probabilities = softmax(&[0.0, 1.0], 0.1).unwrap();
        assert!(probabilities[1] > 0.999);
    }

    #[test]
    fn renders_python_style_json() {
        assert_eq!(
            render_json(&serde_json::json!({"a": [1, true]})),
            r#"{"a": [1, true]}"#
        );
    }
}
