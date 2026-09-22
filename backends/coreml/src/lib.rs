use async_trait::async_trait;
use futures_util::stream;
use ml_runtime_backend::{Backend, BackendCapabilities, BackendCapability};
use ml_runtime_common::{BoxStream, CancellationToken, RuntimeError, RuntimeResult};
use ml_runtime_inference::Input;
#[cfg(target_os = "macos")]
use ml_runtime_inference::{ExecutionMetadata, Output, Tensor};
use ml_runtime_inference::{InferenceChunk, InferenceRequest, InferenceResult};
use ml_runtime_model::{ModelFormat, ModelHandle, ModelLocation, ModelSpec};
use std::sync::Arc;

mod laya;

#[derive(Clone, Debug, Default)]
pub struct CoreMlBackend;

#[derive(Clone)]
struct CoreMlLoadedModel {
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    spec: ModelSpec,
    #[cfg(target_os = "macos")]
    loaded: Arc<native::LoadedModel>,
}

#[cfg(target_os = "macos")]
pub(crate) mod native {
    use objc::rc::autoreleasepool;
    use objc::runtime::Object;
    use objc::{class, msg_send, sel, sel_impl};
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::path::{Path, PathBuf};

    type Id = *mut Object;

    pub struct LoadedModel {
        pub model: Id,
        pub schema: ModelSchema,
        compiled_temporary: Option<PathBuf>,
    }

    #[derive(Debug)]
    pub struct FeatureSchema {
        pub name: String,
        pub data_type: usize,
        pub shape: Vec<usize>,
    }

    #[derive(Debug)]
    pub struct ModelSchema {
        pub inputs: Vec<FeatureSchema>,
        pub outputs: Vec<FeatureSchema>,
    }

    pub struct FloatArray {
        pub shape: Vec<usize>,
        pub values: Vec<f32>,
    }

    unsafe impl Send for LoadedModel {}
    unsafe impl Sync for LoadedModel {}

    impl Drop for LoadedModel {
        fn drop(&mut self) {
            unsafe {
                let _: () = msg_send![self.model, release];
            }
            if let Some(path) = &self.compiled_temporary {
                let _ = std::fs::remove_dir_all(path);
            }
        }
    }

    unsafe fn string(value: &str) -> Id {
        let value = CString::new(value).expect("Core ML names cannot contain NUL");
        msg_send![class!(NSString), stringWithUTF8String: value.as_ptr()]
    }

    unsafe fn error_message(error: Id) -> String {
        if error.is_null() {
            return "unknown Core ML error".to_owned();
        }
        let description: Id = msg_send![error, localizedDescription];
        let bytes: *const c_char = msg_send![description, UTF8String];
        if bytes.is_null() {
            "unknown Core ML error".to_owned()
        } else {
            CStr::from_ptr(bytes).to_string_lossy().into_owned()
        }
    }

    unsafe fn rust_string(value: Id) -> Result<String, String> {
        let bytes: *const c_char = msg_send![value, UTF8String];
        if bytes.is_null() {
            Err("Core ML returned an invalid feature name".to_owned())
        } else {
            Ok(CStr::from_ptr(bytes).to_string_lossy().into_owned())
        }
    }

    unsafe fn inspect_features(descriptions: Id) -> Result<Vec<FeatureSchema>, String> {
        let keys: Id = msg_send![descriptions, allKeys];
        let count: usize = msg_send![keys, count];
        let mut features = Vec::with_capacity(count);
        for index in 0..count {
            let key: Id = msg_send![keys, objectAtIndex: index];
            let name = rust_string(key)?;
            let feature: Id = msg_send![descriptions, objectForKey: key];
            let feature_type: usize = msg_send![feature, type];
            if feature_type != 5 {
                return Err(format!(
                    "Core ML feature {name:?} has type {feature_type}; expected MLMultiArray"
                ));
            }
            let constraint: Id = msg_send![feature, multiArrayConstraint];
            if constraint.is_null() {
                return Err(format!(
                    "Core ML feature {name:?} has no multi-array constraint"
                ));
            }
            let data_type: usize = msg_send![constraint, dataType];
            let dimensions: Id = msg_send![constraint, shape];
            let dimension_count: usize = msg_send![dimensions, count];
            let mut shape = Vec::with_capacity(dimension_count);
            for dimension_index in 0..dimension_count {
                let dimension: Id = msg_send![dimensions, objectAtIndex: dimension_index];
                let value: u64 = msg_send![dimension, unsignedLongLongValue];
                shape.push(value as usize);
            }
            features.push(FeatureSchema {
                name,
                data_type,
                shape,
            });
        }
        features.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(features)
    }

    unsafe fn inspect_model(model: Id) -> Result<ModelSchema, String> {
        let description: Id = msg_send![model, modelDescription];
        let inputs: Id = msg_send![description, inputDescriptionsByName];
        let outputs: Id = msg_send![description, outputDescriptionsByName];
        Ok(ModelSchema {
            inputs: inspect_features(inputs)?,
            outputs: inspect_features(outputs)?,
        })
    }

    pub fn load(path: &Path) -> Result<LoadedModel, String> {
        autoreleasepool(|| load_with_compute_units(path, None))
    }

    pub fn load_cpu_gpu(path: &Path) -> Result<LoadedModel, String> {
        autoreleasepool(|| load_with_compute_units(path, Some(1))) // MLComputeUnitsCPUAndGPU
    }

    pub fn compile_to_cache(source: &Path, destination: &Path) -> Result<(), String> {
        let temporary = autoreleasepool(|| unsafe {
            let source_path = string(&source.to_string_lossy());
            let source_url: Id = msg_send![class!(NSURL), fileURLWithPath: source_path];
            let mut error: Id = std::ptr::null_mut();
            let compiled: Id =
                msg_send![class!(MLModel), compileModelAtURL: source_url error: &mut error];
            if compiled.is_null() {
                return Err(error_message(error));
            }
            let compiled_path: Id = msg_send![compiled, path];
            let bytes: *const c_char = msg_send![compiled_path, UTF8String];
            if bytes.is_null() {
                return Err("Core ML returned an invalid compiled model URL".to_owned());
            }
            Ok(PathBuf::from(
                CStr::from_ptr(bytes).to_string_lossy().into_owned(),
            ))
        })?;
        let parent = destination
            .parent()
            .ok_or_else(|| "compiled cache path has no parent".to_owned())?;
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        match std::fs::rename(&temporary, destination) {
            Ok(()) => Ok(()),
            Err(_error) if destination.is_dir() => {
                let _ = std::fs::remove_dir_all(temporary);
                Ok(())
            }
            Err(error) => {
                let _ = std::fs::remove_dir_all(temporary);
                Err(format!("publish compiled Core ML cache: {error}"))
            }
        }
    }

    fn load_with_compute_units(
        path: &Path,
        compute_units: Option<usize>,
    ) -> Result<LoadedModel, String> {
        unsafe {
            let source_path = string(&path.to_string_lossy());
            let source_url: Id = msg_send![class!(NSURL), fileURLWithPath: source_path];
            let mut error: Id = std::ptr::null_mut();
            let (url, compiled_temporary) =
                if path.extension().is_some_and(|value| value == "mlpackage") {
                    let compiled: Id =
                        msg_send![class!(MLModel), compileModelAtURL: source_url error: &mut error];
                    if compiled.is_null() {
                        return Err(error_message(error));
                    }
                    let compiled_path: Id = msg_send![compiled, path];
                    let bytes: *const c_char = msg_send![compiled_path, UTF8String];
                    if bytes.is_null() {
                        return Err("Core ML returned an invalid compiled model URL".to_owned());
                    }
                    (
                        compiled,
                        Some(PathBuf::from(
                            CStr::from_ptr(bytes).to_string_lossy().into_owned(),
                        )),
                    )
                } else {
                    (source_url, None)
                };
            let model: Id = if let Some(compute_units) = compute_units {
                let configuration: Id = msg_send![class!(MLModelConfiguration), new];
                let _: () = msg_send![configuration, setComputeUnits: compute_units];
                let model: Id = msg_send![
                    class!(MLModel),
                    modelWithContentsOfURL: url
                    configuration: configuration
                    error: &mut error
                ];
                let _: () = msg_send![configuration, release];
                model
            } else {
                msg_send![class!(MLModel), modelWithContentsOfURL: url error: &mut error]
            };
            if model.is_null() {
                return Err(error_message(error));
            }
            let schema = inspect_model(model)?;
            let _: Id = msg_send![model, retain];
            Ok(LoadedModel {
                model,
                schema,
                compiled_temporary,
            })
        }
    }

    unsafe fn number(value: usize) -> Id {
        msg_send![class!(NSNumber), numberWithUnsignedLongLong: value as u64]
    }

    unsafe fn array(values: impl IntoIterator<Item = usize>) -> Id {
        let array: Id = msg_send![class!(NSMutableArray), array];
        for value in values {
            let _: () = msg_send![array, addObject: number(value)];
        }
        array
    }

    pub fn predict(
        loaded: &LoadedModel,
        input_name: &str,
        output_name: &str,
        input_shape: &[usize],
        values: &[f32],
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        autoreleasepool(|| unsafe {
            if input_shape.is_empty() || input_shape.iter().product::<usize>() != values.len() {
                return Err("tensor shape does not match tensor values".to_owned());
            }
            let shape = array(input_shape.iter().copied());
            let mut stride = 1;
            let mut strides_values = vec![1; input_shape.len()];
            for (index, dimension) in input_shape.iter().enumerate().rev() {
                strides_values[index] = stride;
                stride *= dimension;
            }
            let strides = array(strides_values);
            let data_type: usize = 65568; // MLMultiArrayDataTypeFloat32
            let data = values.as_ptr() as *mut c_void;
            let deallocator: *mut Object = std::ptr::null_mut();
            let allocated: Id = msg_send![class!(MLMultiArray), alloc];
            let multi_array: Id = msg_send![
                allocated,
                initWithDataPointer: data
                shape: shape
                dataType: data_type
                strides: strides
                deallocator: deallocator
            ];
            if multi_array.is_null() {
                return Err("could not create MLMultiArray input".to_owned());
            }
            let feature_value: Id =
                msg_send![class!(MLFeatureValue), featureValueWithMultiArray: multi_array];
            let dictionary: Id = msg_send![
                class!(NSDictionary),
                dictionaryWithObject: feature_value
                forKey: string(input_name)
            ];
            let mut error: Id = std::ptr::null_mut();
            let allocated: Id = msg_send![class!(MLDictionaryFeatureProvider), alloc];
            let provider: Id = msg_send![
                allocated,
                initWithDictionary: dictionary
                error: &mut error
            ];
            if provider.is_null() {
                return Err(error_message(error));
            }
            let output: Id =
                msg_send![loaded.model, predictionFromFeatures: provider error: &mut error];
            if output.is_null() {
                return Err(error_message(error));
            }
            let output_feature: Id = msg_send![output, featureValueForName: string(output_name)];
            let output_array: Id = msg_send![output_feature, multiArrayValue];
            if output_array.is_null() {
                return Err("Core ML output is not an MLMultiArray".to_owned());
            }
            let count: usize = msg_send![output_array, count];
            let pointer: *const f32 = msg_send![output_array, dataPointer];
            if pointer.is_null() {
                return Err("Core ML returned an invalid output array".to_owned());
            }
            let dimensions: Id = msg_send![output_array, shape];
            let dimension_count: usize = msg_send![dimensions, count];
            let mut output_shape = Vec::with_capacity(dimension_count);
            for index in 0..dimension_count {
                let dimension: Id = msg_send![dimensions, objectAtIndex: index];
                let value: u64 = msg_send![dimension, unsignedLongLongValue];
                output_shape.push(value as usize);
            }
            Ok((
                output_shape,
                std::slice::from_raw_parts(pointer, count).to_vec(),
            ))
        })
    }

    unsafe fn int32_feature(shape_values: &[usize], values: &[i32]) -> Result<Id, String> {
        if shape_values.iter().product::<usize>() != values.len() {
            return Err("Core ML int32 tensor shape does not match its values".to_owned());
        }
        let shape = array(shape_values.iter().copied());
        let mut error: Id = std::ptr::null_mut();
        let allocated: Id = msg_send![class!(MLMultiArray), alloc];
        let multi_array: Id = msg_send![
            allocated,
            initWithShape: shape
            dataType: 131104usize // MLMultiArrayDataTypeInt32
            error: &mut error
        ];
        if multi_array.is_null() {
            return Err(error_message(error));
        }
        let pointer: *mut i32 = msg_send![multi_array, dataPointer];
        if pointer.is_null() {
            let _: () = msg_send![multi_array, release];
            return Err("Core ML returned an invalid int32 input array".to_owned());
        }
        std::ptr::copy_nonoverlapping(values.as_ptr(), pointer, values.len());
        let feature: Id =
            msg_send![class!(MLFeatureValue), featureValueWithMultiArray: multi_array];
        let _: () = msg_send![multi_array, release];
        Ok(feature)
    }

    unsafe fn float_output(provider: Id, name: &str) -> Result<FloatArray, String> {
        let feature: Id = msg_send![provider, featureValueForName: string(name)];
        if feature.is_null() {
            return Err(format!("Core ML did not return required output {name:?}"));
        }
        let array: Id = msg_send![feature, multiArrayValue];
        if array.is_null() {
            return Err(format!("Core ML output {name:?} is not an MLMultiArray"));
        }
        let data_type: usize = msg_send![array, dataType];
        if data_type != 65568usize {
            return Err(format!(
                "Core ML output {name:?} has data type {data_type}; expected float32"
            ));
        }
        let count: usize = msg_send![array, count];
        let pointer: *const f32 = msg_send![array, dataPointer];
        if pointer.is_null() {
            return Err(format!("Core ML output {name:?} has no data"));
        }
        let dimensions: Id = msg_send![array, shape];
        let dimension_count: usize = msg_send![dimensions, count];
        let mut shape = Vec::with_capacity(dimension_count);
        for index in 0..dimension_count {
            let dimension: Id = msg_send![dimensions, objectAtIndex: index];
            let value: u64 = msg_send![dimension, unsignedLongLongValue];
            shape.push(value as usize);
        }
        Ok(FloatArray {
            shape,
            values: std::slice::from_raw_parts(pointer, count).to_vec(),
        })
    }

    pub fn predict_laya(
        loaded: &LoadedModel,
        inputs: &[(&str, Vec<usize>, Vec<i32>)],
    ) -> Result<(FloatArray, FloatArray), String> {
        autoreleasepool(|| unsafe {
            let dictionary: Id = msg_send![class!(NSMutableDictionary), dictionary];
            for (name, shape, values) in inputs {
                let feature = int32_feature(shape, values)?;
                let _: () = msg_send![dictionary, setObject: feature forKey: string(name)];
            }
            let mut error: Id = std::ptr::null_mut();
            let allocated: Id = msg_send![class!(MLDictionaryFeatureProvider), alloc];
            let provider: Id =
                msg_send![allocated, initWithDictionary: dictionary error: &mut error];
            if provider.is_null() {
                return Err(error_message(error));
            }
            let output: Id =
                msg_send![loaded.model, predictionFromFeatures: provider error: &mut error];
            let _: () = msg_send![provider, release];
            if output.is_null() {
                return Err(error_message(error));
            }
            Ok((
                float_output(output, "logits")?,
                float_output(output, "action_logits")?,
            ))
        })
    }
}

impl CoreMlBackend {
    pub fn descriptor() -> BackendCapability {
        BackendCapability::from_parts("coreml", Self.capabilities())
    }
}

impl ml_runtime_backend::DecisionModelProvider for CoreMlBackend {
    fn name(&self) -> &str {
        "coreml"
    }

    fn supports_artifact(&self, artifact: &std::path::Path) -> bool {
        artifact.join("coreml_config.json").is_file()
    }

    fn load_decision_model(
        &self,
        artifact: &std::path::Path,
    ) -> RuntimeResult<Box<dyn ml_runtime_backend::DecisionModel>> {
        if !cfg!(target_os = "macos") {
            return Err(RuntimeError::backend_unavailable(
                "coreml",
                "typed Laya execution requires macOS",
            ));
        }
        laya::LayaModel::load(artifact)
            .map(|model| Box::new(model) as Box<dyn ml_runtime_backend::DecisionModel>)
    }
}

#[async_trait]
impl Backend for CoreMlBackend {
    fn name(&self) -> &str {
        "coreml"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            available: cfg!(target_os = "macos"),
            local: true,
            streaming: false,
            cancellation: false,
            structured_output: false,
            batching: false,
            max_batch_size: None,
            supported_batch_shapes: Vec::new(),
            supported_formats: vec![ModelFormat::CoreMl],
            accelerators: if cfg!(target_os = "macos") {
                vec!["ane".to_owned(), "gpu".to_owned(), "cpu".to_owned()]
            } else {
                Vec::new()
            },
            hardware: cfg!(target_os = "macos").then_some("coreml".to_owned()),
            notes: vec![if cfg!(target_os = "macos") {
                "Apple Core ML local execution".to_owned()
            } else {
                "unsupported platform: Core ML requires macOS".to_owned()
            }],
        }
    }

    fn supports(&self, model: &ModelSpec) -> bool {
        matches!(model.format, ModelFormat::CoreMl)
    }

    async fn load(&self, model: &ModelSpec) -> RuntimeResult<ModelHandle> {
        if !cfg!(target_os = "macos") {
            return Err(RuntimeError::backend_unavailable(
                self.name(),
                "unsupported platform",
            ));
        }
        let ModelLocation::Path(path) = &model.location else {
            return Err(RuntimeError::InvalidInput {
                reason: "Core ML models require a filesystem artifact".to_owned(),
            });
        };
        #[cfg(not(target_os = "macos"))]
        let _ = path;
        #[cfg(target_os = "macos")]
        let loaded = native::load(std::path::Path::new(path))
            .map_err(|error| RuntimeError::execution("Core ML model load", error))?;
        Ok(ModelHandle::new(
            model.id.clone(),
            self.name(),
            model.format.clone(),
            Arc::new(CoreMlLoadedModel {
                spec: model.clone(),
                #[cfg(target_os = "macos")]
                loaded: Arc::new(loaded),
            }),
        ))
    }

    async fn infer(
        &self,
        model: &ModelHandle,
        request: &InferenceRequest,
        cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<InferenceResult> {
        if cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(RuntimeError::Cancelled);
        }
        if !cfg!(target_os = "macos") {
            return Err(RuntimeError::backend_unavailable(
                self.name(),
                "unsupported platform",
            ));
        }
        let loaded =
            model
                .state::<CoreMlLoadedModel>()
                .ok_or_else(|| RuntimeError::InvalidInput {
                    reason: "model handle does not belong to the Core ML backend".to_owned(),
                })?;
        #[cfg(not(target_os = "macos"))]
        let _ = loaded;
        let Input::Tensor(input) = &request.input else {
            return Err(RuntimeError::InvalidInput {
                reason: "Core ML backend currently accepts tensor input".to_owned(),
            });
        };
        if input.shape.is_empty() || input.shape.iter().product::<usize>() != input.values.len() {
            return Err(RuntimeError::InvalidInput {
                reason: "tensor shape does not match tensor values".to_owned(),
            });
        }
        #[cfg(target_os = "macos")]
        let input_name = loaded
            .spec
            .metadata
            .get("input_name")
            .map(String::as_str)
            .unwrap_or("input");
        #[cfg(target_os = "macos")]
        let output_name = loaded
            .spec
            .metadata
            .get("output_name")
            .map(String::as_str)
            .unwrap_or("output");
        #[cfg(target_os = "macos")]
        let (shape, values) = native::predict(
            loaded.loaded.as_ref(),
            input_name,
            output_name,
            &input.shape,
            &input.values,
        )
        .map_err(|error| RuntimeError::execution("Core ML inference", error))?;
        if cancellation
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(RuntimeError::Cancelled);
        }
        #[cfg(target_os = "macos")]
        return Ok(InferenceResult {
            output: Output::Tensor(Tensor { shape, values }),
            metadata: ExecutionMetadata {
                model: loaded.spec.id.clone(),
                provider: "local".to_owned(),
                backend: self.name().to_owned(),
                hardware: Some("coreml".to_owned()),
                ..ExecutionMetadata::default()
            },
        });
        #[cfg(not(target_os = "macos"))]
        unreachable!()
    }

    async fn infer_stream(
        &self,
        _model: &ModelHandle,
        _request: &InferenceRequest,
        _cancellation: Option<CancellationToken>,
    ) -> RuntimeResult<BoxStream<RuntimeResult<InferenceChunk>>> {
        Ok(Box::pin(stream::iter(vec![Err(
            RuntimeError::CapabilityMismatch {
                reason: "streaming is not available for the coreml scaffold backend".to_owned(),
            },
        )])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn descriptor_matches_coreml_boundary() {
        assert_eq!(CoreMlBackend::descriptor().name, "coreml");
    }

    #[test]
    fn reports_platform_availability_without_fabricating_linux_support() {
        let capabilities = CoreMlBackend.capabilities();
        assert_eq!(capabilities.available, cfg!(target_os = "macos"));
        if !cfg!(target_os = "macos") {
            assert!(capabilities
                .notes
                .iter()
                .any(|note| note.contains("unsupported platform")));
            assert!(capabilities.accelerators.is_empty());
        }
    }

    #[tokio::test]
    async fn rejects_loading_on_unsupported_platform() {
        if cfg!(target_os = "macos") {
            return;
        }
        let result = CoreMlBackend
            .load(&ModelSpec::new(
                "coreml",
                ModelFormat::CoreMl,
                ModelLocation::Memory,
            ))
            .await;
        assert!(matches!(
            result,
            Err(RuntimeError::BackendUnavailable { .. })
        ));
    }
}
