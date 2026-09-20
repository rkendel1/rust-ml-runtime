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
mod native {
    use objc::runtime::Object;
    use objc::{class, msg_send, sel, sel_impl};
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::path::Path;

    type Id = *mut Object;

    pub struct LoadedModel {
        pub model: Id,
    }

    unsafe impl Send for LoadedModel {}
    unsafe impl Sync for LoadedModel {}

    impl Drop for LoadedModel {
        fn drop(&mut self) {
            unsafe {
                let _: () = msg_send![self.model, release];
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

    pub fn load(path: &Path) -> Result<LoadedModel, String> {
        unsafe {
            let path = string(&path.to_string_lossy());
            let url: Id = msg_send![class!(NSURL), fileURLWithPath: path];
            let mut error: Id = std::ptr::null_mut();
            let model: Id =
                msg_send![class!(MLModel), modelWithContentsOfURL: url error: &mut error];
            if model.is_null() {
                return Err(error_message(error));
            }
            let _: Id = msg_send![model, retain];
            Ok(LoadedModel { model })
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
        unsafe {
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
            let data_type: usize = 65600; // MLMultiArrayDataTypeFloat32
            let data = values.as_ptr() as *mut c_void;
            let deallocator: *mut Object = std::ptr::null_mut();
            let multi_array: Id = msg_send![
                msg_send![class!(MLMultiArray), alloc],
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
            let provider: Id = msg_send![
                msg_send![class!(MLDictionaryFeatureProvider), alloc],
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
                output_shape.push(msg_send![dimension, unsignedLongLongValue] as usize);
            }
            Ok((
                output_shape,
                std::slice::from_raw_parts(pointer, count).to_vec(),
            ))
        }
    }
}

impl CoreMlBackend {
    pub fn descriptor() -> BackendCapability {
        BackendCapability::from_parts("coreml", Self::default().capabilities())
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
        let capabilities = CoreMlBackend::default().capabilities();
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
        let result = CoreMlBackend::default()
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
