use ml_runtime::ml_runtime_inference::{InferenceOptions, InferenceRequest, Input};
use ml_runtime::ml_runtime_model::{ModelFormat, ModelLocation, ModelSpec};
use ml_runtime::Runtime;
use ml_runtime_cpu_backend::CpuBackend;

#[tokio::main]
async fn main() {
    let runtime = Runtime::builder().register_backend(CpuBackend::default()).build();
    let model = ModelSpec::new("example-model", ModelFormat::Unknown, ModelLocation::Memory);
    let result = runtime
        .infer(InferenceRequest {
            model: model.into(),
            input: Input::Text("Hello world".to_owned()),
            options: InferenceOptions::default(),
        })
        .await
        .unwrap();
    println!("{:?}", result);
}
