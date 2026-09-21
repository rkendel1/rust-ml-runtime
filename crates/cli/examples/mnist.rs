use ml_runtime::{
    FilesystemModelCatalog, InferenceRequest, Input, ModelReference, Output, Runtime, Tensor,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let values = include_str!("../../../examples/inputs/mnist-8-seven.csv")
        .trim()
        .split(',')
        .map(str::parse)
        .collect::<Result<Vec<f32>, _>>()?;
    let runtime = Runtime::builder()
        .catalog(FilesystemModelCatalog::new("examples/models"))
        .build();
    let result = runtime
        .infer(InferenceRequest::new(
            ModelReference::versioned("mnist-8", "8"),
            Input::Tensor(Tensor::new([1, 1, 28, 28], values)),
        ))
        .await?;
    let Output::Tensor(output) = result.output else {
        return Err("MNIST returned a non-tensor output".into());
    };
    let digit = output
        .values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .ok_or("MNIST returned an empty tensor")?;
    println!(
        "digit={digit} provider={} backend={}",
        result.metadata.provider, result.metadata.backend
    );
    Ok(())
}
