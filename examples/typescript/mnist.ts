import { readFile } from "node:fs/promises"
import { createRuntime } from "../../bindings/typescript/src/index.ts"

async function main() {
  const values = (await readFile("examples/inputs/mnist-8-seven.csv", "utf8"))
    .trim()
    .split(",")
    .map(Number)
  const runtime = await createRuntime({
    binaryPath: process.env.ML_RUNTIME_BINARY ?? "./target/debug/ml-runtime",
    modelsPath: "examples/models",
    backend: "onnx"
  })
  const result = await runtime.infer({
    model: { id: "mnist-8", version: "8" },
    input: { type: "tensor", shape: [1, 1, 28, 28], values }
  })
  console.log(result)
}

void main()
