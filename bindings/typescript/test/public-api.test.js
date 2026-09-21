import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import { test } from "node:test"
import { fileURLToPath } from "node:url"
import { dirname, resolve } from "node:path"
import { RuntimeClientError, createRuntime } from "../dist/index.js"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../..")
const binaryPath = process.env.ML_RUNTIME_BINARY ?? resolve(root, "target/debug/ml-runtime")
const modelsPath = resolve(root, "examples/models")
const values = (await readFile(resolve(root, "examples/inputs/mnist-8-seven.csv"), "utf8"))
  .trim()
  .split(",")
  .map(Number)
const request = {
  model: { id: "mnist-8", version: "8" },
  input: { type: "tensor", shape: [1, 1, 28, 28], values }
}

test("public TypeScript API returns output and truthful metadata", async () => {
  const runtime = await createRuntime({ binaryPath, modelsPath })
  const result = await runtime.infer(request)
  assert.deepEqual(result.output.Tensor.shape, [1, 10])
  assert.equal(result.metadata.model, "mnist-8")
  assert.equal(result.metadata.provider, "local")
  assert.equal(result.metadata.backend, "onnx")
})

test("public TypeScript API exposes structured bridge errors", async () => {
  const runtime = await createRuntime({ binaryPath, modelsPath })
  await assert.rejects(
    runtime.infer({ model: { id: "missing", version: "1" }, input: "test" }),
    error => error instanceof RuntimeClientError && error.kind === "process" && error.exitCode === 1
  )
})

test("public TypeScript API streams normalized runtime events", async () => {
  const runtime = await createRuntime({ binaryPath, modelsPath })
  const events = []
  for await (const event of runtime.inferStream(request)) events.push(event)
  assert.ok("Started" in events[0])
  assert.ok(events.some(event => "Output" in event))
  assert.ok("Completed" in events.at(-1))
})
