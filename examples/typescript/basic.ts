import { createRuntime } from "../../bindings/typescript/src/index"

async function main() {
  const runtime = await createRuntime({ binaryPath: "ml-runtime" })
  const capabilities = await runtime.capabilities()
  console.log(capabilities)
}

void main()
