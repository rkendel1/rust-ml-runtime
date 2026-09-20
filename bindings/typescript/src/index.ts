import { execFile } from "node:child_process"
import { promisify } from "node:util"

const execFileAsync = promisify(execFile)

export interface RuntimeConfig {
  binaryPath?: string
  backend?: string
  provider?: string
  preferAcceleration?: boolean
  allowRemoteFallback?: boolean
}

export interface InferRequest {
  model: string
  input: string
  tensor?: { shape: number[]; values: number[] }
  requireRemote?: boolean
}

async function runJson(binaryPath: string, args: string[]) {
  const { stdout } = await execFileAsync(binaryPath, args)
  try {
    return JSON.parse(stdout)
  } catch (error) {
    const excerpt = stdout.slice(0, 200)
    throw new Error(
      `Failed to parse JSON from ${binaryPath} ${args.join(" ")}: ${excerpt}`,
      { cause: error as Error }
    )
  }
}

export async function createRuntime(config: RuntimeConfig = {}) {
  const binaryPath = config.binaryPath ?? "ml-runtime"

  return {
    async capabilities() {
      return runJson(binaryPath, ["capabilities", "--json"])
    },
    async infer(request: InferRequest) {
      const args = ["run", request.model, "--input", request.input, "--json"]
      if (request.tensor) {
        args.push("--tensor", request.tensor.values.join(","))
        args.push("--shape", request.tensor.shape.join(","))
      }
      if (config.backend) args.push("--backend", config.backend)
      if (config.provider) args.push("--provider", config.provider)
      if (config.preferAcceleration) args.push("--prefer-acceleration")
      if (config.allowRemoteFallback) args.push("--allow-remote-fallback")
      if (request.requireRemote) args.push("--require-remote")
      return runJson(binaryPath, args)
    }
  }
}
