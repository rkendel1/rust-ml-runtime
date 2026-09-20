import { execFile } from "node:child_process"
import { spawn } from "node:child_process"
import { promisify } from "node:util"

const execFileAsync = promisify(execFile)

export interface RuntimeConfig {
  binaryPath?: string
  modelsPath?: string
  backend?: string
  provider?: string
  preferAcceleration?: boolean
  allowRemoteFallback?: boolean
  execution?: ExecutionPolicy
}

export type ExecutionPolicy =
  | "local-only"
  | "remote-only"
  | "prefer-local"
  | "prefer-remote"
  | "local-then-remote"
  | "remote-then-local"

export interface InferRequest {
  model: string
  input: string
  tensor?: { shape: number[]; values: number[] }
  requireRemote?: boolean
  execution?: ExecutionPolicy
}

export type InferenceStreamEvent =
  | { Started: Record<string, unknown> }
  | { Output: unknown }
  | { Completed: Record<string, unknown> }

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
    async models() {
      const args = ["models", "--json"]
      if (config.modelsPath) args.push("--models", config.modelsPath)
      return runJson(binaryPath, args)
    },
    async resolveModel(reference: string) {
      const args = ["inspect", reference, "--json"]
      if (config.modelsPath) args.push("--models", config.modelsPath)
      return runJson(binaryPath, args)
    },
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
      if (request.execution ?? config.execution) {
        args.push("--execution", request.execution ?? config.execution!)
      }
      return runJson(binaryPath, args)
    },
    async *inferStream(request: InferRequest): AsyncGenerator<InferenceStreamEvent> {
      const args = ["run", request.model, "--input", request.input, "--json", "--stream"]
      if (request.tensor) {
        args.push("--tensor", request.tensor.values.join(","))
        args.push("--shape", request.tensor.shape.join(","))
      }
      if (config.backend) args.push("--backend", config.backend)
      if (config.provider) args.push("--provider", config.provider)
      if (config.preferAcceleration) args.push("--prefer-acceleration")
      if (config.allowRemoteFallback) args.push("--allow-remote-fallback")
      if (request.requireRemote) args.push("--require-remote")
      if (request.execution ?? config.execution) {
        args.push("--execution", request.execution ?? config.execution!)
      }
      const child = spawn(binaryPath, args, { stdio: ["ignore", "pipe", "pipe"] })
      let pending = ""
      for await (const chunk of child.stdout) {
        pending += chunk.toString()
        let newline = pending.indexOf("\n")
        while (newline >= 0) {
          const line = pending.slice(0, newline).trim()
          pending = pending.slice(newline + 1)
          if (line) yield JSON.parse(line) as InferenceStreamEvent
          newline = pending.indexOf("\n")
        }
      }
      if (pending.trim()) yield JSON.parse(pending.trim()) as InferenceStreamEvent
    }
  }
}
