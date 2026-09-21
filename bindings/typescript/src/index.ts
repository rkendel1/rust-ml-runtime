import { execFile } from "node:child_process"
import { spawn } from "node:child_process"
import { promisify } from "node:util"

const execFileAsync = promisify(execFile)

export interface RuntimeConfig {
  binaryPath?: string
  modelsPath?: string
  backend?: string
  provider?: string
  endpoint?: string
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

export type ModelReference = string | { id: string; version?: string }

export type InferenceInput =
  | string
  | { type: "text"; value: string }
  | { type: "tensor"; shape: number[]; values: number[] }

export interface InferRequest {
  model: ModelReference
  input: InferenceInput
  /** @deprecated Prefer a tensor-valued `input`. */
  tensor?: { shape: number[]; values: number[] }
  requireRemote?: boolean
  execution?: ExecutionPolicy
}

export interface ExecutionMetadata {
  request_id: string | null
  model: string
  model_version: string | null
  provider: string
  backend: string
  execution_target: string
  routing_policy: string
  fallback: boolean
  fallback_from: string | null
  fallback_reason: string | null
  latency_ms: number | null
  input_tokens: number | null
  output_tokens: number | null
  hardware: string | null
  cache_hit: boolean
  batch_size: number
  queue_wait_ms: number | null
  model_load_ms: number | null
  execution_ms: number | null
  streaming_requested: boolean
  streaming_supported: boolean
  output_event_count: number
  completion_state: string | null
}

export type InferenceOutput =
  | { Text: string }
  | { Tokens: number[] }
  | { Tensor: { shape: number[]; values: number[] } }
  | { Embedding: number[] }
  | { Structured: unknown }
  | { Binary: number[] }

export interface InferenceResult {
  output: InferenceOutput
  metadata: ExecutionMetadata
}

export type RuntimeClientErrorKind = "process" | "protocol"

export class RuntimeClientError extends Error {
  readonly kind: RuntimeClientErrorKind
  readonly command: string[]
  readonly exitCode?: number | string
  readonly stderr?: string

  constructor(
    kind: RuntimeClientErrorKind,
    message: string,
    command: string[],
    exitCode?: number | string,
    stderr?: string,
    options?: ErrorOptions
  ) {
    super(message, options)
    this.name = "RuntimeClientError"
    this.kind = kind
    this.command = command
    this.exitCode = exitCode
    this.stderr = stderr
  }
}

export interface InstallModelRequest {
  source: string
  replace?: boolean
}

export type InferenceStreamEvent =
  | { Started: ExecutionMetadata }
  | { Output: InferenceOutput }
  | { Completed: ExecutionMetadata }

function modelArgument(model: ModelReference) {
  return typeof model === "string"
    ? model
    : model.version
      ? `${model.id}@${model.version}`
      : model.id
}

function inferenceArguments(request: InferRequest) {
  const input = typeof request.input === "string"
    ? request.input
    : request.input.type === "text"
      ? request.input.value
      : ""
  const tensor = typeof request.input === "object" && request.input.type === "tensor"
    ? request.input
    : request.tensor
  const args = ["run", modelArgument(request.model), "--input", input, "--json"]
  if (tensor) {
    args.push("--tensor", tensor.values.join(","))
    args.push("--shape", tensor.shape.join(","))
  }
  return args
}

async function runJson<T>(binaryPath: string, args: string[]): Promise<T> {
  let stdout: string
  try {
    stdout = (await execFileAsync(binaryPath, args)).stdout
  } catch (error) {
    const failure = error as { code?: number | string; stderr?: string }
    throw new RuntimeClientError(
      "process",
      `Runtime process failed with exit code ${failure.code ?? "unknown"}`,
      [binaryPath, ...args],
      failure.code,
      failure.stderr?.trim(),
      { cause: error }
    )
  }
  try {
    return JSON.parse(stdout) as T
  } catch (error) {
    const excerpt = stdout.slice(0, 200)
    throw new RuntimeClientError(
      "protocol",
      `Failed to parse JSON from ${binaryPath} ${args.join(" ")}: ${excerpt}`,
      [binaryPath, ...args],
      undefined,
      undefined,
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
    async installModel(request: InstallModelRequest) {
      const args = ["models", "install", request.source, "--json"]
      if (request.replace) args.push("--replace")
      if (config.modelsPath) args.push("--models", config.modelsPath)
      return runJson(binaryPath, args)
    },
    async verifyModel(reference: string) {
      const args = ["models", "verify", reference, "--json"]
      if (config.modelsPath) args.push("--models", config.modelsPath)
      return runJson(binaryPath, args)
    },
    async capabilities() {
      return runJson(binaryPath, ["capabilities", "--json"])
    },
    async infer(request: InferRequest) {
      const args = inferenceArguments(request)
      if (config.backend) args.push("--backend", config.backend)
      if (config.provider) args.push("--provider", config.provider)
      if (config.endpoint) args.push("--endpoint", config.endpoint)
      if (config.preferAcceleration) args.push("--prefer-acceleration")
      if (config.allowRemoteFallback) args.push("--allow-remote-fallback")
      if (request.requireRemote) args.push("--require-remote")
      if (request.execution ?? config.execution) {
        args.push("--execution", request.execution ?? config.execution!)
      }
      if (config.modelsPath) args.push("--models", config.modelsPath)
      return runJson<InferenceResult>(binaryPath, args)
    },
    async *inferStream(request: InferRequest): AsyncGenerator<InferenceStreamEvent> {
      const args = [...inferenceArguments(request), "--stream"]
      if (config.backend) args.push("--backend", config.backend)
      if (config.provider) args.push("--provider", config.provider)
      if (config.endpoint) args.push("--endpoint", config.endpoint)
      if (config.preferAcceleration) args.push("--prefer-acceleration")
      if (config.allowRemoteFallback) args.push("--allow-remote-fallback")
      if (request.requireRemote) args.push("--require-remote")
      if (request.execution ?? config.execution) {
        args.push("--execution", request.execution ?? config.execution!)
      }
      if (config.modelsPath) args.push("--models", config.modelsPath)
      const child = spawn(binaryPath, args, { stdio: ["ignore", "pipe", "pipe"] })
      let stderr = ""
      child.stderr.on("data", chunk => { stderr += chunk.toString() })
      const completion = new Promise<number | null>((resolve, reject) => {
        child.once("error", reject)
        child.once("close", resolve)
      })
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
      const exitCode = await completion
      if (exitCode !== 0) {
        throw new RuntimeClientError(
          "process",
          `Runtime process failed with exit code ${exitCode ?? "unknown"}`,
          [binaryPath, ...args],
          exitCode ?? undefined,
          stderr.trim()
        )
      }
    }
  }
}
