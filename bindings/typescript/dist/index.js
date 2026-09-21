import { execFile, spawn } from "node:child_process"
import { promisify } from "node:util"

const execFileAsync = promisify(execFile)

export class RuntimeClientError extends Error {
  constructor(kind, message, command, exitCode, stderr, options) {
    super(message, options)
    this.name = "RuntimeClientError"
    this.kind = kind
    this.command = command
    this.exitCode = exitCode
    this.stderr = stderr
  }
}

function modelArgument(model) {
  return typeof model === "string" ? model : model.version ? `${model.id}@${model.version}` : model.id
}

function inferenceArguments(request) {
  const input = typeof request.input === "string"
    ? request.input
    : request.input.type === "text" ? request.input.value : ""
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

async function runJson(binaryPath, args) {
  let stdout
  try {
    stdout = (await execFileAsync(binaryPath, args)).stdout
  } catch (error) {
    throw new RuntimeClientError(
      "process",
      `Runtime process failed with exit code ${error.code ?? "unknown"}`,
      [binaryPath, ...args],
      error.code,
      error.stderr?.trim(),
      { cause: error }
    )
  }
  try {
    return JSON.parse(stdout)
  } catch (error) {
    const excerpt = stdout.slice(0, 200)
    throw new RuntimeClientError(
      "protocol",
      `Failed to parse JSON from ${binaryPath} ${args.join(" ")}: ${excerpt}`,
      [binaryPath, ...args],
      undefined,
      undefined,
      { cause: error }
    )
  }
}

export async function createRuntime(config = {}) {
  const binaryPath = config.binaryPath ?? "ml-runtime"

  return {
    async models() {
      const args = ["models", "--json"]
      if (config.modelsPath) args.push("--models", config.modelsPath)
      return runJson(binaryPath, args)
    },
    async resolveModel(reference) {
      const args = ["inspect", reference, "--json"]
      if (config.modelsPath) args.push("--models", config.modelsPath)
      return runJson(binaryPath, args)
    },
    async verifyModel(reference) {
      const args = ["models", "verify", reference, "--json"]
      if (config.modelsPath) args.push("--models", config.modelsPath)
      return runJson(binaryPath, args)
    },
    async capabilities() {
      return runJson(binaryPath, ["capabilities", "--json"])
    },
    async infer(request) {
      const args = inferenceArguments(request)
      if (config.backend) args.push("--backend", config.backend)
      if (config.provider) args.push("--provider", config.provider)
      if (config.endpoint) args.push("--endpoint", config.endpoint)
      if (config.preferAcceleration) args.push("--prefer-acceleration")
      if (config.allowRemoteFallback) args.push("--allow-remote-fallback")
      if (request.requireRemote) args.push("--require-remote")
      if (request.execution ?? config.execution) {
        args.push("--execution", request.execution ?? config.execution)
      }
      if (config.modelsPath) args.push("--models", config.modelsPath)
      return runJson(binaryPath, args)
    },
    async *inferStream(request) {
      const args = [...inferenceArguments(request), "--stream"]
      if (config.backend) args.push("--backend", config.backend)
      if (config.provider) args.push("--provider", config.provider)
      if (config.endpoint) args.push("--endpoint", config.endpoint)
      if (config.preferAcceleration) args.push("--prefer-acceleration")
      if (config.allowRemoteFallback) args.push("--allow-remote-fallback")
      if (request.requireRemote) args.push("--require-remote")
      if (request.execution ?? config.execution) {
        args.push("--execution", request.execution ?? config.execution)
      }
      if (config.modelsPath) args.push("--models", config.modelsPath)
      const child = spawn(binaryPath, args, { stdio: ["ignore", "pipe", "pipe"] })
      let stderr = ""
      child.stderr.on("data", chunk => { stderr += chunk.toString() })
      const completion = new Promise((resolve, reject) => {
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
          if (line) yield JSON.parse(line)
          newline = pending.indexOf("\n")
        }
      }
      if (pending.trim()) yield JSON.parse(pending.trim())
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
