import { GenerativeML } from "@rust-ml-runtime/node"

// Streams a reply from an OpenAI-compatible endpoint (here: local Ollama).
// Set ML_RUNTIME_API_KEY for endpoints that need a bearer token.
async function main() {
  const ml = new GenerativeML({
    endpoint: "http://127.0.0.1:11434/v1",
    apiKeyEnv: process.env.ML_RUNTIME_API_KEY ? "ML_RUNTIME_API_KEY" : undefined,
  })
  const stream = ml.chatStream({
    model: "llama3.2",
    messages: [{ role: "user", content: "Say hello." }],
  })
  for await (const event of stream) {
    if (event.type === "delta") process.stdout.write(event.content)
    if (event.type === "completed") {
      console.log(`\n${event.metadata.execution} via ${event.metadata.provider}`, event.usage)
    }
  }
}

void main()
