import { GenerativeML } from "@rust-ml-runtime/node"
import type { ChatMessage, ToolCall } from "@rust-ml-runtime/node"

// The model PROPOSES tool calls. rust-ml-runtime never runs them: this
// application validates, authorizes, and executes, then reports the result.

// Deterministic, application-owned tool (no I/O).
function getWeather(args: Record<string, unknown>) {
  return { city: args.city, temp_c: 21, sky: "clear" }
}

// Application policy: only these tools may run, and only with sane arguments.
function authorize(call: ToolCall): boolean {
  return call.name === "get_weather" && typeof call.arguments.city === "string"
}

async function main() {
  const ml = new GenerativeML({
    endpoint: "http://127.0.0.1:11434/v1",
    capabilities: { tools: true, toolChoice: true }, // must be declared; not discoverable
  })
  const options = {
    tools: [{
      name: "get_weather",
      description: "Current weather for a city",
      input_schema: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
        additionalProperties: false,
      },
    }],
    tool_choice: { type: "auto" as const },
  }
  const messages: ChatMessage[] = [{ role: "user", content: "What's the weather in Paris?" }]

  const proposal = await ml.chat({ model: "llama3.2", messages, options })
  const calls = proposal.message.tool_calls ?? []
  if (calls.length === 0) {
    console.log(proposal.message.content) // the model answered directly
    return
  }

  messages.push(proposal.message) // replay the assistant turn unchanged
  for (const call of calls) {
    // A proposal is not an operation. Refusing is also reported as a tool result.
    const content = authorize(call)
      ? JSON.stringify(getWeather(call.arguments))
      : JSON.stringify({ error: "not authorized" })
    messages.push({ role: "tool", tool_call_id: call.id, content })
  }

  const followUp = await ml.chat({ model: "llama3.2", messages, options })
  console.log(followUp.message.content)
}

void main()
