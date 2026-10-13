//! Caller-owned tool calling. The model proposes; this application validates,
//! authorizes, and executes; rust-ml-runtime never runs a tool.

use ml_runtime::{
    ChatMessage, ChatRequest, GenerationOptions, OpenAiCompatibleConfig, OpenAiCompatibleProvider,
    Runtime, ToolCall, ToolChoice, ToolDefinition,
};
use serde_json::{json, Value};

/// Deterministic, application-owned tool (no I/O).
fn get_weather(arguments: &Value) -> Value {
    json!({"city": arguments["city"], "temp_c": 21, "sky": "clear"})
}

/// Application policy, not runtime policy.
fn authorize(call: &ToolCall) -> bool {
    call.name == "get_weather" && call.arguments["city"].is_string()
}

#[tokio::main]
async fn main() -> ml_runtime::RuntimeResult<()> {
    let mut config = OpenAiCompatibleConfig::new("http://127.0.0.1:11434/v1");
    config.capabilities.tools = true; // support cannot be discovered; declare it
    config.capabilities.tool_choice = true;
    let runtime = Runtime::builder()
        .register_generation_provider(OpenAiCompatibleProvider::new(config)?)
        .build();

    let options = GenerationOptions {
        tools: vec![ToolDefinition::new(
            "get_weather",
            json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}),
        )
        .with_description("Current weather for a city")],
        tool_choice: Some(ToolChoice::Auto),
        ..Default::default()
    };
    let mut messages = vec![ChatMessage::user("What's the weather in Paris?")];

    let proposal = runtime
        .chat(
            "openai-compatible",
            ChatRequest::new("llama3.2", messages.clone()).with_options(options.clone()),
            None,
        )
        .await?;
    if proposal.message.tool_calls.is_empty() {
        println!("{}", proposal.message.content);
        return Ok(());
    }

    messages.push(proposal.message.clone()); // replay the assistant turn unchanged
    for call in &proposal.message.tool_calls {
        let result = if authorize(call) {
            get_weather(&call.arguments) // executed HERE, by the application
        } else {
            json!({"error": "not authorized"})
        };
        messages.push(ChatMessage::tool_result(&call.id, result.to_string()));
    }

    let answer = runtime
        .chat(
            "openai-compatible",
            ChatRequest::new("llama3.2", messages).with_options(options),
            None,
        )
        .await?;
    println!("{}", answer.message.content);
    Ok(())
}
