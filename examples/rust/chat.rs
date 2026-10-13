//! Generative inference against an OpenAI-compatible endpoint (for example a
//! local Ollama server). The provider and model are chosen explicitly.

use futures_util::StreamExt;
use ml_runtime::{
    ChatMessage, ChatRequest, ChatStreamEvent, OpenAiCompatibleConfig, OpenAiCompatibleProvider,
    Runtime,
};

#[tokio::main]
async fn main() -> ml_runtime::RuntimeResult<()> {
    let provider =
        OpenAiCompatibleProvider::new(OpenAiCompatibleConfig::new("http://127.0.0.1:11434/v1"))?;
    let runtime = Runtime::builder().register_generation_provider(provider).build();
    let request = ChatRequest::new("llama3.2", vec![ChatMessage::user("Say hello.")]);

    let mut events = runtime.chat_stream("openai-compatible", request, None).await?;
    while let Some(event) = events.next().await {
        match event? {
            ChatStreamEvent::Delta { content } => print!("{content}"),
            ChatStreamEvent::Completed(summary) => {
                println!("\n{:?} {:?}", summary.finish_reason, summary.metadata.execution);
            }
            ChatStreamEvent::Started { .. } => {}
        }
    }
    Ok(())
}
