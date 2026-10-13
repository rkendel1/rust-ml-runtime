use ml_runtime_common::RuntimeError;
use ml_runtime_provider::{
    ChatMessage, ChatRequest, ChatRole, GenerationCapabilities, GenerationOptions, ToolCall,
    ToolChoice, ToolDefinition,
};
use serde_json::json;

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(name, json!({"type":"object"}))
}

fn with_tools(
    messages: Vec<ChatMessage>,
    tools: Vec<ToolDefinition>,
    choice: Option<ToolChoice>,
) -> ChatRequest {
    ChatRequest::new("m", messages).with_options(GenerationOptions {
        tools,
        tool_choice: choice,
        ..Default::default()
    })
}

fn user() -> ChatMessage {
    ChatMessage::user("go")
}

fn call(id: &str) -> ToolCall {
    ToolCall::new(id, "t", json!({"a":1}))
}

fn invalid(request: ChatRequest) -> String {
    match request.validate() {
        Err(RuntimeError::InvalidRequest { reason }) => reason,
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn text_only_wire_shape_is_unchanged() {
    let request = ChatRequest::new("m", vec![ChatMessage::system("s"), user()]);
    let value = serde_json::to_value(&request).unwrap();
    assert_eq!(value["messages"][0], json!({"role":"system","content":"s"}));
    // Tool fields default when absent, so pre-existing JSON still deserializes.
    let old: ChatRequest =
        serde_json::from_value(json!({"model":"m","messages":[{"role":"user","content":"u"}]}))
            .unwrap();
    assert!(old.options.tools.is_empty() && old.options.tool_choice.is_none());
    assert!(old.validate().is_ok());
}

#[test]
fn tool_types_round_trip_and_keep_call_and_result_distinct() {
    let request = with_tools(
        vec![
            user(),
            ChatMessage::assistant_with_tool_calls("thinking", vec![call("c1")]),
            ChatMessage::tool_result("c1", "result"),
        ],
        vec![tool("t").with_description("d")],
        Some(ToolChoice::Tool { name: "t".into() }),
    );
    let text = serde_json::to_string(&request).unwrap();
    assert_eq!(serde_json::from_str::<ChatRequest>(&text).unwrap(), request);
    let value = serde_json::to_value(&request).unwrap();
    assert_eq!(
        value["options"]["tool_choice"],
        json!({"type":"tool","name":"t"})
    );
    assert_eq!(
        value["messages"][1]["tool_calls"][0],
        json!({"id":"c1","name":"t","arguments":{"a":1}})
    );
    assert!(
        value["messages"][1].get("tool_call_id").is_none(),
        "a call is not a result"
    );
    assert_eq!(value["messages"][2]["role"], "tool");
    assert_eq!(value["messages"][2]["tool_call_id"], "c1");
    assert!(
        value["messages"][2].get("tool_calls").is_none(),
        "a result is not a call"
    );
    assert_eq!(
        serde_json::to_value(ToolChoice::Auto).unwrap(),
        json!({"type":"auto"})
    );
    assert_eq!(
        serde_json::to_value(ToolChoice::None).unwrap(),
        json!({"type":"none"})
    );
    assert!(request.validate().is_ok());
}

#[test]
fn definitions_are_validated() {
    assert!(invalid(with_tools(vec![user()], vec![tool("bad name")], None)).contains("tool name"));
    assert!(invalid(with_tools(vec![user()], vec![tool("")], None)).contains("tool name"));
    assert!(
        invalid(with_tools(vec![user()], vec![tool(&"x".repeat(65))], None)).contains("tool name")
    );
    assert!(
        invalid(with_tools(vec![user()], vec![tool("a"), tool("a")], None)).contains("duplicate")
    );
    assert!(invalid(with_tools(
        vec![user()],
        vec![ToolDefinition::new("a", json!("string"))],
        None
    ))
    .contains("JSON object"));
    assert!(
        invalid(with_tools(vec![user()], vec![], Some(ToolChoice::Auto)))
            .contains("at least one tool")
    );
    assert!(invalid(with_tools(
        vec![user()],
        vec![tool("a")],
        Some(ToolChoice::Tool { name: "b".into() })
    ))
    .contains("undefined tool"));
    assert!(
        with_tools(vec![user()], vec![tool("a")], Some(ToolChoice::None))
            .validate()
            .is_ok()
    );
}

#[test]
fn tool_results_must_answer_an_earlier_unanswered_call() {
    let assistant = ChatMessage::assistant_with_tool_calls("", vec![call("c1")]);
    // Result with no proposal, with the wrong id, or answered twice.
    assert!(invalid(with_tools(
        vec![user(), ChatMessage::tool_result("c1", "r")],
        vec![],
        None
    ))
    .contains("does not answer"));
    assert!(invalid(with_tools(
        vec![
            user(),
            assistant.clone(),
            ChatMessage::tool_result("zzz", "r")
        ],
        vec![],
        None
    ))
    .contains("does not answer"));
    assert!(invalid(with_tools(
        vec![
            user(),
            assistant.clone(),
            ChatMessage::tool_result("c1", "r"),
            ChatMessage::tool_result("c1", "again")
        ],
        vec![],
        None
    ))
    .contains("does not answer"));
    // A result before the call it answers.
    assert!(invalid(with_tools(
        vec![user(), ChatMessage::tool_result("c1", "r"), assistant],
        vec![],
        None
    ))
    .contains("does not answer"));
}

#[test]
fn malformed_message_shapes_are_rejected() {
    let mut missing_id = ChatMessage::new(ChatRole::Tool, "r");
    missing_id.tool_call_id = None;
    assert!(invalid(with_tools(vec![user(), missing_id], vec![], None))
        .contains("requires tool_call_id"));

    let mut user_with_calls = user();
    user_with_calls.tool_calls = vec![call("c")];
    assert!(invalid(with_tools(vec![user_with_calls], vec![], None))
        .contains("only assistant and tool"));

    let mut tool_with_calls = ChatMessage::tool_result("c", "r");
    tool_with_calls.tool_calls = vec![call("d")];
    assert!(invalid(with_tools(
        vec![
            user(),
            ChatMessage::assistant_with_tool_calls("", vec![call("c")]),
            tool_with_calls
        ],
        vec![],
        None
    ))
    .contains("cannot propose"));

    let mut assistant_with_id = ChatMessage::assistant("x");
    assistant_with_id.tool_call_id = Some("c".into());
    assert!(
        invalid(with_tools(vec![user(), assistant_with_id], vec![], None)).contains("tool_call_id")
    );

    let bad_args =
        ChatMessage::assistant_with_tool_calls("", vec![ToolCall::new("c", "t", json!([1]))]);
    assert!(invalid(with_tools(vec![user(), bad_args], vec![], None)).contains("JSON object"));
    let bad_name =
        ChatMessage::assistant_with_tool_calls("", vec![ToolCall::new("c", "no good", json!({}))]);
    assert!(invalid(with_tools(vec![user(), bad_name], vec![], None)).contains("valid tool name"));
    let dup = ChatMessage::assistant_with_tool_calls("", vec![call("c"), call("c")]);
    assert!(invalid(with_tools(vec![user(), dup], vec![], None)).contains("duplicate"));
}

#[test]
fn capabilities_never_silently_drop_tool_requests() {
    let caps = |tools, tool_choice| GenerationCapabilities {
        streaming: true,
        stop_sequences: true,
        seed: true,
        json_object: true,
        json_schema: true,
        tools,
        tool_choice,
    };
    let plain = with_tools(vec![user()], vec![tool("t")], None);
    let choosy = with_tools(vec![user()], vec![tool("t")], Some(ToolChoice::Auto));
    let history = with_tools(
        vec![
            user(),
            ChatMessage::assistant_with_tool_calls("", vec![call("c")]),
            ChatMessage::tool_result("c", "r"),
        ],
        vec![],
        None,
    );
    for request in [&plain, &choosy, &history] {
        assert!(matches!(
            caps(false, false).check(request, false),
            Err(RuntimeError::CapabilityMismatch { .. })
        ));
    }
    assert!(caps(true, false).check(&plain, false).is_ok());
    assert!(matches!(
        caps(true, false).check(&choosy, false),
        Err(RuntimeError::CapabilityMismatch { .. })
    ));
    assert!(caps(true, true).check(&choosy, true).is_ok());
    // Text-only requests are unaffected by tool capability flags.
    assert!(caps(false, false)
        .check(&ChatRequest::new("m", vec![user()]), true)
        .is_ok());
}

#[test]
fn debug_output_omits_arguments_and_content() {
    let secret = ToolCall::new("c", "t", json!({"ssn":"123-45-6789"}));
    assert!(!format!("{secret:?}").contains("123-45-6789"));
    let message = ChatMessage::tool_result("c", "sensitive result");
    assert!(!format!("{message:?}").contains("sensitive"));
}
